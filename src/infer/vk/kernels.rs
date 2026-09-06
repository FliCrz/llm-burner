//! Batched compute dispatch for the Vulkan engine.
//!
//! Owns the Vulkan object graph that sits between the raw buffers
//! ([`super::device::Buffer`]) and the engine's forward pass: one compute
//! pipeline per (shader, dtype) pair, a shared host-visible uniform buffer that
//! receives the per-dispatch parameter words (naga can't parse WGSL
//! `push_constant`, so pipelines read parameters from a `var<uniform>` at the
//! last binding), and cached descriptor sets per buffer-site.
//!
//! Every dispatch records into a caller-owned command buffer (see
//! [`Kernels::cmd`], [`Kernels::submit`]). A full compute-stage barrier is
//! inserted after each kernel so neighbors on the scratch buffers and the
//! weight buffer never race; the very first thing recorded is a HOST_WRITE →
//! SHADER barrier so the host-written parameter words are visible.

use std::collections::HashMap;
use std::sync::Arc;

use anyhow::{Context, Result};
use ash::vk::Handle;

use ash::vk;

use super::{Buffer, Chip, Dt};
use crate::infer::vk::shaders;

/// Which shader+dtype combination a pipeline belongs to.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum Key {
    Gemv(Dt),
    Embed(Dt),
    RmsNorm,
    Add,
    Rope,
    StoreKv,
    Attn,
    Act,
}

impl Key {
    fn wgsl(self) -> String {
        match self {
            Key::Gemv(dt) => shaders::gemv_wgsl(dt),
            Key::Embed(dt) => shaders::embed_wgsl(dt),
            Key::RmsNorm => shaders::rmsnorm_wgsl(),
            Key::Add => shaders::add_wgsl(),
            Key::Rope => shaders::rope_wgsl(),
            Key::StoreKv => shaders::store_kv_wgsl(),
            Key::Attn => shaders::attn_wgsl(),
            Key::Act => shaders::act_wgsl(),
        }
    }

    /// Number of parameter words written before each dispatch.
    fn n_params(self) -> usize {
        match self {
            Key::Gemv(_) | Key::Embed(_) | Key::RmsNorm | Key::Add | Key::StoreKv | Key::Act => 4,
            Key::Rope => 8,
            Key::Attn => 12,
        }
    }

    /// Number of storage (non-param) binding slots.
    fn n_storage_slots(self) -> usize {
        match self {
            Key::Gemv(_) | Key::Embed(_) | Key::RmsNorm => 3,
            Key::Add | Key::Act => 2,
            Key::Rope => 3,
            Key::StoreKv | Key::Attn => 4,
        }
    }
}

/// One compute kernel: pipeline + layout + descriptor set layout + pool.
struct Pipeline {
    chip: Arc<Chip>,
    pipeline: vk::Pipeline,
    layout: vk::PipelineLayout,
    set_layout: vk::DescriptorSetLayout,
    pool: vk::DescriptorPool,
}

impl Drop for Pipeline {
    fn drop(&mut self) {
        unsafe {
            let dev = &self.chip.device;
            dev.destroy_descriptor_pool(self.pool, None);
            dev.destroy_pipeline(self.pipeline, None);
            dev.destroy_descriptor_set_layout(self.set_layout, None);
            dev.destroy_pipeline_layout(self.layout, None);
        }
    }
}

/// Identifies one set of bindings: the non-param buffer handles in binding
/// order (offsets are always 0 for now). The parameter uniform buffer is
/// shared by every set.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
struct SetKey(Vec<usize>);

/// Kernel dispatcher. Cheap to create; lazily compiles pipelines on first use.
pub struct Kernels {
    chip: Arc<Chip>,
    /// Shared host-visible uniform buffer holding every kernel's parameter
    /// words (worst case is the attention PC, 12 words; buffer is 64 bytes).
    params: Buffer,
    /// Lazily-built pipelines, keyed by (shader, dtype).
    pipelines: HashMap<Key, Pipeline>,
    /// Cached descriptor sets per (pipeline, binding-site).
    sets: HashMap<(Key, SetKey), vk::DescriptorSet>,
    cmd_pool: vk::CommandPool,
}

impl Kernels {
    pub fn new(chip: Arc<Chip>) -> Result<Self> {
        let params = Buffer::new(
            &chip,
            64,
            vk::BufferUsageFlags::UNIFORM_BUFFER,
            Some(&[0u8; 64]),
        )?;
        let cmd_pool = unsafe {
            let ci = vk::CommandPoolCreateInfo::default().queue_family_index(chip.queue_family);
            chip.device
                .create_command_pool(&ci, None)
                .map_err(|e| anyhow::anyhow!("create_command_pool: {e}"))?
        };
        Ok(Self {
            chip,
            params,
            pipelines: HashMap::new(),
            sets: HashMap::new(),
            cmd_pool,
        })
    }

    pub fn chip(&self) -> &Arc<Chip> {
        &self.chip
    }

    /// Allocate and begin a fresh one-shot command buffer, recording a
    /// HOST_WRITE → SHADER barrier so host-written parameter words (and any
    /// just-copied weight/scratch content) are visible to the dispatches.
    pub fn cmd(&self) -> vk::CommandBuffer {
        unsafe {
            let ai = vk::CommandBufferAllocateInfo::default()
                .command_pool(self.cmd_pool)
                .command_buffer_count(1);
            let bufs = self
                .chip
                .device
                .allocate_command_buffers(&ai)
                .expect("allocate_command_buffers");
            let bi = vk::CommandBufferBeginInfo::default();
            self.chip.device.begin_command_buffer(bufs[0], &bi).unwrap();
            let barrier = vk::MemoryBarrier::default()
                .src_access_mask(vk::AccessFlags::HOST_WRITE)
                .dst_access_mask(vk::AccessFlags::SHADER_READ | vk::AccessFlags::SHADER_WRITE);
            self.chip.device.cmd_pipeline_barrier(
                bufs[0],
                vk::PipelineStageFlags::HOST,
                vk::PipelineStageFlags::COMPUTE_SHADER,
                vk::DependencyFlags::empty(),
                &[barrier],
                &[],
                &[],
            );
            bufs[0]
        }
    }

    /// End the command buffer, submit it on the chip's queue, and wait.
    /// (Synchronous submission: the engine always blocks on the queue before
    /// reading results back, so a fence is unnecessary.)
    pub fn submit(&self, cmd: vk::CommandBuffer) -> Result<()> {
        unsafe {
            self.chip.device.end_command_buffer(cmd)?;
            let cmds = [cmd];
            let si = vk::SubmitInfo::default().command_buffers(&cmds);
            self.chip
                .device
                .queue_submit(self.chip.queue, &[si], vk::Fence::null())?;
            self.chip.device.queue_wait_idle(self.chip.queue)?;
        }
        Ok(())
    }

    fn record_compute_barrier(&self, cmd: vk::CommandBuffer) {
        unsafe {
            let barrier = vk::MemoryBarrier::default()
                .src_access_mask(vk::AccessFlags::SHADER_WRITE)
                .dst_access_mask(vk::AccessFlags::SHADER_READ | vk::AccessFlags::SHADER_WRITE);
            self.chip.device.cmd_pipeline_barrier(
                cmd,
                vk::PipelineStageFlags::COMPUTE_SHADER,
                vk::PipelineStageFlags::COMPUTE_SHADER,
                vk::DependencyFlags::empty(),
                &[barrier],
                &[],
                &[],
            );
        }
    }

    /// Compile + create resources for a key if not already cached.
    fn ensure_pipeline(&mut self, key: Key) -> Result<()> {
        if self.pipelines.contains_key(&key) {
            return Ok(());
        }
        let wgsl = key.wgsl();
        let spv =
            shaders::compile(&wgsl).with_context(|| format!("naga failed for {key:?}"))?;
        let dev = &self.chip.device;
        let pipeline = unsafe {
            let module = dev
                .create_shader_module(&vk::ShaderModuleCreateInfo::default().code(&spv), None)
                .map_err(|e| anyhow::anyhow!("create_shader_module({key:?}): {e}"))?;
            let stage = vk::PipelineShaderStageCreateInfo::default()
                .stage(vk::ShaderStageFlags::COMPUTE)
                .module(module)
                .name(c"main");
            let bindings = {
                let n = key.n_storage_slots();
                let mut v = Vec::with_capacity(n + 1);
                for i in 0..n {
                    v.push(
                        vk::DescriptorSetLayoutBinding::default()
                            .binding(i as u32)
                            .descriptor_type(vk::DescriptorType::STORAGE_BUFFER)
                            .descriptor_count(1)
                            .stage_flags(vk::ShaderStageFlags::COMPUTE),
                    );
                }
                v.push(
                    vk::DescriptorSetLayoutBinding::default()
                        .binding(n as u32)
                        .descriptor_type(vk::DescriptorType::UNIFORM_BUFFER)
                        .descriptor_count(1)
                        .stage_flags(vk::ShaderStageFlags::COMPUTE),
                );
                v
            };
            let set_layout = dev
                .create_descriptor_set_layout(
                    &vk::DescriptorSetLayoutCreateInfo::default().bindings(&bindings),
                    None,
                )
                .map_err(|e| anyhow::anyhow!("create_descriptor_set_layout({key:?}): {e}"))?;
            let sls = [set_layout];
            let layout = dev
                .create_pipeline_layout(
                    &vk::PipelineLayoutCreateInfo::default().set_layouts(&sls),
                    None,
                )
                .map_err(|e| anyhow::anyhow!("create_pipeline_layout({key:?}): {e}"))?;
            let sizes = [
                vk::DescriptorPoolSize::default()
                    .ty(vk::DescriptorType::STORAGE_BUFFER)
                    .descriptor_count(64 * key.n_storage_slots() as u32),
                vk::DescriptorPoolSize::default()
                    .ty(vk::DescriptorType::UNIFORM_BUFFER)
                    .descriptor_count(64),
            ];
            let pool = dev
                .create_descriptor_pool(
                    &vk::DescriptorPoolCreateInfo::default()
                        .max_sets(64)
                        .pool_sizes(&sizes),
                    None,
                )
                .map_err(|e| anyhow::anyhow!("create_descriptor_pool({key:?}): {e}"))?;
            let info = vk::ComputePipelineCreateInfo::default()
                .stage(stage)
                .layout(layout);
            let built = dev
                .create_compute_pipelines(vk::PipelineCache::null(), &[info], None)
                .map_err(|(_, e)| anyhow::anyhow!("create_compute_pipelines({key:?}): {e}"))?;
            dev.destroy_shader_module(module, None);
            Pipeline {
                chip: self.chip.clone(),
                pipeline: built[0],
                layout,
                set_layout,
                pool,
            }
        };
        self.pipelines.insert(key, pipeline);
        Ok(())
    }

    /// Fetch or build the descriptor set that maps the kernel's storage
    /// bindings (in order) onto `binds`. The parameter uniform buffer is
    /// always shared and written by [`Kernels::dispatch`].
    fn set_for(&mut self, key: Key, binds: &[&Buffer]) -> Result<vk::DescriptorSet> {
        self.ensure_pipeline(key)?;
        let skey = SetKey(binds.iter().map(|b| b.buffer.as_raw() as usize).collect());
        if let Some(s) = self.sets.get(&(key, skey.clone())) {
            return Ok(*s);
        }
        let dev = &self.chip.device;
        let pool = self.pipelines[&key].pool;
        let set_layout = self.pipelines[&key].set_layout;
        let set = unsafe {
            let sls = [set_layout];
            let ai = vk::DescriptorSetAllocateInfo::default()
                .descriptor_pool(pool)
                .set_layouts(&sls);
            let sets = dev
                .allocate_descriptor_sets(&ai)
                .map_err(|e| anyhow::anyhow!("allocate_descriptor_sets({key:?}): {e}"))?;
            let set = sets[0];
            let mut infos = Vec::with_capacity(binds.len() + 1);
            for b in binds.iter() {
                infos.push(
                    vk::DescriptorBufferInfo::default()
                        .buffer(b.buffer)
                        .offset(0)
                        .range(vk::WHOLE_SIZE),
                );
            }
            infos.push(
                vk::DescriptorBufferInfo::default()
                    .buffer(self.params.buffer)
                    .offset(0)
                    .range(self.params.size_bytes as u64),
            );
            let mut writes = Vec::with_capacity(binds.len() + 1);
            for (i, _) in binds.iter().enumerate() {
                writes.push(
                    vk::WriteDescriptorSet::default()
                        .dst_set(set)
                        .dst_binding(i as u32)
                        .descriptor_type(vk::DescriptorType::STORAGE_BUFFER)
                        .buffer_info(&infos[i..i + 1]),
                );
            }
            writes.push(
                vk::WriteDescriptorSet::default()
                    .dst_set(set)
                    .dst_binding(binds.len() as u32)
                    .descriptor_type(vk::DescriptorType::UNIFORM_BUFFER)
                    .buffer_info(&infos[binds.len()..]),
            );
            dev.update_descriptor_sets(&writes, &[]);
            set
        };
        self.sets.insert((key, skey), set);
        Ok(set)
    }

    /// Write the parameter words (host memcpy into the coherent mapping) then
    /// bind pipeline + set and dispatch `groups`, finishing with a barrier.
    fn dispatch(
        &mut self,
        cmd: vk::CommandBuffer,
        key: Key,
        binds: &[&Buffer],
        words: &[u32],
        groups: (u32, u32, u32),
    ) -> Result<()> {
        debug_assert!(words.len() <= key.n_params());
        let mut padded = [0u32; 12];
        padded[..words.len()].copy_from_slice(words);
        let n = key.n_params();
        let mut bytes = Vec::with_capacity(n * 4);
        for w in padded[..n].iter() {
            bytes.extend_from_slice(&w.to_le_bytes());
        }
        self.params.write(0, &bytes);
        let set = self.set_for(key, binds)?;
        self.ensure_pipeline(key)?;
        let (pipeline, layout) = {
            let p = &self.pipelines[&key];
            (p.pipeline, p.layout)
        };
        unsafe {
            let dev = &self.chip.device;
            dev.cmd_bind_pipeline(cmd, vk::PipelineBindPoint::COMPUTE, pipeline);
            dev.cmd_bind_descriptor_sets(
                cmd,
                vk::PipelineBindPoint::COMPUTE,
                layout,
                0,
                &[set],
                &[],
            );
            dev.cmd_dispatch(cmd, groups.0, groups.1, groups.2);
        }
        self.record_compute_barrier(cmd);
        Ok(())
    }

    // ------------------------------------------------------------------
    // Per-kernel conveniences. `binds` lists the storage buffers in binding
    // order (weights first where the kernel has a weight slot).
    // ------------------------------------------------------------------

    pub fn gemv(
        &mut self,
        cmd: vk::CommandBuffer,
        dt: Dt,
        wb: &Buffer,
        xv: &Buffer,
        outv: &Buffer,
        base: u32,
        rowlen: u32,
        ncols: u32,
        row0: u32,
        nrows: u32,
    ) -> Result<()> {
        self.dispatch(
            cmd,
            Key::Gemv(dt),
            &[wb, xv, outv],
            &[base, rowlen, ncols, row0],
            (1, nrows, 1),
        )
    }

    pub fn embed(
        &mut self,
        cmd: vk::CommandBuffer,
        dt: Dt,
        wb: &Buffer,
        ids: &Buffer,
        outv: &Buffer,
        base: u32,
        rowlen: u32,
        ncols: u32,
        tok0: u32,
        nrows: u32,
    ) -> Result<()> {
        let nblk = (ncols + 255) / 256;
        self.dispatch(
            cmd,
            Key::Embed(dt),
            &[wb, ids, outv],
            &[base, rowlen, ncols, tok0],
            (nblk * nrows, 1, 1),
        )
    }

    pub fn rmsnorm(
        &mut self,
        cmd: vk::CommandBuffer,
        wb: &Buffer,
        inv: &Buffer,
        outv: &Buffer,
        wbase: u32,
        len: u32,
        nrows: u32,
        eps: f64,
    ) -> Result<()> {
        self.dispatch(
            cmd,
            Key::RmsNorm,
            &[wb, inv, outv],
            &[wbase, len, nrows, (eps as f32).to_bits()],
            (1, nrows, 1),
        )
    }

    pub fn add(&mut self, cmd: vk::CommandBuffer, outv: &Buffer, inv: &Buffer, len: u32) -> Result<()> {
        self.dispatch(
            cmd,
            Key::Add,
            &[outv, inv],
            &[len, 0, 0, 0],
            ((len + 127) / 128, 1, 1),
        )
    }

    pub fn rope(
        &mut self,
        cmd: vk::CommandBuffer,
        qv: &Buffer,
        kv: &Buffer,
        freq: &Buffer,
        pos: u32,
        hd: u32,
        neox: u32,
        qdim: u32,
        kvdim: u32,
    ) -> Result<()> {
        let npairs = (qdim + kvdim) / 2;
        let words = [pos, hd, hd / 2, neox, qdim, kvdim, 0, 0];
        self.dispatch(
            cmd,
            Key::Rope,
            &[qv, kv, freq],
            &words,
            ((npairs + 127) / 128, 1, 1),
        )
    }

    pub fn store_kv(
        &mut self,
        cmd: vk::CommandBuffer,
        kvv: &Buffer,
        vvv: &Buffer,
        kc: &Buffer,
        vc: &Buffer,
        layer: u32,
        pos: u32,
        kvdim: u32,
        cap: u32,
    ) -> Result<()> {
        self.dispatch(
            cmd,
            Key::StoreKv,
            &[kvv, vvv, kc, vc],
            &[layer, pos, kvdim, cap],
            ((kvdim + 127) / 128, 1, 1),
        )
    }

    #[allow(clippy::too_many_arguments)]
    pub fn attn(
        &mut self,
        cmd: vk::CommandBuffer,
        qv: &Buffer,
        kc: &Buffer,
        vc: &Buffer,
        outv: &Buffer,
        layer: u32,
        pos: u32,
        win_start: u32,
        n_heads: u32,
        groups: u32,
        hd: u32,
        scale: f32,
        cap: u32,
        kvdim: u32,
    ) -> Result<()> {
        let words = [
            layer,
            pos,
            win_start,
            n_heads,
            groups,
            hd,
            scale.to_bits(),
            cap,
            kvdim,
            0,
            0,
            0,
        ];
        self.dispatch(cmd, Key::Attn, &[qv, kc, vc, outv], &words, (n_heads, 1, 1))
    }

    pub fn act(
        &mut self,
        cmd: vk::CommandBuffer,
        gv: &Buffer,
        uv: &Buffer,
        len: u32,
        mode: u32,
    ) -> Result<()> {
        self.dispatch(
            cmd,
            Key::Act,
            &[gv, uv],
            &[len, mode, 0, 0],
            ((len + 127) / 128, 1, 1),
        )
    }
}

impl Drop for Kernels {
    fn drop(&mut self) {
        unsafe {
            self.chip
                .device
                .destroy_command_pool(self.cmd_pool, None);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::infer::dequant_ref::dequant_elem;
    use crate::infer::vk::device;

    /// Minimum: open device, create a command pool, allocate one command
    /// buffer. Isolates table/dispatch integrity from the kernel code.
    #[test]
    fn device_cmd_smoke() {
        let chip = device::open().expect("Vulkan device");
        let pool = unsafe {
            let ci = vk::CommandPoolCreateInfo::default().queue_family_index(chip.queue_family);
            chip.device
                .create_command_pool(&ci, None)
                .expect("pool")
        };
        let ai = vk::CommandBufferAllocateInfo::default()
            .command_pool(pool)
            .command_buffer_count(1);
        let bufs = unsafe { chip.device.allocate_command_buffers(&ai).expect("cmds") };
        let bi = vk::CommandBufferBeginInfo::default();
        unsafe { chip.device.begin_command_buffer(bufs[0], &bi).unwrap() };
        unsafe { chip.device.end_command_buffer(bufs[0]).unwrap() };
    }

    /// End-to-end smoke test: F32 GEMV over real Vulkan (llvmpipe/RADV),
    /// compared against the CPU reference row dot.
    #[test]
    fn gemv_f32_roundtrip() {
        let chip = device::open().expect("Vulkan device");
        let kernels = Kernels::new(chip.clone()).expect("kernels");

        let ncols = 16u32;
        let nrows = 2u32;
        let w: Vec<f32> = (0..nrows * ncols)
            .map(|i| (i as f32).sin() * 3.0 + (i as f32) * 0.25)
            .collect();
        let x: Vec<f32> = (0..ncols).map(|i| (i as f32).cos() - 1.0).collect();
        let wb = Buffer::new(
            &chip,
            w.len() * 4,
            vk::BufferUsageFlags::STORAGE_BUFFER,
            Some(bytemuck::cast_slice(&w)),
        )
        .expect("weights");
        let xv = Buffer::new(
            &chip,
            x.len() * 4,
            vk::BufferUsageFlags::STORAGE_BUFFER,
            Some(bytemuck::cast_slice(&x)),
        )
        .expect("x");
        let outv = Buffer::new(
            &chip,
            nrows as usize * 4,
            vk::BufferUsageFlags::STORAGE_BUFFER,
            None,
        )
        .expect("out");

        let mut kernels = kernels;
        let cmd = kernels.cmd();
        kernels
            .gemv(cmd, Dt::F32, &wb, &xv, &outv, 0, ncols * 4, ncols, 0, nrows)
            .expect("dispatch");
        kernels.submit(cmd).expect("submit");

        let got = outv.read(0, nrows as usize * 4);
        let got: Vec<f32> = got
            .chunks_exact(4)
            .map(|c| f32::from_le_bytes(c.try_into().unwrap()))
            .collect();
        for r in 0..nrows as usize {
            let mut exp = 0.0f32;
            for e in 0..ncols as usize {
                exp += dequant_elem(Dt::F32, &bytemuck::cast_slice(&w), r * ncols as usize + e)
                    * x[e];
            }
            assert!(
                (got[r] - exp).abs() <= 1e-3 * exp.abs().max(1.0),
                "row {r}: got {} expected {exp}",
                got[r]
            );
        }
    }
}