//! Vulkan instance/device bootstrap, device selection, and buffer helpers.
//!
//! The engine deliberately targets *unified memory*: an iGPU's
//! host-visible + device-local heap maps straight onto system RAM, so model
//! weights are copied once from the GGUF mmap into one big buffer that the
//! GPU reads directly — no staging buffers, no per-step host→device copies.
//! [`pick_unified_memory`] performs that selection with a
//! `DEVICE_LOCAL|HOST_VISIBLE|HOST_COHERENT` type as first pick.

use std::ffi::{CStr, c_void};
use std::fmt;
use std::sync::Arc;

use anyhow::{Result, bail};

use ash::vk;

/// A minimal handle to a usable compute device plus its scratch resources.
///
/// Everything the engine needs beyond the raw [`ash::Device`] — queue,
/// memory types, name and limits — that outlives a decode step.
pub struct Chip {
    /// Kept alive for the device's sake: `ash::Entry` is the only owner of
    /// the `libvulkan.so` handle (`_lib_guard`), and dropping it dlcloses the
    /// loader, unmapping every function table that [`ash::Instance`] and
    /// [`ash::Device`] point into.
    pub entry: ash::Entry,
    pub instance: ash::Instance,
    pub device: ash::Device,
    /// Compute-capable queue (the only queue the engine uses).
    pub queue: vk::Queue,
    pub queue_family: u32,
    pub physical: vk::PhysicalDevice,
    pub name: String,
    /// Index of the preferred memory type for weight/scratch buffers.
    pub unified_mem: u32,
    /// Upper bound for `max_compute_work_group_invocations`.
    pub max_invocations: u32,
}

impl fmt::Debug for Chip {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("Chip")
            .field("name", &self.name)
            .field("queue_family", &self.queue_family)
            .field("unified_mem", &self.unified_mem)
            .finish_non_exhaustive()
    }
}

impl Drop for Chip {
    fn drop(&mut self) {
        unsafe {
            // Buffers (which hold an `Arc<Chip>`) are dropped before the chip
            // itself, so the device is still alive here.
            self.device.destroy_device(None);
            self.instance.destroy_instance(None);
        }
    }
}

/// Initialize a Vulkan compute device, preferring an integrated GPU when one
/// exists (the iGPU focus of this engine), and hand back a shared [`Chip`]
/// that buffers and the engine both borrow.
pub fn open() -> Result<Arc<Chip>> {
    let app_name = CStr::from_bytes_with_nul(b"llm-burner\0").unwrap();

    unsafe {
        let entry = ash::Entry::load().map_err(|e| anyhow::anyhow!("Vulkan loader: {e}"))?;
        let app_info = vk::ApplicationInfo::default()
            .application_name(app_name)
            .api_version(vk::API_VERSION_1_3);
        let ci = vk::InstanceCreateInfo::default().application_info(&app_info);

        let instance = entry
            .create_instance(&ci, None)
            .map_err(|e| anyhow::anyhow!("create_instance failed: {e}"))?;
        let physical = pick_physical_device(&instance)?;
        let queue_family = compute_queue_family(&instance, physical)?;
        let mem_props = instance.get_physical_device_memory_properties(physical);
        let unified_mem = pick_unified_memory(mem_props)?;
        let name = device_name(&instance, physical);
        let max_invocations = instance
            .get_physical_device_properties(physical)
            .limits
            .max_compute_work_group_invocations;

        let queue_priorities = [1.0f32];
        let queue_create = vk::DeviceQueueCreateInfo::default()
            .queue_family_index(queue_family)
            .queue_priorities(&queue_priorities);
        let dci = vk::DeviceCreateInfo::default()
            .queue_create_infos(std::slice::from_ref(&queue_create));
        let device = instance
            .create_device(physical, &dci, None)
            .map_err(|e| anyhow::anyhow!("create_device failed: {e}"))?;
        let queue = device.get_device_queue(queue_family, 0);

        Ok(Arc::new(Chip {
            entry,
            instance,
            device,
            queue,
            queue_family,
            physical,
            name,
            unified_mem,
            max_invocations,
        }))
    }
}

/// Human-readable device name (null-terminated `c_char` buffer).
fn device_name(instance: &ash::Instance, p: vk::PhysicalDevice) -> String {
    unsafe {
        let props = instance.get_physical_device_properties(p);
        let end = props
            .device_name
            .iter()
            .position(|&c| c == 0)
            .unwrap_or(props.device_name.len());
        let bytes: Vec<u8> = props.device_name[..end].iter().map(|&c| c as u8).collect();
        String::from_utf8_lossy(&bytes).into_owned()
    }
}

/// Pick a physical device: prefer an integrated GPU (the engine's home
/// turf), then any discrete GPU, then a software device. First candidate
/// with a usable compute queue family wins.
fn pick_physical_device(instance: &ash::Instance) -> Result<vk::PhysicalDevice> {
    unsafe {
        let devices = instance
            .enumerate_physical_devices()
            .map_err(|e| anyhow::anyhow!("enumerate_physical_devices: {e}"))?;
        let mut usable: Vec<vk::PhysicalDevice> = devices
            .into_iter()
            .filter(|&p| compute_queue_family(instance, p).is_ok())
            .collect();
        // Stable ranking by preference: integrated first.
        usable.sort_by(|&a, &b| {
            let (ta, tb) = (
                instance.get_physical_device_properties(a).device_type,
                instance.get_physical_device_properties(b).device_type,
            );
            rank(ta).cmp(&rank(tb))
        });
        usable
            .first()
            .copied()
            .ok_or_else(|| anyhow::anyhow!("no Vulkan device with a compute queue family"))
    }
}

fn rank(t: vk::PhysicalDeviceType) -> u8 {
    match t {
        vk::PhysicalDeviceType::INTEGRATED_GPU => 0,
        vk::PhysicalDeviceType::DISCRETE_GPU => 1,
        vk::PhysicalDeviceType::VIRTUAL_GPU => 2,
        _ => 3,
    }
}

/// The smallest-indexed queue family exposing compute support.
fn compute_queue_family(instance: &ash::Instance, p: vk::PhysicalDevice) -> Result<u32> {
    unsafe {
        let families = instance.get_physical_device_queue_family_properties(p);
        for (i, f) in families.iter().enumerate() {
            if f.queue_flags.contains(vk::QueueFlags::COMPUTE) && f.queue_count > 0 {
                return Ok(i as u32);
            }
        }
        bail!("device has no compute-capable queue family")
    }
}

/// Pick a UMA-friendly memory type: prefer `DEVICE_LOCAL|HOST_VISIBLE|HOST_COHERENT`
/// (true unified memory — the GPU reads CPU writes with no explicit flush),
/// then any `HOST_VISIBLE|HOST_COHERENT` as a fallback.
fn pick_unified_memory(mem: vk::PhysicalDeviceMemoryProperties) -> Result<u32> {
    let want = vk::MemoryPropertyFlags::DEVICE_LOCAL
        | vk::MemoryPropertyFlags::HOST_VISIBLE
        | vk::MemoryPropertyFlags::HOST_COHERENT;
    let fallback =
        vk::MemoryPropertyFlags::HOST_VISIBLE | vk::MemoryPropertyFlags::HOST_COHERENT;
    for (i, t) in mem.memory_types.iter().enumerate() {
        if (t.property_flags & want) == want {
            return Ok(i as u32);
        }
    }
    for (i, t) in mem.memory_types.iter().enumerate() {
        if (t.property_flags & fallback) == fallback {
            return Ok(i as u32);
        }
    }
    bail!("no host-visible + coherent memory type available")
}

impl Chip {
    /// Wait for all outstanding work on the queue to finish.
    pub fn wait_idle(&self) -> Result<()> {
        unsafe {
            self.device
                .device_wait_idle()
                .map_err(|e| anyhow::anyhow!("device_wait_idle: {e}"))
        }
    }
}

/// A Vulkan buffer bound to mapped unified memory. For a UMA chip the mapping
/// is coherent, so GPU writes become visible to the CPU (and vice versa) with
/// no explicit host-side flush.
///
/// The buffer keeps an `Arc<Chip>` so `Drop` can tear down the underlying
/// Vulkan objects; the chip is dropped only after every buffer is gone.
pub struct Buffer {
    pub buffer: vk::Buffer,
    pub memory: vk::DeviceMemory,
    pub size_bytes: usize,
    ptr: *mut c_void,
    _chip: Arc<Chip>,
}

// A `Buffer` owns raw pointers plus the device that uses them, and management
// happens on exactly one thread (the engine's). Sending across threads is safe
// because the mapped memory is coherent and kernel launches are serialized on
// the engine's queue.
unsafe impl Send for Buffer {}

impl Buffer {
    /// Create a size-aligned storage buffer in the chip's unified memory,
    /// mapped into this process. Copies `init` in when provided, otherwise
    /// zero-fills.
    pub fn new(
        chip: &Arc<Chip>,
        size_bytes: usize,
        usage: vk::BufferUsageFlags,
        init: Option<&[u8]>,
    ) -> Result<Self> {
        unsafe {
            let bi = vk::BufferCreateInfo::default()
                .size(size_bytes as u64)
                .usage(usage | vk::BufferUsageFlags::STORAGE_BUFFER);
            let buffer = chip
                .device
                .create_buffer(&bi, None)
                .map_err(|e| anyhow::anyhow!("create_buffer({size_bytes}): {e}"))?;
            let reqs = chip.device.get_buffer_memory_requirements(buffer);
            let ai = vk::MemoryAllocateInfo::default()
                .allocation_size(reqs.size)
                .memory_type_index(chip.unified_mem);
            let memory = chip
                .device
                .allocate_memory(&ai, None)
                .map_err(|e| anyhow::anyhow!("allocate_memory: {e}"))?;
            chip.device
                .bind_buffer_memory(buffer, memory, 0)
                .map_err(|e| anyhow::anyhow!("bind_buffer_memory: {e}"))?;
            let ptr = chip
                .device
                .map_memory(memory, 0, reqs.size, vk::MemoryMapFlags::empty())
                .map_err(|e| anyhow::anyhow!("map_memory: {e}"))?;
            if let Some(bytes) = init {
                debug_assert!(bytes.len() <= reqs.size as usize);
                std::ptr::copy_nonoverlapping(bytes.as_ptr(), ptr as *mut u8, bytes.len());
            } else {
                std::ptr::write_bytes(ptr as *mut u8, 0, reqs.size as usize);
            }
            Ok(Self {
                buffer,
                memory,
                size_bytes: reqs.size as usize,
                ptr,
                _chip: chip.clone(),
            })
        }
    }

    /// Write `bytes` at a byte offset in mapped memory.
    pub fn write(&self, offset: usize, bytes: &[u8]) {
        debug_assert!(offset + bytes.len() <= self.size_bytes);
        unsafe {
            std::ptr::copy_nonoverlapping(
                bytes.as_ptr(),
                (self.ptr as *mut u8).add(offset),
                bytes.len(),
            );
        }
    }

    /// Read `len` bytes at a byte offset of mapped memory.
    pub fn read(&self, offset: usize, len: usize) -> Vec<u8> {
        debug_assert!(offset + len <= self.size_bytes);
        let mut out = vec![0u8; len];
        unsafe {
            std::ptr::copy_nonoverlapping(
                (self.ptr as *const u8).add(offset),
                out.as_mut_ptr(),
                len,
            );
        }
        out
    }
}

impl Drop for Buffer {
    fn drop(&mut self) {
        unsafe {
            self._chip.device.unmap_memory(self.memory);
            self._chip.device.free_memory(self.memory, None);
            self._chip.device.destroy_buffer(self.buffer, None);
        }
    }
}