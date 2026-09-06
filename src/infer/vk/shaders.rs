//! WGSL kernels for the engine, compiled to SPIR-V at startup with `naga`.
//!
//! Every kernel is generated from a small template so all dtypes share one
//! parameterized GEMV/embed shader that is specialized per quant format. Byte
//! buffers (weights, biases) expose `array<u32>` and are read through
//! [`byte_readers`]; scratch buffers are plain `array<f32>`. Push constants
//! carry offsets/lengths, so one pipeline covers many tensors.
//!
//! The K-quant dequant kernels are transcribed from the reference decoder
//! (`rlx-gguf` / `ggml-quants.c`); the Rust-side reference in
//! [`crate::infer::dequant_ref`] is unit-tested against `rlx_gguf` and the
//! WGSL here mirrors it line-for-line.

use anyhow::Result;

/// Workgroup size for most kernels.
pub const WG: u32 = 128;

/// Dtypes the engine can run on the GPU.
///
/// Byte layouts and block sizes match [`crate::infer::gguf::GgmlDType`]
/// exactly — `block_bytes(elements)` per super-block.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum Dt {
    F32,
    F16,
    Bf16,
    Q8_0,
    Q4_0,
    Q4_1,
    Q5_0,
    Q5_1,
    Q2K,
    Q3K,
    Q4K,
    Q5K,
    Q6K,
    Q8K,
}

impl Dt {
    /// Numeric elements per (super-)block: 256 for K-quants, else 32 or 1.
    pub fn block_elems(self) -> u32 {
        match self {
            Dt::F32 | Dt::F16 | Dt::Bf16 => 1,
            Dt::Q2K | Dt::Q3K | Dt::Q4K | Dt::Q5K | Dt::Q6K | Dt::Q8K => 256,
            _ => 32,
        }
    }

    /// Bytes per (super-)block for the template-generated kernels.
    pub fn block_bytes(self) -> u32 {
        match self {
            Dt::F32 => 4,
            Dt::F16 | Dt::Bf16 => 2,
            Dt::Q8_0 => 2 + 32,
            Dt::Q4_0 => 2 + 16,
            Dt::Q4_1 => 4 + 16,
            Dt::Q5_0 => 2 + 4 + 16,
            Dt::Q5_1 => 4 + 4 + 16,
            Dt::Q2K => 16 + 64 + 4,
            Dt::Q3K => 32 + 64 + 12 + 2,
            Dt::Q4K => 2 + 2 + 12 + 128,
            Dt::Q5K => 2 + 2 + 12 + 32 + 128,
            Dt::Q6K => 128 + 64 + 16 + 2,
            Dt::Q8K => 4 + 256 + 32,
        }
    }

    /// Stable lower-case name used in pipeline debug labels and cache keys.
    pub fn name(self) -> &'static str {
        match self {
            Dt::F32 => "f32",
            Dt::F16 => "f16",
            Dt::Bf16 => "bf16",
            Dt::Q8_0 => "q8_0",
            Dt::Q4_0 => "q4_0",
            Dt::Q4_1 => "q4_1",
            Dt::Q5_0 => "q5_0",
            Dt::Q5_1 => "q5_1",
            Dt::Q2K => "q2_k",
            Dt::Q3K => "q3_k",
            Dt::Q4K => "q4_k",
            Dt::Q5K => "q5_k",
            Dt::Q6K => "q6_k",
            Dt::Q8K => "q8_k",
        }
    }

    /// Whether this dtype packs elements into 256-element K-quant blocks.
    pub fn is_k_quant(self) -> bool {
        matches!(
            self,
            Dt::Q2K | Dt::Q3K | Dt::Q4K | Dt::Q5K | Dt::Q6K | Dt::Q8K
        )
    }
}

/// The `rd_*` byte readers plus `f16_to_f32`/`as_i8` used by every block
/// quant. Reads go through the module-global `wb: array<u32>`.
fn byte_readers() -> &'static str {
    r#"
fn rd_u8(base: u32) -> u32 {
    let i = base >> 2u;
    let s = (base & 3u) * 8u;
    return (wb[i] >> s) & 0xffu;
}

fn rd_u16(base: u32) -> u32 {
    let lo = base >> 2u;
    let s = (base & 3u) * 8u;
    if (s <= 16u) { return (wb[lo] >> s) & 0xffffu; }
    let hi = lo + 1u;
    return ((wb[lo] >> s) | (wb[hi] << (32u - s))) & 0xffffu;
}

fn rd_u32(base: u32) -> u32 {
    let lo = base >> 2u;
    let s = (base & 3u) * 8u;
    if (s == 0u) { return wb[lo]; }
    let hi = lo + 1u;
    return (wb[lo] >> s) | (wb[hi] << (32u - s));
}

fn rd_f32(base: u32) -> f32 {
    return bitcast<f32>(rd_u32(base));
}

fn as_i8(u: u32) -> i32 {
    if (u >= 128u) { return i32(u) - 256; }
    return i32(u);
}

fn f16_to_f32(h: u32) -> f32 {
    let s = (h >> 15u) & 1u;
    let e = (h >> 10u) & 31u;
    let m = h & 1023u;
    var bits: u32;
    if (e == 0u) {
        if (m == 0u) { bits = s << 31u; }
        else {
            var e2 = 113u;
            var m2 = m;
            while ((m2 & 1024u) == 0u) { m2 = m2 << 1u; e2 = e2 - 1u; }
            bits = (s << 31u) | (e2 << 23u) | ((m2 & 1023u) << 13u);
        }
    } else if (e == 31u) {
        bits = (s << 31u) | 0x7f800000u | (m << 13u);
    } else {
        bits = (s << 31u) | ((e + 112u) << 23u) | (m << 13u);
    }
    return bitcast<f32>(bits);
}
"#
}

/// WGSL function `dequant_elem(base: u32, e: u32) -> f32` for one dtype,
/// where `base` is the byte offset of the *row* start and `e` the element
/// index within the row. Mirrors the reference decoders in `dequant_ref.rs`.
fn dequant_elem_fn(dt: Dt) -> String {
    let body = match dt {
        Dt::F32 => {
            "    return rd_f32(base + e * 4u);".to_string()
        }
        Dt::F16 => {
            "    return f16_to_f32(rd_u16(base + e * 2u));".to_string()
        }
        Dt::Bf16 => {
            "    return bitcast<f32>(rd_u16(base + e * 2u) << 16u);".to_string()
        }
        Dt::Q8_0 => {
            r#"    let bb = base + (e >> 5u) * 34u;
    let d = f16_to_f32(rd_u16(bb));
    let q = as_i8(rd_u8(bb + 2u + (e & 31u)));
    return d * f32(q);"#
                .to_string()
        }
        Dt::Q4_0 => {
            r#"    let bb = base + (e >> 5u) * 18u;
    let d = f16_to_f32(rd_u16(bb));
    let k = e & 31u;
    var v: u32;
    if (k < 16u) { v = rd_u8(bb + 2u + k) & 0x0fu; }
    else { v = rd_u8(bb + 2u + k - 16u) >> 4u; }
    return d * f32(i32(v) - 8);"#
                .to_string()
        }
        Dt::Q4_1 => {
            r#"    let bb = base + (e >> 5u) * 34u;
    let d = f16_to_f32(rd_u16(bb));
    let m = f16_to_f32(rd_u16(bb + 2u));
    let k = e & 31u;
    var v: u32;
    if (k < 16u) { v = rd_u8(bb + 4u + k) & 0x0fu; }
    else { v = rd_u8(bb + 4u + k - 16u) >> 4u; }
    return d * f32(v) + m;"#
                .to_string()
        }
        Dt::Q5_0 => {
            r#"    let bb = base + (e >> 5u) * 22u;
    let d = f16_to_f32(rd_u16(bb));
    let qh = rd_u32(bb + 2u);
    let k = e & 31u;
    var v: u32;
    if (k < 16u) { v = (rd_u8(bb + 6u + k) & 0x0fu) | (((qh >> k) & 1u) << 4u); }
    else { let j = k - 16u; v = (rd_u8(bb + 6u + j) >> 4u) | (((qh >> (j + 16u)) & 1u) << 4u); }
    return d * f32(i32(v) - 16);"#
                .to_string()
        }
        Dt::Q5_1 => {
            r#"    let bb = base + (e >> 5u) * 24u;
    let d = f16_to_f32(rd_u16(bb));
    let m = f16_to_f32(rd_u16(bb + 2u));
    let qh = rd_u32(bb + 4u);
    let k = e & 31u;
    var v: u32;
    if (k < 16u) { v = (rd_u8(bb + 8u + k) & 0x0fu) | (((qh >> k) & 1u) << 4u); }
    else { let j = k - 16u; v = (rd_u8(bb + 8u + j) >> 4u) | (((qh >> (j + 16u)) & 1u) << 4u); }
    return d * f32(v) + m;"#
                .to_string()
        }
        Dt::Q8K => {
            r#"    let bb = base + (e >> 8u) * 292u;
    let d = rd_f32(bb);
    let q = as_i8(rd_u8(bb + 4u + (e & 255u)));
    return d * f32(q);"#
                .to_string()
        }
        Dt::Q2K => {
            r#"    let bb = base + (e >> 8u) * 84u;
    let o = (e >> 7u) & 1u;
    let rem = e & 127u;
    let g = rem >> 5u;
    let sub = rem & 31u;
    let ll = sub & 15u;
    let half = (sub >> 4u) & 1u;
    let shift = g * 2u;
    let qidx = o * 32u + half * 16u + ll;
    let sidx = o * 8u + g * 2u + half;
    let qb = rd_u8(bb + 16u + qidx);
    let sc = rd_u8(bb + sidx);
    let d = f16_to_f32(rd_u16(bb + 80u));
    let mn = f16_to_f32(rd_u16(bb + 82u));
    let val = (qb >> shift) & 3u;
    return d * f32(sc & 0x0fu) * f32(val) - mn * f32(sc >> 4u);"#
                .to_string()
        }
        Dt::Q3K => {
            r#"
fn q3_scale(base: u32, r: u32) -> i32 {
    let m1 = 0x03030303u;
    let m2 = 0x0f0f0f0fu;
    let a0 = rd_u32(base);
    let a1 = rd_u32(base + 4u);
    let a2 = rd_u32(base + 8u);
    let ax0 = (a0 & m2) | ((a2 & m1) << 4u);
    let ax1 = (a1 & m2) | (((a2 >> 2u) & m1) << 4u);
    let ax2 = ((a0 >> 4u) & m2) | (((a2 >> 4u) & m1) << 4u);
    let ax3 = ((a1 >> 4u) & m2) | (((a2 >> 6u) & m1) << 4u);
    let w = r >> 2u;
    var word: u32;
    if (w == 0u) { word = ax0; }
    else if (w == 1u) { word = ax1; }
    else if (w == 2u) { word = ax2; }
    else { word = ax3; }
    let byte = (word >> ((r & 3u) * 8u)) & 0xffu;
    return as_i8(byte);
}

fn dequant_elem(base: u32, e: u32) -> f32 {
    let bb = base + (e >> 8u) * 110u;
    let o = (e >> 7u) & 1u;
    let rem = e & 127u;
    let g = rem >> 5u;
    let sub = rem & 31u;
    let ll = sub & 15u;
    let half = (sub >> 4u) & 1u;
    let shift = g * 2u;
    let qidx = o * 32u + half * 16u + ll;
    let sidx = o * 8u + g * 2u + half;
    let d_all = f16_to_f32(rd_u16(bb + 108u));
    let dl = d_all * f32(q3_scale(bb + 96u, sidx) - 32);
    let qb = rd_u8(bb + 32u + qidx);
    let hb = rd_u8(bb + half * 16u + ll);
    let mbit = 1u << (o * 4u + g);
    let h = select(4u, 0u, (hb & mbit) != 0u);
    let val = i32((qb >> shift) & 3u) - i32(h);
    return dl * f32(val);
}"#
                .to_string()
        }
        Dt::Q4K => {
            r#"    let bb = base + (e >> 8u) * 144u;
    let d = f16_to_f32(rd_u16(bb));
    let dmn = f16_to_f32(rd_u16(bb + 2u));
    let g = e / 32u;
    let s = e % 32u;
    let scb = bb + 4u;
    var sc: u32;
    var mn: u32;
    if (g < 4u) {
        sc = rd_u8(scb + g) & 63u;
        mn = rd_u8(scb + g + 4u) & 63u;
    } else {
        sc = (rd_u8(scb + g + 4u) & 0x0fu) | ((rd_u8(scb + g - 4u) >> 6u) << 4u);
        mn = (rd_u8(scb + g + 4u) >> 4u) | ((rd_u8(scb + g) >> 6u) << 4u);
    }
    let qb = rd_u8(bb + 16u + (g >> 1u) * 32u + s);
    var val: u32;
    if ((g & 1u) == 0u) { val = qb & 0x0fu; }
    else { val = qb >> 4u; }
    return d * f32(sc) * f32(val) - dmn * f32(mn);"#
                .to_string()
        }
        Dt::Q5K => {
            r#"    let bb = base + (e >> 8u) * 176u;
    let d = f16_to_f32(rd_u16(bb));
    let dmn = f16_to_f32(rd_u16(bb + 2u));
    let g = e / 32u;
    let s = e % 32u;
    let scb = bb + 4u;
    var sc: u32;
    var mn: u32;
    if (g < 4u) {
        sc = rd_u8(scb + g) & 63u;
        mn = rd_u8(scb + g + 4u) & 63u;
    } else {
        sc = (rd_u8(scb + g + 4u) & 0x0fu) | ((rd_u8(scb + g - 4u) >> 6u) << 4u);
        mn = (rd_u8(scb + g + 4u) >> 4u) | ((rd_u8(scb + g) >> 6u) << 4u);
    }
    let qb = rd_u8(bb + 48u + (g >> 1u) * 32u + s);
    let u = 1u << (2u * (g >> 1u) + (g & 1u));
    var lo: u32;
    var hi: u32;
    if ((g & 1u) == 0u) {
        lo = qb & 0x0fu;
        hi = select(0u, 16u, (rd_u8(bb + 16u + s) & u) != 0u);
    } else {
        lo = qb >> 4u;
        hi = select(0u, 16u, (rd_u8(bb + 16u + s) & u) != 0u);
    }
    return d * f32(sc) * f32(lo + hi) - dmn * f32(mn);"#
                .to_string()
        }
        Dt::Q6K => {
            r#"    let bb = base + (e >> 8u) * 210u;
    let h = (e >> 7u) & 1u;
    let sub = e & 127u;
    let seg = sub >> 5u;
    let l = sub & 31u;
    let d = f16_to_f32(rd_u16(bb + 208u));
    let qlidx = h * 64u + l + (seg & 1u) * 32u;
    let qh_b = rd_u8(bb + 128u + h * 32u + l);
    var nib: u32;
    if ((seg & 2u) == 0u) { nib = rd_u8(bb + qlidx) & 0x0fu; }
    else { nib = rd_u8(bb + qlidx) >> 4u; }
    let m = (qh_b >> (seg * 2u)) & 3u;
    let sidx = h * 8u + (l >> 4u) + seg * 2u;
    let sc = as_i8(rd_u8(bb + 192u + sidx));
    return d * f32(sc) * f32(i32(nib | (m << 4u)) - 32);"#
                .to_string()
        }
    };
    let head = match dt {
        Dt::Q3K => "",
        _ => "fn dequant_elem(base: u32, e: u32) -> f32 {",
    };
    let tail = if dt == Dt::Q3K { "" } else { "}" };
    format!("{head}\n{body}\n{tail}")
}

/// One full shader module (WGSL) for the GEMV of a given dtype.
pub fn gemv_wgsl(dt: Dt) -> String {
    format!(
        r#"@group(0) @binding(0) var<storage, read> wb: array<u32>;
@group(0) @binding(1) var<storage, read> xv: array<f32>;
@group(0) @binding(2) var<storage, read_write> outv: array<f32>;

{readers}
{dequant}

var<workgroup> red: array<f32, 128>;

struct PC {{ base: u32, rowlen: u32, ncols: u32, row0: u32 }}
@group(0) @binding(3) var<uniform> pc: PC;

@compute @workgroup_size(128)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {{
    let row = pc.row0 + gid.y;
    let tid = gid.x;
    let rbase = pc.base + row * pc.rowlen;
    var acc: f32 = 0.0;
    for (var e = tid; e < pc.ncols; e = e + 128u) {{
        acc = acc + dequant_elem(rbase, e) * xv[e];
    }}
    red[tid] = acc;
    workgroupBarrier();
    for (var s = 64u; s > 0u; s = s >> 1u) {{
        if (tid < s) {{ red[tid] = red[tid] + red[tid + s]; }}
        workgroupBarrier();
    }}
    if (tid == 0u) {{ outv[row] = red[0]; }}
}}"#,
        readers = byte_readers(),
        dequant = dequant_elem_fn(dt),
    )
}

/// Shader that embeds rows of a token-embedding tensor: one workgroup per
/// 256-element slice of a token row, thread `tid` computes element
/// `blk * 256 + tid` (masked past `ncols`).
pub fn embed_wgsl(dt: Dt) -> String {
    format!(
        r#"@group(0) @binding(0) var<storage, read> wb: array<u32>;
@group(0) @binding(1) var<storage, read> ids: array<u32>;
@group(0) @binding(2) var<storage, read_write> outv: array<f32>;

{readers}
{dequant}

struct PC {{ base: u32, rowlen: u32, ncols: u32, tok0: u32 }}
@group(0) @binding(3) var<uniform> pc: PC;

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {{
    let bid = gid.x >> 8u;
    let tid = gid.x & 255u;
    let nblk = (pc.ncols + 255u) / 256u;
    let slc = bid % nblk;
    let tok = pc.tok0 + bid / nblk;
    let e = slc * 256u + tid;
    if (e >= pc.ncols) {{ return; }}
    let tokid = ids[tok];
    let rbase = pc.base + tokid * pc.rowlen;
    outv[e] = dequant_elem(rbase, e);
}}"#,
        readers = byte_readers(),
        dequant = dequant_elem_fn(dt),
    )
}

/// rms-norm: one workgroup per row. `nrows>1` powers per-head q/k norm.
pub fn rmsnorm_wgsl() -> String {
    format!(
        r#"@group(0) @binding(0) var<storage, read> wb: array<u32>;
@group(0) @binding(1) var<storage, read> inv: array<f32>;
@group(0) @binding(2) var<storage, read_write> outv: array<f32>;

{readers}

var<workgroup> red: array<f32, 128>;
var<workgroup> lscale: f32;

struct PC {{ wbase: u32, len: u32, nrows: u32, eps_bits: u32 }}
@group(0) @binding(3) var<uniform> pc: PC;

@compute @workgroup_size(128)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {{
    let row = gid.y;
    let tid = gid.x;
    let base = row * pc.len;
    var acc: f32 = 0.0;
    for (var e = tid; e < pc.len; e = e + 128u) {{
        let v = inv[base + e];
        acc = acc + v * v;
    }}
    red[tid] = acc;
    workgroupBarrier();
    for (var s = 64u; s > 0u; s = s >> 1u) {{
        if (tid < s) {{ red[tid] = red[tid] + red[tid + s]; }}
        workgroupBarrier();
    }}
    if (tid == 0u) {{
        let eps = bitcast<f32>(pc.eps_bits);
        lscale = 1.0 / sqrt(red[0] / f32(pc.len) + eps);
    }}
    workgroupBarrier();
    for (var e = tid; e < pc.len; e = e + 128u) {{
        let w = rd_f32(pc.wbase + (base + e) * 4u);
        outv[base + e] = inv[base + e] * w * lscale;
    }}
}}"#,
        readers = byte_readers(),
    )
}

/// In-place elementwise add: `outv[i] += inv[i]`.
pub fn add_wgsl() -> String {
    r#"@group(0) @binding(0) var<storage, read_write> outv: array<f32>;
@group(0) @binding(1) var<storage, read> inv: array<f32>;

struct PC { len: u32, _p0: u32, _p1: u32, _p2: u32 }
@group(0) @binding(2) var<uniform> pc: PC;

@compute @workgroup_size(128)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    for (var e = gid.x; e < pc.len; e = e + 128u) {
        outv[e] = outv[e] + inv[e];
    }
}"#
    .to_string()
}

/// RoPE on the query and key row buffers (each packed per-head contiguous),
/// using one thread per rotation-pair. `neox=0` selects interleaved pairs.
pub fn rope_wgsl() -> String {
    r#"@group(0) @binding(0) var<storage, read_write> qv: array<f32>;
@group(0) @binding(1) var<storage, read_write> kv: array<f32>;
@group(0) @binding(2) var<storage, read> freq: array<f32>;

struct PC { pos: u32, hd: u32, half: u32, neox: u32, qdim: u32, kvdim: u32, _p0: u32, _p1: u32 }
@group(0) @binding(3) var<uniform> pc: PC;

@compute @workgroup_size(128)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let half = pc.hd / 2u;
    let npairs = (pc.qdim + pc.kvdim) / 2u;
    if (gid.x >= npairs) { return; }
    let is_k = gid.x >= (pc.qdim / 2u);
    let pq = gid.x - select(0u, pc.qdim / 2u, is_k);
    let head = pq / half;
    let within = pq - head * half;
    let theta = freq[within] * f32(pc.pos);
    let c = cos(theta);
    let s = sin(theta);
    var i0: u32 = 0u;
    var i1: u32 = 0u;
    if (pc.neox == 0u) {
        i0 = head * pc.hd + 2u * within;
        i1 = i0 + 1u;
    } else {
        i0 = head * pc.hd + within;
        i1 = i0 + half;
    }
    let base = select(0u, pc.qdim, is_k);
    var n0: f32 = 0.0;
    var n1: f32 = 0.0;
    if (is_k) {
        let a0 = kv[base + i0];
        let a1 = kv[base + i1];
        n0 = c * a0 - s * a1;
        n1 = s * a0 + c * a1;
        kv[base + i0] = n0;
        kv[base + i1] = n1;
    } else {
        let a0 = qv[base + i0];
        let a1 = qv[base + i1];
        n0 = c * a0 - s * a1;
        n1 = s * a0 + c * a1;
        qv[base + i0] = n0;
        qv[base + i1] = n1;
    }
}"#
    .to_string()
}

/// Store the (post-rope) key and value rows into the layer cache.
pub fn store_kv_wgsl() -> String {
    r#"@group(0) @binding(0) var<storage, read> kvv: array<f32>;
@group(0) @binding(1) var<storage, read> vvv: array<f32>;
@group(0) @binding(2) var<storage, read_write> kc: array<f32>;
@group(0) @binding(3) var<storage, read_write> vc: array<f32>;

struct PC { layer: u32, pos: u32, kvdim: u32, cap: u32 }
@group(0) @binding(4) var<uniform> pc: PC;

@compute @workgroup_size(128)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let i = gid.x;
    if (i >= pc.kvdim) { return; }
    kc[pc.layer * pc.cap * pc.kvdim + pc.pos * pc.kvdim + i] = kvv[i];
    vc[pc.layer * pc.cap * pc.kvdim + pc.pos * pc.kvdim + i] = vvv[i];
}"#
    .to_string()
}

/// Flash-style attention with online softmax: one workgroup per query head.
pub fn attn_wgsl() -> String {
    r#"@group(0) @binding(0) var<storage, read> qv: array<f32>;
@group(0) @binding(1) var<storage, read> kc: array<f32>;
@group(0) @binding(2) var<storage, read> vc: array<f32>;
@group(0) @binding(3) var<storage, read_write> outv: array<f32>;

var<workgroup> wred: array<f32, 128>;
var<workgroup> wacc: array<f32, 128>;
var<workgroup> qsh: array<f32, 128>;
var<workgroup> wm: f32;
var<workgroup> wsum: f32;
var<workgroup> wcorr: f32;
var<workgroup> wterm: f32;

struct PC { layer: u32, pos: u32, win_start: u32, n_heads: u32, groups: u32, hd: u32, scale_bits: u32, cap: u32, kvdim: u32, _p0: u32, _p1: u32, _p2: u32 }
@group(0) @binding(4) var<uniform> pc: PC;

@compute @workgroup_size(128)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let hq = gid.x;
    let tid = gid.y;
    let scale = bitcast<f32>(pc.scale_bits);
    let group = hq / pc.groups;
    let qbase = hq * pc.hd;
    let lbase = pc.layer * pc.cap * pc.kvdim + group * pc.hd;
    if (tid < pc.hd) {
        qsh[tid] = qv[qbase + tid];
    }
    wacc[tid] = 0.0;
    workgroupBarrier();
    if (tid == 0u) {
        wm = -1.0e30;
        wsum = 0.0;
    }
    workgroupBarrier();
    for (var p = pc.win_start; p <= pc.pos; p = p + 1u) {
        var contrib: f32 = 0.0;
        if (tid < pc.hd) {
            let cb = lbase + p * pc.kvdim;
            for (var kk = tid; kk < pc.hd; kk = kk + 128u) {
                contrib = contrib + qsh[kk] * kc[cb + kk];
            }
        }
        wred[tid] = contrib;
        workgroupBarrier();
        for (var s = 64u; s > 0u; s = s >> 1u) {
            if (tid < s) { wred[tid] = wred[tid] + wred[tid + s]; }
            workgroupBarrier();
        }
        if (tid == 0u) {
            let sp = wred[0] * scale;
            let nm = max(wm, sp);
            let corr = exp(wm - nm);
            wm = nm;
            wsum = wsum * corr + exp(sp - nm);
            wcorr = corr;
            wterm = exp(sp - nm);
        }
        workgroupBarrier();
        if (tid < pc.hd) {
            wacc[tid] = wacc[tid] * wcorr + wterm * vc[lbase + p * pc.kvdim + tid];
        }
        workgroupBarrier();
    }
    if (tid < pc.hd) {
        outv[qbase + tid] = wacc[tid] / wsum;
    }
}"#
    .to_string()
}

/// Activation (SiLU or tanh-approximated GELU) applied to `uv` scaled by
/// `gv` in place: `uv[e] = act(gv[e]) * uv[e]`.
pub fn act_wgsl() -> String {
    r#"@group(0) @binding(0) var<storage, read> gv: array<f32>;
@group(0) @binding(1) var<storage, read_write> uv: array<f32>;

struct PC { len: u32, mode: u32, _p0: u32, _p1: u32 }
@group(0) @binding(2) var<uniform> pc: PC;

@compute @workgroup_size(128)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    for (var e = gid.x; e < pc.len; e = e + 128u) {
        let g = gv[e];
        var y: f32;
        if (pc.mode == 0u) {
            y = g / (1.0 + exp(-g));
        } else {
            let c = 0.7978845608;
            let x = g + 0.044715 * g * g * g;
            y = 0.5 * g * (1.0 + tanh(c * x));
        }
        uv[e] = y * uv[e];
    }
}"#
    .to_string()
}

/// Compile a WGSL source string into a SPIR-V binary module (u32 words).
pub fn compile(wgsl: &str) -> Result<Vec<u32>> {
    use naga::back::spv;
    use naga::front::wgsl;
    use naga::valid::{Capabilities, ValidationFlags, Validator};

    let module = wgsl::parse_str(wgsl).map_err(|e| anyhow::anyhow!("wgsl parse: {e}"))?;
    let info = Validator::new(ValidationFlags::all(), Capabilities::all())
        .validate(&module)
        .map_err(|e| anyhow::anyhow!("naga validate: {e}"))?;
    let options = spv::Options::default();
    let pipeline = spv::PipelineOptions {
        shader_stage: naga::ShaderStage::Compute,
        entry_point: "main".to_string(),
    };
    let words = spv::write_vec(&module, &info, &options, Some(&pipeline))
        .map_err(|e| anyhow::anyhow!("spv generate: {e}"))?;
    Ok(words)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn block_sizes_match_gguf_reference() {
        // Cross-checked against the reference byte lengths in `gguf.rs`.
        assert_eq!(Dt::F32.block_bytes(), 4);
        assert_eq!(Dt::F16.block_bytes(), 2);
        assert_eq!(Dt::Q8_0.block_bytes(), 34);
        assert_eq!(Dt::Q4_0.block_bytes(), 18);
        assert_eq!(Dt::Q4_1.block_bytes(), 20);
        assert_eq!(Dt::Q5_0.block_bytes(), 22);
        assert_eq!(Dt::Q5_1.block_bytes(), 24);
        assert_eq!(Dt::Q2K.block_bytes(), 84);
        assert_eq!(Dt::Q3K.block_bytes(), 110);
        assert_eq!(Dt::Q4K.block_bytes(), 144);
        assert_eq!(Dt::Q5K.block_bytes(), 176);
        assert_eq!(Dt::Q6K.block_bytes(), 210);
        assert_eq!(Dt::Q8K.block_bytes(), 292);
    }

    #[test]
    fn all_gemv_shaders_compile() {
        for dt in [
            Dt::F32, Dt::F16, Dt::Bf16, Dt::Q8_0, Dt::Q4_0, Dt::Q4_1, Dt::Q5_0,
            Dt::Q5_1, Dt::Q2K, Dt::Q3K, Dt::Q4K, Dt::Q5K, Dt::Q6K, Dt::Q8K,
        ] {
            let wgsl = gemv_wgsl(dt);
            let spv = compile(&wgsl).unwrap_or_else(|e| panic!("gemv {dt:?}: {e}"));
            assert!(spv.len() > 8, "gemv {dt:?} produced empty SPIR-V");
        }
    }

    #[test]
    fn all_embed_shaders_compile() {
        for dt in [
            Dt::F32, Dt::F16, Dt::Bf16, Dt::Q8_0, Dt::Q4_0, Dt::Q4_1, Dt::Q5_0,
            Dt::Q5_1, Dt::Q2K, Dt::Q3K, Dt::Q4K, Dt::Q5K, Dt::Q6K, Dt::Q8K,
        ] {
            let spv = compile(&embed_wgsl(dt))
                .unwrap_or_else(|e| panic!("embed {dt:?}: {e}"));
            assert!(spv.len() > 8, "embed {dt:?} produced empty SPIR-V");
        }
    }

    #[test]
    fn support_shaders_compile() {
        for (name, src) in [
            ("rmsnorm", rmsnorm_wgsl()),
            ("add", add_wgsl()),
            ("rope", rope_wgsl()),
            ("store_kv", store_kv_wgsl()),
            ("attn", attn_wgsl()),
            ("act", act_wgsl()),
        ] {
            let spv = compile(&src).unwrap_or_else(|e| panic!("{name}: {e}"));
            assert!(spv.len() > 8, "{name} produced empty SPIR-V");
        }
    }
}