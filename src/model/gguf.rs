//! Quantized GGUF inference: an mmap-backed engine that runs the transformer
//! forward pass on the CPU with f32 activations against packed weights.
//!
//! Weight matrices loaded from a Q4_K file stay packed in memory (≈ the file
//! size on disk, not the f32 image) and are consumed through a GEMV dot: the
//! activation is quantized to Q8_K once per matrix, then every output row
//! int8-dots against the same packed strip (`rlx_gguf::q4_k_dot_q8_k`).
//! Matrices whose contiguous dimension is not Q4_K-block aligned, and any
//! non-Q4_K dtype, fall back to a block-dequant dot (the loader accepts the
//! full K-quant family plus F32/F16/BF16/Q{4,5,8}_0/Q8_K).
//!
//! Tensor layout and naming mirror the exporter in `crate::export`: 2-D
//! linear weights are stored PyTorch-style `[out, in]` row-major and declared
//! as GGUF `[in, out]` (so `tensor.shape == [cols, rows]`), embeddings are
//! `[vocab, d_model]` as-is, and Gemma norm gains are baked +1 (the loader
//! subtracts 1 before use). RoPE pairing follows the GGUF architecture:
//! `llama` weights are stored rope-permuted (loader applies interleaved
//! rotation), `qwen2`/`gemma` weights are not (NEOX half-split rotation) —
//! see `crate::export::permute_rope_rows`.

use std::collections::HashMap;
use std::path::Path;
use std::sync::Arc;

use anyhow::{Context, Result, bail};
use rlx_gguf::{
    GgmlType, GgufFile, MetaValue, Q4K_BLOCK_BYTES, Q8K_BLOCK_BYTES, QK_K, dequant_q2_k_block,
    dequant_q3_k_block, dequant_q4_k_block, dequant_q5_k_block, dequant_q6_k_block, q4_k_dot_q8_k,
    quantize_q8_k_row,
};

use super::model::LlmModelConfig;
use crate::data::TokenizerStore;
use crate::generate::{GenerateConfig, sample_next_token_cpu};

/// A reference to one 2-D weight matrix living inside the shared mmap.
///
/// `rows`/`cols` describe the PyTorch `[out, in]` storage shape: `rows`
/// counts output rows (each GEMV produces one output element), `cols` is the
/// contiguous activation dimension wrapped around Q4_K super-blocks.
#[derive(Clone)]
struct MatRef {
    file: Arc<GgufFile>,
    name: String,
    rows: usize,
    cols: usize,
}

impl MatRef {
    fn dtype(&self) -> GgmlType {
        self.file.tensors[&self.name].dtype
    }

    fn bytes(&self) -> Result<&[u8]> {
        let t = &self.file.tensors[&self.name];
        self.file
            .tensor_bytes(t)
            .with_context(|| format!("tensor `{}` unreadable", self.name))
    }
}

fn is_k_quant(dt: GgmlType) -> bool {
    matches!(
        dt,
        GgmlType::Q2K | GgmlType::Q3K | GgmlType::Q4K | GgmlType::Q5K | GgmlType::Q6K
    )
}

fn cols_q4k_aligned(cols: usize) -> bool {
    cols.is_multiple_of(QK_K)
}

/// On-disk bytes per QK_K super-block for each K-quant layout.
fn k_block_bytes(dt: GgmlType) -> usize {
    match dt {
        GgmlType::Q2K => 84,
        GgmlType::Q3K => 110,
        GgmlType::Q4K => Q4K_BLOCK_BYTES,
        GgmlType::Q5K => 176,
        GgmlType::Q6K => 210,
        _ => unreachable!("k_block_bytes called for non-K quant"),
    }
}

fn is_supported(dt: GgmlType) -> bool {
    matches!(
        dt,
        GgmlType::F32
            | GgmlType::F16
            | GgmlType::BF16
            | GgmlType::Q4_0
            | GgmlType::Q4_1
            | GgmlType::Q5_0
            | GgmlType::Q5_1
            | GgmlType::Q8_0
            | GgmlType::Q8K
            | GgmlType::Q2K
            | GgmlType::Q3K
            | GgmlType::Q4K
            | GgmlType::Q5K
            | GgmlType::Q6K
    )
}

#[inline]
fn f16_to_f32(bits: u16) -> f32 {
    half::f16::from_bits(bits).to_f32()
}

#[inline]
fn bf16_to_f32(bits: u16) -> f32 {
    f32::from_bits((bits as u32) << 16)
}

fn dequant_k_block(dt: GgmlType, block: &[u8], out: &mut [f32; QK_K]) {
    match dt {
        GgmlType::Q2K => dequant_q2_k_block(block, out),
        GgmlType::Q3K => dequant_q3_k_block(block, out),
        GgmlType::Q4K => dequant_q4_k_block(block, out),
        GgmlType::Q5K => dequant_q5_k_block(block, out),
        GgmlType::Q6K => dequant_q6_k_block(block, out),
        _ => unreachable!(),
    }
}

/// Dot product of one packed row against an f32 activation slice.
///
/// `row` covers exactly `cols` elements of a matrix; `scratch` is reused for
/// K-quant super-block dequantization.
fn row_dot(dt: GgmlType, row: &[u8], cols: usize, x: &[f32], scratch: &mut [f32; QK_K]) -> f32 {
    let mut acc = 0.0f32;
    match dt {
        GgmlType::F32 => {
            for (j, &xj) in x.iter().take(cols).enumerate() {
                let o = j * 4;
                acc += f32::from_le_bytes(row[o..o + 4].try_into().unwrap()) * xj;
            }
        }
        GgmlType::F16 => {
            for (j, &xj) in x.iter().take(cols).enumerate() {
                let o = j * 2;
                acc += f16_to_f32(u16::from_le_bytes([row[o], row[o + 1]])) * xj;
            }
        }
        GgmlType::BF16 => {
            for (j, &xj) in x.iter().take(cols).enumerate() {
                let o = j * 2;
                acc += bf16_to_f32(u16::from_le_bytes([row[o], row[o + 1]])) * xj;
            }
        }
        GgmlType::Q8_0 => {
            let nb = cols / 32;
            for b in 0..nb {
                let o = b * 34;
                let d = f16_to_f32(u16::from_le_bytes([row[o], row[o + 1]]));
                let qs = &row[o + 2..o + 34];
                for j in 0..32 {
                    acc += d * (qs[j] as i8) as f32 * x[b * 32 + j];
                }
            }
        }
        GgmlType::Q4_0 => {
            let nb = cols / 32;
            for b in 0..nb {
                let o = b * 18;
                let d = f16_to_f32(u16::from_le_bytes([row[o], row[o + 1]]));
                let qs = &row[o + 2..o + 18];
                let base = b * 32;
                for j in 0..16 {
                    let lo = (qs[j] & 0x0f) as i32 - 8;
                    let hi = (qs[j] >> 4) as i32 - 8;
                    acc += d * (lo as f32 * x[base + j] + hi as f32 * x[base + j + 16]);
                }
            }
        }
        GgmlType::Q4_1 => {
            let nb = cols / 32;
            for b in 0..nb {
                let o = b * 20;
                let d = f16_to_f32(u16::from_le_bytes([row[o], row[o + 1]]));
                let m = f16_to_f32(u16::from_le_bytes([row[o + 2], row[o + 3]]));
                let qs = &row[o + 4..o + 20];
                let base = b * 32;
                for j in 0..16 {
                    let lo = (qs[j] & 0x0f) as i32;
                    let hi = (qs[j] >> 4) as i32;
                    acc += d * (lo as f32 * x[base + j] + m * x[base + j])
                        + d * (hi as f32 * x[base + j + 16] + m * x[base + j + 16]);
                }
            }
        }
        GgmlType::Q5_0 => {
            let nb = cols / 32;
            for b in 0..nb {
                let o = b * 22;
                let d = f16_to_f32(u16::from_le_bytes([row[o], row[o + 1]]));
                let qh = u32::from_le_bytes([row[o + 2], row[o + 3], row[o + 4], row[o + 5]]);
                let qs = &row[o + 6..o + 22];
                let base = b * 32;
                for j in 0..16 {
                    let lo = ((qs[j] & 0x0f) as u32 | (((qh >> j) & 1) << 4)) as i32 - 16;
                    let hi = ((qs[j] >> 4) as u32 | (((qh >> (j + 16)) & 1) << 4)) as i32 - 16;
                    acc += d * (lo as f32 * x[base + j] + hi as f32 * x[base + j + 16]);
                }
            }
        }
        GgmlType::Q5_1 => {
            let nb = cols / 32;
            for b in 0..nb {
                let o = b * 24;
                let d = f16_to_f32(u16::from_le_bytes([row[o], row[o + 1]]));
                let m = f16_to_f32(u16::from_le_bytes([row[o + 2], row[o + 3]]));
                let qh = u32::from_le_bytes([row[o + 4], row[o + 5], row[o + 6], row[o + 7]]);
                let qs = &row[o + 8..o + 24];
                let base = b * 32;
                for j in 0..16 {
                    let lo = (qs[j] & 0x0f) as u32 | (((qh >> j) & 1) << 4);
                    let hi = (qs[j] >> 4) as u32 | (((qh >> (j + 16)) & 1) << 4);
                    acc += d * lo as f32 * x[base + j]
                        + m * x[base + j]
                        + d * hi as f32 * x[base + j + 16]
                        + m * x[base + j + 16];
                }
            }
        }
        GgmlType::Q8K => {
            let nb = cols / QK_K;
            for b in 0..nb {
                let o = b * Q8K_BLOCK_BYTES;
                let d = f32::from_le_bytes(row[o..o + 4].try_into().unwrap());
                let qs = &row[o + 4..o + 4 + QK_K];
                for j in 0..QK_K {
                    acc += d * (qs[j] as i8) as f32 * x[b * QK_K + j];
                }
            }
        }
        GgmlType::Q2K | GgmlType::Q3K | GgmlType::Q4K | GgmlType::Q5K | GgmlType::Q6K => {
            let nb = cols / QK_K;
            let bsz = k_block_bytes(dt);
            for b in 0..nb {
                let o = b * bsz;
                dequant_k_block(dt, &row[o..o + bsz], scratch);
                for j in 0..QK_K {
                    acc += scratch[j] * x[b * QK_K + j];
                }
            }
        }
        _ => unreachable!("row_dot called for unsupported dtype"),
    }
    acc
}

fn row_bytes(dt: GgmlType, cols: usize) -> usize {
    if is_k_quant(dt) {
        cols / QK_K * k_block_bytes(dt)
    } else {
        match dt {
            GgmlType::F32 => cols * 4,
            GgmlType::F16 | GgmlType::BF16 => cols * 2,
            GgmlType::Q4_0 => cols / 32 * 18,
            GgmlType::Q4_1 => cols / 32 * 20,
            GgmlType::Q5_0 => cols / 32 * 22,
            GgmlType::Q5_1 => cols / 32 * 24,
            GgmlType::Q8_0 => cols / 32 * 34,
            GgmlType::Q8K => cols / QK_K * Q8K_BLOCK_BYTES,
            _ => unreachable!("row_bytes called for unsupported dtype"),
        }
    }
}

/// Compute `out[r] = dot(row_r, x)` for every row of `m`.
///
/// The Q4_K fast path quantizes the shared activation strip once and uses the
/// fixed-point dot on every second large matrix (up to q/k/v/o + gate/up/down,
/// 7 GEMVs per layer); everything else walks the packed rows with the
/// block-dequant dot.
fn gemv(m: &MatRef, x: &[f32], out: &mut [f32], scratch: &mut [f32; QK_K]) {
    debug_assert_eq!(x.len(), m.cols);
    debug_assert_eq!(out.len(), m.rows);
    let dt = m.dtype();
    let bytes = m.bytes().unwrap_or_else(|e| panic!("{e}"));
    if dt == GgmlType::Q4K && cols_q4k_aligned(m.cols) {
        let blocks = m.cols / QK_K;
        let row_len = blocks * Q4K_BLOCK_BYTES;
        let mut q8 = vec![0u8; blocks * Q8K_BLOCK_BYTES];
        quantize_q8_k_row(x, q8.as_mut_slice());
        for (r, o) in out.iter_mut().enumerate() {
            let off = r * row_len;
            let mut acc = 0.0f32;
            for b in 0..blocks {
                // rlx's dot kernel handles a single QK_K super-block; sum the
                // block-wise dots when the row spans several (e.g. MLP down).
                acc += q4_k_dot_q8_k(
                    &bytes[off + b * Q4K_BLOCK_BYTES..off + (b + 1) * Q4K_BLOCK_BYTES],
                    &q8[b * Q8K_BLOCK_BYTES..(b + 1) * Q8K_BLOCK_BYTES],
                );
            }
            *o = acc;
        }
        return;
    }
    let rlen = row_bytes(dt, m.cols);
    for (r, o) in out.iter_mut().enumerate() {
        let off = r * rlen;
        *o = row_dot(dt, &bytes[off..off + rlen], m.cols, x, scratch);
    }
}

// ---------------------------------------------------------------------------
// Normalization / positions
// ---------------------------------------------------------------------------

/// In-place RMS normalization. `w` may already have the Gemma `+1` baked in;
/// callers embed the `-1` into the slice they pass.
fn rms_norm(x: &mut [f32], w: &[f32], eps: f64) {
    debug_assert_eq!(x.len(), w.len());
    let n = x.len();
    let mut sum = 0.0f64;
    for v in x.iter() {
        sum += (*v as f64) * (*v as f64);
    }
    let scale = (sum / n as f64 + eps).sqrt().recip();
    for (xv, wv) in x.iter_mut().zip(w) {
        *xv = (*xv as f64 * scale * *wv as f64) as f32;
    }
}

/// Apply rotary position embedding to a `[n_heads * head_dim]` buffer.
///
/// `interleaved` (true) rotates adjacent pairs, matching llama.cpp's
/// `LLAMA_ROPE_TYPE_NORM` applied to rope-permuted rows; `false` uses the
/// NEOX half-split pairing used by `qwen2`/`gemma`.
fn rotary_apply(vec: &mut [f32], hd: usize, interleaved: bool, inv_freq: &[f32], pos: usize) {
    let half = hd / 2;
    let n_heads = vec.len() / hd;
    if interleaved {
        for h in 0..n_heads {
            let base = h * hd;
            for (i, &freq) in inv_freq.iter().take(half).enumerate() {
                let angle = pos as f32 * freq;
                let a = base + 2 * i;
                let b = a + 1;
                let (f0, f1) = (vec[a], vec[b]);
                vec[a] = f0 * angle.cos() - f1 * angle.sin();
                vec[b] = f1 * angle.cos() + f0 * angle.sin();
            }
        }
    } else {
        for h in 0..n_heads {
            let base = h * hd;
            for (i, &freq) in inv_freq.iter().take(half).enumerate() {
                let angle = pos as f32 * freq;
                let a = base + i;
                let d = a + half;
                let (f0, f1) = (vec[a], vec[d]);
                vec[a] = f0 * angle.cos() - f1 * angle.sin();
                vec[d] = f1 * angle.cos() + f0 * angle.sin();
            }
        }
    }
}

fn inv_freq_table(head_dim: usize, rope_theta: f64) -> Vec<f32> {
    (0..head_dim / 2)
        .map(|i| rope_theta.powf(-(2.0 * i as f64) / head_dim as f64) as f32)
        .collect()
}

// ---------------------------------------------------------------------------
// KV cache
// ---------------------------------------------------------------------------

/// Growable per-layer K/V cache. Each layer stores `capacity * cell` f32
/// values (`cell = n_kv_heads * head_dim`); position `pos` occupies
/// `[pos*cell, (pos+1)*cell)`, with `kv_head` and `j` addressed inside.
pub struct GgufKvCache {
    k: Vec<Vec<f32>>,
    v: Vec<Vec<f32>>,
    cell: usize,
    head_dim: usize,
    capacity: usize,
    /// Number of tokens currently cached.
    pub len: usize,
}

impl GgufKvCache {
    pub fn new(cfg: &LlmModelConfig) -> Self {
        let cell = cfg.n_kv_heads * cfg.head_dim;
        let capacity = 64usize;
        let cell_len = capacity * cell;
        let k = (0..cfg.n_layers).map(|_| vec![0.0; cell_len]).collect();
        let v = (0..cfg.n_layers).map(|_| vec![0.0; cell_len]).collect();
        Self {
            k,
            v,
            cell,
            head_dim: cfg.head_dim,
            capacity,
            len: 0,
        }
    }

    fn grow_to(&mut self, needed: usize) {
        if needed <= self.capacity {
            return;
        }
        let mut new_cap = self.capacity * 2;
        while new_cap < needed {
            new_cap *= 2;
        }
        let new_len = new_cap * self.cell;
        for (k, v) in self.k.iter_mut().zip(self.v.iter_mut()) {
            k.resize(new_len, 0.0);
            v.resize(new_len, 0.0);
        }
        self.capacity = new_cap;
    }

    /// Append the K/V vectors of one key-value head at `pos`.
    fn push_head(&mut self, layer: usize, pos: usize, kv_head: usize, k: &[f32], v: &[f32]) {
        self.grow_to(pos + 1);
        let start = pos * self.cell + kv_head * self.head_dim;
        self.k[layer][start..start + self.head_dim].copy_from_slice(k);
        self.v[layer][start..start + self.head_dim].copy_from_slice(v);
    }

    #[inline(always)]
    fn k_at(&self, layer: usize, pos: usize, kv_head: usize, j: usize) -> f32 {
        self.k[layer][pos * self.cell + kv_head * self.head_dim + j]
    }

    #[inline(always)]
    fn v_at(&self, layer: usize, pos: usize, kv_head: usize, j: usize) -> f32 {
        self.v[layer][pos * self.cell + kv_head * self.head_dim + j]
    }
}

// ---------------------------------------------------------------------------
// Per-layer packed state
// ---------------------------------------------------------------------------

struct GgufLayer {
    input_norm: Vec<f32>,
    post_norm: Vec<f32>,
    attn_q: MatRef,
    attn_k: MatRef,
    attn_v: MatRef,
    attn_o: MatRef,
    q_bias: Vec<f32>,
    k_bias: Vec<f32>,
    v_bias: Vec<f32>,
    qk_norm: Option<(Vec<f32>, Vec<f32>)>,
    ffn_gate: MatRef,
    ffn_up: MatRef,
    ffn_down: MatRef,
}

// ---------------------------------------------------------------------------
// Metadata
// ---------------------------------------------------------------------------

fn meta_str<'a>(meta: &'a HashMap<String, MetaValue>, key: &str) -> Option<&'a str> {
    meta.get(key).and_then(MetaValue::as_str)
}

fn meta_count(meta: &HashMap<String, MetaValue>, key: &str) -> Option<usize> {
    meta.get(key)
        .and_then(MetaValue::as_u64)
        .map(|v| v as usize)
}

fn meta_f32(meta: &HashMap<String, MetaValue>, key: &str) -> Option<f64> {
    match meta.get(key) {
        Some(MetaValue::F32(v)) => Some(*v as f64),
        Some(MetaValue::F64(v)) => Some(*v),
        _ => None,
    }
}

/// Parse [`LlmModelConfig`] from GGUF metadata loops (`{arch}.*` keys).
///
/// Returns the config plus whether the file uses NEOX (half-split) RoPE
/// (every arch except `llama`).
fn config_from_gguf(file: &GgufFile) -> Result<(LlmModelConfig, bool)> {
    let meta = &file.metadata;
    let arch = meta_str(meta, "general.architecture")
        .context("file is missing `general.architecture`")?
        .to_string();
    let key = |suffix: &str| format!("{arch}.{suffix}");

    let get = |suffix: &str, alt: &[&str]| -> Option<usize> {
        meta_count(meta, &key(suffix))
            .or_else(|| alt.iter().find_map(|k| meta_count(meta, &key(k))))
    };

    let d_model = get("embedding_length", &["hidden_size", "n_embd"])
        .context("missing `embedding_length`")?;
    let n_layers =
        get("block_count", &["n_layer", "layer_count"]).context("missing `block_count`")?;
    let n_heads = get("attention.head_count", &["n_head"]).context("missing `head_count`")?;
    let n_kv_heads = get("attention.head_count_kv", &["n_head_kv"]).unwrap_or(n_heads);
    let head_dim =
        get("rope.dimension_count", &["attention.head_dim"]).unwrap_or(d_model / n_heads);
    let intermediate_size =
        get("ffn_dim", &["feed_forward_length", "n_ff"]).unwrap_or(4 * n_heads * head_dim);
    let vocab_size = get("vocab_size", &["n_vocab"])
        .or_else(|| {
            file.tensors
                .get("token_embd.weight")
                .and_then(|t| t.shape.last().copied())
        })
        .context("missing `vocab_size`")?;
    let max_seq_len = get("context_length", &["max_position_embeddings", "n_ctx"]).unwrap_or(4096);
    let rope_theta = meta_f32(meta, &key("rope.freq_base")).unwrap_or(10_000.0);
    let rms_eps = meta_f32(meta, &key("attention.layer_norm_rms_epsilon"))
        .or_else(|| meta_f32(meta, &key("attention.norm_epsilon")))
        .unwrap_or(1e-5);
    let sliding_window = get("attention.sliding_window", &[]).filter(|&w| w > 0);

    let hf_model_type = if arch.starts_with("gemma") {
        "gemma".to_string()
    } else if arch == "qwen2" {
        "qwen2".to_string()
    } else {
        "llama".to_string()
    };
    let neox_rope = !matches!(hf_model_type.as_str(), "llama");
    let is_gemma = hf_model_type == "gemma";
    let has_qk_norm = is_gemma && arch != "gemma";

    let cfg = LlmModelConfig {
        d_model,
        n_layers,
        n_heads,
        n_kv_heads,
        head_dim,
        intermediate_size,
        vocab_size,
        max_seq_len,
        rope_theta,
        rms_eps,
        tie_word_embeddings: !file.tensors.contains_key("output.weight"),
        qkv_bias: file.tensors.contains_key("blk.0.attn_q.bias"),
        hf_model_type,
        sliding_window,
        use_gelu: is_gemma,
        has_qk_norm,
    };
    Ok((cfg, neox_rope))
}

// ---------------------------------------------------------------------------
// Engine
// ---------------------------------------------------------------------------

/// The mmap-backed quantized inference engine.
pub struct GgufEngine {
    cfg: Arc<LlmModelConfig>,
    /// Human-readable model name from `general.name`.
    pub name: String,
    layers: Vec<GgufLayer>,
    embed: MatRef,
    output_norm: Vec<f32>,
    output: Option<MatRef>,
    inv_freq: Vec<f32>,
    neox_rope: bool,
}

/// TEMP-DEBUG: per-stage intermediates captured by `GgufEngine::debug_layer_trace`.
#[doc(hidden)]
pub struct DebugTrace {
    pub normed: Vec<f32>,
    pub q: Vec<f32>,
    pub k: Vec<f32>,
    pub v: Vec<f32>,
    pub qk_normed: bool,
    pub attn: Vec<f32>,
    pub o: Vec<f32>,
    pub out: Vec<f32>,
    pub mlp_in: Vec<f32>,
    pub gate_act: Vec<f32>,
    pub up: Vec<f32>,
    pub down: Vec<f32>,
    pub final_out: Vec<f32>,
}

impl GgufEngine {
    /// Load (mmap) a GGUF file and validate it against a config path.
    pub fn load(path: impl AsRef<Path>) -> Result<Self> {
        let file = GgufFile::from_path_mmap(path.as_ref())
            .with_context(|| format!("failed to map `{}`", path.as_ref().display()))?;
        Self::from_file(Arc::new(file))
    }

    /// Build an engine around an already-parsed file (used by round-trip tests).
    pub fn from_file(file: Arc<GgufFile>) -> Result<Self> {
        let (cfg, neox_rope) = config_from_gguf(&file)?;
        let cfg = Arc::new(cfg);
        Self::from_parts(file, cfg, neox_rope)
    }

    fn from_parts(file: Arc<GgufFile>, cfg: Arc<LlmModelConfig>, neox_rope: bool) -> Result<Self> {
        let d = cfg.d_model;
        let q_dim = cfg.n_heads * cfg.head_dim;
        let kv_dim = cfg.n_kv_heads * cfg.head_dim;

        let mat = |name: &str| -> Result<MatRef> {
            let t = file
                .tensors
                .get(name)
                .with_context(|| format!("missing tensor `{name}`"))?;
            if !is_supported(t.dtype) {
                bail!("tensor `{name}` has unsupported dtype {:?}", t.dtype);
            }
            if t.shape.len() != 2 {
                bail!("tensor `{name}` is not 2-D ({:?})", t.shape);
            }
            let cols = t.shape[0];
            let rows = t.shape[1];
            if cols == 0 || rows == 0 {
                bail!("tensor `{name}` has an empty shape");
            }
            if t.dtype == GgmlType::Q4K && !cols_q4k_aligned(cols) {
                bail!(
                    "tensor `{name}` is Q4_K with non-aligned contiguous dim ({cols}); \
                     re-export with F32 fallback"
                );
            }
            Ok(MatRef {
                file: file.clone(),
                name: name.to_string(),
                rows,
                cols,
            })
        };
        let norm = |name: &str| -> Result<Vec<f32>> {
            file.dequant_f32(name)
                .map(|(v, _)| v)
                .with_context(|| format!("failed to load norm `{name}`"))
        };

        let embed = mat("token_embd.weight")?;
        if embed.cols != d {
            bail!("token_embd width {} does not match d_model {d}", embed.cols);
        }
        let output = if cfg.tie_word_embeddings {
            None
        } else {
            let m = mat("output.weight")?;
            if m.cols != d || m.rows != cfg.vocab_size {
                bail!(
                    "output.weight shape ({}, {}) does not match vocab {} / d_model {}",
                    m.cols,
                    m.rows,
                    cfg.vocab_size,
                    d
                );
            }
            Some(m)
        };

        let mut layers = Vec::with_capacity(cfg.n_layers);
        for li in 0..cfg.n_layers {
            let p = |suffix: &str| format!("blk.{li}.{suffix}");
            let expect_dims = |m: &MatRef, ok_rows: usize| -> Result<()> {
                if m.rows != ok_rows {
                    bail!(
                        "blk.{li} matrix `{}` has {} rows, expected {ok_rows}",
                        m.name,
                        m.rows
                    );
                }
                Ok(())
            };
            let attn_q = mat(&p("attn_q.weight"))?;
            expect_dims(&attn_q, q_dim)?;
            let attn_k = mat(&p("attn_k.weight"))?;
            expect_dims(&attn_k, kv_dim)?;
            let attn_v = mat(&p("attn_v.weight"))?;
            expect_dims(&attn_v, kv_dim)?;
            let attn_o = mat(&p("attn_output.weight"))?;
            expect_dims(&attn_o, d)?;
            for (m, label) in [
                (&attn_q, "attn_q"),
                (&attn_k, "attn_k"),
                (&attn_v, "attn_v"),
            ] {
                if m.cols != d {
                    bail!("blk.{li} `{label}` has {} inputs, expected {d}", m.cols);
                }
            }
            if attn_o.cols != q_dim {
                bail!(
                    "blk.{li} attn_output has {} inputs, expected {q_dim}",
                    attn_o.cols
                );
            }

            let bias_of = |name: &str| -> Option<Result<Vec<f32>>> {
                if file.tensors.contains_key(name) {
                    Some(
                        file.dequant_f32(name)
                            .map(|(v, _)| v)
                            .with_context(|| format!("failed to load bias `{name}`")),
                    )
                } else {
                    None
                }
            };
            let q_bias = bias_of(&p("attn_q.bias")).transpose()?.unwrap_or_default();
            let k_bias = bias_of(&p("attn_k.bias")).transpose()?.unwrap_or_default();
            let v_bias = bias_of(&p("attn_v.bias")).transpose()?.unwrap_or_default();
            if !q_bias.is_empty() && q_bias.len() != q_dim {
                bail!("blk.{li} attn_q.bias length {} != {q_dim}", q_bias.len());
            }
            if !k_bias.is_empty() && k_bias.len() != kv_dim {
                bail!("blk.{li} attn_k.bias length {} != {kv_dim}", k_bias.len());
            }
            if !v_bias.is_empty() && v_bias.len() != kv_dim {
                bail!("blk.{li} attn_v.bias length {} != {kv_dim}", v_bias.len());
            }

            let qk_norm = match (
                file.tensors.contains_key(&p("attn_q_norm.weight")),
                file.tensors.contains_key(&p("attn_k_norm.weight")),
            ) {
                (true, true) => Some((
                    norm(&p("attn_q_norm.weight"))?,
                    norm(&p("attn_k_norm.weight"))?,
                )),
                (false, false) => None,
                _ => bail!("blk.{li} has only one of attn_q_norm/attn_k_norm"),
            };

            let ffn_gate = mat(&p("ffn_gate.weight"))?;
            let ffn_up = mat(&p("ffn_up.weight"))?;
            let ffn_down = mat(&p("ffn_down.weight"))?;
            let inter = cfg.intermediate_size;
            for (m, label) in [
                (&ffn_gate, "ffn_gate"),
                (&ffn_up, "ffn_up"),
                (&ffn_down, "ffn_down"),
            ] {
                if (label == "ffn_down" && (m.cols != inter || m.rows != d))
                    || (label != "ffn_down" && (m.cols != d || m.rows != inter))
                {
                    bail!(
                        "blk.{li} `{label}` shape ({}, {}) does not match inter={inter} d={d}",
                        m.cols,
                        m.rows
                    );
                }
            }

            let input_norm = norm(&p("attn_norm.weight"))?;
            let post_norm = norm(&p("ffn_norm.weight"))?;
            if input_norm.len() != d || post_norm.len() != d {
                bail!("blk.{li} norm length != d_model {d}");
            }

            layers.push(GgufLayer {
                input_norm,
                post_norm,
                attn_q,
                attn_k,
                attn_v,
                attn_o,
                q_bias,
                k_bias,
                v_bias,
                qk_norm,
                ffn_gate,
                ffn_up,
                ffn_down,
            });
        }

        let output_norm = norm("output_norm.weight")?;
        if output_norm.len() != d {
            bail!("output_norm length {} != d_model {d}", output_norm.len());
        }

        let name = meta_str(&file.metadata, "general.name")
            .unwrap_or("untitled")
            .to_string();
        let inv_freq = inv_freq_table(cfg.head_dim, cfg.rope_theta);

        Ok(Self {
            cfg,
            name,
            layers,
            embed,
            output_norm,
            output,
            inv_freq,
            neox_rope,
        })
    }

    /// The parsed model configuration.
    pub fn config(&self) -> &LlmModelConfig {
        &self.cfg
    }

    /// A fresh KV cache sized for this engine's configuration.
    pub fn new_cache(&self) -> GgufKvCache {
        GgufKvCache::new(&self.cfg)
    }

    /// Decode `ids` starting at `start_pos`, returning logits for the final
    /// token. K/V vectors are appended to `cache`; `start_pos` should equal
    /// `cache.len()` so positions stay contiguous.
    pub fn forward(
        &self,
        ids: &[u32],
        cache: &mut GgufKvCache,
        start_pos: usize,
    ) -> Result<Vec<f32>> {
        let d = self.cfg.d_model;
        let mut x = vec![0.0; d];
        let mut scratch = [0.0f32; QK_K];
        for (i, &tok) in ids.iter().enumerate() {
            self.embed_row(tok, &mut x)?;
            let pos = start_pos + i;
            for (li, layer) in self.layers.iter().enumerate() {
                self.decode_token(li, layer, cache, &mut x, pos, &mut scratch)?;
            }
            rms_norm(&mut x, &self.output_norm, self.cfg.rms_eps);
        }
        cache.len = start_pos + ids.len();
        self.project(&x, &mut scratch)
    }

    /// TEMP-DEBUG: residual hidden after `num_layers` layers for the final token.
    #[doc(hidden)]
    pub fn debug_hidden_after(
        &self,
        ids: &[u32],
        cache: &mut GgufKvCache,
        start_pos: usize,
        num_layers: usize,
    ) -> Result<Vec<f32>> {
        let d = self.cfg.d_model;
        let mut x = vec![0.0; d];
        let mut scratch = [0.0f32; QK_K];
        for (i, &tok) in ids.iter().enumerate() {
            self.embed_row(tok, &mut x)?;
            let pos = start_pos + i;
            for (li, layer) in self.layers.iter().enumerate() {
                if li == num_layers {
                    return Ok(x);
                }
                self.decode_token(li, layer, cache, &mut x, pos, &mut scratch)?;
            }
        }
        Ok(x)
    }

    /// TEMP-DEBUG: embed row for the last token.
    #[doc(hidden)]
    pub fn debug_embed_last(&self, ids: &[u32], last: usize) -> Result<Vec<f32>> {
        let mut x = vec![0.0; self.cfg.d_model];
        self.embed_row(ids[last], &mut x)?;
        Ok(x)
    }

    /// TEMP-DEBUG: per-stage intermediates of one token through one layer.
    #[doc(hidden)]
    pub fn debug_layer_trace(
        &self,
        x_in: &[f32],
        li: usize,
        cache: &mut GgufKvCache,
        pos: usize,
        scratch: &mut [f32; QK_K],
    ) -> Result<DebugTrace> {
        let cfg = &self.cfg;
        let d = cfg.d_model;
        let n_heads = cfg.n_heads;
        let n_kv = cfg.n_kv_heads;
        let hd = cfg.head_dim;
        let q_dim = n_heads * hd;
        let kv_dim = n_kv * hd;
        let layer = &self.layers[li];

        let mut x = x_in.to_vec();
        rms_norm(&mut x, &layer.input_norm, cfg.rms_eps);
        let normed = x.clone();

        let mut q = vec![0.0; q_dim];
        let mut k = vec![0.0; kv_dim];
        let mut v = vec![0.0; kv_dim];
        gemv(&layer.attn_q, &x, &mut q, scratch);
        gemv(&layer.attn_k, &x, &mut k, scratch);
        gemv(&layer.attn_v, &x, &mut v, scratch);
        if !layer.q_bias.is_empty() {
            for (o, b) in q.iter_mut().zip(&layer.q_bias) {
                *o += b;
            }
        }
        if !layer.k_bias.is_empty() {
            for (o, b) in k.iter_mut().zip(&layer.k_bias) {
                *o += b;
            }
        }
        if !layer.v_bias.is_empty() {
            for (o, b) in v.iter_mut().zip(&layer.v_bias) {
                *o += b;
            }
        }
        let q_pre_rope = q.clone();
        let k_pre_rope = k.clone();

        let qk_normed = !layer.qk_norm.is_none();
        if let Some((qw, kw)) = &layer.qk_norm {
            for h in 0..n_heads {
                rms_norm(&mut q[h * hd..(h + 1) * hd], qw, cfg.rms_eps);
            }
            for h in 0..n_kv {
                rms_norm(&mut k[h * hd..(h + 1) * hd], kw, cfg.rms_eps);
            }
        }

        let interleaved = !self.neox_rope;
        rotary_apply(&mut q, hd, interleaved, &self.inv_freq, pos);
        rotary_apply(&mut k, hd, interleaved, &self.inv_freq, pos);

        for hkv in 0..n_kv {
            cache.push_head(
                li,
                pos,
                hkv,
                &k[hkv * hd..(hkv + 1) * hd],
                &v[hkv * hd..(hkv + 1) * hd],
            );
        }

        let groups = n_heads / n_kv;
        let win_start = match cfg.sliding_window {
            Some(w) if pos + 1 > w => pos + 1 - w,
            _ => 0,
        };
        let scale = (hd as f64).sqrt().recip() as f32;

        let mut attn = vec![0.0; q_dim];
        for hq in 0..n_heads {
            let group = hq / groups;
            let q_head = &q[hq * hd..(hq + 1) * hd];
            let mut scores = Vec::with_capacity(pos + 1 - win_start);
            let mut max = f32::NEG_INFINITY;
            for t in win_start..=pos {
                let mut s = 0.0f32;
                for (j, &qv) in q_head.iter().enumerate() {
                    s += qv * cache.k_at(li, t, group, j);
                }
                let score = s * scale;
                max = max.max(score);
                scores.push(score);
            }
            let mut sum = 0.0f64;
            for sc in scores.iter_mut() {
                *sc = (*sc - max).exp();
                sum += *sc as f64;
            }
            let inv = (1.0 / sum) as f32;
            let out_head = &mut attn[hq * hd..(hq + 1) * hd];
            out_head.fill(0.0);
            for (i, sc) in scores.iter().enumerate() {
                let t = win_start + i;
                let w = *sc * inv;
                for (j, o) in out_head.iter_mut().enumerate() {
                    *o += w * cache.v_at(li, t, group, j);
                }
            }
        }

        let mut o = vec![0.0; d];
        gemv(&layer.attn_o, &attn, &mut o, scratch);
        let mut out = x_in.to_vec();
        for i in 0..d {
            out[i] += o[i];
        }
        let attn_out = out.clone();

        let inter = cfg.intermediate_size;
        let mut mlp_in = out.clone();
        rms_norm(&mut mlp_in, &layer.post_norm, cfg.rms_eps);
        let mut gate = vec![0.0; inter];
        let mut up = vec![0.0; inter];
        gemv(&layer.ffn_gate, &mlp_in, &mut gate, scratch);
        gemv(&layer.ffn_up, &mlp_in, &mut up, scratch);
        for i in 0..inter {
            let g = gate[i];
            gate[i] = if cfg.use_gelu {
                let inner = (2_f32 / std::f32::consts::PI).sqrt() * (g + 0.044715 * g * g * g);
                0.5 * g * (1.0 + inner.tanh())
            } else {
                g / (1.0 + (-g).exp())
            };
            up[i] *= gate[i];
        }
        let gate_act = gate.clone();
        let mut down = vec![0.0; d];
        gemv(&layer.ffn_down, &up, &mut down, scratch);
        let mut final_out = attn_out.clone();
        for i in 0..d {
            final_out[i] += down[i];
        }
        Ok(DebugTrace {
            normed,
            q: q_pre_rope,
            k: k_pre_rope,
            v,
            qk_normed,
            attn,
            o,
            out,
            mlp_in,
            gate_act,
            up,
            down,
            final_out,
        })
    }

    fn embed_row(&self, tok: u32, out: &mut [f32]) -> Result<()> {
        let row = tok as usize;
        if row >= self.embed.rows {
            bail!("token id {row} out of range for token_embd");
        }
        let dt = self.embed.dtype();
        let bytes = self.embed.bytes()?;
        let cols = self.embed.cols;
        let mut scratch = [0.0f32; QK_K];
        if is_k_quant(dt) && cols_q4k_aligned(cols) {
            let bsz = k_block_bytes(dt);
            let rlen = cols / QK_K * bsz;
            let base = row * rlen;
            for b in 0..cols / QK_K {
                dequant_k_block(
                    dt,
                    &bytes[base + b * bsz..base + (b + 1) * bsz],
                    &mut scratch,
                );
                out[b * QK_K..(b + 1) * QK_K].copy_from_slice(&scratch);
            }
        } else {
            let rlen = row_bytes(dt, cols);
            let base = row * rlen;
            for (j, o) in out.iter_mut().enumerate() {
                *o = match dt {
                    GgmlType::F32 => f32::from_le_bytes(
                        bytes[base + j * 4..base + j * 4 + 4].try_into().unwrap(),
                    ),
                    GgmlType::F16 => f16_to_f32(u16::from_le_bytes([
                        bytes[base + j * 2],
                        bytes[base + j * 2 + 1],
                    ])),
                    GgmlType::BF16 => bf16_to_f32(u16::from_le_bytes([
                        bytes[base + j * 2],
                        bytes[base + j * 2 + 1],
                    ])),
                    _ => {
                        bail!(
                            "token_embd row must be F32/F16/BF16 or K-quant aligned; \
                             got {:?} with unaligned width {cols}",
                            dt
                        )
                    }
                };
            }
        }
        Ok(())
    }

    fn decode_token(
        &self,
        li: usize,
        layer: &GgufLayer,
        cache: &mut GgufKvCache,
        x: &mut [f32],
        pos: usize,
        scratch: &mut [f32; QK_K],
    ) -> Result<()> {
        let cfg = &self.cfg;
        let d = cfg.d_model;
        let n_heads = cfg.n_heads;
        let n_kv = cfg.n_kv_heads;
        let hd = cfg.head_dim;
        let q_dim = n_heads * hd;
        let kv_dim = n_kv * hd;
        let inter = cfg.intermediate_size;

        let input = x.to_vec();
        rms_norm(x, &layer.input_norm, cfg.rms_eps);

        let mut q = vec![0.0; q_dim];
        let mut k = vec![0.0; kv_dim];
        let mut v = vec![0.0; kv_dim];
        gemv(&layer.attn_q, x, &mut q, scratch);
        gemv(&layer.attn_k, x, &mut k, scratch);
        gemv(&layer.attn_v, x, &mut v, scratch);
        if !layer.q_bias.is_empty() {
            for (o, b) in q.iter_mut().zip(&layer.q_bias) {
                *o += b;
            }
        }
        if !layer.k_bias.is_empty() {
            for (o, b) in k.iter_mut().zip(&layer.k_bias) {
                *o += b;
            }
        }
        if !layer.v_bias.is_empty() {
            for (o, b) in v.iter_mut().zip(&layer.v_bias) {
                *o += b;
            }
        }

        if let Some((qw, kw)) = &layer.qk_norm {
            for h in 0..n_heads {
                rms_norm(&mut q[h * hd..(h + 1) * hd], qw, cfg.rms_eps);
            }
            for h in 0..n_kv {
                rms_norm(&mut k[h * hd..(h + 1) * hd], kw, cfg.rms_eps);
            }
        }

        let interleaved = !self.neox_rope;
        rotary_apply(&mut q, hd, interleaved, &self.inv_freq, pos);
        rotary_apply(&mut k, hd, interleaved, &self.inv_freq, pos);

        for hkv in 0..n_kv {
            let k_head = &k[hkv * hd..(hkv + 1) * hd];
            let v_head = &v[hkv * hd..(hkv + 1) * hd];
            cache.push_head(li, pos, hkv, k_head, v_head);
        }

        let groups = n_heads / n_kv;
        let win_start = match cfg.sliding_window {
            Some(w) if pos + 1 > w => pos + 1 - w,
            _ => 0,
        };
        let scale = (hd as f64).sqrt().recip() as f32;

        let mut attn = vec![0.0; q_dim];
        for hq in 0..n_heads {
            let group = hq / groups;
            let q_head = &q[hq * hd..(hq + 1) * hd];
            let mut scores = Vec::with_capacity(pos + 1 - win_start);
            let mut max = f32::NEG_INFINITY;
            for t in win_start..=pos {
                let mut s = 0.0f32;
                for (j, &qv) in q_head.iter().enumerate() {
                    s += qv * cache.k_at(li, t, group, j);
                }
                let score = s * scale;
                max = max.max(score);
                scores.push(score);
            }
            let mut sum = 0.0f64;
            for sc in scores.iter_mut() {
                *sc = (*sc - max).exp();
                sum += *sc as f64;
            }
            let inv = (1.0 / sum) as f32;
            let out_head = &mut attn[hq * hd..(hq + 1) * hd];
            out_head.fill(0.0);
            for (i, sc) in scores.iter().enumerate() {
                let t = win_start + i;
                let w = *sc * inv;
                for (j, o) in out_head.iter_mut().enumerate() {
                    *o += w * cache.v_at(li, t, group, j);
                }
            }
        }

        let mut o = vec![0.0; d];
        gemv(&layer.attn_o, &attn, &mut o, scratch);
        for i in 0..d {
            x[i] = input[i] + o[i];
        }

        let mut mlp_in = x.to_vec();
        rms_norm(&mut mlp_in, &layer.post_norm, cfg.rms_eps);
        let mut gate = vec![0.0; inter];
        let mut up = vec![0.0; inter];
        gemv(&layer.ffn_gate, &mlp_in, &mut gate, scratch);
        gemv(&layer.ffn_up, &mlp_in, &mut up, scratch);
        for i in 0..inter {
            let g = gate[i];
            gate[i] = if cfg.use_gelu {
                let inner = (2_f32 / std::f32::consts::PI).sqrt() * (g + 0.044715 * g * g * g);
                0.5 * g * (1.0 + inner.tanh())
            } else {
                g / (1.0 + (-g).exp())
            };
            up[i] *= gate[i];
        }
        let mut down = vec![0.0; d];
        gemv(&layer.ffn_down, &up, &mut down, scratch);
        for i in 0..d {
            x[i] += down[i];
        }
        Ok(())
    }

    fn project(&self, x: &[f32], scratch: &mut [f32; QK_K]) -> Result<Vec<f32>> {
        let vocab = self.cfg.vocab_size;
        let mut logits = vec![0.0; vocab];
        match &self.output {
            Some(m) => {
                if m.rows != vocab {
                    bail!("output.weight rows {} != vocab {vocab}", m.rows);
                }
                gemv(m, x, &mut logits, scratch);
            }
            None => gemv(&self.embed, x, &mut logits, scratch),
        }
        Ok(logits)
    }

    /// Autoregressive text generation, mirroring `crate::generate::generate`
    /// but on the CPU engine. Stops at `eos` or `max_seq_len`.
    pub fn generate(
        &self,
        tokenizer: &TokenizerStore,
        prompt: &str,
        config: &GenerateConfig,
    ) -> Result<String> {
        let mut ids = tokenizer.encode_raw(prompt)?;
        if ids.is_empty() {
            bail!("prompt produced zero tokens");
        }
        if ids.len() > self.cfg.max_seq_len {
            bail!(
                "prompt is {} tokens, longer than the model's context ({})",
                ids.len(),
                self.cfg.max_seq_len
            );
        }

        let mut cache = self.new_cache();
        let mut last = self.forward(&ids, &mut cache, 0)?;
        let mut next = sample_next_token_cpu(&last, config);
        let mut generated = Vec::new();
        while generated.len() < config.max_tokens {
            if next == tokenizer.eos_id {
                break;
            }
            generated.push(next);
            ids.push(next);
            if ids.len() >= self.cfg.max_seq_len {
                log::warn!(
                    "reached max_seq_len ({}); stopping generation",
                    self.cfg.max_seq_len
                );
                break;
            }
            last = self.forward(&[next], &mut cache, ids.len() - 1)?;
            next = sample_next_token_cpu(&last, config);
        }
        tokenizer.decode(&generated, true)
    }
}

#[cfg(test)]
mod tests {
    use std::sync::Arc;

    use burn::tensor::{Int, Tensor, TensorData};
    use rlx_gguf::GgufFile;

    use super::*;
    use crate::data::TokenizerStore;
    use crate::export::export_gguf;
    use crate::model::LlmModel;
    use crate::train::TestBackend;

    fn test_config(arch: &str) -> LlmModelConfig {
        LlmModelConfig {
            d_model: 256,
            n_layers: 2,
            n_heads: 4,
            n_kv_heads: 2,
            head_dim: 64,
            intermediate_size: 1024,
            vocab_size: 256,
            max_seq_len: 64,
            rope_theta: 10_000.0,
            rms_eps: 1e-6,
            tie_word_embeddings: false,
            qkv_bias: false,
            hf_model_type: arch.to_string(),
            sliding_window: None,
            use_gelu: arch.starts_with("gemma"),
            has_qk_norm: false,
        }
    }

    fn test_tokenizer() -> TokenizerStore {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("tokenizer.json");
        std::fs::write(
            &path,
            r#"{
                "version": "1.0",
                "model": {
                    "type": "WordLevel",
                    "vocab": {"[UNK]": 0, "hello": 1, "world": 2, "burn": 3, "</s>": 4,
                              "red": 5, "green": 6, "blue": 7, "sky": 8, "sea": 9},
                    "unk_token": "[UNK]"
                },
                "normalizer": null,
                "pre_tokenizer": {"type": "Whitespace"},
                "post_processor": null,
                "decoder": null,
                "added_tokens": []
            }"#,
        )
        .unwrap();
        TokenizerStore::from_file(&path).unwrap()
    }

    fn argmax(logits: &[f32]) -> u32 {
        logits
            .iter()
            .enumerate()
            .max_by(|a, b| a.1.partial_cmp(b.1).unwrap())
            .map(|(i, _)| i as u32)
            .unwrap()
    }

    /// Rebuild a burn model whose weights are exactly the *dequantized* GGUF
    /// tensors (reverse of `export_gguf`), so a differential test compares the
    /// engine's arithmetic against a burn reference with identical weights —
    /// isolating computation from Q4_K quantization error.
    fn burn_ref_from_gguf(config: &LlmModelConfig, file: &GgufFile) -> LlmModel<TestBackend> {
        use crate::model::attention::LayerKv;
        use burn::module::Param;
        let device = burn::backend::flex::FlexDevice;
        let mut model = LlmModel::<TestBackend>::new_zeroed(config, &device);
        let arch = config.gguf_architecture();
        let is_gemma = arch == "gemma";

        let deq = |name: &str| -> (Vec<f32>, Vec<usize>) { file.dequant_f32(name).unwrap() };
        let g2b = |v: Vec<f32>| -> Vec<f32> {
            if is_gemma {
                v.into_iter().map(|x| x - 1.0).collect()
            } else {
                v
            }
        };
        let transpose = |flat: &[f32], rows: usize, cols: usize| -> Vec<f32> {
            let mut out = vec![0.0; flat.len()];
            for r in 0..rows {
                for c in 0..cols {
                    out[c * rows + r] = flat[r * cols + c];
                }
            }
            out
        };
        let linear_weight = |name: &str, out: usize, inn: usize| -> Vec<f32> {
            let (flat, shape) = deq(name);
            // stored is PyTorch `[out, in]` row-major (shape[0] = in = cols).
            let rows = out;
            let cols = inn;
            assert_eq!(shape, vec![inn, out], "{name} shape");
            transpose(&flat, rows, cols)
        };

        let (emb, _) = deq("token_embd.weight");
        model.model.embed_tokens.weight = Param::from_data(
            TensorData::new(emb, [config.vocab_size, config.d_model]),
            &device,
        );
        let on = g2b(deq("output_norm.weight").0);
        model.model.norm.weight = Param::from_data(TensorData::new(on, [config.d_model]), &device);
        if let Some(head) = &mut model.lm_head {
            let flat = linear_weight("output.weight", config.vocab_size, config.d_model);
            head.weight = Param::from_data(
                TensorData::new(flat, [config.d_model, config.vocab_size]),
                &device,
            );
        }

        // Un-permute rope-permuted q/k rows back to HF half-split layout.
        let unpermute = |flat: &[f32], out_dim: usize, inn: usize, n_group: usize| -> Vec<f32> {
            let head_dim = out_dim / n_group;
            let half = head_dim / 2;
            let mut out = flat.to_vec();
            for g in 0..n_group {
                for i in 0..half {
                    for c in 0..inn {
                        out[(g * head_dim + i) * inn + c] = flat[(g * head_dim + 2 * i) * inn + c];
                        out[(g * head_dim + half + i) * inn + c] =
                            flat[(g * head_dim + 2 * i + 1) * inn + c];
                    }
                }
            }
            out
        };
        let qk_weight = |name: &str, out_dim: usize, n_group: usize| -> Vec<f32> {
            let (flat, shape) = deq(name);
            assert_eq!(shape, vec![config.d_model, out_dim], "{name} shape");
            let cols = config.d_model;
            let un = if arch == "llama" {
                unpermute(&flat, out_dim, cols, n_group)
            } else {
                flat
            };
            transpose(&un, out_dim, cols)
        };

        for l in 0..config.n_layers {
            let a = &mut model.model.layers[l].self_attn;
            a.q_proj.weight = Param::from_data(
                TensorData::new(
                    qk_weight(
                        &format!("blk.{l}.attn_q.weight"),
                        config.n_heads * config.head_dim,
                        config.n_heads,
                    ),
                    [config.d_model, config.n_heads * config.head_dim],
                ),
                &device,
            );
            a.k_proj.weight = Param::from_data(
                TensorData::new(
                    qk_weight(
                        &format!("blk.{l}.attn_k.weight"),
                        config.n_kv_heads * config.head_dim,
                        config.n_kv_heads,
                    ),
                    [config.d_model, config.n_kv_heads * config.head_dim],
                ),
                &device,
            );
            a.v_proj.weight = Param::from_data(
                TensorData::new(
                    linear_weight(
                        &format!("blk.{l}.attn_v.weight"),
                        config.n_kv_heads * config.head_dim,
                        config.d_model,
                    ),
                    [config.d_model, config.n_kv_heads * config.head_dim],
                ),
                &device,
            );
            a.o_proj.weight = Param::from_data(
                TensorData::new(
                    linear_weight(
                        &format!("blk.{l}.attn_output.weight"),
                        config.d_model,
                        config.n_heads * config.head_dim,
                    ),
                    [config.n_heads * config.head_dim, config.d_model],
                ),
                &device,
            );

            let in_norm = g2b(deq(&format!("blk.{l}.attn_norm.weight")).0);
            model.model.layers[l].input_layernorm.weight =
                Param::from_data(TensorData::new(in_norm, [config.d_model]), &device);
            let po_norm = g2b(deq(&format!("blk.{l}.ffn_norm.weight")).0);
            model.model.layers[l].post_attention_layernorm.weight =
                Param::from_data(TensorData::new(po_norm, [config.d_model]), &device);

            let mlp = &mut model.model.layers[l].mlp;
            mlp.gate_proj.weight = Param::from_data(
                TensorData::new(
                    linear_weight(
                        &format!("blk.{l}.ffn_gate.weight"),
                        config.intermediate_size,
                        config.d_model,
                    ),
                    [config.d_model, config.intermediate_size],
                ),
                &device,
            );
            mlp.up_proj.weight = Param::from_data(
                TensorData::new(
                    linear_weight(
                        &format!("blk.{l}.ffn_up.weight"),
                        config.intermediate_size,
                        config.d_model,
                    ),
                    [config.d_model, config.intermediate_size],
                ),
                &device,
            );
            mlp.down_proj.weight = Param::from_data(
                TensorData::new(
                    linear_weight(
                        &format!("blk.{l}.ffn_down.weight"),
                        config.d_model,
                        config.intermediate_size,
                    ),
                    [config.intermediate_size, config.d_model],
                ),
                &device,
            );
        }
        let _ = LayerKv::<TestBackend>::new();
        model
    }

    /// Greedy-generate 8 tokens with the burn model (full re-prefill every
    /// step) and the GGUF engine (incremental KV cache), asserting identical
    /// tokens — proof that weights, RoPE pairing, norms and the quantized
    /// GEMV agree with the reference implementation.
    fn assert_matches_burn(model: &LlmModel<TestBackend>, engine: &GgufEngine, prompt: &[u32]) {
        let device = burn::backend::flex::FlexDevice;
        let vocab = engine.config().vocab_size;
        let mut cache = GgufKvCache::new(engine.config());
        let mut last = engine.forward(prompt, &mut cache, 0).unwrap();
        let mut ref_ids = prompt.to_vec();
        let mut worst: f32 = 0.0;

        for _ in 0..8 {
            let t = Tensor::<TestBackend, 2, Int>::from_data(
                TensorData::new(ref_ids.clone(), [1, ref_ids.len()]),
                &device,
            );
            // The burn side must recompute the full sequence so it can never
            // inherit KV state from the engine's cache.
            let logits = model.forward(t);
            let seq = ref_ids.len();
            let last_ref = logits
                .slice([0..1, (seq - 1)..seq, 0..vocab])
                .reshape([1, vocab])
                .into_data()
                .to_vec::<f32>()
                .unwrap();
            let burn_tok = argmax(&last_ref);
            let cpu_tok = argmax(&last);
            let dev: f32 = last_ref
                .iter()
                .zip(&last)
                .map(|(a, b)| (a - b).abs())
                .fold(0.0f32, |m, v| m.max(v));
            worst = worst.max(dev);
            if cpu_tok != burn_tok {
                let margin = last_ref[burn_tok as usize] - last_ref[cpu_tok as usize];
                let mut r: Vec<usize> = (0..vocab).collect();
                r.sort_unstable_by(|&a, &b| last_ref[b].partial_cmp(&last_ref[a]).unwrap());
                let mut g: Vec<usize> = (0..vocab).collect();
                g.sort_unstable_by(|&a, &b| last[b].partial_cmp(&last[a]).unwrap());
                let ref_top = r[..5]
                    .iter()
                    .map(|&i| (i, format!("{:.4}", last_ref[i])))
                    .collect::<Vec<_>>();
                let cpu_top = g[..5]
                    .iter()
                    .map(|&i| (i, format!("{:.4}", last[i])))
                    .collect::<Vec<_>>();
                eprintln!(
                    "tight vote step {} (dev {dev:.5}, margin {margin:.5}): REF {ref_top:?}",
                    ref_ids.len() - prompt.len()
                );
                eprintln!("  CPU {cpu_top:?}");
            }
            ref_ids.push(burn_tok);
            let pos = ref_ids.len() - 1;
            // Drive BOTH models from the reference stream so each step compares
            // identical contexts; a flipped tight vote must not cascade.
            last = engine.forward(&[burn_tok], &mut cache, pos).unwrap();
        }
        // q8 activation noise keeps logits within ~0.07 on these tiny weights; a
        // real kernel/layout bug shifts them by >=0.4 (see the failed Q4K
        // multiblock discovery: up to 0.62, and 0.54 on a single layer). Use a
        // bound well above the noise floor but far below bug-scale drift.
        assert!(
            worst < 2e-1,
            "logits drifted up to {worst:.5} vs burn reference (prompt {:?})",
            prompt
        );
    }

    fn roundtrip(arch: &str) {
        let config = test_config(arch);
        let device = burn::backend::flex::FlexDevice;
        let model = LlmModel::<TestBackend>::new(&config, &device);
        let tokenizer = test_tokenizer();

        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("model.gguf");
        export_gguf(&model, &config, &tokenizer, &path, &format!("tiny-{arch}")).unwrap();
        assert!(path.exists());

        let file = Arc::new(GgufFile::from_path(&path).unwrap());
        let engine = GgufEngine::from_file(file.clone()).unwrap();
        assert_eq!(engine.name, format!("tiny-{arch}"));
        assert_eq!(engine.config().vocab_size, 256);

        // Differential reference: burn running the dequantized GGUF weights.
        let burn_ref = burn_ref_from_gguf(engine.config(), &file);
        let prompt = [5, 17, 42, 200];
        assert_matches_burn(&burn_ref, &engine, &prompt);
        // A different starting position must also agree (checks RoPE).
        let prompt2 = [1, 2, 3, 19, 30];
        assert_matches_burn(&burn_ref, &engine, &prompt2);
    }

    #[test]
    fn roundtrip_matches_burn_llama_arch() {
        roundtrip("llama");
    }

    #[test]
    fn roundtrip_matches_burn_gemma_arch() {
        roundtrip("gemma");
    }

    #[test]
    fn forward_rejects_oob_token() {
        let config = test_config("llama");
        let device = burn::backend::flex::FlexDevice;
        let model = LlmModel::<TestBackend>::new(&config, &device);
        let tokenizer = test_tokenizer();
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("model.gguf");
        export_gguf(&model, &config, &tokenizer, &path, "tiny").unwrap();

        let file = Arc::new(GgufFile::from_path(&path).unwrap());
        let engine = GgufEngine::from_file(file).unwrap();
        let mut cache = GgufKvCache::new(engine.config());
        assert!(engine.forward(&[999], &mut cache, 0).is_err());
    }
}
