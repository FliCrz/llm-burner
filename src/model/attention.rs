use super::rms_norm::RmsNorm;
use super::rope;

use burn::module::{Initializer, Module};
use burn::nn::{Linear, LinearConfig};
use burn::tensor::activation::softmax;
use burn::tensor::{Bool, Int, Tensor, backend::Backend};

/// Per-layer cached key and value tensors for autoregressive generation.
#[derive(Debug, Clone, Default)]
pub struct LayerKv<B: Backend> {
    /// Cached key tensor, shape `[batch, n_kv_heads, past_seq, head_dim]`.
    pub k: Option<Tensor<B, 4>>,
    /// Cached value tensor, shape `[batch, n_kv_heads, past_seq, head_dim]`.
    pub v: Option<Tensor<B, 4>>,
}

impl<B: Backend> LayerKv<B> {
    /// Create an empty layer cache.
    pub fn new() -> Self {
        Self { k: None, v: None }
    }

    /// Number of tokens currently stored in the cache.
    pub fn len(&self) -> usize {
        self.k.as_ref().map(|t| t.dims()[2]).unwrap_or(0)
    }

    /// True when the cache contains no tokens.
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }
}

/// Grouped-query attention with rotary positional embeddings, optional query/key
/// normalization (Gemma 2/3) and optional sliding-window masking.
///
/// The module field names (`q_proj`, `k_proj`, `v_proj`, `o_proj`, `q_norm`,
/// `k_norm`) match the Hugging Face checkpoint naming so safetensors weights
/// load 1:1.
#[derive(Module, Debug)]
pub struct CausalAttention<B: Backend> {
    /// Query projection, `hidden -> n_heads * head_dim`.
    pub q_proj: Linear<B>,
    /// Key projection, `hidden -> n_kv_heads * head_dim`.
    pub k_proj: Linear<B>,
    /// Value projection, `hidden -> n_kv_heads * head_dim`.
    pub v_proj: Linear<B>,
    /// Output projection, `n_heads * head_dim -> hidden`.
    pub o_proj: Linear<B>,
    /// Optional per-head query norm (Gemma 2/3).
    pub q_norm: Option<RmsNorm<B>>,
    /// Optional per-head key norm (Gemma 2/3).
    pub k_norm: Option<RmsNorm<B>>,

    /// Number of query heads.
    pub n_heads: usize,
    /// Number of key/value heads used by grouped-query attention.
    pub n_kv_heads: usize,
    /// Dimension of every attention head.
    pub head_dim: usize,
    /// Base for the rotary position embeddings.
    pub rope_theta: f64,
    /// Sliding-window size for the first `n_layers - sliding_window_layers`
    /// layers, if applicable (set per layer during construction).
    pub sliding_window: Option<usize>,
}

impl<B: Backend> CausalAttention<B> {
    /// Create a new attention block.
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        hidden_size: usize,
        n_heads: usize,
        n_kv_heads: usize,
        head_dim: usize,
        rope_theta: f64,
        sliding_window: Option<usize>,
        has_qk_norm: bool,
        qkv_bias: bool,
        rms_eps: f64,
        device: &B::Device,
    ) -> Self {
        Self::new_with_initializer(
            hidden_size,
            n_heads,
            n_kv_heads,
            head_dim,
            rope_theta,
            sliding_window,
            has_qk_norm,
            qkv_bias,
            rms_eps,
            Initializer::Normal {
                mean: 0.0,
                std: 0.02,
            },
            device,
        )
    }

    /// Create an attention block using an explicit initializer (zeros when only
    /// a checkpoint shell is needed, see [`super::model::Transformer::new_zeroed`]).
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn new_with_initializer(
        hidden_size: usize,
        n_heads: usize,
        n_kv_heads: usize,
        head_dim: usize,
        rope_theta: f64,
        sliding_window: Option<usize>,
        has_qk_norm: bool,
        qkv_bias: bool,
        rms_eps: f64,
        initializer: Initializer,
        device: &B::Device,
    ) -> Self {
        // Only the Q/K/V projections take a bias (and only for some
        // architectures, e.g. Qwen2); the output projection never does.
        // Biases are zero-initialized, matching Hugging Face.
        let mut q_proj = LinearConfig::new(hidden_size, n_heads * head_dim)
            .with_bias(qkv_bias)
            .with_initializer(initializer.clone())
            .init(device);
        let mut k_proj = LinearConfig::new(hidden_size, n_kv_heads * head_dim)
            .with_bias(qkv_bias)
            .with_initializer(initializer.clone())
            .init(device);
        let mut v_proj = LinearConfig::new(hidden_size, n_kv_heads * head_dim)
            .with_bias(qkv_bias)
            .with_initializer(initializer.clone())
            .init(device);
        if qkv_bias {
            for proj in [&mut q_proj, &mut k_proj, &mut v_proj] {
                let out_dim = proj.bias.as_ref().unwrap().val().dims()[0];
                let zeros = Tensor::<B, 1>::zeros([out_dim], device);
                if let Some(p) = proj.bias.take() {
                    proj.bias = Some(p.map(|_| zeros));
                }
            }
        }
        let o_proj = LinearConfig::new(n_heads * head_dim, hidden_size)
            .with_bias(false)
            .with_initializer(initializer)
            .init(device);

        let q_norm = if has_qk_norm {
            Some(RmsNorm::new(head_dim, rms_eps, device))
        } else {
            None
        };
        let k_norm = if has_qk_norm {
            Some(RmsNorm::new(head_dim, rms_eps, device))
        } else {
            None
        };

        Self {
            q_proj,
            k_proj,
            v_proj,
            o_proj,
            q_norm,
            k_norm,
            n_heads,
            n_kv_heads,
            head_dim,
            rope_theta,
            sliding_window,
        }
    }

    /// Forward pass over a hidden state of shape `[batch, seq, hidden]`.
    pub fn forward(&self, hidden: Tensor<B, 3>) -> Tensor<B, 3> {
        self.forward_with_cache(hidden, None, 0)
    }

    /// Forward pass supporting an optional per-layer key/value cache.
    ///
    /// When `cache` is provided:
    /// - For prefill (`cache` empty, `start_pos == 0`, `seq >= 1`), attention
    ///   runs with causal masking and stores the full key/value sequence.
    /// - For decode (`cache` warm, `seq == 1`), the new key/value is appended
    ///   to the cache and the query attends to all `past + 1` keys without
    ///   masking (all cached tokens precede `start_pos`).
    pub fn forward_with_cache(
        &self,
        hidden: Tensor<B, 3>,
        cache: Option<&mut LayerKv<B>>,
        start_pos: usize,
    ) -> Tensor<B, 3> {
        let [batch, seq, _hidden] = hidden.dims();
        let device = &hidden.device();

        let q = self.q_proj.forward(hidden.clone());
        let k = self.k_proj.forward(hidden.clone());
        let v = self.v_proj.forward(hidden);

        let q = q
            .reshape([batch, seq, self.n_heads, self.head_dim])
            .permute([0, 2, 1, 3]);
        let k = k
            .reshape([batch, seq, self.n_kv_heads, self.head_dim])
            .permute([0, 2, 1, 3]);
        let v = v
            .reshape([batch, seq, self.n_kv_heads, self.head_dim])
            .permute([0, 2, 1, 3]);

        let q = match &self.q_norm {
            Some(norm) => norm.forward(q),
            None => q,
        };
        let k = match &self.k_norm {
            Some(norm) => norm.forward(k),
            None => k,
        };

        let (cos, sin) =
            rope::rope_cos_sin_offset(start_pos, seq, self.head_dim, self.rope_theta, device);
        let cos = cos.unsqueeze::<4>();
        let sin = sin.unsqueeze::<4>();

        let q = rope::rope_apply(q, cos.clone(), sin.clone());
        let k = rope::rope_apply(k, cos, sin);

        // Update / retrieve cached keys and values.
        let (k_all, v_all, apply_causal_mask) = match cache {
            Some(layer_kv) => {
                let (k_merged, v_merged) = match (layer_kv.k.take(), layer_kv.v.take()) {
                    (Some(cached_k), Some(cached_v)) => {
                        let k_cat = Tensor::cat(vec![cached_k, k], 2);
                        let v_cat = Tensor::cat(vec![cached_v, v], 2);
                        (k_cat, v_cat)
                    }
                    _ => (k, v),
                };
                layer_kv.k = Some(k_merged.clone());
                layer_kv.v = Some(v_merged.clone());
                // Only mask if we are processing multiple tokens at once (prefill).
                // Single-token decode attends to all past + current without masking.
                let need_mask = seq > 1;
                (k_merged, v_merged, need_mask)
            }
            None => (k, v, true),
        };

        let total_kv_seq = k_all.dims()[2];
        let scale = (self.head_dim as f64).sqrt();
        let num_groups = self.n_heads / self.n_kv_heads;
        let expand = |t: Tensor<B, 4>| -> Tensor<B, 4> {
            let t = t.reshape([batch, self.n_kv_heads, 1, total_kv_seq, self.head_dim]);
            t.repeat_dim(2, num_groups)
                .reshape([batch, self.n_heads, total_kv_seq, self.head_dim])
        };
        let k_expanded = expand(k_all);
        let v_expanded = expand(v_all);

        let mut scores = (q / scale).matmul(k_expanded.transpose());

        if apply_causal_mask {
            let attn_mask = causal_mask::<B>(seq, self.sliding_window, device).unsqueeze::<4>();
            scores = scores.mask_fill(attn_mask.bool_not(), f64::NEG_INFINITY);
        } else if let Some(w) = self.sliding_window {
            // For single-token decode with sliding window, mask out keys beyond window.
            if total_kv_seq > w {
                let mask =
                    sliding_window_decode_mask::<B>(total_kv_seq, w, device).unsqueeze::<4>();
                scores = scores.mask_fill(mask.bool_not(), f64::NEG_INFINITY);
            }
        }

        let attn = softmax(scores, 3);
        let out = attn.matmul(v_expanded);

        let out = out
            .permute([0, 2, 1, 3])
            .reshape([batch, seq, self.n_heads * self.head_dim]);
        self.o_proj.forward(out)
    }
}

/// Build the causal (and optional sliding-window) attention mask for a single
/// sequence, shape `[seq, seq]`, as a `Bool` tensor.
fn causal_mask<B: Backend>(
    seq: usize,
    window: Option<usize>,
    device: &B::Device,
) -> Tensor<B, 2, Bool> {
    let rows = Tensor::<B, 1, Int>::arange(0..seq as i64, device).unsqueeze_dim::<2>(1);
    let cols = Tensor::<B, 1, Int>::arange(0..seq as i64, device).unsqueeze_dim::<2>(0);

    let causal = rows.clone().greater_equal(cols.clone());
    match window {
        // allowed = j >= i - (w - 1)
        Some(w) => {
            let window_mask = cols.greater_equal(rows.sub_scalar(w as i64 - 1));
            causal.bool_and(window_mask)
        }
        None => causal,
    }
}

/// Build a 1D/2D mask for single-token decode with sliding-window attention:
/// only keys in `[total - window, total)` are allowed. Shape `[1, total]`.
fn sliding_window_decode_mask<B: Backend>(
    total_kv_seq: usize,
    window: usize,
    device: &B::Device,
) -> Tensor<B, 2, Bool> {
    let cols = Tensor::<B, 1, Int>::arange(0..total_kv_seq as i64, device).unsqueeze_dim::<2>(0);
    let min_col = (total_kv_seq.saturating_sub(window)) as i64;
    cols.greater_equal_elem(min_col)
}
