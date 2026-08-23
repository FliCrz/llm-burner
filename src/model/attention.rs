use super::rms_norm::RmsNorm;
use super::rope;

use burn::module::{Initializer, Module};
use burn::nn::{Linear, LinearConfig};
use burn::tensor::activation::softmax;
use burn::tensor::{Bool, Int, Tensor, backend::Backend};

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
        let initializer = Initializer::Normal {
            mean: 0.0,
            std: 0.02,
        };
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

        let (cos, sin) = rope::rope_cos_sin(seq, self.head_dim, self.rope_theta, device);
        let cos = cos.unsqueeze::<4>();
        let sin = sin.unsqueeze::<4>();

        let q = rope::rope_apply(q, cos.clone(), sin.clone());
        let k = rope::rope_apply(k, cos, sin);

        let scale = (self.head_dim as f64).sqrt();
        let num_groups = self.n_heads / self.n_kv_heads;
        let k = k.repeat_dim(1, num_groups);
        let v = v.repeat_dim(1, num_groups);

        let scores = (q / scale).matmul(k.transpose());

        let attn_mask = causal_mask::<B>(seq, self.sliding_window, device).unsqueeze::<4>();
        let scores = scores.mask_fill(attn_mask.bool_not(), f64::NEG_INFINITY);

        let attn = softmax(scores, 3);
        let out = attn.matmul(v);

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
