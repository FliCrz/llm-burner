use super::attention::{CausalAttention, LayerKv};
use super::mlp::Mlp;
use super::rms_norm::RmsNorm;

use burn::module::Module;
use burn::tensor::{Tensor, backend::Backend};

/// One decoder layer: attention sub-layer + MLP sub-layer, both pre-normalized
/// with residual connections (Llama/Gemma/Qwen style).
#[derive(Module, Debug)]
pub struct DecoderLayer<B: Backend> {
    /// Causal self-attention.
    pub self_attn: CausalAttention<B>,
    /// Feed-forward network.
    pub mlp: Mlp<B>,
    /// Pre-attention RMS norm.
    pub input_layernorm: RmsNorm<B>,
    /// Pre-MLP RMS norm.
    pub post_attention_layernorm: RmsNorm<B>,
}

impl<B: Backend> DecoderLayer<B> {
    /// Forward pass: `x + attn(norm(x))`, then `x + mlp(norm(x))`.
    pub fn forward(&self, x: Tensor<B, 3>) -> Tensor<B, 3> {
        self.forward_with_cache(x, None, 0)
    }

    /// Forward pass supporting an optional per-layer key/value cache.
    pub fn forward_with_cache(
        &self,
        x: Tensor<B, 3>,
        cache: Option<&mut LayerKv<B>>,
        start_pos: usize,
    ) -> Tensor<B, 3> {
        let residual = x.clone();
        let normed = self.input_layernorm.forward(x);
        let attn_out = self.self_attn.forward_with_cache(normed, cache, start_pos);
        let x = attn_out.add(residual);

        let residual = x.clone();
        let normed = self.post_attention_layernorm.forward(x);
        self.mlp.forward(normed).add(residual)
    }
}
