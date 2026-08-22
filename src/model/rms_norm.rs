use burn::module::{Initializer, Module, Param};
use burn::tensor::{DType, Tensor, backend::Backend};

/// Root-mean-square layer normalization (Llama/Gemma style).
///
/// Matches the Hugging Face `LlamaRMSNorm` naming: the learnable gain is a
/// parameter named `weight` of shape `[d_model]`, so safetensors weights load
/// in 1:1 without key remapping.
#[derive(Module, Debug)]
pub struct RmsNorm<B: Backend> {
    /// Learnable per-feature scaling vector, shape `[d_model]`.
    pub weight: Param<Tensor<B, 1>>,
    /// Small constant added for numerical stability.
    pub epsilon: f64,
}

impl<B: Backend> RmsNorm<B> {
    /// Create a new RMS norm initialized with ones.
    pub fn new(d_model: usize, epsilon: f64, device: &B::Device) -> Self {
        let weight = Initializer::Ones.init([d_model], device);
        Self { weight, epsilon }
    }

    /// Apply the normalization over the last dimension.
    ///
    /// `y = x / sqrt(mean(x^2) + eps) * weight`
    pub fn forward<const D: usize>(&self, x: Tensor<B, D>) -> Tensor<B, D> {
        let dtype = x.dtype();
        let rms = (x.clone().cast(DType::F32).square().mean_dim(D - 1) + self.epsilon).sqrt();
        (x / rms.cast(dtype)) * self.weight.val().unsqueeze::<D>()
    }
}
