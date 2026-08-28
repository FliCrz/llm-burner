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
    /// `true` -> `y = x_normed * (1 + weight)` (HF Gemma hidden norms),
    /// `false` -> `y = x_normed * weight` (Llama/Qwen and Gemma Q/K norms).
    pub unit_offset: bool,
}

impl<B: Backend> RmsNorm<B> {
    /// Create a new plain RMS norm initialized with ones.
    pub fn new(d_model: usize, epsilon: f64, device: &B::Device) -> Self {
        let weight = Initializer::Ones.init([d_model], device);
        Self {
            weight,
            epsilon,
            unit_offset: false,
        }
    }

    /// Enable the Gemma `x * (1 + weight)` form.
    pub fn with_unit_offset(mut self) -> Self {
        self.unit_offset = true;
        self
    }

    /// Apply the normalization over the last dimension.
    ///
    /// `y = x / sqrt(mean(x^2) + eps) * weight`
    ///
    /// With [`RmsNorm::unit_offset`] set, the gain becomes `1 + weight`,
    /// matching Hugging Face's `GemmaRMSNorm`:
    /// `y = x_normed * (1 + weight)`.
    pub fn forward<const D: usize>(&self, x: Tensor<B, D>) -> Tensor<B, D> {
        let dtype = x.dtype();
        let rms = (x.clone().cast(DType::F32).square().mean_dim(D - 1) + self.epsilon).sqrt();
        let x = x / rms.cast(dtype);
        if self.unit_offset {
            (x.clone() * self.weight.val().unsqueeze::<D>()) + x
        } else {
            x * self.weight.val().unsqueeze::<D>()
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn unit_offset_applies_one_plus_weight() {
        type B = burn::backend::Flex<f32, i32>;
        let device = burn::backend::flex::FlexDevice;

        let w: [f32; 3] = [0.5, -0.25, 2.0];
        let norm = RmsNorm::<B> {
            weight: burn::module::Param::from_data(w, &device),
            epsilon: 1e-6,
            unit_offset: true,
        };

        // x = [2, 3, 4]
        let x = Tensor::<B, 1>::from_data([2.0, 3.0, 4.0], &device);
        let out = norm.forward(x.clone());
        let out = out.to_data().to_vec::<f32>().unwrap();

        // rms = sqrt(mean(x^2) + eps), x_normed = x / rms
        let mean_sq: f32 = (4.0 + 9.0 + 16.0) / 3.0;
        let rms = mean_sq.sqrt();
        let xv = [2.0, 3.0, 4.0];
        let mut expected = Vec::new();
        for i in 0..3 {
            let xn = xv[i] / rms;
            expected.push(xn * (1.0 + w[i]));
        }
        for (o, e) in out.iter().zip(&expected) {
            assert!((o - e).abs() < 1e-5, "got {o}, expected {e}");
        }

        // Plain norm (no offset) must NOT shift: x_normed * w.
        let plain = RmsNorm::<B> {
            weight: burn::module::Param::from_data(w, &device),
            epsilon: 1e-6,
            unit_offset: false,
        };
        let out = plain.forward(x.clone()).to_data().to_vec::<f32>().unwrap();
        for i in 0..3 {
            let xn = xv[i] / rms;
            let e = xn * w[i];
            assert!((out[i] - e).abs() < 1e-5, "plain: got {}, expected {e}", out[i]);
        }
    }
}
