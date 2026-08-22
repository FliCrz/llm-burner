use burn::module::{Initializer, Module};
use burn::nn::{Linear, LinearConfig};
use burn::tensor::activation::{gelu_approximate, silu};
use burn::tensor::{Tensor, backend::Backend};

/// Feed-forward network with an activation-gated branch, matching the Llama
/// family (GEGLU) and Qwen/SmolLM/TinyLlama (SwiGLU) MLPs.
///
/// `down(gate(x).activate() * up(x))`. Field names match Hugging Face.
#[derive(Module, Debug)]
pub struct Mlp<B: Backend> {
    /// Gating projection, `hidden -> intermediate`.
    pub gate_proj: Linear<B>,
    /// Up projection, `hidden -> intermediate`.
    pub up_proj: Linear<B>,
    /// Down projection, `intermediate -> hidden`.
    pub down_proj: Linear<B>,
    /// `true` -> GELU gate (Gemma), `false` -> SiLU gate (SwiGLU, Llama/Qwen).
    pub use_gelu: bool,
}

impl<B: Backend> Mlp<B> {
    /// Create a new MLP block.
    pub fn new(
        hidden_size: usize,
        intermediate_size: usize,
        use_gelu: bool,
        device: &B::Device,
    ) -> Self {
        let initializer = Initializer::Normal {
            mean: 0.0,
            std: 0.02,
        };
        let gate_proj = LinearConfig::new(hidden_size, intermediate_size)
            .with_bias(false)
            .with_initializer(initializer.clone())
            .init(device);
        let up_proj = LinearConfig::new(hidden_size, intermediate_size)
            .with_bias(false)
            .with_initializer(initializer.clone())
            .init(device);
        let down_proj = LinearConfig::new(intermediate_size, hidden_size)
            .with_bias(false)
            .with_initializer(initializer)
            .init(device);

        Self {
            gate_proj,
            up_proj,
            down_proj,
            use_gelu,
        }
    }

    /// Forward pass over `[batch, seq, hidden]`.
    pub fn forward(&self, x: Tensor<B, 3>) -> Tensor<B, 3> {
        let gate = self.gate_proj.forward(x.clone());
        let gate = if self.use_gelu {
            gelu_approximate(gate)
        } else {
            silu(gate)
        };
        let up = self.up_proj.forward(x);
        self.down_proj.forward(gate.mul(up))
    }
}
