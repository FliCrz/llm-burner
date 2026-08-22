pub mod ablation;
pub mod attention;
pub mod decoder;
pub mod load;
pub mod mlp;
#[allow(clippy::module_inception)]
pub mod model;
pub mod rms_norm;
pub mod rope;
pub mod train;

pub use model::{LlmModel, LlmModelConfig, Transformer};

/// A task-ready training/inference batch for next-token prediction.
pub use train::{CausalLmBatch, CausalLmOutput};

/// Optimizer configuration used by the training CLI.
pub use train::LlmOptimizerConfig;
