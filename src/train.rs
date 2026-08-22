//! Fine-tuning driver.
//
//! Runs a hand-written training loop over a fixed number of optimization
//! steps, using burn's public [`Learner`] so the model/optimizer/scheduler
//! plumbing stays idiomatic, while the step count is exact.

//! # Precision Support
//!
//! Safetensors checkpoints can be emitted in three floating-point dtypes:
//! - `F32`: Full 32-bit floating point (default).
//! - `BF16`: 16-bit bfloat16 (same dynamic range as F32, 50% memory reduction).
//! - `F16`: 16-bit floating point (smaller dynamic range).
//!
//! Training on the Flex backend itself runs with F32 math.

use crate::model::{CausalLmBatch, LlmModel, LlmModelConfig};
use crate::ui::Dashboard;

use burn::module::{AutodiffModule, Module};
use burn::optim::adaptor::OptimizerAdaptor;
use burn::optim::{AdamW, AdamWConfig, GradientsParams};
use burn::tensor::DType;
use burn::train::{Learner, LearningComponentsMarker};

/// The CPU training backend: autodiff over the flex (pure Rust) backend.
pub type TrainBackend = burn::backend::Autodiff<burn::backend::Flex<f32, i32>>;
/// The inference (weight-backed) CPU backend.
pub type FlexBackend = burn::backend::Flex<f32, i32>;

/// Precision modes supported for training and model weights.
#[derive(Debug, Clone, Copy, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
pub enum Precision {
    /// Full 32-bit floating point.
    F32,
    /// 16-bit bfloat16 (same exponent range as F32).
    Bf16,
    /// 16-bit floating point (smaller exponent range).
    F16,
}

impl Default for Precision {
    fn default() -> Self {
        Self::F32
    }
}

impl std::fmt::Display for Precision {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Precision::F32 => write!(f, "f32"),
            Precision::Bf16 => write!(f, "bf16"),
            Precision::F16 => write!(f, "f16"),
        }
    }
}

impl std::str::FromStr for Precision {
    type Err = anyhow::Error;

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        match s.to_lowercase().as_str() {
            "f32" | "f" => Ok(Precision::F32),
            "bf16" => Ok(Precision::Bf16),
            "f16" => Ok(Precision::F16),
            _ => Err(anyhow::anyhow!("Unknown precision: {}", s)),
        }
    }
}

impl Precision {
    /// Tensor dtype used when writing safetensors checkpoints.
    pub fn safetensors_dtype(self) -> DType {
        match self {
            Precision::F32 => DType::F32,
            Precision::Bf16 => DType::BF16,
            Precision::F16 => DType::F16,
        }
    }
}

#[derive(Debug, Clone)]
/// Configuration of a training run.
pub struct TrainConfig {
    /// Number of optimization steps to run.
    pub steps: usize,
    /// Number of windows per batch.
    pub batch_size: usize,
    /// Token sequence length of every window.
    pub seq_len: usize,
    /// Learning rate (AdamW).
    pub lr: f64,
    /// Weight decay (AdamW).
    pub weight_decay: f64,
    /// Report progress every `log_every` steps.
    pub log_every: usize,
    /// Floating-point dtype for exported safetensors weights.
    /// Training on the Flex backend still runs with F32 math.
    pub precision: Precision,
}

impl Default for TrainConfig {
    fn default() -> Self {
        Self {
            steps: 100,
            batch_size: 8,
            seq_len: 64,
            lr: 3e-4,
            weight_decay: 0.1,
            log_every: 10,
            precision: Precision::default(),
        }
    }
}

/// Split token windows into `batch_size`-sized groups. The last partial batch
/// is dropped so every batch is full and the step count stays exact.
pub fn partition_windows(windows: &[Vec<u32>], batch_size: usize) -> Vec<Vec<Vec<u32>>> {
    windows
        .chunks(batch_size)
        .filter(|chunk| chunk.len() == batch_size)
        .map(|chunk| chunk.to_vec())
        .collect()
}

fn build_batch<B: burn::tensor::backend::Backend>(
    windows: &[Vec<u32>],
    pad_id: u32,
    device: &B::Device,
) -> CausalLmBatch<B> {
    crate::data::build_causal_batch(windows, device, pad_id)
}

/// Train `model` for exactly `cfg.steps` steps over `windows`, reporting
/// progress through the Ratatui [`Dashboard`]. Returns the trained model on
/// the inner (non-autodiff) backend.
pub fn train_model(
    model: LlmModel<TrainBackend>,
    windows: &[Vec<u32>],
    pad_id: u32,
    cfg: &TrainConfig,
) -> LlmModel<FlexBackend> {
    assert!(
        windows.len() >= cfg.batch_size,
        "need at least `batch_size` ({}) windows, got {}",
        cfg.batch_size,
        windows.len()
    );

    let device = Default::default();
    let optimizer = AdamWConfig::new()
        .with_weight_decay(cfg.weight_decay as f32)
        .with_beta_1(0.9)
        .with_beta_2(0.95)
        .with_epsilon(1e-8)
        .init::<TrainBackend, LlmModel<TrainBackend>>();
    let lr_scheduler: burn::optim::LearningRate = cfg.lr;

    let mut learner = Learner::<
        LearningComponentsMarker<
            TrainBackend,
            burn::optim::LearningRate,
            LlmModel<TrainBackend>,
            OptimizerAdaptor<AdamW, LlmModel<TrainBackend>, TrainBackend>,
        >,
    >::new(model, optimizer, lr_scheduler);
    learner.lr_step();

    let batches = partition_windows(windows, cfg.batch_size);
    let dashboard = Dashboard::start(cfg.steps);
    let mut last_loss = 0.0;

    for step in 1..=cfg.steps {
        let batch_windows = &batches[(step - 1) % batches.len()];
        let batch: CausalLmBatch<TrainBackend> = build_batch(batch_windows, pad_id, &device);

        let output = learner.train_step(batch);
        let loss = output.item.loss.clone().into_scalar();
        let gradients: GradientsParams = output.grads;

        learner.optimizer_step(gradients);
        learner.lr_step();

        last_loss = loss;
        dashboard.update(step, loss);

        if cfg.log_every > 0 && step % cfg.log_every == 0 {
            log::info!("step {}/{} loss={:.6} lr={}", step, cfg.steps, loss, cfg.lr);
        }
    }

    dashboard.finish(cfg.steps, last_loss);

    let model = learner.model();
    model.valid()
}

/// Move a loaded flex (non-autodiff) model onto the training backend.
pub fn to_train_backend(model: LlmModel<FlexBackend>) -> LlmModel<TrainBackend> {
    model.train::<TrainBackend>()
}

/// Move a trained autodiff model back onto the flex backend.
pub fn to_flex_backend(model: LlmModel<TrainBackend>) -> LlmModel<FlexBackend> {
    model.valid()
}

/// Build a fresh model for the given config on the training backend.
pub fn train_model_from_config(config: &LlmModelConfig) -> LlmModel<TrainBackend> {
    let device = Default::default();
    LlmModel::new(config, &device).train::<TrainBackend>()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn partition_keeps_full_batches_only() {
        let windows: Vec<Vec<u32>> = (0..11).map(|i| vec![i as u32; 8]).collect();
        let batches = partition_windows(&windows, 4);
        assert_eq!(batches.len(), 2);
        for batch in &batches {
            assert_eq!(batch.len(), 4);
        }
    }

    #[test]
    fn empty_windows_yield_no_batches() {
        assert!(partition_windows(&[], 4).is_empty());
    }

    #[test]
    fn train_model_reduces_loss_on_toy_sequence() {
        let config = LlmModelConfig::tiny();
        let model = train_model_from_config(&config);

        let sequence: Vec<u32> = (0..16).cycle().take(64).collect();
        let windows: Vec<Vec<u32>> = sequence.chunks(8).map(|c| c.to_vec()).collect();

        let cfg = TrainConfig {
            steps: 20,
            batch_size: 2,
            seq_len: 8,
            lr: 1e-3,
            weight_decay: 0.0,
            log_every: 0,
            precision: Precision::F32,
        };

        let device = Default::default();
        let init_batch = build_batch::<FlexBackend>(&windows[0..2], 0, &device);
        let init_loss: f32 = model
            .valid()
            .forward_classification(init_batch)
            .loss
            .into_scalar();

        let trained = train_model(model, &windows, 0, &cfg);

        let final_batch = build_batch::<FlexBackend>(&windows[0..2], 0, &device);
        let final_loss: f32 = trained
            .forward_classification(final_batch)
            .loss
            .into_scalar();

        assert!(
            final_loss < init_loss,
            "expected final_loss ({final_loss}) < init_loss ({init_loss})"
        );
    }
}
