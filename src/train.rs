//! Fine-tuning driver.
//!
//! Runs a hand-written training loop over a fixed number of optimization
//! steps, using burn's public [`Learner`] so the model/optimizer/scheduler
//! plumbing stays idiomatic, while the step count is exact.
//!
//! # Precision Support
//!
//! [`Precision`] selects the dtype used for weights *and* training math:
//! - `F32`: full 32-bit floats (default; most stable).
//! - `BF16`: bfloat16 — same exponent range as F32, 50% memory reduction.
//! - `F16`: IEEE half — smaller dynamic range.
//!
//! The whole pipeline (checkpoint load, forward/backward, optimizer state,
//! export) runs in the selected dtype on the compiled backend. F32 remains the
//! default because pure half-precision AdamW can be numerically fragile.

/// The weight-backed inference/export backend.
///
/// With the default `gpu` feature this is Burn's fused wgpu backend (Vulkan
/// on Linux/Windows, Metal on macOS) running in f32; half-precision training
/// instantiates the same backend with bf16/f16 element types at the dispatch
/// site in [`crate::pipeline::run_pipeline`]. Without the gpu feature
/// (`--no-default-features --features flex`) everything runs on the pure-Rust
/// CPU backend, which computes in f32 only.
#[cfg(feature = "gpu")]
pub type InferBackend = burn::backend::Wgpu<f32, i32>;
#[cfg(not(feature = "gpu"))]
pub type InferBackend = burn::backend::Flex<f32, i32>;

/// The training backend: autodiff over [`InferBackend`].
pub type TrainBackend = burn::backend::Autodiff<InferBackend>;

/// Backend used by unit tests.
///
/// Always the pure-Rust CPU backend so `cargo test` never requires a GPU or
/// Vulkan driver, even in default (GPU) builds.
pub type TestBackend = burn::backend::Flex<f32, i32>;

/// Human-readable name of the compiled backend for startup logs.
pub fn backend_label() -> &'static str {
    #[cfg(feature = "gpu")]
    {
        "wgpu/Vulkan (autodiff + fusion)"
    }
    #[cfg(not(feature = "gpu"))]
    {
        "Flex (CPU)"
    }
}

use std::path::PathBuf;

use crate::model::{CausalLmBatch, LlmModel, LlmModelConfig};
use crate::ui::Dashboard;

use burn::module::{AutodiffModule, Module};
use burn::optim::adaptor::OptimizerAdaptor;
use burn::optim::{AdamW, AdamWConfig, GradientsParams};
use burn::tensor::DType;
use burn::tensor::backend::{AutodiffBackend, Backend};
use burn::train::{Learner, LearningComponentsMarker};

/// Precision modes supported for training and model weights.
///
/// On GPU builds the requested dtype drives the whole stack (checkpoint load,
/// forward/backward math, optimizer state, export). If the GPU probe rejects
/// the dtype (buggy driver), or on CPU-only builds, training computes in f32
/// while the checkpoint load/export still honor the requested dtype — fp32
/// master weights are also the numerically safer recipe for half-precision
/// AdamW.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
pub enum Precision {
    /// Full 32-bit floating point.
    #[default]
    F32,
    /// 16-bit bfloat16 (same exponent range as F32).
    Bf16,
    /// 16-bit floating point (smaller exponent range).
    F16,
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
    /// Tensor dtype used for model weights (loading, compute, and exported
    /// safetensors checkpoints).
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
    /// Floating-point dtype for weights, training math, and safetensors export.
    pub precision: Precision,
    /// Show the Ratatui progress dashboard while training. When disabled
    /// (`--no-tui`), progress is reported through the log file only — useful
    /// for tests and non-interactive runs.
    pub tui: bool,
    /// While the training dashboard owns the terminal, raw stdout/stderr are
    /// redirected into this file so library output cannot garble the TUI.
    pub output_redirect: Option<PathBuf>,
    /// Model/dataset labels shown in the dashboard header. Empty by default
    /// (the header line is hidden); the pipeline fills it from the input
    /// directories.
    pub run_info: crate::ui::RunInfo,
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
            tui: true,
            output_redirect: None,
            run_info: Default::default(),
        }
    }
}

/// Build a causal-LM batch from a contiguous slice of consecutive windows.
fn build_batch<B: Backend>(
    flat_tokens: &[u32],
    seq_len: usize,
    pad_id: u32,
    device: &B::Device,
) -> CausalLmBatch<B> {
    crate::data::build_causal_batch_from_flat(flat_tokens, seq_len, device, pad_id)
}

/// Train `model` for exactly `cfg.steps` steps over the token windows in
/// `windows`, reporting progress through the Ratatui [`Dashboard`]. Returns
/// the trained model on the inner (non-autodiff) backend.
pub fn train_model<B>(
    model: LlmModel<B>,
    windows: &crate::data::WindowStore,
    pad_id: u32,
    cfg: &TrainConfig,
) -> LlmModel<B::InnerBackend>
where
    B: AutodiffBackend,
    B::FloatElem: Into<f32>,
{
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
        .init::<B, LlmModel<B>>();
    let lr_scheduler: burn::optim::LearningRate = cfg.lr;

    let mut learner = Learner::<
        LearningComponentsMarker<
            B,
            burn::optim::LearningRate,
            LlmModel<B>,
            OptimizerAdaptor<AdamW, LlmModel<B>, B>,
        >,
    >::new(model, optimizer, lr_scheduler);
    learner.lr_step();

    // A trailing partial batch is dropped so the step count stays exact.
    let batches = windows.batch_count(cfg.batch_size);
    let dashboard = if cfg.tui {
        Some(Dashboard::start_with_output_redirect(
            cfg.run_info.clone(),
            cfg.steps,
            cfg.output_redirect.as_deref(),
        ))
    } else {
        None
    };
    let mut last_loss = 0.0;
    let mut step = 0usize;

    while step < cfg.steps {
        let batch_index = step % batches.max(1);
        let start_window = batch_index * cfg.batch_size;
        let flat = windows.window_tokens(start_window, cfg.batch_size);

        let batch = build_batch::<B>(flat, windows.seq_len(), pad_id, &device);

        let output = learner.train_step(batch);
        let loss: f32 = output.item.loss.clone().into_scalar().into();
        let gradients: GradientsParams = output.grads;

        learner.optimizer_step(gradients);
        learner.lr_step();

        step += 1;
        last_loss = loss;
        if let Some(dashboard) = &dashboard {
            dashboard.update(step, loss);
        }

        if cfg.log_every > 0 && step.is_multiple_of(cfg.log_every) {
            log::info!("step {}/{} loss={:.6} lr={}", step, cfg.steps, loss, cfg.lr);
        }
    }

    if let Some(dashboard) = dashboard {
        dashboard.finish(cfg.steps, last_loss);
    }

    let model = learner.model();
    model.valid()
}

/// Move an inference (non-autodiff) model onto the autodiff training backend.
pub fn to_train_backend<B: Backend>(model: LlmModel<B>) -> LlmModel<burn::backend::Autodiff<B>> {
    model.train()
}

/// Move a trained autodiff model back onto its inference backend.
pub fn to_infer_backend<B: AutodiffBackend>(model: LlmModel<B>) -> LlmModel<B::InnerBackend> {
    model.valid()
}

/// Build a fresh model for the given config on the training backend.
pub fn train_model_from_config<B: AutodiffBackend>(config: &LlmModelConfig) -> LlmModel<B> {
    let device = Default::default();
    LlmModel::new(config, &device).train::<B>()
}

#[cfg(test)]
mod tests {
    // Only the gpu-gated integration test needs the parent scope.
    #[cfg(not(feature = "gpu"))]
    use super::*;

    #[test]
    fn window_store_drops_partial_batches() {
        let mut store = crate::data::WindowStore::new(4);
        store.extend_windows(&[1; 8]);
        store.extend_windows(&[2; 4]);
        store.push_padded_tail(&[3], 0);
        // 4 complete windows: two full batches of 2, none for batch_size 8.
        assert_eq!(store.len(), 4);
        assert_eq!(store.total_tokens(), 16);
        assert_eq!(store.batch_count(2), 2);
        assert_eq!(store.batch_count(8), 0);
        assert_eq!(store.window_tokens(2, 2), &[2, 2, 2, 2, 3, 0, 0, 0]);
    }

    #[test]
    #[cfg(not(feature = "gpu"))] // trains on the compiled TrainBackend (GPU under `gpu`)
    fn train_model_reduces_loss_on_toy_sequence() {
        let config = LlmModelConfig::tiny();
        let model = train_model_from_config::<TrainBackend>(&config);

        let sequence: Vec<u32> = (0..16).cycle().take(64).collect();

        let mut windows = crate::data::WindowStore::new(8);
        let chunked: Vec<Vec<u32>> = sequence.chunks(8).map(|c| c.to_vec()).collect();
        for chunk in &chunked[..chunked.len() - 1] {
            windows.extend_windows(chunk);
        }
        windows.push_padded_tail(chunked.last().unwrap(), 0);

        let cfg = TrainConfig {
            steps: 20,
            batch_size: 2,
            seq_len: 8,
            lr: 1e-3,
            weight_decay: 0.0,
            log_every: 0,
            precision: Precision::F32,
            tui: false,
            output_redirect: None,
            run_info: Default::default(),
        };

        let device = Default::default();
        let init_batch = build_batch::<InferBackend>(windows.window_tokens(0, 2), 8, 0, &device);
        let init_loss: f32 = model
            .valid()
            .forward_classification(init_batch)
            .loss
            .into_scalar();

        let trained = train_model(model, &windows, 0, &cfg);

        let final_batch = build_batch::<InferBackend>(windows.window_tokens(0, 2), 8, 0, &device);
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
