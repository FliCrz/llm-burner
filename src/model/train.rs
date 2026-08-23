use crate::model::{LlmModel, LlmModelConfig};

use burn::config::Config;
use burn::nn::loss::CrossEntropyLoss;
use burn::optim::{AdamWConfig, Optimizer};
use burn::tensor::backend::AutodiffBackend;
use burn::tensor::{Int, Tensor, Transaction};
use burn::train::{InferenceStep, ItemLazy, TrainOutput, TrainStep};

/// CPU backend used for syncing items between training and validation.
///
/// Deliberately pinned to the pure-Rust Flex backend regardless of the
/// compiled training device: synced loss/logit data is materialized on the
/// host so the dashboard never allocates on the GPU per step.
type SyncBackend = burn::backend::Flex<f32, i32>;

/// A next-token prediction batch: input token ids and shifted target ids.
#[derive(Clone)]
pub struct CausalLmBatch<B: burn::tensor::backend::Backend> {
    /// Input token ids, shape `[batch, seq]`.
    pub input: Tensor<B, 2, Int>,
    /// Target token ids (next token), shape `[batch, seq]`.
    pub target: Tensor<B, 2, Int>,
    /// Target value that marks padding; excluded from the loss when set.
    ///
    /// Window tails are padded and every row's final target is forced to this
    /// value, so without masking those positions would teach the model to
    /// emit padding.
    pub pad_id: Option<u32>,
}

impl<B: burn::tensor::backend::Backend> CausalLmBatch<B> {
    /// Create a new unmasked batch.
    pub fn new(input: Tensor<B, 2, Int>, target: Tensor<B, 2, Int>) -> Self {
        Self {
            input,
            target,
            pad_id: None,
        }
    }

    /// Create a batch whose `pad_id` targets are ignored by the loss.
    pub fn new_masked(input: Tensor<B, 2, Int>, target: Tensor<B, 2, Int>, pad_id: u32) -> Self {
        Self {
            input,
            target,
            pad_id: Some(pad_id),
        }
    }
}

/// Output of a causal LM forward pass for training.
#[derive(Clone)]
pub struct CausalLmOutput<B: burn::tensor::backend::Backend> {
    /// Mean cross-entropy loss over non-padded tokens.
    pub loss: Tensor<B, 1>,
    /// Raw logits, shape `[batch, seq, vocab]`.
    pub logits: Tensor<B, 3>,
    /// Target token ids, shape `[batch, seq]`.
    pub targets: Tensor<B, 2, Int>,
}

impl<B: burn::tensor::backend::Backend> CausalLmOutput<B> {
    /// Create a new output.
    pub fn new(loss: Tensor<B, 1>, logits: Tensor<B, 3>, targets: Tensor<B, 2, Int>) -> Self {
        Self {
            loss,
            logits,
            targets,
        }
    }
}

impl<B: burn::tensor::backend::Backend> ItemLazy for CausalLmOutput<B> {
    type ItemSync = CausalLmOutput<SyncBackend>;

    fn sync(self) -> Self::ItemSync {
        let [loss, logits, targets] = Transaction::default()
            .register(self.loss)
            .register(self.logits)
            .register(self.targets)
            .execute()
            .try_into()
            .expect("Correct amount of tensor data");

        let device = &Default::default();
        CausalLmOutput {
            loss: Tensor::from_data(loss, device),
            logits: Tensor::from_data(logits, device),
            targets: Tensor::from_data(targets, device),
        }
    }
}

impl<B: burn::tensor::backend::Backend> LlmModel<B> {
    /// Compute the cross-entropy loss and logits for a batch.
    ///
    /// Positions whose target equals the batch's `pad_id` are excluded from
    /// the mean, so padding never enters the gradient. (burn's own pad-token
    /// handling zeroes excluded positions but still divides by the full
    /// position count, which would dilute the loss whenever a window tail is
    /// padded — hence the manual renormalization.)
    pub fn forward_classification(&self, batch: CausalLmBatch<B>) -> CausalLmOutput<B> {
        let logits = self.forward(batch.input);
        let [batch_size, seq_len, vocab_size] = logits.dims();
        let logits_flat = logits.clone().reshape([batch_size * seq_len, vocab_size]);
        let targets_flat = batch.target.clone().reshape([batch_size * seq_len]);

        let loss = CrossEntropyLoss::new(batch.pad_id.map(|p| p as usize), &logits_flat.device())
            .forward(logits_flat, targets_flat.clone());

        if let Some(pad) = batch.pad_id {
            // Renormalize: burn divided by all positions, we want the mean
            // over retained (non-pad) targets only.
            let positions = (batch_size * seq_len) as f32;
            let device = &logits.device();
            let total = Tensor::<B, 1>::from_floats([positions], device);
            let kept = targets_flat
                .not_equal_elem(pad as i64)
                .int()
                .sum()
                .float()
                .clamp_min(1.0)
                .reshape([1]);
            let loss = loss.mul(total).div(kept);
            return CausalLmOutput::new(loss, logits, batch.target);
        }

        CausalLmOutput::new(loss, logits, batch.target)
    }
}

/// Training step for the autodiff backend.
impl<B: AutodiffBackend> TrainStep for LlmModel<B> {
    type Input = CausalLmBatch<B>;
    type Output = CausalLmOutput<B>;

    fn step(&self, batch: Self::Input) -> TrainOutput<Self::Output> {
        let item = self.forward_classification(batch);
        let grads = item.loss.clone().backward();
        TrainOutput::new(self, grads, item)
    }
}

/// Inference step for a plain (non-autodiff) backend.
impl<B: burn::tensor::backend::Backend> InferenceStep for LlmModel<B> {
    type Input = CausalLmBatch<B>;
    type Output = CausalLmOutput<B>;

    fn step(&self, batch: Self::Input) -> Self::Output {
        self.forward_classification(batch)
    }
}

/// Optimizer used for fine-tuning.
#[derive(Config, Debug)]
pub struct LlmOptimizerConfig {
    /// Learning rate passed to the optimizer.
    #[config(default = 3e-4)]
    pub learning_rate: f64,
}

impl LlmOptimizerConfig {
    /// Initialize the AdamW optimizer for a given training model type.
    pub fn init_optimizer<B: AutodiffBackend, M: burn::module::AutodiffModule<B>>(
        &self,
    ) -> impl Optimizer<M, B> {
        AdamWConfig::new()
            .with_weight_decay(0.1)
            .with_beta_1(0.9)
            .with_beta_2(0.95)
            .with_epsilon(1e-8)
            .init::<B, M>()
    }
}

/// Free helper to build a freshly initialized model for a backend.
pub fn model_from_config<B: burn::tensor::backend::Backend>(
    config: &LlmModelConfig,
    device: &B::Device,
) -> LlmModel<B> {
    LlmModel::new(config, device)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn forward_classification_shapes() {
        type B = burn::backend::Flex<f32, i32>;
        let device = burn::backend::flex::FlexDevice;
        let config = LlmModelConfig::tiny();
        let model = LlmModel::<B>::new(&config, &device);

        let input = Tensor::<B, 2, Int>::from_data([[1, 5, 9, 12]], &device);
        let target = Tensor::<B, 2, Int>::from_data([[5, 9, 12, 3]], &device);
        let output = model.forward_classification(CausalLmBatch::new(input, target));

        assert_eq!(output.logits.dims(), [1, 4, config.vocab_size]);
        assert_eq!(output.loss.dims(), [1]);
        assert_eq!(output.targets.dims(), [1, 4]);
    }

    /// Masked batches must average cross-entropy over the non-pad targets
    /// only: padding must not leak into training gradients.
    #[test]
    fn loss_ignores_pad_targets() {
        type B = burn::backend::Flex<f32, i32>;
        let device = burn::backend::flex::FlexDevice;
        let config = LlmModelConfig::tiny();
        let model = LlmModel::<B>::new(&config, &device);

        let pad = 0u32;
        let input = Tensor::<B, 2, Int>::from_data([[1, 5, 9, 12]], &device);
        let target = Tensor::<B, 2, Int>::from_data([[5, 9, 12, pad]], &device);
        let masked = model.forward_classification(CausalLmBatch::new_masked(
            input.clone(),
            target.clone(),
            pad,
        ));
        let full = model.forward_classification(CausalLmBatch::new(input, target));

        // Reference: plain CE over the three leading (non-pad) positions.
        let [_, _, vocab] = masked.logits.dims();
        let logits3 = masked
            .logits
            .clone()
            .slice([0..1, 0..3])
            .reshape([3, vocab]);
        let targets3 = masked.targets.slice([0..1, 0..3]).reshape([3]);
        let reference = CrossEntropyLoss::new(None, &device).forward(logits3, targets3);

        let masked_loss: f32 = masked.loss.into_scalar();
        let reference_loss: f32 = reference.into_scalar();
        let full_loss: f32 = full.loss.into_scalar();

        assert!(
            (masked_loss - reference_loss).abs() < 1e-4,
            "masked loss {masked_loss} != non-pad-only reference {reference_loss}"
        );
        assert!(
            (full_loss - masked_loss).abs() > 1e-4,
            "pad position should change the unmasked loss"
        );
    }
}
