use crate::model::{LlmModel, LlmModelConfig};

use burn::config::Config;
use burn::nn::loss::CrossEntropyLoss;
use burn::optim::{AdamWConfig, Optimizer};
use burn::tensor::backend::AutodiffBackend;
use burn::tensor::{Int, Tensor, Transaction};
use burn::train::{InferenceStep, ItemLazy, TrainOutput, TrainStep};

/// The CPU backend used for syncing items between training and validation.
type SyncBackend = burn::backend::Flex<f32, i32>;

/// A next-token prediction batch: input token ids and shifted target ids.
#[derive(Clone)]
pub struct CausalLmBatch<B: burn::tensor::backend::Backend> {
    /// Input token ids, shape `[batch, seq]`.
    pub input: Tensor<B, 2, Int>,
    /// Target token ids (next token), shape `[batch, seq]`.
    pub target: Tensor<B, 2, Int>,
}

impl<B: burn::tensor::backend::Backend> CausalLmBatch<B> {
    /// Create a new batch.
    pub fn new(input: Tensor<B, 2, Int>, target: Tensor<B, 2, Int>) -> Self {
        Self { input, target }
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
    pub fn forward_classification(&self, batch: CausalLmBatch<B>) -> CausalLmOutput<B> {
        let logits = self.forward(batch.input);
        let [batch_size, seq_len, vocab_size] = logits.dims();
        let logits_flat = logits.clone().reshape([batch_size * seq_len, vocab_size]);
        let targets_flat = batch.target.clone().reshape([batch_size * seq_len]);

        let loss =
            CrossEntropyLoss::new(None, &logits_flat.device()).forward(logits_flat, targets_flat);

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
}
