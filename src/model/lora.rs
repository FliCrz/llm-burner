//! LoRA (Low-Rank Adaptation) enabled linear layer used for fine-tuning.
//!
//! When LoRA is disabled (the default) the module is exactly a
//! [`burn::nn::Linear`] with identical field names and tensor layout, so base
//! safetensors checkpoints load and export unchanged. Calling
//! [`LoraLinear::enable_lora`] allocates a rank-`r` bypass that is *added* to
//! the (kept frozen) base transformation during the forward pass:
//!
//! ```text
//! out = input @ W^T + b + scale * ((input @ A^T) @ B^T)
//! ```
//!
//! with `A` of shape `[r, d_input]`, `B` of shape `[d_output, r]` (the PEFT
//! layout, so adapters round-trip through Hugging Face) and
//! `scale = lora_alpha / r`. Only `lora_A`/`lora_B` carry gradients while
//! fine-tuning; the base `weight`/`bias` stay frozen.
//!
//! Adapter fields use the Hugging Face PEFT names (`lora_A.weight` /
//! `lora_B.weight`), so exported adapters load into `peft` unchanged.
#![allow(non_snake_case)]

use burn::module::{Initializer, Module, Param};
use burn::nn::{Dropout, DropoutConfig};
use burn::tensor::backend::Backend;
use burn::tensor::module::linear;
use burn::tensor::ops::FloatElem;
use burn::tensor::{ElementConversion, Tensor};

/// Linear layer with an optional LoRA bypass.
#[derive(Module, Debug)]
pub struct LoraLinear<B: Backend> {
    /// Base weight, `[d_input, d_output]` (Burn layout, identical to Linear).
    pub weight: Param<Tensor<B, 2>>,
    /// Optional base bias, `[d_output]`.
    pub bias: Option<Param<Tensor<B, 1>>>,
    /// `[r, d_input]`; `None` when LoRA is disabled.
    pub lora_A: Option<Param<Tensor<B, 2>>>,
    /// `[d_output, r]`; `None` when LoRA is disabled.
    pub lora_B: Option<Param<Tensor<B, 2>>>,
    /// Multiplier applied to the LoRA branch (`lora_alpha / r`); `0` disables it.
    #[module(skip)]
    pub lora_scale: f32,
    /// Dropout applied to the LoRA hidden activation. Burn's Dropout is only
    /// active while autodiff is enabled (training), matching PEFT semantics.
    #[module(skip)]
    pub lora_dropout: Dropout,
}

impl<B: Backend> LoraLinear<B> {
    /// Build a LoRA-disabled linear layer from a `[d_input, d_output]` weight,
    /// mirroring `LinearConfig` (Burn row-major layout, bias via `initializer`).
    pub fn init(
        d_input: usize,
        d_output: usize,
        bias: bool,
        initializer: Initializer,
        device: &B::Device,
    ) -> Self {
        let weight = initializer.init_with(
            [d_input, d_output],
            Some(d_input),
            Some(d_output),
            device,
        );
        let bias = if bias {
            Some(
                initializer.init_with(
                    [d_output],
                    Some(d_input),
                    Some(d_output),
                    device,
                ),
            )
        } else {
            None
        };

        Self {
            weight,
            bias,
            lora_A: None,
            lora_B: None,
            lora_scale: 0.0,
            lora_dropout: DropoutConfig::new(0.0).init(),
        }
    }

    /// True when a LoRA bypass is installed.
    pub fn lora_enabled(&self) -> bool {
        self.lora_A.is_some() && self.lora_B.is_some() && self.lora_scale != 0.0
    }

    /// Allocate `rank`-`r` LoRA adapters over the existing base weight.
    ///
    /// `A` is initialized `[r, d_input]` from `initializer`, `B` zero
    /// `[d_output, r]`, matching the PEFT convention. The branch is scaled by
    /// `lora_alpha / r`.
    pub fn enable_lora(
        &mut self,
        rank: usize,
        lora_alpha: f64,
        dropout_p: f64,
        initializer: Initializer,
        device: &B::Device,
    ) {
        let [d_input, d_output] = self.weight.val().dims();
        let zeros = |shape: [usize; 2]| Tensor::<B, 2>::zeros(shape, device);

        self.lora_A = Some(Param::from_tensor(
            initializer.init_with(
                [rank, d_input],
                Some(d_input),
                Some(d_output),
                device,
            ).val(),
        ));
        self.lora_B = Some(Param::from_tensor(zeros([d_output, rank])));
        self.lora_scale = (lora_alpha / rank as f64) as f32;
        self.lora_dropout = DropoutConfig::new(dropout_p).init();
    }

    /// Drop the LoRA bypass, returning to a plain linear layer. Used after the
    /// adapter has been merged into `weight`.
    pub fn clear_lora(&mut self) {
        self.lora_A = None;
        self.lora_B = None;
        self.lora_scale = 0.0;
        self.lora_dropout = DropoutConfig::new(0.0).init();
    }

    /// Forward pass, identical to [`burn::nn::Linear`] when LoRA is disabled.
    pub fn forward<const D: usize>(&self, input: Tensor<B, D>) -> Tensor<B, D> {
        let out = linear(
            input.clone(),
            self.weight.val(),
            self.bias.as_ref().map(|b| b.val()),
        );

        if !self.lora_enabled() {
            return out;
        }

        let (a, b) = (self.lora_A.as_ref().unwrap(), self.lora_B.as_ref().unwrap());
        // Burn's `linear` treats the weight as `[d_in, d_out]` (no transpose),
        // so the adapter tensors (stored PEFT-style as A `[r, d_in]`, B
        // `[d_out, r]`) must be flipped explicitly to the same `[d_in, d_out]`
        // delta the merge folds: `A^T @ B^T`. Both factors are promoted to
        // input rank with leading unit dims so `matmul` broadcasts like
        // Burn's own `linear` does.
        let axes: Vec<usize> = (0..D.saturating_sub(2)).collect();
        let a = a.val().transpose().unsqueeze_dims::<D>(&axes);
        let h = self.lora_dropout.forward(input.matmul(a));
        let b = b.val().transpose().unsqueeze_dims::<D>(&axes);
        let delta = h
            .matmul(b)
            .mul_scalar(self.lora_scale.elem::<FloatElem<B>>());
        out.add(delta)
    }
}