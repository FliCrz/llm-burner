//! Fine-tuning support: frozen-base LoRA, matching the duties of QLoRA
//! (train a small rank-`r` adapter over a frozen base model) on the dtypes and
//! hardware llm-burner targets.
//!
//! This module wires [`LoraLinear`] into a loaded [`LlmModel`]:
//!
//! - [`inject_lora`] allocates `lora_A`/`lora_B` on every attention and MLP
//!   projection.
//! - [`freeze_base`] marks every base `weight`/`bias` as not-requiring-grad,
//!   so gradients and optimizer state touch only the adapters.
//! - [`merge_lora`] folds the learned delta back into `weight` and drops the
//!   adapters (leaving a plain model ready to export).
//! - [`export_adapter`] writes a PEFT-compatible `adapter_model.safetensors`
//!   plus `adapter_config.json`.

use std::path::Path;

use anyhow::{Context, Result};
use burn::module::{Initializer, Module, ModuleMapper, Param};
use burn::tensor::backend::Backend;
use burn::tensor::ops::FloatElem;
use burn::tensor::{ElementConversion, Tensor};
use burn_store::{ModuleStore, SafetensorsStore};

use crate::lora::LoraConfig;
use crate::model::load::FloatDTypeAdapter;
use crate::model::{LlmModel, LlmModelConfig};

/// LoRA hyperparameters used while fine-tuning.
#[derive(Debug, Clone, PartialEq)]
pub struct LoraTrainConfig {
    /// Rank `r` of the low-rank adapter.
    pub rank: usize,
    /// Scaling hyperparameter; the branch is scaled by `lora_alpha / r`.
    pub alpha: f64,
    /// Dropout probability applied to the adapter activation during training.
    pub dropout: f64,
}

impl Default for LoraTrainConfig {
    fn default() -> Self {
        Self {
            rank: 8,
            alpha: 16.0,
            dropout: 0.0,
        }
    }
}

/// Allocate `lora_A`/`lora_B` on the query, key, value, output, gate, up and
/// down projections of every decoder layer. The base gradients stay frozen
/// (see [`freeze_base`]).
pub fn inject_lora<B: Backend>(
    model: &mut LlmModel<B>,
    config: &LoraTrainConfig,
    device: &B::Device,
) {
    let initializer = Initializer::Normal {
        mean: 0.0,
        std: 0.02,
    };
    for layer in &mut model.model.layers {
        let attn = &mut layer.self_attn;
        for proj in [
            &mut attn.q_proj,
            &mut attn.k_proj,
            &mut attn.v_proj,
            &mut attn.o_proj,
        ] {
            proj.enable_lora(config.rank, config.alpha, config.dropout, initializer.clone(), device);
        }
        let mlp = &mut layer.mlp;
        for proj in [
            &mut mlp.gate_proj,
            &mut mlp.up_proj,
            &mut mlp.down_proj,
        ] {
            proj.enable_lora(config.rank, config.alpha, config.dropout, initializer.clone(), device);
        }
    }
}

/// Tracks a module-traversal path so [`FreezeBase`] can tell adapter params
/// from base params.
#[derive(Default)]
struct PathStack(Vec<String>);

impl PathStack {
    fn inside_lora(&self) -> bool {
        self.0.iter().any(|name| name == "lora_A" || name == "lora_B")
    }
}

/// Module mapper that freezes every param except `lora_A`/`lora_B`.
///
/// `require_grad(false)` keeps the base weights out of automatic
/// differentiation entirely, so the optimizer only ever sees adapter
/// gradients (and only allocates AdamW state for them).
struct FreezeBase {
    path: PathStack,
}

impl<B: Backend> ModuleMapper<B> for FreezeBase {
    fn enter_module(&mut self, name: &str, _container_type: &str) {
        self.path.0.push(name.to_string());
    }

    fn exit_module(&mut self, _name: &str, _container_type: &str) {
        self.path.0.pop();
    }

    fn map_float<const D: usize>(&mut self, param: Param<Tensor<B, D>>) -> Param<Tensor<B, D>> {
        let (id, tensor, mapper) = param.consume();
        let tensor = if self.path.inside_lora() {
            tensor
        } else {
            tensor.set_require_grad(false)
        };
        Param::from_mapped_value(id, tensor, mapper)
    }
}

/// Freeze all base weights so only the LoRA adapters are trainable.
///
/// Consumes and returns the model because [`Module::map`] moves it; the
/// derived module holds skip fields of no state that would be disturbed.
pub fn freeze_base<B: Backend>(model: LlmModel<B>) -> LlmModel<B> {
    model.map(&mut FreezeBase {
        path: PathStack::default(),
    })
}

/// Number of parameters the injected LoRA adapters add for `config`.
///
/// Mirrors [`inject_lora`]: four attention projections (`q/k/v/o`, with `o`
/// reading back the concatenated attention output) and three MLP projections
/// (`gate`/`up` writing `d_model -> intermediate`, `down` going back), ranked
/// `r` each, every adapter contributing `r * (d_input + d_output)`.
pub fn adapter_params(config: &LlmModelConfig, lora: &LoraTrainConfig) -> u64 {
    let d = config.d_model as u64;
    let q_out = (config.n_heads * config.head_dim) as u64;
    let kv_out = (config.n_kv_heads * config.head_dim) as u64;
    let i = config.intermediate_size as u64;
    let r = lora.rank as u64;
    let attn = (d + q_out) + 2 * (d + kv_out) + (q_out + d);
    let mlp = 2 * (d + i) + (i + d);
    r * (attn + mlp) * config.n_layers as u64
}

/// Total number of element reads/writes trained by LoRA.
pub fn trainable_elements<B: Backend>(model: &LlmModel<B>) -> usize {
    // `collect` only materializes per-tensor snapshots; counting the adapter
    // fields directly is cheaper than summing `num_params()` of an adapter-only shell.
    let mut total = 0usize;
    for layer in &model.model.layers {
        for proj in [
            &layer.self_attn.q_proj,
            &layer.self_attn.k_proj,
            &layer.self_attn.v_proj,
            &layer.self_attn.o_proj,
            &layer.mlp.gate_proj,
            &layer.mlp.up_proj,
            &layer.mlp.down_proj,
        ] {
            for p in [&proj.lora_A, &proj.lora_B] {
                if let Some(param) = p {
                    total += param.val().shape().num_elements();
                }
            }
        }
    }
    total
}

/// Fold `scale * (A^T @ B^T)` into each projection's `weight` and drop the
/// adapters. Returns the number of modules merged.
pub fn merge_lora<B: Backend>(model: &mut LlmModel<B>) -> usize {
    let mut merged = 0usize;
    for layer in &mut model.model.layers {
        for proj in [
            &mut layer.self_attn.q_proj,
            &mut layer.self_attn.k_proj,
            &mut layer.self_attn.v_proj,
            &mut layer.self_attn.o_proj,
            &mut layer.mlp.gate_proj,
            &mut layer.mlp.up_proj,
            &mut layer.mlp.down_proj,
        ] {
            if !proj.lora_enabled() {
                continue;
            }
            let scale = proj.lora_scale;
            let a = proj.lora_A.as_ref().unwrap().val();
            let b = proj.lora_B.as_ref().unwrap().val();
            // Burn layout `[in, out]` delta = (B @ A)^T = A^T @ B^T.
            let delta = a
                .transpose()
                .matmul(b.transpose())
                .mul_scalar(scale.elem::<FloatElem<B>>());
            let w = proj.weight.val().add(delta);
            proj.weight = Param::from_tensor(w);
            proj.clear_lora();
            merged += 1;
        }
    }
    merged
}

/// Write a PEFT-compatible adapter (`adapter_model.safetensors` +
/// `adapter_config.json`) for the currently-installed LoRA adapters into
/// `out_dir`. The model may have its adapters still installed (not merged).
pub fn export_adapter<B: Backend>(
    model: &LlmModel<B>,
    out_dir: &Path,
    load_dtype: burn::tensor::DType,
    config: &LoraTrainConfig,
    base_model_name_or_path: &str,
) -> Result<()> {
    std::fs::create_dir_all(out_dir)
        .with_context(|| format!("failed to create `{}`", out_dir.display()))?;

    let adapter_path = out_dir.join("adapter_model.safetensors");
    let mut store = SafetensorsStore::from_file(&adapter_path)
        .with_to_adapter(FloatDTypeAdapter::new(load_dtype))
        .overwrite(true)
        .with_predicate(|name, _full| {
            name.ends_with("lora_A.weight") || name.ends_with("lora_B.weight")
        });
    store
        .collect_from(model)
        .with_context(|| format!("failed to write `{}`", adapter_path.display()))?;
    log::info!("wrote adapter to {}", adapter_path.display());

    let json_path = out_dir.join("adapter_config.json");
    let adapter_config = LoraConfig {
        base_model_name_or_path: Some(base_model_name_or_path.to_string()),
        peft_type: Some("LORA".to_string()),
        r: Some(config.rank),
        lora_alpha: Some(config.alpha),
        lora_dropout: Some(config.dropout),
        target_modules: Some(serde_json::json!([
            "q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"
        ])),
        // The adapter is exported exactly matching the base (scaling baked in
        // via `lora_alpha`), so PEFT re-emits the same branch the model used.
        use_rslora: None,
        bias: None,
        ..Default::default()
    };
    let json = serde_json::to_string_pretty(&adapter_config)
        .with_context(|| format!("failed to serialize `{}`", json_path.display()))?;
    std::fs::write(&json_path, json)
        .with_context(|| format!("failed to write `{}`", json_path.display()))?;
    log::info!("wrote adapter config to {}", json_path.display());

    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::model::LlmModelConfig;
    use burn::nn::LinearConfig;

    fn elements(shape: &burn::tensor::Shape) -> usize {
        shape.iter().product()
    }

    #[test]
    fn injects_and_merges() {
        type B = burn::backend::Flex<f32, i32>;
        let device = burn::backend::flex::FlexDevice;
        let config = LlmModelConfig::tiny();
        let mut model = LlmModel::<B>::new(&config, &device);

        let lora = LoraTrainConfig {
            rank: 2,
            alpha: 4.0,
            dropout: 0.1,
        };
        inject_lora(&mut model, &lora, &device);

        // Every target projection must now have adapters installed.
        assert!(model.model.layers[0].self_attn.q_proj.lora_enabled());
        assert!(model.model.layers[0].self_attn.o_proj.lora_enabled());
        assert!(model.model.layers[0].mlp.gate_proj.lora_enabled());

        // Number of adapter elements matches 7 projections x layers.
        let mut expected = 0usize;
        for layer in &model.model.layers {
            for proj in [
                &layer.self_attn.q_proj,
                &layer.self_attn.k_proj,
                &layer.self_attn.v_proj,
                &layer.self_attn.o_proj,
                &layer.mlp.gate_proj,
                &layer.mlp.up_proj,
                &layer.mlp.down_proj,
            ] {
                expected += elements(&proj.lora_A.as_ref().unwrap().val().shape())
                    + elements(&proj.lora_B.as_ref().unwrap().val().shape());
            }
        }
        assert_eq!(trainable_elements(&model), expected);

        // Install a known adapter (A = B = ones, scale = alpha / r = 2) on the
        // first q_proj and verify merging folds exactly `2 * ones` into weight.
        {
            let proj = &mut model.model.layers[0].self_attn.q_proj;
            let [d_in, d_out] = proj.weight.val().dims();
            let r = lora.rank;
            proj.lora_A = Some(Param::from_tensor(Tensor::<B, 2>::ones([r, d_in], &device)));
            proj.lora_B = Some(Param::from_tensor(Tensor::<B, 2>::ones([d_out, r], &device)));
        }
        let before = model.model.layers[0]
            .self_attn
            .q_proj
            .weight
            .val()
            .clone();

        let merged = merge_lora(&mut model);
        assert_eq!(merged, 7 * config.n_layers);
        assert!(!model.model.layers[0].self_attn.q_proj.lora_enabled());
        assert!(model.model.layers[0].self_attn.q_proj.lora_A.is_none());

        // delta_burn = scale * A^T @ B^T = (alpha/r) * r * ones[d_in, d_out] = 4 * ones.
        let after = model.model.layers[0].self_attn.q_proj.weight.val();
        let drift = after.clone().sub(before);
        let drift_data = drift.into_data();
        let drift_f32: Vec<f32> = drift_data.to_vec().unwrap();
        assert!(
            drift_f32.iter().all(|&v| (v - 4.0).abs() < 1e-5),
            "expected drift of exactly 4.0 per element, got {drift_f32:?}"
        );
    }

    #[test]
    fn enabled_forward_equals_post_merge_forward() {
        type B = burn::backend::Flex<f32, i32>;
        let device = burn::backend::flex::FlexDevice;
        let config = LlmModelConfig::tiny();
        let mut model = LlmModel::<B>::new(&config, &device);
        let lora = LoraTrainConfig {
            rank: 2,
            alpha: 4.0,
            dropout: 0.0,
        };
        inject_lora(&mut model, &lora, &device);

        // Install a known adapter on the first q_proj: A = B = ones, so the
        // branch is `(alpha/r) * r = 4` per element.
        let proj = &mut model.model.layers[0].self_attn.q_proj;
        let [d_in, d_out] = proj.weight.val().dims();
        let r = lora.rank;
        proj.lora_A = Some(Param::from_tensor(Tensor::<B, 2>::ones([r, d_in], &device)));
        proj.lora_B = Some(Param::from_tensor(Tensor::<B, 2>::ones([d_out, r], &device)));

        let input = Tensor::<B, 3>::ones([1, 3, d_in], &device);
        let enabled_out = proj.forward(input.clone());

        let merged = merge_lora(&mut model);
        assert_eq!(merged, 7 * config.n_layers);
        let merged_out = model.model.layers[0].self_attn.q_proj.forward(input.clone());

        let diff: f32 = enabled_out.sub(merged_out).abs().max().into_scalar();
        // Exact real arithmetic would give 0; f32 matmul re-association
        // (`(x@Aᵀ)@Bᵀ` vs `x@(Aᵀ@Bᵀ)` and the add order) leaves a rounding
        // trail well under 1e-3 here.
        assert!(
            diff < 1e-3,
            "forward with adapters must match forward after merge (drift {diff})"
        );
    }

    #[test]
    fn freeze_base_keeps_only_lora_trainable() {
        type B = burn::backend::Autodiff<burn::backend::Flex<f32, i32>>;
        let device = burn::backend::flex::FlexDevice;
        let config = LlmModelConfig::tiny();
        let mut model = LlmModel::<B>::new(&config, &device);
        inject_lora(
            &mut model,
            &LoraTrainConfig {
                rank: 2,
                alpha: 4.0,
                dropout: 0.1,
            },
            &device,
        );
        let model = freeze_base(model);

        for layer in &model.model.layers {
            let proj = &layer.self_attn.q_proj;
            assert!(
                !proj.weight.val().is_require_grad(),
                "base q_proj weight must be frozen"
            );
            assert!(
                proj.lora_A.as_ref().unwrap().val().is_require_grad(),
                "lora_A must stay trainable"
            );
            assert!(
                proj.lora_B.as_ref().unwrap().val().is_require_grad(),
                "lora_B must stay trainable"
            );
        }
    }

    #[test]
    fn lora_disabled_matches_linear() {
        type B = burn::backend::Flex<f32, i32>;
        let device = burn::backend::flex::FlexDevice;
        let config = LlmModelConfig::tiny();
        let model = LlmModel::<B>::new(&config, &device);

        for proj in [
            &model.model.layers[0].self_attn.q_proj,
            &model.model.layers[0].mlp.down_proj,
        ] {
            let [d_in, d_out] = proj.weight.val().dims();
            let input = Tensor::<B, 3>::ones([1, 3, d_in], &device);
            let mut lin = LinearConfig::new(d_in, d_out)
                .with_bias(proj.bias.is_some())
                .with_initializer(burn::module::Initializer::Zeros)
                .init::<B>(&device);
            lin.weight = proj.weight.clone();
            lin.bias = proj.bias.clone();
            let out_lora = proj.forward(input.clone());
            let out_lin = lin.forward(input.clone());
            let diff: f32 = out_lora.sub(out_lin).abs().max().into_scalar();
            assert!(diff < 1e-5, "lora-disabled forward diverged from Linear: {diff}");
        }
    }
}