//! LoRA (Low-Rank Adaptation) adapter parsing, weight merging, and export.
//!
//! Merges PEFT-format LoRA adapter weights (e.g. from Hugging Face `peft` /
//! `adapter_model.safetensors`) into a base [`LlmModel`]:
//!
//! ```text
//! W_merged = W_base + scale * (B x A)
//! ```
//!
//! where `A` has shape `[r, in_features]`, `B` has shape `[out_features, r]`,
//! and `scale = lora_alpha / r` (or `lora_alpha / sqrt(r)` with rsLoRA).

use std::collections::{BTreeMap, HashMap};
use std::path::{Path, PathBuf};

use anyhow::{Context, Result, bail};
use burn::module::Param;
use burn::nn::Linear;
use burn::tensor::backend::Backend;
use burn::tensor::{DType, Tensor};
use burn_store::{ModuleStore, SafetensorsStore, TensorSnapshot};

use crate::config::TransformersConfig;
use crate::data::TokenizerStore;
use crate::export::{export_gguf, export_safetensors};
use crate::hf::classify_download;
use crate::model::{LlmModel, LlmModelConfig};
use crate::train::Precision;

/// Deserialized Hugging Face PEFT `adapter_config.json`.
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize, Default, PartialEq)]
pub struct LoraConfig {
    /// Base model name or path recorded during adapter training.
    #[serde(default)]
    pub base_model_name_or_path: Option<String>,

    /// PEFT adapter type (typically `"LORA"`).
    #[serde(default)]
    pub peft_type: Option<String>,

    /// Default LoRA rank `r`.
    #[serde(default)]
    pub r: Option<usize>,

    /// LoRA alpha scaling hyperparameter.
    #[serde(default)]
    pub lora_alpha: Option<f64>,

    /// Dropout probability used during training.
    #[serde(default)]
    pub lora_dropout: Option<f64>,

    /// Target module names (e.g. `["q_proj", "v_proj"]`).
    #[serde(default)]
    pub target_modules: Option<serde_json::Value>,

    /// Per-module rank overrides.
    #[serde(default)]
    pub rank_pattern: Option<HashMap<String, usize>>,

    /// Per-module alpha overrides.
    #[serde(default)]
    pub alpha_pattern: Option<HashMap<String, f64>>,

    /// Rank-Stabilized LoRA: scales by `alpha / sqrt(r)` instead of `alpha / r`.
    #[serde(default)]
    pub use_rslora: Option<bool>,

    /// Bias training setting (`"none"`, `"all"`, `"lora_only"`).
    #[serde(default)]
    pub bias: Option<String>,
}

impl LoraConfig {
    /// Load and parse `adapter_config.json` from a file or adapter directory.
    pub fn from_path(path: impl AsRef<Path>) -> Result<Self> {
        let path = path.as_ref();
        let file_path = if path.is_dir() {
            path.join("adapter_config.json")
        } else {
            path.to_path_buf()
        };

        if !file_path.exists() {
            bail!("LoRA config file `{}` not found", file_path.display());
        }

        let content = std::fs::read_to_string(&file_path)
            .with_context(|| format!("failed to read `{}`", file_path.display()))?;
        let config: Self = serde_json::from_str(&content)
            .with_context(|| format!("failed to parse `{}`", file_path.display()))?;
        Ok(config)
    }

    /// Try loading `adapter_config.json` from `lora_dir` if present, returning
    /// a default config if missing.
    pub fn load_or_default(lora_dir: &Path) -> Self {
        let config_file = if lora_dir.is_dir() {
            lora_dir.join("adapter_config.json")
        } else {
            lora_dir.to_path_buf()
        };

        if config_file.exists() {
            match Self::from_path(&config_file) {
                Ok(cfg) => cfg,
                Err(e) => {
                    log::warn!(
                        "failed to parse `{}` ({e}); falling back to default LoRA config",
                        config_file.display()
                    );
                    Self::default()
                }
            }
        } else {
            Self::default()
        }
    }

    /// Compute the scaling factor for a specific module, allowing a CLI scale override.
    pub fn compute_scale(&self, module_path: &str, rank: usize, cli_scale: Option<f64>) -> f64 {
        if let Some(scale) = cli_scale {
            return scale;
        }

        let r = self
            .rank_pattern
            .as_ref()
            .and_then(|m| {
                m.iter()
                    .find(|(k, _)| module_path.contains(k.as_str()))
                    .map(|(_, v)| *v)
            })
            .or(self.r)
            .unwrap_or(rank);

        let alpha = self
            .alpha_pattern
            .as_ref()
            .and_then(|m| {
                m.iter()
                    .find(|(k, _)| module_path.contains(k.as_str()))
                    .map(|(_, v)| *v)
            })
            .or(self.lora_alpha)
            .unwrap_or(r as f64);

        let r_f64 = (r as f64).max(1.0);
        if self.use_rslora.unwrap_or(false) {
            alpha / r_f64.sqrt()
        } else {
            alpha / r_f64
        }
    }
}

/// Identifies whether a tensor in an adapter is `lora_A`, `lora_B`, or `bias`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum LoraTensorKind {
    LoraA,
    LoraB,
    LoraBias,
}

/// Parse a PEFT tensor name into its canonical base model module path and tensor kind.
///
/// Handles common PEFT prefixes:
/// - `base_model.model.model.layers.0.self_attn.q_proj.lora_A.weight`
/// - `base_model.model.layers.0.self_attn.q_proj.lora_A.weight`
/// - `model.layers.0.self_attn.q_proj.lora_A.weight`
/// - `layers.0.self_attn.q_proj.lora_A.weight`
/// - `base_model.model.lm_head.lora_A.weight`
pub fn parse_lora_tensor_name(raw_name: &str) -> Option<(String, LoraTensorKind)> {
    let (module_prefix_str, kind) = if let Some(p) = raw_name
        .strip_suffix(".lora_A.weight")
        .or_else(|| raw_name.strip_suffix(".lora_A.default.weight"))
        .or_else(|| raw_name.strip_suffix(".lora_a.weight"))
        .or_else(|| raw_name.strip_suffix(".lora_embedding_A"))
    {
        (p.to_string(), LoraTensorKind::LoraA)
    } else if let Some(p) = raw_name
        .strip_suffix(".lora_B.weight")
        .or_else(|| raw_name.strip_suffix(".lora_B.default.weight"))
        .or_else(|| raw_name.strip_suffix(".lora_b.weight"))
        .or_else(|| raw_name.strip_suffix(".lora_embedding_B"))
    {
        (p.to_string(), LoraTensorKind::LoraB)
    } else if let Some(p) = raw_name
        .strip_suffix(".lora_B.bias")
        .or_else(|| raw_name.strip_suffix(".lora_B.default.bias"))
        .or_else(|| raw_name.strip_suffix(".lora_b.bias"))
    {
        (p.to_string(), LoraTensorKind::LoraBias)
    } else {
        return None;
    };

    let mut clean = module_prefix_str;
    // Strip PEFT `base_model.` prefix if present.
    while let Some(stripped) = clean.strip_prefix("base_model.") {
        clean = stripped.to_string();
    }
    // Strip leading `model.` prefixes iteratively (handles `model.model.layers...`).
    // Stop when the remainder is a known module name or starts with `layers.`.
    while let Some(rest) = clean.strip_prefix("model.") {
        if rest.starts_with("layers.")
            || rest == "lm_head"
            || rest == "q_proj"
            || rest == "k_proj"
            || rest == "v_proj"
            || rest == "o_proj"
            || rest == "gate_proj"
            || rest == "up_proj"
            || rest == "down_proj"
            || rest == "embed_tokens"
        {
            clean = rest.to_string();
            break;
        }
        // Otherwise keep stripping another `model.` layer.
        clean = rest.to_string();
    }

    // Prepend `model.` if the cleaned path starts with `layers.`.
    let canonical = if clean.starts_with("layers.") {
        format!("model.{clean}")
    } else {
        clean.to_string()
    };

    Some((canonical, kind))
}

/// Load all tensor snapshots from safetensors files in `lora_dir`.
pub fn load_lora_snapshots(
    lora_dir: &Path,
) -> Result<BTreeMap<String, TensorSnapshot>> {
    let mut snapshots = BTreeMap::new();
    let mut files = Vec::new();

    if lora_dir.is_file() {
        files.push(lora_dir.to_path_buf());
    } else if lora_dir.is_dir() {
        for entry in std::fs::read_dir(lora_dir)
            .with_context(|| format!("failed to read directory `{}`", lora_dir.display()))?
        {
            let entry = entry?;
            let path = entry.path();
            let is_safetensors = path.is_file()
                && path
                    .extension()
                    .and_then(|s| s.to_str())
                    .map(|ext| ext.eq_ignore_ascii_case("safetensors"))
                    .unwrap_or(false);
            if is_safetensors {
                files.push(path);
            }
        }
        files.sort();
    } else {
        bail!("LoRA path `{}` does not exist", lora_dir.display());
    }

    if files.is_empty() {
        bail!(
            "no `.safetensors` adapter files found in `{}`",
            lora_dir.display()
        );
    }

    for file in &files {
        let mut store = SafetensorsStore::from_file(file).allow_partial(true);
        let shard = store
            .get_all_snapshots()
            .with_context(|| format!("failed to parse LoRA weights from `{}`", file.display()))?;
        for (k, v) in shard {
            snapshots.insert(k.clone(), v.clone());
        }
    }

    Ok(snapshots)
}

/// Helper to get a mutable reference to a `Linear` layer by its canonical path.
fn get_linear_mut<'a, B: Backend>(
    model: &'a mut LlmModel<B>,
    module_path: &str,
) -> Option<&'a mut Linear<B>> {
    let parts: Vec<&str> = module_path.split('.').collect();
    match parts.as_slice() {
        ["lm_head"] | ["model", "lm_head"] => model.lm_head.as_mut(),
        ["model", "layers", layer_str, "self_attn", proj] => {
            let layer_idx: usize = layer_str.parse().ok()?;
            let layer = model.model.layers.get_mut(layer_idx)?;
            match *proj {
                "q_proj" => Some(&mut layer.self_attn.q_proj),
                "k_proj" => Some(&mut layer.self_attn.k_proj),
                "v_proj" => Some(&mut layer.self_attn.v_proj),
                "o_proj" => Some(&mut layer.self_attn.o_proj),
                _ => None,
            }
        }
        ["model", "layers", layer_str, "mlp", proj] => {
            let layer_idx: usize = layer_str.parse().ok()?;
            let layer = model.model.layers.get_mut(layer_idx)?;
            match *proj {
                "gate_proj" => Some(&mut layer.mlp.gate_proj),
                "up_proj" => Some(&mut layer.mlp.up_proj),
                "down_proj" => Some(&mut layer.mlp.down_proj),
                _ => None,
            }
        }
        _ => None,
    }
}

/// Paired weights for a single adapted module.
#[derive(Default)]
struct ModuleLoraSnapshots {
    a: Option<TensorSnapshot>,
    b: Option<TensorSnapshot>,
    bias: Option<TensorSnapshot>,
}

/// Summary statistics from a LoRA merge operation.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct MergeSummary {
    /// Number of distinct model modules merged.
    pub modules_merged: usize,
    /// Total number of LoRA parameter elements merged.
    pub params_merged: usize,
}

/// Merge LoRA adapter weights directly into a loaded base [`LlmModel`].
pub fn merge_lora_into_model<B: Backend>(
    model: &mut LlmModel<B>,
    snapshots: &BTreeMap<String, TensorSnapshot>,
    config: &LoraConfig,
    cli_scale: Option<f64>,
    device: &B::Device,
    load_dtype: DType,
) -> Result<MergeSummary> {
    let mut grouped: BTreeMap<String, ModuleLoraSnapshots> = BTreeMap::new();
    for (key, snapshot) in snapshots {
        if let Some((module_path, kind)) = parse_lora_tensor_name(key) {
            let entry = grouped.entry(module_path).or_default();
            match kind {
                LoraTensorKind::LoraA => entry.a = Some(snapshot.clone()),
                LoraTensorKind::LoraB => entry.b = Some(snapshot.clone()),
                LoraTensorKind::LoraBias => entry.bias = Some(snapshot.clone()),
            }
        } else {
            log::warn!("ignoring unrecognized tensor in LoRA adapter: `{key}`");
        }
    }

    if grouped.is_empty() {
        bail!("no recognizable LoRA tensors found in adapter snapshots");
    }

    let mut modules_merged = 0usize;
    let mut params_merged = 0usize;

    for (module_path, weights) in grouped {
        let (Some(snap_a), Some(snap_b)) = (weights.a, weights.b) else {
            bail!("incomplete LoRA pair for module `{module_path}` (missing lora_A or lora_B)");
        };

        let data_a = snap_a
            .to_data()
            .with_context(|| format!("failed to read lora_A data for `{module_path}`"))?;
        let data_b = snap_b
            .to_data()
            .with_context(|| format!("failed to read lora_B data for `{module_path}`"))?;

        let shape_a = data_a.shape.clone();
        let shape_b = data_b.shape.clone();

        if shape_a.len() != 2 || shape_b.len() != 2 {
            bail!(
                "LoRA tensors for `{module_path}` must be 2D; got A: {:?}, B: {:?}",
                shape_a,
                shape_b
            );
        }

        let [r_a, in_features] = [shape_a[0], shape_a[1]];
        let [out_features, r_b] = [shape_b[0], shape_b[1]];

        if r_a != r_b {
            bail!(
                "rank mismatch for `{module_path}`: lora_A rank {r_a} != lora_B rank {r_b}"
            );
        }

        let rank = r_a;
        let scale = config.compute_scale(&module_path, rank, cli_scale);

        // Check if targeting embed_tokens
        if module_path == "model.embed_tokens" || module_path == "embed_tokens" {
            let embed_w = model.model.embed_tokens.weight.val();
            let [vocab_size, d_model] = embed_w.dims();
            if in_features != vocab_size || out_features != d_model {
                bail!(
                    "embed_tokens shape mismatch for `{module_path}`: base has [{}, {}], adapter has [{}, {}]",
                    vocab_size, d_model, in_features, out_features
                );
            }

            let tensor_a =
                Tensor::<B, 2>::from_data(data_a.convert_dtype(load_dtype), device);
            let tensor_b =
                Tensor::<B, 2>::from_data(data_b.convert_dtype(load_dtype), device);

            // [vocab, r] @ [r, d_model] -> [vocab, d_model]
            // In Burn layout, Linear weight is [in, out], but embed is [vocab, d_model].
            // We need A^T @ B^T: [r, in] @ [r, out]... wait let me think about this.
            // A is [r, in_features], B is [out_features, r]
            // A^T is [in_features, r], B^T is [r, out_features]
            // A^T @ B^T = [in_features, r] @ [r, out_features] = [in_features, out_features]
            // But embed weight is [vocab, d_model] = [in_features, out_features] when targeting embed.
            // The delta should be added as: delta = A^T @ B^T * scale
            // Let me verify: in PyTorch, LoRA delta = scale * B @ A where B:[out, r], A:[r, in]
            // So delta has shape [out, in]. In Burn's Linear, weight is [in, out].
            // So delta in Burn layout = (B @ A)^T = A^T @ B^T = [in, out].
            // Yes, that's correct.
            let delta = tensor_a.transpose().matmul(tensor_b.transpose()).mul_scalar(scale as f32);
            let merged = embed_w.add(delta);
            model.model.embed_tokens.weight = Param::from_tensor(merged);

            modules_merged += 1;
            params_merged += in_features * rank + out_features * rank;
            log::info!(
                "merged LoRA into `{module_path}` (rank {rank}, scale {scale:.4})"
            );
            continue;
        }

        let Some(linear) = get_linear_mut(model, &module_path) else {
            log::warn!(
                "target module `{module_path}` not found in base model; skipping"
            );
            continue;
        };

        let [w_in, w_out] = linear.weight.val().dims();
        if w_in != in_features || w_out != out_features {
            bail!(
                "shape mismatch for `{module_path}`: base weight is [{}, {}], adapter is [{}, {}]",
                w_in,
                w_out,
                in_features,
                out_features
            );
        }

        let tensor_a =
            Tensor::<B, 2>::from_data(data_a.convert_dtype(load_dtype), device);
        let tensor_b =
            Tensor::<B, 2>::from_data(data_b.convert_dtype(load_dtype), device);

        // PyTorch layout: A is [r, in], B is [out, r].
        // Burn layout: Linear weight is [in, out].
        // Delta in Burn layout: A^T @ B^T = [in, r] @ [r, out] -> [in, out].
        let delta = tensor_a.transpose().matmul(tensor_b.transpose()).mul_scalar(scale as f32);
        let merged_w = linear.weight.val().add(delta);
        linear.weight = Param::from_tensor(merged_w);

        if let Some(snap_bias) = weights.bias {
            let data_bias = snap_bias
                .to_data()
                .with_context(|| format!("failed to read bias data for `{module_path}`"))?;
            let tensor_bias =
                Tensor::<B, 1>::from_data(data_bias.convert_dtype(load_dtype), device);

            if let Some(bias) = &mut linear.bias {
                let merged_b = bias.val().add(tensor_bias.mul_scalar(scale as f32));
                *bias = Param::from_tensor(merged_b);
            } else {
                linear.bias = Some(Param::from_tensor(tensor_bias.mul_scalar(scale as f32)));
            }
        }

        modules_merged += 1;
        params_merged += in_features * rank + out_features * rank;
        log::info!(
            "merged LoRA into `{module_path}` (rank {rank}, scale {scale:.4})"
        );
    }

    if modules_merged == 0 {
        bail!("no adapter weights could be mapped to base model modules");
    }

    Ok(MergeSummary {
        modules_merged,
        params_merged,
    })
}

/// Options controlling a base model + LoRA merge pipeline.
#[derive(Debug, Clone)]
pub struct MergePipelineInputs {
    /// Base model directory containing config.json, tokenizer.json, and .safetensors.
    pub base_dir: PathBuf,
    /// LoRA adapter directory containing adapter_config.json and adapter .safetensors.
    pub lora_dir: PathBuf,
    /// Destination directory for the merged model checkpoint.
    pub out_dir: PathBuf,
    /// Manual scale factor overriding `adapter_config.json`.
    pub scale: Option<f64>,
    /// Weight precision for load, compute, and safetensors export.
    pub precision: Precision,
    /// Whether to also export the merged model to GGUF.
    pub export_gguf: bool,
    /// Destination path for GGUF output (defaults to `<out_dir>/model.gguf`).
    pub gguf_output: Option<PathBuf>,
    /// Model name string for GGUF metadata.
    pub model_name: String,
}

/// Copy base model configuration and tokenizer files into the output directory.
pub fn copy_base_metadata(base_dir: &Path, out_dir: &Path) -> Result<()> {
    if let (Ok(from), Ok(to)) = (base_dir.canonicalize(), out_dir.canonicalize())
        && from == to
    {
        return Ok(());
    }

    std::fs::create_dir_all(out_dir)
        .with_context(|| format!("failed to create output directory `{}`", out_dir.display()))?;

    let mut copied = Vec::new();
    for name in ["config.json", "tokenizer.json"] {
        let source = base_dir.join(name);
        if !source.exists() {
            bail!("`{name}` not found in `{}`", base_dir.display());
        }
        std::fs::copy(&source, out_dir.join(name)).with_context(|| {
            format!(
                "failed to copy `{}` into `{}`",
                source.display(),
                out_dir.display()
            )
        })?;
        copied.push(name);
    }

    let tokenizer_config = base_dir.join("tokenizer_config.json");
    if tokenizer_config.exists() {
        std::fs::copy(&tokenizer_config, out_dir.join("tokenizer_config.json")).with_context(
            || {
                format!(
                    "failed to copy `{}` into `{}`",
                    tokenizer_config.display(),
                    out_dir.display()
                )
            },
        )?;
        copied.push("tokenizer_config.json");
    }

    log::info!(
        "copied {} into `{}`",
        copied.join(", "),
        out_dir.display()
    );
    Ok(())
}

/// Run the full end-to-end base + LoRA merge pipeline.
pub fn run_merge<B: Backend>(
    inputs: &MergePipelineInputs,
    device: &B::Device,
    load_dtype: DType,
) -> Result<MergeSummary> {
    log::info!(
        "loading base model from `{}`",
        inputs.base_dir.display()
    );

    let config_path = inputs.base_dir.join("config.json");
    if !config_path.exists() {
        bail!(
            "`config.json` not found in `{}`",
            inputs.base_dir.display()
        );
    }
    let transformers = TransformersConfig::from_path(&config_path)
        .with_context(|| format!("failed to parse `{}`", config_path.display()))?;
    let config = LlmModelConfig::from_transformers(&transformers);

    let tokenizer_path = inputs.base_dir.join("tokenizer.json");
    if !tokenizer_path.exists() {
        bail!(
            "`tokenizer.json` not found in `{}`",
            inputs.base_dir.display()
        );
    }
    let tokenizer = TokenizerStore::from_file(&tokenizer_path)?;

    let shards = classify_download(&inputs.base_dir)?.safetensors;
    if shards.is_empty() {
        bail!(
            "no `.safetensors` weights found in `{}`",
            inputs.base_dir.display()
        );
    }
    let shards_refs: Vec<&Path> = shards.iter().map(PathBuf::as_path).collect();

    let mut model = LlmModel::<B>::new_zeroed(&config, device);
    crate::model::load::load_from_safetensors(&mut model, &shards_refs, load_dtype)?;

    log::info!(
        "loading LoRA adapter from `{}`",
        inputs.lora_dir.display()
    );
    let lora_config = LoraConfig::load_or_default(&inputs.lora_dir);
    let lora_snapshots = load_lora_snapshots(&inputs.lora_dir)?;

    let summary = merge_lora_into_model(
        &mut model,
        &lora_snapshots,
        &lora_config,
        inputs.scale,
        device,
        load_dtype,
    )?;

    std::fs::create_dir_all(&inputs.out_dir).with_context(|| {
        format!(
            "failed to create output directory `{}`",
            inputs.out_dir.display()
        )
    })?;

    let merged_safetensors_path = inputs.out_dir.join("model.safetensors");
    export_safetensors(&model, &merged_safetensors_path, inputs.precision)?;
    copy_base_metadata(&inputs.base_dir, &inputs.out_dir)?;

    if inputs.export_gguf {
        let gguf_path = inputs
            .gguf_output
            .clone()
            .unwrap_or_else(|| inputs.out_dir.join("model.gguf"));
        export_gguf(&model, &config, &tokenizer, &gguf_path, &inputs.model_name)?;
    }

    log::info!(
        "successfully merged {} modules ({} params) into `{}`",
        summary.modules_merged,
        summary.params_merged,
        inputs.out_dir.display()
    );

    Ok(summary)
}

#[cfg(test)]
mod tests {
    use super::*;
    use tempfile::tempdir;

    type B = burn::backend::Flex<f32, i32>;

    #[test]
    fn parses_lora_tensor_names() {
        assert_eq!(
            parse_lora_tensor_name("base_model.model.model.layers.0.self_attn.q_proj.lora_A.weight"),
            Some((
                "model.layers.0.self_attn.q_proj".to_string(),
                LoraTensorKind::LoraA
            ))
        );
        assert_eq!(
            parse_lora_tensor_name("base_model.model.layers.1.mlp.gate_proj.lora_B.weight"),
            Some((
                "model.layers.1.mlp.gate_proj".to_string(),
                LoraTensorKind::LoraB
            ))
        );
        assert_eq!(
            parse_lora_tensor_name("layers.2.self_attn.v_proj.lora_A.default.weight"),
            Some((
                "model.layers.2.self_attn.v_proj".to_string(),
                LoraTensorKind::LoraA
            ))
        );
        assert_eq!(
            parse_lora_tensor_name("base_model.model.lm_head.lora_A.weight"),
            Some(("lm_head".to_string(), LoraTensorKind::LoraA))
        );
        assert_eq!(
            parse_lora_tensor_name("base_model.model.model.layers.0.self_attn.q_proj.lora_B.bias"),
            Some((
                "model.layers.0.self_attn.q_proj".to_string(),
                LoraTensorKind::LoraBias
            ))
        );
        assert_eq!(
            parse_lora_tensor_name("model.layers.0.self_attn.q_proj.weight"),
            None
        );
    }

    #[test]
    fn parses_adapter_config_json() {
        let json_str = r#"{
            "base_model_name_or_path": "meta-llama/Llama-3.2-1B",
            "peft_type": "LORA",
            "r": 16,
            "lora_alpha": 32.0,
            "use_rslora": false,
            "target_modules": ["q_proj", "v_proj"]
        }"#;
        let config: LoraConfig = serde_json::from_str(json_str).unwrap();
        assert_eq!(config.r, Some(16));
        assert_eq!(config.lora_alpha, Some(32.0));
        assert_eq!(config.compute_scale("model.layers.0.self_attn.q_proj", 16, None), 2.0);
    }

    #[test]
    fn computes_rslora_scaling() {
        let config = LoraConfig {
            r: Some(16),
            lora_alpha: Some(32.0),
            use_rslora: Some(true),
            ..Default::default()
        };
        assert_eq!(config.compute_scale("module", 16, None), 32.0 / 4.0); // 8.0
    }

    #[test]
    fn merges_lora_weights_accurately() {
        let device = burn::backend::flex::FlexDevice;
        let config = LlmModelConfig::tiny();
        let mut model = LlmModel::<B>::new(&config, &device);

        let in_dim = config.d_model; // 64
        let out_dim = config.n_heads * config.head_dim; // 4 * 16 = 64
        let rank = 4;

        // Base q_proj weights before merge
        let orig_w = model.model.layers[0].self_attn.q_proj.weight.val();

        // Create synthetic LoRA A [r, in] and LoRA B [out, r]
        let a_vec: Vec<f32> = (0..(rank * in_dim)).map(|i| (i as f32) * 0.01).collect();
        let b_vec: Vec<f32> = (0..(out_dim * rank)).map(|i| (i as f32) * 0.02).collect();

        let lora_dir = tempdir().unwrap();

        let a_bytes: Vec<u8> = a_vec.iter().flat_map(|f| f.to_le_bytes()).collect();
        let b_bytes: Vec<u8> = b_vec.iter().flat_map(|f| f.to_le_bytes()).collect();

        let tensors = vec![
            (
                "base_model.model.model.layers.0.self_attn.q_proj.lora_A.weight".to_string(),
                safetensors::tensor::TensorView::new(
                    safetensors::Dtype::F32,
                    vec![rank, in_dim],
                    &a_bytes,
                )
                .unwrap(),
            ),
            (
                "base_model.model.model.layers.0.self_attn.q_proj.lora_B.weight".to_string(),
                safetensors::tensor::TensorView::new(
                    safetensors::Dtype::F32,
                    vec![out_dim, rank],
                    &b_bytes,
                )
                .unwrap(),
            ),
        ];
        let encoded = safetensors::serialize(tensors, None).unwrap();
        std::fs::write(lora_dir.path().join("adapter_model.safetensors"), encoded).unwrap();

        let lora_cfg = LoraConfig {
            r: Some(rank),
            lora_alpha: Some(8.0), // scale = 8.0 / 4 = 2.0
            ..Default::default()
        };

        let summary = merge_lora_into_model(
            &mut model,
            &load_lora_snapshots(lora_dir.path()).unwrap(),
            &lora_cfg,
            None,
            &device,
            DType::F32,
        )
        .unwrap();

        assert_eq!(summary.modules_merged, 1);

        let merged_w = model.model.layers[0].self_attn.q_proj.weight.val();
        let diff = merged_w.sub(orig_w).into_data().to_vec::<f32>().unwrap();

        // Check against expected Delta = 2.0 * (A^T @ B^T)
        // A is [4, 64], B is [64, 4]
        // (A^T @ B^T)[i, j] = sum_k A^T[i, k] * B^T[k, j] = sum_k A[k, i] * B[j, k]
        for i in 0..in_dim {
            for j in 0..out_dim {
                let mut delta_ij = 0.0f32;
                for k in 0..rank {
                    let a_val = a_vec[k * in_dim + i];
                    let b_val = b_vec[j * rank + k];
                    delta_ij += a_val * b_val;
                }
                let expected_delta = 2.0 * delta_ij;
                let actual_delta = diff[i * out_dim + j];
                assert!(
                    (actual_delta - expected_delta).abs() < 1e-4,
                    "mismatch at [{i}, {j}]: expected {expected_delta}, got {actual_delta}"
                );
            }
        }
    }

    #[test]
    fn run_merge_pipeline_end_to_end() {
        use crate::model::load::save_to_safetensors;

        let device = burn::backend::flex::FlexDevice;
        let config = LlmModelConfig::tiny();

        // 1. Setup base model directory
        let base_dir = tempdir().unwrap();
        let base_model = LlmModel::<B>::new(&config, &device);
        save_to_safetensors(&base_model, &base_dir.path().join("model.safetensors")).unwrap();

        // Write config.json
        let config_json = serde_json::json!({
            "model_type": "llama",
            "hidden_size": config.d_model,
            "intermediate_size": config.intermediate_size,
            "num_attention_heads": config.n_heads,
            "num_key_value_heads": config.n_kv_heads,
            "head_dim": config.head_dim,
            "num_hidden_layers": config.n_layers,
            "vocab_size": config.vocab_size,
            "max_position_embeddings": config.max_seq_len,
            "rms_norm_eps": config.rms_eps,
            "rope_theta": config.rope_theta,
            "tie_word_embeddings": config.tie_word_embeddings,
        });
        std::fs::write(
            base_dir.path().join("config.json"),
            serde_json::to_string(&config_json).unwrap(),
        )
        .unwrap();

        // Write dummy tokenizer.json
        let tok_json = serde_json::json!({
            "version": "1.0",
            "added_tokens": [],
            "normalizer": null,
            "pre_tokenizer": null,
            "post_processor": null,
            "decoder": null,
            "model": {
                "type": "BPE",
                "vocab": (0..config.vocab_size).map(|i| (format!("token_{i}"), i as u32)).collect::<HashMap<_, _>>(),
                "merges": []
            }
        });
        std::fs::write(
            base_dir.path().join("tokenizer.json"),
            serde_json::to_string(&tok_json).unwrap(),
        )
        .unwrap();

        // 2. Setup LoRA adapter directory
        let lora_dir = tempdir().unwrap();
        let rank = 4;
        let in_dim = config.d_model;
        let out_dim = config.n_heads * config.head_dim;

        let a_data: Vec<f32> = vec![0.1; rank * in_dim];
        let b_data: Vec<f32> = vec![0.2; out_dim * rank];

        let a_bytes: Vec<u8> = a_data.iter().flat_map(|f| f.to_le_bytes()).collect();
        let b_bytes: Vec<u8> = b_data.iter().flat_map(|f| f.to_le_bytes()).collect();

        let tensors = vec![
            (
                "base_model.model.model.layers.0.self_attn.q_proj.lora_A.weight".to_string(),
                safetensors::tensor::TensorView::new(
                    safetensors::Dtype::F32,
                    vec![rank, in_dim],
                    &a_bytes,
                )
                .unwrap(),
            ),
            (
                "base_model.model.model.layers.0.self_attn.q_proj.lora_B.weight".to_string(),
                safetensors::tensor::TensorView::new(
                    safetensors::Dtype::F32,
                    vec![out_dim, rank],
                    &b_bytes,
                )
                .unwrap(),
            ),
        ];
        let encoded = safetensors::serialize(tensors, None).unwrap();
        std::fs::write(lora_dir.path().join("adapter_model.safetensors"), encoded).unwrap();

        let lora_config_json = serde_json::json!({
            "r": rank,
            "lora_alpha": 4.0,
            "target_modules": ["q_proj"],
            "peft_type": "LORA"
        });
        std::fs::write(lora_dir.path().join("adapter_config.json"), serde_json::to_string(&lora_config_json).unwrap()).unwrap();

        // 3. Run merge pipeline
        let out_dir = tempdir().unwrap();
        let inputs = MergePipelineInputs {
            base_dir: base_dir.path().to_path_buf(),
            lora_dir: lora_dir.path().to_path_buf(),
            out_dir: out_dir.path().to_path_buf(),
            scale: None,
            precision: Precision::F32,
            export_gguf: true,
            gguf_output: None,
            model_name: "test-merged".to_string(),
        };

        let summary = run_merge::<B>(&inputs, &device, DType::F32).unwrap();
        assert_eq!(summary.modules_merged, 1);

        assert!(out_dir.path().join("model.safetensors").exists());
        assert!(out_dir.path().join("config.json").exists());
        assert!(out_dir.path().join("tokenizer.json").exists());
        assert!(out_dir.path().join("model.gguf").exists());
    }
}