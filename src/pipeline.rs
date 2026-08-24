//! End-to-end pipeline glue: load a checkpoint, prepare the corpus, run the
//! exact-step fine-tune, and export safetensors + GGUF.
//!
//! [`run_pipeline`] maps the requested [`Precision`] onto compile-time backend
//! instantiations and delegates to [`run_pipeline_typed`], which is generic
//! over the backend. Everything downstream — checkpoint casting, forward/
//! backward math, optimizer state, export — then runs in the selected dtype.

use std::path::{Path, PathBuf};

use anyhow::{Context, Result, bail};
#[cfg(feature = "gpu")]
use half::{bf16, f16};

use crate::config::TransformersConfig;
use crate::data::{TokenizerStore, collect_text_files, tokenize_corpus};
use crate::export::{export_gguf, export_safetensors};
use crate::hf::{HfRepo, classify_download};
use crate::model::ablation::{AblationConfig, apply_ablation};
use crate::model::{LlmModel, LlmModelConfig};
use crate::train::{Precision, TrainConfig, backend_label, train_model};

use burn::backend::Autodiff;
use burn::module::Module;
use burn::tensor::DType;
use burn::tensor::backend::Backend;

/// Inputs for a training run.
#[derive(Debug, Clone)]
pub struct PipelineInputs {
    /// Directory containing the model repo (`config.json`, `*.safetensors`,
    /// `tokenizer.json`).
    pub model_dir: PathBuf,
    /// Directory containing the text corpus.
    pub dataset_dir: PathBuf,
    /// Directory to write exports into.
    pub out_dir: PathBuf,
    /// Training hyper-parameters.
    pub train: TrainConfig,
    /// Optional refusal-direction ablation applied before training.
    pub ablation: Option<AblationConfig>,
}

/// Default download destination for a model repo: `models/<owner>--<name>`.
pub fn default_model_dir(out: &Path, repo: &HfRepo) -> PathBuf {
    out.join("models")
        .join(format!("{}--{}", repo.owner, repo.name))
}

/// Default download destination for a dataset repo: `datasets/<owner>--<name>`.
pub fn default_dataset_dir(out: &Path, repo: &crate::data::HfDataset) -> PathBuf {
    out.join("datasets")
        .join(format!("{}--{}", repo.owner, repo.name))
}

/// Load a model checkpoint from `model_dir`, casting all float tensors to
/// `load_dtype` (which must match the model's parameter dtype).
pub fn load_model_from_dir<B>(
    model_dir: &Path,
    load_dtype: DType,
) -> Result<(LlmModel<B>, LlmModelConfig, TokenizerStore)>
where
    B: Backend,
{
    let config_path = model_dir.join("config.json");
    if !config_path.exists() {
        bail!(
            "`config.json` not found in `{}` (run the download step first)",
            model_dir.display()
        );
    }
    let transformers = TransformersConfig::from_path(&config_path)
        .with_context(|| format!("failed to parse `{}`", config_path.display()))?;
    let config = LlmModelConfig::from_transformers(&transformers);

    let tokenizer_path = model_dir.join("tokenizer.json");
    if !tokenizer_path.exists() {
        bail!("`tokenizer.json` not found in `{}`", model_dir.display());
    }
    let tokenizer = TokenizerStore::from_file(&tokenizer_path)?;

    let shards = classify_download(model_dir)?.safetensors;
    if shards.is_empty() {
        bail!(
            "no `.safetensors` weights found in `{}`",
            model_dir.display()
        );
    }
    let shards_refs: Vec<&Path> = shards.iter().map(PathBuf::as_path).collect();

    let mut model = LlmModel::<B>::new(&config, &Default::default());
    crate::model::load::load_from_safetensors(&mut model, &shards_refs, load_dtype)?;

    Ok((model, config, tokenizer))
}

/// Copy the files that `export --model-dir <out_dir>` expects (`config.json`,
/// `tokenizer.json`, plus the optional `tokenizer_config.json`) from the base
/// model directory into the training output directory, making a trained
/// checkpoint self-contained and re-exportable.
fn copy_export_inputs(model_dir: &Path, out_dir: &Path) -> Result<()> {
    if let (Ok(from), Ok(to)) = (model_dir.canonicalize(), out_dir.canonicalize())
        && from == to
    {
        return Ok(());
    }

    let mut copied = Vec::new();
    for name in ["config.json", "tokenizer.json"] {
        let source = model_dir.join(name);
        if !source.exists() {
            bail!("`{name}` not found in `{}`", model_dir.display());
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
    let tokenizer_config = model_dir.join("tokenizer_config.json");
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
        "copied {} into `{}` (output can be re-exported with `export --model-dir`)",
        copied.join(", "),
        out_dir.display()
    );
    Ok(())
}

/// Run the full fine-tune-and-export pipeline on the GPU backend selected by
/// the requested precision.
///
/// The GPU stack can hard-crash (SIGSEGV inside the driver) while compiling
/// half-precision kernels, so callers must validate dtypes with
/// [`crate::probe::spawn_probe`] first and degrade along the ladder of
/// [`crate::probe::BackendPlan`]: requested precision here, then
/// [`run_pipeline_on_gpu_f32`], then [`run_pipeline_on_cpu`].
#[cfg(feature = "gpu")]
pub fn run_pipeline(inputs: &PipelineInputs) -> Result<()> {
    let backend = backend_label();
    match inputs.train.precision {
        Precision::F32 => run_pipeline_typed::<burn::backend::Wgpu<f32, i32>>(
            inputs,
            Precision::F32.safetensors_dtype(),
            backend,
        ),
        Precision::Bf16 => run_pipeline_typed::<burn::backend::Wgpu<bf16, i32>>(
            inputs,
            Precision::Bf16.safetensors_dtype(),
            backend,
        ),
        Precision::F16 => run_pipeline_typed::<burn::backend::Wgpu<f16, i32>>(
            inputs,
            Precision::F16.safetensors_dtype(),
            backend,
        ),
    }
}

/// Run the full fine-tune-and-export pipeline on the GPU backend with f32
/// compute, regardless of the requested precision.
///
/// Middle rung of the fallback ladder: when a buggy driver rejects half
/// precision but accepts f32 (see [`crate::probe`]), training still runs on
/// the GPU while checkpoint ingest and export follow `train.precision`.
#[cfg(feature = "gpu")]
pub fn run_pipeline_on_gpu_f32(inputs: &PipelineInputs) -> Result<()> {
    run_pipeline_typed::<burn::backend::Wgpu<f32, i32>>(inputs, DType::F32, backend_label())
}

/// Run the pipeline on the CPU backend.
///
/// Burn's Flex backend computes in f32 only (`Flex` is not generic over the
/// element type), so half-precision requests become *mixed* precision here:
/// checkpoints are cast to the requested dtype while loading, all training
/// math runs in f32 (fp32 master weights are also the numerically safer
/// recipe for AdamW), and safetensors export casts back down to the requested
/// dtype. This is both the path for `--no-default-features --features flex`
/// builds and the last resort when no dtype survives the GPU probe.
pub fn run_pipeline_on_cpu(inputs: &PipelineInputs) -> Result<()> {
    if inputs.train.precision != Precision::F32 {
        log::warn!(
            "computing in f32 on CPU; the requested {} applies to checkpoint \
             load and export only",
            inputs.train.precision
        );
    }
    // The Flex backend is compiled for f32 parameters, so checkpoints must
    // be cast UP to f32 on ingest regardless of their storage dtype.
    run_pipeline_typed::<burn::backend::Flex<f32, i32>>(inputs, DType::F32, "Flex/CPU")
}

// ---------- memory pre-flight ----------

/// Rough peak-memory footprint of a fine-tuning run, in bytes.
#[derive(Debug, Clone, Copy)]
struct TrainingMemory {
    /// Model parameters.
    weights: u64,
    /// One gradient per parameter, held for the optimizer step.
    gradients: u64,
    /// AdamW keeps two moment buffers (`exp_avg`, `exp_avg_sq`) per parameter.
    optimizer: u64,
    /// Live activations of forward + backward (heuristic upper bound).
    activations: u64,
}

impl TrainingMemory {
    fn total(&self) -> u64 {
        self.weights + self.gradients + self.optimizer + self.activations
    }
}

/// Estimate the peak memory of a fine-tune: weights + gradients + AdamW state
/// (4x the parameter bytes) plus a heuristic activation bound — roughly 16
/// hidden-sized and 8 intermediate-sized tensors per decoder layer (forward
/// values kept alive for backward), plus ~3 vocabulary-sized logits/softmax
/// buffers. Deliberately conservative: better to refuse an oversized run than
/// to let it die mid-autotune inside wgpu.
fn estimate_training_memory(
    params: u64,
    elem_size: usize,
    batch_size: usize,
    seq_len: usize,
    config: &LlmModelConfig,
) -> TrainingMemory {
    let elem = elem_size as u64;
    let p = params.max(1);
    let tokens = batch_size.max(1).saturating_mul(seq_len.max(1)) as u64;
    let per_layer =
        16 * config.d_model as u64 + 8 * config.intermediate_size as u64;
    let logits = 3 * config.vocab_size.max(1) as u64;
    TrainingMemory {
        weights: p * elem,
        gradients: p * elem,
        optimizer: 2 * p * elem,
        activations: tokens * elem * (per_layer * config.n_layers as u64 + logits),
    }
}

/// Extract `MemAvailable:` (converted to bytes) from `/proc/meminfo` content.
fn parse_mem_available(meminfo: &str) -> Option<u64> {
    for line in meminfo.lines() {
        if let Some(rest) = line.strip_prefix("MemAvailable:") {
            let kb: u64 = rest.trim().split_whitespace().next()?.parse().ok()?;
            return Some(kb.saturating_mul(1024));
        }
    }
    None
}

/// Memory the kernel estimates could be handed out without swapping
/// (`/proc/meminfo`'s `MemAvailable`). `None` off Linux or when unreadable.
fn available_memory_bytes() -> Option<u64> {
    parse_mem_available(&std::fs::read_to_string("/proc/meminfo").ok()?)
}

/// Format bytes as Gibibytes with one decimal.
fn gib(bytes: u64) -> String {
    format!("{:.1} GiB", bytes as f64 / (1u64 << 30) as f64)
}

/// Refuse to start a fine-tune whose estimated memory footprint exceeds what
/// the machine can currently provide. An oversized run otherwise dies deep
/// inside the GPU stack with cryptic panics once a buffer allocation fails.
///
/// Skipped silently when the model config cannot be read (the loader reports
/// that with its own error) or `/proc/meminfo` is unavailable (non-Linux).
fn check_memory_fit(model_dir: &Path, train: &TrainConfig) -> Result<()> {
    let Some(available) = available_memory_bytes() else {
        log::debug!("memory pre-flight skipped: /proc/meminfo unavailable");
        return Ok(());
    };
    match check_memory_against(model_dir, train, available) {
        Ok(()) => Ok(()),
        Err(err) => Err(err),
    }
}

/// [`check_memory_fit`] against a caller-supplied availability figure
/// (testable; also the actual decision body).
fn check_memory_against(model_dir: &Path, train: &TrainConfig, available: u64) -> Result<()> {
    let Ok(transformers) = TransformersConfig::from_path(model_dir.join("config.json")) else {
        return Ok(());
    };
    let config = LlmModelConfig::from_transformers(&transformers);
    let params = config.param_count();
    let estimate = |elem: usize| {
        estimate_training_memory(params, elem, train.batch_size, train.seq_len, &config)
    };
    let est = estimate(train.precision.elem_size());

    log::info!(
        "memory pre-flight: need {} (weights {} + gradients {} + AdamW {} + \
         activations {}; {} parameters), {} available",
        gib(est.total()),
        gib(est.weights),
        gib(est.gradients),
        gib(est.optimizer),
        gib(est.activations),
        params,
        gib(available),
    );

    if est.total() <= available {
        return Ok(());
    }

    let remedy = if train.precision == Precision::F32 {
        format!(
            "Try `--precision bf16` (estimated {}), or lower `--batch-size` / `--seq-len`",
            gib(estimate(Precision::Bf16.elem_size()).total())
        )
    } else {
        "Try lowering `--batch-size` / `--seq-len`".to_string()
    };
    bail!(
        "not enough memory to fine-tune `{}` in {}: estimated {} \
         (weights {} + gradients {} + AdamW state {} + activations {} at \
         batch {} x seq_len {}), but only {} is currently available. {remedy}",
        model_dir.display(),
        train.precision,
        gib(est.total()),
        gib(est.weights),
        gib(est.gradients),
        gib(est.optimizer),
        gib(est.activations),
        train.batch_size,
        train.seq_len,
        gib(available),
    )
}

/// Output file name component for a downloaded repo directory: HF snapshot
/// dirs are named `<owner>--<name>`, so keep only `<name>`; anything that is
/// not filename-safe becomes `-`.
fn dir_slug(dir: &Path) -> String {
    let raw = dir
        .file_name()
        .map(|name| name.to_string_lossy().into_owned())
        .unwrap_or_default();
    let name = match raw.rsplit_once("--") {
        // `<owner>--<name>`: keep only the repo name when there is one.
        Some((_, after)) if !after.is_empty() => after,
        _ => raw.as_str(),
    };
    let slug: String = name
        .chars()
        .map(|c| {
            if c.is_ascii_alphanumeric() || matches!(c, '.' | '_' | '-') {
                c
            } else {
                '-'
            }
        })
        .collect();
    let trimmed = slug.trim_matches('-');
    if trimmed.is_empty() {
        "model".to_string()
    } else {
        trimmed.to_string()
    }
}

/// Combined output stem: `<model-name>_<dataset-name>`.
fn output_slug(model_dir: &Path, dataset_dir: &Path) -> String {
    format!("{}_{}", dir_slug(model_dir), dir_slug(dataset_dir))
}

/// Typed pipeline body: every stage runs on backend `B`. `load_dtype` is the
/// parameter dtype of `B` — checkpoints are cast to it on ingest, while
/// safetensors export follows `train.precision` (they differ on the CPU
/// fallback path).
fn run_pipeline_typed<B>(inputs: &PipelineInputs, load_dtype: DType, backend: &str) -> Result<()>
where
    B: Backend,
    Autodiff<B>: burn::tensor::backend::AutodiffBackend<InnerBackend = B>,
    f32: From<<Autodiff<B> as burn::tensor::backend::BackendTypes>::FloatElem>,
{
    // Fail fast (before allocating weights) when this run cannot fit.
    check_memory_fit(&inputs.model_dir, &inputs.train)?;

    log::info!(
        "loading model from `{}` as {}",
        inputs.model_dir.display(),
        inputs.train.precision
    );
    let (mut model, config, mut tokenizer) =
        load_model_from_dir::<B>(&inputs.model_dir, load_dtype)?;

    if config.vocab_size == 0 {
        bail!("model config has zero vocabulary size");
    }
    if inputs.train.seq_len == 0 {
        bail!("seq-len must be at least 1");
    }
    tokenizer.set_seq_len(inputs.train.seq_len);

    if let Some(cfg) = &inputs.ablation {
        log::info!("applying refusal-direction ablation");
        apply_ablation(&mut model, &tokenizer, cfg, config.max_seq_len)?;
    }

    log::info!("tokenizing corpus in `{}`", inputs.dataset_dir.display());
    let files = collect_text_files(&inputs.dataset_dir, &["txt", "text", "md", "jsonl"]);
    if files.is_empty() {
        bail!(
            "no text files (.txt/.text/.md/.jsonl) found in `{}`",
            inputs.dataset_dir.display()
        );
    }
    // Streaming: one file resident at a time; windows land in one flat arena.
    let (windows, total_tokens) =
        tokenize_corpus(&tokenizer, &files, inputs.train.seq_len, tokenizer.pad_id)?;
    if windows.len() < inputs.train.batch_size {
        bail!(
            "corpus produced {} windows ({} tokens), but `batch-size` is {}; shorten `--seq-len` or add more text",
            windows.len(),
            total_tokens,
            inputs.train.batch_size
        );
    }
    log::info!(
        "corpus: {} tokens -> {} windows (seq_len {})",
        total_tokens,
        windows.len(),
        inputs.train.seq_len
    );

    log::info!(
        "training for {} steps (batch {}, lr {}, wd {}, precision {}) on {}",
        inputs.train.steps,
        inputs.train.batch_size,
        inputs.train.lr,
        inputs.train.weight_decay,
        inputs.train.precision,
        backend
    );
    // Label the dashboard header with the repo names behind the input
    // directories, matching how the trained outputs are named.
    let mut train_cfg = inputs.train.clone();
    train_cfg.run_info = crate::ui::RunInfo {
        model: dir_slug(&inputs.model_dir),
        dataset: dir_slug(&inputs.dataset_dir),
    };
    let ad_model: LlmModel<Autodiff<B>> = model.train();
    let trained = train_model(ad_model, &windows, tokenizer.pad_id, &train_cfg);

    // Outputs are grouped under `<model-name>_<dataset-name>` so concurrent
    // fine-tunes of different model/dataset pairs never collide:
    // `<out>/<slug>/model.safetensors` + `<out>/<slug>.gguf`.
    let slug = output_slug(&inputs.model_dir, &inputs.dataset_dir);
    let checkpoint_dir = inputs.out_dir.join(&slug);
    std::fs::create_dir_all(&checkpoint_dir)
        .with_context(|| format!("failed to create `{}`", checkpoint_dir.display()))?;
    copy_export_inputs(&inputs.model_dir, &checkpoint_dir)?;
    let safetensors_path = checkpoint_dir.join("model.safetensors");
    export_safetensors(&trained, &safetensors_path, inputs.train.precision)?;

    let gguf_path = inputs.out_dir.join(format!("{slug}.gguf"));
    export_gguf(&trained, &config, &tokenizer, &gguf_path, &slug)?;

    log::info!("wrote checkpoint to `{}`", checkpoint_dir.display());
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use burn_store::ModuleSnapshot;

    // ---------- output naming ----------

    #[test]
    fn slug_keeps_only_repo_name_from_hf_layout() {
        let base = tempfile::tempdir().unwrap();
        let model_dir = base.path().join("models/Qwen--Qwen2.5-0.5B");
        let dataset_dir = base.path().join("datasets/smcleod--golang-coder");
        std::fs::create_dir_all(&model_dir).unwrap();
        std::fs::create_dir_all(&dataset_dir).unwrap();

        assert_eq!(
            output_slug(&model_dir, &dataset_dir),
            "Qwen2.5-0.5B_golang-coder"
        );
    }

    #[test]
    fn slug_handles_plain_and_unsafe_names() {
        let base = tempfile::tempdir().unwrap();
        for (raw, want) in [
            ("plain", "plain"),
            ("My Model@2!", "My-Model-2"),
            ("--weird--", "weird"),
        ] {
            let dir = base.path().join(raw);
            std::fs::create_dir_all(&dir).unwrap();
            assert_eq!(dir_slug(&dir), want, "wrong slug for `{raw}`");
        }
        // A name made entirely of stripped characters falls back to a stable
        // placeholder instead of producing an empty path component.
        let dashes = base.path().join("---");
        std::fs::create_dir_all(&dashes).unwrap();
        assert_eq!(dir_slug(&dashes), "model");
    }

    // ---------- precision semantics of the CPU path ----------

    /// The CPU fallback promises: *checkpoint* dtype is honored even though
    /// compute is f32. Round-trip a tiny model through a BF16 safetensors
    /// export and an f32-targeted load (the CPU fallback path), then require
    /// the weights to come back within bf16 rounding error (~2^-8 relative).
    #[test]
    fn bf16_checkpoint_round_trip_stays_within_bf16_epsilon() {
        use crate::model::load::load_from_safetensors;
        use burn::tensor::DType;

        type B = burn::backend::Flex<f32, i32>;
        let config = LlmModelConfig::tiny();
        let device = Default::default();
        let original = LlmModel::<B>::new(&config, &device);

        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("bf16.safetensors");
        export_safetensors(&original, &path, Precision::Bf16).unwrap();

        // The file really is stored as BF16, not silently upgraded to F32.
        let header = read_safetensors_header(&path);
        let dtypes: Vec<&str> = header
            .iter()
            .filter_map(|(k, v)| v.get("dtype").and_then(|d| d.as_str()).map(|d| (k, d)))
            .map(|(_, d)| d)
            .collect();
        assert!(!dtypes.is_empty(), "no tensors in exported header");
        assert!(
            dtypes.iter().all(|d| *d == "BF16"),
            "expected every tensor to be BF16, got {dtypes:?}"
        );

        // The CPU model is compiled for f32 parameters: loading with a BF16
        // target would violate the store's dtype contract, so ingest casts
        // up to F32 while the file itself stays BF16.
        let mut reloaded = LlmModel::<B>::new(&config, &device);
        load_from_safetensors(&mut reloaded, &[&path], DType::F32).unwrap();

        let max_diff = max_abs_weight_diff(&original, &reloaded);
        assert!(
            max_diff < 0.01,
            "bf16 round trip drifted too far: {max_diff}"
        );
    }

    /// Read the JSON header of a safetensors file as `(key, object)` pairs.
    fn read_safetensors_header(path: &std::path::Path) -> Vec<(String, serde_json::Value)> {
        use std::io::Read;
        let mut file = std::fs::File::open(path).unwrap();
        let mut len_bytes = [0u8; 8];
        file.read_exact(&mut len_bytes).unwrap();
        let header_len = u64::from_le_bytes(len_bytes)
            .try_into()
            .expect("header length exceeds usize");
        let mut header_json = vec![0u8; header_len];
        file.read_exact(&mut header_json).unwrap();
        let parsed: serde_json::Value = serde_json::from_slice(&header_json).unwrap();
        parsed
            .as_object()
            .expect("safetensors header must be an object")
            .iter()
            .filter(|(k, _)| k.as_str() != "__metadata__")
            .map(|(k, v)| (k.clone(), v.clone()))
            .collect()
    }

    /// Largest absolute difference across all float parameters.
    fn max_abs_weight_diff<B>(a: &LlmModel<B>, b: &LlmModel<B>) -> f64
    where
        B: Backend,
    {
        let sa = a.collect(None, None, false);
        let sb = b.collect(None, None, false);
        let mut max = 0.0f64;
        for (x, y) in sa.iter().zip(sb.iter()) {
            assert_eq!(x.full_path(), y.full_path(), "snapshot order mismatch");
            if !matches!(
                x.dtype,
                burn::tensor::DType::F32 | burn::tensor::DType::Flex32
            ) {
                continue;
            }
            let va = x.to_data().unwrap().to_vec::<f32>().unwrap();
            let vb = y.to_data().unwrap().to_vec::<f32>().unwrap();
            for (p, q) in va.into_iter().zip(vb) {
                max = max.max((p - q).abs() as f64);
            }
        }
        max
    }

    #[test]
    fn makes_output_self_contained_for_reexport() {
        let base = tempfile::tempdir().unwrap();
        let out = tempfile::tempdir().unwrap();
        std::fs::write(base.path().join("config.json"), "{}").unwrap();
        std::fs::write(base.path().join("tokenizer.json"), "{}").unwrap();
        std::fs::write(base.path().join("tokenizer_config.json"), "{\"a\":1}").unwrap();

        copy_export_inputs(base.path(), out.path()).unwrap();

        for (name, want) in [
            ("config.json", "{}"),
            ("tokenizer.json", "{}"),
            ("tokenizer_config.json", "{\"a\":1}"),
        ] {
            let got = std::fs::read_to_string(out.path().join(name)).unwrap();
            assert_eq!(got, want, "wrong content for `{name}`");
        }

        // A base dir without the optional tokenizer_config.json still works.
        let bare = tempfile::tempdir().unwrap();
        std::fs::write(bare.path().join("config.json"), "{}").unwrap();
        std::fs::write(bare.path().join("tokenizer.json"), "{}").unwrap();
        let bare_out = tempfile::tempdir().unwrap();
        copy_export_inputs(bare.path(), bare_out.path()).unwrap();
        assert!(!bare_out.path().join("tokenizer_config.json").exists());

        // Missing required files must fail loudly.
        let empty = tempfile::tempdir().unwrap();
        assert!(copy_export_inputs(empty.path(), out.path()).is_err());

        // Exporting into the model directory itself is a no-op, not a
        // self-copy error.
        copy_export_inputs(base.path(), base.path()).unwrap();
    }
}
