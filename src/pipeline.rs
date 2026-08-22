//! End-to-end pipeline glue: load a checkpoint, prepare the corpus, run the
//! exact-step fine-tune, and export safetensors + GGUF.

use std::path::{Path, PathBuf};

use anyhow::{Context, Result, bail};

use crate::config::TransformersConfig;
use crate::data::{TokenizerStore, collect_text_files, read_corpus, sliding_windows};
use crate::export::{export_gguf, export_safetensors};
use crate::hf::{HfRepo, classify_download};
use crate::model::{LlmModel, LlmModelConfig};
use crate::train::{FlexBackend, TrainBackend, TrainConfig, train_model};

use burn::module::Module;

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
    /// Name recorded in GGUF metadata.
    pub model_name: String,
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

/// Load a model checkpoint from `model_dir` into a flex-backed model.
pub fn load_model_from_dir(
    model_dir: &Path,
) -> Result<(LlmModel<FlexBackend>, LlmModelConfig, TokenizerStore)> {
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

    let mut model = LlmModel::<FlexBackend>::new(&config, &Default::default());
    crate::model::load::load_from_safetensors(&mut model, &shards_refs)?;

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

/// Run the full fine-tune-and-export pipeline.
pub fn run_pipeline(inputs: &PipelineInputs) -> Result<()> {
    log::info!("loading model from `{}`", inputs.model_dir.display());
    let (model, config, mut tokenizer) = load_model_from_dir(&inputs.model_dir)?;

    if config.vocab_size == 0 {
        bail!("model config has zero vocabulary size");
    }
    if inputs.train.seq_len == 0 {
        bail!("seq-len must be at least 1");
    }
    tokenizer.set_seq_len(inputs.train.seq_len);

    log::info!("tokenizing corpus in `{}`", inputs.dataset_dir.display());
    let files = collect_text_files(&inputs.dataset_dir, &["txt", "text", "md", "jsonl"]);
    if files.is_empty() {
        bail!(
            "no text files (.txt/.text/.md/.jsonl) found in `{}`",
            inputs.dataset_dir.display()
        );
    }
    let corpus = read_corpus(&files)?;
    let ids = tokenizer.encode_raw(&corpus)?;
    let windows = sliding_windows(
        &ids,
        inputs.train.seq_len,
        inputs.train.seq_len,
        tokenizer.pad_id,
    );
    if windows.len() < inputs.train.batch_size {
        bail!(
            "corpus produced {} windows, but `batch-size` is {}; shorten `--seq-len` or add more text",
            windows.len(),
            inputs.train.batch_size
        );
    }
    log::info!(
        "corpus: {} tokens -> {} windows (seq_len {})",
        ids.len(),
        windows.len(),
        inputs.train.seq_len
    );

    log::info!(
        "training for {} steps (batch {}, lr {}, wd {})",
        inputs.train.steps,
        inputs.train.batch_size,
        inputs.train.lr,
        inputs.train.weight_decay
    );
    let ad_model: LlmModel<TrainBackend> = model.train();
    let trained = train_model(ad_model, &windows, tokenizer.pad_id, &inputs.train);

    std::fs::create_dir_all(&inputs.out_dir)
        .with_context(|| format!("failed to create `{}`", inputs.out_dir.display()))?;
    copy_export_inputs(&inputs.model_dir, &inputs.out_dir)?;
    let safetensors_path = inputs.out_dir.join("model.safetensors");
    export_safetensors(&trained, &safetensors_path)?;

    let gguf_path = inputs.out_dir.join("model.gguf");
    export_gguf(
        &trained,
        &config,
        &tokenizer,
        &gguf_path,
        &inputs.model_name,
    )?;

    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

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
