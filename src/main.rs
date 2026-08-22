use std::path::{Path, PathBuf};

use anyhow::Context;
use clap::{Parser, Subcommand};
use llm_burner::data::HfDataset;
use llm_burner::hf::HfRepo;
use llm_burner::pipeline::{PipelineInputs, default_dataset_dir, default_model_dir, run_pipeline};
use llm_burner::train::{Precision, TrainConfig};

/// A simplified Gemma-family LLM fine-tuner for Burn.
#[derive(Parser, Debug)]
#[command(
    name = "llm-burner",
    about = "download, fine-tune, and export Gemma-style transformers"
)]
struct Cli {
    #[command(subcommand)]
    command: Command,
}

#[derive(Subcommand, Debug)]
enum Command {
    /// Download a model and dataset from the Hugging Face Hub.
    Download {
        /// Model repo id (`owner/name`).
        #[arg(long)]
        model: String,

        /// Text dataset repo id (`owner/name`).
        #[arg(long)]
        dataset: String,

        /// Base output directory (`models/` and `datasets/` are created under it).
        #[arg(long, default_value = "artifacts")]
        out: PathBuf,

        /// Concurrent file downloads.
        #[arg(long, default_value_t = 8)]
        max_workers: usize,
    },

    /// Fine-tune a downloaded model on a corpus and export safetensors + GGUF.
    Train {
        /// Model repo id (`owner/name`) to download if `--model-dir` is absent.
        #[arg(long)]
        model: Option<String>,

        /// Existing model directory (overrides `--model`).
        #[arg(long)]
        model_dir: Option<PathBuf>,

        /// Dataset repo id (`owner/name`) to download if `--dataset-dir` is absent.
        #[arg(long)]
        dataset: Option<String>,

        /// Existing text corpus directory (overrides `--dataset`).
        #[arg(long)]
        dataset_dir: Option<PathBuf>,

        /// Base output directory (`models/` and `datasets/` are created under it;
        /// fine-tuned weights land in `trained/`).
        #[arg(long, default_value = "artifacts")]
        out: PathBuf,

        /// Exact number of optimization steps.
        #[arg(long, default_value_t = 100)]
        steps: usize,

        /// Windows per batch.
        #[arg(long, default_value_t = 8)]
        batch_size: usize,

        /// Token sequence length per window.
        #[arg(long, default_value_t = 128)]
        seq_len: usize,

        /// Learning rate.
        #[arg(long, default_value_t = 3e-4)]
        lr: f64,

        /// AdamW weight decay.
        #[arg(long, default_value_t = 0.1)]
        weight_decay: f64,

        /// Disable the Ratatui progress dashboard (useful for testing and
        /// non-interactive runs); progress goes to the log file instead.
        #[arg(long)]
        no_tui: bool,

        /// Weight/compute precision: f32, bf16, or f16. Applies to checkpoint
        /// loading, training math, optimizer state, and safetensors export.
        #[arg(long, default_value_t = Precision::F32)]
        precision: Precision,

        /// Remove the model's refusal direction before fine-tuning
        /// (abliteration). The change is baked into the exported weights.
        #[arg(long)]
        ablate_refusal: bool,

        /// Decoder layer used to measure the refusal direction
        /// (default: ~2/3 of the network depth).
        #[arg(long)]
        refusal_layer: Option<usize>,

        /// Ablation strength within [0, 1]; 1 fully removes the component.
        #[arg(long, default_value_t = 1.0)]
        ablate_scale: f64,

        /// File of newline-separated harmful probes (overrides built-ins;
        /// blank lines and `#` comments are ignored).
        #[arg(long)]
        harmful_file: Option<PathBuf>,

        /// File of newline-separated harmless probes (overrides built-ins;
        /// blank lines and `#` comments are ignored).
        #[arg(long)]
        harmless_file: Option<PathBuf>,

        /// Concurrent file downloads when fetching the model/dataset.
        #[arg(long, default_value_t = 8)]
        max_workers: usize,
    },

    /// Export a trained model to GGUF format.
    Export {
        /// Model directory containing config.json, tokenizer.json, and .safetensors.
        #[arg(long)]
        model_dir: PathBuf,

        /// Output path for the .gguf file.
        #[arg(long, default_value = "artifacts/trained/model.gguf")]
        output: PathBuf,

        /// Model name string recorded in GGUF metadata.
        #[arg(long, default_value = "llm-burner-finetune")]
        model_name: String,
    },
}

fn main() -> anyhow::Result<()> {
    let cli = Cli::parse();
    match cli.command {
        Command::Download {
            model,
            dataset,
            out,
            max_workers,
        } => {
            init_stderr_logger();
            let repo = HfRepo::parse(&model)?;
            let ds = HfDataset::parse(&dataset)?;

            log::info!("downloading model `{}`", repo.id());
            let mdir = default_model_dir(&out, &repo);
            let md = repo.download_snapshot(
                &mdir,
                &[
                    "config.json".to_string(),
                    "tokenizer.json".to_string(),
                    "tokenizer_config.json".to_string(),
                    "*.safetensors".to_string(),
                ],
                &["*.safetensors.index.json".to_string()],
                max_workers,
            )?;
            println!("model: {}", md.display());

            log::info!("downloading dataset `{}`", ds.id());
            let ddir = default_dataset_dir(&out, &ds);
            let dd = ds.download_snapshot(
                &ddir,
                &[
                    "*.txt".to_string(),
                    "*.text".to_string(),
                    "*.md".to_string(),
                    "*.jsonl".to_string(),
                ],
                &[],
                max_workers,
            )?;
            println!("dataset: {}", dd.display());
        }
        Command::Train {
            model,
            model_dir,
            dataset,
            dataset_dir,
            out,
            steps,
            batch_size,
            seq_len,
            lr,
            weight_decay,
            max_workers,
            no_tui,
            precision,
            ablate_refusal,
            refusal_layer,
            ablate_scale,
            harmful_file,
            harmless_file,
        } => {
            std::fs::create_dir_all(&out)
                .with_context(|| format!("failed to create `{}`", out.display()))?;
            let trained_dir = out.join("trained");
            let log_path = out.join("train.log");
            let log_file = std::fs::File::create(&log_path)
                .with_context(|| format!("failed to create `{}`", log_path.display()))?;
            env_logger::Builder::from_env(env_logger::Env::default().default_filter_or("info"))
                .target(env_logger::Target::Pipe(Box::new(log_file)))
                .init();
            println!("logging to `{}`", log_path.display());

            let (mdir, ddir) =
                resolve_inputs(&out, model, model_dir, dataset, dataset_dir, max_workers)?;

            for path in [&mdir, &ddir] {
                if !path.exists() {
                    anyhow::bail!("path does not exist: `{}`", path.display());
                }
            }

            if ablate_refusal && !(0.0..=1.0).contains(&ablate_scale) {
                anyhow::bail!("`--ablate-scale` must be within [0, 1], got {ablate_scale}");
            }

            log::info!("training precision: {:?}", precision);
            log::info!("backend: {}", llm_burner::train::backend_label());

            let inputs = PipelineInputs {
                model_dir: mdir,
                dataset_dir: ddir,
                out_dir: trained_dir,
                train: TrainConfig {
                    steps,
                    batch_size,
                    seq_len,
                    lr,
                    weight_decay,
                    log_every: (steps / 20).max(1),
                    precision,
                    tui: !no_tui,
                    output_redirect: Some(log_path.clone()),
                },
                ablation: ablate_refusal.then_some(llm_burner::model::ablation::AblationConfig {
                    direction_layer: refusal_layer,
                    scale: ablate_scale,
                    harmful_file,
                    harmless_file,
                }),
                model_name: "llm-burner-finetune".to_string(),
            };
            run_pipeline(&inputs)?;
            println!("outputs in `{}`", inputs.out_dir.display());
        }
        Command::Export {
            model_dir,
            output,
            model_name,
        } => {
            init_stderr_logger();
            // Load config from the model directory
            let config_path = model_dir.join("config.json");
            if !config_path.exists() {
                anyhow::bail!("`config.json` not found in `{}`", model_dir.display());
            }
            let transformers = llm_burner::config::TransformersConfig::from_path(&config_path)
                .context(format!("failed to parse `{}`", config_path.display()))?;
            let config = llm_burner::model::LlmModelConfig::from_transformers(&transformers);

            // Load tokenizer
            let tokenizer_path = model_dir.join("tokenizer.json");
            if !tokenizer_path.exists() {
                anyhow::bail!("`tokenizer.json` not found in `{}`", model_dir.display());
            }
            let tokenizer = llm_burner::data::TokenizerStore::from_file(&tokenizer_path)?;

            // Load model weights
            let shards = llm_burner::hf::classify_download(&model_dir)?;
            let shards_refs: Vec<&std::path::Path> =
                shards.safetensors.iter().map(PathBuf::as_path).collect();
            if shards_refs.is_empty() {
                anyhow::bail!(
                    "no `.safetensors` weights found in `{}`",
                    model_dir.display()
                );
            }

            let mut model = llm_burner::model::LlmModel::<llm_burner::train::InferBackend>::new(
                &config,
                &Default::default(),
            );
            llm_burner::model::load::load_from_safetensors(
                &mut model,
                &shards_refs,
                burn::tensor::DType::F32,
            )?;

            // Export to GGUF
            let gguf_parent = output.parent().map(|p| p.to_path_buf()).unwrap_or_default();
            std::fs::create_dir_all(&gguf_parent)?;
            let gguf_path = output;
            llm_burner::export::export_gguf(&model, &config, &tokenizer, &gguf_path, &model_name)?;

            log::info!("exported GGUF to {}", gguf_path.display());
        }
    }
    Ok(())
}

/// Resolve `(model_dir, dataset_dir)`, downloading the repos when the caller
/// passed ids instead of local directories.
fn resolve_inputs(
    out: &Path,
    model: Option<String>,
    model_dir: Option<PathBuf>,
    dataset: Option<String>,
    dataset_dir: Option<PathBuf>,
    max_workers: usize,
) -> anyhow::Result<(PathBuf, PathBuf)> {
    let resolved_model = match (model, model_dir) {
        (Some(id), None) => {
            let repo = HfRepo::parse(&id)?;
            log::info!("downloading model `{}`", repo.id());
            repo.download_snapshot(
                &default_model_dir(out, &repo),
                &[
                    "config.json".to_string(),
                    "tokenizer.json".to_string(),
                    "tokenizer_config.json".to_string(),
                    "*.safetensors".to_string(),
                ],
                &["*.safetensors.index.json".to_string()],
                max_workers,
            )?
        }
        (None, Some(dir)) => dir,
        (Some(_), Some(_)) => anyhow::bail!("pass only one of `--model` / `--model-dir`"),
        (None, None) => anyhow::bail!("pass either `--model` or `--model-dir`"),
    };

    let resolved_dataset = match (dataset, dataset_dir) {
        (Some(id), None) => {
            let ds = HfDataset::parse(&id)?;
            log::info!("downloading dataset `{}`", ds.id());
            ds.download_snapshot(
                &default_dataset_dir(out, &ds),
                &[
                    "*.txt".to_string(),
                    "*.text".to_string(),
                    "*.md".to_string(),
                    "*.jsonl".to_string(),
                ],
                &[],
                max_workers,
            )?
        }
        (None, Some(dir)) => dir,
        (Some(_), Some(_)) => anyhow::bail!("pass only one of `--dataset` / `--dataset-dir`"),
        (None, None) => anyhow::bail!("pass either `--dataset` or `--dataset-dir`"),
    };

    Ok((resolved_model, resolved_dataset))
}

/// Log to the terminal for commands that do not own it with a TUI.
fn init_stderr_logger() {
    env_logger::Builder::from_env(env_logger::Env::default().default_filter_or("info")).init();
}
