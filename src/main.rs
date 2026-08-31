use std::path::{Path, PathBuf};

use anyhow::Context;
use clap::{Parser, Subcommand};
#[cfg(feature = "gpu")]
use half::{bf16, f16};
use llm_burner::data::HfDataset;
use llm_burner::hf::HfRepo;
use llm_burner::pipeline::{PipelineInputs, default_dataset_dir, default_model_dir};
use llm_burner::probe::DeviceChoice;
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

        /// Training device: `auto` probes the GPU and degrades (requested
        /// precision → f32 compute → CPU), `cpu` forces the CPU backend and
        /// skips probing, `gpu` fails instead of falling back to CPU.
        #[arg(long, default_value_t = DeviceChoice::Auto)]
        device: DeviceChoice,

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

    /// Validate that the GPU driver can compile and run kernels in a given
    /// precision (exit 0) or die trying (any other exit). Used internally as
    /// the child process of a pre-flight probe before half-precision runs.
    #[command(hide = true)]
    GpuProbe {
        /// Precision to validate on the GPU backend.
        #[arg(long)]
        precision: Precision,
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

        /// Device backend: `auto` uses GPU if available, `cpu` forces the
        /// CPU backend.
        #[arg(long, default_value_t = DeviceChoice::Auto)]
        device: DeviceChoice,
    },

    /// Generate text from a loaded checkpoint.
    Generate {
        /// Model directory containing config.json, tokenizer.json, and .safetensors.
        #[arg(long)]
        model_dir: PathBuf,

        /// Prompt to generate from.
        #[arg(long)]
        prompt: String,

        /// Maximum number of new tokens to emit.
        #[arg(long, default_value_t = 128)]
        max_tokens: usize,

        /// Temperature for sampling; `0.0` disables sampling and uses greedy decoding.
        #[arg(long, default_value_t = 0.7)]
        temperature: f64,

        /// Top-k filtering: keep only the top `k` tokens.
        #[arg(long)]
        top_k: Option<usize>,

        /// Top-p (nucleus) filtering: keep the smallest set of tokens that sums to `p`.
        #[arg(long)]
        top_p: Option<f64>,

        /// Weight dtype for the in-memory model: f32, bf16, or f16. On GPU
        /// builds bf16/f16 halve memory and load bf16/f16 checkpoints without
        /// conversion; on CPU builds Flex computes in f32 and the flag only
        /// changes the load-conversion target.
        #[arg(long, default_value_t = Precision::F32)]
        precision: Precision,

        /// Device backend: `auto` uses GPU if available, `cpu` forces the CPU backend.
        #[arg(long, default_value_t = DeviceChoice::Auto)]
        device: DeviceChoice,
    },

    /// Chat with a quantized GGUF model (CPU-only inference).
    Chat {
        /// Directory containing the exported `model.gguf`, `tokenizer.json`,
        /// and `tokenizer_config.json` (use `--gguf`/`--tokenizer` to point at
        /// specific files elsewhere).
        #[arg(long, default_value = "artifacts/trained")]
        model_dir: PathBuf,

        /// Explicit GGUF file (defaults to `<model_dir>/model.gguf`).
        #[arg(long)]
        gguf: Option<PathBuf>,

        /// Explicit tokenizer file (defaults to `<model_dir>/tokenizer.json`).
        #[arg(long)]
        tokenizer: Option<PathBuf>,

        /// Prompt to answer once and exit; when absent the interactive REPL runs.
        #[arg(long)]
        prompt: Option<String>,

        /// Temperature for sampling; `0.0` uses greedy decoding.
        #[arg(long, default_value_t = 0.7)]
        temperature: f64,

        /// Top-k filtering: keep only the top `k` tokens.
        #[arg(long)]
        top_k: Option<usize>,

        /// Top-p (nucleus) filtering: keep the smallest set of tokens that sums to `p`.
        #[arg(long)]
        top_p: Option<f64>,

        /// Maximum number of tokens per assistant reply.
        #[arg(long, default_value_t = 512)]
        max_tokens: usize,
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
            device,
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
            let _ = std::fs::remove_file(&log_path);
            let log_file = std::fs::OpenOptions::new()
                .create(true)
                .append(true)
                .open(&log_path)
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
                    run_info: Default::default(),
                },
                ablation: ablate_refusal.then_some(llm_burner::model::ablation::AblationConfig {
                    direction_layer: refusal_layer,
                    scale: ablate_scale,
                    harmful_file,
                    harmless_file,
                }),
                // Only an explicit `--device cpu` downgrades the memory
                // pre-flight to a warning; automatic fallbacks stay strict.
                memory_fit_enforced: device != DeviceChoice::Cpu,
            };

            // Pre-flight: half-precision kernels can segfault buggy GPU
            // drivers (Mesa RADV on some AMD iGPUs), which is unrecoverable
            // in-process. Validate dtypes in throwaway children and degrade
            // along the ladder: requested precision → f32 compute on the
            // GPU → CPU training. `--device cpu` skips probing entirely;
            // `--device gpu` refuses the CPU rung.
            #[cfg(feature = "gpu")]
            {
                use llm_burner::probe::{BackendPlan, resolve_backend_plan};
                let plan = match device {
                    DeviceChoice::Cpu => {
                        log::info!("backend: Flex/CPU (`--device cpu`; skipping the GPU probe)");
                        BackendPlan::Cpu
                    }
                    _ => {
                        let exe = std::env::current_exe()
                            .context("cannot locate own executable for the GPU probe")?;
                        let probe_exe = exe.clone();
                        let (plan, notes) = resolve_backend_plan(precision, |p| {
                            llm_burner::probe::spawn_probe(
                                &probe_exe,
                                p,
                                llm_burner::probe::PROBE_TIMEOUT,
                            )
                        });
                        // Fallbacks must be visible even when the TUI later
                        // owns the terminal and train.log is not watched.
                        for note in &notes {
                            log::warn!("{note}");
                            println!("{note}");
                        }
                        if plan == BackendPlan::Cpu && device == DeviceChoice::Gpu {
                            anyhow::bail!(
                                "`--device gpu` was requested, but no precision \
                                 survived the GPU probe; rerun without `--device` \
                                 to train on the CPU"
                            );
                        }
                        plan
                    }
                };
                match plan {
                    BackendPlan::RequestedPrecision => {
                        log::info!("backend: {}", llm_burner::train::backend_label());
                        llm_burner::pipeline::run_pipeline(&inputs)?;
                    }
                    BackendPlan::F32Compute => {
                        log::info!(
                            "backend: {} (f32 compute)",
                            llm_burner::train::backend_label()
                        );
                        llm_burner::pipeline::run_pipeline_on_gpu_f32(&inputs)?;
                    }
                    BackendPlan::Cpu => llm_burner::pipeline::run_pipeline_on_cpu(&inputs)?,
                }
            }
            #[cfg(not(feature = "gpu"))]
            {
                if device == DeviceChoice::Gpu {
                    anyhow::bail!(
                        "`--device gpu` requires a GPU build; rebuild with the \
                         default features (this binary only has the CPU backend)"
                    );
                }
                log::info!("backend: {}", llm_burner::train::backend_label());
                llm_burner::pipeline::run_pipeline_on_cpu(&inputs)?;
            }

            println!("outputs in `{}`", inputs.out_dir.display());
        }
        Command::GpuProbe { precision } => {
            init_stderr_logger();
            llm_burner::probe::run_gpu_probe(precision)?;
        }
        Command::Export {
            model_dir,
            output,
            model_name,
            device,
        } => {
            init_stderr_logger();
            let config_path = model_dir.join("config.json");
            if !config_path.exists() {
                anyhow::bail!("`config.json` not found in `{}`", model_dir.display());
            }
            let transformers = llm_burner::config::TransformersConfig::from_path(&config_path)
                .context(format!("failed to parse `{}`", config_path.display()))?;
            let config = llm_burner::model::LlmModelConfig::from_transformers(&transformers);

            let tokenizer_path = model_dir.join("tokenizer.json");
            if !tokenizer_path.exists() {
                anyhow::bail!("`tokenizer.json` not found in `{}`", model_dir.display());
            }
            let tokenizer = llm_burner::data::TokenizerStore::from_file(&tokenizer_path)?;

            let shards = llm_burner::hf::classify_download(&model_dir)?;
            let shards_refs: Vec<&std::path::Path> =
                shards.safetensors.iter().map(PathBuf::as_path).collect();
            if shards_refs.is_empty() {
                anyhow::bail!(
                    "no `.safetensors` weights found in `{}`",
                    model_dir.display()
                );
            }

            let gguf_parent = output.parent().map(|p| p.to_path_buf()).unwrap_or_default();
            std::fs::create_dir_all(&gguf_parent)?;

            #[cfg(feature = "gpu")]
            match device {
                DeviceChoice::Cpu => {
                    let mut model =
                        llm_burner::model::LlmModel::<burn::backend::Flex<f32, i32>>::new_zeroed(
                            &config,
                            &burn::backend::flex::FlexDevice,
                        );
                    llm_burner::model::load::load_from_safetensors(
                        &mut model,
                        &shards_refs,
                        burn::tensor::DType::F32,
                    )?;
                    llm_burner::export::export_gguf(
                        &model,
                        &config,
                        &tokenizer,
                        &output,
                        &model_name,
                    )?;
                }
                _ => {
                    let mut model =
                        llm_burner::model::LlmModel::<llm_burner::train::InferBackend>::new_zeroed(
                            &config,
                            &Default::default(),
                        );
                    llm_burner::model::load::load_from_safetensors(
                        &mut model,
                        &shards_refs,
                        burn::tensor::DType::F32,
                    )?;
                    llm_burner::export::export_gguf(
                        &model,
                        &config,
                        &tokenizer,
                        &output,
                        &model_name,
                    )?;
                }
            }
            #[cfg(not(feature = "gpu"))]
            {
                let mut model =
                    llm_burner::model::LlmModel::<burn::backend::Flex<f32, i32>>::new_zeroed(
                        &config,
                        &burn::backend::flex::FlexDevice,
                    );
                llm_burner::model::load::load_from_safetensors(
                    &mut model,
                    &shards_refs,
                    burn::tensor::DType::F32,
                )?;
                llm_burner::export::export_gguf(&model, &config, &tokenizer, &output, &model_name)?;
            }

            log::info!("exported GGUF to {}", output.display());
        }
        Command::Generate {
            model_dir,
            prompt,
            max_tokens,
            temperature,
            top_k,
            top_p,
            precision,
            device,
        } => {
            init_stderr_logger();
            let gen_cfg = llm_burner::generate::GenerateConfig {
                max_tokens,
                temperature,
                top_k,
                top_p,
                greedy: temperature <= 0.0,
            };
            let output = dispatch_generate(&model_dir, &prompt, &gen_cfg, device, precision)?;
            println!("{output}");
        }
        Command::Merge {
            base_dir,
            lora_dir,
            out,
            scale,
            export_gguf,
            gguf_output,
            model_name,
            precision,
            device,
        } => {
            init_stderr_logger();
            let inputs = llm_burner::lora::MergePipelineInputs {
                base_dir,
                lora_dir,
                out_dir: out,
                scale,
                precision,
                export_gguf,
                gguf_output,
                model_name,
            };
            dispatch_merge(&inputs, device)?;
        }
        Command::Chat {
            model_dir,
            gguf,
            tokenizer,
            prompt,
            temperature,
            top_k,
            top_p,
            max_tokens,
        } => {
            // The REPL owns the terminal; keep log output out of the way by
            // piping it to a sibling `chat.log` like training does.
            let log_path = model_dir.join("chat.log");
            let _ = std::fs::remove_file(&log_path);
            let log_file = std::fs::OpenOptions::new()
                .create(true)
                .append(true)
                .open(&log_path)
                .with_context(|| format!("failed to create `{}`", log_path.display()))?;
            env_logger::Builder::from_env(env_logger::Env::default().default_filter_or("info"))
                .target(env_logger::Target::Pipe(Box::new(log_file)))
                .init();
            log::info!("logging to `{}`", log_path.display());

            let gguf_path = gguf.unwrap_or_else(|| model_dir.join("model.gguf"));
            if !gguf_path.exists() {
                anyhow::bail!("GGUF file not found: `{}`", gguf_path.display());
            }
            let tokenizer_path = tokenizer.unwrap_or_else(|| model_dir.join("tokenizer.json"));
            if !tokenizer_path.exists() {
                anyhow::bail!(
                    "tokenizer not found: `{}` (export it next to `model.gguf`)",
                    tokenizer_path.display()
                );
            }

            let engine = llm_burner::model::gguf::GgufEngine::load(&gguf_path)?;
            let tokenizer = llm_burner::data::TokenizerStore::from_file(&tokenizer_path)?;
            let gen_cfg = llm_burner::generate::GenerateConfig {
                max_tokens,
                temperature,
                top_k,
                top_p,
                greedy: temperature <= 0.0,
            };

            match prompt {
                Some(p) => {
                    let mut chat = llm_burner::chat::GgufChat::new(engine, tokenizer, gen_cfg);
                    let reply = chat.respond(&p)?;
                    println!("{reply}");
                }
                None => {
                    let mut chat = llm_burner::chat::GgufChat::new(engine, tokenizer, gen_cfg);
                    llm_burner::chat::repl(&mut chat)?;
                }
            }
        }
    }
    Ok(())
}

/// Pick the backend for `generate` and run it.
///
/// On GPU builds the requested [`Precision`] selects the model's element type
/// (`Wgpu<f32|bf16|f16>`), so a bf16 checkpoint loads into bf16 weights with no
/// dtype conversion and half the memory. `--device cpu` and non-GPU builds are
/// always `Flex<f32>`: Burn 0.21's Flex backend computes in f32 only, so the
/// `--precision` flag there only picks the load-conversion target.
fn dispatch_generate(
    model_dir: &Path,
    prompt: &str,
    gen_cfg: &llm_burner::generate::GenerateConfig,
    device: DeviceChoice,
    precision: Precision,
) -> anyhow::Result<String> {
    #[cfg(feature = "gpu")]
    {
        match (device, precision) {
            (DeviceChoice::Cpu, p) => {
                if p != Precision::F32 {
                    log::warn!(
                        "computing in f32 on CPU; the requested {} changes only how \
                         the checkpoint is converted before loading",
                        p
                    );
                }
                run_generate::<burn::backend::Flex<f32, i32>>(
                    model_dir,
                    prompt,
                    gen_cfg,
                    &burn::backend::flex::FlexDevice,
                    burn::tensor::DType::F32,
                )
            }
            (_, Precision::Bf16) => {
                let device = Default::default();
                run_generate::<burn::backend::Wgpu<bf16, i32>>(
                    model_dir,
                    prompt,
                    gen_cfg,
                    &device,
                    Precision::Bf16.safetensors_dtype(),
                )
            }
            (_, Precision::F16) => {
                let device = Default::default();
                run_generate::<burn::backend::Wgpu<f16, i32>>(
                    model_dir,
                    prompt,
                    gen_cfg,
                    &device,
                    Precision::F16.safetensors_dtype(),
                )
            }
            (_, Precision::F32) => {
                let device = Default::default();
                run_generate::<burn::backend::Wgpu<f32, i32>>(
                    model_dir,
                    prompt,
                    gen_cfg,
                    &device,
                    Precision::F32.safetensors_dtype(),
                )
            }
        }
    }
    #[cfg(not(feature = "gpu"))]
    {
        if device == DeviceChoice::Gpu {
            anyhow::bail!(
                "`--device gpu` requires a GPU build; rebuild with the \
                 default features (this binary only has the CPU backend)"
            );
        }
        if precision != Precision::F32 {
            log::warn!(
                "Flex (CPU) computes in f32; the requested {} changes only how \
                 the checkpoint is converted before loading",
                precision
            );
        }
        run_generate::<burn::backend::Flex<f32, i32>>(
            model_dir,
            prompt,
            gen_cfg,
            &burn::backend::flex::FlexDevice,
            burn::tensor::DType::F32,
        )
    }
}

/// Pick the backend for `merge` and run it.
fn dispatch_merge(
    inputs: &llm_burner::lora::MergePipelineInputs,
    device: DeviceChoice,
) -> anyhow::Result<llm_burner::lora::MergeSummary> {
    #[cfg(feature = "gpu")]
    {
        match (device, inputs.precision) {
            (DeviceChoice::Cpu, p) => {
                if p != Precision::F32 {
                    log::warn!(
                        "computing in f32 on CPU; the requested {} changes only how \
                         the checkpoint is converted before loading",
                        p
                    );
                }
                llm_burner::lora::run_merge::<burn::backend::Flex<f32, i32>>(
                    inputs,
                    &burn::backend::flex::FlexDevice,
                    burn::tensor::DType::F32,
                )
            }
            (_, Precision::Bf16) => {
                let device = Default::default();
                llm_burner::lora::run_merge::<burn::backend::Wgpu<bf16, i32>>(
                    inputs,
                    &device,
                    Precision::Bf16.safetensors_dtype(),
                )
            }
            (_, Precision::F16) => {
                let device = Default::default();
                llm_burner::lora::run_merge::<burn::backend::Wgpu<f16, i32>>(
                    inputs,
                    &device,
                    Precision::F16.safetensors_dtype(),
                )
            }
            (_, Precision::F32) => {
                let device = Default::default();
                llm_burner::lora::run_merge::<burn::backend::Wgpu<f32, i32>>(
                    inputs,
                    &device,
                    Precision::F32.safetensors_dtype(),
                )
            }
        }
    }
    #[cfg(not(feature = "gpu"))]
    {
        if device == DeviceChoice::Gpu {
            anyhow::bail!(
                "`--device gpu` requires a GPU build; rebuild with the \
                 default features (this binary only has the CPU backend)"
            );
        }
        if inputs.precision != Precision::F32 {
            log::warn!(
                "Flex (CPU) computes in f32; the requested {} changes only how \
                 the checkpoint is converted before loading",
                inputs.precision
            );
        }
        llm_burner::lora::run_merge::<burn::backend::Flex<f32, i32>>(
            inputs,
            &burn::backend::flex::FlexDevice,
            burn::tensor::DType::F32,
        )
    }
}

/// Build, load, and run a checkpoint shell of type `B`.
///
/// The model is zero-initialized (every weight is immediately overwritten by
/// the checkpoint), loaded from all `.safetensors` shards in `model_dir`, and
/// used to autoregressively generate `prompt`'s continuation.
fn run_generate<B: burn::tensor::backend::Backend>(
    model_dir: &Path,
    prompt: &str,
    gen_cfg: &llm_burner::generate::GenerateConfig,
    device: &B::Device,
    load_dtype: burn::tensor::DType,
) -> anyhow::Result<String> {
    let config_path = model_dir.join("config.json");
    if !config_path.exists() {
        anyhow::bail!("`config.json` not found in `{}`", model_dir.display());
    }
    let transformers = llm_burner::config::TransformersConfig::from_path(&config_path)
        .with_context(|| format!("failed to parse `{}`", config_path.display()))?;
    let config = llm_burner::model::LlmModelConfig::from_transformers(&transformers);

    let tokenizer_path = model_dir.join("tokenizer.json");
    if !tokenizer_path.exists() {
        anyhow::bail!("`tokenizer.json` not found in `{}`", model_dir.display());
    }
    let tokenizer = llm_burner::data::TokenizerStore::from_file(&tokenizer_path)?;

    let shards = llm_burner::hf::classify_download(model_dir)?;
    let shards_refs: Vec<&std::path::Path> =
        shards.safetensors.iter().map(PathBuf::as_path).collect();
    if shards_refs.is_empty() {
        anyhow::bail!(
            "no `.safetensors` weights found in `{}`",
            model_dir.display()
        );
    }

    let (stored_dtype, tensor_count) = llm_burner::model::load::checkpoint_dtype(&shards_refs)?;
    log::info!(
        "checkpoint: {tensor_count} tensor(s) across {} shard(s), stored as {stored_dtype:?}",
        shards_refs.len()
    );
    if stored_dtype != load_dtype {
        log::warn!(
            "checkpoint tensors are stored as {stored_dtype:?} but load as \
             {load_dtype:?}; converting on ingest"
        );
    }

    let mut model = llm_burner::model::LlmModel::<B>::new_zeroed(&config, device);
    llm_burner::model::load::load_from_safetensors(&mut model, &shards_refs, load_dtype)?;

    let output =
        llm_burner::generate::generate(&model, &tokenizer, prompt, gen_cfg, config.max_seq_len)?;
    Ok(output)
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
