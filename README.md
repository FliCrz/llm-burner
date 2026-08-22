# llm-burner

`llm-burner` is a Rust CLI designed to download Hugging Face model and text datasets, fine-tune a simplified Gemma/Llama-style transformer using [Burn](https://github.com/tracel-ai/burn), render a live Ratatui progress dashboard, and export checkpoints to both retrainable `.safetensors` and quantized `.gguf` (`Q4_K_M` via `rlx-gguf`).

---

## Installation & Building

Ensure you have a recent Rust toolchain installed (Edition 2024 / Rust 1.97+).

```bash
# Build a fast debug binary
cargo build

# Build an optimized release binary
cargo build --release
```

The compiled binary is placed at `target/debug/llm-burner` or `target/release/llm-burner`.

---

## Recommended Models & Datasets

Because `llm-burner` implements a lightweight, architecture-compatible transformer body (supporting RMSNorm, RoPE, Grouped-Query Attention, SwiGLU/GeGLU, and tied/untied embeddings), it loads weights directly from standard Hugging Face repositories matching these layouts.

### 1. Small / Test Models (Fast & Lightweight)
* **`Qwen/Qwen2.5-0.5B`** or **`Qwen/Qwen2.5-1.5B`**
  * *Why*: Excellent open weights, standard Llama/Qwen key layouts, highly performant on CPU.
* **`google/gemma-2b`** or **`google/gemma-2-2b`**
  * *Why*: Native Gemma architecture support (including RMS Norm on queries/keys when present).
* **`TinyLlama/TinyLlama-1.1B-Chat-v1.0`**
  * *Why*: Extremely fast to download and fine-tune on local hardware.

### 2. Recommended Text Datasets
* **`wikitext`** (subset: `wikitext-2-raw-v1`)
* **`roneneldan/TinyStories`**
  * *Why*: Small, clean text files ideal for quick end-to-end fine-tuning verification.

---

## Usage & Commands

### 1. Download Model and Dataset
Downloads `config.json`, tokenizer files, and `.safetensors` weights into `./models/` and text corpus files into `./datasets/`.

```bash
cargo run --release -- download \
  --model Qwen/Qwen2.5-0.5B \
  --dataset wikitext \
  --out .
```

### 2. Fine-Tune and Export
Fine-tunes the model on the downloaded corpus for an exact `--steps` count with a live Ratatui dashboard, then exports `model.safetensors` and `model.gguf` (`Q4K`) into the output directory.

```bash
cargo run --release -- train \
  --model-dir ./models/Qwen--Qwen2.5-0.5B \
  --dataset-dir ./datasets/wikitext--wikitext-2-raw-v1 \
  --steps 200 \
  --batch-size 4 \
  --seq-len 128 \
  --lr 3e-4 \
  --out ./artifacts
```

Alternatively, `train` can auto-download the repositories on the fly if provided via `--model` and `--dataset` instead of local directories:

```bash
cargo run --release -- train \
  --model Qwen/Qwen2.5-0.5B \
  --dataset wikitext \
  --steps 100 \
  --out ./artifacts
```

---

## Refusal Ablation (Abliteration)

Fine-tuned chat models often keep refusing requests they were retrained to answer. `train --ablate-refusal` removes the model's *refusal direction* before fine-tuning begins: it measures the difference between residual-stream activations of harmful and harmless probe prompts at one decoder layer, then projects that direction out of every weight matrix writing into the residual stream (`o_proj`, `down_proj`). Because the change is applied to weights, it is permanent and survives both `.safetensors` and GGUF export.

```bash
cargo run --release -- train \
  --model TinyLlama/TinyLlama-1.1B-Chat-v1.0 \
  --dataset wikitext \
  --steps 200 \
  --ablate-refusal \
  --out ./artifacts
```

| Flag | Default | Description |
| --- | --- | --- |
| `--ablate-refusal` | off | Enable refusal-direction ablation before training. |
| `--refusal-layer <N>` | `2 * n_layers / 3` | Decoder layer used to measure the direction. |
| `--ablate-scale <F>` | `1.0` | Removal strength in `[0, 1]`; lower values soften the effect. |
| `--harmful-file <PATH>` | built-in list | Newline-separated harmful probes (one per line; `#` comments allowed). |
| `--harmless-file <PATH>` | built-in list | Newline-separated harmless probe prompts. |

The built-in probe lists are small AdvBench-style corpora (48 + 48 prompts). For best results on a specific model, override them with larger lists such as [FailSpy's `harmful.txt` / `harmless.txt`](https://huggingface.co/datasets/failspy/abliteration) or AdvBench behaviors.

Notes:
* Probes are encoded as raw text; chat-template wrapping is not applied, which is usually sufficient because the direction generalizes across formats.
* The logged `cos(harmful, harmless)` similarity is a quality signal — values close to 1 mean the probe sets barely separate and the measured direction may be noisy.
* Ablation runs once, right after checkpoint load; fine-tuning then proceeds exactly as usual.
