# llm-burner

`llm-burner` is a Rust CLI designed to download Hugging Face model and text datasets, fine-tune a simplified Gemma/Llama-style transformer using [Burn](https://github.com/tracel-ai/burn), render a live Ratatui progress dashboard, and export checkpoints to both retrainable `.safetensors` and quantized `.gguf` (`Q4_K_M` via `rlx-gguf`).

---

## Installation & Building

Ensure you have a recent Rust toolchain installed (Edition 2024 / Rust 1.97+).

```bash
# Build a fast debug binary (GPU/Vulkan backend by default)
cargo build

# Build an optimized release binary
cargo build --release

# CPU-only build (no Vulkan drivers required)
cargo build --release --no-default-features --features flex
```

The compiled binary is placed at `target/debug/llm-burner` or `target/release/llm-burner`.

### Backends

| Feature | Default | Backend |
| --- | --- | --- |
| `gpu` | yes | Burn fused wgpu backend: Vulkan on Linux/Windows, Metal on macOS. The device is picked automatically, preferring high-power GPUs; override with `CUBECL_WGPU_DEFAULT_DEVICE` (e.g. `IntegratedGpu(0)`, `Cpu`). |
| `flex` | no (`--no-default-features --features flex`) | Pure-Rust CPU backend with SIMD; computes in f32 only. |

`--precision f32|bf16|f16` selects the dtype for checkpoint loading, training math,
optimizer state, and safetensors export on GPU builds — the whole stack runs in the
selected precision (F32 remains the default because pure half-precision AdamW can be
numerically fragile). On `flex` builds only f32 is available.

> **Known issue:** on AMD RADV (Mesa ≤ 25.0.x) `--precision bf16` segfaults inside
> `libvulkan_radeon.so` while compiling cubecl's bf16 compute pipelines (bf16 is
> emulated through bit-manipulation SPIR-V that trips this driver). `--precision f16`
> works — WGSL supports it natively. Revisit bf16 after a Mesa upgrade.

Use `--no-tui` to skip the Ratatui dashboard (tests, CI, non-interactive runs);
progress goes to the log file only.

While training runs, raw stdout/stderr are redirected into `train.log` so cubecl/wgpu
compile diagnostics cannot garble the Ratatui dashboard; everything lands in
`artifacts/trained/train.log`.

Corpus tokenization streams files in ~1 MiB line-aligned chunks into a flat window
arena, so peak RAM stays around a few hundred MB even for multi-gigabyte corpora.

> **Tip:** always train with `--release`. Debug builds compile compute shaders far slower and can overflow the stack while cubecl autotunes kernels on first use.

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

All commands default to a single output root, `./artifacts/` (override with `--out`):

```
artifacts/
├── models/<owner>--<name>/     # downloaded model snapshots
├── datasets/<owner>--<name>/   # downloaded text corpora
└── trained/                    # fine-tuned outputs
    ├── model.safetensors       # retrainable checkpoint (--precision)
    ├── model.gguf              # quantized Q4_K_M export
    ├── config.json
    └── tokenizer.json (+ tokenizer_config.json)
```

The `trained/` directory is self-contained: pass it to `export --model-dir` (or share it) without extra files.

### 1. Download Model and Dataset
Downloads `config.json`, tokenizer files, and `.safetensors` weights into `./artifacts/models/` and text corpus files into `./artifacts/datasets/`.

```bash
cargo run --release -- download \
  --model Qwen/Qwen2.5-0.5B \
  --dataset wikitext
```

### 2. Fine-Tune and Export
Fine-tunes the model on the downloaded corpus for an exact `--steps` count with a live Ratatui dashboard, then exports `model.safetensors` and `model.gguf` (`Q4K`) into `./artifacts/trained/`.

```bash
cargo run --release -- train \
  --model-dir ./artifacts/models/Qwen--Qwen2.5-0.5B \
  --dataset-dir ./artifacts/datasets/wikitext--wikitext-2-raw-v1 \
  --steps 200 \
  --batch-size 4 \
  --seq-len 128 \
  --lr 3e-4
```

Alternatively, `train` can auto-download the repositories on the fly if provided via `--model` and `--dataset` instead of local directories:

```bash
cargo run --release -- train \
  --model Qwen/Qwen2.5-0.5B \
  --dataset wikitext \
  --steps 100
```

---

## Refusal Ablation (Abliteration)

Fine-tuned chat models often keep refusing requests they were retrained to answer. `train --ablate-refusal` removes the model's *refusal direction* before fine-tuning begins: it measures the difference between residual-stream activations of harmful and harmless probe prompts at one decoder layer, then projects that direction out of every weight matrix writing into the residual stream (`o_proj`, `down_proj`). Because the change is applied to weights, it is permanent and survives both `.safetensors` and GGUF export.

```bash
cargo run --release -- train \
  --model TinyLlama/TinyLlama-1.1B-Chat-v1.0 \
  --dataset wikitext \
  --steps 200 \
  --ablate-refusal
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
