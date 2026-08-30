//! Export a trained model to GGUF (Q4_K, via the `Q4K` scheme) and to
//! PyTorch-layout `.safetensors`.

use std::path::Path;

use anyhow::{Context, Result, bail};
use burn_store::{
    BurnToPyTorchAdapter, ModuleAdapter, ModuleSnapshot, ModuleStore, SafetensorsStore,
};
use rlx_gguf::{GgmlType, GgufWriter, MetaValue, quantize};

use crate::data::TokenizerStore;
use crate::model::load::FloatDTypeAdapter;
use crate::model::{LlmModel, LlmModelConfig};
use crate::train::Precision;

/// Dtype for tensors that must not be quantized: 1-D RMSNorm weights (llama.cpp
/// multiplies them elementwise, which is unsupported against quantized types)
/// and 2-D weights whose rows are not a multiple of the Q4_K block size.
const FALLBACK_DTYPE: GgmlType = GgmlType::F32;

/// Quantization scheme used for 2D projection/embedding weights.
const QUANT_DTYPE: GgmlType = GgmlType::Q4K;

/// Map a burn tensor path (e.g. `model.layers.0.self_attn.q_proj.weight`) to
/// the GGUF tensor name expected by llama.cpp (`blk.0.attn_q.weight`).
fn gguf_tensor_name(path: &str) -> Option<String> {
    let parts: Vec<&str> = path.split('.').collect();
    match parts.as_slice() {
        ["model", "embed_tokens", "weight"] => Some("token_embd.weight".into()),
        ["model", "norm", "weight"] => Some("output_norm.weight".into()),
        ["lm_head", "weight"] => Some("output.weight".into()),
        ["model", "layers", layer, "self_attn", proj, "weight"] => {
            let proj_name = match *proj {
                "q_proj" => "attn_q",
                "k_proj" => "attn_k",
                "v_proj" => "attn_v",
                "o_proj" => "attn_output",
                _ => return None,
            };
            Some(format!("blk.{layer}.{proj_name}.weight"))
        }
        ["model", "layers", layer, "self_attn", proj, "bias"] => {
            let proj_name = match *proj {
                "q_proj" => "attn_q",
                "k_proj" => "attn_k",
                "v_proj" => "attn_v",
                // Qwen2 trains no output-projection bias; anything else is
                // not a tensor this exporter knows how to place.
                _ => return None,
            };
            Some(format!("blk.{layer}.{proj_name}.bias"))
        }
        ["model", "layers", layer, "mlp", proj, "weight"] => {
            let proj_name = match *proj {
                "gate_proj" => "ffn_gate",
                "up_proj" => "ffn_up",
                "down_proj" => "ffn_down",
                _ => return None,
            };
            Some(format!("blk.{layer}.{proj_name}.weight"))
        }
        ["model", "layers", layer, "input_layernorm", "weight"] => {
            Some(format!("blk.{layer}.attn_norm.weight"))
        }
        [
            "model",
            "layers",
            layer,
            "post_attention_layernorm",
            "weight",
        ] => Some(format!("blk.{layer}.ffn_norm.weight")),
        _ => None,
    }
}

/// True when the tensor is a 2D `Linear` weight. Burn stores these as
/// `[in, out]`; llama.cpp expects the PyTorch layout (`[out, in]` row-major,
/// declared as ne `[in, out]`), so the buffer must be transposed before
/// writing. Embedding weights are `[vocab, d_model]` in both frameworks and
/// must NOT be transposed.
fn is_linear_weight(path: &str) -> bool {
    path.ends_with(".weight") && (path.starts_with("model.layers.") || path == "lm_head.weight")
}

/// True for the attention query/key projections. Their rows are grouped per
/// head, so their ordering depends on how RoPE pairs the head dimensions.
fn is_qk_projection(path: &str) -> bool {
    path.ends_with("self_attn.q_proj.weight") || path.ends_with("self_attn.k_proj.weight")
}

/// Reorder the rows of a PyTorch-layout `[out, in]` q/k projection from HF's
/// half-split RoPE grouping (`x0..x{h/2-1}, y0..y{h/2-1}` within each head)
/// to llama.cpp's interleaved pairing (`x0 y0 x1 y1 ...`). Mirrors the
/// `permute` in llama.cpp's `conversion/llama.py`; only needed for archs with
/// `LLAMA_ROPE_TYPE_NORM` (the llama family), not for gemma (NEOX pairing).
fn permute_rope_rows(flat: &[f32], out_dim: usize, in_dim: usize, n_group: usize) -> Vec<f32> {
    if n_group == 0 || !out_dim.is_multiple_of(n_group) {
        log::warn!("cannot rope-permute q/k projection: {out_dim} rows not divisible by {n_group}");
        return flat.to_vec();
    }
    let head_dim = out_dim / n_group;
    let half = head_dim / 2;
    let mut out = vec![0.0; flat.len()];
    for g in 0..n_group {
        for i in 0..half {
            for c in 0..in_dim {
                out[(g * head_dim + 2 * i) * in_dim + c] = flat[(g * head_dim + i) * in_dim + c];
                out[(g * head_dim + 2 * i + 1) * in_dim + c] =
                    flat[(g * head_dim + half + i) * in_dim + c];
            }
        }
    }
    out
}

/// Transpose a row-major `[rows, cols]` buffer into `[cols, rows]`.
fn transpose(flat: &[f32], rows: usize, cols: usize) -> Vec<f32> {
    let mut out = vec![0.0; flat.len()];
    for r in 0..rows {
        for c in 0..cols {
            out[c * rows + r] = flat[r * cols + c];
        }
    }
    out
}

/// Write the model to a GGUF file using the `Q4K` (≈ Q4_K_M) scheme.
pub fn export_gguf<B: burn::tensor::backend::Backend>(
    model: &LlmModel<B>,
    config: &LlmModelConfig,
    tokenizer: &TokenizerStore,
    path: &Path,
    name: &str,
) -> Result<()> {
    let arch = config.gguf_architecture();
    let snapshots = model.collect(None, None, false);
    if snapshots.is_empty() {
        bail!("model has no weights to export");
    }

    let mut writer = GgufWriter::new();
    writer.set_arch(arch);
    writer.set_meta("general.name", MetaValue::String(name.to_string()));
    writer.set_meta("general.file_type", MetaValue::U32(15)); // Q4_K
    add_model_metadata(&mut writer, config, arch);
    let pre = if arch == "qwen2" { Some("qwen2") } else { None };
    add_tokenizer_metadata(&mut writer, tokenizer, config.vocab_size, pre);

    for snapshot in &snapshots {
        let Some(gguf_name) = gguf_tensor_name(&snapshot.full_path()) else {
            log::warn!("skipping untranslated tensor `{}`", snapshot.full_path());
            continue;
        };

        let shape_vec = snapshot.shape.to_vec();
        log::debug!(
            "exporting `{}`: shape={:?}",
            snapshot.full_path(),
            shape_vec
        );
        let data = snapshot
            .to_data()
            .map_err(|e| anyhow::anyhow!("failed to read `{}`: {e}", snapshot.full_path()))?;
        // The trained model may hold half-precision weights; GGUF writing and
        // Q4_K quantization both operate on f32, so convert any float dtype.
        let floats: Vec<f32> = if data.dtype == burn::tensor::DType::F32 {
            data.to_vec::<f32>().map_err(|e| {
                anyhow::anyhow!("failed to read `{}` as f32: {e}", snapshot.full_path())
            })?
        } else {
            let converted = data.convert_dtype(burn::tensor::DType::F32);
            converted.to_vec::<f32>().map_err(|e| {
                anyhow::anyhow!("failed to read `{}` as f32: {e}", snapshot.full_path())
            })?
        };

        let floats = if arch == "gemma" && snapshot.full_path().ends_with("norm.weight") {
            // HF Gemma's RmsNorm computes `x * (1 + weight)` while llama.cpp's
            // gemma graph applies a plain `x * weight`. The official converter
            // (conversion/gemma.py) therefore stores `weight + 1`. Without this
            // shift every decoder layer's normalization is wrong, collapsing the
            // model into garbage. Applies to attn/ffn/output norms alike.
            floats.into_iter().map(|v| v + 1.0).collect()
        } else {
            floats
        };

        // GGUF declares dimensions fastest-first: ne[0] is the contiguous
        // axis, i.e. ne is the reverse of the row-major shape of the stored
        // buffer, and buffers must be in PyTorch layout:
        // - Linear weights: burn `[in, out]` -> transpose to `[out, in]`
        //   row-major, declared as ne `[in, out]`.
        // - Embeddings: already `[vocab, d_model]`, written as-is and
        //   declared as ne `[d_model, vocab]`.
        let (shape_vec, floats) = if shape_vec.len() == 2 {
            let [d0, d1] = snapshot.shape.dims::<2>();
            if is_linear_weight(&snapshot.full_path()) {
                // Burn `[in, out]` -> PyTorch `[out, in]` row-major.
                let mut pytorch = transpose(&floats, d0, d1);
                if is_qk_projection(&snapshot.full_path()) && arch == "llama" {
                    let n_group = if snapshot.full_path().ends_with("q_proj.weight") {
                        config.n_heads
                    } else {
                        config.n_kv_heads
                    };
                    pytorch = permute_rope_rows(&pytorch, d1, d0, n_group);
                }
                (vec![d0, d1], pytorch)
            } else {
                (vec![d1, d0], floats)
            }
        } else {
            (shape_vec, floats)
        };

        // Quantize only 2-D matrices whose contiguous dimension is a Q4_K
        // block multiple (blocks must never span rows). 1-D tensors — the
        // RMSNorm weights — stay F32: llama.cpp applies them with an
        // elementwise multiply, which does not support quantized operands.
        let dtype = match shape_vec.as_slice() {
            [d0, _] if d0 % Q4K_BLOCK == 0 => QUANT_DTYPE,
            _ => {
                log::debug!("`{gguf_name}` kept as {FALLBACK_DTYPE:?}");
                FALLBACK_DTYPE
            }
        };

        let bytes = quantize(&floats, dtype)
            .map_err(|e| anyhow::anyhow!("failed to quantize `{gguf_name}` with {dtype:?}: {e}"))?;

        writer
            .add_tensor_bytes(gguf_name.clone(), shape_vec, dtype, bytes)
            .with_context(|| format!("failed to add tensor `{gguf_name}`"))?;
    }

    writer
        .write_to_path(path)
        .with_context(|| format!("failed to write `{}`", path.display()))?;
    log::info!("wrote {} ({arch}) to {}", name, path.display());
    Ok(())
}

fn add_model_metadata(writer: &mut GgufWriter, config: &LlmModelConfig, arch: &str) {
    let u = |v: usize| MetaValue::U32(v as u32);
    let kv = |suffix: &str| format!("{arch}.{suffix}");
    writer.set_meta(kv("context_length"), u(config.max_seq_len));
    writer.set_meta(kv("embedding_length"), u(config.d_model));
    writer.set_meta(kv("block_count"), u(config.n_layers));
    writer.set_meta(kv("feed_forward_length"), u(config.intermediate_size));
    writer.set_meta(kv("attention.head_count"), u(config.n_heads));
    writer.set_meta(kv("attention.head_count_kv"), u(config.n_kv_heads));
    writer.set_meta(kv("vocab_size"), u(config.vocab_size));
    writer.set_meta(
        kv("attention.layer_norm_rms_epsilon"),
        MetaValue::F32(config.rms_eps as f32),
    );
    writer.set_meta(
        kv("rope.freq_base"),
        MetaValue::F32(config.rope_theta as f32),
    );
    writer.set_meta(
        kv("attention.sliding_window"),
        MetaValue::U32(config.sliding_window.unwrap_or(0) as u32),
    );
    // llama.cpp requires n_rot == n_embd / n_head for the llama arch, so this
    // must be the full per-head dimension (NOT half of it).
    writer.set_meta(kv("rope.dimension_count"), u(config.head_dim));
}

/// GGUF token type for padding placeholders (`TokenType::UNUSED`).
const TOKEN_TYPE_UNUSED: u32 = 5;

/// Write GGUF `tokenizer.ggml.*` metadata from a tokenizer.
///
/// Two conventions must be honored for llama.cpp to load the file:
///
/// - Vocabulary coverage: HF tokenizers often define fewer ids than
///   `config.vocab_size` reserves in the embedding matrix (Qwen2.5: 151,665
///   tokenizer ids vs 151,936 rows). llama.cpp sizes the model from the
///   `tokenizer.ggml.*` arrays rather than from `llama.vocab_size`, so a short
///   token array makes it expect an embedding of `[d_model, n_tokens]` and
///   reject the real `token_embd.weight` tensor with a "wrong shape" error.
///   The vocabulary is padded up to `vocab_size` with `[PADn]` UNUSED
///   placeholders — the same convention (numbered from 1) as llama.cpp's own
///   HF converter.
/// - Tokenizer family: vocabs with `<0xNN>` byte-fallback tokens follow the
///   SentencePiece layout and are exported as `tokenizer.ggml.model "llama"`
///   with a BOS token; GPT-2-style byte-level BPE vocabs (Qwen2.5, SmolLM...)
///   have none, so exporting them as `"llama"` makes llama.cpp's SPM loader
///   abort while looking up byte tokens. They are exported as `"gpt2"` with
///   their merge rules instead — matching llama.cpp's Qwen converter.
pub fn add_tokenizer_metadata(
    writer: &mut GgufWriter,
    tokenizer: &TokenizerStore,
    vocab_size: usize,
    pre: Option<&str>,
) {
    let spm_style = tokenizer.has_byte_fallback();
    let ggml_model = if spm_style { "llama" } else { "gpt2" };

    let mut pieces = vec![String::new(); vocab_size];
    let scores = vec![MetaValue::F32(0.0); vocab_size];
    let mut types = vec![TOKEN_TYPE_UNUSED; vocab_size];
    let mut defined = vec![false; vocab_size];

    let vocab = tokenizer.vocab_ordered();
    let token_types = tokenizer.token_types();
    for ((token, id), token_type) in vocab.into_iter().zip(token_types) {
        if (id as usize) < vocab_size {
            pieces[id as usize] = token;
            types[id as usize] = token_type;
            defined[id as usize] = true;
        } else {
            log::warn!(
                "tokenizer id {id} is outside the model vocab_size {vocab_size}; dropping it"
            );
        }
    }
    // Pad placeholders numbered from 1, matching llama.cpp's converter so the
    // token-array checksum can still match its known-vocabulary tables.
    if defined.iter().filter(|slot| !**slot).count() > 0 {
        let mut pad_n = 0usize;
        for (id, slot) in defined.iter_mut().enumerate() {
            if !*slot {
                pad_n += 1;
                pieces[id] = format!("[PAD{pad_n}]");
            }
        }
        log::info!(
            "padded {} vocabulary slots up to vocab_size {} with [PADn] UNUSED tokens",
            pad_n,
            vocab_size
        );
    }

    writer.set_meta("tokenizer.ggml.model", MetaValue::String(ggml_model.into()));
    if let Some(pre) = pre {
        writer.set_meta("tokenizer.ggml.pre", MetaValue::String(pre.into()));
    }
    writer.set_meta(
        "tokenizer.ggml.tokens",
        MetaValue::Array(pieces.into_iter().map(MetaValue::String).collect()),
    );
    writer.set_meta("tokenizer.ggml.scores", MetaValue::Array(scores));
    writer.set_meta(
        "tokenizer.ggml.token_type",
        MetaValue::Array(types.into_iter().map(MetaValue::U32).collect()),
    );

    // GPT-2-style BPE is merge-driven; without the rules llama.cpp cannot
    // tokenize at all. SPM-style vocabs don't use merges.
    if !spm_style {
        let merges = tokenizer.merges();
        if merges.is_empty() {
            log::warn!(
                "byte-level BPE tokenizer has no merge rules; llama.cpp will not be able to encode text"
            );
        } else {
            writer.set_meta(
                "tokenizer.ggml.merges",
                MetaValue::Array(
                    merges
                        .iter()
                        .cloned()
                        .map(MetaValue::String)
                        .collect::<Vec<_>>(),
                ),
            );
        }
    }
    // SentencePiece vocabs open with a fixed BOS token (`<s>` = 1). Byte-level
    // BPE vocabs have no BOS concept — claiming one would prepend an arbitrary
    // vocab entry to every prompt — so llama.cpp is told to add none.
    if spm_style {
        writer.set_meta("tokenizer.ggml.bos_token_id", MetaValue::U32(1));
        writer.set_meta("tokenizer.ggml.add_bos_token", MetaValue::Bool(true));
    } else {
        writer.set_meta("tokenizer.ggml.add_bos_token", MetaValue::Bool(false));
    }
    writer.set_meta(
        "tokenizer.ggml.eos_token_id",
        MetaValue::U32(tokenizer.eos_id),
    );
    writer.set_meta(
        "tokenizer.ggml.pad_token_id",
        MetaValue::U32(tokenizer.pad_id),
    );
    writer.set_meta("tokenizer.ggml.add_eos_token", MetaValue::Bool(false));

    // The Jinja chat template lets llama-cli / the llama server apply the
    // prompt format the model was trained on. Without it, runtimes fall back
    // to a generic template, and template-trained models (e.g. Zephyr-style
    // TinyLlama) degenerate into emitting `<|user|>`-style markup as plain
    // text.
    if let Some(template) = &tokenizer.chat_template {
        writer.set_meta(
            "tokenizer.chat_template",
            MetaValue::String(template.clone()),
        );
    }
}

/// Write the model to a PyTorch-compatible `.safetensors` file.
pub fn export_safetensors<B: burn::tensor::backend::Backend>(
    model: &LlmModel<B>,
    path: &Path,
    precision: Precision,
) -> Result<()> {
    let mut store = SafetensorsStore::from_file(path)
        .with_to_adapter(
            FloatDTypeAdapter::new(precision.safetensors_dtype()).chain(BurnToPyTorchAdapter),
        )
        .overwrite(true);
    store
        .collect_from(model)
        .with_context(|| format!("failed to write `{}`", path.display()))?;
    log::info!("wrote safetensors to {}", path.display());
    Ok(())
}

/// Test-only: block size used to decide quantization fallback.
const Q4K_BLOCK: usize = 256;

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn maps_gemma_tensor_names() {
        assert_eq!(
            gguf_tensor_name("model.embed_tokens.weight"),
            Some("token_embd.weight".to_string())
        );
        assert_eq!(
            gguf_tensor_name("model.layers.3.self_attn.q_proj.weight"),
            Some("blk.3.attn_q.weight".to_string())
        );
        assert_eq!(
            gguf_tensor_name("model.layers.1.self_attn.o_proj.weight"),
            Some("blk.1.attn_output.weight".to_string())
        );
        assert_eq!(
            gguf_tensor_name("model.layers.2.self_attn.v_proj.bias"),
            Some("blk.2.attn_v.bias".to_string())
        );
        assert_eq!(
            gguf_tensor_name("model.layers.0.self_attn.o_proj.bias"),
            None
        );
        assert_eq!(
            gguf_tensor_name("model.layers.0.mlp.up_proj.weight"),
            Some("blk.0.ffn_up.weight".to_string())
        );
        assert_eq!(
            gguf_tensor_name("model.layers.0.input_layernorm.weight"),
            Some("blk.0.attn_norm.weight".to_string())
        );
        assert_eq!(
            gguf_tensor_name("model.layers.0.post_attention_layernorm.weight"),
            Some("blk.0.ffn_norm.weight".to_string())
        );
        assert_eq!(
            gguf_tensor_name("model.norm.weight"),
            Some("output_norm.weight".to_string())
        );
        assert_eq!(
            gguf_tensor_name("lm_head.weight"),
            Some("output.weight".to_string())
        );
        assert_eq!(gguf_tensor_name("mystery.weight"), None);
        assert!(is_linear_weight("model.layers.0.self_attn.q_proj.weight"));
        assert!(is_linear_weight("lm_head.weight"));
        assert!(!is_linear_weight("model.embed_tokens.weight"));
    }

    #[test]
    fn transposes_row_major() {
        let flat = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
        assert_eq!(transpose(&flat, 2, 3), vec![1.0, 4.0, 2.0, 5.0, 3.0, 6.0]);
    }

    /// One head of four rows `x0 x1 y0 y1` must interleave to `x0 y0 x1 y1`,
    /// matching the reshape/swapaxes in llama.cpp's converter.
    #[test]
    fn permutes_rope_rows_like_hf_converter() {
        let flat = vec![10.0, 11.0, 20.0, 21.0];
        assert_eq!(
            permute_rope_rows(&flat, 4, 1, 1),
            vec![10.0, 20.0, 11.0, 21.0]
        );
    }

    /// Two heads of two rows: each head holds a single pair, so the order is
    /// already interleaved and the permutation is the identity.
    #[test]
    fn permute_rope_rows_is_identity_for_single_pairs() {
        let flat = vec![1.0, 2.0, 3.0, 4.0];
        assert_eq!(permute_rope_rows(&flat, 4, 1, 2), flat);
    }

    #[test]
    fn check_lm_head_shape() {
        use crate::model::LlmModelConfig;
        use crate::train::TestBackend;
        let device = burn::backend::flex::FlexDevice;
        let config = LlmModelConfig::tiny();
        let model = crate::model::LlmModel::<TestBackend>::new(&config, &device);
        let snapshots = model.collect(None, None, false);
        for s in &snapshots {
            let path = s.full_path();
            if path == "lm_head.weight" {
                let sv = s.shape.to_vec();
                eprintln!("DEBUG lm_head.weight shape = {:?}", sv);
                assert_eq!(sv, vec![config.d_model, config.vocab_size]);
            }
        }
    }

    /// Regression test: llama.cpp requires GGUF dims fastest-first
    /// (`output.weight` as ne `[d_model, vocab]`, `ffn_gate` as ne
    /// `[d_model, n_ff]`, ...). The exporter used to declare the transposed
    /// row-major shape instead, which made llama.cpp reject the file with
    /// `tensor 'output.weight' has wrong shape`.
    #[test]
    fn exported_gguf_declares_llama_cpp_dims() {
        use crate::model::LlmModelConfig;
        use crate::train::TestBackend;
        use rlx_gguf::GgufFile;

        let device = burn::backend::flex::FlexDevice;
        let config = LlmModelConfig::tiny();
        let model = crate::model::LlmModel::<TestBackend>::new(&config, &device);
        let tokenizer = dummy_tokenizer(&config).unwrap();

        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("tiny.gguf");
        export_gguf(&model, &config, &tokenizer, &path, "test").unwrap();

        let kv_dim = config.n_kv_heads * config.head_dim;
        let file = GgufFile::from_path(&path).unwrap();
        let shape = |name: &str| file.tensors[name].shape.clone();
        assert_eq!(
            shape("token_embd.weight"),
            vec![config.d_model, config.vocab_size]
        );
        assert_eq!(
            shape("output.weight"),
            vec![config.d_model, config.vocab_size]
        );
        assert_eq!(
            shape("blk.0.attn_q.weight"),
            vec![config.d_model, config.d_model]
        );
        assert_eq!(shape("blk.0.attn_k.weight"), vec![config.d_model, kv_dim]);
        assert_eq!(shape("blk.0.attn_v.weight"), vec![config.d_model, kv_dim]);
        assert_eq!(
            shape("blk.0.ffn_gate.weight"),
            vec![config.d_model, config.intermediate_size]
        );
        assert_eq!(
            shape("blk.0.ffn_down.weight"),
            vec![config.intermediate_size, config.d_model]
        );

        // Metadata keys llama.cpp requires to size the tensors (canonical
        // names — `llama.head_count` etc. are silently ignored, which made
        // llama.cpp compute zero-sized attention projections).
        for key in [
            "llama.context_length",
            "llama.embedding_length",
            "llama.block_count",
            "llama.feed_forward_length",
            "llama.attention.head_count",
            "llama.attention.head_count_kv",
            "llama.vocab_size",
            "llama.rope.dimension_count",
        ] {
            assert!(
                file.metadata.contains_key(key),
                "missing GGUF metadata key `{key}`"
            );
        }

        // Matrices whose contiguous dim is a Q4_K block multiple are
        // quantized; everything else (incl. all 1-D norm weights) stays F32.
        // Norms must be F32 because llama.cpp applies them with an elementwise
        // multiply that rejects quantized operands.
        for (name, t) in &file.tensors {
            let expected = if t.shape.len() == 2 && t.shape[0] % 256 == 0 {
                rlx_gguf::GgmlType::Q4K
            } else {
                rlx_gguf::GgmlType::F32
            };
            assert_eq!(t.dtype, expected, "wrong dtype for `{name}`");
        }

        // Token types must be classified: type 0 (UNDEFINED) makes llama.cpp
        // detokenize every token to an empty string.
        let types = match &file.metadata["tokenizer.ggml.token_type"] {
            MetaValue::Array(v) => v,
            other => panic!("unexpected token_type metadata: {other:?}"),
        };
        assert_eq!(types.len(), config.vocab_size);
        assert!(
            types.iter().all(|t| !matches!(t, MetaValue::U32(0))),
            "token_type must never be UNDEFINED"
        );

        // Embedding data must be untransposed: row `t` of the stored buffer
        // (contiguous over d_model) must match burn's `embed_tokens.weight`.
        let (embd, _) = file.dequant_f32("token_embd.weight").unwrap();
        let snapshot = model
            .collect(None, None, false)
            .into_iter()
            .find(|s| s.full_path() == "model.embed_tokens.weight")
            .unwrap();
        let floats = snapshot.to_data().unwrap().to_vec::<f32>().unwrap();
        let d = config.d_model;
        for t in [0usize, 1, config.vocab_size / 2, config.vocab_size - 1] {
            for e in [0usize, 1, d - 1] {
                let expected = floats[t * d + e];
                let got = embd[t * d + e];
                assert!(
                    (expected - got).abs() < 0.25,
                    "token_embd[{t}][{e}] mismatch: {got} vs {expected}"
                );
            }
        }

        // q/k projections must be rope-permuted: the exported rows (PyTorch
        // layout) interleave each head's rotary halves. Tiny tensors stay F32,
        // so the comparison is exact.
        let snapshots = model.collect(None, None, false);
        let check_qk = |hf_name: &str, gguf_name: &str, n_group: usize| {
            let snapshot = snapshots.iter().find(|s| s.full_path() == hf_name).unwrap();
            let burn_floats = snapshot.to_data().unwrap().to_vec::<f32>().unwrap();
            let [d0, d1] = snapshot.shape.dims::<2>();
            let transposed = transpose(&burn_floats, d0, d1);
            let expected = permute_rope_rows(&transposed, d1, d0, n_group);
            let (got, _) = file.dequant_f32(gguf_name).unwrap();
            assert_eq!(file.tensors[gguf_name].dtype, rlx_gguf::GgmlType::F32);
            assert_eq!(got.len(), expected.len());
            for (i, (g, e)) in got.iter().zip(expected.iter()).enumerate() {
                assert_eq!(g, e, "{gguf_name} row-permute mismatch at flat index {i}");
            }
        };
        check_qk(
            "model.layers.0.self_attn.q_proj.weight",
            "blk.0.attn_q.weight",
            config.n_heads,
        );
        check_qk(
            "model.layers.0.self_attn.k_proj.weight",
            "blk.0.attn_k.weight",
            config.n_kv_heads,
        );

        // v must NOT be permuted.
        {
            let snapshot = snapshots
                .iter()
                .find(|s| s.full_path() == "model.layers.0.self_attn.v_proj.weight")
                .unwrap();
            let burn_floats = snapshot.to_data().unwrap().to_vec::<f32>().unwrap();
            let [d0, d1] = snapshot.shape.dims::<2>();
            let expected = transpose(&burn_floats, d0, d1);
            let (got, _) = file.dequant_f32("blk.0.attn_v.weight").unwrap();
            assert_eq!(got, expected);
        }
    }

    /// Regression test: Qwen-style tokenizers define fewer ids than
    /// `config.vocab_size` reserves (Qwen2.5: 151,665 ids vs 151,936 rows).
    /// llama.cpp sizes tensors from the GGUF `tokenizer.ggml.*` arrays, so the
    /// short array made it expect an `[d_model, 151665]` embedding and reject
    /// the exported `token_embd.weight` with a shape mismatch. The exporter
    /// must pad the vocabulary up to `vocab_size`.
    #[test]
    fn pads_short_tokenizer_vocab_to_model_vocab_size() {
        use crate::model::LlmModelConfig;
        use crate::train::TestBackend;
        use rlx_gguf::GgufFile;

        let device = burn::backend::flex::FlexDevice;
        let config = LlmModelConfig::tiny();
        let short_config = LlmModelConfig {
            vocab_size: config.vocab_size / 2,
            ..config.clone()
        };
        let model = crate::model::LlmModel::<TestBackend>::new(&config, &device);
        let tokenizer = dummy_tokenizer(&short_config).unwrap();

        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("padded.gguf");
        export_gguf(&model, &config, &tokenizer, &path, "test").unwrap();

        let file = GgufFile::from_path(&path).unwrap();
        let tokens = match &file.metadata["tokenizer.ggml.tokens"] {
            MetaValue::Array(v) => v,
            other => panic!("unexpected tokens metadata: {other:?}"),
        };
        let types = match &file.metadata["tokenizer.ggml.token_type"] {
            MetaValue::Array(v) => v,
            other => panic!("unexpected token_type metadata: {other:?}"),
        };

        // Arrays cover the full embedding vocab, not just the tokenizer's.
        assert_eq!(tokens.len(), config.vocab_size);
        assert_eq!(types.len(), config.vocab_size);

        // Real ids keep their slots; padding fills only the undefined tail.
        let last_defined = short_config.vocab_size - 1;
        assert!(
            matches!(&tokens[last_defined], MetaValue::String(s) if *s == format!("tok{last_defined}")),
            "defined token was displaced: {:?}",
            tokens[last_defined]
        );
        for (n, id) in [
            (1usize, config.vocab_size / 2),
            (
                config.vocab_size - short_config.vocab_size,
                config.vocab_size - 1,
            ),
        ] {
            assert!(
                matches!(tokens[id], MetaValue::String(ref s) if *s == format!("[PAD{n}]")),
                "expected [PAD{n}] placeholder at id {id}, got {:?}",
                tokens[id]
            );
            assert!(matches!(types[id], MetaValue::U32(TOKEN_TYPE_UNUSED)));
        }

        // Metadata still declares the embedding row count.
        assert!(matches!(
            file.metadata["llama.vocab_size"],
            MetaValue::U32(n) if n == config.vocab_size as u32
        ));

        // No byte-fallback tokens -> GPT-2-style BPE family.
        assert!(
            matches!(&file.metadata["tokenizer.ggml.model"], MetaValue::String(s) if s == "gpt2")
        );
    }

    /// Tied-embedding models (Qwen2.5, Gemma) have no `lm_head` parameter, so
    /// no `output.weight` is written. llama.cpp's architecture loaders treat
    /// the output projection as duplicated from `token_embd` for tied models
    /// (e.g. gemma always, llama when `tie_word_embeddings` is set), so
    /// writing an extra `output.weight` would push the tensor count one above
    /// what llama.cpp expects and fail to load. The exporter must therefore
    /// leave `token_embd.weight` to serve both roles.
    #[test]
    fn tied_embeddings_do_not_export_output_weight() {
        use crate::model::LlmModelConfig;
        use crate::train::TestBackend;
        use rlx_gguf::GgufFile;

        let device = burn::backend::flex::FlexDevice;
        let config = LlmModelConfig {
            tie_word_embeddings: true,
            ..LlmModelConfig::tiny()
        };
        let model = crate::model::LlmModel::<TestBackend>::new(&config, &device);
        // No lm_head parameter exists in a tied model.
        assert!(
            !model
                .collect(None, None, false)
                .iter()
                .any(|s| s.full_path() == "lm_head.weight")
        );
        let tokenizer = dummy_tokenizer(&config).unwrap();

        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("tied.gguf");
        export_gguf(&model, &config, &tokenizer, &path, "test").unwrap();
        let file = GgufFile::from_path(&path).unwrap();

        assert!(file.tensors.contains_key("token_embd.weight"));
        // Tied models must NOT emit a separate output.weight: llama.cpp
        // duplicates the output from token_embd and would reject the extra
        // tensor with a wrong-tensor-count error.
        assert!(
            !file.tensors.contains_key("output.weight"),
            "tied model must not emit a separate output.weight"
        );
    }

    /// Qwen2-style checkpoints train non-zero attention QKV biases. They must
    /// survive the GGUF export as `blk.N.attn_{q,k,v}.bias` (F32, no
    /// transpose) — dropping them silently turns every loaded model into
    /// gibberish.
    #[test]
    fn exports_attention_biases() {
        use crate::model::LlmModelConfig;
        use crate::train::TestBackend;
        use rlx_gguf::GgufFile;

        let device = burn::backend::flex::FlexDevice;
        let config = LlmModelConfig {
            qkv_bias: true,
            ..LlmModelConfig::tiny()
        };
        let model = crate::model::LlmModel::<TestBackend>::new(&config, &device);
        let tokenizer = dummy_tokenizer(&config).unwrap();

        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("bias.gguf");
        export_gguf(&model, &config, &tokenizer, &path, "test").unwrap();
        let file = GgufFile::from_path(&path).unwrap();

        for proj in ["attn_q", "attn_k", "attn_v"] {
            let name = format!("blk.0.{proj}.bias");
            assert!(file.tensors.contains_key(&name), "missing `{name}`");
            assert_eq!(file.tensors[&name].dtype, rlx_gguf::GgmlType::F32);
            // Declared fastest-first as a plain [out] vector.
            assert_eq!(file.tensors[&name].shape.len(), 1);
        }
        assert!(!file.tensors.contains_key("blk.0.attn_output.bias"));

        // Values must match the burn parameters exactly (F32 path).
        let snapshots = model.collect(None, None, false);
        let snapshot = snapshots
            .iter()
            .find(|s| s.full_path() == "model.layers.0.self_attn.q_proj.bias")
            .unwrap();
        let floats = snapshot.to_data().unwrap().to_vec::<f32>().unwrap();
        let (got, _) = file.dequant_f32("blk.0.attn_q.bias").unwrap();
        assert_eq!(got.len(), floats.len());
        for (i, (g, e)) in got.iter().zip(floats.iter()).enumerate() {
            assert_eq!(g, e, "q bias mismatch at flat index {i}");
        }
    }

    /// HF Gemma's RmsNorm computes `x * (1 + weight)` while llama.cpp applies a
    /// plain `x * weight`, so every gemma norm must be exported with `+1` baked
    /// in (see conversion/gemma.py). Missing this shift collapses the model to
    /// gibberish. Applies identically to the three norm kinds we export.
    #[test]
    fn gemma_norms_are_shifted_by_one() {
        use crate::model::LlmModelConfig;
        use crate::train::TestBackend;
        use rlx_gguf::GgufFile;

        let device = burn::backend::flex::FlexDevice;
        let config = LlmModelConfig {
            hf_model_type: "gemma".into(),
            use_gelu: true,
            ..LlmModelConfig::tiny()
        };
        let model = crate::model::LlmModel::<TestBackend>::new(&config, &device);
        let tokenizer = dummy_tokenizer(&config).unwrap();

        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("gemma-norm.gguf");
        export_gguf(&model, &config, &tokenizer, &path, "test").unwrap();
        let file = GgufFile::from_path(&path).unwrap();

        let snapshots = model.collect(None, None, false);
        let raw = |p: &str| {
            snapshots
                .iter()
                .find(|s| s.full_path() == p)
                .unwrap()
                .to_data()
                .unwrap()
                .to_vec::<f32>()
                .unwrap()
        };

        // output_norm <- model.norm.weight
        let pairs = [
            ("model.norm.weight", "output_norm.weight"),
            (
                "model.layers.0.input_layernorm.weight",
                "blk.0.attn_norm.weight",
            ),
            (
                "model.layers.0.post_attention_layernorm.weight",
                "blk.0.ffn_norm.weight",
            ),
        ];
        for (hf, gguf) in pairs {
            let expected: Vec<f32> = raw(hf).into_iter().map(|v| v + 1.0).collect();
            let (got, _) = file.dequant_f32(gguf).unwrap();
            assert_eq!(got, expected, "`{gguf}` must be stored as raw + 1");
            assert_eq!(
                file.tensors[gguf].dtype,
                rlx_gguf::GgmlType::F32,
                "`{gguf}` must stay F32"
            );
        }
    }

    /// Value-level round-trip on a gemma-family tiny model: every exported
    /// tensor must match the in-memory parameters after the same
    /// transforms the exporter applies (transpose 2-D linears to PyTorch
    /// row-major, add 1 to norms, leave embeddings/biases alone, no RoPE
    /// permutation for the NEOX gemma family). Tiny tensors stay F32, so the
    /// comparison is exact and any transposition/layout bug is caught here
    /// rather than surfacing as gibberish in llama.cpp.
    #[test]
    fn gemma_export_roundtrips_weight_values() {
        use crate::model::LlmModelConfig;
        use crate::train::TestBackend;
        use rlx_gguf::GgufFile;

        let device = burn::backend::flex::FlexDevice;
        let config = LlmModelConfig {
            hf_model_type: "gemma".into(),
            use_gelu: true,
            qkv_bias: true,
            tie_word_embeddings: true,
            ..LlmModelConfig::tiny()
        };
        let model = crate::model::LlmModel::<TestBackend>::new(&config, &device);
        let tokenizer = dummy_tokenizer(&config).unwrap();

        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("gemma-rt.gguf");
        export_gguf(&model, &config, &tokenizer, &path, "test").unwrap();
        let file = GgufFile::from_path(&path).unwrap();

        let snapshots = model.collect(None, None, false);

        // Recompute the expected flat GGUF buffer for a snapshot, mirroring
        // the exporter's transforms exactly.
        let expected = |snap: &burn_store::TensorSnapshot| -> Vec<f32> {
            let mut floats = snap.to_data().unwrap().to_vec::<f32>().unwrap();
            if config.hf_model_type.starts_with("gemma")
                && snap.full_path().ends_with("norm.weight")
            {
                floats.iter_mut().for_each(|v| *v += 1.0);
            }
            if snap.shape.num_dims() == 2 && is_linear_weight(&snap.full_path()) {
                // Burn `[in, out]` -> PyTorch `[out, in]` row-major. Gemma uses
                // NEOX RoPE: no additional row permutation.
                let [d0, d1] = snap.shape.dims::<2>();
                transpose(&floats, d0, d1)
            } else {
                floats
            }
        };

        for snap in &snapshots {
            let Some(gguf_name) = gguf_tensor_name(&snap.full_path()) else {
                continue;
            };
            let expect = expected(snap);
            let (got, _) = file.dequant_f32(&gguf_name).unwrap();
            assert_eq!(
                got.len(),
                expect.len(),
                "element count mismatch for `{gguf_name}`"
            );
            for (i, (g, e)) in got.iter().zip(expect.iter()).enumerate() {
                assert_eq!(g, e, "value mismatch at flat index {i} for `{gguf_name}`");
            }
        }

        // Sanity: every exported tensor was validated above.
        assert!(!file.tensors.is_empty(), "no tensors validated");
    }

    /// Vocabs containing `<0xNN>` byte-fallback tokens follow SentencePiece
    /// conventions and must export as `tokenizer.ggml.model = "llama"` with a
    /// BOS token; exporting them as `"gpt2"` would make llama.cpp's BPE loader
    /// fail on the missing merge rules, and vice versa a true byte-level BPE
    /// vocab exported as `"llama"` aborts llama.cpp's SPM loader while
    /// resolving byte tokens (`unordered_map::at`).
    #[test]
    fn picks_gguf_tokenizer_family_from_byte_fallback() {
        use crate::model::LlmModelConfig;
        use crate::train::TestBackend;
        use rlx_gguf::GgufFile;

        let device = burn::backend::flex::FlexDevice;
        let config = LlmModelConfig::tiny();
        let model = crate::model::LlmModel::<TestBackend>::new(&config, &device);

        let export = |tokenizer: &crate::data::TokenizerStore, dir: &tempfile::TempDir| {
            let path = dir.path().join("family.gguf");
            export_gguf(&model, &config, tokenizer, &path, "test").unwrap();
            GgufFile::from_path(&path).unwrap()
        };

        // SentencePiece-style: every slot is a byte-fallback token.
        let spm_dir = tempfile::tempdir().unwrap();
        let spm_path = spm_dir.path().join("tok.json");
        let vocab: Vec<String> = (0..config.vocab_size)
            .map(|i| format!("\"<0x{i:02X}>\": {i}"))
            .collect();
        std::fs::write(
            &spm_path,
            format!(
                r#"{{"version":"1.0","added_tokens":[],"normalizer":null,
                "pre_tokenizer":null,"post_processor":null,"decoder":null,
                "model":{{"type":"WordLevel","vocab":{{{}}},"unk_id":0,"unk_token":"<0x00>"}}}}"#,
                vocab.join(",")
            ),
        )
        .unwrap();
        let spm_file = export(
            &crate::data::TokenizerStore::from_file(&spm_path).unwrap(),
            &spm_dir,
        );
        assert!(
            matches!(&spm_file.metadata["tokenizer.ggml.model"], MetaValue::String(s) if s == "llama")
        );
        assert!(matches!(
            spm_file.metadata["tokenizer.ggml.bos_token_id"],
            MetaValue::U32(1)
        ));

        // Byte-level BPE-style: no byte-fallback tokens anywhere.
        let bpe_dir = tempfile::tempdir().unwrap();
        let bpe_file = export(&dummy_tokenizer(&config).unwrap(), &bpe_dir);
        assert!(
            matches!(&bpe_file.metadata["tokenizer.ggml.model"], MetaValue::String(s) if s == "gpt2")
        );
        assert!(
            !bpe_file
                .metadata
                .contains_key("tokenizer.ggml.bos_token_id")
        );
        assert!(matches!(
            bpe_file.metadata["tokenizer.ggml.add_bos_token"],
            MetaValue::Bool(false)
        ));
    }

    /// The Qwen2 family must map to a QKV-biased model even when config.json
    /// omits `attention_bias` (HF's default for that architecture).
    #[test]
    fn qwen_config_enables_attention_bias_by_default() {
        use crate::config::TransformersConfig;
        use crate::model::LlmModelConfig;

        let qwen = LlmModelConfig::from_transformers(
            &TransformersConfig::from_value(&serde_json::json!({
                "model_type": "qwen2",
                "hidden_size": 896,
                "intermediate_size": 4864,
                "num_attention_heads": 14,
                "num_hidden_layers": 24,
                "vocab_size": 151936,
                "max_position_embeddings": 32768,
            }))
            .unwrap(),
        );
        assert!(qwen.qkv_bias);

        // Explicit false must be honored (and non-Qwen stays bias-free).
        let tiny = LlmModelConfig::from_transformers(
            &TransformersConfig::from_value(&serde_json::json!({
                "model_type": "llama",
                "hidden_size": 576,
                "intermediate_size": 1536,
                "num_attention_heads": 12,
                "num_hidden_layers": 4,
                "vocab_size": 32000,
                "max_position_embeddings": 2048,
            }))
            .unwrap(),
        );
        assert!(!tiny.qkv_bias);
    }

    /// Qwen2 models must export under the `qwen2` GGUF architecture with
    /// `qwen2.`-prefixed metadata, a `tokenizer.ggml.pre` of `qwen2`, and —
    /// critically — UNPERMUTED Q/K weights: this build of llama.cpp applies
    /// NEOX-style RoPE directly to HF-layout weights for that architecture.
    /// Exporting as `llama` (with permutation) loads fine but generates
    /// garbage output.
    #[test]
    fn qwen2_exports_own_architecture_without_rope_permute() {
        use crate::config::TransformersConfig;
        use crate::model::LlmModelConfig;
        use crate::train::TestBackend;
        use rlx_gguf::GgufFile;

        let device = burn::backend::flex::FlexDevice;
        let config = LlmModelConfig::from_transformers(
            &TransformersConfig::from_value(&serde_json::json!({
                "model_type": "qwen2",
                "hidden_size": 64,
                "intermediate_size": 256,
                "num_attention_heads": 4,
                "num_key_value_heads": 2,
                "num_hidden_layers": 2,
                "vocab_size": 256,
                "max_position_embeddings": 64,
                "tie_word_embeddings": false,
            }))
            .unwrap(),
        );
        assert_eq!(config.gguf_architecture(), "qwen2");
        assert!(config.qkv_bias);

        let model = crate::model::LlmModel::<TestBackend>::new(&config, &device);
        let tokenizer = dummy_tokenizer(&config).unwrap();

        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("qwen.gguf");
        export_gguf(&model, &config, &tokenizer, &path, "test").unwrap();
        let file = GgufFile::from_path(&path).unwrap();

        // Architecture + prefixed metadata keys.
        assert!(
            matches!(&file.metadata["general.architecture"], MetaValue::String(s) if s == "qwen2")
        );
        for key in [
            "qwen2.context_length",
            "qwen2.embedding_length",
            "qwen2.block_count",
            "qwen2.vocab_size",
            "qwen2.rope.dimension_count",
        ] {
            assert!(file.metadata.contains_key(key), "missing `{key}`");
        }
        assert!(!file.metadata.contains_key("llama.vocab_size"));

        // Tokenizer pre.
        assert!(
            matches!(&file.metadata["tokenizer.ggml.pre"], MetaValue::String(s) if s == "qwen2")
        );

        // Q/K must be plain transposes (no rope row permutation): rebuild the
        // expected PyTorch-layout matrix from the burn snapshot.
        let snapshots = model.collect(None, None, false);
        for hf_name in [
            "model.layers.0.self_attn.q_proj.weight",
            "model.layers.0.self_attn.k_proj.weight",
        ] {
            let s = snapshots.iter().find(|s| s.full_path() == hf_name).unwrap();
            let [d0, d1] = s.shape.dims::<2>();
            let floats = s.to_data().unwrap().to_vec::<f32>().unwrap();
            let expected = transpose(&floats, d0, d1);
            let gguf_name = gguf_tensor_name(hf_name).unwrap();
            let (got, _) = file.dequant_f32(&gguf_name).unwrap();
            assert_eq!(got.len(), expected.len());
            for (i, (g, e)) in got.iter().zip(expected.iter()).enumerate() {
                assert_eq!(g, e, "{gguf_name} must be an unpermuted transpose at {i}");
            }
        }
    }

    /// Minimal vocab so the export test does not need a tokenizer file.
    fn dummy_tokenizer(
        config: &crate::model::LlmModelConfig,
    ) -> anyhow::Result<crate::data::TokenizerStore> {
        let dir = tempfile::tempdir()?;
        let path = dir.path().join("tokenizer.json");
        let vocab: Vec<String> = (0..config.vocab_size)
            .map(|i| format!("\"tok{i}\": {i}"))
            .collect();
        std::fs::write(
            &path,
            format!(
                r#"{{"version":"1.0","added_tokens":[],"normalizer":null,
                "pre_tokenizer":null,"post_processor":null,"decoder":null,
                "model":{{"type":"WordLevel","vocab":{{{}}},"unk_id":0,"unk_token":"tok0"}}}}"#,
                vocab.join(",")
            ),
        )?;
        let mut store = crate::data::TokenizerStore::from_file(&path)?;
        store.set_seq_len(config.max_seq_len);
        Ok(store)
    }
}
