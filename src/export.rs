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
use crate::train::{FlexBackend, Precision};

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
pub fn export_gguf(
    model: &LlmModel<FlexBackend>,
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
    add_model_metadata(&mut writer, config);
    add_tokenizer_metadata(&mut writer, tokenizer, "llama");

    for snapshot in &snapshots {
        // When embeddings are tied, the output projection (lm_head) is the
        // same matrix as the token embeddings; do NOT write a separate
        // output.weight – llama.cpp will use token_embd.weight instead.
        if config.tie_word_embeddings && snapshot.full_path() == "lm_head.weight" {
            log::warn!(
                "skipping output.weight (tied embeddings); token_embd.weight will be used as output projection"
            );
            continue;
        }

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
        let floats: Vec<f32> = data.to_vec::<f32>().map_err(|e| {
            anyhow::anyhow!("failed to read `{}` as f32: {e}", snapshot.full_path())
        })?;

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

fn add_model_metadata(writer: &mut GgufWriter, config: &LlmModelConfig) {
    let u = |v: usize| MetaValue::U32(v as u32);
    writer.set_meta("llama.context_length", u(config.max_seq_len));
    writer.set_meta("llama.embedding_length", u(config.d_model));
    writer.set_meta("llama.block_count", u(config.n_layers));
    writer.set_meta("llama.feed_forward_length", u(config.intermediate_size));
    writer.set_meta("llama.attention.head_count", u(config.n_heads));
    writer.set_meta("llama.attention.head_count_kv", u(config.n_kv_heads));
    writer.set_meta("llama.vocab_size", u(config.vocab_size));
    writer.set_meta(
        "llama.attention.layer_norm_rms_epsilon",
        MetaValue::F32(config.rms_eps as f32),
    );
    writer.set_meta(
        "llama.rope.freq_base",
        MetaValue::F32(config.rope_theta as f32),
    );
    writer.set_meta(
        "llama.attention.sliding_window",
        MetaValue::U32(config.sliding_window.unwrap_or(0) as u32),
    );
    writer.set_meta("llama.expert_count", MetaValue::U32(0));
    // llama.cpp requires n_rot == n_embd / n_head for the llama arch, so this
    // must be the full per-head dimension (NOT half of it).
    writer.set_meta("llama.rope.dimension_count", u(config.head_dim));
}

/// Write GGUF `tokenizer.ggml.*` metadata from a tokenizer.
pub fn add_tokenizer_metadata(
    writer: &mut GgufWriter,
    tokenizer: &TokenizerStore,
    model_name: &str,
) {
    let vocab = tokenizer.vocab_ordered();
    let tokens: Vec<MetaValue> = vocab
        .iter()
        .map(|(t, _)| MetaValue::String(t.clone()))
        .collect();
    let scores: Vec<MetaValue> = vocab.iter().map(|_| MetaValue::F32(0.0)).collect();
    let token_types: Vec<MetaValue> = tokenizer
        .token_types()
        .into_iter()
        .map(MetaValue::U32)
        .collect();

    writer.set_meta(
        "tokenizer.ggml.model",
        MetaValue::String(model_name.to_string()),
    );
    writer.set_meta("tokenizer.ggml.tokens", MetaValue::Array(tokens));
    writer.set_meta("tokenizer.ggml.scores", MetaValue::Array(scores));
    writer.set_meta("tokenizer.ggml.token_type", MetaValue::Array(token_types));
    writer.set_meta("tokenizer.ggml.bos_token_id", MetaValue::U32(1));
    writer.set_meta(
        "tokenizer.ggml.eos_token_id",
        MetaValue::U32(tokenizer.eos_id),
    );
    writer.set_meta(
        "tokenizer.ggml.pad_token_id",
        MetaValue::U32(tokenizer.pad_id),
    );
    writer.set_meta("tokenizer.ggml.add_bos_token", MetaValue::Bool(true));
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
pub fn export_safetensors(
    model: &LlmModel<FlexBackend>,
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
        use crate::train::FlexBackend;
        let device = burn::backend::flex::FlexDevice;
        let config = LlmModelConfig::tiny();
        let model = crate::model::LlmModel::<FlexBackend>::new(&config, &device);
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
        use crate::train::FlexBackend;
        use rlx_gguf::GgufFile;

        let device = burn::backend::flex::FlexDevice;
        let config = LlmModelConfig::tiny();
        let model = crate::model::LlmModel::<FlexBackend>::new(&config, &device);
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
