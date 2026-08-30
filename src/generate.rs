//! Autoregressive text generation with greedy, temperature, top-k, and top-p sampling.

use anyhow::Result;
use burn::tensor::activation::softmax;
use burn::tensor::backend::Backend;
use burn::tensor::{Bool, Distribution, Int, IntDType, Tensor, TensorData};

use crate::data::TokenizerStore;
use crate::model::attention::LayerKv;
use crate::model::model::LlmModel;

/// Sampling and generation hyperparameters.
#[derive(Debug, Clone)]
pub struct GenerateConfig {
    /// Maximum number of new tokens to generate.
    pub max_tokens: usize,
    /// Temperature for scaling logits (higher = more random, lower = more deterministic).
    pub temperature: f64,
    /// Top-k filtering: keep only the top `k` highest-probability tokens.
    pub top_k: Option<usize>,
    /// Top-p (nucleus) filtering: keep the smallest set of tokens with cumulative probability >= `top_p`.
    pub top_p: Option<f64>,
    /// If true, use greedy decoding (argmax) ignoring temperature, top-k, and top-p.
    pub greedy: bool,
}

impl Default for GenerateConfig {
    fn default() -> Self {
        Self {
            max_tokens: 128,
            temperature: 0.7,
            top_k: Some(40),
            top_p: Some(0.9),
            greedy: false,
        }
    }
}

/// CPU-side next-token sampler over raw f32 logits (used by the GGUF engine,
/// which never materializes a burn tensor). Greedy, temperature, top-k and
/// top-p behave like [`sample_next_token`], sampling from a deterministic
/// xorshift PRNG seeded per call.
pub fn sample_next_token_cpu(logits: &[f32], config: &GenerateConfig) -> u32 {
    let vocab = logits.len();
    assert!(vocab > 0, "cannot sample from an empty logits vector");

    if config.greedy || config.temperature <= 0.0 {
        let mut best = (0usize, f32::NEG_INFINITY);
        for (i, &l) in logits.iter().enumerate() {
            if l > best.1 {
                best = (i, l);
            }
        }
        return best.0 as u32;
    }

    let t = config.temperature as f32;
    let scaled: Vec<f32> = logits.iter().map(|&l| l / t).collect();
    let k = config.top_k.map(|k| k.min(vocab)).unwrap_or(vocab);

    let mut cand: Vec<usize> = (0..vocab).collect();
    if k < vocab {
        cand.sort_unstable_by(|&a, &b| {
            scaled[b]
                .partial_cmp(&scaled[a])
                .unwrap_or(std::cmp::Ordering::Equal)
        });
        cand.truncate(k);
    }

    let maxv = cand
        .iter()
        .map(|&i| scaled[i])
        .fold(f32::NEG_INFINITY, f32::max);
    let mut probs: Vec<(usize, f32)> = cand
        .iter()
        .map(|&i| (i, (scaled[i] - maxv).exp()))
        .collect();
    let sum: f32 = probs.iter().map(|&(_, p)| p).sum();
    for (_, p) in probs.iter_mut() {
        *p /= sum;
    }

    // Top-p (nucleus): keep the smallest descending prefix whose cumulative
    // mass reaches `top_p`, always keeping at least the first token.
    if let Some(p) = config.top_p.filter(|&p| p > 0.0 && p < 1.0) {
        probs.sort_unstable_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
        let mut cum = 0.0f32;
        for (i, &(_, prob)) in probs.iter().enumerate() {
            cum += prob;
            if cum >= p as f32 && i + 1 < probs.len() {
                probs.truncate(i + 1);
                break;
            }
        }
    }

    let kept_sum: f32 = probs.iter().map(|&(_, p)| p).sum();
    let u = rng_f32() * kept_sum;
    let mut acc = 0.0f32;
    for &(id, p) in &probs {
        acc += p;
        if acc >= u {
            return id as u32;
        }
    }
    probs[0].0 as u32
}

/// Uniform `[0, 1)` f32 entropy from a tiny xorshift64* PRNG, seeded with the
/// system clock so successive generated runs differ.
fn rng_f32() -> f32 {
    use std::sync::atomic::{AtomicU64, Ordering};

    static COUNTER: AtomicU64 = AtomicU64::new(0);
    let nanos = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|d| d.as_nanos() as u64)
        .unwrap_or(0x9e37_79b9_7f4a_7c15);
    let mut state = COUNTER
        .fetch_add(0x9e37_79b9_7f4a_7c15, Ordering::Relaxed)
        .wrapping_add(nanos);
    state ^= state >> 12;
    state ^= state << 25;
    state ^= state >> 27;
    let x = state.wrapping_mul(0x2545_f491_4f6c_dd1d);
    (x >> 40) as f32 / (1u64 << 24) as f32
}

/// Sample the next token from raw logits of shape `[1, vocab_size]`.
pub fn sample_next_token<B: Backend>(
    logits: Tensor<B, 2>,
    config: &GenerateConfig,
    device: &B::Device,
) -> u32 {
    let vocab_size = logits.dims()[1];

    if config.greedy || config.temperature <= 0.0 {
        let chosen = logits.argmax(1).cast(IntDType::I32);
        let val: i64 = chosen.into_data().to_vec::<i32>().unwrap()[0] as i64;
        return val as u32;
    }

    // Scale by temperature
    let logits = logits.div_scalar(config.temperature as f32);

    // Apply top-k filtering if requested
    let (logits, indices) = if let Some(k) = config.top_k.map(|k| k.min(vocab_size)) {
        if k < vocab_size {
            let (top_vals, top_idx) = logits.topk_with_indices(k, 1);
            (top_vals, Some(top_idx))
        } else {
            (logits, None)
        }
    } else {
        (logits, None)
    };

    let probs = softmax(logits, 1);

    // Apply top-p (nucleus) filtering if requested
    let (probs, final_indices) = if let Some(p) = config.top_p.filter(|&p| p > 0.0 && p < 1.0) {
        // Sort probabilities descending
        let sorted_indices = probs.clone().argsort_descending(1);
        let sorted_indices = sorted_indices.cast(IntDType::I32);
        let sorted_probs = probs.gather(1, sorted_indices.clone());
        let cumsum = sorted_probs.clone().cumsum(1);

        // Mask out tokens where previous cumulative sum already >= p.
        // Keep at least the first token.
        let mask = cumsum.greater_elem(p as f32);
        let shifted_mask = Tensor::<B, 2, Bool>::cat(
            vec![
                Tensor::<B, 2, Bool>::zeros([1, 1], device),
                mask.clone()
                    .slice([0..1, 0..(mask.dims()[1].saturating_sub(1))]),
            ],
            1,
        );

        let filtered_probs = sorted_probs.mask_fill(shifted_mask, 0.0);
        let sum = filtered_probs.clone().sum_dim(1);
        let renorm_probs = filtered_probs.div(sum);

        // Map back through the indices
        let mapped_indices = match indices {
            Some(orig_idx) => orig_idx.gather(1, sorted_indices),
            None => sorted_indices,
        };
        (renorm_probs, Some(mapped_indices))
    } else {
        (probs, indices)
    };

    // Categorical sampling using cumulative distribution
    let cdf = probs.cumsum(1);
    let u = Tensor::<B, 2>::random([1, 1], Distribution::Uniform(0.0, 1.0), device);

    // Find the first index where CDF >= u
    let mask = cdf.greater_equal(u);
    // argmax on boolean returns the index of the first `true` value
    let sampled_slot = mask.int().argmax(1).cast(IntDType::I32);

    let chosen_token_id: i64 = match final_indices {
        Some(idx_tensor) => {
            let gathered = idx_tensor.gather(1, sampled_slot.clone());
            gathered.into_data().to_vec::<i32>().unwrap()[0] as i64
        }
        None => sampled_slot.into_data().to_vec::<i32>().unwrap()[0] as i64,
    };

    chosen_token_id as u32
}

/// Autoregressively generate text starting from `prompt`.
pub fn generate<B: Backend>(
    model: &LlmModel<B>,
    tokenizer: &TokenizerStore,
    prompt: &str,
    config: &GenerateConfig,
    max_seq_len: usize,
) -> Result<String> {
    let mut token_ids = tokenizer.encode_raw(prompt)?;
    if token_ids.is_empty() {
        anyhow::bail!("prompt produced zero tokens");
    }

    let device = model.model.embed_tokens.weight.val().device();
    let n_layers = model.model.layers.len();
    let mut cache: Vec<LayerKv<B>> = (0..n_layers).map(|_| LayerKv::new()).collect();

    // Prefill pass: run full prompt through the model to prime the KV cache
    let prompt_len = token_ids.len();
    let prefill_tensor = Tensor::<B, 2, Int>::from_data(
        TensorData::new(token_ids.clone(), [1, prompt_len]),
        &device,
    );
    let logits = model.forward_with_cache(prefill_tensor, Some(&mut cache), 0);
    let vocab_size = logits.dims()[2];
    let last_logit = logits
        .slice([0..1, (prompt_len - 1)..prompt_len, 0..vocab_size])
        .reshape([1, vocab_size]);

    let mut generated_ids = Vec::new();
    let mut next_token = sample_next_token(last_logit, config, &device);

    for _ in 0..config.max_tokens {
        if next_token == tokenizer.eos_id {
            break;
        }
        if token_ids.len() >= max_seq_len {
            log::warn!("reached max_seq_len ({max_seq_len}); stopping generation");
            break;
        }

        generated_ids.push(next_token);
        token_ids.push(next_token);

        let current_pos = token_ids.len() - 1;
        let step_tensor =
            Tensor::<B, 2, Int>::from_data(TensorData::new(vec![next_token], [1, 1]), &device);
        let step_logits = model.forward_with_cache(step_tensor, Some(&mut cache), current_pos);
        let step_logit = step_logits.reshape([1, vocab_size]);
        next_token = sample_next_token(step_logit, config, &device);
    }

    tokenizer.decode(&generated_ids, true)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::model::LlmModelConfig;
    use crate::train::TestBackend;

    #[test]
    fn sample_greedy_picks_highest_logit() {
        let device = burn::backend::flex::FlexDevice;
        let logits = Tensor::<TestBackend, 2>::from_data([[1.0, 5.0, 2.0, 0.5]], &device);
        let config = GenerateConfig {
            greedy: true,
            ..GenerateConfig::default()
        };
        let token = sample_next_token(logits, &config, &device);
        assert_eq!(token, 1);
    }

    #[test]
    fn sample_top_k_restricts_choices() {
        let device = burn::backend::flex::FlexDevice;
        // token 0 has high logit, token 1 slightly lower, others -100
        let logits = Tensor::<TestBackend, 2>::from_data([[10.0, 9.9, -100.0, -100.0]], &device);
        let config = GenerateConfig {
            temperature: 0.1,
            top_k: Some(2),
            greedy: false,
            ..GenerateConfig::default()
        };
        for _ in 0..20 {
            let token = sample_next_token(logits.clone(), &config, &device);
            assert!(token == 0 || token == 1, "sampled {token} not in top-2");
        }
    }

    #[test]
    fn cpu_greedy_picks_highest_logit() {
        let logits = [1.0, 5.0, 2.0, 0.5];
        let token = sample_next_token_cpu(&logits, &GenerateConfig::default());
        assert_eq!(token, 1);
    }

    #[test]
    fn cpu_top_k_restricts_choices() {
        let logits = [10.0, 9.9, -100.0, -100.0, -100.0];
        let config = GenerateConfig {
            temperature: 0.1,
            top_k: Some(2),
            greedy: false,
            ..GenerateConfig::default()
        };
        for _ in 0..64 {
            let token = sample_next_token_cpu(&logits, &config);
            assert!(token == 0 || token == 1, "sampled {token} not in top-2");
        }
    }

    #[test]
    fn generate_runs_end_to_end_on_tiny_model() {
        let config = LlmModelConfig::tiny();
        let device = burn::backend::flex::FlexDevice;
        let model = LlmModel::<TestBackend>::new(&config, &device);

        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("tokenizer.json");
        std::fs::write(
            &path,
            r#"{
                "version": "1.0",
                "model": {"type": "WordLevel", "vocab": {"[UNK]": 0, "hello": 1, "world": 2, "burn": 3, "</s>": 4},
                          "unk_token": "[UNK]"},
                "normalizer": null,
                "pre_tokenizer": {"type": "Whitespace"},
                "post_processor": null,
                "decoder": null,
                "added_tokens": []
            }"#,
        )
        .unwrap();
        let tokenizer = TokenizerStore::from_file(&path).unwrap();

        let gen_cfg = GenerateConfig {
            max_tokens: 5,
            greedy: true,
            ..GenerateConfig::default()
        };

        let output = generate(&model, &tokenizer, "hello", &gen_cfg, config.max_seq_len);
        assert!(output.is_ok(), "generate failed: {:?}", output.err());
    }
}
