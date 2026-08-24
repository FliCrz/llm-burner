use super::decoder::DecoderLayer;
use super::rms_norm::RmsNorm;

use burn::module::{Content, DisplaySettings, Module, ModuleDisplay};
use burn::nn::{Embedding, EmbeddingConfig, Linear, LinearConfig};
use burn::tensor::{Int, Tensor, backend::Backend};

/// Configuration describing the simplified Gemma-style transformer.
///
/// This is the single source of truth consumed by every sub-module. It is
/// serialized next to trained weights so checkpoints are self-describing.
#[derive(Debug, Clone, PartialEq, serde::Serialize, serde::Deserialize)]
pub struct LlmModelConfig {
    /// Hidden size used by embeddings and projections.
    pub d_model: usize,
    /// Number of decoder layers.
    pub n_layers: usize,
    /// Number of query heads.
    pub n_heads: usize,
    /// Number of key/value heads (grouped-query attention).
    pub n_kv_heads: usize,
    /// Dimension of every attention head.
    pub head_dim: usize,
    /// MLP intermediate size.
    pub intermediate_size: usize,
    /// Vocabulary size.
    pub vocab_size: usize,
    /// Max sequence length supported by the RoPE cache.
    pub max_seq_len: usize,
    /// Base for the rotary position embeddings.
    pub rope_theta: f64,
    /// Epsilon used by RMS norms.
    pub rms_eps: f64,
    /// Share the input embedding matrix with the output projection.
    pub tie_word_embeddings: bool,
    /// Give the query/key/value projections a bias term. Qwen2 trains
    /// non-zero attention biases; dropping them silently degrades every
    /// loaded Qwen checkpoint to gibberish output.
    #[serde(default)]
    pub qkv_bias: bool,
    /// Hugging Face `model_type` this configuration came from (`qwen2`,
    /// `llama`, `gemma`, ...). Selects the GGUF target architecture; empty
    /// for synthetic configs.
    #[serde(default)]
    pub hf_model_type: String,
    /// Sliding-window attention size, if the model uses it.
    pub sliding_window: Option<usize>,
    /// `true` -> GELU gated MLP (Gemma), `false` -> SiLU gated MLP.
    pub use_gelu: bool,
    /// Apply per-head query/key norms (Gemma 2/3).
    pub has_qk_norm: bool,
}

impl LlmModelConfig {
    /// Create the configuration from a parsed Hugging Face config.
    pub fn from_transformers(cfg: &crate::config::TransformersConfig) -> Self {
        Self {
            d_model: cfg.hidden_size,
            n_layers: cfg.num_hidden_layers,
            n_heads: cfg.num_attention_heads,
            n_kv_heads: cfg.num_key_value_heads,
            head_dim: cfg.attention_head_dim,
            intermediate_size: cfg.intermediate_size,
            vocab_size: cfg.vocab_size,
            max_seq_len: cfg.max_position_embeddings,
            rope_theta: cfg.rope_theta,
            rms_eps: cfg.rms_norm_eps,
            tie_word_embeddings: cfg.tie_word_embeddings,
            // Hugging Face's Qwen2Config defaults `attention_bias` to true
            // and most Qwen releases omit the key from config.json.
            qkv_bias: cfg.attention_bias.unwrap_or(cfg.model_type == "qwen2"),
            hf_model_type: cfg.model_type.clone(),
            sliding_window: cfg.sliding_window,
            use_gelu: cfg.hidden_act == "gelu" || cfg.hidden_act == "gelu_pytorch_tanh",
            has_qk_norm: cfg.has_qk_norm,
        }
    }

    /// The GGUF architecture string to use when exporting.
    ///
    /// Qwen2 must be exported under its own `qwen2` architecture — llama.cpp
    /// applies NEOX-style RoPE to unpermuted HF-layout Q/K weights for it,
    /// whereas the `llama` architecture expects interleaved (permuted) Q/K.
    /// Exporting a Qwen model as `llama` loads fine but generates garbage.
    pub fn gguf_architecture(&self) -> &'static str {
        if self.hf_model_type == "qwen2" {
            "qwen2"
        } else if self.has_qk_norm {
            "gemma"
        } else {
            "llama"
        }
    }

    /// Number of trainable parameters this configuration instantiates.
    ///
    /// Mirrors [`Transformer::new`] / [`LlmModel::new`] exactly (projection
    /// shapes, optional QKV biases, optional Q/K norms, tied embeddings) so
    /// callers can size a run — e.g. the memory pre-flight in the pipeline —
    /// without building the model. Verified against a real instantiation in
    /// the unit tests.
    pub fn param_count(&self) -> u64 {
        let d = self.d_model as u64;
        let q_out = (self.n_heads * self.head_dim) as u64;
        let kv_out = (self.n_kv_heads * self.head_dim) as u64;
        let i = self.intermediate_size as u64;
        let vocab = self.vocab_size as u64;

        // Attention: q/k/v write `d_model -> q_out|kv_out`, o_proj reads back
        // `q_out -> d_model`; only q/k/v carry biases.
        let mut attn = q_out * d + 2 * (kv_out * d) + q_out * d;
        if self.qkv_bias {
            attn += q_out + 2 * kv_out;
        }
        if self.has_qk_norm {
            attn += 2 * self.head_dim as u64;
        }
        // MLP: gate and up are `d_model -> intermediate`, down goes back.
        let mlp = 2 * (d * i) + i * d;
        // Two RMSNorm gains per decoder layer plus one final norm gain.
        let layer = attn + mlp + 2 * d;

        let mut n = d * vocab; // embed_tokens
        n += layer * self.n_layers as u64;
        n += d; // model.norm
        if !self.tie_word_embeddings {
            n += d * vocab; // lm_head
        }
        n
    }

    /// A small test configuration.
    pub fn tiny() -> Self {
        Self {
            d_model: 64,
            n_layers: 2,
            n_heads: 4,
            n_kv_heads: 2,
            head_dim: 16,
            intermediate_size: 128,
            vocab_size: 256,
            max_seq_len: 64,
            rope_theta: 10_000.0,
            rms_eps: 1e-6,
            tie_word_embeddings: false,
            qkv_bias: false,
            hf_model_type: String::new(),
            sliding_window: None,
            use_gelu: false,
            has_qk_norm: false,
        }
    }
}

/// The decoder-only transformer body: token embeddings, stacked decoder
/// layers, and a final RMS norm.
#[derive(Module, Debug)]
pub struct Transformer<B: Backend> {
    /// Token embedding matrix, shape `[vocab, d_model]`.
    pub embed_tokens: Embedding<B>,
    /// Stacked decoder layers.
    pub layers: Vec<DecoderLayer<B>>,
    /// Final normalization before the output projection.
    pub norm: RmsNorm<B>,

    /// Whether the output projection shares the embedding weights.
    pub tie_word_embeddings: bool,
    /// Whether the final norm is applied (gemma applies `hidden * (1 + ln)`
    /// only via the tied head; we always apply it).
    pub has_final_norm: bool,
}

impl<B: Backend> Transformer<B> {
    /// Build a transformer from a configuration.
    pub fn new(config: &LlmModelConfig, device: &B::Device) -> Self {
        let embed_tokens = EmbeddingConfig::new(config.vocab_size, config.d_model)
            .with_initializer(burn::module::Initializer::Normal {
                mean: 0.0,
                std: 0.02,
            })
            .init(device);

        let layers = (0..config.n_layers)
            .map(|_| {
                let self_attn = super::attention::CausalAttention::new(
                    config.d_model,
                    config.n_heads,
                    config.n_kv_heads,
                    config.head_dim,
                    config.rope_theta,
                    config.sliding_window,
                    config.has_qk_norm,
                    config.qkv_bias,
                    config.rms_eps,
                    device,
                );
                let mlp = super::mlp::Mlp::new(
                    config.d_model,
                    config.intermediate_size,
                    config.use_gelu,
                    device,
                );
                let input_layernorm = RmsNorm::new(config.d_model, config.rms_eps, device);
                let post_attention_layernorm = RmsNorm::new(config.d_model, config.rms_eps, device);
                DecoderLayer {
                    self_attn,
                    mlp,
                    input_layernorm,
                    post_attention_layernorm,
                }
            })
            .collect();

        let norm = RmsNorm::new(config.d_model, config.rms_eps, device);

        Self {
            embed_tokens,
            layers,
            norm,
            tie_word_embeddings: config.tie_word_embeddings,
            has_final_norm: true,
        }
    }

    /// Forward pass, returning hidden states of shape `[batch, seq, d_model]`.
    pub fn forward(&self, input: Tensor<B, 2, Int>) -> Tensor<B, 3> {
        let x = self.embed_tokens.forward(input);
        let mut x = x;
        for layer in &self.layers {
            x = layer.forward(x);
        }
        self.norm.forward(x)
    }

    /// Forward pass returning residual-stream hidden states taken directly
    /// after decoder layer `after_layer` (0-indexed), skipping the final norm.
    ///
    /// Used to probe intermediate activations (e.g. refusal-direction
    /// extraction); the regular [`Transformer::forward`] should be preferred
    /// everywhere else.
    pub fn forward_hidden_after_layer(
        &self,
        input: Tensor<B, 2, Int>,
        after_layer: usize,
    ) -> Tensor<B, 3> {
        assert!(
            after_layer < self.layers.len(),
            "layer index {} out of range (model has {} layers)",
            after_layer,
            self.layers.len()
        );
        let mut x = self.embed_tokens.forward(input);
        for layer in &self.layers[..=after_layer] {
            x = layer.forward(x);
        }
        x
    }
}

/// The full causal language model: transformer body plus output projection.
///
/// Field names mirror the Hugging Face checkpoint layout (`model.layers.N.*`,
/// `model.norm`, `lm_head`) so safetensors weights load without remapping.
#[derive(Module, Debug)]
#[module(custom_display)]
pub struct LlmModel<B: Backend> {
    /// The transformer body.
    pub model: Transformer<B>,
    /// Output projection to vocab logits. `None` when embeddings are tied.
    pub lm_head: Option<Linear<B>>,
}

impl<B: Backend> LlmModel<B> {
    /// Build a model from a configuration.
    pub fn new(config: &LlmModelConfig, device: &B::Device) -> Self {
        let model = Transformer::new(config, device);
        let lm_head = if config.tie_word_embeddings {
            None
        } else {
            Some(
                LinearConfig::new(config.d_model, config.vocab_size)
                    .with_bias(false)
                    .with_initializer(burn::module::Initializer::Normal {
                        mean: 0.0,
                        std: 0.02,
                    })
                    .init(device),
            )
        };
        Self { model, lm_head }
    }

    /// Forward pass over token indices, returning logits `[batch, seq, vocab]`.
    pub fn forward(&self, input: Tensor<B, 2, Int>) -> Tensor<B, 3> {
        let hidden = self.model.forward(input);
        match &self.lm_head {
            Some(head) => head.forward(hidden),
            None => hidden.matmul(
                self.model
                    .embed_tokens
                    .weight
                    .val()
                    .transpose()
                    .unsqueeze::<3>(),
            ),
        }
    }
}

impl<B: Backend> ModuleDisplay for LlmModel<B> {
    fn custom_settings(&self) -> Option<DisplaySettings> {
        DisplaySettings::new()
            .with_new_line_after_attribute(false)
            .optional()
    }

    fn custom_content(&self, content: Content) -> Option<Content> {
        content
            .add("layers", &self.model.layers.len())
            .add("params", &Module::num_params(self))
            .optional()
    }
}
