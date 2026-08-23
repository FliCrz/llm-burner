use serde_json::Value;
use std::path::Path;

/// Errors produced while parsing a Hugging Face `config.json` into an
/// [`TransformersConfig`].
#[derive(Debug, thiserror::Error)]
pub enum ConfigError {
    #[error("failed to read config file `{path}`: {source}")]
    Io {
        path: String,
        #[source]
        source: std::io::Error,
    },
    #[error("failed to parse config file `{path}`: {source}")]
    Json {
        path: String,
        #[source]
        source: serde_json::Error,
    },
    #[error("missing required field `{field}` in config")]
    MissingField { field: String },
    #[error("field `{field}` has an unexpected type: expected `{expected}`")]
    BadType {
        field: String,
        expected: &'static str,
    },
}

/// A parsed, simplified view over a Hugging Face `config.json` for the
/// Gemma-family of decoder-only transformers (Llama, Qwen, SmolLM, TinyLlama,
/// Gemma).
///
/// The raw config differs between model families:
/// - Most models expose the decoder fields at the top level.
/// - `google/gemma-4-E2B` nests them inside a `text_config` object.
///
/// [`TransformersConfig::from_path`] handles both layouts.
#[derive(Debug, Clone, PartialEq, serde::Serialize)]
pub struct TransformersConfig {
    /// Name reported by `architectures[0]`, e.g. `Qwen2ForCausalLM`.
    pub architecture: String,
    /// `model_type`, e.g. `qwen2`, `llama`, `gemma`.
    pub model_type: String,
    /// Hidden layer dimension (`hidden_size`). Used by embeddings, QKV/Q projections
    /// and layer norms.
    pub hidden_size: usize,
    /// MLP hidden dimension (`intermediate_size`).
    pub intermediate_size: usize,
    /// Number of attention heads.
    pub num_attention_heads: usize,
    /// Number of KV heads used by grouped-query attention. Defaults to
    /// `num_attention_heads` when not present.
    pub num_key_value_heads: usize,
    /// Number of decoder layers.
    pub num_hidden_layers: usize,
    /// Per-head exponent dim. Defaults to `hidden_size / num_attention_heads`.
    pub attention_head_dim: usize,
    /// Vocabulary size.
    pub vocab_size: usize,
    /// Maximum sequence length the model was trained with.
    pub max_position_embeddings: usize,
    /// Epsilon for RMSNorm.
    pub rms_norm_eps: f64,
    /// Base for the rotary position embeddings.
    pub rope_theta: f64,
    /// Whether input/output embeddings share weights.
    pub tie_word_embeddings: bool,
    /// Sliding-window attention window size, if the model uses it.
    pub sliding_window: Option<usize>,
    /// MLP activation function. `silu` -> GEGLU, `gelu` -> GELU.
    pub hidden_act: String,
    /// Whether the query/key/value projections carry a bias term (Qwen2
    /// trains non-zero attention biases; Llama/Gemma/TinyLlama do not).
    pub attention_bias: Option<bool>,
    /// Whether the model applies extra query/key norms (Gemma 2/3).
    pub has_qk_norm: bool,
}

impl TransformersConfig {
    /// Parse a config from a local `config.json` path.
    pub fn from_path(path: impl AsRef<Path>) -> Result<Self, ConfigError> {
        let path = path.as_ref();
        let text = std::fs::read_to_string(path).map_err(|source| ConfigError::Io {
            path: path.display().to_string(),
            source,
        })?;
        let value: Value = serde_json::from_str(&text).map_err(|source| ConfigError::Json {
            path: path.display().to_string(),
            source,
        })?;
        Self::from_value(&value)
    }

    /// Parse a config from a pre-parsed JSON value.
    ///
    /// When a `text_config` object is present it is merged on top of the
    /// top-level object, matching the Gemma-4-E2B layout.
    pub fn from_value(value: &Value) -> Result<Self, ConfigError> {
        let merged = merge_text_config(value);
        let object = merged.as_object().ok_or(ConfigError::BadType {
            field: "$".to_string(),
            expected: "object",
        })?;

        let architectures = get_strings(object, "architectures")?;
        let architecture = architectures
            .first()
            .cloned()
            .unwrap_or_else(|| "UnknownForCausalLM".to_string());

        let model_type = get_string(object, "model_type")?;

        let hidden_size = get_usize(object, "hidden_size")?;
        let num_attention_heads = get_usize(object, "num_attention_heads")?;
        let num_key_value_heads =
            get_usize_opt(object, "num_key_value_heads")?.unwrap_or(num_attention_heads);
        let num_hidden_layers = get_usize(object, "num_hidden_layers")?;
        let vocab_size = get_usize(object, "vocab_size")?;

        let attention_head_dim =
            get_usize_opt(object, "head_dim")?.unwrap_or(hidden_size / num_attention_heads);

        let has_qk_norm = model_type.contains("gemma");

        Ok(Self {
            architecture,
            model_type,
            hidden_size,
            intermediate_size: get_usize(object, "intermediate_size")?,
            num_attention_heads,
            num_key_value_heads,
            num_hidden_layers,
            attention_head_dim,
            vocab_size,
            max_position_embeddings: get_usize(object, "max_position_embeddings")?,
            rms_norm_eps: get_f64_opt(object, "rms_norm_eps")?.unwrap_or(1e-5),
            rope_theta: get_f64_opt(object, "rope_theta")?.unwrap_or(10_000.0),
            tie_word_embeddings: get_bool_opt(object, "tie_word_embeddings")?.unwrap_or(false),
            sliding_window: get_usize_opt(object, "sliding_window")?,
            hidden_act: get_string_opt(object, "hidden_act")?.unwrap_or_else(|| "silu".into()),
            attention_bias: get_bool_opt(object, "attention_bias")?,
            has_qk_norm,
        })
    }

    /// GGUF architecture name for export (`gemma` for gemma model types,
    /// `llama` otherwise, matching llama.cpp's naming).
    pub fn gguf_architecture(&self) -> &'static str {
        if self.model_type.starts_with("gemma") {
            "gemma"
        } else {
            "llama"
        }
    }

    /// Human readable name for the `list-models` command.
    pub fn name(&self) -> String {
        format!("{} ({})", self.architecture, self.model_type)
    }
}

/// If a `text_config` object exists at the top level, merge it over the
/// top-level object. Explicit fields in `text_config` win.
fn merge_text_config(value: &Value) -> Value {
    let Some(obj) = value.as_object() else {
        return value.clone();
    };
    let Some(text) = obj.get("text_config") else {
        return value.clone();
    };
    if let Some(text_obj) = text.as_object() {
        let mut merged = obj.clone();
        for (key, field) in text_obj {
            merged.insert(key.clone(), field.clone());
        }
        Value::Object(merged)
    } else {
        value.clone()
    }
}

fn get_string(obj: &serde_json::Map<String, Value>, key: &str) -> Result<String, ConfigError> {
    match obj.get(key) {
        Some(Value::String(s)) => Ok(s.clone()),
        Some(_) => Err(ConfigError::BadType {
            field: key.to_string(),
            expected: "string",
        }),
        None => Err(ConfigError::MissingField {
            field: key.to_string(),
        }),
    }
}

fn get_string_opt(
    obj: &serde_json::Map<String, Value>,
    key: &str,
) -> Result<Option<String>, ConfigError> {
    match obj.get(key) {
        Some(Value::String(s)) => Ok(Some(s.clone())),
        Some(_) => Err(ConfigError::BadType {
            field: key.to_string(),
            expected: "string",
        }),
        None => Ok(None),
    }
}

fn get_strings(
    obj: &serde_json::Map<String, Value>,
    key: &str,
) -> Result<Vec<String>, ConfigError> {
    match obj.get(key) {
        Some(Value::Array(items)) => items
            .iter()
            .map(|item| match item {
                Value::String(s) => Ok(s.clone()),
                _ => Err(ConfigError::BadType {
                    field: key.to_string(),
                    expected: "array of strings",
                }),
            })
            .collect(),
        Some(_) => Err(ConfigError::BadType {
            field: key.to_string(),
            expected: "array of strings",
        }),
        None => Ok(Vec::new()),
    }
}

fn get_usize(obj: &serde_json::Map<String, Value>, key: &str) -> Result<usize, ConfigError> {
    match obj.get(key) {
        Some(Value::Number(n)) if n.is_u64() => Ok(n.as_u64().unwrap() as usize),
        Some(_) => Err(ConfigError::BadType {
            field: key.to_string(),
            expected: "unsigned integer",
        }),
        None => Err(ConfigError::MissingField {
            field: key.to_string(),
        }),
    }
}

fn get_usize_opt(
    obj: &serde_json::Map<String, Value>,
    key: &str,
) -> Result<Option<usize>, ConfigError> {
    match obj.get(key) {
        Some(Value::Number(n)) if n.is_u64() => Ok(Some(n.as_u64().unwrap() as usize)),
        Some(_) => Err(ConfigError::BadType {
            field: key.to_string(),
            expected: "unsigned integer",
        }),
        None => Ok(None),
    }
}

fn get_f64_opt(
    obj: &serde_json::Map<String, Value>,
    key: &str,
) -> Result<Option<f64>, ConfigError> {
    match obj.get(key) {
        Some(Value::Number(n)) => match n.as_f64() {
            Some(v) => Ok(Some(v)),
            None => Err(ConfigError::BadType {
                field: key.to_string(),
                expected: "number",
            }),
        },
        Some(_) => Err(ConfigError::BadType {
            field: key.to_string(),
            expected: "number",
        }),
        None => Ok(None),
    }
}

fn get_bool_opt(
    obj: &serde_json::Map<String, Value>,
    key: &str,
) -> Result<Option<bool>, ConfigError> {
    match obj.get(key) {
        Some(Value::Bool(b)) => Ok(Some(*b)),
        Some(_) => Err(ConfigError::BadType {
            field: key.to_string(),
            expected: "boolean",
        }),
        None => Ok(None),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    #[test]
    fn parses_flat_llama_config() {
        let config = TransformersConfig::from_value(&json!({
            "architectures": ["LlamaForCausalLM"],
            "model_type": "llama",
            "hidden_size": 576,
            "intermediate_size": 1536,
            "num_attention_heads": 12,
            "num_hidden_layers": 4,
            "vocab_size": 32000,
            "max_position_embeddings": 2048,
            "rms_norm_eps": 1e-5,
            "tie_word_embeddings": false,
        }))
        .unwrap();

        assert_eq!(config.architecture, "LlamaForCausalLM");
        assert_eq!(config.model_type, "llama");
        assert_eq!(config.hidden_size, 576);
        assert_eq!(config.num_key_value_heads, 12);
        assert_eq!(config.attention_head_dim, 48);
        assert_eq!(config.rope_theta, 10_000.0);
        assert_eq!(config.hidden_act, "silu");
        assert!(!config.has_qk_norm);
        assert_eq!(config.sliding_window, None);
        assert_eq!(config.gguf_architecture(), "llama");
    }

    #[test]
    fn parses_qwen_config_with_gqa() {
        let config = TransformersConfig::from_value(&json!({
            "architectures": ["Qwen2ForCausalLM"],
            "model_type": "qwen2",
            "hidden_size": 896,
            "intermediate_size": 4864,
            "num_attention_heads": 14,
            "num_key_value_heads": 2,
            "num_hidden_layers": 24,
            "vocab_size": 151936,
            "max_position_embeddings": 32768,
            "rms_norm_eps": 1e-6,
            "rope_theta": 1000000.0,
            "tie_word_embeddings": false,
        }))
        .unwrap();

        assert_eq!(config.num_key_value_heads, 2);
        assert_eq!(config.attention_head_dim, 64);
        assert_eq!(config.rope_theta, 1_000_000.0);
    }

    #[test]
    fn parses_nested_gemma_text_config() {
        let config = TransformersConfig::from_value(&json!({
            "architectures": "Gemma4ForCausalLM100B",
            "model_type": "gemma-4",
            "attention_dropout": 0.0,
            "text_config": {
                "architectures": ["Gemma4ForCausalLM100B"],
                "attention_head_dim": 256,
                "head_dim": 256,
                "hidden_size": 1536,
                "intermediate_size": 6144,
                "max_position_embeddings": 131072,
                "num_attention_heads": 8,
                "num_hidden_layers": 35,
                "num_key_value_heads": 1,
                "rms_norm_eps": 1e-6,
                "rope_theta": 1000000.0,
                "sliding_window": 512,
                "tie_word_embeddings": true,
                "vocab_size": 262144,
                "hidden_act": "gelu"
            }
        }))
        .unwrap();

        assert_eq!(config.model_type, "gemma-4");
        assert_eq!(config.hidden_size, 1536);
        assert_eq!(config.intermediate_size, 6144);
        assert_eq!(config.num_key_value_heads, 1);
        assert_eq!(config.attention_head_dim, 256);
        assert_eq!(config.vocab_size, 262_144);
        assert_eq!(config.max_position_embeddings, 131_072);
        assert_eq!(config.sliding_window, Some(512));
        assert!(config.tie_word_embeddings);
        assert_eq!(config.hidden_act, "gelu");
        assert!(config.has_qk_norm);
        assert_eq!(config.gguf_architecture(), "gemma");
    }

    #[test]
    fn attention_bias_defaults_follow_model_family() {
        // Qwen2Config defaults attention_bias to true; most Qwen releases
        // omit the key entirely.
        let qwen = TransformersConfig::from_value(&json!({
            "model_type": "qwen2",
            "hidden_size": 896,
            "intermediate_size": 4864,
            "num_attention_heads": 14,
            "num_hidden_layers": 24,
            "vocab_size": 151936,
            "max_position_embeddings": 32768,
        }))
        .unwrap();
        assert_eq!(qwen.attention_bias, None);

        let llama = TransformersConfig::from_value(&json!({
            "model_type": "llama",
            "hidden_size": 576,
            "intermediate_size": 1536,
            "num_attention_heads": 12,
            "num_hidden_layers": 4,
            "vocab_size": 32000,
            "max_position_embeddings": 2048,
        }))
        .unwrap();
        assert_eq!(llama.attention_bias, None);

        let explicit = TransformersConfig::from_value(&json!({
            "model_type": "qwen2",
            "attention_bias": false,
            "hidden_size": 896,
            "intermediate_size": 4864,
            "num_attention_heads": 14,
            "num_hidden_layers": 24,
            "vocab_size": 151936,
            "max_position_embeddings": 32768,
        }))
        .unwrap();
        assert_eq!(explicit.attention_bias, Some(false));
    }

    #[test]
    fn missing_required_field_errors() {
        let err = TransformersConfig::from_value(&json!({
            "model_type": "llama",
            "hidden_size": 576,
            "intermediate_size": 1536,
            "num_attention_heads": 12,
            "num_hidden_layers": 4,
            "vocab_size": 32000
        }))
        .unwrap_err();
        match err {
            ConfigError::MissingField { ref field } => {
                assert_eq!(field, "max_position_embeddings")
            }
            other => panic!("unexpected error: {other:?}"),
        }
    }

    #[test]
    fn parses_errors_path() {
        assert!(TransformersConfig::from_path("definitely-not-a-file.json").is_err());
    }
}
