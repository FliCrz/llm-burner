//! llm-burner: fine-tune a simplified Gemma-style transformer with Burn and
//! export to safetensors and GGUF.

pub mod chat;
pub mod config;
pub mod data;
pub mod export;
pub mod generate;
pub mod hf;
pub mod model;
pub mod pipeline;
pub mod probe;
pub mod train;
pub mod ui;

pub use model::{CausalLmBatch, CausalLmOutput, LlmModel, LlmModelConfig};
