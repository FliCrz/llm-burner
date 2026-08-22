//! Text datasets: downloading Hugging Face text datasets, tokenization, and
//! building causal-LM batches for training.
//!
//! Pure token-level helpers (`sliding_windows`, `pack_batches`) are backend
//! agnostic and unit-tested; tensor construction happens in
//! [`build_causal_batch`].

use std::path::{Path, PathBuf};

use anyhow::{Context, Result};
use hf_hub::{HFClientSync, HFRepositorySync, RepoTypeDataset};
use tokenizers::{
    PaddingDirection, PaddingParams, PaddingStrategy, Tokenizer, TruncationDirection,
    TruncationParams,
};

use crate::model::CausalLmBatch;

/// A reference to a Hugging Face dataset repository (`owner/name`).
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct HfDataset {
    /// Repository owner (user or organization).
    pub owner: String,
    /// Repository name.
    pub name: String,
}

impl HfDataset {
    /// Parse a dataset id of the form `owner/name`.
    pub fn parse(id: &str) -> Result<Self> {
        let (owner, name) = id.split_once('/').ok_or_else(|| {
            anyhow::anyhow!("expected a dataset id of the form `owner/name`, got `{id}`")
        })?;
        if owner.is_empty() || name.is_empty() {
            return Err(anyhow::anyhow!(
                "dataset id `{id}` must have a non-empty owner and name"
            ));
        }
        Ok(Self {
            owner: owner.to_string(),
            name: name.to_string(),
        })
    }

    /// The full `owner/name` identifier.
    pub fn id(&self) -> String {
        format!("{}/{}", self.owner, self.name)
    }

    /// A blocking handle to this dataset repository.
    pub fn client(&self) -> Result<HFRepositorySync<RepoTypeDataset>> {
        let client = HFClientSync::new().context("failed to create the Hugging Face client")?;
        Ok(client.dataset(&self.owner, &self.name))
    }

    /// Download the matching files into `target_dir`. Returns `target_dir`.
    pub fn download_snapshot(
        &self,
        target_dir: &Path,
        allow: &[String],
        ignore: &[String],
        max_workers: usize,
    ) -> Result<PathBuf> {
        let repo = self.client()?;
        std::fs::create_dir_all(target_dir)
            .with_context(|| format!("failed to create `{}`", target_dir.display()))?;
        repo.snapshot_download()
            .maybe_allow_patterns(if allow.is_empty() {
                None
            } else {
                Some(allow.to_vec())
            })
            .maybe_ignore_patterns(if ignore.is_empty() {
                None
            } else {
                Some(ignore.to_vec())
            })
            .local_dir(target_dir.to_path_buf())
            .max_workers(max_workers)
            .send()
            .with_context(|| format!("failed to download dataset `{}`", self.id()))?;
        Ok(target_dir.to_path_buf())
    }
}

/// A loaded Hugging Face tokenizer, configured for fixed-length causal-LM
/// windows with truncation on the right and right-side padding.
pub struct TokenizerStore {
    tokenizer: Tokenizer,
    /// Padding token id (used as the target for the shifted last position).
    pub pad_id: u32,
    /// End-of-sequence token id.
    pub eos_id: u32,
    /// Number of tokens per window.
    pub seq_len: usize,
    /// Jinja chat template from the sibling `tokenizer_config.json`, if any.
    /// Exported as `tokenizer.chat_template` so runtimes (llama-cli, llama
    /// server) prompt the model with the format it was trained on instead of
    /// a fallback template.
    pub chat_template: Option<String>,
}

impl TokenizerStore {
    /// Load a tokenizer from its `tokenizer.json` file. A sibling
    /// `tokenizer_config.json` is consulted for the chat template when
    /// present; its absence is not an error.
    pub fn from_file(path: &Path) -> Result<Self> {
        let tokenizer = Tokenizer::from_file(path).map_err(|e| {
            anyhow::anyhow!("failed to load tokenizer from `{}`: {e}", path.display())
        })?;
        let pad_id = tokenizer
            .token_to_id("<pad>")
            .or_else(|| tokenizer.token_to_id("<unk>"))
            .or_else(|| tokenizer.token_to_id("<eos>"))
            .unwrap_or(0);
        let eos_id = tokenizer
            .token_to_id("</s>")
            .or_else(|| tokenizer.token_to_id("<eos>"))
            .or_else(|| tokenizer.token_to_id("<|endoftext|>"))
            .unwrap_or(0);
        let config_path = path
            .parent()
            .unwrap_or(Path::new("."))
            .join("tokenizer_config.json");
        let chat_template = std::fs::read_to_string(&config_path)
            .ok()
            .and_then(|raw| serde_json::from_str::<serde_json::Value>(&raw).ok())
            .and_then(|config| {
                config
                    .get("chat_template")
                    .and_then(serde_json::Value::as_str)
                    .filter(|t| !t.is_empty())
                    .map(str::to_string)
            });
        Ok(Self {
            tokenizer,
            pad_id,
            eos_id,
            seq_len: 0,
            chat_template,
        })
    }

    /// Configure truncation and fixed-length padding of `seq_len` tokens.
    pub fn set_seq_len(&mut self, seq_len: usize) {
        self.tokenizer
            .with_truncation(Some(TruncationParams {
                max_length: seq_len,
                strategy: tokenizers::TruncationStrategy::LongestFirst,
                stride: 0,
                direction: TruncationDirection::Right,
            }))
            .expect("truncation configuration is valid");
        self.tokenizer.with_padding(Some(PaddingParams {
            strategy: PaddingStrategy::Fixed(seq_len),
            direction: PaddingDirection::Right,
            pad_to_multiple_of: None,
            pad_id: self.pad_id,
            pad_type_id: 0,
            pad_token: "[PAD]".to_string(),
        }));
        self.seq_len = seq_len;
    }

    /// Tokenize a single text into ids (padded/truncated to `seq_len`).
    pub fn encode(&self, text: &str) -> Result<Vec<u32>> {
        self.encode_batch(&[text]).map(|mut rows| rows.remove(0))
    }

    /// Tokenize an arbitrarily long text with no padding or truncation, for
    /// whole-corpus windowing.
    pub fn encode_raw(&self, text: &str) -> Result<Vec<u32>> {
        let mut tokenizer = self.tokenizer.clone();
        tokenizer
            .with_truncation(None)
            .map_err(|e| anyhow::anyhow!("failed to clear truncation: {e}"))?;
        tokenizer.with_padding(None);
        tokenizer
            .encode(text, true)
            .map(|enc| enc.get_ids().to_vec())
            .map_err(|e| anyhow::anyhow!("tokenizer failed: {e}"))
    }

    /// Tokenize many texts into rows of exactly `seq_len` ids.
    pub fn encode_batch(&self, texts: &[&str]) -> Result<Vec<Vec<u32>>> {
        let encodings = self
            .tokenizer
            .encode_batch(texts.to_vec(), true)
            .map_err(|e| anyhow::anyhow!("tokenizer batch encoding failed: {e}"))?;
        Ok(encodings.iter().map(|e| e.get_ids().to_vec()).collect())
    }

    /// The full vocabulary as `(token, id)` pairs ordered by id, suitable for
    /// writing GGUF `tokenizer.ggml.*` metadata.
    pub fn vocab_ordered(&self) -> Vec<(String, u32)> {
        let mut vocab: Vec<(String, u32)> = self.tokenizer.get_vocab(true).into_iter().collect();
        vocab.sort_by_key(|(_, id)| *id);
        vocab
    }

    /// GGUF `tokenizer.ggml.token_type` values ordered by token id.
    ///
    /// llama.cpp maps type 0 (`UNDEFINED`) to a token attribute that matches
    /// none of its piece-conversion branches, so every such token detokenizes
    /// to an empty string — models exported with all-zero types produce no
    /// visible output. Tokens must therefore be classified explicitly.
    pub fn token_types(&self) -> Vec<u32> {
        const TOKEN_TYPE_NORMAL: u32 = 1;
        const TOKEN_TYPE_UNKNOWN: u32 = 2;
        const TOKEN_TYPE_CONTROL: u32 = 3;
        const TOKEN_TYPE_USER_DEFINED: u32 = 4;
        const TOKEN_TYPE_BYTE: u32 = 6;

        let added = self.tokenizer.get_added_tokens_decoder();
        self.vocab_ordered()
            .iter()
            .map(|(token, id)| match added.get(id) {
                // `<unk>` keeps its conventional UNKNOWN class; other
                // special added tokens (<s>, </s>, <|endoftext|>, ...)
                // are control tokens.
                Some(added) if added.special => {
                    if token == "<unk>" {
                        TOKEN_TYPE_UNKNOWN
                    } else {
                        TOKEN_TYPE_CONTROL
                    }
                }
                Some(_) => TOKEN_TYPE_USER_DEFINED,
                // Byte-fallback tokens `<0xNN>` must be BYTE so llama.cpp
                // decodes them as raw bytes instead of literal text.
                None if is_byte_fallback_token(token) => TOKEN_TYPE_BYTE,
                None => TOKEN_TYPE_NORMAL,
            })
            .collect()
    }
}

/// True for byte-fallback tokens of the form `<0xNN>` (e.g. `<0x0A>`).
fn is_byte_fallback_token(token: &str) -> bool {
    token.len() == 6
        && token.starts_with("<0x")
        && token.ends_with('>')
        && token[3..5].chars().all(|c| c.is_ascii_hexdigit())
}

/// Split a flat token sequence into causal window rows using a sliding window
/// with the given stride. Every window has exactly `seq_len` tokens; windows are
/// padded with `pad_id` on the right when the tail is shorter than `seq_len`.
pub fn sliding_windows(
    tokens: &[u32],
    seq_len: usize,
    stride: usize,
    pad_id: u32,
) -> Vec<Vec<u32>> {
    let mut rows = Vec::new();
    if tokens.is_empty() || seq_len == 0 {
        return rows;
    }
    let stride = stride.max(1);
    let mut start = 0;
    while start < tokens.len() {
        let end = (start + seq_len).min(tokens.len());
        let mut row = tokens[start..end].to_vec();
        row.resize(seq_len, pad_id);
        rows.push(row);
        start += stride;
        if end >= tokens.len() {
            break;
        }
    }
    rows
}

/// Shift every window left by one token to produce targets: the target for the
/// first `seq_len - 1` positions is the next token; the last is `pad_id`.
pub fn shifted_targets(windows: &[Vec<u32>], pad_id: u32) -> Vec<Vec<u32>> {
    windows
        .iter()
        .map(|w| {
            let mut target = w[1..].to_vec();
            target.push(pad_id);
            target
        })
        .collect()
}

/// Build a causal-LM batch pair from token windows on a backend.
pub fn build_causal_batch<B: burn::tensor::backend::Backend>(
    windows: &[Vec<u32>],
    device: &B::Device,
    pad_id: u32,
) -> CausalLmBatch<B> {
    use burn::tensor::{Tensor, TensorData};

    let seq_len = windows[0].len();
    let batch_size = windows.len();
    let flat: Vec<u32> = windows.iter().flatten().copied().collect();
    let targets: Vec<u32> = shifted_targets(windows, pad_id)
        .into_iter()
        .flatten()
        .collect();

    let input = Tensor::from_data(TensorData::new(flat, [batch_size, seq_len]), device);
    let target = Tensor::from_data(TensorData::new(targets, [batch_size, seq_len]), device);
    CausalLmBatch::new(input, target)
}

/// Recursively collect files under `dir` with one of the given extensions.
pub fn collect_text_files(dir: &Path, extensions: &[&str]) -> Vec<PathBuf> {
    let mut out = Vec::new();
    let mut stack = vec![dir.to_path_buf()];
    while let Some(current) = stack.pop() {
        let Ok(entries) = std::fs::read_dir(&current) else {
            continue;
        };
        for entry in entries.flatten() {
            let path = entry.path();
            if path.is_dir() {
                stack.push(path);
            } else {
                let ext = path.extension().map(|e| e.to_string_lossy().to_lowercase());
                if ext.as_deref().is_some_and(|e| extensions.contains(&e)) {
                    out.push(path);
                }
            }
        }
    }
    out.sort();
    out
}

/// Read a list of text files, concatenating their contents with newlines.
pub fn read_corpus(paths: &[PathBuf]) -> Result<String> {
    let mut corpus = String::new();
    for path in paths {
        let text = std::fs::read_to_string(path)
            .with_context(|| format!("failed to read `{}`", path.display()))?;
        corpus.push_str(&text);
        corpus.push('\n');
    }
    Ok(corpus)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parses_dataset_id() {
        let ds = HfDataset::parse("wikitext/wikitext-2-raw-v1").unwrap();
        assert_eq!(ds.owner, "wikitext");
        assert_eq!(ds.name, "wikitext-2-raw-v1");
        assert!(HfDataset::parse("no-slash").is_err());
    }

    #[test]
    fn sliding_windows_pad_the_tail() {
        let tokens = vec![1, 2, 3, 4, 5];
        let rows = sliding_windows(&tokens, 4, 4, 0);
        assert_eq!(rows, vec![vec![1, 2, 3, 4], vec![5, 0, 0, 0]]);
    }

    #[test]
    fn sliding_windows_honor_stride() {
        let tokens = vec![1, 2, 3, 4, 5, 6, 7, 8, 9];
        let rows = sliding_windows(&tokens, 4, 2, 9);
        assert_eq!(
            rows,
            vec![
                vec![1, 2, 3, 4],
                vec![3, 4, 5, 6],
                vec![5, 6, 7, 8],
                vec![7, 8, 9, 9]
            ]
        );
    }

    #[test]
    fn shifted_targets_use_pad_for_last_position() {
        let windows = vec![vec![1, 2, 3, 4], vec![5, 6, 7, 8]];
        assert_eq!(
            shifted_targets(&windows, 0),
            vec![vec![2, 3, 4, 0], vec![6, 7, 8, 0]]
        );
    }

    #[test]
    fn empty_input_yields_no_windows() {
        assert!(sliding_windows(&[], 4, 4, 0).is_empty());
    }
}
