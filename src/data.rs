//! Text datasets: downloading Hugging Face text datasets, tokenization, and
//! building causal-LM batches for training.
//!
//! [`WindowStore`] keeps training windows in one flat arena so batches are
//! contiguous slices; tensor construction happens in [`build_causal_batch_from_flat`].

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

/// Flat token arena holding consecutive causal windows.
///
/// Windows produced by [`tokenize_corpus`] tile the token stream with stride ==
/// `seq_len`, so window `i` occupies `tokens[i*seq_len .. (i+1)*seq_len]` and a
/// batch of consecutive windows is a single contiguous slice. Building training
/// tensors therefore never materializes per-window `Vec`s.
#[derive(Debug, Clone)]
pub struct WindowStore {
    tokens: Vec<u32>,
    seq_len: usize,
}

impl WindowStore {
    /// Create an empty arena with the given window length.
    pub fn new(seq_len: usize) -> Self {
        assert!(seq_len > 0, "seq_len must be positive");
        Self {
            tokens: Vec::new(),
            seq_len,
        }
    }

    /// Number of complete windows held.
    pub fn len(&self) -> usize {
        self.tokens.len() / self.seq_len
    }

    /// True when no complete window is held.
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// Tokens per window.
    pub fn seq_len(&self) -> usize {
        self.seq_len
    }

    /// Total number of tokens stored (a multiple of `seq_len`).
    pub fn total_tokens(&self) -> usize {
        self.tokens.len()
    }

    /// Append whole windows from a contiguous buffer whose length is a
    /// multiple of `seq_len`.
    pub fn extend_windows(&mut self, flat: &[u32]) {
        debug_assert_eq!(
            flat.len() % self.seq_len,
            0,
            "expected whole windows, got {} tokens for seq_len {}",
            flat.len(),
            self.seq_len
        );
        self.tokens.extend_from_slice(flat);
    }

    /// Append one final window from a partial tail, right-padded with `pad_id`.
    pub fn push_padded_tail(&mut self, tokens: &[u32], pad_id: u32) {
        if tokens.is_empty() {
            return;
        }
        self.tokens.extend_from_slice(tokens);
        let rem = self.tokens.len() % self.seq_len;
        if rem > 0 {
            self.tokens.resize(self.tokens.len() + (self.seq_len - rem), pad_id);
        }
    }

    /// Full batches available for the given `batch_size`; a trailing partial
    /// batch is dropped so step counts stay exact.
    pub fn batch_count(&self, batch_size: usize) -> usize {
        if batch_size == 0 {
            return 0;
        }
        self.len() / batch_size
    }

    /// Contiguous token buffer covering `n_windows` consecutive windows
    /// starting at window index `start_window`.
    pub fn window_tokens(&self, start_window: usize, n_windows: usize) -> &[u32] {
        let from = start_window * self.seq_len;
        &self.tokens[from..from + n_windows * self.seq_len]
    }
}

/// Target size of the text slices handed to the tokenizer.
///
/// The HF tokenizers library materializes a full `Encoding` (per-token piece
/// strings, offsets, masks) for whatever it is asked to encode — measured at
/// roughly 100x the input size in RAM. Feeding it the corpus in ~1 MiB
/// line-aligned chunks keeps that overhead bounded regardless of file size.
const CHUNK_TARGET_BYTES: usize = 1024 * 1024;

/// Tokenize text files into a flat window arena with bounded memory use.
///
/// Files are read in [`CHUNK_TARGET_BYTES`] line-aligned chunks and encoded one
/// chunk at a time; a carry-over tail of fewer than `seq_len` tokens stitches
/// the token stream across chunk and file boundaries. Peak memory is therefore
/// O(chunk + window arena) instead of O(corpus), and windows are never
/// allocated individually.
///
/// Returns the arena and the total number of encoded tokens (before padding).
/// A newline separator is inserted between files (mirroring whole-corpus
/// concatenation). Subword merges cannot span chunk or file borders; with 1 MiB
/// chunks that only affects a handful of boundary tokens per megabyte.
pub fn tokenize_corpus(
    tokenizer: &TokenizerStore,
    files: &[PathBuf],
    seq_len: usize,
    pad_id: u32,
) -> Result<(WindowStore, usize)> {
    tokenize_corpus_sized(tokenizer, files, seq_len, pad_id, CHUNK_TARGET_BYTES)
}

/// [`tokenize_corpus`] with an explicit chunk target; exposed for tests.
fn tokenize_corpus_sized(
    tokenizer: &TokenizerStore,
    files: &[PathBuf],
    seq_len: usize,
    pad_id: u32,
    chunk_target_bytes: usize,
) -> Result<(WindowStore, usize)> {
    if seq_len == 0 {
        anyhow::bail!("seq-len must be at least 1");
    }
    // One truncation/padding-free clone for the whole run; encoding many
    // chunks through `encode_raw` would re-clone per call.
    let mut raw = tokenizer.tokenizer.clone();
    raw.with_truncation(None)
        .map_err(|e| anyhow::anyhow!("failed to clear truncation: {e}"))?;
    raw.with_padding(None);

    let mut store = WindowStore::new(seq_len);
    let mut carry: Vec<u32> = Vec::new();
    let mut total = 0usize;

    for file in files {
        let f = std::fs::File::open(file)
            .with_context(|| format!("failed to read `{}`", file.display()))?;
        let reader = std::io::BufReader::with_capacity(chunk_target_bytes.max(8192), f);
        let mut chunk = String::new();
        for line in std::io::BufRead::lines(reader) {
            let line = line.with_context(|| format!("failed to read `{}`", file.display()))?;
            chunk.push_str(&line);
            chunk.push('\n');
            if chunk.len() >= chunk_target_bytes {
                total += push_encoded(&raw, &chunk, &mut carry, &mut store)?;
                chunk.clear();
            }
        }
        if !chunk.is_empty() {
            total += push_encoded(&raw, &chunk, &mut carry, &mut store)?;
        }
        // Separator mirrors the previous whole-corpus reader, which joined
        // files with '\n'.
        if !carry.is_empty() || !store.is_empty() {
            total += push_encoded(&raw, "\n", &mut carry, &mut store)?;
        }
    }
    store.push_padded_tail(&carry, pad_id);

    Ok((store, total))
}

/// Encode `text`, append its ids to the carry buffer, and drain every
/// complete window into the arena. Returns the number of encoded tokens.
fn push_encoded(
    raw: &Tokenizer,
    text: &str,
    carry: &mut Vec<u32>,
    store: &mut WindowStore,
) -> Result<usize> {
    let encoding = raw
        .encode(text, true)
        .map_err(|e| anyhow::anyhow!("tokenizer failed: {e}"))?;
    let count = encoding.get_ids().len();
    carry.extend_from_slice(encoding.get_ids());
    let seq_len = store.seq_len();
    let complete = carry.len() - carry.len() % seq_len;
    if complete > 0 {
        store.extend_windows(&carry[..complete]);
        carry.drain(..complete);
    }
    Ok(count)
}

/// Build a causal-LM batch pair from a contiguous buffer of consecutive
/// windows on a backend.
///
/// `flat` must hold whole rows of `seq_len` tokens. Targets are each row
/// shifted left by one token; the last target position of every row is
/// `pad_id`.
pub fn build_causal_batch_from_flat<B: burn::tensor::backend::Backend>(
    flat: &[u32],
    seq_len: usize,
    device: &B::Device,
    pad_id: u32,
) -> CausalLmBatch<B> {
    use burn::tensor::{Tensor, TensorData};

    assert!(seq_len > 0, "seq_len must be positive");
    assert!(
        !flat.is_empty() && flat.len().is_multiple_of(seq_len),
        "batch buffer must hold whole rows, got {} tokens for seq_len {}",
        flat.len(),
        seq_len
    );
    let rows = flat.len() / seq_len;
    let mut targets = Vec::with_capacity(flat.len());
    for chunk in flat.chunks(seq_len) {
        targets.extend_from_slice(&chunk[1..]);
        targets.push(pad_id);
    }

    let input = Tensor::from_data(TensorData::new(flat.to_vec(), [rows, seq_len]), device);
    let target = Tensor::from_data(TensorData::new(targets, [rows, seq_len]), device);
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

#[cfg(test)]
mod tests {
    use super::*;

    type TestBackend = burn::backend::Flex<f32, i32>;

    /// A tiny whitespace/word-level tokenizer: `one`=10 `two`=11 `three`=12
    /// `four`=13; anything else maps to `[UNK]`=0. Newlines vanish under the
    /// whitespace pre-tokenizer.
    fn word_pseudo_tokenizer() -> TokenizerStore {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("tok.json");
        let vocab: ahash::AHashMap<String, u32> = [
            ("one", 10),
            ("two", 11),
            ("three", 12),
            ("four", 13),
            ("[UNK]", 0),
        ]
        .iter()
        .map(|(k, v)| (k.to_string(), *v))
        .collect();
        let mut tokenizer = Tokenizer::new(
            tokenizers::models::wordlevel::WordLevel::builder()
                .vocab(vocab)
                .unk_token("[UNK]".to_string())
                .build()
                .unwrap(),
        );
        tokenizer.with_pre_tokenizer(Some(tokenizers::pre_tokenizers::whitespace::Whitespace));
        std::fs::write(&path, serde_json::to_string(&tokenizer).unwrap()).unwrap();
        let store = TokenizerStore::from_file(&path).unwrap();
        drop(dir); // tokenizer already parsed into memory
        store
    }

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
    fn window_store_batches_are_contiguous_slices() {
        let mut store = WindowStore::new(4);
        store.extend_windows(&[1, 2, 3, 4]);
        store.extend_windows(&[5, 6, 7, 8]);
        store.push_padded_tail(&[9], 0);

        assert_eq!(store.len(), 3);
        assert_eq!(store.total_tokens(), 12);
        assert_eq!(store.window_tokens(0, 2), &[1, 2, 3, 4, 5, 6, 7, 8]);
        assert_eq!(store.window_tokens(2, 1), &[9, 0, 0, 0]);
        assert_eq!(store.batch_count(2), 1);
        assert_eq!(store.batch_count(4), 0);
    }

    #[test]
    fn flat_batch_shifts_targets_by_one() {
        let device = Default::default();
        let batch =
            build_causal_batch_from_flat::<TestBackend>(&[1, 2, 3, 4, 5, 6, 7, 8], 4, &device, 0);
        let input: Vec<i32> = batch.input.into_data().to_vec::<i32>().unwrap();
        let target: Vec<i32> = batch.target.into_data().to_vec::<i32>().unwrap();
        assert_eq!(input, vec![1, 2, 3, 4, 5, 6, 7, 8]);
        assert_eq!(target, vec![2, 3, 4, 0, 6, 7, 8, 0]);
    }

    #[test]
    fn tokenize_corpus_stitches_files_and_pads_tail() {
        let corpus_dir = tempfile::tempdir().unwrap();
        let a = corpus_dir.path().join("a.txt");
        let b = corpus_dir.path().join("b.txt");
        std::fs::write(&a, "one two").unwrap();
        std::fs::write(&b, "three four").unwrap();

        let tokenizer = word_pseudo_tokenizer();
        let (store, total) = tokenize_corpus(&tokenizer, &[a, b], 3, 0).unwrap();

        // Stream: [one two] [three four] -> ids [10,11,12,13]; '\n' yields no
        // tokens under the whitespace pre-tokenizer.
        assert_eq!(total, 4);
        assert_eq!(store.seq_len(), 3);
        // One complete window + one padded tail window.
        assert_eq!(store.len(), 2);
        // The first window spans the file boundary thanks to the carry-over.
        assert_eq!(store.window_tokens(0, 1), &[10, 11, 12]);
        assert_eq!(store.window_tokens(1, 1), &[13, 0, 0]);
    }

    #[test]
    fn tokenize_corpus_chunks_stitch_within_a_file() {
        let corpus_dir = tempfile::tempdir().unwrap();
        let a = corpus_dir.path().join("a.txt");
        std::fs::write(&a, "one two\nthree four\n").unwrap();

        let tokenizer = word_pseudo_tokenizer();
        // 4-byte chunk target forces an encode call per line.
        let (store, total) =
            tokenize_corpus_sized(&tokenizer, &[a], 3, 0, 4).unwrap();

        // Same stream as the whole-file encoding of "one two\nthree four".
        assert_eq!(total, 4);
        assert_eq!(store.len(), 2);
        assert_eq!(store.window_tokens(0, 1), &[10, 11, 12]);
        assert_eq!(store.window_tokens(1, 1), &[13, 0, 0]);
    }

    #[test]
    fn empty_input_yields_no_windows() {
        assert!(sliding_windows(&[], 4, 4, 0).is_empty());
    }
}
