//! Download Hugging Face model repositories to a local directory.
//!
//! Uses the blocking (`HFClientSync`) API from `hf-hub` 1.0. Repositories are
//! downloaded with `snapshot_download` into a caller-provided directory, so
//! no HF cache layout is assumed downstream.

use std::path::{Path, PathBuf};

use anyhow::{Context, Result};
use hf_hub::{HFClientSync, HFRepositorySync, RepoTypeModel};

/// A reference to a Hugging Face model repository (`owner/name`).
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct HfRepo {
    /// Repository owner (user or organization), e.g. `"google"`.
    pub owner: String,
    /// Repository name, e.g. `"gemma-2b"`.
    pub name: String,
}

impl HfRepo {
    /// Parse a repo id of the form `owner/name`.
    pub fn parse(id: &str) -> Result<Self> {
        let (owner, name) = id.split_once('/').ok_or_else(|| {
            anyhow::anyhow!("expected a model id of the form `owner/name`, got `{id}`")
        })?;
        if owner.is_empty() || name.is_empty() {
            return Err(anyhow::anyhow!(
                "model id `{id}` must have a non-empty owner and name"
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

    /// A blocking handle to this repository.
    pub fn client(&self) -> Result<HFRepositorySync<RepoTypeModel>> {
        let client = HFClientSync::new().context("failed to create the Hugging Face client")?;
        Ok(client.model(&self.owner, &self.name))
    }

    /// Download a single file into `local_dir`, returning its final path.
    pub fn download_file(&self, filename: &str, local_dir: &Path) -> Result<PathBuf> {
        let repo = self.client()?;
        let path = repo
            .download_file()
            .filename(filename.to_string())
            .local_dir(local_dir.to_path_buf())
            .send()
            .with_context(|| format!("failed to download `{filename}` from `{}`", self.id()))?;
        Ok(path)
    }

    /// Download the files matching `allow` / `ignore` glob patterns into
    /// `target_dir` with the given level of parallelism.
    ///
    /// Empty `allow` downloads every file that is not ignored. Returned path is
    /// `target_dir` itself (the files live directly under it).
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
            .with_context(|| format!("failed to download `{}`", self.id()))?;
        Ok(target_dir.to_path_buf())
    }
}

/// Classification of a downloaded model repository.
#[derive(Debug, Clone)]
pub struct HfDownload {
    /// The directory files were downloaded into.
    pub dir: PathBuf,
    /// Path to `config.json`, if present.
    pub config_file: Option<PathBuf>,
    /// Paths to all `.safetensors` weight files (sorted).
    pub safetensors: Vec<PathBuf>,
    /// Tokenizer-related files (sorted).
    pub tokenizer_files: Vec<PathBuf>,
}

/// Return the names of every file under `dir` (recursively), as paths
/// relative to `dir`.
fn walk_files(dir: &Path) -> Result<Vec<String>> {
    let mut out = Vec::new();
    if !dir.exists() {
        return Ok(out);
    }
    let mut stack = vec![dir.to_path_buf()];
    while let Some(current) = stack.pop() {
        for entry in std::fs::read_dir(&current)
            .with_context(|| format!("failed to read directory `{}`", current.display()))?
        {
            let entry = entry?;
            let path = entry.path();
            let rel = path
                .strip_prefix(dir)
                .expect("walked paths are always under the root");
            if path.is_dir() {
                stack.push(path);
            } else {
                out.push(rel.to_string_lossy().replace('\\', "/"));
            }
        }
    }
    out.sort();
    Ok(out)
}

/// Classify the contents of a downloaded snapshot directory.
pub fn classify_download(dir: &Path) -> Result<HfDownload> {
    let mut safetensors = Vec::new();
    let mut tokenizer_files = Vec::new();
    let mut config_file = None;

    for rel in walk_files(dir)? {
        let path = dir.join(&rel);
        let lower = rel.to_ascii_lowercase();
        if rel == "config.json" {
            config_file = Some(path);
        } else if lower.ends_with(".safetensors") {
            safetensors.push(path);
        } else if is_tokenizer_file(&lower) {
            tokenizer_files.push(path);
        }
    }

    Ok(HfDownload {
        dir: dir.to_path_buf(),
        config_file,
        safetensors,
        tokenizer_files,
    })
}

fn is_tokenizer_file(lower_rel: &str) -> bool {
    lower_rel == "tokenizer.json"
        || lower_rel == "tokenizer_config.json"
        || lower_rel == "tokenizer.model"
        || lower_rel == "vocab.json"
        || lower_rel == "merges.txt"
        || lower_rel == "special_tokens_map.json"
        || lower_rel == "added_tokens.json"
        || lower_rel.starts_with("tokenizer/")
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parses_owner_and_name() {
        let repo = HfRepo::parse("google/gemma-2b").unwrap();
        assert_eq!(repo.owner, "google");
        assert_eq!(repo.name, "gemma-2b");
        assert_eq!(repo.id(), "google/gemma-2b");
    }

    #[test]
    fn rejects_malformed_ids() {
        assert!(HfRepo::parse("no-slash").is_err());
        assert!(HfRepo::parse("/empty-owner").is_err());
        assert!(HfRepo::parse("empty-name/").is_err());
    }

    #[test]
    fn walk_and_classify_finds_files() {
        let dir = tempfile::tempdir().unwrap();
        let root = dir.path();
        std::fs::create_dir_all(root.join("nested")).unwrap();
        std::fs::write(root.join("config.json"), "{}").unwrap();
        std::fs::write(root.join("model.safetensors"), "w").unwrap();
        std::fs::write(root.join("tokenizer.json"), "{}").unwrap();
        std::fs::write(root.join("nested/inner.bin"), "x").unwrap();

        let download = classify_download(root).unwrap();
        assert_eq!(download.config_file, Some(root.join("config.json")));
        assert_eq!(download.safetensors, vec![root.join("model.safetensors")]);
        assert!(
            download
                .tokenizer_files
                .contains(&root.join("tokenizer.json"))
        );
        assert!(
            !download
                .tokenizer_files
                .iter()
                .any(|p| p.ends_with("inner.bin"))
        );
        assert!(download.dir == root);
    }

    #[test]
    fn classifies_missing_directory_as_empty() {
        let dir = PathBuf::from("/nonexistent/llm-burner-test");
        let download = classify_download(&dir).unwrap();
        assert!(download.config_file.is_none());
        assert!(download.safetensors.is_empty());
        assert!(download.tokenizer_files.is_empty());
    }
}
