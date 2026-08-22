//! Measures peak RSS of the streaming corpus tokenizer.
//!
//! Usage: `cargo run --release --example tokenize_mem -- <model_dir> <dataset_dir>`
//!
//! Peak RSS should stay near `largest single file + flat window arena`
//! instead of growing with the whole corpus at once.

use std::path::Path;

use llm_burner::data::{TokenizerStore, collect_text_files, tokenize_corpus};

/// Highest resident set size observed so far, in KiB (`/proc/self/status`).
fn peak_rss_kib() -> u64 {
    std::fs::read_to_string("/proc/self/status")
        .expect("linux only")
        .lines()
        .find(|line| line.starts_with("VmHWM"))
        .and_then(|line| line.split_whitespace().nth(1))
        .and_then(|kib| kib.parse::<u64>().ok())
        .expect("VmHWM present")
}

fn kib_as_gib(kib: u64) -> f64 {
    kib as f64 / 1024.0 / 1024.0
}

fn main() -> anyhow::Result<()> {
    let mut args = std::env::args().skip(1);
    let model_dir = args.next().expect("usage: tokenize_mem <model_dir> <dataset_dir>");
    let dataset_dir = args.next().expect("usage: tokenize_mem <model_dir> <dataset_dir>");

    let tokenizer = TokenizerStore::from_file(&Path::new(&model_dir).join("tokenizer.json"))?;
    let files = collect_text_files(
        Path::new(&dataset_dir),
        &["txt", "text", "md", "jsonl"],
    );
    anyhow::ensure!(!files.is_empty(), "no text files found in `{dataset_dir}`");
    let total_bytes: u64 = files
        .iter()
        .map(|f| std::fs::metadata(f).map(|m| m.len()).unwrap_or(0))
        .sum();
    println!(
        "{} files, {:.2} GiB of text",
        files.len(),
        total_bytes as f64 / 1024.0 / 1024.0 / 1024.0
    );
    println!("baseline peak RSS: {:.3} GiB", kib_as_gib(peak_rss_kib()));

    let started = std::time::Instant::now();
    let (store, tokens) = tokenize_corpus(&tokenizer, &files, 256, tokenizer.pad_id)?;

    println!(
        "tokenized {} tokens -> {} windows in {:?}",
        tokens,
        store.len(),
        started.elapsed()
    );
    println!("final peak RSS:    {:.3} GiB", kib_as_gib(peak_rss_kib()));
    Ok(())
}
