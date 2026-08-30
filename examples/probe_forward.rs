//! Forward-pass health probe: load a checkpoint on the CPU backend,
//! teacher-force a short text, and report the LM loss plus the top next-token
//! candidates at the final position. A correctly loaded model assigns its
//! natural continuation a low loss and ranks it first; a broken forward
//! yields near-uniform logits (~ln(vocab) loss) and gibberish candidates.
//!
//! Usage: cargo run --release --example probe_forward -- <model_dir> [text]

use std::path::Path;

use burn::tensor::Int;
use llm_burner::data::WindowStore;
use llm_burner::pipeline::load_model_from_dir;

fn main() -> anyhow::Result<()> {
    type B = burn::backend::Wgpu<f32, i32>;
    let mut args = std::env::args().skip(1);
    let model_dir = args
        .next()
        .expect("usage: probe_forward <model_dir> [text]");
    let text = args
        .next()
        .unwrap_or_else(|| "The capital of France is".into());

    let (model, config, tokenizer) =
        load_model_from_dir::<B>(Path::new(&model_dir), burn::tensor::DType::F32)?;
    println!(
        "loaded {} (d_model {}, {} layers, vocab {}) — tie_word_embeddings={}",
        model_dir, config.d_model, config.n_layers, config.vocab_size, config.tie_word_embeddings
    );

    // Weight sanity: embedding rows must not be near-zero or exploded.
    let emb = model.model.embed_tokens.weight.val();
    let [rows, cols] = emb.dims();
    let flat = emb.reshape([rows * cols]);
    let vals = flat.to_data().to_vec::<f32>()?;
    let mean: f32 = vals.iter().sum::<f32>() / vals.len() as f32;
    let var: f32 = vals.iter().map(|v| (v - mean).powi(2)).sum::<f32>() / vals.len() as f32;
    println!(
        "embed: [{rows},{cols}] mean={mean:.4} std={:.4}",
        var.sqrt()
    );

    let vocab: std::collections::HashMap<u32, String> = tokenizer
        .vocab_ordered()
        .into_iter()
        .map(|(t, i)| (i, t))
        .collect();
    let ids = tokenizer.encode_raw(&text)?;
    // Per-layer residual-stream RMS after each decoder layer (position -1).
    {
        let dev = Default::default();
        let input = burn::tensor::Tensor::<B, 2, Int>::from_data(
            burn::tensor::TensorData::new(ids.clone(), [1, ids.len()]),
            &dev,
        );
        print!("residual rms/layer:");
        let mut x = model.model.embed_tokens.forward(input.clone());
        let rms = |t: &burn::tensor::Tensor<B, 3>| -> f32 {
            let v = t.to_data().to_vec::<f32>().unwrap();
            let last = &v[v.len() - config.d_model..];
            (last.iter().map(|e| e * e).sum::<f32>() / last.len() as f32).sqrt()
        };
        print!(" emb={:.2}", rms(&x));
        for (i, layer) in model.model.layers.iter().enumerate() {
            x = layer.forward(x);
            print!(" l{}={:.2}", i, rms(&x));
        }
        println!();
    }

    println!("tokens({}): {:?} …", ids.len(), &ids[..ids.len().min(12)]);

    // Teacher-forced loss over the whole sequence.
    let mut windows = WindowStore::new(ids.len());
    windows.push_padded_tail(&ids, tokenizer.pad_id);
    let flat_tokens = windows.window_tokens(0, 1);
    let batch = llm_burner::data::build_causal_batch_from_flat::<B>(
        flat_tokens,
        ids.len(),
        &Default::default(),
        tokenizer.pad_id,
    );
    let out = model.forward_classification(batch);
    let loss: f32 = out.loss.into_scalar();
    println!(
        "teacher-forced loss: {loss:.4}  (uniform-random baseline: ln(V) = {:.2})",
        (config.vocab_size as f32).ln()
    );

    // Greedy continuation: argmax at each step for a few tokens.
    let mut ids = ids.clone();
    print!("greedy: \"{text}");
    for _ in 0..8 {
        let seq = ids.len();
        let input = burn::tensor::Tensor::<B, 2, Int>::from_data(
            burn::tensor::TensorData::new(ids.clone(), [1, seq]),
            &Default::default(),
        );
        let logits = model.forward(input);
        let last = logits
            .slice([0..1, (seq - 1)..seq])
            .reshape([config.vocab_size]);
        let vals = last.to_data().to_vec::<f32>()?;
        let mut ranked: Vec<(u32, f32)> = vals
            .iter()
            .enumerate()
            .map(|(i, v)| (i as u32, *v))
            .collect();
        ranked.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());
        let top: Vec<String> = ranked[..5.min(ranked.len())]
            .iter()
            .map(|(id, v)| format!("{}({v:.2})", decode(&vocab, *id)))
            .collect();
        println!("  top5: {}", top.join(" "));
        ids.push(ranked[0].0);
        print!("{}", decode(&vocab, ranked[0].0));
    }
    println!("\"");
    Ok(())
}

fn decode(vocab: &std::collections::HashMap<u32, String>, id: u32) -> String {
    vocab
        .get(&id)
        .map(|s| s.replace('\n', "\\n").replace(' ', "\u{b7}"))
        .unwrap_or_else(|| "<unk>".into())
}
