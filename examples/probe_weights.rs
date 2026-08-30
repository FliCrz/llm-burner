//! Dump exact leading values + std of key tensors after load, for comparison
//! against direct safetensors reads.
use burn::tensor::DType;
use burn_store::ModuleSnapshot;
use llm_burner::config::TransformersConfig;
use llm_burner::data::TokenizerStore;
use llm_burner::model::load::load_from_safetensors;
use llm_burner::model::{LlmModel, LlmModelConfig};
use std::path::Path;

fn main() -> anyhow::Result<()> {
    type B = burn::backend::Flex<f32, i32>;
    let dir_arg = std::env::args().nth(1).expect("model_dir");
    let dir = Path::new(&dir_arg);
    let cfg = TransformersConfig::from_path(dir.join("config.json"))?;
    let config = LlmModelConfig::from_transformers(&cfg);
    let _tokenizer = TokenizerStore::from_file(&dir.join("tokenizer.json"))?;
    let shards = llm_burner::hf::classify_download(dir)?.safetensors;
    let refs: Vec<&Path> = shards.iter().map(std::path::PathBuf::as_path).collect();
    let mut model = LlmModel::<B>::new(&config, &Default::default());
    load_from_safetensors(&mut model, &refs, DType::F32)?;

    let want = [
        "model.embed_tokens.weight",
        "model.layers.0.self_attn.q_proj.weight",
        "model.layers.0.self_attn.q_proj.bias",
        "model.layers.0.input_layernorm.weight",
        "model.norm.weight",
        "model.layers.0.mlp.gate_proj.weight",
    ];
    for s in model.collect(None, None, false) {
        let name = s.full_path();
        if !want.contains(&name.as_str()) {
            continue;
        }
        let data = s.to_data()?;
        let dt = data.dtype;
        let data2 = if dt == burn::tensor::DType::F32 {
            data
        } else {
            data.convert_dtype(DType::F32)
        };
        let v: Vec<f32> = data2.to_vec()?;
        let n = v.len() as f32;
        let mean = v.iter().sum::<f32>() / n;
        let var = v.iter().map(|x| (x - mean).powi(2)).sum::<f32>() / n;
        println!(
            "{name}: dtype={dt:?} std={:.4} mean={:.5} first6={:?}",
            var.sqrt(),
            mean,
            &v[..v.len().min(6)]
        );
    }
    // Row-level embed audit.
    let data = model.model.embed_tokens.weight.val().to_data();
    let dt = data.dtype;
    let data2 = if dt == burn::tensor::DType::F32 {
        data
    } else {
        data.convert_dtype(DType::F32)
    };
    let v = data2.to_vec()?;
    let (rows, cols) = (151936usize, 896usize);
    assert_eq!(v.len(), rows * cols);
    let norm = |r: usize| {
        v[r * cols..(r + 1) * cols]
            .iter()
            .map(|x: &f32| x * x)
            .sum::<f32>()
            .sqrt()
    };
    println!("[embed] dims ok, first3={:?}", &v[..3]);
    for r in [
        0usize, 1, 100, 50000, 151000, 151664, 151665, 151900, 151935,
    ] {
        println!(
            "[embed] row {r}: norm={:.4} first3={:?}",
            norm(r),
            &v[r * cols..r * cols + 3]
        );
    }
    Ok(())
}
