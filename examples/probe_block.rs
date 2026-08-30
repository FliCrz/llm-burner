//! Crack open decoder layer 0 of the real checkpoint: print norms after each
//! internal op to find which sub-step explodes.
use burn::tensor::{DType, Int};
use burn_store::ModuleSnapshot;
use std::path::Path;

type B = burn::backend::Flex<f32, i32>;

fn rms(t: &burn::tensor::Tensor<B, 3>) -> f32 {
    let [b, s, d] = t.dims();
    let v = t.to_data().to_vec::<f32>().unwrap();
    let last = &v[(b * s - 1) * d..b * s * d];
    (last.iter().map(|x| x * x).sum::<f32>() / d as f32).sqrt()
}

fn main() -> anyhow::Result<()> {
    let dir_arg = std::env::args().nth(1).expect("model_dir");
    let dir = Path::new(&dir_arg);
    let cfg = llm_burner::config::TransformersConfig::from_path(dir.join("config.json"))?;
    let config = llm_burner::model::LlmModelConfig::from_transformers(&cfg);
    let tok = llm_burner::data::TokenizerStore::from_file(&dir.join("tokenizer.json"))?;
    let shards = llm_burner::hf::classify_download(dir)?.safetensors;
    let refs: Vec<&Path> = shards.iter().map(std::path::PathBuf::as_path).collect();
    let mut model = llm_burner::model::LlmModel::<B>::new(&config, &Default::default());
    llm_burner::model::load::load_from_safetensors(&mut model, &refs, DType::F32)?;

    let ids = tok.encode_raw("The capital of France is")?;
    let input = burn::tensor::Tensor::<B, 2, Int>::from_data(
        burn::tensor::TensorData::new(ids.clone(), [1, ids.len()]),
        &Default::default(),
    );

    // Replicate Transformer::forward manually.
    let x = model.model.embed_tokens.forward(input.clone());
    println!("emb rms(last pos) = {:.4}", rms(&x));
    let layer = &model.model.layers[0];

    let n1 = layer.input_layernorm.forward(x.clone());
    println!("after input_layernorm = {:.4}", rms(&n1));

    // Attention internals.
    let attn = &layer.self_attn;
    let q = attn.q_proj.forward(n1.clone());
    let k = attn.k_proj.forward(n1.clone());
    let v = attn.v_proj.forward(n1.clone());
    let flat = |t: burn::tensor::Tensor<B, 3>| -> (f32, f32) {
        let v = t.to_data().to_vec::<f32>().unwrap();
        let m = v.iter().sum::<f32>() / v.len() as f32;
        let var = v.iter().map(|x| (x - m).powi(2)).sum::<f32>() / v.len() as f32;
        (m, var.sqrt())
    };
    let (qm, qs) = flat(q.clone());
    println!("q mean={qm:.4} std={qs:.4}");
    let _ = k;
    let _ = v;
    let a_out = attn.forward(n1);
    println!("attn out rms = {:.4}", rms(&a_out));

    let x2 = a_out.add(x);
    println!("residual2 rms = {:.4}", rms(&x2));

    let n2 = layer.post_attention_layernorm.forward(x2.clone());
    let mlp_out = layer.mlp.forward(n2);
    println!("mlp out rms = {:.4}", rms(&mlp_out));
    let x3 = mlp_out.add(x2);
    println!(
        "after layer0 rms = {:.4}  (pipeline sweep said 0.23)",
        rms(&x3)
    );

    // Also verify against ModuleSnapshot collect for weight sanity of this layer.
    for s in model.collect(None, None, false) {
        let p = s.full_path();
        if p == "model.layers.0.self_attn.q_proj.weight" || p == "model.norm.weight" {
            let data = s.to_data()?;
            let dt = data.dtype;
            let d2 = if dt == DType::F32 {
                data
            } else {
                data.convert_dtype(DType::F32)
            };
            let v = d2.to_vec()?;
            let m = v.iter().sum::<f32>() / v.len() as f32;
            let var = v.iter().map(|x| (x - m).powi(2)).sum::<f32>() / v.len() as f32;
            println!("{p}: std={:.4} first3={:?}", var.sqrt(), &v[..3]);
        }
    }
    Ok(())
}
