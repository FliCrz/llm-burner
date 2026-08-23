//! Absolute correctness probes that need no pretrained weights.
//!
//! 1. Identity: zeroing o_proj/down_proj makes every decoder layer an exact
//!    identity, so body(x) must equal final_norm(embed(x)).
//! 2. Overfit: a healthy causal-LM stack drives memorization loss toward 0;
//!    a broken one plateaus at ~ln(vocab).
use burn::module::Module;
use burn::tensor::{Int};

type B = burn::backend::Flex<f32, i32>;

fn main() -> anyhow::Result<()> {
    let device = burn::backend::flex::FlexDevice;
    let config = llm_burner::model::LlmModelConfig::tiny();
    let model = llm_burner::model::LlmModel::<B>::new(&config, &device);

    // --- 1. identity ---
    let mut model = model;
    for layer in model.model.layers.iter_mut() {
        let [i, o] = layer.self_attn.o_proj.weight.val().dims();
        layer.self_attn.o_proj.weight =
            burn::module::Param::from_tensor(burn::tensor::Tensor::zeros([i, o], &device));
        let [i, o] = layer.mlp.down_proj.weight.val().dims();
        layer.mlp.down_proj.weight =
            burn::module::Param::from_tensor(burn::tensor::Tensor::zeros([i, o], &device));
    }
    let tokens: Vec<u32> = vec![1, 7, 3, 9, 2];
    let input = burn::tensor::Tensor::<B, 2, Int>::from_data(
        burn::tensor::TensorData::new(tokens.clone(), [1, tokens.len()]),
        &device,
    );
    let hidden = model.model.forward(input.clone());
    let emb = model.model.embed_tokens.forward(input.clone());
    let expected = model.model.norm.forward(emb);
    let diff = hidden.sub(expected).square().sum().into_scalar();
    println!("[identity] ||body - norm(embed)||^2 = {diff:.6e}  (expect ~0)");

    // --- 2. overfit ---
    use burn::optim::adaptor::OptimizerAdaptor;
    use burn::optim::{AdamW, AdamWConfig, GradientsParams, Optimizer};
    type AD = burn::backend::Autodiff<B>;
    let mut opt: OptimizerAdaptor<AdamW, llm_burner::model::LlmModel<AD>, AD> =
        AdamWConfig::new().with_weight_decay(0.0).init::<AD, llm_burner::model::LlmModel<AD>>();
    let mut train_model: llm_burner::model::LlmModel<AD> =
        llm_burner::model::LlmModel::<B>::new(&config, &device).train();
    let seq: Vec<u32> = (0..16u32).cycle().take(64).collect();
    let target: Vec<u32> = seq[1..].to_vec();
    let lr = 3e-3;
    for step in 0..301 {
        let inp = burn::tensor::Tensor::<AD, 2, Int>::from_data(
            burn::tensor::TensorData::new(seq[..63].to_vec(), [1, 63]),
            &device,
        );
        let tgt = burn::tensor::Tensor::<AD, 2, Int>::from_data(
            burn::tensor::TensorData::new(target[..63].to_vec(), [1, 63]),
            &device,
        );
        let logits = train_model.forward(inp);
        let [_, s, v] = logits.dims();
        let loss = burn::nn::loss::CrossEntropyLoss::new(None, &device)
            .forward(logits.reshape([s, v]), tgt.reshape([s]));
        if step % 100 == 0 || step == 300 {
            println!("[overfit] step {step}: loss {:.4}", loss.clone().into_scalar());
        }
        let grads = loss.backward();
        let gs = GradientsParams::from_grads(grads, &train_model);
        train_model = opt.step(lr, train_model, gs);
    }
    Ok(())
}
