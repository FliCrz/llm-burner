use burn::tensor::{Int, Tensor, backend::Backend};

/// Half-split (GPT-NeoX style) rotary position embedding, as used by Llama,
/// Qwen, SmolLM and Gemma.
///
/// The frequency tables `cos`/`sin` are computed on the fly from the sequence
/// length, so nothing is stored in the model record and the module stays easy
/// to save/load.
pub fn rope_apply<B: Backend, const D: usize>(
    x: Tensor<B, D>,
    cos: Tensor<B, D>,
    sin: Tensor<B, D>,
) -> Tensor<B, D> {
    x.clone().mul(cos) + rotate_half(&x).mul(sin)
}

/// `rotate_half(x)` splits the last dimension in two halves `x1, x2` and
/// returns `cat(-x2, x1)`.
pub fn rotate_half<B: Backend, const D: usize>(x: &Tensor<B, D>) -> Tensor<B, D> {
    let head_dim = x.dims()[D - 1];
    let half = head_dim / 2;
    let parts = x.clone().split(half, D - 1);
    let (x1, x2) = (parts[0].clone(), parts[1].clone());
    Tensor::cat(vec![x2.neg(), x1], D - 1)
}

/// Build the `cos` and `sin` tables for RoPE.
///
/// Returns two tensors of shape `[seq_len, head_dim]` where both halves are
/// duplicated so the half-split application matches Hugging Face.
pub fn rope_cos_sin<B: Backend>(
    seq_len: usize,
    head_dim: usize,
    theta: f64,
    device: &B::Device,
) -> (Tensor<B, 2>, Tensor<B, 2>) {
    rope_cos_sin_offset(0, seq_len, head_dim, theta, device)
}

/// Build the `cos` and `sin` tables for RoPE spanning positions
/// `[start_pos, start_pos + seq_len)`.
pub fn rope_cos_sin_offset<B: Backend>(
    start_pos: usize,
    seq_len: usize,
    head_dim: usize,
    theta: f64,
    device: &B::Device,
) -> (Tensor<B, 2>, Tensor<B, 2>) {
    let head_dim = head_dim / 2 * 2;
    let half = head_dim / 2;

    // inv_freq[i] = 1 / theta^(2i/head_dim)
    let i = Tensor::<B, 1, Int>::arange(0..half as i64, device).float();
    let exp = i.mul_scalar(-2.0 / head_dim as f64);
    let inv_freq = exp.mul_scalar(theta.ln()).exp();

    // angles[s, i] = (start_pos + s) * inv_freq[i]
    let start = start_pos as i64;
    let end = (start_pos + seq_len) as i64;
    let positions = Tensor::<B, 1, Int>::arange(start..end, device).float();
    let angles = positions
        .unsqueeze_dim::<2>(1)
        .mul(inv_freq.unsqueeze_dim::<2>(0));

    let cos = angles.clone().cos();
    let sin = angles.sin();

    let cos = Tensor::cat(vec![cos.clone(), cos], 1);
    let sin = Tensor::cat(vec![sin.clone(), sin], 1);

    (cos, sin)
}

#[cfg(test)]
mod tests {
    use super::*;
    type B = burn::backend::Flex<f32, i32>;

    #[test]
    fn rope_offset_matches_slice_of_full_table() {
        let device = burn::backend::flex::FlexDevice;
        let (full_cos, full_sin) = rope_cos_sin::<B>(16, 8, 10000.0, &device);
        let (off_cos, off_sin) = rope_cos_sin_offset::<B>(4, 6, 8, 10000.0, &device);

        let full_cos_slice = full_cos
            .slice([4..10, 0..8])
            .to_data()
            .to_vec::<f32>()
            .unwrap();
        let off_cos_data = off_cos.to_data().to_vec::<f32>().unwrap();
        assert_eq!(full_cos_slice, off_cos_data);

        let full_sin_slice = full_sin
            .slice([4..10, 0..8])
            .to_data()
            .to_vec::<f32>()
            .unwrap();
        let off_sin_data = off_sin.to_data().to_vec::<f32>().unwrap();
        assert_eq!(full_sin_slice, off_sin_data);
    }
}
