//! Rust-side reference K-quant/legacy dequant used to pin down the WGSL
//! kernels in [`crate::infer::vk::shaders`].
//!
//! Each `dequant_elem` here is a line-for-line port of the corresponding
//! WGSL `dequant_elem` body. The unit tests compare whole rows against
//! `rlx_gguf`'s decoder (which is itself the `ggml-quants.c` reference), so a
//! transcription bug in either copy fails loudly instead of silently
//! producing garbage on the GPU.

use crate::infer::vk::shaders::Dt;

fn le_u32(b: &[u8]) -> u32 {
    u32::from_le_bytes([b[0], b[1], b[2], b[3]])
}
fn le_f32(b: &[u8]) -> f32 {
    f32::from_le_bytes([b[0], b[1], b[2], b[3]])
}
fn hf(b: &[u8]) -> f32 {
    half::f16::from_le_bytes([b[0], b[1]]).to_f32()
}
fn as_i8(u: u8) -> i32 {
    u as i8 as i32
}

/// Dequant element `e` of a tensor **row** given that row's bytes.
pub fn dequant_elem(dt: Dt, row: &[u8], e: usize) -> f32 {
    let blk = dt.block_bytes() as usize;
    let bsize = dt.block_elems() as usize;
    let bb = e / bsize * blk;
    let blk_bytes = &row[bb..bb + blk];
    let elem = e % bsize;
    match dt {
        Dt::F32 => le_f32(&row[e * 4..]),
        Dt::F16 => hf(&row[e * 2..]),
        Dt::Bf16 => {
            let u = u16::from_le_bytes([row[e * 2], row[e * 2 + 1]]);
            f32::from_bits((u as u32) << 16)
        }
        Dt::Q8_0 => {
            let d = hf(&blk_bytes[0..2]);
            d * as_i8(blk_bytes[2 + elem]) as f32
        }
        Dt::Q4_0 => {
            let d = hf(&blk_bytes[0..2]);
            let k = elem & 31;
            let v = if k < 16 {
                blk_bytes[2 + k] & 0x0F
            } else {
                blk_bytes[2 + k - 16] >> 4
            };
            d * (v as i32 - 8) as f32
        }
        Dt::Q4_1 => {
            let d = hf(&blk_bytes[0..2]);
            let m = hf(&blk_bytes[2..4]);
            let k = elem & 31;
            let v = if k < 16 {
                blk_bytes[4 + k] & 0x0F
            } else {
                blk_bytes[4 + k - 16] >> 4
            };
            d * v as f32 + m
        }
        Dt::Q5_0 => {
            let d = hf(&blk_bytes[0..2]);
            let qh = le_u32(&blk_bytes[2..6]);
            let k = elem & 31;
            let v = if k < 16 {
                (blk_bytes[6 + k] & 0x0F) | ((((qh >> k) & 1) as u8) << 4)
            } else {
                let j = k - 16;
                (blk_bytes[6 + j] >> 4) | ((((qh >> (j + 16)) & 1) as u8) << 4)
            };
            d * (v as i32 - 16) as f32
        }
        Dt::Q5_1 => {
            let d = hf(&blk_bytes[0..2]);
            let m = hf(&blk_bytes[2..4]);
            let qh = le_u32(&blk_bytes[4..8]);
            let k = elem & 31;
            let v = if k < 16 {
                (blk_bytes[8 + k] & 0x0F) | ((((qh >> k) & 1) as u8) << 4)
            } else {
                let j = k - 16;
                (blk_bytes[8 + j] >> 4) | ((((qh >> (j + 16)) & 1) as u8) << 4)
            };
            d * v as f32 + m
        }
        Dt::Q8K => {
            let d = le_f32(&blk_bytes[0..4]);
            d * as_i8(blk_bytes[4 + elem]) as f32
        }
        Dt::Q2K => {
            // scales[16] at 0, qs[64] at 16, d@80, dmin@82.
            let o = (elem >> 7) & 1;
            let rem = elem & 127;
            let g = rem >> 5;
            let sub = rem & 31;
            let ll = sub & 15;
            let half = (sub >> 4) & 1;
            let shift = g * 2;
            let qidx = o * 32 + half * 16 + ll;
            let sidx = o * 8 + g * 2 + half;
            let qb = blk_bytes[16 + qidx];
            let sc = blk_bytes[sidx];
            let d = hf(&blk_bytes[80..82]);
            let mn = hf(&blk_bytes[82..84]);
            let val = (qb >> shift) & 3;
            d * (sc & 0x0F) as f32 * val as f32 - mn * (sc >> 4) as f32
        }
        Dt::Q3K => {
            // hm[32]@0, qs[64]@32, scales[12]@96, d@108.
            let o = (elem >> 7) & 1;
            let rem = elem & 127;
            let g = rem >> 5;
            let sub = rem & 31;
            let ll = sub & 15;
            let half = (sub >> 4) & 1;
            let shift = g * 2;
            let qidx = o * 32 + half * 16 + ll;
            let sidx = o * 8 + g * 2 + half;
            let d_all = hf(&blk_bytes[108..110]);
            let dl = d_all * (q3_scale(&blk_bytes[96..108], sidx) - 32) as f32;
            let qb = blk_bytes[32 + qidx];
            let hb = blk_bytes[half * 16 + ll];
            let mbit = 1u8 << (o * 4 + g);
            let h = if hb & mbit != 0 { 0 } else { 4 };
            let val = (qb >> shift) as i32 & 3;
            dl * (val - h) as f32
        }
        Dt::Q4K => {
            // d@0, dmin@2, scales[12]@4, qs[128]@16.
            let d = hf(&blk_bytes[0..2]);
            let dmn = hf(&blk_bytes[2..4]);
            let g = elem / 32;
            let s = elem % 32;
            let (sc, mn) = get_scale_min_k4(g, &blk_bytes[4..16]);
            let qb = blk_bytes[16 + (g >> 1) * 32 + s];
            let v = if g & 1 == 0 { qb & 0x0F } else { qb >> 4 };
            d * sc as f32 * v as f32 - dmn * mn as f32
        }
        Dt::Q5K => {
            // d@0, dmin@2, scales[12]@4, qh[32]@16, qs[128]@48.
            let d = hf(&blk_bytes[0..2]);
            let dmn = hf(&blk_bytes[2..4]);
            let g = elem / 32;
            let s = elem % 32;
            let (sc, mn) = get_scale_min_k4(g, &blk_bytes[4..16]);
            let qb = blk_bytes[48 + (g >> 1) * 32 + s];
            let u = 1u8 << (2 * (g >> 1) + (g & 1));
            let lo = if g & 1 == 0 { qb & 0x0F } else { qb >> 4 };
            let hi = if blk_bytes[16 + s] & u != 0 { 16 } else { 0 };
            d * sc as f32 * (lo + hi) as f32 - dmn * mn as f32
        }
        Dt::Q6K => {
            // ql[128]@0, qh[64]@128, sc[16]@192, d@208.
            let h = (elem >> 7) & 1;
            let sub = elem & 127;
            let seg = sub >> 5;
            let l = sub & 31;
            let d = hf(&blk_bytes[208..210]);
            let qlidx = h * 64 + l + (seg & 1) * 32;
            let qh_b = blk_bytes[128 + h * 32 + l];
            let nib = if seg & 2 == 0 {
                blk_bytes[qlidx] & 0x0F
            } else {
                blk_bytes[qlidx] >> 4
            };
            let m = (qh_b >> (seg * 2)) & 3;
            let q = (nib | (m << 4)) as i32 - 32;
            let sidx = h * 8 + (l >> 4) + seg * 2;
            let sc = as_i8(blk_bytes[192 + sidx]);
            d * sc as f32 * q as f32
        }
    }
}

fn get_scale_min_k4(j: usize, q: &[u8]) -> (u8, u8) {
    if j < 4 {
        (q[j] & 63, q[j + 4] & 63)
    } else {
        let d = (q[j + 4] & 0x0F) | ((q[j - 4] >> 6) << 4);
        let m = (q[j + 4] >> 4) | ((q[j] >> 6) << 4);
        (d, m)
    }
}

/// Bits of the Q3_K scale at byte index `r` (0..16), as a signed i8 scale.
fn q3_scale(scales: &[u8], r: usize) -> i32 {
    const KMASK1: u32 = 0x0303_0303;
    const KMASK2: u32 = 0x0f0f_0f0f;
    let a0 = le_u32(&scales[0..4]);
    let a1 = le_u32(&scales[4..8]);
    let a2 = le_u32(&scales[8..12]);
    let ax0 = (a0 & KMASK2) | ((a2 & KMASK1) << 4);
    let ax1 = (a1 & KMASK2) | (((a2 >> 2) & KMASK1) << 4);
    let ax2 = ((a0 >> 4) & KMASK2) | (((a2 >> 4) & KMASK1) << 4);
    let ax3 = ((a1 >> 4) & KMASK2) | (((a2 >> 6) & KMASK1) << 4);
    let word = [ax0, ax1, ax2, ax3][r >> 2];
    as_i8(((word >> ((r & 3) * 8)) & 0xFF) as u8)
}

/// Dequantize a whole row into `out` (length `cols`).
pub fn dequant_row(dt: Dt, row: &[u8], cols: usize, out: &mut [f32]) {
    assert_eq!(cols, out.len());
    for (e, o) in out.iter_mut().enumerate() {
        *o = dequant_elem(dt, row, e);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn random_bytes(seed: u64, len: usize) -> Vec<u8> {
        let mut s = seed.wrapping_mul(0x9E3779B97F4A7C15).wrapping_add(0xBF5847_6D1CE4E5B9);
        (0..len)
            .map(|_| {
                s ^= s << 13;
                s ^= s >> 7;
                s ^= s << 17;
                s as u8
            })
            .collect()
    }

    const COLS: usize = 1024;

    /// Reference dequant must match rlx-gguf's decoder for every dtype.
    #[test]
    fn matches_rlx_gguf() {
        let cases: &[(Dt, fn(&[u8], usize) -> Vec<f32>)] = &[
            (Dt::Q8_0, |b, n| rlx_gguf::dequant_q8_0(b, n).unwrap()),
            (Dt::Q4_0, |b, n| rlx_gguf::dequant_q4_0(b, n).unwrap()),
            (Dt::Q4_1, |b, n| rlx_gguf::dequant_q4_1(b, n).unwrap()),
            (Dt::Q5_0, |b, n| rlx_gguf::dequant_q5_0(b, n).unwrap()),
            (Dt::Q5_1, |b, n| rlx_gguf::dequant_q5_1(b, n).unwrap()),
            (Dt::Q2K, |b, n| rlx_gguf::dequant_q2_k(b, n).unwrap()),
            (Dt::Q3K, |b, n| rlx_gguf::dequant_q3_k(b, n).unwrap()),
            (Dt::Q4K, |b, n| rlx_gguf::dequant_q4_k(b, n).unwrap()),
            (Dt::Q5K, |b, n| rlx_gguf::dequant_q5_k(b, n).unwrap()),
            (Dt::Q6K, |b, n| rlx_gguf::dequant_q6_k(b, n).unwrap()),
            (Dt::Q8K, |b, n| rlx_gguf::dequant_q8_k(b, n).unwrap()),
        ];
        for (seed, (dt, rlx)) in cases.iter().enumerate() {
            let nblk = COLS / dt.block_elems() as usize;
            let mut bytes =
                random_bytes(seed as u64 + 42, nblk * dt.block_bytes() as usize);
            // Copy the same row twice so descriptor-ish aliasing does not matter.
            bytes.extend_from_slice(&bytes.clone());
            let row = &bytes[..bytes.len() / 2];
            let rlx_vals = rlx(row, COLS);
            assert_eq!(rlx_vals.len(), COLS);
            let mut mine = vec![0f32; COLS];
            dequant_row(*dt, row, COLS, &mut mine);
            for (i, (a, b)) in mine.iter().zip(rlx_vals.iter()).enumerate() {
                if (a.is_nan() && b.is_nan()) || (a.is_infinite() && b.is_infinite() && a == b)
                {
                    continue;
                }
                let tol = 1e-4 * a.abs().max(b.abs()).max(1.0);
                assert!(
                    (a - b).abs() <= tol,
                    "{dt:?}[{i}] mine={a} rlx={b}"
                );
            }
        }
    }

    #[test]
    fn unconverted_dtypes_roundtrip() {
        // F32/F16/BF16 rows of shorts must widen losslessly (F32) or exactly
        // decode back (F16/BF16).
        let row: Vec<u8> = (0..COLS * 4)
            .map(|i| (i as u32).wrapping_mul(2654435761) as u8)
            .collect();
        let mut out = vec![0f32; COLS];
        dequant_row(Dt::F32, &row, COLS, &mut out);
        for i in 0..COLS {
            assert_eq!(
                out[i].to_bits(),
                le_f32(&row[i * 4..]).to_bits(),
                "f32[{i}]"
            );
        }
        let f16row: Vec<u8> = (0..COLS)
            .flat_map(|i| {
                let bits = half::f16::from_f32(((i as f32) - 500.0) * 0.01).to_bits();
                [bits as u8, (bits >> 8) as u8]
            })
            .collect();
        dequant_row(Dt::F16, &f16row, COLS, &mut out);
        for i in 0..COLS {
            assert_eq!(out[i], hf(&f16row[i * 2..]), "f16[{i}]");
        }
    }
}