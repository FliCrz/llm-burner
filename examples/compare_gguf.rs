//! Debug: compare dequantized tensors between two GGUF files.
//!
//! Usage: cargo run --release --example compare_gguf -- FILE_A FILE_B

use rlx_gguf::GgufFile;

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let (a_path, b_path) = (&args[1], &args[2]);
    let a = GgufFile::from_path(a_path).unwrap();
    let b = GgufFile::from_path(b_path).unwrap();

    let mut names: Vec<&String> = a.tensors.keys().collect();
    names.sort();
    println!(
        "{:30} {:>10} {:>10} {:>10}",
        "tensor", "mse", "max_abs", "cos_sim"
    );
    for name in names {
        let Some(tb) = b.tensors.get(name) else {
            println!("{name:30} missing in B");
            continue;
        };
        let ta = &a.tensors[name];
        if ta.dtype != tb.dtype || ta.shape != tb.shape {
            println!(
                "{name:30} dtype/shape mismatch A={:?}{:?} B={:?}{:?}",
                ta.dtype, ta.shape, tb.dtype, tb.shape
            );
            continue;
        }
        let (da, _) = a.dequant_f32(name).unwrap();
        let (db, _) = b.dequant_f32(name).unwrap();
        let n = da.len() as f64;
        let mse: f64 = da
            .iter()
            .zip(db.iter())
            .map(|(x, y)| {
                let d = (x - y) as f64;
                d * d
            })
            .sum::<f64>()
            / n;
        let max_abs = da
            .iter()
            .zip(db.iter())
            .map(|(x, y)| (x - y).abs())
            .fold(0f32, f32::max);
        let dot: f64 = da
            .iter()
            .zip(db.iter())
            .map(|(x, y)| (*x * *y) as f64)
            .sum();
        let na: f64 = da.iter().map(|x| (*x * *x) as f64).sum::<f64>().sqrt();
        let nb: f64 = db.iter().map(|y| (*y * *y) as f64).sum::<f64>().sqrt();
        println!(
            "{name:30} {mse:10.3e} {max_abs:10.3e} {:>10}",
            format!("{:.6}", dot / (na * nb))
        );
    }
}
