//! Loading Hugging Face safetensors checkpoints into a [`LlmModel`] and saving
//! them back in a retrainable (PyTorch-compatible) layout.

use std::collections::HashSet;
use std::path::Path;

use crate::model::model::LlmModel;

use anyhow::{Context, Result};
use burn::tensor::{DType, backend::Backend};
use burn_store::{
    BurnToPyTorchAdapter, ModuleAdapter, ModuleSnapshot, ModuleStore, PyTorchToBurnAdapter,
    SafetensorsStore, TensorSnapshot,
};

/// Adapter that casts floating-point tensors to a target dtype.
#[derive(Debug, Clone)]
pub(crate) struct FloatDTypeAdapter {
    target: DType,
}

impl FloatDTypeAdapter {
    pub(crate) fn new(target: DType) -> Self {
        Self { target }
    }
}

impl ModuleAdapter for FloatDTypeAdapter {
    fn adapt(&self, snapshot: &TensorSnapshot) -> TensorSnapshot {
        let is_float = matches!(
            snapshot.dtype,
            DType::F64 | DType::F32 | DType::Flex32 | DType::F16 | DType::BF16
        );
        if !is_float || snapshot.dtype == self.target {
            return snapshot.clone();
        }

        let original_data_fn = snapshot.clone_data_fn();
        let target = self.target;
        let cast_data_fn = std::rc::Rc::new(move || {
            let data = original_data_fn()?;
            Ok(data.convert_dtype(target))
        });

        TensorSnapshot::from_closure(
            cast_data_fn,
            target,
            snapshot.shape.clone(),
            snapshot.path_stack.clone().unwrap_or_default(),
            snapshot.container_stack.clone().unwrap_or_default(),
            snapshot.tensor_id.unwrap_or_default(),
        )
    }

    fn clone_box(&self) -> Box<dyn ModuleAdapter> {
        Box::new(self.clone())
    }
}

/// Load weights from one or more Hugging Face `.safetensors` shard files
/// (PyTorch layout) into the model.
///
/// Floating-point tensors are cast to `target_dtype` while loading so models
/// compiled for F32, BF16 or F16 can reliably ingest checkpoints stored in a
/// different precision.
pub fn load_from_safetensors<B: Backend>(
    model: &mut LlmModel<B>,
    paths: &[&Path],
    target_dtype: DType,
) -> Result<()> {
    let expected: HashSet<String> = model
        .collect(None, None, false)
        .iter()
        .map(|snapshot| snapshot.full_path())
        .collect();

    let mut applied = HashSet::new();
    for path in paths {
        let mut store = SafetensorsStore::from_file(path)
            .with_from_adapter(
                FloatDTypeAdapter::new(target_dtype).chain(PyTorchToBurnAdapter),
            )
            .allow_partial(true)
            .validate(true);
        let result = store
            .apply_to(model)
            .with_context(|| format!("failed to load weights from `{}`", path.display()))?;
        if !result.errors.is_empty() {
            anyhow::bail!(
                "errors while loading `{}`: {:#?}",
                path.display(),
                result.errors
            );
        }
        applied.extend(result.applied);
    }

    let missing: Vec<&String> = expected.difference(&applied).collect();
    if !missing.is_empty() {
        anyhow::bail!(
            "the checkpoint is missing {} expected tensor(s): {:?}",
            missing.len(),
            missing
        );
    }
    Ok(())
}

/// Save the model to a `.safetensors` file in PyTorch-compatible layout
/// (`Linear` weights transposed back to `[out, in]`), so the file is
/// loadable by PyTorch / transformers and retrainable by Burn.
pub fn save_to_safetensors<B: Backend>(model: &LlmModel<B>, path: &Path) -> Result<()> {
    let mut store = SafetensorsStore::from_file(path)
        .with_to_adapter(BurnToPyTorchAdapter)
        .overwrite(true);
    store
        .collect_from(model)
        .with_context(|| format!("failed to write `{}`", path.display()))?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::model::LlmModelConfig;
    use tempfile::tempdir;

    type B = burn::backend::Flex<f32, i32>;

    fn save_with_float_dtype(model: &LlmModel<B>, path: &Path, dtype: DType) {
        let mut store = SafetensorsStore::from_file(path)
            .with_to_adapter(FloatDTypeAdapter::new(dtype).chain(BurnToPyTorchAdapter))
            .overwrite(true);
        store.collect_from(model).unwrap();
    }

    #[test]
    fn round_trip_pytorch_layout_f32() {
        let device = burn::backend::flex::FlexDevice;
        let config = LlmModelConfig::tiny();

        // Save a freshly initialized model in PyTorch-compatible layout.
        let dir = tempdir().unwrap();
        let path = dir.path().join("model.safetensors");

        let source = LlmModel::<B>::new(&config, &device);
        save_to_safetensors(&source, &path).unwrap();

        // Load with F32 precision
        let mut dest = LlmModel::<B>::new(&config, &device);
        load_from_safetensors(&mut dest, &[&path], DType::F32).unwrap();

        // Every tensor must be bit-identical after the round trip.
        let source_views = source.collect(None, None, false);
        let dest_views = dest.collect(None, None, false);
        assert_eq!(source_views.len(), dest_views.len());
        for (s, d) in source_views.iter().zip(dest_views.iter()) {
            assert_eq!(s.full_path(), d.full_path(), "paths differ");
            assert_eq!(
                s.to_data().unwrap().to_vec::<f32>().unwrap(),
                d.to_data().unwrap().to_vec::<f32>().unwrap(),
                "tensor `{}` differs after round trip",
                s.full_path()
            );
        }
    }

    #[test]
    fn round_trip_pytorch_layout_bf16() {
        let device = burn::backend::flex::FlexDevice;
        let config = LlmModelConfig::tiny();

        // Save BF16 checkpoint, then load into a Flex<f32> model.
        let dir = tempdir().unwrap();
        let bf16_path = dir.path().join("model-bf16.safetensors");

        let source = LlmModel::<B>::new(&config, &device);
        save_with_float_dtype(&source, &bf16_path, DType::BF16);

        let mut dest = LlmModel::<B>::new(&config, &device);
        load_from_safetensors(&mut dest, &[&bf16_path], DType::F32).unwrap();

        let source_views = source.collect(None, None, false);
        let dest_views = dest.collect(None, None, false);
        assert_eq!(source_views.len(), dest_views.len());
        for (s, d) in source_views.iter().zip(dest_views.iter()) {
            assert_eq!(s.full_path(), d.full_path(), "paths differ after BF16 load");
            assert_eq!(d.dtype, DType::F32, "loaded tensor must be F32");
        }
    }

    #[test]
    fn round_trip_pytorch_layout_f16() {
        let device = burn::backend::flex::FlexDevice;
        let config = LlmModelConfig::tiny();

        // Save F16 checkpoint, then load into a Flex<f32> model.
        let dir = tempdir().unwrap();
        let f16_path = dir.path().join("model-f16.safetensors");

        let source = LlmModel::<B>::new(&config, &device);
        save_with_float_dtype(&source, &f16_path, DType::F16);

        let mut dest = LlmModel::<B>::new(&config, &device);
        load_from_safetensors(&mut dest, &[&f16_path], DType::F32).unwrap();

        let source_views = source.collect(None, None, false);
        let dest_views = dest.collect(None, None, false);
        assert_eq!(source_views.len(), dest_views.len());
        for (s, d) in source_views.iter().zip(dest_views.iter()) {
            assert_eq!(s.full_path(), d.full_path(), "paths differ after F16 load");
            assert_eq!(d.dtype, DType::F32, "loaded tensor must be F32");
        }
    }

    #[test]
    fn missing_tensor_is_reported() {
        let device = burn::backend::flex::FlexDevice;
        let config = LlmModelConfig::tiny();

        let dir = tempdir().unwrap();
        let path = dir.path().join("model.safetensors");

        let source = LlmModel::<B>::new(&config, &device);
        save_to_safetensors(&source, &path).unwrap();

        // A model with an extra layer must fail to load (missing tensors).
        let bigger = LlmModelConfig {
            n_layers: config.n_layers + 1,
            ..config.clone()
        };
        let mut dest = LlmModel::<B>::new(&bigger, &device);
        let err = load_from_safetensors(&mut dest, &[&path], DType::F32).unwrap_err();
        assert!(err.to_string().contains("missing"), "{}", err);
    }
}
