//! Lazy-loaded inference engines over a quantized GGUF checkpoint.
//!
//! [`choose_engine`] owns the fallback ladder for `chat`: it prefers the
//! raw-Vulkan compute engine ([`vk::VulkanEngine`]) on machines with a
//! Vulkan-capable (integrated) GPU, and falls back to the CPU-only
//! [`crate::model::gguf::GgufEngine`] when Vulkan is unavailable, disabled at
//! build time, or the user passes `--device cpu`.

#[cfg(feature = "infer-vk")]
pub mod dequant_ref;
pub mod gguf;
#[cfg(feature = "infer-vk")]
pub mod vk;