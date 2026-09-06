//! Raw-Vulkan compute engine for the quantized GGUF chat path.
//!
//! The engine compiles WGSL shaders to SPIR-V at startup with `naga`, allocates
//! the model weights into a single unified (host-visible + device-local) buffer
//! on the best available iGPU, and runs the transformer forward pass as a
//! sequence of tiny compute dispatches. No `wgpu`, no `cubecl`, no Burn — just
//! `ash` on top of whatever Vulkan driver the system exposes ([`device::Chip`]).

pub mod device;
pub mod kernels;
pub mod shaders;

pub use device::{Buffer, Chip};
pub use kernels::Kernels;
pub use shaders::Dt;