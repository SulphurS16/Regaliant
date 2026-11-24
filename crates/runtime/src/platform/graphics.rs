//! Shader-construction hooks into the existing renderer.
pub use crate::gpu::fx::Map;
pub use crate::gpu::tune::{plan_fast, precompile};
pub use crate::gpu::precompile::{created_graphics, created};
pub use crate::gpu::{submit, GFX};

#[cfg(windows)]
pub fn sparse_buffer_binds(input: &[ash::vk::SparseMemoryBind]) -> std::borrow::Cow<'_, [ash::vk::SparseMemoryBind]> {
    std::borrow::Cow::Borrowed(input)
}
#[cfg(target_os = "linux")]
pub use super::sparse::binds as sparse_buffer_binds;
