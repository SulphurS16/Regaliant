//! Linux never changes desktop modes; compositor HDR control is unavailable.
use super::Mode;
pub fn set_mode(_device: &str, _mode: Mode) -> bool { false }
pub fn restore() {}
pub fn restore_on_exit() {}
pub fn hdr_now(_device: &str) -> Option<f32> { None }
