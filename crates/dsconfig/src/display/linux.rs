// Display enumeration is pending the Linux window backend. No invented monitor or HDR state.
use super::{Mode, Monitor};
pub fn targets() -> Vec<(String, String, bool, bool, f32)> { Vec::new() }
pub fn monitors() -> Vec<Monitor> { Vec::new() }
pub fn modes(_device: &str) -> Vec<Mode> { Vec::new() }
pub fn current_mode(_device: &str) -> Option<Mode> { None }
pub fn registry_mode(_device: &str) -> Option<Mode> { None }
pub fn signal(_device: &str) -> Option<(&'static str, u32)> { None }
