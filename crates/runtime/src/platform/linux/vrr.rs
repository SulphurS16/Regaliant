//! VRR/display mode control is disabled on Linux.
pub fn start() {}
pub fn active() -> bool {
    false
}
pub fn possible_on(_device: &str) -> Option<bool> {
    None
}
pub fn status_line() -> String {
    "Linux VRR control disabled".into()
}
