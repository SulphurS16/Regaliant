//! Windows sampled-thread diagnostics are not available; no fabricated samples.
pub fn start() {}
pub fn reset() {}
pub fn report() -> String {
    "guest profiler unavailable on Linux".into()
}
pub fn start_trace() {}
pub fn explain(_lo: u64, _hi: u64) -> Option<String> {
    None
}
pub fn explain_cp(_lo: u64, _hi: u64) -> Option<String> {
    None
}
pub fn explain_par(_lo: u64, _hi: u64) -> Option<String> {
    None
}
pub fn explain_submit(_lo: u64, _hi: u64) -> Option<String> {
    None
}
