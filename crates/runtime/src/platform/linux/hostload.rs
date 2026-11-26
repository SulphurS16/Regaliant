//! Cross-process Windows GPU performance counters are unavailable.
pub fn others_vram() -> Option<u64> {
    static SIM: std::sync::OnceLock<Option<u64>> = std::sync::OnceLock::new();
    *SIM.get_or_init(|| std::env::var("DS_SIM_OTHERS_MIB").ok().and_then(|v| v.trim().parse::<u64>().ok()).map(|m| m.saturating_mul(1 << 20)))
}
pub fn start() {}
pub fn window_report_and_reset() -> String {
    "external GPU load unavailable on Linux".into()
}
