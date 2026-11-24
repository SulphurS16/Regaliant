//! Compositor vblank timing is not exposed by this backend yet.
use std::time::{Duration, Instant};
#[derive(Clone, Copy, Debug)]
pub struct Snap {
    /// When the last vblank came and its number.
    pub last: Instant,
    pub count: u64,
    /// The refresh period measured over the last second (up to 128 vblanks).
    pub period: Duration,
    /// The mode's refresh rate (Hz, as Windows gives it: whole numbers).
    pub mode_hz: u32,
}

pub fn start() {}
pub fn snapshot() -> Option<Snap> {
    None
}
pub fn lock() -> Option<Snap> {
    None
}
pub fn forced_refreshes() -> Option<u32> {
    None
}
pub fn vrr_seen() -> bool {
    false
}
pub fn refreshes_for(hz: f64, fps: f64) -> u32 {
    (hz / fps.max(1.0)).round().max(1.0) as u32
}
pub fn sim_hz() -> Option<f64> {
    None
}
pub fn reset_cadence() {}
pub fn note_present() {}
pub fn window_clause() -> String {
    String::new()
}
pub fn status_line() -> String {
    "Linux compositor vblank clock unavailable".into()
}

pub static LOCKS: std::sync::atomic::AtomicU32 = std::sync::atomic::AtomicU32::new(0);
pub static FLIPS_APART: std::sync::atomic::AtomicU32 = std::sync::atomic::AtomicU32::new(0);

/// A free-running software grid is not the compositor's refresh clock. Keep it only
/// for diagnosis; otherwise use videoout's existing ready-frame/cap-deadline path.
pub fn timer_grid() -> bool {
    static GRID: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *GRID.get_or_init(|| std::env::var("DS_LINUX_VBLANK_GRID").ok().as_deref() == Some("1"))
}
pub static MODE_MILLIHZ: std::sync::atomic::AtomicU32 = std::sync::atomic::AtomicU32::new(60000);
pub fn nominal_refresh() -> f64 { MODE_MILLIHZ.load(std::sync::atomic::Ordering::Relaxed) as f64 / 1000.0 }
