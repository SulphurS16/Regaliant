//! The Windows debugger-host process is unavailable. The renderer uses its in-process
//! residency/fault path on Linux; attempts to invoke host-only operations fail explicitly.
use parking_lot::{ReentrantMutex, ReentrantMutexGuard};
static ARM: ReentrantMutex<()> = ReentrantMutex::new(());
pub const HOLD_RUNS: usize = 1024;
#[derive(Clone, Copy, Debug)]
pub struct HeldRequest {
    pub page: u64,
    pub write: bool,
    pub suspended: bool,
    pub rip: u64,
    pub tid: u32,
    pub stamp: i64,
}
pub fn on() -> bool {
    false
}
pub fn init() {}
pub fn wants_host() -> bool {
    false
}
pub fn page_held(_a: u64) -> bool {
    false
}
pub fn held(_a: u64) -> bool {
    false
}
pub fn held_pages() -> u64 {
    0
}
pub fn splitting(_lo: u64, _hi: u64) {}
pub fn set_held_answered() {}
pub fn set_held_served(_on: bool) {}
pub fn set_word_mode() {}
pub fn arm_lock() -> ReentrantMutexGuard<'static, ()> {
    ARM.lock()
}
pub fn try_arm_lock_for(d: std::time::Duration) -> Option<ReentrantMutexGuard<'static, ()>> {
    ARM.try_lock_for(d)
}
pub fn arm(_base: u64, _len: u64) -> Result<(), String> {
    Err("Linux uses in-process faults, not a debugger host".into())
}
pub fn arm_noaccess(base: u64, len: u64) -> Result<(), String> {
    arm(base, len)
}
pub fn hold(base: u64, len: u64) -> Result<(), String> {
    arm(base, len)
}
pub fn hold_by_host(runs: &[(u64, u64)]) -> Vec<bool> {
    vec![false; runs.len()]
}
pub fn leaf_end(a: u64) -> u64 {
    a.saturating_add(4096)
}
pub fn leaf_events(_a: u64) -> u64 {
    0
}
pub fn disarm(_base: u64, _len: u64) {}
pub fn armed_word(_a: u64) -> u64 {
    0
}
pub fn all_armed(_base: u64, _len: u64) -> bool {
    false
}
pub fn disarm_if_armed(_base: u64, _len: u64) {}
pub fn armed(_a: u64) -> bool {
    false
}
pub fn forget(_base: u64, _len: u64) {}
pub fn take_held(_out: &mut Vec<HeldRequest>) {}
pub fn wait_held(_ms: u32) -> bool {
    false
}
pub fn release(_base: u64, _len: u64) -> Result<u64, String> {
    Err("debugger host unavailable on Linux".into())
}
pub fn released_bits(_base: u64, _len: u64) {}
pub fn resume_waiter(_r: &HeldRequest) -> Result<u64, String> {
    Err("debugger host unavailable on Linux".into())
}
pub fn drain_faults(_out: &mut Vec<(u64, bool, u64)>) -> bool {
    false
}
pub fn drain(_out: &mut Vec<u64>) -> bool {
    false
}
pub fn faults() -> u64 {
    0
}
pub fn status() -> String {
    "Linux in-process fault service (no Windows debugger host)".into()
}
