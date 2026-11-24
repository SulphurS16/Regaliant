pub fn patch_main(p: &mut crate::patch::Patcher, ranges: &[(u64, u64)]) {
    crate::exception::install_text_hooks(p, ranges);
    // (The engine's memory map: read only, so on with or without the title fixes.)
    crate::engine::init(p.base, p.image);
    crate::titlefix::apply(p, ranges);
}
pub fn frame_count() -> u64 { crate::gpu::FRAMES.load(std::sync::atomic::Ordering::Relaxed) }
pub fn haptics(left: f32, right: f32) { crate::input::haptics(left, right); }
pub fn arm_watch_current_thread(guest: bool) {
    crate::exception::arm_watch_current_thread(guest);
}
pub fn safe_area() -> Option<f32> { crate::pcsettings::safe_area() }
pub fn current_guest_frames(n: usize) -> Vec<String> { crate::exception::current_guest_frames(n) }
pub fn ajm_lock<T>(m: &parking_lot::Mutex<T>) -> parking_lot::MutexGuard<'_, T> { crate::gpu::lockstat::AJM.lock(m) }
pub fn guest_readable(a: u64, len: u64) -> bool { crate::gpu::cp::guest_readable(a, len) }
pub fn acm_reverb() -> bool { crate::gpu::tune::acm_reverb() }

pub fn desktop_size() -> (i32, i32) { use windows_sys::Win32::UI::WindowsAndMessaging::{GetSystemMetrics, SM_CXSCREEN, SM_CYSCREEN}; unsafe { (GetSystemMetrics(SM_CXSCREEN), GetSystemMetrics(SM_CYSCREEN)) } }
