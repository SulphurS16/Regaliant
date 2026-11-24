pub fn patch_main(p: &mut crate::patch::Patcher, ranges: &[(u64, u64)]) {
    crate::text_hooks::install_text_hooks(p, ranges);
    crate::engine::init(p.base, p.image);
    crate::titlefix::apply(p, ranges);
}
pub fn frame_count() -> u64 {
    crate::gpu::FRAMES.load(std::sync::atomic::Ordering::Relaxed)
}
pub fn haptics(left: f32, right: f32) {
    crate::input::haptics(left, right);
}
pub fn arm_watch_current_thread(guest: bool) {
    let _ = guest; // Hardware debug-register watchpoints are not ported.
}
pub fn safe_area() -> Option<f32> {
    crate::pcsettings::safe_area()
}
pub fn current_guest_frames(n: usize) -> Vec<String> {
    crate::exception::current_guest_frames(n)
}
pub fn ajm_lock<T>(m: &parking_lot::Mutex<T>) -> parking_lot::MutexGuard<'_, T> {
    m.lock()
}
pub fn guest_readable(a: u64, len: u64) -> bool {
    let Some(end) = a
        .checked_add(len)
        .filter(|&e| a >= crate::mem::GUEST_BASE && e <= crate::mem::GUEST_END)
    else {
        return false;
    };
    let memory = crate::mem::MEM.lock();
    let mut at = a;
    while at < end {
        let Some(vma) = memory.find(at) else {
            return false;
        };
        if vma.prot & crate::mem::PROT_CPU_READ == 0 {
            return false;
        }
        at = vma.end().min(end);
    }
    true
}
pub fn acm_reverb() -> bool {
    std::env::var("DS_ACM_REVERB").ok().as_deref() != Some("0")
}

pub fn desktop_size() -> (i32, i32) {
    (
        crate::gpu::vk::window::CLIENT_W.load(std::sync::atomic::Ordering::Relaxed) as i32,
        crate::gpu::vk::window::CLIENT_H.load(std::sync::atomic::Ordering::Relaxed) as i32,
    )
}
