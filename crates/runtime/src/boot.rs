//! Startup shared by the native Windows and Linux entry points.
use std::path::Path;

/// Writable mounts belong to the runtime; the supplied game tree is always read-only.
/// Linux validates these output paths before calling (including existing symlink parents).
pub fn mount_filesystems(app: &dsconfig::zar::Source, runtime: &Path) {
    use crate::hle::kernel::fs::{ensure_dir, mount};
    for name in ["temp0", "download0", "data", "devlog"] {
        ensure_dir(&runtime.join(name));
    }
    crate::hle::kernel::fs::mount_game("/app0", app);
    crate::hle::kernel::fs::mount_game("/hostapp", app);
    mount("/temp0", vec![runtime.join("temp0")], true);
    mount("/temp", vec![runtime.join("temp0")], true);
    mount("/download0", vec![runtime.join("download0")], true);
    mount("/data", vec![runtime.join("data")], true);
    ensure_dir(&runtime.join("devlog").join("logs"));
    mount("/devlog/app", vec![runtime.join("devlog")], true);
}

/// The executable's SDK and initial thread parameters (shared native bootstrap).
pub fn read_proc_param(m: &crate::loader::LoadedModule) -> (u64, String, i32) {
    let mut stack = 0u64;
    let mut name = "GAME_MainThread".to_string();
    let mut prio = crate::hle::kernel::pthread::PRIO_DEFAULT;
    if let Some(pp) = m.proc_param {
        unsafe {
            let size = *(pp as *const u64);
            let magic = *((pp + 8) as *const u32);
            let sdk = *((pp + 0x10) as *const u64);
            crate::hle::kernel::misc::SDK_VERSION.store(sdk, std::sync::atomic::Ordering::Relaxed);
            crate::info!("proc param at {pp:#x}: size {size:#x} magic {magic:#x} sdk {sdk:#x}");
            let field = |off: u64| if off + 8 <= size { *((pp + off) as *const u64) } else { 0 };
            let name_pp = field(0x20);
            if name_pp != 0 && *(name_pp as *const u64) != 0 {
                name = crate::hle::cstr(*(name_pp as *const u64) as *const u8).to_string();
            }
            let prio_p = field(0x28);
            if prio_p != 0 {
                prio = *(prio_p as *const i32);
            }
            let stack_p = field(0x30);
            if stack_p != 0 {
                stack = *(stack_p as *const u32) as u64;
            }
        }
    }
    (stack, name, prio)
}
