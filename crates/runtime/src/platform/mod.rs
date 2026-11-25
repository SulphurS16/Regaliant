//! Host services used by the native guest loader. Windows implementations retain the original
//! APIs; Linux keeps the host FS base intact and gives patched guest TLS its own GS area.
#[cfg(windows)]
#[path = "windows/tls.rs"]
pub mod tls;
#[cfg(target_os = "linux")]
#[path = "linux/tls.rs"]
pub mod tls;
#[cfg(windows)]
#[path = "windows/cache.rs"]
mod cache;
#[cfg(target_os = "linux")]
#[path = "linux/cache.rs"]
mod cache;
pub use cache::flush_instruction_cache;

#[cfg(target_os = "linux")]
#[path = "linux/vm.rs"]
pub mod vm;
#[cfg(target_os = "linux")]
#[path = "linux/signals.rs"]
pub mod signals;
#[cfg(target_os = "linux")]
#[path = "linux/files.rs"]
pub mod files;
#[cfg(windows)]
#[path = "windows/loader_hooks.rs"]
pub mod loader_hooks;
#[cfg(target_os = "linux")]
#[path = "linux/loader_hooks.rs"]
pub mod loader_hooks;
#[cfg(windows)]
#[path = "windows/stack.rs"]
pub mod stack;
#[cfg(target_os = "linux")]
#[path = "linux/stack.rs"]
pub mod stack;
#[cfg(windows)]
#[path = "windows/time.rs"]
pub mod time;
#[cfg(target_os = "linux")]
#[path = "linux/time.rs"]
pub mod time;
#[cfg(windows)]
#[path = "windows/files.rs"]
pub mod files;
#[cfg(windows)]
#[path = "windows/file_hooks.rs"]
pub mod file_hooks;
#[cfg(target_os = "linux")]
#[path = "linux/file_hooks.rs"]
pub mod file_hooks;
#[cfg(target_os = "linux")]
#[path = "linux/frontend.rs"]
pub mod frontend;
pub mod graphics;

#[cfg(windows)]
pub mod os { pub use windows_sys::*; pub use std::os::windows::ffi::OsStrExt; }
#[cfg(target_os = "linux")]
#[path = "linux/os.rs"]
pub mod os;

#[cfg(windows)]
pub mod controller {
    pub use windows_sys::Win32::UI::Input::XboxController::*;
    pub fn prompt_style() -> u8 { 2 }
    pub fn poll_disconnected() -> bool { false }
    pub fn initial_sample_active() -> bool { false }
}
#[cfg(target_os = "linux")]
#[path = "linux/controller.rs"]
pub mod controller;

#[cfg(windows)]
#[path = "windows/thread_times.rs"]
pub mod thread_times;
#[cfg(target_os = "linux")]
#[path = "linux/thread_times.rs"]
pub mod thread_times;

#[cfg(all(test, target_os = "linux"))]
#[path = "linux/mem_tests.rs"]
mod mem_tests;

#[cfg(target_os = "linux")]
#[path = "linux/text.rs"]
pub mod text;

#[cfg(target_os = "linux")]
#[path = "linux/fpu.rs"]
pub mod fpu;

#[cfg(target_os = "linux")]
#[path = "linux/write_watch.rs"]
pub mod write_watch;

#[cfg(target_os = "linux")]
#[path = "linux/present_timing.rs"]
pub mod present_timing;

#[cfg(target_os = "linux")]
#[path = "linux/prepare.rs"]
pub mod prepare;

#[cfg(target_os = "linux")]
#[path = "linux/stalls.rs"]
pub mod stalls;

#[cfg(target_os = "linux")]
#[path = "linux/sparse.rs"]
pub mod sparse;

#[cfg(target_os = "linux")]
#[path = "linux/steady.rs"]
pub mod steady;

#[cfg(target_os = "linux")]
#[path = "linux/hang.rs"]
pub mod hang;

#[cfg(windows)]
#[path = "windows/memory_diagnostics.rs"]
pub mod memory_diagnostics;
#[cfg(target_os = "linux")]
#[path = "linux/memory_diagnostics.rs"]
pub mod memory_diagnostics;
