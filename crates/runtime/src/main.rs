//! dsport: runs the PS5 Demon's Souls executable (PPSA01341 01.007.000) natively
//! on Windows x86-64 with high-level emulation of the PS5 system libraries.

// No console window: players start it from the launcher or Explorer (redirected output still
// reaches development tools).
#![cfg_attr(windows, windows_subsystem = "windows")]

mod platform;
mod boot;
include!("diagnostics.rs");
#[macro_use]
mod log;
mod ehframe;
mod achievements;
mod audio;
mod cpucheck;
#[cfg_attr(target_os = "linux", path = "platform/linux/exception.rs")]
mod exception;
mod text_hooks;
mod gpu;
#[cfg_attr(target_os = "linux", path = "platform/linux/guestprof.rs")]
mod guestprof;
mod heap;
mod heapsites;
#[cfg_attr(target_os = "linux", path = "platform/linux/hosted.rs")]
mod hosted;
mod hostcensus;
mod placement;
#[cfg_attr(target_os = "linux", path = "platform/linux/hostload.rs")]
mod hostload;
mod hle;
mod input;
mod loader;
#[cfg(windows)]
mod pspad;
#[cfg(target_os = "linux")]
#[path = "platform/linux/pspad.rs"]
mod pspad;
mod mem;
mod sections;
mod spatial;
mod patch;
mod pcsettings;
mod jobidle;
mod camhook;
mod gamestep;
mod game;
mod dispmode;
mod display;
#[cfg_attr(target_os = "linux", path = "platform/linux/displayclock.rs")]
mod displayclock;
mod jitter;
mod engine;
mod thread;
mod titlefix;
mod vgdiag;
mod cullread;
mod cpudraw;
mod titlemem;
mod textbox;
mod uistate;
#[cfg_attr(target_os = "linux", path = "platform/linux/vrr.rs")]
mod vrr;
mod tls;
// The shared self-play also contains the Windows launcher's tour/preparation entry points.
#[cfg_attr(target_os = "linux", allow(dead_code))]
mod prepplay;

#[cfg(windows)]
include!("windows_main.rs");
#[cfg(target_os = "linux")]
include!("linux_main.rs");
#[cfg(not(any(windows, target_os = "linux")))]
compile_error!("Regaliant supports Windows and Linux x86-64");
