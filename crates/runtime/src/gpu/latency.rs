//! Latency: NVIDIA Reflex Low Latency and the PC Latency markers, through Streamline's sl.reflex
//! and sl.pcl (`gpu::streamline`; ProgrammingGuideReflex.md, ProgrammingGuidePCL.md). The rest of
//! the port only names the moments of a frame (`sleep`, `marker`): another vendor's latency
//! library could take Streamline's place here.
//!
//! Reflex Low Latency (the Graphics & Display list's NVIDIA Reflex: Off, On, On + Boost; On by
//! default as NVIDIA asks) delays the start of each frame's simulation so that its GPU work does
//! not wait in a queue. DLSS Frame Generation needs it: while frame generation is chosen Reflex is
//! at least On whatever the setting (Off counts as On; On + Boost stays), as NVIDIA's guidelines
//! ask ("users should not be able to disable Reflex while Frame Generation is on").
//!
//! The frame's moments, each named by the port's frame index (the number of the flip that shows
//! the frame, the same index DLSS and DLSS Frame Generation get, `gpu::streamline`):
//! - Reflex's sleep and the simulation's start: where the title's main loop has stepped its game
//!   clock for the frame (`gamestep::after_step`), before it reads input and simulates;
//! - the simulation's end and the render submission's start: the title's render thread hands the
//!   frame's command buffers and flip to the GPU (CRenderContext::SubmitAndFlip,
//!   `gamestep::submit_hook`);
//! - the render submission's end: the port's graphics command processor reaches the frame's flip
//!   (`gpu::cp`, every Vulkan submission of the frame's work recorded);
//! - the present's start and end: around the presenter's vkQueuePresentKHR (`gpu::vk::present`).
//!
//! The PC Latency ping (a window message Streamline sends) and the flash indicator (a left click)
//! come from the window (`gpu::vk::window`). The calls are made whatever the mode, Off included,
//! as the guide requires ("slReflexSleep and slPCLSetMarker must always be called").
//!
//! Reflex's own frame rate limiter (frameLimitUs) stays unused: the port's flips are paced by
//! its own limit (`hle::libs::videoout`), and a second limiter beside it would fight it on a
//! fixed-refresh display's grid.

use super::streamline::{self as sl, SlBase, SlPclState, SlReflexOptions, SlReflexState};
use parking_lot::Mutex;
use std::sync::OnceLock;
use std::sync::atomic::{AtomicBool, AtomicU8, AtomicU32, AtomicU64, Ordering::Relaxed};

/// Reflex modes (sl::ReflexMode).
pub const OFF: u8 = 0;
pub const ON: u8 = 1;
pub const BOOST: u8 = 2;

pub fn mode_name(m: u8) -> &'static str {
    match m {
        ON => "On",
        BOOST => "On + Boost",
        _ => "Off",
    }
}

/// The frame's moments (sl::PCLMarker).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[repr(u32)]
pub enum Marker {
    SimulationStart = 0,
    SimulationEnd = 1,
    RenderSubmitStart = 2,
    RenderSubmitEnd = 3,
    PresentStart = 4,
    PresentEnd = 5,
    TriggerFlash = 7,
    PcLatencyPing = 8,
}

/// The setting (the Graphics & Display list's NVIDIA Reflex; `DS_REFLEX_MODE=0|1|2` decides in
/// development runs).
static USER_MODE: AtomicU8 = AtomicU8::new(ON);
/// The mode last given to Streamline (u8::MAX: none yet).
static APPLIED: AtomicU8 = AtomicU8::new(u8::MAX);

type FnSleep = unsafe extern "C" fn(*const std::ffi::c_void) -> i32;
type FnSetOptions = unsafe extern "C" fn(*const SlReflexOptions) -> i32;
type FnGetState = unsafe extern "C" fn(*mut SlReflexState) -> i32;
type FnPclMarker = unsafe extern "C" fn(u32, *const std::ffi::c_void) -> i32;
type FnPclGetState = unsafe extern "C" fn(*mut SlPclState) -> i32;

struct Fns {
    sleep: FnSleep,
    set_options: FnSetOptions,
    get_state: FnGetState,
    marker: FnPclMarker,
}

static FNS: OnceLock<Fns> = OnceLock::new();
/// Reflex Low Latency can run here (sl::ReflexState::lowLatencyAvailable).
static LOW_LATENCY: AtomicBool = AtomicBool::new(false);
/// The flash indicator is the driver's (the left click is sent as a marker).
static FLASH_DRIVER: AtomicBool = AtomicBool::new(false);
/// The PC Latency ping's window message (0: none).
static PING_MESSAGE: AtomicU32 = AtomicU32::new(0);
/// The last frame whose simulation started (for the ping and flash markers).
static SIM_FRAME: AtomicU32 = AtomicU32::new(0);
/// Counts for the status line: sleeps, markers, failures.
static SLEEPS: AtomicU64 = AtomicU64::new(0);
static MARKERS: AtomicU64 = AtomicU64::new(0);
static FAILED: AtomicU64 = AtomicU64::new(0);
/// slReflexSetOptions is not thread safe (sl_reflex.h).
static OPTIONS_LOCK: Mutex<()> = Mutex::new(());

fn fns() -> Option<&'static Fns> {
    FNS.get()
}

/// Streamline is ready (`gpu::streamline::device_created`, its own thread): Reflex's and PC
/// Latency's functions, the ping message, the options set once (the guide: even when Off).
pub(super) fn streamline_ready() {
    if !sl::supported(sl::FEATURE_REFLEX) || !sl::supported(sl::FEATURE_PCL) {
        return;
    }
    let f = |feature: u32, name: &std::ffi::CStr| sl::feature_fn(feature, name);
    // SAFETY: sl_reflex.h's and sl_pcl.h's signatures (references are pointers).
    let fns = unsafe {
        (|| -> Option<Fns> {
            Some(Fns {
                sleep: std::mem::transmute(f(sl::FEATURE_REFLEX, c"slReflexSleep")?),
                set_options: std::mem::transmute(f(sl::FEATURE_REFLEX, c"slReflexSetOptions")?),
                get_state: std::mem::transmute(f(sl::FEATURE_REFLEX, c"slReflexGetState")?),
                marker: std::mem::transmute(f(sl::FEATURE_PCL, c"slPCLSetMarker")?),
            })
        })()
    };
    let Some(fns) = fns else {
        crate::warn!("reflex: Streamline's Reflex or PC Latency functions are missing: no Reflex");
        return;
    };
    if let Some(get) = sl::feature_fn(sl::FEATURE_PCL, c"slPCLGetState") {
        let get: FnPclGetState = unsafe { std::mem::transmute(get) };
        let mut st = SlPclState { base: SlBase::new(sl::SL_PCL_STATE, 1), stats_window_message: 0 };
        if unsafe { get(&mut st) } == sl::SL_OK {
            PING_MESSAGE.store(st.stats_window_message, Relaxed);
        }
    }
    let _ = FNS.set(fns);
    refresh_state();
    apply();
    crate::info!(
        "reflex: ready (low latency {}, flash indicator {}, PC Latency ping message {:#x}); NVIDIA Reflex {} chosen",
        if LOW_LATENCY.load(Relaxed) { "available" } else { "not available on this GPU or driver: PC Latency markers only" },
        if FLASH_DRIVER.load(Relaxed) { "the driver's" } else { "off" },
        PING_MESSAGE.load(Relaxed),
        mode_name(user_mode())
    );
}

/// Reads Reflex's state (low latency's availability, the flash indicator's owner).
fn refresh_state() {
    let Some(f) = fns() else { return };
    let _g = OPTIONS_LOCK.lock();
    let mut st = SlReflexState::boxed();
    if unsafe { (f.get_state)(&mut *st) } == sl::SL_OK {
        LOW_LATENCY.store(st.low_latency_available, Relaxed);
        FLASH_DRIVER.store(st.flash_indicator_driver_controlled, Relaxed);
        if st.stats_window_message != 0 {
            PING_MESSAGE.store(st.stats_window_message, Relaxed);
        }
    }
}

/// Reflex's functions are there (the markers and the sleep are sent).
pub fn active() -> bool {
    fns().is_some()
}

/// Reflex Low Latency can run here (the setting does something).
pub fn available() -> bool {
    LOW_LATENCY.load(Relaxed)
}

/// The setting (0 Off, 1 On, 2 On + Boost).
pub fn user_mode() -> u8 {
    static ENV: OnceLock<Option<u8>> = OnceLock::new();
    if let Some(m) = *ENV.get_or_init(|| std::env::var("DS_REFLEX_MODE").ok().and_then(|v| v.trim().parse::<u8>().ok()).map(|v| v.min(BOOST))) {
        return m;
    }
    USER_MODE.load(Relaxed)
}

/// The mode in effect: the setting, at least On while frame generation is chosen.
pub fn effective_mode() -> u8 {
    let m = user_mode();
    if m == OFF && super::framegen::requires_reflex() { ON } else { m }
}

/// The setting changed (`pcsettings::apply`).
pub fn set_user_mode(m: u8) {
    let m = m.min(BOOST);
    if USER_MODE.swap(m, Relaxed) != m {
        crate::info!("reflex: NVIDIA Reflex {} chosen{}", mode_name(m), if m == OFF && super::framegen::requires_reflex() { " (On while frame generation is on)" } else { "" });
    }
    apply();
}

/// Gives Streamline the mode in effect when it changed (`force`: again, after a new swapchain:
/// Vulkan's latency mode belongs to the swapchain).
pub fn apply_with(force: bool) {
    let Some(f) = fns() else { return };
    let m = effective_mode();
    if !force && APPLIED.load(Relaxed) == m {
        return;
    }
    let _g = OPTIONS_LOCK.lock();
    let opts = SlReflexOptions { base: SlBase::new(sl::SL_REFLEX_OPTIONS, 1), mode: m as u32, frame_limit_us: 0, use_markers_to_optimize: false, virtual_key: 0, id_thread: 0 };
    let r = unsafe { (f.set_options)(&opts) };
    if r != sl::SL_OK {
        crate::warn!("reflex: slReflexSetOptions({}) failed ({r}: {})", mode_name(m), sl::result_name(r));
        return;
    }
    if APPLIED.swap(m, Relaxed) != m {
        crate::info!("reflex: low latency mode {}{}", mode_name(m), if available() { "" } else { " (not available here: markers only)" });
    }
}

pub fn apply() {
    apply_with(false);
}

/// A swapchain was made through Streamline (its latency mode is the new swapchain's).
pub fn swapchain_created() {
    apply_with(true);
    refresh_state();
}

/// Reflex's sleep for frame `frame` (before its simulation; any thread, blocks).
pub fn sleep(frame: u32) {
    let Some(f) = fns() else { return };
    let Some(t) = sl::token(frame) else { return };
    let r = unsafe { (f.sleep)(token_ptr(t)) };
    SLEEPS.fetch_add(1, Relaxed);
    if r != sl::SL_OK && FAILED.fetch_add(1, Relaxed) < 4 {
        crate::warn!("reflex: slReflexSleep failed ({r}: {})", sl::result_name(r));
    }
}

fn token_ptr(t: sl::Token) -> *const std::ffi::c_void {
    t.ptr()
}

/// A frame's moment.
pub fn marker(m: Marker, frame: u32) {
    let Some(f) = fns() else { return };
    let Some(t) = sl::token(frame) else { return };
    let r = unsafe { (f.marker)(m as u32, token_ptr(t)) };
    MARKERS.fetch_add(1, Relaxed);
    if r != sl::SL_OK && FAILED.fetch_add(1, Relaxed) < 4 {
        crate::warn!("reflex: slPCLSetMarker({m:?}, {frame}) failed ({r}: {})", sl::result_name(r));
    }
}

/// The title's main loop starts frame `frame`'s simulation (`gamestep::after_step`): Reflex's
/// sleep, then the simulation's start. Frame indices only move forward here.
pub fn simulation_start(frame: u32) {
    if fns().is_none() {
        return;
    }
    let last = SIM_FRAME.load(Relaxed);
    let frame = if frame <= last { last + 1 } else { frame };
    SIM_FRAME.store(frame, Relaxed);
    sleep(frame);
    marker(Marker::SimulationStart, frame);
}

/// The title's render thread hands frame `frame` to the GPU (`gamestep::submit_hook`).
pub fn render_submit(frame: u32) {
    if fns().is_none() {
        return;
    }
    marker(Marker::SimulationEnd, frame);
    marker(Marker::RenderSubmitStart, frame);
}

/// Whether `msg` is the PC Latency ping (the window's message loop).
pub fn is_ping(msg: u32) -> bool {
    let m = PING_MESSAGE.load(Relaxed);
    m != 0 && msg == m
}

/// The PC Latency ping arrived: the next frame's simulation picks it up.
pub fn ping() {
    marker(Marker::PcLatencyPing, SIM_FRAME.load(Relaxed) + 1);
}

/// A left click (the flash indicator, where the driver shows it).
pub fn trigger_flash() {
    if FLASH_DRIVER.load(Relaxed) {
        marker(Marker::TriggerFlash, SIM_FRAME.load(Relaxed));
    }
}

/// Status line.
pub fn status_line() -> String {
    if fns().is_none() {
        return String::new();
    }
    format!(
        "    reflex: {} (chosen {}, low latency {}), {} sleeps, {} markers, {} failed\n",
        mode_name(effective_mode()),
        mode_name(user_mode()),
        if available() { "available" } else { "not available" },
        SLEEPS.load(Relaxed),
        MARKERS.load(Relaxed),
        FAILED.load(Relaxed)
    )
}
