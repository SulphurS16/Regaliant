//! Whether the display the game's window is on refreshes at a variable rate now (G-Sync,
//! FreeSync): the screen then shows each frame when it is presented, so the frame limit is
//! kept by the port's own timer and a late frame is shown the moment it is done, instead of
//! everything following the display's fixed ticks (`hle::libs::videoout`, PACING.md).
//!
//! What decides is the display's own behaviour (`displayclock::vrr_seen`: refreshes that come
//! with the presents instead of at the mode's steady period), whatever the vendor: it is
//! what the picture does, and where it cannot be seen (every refresh gets a frame) the two
//! kinds of screen show the same. A screen wrongly taken as fixed still gets evenly spaced
//! frames from the grid.
//!
//! The NVIDIA driver is asked as well (NVAPI through its own nvapi64.dll where installed; a
//! thread of its own, twice a second: `NvAPI_Disp_GetVRRInfo` for the monitor's display id):
//! whether variable refresh is possible there decides the refresh rate exclusive fullscreen
//! picks at Automatic (`possible_on`), and its "active" is logged beside the display's
//! evidence. The interface ids and the structure's layout are from NVIDIA's public NVAPI
//! headers (PROVENANCE.md); no code of theirs is in the port, and the library is the
//! driver's own, loaded only if present.
//!
//! `tune vrr` (`DS_VRR`): 2 follows the display (the default), 1 treats it as variable
//! refresh, 0 as fixed.

use parking_lot::Mutex;
use std::sync::OnceLock;
use std::sync::atomic::{AtomicBool, AtomicU32, Ordering::Relaxed};
use std::time::{Duration, Instant};

/// The driver's answer for the window's monitor: bit 31 set once there is one, then
/// `NV_GET_VRR_INFO`'s flags (bit 0 active now, 1 possible, 2 requested, 3 the driver's
/// indicator on, 4 the display in its variable-refresh mode).
static DRIVER: AtomicU32 = AtomicU32::new(0);
static STARTED: AtomicBool = AtomicBool::new(false);
/// Exclusive fullscreen kept the screen's own refresh rate because the driver said variable
/// refresh was possible (`possible_on`), and the screen then went on refreshing at its mode's
/// steady rate for seconds: variable refresh does not take for this game here (a driver
/// setting), so `possible_on` says no for the session and the mode is chosen again.
static NOT_TAKEN: AtomicBool = AtomicBool::new(false);
/// `possible_on` answered yes for the mode exclusive fullscreen has now.
static ASSUMED: AtomicBool = AtomicBool::new(false);
/// Display ids by device name.
static IDS: Mutex<Vec<(String, u32)>> = Mutex::new(Vec::new());

const KNOWN: u32 = 1 << 31;

type Query = unsafe extern "C" fn(u32) -> *const core::ffi::c_void;
type Init = unsafe extern "C" fn() -> i32;
type GetId = unsafe extern "C" fn(*const u8, *mut u32) -> i32;
type GetVrr = unsafe extern "C" fn(u32, *mut VrrInfo) -> i32;

/// `NV_GET_VRR_INFO_V1`: its version (the size with 1 << 16), the flags, four reserved words.
#[repr(C)]
#[derive(Default)]
struct VrrInfo {
    version: u32,
    flags: u32,
    reserved: [u32; 4],
}

struct Nv {
    get_id: GetId,
    get_vrr: GetVrr,
}

/// The driver's interface, where nvapi64.dll is installed and has the calls (made once).
fn nv() -> Option<&'static Nv> {
    static NV: OnceLock<Option<Nv>> = OnceLock::new();
    NV.get_or_init(|| {
        use windows_sys::Win32::System::LibraryLoader::{GetProcAddress, LOAD_LIBRARY_SEARCH_SYSTEM32, LoadLibraryExW};
        let name: Vec<u16> = "nvapi64.dll".encode_utf16().chain(std::iter::once(0)).collect();
        unsafe {
            // (The driver's own copy in the system directory, never one beside the game.)
            let lib = LoadLibraryExW(name.as_ptr(), std::ptr::null_mut(), LOAD_LIBRARY_SEARCH_SYSTEM32);
            if lib.is_null() {
                return None;
            }
            let query: Query = std::mem::transmute(GetProcAddress(lib, c"nvapi_QueryInterface".as_ptr() as *const u8)?);
            let get = |id: u32| {
                let p = query(id);
                (!p.is_null()).then_some(p)
            };
            let init: Init = std::mem::transmute(get(0x0150_E828)?);
            let get_id: GetId = std::mem::transmute(get(0xAE45_7190)?);
            let get_vrr: GetVrr = std::mem::transmute(get(0xDF8F_DA57)?);
            let r = init();
            if r != 0 {
                crate::info!("vrr: the NVIDIA driver's interface did not start (status {r}): the display clock's evidence serves");
                return None;
            }
            Some(Nv { get_id, get_vrr })
        }
    })
    .as_ref()
}

/// The driver's flags for `device` (a GDI device name), None where it has no answer.
fn ask(device: &str) -> Option<u32> {
    let nv = nv()?;
    let id = {
        let mut ids = IDS.lock();
        match ids.iter().find(|e| e.0 == device) {
            Some(e) => e.1,
            None => {
                let name: Vec<u8> = device.bytes().chain(std::iter::once(0)).collect();
                let mut id = 0u32;
                if unsafe { (nv.get_id)(name.as_ptr(), &mut id) } != 0 {
                    return None;
                }
                ids.push((device.to_string(), id));
                id
            }
        }
    };
    let mut info = VrrInfo { version: std::mem::size_of::<VrrInfo>() as u32 | (1 << 16), ..Default::default() };
    if unsafe { (nv.get_vrr)(id, &mut info) } != 0 {
        // (The id may be stale after a display change: asked for again next time.)
        IDS.lock().retain(|e| e.0 != device);
        return None;
    }
    Some(info.flags & 0x1F)
}

/// Starts the watch (once; it idles while there is no window: runs without one have no display).
pub fn start() {
    if STARTED.swap(true, Relaxed) {
        return;
    }
    std::thread::Builder::new()
        .name("vrr-watch".into())
        .spawn(|| {
            crate::log::set_thread_name("vrr-watch");
            let mut last: Option<(String, u32)> = None;
            // Since when exclusive fullscreen has been held with "possible" and not "active".
            let mut waiting: Option<Instant> = None;
            loop {
                std::thread::sleep(Duration::from_millis(500));
                let Some((_, device)) = crate::gpu::vk::window::current_monitor() else {
                    DRIVER.store(0, Relaxed);
                    continue;
                };
                if nv().is_none() {
                    return;
                }
                let Some(flags) = ask(&device) else {
                    DRIVER.store(0, Relaxed);
                    continue;
                };
                DRIVER.store(KNOWN | flags, Relaxed);
                // Logged: its state when the window opens and every change of "active" or
                // "possible" (the other flags move with the foreground window).
                if last.as_ref().is_none_or(|l| l.0 != device || (l.1 ^ flags) & 3 != 0) {
                    crate::info!(
                        "vrr: {device}: variable refresh {} (the driver: possible {}, requested {}, the display in its variable-refresh mode {})",
                        if flags & 1 != 0 { "active" } else { "not active" },
                        flags >> 1 & 1,
                        flags >> 2 & 1,
                        flags >> 4 & 1
                    );
                }
                last = Some((device.clone(), flags));
                // Exclusive fullscreen held and presenting at a rate kept for variable refresh,
                // and the screen refreshes at its mode's steady rate for five seconds: it does
                // not take here.
                let exclusive = crate::gpu::vk::window::exclusive_wanted().is_some() && crate::gpu::FRAMES.load(Relaxed) > 0;
                if exclusive && ASSUMED.load(Relaxed) && crate::gpu::tune::vrr() == 2 && !NOT_TAKEN.load(Relaxed) && crate::displayclock::snapshot().is_some() && !crate::displayclock::vrr_seen() {
                    let since = *waiting.get_or_insert_with(Instant::now);
                    if since.elapsed() > Duration::from_secs(5) {
                        NOT_TAKEN.store(true, Relaxed);
                        ASSUMED.store(false, Relaxed);
                        crate::info!("vrr: {device}: exclusive fullscreen for five seconds and the screen keeps its mode's steady refreshes (the driver: active {}): paced as a fixed-refresh screen, its mode chosen again", flags & 1);
                        crate::gpu::vk::window::reapply_mode();
                    }
                } else {
                    waiting = None;
                }
            }
        })
        .ok();
}

/// Whether the window's display refreshes at a variable rate now (the display's own
/// behaviour, `displayclock::vrr_seen`).
pub fn active() -> bool {
    match crate::gpu::tune::vrr() {
        0 => false,
        1 => true,
        _ => crate::displayclock::vrr_seen(),
    }
}

/// Whether variable refresh could become active on `device` in exclusive fullscreen (the
/// driver's answer, asked now; None where there is none): the screen then keeps its own
/// refresh rate there (`display::wanted_mode`).
pub fn possible_on(device: &str) -> Option<bool> {
    let r = match crate::gpu::tune::vrr() {
        0 => Some(false),
        1 => Some(true),
        _ if NOT_TAKEN.load(Relaxed) => Some(false),
        _ => ask(device).map(|f| f & 2 != 0),
    };
    ASSUMED.store(r == Some(true), Relaxed);
    r
}

/// The NVIDIA driver's word on the window's monitor now: whether it can refresh at a variable
/// rate there (None: no answer, as on other vendors' GPUs; `tune vrr` 0 and 1 answer for it).
pub fn driver_possible() -> Option<bool> {
    match crate::gpu::tune::vrr() {
        0 => Some(false),
        1 => Some(true),
        _ => {
            let d = DRIVER.load(Relaxed);
            (d & KNOWN != 0).then_some(d & 2 != 0)
        }
    }
}

/// Status line.
pub fn status_line() -> String {
    let d = DRIVER.load(Relaxed);
    let driver = if d & KNOWN != 0 { format!("the driver says active {}, possible {}, requested {}", d & 1, d >> 1 & 1, d >> 2 & 1) } else { "no driver answer".into() };
    format!(
        "    variable refresh (`tune vrr` {}): {}; the display clock {}; {driver}{}\n",
        crate::gpu::tune::vrr(),
        if active() { "taken as active" } else { "taken as off" },
        if crate::displayclock::vrr_seen() { "sees refreshes follow the presents" } else { "sees the mode's steady refreshes" },
        if NOT_TAKEN.load(Relaxed) { " (exclusive fullscreen kept the screen's rate for it and it did not take)" } else { "" }
    )
}
