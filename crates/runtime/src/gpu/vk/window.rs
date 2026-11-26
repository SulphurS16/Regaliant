//! Output window (Win32) running its own message loop thread.
//!
//! Keyboard: key states by virtual-key code (`key_down`), with the left and
//! right Shift, Ctrl and Alt keys apart as well as together.
//!
//! Mouse (`input`): device motion from WM_INPUT (Raw Input: the counts the
//! mouse reports, no pointer acceleration), its buttons and wheel. A click
//! into the client area captures the cursor: hidden and held at the client
//! area's centre while the window is focused, so motion only turns the
//! camera. Leaving the window (Alt+Tab) releases it until the window is
//! focused again; F1 releases it or takes it back. Buttons and motion count
//! only while it is captured (the capturing click is not an attack).
//!
//! Modes: the window is per-monitor DPI aware (its client area in physical
//! pixels, which the presenter fits the frame to) and opens at 1280x720
//! scaled by its monitor's DPI, inside the work area and centred
//! (`DS_WINDOW_SIZE=<w>x<h>`: that client size in pixels). The PC tab's Window
//! Mode: windowed, borderless fullscreen or exclusive fullscreen, on the PC tab's
//! Monitor (`crate::display`); exclusive fullscreen sets that monitor to the chosen
//! Resolution and Refresh Rate and gives it back when the game goes to the
//! background (Alt+Tab: the window minimizes) or ends. Alt+Enter or F11 switches
//! between the window and the last fullscreen mode (`DS_FULLSCREEN=1`: borderless
//! from the start, `2` exclusive, `0` windowed); neither reaches the game as a key.
//! The presenter keeps the driver from making a borderless window exclusive on its
//! own (a driver may change the display's mode for a window covering it). `DS_WINDOW_BEHIND=1`
//! (test runs on a shared desktop): shown without activation at the bottom of the
//! z-order, fullscreen too.
//!
//! Menu pointer (`pcsettings::pointer`): while a menu is open with the keyboard
//! and mouse in use, the cursor is free and shows the port's arrow; its client
//! position, clicks and wheel go to the pointer instead of the layout's mouse
//! bindings.

use std::sync::OnceLock;
use std::sync::atomic::{AtomicBool, AtomicI32, AtomicI64, AtomicU32, Ordering};
use windows_sys::Win32::Foundation::{HWND, LPARAM, LRESULT, POINT, RECT, WPARAM};
use windows_sys::Win32::Graphics::Gdi::{ClientToScreen, GetMonitorInfoW, MONITOR_DEFAULTTONEAREST, MONITORINFO, MonitorFromWindow};
use windows_sys::Win32::UI::HiDpi::{AdjustWindowRectExForDpi, DPI_AWARENESS_CONTEXT_PER_MONITOR_AWARE_V2, GetDpiForWindow, SetThreadDpiAwarenessContext};
use windows_sys::Win32::System::LibraryLoader::GetModuleHandleW;
use windows_sys::Win32::UI::WindowsAndMessaging::*;

pub struct Window {
    pub hwnd: isize,
    pub hinstance: isize,
}

static WINDOW: OnceLock<Window> = OnceLock::new();
pub static CLOSED: AtomicBool = AtomicBool::new(false);
pub static RESIZED: AtomicBool = AtomicBool::new(false);
/// The whole process was made per-monitor DPI aware at its start (`main`): every thread sees
/// the screen's real pixels, the swapchain's thread too.
pub static PROCESS_DPI_AWARE: AtomicBool = AtomicBool::new(false);
pub static CLIENT_W: AtomicU32 = AtomicU32::new(1280);
pub static CLIENT_H: AtomicU32 = AtomicU32::new(720);
/// Virtual-key states (true = down).
pub static KEYS: [AtomicBool; 256] = [const { AtomicBool::new(false) }; 256];

/// Mouse motion not yet taken (`input`), in device counts.
pub static MOUSE_DX: AtomicI64 = AtomicI64::new(0);
pub static MOUSE_DY: AtomicI64 = AtomicI64::new(0);
/// Wheel movement not yet taken, in WHEEL_DELTA (120) units per notch.
pub static MOUSE_WHEEL: AtomicI32 = AtomicI32::new(0);
/// Mouse buttons down: bit 0 left, 1 right, 2 middle, 3 and 4 the side
/// buttons (X1, X2).
pub static MOUSE_BUTTONS: AtomicU32 = AtomicU32::new(0);
/// Mouse buttons kept from the game until they are released (bit as `MOUSE_BUTTONS`): a click the
/// save list took (`hle::libs::savedata`).
pub static MOUSE_SWALLOW: AtomicU32 = AtomicU32::new(0);
/// The cursor is captured (hidden and held in the client area).
pub static CAPTURED: AtomicBool = AtomicBool::new(false);
static FOCUSED: AtomicBool = AtomicBool::new(false);
/// Whether a focused window captures the cursor: from the first click into
/// the client area on (not at launch: the window appears while the title
/// boots); F1 toggles.
static CAPTURE_WANTED: AtomicBool = AtomicBool::new(false);
/// Diagnosis (control channel `capture`): input as if captured without a
/// focused window (headless runs, injected motion).
pub static CAPTURE_TEST: AtomicBool = AtomicBool::new(false);
/// The last absolute position of a mouse reporting absolute coordinates
/// (remote sessions, tablets; i64::MIN: none yet).
static ABS_X: AtomicI64 = AtomicI64::new(i64::MIN);
static ABS_Y: AtomicI64 = AtomicI64::new(i64::MIN);
/// Borderless fullscreen (`apply_fullscreen`).
pub static FULLSCREEN: AtomicBool = AtomicBool::new(false);
/// The window mode in effect: 0 windowed, 1 borderless fullscreen, 2 exclusive fullscreen.
static WINDOW_MODE: AtomicU32 = AtomicU32::new(0);
/// The fullscreen mode Alt+Enter returns to (1 or 2).
static LAST_FULL: AtomicU32 = AtomicU32::new(1);
/// Exclusive fullscreen held (the presenter acquires it), and given up while the game is in the
/// background.
static EXCLUSIVE: AtomicBool = AtomicBool::new(false);
static SUSPENDED: AtomicBool = AtomicBool::new(false);
/// The monitor a fullscreen window covers (HMONITOR; the presenter's exclusive mode names it).
static HMON: std::sync::atomic::AtomicIsize = std::sync::atomic::AtomicIsize::new(0);
/// The windowed placement fullscreen returns to.
static PLACEMENT: parking_lot::Mutex<Option<WINDOWPLACEMENT>> = parking_lot::Mutex::new(None);
/// Asks the UI thread for a mode (wParam 0 windowed, 1 fullscreen, 2 the
/// other one).
const WM_DS_MODE: u32 = WM_APP + 1;
/// `DS_WINDOW_BEHIND=1`: never activated or raised (test runs).
static BEHIND: AtomicBool = AtomicBool::new(false);
/// Asks the UI thread for a client size (wParam width, lParam height).
const WM_DS_SIZE: u32 = WM_APP + 2;
/// The menu pointer turned on or off (`pointer_mode_changed`).
const WM_DS_POINTER: u32 = WM_APP + 3;

/// The menu pointer's cursor (made on the UI thread when first shown).
static POINTER_CURSOR: std::sync::atomic::AtomicUsize = std::sync::atomic::AtomicUsize::new(0);

fn pointer_on() -> bool {
    crate::pcsettings::pointer::ACTIVE.load(Ordering::Relaxed)
}

/// From any thread: the menu pointer turned on or off (the cursor is freed or taken back).
pub fn pointer_mode_changed() {
    if let Some(w) = WINDOW.get() {
        unsafe { PostMessageW(w.hwnd as HWND, WM_DS_POINTER, 0, 0) };
    }
}

/// The pointer's pixels (n x n, premultiplied, grey: any channel order): an arrow in the game's
/// glyph colours (white, a dark outline), its tip at (n/24, n/24).
pub fn pointer_pixels(n: usize) -> Vec<u8> {
    // The arrow on a 16-unit grid, its tip at (1, 1).
    let pts: [(f32, f32); 7] = [(1.0, 1.0), (1.0, 15.5), (4.6, 12.2), (7.4, 17.8), (9.9, 16.8), (7.2, 11.3), (12.0, 11.3)];
    let inside = |x: f32, y: f32| {
        let mut c = false;
        let mut j = pts.len() - 1;
        for i in 0..pts.len() {
            let (xi, yi) = pts[i];
            let (xj, yj) = pts[j];
            if (yi > y) != (yj > y) && x < (xj - xi) * (y - yi) / (yj - yi) + xi {
                c = !c;
            }
            j = i;
        }
        c
    };
    let dist = |x: f32, y: f32| {
        let mut d = f32::MAX;
        for i in 0..pts.len() {
            let (ax, ay) = pts[i];
            let (bx, by) = pts[(i + 1) % pts.len()];
            let (dx, dy) = (bx - ax, by - ay);
            let t = (((x - ax) * dx + (y - ay) * dy) / (dx * dx + dy * dy)).clamp(0.0, 1.0);
            d = d.min(((ax + t * dx - x).powi(2) + (ay + t * dy - y).powi(2)).sqrt());
        }
        d
    };
    let unit = n as f32 / 24.0; // the 16-unit arrow in two thirds of the cursor
    let mut px = vec![0u8; n * n * 4];
    for y in 0..n {
        for x in 0..n {
            let (mut white, mut dark) = (0u32, 0u32);
            for sy in 0..4 {
                for sx in 0..4 {
                    let (u, v) = ((x as f32 + (sx as f32 + 0.5) / 4.0) / unit, (y as f32 + (sy as f32 + 0.5) / 4.0) / unit);
                    if inside(u, v) && dist(u, v) > 0.9 {
                        white += 1;
                    } else if inside(u, v) || dist(u, v) <= 0.7 {
                        dark += 1;
                    }
                }
            }
            let a = (white + dark) as f32 / 16.0;
            let c = if white + dark == 0 { 0.0 } else { white as f32 / (white + dark) as f32 * 0.94 + 0.05 };
            let o = (y * n + x) * 4;
            let v = (c * a * 255.0 + 0.5) as u8;
            px[o..o + 4].copy_from_slice(&[v, v, v, (a * 255.0 + 0.5) as u8]);
        }
    }
    px
}

/// The pointer's cursor: an arrow in the game's glyph colours (white, a dark outline), its
/// size the system cursor's at the window's DPI.
unsafe fn pointer_cursor(hwnd: HWND) -> HCURSOR {
    let have = POINTER_CURSOR.load(Ordering::Relaxed);
    if have != 0 {
        return have as HCURSOR;
    }
    use windows_sys::Win32::Graphics::Gdi::*;
    let dpi = unsafe { GetDpiForWindow(hwnd) }.max(96);
    let n = ((32 * dpi + 48) / 96).clamp(32, 128) as usize;
    let px = pointer_pixels(n);
    let unit = n as f32 / 24.0;
    unsafe {
        let mut bmi: BITMAPINFO = std::mem::zeroed();
        bmi.bmiHeader.biSize = std::mem::size_of::<BITMAPINFOHEADER>() as u32;
        bmi.bmiHeader.biWidth = n as i32;
        bmi.bmiHeader.biHeight = -(n as i32);
        bmi.bmiHeader.biPlanes = 1;
        bmi.bmiHeader.biBitCount = 32;
        bmi.bmiHeader.biCompression = BI_RGB;
        let mut bits: *mut std::ffi::c_void = std::ptr::null_mut();
        let color = CreateDIBSection(std::ptr::null_mut(), &bmi, DIB_RGB_COLORS, &mut bits, std::ptr::null_mut(), 0);
        let mask = CreateBitmap(n as i32, n as i32, 1, 1, std::ptr::null());
        if color.is_null() || bits.is_null() || mask.is_null() {
            return LoadCursorW(std::ptr::null_mut(), IDC_ARROW);
        }
        std::ptr::copy_nonoverlapping(px.as_ptr(), bits as *mut u8, px.len());
        let hot = (unit * 1.0) as u32;
        let ii = ICONINFO { fIcon: 0, xHotspot: hot, yHotspot: hot, hbmMask: mask, hbmColor: color };
        let cur = CreateIconIndirect(&ii);
        DeleteObject(color as HGDIOBJ);
        DeleteObject(mask as HGDIOBJ);
        if cur.is_null() {
            return LoadCursorW(std::ptr::null_mut(), IDC_ARROW);
        }
        POINTER_CURSOR.store(cur as usize, Ordering::Relaxed);
        cur as HCURSOR
    }
}

pub fn get() -> Option<&'static Window> {
    WINDOW.get()
}

pub fn key_down(vk: u8) -> bool {
    KEYS[vk as usize].load(Ordering::Relaxed)
}

/// The left or right key of a Shift, Ctrl or Alt key message (VK_LSHIFT ..
/// VK_RMENU): Shift by its scan code, Ctrl and Alt by the extended-key bit.
fn sided(wp: WPARAM, lp: LPARAM) -> Option<u8> {
    let extended = (lp >> 24) & 1 != 0;
    match wp as u8 {
        0x10 => Some(if (lp >> 16) & 0xFF == 0x36 { 0xA1 } else { 0xA0 }),
        0x11 => Some(if extended { 0xA3 } else { 0xA2 }),
        0x12 => Some(if extended { 0xA5 } else { 0xA4 }),
        _ => None,
    }
}

fn set_key(wp: WPARAM, lp: LPARAM, down: bool) {
    KEYS[(wp & 0xFF) as usize].store(down, Ordering::Relaxed);
    if let Some(k) = sided(wp, lp) {
        KEYS[k as usize].store(down, Ordering::Relaxed);
        // The common key is down while either side is.
        let (l, r) = match wp as u8 {
            0x10 => (0xA0, 0xA1),
            0x11 => (0xA2, 0xA3),
            _ => (0xA4, 0xA5),
        };
        KEYS[(wp & 0xFF) as usize].store(key_down(l) || key_down(r), Ordering::Relaxed);
    }
}

/// Captures the cursor while the window is focused and capture is wanted
/// (held at the client area's centre: Raw Input keeps reporting motion),
/// else releases it.
unsafe fn update_capture(hwnd: HWND) {
    if FOCUSED.load(Ordering::Relaxed) && CAPTURE_WANTED.load(Ordering::Relaxed) && !pointer_on() {
        let mut r: RECT = unsafe { std::mem::zeroed() };
        unsafe { GetClientRect(hwnd, &mut r) };
        let mut p = POINT { x: (r.left + r.right) / 2, y: (r.top + r.bottom) / 2 };
        unsafe { ClientToScreen(hwnd, &mut p) };
        let c = RECT { left: p.x, top: p.y, right: p.x + 1, bottom: p.y + 1 };
        unsafe { ClipCursor(&c) };
        if !CAPTURED.swap(true, Ordering::SeqCst) {
            unsafe { SetCursor(std::ptr::null_mut()) };
            MOUSE_DX.store(0, Ordering::Relaxed);
            MOUSE_DY.store(0, Ordering::Relaxed);
            ABS_X.store(i64::MIN, Ordering::Relaxed);
        }
    } else if CAPTURED.swap(false, Ordering::SeqCst) {
        unsafe {
            ClipCursor(std::ptr::null());
            SetCursor(LoadCursorW(std::ptr::null_mut(), IDC_ARROW));
        }
        MOUSE_BUTTONS.store(0, Ordering::Relaxed);
        MOUSE_WHEEL.store(0, Ordering::Relaxed);
    }
}

/// Asks the window for borderless fullscreen (`Some(true)`), windowed
/// (`Some(false)`) or the other one (`None`); from any thread.
pub fn set_fullscreen(on: Option<bool>) {
    set_window_mode(match on {
        Some(false) => 0,
        Some(true) => 1,
        None => MODE_TOGGLE,
    });
}

/// `WM_DS_MODE` arguments beyond the modes 0-2.
const MODE_TOGGLE: u32 = 3;
const MODE_REAPPLY: u32 = 4;

/// Asks the window for a mode (0 windowed, 1 borderless, 2 exclusive fullscreen); from any
/// thread.
pub fn set_window_mode(mode: u32) {
    if let Some(w) = WINDOW.get() {
        unsafe { PostMessageW(w.hwnd as HWND, WM_DS_MODE, mode as WPARAM, 0) };
    }
}

/// Tests (`window background|foreground`): the activation message Alt+Tab sends.
pub fn simulate_activation(active: bool) {
    if let Some(w) = WINDOW.get() {
        unsafe { PostMessageW(w.hwnd as HWND, WM_ACTIVATEAPP, active as WPARAM, 0) };
    }
}

/// Tests: the message a monitor's mode change sends (with the primary screen's size now).
pub fn simulate_display_change() {
    if let Some(w) = get() {
        let (cx, cy) = unsafe { (GetSystemMetrics(SM_CXSCREEN), GetSystemMetrics(SM_CYSCREEN)) };
        unsafe { PostMessageW(w.hwnd as HWND, WM_DISPLAYCHANGE, 32, ((cy as isize) << 16 | (cx as isize & 0xFFFF)) as LPARAM) };
    }
}

/// The PC tab's Monitor, Resolution or Refresh Rate changed: the window mode is applied again.
pub fn reapply_mode() {
    set_window_mode(MODE_REAPPLY);
}

/// Whether the presenter should hold exclusive fullscreen now, and on which monitor.
pub fn exclusive_wanted() -> Option<isize> {
    (EXCLUSIVE.load(Ordering::Relaxed) && !SUSPENDED.load(Ordering::Relaxed)).then(|| HMON.load(Ordering::Relaxed))
}

/// The window mode in effect (0 windowed, 1 borderless, 2 exclusive fullscreen).
pub fn window_mode() -> u32 {
    WINDOW_MODE.load(Ordering::Relaxed)
}

/// The monitor the window is on (HMONITOR) and its GDI device name.
pub fn current_monitor() -> Option<(isize, String)> {
    let w = WINDOW.get()?;
    unsafe {
        let h = MonitorFromWindow(w.hwnd as HWND, MONITOR_DEFAULTTONEAREST);
        let mut mi: windows_sys::Win32::Graphics::Gdi::MONITORINFOEXW = std::mem::zeroed();
        mi.monitorInfo.cbSize = std::mem::size_of::<windows_sys::Win32::Graphics::Gdi::MONITORINFOEXW>() as u32;
        if GetMonitorInfoW(h, &mut mi as *mut _ as *mut MONITORINFO) == 0 {
            return None;
        }
        let n = mi.szDevice.iter().position(|&c| c == 0).unwrap_or(mi.szDevice.len());
        Some((h as isize, String::from_utf16_lossy(&mi.szDevice[..n])))
    }
}

/// Asks the windowed window for a client size in pixels (ignored in
/// fullscreen); from any thread.
pub fn set_client_size(w: u32, h: u32) {
    if let Some(win) = WINDOW.get() {
        unsafe { PostMessageW(win.hwnd as HWND, WM_DS_SIZE, w as WPARAM, h as LPARAM) };
    }
}

/// UI thread: borderless fullscreen on the window's monitor, or back to the
/// windowed placement it had.
unsafe fn apply_fullscreen(hwnd: HWND, on: bool) {
    unsafe { apply_mode(hwnd, if on { LAST_FULL.load(Ordering::Relaxed).clamp(1, 2) } else { 0 }, false) };
}

/// The area a fullscreen window covers: the PC tab's Monitor (after any mode change), else
/// the monitor the window is on.
unsafe fn fullscreen_rect(hwnd: HWND) -> Option<RECT> {
    if let Some(t) = crate::display::target() {
        return Some(RECT { left: t.rect[0], top: t.rect[1], right: t.rect[2], bottom: t.rect[3] });
    }
    let mut mi: MONITORINFO = unsafe { std::mem::zeroed() };
    mi.cbSize = std::mem::size_of::<MONITORINFO>() as u32;
    (unsafe { GetMonitorInfoW(MonitorFromWindow(hwnd, MONITOR_DEFAULTTONEAREST), &mut mi) } != 0).then_some(mi.rcMonitor)
}

/// UI thread: the window centred on the PC tab's Monitor (windowed).
unsafe fn center_on_target(hwnd: HWND) {
    let Some(t) = crate::display::target() else { return };
    unsafe {
        let mut wr: RECT = std::mem::zeroed();
        GetWindowRect(hwnd, &mut wr);
        let (w, h) = (wr.right - wr.left, wr.bottom - wr.top);
        let inside = wr.left >= t.rect[0] && wr.top >= t.rect[1] && wr.right <= t.rect[2] && wr.bottom <= t.rect[3];
        if !inside {
            let x = t.rect[0] + ((t.rect[2] - t.rect[0] - w) / 2).max(0);
            let y = t.rect[1] + ((t.rect[3] - t.rect[1] - h) / 2).max(0);
            SetWindowPos(hwnd, std::ptr::null_mut(), x, y, 0, 0, SWP_NOSIZE | SWP_NOZORDER | SWP_NOACTIVATE);
        }
    }
}

/// UI thread: puts the window in `mode` (0 windowed, 1 borderless fullscreen, 2 exclusive
/// fullscreen) on the PC tab's Monitor; `force` applies it again when it is already in effect
/// (the monitor, resolution or refresh rate changed).
unsafe fn apply_mode(hwnd: HWND, mode: u32, force: bool) {
    let was = WINDOW_MODE.load(Ordering::Relaxed);
    if mode == was && !force {
        return;
    }
    unsafe {
        // Exclusive fullscreen's display mode: kept when it is the one wanted, else given back.
        let target = crate::display::target();
        let wanted = (mode == 2).then(|| target.as_ref().and_then(|t| crate::display::wanted_mode(&t.device))).flatten();
        let keep = mode == 2 && wanted.is_some() && target.as_ref().is_some_and(|t| crate::display::current_mode(&t.device) == wanted) && crate::display::mode_changed();
        if !keep {
            crate::display::restore();
        }
        EXCLUSIVE.store(false, Ordering::Relaxed);
        SUSPENDED.store(false, Ordering::Relaxed);
        if mode == 0 {
            if was != 0 {
                FULLSCREEN.store(false, Ordering::Relaxed);
                SetWindowLongPtrW(hwnd, GWL_STYLE, (WS_OVERLAPPEDWINDOW | WS_VISIBLE) as isize);
                let saved = PLACEMENT.lock().take();
                if let Some(mut wp) = saved {
                    if BEHIND.load(Ordering::Relaxed) && wp.showCmd == SW_SHOWNORMAL as u32 {
                        wp.showCmd = SW_SHOWNOACTIVATE as u32;
                    }
                    SetWindowPlacement(hwnd, &wp);
                }
                SetWindowPos(hwnd, std::ptr::null_mut(), 0, 0, 0, 0, SWP_NOMOVE | SWP_NOSIZE | SWP_NOZORDER | SWP_NOOWNERZORDER | SWP_FRAMECHANGED | SWP_NOACTIVATE);
            }
            center_on_target(hwnd);
            WINDOW_MODE.store(0, Ordering::Relaxed);
            RESIZED.store(true, Ordering::SeqCst);
            crate::info!("window: windowed");
            return;
        }
        if was == 0 {
            let mut wp: WINDOWPLACEMENT = std::mem::zeroed();
            wp.length = std::mem::size_of::<WINDOWPLACEMENT>() as u32;
            if GetWindowPlacement(hwnd, &mut wp) != 0 {
                *PLACEMENT.lock() = Some(wp);
            }
        }
        if mode == 2 {
            if let (Some(t), Some(m)) = (&target, wanted) {
                if !keep {
                    crate::display::set_mode(&t.device, m);
                }
            }
        }
        let Some(r) = fullscreen_rect(hwnd) else {
            crate::warn!("window: no fullscreen ({})", std::io::Error::last_os_error());
            return;
        };
        FULLSCREEN.store(true, Ordering::Relaxed);
        SetWindowLongPtrW(hwnd, GWL_STYLE, (WS_POPUP | WS_VISIBLE) as isize);
        let behind = BEHIND.load(Ordering::Relaxed);
        let (after, quiet) = if behind { (HWND_BOTTOM, SWP_NOACTIVATE) } else { (HWND_TOP, 0) };
        SetWindowPos(hwnd, after, r.left, r.top, r.right - r.left, r.bottom - r.top, SWP_NOOWNERZORDER | SWP_FRAMECHANGED | quiet);
        HMON.store(MonitorFromWindow(hwnd, MONITOR_DEFAULTTONEAREST) as isize, Ordering::Relaxed);
        WINDOW_MODE.store(mode, Ordering::Relaxed);
        LAST_FULL.store(mode, Ordering::Relaxed);
        EXCLUSIVE.store(mode == 2, Ordering::Relaxed);
        RESIZED.store(true, Ordering::SeqCst);
        crate::info!("window: {} {}x{}", if mode == 2 { "exclusive fullscreen" } else { "borderless fullscreen" }, r.right - r.left, r.bottom - r.top);
    }
}

/// UI thread: a fullscreen window fitted to its monitor again (the desktop's layout changed).
unsafe fn refit(hwnd: HWND) {
    if !FULLSCREEN.load(Ordering::Relaxed) {
        return;
    }
    unsafe {
        let mut mi: MONITORINFO = std::mem::zeroed();
        mi.cbSize = std::mem::size_of::<MONITORINFO>() as u32;
        if GetMonitorInfoW(MonitorFromWindow(hwnd, MONITOR_DEFAULTTONEAREST), &mut mi) != 0 {
            let r = mi.rcMonitor;
            SetWindowPos(hwnd, std::ptr::null_mut(), r.left, r.top, r.right - r.left, r.bottom - r.top, SWP_NOZORDER | SWP_NOACTIVATE | SWP_NOOWNERZORDER);
            HMON.store(MonitorFromWindow(hwnd, MONITOR_DEFAULTTONEAREST) as isize, Ordering::Relaxed);
        }
    }
    RESIZED.store(true, Ordering::SeqCst);
}

/// WM_INPUT: mouse motion, buttons and wheel while the cursor is captured.
unsafe fn raw_input(lp: LPARAM) {
    use windows_sys::Win32::UI::Input::*;
    let mut raw: RAWINPUT = unsafe { std::mem::zeroed() };
    let mut size = std::mem::size_of::<RAWINPUT>() as u32;
    let n = unsafe { GetRawInputData(lp as HRAWINPUT, RID_INPUT, &mut raw as *mut RAWINPUT as *mut _, &mut size, std::mem::size_of::<RAWINPUTHEADER>() as u32) };
    if n == 0 || n == u32::MAX || raw.header.dwType != RIM_TYPEMOUSE || !CAPTURED.load(Ordering::Relaxed) {
        return;
    }
    let m = unsafe { raw.data.mouse };
    // Keyboard and mouse in use (the button prompts): motion past a few counts, a button, the wheel.
    if m.lLastX.abs() + m.lLastY.abs() > 3 && m.usFlags & MOUSE_MOVE_ABSOLUTE == 0 || unsafe { m.Anonymous.Anonymous.usButtonFlags } != 0 {
        crate::input::note_device(crate::input::DEVICE_KBM);
    }
    if m.usFlags & MOUSE_MOVE_ABSOLUTE == 0 {
        MOUSE_DX.fetch_add(m.lLastX as i64, Ordering::Relaxed);
        MOUSE_DY.fetch_add(m.lLastY as i64, Ordering::Relaxed);
    } else {
        // 0..65535 over the (virtual) desktop: motion in pixels.
        let virt = m.usFlags & MOUSE_VIRTUAL_DESKTOP != 0;
        let (w, h) = unsafe {
            if virt {
                (GetSystemMetrics(SM_CXVIRTUALSCREEN), GetSystemMetrics(SM_CYVIRTUALSCREEN))
            } else {
                (GetSystemMetrics(SM_CXSCREEN), GetSystemMetrics(SM_CYSCREEN))
            }
        };
        let x = m.lLastX as i64 * w as i64 / 65535;
        let y = m.lLastY as i64 * h as i64 / 65535;
        let (px, py) = (ABS_X.swap(x, Ordering::Relaxed), ABS_Y.swap(y, Ordering::Relaxed));
        if px != i64::MIN {
            MOUSE_DX.fetch_add(x - px, Ordering::Relaxed);
            MOUSE_DY.fetch_add(y - py, Ordering::Relaxed);
        }
    }
    let f = unsafe { m.Anonymous.Anonymous.usButtonFlags } as u32;
    for (down, up, bit) in [(0x1, 0x2, 0), (0x4, 0x8, 1), (0x10, 0x20, 2), (0x40, 0x80, 3), (0x100, 0x200, 4)] {
        if f & down != 0 {
            MOUSE_BUTTONS.fetch_or(1 << bit, Ordering::Relaxed);
        }
        if f & up != 0 {
            MOUSE_BUTTONS.fetch_and(!(1 << bit), Ordering::Relaxed);
            MOUSE_SWALLOW.fetch_and(!(1 << bit), Ordering::Relaxed);
        }
    }
    if f & 0x400 != 0 {
        MOUSE_WHEEL.fetch_add(unsafe { m.Anonymous.Anonymous.usButtonData } as i16 as i32, Ordering::Relaxed);
    }
}

/// The key that closed the system text box (`ime_dialog`), kept from the game until released.
static IME_SWALLOW: std::sync::atomic::AtomicU32 = std::sync::atomic::AtomicU32::new(0);

unsafe extern "system" fn wndproc(hwnd: HWND, msg: u32, wp: WPARAM, lp: LPARAM) -> LRESULT {
    // Reflex: the flash indicator's left click (`gpu::latency`; the driver shows it where the
    // player turned it on).
    if msg == WM_LBUTTONDOWN {
        crate::gpu::latency::trigger_flash();
    }
    match msg {
        WM_CLOSE => {
            CLOSED.store(true, Ordering::SeqCst);
            0
        }
        WM_SIZE => {
            let w = (lp & 0xFFFF) as u32;
            let h = ((lp >> 16) & 0xFFFF) as u32;
            if w > 0 && h > 0 {
                CLIENT_W.store(w, Ordering::Relaxed);
                CLIENT_H.store(h, Ordering::Relaxed);
                RESIZED.store(true, Ordering::SeqCst);
            }
            if wp as u32 == SIZE_MINIMIZED {
                FOCUSED.store(false, Ordering::Relaxed);
            }
            unsafe { update_capture(hwnd) };
            0
        }
        WM_MOVE => {
            unsafe { update_capture(hwnd) };
            0
        }
        WM_ACTIVATE => {
            // Active and not minimized.
            FOCUSED.store((wp & 0xFFFF) as u32 != WA_INACTIVE && (wp >> 16) & 0xFFFF == 0, Ordering::Relaxed);
            // (The click that brings the window back is no choice in the save list.)
            if (wp & 0xFFFF) as u32 != WA_INACTIVE {
                crate::hle::libs::savedata::note_activation();
            }
            unsafe { update_capture(hwnd) };
            unsafe { DefWindowProcW(hwnd, msg, wp, lp) }
        }
        WM_SETCURSOR if CAPTURED.load(Ordering::Relaxed) && (lp & 0xFFFF) as u32 == HTCLIENT => {
            unsafe { SetCursor(std::ptr::null_mut()) };
            1
        }
        WM_SETCURSOR if pointer_on() && (lp & 0xFFFF) as u32 == HTCLIENT => {
            unsafe { SetCursor(pointer_cursor(hwnd)) };
            1
        }
        WM_DS_POINTER => {
            unsafe { update_capture(hwnd) };
            if pointer_on() {
                unsafe { SetCursor(pointer_cursor(hwnd)) };
            }
            0
        }
        WM_MOUSEMOVE => {
            use crate::pcsettings::pointer::{CLIENT_X, CLIENT_Y, MOVES};
            let (x, y) = ((lp & 0xFFFF) as i16 as i32, ((lp >> 16) & 0xFFFF) as i16 as i32);
            if !CAPTURED.load(Ordering::Relaxed) && (CLIENT_X.swap(x, Ordering::Relaxed) != x || CLIENT_Y.swap(y, Ordering::Relaxed) != y) {
                MOVES.fetch_add(1, Ordering::Relaxed);
                crate::input::note_device(crate::input::DEVICE_KBM);
            }
            0
        }
        // The save list (Load Game's characters, `savedata`): the wheel chooses, a left click
        // loads, a right click goes back.
        WM_LBUTTONDOWN | WM_RBUTTONDOWN if crate::hle::libs::savedata::list_active() => {
            crate::input::note_device(crate::input::DEVICE_KBM);
            // (The button is the list's until it is released: the game does not see it press
            // after the list closes, `MOUSE_SWALLOW`.)
            MOUSE_SWALLOW.fetch_or(if msg == WM_LBUTTONDOWN { 1 } else { 2 }, Ordering::Relaxed);
            crate::hle::libs::savedata::list_click(msg == WM_LBUTTONDOWN);
            0
        }
        WM_MOUSEWHEEL if crate::hle::libs::savedata::list_active() => {
            crate::input::note_device(crate::input::DEVICE_KBM);
            crate::hle::libs::savedata::list_wheel(((wp >> 16) & 0xFFFF) as i16 as i32);
            0
        }
        WM_LBUTTONDOWN | WM_RBUTTONDOWN | WM_MBUTTONDOWN | WM_XBUTTONDOWN
            if crate::pcsettings::bindings::mouse_down(
                Some(match msg {
                    WM_LBUTTONDOWN => 0,
                    WM_RBUTTONDOWN => 1,
                    WM_MBUTTONDOWN => 2,
                    _ => {
                        if (wp >> 16) & 0xFFFF == 1 {
                            3
                        } else {
                            4
                        }
                    }
                }),
                0,
            ) =>
        {
            0
        }
        WM_MOUSEWHEEL if crate::pcsettings::bindings::mouse_down(None, ((wp >> 16) & 0xFFFF) as i16 as i32) => 0,
        WM_LBUTTONDOWN | WM_RBUTTONDOWN if pointer_on() => {
            use crate::pcsettings::pointer::{Event, event};
            crate::input::note_device(crate::input::DEVICE_KBM);
            event(if msg == WM_LBUTTONDOWN { Event::Left } else { Event::Right });
            0
        }
        WM_MOUSEWHEEL if pointer_on() => {
            use crate::pcsettings::pointer::{Event, event};
            let notches = ((wp >> 16) & 0xFFFF) as i16 as i32 / 120;
            if notches != 0 {
                event(Event::Wheel(notches));
            }
            0
        }
        WM_LBUTTONDOWN | WM_RBUTTONDOWN | WM_MBUTTONDOWN if !CAPTURED.load(Ordering::Relaxed) => {
            // A click into the client area takes the cursor again.
            CAPTURE_WANTED.store(true, Ordering::Relaxed);
            FOCUSED.store(true, Ordering::Relaxed);
            unsafe { update_capture(hwnd) };
            0
        }
        WM_INPUT => {
            unsafe { raw_input(lp) };
            unsafe { DefWindowProcW(hwnd, msg, wp, lp) }
        }
        WM_KEYDOWN | WM_SYSKEYDOWN => {
            if wp as u16 == VK_F4_CODE && msg == WM_SYSKEYDOWN {
                // Alt+F4 closes the window.
                return unsafe { DefWindowProcW(hwnd, msg, wp, lp) };
            }
            // The system text box (the character's name, `ime_dialog`) takes the keys while it
            // is open; the key that closed it is kept from the game until it is released.
            let repeat = (lp >> 30) & 1 != 0;
            if msg == WM_KEYDOWN {
                if repeat && IME_SWALLOW.load(Ordering::Relaxed) == wp as u32 {
                    return 0;
                }
                let ctrl = unsafe { windows_sys::Win32::UI::Input::KeyboardAndMouse::GetKeyState(0x11) } < 0;
                if crate::hle::libs::ime_dialog::key_down(wp as u8, repeat, ctrl, ((lp >> 16) & 0x1FF) as u16) {
                    crate::input::note_device(crate::input::DEVICE_KBM);
                    if !crate::hle::libs::ime_dialog::active() {
                        IME_SWALLOW.store(wp as u32, Ordering::Relaxed);
                    }
                    return 0;
                }
                // The save list (Load Game's characters, `savedata`) likewise.
                if crate::hle::libs::savedata::list_key_down(wp as u8, repeat) {
                    crate::input::note_device(crate::input::DEVICE_KBM);
                    if !crate::hle::libs::savedata::list_active() {
                        IME_SWALLOW.store(wp as u32, Ordering::Relaxed);
                    }
                    return 0;
                }
            }
            if (wp as u16 == VK_RETURN_CODE && msg == WM_SYSKEYDOWN) || wp as u16 == VK_F11_CODE {
                // Alt+Enter, F11: windowed / the last fullscreen mode.
                if (lp >> 30) & 1 == 0 {
                    let mode = if WINDOW_MODE.load(Ordering::Relaxed) != 0 { 0 } else { LAST_FULL.load(Ordering::Relaxed).clamp(1, 2) };
                    unsafe { apply_mode(hwnd, mode, false) };
                    // The PC tab's Window Mode follows (its menu shows the mode in use).
                    let _ = crate::pcsettings::store::set_by_name("Setting_PC_WindowMode", &mode.to_string());
                }
                return 0;
            }
            if wp as u16 == VK_F10_CODE {
                // F10: the next 60 frames saved (a glitch as it happens); Ctrl+F10: every
                // thread's wait and stack written to the log (a hang the game still draws
                // through). (As a system key it would open the window's menu.)
                if (lp >> 30) & 1 == 0 {
                    if unsafe { windows_sys::Win32::UI::Input::KeyboardAndMouse::GetKeyState(0x11) } < 0 {
                        crate::stacks_hotkey();
                    } else {
                        crate::hle::libs::videoout::burst_hotkey();
                    }
                }
                return 0;
            }
            if wp as u16 == VK_F1_CODE && (lp >> 30) & 1 == 0 {
                CAPTURE_WANTED.fetch_xor(true, Ordering::Relaxed);
                unsafe { update_capture(hwnd) };
            }
            // A key binding waiting for its key takes it (not repeats).
            if (lp >> 30) & 1 == 0 && crate::pcsettings::bindings::key_down(wp as u8, sided(wp, lp)) {
                if matches!(wp as u8, 0x10 | 0x11 | 0x12) {
                    set_key(wp, lp, true);
                }
                return 0;
            }
            set_key(wp, lp, true);
            crate::input::note_device(crate::input::DEVICE_KBM);
            0
        }
        WM_KEYUP | WM_SYSKEYUP => {
            if IME_SWALLOW.load(Ordering::Relaxed) == wp as u32 {
                IME_SWALLOW.store(0, Ordering::Relaxed);
            }
            set_key(wp, lp, false);
            crate::pcsettings::bindings::key_up(wp as u8, sided(wp, lp));
            0
        }
        // Typed characters go to the system text box while it is open.
        WM_CHAR if crate::hle::libs::ime_dialog::char_typed(wp as u16, (lp >> 30) & 1 != 0, ((lp >> 16) & 0x1FF) as u16) => 0,
        // Alt with a key (Alt is walk): no menu mnemonic and its beep; Alt+Space
        // still opens the window menu.
        WM_SYSCHAR if wp != b' ' as WPARAM => 0,
        WM_DS_MODE => {
            match wp as u32 {
                m @ 0..=2 => unsafe { apply_mode(hwnd, m, false) },
                MODE_TOGGLE => unsafe { apply_fullscreen(hwnd, !FULLSCREEN.load(Ordering::Relaxed)) },
                _ => unsafe { apply_mode(hwnd, WINDOW_MODE.load(Ordering::Relaxed), true) },
            }
            0
        }
        WM_ACTIVATEAPP => {
            // Exclusive fullscreen in the background: the desktop's mode back and the window
            // minimized; in front again: the mode set once more.
            if WINDOW_MODE.load(Ordering::Relaxed) == 2 {
                if wp == 0 {
                    if !SUSPENDED.swap(true, Ordering::Relaxed) {
                        crate::display::restore();
                        RESIZED.store(true, Ordering::SeqCst);
                        unsafe { ShowWindow(hwnd, SW_MINIMIZE) };
                        crate::info!("window: exclusive fullscreen left for the desktop");
                    }
                } else if SUSPENDED.load(Ordering::Relaxed) {
                    unsafe {
                        ShowWindow(hwnd, SW_RESTORE);
                        apply_mode(hwnd, 2, true);
                    }
                }
            }
            unsafe { DefWindowProcW(hwnd, msg, wp, lp) }
        }
        // No screen saver or monitor power-off while the game is in front (a controller alone
        // moves no mouse and presses no key Windows sees; the execution state set at creation
        // keeps the display on, this keeps the screen saver off too).
        // (SC_SCREENSAVE 0xF140, SC_MONITORPOWER 0xF170.)
        WM_SYSCOMMAND if matches!(wp & 0xFFF0, 0xF140 | 0xF170) && FOCUSED.load(Ordering::Relaxed) => 0,
        WM_DISPLAYCHANGE => {
            // A monitor's mode or the desktop's layout changed (the port's own exclusive mode,
            // or the player's): a fullscreen window covers its monitor again.
            crate::info!("window: display changed to {}x{}", lp & 0xFFFF, (lp >> 16) & 0xFFFF);
            unsafe { refit(hwnd) };
            unsafe { DefWindowProcW(hwnd, msg, wp, lp) }
        }
        WM_DS_SIZE if !FULLSCREEN.load(Ordering::Relaxed) => {
            let mut r = RECT { left: 0, top: 0, right: wp as i32, bottom: lp as i32 };
            unsafe {
                AdjustWindowRectExForDpi(&mut r, WS_OVERLAPPEDWINDOW, 0, 0, GetDpiForWindow(hwnd).max(96));
                SetWindowPos(hwnd, std::ptr::null_mut(), 0, 0, r.right - r.left, r.bottom - r.top, SWP_NOMOVE | SWP_NOZORDER | SWP_NOACTIVATE);
            }
            crate::info!("window: client size {wp}x{lp} asked");
            0
        }
        WM_DPICHANGED => {
            // Moved to a monitor of another DPI: the size Windows suggests
            // (fullscreen keeps the monitor's).
            if !FULLSCREEN.load(Ordering::Relaxed) {
                let r = unsafe { &*(lp as *const RECT) };
                unsafe { SetWindowPos(hwnd, std::ptr::null_mut(), r.left, r.top, r.right - r.left, r.bottom - r.top, SWP_NOZORDER | SWP_NOACTIVATE) };
            }
            0
        }
        WM_KILLFOCUS => {
            for k in KEYS.iter() {
                k.store(false, Ordering::Relaxed);
            }
            FOCUSED.store(false, Ordering::Relaxed);
            unsafe { update_capture(hwnd) };
            0
        }
        // Reflex's PC Latency ping (a message Streamline sends): the next frame takes it.
        m if crate::gpu::latency::is_ping(m) => {
            crate::gpu::latency::ping();
            0
        }
        _ => unsafe { DefWindowProcW(hwnd, msg, wp, lp) },
    }
}

const VK_F1_CODE: u16 = 0x70;
const VK_F4_CODE: u16 = 0x73;
const VK_F11_CODE: u16 = 0x7A;
const VK_F10_CODE: u16 = 0x79;
const VK_RETURN_CODE: u16 = 0x0D;

/// `DS_WINDOW_SIZE=<w>x<h>`: the client size asked for (pixels).
fn size_override() -> Option<(i32, i32)> {
    let v = std::env::var("DS_WINDOW_SIZE").ok()?;
    let size = v.trim().split_once(['x', 'X']).and_then(|(w, h)| Some((w.trim().parse::<i32>().ok()?, h.trim().parse::<i32>().ok()?)));
    match size {
        Some((w, h)) if (160..=16384).contains(&w) && (90..=16384).contains(&h) => Some((w, h)),
        _ => {
            crate::warn!("DS_WINDOW_SIZE={v}: not a size (WxH); the default is used");
            None
        }
    }
}

/// UI thread, before the window is shown: its client area at `width`x`height`
/// scaled by the monitor's DPI (or `DS_WINDOW_SIZE`), at most the work area
/// (aspect kept), centred in it.
unsafe fn place_initial(hwnd: HWND, style: WINDOW_STYLE, width: u32, height: u32) {
    unsafe {
        let dpi = GetDpiForWindow(hwnd).max(96);
        let mut mi: MONITORINFO = std::mem::zeroed();
        mi.cbSize = std::mem::size_of::<MONITORINFO>() as u32;
        let have_area = GetMonitorInfoW(MonitorFromWindow(hwnd, MONITOR_DEFAULTTONEAREST), &mut mi) != 0;
        let area = mi.rcWork;
        let frame = |w: i32, h: i32| {
            let mut r = RECT { left: 0, top: 0, right: w, bottom: h };
            AdjustWindowRectExForDpi(&mut r, style, 0, 0, dpi);
            (r.right - r.left, r.bottom - r.top)
        };
        let asked = size_override();
        let (mut w, mut h) = asked.unwrap_or(((width * dpi / 96) as i32, (height * dpi / 96) as i32));
        if have_area && asked.is_none() {
            // Shrink (aspect kept) until the framed window fits the work area.
            let (fw, fh) = frame(w, h);
            let (aw, ah) = (area.right - area.left, area.bottom - area.top);
            if fw > aw || fh > ah {
                let s = ((aw - (fw - w)) as f64 / w as f64).min((ah - (fh - h)) as f64 / h as f64);
                (w, h) = (((w as f64 * s) as i32).max(160), ((h as f64 * s) as i32).max(90));
            }
        }
        let (fw, fh) = frame(w, h);
        let (x, y) = if have_area { (area.left + ((area.right - area.left - fw) / 2).max(0), area.top + ((area.bottom - area.top - fh) / 2).max(0)) } else { (0, 0) };
        SetWindowPos(hwnd, std::ptr::null_mut(), x, y, fw, fh, SWP_NOZORDER | SWP_NOACTIVATE);
        let mut c: RECT = std::mem::zeroed();
        GetClientRect(hwnd, &mut c);
        CLIENT_W.store((c.right - c.left).max(1) as u32, Ordering::Relaxed);
        CLIENT_H.store((c.bottom - c.top).max(1) as u32, Ordering::Relaxed);
        crate::info!(
            "window: client {}x{} at DPI {dpi} (the process {})",
            c.right - c.left,
            c.bottom - c.top,
            if PROCESS_DPI_AWARE.load(Ordering::Relaxed) { "per-monitor DPI aware" } else { "not DPI aware: Windows scaling may shrink the picture" }
        );
    }
}

fn wide(s: &str) -> Vec<u16> {
    s.encode_utf16().chain(std::iter::once(0)).collect()
}

/// Creates the window on a dedicated thread and waits until it exists.
pub fn create(title: &str, width: u32, height: u32) -> Result<&'static Window, String> {
    if let Some(w) = WINDOW.get() {
        return Ok(w);
    }
    let (tx, rx) = std::sync::mpsc::channel::<Result<(isize, isize), String>>();
    let title = title.to_string();
    std::thread::Builder::new()
        .name("ui".into())
        .spawn(move || unsafe {
            crate::log::set_thread_name("ui");
            let hinst = GetModuleHandleW(std::ptr::null());
            let class = wide("dsport_window");
            let wc = WNDCLASSEXW {
                cbSize: std::mem::size_of::<WNDCLASSEXW>() as u32,
                style: CS_HREDRAW | CS_VREDRAW,
                lpfnWndProc: Some(wndproc),
                cbClsExtra: 0,
                cbWndExtra: 0,
                hInstance: hinst,
                // The program icon (build.rs); the title bar's small one is
                // picked from the same icon's sizes.
                hIcon: LoadIconW(hinst, 1 as _),
                hCursor: LoadCursorW(std::ptr::null_mut(), IDC_ARROW),
                hbrBackground: std::ptr::null_mut(),
                lpszMenuName: std::ptr::null(),
                lpszClassName: class.as_ptr(),
                hIconSm: std::ptr::null_mut(),
            };
            RegisterClassExW(&wc);
            // Per-monitor DPI awareness for this thread's windows only (the
            // text entry dialog keeps the system's scaling).
            SetThreadDpiAwarenessContext(DPI_AWARENESS_CONTEXT_PER_MONITOR_AWARE_V2);
            let style = WS_OVERLAPPEDWINDOW;
            let mut rect = windows_sys::Win32::Foundation::RECT { left: 0, top: 0, right: width as i32, bottom: height as i32 };
            AdjustWindowRect(&mut rect, style, 0);
            let t = wide(&title);
            let hwnd = CreateWindowExW(
                0,
                class.as_ptr(),
                t.as_ptr(),
                style,
                CW_USEDEFAULT,
                CW_USEDEFAULT,
                rect.right - rect.left,
                rect.bottom - rect.top,
                std::ptr::null_mut(),
                std::ptr::null_mut(),
                hinst,
                std::ptr::null(),
            );
            if hwnd.is_null() {
                let _ = tx.send(Err("CreateWindowExW failed".into()));
                return;
            }
            CLIENT_W.store(width, Ordering::Relaxed);
            CLIENT_H.store(height, Ordering::Relaxed);
            place_initial(hwnd, style, width, height);
            // The mouse as a Raw Input device of this window (motion
            // without pointer acceleration; only while it is in front).
            let dev = windows_sys::Win32::UI::Input::RAWINPUTDEVICE { usUsagePage: 0x01, usUsage: 0x02, dwFlags: 0, hwndTarget: hwnd };
            if windows_sys::Win32::UI::Input::RegisterRawInputDevices(&dev, 1, std::mem::size_of::<windows_sys::Win32::UI::Input::RAWINPUTDEVICE>() as u32) == 0 {
                crate::warn!("window: no raw mouse input ({}): no mouse look", std::io::Error::last_os_error());
            }
            if std::env::var("DS_WINDOW_BEHIND").is_ok_and(|v| v.trim() == "1") {
                BEHIND.store(true, Ordering::Relaxed);
                ShowWindow(hwnd, SW_SHOWNOACTIVATE);
                SetWindowPos(hwnd, HWND_BOTTOM, 0, 0, 0, 0, SWP_NOMOVE | SWP_NOSIZE | SWP_NOACTIVATE);
            } else {
                ShowWindow(hwnd, SW_SHOW);
            }
            // No screen saver or display sleep while the game runs (a controller alone moves
            // no mouse and presses no key Windows sees).
            windows_sys::Win32::System::Power::SetThreadExecutionState(
                windows_sys::Win32::System::Power::ES_CONTINUOUS | windows_sys::Win32::System::Power::ES_DISPLAY_REQUIRED | windows_sys::Win32::System::Power::ES_SYSTEM_REQUIRED,
            );
            // `DS_FULLSCREEN` (0, 1 or 2) over the PC tab's Window Mode; a windowed start on
            // the PC tab's Monitor.
            let mode = match std::env::var("DS_FULLSCREEN") {
                Ok(v) => v.trim().parse::<u32>().unwrap_or(0).min(2),
                Err(_) => crate::pcsettings::window_mode().unwrap_or(0).min(2),
            };
            if mode != 0 {
                apply_mode(hwnd, mode, false);
            } else {
                center_on_target(hwnd);
            }
            let _ = tx.send(Ok((hwnd as isize, hinst as isize)));
            let mut msg: MSG = std::mem::zeroed();
            while GetMessageW(&mut msg, std::ptr::null_mut(), 0, 0) > 0 {
                TranslateMessage(&msg);
                DispatchMessageW(&msg);
            }
        })
        .map_err(|e| format!("ui thread: {e}"))?;
    let (hwnd, hinstance) = rx.recv().map_err(|e| format!("ui thread died: {e}"))??;
    WINDOW.set(Window { hwnd, hinstance }).ok();
    Ok(WINDOW.get().unwrap())
}

/// A window that is never shown, with a message loop of its own (`present::headless_present`:
/// a swapchain needs a window). Its handle and the module's.
pub fn create_hidden() -> Result<(isize, isize), String> {
    let (tx, rx) = std::sync::mpsc::channel::<Result<(isize, isize), String>>();
    std::thread::Builder::new()
        .name("hidden-window".into())
        .spawn(move || unsafe {
            crate::log::set_thread_name("hidden-window");
            let hinst = GetModuleHandleW(std::ptr::null());
            let class = wide("dsport_hidden");
            let wc = WNDCLASSEXW {
                cbSize: std::mem::size_of::<WNDCLASSEXW>() as u32,
                style: 0,
                lpfnWndProc: Some(DefWindowProcW),
                cbClsExtra: 0,
                cbWndExtra: 0,
                hInstance: hinst,
                hIcon: std::ptr::null_mut(),
                hCursor: std::ptr::null_mut(),
                hbrBackground: std::ptr::null_mut(),
                lpszMenuName: std::ptr::null(),
                lpszClassName: class.as_ptr(),
                hIconSm: std::ptr::null_mut(),
            };
            RegisterClassExW(&wc);
            let hwnd = CreateWindowExW(WS_EX_NOACTIVATE | WS_EX_TOOLWINDOW, class.as_ptr(), class.as_ptr(), WS_POPUP, 0, 0, 64, 64, std::ptr::null_mut(), std::ptr::null_mut(), hinst, std::ptr::null());
            if hwnd.is_null() {
                let _ = tx.send(Err("CreateWindowExW failed".into()));
                return;
            }
            let _ = tx.send(Ok((hwnd as isize, hinst as isize)));
            let mut msg: MSG = std::mem::zeroed();
            while GetMessageW(&mut msg, std::ptr::null_mut(), 0, 0) > 0 {
                DispatchMessageW(&msg);
            }
        })
        .map_err(|e| format!("hidden window thread: {e}"))?;
    rx.recv().map_err(|e| format!("hidden window thread died: {e}"))?
}

#[cfg(test)]
mod tests {
    #[test]
    fn pointer_arrow_draws() {
        let px = super::pointer_pixels(48);
        let opaque = px.chunks_exact(4).filter(|p| p[3] > 200).count();
        let white = px.chunks_exact(4).filter(|p| p[3] > 200 && p[0] > 200).count();
        assert!(opaque > 150 && white > 60, "{opaque} {white}");
        if let Ok(dir) = std::env::var("DS_DUMP_DIR") {
            std::fs::write(std::path::Path::new(&dir).join("pointer_cursor.png"), crate::gpu::png::encode(48, 48, &px)).unwrap();
        }
    }
}
