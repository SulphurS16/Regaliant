//! Windows desktop mode changes, unchanged behind the shared display settings.
use super::{Mode, current_mode, CHANGED, test_hz};
use windows_sys::Win32::Graphics::Gdi::*;
use dsconfig::display::{devmode, targets};

fn wide(s: &str) -> Vec<u16> {
    s.encode_utf16().chain(std::iter::once(0)).collect()
}

pub fn set_mode(device: &str, mode: Mode) -> bool {
    if current_mode(device) == Some(mode) {
        return true;
    }
    let dev = wide(device);
    let mut dm = devmode();
    dm.dmPelsWidth = mode.w;
    dm.dmPelsHeight = mode.h;
    dm.dmDisplayFrequency = mode.hz;
    dm.dmBitsPerPel = 32;
    dm.dmFields = DM_PELSWIDTH | DM_PELSHEIGHT | DM_DISPLAYFREQUENCY | DM_BITSPERPEL;
    let r = unsafe { ChangeDisplaySettingsExW(dev.as_ptr(), &dm, std::ptr::null_mut(), CDS_FULLSCREEN, std::ptr::null()) };
    if r == DISP_CHANGE_SUCCESSFUL {
        *CHANGED.lock() = Some(device.to_string());
        crate::info!("display: {device} set to {}x{} at {} Hz", mode.w, mode.h, mode.hz);
        true
    } else {
        crate::warn!("display: {device} refused {}x{} at {} Hz (code {r})", mode.w, mode.h, mode.hz);
        false
    }
}

pub fn restore() {
    // (A test's rate stays for the run: `restore_on_exit` puts the desktop's back.)
    if test_hz().is_some() {
        return;
    }
    let Some(device) = CHANGED.lock().take() else { return };
    let dev = wide(&device);
    let r = unsafe { ChangeDisplaySettingsExW(dev.as_ptr(), std::ptr::null(), std::ptr::null_mut(), 0, std::ptr::null()) };
    crate::info!("display: {device} back to its desktop mode{}", if r == DISP_CHANGE_SUCCESSFUL { String::new() } else { format!(" (code {r})") });
}

pub fn restore_on_exit() {
    let Some(mut g) = CHANGED.try_lock_for(std::time::Duration::from_millis(200)) else { return };
    let Some(device) = g.take() else { return };
    drop(g);
    let dev = wide(&device);
    unsafe { ChangeDisplaySettingsExW(dev.as_ptr(), std::ptr::null(), std::ptr::null_mut(), 0, std::ptr::null()) };
}

pub fn hdr_now(device: &str) -> Option<f32> {
    targets().into_iter().find(|t| t.0.eq_ignore_ascii_case(device)).filter(|t| t.3).map(|t| t.4)
}
