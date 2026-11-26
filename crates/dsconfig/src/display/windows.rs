use super::{Mode, Monitor};
use windows_sys::Win32::Devices::Display::*;
use windows_sys::Win32::Foundation::{LPARAM, RECT};
use windows_sys::Win32::Graphics::Gdi::*;

fn wide(s: &str) -> Vec<u16> {
    s.encode_utf16().chain(std::iter::once(0)).collect()
}

fn from_wide(w: &[u16]) -> String {
    let n = w.iter().position(|&c| c == 0).unwrap_or(w.len());
    String::from_utf16_lossy(&w[..n])
}

/// Per GDI device name: the monitor's own name and its HDR state (DisplayConfig).
pub fn targets() -> Vec<(String, String, bool, bool, f32)> {
    let mut out = Vec::new();
    unsafe {
        let (mut np, mut nm) = (0u32, 0u32);
        if GetDisplayConfigBufferSizes(QDC_ONLY_ACTIVE_PATHS, &mut np, &mut nm) != 0 {
            return out;
        }
        let mut paths: Vec<DISPLAYCONFIG_PATH_INFO> = vec![std::mem::zeroed(); np as usize];
        let mut modes: Vec<DISPLAYCONFIG_MODE_INFO> = vec![std::mem::zeroed(); nm as usize];
        if QueryDisplayConfig(QDC_ONLY_ACTIVE_PATHS, &mut np, paths.as_mut_ptr(), &mut nm, modes.as_mut_ptr(), std::ptr::null_mut()) != 0 {
            return out;
        }
        paths.truncate(np as usize);
        for p in &paths {
            let mut src: DISPLAYCONFIG_SOURCE_DEVICE_NAME = std::mem::zeroed();
            src.header.r#type = DISPLAYCONFIG_DEVICE_INFO_GET_SOURCE_NAME;
            src.header.size = std::mem::size_of::<DISPLAYCONFIG_SOURCE_DEVICE_NAME>() as u32;
            src.header.adapterId = p.sourceInfo.adapterId;
            src.header.id = p.sourceInfo.id;
            if DisplayConfigGetDeviceInfo(&mut src.header) != 0 {
                continue;
            }
            let device = from_wide(&src.viewGdiDeviceName);
            let mut tgt: DISPLAYCONFIG_TARGET_DEVICE_NAME = std::mem::zeroed();
            tgt.header.r#type = DISPLAYCONFIG_DEVICE_INFO_GET_TARGET_NAME;
            tgt.header.size = std::mem::size_of::<DISPLAYCONFIG_TARGET_DEVICE_NAME>() as u32;
            tgt.header.adapterId = p.targetInfo.adapterId;
            tgt.header.id = p.targetInfo.id;
            let name = if DisplayConfigGetDeviceInfo(&mut tgt.header) == 0 { from_wide(&tgt.monitorFriendlyDeviceName) } else { String::new() };
            let mut aci: DISPLAYCONFIG_GET_ADVANCED_COLOR_INFO = std::mem::zeroed();
            aci.header.r#type = DISPLAYCONFIG_DEVICE_INFO_GET_ADVANCED_COLOR_INFO;
            aci.header.size = std::mem::size_of::<DISPLAYCONFIG_GET_ADVANCED_COLOR_INFO>() as u32;
            aci.header.adapterId = p.targetInfo.adapterId;
            aci.header.id = p.targetInfo.id;
            let (supported, on) = if DisplayConfigGetDeviceInfo(&mut aci.header) == 0 {
                let v = aci.Anonymous.value;
                // Bit 0 advanced color supported, bit 1 enabled, bit 3 force-disabled.
                (v & 1 != 0, v & 2 != 0 && v & 8 == 0)
            } else {
                (false, false)
            };
            let mut white: DISPLAYCONFIG_SDR_WHITE_LEVEL = std::mem::zeroed();
            white.header.r#type = DISPLAYCONFIG_DEVICE_INFO_GET_SDR_WHITE_LEVEL;
            white.header.size = std::mem::size_of::<DISPLAYCONFIG_SDR_WHITE_LEVEL>() as u32;
            white.header.adapterId = p.targetInfo.adapterId;
            white.header.id = p.targetInfo.id;
            // In thousandths of 80 nits.
            let nits = if DisplayConfigGetDeviceInfo(&mut white.header) == 0 && white.SDRWhiteLevel > 0 { white.SDRWhiteLevel as f32 / 1000.0 * 80.0 } else { 80.0 };
            out.push((device, name, supported, on, nits));
        }
    }
    out
}

unsafe extern "system" fn enum_monitor(h: HMONITOR, _dc: HDC, _r: *mut RECT, data: LPARAM) -> windows_sys::core::BOOL {
    let list = unsafe { &mut *(data as *mut Vec<Monitor>) };
    let mut mi: MONITORINFOEXW = unsafe { std::mem::zeroed() };
    mi.monitorInfo.cbSize = std::mem::size_of::<MONITORINFOEXW>() as u32;
    if unsafe { GetMonitorInfoW(h, &mut mi as *mut MONITORINFOEXW as *mut MONITORINFO) } != 0 {
        let r = mi.monitorInfo.rcMonitor;
        let device = from_wide(&mi.szDevice);
        list.push(Monitor {
            name: device.clone(),
            device,
            rect: [r.left, r.top, r.right, r.bottom],
            primary: mi.monitorInfo.dwFlags & windows_sys::Win32::UI::WindowsAndMessaging::MONITORINFOF_PRIMARY != 0,
            sdr_white_nits: 80.0,
            ..Default::default()
        });
    }
    1
}

/// The monitors Windows lists now, the primary first, then left to right.
pub fn monitors() -> Vec<Monitor> {
    let mut list: Vec<Monitor> = Vec::new();
    unsafe { EnumDisplayMonitors(std::ptr::null_mut(), std::ptr::null(), Some(enum_monitor), &mut list as *mut Vec<Monitor> as LPARAM) };
    let t = targets();
    for m in list.iter_mut() {
        if let Some((_, name, sup, on, nits)) = t.iter().find(|x| x.0.eq_ignore_ascii_case(&m.device)) {
            if !name.is_empty() {
                m.name = name.clone();
            }
            (m.hdr_supported, m.hdr_on, m.sdr_white_nits) = (*sup, *on, *nits);
        }
    }
    list.sort_by_key(|m| (!m.primary, m.rect[0], m.rect[1]));
    list
}

pub fn devmode() -> DEVMODEW {
    let mut dm: DEVMODEW = unsafe { std::mem::zeroed() };
    dm.dmSize = std::mem::size_of::<DEVMODEW>() as u16;
    dm
}

/// A monitor's modes (32 bits per pixel, progressive), largest and fastest first.
pub fn modes(device: &str) -> Vec<Mode> {
    let dev = wide(device);
    let mut out = Vec::new();
    let mut i = 0u32;
    loop {
        let mut dm = devmode();
        if unsafe { EnumDisplaySettingsExW(dev.as_ptr(), i, &mut dm, 0) } == 0 {
            break;
        }
        i += 1;
        let interlaced = unsafe { dm.Anonymous2.dmDisplayFlags } & DM_INTERLACED != 0;
        if dm.dmBitsPerPel == 32 && !interlaced && dm.dmPelsWidth >= 640 && dm.dmPelsHeight >= 480 && dm.dmDisplayFrequency > 1 {
            out.push(Mode { w: dm.dmPelsWidth, h: dm.dmPelsHeight, hz: dm.dmDisplayFrequency });
        }
        if i > 4096 {
            break;
        }
    }
    out.sort_by(|a, b| (b.w * b.h, b.w, b.hz).cmp(&(a.w * a.h, a.w, a.hz)));
    out.dedup();
    out
}

/// A monitor's mode now.
pub fn current_mode(device: &str) -> Option<Mode> {
    let dev = wide(device);
    let mut dm = devmode();
    (unsafe { EnumDisplaySettingsExW(dev.as_ptr(), ENUM_CURRENT_SETTINGS, &mut dm, 0) } != 0).then_some(Mode { w: dm.dmPelsWidth, h: dm.dmPelsHeight, hz: dm.dmDisplayFrequency })
}

/// Its mode in the registry (what Windows puts back when the port's mode ends).
pub fn registry_mode(device: &str) -> Option<Mode> {
    let dev = wide(device);
    let mut dm = devmode();
    (unsafe { EnumDisplaySettingsExW(dev.as_ptr(), ENUM_REGISTRY_SETTINGS, &mut dm, 0) } != 0).then_some(Mode { w: dm.dmPelsWidth, h: dm.dmPelsHeight, hz: dm.dmDisplayFrequency })
}

/// The signal Windows sends `device`'s monitor now: its colour encoding (RGB, YCbCr 4:4:4,
/// 4:2:2 or 4:2:0) and bits per colour channel (DisplayConfig's advanced colour information).
pub fn signal(device: &str) -> Option<(&'static str, u32)> {
    unsafe {
        let (mut np, mut nm) = (0u32, 0u32);
        if GetDisplayConfigBufferSizes(QDC_ONLY_ACTIVE_PATHS, &mut np, &mut nm) != 0 {
            return None;
        }
        let mut paths: Vec<DISPLAYCONFIG_PATH_INFO> = vec![std::mem::zeroed(); np as usize];
        let mut modes: Vec<DISPLAYCONFIG_MODE_INFO> = vec![std::mem::zeroed(); nm as usize];
        if QueryDisplayConfig(QDC_ONLY_ACTIVE_PATHS, &mut np, paths.as_mut_ptr(), &mut nm, modes.as_mut_ptr(), std::ptr::null_mut()) != 0 {
            return None;
        }
        paths.truncate(np as usize);
        for p in &paths {
            let mut src: DISPLAYCONFIG_SOURCE_DEVICE_NAME = std::mem::zeroed();
            src.header.r#type = DISPLAYCONFIG_DEVICE_INFO_GET_SOURCE_NAME;
            src.header.size = std::mem::size_of::<DISPLAYCONFIG_SOURCE_DEVICE_NAME>() as u32;
            src.header.adapterId = p.sourceInfo.adapterId;
            src.header.id = p.sourceInfo.id;
            if DisplayConfigGetDeviceInfo(&mut src.header) != 0 || !from_wide(&src.viewGdiDeviceName).eq_ignore_ascii_case(device) {
                continue;
            }
            let mut aci: DISPLAYCONFIG_GET_ADVANCED_COLOR_INFO = std::mem::zeroed();
            aci.header.r#type = DISPLAYCONFIG_DEVICE_INFO_GET_ADVANCED_COLOR_INFO;
            aci.header.size = std::mem::size_of::<DISPLAYCONFIG_GET_ADVANCED_COLOR_INFO>() as u32;
            aci.header.adapterId = p.targetInfo.adapterId;
            aci.header.id = p.targetInfo.id;
            if DisplayConfigGetDeviceInfo(&mut aci.header) != 0 {
                return None;
            }
            let enc = match aci.colorEncoding {
                0 => "RGB",
                1 => "YCbCr 4:4:4",
                2 => "YCbCr 4:2:2",
                3 => "YCbCr 4:2:0",
                _ => "other",
            };
            return Some((enc, aci.bitsPerColorChannel));
        }
    }
    None
}

