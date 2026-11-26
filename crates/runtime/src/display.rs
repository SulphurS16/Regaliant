//! Monitors and display modes: the PC tab's Monitor, Resolution and Refresh Rate, exclusive
//! fullscreen's mode changes (`gpu::vk::window`, `gpu::vk::present`) and each monitor's HDR
//! state (Windows' advanced color).
//!
//! The choices are kept in display.txt beside pcsettings.txt: `monitor=<GDI device name>`
//! (empty: the primary monitor), `resolution=<w>x<h>` (empty: the desktop's) and
//! `refresh=<Hz>` (0: the desktop's rate, or the highest one at another resolution). The
//! menu's lists are made once, at the start, from the monitors then and the chosen one's modes.

use parking_lot::Mutex;
use std::path::PathBuf;
use std::sync::OnceLock;

pub use dsconfig::display::{Mode, Monitor, current_mode, modes, monitors};
use dsconfig::display::registry_mode;

#[cfg_attr(windows, path = "platform/windows/display.rs")]
#[cfg_attr(target_os = "linux", path = "platform/linux/display.rs")]
mod host;
pub use host::{set_mode, restore, restore_on_exit, hdr_now};

#[derive(Default)]
struct Choice {
    monitor: String,
    res: Option<(u32, u32)>,
    hz: u32,
}

static CHOICE: Mutex<Option<Choice>> = Mutex::new(None);
static PATH: Mutex<Option<PathBuf>> = Mutex::new(None);
/// The mode set for exclusive fullscreen and its device, to put back (`restore`).
static CHANGED: Mutex<Option<String>> = Mutex::new(None);

/// The menu's options, made at the start.
struct Lists {
    monitors: Vec<Monitor>,
    resolutions: Vec<(u32, u32)>,
    rates: Vec<u32>,
}

static LISTS: OnceLock<Lists> = OnceLock::new();

/// The chosen monitor (display.txt), else the primary one; None without any monitor.
pub fn target() -> Option<Monitor> {
    let list = monitors();
    let want = CHOICE.lock().as_ref().map(|c| c.monitor.clone()).unwrap_or_default();
    list.iter().find(|m| !want.is_empty() && m.device.eq_ignore_ascii_case(&want)).or_else(|| list.iter().find(|m| m.primary)).or(list.first()).cloned()
}

/// `DS_TEST_DESKTOP_HZ=<hz>` (tests of the pacing on a real screen, `tools/pace_real.sh`): the
/// game's monitor runs at this refresh rate, at its desktop resolution, for the whole run in
/// every window mode (set at the start, put back at the exit).
pub fn test_hz() -> Option<u32> {
    static HZ: OnceLock<Option<u32>> = OnceLock::new();
    *HZ.get_or_init(|| std::env::var("DS_TEST_DESKTOP_HZ").ok().and_then(|v| v.trim().parse().ok()).filter(|h| (24..=500).contains(h)))
}

/// Sets the test's refresh rate (`test_hz`) on the game's monitor, where it is one of its modes.
pub fn apply_test_mode() {
    let (Some(hz), Some(t)) = (test_hz(), target()) else { return };
    let Some(desk) = registry_mode(&t.device).or_else(|| current_mode(&t.device)) else { return };
    let m = Mode { w: desk.w, h: desk.h, hz };
    if !modes(&t.device).contains(&m) {
        crate::warn!("display: DS_TEST_DESKTOP_HZ: {}x{} at {hz} Hz is not a mode of {}", desk.w, desk.h, t.device);
        return;
    }
    crate::info!("display: DS_TEST_DESKTOP_HZ: {} at {hz} Hz for this run", t.device);
    set_mode(&t.device, m);
}

/// The mode exclusive fullscreen takes on `device`: the chosen resolution and rate matched to
/// its modes (the nearest rate at that resolution); None keeps the desktop's mode.
pub fn wanted_mode(device: &str) -> Option<Mode> {
    if let Some(hz) = test_hz() {
        let desk = registry_mode(device).or_else(|| current_mode(device))?;
        return Some(Mode { w: desk.w, h: desk.h, hz });
    }
    let (res, hz) = CHOICE.lock().as_ref().map(|c| (c.res, c.hz)).unwrap_or((None, 0));
    let desk = registry_mode(device).or_else(|| current_mode(device))?;
    let (w, h) = res.unwrap_or((desk.w, desk.h));
    let at: Vec<Mode> = modes(device).into_iter().filter(|m| m.w == w && m.h == h).collect();
    if at.is_empty() {
        crate::warn!("display: {w}x{h} is not a mode of {device}: the desktop's mode stays");
        return None;
    }
    let rate = if hz != 0 {
        at.iter().min_by_key(|m| (m.hz as i64 - hz as i64).abs()).map(|m| m.hz)
    } else if let Some(r) = paced_rate(&at).filter(|_| crate::vrr::possible_on(device) != Some(true)) {
        // (Automatic with the frame rate limited (60 or 30) on a fixed-refresh screen: a rate
        // that makes the limit a whole number of refreshes, which the vblank then follows
        // (`displayclock`). A variable-refresh screen keeps its own rate: it shows each frame
        // as it comes at any rate.)
        Some(r)
    } else if (w, h) == (desk.w, desk.h) {
        Some(desk.hz)
    } else {
        at.iter().map(|m| m.hz).max()
    }?;
    let m = Mode { w, h, hz: rate };
    (m != desk).then_some(m)
}

/// Automatic refresh with the frame rate limited (the PC tab's 60 or 30 FPS, not Uncapped): the
/// highest of the rates `at` that is a whole multiple of 60 Hz (59, 60, 119, 120, 239, 240:
/// within 2%), so the limit is a whole number of refreshes and the game runs at a true 60;
/// None when there is none (or uncapped: the highest rate serves). At a rate that is no
/// multiple (165 Hz) the limit's frames are three refreshes each: an even 55 a second
/// (`displayclock::refreshes_for`).
fn paced_rate(at: &[Mode]) -> Option<u32> {
    if crate::gpu::tune::uncapped() {
        return None;
    }
    at.iter().map(|m| m.hz).filter(|&r| r >= 59 && ((r as f64 / 60.0) - (r as f64 / 60.0).round()).abs() < 0.02).max()
}

/// Whether the port changed a monitor's mode (exclusive fullscreen holds it).
pub fn mode_changed() -> bool {
    CHANGED.lock().is_some()
}

fn save() {
    let Some(p) = PATH.lock().clone() else { return };
    let text = {
        let g = CHOICE.lock();
        let c = g.as_ref();
        let monitor = c.map(|c| c.monitor.clone()).unwrap_or_default();
        let res = c.and_then(|c| c.res).map(|(w, h)| format!("{w}x{h}")).unwrap_or_default();
        let hz = c.map_or(0, |c| c.hz);
        format!("# Display choices (the PC tab's Monitor, Resolution and Refresh Rate); written by the port\nmonitor={monitor}\nresolution={res}\nrefresh={hz}\n")
    };
    if let Err(e) = std::fs::write(&p, text) {
        crate::warn!("display: write {}: {e}", p.display());
    }
}

/// Reads display.txt (`path`) and makes the menu's lists.
pub fn init(path: PathBuf) {
    let mut c = Choice::default();
    if let Ok(text) = std::fs::read_to_string(&path) {
        for line in text.lines() {
            let line = line.trim();
            if line.starts_with('#') {
                continue;
            }
            let Some((k, v)) = line.split_once('=') else { continue };
            let v = v.trim();
            match k.trim() {
                "monitor" => c.monitor = v.to_string(),
                "resolution" => c.res = v.split_once('x').and_then(|(w, h)| Some((w.trim().parse().ok()?, h.trim().parse().ok()?))),
                "refresh" => c.hz = v.parse().unwrap_or(0),
                _ => {}
            }
        }
    }
    *CHOICE.lock() = Some(c);
    *PATH.lock() = Some(path);
    let mons = monitors();
    let t = target();
    let (mut resolutions, mut rates) = (Vec::new(), Vec::new());
    if let Some(t) = &t {
        for m in modes(&t.device) {
            if !resolutions.contains(&(m.w, m.h)) {
                resolutions.push((m.w, m.h));
            }
            if !rates.contains(&m.hz) {
                rates.push(m.hz);
            }
        }
        rates.sort_by(|a, b| b.cmp(a));
        let cur = current_mode(&t.device);
        crate::info!(
            "display: {} monitor(s); {} ({}) {}x{} at {} Hz, {} modes, HDR {}",
            mons.len(),
            t.name,
            t.device,
            cur.map_or(0, |m| m.w),
            cur.map_or(0, |m| m.h),
            cur.map_or(0, |m| m.hz),
            resolutions.len(),
            if t.hdr_on { format!("on (SDR white {:.0} nits)", t.sdr_white_nits) } else if t.hdr_supported { "off (supported)".into() } else { "not supported".into() }
        );
    }
    let _ = LISTS.set(Lists { monitors: mons, resolutions, rates });
}

fn lists() -> Option<&'static Lists> {
    LISTS.get()
}

fn short(s: &str, n: usize) -> String {
    if s.chars().count() > n { s.chars().take(n - 3).collect::<String>() + "..." } else { s.to_string() }
}

/// The Monitor list's options (the menu's), made at the start.
pub fn monitor_options() -> Vec<String> {
    let l = lists().map(|l| l.monitors.as_slice()).unwrap_or(&[]);
    if l.is_empty() {
        return vec!["Primary".into()];
    }
    // (The value column of the Graphics & Display list holds about 15 characters; one monitor
    // goes without its number.)
    if l.len() == 1 {
        return vec![short(&l[0].name, 15)];
    }
    l.iter().enumerate().map(|(i, m)| short(&format!("{}: {}", i + 1, m.name), 15)).collect()
}

/// The Resolution list's options: the desktop's, then the start monitor's resolutions.
pub fn resolution_options() -> Vec<String> {
    std::iter::once("Desktop".to_string()).chain(lists().map(|l| l.resolutions.as_slice()).unwrap_or(&[]).iter().map(|(w, h)| format!("{w} x {h}"))).collect()
}

/// The Refresh Rate list's options: the desktop's (or the highest), then the rates.
pub fn refresh_options() -> Vec<String> {
    std::iter::once("Automatic".to_string()).chain(lists().map(|l| l.rates.as_slice()).unwrap_or(&[]).iter().map(|hz| format!("{hz} Hz"))).collect()
}

/// The option the Monitor list shows (the chosen monitor's, else the primary's).
pub fn monitor_index() -> u32 {
    let Some(l) = lists() else { return 0 };
    let want = CHOICE.lock().as_ref().map(|c| c.monitor.clone()).unwrap_or_default();
    l.monitors.iter().position(|m| !want.is_empty() && m.device.eq_ignore_ascii_case(&want)).or_else(|| l.monitors.iter().position(|m| m.primary)).unwrap_or(0) as u32
}

pub fn resolution_index() -> u32 {
    let Some(l) = lists() else { return 0 };
    let res = CHOICE.lock().as_ref().and_then(|c| c.res);
    res.and_then(|r| l.resolutions.iter().position(|&x| x == r)).map_or(0, |i| i as u32 + 1)
}

pub fn refresh_index() -> u32 {
    let Some(l) = lists() else { return 0 };
    let hz = CHOICE.lock().as_ref().map_or(0, |c| c.hz);
    if hz == 0 {
        return 0;
    }
    l.rates.iter().position(|&x| x == hz).map_or(0, |i| i as u32 + 1)
}

/// The menu changed the Monitor (option `i`): kept, and the window moves.
pub fn set_monitor_index(i: u32) {
    let Some(m) = lists().and_then(|l| l.monitors.get(i as usize)) else { return };
    if let Some(c) = CHOICE.lock().as_mut() {
        c.monitor = if m.primary { String::new() } else { m.device.clone() };
    }
    save();
    crate::info!("display: monitor {} ({})", m.name, m.device);
    crate::gpu::vk::window::reapply_mode();
}

pub fn set_resolution_index(i: u32) {
    let res = if i == 0 { None } else { lists().and_then(|l| l.resolutions.get(i as usize - 1).copied()) };
    if let Some(c) = CHOICE.lock().as_mut() {
        c.res = res;
    }
    save();
    crate::gpu::vk::window::reapply_mode();
}

pub fn set_refresh_index(i: u32) {
    let hz = if i == 0 { 0 } else { lists().and_then(|l| l.rates.get(i as usize - 1).copied()).unwrap_or(0) };
    if let Some(c) = CHOICE.lock().as_mut() {
        c.hz = hz;
    }
    save();
    crate::gpu::vk::window::reapply_mode();
}

#[cfg(test)]
mod tests {
    use super::*;

    fn at(rates: &[u32]) -> Vec<Mode> {
        rates.iter().map(|&hz| Mode { w: 2560, h: 1440, hz }).collect()
    }

    #[test]
    fn paced_rate_is_the_highest_multiple_of_60() {
        // (The user's monitor: 165 Hz desktop, 120 the even one.)
        assert_eq!(paced_rate(&at(&[59, 60, 75, 100, 120, 144, 165])), Some(120));
        assert_eq!(paced_rate(&at(&[60, 75, 144])), Some(60));
        assert_eq!(paced_rate(&at(&[59, 119, 239, 240])), Some(240));
        assert_eq!(paced_rate(&at(&[75, 144, 165])), None);
    }
}
