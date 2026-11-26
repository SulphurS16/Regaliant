//! The Display page: the game's window mode, monitor, fullscreen resolution and refresh rate,
//! render scale and frame rate limit for the next start (the user's request, 2026-11-13). The
//! same settings as the game's SETTINGS > PC tab: kept in the files the game reads at its start
//! (pcsettings.txt and display.txt), written as soon as a choice changes. While the game runs it
//! owns those files: the page waits, and reads them again when the game ends.

use crate::ui::{self, Builder};
use crate::App;
use dsconfig::display::{self as disp, Choice};
use windows_sys::Win32::Foundation::{HWND, LPARAM, WPARAM};
use windows_sys::Win32::UI::WindowsAndMessaging::*;

const ID_MODE: i32 = 401;
const ID_MONITOR: i32 = 402;
const ID_RES: i32 = 403;
const ID_RATE: i32 = 404;
const ID_SCALE: i32 = 405;
const ID_LIMIT: i32 = 406;

/// The page's index among the tabs.
pub const PAGE: usize = 1;

const MODES: [&str; 3] = ["Windowed", "Borderless fullscreen", "Fullscreen"];
/// The game's Render Scale options (the PC tab's: option 0 the Visual Mode's own size).
const SCALES: [&str; 10] = ["Visual Mode (the game's own)", "50%", "60%", "67%", "75%", "85%", "100%", "125%", "150%", "200%"];
const LIMITS: [&str; 3] = ["30 FPS", "60 FPS (the original)", "Uncapped"];

#[derive(Default)]
pub struct Page {
    mode: usize,
    monitor: usize,
    res: usize,
    rate: usize,
    scale: usize,
    limit: usize,
    note: usize,
    /// The monitors listed (device names, the primary first) and the chosen one's resolutions
    /// and rates as listed.
    monitors: Vec<disp::Monitor>,
    resolutions: Vec<(u32, u32)>,
    rates: Vec<u32>,
}

fn h(v: usize) -> HWND {
    v as HWND
}

unsafe fn combo(b: &Builder, r: (i32, i32, i32, i32), id: i32) -> HWND {
    unsafe { b.add("COMBOBOX", "", WS_TABSTOP | WS_VSCROLL | CBS_DROPDOWNLIST as u32, 0, (r.0, r.1, r.2, 320), id) }
}

unsafe fn fill(c: HWND, items: &[String], sel: usize) {
    unsafe {
        SendMessageW(c, CB_RESETCONTENT, 0, 0);
        for s in items {
            let w = ui::wide(s);
            SendMessageW(c, CB_ADDSTRING, 0, w.as_ptr() as LPARAM);
        }
        SendMessageW(c, CB_SETCURSEL, sel.min(items.len().saturating_sub(1)) as WPARAM, 0);
    }
}

unsafe fn selected(c: HWND) -> usize {
    unsafe { SendMessageW(c, CB_GETCURSEL, 0, 0) }.max(0) as usize
}

/// A PC setting's value in pcsettings.txt (`Setting_PC_<id> = <n>`).
fn setting(app: &App, id: &str) -> Option<u32> {
    let key = format!("Setting_PC_{id}");
    dsconfig::paths::read_kv(&app.paths.settings_file()).into_iter().find(|(k, _)| *k == key).and_then(|(_, v)| v.parse().ok())
}

/// Sets PC settings in pcsettings.txt (its other lines kept as they are).
fn set_settings(app: &App, values: &[(&str, u32)]) -> std::io::Result<()> {
    let path = app.paths.settings_file();
    let text = std::fs::read_to_string(&path).unwrap_or_else(|_| "# PC settings (the game's SETTINGS screen, PC tab); written by the port\n".into());
    let mut lines: Vec<String> = text.lines().map(str::to_string).collect();
    for (id, v) in values {
        let key = format!("Setting_PC_{id}");
        let line = format!("{key} = {v}");
        match lines.iter().position(|l| l.split_once('=').is_some_and(|(k, _)| k.trim() == key)) {
            Some(i) => lines[i] = line,
            None => lines.push(line),
        }
    }
    if let Some(dir) = path.parent() {
        std::fs::create_dir_all(dir)?;
    }
    std::fs::write(&path, lines.join("\n") + "\n")
}

pub fn build(app: &mut App) {
    let b: Builder = app.builder();
    unsafe {
        let r = (16, 58, 748, 40);
        app.put(PAGE, b.label("How the game shows for the next start. The game's SETTINGS > PC tab changes the same settings while you play.", r, -1), r);
        let rows = [("Window mode:", ID_MODE), ("Monitor:", ID_MONITOR), ("Fullscreen resolution:", ID_RES), ("Fullscreen refresh rate:", ID_RATE), ("Render scale:", ID_SCALE), ("Frame rate limit:", ID_LIMIT)];
        let mut hs = Vec::new();
        for (i, (label, id)) in rows.iter().enumerate() {
            let y = 108 + 40 * i as i32;
            let r = (16, y + 4, 200, 22);
            app.put(PAGE, b.label(label, r, -1), r);
            let r = (220, y, 340, 26);
            hs.push(app.put(PAGE, combo(&b, r, *id), r) as usize);
        }
        (app.display.mode, app.display.monitor, app.display.res, app.display.rate, app.display.scale, app.display.limit) = (hs[0], hs[1], hs[2], hs[3], hs[4], hs[5]);
        let r = (16, 356, 748, 60);
        app.display.note = app.put(PAGE, b.label("", r, -1), r) as usize;
    }
    crate::dlss::build(app, PAGE);
    refresh(app);
}

/// Reads the files again (at the start and when the game ends) and shows their choices.
pub fn refresh(app: &mut App) {
    let choice = disp::read_choice(&app.paths.display_file());
    app.display.monitors = disp::monitors();
    let mon_i = app.display.monitors.iter().position(|m| !choice.monitor.is_empty() && m.device.eq_ignore_ascii_case(&choice.monitor)).or_else(|| app.display.monitors.iter().position(|m| m.primary)).unwrap_or(0);
    let mons: Vec<String> = if app.display.monitors.is_empty() {
        vec!["Primary".into()]
    } else {
        app.display.monitors.iter().enumerate().map(|(i, m)| format!("{}: {}{}", i + 1, m.name, if m.primary { " (primary)" } else { "" })).collect()
    };
    unsafe {
        fill(h(app.display.mode), &MODES.iter().map(|s| s.to_string()).collect::<Vec<_>>(), setting(app, "WindowMode").unwrap_or(0) as usize);
        fill(h(app.display.monitor), &mons, mon_i);
        fill(h(app.display.scale), &SCALES.iter().map(|s| s.to_string()).collect::<Vec<_>>(), setting(app, "RenderScale").unwrap_or(0) as usize);
        fill(h(app.display.limit), &LIMITS.iter().map(|s| s.to_string()).collect::<Vec<_>>(), setting(app, "FrameRateLimit").unwrap_or(1) as usize);
    }
    modes_for(app, mon_i, &choice);
    update(app);
    crate::dlss::refresh(app);
}

/// The chosen monitor's resolutions and rates, the choice selected.
fn modes_for(app: &mut App, mon_i: usize, choice: &Choice) {
    let (mut res, mut rates) = (Vec::new(), Vec::new());
    if let Some(m) = app.display.monitors.get(mon_i) {
        for md in disp::modes(&m.device) {
            if !res.contains(&(md.w, md.h)) {
                res.push((md.w, md.h));
            }
            if !rates.contains(&md.hz) {
                rates.push(md.hz);
            }
        }
    }
    rates.sort_by(|a, b| b.cmp(a));
    let res_i = choice.res.and_then(|r| res.iter().position(|&x| x == r)).map_or(0, |i| i + 1);
    let rate_i = if choice.hz == 0 { 0 } else { rates.iter().position(|&x| x == choice.hz).map_or(0, |i| i + 1) };
    let res_s: Vec<String> = std::iter::once("Desktop".to_string()).chain(res.iter().map(|(w, h)| format!("{w} x {h}"))).collect();
    let rate_s: Vec<String> = std::iter::once("Automatic".to_string()).chain(rates.iter().map(|hz| format!("{hz} Hz"))).collect();
    unsafe {
        fill(h(app.display.res), &res_s, res_i);
        fill(h(app.display.rate), &rate_s, rate_i);
    }
    app.display.resolutions = res;
    app.display.rates = rates;
}

/// The page while the game runs (it owns the files then) and the note on the mode.
pub fn update(app: &App) {
    let running = app.running.is_some();
    let fullscreen = unsafe { selected(h(app.display.mode)) } == 2;
    unsafe {
        for c in [app.display.mode, app.display.monitor, app.display.scale, app.display.limit] {
            ui::enable(h(c), !running);
        }
        for c in [app.display.res, app.display.rate] {
            ui::enable(h(c), !running && fullscreen);
        }
        let note = if running {
            "The game is running. Change these in its SETTINGS > PC tab, or here after it ends."
        } else if fullscreen {
            "Fullscreen sets the monitor to this resolution and refresh rate while the game is in front. Alt+Tab gives the desktop its own mode back."
        } else {
            "Borderless fullscreen and the window keep the desktop's resolution. Alt+Enter switches between the window and fullscreen while you play."
        };
        ui::set_text(h(app.display.note), note);
    }
}

fn save(app: &mut App) {
    let (mode, scale, limit) = unsafe { (selected(h(app.display.mode)), selected(h(app.display.scale)), selected(h(app.display.limit))) };
    let mon_i = unsafe { selected(h(app.display.monitor)) };
    let (res_i, rate_i) = unsafe { (selected(h(app.display.res)), selected(h(app.display.rate))) };
    let choice = Choice {
        monitor: app.display.monitors.get(mon_i).filter(|m| !m.primary).map(|m| m.device.clone()).unwrap_or_default(),
        res: res_i.checked_sub(1).and_then(|i| app.display.resolutions.get(i).copied()),
        hz: rate_i.checked_sub(1).and_then(|i| app.display.rates.get(i).copied()).unwrap_or(0),
    };
    let r = set_settings(app, &[("WindowMode", mode as u32), ("RenderScale", scale as u32), ("FrameRateLimit", limit as u32)]).and_then(|_| disp::write_choice(&app.paths.display_file(), &choice));
    match r {
        Ok(()) => app.set_status("Display settings saved for the next start."),
        Err(e) => app.set_status(&format!("Could not save the display settings: {e}")),
    }
    update(app);
}

pub fn command(app: &mut App, id: i32, code: u32) {
    if code != CBN_SELCHANGE {
        return;
    }
    if id == ID_MONITOR {
        // Another monitor: its own resolutions and rates (the desktop's and automatic first).
        let mon_i = unsafe { selected(h(app.display.monitor)) };
        modes_for(app, mon_i, &Choice::default());
    }
    if matches!(id, ID_MODE | ID_MONITOR | ID_RES | ID_RATE | ID_SCALE | ID_LIMIT) {
        save(app);
    }
}
