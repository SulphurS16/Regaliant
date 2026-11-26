//! regaliant-launcher: sets up and starts Regaliant, the Demon's Souls PC port, for players.
//!
//! - Game: the game folder, checked when chosen (title, version 01.007.000, the decrypted
//!   executable and libraries, the data folders); what this PC brings; safe mode for one start.
//! - Shaders: the game's shaders prepared ahead, with progress (the first Play prepares them
//!   unless that is turned off).
//! - Saves: the save folder, the backups the game makes and their restore, the PS5 import folder.
//! - Play / Reset settings / Open logs / Join the Discord (the community's invite in the
//!   browser). When the game stops with an error, its last error lines and its log folder are
//!   shown.
//!
//! The settings are the files the game reads (`dsconfig::paths`); the game's own settings are
//! in its menus (SETTINGS > PC).

#![windows_subsystem = "windows"]

mod display;
mod dlss;
mod game;
mod saves;
mod shaders;
mod shot;
mod sys;
mod ui;
mod uninstall;
mod update;

use dsconfig::paths::{Launch, Paths};
use std::cell::RefCell;
use std::path::PathBuf;
use ui::{Builder, Scale, wide};
use windows_sys::Win32::Foundation::{HWND, LPARAM, LRESULT, RECT, WPARAM};
use windows_sys::Win32::Graphics::Gdi::{DeleteObject, HFONT};
use windows_sys::Win32::System::LibraryLoader::GetModuleHandleW;
use windows_sys::Win32::UI::Controls::*;
use windows_sys::Win32::UI::HiDpi::{AdjustWindowRectExForDpi, DPI_AWARENESS_CONTEXT_PER_MONITOR_AWARE_V2, GetDpiForWindow, SetProcessDpiAwarenessContext};
use windows_sys::Win32::UI::WindowsAndMessaging::*;

pub const TITLE: &str = "Regaliant";
/// The release version (the workspace's, this package's own).
pub const VERSION: &str = env!("CARGO_PKG_VERSION");
const CLIENT: (i32, i32) = (780, 600);

/// Posted when the game exits (WPARAM: its exit code).
const WM_GAME_EXIT: u32 = WM_APP + 2;
/// Retries a game exit that arrived while a dialog held the app.
const TIMER_RETRY: usize = 2;

const ID_TABS: i32 = 10;
const ID_PLAY: i32 = 901;
const ID_RESET: i32 = 903;
const ID_LOGS: i32 = 904;
const ID_DISCORD: i32 = 905;
const ID_STATUS_BAR: i32 = 906;
const ID_UNINSTALL: i32 = 907;

/// The community's Discord invite (the button "Join the Discord").
const DISCORD: &str = "https://discord.gg/rSr5JhxxtZ";

/// The game's exit codes when it told the player itself: its files cannot be used, the GPU
/// device was lost, the processor lacks what the game needs (and 0xD5xx, errors it caught).
const EXIT_GAME_FILES: i32 = 3;
const EXIT_DEVICE_LOST: i32 = 5;
const EXIT_CPU: i32 = 8;
const EXIT_RUNNING: i32 = 9;
const EXIT_EARLY: i32 = 10;
const EXIT_MEMORY: i32 = 7;
const EXIT_NO_GPU: i32 = 11;

pub struct App {
    pub hwnd: HWND,
    pub scale: Scale,
    pub font: HFONT,
    pub bold: HFONT,
    /// Every control with its 1/96-inch rectangle (laid out again when the DPI changes).
    pub layout: Vec<(HWND, (i32, i32, i32, i32))>,
    pub paths: Paths,
    pub launch: Launch,
    pub tabs: HWND,
    /// The controls of each page.
    pub pages: Vec<Vec<HWND>>,
    pub page: usize,
    pub status: HWND,
    pub play: HWND,
    pub game: game::Page,
    pub display: display::Page,
    pub shaders: shaders::Page,
    pub saves: saves::Page,
    /// NVIDIA's DLSS file's row on the Display page (`dlss`).
    pub dlss: dlss::Row,
    pub update: update::Page,
    /// The game's process id while it runs.
    pub running: Option<u32>,
}

thread_local! {
    static APP: RefCell<Option<App>> = const { RefCell::new(None) };
    /// Game exits that could not be handled yet (a dialog was open): their exit codes.
    static PENDING: RefCell<Vec<WPARAM>> = const { RefCell::new(Vec::new()) };
}

/// Runs `f` on the app unless it is already borrowed (a message inside a dialog's loop).
pub fn with_app<R>(f: impl FnOnce(&mut App) -> R) -> Option<R> {
    APP.with(|a| a.try_borrow_mut().ok().and_then(|mut g| g.as_mut().map(f)))
}

impl App {
    pub fn builder(&self) -> Builder {
        Builder { parent: self.hwnd, scale: self.scale, font: self.font }
    }

    /// Records a control of page `page` (usize::MAX: on every page).
    pub fn put(&mut self, page: usize, h: HWND, r: (i32, i32, i32, i32)) -> HWND {
        self.layout.push((h, r));
        if page < self.pages.len() {
            self.pages[page].push(h);
        }
        h
    }

    pub fn save_launch(&mut self) {
        if let Err(e) = self.launch.save(&self.paths) {
            self.set_status(&format!("Could not save {}: {e}", self.paths.launch_file().display()));
        }
    }

    pub fn set_status(&self, s: &str) {
        unsafe { ui::set_text(self.status, s) };
    }

    fn show_page(&mut self, n: usize) {
        self.page = n;
        for (i, p) in self.pages.iter().enumerate() {
            for &h in p {
                unsafe { ui::show(h, i == n) };
            }
        }
        // (The Shaders page's progress bar shows only while preparing: showing the page showed
        // it empty, hidden again only by the start-up check's refresh.)
        if n == 2 {
            shaders::refresh(self);
        }
        unsafe { ui::redraw(self.hwnd) };
    }

    pub fn update_play(&self) {
        // (Not while the game folder is being compressed: the game would read it meanwhile.)
        unsafe { ui::enable(self.play, self.game.ok && self.running.is_none() && !game::compressing(self)) };
        game::update(self);
        saves::update(self);
        display::update(self);
    }

    fn relayout(&mut self, dpi: u32) {
        self.scale = Scale(dpi);
        unsafe {
            DeleteObject(self.font as _);
            DeleteObject(self.bold as _);
            self.font = ui::message_font(dpi, false, 100);
            self.bold = ui::message_font(dpi, true, 110);
            for &(h, r) in &self.layout {
                let s = self.scale;
                SetWindowPos(h, std::ptr::null_mut(), s.px(r.0), s.px(r.1), s.px(r.2), s.px(r.3), SWP_NOZORDER | SWP_NOACTIVATE);
                SendMessageW(h, WM_SETFONT, if h == self.play { self.bold } else { self.font } as WPARAM, 1);
            }
        }
    }
}

/// Play: the shaders first when they are not prepared (unless turned off), then the game.
pub fn play(app: &mut App) {
    if app.running.is_some() || !app.game.ok || game::compressing(app) {
        return;
    }
    if update::downloading(app) {
        app.set_status("An update is being downloaded. Press Play when it is done.");
        return;
    }
    if shaders::running(app) {
        let t = "The shaders are still being prepared. Stop preparing them and start the game now?\n\nThe next preparation goes on where this one stops.";
        if unsafe { ui::message(app.hwnd, t, TITLE, MB_ICONQUESTION | MB_YESNO) } == IDYES {
            shaders::stop_then_play(app);
        }
        return;
    }
    // (During the game's check: Play goes on once it answers, the store judged with it.)
    if shaders::defer_play(app) {
        return;
    }
    // (Not again after the player declined preparing for lack of space in this session. Not on
    // battery either: an hour of the PC's full load would come before the game; the page says how
    // to prepare later, 2026-11-21.)
    if app.launch.prepare_shaders && !app.shaders.space_declined && !shaders::prepared(app) {
        if let Some(pct) = sys::battery() {
            shaders::note_battery(app, pct);
            start_game(app);
            return;
        }
        shaders::start(app, true);
        return;
    }
    start_game(app);
}

/// Open logs: Explorer at the logs folder with the newest session's folder selected (the
/// player's or a shader preparation step's; run folders are named by their time, the launcher's
/// quick checks log in `checks`), else the folder.
fn open_logs(app: &App) {
    match newest_log(&app.paths) {
        Some(p) => ui::select_path(&p),
        None => ui::open_path(&app.paths.logs()),
    }
}

/// Starts the game (the launcher minimized while it runs; it comes back when the game exits).
pub fn start_game(app: &mut App) {
    if app.running.is_some() {
        return;
    }
    // (Not while the game folder is being compressed: also when a preparation that ends now
    // would start the game.)
    if game::compressing(app) {
        app.set_status("The game folder is being compressed. Press Play when it is done.");
        return;
    }
    let exe = dsconfig::paths::game_exe_in(&app.paths.exe_dir);
    if !exe.is_file() {
        unsafe { ui::message(app.hwnd, &format!("{} was not found next to the launcher.", dsconfig::paths::GAME_EXE), TITLE, MB_ICONERROR) };
        return;
    }
    // (The translation check runs the game for a few seconds when the launcher opens, and the
    // prepared shaders' probe after it.)
    shaders::wait_check();
    shaders::settle_probe(app);
    let mut cmd = std::process::Command::new(&exe);
    cmd.current_dir(&app.paths.exe_dir);
    let safe = game::safe_mode(app);
    if safe {
        cmd.arg("--safe-mode");
    }
    match cmd.spawn() {
        Ok(mut child) => {
            if safe {
                game::clear_safe_mode(app);
            }
            app.running = Some(child.id());
            app.update_play();
            let hwnd = app.hwnd as usize;
            std::thread::spawn(move || {
                let code = child.wait().ok().and_then(|s| s.code()).unwrap_or(-1);
                unsafe { PostMessageW(hwnd as HWND, WM_GAME_EXIT, code as u32 as WPARAM, 0) };
            });
            app.set_status(if safe { "The game runs in safe mode." } else { "The game runs." });
            unsafe { ShowWindow(app.hwnd, SW_MINIMIZE) };
        }
        Err(e) => unsafe {
            ui::message(app.hwnd, &format!("Could not start {}: {e}", exe.display()), TITLE, MB_ICONERROR);
        },
    }
}

/// The newest run folder in the logs (a session's log): folders named by their time (the
/// launcher's quick checks log in `checks`, `Paths::check_logs`).
fn newest_log(paths: &Paths) -> Option<PathBuf> {
    let dated = |p: &PathBuf| p.is_dir() && p.file_name().and_then(|n| n.to_str()).is_some_and(|n| n.starts_with(|c: char| c.is_ascii_digit()));
    let mut v: Vec<PathBuf> = std::fs::read_dir(paths.logs()).ok()?.flatten().map(|e| e.path()).filter(dated).collect();
    v.sort();
    v.pop()
}

/// The last error lines of a session's log (the time stamps left out).
fn last_errors(dir: &std::path::Path) -> Vec<String> {
    let Ok(b) = std::fs::read(dir.join("dsport.log")) else { return Vec::new() };
    let t = String::from_utf8_lossy(&b);
    let errs: Vec<String> = t.lines().filter_map(|l| l.split_once("] E: ").map(|(_, m)| m.chars().take(300).collect())).collect();
    errs[errs.len().saturating_sub(5)..].to_vec()
}

fn handle_exit(hwnd: HWND, wp: WPARAM) {
    if with_app(|app| game_exited(app, wp as u32 as i32)).is_none() {
        PENDING.with(|p| p.borrow_mut().push(wp));
        unsafe { SetTimer(hwnd, TIMER_RETRY, 250, None) };
    }
}

fn game_exited(app: &mut App, code: i32) {
    app.running = None;
    // (This exit is reported here: its marker would tell the next start of a session that did
    // not end normally.)
    let _ = std::fs::remove_file(app.paths.running_marker());
    app.update_play();
    saves::refresh(app);
    // (The game may have changed its display settings.)
    display::refresh(app);
    unsafe {
        ShowWindow(app.hwnd, SW_RESTORE);
        SetForegroundWindow(app.hwnd);
        ui::redraw(app.hwnd);
    }
    match code {
        0 => {
            app.set_status(shaders::battery_note(app).unwrap_or(""));
            // (The update's backup is no longer needed once a session of the new version ended
            // normally.)
            update::session_ok(app);
        }
        EXIT_GAME_FILES => app.set_status("The game could not use its game files. The Game page shows why."),
        EXIT_NO_GPU => app.set_status("The game could not start the graphics card. Restart Windows, or update the graphics driver."),
        EXIT_CPU | EXIT_DEVICE_LOST | EXIT_MEMORY | EXIT_RUNNING | EXIT_EARLY | 0xD500..=0xD5FF => app.set_status(&format!("The game stopped with an error (code {code:#x}). Open logs shows its log.")),
        _ => {
            let log = newest_log(&app.paths);
            let errs = log.as_deref().map(last_errors).unwrap_or_default();
            app.set_status(&format!("The game stopped with an error (code {code:#x})."));
            let text = format!(
                "The game stopped with an error (code {code:#x}).\n\n{}\n\nIts log is in:\n{}\n\nPlease attach that log when you report the problem. Open the log folder now?",
                if errs.is_empty() { "No error was recorded.".to_string() } else { errs.join("\n") },
                log.as_ref().map(|d| d.display().to_string()).unwrap_or_else(|| app.paths.logs().display().to_string())
            );
            if unsafe { ui::message(app.hwnd, &text, TITLE, MB_ICONERROR | MB_YESNO) } == IDYES {
                ui::open_path(&log.unwrap_or_else(|| app.paths.logs()));
            }
        }
    }
}

/// Reset settings: the PC settings, display choices, key bindings and audio device back to
/// their defaults (the files kept as .bak); the game folder, saves and shaders stay.
fn reset_settings(app: &mut App) {
    if app.running.is_some() {
        app.set_status("Close the game first.");
        return;
    }
    let text = "Reset the game's PC settings, display choices and key bindings to their defaults?\n\nThe current files are kept as .bak files. Saves are not touched.";
    if unsafe { ui::message(app.hwnd, text, TITLE, MB_ICONQUESTION | MB_OKCANCEL) } != IDOK {
        return;
    }
    let files = [app.paths.settings_file(), app.paths.display_file(), app.paths.controls_file(), app.paths.user.join("audio-device.txt")];
    let mut failed = Vec::new();
    for f in files.iter().filter(|f| f.exists()) {
        let mut b = f.clone().into_os_string();
        b.push(".bak");
        if let Err(e) = std::fs::rename(f, PathBuf::from(b)) {
            failed.push(format!("{}: {e}", f.display()));
        }
    }
    // (The Display page shows the defaults now.)
    display::refresh(app);
    if failed.is_empty() {
        app.set_status("Settings reset. The game starts with its defaults.");
    } else {
        unsafe { ui::message(app.hwnd, &format!("Some settings could not be reset:\n{}", failed.join("\n")), TITLE, MB_ICONERROR | MB_OK) };
    }
}

/// Closing while the shaders are prepared: they stop after their current ones.
fn close(app: &mut App) -> bool {
    if game::compressing(app) {
        let t = "The game is still being compressed. Stop and close?\n\nThe unfinished archive is removed.";
        if unsafe { ui::message(app.hwnd, t, TITLE, MB_ICONQUESTION | MB_YESNO) } == IDYES {
            // (The launcher closes once the compressing thread has removed it.)
            game::stop_then_close(app);
        }
        return false;
    }
    if shaders::running(app) && !shaders::stopping(app) {
        let t = "The shaders are still being prepared. Stop and close?\n\nThe next preparation goes on where this one stops.";
        if unsafe { ui::message(app.hwnd, t, TITLE, MB_ICONQUESTION | MB_YESNO) } != IDYES {
            return false;
        }
        shaders::stop(app);
        shaders::end_for_close(app);
    }
    true
}

unsafe extern "system" fn wndproc(hwnd: HWND, msg: u32, wp: WPARAM, lp: LPARAM) -> LRESULT {
    let handled = match msg {
        WM_COMMAND => {
            let (id, code) = ((wp & 0xFFFF) as i32, ((wp >> 16) & 0xFFFF) as u32);
            with_app(|app| match id {
                ID_PLAY => play(app),
                ID_RESET => reset_settings(app),
                ID_LOGS => open_logs(app),
                ID_DISCORD => ui::open_url(DISCORD),
                ID_UNINSTALL => uninstall::run(app),
                update::ID_UPDATE | update::ID_CHECK_UPDATES => update::command(app, id),
                dlss::ID_GET | dlss::ID_LICENSE => dlss::command(app, id),
                100..=199 => game::command(app, id, code),
                200..=299 => shaders::command(app, id, code),
                300..=399 => saves::command(app, id, code),
                400..=499 => display::command(app, id, code),
                _ => {}
            })
            .is_some()
        }
        WM_NOTIFY => {
            let h = unsafe { &*(lp as *const NMHDR) };
            with_app(|app| {
                if h.idFrom == ID_TABS as usize && h.code == TCN_SELCHANGE {
                    let n = unsafe { SendMessageW(app.tabs, TCM_GETCURSEL, 0, 0) }.max(0) as usize;
                    app.show_page(n);
                } else if h.idFrom / 100 == 3 && h.code == LVN_ITEMCHANGED {
                    saves::notify(app);
                }
            })
            .is_some()
        }
        WM_TIMER if wp == TIMER_RETRY => {
            unsafe { KillTimer(hwnd, TIMER_RETRY) };
            let pending: Vec<WPARAM> = PENDING.with(|p| std::mem::take(&mut *p.borrow_mut()));
            for w in pending {
                handle_exit(hwnd, w);
            }
            true
        }
        WM_TIMER if wp == shaders::TIMER => with_app(shaders::tick).is_some(),
        WM_TIMER if wp == shaders::CHECK_TIMER => with_app(shaders::check_tick).is_some(),
        WM_TIMER if wp == game::TIMER => with_app(game::tick).is_some(),
        WM_TIMER if wp == update::TIMER => with_app(update::tick).is_some(),
        WM_TIMER if wp == dlss::TIMER => with_app(dlss::tick).is_some(),
        WM_TIMER if wp == update::RETRY_TIMER => {
            update::retry_now(hwnd);
            true
        }
        // (Came while a dialog held the app: posted again shortly, not lost.)
        update::WM_CHECKED | update::WM_STARTED => {
            if with_app(|app| update::message(app, msg, wp)).is_none() {
                update::retry_later(hwnd, msg, wp);
            }
            true
        }
        WM_GAME_EXIT => {
            handle_exit(hwnd, wp);
            true
        }
        WM_DPICHANGED => {
            let r = unsafe { &*(lp as *const RECT) };
            unsafe { SetWindowPos(hwnd, std::ptr::null_mut(), r.left, r.top, r.right - r.left, r.bottom - r.top, SWP_NOZORDER | SWP_NOACTIVATE) };
            let dpi = (wp & 0xFFFF) as u32;
            with_app(|app| app.relayout(dpi));
            true
        }
        WM_CLOSE => {
            if with_app(close).unwrap_or(true) {
                unsafe { DestroyWindow(hwnd) };
            }
            true
        }
        WM_DESTROY => {
            unsafe { PostQuitMessage(0) };
            true
        }
        _ => false,
    };
    if handled {
        return 0;
    }
    unsafe { DefWindowProcW(hwnd, msg, wp, lp) }
}

/// The opening status line: a game still running, or a session that did not end normally.
fn startup_status(app: &mut App) {
    let marker = app.paths.running_marker();
    let Ok(t) = std::fs::read_to_string(&marker) else {
        return;
    };
    let pid = t.lines().find_map(|l| l.strip_prefix("pid ")).and_then(|p| p.trim().parse::<u32>().ok());
    if let Some(pid) = pid.filter(|&p| sys::process_alive(p)) {
        app.running = Some(pid);
        app.update_play();
        app.set_status("The game is running.");
        // (Its end posted as a game started here would post it: Play comes back then, the
        // review of 2026-11-20.)
        sys::watch_process(pid, app.hwnd as usize, WM_GAME_EXIT);
    } else {
        let _ = std::fs::remove_file(&marker);
        app.set_status("The last session did not end normally. Open logs shows its log.");
    }
}

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let arg = |name: &str| args.iter().position(|a| a == name).and_then(|i| args.get(i + 1)).cloned();
    // Tests: `--test-fetch-dlss <folder>` fetches and checks NVIDIA's DLSS file into that folder as
    // the Display page does (no window), the result in <folder>\fetch-result.txt and the exit code.
    if let Some(dir) = arg("--test-fetch-dlss") {
        let dir = std::path::PathBuf::from(dir);
        let r = dlss::fetch(&dir, &std::sync::atomic::AtomicU64::new(0));
        let _ = std::fs::write(dir.join("fetch-result.txt"), format!("{r:?}\n"));
        std::process::exit(if r.is_ok() { 0 } else { 1 });
    }
    unsafe {
        SetProcessDpiAwarenessContext(DPI_AWARENESS_CONTEXT_PER_MONITOR_AWARE_V2);
        let _ = windows::Win32::System::Com::CoInitializeEx(None, windows::Win32::System::Com::COINIT_APARTMENTTHREADED);
        let icc = INITCOMMONCONTROLSEX { dwSize: std::mem::size_of::<INITCOMMONCONTROLSEX>() as u32, dwICC: ICC_TAB_CLASSES | ICC_LISTVIEW_CLASSES | ICC_BAR_CLASSES | ICC_PROGRESS_CLASS | ICC_STANDARD_CLASSES };
        InitCommonControlsEx(&icc);
        let hinst = GetModuleHandleW(std::ptr::null());
        let class = wide("regaliant_launcher");
        // (Started by an update: the launcher that started it waits, hidden, until this one says it
        // is up, and puts the earlier version back otherwise.)
        let after_update = arg("--after-update").and_then(|p| p.parse::<u32>().ok());
        // One launcher at a time (two could prepare the shaders twice): a second start brings
        // the first one to the front. (`--shot` runs are layout checks: they may run beside one;
        // so does a launcher an update started, beside the hidden one that started it.)
        if arg("--shot").is_none() && after_update.is_none() {
            let other = FindWindowW(class.as_ptr(), std::ptr::null());
            if !other.is_null() {
                ShowWindow(other, SW_RESTORE);
                SetForegroundWindow(other);
                return;
            }
        }
        let wc = WNDCLASSEXW {
            cbSize: std::mem::size_of::<WNDCLASSEXW>() as u32,
            style: CS_HREDRAW | CS_VREDRAW,
            lpfnWndProc: Some(wndproc),
            cbClsExtra: 0,
            cbWndExtra: 0,
            hInstance: hinst,
            hIcon: LoadIconW(hinst, 1 as _),
            hCursor: LoadCursorW(std::ptr::null_mut(), IDC_ARROW),
            hbrBackground: (windows_sys::Win32::Graphics::Gdi::COLOR_BTNFACE + 1) as usize as _,
            lpszMenuName: std::ptr::null(),
            lpszClassName: class.as_ptr(),
            // (The title bar's small icon: picked from the same icon's sizes.)
            hIconSm: std::ptr::null_mut(),
        };
        RegisterClassExW(&wc);
        let style = WS_OVERLAPPED | WS_CAPTION | WS_SYSMENU | WS_MINIMIZEBOX;
        let title = wide(&format!("{TITLE} {}", VERSION));
        let hwnd = CreateWindowExW(0, class.as_ptr(), title.as_ptr(), style, CW_USEDEFAULT, CW_USEDEFAULT, 800, 640, std::ptr::null_mut(), std::ptr::null_mut(), hinst, std::ptr::null());
        if hwnd.is_null() {
            return;
        }
        let dpi = GetDpiForWindow(hwnd).max(96);
        let scale = Scale(dpi);
        let mut r = RECT { left: 0, top: 0, right: scale.px(CLIENT.0), bottom: scale.px(CLIENT.1) };
        AdjustWindowRectExForDpi(&mut r, style, 0, 0, dpi);
        SetWindowPos(hwnd, std::ptr::null_mut(), 0, 0, r.right - r.left, r.bottom - r.top, SWP_NOMOVE | SWP_NOZORDER);

        let paths = Paths::current();
        let launch = Launch::load(&paths);
        let font = ui::message_font(dpi, false, 100);
        let bold = ui::message_font(dpi, true, 110);
        let mut app = App {
            hwnd,
            scale,
            font,
            bold,
            layout: Vec::new(),
            paths,
            launch,
            tabs: std::ptr::null_mut(),
            pages: vec![Vec::new(); 4],
            page: 0,
            status: std::ptr::null_mut(),
            play: std::ptr::null_mut(),
            game: game::Page::default(),
            display: display::Page::default(),
            shaders: shaders::Page::default(),
            saves: saves::Page::default(),
            dlss: dlss::Row::default(),
            update: update::Page::default(),
            running: None,
        };
        let b = app.builder();
        let rt = (12, 10, 756, 30);
        let tabs = b.add("SysTabControl32", "", WS_TABSTOP | WS_CLIPSIBLINGS, 0, rt, ID_TABS);
        app.tabs = app.put(usize::MAX, tabs, rt);
        for (i, t) in ["Game", "Display", "Shaders", "Saves"].iter().enumerate() {
            let mut w = wide(t);
            let item = TCITEMW { mask: TCIF_TEXT, pszText: w.as_mut_ptr(), ..Default::default() };
            SendMessageW(tabs, TCM_INSERTITEMW, i, &item as *const _ as LPARAM);
        }
        let r = (16, 520, 748, 20);
        app.status = app.put(usize::MAX, b.label("", r, ID_STATUS_BAR), r);
        let r = (16, 548, 120, 36);
        app.play = app.put(usize::MAX, b.add("BUTTON", "Play", WS_TABSTOP | BS_DEFPUSHBUTTON as u32, 0, r, ID_PLAY), r);
        SendMessageW(app.play, WM_SETFONT, app.bold as WPARAM, 1);
        let r = (144, 548, 130, 36);
        app.put(usize::MAX, b.button("Reset settings...", r, ID_RESET), r);
        let r = (282, 548, 110, 36);
        app.put(usize::MAX, b.button("Open logs", r, ID_LOGS), r);
        let r = (400, 548, 110, 36);
        app.put(usize::MAX, b.button("Uninstall...", r, ID_UNINSTALL), r);
        // (At the row's right end.)
        let r = (614, 548, 150, 36);
        app.put(usize::MAX, b.button("Join the Discord", r, ID_DISCORD), r);
        // (Saves first: the Game page's check refreshes them.)
        saves::build(&mut app);
        game::build(&mut app);
        display::build(&mut app);
        shaders::build(&mut app);
        update::build(&mut app);
        let first = arg("--page").and_then(|n| n.parse::<usize>().ok()).unwrap_or(0).min(3);
        SendMessageW(app.tabs, TCM_SETCURSEL, first, 0);
        app.show_page(first);
        app.update_play();
        startup_status(&mut app);
        // (The prepared shaders checked against this build's translation, once a game folder
        // passes its check: the check reads the game's shader files, and without one it ran the
        // game only to fail and log an error at every start of a new player's launcher
        // (`game::check` begins it when a folder is chosen); `--shot` runs skip it.)
        if arg("--shot").is_none() && app.game.ok {
            shaders::start_check(&app);
            SetTimer(hwnd, shaders::CHECK_TIMER, 500, None);
        }
        // (A newer release asked for at every start, the player's check box allowing.)
        if arg("--shot").is_none() {
            update::start_check(&app);
        }
        // `--shot <file>`: each page as <file>-<page>.png, then exit (layout checks).
        let shot = arg("--shot").map(PathBuf::from);
        APP.with(|a| *a.borrow_mut() = Some(app));
        // (`DS_WINDOW_BEHIND=1`, tests on a shared desktop, as the game's: shown without taking
        // the focus, below the other windows.)
        if std::env::var("DS_WINDOW_BEHIND").is_ok_and(|v| v.trim() == "1") {
            ShowWindow(hwnd, SW_SHOWNOACTIVATE);
            SetWindowPos(hwnd, HWND_BOTTOM, 0, 0, 0, 0, SWP_NOMOVE | SWP_NOSIZE | SWP_NOACTIVATE);
        } else {
            ShowWindow(hwnd, SW_SHOW);
        }
        ui::redraw(hwnd);
        // (Up: the launcher that started this one may close. Then what the update changed, once,
        // over the shown window.)
        if let Some(pid) = after_update {
            update::signal_ready(pid);
            with_app(update::after_start);
        }
        if let Some(file) = shot {
            for n in 0..4 {
                with_app(|app| {
                    SendMessageW(app.tabs, TCM_SETCURSEL, n, 0);
                    app.show_page(n);
                });
                use windows_sys::Win32::Graphics::Gdi::{RDW_ALLCHILDREN, RDW_ERASE, RDW_INVALIDATE, RDW_UPDATENOW, RedrawWindow};
                RedrawWindow(hwnd, std::ptr::null(), std::ptr::null_mut(), RDW_INVALIDATE | RDW_ERASE | RDW_ALLCHILDREN | RDW_UPDATENOW);
                for _ in 0..5 {
                    let mut msg: MSG = std::mem::zeroed();
                    while PeekMessageW(&mut msg, std::ptr::null_mut(), 0, 0, PM_REMOVE) != 0 {
                        TranslateMessage(&msg);
                        DispatchMessageW(&msg);
                    }
                    std::thread::sleep(std::time::Duration::from_millis(100));
                }
                let f = file.with_file_name(format!("{}-{n}.png", file.file_stem().map(|s| s.to_string_lossy().into_owned()).unwrap_or_default()));
                let _ = shot::window(hwnd, &f);
            }
            DestroyWindow(hwnd);
            return;
        }
        let mut msg: MSG = std::mem::zeroed();
        while GetMessageW(&mut msg, std::ptr::null_mut(), 0, 0) > 0 {
            if IsDialogMessageW(hwnd, &msg) == 0 {
                TranslateMessage(&msg);
                DispatchMessageW(&msg);
            }
        }
    }
}
