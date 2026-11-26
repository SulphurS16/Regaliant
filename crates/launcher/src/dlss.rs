//! NVIDIA's DLSS file on the Display page. nvngx_dlss.dll is NVIDIA's software under NVIDIA's
//! license and never part of the port's package: the player's own copy (README). While it is
//! missing the page offers to fetch it from NVIDIA's own DLSS SDK on GitHub (the v310.9.16
//! release's file, the same as Streamline SDK 2.14.16's copy) once the player accepts NVIDIA's
//! license. The download must be exactly that file (its size and SHA-256), and the game checks
//! NVIDIA's signature before it loads it, as for any copy. (Players' suggestion of 2026-11-26
//! evening: copying the file by hand kept many from DLSS.)

use crate::ui::{self, Builder};
use crate::App;
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicU64, Ordering::Relaxed};
use std::sync::{Arc, Mutex};
use windows_sys::Win32::Foundation::HWND;
use windows_sys::Win32::UI::WindowsAndMessaging::{IDOK, KillTimer, MB_ICONERROR, MB_ICONINFORMATION, MB_ICONQUESTION, MB_OK, MB_OKCANCEL, SetTimer};

/// The file's name in the upscalers folder.
pub const FILE: &str = "nvngx_dlss.dll";
/// NVIDIA's DLSS SDK, the v310.9.16 release's file.
const URL: &str = "https://raw.githubusercontent.com/NVIDIA/DLSS/v310.9.16/lib/Windows_x86_64/rel/nvngx_dlss.dll";
/// NVIDIA's license for it (shown before the download and on the page's button).
pub const LICENSE: &str = "https://github.com/NVIDIA/DLSS/blob/v310.9.16/LICENSE.txt";
const SIZE: u64 = 58_956_912;
const SHA256: &str = "d00b5b0ab29c3ca101b70c3a5ecb3a01b7ef33766a3383a28a18623083e6123e";

pub const TIMER: usize = 21;
pub const ID_GET: i32 = 450;
pub const ID_LICENSE: i32 = 451;

pub struct Job {
    done: Arc<AtomicU64>,
    result: Arc<Mutex<Option<Result<(), String>>>>,
}

#[derive(Default)]
pub struct Row {
    status: usize,
    get: usize,
    job: Option<Job>,
}

fn h(v: usize) -> HWND {
    v as HWND
}

/// The row under the Display page's settings.
pub fn build(app: &mut App, page: usize) {
    let b: Builder = app.builder();
    unsafe {
        let r = (16, 428, 400, 22);
        app.dlss.status = app.put(page, b.label("", r, -1), r) as usize;
        let r = (424, 424, 180, 26);
        app.dlss.get = app.put(page, b.button("Download DLSS...", r, ID_GET), r) as usize;
        let r = (612, 424, 152, 26);
        app.put(page, b.button("NVIDIA's license", r, ID_LICENSE), r);
    }
    refresh(app);
}

fn installed(app: &App) -> bool {
    app.paths.upscalers().join(FILE).is_file()
}

/// The row's text and button as things stand.
pub fn refresh(app: &App) {
    if app.dlss.status == 0 {
        return;
    }
    let have = installed(app);
    let text = if app.dlss.job.is_some() {
        "Downloading NVIDIA's DLSS file."
    } else if have {
        "DLSS: NVIDIA's file is in the upscalers folder."
    } else {
        "DLSS needs NVIDIA's file, nvngx_dlss.dll."
    };
    unsafe {
        ui::set_text(h(app.dlss.status), text);
        ui::enable(h(app.dlss.get), !have && app.dlss.job.is_none());
    }
}

pub fn command(app: &mut App, id: i32) {
    match id {
        ID_LICENSE => ui::open_url(LICENSE),
        ID_GET => start(app),
        _ => {}
    }
}

fn start(app: &mut App) {
    if app.dlss.job.is_some() || installed(app) {
        return;
    }
    let text = format!(
        "Download NVIDIA's DLSS file (nvngx_dlss.dll, 59 MB) from NVIDIA's DLSS SDK on GitHub?\n\nIt is NVIDIA's software. By downloading it you accept NVIDIA's license:\n{LICENSE}\n\nThe launcher checks the file and puts it in the upscalers folder."
    );
    if unsafe { ui::message(app.hwnd, &text, crate::TITLE, MB_ICONQUESTION | MB_OKCANCEL) } != IDOK {
        return;
    }
    let dir = app.paths.upscalers();
    let done = Arc::new(AtomicU64::new(0));
    let result = Arc::new(Mutex::new(None));
    let (d, r) = (done.clone(), result.clone());
    std::thread::spawn(move || {
        let out = fetch(&dir, &d);
        *r.lock().unwrap() = Some(out);
    });
    app.dlss.job = Some(Job { done, result });
    crate::shaders::log(app, "DLSS file: the player accepted NVIDIA's license, the download started");
    unsafe { SetTimer(app.hwnd, TIMER, 250, None) };
    refresh(app);
}

/// The download into `dir` (resumed where an earlier try stopped), checked, then named.
pub fn fetch(dir: &Path, done: &AtomicU64) -> Result<(), String> {
    std::fs::create_dir_all(dir).map_err(|e| format!("the upscalers folder: {e}"))?;
    let part: PathBuf = dir.join(format!("{FILE}.part"));
    crate::update::download(URL, &part, SIZE, done)?;
    let got = std::fs::metadata(&part).map(|m| m.len()).unwrap_or(0);
    let sha = dsconfig::sha::file_sha256(&part).map_err(|e| format!("reading the download: {e}"))?;
    if got != SIZE || !sha.eq_ignore_ascii_case(SHA256) {
        let _ = std::fs::remove_file(&part);
        return Err("the download is not NVIDIA's file as released".into());
    }
    let to = dir.join(FILE);
    if to.exists() {
        // (One appeared meanwhile, the player's own: it stays.)
        let _ = std::fs::remove_file(&part);
        return Ok(());
    }
    std::fs::rename(&part, &to).map_err(|e| format!("naming the file: {e}"))
}

pub fn tick(app: &mut App) {
    let Some(job) = &app.dlss.job else {
        unsafe { KillTimer(app.hwnd, TIMER) };
        return;
    };
    let r = job.result.lock().unwrap().take();
    let Some(r) = r else {
        app.set_status(&format!("Downloading NVIDIA's DLSS file: {} of {} MB.", job.done.load(Relaxed) / 1_000_000, (SIZE + 500_000) / 1_000_000));
        return;
    };
    unsafe { KillTimer(app.hwnd, TIMER) };
    app.dlss.job = None;
    match r {
        Ok(()) => {
            crate::shaders::log(app, "DLSS file: downloaded and checked");
            app.set_status("NVIDIA's DLSS file is ready.");
            let text = if app.running.is_some() {
                "DLSS is ready from the game's next start. Then choose it in SETTINGS > PC, Anti-Aliasing."
            } else {
                "DLSS is ready. Choose it in the game's SETTINGS > PC tab, Anti-Aliasing."
            };
            unsafe { ui::message(app.hwnd, text, crate::TITLE, MB_ICONINFORMATION | MB_OK) };
        }
        Err(e) => {
            crate::shaders::log(app, &format!("DLSS file: not downloaded: {e}"));
            app.set_status("NVIDIA's DLSS file was not downloaded.");
            let text = format!("NVIDIA's DLSS file could not be downloaded: {e}.\n\nYou can copy your own nvngx_dlss.dll into the upscalers folder instead.");
            unsafe { ui::message(app.hwnd, &text, crate::TITLE, MB_ICONERROR | MB_OK) };
        }
    }
    refresh(app);
}
