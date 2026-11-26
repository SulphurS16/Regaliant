//! The Game page: the game folder or its `.zar` archive (checked when chosen, kept in game.txt
//! once it passes), the folder compressed into an archive, what this PC brings (graphics card,
//! memory, processor) and safe mode for the next start.

use crate::sys;
use crate::ui::{self, Builder};
use crate::App;
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering::Relaxed};
use std::sync::{Arc, Mutex};
use std::time::Instant;
use windows_sys::Win32::Foundation::HWND;
use windows_sys::Win32::UI::WindowsAndMessaging::{IDYES, KillTimer, MB_ICONQUESTION, MB_ICONWARNING, MB_OK, MB_YESNO, SetTimer};

const ID_DIR: i32 = 101;
const ID_BROWSE: i32 = 102;
const ID_CHECK: i32 = 103;
const ID_ZAR: i32 = 104;
const ID_COMPRESS: i32 = 105;
const ID_SAFE: i32 = 106;
/// The compression's progress, shown twice a second.
pub const TIMER: usize = 7;

#[derive(Default)]
pub struct Page {
    dir: usize,
    status: usize,
    compress: usize,
    pc: usize,
    safe: usize,
    /// Whether the game folder passed the check.
    pub ok: bool,
    /// The checked game's folder when it is a folder (what Compress makes an archive of).
    folder: Option<PathBuf>,
    /// The game's check, shown again under the compression's progress.
    text: String,
    job: Option<Job>,
    /// Close the launcher once the compressing thread has ended (its partial archive removed).
    close_after: bool,
}

/// A compression into a `.zar` under way: its thread writes, the page's timer shows it.
struct Job {
    from: PathBuf,
    out: PathBuf,
    t0: Instant,
    done: Arc<AtomicU64>,
    total: Arc<AtomicU64>,
    stop: Arc<AtomicBool>,
    result: Arc<Mutex<Option<std::io::Result<dsconfig::zar::Written>>>>,
}

fn h(v: usize) -> HWND {
    v as HWND
}

pub fn build(app: &mut App) {
    let b: Builder = app.builder();
    let dir = app.paths.chosen_game().filter(|d| d.exists()).map(|d| d.display().to_string()).unwrap_or_default();
    unsafe {
        let r = (16, 67, 100, 20);
        app.put(0, b.label("Game:", r, -1), r);
        let r = (116, 64, 340, 24);
        app.game.dir = app.put(0, b.edit(&dir, r, ID_DIR), r) as usize;
        let r = (464, 63, 96, 26);
        app.put(0, b.button("Folder...", r, ID_BROWSE), r);
        let r = (568, 63, 96, 26);
        app.put(0, b.button(".zar file...", r, ID_ZAR), r);
        let r = (672, 63, 92, 26);
        app.put(0, b.button("Check", r, ID_CHECK), r);
        let r = (16, 102, 600, 90);
        app.game.status = app.put(0, b.label("", r, -1), r) as usize;
        let r = (624, 100, 140, 26);
        app.game.compress = app.put(0, b.button("Compress to .zar...", r, ID_COMPRESS), r) as usize;
        let r = (12, 196, 756, 150);
        app.put(0, b.group("This PC", r), r);
        let r = (26, 218, 730, 120);
        app.game.pc = app.put(0, b.label("", r, -1), r) as usize;
        let r = (16, 360, 748, 40);
        app.game.safe = app.put(0, b.check("Start in safe mode once: default settings for that session, in a window, without an upscaler, Frame Generation or Reflex. Use it when the game does not start.", r, ID_SAFE), r) as usize;
        ui::set_check(h(app.game.safe), app.launch.safe_mode);
        // (Also in the launcher's log: what the player was told about this PC, in bug reports and
        // the small-card tests.)
        let pc = pc_text();
        ui::set_text(h(app.game.pc), &pc);
        crate::shaders::log(app, &format!("this PC: {}", pc.replace('\n', " | ")));
    }
    if dir.is_empty() {
        unsafe { ui::set_text(h(app.game.status), "Choose the game's folder or its .zar archive: the PS5 game Demon's Souls (PPSA01341), version 01.007.000, with its executable decrypted.") };
        app.game.ok = false;
    } else {
        check(app, Path::new(&dir), false);
    }
    update(app);
}

/// What this PC brings, in short lines.
fn pc_text() -> String {
    let mut lines = Vec::new();
    match sys::gpu() {
        Ok(g) => {
            lines.push(format!("Graphics card: {}, {} of video memory, driver {}.", g.name, sys::gb(g.vram), g.driver_text));
            if g.api < ash::vk::API_VERSION_1_3 {
                lines.push("It offers an older Vulkan than the game needs (1.3). Update the graphics driver.".into());
            }
            if !g.barycentric && !g.geometry {
                lines.push("This card lacks shader features the game needs. Surfaces will look faceted.".into());
            } else if !g.barycentric {
                lines.push("This card lacks fragment shader barycentrics. The game draws some surfaces another way, which costs some speed.".into());
            }
            if g.vendor == sys::VENDOR_AMD || g.vendor == sys::VENDOR_INTEL {
                lines.push("AMD and Intel graphics cards are untested with this port. Please report how it runs.".into());
            }
            // (Measured in smaller rooms simulated on a 24 GB card, 2026-11-22: from 7 GB of room
            // the picture stays right with textures in system memory in busy places; 4 GB of room
            // and 3.5 GB (what a 4 GB card leaves a game) draw right in the 8-level tour, slowly; what a
            // 3 GB card leaves drew parts of some levels wrong.)
            if g.vram < 3584 << 20 {
                lines.push("Cards with less than 4 GB of video memory are not supported. Parts of the game can draw wrong.".into());
            } else if g.vram < 5632 << 20 {
                lines.push("This card has little video memory for this game. It runs, but slowly. The Low preset in the game's PC tab helps.".into());
            } else if g.vram < 11 << 30 {
                lines.push("On this card the game keeps some textures in system memory. Busy places can run slower.".into());
            }
        }
        Err(e) => lines.push(e),
    }
    let mem = sys::memory();
    lines.push(format!("Memory: {}. Processor: {} threads.", sys::gb(mem), sys::threads()));
    // (The card reads the game's memory through Windows' graphics share of it, about half the
    // installed memory. A 16 GB PC simulated on the tested one: on 2026-11-22 afternoon the game
    // froze at the first level load and lost its graphics device; since that night's eviction
    // and write-back fallback it played every tested level, a first session as fast as with
    // room.)
    if mem < 15 << 30 {
        lines.push("The game needs 16 GB of memory. It may stop when memory runs out.".into());
    } else if mem < 24 << 30 {
        lines.push("16 GB of memory has not been tested on a real PC. 32 GB is recommended.".into());
        lines.push("Keep Windows' page file on automatic and about 30 GB free on the system drive.".into());
    } else if mem < 30 << 30 {
        lines.push("Keep Windows' page file on automatic and about 30 GB free on the system drive.".into());
    }
    if sys::threads() < 8 {
        lines.push("The game was tested with at least 6 processor cores. It may run slowly here.".into());
    }
    lines.join("\n")
}

/// Checks the folder or archive; once it passes (and `keep`), game.txt takes it.
pub fn check(app: &mut App, dir: &Path, keep: bool) {
    let (ok, text) = match dsconfig::game::check(dir) {
        Ok(g) => {
            if keep {
                if let Err(e) = app.paths.set_game(&g.app) {
                    app.set_status(&format!("Could not save the game folder: {e}"));
                }
            }
            let what = if g.source.is_archive() { "Found in the archive" } else { "Found in" };
            let mut t = format!("Demon's Souls {} is ready.\n{what} {}.", g.version, ui::short(&g.app, 90));
            if g.id != dsconfig::game::TITLE_ID {
                t.push_str(&format!("\nThis is the {} release. The port was tested with {}.", g.id, dsconfig::game::TITLE_ID));
            }
            match dsconfig::game::tested_executable(&g) {
                Ok(true) => {}
                Ok(false) => t.push_str("\nIts executable differs from the one this port was tested with. It may not work."),
                Err(e) => t.push_str(&format!("\nIts executable could not be read ({e}).")),
            }
            app.game.folder = (!g.source.is_archive()).then(|| g.app.clone());
            (true, t)
        }
        Err(e) => {
            app.game.folder = None;
            (false, e.to_string())
        }
    };
    app.game.ok = ok;
    app.game.text = text.clone();
    unsafe { ui::set_text(h(app.game.status), &text) };
    // (A folder chosen now: the prepared shaders' check, which needs the game's files, begins;
    // the Shaders page says what it can do now.)
    if keep {
        if ok {
            crate::shaders::start_check(app);
            unsafe { SetTimer(app.hwnd, crate::shaders::CHECK_TIMER, 500, None) };
        }
        crate::shaders::refresh(app);
    }
    update(app);
    app.update_play();
    crate::saves::refresh(app);
    unsafe { ui::redraw(app.hwnd) };
}

/// The Compress button: usable for a checked folder while neither the game runs nor the shaders
/// are being prepared; "Stop" while compressing.
pub fn update(app: &App) {
    let busy = app.game.job.is_some();
    unsafe {
        ui::set_text(h(app.game.compress), if busy { "Stop compressing" } else { "Compress to .zar..." });
        ui::enable(h(app.game.compress), busy || (app.game.ok && app.game.folder.is_some() && app.running.is_none() && !crate::shaders::running(app)));
    }
}

/// Whether the game folder is being compressed (Play waits for it).
pub fn compressing(app: &App) -> bool {
    app.game.job.is_some()
}

/// Safe mode for the next start (the box), cleared once used.
pub fn safe_mode(app: &App) -> bool {
    unsafe { ui::checked(h(app.game.safe)) }
}

pub fn clear_safe_mode(app: &mut App) {
    unsafe { ui::set_check(h(app.game.safe), false) };
    app.launch.safe_mode = false;
    app.save_launch();
}

fn check_field(app: &mut App) {
    let t = unsafe { ui::get_text(h(app.game.dir)) }.trim().trim_matches('"').to_string();
    if t.is_empty() {
        app.game.ok = false;
        app.game.folder = None;
        unsafe { ui::set_text(h(app.game.status), "Choose the game's folder or its .zar archive.") };
        crate::shaders::refresh(app);
        update(app);
        app.update_play();
        return;
    }
    check(app, Path::new(&t), true);
}

fn gib(b: u64) -> String {
    format!("{:.1} GB", b as f64 / (1u64 << 30) as f64)
}

/// Compress: the game folder into a `.zar` the player names, on a thread of its own.
fn start_compress(app: &mut App) {
    let Some(from) = app.game.folder.clone() else { return };
    let folder = from.parent().map(Path::to_path_buf).unwrap_or_else(|| from.clone());
    let name = format!("{}.zar", from.file_name().map(|n| n.to_string_lossy().into_owned()).unwrap_or_else(|| "Demon's Souls".into()));
    let Some(out) = ui::pick_file(app.hwnd, "Save the game as a .zar archive", ".zar archives", "*.zar", Some((&folder, &name))) else { return };
    // (Not inside the game folder: the folders resolved, compared as Windows names them, the
    // letters' case aside.)
    let canon = |p: &Path| std::fs::canonicalize(p).unwrap_or_else(|_| p.to_path_buf()).to_string_lossy().to_lowercase();
    let (into, game) = (canon(out.parent().unwrap_or(&out)), canon(&from));
    let game = game.trim_end_matches('\\').to_string();
    if into == game || into.starts_with(&format!("{game}\\")) {
        unsafe { ui::message(app.hwnd, "Save the archive outside the game folder.", crate::TITLE, MB_ICONWARNING | MB_OK) };
        return;
    }
    // (The tested copy's archive is 66.0% of its folder, 2026-11-23: a little more is asked for.)
    let (files, bytes) = match dsconfig::zar::folder_size(&from) {
        Ok(s) => s,
        Err(e) => {
            unsafe { ui::message(app.hwnd, &format!("The game folder could not be read ({e})."), crate::TITLE, MB_ICONWARNING | MB_OK) };
            return;
        }
    };
    let need = bytes / 100 * 68;
    let free = sys::free_space(out.parent().unwrap_or(&out));
    let mut t = format!(
        "Compress the game ({} in {files} files) into this archive?\n{}\n\nThe archive takes about {} on that drive. The game folder stays as it is. Once the archive works, you can delete the folder yourself.",
        gib(bytes),
        out.display(),
        gib(need)
    );
    if free.is_some_and(|f| f < need) {
        t.push_str(&format!("\n\nThat drive has only {} free. It may run out of space.", gib(free.unwrap_or(0))));
    }
    if unsafe { ui::message(app.hwnd, &t, crate::TITLE, MB_ICONQUESTION | MB_YESNO) } != IDYES {
        return;
    }
    let job = Job {
        from: from.clone(),
        out: out.clone(),
        t0: Instant::now(),
        done: Arc::new(AtomicU64::new(0)),
        total: Arc::new(AtomicU64::new(bytes)),
        stop: Arc::new(AtomicBool::new(false)),
        result: Arc::new(Mutex::new(None)),
    };
    let (done, total, stop, result) = (job.done.clone(), job.total.clone(), job.stop.clone(), job.result.clone());
    // (Every logical processor but one: the launcher stays responsive.)
    let threads = std::thread::available_parallelism().map_or(4, |n| n.get()).saturating_sub(1).max(1);
    crate::shaders::log(app, &format!("compress: {} into {} on {threads} threads (zstd level {})", from.display(), out.display(), dsconfig::zar::LEVEL));
    let spawned = std::thread::Builder::new().name("compress".into()).spawn(move || {
        let r = dsconfig::zar::write_folder(&from, &out, dsconfig::zar::LEVEL, threads, &|d, t| {
            done.store(d, Relaxed);
            total.store(t, Relaxed);
            !stop.load(Relaxed)
        });
        *result.lock().unwrap() = Some(r);
    });
    if let Err(e) = spawned {
        unsafe { ui::message(app.hwnd, &format!("Compressing could not start ({e})."), crate::TITLE, MB_ICONWARNING | MB_OK) };
        return;
    }
    app.game.job = Some(job);
    unsafe { SetTimer(app.hwnd, TIMER, 500, None) };
    update(app);
    app.update_play();
    tick(app);
}

/// Stops the compression (its thread then removes the partial archive).
pub fn stop_compress(app: &mut App) {
    if let Some(j) = &app.game.job {
        j.stop.store(true, Relaxed);
    }
}

/// Stops the compression and closes the launcher once its thread has ended (the window stays
/// responsive meanwhile, its status saying so).
pub fn stop_then_close(app: &mut App) {
    stop_compress(app);
    app.game.close_after = true;
    tick(app);
}

/// The compression's progress, and its end.
pub fn tick(app: &mut App) {
    let Some(j) = &app.game.job else {
        unsafe { KillTimer(app.hwnd, TIMER) };
        return;
    };
    let finished = j.result.lock().unwrap().take();
    let Some(r) = finished else {
        let (d, t) = (j.done.load(Relaxed), j.total.load(Relaxed).max(1));
        let secs = j.t0.elapsed().as_secs_f64();
        let mut s = format!("Compressing the game into {}: {}% ({} of {}).", ui::short(&j.out, 70), d * 100 / t, gib(d), gib(t));
        if j.stop.load(Relaxed) {
            s.push_str("\nStopping...");
        } else if d > 0 && secs > 5.0 {
            let left = (t.saturating_sub(d)) as f64 / (d as f64 / secs);
            s.push_str(&format!("\nAbout {} min left.", (left / 60.0).ceil() as u64));
        }
        unsafe { ui::set_text(h(app.game.status), &s) };
        return;
    };
    unsafe { KillTimer(app.hwnd, TIMER) };
    let j = app.game.job.take().expect("the job");
    let secs = j.t0.elapsed().as_secs_f64();
    update(app);
    app.update_play();
    if std::mem::take(&mut app.game.close_after) {
        // (Closing as asked: stopped, the partial archive removed; or done just before the stop.)
        let what = match &r {
            Ok(_) => "done just before the stop".to_string(),
            Err(e) if e.kind() == std::io::ErrorKind::Interrupted => "stopped".to_string(),
            Err(e) => format!("failed: {e}"),
        };
        crate::shaders::log(app, &format!("compress: {what} after {secs:.0} s; the launcher closes"));
        unsafe { windows_sys::Win32::UI::WindowsAndMessaging::PostMessageW(app.hwnd, windows_sys::Win32::UI::WindowsAndMessaging::WM_CLOSE, 0, 0) };
        return;
    }
    match r {
        Ok(w) => {
            crate::shaders::log(app, &format!("compress: done in {secs:.0} s: {} files, {} bytes read, an archive of {} bytes", w.files, w.bytes_in, w.bytes_out));
            let t = format!(
                "The archive is ready: {} ({}% of the game folder's {}).\n\nPlay from the archive from now on?\n\nThe game folder stays as it is. Delete it yourself to free its space. Keep it if you want to switch back.",
                gib(w.bytes_out),
                w.bytes_out * 100 / w.bytes_in.max(1),
                gib(w.bytes_in)
            );
            unsafe { ui::set_text(h(app.game.status), &app.game.text) };
            if unsafe { ui::message(app.hwnd, &t, crate::TITLE, MB_ICONQUESTION | MB_YESNO) } == IDYES {
                unsafe { ui::set_text(h(app.game.dir), &j.out.display().to_string()) };
                check_field(app);
            }
        }
        Err(e) if e.kind() == std::io::ErrorKind::Interrupted => {
            crate::shaders::log(app, &format!("compress: stopped after {secs:.0} s"));
            unsafe { ui::set_text(h(app.game.status), &format!("{}\n\nCompressing was stopped. No archive was kept.", app.game.text)) };
        }
        Err(e) => {
            crate::shaders::log(app, &format!("compress: failed after {secs:.0} s: {e}"));
            unsafe { ui::set_text(h(app.game.status), &app.game.text) };
            let t = format!("The archive could not be made ({e}).\n\nThe game folder is unchanged: {}", j.from.display());
            unsafe { ui::message(app.hwnd, &t, crate::TITLE, MB_ICONWARNING | MB_OK) };
        }
    }
}

pub fn command(app: &mut App, id: i32, code: u32) {
    match (id, code) {
        (ID_CHECK, 0) => check_field(app),
        (ID_BROWSE, 0) => {
            if let Some(p) = ui::pick_folder(app.hwnd, "The Demon's Souls game folder (PPSA01341, 01.007.000)") {
                unsafe { ui::set_text(h(app.game.dir), &p.display().to_string()) };
                check_field(app);
            }
        }
        (ID_ZAR, 0) => {
            if let Some(p) = ui::pick_file(app.hwnd, "The Demon's Souls game as a .zar archive (PPSA01341, 01.007.000)", ".zar archives", "*.zar", None) {
                unsafe { ui::set_text(h(app.game.dir), &p.display().to_string()) };
                check_field(app);
            }
        }
        (ID_COMPRESS, 0) => {
            if app.game.job.is_some() {
                stop_compress(app);
                tick(app);
            } else {
                start_compress(app);
            }
        }
        (ID_SAFE, 0) => {
            app.launch.safe_mode = safe_mode(app);
            app.save_launch();
            app.set_status(if app.launch.safe_mode { "The next start uses safe mode." } else { "" });
        }
        _ => {}
    }
}
