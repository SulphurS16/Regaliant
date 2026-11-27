//! Updates (docs/UPDATES.md; the user's decision of 2026-11-26): at every start the launcher asks
//! GitHub for the newest release, and a newer one shows the Update button beside Play. Nothing
//! downloads or changes until the player presses it. Update downloads the release's signature, then
//! its zip, checks the signature against the key built in here (`update-public-key.pem`, Ed25519;
//! its private half stays with the user), unpacks the zip's files beside the install and checks
//! that they are the offered version (the README's first line, inside the signed zip: a genuine
//! zip of another version offered under this release's name is refused). It then moves the
//! installed package files into a backup folder and the new ones in, journalled so that an
//! interrupted swap is undone at the next start, and starts the new launcher, which says when it
//! is up; if it does not come up, the earlier version is put back. Saves, settings, logs, prepared
//! shaders, add-ons and the player's own DLSS file are not package files and are never touched.
//! The backup goes after the new version's first game session that ends normally; until then Roll
//! back puts the earlier version back.
//!
//! The check is one HTTPS request (GitHub sees the address it comes from, as with any download);
//! the Game page's check box turns it off (`check_updates = 0` in launch.txt). A checkout's build
//! checks nothing. `DS_UPDATE_URL` (tests) names another release document on this PC's own address
//! (plain HTTP allowed there only).

use crate::{App, TITLE, VERSION, ui};
use dsconfig::paths::{GAME_EXE, LAUNCHER_EXE, Layout, PORTABLE_MARKER};
use std::io::{Seek, SeekFrom, Write};
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicU64, Ordering::Relaxed};
use std::sync::{Arc, Mutex};
use windows_sys::Win32::Foundation::HWND;
use windows_sys::Win32::UI::WindowsAndMessaging::*;

/// Posted when the start's check has its answer.
pub const WM_CHECKED: u32 = WM_APP + 7;
/// Posted when the started launcher said it is up (WPARAM 1) or did not come up (0).
pub const WM_STARTED: u32 = WM_APP + 8;
/// The download's progress and end, looked at twice a second.
pub const TIMER: usize = 11;
/// Messages that came while a dialog held the app, posted again.
pub const RETRY_TIMER: usize = 12;
pub const ID_UPDATE: i32 = 908;
pub const ID_CHECK_UPDATES: i32 = 909;

/// The newest release's document on GitHub.
const LATEST: &str = "https://api.github.com/repos/SulphurS16/Regaliant/releases/latest";
/// The public half of the key the release zips are signed with.
const PUBLIC_KEY_PEM: &str = include_str!("../update-public-key.pem");
/// The largest zip taken (a package is ~100 MB).
const MAX_ZIP: u64 = 1 << 30;
/// The update's folders in the install; each holds `MARKER`, and only such folders are removed.
const STAGE_PREFIX: &str = ".regaliant-update-";
const BACKUP_PREFIX: &str = ".regaliant-backup-";
const ROLLED_PREFIX: &str = ".regaliant-rolled-back-";
const MARKER: &str = ".regaliant-folder";
/// The swap's journal in the install: written before its first move, gone after its last.
const JOURNAL: &str = ".regaliant-update-journal.txt";
/// How long the old launcher waits for the new one to come up before putting itself back.
const START_WAIT_MS: u32 = 60_000;

/// A newer release on offer.
#[derive(Clone, Debug, PartialEq)]
pub struct Offer {
    pub version: String,
    zip_url: String,
    zip_size: u64,
    sig_url: String,
    /// The release notes' "## Changes" section, shown once after the update.
    changes: String,
}

#[derive(Default)]
pub struct Page {
    button: usize,
    check: usize,
    offer: Option<Offer>,
    job: Option<Job>,
    /// A version unpacked but not installed (something ran at that moment): Update installs it.
    staged: Option<Staged>,
    /// The earlier version a backup folder holds (what Roll back puts back).
    backup: Option<String>,
    /// An installed update whose launcher has not said it is up yet: what to put back otherwise.
    handoff: Option<Handoff>,
}

/// A download under way: its thread fetches, checks and unpacks; the timer shows it.
struct Job {
    version: String,
    changes: String,
    total: u64,
    done: Arc<AtomicU64>,
    /// 0 downloading, 1 checking the signature, 2 unpacking.
    step: Arc<AtomicU64>,
    /// The unpacked files' folder and their names, or why not.
    result: Arc<Mutex<Option<Result<(PathBuf, Vec<String>), String>>>>,
}

#[derive(Clone)]
struct Staged {
    version: String,
    changes: String,
    dir: PathBuf,
    names: Vec<String>,
}

/// The swap that installed `to` over `from`: the names it moved in.
struct Handoff {
    from: String,
    to: String,
    names: Vec<String>,
}

static CHECKED: Mutex<Option<Result<Option<Offer>, String>>> = Mutex::new(None);
/// Messages to post again (`RETRY_TIMER`).
static RETRY: Mutex<Vec<(u32, usize)>> = Mutex::new(Vec::new());

fn h(v: usize) -> HWND {
    v as HWND
}

fn log(app: &App, line: &str) {
    crate::shaders::log(app, &format!("update: {line}"));
}

/// A release version as numbers (`0.1.9`, `v0.1.9`); None for anything else (a pre-release's).
pub fn parse_version(s: &str) -> Option<(u32, u32, u32)> {
    let s = s.trim();
    let s = s.strip_prefix('v').unwrap_or(s);
    let mut it = s.split('.');
    let v = (it.next()?.parse().ok()?, it.next()?.parse().ok()?, it.next()?.parse().ok()?);
    it.next().is_none().then_some(v)
}

fn backup_dir(install: &Path, version: &str) -> PathBuf {
    install.join(format!("{BACKUP_PREFIX}{version}"))
}

fn staging_dir(install: &Path, version: &str) -> PathBuf {
    install.join(format!("{STAGE_PREFIX}{version}"))
}

fn rolled_dir(install: &Path, version: &str) -> PathBuf {
    install.join(format!("{ROLLED_PREFIX}{version}"))
}

fn download_dir(app: &App) -> PathBuf {
    app.paths.cache.join("update")
}

/// A folder of the update's own (made here, `MARKER` inside it).
fn ours(p: &Path) -> bool {
    let name = p.file_name().map(|n| n.to_string_lossy().into_owned()).unwrap_or_default();
    [STAGE_PREFIX, BACKUP_PREFIX, ROLLED_PREFIX].iter().any(|pre| name.starts_with(pre)) && p.join(MARKER).is_file()
}

/// Makes `dir` (empty) as a folder of the update's own.
fn make_ours(dir: &Path) -> Result<(), String> {
    if dir.exists() {
        if !ours(dir) {
            return Err(format!("{} is in the way", dir.display()));
        }
        std::fs::remove_dir_all(dir).map_err(|e| format!("{}: {e}", dir.display()))?;
    }
    std::fs::create_dir_all(dir).map_err(|e| format!("{}: {e}", dir.display()))?;
    std::fs::write(dir.join(MARKER), "Made by the Regaliant launcher's update. It removes this folder itself.\r\n").map_err(|e| format!("{}: {e}", dir.display()))
}

/// The update's folders in `install` (Uninstall removes them) and its journal, when there.
pub fn own_files(install: &Path) -> Vec<PathBuf> {
    let mut v: Vec<PathBuf> = std::fs::read_dir(install).into_iter().flatten().flatten().map(|e| e.path()).filter(|p| ours(p)).collect();
    let j = install.join(JOURNAL);
    if j.is_file() {
        v.push(j);
    }
    v
}

/// The package files this version knows (`scripts\package.ps1`'s allowlist: the launcher, the game
/// and what Uninstall removes by name; the portable marker is the player's).
fn package_names() -> Vec<String> {
    let mut v = vec![LAUNCHER_EXE.to_string()];
    v.extend(crate::uninstall::PACKAGE_FILES.iter().filter(|n| **n != PORTABLE_MARKER).map(|n| n.to_string()));
    v
}

/// A file name as an install's relative path (backslashes), or None when it is not one a package
/// may hold: absolute, with a drive or a stream (`:`), `..`, deeper than three folders, a part
/// Windows would change or read as something else (a trailing dot or space, `~` of a short name,
/// a device name), a place of the player's (the portable layout's marker and data, their own DLSS
/// file) or of the update's own.
fn valid_name(n: &str) -> Option<String> {
    let n = n.replace('/', "\\");
    let parts: Vec<&str> = n.split('\\').collect();
    if n.is_empty() || n.len() > 200 || parts.len() > 4 {
        return None;
    }
    for p in &parts {
        if p.is_empty() || p.ends_with('.') || p.ends_with(' ') || p.starts_with(' ') {
            return None;
        }
        if p.chars().any(|c| matches!(c, ':' | '~' | '<' | '>' | '"' | '|' | '?' | '*') || (c as u32) < 32) {
            return None;
        }
        let stem = p.split('.').next().unwrap_or("").to_ascii_uppercase();
        let numbered = (stem.starts_with("COM") || stem.starts_with("LPT")) && stem.len() == 4 && stem.as_bytes()[3].is_ascii_digit();
        if numbered || matches!(stem.as_str(), "CON" | "PRN" | "AUX" | "NUL" | "CONIN$" | "CONOUT$") {
            return None;
        }
    }
    let lower = n.to_ascii_lowercase();
    if lower == PORTABLE_MARKER || lower.starts_with("user\\") || lower == "upscalers\\nvngx_dlss.dll" || lower.starts_with(".regaliant-") {
        return None;
    }
    Some(n)
}

/// The Update button (beside Uninstall, shown only when it has something to do) and the Game
/// page's check box. A swap a crash interrupted is undone first.
pub fn build(app: &mut App) {
    let b = app.builder();
    unsafe {
        let r = (518, 548, 88, 36);
        let btn = b.button("Update", r, ID_UPDATE);
        app.update.button = app.put(usize::MAX, btn, r) as usize;
        ui::show(btn, false);
        let r = (16, 408, 748, 24);
        let c = b.check("Check for a new version of Regaliant at every start", r, ID_CHECK_UPDATES);
        app.update.check = app.put(0, c, r) as usize;
        ui::set_check(c, app.launch.check_updates);
    }
    if let Some(note) = recover(app) {
        app.set_status(&note);
    }
    app.update.backup = find_backup(app);
    refresh(app);
}

fn refresh(app: &App) {
    let btn = h(app.update.button);
    let (text, show) = match (&app.update.job, &app.update.offer, &app.update.backup) {
        (Some(_), _, _) | (None, Some(_), _) => ("Update", true),
        (None, None, Some(_)) => ("Roll back", true),
        _ => ("Update", false),
    };
    unsafe {
        ui::set_text(btn, text);
        ui::show(btn, show);
        ui::enable(btn, app.update.job.is_none() && app.update.handoff.is_none());
    }
}

/// The earlier version whose backup this install keeps: the newest whole one of this version or
/// older (the update's other folders are removed: older or incomplete backups, backups of later
/// versions left by a roll back, roll backs' and unfinished updates' folders). Nothing is removed
/// while a journal waits to be undone.
fn find_backup(app: &App) -> Option<String> {
    let cur = parse_version(VERSION)?;
    let install = &app.paths.exe_dir;
    if install.join(JOURNAL).exists() {
        return None;
    }
    let whole = |p: &Path| p.join(GAME_EXE).is_file() && p.join(LAUNCHER_EXE).is_file();
    let mut older: Vec<((u32, u32, u32), String, PathBuf)> = Vec::new();
    for p in own_files(install) {
        let name = p.file_name().map(|n| n.to_string_lossy().into_owned()).unwrap_or_default();
        let keep = name.strip_prefix(BACKUP_PREFIX).and_then(|v| parse_version(v).map(|n| (n, v.to_string()))).filter(|(n, _)| *n <= cur && whole(&p));
        match keep {
            Some((n, v)) => older.push((n, v, p)),
            None => {
                let _ = std::fs::remove_dir_all(&p);
            }
        }
    }
    older.sort();
    let best = older.pop();
    for (_, _, p) in older {
        let _ = std::fs::remove_dir_all(p);
    }
    best.map(|(_, v, _)| v)
}

/// The start's check, on a thread of its own (none in a checkout or when the player turned it
/// off; `DS_UPDATE_URL` only on this PC's own address).
pub fn start_check(app: &App) {
    let test = std::env::var("DS_UPDATE_URL").ok().filter(|u| !u.is_empty());
    if let Some(u) = &test {
        if !http::local(u) {
            log(app, "DS_UPDATE_URL is not on this PC: no check");
            return;
        }
    }
    if !app.launch.check_updates || (test.is_none() && app.paths.layout == Layout::Dev) {
        return;
    }
    let url = test.unwrap_or_else(|| LATEST.to_string());
    let hwnd = app.hwnd as usize;
    std::thread::spawn(move || {
        let r = check(&url);
        *CHECKED.lock().unwrap() = Some(r);
        unsafe { PostMessageW(hwnd as HWND, WM_CHECKED, 0, 0) };
    });
}

fn check(url: &str) -> Result<Option<Offer>, String> {
    let mut body = Vec::new();
    let status = http::get(url, 0, "application/vnd.github+json", &mut |_, b| {
        if body.len() + b.len() > 4 << 20 {
            return Err("the release document is too large".into());
        }
        body.extend_from_slice(b);
        Ok(())
    })?;
    if status != 200 {
        return Err(format!("GitHub answered {status}"));
    }
    let v: serde_json::Value = serde_json::from_slice(&body).map_err(|e| format!("the release document: {e}"))?;
    offer_from(&v, VERSION)
}

/// The release GitHub's document `v` describes, when it is newer than `current` and carries its
/// zip and the zip's signature (a release without one is not offered).
fn offer_from(v: &serde_json::Value, current: &str) -> Result<Option<Offer>, String> {
    if v["draft"].as_bool() == Some(true) || v["prerelease"].as_bool() == Some(true) {
        return Ok(None);
    }
    let tag = v["tag_name"].as_str().ok_or("the release has no tag")?;
    let (Some(new), Some(cur)) = (parse_version(tag), parse_version(current)) else { return Ok(None) };
    if new <= cur {
        return Ok(None);
    }
    let version = format!("{}.{}.{}", new.0, new.1, new.2);
    let assets = v["assets"].as_array().ok_or("the release has no files")?;
    let find = |name: &str| assets.iter().find(|a| a["name"].as_str() == Some(name));
    let zip_name = format!("Regaliant-{version}.zip");
    let (Some(zip), Some(sig)) = (find(&zip_name), find(&format!("{zip_name}.sig"))) else {
        return Err(format!("{version} is out without a signed {zip_name}"));
    };
    let url = |a: &serde_json::Value| a["browser_download_url"].as_str().map(str::to_string).ok_or_else(|| "a file without its address".to_string());
    let zip_size = zip["size"].as_u64().filter(|&s| s > 0 && s <= MAX_ZIP).ok_or("the zip's size")?;
    Ok(Some(Offer { version, zip_url: url(zip)?, zip_size, sig_url: url(sig)?, changes: changes_of(v["body"].as_str().unwrap_or("")) }))
}

/// The "## Changes" section of release notes, up to the next heading of its level or above.
fn changes_of(body: &str) -> String {
    let mut out = Vec::new();
    let mut inside = false;
    for line in body.lines() {
        let t = line.trim_end();
        if t.starts_with("## ") || t.starts_with("# ") {
            if inside {
                break;
            }
            inside = t.trim_start_matches('#').trim().eq_ignore_ascii_case("changes");
            continue;
        }
        if inside {
            out.push(t);
        }
    }
    out.join("\n").trim().to_string()
}

/// The update's messages (`WM_CHECKED`, `WM_STARTED`).
pub fn message(app: &mut App, msg: u32, wp: usize) {
    match msg {
        WM_CHECKED => checked(app),
        WM_STARTED => started(app, wp != 0),
        _ => {}
    }
}

/// `msg` came while a dialog held the app: posted again shortly.
pub fn retry_later(hwnd: HWND, msg: u32, wp: usize) {
    RETRY.lock().unwrap().push((msg, wp));
    unsafe { SetTimer(hwnd, RETRY_TIMER, 250, None) };
}

pub fn retry_now(hwnd: HWND) {
    unsafe { KillTimer(hwnd, RETRY_TIMER) };
    for (msg, wp) in std::mem::take(&mut *RETRY.lock().unwrap()) {
        unsafe { PostMessageW(hwnd, msg, wp, 0) };
    }
}

/// The start's answer arrived.
fn checked(app: &mut App) {
    let Some(r) = CHECKED.lock().unwrap().take() else { return };
    match r {
        Ok(Some(o)) => {
            log(app, &format!("{} is out (this is {VERSION})", o.version));
            if unsafe { ui::get_text(app.status) }.is_empty() {
                app.set_status(&format!("Regaliant {} is out. Update downloads it and restarts the launcher.", o.version));
            }
            app.update.offer = Some(o);
        }
        Ok(None) => log(app, &format!("{VERSION} is the newest")),
        Err(e) => log(app, &format!("no answer ({e})")),
    }
    refresh(app);
}

/// Why the program files cannot change now, if they cannot (the launcher's own short checks of the
/// game, in its job, do not count).
fn busy(app: &App) -> Option<&'static str> {
    let own = crate::sys::job_processes();
    if app.running.is_some() || crate::sys::processes_in(&app.paths.exe_dir).iter().any(|(pid, n)| n.eq_ignore_ascii_case(GAME_EXE) && !own.contains(pid)) {
        return Some("Close the game first.");
    }
    if crate::shaders::running(app) {
        return Some("The shaders are being prepared. Stop preparing them first.");
    }
    if crate::game::compressing(app) {
        return Some("The game folder is being compressed. Wait until it is done.");
    }
    None
}

/// Whether an update is being downloaded or installed (Play, the shader preparation and
/// Uninstall wait for it).
pub fn downloading(app: &App) -> bool {
    app.update.job.is_some() || app.update.handoff.is_some()
}

pub fn command(app: &mut App, id: i32) {
    match id {
        ID_CHECK_UPDATES => {
            app.launch.check_updates = unsafe { ui::checked(h(app.update.check)) };
            app.save_launch();
        }
        ID_UPDATE if app.update.handoff.is_some() => {}
        ID_UPDATE if app.update.offer.is_some() => update(app),
        ID_UPDATE if app.update.backup.is_some() => roll_back(app),
        _ => {}
    }
}

fn update(app: &mut App) {
    if app.update.job.is_some() {
        return;
    }
    if let Some(why) = busy(app) {
        app.set_status(why);
        return;
    }
    let Some(o) = app.update.offer.clone() else { return };
    // (Unpacked before, but not installed then: no second download.)
    if let Some(s) = app.update.staged.clone().filter(|s| s.version == o.version && s.dir.is_dir()) {
        install(app, &s);
        refresh(app);
        return;
    }
    let t = format!(
        "Download Regaliant {} ({} MB) and update?\n\nThe launcher restarts when it is done. Saves, settings and prepared shaders are kept.",
        o.version,
        (o.zip_size + (1 << 19)) >> 20
    );
    if unsafe { ui::message(app.hwnd, &t, TITLE, MB_ICONQUESTION | MB_YESNO) } != IDYES {
        return;
    }
    let job = Job { version: o.version.clone(), changes: o.changes.clone(), total: o.zip_size, done: Arc::default(), step: Arc::default(), result: Arc::default() };
    let (done, step, result) = (job.done.clone(), job.step.clone(), job.result.clone());
    let (dir, staging) = (download_dir(app), staging_dir(&app.paths.exe_dir, &o.version));
    log(app, &format!("downloading {}", o.version));
    let status = format!("Downloading Regaliant {}.", o.version);
    std::thread::spawn(move || {
        let r = fetch(&o, &dir, &staging, &done, &step);
        *result.lock().unwrap() = Some(r);
    });
    app.update.job = Some(job);
    unsafe { SetTimer(app.hwnd, TIMER, 500, None) };
    refresh(app);
    app.set_status(&status);
}

/// The timer: the download's progress, then its end.
pub fn tick(app: &mut App) {
    let Some(job) = &app.update.job else {
        unsafe { KillTimer(app.hwnd, TIMER) };
        return;
    };
    let r = job.result.lock().unwrap().take();
    let Some(r) = r else {
        let status = match job.step.load(Relaxed) {
            0 => format!("Downloading Regaliant {}: {} of {} MB.", job.version, job.done.load(Relaxed) >> 20, (job.total + (1 << 19)) >> 20),
            1 => format!("Checking the download of Regaliant {}.", job.version),
            _ => format!("Unpacking Regaliant {}.", job.version),
        };
        app.set_status(&status);
        return;
    };
    unsafe { KillTimer(app.hwnd, TIMER) };
    let job = app.update.job.take().unwrap();
    match r {
        Ok((dir, names)) => {
            let s = Staged { version: job.version, changes: job.changes, dir, names };
            app.update.staged = Some(s.clone());
            install(app, &s);
        }
        Err(e) => {
            log(app, &format!("{} not installed: {e}", job.version));
            app.set_status(&format!("The update did not complete: {e}. Regaliant {VERSION} stays."));
        }
    }
    refresh(app);
}

/// The download thread: the signature, then the zip (resumed where an earlier try stopped), the
/// signature checked, the files unpacked into `staging` and checked to be the offered version.
fn fetch(o: &Offer, dir: &Path, staging: &Path, done: &AtomicU64, step: &AtomicU64) -> Result<(PathBuf, Vec<String>), String> {
    std::fs::create_dir_all(dir).map_err(|e| format!("the download folder: {e}"))?;
    let zip = dir.join(format!("Regaliant-{}.zip", o.version));
    let part = zip.with_extension("zip.part");
    // (Earlier versions' leftovers go.)
    for e in std::fs::read_dir(dir).into_iter().flatten().flatten() {
        if e.path() != part && e.file_type().is_ok_and(|t| t.is_file()) {
            let _ = std::fs::remove_file(e.path());
        }
    }
    let mut sig = Vec::new();
    let status = http::get(&o.sig_url, 0, "application/octet-stream", &mut |_, b| {
        if sig.len() + b.len() > 4096 {
            return Err("the signature is too large".into());
        }
        sig.extend_from_slice(b);
        Ok(())
    })?;
    if status != 200 || sig.len() != 64 {
        return Err(format!("the signature's download answered {status} with {} bytes", sig.len()));
    }
    download(&o.zip_url, &part, o.zip_size, done)?;
    step.store(1, Relaxed);
    // (Read whole, then the file goes: a checked, refused or broken download is never kept.)
    let bytes = std::fs::read(&part).map_err(|e| format!("the download: {e}"))?;
    let _ = std::fs::remove_file(&part);
    verify(&bytes, &sig)?;
    step.store(2, Relaxed);
    let names = unpack(&bytes, staging).and_then(|names| {
        let want = parse_version(&o.version);
        match readme_version(staging) {
            Some(v) if Some(v) == want && v > parse_version(VERSION).unwrap_or((0, 0, 0)) => Ok(names),
            Some(v) => Err(format!("the download holds Regaliant {}.{}.{}, not {}", v.0, v.1, v.2, o.version)),
            None => Err("the download does not name its version".into()),
        }
    });
    names.inspect_err(|_| {
        let _ = std::fs::remove_dir_all(staging);
    })
    .map(|n| (staging.to_path_buf(), n))
}

/// The version a package's README names on its first line ("Regaliant X.Y.Z - ...").
fn readme_version(dir: &Path) -> Option<(u32, u32, u32)> {
    let t = std::fs::read_to_string(dir.join("README.txt")).ok()?;
    let first = t.trim_start_matches('\u{feff}').lines().next()?;
    parse_version(first.strip_prefix("Regaliant ")?.split_whitespace().next()?)
}

/// Downloads `url` (`size` bytes) into `part`, going on from what an earlier try left there when
/// the server sends that range (and starting again when it sends the whole).
pub(crate) fn download(url: &str, part: &Path, size: u64, done: &AtomicU64) -> Result<(), String> {
    let io = |e: std::io::Error| format!("the download: {e}");
    for _ in 0..2 {
        let mut f = std::fs::OpenOptions::new().create(true).write(true).truncate(false).open(part).map_err(io)?;
        let mut have = f.seek(SeekFrom::End(0)).map_err(io)?;
        if have >= size {
            f.set_len(0).map_err(io)?;
            have = f.seek(SeekFrom::Start(0)).map_err(io)?;
        }
        let from = have;
        let mut first = true;
        let status = http::get(url, from, "application/octet-stream", &mut |a, b| {
            if first {
                first = false;
                if a.status == 200 && have > 0 {
                    f.set_len(0).map_err(io)?;
                    have = f.seek(SeekFrom::Start(0)).map_err(io)?;
                }
                if a.status == 206 && a.range_start != Some(from) {
                    return Err(format!("the server sent another part of the file ({:?} for {from})", a.range_start));
                }
            }
            if a.status != 200 && a.status != 206 {
                return Ok(());
            }
            have += b.len() as u64;
            if have > size {
                return Err("the download is larger than announced".into());
            }
            f.write_all(b).map_err(io)?;
            done.store(have, Relaxed);
            Ok(())
        })?;
        f.sync_all().map_err(io)?;
        drop(f);
        match status {
            200 | 206 if have == size => return Ok(()),
            200 | 206 => return Err(format!("the download stopped at {} of {} MB", have >> 20, size >> 20)),
            // (What is there no longer fits the file on the server: once more from the start.)
            416 => {
                let _ = std::fs::remove_file(part);
            }
            s => return Err(format!("the download answered {s}")),
        }
    }
    Err("the download could not be resumed".into())
}

/// The public key's 32 bytes (the PEM's SubjectPublicKeyInfo: a fixed 12-byte prefix, then the
/// key).
fn public_key() -> Result<[u8; 32], String> {
    let b64: String = PUBLIC_KEY_PEM.lines().filter(|l| !l.starts_with("-----")).collect();
    let der = base64(&b64).ok_or("the built-in key")?;
    const PREFIX: [u8; 12] = [0x30, 0x2a, 0x30, 0x05, 0x06, 0x03, 0x2b, 0x65, 0x70, 0x03, 0x21, 0x00];
    if der.len() != 44 || der[..12] != PREFIX {
        return Err("the built-in key".into());
    }
    Ok(der[12..].try_into().unwrap())
}

fn base64(s: &str) -> Option<Vec<u8>> {
    let mut out = Vec::new();
    let (mut acc, mut bits) = (0u32, 0);
    for c in s.bytes() {
        let v = match c {
            b'A'..=b'Z' => c - b'A',
            b'a'..=b'z' => c - b'a' + 26,
            b'0'..=b'9' => c - b'0' + 52,
            b'+' => 62,
            b'/' => 63,
            b'=' => break,
            b'\r' | b'\n' | b' ' => continue,
            _ => return None,
        } as u32;
        acc = acc << 6 | v;
        bits += 6;
        if bits >= 8 {
            bits -= 8;
            out.push((acc >> bits) as u8);
            acc &= (1 << bits) - 1;
        }
    }
    Some(out)
}

/// Whether `sig` is the release key's Ed25519 signature of `data`.
fn verify(data: &[u8], sig: &[u8]) -> Result<(), String> {
    verify_with(&public_key()?, data, sig)
}

fn verify_with(key: &[u8; 32], data: &[u8], sig: &[u8]) -> Result<(), String> {
    let pk = ed25519_compact::PublicKey::from_slice(key).map_err(|_| "the built-in key".to_string())?;
    let sig = ed25519_compact::Signature::from_slice(sig).map_err(|_| "the signature is not one".to_string())?;
    pk.verify(data, &sig).map_err(|_| "the download's signature does not match: it is not a Regaliant release".to_string())
}

/// The zip's files into `staging` (made anew as a folder of the update's own), each written to
/// the disk before it counts: their names, checked by `valid_name`; the launcher and the game
/// must be among them.
fn unpack(z: &[u8], staging: &Path) -> Result<Vec<String>, String> {
    let entries = zip::entries(z)?;
    make_ours(staging)?;
    let mut names: Vec<String> = Vec::new();
    for e in entries.iter().filter(|e| !e.name.ends_with('/')) {
        let name = valid_name(&e.name).ok_or_else(|| format!("the download holds a file it may not ({})", e.name))?;
        if names.iter().any(|n| n.eq_ignore_ascii_case(&name)) {
            return Err(format!("the download holds {} twice", e.name));
        }
        let data = zip::read(z, e)?;
        let out = staging.join(&name);
        if let Some(p) = out.parent() {
            std::fs::create_dir_all(p).map_err(|e| format!("the update's folder: {e}"))?;
        }
        let mut f = std::fs::File::create(&out).map_err(|e| format!("{name}: {e}"))?;
        f.write_all(&data).and_then(|_| f.sync_all()).map_err(|e| format!("{name}: {e}"))?;
        names.push(name);
    }
    for need in [GAME_EXE, LAUNCHER_EXE] {
        if !names.iter().any(|n| n.eq_ignore_ascii_case(need)) {
            return Err(format!("the download lacks {need}"));
        }
    }
    Ok(names)
}

/// A rename, tried again for a few seconds while another program holds the file (a virus
/// scanner reading a new program file).
fn rename_retry(from: &Path, to: &Path) -> std::io::Result<()> {
    let mut tries = 0;
    loop {
        match std::fs::rename(from, to) {
            Ok(()) => return Ok(()),
            // (ERROR_ACCESS_DENIED, ERROR_SHARING_VIOLATION, ERROR_LOCK_VIOLATION.)
            Err(e) if matches!(e.raw_os_error(), Some(5 | 32 | 33)) && tries < 20 => {
                tries += 1;
                std::thread::sleep(std::time::Duration::from_millis(250));
            }
            Err(e) => return Err(e),
        }
    }
}

fn mv(from: &Path, to: &Path) -> Result<(), String> {
    if let Some(p) = to.parent() {
        std::fs::create_dir_all(p).map_err(|e| format!("{}: {e}", p.display()))?;
    }
    rename_retry(from, to).map_err(|e| format!("{}: {e}", from.display()))
}

fn write_synced(p: &Path, text: &str) -> Result<(), String> {
    let mut f = std::fs::File::create(p).map_err(|e| format!("{}: {e}", p.display()))?;
    f.write_all(text.as_bytes()).and_then(|_| f.sync_all()).map_err(|e| format!("{}: {e}", p.display()))
}

/// Moves each installed file named in `out_names` into `backup` (made anew), then each of
/// `in_names` from `incoming` into the install: the launcher last out and first in, so the install
/// is without one for two renames only. A journal in the install names the moves before the first
/// (`recover` undoes them at the next start after a crash) and goes after the last. On a failure
/// everything moved goes back; what could not go back is in the error, the journal kept for the
/// next start.
fn swap(install: &Path, incoming: &Path, in_names: &[String], out_names: &[String], backup: &Path) -> Result<(), String> {
    let is_l = |n: &&String| n.eq_ignore_ascii_case(LAUNCHER_EXE);
    let outs: Vec<&String> = out_names.iter().filter(|n| !is_l(n)).chain(out_names.iter().filter(is_l)).collect();
    let ins: Vec<&String> = in_names.iter().filter(is_l).chain(in_names.iter().filter(|n| !is_l(n))).collect();
    make_ours(backup)?;
    let file = |p: &Path| p.file_name().map(|n| n.to_string_lossy().into_owned()).unwrap_or_default();
    let mut j = format!("regaliant update journal\nincoming = {}\nbackup = {}\n", file(incoming), file(backup));
    for n in &ins {
        j += &format!("in = {n}\n");
    }
    for n in &outs {
        j += &format!("out = {n}\n");
    }
    let journal = install.join(JOURNAL);
    write_synced(&journal, &j)?;
    let mut moved_out: Vec<&String> = Vec::new();
    let mut moved_in: Vec<&String> = Vec::new();
    let mut r = Ok(());
    for n in &outs {
        if install.join(n).is_file() {
            if let Err(e) = mv(&install.join(n), &backup.join(n)) {
                r = Err(e);
                break;
            }
            moved_out.push(n);
        }
    }
    if r.is_ok() {
        for n in &ins {
            if let Err(e) = mv(&incoming.join(n), &install.join(n)) {
                r = Err(e);
                break;
            }
            moved_in.push(n);
        }
    }
    let Err(e) = r else {
        let _ = std::fs::remove_file(&journal);
        return Ok(());
    };
    let mut left = Vec::new();
    for n in moved_in.iter().rev() {
        if let Err(e) = mv(&install.join(n), &incoming.join(n)) {
            left.push(e);
        }
    }
    for n in moved_out.iter().rev() {
        if let Err(e) = mv(&backup.join(n), &install.join(n)) {
            left.push(e);
        }
    }
    if !left.is_empty() {
        return Err(format!("{e}, and some files could not be put back ({}): the launcher tries again at its next start", left.join("; ")));
    }
    let _ = std::fs::remove_file(&journal);
    let _ = std::fs::remove_dir_all(backup);
    Err(e)
}

/// At a start: a swap a crash interrupted (its journal still there) is undone (`recover_at`).
fn recover(app: &App) -> Option<String> {
    let note = recover_at(&app.paths)?;
    log(app, &format!("an interrupted swap found: {note}"));
    Some(note)
}

/// The files an interrupted swap moved in go back to where they came from, the ones it moved out
/// come back; the journal goes once all are back. A note for the status line when there was one.
fn recover_at(paths: &dsconfig::paths::Paths) -> Option<String> {
    let install = &paths.exe_dir;
    let journal = install.join(JOURNAL);
    let t = std::fs::read_to_string(&journal).ok()?;
    let mut kv = t.lines().filter_map(|l| l.split_once(" = ")).map(|(k, v)| (k.trim(), v.trim()));
    let mut incoming = None;
    let mut backup = None;
    let (mut ins, mut outs) = (Vec::new(), Vec::new());
    for (k, v) in kv.by_ref() {
        match k {
            "incoming" => incoming = Some(v.to_string()),
            "backup" => backup = Some(v.to_string()),
            "in" => ins.extend(valid_name(v)),
            "out" => outs.extend(valid_name(v)),
            _ => {}
        }
    }
    let dir = |v: Option<String>| v.filter(|n| n.starts_with(".regaliant-") && !n.contains(['\\', '/', ':'])).map(|n| install.join(n));
    let (Some(incoming), Some(backup)) = (dir(incoming), dir(backup)) else {
        let _ = std::fs::remove_file(&journal);
        return None;
    };
    let mut left = Vec::new();
    for n in ins.iter().rev() {
        if install.join(n).is_file() && !incoming.join(n).exists() {
            if let Err(e) = mv(&install.join(n), &incoming.join(n)) {
                left.push(e);
            }
        }
    }
    for n in outs.iter().rev() {
        if backup.join(n).is_file() && !install.join(n).exists() {
            if let Err(e) = mv(&backup.join(n), &install.join(n)) {
                left.push(e);
            }
        }
    }
    if !left.is_empty() {
        return Some(format!("An update was interrupted and could not be undone yet ({}). Close other programs and start the launcher again.", left.join("; ")));
    }
    let _ = std::fs::remove_file(&journal);
    if files_under(&backup).is_empty() {
        let _ = std::fs::remove_dir_all(&backup);
    }
    Some("An update was interrupted and has been undone. Start the launcher again before you play.".into())
}

/// The files under `dir` by their relative names (a backup's), links and junctions left out.
fn files_under(dir: &Path) -> Vec<String> {
    let mut out = Vec::new();
    let mut stack = vec![dir.to_path_buf()];
    while let Some(d) = stack.pop() {
        for e in std::fs::read_dir(&d).into_iter().flatten().flatten() {
            let Ok(t) = e.file_type() else { continue };
            if t.is_symlink() {
                continue;
            }
            if t.is_dir() {
                stack.push(e.path());
            } else if let Ok(rel) = e.path().strip_prefix(dir) {
                let rel = rel.to_string_lossy().into_owned();
                if rel != MARKER {
                    out.push(rel);
                }
            }
        }
    }
    out
}

/// The unpacked version goes in: the installed package files into `.regaliant-backup-<this
/// version>`, the new ones in their place, the new launcher started.
fn install(app: &mut App, s: &Staged) {
    // (The launcher's own check of the game first: it runs the game's program for a moment.)
    crate::shaders::wait_check();
    if let Some(why) = busy(app) {
        log(app, &format!("{} unpacked, not installed: {why}", s.version));
        app.set_status(&format!("{why} Then press Update again."));
        return;
    }
    let mut out = package_names();
    for n in &s.names {
        if !out.iter().any(|o| o.eq_ignore_ascii_case(n)) {
            out.push(n.clone());
        }
    }
    let install = app.paths.exe_dir.clone();
    let backup = backup_dir(&install, VERSION);
    if let Err(e) = swap(&install, &s.dir, &s.names, &out, &backup) {
        log(app, &format!("{} not installed: {e}", s.version));
        app.set_status(&format!("The update could not be installed: {e}. Regaliant {VERSION} stays."));
        return;
    }
    app.update.staged = None;
    let _ = std::fs::remove_dir_all(&s.dir);
    log(app, &format!("{} installed, {VERSION} kept in {}", s.version, backup.file_name().unwrap_or_default().to_string_lossy()));
    let dir = download_dir(app);
    let _ = std::fs::create_dir_all(&dir);
    let _ = std::fs::write(dir.join("changes.txt"), format!("{}\n{}\n", s.version, s.changes));
    app.update.handoff = Some(Handoff { from: VERSION.to_string(), to: s.version.clone(), names: s.names.clone() });
    hand_over(app);
}

/// Starts the installed launcher and waits on a thread of its own for it to say it is up (a named
/// event it sets once its window shows); `WM_STARTED` says how that went.
fn hand_over(app: &mut App) {
    use std::os::windows::io::AsRawHandle;
    use windows_sys::Win32::Foundation::{CloseHandle, HANDLE, WAIT_OBJECT_0};
    use windows_sys::Win32::System::Threading::{CreateEventW, WaitForMultipleObjects};
    let name = ui::wide(&ready_event_name(std::process::id()));
    let ev = unsafe { CreateEventW(std::ptr::null(), 1, 0, name.as_ptr()) };
    let exe = app.paths.exe_dir.join(LAUNCHER_EXE);
    let child = spawn_quiet(std::process::Command::new(&exe).arg("--after-update").arg(std::process::id().to_string()).current_dir(&app.paths.exe_dir));
    let hwnd = app.hwnd as usize;
    match child {
        Ok(child) if !ev.is_null() => {
            unsafe { ShowWindow(app.hwnd, SW_HIDE) };
            let ev = ev as usize;
            std::thread::spawn(move || {
                let handles: [HANDLE; 2] = [ev as HANDLE, child.as_raw_handle() as HANDLE];
                let r = unsafe { WaitForMultipleObjects(2, handles.as_ptr(), 0, START_WAIT_MS) };
                unsafe { CloseHandle(ev as HANDLE) };
                // (Not up in time: ended before the earlier version's files go back under it,
                // which it could be reading or moving meanwhile; the review of 2026-11-27.)
                let mut child = child;
                if r != WAIT_OBJECT_0 {
                    let _ = child.kill();
                    let _ = child.wait();
                }
                drop(child);
                unsafe { PostMessageW(hwnd as HWND, WM_STARTED, (r == WAIT_OBJECT_0) as usize, 0) };
            });
        }
        other => {
            if !ev.is_null() {
                unsafe { CloseHandle(ev) };
            }
            log(app, &format!("the new launcher did not start: {:?}", other.err()));
            unsafe { PostMessageW(hwnd as HWND, WM_STARTED, 0, 0) };
        }
    }
}

/// Starts `cmd` with Windows' own error boxes off for this call (a program file that is not one
/// then fails here, where a system box would have held the window), the program itself given
/// Windows' default error mode.
fn spawn_quiet(cmd: &mut std::process::Command) -> std::io::Result<std::process::Child> {
    use std::os::windows::process::CommandExt;
    use windows_sys::Win32::System::Diagnostics::Debug::{SEM_FAILCRITICALERRORS, SEM_NOGPFAULTERRORBOX, SEM_NOOPENFILEERRORBOX, SetThreadErrorMode};
    const CREATE_DEFAULT_ERROR_MODE: u32 = 0x0400_0000;
    let mut old = 0;
    unsafe { SetThreadErrorMode(SEM_FAILCRITICALERRORS | SEM_NOGPFAULTERRORBOX | SEM_NOOPENFILEERRORBOX, &mut old) };
    let r = cmd.creation_flags(CREATE_DEFAULT_ERROR_MODE).spawn();
    unsafe { SetThreadErrorMode(old, std::ptr::null_mut()) };
    r
}

fn ready_event_name(pid: u32) -> String {
    format!("Local\\RegaliantUpdateReady-{pid}")
}

/// A launcher started by an update (`--after-update <pid>`), its window shown: tells the one that
/// started it.
pub fn signal_ready(pid: u32) {
    use windows_sys::Win32::Foundation::CloseHandle;
    use windows_sys::Win32::System::Threading::{EVENT_MODIFY_STATE, OpenEventW, SetEvent};
    let name = ui::wide(&ready_event_name(pid));
    unsafe {
        let ev = OpenEventW(EVENT_MODIFY_STATE, 0, name.as_ptr());
        if !ev.is_null() {
            SetEvent(ev);
            CloseHandle(ev);
        }
    }
}

/// The new launcher is up (this one closes), or it did not come up: the earlier version back.
fn started(app: &mut App, ok: bool) {
    let Some(h) = app.update.handoff.take() else { return };
    if ok {
        log(app, &format!("{} is up", h.to));
        unsafe { PostMessageW(app.hwnd, WM_CLOSE, 0, 0) };
        return;
    }
    let install = app.paths.exe_dir.clone();
    let from = backup_dir(&install, &h.from);
    let names = files_under(&from);
    let mut out = package_names();
    for n in h.names.iter().chain(names.iter()) {
        if !out.iter().any(|o| o.eq_ignore_ascii_case(n)) {
            out.push(n.clone());
        }
    }
    let r = swap(&install, &from, &names, &out, &rolled_dir(&install, &h.to));
    unsafe {
        ShowWindow(app.hwnd, SW_SHOW);
        SetForegroundWindow(app.hwnd);
    }
    // (The note of what the new version changed is not for this one.)
    let _ = std::fs::remove_file(download_dir(app).join("changes.txt"));
    let text = match r {
        Ok(()) => {
            let _ = std::fs::remove_dir_all(&from);
            log(app, &format!("{} did not come up: {} put back", h.to, h.from));
            format!("Regaliant {} did not start, so Regaliant {} is back. Please report this.", h.to, h.from)
        }
        Err(e) => {
            log(app, &format!("{} did not come up, and {} could not be put back: {e}", h.to, h.from));
            format!("Regaliant {} did not start, and Regaliant {} could not be put back: {e}", h.to, h.from)
        }
    };
    app.update.backup = find_backup(app);
    app.set_status(&text);
    unsafe { ui::message(app.hwnd, &text, TITLE, MB_ICONWARNING | MB_OK) };
    refresh(app);
}

/// Roll back: the backup's version goes back in, the current one into
/// `.regaliant-rolled-back-<version>` (removed at the next start). The earlier launcher may have no
/// update code of its own, so this one's window goes before it starts (one launcher at a time).
fn roll_back(app: &mut App) {
    let Some(old) = app.update.backup.clone() else { return };
    if let Some(why) = busy(app) {
        app.set_status(why);
        return;
    }
    let t = format!("Go back to Regaliant {old}?\n\nThe launcher restarts. Saves, settings and prepared shaders are kept.");
    if unsafe { ui::message(app.hwnd, &t, TITLE, MB_ICONQUESTION | MB_YESNO) } != IDYES {
        return;
    }
    let install = app.paths.exe_dir.clone();
    let from = backup_dir(&install, &old);
    let names = files_under(&from);
    let valid = names.iter().all(|n| valid_name(n).as_deref() == Some(n.as_str()));
    if !valid || !names.iter().any(|n| n.eq_ignore_ascii_case(LAUNCHER_EXE)) || !names.iter().any(|n| n.eq_ignore_ascii_case(GAME_EXE)) {
        app.set_status(&format!("The backup of Regaliant {old} is not complete. It cannot be put back."));
        return;
    }
    let mut out = package_names();
    for n in &names {
        if !out.iter().any(|o| o.eq_ignore_ascii_case(n)) {
            out.push(n.clone());
        }
    }
    if let Err(e) = swap(&install, &from, &names, &out, &rolled_dir(&install, VERSION)) {
        log(app, &format!("roll back to {old} failed: {e}"));
        app.set_status(&format!("Regaliant {old} could not be put back: {e}."));
        return;
    }
    let _ = std::fs::remove_dir_all(&from);
    log(app, &format!("rolled back to {old}"));
    let exe = install.join(LAUNCHER_EXE);
    unsafe { DestroyWindow(app.hwnd) };
    if spawn_quiet(std::process::Command::new(&exe).current_dir(&install)).is_err() {
        let w = ui::wide(&format!("Regaliant {old} is back. Start the launcher again."));
        let t = ui::wide(TITLE);
        unsafe { MessageBoxW(std::ptr::null_mut(), w.as_ptr(), t.as_ptr(), MB_ICONINFORMATION | MB_OK) };
    }
}

/// At a start after an update: what changed, once; the update's leftovers removed.
pub fn after_start(app: &mut App) {
    let notes = download_dir(app).join("changes.txt");
    let Ok(t) = std::fs::read_to_string(&notes) else { return };
    let _ = std::fs::remove_file(&notes);
    let (version, changes) = t.split_once('\n').unwrap_or((t.as_str(), ""));
    if version.trim() != VERSION {
        return;
    }
    log(app, &format!("now {VERSION}"));
    app.set_status(&format!("Updated to Regaliant {VERSION}."));
    let changes = changes_text(changes);
    if !changes.is_empty() {
        let text = format!("Regaliant {VERSION} is installed.\n\nWhat changed:\n{changes}");
        unsafe { ui::message(app.hwnd, &text, TITLE, MB_ICONINFORMATION | MB_OK) };
    }
}

/// The Changes section for a message box: each "- " item one paragraph with a bullet (the
/// changelog's wrapped lines joined), blank lines between items left out.
fn changes_text(changes: &str) -> String {
    let mut items: Vec<String> = Vec::new();
    for line in changes.lines() {
        let t = line.trim();
        if t.is_empty() {
            continue;
        }
        match t.strip_prefix("- ") {
            Some(rest) => items.push(format!("\u{2022} {rest}")),
            None => match items.last_mut() {
                Some(last) => {
                    last.push(' ');
                    last.push_str(t);
                }
                None => items.push(t.to_string()),
            },
        }
    }
    items.join("\n")
}

/// A game session ended normally: the earlier version's backup goes.
pub fn session_ok(app: &mut App) {
    if let Some(old) = app.update.backup.take() {
        let _ = std::fs::remove_dir_all(backup_dir(&app.paths.exe_dir, &old));
        log(app, &format!("the backup of {old} removed after a normal session"));
        refresh(app);
    }
}

/// HTTPS requests through Windows' WinHTTP (no library of our own).
mod http {
    use crate::ui::wide;
    use windows_sys::Win32::Networking::WinHttp::*;

    /// The answer's status, and for a range the first byte it holds.
    pub struct Answer {
        pub status: u32,
        pub range_start: Option<u64>,
    }

    struct H(*mut core::ffi::c_void);
    impl Drop for H {
        fn drop(&mut self) {
            if !self.0.is_null() {
                unsafe { WinHttpCloseHandle(self.0) };
            }
        }
    }

    fn failed(what: &str) -> String {
        format!("{what} failed ({})", std::io::Error::last_os_error())
    }

    /// Whether `url` names this PC itself (tests only).
    pub fn local(url: &str) -> bool {
        let rest = url.strip_prefix("http://").or_else(|| url.strip_prefix("https://")).unwrap_or("");
        let host = rest.split(['/', '?']).next().unwrap_or("");
        let host = host.rsplit_once(':').filter(|(h, p)| !h.is_empty() && p.chars().all(|c| c.is_ascii_digit())).map_or(host, |(h, _)| h);
        matches!(host.to_ascii_lowercase().as_str(), "127.0.0.1" | "localhost" | "[::1]")
    }

    /// GET `url` (https, or http to this PC's own address for tests), from byte `from` on (a
    /// range) when not 0: its status code; its body goes to `sink` in pieces, with the answer.
    pub fn get(url: &str, from: u64, accept: &str, sink: &mut dyn FnMut(&Answer, &[u8]) -> Result<(), String>) -> Result<u32, String> {
        let w = wide(url);
        let mut host = vec![0u16; 256];
        let mut path = vec![0u16; 4096];
        let mut extra = vec![0u16; 4096];
        let mut uc = URL_COMPONENTS {
            dwStructSize: std::mem::size_of::<URL_COMPONENTS>() as u32,
            lpszHostName: host.as_mut_ptr(),
            dwHostNameLength: host.len() as u32,
            lpszUrlPath: path.as_mut_ptr(),
            dwUrlPathLength: path.len() as u32,
            lpszExtraInfo: extra.as_mut_ptr(),
            dwExtraInfoLength: extra.len() as u32,
            ..Default::default()
        };
        if unsafe { WinHttpCrackUrl(w.as_ptr(), 0, 0, &mut uc) } == 0 {
            return Err(format!("not an address: {url}"));
        }
        let host_s = String::from_utf16_lossy(&host[..uc.dwHostNameLength as usize]);
        let secure = uc.nScheme == WINHTTP_INTERNET_SCHEME_HTTPS;
        let local = matches!(host_s.to_ascii_lowercase().as_str(), "127.0.0.1" | "localhost" | "::1" | "[::1]");
        if !secure && !(uc.nScheme == WINHTTP_INTERNET_SCHEME_HTTP && local) {
            return Err(format!("not a secure address: {url}"));
        }
        let mut object: Vec<u16> = path[..uc.dwUrlPathLength as usize].to_vec();
        object.extend_from_slice(&extra[..uc.dwExtraInfoLength as usize]);
        object.push(0);
        let host_w = wide(&host_s);
        unsafe {
            let agent = wide(&format!("Regaliant-launcher/{}", crate::VERSION));
            let s = H(WinHttpOpen(agent.as_ptr(), WINHTTP_ACCESS_TYPE_AUTOMATIC_PROXY, std::ptr::null(), std::ptr::null(), 0));
            if s.0.is_null() {
                return Err(failed("WinHttpOpen"));
            }
            WinHttpSetTimeouts(s.0, 15_000, 15_000, 30_000, 60_000);
            let c = H(WinHttpConnect(s.0, host_w.as_ptr(), uc.nPort, 0));
            if c.0.is_null() {
                return Err(failed("connecting"));
            }
            let verb = wide("GET");
            let r = H(WinHttpOpenRequest(c.0, verb.as_ptr(), object.as_ptr(), std::ptr::null(), std::ptr::null(), std::ptr::null(), if secure { WINHTTP_FLAG_SECURE } else { 0 }));
            if r.0.is_null() {
                return Err(failed("the request"));
            }
            if !secure {
                // (A test server's answer is not followed anywhere else.)
                let never = WINHTTP_OPTION_REDIRECT_POLICY_NEVER;
                WinHttpSetOption(r.0, WINHTTP_OPTION_REDIRECT_POLICY, &never as *const u32 as *const _, 4);
            }
            let mut headers = format!("Accept: {accept}\r\n");
            if from > 0 {
                headers += &format!("Range: bytes={from}-\r\n");
            }
            let hw = wide(&headers);
            WinHttpAddRequestHeaders(r.0, hw.as_ptr(), u32::MAX, WINHTTP_ADDREQ_FLAG_ADD);
            if WinHttpSendRequest(r.0, std::ptr::null(), 0, std::ptr::null(), 0, 0, 0) == 0 {
                return Err(failed("sending the request"));
            }
            if WinHttpReceiveResponse(r.0, std::ptr::null_mut()) == 0 {
                return Err(failed("the answer"));
            }
            let mut status = 0u32;
            let mut len = 4u32;
            if WinHttpQueryHeaders(r.0, WINHTTP_QUERY_STATUS_CODE | WINHTTP_QUERY_FLAG_NUMBER, std::ptr::null(), &mut status as *mut u32 as *mut _, &mut len, std::ptr::null_mut()) == 0 {
                return Err(failed("the answer's status"));
            }
            let mut range_start = None;
            if status == 206 {
                let mut buf = vec![0u16; 128];
                let mut len = (buf.len() * 2) as u32;
                if WinHttpQueryHeaders(r.0, WINHTTP_QUERY_CONTENT_RANGE, std::ptr::null(), buf.as_mut_ptr() as *mut _, &mut len, std::ptr::null_mut()) != 0 {
                    range_start = content_range_start(&String::from_utf16_lossy(&buf[..(len / 2) as usize]));
                }
            }
            let answer = Answer { status, range_start };
            let mut buf = vec![0u8; 1 << 16];
            loop {
                let mut n = 0u32;
                if WinHttpReadData(r.0, buf.as_mut_ptr() as *mut _, buf.len() as u32, &mut n) == 0 {
                    return Err(failed("reading the answer"));
                }
                if n == 0 {
                    break;
                }
                sink(&answer, &buf[..n as usize])?;
            }
            Ok(status)
        }
    }

    /// The first byte of a `Content-Range: bytes <first>-<last>/<size>` header.
    pub fn content_range_start(v: &str) -> Option<u64> {
        v.trim().strip_prefix("bytes ")?.split('-').next()?.trim().parse().ok()
    }
}

/// Reading a zip (stored and deflated entries, no zip64, no encryption: what the package script
/// makes).
mod zip {
    pub struct Entry {
        pub name: String,
        flags: u16,
        method: u16,
        crc: u32,
        csize: u64,
        size: u64,
        offset: u64,
    }

    fn u16at(b: &[u8], o: usize) -> Option<u16> {
        b.get(o..o + 2).map(|s| u16::from_le_bytes([s[0], s[1]]))
    }

    fn u32at(b: &[u8], o: usize) -> Option<u32> {
        b.get(o..o + 4).map(|s| u32::from_le_bytes([s[0], s[1], s[2], s[3]]))
    }

    /// The central directory's entries.
    pub fn entries(z: &[u8]) -> Result<Vec<Entry>, String> {
        const DAMAGED: &str = "the download is not a whole zip file";
        if z.len() < 22 {
            return Err(DAMAGED.into());
        }
        let lo = z.len().saturating_sub(22 + 0xffff);
        let eocd = (lo..=z.len() - 22).rev().find(|&i| u32at(z, i) == Some(0x0605_4b50)).ok_or(DAMAGED)?;
        let count = u16at(z, eocd + 10).ok_or(DAMAGED)? as usize;
        let off = u32at(z, eocd + 16).ok_or(DAMAGED)?;
        if count == 0xffff || off == u32::MAX {
            return Err("the download is a zip this launcher cannot read".into());
        }
        let mut v = Vec::with_capacity(count);
        let mut p = off as usize;
        for _ in 0..count {
            if u32at(z, p) != Some(0x0201_4b50) {
                return Err(DAMAGED.into());
            }
            let f = |o: usize| u16at(z, p + o).ok_or(DAMAGED);
            let g = |o: usize| u32at(z, p + o).ok_or(DAMAGED);
            let (nlen, elen, clen) = (f(28)? as usize, f(30)? as usize, f(32)? as usize);
            let name = z.get(p + 46..p + 46 + nlen).ok_or(DAMAGED)?;
            if !name.is_ascii() {
                return Err("the download holds a file name this launcher cannot read".into());
            }
            v.push(Entry { name: String::from_utf8_lossy(name).into_owned(), flags: f(8)?, method: f(10)?, crc: g(16)?, csize: g(20)? as u64, size: g(24)? as u64, offset: g(42)? as u64 });
            p += 46 + nlen + elen + clen;
        }
        Ok(v)
    }

    /// An entry's contents, their length and CRC-32 checked.
    pub fn read(z: &[u8], e: &Entry) -> Result<Vec<u8>, String> {
        let damaged = || format!("{} in the download is damaged", e.name);
        if e.flags & 1 != 0 {
            return Err(format!("{} in the download is encrypted", e.name));
        }
        let o = e.offset as usize;
        if u32at(z, o) != Some(0x0403_4b50) {
            return Err(damaged());
        }
        let start = o + 30 + u16at(z, o + 26).ok_or_else(damaged)? as usize + u16at(z, o + 28).ok_or_else(damaged)? as usize;
        let data = z.get(start..start + e.csize as usize).ok_or_else(damaged)?;
        let out = match e.method {
            0 => data.to_vec(),
            8 => miniz_oxide::inflate::decompress_to_vec_with_limit(data, e.size as usize).map_err(|_| damaged())?,
            m => return Err(format!("{} in the download is packed in a way this launcher cannot read ({m})", e.name)),
        };
        if out.len() as u64 != e.size || crc32(&out) != e.crc {
            return Err(damaged());
        }
        Ok(out)
    }

    pub fn crc32(b: &[u8]) -> u32 {
        static TABLE: std::sync::OnceLock<[u32; 256]> = std::sync::OnceLock::new();
        let t = TABLE.get_or_init(|| {
            let mut t = [0u32; 256];
            for (i, v) in t.iter_mut().enumerate() {
                let mut c = i as u32;
                for _ in 0..8 {
                    c = if c & 1 != 0 { 0xedb8_8320 ^ (c >> 1) } else { c >> 1 };
                }
                *v = c;
            }
            t
        });
        !b.iter().fold(!0u32, |c, &x| t[((c ^ x as u32) & 0xff) as usize] ^ (c >> 8))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn versions() {
        assert_eq!(parse_version("v0.1.9"), Some((0, 1, 9)));
        assert_eq!(parse_version("0.10.0"), Some((0, 10, 0)));
        assert_eq!(parse_version("v0.2.0-beta"), None);
        assert_eq!(parse_version("0.2"), None);
        assert_eq!(parse_version("0.2.0.1"), None);
        assert!(parse_version("0.1.10") > parse_version("0.1.9"));
    }

    fn release(tag: &str, with_sig: bool) -> serde_json::Value {
        let v = tag.trim_start_matches('v');
        let mut assets = vec![serde_json::json!({"name": format!("Regaliant-{v}.zip"), "size": 104_000_000u64, "browser_download_url": format!("https://github.com/x/releases/download/{tag}/Regaliant-{v}.zip")})];
        if with_sig {
            assets.push(serde_json::json!({"name": format!("Regaliant-{v}.zip.sig"), "size": 64, "browser_download_url": format!("https://github.com/x/releases/download/{tag}/Regaliant-{v}.zip.sig")}));
        }
        serde_json::json!({"tag_name": tag, "draft": false, "prerelease": false, "assets": assets,
            "body": "Intro line.\n\n## Changes\n\n- Fixed one thing.\n- Added another.\n\n## Known issues\n\n- Something."})
    }

    #[test]
    fn offers() {
        let o = offer_from(&release("v0.1.9", true), "0.1.8").unwrap().unwrap();
        assert_eq!(o.version, "0.1.9");
        assert_eq!(o.zip_size, 104_000_000);
        assert!(o.sig_url.ends_with(".zip.sig"));
        assert_eq!(o.changes, "- Fixed one thing.\n- Added another.");
        assert_eq!(offer_from(&release("v0.1.8", true), "0.1.8").unwrap(), None);
        assert_eq!(offer_from(&release("v0.1.7", true), "0.1.8").unwrap(), None);
        assert!(offer_from(&release("v0.1.9", false), "0.1.8").is_err());
        let mut pre = release("v0.1.9", true);
        pre["prerelease"] = serde_json::json!(true);
        assert_eq!(offer_from(&pre, "0.1.8").unwrap(), None);
    }

    #[test]
    fn names() {
        assert_eq!(valid_name("upscalers/libxess.dll").as_deref(), Some("upscalers\\libxess.dll"));
        assert_eq!(valid_name("regaliant.exe").as_deref(), Some("regaliant.exe"));
        assert_eq!(valid_name("README.txt").as_deref(), Some("README.txt"));
        for bad in [
            "../x.exe",
            "a/../../x",
            "/x.exe",
            "C:/x.exe",
            "c:x",
            "x.exe:stream",
            "user/saves/x",
            "user./saves/x",
            "portable.txt",
            "portable.txt.",
            "portable.txt ",
            "upscalers/nvngx_dlss.dll",
            "upscalers/nvngx_dlss.dll.",
            "UPSCAL~1/nvngx_dlss.dll",
            "upscalers/NVNGX_~1.DLL",
            "NUL",
            "con.txt",
            "upscalers/COM1.dll",
            "CONIN$",
            ".regaliant-backup-0.1.8/x",
            "a//b",
            "a/b/c/d/e",
            "",
        ] {
            assert_eq!(valid_name(bad), None, "{bad}");
        }
    }

    #[test]
    fn local_addresses() {
        assert!(http::local("http://127.0.0.1:8765/latest.json"));
        assert!(http::local("https://localhost/x"));
        assert!(http::local("http://[::1]:80/x"));
        assert!(!http::local("https://api.github.com/repos/x"));
        assert!(!http::local("https://127.0.0.1.example.com/x"));
        assert!(!http::local("http://evil.com/127.0.0.1"));
        assert_eq!(http::content_range_start("bytes 1000-1999/5000"), Some(1000));
        assert_eq!(http::content_range_start("bytes */5000"), None);
    }

    #[test]
    fn built_in_key() {
        let k = public_key().unwrap();
        assert_eq!(k.len(), 32);
        assert_eq!(base64("TWFu").unwrap(), b"Man");
        assert_eq!(base64("TWE=").unwrap(), b"Ma");
    }

    fn hex(s: &str) -> Vec<u8> {
        (0..s.len()).step_by(2).map(|i| u8::from_str_radix(&s[i..i + 2], 16).unwrap()).collect()
    }

    /// A throwaway key's signature made by OpenSSL 3.5 (`openssl pkeyutl -sign -rawin`), as the
    /// release step signs: it verifies here, and a changed message or signature does not.
    #[test]
    fn openssl_signature() {
        let key: [u8; 32] = hex("6d2327e6470c99054f21e2ef537aaa78f48a0b6a0e5e42f99c158bc401f6c740").try_into().unwrap();
        let sig = hex("9a657027e67bb616351ea4ca7e8b511c537c8552cb1ad7f236598a3dcd4203e76e876e75c302e125bc8cd98984ddb080e6a79ec77833ae70fea31f0e61216d00");
        let msg = b"Regaliant update test vector";
        assert!(verify_with(&key, msg, &sig).is_ok());
        assert!(verify_with(&key, b"Regaliant update test vectoR", &sig).is_err());
        let mut bad = sig.clone();
        bad[10] ^= 1;
        assert!(verify_with(&key, msg, &bad).is_err());
        assert!(verify_with(&key, msg, &sig[..63]).is_err());
        // (Not the release key.)
        assert!(verify(msg, &sig).is_err());
    }

    /// A zip made here (stored and deflated entries) as the package script makes them.
    fn make_zip(files: &[(&str, &[u8], bool)]) -> Vec<u8> {
        let mut z = Vec::new();
        let mut cd = Vec::new();
        for &(name, data, deflate) in files {
            let packed = if deflate { miniz_oxide::deflate::compress_to_vec(data, 6) } else { data.to_vec() };
            let off = z.len() as u32;
            let crc = zip::crc32(data);
            let head = |sig: u32, central: bool| {
                let mut h = Vec::new();
                h.extend_from_slice(&sig.to_le_bytes());
                if central {
                    h.extend_from_slice(&20u16.to_le_bytes());
                }
                h.extend_from_slice(&20u16.to_le_bytes());
                h.extend_from_slice(&0u16.to_le_bytes());
                h.extend_from_slice(&(if deflate { 8u16 } else { 0 }).to_le_bytes());
                h.extend_from_slice(&[0; 4]);
                h.extend_from_slice(&crc.to_le_bytes());
                h.extend_from_slice(&(packed.len() as u32).to_le_bytes());
                h.extend_from_slice(&(data.len() as u32).to_le_bytes());
                h.extend_from_slice(&(name.len() as u16).to_le_bytes());
                h.extend_from_slice(&0u16.to_le_bytes());
                if central {
                    // (Comment length, disk, internal attributes; external attributes; offset.)
                    h.extend_from_slice(&[0; 6]);
                    h.extend_from_slice(&0u32.to_le_bytes());
                    h.extend_from_slice(&off.to_le_bytes());
                }
                h.extend_from_slice(name.as_bytes());
                h
            };
            z.extend(head(0x0403_4b50, false));
            z.extend_from_slice(&packed);
            cd.extend(head(0x0201_4b50, true));
        }
        let cd_off = z.len() as u32;
        z.extend_from_slice(&cd);
        z.extend_from_slice(&0x0605_4b50u32.to_le_bytes());
        z.extend_from_slice(&[0; 4]);
        z.extend_from_slice(&(files.len() as u16).to_le_bytes());
        z.extend_from_slice(&(files.len() as u16).to_le_bytes());
        z.extend_from_slice(&(cd.len() as u32).to_le_bytes());
        z.extend_from_slice(&cd_off.to_le_bytes());
        z.extend_from_slice(&0u16.to_le_bytes());
        z
    }

    fn temp(name: &str) -> PathBuf {
        let d = std::env::temp_dir().join(format!("regaliant-update-test-{name}-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&d);
        std::fs::create_dir_all(&d).unwrap();
        d
    }

    #[test]
    fn unpacks_and_refuses() {
        let big: Vec<u8> = (0..200_000u32).map(|i| (i % 251) as u8).collect();
        let z = make_zip(&[("regaliant.exe", &big, true), ("regaliant-launcher.exe", b"STORED BYTES", false), ("upscalers/libxess.dll", b"xess", true), ("README.txt", "\u{feff}Regaliant 0.2.0 - a test\r\n".as_bytes(), false)]);
        let d = temp("unpack");
        let stage = d.join(format!("{STAGE_PREFIX}0.2.0"));
        let names = unpack(&z, &stage).unwrap();
        assert_eq!(names, vec!["regaliant.exe", "regaliant-launcher.exe", "upscalers\\libxess.dll", "README.txt"]);
        assert_eq!(std::fs::read(stage.join("regaliant.exe")).unwrap(), big);
        assert!(ours(&stage));
        assert_eq!(readme_version(&stage), Some((0, 2, 0)));
        // A changed byte in a stored file: its CRC-32 no longer matches.
        let mut bad = z.clone();
        let at = bad.windows(12).position(|w| w == b"STORED BYTES").unwrap();
        bad[at] ^= 1;
        assert!(unpack(&bad, &d.join(format!("{STAGE_PREFIX}bad"))).is_err());
        // A name outside the package's places, and a package without the game.
        assert!(unpack(&make_zip(&[("../evil.exe", b"x", false), ("regaliant.exe", b"g", false), ("regaliant-launcher.exe", b"l", false)]), &d.join(format!("{STAGE_PREFIX}evil"))).is_err());
        assert!(unpack(&make_zip(&[("regaliant-launcher.exe", b"l", false)]), &d.join(format!("{STAGE_PREFIX}nogame"))).is_err());
        assert!(unpack(b"not a zip", &d.join(format!("{STAGE_PREFIX}junk"))).is_err());
        // A folder of the same name that is not the update's own is not replaced.
        let theirs = d.join(format!("{STAGE_PREFIX}theirs"));
        std::fs::create_dir_all(&theirs).unwrap();
        std::fs::write(theirs.join("keep.txt"), "mine").unwrap();
        assert!(unpack(&z, &theirs).is_err());
        assert!(theirs.join("keep.txt").is_file());
        let _ = std::fs::remove_dir_all(&d);
    }

    /// A real package as `scripts\package.ps1` made it (in a checkout's `dist`; skipped elsewhere):
    /// every entry read and checked, every name a valid one.
    #[test]
    fn reads_a_release_zip() {
        let root = Path::new(env!("CARGO_MANIFEST_DIR")).join("..").join("..").join("dist");
        let Some(zip) = std::fs::read_dir(&root).into_iter().flatten().flatten().map(|e| e.path()).filter(|p| p.file_name().is_some_and(|n| n.to_string_lossy().starts_with("Regaliant-") && n.to_string_lossy().ends_with(".zip"))).max() else {
            return;
        };
        let z = std::fs::read(&zip).unwrap();
        let entries = zip::entries(&z).unwrap();
        assert!(entries.len() >= 3);
        for e in entries.iter().filter(|e| !e.name.ends_with('/')) {
            assert!(valid_name(&e.name).is_some(), "{}", e.name);
            zip::read(&z, e).unwrap();
        }
    }

    fn put(dir: &Path, files: &[(&str, &str)]) {
        for (n, t) in files {
            let p = dir.join(n);
            std::fs::create_dir_all(p.parent().unwrap()).unwrap();
            std::fs::write(p, t).unwrap();
        }
    }

    fn read(dir: &Path, n: &str) -> String {
        std::fs::read_to_string(dir.join(n)).unwrap_or_default()
    }

    #[test]
    fn swaps_and_puts_back() {
        let d = temp("swap");
        let (inst, inc) = (d.join("inst"), d.join(format!("{STAGE_PREFIX}1")));
        put(&inst, &[("regaliant.exe", "old"), ("regaliant-launcher.exe", "old"), ("upscalers\\libxess.dll", "old"), ("upscalers\\nvngx_dlss.dll", "mine")]);
        make_ours(&inc).unwrap();
        put(&inc, &[("regaliant.exe", "new"), ("regaliant-launcher.exe", "new"), ("upscalers\\libxess.dll", "new")]);
        let names: Vec<String> = vec!["regaliant.exe".into(), "upscalers\\libxess.dll".into(), "regaliant-launcher.exe".into()];
        let mut out = names.clone();
        out.push("README.txt".into());
        let bak = backup_dir(&inst, "1");
        swap(&inst, &inc, &names, &out, &bak).unwrap();
        assert_eq!(read(&inst, "regaliant.exe"), "new");
        assert_eq!(read(&inst, "regaliant-launcher.exe"), "new");
        assert_eq!(read(&bak, "upscalers\\libxess.dll"), "old");
        assert!(ours(&bak));
        // The player's own file beside the package's stays where it is; no journal is left.
        assert_eq!(read(&inst, "upscalers\\nvngx_dlss.dll"), "mine");
        assert!(!inst.join(JOURNAL).exists());
        // A failure halfway (an incoming file missing): everything back as it was, the failed
        // swap's backup folder gone, no journal.
        let inc2 = d.join(format!("{STAGE_PREFIX}2"));
        make_ours(&inc2).unwrap();
        put(&inc2, &[("regaliant-launcher.exe", "newer")]);
        let bak2 = backup_dir(&inst, "2");
        assert!(swap(&inst, &inc2, &names, &out, &bak2).is_err());
        assert_eq!(read(&inst, "regaliant.exe"), "new");
        assert_eq!(read(&inst, "regaliant-launcher.exe"), "new");
        assert_eq!(read(&inst, "upscalers\\libxess.dll"), "new");
        assert_eq!(read(&inc2, "regaliant-launcher.exe"), "newer");
        assert!(!bak2.exists());
        assert!(!inst.join(JOURNAL).exists());
        let _ = std::fs::remove_dir_all(&d);
    }

    /// A swap a crash stopped halfway (the journal left, some files moved): `recover` puts every
    /// file back where it was.
    #[test]
    fn interrupted_swap_undone() {
        let d = temp("recover");
        let inst = d.join("inst");
        put(&inst, &[("regaliant.exe", "old"), ("regaliant-launcher.exe", "old"), ("README.txt", "old")]);
        // (In the install, as a real swap keeps it: the journal names folders there.)
        let inc = inst.join(format!("{STAGE_PREFIX}9.9.9"));
        make_ours(&inc).unwrap();
        put(&inc, &[("regaliant.exe", "new"), ("regaliant-launcher.exe", "new"), ("README.txt", "new")]);
        let bak = backup_dir(&inst, "0.0.1");
        make_ours(&bak).unwrap();
        // As a swap leaves it after moving all out and the launcher in: the journal first.
        let j = format!(
            "regaliant update journal\nincoming = {}\nbackup = {}\nin = regaliant-launcher.exe\nin = regaliant.exe\nin = README.txt\nout = regaliant.exe\nout = README.txt\nout = regaliant-launcher.exe\n",
            inc.file_name().unwrap().to_string_lossy(),
            bak.file_name().unwrap().to_string_lossy()
        );
        std::fs::write(inst.join(JOURNAL), j).unwrap();
        for n in ["regaliant.exe", "README.txt", "regaliant-launcher.exe"] {
            std::fs::rename(inst.join(n), bak.join(n)).unwrap();
        }
        std::fs::rename(inc.join("regaliant-launcher.exe"), inst.join("regaliant-launcher.exe")).unwrap();
        let paths = dsconfig::paths::Paths { layout: Layout::Portable, exe_dir: inst.clone(), user: d.join("user"), cache: d.join("user"), root: None };
        let note = recover_at(&paths).unwrap();
        assert!(note.contains("undone"), "{note}");
        for n in ["regaliant.exe", "regaliant-launcher.exe", "README.txt"] {
            assert_eq!(read(&inst, n), "old", "{n}");
            assert_eq!(read(&inc, n), "new", "{n}");
        }
        assert!(!inst.join(JOURNAL).exists());
        assert!(!bak.exists());
        let _ = std::fs::remove_dir_all(&d);
    }

    #[test]
    fn changes_for_the_box() {
        let c = "- Fixed one thing\n  on two lines.\n\n- Added another.\nA stray line.";
        assert_eq!(changes_text(c), "\u{2022} Fixed one thing on two lines.\n\u{2022} Added another. A stray line.");
        assert_eq!(changes_text("Plain text."), "Plain text.");
        assert_eq!(changes_text(""), "");
    }

    #[test]
    fn changes_section() {
        assert_eq!(changes_of("## Changes\n- A\n- B\n## Next\n- C"), "- A\n- B");
        assert_eq!(changes_of("# Regaliant 0.1.9\n\n## Changes\n\n- A\n\n# Other"), "- A");
        assert_eq!(changes_of("No section"), "");
    }
}
