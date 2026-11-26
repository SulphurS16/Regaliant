//! Updates (docs/UPDATES.md; the user's decision of 2026-11-26): at every start the launcher asks
//! GitHub for the newest release, and a newer one shows the Update button beside Play. Nothing
//! downloads or changes until the player presses it. Update downloads the release's zip and its
//! signature, checks the signature against the key built in here (`update-public-key.pem`,
//! Ed25519; its private half stays with the user), unpacks the zip's files beside the install,
//! moves the installed package files into a backup folder and the new ones in, and starts the new
//! launcher. On a failure the current version stays or is put back. Saves, settings, logs,
//! prepared shaders, add-ons and the player's own DLSS file are not package files and are never
//! touched. The backup goes after the new version's first game session that ends normally; until
//! then Roll back puts the earlier version back.
//!
//! The check is one HTTPS request (GitHub sees the address it comes from, as with any download);
//! the Game page's check box turns it off (`check_updates = 0` in launch.txt). A checkout's build
//! checks nothing. `DS_UPDATE_URL` (tests) names another release document; plain HTTP is taken
//! from this PC's own address only.

use crate::{App, TITLE, VERSION, ui};
use dsconfig::paths::{GAME_EXE, LAUNCHER_EXE, Layout, PORTABLE_MARKER};
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicU64, Ordering::Relaxed};
use std::sync::{Arc, Mutex};
use windows_sys::Win32::Foundation::HWND;
use windows_sys::Win32::UI::WindowsAndMessaging::*;

/// Posted when the start's check has its answer.
pub const WM_CHECKED: u32 = WM_APP + 7;
/// The download's progress and end, looked at twice a second.
pub const TIMER: usize = 11;
pub const ID_UPDATE: i32 = 908;
pub const ID_CHECK_UPDATES: i32 = 909;

/// The newest release's document on GitHub.
const LATEST: &str = "https://api.github.com/repos/SulphurS16/Regaliant/releases/latest";
/// The public half of the key the release zips are signed with.
const PUBLIC_KEY_PEM: &str = include_str!("../update-public-key.pem");
/// The largest zip taken (a package is ~100 MB).
const MAX_ZIP: u64 = 1 << 30;

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
    /// The earlier version a backup folder holds (what Roll back puts back).
    backup: Option<String>,
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

static CHECKED: Mutex<Option<Result<Option<Offer>, String>>> = Mutex::new(None);

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

fn backup_dir(app: &App, version: &str) -> PathBuf {
    app.paths.exe_dir.join(format!(".backup-{version}"))
}

fn staging_dir(app: &App, version: &str) -> PathBuf {
    app.paths.exe_dir.join(format!(".update-{version}"))
}

fn download_dir(app: &App) -> PathBuf {
    app.paths.cache.join("update")
}

/// The package files this version knows (`scripts\package.ps1`'s allowlist: the launcher, the game
/// and what Uninstall removes by name; the portable marker is the player's).
fn package_names() -> Vec<String> {
    let mut v = vec![LAUNCHER_EXE.to_string()];
    v.extend(crate::uninstall::PACKAGE_FILES.iter().filter(|n| **n != PORTABLE_MARKER).map(|n| n.to_string()));
    v
}

/// A zip entry's name as an install's relative path (backslashes), or None when it is not one a
/// package may hold: absolute, with a drive, `..`, deeper than three folders, or a place of the
/// player's (the portable layout's marker and data, their own DLSS file).
fn valid_name(n: &str) -> Option<String> {
    let n = n.replace('/', "\\");
    let parts: Vec<&str> = n.split('\\').collect();
    if n.is_empty() || n.len() > 200 || n.contains(':') || parts.len() > 4 || parts.iter().any(|p| p.is_empty() || *p == "." || *p == "..") {
        return None;
    }
    let lower = n.to_ascii_lowercase();
    if lower == PORTABLE_MARKER || lower.starts_with("user\\") || lower == "upscalers\\nvngx_dlss.dll" {
        return None;
    }
    Some(n)
}

/// The Update button (beside Uninstall, shown only when it has something to do) and the Game
/// page's check box.
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
        ui::enable(btn, app.update.job.is_none());
    }
}

/// The earlier version whose backup this install keeps: the newest one of this version or older
/// (older ones, backups of later versions left by a roll back, roll backs' `.rolled-back-*` and
/// unfinished updates' `.update-*` folders are removed).
fn find_backup(app: &App) -> Option<String> {
    let cur = parse_version(VERSION)?;
    let mut older: Vec<((u32, u32, u32), String, PathBuf)> = Vec::new();
    for e in std::fs::read_dir(&app.paths.exe_dir).ok()?.flatten() {
        let name = e.file_name().to_string_lossy().into_owned();
        if name.starts_with(".rolled-back-") || name.starts_with(".update-") {
            let _ = std::fs::remove_dir_all(e.path());
            continue;
        }
        let Some(v) = name.strip_prefix(".backup-") else { continue };
        let Some(n) = parse_version(v) else { continue };
        if n > cur {
            let _ = std::fs::remove_dir_all(e.path());
        } else {
            older.push((n, v.to_string(), e.path()));
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
/// off).
pub fn start_check(app: &App) {
    let test = std::env::var("DS_UPDATE_URL").ok().filter(|u| !u.is_empty());
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

/// The start's answer arrived.
pub fn checked(app: &mut App) {
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

/// Why the program files cannot change now, if they cannot.
fn busy(app: &App) -> Option<&'static str> {
    if app.running.is_some() || crate::sys::processes_in(&app.paths.exe_dir).iter().any(|(_, n)| n.eq_ignore_ascii_case(GAME_EXE)) {
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

/// Whether an update is being downloaded (Play and the shader preparation wait for it).
pub fn downloading(app: &App) -> bool {
    app.update.job.is_some()
}

pub fn command(app: &mut App, id: i32) {
    match id {
        ID_CHECK_UPDATES => {
            app.launch.check_updates = unsafe { ui::checked(h(app.update.check)) };
            app.save_launch();
        }
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
    let (dir, staging) = (download_dir(app), staging_dir(app, &o.version));
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
        Ok((staging, names)) => install(app, &job.version, &job.changes, &staging, &names),
        Err(e) => {
            log(app, &format!("{} not installed: {e}", job.version));
            app.set_status(&format!("The update did not complete: {e}. Regaliant {VERSION} stays."));
        }
    }
    refresh(app);
}

/// The download thread: the zip (resumed where an earlier one stopped), its signature checked,
/// its files unpacked into `staging`.
fn fetch(o: &Offer, dir: &Path, staging: &Path, done: &AtomicU64, step: &AtomicU64) -> Result<(PathBuf, Vec<String>), String> {
    std::fs::create_dir_all(dir).map_err(|e| format!("the download folder: {e}"))?;
    let zip = dir.join(format!("Regaliant-{}.zip", o.version));
    download(&o.zip_url, &zip, o.zip_size, done)?;
    let mut sig = Vec::new();
    let status = http::get(&o.sig_url, 0, "application/octet-stream", &mut |_, b| {
        if sig.len() + b.len() > 4096 {
            return Err("the signature is too large".into());
        }
        sig.extend_from_slice(b);
        Ok(())
    })?;
    if status != 200 {
        return Err(format!("the signature's download answered {status}"));
    }
    step.store(1, Relaxed);
    // (Read whole, then the file goes: an unpacked, refused or broken download is never kept.)
    let bytes = std::fs::read(&zip).map_err(|e| format!("the download: {e}"))?;
    let _ = std::fs::remove_file(&zip);
    verify(&bytes, &sig)?;
    step.store(2, Relaxed);
    let names = unpack(&bytes, staging).inspect_err(|_| {
        let _ = std::fs::remove_dir_all(staging);
    })?;
    Ok((staging.to_path_buf(), names))
}

/// Downloads `url` (`size` bytes) to `to` through `<to>.part`, going on from a part left by an
/// earlier try when the server takes a range.
fn download(url: &str, to: &Path, size: u64, done: &AtomicU64) -> Result<(), String> {
    use std::io::Write;
    let part = to.with_extension("zip.part");
    let mut have = std::fs::metadata(&part).map_or(0, |m| m.len());
    if have >= size {
        let _ = std::fs::remove_file(&part);
        have = 0;
    }
    for _ in 0..2 {
        let from = have;
        let mut f = std::fs::OpenOptions::new().create(true).append(true).open(&part).map_err(|e| format!("the download: {e}"))?;
        let mut first = true;
        let status = http::get(url, from, "application/octet-stream", &mut |status, b| {
            if first {
                first = false;
                // (A server that ignores the range sends the whole file again.)
                if status == 200 && have > 0 {
                    f.set_len(0).map_err(|e| format!("the download: {e}"))?;
                    have = 0;
                }
            }
            if status != 200 && status != 206 {
                return Ok(());
            }
            have += b.len() as u64;
            if have > size {
                return Err("the download is larger than announced".into());
            }
            f.write_all(b).map_err(|e| format!("the download: {e}"))?;
            done.store(have, Relaxed);
            Ok(())
        })?;
        drop(f);
        match status {
            200 | 206 if have == size => {
                let _ = std::fs::remove_file(to);
                return std::fs::rename(&part, to).map_err(|e| format!("the download: {e}"));
            }
            200 | 206 => return Err(format!("the download stopped at {} of {} MB", have >> 20, size >> 20)),
            // (The part no longer fits the file on the server: once more from the start.)
            416 => {
                let _ = std::fs::remove_file(&part);
                have = 0;
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

/// The zip's files into `staging` (emptied first): their names, checked by `valid_name`; the
/// launcher and the game must be among them.
fn unpack(z: &[u8], staging: &Path) -> Result<Vec<String>, String> {
    let entries = zip::entries(z)?;
    let _ = std::fs::remove_dir_all(staging);
    std::fs::create_dir_all(staging).map_err(|e| format!("the update's folder: {e}"))?;
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
        std::fs::write(&out, &data).map_err(|e| format!("{name}: {e}"))?;
        names.push(name);
    }
    for need in [GAME_EXE, LAUNCHER_EXE] {
        if !names.iter().any(|n| n.eq_ignore_ascii_case(need)) {
            return Err(format!("the download lacks {need}"));
        }
    }
    Ok(names)
}

/// Moves each installed file named in `out_names` into `backup`, then each of `in_names` from
/// `incoming` into the install. On a failure everything moved goes back and the error is
/// returned. (A running program's file can be moved on the same drive: the launcher moves its own.)
fn swap(install: &Path, incoming: &Path, in_names: &[String], out_names: &[String], backup: &Path) -> Result<(), String> {
    let mv = |from: &Path, to: &Path| -> Result<(), String> {
        if let Some(p) = to.parent() {
            std::fs::create_dir_all(p).map_err(|e| format!("{}: {e}", p.display()))?;
        }
        std::fs::rename(from, to).map_err(|e| format!("{}: {e}", from.display()))
    };
    let _ = std::fs::remove_dir_all(backup);
    let mut moved_out: Vec<&String> = Vec::new();
    let mut moved_in: Vec<&String> = Vec::new();
    let mut r = Ok(());
    for n in out_names {
        if install.join(n).is_file() {
            if let Err(e) = mv(&install.join(n), &backup.join(n)) {
                r = Err(e);
                break;
            }
            moved_out.push(n);
        }
    }
    if r.is_ok() {
        for n in in_names {
            if let Err(e) = mv(&incoming.join(n), &install.join(n)) {
                r = Err(e);
                break;
            }
            moved_in.push(n);
        }
    }
    if r.is_err() {
        for n in moved_in.iter().rev() {
            let _ = mv(&install.join(n), &incoming.join(n));
        }
        for n in moved_out.iter().rev() {
            let _ = mv(&backup.join(n), &install.join(n));
        }
    }
    r
}

/// The files under `dir` by their relative names (a backup's).
fn files_under(dir: &Path) -> Vec<String> {
    let mut out = Vec::new();
    let mut stack = vec![dir.to_path_buf()];
    while let Some(d) = stack.pop() {
        for e in std::fs::read_dir(&d).into_iter().flatten().flatten() {
            let p = e.path();
            if p.is_dir() {
                stack.push(p);
            } else if let Ok(rel) = p.strip_prefix(dir) {
                out.push(rel.to_string_lossy().into_owned());
            }
        }
    }
    out
}

/// The unpacked version goes in: the installed package files into `.backup-<this version>`, the
/// new ones in their place, the new launcher started.
fn install(app: &mut App, version: &str, changes: &str, staging: &Path, names: &[String]) {
    if let Some(why) = busy(app) {
        log(app, &format!("{version} unpacked, not installed: {why}"));
        app.set_status(&format!("{why} Then press Update again."));
        return;
    }
    let mut out = package_names();
    for n in names {
        if !out.iter().any(|o| o.eq_ignore_ascii_case(n)) {
            out.push(n.clone());
        }
    }
    let backup = backup_dir(app, VERSION);
    if let Err(e) = swap(&app.paths.exe_dir, staging, names, &out, &backup) {
        log(app, &format!("{version} not installed: {e}"));
        app.set_status(&format!("The update could not be installed: {e}. Regaliant {VERSION} stays."));
        return;
    }
    let _ = std::fs::remove_dir_all(staging);
    log(app, &format!("{version} installed, {VERSION} kept in {}", backup.file_name().unwrap_or_default().to_string_lossy()));
    let dir = download_dir(app);
    let _ = std::fs::create_dir_all(&dir);
    let _ = std::fs::write(dir.join("changes.txt"), format!("{version}\n{changes}\n"));
    restart(app, &format!("Regaliant {version} is installed. Start the launcher again."));
}

/// Roll back: the backup's version goes back in, the current one into `.rolled-back-<version>`
/// (removed at the next start).
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
    let from = backup_dir(app, &old);
    let names = files_under(&from);
    if !names.iter().any(|n| n.eq_ignore_ascii_case(LAUNCHER_EXE)) || !names.iter().any(|n| n.eq_ignore_ascii_case(GAME_EXE)) {
        app.set_status(&format!("The backup of Regaliant {old} is not complete. It cannot be put back."));
        return;
    }
    let mut out = package_names();
    for n in &names {
        if !out.iter().any(|o| o.eq_ignore_ascii_case(n)) {
            out.push(n.clone());
        }
    }
    // (Not `.backup-<this version>`: a backup of this same version is the one being put back.)
    let current = app.paths.exe_dir.join(format!(".rolled-back-{VERSION}"));
    if let Err(e) = swap(&app.paths.exe_dir, &from, &names, &out, &current) {
        log(app, &format!("roll back to {old} failed: {e}"));
        app.set_status(&format!("Regaliant {old} could not be put back: {e}."));
        return;
    }
    let _ = std::fs::remove_dir_all(&from);
    log(app, &format!("rolled back to {old}"));
    restart(app, &format!("Regaliant {old} is back. Start the launcher again."));
}

/// Starts the launcher now in the install (it waits for this one to end) and closes this one.
fn restart(app: &mut App, if_not: &str) {
    let exe = app.paths.exe_dir.join(LAUNCHER_EXE);
    let started = std::process::Command::new(&exe).arg("--after-update").arg(std::process::id().to_string()).current_dir(&app.paths.exe_dir).spawn();
    if started.is_err() {
        unsafe { ui::message(app.hwnd, if_not, TITLE, MB_ICONINFORMATION | MB_OK) };
    }
    unsafe { PostMessageW(app.hwnd, WM_CLOSE, 0, 0) };
}

/// A launcher started by an update (`--after-update <pid>`): waits for the one that started it.
pub fn wait_for(pid: u32) {
    use windows_sys::Win32::Foundation::CloseHandle;
    use windows_sys::Win32::System::Threading::{OpenProcess, PROCESS_SYNCHRONIZE, WaitForSingleObject};
    unsafe {
        let p = OpenProcess(PROCESS_SYNCHRONIZE, 0, pid);
        if !p.is_null() {
            WaitForSingleObject(p, 20_000);
            CloseHandle(p);
        }
    }
}

/// At a start after an update or a roll back: what changed, once; the update's leftovers removed.
pub fn after_start(app: &mut App) {
    for e in std::fs::read_dir(&app.paths.exe_dir).into_iter().flatten().flatten() {
        if e.file_name().to_string_lossy().starts_with(".update-") {
            let _ = std::fs::remove_dir_all(e.path());
        }
    }
    let notes = download_dir(app).join("changes.txt");
    let Ok(t) = std::fs::read_to_string(&notes) else { return };
    let _ = std::fs::remove_file(&notes);
    let (version, changes) = t.split_once('\n').unwrap_or((t.as_str(), ""));
    if version.trim() != VERSION {
        return;
    }
    log(app, &format!("now {VERSION}"));
    app.set_status(&format!("Updated to Regaliant {VERSION}."));
    let changes = changes.trim();
    if !changes.is_empty() {
        let text = format!("Regaliant {VERSION} is installed.\n\nWhat changed:\n{}", changes.replace("\n- ", "\n\u{2022} ").trim_start_matches("- "));
        unsafe { ui::message(app.hwnd, &text, TITLE, MB_ICONINFORMATION | MB_OK) };
    }
}

/// A game session ended normally: the earlier version's backup goes.
pub fn session_ok(app: &mut App) {
    if let Some(old) = app.update.backup.take() {
        let _ = std::fs::remove_dir_all(backup_dir(app, &old));
        log(app, &format!("the backup of {old} removed after a normal session"));
        refresh(app);
    }
}

/// HTTPS requests through Windows' WinHTTP (no library of our own).
mod http {
    use crate::ui::wide;
    use windows_sys::Win32::Networking::WinHttp::*;

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

    /// GET `url` (https, or http to this PC's own address for tests), from byte `from` on (a
    /// range) when not 0: its status code; its body goes to `sink` in pieces, with the status.
    pub fn get(url: &str, from: u64, accept: &str, sink: &mut dyn FnMut(u32, &[u8]) -> Result<(), String>) -> Result<u32, String> {
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
            let mut buf = vec![0u8; 1 << 16];
            loop {
                let mut n = 0u32;
                if WinHttpReadData(r.0, buf.as_mut_ptr() as *mut _, buf.len() as u32, &mut n) == 0 {
                    return Err(failed("reading the answer"));
                }
                if n == 0 {
                    break;
                }
                sink(status, &buf[..n as usize])?;
            }
            Ok(status)
        }
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
        for bad in ["../x.exe", "a/../../x", "/x.exe", "C:/x.exe", "c:x", "user/saves/x", "portable.txt", "upscalers/nvngx_dlss.dll", "a//b", "a/b/c/d/e", ""] {
            assert_eq!(valid_name(bad), None, "{bad}");
        }
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
        let z = make_zip(&[("regaliant.exe", &big, true), ("regaliant-launcher.exe", b"STORED BYTES", false), ("upscalers/libxess.dll", b"xess", true)]);
        let d = temp("unpack");
        let names = unpack(&z, &d.join("stage")).unwrap();
        assert_eq!(names, vec!["regaliant.exe", "regaliant-launcher.exe", "upscalers\\libxess.dll"]);
        assert_eq!(std::fs::read(d.join("stage").join("regaliant.exe")).unwrap(), big);
        // A changed byte in a stored file: its CRC-32 no longer matches.
        let mut bad = z.clone();
        let at = bad.windows(12).position(|w| w == b"STORED BYTES").unwrap();
        bad[at] ^= 1;
        assert!(unpack(&bad, &d.join("bad")).is_err());
        // A name outside the package's places, and a package without the game.
        assert!(unpack(&make_zip(&[("../evil.exe", b"x", false), ("regaliant.exe", b"g", false), ("regaliant-launcher.exe", b"l", false)]), &d.join("evil")).is_err());
        assert!(unpack(&make_zip(&[("regaliant-launcher.exe", b"l", false)]), &d.join("nogame")).is_err());
        assert!(unpack(b"not a zip", &d.join("junk")).is_err());
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

    #[test]
    fn swaps_and_puts_back() {
        let d = temp("swap");
        let (inst, inc, bak) = (d.join("inst"), d.join("inc"), d.join("bak"));
        for (dir, tag) in [(&inst, "old"), (&inc, "new")] {
            std::fs::create_dir_all(dir.join("upscalers")).unwrap();
            std::fs::write(dir.join("regaliant.exe"), tag).unwrap();
            std::fs::write(dir.join("upscalers").join("libxess.dll"), tag).unwrap();
        }
        // The player's own file beside the package's stays where it is.
        std::fs::write(inst.join("upscalers").join("nvngx_dlss.dll"), "mine").unwrap();
        let names: Vec<String> = vec!["regaliant.exe".into(), "upscalers\\libxess.dll".into()];
        let mut out = names.clone();
        out.push("README.txt".into());
        swap(&inst, &inc, &names, &out, &bak).unwrap();
        assert_eq!(std::fs::read_to_string(inst.join("regaliant.exe")).unwrap(), "new");
        assert_eq!(std::fs::read_to_string(bak.join("upscalers").join("libxess.dll")).unwrap(), "old");
        assert_eq!(std::fs::read_to_string(inst.join("upscalers").join("nvngx_dlss.dll")).unwrap(), "mine");
        // A failure halfway (an incoming file missing): everything back as it was.
        let inc2 = d.join("inc2");
        std::fs::create_dir_all(&inc2).unwrap();
        std::fs::write(inc2.join("regaliant.exe"), "newer").unwrap();
        let bak2 = d.join("bak2");
        assert!(swap(&inst, &inc2, &names, &out, &bak2).is_err());
        assert_eq!(std::fs::read_to_string(inst.join("regaliant.exe")).unwrap(), "new");
        assert_eq!(std::fs::read_to_string(inst.join("upscalers").join("libxess.dll")).unwrap(), "new");
        assert_eq!(std::fs::read_to_string(inc2.join("regaliant.exe")).unwrap(), "newer");
        let _ = std::fs::remove_dir_all(&d);
    }

    #[test]
    fn changes_section() {
        assert_eq!(changes_of("## Changes\n- A\n- B\n## Next\n- C"), "- A\n- B");
        assert_eq!(changes_of("# Regaliant 0.1.9\n\n## Changes\n\n- A\n\n# Other"), "- A");
        assert_eq!(changes_of("No section"), "");
    }
}
