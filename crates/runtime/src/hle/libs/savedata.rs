//! libSceSaveData: save directories and save-data memory backed by host
//! directories under local-data/saves/<user>/<title>/<dir>/ (metadata in
//! `sce_sys/`: param.bin, blocks, icon0.png).
//!
//! PS5 API (Mount3/Umount2/Prepare/Commit with transaction resources):
//! structure layouts and error codes per public sources (docs/PROVENANCE.md).
//! Transactions: a title mounts read-only (mode 1), calls Prepare, writes,
//! and calls Commit. Prepare gives the mount point a writable staging copy
//! (`<title>/.staging/<dir>`); Commit swaps it in by two renames (a crash
//! leaves the old or the new directory; `recover` completes it at the next
//! mount); unmounting without a commit discards it. Mounts without mode 1
//! are written directly. The save-data dialog (system UI) is answered
//! automatically and logged; prompts of the delete type are declined.

use crate::hle::{Registry, cstr};
use crate::{fixed, reg, sysv};
use parking_lot::Mutex;
use std::path::{Path, PathBuf};
use std::sync::OnceLock;

const ERR_PARAMETER: i32 = 0x809F_0000u32 as i32;
const ERR_BUSY: i32 = 0x809F_0003u32 as i32;
const ERR_INVALID_LOGIN_USER: i32 = 0x809F_0011u32 as i32;
/// Block count reported for saves created without one (32 KiB blocks).
const DEFAULT_BLOCKS: u64 = 1024;
const ERR_NOT_MOUNTED: i32 = 0x809F_0004u32 as i32;
const ERR_EXISTS: i32 = 0x809F_0007u32 as i32;
const ERR_NOT_FOUND: i32 = 0x809F_0008u32 as i32;
const ERR_INTERNAL: i32 = 0x809F_000Bu32 as i32;
const ERR_MOUNT_FULL: i32 = 0x809F_000Cu32 as i32;

const MOUNT_RDONLY: u32 = 1;
const MOUNT_CREATE: u32 = 4;
const MOUNT_CREATE2: u32 = 32;

static ROOT: OnceLock<PathBuf> = OnceLock::new();
/// A mounted save directory and, between Prepare and Commit, its staging
/// copy (with the transaction resource that prepared it).
#[derive(Clone)]
struct Mounted {
    dir: PathBuf,
    writable: bool,
    staging: Option<(PathBuf, i32)>,
}

static MOUNTS: Mutex<[Option<Mounted>; 16]> = Mutex::new([const { None }; 16]);

pub fn init(root: PathBuf) {
    ROOT.set(root).ok();
}

/// Persist files the title has already written. Do not commit unfinished transactions:
/// recovery at the next mount preserves the previous committed save. This cannot make
/// the guest create a new checkpoint; players should quit through the game's menu first.
#[cfg(target_os = "linux")]
pub fn flush_on_exit() {
    fn sync_tree(path: &Path) -> std::io::Result<u64> {
        let mut files = 0;
        for e in std::fs::read_dir(path)? {
            let e = e?;
            let kind = e.file_type()?;
            if kind.is_dir() { files += sync_tree(&e.path())?; }
            else if kind.is_file() { std::fs::File::open(e.path())?.sync_all()?; files += 1; }
            // Never follow a symlink out of the save tree.
        }
        std::fs::File::open(path)?.sync_all()?;
        Ok(files)
    }
    if let Some(root) = ROOT.get().filter(|r| r.exists()) {
        match sync_tree(root) {
            Ok(n) => crate::info!("savedata shutdown: synced {n} existing files; unfinished transactions retained for recovery"),
            Err(e) => crate::warn!("savedata shutdown flush: {e}"),
        }
    }
}

pub fn title_id() -> String {
    if let Some(t) = super::app::param_json().get("titleId").and_then(|v| v.as_str()) {
        return t.to_string();
    }
    match super::app::param_sfo().get("TITLE_ID") {
        Some(super::app::SfoValue::Str(s)) => s.clone(),
        _ => "PPSA01341".into(),
    }
}

/// SQL-LIKE match used by dir-name searches: `%` any run, `_` one character
/// (case-insensitive).
fn dir_name_match(name: &[u8], pat: &[u8]) -> bool {
    match pat.split_first() {
        None => name.is_empty(),
        Some((b'%', rest)) => (0..=name.len()).any(|i| dir_name_match(&name[i..], rest)),
        Some((&c, rest)) => match name.split_first() {
            Some((&n, nrest)) => (c == b'_' || n.eq_ignore_ascii_case(&c)) && dir_name_match(nrest, rest),
            None => false,
        },
    }
}

/// Mounts `path` at the first free /savedataN point.
fn mount_at_free_slot(path: PathBuf, writable: bool) -> Result<String, i32> {
    let mut mounts = MOUNTS.lock();
    if is_mounted(&mounts, &path) {
        return Err(ERR_BUSY);
    }
    let Some(slot) = mounts.iter().position(|s| s.is_none()) else { return Err(ERR_MOUNT_FULL) };
    mounts[slot] = Some(Mounted { dir: path.clone(), writable, staging: None });
    let mp = format!("/savedata{slot}");
    crate::hle::kernel::fs::mount(&mp, vec![path], writable);
    Ok(mp)
}

fn is_mounted(mounts: &[Option<Mounted>; 16], path: &Path) -> bool {
    mounts.iter().flatten().any(|m| m.dir == path)
}

/// Unmounts slot `i`, discarding uncommitted writes.
fn unmount_slot(mounts: &mut [Option<Mounted>; 16], i: usize, mp: &str) -> bool {
    let Some(m) = mounts[i].take() else { return false };
    crate::hle::kernel::fs::unmount(mp);
    if let Some((staging, resource)) = m.staging {
        let _ = std::fs::remove_dir_all(&staging);
        crate::info!("savedata: {mp} unmounted without commit: staged writes of '{}' (resource {resource}) discarded", dir_label(&m.dir));
    }
    true
}

fn dir_label(dir: &Path) -> String {
    dir.file_name().map(|n| n.to_string_lossy().into_owned()).unwrap_or_default()
}

/// `<title root>/.staging/<dir>`: hidden from directory searches (not a
/// valid directory name).
fn staging_path(dir: &Path) -> PathBuf {
    dir.parent().unwrap_or(dir).join(".staging").join(dir.file_name().unwrap_or_default())
}

fn old_path(dir: &Path) -> PathBuf {
    let mut p = staging_path(dir).into_os_string();
    p.push(".old");
    PathBuf::from(p)
}

/// Copies a directory tree; returns the number of files.
fn copy_dir(from: &Path, to: &Path) -> std::io::Result<u64> {
    std::fs::create_dir_all(to)?;
    let mut n = 0;
    for e in std::fs::read_dir(from)? {
        let e = e?;
        let target = to.join(e.file_name());
        if e.file_type()?.is_dir() {
            n += copy_dir(&e.path(), &target)?;
        } else {
            std::fs::copy(e.path(), &target)?;
            n += 1;
        }
    }
    Ok(n)
}

/// Makes `staging` the content of `dir`: dir -> old, staging -> dir, old
/// removed. Renaming a directory fails on Windows while a file in it is
/// open; then the files are copied over instead (not atomic, logged).
fn commit_dir(staging: &Path, dir: &Path) -> std::io::Result<()> {
    let old = old_path(dir);
    let _ = std::fs::remove_dir_all(&old);
    let had_dir = dir.exists();
    if had_dir {
        if let Err(e) = std::fs::rename(dir, &old) {
            crate::warn!("savedata: commit of '{}' could not rename ({e}); copying the files instead (not atomic)", dir_label(dir));
            copy_dir(staging, dir)?;
            let _ = std::fs::remove_dir_all(staging);
            return Ok(());
        }
    }
    if let Err(e) = std::fs::rename(staging, dir) {
        if had_dir {
            let _ = std::fs::rename(&old, dir);
        }
        return Err(e);
    }
    let _ = std::fs::remove_dir_all(&old);
    Ok(())
}

/// Finishes or rolls back a commit a crash interrupted (see `commit_dir`)
/// and drops staged writes that were never committed. Not for mounted
/// directories.
fn recover(dir: &Path) {
    let (staging, old) = (staging_path(dir), old_path(dir));
    if !dir.exists() && old.exists() {
        // Between the two renames: the staging copy was complete.
        let new = staging.exists() && std::fs::rename(&staging, dir).is_ok();
        if !new {
            let _ = std::fs::rename(&old, dir);
        }
        crate::warn!("savedata: completed an interrupted commit of '{}' ({} data)", dir_label(dir), if new { "new" } else { "previous" });
    }
    let _ = std::fs::remove_dir_all(&old);
    if staging.exists() {
        crate::info!("savedata: removing uncommitted staged writes of '{}'", dir_label(dir));
        let _ = std::fs::remove_dir_all(&staging);
    }
}

/// Backups kept per save directory (`backup_once`).
const BACKUPS: usize = 8;
/// The save directories backed up this session.
static BACKED_UP: Mutex<Vec<PathBuf>> = Mutex::new(Vec::new());

/// Copies `dir` as it is before this session's first change to it (a
/// writable mount, a transaction, a deletion, a save-data memory write) to
/// local-data/save-backups/<user>/<title>/<dir>/<UTC time>/, the newest
/// BACKUPS kept: a session that damages or deletes a save leaves the state
/// before it, restored by copying that folder back over the save's.
fn backup_once(dir: &Path) {
    let Some(root) = ROOT.get() else { return };
    let Ok(rel) = dir.strip_prefix(root) else { return };
    if !dir.is_dir() {
        return;
    }
    {
        let mut done = BACKED_UP.lock();
        if done.iter().any(|d| d == dir) {
            return;
        }
        done.push(dir.to_path_buf());
    }
    let base = root.parent().unwrap_or(root).join("save-backups").join(rel);
    let to = base.join(crate::utc_stamp());
    match copy_dir(dir, &to) {
        Ok(n) => crate::info!("savedata: '{}' backed up before this session's first change ({n} files): {}", dir_label(dir), to.display()),
        Err(e) => {
            crate::warn!("savedata: backup of '{}' to {} failed: {e}", dir_label(dir), to.display());
            let _ = std::fs::remove_dir_all(&to);
            return;
        }
    }
    // (Folder names are UTC times: oldest first.)
    if let Ok(rd) = std::fs::read_dir(&base) {
        let mut old: Vec<PathBuf> = rd.flatten().map(|e| e.path()).filter(|p| p.is_dir()).collect();
        old.sort();
        let n = old.len().saturating_sub(BACKUPS);
        for p in &old[..n] {
            let _ = std::fs::remove_dir_all(p);
        }
    }
}

fn read_blocks(dir: &std::path::Path) -> u64 {
    std::fs::read(dir.join("sce_sys").join("blocks"))
        .ok()
        .and_then(|b| b.get(..8).map(|x| u64::from_le_bytes(x.try_into().unwrap())))
        .unwrap_or(DEFAULT_BLOCKS)
}

fn user_root(user: i32, title: *const u8) -> PathBuf {
    let t = if title.is_null() { title_id() } else { unsafe { cstr(title) }.to_string() };
    let t = if t.is_empty() { title_id() } else { t };
    crate::platform::file_hooks::save_root(ROOT.get().expect("savedata root").join(format!("{user}")).join(t))
}

fn valid_dir_name(s: &str) -> bool {
    !s.is_empty() && s.len() < 32 && s.chars().all(|c| c.is_ascii_alphanumeric() || c == '-' || c == '_')
}

unsafe fn mount_point_index(mp: *const u8) -> Option<usize> {
    let s = unsafe { cstr(mp) };
    s.strip_prefix("/savedata").and_then(|n| n.parse::<usize>().ok()).filter(|&n| n < 16)
}

/// The directory a mount point shows (the staging copy during a transaction).
fn mounted_dir(mp: *const u8) -> Option<PathBuf> {
    let i = unsafe { mount_point_index(mp) }?;
    MOUNTS.lock()[i].as_ref().map(|m| m.staging.as_ref().map_or_else(|| m.dir.clone(), |s| s.0.clone()))
}

#[repr(C)]
struct SaveParam {
    title: [u8; 128],
    sub_title: [u8; 128],
    detail: [u8; 1024],
    user_param: u32,
    pad: i32,
    mtime: i64,
    reserved: [u8; 32],
}

fn memory_file(user: i32) -> PathBuf {
    user_root(user, std::ptr::null()).join("sdmemory").join("memory.dat")
}

sysv! {
    fn sceSaveDataMount(m: *const u8, result: *mut u8) -> i32 {
        if m.is_null() || result.is_null() { return ERR_PARAMETER; }
        let user = *(m as *const i32);
        let title = *(m.add(8) as *const u64) as *const u8;
        let dir = *(m.add(16) as *const u64) as *const u8;
        let blocks = *(m.add(32) as *const u64);
        let mode = *(m.add(40) as *const u32);
        if dir.is_null() { return ERR_PARAMETER; }
        let name = cstr(dir).to_string();
        if !valid_dir_name(&name) { return ERR_PARAMETER; }
        let path = user_root(user, title).join(&name);
        if is_mounted(&MOUNTS.lock(), &path) { return ERR_BUSY; }
        recover(&path);
        let exists = path.join("sce_sys").exists() || path.exists();
        let mut created = false;
        if !exists {
            if mode & (MOUNT_CREATE | MOUNT_CREATE2) == 0 {
                crate::info!("sceSaveDataMount '{name}' mode {mode:#x}: not found");
                return ERR_NOT_FOUND;
            }
            if std::fs::create_dir_all(path.join("sce_sys")).is_err() { return ERR_INTERNAL; }
            created = true;
        } else if mode & MOUNT_CREATE != 0 && mode & MOUNT_CREATE2 == 0 {
            return ERR_EXISTS;
        }
        if mode & MOUNT_RDONLY == 0 && !created { backup_once(&path); }
        let mp = match mount_at_free_slot(path, mode & MOUNT_RDONLY == 0) { Ok(x) => x, Err(e) => return e };
        std::ptr::write_bytes(result, 0, 64);
        crate::hle::write_cstr(result, 16, &mp);
        *(result.add(16) as *mut u64) = blocks;
        *(result.add(28) as *mut u32) = if created { 1 } else { 0 };
        crate::info!("sceSaveDataMount '{name}' mode {mode:#x} blocks {blocks} -> {mp}{}", if created { " (created)" } else { "" });
        0
    }
    fn sceSaveDataUmount(mp: *const u8) -> i32 {
        let Some(i) = mount_point_index(mp) else { return ERR_PARAMETER };
        if !unmount_slot(&mut MOUNTS.lock(), i, cstr(mp)) { return ERR_NOT_MOUNTED; }
        0
    }
    fn sceSaveDataDirNameSearch(cond: *const u8, result: *mut u8) -> i32 {
        if cond.is_null() || result.is_null() { return ERR_PARAMETER; }
        let user = *(cond as *const i32);
        let title = *(cond.add(8) as *const u64) as *const u8;
        let pattern = *(cond.add(16) as *const u64) as *const u8;
        let pat = if pattern.is_null() { String::new() } else { cstr(pattern).to_string() };
        let root = user_root(user, title);
        let mut names: Vec<String> = std::fs::read_dir(&root)
            .map(|rd| rd.flatten().filter(|e| e.path().is_dir()).map(|e| e.file_name().to_string_lossy().into_owned()).collect())
            .unwrap_or_default();
        names.retain(|n| n != "sdmemory");
        names.retain(|n| !n.starts_with("sce_") && valid_dir_name(n));
        if !pat.is_empty() {
            names.retain(|n| dir_name_match(n.as_bytes(), pat.as_bytes()));
        }
        names.sort();
        if *(cond.add(28) as *const u32) == 1 {
            names.reverse(); // descending order
        }
        let dir_names = *(result.add(8) as *const u64) as *mut u8;
        let cap = *(result.add(16) as *const u32) as usize;
        let params = *(result.add(24) as *const u64) as *mut SaveParam;
        let set = names.len().min(cap);
        for (i, n) in names.iter().take(set).enumerate() {
            if !dir_names.is_null() { crate::hle::write_cstr(dir_names.add(i * 32), 32, n); }
            if !params.is_null() {
                let mut p: SaveParam = std::mem::zeroed();
                if let Ok(b) = std::fs::read(root.join(n).join("sce_sys").join("param.bin")) {
                    if b.len() == std::mem::size_of::<SaveParam>() { std::ptr::copy_nonoverlapping(b.as_ptr(), &mut p as *mut _ as *mut u8, b.len()); }
                }
                std::ptr::write_unaligned(params.add(i), p);
            }
        }
        // SaveDataSearchInfo {u64 blocks, u64 free_blocks, reserved[32]} per hit.
        let infos = *(result.add(32) as *const u64) as *mut u64;
        if !infos.is_null() {
            for (i, n) in names.iter().take(set).enumerate() {
                let b = read_blocks(&root.join(n));
                std::ptr::write_bytes(infos.add(i * 6) as *mut u8, 0, 48);
                *infos.add(i * 6) = b;
                *infos.add(i * 6 + 1) = b;
            }
        }
        *(result as *mut u32) = names.len() as u32;
        *(result.add(20) as *mut u32) = set as u32;
        crate::info!("sceSaveDataDirNameSearch '{pat}': {} hit(s)", names.len());
        0
    }
    fn sceSaveDataSetParam(mp: *const u8, kind: u32, buf: *const u8, size: u64) -> i32 {
        let Some(dir) = mounted_dir(mp) else { return ERR_NOT_MOUNTED };
        let file = dir.join("sce_sys").join("param.bin");
        let mut p: SaveParam = std::mem::zeroed();
        if let Ok(b) = std::fs::read(&file) {
            if b.len() == std::mem::size_of::<SaveParam>() { std::ptr::copy_nonoverlapping(b.as_ptr(), &mut p as *mut _ as *mut u8, b.len()); }
        }
        match kind {
            0 if size as usize >= std::mem::size_of::<SaveParam>() => std::ptr::copy_nonoverlapping(buf, &mut p as *mut _ as *mut u8, std::mem::size_of::<SaveParam>()),
            1 => crate::hle::write_cstr(p.title.as_mut_ptr(), 128, cstr(buf)),
            2 => crate::hle::write_cstr(p.sub_title.as_mut_ptr(), 128, cstr(buf)),
            3 => crate::hle::write_cstr(p.detail.as_mut_ptr(), 1024, cstr(buf)),
            4 => p.user_param = *(buf as *const u32),
            _ => return ERR_PARAMETER,
        }
        p.mtime = (super::rtc::now_tick() / 1_000_000) as i64 - 62_135_596_800;
        let bytes = std::slice::from_raw_parts(&p as *const _ as *const u8, std::mem::size_of::<SaveParam>());
        let _ = std::fs::create_dir_all(dir.join("sce_sys"));
        if std::fs::write(&file, bytes).is_err() { return ERR_INTERNAL; }
        0
    }
    fn sceSaveDataSaveIcon(mp: *const u8, icon: *const u64) -> i32 {
        let Some(dir) = mounted_dir(mp) else { return ERR_NOT_MOUNTED };
        if icon.is_null() { return ERR_PARAMETER; }
        let buf = *icon as *const u8;
        let size = *icon.add(2) as usize;
        if buf.is_null() || size == 0 { return ERR_PARAMETER; }
        let _ = std::fs::create_dir_all(dir.join("sce_sys"));
        if std::fs::write(dir.join("sce_sys").join("icon0.png"), std::slice::from_raw_parts(buf, size)).is_err() { return ERR_INTERNAL; }
        0
    }
    fn sceSaveDataDelete(del: *const u8) -> i32 {
        if del.is_null() { return ERR_PARAMETER; }
        let user = *(del as *const i32);
        let title = *(del.add(8) as *const u64) as *const u8;
        let dir = *(del.add(16) as *const u64) as *const u8;
        if dir.is_null() { return ERR_PARAMETER; }
        let name = cstr(dir).to_string();
        if !valid_dir_name(&name) { return ERR_PARAMETER; }
        let path = user_root(user, title).join(&name);
        if !path.exists() { return ERR_NOT_FOUND; }
        crate::info!("sceSaveDataDelete '{name}'");
        backup_once(&path);
        if std::fs::remove_dir_all(&path).is_err() { return ERR_INTERNAL; }
        0
    }
    fn sceSaveDataSetupSaveDataMemory(user: i32, size: u64, param: *const u8) -> i32 {
        let f = memory_file(user);
        let _ = std::fs::create_dir_all(f.parent().unwrap());
        let cur = std::fs::metadata(&f).map(|m| m.len()).unwrap_or(0);
        if cur < size {
            let file = match std::fs::OpenOptions::new().create(true).write(true).truncate(false).open(&f) { Ok(x) => x, Err(_) => return ERR_INTERNAL };
            if file.set_len(size).is_err() { return ERR_INTERNAL; }
        }
        crate::info!("sceSaveDataSetupSaveDataMemory user {user} size {size:#x}");
        0
    }
    fn sceSaveDataGetSaveDataMemory(user: i32, buf: *mut u8, size: u64, offset: i64) -> i32 {
        use crate::platform::files::FileExt;
        let f = match std::fs::File::open(memory_file(user)) { Ok(f) => f, Err(_) => return ERR_NOT_FOUND };
        let out = std::slice::from_raw_parts_mut(buf, size as usize);
        out.fill(0);
        // The kernel writes the buffer: write-tracked pages cannot fault.
        let _w = crate::platform::file_hooks::host_write(buf as u64, size);
        let what = || format!("save data memory read [{offset:#x}+{size:#x}] into {:#x}", buf as u64);
        let _ = crate::mem::kernel_io(&what, buf as u64, size, true, || f.seek_read(&mut out[..], offset as u64));
        0
    }
    // ---- PS5: Mount3 {i32 user, pad, dir_name* @8, u64 blocks @16,
    // u64 system_blocks @24, u32 mode @32, pad, i32 resource @40, reserved}
    // -> MountResult {mount_point[16], u64 required_blocks @16, u32 @24,
    // u32 mount_status @28 (1 = created), reserved} (64 bytes).
    fn sceSaveDataMount3(m: *const u8, result: *mut u8) -> i32 {
        if m.is_null() || result.is_null() { return ERR_PARAMETER; }
        let user = *(m as *const i32);
        let dir = *(m.add(8) as *const u64) as *const u8;
        let blocks = *(m.add(16) as *const u64);
        let mode = *(m.add(32) as *const u32);
        let resource = *(m.add(40) as *const i32);
        if dir.is_null() { return ERR_PARAMETER; }
        if user < 0 { return ERR_INVALID_LOGIN_USER; }
        let name = cstr(dir).to_string();
        if !valid_dir_name(&name) { return ERR_PARAMETER; }
        let path = user_root(user, std::ptr::null()).join(&name);
        if is_mounted(&MOUNTS.lock(), &path) { return ERR_BUSY; }
        recover(&path);
        let exists = path.join("sce_sys").exists();
        let mut created = false;
        if !exists {
            if mode & (MOUNT_CREATE | MOUNT_CREATE2) == 0 {
                crate::info!("sceSaveDataMount3 '{name}' mode {mode:#x}: not found");
                return ERR_NOT_FOUND;
            }
            if std::fs::create_dir_all(path.join("sce_sys")).is_err() { return ERR_INTERNAL; }
            let b = if blocks == 0 { DEFAULT_BLOCKS } else { blocks };
            if std::fs::write(path.join("sce_sys").join("blocks"), b.to_le_bytes()).is_err() { return ERR_INTERNAL; }
            created = true;
        } else if mode & MOUNT_CREATE != 0 && mode & MOUNT_CREATE2 == 0 {
            return ERR_EXISTS;
        }
        if mode & MOUNT_RDONLY == 0 && !created { backup_once(&path); }
        let mp = match mount_at_free_slot(path, mode & MOUNT_RDONLY == 0) { Ok(x) => x, Err(e) => return e };
        std::ptr::write_bytes(result, 0, 64);
        crate::hle::write_cstr(result, 16, &mp);
        *(result.add(28) as *mut u32) = if created { 1 } else { 0 };
        crate::info!("sceSaveDataMount3 '{name}' mode {mode:#x} blocks {blocks} resource {resource} -> {mp}{}", if created { " (created)" } else { "" });
        0
    }
    fn sceSaveDataUmount2(mode: u32, mp: *const u8) -> i32 {
        if mp.is_null() { return ERR_PARAMETER; }
        let Some(i) = mount_point_index(mp) else { return ERR_NOT_FOUND };
        if !unmount_slot(&mut MOUNTS.lock(), i, cstr(mp)) { return ERR_NOT_FOUND; }
        crate::debug!("sceSaveDataUmount2 {} mode {mode:#x}", cstr(mp));
        0
    }
    fn sceSaveDataCreateTransactionResource(size: u32) -> i32 {
        static NEXT: std::sync::atomic::AtomicI32 = std::sync::atomic::AtomicI32::new(1);
        NEXT.fetch_add(1, std::sync::atomic::Ordering::Relaxed)
    }
    fn sceSaveDataDeleteTransactionResource(resource: i32) -> i32 { 0 }
    /// (mount point, {i32 resource, u32 prepare_mode, reserved}): starts a
    /// transaction; the mount point shows a writable staging copy.
    fn sceSaveDataPrepare(mp: *const u8, param: *const u8) -> i32 {
        if mp.is_null() || param.is_null() { return ERR_PARAMETER; }
        let Some(i) = mount_point_index(mp) else { return ERR_NOT_MOUNTED };
        let resource = *(param as *const i32);
        let mode = *(param.add(4) as *const u32);
        let mut mounts = MOUNTS.lock();
        let Some(m) = mounts[i].as_mut() else { return ERR_NOT_MOUNTED };
        if m.staging.is_none() {
            backup_once(&m.dir);
            let staging = staging_path(&m.dir);
            let _ = std::fs::remove_dir_all(&staging);
            match copy_dir(&m.dir, &staging) {
                Ok(n) => crate::info!("sceSaveDataPrepare {} resource {resource} mode {mode:#x}: staging copy of '{}' ({n} files)", cstr(mp), dir_label(&m.dir)),
                Err(e) => {
                    crate::error!("sceSaveDataPrepare {}: staging copy of '{}' failed: {e}", cstr(mp), dir_label(&m.dir));
                    let _ = std::fs::remove_dir_all(&staging);
                    return ERR_INTERNAL;
                }
            }
            crate::hle::kernel::fs::mount(cstr(mp), vec![staging.clone()], true);
            m.staging = Some((staging, resource));
        }
        0
    }
    /// ({i32 resource, u32 commit_mode, reserved}): the staged writes of the
    /// mounts the resource prepared become the save data.
    fn sceSaveDataCommit(param: *const u8) -> i32 {
        if param.is_null() { return ERR_PARAMETER; }
        let resource = *(param as *const i32);
        let mode = *(param.add(4) as *const u32);
        let mut mounts = MOUNTS.lock();
        let mut done = Vec::new();
        for (i, slot) in mounts.iter_mut().enumerate() {
            let Some(m) = slot.as_mut() else { continue };
            let Some((staging, r)) = m.staging.clone() else { continue };
            if r != resource { continue; }
            if let Err(e) = commit_dir(&staging, &m.dir) {
                crate::error!("sceSaveDataCommit: '{}': {e}", dir_label(&m.dir));
                return ERR_INTERNAL;
            }
            m.staging = None;
            crate::hle::kernel::fs::mount(&format!("/savedata{i}"), vec![m.dir.clone()], m.writable);
            done.push(dir_label(&m.dir));
        }
        if done.is_empty() {
            crate::warn!("sceSaveDataCommit resource {resource} mode {mode:#x}: no prepared save directory");
        } else {
            crate::info!("sceSaveDataCommit resource {resource} mode {mode:#x}: committed {done:?}");
        }
        0
    }
    /// (mount point, {u64 blocks, u64 free_blocks, reserved[32]})
    fn sceSaveDataGetMountInfo(mp: *const u8, info: *mut u64) -> i32 {
        if mp.is_null() || info.is_null() { return ERR_PARAMETER; }
        let Some(dir) = mounted_dir(mp) else { return ERR_NOT_MOUNTED };
        let b = read_blocks(&dir);
        std::ptr::write_bytes(info as *mut u8, 0, 48);
        *info = b;
        *info.add(1) = b;
        0
    }

    fn sceSaveDataSetSaveDataMemory(user: i32, buf: *const u8, size: u64, offset: i64) -> i32 {
        use crate::platform::files::FileExt;
        let path = memory_file(user);
        if let Some(d) = path.parent() { backup_once(d); }
        let f = match std::fs::OpenOptions::new().write(true).open(&path) { Ok(f) => f, Err(_) => return ERR_NOT_FOUND };
        let what = || format!("save data memory write [{offset:#x}+{size:#x}] from {:#x}", buf as u64);
        if crate::mem::kernel_io(&what, buf as u64, size, false, || f.seek_write(std::slice::from_raw_parts(buf, size as usize), offset as u64)).is_err() { return ERR_INTERNAL; }
        0
    }
}

fixed! {
    sceSaveDataInitialize => 0,
    sceSaveDataInitialize3 => 0,
    sceSaveDataTerminate => 0,
}

pub fn register(r: &mut Registry) {
    reg!(r, "libSceSaveData", [
        sceSaveDataMount, sceSaveDataUmount, sceSaveDataDirNameSearch, sceSaveDataSetParam, sceSaveDataSaveIcon,
        sceSaveDataDelete, sceSaveDataSetupSaveDataMemory, sceSaveDataGetSaveDataMemory, sceSaveDataSetSaveDataMemory,
        sceSaveDataInitialize, sceSaveDataInitialize3, sceSaveDataTerminate, sceSaveDataMount3, sceSaveDataUmount2,
        sceSaveDataCreateTransactionResource, sceSaveDataDeleteTransactionResource, sceSaveDataPrepare, sceSaveDataCommit,
        sceSaveDataGetMountInfo,
    ]);
    reg!(r, "libSceSaveDataDialog", [
        sceSaveDataDialogInitialize, sceSaveDataDialogTerminate, sceSaveDataDialogOpen, sceSaveDataDialogClose,
        sceSaveDataDialogGetStatus, sceSaveDataDialogUpdateStatus, sceSaveDataDialogGetResult,
        sceSaveDataDialogIsReadyToDisplay, sceSaveDataDialogProgressBarSetValue, sceSaveDataDialogProgressBarInc,
    ]);
}

// ---------------------------------------------------------------------------
// Save-data dialog (system UI). No overlay is drawn yet: each dialog is
// logged and answered. Dialogs with buttons finish after one status update
// (titles expect to observe RUNNING first); progress dialogs stay open until
// the title closes them. Delete-type prompts are declined so no save is
// removed without the user's decision.

const DLG_NONE: i32 = 0;
const DLG_INITIALIZED: i32 = 1;
const DLG_RUNNING: i32 = 2;
const DLG_FINISHED: i32 = 3;

const MODE_USER_MSG: i32 = 2;
const MODE_SYSTEM_MSG: i32 = 3;
const MODE_ERROR_CODE: i32 = 4;
const MODE_PROGRESS_BAR: i32 = 5;
const DISP_DELETE: i32 = 3;
const SYSMSG_PROGRESS: i32 = 5;
const USER_BUTTONS_NONE: i32 = 2;

const BUTTON_OK_YES: i32 = 1;
const BUTTON_NO: i32 = 2;
const RESULT_OK: i32 = 0;
const RESULT_CANCELED: i32 = 1;
const ERR_DIALOG_NOT_FINISHED: i32 = 0x80B8_0006u32 as i32;

struct Dialog {
    status: i32,
    polled: bool,
    holds: bool,
    mode: i32,
    result: i32,
    button: i32,
    user_data: u64,
    dir_name: String,
}

static DIALOG: Mutex<Dialog> =
    Mutex::new(Dialog { status: DLG_NONE, polled: false, holds: false, mode: 0, result: 0, button: 0, user_data: 0, dir_name: String::new() });

/// Whether the save-data dialog is on screen (common dialog in use).
pub fn dialog_running() -> bool {
    DIALOG.lock().status == DLG_RUNNING
}

sysv! {
    fn sceSaveDataDialogInitialize() -> i32 {
        let mut d = DIALOG.lock();
        d.status = DLG_INITIALIZED;
        d.polled = false;
        0
    }
    fn sceSaveDataDialogTerminate() -> i32 {
        DIALOG.lock().status = DLG_NONE;
        0
    }
    /// SceSaveDataDialogParam: base[48], i32 size @48, i32 mode @52,
    /// i32 dispType @56, anim* @64, items* @72, userMsg* @80, sysMsg* @88,
    /// errorCode* @96, progBar* @104, userData @112.
    fn sceSaveDataDialogOpen(p: *const u8) -> i32 {
        if p.is_null() { return ERR_PARAMETER; }
        let mode = *(p.add(52) as *const i32);
        let disp = *(p.add(56) as *const i32);
        let items = *(p.add(72) as *const u64) as *const u8;
        let user_msg = *(p.add(80) as *const u64) as *const u8;
        let sys_msg = *(p.add(88) as *const u64) as *const u8;
        let prog = *(p.add(104) as *const u64) as *const u8;
        let user_data = *(p.add(112) as *const u64);
        let mut desc = format!("mode {mode} type {disp}");
        let mut has_buttons = true;
        if mode == MODE_USER_MSG && !user_msg.is_null() {
            let buttons = *(user_msg as *const i32);
            let msg = *(user_msg.add(8) as *const u64) as *const u8;
            desc.push_str(&format!(" user message (buttons {buttons}) {:?}", if msg.is_null() { "" } else { cstr(msg) }));
            has_buttons = buttons != USER_BUTTONS_NONE;
        }
        if mode == MODE_SYSTEM_MSG && !sys_msg.is_null() {
            let t = *(sys_msg as *const i32);
            desc.push_str(&format!(" system message {t}"));
            has_buttons = t != SYSMSG_PROGRESS;
        }
        if mode == MODE_ERROR_CODE {
            let err = *(p.add(96) as *const u64) as *const i32;
            if !err.is_null() {
                desc.push_str(&format!(" error {:#x}", *err as u32));
            }
        }
        if mode == MODE_PROGRESS_BAR {
            if !prog.is_null() {
                let msg = *(prog.add(8) as *const u64) as *const u8;
                if !msg.is_null() {
                    desc.push_str(&format!(" progress {:?}", cstr(msg)));
                }
            }
            has_buttons = false;
        }
        let mut dir_name = String::new();
        if !items.is_null() {
            let names = *(items.add(16) as *const u64) as *const u8;
            let n = *(items.add(24) as *const u32);
            if !names.is_null() && n > 0 { dir_name = cstr(names).to_string(); }
        }
        let decline = disp == DISP_DELETE;
        let outcome = if !has_buttons {
            "shown until closed"
        } else if decline {
            "declined (delete prompts are not confirmed automatically)"
        } else {
            "accepted"
        };
        let dir_desc = if dir_name.is_empty() { String::new() } else { format!(" dir '{dir_name}'") };
        crate::info!("save-data dialog: {desc}{dir_desc} -> {outcome}");
        let mut d = DIALOG.lock();
        d.status = DLG_RUNNING;
        d.polled = false;
        d.holds = !has_buttons;
        d.mode = mode;
        d.user_data = user_data;
        d.dir_name = dir_name;
        (d.result, d.button) = if decline { (RESULT_CANCELED, BUTTON_NO) } else { (RESULT_OK, BUTTON_OK_YES) };
        0
    }
    fn sceSaveDataDialogClose(param: *const u8) -> i32 {
        let mut d = DIALOG.lock();
        if d.status == DLG_RUNNING { d.status = DLG_FINISHED; }
        0
    }
    fn sceSaveDataDialogGetStatus() -> i32 { DIALOG.lock().status }
    fn sceSaveDataDialogUpdateStatus() -> i32 {
        let mut d = DIALOG.lock();
        if d.status == DLG_RUNNING && !d.holds {
            if d.polled { d.status = DLG_FINISHED; }
            d.polled = true;
        }
        d.status
    }
    /// SceSaveDataDialogResult: i32 mode, i32 result, i32 buttonId, pad,
    /// dirName* @16, param* @24, userData @32.
    fn sceSaveDataDialogGetResult(r: *mut u8) -> i32 {
        if r.is_null() { return ERR_PARAMETER; }
        let d = DIALOG.lock();
        if d.status != DLG_FINISHED { return ERR_DIALOG_NOT_FINISHED; }
        *(r as *mut i32) = d.mode;
        *(r.add(4) as *mut i32) = d.result;
        *(r.add(8) as *mut i32) = d.button;
        let dn = *(r.add(16) as *const u64) as *mut u8;
        if !dn.is_null() && !d.dir_name.is_empty() { crate::hle::write_cstr(dn, 32, &d.dir_name); }
        *(r.add(32) as *mut u64) = d.user_data;
        0
    }
    fn sceSaveDataDialogIsReadyToDisplay() -> i32 { 1 }
    fn sceSaveDataDialogProgressBarSetValue(target: i32, rate: u32) -> i32 { 0 }
    fn sceSaveDataDialogProgressBarInc(target: i32, delta: u32) -> i32 { 0 }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn tree(root: &Path) -> Vec<(String, String)> {
        let mut v = Vec::new();
        for e in std::fs::read_dir(root).unwrap().flatten() {
            if e.path().is_dir() {
                for (n, c) in tree(&e.path()) {
                    v.push((format!("{}/{n}", e.file_name().to_string_lossy()), c));
                }
            } else {
                v.push((e.file_name().to_string_lossy().into_owned(), std::fs::read_to_string(e.path()).unwrap()));
            }
        }
        v.sort();
        v
    }

    #[test]
    fn commit_and_recovery() {
        let base = std::env::temp_dir().join(format!("dsport-savedata-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&base);
        let dir = base.join("SAVE0");
        std::fs::create_dir_all(dir.join("sce_sys")).unwrap();
        std::fs::write(dir.join("USR-DATA"), "old").unwrap();
        std::fs::write(dir.join("sce_sys").join("param.bin"), "p").unwrap();

        // Prepare + write + commit.
        let staging = staging_path(&dir);
        assert_eq!(copy_dir(&dir, &staging).unwrap(), 2);
        std::fs::write(staging.join("USR-DATA"), "new").unwrap();
        commit_dir(&staging, &dir).unwrap();
        assert_eq!(tree(&dir), vec![("USR-DATA".into(), "new".into()), ("sce_sys/param.bin".into(), "p".into())]);
        assert!(!staging.exists() && !old_path(&dir).exists());

        // Crash between the renames: the staged copy wins.
        copy_dir(&dir, &staging).unwrap();
        std::fs::write(staging.join("USR-DATA"), "newer").unwrap();
        std::fs::rename(&dir, old_path(&dir)).unwrap();
        recover(&dir);
        assert_eq!(std::fs::read_to_string(dir.join("USR-DATA")).unwrap(), "newer");
        assert!(!staging.exists() && !old_path(&dir).exists());

        // Prepared but never committed: discarded.
        copy_dir(&dir, &staging).unwrap();
        std::fs::write(staging.join("USR-DATA"), "lost").unwrap();
        recover(&dir);
        assert_eq!(std::fs::read_to_string(dir.join("USR-DATA")).unwrap(), "newer");
        assert!(!staging.exists());
        let _ = std::fs::remove_dir_all(&base);
    }

    /// A save directory is copied once a session, before its first change,
    /// and only the newest BACKUPS copies stay.
    #[test]
    fn backups_once_a_session() {
        let base = std::env::temp_dir().join(format!("dsport-savebackup-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&base);
        ROOT.set(base.join("saves")).ok();
        let root = ROOT.get().unwrap().clone();
        let dir = root.join("1").join("PPSA01341").join("SAVEDATA0PlayerProfile0");
        std::fs::create_dir_all(dir.join("sce_sys")).unwrap();
        std::fs::write(dir.join("USR-DATA"), "before").unwrap();
        let backups = root.parent().unwrap().join("save-backups").join("1").join("PPSA01341").join("SAVEDATA0PlayerProfile0");
        for k in 0..BACKUPS + 2 {
            std::fs::create_dir_all(backups.join(format!("20000101-0000{k:02}"))).unwrap();
        }
        backup_once(&dir);
        std::fs::write(dir.join("USR-DATA"), "after").unwrap();
        backup_once(&dir);
        let mut kept: Vec<String> = std::fs::read_dir(&backups).unwrap().flatten().map(|e| e.file_name().to_string_lossy().into_owned()).collect();
        kept.sort();
        assert_eq!(kept.len(), BACKUPS);
        assert_eq!(std::fs::read_to_string(backups.join(kept.last().unwrap()).join("USR-DATA")).unwrap(), "before");
        assert!(backups.join(kept.last().unwrap()).join("sce_sys").is_dir());
        let _ = std::fs::remove_dir_all(&base);
    }
}
