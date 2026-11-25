//! Persistent Vulkan pipeline caches, one per pipeline
//! (`local-data/pipelines/<kind><key>.bin`).
//!
//! Creating the title's pipelines from scratch takes minutes per run (~700 in
//! 122 s at the 2026-10-11 courtyard), so compiled pipelines are kept across
//! runs. One cache file for all of them grew without bound (Vulkan caches
//! never drop entries: 1.5 GB by 2026-10-11, entries of earlier translations
//! and variants included, copied out of the driver and rewritten 26 times in
//! one run), and over its 1 GiB load limit it was dropped whole, so the next
//! run recompiled everything. Now each pipeline is created with a small cache
//! of its own, loaded from the file named by the pipeline's key (translation
//! and state: `exec::compute_pipe`, the graphics state hash); the file is
//! written when creation added data (a miss, or another translation under
//! the same key) and touched when it was used as is. At startup the files used
//! longest ago go beyond `BUDGET` bytes. Data of another driver or GPU is
//! rejected by the driver's header check and replaced.
//!
//! The data of every one-pipeline cache on the RTX 4090 driver (610.88)
//! carries the same ~2 MB block (2,097,212 bytes, at varying offsets) besides
//! the pipeline's own entries (tens of KB), so files store the pipeline's
//! bytes and a reference to the block, which is kept once (`<hash>.blk`;
//! found by comparing two pipelines' data, `discover`). Files written whole
//! are converted when read.
//!
//! Pipeline binaries (VK_KHR_pipeline_binary, `vk::pipeline_binaries`):
//! the driver's binaries of a pipeline kept in `binaries/<kind><key>.pb`
//! (`create_with`), which later sessions make the pipeline from without a
//! cache object or the driver's own cache. On this driver the cache data
//! alone did not spare a compile once the driver's cache was reset
//! (FINDINGS.md 2026-11-08 evening).

#[cfg(target_os = "linux")]
#[path = "compile_journal.rs"]
mod compile_journal;

use super::vk::ctx;
use ash::vk;
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Arc, OnceLock};
use parking_lot::Mutex;

static DIR: OnceLock<PathBuf> = OnceLock::new();
/// The pipeline binaries' directory (`create_with`).
static BIN_DIR: OnceLock<PathBuf> = OnceLock::new();
/// The binaries' own budget (pruned as the cache files are).
const BIN_BUDGET: u64 = 4 << 30;
/// Pipeline binary files: magic, the global key, the fingerprint of the
/// pipeline's SPIR-V (`fingerprint`: the store's keys name translations, not
/// their code, and binaries of other code must never be used), the binaries
/// (key, data; data of SHARED_MIN bytes or more kept once, in `<key>.shared`,
/// and marked u64::MAX here), an FNV-1a hash of all before it.
const PB_MAGIC: &[u8; 8] = b"DSPBIN04";
const SHARED_MIN: usize = 1 << 20;
/// [pipelines made from binaries, their ns, binaries refused or failed,
/// pipelines whose binaries were captured, their bytes, capture ns, captures
/// failed, ns reading and decoding the files, making the binary objects,
/// making the pipelines, destroying the binary objects].
static PB: [AtomicU64; 11] = [const { AtomicU64::new(0) }; 11];
/// Binary files of other code under a pipeline's key (made again).
static PB_STALE: AtomicU64 = AtomicU64::new(0);
/// Stored binaries made under another global key (the driver's: a store it no longer takes).
static PB_OTHER_KEY: AtomicU64 = AtomicU64::new(0);
// Source creation includes internal-driver/cache hits; it is not proof of a compiler run.
static SOURCE_CALLS: AtomicU64 = AtomicU64::new(0);
static SOURCE_STARTED: AtomicU64 = AtomicU64::new(0);
static SOURCE_ACTIVE: AtomicU64 = AtomicU64::new(0);
static SOURCE_NS: AtomicU64 = AtomicU64::new(0);
static PB_MISSING: AtomicU64 = AtomicU64::new(0);
static PB_SCHEMA: AtomicU64 = AtomicU64::new(0);
static PB_INVALID: AtomicU64 = AtomicU64::new(0);
static PB_EVENTS: Mutex<Vec<serde_json::Value>> = Mutex::new(Vec::new());
static PB_LEDGER: Mutex<Vec<serde_json::Value>> = Mutex::new(Vec::new());
static PB_LEDGER_DROPPED: AtomicU64 = AtomicU64::new(0);
static PB_FRESH: [AtomicU64; 2] = [const { AtomicU64::new(0) }; 2];
fn reuse_record(kind: char, key: u64, fp: u64, outcome: &str, ms: f64) {
    let mut rows = PB_LEDGER.lock();
    if rows.len() < 32768 {
        rows.push(serde_json::json!({"key":format!("{kind}{key:016x}"), "fingerprint":format!("{fp:016x}"), "outcome":outcome, "ms":ms}));
    } else { PB_LEDGER_DROPPED.fetch_add(1, Ordering::Relaxed); }
}

fn reuse_event(kind: char, key: u64, fp: u64, reason: &str) {
    let mut events = PB_EVENTS.lock();
    if events.len() < 256 { events.push(serde_json::json!({"key":format!("{kind}{key:016x}"), "fingerprint":format!("{fp:016x}"), "reason":reason})); }
}
/// Machine-readable actual binary reuse, independent of asynchronous scheduling counters.
#[cfg(target_os = "linux")]
pub fn reuse_report() -> serde_json::Value {
    serde_json::json!({"binary_mode":super::vk::try_ctx().is_some_and(|c| c.pbin.is_some()), "binary_hits":PB[0].load(Ordering::Relaxed), "binary_refused":PB[2].load(Ordering::Relaxed),
        "missing":PB_MISSING.load(Ordering::Relaxed), "invalid":PB_INVALID.load(Ordering::Relaxed),
        "old_schema":PB_SCHEMA.load(Ordering::Relaxed), "other_driver":PB_OTHER_KEY.load(Ordering::Relaxed), "other_code":PB_STALE.load(Ordering::Relaxed),
        "captured":PB[3].load(Ordering::Relaxed), "capture_failed":PB[6].load(Ordering::Relaxed),
        "source_create_calls":SOURCE_CALLS.load(Ordering::Relaxed), "source_create_ms":SOURCE_NS.load(Ordering::Relaxed) as f64/1e6,
        "source_create_started":SOURCE_STARTED.load(Ordering::SeqCst), "source_create_active":SOURCE_ACTIVE.load(Ordering::SeqCst),
        "binary_total_ms":PB[1].load(Ordering::Relaxed) as f64/1e6,
        "cache_create_calls":LAPS[7].load(Ordering::Relaxed), "cache_create_ms":LAPS[2].load(Ordering::Relaxed) as f64/1e6,
        "cache_file_hits":STATS[0].load(Ordering::Relaxed), "cache_files_written":STATS[1].load(Ordering::Relaxed),
        "store":DIR.get(), "identity":IDENTITY.get(), "events":&*PB_EVENTS.lock(),
        "schema":"DSPBIN04/creation-contract-1", "ledger":&*PB_LEDGER.lock(), "ledger_dropped":PB_LEDGER_DROPPED.load(Ordering::Relaxed),
        "store_write_failures":WRITE_FAILS.load(Ordering::Relaxed), "fresh_replay_ok":PB_FRESH[0].load(Ordering::Relaxed), "fresh_replay_failed":PB_FRESH[1].load(Ordering::Relaxed),
        "driver_cache_env": (["__GL_SHADER_DISK_CACHE", "__GL_SHADER_DISK_CACHE_PATH", "__GL_SHADER_DISK_CACHE_SIZE", "__GL_SHADER_DISK_CACHE_SKIP_CLEANUP", "DS_PB_NO_INTERNAL"].map(|k| (k, std::env::var(k).ok()))),
        "note":"Source creates may hit driver caches. Binary hits bypass shader compilation but still instantiate Vulkan objects. Events capped at256."})
}


/// File header of data stored without its shared block: the magic, the
/// block's offset and length (u32 each) and its hash (u64), then the data
/// before and after it.
const MAGIC: &[u8; 8] = b"DSPCDDP1";
const HEADER: usize = 8 + 4 + 4 + 8;
/// Shortest block stored apart.
const MIN_BLOCK: usize = 1 << 20;
/// Known shared blocks by hash (loaded from `<hash>.blk` when first needed).
static BLOCKS: Mutex<Vec<(u64, Arc<Vec<u8>>)>> = Mutex::new(Vec::new());
/// The last data written whole: a shared block is found by comparing two.
static LAST_WHOLE: Mutex<Option<Vec<u8>>> = Mutex::new(None);

fn hash64(b: &[u8]) -> u64 {
    // FNV-1a.
    b.iter().fold(0xcbf2_9ce4_8422_2325u64, |h, &x| (h ^ x as u64).wrapping_mul(0x100_0000_01b3))
}

/// First position of `needle` in `hay`.
fn find(hay: &[u8], needle: &[u8]) -> Option<usize> {
    find_from(hay, needle, 0)
}

/// First position of `needle` in `hay` from `start`.
fn find_from(hay: &[u8], needle: &[u8], start: usize) -> Option<usize> {
    let n = needle.len();
    if n == 0 || n > hay.len() {
        return None;
    }
    let sig = &needle[..n.min(64)];
    let last = hay.len() - n;
    let mut i = start;
    while i <= last {
        let p = hay[i..=last].iter().position(|&b| b == sig[0])?;
        let j = i + p;
        if &hay[j..j + sig.len()] == sig && &hay[j..j + n] == needle {
            return Some(j);
        }
        i = j + 1;
    }
    None
}

/// A run of `b` of at least MIN_BLOCK bytes that `a` also contains, grown
/// from 64-byte windows near the middle of `b` (each occurrence in `a` is
/// tried: the data may repeat short patterns).
fn discover(a: &[u8], b: &[u8]) -> Option<Vec<u8>> {
    if b.len() < MIN_BLOCK || a.len() < MIN_BLOCK {
        return None;
    }
    for k in 0..8 {
        let mid = (b.len() / 2 + k * 4099).min(b.len() - 64);
        let w = &b[mid..mid + 64];
        let mut from = 0;
        for _ in 0..256 {
            let Some(t) = find_from(a, w, from) else { break };
            let (mut s, mut u) = (mid, t);
            while s > 0 && u > 0 && b[s - 1] == a[u - 1] {
                s -= 1;
                u -= 1;
            }
            let (mut e, mut v) = (mid + 64, t + 64);
            while e < b.len() && v < a.len() && b[e] == a[v] {
                e += 1;
                v += 1;
            }
            if e - s >= MIN_BLOCK {
                return Some(b[s..e].to_vec());
            }
            from = t + 1;
        }
    }
    None
}

fn block_path(dir: &Path, h: u64) -> PathBuf {
    dir.join(format!("{h:016x}.blk"))
}

/// The shared block with hash `h` (from its file when first needed).
fn block(h: u64) -> Option<Arc<Vec<u8>>> {
    if let Some((_, b)) = BLOCKS.lock().iter().find(|(k, _)| *k == h) {
        return Some(b.clone());
    }
    let data = std::fs::read(block_path(DIR.get()?, h)).ok()?;
    if hash64(&data) != h {
        return None;
    }
    let b = Arc::new(data);
    BLOCKS.lock().push((h, b.clone()));
    Some(b)
}

/// The blocks stored so far (their files: `<hash>.blk`).
fn load_blocks(dir: &Path) {
    let Ok(rd) = std::fs::read_dir(dir) else { return };
    for e in rd.flatten() {
        let name = e.file_name().to_string_lossy().into_owned();
        if let Some(h) = name.strip_suffix(".blk").and_then(|x| u64::from_str_radix(x, 16).ok()) {
            let _ = block(h);
        }
    }
}

/// Pipeline cache data as stored: without a known shared block when it
/// contains one (a new block is found by comparing with the last data
/// written whole, and stored).
fn encode(data: &[u8]) -> Vec<u8> {
    let known: Vec<(u64, Arc<Vec<u8>>)> = BLOCKS.lock().clone();
    for (h, b) in &known {
        if let Some(o) = find(data, b) {
            let mut out = Vec::with_capacity(HEADER + data.len() - b.len());
            out.extend_from_slice(MAGIC);
            out.extend_from_slice(&(o as u32).to_le_bytes());
            out.extend_from_slice(&(b.len() as u32).to_le_bytes());
            out.extend_from_slice(&h.to_le_bytes());
            out.extend_from_slice(&data[..o]);
            out.extend_from_slice(&data[o + b.len()..]);
            return out;
        }
    }
    // (The launcher's probe stores nothing, a shared block either; the review of 2026-11-21.)
    if READ_ONLY.load(Ordering::Relaxed) {
        return data.to_vec();
    }
    let prev = LAST_WHOLE.lock().replace(data.to_vec());
    if let (Some(prev), Some(dir)) = (prev, DIR.get()) {
        if let Some(b) = discover(&prev, data) {
            let h = hash64(&b);
            if std::fs::write(block_path(dir, h), &b).is_ok() {
                crate::info!("pipeline cache: a {} KiB block shared by the pipelines' data is stored once ({h:016x}.blk)", b.len() >> 10);
                BLOCKS.lock().push((h, Arc::new(b)));
                *LAST_WHOLE.lock() = None;
                return encode(data);
            }
        }
    }
    data.to_vec()
}

/// The pipeline cache data of a file (None: a block it names is missing).
#[cfg(test)]
fn decode(file: &[u8]) -> Option<Vec<u8>> {
    let mut out = Decoded::default();
    decode_into(file, &mut out).then_some(out.data)
}

/// A file's decoded data, and where its shared block lies in it (offset and
/// the block's hash; None: no block, or not known to be in place).
#[derive(Default)]
struct Decoded {
    data: Vec<u8>,
    block_at: Option<(usize, u64)>,
}

/// `decode` into `out` (its capacity kept): false when a block the file
/// names is missing. When `out` holds that block at the same offset already
/// (the last file of this thread put it there), only the pipeline's own
/// bytes around it are copied (53% of the files hold the block at offset
/// 36, after the cache's header).
fn decode_into(file: &[u8], out: &mut Decoded) -> bool {
    if file.len() < HEADER || &file[..8] != MAGIC {
        out.data.clear();
        out.data.extend_from_slice(file);
        out.block_at = None;
        return true;
    }
    let word = |r: std::ops::Range<usize>| file[r].iter().rev().fold(0u64, |a, &b| a << 8 | b as u64);
    let (o, n, h) = (word(8..12) as usize, word(12..16) as usize, word(16..24));
    let rest = &file[HEADER..];
    let Some(b) = block(h) else { return false };
    if b.len() != n || o > rest.len() {
        return false;
    }
    if out.block_at == Some((o, h)) && out.data.len() >= o + n {
        out.data[..o].copy_from_slice(&rest[..o]);
        out.data.truncate(o + n);
        out.data.extend_from_slice(&rest[o..]);
        return true;
    }
    out.data.clear();
    out.data.reserve(rest.len() + n);
    out.data.extend_from_slice(&rest[..o]);
    out.data.extend_from_slice(&b);
    out.data.extend_from_slice(&rest[o..]);
    out.block_at = Some((o, h));
    true
}

thread_local! {
    /// This thread's buffer for a file's decoded data (`with_cache`): ~2 MB
    /// a pipeline with the driver's shared block, which a fresh allocation
    /// committed page by page on every creation.
    static DATA: std::cell::RefCell<Decoded> = std::cell::RefCell::new(Decoded::default());
}

/// Pipeline files used as they were, whose last use the `pcache-touch`
/// thread records (for pruning; ~0.2 ms a file on the creating thread).
static TOUCH: Mutex<Vec<PathBuf>> = Mutex::new(Vec::new());

fn touch_later(p: &Path) {
    // (Nor does the probe change the files' times.)
    if READ_ONLY.load(Ordering::Relaxed) {
        return;
    }
    static STARTED: std::sync::Once = std::sync::Once::new();
    STARTED.call_once(|| {
        let _ = std::thread::Builder::new().name("pcache-touch".into()).spawn(|| loop {
            std::thread::sleep(std::time::Duration::from_secs(2));
            // (Taken first: a guard in the loop's head would be held through it.)
            let due = std::mem::take(&mut *TOUCH.lock());
            for p in due {
                if let Ok(f) = std::fs::File::options().write(true).open(&p) {
                    let _ = f.set_modified(std::time::SystemTime::now());
                }
            }
        });
    });
    TOUCH.lock().push(p.to_path_buf());
}

/// Pipeline files that could not be written (a full drive, no rights; the first 8 logged).
pub static WRITE_FAILS: AtomicU64 = AtomicU64::new(0);

/// Writes a pipeline file (temporary name first: two threads may make the
/// same pipeline).
fn write_file(p: &Path, bytes: &[u8]) -> bool {
    // (The launcher's probe writes nothing: a refused pipeline it makes again must stay refused in
    // the store, or the next probe would find the store sound while the rest of it is not.)
    if READ_ONLY.load(Ordering::Relaxed) {
        return false;
    }
    let tmp = p.with_extension(format!("{}-{:x}.tmp", std::process::id(), std::thread::current().id().as_u64_compat()));
    let write = || {
        if sim_disk_full() {
            Err(std::io::Error::other("there is not enough space on the disk (DS_SIM_DISK_FULL, a test)"))
        } else if sim_unwritable() {
            Err(std::io::Error::other("access is denied (DS_SIM_UNWRITABLE, a test)"))
        } else {
            std::fs::write(&tmp, bytes).and_then(|_| std::fs::rename(&tmp, p))
        }
    };
    match write() {
        Ok(()) => {
            WRITES_OK.fetch_add(1, Ordering::Relaxed);
            return true;
        }
        Err(e) => {
            if WRITE_FAILS.fetch_add(1, Ordering::Relaxed) < 8 {
                crate::warn!("pipeline store: {} not written: {e}", p.display());
            }
        }
    }
    let _ = std::fs::remove_file(&tmp);
    false
}

/// Pipeline files written (`write_file`).
pub static WRITES_OK: AtomicU64 = AtomicU64::new(0);
/// The store is only read (`--pb-check probe`, set before the store opens: `set_read_only`):
/// `write_file` writes nothing.
static READ_ONLY: std::sync::atomic::AtomicBool = std::sync::atomic::AtomicBool::new(false);

/// The store only read from now on (the launcher's probe), set before it opens.
pub fn set_read_only() {
    READ_ONLY.store(true, Ordering::Relaxed);
}

/// The store's folder refuses writes (not for lack of space: a folder whose permissions or a
/// security program block them): at least 16 writes failed and none was written. The steps of
/// the shader preparation end then (`prepplay::EXIT_UNWRITABLE`): nothing they make is kept.
pub fn store_unwritable() -> bool {
    WRITE_FAILS.load(Ordering::Relaxed) >= 16 && WRITES_OK.load(Ordering::Relaxed) == 0
}

/// `DS_SIM_UNWRITABLE=1` (a test): the store's writes fail, its drive with its own free space.
fn sim_unwritable() -> bool {
    static S: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *S.get_or_init(|| std::env::var("DS_SIM_UNWRITABLE").ok().as_deref() == Some("1"))
}

/// `DS_SIM_DISK_FULL=1` (a test): the store's writes fail and its drive has 100 MiB free.
fn sim_disk_full() -> bool {
    static S: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *S.get_or_init(|| std::env::var("DS_SIM_DISK_FULL").ok().as_deref() == Some("1"))
}

/// The free bytes on the store's drive (None: no store, or unknown).
pub fn store_free_bytes() -> Option<u64> {
    use crate::platform::os::OsStrExt;
    let dir = DIR.get()?;
    if sim_disk_full() {
        return Some(100 << 20);
    }
    let w: Vec<u16> = dir.as_os_str().encode_wide().chain(std::iter::once(0)).collect();
    let mut free = 0u64;
    let ok = unsafe { crate::platform::os::Win32::Storage::FileSystem::GetDiskFreeSpaceExW(w.as_ptr(), &mut free, std::ptr::null_mut(), std::ptr::null_mut()) };
    (ok != 0).then_some(free)
}

/// The store's drive is full: pipeline files could not be written and under 1 GiB is free there
/// (the shader preparation's steps end then, the launcher tells the player).
pub fn store_full() -> bool {
    WRITE_FAILS.load(Ordering::Relaxed) > 0 && store_free_bytes().is_some_and(|f| f < 1 << 30)
}

/// Stores the pipeline files written whole without their shared block
/// (before pruning, which goes by size).
fn convert(dir: &Path) {
    load_blocks(dir);
    let Ok(rd) = std::fs::read_dir(dir) else { return };
    let mut n = 0;
    for e in rd.flatten() {
        let path = e.path();
        if path.extension().is_none_or(|x| x != "bin") {
            continue;
        }
        let Ok(file) = std::fs::read(&path) else { continue };
        if file.starts_with(MAGIC) {
            continue;
        }
        let enc = encode(&file);
        if enc.len() < file.len() && write_file(&path, &enc) {
            n += 1;
        }
    }
    if n > 0 {
        crate::info!("pipeline cache: {n} pipeline files stored without their shared block");
    }
}

/// Bytes of pipeline files kept (the least recently used beyond it go at
/// startup, down to three quarters of it). 4 GiB since 2026-11-04: the
/// files the 1-1 walk, the fights and the tours use passed 2 GiB with an
/// experiment's variants, and a prune to 1.5 GiB (4,068 files) left the
/// next suite compiling an arrival's particle programs anew (a 1.85 s frame).
const BUDGET: u64 = 4 << 30;

/// [pipelines created from their file unchanged, files written (misses),
/// bytes written, files pruned].
pub static STATS: [AtomicU64; 4] = [const { AtomicU64::new(0) }; 4];

/// Files rewritten after a slow creation from them (`SLOW_CREATION`): the
/// driver refused their data, and its new data had the same size.
static REFRESHED: AtomicU64 = AtomicU64::new(0);

/// A creation from a file that took this long compiled (one from data the
/// driver accepts takes ~1-2 ms): its data is read back and compared even
/// when its size did not change. (On 2026-11-08, after a reboot, the
/// driver's ~2 MB shared block changed while its size did not: 6,021
/// creations took 776 ms each and only 37 files were rewritten, so every
/// boot compiled them again.)
const SLOW_CREATION: std::time::Duration = std::time::Duration::from_millis(20);

/// Nanoseconds of `with_cache`'s steps, summed: the file's read and its
/// decode, the cache object, the pipeline's creation, reading the cache's
/// size (its data when it grew), writing the file or queueing its last-use
/// mark, destroying the cache; then the calls.
static LAPS: [AtomicU64; 8] = [const { AtomicU64::new(0) }; 8];

/// Shader modules made for the title's pipelines (`note_module`): count, ns.
static MODULES: [AtomicU64; 2] = [const { AtomicU64::new(0) }; 2];

/// A shader module for a pipeline of the title took `ns` (graphics and
/// compute pipelines make theirs before `with_cache`; a cached creation
/// still hands the driver the module's whole SPIR-V).
pub fn note_module(ns: u64) {
    MODULES[0].fetch_add(1, Ordering::Relaxed);
    MODULES[1].fetch_add(ns, Ordering::Relaxed);
}

/// Where the pipeline files live (set once at startup; none: no persistence).
/// Prunes the directory in the background.
pub fn set_dir(dir: PathBuf) {
    if std::fs::create_dir_all(&dir).is_err() || DIR.set(dir.clone()).is_err() {
        return;
    }
    let bin = dir.join("binaries");
    let bins = std::fs::create_dir_all(&bin).is_ok() && BIN_DIR.set(bin.clone()).is_ok();
    // (The probe neither converts nor prunes.)
    if READ_ONLY.load(Ordering::Relaxed) {
        return;
    }
    let _ = std::thread::Builder::new().name("pcache-prune".into()).spawn(move || {
        convert(&dir);
        prune(&dir, "bin", BUDGET);
        if bins {
            prune(&bin, "pb", BIN_BUDGET);
        }
    });
}

/// Removes the files (extension `ext`) used longest ago while the directory
/// is over `budget`.
fn prune(dir: &Path, ext: &str, budget: u64) {
    if READ_ONLY.load(Ordering::Relaxed) {
        return;
    }
    let Ok(rd) = std::fs::read_dir(dir) else { return };
    // (Not the store check's four pipelines (`q`): no session uses them, so they would go first,
    // and the launcher's probe needs them to tell a store the driver refuses; the review of
    // 2026-11-21.)
    let mut files: Vec<(std::time::SystemTime, u64, PathBuf)> = rd
        .flatten()
        .filter(|e| e.path().extension().is_some_and(|x| x == ext) && !e.file_name().to_string_lossy().starts_with('q'))
        .filter_map(|e| {
            let m = e.metadata().ok()?;
            Some((m.modified().ok()?, m.len(), e.path()))
        })
        .collect();
    let mut total: u64 = files.iter().map(|f| f.1).sum();
    crate::info!("pipeline cache: {} pipeline .{ext} files, {} MiB", files.len(), total >> 20);
    if total <= budget {
        return;
    }
    files.sort_by_key(|f| f.0);
    let mut removed = 0u64;
    for (_, len, path) in files {
        if total <= budget / 4 * 3 {
            break;
        }
        if std::fs::remove_file(&path).is_ok() {
            total -= len;
            removed += 1;
        }
    }
    STATS[3].fetch_add(removed, Ordering::Relaxed);
    crate::info!("pipeline cache: {removed} pipeline files used longest ago removed ({} MiB left)", total >> 20);
}

/// Creates a pipeline through the cache file of `key` (`kind`: 'c' compute,
/// 'g' graphics, 'i' the runtime's own): `create` gets the pipeline cache to
/// pass to vkCreate*Pipelines.
/// Pipeline creations running now (any thread; diagnosis of stalled submissions).
pub static CREATING: std::sync::atomic::AtomicU32 = std::sync::atomic::AtomicU32::new(0);

pub fn with_cache<T>(kind: char, key: u64, create: impl FnOnce(vk::PipelineCache) -> T) -> T {
    let d = &ctx().device;
    let t = std::time::Instant::now();
    let lap = |k: usize, t: &mut std::time::Instant| {
        let n = std::time::Instant::now();
        LAPS[k].fetch_add((n - *t).as_nanos() as u64, Ordering::Relaxed);
        *t = n;
    };
    let mut t = t;
    LAPS[7].fetch_add(1, Ordering::Relaxed);
    let path = DIR.get().map(|dir| dir.join(format!("{kind}{key:016x}.bin")));
    let file = path.as_ref().and_then(|p| std::fs::read(p).ok()).unwrap_or_default();
    lap(6, &mut t);
    let whole = !file.starts_with(MAGIC);
    // (Taken out of the thread's slot while in use: `create` may make
    // another pipeline.)
    let mut decoded = DATA.with(|b| std::mem::take(&mut *b.borrow_mut()));
    if !decode_into(&file, &mut decoded) {
        decoded.data.clear();
        decoded.block_at = None;
    }
    let data = &decoded.data;
    lap(0, &mut t);
    let cache = unsafe { d.create_pipeline_cache(&vk::PipelineCacheCreateInfo::default().initial_data(data), None) }
        .or_else(|_| unsafe { d.create_pipeline_cache(&vk::PipelineCacheCreateInfo::default(), None) })
        .unwrap_or(vk::PipelineCache::null());
    lap(1, &mut t);
    CREATING.fetch_add(1, Ordering::Relaxed);
    let tc = std::time::Instant::now();
    if file.is_empty() {
        sim_compile_delay();
    }
    let r = create(cache);
    let slow = !file.is_empty() && tc.elapsed() >= SLOW_CREATION;
    CREATING.fetch_sub(1, Ordering::Relaxed);
    lap(2, &mut t);
    if cache == vk::PipelineCache::null() {
        DATA.with(|b| *b.borrow_mut() = decoded);
        return r;
    }
    if let Some(p) = &path {
        // The creation added nothing when the cache's size is unchanged
        // (the data itself, ~2 MB, is read back only when it grew).
        let mut size = 0usize;
        let sized = unsafe { (d.fp_v1_0().get_pipeline_cache_data)(d.handle(), cache, &mut size, std::ptr::null_mut()) } == vk::Result::SUCCESS;
        let got = if sized && size == data.len() && !(whole && !file.is_empty()) && !slow {
            Ok(None)
        } else {
            unsafe { d.get_pipeline_cache_data(cache) }.map(Some)
        };
        lap(3, &mut t);
        match got {
            Ok(None) => {
                STATS[0].fetch_add(1, Ordering::Relaxed);
                touch_later(p);
            }
            Ok(Some(new)) => match new {
                new if new.len() != data.len() => {
                    let enc = encode(&new);
                    if write_file(p, &enc) {
                        STATS[1].fetch_add(1, Ordering::Relaxed);
                        STATS[2].fetch_add(enc.len() as u64, Ordering::Relaxed);
                    }
                }
                new if whole && !file.is_empty() && !(slow && new != *data) => {
                    // A file written whole: store it without its shared block.
                    STATS[0].fetch_add(1, Ordering::Relaxed);
                    let enc = encode(&new);
                    if enc.len() < file.len() {
                        write_file(p, &enc);
                    }
                }
                new if slow && new != *data => {
                    // Refused data compiled again (`SLOW_CREATION`).
                    let enc = encode(&new);
                    if write_file(p, &enc) {
                        STATS[1].fetch_add(1, Ordering::Relaxed);
                        STATS[2].fetch_add(enc.len() as u64, Ordering::Relaxed);
                        if REFRESHED.fetch_add(1, Ordering::Relaxed) == 0 {
                            crate::info!("pipeline cache: the driver refused {kind}{key:016x}'s stored data (its creation compiled for {} ms): rewritten, as are later ones", tc.elapsed().as_millis());
                        }
                    }
                }
                _ => {
                    STATS[0].fetch_add(1, Ordering::Relaxed);
                    touch_later(p);
                }
            },
            Err(e) => crate::warn!("pipeline cache: reading the data of {kind}{key:016x} failed ({e})"),
        }
        lap(4, &mut t);
    }
    unsafe { d.destroy_pipeline_cache(cache, None) };
    lap(5, &mut t);
    DATA.with(|b| *b.borrow_mut() = decoded);
    r
}

/// Whether the store holds what makes the pipeline of `key` without a
/// compile in this session's mode: its binaries with pipeline binaries on,
/// else its cache file.
pub fn stored_cheap(kind: char, key: u64) -> bool {
    if ctx().pbin.is_some() {
        BIN_DIR.get().is_some_and(|dir| dir.join(format!("{kind}{key:016x}.pb")).exists())
    } else {
        DIR.get().is_some_and(|dir| dir.join(format!("{kind}{key:016x}.bin")).exists())
    }
}

/// How `create_with` has a pipeline made: through `cache`, with its
/// binaries captured (VK_PIPELINE_CREATE_2_CAPTURE_DATA_BIT_KHR in a
/// VkPipelineCreateFlags2CreateInfoKHR with the create info's own flags),
/// or from `binaries` (a VkPipelineBinaryInfoKHR in its chain, no cache).
pub struct Via<'a> {
    pub cache: vk::PipelineCache,
    pub capture: bool,
    pub binaries: Option<&'a [super::vk::pbin::Binary]>,
}

thread_local! {
    /// What this thread is having the driver compile (`Crumb`): named in the log when the
    /// driver faults inside it (`exception::veh_body`; the driver's compiler crashed so in
    /// mass rebuilds: FINDINGS.md 2026-11-15 evening), with its stages' SPIR-V saved into the
    /// run directory under `DS_PIPELINE_CRUMBS=1`.
    static CRUMB: std::cell::RefCell<Option<(String, Vec<(&'static str, Vec<u32>)>)>> = const { std::cell::RefCell::new(None) };
    /// The driver stopped on a debug break in this thread's compile and it was stepped over
    /// (`exception`, under the port's host): at most one a compile, and such a compile's
    /// binaries are not kept for later sessions (the second review of 2026-11-25).
    static STEPPED: std::cell::Cell<bool> = const { std::cell::Cell::new(false) };
}

/// The fault handler may step over a driver's debug break in this thread's compile: the first in
/// a compile only (true, and noted), never a second one.
pub fn note_stepped_break() -> bool {
    STEPPED.with(|s| !s.replace(true))
}

/// This thread's compile had a debug break stepped over (`note_stepped_break`).
pub fn stepped_in_compile() -> bool {
    STEPPED.with(|s| s.get())
}

/// A pipeline compile's note (`CRUMB`) for as long as it lives.
pub struct Crumb {
    #[cfg(target_os = "linux")]
    _signal: crate::platform::signals::CompileNote,
}

impl Crumb {
    pub fn set(what: impl FnOnce() -> String, code: impl FnOnce() -> Vec<(&'static str, Vec<u32>)>) -> Crumb {
        static KEEP: OnceLock<bool> = OnceLock::new();
        let keep = *KEEP.get_or_init(|| std::env::var("DS_PIPELINE_CRUMBS").ok().as_deref() == Some("1"));
        let code = if keep { code() } else { Vec::new() };
        let what = what();
        // Opt-in capture is written before entering the driver, never by a signal handler.
        #[cfg(target_os = "linux")]
        let signal = {
            if keep {
                if let Some(dir) = crate::run_dir() {
                    for (stage, words) in &code {
                        let path = dir.join(format!("fault-{stage}-{:016x}.spv", fingerprint(&[words])));
                        let bytes: Vec<u8> = words.iter().flat_map(|w| w.to_le_bytes()).collect();
                        if std::fs::write(&path, bytes).is_ok() {
                            crate::info!("pipeline compile input: {what}; {stage}={}", path.display());
                        }
                    }
                }
            }
            let mut line = match (crumb_key(&what), crash_identity()) {
                (Some((kind, key)), Some(id)) => format!("{kind}{key:016x} {id} {}", what.replace(['\n', '\r'], " ")),
                _ => what.clone(),
            };
            if keep {
                for (stage, words) in &code {
                    line.push_str(&format!(" {stage}=fault-{stage}-{:016x}.spv", fingerprint(&[words])));
                }
            }
            crate::platform::signals::CompileNote::new(&line)
        };
        inflight_mark(crumb_key(&what));
        STEPPED.with(|s| s.set(false));
        CRUMB.with(|c| {
            if let Ok(mut g) = c.try_borrow_mut() {
                *g = Some((what, code));
            }
        });
        Crumb {
            #[cfg(target_os = "linux")]
            _signal: signal,
        }
    }
}

impl Drop for Crumb {
    fn drop(&mut self) {
        inflight_mark(None);
        CRUMB.with(|c| {
            if let Ok(mut g) = c.try_borrow_mut() {
                *g = None;
            }
        });
    }
}

/// The compiles in flight, kept on disk (`inflight/inflight-<pid>.bin` in the store, 2026-11-24
/// night): a slot per compiling thread holds the pipeline it has the driver compile (`Crumb`),
/// cleared when that compile ends; a clean shutdown removes the file (`clear_crash_suspects`). A
/// session that ended otherwise, by an exit inside the driver no fault handler sees (a player's
/// sessions on NVIDIA's 621.40 ended in their background compiles with nothing in their logs), a
/// hang report (a compile that does not return) or from outside, leaves it, and the next start
/// lists those compiles as suspects (`confirm_left_suspects`): the same compile cannot end every
/// session. The launcher removes the files of the steps it ends when it closes (its
/// `end_for_close`): those ends are the player's.
struct Inflight {
    file: std::fs::File,
    next: std::sync::atomic::AtomicU32,
}

static INFLIGHT: OnceLock<Option<Inflight>> = OnceLock::new();
/// The file: the driver's pipeline cache id (ASCII, zero-padded), then a slot a thread: the kind
/// (a byte), seven bytes of padding, the key (little-endian).
const INFLIGHT_HEADER: u64 = 128;
const INFLIGHT_SLOT: u64 = 16;
const INFLIGHT_SLOTS: u32 = 256;

thread_local! {
    static INFLIGHT_SLOT_OF: std::cell::Cell<Option<u32>> = const { std::cell::Cell::new(None) };
}

/// The files' folder (apart from the store's thousands of pipeline files: a start lists it).
fn inflight_dir() -> Option<PathBuf> {
    DIR.get().map(|d| d.join("inflight"))
}

fn inflight_file(pid: u32) -> Option<PathBuf> {
    inflight_dir().map(|d| d.join(format!("inflight-{pid}.bin")))
}

fn inflight() -> Option<&'static Inflight> {
    INFLIGHT
        .get_or_init(|| {
            use crate::platform::files::FileExt;
            if READ_ONLY.load(Ordering::Relaxed) {
                return None;
            }
            let id = crash_identity()?;
            std::fs::create_dir_all(inflight_dir()?).ok()?;
            let f = std::fs::OpenOptions::new().create(true).write(true).truncate(true).open(inflight_file(std::process::id())?).ok()?;
            let mut header = [0u8; INFLIGHT_HEADER as usize];
            let n = id.len().min(header.len());
            header[..n].copy_from_slice(&id.as_bytes()[..n]);
            f.seek_write(&header, 0).ok()?;
            f.set_len(INFLIGHT_HEADER + INFLIGHT_SLOTS as u64 * INFLIGHT_SLOT).ok()?;
            Some(Inflight { file: f, next: std::sync::atomic::AtomicU32::new(0) })
        })
        .as_ref()
}

/// This thread's compile in flight (`Crumb`): `Some` as it begins, `None` as it ends.
fn inflight_mark(what: Option<(char, u64)>) {
    use crate::platform::files::FileExt;
    let Some(f) = inflight() else { return };
    let slot = INFLIGHT_SLOT_OF.with(|s| {
        s.get().unwrap_or_else(|| {
            let n = f.next.fetch_add(1, Ordering::Relaxed);
            s.set(Some(n));
            n
        })
    });
    if slot >= INFLIGHT_SLOTS {
        return;
    }
    let mut b = [0u8; INFLIGHT_SLOT as usize];
    if let Some((kind, key)) = what {
        b[0] = kind as u8;
        b[8..16].copy_from_slice(&key.to_le_bytes());
    }
    let _ = f.file.seek_write(&b, INFLIGHT_HEADER + slot as u64 * INFLIGHT_SLOT);
}

/// The compiles a dead session's `inflight-<pid>.bin` holds: its cache id and the pipelines, each
/// once (two threads can compile one pipeline at once: one session's evidence counts once).
fn inflight_left(path: &Path) -> Option<(String, Vec<(char, u64)>)> {
    let b = std::fs::read(path).ok()?;
    let header = b.get(..INFLIGHT_HEADER as usize)?;
    let id = String::from_utf8_lossy(&header[..header.iter().position(|&c| c == 0).unwrap_or(header.len())]).into_owned();
    let mut v: Vec<(char, u64)> = Vec::new();
    for s in b[INFLIGHT_HEADER as usize..].chunks_exact(INFLIGHT_SLOT as usize) {
        if let (k @ (b'g' | b'c'), Ok(key)) = (s[0], <[u8; 8]>::try_from(&s[8..16])) {
            let e = (k as char, u64::from_le_bytes(key));
            if !v.contains(&e) {
                v.push(e);
            }
        }
    }
    Some((id, v))
}

/// The driver faulted on this thread (`exception::veh_body`, at first chance): what it was
/// compiling, with the files its SPIR-V went to.
pub fn crumb_report() -> Option<String> {
    CRUMB.with(|c| {
        let g = c.try_borrow().ok()?;
        let (what, code) = g.as_ref()?;
        let mut files = Vec::new();
        if let Some(dir) = crate::run_dir() {
            for (stage, words) in code {
                let p = dir.join(format!("fault-{stage}-{:016x}.spv", fingerprint(&[words])));
                let bytes: Vec<u8> = words.iter().flat_map(|w| w.to_le_bytes()).collect();
                if std::fs::write(&p, bytes).is_ok() {
                    files.push(p.display().to_string());
                }
            }
        }
        Some(if files.is_empty() { what.clone() } else { format!("{what}; SPIR-V in {}", files.join(", ")) })
    })
}

/// Pipelines whose compile crashed the driver (`crashed.txt` in the store, 2026-11-21): the
/// fault handler appends the faulting compile's key with the driver's pipeline cache id
/// (`note_crash`), and background compiles (`draw::precompile_gfx_translations`, the launcher's
/// preparation through it) leave them out under that driver: a pipeline the driver's compiler
/// cannot make ends one session, not every session that would make it ahead (NVIDIA's compiler
/// crashed on one pipeline in two boots and on others of its vertex stage: FINDINGS.md
/// 2026-11-21 afternoon). Since 2026-11-21 night no creation makes one (`create_with`): the draw
/// or dispatch that needs it goes without it, as a crash there would come back every session.
/// Another driver (cache id) reads none of the lines.
static CRASHED: OnceLock<parking_lot::RwLock<Crashed>> = OnceLock::new();

/// The crash list as read (`crashed_set`): `background` every listed pipeline (background
/// compiles leave them out); `foreground` those a creation of any kind refuses: confirmed by a
/// fatal report once, or a suspect two sessions left. A suspect alone (a session that died
/// without its fatal report, or was ended from outside, after the driver faulted at first chance
/// in a compile it may have taken itself) holds back only background compiles: until the review
/// of 2026-11-21 night one became a refusal everywhere, its draws or dispatches dropped in every
/// later session.
#[derive(Default)]
pub struct Crashed {
    background: std::collections::HashSet<(char, u64)>,
    foreground: std::collections::HashSet<(char, u64)>,
}

impl Crashed {
    pub fn len(&self) -> usize {
        self.background.len()
    }
}

/// The marker of a suspect's line in `crashed.txt` (after the cache id).
const SUSPECT_MARK: &str = "suspect:";

/// The compute step's or the store check's process: every compile in it is a background one.
static BACKGROUND_PROCESS: std::sync::atomic::AtomicBool = std::sync::atomic::AtomicBool::new(false);

/// This process compiles only ahead of use (the compute step, the store check): `crashed_skip`
/// applies the background rule to all of it.
pub fn background_process() {
    BACKGROUND_PROCESS.store(true, Ordering::Relaxed);
}

/// Whether this thread compiles ahead of use: the background list's, precompile's, compile-ahead's
/// and the background compiles' workers, or any thread of a background process.
fn background_thread() -> bool {
    thread_local! {
        static BG: std::cell::Cell<Option<bool>> = const { std::cell::Cell::new(None) };
    }
    if BACKGROUND_PROCESS.load(Ordering::Relaxed) {
        return true;
    }
    BG.with(|b| {
        if let Some(v) = b.get() {
            return v;
        }
        let n = crate::log::thread_name();
        let v = ["gpu-prep", "gpu-precompile", "gpu-compile", "gpu-async", "gpu-gfxasync"].iter().any(|p| n.starts_with(p));
        b.set(Some(v));
        v
    })
}
/// Background compiles left out for a pipeline that crashed the driver (status).
pub static CRASH_SKIPS: AtomicU64 = AtomicU64::new(0);

fn crashed_file() -> Option<PathBuf> {
    DIR.get().map(|d| d.join("crashed.txt"))
}

fn crash_identity() -> Option<&'static str> {
    let (_, id) = IDENTITY.get()?;
    #[cfg(target_os = "linux")]
    {
        static SCOPE: OnceLock<String> = OnceLock::new();
        Some(SCOPE.get_or_init(|| format!("{id}/{}", rdna::SOURCE_HASH)).as_str())
    }
    #[cfg(windows)]
    Some(id.as_str())
}

fn crashed_set() -> Option<&'static parking_lot::RwLock<Crashed>> {
    let id = crash_identity()?;
    Some(CRASHED.get_or_init(|| {
        confirm_left_suspects();
        let mut s = Crashed::default();
        let mut suspects: std::collections::HashMap<(char, u64), u32> = std::collections::HashMap::new();
        if let Some(t) = crashed_file().and_then(|p| std::fs::read_to_string(p).ok()) {
            let mut named = 0;
            for l in t.lines() {
                let mut w = l.split_whitespace();
                let (Some(k), Some(cid)) = (w.next(), w.next()) else { continue };
                let suspect = w.next() == Some(SUSPECT_MARK);
                let mut ch = k.chars();
                if let (true, Some(kind @ ('g' | 'c')), Ok(key)) = (cid == id, ch.next(), u64::from_str_radix(ch.as_str(), 16)) {
                    // (Each listed line in the log: which pipeline and shaders, for a player's
                    // report.)
                    if named < 16 {
                        named += 1;
                        crate::info!("pipeline cache: crashed.txt lists {}", l.chars().take(400).collect::<String>());
                    }
                    s.background.insert((kind, key));
                    if suspect {
                        let n = suspects.entry((kind, key)).or_insert(0);
                        *n += 1;
                        if *n >= 2 {
                            s.foreground.insert((kind, key));
                        }
                    } else {
                        s.foreground.insert((kind, key));
                    }
                }
            }
        }
        if !s.background.is_empty() {
            crate::info!(
                "pipeline cache: {} pipelines whose compile crashed this driver are left out of background compiles, {} of them out of every creation (crashed.txt)",
                s.background.len(),
                s.foreground.len()
            );
        }
        parking_lot::RwLock::new(s)
    }))
}

/// `DS_SIM_COMPILER_CRASH=<n>` (a test): pipelines whose key n divides crash in their compile
/// (`create_with`); with `DS_SIM_COMPILER_CRASH_THREADS=<prefix>` only on threads whose name
/// starts with it (`gpu-prep`: the background list's workers, as the full injected run of
/// 2026-11-21 night met every crash).
fn sim_compiler_crash() -> Option<u64> {
    static N: OnceLock<Option<u64>> = OnceLock::new();
    static PREFIX: OnceLock<Option<String>> = OnceLock::new();
    let n = (*N.get_or_init(|| std::env::var("DS_SIM_COMPILER_CRASH").ok().and_then(|v| v.trim().parse::<u64>().ok()).filter(|&n| n >= 2)))?;
    match PREFIX.get_or_init(|| std::env::var("DS_SIM_COMPILER_CRASH_THREADS").ok().filter(|p| !p.is_empty())) {
        Some(p) if !crate::log::thread_name().starts_with(p.as_str()) => None,
        _ => Some(n),
    }
}

thread_local! {
    /// This thread makes the injected compiler crash now (`DS_SIM_COMPILER_CRASH`): the fatal
    /// report lists it as a driver's crash, though it lands in our code.
    static SIM_CRASHING: std::cell::Cell<bool> = const { std::cell::Cell::new(false) };
}

/// Whether this thread's fault is the injected compiler crash (`exception`'s fatal report).
pub fn sim_crash_now() -> bool {
    SIM_CRASHING.with(|c| c.get())
}

/// Whether a creation leaves pipeline `key` of `kind` out: its compile crashed this driver in an
/// earlier session (`Crashed`: a background compile by any listing, a creation for use only when
/// confirmed; counted).
pub fn crashed_skip(kind: char, key: u64) -> bool {
    let bg = background_thread();
    let hit = crashed_set().is_some_and(|s| {
        let s = s.read();
        if bg { s.background.contains(&(kind, key)) } else { s.foreground.contains(&(kind, key)) }
    });
    if hit {
        CRASH_SKIPS.fetch_add(1, Ordering::Relaxed);
    }
    hit
}

/// The pipeline a crumb's text names: `graphics pipeline g<16 hex>...` or `compute pipeline
/// c<16 hex>`.
fn crumb_key(what: &str) -> Option<(char, u64)> {
    let rest = what.split("pipeline ").nth(1)?;
    let kind = rest.chars().next().filter(|k| matches!(k, 'g' | 'c'))?;
    let hex = rest.get(1..17)?;
    Some((kind, u64::from_str_radix(hex, 16).ok()?))
}

#[cfg(test)]
mod crash_tests {
    #[test]
    #[cfg(target_os = "linux")]
    fn inflight_preserves_linux_scope_and_counts_a_dead_session_once() {
        const NAME: &str = "gpu::pcache::crash_tests::inflight_preserves_linux_scope_and_counts_a_dead_session_once";
        if std::env::var_os("REGALIANT_INFLIGHT_CHILD").is_none() {
            assert!(std::process::Command::new(std::env::current_exe().unwrap())
                .args(["--exact", NAME, "--nocapture"]).env("REGALIANT_INFLIGHT_CHILD", "1").status().unwrap().success());
            return;
        }
        let dir = std::env::temp_dir().join(format!("regaliant-inflight-{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        super::DIR.set(dir.clone()).unwrap();
        super::IDENTITY.set(("test-driver".into(), "test-cache".into())).unwrap();
        let crumb = super::Crumb::set(|| "compute pipeline c00000000deadbeef".into(), Vec::new);
        let path = super::inflight_file(std::process::id()).unwrap();
        let (scope, entries) = super::inflight_left(&path).unwrap();
        assert_eq!(scope, format!("test-cache/{}", rdna::SOURCE_HASH));
        assert_eq!(entries, [('c', 0xdeadbeef)]);
        let mut child = std::process::Command::new("/bin/true").spawn().unwrap();
        let dead = child.id();
        child.wait().unwrap();
        let dead_path = super::inflight_file(dead).unwrap();
        let mut bytes = std::fs::read(&path).unwrap();
        // Two worker slots for the same pipeline must still count as one session.
        let slot = bytes[128..144].to_vec();
        bytes[144..160].copy_from_slice(&slot);
        std::fs::write(&dead_path, bytes).unwrap();
        super::confirm_left_suspects();
        super::confirm_left_suspects();
        let history = std::fs::read_to_string(dir.join("crashed.txt")).unwrap();
        assert_eq!(history.lines().count(), 1);
        assert!(history.contains(&scope));
        assert!(!dead_path.exists());
        assert!(path.exists()); // The live session is never claimed.
        drop(crumb);
        assert!(super::inflight_left(&path).unwrap().1.is_empty());
        super::clear_crash_suspects_keep_inflight();
        assert!(path.exists());
        super::clear_crash_suspects();
        assert!(!path.exists());
        std::fs::remove_dir_all(dir).unwrap();
    }

    #[test]
    fn crumb_keys() {
        let g = "graphics pipeline g317775312c680825: vs 0x100a276d00 (translation e7560359797a10f9), ps 0x100c8bb500";
        assert_eq!(super::crumb_key(g), Some(('g', 0x317775312c680825)));
        assert_eq!(super::crumb_key("compute pipeline c00000000deadbeef"), Some(('c', 0xdeadbeef)));
        assert_eq!(super::crumb_key("graphics pipeline x317775312c680825"), None);
        assert_eq!(super::crumb_key("graphics pipeline g3177"), None);
        assert_eq!(super::crumb_key("something else"), None);
    }
}

/// The fault handler (`exception::veh_body`): the driver faulted inside the compile this
/// thread's crumb names (`graphics pipeline g<key>` or `compute pipeline c<key>`); its key goes
/// to `crashed.txt` with the driver's pipeline cache id.
pub fn note_crash() {
    if let (Some(line), Some(path)) = (crash_line(), crashed_file()) {
        append_line(&path, &line);
    }
}

/// The crash list's line for what this thread's crumb names: `<kind><key> <cache id> <what>`.
fn crash_line() -> Option<String> {
    if READ_ONLY.load(Ordering::Relaxed) {
        return None;
    }
    let what = CRUMB.with(|c| c.try_borrow().ok().and_then(|g| g.as_ref().map(|(w, _)| w.clone())))?;
    let (kind, key) = crumb_key(&what)?;
    let (_, id) = IDENTITY.get()?;
    Some(format!("{kind}{key:016x} {id} {}", what.replace(['\n', '\r'], " ")))
}

fn append_line(path: &Path, line: &str) {
    use std::io::Write;
    if let Ok(mut f) = std::fs::OpenOptions::new().create(true).append(true).open(path) {
        let _ = writeln!(f, "{line}");
    }
}

/// The suspected compiles (`crash-suspect.txt`): a fault inside the driver at its first chance,
/// which the driver's own handlers may still take. A fatal end confirms it (`note_crash` from the
/// fault report); a clean shutdown drops this process's (`clear_crash_suspects`); a session that
/// ended neither way left its suspects, which the next session confirms (`confirm_left_suspects`).
/// Until 2026-11-21 night the first chance went straight to the crash list, so a fault the driver
/// handled itself would have left a pipeline out for good.
fn suspect_file() -> Option<PathBuf> {
    DIR.get().map(|d| d.join("crash-suspect.txt"))
}

/// A fault inside the driver at its first chance, while this thread compiles (`exception`).
pub fn note_crash_suspect() {
    if let (Some(line), Some(path)) = (crash_line(), suspect_file()) {
        append_line(&path, &format!("pid={} {line}", std::process::id()));
    }
}

/// A clean shutdown: this process's suspects were handled by the driver, not fatal; its compiles
/// in flight (`Inflight`) ended with it.
pub fn clear_crash_suspects() {
    if INFLIGHT.get().is_some_and(|f| f.is_some())
        && let Some(p) = inflight_file(std::process::id())
    {
        let _ = std::fs::remove_file(p);
    }
    clear_first_chance_suspects();
}

/// A session found hung (`crate::hung`): its first-chance suspects go as at a clean end, its
/// compiles in flight stay for the next start (a compile that does not return is what the
/// compute step's and the self-play's watches end a session for: the review of 2026-11-24 night).
pub fn clear_crash_suspects_keep_inflight() {
    clear_first_chance_suspects();
}

fn clear_first_chance_suspects() {
    let Some(path) = suspect_file() else { return };
    let Ok(t) = std::fs::read_to_string(&path) else { return };
    let mine = format!("pid={} ", std::process::id());
    let rest: String = t.lines().filter(|l| !l.starts_with(&mine)).map(|l| format!("{l}\n")).collect();
    let _ = if rest.is_empty() { std::fs::remove_file(&path) } else { std::fs::write(&path, rest) };
}

/// At this session's start (`crashed_set`): suspects other sessions left, which ended neither by
/// a clean exit nor through the fault report (a fail-fast exit, or ended from outside), go to the
/// crash list as suspects (`Crashed`); a session still running keeps its own.
fn confirm_left_suspects() {
    let (Some(path), Some(crashed)) = (suspect_file(), crashed_file()) else { return };
    if READ_ONLY.load(Ordering::Relaxed) {
        return;
    }
    // (Pipelines a dead session's own first-chance note named: its compiles in flight below are
    // not counted again for that session.)
    let mut noted: std::collections::HashSet<(u32, String)> = std::collections::HashSet::new();
    if let Ok(t) = std::fs::read_to_string(&path) {
        let mut kept = String::new();
        for l in t.lines() {
            let Some((pid, line)) = l.split_once(' ') else { continue };
            let pid = pid.strip_prefix("pid=").and_then(|p| p.parse::<u32>().ok()).unwrap_or(0);
            if pid == std::process::id() || process_alive(pid) {
                kept.push_str(l);
                kept.push('\n');
                continue;
            }
            // (`<kind><key> <cache id> suspect: <what>`.)
            let mut w = line.splitn(3, ' ');
            if let (Some(k), Some(cid)) = (w.next(), w.next()) {
                append_line(&crashed, &format!("{k} {cid} {SUSPECT_MARK} {}", w.next().unwrap_or("")));
                noted.insert((pid, k.to_string()));
            }
        }
        let _ = if kept.is_empty() { std::fs::remove_file(&path) } else { std::fs::write(&path, kept) };
    }
    // Compiles in flight when a session ended without a clean exit (`Inflight`): each dead
    // session's file claimed by renaming it first (several processes start at once: the evidence
    // counts once).
    let Some(dir) = inflight_dir() else { return };
    let Ok(rd) = std::fs::read_dir(&dir) else { return };
    for e in rd.flatten() {
        let name = e.file_name().to_string_lossy().into_owned();
        let Some(pid) = name.strip_prefix("inflight-").and_then(|s| s.strip_suffix(".bin")).and_then(|s| s.parse::<u32>().ok()) else { continue };
        if pid == std::process::id() || process_alive(pid) {
            continue;
        }
        let claimed = dir.join(format!("inflight-{pid}.claimed-{}", std::process::id()));
        if std::fs::rename(e.path(), &claimed).is_err() {
            continue;
        }
        if let Some((cid, pipelines)) = inflight_left(&claimed) {
            for (kind, key) in pipelines {
                let k = format!("{kind}{key:016x}");
                if !noted.contains(&(pid, k.clone())) {
                    crate::warn!("pipeline cache: {k} was being compiled when an earlier session (process {pid}) ended without a clean exit: a suspect (crashed.txt)");
                    append_line(&crashed, &format!("{k} {cid} {SUSPECT_MARK} in flight when session {pid} ended"));
                }
            }
        }
        let _ = std::fs::remove_file(&claimed);
    }
}

/// Whether process `pid` still runs (a suspect's session beside this one).
fn process_alive(pid: u32) -> bool {
    use crate::platform::os::Win32::System::Threading::{GetExitCodeProcess, OpenProcess, PROCESS_QUERY_LIMITED_INFORMATION};
    if pid == 0 {
        return false;
    }
    unsafe {
        let h = OpenProcess(PROCESS_QUERY_LIMITED_INFORMATION, 0, pid);
        if h.is_null() {
            return false;
        }
        let mut code = 0u32;
        let ok = GetExitCodeProcess(h, &mut code) != 0;
        crate::platform::os::Win32::Foundation::CloseHandle(h);
        ok && code == 259 // STILL_ACTIVE
    }
}

/// Creates the pipeline of `key` (`kind` 'c' compute, 'g' graphics) through
/// the store: with pipeline binaries on (`vk::pipeline_binaries`), from its
/// stored binaries when it has them (and they were made from the same code,
/// `fp`), else made anew with its binaries captured into the store; without
/// them through its cache file (`with_cache`).
pub fn create_with(kind: char, key: u64, fp: u64, create: impl Fn(Via) -> Result<vk::Pipeline, String>) -> Result<vk::Pipeline, String> {
    let create = |via: Via<'_>| {
        let source = via.binaries.is_none();
        let t = std::time::Instant::now();
        #[cfg(target_os = "linux")]
        let pending = if source && !READ_ONLY.load(Ordering::Relaxed) {
            match (DIR.get(), crash_identity()) {
                (Some(dir), Some(id)) => Some(compile_journal::Pending::begin(dir, id, kind, key, fp)?),
                _ => None,
            }
        } else { None };
        struct Active(bool);
        impl Drop for Active {
            fn drop(&mut self) { if self.0 { SOURCE_ACTIVE.fetch_sub(1, Ordering::SeqCst); } }
        }
        let active = Active(source);
        if source {
            SOURCE_ACTIVE.fetch_add(1, Ordering::SeqCst);
            SOURCE_STARTED.fetch_add(1, Ordering::SeqCst);
        }
        let result = create(via);
        #[cfg(target_os = "linux")]
        if let Some(pending) = pending {
            if let Err(e) = pending.finish(result.is_ok()) { crate::warn!("{e}"); }
        }
        if source {
            reuse_record(kind, key, fp, if result.is_ok() { "source" } else { "source_failed" }, t.elapsed().as_secs_f64() * 1000.0);
            SOURCE_CALLS.fetch_add(1, Ordering::Relaxed);
            SOURCE_NS.fetch_add(t.elapsed().as_nanos() as u64, Ordering::Relaxed);
        }
        drop(active);
        result
    };
    // (A pipeline whose compile crashed this driver before (`crashed.txt`) is not made again,
    // however it is asked for: its draw or dispatch goes without it. Until 2026-11-21 night only
    // background compiles left it out, so a draw needing it crashed every session that reached
    // it, and a preparation step begun again after such a crash would have met it again.)
    if crashed_skip(kind, key) {
        return Err(format!("{kind}{key:016x} left out: its compile crashed this graphics driver before (crashed.txt)"));
    }
    // (`DS_SIM_COMPILER_CRASH=<n>`, a test of a driver whose compiler crashes on some pipelines,
    // as NVIDIA's did on a few here and on the other PC: a pipeline whose key n divides faults
    // inside its compile, its note set, as that crash does. The fault handler records it in
    // crashed.txt and the process ends; each session meets it again until it is left out.)
    if let Some(n) = sim_compiler_crash() {
        if kind != 'q' && key % n == 0 {
            crate::warn!("pipeline cache: {kind}{key:016x} faults now in its compile (DS_SIM_COMPILER_CRASH={n})");
            crate::log::try_flush();
            // (`DS_SIM_COMPILER_CRASH_MODE=abort`: a fail-fast exit no fault handler sees, as the
            // sessions on NVIDIA's 621.40 ended: only the compiles in flight tell, `Inflight`.)
            if std::env::var("DS_SIM_COMPILER_CRASH_MODE").ok().as_deref() == Some("abort") {
                std::process::abort();
            }
            // (`DS_SIM_COMPILER_CRASH_MODE=breakpoint`: a breakpoint inside another module, as
            // NVIDIA's 621.40 compiler stops on one (status 0x80000003), then the compile goes
            // on as a debugger-only break does: under the port's host it is stepped over, without
            // a debugger it ends the process.)
            if std::env::var("DS_SIM_COMPILER_CRASH_MODE").ok().as_deref() == Some("breakpoint") {
                // SAFETY: a deliberate breakpoint exception, as the driver's own.
                #[cfg(windows)]
                unsafe { windows_sys::Win32::System::Diagnostics::Debug::DebugBreak() };
                #[cfg(target_os = "linux")]
                unsafe { libc::raise(libc::SIGTRAP) };
            } else {
                SIM_CRASHING.with(|c| c.set(true));
                // SAFETY: none; a deliberate access violation, as the driver's own crash.
                unsafe { std::ptr::write_volatile(std::ptr::without_provenance_mut::<u64>(8), 0) };
            }
        }
    }
    let Some(pb) = ctx().pbin.as_ref() else {
        return with_cache(kind, key, |cache| create(Via { cache, capture: false, binaries: None }));
    };
    let path = BIN_DIR.get().map(|dir| dir.join(format!("{kind}{key:016x}.pb")));
    // (Under the validation layer the driver refuses some graphics
    // pipelines' binaries that it takes without the layer (2026-11-08
    // night): such a boot keeps the files as they are.)
    // (Nor does the launcher's probe change them: a refused binary stays, the pipeline made without
    // capturing.)
    let keep = ctx().validation || READ_ONLY.load(Ordering::Relaxed);
    if let Some(p) = &path {
        let t = std::time::Instant::now();
        let stored = match std::fs::read(p) {
            Ok(b) => {
                let decoded = decode_pb(&b, &pb.global, fp);
                if decoded.is_none() {
                    let reason = pb_rejection(&b, &pb.global, fp);
                    match reason {
                        "old_schema" => &PB_SCHEMA, "other_driver" => &PB_OTHER_KEY,
                        "other_code_or_contract" => &PB_STALE, _ => &PB_INVALID,
                    }.fetch_add(1, Ordering::Relaxed);
                    reuse_record(kind, key, fp, reason, 0.0);
                    reuse_event(kind, key, fp, reason);
                }
                decoded
            }
            Err(e) => { PB_MISSING.fetch_add(1, Ordering::Relaxed); reuse_event(kind, key, fp, &format!("read: {e}")); None }
        };
        PB[7].fetch_add(t.elapsed().as_nanos() as u64, Ordering::Relaxed);
        if let Some(stored) = stored {
            let tb = std::time::Instant::now();
            let shared_ok = stored.iter().all(|part| match part {
                super::vk::pbin::Part::Shared(k) => pb.has_shared(k) || load_shared(k).is_some_and(|d| pb.share(k, &d).is_ok()),
                super::vk::pbin::Part::Own(..) => true,
            });
            // (`DS_SIM_PB_REFUSE=1`, a test: every stored binary refused, as the other PC's
            // driver refused the preparation's.)
            let shared_ok = shared_ok && !sim_refuse();
            let r = match shared_ok.then(|| pb.from_parts(&stored)) {
                Some(Ok((handles, own))) => {
                    PB[8].fetch_add(tb.elapsed().as_nanos() as u64, Ordering::Relaxed);
                    let tc = std::time::Instant::now();
                    CREATING.fetch_add(1, Ordering::Relaxed);
                    let r = create(Via { cache: vk::PipelineCache::null(), capture: false, binaries: Some(&handles) });
                    CREATING.fetch_sub(1, Ordering::Relaxed);
                    PB[9].fetch_add(tc.elapsed().as_nanos() as u64, Ordering::Relaxed);
                    let td = std::time::Instant::now();
                    pb.destroy(&own);
                    PB[10].fetch_add(td.elapsed().as_nanos() as u64, Ordering::Relaxed);
                    r
                }
                Some(Err(e)) => Err(format!("binaries from their data: {e}")),
                None if sim_refuse() => Err("refused (DS_SIM_PB_REFUSE)".to_string()),
                None => Err("a shared binary is missing".to_string()),
            };
            match r {
                Ok(pipe) => {
                    PB[0].fetch_add(1, Ordering::Relaxed);
                    PB[1].fetch_add(t.elapsed().as_nanos() as u64, Ordering::Relaxed);
                    reuse_record(kind, key, fp, "hit", t.elapsed().as_secs_f64() * 1000.0);
                    touch_later(p);
                    return Ok(pipe);
                }
                Err(e) => {
                    reuse_record(kind, key, fp, "refused", t.elapsed().as_secs_f64() * 1000.0);
                    reuse_event(kind, key, fp, &e);
                    if PB[2].fetch_add(1, Ordering::Relaxed) < 8 {
                        crate::warn!("pipeline cache: the binaries of {kind}{key:016x} were not used ({e}); made anew");
                    }
                    if !keep {
                        note_refusal();
                    }
                    if keep {
                        CREATING.fetch_add(1, Ordering::Relaxed);
                        let r = create(Via { cache: vk::PipelineCache::null(), capture: false, binaries: None });
                        CREATING.fetch_sub(1, Ordering::Relaxed);
                        return r;
                    }
                    let _ = std::fs::remove_file(p);
                }
            }
        }
    }
    // (Made with CAPTURE_DATA, without a cache object (VUID 09617), its
    // binaries then captured from it: what the driver's internal cache holds,
    // when it holds it, `pbin::Fns::probe`, gives the same.)
    CREATING.fetch_add(1, Ordering::Relaxed);
    sim_compile_delay();
    let r = create(Via { cache: vk::PipelineCache::null(), capture: true, binaries: None });
    CREATING.fetch_sub(1, Ordering::Relaxed);
    // (A compile whose driver stopped on a debug break that was stepped over: made for this
    // session, its binaries not kept, as the break may have been an assertion.)
    let stepped = stepped_in_compile();
    if stepped {
        crate::warn!("pipeline cache: {kind}{key:016x}'s binaries not kept: its compile had a driver debug break stepped over");
    }
    if let (Ok(pipe), Some(p), false) = (&r, &path, stepped) {
        let t = std::time::Instant::now();
        match pb.capture(*pipe, &|k| shared_path(k).is_some_and(|p| p.exists())) {
            Ok(bins) => {
                let enc = match encode_pb(&pb.global, fp, &bins) {
                    Ok(enc) => enc,
                    Err(e) => {
                        reuse_record(kind, key, fp, "storage_failed", 0.0);
                        reuse_event(kind, key, fp, &e);
                        PB[5].fetch_add(t.elapsed().as_nanos() as u64, Ordering::Relaxed);
                        return r;
                    }
                };
                // Explicit relay diagnostic: reconstruct from exactly the serialized parts and
                // replay while the original create-info, layouts and modules are still alive.
                // No internal-cache lookup is involved. Do not count these as ordinary hits.
                if std::env::var("DS_PB_VERIFY_CAPTURE").ok().as_deref() == Some("1") {
                    let started = std::time::Instant::now();
                    let parts = decode_pb(&enc, &pb.global, fp).expect("just encoded pipeline binary");
                    let ready = parts.iter().all(|p| match p {
                        super::vk::pbin::Part::Shared(k) => pb.has_shared(k) || load_shared(k).is_some_and(|d| pb.share(k, &d).is_ok()),
                        _ => true,
                    });
                    let result = if ready { pb.from_parts(&parts).map_err(|e| e.to_string()).and_then(|(all, own)| {
                        let result = create(Via { cache: vk::PipelineCache::null(), capture: false, binaries: Some(&all) });
                        pb.destroy(&own);
                        result
                    }) } else { Err("shared data unavailable".into()) };
                    let outcome = match result {
                        Ok(replay) => { unsafe { ctx().device.destroy_pipeline(replay, None) }; PB_FRESH[0].fetch_add(1, Ordering::Relaxed); "fresh_hit" }
                        Err(e) => { PB_FRESH[1].fetch_add(1, Ordering::Relaxed); reuse_event(kind, key, fp, &format!("fresh replay: {e}")); "fresh_refused" }
                    };
                    reuse_record(kind, key, fp, outcome, started.elapsed().as_secs_f64() * 1000.0);
                }
                if write_file(p, &enc) {
                    reuse_record(kind, key, fp, "captured", 0.0);
                    PB[3].fetch_add(1, Ordering::Relaxed);
                    PB[4].fetch_add(enc.len() as u64, Ordering::Relaxed);
                }
            }
            Err(e) => {
                if PB[6].fetch_add(1, Ordering::Relaxed) < 8 {
                    crate::warn!("pipeline cache: capturing the binaries of {kind}{key:016x} failed ({e})");
                }
            }
        }
        PB[5].fetch_add(t.elapsed().as_nanos() as u64, Ordering::Relaxed);
    }
    r
}

/// The code fingerprint of a pipeline made from translations: `fingerprint` of its parts, and
/// whether its modules declare maximal reconvergence (`exec::module_code`, `vk::max_reconvergence`:
/// binaries made without the declaration are not taken for modules with it, nor the other way).
pub fn translated_fingerprint(parts: &[&[u32]]) -> u64 {
    let h = fingerprint(parts);
    let h = if super::vk::max_reconvergence() { fingerprint(&[&[h as u32, (h >> 32) as u32], &[0x4d52_4331]]) } else { h };
    // (`vk::nan_preserve`: modules declaring it are other code.)
    if super::vk::nan_preserve() { fingerprint(&[&[h as u32, (h >> 32) as u32], &[0x4e50_5631, super::vk::nan_preserve_widths()]]) } else { h }
}

/// The variable create-info fields not covered by shader code or the graphics state key.
/// No handles, pointers or padding enter this fingerprint. Fixed state/chain changes bump
/// this schema. Viewport/scissor extents are dynamic and are deliberately absent.
pub fn creation_fingerprint(code: u64, bindings: &[vk::DescriptorSetLayoutBinding], ranges: &[vk::PushConstantRange], layout_flags: vk::DescriptorSetLayoutCreateFlags, flags: vk::PipelineCreateFlags, subgroups: vk::ShaderStageFlags) -> u64 {
    let mut words = vec![1, code as u32, (code >> 32) as u32, layout_flags.as_raw(), flags.as_raw(), subgroups.as_raw(), bindings.len() as u32];
    for b in bindings { words.extend([b.binding, b.descriptor_type.as_raw() as u32, b.descriptor_count, b.stage_flags.as_raw()]); }
    words.push(ranges.len() as u32);
    for r in ranges { words.extend([r.stage_flags.as_raw(), r.offset, r.size]); }
    fingerprint(&[&words])
}

/// A pipeline's code fingerprint (`create_with`): its stages' SPIR-V words
/// and any other words that shape it.
pub fn fingerprint(parts: &[&[u32]]) -> u64 {
    let mut h = 0x9E37_79B9_7F4A_7C15u64;
    for p in parts {
        h = (h.rotate_left(5) ^ p.len() as u64).wrapping_mul(0x517C_C1B7_2722_0A95);
        for &w in p.iter() {
            h = (h.rotate_left(5) ^ w as u64).wrapping_mul(0x517C_C1B7_2722_0A95);
        }
    }
    h
}

fn fnv64(b: &[u8]) -> u64 {
    b.iter().fold(0xcbf2_9ce4_8422_2325u64, |h, &x| (h ^ x as u64).wrapping_mul(0x100_0000_01b3))
}

fn hex(b: &[u8]) -> String {
    b.iter().map(|x| format!("{x:02x}")).collect()
}

/// Shared data is valid only within its device compatibility namespace, just like
/// the parent pipeline file. A matching binary key alone cannot validate old data.
fn shared_name(global: &[u8], key: &[u8]) -> String {
    format!("{}-{}.shared", hex(global), hex(key))
}
fn shared_path(key: &[u8]) -> Option<PathBuf> {
    let dir = BIN_DIR.get()?;
    let pb = ctx().pbin.as_ref()?;
    Some(dir.join(shared_name(&pb.global, key)))
}

/// Stores a shared binary's data once (`SHARED_MIN`).
fn store_shared(key: &[u8], data: &[u8]) -> bool {
    let Some(p) = shared_path(key) else { return false };
    store_shared_at(&p, data)
}
fn store_shared_at(p: &Path, data: &[u8]) -> bool {
    if load_shared_at(p).is_some() { return true; }
    if data.is_empty() { return false; }
    let mut w = data.to_vec();
    w.extend_from_slice(&fnv64(data).to_le_bytes());
    write_file(p, &w)
}

fn load_shared(key: &[u8]) -> Option<Vec<u8>> {
    load_shared_at(&shared_path(key)?)
}
fn load_shared_at(path: &Path) -> Option<Vec<u8>> {
    let mut b = std::fs::read(path).ok()?;
    if b.len() < 8 {
        return None;
    }
    let tail = b.split_off(b.len() - 8);
    if fnv64(&b) != u64::from_le_bytes(tail.try_into().ok()?) {
        // (Damaged: removed, so the next capture stores it again; not by the launcher's probe.)
        if !READ_ONLY.load(Ordering::Relaxed) {
            let _ = std::fs::remove_file(path);
        }
        return None;
    }
    Some(b)
}

/// A pipeline binary file (`PB_MAGIC`); shared data stored once apart.
fn encode_pb(global: &[u8], fp: u64, bins: &[(Vec<u8>, Vec<u8>)]) -> Result<Vec<u8>, String> {
    encode_pb_with(global, fp, bins, store_shared)
}
fn encode_pb_with(global: &[u8], fp: u64, bins: &[(Vec<u8>, Vec<u8>)], store: impl Fn(&[u8], &[u8]) -> bool) -> Result<Vec<u8>, String> {
    let mut w = Vec::with_capacity(64 + bins.iter().map(|b| b.0.len() + b.1.len().min(SHARED_MIN) + 12).sum::<usize>());
    w.extend_from_slice(PB_MAGIC);
    w.extend_from_slice(&(global.len() as u32).to_le_bytes());
    w.extend_from_slice(global);
    w.extend_from_slice(&fp.to_le_bytes());
    w.extend_from_slice(&(bins.len() as u32).to_le_bytes());
    for (k, d) in bins {
        w.extend_from_slice(&(k.len() as u32).to_le_bytes());
        w.extend_from_slice(k);
        if d.is_empty() || d.len() >= SHARED_MIN {
            // (Empty: a shared binary stored before, `pbin::Fns::capture`.)
            if !store(k, d) { return Err(format!("shared binary {} could not be stored/read; pipeline record not written", hex(k))); }
            w.extend_from_slice(&u64::MAX.to_le_bytes());
        } else {
            w.extend_from_slice(&(d.len() as u64).to_le_bytes());
            w.extend_from_slice(d);
        }
    }
    let h = fnv64(&w);
    w.extend_from_slice(&h.to_le_bytes());
    Ok(w)
}

/// `encode_pb`'s inverse; None for another global key, other code (`fp`)
/// or a damaged file.
fn pb_rejection(b: &[u8], global: &[u8], fp: u64) -> &'static str {
    if b.get(..8).is_some_and(|v| v == b"DSPBIN03") { return "old_schema"; }
    if b.len() < 28 || b.get(..8) != Some(PB_MAGIC) { return "invalid"; }
    let (body, tail) = b.split_at(b.len() - 8);
    if fnv64(body) != u64::from_le_bytes(tail.try_into().unwrap()) { return "invalid"; }
    let n = u32::from_le_bytes(b[8..12].try_into().unwrap()) as usize;
    let Some(g) = body.get(12..12usize.saturating_add(n)) else { return "invalid" };
    if g != global { return "other_driver"; }
    let Some(f) = body.get(12+n..20+n) else { return "invalid" };
    if u64::from_le_bytes(f.try_into().unwrap()) != fp { return "other_code_or_contract"; }
    "invalid"
}

fn decode_pb(b: &[u8], global: &[u8], fp: u64) -> Option<Vec<super::vk::pbin::Part>> {
    if b.len() < 8 + 4 + 4 + 8 || &b[..8] != PB_MAGIC {
        return None;
    }
    let (body, tail) = b.split_at(b.len() - 8);
    if fnv64(body) != u64::from_le_bytes(tail.try_into().ok()?) {
        return None;
    }
    let mut at = 8usize;
    let mut take = |n: usize| -> Option<&[u8]> {
        let s = body.get(at..at.checked_add(n)?)?;
        at += n;
        Some(s)
    };
    let gl = u32::from_le_bytes(take(4)?.try_into().ok()?) as usize;
    if take(gl)? != global {
        return None;
    }
    if u64::from_le_bytes(take(8)?.try_into().ok()?) != fp {
        return None;
    }
    let n = u32::from_le_bytes(take(4)?.try_into().ok()?) as usize;
    if n == 0 || n > 64 {
        return None;
    }
    let mut out = Vec::with_capacity(n);
    for _ in 0..n {
        let kl = u32::from_le_bytes(take(4)?.try_into().ok()?) as usize;
        if kl == 0 || kl > super::vk::pbin::MAX_KEY { return None; }
        let k = take(kl)?.to_vec();
        let dl = u64::from_le_bytes(take(8)?.try_into().ok()?);
        if dl == u64::MAX {
            out.push(super::vk::pbin::Part::Shared(k));
        } else {
            let d = take(dl as usize)?.to_vec();
            out.push(super::vk::pbin::Part::Own(k, d));
        }
    }
    if at != body.len() { return None; }
    Some(out)
}

/// One status line.
pub fn status() -> String {
    let s: Vec<u64> = STATS.iter().map(|c| c.load(Ordering::Relaxed)).collect();
    let l: Vec<u64> = LAPS.iter().map(|c| c.load(Ordering::Relaxed)).collect();
    let n = l[7].max(1) as f64 * 1000.0;
    format!(
        "    pipeline cache files: {} pipelines from their file, {} written ({} MiB; {} of them after the driver refused their data), {} pruned; {} creations, us each: file read {:.0}, decode {:.0}, cache object {:.0}, creation {:.0}, its size or data {:.0}, write or last-use mark {:.0}, destroy {:.0}; shader modules before them: {} at {:.0} us{}\n",
        s[0],
        s[1],
        s[2] >> 20,
        REFRESHED.load(Ordering::Relaxed),
        s[3],
        l[7],
        l[6] as f64 / n,
        l[0] as f64 / n,
        l[1] as f64 / n,
        l[2] as f64 / n,
        l[3] as f64 / n,
        l[4] as f64 / n,
        l[5] as f64 / n,
        MODULES[0].load(Ordering::Relaxed),
        MODULES[1].load(Ordering::Relaxed) as f64 / MODULES[0].load(Ordering::Relaxed).max(1) as f64 / 1000.0,
        match (CRASH_SKIPS.load(Ordering::Relaxed), CRASHED.get().map_or(0, |s| s.read().len())) {
            (0, 0) => String::new(),
            (n, m) => format!("; {n} background compiles left out, their pipeline having crashed this driver ({m} listed in crashed.txt)"),
        }
    ) + &{
        let b: Vec<u64> = PB.iter().map(|c| c.load(Ordering::Relaxed)).collect();
        if super::vk::try_ctx().is_none_or(|c| c.pbin.is_none()) {
            String::new()
        } else {
            format!(
                "    pipeline binaries: {} pipelines made from them ({:.0} us each: file {:.0}, binary objects {:.0}, pipeline {:.0}, destroy {:.0}), {} refused or failed, {} of other code; {} captured ({} MiB, {:.0} us each), {} captures failed\n",
                b[0],
                b[1] as f64 / b[0].max(1) as f64 / 1000.0,
                b[7] as f64 / b[0].max(1) as f64 / 1000.0,
                b[8] as f64 / b[0].max(1) as f64 / 1000.0,
                b[9] as f64 / b[0].max(1) as f64 / 1000.0,
                b[10] as f64 / b[0].max(1) as f64 / 1000.0,
                b[2],
                PB_STALE.load(Ordering::Relaxed),
                b[3],
                b[4] >> 20,
                b[5] as f64 / (b[3] + b[6]).max(1) as f64 / 1000.0,
                b[6]
            )
        }
    }
}

/// `ThreadId::as_u64` is unstable: a stable number per thread from its debug
/// form.
trait ThreadIdNumber {
    fn as_u64_compat(&self) -> u64;
}

impl ThreadIdNumber for std::thread::ThreadId {
    fn as_u64_compat(&self) -> u64 {
        format!("{self:?}").bytes().filter(u8::is_ascii_digit).fold(0u64, |a, b| a.wrapping_mul(10).wrapping_add((b - b'0') as u64))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn fake_driver_capture_store_load_replay() {
        // Fake driver's ordered capture: private code plus a large shared dictionary.
        // Exercise the production serializer, atomic writer and shared-data reader on disk.
        let root = std::env::temp_dir().join(format!("regaliant-pb-roundtrip-{}", std::process::id()));
        std::fs::create_dir_all(&root).unwrap();
        let global = [1; 36];
        let bins = vec![(vec![2; 8], vec![5; 32]), (vec![3; 8], vec![7; SHARED_MIN])];
        let store = |key: &[u8], bytes: &[u8]| store_shared_at(&root.join(shared_name(&global, key)), bytes);
        let encoded = encode_pb_with(&global, 123, &bins, store).unwrap();
        let path = root.join("c123.pb");
        assert!(write_file(&path, &encoded));
        let parts = decode_pb(&std::fs::read(&path).unwrap(), &global, 123).unwrap();
        let replay: Vec<_> = parts.into_iter().map(|p| match p {
            super::super::vk::pbin::Part::Own(k, d) => (k, d),
            super::super::vk::pbin::Part::Shared(k) => { let d = load_shared_at(&root.join(shared_name(&global, &k))).unwrap(); (k, d) }
        }).collect();
        assert_eq!(replay, bins, "fake driver accepts only its original ordered keys/data");
        // A capture that skipped already-known shared bytes still reloads the stored data.
        let mut known = bins.clone(); known[1].1.clear();
        assert!(encode_pb_with(&global, 123, &known, store).is_ok());
        // Neither a failed shared write nor missing known data may publish a dangling record.
        assert!(encode_pb_with(&global, 123, &bins, |_, _| false).is_err());
        std::fs::remove_file(root.join(shared_name(&global, &bins[1].0))).unwrap();
        assert!(encode_pb_with(&global, 123, &known, store).is_err());
        std::fs::remove_dir_all(root).unwrap();
    }

    #[test]
    fn shared_data_is_scoped_to_device_contract() {
        assert_ne!(shared_name(&[1, 2], &[3, 4]), shared_name(&[1, 5], &[3, 4]));
        assert_ne!(shared_name(&[1, 2], &[3, 4]), shared_name(&[1, 2], &[3, 5]));
        assert_eq!(shared_name(&[1, 2], &[3, 4]), "0102-0304.shared");
    }

    #[test]
    fn binary_file_preserves_order_and_rejects_stale_contracts() {
        let bins = vec![(vec![7; 4], vec![1; 40]), (vec![3; 8], vec![2; 20])];
        let bytes = encode_pb(&[9; 28], 42, &bins).unwrap();
        let parts = decode_pb(&bytes, &[9; 28], 42).unwrap();
        for (part, (key, data)) in parts.iter().zip(&bins) {
            match part { super::super::vk::pbin::Part::Own(k, d) => { assert_eq!(k, key); assert_eq!(d, data); }, _ => panic!("not shared") }
        }
        assert!(decode_pb(&bytes, &[9; 28], 43).is_none());
        assert_eq!(pb_rejection(&bytes, &[9; 28], 43), "other_code_or_contract");
        assert_eq!(pb_rejection(&bytes, &[8; 28], 42), "other_driver");
        let mut old = bytes.clone(); old[..8].copy_from_slice(b"DSPBIN03");
        assert_eq!(pb_rejection(&old, &[9; 28], 42), "old_schema");
        let mut corrupt = bytes; corrupt[30] ^= 1;
        assert_eq!(pb_rejection(&corrupt, &[9; 28], 42), "invalid");
    }

    #[test]
    fn pipeline_contract_includes_layout_flags_and_subgroup_size() {
        let b = [vk::DescriptorSetLayoutBinding::default().binding(2).descriptor_count(1).descriptor_type(vk::DescriptorType::STORAGE_BUFFER).stage_flags(vk::ShaderStageFlags::COMPUTE)];
        let f = |flags, sg, binding: &[vk::DescriptorSetLayoutBinding]| creation_fingerprint(7, binding, &[], vk::DescriptorSetLayoutCreateFlags::empty(), flags, sg);
        let flags = vk::PipelineCreateFlags::DISPATCH_BASE;
        let h = f(flags, vk::ShaderStageFlags::COMPUTE, &b);
        assert_ne!(h, f(flags, vk::ShaderStageFlags::empty(), &b));
        assert_ne!(h, f(vk::PipelineCreateFlags::empty(), vk::ShaderStageFlags::COMPUTE, &b));
        let mut other = b; other[0].descriptor_count = 2;
        assert_ne!(h, f(flags, vk::ShaderStageFlags::COMPUTE, &other));
        assert_eq!(h, f(flags, vk::ShaderStageFlags::COMPUTE, &b));
    }

    /// Files decoded one after another into one buffer (the block kept in
    /// place when the offset repeats) come back as decoded fresh.
    #[test]
    fn decode_reuses_the_block() {
        let block: Vec<u8> = (0..(MIN_BLOCK + 3000)).map(|i| (i * 7 + i / 300) as u8).collect();
        let h = hash64(&block);
        BLOCKS.lock().push((h, std::sync::Arc::new(block.clone())));
        let file = |pre: &[u8], post: &[u8]| {
            let mut f = Vec::new();
            f.extend_from_slice(MAGIC);
            f.extend_from_slice(&(pre.len() as u32).to_le_bytes());
            f.extend_from_slice(&(block.len() as u32).to_le_bytes());
            f.extend_from_slice(&h.to_le_bytes());
            f.extend_from_slice(pre);
            f.extend_from_slice(post);
            f
        };
        let whole = |pre: &[u8], post: &[u8]| [pre, &block[..], post].concat();
        let mut d = Decoded::default();
        for (pre, post) in [(&[1u8; 36][..], &[2u8; 500][..]), (&[3u8; 36][..], &[4u8; 20][..]), (&[5u8; 40][..], &[6u8; 900][..]), (&[7u8; 36][..], &[8u8; 3000][..])] {
            assert!(decode_into(&file(pre, post), &mut d));
            assert_eq!(d.data, whole(pre, post));
        }
    }

    /// Two pipelines' data sharing a 1.5 MiB block at different offsets:
    /// `discover` finds it, and stored without it the data comes back whole.
    #[test]
    fn shared_block_round_trip() {
        // xorshift64 bytes, with a short repeated pattern at the window the
        // search starts from (as the driver's data may hold).
        let mut x = 0x9E37_79B9_7F4A_7C15u64;
        let mut block: Vec<u8> = (0..(3 << 19))
            .map(|_| {
                x ^= x << 13;
                x ^= x >> 7;
                x ^= x << 17;
                x as u8
            })
            .collect();
        let pat: Vec<u8> = block[700_000..700_200].to_vec();
        block[100..300].copy_from_slice(&pat);
        let a: Vec<u8> = [&b"header-a"[..], &block, &[7u8; 300]].concat();
        let b: Vec<u8> = [&[9u8; 5000][..], &block, &b"tail-b"[..]].concat();
        let found = discover(&a, &b).expect("block");
        assert_eq!(found, block);
        let h = hash64(&found);
        BLOCKS.lock().push((h, Arc::new(found)));
        for data in [&a, &b] {
            let enc = encode(data);
            assert!(enc.starts_with(MAGIC) && enc.len() < data.len() / 100);
            assert_eq!(decode(&enc).as_deref(), Some(&data[..]));
        }
        // Data without the block is stored whole.
        let c = b"short pipeline data".to_vec();
        assert_eq!(decode(&encode(&c)).as_deref(), Some(&c[..]));
    }
}

/// `dsport --pb-test` (diagnosis): pipeline binaries on one small compute
/// pipeline with no title running: made, its binaries probed from the
/// driver's internal cache, then made with CAPTURE_DATA and its binaries
/// captured, each step logged before its call (the driver faulted in both
/// in runs, never under the validation layer).
/// `DS_SIM_COMPILE_DELAY_MS=<ms>[:<every>:<long ms>]` (a test, 2026-11-20): every pipeline the
/// driver compiles (not one from the store) takes that much longer, and every `every`-th one
/// `long ms` instead: a slower PC's driver (the other PC's took ~0.3 s a compute pipeline in its
/// menus, 3-21 s for its largest, 2026-11-19).
pub fn sim_compile_delay() {
    static D: OnceLock<Option<(u64, u64, u64)>> = OnceLock::new();
    static N: AtomicU64 = AtomicU64::new(0);
    let d = *D.get_or_init(|| {
        let v = std::env::var("DS_SIM_COMPILE_DELAY_MS").ok()?;
        let mut it = v.split(':').map(|x| x.trim().parse::<u64>().ok());
        let ms = it.next()??;
        Some((ms, it.next().flatten().unwrap_or(0), it.next().flatten().unwrap_or(ms)))
    });
    if let Some((ms, every, long)) = d {
        let n = N.fetch_add(1, Ordering::Relaxed) + 1;
        let w = if every > 0 && n % every == 0 { long } else { ms };
        std::thread::sleep(std::time::Duration::from_millis(w));
    }
}

/// `DS_SIM_PB_REFUSE=1` (a test of the fallback): stored binaries are treated as refused.
fn sim_refuse() -> bool {
    static ON: OnceLock<bool> = OnceLock::new();
    *ON.get_or_init(|| std::env::var("DS_SIM_PB_REFUSE").ok().as_deref() == Some("1"))
}

/// This device's name and pipeline cache id (`vk::init`): the store's binary mode is theirs.
static IDENTITY: OnceLock<(String, String)> = OnceLock::new();

pub fn set_identity(device: &str, cache_id: &str) {
    crate::info!("pipeline store identity: {device}; {cache_id}; {:?}", DIR.get());
    let _ = IDENTITY.set((device.to_string(), cache_id.to_string()));
    #[cfg(target_os = "linux")]
    if !READ_ONLY.load(Ordering::Relaxed) {
        if let Some(path) = crashed_file() { crate::platform::signals::compiler_crash_file(&path); }
    }
}

/// The store's pipeline-binary mode (`binaries-mode.txt` in the store, 2026-11-20): `binaries =
/// off` for this device and pipeline cache id when its driver refused binaries another process
/// made (a session's own count, `note_refusal`, or the launcher's check, `pb_check`). The store
/// then keeps cache data with the driver's own cache on, as before 2026-11-08 (the other PC's
/// driver refused 259 of the 262 binaries its preparation had made, 2026-11-19).
fn mode_file() -> Option<PathBuf> {
    DIR.get().map(|d| d.join("binaries-mode.txt"))
}

/// Why the store's binaries are off for this device and cache id (None: they are not).
pub fn binaries_off_for(device: &str, cache_id: &str) -> Option<String> {
    let t = std::fs::read_to_string(mode_file()?).ok()?;
    let get = |k: &str| t.lines().find_map(|l| l.split_once('=').filter(|(a, _)| a.trim() == k).map(|(_, b)| b.trim().to_string()));
    (get("binaries").as_deref() == Some("off") && get("device").as_deref() == Some(device) && get("cache_id").as_deref() == Some(cache_id)).then(|| get("why").unwrap_or_default())
}

/// Writes the mode file: binaries off for this device and cache id, and why.
pub fn turn_binaries_off(why: &str) -> bool {
    let (Some(p), Some((dev, id))) = (mode_file(), IDENTITY.get()) else { return false };
    let t = format!("# The pipeline store keeps cache data, not binaries, for this graphics card and driver (written by the game)\nbinaries = off\ndevice = {dev}\ncache_id = {id}\nwhy = {why}\n");
    std::fs::write(p, t).is_ok()
}

/// A session's own count of refused binaries: once 24 were refused and at most a quarter as many
/// used, the next sessions keep cache data (`turn_binaries_off`).
fn note_refusal() {
    static DONE: std::sync::atomic::AtomicBool = std::sync::atomic::AtomicBool::new(false);
    let (made, refused) = (PB[0].load(Ordering::Relaxed), PB[2].load(Ordering::Relaxed));
    if refused >= 24 && made * 4 <= refused && !DONE.swap(true, Ordering::Relaxed) {
        let why = format!("a session's driver refused {refused} of its stored binaries and took {made}");
        if turn_binaries_off(&why) {
            crate::warn!("pipeline cache: the driver refused {refused} stored binaries and took {made}: the next sessions keep cache data instead (binaries-mode.txt)");
        }
    }
}

/// `--pb-check write|read` (the launcher's check before a preparation, 2026-11-20): whether this
/// PC's driver takes what another process stored, as a game session must take the preparation's.
/// `write` makes the check's four compute pipelines through the store (their binaries, or with
/// binaries off their cache data, kept as any pipeline's); `read`, in the next process, makes
/// them again. Exit status of `read`: 0 when at least three came from the store without a
/// compile, 20 when the driver compiled them again (binaries refused, or cache data that did not
/// serve), 21 when nothing was stored; 1 Vulkan failed.
pub fn pb_check(phase: &str) -> i32 {
    // (No usable graphics card: the launcher ends the preparation at once, not after the steps
    // after it failed too.)
    if let Err(e) = super::vk::init() {
        crate::error!("pb-check: Vulkan: {e}");
        println!("pb-check = {}", crate::EXIT_NO_GPU);
        return crate::EXIT_NO_GPU as i32;
    }
    let c = ctx();
    // Linux's supervisor captures stdout, not the runtime's info-only log sink.
    // This accompanies the existing stdout result protocol and keeps driver changes visible.
    #[cfg(target_os = "linux")]
    if let Some((device, id)) = IDENTITY.get() {
        println!("pipeline store identity: {device}; {id}");
    }
    // (`probe`, the launcher's at its start once a store is prepared: a read that changes nothing,
    // to find a store whose pipelines the driver no longer takes; 2026-11-20 night: this PC's
    // driver refused ~3,400 stored pipelines, binaries and cache data alike, after a Windows
    // session event, with the same driver and cache id.)
    let read = phase == "read" || phase == "probe";
    let binaries = c.pbin.is_some();
    let code = rdna::kernels::timestamp_store_many();
    let Ok(module) = (unsafe { c.device.create_shader_module(&vk::ShaderModuleCreateInfo::default().code(&code), None) }) else { return 1 };
    let (mut served, mut compiled, mut missing) = (0u32, 0u32, 0u32);
    let (mut errors, mut lost) = (0u32, false);
    for k in 0..4u32 {
        // (Four pipelines: the kernel's two buffers and k unused ones, so four layouts.)
        let bindings: Vec<vk::DescriptorSetLayoutBinding> = (0..2 + k)
            .map(|b| vk::DescriptorSetLayoutBinding::default().binding(b).descriptor_type(vk::DescriptorType::STORAGE_BUFFER).descriptor_count(1).stage_flags(vk::ShaderStageFlags::COMPUTE))
            .collect();
        let Ok(set_layout) = (unsafe { c.device.create_descriptor_set_layout(&vk::DescriptorSetLayoutCreateInfo::default().bindings(&bindings), None) }) else { return 1 };
        let layouts = [set_layout];
        let ranges = [vk::PushConstantRange { stage_flags: vk::ShaderStageFlags::COMPUTE, offset: 0, size: 4 * (4 + 2 * rdna::kernels::TS_MANY) }];
        let Ok(layout) = (unsafe { c.device.create_pipeline_layout(&vk::PipelineLayoutCreateInfo::default().set_layouts(&layouts).push_constant_ranges(&ranges), None) }) else { return 1 };
        let key = 0x5042_4348_4b00_0000u64 | k as u64;
        let fp = fingerprint(&[&code, &[k]]);
        // (This mode's file only: one the other mode left says nothing about this one.)
        let stored_before = stored_cheap('q', key);
        let pb = [PB[0].load(Ordering::Relaxed), PB[2].load(Ordering::Relaxed), PB_OTHER_KEY.load(Ordering::Relaxed)];
        let stage = vk::PipelineShaderStageCreateInfo::default().stage(vk::ShaderStageFlags::COMPUTE).module(module).name(c"main");
        // (Cache data read: the driver must serve it without compiling, else the pipeline is not
        // made at all: pipelineCreationCacheControl.)
        let fail = read && !binaries;
        let flags = if fail { vk::PipelineCreateFlags::FAIL_ON_PIPELINE_COMPILE_REQUIRED } else { vk::PipelineCreateFlags::empty() };
        let ci = vk::ComputePipelineCreateInfo::default().stage(stage).layout(layout).flags(flags);
        let make = |via: Via| -> Result<vk::Pipeline, String> {
            let mut ci = ci;
            let mut f2 = super::vk::pbin::creation_flags(ci.flags, via.capture);
            if via.capture || via.binaries.is_some() { ci = ci.push_next(&mut f2); }
            let bi;
            if let Some(b) = via.binaries {
                bi = super::vk::pbin::BinaryInfo { s_type: super::vk::pbin::ST_BINARY_INFO, p_next: ci.p_next, binary_count: b.len() as u32, p_binaries: b.as_ptr() };
                ci.p_next = &bi as *const super::vk::pbin::BinaryInfo as *const std::ffi::c_void;
            }
            unsafe { c.device.create_compute_pipelines(via.cache, &[ci], None) }.map(|v| v[0]).map_err(|(_, e)| format!("{e}"))
        };
        let r = create_with('q', key, fp, make);
        let from_bins = PB[0].load(Ordering::Relaxed) > pb[0];
        let refused = PB[2].load(Ordering::Relaxed) > pb[1];
        let outcome = match (&r, binaries) {
            _ if !read => "made",
            (Ok(_), true) if from_bins => "from its binaries",
            (Ok(_), true) if refused => "its binaries refused, compiled again",
            // (Stored under another key of the driver's: the store is stale, as when refused. Stored
            // but for another fingerprint (the check's kernel changed in a new version) or damaged:
            // the check's own file is what does not hold, so it counts as not stored; the reviews
            // of 2026-11-21.)
            (Ok(_), true) if stored_before && PB_OTHER_KEY.load(Ordering::Relaxed) > pb[2] => "its stored binaries are for another driver key, compiled again",
            (Ok(_), true) if stored_before => "its stored file is stale or damaged, compiled",
            (Ok(_), true) => "no binaries stored, compiled",
            (Ok(_), false) => "served by its cache data",
            (Err(e), _) if e.contains("DEVICE_LOST") => "the device lost",
            (Err(e), false) if e.contains("COMPILE_REQUIRED") && stored_before => "its cache data did not serve it",
            (Err(e), false) if e.contains("COMPILE_REQUIRED") => "no cache data stored",
            (Err(_), _) => "failed",
        };
        crate::info!("pb-check {phase}: pipeline {k} ({}): {outcome}{}", if binaries { "binaries" } else { "cache data" }, r.as_ref().err().map(|e| format!(" ({e})")).unwrap_or_default());
        // (Errors are not refusals: a driver that fails pipelines, or a lost device, says nothing
        // about the store's way; the review of 2026-11-20: they had turned binaries off for good.)
        match outcome {
            "the device lost" => lost = true,
            "failed" => errors += 1,
            _ => {}
        }
        if read {
            match outcome {
                "from its binaries" | "served by its cache data" => served += 1,
                "no binaries stored, compiled" | "no cache data stored" | "its stored file is stale or damaged, compiled" => missing += 1,
                "the device lost" | "failed" => {}
                _ => compiled += 1,
            }
        }
        if let Ok(p) = r {
            unsafe { c.device.destroy_pipeline(p, None) };
        }
        unsafe {
            c.device.destroy_pipeline_layout(layout, None);
            c.device.destroy_descriptor_set_layout(set_layout, None);
        }
    }
    unsafe { c.device.destroy_shader_module(module, None) };
    if lost {
        crate::error!("pb-check {phase}: the graphics device was lost");
        println!("pb-check = 5");
        return crate::EXIT_DEVICE_LOST as i32;
    }
    if !read {
        // (The store not written, its drive full or its folder refusing writes: not the driver's
        // doing. Its own status, so the launcher says why and does not conclude that this PC's
        // driver keeps nothing (until 2026-11-20 night the read would have found nothing stored
        // and recorded "cannot help" for the driver for good).)
        let (fails, written) = (WRITE_FAILS.load(Ordering::Relaxed), WRITES_OK.load(Ordering::Relaxed));
        if fails > 0 && (store_full() || written == 0) {
            let code = if store_full() { crate::prepplay::EXIT_DISK_FULL } else { crate::prepplay::EXIT_UNWRITABLE };
            crate::error!("pb-check write: the store not written ({fails} files failed, {written} written, {} MiB free)", store_free_bytes().unwrap_or(0) >> 20);
            println!("pb-check = {code}");
            return code;
        }
        crate::info!("pb-check write: done ({}; {errors} failed)", if binaries { "binaries" } else { "cache data" });
        return if errors > 0 { 21 } else { 0 };
    }
    let code = if served >= 3 {
        0
    } else if compiled > 0 && errors == 0 {
        20
    } else if phase == "probe" && missing == 4 {
        // (The check's pipelines are not in the store: the probe cannot tell; the launcher has the
        // check run again at the next preparation.)
        22
    } else {
        21
    };
    // (Binaries refused across processes: the store keeps cache data from now on, the mode file
    // written here, where the driver's cache id is known; the launcher checks that way next.)
    if code == 20 && binaries && phase == "read" && turn_binaries_off(&format!("the launcher's check: the driver refused {compiled} of 4 binaries another process made")) {
        crate::warn!("pb-check read: binaries refused across processes: the store keeps cache data from now on (binaries-mode.txt)");
    }
    crate::info!("pb-check {phase}: {served} served from the store, {compiled} compiled again, {missing} not stored ({}): exit {code}", if binaries { "binaries" } else { "cache data" });
    println!("pb-check = {code}");
    code
}

pub fn pb_test() -> i32 {
    if let Err(e) = super::vk::init() {
        crate::error!("pb-test: Vulkan: {e}");
        return 1;
    }
    let c = ctx();
    let Some(pb) = c.pbin.as_ref() else {
        crate::error!("pb-test: pipeline binaries are off (DS_PIPELINE_BINARIES=1)");
        return 1;
    };
    unsafe {
        let bindings: Vec<vk::DescriptorSetLayoutBinding> = (0..2)
            .map(|b| vk::DescriptorSetLayoutBinding::default().binding(b).descriptor_type(vk::DescriptorType::STORAGE_BUFFER).descriptor_count(1).stage_flags(vk::ShaderStageFlags::COMPUTE))
            .collect();
        let Ok(set_layout) = c.device.create_descriptor_set_layout(&vk::DescriptorSetLayoutCreateInfo::default().bindings(&bindings), None) else { return 1 };
        let layouts = [set_layout];
        let ranges = [vk::PushConstantRange { stage_flags: vk::ShaderStageFlags::COMPUTE, offset: 0, size: 4 * (4 + 2 * rdna::kernels::TS_MANY) }];
        let Ok(layout) = c.device.create_pipeline_layout(&vk::PipelineLayoutCreateInfo::default().set_layouts(&layouts).push_constant_ranges(&ranges), None) else { return 1 };
        let code = rdna::kernels::timestamp_store_many();
        let Ok(module) = c.device.create_shader_module(&vk::ShaderModuleCreateInfo::default().code(&code), None) else { return 1 };
        let stage = vk::PipelineShaderStageCreateInfo::default().stage(vk::ShaderStageFlags::COMPUTE).module(module).name(c"main");
        let ci = vk::ComputePipelineCreateInfo::default().stage(stage).layout(layout);
        crate::info!("pb-test: making the pipeline");
        let p1 = c.device.create_compute_pipelines(vk::PipelineCache::null(), &[ci], None);
        crate::info!("pb-test: made: {:?}", p1.as_ref().map(|v| v[0]).map_err(|e| e.1));
        if std::env::var("DS_PB_TEST").ok().as_deref() != Some("capture") {
            crate::info!("pb-test: probing its binaries by its create info");
            let r = pb.probe(&ci as *const vk::ComputePipelineCreateInfo as *const std::ffi::c_void);
            crate::info!("pb-test: probed: {:?}", r.as_ref().map(|v| v.iter().map(|b| (b.0.len(), b.1.len())).collect::<Vec<_>>()));
        }
        let mut f2 = vk::PipelineCreateFlags2CreateInfoKHR::default().flags(vk::PipelineCreateFlags2KHR::from_raw(super::vk::pbin::CAPTURE_DATA));
        let ci2 = ci.push_next(&mut f2);
        crate::info!("pb-test: making it with CAPTURE_DATA");
        let p2 = c.device.create_compute_pipelines(vk::PipelineCache::null(), &[ci2], None);
        crate::info!("pb-test: made: {:?}", p2.as_ref().map(|v| v[0]).map_err(|e| e.1));
        if let Ok(v) = &p2 {
            crate::info!("pb-test: capturing its binaries");
            let r = pb.capture(v[0], &|_| false);
            crate::info!("pb-test: captured: {:?}", r.as_ref().map(|v| v.iter().map(|b| (b.0.len(), b.1.len())).collect::<Vec<_>>()));
        }
        c.device.destroy_shader_module(module, None);
    }
    crate::info!("pb-test: done");
    0
}
