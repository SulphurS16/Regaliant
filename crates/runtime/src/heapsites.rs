//! `DS_HEAP_SITES=<min bytes>` (diagnosis): the host heap's live
//! allocations of at least that many bytes by allocation site (the
//! requesting host frames), the largest sites logged every 20 seconds with
//! their frames named. The bookkeeping lives in static tables (no heap use
//! inside the allocator) under a spin lock; allocations made before `main`
//! read the variable go untracked, and frees of untracked blocks are
//! ignored.

use std::cell::UnsafeCell;
use std::sync::atomic::{AtomicBool, AtomicU64, AtomicUsize, Ordering};

/// The smallest tracked size (0: off).
pub static MIN: AtomicUsize = AtomicUsize::new(0);

const FRAMES: usize = 14;
const SITES: usize = 8192;
const PTRS: usize = 1 << 21;
const EMPTY: usize = 0;
const GONE: usize = 1;

#[derive(Clone, Copy)]
struct Site {
    hash: u64,
    frames: [usize; FRAMES],
    n: u32,
    live: u64,
    count: u64,
    allocs: u64,
}

const NO_SITE: Site = Site { hash: 0, frames: [0; FRAMES], n: 0, live: 0, count: 0, allocs: 0 };

struct Table {
    /// (pointer, size, site + 1).
    ptrs: [(usize, usize, u32); PTRS],
    sites: [Site; SITES],
    nsites: usize,
}

struct Shared(UnsafeCell<Table>);
// (Accessed only under `LOCK`.)
unsafe impl Sync for Shared {}

static T: Shared = Shared(UnsafeCell::new(Table { ptrs: [(EMPTY, 0, 0); PTRS], sites: [NO_SITE; SITES], nsites: 0 }));
static LOCK: AtomicBool = AtomicBool::new(false);
/// [allocations not tracked (tables full), frees of blocks not found].
static MISSED: [AtomicU64; 2] = [AtomicU64::new(0), AtomicU64::new(0)];

fn lock() {
    while LOCK.compare_exchange_weak(false, true, Ordering::Acquire, Ordering::Relaxed).is_err() {
        std::hint::spin_loop();
    }
}

fn unlock() {
    LOCK.store(false, Ordering::Release);
}

fn slot(p: usize) -> usize {
    (((p >> 4) as u64).wrapping_mul(0x9E37_79B9_7F4A_7C15) >> (64 - 21)) as usize
}

/// Whether a block of `size` bytes is tracked.
#[inline]
pub fn tracked(size: usize) -> bool {
    let m = MIN.load(Ordering::Relaxed);
    m != 0 && size >= m
}

/// A tracked block was allocated at `p` (frames taken here).
#[cold]
pub fn note_alloc(p: *mut u8, size: usize) {
    if p.is_null() {
        return;
    }
    let mut frames = [std::ptr::null_mut::<std::ffi::c_void>(); FRAMES];
    let n = unsafe { crate::platform::os::Win32::System::Diagnostics::Debug::RtlCaptureStackBackTrace(2, FRAMES as u32, frames.as_mut_ptr(), std::ptr::null_mut()) } as usize;
    let mut h: u64 = 0xcbf2_9ce4_8422_2325;
    for f in &frames[..n] {
        h = (h ^ *f as u64).wrapping_mul(0x100_0000_01b3);
    }
    let h = h | 1;
    lock();
    let t = unsafe { &mut *T.0.get() };
    // The site.
    let mut s = (h as usize) % SITES;
    let mut site = None;
    for _ in 0..SITES {
        if t.sites[s].hash == h {
            site = Some(s);
            break;
        }
        if t.sites[s].hash == 0 {
            let e = &mut t.sites[s];
            e.hash = h;
            e.n = n as u32;
            for (k, f) in frames[..n].iter().enumerate() {
                e.frames[k] = *f as usize;
            }
            t.nsites += 1;
            site = Some(s);
            break;
        }
        s = (s + 1) % SITES;
    }
    let Some(s) = site else {
        unlock();
        MISSED[0].fetch_add(1, Ordering::Relaxed);
        return;
    };
    // The block.
    let mut k = slot(p as usize);
    for _ in 0..PTRS {
        let e = &mut t.ptrs[k];
        if e.0 == EMPTY || e.0 == GONE {
            *e = (p as usize, size, s as u32 + 1);
            let st = &mut t.sites[s];
            st.live += size as u64;
            st.count += 1;
            st.allocs += 1;
            unlock();
            return;
        }
        k = (k + 1) % PTRS;
    }
    unlock();
    MISSED[0].fetch_add(1, Ordering::Relaxed);
}

/// A tracked block at `p` is freed (or moved by a reallocation).
#[cold]
pub fn note_free(p: *mut u8) {
    lock();
    let t = unsafe { &mut *T.0.get() };
    let mut k = slot(p as usize);
    for _ in 0..PTRS {
        let e = t.ptrs[k];
        if e.0 == EMPTY {
            break;
        }
        if e.0 == p as usize {
            let st = &mut t.sites[e.2 as usize - 1];
            st.live = st.live.saturating_sub(e.1 as u64);
            st.count = st.count.saturating_sub(1);
            t.ptrs[k] = (GONE, 0, 0);
            unlock();
            return;
        }
        k = (k + 1) % PTRS;
    }
    unlock();
    MISSED[1].fetch_add(1, Ordering::Relaxed);
}

unsafe extern "C" {
    fn mi_stats_print_out(out: Option<unsafe extern "C" fn(*const std::ffi::c_char, *mut std::ffi::c_void)>, arg: *mut std::ffi::c_void);
}

static MI_TEXT: parking_lot::Mutex<String> = parking_lot::Mutex::new(String::new());

unsafe extern "C" fn mi_out(msg: *const std::ffi::c_char, _arg: *mut std::ffi::c_void) {
    if msg.is_null() {
        return;
    }
    let t = unsafe { std::ffi::CStr::from_ptr(msg) }.to_string_lossy();
    MI_TEXT.lock().push_str(&t);
}

/// mimalloc's own statistics (its OS-level lines: what it reserved,
/// committed, reset and purged).
fn mi_lines() -> String {
    MI_TEXT.lock().clear();
    unsafe { mi_stats_print_out(Some(mi_out), std::ptr::null_mut()) };
    let t = std::mem::take(&mut *MI_TEXT.lock());
    t.lines()
        .filter(|l| ["reserved", "committed", "reset", "purged", "touched", "segments", "pages", "mmaps", "heap stats", "normal", "large", "huge", "malloc req", "process"].iter().any(|k| l.trim_start().starts_with(k)))
        .map(|l| format!("\n    mimalloc: {}", l.trim_end()))
        .collect()
}

/// Logs the largest sites by live bytes.
pub fn report() {
    const TOP: usize = 16;
    let mut top = [NO_SITE; TOP];
    let (mut live, mut count, nsites);
    {
        lock();
        let t = unsafe { &*T.0.get() };
        live = 0u64;
        count = 0u64;
        nsites = t.nsites;
        for s in t.sites.iter().filter(|s| s.hash != 0) {
            live += s.live;
            count += s.count;
            // (Insertion into the sorted top list.)
            if s.live > top[TOP - 1].live {
                let mut i = TOP - 1;
                while i > 0 && top[i - 1].live < s.live {
                    top[i] = top[i - 1];
                    i -= 1;
                }
                top[i] = *s;
            }
        }
        unlock();
    }
    let mut out = format!(
        "heap sites: {:.2} GiB live in {count} blocks of at least {} bytes over {nsites} sites ({} not tracked, {} frees not found); the largest:",
        live as f64 / (1u64 << 30) as f64,
        MIN.load(Ordering::Relaxed),
        MISSED[0].load(Ordering::Relaxed),
        MISSED[1].load(Ordering::Relaxed)
    );
    for s in top.iter().filter(|s| s.hash != 0 && s.live > 0) {
        // (The frames past the allocator's own.)
        let names: Vec<String> = s.frames[..s.n as usize].iter().map(|&f| crate::exception::host_symbol(f as u64)).filter(|n| !(n.contains("alloc::") || n.contains("GuardedAlloc") || n.contains("heap::") || n.contains("mi_") || n.contains("__rust"))).take(7).collect();
        out.push_str(&format!("\n    {:.1} MiB in {} blocks ({} allocated): {}", s.live as f64 / (1u64 << 20) as f64, s.count, s.allocs, names.join(" < ")));
    }
    out.push_str(&mi_lines());
    crate::info!("{out}");
}

/// Reads `DS_HEAP_SITES` and starts the reporting thread.
pub fn start() {
    let Some(min) = std::env::var("DS_HEAP_SITES").ok().and_then(|v| v.trim().parse::<usize>().ok()).filter(|&n| n > 0) else { return };
    MIN.store(min, Ordering::Relaxed);
    crate::info!("heap sites: tracking live host allocations of at least {min} bytes (DS_HEAP_SITES)");
    let _ = std::thread::Builder::new().name("heap-sites".into()).spawn(|| {
        loop {
            std::thread::sleep(std::time::Duration::from_secs(20));
            report();
        }
    });
}
