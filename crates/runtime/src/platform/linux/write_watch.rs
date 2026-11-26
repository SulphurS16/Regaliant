//! Exact section write watching with userfaultfd async WP and PAGEMAP_SCAN (Linux 6.7+).
//! Only guest views are registered, never Vulkan's aliases. A scan harvests dirty PTEs
//! and rearms them atomically; writes after it remain dirty for the next take. No handler
//! thread, signal delivery, or per-write mprotect. Guest mapping changes exclude takes
//! with resident::ARM_LOCK; announced alias/host writes also have an explicit journal.
//! See https://docs.kernel.org/admin-guide/mm/pagemap.html and mm/userfaultfd.html.
use super::os;
use crate::gpu::ranges::Ranges;
use parking_lot::Mutex;
use std::collections::BTreeMap;
use std::os::fd::{AsRawFd, FromRawFd, OwnedFd};
use std::sync::{OnceLock, atomic::{AtomicU64, Ordering::Relaxed}};

const PAGE: u64 = 4096;
const WRITTEN: u64 = 2;
const FEATURES: u64 = (1 << 12) | (1 << 13) | (1 << 15); // shmem WP, unpopulated, async
const fn ioc(dir: u64, ty: u64, nr: u64, size: usize) -> libc::c_ulong {
    ((dir << 30) | ((size as u64) << 16) | (ty << 8) | nr) as libc::c_ulong
}
#[repr(C)]
#[derive(Default)]
struct Api { api: u64, features: u64, ioctls: u64 }
#[repr(C)]
#[derive(Default, Clone, Copy)]
struct Range { start: u64, len: u64 }
#[repr(C)]
#[derive(Default)]
struct Register { range: Range, mode: u64, ioctls: u64 }
#[repr(C)]
#[derive(Default, Clone, Copy)]
struct PageRegion { start: u64, end: u64, categories: u64 }
#[repr(C)]
#[derive(Default)]
struct Scan {
    size: u64, flags: u64, start: u64, end: u64, walk_end: u64, vec: u64,
    vec_len: u64, max_pages: u64, category_inverted: u64, category_mask: u64,
    category_anyof_mask: u64, return_mask: u64,
}
const API: libc::c_ulong = ioc(3, 0xaa, 0x3f, size_of::<Api>());
const REGISTER: libc::c_ulong = ioc(3, 0xaa, 0, size_of::<Register>());
const UNREGISTER: libc::c_ulong = ioc(2, 0xaa, 1, size_of::<Range>());
const SCAN: libc::c_ulong = ioc(3, b'f' as u64, 16, size_of::<Scan>());
fn error(op: &str) -> String { format!("{op}: {}", std::io::Error::last_os_error()) }

#[derive(Clone, Copy)]
struct Region { end: u64, committed: bool, registered: bool }
struct Watch {
    uffd: OwnedFd,
    pagemap: std::fs::File,
    regions: BTreeMap<u64, Region>,
    dirty: Ranges,
    // Overlapping writes and identical nested announcements retain separate counts.
    writers: BTreeMap<(u64, u64), u32>,
}
static WATCH: OnceLock<Option<Mutex<Watch>>> = OnceLock::new();
// takes, scans, dirty pages, ns taking (including lock), registrations, failures, host writes
static STATS: [AtomicU64; 7] = [const { AtomicU64::new(0) }; 7];
// Optional 1/64 samples: count, mapping wait ns, watch mutex wait ns, take work ns;
// followed by last sampled gauges: region intervals, journal intervals, active writers.
static DETAIL: [AtomicU64; 7] = [const { AtomicU64::new(0) }; 7];
pub fn detail() -> [u64; 7] { DETAIL.each_ref().map(|v| v.load(Relaxed)) }
pub fn play_telemetry() -> bool {
    static ON: OnceLock<bool> = OnceLock::new();
    *ON.get_or_init(|| std::env::var("DS_PLAY_TELEMETRY").ok().as_deref() == Some("1"))
}

pub fn enabled() -> bool { WATCH.get().is_some_and(Option::is_some) }
pub fn initialize() {
    WATCH.get_or_init(|| {
        let policy = std::env::var("DS_LINUX_WRITE_WATCH").unwrap_or_else(|_| "uffd".into());
        if !matches!(policy.as_str(), "uffd" | "auto") { return None; }
        if !crate::sections::on() || crate::gpu::resident::wanted() {
            crate::warn!("Linux async write watch unavailable outside section tracking; using mprotect");
            return None;
        }
        match Watch::new() {
            Ok(w) => { crate::info!("Linux write watch: uffd async WP + PAGEMAP_SCAN (guest views only)"); Some(Mutex::new(w)) }
            Err(e) => { crate::warn!("Linux async write watch unavailable: {e}; falling back to mprotect (tour corruption reported; use a kernel permitting async WP for preparation)"); None }
        }
    });
    if !enabled() { crate::info!("Linux write watch: mprotect"); }
}

impl Watch {
    fn new() -> Result<Self, String> {
        let fd = unsafe { libc::syscall(libc::SYS_userfaultfd, libc::O_CLOEXEC | libc::O_NONBLOCK | 1) };
        if fd < 0 { return Err(error("userfaultfd(USER_MODE_ONLY)")); }
        let uffd = unsafe { OwnedFd::from_raw_fd(fd as i32) };
        let mut api = Api { api: 0xaa, features: FEATURES, ..Default::default() };
        if unsafe { libc::ioctl(uffd.as_raw_fd(), API, &mut api) } != 0 { return Err(error("UFFDIO_API async WP")); }
        if api.features & FEATURES != FEATURES { return Err("kernel lacks async WP/unpopulated/shmem features".into()); }
        let pagemap = std::fs::File::open("/proc/self/pagemap").map_err(|e| format!("pagemap: {e}"))?;
        Ok(Self { uffd, pagemap, regions: BTreeMap::new(), dirty: Ranges::new(), writers: BTreeMap::new() })
    }

    fn register(&self, a: u64, e: u64) -> Result<(), String> {
        let mut r = Register { range: Range { start: a, len: e-a }, mode: 2, ..Default::default() };
        if unsafe { libc::ioctl(self.uffd.as_raw_fd(), REGISTER, &mut r) } != 0 { return Err(error("UFFDIO_REGISTER WP")); }
        STATS[4].fetch_add(1, Relaxed);
        Ok(())
    }

    fn ensure(&mut self, lo: u64, hi: u64) -> Result<(), String> {
        let mut p = lo;
        while p < hi {
            if let Some((_, r)) = self.regions.range(..=p).next_back().filter(|(_, r)| r.end > p) {
                p = r.end;
                continue;
            }
            let mut info = unsafe { std::mem::zeroed::<os::MEMORY_BASIC_INFORMATION>() };
            if unsafe { os::VirtualQuery(p as _, &mut info, size_of_val(&info)) } == 0 { return Err(error("query write-watch mapping")); }
            let end = (info.BaseAddress as u64 + info.RegionSize as u64).min(crate::mem::GUEST_END);
            // Register the remainder of this mapping once, but never overlap a cached part.
            let end = self.regions.range(p..).next().map_or(end, |(&a, _)| end.min(a));
            if end <= p { return Err("write-watch region made no progress".into()); }
            let committed = info.State == os::MEM_COMMIT && info.Protect & 0xff != os::PAGE_NOACCESS;
            if committed { self.register(p, end)?; }
            self.regions.insert(p, Region { end, committed, registered: committed });
            p = end;
        }
        Ok(())
    }

    /// The bookkeeping half of a take, under the watch mutex: registers what the range needs,
    /// and returns the committed parts to scan plus the journal's dirty pages over the range
    /// (removed from the journal: this take reports them). The kernel scans happen after the
    /// mutex is released (`scan_fd`): PAGEMAP_SCAN is safe to run concurrently, and the
    /// caller's mapping guard keeps remaps (`forget`) out while it runs.
    fn plan(&mut self, lo: u64, hi: u64, parts: &mut Vec<(u64, u64)>, dirty: &mut Vec<(u64, u64)>) -> Result<(), String> {
        self.ensure(lo, hi)?;
        // A syscall/host write can use an untracked alias, or span this take. Announced
        // writes remain conservatively dirty until end, which journals them once again.
        dirty.extend(self.dirty.intersect(lo, hi-lo));
        for (&(a, e), _) in &self.writers {
            if a < hi && e > lo { dirty.push((a.max(lo), e.min(hi)-a.max(lo))); }
        }
        for (&a, r) in self.regions.range(..hi).rev().take_while(|(_, r)| r.end > lo) {
            if r.committed { parts.push((a.max(lo), r.end.min(hi))); }
        }
        self.dirty.remove(lo, hi-lo);
        Ok(())
    }

    fn scan(&self, lo: u64, hi: u64, out: &mut Vec<u64>) -> Result<(), String> {
        scan_fd(self.pagemap.as_raw_fd(), lo, hi, out)
    }

    fn take(&mut self, lo: u64, hi: u64, out: &mut Vec<u64>) -> Result<(), String> {
        self.ensure(lo, hi)?;
        // A syscall/host write can use an untracked alias, or span this take. Announced
        // writes remain conservatively dirty until end, which journals them once again.
        let mut dirty = self.dirty.intersect(lo, hi-lo);
        for (&(a, e), _) in &self.writers {
            if a < hi && e > lo { dirty.push((a.max(lo), e.min(hi)-a.max(lo))); }
        }
        let first = out.len();
        for (&a, r) in self.regions.range(..hi).rev().take_while(|(_, r)| r.end > lo) {
            if !r.committed { continue; }
            let (a, e) = (a.max(lo), r.end.min(hi));
            self.scan(a, e, out)?;
            for &(d, n) in &dirty {
                out.extend((a.max(d)..e.min(d+n)).step_by(PAGE as usize));
            }
        }
        out[first..].sort_unstable();
        // Adjacent kernel runs don't overlap; the explicit host journal can.
        let mut write = first;
        for read in first..out.len() {
            if write == first || out[read] != out[write-1] { out[write] = out[read]; write += 1; }
        }
        out.truncate(write);
        self.dirty.remove(lo, hi-lo);
        Ok(())
    }

    fn forget(&mut self, lo: u64, hi: u64) {
        let old: Vec<_> = self.regions.range(..hi).rev().take_while(|(_, r)| r.end > lo).map(|(&a, &r)| (a, r)).collect();
        for (a, r) in old {
            let (start, end) = (a.max(lo), r.end.min(hi));
            if r.registered {
                let range = Range { start, len: end-start };
                // mmap/munmap already dropped registration; mprotect may have retained it.
                if unsafe { libc::ioctl(self.uffd.as_raw_fd(), UNREGISTER, &range) } != 0 && std::io::Error::last_os_error().raw_os_error() != Some(libc::EINVAL) {
                    // Do not silently keep stale PTE tracking after a remap.
                    crate::fatal_exit(1, &error("UFFDIO_UNREGISTER after mapping change"));
                }
            }
            self.regions.remove(&a);
            if a < lo { self.regions.insert(a, Region { end: lo, ..r }); }
            if r.end > hi { self.regions.insert(hi, r); }
        }
        self.dirty.remove(lo, hi-lo);
    }
}

static SCAN_LOGGED: AtomicU64 = AtomicU64::new(0);
/// PAGEMAP_SCAN over [lo, hi): appends the pages written since their last scan and
/// write-protects them again (WP_MATCHING | CHECK_WPASYNC). Needs no watch mutex.
fn scan_fd(fd: i32, lo: u64, hi: u64, out: &mut Vec<u64>) -> Result<(), String> {
    let mut at = lo;
    let mut regions = [PageRegion::default(); 128];
    while at < hi {
        let mut arg = Scan { size: size_of::<Scan>() as u64, flags: 3,
            start: at, end: hi, vec: regions.as_mut_ptr() as u64, vec_len: regions.len() as u64,
            category_mask: WRITTEN, return_mask: WRITTEN, ..Default::default() };
        let ts = std::time::Instant::now();
        let n = unsafe { libc::ioctl(fd, SCAN, &mut arg) };
        let us = ts.elapsed().as_micros() as u64;
        // (A scan waits for the mm's lock behind a mapping change, which waits for the
        // scans before it: long ones are logged, 2026-11-26.)
        if us >= 5_000 && SCAN_LOGGED.fetch_add(1, Relaxed) < 100 {
            crate::info!("pagemap scan {at:#x}..{:#x} ({} pages walked, {n} regions) took {us} us", arg.walk_end, (arg.walk_end.max(at) - at) / PAGE);
        }
        STATS[1].fetch_add(1, Relaxed);
        if n < 0 { return Err(error("PAGEMAP_SCAN")); }
        if n as usize > regions.len() || arg.walk_end <= at || arg.walk_end > hi { return Err("PAGEMAP_SCAN invalid progress".into()); }
        for r in &regions[..n as usize] {
            if r.start < at || r.end > arg.walk_end || r.start >= r.end || r.categories & WRITTEN == 0 {
                return Err("PAGEMAP_SCAN invalid region".into());
            }
            out.extend((r.start..r.end).step_by(PAGE as usize));
        }
        at = arg.walk_end;
    }
    Ok(())
}

/// Whether takes scan outside the watch mutex (default; DS_LINUX_WW_UNLOCKED_SCAN=0: the
/// whole take under it, as before).
fn unlocked_scan() -> bool {
    static ON: OnceLock<bool> = OnceLock::new();
    *ON.get_or_init(|| std::env::var("DS_LINUX_WW_UNLOCKED_SCAN").ok().as_deref() != Some("0"))
}

/// `take` with the kernel scans outside the watch mutex (`Watch::plan`, `scan_fd`).
fn take_unlocked(base: u64, size: u64, out: &mut Vec<u64>) -> Option<()> {
    if size == 0 { return Some(()); }
    let (lo, hi) = (base & !(PAGE-1), base.checked_add(size)?.checked_next_multiple_of(PAGE)?);
    if lo < crate::mem::GUEST_BASE || hi > crate::mem::GUEST_END { return None; }
    thread_local! {
        static SAMPLE: std::cell::Cell<u32> = const { std::cell::Cell::new(0) };
        static PLAN: std::cell::RefCell<(Vec<(u64, u64)>, Vec<(u64, u64)>)> = const { std::cell::RefCell::new((Vec::new(), Vec::new())) };
    }
    let sampled = play_telemetry() && SAMPLE.with(|n| { let x = n.get().wrapping_add(1); n.set(x); x % 64 == 0 });
    let before = sampled.then(crate::gpu::record_opts::Timer::start);
    // Shared: remaps (`remapped`, under the exclusive mapping guard) cannot run until the
    // scans below are done, so the parts planned stay registered and mapped.
    let _mapping = crate::gpu::resident::arm_guard();
    let mapping_ns = before.map(|t| t.elapsed().as_nanos() as u64).unwrap_or(0);
    let t = crate::gpu::record_opts::Timer::start();
    let watch = WATCH.get()?.as_ref()?;
    PLAN.with(|plan| {
        let (parts, dirty) = &mut *plan.borrow_mut();
        parts.clear();
        dirty.clear();
        let (fd, lock_ns, gauges) = {
            let mut w = watch.lock();
            let lock_ns = if sampled { t.elapsed().as_nanos() as u64 } else { 0 };
            if let Err(e) = w.plan(lo, hi, parts, dirty) {
                w.dirty.insert(lo, hi-lo);
                if STATS[5].fetch_add(1, Relaxed) < 8 { crate::warn!("Linux async write watch: {lo:#x}..{hi:#x}: {e}"); }
                return None;
            }
            (w.pagemap.as_raw_fd(), lock_ns, [w.regions.len(), w.dirty.len(), w.writers.len()])
        };
        let first = out.len();
        for &(a, e) in parts.iter() {
            if let Err(err) = scan_fd(fd, a, e, out) {
                // A partial scan can already have reset dirty bits: keep the whole request
                // dirty for the next attempt; callers treat None conservatively now.
                out.truncate(first);
                watch.lock().dirty.insert(lo, hi-lo);
                if STATS[5].fetch_add(1, Relaxed) < 8 { crate::warn!("Linux async write watch: {lo:#x}..{hi:#x}: {err}"); }
                return None;
            }
            for &(d, n) in dirty.iter() {
                out.extend((a.max(d)..e.min(d+n)).step_by(PAGE as usize));
            }
        }
        out[first..].sort_unstable();
        // Adjacent kernel runs don't overlap; the explicit host journal can.
        let mut write = first;
        for read in first..out.len() {
            if write == first || out[read] != out[write-1] { out[write] = out[read]; write += 1; }
        }
        out.truncate(write);
        STATS[0].fetch_add(1, Relaxed);
        STATS[2].fetch_add((out.len()-first) as u64, Relaxed);
        let total_ns = t.elapsed().as_nanos() as u64;
        STATS[3].fetch_add(total_ns, Relaxed);
        if sampled {
            for (i, value) in [1, mapping_ns, lock_ns, total_ns.saturating_sub(lock_ns)].into_iter().enumerate() {
                DETAIL[i].fetch_add(value, Relaxed);
            }
            for (i, value) in gauges.into_iter().enumerate() {
                DETAIL[i+4].store(value as u64, Relaxed);
            }
        }
        Some(())
    })
}

pub fn take(base: u64, size: u64, out: &mut Vec<u64>) -> Option<()> {
    if unlocked_scan() { return take_unlocked(base, size, out); }
    if size == 0 { return Some(()); }
    let (lo, hi) = (base & !(PAGE-1), base.checked_add(size)?.checked_next_multiple_of(PAGE)?);
    if lo < crate::mem::GUEST_BASE || hi > crate::mem::GUEST_END { return None; }
    // Same mapping lock/order as the mprotect backend. No signal calls this module.
    thread_local! { static SAMPLE: std::cell::Cell<u32> = const { std::cell::Cell::new(0) }; }
    let sampled = play_telemetry() && SAMPLE.with(|n| { let x = n.get().wrapping_add(1); n.set(x); x % 64 == 0 });
    let before = sampled.then(crate::gpu::record_opts::Timer::start);
    let _mapping = crate::gpu::resident::arm_guard();
    let mapping_ns = before.map(|t| t.elapsed().as_nanos() as u64).unwrap_or(0);
    let t = crate::gpu::record_opts::Timer::start();
    let mut w = WATCH.get()?.as_ref()?.lock();
    let lock_ns = if sampled { t.elapsed().as_nanos() as u64 } else { 0 };
    let first = out.len();
    if let Err(e) = w.take(lo, hi, out) {
        // A partial scan can already have reset dirty bits. Keep the whole request dirty
        // for the next attempt too; callers treat None conservatively on this attempt.
        out.truncate(first);
        w.dirty.insert(lo, hi-lo);
        if STATS[5].fetch_add(1, Relaxed) < 8 { crate::warn!("Linux async write watch: {lo:#x}..{hi:#x}: {e}"); }
        return None;
    }
    STATS[0].fetch_add(1, Relaxed);
    STATS[2].fetch_add((out.len()-first) as u64, Relaxed);
    let total_ns = t.elapsed().as_nanos() as u64;
    STATS[3].fetch_add(total_ns, Relaxed);
    if sampled {
        for (i, value) in [1, mapping_ns, lock_ns, total_ns.saturating_sub(lock_ns)].into_iter().enumerate() {
            DETAIL[i].fetch_add(value, Relaxed);
        }
        for (i, value) in [w.regions.len(), w.dirty.len(), w.writers.len()].into_iter().enumerate() {
            DETAIL[i+4].store(value as u64, Relaxed);
        }
    }
    Some(())
}

/// Called under the memory manager's exclusive mapping guard, after its OS operation.
pub fn remapped(base: u64, size: u64) {
    if let Some(Some(w)) = WATCH.get() { w.lock().forget(base & !(PAGE-1), base.saturating_add(size).next_multiple_of(PAGE)); }
}

pub fn host_write(base: u64, size: u64, begin: bool) {
    if size == 0 { return; }
    let Some(Some(w)) = WATCH.get() else { return };
    let (a, e) = (base & !(PAGE-1), base.saturating_add(size).next_multiple_of(PAGE));
    let mut w = w.lock();
    w.dirty.insert(a, e-a);
    if begin {
        *w.writers.entry((a, e)).or_default() += 1;
        STATS[6].fetch_add(1, Relaxed);
    } else if let Some(n) = w.writers.get_mut(&(a, e)) {
        *n -= 1;
        if *n == 0 { w.writers.remove(&(a, e)); }
    }
}

pub fn counters() -> [u64; 7] { STATS.each_ref().map(|v| v.load(Relaxed)) }
pub fn status() -> String {
    if !enabled() { return String::new(); }
    let [takes, scans, pages, ns, registrations, errors, host] = counters();
    format!("Linux async write watch: {takes} takes, {scans} scans, {pages} pages, {:.2} us/take including lock; {registrations} registrations, {errors} errors, {host} announced host writes", ns as f64 / takes.max(1) as f64 / 1000.0)
}

/// CPU-only functional check and repeatable tracking workload, used by the GPU relay before
/// opening any window. Select the backend with DS_LINUX_WRITE_WATCH=mprotect|uffd.
pub fn check() -> Result<(), String> {
    unsafe { std::env::set_var("DS_SECTIONS", "1"); std::env::set_var("DS_RESIDENT", "0"); }
    crate::mem::reserve_guest_range()?;
    crate::gpu::resident::init_linux_faults();
    if std::env::var("DS_LINUX_WRITE_WATCH").ok().as_deref() == Some("uffd") && !enabled() {
        return Err("async WP requested but unavailable (no mprotect benchmark substitution)".into());
    }
    crate::platform::signals::ensure_installed().map_err(|e| e.to_string())?;
    let _stack = crate::platform::signals::SignalStack::new().map_err(|e| e.to_string())?;
    let a = crate::mem::MMAP_BASE + (64 << 20);
    let pages = 512u64;
    let size = pages*PAGE;
    crate::mem::MEM.lock().map(a, size, crate::mem::VmaKind::Flexible, 3, 0, 0, "write-watch-check").map_err(|e| format!("mapping check: {e:?}"))?;
    let take = |out: &mut Vec<u64>| crate::gpu::resident::take_written(a, size, out).ok_or("write-watch take failed".to_string());
    let mut out = Vec::new();
    take(&mut out)?;
    if out.len() != pages as usize { return Err(format!("first take: {} pages, expected {pages}", out.len())); }
    out.clear(); take(&mut out)?;
    if !out.is_empty() { return Err("quiet take was dirty".into()); }
    let rounds = 4000u64;
    let before = crate::platform::signals::RECOVERED.load(Relaxed);
    let mut cpu: libc::timespec = unsafe { std::mem::zeroed() };
    unsafe { libc::clock_gettime(libc::CLOCK_THREAD_CPUTIME_ID, &mut cpu); }
    let cpu0 = cpu.tv_sec as f64 + cpu.tv_nsec as f64/1e9;
    let start = std::time::Instant::now();
    for i in 0..rounds {
        // 32 scattered 4K pages: catches scans whose page_region output fills up,
        // and models frequent small CPU writes followed by larger GPU reads.
        let mut expected = [0; 32];
        for j in 0..32 { expected[j] = a + ((i*7+j as u64*13) % pages)*PAGE; }
        for &p in &expected { unsafe { std::ptr::write_volatile(p as *mut u64, i); } }
        out.clear(); take(&mut out)?;
        expected.sort_unstable();
        if out != expected { return Err(format!("take {i}: {} pages, expected 32 exact addresses", out.len())); }
    }
    let wall = start.elapsed().as_secs_f64();
    unsafe { libc::clock_gettime(libc::CLOCK_THREAD_CPUTIME_ID, &mut cpu); }
    let cpu_s = cpu.tv_sec as f64 + cpu.tv_nsec as f64/1e9-cpu0;
    let faults = crate::platform::signals::RECOVERED.load(Relaxed)-before;
    println!("write-watch benchmark: backend={} rounds={rounds} writes={} wall_s={wall:.6} cpu_s={cpu_s:.6} signals={faults}", if enabled() { "uffd" } else { "mprotect" }, rounds*32);
    println!("{}", status());
    if crate::hosted::on() { arming_check(a, pages)?; }
    crate::mem::MEM.lock().unmap_range(a, a+size);
    Ok(())
}

/// `DS_LINUX_ARMING=1` with `--write-watch-check`: page arming over the check's mapping
/// [a, a + pages*4K) (`hosted`): faults posted exactly once per written page, other pages
/// still armed, writes landing, concurrent writers, kernel copies after `disarm_if_armed`,
/// and the uffd take still seeing the armed pages' writes.
fn arming_check(a: u64, pages: u64) -> Result<(), String> {
    let size = pages*PAGE;
    let mut faults = Vec::new();
    let mut out = Vec::new();
    let take = |out: &mut Vec<u64>| crate::gpu::resident::take_written(a, size, out).ok_or("write-watch take failed".to_string());
    let drain = |f: &mut Vec<(u64, bool, u64)>| -> Result<Vec<u64>, String> {
        f.clear();
        if !crate::hosted::drain_faults(f) { return Err("arming: faults lost".into()); }
        let mut v: Vec<u64> = f.iter().map(|x| x.0).collect(); v.sort_unstable(); Ok(v)
    };
    drain(&mut faults)?;
    take(&mut out)?;
    // 1. One thread: 32 scattered pages written, exactly those faulted and disarmed.
    crate::hosted::arm(a, size)?;
    if !crate::hosted::all_armed(a, size) { return Err("arming: not all armed after arm".into()); }
    let mut expected: Vec<u64> = (0..32u64).map(|j| a + (j*13 % pages)*PAGE).collect();
    expected.sort_unstable(); expected.dedup();
    for (k, &p) in expected.iter().enumerate() { unsafe { std::ptr::write_volatile((p+8) as *mut u64, 0xA000 + k as u64); } }
    for (k, &p) in expected.iter().enumerate() {
        if unsafe { std::ptr::read_volatile((p+8) as *const u64) } != 0xA000 + k as u64 { return Err(format!("arming: write to {p:#x} lost")); }
    }
    let got = drain(&mut faults)?;
    if got != expected { return Err(format!("arming: {} faults posted, expected {} pages", got.len(), expected.len())); }
    for p in (a..a+size).step_by(PAGE as usize) {
        if crate::hosted::armed(p) == expected.contains(&p) { return Err(format!("arming: page {p:#x} armed state wrong after the writes")); }
    }
    out.clear(); take(&mut out)?;
    if out != expected { return Err(format!("arming: take after armed writes found {} pages, expected {}", out.len(), expected.len())); }
    // 2. Eight threads writing the same 64 pages at once: each page posted once, no write lost.
    crate::hosted::disarm(a, size);
    crate::hosted::arm(a, 64*PAGE)?;
    std::thread::scope(|sc| for t in 0..8u64 {
        sc.spawn(move || for i in 0..64u64 { unsafe { std::ptr::write_volatile((a + i*PAGE + t*8) as *mut u64, t + 1); } });
    });
    for i in 0..64u64 { for t in 0..8u64 {
        if unsafe { std::ptr::read_volatile((a + i*PAGE + t*8) as *const u64) } != t + 1 { return Err("arming: concurrent write lost".into()); }
    } }
    let got = drain(&mut faults)?;
    if got != (0..64u64).map(|i| a + i*PAGE).collect::<Vec<_>>() { return Err(format!("arming: concurrent writers posted {} faults for 64 pages", got.len())); }
    // 3. A kernel copy into armed pages after the I/O preparation (read(2) gets EFAULT, no
    // signal, on a read-only page).
    crate::hosted::arm(a, 8*PAGE)?;
    crate::platform::file_hooks::prepare_kernel_io(a, 8*PAGE, true);
    let zero = std::fs::File::open("/dev/zero").map_err(|e| e.to_string())?;
    use std::os::fd::AsRawFd;
    if unsafe { libc::read(zero.as_raw_fd(), a as *mut libc::c_void, (8*PAGE) as usize) } != (8*PAGE) as isize {
        return Err(format!("arming: kernel copy into disarmed pages failed: {}", std::io::Error::last_os_error()));
    }
    if (0..8u64).any(|i| crate::hosted::armed(a + i*PAGE)) { return Err("arming: pages still armed after the I/O preparation".into()); }
    drain(&mut faults)?;
    // 4. Cost: arm all, one write per page, timed.
    crate::hosted::disarm(a, size);
    let t = std::time::Instant::now();
    crate::hosted::arm(a, size)?;
    let arm_us = t.elapsed().as_secs_f64()*1e6;
    let t = std::time::Instant::now();
    for p in (a..a+size).step_by(PAGE as usize) { unsafe { std::ptr::write_volatile(p as *mut u64, 1); } }
    let fault_us = t.elapsed().as_secs_f64()*1e6 / pages as f64;
    let got = drain(&mut faults)?;
    if got.len() != pages as usize { return Err(format!("arming: {} faults for {pages} written pages", got.len())); }
    crate::hosted::disarm(a, size);
    println!("arming check passed: arm of {pages} pages {arm_us:.0} us; {fault_us:.2} us per armed write fault");
    println!("{}", crate::hosted::status());
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn async_wp_shared_pages_kernel_writes_remaps_and_partial_scans() {
        // Pure kernel tests have no Vulkan dependency. Skip only a kernel/permission
        // restriction; --write-watch-check in the relay requires success explicitly.
        let Ok(mut w) = Watch::new() else { eprintln!("async WP unavailable on this kernel/sandbox"); return };
        let fd = unsafe { libc::memfd_create(c"async-wp-test".as_ptr(), libc::MFD_CLOEXEC) };
        assert!(fd >= 0);
        let fd = unsafe { OwnedFd::from_raw_fd(fd) };
        let n = 1024*PAGE;
        assert_eq!(unsafe { libc::ftruncate(fd.as_raw_fd(), n as i64) }, 0);
        let p = unsafe { libc::mmap(std::ptr::null_mut(), n as usize, 3, libc::MAP_SHARED, fd.as_raw_fd(), 0) };
        assert_ne!(p, libc::MAP_FAILED);
        let a = p as u64;
        let alias = unsafe { libc::mmap(std::ptr::null_mut(), n as usize, 3, libc::MAP_SHARED, fd.as_raw_fd(), 0) };
        assert_ne!(alias, libc::MAP_FAILED);
        w.register(a, a+n).unwrap();
        w.regions.insert(a, Region { end: a+n, committed: true, registered: true });
        let mut out = Vec::new();
        w.take(a, a+n, &mut out).unwrap();
        assert_eq!(out.len(), 1024, "unpopulated pages are initially dirty and protected");
        out.clear();
        for i in (0..1024).step_by(2) { unsafe { std::ptr::write_volatile((a+i*PAGE) as *mut u64, i); } }
        w.take(a, a+n, &mut out).unwrap();
        assert_eq!(out, (0..1024).step_by(2).map(|i| a+i*PAGE).collect::<Vec<_>>(), "more than 128 disjoint output regions resumes at walk_end");
        out.clear(); w.take(a, a+n, &mut out).unwrap(); assert!(out.is_empty());
        // Writes through a GPU alias do not masquerade as CPU writes.
        unsafe { std::ptr::write_volatile(alias as *mut u64, 123); }
        w.take(a, a+n, &mut out).unwrap(); assert!(out.is_empty());
        // Kernel copies touch protected PTEs without a userspace signal or EFAULT.
        let zero = std::fs::File::open("/dev/zero").unwrap();
        assert_eq!(unsafe { libc::read(zero.as_raw_fd(), (a+PAGE) as _, 8) }, 8);
        w.take(a, a+n, &mut out).unwrap(); assert_eq!(out, [a+PAGE]);
        // An explicit alias-write announcement is conservatively dirty across takes.
        out.clear();
        w.writers.insert((a, a+PAGE), 1);
        w.take(a, a+n, &mut out).unwrap(); assert_eq!(out, [a]);
        out.clear(); w.take(a, a+n, &mut out).unwrap(); assert_eq!(out, [a]);
        w.writers.clear(); w.dirty.insert(a, PAGE);
        out.clear(); w.take(a, a+n, &mut out).unwrap(); assert_eq!(out, [a]);
        out.clear(); w.take(a, a+n, &mut out).unwrap(); assert!(out.is_empty());
        // Race a scan against first writes on another CPU. Each write must be
        // observed by this scan or the next, even if protection changes under it.
        let barrier = std::sync::Barrier::new(2);
        std::thread::scope(|scope| {
            let writer = scope.spawn(|| {
                barrier.wait();
                for i in 0..1024 { unsafe { std::ptr::write_volatile((a+i*PAGE) as *mut u64, i); } }
            });
            barrier.wait();
            out.clear(); w.take(a, a+n, &mut out).unwrap();
            writer.join().unwrap();
            w.take(a, a+n, &mut out).unwrap();
            out.sort_unstable(); out.dedup();
            assert_eq!(out, (0..1024).map(|i| a+i*PAGE).collect::<Vec<_>>());
        });
        out.clear(); w.take(a, a+n, &mut out).unwrap(); assert!(out.is_empty());
        // Unregister only a middle part; dirty tracking in the neighbors survives.
        w.forget(a+PAGE, a+2*PAGE);
        unsafe { std::ptr::write_volatile(a as *mut u64, 7); }
        w.scan(a, a+PAGE, &mut out).unwrap(); assert_eq!(out, [a]);
        w.register(a+PAGE, a+2*PAGE).unwrap();
        out.clear(); w.scan(a+PAGE, a+2*PAGE, &mut out).unwrap(); assert_eq!(out, [a+PAGE]);
        // No protected bit may leak from an old mapping into a fresh incarnation.
        assert_eq!(unsafe { libc::munmap(p, n as usize) }, 0);
        w.forget(a, a+n);
        let new = unsafe { libc::mmap(p, n as usize, 3, libc::MAP_SHARED | libc::MAP_FIXED_NOREPLACE, fd.as_raw_fd(), 0) };
        assert_eq!(new, p);
        w.register(a, a+n).unwrap();
        out.clear(); w.scan(a, a+n, &mut out).unwrap(); assert_eq!(out.len(), 1024);
        unsafe { libc::munmap(new, n as usize); libc::munmap(alias, n as usize); }
    }

    #[test]
    fn concurrent_guest_writes_rearm_and_masked_alias_writeback() {
        const NAME: &str = "platform::write_watch::tests::concurrent_guest_writes_rearm_and_masked_alias_writeback";
        let Ok(mode) = std::env::var("REGALIANT_WATCH_STRESS") else {
            for mode in ["mprotect", "uffd"] {
                let out = std::process::Command::new(std::env::current_exe().unwrap()).args(["--exact", NAME, "--nocapture"])
                    .env("REGALIANT_WATCH_STRESS", mode).env("DS_LINUX_WRITE_WATCH", mode)
                    .env("DS_SECTIONS", "1").env("DS_RESIDENT", "0").env("DS_LINUX_VOLATILE_BACKOFF", "0")
                    .env("DS_LINUX_DIRECT_FAULT", "1").output().unwrap();
                assert!(out.status.success(), "{mode}: {}\n{}", String::from_utf8_lossy(&out.stdout), String::from_utf8_lossy(&out.stderr));
                eprintln!("{mode}: {}", String::from_utf8_lossy(&out.stdout));
            }
            return;
        };
        if mode == "uffd" && Watch::new().is_err() { println!("SKIP uffd stress: kernel/sandbox denies async WP"); return; }
        crate::mem::reserve_guest_range().unwrap();
        crate::gpu::resident::init_linux_faults();
        crate::platform::signals::ensure_installed().unwrap();
        let _stack = crate::platform::signals::SignalStack::new().unwrap();
        const PAGES: usize = 16;
        const WORDS: usize = PAGES * 4096 / 8;
        const ROUNDS: u64 = 64;
        let a = crate::mem::MMAP_BASE + (128 << 20);
        let size = PAGES as u64 * PAGE;
        crate::mem::MEM.lock().map(a, size, crate::mem::VmaKind::Flexible, 3, 0, 0, "watch-stress").unwrap();
        let slot = crate::mem::bridge_aliases(a, a+size).unwrap();
        let alias = crate::sections::alias_of(a, slot);
        let word = |base: u64, i: usize| unsafe { &*((base + i as u64 * 8) as *const AtomicU64) };
        let phase = std::sync::Barrier::new(6); // four guest writers, alias writeback, scanner
        let done = std::sync::atomic::AtomicUsize::new(0);
        let mut snapshot = vec![0; WORDS];
        let mut out = Vec::new();
        let mut scan = |snapshot: &mut [u64]| {
            out.clear(); crate::gpu::resident::take_written(a, size, &mut out).unwrap();
            for &page in &out {
                let start = (page-a) as usize/8;
                // The CPU upload snapshot reads only CPU-owned words. Concurrent GPU
                // writes through the alias are intentionally NOT CPU dirtiness.
                for i in (start..start+512).step_by(2) { snapshot[i] = word(a, i).load(std::sync::atomic::Ordering::Acquire); }
            }
        };
        scan(&mut snapshot);
        let mut mismatch = None;
        std::thread::scope(|scope| {
            for worker in 0..4 {
                let (phase, done, word) = (&phase, &done, &word);
                scope.spawn(move || {
                    let _stack = crate::platform::signals::SignalStack::new().unwrap();
                    for round in 1..=ROUNDS {
                        phase.wait();
                        for i in (worker*2..WORDS).step_by(8) {
                            word(a, i).store((round << 32) | i as u64, std::sync::atomic::Ordering::Release);
                        }
                        done.fetch_add(1, std::sync::atomic::Ordering::Release);
                        phase.wait();
                    }
                });
            }
            let (phase_ref, done_ref, word_ref) = (&phase, &done, &word);
            scope.spawn(move || {
                for round in 1..=ROUNDS {
                    phase_ref.wait();
                    // Model FE2's marked-word stores into its unprotected alias, NOT a
                    // whole-page copy: CPU/GPU own disjoint words within every page.
                    for i in (1..WORDS).step_by(2) { word_ref(alias, i).store((round << 32) | i as u64, std::sync::atomic::Ordering::Release); }
                    done_ref.fetch_add(1, std::sync::atomic::Ordering::Release);
                    phase_ref.wait();
                }
            });
            for round in 1..=ROUNDS {
                done.store(0, std::sync::atomic::Ordering::Release);
                phase.wait();
                while done.load(std::sync::atomic::Ordering::Acquire) != 5 { scan(&mut snapshot); std::thread::yield_now(); }
                phase.wait();
                scan(&mut snapshot);
                for i in 0..WORDS {
                    let expected = (round << 32) | i as u64;
                    for (where_, actual) in [("guest", word(a, i).load(std::sync::atomic::Ordering::Acquire)),
                        ("alias", word(alias, i).load(std::sync::atomic::Ordering::Acquire)),
                        ("snapshot", if i % 2 == 0 { snapshot[i] } else { expected })] {
                        if actual != expected && mismatch.is_none() { mismatch = Some(format!("{mode} {where_} word {i} round {round}: {actual:#x} != {expected:#x}")); }
                    }
                }
            }
        });
        assert!(mismatch.is_none(), "{}", mismatch.unwrap_or_default());
        crate::mem::MEM.lock().unmap_range(a, a+size);
        println!("{mode}: {ROUNDS} rounds, {} words checked in guest/alias; four writers + rearming + masked writeback", WORDS as u64 * ROUNDS);
    }

    #[test]
    fn integrated_tracking_backends() {
        if let Ok(mode) = std::env::var("REGALIANT_WATCH_CHECK") {
            // An unavailable optional kernel mechanism is covered by the low-level test;
            // the external relay separately fails instead of silently substituting modes.
            if mode == "uffd" && Watch::new().is_err() { return; }
            check().unwrap();
            return;
        }
        for mode in ["mprotect", "uffd"] {
            let out = std::process::Command::new(std::env::current_exe().unwrap())
                .args(["--exact", "platform::write_watch::tests::integrated_tracking_backends", "--nocapture"])
                .env("REGALIANT_WATCH_CHECK", mode).env("DS_LINUX_WRITE_WATCH", mode)
                .env("DS_LINUX_VOLATILE_BACKOFF", "0").env("DS_LINUX_DIRECT_FAULT", "1")
                .output().unwrap();
            assert!(out.status.success(), "{mode}: {}\n{}", String::from_utf8_lossy(&out.stdout), String::from_utf8_lossy(&out.stderr));
        }
    }
}
