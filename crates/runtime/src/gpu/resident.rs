//! GPU memory residency (RENDERER.md, "Residency design"; `DS_RESIDENT=1`).
//!
//! Stage 1, the foundation; nothing is bound from the mirror yet:
//! - The page table: per 4 KiB guest page a state (`ST_*`), kept in 64 MiB
//!   chunks made on first use (guest addresses are host addresses).
//! - Arming: a page whose contents the mirror holds is made read-only
//!   (`arm`); the CPU's next write faults (`on_fault`, asked first by the
//!   exception handler), and the page becomes writable and device-invalid
//!   again. Arm before copying a page out: a write in between then faults and
//!   invalidates the copy. Only pages that are plain read-write now are
//!   armed; imported pages must never be (protecting pages imported into the
//!   GPU lost the device: `track`); a guest protection change, commit or
//!   decommit resets the pages it covers (`remapped`). A page's arming and
//!   disarming hold its `ST_BUSY` bit, so a write fault racing either waits
//!   for the protection change instead of being taken for a crash.
//! - The mirror: one sparse buffer over guest [MIRROR_BASE, MIRROR_END), its
//!   64 KiB pages bound on first use (`back`) to device memory taken from
//!   64 MiB blocks.
//! - Staging copies into and out of the mirror (`upload_now`,
//!   `readback_now`): synchronous here (the boot self-test).
//!
//! Stage 2: read-only buffer bindings of pages the mirror holds bind it
//! (`Exec::resident_bind`). A page is uploaded by the batch that first binds
//! it: claimed (`claim_pages`: its upload pending), armed, copied (memcpy)
//! into the slot's staging ring and from there into the mirror in the
//! batch's prologue; once that batch is submitted (`submitted`) every later
//! batch may bind it (each batch begins behind a full barrier). While an
//! upload is pending no other batch uploads or binds the page (two pending
//! copies could land in either order). A page written again within
//! VOLATILE_FRAMES of its last write fault is volatile: left unarmed (each
//! fault and re-arm costs microseconds), it keeps the old placements, for
//! VOL_BASE << n frames the n-th time (n up to VOL_MAX_LEVEL), then armed
//! again by the next take (pages written during a load and quiet after it
//! stop counting as written).
//! Pages an import covers are host class (`mark_host`, called before every
//! import): protecting imported pages lost the device.
//!
//! CPU-write tracking for section-backed guest memory (`sections`, which has
//! no write watching): `take_written` reports the pages not taken since they
//! were last written (no `ST_CLEAN`), marks them clean and arms them; a write
//! fault, a file read or a remap clears the bit. The table and fault path run
//! whenever sections are on (`tracking`), the mirror only with `DS_RESIDENT`. Writes that cannot
//! fault (file reads: the kernel fails them on a read-only page, "Invalid
//! access to memory location") announce themselves (`host_write_begin` and
//! `_end`, from `track::host_write`'s guard): their pages are disarmed and
//! not armed again until the write ends.
//! Hosted (`--host`), that tracking arms through the host (`via_host`,
//! `hosted::arm`): the host answers its write faults (each one handled here
//! overwrites the guest's red zone: 459,145 of them in front end 2's probe
//! on sections, FINDINGS.md 2026-10-28) and clears the page's bit, so takes
//! compare the pages armed so (`Chunk::harmed`) with the host's bits
//! (`reconcile`) and end the arming of those whose bit is gone as a fault
//! here would. Pages another armed through the host (front end 2) are
//! joined, not taken for read-only to the guest (`host_mark`).

use super::vk::ctx;
use ash::vk;
use parking_lot::Mutex;
use std::sync::atomic::{AtomicBool, AtomicPtr, AtomicU32, AtomicU64, AtomicU8, Ordering};

pub const PAGE: u64 = 4096;
/// Page states per chunk of the table: 64 MiB of guest memory.
const CHUNK_SHIFT: u32 = 26;
const CHUNK_PAGES: usize = 1 << (CHUNK_SHIFT - 12);
/// Guest span the mirror covers (the title's direct memory and the GPU
/// buffers seen at the courtyard lie in [0x10_0000_0000, 0x15_0000_0000)).
pub const MIRROR_BASE: u64 = 0x10_0000_0000;
pub const MIRROR_END: u64 = 0x18_0000_0000;
/// Device memory blocks the mirror's pages are bound from.
const BLOCK: u64 = 64 << 20;

/// The mirror holds the page's current contents (no CPU write since).
pub const ST_VALID: u8 = 1;
/// Made read-only by us: the CPU's next write faults.
pub const ST_ARMED: u8 = 2;
/// A protection change of the page is in progress (`arm`, `on_fault`).
const ST_BUSY: u8 = 4;
/// Host class: imported or written by CP writes the GPU performs; never armed.
pub const ST_HOST: u8 = 8;
/// Written by the CPU in nearby frames again and again: never armed.
pub const ST_VOLATILE: u8 = 16;
/// GPU work wrote it in guest memory (`gpu_wrote`; cleared by remaps): read
/// in place rather than uploaded again after every such write (a binding of
/// it is refused; `DS_RES_GPU_PAGES=1` keeps such pages in the mirror).
const ST_GPU_W: u8 = 32;
/// Taken since its last write (`take_written`; cleared by write faults, file
/// reads, remaps).
const ST_CLEAN: u8 = 64;
/// The guest cannot write it (read-only or no-access protection when it was
/// to be armed): clean once taken, without arming (cleared by remaps).
const ST_GUEST_RO: u8 = 128;
/// A page's upload sequence while the batch copying it is not submitted.
const PENDING: u64 = u64::MAX;
/// A write fault this many frames after the page's last one makes it volatile.
const VOLATILE_FRAMES: u32 = 8;
/// A volatile page stays unarmed VOL_BASE << n frames the n-th time it
/// turns volatile (n up to VOL_MAX_LEVEL: 32 frames, then 64, .. 1024).
const VOL_BASE: u32 = 16;
const VOL_MAX_LEVEL: u8 = 6;
/// A page whose next write fault comes this many frames after its stay
/// unarmed ended loses its level (a burst of writes, a load's, is over; a
/// page written all the time faults right after its stay).
const QUIET_FRAMES: u32 = 64;

struct Chunk {
    pages: Box<[AtomicU8]>,
    /// The batch that uploaded the page (PENDING: not submitted yet; 0: none).
    seqs: Box<[AtomicU64]>,
    /// The frame of the page's last write fault, plus one (0: none).
    faulted: Box<[AtomicU32]>,
    /// Writes that cannot fault in progress over the page (`host_write_begin`).
    writing: Box<[AtomicU8]>,
    /// The synchronization epoch (`precache::epoch`, low bits) a page that
    /// cannot be armed was last reported written in (`take_written`).
    stamp: Box<[AtomicU32]>,
    /// Volatile pages: the frame (`FRAMES` plus one) their stay unarmed
    /// ends, and how many times they turned volatile.
    vol_until: Box<[AtomicU32]>,
    vol_level: Box<[AtomicU8]>,
    /// Mirror mode: one bit per page the mirror holds current (a placement
    /// need not look at it): set by placements that found it so
    /// (`mm_page_in`), cleared with its attention bit (`flag`) and wherever
    /// it stops being device-valid otherwise (`uncur`).
    cur: Box<[AtomicU64]>,
    /// Mirror mode: one bit per page with dwords GPU work wrote that guest
    /// memory lacks (no-access until released: `DIRTY`, `release_dirty`).
    gdirty: Box<[AtomicU64]>,
    /// Lazy mirror mode: one bit per page made no-access for GPU ownership
    /// (`own_begin`: from before the writing batch's submission; dirty or
    /// not) until a CPU access settles it (`settle`).
    gprot: Box<[AtomicU64]>,
    /// The batch sequence the next batch would have had when the page was
    /// last remapped: masks of batches submitted before it belong to the old
    /// mapping and are dropped (`apply_masks`).
    rseq: Box<[AtomicU64]>,
    /// Attention bits, one per page: set when the page may not be clean
    /// (write faults, disarming, file reads, remaps, new chunks; pages that
    /// cannot be armed keep it), cleared by the take that makes it clean.
    /// Takes visit only these pages (`take_written`).
    attn: Box<[AtomicU64]>,
    /// Hosted write tracking: one bit per page armed through the host (by
    /// us, or joined: its host bit was set already). The host clears its
    /// own bit when it answers a write fault there; takes find such pages
    /// (`reconcile`).
    harmed: Box<[AtomicU64]>,
    /// The host's count of bits cleared in this chunk's 128 MiB
    /// (`hosted::leaf_events`) as of the last `reconcile` of the chunk.
    host_ev: AtomicU64,
}

/// Flags page `i` of `c` for the next take's attention (`Chunk::attn`); it
/// is no longer known current in the mirror (`Chunk::cur`).
#[inline]
fn flag(c: &Chunk, i: usize) {
    c.attn[i >> 6].fetch_or(1u64 << (i & 63), Ordering::AcqRel);
    uncur(c, i);
}

/// Page `i` of `c` is no longer known current in the mirror (after its
/// state changed).
#[inline]
fn uncur(c: &Chunk, i: usize) {
    if c.cur[i >> 6].load(Ordering::Relaxed) & (1u64 << (i & 63)) != 0 {
        c.cur[i >> 6].fetch_and(!(1u64 << (i & 63)), Ordering::AcqRel);
    }
}

/// Clears the attention bits of the pages [a, e) of chunk `c` (one chunk).
fn clear_attn(c: &Chunk, a: u64, e: u64) {
    let (mut i, j) = (idx(a), idx(e - PAGE) + 1);
    while i < j {
        let end = (((i >> 6) + 1) << 6).min(j);
        let n = end - i;
        let mask = if n == 64 { u64::MAX } else { ((1u64 << n) - 1) << (i & 63) };
        c.attn[i >> 6].fetch_and(!mask, Ordering::AcqRel);
        i = end;
    }
}

/// The first page of [p, stop) (one chunk) whose attention bit is set.
fn next_flagged(c: &Chunk, p: u64, stop: u64) -> Option<u64> {
    let base = p & !((1u64 << CHUNK_SHIFT) - 1);
    let (mut i, j) = (idx(p), idx(stop - PAGE) + 1);
    while i < j {
        let w = c.attn[i >> 6].load(Ordering::Acquire) >> (i & 63);
        if w != 0 {
            let k = i + w.trailing_zeros() as usize;
            return (k < j).then_some(base + ((k as u64) << 12));
        }
        i = ((i >> 6) + 1) << 6;
    }
    None
}

/// `DS_TAKE_FULL=1`: takes visit every page of their range, as before the
/// attention bits (A/B).
fn take_full() -> bool {
    static F: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *F.get_or_init(|| std::env::var("DS_TAKE_FULL").ok().as_deref() == Some("1"))
}

/// Chunks by guest address >> CHUNK_SHIFT, over the guest range; never
/// removed (288 KiB each). Lock-free: a take looks one up for every page it
/// walks (a shared lock and a hash map there cost ~0.4 us per page with
/// several queue threads taking, tour 20261015-151613).
const CHUNK_SLOTS: usize = (crate::mem::GUEST_END >> CHUNK_SHIFT) as usize;
static TABLE: [AtomicPtr<Chunk>; CHUNK_SLOTS] = [const { AtomicPtr::new(std::ptr::null_mut()) }; CHUNK_SLOTS];
static ON: AtomicBool = AtomicBool::new(false);

/// Counters (`status`): pages armed, write faults taken, arm calls, their
/// protection calls and ns, mirror pages backed, uploads, readbacks, resets,
/// fault races, pages claimed for upload, pages made volatile, pages made
/// host class, bindings served from the mirror, and refusals by REFUSE_NAMES.
static STATS: [AtomicU64; 23] = [const { AtomicU64::new(0) }; 23];
/// Levels reset by a write fault long after the page's stay (`QUIET_FRAMES`).
const S_VOL_DECAYED: usize = 21;
/// Write faults the host answered, read from its ring (`host_fault`).
const S_HOST_FAULTS: usize = 22;
/// Mirror pages made stale by GPU writes in guest memory (`gpu_wrote`).
const S_GPU_INVAL: usize = 20;
/// Volatile pages armed again after their stay (`take_written`).
const S_VOL_REARMED: usize = 19;
/// Pages left unarmed by `arm` for their protection (neither plain
/// read-write nor unwritable: executable read-write, guard): a take reports
/// them every time.
const S_OTHER_PROT: usize = 18;
/// Protection calls and ns that disarmed (the others armed).
const S_UNPROTECTS: usize = 14;
const S_UNPROTECT_NS: usize = 15;
/// Host writes announced, and pages disarmed by them.
const S_HOST_WRITES: usize = 16;
const S_HOST_WRITE_PAGES: usize = 17;
const S_CLAIMED: usize = 9;
const S_VOLATILE: usize = 10;
const S_HOSTED: usize = 11;
const S_BINDS: usize = 12;
const S_BIND_PAGES: usize = 13;
/// Why a read-only binding was not served from the mirror (`refused`).
pub const REFUSE_NAMES: [&str; 9] = ["outside the mirror or too large", "GPU or CP writes pending", "host or volatile pages", "another batch's upload pending", "imported pages", "staging full", "arming failed", "mirror pages not bound", "GPU-written pages"];
static REFUSED: [AtomicU64; 9] = [const { AtomicU64::new(0) }; 9];

/// Counts a refusal (`REFUSE_NAMES` index).
pub fn refused(why: usize) {
    REFUSED[why.min(8)].fetch_add(1, Ordering::Relaxed);
}

/// Counts a binding served from the mirror and its pages.
pub fn served(pages: u64) {
    STATS[S_BINDS].fetch_add(1, Ordering::Relaxed);
    STATS[S_BIND_PAGES].fetch_add(pages, Ordering::Relaxed);
}
const S_ARMED: usize = 0;
const S_FAULTS: usize = 1;
const S_PROTECTS: usize = 2;
const S_PROTECT_NS: usize = 3;
const S_BACKED: usize = 4;
const S_UPLOADS: usize = 5;
const S_READBACKS: usize = 6;
const S_RESETS: usize = 7;
const S_RACES: usize = 8;

/// `DS_RESIDENT=1`: residency is wanted (the device enables sparse buffers);
/// `DS_RESIDENT=2`: mirror mode (`mirror_mode`).
pub fn wanted() -> bool {
    static W: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *W.get_or_init(|| matches!(std::env::var("DS_RESIDENT").ok().as_deref(), Some("1" | "2")))
}

fn wanted_mode2() -> bool {
    static W: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *W.get_or_init(|| std::env::var("DS_RESIDENT").ok().as_deref() == Some("2"))
}

/// Mirror mode was asked for (`DS_RESIDENT=2`; whether it runs is known
/// after its self-test: `mirror_mode`). Shader specs follow this, so the
/// pipelines precompiled at a session's start match its translations.
pub fn mirror_wanted() -> bool {
    wanted_mode2()
}

/// Residency runs (wanted, and the mirror was made).
pub fn enabled() -> bool {
    ON.load(Ordering::Acquire)
}

/// The page table, arming and write faults run: residency, or section-backed
/// guest memory (its CPU-write tracking, `take_written`).
pub fn tracking() -> bool {
    ON.load(Ordering::Acquire) || crate::sections::on() || crate::mem::views_on()
}

/// The parts of [lo, hi) the write tracking covers: all guest memory with
/// section-backed memory, the mirror's span with residency alone, else the
/// mixed range's view windows (`mem::view_window`).
fn tracked_parts(lo: u64, hi: u64) -> Vec<(u64, u64)> {
    if crate::sections::on() {
        return vec![(lo, hi)];
    }
    if enabled() {
        let (a, e) = (lo.max(MIRROR_BASE), hi.min(MIRROR_END));
        return if a < e { vec![(a, e)] } else { Vec::new() };
    }
    let mut out: Vec<(u64, u64)> = Vec::new();
    let mut p = lo;
    while p < hi {
        // (2 MiB steps: no window of the mixed range is smaller.)
        let e = (((p >> 21) + 1) << 21).min(hi);
        if crate::mem::view_window(p) {
            match out.last_mut() {
                Some(l) if l.1 == p => l.1 = e,
                _ => out.push((p, e)),
            }
        }
        p = e;
    }
    out
}

fn chunk(addr: u64, create: bool) -> Option<&'static Chunk> {
    let slot = TABLE.get((addr >> CHUNK_SHIFT) as usize)?;
    let p = slot.load(Ordering::Acquire);
    if !p.is_null() {
        return Some(unsafe { &*p });
    }
    if !create {
        return None;
    }
    let new = Box::into_raw(Box::new(Chunk {
        pages: (0..CHUNK_PAGES).map(|_| AtomicU8::new(0)).collect(),
        seqs: (0..CHUNK_PAGES).map(|_| AtomicU64::new(0)).collect(),
        faulted: (0..CHUNK_PAGES).map(|_| AtomicU32::new(0)).collect(),
        writing: (0..CHUNK_PAGES).map(|_| AtomicU8::new(0)).collect(),
        stamp: (0..CHUNK_PAGES).map(|_| AtomicU32::new(0)).collect(),
        vol_until: (0..CHUNK_PAGES).map(|_| AtomicU32::new(0)).collect(),
        vol_level: (0..CHUNK_PAGES).map(|_| AtomicU8::new(0)).collect(),
        attn: (0..CHUNK_PAGES / 64).map(|_| AtomicU64::new(u64::MAX)).collect(),
        cur: (0..CHUNK_PAGES / 64).map(|_| AtomicU64::new(0)).collect(),
        gdirty: (0..CHUNK_PAGES / 64).map(|_| AtomicU64::new(0)).collect(),
        gprot: (0..CHUNK_PAGES / 64).map(|_| AtomicU64::new(0)).collect(),
        rseq: (0..CHUNK_PAGES).map(|_| AtomicU64::new(0)).collect(),
        harmed: (0..CHUNK_PAGES / 64).map(|_| AtomicU64::new(0)).collect(),
        host_ev: AtomicU64::new(0),
    }));
    match slot.compare_exchange(std::ptr::null_mut(), new, Ordering::AcqRel, Ordering::Acquire) {
        Ok(_) => Some(unsafe { &*new }),
        Err(cur) => {
            // Another thread made it first.
            drop(unsafe { Box::from_raw(new) });
            Some(unsafe { &*cur })
        }
    }
}

fn idx(addr: u64) -> usize {
    ((addr >> 12) as usize) & (CHUNK_PAGES - 1)
}

fn slot(c: &Chunk, addr: u64) -> &AtomicU8 {
    &c.pages[idx(addr)]
}

/// The state of the page holding `addr` (0 when untracked).
pub fn state(addr: u64) -> u8 {
    chunk(addr, false).map_or(0, |c| slot(c, addr).load(Ordering::Acquire) & !ST_BUSY)
}

/// Marks [base, base+len) host class: never armed (disarmed now if armed).
pub fn mark_host(base: u64, len: u64) {
    let (lo, hi) = (base & !(PAGE - 1), (base + len).next_multiple_of(PAGE));
    disarm_range(lo, hi);
    let mut p = lo;
    let mut n = 0;
    while p < hi {
        if let Some(c) = chunk(p, true) {
            if slot(c, p).fetch_or(ST_HOST, Ordering::AcqRel) & ST_HOST == 0 {
                n += 1;
            }
            slot(c, p).fetch_and(!ST_CLEAN, Ordering::AcqRel);
            flag(c, idx(p));
        }
        p += PAGE;
    }
    STATS[S_HOSTED].fetch_add(n, Ordering::Relaxed);
}

// Linux section tracking has logical guest permissions in `os`, and temporary read-only
// permissions here. Keeping those separate avoids allocating/splitting the VMA map per fault.
// Mirror/held-page modes still use the service thread (they can require GPU readback).
#[cfg(target_os = "linux")]
static LINUX_DIRECT: AtomicBool = AtomicBool::new(false);
// Leaving a page unarmed saves faults but reports it dirty at every epoch. On Linux
// the old 32..1024-frame backoff copied unchanged pages into VRAM thousands of times.
// Keep it available for relay comparisons; exact tracking is the default. Read by
// the signal handler, so initialize outside it and use only an atomic load there.
#[cfg(target_os = "linux")]
static LINUX_VOLATILE_BACKOFF: AtomicBool = AtomicBool::new(false);
#[cfg(target_os = "linux")]
static LINUX_REMAP: AtomicU32 = AtomicU32::new(0);
#[cfg(target_os = "linux")]
const LINUX_WRITER: u32 = 1 << 31;
#[cfg(target_os = "linux")]
static LINUX_FAULT_TIMING: AtomicBool = AtomicBool::new(false);
#[cfg(target_os = "linux")]
static LINUX_FAULT_SAMPLES: [AtomicU64; 3] = [const { AtomicU64::new(0) }; 3];

#[cfg(target_os = "linux")]
fn linux_clock_ns(clock: libc::clockid_t) -> u64 {
    let mut ts = libc::timespec { tv_sec: 0, tv_nsec: 0 };
    if unsafe { libc::syscall(libc::SYS_clock_gettime, clock, &mut ts) } != 0 { return 0; }
    ts.tv_sec as u64 * 1_000_000_000 + ts.tv_nsec as u64
}

#[cfg(target_os = "linux")]
pub fn linux_fault_samples() -> [u64; 3] {
    LINUX_FAULT_SAMPLES.each_ref().map(|x| x.load(Ordering::Relaxed))
}

#[cfg(target_os = "linux")]
pub fn linux_fault_timing() -> String {
    let [n, wall, cpu] = LINUX_FAULT_SAMPLES.each_ref().map(|x| x.load(Ordering::Relaxed));
    format!("Linux direct fault timing: {n} samples (1/256 attempts), mean {:.2} us wall / {:.2} us thread CPU; excludes kernel signal delivery and service-thread faults", wall as f64 / n.max(1) as f64 / 1000.0, cpu as f64 / n.max(1) as f64 / 1000.0)
}
#[cfg(target_os = "linux")]
pub fn init_linux_faults() {
    crate::platform::write_watch::initialize();
    let on = crate::sections::on() && !wanted() && !crate::platform::write_watch::enabled() && std::env::var("DS_LINUX_DIRECT_FAULT").ok().as_deref() != Some("0");
    let backoff = std::env::var("DS_LINUX_VOLATILE_BACKOFF").ok().as_deref() == Some("1");
    LINUX_VOLATILE_BACKOFF.store(backoff, Ordering::Release);
    LINUX_FAULT_TIMING.store(std::env::var("DS_LINUX_FAULT_TIMING").ok().as_deref() == Some("1"), Ordering::Release);
    LINUX_DIRECT.store(on, Ordering::Release);
    REMAP_RANGE_GATE.store(std::env::var("DS_REMAP_RANGE_GATE").ok().as_deref() != Some("0"), Ordering::Release);
    crate::info!("Linux signal readers during remaps: {}", if REMAP_RANGE_GATE.load(Ordering::Relaxed) { "admitted outside the remap's range" } else { "all wait (DS_REMAP_RANGE_GATE=0)" });
    crate::info!("Linux resident direct write-fault recovery: {on} (section tracking only)");
    crate::info!("Linux resident volatile backoff: {backoff} (false: exact dirty tracking; true: legacy 32..1024-frame cooldown)");
}

/// VirtualQuery's actual permissions: arming is temporary, distinct from guest permissions.
/// Bound each query's scan; callers already walk successive regions.
#[cfg(target_os = "linux")]
pub fn linux_protection_span(a: u64, end: u64) -> (u64, bool) {
    if !LINUX_DIRECT.load(Ordering::Acquire) { return (end, false); }
    let armed = state(a) & ST_ARMED != 0;
    let mut p = (a & !(PAGE - 1)) + PAGE;
    let stop = end.min(p.saturating_add(63 * PAGE));
    while p < stop && (state(p) & ST_ARMED != 0) == armed { p += PAGE; }
    (p.min(end), armed)
}

/// Admits a signal-context reader of guest mappings (async-signal-safe): false while a remap
/// holds or is taking `remap_guard` (the fault then goes to the service thread). Pair with
/// `signal_leave`.
#[cfg(target_os = "linux")]
pub fn signal_admit() -> bool {
    let gate = LINUX_REMAP.load(Ordering::Acquire);
    gate & LINUX_WRITER == 0 && LINUX_REMAP.compare_exchange(gate, gate + 1, Ordering::AcqRel, Ordering::Acquire).is_ok()
}

/// `signal_admit` for a reader of page `page` alone: admitted during a remap too, when the
/// page lies outside the range the remap changes (`remap_guard_for`). The reader counts as
/// admitted before it reads the range, so a remap that publishes its range later waits for
/// it; a range read while a later remap replaces it belongs to one that has not begun its
/// change (it is still waiting for this reader). Pair with `signal_leave`.
#[cfg(target_os = "linux")]
pub fn signal_admit_page(page: u64) -> bool {
    if !REMAP_RANGE_GATE.load(Ordering::Relaxed) {
        return signal_admit();
    }
    let mut gate = LINUX_REMAP.load(Ordering::Acquire);
    loop {
        match LINUX_REMAP.compare_exchange_weak(gate, gate + 1, Ordering::AcqRel, Ordering::Acquire) {
            Ok(_) => break,
            Err(g) => gate = g,
        }
    }
    if gate & LINUX_WRITER == 0 {
        return true;
    }
    let (lo, hi) = (REMAP_LO.load(Ordering::Acquire), REMAP_HI.load(Ordering::Acquire));
    if page.saturating_add(PAGE) > lo && page < hi {
        LINUX_REMAP.fetch_sub(1, Ordering::Release);
        return false;
    }
    REMAP_OUTSIDE.fetch_add(1, Ordering::Relaxed);
    true
}

/// The range the remap holding (or taking) `remap_guard` changes; the whole space for one
/// that names none.
#[cfg(target_os = "linux")]
static REMAP_LO: AtomicU64 = AtomicU64::new(0);
#[cfg(target_os = "linux")]
static REMAP_HI: AtomicU64 = AtomicU64::new(u64::MAX);
/// Signal readers admitted during a remap, their page outside its range.
#[cfg(target_os = "linux")]
pub static REMAP_OUTSIDE: AtomicU64 = AtomicU64::new(0);
/// `DS_REMAP_RANGE_GATE=0`: every signal reader waits out every remap (the old gate).
#[cfg(target_os = "linux")]
pub static REMAP_RANGE_GATE: AtomicBool = AtomicBool::new(true);

#[cfg(target_os = "linux")]
pub fn signal_leave() {
    LINUX_REMAP.fetch_sub(1, Ordering::Release);
}

/// Async-signal-safe: permanent table chunks, atomics and one mprotect syscall only.
/// Never wait on ST_BUSY: the interrupted thread might hold it. Remaps exclude this path
/// with a lock-free reader count; the slow service handles collisions and readback faults.
#[cfg(target_os = "linux")]
pub fn linux_write_fault(addr: u64) -> bool {
    if !LINUX_DIRECT.load(Ordering::Acquire) { return false; }
    let Some(c) = chunk(addr, false) else { return false };
    let gate = LINUX_REMAP.load(Ordering::Acquire);
    if gate & LINUX_WRITER != 0 || LINUX_REMAP.compare_exchange(gate, gate + 1, Ordering::AcqRel, Ordering::Acquire).is_err() { return false; }
    let timed = LINUX_FAULT_TIMING.load(Ordering::Relaxed) && STATS[S_FAULTS].load(Ordering::Relaxed) % 256 == 0;
    let start = if timed { (linux_clock_ns(libc::CLOCK_MONOTONIC), linux_clock_ns(libc::CLOCK_THREAD_CPUTIME_ID)) } else { (0, 0) };
    let i = idx(addr);
    let s = &c.pages[i];
    let cur = s.load(Ordering::Acquire);
    let mut ok = false;
    if cur & (ST_ARMED | ST_BUSY) == ST_ARMED && s.compare_exchange(cur, cur | ST_BUSY, Ordering::AcqRel, Ordering::Acquire).is_ok() {
        let page = addr & !(PAGE - 1);
        if unsafe { libc::syscall(libc::SYS_mprotect, page, PAGE, libc::PROT_READ | libc::PROT_WRITE) } == 0 {
            let frame = super::FRAMES.load(Ordering::Relaxed) as u32 + 1;
            let volatile = note_write_fault(c, i, frame);
            // Publish dirty state before attention, as on_fault does. A take can clear
            // attention concurrently: flagging first let it still see CLEAN|ARMED|BUSY,
            // skip the page, then lose this fault when the state became dirty afterwards.
            s.store((cur & !(ST_ARMED | ST_VALID | ST_BUSY | ST_CLEAN)) | if volatile { ST_VOLATILE } else { 0 }, Ordering::Release);
            flag(c, i);
            STATS[S_FAULTS].fetch_add(1, Ordering::Relaxed);
            // The direct counter is in signals; protect()'s call/time averages cover only
            // its timed calls, not these signal-context syscalls.
            ok = true;
        } else { s.store(cur, Ordering::Release); }
    }
    LINUX_REMAP.fetch_sub(1, Ordering::Release);
    if timed && ok {
        let cpu = linux_clock_ns(libc::CLOCK_THREAD_CPUTIME_ID).saturating_sub(start.1);
        let wall = linux_clock_ns(libc::CLOCK_MONOTONIC).saturating_sub(start.0);
        LINUX_FAULT_SAMPLES[1].fetch_add(wall, Ordering::Relaxed);
        LINUX_FAULT_SAMPLES[2].fetch_add(cpu, Ordering::Relaxed);
        LINUX_FAULT_SAMPLES[0].fetch_add(1, Ordering::Relaxed);
    }
    ok
}

fn protect(a: u64, n: u64, prot: u32) -> bool {
    use crate::platform::os::Win32::System::Memory::{VirtualProtect, PAGE_READONLY};
    let t = std::time::Instant::now();
    let mut old = 0u32;
    #[cfg(windows)]
    let ok = unsafe { VirtualProtect(a as _, n as usize, prot, &mut old) } != 0;
    #[cfg(target_os = "linux")]
    let ok = if LINUX_DIRECT.load(Ordering::Acquire) && matches!(prot, PAGE_READONLY | crate::platform::os::Win32::System::Memory::PAGE_READWRITE) {
        let native = if prot == PAGE_READONLY { libc::PROT_READ } else { libc::PROT_READ | libc::PROT_WRITE };
        unsafe { libc::mprotect(a as _, n as usize, native) == 0 }
    } else { unsafe { VirtualProtect(a as _, n as usize, prot, &mut old) != 0 } };
    let k = if prot == PAGE_READONLY { 0 } else { S_UNPROTECTS - S_PROTECTS };
    STATS[S_PROTECTS + k].fetch_add(1, Ordering::Relaxed);
    STATS[S_PROTECT_NS + k].fetch_add(t.elapsed().as_nanos() as u64, Ordering::Relaxed);
    ok
}

/// Protects [a, e) with `prot`, one call per memory region (a call cannot
/// span two section views). With `need_rw` only when every region is plain
/// read-write now; false (nothing changed) otherwise.
fn protect_regions(a: u64, e: u64, prot: u32, need_rw: bool) -> bool {
    use crate::platform::os::Win32::System::Memory::{VirtualQuery, MEMORY_BASIC_INFORMATION, MEM_COMMIT, PAGE_READWRITE};
    let mut regions = Vec::new();
    let mut p = a;
    while p < e {
        let mut info: MEMORY_BASIC_INFORMATION = unsafe { std::mem::zeroed() };
        if unsafe { VirtualQuery(p as _, &mut info, std::mem::size_of::<MEMORY_BASIC_INFORMATION>()) } == 0 {
            return false;
        }
        if need_rw && (info.State != MEM_COMMIT || info.Protect != PAGE_READWRITE) {
            return false;
        }
        let end = (info.BaseAddress as u64 + info.RegionSize as u64).min(e);
        regions.push((p, end));
        p = end;
    }
    for (k, &(s, t)) in regions.iter().enumerate() {
        if !protect(s, t - s, prot) {
            for &(s0, t0) in &regions[..k] {
                protect(s0, t0 - s0, PAGE_READWRITE);
            }
            return false;
        }
    }
    true
}

#[derive(Clone, Copy, PartialEq, Eq)]
enum RegionKind {
    /// Committed, plain read-write.
    Rw,
    /// Committed, and the guest cannot write it (read-only, no access).
    NotWritable,
    /// Not committed (reserved, placeholders, unmapped tiles of partially
    /// resident textures): nothing to track.
    Uncommitted,
    /// Anything else (guard, executable read-write).
    Other,
}

/// The memory region holding `p`, clipped to `e`: (end, kind).
fn region_kind(p: u64, e: u64) -> (u64, RegionKind) {
    use crate::platform::os::Win32::System::Memory::{VirtualQuery, MEMORY_BASIC_INFORMATION, MEM_COMMIT, PAGE_EXECUTE, PAGE_EXECUTE_READ, PAGE_NOACCESS, PAGE_READONLY, PAGE_READWRITE};
    let mut info: MEMORY_BASIC_INFORMATION = unsafe { std::mem::zeroed() };
    if unsafe { VirtualQuery(p as _, &mut info, std::mem::size_of::<MEMORY_BASIC_INFORMATION>()) } == 0 {
        return (e, RegionKind::Other);
    }
    let end = (info.BaseAddress as u64 + info.RegionSize as u64).min(e).max(p + PAGE);
    let kind = if info.State != MEM_COMMIT {
        RegionKind::Uncommitted
    } else if info.Protect == PAGE_READWRITE {
        RegionKind::Rw
    } else if matches!(info.Protect, PAGE_READONLY | PAGE_EXECUTE_READ | PAGE_NOACCESS | PAGE_EXECUTE) {
        RegionKind::NotWritable
    } else {
        RegionKind::Other
    };
    (end, kind)
}

/// Whether [a, a+n) is committed plain read-write memory now.
/// `DS_TRACE_PAGES=<hex lo>-<hex hi>` (diagnosis): mirror mode's page
/// events in that range are logged (GPU-written dword runs per batch at
/// completion, CPU write faults with the faulting instruction, releases).
fn trace_range() -> (u64, u64) {
    static R: std::sync::OnceLock<(u64, u64)> = std::sync::OnceLock::new();
    *R.get_or_init(|| {
        let v = std::env::var("DS_TRACE_PAGES").unwrap_or_default();
        let num = |x: &str| u64::from_str_radix(x.trim().trim_start_matches("0x"), 16).ok();
        v.split_once('-').and_then(|(a, b)| Some((num(a)?, num(b)?))).unwrap_or((0, 0))
    })
}

fn traced(p: u64) -> bool {
    let (lo, hi) = trace_range();
    p + PAGE > lo && p < hi
}

/// Logs a traced page event (at most 3,000 lines).
fn trace_event(p: u64, what: std::fmt::Arguments) {
    static N: AtomicU64 = AtomicU64::new(0);
    if N.fetch_add(1, Ordering::Relaxed) < 3000 {
        crate::info!("mm trace {p:#x}: {what} (flip {}, {})", super::FRAMES.load(Ordering::Relaxed), crate::log::thread_name());
    }
}

/// The runs of set bits of 32 mask words as "first+count" dword runs.
fn mask_runs(masks: &[u32]) -> String {
    let mut out: Vec<String> = Vec::new();
    let mut k = 0usize;
    while k < 32 * masks.len() {
        if masks[k / 32] >> (k % 32) & 1 != 0 {
            let s = k;
            while k < 32 * masks.len() && masks[k / 32] >> (k % 32) & 1 != 0 {
                k += 1;
            }
            out.push(format!("{s}+{}", k - s));
        } else {
            k += 1;
        }
    }
    out.join(",")
}

/// Whether the page holding `a` is committed and readable now.
fn readable_now(a: u64) -> bool {
    use crate::platform::os::Win32::System::Memory::{VirtualQuery, MEMORY_BASIC_INFORMATION, MEM_COMMIT, PAGE_GUARD, PAGE_NOACCESS};
    let mut info: MEMORY_BASIC_INFORMATION = unsafe { std::mem::zeroed() };
    if unsafe { VirtualQuery(a as _, &mut info, std::mem::size_of::<MEMORY_BASIC_INFORMATION>()) } == 0 {
        return false;
    }
    info.State == MEM_COMMIT && info.Protect & (PAGE_NOACCESS | PAGE_GUARD) == 0 && info.Protect != 0
}

fn plain_rw(a: u64, n: u64) -> bool {
    use crate::platform::os::Win32::System::Memory::{VirtualQuery, MEMORY_BASIC_INFORMATION, MEM_COMMIT, PAGE_READWRITE};
    let mut p = a;
    while p < a + n {
        let mut info: MEMORY_BASIC_INFORMATION = unsafe { std::mem::zeroed() };
        if unsafe { VirtualQuery(p as _, &mut info, std::mem::size_of::<MEMORY_BASIC_INFORMATION>()) } == 0 {
            return false;
        }
        if info.State != MEM_COMMIT || info.Protect != PAGE_READWRITE {
            return false;
        }
        p = info.BaseAddress as u64 + info.RegionSize as u64;
    }
    true
}

/// Hosted write tracking (`status`): pages takes found answered by the host
/// (or disarmed through it), write faults on pages armed through the host
/// that reached us anyway, pages answered between their arming and its
/// marks, pages joined (armed through the host by another already), chunk
/// scans (`reconcile`), armings through the host that failed.
static HOST_STATS: [AtomicU64; 6] = [const { AtomicU64::new(0) }; 6];
const H_ANSWERED: usize = 0;
const H_IN_PROCESS: usize = 1;
const H_LOST: usize = 2;
const H_JOINED: usize = 3;
const H_SCANS: usize = 4;
const H_FAILED: usize = 5;

/// Write tracking arms through the host (`--host`, section-backed memory):
/// the host answers its write faults, so they leave the guest's red zone
/// intact (`hosted`).
#[inline]
fn via_host() -> bool {
    crate::hosted::on() && (crate::sections::on() || crate::mem::views_on())
}

/// Whether the host's bit of page `q` is set (armed through the host).
#[inline]
fn host_armed(q: u64) -> bool {
    crate::hosted::armed_word(q & !(64 * PAGE - 1)) >> ((q >> 12) & 63) & 1 != 0
}

/// Arms [a, a+n) (one region, plain read-write) through the host.
fn arm_host(a: u64, n: u64) -> bool {
    let t = std::time::Instant::now();
    let r = crate::hosted::arm(a, n);
    STATS[S_PROTECTS].fetch_add(1, Ordering::Relaxed);
    STATS[S_PROTECT_NS].fetch_add(t.elapsed().as_nanos() as u64, Ordering::Relaxed);
    if let Err(e) = &r {
        if HOST_STATS[H_FAILED].fetch_add(1, Ordering::Relaxed) < 8 {
            crate::warn!("resident: arming {a:#x}+{n:#x} through the host failed: {e}");
        }
    }
    r.is_ok()
}

/// Clears page `i`'s host-armed bit; whether it was set.
#[inline]
fn clear_harmed(c: &Chunk, i: usize) -> bool {
    let m = 1u64 << (i & 63);
    c.harmed[i >> 6].load(Ordering::Acquire) & m != 0 && c.harmed[i >> 6].fetch_and(!m, Ordering::AcqRel) & m != 0
}

/// Hosted write tracking: the pages of chunk `c` (at `base`) armed through
/// the host whose bit the host has cleared since (a write fault it answered
/// for the page or for its 64-page word, a disarm, a mapping change) end
/// their arming as a write fault here would (`on_fault`). Skipped while the
/// host's count of cleared bits for the chunk is the one a scan read before
/// it began (an arming whose bits are cleared before its marks finds that
/// itself: `arm_with`).
fn reconcile(c: &Chunk, base: u64) {
    let ev = crate::hosted::leaf_events(base);
    if c.host_ev.load(Ordering::Acquire) == ev {
        return;
    }
    HOST_STATS[H_SCANS].fetch_add(1, Ordering::Relaxed);
    let frame = super::FRAMES.load(Ordering::Relaxed) as u32 + 1;
    for w in 0..CHUNK_PAGES / 64 {
        let ours = c.harmed[w].load(Ordering::Acquire);
        if ours == 0 {
            continue;
        }
        let gone = ours & !crate::hosted::armed_word(base + (w as u64) * 64 * PAGE);
        if gone == 0 {
            continue;
        }
        // (Claimed: each page is done by one scan.)
        let mut r = c.harmed[w].fetch_and(!gone, Ordering::AcqRel) & gone;
        while r != 0 {
            let i = w * 64 + r.trailing_zeros() as usize;
            r &= r - 1;
            host_answered(c, i, frame);
        }
    }
    c.host_ev.store(ev, Ordering::Release);
}

/// Page `i` of `c` lost its host bit while armed through the host: no
/// longer armed or clean, volatile when written again soon (as `on_fault`).
/// With the faults read from the host's ring (`faults_from_ring`) only the
/// page written counts toward volatility (`host_fault`): the host's answer
/// for a 64-page word (`hosted::set_word_mode`) ends the arming of the rest
/// with no write of theirs, and counting them made a hot page's whole word
/// volatile, reported in every epoch.
fn host_answered(c: &Chunk, i: usize, frame: u32) {
    let s = &c.pages[i];
    loop {
        let cur = s.load(Ordering::Acquire);
        if cur & ST_BUSY != 0 {
            // A protection change in progress (a disarm): retry after.
            std::hint::spin_loop();
            continue;
        }
        if cur & ST_ARMED == 0 {
            // Disarmed or reset meanwhile: their marks stand.
            return;
        }
        if s.compare_exchange(cur, cur | ST_BUSY, Ordering::AcqRel, Ordering::Acquire).is_ok() {
            let volatile = !faults_from_ring() && note_write_fault(c, i, frame);
            s.store((cur & !(ST_ARMED | ST_VALID | ST_BUSY | ST_CLEAN)) | if volatile { ST_VOLATILE } else { 0 }, Ordering::Release);
            flag(c, i);
            HOST_STATS[H_ANSWERED].fetch_add(1, Ordering::Relaxed);
            return;
        }
    }
}

/// A write fault on page `i` of `c` in `frame`: whether the page turns
/// volatile (written again within VOLATILE_FRAMES of its last fault: it
/// stays unarmed VOL_BASE << level frames, its level one up). A fault more
/// than QUIET_FRAMES after the page's last stay ended resets its level.
fn note_write_fault(c: &Chunk, i: usize, frame: u32) -> bool {
    #[cfg(target_os = "linux")]
    if !LINUX_VOLATILE_BACKOFF.load(Ordering::Relaxed) { return false; }
    let last = c.faulted[i].swap(frame, Ordering::AcqRel);
    #[cfg(windows)]
    let volatile = last != 0 && frame.wrapping_sub(last) <= VOLATILE_FRAMES;
    #[cfg(target_os = "linux")]
    let volatile = last != 0 && (frame.wrapping_sub(last) <= 64 ||
        (c.vol_level[i].load(Ordering::Acquire) != 0 && frame.wrapping_sub(c.vol_until[i].load(Ordering::Acquire)) <= VOLATILE_FRAMES));
    if volatile {
        let level = c.vol_level[i].load(Ordering::Acquire).saturating_add(1).min(VOL_MAX_LEVEL);
        c.vol_level[i].store(level, Ordering::Release);
        c.vol_until[i].store(frame.wrapping_add(VOL_BASE << level), Ordering::Release);
        STATS[S_VOLATILE].fetch_add(1, Ordering::Relaxed);
    } else if c.vol_level[i].load(Ordering::Acquire) != 0 && frame.wrapping_sub(c.vol_until[i].load(Ordering::Acquire)) as i32 > QUIET_FRAMES as i32 {
        c.vol_level[i].store(0, Ordering::Release);
        STATS[S_VOL_DECAYED].fetch_add(1, Ordering::Relaxed);
    }
    volatile
}

/// Whether the host's answered faults reach `host_fault` from its ring
/// (front end 2's fault reader, `fe2::watch`, runs: hosted with front end 2
/// on), so that `host_answered` leaves volatility to them.
fn faults_from_ring() -> bool {
    crate::hosted::on() && super::fe2::on() && !crate::sections::on()
}

/// A write fault the host answered on guest `page` (front end 2's reader of
/// the host's ring, for view windows): the page written, so its volatility
/// (`note_write_fault`); its arming ends when a take finds its host bit gone
/// (`reconcile`), before or after this.
pub fn host_fault(page: u64) {
    if !via_host() || !faults_from_ring() {
        return;
    }
    let Some(c) = chunk(page, false) else { return };
    let i = idx(page);
    STATS[S_HOST_FAULTS].fetch_add(1, Ordering::Relaxed);
    let frame = super::FRAMES.load(Ordering::Relaxed) as u32 + 1;
    if note_write_fault(c, i, frame) {
        c.pages[i].fetch_or(ST_VOLATILE, Ordering::AcqRel);
        flag(c, i);
    }
}

/// Hosted write tracking, under the host's arm lock (`arm_with`): the mark
/// of page `q` (busy), just armed through the host (`armed_now`) or in a
/// region read-only when looked at (for the guest, or for another's arming
/// through the host).
fn host_mark(c: &Chunk, q: u64, armed_now: bool) -> u8 {
    let i = idx(q);
    if host_armed(q) {
        // Ours to watch while its host bit stays (marked before the second
        // look: a bit the host clears after it is found by `reconcile`).
        c.harmed[i >> 6].fetch_or(1u64 << (i & 63), Ordering::AcqRel);
        if host_armed(q) {
            if !armed_now {
                HOST_STATS[H_JOINED].fetch_add(1, Ordering::Relaxed);
            }
            return ST_ARMED;
        }
        clear_harmed(c, i);
        HOST_STATS[H_LOST].fetch_add(1, Ordering::Relaxed);
        return 0;
    }
    if armed_now {
        // The host answered a write there already.
        HOST_STATS[H_LOST].fetch_add(1, Ordering::Relaxed);
        return 0;
    }
    // Read-only without a host bit: no arming can begin under the lock and
    // the host only makes pages writable, so the guest's own protection,
    // unless the host answered a fault there since the region was looked
    // at (writable now: flagged for the next take).
    if region_kind(q, q + PAGE).1 == RegionKind::NotWritable {
        ST_GUEST_RO
    } else {
        0
    }
}

/// Arms the pages of [base, base+len) (rounded out to pages): read-only,
/// device-valid. Pages of the host class, busy or already armed are left;
/// a run that is not plain read-write memory is not armed. Returns the
/// pages armed.
pub fn arm(base: u64, len: u64) -> u64 {
    arm_with(base, len, ST_VALID)
}

/// `arm`, setting `extra` besides ST_ARMED (ST_VALID for the mirror; nothing
/// for write tracking).
fn arm_with(base: u64, len: u64, extra: u8) -> u64 {
    use crate::platform::os::Win32::System::Memory::PAGE_READONLY;
    // No mapping change meanwhile (`ARM_LOCK`). Linux takes hold this guard across
    // queries and attention clearing too; recursive read cannot block behind a writer.
    #[cfg(windows)]
    let _arming = ARM_LOCK.read();
    #[cfg(target_os = "linux")]
    let _arming = ARM_LOCK.read_recursive();
    // Write tracking through the host (`via_host`): no other arm or disarm
    // between a page's host bit and protection being looked at and its
    // marks (`hosted::arm_lock`; the host itself only makes pages
    // writable).
    let host = extra == 0 && via_host();
    let _host_lock = host.then(crate::hosted::arm_lock);
    let (lo, hi) = (base & !(PAGE - 1), (base + len).next_multiple_of(PAGE));
    let mut armed = 0;
    let mut p = lo;
    // Runs of pages taken busy, protected with one call each.
    let mut run: Option<(u64, u64)> = None;
    // Region by region (a protection call cannot span two section views):
    // plain read-write regions are armed, regions the guest cannot write are
    // marked so (clean once taken), others are left.
    let flush = |run: &mut Option<(u64, u64)>, armed: &mut u64| {
        if let Some((a, e)) = run.take() {
            let mut p = a;
            while p < e {
                let (r_end, kind) = region_kind(p, e);
                let ok = kind == RegionKind::Rw && if host { arm_host(p, r_end - p) } else { protect(p, r_end - p, PAGE_READONLY) };
                let mark = if ok {
                    ST_ARMED | extra
                } else if matches!(kind, RegionKind::NotWritable | RegionKind::Uncommitted) {
                    ST_GUEST_RO
                } else {
                    if kind == RegionKind::Other {
                        STATS[S_OTHER_PROT].fetch_add((r_end - p) / PAGE, Ordering::Relaxed);
                    }
                    0
                };
                let mut q = p;
                while q < r_end {
                    if let Some(c) = chunk(q, false) {
                        let m = if host && (ok || kind == RegionKind::NotWritable) { host_mark(c, q, ok) } else { mark };
                        let s = slot(c, q);
                        s.store((s.load(Ordering::Acquire) | m) & !ST_BUSY, Ordering::Release);
                        if m == 0 {
                            flag(c, idx(q));
                        }
                        if m & ST_ARMED != 0 {
                            *armed += 1;
                            // (A view page armed at a take: front end 2's
                            // arming too, `fe2::adopt_armed`.)
                            if host && crate::mem::view_window(q) {
                                super::fe2::adopt_armed(q);
                            }
                        }
                    }
                    q += PAGE;
                }
                p = r_end;
            }
        }
    };
    while p < hi {
        let c = chunk(p, true).expect("chunk");
        let s = slot(c, p);
        let cur = s.load(Ordering::Acquire);
        let take = cur & (ST_HOST | ST_BUSY | ST_ARMED) == 0 && c.writing[idx(p)].load(Ordering::Acquire) == 0 && !is_dirty(c, idx(p)) && s.compare_exchange(cur, cur | ST_BUSY, Ordering::AcqRel, Ordering::Acquire).is_ok();
        if take {
            run = Some(match run {
                Some((a, e)) if e == p => (a, p + PAGE),
                other => {
                    let mut r = other;
                    flush(&mut r, &mut armed);
                    (p, p + PAGE)
                }
            });
        } else {
            flush(&mut run, &mut armed);
        }
        p += PAGE;
    }
    flush(&mut run, &mut armed);
    STATS[S_ARMED].fetch_add(armed, Ordering::Relaxed);
    armed
}

/// Disarms the armed pages of [lo, hi) (page-aligned): writable again and
/// device-invalid. Runs of armed pages are taken busy and made writable with
/// one call each (an import window covers up to 16,384 pages).
fn disarm_range(lo: u64, hi: u64) {
    use crate::platform::os::Win32::System::Memory::PAGE_READWRITE;
    let host = via_host();
    let release = |a: u64, e: u64| {
        // Pages armed through the host: writable, then their host bits
        // cleared (`hosted::disarm`), before anything else is made
        // writable (a writable page keeping its bit would hide writes from
        // front end 2).
        if host {
            crate::hosted::disarm(a, e - a);
        }
        protect_regions(a, e, PAGE_READWRITE, false);
        let mut q = a;
        while q < e {
            if let Some(c) = chunk(q, false) {
                slot(c, q).fetch_and(!(ST_ARMED | ST_VALID | ST_BUSY | ST_CLEAN), Ordering::AcqRel);
                clear_harmed(c, idx(q));
                flag(c, idx(q));
            }
            q += PAGE;
        }
    };
    let mut run: Option<(u64, u64)> = None;
    let mut p = lo;
    while p < hi {
        let Some(c) = chunk(p, false) else {
            if let Some((a, e)) = run.take() {
                release(a, e);
            }
            p = ((p >> CHUNK_SHIFT) + 1) << CHUNK_SHIFT;
            continue;
        };
        let s = slot(c, p);
        let took = loop {
            let cur = s.load(Ordering::Acquire);
            if cur & ST_BUSY != 0 {
                std::hint::spin_loop();
                continue;
            }
            if cur & ST_ARMED == 0 {
                break false;
            }
            if s.compare_exchange(cur, cur | ST_BUSY, Ordering::AcqRel, Ordering::Acquire).is_ok() {
                break true;
            }
        };
        match (took, run) {
            (true, Some((a, e))) if e == p => run = Some((a, p + PAGE)),
            (true, r) => {
                if let Some((a, e)) = r {
                    release(a, e);
                }
                run = Some((p, p + PAGE));
            }
            (false, Some((a, e))) => {
                release(a, e);
                run = None;
            }
            (false, None) => {}
        }
        p += PAGE;
    }
    if let Some((a, e)) = run {
        release(a, e);
    }
}

/// A guest commit, decommit or protection change of [base, base+len): its
/// pages are no longer armed or valid (the new protection is the guest's).
pub fn remapped(base: u64, len: u64) {
    if !tracking() || len == 0 {
        return;
    }
    let (lo, hi) = (base & !(PAGE - 1), base.saturating_add(len).next_multiple_of(PAGE));
    let mut p = lo;
    while p < hi {
        let Some(c) = chunk(p, false) else {
            p = ((p >> CHUNK_SHIFT) + 1) << CHUNK_SHIFT;
            continue;
        };
        let s = slot(c, p);
        loop {
            let cur = s.load(Ordering::Acquire);
            if cur & ST_BUSY != 0 {
                std::hint::spin_loop();
                continue;
            }
            if cur & (ST_ARMED | ST_VALID | ST_CLEAN | ST_GUEST_RO | ST_GPU_W) == 0 || s.compare_exchange(cur, cur & !(ST_ARMED | ST_VALID | ST_CLEAN | ST_GUEST_RO | ST_GPU_W), Ordering::AcqRel, Ordering::Acquire).is_ok() {
                break;
            }
        }
        // (Its host bit goes with the change: `settled`.)
        clear_harmed(c, idx(p));
        // Uncommitted pages a take skipped have no attention: committing
        // them comes here.
        flag(c, idx(p));
        if is_dirty(c, idx(p)) {
            undirty(c, idx(p));
            if let Some(m) = DIRTY.lock().as_mut() {
                m.remove(&p);
            }
            OWN_STATS[O_REMAP_DROPS].fetch_add(1, Ordering::Relaxed);
        }
        if mirror_mode() {
            baseline_drop(p, p + PAGE);
            // The remap set the page's protection anew; batches submitted
            // before it wrote the old mapping (`apply_masks` drops theirs).
            clear_gprot(c, idx(p));
            c.rseq[idx(p)].store(super::exec::next_seq(), Ordering::Release);
        }
        p += PAGE;
    }
    STATS[S_RESETS].fetch_add(1, Ordering::Relaxed);
}

/// Arming and the memory manager's mapping changes exclude each other: a
/// change holds it exclusively from just before the operating-system call
/// until its pages' states are reset after it (`remap_guard`, `settled`);
/// `arm_with` holds it shared. Without it a page armed between the change
/// and the reset kept its read-only protection with its state reset, and
/// the guest's next write to it was not ours to handle (fatal, gate window
/// 20261015-163016).
static ARM_LOCK: parking_lot::RwLock<()> = parking_lot::RwLock::new(());

/// Held by the memory manager across a mapping or protection change of
/// guest memory and the `settled` after it (taken after `mem::GUEST_READERS`
/// where both are; arming threads may hold that one shared).
#[cfg(windows)]
pub fn remap_guard() -> parking_lot::RwLockWriteGuard<'static, ()> {
    ARM_LOCK.write()
}

#[cfg(target_os = "linux")]
pub struct RemapGuard {
    _lock: parking_lot::RwLockWriteGuard<'static, ()>,
    /// Where it was taken, when, and how long taking it waited for the readers.
    at: &'static std::panic::Location<'static>,
    held: std::time::Instant,
    wait_us: u64,
}
#[cfg(target_os = "linux")]
impl Drop for RemapGuard {
    fn drop(&mut self) {
        // (Readers of pages outside the range may still be in: only the writer bit goes.)
        LINUX_REMAP.fetch_and(!LINUX_WRITER, Ordering::AcqRel);
        // Every take and check waits while this is held (`arm_guard`): a long hold is a
        // hitch on all of them at once (2026-11-26: four gfx workers' one-page checks
        // at 107-108 ms together, after mapping changes).
        let us = self.held.elapsed().as_micros() as u64;
        REMAP_STATS[0].fetch_add(1, Ordering::Relaxed);
        REMAP_STATS[1].fetch_add(us, Ordering::Relaxed);
        REMAP_STATS[2].fetch_add(self.wait_us, Ordering::Relaxed);
        REMAP_STATS[3].fetch_max(us, Ordering::Relaxed);
        let (at, wait) = (self.at, self.wait_us);
        if us >= 5_000 && REMAP_LOGGED.fetch_add(1, Ordering::Relaxed) < 400 {
            crate::info!("mapping change at {}:{} held every take {us} us (waited {wait} us for readers)", at.file().rsplit('/').next().unwrap_or(""), at.line());
        }
        super::hitch::note_since(super::hitch::Kind::Mirror, self.held, || {
            format!("mapping change held every take ({}:{}; it waited {} us for the readers first)", at.file().rsplit('/').next().unwrap_or(""), at.line(), wait)
        });
    }
}
/// Mapping changes under `remap_guard`: [count, us held, us waited for readers, longest hold us].
#[cfg(target_os = "linux")]
static REMAP_LOGGED: AtomicU64 = AtomicU64::new(0);
#[cfg(target_os = "linux")]
pub static REMAP_STATS: [AtomicU64; 4] = [const { AtomicU64::new(0) }; 4];
#[cfg(target_os = "linux")]
#[track_caller]
pub fn remap_guard() -> RemapGuard {
    remap_guard_at(0, u64::MAX, std::panic::Location::caller())
}

/// `remap_guard` for a change of [base, base+len) alone: signal readers of other pages go on.
#[cfg(target_os = "linux")]
#[track_caller]
pub fn remap_guard_for(base: u64, len: u64) -> RemapGuard {
    let lo = base & !(PAGE - 1);
    remap_guard_at(lo, base.saturating_add(len).saturating_add(PAGE - 1) & !(PAGE - 1), std::panic::Location::caller())
}

#[cfg(windows)]
pub fn remap_guard_for(_base: u64, _len: u64) -> parking_lot::RwLockWriteGuard<'static, ()> {
    ARM_LOCK.write()
}

#[cfg(target_os = "linux")]
fn remap_guard_at(lo: u64, hi: u64, at: &'static std::panic::Location<'static>) -> RemapGuard {
    let t = std::time::Instant::now();
    let lock = ARM_LOCK.write();
    // (Published before the writer bit: a reader that sees the bit reads this range or a
    // later remap's, `signal_admit_page`.)
    REMAP_LO.store(lo, Ordering::Release);
    REMAP_HI.store(hi, Ordering::Release);
    // Only one writer holds ARM_LOCK. Stop admitting signal readers, then wait for the
    // at most one syscall each already admitted reader performs.
    LINUX_REMAP.fetch_or(LINUX_WRITER, Ordering::AcqRel);
    while LINUX_REMAP.load(Ordering::Acquire) != LINUX_WRITER { std::hint::spin_loop(); }
    let wait_us = t.elapsed().as_micros() as u64;
    RemapGuard { _lock: lock, at, held: std::time::Instant::now(), wait_us }
}

/// A mapping or protection change of [base, base+len) is done (called under
/// `remap_guard`): its page states are reset (`remapped`), now that the
/// change set the pages' protection (a page marked armed and clean but
/// writable would hide CPU writes from every later take; one reset before
/// the change left armed pages read-only and unmarked until it landed).
pub fn settled(base: u64, len: u64) {
    #[cfg(target_os = "linux")]
    crate::platform::write_watch::remapped(base, len);
    remapped(base, len);
    crate::hosted::forget(base, len);
    // (Front end 4's held pages there: their pending words died with the
    // mapping; marks and held bits cleared.)
    super::fe2::c4_forgotten(base, len);
    super::fe2::hash_clear_range(base, len);
}

/// An access violation at `addr` (`write`: a write): handled when it is a
/// write to a page we armed (it becomes writable and device-invalid) or one
/// that another thread disarmed meanwhile. Called by the exception handler
/// before anything else; takes no lock a faulting thread of ours could hold
/// (the table's read lock only).
/// Whether recovery can wait for GPU ownership/readback. The Linux fault dispatcher sends
/// these to independent workers so a wait cannot block the faults of a GPU worker.
#[cfg(target_os = "linux")]
pub fn fault_may_wait(addr: u64) -> bool {
    mirror_mode() && chunk(addr, false).is_some_and(|c| is_gprot(c, idx(addr)) || is_dirty(c, idx(addr)))
}

pub fn on_fault(addr: u64, write: bool) -> bool {
    #[cfg(target_os = "linux")]
    let _remap = LINUX_DIRECT.load(Ordering::Acquire).then(|| ARM_LOCK.read());
    use crate::platform::os::Win32::System::Memory::PAGE_READWRITE;
    if !tracking() {
        return false;
    }
    let page = addr & !(PAGE - 1);
    // Mirror mode: a page with GPU-written dwords guest memory lacks gets
    // them first (any access); a page owned by batches in flight waits for
    // them (lazy write-backs: `settle`).
    if mirror_mode() {
        if let Some(c) = chunk(page, false) {
            if is_gprot(c, idx(page)) {
                return settle(page, true);
            }
            if is_dirty(c, idx(page)) {
                return release_dirty(page, true);
            }
        }
        // Settled by another thread between the fault and these checks:
        // the access may be retried when the page now allows it.
        if !write && readable_now(page) {
            STATS[S_RACES].fetch_add(1, Ordering::Relaxed);
            return true;
        }
    }
    if !write {
        return false;
    }
    let Some(c) = chunk(page, false) else { return false };
    let s = slot(c, page);
    loop {
        let cur = s.load(Ordering::Acquire);
        if cur & ST_BUSY != 0 {
            // Armed or disarmed by another thread right now: retry after.
            std::hint::spin_loop();
            continue;
        }
        if cur & ST_ARMED == 0 {
            // Disarmed meanwhile (the write can be retried) or not ours.
            if plain_rw(page, PAGE) {
                STATS[S_RACES].fetch_add(1, Ordering::Relaxed);
                return true;
            }
            return false;
        }
        if s.compare_exchange(cur, cur | ST_BUSY, Ordering::AcqRel, Ordering::Acquire).is_ok() {
            // (Armed through the host, whose bit is gone without its
            // answer: none expected.)
            if clear_harmed(c, idx(page)) {
                HOST_STATS[H_IN_PROCESS].fetch_add(1, Ordering::Relaxed);
            }
            if traced(page) {
                let rip = FAULT_RIP.with(|r| r.get());
                let at = if rip >= crate::mem::GUEST_BASE && rip < crate::mem::GUEST_END { crate::exception::describe_guest(rip) } else { crate::exception::host_symbol(rip) };
                trace_event(page, format_args!("CPU write fault at {at} (state {cur:#x})"));
            }
            // Mirror mode: the page's bytes before the CPU's writes (the
            // mirror's from the CPU's side) for its next copy there.
            if mirror_mode() && cur & ST_VALID != 0 {
                baseline_snap(page);
            }
            protect(page, PAGE, PAGE_READWRITE);
            // Written again soon after the last write: volatile from now on.
            let frame = super::FRAMES.load(Ordering::Relaxed) as u32 + 1;
            let volatile = note_write_fault(c, idx(page), frame);
            s.store((cur & !(ST_ARMED | ST_VALID | ST_BUSY | ST_CLEAN)) | if volatile { ST_VOLATILE } else { 0 }, Ordering::Release);
            flag(c, idx(page));
            STATS[S_FAULTS].fetch_add(1, Ordering::Relaxed);
            return true;
        }
    }
}

/// Statistics of `take_written`: [calls, pages reported, pages armed, pages
/// walked (committed), region queries, reports of unarmable pages].
pub static TAKE_STATS: [AtomicU64; 6] = [const { AtomicU64::new(0) }; 6];

/// The 4 KiB pages of [base, base+size) the CPU may have written since they
/// were last taken (section-backed memory: `mem::take_written`), into `out`.
/// Reported pages are marked clean and armed (read-only), so their next
/// write faults and clears the mark; volatile and host-class pages, and
/// pages a file read is writing, stay unarmed and are always reported.
pub fn take_written(base: u64, size: u64, out: &mut Vec<u64>) -> Option<()> {
    #[cfg(target_os = "linux")]
    if crate::platform::write_watch::enabled() { return crate::platform::write_watch::take(base, size, out); }
    // Match uffd: a commit/remap must not flag a page between a cached Uncommitted
    // region query and clear_attn below, which would discard the new mapping's dirtiness.
    #[cfg(target_os = "linux")]
    let _mapping = arm_guard();
    if size == 0 {
        return Some(());
    }
    let (lo, hi) = (base & !(PAGE - 1), base.saturating_add(size).next_multiple_of(PAGE));
    let first = out.len();
    let epoch = super::precache::epoch() as u32;
    let frame = super::FRAMES.load(Ordering::Relaxed) as u32 + 1;
    let full = take_full();
    // The run of pages to arm, [a, e), armed with one call per region.
    let mut run: Option<(u64, u64)> = None;
    let mut armed = 0;
    let flush = |run: &mut Option<(u64, u64)>, armed: &mut u64| {
        if let Some((a, e)) = run.take() {
            *armed += arm_with(a, e - a, 0);
        }
    };
    // Uncommitted regions are skipped whole (write watching never reported
    // them; partially resident textures span ~1 TiB of them): no pages to
    // report, no table chunks made, their attention cleared until a commit.
    let (mut region_end, mut region) = (lo, RegionKind::Other);
    let (mut walked, mut queries, mut unarmed) = (0u64, 0u64, 0u64);
    // Hosted: each chunk's pages the host answered are flagged first.
    let host = via_host();
    let mut seen = u64::MAX;
    let mut p = lo;
    while p < hi {
        let chunk_end = (((p >> CHUNK_SHIFT) + 1) << CHUNK_SHIFT).min(hi);
        let c = match chunk(p, false) {
            Some(c) => c,
            None => {
                if p >= region_end {
                    (region_end, region) = region_kind(p, hi);
                    queries += 1;
                }
                if region == RegionKind::Uncommitted {
                    flush(&mut run, &mut armed);
                    p = region_end;
                    continue;
                }
                chunk(p, true)?
            }
        };
        if host && p >> CHUNK_SHIFT != seen {
            seen = p >> CHUNK_SHIFT;
            reconcile(c, seen << CHUNK_SHIFT);
        }
        // Only pages flagged for attention can be other than clean.
        let Some(q) = (if full { Some(p) } else { next_flagged(c, p, chunk_end) }) else {
            flush(&mut run, &mut armed);
            p = chunk_end;
            continue;
        };
        if q != p {
            flush(&mut run, &mut armed);
            p = q;
        }
        if p >= region_end {
            (region_end, region) = region_kind(p, hi);
            queries += 1;
        }
        if region == RegionKind::Uncommitted {
            flush(&mut run, &mut armed);
            let e = region_end.min(chunk_end);
            clear_attn(c, p, e);
            p = e;
            continue;
        }
        let i = idx(p);
        walked += 1;
        let mut s = c.pages[i].load(Ordering::Acquire);
        if s & (ST_VOLATILE | ST_HOST | ST_BUSY) == ST_VOLATILE && frame.wrapping_sub(c.vol_until[i].load(Ordering::Acquire)) as i32 >= 0 {
            // Its stay unarmed is over: tracked again (reported and armed
            // below; written again soon, it turns volatile for longer).
            if c.pages[i].compare_exchange(s, s & !ST_VOLATILE, Ordering::AcqRel, Ordering::Acquire).is_ok() {
                s &= !ST_VOLATILE;
                STATS[S_VOL_REARMED].fetch_add(1, Ordering::Relaxed);
            }
        }
        let unarmable = s & (ST_VOLATILE | ST_HOST) != 0 || c.writing[i].load(Ordering::Acquire) != 0;
        if unarmable {
            // Not tracked: written as far as anyone knows, reported once per
            // synchronization epoch (the title orders its writes for our
            // commands before an epoch ends; write watching reported a page
            // once per write). Its attention stays.
            if c.stamp[i].swap(epoch, Ordering::AcqRel) != epoch {
                out.push(p);
                unarmed += 1;
            }
        } else {
            // Attention cleared before the page is made clean: a fault
            // after this flags it again.
            c.attn[i >> 6].fetch_and(!(1u64 << (i & 63)), Ordering::AcqRel);
            let s = c.pages[i].load(Ordering::Acquire);
            if !(s & ST_CLEAN != 0 && s & (ST_ARMED | ST_GUEST_RO) != 0) {
                out.push(p);
                // Clean before armed: a write in between needs no fault, as
                // the caller reads the memory after this returns.
                c.pages[i].fetch_or(ST_CLEAN, Ordering::AcqRel);
            }
        }
        // Runs of pages to arm (clean, neither armed nor known unwritable;
        // pages left unarmed are flagged again by `arm_with`).
        let want = !unarmable && c.pages[i].load(Ordering::Acquire) & (ST_ARMED | ST_CLEAN | ST_GUEST_RO) == ST_CLEAN;
        if want {
            run = Some(match run {
                Some((a, e)) if e == p => (a, p + PAGE),
                mut other => {
                    flush(&mut other, &mut armed);
                    (p, p + PAGE)
                }
            });
        } else {
            flush(&mut run, &mut armed);
        }
        p += PAGE;
    }
    flush(&mut run, &mut armed);
    TAKE_STATS[0].fetch_add(1, Ordering::Relaxed);
    TAKE_STATS[1].fetch_add((out.len() - first) as u64, Ordering::Relaxed);
    TAKE_STATS[2].fetch_add(armed, Ordering::Relaxed);
    TAKE_STATS[3].fetch_add(walked, Ordering::Relaxed);
    TAKE_STATS[4].fetch_add(queries, Ordering::Relaxed);
    TAKE_STATS[5].fetch_add(unarmed, Ordering::Relaxed);
    Some(())
}

/// GPU work writes `writes` of guest memory where it lives (in place
/// through an import, a shadow's merge, a fill, a CP write): the mirror's
/// copies of those pages are stale (with section-backed memory such pages
/// may be in the mirror; the writes bypass the guest view's protection).
pub fn gpu_wrote(writes: &super::ranges::Ranges) {
    if !enabled() || writes.is_empty() {
        return;
    }
    let mut stale = 0;
    for (a, n) in writes.iter() {
        let (lo, hi) = ((a & !(PAGE - 1)).max(MIRROR_BASE), a.saturating_add(n).next_multiple_of(PAGE).min(MIRROR_END));
        let mut p = lo;
        while p < hi {
            match chunk(p, false) {
                Some(c) => {
                    if c.pages[idx(p)].fetch_update(Ordering::AcqRel, Ordering::Acquire, |s| Some((s & !ST_VALID) | ST_GPU_W)).is_ok_and(|s| s & ST_VALID != 0) {
                        stale += 1;
                    }
                    p += PAGE;
                }
                None => p = ((p >> CHUNK_SHIFT) + 1) << CHUNK_SHIFT,
            }
        }
    }
    STATS[S_GPU_INVAL].fetch_add(stale, Ordering::Relaxed);
}

/// What a read-only binding may do with a page (`page`).
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum Page {
    /// The mirror holds it (uploaded by a submitted batch).
    Ready,
    /// A batch not submitted yet uploads it (usable by that batch only).
    Pending,
    /// It needs an upload (`claim_pages`).
    Upload,
    /// Host class, volatile or busy: not from the mirror.
    Refuse,
    /// GPU work writes it in guest memory (`ST_GPU_W`): not from the mirror.
    GpuWritten,
}

/// `DS_RES_GPU_PAGES=1`: pages GPU work writes in guest memory stay in the
/// mirror (uploaded again after each such write; A/B).
fn gpu_pages_kept() -> bool {
    static K: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *K.get_or_init(|| std::env::var("DS_RES_GPU_PAGES").ok().as_deref() == Some("1"))
}

/// The page holding `addr` for a read-only binding.
pub fn page(addr: u64) -> Page {
    let Some(c) = chunk(addr, false) else { return Page::Upload };
    let i = idx(addr);
    let s = c.pages[i].load(Ordering::Acquire);
    if s & (ST_HOST | ST_VOLATILE | ST_BUSY) != 0 || c.writing[i].load(Ordering::Acquire) != 0 {
        return Page::Refuse;
    }
    if s & ST_GPU_W != 0 && !gpu_pages_kept() {
        return Page::GpuWritten;
    }
    let seq = c.seqs[i].load(Ordering::Acquire);
    if seq == PENDING {
        Page::Pending
    } else if s & (ST_VALID | ST_ARMED) == ST_VALID | ST_ARMED && seq != 0 {
        Page::Ready
    } else {
        Page::Upload
    }
}

/// Claims the uploads of `pages` (ascending, each `Page::Upload` when
/// looked at): their uploads become pending and they are armed (read-only,
/// device-valid), so the caller may copy them out. Returns how many were
/// claimed, in order: the caller uploads those and gives up on the rest.
pub fn claim_pages(pages: &[u64]) -> usize {
    // Take the uploads (stop at the first another thread holds).
    let mut taken = 0;
    for &p in pages {
        let c = chunk(p, true).expect("chunk");
        let i = idx(p);
        let cur = c.seqs[i].load(Ordering::Acquire);
        if cur == PENDING || c.seqs[i].compare_exchange(cur, PENDING, Ordering::AcqRel, Ordering::Acquire).is_err() {
            break;
        }
        taken += 1;
    }
    // Arm them, contiguous runs with one call each (pages still armed are
    // unchanged since: valid again).
    let mut k = 0;
    while k < taken {
        let mut e = k + 1;
        while e < taken && pages[e] == pages[e - 1] + PAGE {
            e += 1;
        }
        arm(pages[k], pages[e - 1] + PAGE - pages[k]);
        k = e;
    }
    // The claim holds up to the first page that could not be armed; the
    // pages after it are given up (never back to an older upload: device
    // memory may hold a copy older than what the host has now).
    let mut n = 0;
    for (j, &p) in pages[..taken].iter().enumerate() {
        let c = chunk(p, false).expect("chunk");
        let i = idx(p);
        let s = c.pages[i].load(Ordering::Acquire);
        let ok = j == n
            && s & ST_ARMED != 0
            && s & (ST_HOST | ST_VOLATILE) == 0
            && c.writing[i].load(Ordering::Acquire) == 0;
        if ok {
            c.pages[i].fetch_or(ST_VALID, Ordering::AcqRel);
            n += 1;
        } else {
            c.pages[i].fetch_and(!ST_VALID, Ordering::AcqRel);
            uncur(c, i);
            c.seqs[i].store(0, Ordering::Release);
        }
    }
    STATS[S_CLAIMED].fetch_add(n as u64, Ordering::Relaxed);
    n
}

/// A write that cannot fault (a file read into guest memory) begins over
/// [base, base+len): its pages in the mirror's span are disarmed and stay
/// out of the mirror until `host_write_end`.
pub fn host_write_begin(base: u64, len: u64) {
    #[cfg(target_os = "linux")]
    if crate::platform::write_watch::enabled() { crate::platform::write_watch::host_write(base, len, true); return; }
    if !tracking() || len == 0 {
        return;
    }
    // Residency alone arms only the mirror's span; section-backed memory's
    // write tracking arms all guest memory, the mixed range's its views.
    for (lo, hi) in tracked_parts(base & !(PAGE - 1), base.saturating_add(len).next_multiple_of(PAGE)) {
        host_write_begin_in(lo, hi);
    }
}

fn host_write_begin_in(lo: u64, hi: u64) {
    if lo >= hi {
        return;
    }
    // Mirror mode: GPU-written dwords first (a file read may cover pages in
    // part).
    release_range(lo, hi - lo);
    let mut p = lo;
    while p < hi {
        if let Some(c) = chunk(p, true) {
            c.writing[idx(p)].fetch_add(1, Ordering::AcqRel);
            c.pages[idx(p)].fetch_and(!ST_CLEAN, Ordering::AcqRel);
            flag(c, idx(p));
        }
        p += PAGE;
    }
    disarm_range(lo, hi);
    STATS[S_HOST_WRITES].fetch_add(1, Ordering::Relaxed);
    STATS[S_HOST_WRITE_PAGES].fetch_add((hi - lo) / PAGE, Ordering::Relaxed);
}

/// A kernel-mode write into [base, base+len) failed (`mem::kernel_io`): its
/// armed pages are made writable again before it is retried.
pub fn disarm_for_host(base: u64, len: u64) {
    if !tracking() || len == 0 {
        return;
    }
    for (lo, hi) in tracked_parts(base & !(PAGE - 1), base.saturating_add(len).next_multiple_of(PAGE)) {
        disarm_range(lo, hi);
    }
}

/// The tracking states of [base, base+len)'s pages (diagnostics).
pub fn describe(base: u64, len: u64) -> String {
    if !tracking() {
        return "no write tracking".into();
    }
    let (lo, hi) = (base & !(PAGE - 1), base.saturating_add(len).next_multiple_of(PAGE));
    // armed, clean, guest read-only, busy, being written, volatile, host class
    let mut n = [0u64; 7];
    let bits = [ST_ARMED, ST_CLEAN, ST_GUEST_RO, ST_BUSY, 0, ST_VOLATILE, ST_HOST];
    let mut p = lo;
    while p < hi {
        if let Some(c) = chunk(p, false) {
            let i = idx(p);
            let s = c.pages[i].load(Ordering::Acquire);
            for (k, &b) in bits.iter().enumerate() {
                if b != 0 && s & b != 0 {
                    n[k] += 1;
                }
            }
            if c.writing[i].load(Ordering::Acquire) != 0 {
                n[4] += 1;
            }
        }
        p += PAGE;
    }
    format!(
        "{} pages: armed {}, clean {}, guest read-only {}, busy {}, being written {}, volatile {}, host class {}",
        (hi - lo) / PAGE,
        n[0],
        n[1],
        n[2],
        n[3],
        n[4],
        n[5],
        n[6]
    )
}

/// The write announced by `host_write_begin(base, len)` ended: its pages
/// may be armed again (the mirror's copies of them are stale).
pub fn host_write_end(base: u64, len: u64) {
    #[cfg(target_os = "linux")]
    if crate::platform::write_watch::enabled() { crate::platform::write_watch::host_write(base, len, false); return; }
    if !tracking() || len == 0 {
        return;
    }
    for (lo, hi) in tracked_parts(base & !(PAGE - 1), base.saturating_add(len).next_multiple_of(PAGE)) {
        host_write_end_in(lo, hi);
    }
}

fn host_write_end_in(lo: u64, hi: u64) {
    let mut p = lo;
    while p < hi {
        if let Some(c) = chunk(p, false) {
            let i = idx(p);
            c.pages[i].fetch_and(!ST_VALID, Ordering::AcqRel);
            let _ = c.writing[i].fetch_update(Ordering::AcqRel, Ordering::Acquire, |w| w.checked_sub(1));
        }
        p += PAGE;
    }
}

/// The batch that uploads `pages` was submitted as `seq`.
pub fn submitted(pages: &[u64], seq: u64) {
    for &p in pages {
        if let Some(c) = chunk(p, false) {
            let _ = c.seqs[idx(p)].compare_exchange(PENDING, seq, Ordering::AcqRel, Ordering::Acquire);
        }
    }
}

/// The batch that would have uploaded `pages` was dropped: they need an
/// upload again.
pub fn dropped(pages: &[u64]) {
    for &p in pages {
        if let Some(c) = chunk(p, false) {
            let i = idx(p);
            if c.seqs[i].compare_exchange(PENDING, 0, Ordering::AcqRel, Ordering::Acquire).is_ok() {
                c.pages[i].fetch_and(!ST_VALID, Ordering::AcqRel);
                uncur(c, i);
            }
        }
    }
}

/// The mirror buffer and its sparse page size.
pub fn mirror() -> Option<(vk::Buffer, u64)> {
    if !enabled() {
        return None;
    }
    MIRROR.get().map(|m| (m.buf, m.page))
}

/// The mirror buffer and its sparse page size once made, residency on or
/// not (front end 2: `fe2`).
pub fn mirror_buffer() -> Option<(vk::Buffer, u64)> {
    MIRROR.get().map(|m| (m.buf, m.page))
}

/// The mirror's device address (guest MIRROR_BASE's), 0 before it is made
/// (front end 2's GPU-resolved chains, `rdna::translate::Spec::pt_linear`).
pub fn mirror_address() -> u64 {
    static VA: std::sync::OnceLock<u64> = std::sync::OnceLock::new();
    match MIRROR.get() {
        Some(m) => *VA.get_or_init(|| ctx().buffer_address(m.buf)),
        None => 0,
    }
}

const _: () = assert!(MIRROR_BASE == rdna::translate::PT_BASE && MIRROR_END == rdna::translate::PT_END);

/// Front end 2 (`fe2`): the mirror and the written-bit map, made without
/// turning residency on (none of its placements run). Ok: what was made.
pub fn make_for_fe2() -> Result<String, String> {
    if MIRROR.get().is_none() {
        let m = make_mirror()?;
        let _ = MIRROR.set(m);
    }
    let m = MIRROR.get().ok_or("no mirror")?;
    let at = MIRROR_END - m.page;
    let data: Vec<u8> = (0..m.page).map(|i| (i * 7 + 3) as u8).collect();
    upload_now(at, &data)?;
    if readback_now(at, m.page)? != data {
        return Err("a page through the mirror and back differs".into());
    }
    if MARKS.get().is_none() {
        make_marks()?;
    }
    Ok(format!("mirror of {} GiB (sparse pages of {} KiB), written-bit map of {} MiB", (MIRROR_END - MIRROR_BASE) >> 30, m.page >> 10, MARK_BYTES >> 20))
}

/// Shared while pages are armed: mapping changes exclude it (`remap_guard`).
pub fn arm_guard() -> parking_lot::RwLockReadGuard<'static, ()> {
    ARM_LOCK.read_recursive()
}

/// `arm_guard` within `d`, else None.
pub fn try_arm_guard_for(d: std::time::Duration) -> Option<parking_lot::RwLockReadGuard<'static, ()>> {
    ARM_LOCK.try_read_recursive_for(d)
}

/// The mirror: a sparse buffer over [MIRROR_BASE, MIRROR_END).
struct Mirror {
    buf: vk::Buffer,
    /// Sparse block size (the alignment of its memory requirements).
    page: u64,
    mem_type: u32,
    /// Device memory blocks, the pages claimed and the binds waiting
    /// (`back_in`).
    blocks: Mutex<Backing>,
    /// Woken when a bind completes or fails.
    bound: parking_lot::Condvar,
    /// One bit per mirror page: bound to memory.
    backed: Box<[AtomicU64]>,
    /// Per word of `backed` (64 pages): the last flip a placement covered
    /// any of its pages (`back_in`; `cold_mib`).
    used: Box<[std::sync::atomic::AtomicU32]>,
}

/// A mirror's backing state (`Mirror::blocks`): one thread at a time binds
/// every page claimed so far in one `vkQueueBindSparse` and waits for it;
/// threads needing pages meanwhile claim theirs and wait for that bind or
/// make the next one (a bind and its wait cost ~0.4-0.9 ms whatever its
/// size, FINDINGS.md 2026-11-01).
struct Backing {
    /// Device memory blocks: the mirror's pages come from `cur` (its next
    /// free offset `next`), the maps' bit pages from blocks of their own
    /// (`mark_cur`: block and next offset), so that a block of mirror pages
    /// can be freed whole (`unback_cold`; the maps' pages stay bound).
    mems: Vec<vk::DeviceMemory>,
    cur: Option<usize>,
    next: u64,
    mark_cur: Option<(usize, u64)>,
    /// One bit per page: claimed for a bind (waiting, in flight or done).
    claimed: Vec<u64>,
    /// Binds of claimed pages no bind has taken yet.
    pending: Vec<vk::SparseMemoryBind>,
    /// A thread is binding.
    binding: bool,
    /// The sparse written-bit maps (`SparseMarks`): one bit per bit page
    /// claimed (in every map); the binds no bind has taken yet (map, bind)
    /// and their bit pages, zeroed once bound.
    mark_claimed: Vec<u64>,
    mark_pending: Vec<(usize, vk::SparseMemoryBind)>,
    mark_new: Vec<u64>,
    /// Per block of `mems`: its pages bound now; a block of mirror pages
    /// left with none is freed (`unback_cold`).
    block_used: Vec<u32>,
    /// Pages of blocks the mirror unbound, taken first (`take_page`).
    free: Vec<(usize, u64)>,
    /// Each mirror page bound or being bound: its block and offset.
    page_mem: std::collections::HashMap<u64, (usize, u64)>,
    /// Cold words whose backed bits a sweep cleared, still bound
    /// (`unback_cold`): their bits, their stamp then, the flip.
    draining: std::collections::HashMap<usize, (u64, u32, u32)>,
    /// A block allocated outside this lock (`refill_spare`), which `take_page` takes before
    /// allocating under it.
    spare: Option<vk::DeviceMemory>,
}

/// The spare block's state (`Backing::spare`): none there, one thread making one, and the
/// blocks taken from it and made for it (status).
static SPARE_EMPTY: std::sync::atomic::AtomicBool = std::sync::atomic::AtomicBool::new(true);
static SPARE_FILLING: std::sync::atomic::AtomicBool = std::sync::atomic::AtomicBool::new(false);
/// The refill is inside its vkAllocateMemory (not one the refusal window turns down at once):
/// `take_page`'s busy answer and `back_in`'s wait. With `SPARE_FILLING` there, every refill
/// attempt made pages busy, most of them refused in microseconds, and under pressure threads kept
/// waiting for refills that never reached the driver (the short injected run of 2026-11-22 small
/// hours: its new game's frozen time 83 against 29 s).
static SPARE_ALLOCATING: std::sync::atomic::AtomicBool = std::sync::atomic::AtomicBool::new(false);
/// [taken, made, pages gone without after waiting for one in the making].
pub static SPARE_STATS: [AtomicU64; 3] = [const { AtomicU64::new(0) }; 3];

/// Video memory the mirror's blocks hold now (its pages, the written-bit maps' pages and the
/// spare block), bytes (`images::over_budget_line`).
pub fn mirror_bytes() -> Option<u64> {
    let Some(m) = MIRROR.get() else { return Some(0) };
    // (Not waited for: the reaper asks, which must not stand behind a page's allocation.)
    let g = m.blocks.try_lock()?;
    Some((g.mems.iter().filter(|&&x| x != vk::DeviceMemory::null()).count() as u64 + g.spare.is_some() as u64) * BLOCK)
}

/// `take_page`'s answer while another thread makes the spare block under pressure: `back_in` then
/// waits for it with the backing's lock released (`SPARE_WAIT`) and claims on.
const SPARE_BUSY: &str = "vkAllocateMemory (mirror block): the spare block is still being made";
/// The longest `back_in` waits for a spare block in the making before its pages go without.
/// (Until 2026-11-22 small hours a page went without at once: the PGO training's walk on this PC,
/// its card shared and near its budget, drew mirror pages from zeros whenever a refill was in
/// flight; the review of 2026-11-21 night had named it as a risk.)
const SPARE_WAIT: std::time::Duration = std::time::Duration::from_secs(2);

/// The spare block's status line (`SPARE_STATS`).
pub fn spare_line() -> String {
    let (taken, made, refused) = (SPARE_STATS[0].load(Ordering::Relaxed), SPARE_STATS[1].load(Ordering::Relaxed), SPARE_STATS[2].load(Ordering::Relaxed));
    if made == 0 && refused == 0 {
        return String::new();
    }
    format!("    the mirror's spare blocks: {made} made outside the backing's lock, {taken} taken under it with no driver call, {refused} pages gone without while one was still being made\n")
}

/// Makes the spare block when there is none, outside the backing's lock, one thread at a time
/// (`back`): `take_page` then needs no driver call under that lock. A thread there had blocked in
/// vkAllocateMemory with the lock held while video memory ran out, every recording thread and the
/// graphics CP waiting behind it until the hang watch ended the session (the other PC's 12 GB
/// card, 2026-11-21). An allocation that fails leaves `take_page` to try under the lock as before.
fn refill_spare(m: &Mirror) {
    use std::sync::atomic::Ordering::{AcqRel, Acquire, Release};
    if !SPARE_EMPTY.load(Acquire) || SPARE_FILLING.swap(true, AcqRel) {
        return;
    }
    let c = ctx();
    // (Not inside the refusal window after a failure, `Ctx::refuse_alloc`: every placement over
    // pages not yet backed comes here, and while memory was out each made a failing driver call.)
    if !c.refuse_alloc(m.mem_type, BLOCK) {
        let mut fl = vk::MemoryAllocateFlagsInfo::default().flags(vk::MemoryAllocateFlags::DEVICE_ADDRESS);
        let ai = vk::MemoryAllocateInfo::default().allocation_size(BLOCK).memory_type_index(m.mem_type);
        let ai = if c.device_addresses { ai.push_next(&mut fl) } else { ai };
        let ta = std::time::Instant::now();
        SPARE_ALLOCATING.store(true, Release);
        let r = unsafe { c.device.allocate_memory(&ai, None) };
        SPARE_ALLOCATING.store(false, Release);
        if let Ok(mem) = r {
            super::vk::note_alloc("a spare mirror block", BLOCK, ta);
            let mut g = m.blocks.lock();
            if g.spare.is_none() {
                g.spare = Some(mem);
                SPARE_EMPTY.store(false, Release);
                SPARE_STATS[1].fetch_add(1, Ordering::Relaxed);
            } else {
                drop(g);
                unsafe { c.device.free_memory(mem, None) };
            }
        } else {
            c.note_alloc_fail(m.mem_type, BLOCK);
        }
    }
    SPARE_FILLING.store(false, Release);
}

/// The written-bit maps as sparse buffers (`make_marks`): the bit words of
/// mirror pages are bound when those pages are (`back_in`, in the same
/// bind) and zeroed before the pages count as backed; the page bits and
/// touched-page lists when the maps are made. Maps over the mirror's whole
/// span took 1 GiB of video memory each, where the 4.8 GB the swamp's
/// mirror held needs ~150-300 MB (FINDINGS.md 2026-11-02).
struct SparseMarks {
    /// Every map (`MARKS`, then `MARKS_C`), their sparse page size.
    bufs: Vec<vk::Buffer>,
    page: u64,
}

static SPARSE_MARKS: std::sync::OnceLock<SparseMarks> = std::sync::OnceLock::new();
/// The maps' bit and tail pages bound (all maps together).
static MARK_PAGES: AtomicU64 = AtomicU64::new(0);

/// A page of device memory for a sparse bind: a mirror page (freed pages
/// first) or, with `marks`, a bit page of the written-bit maps.
fn take_page(g: &mut Backing, m: &Mirror, alloc_ns: &mut u64, marks: bool) -> Result<(vk::DeviceMemory, u64, usize), String> {
    if !marks {
        if let Some((b, off)) = g.free.pop() {
            g.block_used[b] += 1;
            return Ok((g.mems[b], off, b));
        }
    }
    let open = if marks { g.mark_cur } else { g.cur.map(|b| (b, g.next)) };
    let (b, off) = match open {
        Some((b, off)) if off + m.page <= BLOCK => (b, off),
        _ => {
            // (The spare block first, made outside this lock: `refill_spare`.)
            let mem = if let Some(mem) = g.spare.take() {
                SPARE_EMPTY.store(true, Ordering::Release);
                SPARE_STATS[0].fetch_add(1, Ordering::Relaxed);
                mem
            } else if SPARE_ALLOCATING.load(Ordering::Acquire) && super::images::under_pressure() {
                // (Under video-memory pressure another thread is making the spare and has not
                // finished in `back`'s wait: the driver is slow to give memory, and an allocation
                // here would hold the lock while it waits, every recording thread behind it. The
                // page goes without, as when memory runs out; the review of 2026-11-21 night.
                // Without pressure a refill takes milliseconds, and the allocation below is safe.)
                return Err(SPARE_BUSY.into());
            } else {
                let c = ctx();
                let mut fl = vk::MemoryAllocateFlagsInfo::default().flags(vk::MemoryAllocateFlags::DEVICE_ADDRESS);
                let ai = vk::MemoryAllocateInfo::default().allocation_size(BLOCK).memory_type_index(m.mem_type);
                let ai = if c.device_addresses { ai.push_next(&mut fl) } else { ai };
                let ta = std::time::Instant::now();
                // (The refusal window after a failure and `DS_SIM_VRAM_MIB`: `Ctx::refuse_alloc`.)
                if c.refuse_alloc(m.mem_type, BLOCK) {
                    return Err(format!("vkAllocateMemory (mirror block): {} (refused: video memory ran out moments ago, or DS_SIM_VRAM_MIB)", super::vk::SIM_OOM));
                }
                let mem = unsafe { c.device.allocate_memory(&ai, None) }.map_err(|e| {
                    c.note_alloc_fail(m.mem_type, BLOCK);
                    format!("vkAllocateMemory (mirror block): {e}")
                })?;
                super::vk::note_alloc("a mirror block", BLOCK, ta);
                *alloc_ns += ta.elapsed().as_nanos() as u64;
                mem
            };
            g.mems.push(mem);
            g.block_used.push(0);
            (g.mems.len() - 1, 0)
        }
    };
    if marks {
        g.mark_cur = Some((b, off + m.page));
    } else {
        g.cur = Some(b);
        g.next = off + m.page;
    }
    g.block_used[b] += 1;
    Ok((g.mems[b], off, b))
}

/// Claims the written-bit maps' bit page covering mirror page `p` (in
/// every map) for the next bind, when it has none yet.
fn claim_mark_page(g: &mut Backing, m: &Mirror, p: u64, alloc_ns: &mut u64) -> Result<(), String> {
    let Some(sm) = SPARSE_MARKS.get() else { return Ok(()) };
    // (A dword's bit: a page's bits are page / 32 bytes at page_index * page / 32.)
    let k = p * m.page / 32 / sm.page;
    let (w, bit) = ((k / 64) as usize, 1u64 << (k % 64));
    if g.mark_claimed[w] & bit != 0 {
        return Ok(());
    }
    for i in 0..sm.bufs.len() {
        let (mem, off, _) = take_page(g, m, alloc_ns, true)?;
        g.mark_pending.push((i, vk::SparseMemoryBind::default().resource_offset(k * sm.page).size(sm.page).memory(mem).memory_offset(off)));
    }
    g.mark_claimed[w] |= bit;
    g.mark_new.push(k);
    Ok(())
}

/// Zeroes the maps' bit pages `pages` (just bound), on the copy queue when
/// there is one (the main queue would wait for the batches queued there),
/// and waits.
fn zero_mark_pages(sm: &SparseMarks, pages: &[u64]) -> Result<(), String> {
    one_off_side(|cmd: vk::CommandBuffer| unsafe {
        for &b in &sm.bufs {
            for &k in pages {
                ctx().device.cmd_fill_buffer(cmd, b, k * sm.page, sm.page.min(MARK_BYTES - k * sm.page), 0);
            }
        }
    })
}

/// A one-off command buffer on the copy queue when there is one (the main
/// queue would first finish the batches queued there), else on the main
/// queue; waited for. Nothing orders it with the other queues' work but
/// the wait: for writes those queues' later batches may see late.
pub fn one_off_side(record: impl FnOnce(vk::CommandBuffer)) -> Result<(), String> {
    let c = ctx();
    let Some(q) = c.copy.as_ref() else { return one_off(record) };
    unsafe {
        let pool = c.device.create_command_pool(&vk::CommandPoolCreateInfo::default().queue_family_index(q.family), None).map_err(|e| format!("pool: {e}"))?;
        let r = (|| {
            let cmd = c.device.allocate_command_buffers(&vk::CommandBufferAllocateInfo::default().command_pool(pool).command_buffer_count(1)).map_err(|e| format!("cmd: {e}"))?[0];
            c.device.begin_command_buffer(cmd, &vk::CommandBufferBeginInfo::default().flags(vk::CommandBufferUsageFlags::ONE_TIME_SUBMIT)).map_err(|e| format!("begin: {e}"))?;
            record(cmd);
            c.device.end_command_buffer(cmd).map_err(|e| format!("end: {e}"))?;
            let fence = c.device.create_fence(&vk::FenceCreateInfo::default(), None).map_err(|e| format!("fence: {e}"))?;
            let r = c.submit_timeline(true, &[cmd], &[], &[], fence).and_then(|_| c.device.wait_for_fences(&[fence], true, u64::MAX).map_err(|e| format!("wait: {e}")));
            c.device.destroy_fence(fence, None);
            r
        })();
        c.device.destroy_command_pool(pool, None);
        r
    }
}

static MIRROR: std::sync::OnceLock<Mirror> = std::sync::OnceLock::new();

fn make_mirror() -> Result<Mirror, String> {
    let c = ctx();
    if !c.sparse_buffers {
        return Err("sparse buffers not enabled".into());
    }
    let size = MIRROR_END - MIRROR_BASE;
    let usage = vk::BufferUsageFlags::STORAGE_BUFFER
        | vk::BufferUsageFlags::TRANSFER_SRC
        | vk::BufferUsageFlags::TRANSFER_DST
        | vk::BufferUsageFlags::INDEX_BUFFER
        | vk::BufferUsageFlags::INDIRECT_BUFFER;
    let mut bi = vk::BufferCreateInfo::default().size(size).usage(c.addressable(usage)).flags(vk::BufferCreateFlags::SPARSE_BINDING | vk::BufferCreateFlags::SPARSE_RESIDENCY);
    if c.families.len() > 1 {
        bi = bi.sharing_mode(vk::SharingMode::CONCURRENT).queue_family_indices(&c.families);
    }
    let buf = unsafe { c.device.create_buffer(&bi, None) }.map_err(|e| format!("vkCreateBuffer (sparse, {} GiB): {e}", size >> 30))?;
    let req = unsafe { c.device.get_buffer_memory_requirements(buf) };
    let Some(mem_type) = c.memory_type(req.memory_type_bits, vk::MemoryPropertyFlags::DEVICE_LOCAL) else {
        unsafe { c.device.destroy_buffer(buf, None) };
        return Err("no device-local memory type for the mirror".into());
    };
    let pages = (size / req.alignment) as usize;
    // (Its device address too, when buffers have them: device faults name device addresses,
    // `vk::describe_address`.)
    let va = if c.device_addresses { c.buffer_address(buf) } else { 0 };
    crate::info!(
        "resident: mirror of guest {MIRROR_BASE:#x}..{MIRROR_END:#x} ({} GiB, sparse pages of {} KiB, memory type {mem_type} of type bits {:#x}; device sparse address space {} GiB; device address {va:#x})",
        size >> 30,
        req.alignment >> 10,
        req.memory_type_bits,
        c.limits.sparse_address_space_size >> 30
    );
    let words = pages.div_ceil(64);
    let blocks = Mutex::new(Backing { mems: Vec::new(), cur: None, next: 0, mark_cur: None, claimed: vec![0; words], pending: Vec::new(), binding: false, mark_claimed: Vec::new(), mark_pending: Vec::new(), mark_new: Vec::new(), block_used: Vec::new(), free: Vec::new(), page_mem: std::collections::HashMap::new(), draining: std::collections::HashMap::new(), spare: None });
    Ok(Mirror { buf, page: req.alignment, mem_type, blocks, bound: parking_lot::Condvar::new(), backed: (0..words).map(|_| AtomicU64::new(0)).collect(), used: (0..words).map(|_| std::sync::atomic::AtomicU32::new(0)).collect() })
}

/// Binds device memory to the mirror's pages over guest [base, base+len)
/// that have none yet (synchronously: the bind completes before return).
pub fn back(base: u64, len: u64) -> Result<(), String> {
    let m = MIRROR.get().ok_or("no mirror")?;
    refill_spare(m);
    // (Another thread making the spare: waited for here, outside the lock, at most 200 ms; a
    // driver slower than that leaves `take_page` to go without a new block.)
    if SPARE_EMPTY.load(Ordering::Acquire) && SPARE_FILLING.load(Ordering::Acquire) {
        let t = std::time::Instant::now();
        while SPARE_FILLING.load(Ordering::Acquire) && t.elapsed() < std::time::Duration::from_millis(200) {
            std::thread::sleep(std::time::Duration::from_millis(1));
        }
    }
    back_in(m, base, len, &STATS[S_BACKED])
}

/// Whether the mirror's pages of [base, base + len) are all backed, their words' last placement
/// stamped as `back` stamps it (front end 2's lean placement, `Exec::fe2_place_quick`). False:
/// `back` binds what is missing.
#[inline]
pub fn backed_now(base: u64, len: u64) -> bool {
    let Some(m) = MIRROR.get() else { return false };
    if len == 0 || base < MIRROR_BASE || base + len > MIRROR_END {
        return false;
    }
    let first = (base - MIRROR_BASE) / m.page;
    let last = (base + len - 1 - MIRROR_BASE) / m.page;
    let flip = super::FRAMES.load(Ordering::Relaxed) as u32;
    for w in (first / 64)..=(last / 64) {
        let u = &m.used[w as usize];
        if u.load(Ordering::Relaxed) != flip {
            u.store(flip, Ordering::Relaxed);
        }
    }
    (first..=last).all(|p| m.backed[(p / 64) as usize].load(Ordering::Acquire) & (1u64 << (p % 64)) != 0)
}

/// Backing ahead: a recording thread's backing that bound granules right
/// after a backed one (a region growing in order: ~40% of the walk's backing
/// time) has the next `BACK_AHEAD` granules bound by a helper thread, so the
/// next first use finds them bound (2026-11-12: the walk x1.015, ~100 MiB
/// more of the mirror backed by its end).
const BACK_AHEAD: u64 = 4;
/// Ranges a recording thread's in-order backing queued for the helper:
/// [granule ranges queued, backed by the helper, found backed already, the
/// helper's ns].
static AHEAD_Q: parking_lot::Mutex<Vec<(u64, u64)>> = parking_lot::Mutex::new(Vec::new());
static AHEAD_CV: parking_lot::Condvar = parking_lot::Condvar::new();
static AHEAD: [AtomicU64; 4] = [const { AtomicU64::new(0) }; 4];
thread_local! {
    /// This thread is the helper (its backings queue nothing ahead).
    static IN_AHEAD: std::cell::Cell<bool> = const { std::cell::Cell::new(false) };
}

/// Queues [base, base + len) for the helper (`back_ahead_thread`).
fn back_ahead_queue(base: u64, len: u64) {
    static START: std::sync::Once = std::sync::Once::new();
    START.call_once(|| {
        let _ = std::thread::Builder::new().name("gpu-backahead".into()).spawn(back_ahead_thread);
    });
    let mut q = AHEAD_Q.lock();
    if q.len() < 256 && !q.iter().any(|e| e.0 == base) {
        q.push((base, len));
        AHEAD[0].fetch_add(1, Ordering::Relaxed);
    }
    drop(q);
    AHEAD_CV.notify_one();
}

/// The helper: backs what in-order growth queued (`back_in`, the same
/// synchronous bind), skipping ranges bound meanwhile.
fn back_ahead_thread() {
    IN_AHEAD.with(|f| f.set(true));
    let mut work: Vec<(u64, u64)> = Vec::new();
    loop {
        {
            let mut q = AHEAD_Q.lock();
            while q.is_empty() {
                AHEAD_CV.wait_for(&mut q, std::time::Duration::from_millis(100));
            }
            std::mem::swap(&mut work, &mut *q);
        }
        if super::shutting_down() {
            work.clear();
            continue;
        }
        let Some(m) = MIRROR.get() else {
            work.clear();
            continue;
        };
        for &(base, len) in &work {
            let t = std::time::Instant::now();
            let before = STATS[S_BACKED].load(Ordering::Relaxed);
            let _ = back_in(m, base, len, &STATS[S_BACKED]);
            if STATS[S_BACKED].load(Ordering::Relaxed) > before {
                AHEAD[1].fetch_add(1, Ordering::Relaxed);
            } else {
                AHEAD[2].fetch_add(1, Ordering::Relaxed);
            }
            AHEAD[3].fetch_add(t.elapsed().as_nanos() as u64, Ordering::Relaxed);
        }
        work.clear();
    }
}

/// Backing ahead's status line ("" before any).
pub fn back_ahead_line() -> String {
    let v: Vec<u64> = AHEAD.iter().map(|c| c.load(Ordering::Relaxed)).collect();
    if v[0] == 0 {
        return String::new();
    }
    format!("    mirror backings ahead of in-order growth ({} granules): {} ranges queued, {} backed by the helper, {} found backed already, {:.0} ms of the helper\n", BACK_AHEAD, v[0], v[1], v[2], v[3] as f64 / 1e6)
}

fn back_in(m: &Mirror, base: u64, len: u64, count: &AtomicU64) -> Result<(), String> {
    if base < MIRROR_BASE || base + len > MIRROR_END {
        return Err(format!("{base:#x}+{len:#x} is outside the mirror"));
    }
    let c = ctx();
    let first = (base - MIRROR_BASE) / m.page;
    let last = (base + len - 1 - MIRROR_BASE) / m.page;
    // (The flip each word was last placed in: stored only when it moves, so
    // the threads placing the same ranges mostly read the line.)
    let flip = super::FRAMES.load(Ordering::Relaxed) as u32;
    for w in (first / 64)..=(last / 64) {
        let u = &m.used[w as usize];
        if u.load(Ordering::Relaxed) != flip {
            u.store(flip, Ordering::Relaxed);
        }
    }
    // Backed already (most calls): no lock; whole words of 64 pages (4 MiB
    // with 64 KiB pages) at a time.
    let backed = |p: u64| m.backed[(p / 64) as usize].load(Ordering::Acquire) & (1u64 << (p % 64)) != 0;
    let mut p = first;
    let mut all = true;
    while p <= last {
        if p % 64 == 0 && p + 63 <= last {
            if m.backed[(p / 64) as usize].load(Ordering::Acquire) != u64::MAX {
                all = false;
                break;
            }
            p += 64;
        } else {
            if !backed(p) {
                all = false;
                break;
            }
            p += 1;
        }
    }
    if all {
        return Ok(());
    }
    let t0 = std::time::Instant::now();
    // The aligned granule around the range is claimed with it (`tune
    // fe2_back_granule` pages): the binds a range's neighbours would make.
    let gran = back_granule();
    let pages = (MIRROR_END - MIRROR_BASE) / m.page;
    let (g0, g1) = (first & !(gran - 1), (last | (gran - 1)).min(pages - 1));
    let mut g = m.blocks.lock();
    restore_draining(&mut g, m, first / 64, last / 64);
    let (mut alloc_ns, mut binds_here, mut pages_here, mut bind_ns) = (0u64, 0u64, 0u64, 0u64);
    let mut claim = |g: &mut Backing, a: u64, b: u64| -> Result<(), String> {
        for p in a..=b {
            let (w, bit) = ((p / 64) as usize, 1u64 << (p % 64));
            if g.claimed[w] & bit != 0 {
                continue;
            }
            let (mem, off, blk) = take_page(g, m, &mut alloc_ns, false)?;
            g.pending.push(vk::SparseMemoryBind::default().resource_offset(p * m.page).size(m.page).memory(mem).memory_offset(off));
            g.page_mem.insert(p, (blk, off));
            g.claimed[w] |= bit;
            // (The written-bit maps' bits of the page, in the same bind.)
            claim_mark_page(g, m, p, &mut alloc_ns)?;
        }
        Ok(())
    };
    // (A page's block while another thread makes the spare under pressure (`SPARE_BUSY`): waited
    // for with the lock released, at most `SPARE_WAIT`, then the claim goes on from where it
    // stopped (claimed pages stay claimed); no driver call is made under the lock.)
    let mut claim_on = |g: &mut parking_lot::MutexGuard<'_, Backing>, a: u64, b: u64| -> Result<(), String> {
        let t = std::time::Instant::now();
        loop {
            match claim(&mut **g, a, b) {
                Err(e) if e == SPARE_BUSY && t.elapsed() < SPARE_WAIT => {
                    parking_lot::MutexGuard::unlocked(g, || {
                        while SPARE_ALLOCATING.load(Ordering::Acquire) && t.elapsed() < SPARE_WAIT {
                            std::thread::sleep(std::time::Duration::from_millis(1));
                        }
                    });
                }
                Err(e) if e == SPARE_BUSY => {
                    SPARE_STATS[2].fetch_add(1, Ordering::Relaxed);
                    return Err(e);
                }
                r => return r,
            }
        }
    };
    claim_on(&mut g, g0, g1)?;
    loop {
        if (first..=last).all(backed) {
            break;
        }
        if !g.binding && g.pending.is_empty() {
            // (Pages a failed bind gave up: claimed again.)
            claim_on(&mut g, first, last)?;
        }
        if g.binding || g.pending.is_empty() {
            m.bound.wait(&mut g);
            continue;
        }
        let binds = std::mem::take(&mut g.pending);
        let mbinds = std::mem::take(&mut g.mark_pending);
        let mnew = std::mem::take(&mut g.mark_new);
        g.binding = true;
        // (Waited for: binds not waited for, every later submission waiting
        // for a timeline value instead, cost the arrival no less and stalled
        // the queues for seconds at times: FINDINGS.md 2026-11-01.)
        let tb = std::time::Instant::now();
        let r = parking_lot::MutexGuard::unlocked(&mut g, || {
            let sm = SPARSE_MARKS.get();
            let per_map: Vec<Vec<vk::SparseMemoryBind>> = (0..sm.map_or(0, |s| s.bufs.len())).map(|i| mbinds.iter().filter(|b| b.0 == i).map(|b| b.1).collect()).collect();
            let driver_binds = crate::platform::graphics::sparse_buffer_binds(&binds);
            let driver_maps: Vec<_> = per_map.iter().map(|v| crate::platform::graphics::sparse_buffer_binds(v)).collect();
            let mut bufs = vec![vk::SparseBufferMemoryBindInfo::default().buffer(m.buf).binds(&driver_binds)];
            if let Some(sm) = sm {
                for (i, v) in driver_maps.iter().enumerate().filter(|(_, v)| !v.is_empty()) {
                    bufs.push(vk::SparseBufferMemoryBindInfo::default().buffer(sm.bufs[i]).binds(v));
                }
            }
            let info = vk::BindSparseInfo::default().buffer_binds(&bufs);
            let fence = unsafe { c.device.create_fence(&vk::FenceCreateInfo::default(), None) }.map_err(|e| format!("vkCreateFence: {e}"))?;
            let r = {
                // (On the main family's second queue it cost the same:
                // FINDINGS.md 2026-11-01; on the copy family's second queue
                // too, the call's cost being the paging operation: TRIED.md
                // `bind_queue`, 2026-11-16.)
                let t0 = std::time::Instant::now();
                let q = c.queue.lock();
                let t1 = std::time::Instant::now();
                let r = unsafe { c.device.queue_bind_sparse(*q, &[info], fence) };
                drop(q);
                super::vk::note_submit("vkQueueBindSparse", t0, t1);
                r
            };
            let tw = std::time::Instant::now();
            let r = r.map_err(|e| format!("vkQueueBindSparse: {e}")).and_then(|_| unsafe { c.device.wait_for_fences(&[fence], true, u64::MAX) }.map_err(|e| format!("bind fence: {e}")));
            unsafe { c.device.destroy_fence(fence, None) };
            // (`DS_IMPORT_LOG=1`: each bind with its wait, beside the imports and submissions.)
            static LOG: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
            if *LOG.get_or_init(|| std::env::var("DS_IMPORT_LOG").ok().as_deref() == Some("1")) {
                crate::info!("gpu: sparse bind of {} pages: call and wait {} us (wait {} us)", binds.len(), tb.elapsed().as_micros(), tw.elapsed().as_micros());
            }
            // The maps' new bit pages zeroed before their mirror pages count
            // as backed (no shader marks them before that).
            match (r, sm) {
                (Ok(()), Some(sm)) if !mnew.is_empty() => zero_mark_pages(sm, &mnew),
                (r, _) => r,
            }
        });
        bind_ns += tb.elapsed().as_nanos() as u64;
        g.binding = false;
        for b in &binds {
            let p = b.resource_offset / m.page;
            let (w, bit) = ((p / 64) as usize, 1u64 << (p % 64));
            if r.is_ok() {
                m.backed[w].fetch_or(bit, Ordering::AcqRel);
                if let Some(t) = super::fe2::traced(MIRROR_BASE + p * m.page, MIRROR_BASE + (p + 1) * m.page) {
                    super::fe2::trace_event(t, "its mirror page bound (back_in)");
                }
            } else {
                g.claimed[w] &= !bit;
                if let Some((blk, off)) = g.page_mem.remove(&p) {
                    g.block_used[blk] -= 1;
                    g.free.push((blk, off));
                }
            }
        }
        if r.is_ok() {
            MARK_PAGES.fetch_add(mbinds.len() as u64, Ordering::Relaxed);
        } else {
            for &k in &mnew {
                g.mark_claimed[(k / 64) as usize] &= !(1u64 << (k % 64));
            }
        }
        m.bound.notify_all();
        r?;
        count.fetch_add(binds.len() as u64, Ordering::Relaxed);
        binds_here += 1;
        pages_here += binds.len() as u64;
    }
    // (Census, `DS_BACK_CENSUS=1`: by the engine allocation owning the
    // range, whether the granule before it was backed already (a region
    // growing in order), with the pages and time.)
    let seq = g0 > 0 && (m.backed[((g0 - 1) / 64) as usize].load(Ordering::Acquire) >> ((g0 - 1) % 64)) & 1 != 0;
    drop(g);
    // (Backing ahead: a recording thread's bind that grew a region in order
    // has the next granules bound ahead by the helper.)
    if binds_here > 0 && seq && !IN_AHEAD.with(|f| f.get()) && g1 + 1 < pages {
        let next = g1 + 1;
        let n = (BACK_AHEAD * gran).min(pages - next);
        back_ahead_queue(MIRROR_BASE + next * m.page, n * m.page);
    }
    let all_ns = t0.elapsed().as_nanos() as u64;
    if binds_here > 0 && back_census_on() {
        let mut c = BACK_CENSUS.lock();
        let e = c.get_or_insert_with(std::collections::HashMap::new).entry((crate::engine::owner(base), seq)).or_insert([0; 3]);
        e[0] += 1;
        e[1] += pages_here;
        e[2] += all_ns;
    }
    for (k, v) in [binds_here, pages_here, alloc_ns, bind_ns, all_ns, 0, 1].into_iter().enumerate() {
        BACKING[k].fetch_add(v, Ordering::Relaxed);
    }
    BACKING[5].fetch_max(all_ns, Ordering::Relaxed);
    // (A long one is a hitch cause: `gpu::hitch`.)
    super::hitch::note_since(super::hitch::Kind::Mirror, t0, || {
        format!("backing {} pages of the mirror at {base:#x} ({binds_here} binds here, {pages_here} pages, {} ms)", last + 1 - first, bind_ns / 1_000_000)
    });
    Ok(())
}

/// The backing granule in pages (`tune fe2_back_granule`; 0, the default:
/// 16 pages (1 MiB) with a device-local heap of 16 GiB or more, else 1: at
/// Castlewall's arrival 1 MiB granules cut the backings' thread time 55%
/// for 30% more video memory in the mirror, FINDINGS.md 2026-11-01).
fn back_granule() -> u64 {
    match super::tune::fe2_back_granule() {
        0 => {
            static AUTO: std::sync::OnceLock<u64> = std::sync::OnceLock::new();
            // (The card's size as `vk::card_size` sees it: under `DS_SIM_VRAM_MIB` the smaller
            // card's single pages, as on a real one.)
            *AUTO.get_or_init(|| if super::vk::card_size() >= 16 << 30 { 16 } else { 1 })
        }
        g => g.clamp(1, 4096).next_power_of_two(),
    }
}

/// `DS_BACK_CENSUS=1` (diagnosis): the mirror's backings that bound by
/// (engine allocation owning the range, the granule before it backed already)
/// -> [calls, pages bound, ns]; status line `mirror backings by allocation`.
fn back_census_on() -> bool {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ON.get_or_init(|| std::env::var("DS_BACK_CENSUS").ok().as_deref() == Some("1"))
}
static BACK_CENSUS: parking_lot::Mutex<Option<std::collections::HashMap<(u8, bool), [u64; 3]>>> = parking_lot::Mutex::new(None);

/// The backing census's status line ("" when off or empty).
pub fn back_census_line() -> String {
    if !back_census_on() {
        return String::new();
    }
    let g = BACK_CENSUS.lock();
    let Some(m) = g.as_ref() else { return String::new() };
    let names: std::collections::HashMap<u8, String> = crate::engine::allocators().into_iter().map(|a| (a.id, a.name)).collect();
    let mut v: Vec<(&(u8, bool), &[u64; 3])> = m.iter().collect();
    v.sort_by_key(|x| std::cmp::Reverse(x.1[2]));
    let parts: Vec<String> = v.iter().take(14).map(|((o, seq), c)| format!("{}{} {} calls, {} pages, {:.0} ms", names.get(o).cloned().unwrap_or_else(|| format!("owner {o}")), if *seq { " (in order)" } else { "" }, c[0], c[1], c[2] as f64 / 1e6)).collect();
    format!("    mirror backings by allocation (`DS_BACK_CENSUS`, since boot): {}
", parts.join("; "))
}

/// The mirror's backings (`back_in`): [binds, pages bound, allocation ns,
/// bind and wait ns, ns in calls that bound or waited, the longest such
/// call, such calls]; `profile reset` clears them.
static BACKING: [AtomicU64; 7] = [const { AtomicU64::new(0) }; 7];

/// Clears the backing counts (`profile reset`).
pub fn reset_backing() {
    for c in &BACKING {
        c.store(0, Ordering::Relaxed);
    }
}

/// Profile line: the mirror's backings in the window (each binds video
/// memory with `vkQueueBindSparse` on the main queue and waits for it), and
/// what aligned granules of 1, 2 and 4 MiB over the pages backed now would
/// hold (`tune fe2_back_granule`).
pub fn backing_line() -> String {
    let v: Vec<u64> = BACKING.iter().map(|c| c.load(Ordering::Relaxed)).collect();
    let Some(m) = MIRROR.get() else { return String::new() };
    if v[6] == 0 {
        return String::new();
    }
    let page = m.page;
    let (mut now, mut g) = (0u64, [0u64; 3]);
    for w in m.backed.iter() {
        let w = w.load(Ordering::Relaxed);
        now += w.count_ones() as u64;
        for (k, n) in [16u32, 32, 64].into_iter().enumerate() {
            let mask = if n == 64 { u64::MAX } else { (1u64 << n) - 1 };
            g[k] += (0..64 / n).filter(|i| (w >> (i * n)) & mask != 0).count() as u64 * n as u64;
        }
    }
    format!(
        "    mirror backings in the window: {} binds ({} pages, {} MiB; {:.1} ms with their waits, allocations {:.1} ms) for {} calls that bound or waited ({:.1} ms in all, longest {:.1} ms); backed now {} MiB, in whole granules of 1 / 2 / 4 MiB {} / {} / {} MiB\n",
        v[0],
        v[1],
        (v[1] * page) >> 20,
        v[3] as f64 / 1e6,
        v[2] as f64 / 1e6,
        v[6],
        v[4] as f64 / 1e6,
        v[5] as f64 / 1e6,
        (now * page) >> 20,
        (g[0] * page) >> 20,
        (g[1] * page) >> 20,
        (g[2] * page) >> 20
    )
}

/// Video memory backing the mirror so far, in MiB (pages are never
/// unbound).
pub fn backed_mib() -> u64 {
    let page = MIRROR.get().map_or(64 << 10, |m| m.page);
    (STATS[S_BACKED].load(Ordering::Relaxed) * page) >> 20
}

/// Video memory backing mirror pages no placement covered in the last
/// `flips` flips, in MiB (by words of 64 pages: a word placed lately counts
/// whole as used). What evicting cold mirror pages could give back.
pub fn cold_mib(flips: u32) -> u64 {
    let Some(m) = MIRROR.get() else { return 0 };
    let now = super::FRAMES.load(Ordering::Relaxed) as u32;
    let mut pages = 0u64;
    for (w, b) in m.backed.iter().enumerate() {
        let bits = b.load(Ordering::Relaxed);
        if bits != 0 && now.wrapping_sub(m.used[w].load(Ordering::Relaxed)) > flips {
            pages += bits.count_ones() as u64;
        }
    }
    (pages * m.page) >> 20
}

/// The mirror's cold pages given back (`unback_cold`): [words, pages,
/// blocks freed, words a placement kept while draining].
static UNBACKED: [AtomicU64; 4] = [const { AtomicU64::new(0) }; 4];

/// Front end 2's status: the mirror's video memory unbound since boot
/// (MiB), its blocks freed, cold words a placement kept.
pub fn unbacked_status() -> (u64, u64, u64) {
    let page = MIRROR.get().map_or(64 << 10, |m| m.page);
    ((UNBACKED[1].load(Ordering::Relaxed) * page) >> 20, UNBACKED[2].load(Ordering::Relaxed), UNBACKED[3].load(Ordering::Relaxed))
}

/// A cold word drains this many flips before it is unbound
/// (`unback_cold`): every stamp of a placement made before it began is
/// seen by then.
const DRAIN_FLIPS: u32 = 10;

/// `unback_cold` every this many milliseconds (its own thread).
const UNBACK_EVERY_MS: u64 = 5000;
/// Under video-memory pressure (`images::under_pressure`): a sweep this often, and a word cold
/// after this many flips without a placement (`tune mirror_cold` when that is less).
const PRESSURE_EVERY_MS: u64 = 1000;
const PRESSURE_COLD_FLIPS: u32 = 300;
const PRESSURE_COLD_SECS: f64 = 15.0;

/// Starts the thread that runs `unback_cold` (front end 2 only; called by
/// the completion thread, which it keeps clear of the unbinds' waits).
pub fn maybe_unback() {
    static STARTED: std::sync::Once = std::sync::Once::new();
    if !super::fe2::on() || MIRROR.get().is_none() {
        return;
    }
    STARTED.call_once(|| {
        let r = std::thread::Builder::new().name("gpu-unback".into()).spawn(|| {
            // (The flip rate between sweeps, for a coldness in time under pressure.)
            let (mut last_flips, mut last_at) = (super::FRAMES.load(Ordering::Relaxed), std::time::Instant::now());
            let mut rate = 60.0f64;
            loop {
                // (Under video-memory pressure, `images::under_pressure`: a sweep every second,
                // words cold for PRESSURE_COLD_SECS (at least PRESSURE_COLD_FLIPS flips), more of
                // them a sweep; the other PC's 12 GB card ran 2 GB over its budget with the mirror
                // at 3.2 GB, 454 MiB of it cold, 2026-11-21. In time, not flips: 300 flips were
                // 2.5 s at the tour's 120 a second or uncapped, the review of 2026-11-21 night.)
                let pressure = super::images::under_pressure();
                std::thread::sleep(std::time::Duration::from_millis(if pressure { PRESSURE_EVERY_MS } else { UNBACK_EVERY_MS }));
                let (f, at) = (super::FRAMES.load(Ordering::Relaxed), std::time::Instant::now());
                let dt = at.duration_since(last_at).as_secs_f64();
                if dt > 0.5 && f >= last_flips {
                    rate = (f - last_flips) as f64 / dt;
                }
                (last_flips, last_at) = (f, at);
                let cold = super::tune::mirror_cold().min(u32::MAX as u64) as u32;
                if cold == 0 {
                    continue;
                }
                let pressure_flips = ((rate * PRESSURE_COLD_SECS) as u32).max(PRESSURE_COLD_FLIPS);
                let (flips, words) = if super::images::under_pressure() { (cold.min(pressure_flips), 2048) } else { (cold, 256) };
                if let Err(e) = unback_cold(flips, words) {
                    crate::warn!("resident: cold mirror pages not unbound: {e}");
                }
            }
        });
        if let Err(e) = r {
            crate::warn!("resident: no thread for the mirror's cold pages: {e}");
        }
    });
}

/// Gives back the mirror's words of 64 pages that no placement covered in
/// the last `flips` flips (earlier levels' data, places left behind: after
/// six level loads 3.5 of the mirror's 6.7 GB, FINDINGS.md 2026-11-02), at
/// most `max_words` new ones a call, in two phases. A cold word first
/// drains: its backed bits are cleared, so a placement reaching it takes
/// `back_in`'s locked path, which sets them again and keeps the word
/// (`restore_draining`). A sweep at least DRAIN_FLIPS flips later unbinds a
/// drained word whose stamp has not moved (no placement since: one that
/// read the bits before they were cleared stamped the word first, and its
/// stamp is seen by then): its memory goes to later backings first
/// (`take_page`), a block left with no page bound is freed, and front end 2
/// holds its guest pages invalid (`fe2::invalidate`), so a later placement
/// backs and uploads them again. No batch can read it then: every batch
/// that placed it was recorded `flips` flips before.
fn unback_cold(flips: u32, max_words: usize) -> Result<(), String> {
    let m = MIRROR.get().ok_or("no mirror")?;
    let t0 = std::time::Instant::now();
    let now = super::FRAMES.load(Ordering::Relaxed) as u32;
    if now < flips {
        return Ok(());
    }
    let mut g = m.blocks.lock();
    if g.binding || !g.pending.is_empty() {
        return Ok(());
    }
    // Drained words: unbound unless placed since.
    let mut words: Vec<(usize, u64)> = Vec::new();
    let ready: Vec<usize> = g.draining.iter().filter(|&(_, &(_, _, f))| now.wrapping_sub(f) >= DRAIN_FLIPS).map(|(&w, _)| w).collect();
    // (Front end 4's held pages: the mirror holds their newest words, never
    // unbound.)
    let held = |w: usize| {
        let lo = MIRROR_BASE + w as u64 * 64 * m.page;
        super::fe2::c4_any_held(lo, lo + 64 * m.page)
    };
    for w in ready {
        let (bits, u, _) = g.draining.remove(&w).unwrap();
        if m.used[w].load(Ordering::Acquire) != u || held(w) {
            m.backed[w].fetch_or(bits, Ordering::AcqRel);
            UNBACKED[3].fetch_add(1, Ordering::Relaxed);
        } else {
            words.push((w, bits));
        }
    }
    // New cold words drain.
    let mut started = 0;
    for w in 0..m.backed.len() {
        if started >= max_words {
            break;
        }
        let bits = m.backed[w].load(Ordering::Acquire);
        if bits == 0 || g.claimed[w] != bits || g.draining.contains_key(&w) {
            continue;
        }
        let u = m.used[w].load(Ordering::Acquire);
        if now.wrapping_sub(u) <= flips || held(w) {
            continue;
        }
        m.backed[w].fetch_and(!bits, Ordering::AcqRel);
        g.draining.insert(w, (bits, u, now));
        started += 1;
    }
    if words.is_empty() {
        return Ok(());
    }
    let mut unbinds = Vec::new();
    for &(w, bits) in &words {
        for i in 0..64u64 {
            if bits & (1 << i) != 0 {
                unbinds.push(vk::SparseMemoryBind::default().resource_offset((w as u64 * 64 + i) * m.page).size(m.page));
            }
        }
    }
    // Other threads' binds go on meanwhile (`binding` stays as it is): the
    // words' pages stay claimed until the unbind is done, so a placement
    // reaching them waits for it (`bound`) and binds them again after.
    let c = ctx();
    let r = parking_lot::MutexGuard::unlocked(&mut g, || {
        let bufs = [vk::SparseBufferMemoryBindInfo::default().buffer(m.buf).binds(&unbinds)];
        let info = vk::BindSparseInfo::default().buffer_binds(&bufs);
        let fence = unsafe { c.device.create_fence(&vk::FenceCreateInfo::default(), None) }.map_err(|e| format!("vkCreateFence: {e}"))?;
        let r = {
            let q = c.queue.lock();
            unsafe { c.device.queue_bind_sparse(*q, &[info], fence) }
        };
        let r = r.map_err(|e| format!("vkQueueBindSparse (unbinding): {e}")).and_then(|_| unsafe { c.device.wait_for_fences(&[fence], true, u64::MAX) }.map_err(|e| format!("unbind fence: {e}")));
        unsafe { c.device.destroy_fence(fence, None) };
        r
    });
    let (mut pages, mut freed) = (0u64, 0u64);
    match &r {
        Ok(()) => {
            for &(w, bits) in &words {
                for i in 0..64u64 {
                    if bits & (1 << i) == 0 {
                        continue;
                    }
                    let p = w as u64 * 64 + i;
                    g.claimed[w] &= !(1 << i);
                    if let Some((blk, off)) = g.page_mem.remove(&p) {
                        g.block_used[blk] -= 1;
                        g.free.push((blk, off));
                    }
                    pages += 1;
                }
                super::fe2::invalidate(MIRROR_BASE + w as u64 * 64 * m.page, 64 * m.page);
            }
            // Blocks left with no page bound (not the one being filled;
            // the maps' blocks stay bound).
            for b in 0..g.mems.len() {
                if Some(b) != g.cur && g.block_used[b] == 0 && g.mems[b] != vk::DeviceMemory::null() {
                    g.free.retain(|&(fb, _)| fb != b);
                    unsafe { c.device.free_memory(g.mems[b], None) };
                    g.mems[b] = vk::DeviceMemory::null();
                    freed += 1;
                }
            }
        }
        Err(_) => {
            // Still bound: the bits back.
            for &(w, bits) in &words {
                m.backed[w].fetch_or(bits, Ordering::AcqRel);
            }
        }
    }
    m.bound.notify_all();
    drop(g);
    STATS[S_BACKED].fetch_sub(pages, Ordering::Relaxed);
    let first = UNBACKED[0].fetch_add(words.len() as u64, Ordering::Relaxed) == 0;
    UNBACKED[1].fetch_add(pages, Ordering::Relaxed);
    UNBACKED[2].fetch_add(freed, Ordering::Relaxed);
    if first || freed > 0 {
        crate::info!("resident: {} cold words of the mirror unbound ({} MiB; {} blocks of {} MiB freed) in {} ms", words.len(), (pages * m.page) >> 20, freed, BLOCK >> 20, t0.elapsed().as_millis());
    }
    r
}

/// A placement reaching draining words (`unback_cold`) keeps them: their
/// backed bits set again (under the backing's lock).
fn restore_draining(g: &mut Backing, m: &Mirror, w0: u64, w1: u64) {
    if g.draining.is_empty() {
        return;
    }
    for w in w0..=w1 {
        if let Some((bits, _, _)) = g.draining.remove(&(w as usize)) {
            m.backed[w as usize].fetch_or(bits, Ordering::AcqRel);
            UNBACKED[3].fetch_add(1, Ordering::Relaxed);
        }
    }
}

/// Video memory of the written-bit maps, in MiB (sparse: their bound
/// pages; dense: their whole size).
pub fn marks_mib() -> u64 {
    match SPARSE_MARKS.get() {
        Some(sm) => (MARK_PAGES.load(Ordering::Relaxed) * sm.page) >> 20,
        None => (MARKS.get().is_some() as u64 + MARKS_C.get().is_some() as u64) * (MARK_BYTES >> 20),
    }
}

/// Runs `record` in a one-off command buffer on the main queue and waits.
fn one_off(record: impl FnOnce(vk::CommandBuffer)) -> Result<(), String> {
    let c = ctx();
    unsafe {
        let pool = c.device.create_command_pool(&vk::CommandPoolCreateInfo::default().queue_family_index(c.queue_family), None).map_err(|e| format!("pool: {e}"))?;
        let r = (|| {
            let cmd = c.device.allocate_command_buffers(&vk::CommandBufferAllocateInfo::default().command_pool(pool).command_buffer_count(1)).map_err(|e| format!("cmd: {e}"))?[0];
            c.device.begin_command_buffer(cmd, &vk::CommandBufferBeginInfo::default().flags(vk::CommandBufferUsageFlags::ONE_TIME_SUBMIT)).map_err(|e| format!("begin: {e}"))?;
            record(cmd);
            c.device.end_command_buffer(cmd).map_err(|e| format!("end: {e}"))?;
            let fence = c.device.create_fence(&vk::FenceCreateInfo::default(), None).map_err(|e| format!("fence: {e}"))?;
            let r = c.submit_cmds(&[cmd], &[], &[], fence).and_then(|_| c.device.wait_for_fences(&[fence], true, u64::MAX).map_err(|e| format!("wait: {e}")));
            c.device.destroy_fence(fence, None);
            r
        })();
        c.device.destroy_command_pool(pool, None);
        r
    }
}

/// Copies `data` into the mirror at guest address `at` (backed first).
pub fn upload_now(at: u64, data: &[u8]) -> Result<(), String> {
    let m = MIRROR.get().ok_or("no mirror")?;
    let n = data.len() as u64;
    back(at, n)?;
    let c = ctx();
    let (sbuf, smem, ptr) = c.host_buffer(n.max(4), vk::BufferUsageFlags::TRANSFER_SRC)?;
    unsafe { std::ptr::copy_nonoverlapping(data.as_ptr(), ptr, data.len()) };
    let r = one_off(|cmd| unsafe {
        c.device.cmd_copy_buffer(cmd, sbuf, m.buf, &[vk::BufferCopy { src_offset: 0, dst_offset: at - MIRROR_BASE, size: n }]);
    });
    unsafe {
        c.device.destroy_buffer(sbuf, None);
        c.device.free_memory(smem, None);
    }
    STATS[S_UPLOADS].fetch_add(1, Ordering::Relaxed);
    r
}

/// Reads `n` bytes of the mirror at guest address `at` (backed).
pub fn readback_now(at: u64, n: u64) -> Result<Vec<u8>, String> {
    let m = MIRROR.get().ok_or("no mirror")?;
    let c = ctx();
    let (dbuf, dmem, ptr) = c.host_buffer(n.max(4), vk::BufferUsageFlags::TRANSFER_DST)?;
    let r = one_off(|cmd| unsafe {
        c.device.cmd_copy_buffer(cmd, m.buf, dbuf, &[vk::BufferCopy { src_offset: at - MIRROR_BASE, dst_offset: 0, size: n }]);
    });
    let out = r.map(|_| unsafe { std::slice::from_raw_parts(ptr, n as usize) }.to_vec());
    unsafe {
        c.device.destroy_buffer(dbuf, None);
        c.device.free_memory(dmem, None);
    }
    STATS[S_READBACKS].fetch_add(1, Ordering::Relaxed);
    out
}

/// Makes the mirror and runs the self-test (after Vulkan initialization,
/// when `wanted`). On failure residency stays off.
pub fn init() {
    if !wanted() || MIRROR.get().is_some() {
        return;
    }
    match make_mirror() {
        Ok(m) => {
            let _ = MIRROR.set(m);
        }
        Err(e) => {
            crate::error!("resident: DS_RESIDENT=1 but no mirror: {e}");
            return;
        }
    }
    ON.store(true, Ordering::Release);
    match self_test() {
        Ok(s) => {
            crate::info!("resident: self-test passed ({s})");
            if wanted_mode2() {
                match owned_self_test() {
                    Ok(s) => {
                        MODE2.store(true, Ordering::Release);
                        crate::info!("resident: mirror mode (DS_RESIDENT=2): {s}");
                    }
                    Err(e) => crate::error!("resident: mirror mode's self-test failed: {e}; mirror mode off"),
                }
            }
        }
        Err(e) => {
            crate::error!("resident: self-test failed: {e}; residency off");
            ON.store(false, Ordering::Release);
        }
    }
}

/// Write faults on armed pages of a private region, and a pattern through
/// the mirror's last page and back.
fn self_test() -> Result<String, String> {
    use crate::platform::os::Win32::System::Memory::{VirtualAlloc, VirtualFree, MEM_COMMIT, MEM_RELEASE, MEM_RESERVE, PAGE_READWRITE};
    let n = 16 * PAGE;
    let a = unsafe { VirtualAlloc(std::ptr::null(), n as usize, MEM_COMMIT | MEM_RESERVE, PAGE_READWRITE) } as u64;
    if a == 0 {
        return Err("VirtualAlloc".into());
    }
    let r = (|| {
        let armed = arm(a, 8 * PAGE);
        if armed != 8 {
            return Err(format!("armed {armed} of 8 pages"));
        }
        let faults = STATS[S_FAULTS].load(Ordering::Relaxed);
        // Writes to pages 0 and 3 (armed) fault and disarm them; page 9 is not armed.
        for i in [0u64, 3, 9] {
            unsafe { std::ptr::write_volatile((a + i * PAGE + 8) as *mut u64, 0x5EED_0000 + i) };
        }
        let taken = STATS[S_FAULTS].load(Ordering::Relaxed) - faults;
        let st: Vec<u8> = (0..10).map(|i| state(a + i * PAGE)).collect();
        if taken != 2 || st[0] != 0 || st[3] != 0 || st[1] != ST_ARMED | ST_VALID || st[9] != 0 {
            return Err(format!("{taken} faults, states {st:?}"));
        }
        if (0..3u64).map(|i| unsafe { std::ptr::read_volatile((a + i * PAGE * 3 + 8) as *const u64) }).collect::<Vec<_>>() != [0x5EED_0000, 0x5EED_0003, 0] {
            return Err("written values".into());
        }
        Ok(())
    })();
    disarm_range(a, a + n);
    unsafe { VirtualFree(a as _, 0, MEM_RELEASE) };
    r?;
    let m = MIRROR.get().ok_or("no mirror")?;
    let at = MIRROR_END - m.page;
    let data: Vec<u8> = (0..m.page).map(|i| (i * 7 + 3) as u8).collect();
    upload_now(at, &data)?;
    let back = readback_now(at, m.page)?;
    if back != data {
        let first = back.iter().zip(&data).position(|(x, y)| x != y).unwrap_or(0);
        return Err(format!("mirror readback differs from byte {first}"));
    }
    Ok(format!("2 write faults on 8 armed pages; {} KiB through the mirror and back", m.page >> 10))
}

/// Feasibility of section-backed guest memory (`DS_SECTION_TEST=1`, at the
/// first GPU use): a 64 MiB pagefile section mapped as two views in
/// placeholders; the GPU imports one view and fills parts of it while the
/// other view shows the same pages, then while the other view is
/// read-only. If imports of section views work and protecting one view
/// leaves the GPU's writes through the other intact, guest memory could be
/// protected for residency while the GPU imports an alias of it.
pub fn section_test_maybe() {
    if std::env::var("DS_SECTION_TEST").ok().as_deref() != Some("1") {
        return;
    }
    match section_test() {
        Ok(s) => crate::info!("section test passed: {s}"),
        Err(e) => crate::error!("section test failed: {e}"),
    }
}

/// `DS_MIXED_TEST=1` (boot test): one GPU import over two adjacent private
/// write-watched allocations made in placeholders (the mixed guest range,
/// FRONTEND2.md "GPU-quiet mappings on sections"): the import, GPU fills of
/// both parts as the CPU sees them, and write watching reporting the CPU's
/// writes but none of the GPU's (front end 2's write-backs must not read as
/// CPU writes).
pub fn mixed_test_maybe() {
    if std::env::var("DS_MIXED_TEST").ok().as_deref() != Some("1") {
        return;
    }
    match mixed_test() {
        Ok(s) => crate::info!("mixed-range test passed: {s}"),
        Err(e) => crate::error!("mixed-range test failed: {e}"),
    }
}

fn mixed_test() -> Result<String, String> {
    use crate::platform::os::Win32::Foundation::GetLastError;
    use crate::platform::os::Win32::System::Memory::{GetWriteWatch, VirtualAlloc2, VirtualFree, MEM_COMMIT, MEM_PRESERVE_PLACEHOLDER, MEM_RELEASE, MEM_REPLACE_PLACEHOLDER, MEM_RESERVE, MEM_RESERVE_PLACEHOLDER, PAGE_NOACCESS, PAGE_READWRITE};
    const MEM_WRITE_WATCH: u32 = 0x0020_0000;
    let m: usize = 4 << 20;
    let c = ctx();
    unsafe {
        let base = VirtualAlloc2(std::ptr::null_mut(), std::ptr::null(), 2 * m, MEM_RESERVE | MEM_RESERVE_PLACEHOLDER, PAGE_NOACCESS, std::ptr::null_mut(), 0);
        if base.is_null() {
            return Err(format!("placeholder: error {}", GetLastError()));
        }
        let b0 = base as u64;
        if VirtualFree(base, m, MEM_RELEASE | MEM_PRESERVE_PLACEHOLDER) == 0 {
            return Err(format!("splitting the placeholder: error {}", GetLastError()));
        }
        for k in 0..2u64 {
            let q = VirtualAlloc2(std::ptr::null_mut(), (b0 + k * m as u64) as _, m, MEM_RESERVE | MEM_COMMIT | MEM_REPLACE_PLACEHOLDER | MEM_WRITE_WATCH, PAGE_READWRITE, std::ptr::null_mut(), 0);
            if q.is_null() {
                return Err(format!("private write-watched allocation {k}: error {}", GetLastError()));
            }
        }
        let r = (|| -> Result<String, String> {
            let mut out = Vec::new();
            let take = |at: u64| -> Result<Vec<u64>, String> {
                let mut addrs = vec![std::ptr::null_mut::<std::ffi::c_void>(); 2048];
                let (mut count, mut gran) = (addrs.len(), 0u32);
                if GetWriteWatch(1, at as _, m, addrs.as_mut_ptr(), &mut count, &mut gran) != 0 {
                    return Err(format!("GetWriteWatch at {at:#x}: error {}", GetLastError()));
                }
                Ok(addrs[..count].iter().map(|&a| a as u64 - at).collect())
            };
            // (Committing reads as no write; the marks start clear.)
            take(b0)?;
            take(b0 + m as u64)?;
            let (buf, mem, _) = c.import_host(b0, 2 * m as u64, vk::BufferUsageFlags::TRANSFER_DST | vk::BufferUsageFlags::STORAGE_BUFFER)?;
            out.push("one import over two allocations");
            // (What the import itself leaves in the marks: the driver locks
            // the pages for writing.)
            let (i0, i1) = (take(b0)?.len(), take(b0 + m as u64)?.len());
            let at_import = format!("the import left {i0} and {i1} (of up to 64 each) pages marked written");
            out.push(&at_import);
            let r = (|| -> Result<(), String> {
                // GPU fills across the boundary.
                one_off(|cmd| c.device.cmd_fill_buffer(cmd, buf, m as u64 - (1 << 20), 2 << 20, 0x5A5A_5A5A))?;
                let p = b0 as *const u32;
                let seen = (0..(1 << 19)).step_by(997).all(|i| std::ptr::read_volatile(p.add((m - (1 << 20)) / 4 + i)) == 0x5A5A_5A5A);
                if !seen {
                    return Err("the CPU does not see the GPU's fill across the allocations".into());
                }
                out.push("GPU fills across the boundary seen by the CPU");
                let (g0, g1) = (take(b0)?, take(b0 + m as u64)?);
                // Once more, the import resident since the first fill: what
                // the GPU's writes alone leave in the marks.
                one_off(|cmd| c.device.cmd_fill_buffer(cmd, buf, m as u64 - (1 << 20), 2 << 20, 0x6B6B_6B6B))?;
                let (h0, h1) = (take(b0)?, take(b0 + m as u64)?);
                if !h0.is_empty() || !h1.is_empty() {
                    return Err(format!("write watching reported the GPU's writes: after the first fill (the import's first use) {} and {} pages, after the second {} and {} (of 1024 each; the import itself left {i0} and {i1})", g0.len(), g1.len(), h0.len(), h1.len()));
                }
                crate::info!("mixed-range test: write watching after the first GPU fill (the import's first use) {} and {} pages, after the second none (the import itself left {i0} and {i1})", g0.len(), g1.len());
                out.push("write watching reports no GPU write once the import is in use");
                // A CPU write in each part is reported in its own.
                std::ptr::write_volatile((b0 + 0x3000) as *mut u32, 1);
                std::ptr::write_volatile((b0 + m as u64 + 0x7000) as *mut u32, 2);
                let (g0, g1) = (take(b0)?, take(b0 + m as u64)?);
                if g0 != vec![0x3000] || g1 != vec![0x7000] {
                    return Err(format!("CPU writes reported as {g0:x?} and {g1:x?}"));
                }
                out.push("CPU writes reported per allocation");
                Ok(())
            })();
            c.device.destroy_buffer(buf, None);
            c.device.free_memory(mem, None);
            r.map(|_| out.join("; "))
        })();
        VirtualFree(base, 0, MEM_RELEASE);
        VirtualFree((b0 + m as u64) as _, 0, MEM_RELEASE);
        r
    }
}

fn section_test() -> Result<String, String> {
    use crate::platform::os::Win32::Foundation::{CloseHandle, GetLastError, INVALID_HANDLE_VALUE};
    use crate::platform::os::Win32::System::Memory::{
        CreateFileMappingW, MapViewOfFile3, UnmapViewOfFile2, VirtualAlloc2, VirtualFree, VirtualProtect, MEM_PRESERVE_PLACEHOLDER, MEM_RELEASE, MEM_REPLACE_PLACEHOLDER,
        MEM_RESERVE, MEM_RESERVE_PLACEHOLDER, PAGE_NOACCESS, PAGE_READONLY, PAGE_READWRITE,
    };
    use crate::platform::os::Win32::System::Threading::GetCurrentProcess;
    let size: usize = 64 << 20;
    let c = ctx();
    unsafe {
        let sec = CreateFileMappingW(INVALID_HANDLE_VALUE, std::ptr::null(), PAGE_READWRITE, 0, size as u32, std::ptr::null());
        if sec.is_null() {
            return Err(format!("CreateFileMappingW: {}", GetLastError()));
        }
        let base = VirtualAlloc2(std::ptr::null_mut(), std::ptr::null(), 2 * size, MEM_RESERVE | MEM_RESERVE_PLACEHOLDER, PAGE_NOACCESS, std::ptr::null_mut(), 0);
        if base.is_null() {
            CloseHandle(sec);
            return Err(format!("VirtualAlloc2 placeholder: {}", GetLastError()));
        }
        if VirtualFree(base, size, MEM_RELEASE | MEM_PRESERVE_PLACEHOLDER) == 0 {
            return Err(format!("splitting the placeholder: {}", GetLastError()));
        }
        let a = MapViewOfFile3(sec, GetCurrentProcess(), base, 0, size, MEM_REPLACE_PLACEHOLDER, PAGE_READWRITE, std::ptr::null_mut(), 0);
        let b = MapViewOfFile3(sec, GetCurrentProcess(), (base as usize + size) as _, 0, size, MEM_REPLACE_PLACEHOLDER, PAGE_READWRITE, std::ptr::null_mut(), 0);
        if a.Value.is_null() || b.Value.is_null() {
            return Err(format!("MapViewOfFile3: {}", GetLastError()));
        }
        let (pa, pb) = (a.Value as *mut u32, b.Value as *mut u32);
        let r = (|| -> Result<String, String> {
            // The two views share their pages.
            std::ptr::write_volatile(pa.add(7), 0x1234_5678);
            if std::ptr::read_volatile(pb.add(7)) != 0x1234_5678 {
                return Err("the views do not share pages".into());
            }
            // The GPU imports view b and fills parts of it.
            let (buf, mem, _) = c.import_host(pb as u64, size as u64, vk::BufferUsageFlags::TRANSFER_DST | vk::BufferUsageFlags::STORAGE_BUFFER)?;
            let fill = |off: u64, v: u32| one_off(|cmd| c.device.cmd_fill_buffer(cmd, buf, off, 1 << 20, v));
            let check = |off: u64, v: u32| (0..(1 << 18)).step_by(997).all(|i| std::ptr::read_volatile(pa.add(off as usize / 4 + i)) == v);
            let mut out = Vec::new();
            let r = (|| -> Result<(), String> {
                fill(0, 0xA5A5_A5A5)?;
                if !check(0, 0xA5A5_A5A5) {
                    return Err("view a does not show the GPU's fill of view b".into());
                }
                out.push("GPU writes to an imported view show in the other");
                // View a read-only: the GPU keeps writing through view b.
                let mut old = 0u32;
                if VirtualProtect(pa as _, size, PAGE_READONLY, &mut old) == 0 {
                    return Err(format!("VirtualProtect view a: {}", GetLastError()));
                }
                for k in 0..16u64 {
                    fill((k + 1) << 20, 0x5A00_0000 + k as u32)?;
                }
                if !(0..16u64).all(|k| check((k + 1) << 20, 0x5A00_0000 + k as u32)) {
                    return Err("GPU fills through view b missing in read-only view a".into());
                }
                VirtualProtect(pa as _, size, PAGE_READWRITE, &mut old);
                out.push("16 GPU fills through view b while view a was read-only, no device loss");
                Ok(())
            })();
            c.device.destroy_buffer(buf, None);
            c.device.free_memory(mem, None);
            r.map(|_| out.join("; "))
        })();
        UnmapViewOfFile2(GetCurrentProcess(), a, MEM_PRESERVE_PLACEHOLDER);
        UnmapViewOfFile2(GetCurrentProcess(), b, MEM_PRESERVE_PLACEHOLDER);
        // Granularity: a 16 KiB view at a 16 KiB (not 64 KiB) aligned address
        // and section offset, split out of a placeholder.
        let fine = (|| -> Result<&'static str, String> {
            let at = base as usize + (1 << 20) + (16 << 10);
            if VirtualFree(at as _, 16 << 10, MEM_RELEASE | MEM_PRESERVE_PLACEHOLDER) == 0 {
                return Err(format!("splitting a placeholder at 16 KiB granularity: error {}", GetLastError()));
            }
            let v = MapViewOfFile3(sec, GetCurrentProcess(), at as _, 16 << 10, 16 << 10, MEM_REPLACE_PLACEHOLDER, PAGE_READWRITE, std::ptr::null_mut(), 0);
            if v.Value.is_null() {
                return Err(format!("a 16 KiB view at 16 KiB alignment: error {}", GetLastError()));
            }
            UnmapViewOfFile2(GetCurrentProcess(), v, MEM_PRESERVE_PLACEHOLDER);
            VirtualFree(at as _, 0, MEM_RELEASE);
            VirtualFree((at + (16 << 10)) as _, 0, MEM_RELEASE);
            Ok("16 KiB views at 16 KiB granularity work")
        })();
        // One import spanning four adjacent 64 KiB views of non-contiguous
        // section offsets (the title maps buffers in batch-mapped pieces).
        let multi = (|| -> Result<&'static str, String> {
            let at = base as usize + size;
            let piece = 64usize << 10;
            let offs = [3usize, 1, 2, 0].map(|k| (8 << 20) + k * piece);
            let mut views = Vec::new();
            for (i, &off) in offs.iter().enumerate() {
                let va = at + i * piece;
                if VirtualFree(va as _, piece, MEM_RELEASE | MEM_PRESERVE_PLACEHOLDER) == 0 {
                    return Err(format!("splitting placeholders: error {}", GetLastError()));
                }
                let v = MapViewOfFile3(sec, GetCurrentProcess(), va as _, off as u64, piece, MEM_REPLACE_PLACEHOLDER, PAGE_READWRITE, std::ptr::null_mut(), 0);
                if v.Value.is_null() {
                    return Err(format!("mapping piece {i}: error {}", GetLastError()));
                }
                views.push(v);
            }
            let r = (|| -> Result<&'static str, String> {
                let (buf, mem, _) = c.import_host(at as u64, 4 * piece as u64, vk::BufferUsageFlags::TRANSFER_DST | vk::BufferUsageFlags::STORAGE_BUFFER).map_err(|e| format!("importing across views: {e}"))?;
                let r = one_off(|cmd| {
                    for i in 0..4u64 {
                        c.device.cmd_fill_buffer(cmd, buf, i * piece as u64, piece as u64, 0xC0DE_0000 + i as u32);
                    }
                });
                c.device.destroy_buffer(buf, None);
                c.device.free_memory(mem, None);
                r?;
                if !(0..4usize).all(|i| std::ptr::read_volatile(((at + i * piece) + 4092) as *const u32) == 0xC0DE_0000 + i as u32) {
                    return Err("GPU fills across views landed wrong".into());
                }
                Ok("one import spans adjacent views of scattered section offsets")
            })();
            for (i, v) in views.into_iter().enumerate() {
                UnmapViewOfFile2(GetCurrentProcess(), v, MEM_PRESERVE_PLACEHOLDER);
                VirtualFree((at + i * piece) as _, 0, MEM_RELEASE);
            }
            VirtualFree((at + 4 * piece) as _, 0, MEM_RELEASE);
            r
        })();
        VirtualFree(base, 0, MEM_RELEASE);
        VirtualFree((base as usize + size) as _, 0, MEM_RELEASE);
        CloseHandle(sec);
        let fine = match fine {
            Ok(s) => s.to_string(),
            Err(e) => format!("no fine views ({e})"),
        };
        let multi = match multi {
            Ok(s) => s.to_string(),
            Err(e) => format!("no imports across views ({e})"),
        };
        r.map(|s| format!("{s}; {fine}; {multi}"))
    }
}

/// Whether changing the protection of host pages the GPU imports, as guest
/// memory is imported, disturbs the GPU (`DS_PROTECT_TEST=1`, at the first
/// GPU use). Write tracking by protection (`track`, DS_TRACK) lost the
/// device in 3 of 5 runs, but it took its faults inside the title's code,
/// which overwrites the title's values below its stack pointer (FINDINGS.md
/// 2026-10-23, 2026-10-26): this separates the two. A 64 MiB region
/// allocated as guest memory is (write-watched) is imported; for 2,000
/// rounds the GPU fills parts of it and copies out of other parts while a
/// second thread keeps switching those parts between read-only, no-access
/// and read-write; then every part is checked, and the write marks of the
/// GPU's writes are counted (none expected: the GPU does not set them).
pub fn protect_test_maybe() {
    if std::env::var("DS_PROTECT_TEST").ok().as_deref() != Some("1") {
        return;
    }
    match protect_test() {
        Ok(s) => crate::info!("protect test passed: {s}"),
        Err(e) => crate::error!("protect test failed: {e}"),
    }
}

fn protect_test() -> Result<String, String> {
    use crate::platform::os::Win32::System::Memory::{GetWriteWatch, VirtualAlloc, VirtualFree, VirtualProtect, MEM_COMMIT, MEM_RELEASE, MEM_RESERVE, PAGE_NOACCESS, PAGE_READONLY, PAGE_READWRITE};
    const MEM_WRITE_WATCH: u32 = 0x0020_0000;
    const PART: usize = 1 << 20;
    const PARTS: usize = 64;
    const ROUNDS: u32 = 2000;
    let size = PART * PARTS;
    let c = ctx();
    let base = unsafe { VirtualAlloc(std::ptr::null(), size, MEM_RESERVE | MEM_COMMIT | MEM_WRITE_WATCH, PAGE_READWRITE) } as usize;
    if base == 0 {
        return Err("VirtualAlloc failed".into());
    }
    let part = |p: usize| unsafe { std::slice::from_raw_parts_mut((base + p * PART) as *mut u32, PART / 4) };
    for p in 0..PARTS {
        part(p).fill(0x1000_0000 + p as u32);
    }
    let (buf, mem, _) = match c.import_host(base as u64, size as u64, vk::BufferUsageFlags::TRANSFER_SRC | vk::BufferUsageFlags::TRANSFER_DST | vk::BufferUsageFlags::STORAGE_BUFFER) {
        Ok(x) => x,
        Err(e) => {
            unsafe { VirtualFree(base as _, 0, MEM_RELEASE) };
            return Err(format!("import: {e}"));
        }
    };
    // Write marks so far (the CPU's pattern) are taken first.
    let marks = || -> usize {
        let mut addrs = vec![std::ptr::null_mut::<std::ffi::c_void>(); PARTS * PART / 4096];
        let (mut n, mut gran) = (addrs.len(), 0u32);
        unsafe { GetWriteWatch(1, base as _, size, addrs.as_mut_ptr(), &mut n, &mut gran) };
        n
    };
    let cpu_marks = marks();
    let stop = AtomicBool::new(false);
    let toggles = AtomicU64::new(0);
    let gpu = std::thread::scope(|s| {
        // Parts 32..64: switched between protections all along; the GPU
        // fills the even ones and copies out of the odd ones.
        s.spawn(|| {
            let mut i = 0usize;
            let mut old = 0u32;
            while !stop.load(Ordering::Relaxed) {
                let prot = [PAGE_READONLY, PAGE_NOACCESS, PAGE_READWRITE][i % 3];
                unsafe { VirtualProtect((base + (32 + (i / 3) % 32) * PART) as _, PART, prot, &mut old) };
                i += 1;
                toggles.fetch_add(1, Ordering::Relaxed);
                if i % 64 == 0 {
                    std::thread::yield_now();
                }
            }
        });
        let r = (|| -> Result<(), String> {
            for round in 0..ROUNDS {
                let k = round as usize % 16;
                one_off(|cmd| unsafe {
                    c.device.cmd_fill_buffer(cmd, buf, ((32 + 2 * k) * PART) as u64, PART as u64, round);
                    let copy = vk::BufferCopy { src_offset: ((33 + 2 * k) * PART) as u64, dst_offset: (k * PART) as u64, size: PART as u64 };
                    c.device.cmd_copy_buffer(cmd, buf, buf, &[copy]);
                })
                .map_err(|e| format!("round {round}: {e}"))?;
            }
            Ok(())
        })();
        stop.store(true, Ordering::Relaxed);
        r
    });
    let mut old = 0u32;
    unsafe { VirtualProtect(base as _, size, PAGE_READWRITE, &mut old) };
    let gpu_marks = marks();
    // Imports made while pages are already protected (write tracking by
    // protection arms pages that later bindings import): parts 4-5 with
    // part 4 read-only, parts 6-7 with part 6 no-access; the GPU then fills
    // the second part of each.
    let import_over = |p: usize, prot: u32| -> String {
        let mut old = 0u32;
        unsafe { VirtualProtect((base + p * PART) as _, PART, prot, &mut old) };
        let r = c.import_host((base + p * PART) as u64, 2 * PART as u64, vk::BufferUsageFlags::TRANSFER_SRC | vk::BufferUsageFlags::TRANSFER_DST | vk::BufferUsageFlags::STORAGE_BUFFER);
        let out = match r {
            Ok((b, m, _)) => {
                let f = one_off(|cmd| unsafe { c.device.cmd_fill_buffer(cmd, b, PART as u64, PART as u64, 0x77) });
                unsafe {
                    c.device.destroy_buffer(b, None);
                    c.device.free_memory(m, None);
                }
                match f {
                    Ok(()) => "imported, GPU fill done".to_string(),
                    Err(e) => format!("imported, GPU fill failed: {e}"),
                }
            }
            Err(e) => format!("import failed: {e}"),
        };
        unsafe { VirtualProtect((base + p * PART) as _, PART, PAGE_READWRITE, &mut old) };
        out
    };
    let ro = import_over(4, PAGE_READONLY);
    let na = import_over(6, PAGE_NOACCESS);
    let r = gpu.and_then(|_| {
        for k in 0..16usize {
            let last = (0..ROUNDS).rev().find(|r| *r as usize % 16 == k).unwrap();
            if let Some(i) = part(32 + 2 * k).iter().position(|&w| w != last) {
                return Err(format!("part {} word {i}: {:#x}, not the last fill {last:#x}", 32 + 2 * k, part(32 + 2 * k)[i]));
            }
            let want = 0x1000_0000 + (33 + 2 * k) as u32;
            if let Some(i) = part(k).iter().position(|&w| w != want) {
                return Err(format!("part {k} word {i}: {:#x}, not the copy of {want:#x}", part(k)[i]));
            }
        }
        Ok(format!(
            "{ROUNDS} rounds of GPU fills into and copies out of imported pages while {} protection changes switched them between read-only, no-access and read-write; contents right, no device loss; write marks: {cpu_marks} pages after the CPU's writes, {gpu_marks} after the GPU's; an import over pages already read-only: {ro}; over pages already no-access: {na}",
            toggles.load(Ordering::Relaxed)
        ))
    });
    unsafe {
        c.device.destroy_buffer(buf, None);
        c.device.free_memory(mem, None);
        VirtualFree(base as _, 0, MEM_RELEASE);
    }
    r
}

/// Status line.
/// The write tracking of the mixed range's view windows (`mem::views_state`):
/// its takes, reports, arming, faults and volatility.
pub fn view_census() -> String {
    let v: Vec<u64> = STATS.iter().map(|c| c.load(Ordering::Relaxed)).collect();
    let t: Vec<u64> = TAKE_STATS.iter().map(|c| c.load(Ordering::Relaxed)).collect();
    let h: Vec<u64> = HOST_STATS.iter().map(|c| c.load(Ordering::Relaxed)).collect();
    format!(
        " (pages visited {}, {} of the reports for pages left unarmed (volatile, or a file read writing them: once per epoch), {} pages armed in {} calls ({:.1} us each); write faults {} here, {} answered by the host ({} of them read from its ring); {} turned volatile, {} armed again after their stay, {} levels reset after a quiet stay)",
        t[3],
        t[5],
        t[2],
        v[S_PROTECTS],
        v[S_PROTECT_NS] as f64 / v[S_PROTECTS].max(1) as f64 / 1000.0,
        v[S_FAULTS],
        h[H_ANSWERED],
        v[S_HOST_FAULTS],
        v[S_VOLATILE],
        v[S_VOL_REARMED],
        v[S_VOL_DECAYED]
    )
}

/// Lightweight relay counters; no page walks or GPU synchronization.
#[cfg(target_os = "linux")]
pub fn dirty_counters() -> [u64; 2] {
    if crate::platform::write_watch::enabled() { return [crate::platform::write_watch::counters()[2], 0]; }
    [TAKE_STATS[1].load(Ordering::Relaxed), TAKE_STATS[5].load(Ordering::Relaxed)]
}

pub fn status() -> String {
    let v: Vec<u64> = STATS.iter().map(|c| c.load(Ordering::Relaxed)).collect();
    // Section-backed memory's write tracking (with residency too), then
    // residency's line.
    let mut out = String::new();
    if crate::sections::on() {
        let t: Vec<u64> = TAKE_STATS.iter().map(|c| c.load(Ordering::Relaxed)).collect();
        out = format!(
            "    write tracking by protection (sections): {} takes visited {} pages ({} region queries), reported {} ({} of them unarmable: once per epoch), armed {} ({} left for other protections); {} write faults ({} races){}, {} turned volatile ({} armed again after their stay), {} host class; {} arming calls ({:.1} us each), {} disarming calls ({:.1} us each); {} host writes ({} pages); {} remaps; {}; {:.0} MiB zeroed for new mappings (direct {:.0}); {} imports without an alias slot ({} alias views bridged); kernel-mode accesses retried {} ({} retries, {} failed for good); unmapped with GPU writes pending {} (direct {}), mapped again while pending {} ({} waited for)\n",
            t[0],
            t[3],
            t[4],
            t[1],
            t[5],
            t[2],
            v[S_OTHER_PROT],
            v[S_FAULTS],
            v[S_RACES],
            if via_host() {
                let h: Vec<u64> = HOST_STATS.iter().map(|c| c.load(Ordering::Relaxed)).collect();
                format!(
                    " handled here; armed through the host: {} pages the host answered (or disarmed) found by takes ({} chunk scans), {} joined (armed through the host by another), {} answered before their marks, {} faults on them that reached us, {} armings that failed",
                    h[H_ANSWERED], h[H_SCANS], h[H_JOINED], h[H_LOST], h[H_IN_PROCESS], h[H_FAILED]
                )
            } else {
                String::new()
            },
            v[S_VOLATILE],
            v[S_VOL_REARMED],
            v[S_HOSTED],
            v[S_PROTECTS],
            v[S_PROTECT_NS] as f64 / v[S_PROTECTS].max(1) as f64 / 1000.0,
            v[S_UNPROTECTS],
            v[S_UNPROTECT_NS] as f64 / v[S_UNPROTECTS].max(1) as f64 / 1000.0,
            v[S_HOST_WRITES],
            v[S_HOST_WRITE_PAGES],
            v[S_RESETS],
            crate::sections::split_line(),
            (crate::sections::zeroed().0 + crate::sections::zeroed().1) as f64 / (1u64 << 20) as f64,
            crate::sections::zeroed().0 as f64 / (1u64 << 20) as f64,
            super::exec::alias_misses(),
            crate::mem::BRIDGES.load(Ordering::Relaxed),
            crate::mem::KERNEL_IO[0].load(Ordering::Relaxed),
            crate::mem::KERNEL_IO[1].load(Ordering::Relaxed),
            crate::mem::KERNEL_IO[2].load(Ordering::Relaxed),
            crate::mem::PENDING_STATS[0].load(Ordering::Relaxed),
            crate::mem::PENDING_STATS[1].load(Ordering::Relaxed),
            crate::mem::PENDING_STATS[2].load(Ordering::Relaxed),
            crate::mem::PENDING_STATS[3].load(Ordering::Relaxed)
        );
    }
    if !wanted() {
        return out;
    }
    let refused: Vec<String> = REFUSE_NAMES.iter().zip(REFUSED.iter()).map(|(n, c)| format!("{n} {}", c.load(Ordering::Relaxed))).collect();
    out + &format!(
        "    residency ({}): {} bindings from the mirror ({} pages), {} pages uploaded, {} made stale by GPU writes; {} pages armed, {} write faults ({} races), {} volatile, {} host class; {} arming calls ({:.1} us each), {} disarming calls ({:.1} us each); {} host writes ({} pages); {} mirror pages backed, {} uploads, {} readbacks, {} remaps; refused: {}\n",
        if enabled() { "on" } else { "off" },
        v[S_BINDS],
        v[S_BIND_PAGES],
        v[S_CLAIMED],
        v[S_GPU_INVAL],
        v[S_ARMED],
        v[S_FAULTS],
        v[S_RACES],
        v[S_VOLATILE],
        v[S_HOSTED],
        v[S_PROTECTS],
        v[S_PROTECT_NS] as f64 / v[S_PROTECTS].max(1) as f64 / 1000.0,
        v[S_UNPROTECTS],
        v[S_UNPROTECT_NS] as f64 / v[S_UNPROTECTS].max(1) as f64 / 1000.0,
        v[S_HOST_WRITES],
        v[S_HOST_WRITE_PAGES],
        v[S_BACKED],
        v[S_UPLOADS],
        v[S_READBACKS],
        v[S_RESETS],
        refused.join(", ")
    )
}

// ---------------------------------------------------------------------------
// Mirror mode (`DS_RESIDENT=2`; RENDERER.md "Mirror mode"): GPU writes land in
// the mirror. Translated shaders mark every dword they store in the
// written-bit map (`MARKS`, `rdna::translate::Spec::write_marks`); at each
// batch's end the gather kernel (`rdna::kernels::mirror_gather`) records the
// pages with marked dwords and those dwords in host memory and clears their
// bits, and the batch's first completion action stores them to guest memory
// (`apply_masks_copy`), so the title sees them when it sees the batch done.
// Dwords no shader wrote keep what the CPU wrote (a binding's range can share
// pages with CPU data: allocator bitmaps, job stacks). All batches run on one
// Vulkan queue in submission order, so one map serves every queue.

static MODE2: AtomicBool = AtomicBool::new(false);

/// Mirror mode is on (`DS_RESIDENT=2` and its self-test passed).
pub fn mirror_mode() -> bool {
    MODE2.load(Ordering::Acquire)
}

/// Bytes of the written-bit map (`rdna::kernels::MARK_TOTAL_WORDS`: one bit
/// per dword of the mirror's span, one per page, the touched-page list).
pub const MARK_BYTES: u64 = 4 * rdna::kernels::MARK_TOTAL_WORDS as u64;
const _: () = assert!(MIRROR_END - MIRROR_BASE == rdna::kernels::MARK_SPAN_BYTES);
/// The written-bit map (video memory, zeroed by mirror mode's self-test),
/// and the compute family's queue's own (ordered async compute: batches
/// there mark and write back from it; one map for both queues would let one
/// queue's write-back clear marks and empty the touched-page list under the
/// other's stores).
static MARKS: std::sync::OnceLock<vk::Buffer> = std::sync::OnceLock::new();
static MARKS_C: std::sync::OnceLock<vk::Buffer> = std::sync::OnceLock::new();

/// The written-bit map (mirror mode).
pub fn marks() -> Option<vk::Buffer> {
    MARKS.get().copied()
}

/// The written-bit map of a batch on the compute family's queue (`on_c`)
/// or on the main queue.
pub fn marks_for(on_c: bool) -> Option<vk::Buffer> {
    if on_c { MARKS_C.get().copied() } else { MARKS.get().copied() }
}

/// Mirror mode's statistics (`owned_line`).
static OWN_STATS: [AtomicU64; 22] = [const { AtomicU64::new(0) }; 22];
/// Baselines: taken at CPU writes, copies diffed against one, dwords those
/// copies took.
const O_BASE_SNAPS: usize = 19;
const O_BASE_DIFFS: usize = 20;
const O_BASE_DWORDS: usize = 21;
/// Write-backs: batches, pages (records), dwords stored, protection changes
/// around the stores, pages left out (not writable guest memory any more),
/// nanoseconds; records a batch's output had no room for (gathered later).
const O_WB_BATCHES: usize = 0;
const O_WB_PAGES: usize = 1;
const O_WB_DWORDS: usize = 2;
const O_WB_PROTECTS: usize = 3;
const O_WB_SKIPPED: usize = 4;
const O_WB_NS: usize = 5;
const O_WB_OVER: usize = 6;
/// Pages written back that a later batch had uploaded into the mirror
/// meanwhile (made not device-valid: that upload may predate the dwords).
const O_STALE: usize = 7;
/// Pages copied into the mirror again at a batch's close after deferred CP
/// writes into them (`needs_copy`).
const O_CP_COPIES: usize = 8;
/// Lazy write-backs: pages made GPU-dirty, protection calls for it, faults
/// on dirty pages, runs read back, pages read back, dwords stored, read-back
/// nanoseconds, pages released by host writes and kernel I/O (explicit),
/// dirty pages dropped by remaps.
const O_DIRTIED: usize = 9;
const O_DIRTY_PROTECTS: usize = 10;
const O_DIRTY_FAULTS: usize = 11;
const O_RB_RUNS: usize = 12;
const O_RB_PAGES: usize = 13;
const O_RB_DWORDS: usize = 14;
const O_RB_NS: usize = 15;
const O_EXPLICIT: usize = 16;
const O_REMAP_DROPS: usize = 17;
/// Records of lazy gathers (pages made or kept dirty).
const O_MASK_RECORDS: usize = 18;

/// Lazy write-backs (`DS_MM_LAZY=1`): pages with marked dwords stay
/// no-access until the CPU side touches them (`apply_masks`,
/// `release_dirty`); by default they are copied from the mirror at their
/// batch's completion (`apply_masks_copy`; the image cache reads nearly
/// every page GPU work writes over textures, FINDINGS.md 2026-10-21
/// morning).
pub fn lazy_writebacks() -> bool {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ON.get_or_init(|| std::env::var("DS_MM_LAZY").ok().as_deref() == Some("1"))
}

/// The copy context of eager write-backs by copy (the completion thread's):
/// command buffer, fence and a host staging buffer that grows.
struct EagerCtx {
    _pool: vk::CommandPool,
    cmd: vk::CommandBuffer,
    fence: vk::Fence,
    copy: bool,
    stage: (vk::Buffer, vk::DeviceMemory, *mut u8, u64),
}
unsafe impl Send for EagerCtx {}
static EAGER: Mutex<Option<EagerCtx>> = Mutex::new(None);

fn eager_ctx() -> Result<EagerCtx, String> {
    let c = ctx();
    let (family, copy) = match &c.copy {
        Some(q) => (q.family, true),
        None => (c.queue_family, false),
    };
    unsafe {
        let pool = c.device.create_command_pool(&vk::CommandPoolCreateInfo::default().queue_family_index(family).flags(vk::CommandPoolCreateFlags::RESET_COMMAND_BUFFER), None).map_err(|e| format!("pool: {e}"))?;
        let cmd = c.device.allocate_command_buffers(&vk::CommandBufferAllocateInfo::default().command_pool(pool).command_buffer_count(1)).map_err(|e| format!("cmd: {e}"))?[0];
        let fence = c.device.create_fence(&vk::FenceCreateInfo::default(), None).map_err(|e| format!("fence: {e}"))?;
        let (b, m, ptr) = c.host_buffer(16 << 20, vk::BufferUsageFlags::TRANSFER_DST)?;
        Ok(EagerCtx { _pool: pool, cmd, fence, copy, stage: (b, m, ptr, 16 << 20) })
    }
}

/// Eager write-back by copy (the first completion action of the batch whose
/// effects this thread performs): `out` holds the gather kernel's mask
/// records (at most `cap`); the pages are copied from the mirror into host
/// memory on the copy queue (one submission, runs of pages) and their marked
/// dwords stored to guest memory, run by run: runs no longer mapped are left
/// out (the guest mapping table, consistent under the arming lock); pages
/// made writable around the stores get their protection back (armed: read-
/// only; others as the guest mapped them). A page a later batch uploaded
/// into the mirror meanwhile is made not device-valid (its upload may
/// predate these dwords in guest memory).
/// Returns the records the output had no room for.
pub fn apply_masks_copy(out: *const u32, cap: u32) -> u32 {
    use rdna::kernels::{GATHER_HEADER_DWORDS, GATHER_MASK_RECORD_DWORDS};
    use crate::platform::os::Win32::System::Memory::{VirtualProtect, PAGE_READONLY, PAGE_READWRITE};
    let t = std::time::Instant::now();
    let seq = super::cp::after_batch();
    let count = unsafe { std::ptr::read_volatile(out) };
    let n = count.min(cap) as usize;
    if n == 0 {
        return count.saturating_sub(cap);
    }
    let rec = |k: usize| unsafe { out.add(GATHER_HEADER_DWORDS as usize + k * GATHER_MASK_RECORD_DWORDS as usize) };
    let mut order: Vec<(u32, usize)> = (0..n).map(|k| (unsafe { *rec(k) }, k)).collect();
    order.sort_unstable();
    order.dedup_by_key(|x| x.0);
    note_shared(&order);
    // Runs of consecutive pages: (first index in `order`, count).
    let mut runs: Vec<(usize, usize)> = Vec::new();
    for i in 0..order.len() {
        match runs.last_mut() {
            Some((a, k)) if order[*a + *k - 1].0 + 1 == order[i].0 => *k += 1,
            _ => runs.push((i, 1)),
        }
    }
    let Some(m) = MIRROR.get() else { return 0 };
    let mut g = EAGER.lock();
    if g.is_none() {
        match eager_ctx() {
            Ok(x) => *g = Some(x),
            Err(e) => {
                crate::error!("resident: no eager write-back context: {e}");
                return 0;
            }
        }
    }
    let x = g.as_mut().expect("eager context");
    let c = ctx();
    let bytes = order.len() as u64 * PAGE;
    if x.stage.3 < bytes {
        unsafe {
            c.device.destroy_buffer(x.stage.0, None);
            c.device.free_memory(x.stage.1, None);
        }
        let capb = bytes.next_power_of_two();
        match c.host_buffer(capb, vk::BufferUsageFlags::TRANSFER_DST) {
            Ok((b, mm, ptr)) => x.stage = (b, mm, ptr, capb),
            Err(e) => {
                crate::error!("resident: no staging for an eager write-back of {} pages: {e}", order.len());
                x.stage = (vk::Buffer::null(), vk::DeviceMemory::null(), std::ptr::null_mut(), 0);
                return 0;
            }
        }
    }
    let regions: Vec<vk::BufferCopy> = runs.iter().map(|&(a, k)| vk::BufferCopy { src_offset: order[a].0 as u64 * PAGE, dst_offset: a as u64 * PAGE, size: k as u64 * PAGE }).collect();
    let copied = unsafe {
        (|| -> Result<(), String> {
            c.device.reset_command_buffer(x.cmd, vk::CommandBufferResetFlags::empty()).map_err(|e| format!("reset: {e}"))?;
            c.device.begin_command_buffer(x.cmd, &vk::CommandBufferBeginInfo::default().flags(vk::CommandBufferUsageFlags::ONE_TIME_SUBMIT)).map_err(|e| format!("begin: {e}"))?;
            c.device.cmd_copy_buffer(x.cmd, m.buf, x.stage.0, &regions);
            c.device.end_command_buffer(x.cmd).map_err(|e| format!("end: {e}"))?;
            c.device.reset_fences(&[x.fence]).map_err(|e| format!("fence reset: {e}"))?;
            c.submit_timeline(x.copy, &[x.cmd], &[], &[], x.fence)?;
            c.device.wait_for_fences(&[x.fence], true, 10_000_000_000).map_err(|e| format!("wait: {e}"))
        })()
    };
    if let Err(err) = copied {
        crate::error!("resident: eager write-back copy of {} pages: {err}", order.len());
        return 0;
    }
    let stage = x.stage.2 as *const u8;
    let (mut n_dw, mut n_pg, mut n_prot, mut n_skip, mut n_stale) = (0u64, 0u64, 0u64, 0u64, 0u64);
    // The memory manager's lock order: its table, guest readers, arming (it
    // holds the first across mapping changes, which take the others).
    let st = crate::mem::MEM.lock();
    let _readers = crate::mem::GUEST_READERS.read();
    let _arming = ARM_LOCK.read();
    for &(ra, rk) in &runs {
        let a = MIRROR_BASE + order[ra].0 as u64 * PAGE;
        let e = a + rk as u64 * PAGE;
        // Mapped still, and how the guest mapped it.
        let guest_prot = {
            let mut at = a;
            let mut prot = None;
            while at < e {
                match st.find(at) {
                    Some(v) if v.kind != crate::mem::VmaKind::Reserved && v.prot != 0 => {
                        let wp = crate::mem::win_prot(v.prot);
                        prot = Some(match prot {
                            None => wp,
                            Some(p) if p == wp => p,
                            Some(_) => PAGE_READONLY,
                        });
                        at = v.start + v.size;
                    }
                    _ => {
                        prot = None;
                        break;
                    }
                }
            }
            prot
        };
        let Some(guest_prot) = guest_prot else {
            n_skip += rk as u64;
            continue;
        };
        busy_range(a, e);
        let plain = guest_prot == PAGE_READWRITE && {
            let mut all_rw = true;
            let mut q = a;
            while q < e {
                if chunk(q, false).is_some_and(|c| c.pages[idx(q)].load(Ordering::Acquire) & (ST_ARMED | ST_GUEST_RO) != 0) {
                    all_rw = false;
                    break;
                }
                q += PAGE;
            }
            all_rw
        };
        let mut old = 0u32;
        if !plain {
            unsafe { VirtualProtect(a as _, (e - a) as usize, PAGE_READWRITE, &mut old) };
            n_prot += 1;
        }
        for j in 0..rk {
            let r = rec(order[ra + j].1);
            let masks = unsafe { r.add(1) };
            let src = unsafe { stage.add((ra + j) * PAGE as usize) } as *const u32;
            let pg = a + j as u64 * PAGE;
            baseline_merge(pg, unsafe { std::slice::from_raw_parts(masks, 32) }, src);
            note_writeback(pg, seq, unsafe { std::slice::from_raw_parts(masks, 32) });
            for w in 0..32usize {
                let mut mm = unsafe { *masks.add(w) };
                while mm != 0 {
                    let d = w * 32 + mm.trailing_zeros() as usize;
                    unsafe { std::ptr::write_volatile((pg + 4 * d as u64) as *mut u32, *src.add(d)) };
                    n_dw += 1;
                    mm &= mm - 1;
                }
            }
            n_pg += 1;
        }
        if !plain {
            // Back: armed pages read-only, others as the guest mapped them
            // (runs of pages with the same protection).
            let target = |q: u64| {
                let armed = chunk(q, false).is_some_and(|c| c.pages[idx(q)].load(Ordering::Acquire) & ST_ARMED != 0);
                if armed { PAGE_READONLY } else { guest_prot }
            };
            let mut q = a;
            while q < e {
                let pt = target(q);
                let mut q2 = q + PAGE;
                while q2 < e && target(q2) == pt {
                    q2 += PAGE;
                }
                let mut tmp = 0u32;
                unsafe { VirtualProtect(q as _, (q2 - q) as usize, pt, &mut tmp) };
                q = q2;
            }
        }
        // Uploaded by a later batch (or one recording): its copy may lack
        // these dwords.
        let mut q = a;
        while q < e {
            if let Some(c) = chunk(q, false) {
                let k = idx(q);
                let up = c.seqs[k].load(Ordering::Acquire);
                if up == PENDING || (up > seq && seq != 0) {
                    c.pages[k].fetch_and(!ST_VALID, Ordering::AcqRel);
                    flag(c, k);
                    n_stale += 1;
                }
            }
            q += PAGE;
        }
        unbusy_range(a, e);
    }
    OWN_STATS[O_WB_BATCHES].fetch_add(1, Ordering::Relaxed);
    OWN_STATS[O_WB_PAGES].fetch_add(n_pg, Ordering::Relaxed);
    OWN_STATS[O_WB_DWORDS].fetch_add(n_dw, Ordering::Relaxed);
    OWN_STATS[O_WB_PROTECTS].fetch_add(n_prot, Ordering::Relaxed);
    OWN_STATS[O_WB_SKIPPED].fetch_add(n_skip, Ordering::Relaxed);
    OWN_STATS[O_WB_NS].fetch_add(t.elapsed().as_nanos() as u64, Ordering::Relaxed);
    OWN_STATS[O_STALE].fetch_add(n_stale, Ordering::Relaxed);
    count.saturating_sub(cap)
}

/// A GPU-dirty page: the mask words of the dwords guest memory lacks and
/// the protection it had before it was made no-access.
struct DirtyPage {
    masks: [u32; 32],
    prot: u32,
}

/// GPU-dirty pages (`Chunk::gdirty`), by address.
static DIRTY: Mutex<Option<std::collections::HashMap<u64, DirtyPage>>> = Mutex::new(None);

#[inline]
fn is_gprot(c: &Chunk, i: usize) -> bool {
    c.gprot[i >> 6].load(Ordering::Acquire) & (1u64 << (i & 63)) != 0
}

#[inline]
fn set_gprot(c: &Chunk, i: usize) {
    c.gprot[i >> 6].fetch_or(1u64 << (i & 63), Ordering::AcqRel);
}

#[inline]
fn clear_gprot(c: &Chunk, i: usize) {
    if is_gprot(c, i) {
        c.gprot[i >> 6].fetch_and(!(1u64 << (i & 63)), Ordering::AcqRel);
    }
}

/// Lazy mirror mode: one batch's written ranges, no-access from before its
/// submission (`own_begin`) until a CPU access settles each page (`settle`),
/// so no CPU access can fall between the GPU's writes and their masks: the
/// title may see a batch's label (written by the GPU in place) before our
/// completion thread notes its masks, and memory it reused meanwhile then
/// took stale GPU dwords at the release (the allocator crash at
/// eboot+0xad92d6, FINDINGS.md 2026-10-21 and 2026-10-23).
pub struct Owned {
    /// The batch's sequence (0 until `own_submitted`).
    seq: AtomicU64,
    ranges: Vec<(u64, u64)>,
}

/// Batches in flight that own ranges (removed by `own_done` after their
/// masks are noted).
static INFLIGHT_OWNED: Mutex<Vec<std::sync::Arc<Owned>>> = Mutex::new(Vec::new());

/// Ownership statistics (`owned_line`): batches registered, pages made
/// no-access at submission, protection calls for it, settles by faults,
/// settles by explicit releases, settles that waited for a batch, masks
/// dropped for pages remapped after their batch's submission, settles that
/// gave up waiting.
static OWN_PROT: [AtomicU64; 9] = [const { AtomicU64::new(0) }; 9];
const O7_BATCHES: usize = 0;
const O7_PAGES: usize = 1;
const O7_PROTECTS: usize = 2;
const O7_FAULTS: usize = 3;
const O7_EXPLICIT: usize = 4;
const O7_WAITED: usize = 5;
const O7_REMAPPED: usize = 6;
const O7_TIMEOUTS: usize = 7;
const O7_UNPROTECTED: usize = 8;

/// Lazy mirror mode, before a batch with written ranges is submitted: the
/// pages of `ranges` not yet owned become no-access (committed regions
/// only; host-class pages never) and the batch is registered. None outside
/// lazy mirror mode.
pub fn own_begin(ranges: &[(u64, u64)]) -> Option<std::sync::Arc<Owned>> {
    use crate::platform::os::Win32::System::Memory::{VirtualQuery, MEMORY_BASIC_INFORMATION, MEM_COMMIT, PAGE_NOACCESS};
    if !mirror_mode() || !lazy_writebacks() || ranges.is_empty() {
        return None;
    }
    let _arming = ARM_LOCK.read();
    let mut runs: Vec<(u64, u64)> = Vec::new();
    for &(a, e) in ranges {
        let (lo, hi) = ((a & !(PAGE - 1)).max(MIRROR_BASE), e.next_multiple_of(PAGE).min(MIRROR_END));
        let mut p = lo;
        while p < hi {
            let Some(c) = chunk(p, true) else {
                p += PAGE;
                continue;
            };
            let i = idx(p);
            if i & 63 == 0 && p + 64 * PAGE <= hi && c.gprot[i >> 6].load(Ordering::Acquire) == u64::MAX {
                p += 64 * PAGE;
                continue;
            }
            if !is_gprot(c, i) && c.pages[i].load(Ordering::Acquire) & ST_HOST == 0 {
                match runs.last_mut() {
                    Some(r) if r.1 == p => r.1 = p + PAGE,
                    _ => runs.push((p, p + PAGE)),
                }
            }
            p += PAGE;
        }
    }
    // Registered first: an access that faults from here on finds its
    // owner (and waits for the batch's sequence).
    let o = std::sync::Arc::new(Owned { seq: AtomicU64::new(0), ranges: ranges.to_vec() });
    INFLIGHT_OWNED.lock().push(o.clone());
    OWN_PROT[O7_BATCHES].fetch_add(1, Ordering::Relaxed);
    let mark = |a: u64, e: u64, on: bool| {
        let mut q = a;
        while q < e {
            if let Some(c) = chunk(q, false) {
                if on {
                    set_gprot(c, idx(q));
                } else {
                    clear_gprot(c, idx(q));
                }
            }
            q += PAGE;
        }
    };
    for &(a, e) in &runs {
        busy_range(a, e);
        // Region by region: committed ones only (a range past its mapping
        // keeps the rest as it is). The bits go first, as `apply_masks`
        // does: an access faulting while the protection changes is ours.
        let mut p = a;
        while p < e {
            let mut info: MEMORY_BASIC_INFORMATION = unsafe { std::mem::zeroed() };
            if unsafe { VirtualQuery(p as _, &mut info, std::mem::size_of::<MEMORY_BASIC_INFORMATION>()) } == 0 {
                break;
            }
            let end = (info.BaseAddress as u64 + info.RegionSize as u64).min(e).max(p + PAGE);
            if info.State == MEM_COMMIT {
                mark(p, end, true);
                if protect(p, end - p, PAGE_NOACCESS) {
                    OWN_PROT[O7_PROTECTS].fetch_add(1, Ordering::Relaxed);
                    OWN_PROT[O7_PAGES].fetch_add((end - p) / PAGE, Ordering::Relaxed);
                } else {
                    mark(p, end, false);
                }
            }
            p = end;
        }
        unbusy_range(a, e);
    }
    Some(o)
}

/// The batch `o` was registered for was handed over with sequence `seq`.
pub fn own_submitted(o: &Owned, seq: u64) {
    o.seq.store(seq, Ordering::Release);
}

/// The batch's masks are noted (its completion's first action): it no
/// longer owns anything in flight.
pub fn own_done(o: &std::sync::Arc<Owned>) {
    INFLIGHT_OWNED.lock().retain(|x| !std::sync::Arc::ptr_eq(x, o));
}

/// A CPU access (`fault`) or an explicit release reached a page owned for
/// GPU writes: the batches in flight that own it complete first (their
/// masks noted; on the completion thread only those up to the batch it is
/// completing, which it cannot wait past), then its GPU-written dwords are
/// stored to guest memory and its protection restored. True when the
/// access may be retried.
fn settle(page: u64, fault: bool) -> bool {
    use crate::platform::os::Win32::System::Memory::{PAGE_READONLY, PAGE_READWRITE};
    let Some(c) = chunk(page, false) else { return false };
    let i = idx(page);
    if !is_gprot(c, i) {
        return true;
    }
    OWN_PROT[if fault { O7_FAULTS } else { O7_EXPLICIT }].fetch_add(1, Ordering::Relaxed);
    let limit = if crate::log::thread_name() == "gpu-complete" { super::cp::after_batch() } else { u64::MAX };
    let t0 = std::time::Instant::now();
    loop {
        let (max_seq, unknown) = {
            let g = INFLIGHT_OWNED.lock();
            g.iter().filter(|o| o.ranges.iter().any(|&(a, e)| page + PAGE > a && page < e)).fold((0u64, false), |(m, u), o| {
                let s = o.seq.load(Ordering::Acquire);
                if s == 0 {
                    (m, true)
                } else if s <= limit {
                    (m.max(s), u)
                } else {
                    (m, u)
                }
            })
        };
        if unknown && limit == u64::MAX {
            if t0.elapsed() > std::time::Duration::from_secs(10) {
                OWN_PROT[O7_TIMEOUTS].fetch_add(1, Ordering::Relaxed);
                crate::error!("resident: page {page:#x} owned by a batch never handed over (10 s)");
                break;
            }
            std::thread::yield_now();
            continue;
        }
        if max_seq > 0 && super::exec::completed_seq() < max_seq {
            OWN_PROT[O7_WAITED].fetch_add(1, Ordering::Relaxed);
            if !super::exec::wait_done_for(max_seq, std::time::Duration::from_secs(30)) {
                OWN_PROT[O7_TIMEOUTS].fetch_add(1, Ordering::Relaxed);
                crate::error!("resident: page {page:#x}: its owner batch {max_seq} did not complete in 30 s");
            }
        }
        break;
    }
    if is_dirty(c, i) {
        return release_dirty(page, fault);
    }
    // Owned but not written by GPU work: its protection back (after a
    // protection change in progress: `own_begin` holds the page busy).
    let _arming = ARM_LOCK.read();
    busy_range(page, page + PAGE);
    if !is_gprot(c, i) {
        unbusy_range(page, page + PAGE);
        return true;
    }
    let st = c.pages[i].load(Ordering::Acquire);
    let prot = if st & (ST_ARMED | ST_GUEST_RO) != 0 { PAGE_READONLY } else { PAGE_READWRITE };
    protect(page, PAGE, prot);
    clear_gprot(c, i);
    OWN_PROT[O7_UNPROTECTED].fetch_add(1, Ordering::Relaxed);
    unbusy_range(page, page + PAGE);
    true
}

#[inline]
fn is_dirty(c: &Chunk, i: usize) -> bool {
    c.gdirty[i >> 6].load(Ordering::Acquire) & (1u64 << (i & 63)) != 0
}

#[inline]
fn undirty(c: &Chunk, i: usize) {
    c.gdirty[i >> 6].fetch_and(!(1u64 << (i & 63)), Ordering::AcqRel);
}

/// Mirror mode's lazy write-back (the first completion action of the batch
/// whose effects this thread performs): `out` holds the gather kernel's
/// mask records (at most `cap`): the pages get the dwords' marks in `DIRTY`
/// and are made no-access, runs at once (a page dirty already keeps its
/// protection from before); the CPU side's first access reads them back
/// (`release_dirty`). Returns the records the output had no room for.
pub fn apply_masks(out: *const u32, cap: u32) -> u32 {
    use rdna::kernels::{GATHER_HEADER_DWORDS, GATHER_MASK_RECORD_DWORDS};
    use crate::platform::os::Win32::System::Memory::{PAGE_READONLY, PAGE_READWRITE};
    let count = unsafe { std::ptr::read_volatile(out) };
    let n = count.min(cap) as usize;
    let rec = |k: usize| unsafe { out.add(GATHER_HEADER_DWORDS as usize + k * GATHER_MASK_RECORD_DWORDS as usize) };
    let mut order: Vec<(u32, usize)> = (0..n).map(|k| (unsafe { *rec(k) }, k)).collect();
    order.sort_unstable();
    OWN_STATS[O_MASK_RECORDS].fetch_add(n as u64, Ordering::Relaxed);
    note_shared(&order);
    // The batch whose completion this is (its masks of pages remapped since
    // it was submitted belong to the old mapping).
    let seq = super::cp::after_batch();
    let _arming = ARM_LOCK.read();
    let mut g = DIRTY.lock();
    let map = g.get_or_insert_with(Default::default);
    // Pages newly dirty, in runs, protected after their marks are noted
    // (pages owned since the batch's submission already are: `own_begin`).
    let mut fresh: Vec<u64> = Vec::new();
    for &(page, k) in &order {
        let p = MIRROR_BASE + page as u64 * PAGE;
        let Some(c) = chunk(p, false) else { continue };
        let i = idx(p);
        if seq != 0 && c.rseq[i].load(Ordering::Acquire) > seq {
            OWN_PROT[O7_REMAPPED].fetch_add(1, Ordering::Relaxed);
            continue;
        }
        let masks = unsafe { std::slice::from_raw_parts(rec(k).add(1), 32) };
        if traced(p) {
            trace_event(p, format_args!("batch {seq} GPU wrote dwords {}", mask_runs(masks)));
        }
        match map.get_mut(&p) {
            Some(d) => {
                for w in 0..32 {
                    d.masks[w] |= masks[w];
                }
            }
            None => {
                // The page's protection now (restored at its release), from
                // its state (queries cost the more the larger their region).
                let st = c.pages[i].load(Ordering::Acquire);
                let prot = if st & (ST_ARMED | ST_GUEST_RO) != 0 { PAGE_READONLY } else { PAGE_READWRITE };
                let mut m = [0u32; 32];
                m.copy_from_slice(masks);
                map.insert(p, DirtyPage { masks: m, prot });
                c.gdirty[i >> 6].fetch_or(1u64 << (i & 63), Ordering::AcqRel);
                if !is_gprot(c, i) {
                    fresh.push(p);
                }
            }
        }
    }
    drop(g);
    OWN_STATS[O_DIRTIED].fetch_add(fresh.len() as u64, Ordering::Relaxed);
    // No-access, run by run (busy meanwhile: a fault on one waits).
    let mut i = 0;
    while i < fresh.len() {
        let mut j = i + 1;
        while j < fresh.len() && fresh[j] == fresh[j - 1] + PAGE {
            j += 1;
        }
        let (a, e) = (fresh[i], fresh[j - 1] + PAGE);
        busy_range(a, e);
        if protect_regions(a, e, crate::platform::os::Win32::System::Memory::PAGE_NOACCESS, false) {
            let mut q = a;
            while q < e {
                if let Some(c) = chunk(q, false) {
                    set_gprot(c, idx(q));
                }
                q += PAGE;
            }
        }
        OWN_STATS[O_DIRTY_PROTECTS].fetch_add(1, Ordering::Relaxed);
        unbusy_range(a, e);
        i = j;
    }
    count.saturating_sub(cap)
}

/// Mirror mode: the baselines of pages the CPU writes: the page's dwords as
/// the mirror holds them from the CPU's side (its guest bytes at their last
/// copy into the mirror, GPU-written dwords merged since). A copy into the
/// mirror takes only the dwords that differ from the baseline, so dwords
/// GPU work wrote there meanwhile (a batch still running, its write-back
/// not stored yet) stay. Taken at an armed page's first CPU write
/// (`on_fault`), kept for pages copied without a claim (volatile) at their
/// copies; dropped when a claim arms the page again, at remaps.
static BASE: Mutex<Option<std::collections::HashMap<u64, Box<[u32; 1024]>>>> = Mutex::new(None);
static BASE_N: AtomicU64 = AtomicU64::new(0);

/// Takes page `page`'s baseline from its guest bytes unless it has one.
fn baseline_snap(page: u64) {
    let mut b = Box::new([0u32; 1024]);
    unsafe { std::ptr::copy_nonoverlapping(page as *const u32, b.as_mut_ptr(), 1024) };
    let mut g = BASE.lock();
    let m = g.get_or_insert_with(Default::default);
    if !m.contains_key(&page) {
        m.insert(page, b);
        BASE_N.store(m.len() as u64, Ordering::Release);
        OWN_STATS[O_BASE_SNAPS].fetch_add(1, Ordering::Relaxed);
    }
}

/// The dword runs (first dword, dwords) of page `page`, whose current bytes
/// are at `now`, that differ from its baseline (None: no baseline, the whole
/// page). `keep`: the baseline becomes `now` (pages copied without a claim);
/// otherwise it is dropped (a claim armed the page: its next CPU write takes
/// one again).
pub fn baseline_diff(page: u64, now: *const u32, keep: bool) -> Option<Vec<(u32, u32)>> {
    if BASE_N.load(Ordering::Acquire) == 0 && !keep {
        return None;
    }
    // Read before the lock: the page may be GPU-dirty (lazy write-backs),
    // and its fault's release takes the lock too (`baseline_merge`).
    let mut copy = [0u32; 1024];
    unsafe { std::ptr::copy_nonoverlapping(now, copy.as_mut_ptr(), 1024) };
    let mut g = BASE.lock();
    let m = g.get_or_insert_with(Default::default);
    let now = &copy[..];
    let runs = match m.get_mut(&page) {
        Some(b) => {
            let mut runs: Vec<(u32, u32)> = Vec::new();
            for (d, (&x, y)) in now.iter().zip(b.iter_mut()).enumerate() {
                if x != *y {
                    match runs.last_mut() {
                        Some((s, n)) if *s + *n == d as u32 => *n += 1,
                        _ => runs.push((d as u32, 1)),
                    }
                    if keep {
                        *y = x;
                    }
                }
            }
            OWN_STATS[O_BASE_DIFFS].fetch_add(1, Ordering::Relaxed);
            OWN_STATS[O_BASE_DWORDS].fetch_add(runs.iter().map(|r| r.1 as u64).sum(), Ordering::Relaxed);
            if !keep {
                m.remove(&page);
            }
            Some(runs)
        }
        None => {
            if keep {
                let mut b = Box::new([0u32; 1024]);
                b.copy_from_slice(now);
                m.insert(page, b);
            }
            None
        }
    };
    BASE_N.store(m.len() as u64, Ordering::Release);
    runs
}

/// A write-back stored GPU-written dwords of page `page` (masks: bit k of
/// word j for dword 32j+k, values at `src`): its baseline takes them too
/// (they are the mirror's, not CPU writes).
fn baseline_merge(page: u64, masks: &[u32], src: *const u32) {
    if BASE_N.load(Ordering::Acquire) == 0 {
        return;
    }
    let mut g = BASE.lock();
    let Some(b) = g.as_mut().and_then(|m| m.get_mut(&page)) else { return };
    for (w, &mw) in masks.iter().enumerate() {
        let mut mm = mw;
        while mm != 0 {
            let d = w * 32 + mm.trailing_zeros() as usize;
            b[d] = unsafe { *src.add(d) };
            mm &= mm - 1;
        }
    }
}

/// Drops the baselines of [lo, hi) (remaps).
fn baseline_drop(lo: u64, hi: u64) {
    if BASE_N.load(Ordering::Acquire) == 0 {
        return;
    }
    let mut g = BASE.lock();
    if let Some(m) = g.as_mut() {
        let mut p = lo;
        while p < hi {
            m.remove(&p);
            p += PAGE;
        }
        BASE_N.store(m.len() as u64, Ordering::Release);
    }
}

/// Diagnostics: mirror mode's write-backs by page (the last eight: time in
/// ms, batch, dwords, first dword), for fault reports (`writeback_history`).
static WB_LOG: Mutex<Option<std::collections::HashMap<u64, std::collections::VecDeque<(u64, u64, u32, u32)>>>> = Mutex::new(None);

fn note_writeback(page: u64, seq: u64, masks: &[u32]) {
    let n: u32 = masks.iter().map(|m| m.count_ones()).sum();
    let first = masks.iter().enumerate().find(|&(_, &m)| m != 0).map(|(w, m)| w as u32 * 32 + m.trailing_zeros()).unwrap_or(0);
    let t = crate::log::elapsed_ms() as u64;
    let mut g = WB_LOG.lock();
    let m = g.get_or_insert_with(Default::default);
    if m.len() > 1 << 20 {
        m.clear();
    }
    let e = m.entry(page).or_default();
    if e.len() >= 8 {
        e.pop_front();
    }
    e.push_back((t, seq, n, first));
}

/// The write-backs into the page holding `addr` (fault reports).
pub fn writeback_history(addr: u64) -> Option<String> {
    let g = WB_LOG.try_lock()?;
    let e = g.as_ref()?.get(&(addr & !(PAGE - 1)))?;
    Some(e.iter().map(|(t, s, n, f)| format!("at {t} ms batch {s}: {n} dwords from dword {f}")).collect::<Vec<_>>().join("; "))
}

/// Diagnostics: pages GPU work wrote (gather records) that the CPU wrote
/// too since their last upload (not device-valid, or volatile): the first
/// 64 logged.
fn note_shared(order: &[(u32, usize)]) {
    static LOGGED: AtomicU64 = AtomicU64::new(0);
    if LOGGED.load(Ordering::Relaxed) >= 64 {
        return;
    }
    for &(page, _) in order {
        let p = MIRROR_BASE + page as u64 * PAGE;
        let Some(c) = chunk(p, false) else { continue };
        let s = c.pages[idx(p)].load(Ordering::Acquire);
        if s & ST_VALID == 0 || s & ST_VOLATILE != 0 {
            let n = LOGGED.fetch_add(1, Ordering::Relaxed);
            if n >= 64 {
                return;
            }
            crate::info!("resident: page {p:#x} written by GPU work and by the CPU since its upload (state {s:#x})");
        }
    }
}

/// Takes [a, e)'s pages busy (spinning while another thread has one).
fn busy_range(a: u64, e: u64) {
    let mut q = a;
    while q < e {
        if let Some(c) = chunk(q, false) {
            let st = &c.pages[idx(q)];
            loop {
                let cur = st.load(Ordering::Acquire);
                if cur & ST_BUSY != 0 {
                    std::hint::spin_loop();
                    continue;
                }
                if st.compare_exchange(cur, cur | ST_BUSY, Ordering::AcqRel, Ordering::Acquire).is_ok() {
                    break;
                }
            }
        }
        q += PAGE;
    }
}

fn unbusy_range(a: u64, e: u64) {
    let mut q = a;
    while q < e {
        if let Some(c) = chunk(q, false) {
            c.pages[idx(q)].fetch_and(!ST_BUSY, Ordering::AcqRel);
        }
        q += PAGE;
    }
}

/// Pages read back per run at most.
const READBACK_MAX: u64 = 1 << 20;

/// Releases the GPU-dirty run around `page` (at most READBACK_MAX, within
/// its chunk): the mirror's pages are copied into host memory on the copy
/// queue, their marked dwords stored to guest memory, and the pages get
/// their protection back. `fault`: for an access that faulted (statistics).
/// True when the access may be retried.
fn release_dirty(page: u64, fault: bool) -> bool {
    use crate::platform::os::Win32::System::Memory::{VirtualProtect, PAGE_READWRITE};
    let t = std::time::Instant::now();
    let Some(c) = chunk(page, false) else { return false };
    let mut g = READBACK.lock();
    if !is_dirty(c, idx(page)) {
        // Released by another thread meanwhile.
        return true;
    }
    if g.is_none() {
        match readback_ctx() {
            Ok(r) => *g = Some(r),
            Err(e) => {
                crate::error!("resident: no read-back context: {e}");
                return false;
            }
        }
    }
    let rb = g.as_ref().expect("read-back context");
    let base = page & !(READBACK_MAX - 1);
    let chunk_lo = page & !((1u64 << CHUNK_SHIFT) - 1);
    let lo_bound = base.max(chunk_lo);
    let hi_bound = (base + READBACK_MAX).min(chunk_lo + (1u64 << CHUNK_SHIFT));
    let (mut a, mut e) = (page, page + PAGE);
    while a > lo_bound && is_dirty(c, idx(a - PAGE)) {
        a -= PAGE;
    }
    while e < hi_bound && is_dirty(c, idx(e)) {
        e += PAGE;
    }
    let Some(m) = MIRROR.get() else { return false };
    let cx = ctx();
    let n = e - a;
    let copied = unsafe {
        (|| -> Result<(), String> {
            cx.device.reset_command_buffer(rb.cmd, vk::CommandBufferResetFlags::empty()).map_err(|e| format!("reset: {e}"))?;
            cx.device.begin_command_buffer(rb.cmd, &vk::CommandBufferBeginInfo::default().flags(vk::CommandBufferUsageFlags::ONE_TIME_SUBMIT)).map_err(|e| format!("begin: {e}"))?;
            cx.device.cmd_copy_buffer(rb.cmd, m.buf, rb.buf, &[vk::BufferCopy { src_offset: a - MIRROR_BASE, dst_offset: 0, size: n }]);
            cx.device.end_command_buffer(rb.cmd).map_err(|e| format!("end: {e}"))?;
            cx.device.reset_fences(&[rb.fence]).map_err(|e| format!("fence reset: {e}"))?;
            cx.submit_timeline(rb.copy, &[rb.cmd], &[], &[], rb.fence)?;
            cx.device.wait_for_fences(&[rb.fence], true, 10_000_000_000).map_err(|e| format!("wait: {e}"))
        })()
    };
    if let Err(err) = copied {
        crate::error!("resident: reading back {a:#x}..{e:#x}: {err}");
        return false;
    }
    let _arming = ARM_LOCK.read();
    busy_range(a, e);
    let mut dg = DIRTY.lock();
    let map = dg.get_or_insert_with(Default::default);
    let mut dwords = 0u64;
    let mut q = a;
    while q < e {
        let i = idx(q);
        if let Some(d) = map.remove(&q) {
            let mut old = 0u32;
            unsafe { VirtualProtect(q as _, PAGE as usize, PAGE_READWRITE, &mut old) };
            let src = unsafe { rb.ptr.add((q - a) as usize) } as *const u32;
            if traced(q) {
                let rip = FAULT_RIP.with(|r| r.get());
                let at = if !fault { "explicit".to_string() } else if rip >= crate::mem::GUEST_BASE && rip < crate::mem::GUEST_END { crate::exception::describe_guest(rip) } else { crate::exception::host_symbol(rip) };
                trace_event(q, format_args!("released dwords {} (access at {at}); first dwords now {:08x} {:08x}, mirror {:08x} {:08x}", mask_runs(&d.masks), unsafe { *(q as *const u32) }, unsafe { *((q + 4) as *const u32) }, unsafe { *src }, unsafe { *src.add(1) }));
            }
            baseline_merge(q, &d.masks, src);
            note_writeback(q, 0, &d.masks);
            for (w, &mw) in d.masks.iter().enumerate() {
                let mut mm = mw;
                while mm != 0 {
                    let k = w * 32 + mm.trailing_zeros() as usize;
                    unsafe { std::ptr::write_volatile((q + 4 * k as u64) as *mut u32, *src.add(k)) };
                    dwords += 1;
                    mm &= mm - 1;
                }
            }
            let mut tmp = 0u32;
            unsafe { VirtualProtect(q as _, PAGE as usize, d.prot, &mut tmp) };
        }
        undirty(c, i);
        clear_gprot(c, i);
        q += PAGE;
    }
    drop(dg);
    unbusy_range(a, e);
    OWN_STATS[if fault { O_DIRTY_FAULTS } else { O_EXPLICIT }].fetch_add(1, Ordering::Relaxed);
    if fault {
        let site = (crate::log::thread_name(), FAULT_RIP.with(|r| r.get()));
        let mut g = DIRTY_SITES.lock();
        let e = g.get_or_insert_with(Default::default).entry(site).or_default();
        e.0 += 1;
        e.1 += n / PAGE;
    }
    OWN_STATS[O_RB_RUNS].fetch_add(1, Ordering::Relaxed);
    OWN_STATS[O_RB_PAGES].fetch_add(n / PAGE, Ordering::Relaxed);
    OWN_STATS[O_RB_DWORDS].fetch_add(dwords, Ordering::Relaxed);
    OWN_STATS[O_RB_NS].fetch_add(t.elapsed().as_nanos() as u64, Ordering::Relaxed);
    true
}

/// Faults on GPU-dirty pages by (thread, instruction): (faults, pages read
/// back) (`owned_line`).
static DIRTY_SITES: Mutex<Option<std::collections::HashMap<(String, u64), (u64, u64)>>> = Mutex::new(None);

fn dirty_sites_line() -> String {
    let g = DIRTY_SITES.lock();
    let Some(m) = g.as_ref() else { return String::new() };
    let mut v: Vec<(&(String, u64), &(u64, u64))> = m.iter().collect();
    v.sort_by(|a, b| b.1 .1.cmp(&a.1 .1));
    let parts: Vec<String> = v
        .iter()
        .take(16)
        .map(|((t, rip), (n, pages))| {
            let at = if *rip >= crate::mem::GUEST_BASE && *rip < crate::mem::GUEST_END { crate::exception::describe_guest(*rip) } else { crate::exception::host_symbol(*rip) };
            format!("{t} at {at}: {n} faults, {pages} pages")
        })
        .collect();
    format!("      faults on GPU-dirty pages by site (pages read back): {}
", parts.join("; "))
}

/// Releases the GPU-dirty pages of [base, base+len) (before kernel-mode
/// I/O, host writes and protection changes there).
pub fn release_range(base: u64, len: u64) {
    if !mirror_mode() || len == 0 {
        return;
    }
    let (lo, hi) = ((base & !(PAGE - 1)).max(MIRROR_BASE), base.saturating_add(len).next_multiple_of(PAGE).min(MIRROR_END));
    let mut p = lo;
    while p < hi {
        let Some(c) = chunk(p, false) else {
            p = ((p >> CHUNK_SHIFT) + 1) << CHUNK_SHIFT;
            continue;
        };
        let i = idx(p);
        if i & 63 == 0 && c.gdirty[i >> 6].load(Ordering::Acquire) == 0 && c.gprot[i >> 6].load(Ordering::Acquire) == 0 {
            p += 64 * PAGE;
            continue;
        }
        if is_gprot(c, i) {
            settle(p, false);
        } else if is_dirty(c, i) {
            release_dirty(p, false);
        }
        p += PAGE;
    }
}


/// Records a batch may take (grown when one has no room).
static GATHER_CAP: AtomicU64 = AtomicU64::new(2048);
/// The most `GATHER_CAP` grows to.
const GATHER_CAP_MAX: u64 = 16384;

/// Pages a batch's gather may record.
pub fn gather_cap() -> u64 {
    GATHER_CAP.load(Ordering::Relaxed)
}

/// Written ranges whose gather had no room for all records (their pages
/// keep their bits): the next batch's gather takes them too.
static REGATHER: Mutex<Vec<(u64, u64)>> = Mutex::new(Vec::new());

/// A batch's gather over `segs` left `over` records out: its ranges are
/// gathered again by the next batch, with room for more.
pub fn regather(segs: &[(u64, u64)], over: u32) {
    OWN_STATS[O_WB_OVER].fetch_add(over as u64, Ordering::Relaxed);
    let cap = GATHER_CAP.load(Ordering::Relaxed);
    if cap < GATHER_CAP_MAX {
        GATHER_CAP.store((cap * 2).min(GATHER_CAP_MAX), Ordering::Relaxed);
    }
    static LOGGED: AtomicU64 = AtomicU64::new(0);
    if LOGGED.fetch_add(1, Ordering::Relaxed) < 8 {
        crate::info!("resident: a write-back gather had no room for {over} pages (capacity {cap}); gathered again by the next batch");
    }
    REGATHER.lock().extend_from_slice(segs);
}

/// The ranges `regather` left for the next batch's gather.
pub fn take_regather() -> Vec<(u64, u64)> {
    std::mem::take(&mut *REGATHER.lock())
}

/// Mirror mode, CP writes the GPU performs (`host_for_cp`): [calls, pages
/// made host class, of them pages with GPU-written dwords released first].
static CP_HOSTED: [AtomicU64; 3] = [const { AtomicU64::new(0) }; 3];

/// Mirror mode: the pages of [a, a+n) a CP write the GPU performs targets
/// (labels, WRITE_DATA, timestamps) become host class, as RENDERER.md's
/// design has it: imported exactly (`exec::span` takes ranges made only of
/// host-class pages), never armed or mirrored, read and written in place,
/// so the GPU writes them in order and other queues may pass on its
/// promised writes. GPU-written dwords the mirror holds there are released
/// into guest memory first. False outside mirror mode or the span.
pub fn host_for_cp(a: u64, n: u64) -> bool {
    if !mirror_mode() || n == 0 {
        return false;
    }
    let (lo, hi) = (a & !(PAGE - 1), (a + n).next_multiple_of(PAGE));
    if lo < MIRROR_BASE || hi > MIRROR_END {
        return false;
    }
    CP_HOSTED[0].fetch_add(1, Ordering::Relaxed);
    let mut p = lo;
    while p < hi {
        if state(p) & ST_HOST == 0 {
            if let Some(c) = chunk(p, false) {
                if is_gprot(c, idx(p)) {
                    CP_HOSTED[2].fetch_add(1, Ordering::Relaxed);
                    settle(p, false);
                } else if is_dirty(c, idx(p)) {
                    CP_HOSTED[2].fetch_add(1, Ordering::Relaxed);
                    release_dirty(p, false);
                }
            }
            mark_host(p, PAGE);
            CP_HOSTED[1].fetch_add(1, Ordering::Relaxed);
        }
        p += PAGE;
    }
    true
}

/// Whether every page of [a, a+n) is host class (an exact import of it is
/// allowed in mirror mode).
pub fn all_host(a: u64, n: u64) -> bool {
    let (lo, hi) = (a & !(PAGE - 1), a.saturating_add(n).next_multiple_of(PAGE));
    let mut p = lo;
    while p < hi {
        if state(p) & ST_HOST == 0 {
            return false;
        }
        p += PAGE;
    }
    hi > lo
}

/// Mirror mode: pages of the mirror's span CP writes performed by the CPU
/// changed (`cp_wrote`), until a placement over them copies them into the
/// mirror again (`take_cp_pages`).
static CP_PAGES: Mutex<std::collections::BTreeSet<u64>> = Mutex::new(std::collections::BTreeSet::new());

/// A CP write the CPU performed changed [a, a+n).
pub fn cp_wrote(a: u64, n: u64) {
    if n == 0 || a >= MIRROR_END || a + n <= MIRROR_BASE {
        return;
    }
    let (lo, hi) = ((a & !(PAGE - 1)).max(MIRROR_BASE), (a + n).next_multiple_of(PAGE).min(MIRROR_END));
    let mut g = CP_PAGES.lock();
    let mut p = lo;
    while p < hi {
        g.insert(p);
        p += PAGE;
    }
}

/// The pages of [lo, hi) CP writes changed since their last copy into the
/// mirror (taken out first: a write after this copy notes its page again).
pub fn take_cp_pages(lo: u64, hi: u64) -> Vec<u64> {
    if lo >= hi {
        return Vec::new();
    }
    let mut g = CP_PAGES.lock();
    let v: Vec<u64> = g.range(lo..hi).copied().collect();
    for p in &v {
        g.remove(p);
    }
    OWN_STATS[O_CP_COPIES].fetch_add(v.len() as u64, Ordering::Relaxed);
    v
}

thread_local! {
    /// The faulting instruction of the access violation being handled (set
    /// by the exception handler before `on_fault`).
    pub static FAULT_RIP: std::cell::Cell<u64> = const { std::cell::Cell::new(0) };
}

/// The mirror's read-back context (the boot self-test's copy).
struct Readback {
    _pool: vk::CommandPool,
    cmd: vk::CommandBuffer,
    fence: vk::Fence,
    buf: vk::Buffer,
    ptr: *mut u8,
    copy: bool,
}
unsafe impl Send for Readback {}
static READBACK: Mutex<Option<Readback>> = Mutex::new(None);

fn readback_ctx() -> Result<Readback, String> {
    let c = ctx();
    let (family, copy) = match &c.copy {
        Some(q) => (q.family, true),
        None => (c.queue_family, false),
    };
    unsafe {
        let pool = c.device.create_command_pool(&vk::CommandPoolCreateInfo::default().queue_family_index(family).flags(vk::CommandPoolCreateFlags::RESET_COMMAND_BUFFER), None).map_err(|e| format!("pool: {e}"))?;
        let cmd = c.device.allocate_command_buffers(&vk::CommandBufferAllocateInfo::default().command_pool(pool).command_buffer_count(1)).map_err(|e| format!("cmd: {e}"))?[0];
        let fence = c.device.create_fence(&vk::FenceCreateInfo::default(), None).map_err(|e| format!("fence: {e}"))?;
        let (buf, _mem, ptr) = c.host_buffer(READBACK_MAX, vk::BufferUsageFlags::TRANSFER_DST)?;
        Ok(Readback { _pool: pool, cmd, fence, buf, ptr, copy })
    }
}

/// What mirror mode's placement does with a page (`mm_page`).
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum MmPage {
    /// The mirror holds its current contents (uploaded by a submitted batch,
    /// or GPU-owned).
    Current,
    /// Needs an upload with a claim (armed, so a later CPU write faults).
    Claim,
    /// Copied into this batch without a claim: its writes are not tracked
    /// (volatile, being written by a host write, not plain read-write) or
    /// another batch's upload of it is pending.
    Copy,
    /// Host class: imported (CP writes), never in the mirror.
    Host,
}

/// Mirror mode's view of page `i` of chunk `c` (at `p`; `mine`: pages this
/// batch uploads) whose current bit is clear: a page
/// found current (uploaded by a submitted batch, armed since) gets its bit
/// (set, then the state looked at again: a change in between clears it).
fn mm_page_in(c: &Chunk, i: usize, p: u64, mine: &super::fx::Set<u64>) -> MmPage {
    let s = c.pages[i].load(Ordering::Acquire);
    if s & ST_HOST != 0 {
        return MmPage::Host;
    }
    if mine.contains(&p) {
        return MmPage::Current;
    }
    let current = |c: &Chunk| {
        let s = c.pages[i].load(Ordering::Acquire);
        let seq = c.seqs[i].load(Ordering::Acquire);
        s & (ST_VOLATILE | ST_BUSY | ST_HOST) == 0 && c.writing[i].load(Ordering::Acquire) == 0 && seq != PENDING && seq != 0 && s & (ST_VALID | ST_ARMED) == ST_VALID | ST_ARMED
    };
    let seq = c.seqs[i].load(Ordering::Acquire);
    if s & (ST_VOLATILE | ST_BUSY) != 0 || c.writing[i].load(Ordering::Acquire) != 0 || seq == PENDING {
        return MmPage::Copy;
    }
    if current(c) {
        c.cur[i >> 6].fetch_or(1u64 << (i & 63), Ordering::AcqRel);
        if !current(c) {
            uncur(c, i);
        }
        MmPage::Current
    } else {
        MmPage::Claim
    }
}

/// Mirror mode's placement scan of [lo, hi) (page-aligned): the pages to
/// upload with a claim into `claim`, those to copy without one into `copy`;
/// false (stopped) at a host-class page. 64 pages the mirror holds current
/// (`Chunk::cur`) are passed at once.
pub fn mm_scan(lo: u64, hi: u64, mine: &super::fx::Set<u64>, claim: &mut Vec<u64>, copy: &mut Vec<u64>) -> bool {
    let mut p = lo;
    while p < hi {
        let Some(c) = chunk(p, false) else {
            let e = (((p >> CHUNK_SHIFT) + 1) << CHUNK_SHIFT).min(hi);
            while p < e {
                claim.push(p);
                p += PAGE;
            }
            continue;
        };
        let i = idx(p);
        let word = c.cur[i >> 6].load(Ordering::Acquire);
        if i & 63 == 0 && word == u64::MAX && p + 64 * PAGE <= hi {
            p += 64 * PAGE;
            continue;
        }
        if word & (1u64 << (i & 63)) == 0 {
            match mm_page_in(c, i, p, mine) {
                MmPage::Current => {}
                MmPage::Claim => claim.push(p),
                MmPage::Copy => copy.push(p),
                MmPage::Host => return false,
            }
        }
        p += PAGE;
    }
    true
}

/// Mirror mode's status line part.
pub fn owned_line() -> String {
    if !mirror_mode() {
        return String::new();
    }
    let v: Vec<u64> = OWN_STATS.iter().map(|c| c.load(Ordering::Relaxed)).collect();
    format!(
        "    mirror mode: write-backs of marked dwords: {} batches, {} pages, {} dwords ({} MiB), {:.1} ms ({:.1} us per batch), {} protection changes, {} pages left out (not writable), {} records without room (gathered again; capacity now {}), {} pages uploaded meanwhile (made stale); {} pages copied again after CP writes; lazy: {} records, {} pages made GPU-dirty ({} protection calls), {} faults and {} explicit releases, {} runs read back ({} pages, {} dwords stored, {:.1} us each), {} dirty pages dropped by remaps; baselines: {} taken at CPU writes, {} copies by difference ({} dwords)\n",
        v[O_WB_BATCHES],
        v[O_WB_PAGES],
        v[O_WB_DWORDS],
        (4 * v[O_WB_DWORDS]) >> 20,
        v[O_WB_NS] as f64 / 1e6,
        v[O_WB_NS] as f64 / v[O_WB_BATCHES].max(1) as f64 / 1e3,
        v[O_WB_PROTECTS],
        v[O_WB_SKIPPED],
        v[O_WB_OVER],
        gather_cap(),
        v[O_STALE],
        v[O_CP_COPIES],
        v[O_MASK_RECORDS],
        v[O_DIRTIED],
        v[O_DIRTY_PROTECTS],
        v[O_DIRTY_FAULTS],
        v[O_EXPLICIT],
        v[O_RB_RUNS],
        v[O_RB_PAGES],
        v[O_RB_DWORDS],
        v[O_RB_NS] as f64 / v[O_RB_RUNS].max(1) as f64 / 1e3,
        v[O_REMAP_DROPS],
        v[O_BASE_SNAPS],
        v[O_BASE_DIFFS],
        v[O_BASE_DWORDS]
    ) + &format!(
        "    mirror mode, CP writes the GPU performs: {} calls, {} pages made host class ({} with GPU-written dwords released first)\n    mirror mode, ownership at submission (lazy): {} batches, {} pages made no-access ({} calls); settled by {} faults and {} releases ({} waited for their batch, {} gave up waiting, {} without GPU-written dwords); {} masks dropped for pages remapped since their batch\n",
        CP_HOSTED[0].load(Ordering::Relaxed),
        CP_HOSTED[1].load(Ordering::Relaxed),
        CP_HOSTED[2].load(Ordering::Relaxed),
        OWN_PROT[O7_BATCHES].load(Ordering::Relaxed),
        OWN_PROT[O7_PAGES].load(Ordering::Relaxed),
        OWN_PROT[O7_PROTECTS].load(Ordering::Relaxed),
        OWN_PROT[O7_FAULTS].load(Ordering::Relaxed),
        OWN_PROT[O7_EXPLICIT].load(Ordering::Relaxed),
        OWN_PROT[O7_WAITED].load(Ordering::Relaxed),
        OWN_PROT[O7_TIMEOUTS].load(Ordering::Relaxed),
        OWN_PROT[O7_UNPROTECTED].load(Ordering::Relaxed),
        OWN_PROT[O7_REMAPPED].load(Ordering::Relaxed)
    ) + &dirty_sites_line()
}

/// Mirror mode's boot self-test: the read-back context can be made and a
/// copy through it returns what the mirror holds (ownership itself is
/// exercised in the game: guest pages are needed).
fn owned_self_test() -> Result<String, String> {
    let m = MIRROR.get().ok_or("no mirror")?;
    let at = MIRROR_END - m.page;
    let data: Vec<u8> = (0..PAGE).map(|i| (i * 13 + 5) as u8).collect();
    upload_now(at, &data)?;
    let rb = readback_ctx()?;
    let c = ctx();
    unsafe {
        c.device.begin_command_buffer(rb.cmd, &vk::CommandBufferBeginInfo::default().flags(vk::CommandBufferUsageFlags::ONE_TIME_SUBMIT)).map_err(|e| format!("begin: {e}"))?;
        c.device.cmd_copy_buffer(rb.cmd, m.buf, rb.buf, &[vk::BufferCopy { src_offset: at - MIRROR_BASE, dst_offset: 0, size: PAGE }]);
        c.device.end_command_buffer(rb.cmd).map_err(|e| format!("end: {e}"))?;
        c.submit_timeline(rb.copy, &[rb.cmd], &[], &[], rb.fence)?;
        c.device.wait_for_fences(&[rb.fence], true, 10_000_000_000).map_err(|e| format!("wait: {e}"))?;
        c.device.reset_fences(&[rb.fence]).map_err(|e| format!("fence reset: {e}"))?;
    }
    let back = unsafe { std::slice::from_raw_parts(rb.ptr, PAGE as usize) };
    if back != data.as_slice() {
        return Err("read-back context returned other data".into());
    }
    let queue = if rb.copy { "copy queue" } else { "main queue" };
    *READBACK.lock() = Some(rb);
    make_marks()?;
    Ok(format!("read-back context on the {queue}, written-bit map of {} MiB", MARK_BYTES >> 20))
}

/// The written-bit maps, made zeroed (`marks`; the compute family's queue's
/// only when it exists): sparse beside the mirror (`SparseMarks`), else
/// dense.
fn make_marks() -> Result<(), String> {
    if MARKS.get().is_some() {
        return Ok(());
    }
    let c = ctx();
    if let (true, Some(m)) = (c.sparse_buffers, MIRROR.get()) {
        match make_sparse_marks(m) {
            Ok(()) => return Ok(()),
            Err(e) => crate::warn!("resident: sparse written-bit maps: {e}; dense ones of {} MiB each instead", MARK_BYTES >> 20),
        }
    }
    let make = || -> Result<vk::Buffer, String> {
        let (buf, _mem) = c.device_buffer(MARK_BYTES, vk::BufferUsageFlags::STORAGE_BUFFER | vk::BufferUsageFlags::TRANSFER_DST | vk::BufferUsageFlags::TRANSFER_SRC | vk::BufferUsageFlags::INDIRECT_BUFFER)?;
        // Zeroed; the touched-page list's header [count, 1, 1, 0] is the list
        // gather's indirect dispatch.
        let header: Vec<u8> = [0u32, 1, 1, 0].iter().flat_map(|x| x.to_le_bytes()).collect();
        one_off(|cmd| unsafe {
            c.device.cmd_fill_buffer(cmd, buf, 0, vk::WHOLE_SIZE, 0);
            let mb = vk::MemoryBarrier::default().src_access_mask(vk::AccessFlags::TRANSFER_WRITE).dst_access_mask(vk::AccessFlags::TRANSFER_WRITE);
            c.device.cmd_pipeline_barrier(cmd, vk::PipelineStageFlags::TRANSFER, vk::PipelineStageFlags::TRANSFER, vk::DependencyFlags::empty(), &[mb], &[], &[]);
            c.device.cmd_update_buffer(cmd, buf, rdna::kernels::MARK_LIST_AT as u64 * 4, &header);
        })?;
        Ok(buf)
    };
    let _ = MARKS.set(make()?);
    if c.compute.is_some() {
        let _ = MARKS_C.set(make()?);
    }
    Ok(())
}

/// The written-bit maps as sparse buffers (`SparseMarks`): made, their
/// tails (page bits, touched-page lists) and the bit pages of the mirror's
/// pages backed so far bound, zeroed, the lists' headers written; then
/// published, under the mirror's backing lock (no bind meanwhile).
fn make_sparse_marks(m: &Mirror) -> Result<(), String> {
    let c = ctx();
    let usage = vk::BufferUsageFlags::STORAGE_BUFFER | vk::BufferUsageFlags::TRANSFER_DST | vk::BufferUsageFlags::TRANSFER_SRC | vk::BufferUsageFlags::INDIRECT_BUFFER;
    let n = 1 + c.compute.is_some() as usize;
    let mut bufs = Vec::new();
    for _ in 0..n {
        let mut bi = vk::BufferCreateInfo::default().size(MARK_BYTES).usage(usage).flags(vk::BufferCreateFlags::SPARSE_BINDING | vk::BufferCreateFlags::SPARSE_RESIDENCY);
        if c.families.len() > 1 {
            bi = bi.sharing_mode(vk::SharingMode::CONCURRENT).queue_family_indices(&c.families);
        }
        bufs.push(unsafe { c.device.create_buffer(&bi, None) }.map_err(|e| format!("vkCreateBuffer (sparse, {} MiB): {e}", MARK_BYTES >> 20))?);
    }
    let drop_all = |bufs: &[vk::Buffer]| unsafe {
        for &b in bufs {
            c.device.destroy_buffer(b, None);
        }
    };
    let req = unsafe { c.device.get_buffer_memory_requirements(bufs[0]) };
    if req.alignment != m.page || req.memory_type_bits & (1 << m.mem_type) == 0 || (MARK_WORDS_BYTES % req.alignment) != 0 {
        drop_all(&bufs);
        return Err(format!("sparse page {} KiB, memory types {:#x} (the mirror's: {} KiB, type {})", req.alignment >> 10, req.memory_type_bits, m.page >> 10, m.mem_type));
    }
    let page = req.alignment;
    let pages = MARK_BYTES.div_ceil(page);
    let mut g = m.blocks.lock();
    g.mark_claimed = vec![0; (pages as usize).div_ceil(64)];
    let mut alloc_ns = 0u64;
    let mut list: Vec<u64> = ((MARK_WORDS_BYTES / page)..pages).collect();
    let sm = SparseMarks { bufs: bufs.clone(), page };
    for p in 0..(MIRROR_END - MIRROR_BASE) / m.page {
        if m.backed[(p / 64) as usize].load(Ordering::Acquire) & (1u64 << (p % 64)) != 0 {
            let k = p * m.page / 32 / page;
            if !list.contains(&k) {
                list.push(k);
            }
        }
    }
    let mut binds: Vec<Vec<vk::SparseMemoryBind>> = vec![Vec::new(); n];
    for &k in &list {
        for b in binds.iter_mut() {
            let (mem, off, _) = take_page(&mut g, m, &mut alloc_ns, true)?;
            b.push(vk::SparseMemoryBind::default().resource_offset(k * page).size(page).memory(mem).memory_offset(off));
        }
        g.mark_claimed[(k / 64) as usize] |= 1u64 << (k % 64);
    }
    let infos: Vec<vk::SparseBufferMemoryBindInfo> = bufs.iter().zip(&binds).map(|(&b, v)| vk::SparseBufferMemoryBindInfo::default().buffer(b).binds(v)).collect();
    let info = vk::BindSparseInfo::default().buffer_binds(&infos);
    let fence = unsafe { c.device.create_fence(&vk::FenceCreateInfo::default(), None) }.map_err(|e| format!("vkCreateFence: {e}"))?;
    let r = {
        let q = c.queue.lock();
        unsafe { c.device.queue_bind_sparse(*q, &[info], fence) }
    };
    let r = r.map_err(|e| format!("vkQueueBindSparse: {e}")).and_then(|_| unsafe { c.device.wait_for_fences(&[fence], true, u64::MAX) }.map_err(|e| format!("bind fence: {e}")));
    unsafe { c.device.destroy_fence(fence, None) };
    if let Err(e) = r {
        g.mark_claimed.clear();
        drop_all(&bufs);
        return Err(e);
    }
    // Zeroed; the touched-page list's header [count, 1, 1, 0] is the list
    // gather's indirect dispatch.
    let header: Vec<u8> = [0u32, 1, 1, 0].iter().flat_map(|x| x.to_le_bytes()).collect();
    one_off(|cmd| unsafe {
        // (The last page reaches past the map's end: MARK_BYTES is not a
        // whole number of pages.)
        for &b in &bufs {
            for &k in &list {
                c.device.cmd_fill_buffer(cmd, b, k * page, page.min(MARK_BYTES - k * page), 0);
            }
        }
        let mb = vk::MemoryBarrier::default().src_access_mask(vk::AccessFlags::TRANSFER_WRITE).dst_access_mask(vk::AccessFlags::TRANSFER_WRITE);
        c.device.cmd_pipeline_barrier(cmd, vk::PipelineStageFlags::TRANSFER, vk::PipelineStageFlags::TRANSFER, vk::DependencyFlags::empty(), &[mb], &[], &[]);
        for &b in &bufs {
            c.device.cmd_update_buffer(cmd, b, rdna::kernels::MARK_LIST_AT as u64 * 4, &header);
        }
    })?;
    MARK_PAGES.fetch_add((list.len() * n) as u64, Ordering::Relaxed);
    let _ = MARKS.set(bufs[0]);
    if n > 1 {
        let _ = MARKS_C.set(bufs[1]);
    }
    let _ = SPARSE_MARKS.set(sm);
    drop(g);
    crate::info!("resident: written-bit maps sparse ({n} of {} MiB each, pages of {} KiB; {} MiB bound now)", MARK_BYTES >> 20, page >> 10, (list.len() as u64 * n as u64 * page) >> 20);
    Ok(())
}

/// The written-bit map's bytes of dword bits (`rdna::kernels::MARK_WORDS`):
/// the part a sparse map binds with the mirror's pages.
const MARK_WORDS_BYTES: u64 = 4 * rdna::kernels::MARK_WORDS as u64;

/// Runs `record` in a one-off command buffer on the main queue and waits
/// (front end 2's self-test).
pub fn one_off_now(record: impl FnOnce(vk::CommandBuffer)) -> Result<(), String> {
    one_off(record)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::platform::os::Win32::System::Memory::{VirtualAlloc, VirtualFree, MEM_COMMIT, MEM_DECOMMIT, MEM_RELEASE, MEM_RESERVE, PAGE_READWRITE};

    /// Exercise real signal recovery in isolated processes: policy initialization and
    /// section tracking must not race the other tests' global residency state.
    #[cfg(target_os = "linux")]
    #[test]
    fn linux_exact_tracking_across_epochs() {
        let Ok(policy) = std::env::var("REGALIANT_TRACKING_TEST") else {
            for policy in ["0", "1"] {
                let out = std::process::Command::new(std::env::current_exe().unwrap())
                    .args(["--exact", "gpu::resident::tests::linux_exact_tracking_across_epochs", "--nocapture"])
                    .env("REGALIANT_TRACKING_TEST", policy).env("DS_LINUX_VOLATILE_BACKOFF", policy).env("DS_LINUX_WRITE_WATCH", "mprotect")
                    .env("DS_SECTIONS", "1").env("DS_RESIDENT", "0").env("DS_LINUX_DIRECT_FAULT", "1")
                    .output().unwrap();
                assert!(out.status.success(), "policy {policy}: {}\n{}", String::from_utf8_lossy(&out.stdout), String::from_utf8_lossy(&out.stderr));
            }
            return;
        };
        crate::mem::reserve_guest_range().unwrap();
        crate::platform::signals::ensure_installed().unwrap();
        let _stack = crate::platform::signals::SignalStack::new().unwrap();
        init_linux_faults();
        let a = commit_at(MIRROR_END - (16 << 20), PAGE).unwrap();
        let take = || { let mut out = Vec::new(); take_written(a, PAGE, &mut out).unwrap(); out };
        assert_eq!(take(), vec![a]);
        for value in 1..=2 {
            unsafe { std::ptr::write_volatile(a as *mut u64, value); }
            assert_eq!(take(), vec![a]);
        }
        let direct = crate::platform::signals::DIRECT.load(Ordering::Relaxed);
        assert!(direct >= 2, "stores must exercise direct signal recovery");
        let mut redundant = 0;
        for _ in 0..64 {
            super::super::precache::end_scope(0);
            redundant += take().len();
        }
        assert_eq!(redundant, if policy == "1" { 64 } else { 0 });
        // Kernel/host writes still invalidate a clean page, without relying on a fault.
        host_write_begin(a, 8);
        unsafe { std::ptr::write_volatile(a as *mut u64, 3); }
        host_write_end(a, 8);
        super::super::precache::end_scope(0);
        assert_eq!(take(), vec![a]);
        assert_eq!(unsafe { std::ptr::read_volatile(a as *const u64) }, 3);
        disarm_range(a, a + PAGE);
        unsafe { VirtualFree(a as _, PAGE as usize, MEM_DECOMMIT) };
    }

    /// Commits [want, want+n) inside the mirror's span: in the guest range,
    /// which the first heap allocation reserved (tests too), or reserved
    /// here when it was not. None when that fails (the test is skipped).
    fn commit_at(want: u64, n: u64) -> Option<u64> {
        let mut a = unsafe { VirtualAlloc(want as _, n as usize, MEM_COMMIT, PAGE_READWRITE) } as u64;
        if a == 0 {
            a = unsafe { VirtualAlloc(want as _, n as usize, MEM_COMMIT | MEM_RESERVE, PAGE_READWRITE) } as u64;
        }
        if a != want {
            eprintln!("{want:#x}+{n:#x} could not be committed: test skipped");
            return None;
        }
        Some(a)
    }

    /// Arming, a simulated write fault and disarming, without the handler.
    #[test]
    fn arm_fault_disarm() {
        ON.store(true, Ordering::Release);
        let n = 8 * PAGE;
        #[cfg(windows)]
        let a = unsafe { VirtualAlloc(std::ptr::null(), n as usize, MEM_COMMIT | MEM_RESERVE, PAGE_READWRITE) } as u64;
        // Linux mmap(NULL) normally lives above the guest tracking table's span.
        #[cfg(target_os = "linux")]
        let a = commit_at(MIRROR_END - (8 << 20), n).expect("test memory inside the tracked guest span");
        assert_ne!(a, 0);
        assert_eq!(arm(a + 10, 2 * PAGE), 3);
        assert_eq!(state(a), ST_ARMED | ST_VALID);
        assert_eq!(state(a + 3 * PAGE), 0);
        assert!(!plain_rw(a, PAGE));
        // The handler's work for a write to page 1.
        assert!(on_fault(a + PAGE + 5, true));
        assert_eq!(state(a + PAGE), 0);
        assert!(plain_rw(a + PAGE, PAGE));
        unsafe { std::ptr::write_volatile((a + PAGE) as *mut u32, 7) };
        // Reads are not ours, nor pages never armed.
        assert!(!on_fault(a, false));
        assert!(on_fault(a + 3 * PAGE, true), "a writable page: retried as a race");
        // Host class pages are not armed; marking disarms.
        mark_host(a, PAGE);
        assert_eq!(state(a), ST_HOST);
        assert!(plain_rw(a, PAGE));
        assert_eq!(arm(a, PAGE), 0);
        // A remap resets armed pages' state (the guest's protection applies).
        remapped(a + 2 * PAGE, PAGE);
        assert_eq!(state(a + 2 * PAGE), 0);
        #[cfg(target_os = "linux")]
        {
            let _stack = crate::platform::signals::SignalStack::new().unwrap();
            crate::platform::signals::ensure_installed().unwrap();
            assert_eq!(arm(a + 4 * PAGE, PAGE), 1);
            // A real SIGSEGV passes through the Linux service, updates the shared resident
            // state, and resumes the interrupted store without touching the guest red zone.
            unsafe { std::ptr::write_volatile((a + 4 * PAGE) as *mut u32, 123); }
            assert_eq!(state(a + 4 * PAGE) & ST_ARMED, 0);
            assert_eq!(unsafe { std::ptr::read_volatile((a + 4 * PAGE) as *const u32) }, 123);
        }
        disarm_range(a, a + n);
        unsafe { VirtualFree(a as _, 0, MEM_RELEASE) };
    }

    /// Takes report pages written since their last take once: new pages,
    /// a write fault (simulated), a remap, a file read; uncommitted pages
    /// are skipped until committed (inside the mirror's span: file reads
    /// count there with residency on; skipped when that address is taken).
    #[test]
    fn takes_report_writes() {
        ON.store(true, Ordering::Release);
        let n = 16 * PAGE;
        let Some(a) = commit_at(MIRROR_END - (4 << 20), 8 * PAGE) else { return };
        let pages = |v: &[u64]| v.iter().map(|i| a + i * PAGE).collect::<Vec<_>>();
        let take = || {
            let mut out = Vec::new();
            take_written(a, n, &mut out).unwrap();
            out
        };
        assert_eq!(take(), pages(&[0, 1, 2, 3, 4, 5, 6, 7]));
        assert_eq!(take(), pages(&[]));
        assert!(!plain_rw(a, 8 * PAGE), "armed");
        // A write to page 3 (the handler's work, then the write).
        assert!(on_fault(a + 3 * PAGE + 8, true));
        unsafe { std::ptr::write_volatile((a + 3 * PAGE) as *mut u32, 1) };
        assert_eq!(take(), pages(&[3]));
        // A remap of pages 5..7 and a file read over page 1.
        remapped(a + 5 * PAGE, 2 * PAGE);
        host_write_begin(a + PAGE, 8);
        host_write_end(a + PAGE, 8);
        assert_eq!(take(), pages(&[1, 5, 6]));
        assert_eq!(take(), pages(&[]));
        // Page 2 written twice within VOLATILE_FRAMES: volatile, unarmed and
        // reported once per epoch until its stay ends, then armed again.
        #[cfg(windows)]
        {
        assert!(on_fault(a + 2 * PAGE, true));
        assert_eq!(take(), pages(&[2]));
        assert!(on_fault(a + 2 * PAGE, true));
        assert_eq!(state(a + 2 * PAGE) & ST_VOLATILE, ST_VOLATILE);
        assert_eq!(take(), pages(&[2]));
        assert_eq!(take(), pages(&[]), "once per epoch");
        assert!(plain_rw(a + 2 * PAGE, PAGE));
        crate::gpu::FRAMES.fetch_add(((VOL_BASE as u64) << VOL_MAX_LEVEL) + 1, Ordering::Relaxed);
        assert_eq!(take(), pages(&[2]));
        assert!(!plain_rw(a + 2 * PAGE, PAGE), "armed again");
        assert_eq!(state(a + 2 * PAGE) & ST_VOLATILE, 0);
        }
        #[cfg(target_os = "linux")]
        {
            for _ in 0..4 {
                assert!(on_fault(a + 2 * PAGE, true));
                assert_eq!(take(), pages(&[2]));
                crate::gpu::precache::end_scope(0);
                assert_eq!(take(), pages(&[]), "unchanged page stays clean across epochs");
                assert_eq!(state(a + 2 * PAGE) & ST_VOLATILE, 0);
                assert!(!plain_rw(a + 2 * PAGE, PAGE), "armed after every take");
            }
        }
        // The rest committed (mem.rs resets their states): reported next.
        assert_ne!(unsafe { VirtualAlloc((a + 8 * PAGE) as _, (8 * PAGE) as usize, MEM_COMMIT, PAGE_READWRITE) }, std::ptr::null_mut());
        remapped(a + 8 * PAGE, 8 * PAGE);
        assert_eq!(take(), pages(&[8, 9, 10, 11, 12, 13, 14, 15]));
        assert_eq!(take(), pages(&[]));
        disarm_range(a, a + n);
        unsafe { VirtualFree(a as _, n as usize, MEM_DECOMMIT) };
    }

    /// Claims, submission, and a write that cannot fault, on pages inside
    /// the mirror's span (skipped when that address is taken).
    #[test]
    fn claim_submit_host_write() {
        ON.store(true, Ordering::Release);
        let n = 8 * PAGE;
        let Some(a) = commit_at(MIRROR_END - (1 << 20), n) else { return };
        let pages: Vec<u64> = (0..4).map(|i| a + i * PAGE).collect();
        assert_eq!(page(a), Page::Upload);
        assert_eq!(claim_pages(&pages), 4);
        assert_eq!(page(a), Page::Pending);
        assert_eq!(state(a + PAGE), ST_ARMED | ST_VALID);
        // Another claim of a pending page fails.
        assert_eq!(claim_pages(&pages[..1]), 0);
        submitted(&pages, 7);
        assert_eq!(page(a + 2 * PAGE), Page::Ready);
        // A file read over pages 1..2: disarmed and kept out until it ends.
        host_write_begin(a + PAGE + 100, PAGE);
        assert_eq!(page(a + PAGE), Page::Refuse);
        assert!(plain_rw(a + PAGE, 2 * PAGE));
        assert_eq!(claim_pages(&[a + PAGE]), 0);
        host_write_end(a + PAGE + 100, PAGE);
        assert_eq!(page(a + PAGE), Page::Upload);
        assert_eq!(page(a + 3 * PAGE), Page::Ready);
        // A dropped batch's pages need an upload again.
        assert_eq!(claim_pages(&[a + PAGE]), 1);
        dropped(&[a + PAGE]);
        assert_eq!(page(a + PAGE), Page::Upload);
        disarm_range(a, a + n);
        unsafe { VirtualFree(a as _, n as usize, MEM_DECOMMIT) };
    }
}
