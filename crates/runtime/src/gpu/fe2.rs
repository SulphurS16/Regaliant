//! Front end 2 (docs/FRONTEND2.md; the default since 2026-10-30, `DS_FE2=0`
//! the old front end as a fallback): the GPU front end on one GPU address
//! space. Every buffer GPU work uses inside the mirror's span
//! ([MIRROR_BASE, MIRROR_END)) is bound from the mirror M, the sparse buffer
//! `resident` makes (backed on use); this module keeps which pages M holds
//! current and runs the write-back's kernel.
//!
//! - A page is valid when M holds what the CPU last wrote there (as of its
//!   last upload) and every GPU write since: GPU work writes only M, and the
//!   write-back (`rdna::kernels::mirror_scatter`, recorded by the executor
//!   before each CP write the GPU performs and at each batch's end) stores
//!   the dwords it marked into guest memory, so guest memory is current at
//!   batch boundaries for the CPU, the command processors and the image
//!   cache.
//! - CPU writes are found by write watching. A take stamps the pages it
//!   found unwritten with the synchronization epoch (`precache::epoch`) read
//!   before it; a valid page that is not armed is trusted by a check whose
//!   queue's view (`views_on`: a compute queue's last synchronization point;
//!   the epoch now for the graphics queue and its workers) is at most its
//!   stamp. Every take, ours or another's (`precache` applies them all under
//!   one protocol), clears the valid bit of the pages it reports
//!   (`marks_taken`), as do file reads, CP writes the CPU performs and
//!   mapping changes (`invalidate`).
//! - Hosted (the default, `hosted::wants_host`): pages quiet for ARM_QUIET flips (doubled each time an
//!   arming ended in a write, up to ARM_MAX_LEVEL) and found clean by takes
//!   in ARM_USES epochs since, inside import windows at least SETTLE flips
//!   old (protecting a new import's pages lost the device: FINDINGS.md
//!   2026-10-26) are armed read-only through the host; while the host's bit
//!   stays set (no write fault, disarm or mapping change since) such a page
//!   needs no take. Pages GPU work writes are armed too (since 2026-11-05:
//!   GPU writes do not fault and M holds them; `DS_FE2_ARM_GPU=0` leaves
//!   them unarmed, the pages the GPU writes through their imports). Each fault
//!   stops the whole process for 15-50 us, and the host answers it for its
//!   page's whole 64-page word (`hosted::set_word_mode`): a thread reads the
//!   faults the host answered (`watch`; the word's pages take the faulted
//!   page's backoff level), and a 64 MiB chunk with CHUNK_BURST of them
//!   within BURST_MS (or with some while all chunks together had ALL_BURST:
//!   a load rewriting memory armed before it) has its armed pages disarmed
//!   and its arming held CHUNK_HOLD flips (doubled for each burst within
//!   HOLD_MEMORY flips of its last hold, up to CHUNK_HOLD_MAX_LEVEL); the
//!   other chunks keep theirs.
//! - A take only spares uploads of pages that did not change: small runs of
//!   pages found written within HOT_FLIPS flips (the title's rings of
//!   per-draw data, rewritten every frame) are uploaded without one.
//! - Uploads are GPU copies from the import windows in the batch's
//!   prologue, on the main queue after every earlier batch, so after their
//!   write-backs (a copy the CPU made now could predate GPU writes of
//!   batches in flight and undo them in M). A batch's uploads count for
//!   other batches once it is queued (`uploaded`): batches queued later run
//!   after it; until then other batches upload such pages themselves.

use super::resident::{MIRROR_BASE, MIRROR_END};
use super::vk::ctx;
use ash::vk;
use parking_lot::Mutex;
use std::sync::atomic::{AtomicBool, AtomicPtr, AtomicU32, AtomicU64, AtomicU8, Ordering};

const PAGE: u64 = 4096;
/// Page states per chunk: 64 MiB of the span.
const CHUNK_SHIFT: u32 = 26;
const CHUNK_PAGES: usize = 1 << (CHUNK_SHIFT - 12);
const WORDS: usize = CHUNK_PAGES / 64;
const CHUNKS: usize = ((MIRROR_END - MIRROR_BASE) >> CHUNK_SHIFT) as usize;
/// Flips a page stays unwritten before it is armed the first time.
const ARM_QUIET: u64 = 30;
const ARM_MAX_LEVEL: u8 = 6;
/// Epochs a page must have been found clean in since its last write.
const ARM_USES: u8 = 8;
/// Faults on armed pages within BURST_MS that disarm a chunk (its own, or
/// all chunks' together while it had some), the flips its arming then
/// waits, and how a burst within HOLD_MEMORY flips of the chunk's last hold
/// doubles that (up to CHUNK_HOLD_MAX_LEVEL times). A burst is a load-like
/// storm: 64 faults in 100 ms cost ~1.7 ms of stopped process. (16 and 64
/// until 2026-10-28 small hours: at the gate, ~1.2 faults a flip in
/// clusters still held ~20 chunks a window, and their pages were taken every
/// epoch. 256 and 1024, tried the same night, let the page-by-page backoff
/// learn the gate's writes alone: 52 faults a flip against 6, 12 flips/s.)
const CHUNK_BURST: u32 = 64;
const ALL_BURST: u32 = 256;
const BURST_MS: u128 = 100;
/// Holds double only while bursts follow each other (a load going on): the
/// gate's storms, ~20 s apart, held their chunks 120 flips each with the
/// doubling counted over 600 flips, and a held chunk's pages are taken
/// every epoch; 8-flip holds made arming churn (~300 pages armed and
/// disarmed a flip, faults 6 -> 10 a flip; 2026-10-28).
const CHUNK_HOLD: u64 = 60;
const CHUNK_HOLD_MAX_LEVEL: u8 = 1;
const HOLD_MEMORY: u64 = 120;
/// Flips an import window exists before pages in it are armed.
pub const SETTLE: u64 = 60;
/// Runs of at most HOT_PAGES pages found written within HOT_FLIPS flips, or
/// within their quiet time before arming (`hot_quiet`), are uploaded without a
/// take (larger ones, as loads write them, would be uploaded again at every
/// epoch where a take finds them unchanged). A take of a page within its
/// quiet time found it unchanged 93% of the time over a 1-1 walk (~1,880
/// takes a flip, 3.6 us each, 87% of ranges under 64 KiB: FINDINGS.md
/// 2026-10-28 night), and each move of a view repeats it; an upload is a
/// copy region in the batch's prologue.
const HOT_FLIPS: u64 = 8;
const HOT_PAGES: u64 = 16;
/// Runs to check at most this many pages apart get one take (a take's
/// query costs ~0.75 us per MiB but microseconds, often tens of them with
/// other queries in flight, to make).
const TAKE_GAP: u64 = 256;

// The host's armed-page words cover the same 64 pages as ours.
const _: () = assert!((MIRROR_BASE - crate::mem::GUEST_BASE) % (64 * PAGE) == 0);

struct Chunk {
    /// M holds the page's current contents.
    valid: [AtomicU64; WORDS],
    /// Armed by us through the host (the host's bit says whether still).
    armed: [AtomicU64; WORDS],
    /// Armed and taken since (by one of our takes that began with the page
    /// armed): from then on only a write fault may change it; a take that
    /// finds such a page written found a write the protection missed
    /// (`take_run`: a hole).
    varmed: [AtomicU64; WORDS],
    /// Written by GPU work (marked only with `DS_FE2_ARM_GPU=0`, unarmed). Bits
    /// are only ever set.
    gpu: [AtomicU64; WORDS],
    /// Every `gpu` bit of the chunk set (a written range's scan covered it
    /// whole): its scans need set none, so its summary serves them too.
    gpu_all: AtomicBool,
    /// The epoch every valid page of the word was found clean in (a take
    /// over the whole word), else 0.
    wclean: [AtomicU32; WORDS],
    /// At least the newest epoch any page of the word has in `clean` (a
    /// take over part of the word): older than a check's requirement, no
    /// page of it meets it by its own stamp.
    wmax: [AtomicU32; WORDS],
    /// Bumped by every invalidation of a page of the word.
    wgen: [AtomicU32; WORDS],
    /// The epoch (low 32 bits) a take found the page clean in.
    clean: Box<[AtomicU32]>,
    /// The flip, plus one, a CPU write to the page was last found.
    wrote: Box<[AtomicU32]>,
    /// Bumped by every CPU write to the page found (and every invalidation
    /// of it): run-shared uploads hold while it stays (`run_shared`).
    pgen: Box<[AtomicU32]>,
    /// Armings of the page that ended in a write (each doubles the quiet
    /// time before the next).
    level: Box<[AtomicU8]>,
    /// Epochs a take looked at the word in since a write to one of its pages
    /// was last found (at most ARM_USES), and the last epoch counted. (Per
    /// page until 2026-10-27 late night: every take over a pool of ~113 MB
    /// then updated ~27,000 pages one by one.)
    uses: [AtomicU8; WORDS],
    use_ep: [AtomicU32; WORDS],
    /// Pages whose `wrote` is set (a page seen for the first time counts as
    /// written then: `set_clean`).
    seen: [AtomicU64; WORDS],
    /// Bumped by every invalidation in the chunk and every arming that
    /// ended there (the summary below goes with it). On a cache line of its
    /// own: every range-cache hit over the chunk reads it, and it lay beside
    /// the summary that scans store.
    cgen: Line<AtomicU32>,
    /// The chunk's summary: every page of it needed no check as of (`cgen`
    /// << 32 | the requirement it met (`ensure`'s `want`: met for any lower
    /// one too), or SUM_ARMED for every page armed: any) and the host's
    /// events of its leaf (`hosted::leaf_events`); 0: none.
    sum: AtomicU64,
    sum_host: AtomicU64,
    /// No page of the chunk is armed before this flip (after a burst of
    /// faults: `burst`), and the doublings of its last hold.
    hold_until: AtomicU64,
    hold_level: AtomicU8,
}

/// A value on a cache line of its own (`Chunk::cgen`).
#[repr(align(64))]
struct Line<T>(T);

impl<T> std::ops::Deref for Line<T> {
    type Target = T;
    fn deref(&self) -> &T {
        &self.0
    }
}

/// `Chunk::sum`'s epoch part when every page of the chunk is armed.
const SUM_ARMED: u32 = u32::MAX;

/// A queue without a view of its own (`view_cell`): its checks need pages
/// current as of the epoch now (the graphics queue and its workers).
pub const NO_VIEW: u32 = u32::MAX;

/// Per-queue views (`DS_FE2_VIEWS`, on by default; 0: every check needs the
/// epoch now): a compute queue's checks need pages current as of its own
/// last synchronization point (a title submission, a wait the CPU released:
/// the epoch then; a wait on a label another queue's CP wrote: that queue's
/// view, the epoch now for queues without one), not as of every queue's.
/// The epoch ends at any queue's (~260 times a flip at the gate), and queue
/// 0x50 checked its pools again after each. A take's pages are stamped with
/// the epoch before it, so a page stamped at or after a queue's view
/// reflects every CPU write before that queue's synchronization point.
static VIEWS: Mutex<Vec<(String, std::sync::Arc<AtomicU32>)>> = Mutex::new(Vec::new());

pub fn views_on() -> bool {
    static W: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *W.get_or_init(|| std::env::var("DS_FE2_VIEWS").ok().as_deref() != Some("0"))
}

/// `DS_FE2_GFX_VIEWS` (on by default; 0: the graphics queue and its
/// workers need the epoch now, as before 2026-10-28 night): the graphics
/// queue has a view of its own too (its submissions, waits the CPU
/// released, REWINDs, waits in parallel runs the CPU released; the waits on
/// other queues' labels adopt theirs), and each parallel run's workers
/// check as of the view its command processor had when it planned the run.
/// While the camera moved its checks re-took the pages of every epoch any
/// queue ended (~3,400 takes a flip in a 1-1 walk window).
pub fn gfx_views() -> bool {
    static W: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *W.get_or_init(|| std::env::var("DS_FE2_GFX_VIEWS").ok().as_deref() != Some("0"))
}

/// The view of the queue (or parallel worker) named `name` (made at
/// NO_VIEW).
pub fn view_cell(name: &str) -> std::sync::Arc<AtomicU32> {
    let mut g = VIEWS.lock();
    if let Some((_, c)) = g.iter().find(|(n, _)| n == name) {
        return c.clone();
    }
    let c = std::sync::Arc::new(AtomicU32::new(NO_VIEW));
    g.push((name.to_string(), c.clone()));
    c
}

/// The view of the queue named `name` (None: it has none).
pub fn view_of(name: &str) -> Option<u32> {
    let g = VIEWS.lock();
    g.iter().find(|(n, _)| n == name).map(|(_, c)| c.load(Ordering::Acquire)).filter(|&v| v != NO_VIEW)
}

/// A queue synchronized: its view becomes at least `v`; whether that
/// raised it.
pub fn sync_view(cell: &AtomicU32, v: u32) -> bool {
    let old = cell.fetch_update(Ordering::AcqRel, Ordering::Acquire, |c| Some(if c == NO_VIEW { v } else { c.max(v) })).unwrap_or(NO_VIEW);
    VIEW_SYNCS[0].fetch_add(1, Ordering::Relaxed);
    let raised = old == NO_VIEW || old < v;
    if raised {
        VIEW_SYNCS[1].fetch_add(1, Ordering::Relaxed);
    }
    raised
}

/// Views synchronized, and those it raised (`status`).
static VIEW_SYNCS: [AtomicU64; 2] = [const { AtomicU64::new(0) }; 2];
/// Views raised by the queues' own synchronization (`Cp::fe2_sync`): the
/// graphics command processor's, the compute queues' (`status`).
static VIEW_RAISES: [AtomicU64; 2] = [const { AtomicU64::new(0) }; 2];

/// Queue `name`'s own synchronization raised its view (`status`).
pub fn note_raise(compute: bool) {
    VIEW_RAISES[compute as usize].fetch_add(1, Ordering::Relaxed);
}

/// Ranges per thread whose last check found every page fine (`RANGES`).
const RANGE_CACHE: usize = 32;
/// Ranges over more chunks or host leaves than these are not kept there.
const FINE_CHUNKS: usize = 8;
const FINE_LEAVES: usize = 5;

/// A range a thread found needing no check: the generations of the chunks
/// it covers (`Chunk::cgen`: every invalidation of a valid page and every
/// ended arming there bumps them) and the host's events of its leaves
/// (`hosted::leaf_events`) as they were before that check, and the
/// requirement it met (`ensure`'s `want`; SUM_ARMED: every page was armed,
/// any). While they are unchanged it still needs none. (Until 2026-10-27
/// late night one generation over the whole span: queue 0x50's pools of
/// ~113 MB were scanned again after any invalidation anywhere, ~3 us a
/// binding.)
#[derive(Clone, Copy)]
struct Fine {
    lo: u64,
    hi: u64,
    gens: [u32; FINE_CHUNKS],
    host: [u64; FINE_LEAVES],
    ep: u32,
    /// Its words were marked GPU-written (a written binding's check).
    marked: bool,
}

/// This thread's ranges found fine (`RANGES`): a ring, newest at `head`, the ranges' bounds
/// packed apart from their records so a lookup reads 512 bytes, not the records' 3 KiB, and an
/// insert writes one entry instead of moving the list (2026-11-16: the list's insert at its front
/// and its retain were most of a full check's bookkeeping). A dropped entry has `hi` 0.
struct Ring {
    keys: [(u64, u64); RANGE_CACHE],
    fines: [Fine; RANGE_CACHE],
    head: usize,
}

/// This thread's exact ranges found fine (`EXACT_SLOTS`, by `exact_slot`): the bounds packed
/// apart (64 KiB a thread, the records' 384 KiB read only on a match); an empty slot has `hi` 0.
struct Exact {
    keys: Vec<(u64, u64)>,
    fines: Vec<Fine>,
}

const FINE_NONE: Fine = Fine { lo: 0, hi: 0, gens: [0; FINE_CHUNKS], host: [0; FINE_LEAVES], ep: 0, marked: false };

thread_local! {
    /// This thread's ranges found fine, most recent first.
    static RANGES: std::cell::RefCell<Ring> = const { std::cell::RefCell::new(Ring { keys: [(0, 0); RANGE_CACHE], fines: [FINE_NONE; RANGE_CACHE], head: 0 }) };
    /// This thread's exact ranges found fine.
    static EXACT: std::cell::RefCell<Exact> = const { std::cell::RefCell::new(Exact { keys: Vec::new(), fines: Vec::new() }) };
}

/// Slots of each thread's table of exact ranges found fine, beside the
/// RANGE_CACHE ranges that may cover a binding: a frame's distinct bindings
/// outnumber those (the graphics CP found no covering range in 42% of its
/// ~15,850 checks a flip at Sec09 Firebridge, 2026-10-28 night). A slot
/// holds the last range hashed there (`DS_FE2_EXACT=0`: none).
const EXACT_SLOTS: usize = 4096;

fn exact_on() -> bool {
    static W: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *W.get_or_init(|| std::env::var("DS_FE2_EXACT").ok().as_deref() != Some("0"))
}

/// The slot of page-aligned [lo, hi) in `EXACT`.
#[inline]
fn exact_slot(lo: u64, hi: u64) -> usize {
    let h = (lo >> 12).wrapping_mul(0x9E37_79B9_7F4A_7C15) ^ (hi >> 12).wrapping_mul(0xC2B2_AE3D_27D4_EB4F);
    (h >> (64 - EXACT_SLOTS.trailing_zeros())) as usize
}

/// The generations of the chunks over [lo, hi) and the host's events of its
/// leaves (None: more than FINE_CHUNKS or FINE_LEAVES of them).
#[inline]
fn snapshot(lo: u64, hi: u64) -> Option<([u32; FINE_CHUNKS], [u64; FINE_LEAVES])> {
    let (c0, c1) = (((lo - MIRROR_BASE) >> CHUNK_SHIFT) as usize, ((hi - 1 - MIRROR_BASE) >> CHUNK_SHIFT) as usize);
    if c1 - c0 >= FINE_CHUNKS {
        return None;
    }
    let mut gens = [0u32; FINE_CHUNKS];
    for (k, g) in (c0..=c1).zip(gens.iter_mut()) {
        let p = TABLE[k].load(Ordering::Acquire);
        // (A chunk made later starts at generation 0.)
        if !p.is_null() {
            *g = unsafe { &*p }.cgen.load(Ordering::Acquire);
        }
    }
    let mut host = [0u64; FINE_LEAVES];
    let mut a = lo;
    let mut k = 0;
    while a < hi {
        if k == FINE_LEAVES {
            return None;
        }
        host[k] = crate::hosted::leaf_events(a);
        k += 1;
        a = crate::hosted::leaf_end(a);
    }
    Some((gens, host))
}

/// Whether [lo, hi) lies in a range this thread found fine and nothing
/// changed there since.
#[inline]
fn range_fine(lo: u64, hi: u64, want: u32, marked: bool) -> bool {
        let _cpu = crate::gpu::cpu_profile::scope(crate::gpu::cpu_profile::Part::Fe2Versions);
    // The exact range first.
    let mut exact_why = None;
    if exact_on() {
        let hit = EXACT.with(|t| {
            let mut t = t.borrow_mut();
            let k = exact_slot(lo, hi);
            if *t.keys.get(k)? != (lo, hi) {
                return None;
            }
            let f = &t.fines[k];
            if marked && !f.marked {
                return None;
            }
            match changed(f) {
                Some(c) => {
                    t.keys[k].1 = 0;
                    Some(Err(c))
                }
                None if f.ep >= want => Some(Ok(())),
                None => Some(Err(C_EPOCH)),
            }
        });
        match hit {
            Some(Ok(())) => return true,
            Some(Err(c)) => exact_why = Some(c),
            None => {}
        }
    }
    RANGES.with(|r| {
        let mut r = r.borrow_mut();
        // (Why not, for `DS_FE2_CALLERS`: the last covering range's reason.)
        let mut why = exact_why.unwrap_or(C_NO_RANGE);
        let head = r.head;
        for j in 0..RANGE_CACHE {
            let i = (head + j) % RANGE_CACHE;
            let (flo, fhi) = r.keys[i];
            if !(flo <= lo && hi <= fhi) {
                continue;
            }
            let f = &r.fines[i];
            if !(f.marked || !marked) {
                continue;
            }
            let ep_ok = f.ep >= want;
            match changed(f) {
                // Generations and events only grow: it never holds again.
                Some(c) => {
                    why = c;
                    r.keys[i].1 = 0;
                }
                None if ep_ok => {
                    // (One step toward the newest a hit.)
                    if j > 0 {
                        let n = (head + j - 1) % RANGE_CACHE;
                        r.keys.swap(i, n);
                        r.fines.swap(i, n);
                    }
                    return true;
                }
                None => why = C_EPOCH,
            }
        }
        if callers_on() {
            caller_count(why, 1);
        }
        false
    })
}

/// What changed in `f`'s chunks (C_GENS) or host leaves (C_HOST) since it
/// was found fine, compared as read (None: nothing).
#[inline]
fn changed(f: &Fine) -> Option<usize> {
        let _cpu = crate::gpu::cpu_profile::scope(crate::gpu::cpu_profile::Part::Fe2Versions);
    let (c0, c1) = (((f.lo - MIRROR_BASE) >> CHUNK_SHIFT) as usize, ((f.hi - 1 - MIRROR_BASE) >> CHUNK_SHIFT) as usize);
    for (k, &g) in (c0..=c1).zip(f.gens.iter()) {
        let p = TABLE[k].load(Ordering::Acquire);
        // (A chunk made later starts at generation 0.)
        let now = if p.is_null() { 0 } else { unsafe { &*p }.cgen.load(Ordering::Acquire) };
        if now != g {
            return Some(C_GENS);
        }
    }
    let (mut a, mut k) = (f.lo, 0);
    while a < f.hi && k < FINE_LEAVES {
        if crate::hosted::leaf_events(a) != f.host[k] {
            return Some(C_HOST);
        }
        k += 1;
        a = crate::hosted::leaf_end(a);
    }
    None
}

thread_local! {
    /// How the last `ensure` on this thread was answered: 0 by its range cache (`range_fine`),
    /// 1 by a full check with nothing to do, 2 by one with takes or uploads.
    static LAST_FINE: std::cell::Cell<u8> = const { std::cell::Cell::new(2) };
}

/// The invalidation generation of the 64-page word holding page `p` (`Chunk::wgen`, bumped by
/// every invalidation of a page of it; 0 outside the span or before its chunk exists): the
/// executors' placement and page memos keep it from before their placement's check and answer
/// only while it stays (`Exec::fe2_place_memo`).
#[inline]
pub fn word_gen(p: u64) -> u32 {
    chunk(p, false).map_or(0, |c| c.wgen[idx(p) / 64].load(Ordering::Acquire))
}

/// How the last `ensure` on this thread was answered (`LAST_FINE`).
pub fn last_fine() -> u8 {
    LAST_FINE.with(|c| c.get())
}

/// Diagnosis (`DS_IMM_STATE`, 2026-11-19): page `a`'s state here, for logs: valid, armed by us,
/// the host's bit, armed and taken since, GPU-written, its clean stamp, its word's clean and
/// newest stamps against `want`, the flip (plus one) a CPU write to it was last found and the
/// flip now, its write generation and its arming level.
pub fn page_state(a: u64, want: u32) -> String {
    let Some(c) = chunk(a, false) else { return "no chunk".into() };
    let i = idx(a);
    let w = i / 64;
    let m = 1u64 << (i % 64);
    let bit = |x: &AtomicU64| (x.load(Ordering::Acquire) & m != 0) as u8;
    let host = (crate::hosted::armed_word(a & !(64 * PAGE - 1)) & m != 0) as u8;
    format!(
        "valid {} armed {} host {host} varmed {} gpu {} clean {} wclean {} wmax {} want {want} wrote {} flip {} pgen {} level {}",
        bit(&c.valid[w]),
        bit(&c.armed[w]),
        bit(&c.varmed[w]),
        bit(&c.gpu[w]),
        c.clean[i].load(Ordering::Acquire),
        c.wclean[w].load(Ordering::Acquire),
        c.wmax[w].load(Ordering::Acquire),
        c.wrote[i].load(Ordering::Relaxed),
        super::FRAMES.load(Ordering::Relaxed),
        c.pgen[i].load(Ordering::Relaxed),
        c.level[i].load(Ordering::Relaxed)
    )
}

/// The flip, plus one, a CPU write to the page of `a` was last found (`Chunk::wrote`; 0: never
/// found, or no chunk of the mirror's table holds it). For censuses.
pub fn wrote_flip(a: u64) -> u32 {
    chunk(a, false).map_or(0, |c| c.wrote[((a - MIRROR_BASE) >> 12) as usize & (CHUNK_PAGES - 1)].load(Ordering::Relaxed))
}

/// Front end 2's lean check (`Exec::fe2_place_quick`): whether every page of the read-only range
/// [lo, hi) (page-aligned, a few words at most) is current for `want` as its state stands
/// (valid and clean for it, or armed), which `ensure` would find with nothing to do. False:
/// `ensure` runs (pages its caller's batch uploaded are current too, which only it knows).
#[inline]
pub fn quick_fine(lo: u64, hi: u64, want: u32) -> bool {
        let _cpu = crate::gpu::cpu_profile::scope(crate::gpu::cpu_profile::Part::Fe2Versions);
    let mut a = lo;
    while a < hi {
        let e = word_end(a, hi);
        let Some(c) = chunk(a, false) else { return false };
        let m = word_mask(a, e);
        if word_ok(c, a, e, want) & m != m {
            return false;
        }
        a = e;
    }
    true
}

/// `DS_STEP_LAPS=1`: front end 2's checks with work timed by the kind of each run they
/// handled (`WORK_PARTS`).
fn work_laps() -> bool {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ON.get_or_init(|| std::env::var("DS_STEP_LAPS").ok().as_deref() == Some("1"))
}

/// Checks with work by run kind (`DS_STEP_LAPS`), TSC cycles: [runs of pages written lately
/// uploaded without a take: cycles, runs, pages; runs another thread's take covered: cycles,
/// runs, pages; runs taken: cycles, runs, pages; the checks with work: cycles of their run
/// loops, checks], added per thread every 1,024 checks.
pub static WORK_PARTS: [AtomicU64; 11] = [const { AtomicU64::new(0) }; 11];

thread_local! {
    static WORK_LOCAL: std::cell::Cell<[u64; 11]> = const { std::cell::Cell::new([0; 11]) };
}

/// Adds run kind `k`'s cycles and pages (`WORK_PARTS`).
fn work_part(k: usize, cycles: u64, pages: u64) {
    WORK_LOCAL.with(|c| {
        let mut v = c.get();
        v[3 * k] += cycles;
        v[3 * k + 1] += 1;
        v[3 * k + 2] += pages;
        c.set(v);
    });
}

/// Ends a check with work (`WORK_PARTS`).
fn work_check(cycles: u64) {
    WORK_LOCAL.with(|c| {
        let mut v = c.get();
        v[9] += cycles;
        v[10] += 1;
        if v[10] >= 1024 {
            for (a, x) in WORK_PARTS.iter().zip(v.iter()) {
                a.fetch_add(*x, Ordering::Relaxed);
            }
            v = [0; 11];
        }
        c.set(v);
    });
}

/// `WORK_PARTS` as a status line ("" before any).
pub fn work_parts_line() -> String {
    let v: Vec<u64> = WORK_PARTS.iter().map(|c| c.load(Ordering::Relaxed)).collect();
    if v[10] == 0 {
        return String::new();
    }
    let hz = super::threadcpu::tsc_hz();
    let us = |c: u64, n: u64| if n == 0 { 0.0 } else { c as f64 * 1e6 / hz / n as f64 };
    format!(
        "    front end 2's checks with work (`DS_STEP_LAPS`): {} at {:.2} us of run handling; runs written lately uploaded without a take {} at {:.2} us ({} pages), runs another thread's take covered {} at {:.2} us ({} pages), runs taken {} at {:.2} us ({} pages)\n",
        v[10],
        us(v[9], v[10]),
        v[1],
        us(v[0], v[1]),
        v[2],
        v[4],
        us(v[3], v[4]),
        v[5],
        v[7],
        us(v[6], v[7]),
        v[8]
    )
}

/// Resets `WORK_PARTS` (`profile reset`).
pub fn reset_work_parts() {
    for c in WORK_PARTS.iter() {
        c.store(0, Ordering::Relaxed);
    }
    if let Some(m) = BIG_RUNS.lock().as_mut() {
        m.clear();
    }
}

/// `DS_STEP_LAPS`: runs of at least `BIG_RUN_PAGES` pages that checks took or waited on, by
/// range: [takes, their cycles, waits on another thread's take, their cycles, pages written found].
static BIG_RUNS: parking_lot::Mutex<Option<std::collections::HashMap<(u64, u64), [u64; 5]>>> = parking_lot::Mutex::new(None);
const BIG_RUN_PAGES: u64 = 256;

/// Notes a big run's take (`k` 0) or wait (`k` 1) (`BIG_RUNS`).
fn note_big_run(a: u64, b: u64, k: usize, cycles: u64, written: u64) {
    let mut g = BIG_RUNS.lock();
    let m = g.get_or_insert_with(std::collections::HashMap::new);
    if m.len() >= 4096 && !m.contains_key(&(a, b)) {
        return;
    }
    let e = m.entry((a, b)).or_insert([0; 5]);
    e[2 * k] += 1;
    e[2 * k + 1] += cycles;
    e[4] += written;
}

/// The big runs taken or waited on most, by time (`BIG_RUNS`; "" before any).
pub fn big_runs_line() -> String {
    let g = BIG_RUNS.lock();
    let Some(m) = g.as_ref().filter(|m| !m.is_empty()) else { return String::new() };
    let hz = super::threadcpu::tsc_hz();
    let mut v: Vec<(&(u64, u64), &[u64; 5])> = m.iter().collect();
    v.sort_unstable_by_key(|(_, e)| std::cmp::Reverse(e[1] + e[3]));
    let total: u64 = m.values().map(|e| e[1] + e[3]).sum();
    let names = crate::engine::names();
    let parts: Vec<String> = v
        .iter()
        .take(12)
        .map(|&(r, e)| {
            let (a, b) = *r;
            format!(
                "{a:#x}+{:#x} ({}) taken {} at {:.0} us, waited on {} at {:.0} us, {} pages found written",
                b - a,
                names.get(&crate::engine::owner(a)).map_or("?", |s| s.as_str()),
                e[0],
                e[1] as f64 * 1e6 / hz / e[0].max(1) as f64,
                e[2],
                e[3] as f64 * 1e6 / hz / e[2].max(1) as f64,
                e[4]
            )
        })
        .collect();
    format!("    front end 2's big runs (`DS_STEP_LAPS`, {} ranges, {:.0} ms in all): {}\n", m.len(), total as f64 * 1e3 / hz, parts.join("; "))
}

/// `DS_FE2_CALLERS=1` (diagnosis): placement checks counted by caller
/// (`set_caller`), with why the range cache missed.
pub fn callers_on() -> bool {
    static W: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *W.get_or_init(|| std::env::var("DS_FE2_CALLERS").ok().as_deref() == Some("1"))
}

/// `DS_FE2_CALLERS_MEMO=1` with `DS_FE2_CALLERS=1`: the executors' placement memo stays on, so
/// the census sees the placements the memo misses (its parts as in a normal run).
pub fn callers_keep_memo() -> bool {
    static W: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *W.get_or_init(|| std::env::var("DS_FE2_CALLERS_MEMO").ok().as_deref() == Some("1"))
}

const CALLER_NAMES: [&str; 4] = ["graphics CP", "its parallel workers", "queue 0x50", "other compute queues"];
/// Per caller: [checks, range-cache hits, misses with no range covering,
/// misses on another epoch, misses on chunk generations, misses on host
/// events, words scanned, takes, TSC cycles in all, placements (the
/// executor's `fe2_place`), their TSC cycles].
static BY_CALLER: [[AtomicU64; 25]; 4] = [const { [const { AtomicU64::new(0) }; 25] }; 4];
/// The same for placements made while a batch's (or parallel chunk's) first
/// command is bound (`set_first_command`): first commands cost 100-216 us
/// against ~20 us for later ones at Castlewall's arrival, most of it
/// binding (FINDINGS.md 2026-10-31).
static BY_CALLER_FIRST: [[AtomicU64; 25]; 4] = [const { [const { AtomicU64::new(0) }; 25] }; 4];

thread_local! {
    /// The command being bound opens its batch (`set_first_command`).
    static FIRST_COMMAND: std::cell::Cell<bool> = const { std::cell::Cell::new(false) };
}

/// Whether the command this thread binds now opens its batch (the
/// executor's draws and dispatches; `DS_FE2_CALLERS`).
#[inline]
pub fn set_first_command(first: bool) {
    if callers_on() {
        FIRST_COMMAND.with(|f| f.set(first));
    }
}
const C_PLACES: usize = 9;
const C_PLACE_CYCLES: usize = 10;
/// A placement's parts outside `ensure` (TSC cycles): finding its mapping
/// and backing the mirror, beginning the batch and the written range's
/// import, the bookkeeping after `ensure` (its uploads' pages, deferred CP
/// writes).
const C_PART_MAP: usize = 11;
const C_PART_BEGIN: usize = 12;
const C_PART_AFTER: usize = 13;
/// Of C_PART_BEGIN, the batch's begin alone (the rest: the written range's
/// import, `exec::span`).
const C_PART_BATCH: usize = 18;
/// The checks' parts (TSC cycles): scanning words, arming (`try_arm`),
/// takes, making runs current (`fix_run`).
const C_SCAN: usize = 14;
const C_ARM: usize = 15;
const C_TAKE: usize = 16;
const C_FIX: usize = 17;
/// More of the checks' parts: the range caches' lookup, following another
/// thread's take (`follow_take`, `followed`), uploads made without a take
/// (pages written lately, pages a take of the epoch found lacking).
const C_CACHE: usize = 19;
const C_FOLLOW: usize = 20;
const C_UPLOAD: usize = 21;
/// Placements of a range (base, size, written) its batch placed before at
/// the same epoch, and at an earlier one (`Exec::fe2_memo`, the census of
/// what a per-batch memo of placements could skip).
const C_REP_SAME: usize = 22;
const C_REP_BATCH: usize = 23;
/// The cycles of the same-epoch repeats (what such a memo would spare, less its lookup).
const C_REP_CYCLES: usize = 24;

/// The TSC when `DS_FE2_CALLERS` is on, else 0.
#[inline]
fn tsc() -> u64 {
    if callers_on() {
        unsafe { core::arch::x86_64::_rdtsc() }
    } else {
        0
    }
}

/// Adds the cycles since `t0` (`tsc`) to part `what`.
#[inline]
fn caller_since(what: usize, t0: u64) {
    if t0 != 0 {
        caller_count(what, unsafe { core::arch::x86_64::_rdtsc() }.wrapping_sub(t0));
    }
}

/// Cycles of a placement's parts (`DS_FE2_CALLERS`): map, begin (of it
/// the batch's begin alone), after.
/// A placement repeating one of its batch (`C_REP_SAME` / `C_REP_BATCH`).
pub fn note_place_repeat(same_epoch: bool) {
    caller_count(if same_epoch { C_REP_SAME } else { C_REP_BATCH }, 1);
}

/// A same-epoch repeat's cycles (`C_REP_CYCLES`).
pub fn note_repeat_cycles(cycles: u64) {
    caller_count(C_REP_CYCLES, cycles);
}

pub fn note_place_parts(map: u64, begin: u64, batch: u64, after: u64) {
    caller_count(C_PART_MAP, map);
    caller_count(C_PART_BEGIN, begin);
    caller_count(C_PART_BATCH, batch);
    caller_count(C_PART_AFTER, after);
}

/// Clears the checks by caller (`profile reset`).
pub fn reset_callers() {
    reset_up_census();
    take_census_reset();
    for c in BY_CALLER.iter().flatten().chain(BY_CALLER_FIRST.iter().flatten()).chain(QUIET_TAKES.iter().flatten()).chain(BY_SOURCE.iter().flatten()) {
        c.store(0, Ordering::Relaxed);
    }
}

/// `DS_FE2_QUIET_CENSUS=1` (diagnosis): each take classed by whether every
/// page of its range is GPU-written with no CPU write found for QUIET_FLIPS
/// flips (what view-backed GPU-quiet mappings would spare: FRONTEND2.md
/// "GPU-quiet mappings on sections"), by caller (`set_caller`).
pub fn quiet_census_on() -> bool {
    static W: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *W.get_or_init(|| std::env::var("DS_FE2_QUIET_CENSUS").ok().as_deref() == Some("1"))
}

/// `DS_FE2_UPCENSUS=1` (diagnosis): every page uploaded into the mirror
/// with the flip and its generation (`Chunk::pgen`: CPU writes takes found,
/// invalidations) at its last upload, so an upload of a page unchanged since
/// an upload of it in the same flip shows: a copy of what another batch's
/// prologue holds already (pages become valid only when their batch is
/// queued, and a parallel run's chunks check the same ranges together), by
/// caller and by the path that uploaded it (`UP_KINDS`). Pages written lately
/// are uploaded without a take, so their generation misses writes since the
/// last take: theirs may be new contents.
pub fn up_census_on() -> bool {
    static W: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *W.get_or_init(|| std::env::var("DS_FE2_UPCENSUS").ok().as_deref() == Some("1"))
}

/// The paths that upload pages into the mirror (`up_census_on`).
const UP_KINDS: [&str; 5] = ["other", "written lately", "a take of the epoch looked at", "after another's take", "after a take"];

thread_local! {
    /// The path uploading now (`UP_KINDS`).
    static UP_KIND: std::cell::Cell<usize> = const { std::cell::Cell::new(0) };
}

/// Sets this thread's upload path until dropped (`up_census_on`).
struct UpKind(usize);

impl UpKind {
    #[inline]
    fn new(k: usize) -> UpKind {
        if !up_census_on() {
            return UpKind(usize::MAX);
        }
        UpKind(UP_KIND.with(|c| c.replace(k)))
    }
}

impl Drop for UpKind {
    #[inline]
    fn drop(&mut self) {
        if self.0 != usize::MAX {
            UP_KIND.with(|c| c.set(self.0));
        }
    }
}

/// Per page of the span: (flip + 1) << 32 | its generation at its last
/// upload; 0: never uploaded.
static UP_LAST: std::sync::OnceLock<Box<[AtomicU64]>> = std::sync::OnceLock::new();
/// By caller and path: [pages uploaded, of them unchanged since an upload
/// in the same flip, unchanged since their last upload in an earlier flip].
static UP_COUNTS: [[[AtomicU64; 3]; 5]; 4] = [const { [const { [const { AtomicU64::new(0) }; 3] }; 5] }; 4];
/// The flip being counted and the flips with the most pages uploaded again
/// (`up_census_on`).
struct UpFlips {
    flip: u64,
    pages: u64,
    again: u64,
    /// This flip's pages uploaded again by caller and path.
    by: [[u64; 5]; 4],
    /// (again, pages, flip, its largest callers and paths), most first.
    worst: Vec<(u64, u64, u64, String)>,
    flips: u64,
}

static UP_FLIPS: Mutex<UpFlips> = Mutex::new(UpFlips { flip: 0, pages: 0, again: 0, by: [[0; 5]; 4], worst: Vec::new(), flips: 0 });

/// Pages [a, b) are uploaded into the mirror by this thread's batch
/// (`up_census_on`).
pub fn upload_census(a: u64, b: u64) {
    crate::engine::note(crate::engine::C_UPLOAD, a, b.saturating_sub(a));
    if hold_census() && HELD_MAP.get().is_some() {
        // (The hold census: pages uploaded while held.)
        let n = (a & !(PAGE - 1)..b).step_by(PAGE as usize).filter(|&p| crate::hosted::held(p)).count() as u64;
        if n > 0 {
            HOLD[H_UPLOADED].fetch_add(n, Ordering::Relaxed);
        }
    }
    if !up_census_on() {
        return;
    }
    // (Zeroed pages: only those of pages uploaded become resident.)
    let last = UP_LAST.get_or_init(|| unsafe { Box::<[AtomicU64]>::new_zeroed_slice(((MIRROR_END - MIRROR_BASE) / PAGE) as usize).assume_init() });
    let flip = super::FRAMES.load(Ordering::Relaxed) + 1;
    let (mut n, mut again, mut later) = (0u64, 0u64, 0u64);
    let mut p = (a & !(PAGE - 1)).max(MIRROR_BASE);
    let hi = b.min(MIRROR_END);
    while p < hi {
        let pg = chunk(p, false).map_or(0, |c| c.pgen[idx(p)].load(Ordering::Acquire));
        let old = last[((p - MIRROR_BASE) / PAGE) as usize].swap(flip << 32 | pg as u64, Ordering::AcqRel);
        n += 1;
        if old != 0 && old as u32 == pg {
            if old >> 32 == flip {
                again += 1;
            } else {
                later += 1;
            }
        }
        p += PAGE;
    }
    let (caller, kind) = (CALLER.with(|c| c.get()), UP_KIND.with(|c| c.get()));
    let c = &UP_COUNTS[caller][kind];
    c[0].fetch_add(n, Ordering::Relaxed);
    c[1].fetch_add(again, Ordering::Relaxed);
    c[2].fetch_add(later, Ordering::Relaxed);
    let mut g = UP_FLIPS.lock();
    if g.flip != flip {
        if g.flip != 0 {
            // (Its three largest callers and paths of pages uploaded again.)
            let mut by: Vec<(u64, usize, usize)> = (0..4).flat_map(|i| (0..5).map(move |k| (i, k))).map(|(i, k)| (g.by[i][k], i, k)).filter(|x| x.0 != 0).collect();
            by.sort_unstable_by(|x, y| y.cmp(x));
            let top: Vec<String> = by.iter().take(3).map(|&(v, i, k)| format!("{} {} {}", CALLER_NAMES[i], UP_KINDS[k], v)).collect();
            let row = (g.again, g.pages, g.flip, top.join(", "));
            g.worst.push(row);
            g.worst.sort_unstable_by(|x, y| (y.0, y.1, y.2).cmp(&(x.0, x.1, x.2)));
            g.worst.truncate(8);
            g.flips += 1;
        }
        (g.flip, g.pages, g.again, g.by) = (flip, 0, 0, [[0; 5]; 4]);
    }
    g.pages += n;
    g.again += again;
    g.by[caller][kind] += again;
}

/// Run-shared uploads (`Exec::fe2_share`): per page of the span, the
/// parallel run and part whose batch claimed it for its copy and the page's
/// generation then (`Chunk::pgen`, low 16 bits): run << 24 | part << 16 |
/// generation; 0: none. Of a run, the earliest part's claim is kept.
static RUN_UP: std::sync::OnceLock<Box<[AtomicU64]>> = std::sync::OnceLock::new();
/// Per 64-page word of the span, the run that last claimed a page there
/// (`run_claim`): a lookup for another run needs no page entry (every page a
/// check needs is looked up, and at an arrival the page entries' loads cost
/// the workers ~6% more CPU a flip; 2026-10-31). Runs of other queues
/// claiming in the same word take it from each other: their lookups then
/// miss, and those pages are copied again, as without sharing.
static RUN_WORD: std::sync::OnceLock<Box<[AtomicU64]>> = std::sync::OnceLock::new();
/// Runs that claimed a page, by their low bits (`run_any`): a check by a
/// part of a run without claims passes its pages by whole runs as a batch
/// without uploads does (`ensure`'s `mine_any`).
static RUN_ANY: [AtomicU64; 256] = [const { AtomicU64::new(0) }; 256];
/// [page lookups that found a page claimed by an earlier part or batch of
/// the run, pages claimed, pages left to an earlier part's claim].
static RUN_SHARED: [AtomicU64; 3] = [const { AtomicU64::new(0) }; 3];
/// Parts past this share nothing (the entry's field is 8 bits).
pub const RUN_CHUNKS: u32 = 254;

fn run_up() -> &'static [AtomicU64] {
    // (Zeroed pages: only those of pages a run uploads become resident.)
    RUN_UP.get_or_init(|| unsafe { Box::<[AtomicU64]>::new_zeroed_slice(((MIRROR_END - MIRROR_BASE) / PAGE) as usize).assume_init() })
}

fn run_word() -> &'static [AtomicU64] {
    RUN_WORD.get_or_init(|| unsafe { Box::<[AtomicU64]>::new_zeroed_slice(((MIRROR_END - MIRROR_BASE) / PAGE / 64) as usize).assume_init() })
}

#[inline]
fn page_gen16(p: u64) -> u64 {
    chunk(p, false).map_or(0, |c| c.pgen[idx(p)].load(Ordering::Acquire) as u64 & 0xffff)
}

/// Claims pages [a, b) (page-aligned) for the batch of part `chunk` of
/// parallel run `run` (`Exec::fe2_share`; `chunk` at most RUN_CHUNKS;
/// part 0 the main command processor's own, before the workers' chunks): a
/// page an earlier part or batch of the run claimed, unchanged since, is
/// left to that claim; the others are claimed and pushed onto `out` as runs,
/// for copies into this batch's prologue. One atomic step per page: the
/// workers that followed one take wake together, and looked up before any of
/// them had claimed, each would copy every page. Pages outside the span are
/// pushed as they are.
pub fn run_claim(a: u64, b: u64, run: u64, chunk: u32, out: &mut Vec<(u64, u64)>) {
    let mut push = |x: u64, y: u64| match out.last_mut() {
        Some(l) if l.1 == x => l.1 = y,
        _ => out.push((x, y)),
    };
    if a < MIRROR_BASE || b > MIRROR_END {
        push(a, b);
        return;
    }
    let t = run_up();
    let words = run_word();
    let r = run & 0xff_ffff_ffff;
    let any = &RUN_ANY[(r & 255) as usize];
    if any.load(Ordering::Relaxed) != r {
        any.store(r, Ordering::Release);
    }
    let (mut claimed, mut left) = (0u64, 0u64);
    let mut p = a;
    while p < b {
        // (The word's tag before the page's claim: a lookup that sees the
        // claim sees the tag.)
        let w = &words[((p - MIRROR_BASE) / PAGE / 64) as usize];
        if w.load(Ordering::Relaxed) != r {
            w.store(r, Ordering::Release);
        }
        let g = page_gen16(p);
        let new = r << 24 | (chunk as u64) << 16 | g;
        let e = &t[((p - MIRROR_BASE) / PAGE) as usize];
        let won = e
            .fetch_update(Ordering::AcqRel, Ordering::Acquire, |old| {
                let earlier = old >> 24 == r && (old >> 16 & 0xff) as u32 <= chunk && old & 0xffff == g;
                (!earlier).then_some(new)
            })
            .is_ok();
        if won {
            claimed += 1;
            push(p, p + PAGE);
        } else {
            left += 1;
        }
        if traced(p, p + PAGE).is_some() {
            trace_event(p, &format!("run {r:#x} part {chunk}: {} (page generation {g})", if won { "claimed for this part's copy" } else { "left to an earlier part's copy" }));
        }
        p += PAGE;
    }
    RUN_SHARED[1].fetch_add(claimed, Ordering::Relaxed);
    RUN_SHARED[2].fetch_add(left, Ordering::Relaxed);
}

/// Whether a part of parallel run `run` claimed a page (`run_claim`; false
/// as well once a run 256 ids later claimed one).
#[inline]
pub fn run_any(run: u64) -> bool {
    RUN_ANY[(run & 255) as usize].load(Ordering::Acquire) == run & 0xff_ffff_ffff
}

/// Whether page `p` is claimed by a batch of part `chunk` or an earlier part
/// of parallel run `run`, unchanged since (`run_claim`):
/// that batch is queued before the asking one, its copies landing first.
#[inline]
pub fn run_shared(p: u64, run: u64, chunk: u32) -> bool {
    let (Some(t), Some(words)) = (RUN_UP.get(), RUN_WORD.get()) else { return false };
    if !(MIRROR_BASE..MIRROR_END).contains(&p) {
        return false;
    }
    let i = ((p - MIRROR_BASE) / PAGE) as usize;
    if words[i / 64].load(Ordering::Acquire) != run & 0xff_ffff_ffff {
        return false;
    }
    let v = t[i].load(Ordering::Acquire);
    let hit = v >> 24 == run & 0xff_ffff_ffff && (v >> 16 & 0xff) as u32 <= chunk && v & 0xffff == page_gen16(p);
    if hit {
        // (Counted per thread: every worker adding to one word made the
        // lookups slow.)
        RUN_HITS.with(|c| {
            let n = c.get() + 1;
            if n >= 4096 {
                RUN_SHARED[0].fetch_add(n, Ordering::Relaxed);
                c.set(0);
            } else {
                c.set(n);
            }
        });
    }
    hit
}

thread_local! {
    /// This thread's lookups served not yet added to RUN_SHARED.
    static RUN_HITS: std::cell::Cell<u64> = const { std::cell::Cell::new(0) };
}

/// Status part: run-shared uploads (`Exec::fe2_share`).
fn run_share_line() -> String {
    let [hits, claimed, left] = [0, 1, 2].map(|k| RUN_SHARED[k].load(Ordering::Relaxed));
    if hits == 0 && claimed == 0 {
        return String::new();
    }
    format!("    front end 2's run-shared uploads (`Exec::fe2_share`): {claimed} pages claimed by parallel runs' parts for their copies, {left} left to an earlier part's claim at the copy; {hits} page lookups found a page claimed by an earlier part or batch of the run\n")
}

/// Clears the upload census's counts (`profile reset`; the pages' last
/// uploads stay).
fn reset_up_census() {
    for c in UP_COUNTS.iter().flatten().flatten() {
        c.store(0, Ordering::Relaxed);
    }
    *UP_FLIPS.lock() = UpFlips { flip: 0, pages: 0, again: 0, by: [[0; 5]; 4], worst: Vec::new(), flips: 0 };
}

/// The upload census (`DS_FE2_UPCENSUS=1`; "" otherwise).
fn up_census_line() -> String {
    if !up_census_on() {
        return String::new();
    }
    let mib = |pages: u64| pages as f64 * PAGE as f64 / (1u64 << 20) as f64;
    let mut parts = Vec::new();
    let mut all = [0u64; 3];
    for (name, rows) in CALLER_NAMES.iter().zip(UP_COUNTS.iter()) {
        let mut kinds = Vec::new();
        for (kind, c) in UP_KINDS.iter().zip(rows.iter()) {
            let v: Vec<u64> = c.iter().map(|x| x.load(Ordering::Relaxed)).collect();
            if v[0] == 0 {
                continue;
            }
            for k in 0..3 {
                all[k] += v[k];
            }
            kinds.push(format!("{kind} {} pages ({:.0} MiB; unchanged since an upload in the flip {}, since one before it {})", v[0], mib(v[0]), v[1], v[2]));
        }
        if !kinds.is_empty() {
            parts.push(format!("{name}: {}", kinds.join(", ")));
        }
    }
    let g = UP_FLIPS.lock();
    let flips = g.flips.max(1);
    let worst: Vec<String> = g.worst.iter().map(|(again, pages, flip, top)| format!("flip {} {} of {} pages ({:.0} of {:.0} MiB; {top})", flip - 1, again, pages, mib(*again), mib(*pages))).collect();
    format!(
        "    front end 2's uploads into the mirror (`DS_FE2_UPCENSUS`): {} pages ({:.1} MiB a flip over {} flips), {} unchanged since an upload in the same flip ({:.1} MiB a flip), {} since one in an earlier flip; by caller and path: {}; most uploaded again in a flip: {}\n",
        all[0],
        mib(all[0]) / flips as f64,
        flips,
        all[1],
        mib(all[1]) / flips as f64,
        all[2],
        parts.join("; "),
        worst.join(", ")
    )
}

/// Flips without a CPU write found (`Chunk::wrote`; a page's first sight
/// counts as one) that make a GPU-written page quiet (`quiet_census_on`).
const QUIET_FLIPS: u64 = 300;

/// Per caller: [takes, ns, quiet takes, their ns] (`quiet_census_on`).
static QUIET_TAKES: [[AtomicU64; 4]; 4] = [const { [const { AtomicU64::new(0) }; 4] }; 4];

/// Whether every page of [a, b) is GPU-written and quiet (`QUIET_FLIPS`).
fn quiet_range(a: u64, b: u64, frame: u64) -> bool {
    let mut x = a;
    while x < b {
        let Some(c) = chunk(x, false) else { return false };
        let i = idx(x);
        if c.gpu[i / 64].load(Ordering::Relaxed) >> (i % 64) & 1 == 0 {
            return false;
        }
        let w = c.wrote[i].load(Ordering::Relaxed) as u64;
        if w != 0 && frame.saturating_sub(w - 1) < QUIET_FLIPS {
            return false;
        }
        x += PAGE;
    }
    true
}

/// A placement took `cycles` (TSC; `DS_FE2_CALLERS`).
pub fn note_place(cycles: u64) {
    caller_count(C_PLACES, 1);
    caller_count(C_PLACE_CYCLES, cycles);
}
const C_CHECKS: usize = 0;
const C_HITS: usize = 1;
const C_NO_RANGE: usize = 2;
const C_EPOCH: usize = 3;
const C_GENS: usize = 4;
const C_HOST: usize = 5;
const C_WORDS: usize = 6;
const C_TAKES: usize = 7;
const C_CYCLES: usize = 8;

thread_local! {
    static CALLER: std::cell::Cell<usize> = const { std::cell::Cell::new(3) };
}

/// The caller class of an executor named `name` (`DS_FE2_CALLERS`).
pub fn set_caller(name: &str) {
    let k = match name {
        "gpu-gfx" => 0,
        n if n.starts_with("gpu-gfx-par") => 1,
        "gpu-acb0x50" => 2,
        _ => 3,
    };
    CALLER.with(|c| c.set(k));
}

fn caller_count(what: usize, n: u64) {
    let k = CALLER.with(|c| c.get());
    BY_CALLER[k][what].fetch_add(n, Ordering::Relaxed);
    if FIRST_COMMAND.with(|f| f.get()) {
        BY_CALLER_FIRST[k][what].fetch_add(n, Ordering::Relaxed);
    }
}

fn note_range_fine(lo: u64, hi: u64, gens: [u32; FINE_CHUNKS], host: [u64; FINE_LEAVES], ep: u32, marked: bool) {
    let f = Fine { lo, hi, gens, host, ep, marked };
    if exact_on() {
        EXACT.with(|t| {
            let mut t = t.borrow_mut();
            if t.keys.is_empty() {
                t.keys.resize(EXACT_SLOTS, (0, 0));
                t.fines.resize(EXACT_SLOTS, FINE_NONE);
            }
            let k = exact_slot(lo, hi);
            t.keys[k] = (lo, hi);
            t.fines[k] = f;
        });
    }
    RANGES.with(|r| {
        let mut r = r.borrow_mut();
        // (Ranges the new one covers dropped, as the list's retain did.)
        for i in 0..RANGE_CACHE {
            let (flo, fhi) = r.keys[i];
            if fhi != 0 && lo <= flo && fhi <= hi && (marked || !r.fines[i].marked) {
                r.keys[i].1 = 0;
            }
        }
        let head = (r.head + RANGE_CACHE - 1) % RANGE_CACHE;
        r.head = head;
        r.keys[head] = (lo, hi);
        r.fines[head] = f;
    });
}

static TABLE: [AtomicPtr<Chunk>; CHUNKS] = [const { AtomicPtr::new(std::ptr::null_mut()) }; CHUNKS];
static ON: AtomicBool = AtomicBool::new(false);

/// Front end 2 wanted: the default since 2026-10-30 (the user's direction);
/// `DS_FE2=0` runs the old front end, kept as a fallback only. Read once:
/// shader specs follow it, so the pipelines precompiled at a session's start
/// match.
pub fn wanted() -> bool {
    static W: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *W.get_or_init(|| std::env::var("DS_FE2").ok().as_deref() != Some("0"))
}

/// Pages GPU work writes are armed like others (GPU writes reach guest
/// memory by the write-back, which does not fault, and M holds them
/// already): the default since 2026-11-05 afternoon, with the user's
/// approval, after 16 armed 8-2 boots, three armed 8-level tours and an
/// armed player suite without a loss or a broken level once the write-back's
/// hash skip was off (the morning's armed failures all had it on; October's
/// losses predate it, cause open: FINDINGS.md 2026-11-05). It answers ~70% of
/// the graphics CP's takes with the host's bit. `DS_FE2_ARM_GPU=0`: never
/// armed (the fallback; they are the pages the GPU writes through their
/// imports); `=2`: protected but kept unarmed in front end 2's records
/// (`arm_gpu_os`, diagnosis).
fn arm_gpu() -> bool {
    static W: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *W.get_or_init(|| !matches!(std::env::var("DS_FE2_ARM_GPU").ok().as_deref(), Some("0") | Some("2")))
}

/// `DS_FE2_ARM_GPU=2` (diagnosis of the device losses): the pages
/// arming would arm are protected through the host as armed pages
/// are, the GPU-written ones among them kept unarmed in front end 2's own
/// records (marked GPU-written, taken in every epoch as by default): the
/// protection of pages the GPU writes through their imports, without
/// arming's other changes.
fn arm_gpu_os() -> bool {
    static W: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *W.get_or_init(|| std::env::var("DS_FE2_ARM_GPU").ok().as_deref() == Some("2"))
}

/// Diagnosis (`DS_FE2_CHECK_COPIES=1`): copy kernels' dispatches outside the
/// graphics CP checked after their batches (`Exec::dispatch_inner`).
pub fn check_copies() -> bool {
    static W: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *W.get_or_init(|| std::env::var("DS_FE2_CHECK_COPIES").ok().as_deref() == Some("1"))
}

/// `DS_FE2_STRETCH_AUDIT=1` (diagnosis): at its label every stretch past a
/// label wait whose checks needed an older epoch than its label does is
/// checked again (`Exec::past_current_for`) and counted stale or not, as
/// waits passed on GPU writes check them, but kept either way.
pub fn stretch_audit() -> bool {
    static W: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *W.get_or_init(|| std::env::var("DS_FE2_STRETCH_AUDIT").ok().as_deref() == Some("1"))
}

/// Held past a label wait, the graphics command processor's checks need the
/// epoch now, not its view (`DS_FE2_STRETCH_VIEW=1`: the view, as before
/// 2026-10-28 night's fix; `Exec::fe2_place_inner`).
pub fn stretch_epoch() -> bool {
    static W: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *W.get_or_init(|| std::env::var("DS_FE2_STRETCH_VIEW").ok().as_deref() != Some("1"))
}

/// Small runs are uploaded without a take while their pages are within their
/// quiet time, not only HOT_FLIPS after a write (`hot_run`; `DS_FE2_HOT_QUIET=0`:
/// HOT_FLIPS only, as before 2026-10-28).
fn hot_quiet() -> bool {
    static W: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *W.get_or_init(|| std::env::var("DS_FE2_HOT_QUIET").ok().as_deref() != Some("0"))
}

/// Where pages may be armed (the executor's import list): calls its third
/// argument with the parts of [a, b) inside settled import windows (asked
/// again under the arm lock: `arm_part`); false: the list was busy.
pub type Armable = dyn Fn(u64, u64, &mut dyn FnMut(u64, u64)) -> bool;

/// Front end 2 runs (wanted, and its self-test passed).
#[inline]
pub fn on() -> bool {
    ON.load(Ordering::Acquire)
}

/// Whether guest [base, base+size) lies in the mirror's span.
#[inline]
pub fn in_span(base: u64, size: u64) -> bool {
    base >= MIRROR_BASE && base.saturating_add(size) <= MIRROR_END
}

#[inline]
fn chunk(addr: u64, create: bool) -> Option<&'static Chunk> {
    if !(MIRROR_BASE..MIRROR_END).contains(&addr) {
        return None;
    }
    let slot = &TABLE[((addr - MIRROR_BASE) >> CHUNK_SHIFT) as usize];
    let p = slot.load(Ordering::Acquire);
    if !p.is_null() {
        return Some(unsafe { &*p });
    }
    if !create {
        return None;
    }
    Some(make_chunk(slot))
}

/// A new chunk in `slot` (or the one another thread put there first); out
/// of line: its initializer's frame would make every `chunk` call probe the
/// stack.
#[cold]
#[inline(never)]
fn make_chunk(slot: &'static AtomicPtr<Chunk>) -> &'static Chunk {
    let new = Box::into_raw(Box::new(Chunk {
        valid: std::array::from_fn(|_| AtomicU64::new(0)),
        armed: std::array::from_fn(|_| AtomicU64::new(0)),
        varmed: std::array::from_fn(|_| AtomicU64::new(0)),
        gpu: std::array::from_fn(|_| AtomicU64::new(0)),
        gpu_all: AtomicBool::new(false),
        wclean: std::array::from_fn(|_| AtomicU32::new(0)),
        wmax: std::array::from_fn(|_| AtomicU32::new(0)),
        wgen: std::array::from_fn(|_| AtomicU32::new(0)),
        clean: (0..CHUNK_PAGES).map(|_| AtomicU32::new(0)).collect(),
        wrote: (0..CHUNK_PAGES).map(|_| AtomicU32::new(0)).collect(),
        pgen: (0..CHUNK_PAGES).map(|_| AtomicU32::new(0)).collect(),
        level: (0..CHUNK_PAGES).map(|_| AtomicU8::new(0)).collect(),
        uses: std::array::from_fn(|_| AtomicU8::new(0)),
        use_ep: std::array::from_fn(|_| AtomicU32::new(0)),
        seen: std::array::from_fn(|_| AtomicU64::new(0)),
        cgen: Line(AtomicU32::new(0)),
        sum: AtomicU64::new(0),
        sum_host: AtomicU64::new(0),
        hold_until: AtomicU64::new(0),
        hold_level: AtomicU8::new(0),
    }));
    match slot.compare_exchange(std::ptr::null_mut(), new, Ordering::AcqRel, Ordering::Acquire) {
        Ok(_) => unsafe { &*new },
        Err(cur) => {
            drop(unsafe { Box::from_raw(new) });
            unsafe { &*cur }
        }
    }
}

const CHUNK_BYTES: u64 = 1 << CHUNK_SHIFT;

/// Whether chunk `c` (at `a`) has a summary that meets `want`.
#[inline]
fn summary_ok(c: &Chunk, a: u64, want: u32) -> bool {
    let s = c.sum.load(Ordering::Acquire);
    if s == 0 || (s >> 32) as u32 != c.cgen.load(Ordering::Acquire) {
        return false;
    }
    let e = s as u32;
    e >= want && c.sum_host.load(Ordering::Acquire) == crate::hosted::leaf_events(a)
}

/// Page index of `addr` in its chunk.
#[inline]
fn idx(addr: u64) -> usize {
    ((addr >> 12) as usize) & (CHUNK_PAGES - 1)
}

/// The bits of pages [a, e) within the 64-page word holding `a` (e at most
/// that word's end).
#[inline]
fn word_mask(a: u64, e: u64) -> u64 {
    let first = idx(a) % 64;
    let n = ((e - a) / PAGE) as usize;
    if n >= 64 {
        u64::MAX
    } else {
        ((1u64 << n) - 1) << first
    }
}

/// The end of the 64-page word holding `a`, at most `hi`.
#[inline]
fn word_end(a: u64, hi: u64) -> u64 {
    ((a & !(64 * PAGE - 1)) + 64 * PAGE).min(hi)
}

/// Counters (`status`).
static STATS: [AtomicU64; 34] = [const { AtomicU64::new(0) }; 34];
const S_ENSURES: usize = 0;
const S_ENSURE_PAGES: usize = 1;
const S_FAST: usize = 2;
const S_TAKES: usize = 3;
const S_TAKE_WRITTEN: usize = 4;
const S_UPLOADS: usize = 5;
const S_UPLOAD_PAGES: usize = 6;
const S_UPLOAD_FAILED: usize = 7;
const S_ARM_RUNS: usize = 8;
const S_ARMED: usize = 9;
const S_ARM_ENDED: usize = 10;
const S_ARM_FAILED: usize = 11;
const S_INVAL_CALLS: usize = 12;
const S_INVAL_TAKEN: usize = 13;
const S_SCATTERS: usize = 14;
const S_CP_MIRROR: usize = 15;
const S_VALIDATED: usize = 16;
const S_RACED: usize = 17;
const S_MEMCPY: usize = 19;
const S_UNSCATTERABLE: usize = 20;
const S_TAKE_NS: usize = 21;
const S_SLOW_NS: usize = 22;
const S_TAKE_FAILED: usize = 23;
/// Chunks a placement passed by their summary, placements its thread's
/// range cache passed.
const S_SUMMARY: usize = 24;
const S_RANGE_HITS: usize = 25;
/// Runs uploaded without a take (written lately) and their pages.
const S_HOT_RUNS: usize = 26;
const S_HOT_PAGES: usize = 27;
// (28-30: the checks deferred to batch closes, `DS_FE2_DEFER`, removed on
// 2026-11-01.)
// (31, 32: runs uploaded after another take of their epoch, `tune
// fe2_taken_upload`, removed on 2026-11-01.)
/// View pages the residency armed at its takes, armed here too (`adopt_armed`).
const S_ADOPTED: usize = 33;

/// Placement checks by size (pages: <16, <256, <4096, more): [checks,
/// pages, takes, take ns].
static BY_SIZE: [[AtomicU64; 4]; 4] = [const { [const { AtomicU64::new(0) }; 4] }; 4];

fn size_class(pages: u64) -> usize {
    match pages {
        0..16 => 0,
        16..256 => 1,
        256..4096 => 2,
        _ => 3,
    }
}

/// Counts a write-back the executor recorded.
pub fn note_scatter() {
    STATS[S_SCATTERS].fetch_add(1, Ordering::Relaxed);
}

/// Counts a CP write the GPU performs that also went into the mirror.
pub fn note_cp_mirror() {
    STATS[S_CP_MIRROR].fetch_add(1, Ordering::Relaxed);
}

/// Pages of [lo, hi) GPU work wrote (as far as front end 2 knows: its
/// write marks' pages; `fe2census`).
pub fn gpu_pages(lo: u64, hi: u64) -> u64 {
    let (mut x, hi) = ((lo & !(PAGE - 1)).max(MIRROR_BASE), hi.min(MIRROR_END));
    let mut n = 0;
    while x < hi {
        let e = word_end(x, hi);
        if let Some(c) = chunk(x, false) {
            n += (c.gpu[idx(x) / 64].load(Ordering::Relaxed) & word_mask(x, e)).count_ones() as u64;
        }
        x = e;
    }
    n
}

/// Counts pages copied into the mirror by the CPU (no import window).
pub fn note_memcpy(pages: u64) {
    STATS[S_MEMCPY].fetch_add(pages, Ordering::Relaxed);
}

/// A page of a view window the residency armed through the host at a take
/// (`resident::take_written`): armed for front end 2 as if armed here. Its
/// own arming refuses such pages (read-only already), so without this they
/// were taken again after every view raise (2026-10-31, views boot
/// -135103). A take arms a page only after reporting it or finding it clean
/// and armed, and every take's written pages reach `marks_taken`, so the
/// host's bit set at a take leaves no write unseen; a fault clears it, and
/// `watch` ends the arming here as for pages armed here.
pub fn adopt_armed(page: u64) {
    if !on() {
        return;
    }
    let Some(c) = chunk(page, false) else { return };
    let i = idx(page);
    c.armed[i / 64].fetch_or(1u64 << (i % 64), Ordering::AcqRel);
    STATS[S_ADOPTED].fetch_add(1, Ordering::Relaxed);
    if traced(page, page + PAGE).is_some() {
        trace_event(page, "armed (adopted from the host)");
    }
}

/// Counts written bindings whose pages no import window covers (their GPU
/// writes cannot go back).
pub fn note_unscatterable() {
    STATS[S_UNSCATTERABLE].fetch_add(1, Ordering::Relaxed);
}

/// Pages of [a, e) (one word) that need no check for requirement `want`:
/// valid, and armed (the host's bit still set) or found clean at or after
/// epoch `want`.
#[inline]
fn word_ok(c: &Chunk, a: u64, e: u64, want: u32) -> u64 {
    let w = idx(a) / 64;
    let mask = word_mask(a, e);
    let v = c.valid[w].load(Ordering::Acquire) & mask;
    if v == 0 {
        return 0;
    }
    let ours = c.armed[w].load(Ordering::Acquire) & v;
    let armed = if ours != 0 { ours & crate::hosted::armed_word(a & !(64 * PAGE - 1)) } else { 0 };
    let rest = v & !armed;
    if rest == 0 {
        return armed;
    }
    if c.wclean[w].load(Ordering::Acquire) >= want {
        return v;
    }
    // (A stamp stored after this read only makes the check conservative.)
    if c.wmax[w].load(Ordering::Acquire) < want {
        return armed;
    }
    let mut ok = armed;
    let base = w * 64;
    let mut r = rest;
    while r != 0 {
        let b = r.trailing_zeros() as usize;
        r &= r - 1;
        if c.clean[base + b].load(Ordering::Acquire) >= want {
            ok |= 1u64 << b;
        }
    }
    ok
}

/// The mirror's pages owned by engine allocator `owner` whose CPU write
/// front end 2 found within the last `n` flips (`engine`'s allocation census).
pub fn hot_pages_of(owner: u8, n: u64) -> Vec<u64> {
    let frame = super::FRAMES.load(Ordering::Relaxed);
    let mut out = Vec::new();
    for k in 0..CHUNKS {
        let p = TABLE[k].load(Ordering::Acquire);
        if p.is_null() {
            continue;
        }
        let c = unsafe { &*p };
        for i in 0..CHUNK_PAGES {
            let w = c.wrote[i].load(Ordering::Relaxed) as u64;
            if w == 0 || frame.saturating_sub(w - 1) >= n {
                continue;
            }
            let a = MIRROR_BASE + ((k as u64) << CHUNK_SHIFT) + i as u64 * PAGE;
            if crate::engine::owner(a) == owner {
                out.push(a);
            }
        }
    }
    out
}

/// Front end 4's census (`engine report`, docs/FRONTEND3.md's coherence per
/// epoch): the mirror's pages whose CPU write front end 2 found within the
/// last `n` flips (`Chunk::wrote`), by the engine allocator owning them
/// (`engine::owner`): pages, the 1 MiB granules and runs (of pages with one
/// owner) they form, and for the GPU arena how many lie inside its frame
/// allocations (`engine::arena_allocated`).
pub fn hot_by_owner(n: u64) -> String {
    let frame = super::FRAMES.load(Ordering::Relaxed);
    let arena = crate::engine::arena_owner();
    // owner -> [pages, granules, runs, pages inside the arena's allocations]
    let mut by: std::collections::BTreeMap<u8, [u64; 4]> = std::collections::BTreeMap::new();
    let mut last_mib: std::collections::HashMap<u8, u64> = std::collections::HashMap::new();
    let mut prev: Option<(u64, u8)> = None;
    for k in 0..CHUNKS {
        let p = TABLE[k].load(Ordering::Acquire);
        if p.is_null() {
            continue;
        }
        let c = unsafe { &*p };
        for i in 0..CHUNK_PAGES {
            let w = c.wrote[i].load(Ordering::Relaxed) as u64;
            if w == 0 || frame.saturating_sub(w - 1) >= n {
                continue;
            }
            let a = MIRROR_BASE + ((k as u64) << CHUNK_SHIFT) + i as u64 * PAGE;
            let o = crate::engine::owner(a);
            let e = by.entry(o).or_insert([0; 4]);
            e[0] += 1;
            if last_mib.insert(o, a >> 20) != Some(a >> 20) {
                e[1] += 1;
            }
            if prev != Some((a - PAGE, o)) {
                e[2] += 1;
            }
            if o == arena && arena != 0 && crate::engine::arena_allocated(a) {
                e[3] += 1;
            }
            prev = Some((a, o));
        }
    }
    let names = crate::engine::names();
    let parts: Vec<String> = by
        .iter()
        .map(|(o, v)| {
            let name = names.get(o).cloned().unwrap_or_else(|| if *o == 0 { "none".into() } else { "?".into() });
            let inside = if *o == arena && arena != 0 { format!(", {} inside its frame allocations", v[3]) } else { String::new() };
            format!("{o} {name}: {} pages ({:.1} MiB) in {} granules of 1 MiB, {} runs{inside}", v[0], v[0] as f64 / 256.0, v[1], v[2])
        })
        .collect();
    format!("    front end 2's pages with a CPU write found within {n} flips, by owning engine allocator: {}\n", parts.join("; "))
}

/// The checks' commonest counts, kept per thread and added to STATS and
/// BY_SIZE every LOCAL_FLUSH checks (shared atomics on every check, from
/// every recording thread, bounced their cache lines: ~45,000 checks a flip
/// at the vista, four to six adds each). Slots: checks, their pages, range
/// cache hits, checks needing nothing, chunks passed by their summary, then
/// checks and pages per size class.
const L_CHECKS: usize = 0;
const L_PAGES: usize = 1;
const L_HITS: usize = 2;
const L_FAST: usize = 3;
const L_SUMMARY: usize = 4;
const L_CLASS: usize = 5;
const LOCAL_SLOTS: usize = L_CLASS + 8;
const LOCAL_FLUSH: u64 = 256;

thread_local! {
    static LOCAL: std::cell::RefCell<[u64; LOCAL_SLOTS]> = const { std::cell::RefCell::new([0; LOCAL_SLOTS]) };
}

#[inline]
fn local_add(k: usize, n: u64) {
    LOCAL.with(|l| l.borrow_mut()[k] += n);
}

/// A check of `pages` pages in size class `class` begins (the thread's
/// counts go to the shared ones every LOCAL_FLUSH checks).
#[inline]
fn local_check(class: usize, pages: u64) {
    LOCAL.with(|l| {
        let mut l = l.borrow_mut();
        l[L_CHECKS] += 1;
        l[L_PAGES] += pages;
        l[L_CLASS + 2 * class] += 1;
        l[L_CLASS + 2 * class + 1] += pages;
        if l[L_CHECKS] >= LOCAL_FLUSH {
            for (k, s) in [(L_CHECKS, S_ENSURES), (L_PAGES, S_ENSURE_PAGES), (L_HITS, S_RANGE_HITS), (L_FAST, S_FAST), (L_SUMMARY, S_SUMMARY)] {
                STATS[s].fetch_add(std::mem::take(&mut l[k]), Ordering::Relaxed);
            }
            for c in 0..4 {
                BY_SIZE[c][0].fetch_add(std::mem::take(&mut l[L_CLASS + 2 * c]), Ordering::Relaxed);
                BY_SIZE[c][1].fetch_add(std::mem::take(&mut l[L_CLASS + 2 * c + 1]), Ordering::Relaxed);
            }
        }
    });
}

/// `DS_SLOW_PLACE=<us>` (diagnosis; 0 off): placements taking at least that
/// long are reported with what their checks did (`probe_take`,
/// `exec::note_slow_place`).
pub fn slow_place_us() -> u64 {
    static S: std::sync::OnceLock<u64> = std::sync::OnceLock::new();
    *S.get_or_init(|| std::env::var("DS_SLOW_PLACE").ok().and_then(|x| x.trim().parse().ok()).unwrap_or(0))
}

/// What this thread's checks did since `probe_take` (`DS_SLOW_PLACE`):
/// `PROBE_NAMES`.
pub const PROBE_N: usize = 16;
pub const PROBE_NAMES: [&str; PROBE_N] = ["checks", "range-cache hits", "words scanned", "chunks by summary", "runs to check", "their pages", "hot runs uploaded", "their pages", "runs after another's take", "runs taken", "their pages", "scan us", "runs us", "arming us", "takes us", "uploads and clean marks us"];
const P_CHECKS: usize = 0;
const P_HITS: usize = 1;
const P_WORDS: usize = 2;
const P_SUMMARY: usize = 3;
const P_RUNS: usize = 4;
const P_RUN_PAGES: usize = 5;
const P_HOT: usize = 6;
const P_HOT_PAGES: usize = 7;
const P_FOLLOW: usize = 8;
const P_TAKES: usize = 9;
const P_TAKE_PAGES: usize = 10;
const P_SCAN_NS: usize = 11;
const P_RUNS_NS: usize = 12;
const P_ARM_NS: usize = 13;
const P_TAKE_NS: usize = 14;
const P_FIX_NS: usize = 15;

thread_local! {
    static PROBE: std::cell::Cell<[u64; PROBE_N]> = const { std::cell::Cell::new([0; PROBE_N]) };
}

#[inline]
fn probe_add(k: usize, n: u64) {
    if slow_place_us() != 0 {
        PROBE.with(|p| {
            let mut v = p.get();
            v[k] += n;
            p.set(v);
        });
    }
}

/// This thread's probe counts since the last call (`DS_SLOW_PLACE`).
pub fn probe_take() -> [u64; PROBE_N] {
    PROBE.with(|p| p.replace([0; PROBE_N]))
}

thread_local! {
    /// Scratch of `ensure`: runs to check, pages a take reported.
    static SCRATCH: std::cell::RefCell<(Vec<(u64, u64)>, Vec<u64>)> = const { std::cell::RefCell::new((Vec::new(), Vec::new())) };
}

/// Makes M current over the pages of [lo, hi) (page-aligned, inside the
/// span) for GPU work recorded now in epoch `ep` (`precache::epoch`, low
/// bits). Pages not known current and not `mine` (uploaded by the caller's
/// batch: current for it) are checked for CPU writes by a take (armed first
/// where `armable` allows: the import window over them is settled), and
/// those M lacks are uploaded by `upload(a, b)` (the GPU copy of guest
/// [a, b) into M, recorded into the caller's prologue; false: it could not).
/// `gpu_writes`: GPU work will write the range (its pages are marked so,
/// and never armed with `DS_FE2_ARM_GPU=0`; by default it is checked like any).
/// Err: an upload failed (M lacks some of the range).
/// `want`: pages must reflect every CPU write before epoch `want`
/// began (the caller's view, `views_on`); `ep`: the epoch now, read before
/// any take (what takes stamp their pages with).
#[allow(clippy::too_many_arguments)]
pub fn ensure(lo: u64, hi: u64, want: u32, ep: u32, gpu_writes: bool, mine: &dyn Fn(u64) -> bool, mine_any: bool, upload: &mut dyn FnMut(u64, u64) -> bool, armable: &Armable) -> Result<(), &'static str> {
        let _cpu = crate::gpu::cpu_profile::scope(crate::gpu::cpu_profile::Part::Fe2Ensure);
    if !callers_on() {
        return ensure_inner(lo, hi, want, ep, gpu_writes, mine, mine_any, upload, armable);
    }
    let t0 = unsafe { core::arch::x86_64::_rdtsc() };
    caller_count(C_CHECKS, 1);
    source_count(B_CHECKS, 1);
    let r = ensure_inner(lo, hi, want, ep, gpu_writes, mine, mine_any, upload, armable);
    let cycles = unsafe { core::arch::x86_64::_rdtsc() }.wrapping_sub(t0);
    caller_count(C_CYCLES, cycles);
    source_count(B_CYCLES, cycles);
    r
}

thread_local! {
    /// The stage whose bindings this thread places (`StageSource`; for
    /// `DS_FE2_CALLERS`): 0 none, else `Prepared::ahead`.
    static SOURCE: std::cell::Cell<u8> = const { std::cell::Cell::new(0) };
}

const SOURCE_NAMES: [&str; 3] = ["placements of no stage", "stages it prepared", "stages a worker prepared"];
/// The graphics CP's checks by the stage they bind (`SOURCE`;
/// `DS_FE2_CALLERS`): [stage bindings, checks, range-cache hits, TSC cycles,
/// runs taken, their pages, runs only of pages a take of the epoch wanted
/// had looked at and found lacking, runs with a valid page not looked at
/// since that epoch began (and no lacking one), runs with a page M lacks not
/// looked at since; pages not looked at since: stamped one epoch before,
/// older].
static BY_SOURCE: [[AtomicU64; 11]; 3] = [const { [const { AtomicU64::new(0) }; 11] }; 3];
const B_BINDINGS: usize = 0;
const B_CHECKS: usize = 1;
const B_HITS: usize = 2;
const B_CYCLES: usize = 3;
const B_TAKES: usize = 4;
const B_PAGES: usize = 5;
const B_LACKING: usize = 6;
const B_VALID_STALE: usize = 7;
const B_INVALID_STALE: usize = 8;
const B_ONE_BEHIND: usize = 9;
const B_OLDER: usize = 10;

/// The stage whose bindings this thread places, until dropped (`SOURCE`).
pub struct StageSource;

impl StageSource {
    pub fn set(ahead: u8) -> StageSource {
        if callers_on() {
            SOURCE.with(|s| s.set(ahead.min(2)));
            source_count(B_BINDINGS, 1);
        }
        StageSource
    }
}

impl Drop for StageSource {
    fn drop(&mut self) {
        if callers_on() {
            SOURCE.with(|s| s.set(0));
        }
    }
}

/// Counts `n` for the graphics CP's checks of the stage being bound.
fn source_count(what: usize, n: u64) {
    if CALLER.with(|c| c.get()) == 0 {
        BY_SOURCE[SOURCE.with(|s| s.get()) as usize][what].fetch_add(n, Ordering::Relaxed);
    }
}

/// Why the pages of a run about to be taken need it, for a check wanting
/// epoch `want`: [pages M lacks that a take of that epoch looked at, valid
/// pages not looked at since, pages M lacks not looked at since, of those
/// two stamped one epoch before `want`] (pages in the run that need nothing,
/// merged by `TAKE_GAP`, are not counted).
fn take_reasons(a: u64, b: u64, want: u32) -> [u64; 4] {
    let mut k = [0u64; 4];
    let mut x = a;
    while x < b {
        let e = word_end(x, b);
        let Some(c) = chunk(x, false) else { break };
        let w = idx(x) / 64;
        let need = word_mask(x, e) & !word_ok(c, x, e, want);
        let valid = c.valid[w].load(Ordering::Acquire);
        let wclean = c.wclean[w].load(Ordering::Acquire);
        let mut r = need;
        while r != 0 {
            let bit = r.trailing_zeros() as usize;
            r &= r - 1;
            let stamp = wclean.max(c.clean[w * 64 + bit].load(Ordering::Acquire));
            if valid >> bit & 1 != 0 {
                k[1] += 1;
            } else if stamp >= want {
                k[0] += 1;
                continue;
            } else {
                k[2] += 1;
            }
            if stamp.wrapping_add(1) == want {
                k[3] += 1;
            }
        }
        x = e;
    }
    k
}

/// A run the graphics CP took, with `take_reasons`' counts.
fn note_source_take(a: u64, b: u64, k: [u64; 4]) {
    let c = &BY_SOURCE[SOURCE.with(|s| s.get()) as usize];
    c[B_TAKES].fetch_add(1, Ordering::Relaxed);
    c[B_PAGES].fetch_add((b - a) / PAGE, Ordering::Relaxed);
    let class = if k[2] != 0 {
        B_INVALID_STALE
    } else if k[1] != 0 {
        B_VALID_STALE
    } else {
        B_LACKING
    };
    c[class].fetch_add(1, Ordering::Relaxed);
    c[B_ONE_BEHIND].fetch_add(k[3], Ordering::Relaxed);
    c[B_OLDER].fetch_add(k[1] + k[2] - k[3], Ordering::Relaxed);
}

#[allow(clippy::too_many_arguments)]
fn ensure_inner(lo: u64, hi: u64, want: u32, ep: u32, gpu_writes: bool, mine: &dyn Fn(u64) -> bool, mine_any: bool, upload: &mut dyn FnMut(u64, u64) -> bool, armable: &Armable) -> Result<(), &'static str> {
    // (No page of the caller's batch to pass over: runs go in whole.)
    let mine_fast = !mine_any;
    let sc = size_class((hi - lo) / PAGE);
    local_check(sc, (hi - lo) / PAGE);
    let class = &BY_SIZE[sc];
    // Written ranges mark their pages GPU-written (kept unarmed) unless
    // `DS_FE2_ARM_GPU=0`; a range found fine with its marks set passes the
    // cache like a read one (marking every word on every use, ~430 words
    // for a pool of ~113 MB, made queue 0x50's dispatches slow).
    // (Pages of the mixed range's view windows too: their imports are of
    // the aliases, `mem::view_rules`.)
    let gpu_writes = gpu_writes && !arm_gpu() && !crate::mem::view_span(lo, hi);
    let tc = tsc();
    let fine = range_fine(lo, hi, want, gpu_writes);
    LAST_FINE.with(|c| c.set(if fine { 0 } else { 2 }));
    caller_since(C_CACHE, tc);
    probe_add(P_CHECKS, 1);
    if fine {
        probe_add(P_HITS, 1);
        local_add(L_HITS, 1);
        local_add(L_FAST, 1);
        if callers_on() {
            caller_count(C_HITS, 1);
            source_count(B_HITS, 1);
        }
        return Ok(());
    }
    // (Before the scan: a change during it shows as a new value.)
    let snap = snapshot(lo, hi);
    let ts = tsc();
    let probe_t = if slow_place_us() != 0 { Some(std::time::Instant::now()) } else { None };
    SCRATCH.with(|s| {
        let mut s = s.borrow_mut();
        let (bad, written) = &mut *s;
        bad.clear();
        // The pages that need a check, as runs (a whole chunk by its
        // summary when it has a current one).
        let mut a = lo;
        // (The chunk being scanned whole: its generation and host events
        // before the scan, and whether every page so far was ok and armed.)
        let mut whole: Option<(u64, u32, u64, bool, bool)> = None;
        // Every page of the range armed so far (the host's bit set), and no
        // page passed for being the caller's upload.
        let mut all_armed = true;
        let mut mine_none = true;
        // (Words looked at: `DS_FE2_CALLERS`.)
        let mut words = 0u64;
        while a < hi {
            if a & (CHUNK_BYTES - 1) == 0 && a + CHUNK_BYTES <= hi {
                let c = chunk(a, true).expect("inside the span");
                if (!gpu_writes || c.gpu_all.load(Ordering::Acquire)) && summary_ok(c, a, want) {
                    local_add(L_SUMMARY, 1);
                    probe_add(P_SUMMARY, 1);
                    if c.sum.load(Ordering::Acquire) as u32 != SUM_ARMED {
                        all_armed = false;
                    }
                    a += CHUNK_BYTES;
                    continue;
                }
                whole = Some((a, c.cgen.load(Ordering::Acquire), crate::hosted::leaf_events(a), true, true));
            }
            let e = word_end(a, hi);
            let c = chunk(a, true).expect("inside the span");
            let w = idx(a) / 64;
            words += 1;
            if gpu_writes {
                // (Set bits stay set: the chunks of a run start together and
                // scan the same pools, and a write per word made each scan's
                // words move between their cores.)
                let m = word_mask(a, e);
                if c.gpu[w].load(Ordering::Acquire) & m != m {
                    c.gpu[w].fetch_or(m, Ordering::AcqRel);
                }
            }
            let need = word_mask(a, e) & !word_ok(c, a, e, want);
            if all_armed && need == 0 {
                let armed = c.armed[w].load(Ordering::Acquire) & crate::hosted::armed_word(a & !(64 * PAGE - 1));
                all_armed = armed & word_mask(a, e) == word_mask(a, e);
            }
            if let Some(wh) = whole.as_mut() {
                if need != 0 {
                    wh.3 = false;
                } else if wh.4 {
                    // Every page of the word armed (the host's bit set)?
                    let armed = c.armed[w].load(Ordering::Acquire) & crate::hosted::armed_word(a & !(64 * PAGE - 1));
                    wh.4 = armed == u64::MAX;
                }
            }
            let mut r = need;
            while r != 0 {
                let b = r.trailing_zeros() as u64;
                // The run of set bits from b.
                let run = (!(r >> b)).trailing_zeros().min(64 - b as u32) as u64;
                let pa = (a & !(64 * PAGE - 1)) + b * PAGE;
                let pe = pa + run * PAGE;
                r &= !(if run >= 64 { u64::MAX } else { ((1u64 << run) - 1) << b });
                if mine_fast {
                    match bad.last_mut() {
                        Some(l) if l.1 + TAKE_GAP * PAGE >= pa => l.1 = pe,
                        _ => bad.push((pa, pe)),
                    }
                    continue;
                }
                // Pages the caller's batch uploaded need nothing.
                let mut x = pa;
                while x < pe {
                    if mine(x) {
                        mine_none = false;
                        x += PAGE;
                        continue;
                    }
                    let mut y = x + PAGE;
                    while y < pe && !mine(y) {
                        y += PAGE;
                    }
                    match bad.last_mut() {
                        Some(l) if l.1 + TAKE_GAP * PAGE >= x => l.1 = y,
                        _ => bad.push((x, y)),
                    }
                    x = y;
                }
            }
            a = e;
            // A chunk scanned whole without a page to check: its summary.
            if let Some((start, g, h, ok, armed)) = whole {
                if a == start + CHUNK_BYTES {
                    let c = chunk(start, false).expect("scanned");
                    // (Stored only when they change: threads rescanning a
                    // chunk kept its line moving between their cores.)
                    let sum = (g as u64) << 32 | if armed { SUM_ARMED } else { want } as u64;
                    if ok && (c.sum.load(Ordering::Relaxed) != sum || c.sum_host.load(Ordering::Relaxed) != h) {
                        c.sum_host.store(h, Ordering::Release);
                        c.sum.store(sum, Ordering::Release);
                    }
                    if gpu_writes && !c.gpu_all.load(Ordering::Relaxed) {
                        c.gpu_all.store(true, Ordering::Release);
                    }
                    whole = None;
                }
            }
        }
        if callers_on() {
            caller_count(C_WORDS, words);
            caller_since(C_SCAN, ts);
        }
        if let Some(t) = probe_t {
            probe_add(P_WORDS, words);
            probe_add(P_SCAN_NS, t.elapsed().as_nanos() as u64);
            probe_add(P_RUNS, bad.len() as u64);
            probe_add(P_RUN_PAGES, bad.iter().map(|r| (r.1 - r.0) / PAGE).sum());
        }
        let probe_r = if slow_place_us() != 0 { Some(std::time::Instant::now()) } else { None };
        let _probe_runs = ProbeRuns(probe_r);
        if bad.is_empty() {
            local_add(L_FAST, 1);
            LAST_FINE.with(|c| c.set(1));
            // (Not for ranges with pages the caller's batch uploaded: they are
            // fine for its batch only, until it is queued.)
            if let (true, Some((gens, host))) = (mine_none, snap) {
                note_range_fine(lo, hi, gens, host, if all_armed { SUM_ARMED } else { want }, gpu_writes);
            }
            return Ok(());
        }
        let t0 = std::time::Instant::now();
        let wl = work_laps();
        let rd = || if wl { unsafe { core::arch::x86_64::_rdtsc() } } else { 0 };
        let w0 = rd();
        let frame = super::FRAMES.load(Ordering::Relaxed);
        for &(a, b) in bad.iter() {
            let wr = rd();
            note_arming_ended(a, b);
            // Every page written lately: uploaded as it is, without a take.
            if hot_run(a, b, frame) {
                probe_add(P_HOT, 1);
                probe_add(P_HOT_PAGES, (b - a) / PAGE);
                STATS[S_HOT_RUNS].fetch_add(1, Ordering::Relaxed);
                let _k = UpKind::new(1);
                let tu = tsc();
                let mut x = a;
                while x < b {
                    if mine(x) {
                        x += PAGE;
                        continue;
                    }
                    let mut y = x + PAGE;
                    while y < b && !mine(y) {
                        y += PAGE;
                    }
                    if !upload(x, y) {
                        STATS[S_UPLOAD_FAILED].fetch_add(1, Ordering::Relaxed);
                        STATS[S_SLOW_NS].fetch_add(t0.elapsed().as_nanos() as u64, Ordering::Relaxed);
                        return Err("an upload into the mirror failed");
                    }
                    STATS[S_UPLOADS].fetch_add(1, Ordering::Relaxed);
                    STATS[S_UPLOAD_PAGES].fetch_add((y - x) / PAGE, Ordering::Relaxed);
                    STATS[S_HOT_PAGES].fetch_add((y - x) / PAGE, Ordering::Relaxed);
                    x = y;
                }
                caller_since(C_UPLOAD, tu);
                if wl {
                    work_part(0, rd().wrapping_sub(wr), (b - a) / PAGE);
                }
                continue;
            }
            // Another thread's take around this run, for a requirement at
            // least ours, instead of one of our own (`follow_take`).
            let tf = tsc();
            if follow_take(a, b, want) {
                probe_add(P_FOLLOW, 1);
                let r = {
                    let _k = UpKind::new(3);
                    followed(a, b, want, mine, upload)
                };
                caller_since(C_FOLLOW, tf);
                match r {
                    Ok(true) => {
                        if wl {
                            let c = rd().wrapping_sub(wr);
                            work_part(1, c, (b - a) / PAGE);
                            if (b - a) / PAGE >= BIG_RUN_PAGES {
                                note_big_run(a, b, 1, c, 0);
                            }
                        }
                        continue;
                    }
                    Ok(false) => {}
                    Err(e) => {
                        STATS[S_SLOW_NS].fetch_add(t0.elapsed().as_nanos() as u64, Ordering::Relaxed);
                        return Err(e);
                    }
                }
            } else {
                caller_since(C_FOLLOW, tf);
            }
            // (Why the graphics CP takes it, for `DS_FE2_CALLERS`.)
            let reasons = (callers_on() && CALLER.with(|c| c.get()) == 0).then(|| take_reasons(a, b, want));
            probe_add(P_TAKES, 1);
            probe_add(P_TAKE_PAGES, (b - a) / PAGE);
            if let Err(e) = check_run(a, b, ep, frame, !gpu_writes, mine, upload, armable, written, class) {
                STATS[S_SLOW_NS].fetch_add(t0.elapsed().as_nanos() as u64, Ordering::Relaxed);
                return Err(e);
            }
            if let Some(k) = reasons {
                note_source_take(a, b, k);
            }
            if wl {
                let c = rd().wrapping_sub(wr);
                work_part(2, c, (b - a) / PAGE);
                if (b - a) / PAGE >= BIG_RUN_PAGES {
                    note_big_run(a, b, 0, c, written.len() as u64);
                }
            }
        }
        if wl {
            work_check(rd().wrapping_sub(w0));
        }
        STATS[S_SLOW_NS].fetch_add(t0.elapsed().as_nanos() as u64, Ordering::Relaxed);
        // (A long one is a hitch cause: `gpu::hitch`.)
        super::hitch::note_since(super::hitch::Kind::Mirror, t0, || format!("check of {lo:#x}..{hi:#x}: {} runs, {} pages", bad.len(), bad.iter().map(|r| (r.1 - r.0) / PAGE).sum::<u64>()));
        Ok(())
    })
}

/// Adds the time since its start to the probe's runs part when dropped
/// (`DS_SLOW_PLACE`).
struct ProbeRuns(Option<std::time::Instant>);

impl Drop for ProbeRuns {
    fn drop(&mut self) {
        if let Some(t) = self.0 {
            probe_add(P_RUNS_NS, t.elapsed().as_nanos() as u64);
        }
    }
}

/// Arms (`arm`: hosted, where allowed) and takes [a, b), uploads its pages
/// M lacks (not `mine`), and marks its valid pages clean in epoch `ep`.
#[allow(clippy::too_many_arguments)]
fn check_run(a: u64, b: u64, ep: u32, frame: u64, arm: bool, mine: &dyn Fn(u64) -> bool, upload: &mut dyn FnMut(u64, u64) -> bool, armable: &Armable, written: &mut Vec<u64>, class: &[AtomicU64; 4]) -> Result<(), &'static str> {
    let (ta, tb) = widen(a, b);
    // (Until its pages are stamped: `follow_take`.)
    let _taking = Taking::new(ta, tb, ep);
    take_run(a, b, ta, tb, frame, arm, armable, written, class);
    let t = tsc();
    let tp = (slow_place_us() != 0).then(std::time::Instant::now);
    let r = fix_run(a, b, ep, mine, upload);
    if let Some(tp) = tp {
        probe_add(P_FIX_NS, tp.elapsed().as_nanos() as u64);
    }
    // The words around a small run were taken too: clean in this epoch.
    if ta < a {
        set_clean(ta, a, ep, false);
    }
    if b < tb {
        set_clean(b, tb, ep, false);
    }
    caller_since(C_FIX, t);
    r
}

/// Runs of at most WIDE_TAKE pages are taken with their whole 64-page words,
/// within the readable mappings around them (`widen`): a take costs its call,
/// not its pages, and a word's other pages are then clean in its epoch, so
/// later checks of small ranges beside it in the same view need no take of
/// their own (at the gate ~3,650 takes a flip were of ranges under 64 KiB with
/// GPU-written pages unarmed: 2026-10-28; `DS_FE2_WIDE_TAKES=0`: the run
/// alone).
const WIDE_TAKE: u64 = 16;

fn wide_takes() -> bool {
    static W: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *W.get_or_init(|| std::env::var("DS_FE2_WIDE_TAKES").ok().as_deref() != Some("0"))
}

/// The range a check's take of [a, b) covers (`WIDE_TAKE`): the run of
/// GPU-written pages around it within its words and mappings. (Whole words
/// took the CPU's per-frame data beside small ranges too: a take that finds
/// written pages resets them, its costly part, and the gate's takes went
/// from ~4,100 to ~2,250 a flip but from ~27 to ~31 ms; 2026-10-28.)
fn widen(a: u64, b: u64) -> (u64, u64) {
    if (b - a) / PAGE > WIDE_TAKE || !wide_takes() {
        return (a, b);
    }
    // (`tune fe2_wide_pages`: the window in pages, whole words.)
    let w = super::tune::fe2_wide_pages().max(64).next_multiple_of(64) * PAGE;
    let (wa, wb) = (a - a % w, b.next_multiple_of(w));
    let Some((s, e)) = crate::mem::readable_span(a).filter(|&(s, e)| s <= a && b <= e) else { return (a, b) };
    let (lo, hi) = (wa.max(s.next_multiple_of(PAGE)), wb.min(e & !(PAGE - 1)));
    // (Past one word only pages a take saw before: a first look at a page
    // counts as a write, `set_clean`, and its next checks would upload it.)
    let seen_only = w > 64 * PAGE;
    let gpu = |p: u64| chunk(p, false).is_some_and(|c| (c.gpu[idx(p) / 64].load(Ordering::Relaxed) & if seen_only { c.seen[idx(p) / 64].load(Ordering::Relaxed) } else { u64::MAX }) >> (idx(p) % 64) & 1 != 0);
    let (mut ta, mut tb) = (a, b);
    while ta > lo && gpu(ta - PAGE) {
        ta -= PAGE;
    }
    while tb < hi && gpu(tb) {
        tb += PAGE;
    }
    (ta, tb)
}

/// `DS_FE2_TAKE_CENSUS=1` (diagnosis): the takes of the threads that take
/// without arming (the graphics CP, queue 0x50's), by thread kind: [takes,
/// pages, ns, pages found written, takes whose every page would pass front
/// end 2's arming rules now (`armable_now`), their pages, their ns, pages
/// armable in all takes, takes of at most 16 pages, their ns]; status line.
fn take_census_on() -> bool {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ON.get_or_init(|| std::env::var("DS_FE2_TAKE_CENSUS").ok().as_deref() == Some("1"))
}
static TAKE_CENSUS: [[AtomicU64; 10]; 2] = [const { [const { AtomicU64::new(0) }; 10] }; 2];
/// The armable pages by why the others were not: [no chunk, armed already,
/// chunk held, too few clean epochs, written lately, outside settled import
/// windows], by thread kind.
static TAKE_CENSUS_NOT: [[AtomicU64; 6]; 2] = [const { [const { AtomicU64::new(0) }; 6] }; 2];

/// The pages of [a, b) that `try_arm` would arm now (its per-page rules and
/// the settled import windows), with the others' reasons added to `not`.
fn armable_now(a: u64, b: u64, frame: u64, not: &mut [u64; 6]) -> u64 {
    let mut ok = 0u64;
    let mut x = a;
    while x < b {
        let e = word_end(x, b);
        let n = (e - x) / PAGE;
        let Some(c) = chunk(x, false) else {
            not[0] += n;
            x = e;
            continue;
        };
        let w = idx(x) / 64;
        let mask = word_mask(x, e);
        let armed = c.armed[w].load(Ordering::Acquire) & mask;
        not[1] += armed.count_ones() as u64;
        let mut cand = mask & !armed;
        if cand != 0 && frame < c.hold_until.load(Ordering::Acquire) {
            not[2] += cand.count_ones() as u64;
            cand = 0;
        }
        if cand != 0 && c.uses[w].load(Ordering::Relaxed) < ARM_USES {
            not[3] += cand.count_ones() as u64;
            cand = 0;
        }
        let base = w * 64;
        let mut r = cand;
        while r != 0 {
            let k = r.trailing_zeros() as usize;
            r &= r - 1;
            let wrote = c.wrote[base + k].load(Ordering::Acquire) as u64;
            let need = ARM_QUIET << c.level[base + k].load(Ordering::Acquire);
            if wrote == 0 || frame.saturating_sub(wrote - 1) < need {
                not[4] += 1;
                cand &= !(1u64 << k);
            }
        }
        ok += cand.count_ones() as u64;
        x = e;
    }
    if ok > 0 && super::exec::fe2_settled_over(a, b) != Some(true) {
        not[5] += ok;
        return 0;
    }
    ok
}

/// The census's status line ("" when off).
fn take_census_line() -> String {
    if !take_census_on() {
        return String::new();
    }
    let mut parts = Vec::new();
    for (k, name) in ["the graphics CP", "queue 0x50's threads"].iter().enumerate() {
        let v: [u64; 10] = std::array::from_fn(|j| TAKE_CENSUS[k][j].load(Ordering::Relaxed));
        let n: [u64; 6] = std::array::from_fn(|j| TAKE_CENSUS_NOT[k][j].load(Ordering::Relaxed));
        if v[0] == 0 {
            continue;
        }
        parts.push(format!(
            "{name}: {} takes ({} pages, {:.1} ms, {} pages found written), all pages armable {} ({} pages, {:.1} ms), pages armable {} (not: no chunk {}, armed {}, chunk held {}, too few clean epochs {}, written lately {}, outside settled windows {}), takes of at most 16 pages {} ({:.1} ms)",
            v[0],
            v[1],
            v[2] as f64 / 1e6,
            v[3],
            v[4],
            v[5],
            v[6] as f64 / 1e6,
            v[7],
            n[0],
            n[1],
            n[2],
            n[3],
            n[4],
            n[5],
            v[8],
            v[9] as f64 / 1e6
        ));
    }
    if parts.is_empty() {
        return String::new();
    }
    format!("    front end 2's takes without arming (`DS_FE2_TAKE_CENSUS`, since boot or `profile reset`): {}\n", parts.join("; "))
}

/// `profile reset`: the take census starts again.
pub fn take_census_reset() {
    for t in TAKE_CENSUS.iter() {
        for x in t.iter() {
            x.store(0, Ordering::Relaxed);
        }
    }
    for t in TAKE_CENSUS_NOT.iter() {
        for x in t.iter() {
            x.store(0, Ordering::Relaxed);
        }
    }
}

/// Arms [a, b) (`arm`: hosted, where allowed) and takes [ta, tb) (around it:
/// `widen`): pages written since their last take lose their valid bit.
#[allow(clippy::too_many_arguments)]
fn take_run(a: u64, b: u64, ta: u64, tb: u64, frame: u64, arm: bool, armable: &Armable, written: &mut Vec<u64>, class: &[AtomicU64; 4]) {
    if why_unarmed_pages() != 0 && (tb - ta) / PAGE >= why_unarmed_pages() {
        why_unarmed(ta, tb, frame, armable);
    }
    // (Queue 0x50's threads and the graphics CP take without arming: their
    // chains set the frame, and the pages stay armable for other threads'
    // checks: queue 0x50 +0.5% at the vista, level at Doran, FINDINGS.md
    // 2026-11-06; the graphics CP, whose large checks of the title's ring
    // spent most of their time arming, x1.007 at the vista in four of four
    // cycles and level at Doran, 2026-11-08 evening.)
    // (Runs longer than `tune arm_inline_max` pages armed behind on every thread: inside the
    // take, an arrival's long runs held the thread and its take's followers for milliseconds.)
    let inline_max = super::tune::arm_inline_max();
    let long = inline_max != 0 && (b - a) / PAGE > inline_max;
    if long && arm && crate::hosted::on() && !matches!(super::prio::role(), 0 | 2) {
        LONG_BEHIND[0].fetch_add(1, Ordering::Relaxed);
        LONG_BEHIND[1].fetch_add((b - a) / PAGE, Ordering::Relaxed);
    }
    if arm && crate::hosted::on() && !matches!(super::prio::role(), 0 | 2) && !long {
        let t = tsc();
        let tp = (slow_place_us() != 0).then(std::time::Instant::now);
        try_arm(a, b, frame, armable);
        if let Some(tp) = tp {
            probe_add(P_ARM_NS, tp.elapsed().as_nanos() as u64);
        }
        caller_since(C_ARM, t);
    } else if arm && crate::hosted::on() {
        // (Armed behind them: `behind_thread`.)
        behind_queue(a, b);
    }
    // The take: pages written since their last take lose their valid bit
    // (`marks_taken`, for any take's pages).
    written.clear();
    let quiet = quiet_census_on().then(|| quiet_range(ta, tb, frame));
    hole_snapshot(ta, tb);
    let tt = std::time::Instant::now();
    let t = tsc();
    let r = super::precache::take_written_for(ta, tb - ta, written, None, crate::mem::WW_WHO_FE2);
    if slow_place_us() != 0 {
        probe_add(P_TAKE_NS, tt.elapsed().as_nanos() as u64);
    }
    if let Some(p) = traced(ta, tb) {
        trace_event(p, &format!("taken with {ta:#x}..{tb:#x} for {a:#x}..{b:#x} (arm {arm}, role {}): found written {}", super::prio::role(), if r.is_none() { "unknown".to_string() } else { written.contains(&p).to_string() }));
    }
    caller_since(C_TAKE, t);
    if r.is_some() {
        hole_check(written);
    }
    if wb_buckets() {
        for &pg in written.iter() {
            note_wb_pages(&WB_CPU_PAGES, pg, PAGE);
        }
    }
    super::hitch::note_since(super::hitch::Kind::Mirror, tt, || format!("take of {ta:#x}..{tb:#x} ({} pages, {} written)", (tb - ta) / PAGE, written.len()));
    let tns = tt.elapsed().as_nanos() as u64;
    STATS[S_TAKE_NS].fetch_add(tns, Ordering::Relaxed);
    STATS[S_TAKES].fetch_add(1, Ordering::Relaxed);
    if callers_on() {
        caller_count(C_TAKES, 1);
    }
    class[2].fetch_add(1, Ordering::Relaxed);
    class[3].fetch_add(tns, Ordering::Relaxed);
    if let Some(q) = quiet {
        let s = &QUIET_TAKES[CALLER.with(|c| c.get())];
        s[0].fetch_add(1, Ordering::Relaxed);
        s[1].fetch_add(tns, Ordering::Relaxed);
        if q {
            s[2].fetch_add(1, Ordering::Relaxed);
            s[3].fetch_add(tns, Ordering::Relaxed);
        }
    }
    STATS[S_TAKE_WRITTEN].fetch_add(written.len() as u64, Ordering::Relaxed);
    if take_census_on() && crate::hosted::on() {
        let role = super::prio::role();
        if matches!(role, 0 | 2) {
            let k = (role == 2) as usize;
            let t = &TAKE_CENSUS[k];
            let pages = (tb - ta) / PAGE;
            t[0].fetch_add(1, Ordering::Relaxed);
            t[1].fetch_add(pages, Ordering::Relaxed);
            t[2].fetch_add(tns, Ordering::Relaxed);
            t[3].fetch_add(written.len() as u64, Ordering::Relaxed);
            let mut not = [0u64; 6];
            let ok = armable_now(ta, tb, frame, &mut not);
            if ok == pages {
                t[4].fetch_add(1, Ordering::Relaxed);
                t[5].fetch_add(pages, Ordering::Relaxed);
                t[6].fetch_add(tns, Ordering::Relaxed);
            }
            t[7].fetch_add(ok, Ordering::Relaxed);
            if pages <= 16 {
                t[8].fetch_add(1, Ordering::Relaxed);
                t[9].fetch_add(tns, Ordering::Relaxed);
            }
            for (j, &v) in not.iter().enumerate() {
                if v != 0 {
                    TAKE_CENSUS_NOT[k][j].fetch_add(v, Ordering::Relaxed);
                }
            }
        }
    }
    if super::fe2census::on() {
        super::fe2census::note_take(ta, tb, written, CALLER.with(|c| c.get()));
    }
    if r.is_none() {
        STATS[S_TAKE_FAILED].fetch_add(1, Ordering::Relaxed);
        invalidate(ta, tb - ta);
    } else if take_all() {
        // Diagnosis: as if the take found every page written (what takes
        // on section-backed memory report for pages they cannot arm).
        invalidate(ta, tb - ta);
    }
}

thread_local! {
    /// `take_run`'s words with armed pages as its take began: (word address,
    /// pages armed and taken since, pages armed).
    static HOLE_WORDS: std::cell::RefCell<Vec<(u64, u64, u64)>> = const { std::cell::RefCell::new(Vec::new()) };
}

/// Pages our takes found written while armed and taken since (no fault
/// ended the arming: a write the protection missed, whose page the mirror
/// would have kept stale where it trusts armed pages).
static HOLES: AtomicU64 = AtomicU64::new(0);

/// Before a take of [ta, tb): its pages armed (the host's bit set), and
/// those of them taken since their arming (`Chunk::varmed`).
fn hole_snapshot(ta: u64, tb: u64) {
    HOLE_WORDS.with(|h| {
        let mut h = h.borrow_mut();
        h.clear();
        let (mut x, hi) = (ta.max(MIRROR_BASE), tb.min(MIRROR_END));
        while x < hi {
            let e = word_end(x, hi);
            if let Some(c) = chunk(x, false) {
                let w = idx(x) / 64;
                let armed = c.armed[w].load(Ordering::Acquire) & word_mask(x, e);
                if armed != 0 {
                    let wa = x & !(64 * PAGE - 1);
                    let now = armed & crate::hosted::armed_word(wa);
                    if now != 0 {
                        h.push((wa, c.varmed[w].load(Ordering::Acquire) & now, now));
                    }
                }
            }
            x = e;
        }
    });
}

/// After that take (it found `written`): a page armed and taken since
/// before it began, found written and still armed and taken since (no
/// fault cleared its bit, no new arming began) is a hole; the pages armed
/// as it began and still armed are taken since their arming now.
fn hole_check(written: &[u64]) {
    HOLE_WORDS.with(|h| {
        let h = h.borrow();
        if h.is_empty() {
            return;
        }
        for &p in written {
            let wa = p & !(64 * PAGE - 1);
            let Ok(k) = h.binary_search_by_key(&wa, |e| e.0) else { continue };
            let bit = 1u64 << (idx(p) % 64);
            if h[k].1 & bit == 0 {
                continue;
            }
            let Some(c) = chunk(p, false) else { continue };
            let w = idx(p) / 64;
            if c.varmed[w].load(Ordering::Acquire) & c.armed[w].load(Ordering::Acquire) & crate::hosted::armed_word(wa) & bit != 0 {
                let n = HOLES.fetch_add(1, Ordering::Relaxed);
                if n < 32 {
                    // (try_lock: the caller may hold the memory map.)
                    let map = crate::mem::MEM.try_lock().map_or_else(|| "busy".into(), |m| m.find(p).map_or_else(|| "none".into(), |v| format!("{:#x}+{:#x} {:?} '{}'", v.start, v.size, v.kind, v.name)));
                    crate::warn!("fe2: a take found page {p:#x} written while armed and taken since (no write fault; flip {}, thread {}; mapping {map})", super::FRAMES.load(Ordering::Relaxed), std::thread::current().name().unwrap_or("?"));
                }
            }
        }
        for &(wa, _, now) in h.iter() {
            let Some(c) = chunk(wa, false) else { continue };
            let w = idx(wa) / 64;
            let live = now & c.armed[w].load(Ordering::Acquire) & crate::hosted::armed_word(wa);
            if live != 0 {
                c.varmed[w].fetch_or(live, Ordering::AcqRel);
            }
        }
    });
}

/// `DS_FE2_TAKE_ALL=1` (diagnosis): every take invalidates its whole range,
/// so its pages, GPU-written ones included, are uploaded into the mirror
/// again in every epoch they are checked in.
/// A dry run of a take per title submission (diagnosis, `tune fe3_scan`;
/// docs/FRONTEND3.md, phase 1): every run of valid pages not armed is
/// queried once without resetting (front end 2's own takes reset those
/// pages, so what it reports was written since their last take).
/// (runs, pages queried, pages reported written, failed queries, ns.)
pub fn fe3_scan_dry() -> (u64, u64, u64, u64, u64) {
    use crate::platform::os::Win32::System::Memory::GetWriteWatch;
    let t = std::time::Instant::now();
    let (mut runs, mut queried, mut written, mut failed) = (0u64, 0u64, 0u64, 0u64);
    let mut addrs = [std::ptr::null_mut::<std::ffi::c_void>(); 512];
    let mut query = |a: u64, e: u64| {
        runs += 1;
        queried += (e - a) / PAGE;
        let mut at = a;
        while at < e {
            let mut count = addrs.len();
            let mut gran = 0u32;
            if unsafe { GetWriteWatch(0, at as _, (e - at) as usize, addrs.as_mut_ptr(), &mut count, &mut gran) } != 0 {
                failed += 1;
                return;
            }
            written += count as u64;
            if count < addrs.len() {
                return;
            }
            at = addrs[count - 1] as u64 + PAGE;
        }
    };
    for (ci, slot) in TABLE.iter().enumerate() {
        let p = slot.load(Ordering::Acquire);
        if p.is_null() {
            continue;
        }
        let c = unsafe { &*p };
        let base = MIRROR_BASE + ((ci as u64) << CHUNK_SHIFT);
        let mut start: Option<u64> = None;
        for w in 0..WORDS {
            let bits = c.valid[w].load(Ordering::Relaxed) & !c.armed[w].load(Ordering::Relaxed);
            let wa = base + (w as u64) * 64 * PAGE;
            if bits == u64::MAX {
                start.get_or_insert(wa);
                continue;
            }
            let mut k = 0u32;
            while k < 64 {
                let set = (bits >> k) & 1 != 0;
                let a = wa + k as u64 * PAGE;
                match (set, start) {
                    (true, None) => start = Some(a),
                    (false, Some(s)) => {
                        query(s, a);
                        start = None;
                    }
                    _ => {}
                }
                k += 1;
            }
        }
        if let Some(s) = start {
            query(s, base + (1u64 << CHUNK_SHIFT));
        }
    }
    (runs, queried, written, failed, t.elapsed().as_nanos() as u64)
}

/// `tune fe3_scan 2` (FRONTEND3.md, its blocker): the same dry run with
/// the cheapest calls (per 64 MiB window one query over the span of its
/// valid pages not armed, or one per run when that covers fewer pages at
/// ~2 us a call and ~2.7 ns a page), and those pages by the age of the last
/// CPU write front end 2 found on them: what a scan per epoch must cover
/// now, and what arming more of them would leave.
/// [calls, pages queried, pages written since their last take, failed
/// queries, ns, pages not armed, of them last written < 2 flips ago, < 16,
/// < 128, longer, never found written, GPU-written ones; then pages of runs
/// the title cannot write on the CPU (its mapping's protection), pages of
/// runs no settled import covers, unarmed pages written within 16 flips in
/// the title's 'batch' mappings, the same elsewhere, the 'batch' mappings'
/// pages in the mirror's span, ns of one query-only scan of them whole, the
/// rings' spans of pages written within 2 flips summed, those pages, rings
/// holding any, the spans by 64 MiB window summed].
pub fn fe3_scan_dry2() -> [u64; 22] {
    use crate::platform::os::Win32::System::Memory::GetWriteWatch;
    let t = std::time::Instant::now();
    let frame = super::FRAMES.load(Ordering::Relaxed);
    let mut out = [0u64; 22];
    // The title's rings: its mappings named 'batch' in the mirror's span.
    let rings: Vec<(u64, u64)> = {
        let g = crate::mem::MEM.lock();
        g.vmas.values().filter(|v| v.name == "batch" && v.start >= MIRROR_BASE && v.end() <= MIRROR_END).map(|v| (v.start, v.end())).collect()
    };
    let in_ring = |a: u64| rings.iter().any(|&(s, e)| s <= a && a < e);
    // (Per ring: the first and last page written within 2 flips, and how
    // many; per window the same; and within 16 flips.)
    let mut spans: Vec<(u64, u64, u64)> = vec![(u64::MAX, 0, 0); rings.len()];
    let mut spans16: Vec<(u64, u64)> = vec![(u64::MAX, 0); rings.len()];
    let mut wspans: std::collections::HashMap<u64, (u64, u64)> = std::collections::HashMap::new();
    let mut addrs = [std::ptr::null_mut::<std::ffi::c_void>(); 512];
    let mut query = |a: u64, e: u64, out: &mut [u64; 22]| {
        out[0] += 1;
        out[1] += (e - a) / PAGE;
        let mut at = a;
        while at < e {
            let mut count = addrs.len();
            let mut gran = 0u32;
            if unsafe { GetWriteWatch(0, at as _, (e - at) as usize, addrs.as_mut_ptr(), &mut count, &mut gran) } != 0 {
                out[3] += 1;
                return;
            }
            out[2] += count as u64;
            if count < addrs.len() {
                return;
            }
            at = addrs[count - 1] as u64 + PAGE;
        }
    };
    let mut runs: Vec<(u64, u64)> = Vec::new();
    for (ci, slot) in TABLE.iter().enumerate() {
        let p = slot.load(Ordering::Acquire);
        if p.is_null() {
            continue;
        }
        let c = unsafe { &*p };
        let base = MIRROR_BASE + ((ci as u64) << CHUNK_SHIFT);
        runs.clear();
        let mut start: Option<u64> = None;
        for w in 0..WORDS {
            let bits = c.valid[w].load(Ordering::Relaxed) & !c.armed[w].load(Ordering::Relaxed);
            let wa = base + (w as u64) * 64 * PAGE;
            if bits != 0 {
                out[5] += bits.count_ones() as u64;
                out[11] += (bits & c.gpu[w].load(Ordering::Relaxed)).count_ones() as u64;
                let mut r = bits;
                while r != 0 {
                    let k = r.trailing_zeros() as usize;
                    r &= r - 1;
                    let wrote = c.wrote[w * 64 + k].load(Ordering::Relaxed) as u64;
                    let slot = if wrote == 0 {
                        10
                    } else {
                        let age = (frame + 1).saturating_sub(wrote);
                        if age < 2 { 6 } else if age < 16 { 7 } else if age < 128 { 8 } else { 9 }
                    };
                    out[slot] += 1;
                    if slot == 6 || slot == 7 {
                        if in_ring(wa + k as u64 * PAGE) {
                            out[14] += 1;
                        } else {
                            out[15] += 1;
                        }
                        let pa = wa + k as u64 * PAGE;
                        if let Some(i) = rings.iter().position(|&(s, e)| s <= pa && pa < e) {
                            let sp = &mut spans16[i];
                            sp.0 = sp.0.min(pa);
                            sp.1 = sp.1.max(pa + PAGE);
                        }
                    }
                    if slot == 6 {
                        let pa = wa + k as u64 * PAGE;
                        if let Some(i) = rings.iter().position(|&(s, e)| s <= pa && pa < e) {
                            let sp = &mut spans[i];
                            sp.0 = sp.0.min(pa);
                            sp.1 = sp.1.max(pa + PAGE);
                            sp.2 += 1;
                        }
                        let ws = wspans.entry(pa >> CHUNK_SHIFT).or_insert((u64::MAX, 0));
                        ws.0 = ws.0.min(pa);
                        ws.1 = ws.1.max(pa + PAGE);
                    }
                }
            }
            if bits == u64::MAX {
                start.get_or_insert(wa);
                continue;
            }
            for k in 0..64u32 {
                let set = (bits >> k) & 1 != 0;
                let a = wa + k as u64 * PAGE;
                match (set, start) {
                    (true, None) => start = Some(a),
                    (false, Some(s)) => {
                        runs.push((s, a));
                        start = None;
                    }
                    _ => {}
                }
            }
        }
        if let Some(s) = start {
            runs.push((s, base + (1u64 << CHUNK_SHIFT)));
        }
        if runs.is_empty() {
            continue;
        }
        for &(a, e) in &runs {
            if crate::mem::prot_at(a).is_some_and(|p| p & crate::mem::PROT_CPU_WRITE == 0) {
                out[12] += (e - a) / PAGE;
            }
            if super::exec::fe2_settled_over(a, e) != Some(true) {
                out[13] += (e - a) / PAGE;
            }
        }
        // (Per run, or the window's span once: whichever the model prices
        // lower, ~2,000 ns a call and ~2.7 ns a page.)
        let pages: u64 = runs.iter().map(|r| (r.1 - r.0) / PAGE).sum();
        let (lo, hi) = (runs[0].0, runs[runs.len() - 1].1);
        let per_run = runs.len() as u64 * 2000 + pages * 27 / 10;
        let span = 2000 + (hi - lo) / PAGE * 27 / 10;
        if span <= per_run {
            query(lo, hi, &mut out);
        } else {
            for &(a, e) in runs.iter() {
                query(a, e, &mut out);
            }
        }
    }
    out[4] = t.elapsed().as_nanos() as u64;
    for sp in spans.iter().filter(|sp| sp.2 > 0) {
        out[18] += (sp.1 - sp.0) / PAGE;
        out[19] += sp.2;
        out[20] += 1;
    }
    out[21] = wspans.values().map(|w| (w.1 - w.0) / PAGE).sum();
    // (The rings' spans of pages written within 16 flips, one query each:
    // what a scan per epoch over the rings would cover.)
    let tr = std::time::Instant::now();
    let mut dummy = [0u64; 22];
    for &(a, e) in spans16.iter().filter(|sp| sp.1 > sp.0) {
        out[16] += (e - a) / PAGE;
        query(a, e, &mut dummy);
    }
    out[17] = tr.elapsed().as_nanos() as u64;
    out
}

fn take_all() -> bool {
    static W: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *W.get_or_init(|| std::env::var("DS_FE2_TAKE_ALL").ok().as_deref() == Some("1"))
}

/// Uploads the pages of [a, b) M lacks (not `mine`) and marks its valid
/// pages clean in epoch `ep` (after its take).
fn fix_run(a: u64, b: u64, ep: u32, mine: &dyn Fn(u64) -> bool, upload: &mut dyn FnMut(u64, u64) -> bool) -> Result<(), &'static str> {
    let (_, failed) = {
        let _k = UpKind::new(4);
        upload_lacking(a, b, mine, upload)
    };
    // Valid pages are clean in this epoch (uploaded ones become valid when
    // the caller's batch is queued: `uploaded`).
    let tc = std::time::Instant::now();
    set_clean(a, b, ep, false);
    super::hitch::note_since(super::hitch::Kind::Mirror, tc, || format!("clean marks of {a:#x}..{b:#x}"));
    if failed {
        return Err("an upload into the mirror failed");
    }
    Ok(())
}

/// Diagnosis (`exec`'s import audit, `DS_IMPORT_AUDIT`): up to `n` pages M
/// holds as current that are armed now (a CPU write faults, which ends the
/// arming) and were never written by GPU work: such a page reads in M as in
/// guest memory. `seed` varies the sample.
pub fn audit_sample(n: usize, seed: u64) -> Vec<u64> {
    let mut out = Vec::with_capacity(n);
    let mut h = seed | 1;
    let mut next = || {
        h ^= h << 13;
        h ^= h >> 7;
        h ^= h << 17;
        h
    };
    for _ in 0..n * 256 {
        if out.len() >= n {
            break;
        }
        let r = next();
        let ci = (r % CHUNKS as u64) as usize;
        let p = TABLE[ci].load(Ordering::Acquire);
        if p.is_null() {
            continue;
        }
        let c = unsafe { &*p };
        let w = ((r >> 40) % WORDS as u64) as usize;
        let wa = MIRROR_BASE + ((ci as u64) << CHUNK_SHIFT) + w as u64 * 64 * PAGE;
        let bits = audit_bits(c, w, wa);
        if bits == 0 {
            continue;
        }
        let mut b = bits;
        for _ in 0..next() % bits.count_ones() as u64 {
            b &= b - 1;
        }
        out.push(wa + b.trailing_zeros() as u64 * PAGE);
    }
    out
}

fn audit_bits(c: &Chunk, w: usize, wa: u64) -> u64 {
    let valid = c.valid[w].load(Ordering::Acquire);
    let gpu = c.gpu[w].load(Ordering::Acquire);
    let armed = valid & c.armed[w].load(Ordering::Acquire) & crate::hosted::armed_word(wa) & !gpu;
    // (`DS_IMPORT_AUDIT_GPU=1`: GPU-written pages current in M too, unarmed;
    // the caller leaves out those the CPU wrote since their last take.)
    if audit_gpu() {
        armed | (valid & gpu)
    } else {
        armed
    }
}

/// `DS_IMPORT_AUDIT_GPU=1` (diagnosis, with `DS_IMPORT_AUDIT`): the mirror
/// audit also samples GPU-written pages current in the mirror (never armed
/// with `DS_FE2_ARM_GPU=0`), compared while no unfinished batch writes
/// them and the CPU has not written them since their last take: there the
/// write-backs must have made guest memory equal to the mirror.
pub fn audit_gpu() -> bool {
    static W: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *W.get_or_init(|| std::env::var("DS_IMPORT_AUDIT_GPU").ok().as_deref() == Some("1"))
}

/// Whether page `p` is marked GPU-written (`Chunk::gpu`).
pub fn audit_gpu_page(p: u64) -> bool {
    chunk(p, false).is_some_and(|c| c.gpu[idx(p) / 64].load(Ordering::Acquire) >> (idx(p) % 64) & 1 != 0)
}

/// Page `p`'s arming record for the audit's log (`DS_IMPORT_AUDIT`): the
/// flip of its last CPU write found, its arming level, the epochs it was
/// found clean in, whether it was taken since its arming.
/// Whether page `p` is valid in the mirror (diagnosis: `DS_WATCH_WRITES`).
pub fn page_valid(p: u64) -> bool {
    chunk(p, false).is_some_and(|c| c.valid[idx(p) / 64].load(Ordering::Acquire) >> (idx(p) % 64) & 1 != 0)
}

/// Whether every page of [a, a+n) is armed now (front end 2's bit and the
/// host's): a CPU write to it faults first (R2's census of what stage
/// preparations read, `DS_R2_ARMED_CENSUS`).
pub fn range_armed(a: u64, n: u64) -> bool {
    if n == 0 || !in_span(a, n) {
        return false;
    }
    let (lo, hi) = (a & !(PAGE - 1), (a + n).next_multiple_of(PAGE));
    let mut x = lo;
    while x < hi {
        let e = word_end(x, hi);
        let Some(c) = chunk(x, false) else { return false };
        let m = word_mask(x, e);
        let armed = c.armed[idx(x) / 64].load(Ordering::Acquire) & crate::hosted::armed_word(x & !(64 * PAGE - 1));
        if armed & m != m {
            return false;
        }
        x = e;
    }
    true
}

pub fn audit_page_state(p: u64) -> String {
    let Some(c) = chunk(p, false) else { return "no chunk".into() };
    let (i, w, bit) = (idx(p), idx(p) / 64, idx(p) % 64);
    format!(
        "last CPU write found at flip {}, arming level {}, clean stamp {} (word {}), taken since arming {}, uses {}",
        c.wrote[i].load(Ordering::Acquire),
        c.level[i].load(Ordering::Acquire),
        c.clean[i].load(Ordering::Acquire),
        c.wclean[w].load(Ordering::Acquire),
        c.varmed[w].load(Ordering::Acquire) >> bit & 1,
        c.uses[w].load(Ordering::Relaxed)
    )
}

/// Whether page `p` (`audit_sample`) is still current in M, armed and never
/// GPU-written.
pub fn audit_still(p: u64) -> bool {
    let Some(c) = chunk(p, false) else { return false };
    audit_bits(c, idx(p) / 64, p & !(64 * PAGE - 1)) & (1u64 << (idx(p) % 64)) != 0
}

/// Uploads the pages of [a, b) M lacks (not `mine`), as runs of consecutive
/// ones (word by word: only the invalid pages are looked at one by one):
/// (pages, whether an upload failed; the rest of the range is still made
/// current).
fn upload_lacking(a: u64, b: u64, mine: &dyn Fn(u64) -> bool, upload: &mut dyn FnMut(u64, u64) -> bool) -> (u64, bool) {
    let tu = std::time::Instant::now();
    let mut failed = false;
    let mut pages = 0u64;
    let mut run: Option<(u64, u64)> = None;
    let mut flush = |run: &mut Option<(u64, u64)>, failed: &mut bool, pages: &mut u64| {
        if let Some((x, y)) = run.take() {
            if !upload(x, y) {
                STATS[S_UPLOAD_FAILED].fetch_add(1, Ordering::Relaxed);
                *failed = true;
            } else {
                STATS[S_UPLOADS].fetch_add(1, Ordering::Relaxed);
                STATS[S_UPLOAD_PAGES].fetch_add((y - x) / PAGE, Ordering::Relaxed);
                *pages += (y - x) / PAGE;
            }
        }
    };
    let mut x = a;
    while x < b {
        let e = word_end(x, b);
        let c = chunk(x, true).expect("inside the span");
        let lacks = word_mask(x, e) & !c.valid[idx(x) / 64].load(Ordering::Acquire);
        let wa = x & !(64 * PAGE - 1);
        let mut r = lacks;
        while r != 0 {
            let p = wa + r.trailing_zeros() as u64 * PAGE;
            r &= r - 1;
            if mine(p) {
                flush(&mut run, &mut failed, &mut pages);
                continue;
            }
            match run.as_mut() {
                Some(l) if l.1 == p => l.1 = p + PAGE,
                _ => {
                    flush(&mut run, &mut failed, &mut pages);
                    run = Some((p, p + PAGE));
                }
            }
        }
        x = e;
    }
    flush(&mut run, &mut failed, &mut pages);
    // (Long parts are hitch causes: `gpu::hitch`.)
    super::hitch::note_since(super::hitch::Kind::Mirror, tu, || format!("uploads of {a:#x}..{b:#x} into the mirror"));
    (pages, failed)
}

/// Our takes in flight, as `check_run` makes them: (taken range, the epoch
/// its pages are stamped with, id). The chunks of a parallel run start
/// together, and each one's first command checked the same GPU-written
/// ranges at the run's view: every worker took them, one after another
/// (~10 commands a flip at ~1.3 ms each on the workers at the vista,
/// 2026-10-28). A check of a run inside one of them, for a requirement it
/// meets, waits for it and uses what it found (`follow_take`, `followed`).
static TAKING: Mutex<Vec<(u64, u64, u32, u64)>> = Mutex::new(Vec::new());
static TAKING_IDS: AtomicU64 = AtomicU64::new(1);
/// Bumped as each of them ends (the waiters watch it).
static TAKING_ENDED: AtomicU64 = AtomicU64::new(0);
/// [runs that followed another's take without one of their own, pages they
/// uploaded, runs that followed and still needed a take].
static FOLLOW: [AtomicU64; 3] = [const { AtomicU64::new(0) }; 3];

/// `DS_FE2_FOLLOW=0` (A/B): every check takes for itself, as before
/// 2026-10-28 evening.
fn follow_on() -> bool {
    static W: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *W.get_or_init(|| std::env::var("DS_FE2_FOLLOW").ok().as_deref() != Some("0"))
}

/// Takes in flight from here until dropped.
struct Taking(u64);

impl Taking {
    fn new(ta: u64, tb: u64, ep: u32) -> Taking {
        let id = TAKING_IDS.fetch_add(1, Ordering::Relaxed);
        TAKING.lock().push((ta, tb, ep, id));
        Taking(id)
    }
}

impl Drop for Taking {
    fn drop(&mut self) {
        TAKING.lock().retain(|e| e.3 != self.0);
        TAKING_ENDED.fetch_add(1, Ordering::AcqRel);
    }
}

/// Whether another thread's take covering [a, b) whose stamps meet `want`
/// was in flight: waited for until it ended (its clean stamps set, the
/// pages it found written invalid).
fn follow_take(a: u64, b: u64, want: u32) -> bool {
    if !follow_on() {
        return false;
    }
    let id = {
        let g = TAKING.lock();
        match g.iter().find(|e| e.0 <= a && b <= e.1 && e.2 >= want) {
            Some(e) => e.3,
            None => return false,
        }
    };
    let mut k = 0u32;
    let tl = super::timeline::mark();
    loop {
        // (Read before the list: an end after it shows as a change.)
        let seen = TAKING_ENDED.load(Ordering::Acquire);
        if !TAKING.lock().iter().any(|e| e.3 == id) {
            super::timeline::end_min(tl, 100, || format!("wait: another thread's take around {a:#x}..{b:#x} ({} pages)", (b - a) / PAGE));
            return true;
        }
        while TAKING_ENDED.load(Ordering::Acquire) == seen {
            k += 1;
            if k < 64 {
                std::hint::spin_loop();
            } else {
                std::thread::yield_now();
            }
        }
    }
}

/// After another's take over [a, b) (`follow_take`): Ok(true) when nothing
/// needs a take of our own (pages valid and clean for `want`; pages not
/// valid, which that take found written or the mirror never held, uploaded
/// here as after a take: they become valid and clean when the caller's
/// batch is queued, `uploaded`); Ok(false) when a valid page is not clean
/// for `want`.
fn followed(a: u64, b: u64, want: u32, mine: &dyn Fn(u64) -> bool, upload: &mut dyn FnMut(u64, u64) -> bool) -> Result<bool, &'static str> {
    let mut x = a;
    while x < b {
        let e = word_end(x, b);
        let c = chunk(x, true).expect("inside the span");
        let need = word_mask(x, e) & !word_ok(c, x, e, want);
        if need & c.valid[idx(x) / 64].load(Ordering::Acquire) != 0 {
            FOLLOW[2].fetch_add(1, Ordering::Relaxed);
            return Ok(false);
        }
        x = e;
    }
    let (pages, failed) = upload_lacking(a, b, mine, upload);
    FOLLOW[0].fetch_add(1, Ordering::Relaxed);
    FOLLOW[1].fetch_add(pages, Ordering::Relaxed);
    if failed {
        return Err("an upload into the mirror failed");
    }
    Ok(true)
}

/// The runs of pages of [lo, hi) (page-aligned) each found written within
/// its hot window (`hot_run`'s): a stretch past a label wait uploads them
/// as they are when it runs, whatever its checks found, since the CPU may
/// write them again before its label (`Exec::fe2_place_inner`; the
/// title's per-frame blocks, rewritten after the graphics queue's
/// submission and before the compute queue's whose label it waits on).
pub fn hot_pages(lo: u64, hi: u64, out: &mut Vec<(u64, u64)>) {
    let frame = super::FRAMES.load(Ordering::Relaxed);
    let quiet = hot_quiet();
    let mut start: Option<u64> = None;
    let mut x = lo;
    while x < hi {
        let hot = chunk(x, false).is_some_and(|c| {
            let i = idx(x);
            let wrote = c.wrote[i].load(Ordering::Relaxed) as u64;
            let window = if quiet { HOT_FLIPS.max(ARM_QUIET << c.level[i].load(Ordering::Relaxed)) } else { HOT_FLIPS };
            wrote != 0 && frame.saturating_sub(wrote - 1) < window
        });
        match (hot, start) {
            (true, None) => start = Some(x),
            (false, Some(s)) => {
                out.push((s, x));
                start = None;
            }
            _ => {}
        }
        x += PAGE;
    }
    if let Some(s) = start {
        out.push((s, hi));
    }
}

/// Whether [a, b) is at most HOT_PAGES pages, every one found written
/// within HOT_FLIPS flips.
fn hot_run(a: u64, b: u64, frame: u64) -> bool {
    if (b - a) / PAGE > HOT_PAGES {
        return false;
    }
    let quiet = hot_quiet();
    let mut x = a;
    while x < b {
        let Some(c) = chunk(x, false) else { return false };
        let i = idx(x);
        let wrote = c.wrote[i].load(Ordering::Relaxed) as u64;
        let window = if quiet { HOT_FLIPS.max(ARM_QUIET << c.level[i].load(Ordering::Relaxed)) } else { HOT_FLIPS };
        if wrote == 0 || frame.saturating_sub(wrote - 1) >= window {
            return false;
        }
        x += PAGE;
    }
    true
}

/// Marks pages of [a, b) clean in epoch `ep` (whole words: `wclean` too);
/// `validate`: they become valid as well. A page seen for the first time
/// counts as written now (its quiet time before arming starts here: a
/// take cannot tell what the CPU will write next into memory it just
/// filled).
fn set_clean(a: u64, b: u64, ep: u32, validate: bool) {
    let f = super::FRAMES.load(Ordering::Relaxed) as u32 + 1;
    let mut x = a;
    while x < b {
        let e = word_end(x, b);
        let Some(c) = chunk(x, true) else { break };
        let (i, w) = (idx(x), idx(x) / 64);
        let mask = word_mask(x, e);
        // A whole word goes by its stamp (`word_ok` reads the pages' own only
        // when the word's is older).
        let whole = mask == u64::MAX;
        if !whole {
            for k in i..i + ((e - x) / PAGE) as usize {
                c.clean[k].store(ep, Ordering::Release);
            }
            c.wmax[w].fetch_max(ep, Ordering::AcqRel);
        }
        // A take's first look at the word in this epoch counts as a use.
        if !validate && c.use_ep[w].swap(ep, Ordering::AcqRel) != ep {
            let u = &c.uses[w];
            if u.load(Ordering::Relaxed) < ARM_USES {
                u.fetch_add(1, Ordering::Relaxed);
            }
        }
        let unseen = mask & !c.seen[w].load(Ordering::Acquire);
        if unseen != 0 {
            let mut r = unseen;
            while r != 0 {
                let k = w * 64 + r.trailing_zeros() as usize;
                r &= r - 1;
                let _ = c.wrote[k].compare_exchange(0, f, Ordering::AcqRel, Ordering::Relaxed);
            }
            c.seen[w].fetch_or(unseen, Ordering::AcqRel);
        }
        if validate {
            c.valid[w].fetch_or(mask, Ordering::AcqRel);
        }
        if whole {
            c.wclean[w].store(ep, Ordering::Release);
        }
        x = e;
    }
    if let Some(p) = traced(a, b) {
        trace_event(p, &format!("set clean in epoch {ep} (validate {validate}) with {a:#x}..{b:#x}"));
    }
}

/// Pages of [a, b) we armed whose host bit is gone (a write fault, its
/// word's, a disarm or a mapping change): no longer ours; the next arming
/// waits longer, its quiet time from now (the fault watcher may not have
/// seen the fault yet).
fn note_arming_ended(a: u64, b: u64) {
    let mut x = a;
    while x < b {
        let e = word_end(x, b);
        let Some(c) = chunk(x, false) else { return };
        let w = idx(x) / 64;
        let ours = c.armed[w].load(Ordering::Acquire) & word_mask(x, e);
        if ours != 0 {
            let gone = ours & !crate::hosted::armed_word(x & !(64 * PAGE - 1));
            if gone != 0 {
                c.armed[w].fetch_and(!gone, Ordering::AcqRel);
                c.cgen.fetch_add(1, Ordering::AcqRel);
                if let Some(p) = traced(x, e) {
                    if gone >> (idx(p) % 64) & 1 != 0 {
                        trace_event(p, "a take saw its arming ended (note_arming_ended)");
                    }
                }
                let base = w * 64;
                let f = super::FRAMES.load(Ordering::Relaxed) as u32 + 1;
                let mut r = gone;
                while r != 0 {
                    let k = r.trailing_zeros() as usize;
                    r &= r - 1;
                    let l = &c.level[base + k];
                    let _ = l.fetch_update(Ordering::AcqRel, Ordering::Acquire, |v| Some((v + 1).min(ARM_MAX_LEVEL)));
                    c.wrote[base + k].store(f, Ordering::Release);
                }
                STATS[S_ARM_ENDED].fetch_add(gone.count_ones() as u64, Ordering::Relaxed);
            }
        }
        x = e;
    }
}

/// Writes that cannot fault in progress (file reads into guest memory):
/// their pages are not armed meanwhile (the kernel fails a write into a
/// read-only page instead of faulting).
static HOST_WRITES: Mutex<Vec<(u64, u64)>> = Mutex::new(Vec::new());
static HOST_WRITES_N: AtomicU64 = AtomicU64::new(0);

/// A write that cannot fault begins over guest [base, base+len)
/// (`track::host_write`; its pages were disarmed).
pub fn host_write_begin(base: u64, len: u64) {
    if !on() || len == 0 {
        return;
    }
    hold_begin(base, len, "a file read");
    invalidate(base, len);
}

/// That write ended: its pages hold new contents.
pub fn host_write_end(base: u64, len: u64) {
    if !on() || len == 0 {
        return;
    }
    hold_end(base, len);
    invalidate(base, len);
}

/// Guest [base, base+len) armed by none until `hold_end` (`HOST_WRITES`).
fn hold_begin(base: u64, len: u64, who: &str) {
    // (An arming that checked before this is done by now, and the caller's
    // disarm after this undoes it: `arm_part`.)
    let t = std::time::Instant::now();
    let _lock = crate::hosted::arm_lock();
    super::hitch::note_since(super::hitch::Kind::Mirror, t, || format!("{who} into {base:#x}+{len:#x} waiting for an arming"));
    HOST_WRITES.lock().push((base, base.saturating_add(len)));
    HOST_WRITES_N.fetch_add(1, Ordering::AcqRel);
}

fn hold_end(base: u64, len: u64) {
    let mut g = HOST_WRITES.lock();
    if let Some(i) = g.iter().position(|&(a, e)| a == base && e == base.saturating_add(len)) {
        g.swap_remove(i);
        HOST_WRITES_N.fetch_sub(1, Ordering::AcqRel);
    }
}

/// Guest [base, base+len) held unarmed while this lives (`hold_begin`).
pub struct Hold(Option<(u64, u64)>);

impl Drop for Hold {
    fn drop(&mut self) {
        if let Some((base, len)) = self.0 {
            hold_end(base, len);
        }
    }
}

/// An import of guest [base, base+len) begins (`exec::import_span`, before
/// its disarm and the driver's call): its pages are armed by none until it
/// is listed, young (the import list is no longer held while arming).
pub fn hold(base: u64, len: u64) -> Hold {
    if !on() || len == 0 {
        return Hold(None);
    }
    hold_begin(base, len, "an import");
    Hold(Some((base, len)))
}

fn host_writing(a: u64, b: u64) -> bool {
    HOST_WRITES_N.load(Ordering::Acquire) != 0 && HOST_WRITES.lock().iter().any(|&(x, e)| x < b && a < e)
}

/// Arms the quiet pages of [a, b) not armed now (nor written by GPU work,
/// with `DS_FE2_ARM_GPU=0`) outside held chunks, by runs inside settled
/// import windows (`armable`), plain read-write memory only.
fn try_arm(a: u64, b: u64, frame: u64, armable: &Armable) {
    let arm_gpu = arm_gpu();
    let arm_gpu_os = arm_gpu_os();
    // (The armer behind the graphics CP asks for longer quiet times: its
    // pages are the ones only the non-arming threads check, and in the walk
    // the title streams into them.)
    let quiet_shift = if BEHIND_ARMING.with(|f| f.get()) { super::tune::behind_quiet() as u8 } else { 0 };
    // A word at a time, the pages not armed counted here and added once
    // (until 2026-10-28 page by page, each such page a shared atomic add:
    // ~170,000 pages of held chunks a flip at the gate). A page's reason is
    // the first of: armed already, its chunk held, GPU-written, its word
    // found clean in too few epochs, its quiet time.
    let mut not = [0u64; 8];
    // The run of candidates being gathered (armed when a page that is not
    // one ends it).
    let mut run: Option<(u64, u64)> = None;
    let mut x = a;
    while x < b {
        let e = word_end(x, b);
        let Some(c) = chunk(x, false) else {
            not[R_QUIET] += (e - x) / PAGE;
            if let Some((p, q)) = run.take() {
                arm_run(p, q, frame, armable);
            }
            x = e;
            continue;
        };
        let w = idx(x) / 64;
        let mask = word_mask(x, e);
        let mut armed = c.armed[w].load(Ordering::Acquire) & mask;
        if arm_gpu_os {
            // (GPU-written pages protected and kept unarmed: the host's bit.)
            armed |= c.gpu[w].load(Ordering::Acquire) & crate::hosted::armed_word(x & !(64 * PAGE - 1)) & mask;
        }
        not[R_ARMED] += armed.count_ones() as u64;
        let mut cand = mask & !armed;
        if cand != 0 && frame < c.hold_until.load(Ordering::Acquire) {
            not[R_HELD] += cand.count_ones() as u64;
            cand = 0;
        }
        if cand != 0 && !arm_gpu && !arm_gpu_os && !hold_census() && !crate::mem::view_window(x) {
            let g = c.gpu[w].load(Ordering::Acquire) & cand;
            not[R_GPU] += g.count_ones() as u64;
            cand &= !g;
        }
        if cand != 0 && c.uses[w].load(Ordering::Relaxed) < ARM_USES {
            not[R_USES] += cand.count_ones() as u64;
            cand = 0;
        }
        let base = w * 64;
        let mut r = cand;
        while r != 0 {
            let k = r.trailing_zeros() as usize;
            r &= r - 1;
            let wrote = c.wrote[base + k].load(Ordering::Acquire) as u64;
            let need = ARM_QUIET << (c.level[base + k].load(Ordering::Acquire) + quiet_shift).min(16);
            if wrote == 0 || frame.saturating_sub(wrote - 1) < need {
                not[R_QUIET] += 1;
                cand &= !(1u64 << k);
            }
        }
        // The word's runs of candidates (a run goes on across words).
        let wa = x & !(64 * PAGE - 1);
        let (mut pos, end) = ((idx(x) % 64) as u32, (idx(x) % 64) as u32 + ((e - x) / PAGE) as u32);
        while pos < end {
            let rest = cand >> pos;
            if rest & 1 != 0 {
                let ones = (!rest).trailing_zeros().min(end - pos);
                let (p, q) = (wa + pos as u64 * PAGE, wa + (pos + ones) as u64 * PAGE);
                match run.as_mut() {
                    Some(l) if l.1 == p => l.1 = q,
                    _ => {
                        if let Some((s, t)) = run.replace((p, q)) {
                            arm_run(s, t, frame, armable);
                        }
                    }
                }
                pos += ones;
            } else {
                if let Some((s, t)) = run.take() {
                    arm_run(s, t, frame, armable);
                }
                pos += if rest == 0 { end - pos } else { rest.trailing_zeros().min(end - pos) };
            }
        }
        x = e;
    }
    if let Some((p, q)) = run.take() {
        arm_run(p, q, frame, armable);
    }
    for (k, &n) in not.iter().enumerate() {
        if n != 0 {
            not_armed(k, n);
        }
    }
}

/// `DS_FE2_WHY_UNARMED=<pages>` (diagnosis): the least pages of a take
/// `why_unarmed` reports on (0: none).
fn why_unarmed_pages() -> u64 {
    static N: std::sync::OnceLock<u64> = std::sync::OnceLock::new();
    *N.get_or_init(|| std::env::var("DS_FE2_WHY_UNARMED").ok().and_then(|v| v.trim().parse().ok()).unwrap_or(0))
}

/// Logs (the first 40 takes) why the pages of [a, b) are not armed, as
/// `try_arm` would decide now (nothing armed): its reasons, the ages of the
/// pages written lately (flips since their last write against the quiet time
/// their level needs), and the candidates outside settled import windows.
fn why_unarmed(a: u64, b: u64, frame: u64, armable: &Armable) {
    // (`DS_FE2_WHY_FROM=<flip>`: from that flip on.)
    static FROM: std::sync::OnceLock<u64> = std::sync::OnceLock::new();
    if frame < *FROM.get_or_init(|| std::env::var("DS_FE2_WHY_FROM").ok().and_then(|v| v.trim().parse().ok()).unwrap_or(0)) {
        return;
    }
    static LOGGED: AtomicU64 = AtomicU64::new(0);
    if LOGGED.fetch_add(1, Ordering::Relaxed) >= 40 {
        return;
    }
    let arm_gpu = arm_gpu();
    let mut not = [0u64; 8];
    let (mut cand_pages, mut outside, mut nochunk) = (0u64, 0u64, 0u64);
    // Written lately: flips since the last write (<8, <30, <120, more) and
    // the levels seen.
    let mut ages = [0u64; 4];
    let mut levels = [0u64; ARM_MAX_LEVEL as usize + 1];
    let mut never = 0u64;
    let mut x = a;
    while x < b {
        let e = word_end(x, b);
        let Some(c) = chunk(x, false) else {
            nochunk += (e - x) / PAGE;
            x = e;
            continue;
        };
        let w = idx(x) / 64;
        let mask = word_mask(x, e);
        let armed = c.armed[w].load(Ordering::Acquire) & mask;
        not[R_ARMED] += armed.count_ones() as u64;
        let mut cand = mask & !armed;
        if cand != 0 && frame < c.hold_until.load(Ordering::Acquire) {
            not[R_HELD] += cand.count_ones() as u64;
            cand = 0;
        }
        if cand != 0 && !arm_gpu && !crate::mem::view_window(x) {
            let g = c.gpu[w].load(Ordering::Acquire) & cand;
            not[R_GPU] += g.count_ones() as u64;
            cand &= !g;
        }
        if cand != 0 && c.uses[w].load(Ordering::Relaxed) < ARM_USES {
            not[R_USES] += cand.count_ones() as u64;
            cand = 0;
        }
        let base = w * 64;
        let mut r = cand;
        while r != 0 {
            let k = r.trailing_zeros() as usize;
            r &= r - 1;
            let wrote = c.wrote[base + k].load(Ordering::Acquire) as u64;
            let level = c.level[base + k].load(Ordering::Acquire);
            let need = ARM_QUIET << level;
            if wrote == 0 || frame.saturating_sub(wrote - 1) < need {
                not[R_QUIET] += 1;
                cand &= !(1u64 << k);
                levels[(level as usize).min(ARM_MAX_LEVEL as usize)] += 1;
                if wrote == 0 {
                    never += 1;
                } else {
                    let age = frame.saturating_sub(wrote - 1);
                    ages[if age < 8 { 0 } else if age < 30 { 1 } else if age < 120 { 2 } else { 3 }] += 1;
                }
            }
        }
        // The candidates' runs inside settled import windows.
        let wa = x & !(64 * PAGE - 1);
        let mut r = cand;
        while r != 0 {
            let k = r.trailing_zeros() as u64;
            let ones = (!(r >> k)).trailing_zeros().min(64 - k as u32) as u64;
            let (p, q) = (wa + k * PAGE, wa + (k + ones) * PAGE);
            cand_pages += ones;
            let mut inside = 0u64;
            if armable(p, q, &mut |s, t| inside += t - s) {
                outside += (q - p - inside) / PAGE;
            }
            r &= !(if k + ones >= 64 { u64::MAX << k } else { ((1u64 << ones) - 1) << k });
        }
        x = e;
    }
    crate::info!(
        "fe2: why unarmed: take of {a:#x}..{b:#x} ({} pages): armed already {}, chunk held {}, GPU-written {}, too few clean epochs {}, written lately {} (flips since their write <8 {}, 8-29 {}, 30-119 {}, more {}; never seen {}; levels {:?}), no chunk {}, candidates {} ({} outside settled import windows)",
        (b - a) / PAGE,
        not[R_ARMED],
        not[R_HELD],
        not[R_GPU],
        not[R_USES],
        not[R_QUIET],
        ages[0],
        ages[1],
        ages[2],
        ages[3],
        never,
        levels,
        nochunk,
        cand_pages,
        outside
    );
}

/// Pages checked by takes that were not armed, by reason: armed already
/// (by us, its host bit set), written by GPU work (`DS_FE2_ARM_GPU`),
/// written within their quiet time, found clean in fewer than ARM_USES
/// epochs, outside settled import windows, not plain read-write (or a file
/// read or an import in progress), in a chunk held after a burst of faults,
/// the import list busy.
static NOT_ARMED: [AtomicU64; 8] = [const { AtomicU64::new(0) }; 8];
const R_ARMED: usize = 0;
const R_GPU: usize = 1;
const R_QUIET: usize = 2;
const R_USES: usize = 3;
const R_WINDOW: usize = 4;
const R_RW: usize = 5;
const R_HELD: usize = 6;
const R_BUSY: usize = 7;

/// The same reasons for C4's own arming (`c4_round`: the pages it offered).
static C4_NOT_ARMED: [AtomicU64; 8] = [const { AtomicU64::new(0) }; 8];

/// Counts `n` pages not armed for reason `k` (C4's own arming apart).
fn not_armed(k: usize, n: u64) {
    if C4_ARMING.with(|f| f.get()) {
        C4_NOT_ARMED[k].fetch_add(n, Ordering::Relaxed);
    } else if BEHIND_ARMING.with(|f| f.get()) {
        BEHIND_NOT_ARMED[k].fetch_add(n, Ordering::Relaxed);
    } else {
        NOT_ARMED[k].fetch_add(n, Ordering::Relaxed);
    }
}

// ---------------------------------------------------------------------------
// Arming behind the threads that take without arming (2026-11-11 night).
// The graphics CP's and queue 0x50's checks take without arming (arming on
// those threads cost them more than the takes it saved: FINDINGS.md
// 2026-11-06, 2026-11-08 evening), so pages only they check were never
// armed and every epoch's check of them took again: at Castlewall Front
// ~1.6 ms a flip of the graphics CP's takes, 77% of them over pages that
// pass front end 2's arming rules (`DS_FE2_TAKE_CENSUS`). Their checks now
// queue the runs they took, and the `fe2-armer` thread arms them by front
// end 2's own rules (`try_arm`); a later check finds them armed and takes
// nothing.
// ---------------------------------------------------------------------------

/// Arming behind is wanted (`DS_ARM_BEHIND=0` off).
fn arm_behind_wanted() -> bool {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ON.get_or_init(|| std::env::var("DS_ARM_BEHIND").ok().as_deref() != Some("0"))
}

/// `armbehind off` (in-run A/Bs): nothing queued or armed, and the pages it
/// armed disarmed.
static BEHIND_OFF: AtomicBool = AtomicBool::new(false);
/// The runs queued for the armer (at most BEHIND_MAX; more are dropped).
static BEHIND_RUNS: Mutex<Vec<(u64, u64)>> = Mutex::new(Vec::new());
const BEHIND_MAX: usize = 16384;
/// [runs queued, runs dropped (queue full), rounds, merged runs offered,
/// pages offered, pages armed, us arming, pages disarmed by `armbehind off`,
/// takes after an arming, pages they found written, runs skipped as offered
/// lately, rounds that ran out of their time].
static BEHIND: [AtomicU64; 12] = [const { AtomicU64::new(0) }; 12];
/// A run offered within this many flips is not offered again (`behind_thread`).
const BEHIND_AGAIN: u64 = 30;
const AB_QUEUED: usize = 0;
const AB_DROPPED: usize = 1;
const AB_ROUNDS: usize = 2;
const AB_RUNS: usize = 3;
const AB_PAGES: usize = 4;
const AB_ARMED: usize = 5;
const AB_US: usize = 6;
const AB_DISARMED: usize = 7;
const AB_TAKES: usize = 8;
const AB_TAKE_WRITTEN: usize = 9;
const AB_SKIPPED: usize = 10;
const AB_OVER: usize = 11;
/// Front end 2's reasons for the pages the armer offered and did not arm.
static BEHIND_NOT_ARMED: [AtomicU64; 8] = [const { AtomicU64::new(0) }; 8];
/// Runs longer than `tune arm_inline_max` pages that threads arming as they take left to the
/// armer: [runs, pages].
static LONG_BEHIND: [AtomicU64; 2] = [const { AtomicU64::new(0) }; 2];

thread_local! {
    /// The armer's thread (`not_armed`, `arm_part` count for it).
    static BEHIND_ARMING: std::cell::Cell<bool> = const { std::cell::Cell::new(false) };
}

/// The pages the armer armed (a bit per page of the mirror's span):
/// `armbehind off` disarms those still armed.
fn behind_map() -> &'static [AtomicU32] {
    static M: std::sync::OnceLock<Box<[AtomicU32]>> = std::sync::OnceLock::new();
    M.get_or_init(|| (0..HELD_MAP_WORDS).map(|_| AtomicU32::new(0)).collect())
}

/// A check of [a, b) on a thread that takes without arming took it: queued
/// for the armer.
fn behind_queue(a: u64, b: u64) {
    if !arm_behind_wanted() || BEHIND_OFF.load(Ordering::Relaxed) {
        return;
    }
    let mut g = BEHIND_RUNS.lock();
    if g.len() >= BEHIND_MAX {
        drop(g);
        BEHIND[AB_DROPPED].fetch_add(1, Ordering::Relaxed);
        return;
    }
    g.push((a, b));
    drop(g);
    BEHIND[AB_QUEUED].fetch_add(1, Ordering::Relaxed);
}

/// The armer (`fe2-armer`): every 2 ms the queued runs, sorted and merged,
/// offered to `try_arm`.
fn behind_thread() {
    BEHIND_ARMING.with(|f| f.set(true));
    // (Below the recording threads: its work is never on their path.)
    unsafe {
        crate::platform::os::Win32::System::Threading::SetThreadPriority(crate::platform::os::Win32::System::Threading::GetCurrentThread(), crate::platform::os::Win32::System::Threading::THREAD_PRIORITY_BELOW_NORMAL);
    }
    let mut runs: Vec<(u64, u64)> = Vec::new();
    let mut merged: Vec<(u64, u64)> = Vec::new();
    let mut written: Vec<u64> = Vec::new();
    // The runs offered lately, by a hash of their start: (start, end, flip).
    // A run offered within BEHIND_AGAIN flips is skipped: the checks queue
    // the same runs every epoch, and pages that were not armable then are
    // rarely armable a moment later.
    let mut offered: Vec<(u64, u64, u64)> = vec![(0, 0, 0); 1 << 14];
    loop {
        std::thread::sleep(std::time::Duration::from_millis(2));
        std::mem::swap(&mut runs, &mut *BEHIND_RUNS.lock());
        if runs.is_empty() {
            continue;
        }
        if BEHIND_OFF.load(Ordering::Acquire) {
            runs.clear();
            continue;
        }
        runs.sort_unstable();
        runs.dedup();
        // (Runs offered within BEHIND_AGAIN flips skipped: the checks queue
        // the same runs every epoch.)
        let frame = super::FRAMES.load(Ordering::Relaxed);
        let before_filter = runs.len();
        runs.retain(|&(a, b)| {
            let h = (((a >> 12) ^ (b >> 12).rotate_left(17)).wrapping_mul(0x9E37_79B9_7F4A_7C15) >> 50) as usize;
            let e = offered[h];
            if e.0 == a && e.1 == b && frame.saturating_sub(e.2) < BEHIND_AGAIN {
                false
            } else {
                offered[h] = (a, b, frame);
                true
            }
        });
        BEHIND[AB_SKIPPED].fetch_add((before_filter - runs.len()) as u64, Ordering::Relaxed);
        merged.clear();
        for &(a, b) in &runs {
            match merged.last_mut() {
                Some(l) if a <= l.1 => l.1 = l.1.max(b),
                _ => merged.push((a, b)),
            }
        }
        runs.clear();
        let t = std::time::Instant::now();
        let budget = std::time::Duration::from_micros(super::tune::behind_us().max(1));
        let mut pages = 0u64;
        let mut offered_now = 0u64;
        for &(a, b) in &merged {
            // (A round's time is bounded: the rest are queued again by
            // their next checks.)
            if t.elapsed() > budget {
                BEHIND[AB_OVER].fetch_add(1, Ordering::Relaxed);
                break;
            }
            offered_now += 1;
            pages += (b - a) / PAGE;
            let before = BEHIND[AB_ARMED].load(Ordering::Relaxed);
            try_arm(a, b, frame, &super::exec::fe2_armable);
            if BEHIND[AB_ARMED].load(Ordering::Relaxed) != before {
                // (As `take_run`: armed first, then taken. A CPU write after
                // the check's take and before this arming is found here and
                // its page made invalid (`marks_taken`), and the pages armed
                // are taken since their arming (`hole_check`).)
                hole_snapshot(a, b);
                written.clear();
                let r = super::precache::take_written_for(a, b - a, &mut written, None, crate::mem::WW_WHO_FE2);
                if r.is_some() {
                    hole_check(&written);
                } else {
                    invalidate(a, b - a);
                }
                BEHIND[AB_TAKES].fetch_add(1, Ordering::Relaxed);
                BEHIND[AB_TAKE_WRITTEN].fetch_add(written.len() as u64, Ordering::Relaxed);
            }
        }
        BEHIND[AB_ROUNDS].fetch_add(1, Ordering::Relaxed);
        BEHIND[AB_RUNS].fetch_add(offered_now, Ordering::Relaxed);
        BEHIND[AB_PAGES].fetch_add(pages, Ordering::Relaxed);
        BEHIND[AB_US].fetch_add(t.elapsed().as_micros() as u64, Ordering::Relaxed);
    }
}

/// `arm_part` armed [x, y) on the armer's thread.
fn behind_note_armed(x: u64, y: u64) {
    let m = behind_map();
    let (lo, hi) = (x.max(MIRROR_BASE), y.min(MIRROR_END));
    let mut p = lo;
    while p < hi {
        let i = ((p - MIRROR_BASE) / PAGE) as usize;
        m[i / 32].fetch_or(1 << (i % 32), Ordering::Relaxed);
        p += PAGE;
    }
    BEHIND[AB_ARMED].fetch_add(hi.saturating_sub(lo) / PAGE, Ordering::Relaxed);
}

/// Disarms the pages the armer armed that are still armed (front end 2's
/// bit and the host's), held pages apart: `armbehind off`.
fn behind_disarm_all() -> u64 {
    let m = behind_map();
    let _readers = crate::mem::GUEST_READERS.read_recursive();
    let _arming = super::resident::arm_guard();
    // (No arming between a page's bit cleared and its protection gone: an
    // arming in between would leave the bit clear over a protected page, or
    // set over a writable one.)
    let _lock = crate::hosted::arm_lock();
    let mut n = 0u64;
    let mut pend: Option<(u64, u64)> = None;
    for (i, w) in m.iter().enumerate() {
        let v = w.swap(0, Ordering::AcqRel);
        let mut r = v;
        while r != 0 {
            let bit = r.trailing_zeros() as u64;
            r &= r - 1;
            let p = MIRROR_BASE + (i as u64 * 32 + bit) * PAGE;
            let Some(c) = chunk(p, false) else { continue };
            let (k, wi) = (idx(p) % 64, idx(p) / 64);
            if c.armed[wi].load(Ordering::Acquire) >> k & 1 == 0 || c4_held(p) {
                continue;
            }
            // (Front end 2's bit first, as a burst's disarm: a check that
            // reads it set after the host's protection is gone only takes.)
            if c.armed[wi].fetch_and(!(1u64 << k), Ordering::AcqRel) >> k & 1 == 0 {
                continue;
            }
            c.cgen.fetch_add(1, Ordering::AcqRel);
            if traced(p, p + PAGE).is_some() {
                trace_event(p, "disarmed with everything armed behind (behind_disarm_all)");
            }
            n += 1;
            pend = match pend {
                Some((a, b)) if b == p => Some((a, p + PAGE)),
                Some((a, b)) => {
                    crate::hosted::disarm(a, b - a);
                    Some((p, p + PAGE))
                }
                None => Some((p, p + PAGE)),
            };
        }
    }
    if let Some((a, b)) = pend {
        crate::hosted::disarm(a, b - a);
    }
    BEHIND[AB_DISARMED].fetch_add(n, Ordering::Relaxed);
    n
}

/// `armbehind off|on` (in-run A/Bs of arming behind).
pub fn behind_control(what: &str) -> Result<String, String> {
    if !on() || !crate::hosted::on() || !arm_behind_wanted() {
        return Err("armbehind: not on (front end 2 hosted, `DS_ARM_BEHIND` not 0)".into());
    }
    match what {
        "off" => {
            BEHIND_OFF.store(true, Ordering::Release);
            BEHIND_RUNS.lock().clear();
            let n = behind_disarm_all();
            Ok(format!("arming behind off, {n} pages it armed disarmed"))
        }
        "on" => {
            BEHIND_OFF.store(false, Ordering::Release);
            Ok("arming behind on".into())
        }
        _ => Err("usage: armbehind off|on".into()),
    }
}

/// The armer's status line ("" when off).
fn behind_line() -> String {
    if !arm_behind_wanted() || !crate::hosted::on() {
        return String::new();
    }
    let v: [u64; 12] = std::array::from_fn(|k| BEHIND[k].load(Ordering::Relaxed));
    let na: [u64; 8] = std::array::from_fn(|k| BEHIND_NOT_ARMED[k].load(Ordering::Relaxed));
    format!(
        "    front end 2's arming behind the graphics CP and queue 0x50 (`armbehind`{}, since boot; `tune behind_quiet` {}, `behind_us` {}; long runs of other threads, `tune arm_inline_max` {}: {} ({} pages)): {} runs queued ({} dropped), {} rounds ({} out of time), {} merged runs offered ({} pages; {} skipped as offered lately), {} pages armed, {:.1} ms arming, {} takes after an arming ({} pages found written), {} disarmed by `armbehind off`; not armed: armed already {}, too few clean epochs {}, written lately {}, chunk held {}, no settled window {}, not plain read-write {}, import list busy {}\n",
        if BEHIND_OFF.load(Ordering::Relaxed) { " off" } else { "" },
        super::tune::behind_quiet(),
        super::tune::behind_us(),
        super::tune::arm_inline_max(),
        LONG_BEHIND[0].load(Ordering::Relaxed),
        LONG_BEHIND[1].load(Ordering::Relaxed),
        v[AB_QUEUED],
        v[AB_DROPPED],
        v[AB_ROUNDS],
        v[AB_OVER],
        v[AB_RUNS],
        v[AB_PAGES],
        v[AB_SKIPPED],
        v[AB_ARMED],
        v[AB_US] as f64 / 1000.0,
        v[AB_TAKES],
        v[AB_TAKE_WRITTEN],
        v[AB_DISARMED],
        na[R_ARMED],
        na[R_USES],
        na[R_QUIET],
        na[R_HELD],
        na[R_WINDOW],
        na[R_RW],
        na[R_BUSY]
    )
}

/// The faults `watch` read: [on pages of the span, those that ended an
/// arming of ours (its level raised there), bursts, pages they disarmed,
/// faults outside the span, ring overflows seen].
static FAULTS: [AtomicU64; 7] = [const { AtomicU64::new(0) }; 7];
const F_SPAN: usize = 0;
const F_ENDED: usize = 1;
const F_BURSTS: usize = 2;
const F_DISARMED: usize = 3;
const F_OUTSIDE: usize = 4;
const F_LOST: usize = 5;
/// Pages of ours the host made read-write with a faulted page's word
/// (`word_ended`).
const F_WORD_DISARMED: usize = 6;

/// Reads the faults the host answered, every millisecond (a thread of its
/// own, hosted runs only): each ends its page's arming (the next waits
/// longer) and restarts its quiet time; a chunk with CHUNK_BURST of them
/// within BURST_MS, or with some while all chunks together had ALL_BURST,
/// is disarmed and held (`burst`).
fn watch() {
    let mut got: Vec<(u64, bool, u64)> = Vec::new();
    let mut seen = crate::hosted::faults();
    // This window's faults per chunk, whether the chunk burst in it, the
    // chunks with any, and all of them.
    let mut counts = vec![0u32; CHUNKS];
    let mut fired = vec![false; CHUNKS];
    // (The code of each chunk's first fault in the window: the burst log.)
    let mut first_rip = vec![0u64; CHUNKS];
    let mut touched: Vec<usize> = Vec::new();
    let mut total = 0u32;
    let mut window = std::time::Instant::now();
    let mut logged = 0u32;
    // `DS_FE2_FAULT_LOG=1` (diagnosis): every fault as "flip page code"
    // into the run's fe2-faults.txt.
    let mut flog = (std::env::var("DS_FE2_FAULT_LOG").ok().as_deref() == Some("1"))
        .then(|| crate::run_dir().and_then(|d| std::fs::File::create(d.join("fe2-faults.txt")).ok()))
        .flatten()
        .map(std::io::BufWriter::new);
    let mut names: std::collections::HashMap<u64, String> = std::collections::HashMap::new();
    // (This read's words with an arming a fault ended, and the level.)
    let mut hit: Vec<(u64, u8)> = Vec::new();
    let mut reconciled = std::time::Instant::now();
    loop {
        std::thread::sleep(std::time::Duration::from_millis(1));
        if hold_census() && HELD_MAP.get().is_some() {
            hold_tick(&mut reconciled);
        }
        if window.elapsed().as_millis() >= BURST_MS {
            for &k in &touched {
                counts[k] = 0;
                fired[k] = false;
            }
            touched.clear();
            total = 0;
            window = std::time::Instant::now();
        }
        let n = crate::hosted::faults();
        if n == seen {
            continue;
        }
        seen = n;
        got.clear();
        if !crate::hosted::drain_faults(&mut got) {
            FAULTS[F_LOST].fetch_add(1, Ordering::Relaxed);
        }
        let frame = super::FRAMES.load(Ordering::Relaxed);
        hit.clear();
        if let Some(f) = flog.as_mut() {
            use std::io::Write;
            for &(page, _, rip) in &got {
                let name = names.entry(rip).or_insert_with(|| describe_code(rip));
                let _ = writeln!(f, "{frame} {page:#x} {name}");
            }
            let _ = f.flush();
        }
        for &(page, write, rip) in &got {
            if write && wb_buckets() {
                note_wb_pages(&WB_CPU_PAGES, page, PAGE);
            }
            note_wb_lazy_fault(page, write, rip);
            if held_mark(page, false) {
                note_hold_fault(page, write, rip);
            }
            // (Pages of view windows: the residency tracks their writes, and
            // only the page written turns volatile.)
            if write && crate::mem::view_window(page) {
                super::resident::host_fault(page);
            }
            let Some(c) = chunk(page, false) else {
                FAULTS[F_OUTSIDE].fetch_add(1, Ordering::Relaxed);
                continue;
            };
            FAULTS[F_SPAN].fetch_add(1, Ordering::Relaxed);
            if traced(page, page + PAGE).is_some() {
                trace_event(page, &format!("a {} fault the host answered (watcher, before its handling)", if write { "write" } else { "read" }));
            }
            let (i, w) = (idx(page), idx(page) / 64);
            let bit = 1u64 << (i % 64);
            if crate::engine::census_on() {
                crate::engine::note_count(crate::engine::C_ARM_FAULT, page);
                if c.gpu[w].load(Ordering::Acquire) & bit != 0 {
                    crate::engine::note_count(crate::engine::C_ARM_FAULT_GPU, page);
                }
            }
            // (A take's `note_arming_ended` may have seen it first.)
            if c.armed[w].fetch_and(!bit, Ordering::AcqRel) & bit != 0 {
                let v = c.level[i].fetch_update(Ordering::AcqRel, Ordering::Acquire, |v| Some((v + 1).min(ARM_MAX_LEVEL))).unwrap_or(0);
                c.cgen.fetch_add(1, Ordering::AcqRel);
                FAULTS[F_ENDED].fetch_add(1, Ordering::Relaxed);
                STATS[S_ARM_ENDED].fetch_add(1, Ordering::Relaxed);
                let wa = page & !(64 * PAGE - 1);
                let lv = (v + 1).min(ARM_MAX_LEVEL);
                match hit.iter_mut().find(|h| h.0 == wa) {
                    Some(h) => h.1 = h.1.max(lv),
                    None => hit.push((wa, lv)),
                }
            }
            c.wrote[i].store(frame as u32 + 1, Ordering::Release);
            c.uses[w].store(0, Ordering::Relaxed);
            let k = ((page - MIRROR_BASE) >> CHUNK_SHIFT) as usize;
            if counts[k] == 0 {
                touched.push(k);
                first_rip[k] = rip;
            }
            counts[k] += 1;
            total += 1;
            if logged < 40 {
                logged += 1;
                crate::info!("fe2: write fault on armed page {page:#x} at {}", describe_code(rip));
            }
        }
        for &(wa, lv) in &hit {
            let n = word_ended(wa, lv, frame);
            FAULTS[F_WORD_DISARMED].fetch_add(n, Ordering::Relaxed);
        }
        for &k in &touched {
            if !fired[k] && (counts[k] >= CHUNK_BURST || total >= ALL_BURST) {
                fired[k] = true;
                burst(k, counts[k], total, frame, first_rip[k]);
            }
        }
    }
}

/// A fault ended an arming in the 64-page word at `wa`, and the host's
/// answer made the word's other armed pages read-write too
/// (`hosted::set_word_mode`): those of ours are no longer armed, their
/// quiet time restarts and their level becomes at least `level` (the
/// faulted page's). Titles write runs of pages (at the gate a word's pages
/// faulted one after another within the millisecond, each page written once
/// in ~200 flips), and the backoff learnt page by page. The pages.
fn word_ended(wa: u64, level: u8, frame: u64) -> u64 {
    let Some(c) = chunk(wa, false) else { return 0 };
    let w = idx(wa) / 64;
    let gone = c.armed[w].load(Ordering::Acquire) & !crate::hosted::armed_word(wa);
    if gone == 0 {
        return 0;
    }
    c.armed[w].fetch_and(!gone, Ordering::AcqRel);
    if let Some(p) = traced(wa, wa + 64 * PAGE) {
        if gone >> (idx(p) % 64) & 1 != 0 {
            trace_event(p, "its word's arming ended by a fault (word_ended)");
        }
    }
    let base = w * 64;
    let mut r = gone;
    while r != 0 {
        let k = r.trailing_zeros() as usize;
        r &= r - 1;
        c.wrote[base + k].store(frame as u32 + 1, Ordering::Release);
        c.level[base + k].fetch_max(level, Ordering::AcqRel);
    }
    c.cgen.fetch_add(1, Ordering::AcqRel);
    gone.count_ones() as u64
}

/// Where code at `rip` is (the title's or a host module's).
fn describe_code(rip: u64) -> String {
    if (crate::mem::GUEST_BASE..crate::mem::GUEST_END).contains(&rip) {
        crate::exception::describe_guest(rip)
    } else {
        crate::exception::host_symbol(rip)
    }
}

/// Chunk `k` had `n` faults within BURST_MS (`total` in all chunks): its
/// armed pages are disarmed and arming there waits CHUNK_HOLD flips (twice
/// as long for each burst within HOLD_MEMORY flips of the last hold's end).
fn burst(k: usize, n: u32, total: u32, frame: u64, rip: u64) {
    let base = MIRROR_BASE + ((k as u64) << CHUNK_SHIFT);
    let Some(c) = chunk(base, false) else { return };
    let until = c.hold_until.load(Ordering::Acquire);
    let level = if until != 0 && frame < until + HOLD_MEMORY { (c.hold_level.load(Ordering::Acquire) + 1).min(CHUNK_HOLD_MAX_LEVEL) } else { 0 };
    c.hold_level.store(level, Ordering::Release);
    let hold = CHUNK_HOLD << level;
    c.hold_until.store(frame + hold, Ordering::Release);
    let mut pages = 0u64;
    {
        // No mapping change meanwhile, no arming between our bits and the
        // host's (ours go first: a take seeing the host's gone and ours set
        // would count the arming as ended by a write).
        let _readers = crate::mem::GUEST_READERS.read_recursive();
        let _arming = super::resident::arm_guard();
        let _lock = crate::hosted::arm_lock();
        let mut pend: Option<(u64, u64)> = None;
        for w in 0..WORDS {
            let ours = c.armed[w].load(Ordering::Acquire);
            if ours == 0 {
                continue;
            }
            let wa = base + (w as u64) * 64 * PAGE;
            let live = ours & crate::hosted::armed_word(wa);
            if live == 0 {
                continue;
            }
            c.armed[w].fetch_and(!live, Ordering::AcqRel);
            if let Some(t) = traced(wa, wa + 64 * PAGE) {
                if live >> (idx(t) % 64) & 1 != 0 {
                    trace_event(t, &format!("disarmed by a fault burst in its chunk ({n} faults, the first at {})", describe_code(rip)));
                }
            }
            let mut r = live;
            while r != 0 {
                let b = r.trailing_zeros() as u64;
                let run = (!(r >> b)).trailing_zeros().min(64 - b as u32) as u64;
                r &= !(if run >= 64 { u64::MAX } else { ((1u64 << run) - 1) << b });
                let (x, y) = (wa + b * PAGE, wa + (b + run) * PAGE);
                pages += run;
                pend = match pend {
                    Some((p, q)) if q == x => Some((p, y)),
                    Some((p, q)) => {
                        crate::hosted::disarm(p, q - p);
                        Some((x, y))
                    }
                    None => Some((x, y)),
                };
            }
        }
        if let Some((p, q)) = pend {
            crate::hosted::disarm(p, q - p);
        }
        c.cgen.fetch_add(1, Ordering::AcqRel);
    }
    let b = FAULTS[F_BURSTS].fetch_add(1, Ordering::Relaxed);
    FAULTS[F_DISARMED].fetch_add(pages, Ordering::Relaxed);
    // (The first 40, then every 16th.)
    if b < 40 || b % 16 == 0 {
        crate::info!("fe2: burst {b}: {n} faults within {BURST_MS} ms in {base:#x}+64M ({total} in all; the first at {}): {pages} armed pages there disarmed, arming there held {hold} flips", describe_code(rip));
    }
}

/// Arms the parts of the candidate run [x, y) inside settled import windows.
fn arm_run(x: u64, y: u64, frame: u64, armable: &Armable) {
    let mut inside = 0u64;
    let listed = armable(x, y, &mut |p, q| {
        inside += q - p;
        arm_part(p, q, frame);
    });
    if !listed {
        not_armed(R_BUSY, (y - x) / PAGE);
    } else if inside < y - x {
        not_armed(R_WINDOW, (y - x - inside) / PAGE);
    }
}

/// Arms [x, y) (inside settled import windows, asked again under the arm
/// lock: an import beginning meanwhile waits for it, then disarms it).
fn arm_part(x: u64, y: u64, frame: u64) {
    // R2: the refusals that need no system call checked first without the
    // locks (most candidates are refused; each took the global arming lock,
    // which the parallel workers then queued on: 16.7% of their busy samples
    // in the 1-1 walk once mapping lookups left it, FINDINGS.md 2026-11-10
    // evening). `plain_rw`'s VirtualQuery stays under the lock: unlocked,
    // the threads' queries met in the kernel's address-space lock and every
    // write-watch take slowed with them (~3.6 -> ~7.8 us). Not arming is
    // always safe (the page is taken again); what passes is checked again.
    if super::tune::r2() {
        // (A page front end 4 holds is no-access: not plain read-write.)
        if host_writing(x, y) || c4_any_held(x, y) {
            not_armed(R_RW, (y - x) / PAGE);
            return;
        }
        if super::exec::fe2_settled_over(x, y) != Some(true) {
            not_armed(R_WINDOW, (y - x) / PAGE);
            return;
        }
        if held(x, y, frame) {
            not_armed(R_HELD, (y - x) / PAGE);
            return;
        }
    }
    // No mapping change meanwhile (`resident::remap_guard`), and no other
    // arm or disarm between the checks and the arming (`hosted::arm_lock`:
    // a file read beginning now waits, then disarms what is armed).
    let _readers = crate::mem::GUEST_READERS.read_recursive();
    let _arming = super::resident::arm_guard();
    let _lock = crate::hosted::arm_lock();
    if host_writing(x, y) || !plain_rw(x, y - x) {
        not_armed(R_RW, (y - x) / PAGE);
        return;
    }
    // (Still inside settled imports: the caller looked without the arm lock,
    // and an import listed since is young.)
    if super::exec::fe2_settled_over(x, y) != Some(true) {
        not_armed(R_WINDOW, (y - x) / PAGE);
        return;
    }
    if held(x, y, frame) {
        not_armed(R_HELD, (y - x) / PAGE);
        return;
    }
    if hold_census() && HELD_MAP.get().is_some() {
        hold_part(x, y);
        return;
    }
    STATS[S_ARM_RUNS].fetch_add(1, Ordering::Relaxed);
    let t = std::time::Instant::now();
    let r = crate::hosted::arm(x, y - x);
    super::hitch::note_since(super::hitch::Kind::Mirror, t, || format!("arming {x:#x}+{:#x}", y - x));
    if r.is_err() {
        STATS[S_ARM_FAILED].fetch_add(1, Ordering::Relaxed);
        return;
    }
    let mut p = x;
    while p < y {
        let e = word_end(p, y);
        if let Some(c) = chunk(p, false) {
            // (`DS_FE2_ARM_GPU=2`: GPU-written pages protected, not armed.)
            let m = word_mask(p, e) & if arm_gpu_os() { !c.gpu[idx(p) / 64].load(Ordering::Acquire) } else { u64::MAX };
            // (Not taken since this arming yet: `take_run`.)
            c.varmed[idx(p) / 64].fetch_and(!m, Ordering::AcqRel);
            c.armed[idx(p) / 64].fetch_or(m, Ordering::AcqRel);
            if let Some(t) = traced(p, e) {
                if m >> (idx(t) % 64) & 1 != 0 {
                    trace_event(t, "armed (arm_part)");
                }
            }
        }
        p = e;
    }
    STATS[S_ARMED].fetch_add((y - x) / PAGE, Ordering::Relaxed);
    if C4_ARMING.with(|f| f.get()) {
        c4_note_armed(x, y);
    } else if BEHIND_ARMING.with(|f| f.get()) {
        behind_note_armed(x, y);
    }
}

/// `arm_part` under the hold census (its checks passed, the locks held):
/// the GPU-written pages of [x, y) held (`hosted::hold`, their bits in the
/// census's map), the others armed against writes, run by run.
fn hold_part(x: u64, y: u64) {
    // (0: not GPU-written, armed; 1: GPU-written and written back lately,
    // held; 2: GPU-written, not written back lately, left alone. Since GPU-
    // written pages are armed (`arm_gpu`, 2026-11-05) the placement check
    // sets no GPU-written bits, so nothing is held: taking every page
    // written back lately instead held label pages the GPU writes in place
    // and lost a label (2026-11-10, run -002728: a 3 s wait, stopped).)
    let frame = super::FRAMES.load(Ordering::Relaxed);
    let quiet = hold_quiet();
    let kind = |p: u64| -> u8 {
        let Some(c) = chunk(p, false) else { return 0 };
        if c.gpu[idx(p) / 64].load(Ordering::Acquire) >> (idx(p) % 64) & 1 == 0 {
            0
        } else if !wb_recent(p) || never_hold(p) {
            2
        } else if quiet > 0 && frame.saturating_sub((c.wrote[idx(p)].load(Ordering::Acquire) as u64).saturating_sub(1)) < quiet {
            3
        } else {
            1
        }
    };
    let mut p = x;
    while p < y {
        let k = kind(p);
        let mut q = p + PAGE;
        while q < y && kind(q) == k {
            q += PAGE;
        }
        if k == 2 || k == 3 {
            if k == 2 {
                HOLD_STALE.fetch_add((q - p) / PAGE, Ordering::Relaxed);
            } else {
                HOLD_YOUNG.fetch_add((q - p) / PAGE, Ordering::Relaxed);
            }
            p = q;
            continue;
        }
        let g = k == 1;
        STATS[S_ARM_RUNS].fetch_add(1, Ordering::Relaxed);
        let t = std::time::Instant::now();
        let r = if g { crate::hosted::hold(p, q - p) } else { crate::hosted::arm(p, q - p) };
        super::hitch::note_since(super::hitch::Kind::Mirror, t, || format!("{} {p:#x}+{:#x}", if g { "holding" } else { "arming" }, q - p));
        if r.is_err() {
            STATS[S_ARM_FAILED].fetch_add(1, Ordering::Relaxed);
            if g {
                HOLD[H_FAILED].fetch_add(1, Ordering::Relaxed);
            }
            p = q;
            continue;
        }
        let mut a = p;
        while a < q {
            let e = word_end(a, q);
            if let Some(c) = chunk(a, false) {
                c.varmed[idx(a) / 64].fetch_and(!word_mask(a, e), Ordering::AcqRel);
                c.armed[idx(a) / 64].fetch_or(word_mask(a, e), Ordering::AcqRel);
                if let Some(t) = traced(a, e) {
                    trace_event(t, "armed (a run)");
                }
            }
            a = e;
        }
        if g {
            let mut n = 0u64;
            for a in (p..q).step_by(PAGE as usize) {
                if !held_mark(a, true) {
                    n += 1;
                }
            }
            HOLD[H_PAGES].fetch_add(n, Ordering::Relaxed);
            HOLD[H_RUNS].fetch_add(1, Ordering::Relaxed);
            HOLD[H_HELD_NOW].fetch_add(n, Ordering::Relaxed);
            crate::engine::note_value(crate::engine::C_HELD, p, n);
            crate::engine::note_held_record(p, n);
        }
        STATS[S_ARMED].fetch_add((q - p) / PAGE, Ordering::Relaxed);
        p = q;
    }
}

/// Whether a chunk of [x, y) holds its arming at `frame` (`burst`).
fn held(x: u64, y: u64, frame: u64) -> bool {
    let mut p = x & !(CHUNK_BYTES - 1);
    while p < y {
        if chunk(p, false).is_some_and(|c| frame < c.hold_until.load(Ordering::Acquire)) {
            return true;
        }
        p += CHUNK_BYTES;
    }
    false
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

/// Generations of the words the runs touch, in order (taken right before
/// their batch is queued: `uploaded`).
pub fn generations(runs: &[(u64, u64)], out: &mut Vec<u32>) {
    out.clear();
    for &(a, b) in runs {
        let mut x = a;
        while x < b {
            let e = word_end(x, b);
            out.push(chunk(x, true).map_or(0, |c| c.wgen[idx(x) / 64].load(Ordering::Acquire)));
            x = e;
        }
    }
}

/// The batch whose prologue uploads `runs` was queued: their pages are
/// valid and clean in epoch `ep` (loaded before it was queued), except in
/// words invalidated since `gens` (`generations`) were taken (the upload may
/// have run before that write).
pub fn uploaded(runs: &[(u64, u64)], gens: &[u32], ep: u32) {
    let mut k = 0;
    for &(a, b) in runs {
        let mut x = a;
        while x < b {
            let e = word_end(x, b);
            let g = gens.get(k).copied();
            k += 1;
            if let Some(c) = chunk(x, true) {
                if Some(c.wgen[idx(x) / 64].load(Ordering::Acquire)) == g {
                    set_clean(x, e, ep, true);
                    STATS[S_VALIDATED].fetch_add((e - x) / PAGE, Ordering::Relaxed);
                    if let Some(p) = traced(x, e) {
                        trace_event(p, &format!("its upload's batch queued: validated in epoch {ep}"));
                    }
                } else {
                    STATS[S_RACED].fetch_add((e - x) / PAGE, Ordering::Relaxed);
                    if let Some(p) = traced(x, e) {
                        trace_event(p, &format!("its upload's batch queued: raced (word generation {g:?} then)"));
                    }
                }
            }
            x = e;
        }
    }
}

/// A write-watch take over guest [base, base+size) (anyone's: `precache`):
/// the pages it reports (None: unknown, all of it) are no longer valid.
pub fn marks_taken(r: Option<()>, base: u64, size: u64, pages: &[u64]) {
    if !on() {
        return;
    }
    match r {
        Some(()) => cpu_written(pages),
        None => invalidate(base, size),
    }
}

/// `DS_FE2_TRACE_PAGE=<hex page>` (diagnosis, boot; 2026-11-21): front end 2's events on that
/// guest page logged with its state (the first 4,000): takes and what they found, writes found,
/// invalidations, uploads recorded and validated, run claims, armings, faults.
pub fn trace_page() -> u64 {
    static P: std::sync::OnceLock<u64> = std::sync::OnceLock::new();
    *P.get_or_init(|| std::env::var("DS_FE2_TRACE_PAGE").ok().and_then(|v| u64::from_str_radix(v.trim().trim_start_matches("0x"), 16).ok()).map_or(0, |p| p & !(PAGE - 1)))
}

/// The traced page when it lies in [a, b).
#[inline]
pub fn traced(a: u64, b: u64) -> Option<u64> {
    let p = trace_page();
    (p != 0 && a <= p && p < b).then_some(p)
}

/// Logs an event on the traced page `p` with its state.
pub fn trace_event(p: u64, what: &str) {
    static N: AtomicU64 = AtomicU64::new(0);
    if N.fetch_add(1, Ordering::Relaxed) >= 4000 {
        return;
    }
    let st = match chunk(p, false) {
        Some(c) => {
            let (i, w, bit) = (idx(p), idx(p) / 64, idx(p) % 64);
            format!(
                "valid {}, armed {} host {}, cgen {} wgen {} pgen {}, wrote {}, clean {} wclean {}, epoch {}",
                c.valid[w].load(Ordering::Acquire) >> bit & 1,
                c.armed[w].load(Ordering::Acquire) >> bit & 1,
                crate::hosted::armed_word(p & !(64 * PAGE - 1)) >> bit & 1,
                c.cgen.load(Ordering::Acquire),
                c.wgen[w].load(Ordering::Acquire),
                c.pgen[i].load(Ordering::Acquire),
                c.wrote[i].load(Ordering::Acquire),
                c.clean[i].load(Ordering::Acquire),
                c.wclean[w].load(Ordering::Acquire),
                super::precache::epoch()
            )
        }
        None => "no chunk".into(),
    };
    crate::info!("fe2 trace {p:#x} at flip {} on {}: {what} [{st}]", super::FRAMES.load(Ordering::Relaxed), std::thread::current().name().unwrap_or("?"));
}

/// CPU writes to these pages were found (by a take).
pub fn cpu_written(pages: &[u64]) {
    if !on() || pages.is_empty() {
        return;
    }
    let f = super::FRAMES.load(Ordering::Relaxed) as u32 + 1;
    let mut n = 0;
    for &p in pages {
        let Some(c) = chunk(p, false) else { continue };
        let (i, w) = (idx(p), idx(p) / 64);
        let bit = 1u64 << (i % 64);
        // (The chunk's generation only for a page that was valid: ranges
        // found fine cover valid pages only, and every thread's range cache
        // reads it; an upload in flight still sees the word's generation.)
        if c.valid[w].fetch_and(!bit, Ordering::AcqRel) & bit != 0 {
            c.cgen.fetch_add(1, Ordering::AcqRel);
        }
        c.wgen[w].fetch_add(1, Ordering::AcqRel);
        c.wrote[i].store(f, Ordering::Release);
        c.pgen[i].fetch_add(1, Ordering::AcqRel);
        c.uses[w].store(0, Ordering::Relaxed);
        n += 1;
        if traced(p, p + PAGE).is_some() {
            trace_event(p, "a take found it written (cpu_written)");
        }
    }
    STATS[S_INVAL_TAKEN].fetch_add(n, Ordering::Relaxed);
}

/// Guest [base, base+len) may have changed without a write mark (file
/// reads, CP writes the CPU performs, mapping changes): M no longer holds
/// its pages.
pub fn invalidate(base: u64, len: u64) {
    if !on() || len == 0 {
        return;
    }
    let lo = (base & !(PAGE - 1)).max(MIRROR_BASE);
    let hi = base.saturating_add(len).next_multiple_of(PAGE).min(MIRROR_END);
    if lo >= hi {
        return;
    }
    STATS[S_INVAL_CALLS].fetch_add(1, Ordering::Relaxed);
    let f = super::FRAMES.load(Ordering::Relaxed) as u32 + 1;
    let mut x = lo;
    while x < hi {
        let e = word_end(x, hi);
        if let Some(c) = chunk(x, false) {
            let w = idx(x) / 64;
            let mask = word_mask(x, e);
            // (The chunk's generation only where a page was valid, as
            // `cpu_written`.)
            if c.valid[w].fetch_and(!mask, Ordering::AcqRel) & mask != 0 {
                c.cgen.fetch_add(1, Ordering::AcqRel);
            }
            c.wgen[w].fetch_add(1, Ordering::AcqRel);
            for k in idx(x)..idx(x) + ((e - x) / PAGE) as usize {
                c.wrote[k].store(f, Ordering::Release);
                c.pgen[k].fetch_add(1, Ordering::AcqRel);
            }
            c.uses[w].store(0, Ordering::Relaxed);
            x = e;
        } else {
            // No chunk: nothing valid in it.
            x = ((x >> CHUNK_SHIFT) + 1) << CHUNK_SHIFT;
        }
    }
    if let Some(p) = traced(lo, hi) {
        trace_event(p, &format!("invalidated with {base:#x}+{len:#x}"));
    }
}

/// The write-back's pipeline, its one descriptor set (the written-bit map
/// and the counters never change), the counters (host visible), whether
/// its kernel empties the touched-page list itself and the counters it
/// adds to in video memory (copied into the host-visible ones every
/// COUNTER_COPY_EVERY write-backs).
struct Scatter {
    pipeline: vk::Pipeline,
    layout: vk::PipelineLayout,
    /// Its descriptor sets: the main queue's written-bit map, the compute
    /// family's queue's (null without it).
    set: vk::DescriptorSet,
    set_c: vk::DescriptorSet,
    counters: *mut u32,
    host: vk::Buffer,
    /// The counters in video memory (two host-memory atomics a page written
    /// back cost ~2.3 ms of GPU a vista frame), copied into `host` now and
    /// then.
    dev: vk::Buffer,
}

/// Write-backs between copies of the video-memory counters into the
/// host-visible ones (the status line's counts lag by at most that many).
const COUNTER_COPY_EVERY: u64 = 64;

static SCATTERS_RECORDED: AtomicU64 = AtomicU64::new(0);

unsafe impl Send for Scatter {}
unsafe impl Sync for Scatter {}

static SCATTER: std::sync::OnceLock<Scatter> = std::sync::OnceLock::new();

fn make_scatter(marks: vk::Buffer, marks_c: Option<vk::Buffer>) -> Result<Scatter, String> {
    let c = ctx();
    unsafe {
        let bindings: Vec<vk::DescriptorSetLayoutBinding> = (0..2)
            .map(|b| vk::DescriptorSetLayoutBinding::default().binding(b).descriptor_type(vk::DescriptorType::STORAGE_BUFFER).descriptor_count(1).stage_flags(vk::ShaderStageFlags::COMPUTE))
            .collect();
        let set_layout = c.device.create_descriptor_set_layout(&vk::DescriptorSetLayoutCreateInfo::default().bindings(&bindings), None).map_err(|e| format!("scatter set layout: {e}"))?;
        let layouts = [set_layout];
        // (40 bytes: the mirror's and the table's addresses, the variants'
        // own, the zero-known bitmap's at 32.)
        let ranges = [vk::PushConstantRange { stage_flags: vk::ShaderStageFlags::COMPUTE, offset: 0, size: 40 }];
        let layout = c.device.create_pipeline_layout(&vk::PipelineLayoutCreateInfo::default().set_layouts(&layouts).push_constant_ranges(&ranges), None).map_err(|e| format!("scatter layout: {e}"))?;
        // (The kernel empties the touched-page list itself; `DS_WB_BUCKETS`
        // also counts by bucket.)
        // (`DS_FE2_HOLD`: the hold census's kernel with its map in host
        // memory, read by address.)
        if held_map_wanted() && HELD_MAP.get().is_none() {
            let bytes = (HELD_MAP_WORDS * 4) as u64;
            match c.host_buffer(bytes, vk::BufferUsageFlags::STORAGE_BUFFER) {
                Ok((buf, _mem, ptr)) => {
                    std::ptr::write_bytes(ptr, 0, bytes as usize);
                    let addr = c.buffer_address(buf);
                    if addr != 0 {
                        let _ = HELD_MAP.set(HeldMap { _buf: buf, words: ptr as usize, addr });
                    } else {
                        crate::warn!("fe2: hold census off: no buffer device addresses");
                    }
                }
                Err(e) => crate::warn!("fe2: hold census off: {e}"),
            }
        }
        let split_census = wb_buckets() && std::env::var("DS_WB_BUCKETS_SPLIT").ok().as_deref() == Some("1");
        let code = if split_census {
            rdna::kernels::mirror_scatter_split_counting()
        } else if wb_buckets() {
            rdna::kernels::mirror_scatter_counting()
        } else if wb_same_mode() == 1 {
            rdna::kernels::mirror_scatter_same_census()
        } else if c4_wanted() && HELD_MAP.get().is_some() {
            // (Front end 4: held pages left pending in the mirror.)
            rdna::kernels::mirror_scatter_held()
        } else if HELD_MAP.get().is_some() {
            rdna::kernels::mirror_scatter_hold_census()
        } else {
            // (Partly marked lines stored one dword per line in each store
            // instruction: ~6-7x faster into host memory than a line with
            // holes from one instruction, the same dwords; FINDINGS.md
            // 2026-11-09 "Holey write-backs". `DS_WB_OLD=1`, diagnosis: the
            // write-back before, for the bench's comparison.)
            if std::env::var("DS_WB_OLD").ok().as_deref() == Some("1") {
                rdna::kernels::mirror_scatter_reset()
            } else {
                rdna::kernels::mirror_scatter_split()
            }
        };
        let make = |code: &[u32], key: u64| -> Result<vk::Pipeline, String> {
            let module = c.device.create_shader_module(&vk::ShaderModuleCreateInfo::default().code(code), None).map_err(|e| format!("scatter module: {e}"))?;
            let stage = vk::PipelineShaderStageCreateInfo::default().stage(vk::ShaderStageFlags::COMPUTE).module(module).name(c"main");
            let ci = vk::ComputePipelineCreateInfo::default().stage(stage).layout(layout);
            let pipeline = super::pcache::with_cache('i', key, |cache| c.device.create_compute_pipelines(cache, &[ci], None)).map_err(|(_, e)| format!("scatter pipeline: {e}"));
            c.device.destroy_shader_module(module, None);
            Ok(pipeline?[0])
        };
        let pipeline = make(&code, if split_census { 0xfe210001 } else if wb_buckets() { 27 } else if wb_same_mode() == 1 { 34 } else if c4_wanted() && HELD_MAP.get().is_some() { 44 } else if HELD_MAP.get().is_some() { 28 } else if std::env::var("DS_WB_OLD").ok().as_deref() == Some("1") { 20 } else { 42 })?;
        if wb_same_mode() == 2 && SAME_SKIP.get().is_none() {
            // (The tables first: zeroed, and the buffer of zeros.)
            let mut tables = Vec::new();
            for _ in [Some(marks), marks_c].into_iter().flatten() {
                let (b, _m, a) = c.device_buffer_addressed(HASH_TABLE_BYTES, vk::BufferUsageFlags::STORAGE_BUFFER | vk::BufferUsageFlags::TRANSFER_DST)?;
                if a == 0 {
                    return Err("the hash tables need device addresses".into());
                }
                tables.push((b, a));
            }
            let (zeros, _zm) = c.device_buffer(HASH_ZERO_BYTES, vk::BufferUsageFlags::TRANSFER_SRC | vk::BufferUsageFlags::TRANSFER_DST)?;
            let tb: Vec<vk::Buffer> = tables.iter().map(|t| t.0).collect();
            super::resident::one_off_now(|cmd| {
                for &t in &tb {
                    c.device.cmd_fill_buffer(cmd, t, 0, vk::WHOLE_SIZE, 0);
                }
                c.device.cmd_fill_buffer(cmd, zeros, 0, vk::WHOLE_SIZE, 0);
            })?;
            let _ = HASHES.set(HashTables { tables, zeros });
            let _ = SAME_SKIP.set(make(&rdna::kernels::mirror_scatter_same_skip(), 36)?);
        }
        let sizes = [vk::DescriptorPoolSize { ty: vk::DescriptorType::STORAGE_BUFFER, descriptor_count: 4 }];
        let pool = c.device.create_descriptor_pool(&vk::DescriptorPoolCreateInfo::default().max_sets(2).pool_sizes(&sizes), None).map_err(|e| format!("scatter pool: {e}"))?;
        let bytes = 4 * rdna::kernels::SCATTER_COUNTERS as u64;
        let (cbuf, _cmem, cptr) = c.host_buffer(bytes, vk::BufferUsageFlags::STORAGE_BUFFER | vk::BufferUsageFlags::TRANSFER_DST)?;
        std::ptr::write_bytes(cptr, 0, bytes as usize);
        let (dev, _dmem) = c.device_buffer(bytes, vk::BufferUsageFlags::STORAGE_BUFFER | vk::BufferUsageFlags::TRANSFER_SRC | vk::BufferUsageFlags::TRANSFER_DST)?;
        super::resident::one_off_now(|cmd| c.device.cmd_fill_buffer(cmd, dev, 0, bytes, 0))?;
        // One set per map; the counters are shared (atomics).
        let make_set = |m: vk::Buffer| -> Result<vk::DescriptorSet, String> {
            let set = c.device.allocate_descriptor_sets(&vk::DescriptorSetAllocateInfo::default().descriptor_pool(pool).set_layouts(&layouts)).map_err(|e| format!("scatter set: {e}"))?[0];
            let infos = [vk::DescriptorBufferInfo { buffer: m, offset: 0, range: super::resident::MARK_BYTES }, vk::DescriptorBufferInfo { buffer: dev, offset: 0, range: bytes }];
            let writes: Vec<vk::WriteDescriptorSet> = infos
                .iter()
                .enumerate()
                .map(|(k, info)| vk::WriteDescriptorSet::default().dst_set(set).dst_binding(k as u32).descriptor_type(vk::DescriptorType::STORAGE_BUFFER).buffer_info(std::slice::from_ref(info)))
                .collect();
            c.device.update_descriptor_sets(&writes, &[]);
            Ok(set)
        };
        let set = make_set(marks)?;
        let set_c = match marks_c {
            Some(m) => make_set(m)?,
            None => vk::DescriptorSet::null(),
        };
        Ok(Scatter { pipeline, layout, set, set_c, counters: cptr as *mut u32, host: cbuf, dev })
    }
}

/// `DS_WB_BUCKETS=1` (diagnosis): the write-back counts the dwords and
/// pages it writes back per 1 MiB of the span (`rdna::kernels::
/// mirror_scatter_counting`, the map's aux region), and status reports
/// class those buckets by the pages the CPU wrote (takes, write faults) and
/// the command processors read (`sync_for`) since the last report: how much
/// of the volume lies where neither touches it (pages a write-back could
/// leave out while their reads are detected).
pub fn wb_buckets() -> bool {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ON.get_or_init(|| std::env::var("DS_WB_BUCKETS").ok().as_deref() == Some("1"))
}

/// Whether the write-back compares pages with guest memory (`wb_same_mode`
/// 1 or 2).
pub fn wb_same() -> bool {
    wb_same_mode() != 0
}

/// The write-back's handling of pages it would store unchanged: 0 (the
/// default) stores every page, as do the diagnoses with write-back kernels
/// of their own (`DS_WB_BUCKETS=1`, `DS_FE2_HOLD=1`); `DS_WB_SAME=2` (the
/// default from efb0a1d until 2026-11-05 morning) leaves them out
/// (`rdna::kernels::mirror_scatter_same_skip`: a 64-bit hash of the page's
/// mirror copy against the one kept at its last store; FINDINGS.md 2026-11-05
/// small hours), the plain write-back's pipeline made too and the control
/// `wbs on|off` choosing (in-run A/Bs); `DS_WB_SAME=1` (diagnosis) counts
/// them by reading guest memory's copy (`mirror_scatter_same_census`).
pub fn wb_same_mode() -> u8 {
    static M: std::sync::OnceLock<u8> = std::sync::OnceLock::new();
    *M.get_or_init(|| {
        if wb_buckets() || std::env::var("DS_FE2_HOLD").ok().as_deref() == Some("1") {
            return 0;
        }
        // (Off by default since 2026-11-05: a page the CPU rewrote after a
        // batch's check, which the batch then wrote again with the values it
        // held, was left out and guest memory kept the CPU's words; FINDINGS.md
        // 2026-11-05 morning, later.)
        std::env::var("DS_WB_SAME").ok().and_then(|v| v.parse().ok()).filter(|&v: &u8| v <= 2).unwrap_or(0)
    })
}

/// `DS_WB_SAME=2`: the skipping write-back's pipeline, and whether it is used
/// (`wbs on|off`).
static SAME_SKIP: std::sync::OnceLock<vk::Pipeline> = std::sync::OnceLock::new();
static SAME_SKIP_ON: AtomicBool = AtomicBool::new(true);

/// `DS_WB_SAME=2`: the pages' hashes at their last store, a table per
/// written-bit map (device-local, 8 bytes a page of the span: buffer,
/// address), and a buffer of zeros the uploads' clears copy from.
struct HashTables {
    tables: Vec<(vk::Buffer, u64)>,
    zeros: vk::Buffer,
}

static HASHES: std::sync::OnceLock<HashTables> = std::sync::OnceLock::new();
const HASH_TABLE_BYTES: u64 = (MIRROR_END - MIRROR_BASE) / PAGE * 8;
const HASH_ZERO_BYTES: u64 = 64 << 10;
/// Tables to clear before their family's next write-back (mapping changes,
/// `wbs on`): a bit per table.
static HASH_CLEAR: AtomicU32 = AtomicU32::new(0);

/// Guest [base, base+len)'s mapping changed: its pages' kept hashes dropped
/// before each family's next write-back (an upload may not precede GPU work
/// that only writes there).
pub fn hash_clear_range(base: u64, len: u64) {
    // (The zero-known bits too: guest memory's copy may be other than zeros.)
    super::zk::cpu_dirty_range(base, len);
    if HASHES.get().is_none() || len == 0 {
        return;
    }
    let lo = base.max(MIRROR_BASE) & !(PAGE - 1);
    let hi = base.saturating_add(len).min(MIRROR_END).next_multiple_of(PAGE);
    if lo < hi {
        let mut q = HASH_CLEAR_RANGES.lock();
        for l in q.iter_mut() {
            l.push((lo, hi));
        }
    }
}

/// Ranges whose hashes to clear, a list per table.
static HASH_CLEAR_RANGES: parking_lot::Mutex<[Vec<(u64, u64)>; 2]> = parking_lot::Mutex::new([Vec::new(), Vec::new()]);

/// Every page's kept hash dropped before the next write-backs (`wbs on`: the
/// tables were not kept meanwhile). (Guest memory changed under a page --
/// a CPU write, a mapping change -- reaches the mirror only by an upload,
/// which clears the page's hash: `hash_clear_uploads`.)
pub fn hash_clear_all() {
    if HASHES.get().is_some() {
        HASH_CLEAR.store(u32::MAX, Ordering::Release);
    }
}

/// Uploads into the mirror were recorded into `cmd` (`exec::record_copies`):
/// their pages' kept hashes cleared there too, after them.
pub fn hash_clear_uploads(cmd: vk::CommandBuffer, copies: &[(vk::Buffer, vk::Buffer, vk::BufferCopy)]) {
    let Some(h) = HASHES.get() else { return };
    let Some((mbuf, _)) = super::resident::mirror_buffer() else { return };
    // (The uploads' pages as disjoint runs: several uploads may cover a page.)
    let mut runs: Vec<(u64, u64)> = copies.iter().filter(|c| c.1 == mbuf).map(|c| (c.2.dst_offset / PAGE, (c.2.dst_offset + c.2.size).div_ceil(PAGE))).collect();
    runs.sort_unstable();
    let mut merged: Vec<(u64, u64)> = Vec::with_capacity(runs.len());
    for (a, b) in runs {
        match merged.last_mut() {
            Some(l) if a <= l.1 => l.1 = l.1.max(b),
            _ => merged.push((a, b)),
        }
    }
    let mut regions: Vec<vk::BufferCopy> = Vec::new();
    for (p0, p1) in merged {
        let mut p = p0;
        while p < p1 {
            let n = (p1 - p).min(HASH_ZERO_BYTES / 8);
            regions.push(vk::BufferCopy { src_offset: 0, dst_offset: p * 8, size: n * 8 });
            p += n;
        }
    }
    if regions.is_empty() {
        return;
    }
    let c = ctx();
    unsafe {
        // (Transfer stages only: the copy queue records these too.)
        let mb = vk::MemoryBarrier::default().src_access_mask(vk::AccessFlags::TRANSFER_WRITE).dst_access_mask(vk::AccessFlags::TRANSFER_WRITE);
        c.device.cmd_pipeline_barrier(cmd, vk::PipelineStageFlags::TRANSFER, vk::PipelineStageFlags::TRANSFER, vk::DependencyFlags::empty(), &[mb], &[], &[]);
        for &(t, _) in &h.tables {
            c.device.cmd_copy_buffer(cmd, h.zeros, t, &regions);
        }
    }
}

/// Control `wbs on|off` (`DS_WB_SAME=2`).
pub fn wbs_control(what: &str) -> Result<String, String> {
    if SAME_SKIP.get().is_none() {
        return Err("wbs: no skipping write-back (`DS_WB_SAME=2`)".into());
    }
    match what {
        "on" => {
            // (The tables were not kept while off.)
            hash_clear_all();
            SAME_SKIP_ON.store(true, Ordering::Release);
            Ok("write-backs leave out pages stored unchanged".into())
        }
        "off" => {
            SAME_SKIP_ON.store(false, Ordering::Release);
            Ok("write-backs store every page".into())
        }
        _ => Err("usage: wbs on|off".into()),
    }
}

/// Status line of `DS_WB_SAME` (since the last one).
fn wb_same_line() -> String {
    if !wb_same() || SCATTER.get().is_none() {
        return String::new();
    }
    static LAST: parking_lot::Mutex<(u64, u64, u64, u64, u64)> = parking_lot::Mutex::new((0, 0, 0, 0, 0));
    let frame = super::FRAMES.load(Ordering::Relaxed);
    let now = [WB64[0].load(Ordering::Relaxed), WB64[1].load(Ordering::Relaxed), WB64[2].load(Ordering::Relaxed), WB64[3].load(Ordering::Relaxed)];
    let mut l = LAST.lock();
    let flips = frame.saturating_sub(l.0).max(1) as f64;
    let d = [now[0].saturating_sub(l.1), now[1].saturating_sub(l.2), now[2].saturating_sub(l.3), now[3].saturating_sub(l.4)];
    *l = (frame, now[0], now[1], now[2], now[3]);
    format!(
        "    write-backs unchanged (`DS_WB_SAME={}`{}, over {flips:.0} flips): {:.0} of {:.0} pages a flip ({:.1}%) {} every marked dword as guest memory held it, {:.3} of {:.3} M dwords ({:.1}%)\n",
        wb_same_mode(),
        if wb_same_mode() == 2 { if SAME_SKIP_ON.load(Ordering::Relaxed) { ", on" } else { ", off" } } else { "" },
        d[3] as f64 / flips,
        d[0] as f64 / flips,
        100.0 * d[3] as f64 / d[0].max(1) as f64,
        if wb_same_mode() == 2 { "left out, holding" } else { "stored" },
        d[2] as f64 / 1e6 / flips,
        if wb_same_mode() == 2 { (d[1] + d[2]) as f64 / 1e6 / flips } else { d[1] as f64 / 1e6 / flips },
        if wb_same_mode() == 2 { 100.0 * d[2] as f64 / (d[1] + d[2]).max(1) as f64 } else { 100.0 * d[2] as f64 / d[1].max(1) as f64 }
    )
}

/// `DS_FE2_HOLD=1` (diagnosis, the hold census; FRONTEND2.md "GPU-held
/// pages"; hosted runs): the GPU-written pages the arming picks are held
/// through the host (no-access, their held bits set) instead of armed
/// read-only, the host answering their faults at once as for armed pages
/// (no thread waits; their words are written back as ever); the write-back
/// counts the dwords holding would leave out (`rdna::kernels::
/// mirror_scatter_hold_census`, a bitmap of held pages in host memory);
/// each fault on a held page is counted by origin and access, with the wait
/// an exact release would have had for GPU work writing the page then.
pub fn hold_census() -> bool {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ON.get_or_init(|| std::env::var("DS_FE2_HOLD").ok().as_deref() == Some("1") && !wb_buckets())
}

/// The hold census's bitmap: a bit per page of the span (bit i of word i /
/// 32), in host memory the write-back reads at `addr`.
struct HeldMap {
    _buf: vk::Buffer,
    words: usize,
    addr: u64,
}

static HELD_MAP: std::sync::OnceLock<HeldMap> = std::sync::OnceLock::new();
const HELD_MAP_WORDS: usize = ((MIRROR_END - MIRROR_BASE) / PAGE / 32) as usize;

/// Sets (or clears) page `page`'s bit in the hold census's map: whether it
/// was set.
fn held_mark(page: u64, on: bool) -> bool {
    let Some(m) = HELD_MAP.get() else { return false };
    if !(MIRROR_BASE..MIRROR_END).contains(&page) {
        return false;
    }
    let i = ((page - MIRROR_BASE) / PAGE) as usize;
    // SAFETY: HELD_MAP_WORDS host-memory words, mapped for the process's life.
    let w = unsafe { &*(m.words as *const AtomicU32).add(i / 32) };
    let bit = 1u32 << (i % 32);
    let old = if on { w.fetch_or(bit, Ordering::AcqRel) } else { w.fetch_and(!bit, Ordering::AcqRel) };
    old & bit != 0
}

/// The hold census's counts: pages held, runs, runs that failed; faults on
/// held pages [the title, ours, other code] x [read, write]; of them with
/// GPU work writing the page in flight, none in flight; waits resolved,
/// their total and longest (us); holds ended otherwise (disarms, mapping
/// changes, word answers: the reconcile).
static HOLD: [AtomicU64; 17] = [const { AtomicU64::new(0) }; 17];
/// (Pages uploaded into the mirror while held.)
const H_UPLOADED: usize = 16;
/// GPU-written pages the census did not hold: not written back lately.
static HOLD_STALE: AtomicU64 = AtomicU64::new(0);
/// Pages written back within the last two periods of `hold_thread` (a bit
/// per page of the span), and how many at its last look.
static WB_RECENT: std::sync::OnceLock<Box<[AtomicU32]>> = std::sync::OnceLock::new();
static WB_RECENT_PAGES: AtomicU64 = AtomicU64::new(0);

/// `DS_FE2_HOLD_ONCE=1` (the hold census): a page that faulted while held
/// is never held again (`NO_HOLD`, a bit per page of the span).
fn hold_once() -> bool {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ON.get_or_init(|| std::env::var("DS_FE2_HOLD_ONCE").ok().as_deref() == Some("1"))
}

static NO_HOLD: std::sync::OnceLock<Box<[AtomicU32]>> = std::sync::OnceLock::new();
static NO_HOLD_PAGES: AtomicU64 = AtomicU64::new(0);

fn no_hold() -> &'static [AtomicU32] {
    NO_HOLD.get_or_init(|| (0..HELD_MAP_WORDS).map(|_| AtomicU32::new(0)).collect())
}

/// `DS_FE2_HOLD_QUIET=<flips>` (the hold census): a GPU-written page is held
/// only after that many flips without a CPU write seen on it (0: the
/// arming's own quiet time).
fn hold_quiet() -> u64 {
    static Q: std::sync::OnceLock<u64> = std::sync::OnceLock::new();
    *Q.get_or_init(|| std::env::var("DS_FE2_HOLD_QUIET").ok().and_then(|v| v.parse().ok()).unwrap_or(0))
}

/// Pages the census did not hold for want of its own quiet time.
static HOLD_YOUNG: AtomicU64 = AtomicU64::new(0);

/// Whether page `p` faulted while held before (`hold_once`).
fn never_hold(p: u64) -> bool {
    if !hold_once() || !(MIRROR_BASE..MIRROR_END).contains(&p) {
        return false;
    }
    let i = ((p - MIRROR_BASE) / PAGE) as usize;
    no_hold()[i / 32].load(Ordering::Relaxed) >> (i % 32) & 1 != 0
}

/// Whether page `p` was written back lately (`WB_RECENT`).
fn wb_recent(p: u64) -> bool {
    let Some(r) = WB_RECENT.get() else { return false };
    if !(MIRROR_BASE..MIRROR_END).contains(&p) {
        return false;
    }
    let i = ((p - MIRROR_BASE) / PAGE) as usize;
    r[i / 32].load(Ordering::Relaxed) >> (i % 32) & 1 != 0
}

/// The hold census's thread: every 500 ms the written-back bits of both maps'
/// aux regions read back and cleared (a write-back racing it may lose its
/// bit to the clear), `WB_RECENT` made of this period's and the last one's.
fn hold_thread() {
    let c = ctx();
    let words = HELD_MAP_WORDS as u64;
    let at = rdna::kernels::MARK_AUX_AT as u64 * 4;
    let recent = WB_RECENT.get_or_init(|| (0..HELD_MAP_WORDS).map(|_| AtomicU32::new(0)).collect());
    let Ok((hbuf, _hmem, hptr)) = c.host_buffer(2 * words * 4, vk::BufferUsageFlags::TRANSFER_DST) else {
        crate::warn!("fe2: hold census: no host buffer for the written-back bits");
        return;
    };
    let mut prev = vec![0u32; HELD_MAP_WORDS];
    loop {
        std::thread::sleep(std::time::Duration::from_millis(500));
        let maps: Vec<vk::Buffer> = [false, true].iter().filter_map(|&on_c| super::resident::marks_for(on_c)).collect();
        let r = super::resident::one_off_now(|cmd| unsafe {
            for (k, &m) in maps.iter().enumerate() {
                c.device.cmd_copy_buffer(cmd, m, hbuf, &[vk::BufferCopy { src_offset: at, dst_offset: k as u64 * words * 4, size: words * 4 }]);
            }
            let mb = vk::MemoryBarrier::default().src_access_mask(vk::AccessFlags::TRANSFER_READ).dst_access_mask(vk::AccessFlags::TRANSFER_WRITE);
            c.device.cmd_pipeline_barrier(cmd, vk::PipelineStageFlags::TRANSFER, vk::PipelineStageFlags::TRANSFER, vk::DependencyFlags::empty(), &[mb], &[], &[]);
            for &m in &maps {
                c.device.cmd_fill_buffer(cmd, m, at, words * 4, 0);
            }
        });
        if r.is_err() {
            continue;
        }
        let mut n = 0u64;
        // (This period's written-back pages by engine allocation, and the
        // pages with their allocator for the census by call site.)
        let mut by_rec: std::collections::HashMap<(u64, u64, u8), u64> = std::collections::HashMap::new();
        let mut wb_pages: Vec<(u64, u8)> = Vec::new();
        for i in 0..HELD_MAP_WORDS {
            let mut cur = 0u32;
            for k in 0..maps.len() {
                cur |= unsafe { std::ptr::read_volatile((hptr as *const u32).add(k * HELD_MAP_WORDS + i)) };
            }
            let v = cur | prev[i];
            recent[i].store(v, Ordering::Relaxed);
            n += v.count_ones() as u64;
            prev[i] = cur;
            let mut r = cur;
            while r != 0 {
                let b = r.trailing_zeros() as u64;
                r &= r - 1;
                let page = MIRROR_BASE + (i as u64 * 32 + b) * PAGE;
                let k = crate::engine::record_key(page);
                *by_rec.entry(k).or_insert(0) += 1;
                wb_pages.push((page, k.2));
            }
        }
        crate::engine::note_wb_period(by_rec);
        crate::engine::note_wb_sites(&wb_pages);
        WB_RECENT_PAGES.store(n, Ordering::Relaxed);
        c4_round();
    }
}
const H_PAGES: usize = 0;
const H_RUNS: usize = 1;
const H_FAILED: usize = 2;
const H_FAULTS: usize = 3;
const H_INFLIGHT: usize = 9;
const H_NONE: usize = 10;
const H_WAITS: usize = 11;
const H_WAIT_US: usize = 12;
const H_WAIT_MAX: usize = 13;
const H_OTHER: usize = 14;
const H_HELD_NOW: usize = 15;
/// Faults on held pages by their page's quiet time at the fault (flips since
/// the last CPU write seen on it): < 64, < 128, ... < 8192, more.
static HOLD_AGES: [AtomicU64; 9] = [const { AtomicU64::new(0) }; 9];

/// Faults on held pages by (instruction, a write) since the last status
/// line.
static HOLD_RIPS: parking_lot::Mutex<Option<std::collections::HashMap<(u64, bool), u64>>> = parking_lot::Mutex::new(None);
/// The waits being timed: (the writer's sequence, the fault's time).
static HOLD_PENDING: parking_lot::Mutex<Vec<(u64, std::time::Instant)>> = parking_lot::Mutex::new(Vec::new());

/// A fault on a held page (`watch`): counted by origin and access, the
/// first 40 logged with their code; the GPU work writing the page then.
fn note_hold_fault(page: u64, write: bool, rip: u64) {
    let (b, e) = crate::exception::host_image();
    let origin = if (crate::mem::GUEST_BASE..crate::mem::GUEST_END).contains(&rip) {
        0
    } else if (b..e).contains(&rip) {
        1
    } else {
        2
    };
    let n = HOLD[H_FAULTS + 2 * origin + write as usize].fetch_add(1, Ordering::Relaxed);
    HOLD[H_HELD_NOW].fetch_sub(1, Ordering::Relaxed);
    crate::engine::note_hold_fault_record(page, origin, write);
    if origin == 0 {
        crate::engine::note_count(if write { crate::engine::C_HOLD_TW } else { crate::engine::C_HOLD_TR }, page);
    }
    if hold_once() && (MIRROR_BASE..MIRROR_END).contains(&page) {
        let i = ((page - MIRROR_BASE) / PAGE) as usize;
        if no_hold()[i / 32].fetch_or(1 << (i % 32), Ordering::Relaxed) >> (i % 32) & 1 == 0 {
            NO_HOLD_PAGES.fetch_add(1, Ordering::Relaxed);
        }
    }
    *HOLD_RIPS.lock().get_or_insert_with(Default::default).entry((rip, write)).or_insert(0) += 1;
    if let Some(c) = chunk(page, false) {
        let frame = super::FRAMES.load(Ordering::Relaxed);
        let age = frame.saturating_sub((c.wrote[idx(page)].load(Ordering::Acquire) as u64).saturating_sub(1));
        let k = ((64 - age.max(1).leading_zeros()) as usize).saturating_sub(6).min(8);
        HOLD_AGES[k].fetch_add(1, Ordering::Relaxed);
    }
    let writer = super::exec::inflight_writer(page, PAGE);
    match writer {
        Some(seq) => {
            HOLD[H_INFLIGHT].fetch_add(1, Ordering::Relaxed);
            HOLD_PENDING.lock().push((seq, std::time::Instant::now()));
        }
        None => {
            HOLD[H_NONE].fetch_add(1, Ordering::Relaxed);
        }
    }
    static LOGGED: AtomicU64 = AtomicU64::new(0);
    if LOGGED.fetch_add(1, Ordering::Relaxed) < 40 {
        crate::info!("fe2: hold census: a {} of held page {page:#x} by {} ({}; this kind's {} so far){}", if write { "write" } else { "read" }, ["the title", "ours", "other code"][origin], describe_code(rip), n + 1, if writer.is_some() { "; GPU work writing it in flight" } else { "" });
    }
}

/// The hold census's tick (`watch`, every millisecond): the waits whose
/// writer completed timed; every second, map bits of pages no longer held
/// (their holds ended without a fault of their own) cleared and counted.
fn hold_tick(last_reconcile: &mut std::time::Instant) {
    {
        let mut p = HOLD_PENDING.lock();
        if !p.is_empty() {
            let done = super::exec::done_now();
            p.retain(|&(seq, t)| {
                if seq > done {
                    return true;
                }
                let us = t.elapsed().as_micros() as u64;
                HOLD[H_WAITS].fetch_add(1, Ordering::Relaxed);
                HOLD[H_WAIT_US].fetch_add(us, Ordering::Relaxed);
                HOLD[H_WAIT_MAX].fetch_max(us, Ordering::Relaxed);
                false
            });
        }
    }
    if last_reconcile.elapsed().as_millis() < 1000 {
        return;
    }
    *last_reconcile = std::time::Instant::now();
    let Some(m) = HELD_MAP.get() else { return };
    // SAFETY: as `held_mark`.
    let words = unsafe { std::slice::from_raw_parts(m.words as *const AtomicU32, HELD_MAP_WORDS) };
    let mut ended = 0u64;
    for (i, w) in words.iter().enumerate() {
        let mut bits = w.load(Ordering::Acquire);
        while bits != 0 {
            let k = bits.trailing_zeros();
            bits &= bits - 1;
            let page = MIRROR_BASE + (i as u64 * 32 + k as u64) * PAGE;
            if !crate::hosted::held(page) && w.fetch_and(!(1u32 << k), Ordering::AcqRel) & (1u32 << k) != 0 {
                ended += 1;
            }
        }
    }
    if ended > 0 {
        HOLD[H_OTHER].fetch_add(ended, Ordering::Relaxed);
        HOLD[H_HELD_NOW].fetch_sub(ended, Ordering::Relaxed);
    }
}

/// Status line of the hold census (since the last one: per flip).
fn hold_line() -> String {
    if !hold_census() || HELD_MAP.get().is_none() {
        return String::new();
    }
    static LAST: parking_lot::Mutex<(u64, [u64; 17], u64, u64, u64)> = parking_lot::Mutex::new((0, [0; 17], 0, 0, 0));
    let frame = super::FRAMES.load(Ordering::Relaxed);
    let now: [u64; 17] = std::array::from_fn(|k| HOLD[k].load(Ordering::Relaxed));
    let (dw, pg, all) = (WB64[2].load(Ordering::Relaxed), WB64[3].load(Ordering::Relaxed), WB64[1].load(Ordering::Relaxed));
    let mut l = LAST.lock();
    let flips = frame.saturating_sub(l.0).max(1) as f64;
    let d = |k: usize| now[k].saturating_sub(l.1[k]);
    let (ddw, dpg, dall) = (dw.saturating_sub(l.2), pg.saturating_sub(l.3), all.saturating_sub(l.4));
    let waits = d(H_WAITS);
    let line = format!(
        "    hold census (`DS_FE2_HOLD`, over {flips:.0} flips): pages held now {} ({} held in {} runs since, {} runs failed); held pages in the write-backs {:.3} M dwords and {:.0} pages a flip ({:.1}% of the dwords written back); faults on held pages {:.3} a flip: the title {} reads / {} writes, ours {} / {}, other code {} / {}; GPU work writing the page in flight at {} of them (until it completed {:.2} ms on average, {:.1} ms at most so far, over {waits}), none at {}; holds ended otherwise (disarms, mapping changes, word answers) {}; pages uploaded into the mirror while held {}; pages written back within the last second {}, GPU-written candidates not held (not written back lately, or faulted before: `DS_FE2_HOLD_ONCE`, {} pages) {}, not quiet for `DS_FE2_HOLD_QUIET` {}\n",
        now[H_HELD_NOW] as i64,
        d(H_PAGES),
        d(H_RUNS),
        d(H_FAILED),
        ddw as f64 / 1e6 / flips,
        dpg as f64 / flips,
        100.0 * ddw as f64 / dall.max(1) as f64,
        (H_FAULTS..H_FAULTS + 6).map(d).sum::<u64>() as f64 / flips,
        d(H_FAULTS),
        d(H_FAULTS + 1),
        d(H_FAULTS + 2),
        d(H_FAULTS + 3),
        d(H_FAULTS + 4),
        d(H_FAULTS + 5),
        d(H_INFLIGHT),
        d(H_WAIT_US) as f64 / 1000.0 / waits.max(1) as f64,
        now[H_WAIT_MAX] as f64 / 1000.0,
        d(H_NONE),
        d(H_OTHER),
        d(H_UPLOADED),
        WB_RECENT_PAGES.load(Ordering::Relaxed),
        NO_HOLD_PAGES.load(Ordering::Relaxed),
        HOLD_STALE.load(Ordering::Relaxed),
        HOLD_YOUNG.load(Ordering::Relaxed)
    );
    *l = (frame, now, dw, pg, all);
    let mut rips: Vec<((u64, bool), u64)> = HOLD_RIPS.lock().take().map(|m| m.into_iter().collect()).unwrap_or_default();
    rips.sort_by(|a, b| b.1.cmp(&a.1));
    if rips.is_empty() {
        return line;
    }
    let top: Vec<String> = rips.iter().take(8).map(|&((rip, w), n)| format!("{n} {} at {}", if w { "writes" } else { "reads" }, describe_code(rip))).collect();
    let ages: Vec<String> = HOLD_AGES.iter().map(|a| a.swap(0, Ordering::Relaxed).to_string()).collect();
    line + &format!("    hold census: faults by code: {} ({} places); by the page's quiet time (flips; < 64, < 128, ..., < 8192, more): {}\n", top.join(", "), rips.len(), ages.join(" "))
}

/// `DS_WB_BUCKETS`: pages of the span the CPU wrote, and those the command
/// processors read, since the last report (one bit a page).
static WB_CPU_PAGES: std::sync::OnceLock<Box<[AtomicU64]>> = std::sync::OnceLock::new();
static WB_READ_PAGES: std::sync::OnceLock<Box<[AtomicU64]>> = std::sync::OnceLock::new();
/// The flip of the last report.
static WB_FLIP: AtomicU64 = AtomicU64::new(0);
/// The buckets the last report found untouched (`wbb lazy`).
static WB_UNTOUCHED: parking_lot::Mutex<Vec<u32>> = parking_lot::Mutex::new(Vec::new());
/// `wbb lazy`'s buckets (a bit each) and the page runs it armed no-access.
static WB_LAZY_BITS: std::sync::OnceLock<Box<[AtomicU64]>> = std::sync::OnceLock::new();
static WB_LAZY_RUNS: parking_lot::Mutex<Vec<(u64, u64)>> = parking_lot::Mutex::new(Vec::new());
/// Faults the host answered on those pages: [title, ours, other] x [read,
/// write], then pages armed, runs failed.
static WB_LAZY_FAULTS: [AtomicU64; 8] = [const { AtomicU64::new(0) }; 8];

fn wb_lazy_bits() -> &'static [AtomicU64] {
    WB_LAZY_BITS.get_or_init(|| (0..rdna::kernels::WB_BUCKETS as usize / 64).map(|_| AtomicU64::new(0)).collect())
}

/// A fault the host answered (`watch`): counted when its page lies in a lazy
/// bucket (`wbb lazy`), the first 30 logged with their code.
fn note_wb_lazy_fault(page: u64, write: bool, rip: u64) {
    if !wb_buckets() || !(MIRROR_BASE..MIRROR_END).contains(&page) {
        return;
    }
    let k = ((page - MIRROR_BASE) >> (rdna::kernels::WB_BUCKET_SHIFT as u64 + 12)) as usize;
    if wb_lazy_bits()[k / 64].load(Ordering::Relaxed) & (1 << (k % 64)) == 0 {
        return;
    }
    let (b, e) = crate::exception::host_image();
    let origin = if (crate::mem::GUEST_BASE..crate::mem::GUEST_END).contains(&rip) {
        0
    } else if (b..e).contains(&rip) {
        1
    } else {
        2
    };
    let n = WB_LAZY_FAULTS[2 * origin + write as usize].fetch_add(1, Ordering::Relaxed);
    static LOGGED: AtomicU64 = AtomicU64::new(0);
    if LOGGED.fetch_add(1, Ordering::Relaxed) < 30 {
        crate::info!("fe2: wbb lazy: a {} of {page:#x} by {} ({}; this kind's {} so far)", if write { "write" } else { "read" }, ["the title", "ours", "other code"][origin], describe_code(rip), n + 1);
    }
}

/// Control `wbb lazy` (diagnosis, wrong results where the title or a
/// command processor reads such pages): the counting write-back leaves out
/// the stores of the pages of the buckets the last report found untouched
/// (their marks cleared, their dwords counted); `wbb eager`: none again.
pub fn wbb_control(what: &str) -> Result<String, String> {
    if !wb_buckets() {
        return Err("wbb needs DS_WB_BUCKETS=1".into());
    }
    let n = (rdna::kernels::WB_BUCKETS / 32) as usize;
    let mut bits = vec![0u32; n];
    let count = match what {
        "lazy" => {
            let u = WB_UNTOUCHED.lock();
            for &k in u.iter() {
                bits[(k / 32) as usize] |= 1 << (k % 32);
            }
            u.len()
        }
        "eager" => 0,
        _ => return Err("usage: wbb lazy|eager".into()),
    };
    // The bucket bits for the fault count, and their pages armed no-access
    // (those nothing else armed, in runs) or disarmed again.
    let lb = wb_lazy_bits();
    for (i, w) in lb.iter().enumerate() {
        let v = (bits[2 * i] as u64) | (bits[2 * i + 1] as u64) << 32;
        w.store(v, Ordering::Relaxed);
    }
    let mut runs = WB_LAZY_RUNS.lock();
    for &(a, n) in runs.iter() {
        crate::hosted::disarm_if_armed(a, n);
    }
    runs.clear();
    if what == "lazy" && crate::hosted::on() {
        for &k in WB_UNTOUCHED.lock().iter() {
            let lo = MIRROR_BASE + ((k as u64) << (rdna::kernels::WB_BUCKET_SHIFT as u64 + 12));
            let hi = lo + (1u64 << (rdna::kernels::WB_BUCKET_SHIFT as u64 + 12));
            let mut p = lo;
            while p < hi {
                // A run of pages nothing armed, readable and writable now.
                if crate::hosted::armed(p) || !crate::mem::os_readable(p, PAGE) {
                    p += PAGE;
                    continue;
                }
                let mut q = p + PAGE;
                while q < hi && !crate::hosted::armed(q) && crate::mem::os_readable(q, PAGE) {
                    q += PAGE;
                }
                if crate::hosted::arm_noaccess(p, q - p).is_ok() {
                    runs.push((p, q - p));
                    WB_LAZY_FAULTS[6].fetch_add((q - p) / PAGE, Ordering::Relaxed);
                } else {
                    WB_LAZY_FAULTS[7].fetch_add(1, Ordering::Relaxed);
                }
                p = q;
            }
        }
    }
    let bytes: Vec<u8> = bits.iter().flat_map(|x| x.to_le_bytes()).collect();
    let at = rdna::kernels::WB_LAZY_AT as u64 * 4;
    let c = ctx();
    for on_c in [false, true] {
        let Some(marks) = super::resident::marks_for(on_c) else { continue };
        super::resident::one_off_now(|cmd| unsafe {
            c.device.cmd_update_buffer(cmd, marks, at, &bytes);
        })?;
    }
    Ok(format!("{count} buckets lazy"))
}

fn wb_bits(b: &std::sync::OnceLock<Box<[AtomicU64]>>) -> &[AtomicU64] {
    b.get_or_init(|| (0..(MIRROR_END - MIRROR_BASE) / PAGE / 64).map(|_| AtomicU64::new(0)).collect())
}

fn note_wb_pages(b: &std::sync::OnceLock<Box<[AtomicU64]>>, a: u64, n: u64) {
    let bits = wb_bits(b);
    let lo = a.max(MIRROR_BASE);
    let hi = a.saturating_add(n.max(1)).min(MIRROR_END);
    let mut p = (lo - MIRROR_BASE) / PAGE;
    while lo < hi && MIRROR_BASE + p * PAGE < hi {
        bits[(p / 64) as usize].fetch_or(1 << (p % 64), Ordering::Relaxed);
        p += 1;
    }
}

/// `DS_WB_BUCKETS`: reads the command processors make of guest memory
/// (`Exec::sync_for`).
pub fn note_wb_reads(reads: &[(u64, u64)]) {
    if wb_buckets() {
        for &(a, n) in reads {
            note_wb_pages(&WB_READ_PAGES, a, n);
        }
    }
}

/// Status part (`DS_WB_BUCKETS`): the window's write-back volume by bucket
/// class (the counts read back from both written-bit maps' aux regions,
/// which are cleared; a write-back racing the read-back may be lost to it),
/// the largest buckets nothing else touched by mapping.
pub fn wb_buckets_line() -> String {
    if !wb_buckets() {
        return String::new();
    }
    let c = ctx();
    let nb = rdna::kernels::WB_BUCKETS as usize;
    let words = 2 * nb as u64;
    let at = rdna::kernels::MARK_AUX_AT as u64 * 4;
    let mut gpu = vec![0u64; 2 * nb];
    for on_c in [false, true] {
        let Some(marks) = super::resident::marks_for(on_c) else { continue };
        let Ok((hbuf, hmem, hptr)) = c.host_buffer(words * 4, vk::BufferUsageFlags::TRANSFER_DST) else {
            return "    write-back buckets (`DS_WB_BUCKETS`): no host buffer\n".into();
        };
        let r = super::resident::one_off_now(|cmd| unsafe {
            c.device.cmd_copy_buffer(cmd, marks, hbuf, &[vk::BufferCopy { src_offset: at, dst_offset: 0, size: words * 4 }]);
            let mb = vk::MemoryBarrier::default().src_access_mask(vk::AccessFlags::TRANSFER_READ).dst_access_mask(vk::AccessFlags::TRANSFER_WRITE);
            c.device.cmd_pipeline_barrier(cmd, vk::PipelineStageFlags::TRANSFER, vk::PipelineStageFlags::TRANSFER, vk::DependencyFlags::empty(), &[mb], &[], &[]);
            c.device.cmd_fill_buffer(cmd, marks, at, words * 4, 0);
        });
        if r.is_ok() {
            for (i, g) in gpu.iter_mut().enumerate() {
                *g += unsafe { std::ptr::read_volatile((hptr as *const u32).add(i)) } as u64;
            }
        }
        unsafe {
            c.device.destroy_buffer(hbuf, None);
            c.device.free_memory(hmem, None);
        }
        if let Err(e) = r {
            return format!("    write-back buckets (`DS_WB_BUCKETS`): read-back failed: {e}\n");
        }
    }
    let frame = super::FRAMES.load(Ordering::Relaxed);
    let flips = frame.saturating_sub(WB_FLIP.swap(frame, Ordering::Relaxed)).max(1) as f64;
    let (cpu, rd) = (wb_bits(&WB_CPU_PAGES), wb_bits(&WB_READ_PAGES));
    let per = (1usize << rdna::kernels::WB_BUCKET_SHIFT) / 64;
    // Classes: neither, read by a CP, CPU-written, both: [buckets, dwords, pages].
    let mut cls = [[0u64; 3]; 4];
    let mut quiet: Vec<(u64, u32)> = Vec::new();
    // (By the engine allocator owning the bucket's MiB: `engine::owner`;
    // and by the page allocator's record holding the bucket's first page.)
    let mut by_owner = [[0u64; 2]; 256];
    let mut by_rec: std::collections::HashMap<(u64, u64, u8), [u64; 2]> = std::collections::HashMap::new();
    let mut weighted: Vec<(u64, u8, u64)> = Vec::new();
    let (mut cpu_pages_mixed, mut pages_mixed) = (0u64, 0u64);
    for k in 0..nb {
        let (dw, pg) = (gpu[k], gpu[nb + k]);
        if dw == 0 && pg == 0 {
            continue;
        }
        let cw: u32 = (0..per).map(|j| cpu[k * per + j].load(Ordering::Relaxed).count_ones()).sum();
        let rw: u32 = (0..per).map(|j| rd[k * per + j].load(Ordering::Relaxed).count_ones()).sum();
        let ci = (rw > 0) as usize + 2 * (cw > 0) as usize;
        let id = crate::engine::owner(MIRROR_BASE + ((k as u64) << (rdna::kernels::WB_BUCKET_SHIFT + 12))) as usize;
        by_owner[id][0] += dw;
        by_owner[id][1] += pg;
        let key = crate::engine::record_key(MIRROR_BASE + ((k as u64) << (rdna::kernels::WB_BUCKET_SHIFT + 12)));
        let r = by_rec.entry(key).or_default();
        r[0] += dw;
        r[1] += pg;
        weighted.push((MIRROR_BASE + ((k as u64) << (rdna::kernels::WB_BUCKET_SHIFT + 12)), key.2, dw));
        cls[ci][0] += 1;
        cls[ci][1] += dw;
        cls[ci][2] += pg;
        if ci == 0 {
            quiet.push((dw, k as u32));
        } else if cw > 0 {
            cpu_pages_mixed += cw as u64;
            pages_mixed += 1 << rdna::kernels::WB_BUCKET_SHIFT;
        }
    }
    for b in [cpu, rd] {
        for x in b.iter() {
            x.store(0, Ordering::Relaxed);
        }
    }
    let (dw_all, pg_all): (u64, u64) = (cls.iter().map(|x| x[1]).sum(), cls.iter().map(|x| x[2]).sum());
    let names = ["untouched", "read by our CPs", "CPU-written", "CPU-written and read"];
    let parts: Vec<String> = cls
        .iter()
        .zip(names)
        .map(|(x, n)| format!("{n}: {} buckets, {:.0}% of dwords ({:.2} M a flip), {:.0}% of pages ({:.0} a flip)", x[0], 100.0 * x[1] as f64 / dw_all.max(1) as f64, x[1] as f64 / flips / 1e6, 100.0 * x[2] as f64 / pg_all.max(1) as f64, x[2] as f64 / flips))
        .collect();
    // The untouched buckets by mapping, the largest first.
    let mut by_map: std::collections::HashMap<(u64, u64, String), u64> = std::collections::HashMap::new();
    *WB_UNTOUCHED.lock() = quiet.iter().map(|q| q.1).collect();
    for &(dw, k) in &quiet {
        let a = MIRROR_BASE + ((k as u64) << (rdna::kernels::WB_BUCKET_SHIFT + 12));
        *by_map.entry(crate::mem::vma_label(a).unwrap_or((a, 0, "?".into()))).or_insert(0) += dw;
    }
    let mut maps: Vec<_> = by_map.into_iter().collect();
    maps.sort_by(|a, b| b.1.cmp(&a.1));
    let top: Vec<String> = maps.iter().take(8).map(|((s, z, n), dw)| format!("{n} {s:#x}+{z:#x} {:.2} M a flip", *dw as f64 / flips / 1e6)).collect();
    let lf: Vec<u64> = WB_LAZY_FAULTS.iter().map(|c| c.load(Ordering::Relaxed)).collect();
    let anames = crate::engine::names();
    let mut owners: Vec<(usize, [u64; 2])> = by_owner.iter().copied().enumerate().filter(|x| x.1[0] + x.1[1] > 0).collect();
    owners.sort_by(|a, b| b.1[0].cmp(&a.1[0]));
    let owners: Vec<String> = owners.iter().map(|(id, x)| format!("{id} {}: {:.2} M dwords ({:.0}%), {:.0} pages a flip", anames.get(&(*id as u8)).map_or("none", |s| s.as_str()), x[0] as f64 / flips / 1e6, 100.0 * x[0] as f64 / dw_all.max(1) as f64, x[1] as f64 / flips)).collect();
    let owners = if owners.is_empty() { String::new() } else { format!("    write-backs by owning engine allocator: {}\n", owners.join("; ")) };
    let mut recs: Vec<((u64, u64, u8), [u64; 2])> = by_rec.into_iter().collect();
    recs.sort_by(|a, b| b.1[0].cmp(&a.1[0]));
    let recs: Vec<String> = recs
        .iter()
        .take(30)
        .map(|((s, e, id), x)| format!("{s:#x}+{:#x} ({id} {}): {:.3} M dwords ({:.1}%), {:.0} pages a flip", e - s, anames.get(id).map_or("none", |s| s.as_str()), x[0] as f64 / flips / 1e6, 100.0 * x[0] as f64 / dw_all.max(1) as f64, x[1] as f64 / flips))
        .collect();
    let owners = owners + &if recs.is_empty() { String::new() } else { format!("    write-backs by engine allocation (the record holding each MiB's first page): {}\n", recs.join("; ")) };
    let sites = crate::engine::sites_weighted(&weighted, 24, flips);
    let owners = owners + &if sites.is_empty() { String::new() } else { format!("    write-back dwords a flip by call site (each MiB's first page): {sites}\n") };
    owners + &format!(
        "    write-back buckets (`DS_WB_BUCKETS`, 1 MiB, over {flips:.0} flips): {:.2} M dwords and {:.0} pages a flip; {}; the CPU wrote {:.1}% of the pages of the buckets it wrote; untouched by mapping: {}; `wbb lazy` pages armed no-access {} ({} runs failed), faults on them so far: the title {} reads / {} writes, ours {} / {}, other code {} / {}\n",
        dw_all as f64 / flips / 1e6,
        pg_all as f64 / flips,
        parts.join("; "),
        100.0 * cpu_pages_mixed as f64 / pages_mixed.max(1) as f64,
        top.join(", "),
        lf[6],
        lf[7],
        lf[0],
        lf[1],
        lf[2],
        lf[3],
        lf[4],
        lf[5]
    )
}

/// Records the write-back into `cmd` (outside a render pass): a barrier
/// after all earlier work, the scatter over the touched-page list, the list
/// emptied (by the kernel itself), a barrier before all later work. `on_c`:
/// a batch on the compute family's queue, with its written-bit map (ordered
/// async compute). Returns the pipeline it bound (the caller's bound
/// compute state is gone).
pub fn record_scatter(cmd: vk::CommandBuffer, on_c: bool) -> Option<vk::Pipeline> {
    record_scatter_at(cmd, on_c, 0)
}

/// `record_scatter` at site `site` (`tune wb_probe_site`: 1 a batch's
/// close, 2 before a CP write the GPU performs, 3 a workers' chunk batch's
/// close; 0 elsewhere).
pub fn record_scatter_at(cmd: vk::CommandBuffer, on_c: bool, site: u8) -> Option<vk::Pipeline> {
    let s = SCATTER.get()?;
    let marks = super::resident::marks_for(on_c)?;
    let set = if on_c { s.set_c } else { s.set };
    let (mbuf, _) = super::resident::mirror_buffer()?;
    let c = ctx();
    let m = c.buffer_address(mbuf);
    let t = super::pagetable::address();
    let mut h = HELD_MAP.get().map_or(0, |m| m.addr);
    // (The other family's hash table, whose entries of the pages this
    // write-back stores the kernel clears; 0: none.)
    let mut other = 0u64;
    // (`DS_WB_SAME=2`: the family's hash table; cleared first when asked.)
    if let Some(ht) = HASHES.get() {
        let k = usize::from(on_c).min(ht.tables.len() - 1);
        h = ht.tables[k].1;
        if ht.tables.len() > 1 {
            other = ht.tables[1 - k].1;
        }
        let bit = 1u32 << k;
        let all = HASH_CLEAR.load(Ordering::Acquire) & bit != 0 && HASH_CLEAR.fetch_and(!bit, Ordering::AcqRel) & bit != 0;
        let ranges = std::mem::take(&mut HASH_CLEAR_RANGES.lock()[k]);
        if all || !ranges.is_empty() {
            unsafe {
                if all {
                    ctx().device.cmd_fill_buffer(cmd, ht.tables[k].0, 0, vk::WHOLE_SIZE, 0);
                } else {
                    for (lo, hi) in ranges {
                        ctx().device.cmd_fill_buffer(cmd, ht.tables[k].0, (lo - MIRROR_BASE) / PAGE * 8, (hi - lo) / PAGE * 8, 0);
                    }
                }
                let mb = vk::MemoryBarrier::default().src_access_mask(vk::AccessFlags::TRANSFER_WRITE).dst_access_mask(vk::AccessFlags::SHADER_READ | vk::AccessFlags::SHADER_WRITE);
                ctx().device.cmd_pipeline_barrier(cmd, vk::PipelineStageFlags::TRANSFER, vk::PipelineStageFlags::COMPUTE_SHADER, vk::DependencyFlags::empty(), &[mb], &[], &[]);
            }
        }
    }
    // (The zero-known bitmap at 8-9: the kernel clears the bits of the pages it lists, with
    // `tune zk` off too, so that the bits stay true when it is switched on again.)
    let zk = super::zk::address();
    let pc: Vec<u8> = [m as u32, (m >> 32) as u32, t as u32, (t >> 32) as u32, h as u32, (h >> 32) as u32, other as u32, (other >> 32) as u32, zk as u32, (zk >> 32) as u32].iter().flat_map(|x| x.to_le_bytes()).collect();
    let pipeline = s.pipeline;
    unsafe {
        let all = vk::MemoryBarrier::default().src_access_mask(vk::AccessFlags::MEMORY_WRITE).dst_access_mask(vk::AccessFlags::MEMORY_READ | vk::AccessFlags::MEMORY_WRITE);
        c.device.cmd_pipeline_barrier(cmd, vk::PipelineStageFlags::ALL_COMMANDS, vk::PipelineStageFlags::COMPUTE_SHADER | vk::PipelineStageFlags::DRAW_INDIRECT, vk::DependencyFlags::empty(), &[all], &[], &[]);
        // (`DS_WB_SAME=2`: the skipping write-back while `wbs on`.)
        let pipeline = match SAME_SKIP.get() {
            Some(&p) if SAME_SKIP_ON.load(Ordering::Acquire) => p,
            _ => pipeline,
        };
        c.device.cmd_bind_pipeline(cmd, vk::PipelineBindPoint::COMPUTE, pipeline);
        c.device.cmd_bind_descriptor_sets(cmd, vk::PipelineBindPoint::COMPUTE, s.layout, 0, &[set], &[]);
        c.device.cmd_push_constants(cmd, s.layout, vk::ShaderStageFlags::COMPUTE, 0, &pc);
        c.device.cmd_dispatch_indirect(cmd, marks, rdna::kernels::MARK_LIST_AT as u64 * 4);
        if SCATTERS_RECORDED.fetch_add(1, Ordering::Relaxed) % COUNTER_COPY_EVERY == COUNTER_COPY_EVERY - 1 {
            record_counter_copy(s, cmd);
        }
        // (The kernel emptied the list; the barrier's first scope holds the
        // dispatch's read of its arguments, a logically earlier stage.)
        let after = vk::MemoryBarrier::default().src_access_mask(vk::AccessFlags::SHADER_WRITE).dst_access_mask(vk::AccessFlags::MEMORY_READ | vk::AccessFlags::MEMORY_WRITE | vk::AccessFlags::HOST_READ);
        c.device.cmd_pipeline_barrier(cmd, vk::PipelineStageFlags::COMPUTE_SHADER, vk::PipelineStageFlags::ALL_COMMANDS | vk::PipelineStageFlags::HOST, vk::DependencyFlags::empty(), &[after], &[], &[]);
        // (`tune wb_probe`, diagnosis: one workgroup of a ballast kernel
        // running that many steps of a dependent chain into a buffer of its
        // own, between full barriers: only GPU time and order added.)
        let probe = super::tune::wb_probe();
        let want = match super::tune::wb_probe_site() {
            0 => true,
            1 => site == 1 || site == 3,
            k => site as u64 == k,
        };
        if probe != 0 && want {
            if let Some(b) = ballast() {
                c.device.cmd_pipeline_barrier(cmd, vk::PipelineStageFlags::ALL_COMMANDS, vk::PipelineStageFlags::COMPUTE_SHADER, vk::DependencyFlags::empty(), &[all], &[], &[]);
                c.device.cmd_bind_pipeline(cmd, vk::PipelineBindPoint::COMPUTE, b.pipeline);
                c.device.cmd_bind_descriptor_sets(cmd, vk::PipelineBindPoint::COMPUTE, b.layout, 0, &[b.set], &[]);
                let pcs: Vec<u8> = [probe.min(u32::MAX as u64) as u32, 0].iter().flat_map(|x| x.to_le_bytes()).collect();
                c.device.cmd_push_constants(cmd, b.layout, vk::ShaderStageFlags::COMPUTE, 0, &pcs);
                c.device.cmd_dispatch(cmd, 1, 1, 1);
                c.device.cmd_pipeline_barrier(cmd, vk::PipelineStageFlags::COMPUTE_SHADER, vk::PipelineStageFlags::ALL_COMMANDS | vk::PipelineStageFlags::HOST, vk::DependencyFlags::empty(), &[after], &[], &[]);
                PROBES.fetch_add(1, Ordering::Relaxed);
                note_scatter();
                return Some(b.pipeline);
            }
        }
    }
    note_scatter();
    Some(pipeline)
}

/// Write-backs followed by `tune wb_probe`'s ballast.
static PROBES: AtomicU64 = AtomicU64::new(0);

/// `tune wb_probe`'s ballast: `rdna::kernels::latency_test` storing into a
/// buffer of its own (one workgroup's dwords), made at first use.
struct Ballast {
    pipeline: vk::Pipeline,
    layout: vk::PipelineLayout,
    set: vk::DescriptorSet,
}

// SAFETY: Vulkan handles, used from recording threads under the API's
// external synchronization rules (the set is never updated after it is made).
unsafe impl Send for Ballast {}
unsafe impl Sync for Ballast {}

fn ballast() -> Option<&'static Ballast> {
    static B: std::sync::OnceLock<Option<Ballast>> = std::sync::OnceLock::new();
    B.get_or_init(|| {
        let r = (|| -> Result<Ballast, String> {
            let c = ctx();
            let d = &c.device;
            let (buf, mem) = c.device_buffer(4 * rdna::kernels::LATENCY_LOCAL_SIZE as u64, vk::BufferUsageFlags::STORAGE_BUFFER)?;
            std::mem::forget(mem);
            unsafe {
                let binding = [vk::DescriptorSetLayoutBinding::default().binding(0).descriptor_type(vk::DescriptorType::STORAGE_BUFFER).descriptor_count(1).stage_flags(vk::ShaderStageFlags::COMPUTE)];
                let set_layout = d.create_descriptor_set_layout(&vk::DescriptorSetLayoutCreateInfo::default().bindings(&binding), None).map_err(|e| format!("set layout: {e}"))?;
                let range = [vk::PushConstantRange::default().stage_flags(vk::ShaderStageFlags::COMPUTE).offset(0).size(8)];
                let layout = d.create_pipeline_layout(&vk::PipelineLayoutCreateInfo::default().set_layouts(&[set_layout]).push_constant_ranges(&range), None).map_err(|e| format!("layout: {e}"))?;
                let code = rdna::kernels::latency_test(0xba11);
                let module = d.create_shader_module(&vk::ShaderModuleCreateInfo::default().code(&code), None).map_err(|e| format!("module: {e}"))?;
                let stage = vk::PipelineShaderStageCreateInfo::default().stage(vk::ShaderStageFlags::COMPUTE).module(module).name(c"main");
                let pipeline = d.create_compute_pipelines(vk::PipelineCache::null(), &[vk::ComputePipelineCreateInfo::default().stage(stage).layout(layout)], None).map_err(|(_, e)| format!("pipeline: {e}"))?[0];
                d.destroy_shader_module(module, None);
                let sizes = [vk::DescriptorPoolSize { ty: vk::DescriptorType::STORAGE_BUFFER, descriptor_count: 1 }];
                let pool = d.create_descriptor_pool(&vk::DescriptorPoolCreateInfo::default().max_sets(1).pool_sizes(&sizes), None).map_err(|e| format!("pool: {e}"))?;
                let set = d.allocate_descriptor_sets(&vk::DescriptorSetAllocateInfo::default().descriptor_pool(pool).set_layouts(&[set_layout])).map_err(|e| format!("set: {e}"))?[0];
                let bi = [vk::DescriptorBufferInfo::default().buffer(buf).offset(0).range(vk::WHOLE_SIZE)];
                d.update_descriptor_sets(&[vk::WriteDescriptorSet::default().dst_set(set).dst_binding(0).descriptor_type(vk::DescriptorType::STORAGE_BUFFER).buffer_info(&bi)], &[]);
                Ok(Ballast { pipeline, layout, set })
            }
        })();
        match r {
            Ok(b) => {
                crate::info!("fe2: `tune wb_probe`'s ballast kernel made");
                Some(b)
            }
            Err(e) => {
                crate::warn!("fe2: `tune wb_probe`'s ballast kernel: {e}");
                None
            }
        }
    })
    .as_ref()
}

/// The probe's line ("" before any).
pub fn wb_probe_line() -> String {
    let n = PROBES.load(Ordering::Relaxed);
    if n == 0 {
        return String::new();
    }
    format!("    front end 2's write-backs followed by `tune wb_probe`'s ballast: {n}\n")
}

/// Records a copy of the write-back's video-memory counters into the
/// host-visible ones (after the write-backs before it; the barrier after
/// the write-back that follows makes it visible to the host).
unsafe fn record_counter_copy(s: &Scatter, cmd: vk::CommandBuffer) {
    let d = s.dev;
    let c = ctx();
    let to_copy = vk::MemoryBarrier::default().src_access_mask(vk::AccessFlags::SHADER_WRITE).dst_access_mask(vk::AccessFlags::TRANSFER_READ);
    let bytes = 4 * rdna::kernels::SCATTER_COUNTERS as u64;
    let after = vk::MemoryBarrier::default().src_access_mask(vk::AccessFlags::TRANSFER_WRITE).dst_access_mask(vk::AccessFlags::HOST_READ);
    unsafe {
        c.device.cmd_pipeline_barrier(cmd, vk::PipelineStageFlags::COMPUTE_SHADER, vk::PipelineStageFlags::TRANSFER, vk::DependencyFlags::empty(), &[to_copy], &[], &[]);
        c.device.cmd_copy_buffer(cmd, d, s.host, &[vk::BufferCopy { src_offset: 0, dst_offset: 0, size: bytes }]);
        c.device.cmd_pipeline_barrier(cmd, vk::PipelineStageFlags::TRANSFER, vk::PipelineStageFlags::HOST, vk::DependencyFlags::empty(), &[after], &[], &[]);
    }
}

/// The buffer the write-back kernel counts into (its pages and dwords
/// first: `rdna::kernels::SCATTER_COUNTERS`), in video memory.
pub fn counter_buffer() -> Option<vk::Buffer> {
    SCATTER.get().map(|s| s.dev)
}

/// Whether front end 2 is on with its write-back kernel (which empties the
/// touched-page list itself).
pub fn scatter_resets() -> bool {
    on() && SCATTER.get().is_some()
}

/// The write-back's counters summed in 64 bits (the kernel's are 32-bit
/// and wrap within seconds at an arrival's volume: ~10 million dwords a
/// flip), sampled at every flip (`census_sample`): pages, dwords and four
/// counters the lazy write-back study used (FINDINGS.md 2026-10-31 night),
/// then the last values seen of the six.
static WB64: [AtomicU64; 12] = [const { AtomicU64::new(0) }; 12];

/// Adds the counters' growth since the last sample (at each flip, from one
/// thread; the status line reads the sums alone, two samplers could race
/// on the last values).
pub fn census_sample() {
    if !on() || SCATTER.get().is_none() {
        return;
    }
    let k = counters();
    for (i, now) in [k[0], k[1], k[4], k[5], k[6], k[7]].into_iter().enumerate() {
        let prev = WB64[6 + i].swap(now as u64, Ordering::Relaxed) as u32;
        WB64[i].fetch_add(now.wrapping_sub(prev) as u64, Ordering::Relaxed);
    }
    // (Counter 7: the last page the write-back found no table entry for, its number + 1.)
    static LAST_LOST: AtomicU32 = AtomicU32::new(0);
    static LOST_LOGGED: AtomicU32 = AtomicU32::new(0);
    if k[7] != 0 && LAST_LOST.swap(k[7], Ordering::Relaxed) != k[7] && LOST_LOGGED.fetch_add(1, Ordering::Relaxed) < 40 {
        let a = rdna::translate::PT_BASE + (((k[7] - 1) as u64) << rdna::translate::PT_PAGE_SHIFT);
        crate::info!(
            "fe2: write-back lost page {a:#x} (engine owner {}; its import window last used {}; a listed import over it now: {}; {} pages lost so far)",
            crate::engine::owner(a),
            super::exec::window_age(a).map_or("never".to_string(), |n| format!("{n} flips ago")),
            super::exec::listed_over(a).map_or("not known (the list busy)".to_string(), |b| b.to_string()),
            k[2]
        );
    }
}

/// The write-back's counters (pages, dwords, pages without a mapping,
/// overflowed lists, the census's dwords and unchanged ones), as the GPU left them so far (as of the last
/// copy).
fn counters() -> [u32; rdna::kernels::SCATTER_COUNTERS as usize] {
    let Some(s) = SCATTER.get() else { return [0; rdna::kernels::SCATTER_COUNTERS as usize] };
    // SAFETY: host-visible words the kernel adds to atomically.
    std::array::from_fn(|k| unsafe { std::ptr::read_volatile(s.counters.add(k)) })
}

/// Makes the mirror, the written-bit map, the guest page table and the
/// write-back, switches off the old placement mechanisms, and runs the
/// self-test (after Vulkan initialization, when `wanted`). On failure front
/// end 2 stays off.
pub fn init() {
    if !wanted() || on() {
        return;
    }
    // The old placement and coherence mechanisms are off under it, and so
    // are the recording schemes that exist to hide their costs.
    // (Parallel runs stay: their chunks run in program order, each with
    // its own uploads and write-backs; so does recording past label waits
    // and runs: held batches are deferred ones, their uploads valid once
    // they are queued, and a dropped stretch's never happen. So do the
    // per-view copies as buffer updates, forwarded to the CPU-side reads of
    // their blocks: the updates write the mirror too.)
    // (The host census and mirror mode would read the host's faults too.)
    // (GPU-resolved chains stay as set: stage F3, they read the mirror at
    // guest offsets, `Spec::pt_linear`, their per-item ranges placed per
    // command, `exec::fe2_gpu_ranges`.)
    // (So do long runs, off by default: a chunk recorded before an earlier
    // one still uploads what it finds invalid, its uploads read guest memory
    // when they execute, after the earlier chunks' write-backs, and its pages
    // are valid only once it is queued, in program order; its CPU reads are
    // checked at its turn against the run's earlier writes as before.)
    const OFF: [&str; 4] = ["homes", "shadow_args", "pre_share", "host_census"];
    for name in OFF {
        if !super::tune::set(name, 0) {
            crate::warn!("fe2: no switch {name}");
        }
    }
    // Compute queues' synchronization-free runs are recorded in parallel
    // (four workers, flushes kept inside runs, runs from 64 commands: +6-10%
    // at the vista once checks follow each other's takes; a loss in the
    // default front end, FINDINGS.md 2026-10-28 night; `cp::compute_workers`,
    // without switches since 2026-11-01 but the run length), and waits passed on
    // another queue's promise adopt its writer's view instead of ending an
    // epoch (+3.6% at the vista with stretches checked at their labels,
    // FINDINGS.md 2026-10-29), and waits that release nothing new (gone
    // past, a counter's value passed on again) end none (epoch ends -38%,
    // takes -10%, level), and the write-back's kernel empties its
    // touched-page list itself (level to +1%), counts into video memory
    // (GPU busy -2.3 ms a frame, +2-3%) and counts as a full barrier, CP
    // writes right after it recording none before them (level to +0.3%;
    // FINDINGS.md 2026-10-29 morning): those five without switches since
    // 2026-11-01; the parallel runs' length unless set explicitly.
    const PAR: [(&str, u64); 1] = [("par_min_compute", 64)];
    for (name, v) in PAR {
        if !super::tune::set_default(name, v) {
            crate::warn!("fe2: no switch {name}");
        }
    }
    let made = match super::resident::make_for_fe2() {
        Ok(s) => s,
        Err(e) => {
            crate::error!("fe2: no mirror: {e}; front end 2 off");
            return;
        }
    };
    if super::pagetable::address() == 0 {
        crate::error!("fe2: no guest page table (device addresses); front end 2 off");
        return;
    }
    let Some(marks) = super::resident::marks() else {
        crate::error!("fe2: no written-bit map; front end 2 off");
        return;
    };
    match make_scatter(marks, super::resident::marks_for(true)) {
        Ok(s) => {
            let _ = SCATTER.set(s);
            super::zk::init();
        }
        Err(e) => {
            crate::error!("fe2: write-back kernel: {e}; front end 2 off");
            return;
        }
    }
    match self_test() {
        Ok(s) => {
            if std::env::var("DS_FE2_SCATTER_BENCH").ok().is_some_and(|v| !v.is_empty() && v != "0") {
                match scatter_bench() {
                    Ok(l) => crate::info!("fe2: {l}"),
                    Err(e) => crate::warn!("fe2: write-back bench: {e}"),
                }
            }
            ON.store(true, Ordering::Release);
            if crate::hosted::on() {
                // (`DS_FE2_WORD=0`, diagnosis: the host answers a fault for
                // its page only.)
                if std::env::var("DS_FE2_WORD").ok().as_deref() != Some("0") {
                    crate::hosted::set_word_mode();
                }
                if hold_census() {
                    crate::hosted::set_held_answered();
                    if HELD_MAP.get().is_some() {
                        let _ = std::thread::Builder::new().name("fe2-hold".into()).spawn(hold_thread);
                    }
                } else if c4_on() && HELD_MAP.get().is_some() {
                    // (Front end 4: the release service first: a page is
                    // held only while it runs.)
                    crate::hosted::set_held_served(true);
                    let _ = std::thread::Builder::new().name("fe2-c4-release".into()).spawn(c4_release_thread);
                    let _ = std::thread::Builder::new().name("fe2-hold".into()).spawn(hold_thread);
                    crate::info!("fe2: front end 4's held pages on (`DS_C4=0` turns them off): GPU-written pages of the GPU page allocator's and heaps' records held in the mirror, released on a CPU access");
                }
                let _ = std::thread::Builder::new().name("fe2-faults".into()).spawn(watch);
                if arm_behind_wanted() {
                    let _ = std::thread::Builder::new().name("fe2-armer".into()).spawn(behind_thread);
                }
            }
            crate::info!(
                "fe2: front end 2 on ({made}; {s}; hosted arming {}{}; switched off: {}; compute runs recorded in parallel on 4 workers, from {} commands)",
                if crate::hosted::on() { "on" } else { "off (not under the host)" },
                if arm_gpu() { "" } else if arm_gpu_os() { ", GPU-written pages protected but kept unarmed (DS_FE2_ARM_GPU=2)" } else { ", GPU-written pages never armed" },
                OFF.join(", "),
                super::tune::par_min_compute()
            );
        }
        Err(e) => crate::error!("fe2: the write-back's self-test failed: {e}; front end 2 off"),
    }
    if std::env::var("DS_PROT_TEST").ok().as_deref() == Some("1") {
        crate::info!("{}", protection_test());
    }
}

/// `DS_PROT_TEST=1` (diagnosis, at front end 2's start): whether GPU work
/// reaches an imported host page whose CPU protection is no-access (and
/// read-only): a private page with a pattern, imported, protected, then on
/// the GPU a copy out of it and a fill of its first half, waited for; the
/// page made read-write again and checked (FRONTEND4.md C4: a held page is
/// no-access while GPU work may still reach it).
fn protection_test() -> String {
    use crate::platform::os::Win32::System::Memory::{VirtualAlloc, VirtualFree, VirtualProtect, MEM_COMMIT, MEM_RELEASE, MEM_RESERVE, PAGE_NOACCESS, PAGE_READONLY, PAGE_READWRITE};
    let c = ctx();
    let mut out = Vec::new();
    // (Also: a page made no-access before its import.)
    for (name, prot, before) in [("no-access", PAGE_NOACCESS, false), ("read-only", PAGE_READONLY, false), ("read-write", PAGE_READWRITE, false), ("no-access before the import", PAGE_NOACCESS, true)] {
        let n = 64 << 10;
        let h = unsafe { VirtualAlloc(std::ptr::null(), n, MEM_COMMIT | MEM_RESERVE, PAGE_READWRITE) } as u64;
        if h == 0 {
            out.push(format!("{name}: VirtualAlloc failed"));
            continue;
        }
        let r = (|| -> Result<String, String> {
            let words = h as *mut u32;
            for i in 0..1024u32 {
                unsafe { words.add(i as usize).write_volatile(0x1111_0000 + i) };
            }
            let usage = vk::BufferUsageFlags::STORAGE_BUFFER | vk::BufferUsageFlags::TRANSFER_SRC | vk::BufferUsageFlags::TRANSFER_DST;
            let mut old = 0u32;
            if before && unsafe { VirtualProtect(h as _, PAGE as usize, prot, &mut old) } == 0 {
                return Err("VirtualProtect before the import".into());
            }
            let imported = c.import_host(h, PAGE, usage);
            if before {
                unsafe { VirtualProtect(h as _, PAGE as usize, PAGE_READWRITE, &mut old) };
            }
            let (hb, hm, _) = imported.map_err(|e| format!("the import failed: {e}"))?;
            if before {
                unsafe { VirtualProtect(h as _, PAGE as usize, prot, &mut old) };
            }
            let (rb, rbm, rbp) = c.host_buffer(4096, vk::BufferUsageFlags::TRANSFER_DST)?;
            if !before && unsafe { VirtualProtect(h as _, PAGE as usize, prot, &mut old) } == 0 {
                unsafe {
                    c.device.destroy_buffer(rb, None);
                    c.device.free_memory(rbm, None);
                    c.device.destroy_buffer(hb, None);
                    c.device.free_memory(hm, None);
                }
                return Err("VirtualProtect".into());
            }
            let t = std::time::Instant::now();
            let r = super::resident::one_off_now(|cmd| unsafe {
                c.device.cmd_copy_buffer(cmd, hb, rb, &[vk::BufferCopy { src_offset: 0, dst_offset: 0, size: 4096 }]);
                let mb = vk::MemoryBarrier::default().src_access_mask(vk::AccessFlags::TRANSFER_READ).dst_access_mask(vk::AccessFlags::TRANSFER_WRITE);
                c.device.cmd_pipeline_barrier(cmd, vk::PipelineStageFlags::TRANSFER, vk::PipelineStageFlags::TRANSFER, vk::DependencyFlags::empty(), &[mb], &[], &[]);
                c.device.cmd_fill_buffer(cmd, hb, 0, 2048, 0xDEAD_BEEF);
            });
            let ms = t.elapsed().as_secs_f64() * 1e3;
            unsafe { VirtualProtect(h as _, PAGE as usize, PAGE_READWRITE, &mut old) };
            let read_ok = (0..1024u32).all(|i| unsafe { std::ptr::read_volatile((rbp as *const u32).add(i as usize)) } == 0x1111_0000 + i);
            let read_zero = (0..1024u32).all(|i| unsafe { std::ptr::read_volatile((rbp as *const u32).add(i as usize)) } == 0);
            let filled = (0..512u32).filter(|&i| unsafe { words.add(i as usize).read_volatile() } == 0xDEAD_BEEF).count();
            let kept = (512..1024u32).filter(|&i| unsafe { words.add(i as usize).read_volatile() } == 0x1111_0000 + i).count();
            unsafe {
                c.device.destroy_buffer(rb, None);
                c.device.free_memory(rbm, None);
                c.device.destroy_buffer(hb, None);
                c.device.free_memory(hm, None);
            }
            r?;
            Ok(format!("{name}: the GPU's read {}, its fill landed in {filled} of 512 dwords ({kept} of 512 kept), {ms:.2} ms", if read_ok { "saw the pattern" } else if read_zero { "saw zeros" } else { "saw other words" }))
        })();
        unsafe { VirtualFree(h as _, 0, MEM_RELEASE) };
        out.push(r.unwrap_or_else(|e| format!("{name}: failed: {e}")));
    }
    format!("fe2: protection test (`DS_PROT_TEST`): {}", out.join("; "))
}

/// The write-back over one page: guest memory a private page imported like
/// a window, mapped into the guest page table at the span's last page; the
/// mirror's page a pattern; five dwords marked. The marked ones must land,
/// the others keep what the CPU wrote, and the marks and the touched bit
/// must be cleared.
fn self_test() -> Result<String, String> {
    use crate::platform::os::Win32::System::Memory::{VirtualAlloc, VirtualFree, MEM_COMMIT, MEM_RELEASE, MEM_RESERVE, PAGE_READWRITE};
    let c = ctx();
    let marks = super::resident::marks().ok_or("no written-bit map")?;
    let table = super::pagetable::buffer().ok_or("no page table")?;
    let guest = MIRROR_END - PAGE;
    let page = (guest - MIRROR_BASE) / PAGE;
    let pattern: Vec<u8> = (0..1024u32).flat_map(|i| (0x5000_0000 + i).to_le_bytes()).collect();
    super::resident::upload_now(guest, &pattern)?;
    let n = 64 << 10;
    let h = unsafe { VirtualAlloc(std::ptr::null(), n, MEM_COMMIT | MEM_RESERVE, PAGE_READWRITE) } as u64;
    if h == 0 {
        return Err("VirtualAlloc".into());
    }
    let r = (|| {
        let words = h as *mut u32;
        for i in 0..1024 {
            unsafe { words.add(i).write_volatile(0xAAAA_AAAA) };
        }
        let usage = vk::BufferUsageFlags::STORAGE_BUFFER | vk::BufferUsageFlags::TRANSFER_SRC | vk::BufferUsageFlags::TRANSFER_DST;
        let (hb, hm, hva) = c.import_host(h, PAGE, usage)?;
        let dwords = [0u32, 5, 31, 32, 1023];
        let mut mask = [0u32; 32];
        for d in dwords {
            mask[(d / 32) as usize] |= 1 << (d % 32);
        }
        let mask_bytes: Vec<u8> = mask.iter().flat_map(|x| x.to_le_bytes()).collect();
        let list: Vec<u8> = [1u32, 1, 1, 0, page as u32].iter().flat_map(|x| x.to_le_bytes()).collect();
        let pbit_at = 4 * (rdna::kernels::MARK_PAGE_BITS_AT as u64 + page / 32);
        let pbit = (1u32 << (page % 32)).to_le_bytes();
        let before = counters();
        let (rb, rbm, rbp) = c.host_buffer(256, vk::BufferUsageFlags::TRANSFER_DST)?;
        let r = super::resident::one_off_now(|cmd| unsafe {
            c.device.cmd_update_buffer(cmd, table, page * 8, &hva.to_le_bytes());
            c.device.cmd_update_buffer(cmd, marks, page * 128, &mask_bytes);
            c.device.cmd_update_buffer(cmd, marks, pbit_at, &pbit);
            c.device.cmd_update_buffer(cmd, marks, rdna::kernels::MARK_LIST_AT as u64 * 4, &list);
            let mb = vk::MemoryBarrier::default().src_access_mask(vk::AccessFlags::TRANSFER_WRITE).dst_access_mask(vk::AccessFlags::MEMORY_READ | vk::AccessFlags::MEMORY_WRITE);
            c.device.cmd_pipeline_barrier(cmd, vk::PipelineStageFlags::TRANSFER, vk::PipelineStageFlags::ALL_COMMANDS, vk::DependencyFlags::empty(), &[mb], &[], &[]);
            record_scatter(cmd, false);
            if let Some(s) = SCATTER.get() {
                record_counter_copy(s, cmd);
            }
            // What the kernel left in the map, then the table entry cleared.
            c.device.cmd_copy_buffer(cmd, marks, rb, &[vk::BufferCopy { src_offset: page * 128, dst_offset: 0, size: 128 }, vk::BufferCopy { src_offset: pbit_at, dst_offset: 128, size: 4 }, vk::BufferCopy { src_offset: rdna::kernels::MARK_LIST_AT as u64 * 4, dst_offset: 132, size: 4 }]);
            c.device.cmd_update_buffer(cmd, table, page * 8, &0u64.to_le_bytes());
        });
        let after = counters();
        let left: Vec<u32> = (0..35).map(|k| unsafe { std::ptr::read_volatile((rbp as *const u32).add(k)) }).collect();
        unsafe {
            c.device.destroy_buffer(rb, None);
            c.device.free_memory(rbm, None);
            c.device.destroy_buffer(hb, None);
            c.device.free_memory(hm, None);
        }
        r?;
        for i in 0..1024u32 {
            let got = unsafe { words.add(i as usize).read_volatile() };
            let want = if dwords.contains(&i) { 0x5000_0000 + i } else { 0xAAAA_AAAA };
            if got != want {
                return Err(format!("dword {i}: {got:#x}, not {want:#x}"));
            }
        }
        if left[..32].iter().any(|&w| w != 0) {
            return Err("mask words left set".into());
        }
        if left[32] & (1 << (page % 32)) != 0 {
            return Err("the page's touched bit left set".into());
        }
        if left[33] != 0 {
            return Err(format!("the list's count left at {}", left[33]));
        }
        if after[0] != before[0] + 1 || after[1] != before[1] + 5 || after[2] != before[2] {
            return Err(format!("counters {before:?} -> {after:?}"));
        }
        Ok(format!("write-back self-test passed: 5 marked dwords of an imported page written back, 1019 left"))
    })();
    unsafe { VirtualFree(h as _, 0, MEM_RELEASE) };
    r
}

/// `DS_FE2_SCATTER_BENCH=1` (diagnosis, at front end 2's start): the
/// write-back's GPU time on synthetic inputs as the segment profile counts
/// it (from after the barrier behind its inputs to after its last barrier;
/// the median of BENCH_REPS): an empty list, then 64 private pages (`=<n>`:
/// n, 64-16384 in words of 32) imported and mapped at the span's end (as
/// the self-test's page) listed without marks and with 1, 32 consecutive,
/// 32 spread (one in 32) and all 1024 dwords of each marked; then, for the
/// rate of stores into host memory by other means, the same pages copied
/// from the mirror into the import and the import filled (each between the
/// same barriers). Its pages and marks are cleared again after.
fn scatter_bench() -> Result<String, String> {
    use crate::platform::os::Win32::System::Memory::{VirtualAlloc, VirtualFree, MEM_COMMIT, MEM_RELEASE, MEM_RESERVE, PAGE_READWRITE};
    let want: u64 = std::env::var("DS_FE2_SCATTER_BENCH").ok().and_then(|v| v.parse().ok()).unwrap_or(1);
    let bench_pages: u64 = if want <= 1 { 64 } else { want.clamp(64, 16384).next_multiple_of(32) };
    const BENCH_REPS: usize = 15;
    // (`cmd_update_buffer` takes at most 64 KiB.)
    let update = |cmd: vk::CommandBuffer, buf: vk::Buffer, at: u64, data: &[u8]| {
        for (k, part) in data.chunks(65536).enumerate() {
            unsafe { ctx().device.cmd_update_buffer(cmd, buf, at + 65536 * k as u64, part) };
        }
    };
    let c = ctx();
    let marks = super::resident::marks().ok_or("no written-bit map")?;
    let table = super::pagetable::buffer().ok_or("no page table")?;
    let guest = MIRROR_END - bench_pages * PAGE;
    let p0 = (guest - MIRROR_BASE) / PAGE;
    if p0 % 32 != 0 {
        return Err("its pages do not start a word of touched bits".into());
    }
    super::resident::back(guest, bench_pages * PAGE)?;
    let mirror = super::resident::mirror_buffer().ok_or("no mirror")?.0;
    let h = unsafe { VirtualAlloc(std::ptr::null(), (bench_pages * PAGE) as usize, MEM_COMMIT | MEM_RESERVE, PAGE_READWRITE) } as u64;
    if h == 0 {
        return Err("VirtualAlloc".into());
    }
    let r = (|| {
        let usage = vk::BufferUsageFlags::STORAGE_BUFFER | vk::BufferUsageFlags::TRANSFER_SRC | vk::BufferUsageFlags::TRANSFER_DST;
        let (hb, hm, hva) = c.import_host(h, bench_pages * PAGE, usage)?;
        let entries: Vec<u8> = (0..bench_pages).flat_map(|i| (hva + i * PAGE).to_le_bytes()).collect();
        let n = 2 * BENCH_REPS as u32;
        let qp = match unsafe { c.device.create_query_pool(&vk::QueryPoolCreateInfo::default().query_type(vk::QueryType::TIMESTAMP).query_count(n), None) } {
            Ok(q) => q,
            Err(e) => {
                unsafe {
                    c.device.destroy_buffer(hb, None);
                    c.device.free_memory(hm, None);
                }
                return Err(format!("query pool: {e}"));
            }
        };
        let period = c.limits.timestamp_period as f64;
        let bits_at = 4 * (rdna::kernels::MARK_PAGE_BITS_AT as u64 + p0 / 32);
        let names = ["an empty list", "pages without marks", "1 dword a page", "32 consecutive dwords a page", "32 dwords a page one in 32", "every dword", "copied whole", "filled", "driver memory filled", "2 of every 3 dwords", "every other dword", "2 of every 3 dwords, one page in 4", "31 of every 32 dwords", "two runs of 15 a line"];
        // (Host memory the driver allocated, for case 8.)
        let (drv, drv_mem, _) = c.host_buffer(bench_pages * PAGE, vk::BufferUsageFlags::TRANSFER_DST | vk::BufferUsageFlags::TRANSFER_SRC)?;
        let pattern = |k: usize| -> [u32; 32] {
            let mut m = [0u32; 32];
            match k {
                2 => m[0] = 1,
                3 => m[0] = u32::MAX,
                4 => m = [1; 32],
                5 => m = [u32::MAX; 32],
                // (The swamp's records: dwords 1-2 of 12-byte records.)
                9 | 11 => {
                    for (w, x) in m.iter_mut().enumerate() {
                        for b in 0..32 {
                            if (w * 32 + b) % 3 != 0 {
                                *x |= 1 << b;
                            }
                        }
                    }
                }
                10 => m = [0x5555_5555; 32],
                12 => m = [0x7FFF_FFFF; 32],
                13 => m = [0x7FFF_7FFF; 32],
                _ => {}
            }
            m
        };
        let mut lines = Vec::new();
        let res = (|| {
            for (k, name) in names.iter().enumerate() {
                // (6-8: no write-back; the list stays empty. 11: a page in 4.)
                let listed = if k == 0 || (6..=8).contains(&k) { 0 } else if k == 11 { bench_pages / 4 } else { bench_pages };
                let mask: Vec<u8> = pattern(k).iter().flat_map(|x| x.to_le_bytes()).collect();
                let masks: Vec<u8> = (0..bench_pages).flat_map(|_| mask.iter().copied()).collect();
                let touched: Vec<u8> = (0..bench_pages / 32).flat_map(|_| (if listed > 0 { u32::MAX } else { 0u32 }).to_le_bytes()).collect();
                let step = if k == 11 { 4 } else { 1 };
                let list: Vec<u8> = [listed as u32, 1, 1, 0].into_iter().chain((0..listed).map(|i| (p0 + step * i) as u32)).flat_map(|x| x.to_le_bytes()).collect();
                let bytes = bench_pages * PAGE;
                super::resident::one_off_now(|cmd| unsafe {
                    c.device.cmd_reset_query_pool(cmd, qp, 0, n);
                    update(cmd, table, p0 * 8, &entries);
                    for r in 0..BENCH_REPS as u32 {
                        update(cmd, marks, p0 * 128, &masks);
                        update(cmd, marks, bits_at, &touched);
                        update(cmd, marks, rdna::kernels::MARK_LIST_AT as u64 * 4, &list);
                        let mb = vk::MemoryBarrier::default().src_access_mask(vk::AccessFlags::TRANSFER_WRITE).dst_access_mask(vk::AccessFlags::MEMORY_READ | vk::AccessFlags::MEMORY_WRITE);
                        c.device.cmd_pipeline_barrier(cmd, vk::PipelineStageFlags::TRANSFER, vk::PipelineStageFlags::ALL_COMMANDS, vk::DependencyFlags::empty(), &[mb], &[], &[]);
                        c.device.cmd_write_timestamp(cmd, vk::PipelineStageFlags::BOTTOM_OF_PIPE, qp, 2 * r);
                        if k < 6 || k >= 9 {
                            record_scatter(cmd, false);
                        } else {
                            let all = vk::MemoryBarrier::default().src_access_mask(vk::AccessFlags::MEMORY_WRITE).dst_access_mask(vk::AccessFlags::MEMORY_READ | vk::AccessFlags::MEMORY_WRITE);
                            c.device.cmd_pipeline_barrier(cmd, vk::PipelineStageFlags::ALL_COMMANDS, vk::PipelineStageFlags::TRANSFER, vk::DependencyFlags::empty(), &[all], &[], &[]);
                            if k == 6 {
                                c.device.cmd_copy_buffer(cmd, mirror, hb, &[vk::BufferCopy { src_offset: guest - MIRROR_BASE, dst_offset: 0, size: bytes }]);
                            } else {
                                c.device.cmd_fill_buffer(cmd, if k == 8 { drv } else { hb }, 0, bytes, 0);
                            }
                            let after = vk::MemoryBarrier::default().src_access_mask(vk::AccessFlags::TRANSFER_WRITE).dst_access_mask(vk::AccessFlags::MEMORY_READ | vk::AccessFlags::MEMORY_WRITE | vk::AccessFlags::HOST_READ);
                            c.device.cmd_pipeline_barrier(cmd, vk::PipelineStageFlags::TRANSFER, vk::PipelineStageFlags::ALL_COMMANDS | vk::PipelineStageFlags::HOST, vk::DependencyFlags::empty(), &[after], &[], &[]);
                        }
                        c.device.cmd_write_timestamp(cmd, vk::PipelineStageFlags::BOTTOM_OF_PIPE, qp, 2 * r + 1);
                        // (The next round's inputs after it.)
                        let next = vk::MemoryBarrier::default().src_access_mask(vk::AccessFlags::MEMORY_WRITE).dst_access_mask(vk::AccessFlags::TRANSFER_WRITE);
                        c.device.cmd_pipeline_barrier(cmd, vk::PipelineStageFlags::ALL_COMMANDS, vk::PipelineStageFlags::TRANSFER, vk::DependencyFlags::empty(), &[next], &[], &[]);
                    }
                })?;
                let mut t = vec![0u64; n as usize];
                unsafe { c.device.get_query_pool_results(qp, 0, &mut t, vk::QueryResultFlags::TYPE_64 | vk::QueryResultFlags::WAIT) }.map_err(|e| format!("query: {e}"))?;
                let mut us: Vec<f64> = (0..BENCH_REPS).map(|r| t[2 * r + 1].wrapping_sub(t[2 * r]) as f64 * period / 1000.0).collect();
                us.sort_by(|a, b| a.total_cmp(b));
                let m = us[BENCH_REPS / 2];
                // Bytes stored into host memory: the marked dwords, or all of it.
                let stored = match k {
                    2 => bench_pages * 4,
                    3 | 4 => bench_pages * 128,
                    5..=8 => bytes,
                    9 => bench_pages * 683 * 4,
                    10 => bench_pages * 512 * 4,
                    11 => bench_pages / 4 * 683 * 4,
                    12 => bench_pages * 992 * 4,
                    13 => bench_pages * 960 * 4,
                    _ => 0,
                };
                if stored >= 1 << 20 {
                    lines.push(format!("{name} {m:.1} us ({:.1} GB/s)", stored as f64 / (m * 1000.0)));
                } else {
                    lines.push(format!("{name} {m:.1} us"));
                }
            }
            // The copy queue's engine storing the pages into the import:
            // BENCH_REPS copies in one submission, timed on the CPU.
            if let Some(q) = c.copy.as_ref() {
                let bytes = bench_pages * PAGE;
                unsafe {
                    let pool = c.device.create_command_pool(&vk::CommandPoolCreateInfo::default().queue_family_index(q.family), None).map_err(|e| format!("copy pool: {e}"))?;
                    let r = (|| {
                        let cb = c.device.allocate_command_buffers(&vk::CommandBufferAllocateInfo::default().command_pool(pool).command_buffer_count(1)).map_err(|e| format!("copy cmd: {e}"))?[0];
                        c.device.begin_command_buffer(cb, &vk::CommandBufferBeginInfo::default().flags(vk::CommandBufferUsageFlags::ONE_TIME_SUBMIT)).map_err(|e| format!("begin: {e}"))?;
                        for _ in 0..BENCH_REPS {
                            c.device.cmd_copy_buffer(cb, mirror, hb, &[vk::BufferCopy { src_offset: guest - MIRROR_BASE, dst_offset: 0, size: bytes }]);
                            let ww = vk::MemoryBarrier::default().src_access_mask(vk::AccessFlags::TRANSFER_WRITE).dst_access_mask(vk::AccessFlags::TRANSFER_WRITE | vk::AccessFlags::HOST_READ);
                            c.device.cmd_pipeline_barrier(cb, vk::PipelineStageFlags::TRANSFER, vk::PipelineStageFlags::TRANSFER | vk::PipelineStageFlags::HOST, vk::DependencyFlags::empty(), &[ww], &[], &[]);
                        }
                        c.device.end_command_buffer(cb).map_err(|e| format!("end: {e}"))?;
                        let fence = c.device.create_fence(&vk::FenceCreateInfo::default(), None).map_err(|e| format!("fence: {e}"))?;
                        let t0 = std::time::Instant::now();
                        let sent = c.submit_timeline(true, &[cb], &[], &[], fence);
                        let waited = sent.and_then(|_| c.device.wait_for_fences(&[fence], true, 10_000_000_000).map_err(|e| format!("wait: {e}")));
                        let us = t0.elapsed().as_secs_f64() * 1e6 / BENCH_REPS as f64;
                        c.device.destroy_fence(fence, None);
                        waited?;
                        lines.push(format!("copied whole on the copy queue {us:.1} us ({:.1} GB/s, CPU-timed)", bytes as f64 / (us * 1000.0)));
                        Ok::<(), String>(())
                    })();
                    c.device.destroy_command_pool(pool, None);
                    r?;
                }
            }
            Ok::<(), String>(())
        })();
        unsafe {
            c.device.destroy_buffer(drv, None);
            c.device.free_memory(drv_mem, None);
        }
        // Its pages unmapped and their marks cleared again, whatever happened.
        let zeros_t = vec![0u8; entries.len()];
        let zeros_m = vec![0u8; (bench_pages * 128) as usize];
        let zeros_b = vec![0u8; (bench_pages / 32 * 4) as usize];
        let cleared = super::resident::one_off_now(|cmd| unsafe {
            update(cmd, table, p0 * 8, &zeros_t);
            update(cmd, marks, p0 * 128, &zeros_m);
            update(cmd, marks, bits_at, &zeros_b);
            c.device.cmd_update_buffer(cmd, marks, rdna::kernels::MARK_LIST_AT as u64 * 4, &[0u8; 4]);
        });
        unsafe {
            c.device.destroy_query_pool(qp, None);
            c.device.destroy_buffer(hb, None);
            c.device.free_memory(hm, None);
        }
        res?;
        cleared?;
        Ok(format!("write-back bench ({bench_pages} pages, median of {BENCH_REPS}): {}", lines.join(", ")))
    })();
    unsafe { VirtualFree(h as _, 0, MEM_RELEASE) };
    r
}

/// One status line ("" when off).
/// Scheduled upload bytes and completed write-back bytes (different pipeline stages).
/// Read atomics only: unlike status(), safe to sample without GPU counter readbacks.
#[cfg(target_os = "linux")]
pub fn traffic_counters() -> [u64; 3] {
    [STATS[S_UPLOAD_PAGES].load(Ordering::Relaxed) * PAGE,
     WB64[1].load(Ordering::Relaxed) * 4, STATS[S_UPLOADS].load(Ordering::Relaxed)]
}

pub fn status() -> String {
    if !on() {
        return String::new();
    }
    let s: Vec<u64> = STATS.iter().map(|c| c.load(Ordering::Relaxed)).collect();
    let k = counters();
    let spans = super::exec::fe2_span_census();
    let ends = super::precache::epoch_ends();
    let ms = |ns: u64| ns as f64 / 1e6;
    let sizes: Vec<String> = ["<64K", "<1M", "<16M", ">=16M"]
        .iter()
        .zip(BY_SIZE.iter())
        .map(|(n, c)| {
            let v: Vec<u64> = c.iter().map(|x| x.load(Ordering::Relaxed)).collect();
            format!("{n} {} ({} MiB; {} takes, {:.1} ms)", v[0], (v[1] * PAGE) >> 20, v[2], ms(v[3]))
        })
        .collect();
    let w: Vec<u64> = WB64.iter().map(|c| c.load(Ordering::Relaxed)).collect();
    format!(
        "    front end 2: {} placement checks ({} pages; {} needed nothing; {} chunks passed by their summary, {} placements by the range cache), {} takes ({} pages written, {} failed; {:.1} ms), checking {:.1} ms; uploads {} ({} pages, {} MiB; runs written lately, without a take: {}, {} pages), {} failed, {} pages valid once queued ({} raced a write); hosted arming: {} pages in {} runs ({} failed; {} view pages armed at the residency's takes adopted), {} armings ended by writes, {} writes found on pages armed and taken since (holes); checks that followed another's take {} ({} pages uploaded without a take of their own, {} took anyway); faults read {} ({} ending an arming of ours, {} outside the span, {} ring overflows), {} chunk bursts ({} pages disarmed), {} more pages unarmed with a faulted page's word; invalidated: {} ranges, {} pages by takes; write-backs recorded {}: {} pages, {} dwords, {} pages unmapped (lost to the write-back, their marks kept{}), {} lists past their capacity (before GPU-performed CP writes {}, at batch closes {}, {} of them with no label write and no completion action; CP writes with work to write back {} of labels a CP of ours waits on, {} left to the batch's close, {} with their guest-memory copy at the close); CP writes also into the mirror {}, pages copied by the CPU {}, written bindings without an import {}; imports {} live ({} uploaded from, {} settled for arming); views synchronized {} times ({} raised; by the graphics CP's own synchronization {}, the compute queues' {}; epochs ended by {} title submissions, {} waits the CPU released, {} REWIND, {} waits passed on GPU writes ended none, {} passed again on a counter's value, {} gone past ended none); the mirror's video memory {} MiB (no placement in the last 1,800 flips: {} MiB; unbound since boot {} MiB, {} blocks freed, {} cold words kept by placements), its written-bit maps' {} MiB\n    front end 2's placement checks by size: {}\n    front end 2's pages taken but not armed: {}\n",
        s[S_ENSURES],
        s[S_ENSURE_PAGES],
        s[S_FAST],
        s[S_SUMMARY],
        s[S_RANGE_HITS],
        s[S_TAKES],
        s[S_TAKE_WRITTEN],
        s[S_TAKE_FAILED],
        ms(s[S_TAKE_NS]),
        ms(s[S_SLOW_NS]),
        s[S_UPLOADS],
        s[S_UPLOAD_PAGES],
        (s[S_UPLOAD_PAGES] * PAGE) >> 20,
        s[S_HOT_RUNS],
        s[S_HOT_PAGES],
        s[S_UPLOAD_FAILED],
        s[S_VALIDATED],
        s[S_RACED],
        s[S_ARMED],
        s[S_ARM_RUNS],
        s[S_ARM_FAILED],
        s[S_ADOPTED],
        s[S_ARM_ENDED],
        HOLES.load(Ordering::Relaxed),
        FOLLOW[0].load(Ordering::Relaxed),
        FOLLOW[1].load(Ordering::Relaxed),
        FOLLOW[2].load(Ordering::Relaxed),
        FAULTS[F_SPAN].load(Ordering::Relaxed),
        FAULTS[F_ENDED].load(Ordering::Relaxed),
        FAULTS[F_OUTSIDE].load(Ordering::Relaxed),
        FAULTS[F_LOST].load(Ordering::Relaxed),
        FAULTS[F_BURSTS].load(Ordering::Relaxed),
        FAULTS[F_DISARMED].load(Ordering::Relaxed),
        FAULTS[F_WORD_DISARMED].load(Ordering::Relaxed),
        s[S_INVAL_CALLS],
        s[S_INVAL_TAKEN],
        s[S_SCATTERS],
        w[0],
        w[1],
        k[2],
        if k[7] != 0 { format!(", the last at {:#x}", rdna::translate::PT_BASE + (((k[7] - 1) as u64) << rdna::translate::PT_PAGE_SHIFT)) } else { String::new() },
        k[3],
        super::exec::SCATTER_SITES[0].load(Ordering::Relaxed),
        super::exec::SCATTER_SITES[1].load(Ordering::Relaxed),
        super::exec::SCATTER_SITES[2].load(Ordering::Relaxed),
        super::exec::SCATTER_SITES[3].load(Ordering::Relaxed),
        super::exec::SCATTER_SITES[4].load(Ordering::Relaxed),
        super::exec::SCATTER_SITES[5].load(Ordering::Relaxed),
        s[S_CP_MIRROR],
        s[S_MEMCPY],
        s[S_UNSCATTERABLE],
        spans.0,
        spans.1,
        spans.2,
        VIEW_SYNCS[0].load(Ordering::Relaxed),
        VIEW_SYNCS[1].load(Ordering::Relaxed),
        VIEW_RAISES[0].load(Ordering::Relaxed),
        VIEW_RAISES[1].load(Ordering::Relaxed),
        ends[0],
        ends[1],
        ends[2],
        super::cp::EPOCH_GPU_WAITS.load(Ordering::Relaxed),
        super::cp::QUIET_WAITS[0].load(Ordering::Relaxed),
        super::cp::QUIET_WAITS[1].load(Ordering::Relaxed),
        super::resident::backed_mib(),
        super::resident::cold_mib(1800),
        super::resident::unbacked_status().0,
        super::resident::unbacked_status().1,
        super::resident::unbacked_status().2,
        super::resident::marks_mib(),
        sizes.join(", "),
        ["armed already", "GPU-written", "written lately", "too few clean epochs", "no settled window", "not plain read-write", "chunk held", "import list busy"]
            .iter()
            .zip(NOT_ARMED.iter())
            .map(|(n, c)| format!("{n} {}", c.load(Ordering::Relaxed)))
            .collect::<Vec<_>>()
            .join(", "),
    ) + &census_line()
        + &run_share_line()
        + &callers_line()
        + &wb_buckets_line()
        + &hold_line()
        + &c4_line()
        + &behind_line()
        + &take_census_line()
        + &wb_same_line()
        + &wb_probe_line()
}

/// One status line: the mirror's pages by state now (a scan of every
/// chunk's words): valid, armed (the host's bit set), GPU-written, and those
/// found written (or first seen) within the last 8, 60 and 600 flips with the
/// runs they form (the volatile set a coherence without per-binding checks
/// would have to keep current: FRONTEND2.md, F3).
fn census_line() -> String {
    let frame = super::FRAMES.load(Ordering::Relaxed);
    let (mut valid, mut armed, mut gpu) = (0u64, 0u64, 0u64);
    // (GPU-written pages protected now (the host's bit), and those of them
    // armed: pages armed before GPU work first wrote them stay so.)
    let (mut gpu_prot, mut gpu_armed) = (0u64, 0u64);
    let mut lately = [0u64; 3];
    let mut runs = [0u64; 3];
    let windows = [8u64, 60, 600];
    for (k, slot) in TABLE.iter().enumerate() {
        let p = slot.load(Ordering::Acquire);
        if p.is_null() {
            continue;
        }
        let c = unsafe { &*p };
        let base = MIRROR_BASE + ((k as u64) << CHUNK_SHIFT);
        let mut prev = [false; 3];
        for w in 0..WORDS {
            let v = c.valid[w].load(Ordering::Relaxed);
            valid += v.count_ones() as u64;
            let host = crate::hosted::armed_word(base + (w as u64) * 64 * PAGE);
            let (a, g) = (c.armed[w].load(Ordering::Relaxed) & host, c.gpu[w].load(Ordering::Relaxed));
            armed += a.count_ones() as u64;
            gpu += g.count_ones() as u64;
            gpu_prot += (g & host).count_ones() as u64;
            gpu_armed += (g & a).count_ones() as u64;
            if c.seen[w].load(Ordering::Relaxed) == 0 {
                prev = [false; 3];
                continue;
            }
            for b in 0..64 {
                let wrote = c.wrote[w * 64 + b].load(Ordering::Relaxed) as u64;
                for (i, &n) in windows.iter().enumerate() {
                    let hit = wrote != 0 && frame.saturating_sub(wrote - 1) < n;
                    if hit {
                        lately[i] += 1;
                        if !prev[i] {
                            runs[i] += 1;
                        }
                    }
                    prev[i] = hit;
                }
            }
        }
    }
    let mib = |pages: u64| (pages * PAGE) >> 20;
    format!(
        "    front end 2's pages now: valid {} ({} MiB), armed {} ({} MiB), GPU-written {} ({} protected now, {} of them armed); written or first seen within 8 / 60 / 600 flips: {} / {} / {} pages ({} / {} / {} MiB) in {} / {} / {} runs
",
        valid,
        mib(valid),
        armed,
        mib(armed),
        gpu,
        gpu_prot,
        gpu_armed,
        lately[0],
        lately[1],
        lately[2],
        mib(lately[0]),
        mib(lately[1]),
        mib(lately[2]),
        runs[0],
        runs[1],
        runs[2]
    )
}

/// The placement checks by caller (`DS_FE2_CALLERS=1`; "" otherwise).
pub fn callers_line() -> String {
    let mut quiet = up_census_line();
    if quiet_census_on() {
        let parts: Vec<String> = CALLER_NAMES
            .iter()
            .zip(QUIET_TAKES.iter())
            .map(|(n, c)| {
                let v: Vec<u64> = c.iter().map(|x| x.load(Ordering::Relaxed)).collect();
                format!("{n} {} of {} ({:.1} of {:.1} ms)", v[2], v[0], v[3] as f64 / 1e6, v[1] as f64 / 1e6)
            })
            .collect();
        quiet += &format!("    front end 2's takes of ranges all GPU-written with no CPU write found for {QUIET_FLIPS} flips (what view-backed mappings would spare), by caller: {}\n", parts.join("; "));
    }
    if !callers_on() {
        return quiet;
    }
    let hz = super::threadcpu::tsc_hz();
    let line = |rows: &[[AtomicU64; 25]; 4]| -> Vec<String> {
        CALLER_NAMES
        .iter()
        .zip(rows.iter())
        .map(|(n, c)| {
            let v: Vec<u64> = c.iter().map(|x| x.load(Ordering::Relaxed)).collect();
            format!(
                "{n}: {} placements ({} repeating one of the batch at the same epoch ({:.1} ms), {} at an earlier one; {:.1} ms; outside the checks: mapping and backing {:.1}, batch and import {:.1} (the batch's begin {:.1}), after {:.1}), {} checks ({} by the range cache; missed: no range {}, another epoch {}, chunk generations {}, host events {}), {} words scanned, {} takes, {:.1} ms (range caches {:.1}, scans {:.1}, arming {:.1}, takes {:.1}, runs made current {:.1}, following other takes {:.1}, uploads without a take {:.1}, the rest {:.1})",
                v[C_PLACES],
                v[C_REP_SAME],
                v[C_REP_CYCLES] as f64 * 1e3 / hz,
                v[C_REP_BATCH],
                v[C_PLACE_CYCLES] as f64 * 1e3 / hz,
                v[C_PART_MAP] as f64 * 1e3 / hz,
                v[C_PART_BEGIN] as f64 * 1e3 / hz,
                v[C_PART_BATCH] as f64 * 1e3 / hz,
                v[C_PART_AFTER] as f64 * 1e3 / hz,
                v[C_CHECKS],
                v[C_HITS],
                v[C_NO_RANGE],
                v[C_EPOCH],
                v[C_GENS],
                v[C_HOST],
                v[C_WORDS],
                v[C_TAKES],
                v[C_CYCLES] as f64 * 1e3 / hz,
                v[C_CACHE] as f64 * 1e3 / hz,
                v[C_SCAN] as f64 * 1e3 / hz,
                v[C_ARM] as f64 * 1e3 / hz,
                v[C_TAKE] as f64 * 1e3 / hz,
                v[C_FIX] as f64 * 1e3 / hz,
                v[C_FOLLOW] as f64 * 1e3 / hz,
                v[C_UPLOAD] as f64 * 1e3 / hz,
                v[C_CYCLES].saturating_sub(v[C_CACHE] + v[C_SCAN] + v[C_ARM] + v[C_TAKE] + v[C_FIX] + v[C_FOLLOW] + v[C_UPLOAD]) as f64 * 1e3 / hz
            )
        })
        .collect()
    };
    let parts = line(&BY_CALLER);
    let firsts = line(&BY_CALLER_FIRST);
    let sources: Vec<String> = SOURCE_NAMES
        .iter()
        .zip(BY_SOURCE.iter())
        .map(|(n, c)| {
            let v: Vec<u64> = c.iter().map(|x| x.load(Ordering::Relaxed)).collect();
            format!(
                "{n}: {} bindings, {} checks ({} by the range cache), {:.1} ms, {} runs taken ({} pages; only pages found lacking by a take of the epoch wanted {}, with a valid page not looked at since {}, with a page M lacks not looked at since {}; pages not looked at since: one epoch before {}, older {})",
                v[B_BINDINGS],
                v[B_CHECKS],
                v[B_HITS],
                v[B_CYCLES] as f64 * 1e3 / hz,
                v[B_TAKES],
                v[B_PAGES],
                v[B_LACKING],
                v[B_VALID_STALE],
                v[B_INVALID_STALE],
                v[B_ONE_BEHIND],
                v[B_OLDER]
            )
        })
        .collect();
    format!(
        "    front end 2's checks by caller: {}\n    of them at batches' and chunks' first commands: {}\n    the graphics CP's checks by the stage they bind: {}\n{quiet}",
        parts.join("; "),
        firsts.join("; "),
        sources.join("; ")
    )
}

// ---------------------------------------------------------------------------
// Front end 4's deferred write-backs (C4; FRONTEND4.md, `DS_C4=1`, hosted).
// GPU-written pages of the engine's long-lived GPU allocations (the page
// allocator's and the GPU heap's records: ~80-87% of the write-back volume,
// FINDINGS.md 2026-11-10 night) are held: no-access through the host first,
// then their bit in `HELD_MAP` set, from which on the write-back leaves their
// dwords pending in the mirror (`rdna::kernels::mirror_scatter_held`: marks
// kept). GPU work reaches a no-access page through its import as ever
// (`DS_PROT_TEST`), so protecting first leaves no moment where the CPU could
// read a page missing GPU words. Any CPU access faults; the host suspends
// the thread and `c4_release_thread` releases the page: with no batch that
// writes it in flight and none pushed meanwhile (`exec::NOTE_GATE`), its
// mirror copy and both maps' mask words are copied out and the masks
// cleared, its held bit cleared, and with every other thread suspended the
// page is made read-write and its marked dwords stored; the thread resumes.
// Our own readers and writers release first instead of faulting (imports,
// which cannot pin a no-access page; our command processors' reads; CP
// effects and writes, never held again; file I/O; commits; textures read
// from guest memory); uploads into the mirror skip held pages; the mirror
// never unbinds them; a mapping change drops them (marks cleared).
// ---------------------------------------------------------------------------

/// Front end 4's held pages: the default since 2026-11-10 night (x1.15 at the
/// swamp, x1.015 at the vista: FINDINGS.md); `DS_C4=0` turns them off.
/// Front end 2, hosted; not with the hold census, the bucket census or
/// `DS_WB_SAME=2`.
pub fn c4_on() -> bool {
    c4_wanted() && on()
}

/// C4 is wanted (front end 2's own state apart: its write-back is made
/// before it turns on, and the first callers may come before that).
fn c4_wanted() -> bool {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ON.get_or_init(|| std::env::var("DS_C4").ok().as_deref() != Some("0") && !wb_buckets() && wb_same_mode() != 2 && std::env::var("DS_FE2_HOLD").ok().as_deref() != Some("1")) && (hosted_c4() || SPECS_AS_HOSTED.load(Ordering::Acquire))
}

/// The host serves C4's held pages: Windows' debugger host. Linux's in-process arming
/// (`DS_LINUX_ARMING=1`) serves no held pages, so C4 stays off there: its translations
/// (`Spec::marks_held`) stay the ones the pipeline store holds (turned on with arming, every
/// affected pipeline was made anew in play, 3,412 of them in a 2.5-minute session).
fn hosted_c4() -> bool {
    cfg!(windows) && crate::hosted::on()
}

/// This process only makes translations for the game's sessions (`--precompile-all`, the
/// translation fingerprint, the compile bench), which run hosted (the default): its specs take
/// C4 and C5 as those sessions do (`Spec::marks_held` keys translations). The launcher's
/// preparation ran unhosted, so from C5 on (2026-11-11) no prepared key matched a hosted
/// session's (FINDINGS.md 2026-11-13).
static SPECS_AS_HOSTED: AtomicBool = AtomicBool::new(false);

/// Before any spec is made: this process's specs are made as a hosted session's
/// (`SPECS_AS_HOSTED`).
pub fn specs_as_hosted() {
    SPECS_AS_HOSTED.store(true, Ordering::Release);
}

/// C5 is wanted: stores into held pages set no marks (`Spec::marks_held`;
/// `DS_C5=0` off). Decided at start like C4 (translations are keyed by it).
pub fn c5_wanted() -> bool {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ON.get_or_init(|| std::env::var("DS_C5").ok().as_deref() != Some("0")) && c4_wanted()
}

/// `c5 off` (in-run A/Bs): the GPU's held region kept zero, so every store
/// marks as without C5 (the shaders' held test stays).
static C5_OFF: AtomicBool = AtomicBool::new(false);
/// [copies into the GPU's held region, words copied, us].
static C5X: [AtomicU64; 3] = [const { AtomicU64::new(0) }; 3];
/// One copy at a time, each reading the host map's words when it records
/// them: the last copy leaves the GPU's region as the host map is then.
static C5_LOCK: parking_lot::Mutex<()> = parking_lot::Mutex::new(());

/// C5: records into `cmd` the copy of the held map's words over `pages`
/// (all of them with `pages` None) into both written-bit maps' held region
/// (`rdna::kernels::MARK_HELD_AT`), zeros while `c5 off`. Caller holds
/// C5_LOCK and records the barriers around it.
fn c5_record(cmd: vk::CommandBuffer, pages: Option<&[u64]>) -> u64 {
    c5_record_clearing(cmd, pages, false)
}

/// `c5_record`; with `clear`, the bits of `pages` written clear whatever
/// the host map says (a release, before it clears them there).
fn c5_record_clearing(cmd: vk::CommandBuffer, pages: Option<&[u64]>, clear: bool) -> u64 {
    let Some(m) = HELD_MAP.get() else { return 0 };
    let cleared: std::collections::HashMap<usize, u32> = match (clear, pages) {
        (true, Some(pages)) => {
            let mut h = std::collections::HashMap::new();
            for &p in pages.iter().filter(|&&p| (MIRROR_BASE..MIRROR_END).contains(&p)) {
                let i = ((p - MIRROR_BASE) / PAGE) as usize;
                *h.entry(i / 32).or_insert(0u32) |= 1u32 << (i % 32);
            }
            h
        }
        _ => std::collections::HashMap::new(),
    };
    let maps: Vec<vk::Buffer> = [false, true].iter().filter_map(|&on_c| super::resident::marks_for(on_c)).collect();
    // The words, as runs [first, end).
    let mut runs: Vec<(usize, usize)> = Vec::new();
    match pages {
        None => runs.push((0, HELD_MAP_WORDS)),
        Some(pages) => {
            let mut ws: Vec<usize> = pages.iter().filter(|&&p| (MIRROR_BASE..MIRROR_END).contains(&p)).map(|&p| ((p - MIRROR_BASE) / PAGE / 32) as usize).collect();
            ws.sort_unstable();
            ws.dedup();
            for w in ws {
                match runs.last_mut() {
                    Some(r) if r.1 == w => r.1 = w + 1,
                    _ => runs.push((w, w + 1)),
                }
            }
        }
    }
    let off = C5_OFF.load(Ordering::Acquire);
    // SAFETY: HELD_MAP_WORDS host-memory words, mapped for the process's life.
    let host = unsafe { std::slice::from_raw_parts(m.words as *const AtomicU32, HELD_MAP_WORDS) };
    let mut words = 0u64;
    let c = ctx();
    for (a, e) in runs {
        // (vkCmdUpdateBuffer takes at most 65,536 bytes a call.)
        let mut at = a;
        while at < e {
            let n = (e - at).min(16384);
            let data: Vec<u8> = (at..at + n).flat_map(|i| if off { 0u32 } else { host[i].load(Ordering::Acquire) & !cleared.get(&i).copied().unwrap_or(0) }.to_le_bytes()).collect();
            for &mb in &maps {
                unsafe { c.device.cmd_update_buffer(cmd, mb, (rdna::kernels::MARK_HELD_AT as u64 + at as u64) * 4, &data) };
            }
            words += n as u64;
            at += n;
        }
    }
    words
}

/// C5: the GPU's held region made the host map's over `pages` (all with
/// None), on the copy queue (a one-off on the main queue would wait for its
/// queued batches and drain it every hold round), waited for under C5_LOCK
/// (a release's clear, in its own one-off on the main queue, comes after
/// it). Holds and `c5 on|off`: batches already queued may see it late,
/// which only marks some stores into held pages.
fn c5_sync(pages: Option<&[u64]>) {
    if !c5_wanted() || HELD_MAP.get().is_none() {
        return;
    }
    let t = std::time::Instant::now();
    let _g = C5_LOCK.lock();
    let mut words = 0u64;
    let r = super::resident::one_off_side(|cmd| unsafe {
        let c = ctx();
        let mb = vk::MemoryBarrier::default().src_access_mask(vk::AccessFlags::MEMORY_READ | vk::AccessFlags::MEMORY_WRITE).dst_access_mask(vk::AccessFlags::TRANSFER_WRITE);
        c.device.cmd_pipeline_barrier(cmd, vk::PipelineStageFlags::ALL_COMMANDS, vk::PipelineStageFlags::TRANSFER, vk::DependencyFlags::empty(), &[mb], &[], &[]);
        words = c5_record(cmd, pages);
        let mb = vk::MemoryBarrier::default().src_access_mask(vk::AccessFlags::TRANSFER_WRITE).dst_access_mask(vk::AccessFlags::MEMORY_READ | vk::AccessFlags::MEMORY_WRITE);
        c.device.cmd_pipeline_barrier(cmd, vk::PipelineStageFlags::TRANSFER, vk::PipelineStageFlags::ALL_COMMANDS, vk::DependencyFlags::empty(), &[mb], &[], &[]);
    });
    if let Err(e) = r {
        crate::error!("fe2: C5: copying the held map to the GPU failed: {e}");
    }
    C5X[0].fetch_add(1, Ordering::Relaxed);
    C5X[1].fetch_add(words, Ordering::Relaxed);
    C5X[2].fetch_add(t.elapsed().as_micros() as u64, Ordering::Relaxed);
}

/// Control `c5 off|on` (in-run A/Bs; `DS_C5` not 0): off zeroes the GPU's
/// held region (every store marks), on copies the host map there.
pub fn c5_control(what: &str) -> Result<String, String> {
    if !c5_wanted() || !c4_on() {
        return Err("c5: not on (front end 4 on, `DS_C5` not 0)".into());
    }
    match what {
        "off" => {
            C5_OFF.store(true, Ordering::Release);
            c5_sync(None);
            Ok("held pages marked again".into())
        }
        "on" => {
            C5_OFF.store(false, Ordering::Release);
            c5_sync(None);
            Ok("held pages unmarked".into())
        }
        _ => Err("usage: c5 off|on".into()),
    }
}

/// The held-page bitmap is wanted (the hold census or C4).
fn held_map_wanted() -> bool {
    hold_census() || c4_wanted()
}

/// `c4 off` (live): no new holds (and every held page released).
static C4_OFF: AtomicBool = AtomicBool::new(false);
/// `c4 off` asked the release thread to release every held page.
static C4_ALL: AtomicBool = AtomicBool::new(false);
/// Synchronous releases asked: ([lo, hi), not held again, done).
static C4_SYNC: parking_lot::Mutex<Vec<(u64, u64, bool, std::sync::Arc<AtomicBool>)>> = parking_lot::Mutex::new(Vec::new());

/// C4's counts (since boot).
static C4X: [AtomicU64; 28] = [const { AtomicU64::new(0) }; 28];
const X_HELD: usize = 0;
const X_RUNS: usize = 1;
const X_FAILED: usize = 2;
const X_NOW: usize = 3;
/// Releases on faults: [the title, ours, other code] x [read, write].
const X_FAULTS: usize = 4;
const X_SYNC: usize = 10;
const X_ROUNDS: usize = 11;
const X_TAKE_US: usize = 12;
const X_WAITS: usize = 13;
const X_WAIT_MS: usize = 14;
const X_DROPPED: usize = 15;
const X_UP_SKIPPED: usize = 16;
const X_SYNC_LATE: usize = 17;
const X_FAILED_TAKES: usize = 18;
const X_SUSPEND_US: usize = 19;
const X_WAIT_TIMEOUTS: usize = 20;
const X_CANDIDATES: usize = 21;
const X_PLAIN: usize = 22;
const X_RELEASED: usize = 23;
/// Pages released through host memory (no valid window).
const X_CPU_PAGES: usize = 24;
/// Runs the host refused: a page of them took a write fault (or was
/// disarmed) since it was looked at.
const X_UNDONE: usize = 25;
/// C4's own arming: pages its rounds offered to front end 2's rules, and
/// those armed.
const X_ARM_TRIED: usize = 26;
const X_ARM_DONE: usize = 27;

thread_local! {
    /// The release thread (it never waits for itself).
    static C4_RELEASER: std::cell::Cell<bool> = const { std::cell::Cell::new(false) };
    /// The hold round arming for C4 (`c4_round`): `arm_part` notes what it
    /// armed, `not_armed` counts apart.
    static C4_ARMING: std::cell::Cell<bool> = const { std::cell::Cell::new(false) };
}

/// C4's own arming (`DS_C4_ARM=0` off): target pages written back lately
/// that front end 2 has not armed are armed by the hold round itself, by
/// front end 2's rules (`try_arm`: quiet, found clean in enough epochs,
/// inside settled import windows), so a later round can hold them. Front
/// end 2 arms only at its checks, and the graphics CP's checks take without
/// arming: pages only the graphics CP's work reads were never armed, so
/// never held (the 8 MiB per-view buffer at the swamp, 581,235 of its
/// 585,727 page-rounds written back lately and not held; `DS_C4_WHY`,
/// 2026-11-11 evening).
fn c4_arm_wanted() -> bool {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ON.get_or_init(|| std::env::var("DS_C4_ARM").ok().as_deref() != Some("0"))
}

/// `c4arm off` (in-run A/Bs): the rounds arm nothing, and the pages they
/// armed are released and not held again until `c4arm on`.
static C4_ARM_OFF: AtomicBool = AtomicBool::new(false);

/// The pages C4's rounds armed (a bit per page of the mirror's span, as the
/// held map's).
fn c4_armed_map() -> &'static [AtomicU32] {
    static M: std::sync::OnceLock<Box<[AtomicU32]>> = std::sync::OnceLock::new();
    M.get_or_init(|| (0..HELD_MAP_WORDS).map(|_| AtomicU32::new(0)).collect())
}

/// `arm_part` armed [x, y) in C4's round (`C4_ARMING`).
fn c4_note_armed(x: u64, y: u64) {
    let m = c4_armed_map();
    let (lo, hi) = (x.max(MIRROR_BASE), y.min(MIRROR_END));
    let mut p = lo;
    while p < hi {
        let i = ((p - MIRROR_BASE) / PAGE) as usize;
        m[i / 32].fetch_or(1 << (i % 32), Ordering::Relaxed);
        p += PAGE;
    }
    C4X[X_ARM_DONE].fetch_add(hi.saturating_sub(lo) / PAGE, Ordering::Relaxed);
}

/// Whether C4's rounds armed page `p` (`c4_note_armed`).
fn c4_armed_by_round(p: u64) -> bool {
    let i = ((p - MIRROR_BASE) / PAGE) as usize;
    c4_armed_map()[i / 32].load(Ordering::Relaxed) >> (i % 32) & 1 != 0
}

/// `c4arm off|on` (in-run A/Bs of C4's own arming): off releases the held
/// pages its rounds armed (through the release thread, not waited for) and
/// holds none of them; on arms and holds them again from the next rounds.
pub fn c4_arm_control(what: &str) -> Result<String, String> {
    if !c4_on() || !c4_arm_wanted() {
        return Err("c4arm: not on (front end 4 on, `DS_C4_ARM` not 0)".into());
    }
    match what {
        "off" => {
            C4_ARM_OFF.store(true, Ordering::Release);
            let Some(m) = HELD_MAP.get() else { return Ok("the rounds' arming off".into()) };
            // SAFETY: as `held_mark`.
            let held = unsafe { std::slice::from_raw_parts(m.words as *const AtomicU32, HELD_MAP_WORDS) };
            let a = c4_armed_map();
            let mut runs: Vec<(u64, u64)> = Vec::new();
            for i in 0..HELD_MAP_WORDS {
                let mut v = held[i].load(Ordering::Acquire) & a[i].load(Ordering::Relaxed);
                while v != 0 {
                    let bit = v.trailing_zeros() as u64;
                    v &= v - 1;
                    let p = MIRROR_BASE + (i as u64 * 32 + bit) * PAGE;
                    match runs.last_mut() {
                        Some(l) if l.1 == p => l.1 = p + PAGE,
                        _ => runs.push((p, p + PAGE)),
                    }
                }
            }
            let n: u64 = runs.iter().map(|r| (r.1 - r.0) / PAGE).sum();
            let mut g = C4_SYNC.lock();
            for (p, q) in runs {
                g.push((p, q, false, std::sync::Arc::new(AtomicBool::new(false))));
            }
            Ok(format!("the rounds' arming off, {n} held pages it armed released"))
        }
        "on" => {
            C4_ARM_OFF.store(false, Ordering::Release);
            Ok("the rounds' arming on".into())
        }
        _ => Err("usage: c4arm off|on".into()),
    }
}

/// Whether a page of guest [lo, hi) is held (the bitmap; false without C4).
pub fn c4_any_held(lo: u64, hi: u64) -> bool {
    if !c4_on() {
        return false;
    }
    let Some(m) = HELD_MAP.get() else { return false };
    let (lo, hi) = (lo.max(MIRROR_BASE), hi.min(MIRROR_END));
    if lo >= hi {
        return false;
    }
    let (a, b) = (((lo - MIRROR_BASE) / PAGE) as usize, ((hi - MIRROR_BASE).div_ceil(PAGE)) as usize);
    // SAFETY: HELD_MAP_WORDS host-memory words, mapped for the process's life.
    let words = unsafe { std::slice::from_raw_parts(m.words as *const AtomicU32, HELD_MAP_WORDS) };
    let mut i = a;
    while i < b {
        let n = (32 - i % 32).min(b - i);
        let mask = if n >= 32 { u32::MAX } else { (1u32 << n) - 1 };
        if words[i / 32].load(Ordering::Acquire) >> (i % 32) & mask != 0 {
            return true;
        }
        i += n;
    }
    false
}

/// Whether every page of guest [lo, hi) is held (the bitmap; false without
/// C4 or for an empty range).
pub fn c4_all_held(lo: u64, hi: u64) -> bool {
    if !c4_on() || lo >= hi || lo < MIRROR_BASE || hi > MIRROR_END {
        return false;
    }
    let Some(m) = HELD_MAP.get() else { return false };
    let (a, b) = (((lo - MIRROR_BASE) / PAGE) as usize, ((hi - MIRROR_BASE).div_ceil(PAGE)) as usize);
    // SAFETY: HELD_MAP_WORDS host-memory words, mapped for the process's life.
    let words = unsafe { std::slice::from_raw_parts(m.words as *const AtomicU32, HELD_MAP_WORDS) };
    let mut i = a;
    while i < b {
        let n = (32 - i % 32).min(b - i);
        let mask = if n >= 32 { u32::MAX } else { (1u32 << n) - 1 };
        if (words[i / 32].load(Ordering::Acquire) >> (i % 32)) & mask != mask {
            return false;
        }
        i += n;
    }
    true
}

/// Whether page `p` is held (the bitmap).
fn c4_held(p: u64) -> bool {
    c4_any_held(p, p + PAGE)
}

/// The runs of [lo, hi) outside held pages, appended to `out` (uploads into
/// the mirror; the pages left out counted).
pub fn c4_unheld_runs(lo: u64, hi: u64, out: &mut Vec<(u64, u64)>) {
    let mut a = lo;
    let mut skipped = 0u64;
    while a < hi {
        let pe = ((a & !(PAGE - 1)) + PAGE).min(hi);
        if c4_held(a & !(PAGE - 1)) {
            skipped += 1;
            a = pe;
            continue;
        }
        match out.last_mut() {
            Some(l) if l.1 == a => l.1 = pe,
            _ => out.push((a, pe)),
        }
        a = pe;
    }
    if skipped > 0 {
        let n = C4X[X_UP_SKIPPED].fetch_add(skipped, Ordering::Relaxed);
        if n < 16 {
            // (Front end 2 took the page for CPU-written: the mirror would be
            // stale if it was. Expected 0 but for runs uploaded whole.)
            let wrote = chunk(lo, false).map_or(0, |c| c.wrote[idx(lo)].load(Ordering::Acquire));
            crate::warn!("fe2: C4: an upload of {lo:#x}..{hi:#x} left out {skipped} held pages (last CPU write seen at flip {}, now {})", wrote.saturating_sub(1), super::FRAMES.load(Ordering::Relaxed));
        }
    }
}

/// Uploads that left out held pages (`exec`'s copies without a window).
pub fn c4_note_upload_skips(n: u64) {
    C4X[X_UP_SKIPPED].fetch_add(n, Ordering::Relaxed);
}

/// Pages never held by the engine allocator owning them (`engine::owner`)
/// and why: [our command processors' reads, CP and CPU-action targets,
/// faults] (status line: what front end 4's next step would let C4 hold).
static NO_HOLD_BY: [[AtomicU64; 3]; 256] = [const { [const { AtomicU64::new(0) }; 3] }; 256];
pub const NO_HOLD_READ: usize = 0;
pub const NO_HOLD_TARGET: usize = 1;
pub const NO_HOLD_FAULT: usize = 2;

/// Pages of guest [a, a+n) are never held (CP targets, our labels; `why`:
/// NO_HOLD_*).
pub fn c4_no_hold(a: u64, n: u64, why: usize) {
    if !c4_on() {
        return;
    }
    let lo = (a & !(PAGE - 1)).max(MIRROR_BASE);
    let hi = a.saturating_add(n.max(1)).next_multiple_of(PAGE).min(MIRROR_END);
    let mut p = lo;
    while p < hi {
        let i = ((p - MIRROR_BASE) / PAGE) as usize;
        let bit = 1u32 << (i % 32);
        // (Tested first: every preparation's reads come here, ~20-34 pages a
        // draw, nearly all marked already; a shared read-modify-write each
        // made the word's line bounce between the recording threads.)
        let w = &no_hold()[i / 32];
        if w.load(Ordering::Relaxed) & bit == 0 && w.fetch_or(bit, Ordering::Relaxed) & bit == 0 {
            NO_HOLD_PAGES.fetch_add(1, Ordering::Relaxed);
            NO_HOLD_BY[crate::engine::owner(p) as usize][why.min(2)].fetch_add(1, Ordering::Relaxed);
        }
        p += PAGE;
    }
}

/// A CP write or a CPU action targets guest [a, a+n): its pages are never
/// held, and released first if they are.
pub fn c4_cp_target(a: u64, n: u64) {
    if !c4_on() {
        return;
    }
    c4_no_hold(a, n, NO_HOLD_TARGET);
    c4_release_sync(a, n, true);
}

thread_local! {
    /// Pages this thread's reads marked never held and found unheld
    /// (`c4_release_reads`), direct-mapped by page number: a read on such pages
    /// only needs neither again (a never-held mark is never cleared, and a hold
    /// that raced the mark ends by the read's fault as before). The
    /// preparations' reads repeat their pages draw after draw (the marking was
    /// ~0.8% of the graphics CP's busy samples and ~1.4% of its workers' at
    /// Castlewall Front, 2026-11-11). (Until 2026-11-16 night the last 16 page
    /// ranges, scanned per read: ~200 ns of a stage's ~15 reads, the largest
    /// part of its shader-read sync.)
    static READ_MARKED: std::cell::RefCell<Box<[u64; READ_MARKED_SLOTS]>> = std::cell::RefCell::new(Box::new([u64::MAX; READ_MARKED_SLOTS]));
}
const READ_MARKED_SLOTS: usize = 1024;

/// Whether `c4_release_reads` would do nothing for `reads`: front end 4 off, or every page of
/// them in this thread's read memo (`READ_MARKED`; the lean shader-read sync).
#[inline]
pub fn c4_reads_known(reads: &[(u64, u64)]) -> bool {
    if !c4_on() {
        return true;
    }
    READ_MARKED.with(|m| {
        let m = m.borrow();
        reads.iter().all(|&(a, n)| {
            let (lo, hi) = (a & !(PAGE - 1), a.saturating_add(n.max(1)).next_multiple_of(PAGE));
            let mut p = lo;
            while p < hi {
                if m[(p / PAGE) as usize & (READ_MARKED_SLOTS - 1)] != p {
                    return false;
                }
                p += PAGE;
            }
            true
        })
    })
}

/// Our command processors read `reads` on the CPU: held pages released first.
pub fn c4_release_reads(reads: &[(u64, u64)]) {
    if !c4_on() {
        return;
    }
    READ_MARKED.with(|m| {
        let mut m = m.borrow_mut();
        for &(a, n) in reads {
            let (lo, hi) = (a & !(PAGE - 1), a.saturating_add(n.max(1)).next_multiple_of(PAGE));
            let slot = |p: u64| (p / PAGE) as usize & (READ_MARKED_SLOTS - 1);
            let mut p = lo;
            while p < hi && m[slot(p)] == p {
                p += PAGE;
            }
            if p >= hi {
                continue;
            }
            // (Pages our command processors read are not held: marked first,
            // so that a later selection leaves them out.)
            c4_no_hold(a, n, NO_HOLD_READ);
            if c4_any_held(a, a.saturating_add(n.max(1))) {
                c4_release_sync(a, n, false);
            }
            if hi - lo <= 16 * PAGE {
                let mut p = lo;
                while p < hi {
                    m[slot(p)] = p;
                    p += PAGE;
                }
            }
        }
    });
}

/// Releases the held pages of guest [base, base+len) and waits (5 s at
/// most, then logs and goes on): imports, file I/O, commits, textures and
/// our reads, before they touch the pages. `never`: not held again.
pub fn c4_release_sync(base: u64, len: u64, never: bool) {
    if !c4_on() {
        return;
    }
    let hi = base.saturating_add(len.max(1));
    if !c4_any_held(base, hi) {
        return;
    }
    if C4_RELEASER.with(|r| r.get()) {
        crate::error!("fe2: C4: the release thread asked to release {base:#x}+{len:#x} itself (skipped)");
        return;
    }
    let done = std::sync::Arc::new(AtomicBool::new(false));
    C4_SYNC.lock().push((base, hi, never, done.clone()));
    let t = std::time::Instant::now();
    while !done.load(Ordering::Acquire) {
        if t.elapsed() > std::time::Duration::from_secs(5) {
            if C4X[X_SYNC_LATE].fetch_add(1, Ordering::Relaxed) < 8 {
                crate::error!("fe2: C4: held pages of {base:#x}+{len:#x} not released within 5 s");
            }
            return;
        }
        std::thread::sleep(std::time::Duration::from_micros(100));
    }
}

/// Guest [base, base+len) is about to be decommitted (`mem::host_decommit`):
/// its held pages' pending words die with it; their bits and both maps'
/// mask words cleared and the pages made read-write again (as unheld pages
/// are before a decommit), no words put back.
pub fn c4_drop_sync(base: u64, len: u64) {
    if !c4_on() || !c4_any_held(base, base.saturating_add(len)) {
        return;
    }
    let lo = (base & !(PAGE - 1)).max(MIRROR_BASE);
    let hi = base.saturating_add(len).next_multiple_of(PAGE).min(MIRROR_END);
    let _lock = crate::hosted::arm_lock();
    let mut pages = Vec::new();
    let mut p = lo;
    while p < hi {
        if held_mark(p, false) {
            pages.push(p);
        }
        p += PAGE;
    }
    if pages.is_empty() {
        return;
    }
    let maps: Vec<vk::Buffer> = [false, true].iter().filter_map(|&on_c| super::resident::marks_for(on_c)).collect();
    let _c5 = C5_LOCK.lock();
    let r = super::resident::one_off_now(|cmd| unsafe {
        let c = ctx();
        let mb = vk::MemoryBarrier::default().src_access_mask(vk::AccessFlags::MEMORY_WRITE).dst_access_mask(vk::AccessFlags::TRANSFER_WRITE);
        c.device.cmd_pipeline_barrier(cmd, vk::PipelineStageFlags::ALL_COMMANDS, vk::PipelineStageFlags::TRANSFER, vk::DependencyFlags::empty(), &[mb], &[], &[]);
        for &m in &maps {
            for &p in &pages {
                c.device.cmd_fill_buffer(cmd, m, (p - MIRROR_BASE) / PAGE * 128, 128, 0);
            }
        }
        // (C5: their stores marked again.)
        if c5_wanted() {
            c5_record(cmd, Some(&pages));
        }
        let mb = vk::MemoryBarrier::default().src_access_mask(vk::AccessFlags::TRANSFER_WRITE).dst_access_mask(vk::AccessFlags::MEMORY_READ | vk::AccessFlags::MEMORY_WRITE);
        c.device.cmd_pipeline_barrier(cmd, vk::PipelineStageFlags::TRANSFER, vk::PipelineStageFlags::ALL_COMMANDS, vk::DependencyFlags::empty(), &[mb], &[], &[]);
    });
    if let Err(e) = r {
        crate::error!("fe2: C4: clearing the marks of {} held pages before a decommit failed: {e}", pages.len());
    }
    let mut a = 0;
    while a < pages.len() {
        let mut e = a + 1;
        while e < pages.len() && pages[e] == pages[e - 1] + PAGE {
            e += 1;
        }
        let _ = crate::hosted::release(pages[a], (e - a) as u64 * PAGE);
        a = e;
    }
    C4X[X_DROPPED].fetch_add(pages.len() as u64, Ordering::Relaxed);
    C4X[X_NOW].fetch_sub(pages.len() as u64, Ordering::Relaxed);
}

/// A mapping change of guest [base, base+len) is settled (`resident::
/// settled`, the host's bits cleared): held pages there are dropped, their
/// pending words dead; their bits and both maps' mask words cleared (now,
/// before the new mapping is used).
pub fn c4_forgotten(base: u64, len: u64) {
    if !c4_on() || !c4_any_held(base, base.saturating_add(len)) {
        return;
    }
    let lo = (base & !(PAGE - 1)).max(MIRROR_BASE);
    let hi = base.saturating_add(len).next_multiple_of(PAGE).min(MIRROR_END);
    let mut pages = Vec::new();
    let mut p = lo;
    while p < hi {
        if held_mark(p, false) {
            pages.push(p);
        }
        p += PAGE;
    }
    if pages.is_empty() {
        return;
    }
    C4X[X_DROPPED].fetch_add(pages.len() as u64, Ordering::Relaxed);
    C4X[X_NOW].fetch_sub(pages.len() as u64, Ordering::Relaxed);
    let maps: Vec<vk::Buffer> = [false, true].iter().filter_map(|&on_c| super::resident::marks_for(on_c)).collect();
    let _c5 = C5_LOCK.lock();
    let r = super::resident::one_off_now(|cmd| unsafe {
        let c = ctx();
        let mb = vk::MemoryBarrier::default().src_access_mask(vk::AccessFlags::MEMORY_WRITE).dst_access_mask(vk::AccessFlags::TRANSFER_WRITE);
        c.device.cmd_pipeline_barrier(cmd, vk::PipelineStageFlags::ALL_COMMANDS, vk::PipelineStageFlags::TRANSFER, vk::DependencyFlags::empty(), &[mb], &[], &[]);
        for &m in &maps {
            for &p in &pages {
                c.device.cmd_fill_buffer(cmd, m, (p - MIRROR_BASE) / PAGE * 128, 128, 0);
            }
        }
        // (C5: their stores marked again.)
        if c5_wanted() {
            c5_record(cmd, Some(&pages));
        }
        let mb = vk::MemoryBarrier::default().src_access_mask(vk::AccessFlags::TRANSFER_WRITE).dst_access_mask(vk::AccessFlags::MEMORY_READ | vk::AccessFlags::MEMORY_WRITE);
        c.device.cmd_pipeline_barrier(cmd, vk::PipelineStageFlags::TRANSFER, vk::PipelineStageFlags::ALL_COMMANDS, vk::DependencyFlags::empty(), &[mb], &[], &[]);
    });
    if let Err(e) = r {
        crate::error!("fe2: C4: clearing the marks of {} dropped held pages failed: {e}", pages.len());
    }
}

/// `DS_C4_WHY=1` (diagnosis): per allocation record of C4's target
/// allocators (`engine::record_key`), its pages written back lately and not
/// held, summed over the hold rounds: [all, never held (`NO_HOLD`), not
/// armed by front end 2, not valid in the mirror, not armed by the host,
/// written by the CPU within `DS_C4_QUIET` flips, candidates].
fn c4_why() -> bool {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ON.get_or_init(|| std::env::var("DS_C4_WHY").ok().as_deref() == Some("1"))
}
static C4_WHY: parking_lot::Mutex<Option<std::collections::HashMap<(u64, u64, u8), [u64; 7]>>> = parking_lot::Mutex::new(None);

/// Pages written back within the last two hold periods, summed over the
/// hold rounds, by the engine allocator owning them (`c4_round`; status).
static WB_BY_OWNER: [AtomicU64; 256] = [const { AtomicU64::new(0) }; 256];
static WB_ROUNDS: AtomicU64 = AtomicU64::new(0);

/// `DS_C4_QUIET=<flips>`: a page is held only that long after the last CPU
/// write front end 2 saw on it (600 by default; 0: front end 2's arming
/// alone).
fn c4_quiet() -> u64 {
    static Q: std::sync::OnceLock<u64> = std::sync::OnceLock::new();
    *Q.get_or_init(|| std::env::var("DS_C4_QUIET").ok().and_then(|v| v.parse().ok()).unwrap_or(600))
}

/// The allocator IDs whose records C4 holds: the engine's GPU page
/// allocator and GPU heaps (long-lived GPU allocations). Not the GPU arena's
/// frame allocations: held, the engine's frame allocation code wrote them
/// and released them ~0.3 times a flip (the swamp -10% in boots,
/// TRIED.md 2026-11-11).
fn c4_targets() -> [bool; 256] {
    let mut t = [false; 256];
    for a in crate::engine::allocators() {
        if a.mtype == 2 && (a.name.starts_with("PageAllocator") || a.name.starts_with("HeapAllocator")) {
            t[a.id as usize] = true;
        }
    }
    t
}

/// One selection round (`hold_thread`, after the written-back bits): pages
/// written back within the last two periods, in a target allocator's record,
/// armed by front end 2 (quiet, valid in the mirror) and through the host,
/// not held, never refused: held in runs.
fn c4_round() {
    if !c4_on() || C4_OFF.load(Ordering::Acquire) {
        return;
    }
    let Some(recent) = WB_RECENT.get() else { return };
    let Some(m) = HELD_MAP.get() else { return };
    // SAFETY: as `held_mark`.
    let held = unsafe { std::slice::from_raw_parts(m.words as *const AtomicU32, HELD_MAP_WORDS) };
    let targets = c4_targets();
    let nh = no_hold();
    let frame = super::FRAMES.load(Ordering::Relaxed);
    let quiet = c4_quiet();
    // (Census: the pages written back lately, by the engine allocation
    // owning them: where front end 2's write-back volume is left.)
    WB_ROUNDS.fetch_add(1, Ordering::Relaxed);
    for (i, w) in recent.iter().enumerate() {
        let mut v = w.load(Ordering::Relaxed);
        while v != 0 {
            let b = v.trailing_zeros() as u64;
            v &= v - 1;
            let p = MIRROR_BASE + (i as u64 * 32 + b) * PAGE;
            WB_BY_OWNER[crate::engine::owner(p) as usize].fetch_add(1, Ordering::Relaxed);
        }
    }
    if c4_why() {
        let mut why: std::collections::HashMap<(u64, u64, u8), [u64; 7]> = std::collections::HashMap::new();
        for i in 0..HELD_MAP_WORDS {
            let mut v = recent[i].load(Ordering::Relaxed) & !held[i].load(Ordering::Acquire);
            while v != 0 {
                let bit = v.trailing_zeros() as u64;
                v &= v - 1;
                let p = MIRROR_BASE + (i as u64 * 32 + bit) * PAGE;
                let key = crate::engine::record_key(p);
                if !targets[key.2 as usize] {
                    continue;
                }
                let e = why.entry(key).or_insert([0; 7]);
                e[0] += 1;
                if nh[i].load(Ordering::Relaxed) >> bit & 1 != 0 {
                    e[1] += 1;
                    continue;
                }
                let Some(c) = chunk(p, false) else {
                    e[3] += 1;
                    continue;
                };
                let (k, w) = (idx(p) % 64, idx(p) / 64);
                if (c.armed[w].load(Ordering::Acquire) & c.varmed[w].load(Ordering::Acquire)) >> k & 1 == 0 {
                    e[2] += 1;
                } else if c.valid[w].load(Ordering::Acquire) >> k & 1 == 0 {
                    e[3] += 1;
                } else if !crate::hosted::armed(p) {
                    e[4] += 1;
                } else {
                    let wrote = c.wrote[idx(p)].load(Ordering::Acquire) as u64;
                    if wrote != 0 && frame.saturating_sub(wrote - 1) < quiet {
                        e[5] += 1;
                    } else {
                        e[6] += 1;
                    }
                }
            }
        }
        let mut g = C4_WHY.lock();
        let m = g.get_or_insert_with(std::collections::HashMap::new);
        for (k, v) in why {
            let e = m.entry(k).or_insert([0; 7]);
            for j in 0..7 {
                e[j] += v[j];
            }
        }
    }
    let mut runs: Vec<(u64, u64)> = Vec::new();
    let mut cands = 0u64;
    // (C4's own arming: the unarmed pages, offered to front end 2's rules
    // after the holds; with `c4arm off` the pages it armed are left alone.)
    let arm_off = c4_arm_wanted() && C4_ARM_OFF.load(Ordering::Acquire);
    let arm_rounds = c4_arm_wanted() && !arm_off;
    let mut aruns: Vec<(u64, u64)> = Vec::new();
    for i in 0..HELD_MAP_WORDS {
        let mut v = recent[i].load(Ordering::Relaxed) & !held[i].load(Ordering::Acquire) & !nh[i].load(Ordering::Relaxed);
        while v != 0 {
            let b = v.trailing_zeros() as u64;
            v &= v - 1;
            let p = MIRROR_BASE + (i as u64 * 32 + b) * PAGE;
            if !targets[crate::engine::record_key(p).2 as usize] {
                continue;
            }
            let Some(c) = chunk(p, false) else { continue };
            let (k, w) = (idx(p) % 64, idx(p) / 64);
            if c.armed[w].load(Ordering::Acquire) >> k & 1 == 0 {
                if arm_rounds {
                    match aruns.last_mut() {
                        Some(l) if l.1 == p => l.1 = p + PAGE,
                        _ => aruns.push((p, p + PAGE)),
                    }
                }
                continue;
            }
            if arm_off && c4_armed_by_round(p) {
                continue;
            }
            // (Valid, and taken since its arming (`Chunk::varmed`): a CPU
            // write between the page's last take and its arming is found by
            // the take after the arming and the page made invalid. A page
            // held with such a write missing from the mirror would lose it
            // at its release, which copies the whole page from the mirror:
            // C4's own arming took nothing after arming until 2026-11-11
            // night, and a walk boot crashed in the engine's heap allocator
            // code that reads held pages.)
            if c.valid[w].load(Ordering::Acquire) >> k & 1 == 0 || c.varmed[w].load(Ordering::Acquire) >> k & 1 == 0 || !crate::hosted::armed(p) {
                continue;
            }
            // (Pages the CPU wrote within `c4_quiet` flips are left alone:
            // the title refills streamed GPU heap blocks, and each such hold
            // ends in a release.)
            let wrote = c.wrote[idx(p)].load(Ordering::Acquire) as u64;
            if wrote != 0 && frame.saturating_sub(wrote - 1) < quiet {
                continue;
            }
            cands += 1;
            match runs.last_mut() {
                Some(l) if l.1 == p => l.1 = p + PAGE,
                _ => runs.push((p, p + PAGE)),
            }
        }
    }
    C4X[X_CANDIDATES].fetch_add(cands, Ordering::Relaxed);
    c4_hold_runs(&runs);
    // C4's own arming, after the holds: a page armed now is held a round
    // later at the earliest (a CPU write meanwhile ends its arming as any
    // other's, with front end 2's backoff).
    if !aruns.is_empty() {
        C4X[X_ARM_TRIED].fetch_add(aruns.iter().map(|r| (r.1 - r.0) / PAGE).sum::<u64>(), Ordering::Relaxed);
        C4_ARMING.with(|f| f.set(true));
        let mut written: Vec<u64> = Vec::new();
        for &(p, q) in &aruns {
            let before = C4X[X_ARM_DONE].load(Ordering::Relaxed);
            try_arm(p, q, frame, &super::exec::fe2_armable);
            if C4X[X_ARM_DONE].load(Ordering::Relaxed) != before {
                // (As `take_run`: armed, then taken. A CPU write after the
                // page's last take and before this arming is found here and
                // the page made invalid; the pages armed are taken since
                // their arming, which a hold requires.)
                hole_snapshot(p, q);
                written.clear();
                let r = super::precache::take_written_for(p, q - p, &mut written, None, crate::mem::WW_WHO_FE2);
                if r.is_some() {
                    hole_check(&written);
                } else {
                    invalidate(p, q - p);
                }
            }
        }
        C4_ARMING.with(|f| f.set(false));
    }
}

/// Holds `runs` (sorted guest ranges): through the host, which protects
/// them in its event loop in order with the faults it answers
/// (`hosted::hold_by_host`: a run whose page took a write fault since it
/// was looked at is refused), then the bitmap's bits of those it held (the
/// write-back leaves their dwords pending from then on).
fn c4_hold_runs(runs: &[(u64, u64)]) {
    // (C5: the pages held, copied to the GPU's held region once no lock is
    // held: a wait for the main queue under the host's arming lock could
    // wait on a completion that needs it. Late is safe: until then their
    // stores mark as before.)
    let mut c5_pages: Vec<u64> = Vec::new();
    c4_hold_parts(runs, &mut c5_pages);
    if !c5_pages.is_empty() {
        c5_sync(Some(&c5_pages));
    }
}

fn c4_hold_parts(runs: &[(u64, u64)], c5_pages: &mut Vec<u64>) {
    for part in runs.chunks(crate::hosted::HOLD_RUNS) {
        let _readers = crate::mem::GUEST_READERS.read_recursive();
        let _arming = super::resident::arm_guard();
        let _lock = crate::hosted::arm_lock();
        if C4_OFF.load(Ordering::Acquire) {
            return;
        }
        let want: Vec<(u64, u64)> = part.iter().copied().filter(|&(p, q)| !host_writing(p, q) && crate::hosted::all_armed(p, q - p)).collect();
        if want.is_empty() {
            continue;
        }
        let t = std::time::Instant::now();
        let held = crate::hosted::hold_by_host(&want);
        super::hitch::note_since(super::hitch::Kind::Mirror, t, || format!("C4 holding {} runs", want.len()));
        for (i, &(p, q)) in want.iter().enumerate() {
            if !held[i] {
                C4X[X_UNDONE].fetch_add(1, Ordering::Relaxed);
                continue;
            }
            let mut n = 0u64;
            for a in (p..q).step_by(PAGE as usize) {
                if !held_mark(a, true) {
                    n += 1;
                }
            }
            C4X[X_HELD].fetch_add(n, Ordering::Relaxed);
            C4X[X_RUNS].fetch_add(1, Ordering::Relaxed);
            C4X[X_NOW].fetch_add(n, Ordering::Relaxed);
        }
        if c5_wanted() {
            c5_pages.extend(want.iter().enumerate().filter(|&(i, _)| held[i]).flat_map(|(_, &(p, q))| (p..q).step_by(PAGE as usize)));
        }
    }
}

/// The release thread (`DS_C4`): faults on held pages (the host's
/// requests), synchronous releases and `c4 off`, served in rounds.
fn c4_release_thread() {
    C4_RELEASER.with(|r| r.set(true));
    let mut reqs: Vec<crate::hosted::HeldRequest> = Vec::new();
    loop {
        crate::hosted::wait_held(2);
        reqs.clear();
        crate::hosted::take_held(&mut reqs);
        let sync = std::mem::take(&mut *C4_SYNC.lock());
        let all = C4_ALL.swap(false, Ordering::AcqRel);
        if reqs.is_empty() && sync.is_empty() && !all {
            continue;
        }
        let mut pages: Vec<u64> = reqs.iter().map(|r| r.page).collect();
        for &(lo, hi, never, _) in &sync {
            let mut p = (lo & !(PAGE - 1)).max(MIRROR_BASE);
            while p < hi.min(MIRROR_END) {
                if c4_held(p) {
                    pages.push(p);
                }
                if never {
                    c4_no_hold(p, PAGE, NO_HOLD_TARGET);
                }
                p += PAGE;
            }
        }
        if all {
            if let Some(m) = HELD_MAP.get() {
                // SAFETY: as `held_mark`.
                let held = unsafe { std::slice::from_raw_parts(m.words as *const AtomicU32, HELD_MAP_WORDS) };
                for (i, w) in held.iter().enumerate() {
                    let mut v = w.load(Ordering::Acquire);
                    while v != 0 {
                        let b = v.trailing_zeros() as u64;
                        v &= v - 1;
                        pages.push(MIRROR_BASE + (i as u64 * 32 + b) * PAGE);
                    }
                }
            }
        }
        pages.sort_unstable();
        pages.dedup();
        // (A fault on a page the host holds without a bitmap bit, or none
        // held any more: released plainly; nothing is pending there. Told
        // apart under the arming lock, under which a hold protects the page
        // and sets its bit.)
        let held = {
            let _lock = crate::hosted::arm_lock();
            let (held, plain): (Vec<u64>, Vec<u64>) = pages.into_iter().partition(|&p| c4_held(p));
            for &p in &plain {
                if crate::hosted::held(p) {
                    let _ = crate::hosted::release(p, PAGE);
                    C4X[X_PLAIN].fetch_add(1, Ordering::Relaxed);
                }
            }
            held
        };
        if !held.is_empty() {
            for part in held.chunks(16384) {
                if let Err(e) = c4_release_pages(part)
                    && C4X[X_FAILED_TAKES].fetch_add(1, Ordering::Relaxed) < 8
                {
                    crate::error!("fe2: C4: releasing {} held pages failed: {e}", part.len());
                }
            }
        }
        let frame = super::FRAMES.load(Ordering::Relaxed);
        let (b, e) = crate::exception::host_image();
        for r in &reqs {
            let origin = if (crate::mem::GUEST_BASE..crate::mem::GUEST_END).contains(&r.rip) {
                0
            } else if (b..e).contains(&r.rip) {
                1
            } else {
                2
            };
            C4X[X_FAULTS + 2 * origin + r.write as usize].fetch_add(1, Ordering::Relaxed);
            *HOLD_RIPS.lock().get_or_insert_with(Default::default).entry((r.rip, r.write)).or_insert(0) += 1;
            crate::engine::note_hold_fault_record(r.page, origin, r.write);
            // (A page the CPU touched is not held again; its arming level
            // raised as a fault's.)
            c4_no_hold(r.page, PAGE, NO_HOLD_FAULT);
            if let Some(c) = chunk(r.page, false) {
                let _ = c.level[idx(r.page)].fetch_update(Ordering::AcqRel, Ordering::Acquire, |v| Some((v + 1).min(ARM_MAX_LEVEL)));
                if r.write {
                    c.wrote[idx(r.page)].store(frame as u32 + 1, Ordering::Release);
                    c.uses[idx(r.page) / 64].store(0, Ordering::Relaxed);
                }
            }
            let _ = crate::hosted::resume_waiter(r);
        }
        C4X[X_SYNC].fetch_add(sync.len() as u64, Ordering::Relaxed);
        for (_, _, _, done) in sync {
            done.store(true, Ordering::Release);
        }
    }
}

/// Releases held `pages` (sorted, at most 16,384): see the section's head.
/// A held page's mirror copy is whole and current (it was valid and armed
/// when held, and no CPU write reached it since), so the release copies whole
/// pages: on the GPU into the page's import window while the page is still
/// no-access (GPU work reaches it there), then the page made read-write; a
/// page without a valid window goes through host memory and the CPU, with
/// every other thread suspended meanwhile. No mapping change meanwhile
/// (`mem::GUEST_READERS`, `resident::arm_guard`): a page committed again
/// would take the old words.
fn c4_release_pages(pages: &[u64]) -> Result<(), String> {
    let _readers = crate::mem::GUEST_READERS.read_recursive();
    let _arming = super::resident::arm_guard();
    let c = ctx();
    let mut ranges: Vec<(u64, u64)> = Vec::new();
    for &p in pages {
        match ranges.last_mut() {
            Some(l) if l.0 + l.1 == p => l.1 += PAGE,
            _ => ranges.push((p, PAGE)),
        }
    }
    // No batch that writes the pages in flight, none pushed meanwhile. A
    // writer is waited for 50 ms at most: one can depend on work of the
    // very thread that waits for this release (a label another queue's
    // unsubmitted batch writes: a 2 s stall at a load, run -013954). Past
    // that the release goes on beside it: exact unless that writer's
    // write-back stores a dword of the page between this copy's read and
    // its store of it (a microsecond window on the other queue; counted).
    let gate = loop {
        let g = super::exec::NOTE_GATE.write();
        if let Some(seq) = super::exec::inflight_latest(&ranges) {
            drop(g);
            let t = std::time::Instant::now();
            C4X[X_WAITS].fetch_add(1, Ordering::Relaxed);
            if !super::exec::wait_done_for(seq, std::time::Duration::from_millis(50)) {
                if C4X[X_WAIT_TIMEOUTS].fetch_add(1, Ordering::Relaxed) < 8 {
                    crate::warn!("fe2: C4: batch {seq} writing held pages from {:#x} not done within 50 ms: released beside it", pages[0]);
                }
                C4X[X_WAIT_MS].fetch_add(t.elapsed().as_millis() as u64, Ordering::Relaxed);
                break super::exec::NOTE_GATE.write();
            }
            C4X[X_WAIT_MS].fetch_add(t.elapsed().as_millis() as u64, Ordering::Relaxed);
            continue;
        }
        break g;
    };
    let t0 = std::time::Instant::now();
    // (C5: no other copy of the held map to the GPU until the host's bits
    // of these pages are clear; this one-off clears their GPU bits first.)
    let c5_guard = c5_wanted().then(|| C5_LOCK.lock());
    // (The windows stay listed, none freed, until the release ends.)
    let spans = super::exec::spans_guard();
    let wins: Vec<Option<(vk::Buffer, u64)>> = pages.iter().map(|&p| spans.window(p)).collect();
    let cpu: Vec<usize> = (0..pages.len()).filter(|&i| wins[i].is_none()).collect();
    let maps: Vec<vk::Buffer> = [false, true].iter().filter_map(|&on_c| super::resident::marks_for(on_c)).collect();
    let (mbuf, _) = super::resident::mirror_buffer().ok_or("no mirror")?;
    let staged = if cpu.is_empty() { None } else { Some(c.host_buffer(cpu.len() as u64 * PAGE, vk::BufferUsageFlags::TRANSFER_DST)?) };
    let r = super::resident::one_off_now(|cmd| unsafe {
        let mb = vk::MemoryBarrier::default().src_access_mask(vk::AccessFlags::MEMORY_WRITE).dst_access_mask(vk::AccessFlags::TRANSFER_READ | vk::AccessFlags::TRANSFER_WRITE);
        c.device.cmd_pipeline_barrier(cmd, vk::PipelineStageFlags::ALL_COMMANDS, vk::PipelineStageFlags::TRANSFER, vk::DependencyFlags::empty(), &[mb], &[], &[]);
        // (C5: the GPU marks stores into these pages again from here, so a
        // writer released beside, past its 50 ms, marks its later stores;
        // its earlier ones are in the mirror the copies below read.)
        if c5_guard.is_some() {
            c5_record_clearing(cmd, Some(pages), true);
            let mb = vk::MemoryBarrier::default().src_access_mask(vk::AccessFlags::TRANSFER_WRITE).dst_access_mask(vk::AccessFlags::TRANSFER_READ | vk::AccessFlags::TRANSFER_WRITE);
            c.device.cmd_pipeline_barrier(cmd, vk::PipelineStageFlags::TRANSFER, vk::PipelineStageFlags::TRANSFER, vk::DependencyFlags::empty(), &[mb], &[], &[]);
        }
        // (Whole pages into their windows, a copy per window.)
        let mut by_buf: Vec<(vk::Buffer, Vec<vk::BufferCopy>)> = Vec::new();
        for (i, &p) in pages.iter().enumerate() {
            let Some((buf, off)) = wins[i] else { continue };
            let region = vk::BufferCopy { src_offset: p - MIRROR_BASE, dst_offset: off, size: PAGE };
            match by_buf.iter_mut().find(|x| x.0 == buf) {
                Some(x) => match x.1.last_mut() {
                    Some(l) if l.src_offset + l.size == region.src_offset && l.dst_offset + l.size == region.dst_offset => l.size += PAGE,
                    _ => x.1.push(region),
                },
                None => by_buf.push((buf, vec![region])),
            }
        }
        for (buf, regions) in &by_buf {
            c.device.cmd_copy_buffer(cmd, mbuf, *buf, regions);
        }
        // (Guest memory's copies of the pages change: their zero-known bits cleared.)
        for &p in pages.iter() {
            super::zk::clear(cmd, p, PAGE);
        }
        if let Some((sbuf, _, _)) = &staged {
            let regions: Vec<vk::BufferCopy> = cpu.iter().enumerate().map(|(j, &i)| vk::BufferCopy { src_offset: pages[i] - MIRROR_BASE, dst_offset: j as u64 * PAGE, size: PAGE }).collect();
            c.device.cmd_copy_buffer(cmd, mbuf, *sbuf, &regions);
        }
        // (The pages' mask words cleared: their dwords are stored.)
        for &m in &maps {
            for &p in pages {
                c.device.cmd_fill_buffer(cmd, m, (p - MIRROR_BASE) / PAGE * 128, 128, 0);
            }
        }
        let mb = vk::MemoryBarrier::default().src_access_mask(vk::AccessFlags::TRANSFER_WRITE).dst_access_mask(vk::AccessFlags::HOST_READ | vk::AccessFlags::MEMORY_READ | vk::AccessFlags::MEMORY_WRITE);
        c.device.cmd_pipeline_barrier(cmd, vk::PipelineStageFlags::TRANSFER, vk::PipelineStageFlags::ALL_COMMANDS | vk::PipelineStageFlags::HOST, vk::DependencyFlags::empty(), &[mb], &[], &[]);
    });
    if let Err(e) = r {
        if let Some((sbuf, smem, _)) = staged {
            unsafe {
                c.device.destroy_buffer(sbuf, None);
                c.device.free_memory(smem, None);
            }
        }
        drop(spans);
        drop(gate);
        return Err(format!("the take: {e} (the pages stay held)"));
    }
    drop(spans);
    C4X[X_TAKE_US].fetch_add(t0.elapsed().as_micros() as u64, Ordering::Relaxed);
    // From here the write-back stores these pages again.
    for &p in pages {
        held_mark(p, false);
    }
    drop(c5_guard);
    let chunks: Vec<Option<&'static Chunk>> = pages.iter().map(|&p| chunk(p, false)).collect();
    // Front end 2's arming of each page ended (a release, not a fault: its
    // level stays).
    let unarm = |i: usize, p: u64| {
        if let Some(c) = chunks[i] {
            let (k, w) = (idx(p), idx(p) / 64);
            let bit = 1u64 << (k % 64);
            if c.armed[w].fetch_and(!bit, Ordering::AcqRel) & bit != 0 {
                c.cgen.fetch_add(1, Ordering::AcqRel);
            }
        }
    };
    let lock = crate::hosted::arm_lock();
    use crate::platform::os::Win32::System::Memory::{VirtualProtect, PAGE_READWRITE};
    let mut old = 0u32;
    // The pages with a window: their guest copy is current; read-write now.
    for (i, &p) in pages.iter().enumerate() {
        if wins[i].is_some() {
            unsafe { VirtualProtect(p as _, PAGE as usize, PAGE_READWRITE, &mut old) };
            crate::hosted::released_bits(p, PAGE);
            unarm(i, p);
        }
    }
    // The others through host memory, every other thread suspended.
    if let Some((sbuf, smem, sptr)) = staged {
        let mut handles: Vec<crate::platform::os::Win32::Foundation::HANDLE> = Vec::with_capacity(4096);
        let mut tids: Vec<u32> = Vec::with_capacity(4096);
        let t1 = std::time::Instant::now();
        suspend_others(&mut handles, &mut tids);
        // SAFETY: every other thread is suspended; the pages are committed
        // (held: no mapping change dropped them).
        unsafe {
            for (j, &i) in cpu.iter().enumerate() {
                let p = pages[i];
                VirtualProtect(p as _, PAGE as usize, PAGE_READWRITE, &mut old);
                std::ptr::copy_nonoverlapping(sptr.add(j * PAGE as usize), p as *mut u8, PAGE as usize);
                crate::hosted::released_bits(p, PAGE);
                unarm(i, p);
            }
        }
        resume_others(&handles);
        C4X[X_SUSPEND_US].fetch_add(t1.elapsed().as_micros() as u64, Ordering::Relaxed);
        C4X[X_CPU_PAGES].fetch_add(cpu.len() as u64, Ordering::Relaxed);
        for h in handles {
            unsafe { crate::platform::os::Win32::Foundation::CloseHandle(h) };
        }
        unsafe {
            c.device.destroy_buffer(sbuf, None);
            c.device.free_memory(smem, None);
        }
    }
    drop(lock);
    drop(gate);
    C4X[X_ROUNDS].fetch_add(1, Ordering::Relaxed);
    C4X[X_RELEASED].fetch_add(pages.len() as u64, Ordering::Relaxed);
    C4X[X_NOW].fetch_sub(pages.len() as u64, Ordering::Relaxed);
    Ok(())
}

/// Front end 2's write-backs without an import (`Fallback`): [batches, pages stored, dwords
/// stored, pages left out (remapped since their placement, or held), pages that could not be made
/// writable, pages left to a later write-back (no staging buffer), batches whose stores waited for
/// the locks on a thread of their own].
pub static UNIMPORTED_WB: [AtomicU64; 7] = [const { AtomicU64::new(0) }; 7];

/// The status line of `UNIMPORTED_WB` ("" with none).
pub fn unimported_line() -> String {
    let v: Vec<u64> = UNIMPORTED_WB.iter().map(|x| x.load(Ordering::Relaxed)).collect();
    if v[0] == 0 && v[5] == 0 {
        return String::new();
    }
    format!("    front end 2's write-backs without an import, stored by the CPU at their batch's completion: {} batches ({} of them later, the locks held), {} pages ({} dwords), {} pages left out (remapped since or held), {} left to a later write-back (no staging), {} failures\n", v[0], v[6], v[1], v[2], v[3], v[5], v[4])
}

/// Pages one staging buffer of `Fallback` holds (with their mask words: 4.1 MiB), and how many
/// such buffers are made at most: made from an executor's first failed import on and kept, since
/// the host heap is at its budget when imports fail, where a new allocation may fail too.
const FB_PART: usize = 1024;
const FB_STAGES_MAX: u64 = 16;
struct Stage {
    buf: vk::Buffer,
    _mem: vk::DeviceMemory,
    ptr: *mut u8,
}
unsafe impl Send for Stage {}
static FB_POOL: Mutex<Vec<Stage>> = Mutex::new(Vec::new());
static FB_MADE: AtomicU64 = AtomicU64::new(0);

fn fb_stage() -> Option<Stage> {
    if let Some(s) = FB_POOL.lock().pop() {
        return Some(s);
    }
    fb_new_stage()
}

/// A staging buffer made now (none from the pool), within `FB_STAGES_MAX`.
fn fb_new_stage() -> Option<Stage> {
    if FB_MADE.fetch_add(1, Ordering::AcqRel) >= FB_STAGES_MAX {
        FB_MADE.fetch_sub(1, Ordering::AcqRel);
        return None;
    }
    match ctx().host_buffer(FB_PART as u64 * (PAGE + 128), vk::BufferUsageFlags::TRANSFER_DST) {
        Ok((buf, mem, ptr)) => Some(Stage { buf, _mem: mem, ptr }),
        Err(e) => {
            FB_MADE.fetch_sub(1, Ordering::AcqRel);
            static N: AtomicU64 = AtomicU64::new(0);
            if N.fetch_add(1, Ordering::Relaxed) < 8 {
                crate::warn!("fe2: no staging for write-backs without an import: {e}");
            }
            None
        }
    }
}

/// Staging buffers made ahead (`FB_AHEAD`) where the host heap's budget is small (a 16 GB PC's
/// graphics share: an import there may fail, and a new buffer then too); called every second.
pub fn fallback_reserve(budget: u64) {
    const FB_AHEAD: u64 = 8;
    static DONE: AtomicBool = AtomicBool::new(false);
    if budget >= 12 << 30 || DONE.swap(true, Ordering::AcqRel) {
        return;
    }
    while FB_MADE.load(Ordering::Acquire) < FB_AHEAD {
        match fb_new_stage() {
            Some(s) => FB_POOL.lock().push(s),
            None => break,
        }
    }
    crate::info!("fe2: {} staging buffers made ahead for write-backs without an import (the host heap's budget {} MiB)", FB_MADE.load(Ordering::Acquire), budget >> 20);
}

/// One staging buffer made now, at an executor's first failed import of a written range in a
/// batch (while there may still be room), kept for `record_fallback`.
pub fn fallback_ready() {
    if FB_MADE.load(Ordering::Acquire) == 0 {
        if let Some(s) = fb_stage() {
            FB_POOL.lock().push(s);
        }
    }
}

/// Written ranges whose fallback has not stored yet: (lo, hi, the fallback's id, its batch's
/// sequence once pushed, else 0); `fallback_over`.
static FB_PENDING: Mutex<Vec<(u64, u64, u64, u64)>> = Mutex::new(Vec::new());
static FB_PENDING_N: AtomicU64 = AtomicU64::new(0);
static FB_ID: AtomicU64 = AtomicU64::new(1);

/// A batch's written ranges whose import failed (`Exec`'s `import_written`): copied out after
/// its last write-back by the batch itself (`record_fallback`) and stored by the CPU at its
/// completion (`store_fallback`). Dropped unstored (a batch never submitted), its buffers go back
/// and its ranges stop being pending.
pub struct Fallback {
    /// Each page copied, with the repage sequence at its range's (latest) placement.
    pages: Vec<(u64, u64)>,
    stages: Vec<Stage>,
    id: u64,
}

impl Fallback {
    pub fn id(&self) -> u64 {
        self.id
    }
}

impl Drop for Fallback {
    fn drop(&mut self) {
        // (Mask words zeroed before reuse: a stage whose next batch never ran, after a device
        // loss, then stores nothing of this one's.)
        for st in &self.stages {
            unsafe { std::ptr::write_bytes(st.ptr.add(FB_PART * PAGE as usize), 0, FB_PART * 128) };
        }
        FB_POOL.lock().extend(self.stages.drain(..));
        let mut g = FB_PENDING.lock();
        g.retain(|e| e.2 != self.id);
        FB_PENDING_N.store(g.len() as u64, Ordering::Release);
    }
}

/// Records into `cmd`, after the batch's last write-back (which kept the mask words of the pages
/// it found no page-table entry for, `rdna::kernels::build_scatter`): the mirror pages of `ranges`
/// (lo, hi, the repage sequence at the placement) and their mask words in the batch's own map
/// (`on_c`) copied into host memory, those mask words cleared, the pages' zero-known bits cleared.
/// Pages front end 4 holds are its own; pages beyond the staging there is keep their marks for a
/// later write-back (counted). None: nothing to copy. (Until 2026-11-23 the completion thread
/// made these copies itself after the batch: it waited 50 ms for the batch it was completing,
/// could deadlock with the note gate or a batch waiting for its CPU effects, and stored after
/// the batch's labels: the review of 626d6cc7.)
pub fn record_fallback(cmd: vk::CommandBuffer, on_c: bool, ranges: &[(u64, u64, u64)]) -> Option<Fallback> {
    let mut pages: Vec<(u64, u64)> = Vec::new();
    for &(lo, hi, since) in ranges {
        let mut p = lo.max(MIRROR_BASE) & !(PAGE - 1);
        let e = hi.min(MIRROR_END);
        while p < e {
            pages.push((p, since));
            p += PAGE;
        }
    }
    // (A page in two ranges takes the later placement's sequence: GPU writes after a remap are
    // the new mapping's.)
    pages.sort_unstable_by(|a, b| a.0.cmp(&b.0).then(b.1.cmp(&a.1)));
    pages.dedup_by_key(|x| x.0);
    let before = pages.len();
    pages.retain(|&(p, _)| !c4_held(p));
    UNIMPORTED_WB[3].fetch_add((before - pages.len()) as u64, Ordering::Relaxed);
    let marks = super::resident::marks_for(on_c)?;
    let (mbuf, _) = super::resident::mirror_buffer()?;
    let mut stages = Vec::new();
    while stages.len() * FB_PART < pages.len() {
        match fb_stage() {
            Some(s) => stages.push(s),
            None => break,
        }
    }
    let kept = (stages.len() * FB_PART).min(pages.len());
    UNIMPORTED_WB[5].fetch_add((pages.len() - kept) as u64, Ordering::Relaxed);
    pages.truncate(kept);
    if pages.is_empty() {
        FB_POOL.lock().extend(stages);
        return None;
    }
    let c = ctx();
    unsafe {
        let mb = vk::MemoryBarrier::default().src_access_mask(vk::AccessFlags::MEMORY_WRITE).dst_access_mask(vk::AccessFlags::TRANSFER_READ | vk::AccessFlags::TRANSFER_WRITE);
        c.device.cmd_pipeline_barrier(cmd, vk::PipelineStageFlags::ALL_COMMANDS, vk::PipelineStageFlags::TRANSFER, vk::DependencyFlags::empty(), &[mb], &[], &[]);
        for (si, st) in stages.iter().enumerate() {
            let part = &pages[si * FB_PART..((si + 1) * FB_PART).min(pages.len())];
            let (mut pr, mut mr): (Vec<vk::BufferCopy>, Vec<vk::BufferCopy>) = (Vec::new(), Vec::new());
            for (j, &(p, _)) in part.iter().enumerate() {
                let (src, dst) = (p - MIRROR_BASE, j as u64 * PAGE);
                match pr.last_mut() {
                    Some(r) if r.src_offset + r.size == src && r.dst_offset + r.size == dst => r.size += PAGE,
                    _ => pr.push(vk::BufferCopy { src_offset: src, dst_offset: dst, size: PAGE }),
                }
                let (src, dst) = ((p - MIRROR_BASE) / PAGE * 128, FB_PART as u64 * PAGE + j as u64 * 128);
                match mr.last_mut() {
                    Some(r) if r.src_offset + r.size == src && r.dst_offset + r.size == dst => r.size += 128,
                    _ => mr.push(vk::BufferCopy { src_offset: src, dst_offset: dst, size: 128 }),
                }
            }
            c.device.cmd_copy_buffer(cmd, mbuf, st.buf, &pr);
            c.device.cmd_copy_buffer(cmd, marks, st.buf, &mr);
        }
        let mb = vk::MemoryBarrier::default().src_access_mask(vk::AccessFlags::TRANSFER_READ | vk::AccessFlags::TRANSFER_WRITE).dst_access_mask(vk::AccessFlags::TRANSFER_WRITE);
        c.device.cmd_pipeline_barrier(cmd, vk::PipelineStageFlags::TRANSFER, vk::PipelineStageFlags::TRANSFER, vk::DependencyFlags::empty(), &[mb], &[], &[]);
        // (Their mask words cleared: the CPU stores those dwords. Their zero-known bits cleared:
        // guest memory's copies of the pages change.)
        let mut k = 0;
        while k < pages.len() {
            let mut n = 1;
            while k + n < pages.len() && pages[k + n].0 == pages[k].0 + n as u64 * PAGE {
                n += 1;
            }
            let p = pages[k].0;
            c.device.cmd_fill_buffer(cmd, marks, (p - MIRROR_BASE) / PAGE * 128, n as u64 * 128, 0);
            super::zk::clear(cmd, p, n as u64 * PAGE);
            k += n;
        }
        let mb = vk::MemoryBarrier::default().src_access_mask(vk::AccessFlags::TRANSFER_WRITE).dst_access_mask(vk::AccessFlags::HOST_READ | vk::AccessFlags::MEMORY_READ | vk::AccessFlags::MEMORY_WRITE);
        c.device.cmd_pipeline_barrier(cmd, vk::PipelineStageFlags::TRANSFER, vk::PipelineStageFlags::ALL_COMMANDS | vk::PipelineStageFlags::HOST, vk::DependencyFlags::empty(), &[mb], &[], &[]);
    }
    let id = FB_ID.fetch_add(1, Ordering::Relaxed);
    let mut g = FB_PENDING.lock();
    g.extend(ranges.iter().map(|&(lo, hi, _)| (lo, hi, id, 0)));
    FB_PENDING_N.store(g.len() as u64, Ordering::Release);
    drop(g);
    Some(Fallback { pages, stages, id })
}

/// The batch holding fallback `id` was pushed as `seq`: later placements over its ranges wait
/// for its stores (`fallback_over`).
pub fn fallback_pushed(id: u64, seq: u64) {
    for e in FB_PENDING.lock().iter_mut().filter(|e| e.2 == id) {
        e.3 = seq;
    }
}

/// The newest pushed batch whose fallback over [lo, hi) has not stored yet (0: none). A batch
/// that places the range waits for those stores (`Exec::cpu_wait`): its write-back through an
/// import made since, or its prologue's uploads of the pages, would otherwise come before them.
#[inline]
pub fn fallback_over(lo: u64, hi: u64) -> u64 {
    if FB_PENDING_N.load(Ordering::Acquire) == 0 {
        return 0;
    }
    FB_PENDING.lock().iter().filter(|e| e.3 != 0 && e.0 < hi && lo < e.1).map(|e| e.3).max().unwrap_or(0)
}

/// The newest pushed batch whose fallback has not stored yet, anywhere (0: none): a deferred
/// batch submitted now waits for those stores (`Exec::submit_closed_list`).
pub fn fallback_newest_pending() -> u64 {
    if FB_PENDING_N.load(Ordering::Acquire) == 0 {
        return 0;
    }
    FB_PENDING.lock().iter().map(|e| e.3).max().unwrap_or(0)
}

/// Whether a fallback of a batch up to `seq` has not stored yet (`Exec::depend_on`: what follows a
/// parallel run's batches then follows their CPU stores).
pub fn fallback_pending_upto(seq: u64) -> bool {
    FB_PENDING_N.load(Ordering::Acquire) != 0 && FB_PENDING.lock().iter().any(|e| e.3 != 0 && e.3 <= seq)
}

/// Front end 2's arming of page `p` ended (a release, as `c4_release_pages`', not a fault).
fn unarm_page(p: u64) {
    if let Some(c) = chunk(p, false) {
        let (k, w) = (idx(p), idx(p) / 64);
        let bit = 1u64 << (k % 64);
        if c.armed[w].fetch_and(!bit, Ordering::AcqRel) & bit != 0 {
            c.cgen.fetch_add(1, Ordering::AcqRel);
        }
    }
}

/// The CPU's part of a fallback, at its batch's completion and before the batch's CP effects (the
/// first of its actions: a title that sees a label the batch wrote sees these dwords too). Each
/// page's marked dwords are stored from the staging copy under the guest memory readers' lock (no
/// mapping change meanwhile; a page remapped since its placement, `mem::repaged_since`, is left
/// out). A page the host had armed is left read-write and unarmed: a title write while it was
/// writable reaches front end 2 through its write-watch takes, as after `c4_release_pages`.
/// Another page gets its protection back. Nothing here waits for the GPU or submits work: the
/// completion thread runs it.
pub fn store_fallback(f: Fallback) {
    if super::exec::DEVICE_LOST.load(Ordering::Relaxed) {
        return;
    }
    UNIMPORTED_WB[0].fetch_add(1, Ordering::Relaxed);
    // (The locks taken with a timeout: a mapping change or front end 4's release may hold one
    // while it waits for a GPU copy queued behind a batch that waits for this batch's CPU effects,
    // which would wait for this. Then the stores wait on a thread of their own, after the batch's
    // CP effects: late, which beats a hang. The review of 2026-11-23.)
    // (Behind a late one, in order: `FB_LATE`.)
    if !FB_LATE.lock().0.is_empty() {
        late_store(f);
        return;
    }
    let d = std::time::Duration::from_millis(25);
    let readers = crate::mem::GUEST_READERS.try_read_recursive_for(d);
    let arming = readers.as_ref().and_then(|_| super::resident::try_arm_guard_for(d));
    let lock = arming.as_ref().and_then(|_| crate::hosted::try_arm_lock_for(d));
    match (readers, arming, lock) {
        (Some(_readers), Some(_arming), Some(lock)) => store_pages(&f, lock),
        _ => late_store(f),
    }
    // (`f` dropped once stored: its buffers back to the pool, its ranges no longer pending.)
}

/// Fallbacks whose locks were held at their batch's completion, stored in order by one worker
/// (`late_worker`), and whether that worker runs. Their stores come after the batch's CP effects
/// and after the waits on its CPU effects passed: a writer of the same words meanwhile (a later
/// batch, the title acting on the batch's labels) can be undone by them, a rare residual.
static FB_LATE: Mutex<(std::collections::VecDeque<Fallback>, bool)> = Mutex::new((std::collections::VecDeque::new(), false));
static FB_LATE_CV: parking_lot::Condvar = parking_lot::Condvar::new();

fn late_store(f: Fallback) {
    UNIMPORTED_WB[6].fetch_add(1, Ordering::Relaxed);
    let mut g = FB_LATE.lock();
    g.0.push_back(f);
    if !g.1 {
        match std::thread::Builder::new().name("fe2-late-stores".into()).spawn(late_worker) {
            Ok(_) => g.1 = true,
            Err(e) => {
                // (No thread: the stores are dropped, as before the fallback existed.)
                g.0.clear();
                crate::warn!("fe2: no thread for late write-backs without an import ({e}): their stores dropped");
            }
        }
    }
    FB_LATE_CV.notify_one();
}

fn late_worker() {
    loop {
        // (The front stays queued until stored: `store_fallback` queues behind it meanwhile.)
        let f = {
            let mut g = FB_LATE.lock();
            while g.0.is_empty() {
                FB_LATE_CV.wait(&mut g);
            }
            g.0.pop_front()
        };
        let Some(f) = f else { continue };
        if !super::exec::DEVICE_LOST.load(Ordering::Relaxed) {
            let _readers = crate::mem::GUEST_READERS.read_recursive();
            let _arming = super::resident::arm_guard();
            let lock = crate::hosted::arm_lock();
            store_pages(&f, lock);
        }
    }
}

/// `store_fallback`'s stores, its locks held (`lock`, the host's arm lock, released here).
fn store_pages(f: &Fallback, lock: parking_lot::ReentrantMutexGuard<'static, ()>) {
    use crate::platform::os::Win32::System::Memory::{VirtualProtect, PAGE_READWRITE};
    let (mut n_pg, mut n_dw, mut left, mut failed) = (0u64, 0u64, 0u64, 0u64);
    let hosted = crate::hosted::on();
    for (k, &(p, since)) in f.pages.iter().enumerate() {
        let st = &f.stages[k / FB_PART];
        let j = k % FB_PART;
        let at = unsafe { st.ptr.add(FB_PART * PAGE as usize + j * 128) } as *const u32;
        let mut m = [0u32; 32];
        for (w, mw) in m.iter_mut().enumerate() {
            *mw = unsafe { std::ptr::read_volatile(at.add(w)) };
        }
        if m.iter().all(|&w| w == 0) {
            continue;
        }
        // (Held by front end 4 since the batch's close: its release puts back the mirror's
        // words, these among them.)
        if crate::mem::repaged_since(p, p + PAGE, since) || c4_held(p) {
            left += 1;
            continue;
        }
        let armed = hosted && crate::hosted::armed(p);
        let mut old = 0u32;
        if unsafe { VirtualProtect(p as _, PAGE as usize, PAGE_READWRITE, &mut old) } == 0 {
            failed += 1;
            continue;
        }
        let src = unsafe { st.ptr.add(j * PAGE as usize) } as *const u32;
        for (w, &mw) in m.iter().enumerate() {
            let mut mm = mw;
            while mm != 0 {
                let d = w * 32 + mm.trailing_zeros() as usize;
                unsafe { std::ptr::write_volatile((p + 4 * d as u64) as *mut u32, std::ptr::read_volatile(src.add(d))) };
                n_dw += 1;
                mm &= mm - 1;
            }
        }
        if armed {
            crate::hosted::released_bits(p, PAGE);
            unarm_page(p);
        } else if old != PAGE_READWRITE {
            let mut tmp = 0u32;
            unsafe { VirtualProtect(p as _, PAGE as usize, old, &mut tmp) };
        }
        n_pg += 1;
    }
    drop(lock);
    UNIMPORTED_WB[1].fetch_add(n_pg, Ordering::Relaxed);
    UNIMPORTED_WB[2].fetch_add(n_dw, Ordering::Relaxed);
    UNIMPORTED_WB[3].fetch_add(left, Ordering::Relaxed);
    if failed != 0 && UNIMPORTED_WB[4].fetch_add(failed, Ordering::Relaxed) < 8 {
        crate::warn!("fe2: {failed} pages of write-backs without an import could not be made writable (their dwords not stored)");
    }
}

/// Suspends every other thread of the process (handles and thread IDs into
/// the caller's vectors, whose capacity must suffice: nothing allocates
/// once a thread is suspended), snapshots taken until none is new.
fn suspend_others(handles: &mut Vec<crate::platform::os::Win32::Foundation::HANDLE>, tids: &mut Vec<u32>) {
    use crate::platform::os::Win32::System::Diagnostics::ToolHelp::{CreateToolhelp32Snapshot, Thread32First, Thread32Next, TH32CS_SNAPTHREAD, THREADENTRY32};
    use crate::platform::os::Win32::System::Threading::{GetCurrentProcessId, GetCurrentThreadId, OpenThread, SuspendThread, THREAD_GET_CONTEXT, THREAD_SUSPEND_RESUME};
    let (pid, me) = unsafe { (GetCurrentProcessId(), GetCurrentThreadId()) };
    for _ in 0..4 {
        let snap = unsafe { CreateToolhelp32Snapshot(TH32CS_SNAPTHREAD, 0) };
        if snap == crate::platform::os::Win32::Foundation::INVALID_HANDLE_VALUE {
            return;
        }
        let mut new = 0;
        let mut te: THREADENTRY32 = unsafe { std::mem::zeroed() };
        te.dwSize = std::mem::size_of::<THREADENTRY32>() as u32;
        let mut ok = unsafe { Thread32First(snap, &mut te) } != 0;
        while ok {
            if te.th32OwnerProcessID == pid && te.th32ThreadID != me && !tids.contains(&te.th32ThreadID) && tids.len() < tids.capacity() {
                let h = unsafe { OpenThread(THREAD_SUSPEND_RESUME | THREAD_GET_CONTEXT, 0, te.th32ThreadID) };
                if !h.is_null() {
                    if unsafe { SuspendThread(h) } != u32::MAX {
                        handles.push(h);
                        tids.push(te.th32ThreadID);
                        new += 1;
                    } else {
                        unsafe { crate::platform::os::Win32::Foundation::CloseHandle(h) };
                    }
                }
            }
            ok = unsafe { Thread32Next(snap, &mut te) } != 0;
        }
        unsafe { crate::platform::os::Win32::Foundation::CloseHandle(snap) };
        if new == 0 {
            break;
        }
    }
    // (A suspension takes effect asynchronously: a context read waits for it.)
    #[repr(C, align(16))]
    struct Ctx16(crate::platform::os::Win32::System::Diagnostics::Debug::CONTEXT);
    for &h in handles.iter() {
        let mut ctx: Ctx16 = unsafe { std::mem::zeroed() };
        // (CONTEXT_CONTROL for AMD64.)
        ctx.0.ContextFlags = 0x0010_0001;
        unsafe { crate::platform::os::Win32::System::Diagnostics::Debug::GetThreadContext(h, &mut ctx.0) };
    }
}

fn resume_others(handles: &[crate::platform::os::Win32::Foundation::HANDLE]) {
    for &h in handles {
        unsafe { crate::platform::os::Win32::System::Threading::ResumeThread(h) };
    }
}

/// Control `c4 off|on` (`DS_C4`, in-run A/Bs): off stops holding and
/// releases every held page.
pub fn c4_control(what: &str) -> Result<String, String> {
    if !c4_on() {
        return Err("c4: not on (front end 2 hosted, `DS_C4` not 0)".into());
    }
    match what {
        "off" => {
            C4_OFF.store(true, Ordering::Release);
            C4_ALL.store(true, Ordering::Release);
            Ok("holding off, every held page released".into())
        }
        "on" => {
            C4_OFF.store(false, Ordering::Release);
            Ok("holding on".into())
        }
        _ => Err("usage: c4 off|on".into()),
    }
}

/// Status line of C4 (since the last one, per flip where it says so).
fn c4_line() -> String {
    if !c4_on() || HELD_MAP.get().is_none() {
        return String::new();
    }
    static LAST: parking_lot::Mutex<(u64, [u64; 28], u64, u64, u64)> = parking_lot::Mutex::new((0, [0; 28], 0, 0, 0));
    let frame = super::FRAMES.load(Ordering::Relaxed);
    let now: [u64; 28] = std::array::from_fn(|k| C4X[k].load(Ordering::Relaxed));
    let (dw, pg, all) = (WB64[2].load(Ordering::Relaxed), WB64[3].load(Ordering::Relaxed), WB64[1].load(Ordering::Relaxed));
    let mut l = LAST.lock();
    let flips = frame.saturating_sub(l.0).max(1) as f64;
    let d = |k: usize| now[k].saturating_sub(l.1[k]);
    let (ddw, dpg, dall) = (dw.saturating_sub(l.2), pg.saturating_sub(l.3), all.saturating_sub(l.4));
    let line = format!(
        "    front end 4's held pages (C4{}{}, over {flips:.0} flips): {} held now ({} held in {} runs, {} runs failed, {} refused by the host, {} candidates seen); the write-back left out {:.3} M dwords and {:.0} pages a flip ({:.1}% of the dwords it would have stored); releases on faults: the title {} reads / {} writes, ours {} / {}, other code {} / {}; synchronous releases {} ({} late), pages released {} in {} rounds ({:.0} us of take a round; {} through host memory, {:.0} us suspended), plain {}; writer waits {} ({} ms, {} past 50 ms); dropped by mapping changes {}; uploads that left out held pages {}; failed takes {}; pages never held {}\n",
        if C4_OFF.load(Ordering::Relaxed) { ", off" } else { "" },
        if !c5_wanted() {
            String::new()
        } else {
            format!(
                "; C5{}: {} copies of the held map to the GPU, {} words, {:.1} ms",
                if C5_OFF.load(Ordering::Relaxed) { " off" } else { "" },
                C5X[0].load(Ordering::Relaxed),
                C5X[1].load(Ordering::Relaxed),
                C5X[2].load(Ordering::Relaxed) as f64 / 1000.0
            )
        },
        now[X_NOW],
        d(X_HELD),
        d(X_RUNS),
        d(X_FAILED),
        d(X_UNDONE),
        d(X_CANDIDATES),
        ddw as f64 / 1e6 / flips,
        dpg as f64 / flips,
        100.0 * ddw as f64 / (ddw + dall).max(1) as f64,
        d(X_FAULTS),
        d(X_FAULTS + 1),
        d(X_FAULTS + 2),
        d(X_FAULTS + 3),
        d(X_FAULTS + 4),
        d(X_FAULTS + 5),
        d(X_SYNC),
        d(X_SYNC_LATE),
        d(X_RELEASED),
        d(X_ROUNDS),
        d(X_TAKE_US) as f64 / d(X_ROUNDS).max(1) as f64,
        d(X_CPU_PAGES),
        d(X_SUSPEND_US) as f64,
        d(X_PLAIN),
        d(X_WAITS),
        d(X_WAIT_MS),
        d(X_WAIT_TIMEOUTS),
        d(X_DROPPED),
        d(X_UP_SKIPPED),
        d(X_FAILED_TAKES),
        NO_HOLD_PAGES.load(Ordering::Relaxed)
    );
    *l = (frame, now, dw, pg, all);
    drop(l);
    let line = if c4_arm_wanted() {
        let na: [u64; 8] = std::array::from_fn(|k| C4_NOT_ARMED[k].load(Ordering::Relaxed));
        line + &format!(
            "    C4's own arming{} (since boot): {} pages offered, {} armed; not armed: armed already {}, too few clean epochs {}, written lately {}, chunk held {}, no settled window {}, not plain read-write {}, import list busy {}\n",
            if C4_ARM_OFF.load(Ordering::Relaxed) { " (off)" } else { "" },
            now[X_ARM_TRIED],
            now[X_ARM_DONE],
            na[R_ARMED],
            na[R_USES],
            na[R_QUIET],
            na[R_HELD],
            na[R_WINDOW],
            na[R_RW],
            na[R_BUSY]
        )
    } else {
        line
    };
    // Pages never held by owning allocation and why.
    let names: std::collections::HashMap<u8, String> = crate::engine::allocators().into_iter().map(|a| (a.id, a.name)).collect();
    let mut by: Vec<(usize, [u64; 3])> = (0..256).map(|i| (i, std::array::from_fn(|k| NO_HOLD_BY[i][k].load(Ordering::Relaxed)))).filter(|x: &(usize, [u64; 3])| x.1.iter().sum::<u64>() > 0).collect();
    by.sort_by_key(|x| std::cmp::Reverse(x.1.iter().sum::<u64>()));
    let line = if by.is_empty() {
        line
    } else {
        let parts: Vec<String> = by.iter().take(8).map(|(i, v)| format!("{} {} (our reads {}, targets {}, faults {})", names.get(&(*i as u8)).cloned().unwrap_or_else(|| format!("owner {i}")), v.iter().sum::<u64>(), v[0], v[1], v[2])).collect();
        line + &format!("    front end 4's pages never held by engine allocator: {}\n", parts.join(", "))
    };
    let line = match C4_WHY.lock().as_ref() {
        Some(m) if !m.is_empty() => {
            let mut v: Vec<(&(u64, u64, u8), &[u64; 7])> = m.iter().collect();
            v.sort_by_key(|x| std::cmp::Reverse(x.1[0]));
            let parts: Vec<String> = v
                .iter()
                .take(8)
                .map(|(k, c)| format!("{:#x}+{:#x} ({}): {} (never held {}, unarmed {}, invalid {}, host-unarmed {}, CPU-written lately {}, candidates {})", k.0, k.1, names.get(&k.2).cloned().unwrap_or_else(|| format!("owner {}", k.2)), c[0], c[1], c[2], c[3], c[4], c[5], c[6]))
                .collect();
            line + &format!("    C4's targets' pages written back lately and not held, by record (`DS_C4_WHY`, page-rounds): {}\n", parts.join("; "))
        }
        _ => line,
    };
    let rounds = WB_ROUNDS.load(Ordering::Relaxed);
    let mut wb: Vec<(usize, u64)> = (0..256).map(|i| (i, WB_BY_OWNER[i].load(Ordering::Relaxed))).filter(|x| x.1 > 0).collect();
    wb.sort_by_key(|x| std::cmp::Reverse(x.1));
    let line = if wb.is_empty() || rounds == 0 {
        line
    } else {
        let parts: Vec<String> = wb.iter().take(8).map(|&(i, n)| format!("{} {:.0}", names.get(&(i as u8)).cloned().unwrap_or_else(|| format!("owner {i}")), n as f64 / rounds as f64)).collect();
        line + &format!("    pages written back lately (the last two hold periods, a round's mean over {rounds}) by engine allocator: {}\n", parts.join(", "))
    };
    let mut rips: Vec<((u64, bool), u64)> = HOLD_RIPS.lock().take().map(|m| m.into_iter().collect()).unwrap_or_default();
    rips.sort_by(|a, b| b.1.cmp(&a.1));
    if rips.is_empty() {
        return line;
    }
    let top: Vec<String> = rips.iter().take(8).map(|&((rip, w), n)| format!("{n} {} at {}", if w { "writes" } else { "reads" }, describe_code(rip))).collect();
    line + &format!("    front end 4's held pages: releases by code: {} ({} places)\n", top.join(", "), rips.len())
}
