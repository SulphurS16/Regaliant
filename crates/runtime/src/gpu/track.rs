//! Write tracking of guest memory regions (texture cache invalidation).
//! Regions are marked dirty by announced writes: file reads and driver
//! imports (`host_write`), GPU work (`gpu_write`) and mapping changes
//! (`remapped`).
//!
//! CPU writes by guest code can additionally be caught by page protection
//! (`DS_TRACK=1`, experimental): a watched region's pages are made read-only,
//! the first CPU write faults, the exception handler marks the regions on the
//! page dirty and restores its protection, and the write is retried. It is off
//! by default: changing the protection of guest pages that are imported into
//! the GPU (VK_EXT_external_memory_host) led to device losses on the RTX 4090
//! (3 of 5 runs with it, 0 of 5 without), so consumers detect CPU writes by
//! sampled hashing instead (`protected()` tells which applies).
//!
//! Protocol for a consumer: `take_dirty(id)`; if dirty, `rearm(id)` and then
//! read the memory (writes after re-arming fault again). `take_dirty` names
//! the parts written when they are known (file reads, mapping changes, CPU
//! write faults), so a consumer may read only those.

use parking_lot::Mutex;
use std::collections::{BTreeMap, HashMap};
use std::sync::atomic::{AtomicU64, Ordering};
use crate::platform::os::Win32::System::Memory::{VirtualProtect, PAGE_EXECUTE_READ, PAGE_EXECUTE_READWRITE, PAGE_READONLY, PAGE_READWRITE};

const PAGE: u64 = 4096;

struct Region {
    base: u64,
    size: u64,
    /// Moves whenever the region turns dirty or GPU-written (`gen_of`): a
    /// consumer's check of it holds while it stays (the image cache's view
    /// memos).
    generation: &'static std::sync::atomic::AtomicU32,
    /// Written by the CPU (or mapping changes) since armed.
    dirty: bool,
    /// Written by GPU work since last checked (contents may be unchanged).
    gpu: bool,
    /// What made it dirty last (diagnostics).
    why: Why,
    /// The ranges written since it was armed, unless `whole` (unknown parts,
    /// or more than `MAX_PARTS`).
    parts: Vec<(u64, u64)>,
    whole: bool,
}

/// Written ranges kept per region before it counts as written all over.
const MAX_PARTS: usize = 64;

impl Region {
    /// Marks the region dirty: `part` written (None: unknown parts).
    fn mark(&mut self, part: Option<(u64, u64)>, why: Why) {
        self.generation.fetch_add(1, Ordering::Release);
        self.dirty = true;
        self.why = why;
        match part {
            Some(p) if !self.whole && self.parts.len() < MAX_PARTS => self.parts.push(p),
            _ => {
                self.whole = true;
                self.parts.clear();
            }
        }
    }
}

#[derive(Clone, Copy, Debug)]
pub enum Why {
    None,
    Fault(u64),
    HostWrite(u64),
    GpuWrite(u64, u64),
    /// A mapping change of [base, base+len): `REMAP_COMMIT`, `REMAP_DECOMMIT`
    /// or `REMAP_PROTECT`.
    Remap(u64, u64, u8),
    Unmapped,
    /// Requested (diagnosis: `reupload`).
    Forced,
    /// Sampled bytes changed (texture scrubbing: CPU writes).
    Sampled,
    /// CPU writes reported by write-watch marks (`cpu_pages`).
    Written(u64),
}

#[derive(Default)]
struct Tracker {
    regions: Vec<Option<Region>>,
    /// Region ids by base address (regions are never removed) and the
    /// largest region size: overlap queries look at the bases in (start -
    /// max_size, end) only. Scanning every region took ~0.8 us per GPU
    /// write, two per dispatch (the per-item culls, ~4,300 per frame).
    by_base: BTreeMap<u64, Vec<u32>>,
    max_size: u64,
    /// Region ids by 1 MiB bucket of the guest memory they cover (every
    /// bucket each region touches): overlap queries of ranges under 1 GiB
    /// look at the buckets of the queried range only. (Since 2026-11-08: the
    /// bases from start - max_size alone, with ~350 MiB textures tracked,
    /// looked at ~18 ids a query under this lock the recording threads share;
    /// the buckets ~0.1, the vista x1.006 and its stalled runs of the graphics
    /// workers halved, FINDINGS.md.)
    buckets: HashMap<u64, Vec<u32>>,
    /// Overlap queries and the region ids they looked at, by path (0: the
    /// bases, ranges of 1 GiB or more; 1: buckets), since boot (`status`).
    queries: [u64; 2],
    looked: [u64; 2],
    /// Watched page -> (regions on it, protection to restore, protected now).
    pages: HashMap<u64, (Vec<u32>, u32, bool)>,
    /// Ranges being written by code that cannot fault (kernel reads, driver
    /// imports): their pages stay writable until the writer is done.
    busy: Vec<(u64, u64, u64)>,
    next_busy: u64,
}

static TRACK: Mutex<Option<Tracker>> = Mutex::new(None);
pub static FAULTS: AtomicU64 = AtomicU64::new(0);
/// Changes whenever a region becomes dirty or GPU-written: consumers may
/// keep results that checked the regions while it stays the same.
pub static GEN: AtomicU64 = AtomicU64::new(0);

/// GEN changes by cause: [forced dirty, page not writable, CPU write fault,
/// host write, GPU write, remap] (profile line `image memos`).
pub static GEN_WHY: [AtomicU64; 6] = [const { AtomicU64::new(0) }; 6];

/// Changes whenever a region is added, its GPU-written flag is reset, or a
/// mapping change hits regions: while it stays the same, a `gpu_write` of a
/// range noted before changes nothing (`Exec`'s write-note memo).
pub static RESET_GEN: AtomicU64 = AtomicU64::new(0);

fn changed(why: usize) {
    GEN.fetch_add(1, Ordering::Release);
    GEN_WHY[why].fetch_add(1, Ordering::Relaxed);
}

fn read_only(prot: u32) -> u32 {
    match prot {
        PAGE_READWRITE => PAGE_READONLY,
        PAGE_EXECUTE_READWRITE => PAGE_EXECUTE_READ,
        p => p,
    }
}

/// Regions per 64 KiB and per 4 MiB of guest memory (hashed; regions are
/// never removed): GPU writes where no region lies skip the lock
/// (`gpu_write` ran for every written binding of every command and its lock
/// was contended between the recording threads).
const FILTER: usize = 1 << 16;
static REGION_PAGES: [AtomicU64; FILTER] = [const { AtomicU64::new(0) }; FILTER];
static REGION_COARSE: [AtomicU64; 1 << 12] = [const { AtomicU64::new(0) }; 1 << 12];

fn filter_bucket(page: u64, bits: u32) -> usize {
    (page.wrapping_mul(0x9E37_79B9_7F4A_7C15) >> (64 - bits)) as usize
}

fn note_region(base: u64, size: u64) {
    let size = size.max(1);
    for page in (base >> 16)..=((base + size - 1) >> 16) {
        REGION_PAGES[filter_bucket(page, 16)].fetch_add(1, Ordering::AcqRel);
    }
    for page in (base >> 22)..=((base + size - 1) >> 22) {
        REGION_COARSE[filter_bucket(page, 12)].fetch_add(1, Ordering::AcqRel);
    }
}

/// Live regions per 64 KiB and per 4 MiB of the GPU span [LIVE_LO,
/// LIVE_HI) (front end 2's mirror span), exact: counted when a region is
/// added and uncounted when it is dropped (`unwatch`). The hashed filter
/// above remembers every region ever made, so with a session's textures
/// ~97% of the GPU writes announced at their batch's completion passed it
/// with no region under them (12.3 s of the completion thread's actions in
/// a walk boot against 0.7, -094423, 2026-11-11); it stays for regions
/// outside the span.
const LIVE_LO: u64 = super::resident::MIRROR_BASE;
const LIVE_HI: u64 = super::resident::MIRROR_END;
static LIVE_64K: [std::sync::atomic::AtomicU16; ((LIVE_HI - LIVE_LO) >> 16) as usize] = [const { std::sync::atomic::AtomicU16::new(0) }; ((LIVE_HI - LIVE_LO) >> 16) as usize];
static LIVE_4M: [std::sync::atomic::AtomicU32; ((LIVE_HI - LIVE_LO) >> 22) as usize] = [const { std::sync::atomic::AtomicU32::new(0) }; ((LIVE_HI - LIVE_LO) >> 22) as usize];

/// Counts a region in (`add`) or out of the live counts (its part in the span).
fn live_count(base: u64, size: u64, add: bool) {
    let (lo, hi) = (base.max(LIVE_LO), base.saturating_add(size.max(1)).min(LIVE_HI));
    if lo >= hi {
        return;
    }
    for p in ((lo - LIVE_LO) >> 16)..=((hi - 1 - LIVE_LO) >> 16) {
        if add {
            LIVE_64K[p as usize].fetch_add(1, Ordering::AcqRel);
        } else {
            LIVE_64K[p as usize].fetch_sub(1, Ordering::AcqRel);
        }
    }
    for c in ((lo - LIVE_LO) >> 22)..=((hi - 1 - LIVE_LO) >> 22) {
        if add {
            LIVE_4M[c as usize].fetch_add(1, Ordering::AcqRel);
        } else {
            LIVE_4M[c as usize].fetch_sub(1, Ordering::AcqRel);
        }
    }
}

/// Whether a region may overlap [base, base+len) (false: certainly none):
/// exact inside the GPU span, the hashed filter outside it.
pub(super) fn may_have_region(base: u64, len: u64) -> bool {
    let len = len.max(1);
    let end = base.saturating_add(len);
    let (lo, hi) = (base.max(LIVE_LO), end.min(LIVE_HI));
    if lo < hi {
        for c in ((lo - LIVE_LO) >> 22)..=((hi - 1 - LIVE_LO) >> 22) {
            if LIVE_4M[c as usize].load(Ordering::Acquire) == 0 {
                continue;
            }
            let (clo, chi) = (lo.max(LIVE_LO + (c << 22)), hi.min(LIVE_LO + ((c + 1) << 22)));
            if (((clo - LIVE_LO) >> 16)..=((chi - 1 - LIVE_LO) >> 16)).any(|p| LIVE_64K[p as usize].load(Ordering::Acquire) != 0) {
                return true;
            }
        }
    }
    (base < LIVE_LO && hashed_may_have(base, end.min(LIVE_LO) - base)) || (end > LIVE_HI && hashed_may_have(base.max(LIVE_HI), end - base.max(LIVE_HI)))
}

/// The hashed filter's answer for [base, base+len).
fn hashed_may_have(base: u64, len: u64) -> bool {
    let len = len.max(1);
    let (first, last) = (base >> 16, (base + len - 1) >> 16);
    if last - first < 64 {
        return (first..=last).any(|p| REGION_PAGES[filter_bucket(p, 16)].load(Ordering::Acquire) != 0);
    }
    let (first, last) = (base >> 22, (base + len - 1) >> 22);
    last - first >= 64 || (first..=last).any(|p| REGION_COARSE[filter_bucket(p, 12)].load(Ordering::Acquire) != 0)
}

impl Tracker {
    fn add(&mut self, base: u64, size: u64) -> u32 {
        note_region(base, size);
        live_count(base, size, true);
        RESET_GEN.fetch_add(1, Ordering::Release);
        let id = self.regions.len() as u32;
        // (Its generation's counter is leaked: 4 bytes per region ever made, read without the lock.)
        let generation = Box::leak(Box::new(std::sync::atomic::AtomicU32::new(0)));
        self.regions.push(Some(Region { base, size, generation, dirty: false, gpu: false, why: Why::None, parts: Vec::new(), whole: false }));
        self.by_base.entry(base).or_default().push(id);
        self.max_size = self.max_size.max(size);
        for b in bucket_span(base, size) {
            self.buckets.entry(b).or_default().push(id);
        }
        id
    }

    /// Calls `f` for every region overlapping [base, base+len).
    fn each_overlapping(&mut self, base: u64, len: u64, mut f: impl FnMut(&mut Region)) {
        self.each_overlapping_id(base, len, |_, r| f(r));
    }

    /// `each_overlapping` with the region ids.
    fn each_overlapping_id(&mut self, base: u64, len: u64, mut f: impl FnMut(u32, &mut Region)) {
        let end = base.saturating_add(len);
        // (Ranges of 1 GiB or more go through the bases: a bucket walk is
        // per MiB of the range, the bases' per region.)
        let mode = (len >> BUCKET_SHIFT < 1024) as usize;
        let Tracker { regions, by_base, max_size, buckets, queries, looked, .. } = self;
        queries[mode] += 1;
        if mode == 1 {
            // A region is visited in the first bucket it shares with the
            // range: once, however many buckets both cover.
            let first = base >> BUCKET_SHIFT;
            for b in bucket_span(base, len) {
                let Some(ids) = buckets.get(&b) else { continue };
                looked[1] += ids.len() as u64;
                for &id in ids {
                    if let Some(Some(r)) = regions.get_mut(id as usize) {
                        if r.base < end && base < r.base + r.size && b == first.max(r.base >> BUCKET_SHIFT) {
                            f(id, r);
                        }
                    }
                }
            }
            return;
        }
        for (_, ids) in by_base.range(base.saturating_sub(*max_size)..end) {
            looked[0] += ids.len() as u64;
            for &id in ids {
                if let Some(Some(r)) = regions.get_mut(id as usize) {
                    if r.base < end && base < r.base + r.size {
                        f(id, r);
                    }
                }
            }
        }
    }
}

/// `Tracker::buckets`' granule: 1 MiB.
const BUCKET_SHIFT: u32 = 20;

/// The buckets [base, base+len) touches (at least one).
fn bucket_span(base: u64, len: u64) -> std::ops::RangeInclusive<u64> {
    (base >> BUCKET_SHIFT)..=(base.saturating_add(len.max(1) - 1) >> BUCKET_SHIFT)
}

fn set_prot(page: u64, n: u64, prot: u32) -> bool {
    let mut old = 0u32;
    unsafe { VirtualProtect(page as _, (n * PAGE) as usize, prot, &mut old) != 0 }
}

/// Writable guest mappings covering [base, base+size): (start, end, Windows
/// protection) runs, or None if any page is unmapped or not writable.
fn spans(base: u64, size: u64) -> Option<Vec<(u64, u64, u32)>> {
    let first = base & !(PAGE - 1);
    let end = (base + size).next_multiple_of(PAGE);
    let mut v = Vec::new();
    let st = crate::mem::MEM.lock();
    let mut a = first;
    while a < end {
        let m = st.find(a)?;
        if m.kind == crate::mem::VmaKind::Reserved {
            return None;
        }
        let e = m.end().min(end);
        let prot = crate::mem::win_prot(m.prot);
        if prot != PAGE_READWRITE && prot != PAGE_EXECUTE_READWRITE {
            return None;
        }
        v.push((a, e, prot));
        a = e;
    }
    Some(v)
}

/// Starts watching [base, base+size); returns its id. Pages must be mapped,
/// readable and writable guest memory (others are not watched).
/// CPU writes are caught by page protection (`DS_TRACK=1`).
pub fn protected() -> bool {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ON.get_or_init(|| std::env::var("DS_TRACK").ok().as_deref() == Some("1"))
}

/// Drops a region (its texture left the cache): no longer reported or
/// marked. Its pages keep the protection they have (they are made writable
/// again at their next fault).
pub fn unwatch(id: u32) {
    let mut g = TRACK.lock();
    let Some(t) = g.as_mut() else { return };
    let Some(slot) = t.regions.get_mut(id as usize) else { return };
    let Some(r) = slot.take() else { return };
    live_count(r.base, r.size, false);
    if let Some(v) = t.by_base.get_mut(&r.base) {
        v.retain(|&x| x != id);
        if v.is_empty() {
            t.by_base.remove(&r.base);
        }
    }
    for b in bucket_span(r.base, r.size) {
        if let Some(v) = t.buckets.get_mut(&b) {
            v.retain(|&x| x != id);
            if v.is_empty() {
                t.buckets.remove(&b);
            }
        }
    }
    if !t.pages.is_empty() {
        let mut p = r.base & !(PAGE - 1);
        while p < r.base + r.size {
            if let Some(e) = t.pages.get_mut(&p) {
                e.0.retain(|&x| x != id);
            }
            p += PAGE;
        }
    }
    RESET_GEN.fetch_add(1, Ordering::Release);
}

pub fn watch(base: u64, size: u64) -> Option<u32> {
    if !protected() {
        let mut g = TRACK.lock();
        let t = g.get_or_insert_with(Tracker::default);
        return Some(t.add(base, size));
    }
    let sp = spans(base, size)?;
    let mut g = TRACK.lock();
    let t = g.get_or_insert_with(Tracker::default);
    let id = t.add(base, size);
    arm_locked(t, id, &sp);
    Some(id)
}

/// Registers the region's pages (current protections) and write-protects
/// the unprotected ones.
fn arm_locked(t: &mut Tracker, id: u32, spans: &[(u64, u64, u32)]) {
    for &(a, e, prot) in spans {
        let mut p = a;
        while p < e {
            let ent = t.pages.entry(p).or_insert((Vec::new(), prot, false));
            if !ent.0.contains(&id) {
                ent.0.push(id);
            }
            p += PAGE;
        }
    }
    for &(a, e, _) in spans {
        let mut p = a;
        while p < e {
            let (prot, protected) = match t.pages.get(&p) {
                Some(x) => (x.1, x.2),
                None => {
                    p += PAGE;
                    continue;
                }
            };
            if protected {
                p += PAGE;
                continue;
            }
            if t.busy.iter().any(|&(_, b, n)| p < b + n && b < p + PAGE) {
                // A non-faulting writer is active here: stay writable, and
                // re-arm on the next check.
                if let Some(Some(r)) = t.regions.get_mut(id as usize) {
                    let why = r.why;
                    r.mark(None, why);
                }
                p += PAGE;
                continue;
            }
            let mut q = p + PAGE;
            while q < e && t.pages.get(&q).is_some_and(|x| !x.2 && x.1 == prot) && !t.busy.iter().any(|&(_, b, n)| q < b + n && b < q + PAGE) {
                q += PAGE;
            }
            if set_prot(p, (q - p) / PAGE, read_only(prot)) {
                let mut x = p;
                while x < q {
                    if let Some(en) = t.pages.get_mut(&x) {
                        en.2 = true;
                    }
                    x += PAGE;
                }
            }
            p = q;
        }
    }
}

/// Whether the region was written by the CPU since it was last armed (call
/// `rearm` before reading the memory): why (the last cause), and the ranges
/// written when known (None: any part may have changed).
pub fn take_dirty(id: u32) -> Option<(Why, Option<Vec<(u64, u64)>>)> {
    let mut g = TRACK.lock();
    let t = g.as_mut()?;
    match t.regions.get_mut(id as usize).and_then(|r| r.as_mut()) {
        Some(r) if r.dirty => {
            r.dirty = false;
            r.gpu = false;
            RESET_GEN.fetch_add(1, Ordering::Release);
            let whole = std::mem::replace(&mut r.whole, false);
            let parts = std::mem::take(&mut r.parts);
            if events_on() {
                let mut h = TAKES.lock();
                if h.len() >= 1 << 14 {
                    h.pop_front();
                }
                h.push_back((std::time::Instant::now(), id, whole, parts.clone()));
            }
            Some((r.why, (!whole).then_some(parts)))
        }
        _ => None,
    }
}

/// Marks a region dirty, written by the CPU somewhere (sampled hashes
/// changed: texture scrubbing). The written ranges known so far stay: its
/// consumer finds CPU-written pages by their write-watch marks.
pub fn sampled_dirty(id: u32) {
    let mut g = TRACK.lock();
    if let Some(r) = g.as_mut().and_then(|t| t.regions.get_mut(id as usize)).and_then(|r| r.as_mut()) {
        if !r.dirty {
            changed(0);
        }
        r.mark(Some((0, 0)), Why::Sampled);
    }
}

/// The CPU wrote `pages` (ascending 4 KiB page addresses: write-watch marks,
/// which their taker reset): regions on them other than `except` are dirty
/// with those parts.
pub fn cpu_pages(pages: &[u64], except: Option<u32>) {
    let mut runs: Vec<(u64, u64)> = Vec::new();
    for &p in pages {
        match runs.last_mut() {
            Some(r) if r.0 + r.1 == p => r.1 += PAGE,
            _ => runs.push((p, PAGE)),
        }
    }
    runs.retain(|&(b, n)| may_have_region(b, n));
    for &(b, n) in &runs {
        note_event("CPU write (write watch)", b, n);
    }
    if runs.is_empty() {
        return;
    }
    let mut g = TRACK.lock();
    let Some(t) = g.as_mut() else { return };
    for (b, n) in runs {
        t.each_overlapping_id(b, n, |id, r| {
            if Some(id) != except {
                if !r.dirty {
                    changed(2);
                }
                r.mark(Some((b, n)), Why::Written(b));
            }
        });
    }
}

/// Write marks of [base, base+len) could not be taken: regions on it other
/// than `except` count it as written.
pub fn cpu_range(base: u64, len: u64, except: Option<u32>) {
    if !may_have_region(base, len) {
        return;
    }
    let mut g = TRACK.lock();
    let Some(t) = g.as_mut() else { return };
    t.each_overlapping_id(base, len, |id, r| {
        if Some(id) != except {
            if !r.dirty {
                changed(2);
            }
            r.mark(Some((base, len)), Why::Written(base));
        }
    });
}

/// Marks a region dirty (its texture is uploaded again at its next use).
pub fn force_dirty(id: u32) {
    let mut g = TRACK.lock();
    if let Some(r) = g.as_mut().and_then(|t| t.regions.get_mut(id as usize)).and_then(|r| r.as_mut()) {
        if !r.dirty {
            changed(0);
        }
        r.mark(None, Why::Forced);
    }
}

/// Whether the region holds written parts not taken yet that overlap
/// [base, base+len) (or is written all over).
pub fn pending_overlaps(id: u32, base: u64, len: u64) -> bool {
    let g = TRACK.lock();
    let Some(r) = g.as_ref().and_then(|t| t.regions.get(id as usize)).and_then(|r| r.as_ref()) else { return false };
    // (A GPU write announced over it: the whole region is hashed again at
    // its next use.)
    r.gpu || (r.dirty && (r.whole || r.parts.iter().any(|&(b, n)| b < base + len && base < b.saturating_add(n))))
}

/// Whether the region is marked dirty (not yet taken).
pub fn is_dirty(id: u32) -> bool {
    let g = TRACK.lock();
    g.as_ref().and_then(|t| t.regions.get(id as usize)).and_then(|r| r.as_ref()).is_some_and(|r| r.dirty)
}

/// A region's guest range.
/// Region `id`'s generation (it moves whenever the region turns dirty or GPU-written).
pub fn gen_of(id: u32) -> Option<&'static std::sync::atomic::AtomicU32> {
    TRACK.lock().as_ref().and_then(|t| t.regions.get(id as usize)).and_then(|r| r.as_ref()).map(|r| r.generation)
}

pub fn range(id: u32) -> Option<(u64, u64)> {
    let g = TRACK.lock();
    g.as_ref()?.regions.get(id as usize)?.as_ref().map(|r| (r.base, r.size))
}

/// Whether GPU work wrote the region since the last call.
pub fn take_gpu_written(id: u32) -> bool {
    let mut g = TRACK.lock();
    let Some(t) = g.as_mut() else { return false };
    match t.regions.get_mut(id as usize).and_then(|r| r.as_mut()) {
        Some(r) => {
            let was = std::mem::replace(&mut r.gpu, false);
            if was {
                RESET_GEN.fetch_add(1, Ordering::Release);
            }
            was
        }
        None => false,
    }
}

/// Write-protects the region again (pages re-registered with their current
/// protections, so mapping changes are followed).
pub fn rearm(id: u32) {
    if !protected() {
        return;
    }
    let range = {
        let g = TRACK.lock();
        g.as_ref().and_then(|t| t.regions.get(id as usize)).and_then(|r| r.as_ref()).map(|r| (r.base, r.size))
    };
    let Some((base, size)) = range else { return };
    let sp = spans(base, size);
    let mut g = TRACK.lock();
    let Some(t) = g.as_mut() else { return };
    match sp {
        Some(sp) => arm_locked(t, id, &sp),
        // Unmapped or not writable now: report dirty on every check.
        None => {
            if let Some(Some(r)) = t.regions.get_mut(id as usize) {
                r.mark(None, Why::Unmapped);
                changed(1);
            }
        }
    }
}

/// Marks the regions on `page` dirty and makes it writable.
fn unprotect_page(t: &mut Tracker, page: u64, why: Why) -> bool {
    let Some(e) = t.pages.get_mut(&page) else { return false };
    if e.2 {
        if !set_prot(page, 1, e.1) {
            return false;
        }
        e.2 = false;
    }
    let ids = e.0.clone();
    for id in ids {
        if let Some(Some(r)) = t.regions.get_mut(id as usize) {
            r.mark(Some((page, PAGE)), why);
            changed(2);
        }
    }
    true
}

/// Exception handler hook: a CPU write to a watched page. Returns true when
/// the write may be retried.
pub fn on_write_fault(addr: u64) -> bool {
    let mut g = TRACK.lock();
    let Some(t) = g.as_mut() else { return false };
    let ok = unprotect_page(t, addr & !(PAGE - 1), Why::Fault(addr));
    if ok {
        FAULTS.fetch_add(1, Ordering::Relaxed);
    }
    ok
}

/// A write that cannot fault is in progress (see `host_write`).
/// (busy-range id, written range: homes copy it in when the write is done).
pub struct HostWrite(Option<u64>, u64, u64);

impl Drop for HostWrite {
    fn drop(&mut self) {
        super::homes::cpu_written(self.1, self.2);
        super::resident::host_write_end(self.1, self.2);
        super::fe2::host_write_end(self.1, self.2);
        if let Some(id) = self.0 {
            if let Some(t) = TRACK.lock().as_mut() {
                t.busy.retain(|b| b.0 != id);
            }
        }
    }
}

/// Announces a write that cannot fault (kernel-mode writes such as file
/// reads into guest memory, the driver pinning pages for an import): watched
/// pages in the range become writable, their regions dirty, and they stay
/// writable while the returned guard lives.
pub fn host_write(base: u64, len: u64) -> HostWrite {
    note_event("host write", base, len);
    super::precache::invalidate(base, len);
    super::resident::host_write_begin(base, len);
    super::fe2::host_write_begin(base, len);
    // (Front end 4's held pages: released first, their newest words put
    // back; a kernel write fails on a no-access page.)
    super::fe2::c4_release_sync(base, len, false);
    // Pages armed through the host (the GPU-resolved mirror, the copy
    // cache's protection): writable first (the kernel fails a file read
    // into a read-only page instead of faulting).
    crate::hosted::disarm_if_armed(base, len);
    if len == 0 {
        return HostWrite(None, base, 0);
    }
    let mut g = TRACK.lock();
    let Some(t) = g.as_mut() else { return HostWrite(None, base, len) };
    if t.regions.is_empty() {
        return HostWrite(None, base, len);
    }
    let (first, end) = (base & !(PAGE - 1), base + len);
    if (end - first) / PAGE > t.pages.len() as u64 {
        let pages: Vec<u64> = t.pages.keys().copied().filter(|&p| p >= first && p < end).collect();
        for p in pages {
            unprotect_page(t, p, Why::HostWrite(base));
        }
    } else {
        let mut p = first;
        while p < end {
            unprotect_page(t, p, Why::HostWrite(base));
            p += PAGE;
        }
    }
    // Regions without watched pages (no page protection) learn of the
    // write here.
    t.each_overlapping(base, len, |r| {
        if !r.dirty {
            changed(3);
        }
        r.mark(Some((base, len)), Why::HostWrite(base));
    });
    t.next_busy += 1;
    let id = t.next_busy;
    t.busy.push((id, base, len));
    HostWrite(Some(id), base, len)
}

/// Announces a GPU write to [base, base+len) once its batch completed (its
/// data is in guest memory: `exec::track_completed`): overlapping regions
/// are hashed again at their next use (their pages need no change). Not at
/// recording: a use before the batch completed hashed guest memory without
/// the write and took the mark (2026-11-11).
pub fn gpu_written(base: u64, len: u64) {
    if !may_have_region(base, len) {
        return;
    }
    note_event("GPU write (completed)", base, len);
    let t0 = std::time::Instant::now();
    let mut g = TRACK.lock();
    let Some(t) = g.as_mut() else { return };
    let mut hit = 0u64;
    t.each_overlapping(base, len, |r| {
        if !r.gpu {
            changed(4);
            r.generation.fetch_add(1, Ordering::Release);
        }
        hit += 1;
        r.gpu = true;
        r.why = Why::GpuWrite(base, len);
    });
    drop(g);
    GPU_WRITTEN[0].fetch_add(1, Ordering::Relaxed);
    GPU_WRITTEN[1].fetch_add(hit, Ordering::Relaxed);
    GPU_WRITTEN[2].fetch_add(t0.elapsed().as_nanos() as u64, Ordering::Relaxed);
}

/// `gpu_written`: [ranges announced, regions they marked, ns].
static GPU_WRITTEN: [AtomicU64; 3] = [const { AtomicU64::new(0) }; 3];

/// DS_UPLOAD_CHECK=1: the writes announced over tracked regions, newest
/// last ((when, what, base, len); diagnostics of changed texture parts).
static EVENTS: parking_lot::Mutex<std::collections::VecDeque<(std::time::Instant, &'static str, u64, u64)>> = parking_lot::Mutex::new(std::collections::VecDeque::new());

fn events_on() -> bool {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ON.get_or_init(|| std::env::var("DS_UPLOAD_CHECK").ok().as_deref() == Some("1"))
}

fn note_event(what: &'static str, base: u64, len: u64) {
    if !events_on() || !may_have_region(base, len) {
        return;
    }
    let mut e = EVENTS.lock();
    if e.len() >= 1 << 16 {
        e.pop_front();
    }
    e.push_back((std::time::Instant::now(), what, base, len));
}

/// DS_UPLOAD_CHECK=1: `take_dirty` results ((when, region, whole, parts)).
static TAKES: parking_lot::Mutex<std::collections::VecDeque<(std::time::Instant, u32, bool, Vec<(u64, u64)>)>> = parking_lot::Mutex::new(std::collections::VecDeque::new());

/// The region's `take_dirty` results since `since` (diagnostics).
pub fn takes_since(id: u32, since: std::time::Instant) -> Vec<String> {
    let h = TAKES.lock();
    h.iter().filter(|x| x.1 == id && x.0 >= since).map(|x| format!("{} ms after: {}", (x.0 - since).as_millis(), if x.2 { "whole".to_string() } else { format!("{:x?}", x.3) })).collect()
}

/// The noted writes overlapping [base, base+len) since `since` (diagnostics).
pub fn events_since(base: u64, len: u64, since: std::time::Instant) -> Vec<String> {
    let e = EVENTS.lock();
    e.iter()
        .filter(|x| x.0 >= since && x.2 < base + len && base < x.2.saturating_add(x.3))
        .map(|x| format!("{} {:#x}+{:#x} {} ms after", x.1, x.2, x.3, (x.0 - since).as_millis()))
        .collect()
}

/// Kinds of mapping change (`Why::Remap`).
pub const REMAP_COMMIT: u8 = 0;
pub const REMAP_DECOMMIT: u8 = 1;
pub const REMAP_PROTECT: u8 = 2;

/// The memory manager changed the mapping or protection of [base,
/// base+len) (`op`: `REMAP_*`): its page states are forgotten (re-registered
/// on re-arming) and overlapping regions are dirty.
pub fn remapped(base: u64, len: u64, op: u8) {
    note_event(["commit", "decommit", "protection"][(op as usize).min(2)], base, len);
    super::exec::note_remap_over_imports(base, len, op);
    super::precache::invalidate(base, len);
    super::homes::remapped(base, len);
    // Page states are reset by `resident::settled` after the change, under
    // `resident::remap_guard`: reset here, before it, armed pages stayed
    // read-only without being marked ours until the change landed, and a
    // guest write meanwhile was fatal (gate window 20261015-172335).
    RESET_GEN.fetch_add(1, Ordering::Release);
    let mut g = TRACK.lock();
    let Some(t) = g.as_mut() else { return };
    if t.regions.is_empty() {
        return;
    }
    let (first, end) = (base & !(PAGE - 1), base.saturating_add(len));
    if (end - first) / PAGE > t.pages.len() as u64 {
        t.pages.retain(|&p, _| p < first || p >= end);
    } else {
        let mut p = first;
        while p < end {
            t.pages.remove(&p);
            p += PAGE;
        }
    }
    t.each_overlapping(base, len, |r| {
        if !r.dirty {
            changed(5);
        }
        r.mark(Some((base, len)), Why::Remap(base, len, op));
    });
}

/// One status line.
pub fn status() -> String {
    let g = TRACK.lock();
    let (regions, pages) = g.as_ref().map(|t| (t.regions.iter().flatten().count(), t.pages.len())).unwrap_or((0, 0));
    let (q, l, mx) = g.as_ref().map(|t| (t.queries, t.looked, t.max_size)).unwrap_or(([0; 2], [0; 2], 0));
    let per = |k: usize| l[k] as f64 / q[k].max(1) as f64;
    format!(
        "    write tracking: {regions} regions, {pages} pages, {} write faults; overlap queries since boot by 1 MiB buckets {} ({:.1} ids looked at each), of 1 GiB or more by bases from start - the largest region ({} MiB) {} ({:.1} each); GPU writes announced at their batch's completion {} ({} regions marked, {} ms)\n",
        FAULTS.load(Ordering::Relaxed),
        q[1],
        per(1),
        mx >> 20,
        q[0],
        per(0),
        GPU_WRITTEN[0].load(Ordering::Relaxed),
        GPU_WRITTEN[1].load(Ordering::Relaxed),
        GPU_WRITTEN[2].load(Ordering::Relaxed) / 1_000_000
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    fn hits(t: &mut Tracker, base: u64, len: u64) -> Vec<u64> {
        let mut v = Vec::new();
        t.each_overlapping(base, len, |r| v.push(r.base));
        v.sort_unstable();
        v
    }

    #[test]
    fn live_filter_is_exact_in_the_span_and_forgets_dropped_regions() {
        let mut t = Tracker::default();
        // (An address no other test uses: the counts are global.)
        let base = 0x17_8000_0000;
        assert!(!may_have_region(base, 0x1000));
        t.add(base + 0x2_0000, 0x1_0000);
        assert!(may_have_region(base + 0x2_0000, 4));
        assert!(may_have_region(base + 0x2_fffc, 4));
        assert!(may_have_region(base, 0x100_0000));
        assert!(!may_have_region(base, 0x1_0000));
        assert!(!may_have_region(base + 0x3_0000, 0x40_0000));
        live_count(base + 0x2_0000, 0x1_0000, false);
        assert!(!may_have_region(base + 0x2_0000, 4));
        assert!(!may_have_region(base, 0x100_0000));
    }

    #[test]
    fn bucket_index_visits_each_overlapping_region_once() {
        let mut t = Tracker::default();
        t.add(0x10_0000_0000, 360 << 20);
        t.add(0x10_0010_0000, 0x1000);
        t.add(0x10_2000_0000, 0x10_0000);
        // Five buckets queried, the large region in all of them: once each.
        assert_eq!(hits(&mut t, 0x10_0008_0000, 5 << 20), vec![0x10_0000_0000, 0x10_0010_0000]);
        assert_eq!(hits(&mut t, 0x10_2000_0800, 16), vec![0x10_2000_0000]);
        assert_eq!(hits(&mut t, 0x10_1670_0000, 0x20_0000), vec![0x10_0000_0000]);
        assert_eq!(hits(&mut t, 0x10_3000_0000, 16), Vec::<u64>::new());
    }

    #[test]
    fn overlap_index_finds_every_overlapping_region() {
        let mut t = Tracker::default();
        t.add(0x1000, 0x1000);
        t.add(0x8000, 0x10_0000);
        t.add(0x20_0000, 0x10);
        t.add(0x8000, 0x10);
        assert_eq!(hits(&mut t, 0x1800, 4), vec![0x1000]);
        // A large region starting well before the write.
        assert_eq!(hits(&mut t, 0x10_0000, 4), vec![0x8000]);
        assert_eq!(hits(&mut t, 0x8008, 4), vec![0x8000, 0x8000]);
        assert_eq!(hits(&mut t, 0x2000, 0x6000), Vec::<u64>::new());
        assert_eq!(hits(&mut t, 0, u64::MAX), vec![0x1000, 0x8000, 0x8000, 0x20_0000]);
        assert_eq!(hits(&mut t, 0x20_0010, 0x100), Vec::<u64>::new());
    }
}
