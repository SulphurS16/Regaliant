//! Guest virtual memory.
//!
//! The PS4 address range used by the guest is reserved once in the host process
//! and pages are committed/decommitted at 4 KiB granularity inside it, so guest
//! pointers are host pointers. A VMA map mirrors what the guest has mapped, and a
//! separate allocator models PS4 "direct memory" (physical) ranges. The range is
//! one placeholder whose 64 MiB windows become write-watched allocations of
//! their own at first use (`mixed`; `DS_MIXED=0`: one write-watched
//! reservation): the kernel serializes write-watch queries within one
//! allocation.
//!
//! Limitation (reported, never silent): mapping the same direct-memory range at
//! two virtual addresses (aliasing) is not shared; it is detected and logged.
//! With `DS_SECTIONS=1` guest memory is section-backed instead (`sections`):
//! mappings are views, physical pages keep their contents across remaps and
//! aliases share them, and the GPU imports alias views.

use parking_lot::Mutex;
use std::collections::BTreeMap;
use crate::platform::os::Win32::System::Memory::*;

pub const GUEST_BASE: u64 = 0x4_0000_0000; // 16 GiB
pub const GUEST_END: u64 = 0x800_0000_0000; // 8 TiB (the title reserves ~1 TiB for virtual textures)
pub const MODULE_BASE: u64 = 0x8_0000_0000; // eboot image base (matches reference loaders)
pub const MODULE_END: u64 = 0x10_0000_0000;
pub const MMAP_BASE: u64 = 0x10_0000_0000; // default search start for guest mappings
pub const INTERNAL_BASE: u64 = 0x4_0000_0000; // runtime-owned guest-visible memory
pub const PAGE: u64 = 0x4000; // PS5 page size
/// Direct memory available to the title: PS5 game memory of 13.5 GiB minus the
/// default 1 GiB flexible budget (values from public reverse engineering, unconfirmed).
pub const DMEM_SIZE: u64 = 12800 << 20;

pub const PROT_CPU_READ: u32 = 0x1;
pub const PROT_CPU_WRITE: u32 = 0x2;
pub const PROT_CPU_EXEC: u32 = 0x4;
pub const PROT_GPU_READ: u32 = 0x10;
pub const PROT_GPU_WRITE: u32 = 0x20;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum VmaKind {
    Reserved,
    Direct,
    Flexible,
    Module,
    Internal,
}

#[derive(Clone, Debug)]
pub struct Vma {
    pub start: u64,
    pub size: u64,
    pub kind: VmaKind,
    pub prot: u32,
    pub phys: u64,
    pub mtype: i32,
    pub name: String,
    /// Section-backed memory (`sections`): the alias slots holding alias
    /// views of its pages that it references (bit per slot: the slot its
    /// mapping took, and slots `bridge_aliases` added) and, for memory other
    /// than direct, its offset in the anonymous section.
    pub slots: u8,
    pub off: u64,
    /// The flip it was mapped at (diagnostics; its pieces keep it).
    pub born: u64,
}
/// The anonymous section's size (`sections`): flexible memory, modules and
/// runtime-owned guest memory.
const ANON_SIZE: u64 = 4 << 30;

impl Vma {
    pub fn end(&self) -> u64 {
        self.start + self.size
    }
}

#[derive(Clone, Copy, Debug)]
pub struct DmemBlock {
    pub size: u64,
    pub mtype: i32,
}

pub struct MemState {
    pub vmas: BTreeMap<u64, Vma>,
    pub dmem: BTreeMap<u64, DmemBlock>,
    pub flexible_used: u64,
    internal_cursor: u64,
}

/// Direct-memory mappings whose contents live with the physical pages on the
/// console but with the virtual pages here (diagnostics): `in_place` maps the
/// same physical pages again over their own mapping (we zero them), `moved`
/// maps physical pages last mapped at another address (their data stays
/// there). `released`: physical start -> (address, size) of unmapped ranges.
struct RemapStats {
    in_place: u64,
    in_place_bytes: u64,
    moved: u64,
    moved_bytes: u64,
    logged: u32,
    released: BTreeMap<u64, (u64, u64)>,
}

static REMAP: Mutex<RemapStats> = Mutex::new(RemapStats { in_place: 0, in_place_bytes: 0, moved: 0, moved_bytes: 0, logged: 0, released: BTreeMap::new() });

/// `readable_span` calls and cache misses (diagnostics). Calls are counted
/// per thread and added here every SPAN_CALLS_BATCH (one shared counter
/// bumped by every recording thread ~2.3 million times per second was
/// 0.5-1% of their busy samples, run 20261013-115322).
static SPAN_CALLS: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);
const SPAN_CALLS_BATCH: u32 = 1024;
static SPAN_MISSES: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);
/// Per-thread span caches found full by a miss (`readable_span`).
static SPAN_FULL: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);

/// One status line (diagnostics).
/// Contents of unmapped direct memory kept with its physical pages
/// (`DS_PHYS_CARRY=1`). On the console a physical page keeps its data while
/// unmapped and shows it wherever it is mapped next; here a mapping's pages
/// are private to its addresses, so the title's remaps of physical memory to
/// new addresses (~27k per tour, 8 GB, mostly `sceKernelBatchMap`) arrived
/// zeroed. Unmapping parks the non-zero 4 KiB pages by physical address
/// (`park`), the next mapping of them copies them in (`unpark`), releasing
/// the physical memory drops them (`drop_parked`).
static PARKED: Mutex<BTreeMap<u64, Box<[u8]>>> = Mutex::new(BTreeMap::new());
/// [pages parked, zero pages left out, pages copied into a new mapping,
/// pages dropped by releases, bytes held, pages not parked (cap)].
pub static PARK_STATS: [std::sync::atomic::AtomicU64; 6] = [const { std::sync::atomic::AtomicU64::new(0) }; 6];
/// Most bytes kept parked.
const PARK_CAP: u64 = 4 << 30;
/// Direct mappings whose address, size and physical offset are all 64 KiB
/// aligned, 16 KiB aligned, or neither.
static MAP_ALIGN: [std::sync::atomic::AtomicU64; 6] = [const { std::sync::atomic::AtomicU64::new(0) }; 6];
const PARK_PAGE: u64 = 4096;

/// `DS_PHYS_CARRY=1`: unmapped direct memory keeps its contents (`PARKED`).
pub fn phys_carry() -> bool {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ON.get_or_init(|| env_is("DS_PHYS_CARRY", "1"))
}

/// Parks the non-zero pages of the direct mapping [va, va+size) of physical
/// `phys` (before its pages are decommitted).
fn park(va: u64, phys: u64, size: u64) {
    use std::sync::atomic::Ordering::Relaxed;
    if !os_readable(va, size) {
        return;
    }
    let mut g = PARKED.lock();
    let mut off = 0;
    while off < size {
        let page = unsafe { std::slice::from_raw_parts((va + off) as *const u64, (PARK_PAGE / 8) as usize) };
        let key = phys + off;
        if page.iter().all(|&w| w == 0) {
            if g.remove(&key).is_some() {
                PARK_STATS[4].fetch_sub(PARK_PAGE, Relaxed);
            }
            PARK_STATS[1].fetch_add(1, Relaxed);
        } else if PARK_STATS[4].load(Relaxed) + PARK_PAGE > PARK_CAP && !g.contains_key(&key) {
            PARK_STATS[5].fetch_add(1, Relaxed);
        } else {
            let bytes = unsafe { std::slice::from_raw_parts((va + off) as *const u8, PARK_PAGE as usize) };
            if g.insert(key, bytes.into()).is_none() {
                PARK_STATS[4].fetch_add(PARK_PAGE, Relaxed);
            }
            PARK_STATS[0].fetch_add(1, Relaxed);
        }
        off += PARK_PAGE;
    }
}

/// Copies the parked pages of physical [phys, phys+size) into the new
/// direct mapping at `va` (protection `prot`), and forgets them.
fn unpark(va: u64, phys: u64, size: u64, prot: u32) {
    use std::sync::atomic::Ordering::Relaxed;
    let mut g = PARKED.lock();
    let keys: Vec<u64> = g.range(phys..phys + size).map(|(k, _)| *k).collect();
    if keys.is_empty() {
        return;
    }
    let writable = matches!(win_prot(prot), PAGE_READWRITE | PAGE_EXECUTE_READWRITE);
    let mut old = 0u32;
    if !writable {
        for (a, e) in host_pieces(va, va + size) {
            unsafe { VirtualProtect(a as _, (e - a) as usize, PAGE_READWRITE, &mut old) };
        }
    }
    for k in keys {
        if let Some(b) = g.remove(&k) {
            unsafe { std::ptr::copy_nonoverlapping(b.as_ptr(), (va + (k - phys)) as *mut u8, PARK_PAGE as usize) };
            PARK_STATS[4].fetch_sub(PARK_PAGE, Relaxed);
            PARK_STATS[2].fetch_add(1, Relaxed);
        }
    }
    if !writable {
        for (a, e) in host_pieces(va, va + size) {
            unsafe { VirtualProtect(a as _, (e - a) as usize, win_prot(prot), &mut old) };
        }
    }
}

/// Forgets the parked pages of released physical [start, end).
fn drop_parked(start: u64, end: u64) {
    use std::sync::atomic::Ordering::Relaxed;
    let mut g = PARKED.lock();
    let keys: Vec<u64> = g.range(start..end).map(|(k, _)| *k).collect();
    for k in keys {
        g.remove(&k);
        PARK_STATS[4].fetch_sub(PARK_PAGE, Relaxed);
        PARK_STATS[3].fetch_add(1, Relaxed);
    }
}

/// A mapping's access class from the title's protection bits: "CPU+GPU",
/// "GPU only" (no CPU bit: the title's CPU code cannot touch it), "CPU only"
/// or "none".
pub fn prot_class(prot: u32) -> &'static str {
    match (prot & 7 != 0, prot & (PROT_GPU_READ | PROT_GPU_WRITE) != 0) {
        (true, true) => "CPU+GPU",
        (false, true) => "GPU only",
        (true, false) => "CPU only",
        (false, false) => "none",
    }
}

/// The title's protection bits of the mapping holding `a` (None: unmapped).
pub fn prot_at(a: u64) -> Option<u32> {
    MEM.lock().find(a).filter(|v| v.kind != VmaKind::Reserved).map(|v| v.prot)
}

/// The memory type the title allocated the direct memory under `a` with
/// (None: unmapped or not direct memory).
pub fn mtype_at(a: u64) -> Option<i32> {
    let g = MEM.lock();
    let v = g.find(a).filter(|v| v.kind == VmaKind::Direct)?;
    let phys = v.phys + (a - v.start);
    g.dmem.range(..=phys).next_back().filter(|(s, b)| phys < **s + b.size).map(|(_, b)| b.mtype)
}

/// Guest mappings by kind, access class and memory type (MiB, mappings;
/// status report). The memory type is the one the title asked for (direct
/// memory: `sceKernelAllocateDirectMemory`'s), which says how it means to
/// use the memory (CPU-cached for what the CPU reads back, GPU-cached or
/// write-combined for GPU work).
pub fn prot_census() -> String {
    let g = MEM.lock();
    let mut m: std::collections::BTreeMap<(String, &'static str, i32), (u64, u64)> = std::collections::BTreeMap::new();
    for v in g.vmas.values() {
        if v.kind == VmaKind::Reserved {
            continue;
        }
        let e = m.entry((format!("{:?}", v.kind), prot_class(v.prot), v.mtype)).or_insert((0, 0));
        e.0 += v.size;
        e.1 += 1;
    }
    // Direct memory by the type the title allocated it with.
    let mut types: std::collections::BTreeMap<i32, (u64, u64)> = std::collections::BTreeMap::new();
    for b in g.dmem.values() {
        let e = types.entry(b.mtype).or_insert((0, 0));
        e.0 += b.size;
        e.1 += 1;
    }
    drop(g);
    let dm: Vec<String> = types.iter().map(|(t, (b, n))| format!("type {t} {} MiB ({n})", b >> 20)).collect();
    let parts: Vec<String> = m.iter().map(|((k, c, t), (b, n))| format!("{k} {c} type {t} {} MiB ({n})", b >> 20)).collect();
    format!("    direct memory by the type the title allocated it with: {}\n", dm.join(", "))
        + &format!("    guest mappings by kind, the title's access bits and memory type: {}\n", parts.join(", "))
}

pub fn remap_status() -> String {
    let r = REMAP.lock();
    let o = std::sync::atomic::Ordering::Relaxed;
    let carry = if phys_carry() {
        let v: Vec<u64> = PARK_STATS.iter().map(|c| c.load(o)).collect();
        format!(
            "; contents carried (DS_PHYS_CARRY): {} pages parked ({} zero left out), {} copied into later mappings, {} dropped by releases, {} MiB held, {} not parked (cap)",
            v[0],
            v[1],
            v[2],
            v[3],
            v[4] >> 20,
            v[5]
        )
    } else {
        String::new()
    };
    format!(
        "    direct memory remaps: {} in place ({} MiB, zeroed here), {} of pages last mapped elsewhere ({} MiB, contents not carried); mapping changes {}; readable-range checks {} ({} cache misses, {} with the cache full, {} served by the shared runs); decommits deferred under GPU imports {} ({} MiB; {} done later, {} mapped again first, {} waiting)\n",
        r.in_place,
        r.in_place_bytes >> 20,
        r.moved,
        r.moved_bytes >> 20,
        MAP_GEN.load(o),
        SPAN_CALLS.load(o),
        SPAN_MISSES.load(o),
        SPAN_FULL.load(o),
        SPAN_SHARED_HITS.load(o),
        DEFERRED_STATS[0].load(o),
        DEFERRED_STATS[3].load(o) >> 20,
        DEFERRED_STATS[1].load(o),
        DEFERRED_STATS[2].load(o),
        DEFERRED_LEN.load(o)
    )
    .replace(
        " waiting)\n",
        &format!(
            " waiting){carry}; direct mappings aligned to 64 KiB {}, to 16 KiB {}, less {}; other mappings {}, {}, {}{}\n",
            MAP_ALIGN[0].load(o),
            MAP_ALIGN[1].load(o),
            MAP_ALIGN[2].load(o),
            MAP_ALIGN[3].load(o),
            MAP_ALIGN[4].load(o),
            MAP_ALIGN[5].load(o),
            if mixed() { format!("; {}", mixed_state()) } else { String::new() }
        ),
    )
}

/// Incremented whenever guest mappings or their protections change (inside
/// the MEM lock): per-thread caches of readable ranges are valid for one
/// value (`readable_span`).
pub static MAP_GEN: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);

/// The latest mapping changes: (generation it made, start, end), oldest
/// first. A thread's cached readable runs that no change since its
/// generation touched stay valid: the title's texture streaming remaps
/// 64 KiB pieces many times per second, and every change used to empty
/// every command processor's cache (`readable_span` was ~12% of the
/// graphics thread's busy samples, run 20261011-111123).
static CHANGES: parking_lot::RwLock<std::collections::VecDeque<(u64, u64, u64)>> = parking_lot::RwLock::new(std::collections::VecDeque::new());
const CHANGES_KEEP: usize = 256;

/// Notes a mapping change of [start, end) (inside the MEM lock).
fn bump_gen(start: u64, end: u64) {
    let mut c = CHANGES.write();
    let generation = MAP_GEN.fetch_add(1, std::sync::atomic::Ordering::SeqCst) + 1;
    if c.len() >= CHANGES_KEEP {
        c.pop_front();
    }
    c.push_back((generation, start, end));
    drop(c);
    if let Some(p) = crate::gpu::fe2::traced(start, end) {
        crate::gpu::fe2::trace_event(p, &format!("a mapping change over it: {start:#x}..{end:#x} (generation {generation})"));
    }
}

/// Brings a thread's cached runs from generation `from` to `to`: drops the
/// runs a change in between touched; false when the changes are not all
/// known any more (the caller empties the cache).
fn revalidate(runs: &mut Vec<(u64, u64, bool)>, from: u64, to: u64) -> bool {
    if from == u64::MAX {
        return false;
    }
    // (Read by every recording thread after each change: shared, and only
    // the entries after `from`, newest first: generations rise along it.)
    let c = CHANGES.read();
    if c.front().is_none_or(|&(g, _, _)| g > from + 1) {
        return false;
    }
    for &(g, s, e) in c.iter().rev() {
        if g <= from {
            break;
        }
        if g <= to {
            runs.retain(|&(rs, re, _)| e <= rs || re <= s);
        }
    }
    true
}

thread_local! {
    /// This thread's recently found mappings (`vma_bounds`): [start, end),
    /// sorted and disjoint, valid for one MAP_GEN value.
    static VMAS: std::cell::RefCell<(u64, Vec<(u64, u64)>)> = const { std::cell::RefCell::new((u64::MAX, Vec::new())) };
}

/// Mappings cached per thread (`vma_bounds`).
const VMA_CACHED: usize = 256;

/// The guest mapping holding `a`: start, size, name (diagnostics: takes the
/// mapping table's lock).
pub fn vma_label(a: u64) -> Option<(u64, u64, String)> {
    MEM.lock().find(a).map(|v| (v.start, v.size, v.name.clone()))
}

/// [start, end) of the guest mapping holding `a` as the mapping table has
/// it (None in a gap or a reservation), from a per-thread cache while the
/// table has not changed since (`tune vma_cache`: the GPU executors look up
/// every range of a GPU-resolved command's chains and every unsized
/// binding, and the table's lock is shared with the title's threads).
pub fn vma_bounds(a: u64) -> Option<(u64, u64)> {
    let generation = MAP_GEN.load(std::sync::atomic::Ordering::Acquire);
    let hit = VMAS.with(|c| {
        let mut c = c.borrow_mut();
        if c.0 != generation {
            c.0 = generation;
            c.1.clear();
            return None;
        }
        let i = c.1.partition_point(|r| r.0 <= a);
        i.checked_sub(1).map(|i| c.1[i]).filter(|r| a < r.1)
    });
    if hit.is_some() {
        return hit;
    }
    let found = {
        let st = MEM.lock();
        st.find(a).filter(|v| v.kind != VmaKind::Reserved).map(|v| (v.start, v.end()))
    };
    if let Some((s, e)) = found {
        VMAS.with(|c| {
            let mut c = c.borrow_mut();
            if c.0 != generation {
                c.0 = generation;
                c.1.clear();
            }
            c.1.retain(|r| r.1 <= s || e <= r.0);
            if c.1.len() >= VMA_CACHED {
                c.1.clear();
            }
            let i = c.1.partition_point(|r| r.0 < s);
            c.1.insert(i, (s, e));
        });
    }
    found
}

/// Cached runs per thread, all cleared when full (512 with one evicted at a
/// time and a last-hit check were level in replay A/Bs: FINDINGS.md 2026-10-29).
const READABLE_RUNS: usize = 64;

/// The runs the threads' misses found (`readable_span`), shared: [start,
/// end) and readability, sorted and disjoint, for one
/// MAP_GEN value (carried over changes by `revalidate`, as the threads' own
/// are). Written under the MEM lock; read without it, as a thread's own
/// cache is (a change right after the generation was read is seen by the
/// next call). Every recording thread misses the runs a change dropped, and
/// each miss took the mapping table's lock for a walk of up to 64 mappings
/// either side (~3% of the graphics CP, half of it contended: sampler
/// -055047; in-run A/B: the graphics CP 0.4-0.6 ms a flip less, FINDINGS.md
/// 2026-11-03 morning).
static SHARED_RUNS: parking_lot::RwLock<(u64, Vec<(u64, u64, bool)>)> = parking_lot::RwLock::new((u64::MAX, Vec::new()));
const SHARED_RUNS_MAX: usize = 4096;
/// Misses `SHARED_RUNS` served (diagnostics).
static SPAN_SHARED_HITS: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);

/// The shared run holding `a` if it is current for generation `generation`.
fn shared_run(a: u64, generation: u64) -> Option<(u64, u64, bool)> {
    let sh = SHARED_RUNS.read();
    if sh.0 != generation {
        return None;
    }
    let i = sh.1.partition_point(|r| r.0 <= a);
    i.checked_sub(1).map(|i| sh.1[i]).filter(|r| a < r.1)
}

/// Adds a run found under the MEM lock (`generation`: MAP_GEN read there).
fn share_run(run: (u64, u64, bool), generation: u64) {
    let mut sh = SHARED_RUNS.write();
    if sh.0 != generation {
        let from = sh.0;
        if !crate::gpu::tune::map_ranges() || !revalidate(&mut sh.1, from, generation) {
            sh.1.clear();
        }
        sh.0 = generation;
    }
    let (start, end, _) = run;
    sh.1.retain(|r| r.1 <= start || end <= r.0);
    if sh.1.len() >= SHARED_RUNS_MAX {
        sh.1.clear();
    }
    let i = sh.1.partition_point(|r| r.0 < start);
    sh.1.insert(i, run);
}

// ---------------------------------------------------------------------------
// Readability of guest memory per 64 KiB granule (R2): kept under the MEM
// lock at every change of the mapping table (`MemState::rg_update`) and read
// without it (`rg_readable`). A granule's `ok` bit: it lies in one usable
// mapping (not reserved, some protection: `readable_span`'s rule); `mixed`:
// part of it is usable, or several mappings share it. Nearly every mapping is 64 KiB aligned, so the
// recording threads' readability checks need no lock, cache or walk of the
// table; mixed granules go the earlier way. (The title's texture streaming
// remaps 64 KiB pieces ~250 times a second in a walk: every change dropped
// the threads' cached runs, and each miss walked up to 64 mappings either
// side under the lock the title's allocator holds: 7.5% of the parallel
// workers' busy samples in the 1-1 walk, FINDINGS.md 2026-11-10 evening.)
// ---------------------------------------------------------------------------

const RG_SHIFT: u32 = 16;
const RG_GRAN: u64 = 1 << RG_SHIFT;
const RG_LEAF_SHIFT: u32 = 30;
const RG_LEAF_GRANULES: usize = 1 << (RG_LEAF_SHIFT - RG_SHIFT);
const RG_LEAF_WORDS: usize = RG_LEAF_GRANULES / 64;
const RG_LEAVES: usize = (GUEST_END >> RG_LEAF_SHIFT) as usize;

struct RgLeaf {
    ok: [std::sync::atomic::AtomicU64; RG_LEAF_WORDS],
    mixed: [std::sync::atomic::AtomicU64; RG_LEAF_WORDS],
}

static RG: [std::sync::atomic::AtomicPtr<RgLeaf>; RG_LEAVES] = [const { std::sync::atomic::AtomicPtr::new(std::ptr::null_mut()) }; RG_LEAVES];

/// Sets granules [g0, g1) to (`ok`, `mixed`) (under the MEM lock). Leaves are
/// made only to hold set bits: a missing leaf reads as nothing usable.
fn rg_set_range(g0: u64, g1: u64, ok: bool, mixed: bool) {
    use std::sync::atomic::Ordering;
    let per = RG_LEAF_GRANULES as u64;
    let mut g = g0;
    while g < g1 {
        let li = (g / per) as usize;
        let lend = ((g / per + 1) * per).min(g1);
        if li >= RG_LEAVES {
            return;
        }
        let mut p = RG[li].load(Ordering::Acquire);
        if p.is_null() {
            if !ok && !mixed {
                g = lend;
                continue;
            }
            let leaf = Box::new(RgLeaf { ok: [const { std::sync::atomic::AtomicU64::new(0) }; RG_LEAF_WORDS], mixed: [const { std::sync::atomic::AtomicU64::new(0) }; RG_LEAF_WORDS] });
            p = Box::into_raw(leaf);
            RG[li].store(p, Ordering::Release);
        }
        let leaf = unsafe { &*p };
        let mut i = (g % per) as usize;
        let iend = (lend - (g / per) * per) as usize;
        while i < iend {
            let w = i / 64;
            let b0 = i % 64;
            let n = (iend - i).min(64 - b0);
            let m = if n == 64 { u64::MAX } else { ((1u64 << n) - 1) << b0 };
            if ok {
                leaf.ok[w].fetch_or(m, Ordering::AcqRel);
            } else {
                leaf.ok[w].fetch_and(!m, Ordering::AcqRel);
            }
            if mixed {
                leaf.mixed[w].fetch_or(m, Ordering::AcqRel);
            } else {
                leaf.mixed[w].fetch_and(!m, Ordering::AcqRel);
            }
            i += n;
        }
        g = lend;
    }
}

/// Usable granules over [start, end) made mixed at once (before an
/// unmapping or a protection change: readers go the earlier way, through the
/// table's lock, until the update after the change sets what holds then; a
/// change that fails between leaves them so).
fn rg_clear(start: u64, end: u64) {
    use std::sync::atomic::Ordering;
    if end <= start {
        return;
    }
    let per = RG_LEAF_GRANULES as u64;
    let (g0, g1) = (start >> RG_SHIFT, end.saturating_add(RG_GRAN - 1) >> RG_SHIFT);
    let mut g = g0;
    while g < g1 {
        let li = (g / per) as usize;
        let lend = ((g / per + 1) * per).min(g1);
        if li >= RG_LEAVES {
            return;
        }
        let p = RG[li].load(Ordering::Acquire);
        if !p.is_null() {
            let leaf = unsafe { &*p };
            let mut i = (g % per) as usize;
            let iend = (lend - (g / per) * per) as usize;
            while i < iend {
                let w = i / 64;
                let b0 = i % 64;
                let n = (iend - i).min(64 - b0);
                let m = if n == 64 { u64::MAX } else { ((1u64 << n) - 1) << b0 };
                let ok = leaf.ok[w].load(Ordering::Acquire) & m;
                if ok != 0 {
                    leaf.mixed[w].fetch_or(ok, Ordering::AcqRel);
                    leaf.ok[w].fetch_and(!ok, Ordering::AcqRel);
                }
                i += n;
            }
        }
        g = lend;
    }
}

/// Whether guest [a, a + len) is accessible by the readability map: Some
/// when it decides (every granule usable, or one with nothing usable), None
/// when a granule is mixed or the range lies outside the guest span (then
/// `readable_span` decides).
#[inline]
pub fn rg_readable(a: u64, len: u64) -> Option<bool> {
    use std::sync::atomic::Ordering;
    let end = a.checked_add(len.max(1))?;
    if a < GUEST_BASE || end > GUEST_END {
        return None;
    }
    let (g0, g1) = (a >> RG_SHIFT, (end - 1) >> RG_SHIFT);
    let mut mixed = false;
    let per = RG_LEAF_GRANULES as u64;
    let mut g = g0;
    while g <= g1 {
        let p = RG[(g / per) as usize].load(Ordering::Acquire);
        if p.is_null() {
            return rg_checked(a, len, end, false);
        }
        let leaf = unsafe { &*p };
        let i = (g % per) as usize;
        let (w, b) = (i / 64, 1u64 << (i % 64));
        if leaf.ok[w].load(Ordering::Acquire) & b == 0 {
            if leaf.mixed[w].load(Ordering::Acquire) & b == 0 {
                return rg_checked(a, len, end, false);
            }
            mixed = true;
        }
        g += 1;
    }
    if mixed {
        None
    } else {
        rg_checked(a, len, end, true)
    }
}

/// `rg_readable`'s decided answer `b` (`DS_R2_CHECK=1`: checked against the
/// table's: `readable_span`, else `mapped`).
fn rg_checked(a: u64, len: u64, end: u64, b: bool) -> Option<bool> {
    // (Only "readable" is used: `readable`, `Reader::readable`.)
    if b && crate::gpu::shaders::r2_check() {
        use std::sync::atomic::Ordering;
        let old = match readable_span(a) {
            Some((_, e)) if end <= e => true,
            Some(_) => mapped(a, len),
            None => false,
        };
        crate::gpu::shaders::R2_CHECKS[0].fetch_add(1, Ordering::Relaxed);
        if old != b && crate::gpu::shaders::R2_CHECKS[1].fetch_add(1, Ordering::Relaxed) < 40 {
            crate::info!("r2 check: readability map says {b} for {a:#x}+{len:#x}, the table {old}");
        }
    }
    Some(b)
}

impl MemState {
    /// The readability map's granules over [start, end) made again from the
    /// table (under the MEM lock, after a change).
    fn rg_update(&self, start: u64, end: u64) {
        if end <= start {
            return;
        }
        let usable = |v: &Vma| v.kind != VmaKind::Reserved && v.prot != 0;
        let g1 = end.saturating_add(RG_GRAN - 1) >> RG_SHIFT;
        let mut g = start >> RG_SHIFT;
        while g < g1 {
            let gs = g << RG_SHIFT;
            let ge = gs + RG_GRAN;
            match self.find(gs) {
                // Granules wholly inside one mapping: by its kind.
                Some(v) if v.end() >= ge => {
                    let last = (v.end() >> RG_SHIFT).min(g1).max(g + 1);
                    rg_set_range(g, last, usable(v), false);
                    g = last;
                    continue;
                }
                // Granules wholly inside a gap: nothing usable.
                None => {
                    let next = self.vmas.range(gs..).next().map(|(&k, _)| k).unwrap_or(u64::MAX);
                    if next >= ge {
                        let last = (next >> RG_SHIFT).min(g1).max(g + 1);
                        rg_set_range(g, last, false, false);
                        g = last;
                        continue;
                    }
                }
                _ => {}
            }
            // A granule shared by several mappings or a gap: mixed when any
            // piece is usable (`ok` is one usable mapping over the whole
            // granule: k such granules hold at most k mappings, which
            // `fe2_place_inner`'s run ends rely on).
            let (mut any_ok, mut any_bad) = (false, false);
            let mut p = gs;
            while p < ge {
                match self.find(p) {
                    Some(v) => {
                        if usable(v) {
                            any_ok = true;
                        } else {
                            any_bad = true;
                        }
                        p = v.end();
                    }
                    None => {
                        any_bad = true;
                        p = self.vmas.range(p..).next().map(|(&k, _)| k).unwrap_or(ge).max(p + 1);
                    }
                }
            }
            let _ = any_bad;
            rg_set_range(g, g + 1, false, any_ok);
            g += 1;
        }
    }
}

thread_local! {
    /// This thread's recently confirmed runs: [start, end) and whether they
    /// are readable (unmapped gaps are kept too: descriptors of unbound
    /// resources point there again and again), sorted by start and
    /// disjoint; valid for one MAP_GEN value (`revalidate` carries them
    /// over changes elsewhere).
    static READABLE: std::cell::RefCell<(u64, Vec<(u64, u64, bool)>)> = const { std::cell::RefCell::new((u64::MAX, Vec::new())) };
    /// This thread's `readable_span` calls not yet added to SPAN_CALLS.
    static SPAN_CALLS_LOCAL: std::cell::Cell<u32> = const { std::cell::Cell::new(0) };
}

/// The run of adjacent accessible mappings containing `a` as [start, end)
/// (at most 64 mappings either side), from a per-thread cache when the
/// mappings have not changed since (the GPU command processors check every
/// descriptor and buffer they read, ~2.3 million times per second in the
/// courtyard; the mapping table lock is shared with the title's threads).
pub fn readable_span(a: u64) -> Option<(u64, u64)> {
    SPAN_CALLS_LOCAL.with(|n| {
        let k = n.get() + 1;
        if k == SPAN_CALLS_BATCH {
            SPAN_CALLS.fetch_add(k as u64, std::sync::atomic::Ordering::Relaxed);
            n.set(0);
        } else {
            n.set(k);
        }
    });
    let generation = MAP_GEN.load(std::sync::atomic::Ordering::Acquire);
    let hit = READABLE.with(|c| {
        let mut c = c.borrow_mut();
        if c.0 != generation {
            let from = c.0;
            if !crate::gpu::tune::map_ranges() || !revalidate(&mut c.1, from, generation) {
                c.1.clear();
            }
            c.0 = generation;
        }
        let i = c.1.partition_point(|r| r.0 <= a);
        i.checked_sub(1).map(|i| c.1[i]).filter(|r| a < r.1)
    });
    if let Some((s, e, ok)) = hit {
        return ok.then_some((s, e));
    }
    SPAN_MISSES.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
    let found = shared_run(a, generation);
    if found.is_some() {
        SPAN_SHARED_HITS.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
    }
    let (start, end, ok) = found.unwrap_or_else(|| {
        let st = MEM.lock();
        // (Unchanged while the lock is held: changes bump it inside.)
        let now = MAP_GEN.load(std::sync::atomic::Ordering::Acquire);
        let usable = |v: &Vma| v.kind != VmaKind::Reserved && v.prot != 0;
        let run = match st.find(a) {
            Some(v) if usable(v) => {
                let (mut start, mut end) = (v.start, v.end());
                for _ in 0..64 {
                    match st.vmas.range(..start).next_back().map(|(_, p)| p).filter(|p| p.end() == start && usable(p)) {
                        Some(p) => start = p.start,
                        None => break,
                    }
                }
                for _ in 0..64 {
                    match st.vmas.get(&end).filter(|p| usable(p)) {
                        Some(p) => end = p.end(),
                        None => break,
                    }
                }
                (start, end, true)
            }
            // Reserved or inaccessible: that mapping.
            Some(v) => (v.start, v.end(), false),
            // Between mappings: the gap.
            None => {
                let lo = st.vmas.range(..=a).next_back().map(|(_, p)| p.end()).filter(|&e| e <= a).unwrap_or(a);
                let hi = st.vmas.range(a..).next().map(|(&k, _)| k).filter(|&k| k > a).unwrap_or(a + 1);
                (lo, hi, false)
            }
        };
        share_run(run, now);
        run
    });
    READABLE.with(|c| {
        let mut c = c.borrow_mut();
        if c.0 != generation {
            c.0 = generation;
            c.1.clear();
        }
        // Newer runs replace the ones they overlap.
        c.1.retain(|r| r.1 <= start || end <= r.0);
        let mut i = c.1.partition_point(|r| r.0 < start);
        if c.1.len() >= READABLE_RUNS {
            SPAN_FULL.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
            c.1.clear();
            i = 0;
        }
        c.1.insert(i, (start, end, ok));
    });
    ok.then_some((start, end))
}

/// Whether guest [a, a+len) is accessible (`readable_span`, else `mapped`).
pub fn readable(a: u64, len: u64) -> bool {
    // (R2: the map's "readable" stands; any other answer goes the table's
    // way, which waits for a change in progress.)
    if crate::gpu::tune::r2() && rg_readable(a, len) == Some(true) {
        return true;
    }
    let end = a.saturating_add(len.max(1));
    match readable_span(a) {
        Some((_, e)) if end <= e => true,
        Some(_) => mapped(a, len),
        None => false,
    }
}

/// Guest ranges whose host pages were committed or decommitted since the
/// GPU executor last looked: its imports of guest memory pin the pages they
/// were made with, so an import over a range that was remapped (even with
/// identical bounds, as the title's sceKernelBatchMap streaming does) is stale.
static REPAGED: Mutex<Vec<(u64, u64)>> = Mutex::new(Vec::new());
static REPAGED_PENDING: std::sync::atomic::AtomicBool = std::sync::atomic::AtomicBool::new(false);

/// Every repaged range with its sequence, the last `REPAGE_LOG` of them (`repaged_since`).
static REPAGE_SEQ: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);
static REPAGE_RING: Mutex<std::collections::VecDeque<(u64, u64, u64)>> = Mutex::new(std::collections::VecDeque::new());
const REPAGE_LOG: usize = 8192;

/// The repage sequence now (`repaged_since`).
pub fn repage_seq() -> u64 {
    REPAGE_SEQ.load(std::sync::atomic::Ordering::Acquire)
}

/// Whether guest [lo, hi) had pages committed or decommitted after repage sequence `since`
/// (true too when the log no longer reaches back that far). For the GPU's write-backs stored by
/// the CPU after their batch (`gpu::fe2::release_unimported`): a page remapped meanwhile is
/// another page.
pub fn repaged_since(lo: u64, hi: u64, since: u64) -> bool {
    let r = REPAGE_RING.lock();
    if REPAGE_SEQ.load(std::sync::atomic::Ordering::Acquire) <= since {
        return false;
    }
    if r.front().is_none_or(|&(first, _, _)| first > since + 1) {
        return true;
    }
    r.iter().rev().take_while(|e| e.0 > since).any(|&(_, a, b)| a < hi && lo < b)
}

fn repaged(addr: u64, size: u64) {
    {
        let mut ring = REPAGE_RING.lock();
        let seq = REPAGE_SEQ.load(std::sync::atomic::Ordering::Relaxed) + 1;
        if ring.len() >= REPAGE_LOG {
            ring.pop_front();
        }
        ring.push_back((seq, addr, addr + size));
        REPAGE_SEQ.store(seq, std::sync::atomic::Ordering::Release);
    }
    let mut r = REPAGED.lock();
    if r.len() >= 1024 {
        // Nobody collected them (no GPU work in between): one covering range.
        let (lo, hi) = r.iter().fold((addr, addr + size), |(a, b), &(s, e)| (a.min(s), b.max(e)));
        r.clear();
        r.push((lo, hi));
    } else {
        r.push((addr, addr + size));
    }
    REPAGED_PENDING.store(true, std::sync::atomic::Ordering::Release);
}

/// Takes the ranges whose pages changed since the last call ([start, end)).
pub fn take_repaged() -> Vec<(u64, u64)> {
    if !REPAGED_PENDING.swap(false, std::sync::atomic::Ordering::Acquire) {
        return Vec::new();
    }
    std::mem::take(&mut *REPAGED.lock())
}

/// Whether guest [a, a+len) is covered by accessible mappings (from the
/// mapping table: VirtualQuery walks page tables and is very slow on the
/// title's multi-GiB mappings).
pub fn mapped(a: u64, len: u64) -> bool {
    let st = MEM.lock();
    let end = a.saturating_add(len.max(1));
    let mut p = a;
    while p < end {
        match st.find(p) {
            Some(v) if v.kind != VmaKind::Reserved && v.prot != 0 => p = v.end(),
            _ => return false,
        }
    }
    true
}

/// Whether the mapping table holds [a, a+len) in mappings whose host pages
/// are committed and readable (`win_prot` not PAGE_NOACCESS): what
/// `os_readable` asks the operating system, from the table (bulk readers
/// under `GUEST_READERS` use it from before the lock: `table_view_held`).
pub fn host_readable(a: u64, len: u64) -> bool {
    use crate::platform::os::Win32::System::Memory::PAGE_NOACCESS;
    let st = MEM.lock();
    let end = a.saturating_add(len.max(1));
    let mut p = a;
    while p < end {
        match st.find(p) {
            Some(v) if v.kind != VmaKind::Reserved && win_prot(v.prot) != PAGE_NOACCESS => p = v.end(),
            _ => return false,
        }
    }
    true
}

/// The mapped, readable mapping containing `a` as [start, end), if any.
pub fn mapped_span(a: u64) -> Option<(u64, u64)> {
    let st = MEM.lock();
    st.find(a).filter(|v| v.kind != VmaKind::Reserved && v.prot != 0).map(|v| (v.start, v.end()))
}

pub static MEM: Mutex<MemState> = Mutex::new(MemState {
    vmas: BTreeMap::new(),
    dmem: BTreeMap::new(),
    flexible_used: 0,
    internal_cursor: INTERNAL_BASE,
});

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum MemError {
    NoSpace,
    Invalid,
    Busy,
    HostFailure(u32),
}

pub fn align_up(v: u64, a: u64) -> u64 {
    if a <= 1 { v } else { (v + a - 1) & !(a - 1) }
}
pub fn align_down(v: u64, a: u64) -> u64 {
    if a <= 1 { v } else { v & !(a - 1) }
}

pub fn win_prot(prot: u32) -> u32 {
    let cpu = prot & 7;
    let gpu_rw = prot & (PROT_GPU_READ | PROT_GPU_WRITE) != 0;
    let exec = cpu & PROT_CPU_EXEC != 0;
    let write = cpu & PROT_CPU_WRITE != 0 || gpu_rw;
    let read = cpu != 0 || gpu_rw;
    match (read, write, exec) {
        (_, true, true) => PAGE_EXECUTE_READWRITE,
        (_, false, true) => PAGE_EXECUTE_READ,
        (_, true, false) => PAGE_READWRITE,
        (true, false, false) => PAGE_READONLY,
        (false, false, false) => PAGE_NOACCESS,
    }
}

fn last_error() -> u32 {
    unsafe { crate::platform::os::Win32::Foundation::GetLastError() }
}

static RESERVED_EARLY: std::sync::atomic::AtomicBool = std::sync::atomic::AtomicBool::new(false);
/// The guest range was reserved with write watching (`take_written`).
static WRITE_WATCH: std::sync::atomic::AtomicBool = std::sync::atomic::AtomicBool::new(false);
/// Why a requested write-watched reservation failed (Win32 error; 0: none).
static WRITE_WATCH_ERROR: std::sync::atomic::AtomicU32 = std::sync::atomic::AtomicU32::new(0);

/// Whether environment variable `name` (ASCII) equals `value`, read without
/// allocating (the heap may not be set up yet).
pub fn env_is(name: &str, value: &str) -> bool {
    use crate::platform::os::Win32::System::Environment::GetEnvironmentVariableW;
    let mut n = [0u16; 64];
    if name.len() >= n.len() {
        return false;
    }
    for (d, c) in n.iter_mut().zip(name.bytes()) {
        *d = c as u16;
    }
    let mut buf = [0u16; 64];
    let len = unsafe { GetEnvironmentVariableW(n.as_ptr(), buf.as_mut_ptr(), buf.len() as u32) } as usize;
    len == value.len() && buf[..len].iter().zip(value.bytes()).all(|(&a, b)| a == b as u16)
}

/// Reserves the whole guest range once, allocating nothing: called by
/// `reserve_guest_range` at the start of `main` and by the first heap
/// allocation should one come earlier (`heap`). With write watching (the
/// system notes which pages the CPU writes; `take_written`) unless
/// `DS_WRITE_WATCH=0`, in 64 MiB windows made at first use unless
/// `DS_MIXED=0` (`mixed`).
pub fn reserve_guest_range_early() {
    if RESERVED_EARLY.load(std::sync::atomic::Ordering::Acquire) {
        return;
    }
    // (Something taking part of the range while the process starts may let go of it soon, e.g.
    // a module another program puts into every process, in its own setup: the first call tries
    // again for up to a second (issue #15, 2026-11-25: a player's every shader preparation ended
    // at status 10). Here, at the first allocation and before the heap is chosen: a retry after
    // it would meet the heap's own memory, mimalloc's segments are hinted into 2-6 TiB (the
    // review of 2026-11-25). Sleeping allocates nothing.)
    static TRIED: std::sync::atomic::AtomicBool = std::sync::atomic::AtomicBool::new(false);
    let tries = if TRIED.swap(true, std::sync::atomic::Ordering::AcqRel) { 1 } else { 11 };
    for i in 0..tries {
        if i > 0 {
            std::thread::sleep(std::time::Duration::from_millis(100));
        }
        reserve_guest_range_once();
        if RESERVED_EARLY.load(std::sync::atomic::Ordering::Acquire) {
            return;
        }
    }
}

/// Whether the guest range is reserved (`reserve_guest_range_early`): the heap uses Windows'
/// own when it is not, so a failed start's report names only what really holds the range.
pub fn guest_range_reserved() -> bool {
    RESERVED_EARLY.load(std::sync::atomic::Ordering::Acquire)
}

/// One attempt at the whole range (`reserve_guest_range_early`).
fn reserve_guest_range_once() {
    let size = (GUEST_END - GUEST_BASE) as usize;
    // (`DS_SIM_OCCUPIED=1`, a test: 64 KiB of the range taken first, as another program's
    // module or memory took part of it on a player's PC (issue #15): the start then ends with
    // status 10, naming the region.)
    if env_is("DS_SIM_OCCUPIED", "1") {
        unsafe { VirtualAlloc((GUEST_BASE + 0x1000_0000) as _, 0x1_0000, MEM_RESERVE, PAGE_NOACCESS) };
        crate::platform::memory_diagnostics::sim_module();
    }
    // Section-backed guest memory (no write watching: views have none).
    if env_is("DS_SECTIONS", "1") && crate::sections::reserve(GUEST_BASE, size as u64, DMEM_SIZE, ANON_SIZE).is_ok() {
        RESERVED_EARLY.store(true, std::sync::atomic::Ordering::Release);
        return;
    }
    // The mixed range (`mixed`): one placeholder, windows made at first use.
    if !env_is("DS_MIXED", "0") && !env_is("DS_WRITE_WATCH", "0") {
        let p = unsafe { VirtualAlloc2(std::ptr::null_mut(), GUEST_BASE as _, size, MEM_RESERVE | MEM_RESERVE_PLACEHOLDER, PAGE_NOACCESS, std::ptr::null_mut(), 0) };
        if !p.is_null() {
            if let Some(s) = env_u32("DS_MIXED_SHIFT").filter(|s| (MIXED_MIN_SHIFT..=30).contains(s)) {
                MIXED_SHIFT.store(s, std::sync::atomic::Ordering::Relaxed);
            }
            MIXED.store(true, std::sync::atomic::Ordering::Release);
            WRITE_WATCH.store(true, std::sync::atomic::Ordering::Release);
            RESERVED_EARLY.store(true, std::sync::atomic::Ordering::Release);
            return;
        }
        WRITE_WATCH_ERROR.store(last_error(), std::sync::atomic::Ordering::Release);
    }
    if !env_is("DS_WRITE_WATCH", "0") {
        let p = unsafe { VirtualAlloc(GUEST_BASE as _, size, MEM_RESERVE | MEM_WRITE_WATCH, PAGE_NOACCESS) };
        if !p.is_null() {
            WRITE_WATCH.store(true, std::sync::atomic::Ordering::Release);
            RESERVED_EARLY.store(true, std::sync::atomic::Ordering::Release);
            return;
        }
        WRITE_WATCH_ERROR.store(last_error(), std::sync::atomic::Ordering::Release);
    }
    let p = unsafe { VirtualAlloc(GUEST_BASE as _, size, MEM_RESERVE, PAGE_NOACCESS) };
    if !p.is_null() {
        RESERVED_EARLY.store(true, std::sync::atomic::Ordering::Release);
    }
}

/// VirtualAlloc's MEM_WRITE_WATCH and GetWriteWatch's WRITE_WATCH_FLAG_RESET
/// (not named by windows-sys).
const MEM_WRITE_WATCH: u32 = 0x0020_0000;
const WRITE_WATCH_FLAG_RESET: u32 = 1;

/// The mixed guest range (default; `DS_MIXED=0`, boot: one write-watched
/// reservation as before), first stage (FRONTEND2.md "GPU-quiet mappings on
/// sections"): the guest range is one placeholder,
/// and each 64 MiB window of it (`mixed_shift`: front end 2's chunks, the
/// import windows) is replaced at its first commit by a reserve-only private
/// allocation with write watching, in which pages are committed, decommitted
/// and protected as in the one reservation otherwise. No host call or
/// write-watch query may span two allocations (commits and protection
/// changes fail with error 487, decommits with 87:
/// `sections::tests::windowed_private_allocations`): they split at window
/// boundaries (`host_pieces`), and a take skips windows never made. Every
/// mapping is private so far (views for GPU-quiet mappings come next).
/// Write-watch queries in different allocations do not wait for each other
/// (in one they queue in the kernel, ~10 us more for each in flight): +5.3%
/// for front end 2 at the vista, +5.0% for the default (FINDINGS.md
/// 2026-10-29).
static MIXED: std::sync::atomic::AtomicBool = std::sync::atomic::AtomicBool::new(false);
/// The window size as a power of two (`DS_MIXED_SHIFT=<21..30>`, boot; 26:
/// 64 MiB).
static MIXED_SHIFT: std::sync::atomic::AtomicU32 = std::sync::atomic::AtomicU32::new(26);
const MIXED_MIN_SHIFT: u32 = 21;
const MIXED_MAX_WINDOWS: usize = ((GUEST_END - GUEST_BASE) >> MIXED_MIN_SHIFT) as usize;
/// The windows made (their allocation in place), a bit each.
static MIXED_MADE: [std::sync::atomic::AtomicU64; MIXED_MAX_WINDOWS.div_ceil(64)] = [const { std::sync::atomic::AtomicU64::new(0) }; MIXED_MAX_WINDOWS.div_ceil(64)];
/// Serializes the making of windows.
static MIXED_LOCK: Mutex<()> = Mutex::new(());
/// [windows made, failures to make one, host calls split at a window
/// boundary, takes split].
static MIXED_STATS: [std::sync::atomic::AtomicU64; 4] = [const { std::sync::atomic::AtomicU64::new(0) }; 4];
/// Takes split at window boundaries by caller (`WW_WHO_NAMES`).
static MIXED_SPLIT_WHO: [std::sync::atomic::AtomicU64; 10] = [const { std::sync::atomic::AtomicU64::new(0) }; 10];
const _: () = assert!(GUEST_BASE % (1 << 30) == 0 && GUEST_END % (1 << 30) == 0);

/// The windows' size as a power of two (`MIXED_SHIFT`).
#[inline]
fn mixed_shift() -> u32 {
    MIXED_SHIFT.load(std::sync::atomic::Ordering::Relaxed)
}

/// The windows' size.
#[inline]
fn mixed_window() -> u64 {
    1 << mixed_shift()
}

/// Environment variable `name` (ASCII) as a decimal number, read without
/// allocating (`env_is`).
fn env_u32(name: &str) -> Option<u32> {
    use crate::platform::os::Win32::System::Environment::GetEnvironmentVariableW;
    let mut n = [0u16; 64];
    if name.len() >= n.len() {
        return None;
    }
    for (d, c) in n.iter_mut().zip(name.bytes()) {
        *d = c as u16;
    }
    let mut buf = [0u16; 16];
    let len = unsafe { GetEnvironmentVariableW(n.as_ptr(), buf.as_mut_ptr(), buf.len() as u32) } as usize;
    if len == 0 || len >= buf.len() {
        return None;
    }
    let mut v = 0u32;
    for &c in &buf[..len] {
        let d = (c as u32).checked_sub('0' as u32).filter(|d| *d < 10)?;
        v = v.checked_mul(10)?.checked_add(d)?;
    }
    Some(v)
}

/// The mixed guest range is on (unless `DS_MIXED=0`; set at the reservation).
pub fn mixed() -> bool {
    MIXED.load(std::sync::atomic::Ordering::Acquire)
}

/// Whether the window holding guest address `a` has its allocation.
fn mixed_made(a: u64) -> bool {
    if !(GUEST_BASE..GUEST_END).contains(&a) {
        return false;
    }
    let w = ((a - GUEST_BASE) >> mixed_shift()) as usize;
    MIXED_MADE[w / 64].load(std::sync::atomic::Ordering::Acquire) >> (w % 64) & 1 != 0
}

/// Makes the allocations of the windows overlapping guest [a, e) that have
/// none (their placeholders split off and replaced); the Win32 error.
fn mixed_make(a: u64, e: u64) -> Result<(), u32> {
    use std::sync::atomic::Ordering;
    let win = mixed_window();
    let mut w = a & !(win - 1);
    while w < e {
        if !mixed_made(w) && !view_window(w) {
            let _g = MIXED_LOCK.lock();
            if !mixed_made(w) {
                if let Err(err) = crate::sections::isolate(w, win) {
                    MIXED_STATS[1].fetch_add(1, Ordering::Relaxed);
                    crate::warn!("mem: making the mixed range's window {w:#x}: {err}");
                    return Err(0);
                }
                let p = unsafe { VirtualAlloc2(std::ptr::null_mut(), w as _, win as usize, MEM_RESERVE | MEM_REPLACE_PLACEHOLDER | MEM_WRITE_WATCH, PAGE_NOACCESS, std::ptr::null_mut(), 0) };
                if p.is_null() {
                    let err = last_error();
                    MIXED_STATS[1].fetch_add(1, Ordering::Relaxed);
                    crate::warn!("mem: making the mixed range's window {w:#x}: error {err}");
                    return Err(err);
                }
                let i = ((w - GUEST_BASE) >> mixed_shift()) as usize;
                MIXED_MADE[i / 64].fetch_or(1 << (i % 64), Ordering::AcqRel);
                MIXED_STATS[0].fetch_add(1, Ordering::Relaxed);
                if let Some(p) = crate::gpu::fe2::traced(w, w + win) {
                    crate::gpu::fe2::trace_event(p, &format!("its mixed window {w:#x} made"));
                }
            }
        }
        w += win;
    }
    Ok(())
}

/// The pieces of guest [a, e) one host memory call may take: all of it, or
/// with the mixed range (`mixed`) its part in each window.
pub(crate) fn host_pieces(a: u64, e: u64) -> impl Iterator<Item = (u64, u64)> {
    if mixed() && e > a && a >> mixed_shift() != (e - 1) >> mixed_shift() {
        MIXED_STATS[2].fetch_add(1, std::sync::atomic::Ordering::Relaxed);
    }
    window_pieces(a, e)
}

/// `host_pieces` without its count (takes count their own).
fn window_pieces(a: u64, e: u64) -> impl Iterator<Item = (u64, u64)> {
    let split = mixed();
    let win = mixed_window();
    let mut p = a;
    std::iter::from_fn(move || {
        if p >= e {
            return None;
        }
        let q = if split { ((p | (win - 1)) + 1).min(e) } else { e };
        let r = (p, q);
        p = q;
        Some(r)
    })
}

/// `DS_MIXED_VIEWS=<name>:<hex size>,...` (boot; the mixed range): direct
/// mappings of that name and size map their whole windows that are still
/// placeholders as views of the direct-memory section at their physical
/// offset, each with an alias view that GPU imports are made of (FRONTEND2.md
/// "GPU-quiet mappings on sections", step 2); their other parts stay private.
/// Takes in view windows are the residency's protection tracking through the
/// host (`gpu::resident::take_written`), and front end 2 may arm the
/// GPU-written pages there (`fe2::try_arm`): the pages the GPU writes through
/// its imports are never the ones protected.
fn view_rules() -> &'static [(String, u64)] {
    static R: std::sync::OnceLock<Vec<(String, u64)>> = std::sync::OnceLock::new();
    R.get_or_init(|| {
        let Ok(v) = std::env::var("DS_MIXED_VIEWS") else { return Vec::new() };
        v.split(',')
            .filter_map(|r| {
                let (name, size) = r.split_once(':')?;
                Some((name.trim().to_string(), u64::from_str_radix(size.trim().trim_start_matches("0x"), 16).ok()?))
            })
            .collect()
    })
}

/// The view windows (`view_rules`), a bit each.
static MIXED_VIEW: [std::sync::atomic::AtomicU64; MIXED_MAX_WINDOWS.div_ceil(64)] = [const { std::sync::atomic::AtomicU64::new(0) }; MIXED_MAX_WINDOWS.div_ceil(64)];
/// A window was made a view: the per-window checks run from then on.
static VIEWS_ON: std::sync::atomic::AtomicBool = std::sync::atomic::AtomicBool::new(false);
/// Each view window's alias slot.
static VIEW_SLOTS: Mutex<BTreeMap<u64, u8>> = Mutex::new(BTreeMap::new());
/// [windows made views, released, made private again, windows of chosen
/// mappings left private (made already), takes of view windows, pages they
/// reported, their ns].
static VIEW_STATS: [std::sync::atomic::AtomicU64; 7] = [const { std::sync::atomic::AtomicU64::new(0) }; 7];

/// Some window of the mixed range is (or was) a view (`view_rules`).
#[inline]
pub fn views_on() -> bool {
    VIEWS_ON.load(std::sync::atomic::Ordering::Acquire)
}

/// Whether guest address `a` lies in a view window.
#[inline]
pub fn view_window(a: u64) -> bool {
    if !views_on() || !(GUEST_BASE..GUEST_END).contains(&a) {
        return false;
    }
    let w = ((a - GUEST_BASE) >> mixed_shift()) as usize;
    MIXED_VIEW[w / 64].load(std::sync::atomic::Ordering::Acquire) >> (w % 64) & 1 != 0
}

/// Whether every window of [a, e) is a view window.
pub fn view_span(a: u64, e: u64) -> bool {
    if !views_on() || e <= a {
        return false;
    }
    let win = mixed_window();
    let mut w = a & !(win - 1);
    while w < e {
        if !view_window(w) {
            return false;
        }
        w += win;
    }
    true
}

/// [lo, hi) clipped to the windows around `a` backed as its own is (views or
/// private memory): one GPU import cannot be of both.
pub fn same_backing(a: u64, lo: u64, hi: u64) -> (u64, u64) {
    if !views_on() {
        return (lo, hi);
    }
    let win = mixed_window();
    let v = view_window(a);
    let mut s = a & !(win - 1);
    while s > lo && s > GUEST_BASE && view_window(s - win) == v {
        s -= win;
    }
    let mut e = (a & !(win - 1)) + win;
    while e < hi && e < GUEST_END && view_window(e) == v {
        e += win;
    }
    (lo.max(s), hi.min(e))
}

fn set_view(w: u64, on: bool) {
    use std::sync::atomic::Ordering;
    let i = ((w - GUEST_BASE) >> mixed_shift()) as usize;
    if on {
        MIXED_VIEW[i / 64].fetch_or(1 << (i % 64), Ordering::AcqRel);
    } else {
        MIXED_VIEW[i / 64].fetch_and(!(1 << (i % 64)), Ordering::AcqRel);
    }
}

/// The parts of guest [a, e) outside view windows.
fn private_parts(a: u64, e: u64) -> Vec<(u64, u64)> {
    let mut out: Vec<(u64, u64)> = Vec::new();
    for (p, q) in window_pieces(a, e) {
        if view_window(p) {
            continue;
        }
        match out.last_mut() {
            Some(last) if last.1 == p => last.1 = q,
            _ => out.push((p, q)),
        }
    }
    out
}

/// Maps the whole windows of the chosen direct mapping [addr, addr+size)
/// (physical `phys`) that are still placeholders as views with aliases
/// (`view_rules`); returns how many.
fn mixed_views_make(addr: u64, size: u64, phys: u64, prot: u32) -> u64 {
    use crate::sections::{self as sec, Sec};
    use std::sync::atomic::Ordering;
    let win = mixed_window();
    let (mut w, end) = (addr.next_multiple_of(win), (addr + size) & !(win - 1));
    static N: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);
    if N.fetch_add(1, Ordering::Relaxed) < 16 {
        crate::info!("mem: chosen mapping {addr:#x}+{size:#x}: whole windows {w:#x}..{end:#x} (sections made: {})", sec::ready());
    }
    if w >= end {
        return 0;
    }
    if !sec::ready() {
        if let Err(e) = sec::reserve_aliases(GUEST_BASE, GUEST_END - GUEST_BASE, DMEM_SIZE, ANON_SIZE) {
            crate::warn!("mem: the mixed range's views need the sections and alias ranges: error {e} (mappings stay private)");
            return 0;
        }
    }
    let mut made = 0;
    while w < end {
        if mixed_made(w) || view_window(w) {
            VIEW_STATS[3].fetch_add(1, Ordering::Relaxed);
            w += win;
            continue;
        }
        let off = phys + (w - addr);
        mapped_over_gpu_writes(w, win, true, off);
        let _arm = crate::gpu::resident::remap_guard();
        match sec::map_new_view(w, win, Sec::Dmem, off, win_prot(prot)) {
            Ok(()) => match sec::map_alias(w, win, Sec::Dmem, off) {
                Ok(slot) => {
                    VIEW_SLOTS.lock().insert(w, slot);
                    VIEWS_ON.store(true, Ordering::Release);
                    set_view(w, true);
                    crate::gpu::resident::settled(w, win);
                    VIEW_STATS[0].fetch_add(1, Ordering::Relaxed);
                    made += 1;
                }
                Err(e) => {
                    crate::warn!("mem: the mixed range's view at {w:#x}: {e} (the window stays private)");
                    let _ = sec::unmap_view(w);
                }
            },
            Err(e) => crate::warn!("mem: the mixed range's view at {w:#x}: {e} (the window stays private)"),
        }
        w += win;
    }
    made
}

/// Unmaps view window `w`: its guest view now (a placeholder again), its
/// alias once no GPU import may use it (`sections::alias_unref`).
fn mixed_view_release(w: u64) {
    let win = mixed_window();
    {
        let _readers_out = GUEST_READERS.write();
        let _arm = crate::gpu::resident::remap_guard();
        if let Err(e) = crate::sections::unmap_view(w) {
            crate::warn!("mem: {e}");
        }
        set_view(w, false);
        crate::gpu::resident::settled(w, win);
    }
    if let Some(slot) = VIEW_SLOTS.lock().remove(&w) {
        crate::sections::alias_unref(w, win, slot, crate::gpu::imports_may_cover);
    }
    VIEW_STATS[1].fetch_add(1, std::sync::atomic::Ordering::Relaxed);
}

/// View window `w` becomes private memory with the same contents (its
/// mapping is split inside it: views cannot be unmapped in part): GPU work
/// writing it through its alias is waited for, its pages copied out, the
/// view released, the window made private and its pages committed with the
/// copy and the mapping's protection `prot`.
fn mixed_view_privatize(w: u64, prot: u32) {
    use std::sync::atomic::Ordering;
    let win = mixed_window();
    if let Some(seq) = crate::gpu::exec::pending_writes(w, win) {
        if !crate::gpu::exec::wait_done_for(seq, std::time::Duration::from_secs(2)) {
            crate::warn!("mem: view window {w:#x} made private while batch {seq} may still write it");
        }
    }
    let mut copy = vec![0u8; win as usize];
    {
        let _readers_out = GUEST_READERS.write();
        let _arm = crate::gpu::resident::remap_guard();
        // (Readable first: the guest's protection may forbid our reads.)
        let mut old = 0u32;
        unsafe {
            VirtualProtect(w as _, win as usize, PAGE_READONLY, &mut old);
            std::ptr::copy_nonoverlapping(w as *const u8, copy.as_mut_ptr(), win as usize);
        }
        if let Err(e) = crate::sections::unmap_view(w) {
            crate::warn!("mem: {e}");
        }
        set_view(w, false);
        crate::gpu::resident::settled(w, win);
    }
    if let Some(slot) = VIEW_SLOTS.lock().remove(&w) {
        crate::sections::alias_unref(w, win, slot, crate::gpu::imports_may_cover);
    }
    let ok = mixed_make(w, w + win).is_ok() && {
        let _readers_out = GUEST_READERS.write();
        let _arm = crate::gpu::resident::remap_guard();
        let p = unsafe { VirtualAlloc(w as _, win as usize, MEM_COMMIT, PAGE_READWRITE) };
        if !p.is_null() {
            let mut old = 0u32;
            unsafe {
                std::ptr::copy_nonoverlapping(copy.as_ptr(), w as *mut u8, win as usize);
                VirtualProtect(w as _, win as usize, win_prot(prot), &mut old);
            }
        }
        crate::gpu::resident::settled(w, win);
        !p.is_null()
    };
    let n = VIEW_STATS[2].fetch_add(1, Ordering::Relaxed);
    if !ok {
        crate::error!("mem: view window {w:#x} could not be made private: its contents are lost");
    } else if n < 8 {
        crate::info!("mem: view window {w:#x} made private (its mapping split inside it)");
    }
}

/// The views' part of the mixed range's state.
fn views_state() -> String {
    let v: Vec<u64> = VIEW_STATS.iter().map(|c| c.load(std::sync::atomic::Ordering::Relaxed)).collect();
    format!(
        "; views ({}): {} windows made ({} MiB live), {} released, {} made private again, {} windows of chosen mappings left private; {} takes of view windows ({:.2} us each) reported {} pages{}",
        view_rules().iter().map(|(n, s)| format!("'{n}' of {} MiB", s >> 20)).collect::<Vec<_>>().join(", "),
        v[0],
        VIEW_SLOTS.lock().len() as u64 * (mixed_window() >> 20),
        v[1],
        v[2],
        v[3],
        v[4],
        v[6] as f64 / v[4].max(1) as f64 / 1000.0,
        v[5],
        crate::gpu::resident::view_census()
    )
}

/// The mixed range's state, for the start-up log and the status.
fn mixed_state() -> String {
    let v: Vec<u64> = MIXED_STATS.iter().map(|c| c.load(std::sync::atomic::Ordering::Relaxed)).collect();
    let who: Vec<String> = WW_WHO_NAMES.iter().zip(MIXED_SPLIT_WHO.iter()).map(|(n, c)| (n, c.load(std::sync::atomic::Ordering::Relaxed))).filter(|(_, c)| *c > 0).map(|(n, c)| format!("{n} {c}")).collect();
    let views = if view_rules().is_empty() { String::new() } else { views_state() };
    format!("mixed range: {} windows of {} MiB made ({} failed), {} host calls and {} takes split at window boundaries (takes by caller: {}){views}", v[0], mixed_window() >> 20, v[1], v[2], v[3], who.join(", "))
}

/// Whether CPU writes to the guest range are tracked (`take_written`):
/// write watching, or page protection for section-backed memory.
pub fn write_watched() -> bool {
    WRITE_WATCH.load(std::sync::atomic::Ordering::Acquire) || crate::sections::on()
}

/// Write watching of the guest range, for the start-up log.
pub fn write_watch_state() -> String {
    match (write_watched(), WRITE_WATCH_ERROR.load(std::sync::atomic::Ordering::Acquire)) {
        (true, _) if mixed() => format!("on ({})", mixed_state()),
        (true, _) => "on".into(),
        (false, 0) => "off".into(),
        (false, e) => format!("off (reserving with write watch failed: error {e})"),
    }
}

/// Statistics of `take_written`: [calls, ns, pages reported].
pub static WRITE_WATCH_STATS: [std::sync::atomic::AtomicU64; 3] = [const { std::sync::atomic::AtomicU64::new(0) }; 3];
/// Takes that the probe query found clean (`tune ww_probe`).
pub static PROBE_CLEAN: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);
/// `take_written` by caller (`WW_WHO_NAMES`) and by size (log2 of the 4 KiB
/// pages, the last bucket 2^20 and more): [calls, ns, found clean by the
/// probe] (profile windows).
pub static WW_BY_WHO: [[std::sync::atomic::AtomicU64; 3]; 10] = [const { [const { std::sync::atomic::AtomicU64::new(0) }; 3] }; 10];
pub static WW_BY_SIZE: [[std::sync::atomic::AtomicU64; 3]; 21] = [const { [const { std::sync::atomic::AtomicU64::new(0) }; 3] }; 21];
pub const WW_WHO_NAMES: [&str; 10] = ["copy-cache checks", "copy-cache inserts", "homes watch", "texture scrubbing", "texture re-uploads", "homes candidates", "shadow windows", "other", "resident data", "front end 2"];
/// `take_written` for front end 2's placements (`gpu::fe2`).
pub const WW_WHO_FE2: usize = 9;

/// Queries in flight (`InFlight`), and the calls and ns by how many other
/// queries were in flight when each began (0..7, 8 or more).
static WW_INFLIGHT: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);
pub static WW_BY_CONC: [[std::sync::atomic::AtomicU64; 3]; 9] = [const { [const { std::sync::atomic::AtomicU64::new(0) }; 3] }; 9];

/// A query in flight (counted while it lives); `others`: those in flight
/// when it began.
struct InFlight {
    others: u64,
}

impl InFlight {
    fn new() -> InFlight {
        InFlight { others: WW_INFLIGHT.fetch_add(1, std::sync::atomic::Ordering::Relaxed) }
    }
}

impl Drop for InFlight {
    fn drop(&mut self) {
        WW_INFLIGHT.fetch_sub(1, std::sync::atomic::Ordering::Relaxed);
    }
}

/// The slow ones by caller: [of 1 ms or more, of 5 ms or more, their ns,
/// the longest's ns] (stutters' takes of a few pages took 12-16 ms at
/// Castlewall's arrival, FINDINGS.md 2026-10-31).
pub static WW_TAIL: [[std::sync::atomic::AtomicU64; 4]; 10] = [const { [const { std::sync::atomic::AtomicU64::new(0) }; 4] }; 10];

fn ww_note(who: usize, pages: u64, ns: u64, clean: bool, others: u64) {
    use std::sync::atomic::Ordering::Relaxed;
    if ns >= 1_000_000 {
        let t = &WW_TAIL[who.min(9)];
        t[0].fetch_add(1, Relaxed);
        if ns >= 5_000_000 {
            t[1].fetch_add(1, Relaxed);
        }
        t[2].fetch_add(ns, Relaxed);
        t[3].fetch_max(ns, Relaxed);
    }
    let b = (64 - pages.max(1).leading_zeros() as usize - 1).min(20);
    for s in [&WW_BY_WHO[who.min(9)], &WW_BY_SIZE[b], &WW_BY_CONC[(others as usize).min(8)]] {
        s[0].fetch_add(1, Relaxed);
        s[1].fetch_add(ns, Relaxed);
        s[2].fetch_add(clean as u64, Relaxed);
    }
}

/// The breakdown of `take_written` since the last profile reset.
pub fn ww_lines() -> String {
    use std::sync::atomic::Ordering::Relaxed;
    let f = |s: &[std::sync::atomic::AtomicU64; 3]| {
        let n = s[0].load(Relaxed);
        format!("{n} ({:.1} us, {} clean)", s[1].load(Relaxed) as f64 / n.max(1) as f64 / 1000.0, s[2].load(Relaxed))
    };
    let who: Vec<String> = WW_WHO_NAMES.iter().zip(WW_BY_WHO.iter()).filter(|(_, s)| s[0].load(Relaxed) > 0).map(|(n, s)| format!("{n} {}", f(s))).collect();
    let size: Vec<String> = WW_BY_SIZE.iter().enumerate().filter(|(_, s)| s[0].load(Relaxed) > 0).map(|(k, s)| format!("{}+ {}", 1u64 << k, f(s))).collect();
    let conc: Vec<String> = WW_BY_CONC.iter().enumerate().filter(|(_, s)| s[0].load(Relaxed) > 0).map(|(k, s)| format!("{k}{} {}", if k == 8 { "+" } else { "" }, f(s))).collect();
    let tail: Vec<String> = WW_WHO_NAMES
        .iter()
        .zip(WW_TAIL.iter())
        .filter(|(_, t)| t[0].load(Relaxed) > 0)
        .map(|(n, t)| format!("{n} {} (5 ms or more {}; {} ms, longest {:.1} ms)", t[0].load(Relaxed), t[1].load(Relaxed), t[2].load(Relaxed) / 1_000_000, t[3].load(Relaxed) as f64 / 1e6))
        .collect();
    format!("    write-watch queries by caller: {}\n    by pages: {}\n    by other queries in flight when one began: {}\n    of 1 ms or more by caller: {}\n", who.join(", "), size.join(", "), conc.join(", "), tail.join(", ")) + &ww_census_lines()
}

pub fn ww_reset() {
    use std::sync::atomic::Ordering::Relaxed;
    for c in WW_BY_WHO.iter().chain(WW_BY_SIZE.iter()).chain(WW_BY_CONC.iter()).flatten().chain(WW_TAIL.iter().flatten()) {
        c.store(0, Relaxed);
    }
}

/// The 4 KiB pages of guest [base, base+size) the CPU wrote since they were
/// last taken (their write marks are reset), into `out` (page addresses);
/// None without write watching or when the query fails (treat as written).
/// `who`: the caller (`WW_WHO_NAMES`; statistics).
/// Whether the CPU wrote guest page `page` (4 KiB) since its last take,
/// without resetting anything (a diagnosis's look: `DS_IMPORT_AUDIT_GPU`).
/// None: not write-watched memory or the query failed.
pub fn written_since_take(page: u64) -> Option<bool> {
    let mut addrs = [std::ptr::null_mut::<std::ffi::c_void>(); 1];
    let mut count: usize = 1;
    let mut gran = 0u32;
    let r = unsafe { GetWriteWatch(0, (page & !4095) as _, 4096, addrs.as_mut_ptr(), &mut count, &mut gran) };
    (r == 0).then_some(count != 0)
}

pub fn take_written(base: u64, size: u64, out: &mut Vec<u64>, who: usize) -> Option<()> {
    let first = out.len();
    let r = take_written_census(base, size, out, who);
    if ww_census_on() && out.len() > first {
        ww_census(&out[first..]);
    }
    // (Pages found written lose their zero-known bits before the next zero store over them:
    // the take resets what the store's own query would see.)
    if out.len() > first {
        crate::gpu::zk::cpu_written(&out[first..]);
    }
    r
}

/// The pages of guest [lo, hi) written since their last take, into `out` (page addresses in
/// order), resetting nothing (`gpu::zk`: the pages of a zero store the CPU wrote); None where a
/// query cannot tell (no write watching, section-backed memory, a view window, whose writes
/// faults tell).
pub fn written_pages(lo: u64, hi: u64, out: &mut Vec<u64>) -> Option<()> {
    if !write_watched() || crate::sections::on() || hi <= lo {
        return None;
    }
    let mut addrs = vec![std::ptr::null_mut::<std::ffi::c_void>(); 4096];
    let mut query = |a: u64, e: u64, out: &mut Vec<u64>| -> Option<()> {
        let mut at = a;
        while at < e {
            let mut count: usize = addrs.len();
            let mut gran = 0u32;
            if unsafe { GetWriteWatch(0, at as _, (e - at) as usize, addrs.as_mut_ptr(), &mut count, &mut gran) } != 0 {
                return None;
            }
            out.extend(addrs[..count].iter().map(|&p| p as u64));
            if count < addrs.len() {
                break;
            }
            at = addrs[count - 1] as u64 + 4096;
        }
        Some(())
    };
    let (lo, hi) = (lo & !4095, hi.next_multiple_of(4096));
    if mixed() {
        for (a, e) in window_pieces(lo, hi) {
            if view_window(a) {
                return None;
            }
            if mixed_made(a) {
                query(a, e, out)?;
            }
        }
    } else {
        query(lo, hi, out)?;
    }
    Some(())
}

/// `DS_WW_CENSUS=1` (diagnosis; docs/FRONTEND3.md, sizing a take per title
/// submission): where front end 2's per-binding takes find the CPU's writes,
/// by 64 MiB window, and how many windows each epoch's writes touch (epochs
/// end at title submissions and CPU-released waits, `precache::epoch`).
fn ww_census_on() -> bool {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ON.get_or_init(|| std::env::var("DS_WW_CENSUS").ok().as_deref() == Some("1"))
}

/// window -> pages found written; the current epoch, its windows and pages;
/// closed epochs with writes: [count, windows summed, pages summed] and a
/// histogram of their windows (1, 2, 3, 4-7, 8+); pages each closed epoch.
struct WwCensus {
    by_window: std::collections::HashMap<u64, u64>,
    /// 1 MiB granule -> pages found written.
    by_mib: std::collections::HashMap<u64, u64>,
    epoch: u64,
    epoch_windows: std::collections::HashSet<u64>,
    epoch_pages: u64,
    closed: [u64; 3],
    hist: [u64; 5],
    pages_in_epochs: Vec<u64>,
}

static WW_CENSUS: Mutex<Option<WwCensus>> = Mutex::new(None);

fn ww_census(pages: &[u64]) {
    let ep = crate::gpu::precache::epoch();
    let mut g = WW_CENSUS.lock();
    let c = g.get_or_insert_with(|| WwCensus { by_window: Default::default(), by_mib: Default::default(), epoch: ep, epoch_windows: Default::default(), epoch_pages: 0, closed: [0; 3], hist: [0; 5], pages_in_epochs: Vec::new() });
    if c.epoch != ep {
        if !c.epoch_windows.is_empty() {
            let n = c.epoch_windows.len() as u64;
            c.closed[0] += 1;
            c.closed[1] += n;
            c.closed[2] += c.epoch_pages;
            c.hist[match n {
                1 => 0,
                2 => 1,
                3 => 2,
                4..=7 => 3,
                _ => 4,
            }] += 1;
            if c.pages_in_epochs.len() < 1 << 16 {
                c.pages_in_epochs.push(c.epoch_pages);
            }
        }
        c.epoch = ep;
        c.epoch_windows.clear();
        c.epoch_pages = 0;
    }
    for &p in pages {
        *c.by_window.entry(p >> 26).or_insert(0) += 1;
        *c.by_mib.entry(p >> 20).or_insert(0) += 1;
        c.epoch_windows.insert(p >> 26);
    }
    c.epoch_pages += pages.len() as u64;
}

/// The census as a status line (`DS_WW_CENSUS=1`).
pub fn ww_census_lines() -> String {
    if !ww_census_on() {
        return String::new();
    }
    let g = WW_CENSUS.lock();
    let Some(c) = g.as_ref() else { return String::new() };
    let flips = crate::gpu::FRAMES.load(std::sync::atomic::Ordering::Relaxed).max(1) as f64;
    let mut w: Vec<(u64, u64)> = c.by_window.iter().map(|(&k, &v)| (k, v)).collect();
    w.sort_by(|a, b| b.1.cmp(&a.1));
    let total: u64 = w.iter().map(|x| x.1).sum();
    let top: Vec<String> = w.iter().take(12).map(|(k, v)| format!("{:#x} {:.1}", k << 26, *v as f64 / flips)).collect();
    let mut m: Vec<u64> = c.by_mib.values().copied().collect();
    m.sort_unstable_by(|a, b| b.cmp(a));
    let cover = |share: f64| {
        let (mut acc, mut k) = (0u64, 0usize);
        while k < m.len() && (acc as f64) < share * total as f64 {
            acc += m[k];
            k += 1;
        }
        k
    };
    let mib = format!("; 1 MiB granules written: {}, holding half the pages {}, 90% {}, 99% {}", m.len(), cover(0.5), cover(0.9), cover(0.99));
    let mut pe = c.pages_in_epochs.clone();
    pe.sort_unstable();
    let pct = |q: f64| pe.get(((pe.len().max(1) as f64 - 1.0) * q).round() as usize).copied().unwrap_or(0);
    format!(
        "    write-watch census (since boot, {flips:.0} flips): pages found written {:.1} a flip in {} windows of 64 MiB, a flip by window: {}; epochs with writes {:.1} a flip, windows each {:.2} on average (1: {}, 2: {}, 3: {}, 4-7: {}, 8+: {}), pages each p50 {} p90 {} max {}\n",
        total as f64 / flips,
        w.len(),
        top.join(", "),
        c.closed[0] as f64 / flips,
        c.closed[1] as f64 / c.closed[0].max(1) as f64,
        c.hist[0],
        c.hist[1],
        c.hist[2],
        c.hist[3],
        c.hist[4],
        pct(0.5),
        pct(0.9),
        pe.last().copied().unwrap_or(0)
    )
    .replace("
", &(mib + "
"))
}

fn take_written_census(base: u64, size: u64, out: &mut Vec<u64>, who: usize) -> Option<()> {
    use std::sync::atomic::Ordering::Relaxed;
    if crate::sections::on() {
        let inflight = InFlight::new();
        let t = crate::gpu::record_opts::Timer::start();
        let r = crate::gpu::resident::take_written(base, size, out);
        let ns = t.elapsed().as_nanos() as u64;
        ww_note(who, size.div_ceil(4096), ns, out.is_empty(), inflight.others);
        WRITE_WATCH_STATS[0].fetch_add(1, Relaxed);
        WRITE_WATCH_STATS[1].fetch_add(ns, Relaxed);
        WRITE_WATCH_STATS[2].fetch_add(out.len() as u64, Relaxed);
        return r;
    }
    if !write_watched() || size == 0 {
        return None;
    }
    let (lo, hi) = (base & !4095, (base + size).next_multiple_of(4096));
    take_written_inner(lo, hi, out, who)
}

fn take_written_inner(lo: u64, hi: u64, out: &mut Vec<u64>, who: usize) -> Option<()> {
    if mixed() {
        // One query per window's allocation (none may span two); a window
        // never made holds no written page.
        if lo < GUEST_BASE || hi > GUEST_END {
            return None;
        }
        let split = lo >> mixed_shift() != (hi - 1) >> mixed_shift();
        if split {
            MIXED_STATS[3].fetch_add(1, std::sync::atomic::Ordering::Relaxed);
            MIXED_SPLIT_WHO[who.min(9)].fetch_add(1, std::sync::atomic::Ordering::Relaxed);
        }
        if split || views_on() {
            for (a, e) in window_pieces(lo, hi) {
                if view_window(a) {
                    // (Views have no write watching: their pages are armed
                    // through the host and faults tell their writes.)
                    let (first, t) = (out.len(), crate::gpu::record_opts::Timer::start());
                    crate::gpu::resident::take_written(a, e - a, out)?;
                    VIEW_STATS[4].fetch_add(1, std::sync::atomic::Ordering::Relaxed);
                    VIEW_STATS[5].fetch_add((out.len() - first) as u64, std::sync::atomic::Ordering::Relaxed);
                    VIEW_STATS[6].fetch_add(t.elapsed().as_nanos() as u64, std::sync::atomic::Ordering::Relaxed);
                } else if mixed_made(a) {
                    take_written_query(a, e, out, who)?;
                }
            }
            return Some(());
        }
        if !mixed_made(lo) {
            return Some(());
        }
    }
    take_written_query(lo, hi, out, who)
}

/// Front end 4's census (`engine::arena_scan_census`): pages of [lo, hi)
/// written since their last take, by write-watch queries that reset nothing
/// (one per window piece; front end 2's takes see the same pages after it).
/// (pages, queries); None where the memory is not write-watched.
pub fn query_written(lo: u64, hi: u64) -> Option<(u64, u64)> {
    if !write_watched() || crate::sections::on() || hi <= lo {
        return None;
    }
    let mut addrs = vec![std::ptr::null_mut::<std::ffi::c_void>(); 4096];
    let (mut pages, mut calls) = (0u64, 0u64);
    let mut query = |a: u64, e: u64| -> Option<()> {
        let mut at = a;
        while at < e {
            let mut count: usize = addrs.len();
            let mut gran = 0u32;
            calls += 1;
            if unsafe { GetWriteWatch(0, at as _, (e - at) as usize, addrs.as_mut_ptr(), &mut count, &mut gran) } != 0 {
                return None;
            }
            pages += count as u64;
            if count < addrs.len() {
                break;
            }
            at = addrs[count - 1] as u64 + 4096;
        }
        Some(())
    };
    let (lo, hi) = (lo & !4095, hi.next_multiple_of(4096));
    if mixed() {
        for (a, e) in window_pieces(lo, hi) {
            if !view_window(a) && mixed_made(a) {
                query(a, e)?;
            }
        }
    } else {
        query(lo, hi)?;
    }
    Some((pages, calls))
}

/// One write-watch take of [lo, hi) (inside one allocation).
fn take_written_query(lo: u64, hi: u64, out: &mut Vec<u64>, who: usize) -> Option<()> {
    use std::sync::atomic::Ordering::Relaxed;
    let inflight = InFlight::new();
    let t = crate::gpu::record_opts::Timer::start();
    let mut addrs = [std::ptr::null_mut::<std::ffi::c_void>(); 256];
    let mut at = lo;
    if crate::gpu::tune::ww_probe() {
        // Clean ranges (most takes) end with one query that resets nothing;
        // otherwise the resetting queries start at the first written page.
        let mut count: usize = 1;
        let mut gran = 0u32;
        let r = unsafe { GetWriteWatch(0, lo as _, (hi - lo) as usize, addrs.as_mut_ptr(), &mut count, &mut gran) };
        if r == 0 && count == 0 {
            let ns = t.elapsed().as_nanos() as u64;
            ww_note(who, (hi - lo) / 4096, ns, true, inflight.others);
            WRITE_WATCH_STATS[0].fetch_add(1, Relaxed);
            WRITE_WATCH_STATS[1].fetch_add(ns, Relaxed);
            PROBE_CLEAN.fetch_add(1, Relaxed);
            return Some(());
        }
        if r == 0 && count == 1 {
            at = (addrs[0] as u64).clamp(lo, hi);
        }
    }
    while at < hi {
        let mut count: usize = addrs.len();
        let mut gran = 0u32;
        let r = unsafe { GetWriteWatch(WRITE_WATCH_FLAG_RESET, at as _, (hi - at) as usize, addrs.as_mut_ptr(), &mut count, &mut gran) };
        if r != 0 {
            return None;
        }
        out.extend(addrs[..count].iter().map(|&a| a as u64));
        if count < addrs.len() {
            break;
        }
        // More than fit: continue after the last page reported.
        at = addrs[count - 1] as u64 + 4096;
    }
    let ns = t.elapsed().as_nanos() as u64;
    ww_note(who, (hi - lo) / 4096, ns, false, inflight.others);
    WRITE_WATCH_STATS[0].fetch_add(1, Relaxed);
    WRITE_WATCH_STATS[1].fetch_add(ns, Relaxed);
    WRITE_WATCH_STATS[2].fetch_add(out.len() as u64, Relaxed);
    Some(())
}

/// Reserves the whole guest range (`reserve_guest_range_early`). Must run
/// before anything else allocates widely.
pub fn reserve_guest_range() -> Result<(), String> {
    // (Tried again for up to a second at the first allocation: `reserve_guest_range_early`.)
    reserve_guest_range_early();
    if !RESERVED_EARLY.load(std::sync::atomic::Ordering::Acquire) {
        let err = last_error();
        // Find what occupies the range for a useful diagnostic: each region's kind and, for a
        // module or a mapped file, its file (it names the program that put it there).
        let mut info: MEMORY_BASIC_INFORMATION = unsafe { std::mem::zeroed() };
        let mut addr = GUEST_BASE;
        let mut busy = Vec::new();
        let mut private = false;
        while addr < GUEST_END {
            let n = unsafe { VirtualQuery(addr as _, &mut info, std::mem::size_of::<MEMORY_BASIC_INFORMATION>()) };
            if n == 0 {
                break;
            }
            if info.State != MEM_FREE && info.Type != MEM_IMAGE && info.Type != MEM_MAPPED {
                private = true;
            }
            if info.State != MEM_FREE && busy.len() < 16 {
                let kind = match info.Type {
                    0x0100_0000 => "a module",
                    MEM_MAPPED => "a mapped file",
                    _ => "private memory",
                };
                let file = crate::platform::files::mapped_file_name(info.BaseAddress as u64);
                let file = if file.is_empty() { file } else { format!(" {}", users_folder_hidden(&crate::log::redact(&file))) };
                busy.push(format!("{:#x}+{:#x} {kind}{file}", info.BaseAddress as u64, info.RegionSize));
            }
            addr = info.BaseAddress as u64 + info.RegionSize as u64;
        }
        // (Private memory names no owner: the modules other programs put into this process
        // before it reserved the range are its likely owners, issue #15. Listed only then: a
        // module or mapped file names its own, and a bystander listed beside it would be blamed;
        // the review of 2026-11-25 evening.)
        let others = if private { crate::platform::memory_diagnostics::foreign_modules() } else { Vec::new() };
        let others = if others.is_empty() { String::new() } else { format!("; private memory names no owner, other programs' modules loaded then: {}", others.join(", ")) };
        return Err(format!("reserving guest range failed (error {err}); occupied: {}{others}", busy.join("; ")));
    }
    Ok(())
}

/// `path` with the folder after `\Users\` shown as `<user>`: a mapped file's name comes as a
/// device path (`\Device\HarddiskVolumeN\...`), which the log's own rule for the profile folder
/// does not meet (the review of 2026-11-25).
pub(crate) fn users_folder_hidden(path: &str) -> String {
    let low = path.to_ascii_lowercase();
    let Some(i) = low.find("\\users\\") else { return path.to_string() };
    let start = i + "\\users\\".len();
    let end = path[start..].find('\\').map_or(path.len(), |j| start + j);
    format!("{}<user>{}", &path[..start], &path[end..])
}

/// Host memory calls for guest mappings in the profile window
/// (`host_calls_line`): [commits, ns, decommits, ns, decommits deferred
/// under GPU imports, protection changes, ns, unmap notes, ns].
pub static HOST_CALLS: [std::sync::atomic::AtomicU64; 9] = [const { std::sync::atomic::AtomicU64::new(0) }; 9];
static FAULTS_AT_RESET: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);

/// Adds a call and its time to HOST_CALLS[k], [k + 1].
fn note_host_call(k: usize, t0: std::time::Instant) {
    use std::sync::atomic::Ordering::Relaxed;
    HOST_CALLS[k].fetch_add(1, Relaxed);
    HOST_CALLS[k + 1].fetch_add(t0.elapsed().as_nanos() as u64, Relaxed);
    // (A long one, its waits for the guest memory's readers and armings
    // included, is a hitch cause.)
    crate::gpu::hitch::note_since(crate::gpu::hitch::Kind::Mirror, t0, || ["a commit", "", "a decommit", "", "", "a protection change", "", "an unmap note"].get(k).copied().unwrap_or("a host memory call").to_string());
}

/// This process's page faults so far.
fn page_faults() -> u64 {
    use crate::platform::os::Win32::System::ProcessStatus::{GetProcessMemoryInfo, PROCESS_MEMORY_COUNTERS};
    let mut c: PROCESS_MEMORY_COUNTERS = unsafe { std::mem::zeroed() };
    c.cb = std::mem::size_of::<PROCESS_MEMORY_COUNTERS>() as u32;
    let ok = unsafe { GetProcessMemoryInfo(crate::platform::os::Win32::System::Threading::GetCurrentProcess(), &mut c, c.cb) };
    if ok != 0 { c.PageFaultCount as u64 } else { 0 }
}

/// Clears the window's host-call counts (`profile reset`).
pub fn reset_host_calls() {
    use std::sync::atomic::Ordering::Relaxed;
    for c in &HOST_CALLS {
        c.store(0, Relaxed);
    }
    FAULTS_AT_RESET.store(page_faults(), Relaxed);
}

/// One profile line: the window's host memory calls and page faults.
pub fn host_calls_line() -> String {
    use std::sync::atomic::Ordering::Relaxed;
    let v: Vec<u64> = HOST_CALLS.iter().map(|c| c.load(Relaxed)).collect();
    let faults = page_faults().saturating_sub(FAULTS_AT_RESET.load(Relaxed));
    format!(
        "    host memory calls in the window: commits {} ({} ms), decommits {} ({} ms; {} more deferred under GPU imports), protection changes {} ({} ms), unmap notes {} ({} ms); page faults {}\n",
        v[0],
        v[1] / 1_000_000,
        v[2],
        v[3] / 1_000_000,
        v[4],
        v[5],
        v[6] / 1_000_000,
        v[7],
        v[8] / 1_000_000,
        faults
    )
}

/// A commit Windows refused for want of memory (its commit limit: RAM and page file): tried
/// again for about 3 s while memory frees up, then the game ends with a message (the title
/// would fail on the refused mapping in a way the player cannot read). Other failures: false.
fn commit_retry(a: u64, n: u64, prot: u32) -> bool {
    const ERROR_NOT_ENOUGH_MEMORY: u32 = 8;
    const ERROR_COMMITMENT_LIMIT: u32 = 1455;
    let e = last_error();
    if e != ERROR_NOT_ENOUGH_MEMORY && e != ERROR_COMMITMENT_LIMIT {
        return false;
    }
    crate::warn!("mem: Windows refused to commit {n:#x} bytes at {a:#x} (error {e}: out of memory); trying again");
    for _ in 0..30 {
        std::thread::sleep(std::time::Duration::from_millis(100));
        if !unsafe { VirtualAlloc(a as _, n as usize, MEM_COMMIT, prot) }.is_null() {
            crate::info!("mem: the commit at {a:#x} went through on a later try");
            return true;
        }
    }
    crate::error!("mem: Windows still refuses to commit {n:#x} bytes at {a:#x} (error {}): out of memory", last_error());
    crate::fatal_exit(crate::EXIT_MEMORY, "out of memory")
}

fn host_commit(addr: u64, size: u64, prot: u32) -> Result<(), MemError> {
    let t0 = std::time::Instant::now();
    crate::gpu::track::remapped(addr, size, crate::gpu::track::REMAP_COMMIT);
    repaged(addr, size);
    if mixed() {
        mixed_make(addr, addr + size).map_err(MemError::HostFailure)?;
    }
    // Parts whose decommit was deferred are still committed with their old
    // contents: a new mapping starts zeroed, as after a decommit.
    let stale = cancel_deferred(addr, addr + size);
    // (Its parts, when long, are hitch causes of their own: `gpu::hitch`.)
    use crate::gpu::hitch::{note_since, Kind::Mirror};
    if !stale.is_empty() {
        let t = std::time::Instant::now();
        let _readers_out = GUEST_READERS.write();
        note_since(Mirror, t, || format!("a commit of {addr:#x}+{size:#x} waiting for the guest memory's readers"));
        let t = std::time::Instant::now();
        for &(a, e) in &stale {
            for (a, e) in host_pieces(a, e).filter(|&(a, _)| !view_window(a)) {
                let p = unsafe { VirtualAlloc(a as _, (e - a) as usize, MEM_COMMIT, PAGE_READWRITE) };
                if !p.is_null() {
                    unsafe { std::ptr::write_bytes(a as *mut u8, 0, (e - a) as usize) };
                }
            }
        }
        note_since(Mirror, t, || format!("a commit of {addr:#x}+{size:#x} zeroing {} bytes left by deferred decommits", stale.iter().map(|r| r.1 - r.0).sum::<u64>()));
    }
    // (Front end 4's held pages there released first: a commit resets their
    // protection.)
    crate::gpu::fe2::c4_release_sync(addr, size, false);
    let t = std::time::Instant::now();
    let _arm = crate::gpu::resident::remap_guard();
    note_since(Mirror, t, || format!("a commit of {addr:#x}+{size:#x} waiting for armings"));
    // Pages armed through the host read-write first: whether committing
    // already committed pages resets their protection is not ours to rely
    // on, and `settled` only clears their bits.
    let t = std::time::Instant::now();
    crate::hosted::disarm(addr, size);
    note_since(Mirror, t, || format!("a commit of {addr:#x}+{size:#x}: its disarm"));
    let t = std::time::Instant::now();
    let mut ok = true;
    // (View windows are committed with their views.)
    for (a, e) in host_pieces(addr, addr + size).filter(|&(a, _)| !view_window(a)) {
        if unsafe { VirtualAlloc(a as _, (e - a) as usize, MEM_COMMIT, win_prot(prot)) }.is_null() && !commit_retry(a, e - a, win_prot(prot)) {
            ok = false;
            break;
        }
    }
    note_since(Mirror, t, || format!("a commit of {addr:#x}+{size:#x}: VirtualAlloc"));
    let t = std::time::Instant::now();
    crate::gpu::resident::settled(addr, size);
    note_since(Mirror, t, || format!("a commit of {addr:#x}+{size:#x}: its pages' states reset"));
    note_host_call(0, t0);
    if !ok { Err(MemError::HostFailure(last_error())) } else { Ok(()) }
}

/// Guest ranges [start, end) unmapped while a GPU import covered them: their
/// host pages stay committed until no import does (`retry_deferred`). A
/// decommit under an import that batches in flight still used lost the
/// device at level loads (the title unmaps memory it is done with inside
/// the 64 MiB windows the GPU imports; FINDINGS.md 2026-10-13).
static DEFERRED: Mutex<Vec<(u64, u64)>> = Mutex::new(Vec::new());
static DEFERRED_LEN: std::sync::atomic::AtomicUsize = std::sync::atomic::AtomicUsize::new(0);
/// [deferred, decommitted later, mapped again first, bytes deferred].
pub static DEFERRED_STATS: [std::sync::atomic::AtomicU64; 4] = [const { std::sync::atomic::AtomicU64::new(0) }; 4];

/// Decommits under GPU imports wait for them (`DS_DEFER_DECOMMIT=0`: at
/// once, as before; for A/B).
pub fn defer_decommits() -> bool {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ON.get_or_init(|| std::env::var("DS_DEFER_DECOMMIT").ok().as_deref() != Some("0"))
}

/// Removes [start, end) from the deferred decommits; the parts that were
/// deferred (still committed).
fn cancel_deferred(start: u64, end: u64) -> Vec<(u64, u64)> {
    use std::sync::atomic::Ordering::Relaxed;
    if DEFERRED_LEN.load(std::sync::atomic::Ordering::Acquire) == 0 {
        return Vec::new();
    }
    let mut d = DEFERRED.lock();
    let mut hit = Vec::new();
    let mut keep = Vec::with_capacity(d.len());
    for &(a, e) in d.iter() {
        if e <= start || end <= a {
            keep.push((a, e));
            continue;
        }
        hit.push((a.max(start), e.min(end)));
        if a < start {
            keep.push((a, start));
        }
        if end < e {
            keep.push((end, e));
        }
    }
    if !hit.is_empty() {
        DEFERRED_STATS[2].fetch_add(hit.len() as u64, Relaxed);
    }
    *d = keep;
    DEFERRED_LEN.store(d.len(), std::sync::atomic::Ordering::Release);
    hit
}

/// Decommits the deferred ranges no import covers any more (the GPU
/// completion thread, after freeing dropped imports).
pub fn retry_deferred() {
    if crate::sections::on() {
        crate::sections::retry_deferred(crate::gpu::imports_may_cover);
        return;
    }
    if views_on() {
        crate::sections::retry_deferred(crate::gpu::imports_may_cover);
    }
    if DEFERRED_LEN.load(std::sync::atomic::Ordering::Acquire) == 0 {
        return;
    }
    // The memory manager's lock: no mapping changes meanwhile.
    let _st = MEM.lock();
    let mut d = DEFERRED.lock();
    let mut keep = Vec::new();
    for &(a, e) in d.iter() {
        if crate::gpu::imports_may_cover(a, e - a) {
            keep.push((a, e));
            continue;
        }
        let _readers_out = GUEST_READERS.write();
        let _arm = crate::gpu::resident::remap_guard();
        for (a, e) in host_pieces(a, e) {
            unsafe { VirtualFree(a as _, (e - a) as usize, MEM_DECOMMIT) };
        }
        // Their page states go with them, armed bits too: decommitted pages
        // still marked armed made a disarm over them and their armed
        // neighbours fail (FINDINGS.md 2026-10-27 night).
        crate::gpu::resident::settled(a, e - a);
        DEFERRED_STATS[1].fetch_add(1, std::sync::atomic::Ordering::Relaxed);
    }
    *d = keep;
    DEFERRED_LEN.store(d.len(), std::sync::atomic::Ordering::Release);
}

/// Bulk readers of guest memory on our threads (texture uploads) hold this
/// shared while they check that pages are mapped and read them; decommits and
/// protection changes take it exclusively. A title thread unmapped a layer of
/// a streamed texture array during a level load while a queue thread
/// detiled it (fatal access violation, run 20261012-082109). Holders must not
/// take `MEM` (decommits hold it while they wait here): they check pages with
/// `os_readable`, or with the mapping table before taking this and
/// `table_view_held` after.
pub static GUEST_READERS: parking_lot::RwLock<()> = parking_lot::RwLock::new(());

/// Whether a view of the mapping table taken at generation `g0` (`MAP_GEN`
/// read before the table checks) still holds now that `GUEST_READERS` is
/// held: no mapping or protection change since. Unmaps and protection
/// changes move the generation before they change pages, and wait for the
/// readers, so ranges the table showed readable stay readable while the
/// lock is held. (`VirtualQuery` sizes its region by walking page tables:
/// thousands of checks per streamed texture array cost 0.5-0.6 s.)
pub fn table_view_held(g0: u64) -> bool {
    MAP_GEN.load(std::sync::atomic::Ordering::SeqCst) == g0
}

/// `table_view_held` for one range: no mapping or protection change since generation `g0`
/// touched guest [start, end) (the latest changes are kept, `CHANGES`; when they no longer
/// reach back to `g0`, as `table_view_held`). A change elsewhere leaves the range's table view
/// and import windows as they were: arrivals map and unmap memory many times a second, and
/// textures uploaded meanwhile (the 1-1 arrival's 138 and 184 MB arrays) fell back to the CPU's
/// detiling on the recording threads (2026-11-21).
pub fn table_view_held_for(g0: u64, start: u64, end: u64) -> bool {
    if MAP_GEN.load(std::sync::atomic::Ordering::SeqCst) == g0 {
        return true;
    }
    let c = CHANGES.read();
    // (Every change after g0 still listed: generations rise by one along the list.)
    if c.front().is_none_or(|&(g, _, _)| g > g0 + 1) {
        return false;
    }
    !c.iter().rev().take_while(|&&(g, _, _)| g > g0).any(|&(_, s, e)| s < end && start < e)
}

/// Whether every page of [a, a+len) is committed and readable now (operating
/// system query; takes none of our locks).
pub fn os_readable(a: u64, len: u64) -> bool {
    use crate::platform::os::Win32::System::Memory::{MEMORY_BASIC_INFORMATION, MEM_COMMIT, PAGE_GUARD, PAGE_NOACCESS, VirtualQuery};
    let end = a.saturating_add(len.max(1));
    let mut p = a;
    while p < end {
        let mut mbi: MEMORY_BASIC_INFORMATION = unsafe { std::mem::zeroed() };
        if unsafe { VirtualQuery(p as *const _, &mut mbi, std::mem::size_of::<MEMORY_BASIC_INFORMATION>()) } == 0 {
            return false;
        }
        if mbi.State != MEM_COMMIT || mbi.Protect == 0 || mbi.Protect & PAGE_GUARD != 0 {
            return false;
        }
        let next = mbi.BaseAddress as u64 + mbi.RegionSize as u64;
        if next <= p {
            return false;
        }
        // (No-access pages armed through the host read as readable: an
        // access is answered there, or waits for the page's words when it is
        // held: `hosted::hold`.)
        if mbi.Protect & PAGE_NOACCESS != 0 && !(p & !4095..next.min(end)).step_by(4096).all(crate::hosted::armed) {
            return false;
        }
        p = next;
    }
    true
}

/// The operating system's regions over [a, a+len): one clause each
/// (state, protection, type), at most `max` (diagnostics of failed imports).
pub fn os_regions(a: u64, len: u64, max: usize) -> String {
    use crate::platform::os::Win32::System::Memory::{MEMORY_BASIC_INFORMATION, VirtualQuery};
    let end = a.saturating_add(len.max(1));
    let mut p = a;
    let mut out = Vec::new();
    while p < end && out.len() < max {
        let mut mbi: MEMORY_BASIC_INFORMATION = unsafe { std::mem::zeroed() };
        if unsafe { VirtualQuery(p as *const _, &mut mbi, std::mem::size_of::<MEMORY_BASIC_INFORMATION>()) } == 0 {
            out.push(format!("{p:#x}: query failed"));
            break;
        }
        let next = (mbi.BaseAddress as u64 + mbi.RegionSize as u64).min(end);
        let state = match mbi.State {
            0x1000 => "commit",
            0x2000 => "reserve",
            0x10000 => "free",
            _ => "?",
        };
        let ty = match mbi.Type {
            0x20000 => "private",
            0x40000 => "mapped",
            0x1000000 => "image",
            _ => "-",
        };
        out.push(format!("{p:#x}+{:#x} {state} prot {:#x} {ty}", next - p, mbi.Protect));
        if next <= p {
            break;
        }
        p = next;
    }
    out.join(", ")
}

/// Kernel-mode accesses of guest memory that failed and were retried
/// (`kernel_io`): [calls, retries, calls that failed for good].
pub static KERNEL_IO: [std::sync::atomic::AtomicU64; 3] = [const { std::sync::atomic::AtomicU64::new(0) }; 3];

/// Runs `op`, one kernel-mode access of guest [base, base+len) (a file read
/// into it: `into_guest`, or a file write from it). Such accesses cannot
/// fault into our exception handler: a page inaccessible for a moment (a
/// section view being split, `sections::wait_split`; a page armed for write
/// tracking) fails them with os error 998 or 1784 instead, and the title
/// sees a failed read (the 8-1 load of tour 20261015-151613 never ended).
/// They are run again after the split (pages disarmed again), a few times,
/// with what the range looked like logged.
pub fn kernel_io<T>(what: &dyn Fn() -> String, base: u64, len: u64, into_guest: bool, mut op: impl FnMut() -> std::io::Result<T>) -> std::io::Result<T> {
    use std::sync::atomic::Ordering::Relaxed;
    // Mirror mode: no-access pages with GPU-written dwords first.
    crate::gpu::resident::release_range(base, len);
    const TRIES: u32 = 8;
    let mut tries = 0;
    let traced = crate::gpu::fe2::traced(base & !4095, base.saturating_add(len));
    loop {
        if let Some(p) = traced {
            crate::gpu::fe2::trace_event(p, &format!("kernel I/O over it begins (try {tries}, into guest {into_guest}): {} ({base:#x}+{len:#x})", what()));
        }
        let r = crate::platform::file_hooks::with_kernel_io_pages(base, len, into_guest, &mut op);
        if let Some(p) = traced {
            crate::gpu::fe2::trace_event(p, &format!("kernel I/O over it ended (try {tries}): {}", if r.is_ok() { "ok".to_string() } else { format!("{:?}", r.as_ref().err().and_then(|e| e.raw_os_error())) }));
        }
        match &r {
            Err(e) if crate::platform::file_hooks::is_memory_fault(e) && tries < TRIES => {
                if tries == 0 && KERNEL_IO[0].fetch_add(1, Relaxed) < 32 {
                    crate::warn!("{}: {e}; retrying ({})", what(), describe_host(base, len));
                }
                tries += 1;
                KERNEL_IO[1].fetch_add(1, Relaxed);
                crate::sections::wait_split_range(base, base.saturating_add(len));
                if into_guest {
                    crate::gpu::resident::disarm_for_host(base, len);
                }
                // Pages armed through the host (`hosted`; reads from
                // no-access pages fail too); front end 4's held pages
                // released first.
                crate::hostcensus::release(base, len);
                crate::gpu::fe2::c4_release_sync(base, len, false);
                crate::hosted::disarm(base, len);
                std::thread::sleep(std::time::Duration::from_millis(1 << tries.min(5)));
                continue;
            }
            Err(e) if tries > 0 => {
                KERNEL_IO[2].fetch_add(1, Relaxed);
                crate::warn!("{}: still failing after {tries} retries: {e} ({})", what(), describe_host(base, len));
            }
            Ok(_) if tries > 0 && KERNEL_IO[0].load(Relaxed) <= 32 => crate::info!("{}: succeeded on retry {tries}", what()),
            _ => {}
        }
        return r;
    }
}

/// [positive short syscalls, EOF results, terminal read errors].
pub static KERNEL_READ_STATUS: [std::sync::atomic::AtomicU64; 3] = [const { std::sync::atomic::AtomicU64::new(0) }; 3];

/// A positional file read into guest memory. Short successful reads advance both the
/// destination and file offset; EFAULT retries repeat only the unfinished suffix. EOF is
/// returned as a short total (APR requires the full total; ordinary reads may return EOF).
pub fn kernel_read_at(what: &dyn Fn(u64) -> String, buf: &mut [u8], offset: u64,
    mut read: impl FnMut(&mut [u8], u64) -> std::io::Result<usize>) -> std::io::Result<usize> {
    let mut done = 0;
    while done < buf.len() {
        let at = offset.checked_add(done as u64).ok_or_else(|| std::io::Error::other("file offset overflow"))?;
        let base = buf.as_ptr() as u64 + done as u64;
        let len = buf.len() - done;
        let label = || what(done as u64);
        let result = kernel_io(&label, base, len as u64, true, || read(&mut buf[done..], at));
        crate::platform::file_hooks::note_kernel_read(base, len as u64, at, &result, &label);
        match result {
            Ok(0) => { KERNEL_READ_STATUS[1].fetch_add(1, std::sync::atomic::Ordering::Relaxed); break; }
            Ok(n) if n <= len => {
                if n < len { KERNEL_READ_STATUS[0].fetch_add(1, std::sync::atomic::Ordering::Relaxed); }
                done += n;
            }
            Ok(_) => return Err(std::io::Error::other("file read exceeded destination")),
            Err(e) if e.kind() == std::io::ErrorKind::Interrupted => continue,
            Err(e) => {
                KERNEL_READ_STATUS[2].fetch_add(1, std::sync::atomic::Ordering::Relaxed);
                crate::warn!("{}: file read failed after {done} of {} bytes: {e}", what(done as u64), buf.len());
                return Err(e);
            }
        }
    }
    Ok(done)
}

/// What guest [base, base+len) looks like now (diagnostics): its mappings,
/// the operating system's regions, the write tracking's page states and the
/// last view split.
pub fn describe_host(base: u64, len: u64) -> String {
    let end = base.saturating_add(len.max(1));
    let mut out = format!("{base:#x}+{len:#x}:");
    match MEM.try_lock_for(std::time::Duration::from_millis(20)) {
        Some(st) => {
            for v in st.overlapping(base, end).iter().take(4) {
                out += &format!(" mapping {:#x}+{:#x} {:?} prot {:#x} '{}';", v.start, v.size, v.kind, v.prot, v.name);
            }
        }
        None => out += " (mappings busy);",
    }
    let mut p = base;
    let mut n = 0;
    while p < end && n < 6 {
        let mut m: MEMORY_BASIC_INFORMATION = unsafe { std::mem::zeroed() };
        if unsafe { VirtualQuery(p as _, &mut m, std::mem::size_of::<MEMORY_BASIC_INFORMATION>()) } == 0 {
            out += " (no region)";
            break;
        }
        out += &format!(" region {:#x}+{:#x} state {:#x} protect {:#x} type {:#x};", m.BaseAddress as u64, m.RegionSize, m.State, m.Protect, m.Type);
        p = m.BaseAddress as u64 + m.RegionSize as u64;
        n += 1;
    }
    out += &format!(" {}; {}", crate::gpu::resident::describe(base, len), crate::sections::split_line());
    out
}

fn host_decommit(addr: u64, size: u64) {
    let t0 = std::time::Instant::now();
    // (Front end 4's held pages there: dropped, their pending words dead.)
    crate::gpu::fe2::c4_drop_sync(addr, size);
    crate::gpu::track::remapped(addr, size, crate::gpu::track::REMAP_DECOMMIT);
    repaged(addr, size);
    // View windows go now (their guest views: imports are of their aliases),
    // the private parts as below.
    let parts = if views_on() {
        let win = mixed_window();
        let mut w = addr & !(win - 1);
        while w < addr + size {
            if view_window(w) {
                mixed_view_release(w);
            }
            w += win;
        }
        private_parts(addr, addr + size)
    } else {
        vec![(addr, addr + size)]
    };
    let mut decommitted = false;
    for (addr, end) in parts {
        let size = end - addr;
        if defer_decommits() && crate::gpu::imports_may_cover(addr, size) {
            let mut d = DEFERRED.lock();
            d.push((addr, addr + size));
            DEFERRED_LEN.store(d.len(), std::sync::atomic::Ordering::Release);
            DEFERRED_STATS[0].fetch_add(1, std::sync::atomic::Ordering::Relaxed);
            DEFERRED_STATS[3].fetch_add(size, std::sync::atomic::Ordering::Relaxed);
            HOST_CALLS[4].fetch_add(1, std::sync::atomic::Ordering::Relaxed);
            continue;
        }
        let _readers_out = GUEST_READERS.write();
        let _arm = crate::gpu::resident::remap_guard();
        for (a, e) in host_pieces(addr, addr + size) {
            unsafe { VirtualFree(a as _, (e - a) as usize, MEM_DECOMMIT) };
        }
        crate::gpu::resident::settled(addr, size);
        decommitted = true;
    }
    // (Counted as before: deferred decommits are counted apart.)
    if decommitted {
        note_host_call(2, t0);
    }
}

pub fn host_protect(addr: u64, size: u64, prot: u32) -> Result<(), MemError> {
    let t0 = std::time::Instant::now();
    // Mirror mode: GPU-written dwords reach guest memory before its
    // protection changes (their pages' protections are restored first);
    // front end 4's held pages likewise (released first).
    crate::gpu::resident::release_range(addr, size);
    crate::gpu::fe2::c4_release_sync(addr, size, false);
    crate::gpu::track::remapped(addr, size, crate::gpu::track::REMAP_PROTECT);
    let mut old = 0u32;
    let _readers_out = GUEST_READERS.write();
    let _arm = crate::gpu::resident::remap_guard();
    let (mut ok, mut done) = (1, addr);
    for (a, e) in host_pieces(addr, addr + size) {
        ok = unsafe { VirtualProtect(a as _, (e - a) as usize, win_prot(prot), &mut old) };
        if ok == 0 {
            break;
        }
        done = e;
    }
    // After the change (a failed one left armed pages armed).
    if done > addr {
        crate::gpu::resident::settled(addr, done - addr);
    }
    note_host_call(5, t0);
    if ok == 0 { Err(MemError::HostFailure(last_error())) } else { Ok(()) }
}

/// A recent unmap or protection removal (`note_unmap`): its guest range,
/// when (ms since the first note, flip), the thread and its guest frames.
struct UnmapNote {
    start: u64,
    end: u64,
    op: &'static str,
    ms: u64,
    flip: u64,
    thread: String,
    /// Guest return addresses (`exception::current_guest_returns`).
    frames: Vec<u64>,
}

/// The last UNMAP_NOTES unmaps and protection removals (a separate lock:
/// fault reports read it while the memory map's lock may be held).
static UNMAPS: Mutex<std::collections::VecDeque<UnmapNote>> = Mutex::new(std::collections::VecDeque::new());
const UNMAP_NOTES: usize = 8192;

/// `DS_UNMAP_FRAMES=0` (boot): unmap notes without the unmapping thread's
/// guest frames.
fn unmap_frames() -> bool {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ON.get_or_init(|| !env_is("DS_UNMAP_FRAMES", "0"))
}

/// Notes an unmap (or a protection change to no access) for fault reports
/// (`unmaps_covering`): the title's level changes free memory that jobs
/// still read in a share of loads (FINDINGS.md 2026-10-19).
fn note_unmap(st: &MemState, op: &'static str, start: u64, end: u64) {
    static T0: std::sync::OnceLock<std::time::Instant> = std::sync::OnceLock::new();
    let t = std::time::Instant::now();
    let ms = T0.get_or_init(std::time::Instant::now).elapsed().as_millis() as u64;
    let n = UnmapNote {
        start,
        end,
        op,
        ms,
        flip: crate::gpu::FRAMES.load(std::sync::atomic::Ordering::Relaxed),
        thread: crate::log::thread_name(),
        // Guest frames checked against the memory map (the caller holds
        // it), not by an operating-system query per frame: that took ~10 ms
        // per unmap while a level streamed in (11.5 s of a 20 s window at
        // 1-1, the frame rate a third of normal; FINDINGS.md 2026-10-19).
        frames: if unmap_frames() {
            crate::exception::current_guest_returns(6, &|a| st.find(a).is_some_and(|v| v.kind != VmaKind::Reserved && v.prot != 0))
        } else {
            Vec::new()
        },
    };
    // The first notes with frames, once (the walk works where it matters).
    static SHOWN: std::sync::atomic::AtomicU32 = std::sync::atomic::AtomicU32::new(0);
    if !n.frames.is_empty() && SHOWN.fetch_add(1, std::sync::atomic::Ordering::Relaxed) < 3 {
        let v: Vec<String> = n.frames.iter().map(|&a| crate::exception::describe_guest(a)).collect();
        crate::info!("mem: unmap note {op} {start:#x}..{end:#x} by {}: {}", n.thread, v.join(" <- "));
    }
    let mut g = UNMAPS.lock();
    if g.len() >= UNMAP_NOTES {
        g.pop_front();
    }
    g.push_back(n);
    drop(g);
    note_host_call(7, t);
}

/// The noted unmaps covering `addr`, newest first (at most four): fault
/// reports of reads from memory the title freed. Empty while the notes are
/// being written.
pub fn unmaps_covering(addr: u64) -> Vec<String> {
    let Some(g) = UNMAPS.try_lock() else { return vec!["(notes busy)".into()] };
    let now = g.back().map_or(0, |n| n.ms);
    g.iter()
        .rev()
        .filter(|n| n.start <= addr && addr < n.end)
        .take(4)
        .map(|n| {
            let frames = if n.frames.is_empty() {
                String::new()
            } else {
                let v: Vec<String> = n.frames.iter().map(|&a| crate::exception::describe_guest(a)).collect();
                format!(": {}", v.join(" <- "))
            };
            format!("{} {:#x}..{:#x} at flip {} ({} ms before the last note) by {}{frames}", n.op, n.start, n.end, n.flip, now.saturating_sub(n.ms), n.thread)
        })
        .collect()
}

/// `DS_TRACE_MAP=<hex lo>-<hex hi>`: logs every map, unmap and protection
/// change that touches [lo, hi), with the calling thread's guest frames.
fn trace_map(op: &str, start: u64, end: u64, detail: std::fmt::Arguments) {
    static RANGE: std::sync::OnceLock<Option<(u64, u64)>> = std::sync::OnceLock::new();
    let r = RANGE.get_or_init(|| {
        let v = std::env::var("DS_TRACE_MAP").ok()?;
        let (lo, hi) = v.split_once('-')?;
        let h = |x: &str| u64::from_str_radix(x.trim().trim_start_matches("0x"), 16).ok();
        Some((h(lo)?, h(hi)?))
    });
    if let Some((lo, hi)) = *r {
        if start < hi && lo < end {
            crate::info!("mem trace: {op} {start:#x}..{end:#x} {detail} [{}]: {}", crate::log::thread_name(), crate::exception::current_guest_frames(8).join(" <- "));
        }
    }
}

/// Maps the views of a new mapping (`sections`): the guest view and its
/// alias. Returns (alias slots, as `Vma::slots`; anonymous-section offset).
fn section_commit(addr: u64, size: u64, kind: VmaKind, prot: u32, phys: u64) -> Result<(u8, u64), MemError> {
    use crate::sections::{self as sec, Sec};
    crate::gpu::track::remapped(addr, size, crate::gpu::track::REMAP_COMMIT);
    repaged(addr, size);
    let (which, off) = if kind == VmaKind::Direct { (Sec::Dmem, phys) } else { (Sec::Anon, sec::anon_alloc(size).ok_or(MemError::NoSpace)?) };
    mapped_over_gpu_writes(addr, size, which == Sec::Dmem, off);
    let _arm = crate::gpu::resident::remap_guard();
    if let Err(e) = sec::map_new_view(addr, size, which, off, win_prot(prot)) {
        crate::warn!("mem: {e}");
        if which == Sec::Anon {
            sec::anon_free(off, size);
        }
        return Err(MemError::HostFailure(0));
    }
    let slots = match sec::map_alias(addr, size, which, off) {
        Ok(s) => 1u8 << s,
        Err(e) => {
            crate::warn!("mem: {e} (the GPU cannot import it)");
            0
        }
    };
    crate::gpu::resident::settled(addr, size);
    Ok((slots, if which == Sec::Anon { off } else { 0 }))
}

/// Unmaps the views of mapping `v` (`sections`): its guest view now, its
/// alias once nothing maps through it and no GPU import may use it.
fn section_release(v: &Vma) {
    crate::gpu::track::remapped(v.start, v.size, crate::gpu::track::REMAP_DECOMMIT);
    repaged(v.start, v.size);
    {
        let _readers_out = GUEST_READERS.write();
        let _arm = crate::gpu::resident::remap_guard();
        if let Err(e) = crate::sections::unmap_view(v.start) {
            crate::warn!("mem: {e}");
        }
        crate::gpu::resident::settled(v.start, v.size);
    }
    for slot in 0..crate::sections::ALIAS_SLOTS as u8 {
        if v.slots & (1 << slot) != 0 {
            crate::sections::alias_unref(v.start, v.size, slot, crate::gpu::imports_may_cover);
        }
    }
    if v.kind != VmaKind::Direct {
        crate::sections::anon_free(v.off, v.size);
    }
}

/// Guest mappings unmapped while unfinished GPU work writes them: their
/// physical pages (section offset) and the batch: with section-backed memory
/// those writes land in the pages wherever they are mapped next
/// (`section_commit` waits for the batch with `DS_PHYS_WAIT=1`). Counts:
/// [unmapped with writes pending, of them direct, mapped again while
/// pending, waits that ended in time].
static PENDING_REUSE: Mutex<Vec<(bool, u64, u64, u64, u64)>> = Mutex::new(Vec::new());
pub static PENDING_STATS: [std::sync::atomic::AtomicU64; 4] = [const { std::sync::atomic::AtomicU64::new(0) }; 4];

fn phys_wait() -> bool {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ON.get_or_init(|| env_is("DS_PHYS_WAIT", "1"))
}

fn unmapped_with_gpu_writes(v: &Vma) {
    use std::sync::atomic::Ordering::Relaxed;
    let Some(seq) = crate::gpu::exec::pending_writes(v.start, v.size) else { return };
    let direct = v.kind == VmaKind::Direct;
    let n = PENDING_STATS[0].fetch_add(1, Relaxed);
    if direct {
        PENDING_STATS[1].fetch_add(1, Relaxed);
    }
    if n < 16 {
        crate::info!(
            "mem: {:?} {:#x}+{:#x} '{}' unmapped while batch {seq} (completed up to {}) writes it",
            v.kind,
            v.start,
            v.size,
            v.name,
            crate::gpu::exec::completed_seq()
        );
    }
    if crate::sections::on() {
        let off = if direct { v.phys } else { v.off };
        let mut g = PENDING_REUSE.lock();
        let done = crate::gpu::exec::completed_seq();
        g.retain(|e| e.3 > done);
        if g.len() < 4096 {
            g.push((direct, off, v.size, seq, v.start));
        }
    }
}

/// A new mapping of section pages [off, off+size) (`direct`: direct memory,
/// else anonymous): GPU work still writing them through an earlier mapping
/// is counted, and waited for with `DS_PHYS_WAIT=1` (at most 250 ms).
fn mapped_over_gpu_writes(addr: u64, size: u64, direct: bool, off: u64) {
    use std::sync::atomic::Ordering::Relaxed;
    let hit = {
        let mut g = PENDING_REUSE.lock();
        if g.is_empty() {
            return;
        }
        let done = crate::gpu::exec::completed_seq();
        g.retain(|e| e.3 > done);
        g.iter().filter(|e| e.0 == direct && e.1 < off + size && off < e.1 + e.2).max_by_key(|e| e.3).copied()
    };
    let Some((_, o, n, seq, va)) = hit else { return };
    let k = PENDING_STATS[2].fetch_add(1, Relaxed);
    let waited = phys_wait() && crate::gpu::exec::wait_done_for(seq, std::time::Duration::from_millis(250));
    if waited {
        PENDING_STATS[3].fetch_add(1, Relaxed);
    }
    if k < 16 {
        crate::info!(
            "mem: {} {off:#x}+{size:#x} mapped at {addr:#x} while batch {seq} still writes {o:#x}+{n:#x} through its previous mapping at {va:#x}{}",
            if direct { "direct memory" } else { "anonymous memory" },
            if phys_wait() { if waited { " (waited for it)" } else { " (wait timed out)" } } else { "" }
        );
    }
}

/// Alias views mapped by `bridge_aliases`.
pub static BRIDGES: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);

/// Makes one alias slot cover guest [start, end) without a gap (`sections`;
/// a GPU import is of one slot's alias views): a binding over mappings whose
/// aliases lie in different slots (a mapping takes another slot while an
/// older alias at its addresses is still referenced or imported) could not
/// be imported (24,448 such bindings in tour 20261015-163640, bound as zeros
/// or left out). Each mapping there without an alias in the first slot that
/// is free or covered over all of them gets one, referenced like its first.
/// Returns that slot.
pub fn bridge_aliases(start: u64, end: u64) -> Option<u8> {
    use crate::sections::{self as sec, Sec};
    let mut st = MEM.lock();
    let vmas = st.overlapping(start, end);
    let mut p = start;
    for v in &vmas {
        if v.start > p || v.kind == VmaKind::Reserved {
            return None;
        }
        p = v.end();
    }
    if p < end {
        return None;
    }
    let slot = (0..sec::ALIAS_SLOTS as u8).find(|&slot| vmas.iter().all(|v| sec::alias_covers(v.start, v.size, slot) || sec::alias_free(v.start, v.size, slot)))?;
    for v in &vmas {
        if sec::alias_covers(v.start, v.size, slot) {
            continue;
        }
        let (which, off) = if v.kind == VmaKind::Direct { (Sec::Dmem, v.phys) } else { (Sec::Anon, v.off) };
        if let Err(e) = sec::map_alias_in(v.start, v.size, which, off, slot) {
            crate::warn!("mem: bridging aliases for {start:#x}..{end:#x}: {e}");
            return None;
        }
        if let Some(m) = st.vmas.get_mut(&v.start) {
            m.slots |= 1 << slot;
        }
        BRIDGES.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
    }
    Some(slot)
}

/// Splits the guest view of mapping `v` at `at` (`sections`): views cannot
/// be unmapped in part, so it is unmapped and mapped again as two views of
/// the same section pages (contents kept). Faults of other threads inside it
/// meanwhile wait (`sections::wait_split`); the residency's protections of its
/// pages are reset.
fn section_split(v: &Vma, at: u64, why: usize) {
    use crate::sections::{self as sec, Sec};
    let (which, off) = if v.kind == VmaKind::Direct { (Sec::Dmem, v.phys) } else { (Sec::Anon, v.off) };
    crate::gpu::track::remapped(v.start, v.size, crate::gpu::track::REMAP_PROTECT);
    let _readers_out = GUEST_READERS.write();
    let _arm = crate::gpu::resident::remap_guard();
    sec::split_begin(v.start, v.end(), why);
    let lo = at - v.start;
    let r = sec::unmap_view(v.start)
        .and_then(|_| sec::map_view(v.start, lo, which, off, win_prot(v.prot)))
        .and_then(|_| sec::map_view(at, v.end() - at, which, off + lo, win_prot(v.prot)));
    sec::split_end();
    crate::gpu::resident::settled(v.start, v.size);
    if let Err(e) = r {
        crate::error!("mem: splitting the view {:#x}+{:#x} at {at:#x}: {e}", v.start, v.size);
    }
}

impl MemState {
    /// Finds a free, aligned range of `size` bytes at or above `hint`.
    pub fn find_free(&self, hint: u64, size: u64, align: u64, limit: u64) -> Option<u64> {
        let align = align.max(PAGE);
        let mut cand = align_up(hint, align);
        for (_, v) in self.vmas.range(..) {
            if v.end() <= cand {
                continue;
            }
            if v.start >= cand + size {
                break;
            }
            cand = align_up(v.end(), align);
        }
        if cand + size <= limit { Some(cand) } else { None }
    }

    pub fn find(&self, addr: u64) -> Option<&Vma> {
        self.vmas.range(..=addr).next_back().map(|(_, v)| v).filter(|v| addr < v.end())
    }

    /// Ensures a VMA boundary exists at `addr`.
    fn split_at(&mut self, addr: u64, why: usize) {
        let Some(v) = self.vmas.range(..addr).next_back().map(|(_, v)| v.clone()) else { return };
        if addr > v.start && addr < v.end() {
            if view_window(addr) && addr & (mixed_window() - 1) != 0 {
                mixed_view_privatize(addr & !(mixed_window() - 1), v.prot);
            }
            if crate::sections::on() && v.kind != VmaKind::Reserved {
                section_split(&v, addr, why);
            }
            let mut lo = v.clone();
            lo.size = addr - v.start;
            let mut hi = v;
            hi.size = hi.end() - addr;
            if hi.kind == VmaKind::Direct {
                hi.phys += lo.size;
            } else {
                hi.off += lo.size;
            }
            hi.start = addr;
            self.vmas.insert(lo.start, lo);
            self.vmas.insert(hi.start, hi);
        }
    }

    fn overlapping(&self, start: u64, end: u64) -> Vec<Vma> {
        let mut out = Vec::new();
        if let Some((_, v)) = self.vmas.range(..start).next_back() {
            if v.end() > start {
                out.push(v.clone());
            }
        }
        for (_, v) in self.vmas.range(start..end) {
            out.push(v.clone());
        }
        out
    }

    /// Removes mappings in [start, end), decommitting host pages. Reserved ranges
    /// are left reserved only if `keep_reserved`.
    pub fn unmap_range(&mut self, start: u64, end: u64) {
        self.unmap_range_for(start, end, crate::sections::SPLIT_UNMAP);
    }

    /// `unmap_range`; `why` names the cause of view splits (diagnostics).
    fn unmap_range_for(&mut self, start: u64, end: u64, why: usize) {
        trace_map("unmap", start, end, format_args!(""));
        note_unmap(self, "unmap", start, end);
        rg_clear(start, end);
        bump_gen(start, end);
        self.split_at(start, why);
        self.split_at(end, why);
        let keys: Vec<u64> = self.vmas.range(start..end).map(|(k, _)| *k).collect();
        for k in keys {
            let v = self.vmas.remove(&k).unwrap();
            if matches!(v.kind, VmaKind::Direct | VmaKind::Flexible) {
                unmapped_with_gpu_writes(&v);
            }
            if v.kind == VmaKind::Direct {
                let mut r = REMAP.lock();
                if r.released.len() > 1 << 16 {
                    r.released.clear();
                }
                r.released.insert(v.phys, (v.start, v.size));
            }
            if v.kind == VmaKind::Direct && phys_carry() && !crate::sections::on() {
                park(v.start, v.phys, v.size);
            }
            if v.kind != VmaKind::Reserved {
                if crate::sections::on() {
                    section_release(&v);
                } else {
                    host_decommit(v.start, v.size);
                }
            }
            if v.kind == VmaKind::Flexible {
                self.flexible_used = self.flexible_used.saturating_sub(v.size);
            }
        }
        self.rg_update(start, end);
    }

    /// Maps [addr, addr+size). For fixed mappings any existing mapping there is replaced.
    #[allow(clippy::too_many_arguments)]
    pub fn map(
        &mut self,
        addr: u64,
        size: u64,
        kind: VmaKind,
        prot: u32,
        phys: u64,
        mtype: i32,
        name: &str,
    ) -> Result<u64, MemError> {
        if addr < GUEST_BASE || addr + size > GUEST_END || size == 0 {
            return Err(MemError::Invalid);
        }
        let existing = self.overlapping(addr, addr + size);
        if kind == VmaKind::Direct {
            // Diagnostics: contents the console would keep.
            let same = existing
                .iter()
                .filter(|v| v.kind == VmaKind::Direct && v.phys.wrapping_sub(phys) == v.start.wrapping_sub(addr))
                .map(|v| v.end().min(addr + size) - v.start.max(addr))
                .sum::<u64>();
            let mut r = REMAP.lock();
            if same > 0 {
                r.in_place += 1;
                r.in_place_bytes += same;
                if r.logged < 24 {
                    r.logged += 1;
                    crate::info!("mem: direct {addr:#x}+{size:#x} (phys {phys:#x}) mapped again over its own pages ({same:#x} bytes zeroed here) '{name}'");
                }
            }
            let prior: Vec<(u64, u64, u64)> = r
                .released
                .range(phys.saturating_sub(1 << 30)..phys + size)
                .filter(|(p, (_, n))| **p < phys + size && phys < **p + *n)
                .map(|(p, (va, n))| (*p, *va, *n))
                .collect();
            for (p, va, n) in prior {
                if va.wrapping_sub(p) != addr.wrapping_sub(phys) {
                    r.moved += 1;
                    r.moved_bytes += n.min(size);
                    if r.logged < 24 {
                        r.logged += 1;
                        crate::info!("mem: direct phys {p:#x}+{n:#x}, last mapped at {va:#x}, now mapped at {:#x} '{name}' (contents not carried over)", addr + p.saturating_sub(phys));
                    }
                }
            }
        }
        if !existing.is_empty() {
            self.unmap_range_for(addr, addr + size, crate::sections::SPLIT_MAP_OVER);
        }
        let (mut slots, mut off) = (0u8, 0);
        if kind == VmaKind::Direct && size >= 64 << 20 {
            // (What `DS_MIXED_VIEWS` rules name.)
            static N: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);
            if N.fetch_add(1, std::sync::atomic::Ordering::Relaxed) < 64 {
                crate::info!("mem: direct mapping '{name}' {addr:#x}+{size:#x} (phys {phys:#x}, prot {prot:#x})");
            }
        }
        if kind == VmaKind::Direct && mixed() && !crate::sections::on() && view_rules().iter().any(|(n, s)| n == name && *s == size) {
            mixed_views_make(addr, size, phys, prot);
        }
        if kind != VmaKind::Reserved {
            if crate::sections::on() {
                (slots, off) = section_commit(addr, size, kind, prot, phys)?;
            } else if let Err(e) = host_commit(addr, size, prot) {
                if views_on() {
                    let win = mixed_window();
                    let mut w = addr & !(win - 1);
                    while w < addr + size {
                        if view_window(w) {
                            mixed_view_release(w);
                        }
                        w += win;
                    }
                }
                return Err(e);
            }
        }
        if kind == VmaKind::Direct && phys_carry() && !crate::sections::on() {
            unpark(addr, phys, size, prot);
        }
        if kind != VmaKind::Reserved {
            // Alignment of mappings (64 KiB: what section views may need);
            // direct ones counted with their physical offset.
            let bits = if kind == VmaKind::Direct { addr | size | phys } else { addr | size };
            let k = if bits & 0xFFFF == 0 { 0 } else if bits & 0x3FFF == 0 { 1 } else { 2 };
            MAP_ALIGN[k + if kind == VmaKind::Direct { 0 } else { 3 }].fetch_add(1, std::sync::atomic::Ordering::Relaxed);
        }
        if kind == VmaKind::Direct {
            // Diagnostics: physical memory mapped without an allocation
            // (with sections it shows whatever those pages hold).
            let covered: u64 = self.dmem.range(..phys + size).filter(|(s, b)| **s + b.size > phys).map(|(s, b)| (s + b.size).min(phys + size) - (*s).max(phys)).sum();
            if covered < size {
                static N: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);
                if N.fetch_add(1, std::sync::atomic::Ordering::Relaxed) < 16 {
                    crate::warn!("mem: direct {addr:#x}+{size:#x} maps phys {phys:#x} of which {:#x} bytes are not allocated '{name}'", size - covered);
                }
            }
        }
        if kind == VmaKind::Direct && !crate::sections::on() {
            for v in self.vmas.values() {
                if v.kind == VmaKind::Direct && phys < v.phys + v.size && v.phys < phys + size {
                    crate::warn!(
                        "direct memory aliasing: phys {:#x}+{:#x} already mapped at {:#x} (new va {:#x}); contents are NOT shared",
                        phys,
                        size,
                        v.start,
                        addr
                    );
                    break;
                }
            }
        }
        if kind == VmaKind::Flexible {
            self.flexible_used += size;
        }
        self.vmas.insert(addr, Vma { start: addr, size, kind, prot, phys, mtype, name: name.to_string(), slots, off, born: crate::gpu::FRAMES.load(std::sync::atomic::Ordering::Relaxed) });
        self.rg_update(addr, addr + size);
        bump_gen(addr, addr + size);
        trace_map("map", addr, addr + size, format_args!("{kind:?} prot {prot:#x} phys {phys:#x} '{name}'"));
        Ok(addr)
    }

    /// Maximal run of adjacent committed direct/flexible mappings containing
    /// `addr`: [start, end).
    pub fn committed_span(&self, addr: u64) -> Option<(u64, u64)> {
        let usable = |v: &Vma| matches!(v.kind, VmaKind::Direct | VmaKind::Flexible);
        let v = self.find(addr).filter(|v| usable(v))?;
        let (mut start, mut end) = (v.start, v.end());
        while let Some(p) = self.vmas.range(..start).next_back().map(|(_, p)| p).filter(|p| p.end() == start && usable(p)) {
            start = p.start;
        }
        while let Some(n) = self.vmas.get(&end).filter(|n| usable(n)) {
            end = n.end();
        }
        Some((start, end))
    }

    pub fn protect(&mut self, start: u64, end: u64, prot: u32) -> Result<(), MemError> {
        trace_map("protect", start, end, format_args!("prot {prot:#x}"));
        if prot & 7 == 0 {
            note_unmap(self, "protect none", start, end);
        }
        self.split_at(start, crate::sections::SPLIT_PROTECT);
        self.split_at(end, crate::sections::SPLIT_PROTECT);
        rg_clear(start, end);
        // Before the pages change too: a bulk reader that checked the table
        // before taking `GUEST_READERS` sees the generation move
        // (`table_view_held`).
        bump_gen(start, end);
        let keys: Vec<u64> = self.vmas.range(start..end).map(|(k, _)| *k).collect();
        for k in keys {
            let v = self.vmas.get_mut(&k).unwrap();
            if v.kind == VmaKind::Reserved {
                continue;
            }
            v.prot = prot;
            host_protect(v.start, v.size, prot)?;
        }
        self.rg_update(start, end);
        bump_gen(start, end);
        Ok(())
    }

    pub fn dmem_allocate(&mut self, search_start: u64, search_end: u64, len: u64, align: u64, mtype: i32) -> Option<u64> {
        let align = align.max(PAGE);
        let end_limit = search_end.min(DMEM_SIZE);
        let mut cand = align_up(search_start, align);
        for (&s, b) in self.dmem.iter() {
            if s + b.size <= cand {
                continue;
            }
            if s >= cand + len {
                break;
            }
            cand = align_up(s + b.size, align);
        }
        if cand + len > end_limit {
            return None;
        }
        self.dmem.insert(cand, DmemBlock { size: len, mtype });
        if crate::sections::on() || (mixed() && crate::sections::ready()) {
            crate::sections::dmem_allocated(cand, len);
        }
        Some(cand)
    }

    pub fn dmem_release(&mut self, start: u64, len: u64) {
        let end = start + len;
        let keys: Vec<u64> = self.dmem.range(..end).filter(|(s, b)| **s + b.size > start).map(|(s, _)| *s).collect();
        for s in keys {
            let b = self.dmem.remove(&s).unwrap();
            if s < start {
                self.dmem.insert(s, DmemBlock { size: start - s, mtype: b.mtype });
            }
            if s + b.size > end {
                self.dmem.insert(end, DmemBlock { size: s + b.size - end, mtype: b.mtype });
            }
        }
        // Unmap any virtual mappings of the released physical range.
        let vas: Vec<(u64, u64)> = self
            .vmas
            .values()
            .filter(|v| v.kind == VmaKind::Direct && v.phys < end && start < v.phys + v.size)
            .map(|v| {
                let s = v.start + start.saturating_sub(v.phys);
                let e = v.start + (end.min(v.phys + v.size) - v.phys);
                (s, e)
            })
            .collect();
        for (s, e) in vas {
            self.unmap_range(s, e);
        }
        if phys_carry() {
            drop_parked(start, end);
        }
    }

    pub fn dmem_used(&self) -> u64 {
        self.dmem.values().map(|b| b.size).sum()
    }

    /// Largest free direct-memory block within [start, end).
    pub fn dmem_largest_free(&self, start: u64, end: u64) -> (u64, u64) {
        let end = end.min(DMEM_SIZE);
        let mut best = (0u64, 0u64);
        let mut cursor = start;
        for (&s, b) in self.dmem.iter() {
            if s + b.size <= cursor {
                continue;
            }
            if s > cursor && s.min(end) > cursor {
                let len = s.min(end) - cursor;
                if len > best.1 {
                    best = (cursor, len);
                }
            }
            cursor = cursor.max(s + b.size);
            if cursor >= end {
                break;
            }
        }
        if end > cursor && end - cursor > best.1 {
            best = (cursor, end - cursor);
        }
        best
    }

    /// Allocates runtime-owned guest-visible memory (stub code, HLE data objects).
    pub fn alloc_internal(&mut self, size: u64, prot: u32, name: &str) -> Result<u64, MemError> {
        let size = align_up(size, PAGE);
        let addr = self.find_free(self.internal_cursor, size, PAGE, MODULE_BASE).ok_or(MemError::NoSpace)?;
        self.map(addr, size, VmaKind::Internal, prot, 0, 0, name)?;
        self.internal_cursor = addr + size;
        Ok(addr)
    }
}

/// Simple bump allocator over internal guest memory for small HLE-owned objects.
pub struct GuestArena {
    cur: u64,
    end: u64,
}

pub static ARENA: Mutex<GuestArena> = Mutex::new(GuestArena { cur: 0, end: 0 });

pub fn arena_alloc(size: u64, align: u64) -> u64 {
    let mut a = ARENA.lock();
    let mut p = align_up(a.cur, align.max(8));
    if a.cur == 0 || p + size > a.end {
        let chunk = align_up(size.max(1 << 20), PAGE);
        let base = MEM
            .lock()
            .alloc_internal(chunk, PROT_CPU_READ | PROT_CPU_WRITE, "hle-arena")
            .expect("internal arena exhausted");
        a.cur = base;
        a.end = base + chunk;
        p = align_up(a.cur, align.max(8));
    }
    a.cur = p + size;
    p
}

/// Copies a NUL-terminated string into the guest arena and returns its address.
pub fn arena_cstr(s: &str) -> u64 {
    let p = arena_alloc(s.len() as u64 + 1, 1);
    unsafe {
        std::ptr::copy_nonoverlapping(s.as_ptr(), p as *mut u8, s.len());
        *((p + s.len() as u64) as *mut u8) = 0;
    }
    p
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Cached readable runs survive mapping changes that do not touch them.
    #[test]
    fn positional_read_eof_and_error_are_not_full_completion() {
        let mut out = [0u8; 16];
        let mut calls = 0;
        let n = kernel_read_at(&|_| "EOF test".into(), &mut out, 41, |buf, off| {
            calls += 1;
            if calls == 1 { assert_eq!(off, 41); buf[..7].fill(0x5a); Ok(7) }
            else { assert_eq!(off, 48); Ok(0) }
        }).unwrap();
        assert_eq!(n, 7);
        assert_eq!(&out[..7], &[0x5a; 7]);
        assert_eq!(&out[7..], &[0; 9]);
        calls = 0;
        let r = kernel_read_at(&|_| "error test".into(), &mut out, 0, |_, off| {
            calls += 1;
            if calls == 1 { Ok(7) } else { assert_eq!(off, 7); Err(std::io::Error::other("read failed")) }
        });
        assert!(r.is_err());
    }

    #[test]
    fn readable_runs_survive_unrelated_changes() {
        let g0 = MAP_GEN.load(std::sync::atomic::Ordering::SeqCst);
        let mut runs = vec![(0x1_0000, 0x2_0000, true), (0x5_0000, 0x6_0000, false)];
        bump_gen(0x2_0000, 0x3_0000); // adjacent to the first run
        bump_gen(0x5_8000, 0x5_9000); // inside the second
        let g1 = MAP_GEN.load(std::sync::atomic::Ordering::SeqCst);
        assert!(revalidate(&mut runs, g0, g1));
        assert_eq!(runs, vec![(0x1_0000, 0x2_0000, true)]);
        assert!(!revalidate(&mut runs, u64::MAX, g1), "a cache never filled");
        for _ in 0..CHANGES_KEEP + 1 {
            bump_gen(0x9_0000, 0x9_1000);
        }
        let g2 = MAP_GEN.load(std::sync::atomic::Ordering::SeqCst);
        assert!(!revalidate(&mut runs, g1, g2), "changes no longer all known");
    }

    /// `vma_bounds` answers as the mapping table does, and a mapping change
    /// empties its per-thread cache.
    #[test]
    fn vma_bounds_follows_the_table() {
        let (a, n) = (0x7F12_0000_0000u64, 0x30_0000u64);
        let vma = |start: u64, size: u64, kind: VmaKind| Vma { start, size, kind, prot: 3, phys: 0, mtype: 0, name: "test".into(), slots: 0, off: 0, born: 0 };
        {
            let mut st = MEM.lock();
            st.vmas.insert(a, vma(a, n, VmaKind::Flexible));
            st.vmas.insert(a + n, vma(a + n, 0x1_0000, VmaKind::Reserved));
            bump_gen(a, a + n + 0x1_0000);
        }
        assert_eq!(vma_bounds(a + 0x123), Some((a, a + n)));
        assert_eq!(vma_bounds(a + n - 1), Some((a, a + n)), "cached");
        assert_eq!(vma_bounds(a + n), None, "a reservation");
        assert_eq!(vma_bounds(a + n + 0x1_0000), None, "a gap");
        // The mapping split in two: the cached bounds go with the change.
        {
            let mut st = MEM.lock();
            st.vmas.insert(a, vma(a, 0x10_0000, VmaKind::Flexible));
            st.vmas.insert(a + 0x10_0000, vma(a + 0x10_0000, n - 0x10_0000, VmaKind::Flexible));
            bump_gen(a, a + n);
        }
        assert_eq!(vma_bounds(a + 0x123), Some((a, a + 0x10_0000)));
        assert_eq!(vma_bounds(a + 0x20_0000), Some((a + 0x10_0000, a + n)));
        {
            let mut st = MEM.lock();
            st.vmas.remove(&a);
            st.vmas.remove(&(a + 0x10_0000));
            st.vmas.remove(&(a + n));
            bump_gen(a, a + n + 0x1_0000);
        }
        assert_eq!(vma_bounds(a + 0x123), None, "unmapped");
    }
}
