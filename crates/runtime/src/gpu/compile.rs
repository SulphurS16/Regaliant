//! Pipelines compiled ahead of their first use (`tune compile_ahead`).
//!
//! The first use of a pipeline stalls its command processor while the
//! driver compiles it: 0.2-2.9 s for the title's large compute programs
//! (40,000-450,000 SPIR-V words; the user's play test of 2026-10-16 stalled
//! 27 s in four minutes), and such uses come in bursts: the programs of an
//! effect or an area appear together in one command stream, each compiled
//! only when execution reaches it. When a dispatch's pipeline is missing,
//! its command processor (or a parallel run's chunk worker, over its chunk)
//! scans the commands after it
//! (`Cp::prefetch_compiles`) and hands their compute stages to these
//! workers, which prepare each with the registers the scan saw (as
//! `gpu::ahead` does) and create its pipeline if it is missing
//! (`exec::precompile`). Meanwhile the command processor creates its own,
//! and later waits for the ones in flight instead of compiling them again
//! (`exec::compute_pipe`). Compute stages that a command processor prepares
//! while it waits on a label (`tune wait_ahead`) and whose pipeline is
//! missing are handed over too, already prepared (`submit_prepared`). A
//! preparation made from inputs that differ at execution only costs a
//! compile: execution prepares its stages itself.
//!
//! (Graphics pipelines the same way, `tune compile_gfx`, saw too little in
//! their scans: removed on 2026-11-01.) Since 2026-11-19 a draw that waits for its
//! missing pipeline (one that writes memory, clears depth or stencil, or draws into a target
//! not drawn lately: the rest are left out while theirs is made in the background) scans the
//! commands after it (`Cp::prefetch_gfx`): their draws' stages are prepared here and their
//! pipelines queued for the background graphics workers (`submit_draws`, `DS_GFX_AHEAD=0`
//! off).

use super::ahead::Slot;
use super::shaders::{self, Prepared, RegRead, StageKind};
use parking_lot::{Condvar, Mutex};
use std::collections::VecDeque;
use std::sync::atomic::{AtomicU64, AtomicUsize, Ordering};
use std::sync::Arc;

/// A stage to prepare from scanned registers, one prepared already, or a
/// program's batched form to prepare and make (`submit_batched`).
enum Job {
    Regs(Arc<Slot>),
    Prepared(Box<Prepared>),
    Batched(Arc<Slot>),
    /// A scanned draw: its stages and baked-state registers (`submit_draws`).
    Draw(Box<(Arc<Slot>, Option<Arc<Slot>>, super::exec::GfxRegs)>),
}

struct Pool {
    jobs: Mutex<VecDeque<Job>>,
    cv: Condvar,
}

static POOL: Pool = Pool { jobs: Mutex::new(VecDeque::new()), cv: Condvar::new() };
/// Translations of the prepared stages queued (`submit_prepared`).
static QUEUED: Mutex<Vec<u64>> = Mutex::new(Vec::new());
static WORKERS: AtomicUsize = AtomicUsize::new(0);

/// Statistics: indices into STATS.
const SCANS: usize = 0;
const SCAN_NS: usize = 1;
const JOBS: usize = 2;
const PREPARED: usize = 3;
const FAILED: usize = 4;
const COMPILED: usize = 5;
const COMPILE_NS: usize = 6;
const PRESENT: usize = 7;
const DROPPED: usize = 8;
const AT_WAITS: usize = 9;
/// Draws: handed over, over the limit, prepared, pipelines created, their
/// ns, present already, failed or recording nothing.
/// Programs with a pipeline already handed over again (another spec of
/// them is a first use too).
const AGAIN: usize = 10;
/// Batched forms handed over (`submit_batched`), and those made.
const BATCHED: usize = 11;
const BATCHED_MADE: usize = 12;
/// Batched programs' run-time variants made beside them (`run_batched`).
const BATCHED_RT: usize = 13;
static STATS: [AtomicU64; 14] = [const { AtomicU64::new(0) }; 14];

/// Compute programs whose pipeline was created or found present: scans skip
/// them before their per-scan limit (it filled with programs compiled
/// already and dropped the new ones further on).
static KNOWN: Mutex<Option<std::collections::HashSet<u64>>> = Mutex::new(None);

/// A compute program has a pipeline now (`exec`).
pub fn note_program(addr: u64) {
    KNOWN.lock().get_or_insert_with(Default::default).insert(addr);
}

/// Whether compute program `addr` has a pipeline (made or found present).
pub fn has_pipeline(addr: u64) -> bool {
    KNOWN.lock().as_ref().is_some_and(|k| k.contains(&addr))
}

/// Most compute programs one scan hands over (16 lost the gain: a level's
/// first frames meet hundreds of programs; of those compiled ahead with 64,
/// 93% were used, run 20261016-214851).
const MAX_JOBS: usize = 64;

/// Whether missing pipelines are compiled ahead (`tune compile_ahead`:
/// worker threads; 0 off).
pub fn enabled() -> bool {
    super::tune::compile_ahead() > 0
}

/// `DS_COMPILE_AHEAD_MODE` (diagnosis): bit 0 scans at first uses, bit 1
/// stages prepared at label waits (default 3: both); bit 2 drops what a
/// scan found, bit 3 prepares it without creating pipelines.
fn mode() -> u32 {
    static M: std::sync::OnceLock<u32> = std::sync::OnceLock::new();
    *M.get_or_init(|| std::env::var("DS_COMPILE_AHEAD_MODE").ok().and_then(|v| v.parse().ok()).unwrap_or(3))
}

/// Scans at first uses wanted.
pub fn at_first_uses() -> bool {
    enabled() && mode() & 1 != 0
}

/// Stages prepared at label waits handed over.
pub fn at_waits() -> bool {
    enabled() && mode() & 2 != 0
}

fn ensure_workers() {
    let want = super::tune::compile_ahead().min(16) as usize;
    while WORKERS.load(Ordering::Acquire) < want {
        let n = WORKERS.fetch_add(1, Ordering::AcqRel);
        if n >= want {
            WORKERS.fetch_sub(1, Ordering::AcqRel);
            break;
        }
        let name = format!("gpu-compile{n}");
        let r = crate::thread::spawn_host(
            &name.clone(),
            8 << 20,
            Box::new(move || {
                crate::log::set_thread_name(&name);
                crate::exception::arm_watch_current_thread(false);
                super::profile_this_thread("compile");
                worker();
            }),
        );
        if let Err(e) = r {
            crate::warn!("gpu: no compile worker ({e})");
            WORKERS.fetch_sub(1, Ordering::AcqRel);
            break;
        }
    }
}

fn worker() {
    loop {
        let job = {
            let mut q = POOL.jobs.lock();
            loop {
                if let Some(s) = q.pop_front() {
                    break s;
                }
                POOL.cv.wait(&mut q);
            }
        };
        // A panic must not end the worker (its jobs would wait forever).
        let _ = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| match job {
            Job::Regs(slot) => run(&slot),
            Job::Prepared(p) => {
                compile(&p);
                QUEUED.lock().retain(|&k| k != p.tr.key);
            }
            Job::Batched(slot) => run_batched(&slot),
            Job::Draw(d) => run_draw(&d.0, d.1.as_deref(), &d.2),
        }));
    }
}

/// Whether a draw that waits for its missing pipeline scans the draws after it
/// (`Cp::prefetch_gfx`; `DS_GFX_AHEAD=0` off; needs the workers, `enabled`).
pub fn gfx_ahead() -> bool {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    enabled() && *ON.get_or_init(|| std::env::var("DS_GFX_AHEAD").ok().as_deref() != Some("0"))
}

/// Scanned draws handed over once each per (vertex stage's registers, pixel stage's registers,
/// baked-state registers), by hash: the same programs with other descriptors are other
/// translations and pipelines (keyed by their programs' addresses, a first walk's worker waited
/// ~140 ms each for five pipelines of two program pairs one after another, 2026-11-19). At most
/// `DRAWS_SEEN_MAX` (then forgotten).
static DRAWS_SEEN: Mutex<Option<std::collections::HashSet<(u64, u64, u64)>>> = Mutex::new(None);
const DRAWS_SEEN_MAX: usize = 200_000;

/// A stage snapshot's registers by hash (`DRAWS_SEEN`).
fn regs_hash(r: &shaders::StageRegs) -> u64 {
    use std::hash::{Hash, Hasher};
    let mut h = std::collections::hash_map::DefaultHasher::new();
    r.sh.hash(&mut h);
    r.ctx.hash(&mut h);
    h.finish()
}
/// At most this many new draws a scan (64 left 89 of 305 out in a first boot's 240 s).
const MAX_DRAW_JOBS: usize = 128;
/// [scans, draws found, handed over, prepared, pipelines queued, made or queued already,
/// recording nothing, not prepared, over the per-scan limit, scan ns].
static GFX: [AtomicU64; 10] = [const { AtomicU64::new(0) }; 10];

/// Hands the draws of a scan at a draw that waits (`Cp::prefetch_gfx`, after `ns` of
/// scanning) to the workers: each (programs, state) once.
pub fn submit_draws(draws: Vec<(Arc<Slot>, Option<Arc<Slot>>, super::exec::GfxRegs)>, ns: u64) {
    GFX[0].fetch_add(1, Ordering::Relaxed);
    GFX[1].fetch_add(draws.len() as u64, Ordering::Relaxed);
    GFX[9].fetch_add(ns, Ordering::Relaxed);
    let mut jobs = Vec::new();
    {
        let mut g = DRAWS_SEEN.lock();
        let seen = g.get_or_insert_with(Default::default);
        if seen.len() > DRAWS_SEEN_MAX {
            seen.clear();
        }
        for (vs, ps, regs) in draws {
            let key = (regs_hash(vs.regs()), ps.as_ref().map_or(0, |p| regs_hash(p.regs())), regs.hash64());
            if shaders::ngg_addr_of(vs.regs()) == 0 || seen.contains(&key) {
                continue;
            }
            // (Over the limit: not marked seen, so a later scan hands it over.)
            if jobs.len() >= MAX_DRAW_JOBS {
                GFX[8].fetch_add(1, Ordering::Relaxed);
                continue;
            }
            seen.insert(key);
            jobs.push(Job::Draw(Box::new((vs, ps, regs))));
        }
    }
    if jobs.is_empty() {
        return;
    }
    GFX[2].fetch_add(jobs.len() as u64, Ordering::Relaxed);
    ensure_workers();
    POOL.jobs.lock().extend(jobs);
    POOL.cv.notify_all();
}

/// A scanned draw: its stages prepared from the registers the scan saw, then its pipeline
/// queued when missing (`exec::queue_gfx_ahead`).
fn run_draw(vs: &Slot, ps: Option<&Slot>, g: &super::exec::GfxRegs) {
    let prep = |s: &Slot| {
        let r = s.regs();
        let p = match r.kind {
            StageKind::Vertex => shaders::prepare_ngg(r),
            StageKind::Pixel => shaders::prepare_ps(r),
            StageKind::Compute => return None,
        };
        // (`DS_COMPILE_TRACE=1`: the first stages not prepared and why.)
        if super::exec::compile_trace_on() && (p.is_err() || r.missed.load(Ordering::Relaxed)) {
            static N: std::sync::atomic::AtomicU32 = std::sync::atomic::AtomicU32::new(0);
            if N.fetch_add(1, Ordering::Relaxed) < 24 {
                let why = match &p {
                    Err(e) => e.clone(),
                    Ok(_) => format!("register {:#x} outside the snapshot", r.missed_reg.load(Ordering::Relaxed)),
                };
                crate::info!("compile ahead: a scanned draw's {:?} stage not prepared: {why}", r.kind);
            }
        }
        p.ok().filter(|_| !r.missed.load(Ordering::Relaxed))
    };
    let v = match g.base_inst() {
        Some(slot) => shaders::with_base_instance(slot, || prep(vs)),
        None => prep(vs),
    };
    let Some(v) = v else {
        GFX[7].fetch_add(1, Ordering::Relaxed);
        return;
    };
    let p = match ps {
        Some(s) => match prep(s) {
            Some(p) => Some(p),
            None => {
                GFX[7].fetch_add(1, Ordering::Relaxed);
                return;
            }
        },
        None => None,
    };
    GFX[3].fetch_add(1, Ordering::Relaxed);
    let q = super::exec::queue_gfx_ahead(g, &v, p.as_ref());
    if q.is_none() && super::exec::compile_trace_on() {
        static N: std::sync::atomic::AtomicU32 = std::sync::atomic::AtomicU32::new(0);
        if N.fetch_add(1, Ordering::Relaxed) < 12 {
            crate::info!("compile ahead: a scanned draw of vs {:#x} ps {:#x} records nothing (as its registers read)", v.addr, p.as_ref().map_or(0, |p| p.addr));
        }
    }
    let k = match q {
        Some(true) => 4,
        Some(false) => 5,
        None => 6,
    };
    GFX[k].fetch_add(1, Ordering::Relaxed);
}

/// Prepares a program's batched form from `slot`'s registers and makes its
/// pipeline; the program is batched from then on (`shaders::batched_ready`).
fn run_batched(slot: &Slot) {
    let regs = slot.regs();
    let addr = ((regs.sh(0x20C) as u64) | ((regs.sh(0x20D) as u64 & 0xFF) << 32)) << 8;
    let made = match shaders::with_batched(addr, || shaders::prepare_compute(regs, slot.wave64())) {
        Ok(p) if !regs.missed.load(Ordering::Relaxed) && p.tr.spec.batched => {
            STATS[PREPARED].fetch_add(1, Ordering::Relaxed);
            let t = std::time::Instant::now();
            let r = super::exec::precompile(&p);
            if r == Some(true) {
                STATS[COMPILED].fetch_add(1, Ordering::Relaxed);
                STATS[COMPILE_NS].fetch_add(t.elapsed().as_nanos() as u64, Ordering::Relaxed);
            } else if r == Some(false) {
                STATS[PRESENT].fetch_add(1, Ordering::Relaxed);
            }
            r.map(|_| p.tr.key)
        }
        _ => None,
    };
    if made.is_some() {
        STATS[BATCHED_MADE].fetch_add(1, Ordering::Relaxed);
    } else {
        STATS[FAILED].fetch_add(1, Ordering::Relaxed);
    }
    shaders::batched_ready(addr, made.is_some());
    // Its run-time variant too when its formatted buffers were specialized (another key): a
    // later batched dispatch binding other formats takes it while its own key is made in the
    // background, instead of compiling that key where it stands (`shaders::with_fmt_rt`).
    if let Some(key) = made {
        if let Ok(q) = shaders::with_fmt_rt(|| shaders::with_batched(addr, || shaders::prepare_compute(regs, slot.wave64()))) {
            if !regs.missed.load(Ordering::Relaxed) && q.tr.spec.batched && q.tr.key != key {
                let t = std::time::Instant::now();
                if super::exec::precompile(&q) == Some(true) {
                    STATS[BATCHED_RT].fetch_add(1, Ordering::Relaxed);
                    STATS[COMPILE_NS].fetch_add(t.elapsed().as_nanos() as u64, Ordering::Relaxed);
                }
            }
        }
    }
}

/// Hands the stage of a program marked batchable while it had a pipeline
/// (`shaders::mark_batchable`) to the workers: its batched pipeline is made
/// there, its dispatches stay unbatched meanwhile.
pub fn submit_batched(regs: shaders::StageRegs, wave64: bool) {
    ensure_workers();
    STATS[BATCHED].fetch_add(1, Ordering::Relaxed);
    POOL.jobs.lock().push_front(Job::Batched(Slot::new(regs, wave64)));
    POOL.cv.notify_one();
}

fn run(slot: &Slot) {
    let regs = slot.regs();
    let p = match shaders::prepare_compute(regs, slot.wave64()) {
        Ok(p) if !regs.missed.load(Ordering::Relaxed) => p,
        r => {
            STATS[FAILED].fetch_add(1, Ordering::Relaxed);
            if super::exec::compile_trace_on() {
                // Each program's first failure and why (the scanned
                // registers' program address, COMPUTE_PGM_LO/HI).
                let why = match r {
                    Err(e) => e,
                    Ok(_) => format!("register {:#x} outside the snapshot", regs.missed_reg.load(Ordering::Relaxed)),
                };
                let addr = ((regs.sh(0x20C) as u64) | ((regs.sh(0x20D) as u64 & 0xFF) << 32)) << 8;
                static SEEN: Mutex<Option<std::collections::HashSet<u64>>> = Mutex::new(None);
                if SEEN.lock().get_or_insert_with(Default::default).insert(addr) {
                    crate::info!("compile ahead: cs {addr:#x} not prepared: {why}");
                }
            }
            return;
        }
    };
    STATS[PREPARED].fetch_add(1, Ordering::Relaxed);
    if mode() & 8 != 0 {
        return;
    }
    compile(&p);
}

fn compile(p: &Prepared) {
    let t = std::time::Instant::now();
    let r = super::exec::precompile(p);
    if r.is_some() {
        note_program(p.addr);
    }
    match r {
        Some(true) => {
            STATS[COMPILED].fetch_add(1, Ordering::Relaxed);
            STATS[COMPILE_NS].fetch_add(t.elapsed().as_nanos() as u64, Ordering::Relaxed);
        }
        Some(false) => {
            STATS[PRESENT].fetch_add(1, Ordering::Relaxed);
        }
        None => {
            STATS[FAILED].fetch_add(1, Ordering::Relaxed);
            if super::exec::compile_trace_on() {
                crate::info!("compile ahead: cs {:#x} pipeline not created", p.addr);
            }
        }
    }
}

/// Hands the compute stages of a scan (`Cp::prefetch_compiles`, after `ns`
/// of scanning) to the workers: the first of each program only.
pub fn submit(slots: Vec<Arc<Slot>>, ns: u64) {
    STATS[SCANS].fetch_add(1, Ordering::Relaxed);
    STATS[SCAN_NS].fetch_add(ns, Ordering::Relaxed);
    let known = KNOWN.lock().clone().unwrap_or_default();
    let mut seen: Vec<(u64, bool)> = Vec::new();
    let mut jobs = Vec::new();
    // Programs with a pipeline already come after the others, within the
    // limit: a new spec of one (other descriptors: the title's engine
    // programs at a level change) is a first use too, which the workers
    // find only by preparing it (20 deferred-lighting programs of 1-2's
    // first frames compiled one by one on the graphics queue, 1-1.9 s each,
    // FINDINGS.md 2026-10-24).
    let mut again = Vec::new();
    for s in slots {
        if s.regs().kind != StageKind::Compute {
            continue;
        }
        let r = s.regs();
        let addr = ((r.sh(0x20C) as u64) | ((r.sh(0x20D) as u64 & 0xFF) << 32)) << 8;
        let k = (addr, s.wave64());
        if addr == 0 || seen.contains(&k) {
            continue;
        }
        seen.push(k);
        if known.contains(&addr) {
            again.push(s);
            continue;
        }
        if jobs.len() >= MAX_JOBS {
            STATS[DROPPED].fetch_add(1, Ordering::Relaxed);
            continue;
        }
        jobs.push(s);
    }
    let room = MAX_JOBS.saturating_sub(jobs.len());
    STATS[AGAIN].fetch_add(again.len().min(room) as u64, Ordering::Relaxed);
    jobs.extend(again.into_iter().take(room));
    if jobs.is_empty() || mode() & 4 != 0 {
        return;
    }
    ensure_workers();
    STATS[JOBS].fetch_add(jobs.len() as u64, Ordering::Relaxed);
    POOL.jobs.lock().extend(jobs.into_iter().map(Job::Regs));
    POOL.cv.notify_all();
}

/// Hands a compute stage prepared while its command processor waited on a
/// label, whose pipeline is missing, to the workers (ahead of the queued
/// scans: it is the nearest to execution).
pub fn submit_prepared(p: Prepared) {
    // Once while queued (each label wait prepares the stages after it).
    {
        let mut q = QUEUED.lock();
        if q.contains(&p.tr.key) {
            return;
        }
        q.push(p.tr.key);
    }
    ensure_workers();
    STATS[AT_WAITS].fetch_add(1, Ordering::Relaxed);
    POOL.jobs.lock().push_front(Job::Prepared(Box::new(p)));
    POOL.cv.notify_one();
}

/// Scans at first uses by why they ended (`ahead::END_*` or an opcode:
/// buffer end, the command limit, another packet) and the commands found.
static SCAN_ENDS: [AtomicU64; 4] = [const { AtomicU64::new(0) }; 4];
/// Those ended by another packet, by `why` (an opcode, 0x100 + a private
/// operation, or another `ahead::END_*`).
static SCAN_STOPS: [AtomicU64; 0x140] = [const { AtomicU64::new(0) }; 0x140];

/// A scan at a first use ended for `why` after finding `n` commands.
pub fn note_scan_end(why: usize, n: usize) {
    let k = match why {
        super::ahead::END_BUFFER => 0,
        super::ahead::END_MAX => 1,
        _ => 2,
    };
    if k == 2 {
        SCAN_STOPS[why.min(0x13F)].fetch_add(1, Ordering::Relaxed);
    }
    SCAN_ENDS[k].fetch_add(1, Ordering::Relaxed);
    SCAN_ENDS[3].fetch_add(n as u64, Ordering::Relaxed);
}

/// One status line ("" before any scan).
pub fn status() -> String {
    let s: Vec<u64> = STATS.iter().map(|c| c.load(Ordering::Relaxed)).collect();
    let g: Vec<u64> = GFX.iter().map(|c| c.load(Ordering::Relaxed)).collect();
    let gfx = if g[0] > 0 {
        format!(
            "    graphics pipelines queued ahead at draws that waited: {} scans ({} ms), {} draws found, {} handed over ({} over the per-scan limit), {} prepared, {} pipelines queued, {} made or queued already, {} recording nothing, {} not prepared\n",
            g[0],
            g[9] / 1_000_000,
            g[1],
            g[2],
            g[8],
            g[3],
            g[4],
            g[5],
            g[6],
            g[7]
        )
    } else {
        String::new()
    };
    if s[SCANS] == 0 && s[AT_WAITS] == 0 {
        return gfx;
    }
    let (waits, wait_ns, dups) = super::exec::inflight_stats();
    let mut stops: Vec<(usize, u64)> = SCAN_STOPS.iter().enumerate().map(|(k, c)| (k, c.load(Ordering::Relaxed))).filter(|x| x.1 > 0).collect();
    stops.sort_by(|a, b| b.1.cmp(&a.1));
    let stops: Vec<String> = stops.iter().take(6).map(|(k, n)| format!("{k:#x} {n}")).collect();
    let stops = if stops.is_empty() { String::new() } else { format!(" (by {})", stops.join(", ")) };
    format!(
        "    pipelines compiled ahead: {} scans at first uses ({} ms; ended by their buffer's end {}, the command limit {}, another packet {}{stops}; {} commands found), {} programs handed to {} workers ({} of them with a pipeline already, for other specs; {} over the per-scan limit) and {} stages prepared at label waits, {} prepared, {} pipelines created ({} ms of compiles), {} present already, {} failed or not prepared; command processors waited {} times for a pipeline in flight ({} ms), {} compiles of one pipeline avoided, {} programs batchable from a run's first dispatch, {} batched forms made in the background ({} handed over; {} run-time variants beside them)\n",
        s[SCANS],
        s[SCAN_NS] / 1_000_000,
        SCAN_ENDS[0].load(Ordering::Relaxed),
        SCAN_ENDS[1].load(Ordering::Relaxed),
        SCAN_ENDS[2].load(Ordering::Relaxed),
        SCAN_ENDS[3].load(Ordering::Relaxed),
        s[JOBS],
        WORKERS.load(Ordering::Relaxed),
        s[AGAIN],
        s[DROPPED],
        s[AT_WAITS],
        s[PREPARED],
        s[COMPILED],
        s[COMPILE_NS] / 1_000_000,
        s[PRESENT],
        s[FAILED],
        waits,
        wait_ns / 1_000_000,
        dups,
        super::exec::BATCHABLE_AHEAD.load(Ordering::Relaxed),
        s[BATCHED_MADE],
        s[BATCHED],
        s[BATCHED_RT]
    ) + &gfx
}

/// Optional aggregate limit for driver shader compiles, shared by foreground, compute,
/// graphics and preparation callers. Relay pools are small too, but separate pools must
/// not each consume the whole CPU budget. Unset preserves normal Windows/player policy.
struct CompileGate {
    active: Mutex<usize>,
    wake: Condvar,
    limit: usize,
}
impl CompileGate {
    fn enter(&self) -> CompilePermit<'_> {
        let mut active = self.active.lock();
        while *active >= self.limit { self.wake.wait(&mut active); }
        *active += 1;
        CompilePermit(Some(self))
    }
}
pub struct CompilePermit<'a>(Option<&'a CompileGate>);
impl Drop for CompilePermit<'_> {
    fn drop(&mut self) {
        if let Some(gate) = self.0 {
            *gate.active.lock() -= 1;
            gate.wake.notify_one();
        }
    }
}
pub fn driver_permit() -> CompilePermit<'static> {
    static GATE: std::sync::OnceLock<Option<CompileGate>> = std::sync::OnceLock::new();
    match GATE.get_or_init(|| std::env::var("DS_COMPILE_WORKERS").ok().and_then(|s| s.parse::<usize>().ok())
        .filter(|n| *n > 0).map(|n| CompileGate { active: Mutex::new(0), wake: Condvar::new(), limit: n.min(16) })) {
        Some(gate) => gate.enter(),
        None => CompilePermit(None),
    }
}

#[cfg(test)]
mod compile_limit_tests {
    use super::*;
    #[test]
    fn aggregate_limit_and_release() {
        let gate = CompileGate { active: Mutex::new(0), wake: Condvar::new(), limit: 2 };
        let peak = AtomicUsize::new(0);
        let live = AtomicUsize::new(0);
        std::thread::scope(|scope| {
            for _ in 0..8 {
                scope.spawn(|| {
                    let _permit = gate.enter();
                    let n = live.fetch_add(1, Ordering::SeqCst) + 1;
                    peak.fetch_max(n, Ordering::SeqCst);
                    std::thread::sleep(std::time::Duration::from_millis(5));
                    live.fetch_sub(1, Ordering::SeqCst);
                });
            }
        });
        assert!((1..=2).contains(&peak.load(Ordering::SeqCst)));
        assert_eq!(*gate.active.lock(), 0);
    }
}
