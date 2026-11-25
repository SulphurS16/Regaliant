//! Parallel recording of synchronization-free runs (STATUS: Next steps 1,
//! parallel recording).
//!
//! A long run of commands without synchronization packets between them is
//! split at command boundaries into chunks. (The graphics queue's runs also
//! hold the title's GPU timestamps, label waits that held when the run was
//! planned and multi-draws, `Cp::run_pass`, `Cp::run_multi`: none of them
//! orders later commands after earlier ones on the hardware.) The main command processor
//! executes the first chunk itself; the others form a queue (`Run`) that the
//! workers take in program order: a worker is a command processor of its own
//! (`Cp`, started from a copy of the register state at the chunk's start)
//! whose executor is deferred (`Exec::deferred`): it records the chunk's
//! draws and dispatches with the same code into batches that it closes, then
//! waits for the chunk's turn, submits them, posts the state at the chunk's
//! end and takes the next chunk not taken. The main command processor gives
//! a chunk its turn once its own work before the chunk went to the GPU
//! (`Cp::par_collect`) and continues with the state at the chunk's end; a
//! chunk that ran to its end passes the turn to the next itself, so the GPU
//! starts on a run while its later chunks are still being recorded (`tune
//! par_split`: chunks per recorder). A command the worker cannot record
//! without flushing or waiting for its own unsubmitted work stops the chunk
//! there; the main command processor executes the rest of that chunk itself
//! and then gives the next chunk its turn. (Long runs through the title's
//! GPU-side synchronization, `tune long_runs`, chained, parked and sealed
//! work and runs past label waits were level and are removed: FINDINGS.md
//! 2026-11-01.) Inside a stretch the graphics command processor records past
//! a label wait, runs are held (`Run::held`, `tune held_runs`, since
//! 2026-11-13): the workers post each chunk once recorded and keep its
//! batches until the stretch settles, when the command processor submits
//! them in program order (`Cp::held_collect`, `Cp::held_commit`).

use super::cp::{ChunkState, Cp};
use parking_lot::{Condvar, Mutex};
use std::sync::atomic::{AtomicBool, AtomicU64, AtomicUsize, Ordering};
use std::sync::Arc;

/// Whole synchronization-free graphics runs on workers, including their first chunk.
/// Experimental: keep off until cooled A/B and boundary images establish a benefit.
pub fn whole_run() -> bool {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ON.get_or_init(|| std::env::var("DS_PAR_WHOLE_RUN").ok().as_deref() == Some("1"))
}
pub static WHOLE_RUNS: AtomicU64 = AtomicU64::new(0);
pub static WHOLE_COMMANDS: AtomicU64 = AtomicU64::new(0);

/// Chunk workers wanted (`tune par`; 0: runs are recorded by the command
/// processor alone). By default half the logical CPUs, 2 to 8: on 24 logical
/// CPUs 6, 7 and 8 workers recorded +2.0%, +2.2% and +2.4% against 5
/// (FINDINGS.md 2026-10-14) and 12 no more than 8; on 8 logical CPUs 4
/// workers +6% against 2 and 8 none more (2026-10-21, with the title's idle
/// job workers asleep: a third was for CPUs shared with their polling).
pub fn workers() -> usize {
    if super::nan_probe::early_on() { return 0; }
    match super::tune::par() {
        1000 => {
            static AUTO: std::sync::OnceLock<usize> = std::sync::OnceLock::new();
            *AUTO.get_or_init(|| std::thread::available_parallelism().map_or(5, |n| (n.get() / 2).clamp(2, 8)))
        }
        v => (v as usize).min(16),
    }
}

/// The fewest commands a graphics run is split at (`tune par_min`; 0:
/// automatic): 64 with eight workers or more, else 128. On 24 logical CPUs
/// 64 against 128 gave +1.6% flips/s at +5.4% CPU per command (more, smaller
/// chunks; FINDINGS.md 2026-10-16): worth it while CPUs are spare, a loss
/// once they are not.
/// The shortest run split into parallel chunks whatever `par_min` and
/// `par_min_compute` say (`tune par_floor`, default 48).
pub fn floor() -> usize {
    super::tune::par_floor().max(2) as usize
}

pub fn min_run() -> usize {
    match super::tune::par_min() {
        0 => {
            if workers() >= 8 {
                64
            } else {
                128
            }
        }
        v => v as usize,
    }
}

/// The fewest commands a graphics run is split at (`min_run` and `floor`). With `tune par_min`
/// 0 (automatic) and eight workers or more it follows the bottleneck: 32 while the GPU is busy
/// under 70% of the frames (`hitch::gpu_busy_share`: the command processors leave it waiting;
/// TowerL Top x1.013 against 64, its CP 14.5 against 15.1 ms a flip), the 64 again once it is
/// busy over 85% (where the GPU limits, the more chunks' seams cost it: x0.98 at the swamp;
/// FINDINGS.md 2026-11-13).
pub fn graphics_min() -> usize {
    let fixed = min_run().max(floor());
    if super::tune::par_min() != 0 || workers() < 8 {
        return fixed;
    }
    static CP_LIMITED: AtomicBool = AtomicBool::new(false);
    let share = super::hitch::gpu_busy_share();
    let was = CP_LIMITED.load(Ordering::Relaxed);
    let limited = if share < 0.70 {
        true
    } else if share > 0.85 {
        false
    } else {
        was
    };
    if limited != was {
        CP_LIMITED.store(limited, Ordering::Relaxed);
        MIN_SWITCHES.fetch_add(1, Ordering::Relaxed);
    }
    if limited {
        MIN_LOW.fetch_add(1, Ordering::Relaxed);
        32.min(fixed)
    } else {
        MIN_HIGH.fetch_add(1, Ordering::Relaxed);
        fixed
    }
}

/// `graphics_min`'s answers: [32 (the GPU waiting), the fixed threshold, switches between them].
static MIN_LOW: AtomicU64 = AtomicU64::new(0);
static MIN_HIGH: AtomicU64 = AtomicU64::new(0);
static MIN_SWITCHES: AtomicU64 = AtomicU64::new(0);

/// Chunks per recorder (the main command processor and each worker) a run
/// is split into (`tune par_split`; 1: one each, all submitted when the
/// main command processor's own chunk is done).
pub fn split() -> usize {
    (super::tune::par_split() as usize).clamp(1, 16)
}

/// The main command processor's chunk relative to a worker's.
pub fn main_weight() -> f64 {
    let fixed = super::tune::par_weight();
    if fixed > 0 {
        fixed.min(4000) as f64 / 1000.0
    } else {
        1.0
    }
}

/// A chunk as its worker left it.
pub struct Done {
    /// Where it stopped (None: it ran to its end).
    pub stopped: Option<u64>,
    /// The command-processor state where it stopped or ended.
    pub state: ChunkState,
    /// The sequence of its last batch (None: it submitted none).
    pub last: Option<u64>,
    /// Why it stopped ("" when it ran to its end).
    pub why: &'static str,
    /// TSC cycles each command recorded took, in order (`tune par_cost`;
    /// the first one with the chunk's setup).
    pub costs: Vec<u32>,
    /// What its batches write (`DS_RUN_PROBE` only).
    pub writes: super::ranges::Ranges,
    /// A held run's chunk (`Run::held`): its record (None: failed, its batches dropped), and
    /// the worker whose executor keeps its batches (`Pool::get`).
    pub held: Option<Box<super::exec::HeldChunk>>,
    pub worker: usize,
}

/// The chunks of a parallel run that the workers record (all but the main
/// command processor's), in program order.
pub struct Run {
    /// Chunk i: its start state (taken by the worker that records it), its
    /// first packet and the packet after it.
    chunks: Vec<(Mutex<Option<Box<ChunkState>>>, u64, u64)>,
    /// The next chunk not taken.
    next: AtomicUsize,
    /// Chunks whose turn came: chunk i submits once this exceeds i.
    turn: AtomicUsize,
    /// Chunks posted (in order: chunk i's once this exceeds i).
    posted: AtomicUsize,
    results: Mutex<Vec<Option<Done>>>,
    cancelled: AtomicBool,
    /// Workers started on the run that have not left it.
    active: AtomicUsize,
    lock: Mutex<()>,
    cv: Condvar,
    /// When the run was planned (the workers' start latency).
    born: std::time::Instant,
    /// The workers time each command (`tune par_cost`).
    pub timed: bool,
    /// Front end 2: the command processor's view when it planned the run
    /// (`fe2::gfx_views`; NO_VIEW: the epoch now).
    pub fe2_view: u32,
    /// Front end 2: the run's id for uploads its parts share (`Exec::
    /// fe2_share`): the main command processor's own part before the
    /// workers' chunks is part 0, chunk i part i + 1.
    pub id: u64,
    /// The label waits the run holds (their packets' addresses, ascending):
    /// they held in guest memory when it was planned (`Cp::run_pass`), or
    /// on a label write earlier in the run (`Cp::compute_pass`).
    pub passed_waits: std::sync::Arc<[u64]>,
    /// The CP writes the run holds (`Cp::compute_pass`): packet address,
    /// guest address, bytes.
    pub cp_writes: std::sync::Arc<[(u64, u64, u64)]>,
    /// The graphics queue's handshake census: the run's segment
    /// (`gpu::hscensus`; 0: not counted).
    pub hs_seg: u64,
    /// Planned inside a stretch recorded past a label wait (`tune held_runs`): its workers
    /// record each chunk held (`Exec::held_chunk_begin`) and post it as recorded without
    /// waiting for a turn, its batches left in their executors until the stretch settles
    /// (`Cp::held_collect`, `Cp::held_commit`); `id` is the stretch's share and chunk i its
    /// part `part_base` + i + 1. `held_writes` and `held_keys`: what the held work before it
    /// writes and the images that work notes (a chunk's CPU read or use of one stops it).
    pub held: bool,
    pub part_base: u32,
    pub held_writes: Arc<super::ranges::Ranges>,
    pub held_keys: Arc<std::collections::HashMap<super::images::ImageKey, u8>>,
    /// A held run's chunks recorded (posted in any order: `collect_held`).
    recorded: Vec<AtomicBool>,
}

/// Run ids (`Run::id`; 0: none).
static RUN_IDS: AtomicU64 = AtomicU64::new(1);

/// A new id in the runs' space (`Run::id`): a stretch's share of front end 2's uploads
/// (`Cp::past_wait_start`, `tune held_runs`).
pub fn new_run_id() -> u64 {
    RUN_IDS.fetch_add(1, Ordering::Relaxed)
}

impl Run {
    /// The run of `chunks` ([start state, first packet, end]).
    pub fn new(chunks: Vec<(Box<ChunkState>, u64, u64)>) -> Run {
        let n = chunks.len();
        Run {
            chunks: chunks.into_iter().map(|(s, a, b)| (Mutex::new(Some(s)), a, b)).collect(),
            next: AtomicUsize::new(0),
            turn: AtomicUsize::new(0),
            posted: AtomicUsize::new(0),
            results: Mutex::new((0..n).map(|_| None).collect()),
            cancelled: AtomicBool::new(false),
            active: AtomicUsize::new(0),
            lock: Mutex::new(()),
            cv: Condvar::new(),
            born: std::time::Instant::now(),
            timed: false,
            fe2_view: super::fe2::NO_VIEW,
            id: RUN_IDS.fetch_add(1, Ordering::Relaxed),
            passed_waits: std::sync::Arc::from(Vec::new()),
            cp_writes: std::sync::Arc::from(Vec::new()),
            hs_seg: 0,
            held: false,
            part_base: 0,
            held_writes: Arc::new(super::ranges::Ranges::new()),
            held_keys: Arc::new(std::collections::HashMap::new()),
            recorded: (0..n).map(|_| AtomicBool::new(false)).collect(),
        }
    }

    /// Waits for chunk `i` of a held run to be recorded (`Run::held`): it and how long the
    /// wait took (ns).
    pub fn collect_held(&self, i: usize) -> (Done, u64) {
        let ready = || self.recorded[i].load(Ordering::Acquire);
        let ns = if ready() {
            0
        } else {
            let t = std::time::Instant::now();
            let mut spins = 0;
            let limit = super::tune::par_spins();
            while spins < limit && !ready() {
                std::hint::spin_loop();
                spins += 1;
            }
            let mut g = self.lock.lock();
            let mut logged = false;
            while !ready() {
                if self.cv.wait_for(&mut g, std::time::Duration::from_secs(3)).timed_out() && !logged && !ready() {
                    logged = true;
                    crate::warn!("gpu: waiting > 3 s for chunk {i} of a held run of {} chunks: taken {}, workers on it {}, cancelled {}", self.chunks.len(), self.next.load(Ordering::Acquire), self.active.load(Ordering::Acquire), self.cancelled.load(Ordering::Acquire));
                }
            }
            t.elapsed().as_nanos() as u64
        };
        WAIT_NS.fetch_add(ns, Ordering::Relaxed);
        let d = self.results.lock()[i].take().expect("held chunk collected twice");
        (d, ns)
    }

    fn notify(&self) {
        let _g = self.lock.lock();
        self.cv.notify_all();
    }

    /// Waits until `ready` holds (spinning briefly first: the chunks of a run
    /// are handed over within microseconds of each other); how long (ns).
    fn wait_until(&self, ready: impl Fn() -> bool) -> u64 {
        if ready() {
            return 0;
        }
        let t = std::time::Instant::now();
        let mut spins = 0;
        let limit = super::tune::par_spins();
        while spins < limit && !ready() {
            std::hint::spin_loop();
            spins += 1;
        }
        if !ready() {
            let mut g = self.lock.lock();
            while !ready() {
                self.cv.wait(&mut g);
            }
        }
        t.elapsed().as_nanos() as u64
    }

    /// Chunk `i` may submit (the main command processor's work before it went
    /// to the GPU).
    pub fn grant(&self, i: usize) {
        if self.turn.fetch_max(i + 1, Ordering::AcqRel) <= i {
            self.notify();
        }
    }

    /// Waits for chunk `i` to be posted: it and how long the wait took (ns).
    /// (A wait of more than 3 s logs the run's state once: a run that cannot
    /// go on.)
    pub fn collect(&self, i: usize) -> (Done, u64) {
        let ready = || self.posted.load(Ordering::Acquire) > i;
        let ns = if ready() {
            0
        } else {
            let t = std::time::Instant::now();
            let mut spins = 0;
            let limit = super::tune::par_spins();
            while spins < limit && !ready() {
                std::hint::spin_loop();
                spins += 1;
            }
            let mut g = self.lock.lock();
            let mut logged = false;
            while !ready() {
                if self.cv.wait_for(&mut g, std::time::Duration::from_secs(3)).timed_out() && !logged && !ready() {
                    logged = true;
                    crate::warn!(
                        "gpu: waiting > 3 s for chunk {i} of a parallel run of {} chunks: posted {}, turn {}, taken {}, workers on it {}, cancelled {}",
                        self.chunks.len(),
                        self.posted.load(Ordering::Acquire),
                        self.turn.load(Ordering::Acquire),
                        self.next.load(Ordering::Acquire),
                        self.active.load(Ordering::Acquire),
                        self.cancelled.load(Ordering::Acquire)
                    );
                }
            }
            t.elapsed().as_nanos() as u64
        };
        WAIT_NS.fetch_add(ns, Ordering::Relaxed);
        let d = self.results.lock()[i].take().expect("parallel chunk posted twice");
        (d, ns)
    }

    /// Abandons the run: chunks not yet submitted are dropped by their
    /// workers; returns once every worker left it.
    pub fn cancel(&self) {
        self.cancelled.store(true, Ordering::SeqCst);
        self.notify();
        let t = std::time::Instant::now();
        let mut logged = false;
        while self.active.load(Ordering::SeqCst) != 0 {
            let mut g = self.lock.lock();
            if self.active.load(Ordering::SeqCst) == 0 {
                break;
            }
            let _ = self.cv.wait_for(&mut g, std::time::Duration::from_millis(100));
            if !logged && t.elapsed() > std::time::Duration::from_secs(3) {
                logged = true;
                crate::warn!("gpu: a cancelled run waits > 3 s for its workers to leave ({} on it, posted {} of {}, turn {}, taken {})", self.active.load(Ordering::Acquire), self.posted.load(Ordering::Acquire), self.chunks.len(), self.turn.load(Ordering::Acquire), self.next.load(Ordering::Acquire));
            }
        }
    }

    fn leave(&self) {
        if self.active.fetch_sub(1, Ordering::AcqRel) == 1 {
            self.notify();
        }
    }
}

pub struct Worker {
    /// Its index in its pool (`Pool::get`).
    idx: usize,
    /// The command processor it records chunks with (its own executor).
    cp: Mutex<Cp>,
    /// Runs given it, taken in order, and whether there is one (what a
    /// spinning worker looks at).
    job: Mutex<std::collections::VecDeque<Arc<Run>>>,
    has_job: AtomicBool,
    /// Its pool's wake word (`Pool::wake`): an idle worker sleeps on it.
    wake: Arc<AtomicU64>,
}

/// Statistics: runs split, chunks handed to workers, chunks that stopped,
/// commands per chunk..., main command processor's waits for workers (ns),
/// workers' waits for their chunks' turns (ns).
pub static RUNS: AtomicU64 = AtomicU64::new(0);
pub static CHUNKS: AtomicU64 = AtomicU64::new(0);
pub static STOPS: AtomicU64 = AtomicU64::new(0);
pub static COMMANDS: AtomicU64 = AtomicU64::new(0);
pub static WAIT_NS: AtomicU64 = AtomicU64::new(0);
pub static TURN_NS: AtomicU64 = AtomicU64::new(0);
/// Workers' time per phase (ns): from the run's planning to their first
/// chunk of it (per worker and run), setting a chunk's state up, recording
/// it, closing its batch, submitting and posting it; and the count of first
/// chunks.
static PHASE_NS: [AtomicU64; 5] = [const { AtomicU64::new(0) }; 5];
static STARTS: AtomicU64 = AtomicU64::new(0);
pub static CANCELLED: AtomicU64 = AtomicU64::new(0);
/// The workers' idle census ([graphics, compute] x [wake-ups from their idle
/// sleep, of them finding no run (the pool's word wakes every idle worker),
/// their own cycles in runs, of them waiting for chunks' turns]; the cycles
/// with `DS_IDLE_CENSUS=1`), and the kinds' thread cycles and the flip count at
/// the window's start (`reset_stats`).
static IDLE: [[AtomicU64; 4]; 2] = [const { [const { AtomicU64::new(0) }; 4] }; 2];
static IDLE_BASE: Mutex<([u64; 2], u64)> = Mutex::new(([0; 2], 0));
static STOP_WHY: Mutex<Vec<(&'static str, u64)>> = Mutex::new(Vec::new());

/// Recording past runs (`tune past_runs`): [runs gone past, committed,
/// dropped, commands recorded past them, ns from going past to the stop,
/// ns waited at the stop for the chunks]; stops and drops by reason.
pub static PAST: [AtomicU64; 6] = [const { AtomicU64::new(0) }; 6];
static PAST_STOPS: Mutex<Vec<(&'static str, u64)>> = Mutex::new(Vec::new());
static PAST_DROPS: Mutex<Vec<(&'static str, u64)>> = Mutex::new(Vec::new());

/// A stretch recorded past a run ended at a stop (`why`); `dropped`: why
/// it was dropped.
pub fn note_past(why: &'static str, dropped: Option<&'static str>) {
    let bump = |l: &Mutex<Vec<(&'static str, u64)>>, w: &'static str| {
        let mut g = l.lock();
        match g.iter_mut().find(|e| e.0 == w) {
            Some(e) => e.1 += 1,
            None => g.push((w, 1)),
        }
    };
    bump(&PAST_STOPS, why);
    if let Some(d) = dropped {
        bump(&PAST_DROPS, d);
    }
}

/// Runs the planner saw, by length (commands): [<32, <64, <192, <512,
/// <2048, more] x [runs, commands]; and by what ended them (PM4 opcode,
/// 0x100 + private operation, 0x130 buffer end, 0x131 limit, 0x133
/// predicated) x [runs, commands].
static LENGTHS: Mutex<[[u64; 2]; 6]> = Mutex::new([[0; 2]; 6]);
static ENDS: Mutex<Vec<(usize, u64, u64)>> = Mutex::new(Vec::new());
/// The same for the compute queues' runs (`cp::compute_workers`).
static LENGTHS_C: Mutex<[[u64; 2]; 6]> = Mutex::new([[0; 2]; 6]);
static ENDS_C: Mutex<Vec<(usize, u64, u64)>> = Mutex::new(Vec::new());
/// The graphics queue's runs of fewer than 32 commands by what ended them.
static ENDS_SHORT: Mutex<Vec<(usize, u64, u64)>> = Mutex::new(Vec::new());

/// A run of `n` commands ended by `why` (planner statistics); `compute`: of
/// a compute queue.
pub fn note_run(n: usize, why: usize, compute: bool) {
    let (lengths, ends) = if compute { (&LENGTHS_C, &ENDS_C) } else { (&LENGTHS, &ENDS) };
    let b = match n {
        0..32 => 0,
        32..64 => 1,
        64..192 => 2,
        192..512 => 3,
        512..2048 => 4,
        _ => 5,
    };
    let mut g = lengths.lock();
    g[b][0] += 1;
    g[b][1] += n as u64;
    let note = |ends: &Mutex<Vec<(usize, u64, u64)>>| {
        let mut e = ends.lock();
        match e.iter_mut().find(|x| x.0 == why) {
            Some(x) => {
                x.1 += 1;
                x.2 += n as u64;
            }
            None => e.push((why, 1, n as u64)),
        }
    };
    note(ends);
    if !compute && n < 32 {
        note(&ENDS_SHORT);
    }
}

/// A chunk stopped early for `why`.
pub fn note_stop(why: &'static str) {
    let mut g = STOP_WHY.lock();
    match g.iter_mut().find(|e| e.0 == why) {
        Some(e) => e.1 += 1,
        None => g.push((why, 1)),
    }
}

/// The workers of a queue's command processor (created on first use).
pub struct Pool {
    workers: Vec<Arc<Worker>>,
    /// Moves when runs were handed out (`Worker::give`, `wake`): the idle
    /// workers sleep on it (`WaitOnAddress`), so one call wakes them all
    /// (a condition variable each took ~21 us of the planner per run,
    /// FINDINGS.md 2026-10-16).
    wake: Arc<AtomicU64>,
}

impl Pool {
    pub fn new() -> Pool {
        Pool { workers: Vec::new(), wake: Arc::new(AtomicU64::new(0)) }
    }

    /// Wakes the idle workers to the runs given them (`Worker::give`).
    pub fn wake(&self) {
        self.wake.fetch_add(1, Ordering::AcqRel);
        unsafe { crate::platform::os::Win32::System::Threading::WakeByAddressAll(Arc::as_ptr(&self.wake) as *const core::ffi::c_void) };
    }

    /// Worker `i` of the command processor `name` (started on first use);
    /// None if it cannot be made.
    pub fn worker(&mut self, i: usize, name: &str, compute: bool) -> Option<Arc<Worker>> {
        while self.workers.len() <= i {
            let n = self.workers.len();
            let wname = format!("{name}-par{n}");
            let mut cp = Cp::new(&wname, compute);
            if let Err(e) = cp.make_worker_exec() {
                crate::warn!("gpu[{wname}]: no executor for a parallel worker ({e})");
                return None;
            }
            let w = Arc::new(Worker { idx: n, cp: Mutex::new(cp), job: Mutex::new(Default::default()), has_job: AtomicBool::new(false), wake: self.wake.clone() });
            let w2 = w.clone();
            let tname = wname.clone();
            let qname = name.to_string();
            let r = crate::thread::spawn_host(
                &wname,
                8 << 20,
                Box::new(move || {
                    crate::log::set_thread_name(&tname);
                    crate::exception::arm_watch_current_thread(false);
                    // (The compute queues' workers sample as their own kind; their
                    // CPU counts as "par" in `threadcpu`.)
                    super::profile_this_thread(if compute { "cpar" } else { "par" });
                    if !compute {
                        super::forward::enable_here();
                    } else if let Some(id) = qname.strip_prefix("gpu-acb").and_then(|h| u32::from_str_radix(h.trim_start_matches("0x"), 16).ok()) {
                        super::prio::set_compute_worker(id);
                    }
                    run_worker(&w2, &tname, compute);
                }),
            );
            if let Err(e) = r {
                crate::warn!("gpu[{wname}]: no thread for a parallel worker ({e})");
                return None;
            }
            self.workers.push(w);
        }
        Some(self.workers[i].clone())
    }
}

impl Default for Pool {
    fn default() -> Pool {
        Pool::new()
    }
}

impl Pool {
    /// Worker `i`, if it was started.
    pub fn get(&self, i: usize) -> Option<Arc<Worker>> {
        self.workers.get(i).cloned()
    }

    /// Workers started.
    pub fn count(&self) -> usize {
        self.workers.len()
    }
}

fn run_worker(w: &Worker, name: &str, compute: bool) {
    let k = compute as usize;
    let census = super::threadcpu::idle_census();
    loop {
        super::prio::apply();
        let mut woke = false;
        let run = loop {
            // The word before looking: a run given after this wakes the
            // sleep below (or it does not start).
            let seen = w.wake.load(Ordering::Acquire);
            {
                let mut job = w.job.lock();
                if let Some(r) = job.pop_front() {
                    w.has_job.store(!job.is_empty(), Ordering::Release);
                    break r;
                }
            }
            if woke {
                IDLE[k][1].fetch_add(1, Ordering::Relaxed);
            }
            unsafe {
                crate::platform::os::Win32::System::Threading::WaitOnAddress(
                    Arc::as_ptr(&w.wake) as *const core::ffi::c_void,
                    &seen as *const u64 as *const core::ffi::c_void,
                    8,
                    crate::platform::os::Win32::System::Threading::INFINITE,
                )
            };
            IDLE[k][0].fetch_add(1, Ordering::Relaxed);
            woke = true;
        };
        let c0 = if census { super::threadcpu::own_cycles() } else { 0 };
        // (Counted on the run from here: a cancel waits for the workers
        // recording or committing its chunks, not for those given it that
        // are still busy with earlier runs.)
        run.active.fetch_add(1, Ordering::SeqCst);
        take_chunks(w, &run, name, k, census);
        run.leave();
        if census {
            IDLE[k][2].fetch_add(super::threadcpu::own_cycles().wrapping_sub(c0), Ordering::Relaxed);
        }
    }
}

/// Records, submits and posts chunks of `run` until none is left or the run
/// is cancelled.
fn take_chunks(w: &Worker, run: &Run, name: &str, k: usize, census: bool) {
    let mut first = true;
    let lap = |k: usize, t: &mut std::time::Instant| {
        let now = std::time::Instant::now();
        PHASE_NS[k].fetch_add((now - *t).as_nanos() as u64, Ordering::Relaxed);
        *t = now;
    };
    loop {
        if run.cancelled.load(Ordering::SeqCst) {
            return;
        }
        let i = run.next.fetch_add(1, Ordering::AcqRel);
        let Some((state, from, to)) = run.chunks.get(i) else { return };
        let Some(state) = state.lock().take() else { return };
        let (from, to) = (*from, *to);
        let mut t = std::time::Instant::now();
        if std::mem::take(&mut first) {
            PHASE_NS[0].fetch_add((t - run.born).as_nanos() as u64, Ordering::Relaxed);
            STARTS.fetch_add(1, Ordering::Relaxed);
        }
        let mut cp = w.cp.lock();
        cp.set_chunk_state(&state);
        cp.set_hs_seg(run.hs_seg);
        cp.chunk_waits = run.passed_waits.clone();
        cp.set_run_writes(run.cp_writes.iter().filter(|w| w.0 < from).map(|w| (w.1, w.2)).collect());
        cp.fe2_set_view(run.fe2_view);
        cp.fe2_set_run(run.id, if run.held { run.part_base as usize + i + 1 } else { i + 1 });
        cp.chunk_start();
        if run.held {
            cp.held_chunk_begin(run.held_writes.clone(), run.held_keys.clone());
        }
        cp.chunk_costs = run.timed.then(|| Vec::with_capacity(512));
        lap(1, &mut t);
        let r = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| cp.run_chunk(from, to)));
        let failed = r.is_err();
        let stopped = r.unwrap_or(Some(from));
        lap(2, &mut t);
        cp.chunk_finish();
        cp.fe2_set_run(0, 0);
        if run.held {
            let held = cp.held_chunk_end();
            lap(3, &mut t);
            post_held(run, i, cp, stopped, failed, state, held, w.idx);
            continue;
        }
        lap(3, &mut t);
        let ready = || run.turn.load(Ordering::Acquire) > i || run.cancelled.load(Ordering::Acquire);
        if !ready() {
            let tl = super::timeline::mark();
            let c0 = if census { super::threadcpu::own_cycles() } else { 0 };
            TURN_NS.fetch_add(run.wait_until(ready), Ordering::Relaxed);
            if census {
                IDLE[k][3].fetch_add(super::threadcpu::own_cycles().wrapping_sub(c0), Ordering::Relaxed);
            }
            super::timeline::end(name, tl, || "wait: chunk turn".into());
        }
        commit(run, i, cp, stopped, failed, state);
    }
}

/// Chunk `i` of `run` at its turn, recorded in `cp` (or the run cancelled
/// before its turn: discarded, not posted): submitted and posted, or
/// discarded and posted for the main command processor to record again;
/// one that ran to its end gives the next chunk its turn.
fn commit(run: &Run, i: usize, mut cp: parking_lot::MutexGuard<'_, Cp>, stopped: Option<u64>, failed: bool, state: Box<ChunkState>) {
    let mut t = std::time::Instant::now();
    let granted = run.turn.load(Ordering::Acquire) > i;
    if !granted || failed {
        // Cancelled before its turn or failed (the main command processor
        // executes it from its start): never submitted.
        cp.worker_discard();
    }
    if !granted {
        return;
    }
    let done = if failed {
        Done { stopped, state: *state, last: None, why: "failed", costs: Vec::new(), writes: Default::default(), held: None, worker: 0 }
    } else {
        let writes = if super::runprobe::on() || super::tune::past_runs() { cp.worker_writes() } else { Default::default() };
        let last = cp.worker_submit();
        let why = if stopped.is_some() { cp.worker_stop_why() } else { "" };
        let state = cp.chunk_state();
        Done { stopped, state, last, why, costs: cp.chunk_costs.take().unwrap_or_default(), writes, held: None, worker: 0 }
    };
    drop(cp);
    run.results.lock()[i] = Some(done);
    run.posted.store(i + 1, Ordering::Release);
    if stopped.is_none() {
        // The next chunk follows this one's end directly.
        run.turn.fetch_max(i + 2, Ordering::AcqRel);
    }
    run.notify();
    lap4(&mut t);
}

/// Chunk `i` of a held run (`Run::held`), recorded in `cp`: posted as recorded with what the
/// stretch's checks need, its batches left closed in `cp`'s executor for the stretch's commit
/// (`Cp::held_commit`); cancelled, its batches are dropped and it is not posted; failed, they
/// are dropped and it is posted for the main command processor to record from its start.
#[allow(clippy::too_many_arguments)]
fn post_held(run: &Run, i: usize, mut cp: parking_lot::MutexGuard<'_, Cp>, stopped: Option<u64>, failed: bool, state: Box<ChunkState>, held: super::exec::HeldChunk, worker: usize) {
    let mut t = std::time::Instant::now();
    let cancelled = run.cancelled.load(Ordering::SeqCst);
    if failed || cancelled {
        let keep = cp.closed_count().saturating_sub(held.batches);
        cp.discard_closed_tail(keep);
        if cancelled {
            return;
        }
    }
    let done = if failed {
        Done { stopped, state: *state, last: None, why: "failed", costs: Vec::new(), writes: Default::default(), held: None, worker }
    } else {
        let why = if stopped.is_some() { cp.worker_stop_why() } else { "" };
        let state = cp.chunk_state();
        Done { stopped, state, last: None, why, costs: cp.chunk_costs.take().unwrap_or_default(), writes: Default::default(), held: Some(Box::new(held)), worker }
    };
    drop(cp);
    run.results.lock()[i] = Some(done);
    run.recorded[i].store(true, Ordering::Release);
    run.notify();
    lap4(&mut t);
}

fn lap4(t: &mut std::time::Instant) {
    PHASE_NS[4].fetch_add(t.elapsed().as_nanos() as u64, Ordering::Relaxed);
}

impl Worker {
    /// Runs `f` on its command processor (waiting while it records a chunk): the commit or
    /// drop of a held run's batches its executor keeps (`Cp::held_commit`).
    pub fn with_cp<R>(&self, f: impl FnOnce(&mut Cp) -> R) -> R {
        f(&mut self.cp.lock())
    }

    /// Gives it `run` to take chunks of; it starts once its pool wakes
    /// (`Pool::wake`; `Run::active` counts it from then on).
    pub fn give(&self, run: Arc<Run>) {
        let mut job = self.job.lock();
        job.push_back(run);
        self.has_job.store(true, Ordering::Release);
    }
}

/// Clears the statistics (`profile reset`: the status line and the profile
/// report then cover the window).
pub fn reset_stats() {
    for c in [&RUNS, &CHUNKS, &STOPS, &COMMANDS, &WAIT_NS, &TURN_NS, &CANCELLED, &STARTS, &super::cp::PAR_COSTED] {
        c.store(0, Ordering::Relaxed);
    }
    for c in &PHASE_NS {
        c.store(0, Ordering::Relaxed);
    }
    STOP_WHY.lock().clear();
    for c in &PAST {
        c.store(0, Ordering::Relaxed);
    }
    PAST_STOPS.lock().clear();
    PAST_DROPS.lock().clear();
    for c in IDLE.iter().flatten() {
        c.store(0, Ordering::Relaxed);
    }
    *IDLE_BASE.lock() = ([super::threadcpu::kind_cycles("par"), super::threadcpu::kind_cycles("cpar")], super::FRAMES.load(Ordering::Relaxed));
    *LENGTHS.lock() = [[0; 2]; 6];
    ENDS.lock().clear();
    *LENGTHS_C.lock() = [[0; 2]; 6];
    ENDS_C.lock().clear();
    ENDS_SHORT.lock().clear();
}

/// The workers' idle census line (`DS_IDLE_CENSUS=1`), per flip since the
/// window's start.
fn idle_line() -> String {
    if !super::threadcpu::idle_census() {
        return String::new();
    }
    let (base, f0) = *IDLE_BASE.lock();
    let flips = super::FRAMES.load(Ordering::Relaxed).saturating_sub(f0).max(1) as f64;
    let now = [super::threadcpu::kind_cycles("par"), super::threadcpu::kind_cycles("cpar")];
    let ms = |c: u64| super::threadcpu::ms(c) / flips;
    let v = |k: usize, i: usize| IDLE[k][i].load(Ordering::Relaxed);
    let thread = [ms(now[0].saturating_sub(base[0])), ms(now[1].saturating_sub(base[1]))];
    let runs = [ms(v(0, 2)), ms(v(1, 2))];
    format!(
        "    chunk workers' CPU (graphics, compute queues' workers; ms a flip over {flips:.0} flips): threads {:.1}, {:.1}; in runs {:.1}, {:.1} (of it waiting for chunks' turns {:.1}, {:.1}); outside runs {:.1}, {:.1}; wake-ups {:.0}, {:.0} a flip ({:.0}, {:.0} finding no run)\n",
        thread[0],
        thread[1],
        runs[0],
        runs[1],
        ms(v(0, 3)),
        ms(v(1, 3)),
        thread[0] - runs[0],
        thread[1] - runs[1],
        v(0, 0) as f64 / flips,
        v(1, 0) as f64 / flips,
        v(0, 1) as f64 / flips,
        v(1, 1) as f64 / flips
    )
}

/// One status line ("" before any parallel run).
pub fn status() -> String {
    let runs = RUNS.load(Ordering::Relaxed);
    if runs == 0 && ENDS.lock().is_empty() && ENDS_C.lock().is_empty() {
        return String::new();
    }
    let names = ["<32", "<64", "<192", "<512", "<2048", ">=2048"];
    let lens_of = |l: &Mutex<[[u64; 2]; 6]>| -> Vec<String> { names.iter().zip(l.lock().iter()).map(|(n, x)| format!("{n} {}/{}", x[0], x[1])).collect() };
    let lens = lens_of(&LENGTHS);
    let lens_c = lens_of(&LENGTHS_C);
    let mut ends = ENDS.lock().clone();
    ends.sort_by(|a, b| b.2.cmp(&a.2));
    let mut ends_c = ENDS_C.lock().clone();
    ends_c.sort_by(|a, b| b.2.cmp(&a.2));
    let mut ends_s = ENDS_SHORT.lock().clone();
    ends_s.sort_by(|a, b| b.2.cmp(&a.2));
    let end_name = |w: usize| match w {
        0x130 => "buffer end".to_string(),
        0x131 => "limit".to_string(),
        0x133 => "predicated".to_string(),
        w if w >= 0x200 => format!("event {:#x}", w - 0x200),
        w if w >= 0x100 => format!("private {:#x}", w - 0x100),
        w => format!("{w:#x}"),
    };
    let ends: Vec<String> = ends.iter().take(10).map(|(w, r, c)| format!("{} {r}/{c}", end_name(*w))).collect();
    let ends_c: Vec<String> = ends_c.iter().take(10).map(|(w, r, c)| format!("{} {r}/{c}", end_name(*w))).collect();
    let ends_s: Vec<String> = ends_s.iter().take(10).map(|(w, r, c)| format!("{} {r}/{c}", end_name(*w))).collect();
    let mut why = STOP_WHY.lock().clone();
    why.sort_by(|a, b| b.1.cmp(&a.1));
    let why: Vec<String> = why.iter().map(|(w, n)| format!("{w} {n}")).collect();
    let past = if PAST[0].load(Ordering::Relaxed) > 0 {
        let list = |l: &Mutex<Vec<(&'static str, u64)>>| {
            let mut v = l.lock().clone();
            v.sort_by(|a, b| b.1.cmp(&a.1));
            v.iter().map(|(w, n)| format!("{w} {n}")).collect::<Vec<_>>().join(", ")
        };
        let p = |i: usize| PAST[i].load(Ordering::Relaxed);
        format!(
            "    recording past runs: {} runs gone past, {} committed, {} dropped ({}), {} commands recorded past them ({:.1} each), {:.0} us from going past to the stop, then {:.0} us waited for the chunks; stops: {}\n",
            p(0),
            p(1),
            p(2),
            list(&PAST_DROPS),
            p(3),
            p(3) as f64 / p(0).max(1) as f64,
            p(4) as f64 / 1e3 / p(0).max(1) as f64,
            p(5) as f64 / 1e3 / p(0).max(1) as f64,
            list(&PAST_STOPS)
        )
    } else {
        String::new()
    };
    let whole = format!("    whole-run recording: enabled {}, runs {}, commands {} (first chunk handed to workers; ordered primary batches)\n",
        whole_run(), WHOLE_RUNS.load(Ordering::Relaxed), WHOLE_COMMANDS.load(Ordering::Relaxed));
    past + &whole + &format!(
        "    parallel runs: {runs} runs ({} split at equal recording cost), {} chunks to workers ({} commands; {} per recorder), {} stopped early ({}), {} cancelled; main waited {} ms for workers (its chunk {:.2}x a worker's), workers {} ms for their turns ({} batch slots added), per chunk: start latency {:.0} us (first chunks), setup {:.1} us, recording {:.0} us ({:.0} commands), closing {:.0} us, submitting and posting {:.1} us; runs seen by length (runs/commands) {}; by end {}; runs under 32 commands by end {}; compute queues' runs by length {}; by end {}{}\n",
        super::cp::PAR_COSTED.load(Ordering::Relaxed),
        CHUNKS.load(Ordering::Relaxed),
        COMMANDS.load(Ordering::Relaxed),
        split(),
        STOPS.load(Ordering::Relaxed),
        why.join(", "),
        CANCELLED.load(Ordering::Relaxed),
        WAIT_NS.load(Ordering::Relaxed) / 1_000_000,
        main_weight(),
        TURN_NS.load(Ordering::Relaxed) / 1_000_000,
        super::exec::WORKER_SLOTS.load(Ordering::Relaxed),
        PHASE_NS[0].load(Ordering::Relaxed) as f64 / STARTS.load(Ordering::Relaxed).max(1) as f64 / 1e3,
        PHASE_NS[1].load(Ordering::Relaxed) as f64 / CHUNKS.load(Ordering::Relaxed).max(1) as f64 / 1e3,
        PHASE_NS[2].load(Ordering::Relaxed) as f64 / CHUNKS.load(Ordering::Relaxed).max(1) as f64 / 1e3,
        COMMANDS.load(Ordering::Relaxed) as f64 / CHUNKS.load(Ordering::Relaxed).max(1) as f64,
        PHASE_NS[3].load(Ordering::Relaxed) as f64 / CHUNKS.load(Ordering::Relaxed).max(1) as f64 / 1e3,
        PHASE_NS[4].load(Ordering::Relaxed) as f64 / CHUNKS.load(Ordering::Relaxed).max(1) as f64 / 1e3,
        lens.join(", "),
        ends.join(", "),
        ends_s.join(", "),
        lens_c.join(", "),
        ends_c.join(", "),
        super::cp::run_passed_line()
    ) + &super::hscensus::line()
        + &idle_line()
        + &{
            let (lo, hi) = (MIN_LOW.load(Ordering::Relaxed), MIN_HIGH.load(Ordering::Relaxed));
            if lo + hi == 0 {
                String::new()
            } else {
                format!(
                    "    graphics split threshold (automatic, `graphics_min`): 32 for {lo} plans while the GPU waited, the fixed one for {hi}, {} switches; the GPU's busy share of the frames now {:.2}
",
                    MIN_SWITCHES.load(Ordering::Relaxed),
                    super::hitch::gpu_busy_share()
                )
            }
        }
}
