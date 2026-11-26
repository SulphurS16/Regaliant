//! gfx10 PM4 command processor.
//!
//! Executes command buffers submitted through libSceAgcDriver on a GPU queue
//! thread: register state (context, SH, uconfig), control flow (indirect
//! buffers, COND_EXEC, REWIND, the Agc branch), synchronisation (WAIT_REG_MEM,
//! RELEASE_MEM labels and end-of-pipe interrupts), memory packets (WRITE_DATA,
//! COPY_DATA, DMA_DATA), flips and the private Agc packets
//! (`hle::libs::agc_cmd`).
//!
//! With `DS_GPU=1`, dispatches run on Vulkan (`exec`); the recorded GPU work
//! is completed before any packet that touches memory or synchronizes, so
//! labels and CPU/GPU synchronisation stay ordered as on hardware. Draws are
//! decoded and counted but not rendered yet.

use crate::hle::libs::agc_cmd::{DS_MAGIC, op, pop};
use std::collections::{HashMap, HashSet};
use std::sync::Arc;
use std::sync::atomic::{AtomicU64, Ordering};
use std::time::{Duration, Instant};

pub const CTX_BASE: u32 = 0x28000;
pub const SH_BASE: u32 = 0xB000;
pub const UC_BASE: u32 = 0x30000;
const CTX_DWORDS: usize = 0x2000;
const SH_DWORDS: usize = 0x400;
const UC_DWORDS: usize = 0x4000;

pub static DRAWS: AtomicU64 = AtomicU64::new(0);
pub static DISPATCHES: AtomicU64 = AtomicU64::new(0);
pub static PACKETS: AtomicU64 = AtomicU64::new(0);
pub static EOP_INTERRUPTS: AtomicU64 = AtomicU64::new(0);
/// GPU completions waited for, by the PM4 opcode that required them.
pub static SYNC_BY_OPCODE: [AtomicU64; 256] = [const { AtomicU64::new(0) }; 256];
/// CP effects deferred until earlier GPU work completed, by opcode and why:
/// [end of pipe, after a sync event, behind an earlier deferred effect].
pub static DEFERRED_BY_OPCODE: [[AtomicU64; 3]; 256] = [const { [const { AtomicU64::new(0) }; 3] }; 256];
/// RELEASE_MEM packets by DATA_SEL and INT_SEL (8 x 8).
pub static RELEASE_KINDS: [AtomicU64; 64] = [const { AtomicU64::new(0) }; 64];
/// Waits passed on a value the waiting queue's own recording batch writes
/// on the GPU (`tune gpu_cp_writes`).
pub static OPEN_PROMISED: AtomicU64 = AtomicU64::new(0);

/// Most frequent synchronizing opcodes (diagnostics).
pub fn sync_summary() -> String {
    let mut v: Vec<(u64, usize)> = SYNC_BY_OPCODE.iter().enumerate().map(|(i, c)| (c.load(Ordering::Relaxed), i)).filter(|x| x.0 > 0).collect();
    v.sort_by(|a, b| b.cmp(a));
    let waits = v.iter().take(6).map(|(n, op)| format!("{op:#x}: {n}")).collect::<Vec<_>>().join(", ");
    let mut d: Vec<(u64, usize, [u64; 3])> = DEFERRED_BY_OPCODE
        .iter()
        .enumerate()
        .map(|(i, c)| {
            let x = [c[0].load(Ordering::Relaxed), c[1].load(Ordering::Relaxed), c[2].load(Ordering::Relaxed)];
            (x.iter().sum(), i, x)
        })
        .filter(|x| x.0 > 0)
        .collect();
    d.sort_by(|a, b| b.cmp(a));
    let deferred = d.iter().take(6).map(|(_, op, x)| format!("{op:#x}: {} end of pipe / {} after sync / {} behind deferred", x[0], x[1], x[2])).collect::<Vec<_>>().join(", ");
    let kinds: Vec<String> = RELEASE_KINDS.iter().enumerate().filter(|(_, c)| c.load(Ordering::Relaxed) > 0).map(|(i, c)| format!("data {} int {}: {}", i / 8, i % 8, c.load(Ordering::Relaxed))).collect();
    format!(
        "{waits}; deferred CP effects {deferred}; RELEASE_MEM by kind {}; waits passed on the own batch's GPU writes {}; label writes recorded while another queue waited (submitted at once) {}",
        kinds.join(", "),
        OPEN_PROMISED.load(Ordering::Relaxed),
        super::exec::LABEL_EARLY.load(Ordering::Relaxed)
    )
}

/// Register file of one queue.
#[derive(Clone)]
pub struct Regs {
    pub ctx: Vec<u32>,
    pub sh: Vec<u32>,
    pub uc: Vec<u32>,
}

impl Regs {
    fn new() -> Regs {
        Regs { ctx: vec![0; CTX_DWORDS], sh: vec![0; SH_DWORDS], uc: vec![0; UC_DWORDS] }
    }
}

/// The highest register index written so far plus one, per space (context,
/// SH, UCONFIG), over all command processors and chunk states: every register
/// at or above it is zero everywhere, so state snapshots (`ChunkState`) hold
/// only what lies below (the UCONFIG file alone is 64 KiB; a parallel run
/// copied ~20 whole register files).
static HIGH: [std::sync::atomic::AtomicUsize; 3] = [const { std::sync::atomic::AtomicUsize::new(0) }; 3];

fn note_high(space: Space, index: usize) {
    let h = &HIGH[space as usize];
    if index >= h.load(Ordering::Relaxed) {
        h.fetch_max(index + 1, Ordering::AcqRel);
    }
}

fn high(space: Space, cap: usize) -> usize {
    HIGH[space as usize].load(Ordering::Acquire).min(cap)
}

/// Copies a snapshot's registers into `dst` and zeroes the rest up to `hi`
/// (registers the snapshot does not hold were zero when it was taken).
fn copy_prefix(dst: &mut [u32], src: &[u32], hi: usize) {
    let n = src.len().min(dst.len());
    dst[..n].copy_from_slice(&src[..n]);
    let end = hi.min(dst.len());
    if end > n {
        dst[n..end].fill(0);
    }
}

/// `tune plan_check`: logs the first registers where a chunk start state made
/// with `put_regs` differs from the one made a register at a time.
fn plan_check_states(a: &ChunkState, b: &ChunkState, at: u64) {
    static CHECKED: AtomicU64 = AtomicU64::new(0);
    static BAD: AtomicU64 = AtomicU64::new(0);
    let n = CHECKED.fetch_add(1, Ordering::Relaxed) + 1;
    let mut diff = Vec::new();
    for (name, x, y) in [("ctx", &a.regs.ctx, &b.regs.ctx), ("sh", &a.regs.sh, &b.regs.sh), ("uc", &a.regs.uc, &b.regs.uc)] {
        for i in 0..x.len().max(y.len()) {
            let (u, v) = (x.get(i).copied().unwrap_or(0), y.get(i).copied().unwrap_or(0));
            if u != v && diff.len() < 6 {
                diff.push(format!("{name}[{i:#x}] {u:#x} against {v:#x}"));
            }
        }
    }
    let other = (a.index_base, a.index_count, a.num_instances, a.draw_indirect_base, a.dispatch_indirect_base) != (b.index_base, b.index_count, b.num_instances, b.draw_indirect_base, b.dispatch_indirect_base);
    if !diff.is_empty() || other {
        if BAD.fetch_add(1, Ordering::Relaxed) < 20 {
            crate::warn!("gpu: plan_check: chunk start at {at:#x}: {} (other fields differ: {other})", diff.join(", "));
        }
    } else if n % 10000 == 0 {
        crate::info!("gpu: plan_check: {n} chunk start states checked, {} differed", BAD.load(Ordering::Relaxed));
    }
}

/// Writes `count` registers of a snapshot from `first` (growing it) with the
/// dwords at `a` (`put_reg` for each, in one copy).
fn put_regs(v: &mut Vec<u32>, space: Space, cap: usize, first: usize, f: &Fetched, a: u64, count: usize) {
    if count == 0 || first >= cap {
        return;
    }
    let end = (first + count).min(cap);
    if end > v.len() {
        v.resize(end, 0);
    }
    let dst = &mut v[first..end];
    match f.slice(a, dst.len()) {
        Some(src) => dst.copy_from_slice(src),
        None => {
            for (j, d) in dst.iter_mut().enumerate() {
                *d = f.rd(a + 4 * j as u64);
            }
        }
    }
    note_high(space, end - 1);
}

/// Writes register `i` of a snapshot (growing it), notes the index.
fn put_reg(v: &mut Vec<u32>, space: Space, cap: usize, i: usize, val: u32) {
    if i >= cap {
        return;
    }
    if i >= v.len() {
        v.resize(i + 1, 0);
    }
    v[i] = val;
    note_high(space, i);
}

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum Space {
    Ctx,
    Sh,
    Uc,
}

/// Draws, those with the last draw's context registers on the same command
/// processor, and of those the ones with its programs and primitive type too
/// (`Cp::draw`; `draw_state_line`).
pub static DRAW_STATE: [AtomicU64; 3] = [const { AtomicU64::new(0) }; 3];

/// Runs of draws with the last draw's context registers, programs and
/// primitive type, by length (1, 2, 3-4, 5-8, 9-16, 17-32, more): [runs,
/// draws].
static DRAW_RUNS: [[AtomicU64; 2]; 7] = [const { [const { AtomicU64::new(0) }; 2] }; 7];

fn note_draw_run(n: u32) {
    if n == 0 {
        return;
    }
    let k = match n {
        1 => 0,
        2 => 1,
        3..=4 => 2,
        5..=8 => 3,
        9..=16 => 4,
        17..=32 => 5,
        _ => 6,
    };
    DRAW_RUNS[k][0].fetch_add(1, Ordering::Relaxed);
    DRAW_RUNS[k][1].fetch_add(n as u64, Ordering::Relaxed);
}

/// `profile reset`: `DRAW_STATE` and `DRAW_RUNS`.
pub fn reset_draw_state() {
    for c in DRAW_STATE.iter().chain(DRAW_RUNS.iter().flatten()) {
        c.store(0, Ordering::Relaxed);
    }
}

/// For the status report (`DRAW_STATE`, `DRAW_RUNS`).
pub fn draw_state_line() -> String {
    let d: Vec<u64> = DRAW_STATE.iter().map(|c| c.load(Ordering::Relaxed)).collect();
    if d[0] == 0 {
        return String::new();
    }
    let runs: Vec<String> = ["1", "2", "3-4", "5-8", "9-16", "17-32", ">32"]
        .iter()
        .zip(DRAW_RUNS.iter())
        .map(|(l, r)| format!("{l}: {} runs, {} draws", r[0].load(Ordering::Relaxed), r[1].load(Ordering::Relaxed)))
        .collect();
    format!(
        "    draws with the last draw's context registers: {} of {} ({:.1}%), {} with its programs and primitive type too ({:.1}%); runs of such draws by length: {}\n",
        d[1],
        d[0],
        100.0 * d[1] as f64 / d[0] as f64,
        d[2],
        100.0 * d[2] as f64 / d[0] as f64,
        runs.join(", ")
    )
}

/// Chains of repeatable draws ended, by the opcode of the packet that ended
/// them (`keeps_repeat`; the status line's top ones).
pub static REPEAT_ENDS: [AtomicU64; 256] = [const { AtomicU64::new(0) }; 256];
/// The last value given out as a `Cp::ps_epoch` (unique over all command processors).
static PS_EPOCH: AtomicU64 = AtomicU64::new(0);

/// The opcodes that ended most chains of repeatable draws (`REPEAT_ENDS`).
pub fn repeat_ends_line() -> String {
    let mut v: Vec<(u64, usize)> = REPEAT_ENDS.iter().enumerate().map(|(k, c)| (c.load(Ordering::Relaxed), k)).filter(|x| x.0 > 0).collect();
    v.sort_unstable_by(|a, b| b.cmp(a));
    v.iter().take(6).map(|(c, k)| format!("{k:#x} {c}")).collect::<Vec<_>>().join(", ")
}

pub struct Cp {
    pub name: String,
    /// Outcome of the last draw (frame captures; no allocation per draw).
    pub last_draw: std::borrow::Cow<'static, str>,
    /// `DS_ARGS_CHECK`: the address and words the indirect draw being executed read.
    args_checked: Option<(u64, [u32; 5])>,
    /// The last packet was an indexed indirect draw with GPU-read arguments
    /// that the executor can repeat (`tune draw_repeat`): no other packet
    /// (but NOPs) since, so the registers are the same.
    repeat_ok: bool,
    /// A context register changed value since the last draw (measurement:
    /// `DRAW_STATE`), and the last draw's vertex and pixel programs and
    /// primitive type.
    ctx_dirty: bool,
    /// A register of the pixel program (its address, RSRC1/2 or user data: shader registers
    /// 0x08-0x2B) changed value since the last draw.
    ps_dirty: bool,
    /// Names the context registers and the pixel program's registers of the last draw: a
    /// new value from `PS_EPOCH` at each draw that finds one of them changed
    /// (`Exec::ps_epoch`: a pixel stage that reads no memory is then the same stage).
    ps_epoch: u64,
    last_draw_progs: (u64, u64, u32),
    /// Draws in the current run of draws with the last draw's context
    /// registers, programs and primitive type (`note_draw_run`).
    same_run: u32,
    pub compute: bool,
    pub regs: Regs,
    index_base: u64,
    index_count: u32,
    num_instances: u32,
    draw_indirect_base: u64,
    dispatch_indirect_base: u64,
    markers: Vec<String>,
    ctx_stack: Vec<Vec<u32>>,
    /// Address of the packet being executed / address being polled.
    cur_packet: u64,
    /// Opcode of the packet being executed (private packets: 0x100 + op).
    cur_opcode: u32,
    waiting_on: u64,
    /// Set while the CP polls memory (WAIT_REG_MEM): submitters must not
    /// wait for this queue then (the label may come from them).
    pub wait_flag: Option<&'static std::sync::atomic::AtomicBool>,
    unknown_seen: HashSet<u32>,
    notes_seen: HashSet<&'static str>,
    depth: u32,
    /// Vulkan executor (created on the first dispatch when enabled).
    /// Boxed: commands take it out of the processor and put it back (one
    /// pointer moved per draw and dispatch, not the whole executor).
    exec: Option<Box<super::exec::Exec>>,
    exec_failed: bool,
    /// Front end 2's view of this queue (`fe2::views_on`; the graphics
    /// queue's and its workers' with `fe2::gfx_views`): advanced at its
    /// synchronization points (`fe2_sync`), a worker's set from its run.
    fe2_view: Option<Arc<std::sync::atomic::AtomicU32>>,
    /// A partial flush / acquire was requested: the next CP memory write
    /// must wait for the GPU work before it.
    fence_pending: bool,
    /// The last such request: its EVENT_WRITE event type, 64 ACQUIRE_MEM
    /// (diagnosis: `exec::CPW_FENCES`).
    fence_kind: u8,
    /// SET_PREDICATION state for packets with the predicate bit.
    pred: Option<Pred>,
    /// When the title submitted the command buffer being executed.
    pub submitted_at: Option<Instant>,
    /// Another command buffer follows this one at once (the title's
    /// submission holds more, or its next submission is queued): its end
    /// need not submit the batch (`tune cmdbuf_merge`).
    pub more_follows: Option<Box<dyn Fn() -> bool + Send>>,
    pub next_in_submission: bool,
    /// The queue's number of the title submission being run (set by the
    /// queue's worker), and where to run from again when what was recorded
    /// past a label wait in an earlier submission is dropped (`take_rewind_sub`:
    /// submission, buffer, address; the worker keeps the submissions such a
    /// stretch runs through).
    pub sub_index: u64,
    rewind_sub: Option<(u64, usize, u64)>,
    /// Stage preparations made ahead for the commands after a label wait
    /// (`scan_through`), in packet order.
    ahead: std::collections::VecDeque<AheadCmd>,
    /// The executing command's preparations made ahead (taken by the
    /// executor through `take_ahead`; [vertex, pixel] or [compute, -]).
    cur_ahead: std::cell::RefCell<[Option<Arc<super::ahead::Slot>>; 2]>,
    /// The scan's copies of the SH and context register files (kept to
    /// avoid allocations).
    scan_regs: std::cell::RefCell<(Vec<u32>, Vec<u32>)>,
    /// The fetched buffer's base, the packet after the dispatch about to
    /// execute and the buffer's end (`tune compile_ahead`): scanned when
    /// the dispatch's pipeline is missing (`prefetch_compiles`).
    compile_ctx: std::cell::Cell<Option<(u64, u64, u64)>>,
    /// The buffer, the packet after it and the buffer's end of the label
    /// wait about to execute (`tune wait_ahead`): scanned if it blocks.
    wait_ctx: Option<(Fetched, u64, u64)>,
    /// A parallel run in progress (`gpu::par`).
    par: Option<ParRun>,
    /// The last WAIT_REG_MEM passed on a GPU queue's write (a promise, an
    /// open promise of this queue, a latched CP write): no CPU write
    /// released it, so it ends no copy-cache epoch (`tune epoch_gpu_waits`).
    last_wait_gpu: bool,
    /// How the last WAIT_REG_MEM passed (its `label_event` kind; diagnostics).
    last_wait_kind: u8,
    /// Its comparison function (diagnostics: `EPOCH_REPEATS`).
    last_wait_func: u32,
    /// Front end 2: per label this queue's waits
    /// passed on as released by the CPU, the value then, the distinct
    /// values seen, and whether each was greater than the one before.
    passed: HashMap<u64, (u64, u32, bool)>,
    /// The view of the writer of the promise the last WAIT_REG_MEM passed on
    /// (0: this queue's own open promise; front end 2).
    last_wait_epoch: u32,
    /// The end of the last run the planner found too short (no new plan
    /// until execution passes it: each command would scan the rest again).
    par_skip: (u64, u64),
    /// Its commands (`tune cp_classes`: serial commands by run length).
    par_skip_n: usize,
    /// The next run planned before the stretch past a run settled (`plan_ahead`).
    pre_plan: Option<Box<PrePlan>>,
    /// The serial run being watched (`RUN_EPOCHS`): its end, its epoch when
    /// planned, when.
    run_watch: Option<(u64, u32, Instant)>,
    /// This command processor's chunk workers.
    par_pool: super::par::Pool,
    /// A worker's per-command recording cost in its chunk (`tune par_cost`).
    pub chunk_costs: Option<Vec<u32>>,
    /// A parallel worker: the label waits its run holds (`par::Run::
    /// passed_waits`).
    pub chunk_waits: std::sync::Arc<[u64]>,
    /// The stretch `handshake_census` scanned last: its buffer's end and
    /// its end.
    hs_until: (u64, u64),
    /// Per-command recording costs of the parallel runs seen, by run
    /// signature (`tune par_cost`).
    run_costs: HashMap<u64, RunCost>,
    /// The frame `run_costs` was last pruned in (at most once a frame: a
    /// vista's runs keep it above its bound).
    run_costs_pruned: u64,
    /// Its state while it records past a run or a label wait into held
    /// batches (`past_start`; None: recording as usual).
    pub(super) held: Option<Box<HeldCp>>,
    /// The graphics queue's command processor (`gpu-gfx`).
    main_gfx: bool,
    /// Recording past a parallel run (`tune past_runs`).
    past: Option<Box<Past>>,
    /// Recording past a label wait (`Cp::past_wait_ok`), and whether such a
    /// stretch is settling (its wait is waited for normally then).
    past_wait: Option<Box<PastWait>>,
    settling: bool,
    /// Label waits the stretch past a label wait passed on memory (`held_wait`, a held run's
    /// planning): label, 64-bit, compare function, mask, reference. A write of one by a batch
    /// submitted meanwhile drops the stretch only when the wait no longer passes at the
    /// settle (`past_wait_settle`; since 2026-11-17: 41-79 stretches of ~650 a 13 s turning
    /// window at Castlewall were dropped for such a read, each followed by waits not gone
    /// past, FINDINGS.md).
    held_polls: Vec<(u64, bool, u32, u64, u64)>,
    /// The index of the submission's command buffer being run (set by the
    /// queue's worker), and where the worker is to run from next when what
    /// was recorded past a run is dropped in a later buffer than the run's
    /// (`take_rewind`: buffer, address).
    pub cur_buf: usize,
    rewind: Option<(usize, u64)>,
}

/// The recording cost of a parallel run's commands as its workers measured
/// it (TSC cycles, smoothed over the times the run was seen; NaN: not
/// measured yet), and the frame it was last planned in.
struct RunCost {
    costs: Vec<f32>,
    seen: u64,
}

/// Splits commands [main_n, n) into `m` chunks of about equal summed cost
/// (`costs` per command), each of at least `min` commands: the first command
/// of each chunk. None without costs for all of them.
fn cost_firsts(costs: &[f32], main_n: usize, n: usize, m: usize, min: usize) -> Option<Vec<usize>> {
    let region = costs.get(main_n..n)?;
    if m < 2 || region.iter().any(|c| !c.is_finite()) || n - main_n < m * min {
        return None;
    }
    let total: f64 = region.iter().map(|&c| c as f64).sum();
    if total <= 0.0 {
        return None;
    }
    let mut firsts = Vec::with_capacity(m);
    firsts.push(main_n);
    let mut acc = 0.0;
    for (j, &c) in region.iter().enumerate() {
        let k = firsts.len();
        if k == m {
            break;
        }
        acc += c as f64;
        if acc >= total * k as f64 / m as f64 {
            let lo = firsts[k - 1] + min;
            let hi = n - (m - k) * min;
            firsts.push((main_n + j + 1).clamp(lo, hi));
        }
    }
    while firsts.len() < m {
        let k = firsts.len();
        firsts.push((firsts[k - 1] + min).min(n - (m - k) * min));
    }
    firsts.windows(2).all(|w| w[1] >= w[0] + min).then_some(firsts)
}

/// The context registers' clear-state values: zero, except the registers
/// with a non-zero Agc register default (see `Cp::clear_context`).
fn context_defaults() -> &'static [(u32, u32)] {
    static DEFAULTS: std::sync::OnceLock<Vec<(u32, u32)>> = std::sync::OnceLock::new();
    DEFAULTS.get_or_init(|| {
        let mut v = Vec::new();
        for def in crate::hle::libs::agc_defaults::PUBLIC.iter().chain(crate::hle::libs::agc_defaults::INTERNAL) {
            if matches!(def.table(), 0 | 3) {
                v.extend(def.regs.iter().filter(|r| r.1 != 0).copied());
            }
        }
        v
    })
}

/// `clear_context` on a walked state.
fn walk_clear(st: &mut ChunkState) {
    st.regs.ctx.fill(0);
    for &(i, val) in context_defaults() {
        put_reg(&mut st.regs.ctx, Space::Ctx, CTX_DWORDS, i as usize, val);
    }
}

/// A command processor's state as `Cp::walk_state` follows it.
#[derive(Clone)]
pub struct WalkState {
    pub st: ChunkState,
    /// The context-state stack (`pop::CONTEXT_STATE`).
    pub stack: Vec<Vec<u32>>,
    /// SH registers indirect draws set from their arguments.
    pub unknown: Vec<u32>,
    /// Memory writes of the walked packets whose values are known before
    /// they execute (WRITE_DATA, RELEASE_MEM): (address, value, bytes; 0
    /// bytes: a timestamp, value unknown), in order.
    pub labels: Vec<(u64, u64, u8)>,
    /// The FLIP packet the walk stopped before: (address, bytes).
    pub flip: Option<(u64, u64)>,
}

impl WalkState {
    /// Registers that differ from `other`'s: (space 0 ctx / 1 sh / 2 uc,
    /// index), SH registers in `unknown` left out; and whether the CP's own
    /// fields and stack depth agree.
    pub fn diff(&self, other: &WalkState) -> (Vec<(usize, u32)>, bool) {
        let mut out = Vec::new();
        let spaces = [(&self.st.regs.ctx, &other.st.regs.ctx), (&self.st.regs.sh, &other.st.regs.sh), (&self.st.regs.uc, &other.st.regs.uc)];
        for (k, (a, b)) in spaces.iter().enumerate() {
            let n = a.len().max(b.len());
            for i in 0..n {
                let (x, y) = (a.get(i).copied().unwrap_or(0), b.get(i).copied().unwrap_or(0));
                if x != y && !(k == 1 && (self.unknown.contains(&(i as u32)) || other.unknown.contains(&(i as u32)))) {
                    out.push((k, i as u32));
                }
            }
        }
        let fields = self.st.index_base == other.st.index_base
            && self.st.index_count == other.st.index_count
            && self.st.num_instances == other.st.num_instances
            && self.st.draw_indirect_base == other.st.draw_indirect_base
            && self.st.dispatch_indirect_base == other.st.dispatch_indirect_base
            && self.stack.len() == other.stack.len();
        (out, fields)
    }
}

/// The command-processor state a chunk of a parallel run starts from and
/// ends with (`gpu::par`).
#[derive(Clone)]
pub struct ChunkState {
    regs: Regs,
    index_base: u64,
    index_count: u32,
    num_instances: u32,
    draw_indirect_base: u64,
    dispatch_indirect_base: u64,
}

/// A parallel run in progress: chunk i + 1 starts at `starts[i]` and is
/// chunk i of `run` (the workers'); it ends where the next starts (the last
/// at `end`). `next`: the chunk to collect next.
struct ParRun {
    starts: Vec<u64>,
    end: u64,
    next: usize,
    run: Arc<super::par::Run>,
    /// How long the main command processor waited for the workers so far.
    waited_ns: u64,
    /// The run holds flushes or cache actions (`par_event`): later CP
    /// writes follow the GPU work before them.
    sync: bool,
    /// Its signature and command count, and each worker chunk's first
    /// command (`tune par_cost`; signature 0: not timed).
    sig: u64,
    n: usize,
    firsts: Vec<usize>,
    /// What its chunks' batches write (`DS_RUN_PROBE` only).
    writes: super::ranges::Ranges,
    /// The state at its end, walked when it was planned (`tune
    /// past_runs`), and the SH registers its last chunk's indirect draws set
    /// from their arguments (a draw that hands its start instance to Vulkan,
    /// `tune cpu_args_base`, leaves them as they were: not compared).
    end_state: Option<ChunkState>,
    end_unknown: Vec<u32>,
    /// The command buffer and title submission it lies in.
    buf: usize,
    sub: u64,
    /// Planned inside a stretch past a label wait (`par::Run::held`, `held_collect`).
    held: bool,
    /// Its stretch settled before all its chunks were collected (`past_wait_settle`).
    unheld: Option<Box<Unheld>>,
}

/// A held run whose stretch settled before all its chunks were collected (`Cp::
/// past_wait_settle`): its remaining chunks go to the GPU as they are collected, each checked as
/// the stretch was (`Cp::unheld_collect`): its CPU reads against what other queues wrote during
/// the stretch (`writes`), its images' pending clears and reloads, its placed ranges at the
/// label's epoch (`need`; pages the stretch uploaded, `up`, count as current).
struct Unheld {
    writes: super::ranges::Ranges,
    need: Option<u32>,
    up: HashSet<u64>,
}

/// What the planner's scan of a run found (`Cp::plan_scan`): its commands' ends, the
/// indirect arguments read in it, the label waits it holds (packet, label, 64-bit), those a
/// write earlier in the run meets, its CP writes (packet, address, bytes), whether it holds
/// flushes or cache actions, its signature (packet headers), why it ends and where.
struct PlanScan {
    ends: Vec<u64>,
    args: Vec<(u64, u64)>,
    passed: Vec<(u64, u64, bool)>,
    own_waits: Vec<u64>,
    cp_writes: Vec<(u64, u64, u64)>,
    sync: bool,
    sig: u64,
    why: usize,
    run_end: u64,
}

/// A split run's chunks (`Cp::plan_chunks`): the main command processor's command count,
/// each worker chunk's first command and start, their start states, the state at the run's
/// end (`tune past_runs`) with the SH registers its indirect draws set, and the signature
/// with the count (`tune par_cost`).
struct PlanChunks {
    main_n: usize,
    firsts: Vec<usize>,
    starts: Vec<u64>,
    states: Vec<Box<ChunkState>>,
    end_state: Option<ChunkState>,
    end_unknown: Vec<u32>,
    sig: u64,
}

/// The next parallel run planned before the run recorded past is collected (`Cp::plan_ahead`,
/// since 2026-11-18): made at the stop "next run" while that run's chunks may still record,
/// taken by `par_plan` when it plans the same run in the same command buffer and submission
/// from the same state.
struct PrePlan {
    from: u64,
    end: u64,
    buf: usize,
    sub: u64,
    start: ChunkState,
    scan: PlanScan,
    chunks: PlanChunks,
}

/// Runs planned ahead (`Cp::plan_ahead`): [made, taken, not taken (another run, a dropped
/// stretch, another state), ns making them].
static PLAN_AHEAD: [AtomicU64; 4] = [const { AtomicU64::new(0) }; 4];

/// The graphics command processor recording past a parallel run before
/// collecting its chunks (`Cp::past_start`): the run, and what is put back
/// if what it recorded past the run is dropped.
struct Past {
    pr: ParRun,
    end_state: ChunkState,
    end_unknown: Vec<u32>,
    /// Whether it may go on into the submission's later command buffers.
    across: bool,
    stack: Vec<Vec<u32>>,
    markers: Vec<String>,
    pred: Option<Pred>,
    fence_pending: bool,
    began: Instant,
}

/// The graphics command processor recording past a wait on a label a
/// compute queue wrote last (`Cp::past_wait_start`): the wait, and what is
/// put back if what it recorded past it is dropped.
struct PastWait {
    control: u32,
    a: u64,
    wide: bool,
    reference: u64,
    mask: u64,
    /// The wait's packet, the command buffer holding it and its
    /// submission.
    at: u64,
    buf: usize,
    sub: u64,
    state: ChunkState,
    stack: Vec<Vec<u32>>,
    markers: Vec<String>,
    pred: Option<Pred>,
    fence_pending: bool,
    began: Instant,
    /// Packets since the label was last looked at.
    ticks: u32,
    /// Held runs (`tune held_runs`): the stretch's share of front end 2's uploads (`par::
    /// new_run_id`; 0: none) and the part the command processor records now (`Exec::
    /// fe2_share`); the commit's segments in program order; per worker (`par::Pool::get`), the
    /// batches of collected chunks its executor holds; held runs planned.
    share: u64,
    part: u32,
    segs: Vec<HeldSeg>,
    counts: Vec<usize>,
    runs: u32,
    /// The share is in use (from the stretch's first held run: `Cp::par_plan`).
    sharing: bool,
}

/// A segment of a stretch's held work, in program order (`Cp::held_commit`): the command
/// processor's closed batches up to this count, or a held run's chunk (its worker and its
/// batches, the first ones that worker's executor holds after earlier segments').
enum HeldSeg {
    Cp(usize),
    Chunk(usize, usize),
}

/// Held runs (`tune held_runs`): [runs planned, worker chunks, chunks stopped, chunk batches
/// submitted at commits, ns waited for chunks, runs cancelled, stretches with held runs
/// committed, dropped, commands planned for the workers, batches left in workers' executors at
/// a commit (none expected), runs not planned (fast clears pending, indirect arguments its held
/// work writes), plannings that stopped the stretch, the stretch's own fast clears applied at a
/// held run's start, chunks collected after their stretch settled, of them dropped, stretches
/// without held runs by the backoff].
pub static HELD_RUNS: [AtomicU64; 16] = [const { AtomicU64::new(0) }; 16];

/// Recording past label waits (`Cp::past_wait_ok`): [stretches begun,
/// committed, dropped, commands recorded past the waits, ns from a wait to
/// its settling, settled once the label was promised, settled at a stop,
/// ns the stops waited for the label].
pub static PAST_WAITS: [AtomicU64; 8] = [const { AtomicU64::new(0) }; 8];
/// Label polls of stretches past label waits kept at their settle (`Cp::past_wait_settle`):
/// [polls kept, stretches they would have dropped].
pub static PAST_POLLS: [AtomicU64; 2] = [const { AtomicU64::new(0) }; 2];
static PAST_WAIT_STOPS: parking_lot::Mutex<Vec<(&'static str, u64)>> = parking_lot::Mutex::new(Vec::new());
static PAST_WAIT_DROPS: parking_lot::Mutex<Vec<(&'static str, u64)>> = parking_lot::Mutex::new(Vec::new());
/// The graphics CP's waits on labels queue 0x50 wrote last that did not pass at once and were
/// not gone past, by the first condition of `past_wait_ok` that failed (`past_wait_not`).
static PAST_WAIT_NOT: parking_lot::Mutex<Vec<(&'static str, u64)>> = parking_lot::Mutex::new(Vec::new());
/// Held runs' chunks collected after their stretch settled and dropped, by reason (`Cp::unheld_collect`).
static HELD_DROPS: parking_lot::Mutex<Vec<(&'static str, u64)>> = parking_lot::Mutex::new(Vec::new());

fn note_why(l: &parking_lot::Mutex<Vec<(&'static str, u64)>>, why: &'static str) {
    let mut g = l.lock();
    match g.iter_mut().find(|x| x.0 == why) {
        Some(x) => x.1 += 1,
        None => g.push((why, 1)),
    }
}

/// Clears the statistics of recording past label waits (`profile reset`).
pub fn past_wait_reset() {
    for c in PAST_WAITS.iter().chain(PAST_POLLS.iter()) {
        c.store(0, Ordering::Relaxed);
    }
    PAST_SKIPPED.store(0, Ordering::Relaxed);
    PAST_WAIT_STOPS.lock().clear();
    PAST_WAIT_DROPS.lock().clear();
    PAST_WAIT_NOT.lock().clear();
    HELD_DROPS.lock().clear();
    for c in HELD_RUNS.iter().chain(super::exec::HELD_RUN_STOPS.iter()).chain(super::exec::PAST_REPAIRS.iter()) {
        c.store(0, Ordering::Relaxed);
    }
}

/// Per label address gone past: (its stretches' recent drop rate in 1/1024,
/// waits not gone past since the last probe) (`tune past_backoff`).
static PAST_BACKOFF: parking_lot::Mutex<Option<HashMap<u64, (u32, u32)>>> = parking_lot::Mutex::new(None);
/// Waits not gone past because their label's stretches were dropped.
static PAST_SKIPPED: AtomicU64 = AtomicU64::new(0);

/// Whether a wait on label `a` may be gone past: while most of its recent
/// stretches were dropped, only every `past_backoff`-th wait (a probe).
fn past_backoff_pass(a: u64) -> bool {
    let probe = super::tune::past_backoff() as u32;
    if probe == 0 {
        return true;
    }
    let mut g = PAST_BACKOFF.lock();
    let Some(e) = g.as_mut().and_then(|m| m.get_mut(&(a & !7))) else { return true };
    if e.0 < 512 {
        return true;
    }
    e.1 += 1;
    if e.1 >= probe {
        e.1 = 0;
        return true;
    }
    PAST_SKIPPED.fetch_add(1, Ordering::Relaxed);
    false
}

/// Per label address: the recent drop rate of its stretches with held runs (1/1024) and
/// stretches without them since the last probe (`tune held_runs`): a label whose stretches
/// drop with held runs (they read further ahead than the command processor alone, e.g. what
/// queue 0x50 writes before its label at Castlewall Front) goes on without them, probing every
/// 16th stretch.
static HELD_BACKOFF: parking_lot::Mutex<Option<HashMap<u64, (u32, u32)>>> = parking_lot::Mutex::new(None);

/// Whether a stretch past a wait on label `a` may plan held runs (`HELD_BACKOFF`).
fn held_backoff_pass(a: u64) -> bool {
    let mut g = HELD_BACKOFF.lock();
    let Some(e) = g.as_mut().and_then(|m| m.get_mut(&(a & !7))) else { return true };
    if e.0 < 512 {
        return true;
    }
    e.1 += 1;
    if e.1 >= 16 {
        e.1 = 0;
        return true;
    }
    HELD_RUNS[15].fetch_add(1, Ordering::Relaxed);
    false
}

/// A stretch with held runs past a wait on label `a` was dropped or committed (`HELD_BACKOFF`).
fn held_backoff_note(a: u64, dropped: bool) {
    let mut g = HELD_BACKOFF.lock();
    let e = g.get_or_insert_with(HashMap::new).entry(a & !7).or_insert((0, 0));
    e.0 = e.0 * 3 / 4 + if dropped { 256 } else { 0 };
}

/// A stretch past a wait on label `a` was dropped (`dropped`) or committed:
/// the label's drop rate, each outcome weighing a fourth.
fn past_backoff_note(a: u64, dropped: bool) {
    let mut g = PAST_BACKOFF.lock();
    let e = g.get_or_insert_with(HashMap::new).entry(a & !7).or_insert((0, 0));
    e.0 = e.0 * 3 / 4 + if dropped { 256 } else { 0 };
}

/// Per label address gone past: the CPU reads of its dropped stretches that
/// other queues wrote meanwhile (most recent last, at most PAST_CONFLICTS_MAX;
/// e.g. the indirect arguments of particle draws queue 0x50 writes before
/// its label). A later stretch past a wait on that label stops before such
/// a read and settles once the label passes, keeping what it recorded, where
/// reading it would drop the stretch (FINDINGS.md 2026-10-27 late night).
/// Each with the flip it was learned at and its hold's doublings: past its
/// hold (`tune past_conflict_hold` flips, doubled each time the same read
/// drops a stretch again, up to 32x) a stretch goes past it again (a
/// conflict learned once, e.g. during a load, stopped every later stretch:
/// 0-271 stops a window by boot, ~1 ms each, FINDINGS.md 2026-10-29).
static PAST_CONFLICTS: parking_lot::Mutex<Option<HashMap<u64, Vec<PastConflict>>>> = parking_lot::Mutex::new(None);
const PAST_CONFLICTS_MAX: usize = 16;

#[derive(Clone, Copy)]
struct PastConflict {
    read: (u64, u64),
    learned: u64,
    level: u32,
}

fn past_conflicts_note(a: u64, reads: &[(u64, u64)]) {
    if reads.is_empty() {
        return;
    }
    let frame = super::FRAMES.load(Ordering::Relaxed);
    let mut g = PAST_CONFLICTS.lock();
    let v = g.get_or_insert_with(HashMap::new).entry(a & !7).or_default();
    for &r in reads {
        let level = match v.iter().position(|c| c.read == r) {
            Some(i) => (v.remove(i).level + 1).min(5),
            None => 0,
        };
        v.push(PastConflict { read: r, learned: frame, level });
    }
    if v.len() > PAST_CONFLICTS_MAX {
        let n = v.len() - PAST_CONFLICTS_MAX;
        v.drain(..n);
    }
}

/// Whether `reads` overlap a read that dropped a stretch past a wait on
/// label `a` within its hold.
fn past_conflicts_hit(a: u64, reads: &[(u64, u64)]) -> bool {
    let g = PAST_CONFLICTS.lock();
    let Some(v) = g.as_ref().and_then(|m| m.get(&(a & !7))) else { return false };
    let (hold, frame) = (super::tune::past_conflict_hold(), super::FRAMES.load(Ordering::Relaxed));
    reads.iter().any(|&(x, n)| v.iter().any(|c| x < c.read.0 + c.read.1 && c.read.0 < x + n && (hold == 0 || frame < c.learned + (hold << c.level))))
}

/// The counters at the frame window's start (`past_wait_window_clause`).
static PAST_WINDOW: parking_lot::Mutex<Option<[u64; 9]>> = parking_lot::Mutex::new(None);

/// The frame window's stretches past label waits per flip, how many were
/// dropped and settled at a stop, and their commands: one clause of the
/// `frames at` line ("" without stretches).
pub fn past_wait_window_clause(flips: u64) -> String {
    let now: [u64; 9] = std::array::from_fn(|i| if i < 8 { PAST_WAITS[i].load(Ordering::Relaxed) } else { PAST_SKIPPED.load(Ordering::Relaxed) });
    let Some(w0) = PAST_WINDOW.lock().replace(now) else { return String::new() };
    let d: [u64; 9] = std::array::from_fn(|i| now[i].saturating_sub(w0[i]));
    if d[0] == 0 && d[8] == 0 {
        return String::new();
    }
    let flips = flips.max(1) as f64;
    format!(
        "; past label waits per flip: {:.1} stretches ({:.1} dropped, {:.1} settled at a stop), {:.0} commands each, {:.1} waits not gone past after drops",
        d[0] as f64 / flips,
        d[2] as f64 / flips,
        d[6] as f64 / flips,
        d[3] as f64 / d[0].max(1) as f64,
        d[8] as f64 / flips
    )
}

/// One status line ("" before any stretch past a label wait).
pub fn past_wait_status() -> String {
    let p = |i: usize| PAST_WAITS[i].load(Ordering::Relaxed);
    if p(0) == 0 {
        return String::new();
    }
    let list = |l: &parking_lot::Mutex<Vec<(&'static str, u64)>>| {
        let mut v = l.lock().clone();
        v.sort_by(|a, b| b.1.cmp(&a.1));
        v.iter().map(|(w, n)| format!("{w} {n}")).collect::<Vec<_>>().join(", ")
    };
    let n = p(0).max(1) as f64;
    format!(
        "    recording past label waits: {} stretches, {} committed, {} dropped ({}; {} waits not gone past after drops), {} commands recorded past the waits ({:.1} each), {:.0} us from a wait to its settling; settled once promised {}, at a stop {} ({}; {:.0} us waited for the label there on average); checked again at their label for a later epoch {} ({} ranges, {} stale, {:.0} us each); hot pages of small bindings uploaded as they are when the stretch runs: {} runs, {} pages\n",
        p(0),
        p(1),
        p(2),
        list(&PAST_WAIT_DROPS),
        PAST_SKIPPED.load(Ordering::Relaxed),
        p(3),
        p(3) as f64 / n,
        p(4) as f64 / 1e3 / n,
        p(5),
        p(6),
        list(&PAST_WAIT_STOPS),
        p(7) as f64 / 1e3 / p(6).max(1) as f64,
        super::exec::PAST_CHECKS[0].load(Ordering::Relaxed),
        super::exec::PAST_CHECKS[1].load(Ordering::Relaxed),
        super::exec::PAST_CHECKS[2].load(Ordering::Relaxed),
        super::exec::PAST_CHECKS[3].load(Ordering::Relaxed) as f64 / 1e3 / super::exec::PAST_CHECKS[0].load(Ordering::Relaxed).max(1) as f64,
        super::exec::PAST_HOT[0].load(Ordering::Relaxed),
        super::exec::PAST_HOT[1].load(Ordering::Relaxed)
    ) + &{
        let l = list(&PAST_WAIT_NOT);
        if l.is_empty() {
            String::new()
        } else {
            format!("    the graphics CP's waits on queue 0x50's labels not gone past, by why: {l}
")
        }
    } + &{
        let r = |i: usize| super::exec::PAST_REPAIRS[i].load(Ordering::Relaxed);
        if r(0) == 0 {
            String::new()
        } else {
            format!("    stale pages uploaded before their stretch (`tune past_repair`): {} stretches, {} runs, {} pages\n", r(0), r(1), r(2))
        }
    } + &{
        let k = |i: usize| PAST_POLLS[i].load(Ordering::Relaxed);
        if k(1) == 0 {
            String::new()
        } else {
            format!("    label waits passed on memory kept at their stretch's settle: {} waits in {} stretches a batch submitted meanwhile would have dropped\n", k(0), k(1))
        }
    } + &{
        let n = super::exec::COMMITS_AFTER.load(Ordering::Relaxed);
        if n == 0 { String::new() } else { format!("    committed stretches whose first batch waits for their label's writer: {n}\n") }
    } + &held_runs_line()
}

/// The held runs' clause of the status ("" without any).
fn held_runs_line() -> String {
    let h = |i: usize| HELD_RUNS[i].load(Ordering::Relaxed);
    if h(0) == 0 && h(10) == 0 && h(11) == 0 {
        return String::new();
    }
    format!(
        "    held runs (`tune held_runs`): {} runs planned in stretches ({} worker chunks, {} commands planned for them; {:.0} us waited for a chunk on average), {} chunks stopped, {} runs cancelled, {} runs not planned (clears pending or arguments of its held work), {} plannings stopped the stretch; stretches with held runs: {} committed, {} dropped; {} chunk batches submitted at commits ({} left in the workers' executors); chunks stopped reading held work before their run {}, using an image it clears {} or overwrote {} ({} of its fast clears applied at a run's start); chunks collected after their stretch settled {} ({} dropped: {}); stretches without held runs after drops {}\n",
        h(0),
        h(1),
        h(8),
        h(4) as f64 / 1e3 / h(1).max(1) as f64,
        h(2),
        h(5),
        h(10),
        h(11),
        h(6),
        h(7),
        h(3),
        h(9),
        super::exec::HELD_RUN_STOPS[0].load(Ordering::Relaxed),
        super::exec::HELD_RUN_STOPS[1].load(Ordering::Relaxed),
        super::exec::HELD_RUN_STOPS[2].load(Ordering::Relaxed),
        h(12),
        h(13),
        h(14),
        {
            let mut v = HELD_DROPS.lock().clone();
            v.sort_by(|a, b| b.1.cmp(&a.1));
            v.iter().map(|(w, n)| format!("{w} {n}")).collect::<Vec<_>>().join(", ")
        },
        h(15)
    )
}

/// Two command-processor states are the same (registers past the shorter
/// one's held count zero), SH registers `sh_skip` left out.
fn chunk_states_equal(a: &ChunkState, b: &ChunkState, sh_skip: &[u32]) -> bool {
    let same = |x: &[u32], y: &[u32]| {
        let m = x.len().min(y.len());
        x[..m] == y[..m] && x[m..].iter().all(|&v| v == 0) && y[m..].iter().all(|&v| v == 0)
    };
    let sh = if sh_skip.is_empty() {
        same(&a.regs.sh, &b.regs.sh)
    } else {
        let n = a.regs.sh.len().max(b.regs.sh.len());
        (0..n).all(|i| sh_skip.contains(&(i as u32)) || a.regs.sh.get(i).copied().unwrap_or(0) == b.regs.sh.get(i).copied().unwrap_or(0))
    };
    same(&a.regs.ctx, &b.regs.ctx)
        && sh
        && same(&a.regs.uc, &b.regs.uc)
        && a.index_base == b.index_base
        && a.index_count == b.index_count
        && a.num_instances == b.num_instances
        && a.draw_indirect_base == b.draw_indirect_base
        && a.dispatch_indirect_base == b.dispatch_indirect_base
}

/// Commands per chunk at least (a third of `tune par_min`, the commands a
/// run needs to be split, when that is below 192), and at most per
/// parallel run.
const PAR_CHUNK_MIN: usize = 64;
const PAR_MAX: usize = 8192;

/// `Cp::run_pass`, `Cp::compute_pass`: a timestamp, a label wait (met in
/// memory), a CP write, a label wait met by a write earlier in the run.
const RUN_PASS_TS: u8 = 1;
const RUN_PASS_WAIT: u8 = 2;
const COMPUTE_PASS_WRITE: u8 = 3;
const RUN_PASS_WAIT_OWN: u8 = 4;
/// Runs' held packets (`Cp::run_pass`, `Cp::compute_pass`): CP writes
/// workers performed, label waits passed by split runs (their effect
/// applied when planned), timestamps and label waits workers recorded in
/// their chunks.
pub static RUN_PASSED: [AtomicU64; 4] = [const { AtomicU64::new(0) }; 4];

/// The status line's part for `RUN_PASSED` (empty while it counted nothing).
pub fn run_passed_line() -> String {
    let v: Vec<u64> = RUN_PASSED.iter().map(|c| c.load(Ordering::Relaxed)).collect();
    if v.iter().all(|&x| x == 0) {
        return String::new();
    }
    format!("; runs' held packets: label waits passed when planned {}, timestamps recorded by workers {}, label waits passed by workers {}, CP writes performed by workers {} (chunks stopped reading an earlier chunk's {}); batches of labels and stamps kept open into a run {}", v[1], v[2], v[3], v[0], super::exec::RUN_WRITE_STOPS.load(Ordering::Relaxed), PRE_RUN_KEPT.load(Ordering::Relaxed)) + &plan_ahead_part()
}

/// The runs' status line's part for `PLAN_AHEAD` (empty while none was made).
fn plan_ahead_part() -> String {
    let v: Vec<u64> = PLAN_AHEAD.iter().map(|c| c.load(Ordering::Relaxed)).collect();
    if v[0] == 0 {
        return String::new();
    }
    format!("; runs planned ahead {} ({:.1} us each), taken {}, not taken {}", v[0], v[3] as f64 / v[0] as f64 / 1e3, v[1], v[2])
}

/// Batches with no command, writing labels and stamps only, that went on as
/// the graphics command processor's own part of a run (`Cp::par_plan`).
pub static PRE_RUN_KEPT: AtomicU64 = AtomicU64::new(0);

/// Command-buffer ends that left the batch open (`tune cmdbuf_merge`).
pub static CMDBUF_MERGED: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);

fn is_command(op: u32) -> bool {
    matches!(op, op::DRAW_INDEX_2 | op::DRAW_INDEX_AUTO | op::DRAW_INDEX_OFFSET_2 | op::DRAW_INDIRECT | op::DRAW_INDEX_INDIRECT | op::DISPATCH_DIRECT | op::DISPATCH_INDIRECT)
}

/// Packets that leave a run open (state only; see `Cp::scan_commands`).
/// Flushes and cache actions that compute queues' parallel runs keep inside
/// them (front end 2: their chunks keep a barrier in the batch and order
/// later CP writes after the GPU work; the graphics queue's end its runs).
fn par_event(op: u32, compute: bool) -> bool {
    compute && matches!(op, op::EVENT_WRITE | op::ACQUIRE_MEM) && super::fe2::on()
}

/// Parallel workers of a compute queue's command processor for its
/// synchronization-free runs: four under front end 2 (+6-10% at the vista,
/// FINDINGS.md 2026-10-28 night), none in the old front end (a loss there).
fn compute_workers() -> usize {
    if !super::fe2::on() {
        return 0;
    }
    match super::tune::par_compute() {
        1000 => super::par::workers(),
        v => (v as usize).min(16),
    }
}

fn is_state(op: u32) -> bool {
    matches!(
        op,
        op::SET_CONTEXT_REG
            | op::SET_CONTEXT_REG_INDEX
            | op::SET_SH_REG
            | op::SET_SH_REG_INDEX
            | op::SET_UCONFIG_REG
            | op::SET_UCONFIG_REG_INDEX
            | op::LOAD_CONTEXT_REG_INDEX
            | op::LOAD_SH_REG_INDEX
            | op::LOAD_UCONFIG_REG
            | op::INDEX_TYPE
            | op::INDEX_BASE
            | op::INDEX_BUFFER_SIZE
            | op::NUM_INSTANCES
            | op::CONTEXT_CONTROL
            | op::SET_BASE
    )
}

/// A command the scan ahead saw: its packet address and the preparations
/// handed to the workers.
struct AheadCmd {
    at: u64,
    stages: [Option<Arc<super::ahead::Slot>>; 2],
    /// A draw's baked-state registers (scans for `prefetch_gfx` only).
    gfx: Option<super::exec::GfxRegs>,
}


/// Register view of a scan ahead: the command processor's register files
/// with the writes of the packets scanned so far.
struct ScanRegs<'a> {
    sh: &'a [u32],
    ctx: &'a [u32],
}

impl super::shaders::RegRead for ScanRegs<'_> {
    fn sh(&self, i: u32) -> u32 {
        self.sh.get(i as usize).copied().unwrap_or(0)
    }
    fn ctx(&self, a: u32) -> u32 {
        self.ctx.get(((a - 0x28000) / 4) as usize).copied().unwrap_or(0)
    }
}

/// SET_PREDICATION (as Agc encodes it: control = operation << 16 |
/// wait << 12 | condition << 8, address, dwords covered).
#[derive(Clone, Copy, Debug)]
struct Pred {
    op: u32,
    cond: u32,
    addr: u64,
    count: u32,
}

unsafe fn rd32(a: u64) -> u32 {
    unsafe { std::ptr::read_volatile(a as *const u32) }
}
unsafe fn rd64(a: u64) -> u64 {
    unsafe { std::ptr::read_volatile(a as *const u64) }
}
unsafe fn wr32(a: u64, v: u32) {
    super::precache::invalidate(a, 4);
    unsafe { std::ptr::write_volatile(a as *mut u32, v) }
}
unsafe fn wr64(a: u64, v: u64) {
    super::precache::invalidate(a, 8);
    unsafe { std::ptr::write_volatile(a as *mut u64, v) }
}

fn addr64(lo: u32, hi: u32) -> u64 {
    lo as u64 | ((hi as u64 & 0xFFFF_FFFF) << 32)
}

/// Guest addresses the command processor may touch.
pub(crate) fn valid(a: u64, len: u64) -> bool {
    a >= crate::mem::GUEST_BASE && a.checked_add(len).is_some_and(|e| e <= crate::mem::GUEST_END)
}

/// Whether guest memory [a, a+len) is mapped and readable.
pub fn guest_readable(a: u64, len: u64) -> bool {
    valid(a, len) && crate::mem::readable(a, len)
}

/// Compute shaders whose dispatches `DS_TRACE_CS` traces.
fn trace_cs_list() -> &'static [u64] {
    static L: std::sync::OnceLock<Vec<u64>> = std::sync::OnceLock::new();
    L.get_or_init(|| std::env::var("DS_TRACE_CS").map(|v| v.split(',').filter_map(|a| u64::from_str_radix(a.trim().trim_start_matches("0x"), 16).ok()).collect()).unwrap_or_default())
}

/// The title's timestamp at `a` written now by the CPU (its page never
/// held by front end 4; the store is a CPU write front end 2's takes find).
fn cpu_timestamp(a: u64) {
    if !valid(a, 8) {
        return;
    }
    super::fe2::c4_cp_target(a, 8);
    unsafe { wr64(a, gpu_clock()) };
    TS_CPU.fetch_add(1, Ordering::Relaxed);
}

/// The title's timestamps written (`cpu_timestamp`).
pub static TS_CPU: AtomicU64 = AtomicU64::new(0);

/// GPU timestamp (100 MHz, like the console's GPU clock counter).
pub fn gpu_clock() -> u64 {
    static START: std::sync::OnceLock<Instant> = std::sync::OnceLock::new();
    let s = START.get_or_init(Instant::now);
    (s.elapsed().as_nanos() / 10) as u64
}

/// Command buffers up to this size are copied when execution starts.
const FETCH_DWORDS: u32 = 512;
/// Payload dwords of each packet of a larger buffer copied when execution
/// reaches the packet (`Cp::run`).
const PACKET_FETCH: usize = 32;

/// A command buffer as the CP fetched it (see `Cp::run`): small buffers are
/// copied, larger ones read from guest memory.
#[derive(Clone)]
struct Fetched {
    base: u64,
    words: Vec<u32>,
    /// Packets the CPU streams behind a REWIND (not scanned ahead).
    live: bool,
}

impl Fetched {
    fn new(base: u64, size: u32) -> Fetched {
        static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
        let on = *ON.get_or_init(|| std::env::var("DS_CP_FETCH").ok().as_deref() != Some("0"));
        let n = size as usize;
        let words = if on && size > 0 && size <= FETCH_DWORDS && valid(base, 4 * n as u64) { unsafe { std::slice::from_raw_parts(base as *const u32, n) }.to_vec() } else { Vec::new() };
        Fetched { base, words, live: false }
    }

    /// The `len` dwords at `a` when they are all fetched, or all live (none
    /// fetched); None when they straddle the fetched range.
    fn slice(&self, a: u64, len: usize) -> Option<&[u32]> {
        if self.words.is_empty() {
            return valid(a, 4 * len as u64).then(|| unsafe { std::slice::from_raw_parts(a as *const u32, len) });
        }
        let i = (a.checked_sub(self.base)? / 4) as usize;
        self.words.get(i..i + len)
    }

    /// The dword at `a`: fetched, or live memory past the fetched range.
    fn rd(&self, a: u64) -> u32 {
        if a >= self.base {
            if let Some(&w) = self.words.get(((a - self.base) / 4) as usize) {
                return w;
            }
        }
        unsafe { rd32(a) }
    }
}

fn compare(func: u32, value: u64, reference: u64) -> bool {
    match func & 7 {
        0 => true,
        1 => value < reference,
        2 => value <= reference,
        3 => value == reference,
        4 => value != reference,
        5 => value >= reference,
        6 => value > reference,
        _ => false,
    }
}

/// Last GPU write to each label-sized address: (queue, value, packet address)
/// for wait diagnostics.
/// (and the exact address written: two 4-byte labels share a slot).
static LAST_WRITE: parking_lot::Mutex<Option<std::collections::HashMap<u64, (String, u64, u64, u64)>>> = parking_lot::Mutex::new(None);

/// Recent CP writes to each polled label (time, value), newest last: a
/// queue that reaches a WAIT_REG_MEM late still sees the writes made after
/// its command buffer was submitted (`missed_write`).
static LABEL_WRITES: parking_lot::Mutex<Option<HashMap<u64, std::collections::VecDeque<(Instant, u64)>>>> = parking_lot::Mutex::new(None);
const LABEL_WRITES_KEEP: usize = 64;
/// Values added to `LABEL_WRITES`, over all labels: a waiting queue checks `counter_passed` again
/// when it moves. (The history can lag the label's memory: a GPU-performed write reaches it when
/// its batch completes, a CP write just after memory; a check made between them, keyed by the
/// value in memory, came too early and was never made again: the review of 2026-11-24.)
static LABEL_WRITES_GEN: AtomicU64 = AtomicU64::new(0);
/// Waits passed on a write made after their submission (diagnostics).
pub static MISSED: AtomicU64 = AtomicU64::new(0);

/// Whether a CP write to `addr` made at or after `since` satisfies the
/// comparison.
fn missed_write(addr: u64, wide: bool, func: u32, mask: u64, reference: u64, since: Instant) -> bool {
    let g = LABEL_WRITES.lock();
    let Some(h) = g.as_ref().and_then(|m| m.get(&addr)) else { return false };
    h.iter().rev().take_while(|(t, _)| *t >= since).any(|&(_, v)| {
        let v = if wide { v } else { v & 0xFFFF_FFFF };
        compare(func, v & mask, reference & mask)
    })
}

/// Whether an equality wait for `reference` waits on a counter that has
/// moved past it: every recorded write from `reference`'s last one on (or,
/// when the counter passed it before the recorded window, every recorded
/// write; at least three, starting a whole number of steps and less than 64
/// steps above it) kept the value or added one step (1..4, the same
/// throughout; the title's frame counters count by 1 and by 2), and it rose
/// by two steps or more (a frame counter; a flag set from 0 to 1 is not
/// one). Run 20261011-103835: acb0x48 waited for 0x801 at 0x805 after cold
/// pipeline compiles and every queue stopped.
fn counter_passed(addr: u64, wide: bool, func: u32, mask: u64, reference: u64) -> bool {
    if func & 7 != 3 {
        return false;
    }
    let g = LABEL_WRITES.lock();
    let Some(h) = g.as_ref().and_then(|m| m.get(&addr)) else { return false };
    let val = |v: u64| (if wide { v } else { v & 0xFFFF_FFFF }) & mask;
    let r = reference & mask;
    let vals: Vec<u64> = h.iter().map(|&(_, v)| val(v)).collect();
    // (The step of the latest writes.)
    let Some(step) = vals.windows(2).rev().map(|w| w[1].wrapping_sub(w[0])).find(|&d| d != 0) else { return false };
    if !(1..=4).contains(&step) {
        return false;
    }
    // The run of writes since the last one that neither kept the value nor added the step (a
    // reset, a ring's wrap, a flag going back): a counter only when the whole window is one run,
    // or when the run holds 8 steps or more (a ring of fewer values never does; a counter reset
    // at a load is trusted again after 8 steps). (The review of 2026-11-24, once the check ran
    // while waiting too: a ring's queue waiting at 2 for its next 1 must not pass at 3; a whole
    // window without a break, tried first, kept late waits after a reset blocked for 64 writes.)
    let ok = |w: &[u64]| w[1] == w[0] || w[1] == w[0].wrapping_add(step);
    let start = vals.windows(2).rposition(|w| !ok(w)).map_or(0, |i| i + 1);
    let run = &vals[start..];
    if start > 0 && run.windows(2).filter(|w| w[1] != w[0]).count() < 8 {
        return false;
    }
    let later = match run.iter().rposition(|&v| v == r) {
        Some(i) => &run[i..],
        None if start == 0 && run.len() >= 3 && run[0] > r && (run[0] - r) % step == 0 && run[0] - r < 64 * step => run,
        None => return false,
    };
    later.last().is_some_and(|&l| l >= r.wrapping_add(2 * step))
}

/// The recorded writes of a waited-on label (wait reports).
fn label_writes(addr: u64) -> String {
    let g = LABEL_WRITES.lock();
    let Some(h) = g.as_ref().and_then(|m| m.get(&addr)) else { return "none".into() };
    let v: Vec<String> = h.iter().rev().take(12).rev().map(|&(_, v)| format!("{v:#x}")).collect();
    v.join(" ")
}
/// Equality waits passed because their counter had moved on (diagnostics).
pub static COUNTER_PASSED: AtomicU64 = AtomicU64::new(0);

/// CP memory writes the CPU performed, for the race checks (`tune
/// race_checks`): a global order number and, per 8-byte granule, the last
/// such write's (number, batch it ran after (0: performed on the CP's
/// thread when the CP reached it), queue, packet, opcode).
pub static CPW_SEQ: AtomicU64 = AtomicU64::new(0);
type CpwEntry = (u64, u64, Arc<str>, u64, u32);
static CPW_LOG: parking_lot::Mutex<Option<HashMap<u64, CpwEntry>>> = parking_lot::Mutex::new(None);

thread_local! {
    /// The batch whose deferred CP effects this thread is performing (the
    /// completion thread; 0 elsewhere).
    static AFTER_BATCH: std::cell::Cell<u64> = const { std::cell::Cell::new(0) };
}

/// The completion thread performs batch `seq`'s CP effects (0: done).
pub fn set_after_batch(seq: u64) {
    AFTER_BATCH.with(|c| c.set(seq));
}

/// The batch whose effects this thread performs (the completion thread's
/// actions; 0 elsewhere).
pub fn after_batch() -> u64 {
    AFTER_BATCH.with(|c| c.get())
}

fn cpw_note(queue: &str, a: u64, n: u64, at: u64, opcode: u32) {
    if !super::tune::race_checks() {
        return;
    }
    let s = CPW_SEQ.fetch_add(1, Ordering::AcqRel) + 1;
    let after = AFTER_BATCH.with(|c| c.get());
    let q: Arc<str> = queue.into();
    let mut g = CPW_LOG.lock();
    let m = g.get_or_insert_with(HashMap::new);
    if m.len() > 1 << 21 {
        m.clear();
    }
    let mut g8 = a & !7;
    while g8 < a + n.min(1 << 16) {
        m.insert(g8, (s, after, q.clone(), at, opcode));
        g8 += 8;
    }
}

/// The latest CPU-performed CP write into [a, a + n) with an order number
/// above `after` (as `cpw_since`).
pub fn cpw_since_range(a: u64, n: u64, after: u64) -> Option<(u64, u64, String, u64, u32)> {
    let g = CPW_LOG.lock();
    let m = g.as_ref()?;
    let mut best: Option<&CpwEntry> = None;
    let mut g8 = a & !7;
    while g8 < a + n {
        if let Some(e) = m.get(&g8).filter(|e| e.0 > after) {
            if best.is_none_or(|b| e.0 > b.0) {
                best = Some(e);
            }
        }
        g8 += 8;
    }
    best.map(|e| (e.0, e.1, e.2.to_string(), e.3, e.4))
}

/// The last CPU-performed CP write to the granule of `addr` with an order
/// number above `after`: (number, batch it ran after, queue, packet, opcode).
pub fn cpw_since(addr: u64, after: u64) -> Option<(u64, u64, String, u64, u32)> {
    let g = CPW_LOG.lock();
    let e = g.as_ref()?.get(&(addr & !7))?;
    (e.0 > after).then(|| (e.0, e.1, e.2.to_string(), e.3, e.4))
}

fn note_write(queue: &str, addr: u64, value: u64, at: u64) {
    super::homes::cpu_written(addr, 8);
    if WAITED.lock().as_ref().is_some_and(|w| w.contains(&(addr & !7))) {
        let mut g = LABEL_WRITES.lock();
        let h = g.get_or_insert_with(HashMap::new).entry(addr).or_default();
        // A repeated value refreshes its time: counters several queues
        // write each frame keep their recent frames in the window.
        match h.back_mut() {
            Some(last) if last.1 == value => last.0 = Instant::now(),
            _ => {
                if h.len() >= LABEL_WRITES_KEEP {
                    h.pop_front();
                }
                h.push_back((Instant::now(), value));
                LABEL_WRITES_GEN.fetch_add(1, Ordering::Release);
            }
        }
    }
    {
        let mut g = LAST_WRITE.lock();
        let m = g.get_or_insert_with(Default::default);
        if m.len() > 1 << 16 {
            m.clear();
        }
        m.insert(addr & !7, (queue.to_string(), value, at, addr));
    }
    latch_waits(addr & !7, 8);
    label_event(queue, b'W', addr, value, 0, at);
}

/// A WAIT_REG_MEM in progress. Hardware polls continuously, so a label one
/// queue sets is seen by a queue already waiting on it even if the CPU
/// resets it soon after (the title resets per-frame labels while building
/// the next frame). Our pollers sleep between checks and missed such
/// short-lived values, leaving every queue waiting on another (runs
/// 20261008-185032, -204128, -210312). CP writes therefore evaluate the
/// waits on the written word at once and latch the ones they satisfy.
struct ActiveWait {
    addr: u64,
    wide: bool,
    func: u32,
    mask: u64,
    reference: u64,
    hit: std::sync::Arc<std::sync::atomic::AtomicBool>,
}

static ACTIVE_WAITS: parking_lot::Mutex<Vec<ActiveWait>> = parking_lot::Mutex::new(Vec::new());
/// Blocking waits in progress (cheap check before taking the lock).
static ACTIVE_COUNT: AtomicU64 = AtomicU64::new(0);

/// A wait on a label a GPU queue writes sleeps on the label's own address
/// (`WaitOnAddress`, `tune label_wake`) instead of polling: a promise for
/// the label, a latched CP write or a GPU-performed write to it wakes the
/// waits on that address (and a change of its value in memory before they
/// sleep). The compute queues wait on the graphics queue's labels 12-23 ms
/// at a time, ~4 times per frame each, and their polls (spins, yields,
/// 100 us sleeps) took the front end's CPUs (FINDINGS.md 2026-10-16).
fn wake_label(addr: u64, bytes: u8) {
    if ACTIVE_COUNT.load(Ordering::Acquire) == 0 {
        return;
    }
    unsafe { windows_sys::Win32::System::Threading::WakeByAddressAll(addr as *const core::ffi::c_void) };
    if bytes >= 8 {
        // A 32-bit wait on its upper half.
        unsafe { windows_sys::Win32::System::Threading::WakeByAddressAll((addr + 4) as *const core::ffi::c_void) };
    }
}

/// Whether a queue blocks on the label-sized word at `a` now.
pub(super) fn waited_now(a: u64) -> bool {
    if ACTIVE_COUNT.load(Ordering::Acquire) == 0 {
        return false;
    }
    let a = a & !7;
    ACTIVE_WAITS.lock().iter().any(|w| w.addr & !7 == a)
}
/// Waits satisfied by a latched CP write whose value was gone by the time
/// the waiter looked again (diagnostics).
pub static LATCHED: AtomicU64 = AtomicU64::new(0);
/// Parallel runs split at equal recording cost (`tune par_cost`).
pub static PAR_COSTED: AtomicU64 = AtomicU64::new(0);
/// Batches the graphics queue ended for `tune split_every` (diagnostics).
pub static SPLITS: AtomicU64 = AtomicU64::new(0);
/// Sleeps of label waits on their label's address (diagnostics).
pub static WAKE_SLEEPS: AtomicU64 = AtomicU64::new(0);

/// Latches the active waits overlapping [addr, addr+bytes) that the
/// current memory contents satisfy (called right after a CP write).
fn latch_waits(addr: u64, bytes: u64) {
    let g = ACTIVE_WAITS.lock();
    let mut hits: Vec<u64> = Vec::new();
    for w in g.iter() {
        let n = if w.wide { 8 } else { 4 };
        if w.addr < addr + bytes && addr < w.addr + n {
            let v = if w.wide { unsafe { rd64(w.addr) } } else { (unsafe { rd32(w.addr) }) as u64 };
            if compare(w.func, v & w.mask, w.reference & w.mask) {
                w.hit.store(true, Ordering::SeqCst);
                hits.push(w.addr);
            }
        }
    }
    drop(g);
    for a in hits {
        wake_label(a, 4);
    }
}

/// A label write the GPU performed in order within a batch
/// (`Exec::gpu_cp_write`), seen once the batch completed: recorded like a
/// CPU-performed CP write (the history late waiters pass on, `missed_write`)
/// and latched for the active waits its value satisfies (by the value: the
/// title may have reset the label since). Without this, a queue reaching its
/// wait after the batch completed and the label was reset blocked forever:
/// run 20261012-141114 stopped at flip 737 with graphics and acb0x50 waiting
/// for 1 at 0x11d37ff7e0, which acb0x20's GPU-side write had set ("recorded
/// writes none").
pub(super) fn gpu_label_written(queue: &str, addr: u64, value: u64, bytes: u8) {
    if WAITED.lock().as_ref().is_some_and(|w| w.contains(&(addr & !7))) {
        let mut g = LABEL_WRITES.lock();
        let h = g.get_or_insert_with(HashMap::new).entry(addr).or_default();
        match h.back_mut() {
            Some(last) if last.1 == value => last.0 = Instant::now(),
            _ => {
                if h.len() >= LABEL_WRITES_KEEP {
                    h.pop_front();
                }
                h.push_back((Instant::now(), value));
                LABEL_WRITES_GEN.fetch_add(1, Ordering::Release);
            }
        }
    }
    let mut woke: Vec<u64> = Vec::new();
    {
        let g = ACTIVE_WAITS.lock();
        for w in g.iter() {
            // The waited word within the written bytes.
            let v = if w.addr == addr {
                if w.wide {
                    if bytes < 8 {
                        continue;
                    }
                    value
                } else {
                    value & 0xFFFF_FFFF
                }
            } else if !w.wide && bytes >= 8 && w.addr == addr + 4 {
                value >> 32
            } else {
                continue;
            };
            if compare(w.func, v & w.mask, w.reference & w.mask) {
                w.hit.store(true, Ordering::SeqCst);
                woke.push(w.addr);
            }
        }
    }
    for a in woke {
        wake_label(a, 4);
    }
    label_event(queue, b'W', addr, value, 0, 0);
}

/// Label activity of all queues (CP writes, waits) on addresses some
/// WAIT_REG_MEM polls: a flight recorder dumped for the label when a wait
/// stalls. A value that changes between CP events was written by the CPU
/// (or by shader work).
struct LabelEvent {
    us: u64,
    queue: String,
    /// W = CP write, I = wait satisfied at once, S = wait blocks, E = wait
    /// satisfied, L = wait satisfied by a latched CP write.
    kind: u8,
    addr: u64,
    value: u64,
    reference: u64,
    packet: u64,
    /// The title submission its command processor was running (`CUR_SUB`).
    sub: u64,
}

thread_local! {
    /// The title submission the calling command processor runs (`Cp::new_submission`;
    /// the label log's events carry it: `label_dump`).
    static CUR_SUB: std::cell::Cell<u64> = const { std::cell::Cell::new(0) };
}

static LABEL_LOG: parking_lot::Mutex<std::collections::VecDeque<LabelEvent>> = parking_lot::Mutex::new(std::collections::VecDeque::new());
static WAITED: parking_lot::Mutex<Option<HashSet<u64>>> = parking_lot::Mutex::new(None);

/// Whether a command processor of ours has waited on the label at `a`
/// (WAIT_REG_MEM on memory: `tune fe2_cpw_waited`).
pub(super) fn cp_waited(a: u64) -> bool {
    WAITED.lock().as_ref().is_some_and(|w| w.contains(&(a & !7)))
}

/// Label values deferred CP writes will store: address -> (batch sequence,
/// value, bytes, whether the CPU stores it after the batch completed rather
/// than the GPU within it, when it was made, its writer's view then: the
/// epoch the CPU writes the title ordered before the write are older than,
/// every earlier wait of the writer having passed). A WAIT_REG_MEM that a promised value satisfies passes at
/// once and orders its queue after that batch (`Exec::depend_on`): its later
/// GPU work follows the batch on the device queue, its CP writes wait for
/// it, its CP reads wait for unfinished writers (`Exec::sync_for_reads`), so
/// the command processors record ahead of the GPU instead of idling until
/// each batch completes. `DS_PROMISES=0` turns this off.
static PROMISES: parking_lot::Mutex<Option<HashMap<u64, (u64, u64, u8, bool, Instant, u32)>>> = parking_lot::Mutex::new(None);
/// Moves with every promise made: a polling wait looks at the promises
/// again only then (one that did not satisfy it cannot start to; it can
/// only go stale). Every poll took PROMISES' lock before: ~6% of the
/// compute queues' busy samples, run 20261013-115322.
static PROMISE_GEN: AtomicU64 = AtomicU64::new(0);
/// Waits passed on a promise (diagnostics).
pub static PROMISED: AtomicU64 = AtomicU64::new(0);
/// Multi-draw packets (DRAW_INDIRECT_MULTI / DRAW_INDEX_INDIRECT_MULTI), the
/// draws they expanded to, and those packets whose count GPU work wrote
/// (diagnostics: how much of the per-draw work they cause).
pub static MDI_PACKETS: AtomicU64 = AtomicU64::new(0);
/// Waits passed on a GPU queue's write that ended no copy-cache epoch
/// (`Cp::last_wait_gpu`).
pub static EPOCH_GPU_WAITS: AtomicU64 = AtomicU64::new(0);
/// Waits that ended no epoch under front end 2: [passed again on
/// a counter's value, gone past].
pub static QUIET_WAITS: [AtomicU64; 2] = [const { AtomicU64::new(0) }; 2];
pub static MDI_DRAWS: AtomicU64 = AtomicU64::new(0);
pub static MDI_COUNT_FROM_MEMORY: AtomicU64 = AtomicU64::new(0);
/// Indexed indirect draws whose arguments ran past the bound index buffer
/// (`Cp::checked_index_args`): [dropped, drawn as read again].
pub static ARGS_BEYOND: [AtomicU64; 2] = [AtomicU64::new(0), AtomicU64::new(0)];

fn promises_on() -> bool {
    super::tune::promises()
}

pub(super) fn promise(seq: u64, addr: u64, value: u64, bytes: u8, by_cpu: bool, view: u32) {
    if !promises_on() {
        return;
    }
    let mut g = PROMISES.lock();
    let m = g.get_or_insert_with(HashMap::new);
    if m.len() > 1 << 14 {
        let done = super::exec::completed_seq();
        m.retain(|_, p| p.0 > done);
    }
    // (Made where its writer's batch was submitted, or for a CP effect
    // recorded by a command processor holding nothing: every wait before
    // the write had passed and its view took them in.)
    m.insert(addr, (seq, value, bytes, by_cpu, Instant::now(), view));
    drop(g);
    PROMISE_GEN.fetch_add(1, Ordering::Release);
    wake_label(addr, bytes);
}

/// The batch whose promised write to `addr` satisfies the comparison,
/// whether the CPU performs that write, and its writer's view then.
fn promised(addr: u64, wide: bool, func: u32, mask: u64, reference: u64) -> Option<(u64, bool, u32)> {
    let g = PROMISES.lock();
    let &(seq, value, bytes, by_cpu, _, epoch) = g.as_ref()?.get(&addr)?;
    if seq <= super::exec::completed_seq() || (wide && bytes < 8) {
        return None;
    }
    let v = if wide { value } else { value & 0xFFFF_FFFF };
    compare(func, v & mask, reference & mask).then_some((seq, by_cpu, epoch))
}

/// The unfinished batch whose promised write to `addr` is `v` (the value a
/// wait read there: the GPU may have stored it already), for ordered async
/// compute (`Exec::xq_label`).
pub(super) fn promised_writer(addr: u64, v: u64, wide: bool) -> Option<u64> {
    let g = PROMISES.lock();
    let &(seq, value, bytes, _, _, _) = g.as_ref()?.get(&addr)?;
    let same = if wide { bytes >= 8 && value == v } else { value & 0xFFFF_FFFF == v & 0xFFFF_FFFF };
    (same && seq > super::exec::completed_seq()).then_some(seq)
}
const LABEL_LOG_CAP: usize = 1 << 15;

fn label_event(queue: &str, kind: u8, addr: u64, value: u64, reference: u64, packet: u64) {
    let a = addr & !7;
    if kind == b'W' && !WAITED.lock().as_ref().is_some_and(|w| w.contains(&a)) {
        return;
    }
    static T0: std::sync::OnceLock<Instant> = std::sync::OnceLock::new();
    let us = T0.get_or_init(Instant::now).elapsed().as_micros() as u64;
    let mut g = LABEL_LOG.lock();
    if g.len() >= LABEL_LOG_CAP {
        g.pop_front();
    }
    g.push_back(LabelEvent { us, queue: queue.to_string(), kind, addr: a, value, reference, packet, sub: CUR_SUB.with(|c| c.get()) });
}

/// The last `n` events of the label log, oldest first (control `labels <n>`: the queues'
/// label choreography over a few frames, `tools/labelflow.py`). Kinds beyond `LabelEvent`'s:
/// F a flip (value: the title submission), G a wait gone past (a stretch begins), C / D that
/// stretch committed / dropped, and inside held recording i (memory holds), o (its own held
/// write), p (a promise), m (a missed write or counter), x (the stretch stops there).
pub fn label_dump(n: usize) -> String {
    let g = LABEL_LOG.lock();
    let mut out = String::with_capacity(n.min(g.len()) * 96);
    for e in g.iter().skip(g.len().saturating_sub(n)) {
        let writer = last_write(e.addr).map(|w| w.0).unwrap_or_default();
        out.push_str(&format!("{} {} {} {} {:#x} {:#x} {:#x} {:#x} {}
", e.us, e.queue, e.sub, e.kind as char, e.addr, e.value, e.reference, e.packet, if writer.is_empty() { "-" } else { &writer }));
    }
    out
}

/// The recorded history of one label (newest last).
fn label_history(addr: u64, max: usize) -> String {
    let a = addr & !7;
    let g = LABEL_LOG.lock();
    let ev: Vec<&LabelEvent> = g.iter().filter(|e| e.addr == a).collect();
    let mut out = String::new();
    for e in &ev[ev.len().saturating_sub(max)..] {
        let what = match e.kind {
            b'W' => format!("wrote {:#x}", e.value),
            b'I' => format!("wait ref {:#x}: satisfied at once (value {:#x})", e.reference, e.value),
            b'S' => format!("wait ref {:#x}: blocks (value {:#x})", e.reference, e.value),
            b'L' => format!("wait ref {:#x}: satisfied by a CP write it latched (value now {:#x})", e.reference, e.value),
            b'P' => format!("wait ref {:#x}: passed on a promised write (value now {:#x})", e.reference, e.value),
            b'M' => format!("wait ref {:#x}: passed on a write made after its submission (value now {:#x})", e.reference, e.value),
            b'K' => format!("wait ref {:#x}: passed, its counter moved on past it (value now {:#x})", e.reference, e.value),
            _ => format!("wait ref {:#x}: satisfied (value {:#x})", e.reference, e.value),
        };
        out.push_str(&format!("\n    {:>10.3} ms {} {what} (packet {:#x})", e.us as f64 / 1000.0, e.queue, e.packet));
    }
    out
}

/// Records the label-sized words of a block the CP wrote (DMA fills and
/// copies up to 16 KiB; larger blocks are data, not labels).
fn note_block(queue: &str, addr: u64, bytes: u64, at: u64) {
    super::homes::cpu_written(addr, bytes);
    if bytes > 16 << 10 {
        return;
    }
    let mut a = addr & !7;
    while a < addr + bytes {
        let v = if valid(a, 8) { unsafe { std::ptr::read_volatile(a as *const u64) } } else { 0 };
        note_write(queue, a, v, at);
        a += 8;
    }
}

/// Blocking label waits per (waiting queue, queue that last wrote the
/// label): count and nanoseconds (`profile report`).
static WAIT_STATS: parking_lot::Mutex<Option<HashMap<(String, String), (u64, u64)>>> = parking_lot::Mutex::new(None);

pub fn reset_wait_stats() {
    *WAIT_STATS.lock() = None;
    *EPOCH_WAITS.lock() = None;
    *EPOCH_CLASSES.lock() = None;
    *EPOCH_REPEATS.lock() = None;
    *LONG_WAITS.lock() = None;
    LONG_WAITS_LOGGED.store(0, Ordering::Relaxed);
}

/// Waits the CPU released (new copy-cache epochs, DS_PROFILE_GPU): (queue,
/// label, last CP write there) -> count.
static EPOCH_WAITS: parking_lot::Mutex<Option<HashMap<(String, u64, String), u64>>> = parking_lot::Mutex::new(None);
/// The same by (queue, how the wait passed (`Cp::last_wait_kind`), what the
/// label held) -> (count, labels).
static EPOCH_CLASSES: parking_lot::Mutex<Option<HashMap<(String, u8, &'static str), (u64, std::collections::HashSet<u64>)>>> = parking_lot::Mutex::new(None);
/// The same by (queue, label) -> (the value at the last such pass, passes
/// on that same value again, passes on a new one, the comparison).
static EPOCH_REPEATS: parking_lot::Mutex<Option<HashMap<(String, u64), (u64, u64, u64, u32)>>> = parking_lot::Mutex::new(None);

fn epoch_wait_lines() -> String {
    let g = EPOCH_WAITS.lock();
    let Some(m) = g.as_ref() else { return String::new() };
    let mut v: Vec<(&(String, u64, String), &u64)> = m.iter().collect();
    v.sort_by(|a, b| b.1.cmp(a.1));
    let total: u64 = v.iter().map(|x| *x.1).sum();
    let mut s = format!("    waits the CPU released (copy-cache epochs), {total} in all:\n");
    for ((q, a, by), n) in v.iter().take(12) {
        s.push_str(&format!("      {q} label {a:#x} ({by}): {n}\n"));
    }
    drop(g);
    let g = EPOCH_CLASSES.lock();
    if let Some(m) = g.as_ref() {
        let mut v: Vec<_> = m.iter().collect();
        v.sort_by(|a, b| b.1 .0.cmp(&a.1 .0));
        s.push_str("    the same by queue, how the wait passed (I at once, M a write seen since or a counter past it, E after polling, P/L on a GPU write) and what the label held:\n");
        for ((q, k, what), (n, labels)) in v.iter().take(24) {
            s.push_str(&format!("      {q} {} {what}: {n} ({} labels)\n", *k as char, labels.len()));
        }
    }
    drop(g);
    let g = EPOCH_REPEATS.lock();
    if let Some(m) = g.as_ref() {
        let mut v: Vec<_> = m.iter().collect();
        v.sort_by(|a, b| (b.1 .1 + b.1 .2).cmp(&(a.1 .1 + a.1 .2)));
        let (same, new) = m.values().fold((0u64, 0u64), |(s, n), r| (s + r.1, n + r.2));
        s.push_str(&format!("    the same by label: {same} passed on the value the queue's last such pass there saw, {new} on a new one; by label (same / new, comparison):\n"));
        for ((q, a), (_, same, new, func)) in v.iter().take(16) {
            s.push_str(&format!("      {q} label {a:#x}: {same} / {new} (func {func})\n"));
        }
    }
    s
}

/// The graphics command processor's time by packet class (`tune
/// cp_classes`): [ns without waits, packets, ns of waits] for draws and
/// dispatches of own chunks of parallel runs, draws and dispatches of serial
/// runs, CP effects (other packets), state packets, NOPs and private
/// operations, planning parallel runs, collecting their chunks, submissions
/// at buffer ends.
static CP_CLASSES: [[AtomicU64; 3]; 10] = [const { [const { AtomicU64::new(0) }; 3] }; 10];
/// The graphics command processor's serial commands by why they were not in
/// a parallel run (`tune cp_classes`): `SERIAL_WHY_NAMES`.
static SERIAL_WHY: [AtomicU64; 14] = [const { AtomicU64::new(0) }; 14];
/// The same for the compute queue `tune cp_classes_acb` names, and its
/// commands recorded in its own chunks of runs (index 14).
static SERIAL_WHY_ACB: [AtomicU64; 15] = [const { AtomicU64::new(0) }; 15];
/// Census (`tune cp_classes`, the queue `cp_classes_acb` names): its runs as
/// they would be if label writes, label waits and flushes did not end them
/// (`Cp::handshake_census`): stretches scanned, their commands, commands in
/// stretches long enough to split, the ends met (other packets, buffer ends).
static HS_CENSUS: [AtomicU64; 4] = [const { AtomicU64::new(0) }; 4];
/// The graphics CP's serial runs (`tune cp_classes`): [runs planned and left
/// serial, commands executed in them, of those with the epoch unchanged since
/// the run was planned, ns from planning to each command summed].
static RUN_EPOCHS: [AtomicU64; 4] = [const { AtomicU64::new(0) }; 4];
const SERIAL_WHY_NAMES: [&str; 14] = ["small fetched buffers", "streamed", "runs of 1", "runs of 2-7", "runs of 8-31", "runs of 32-127", "runs of 128 or more", "other", "small fetched buffers' count", "past a run", "past a label wait", "predicated", "held otherwise", "no plan made"];
/// FRAMES when the classes were last cleared.
static CP_CLASSES_FROM: AtomicU64 = AtomicU64::new(0);
/// The same per PM4 opcode for the compute queue `tune cp_classes_acb`
/// names: [ns without waits, packets, ns of waits].
static ACB_OPS: [[AtomicU64; 3]; 256] = [const { [const { AtomicU64::new(0) }; 3] }; 256];
/// The graphics command processor's packets recorded past label waits
/// (`tune cp_classes`), per PM4 opcode: [ns without waits, packets, ns of
/// waits].
static STRETCH_OPS: [[AtomicU64; 3]; 256] = [const { [const { AtomicU64::new(0) }; 3] }; 256];

/// The graphics CP's indexed indirect draws by part (`tune cp_classes`):
/// [count, ns] per `IDRAW_PART_NAMES`.
pub static IDRAW_PARTS: [[AtomicU64; 2]; 18] = [const { [const { AtomicU64::new(0) }; 2] }; 18];
const IDRAW_PART_NAMES: [&str; 6] = ["argument sync", "location checks", "repeats", "their index bindings", "indirect drawn in full", "every draw in full"];
const IDRAW_GROUP_NAMES: [&str; 3] = ["the graphics CP outside stretches", "the graphics CP past label waits", "its workers"];

thread_local! {
    /// This thread's `IDRAW_PARTS` group (0 the graphics CP, 1 the same in
    /// a stretch past a label wait, 2 its workers).
    static IDRAW_GROUP: std::cell::Cell<usize> = const { std::cell::Cell::new(0) };
}

/// Adds a part's time since `t` (`IDRAW_PARTS`).
pub fn idraw_part(k: usize, t: Option<Instant>) {
    if let Some(t) = t {
        let k = k + 6 * IDRAW_GROUP.with(|g| g.get());
        IDRAW_PARTS[k][0].fetch_add(1, Ordering::Relaxed);
        IDRAW_PARTS[k][1].fetch_add(t.elapsed().as_nanos() as u64, Ordering::Relaxed);
    }
}

/// Times every draw recorded in full (`IDRAW_PARTS`) when dropped.
struct EveryLap(Option<Instant>);

impl Drop for EveryLap {
    fn drop(&mut self) {
        idraw_part(5, self.0);
    }
}

/// Times a draw recorded in full (`IDRAW_PARTS`) when dropped.
struct FullLap(Option<Instant>);

impl Drop for FullLap {
    fn drop(&mut self) {
        idraw_part(4, self.0);
    }
}

fn stretch_op_end(opcode: u32, t: Option<(Instant, u64)>) {
    let Some((t0, w0)) = t else { return };
    let waited = super::timeline::waited_ns().saturating_sub(w0);
    let ns = (t0.elapsed().as_nanos() as u64).saturating_sub(waited);
    let c = &STRETCH_OPS[(opcode & 0xFF) as usize];
    c[0].fetch_add(ns, Ordering::Relaxed);
    c[1].fetch_add(1, Ordering::Relaxed);
    c[2].fetch_add(waited, Ordering::Relaxed);
}

fn acb_op_end(opcode: u32, t: Option<(Instant, u64)>) {
    let Some((t0, w0)) = t else { return };
    let waited = super::timeline::waited_ns().saturating_sub(w0);
    let ns = (t0.elapsed().as_nanos() as u64).saturating_sub(waited);
    let c = &ACB_OPS[(opcode & 0xFF) as usize];
    c[0].fetch_add(ns, Ordering::Relaxed);
    c[1].fetch_add(1, Ordering::Relaxed);
    c[2].fetch_add(waited, Ordering::Relaxed);
}

fn op_name(opcode: u32) -> String {
    let n = match opcode {
        op::NOP => "NOP",
        op::SET_BASE => "SET_BASE",
        op::DISPATCH_DIRECT => "DISPATCH_DIRECT",
        op::DISPATCH_INDIRECT => "DISPATCH_INDIRECT",
        op::COND_EXEC => "COND_EXEC",
        op::WRITE_DATA => "WRITE_DATA",
        op::WAIT_REG_MEM => "WAIT_REG_MEM",
        op::INDIRECT_BUFFER => "INDIRECT_BUFFER",
        op::COPY_DATA => "COPY_DATA",
        op::EVENT_WRITE => "EVENT_WRITE",
        op::RELEASE_MEM => "RELEASE_MEM",
        op::DMA_DATA => "DMA_DATA",
        op::ACQUIRE_MEM => "ACQUIRE_MEM",
        op::SET_SH_REG => "SET_SH_REG",
        op::SET_SH_REG_INDEX => "SET_SH_REG_INDEX",
        op::SET_UCONFIG_REG => "SET_UCONFIG_REG",
        op::LOAD_SH_REG_INDEX => "LOAD_SH_REG_INDEX",
        op::WAIT_REG_MEM64 => "WAIT_REG_MEM64",
        op::DRAW_INDEX_2 => "DRAW_INDEX_2",
        op::DRAW_INDEX_AUTO => "DRAW_INDEX_AUTO",
        op::DRAW_INDEX_OFFSET_2 => "DRAW_INDEX_OFFSET_2",
        op::DRAW_INDIRECT => "DRAW_INDIRECT",
        op::DRAW_INDEX_INDIRECT => "DRAW_INDEX_INDIRECT",
        op::SET_CONTEXT_REG => "SET_CONTEXT_REG",
        1 => "label submissions after packets",
        _ => return format!("op {opcode:#04x}"),
    };
    n.to_string()
}
/// The planner's time (`tune cp_classes`; ns): [runs left serial, their
/// count, then for runs split: the scan, setup (fast clears, homes,
/// indirect-argument reads), the chunks' start states, the rest of starting
/// the workers, their count; then of starting: getting the workers and the
/// unsubmitted writes, submitting the batch before the run, making the run,
/// waking the workers].
static PLAN_PHASES: [AtomicU64; 11] = [const { AtomicU64::new(0) }; 11];
/// The chunk start states' and the end state's clones inside `PLAN_PHASES[4]`
/// (ns; `tune cp_classes`).
static PLAN_CLONE_NS: AtomicU64 = AtomicU64::new(0);
/// The chunk start states' walk (`tune cp_classes`): [packets walked, indirect draws among
/// them, ns applying those (their arguments read for the user SGPRs they set)].
static PLAN_WALK: [AtomicU64; 3] = [const { AtomicU64::new(0) }; 3];
/// Runs planned while the batch being recorded had writes (`tune
/// cp_classes`): [such runs, runs whose writes the forwarded copies all
/// cover, the written ranges, the ranges' pieces not covered].
static PLAN_MAIN_WRITES: [AtomicU64; 4] = [const { AtomicU64::new(0) }; 4];

/// Lap clock of `par_plan` (`PLAN_PHASES`).
struct PlanLaps(Option<Instant>);

impl PlanLaps {
    fn mark(&mut self, k: usize) {
        if let Some(t) = self.0 {
            let now = Instant::now();
            PLAN_PHASES[k].fetch_add((now - t).as_nanos() as u64, Ordering::Relaxed);
            self.0 = Some(now);
        }
    }
}

const CP_CLASS_NAMES: [&str; 10] = ["own parallel chunks: draws", "own parallel chunks: dispatches", "serial draws", "serial dispatches", "CP effects", "state", "NOPs and private operations", "planning", "collecting", "buffer-end submissions"];

/// A packet class's start (`cp_classes`): the clock and the thread's waits.
fn class_start(on: bool) -> Option<(Instant, u64)> {
    on.then(|| (Instant::now(), super::timeline::waited_ns()))
}

fn class_end(k: usize, t: Option<(Instant, u64)>) {
    let Some((t0, w0)) = t else { return };
    let waited = super::timeline::waited_ns().saturating_sub(w0);
    let ns = (t0.elapsed().as_nanos() as u64).saturating_sub(waited);
    let c = &CP_CLASSES[k];
    c[0].fetch_add(ns, Ordering::Relaxed);
    c[1].fetch_add(1, Ordering::Relaxed);
    c[2].fetch_add(waited, Ordering::Relaxed);
}

/// `DS_OP_CENSUS=1` (diagnosis, 2026-11-16 night): the main graphics command processor's
/// packets outside its own share of runs, by opcode and nesting (top-level buffer, deeper),
/// and those a held recorder refuses (`Cp::held_refuses`) by reason: where a command processor
/// that only walks and plans (its stretches recorded held on the workers) would have to stop.
fn op_census_on() -> bool {
    static W: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *W.get_or_init(|| std::env::var("DS_OP_CENSUS").ok().as_deref() == Some("1"))
}
static OP_CENSUS: [[AtomicU64; 256]; 2] = [const { [const { AtomicU64::new(0) }; 256] }; 2];
static OP_REFUSED: parking_lot::Mutex<Vec<(&'static str, u64)>> = parking_lot::Mutex::new(Vec::new());

fn op_census(opcode: u32, depth: u32, refused: Option<&'static str>) {
    OP_CENSUS[(depth > 1) as usize][(opcode & 0xFF) as usize].fetch_add(1, Ordering::Relaxed);
    if let Some(why) = refused {
        let mut g = OP_REFUSED.lock();
        match g.iter_mut().find(|e| e.0 == why) {
            Some(e) => e.1 += 1,
            None => g.push((why, 1)),
        }
    }
}

/// The census line (`DS_OP_CENSUS=1`; "" otherwise), per flip since the profile's reset.
pub fn op_census_line() -> String {
    if !op_census_on() {
        return String::new();
    }
    let flips = super::FRAMES.load(Ordering::Relaxed).saturating_sub(CP_CLASSES_FROM.load(Ordering::Relaxed)).max(1) as f64;
    let mut parts = Vec::new();
    for (d, name) in [(0usize, "top level"), (1, "nested")] {
        let mut v: Vec<(usize, u64)> = OP_CENSUS[d].iter().enumerate().map(|(k, c)| (k, c.load(Ordering::Relaxed))).filter(|x| x.1 > 0).collect();
        v.sort_by_key(|x| std::cmp::Reverse(x.1));
        let s: Vec<String> = v.iter().map(|&(k, n)| format!("{} {:.1}", op_name(k as u32), n as f64 / flips)).collect();
        parts.push(format!("{name}: {}", s.join(", ")));
    }
    let mut r = OP_REFUSED.lock().clone();
    r.sort_by_key(|x| std::cmp::Reverse(x.1));
    let rs: Vec<String> = r.iter().map(|&(w, n)| format!("{w} {:.1}", n as f64 / flips)).collect();
    format!("    graphics command processor's own packets per flip outside its run shares (`DS_OP_CENSUS`): {}; held recording refuses: {}\n", parts.join("; "), rs.join(", "))
}

/// Clears the packet classes (`profile reset`).
pub fn reset_classes() {
    PLAN_CLONE_NS.store(0, Ordering::Relaxed);
    for c in OP_CENSUS.iter().flatten() {
        c.store(0, Ordering::Relaxed);
    }
    OP_REFUSED.lock().clear();
    for c in CP_CLASSES.iter().flatten().chain(ACB_OPS.iter().flatten()).chain(STRETCH_OPS.iter().flatten()).chain(IDRAW_PARTS.iter().flatten()).chain(SERIAL_WHY.iter()).chain(SERIAL_WHY_ACB.iter()).chain(HS_CENSUS.iter()).chain(PLAN_PHASES.iter()).chain(PLAN_WALK.iter()).chain(PLAN_MAIN_WRITES.iter()).chain(RUN_EPOCHS.iter()).chain(CB_CENSUS.iter().flatten()) {
        c.store(0, Ordering::Relaxed);
    }
    CP_CLASSES_FROM.store(super::FRAMES.load(Ordering::Relaxed), Ordering::Relaxed);
}

/// `DS_CB_CENSUS=1` (diagnosis): every command buffer a command processor
/// runs (as `Cp::run` enters or chains to it) hashed by its contents, with
/// the registers at its start, and with its address too, against those its
/// command processor ran in the last CB_FLIPS flips: how much of a frame's
/// recording repeats a recent frame's (the reach of reusing recordings
/// across frames, before the memory their commands read).
fn cb_census_on() -> bool {
    static W: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *W.get_or_init(|| std::env::var("DS_CB_CENSUS").ok().as_deref() == Some("1"))
}

/// Diagnosis (`DS_CB_SHAPE=<flip>`): the graphics command processor's
/// stream during that flip, logged as `cb shape:` lines (buffers entered
/// and left with their depth, top-level packets, runs gone past and the
/// stops of recording past them; at most 1,500 lines).
fn cb_shape_flip() -> u64 {
    static F: std::sync::OnceLock<u64> = std::sync::OnceLock::new();
    *F.get_or_init(|| std::env::var("DS_CB_SHAPE").ok().and_then(|v| v.trim().parse().ok()).unwrap_or(u64::MAX))
}

static CB_SHAPE_LINES: AtomicU64 = AtomicU64::new(0);

const CB_FLIPS: u64 = 8;

/// Per kind (the graphics queue's command processor, the compute queues'):
/// [buffers, dwords, buffers with contents one of the last CB_FLIPS flips
/// ran, their dwords, those with the same registers at their start too,
/// their dwords, those at the same address as well, their dwords].
static CB_CENSUS: [[AtomicU64; 8]; 2] = [const { [const { AtomicU64::new(0) }; 8] }; 2];

thread_local! {
    /// This command processor's buffers by hash: the last flip each ran in.
    static CB_SEEN: std::cell::RefCell<HashMap<u64, u64>> = std::cell::RefCell::new(HashMap::new());
}

fn hash_words(h: &mut super::fx::FxHasher, w: &[u32]) {
    use std::hash::Hasher;
    let mut it = w.chunks_exact(2);
    for p in &mut it {
        h.write_u64(p[0] as u64 | (p[1] as u64) << 32);
    }
    for &x in it.remainder() {
        h.write_u32(x);
    }
}

/// The command buffers' census (`DS_CB_CENSUS=1`; "" otherwise).
pub fn cb_census_line() -> String {
    if !cb_census_on() {
        return String::new();
    }
    let flips = super::FRAMES.load(Ordering::Relaxed).saturating_sub(CP_CLASSES_FROM.load(Ordering::Relaxed)).max(1) as f64;
    let parts: Vec<String> = ["the graphics queue's", "the compute queues'"]
        .iter()
        .zip(CB_CENSUS.iter())
        .map(|(name, c)| {
            let v: Vec<u64> = c.iter().map(|x| x.load(Ordering::Relaxed)).collect();
            let pct = |a: u64, b: u64| 100.0 * a as f64 / b.max(1) as f64;
            format!(
                "{name} {:.1} buffers a flip ({:.0} dwords): with contents one its CP ran in the last {CB_FLIPS} flips {:.1}% ({:.1}% of the dwords), with its registers at their start too {:.1}% ({:.1}%), at the same address as well {:.1}% ({:.1}%)",
                v[0] as f64 / flips,
                v[1] as f64 / flips,
                pct(v[2], v[0]),
                pct(v[3], v[1]),
                pct(v[4], v[0]),
                pct(v[5], v[1]),
                pct(v[6], v[0]),
                pct(v[7], v[1])
            )
        })
        .collect();
    format!("    command buffers run (`DS_CB_CENSUS`): {}\n", parts.join("; "))
}

/// The packet classes per flip since they were cleared ("" when none).
pub fn classes_line() -> String {
    let flips = super::FRAMES.load(Ordering::Relaxed).saturating_sub(CP_CLASSES_FROM.load(Ordering::Relaxed)).max(1) as f64;
    let total: u64 = CP_CLASSES.iter().map(|c| c[0].load(Ordering::Relaxed)).sum();
    if total == 0 {
        return String::new();
    }
    let parts: Vec<String> = CP_CLASSES
        .iter()
        .zip(CP_CLASS_NAMES)
        .map(|(c, name)| {
            let (ns, n, w) = (c[0].load(Ordering::Relaxed), c[1].load(Ordering::Relaxed), c[2].load(Ordering::Relaxed));
            format!("{name} {:.2} ms ({:.0} packets, {:.1} us each; waits {:.2} ms)", ns as f64 / flips / 1e6, n as f64 / flips, ns as f64 / n.max(1) as f64 / 1e3, w as f64 / flips / 1e6)
        })
        .collect();
    let mut s = format!("    graphics command processor per flip by packet class (waits apart; {:.2} ms in all): {}\n", total as f64 / flips / 1e6, parts.join("; "));
    let why: Vec<String> = SERIAL_WHY.iter().zip(SERIAL_WHY_NAMES).map(|(c, name)| format!("{name} {:.1}", c.load(Ordering::Relaxed) as f64 / flips)).collect();
    s.push_str(&format!("    its serial commands per flip by why they were not in a parallel run: {}\n", why.join(", ")));
    let re: Vec<u64> = RUN_EPOCHS.iter().map(|c| c.load(Ordering::Relaxed)).collect();
    s.push_str(&format!(
        "    its serial runs planned: {:.1} a flip, {:.1} commands a flip run in them, {:.1}% of those at the epoch their run was planned in, {:.0} us after planning on average\n",
        re[0] as f64 / flips,
        re[1] as f64 / flips,
        100.0 * re[2] as f64 / re[1].max(1) as f64,
        re[3] as f64 / re[1].max(1) as f64 / 1e3
    ));
    let ph: Vec<u64> = PLAN_PHASES.iter().map(|c| c.load(Ordering::Relaxed)).collect();
    let us = |ns: u64, n: u64| ns as f64 / n.max(1) as f64 / 1e3;
    s.push_str(&format!(
        "    planning per flip: {:.1} runs left serial at {:.1} us each; {:.1} runs split at {:.1} us each (scan {:.1}, setup {:.1}, chunk start states {:.1} (clones {:.1}; {:.0} packets walked a run, {:.0} indirect draws among them taking {:.1} us; register files' high-water marks {}/{}/{} dwords), starting workers {:.1}: getting them and the unsubmitted writes {:.1}, submitting the batch before the run {:.1}, making the run {:.1}, waking them {:.1}, the rest {:.1})\n",
        ph[1] as f64 / flips,
        us(ph[0], ph[1]),
        ph[6] as f64 / flips,
        us(ph[2] + ph[3] + ph[4] + ph[5] + ph[7] + ph[8] + ph[9] + ph[10], ph[6]),
        us(ph[2], ph[6]),
        us(ph[3], ph[6]),
        us(ph[4], ph[6]),
        us(PLAN_CLONE_NS.load(Ordering::Relaxed), ph[6]),
        PLAN_WALK[0].load(Ordering::Relaxed) as f64 / ph[6].max(1) as f64,
        PLAN_WALK[1].load(Ordering::Relaxed) as f64 / ph[6].max(1) as f64,
        us(PLAN_WALK[2].load(Ordering::Relaxed), ph[6]),
        high(Space::Ctx, CTX_DWORDS),
        high(Space::Sh, SH_DWORDS),
        high(Space::Uc, UC_DWORDS),
        us(ph[5] + ph[7] + ph[8] + ph[9] + ph[10], ph[6]),
        us(ph[7], ph[6]),
        us(ph[8], ph[6]),
        us(ph[9], ph[6]),
        us(ph[10], ph[6]),
        us(ph[5], ph[6])
    ));
    let mw: Vec<u64> = PLAN_MAIN_WRITES.iter().map(|c| c.load(Ordering::Relaxed)).collect();
    s.push_str(&format!(
        "    runs planned with the batch's writes unsubmitted: {:.1} per flip, {:.1} of them all covered by forwarded copies ({} written ranges, {} pieces not covered)\n",
        mw[0] as f64 / flips,
        mw[1] as f64 / flips,
        mw[2],
        mw[3]
    ));
    let mut ops: Vec<(usize, u64, u64, u64)> = ACB_OPS
        .iter()
        .enumerate()
        .map(|(k, c)| (k, c[0].load(Ordering::Relaxed), c[1].load(Ordering::Relaxed), c[2].load(Ordering::Relaxed)))
        .filter(|&(_, ns, _, w)| ns + w > 0)
        .collect();
    if !ops.is_empty() {
        let all: u64 = ops.iter().map(|o| o.1).sum();
        ops.sort_by_key(|o| std::cmp::Reverse(o.1));
        let parts: Vec<String> = ops
            .iter()
            .map(|&(k, ns, n, w)| format!("{} {:.2} ms ({:.0} packets, {:.1} us each; waits {:.2} ms)", op_name(k as u32), ns as f64 / flips / 1e6, n as f64 / flips, ns as f64 / n.max(1) as f64 / 1e3, w as f64 / flips / 1e6))
            .collect();
        s.push_str(&format!("    compute queue {:#x} per flip by packet (waits apart; {:.2} ms in all): {}\n", super::tune::cp_classes_acb(), all as f64 / flips / 1e6, parts.join("; ")));
        let why: Vec<String> = SERIAL_WHY_ACB.iter().zip(SERIAL_WHY_NAMES.iter().copied().chain(["in its own chunks of runs"])).map(|(c, name)| format!("{name} {:.1}", c.load(Ordering::Relaxed) as f64 / flips)).collect();
        s.push_str(&format!("    its commands per flip by why they were not in a parallel run: {}\n", why.join(", ")));
        let h: Vec<u64> = HS_CENSUS.iter().map(|c| c.load(Ordering::Relaxed)).collect();
        s.push_str(&format!("    its stretches through label writes, label waits and flushes (census): {:.1} a flip, {:.1} commands a flip in them, {:.1} of those in stretches long enough to split\n", h[0] as f64 / flips, h[1] as f64 / flips, h[2] as f64 / flips));
    }
    let mut ops: Vec<(usize, u64, u64, u64)> = STRETCH_OPS
        .iter()
        .enumerate()
        .map(|(k, c)| (k, c[0].load(Ordering::Relaxed), c[1].load(Ordering::Relaxed), c[2].load(Ordering::Relaxed)))
        .filter(|&(_, ns, _, w)| ns + w > 0)
        .collect();
    if !ops.is_empty() {
        let all: u64 = ops.iter().map(|o| o.1).sum();
        ops.sort_by_key(|o| std::cmp::Reverse(o.1));
        let parts: Vec<String> = ops
            .iter()
            .take(14)
            .map(|&(k, ns, n, w)| format!("{} {:.2} ms ({:.0} packets, {:.1} us each; waits {:.2} ms)", op_name(k as u32), ns as f64 / flips / 1e6, n as f64 / flips, ns as f64 / n.max(1) as f64 / 1e3, w as f64 / flips / 1e6))
            .collect();
        s.push_str(&format!("    the graphics command processor past label waits per flip by packet (waits apart; {:.2} ms in all): {}\n", all as f64 / flips / 1e6, parts.join("; ")));
    }
    for (g, gname) in IDRAW_GROUP_NAMES.iter().enumerate() {
        let id: Vec<String> = IDRAW_PARTS[6 * g..6 * g + 6]
            .iter()
            .zip(IDRAW_PART_NAMES)
            .map(|(c, name)| {
                let (n, ns) = (c[0].load(Ordering::Relaxed), c[1].load(Ordering::Relaxed));
                format!("{name} {:.0} a flip at {:.2} us ({:.2} ms)", n as f64 / flips, ns as f64 / n.max(1) as f64 / 1e3, ns as f64 / flips / 1e6)
            })
            .collect();
        s.push_str(&format!("    draws of {gname} by part: {}\n", id.join("; ")));
    }
    s
}

pub fn wait_stats() -> String {
    let mut s = epoch_wait_lines();
    s.push_str(&wait_stats_inner());
    s
}

/// Blocking label waits by waiter and writer, longest total first.
fn wait_stats_inner() -> String {
    let g = WAIT_STATS.lock();
    let Some(m) = g.as_ref() else { return String::new() };
    let mut v: Vec<(&(String, String), &(u64, u64))> = m.iter().collect();
    v.sort_by(|a, b| b.1 .1.cmp(&a.1 .1));
    let mut s = format!("    label waits (waiter <- last writer: count, total ms, mean us; {} sleeps on wakes, `label_wake`):\n", WAKE_SLEEPS.load(Ordering::Relaxed));
    for ((w, by), (n, ns)) in v.iter().take(24) {
        s.push_str(&format!("      {w} <- {by}: {n}, {} ms, {} us\n", ns / 1_000_000, ns / 1000 / (*n).max(1)));
    }
    s
}

/// Blocking label waits of 1 ms or more by (waiter, last writer, label):
/// [count, ns, ended in memory with no promise for the label, ended in
/// memory with a promise for it (of a value that did not satisfy the wait or
/// made after the wait ended), passed on a promise, latched]; the first 40
/// are logged in detail (`long_wait`).
static LONG_WAITS: parking_lot::Mutex<Option<HashMap<(String, String, u64), [u64; 7]>>> = parking_lot::Mutex::new(None);
static LONG_WAITS_LOGGED: AtomicU64 = AtomicU64::new(0);

/// A blocking wait on `a` that lasted `ns` and ended as `kind` (b'E': the
/// value in memory, b'P': a promise, b'L': latched).
#[allow(clippy::too_many_arguments)]
fn long_wait(waiter: &str, a: u64, wide: bool, func: u32, mask: u64, reference: u64, at: u64, t0: Instant, kind: u8) {
    let ns = t0.elapsed().as_nanos() as u64;
    if ns < 1_000_000 {
        return;
    }
    let (by, wv, wpk) = last_write(a).unwrap_or_else(|| ("cpu".into(), 0, 0));
    if waiter == "gpu-gfx" {
        super::hitch::note_since(super::hitch::Kind::CpWait, t0, || format!("label {a:#x} written by {by}"));
    }
    let p = PROMISES.lock().as_ref().and_then(|m| m.get(&a).copied());
    let slot = match (kind, p) {
        (b'P', _) => 4,
        (b'L', _) => 5,
        (b'K', _) => 6,
        (_, None) => 2,
        _ => 3,
    };
    {
        let mut g = LONG_WAITS.lock();
        let e = g.get_or_insert_with(HashMap::new).entry((waiter.to_string(), by.clone(), a)).or_default();
        e[0] += 1;
        e[1] += ns;
        e[slot] += 1;
    }
    // DS_TRACE_WAITS=<queue>: that queue's long waits (the first 200), with
    // the same detail, whenever they happen; `tune trace_waits <n>`: the
    // graphics CP's next n.
    static TRACE_Q: std::sync::OnceLock<Option<String>> = std::sync::OnceLock::new();
    static TRACED_Q: AtomicU64 = AtomicU64::new(0);
    let traced = (TRACE_Q.get_or_init(|| std::env::var("DS_TRACE_WAITS").ok()).as_deref() == Some(waiter) && TRACED_Q.fetch_add(1, Ordering::Relaxed) < 200) || (waiter == "gpu-gfx" && super::tune::take_trace_waits());
    if traced || LONG_WAITS_LOGGED.fetch_add(1, Ordering::Relaxed) < 40 {
        let promise = match p {
            None => "none".to_string(),
            Some((seq, v, bytes, by_cpu, t, _)) => {
                let since = t.checked_duration_since(t0).map(|d| format!("{:.2} ms into the wait", d.as_secs_f64() * 1e3)).unwrap_or_else(|| format!("{:.2} ms before the wait", t0.duration_since(t).as_secs_f64() * 1e3));
                let vv = if wide { v } else { v & 0xFFFF_FFFF };
                format!("batch {seq} ({}; completed through {}), value {v:#x} ({bytes} bytes{}), made {since}, satisfies: {}", if by_cpu { "CPU after it" } else { "GPU in it" }, super::exec::completed_seq(), "", !(wide && bytes < 8) && compare(func, vv & mask, reference & mask))
            }
        };
        let header = if wpk != 0 && valid(wpk, 4) { unsafe { rd32(wpk) } } else { 0 };
        let op = if header >> 30 == 3 { format!("{:#04x}", (header >> 8) & 0xFF) } else { "?".into() };
        crate::info!(
            "gpu[{waiter}]: label wait {:.2} ms at {at:#x} on [{a:#x}] func {func} ref {reference:#x} mask {mask:#x} ended {} (value now {:#x}); last write {wv:#x} by {by} (packet {wpk:#x}, opcode {op}); promise: {promise}; label history:{}",
            ns as f64 / 1e6,
            match kind {
                b'P' => "on a promise",
                b'L' => "latched",
                b'K' => "as its counter moved on past it",
                _ => "in memory",
            },
            if wide { unsafe { rd64(a) } } else { (unsafe { rd32(a) }) as u64 },
            label_history(a, 10)
        );
    }
}

/// Long label waits by waiter, writer and label (profile report).
pub fn long_waits_line() -> String {
    let g = LONG_WAITS.lock();
    let Some(m) = g.as_ref() else { return String::new() };
    let mut v: Vec<_> = m.iter().collect();
    v.sort_by_key(|(_, e)| std::cmp::Reverse(e[1]));
    let parts: Vec<String> = v
        .iter()
        .take(10)
        .map(|((w, by, a), e)| format!("{w} <- {by} [{a:#x}]: {} waits {:.1} ms (in memory without a promise {}, with one {}, on a promise {}, latched {}, its counter moved on {})", e[0], e[1] as f64 / 1e6, e[2], e[3], e[4], e[5], e[6]))
        .collect();
    format!("    label waits of 1 ms or more: {}\n", parts.join("; "))
}

/// The last CP write to the label-sized word at `addr` (wait traces).
pub fn last_write_info(addr: u64) -> Option<String> {
    last_write(addr).map(|(q, v, pk)| format!("{q} wrote {v:#x} (packet {pk:#x})"))
}

fn last_write(addr: u64) -> Option<(String, u64, u64)> {
    LAST_WRITE.lock().as_ref()?.get(&(addr & !7)).map(|(q, v, pk, _)| (q.clone(), *v, *pk))
}

/// `last_write` with the exact address written.
fn last_write_at(addr: u64) -> Option<(String, u64, u64, u64)> {
    LAST_WRITE.lock().as_ref()?.get(&(addr & !7)).cloned()
}

/// Whether a CP write to exactly `addr` recorded for the label (`LABEL_WRITES`)
/// had value `v` (the low half for 4-byte labels).
fn label_wrote(addr: u64, wide: bool, v: u64) -> bool {
    let g = LABEL_WRITES.lock();
    let Some(h) = g.as_ref().and_then(|m| m.get(&addr)) else { return false };
    h.iter().any(|&(_, w)| if wide { w == v } else { w as u32 == v as u32 })
}

/// What the queue thread is doing (diagnostics).
pub static STATE: parking_lot::Mutex<Vec<(String, String)>> = parking_lot::Mutex::new(Vec::new());

/// A command processor's state while it records past a run or a label
/// wait into held batches (`Cp::held`, `Cp::past_start`).
pub struct HeldCp {
    /// Why and where (packet address) it stopped.
    pub stop: Option<(&'static str, u64)>,
    /// Draws and dispatches recorded.
    pub commands: u64,
}

fn set_state(name: &str, s: String) {
    let mut g = STATE.lock();
    match g.iter_mut().find(|(n, _)| n == name) {
        Some(e) => e.1 = s,
        None => g.push((name.to_string(), s)),
    }
}

impl Cp {
    pub fn new(name: &str, compute: bool) -> Cp {
        Cp {
            name: name.to_string(),
            last_draw: std::borrow::Cow::Borrowed(""),
            args_checked: None,
            repeat_ok: false,
            ctx_dirty: true,
            ps_dirty: true,
            ps_epoch: 0,
            last_draw_progs: (0, 0, 0),
            same_run: 0,
            compute,
            regs: Regs::new(),
            index_base: 0,
            index_count: 0,
            num_instances: 1,
            draw_indirect_base: 0,
            dispatch_indirect_base: 0,
            markers: Vec::new(),
            ctx_stack: Vec::new(),
            cur_packet: 0,
            cur_opcode: 0,
            waiting_on: 0,
            wait_flag: None,
            unknown_seen: HashSet::new(),
            notes_seen: HashSet::new(),
            depth: 0,
            exec: None,
            exec_failed: false,
            fe2_view: (super::fe2::wanted() && super::fe2::views_on() && (compute || super::fe2::gfx_views())).then(|| super::fe2::view_cell(name)),
            fence_pending: false,
            fence_kind: 0,
            pred: None,
            submitted_at: None,
            sub_index: 0,
            rewind_sub: None,
            more_follows: None,
            next_in_submission: false,
            ahead: std::collections::VecDeque::new(),
            cur_ahead: std::cell::RefCell::new([None, None]),
            scan_regs: std::cell::RefCell::new((Vec::new(), Vec::new())),
            compile_ctx: std::cell::Cell::new(None),
            wait_ctx: None,
            par: None,
            par_skip: (0, 0),
            par_skip_n: 0,
            pre_plan: None,
            run_watch: None,
            par_pool: super::par::Pool::new(),
            chunk_costs: None,
            chunk_waits: std::sync::Arc::from(Vec::new()),
            hs_until: (0, 0),
            run_costs: HashMap::new(),
            run_costs_pruned: u64::MAX,
            last_wait_gpu: false,
            last_wait_kind: b'-',
            last_wait_func: 0,
            passed: HashMap::new(),
            last_wait_epoch: 0,
            held: None,
            main_gfx: name == "gpu-gfx",
            past: None,
            past_wait: None,
            settling: false,
            held_polls: Vec::new(),
            cur_buf: 0,
            rewind: None,
        }
    }

    /// Where to run from next (buffer index and address) after a stretch
    /// recorded past a run was dropped (`past_settle`).
    pub fn take_rewind(&mut self) -> Option<(usize, u64)> {
        self.rewind.take()
    }

    /// Where to run from after a stretch past a label wait that began in an
    /// earlier submission was dropped: (submission, buffer, address).
    pub fn take_rewind_sub(&mut self) -> Option<(u64, usize, u64)> {
        self.rewind_sub.take()
    }

    /// Whether a rewind into an earlier submission is pending (the queue's
    /// worker stops running the current one).
    pub fn rewind_sub_pending(&self) -> bool {
        self.rewind_sub.is_some()
    }

    /// Whether recording past a run goes on at the end of the command buffer
    /// being run into the submission's next one instead of stopping there,
    /// and past a label wait also into the next title submission when one is
    /// queued. (Past a run into the next submission too, `tune past_subs`,
    /// lost 2.8% in the old front end, FINDINGS.md 2026-10-23: longer
    /// stretches held work back from the GPU; removed on 2026-11-01.)
    fn past_goes_on(&self) -> bool {
        let more = || self.more_follows.as_ref().is_some_and(|f| f());
        self.depth == 1 && (self.past.as_ref().is_some_and(|p| p.across && (self.next_in_submission || (super::tune::past_subs() && more()))) || (self.past_wait.is_some() && (self.next_in_submission || more())))
    }

    /// Whether what this command processor records now may be recorded
    /// again from an earlier title submission (a stretch past a label
    /// wait, or past a run with `tune past_subs`): the queue's worker keeps
    /// the submissions it runs through meanwhile (`take_rewind_sub`).
    pub fn holds_across(&self) -> bool {
        self.past_wait.is_some() || (super::tune::past_subs() && self.past.is_some())
    }

    pub fn chunk_state(&self) -> ChunkState {
        let (c, s, u) = if super::tune::lean_regs() { (high(Space::Ctx, CTX_DWORDS), high(Space::Sh, SH_DWORDS), high(Space::Uc, UC_DWORDS)) } else { (CTX_DWORDS, SH_DWORDS, UC_DWORDS) };
        ChunkState {
            regs: Regs { ctx: self.regs.ctx[..c].to_vec(), sh: self.regs.sh[..s].to_vec(), uc: self.regs.uc[..u].to_vec() },
            index_base: self.index_base,
            index_count: self.index_count,
            num_instances: self.num_instances,
            draw_indirect_base: self.draw_indirect_base,
            dispatch_indirect_base: self.dispatch_indirect_base,
        }
    }

    pub fn set_chunk_state(&mut self, s: &ChunkState) {
        self.repeat_ok = false;
        self.ctx_dirty = true;
        self.ps_dirty = true;
        copy_prefix(&mut self.regs.ctx, &s.regs.ctx, high(Space::Ctx, CTX_DWORDS));
        copy_prefix(&mut self.regs.sh, &s.regs.sh, high(Space::Sh, SH_DWORDS));
        copy_prefix(&mut self.regs.uc, &s.regs.uc, high(Space::Uc, UC_DWORDS));
        self.index_base = s.index_base;
        self.index_count = s.index_count;
        self.num_instances = s.num_instances;
        self.draw_indirect_base = s.draw_indirect_base;
        self.dispatch_indirect_base = s.dispatch_indirect_base;
    }

    /// Gives this (worker) command processor a deferred executor.
    pub fn make_worker_exec(&mut self) -> Result<(), String> {
        let mut e = super::exec::Exec::new(&self.name)?;
        e.set_deferred();
        e.set_par_worker();
        self.exec = Some(Box::new(e));
        Ok(())
    }

    /// Whether the shape trace logs now (`DS_CB_SHAPE`).
    fn shape(&self) -> bool {
        self.main_gfx && super::FRAMES.load(Ordering::Relaxed) == cb_shape_flip() && CB_SHAPE_LINES.fetch_add(1, Ordering::Relaxed) < 1500
    }

    fn held_tag(&self) -> &'static str {
        if self.past.is_some() {
            " (past a run)"
        } else if self.past_wait.is_some() {
            " (past a label wait)"
        } else {
            ""
        }
    }

    /// Stops recording ahead at the packet at `at` (it and what follows are
    /// left to the main command processor).
    fn held_stop(&mut self, why: &'static str, at: u64) {
        if self.shape() {
            crate::info!("cb shape: stop {why} at {at:#x}, depth {}{}", self.depth, self.held_tag());
        }
        if let Some(e) = self.exec.as_mut() {
            e.stop_held(why);
        }
        if let Some(h) = self.held.as_mut() {
            if h.stop.is_none() {
                h.stop = Some((why, at));
            }
        }
    }

    /// Whether recording into held batches stops after the packet at `cur`:
    /// it stopped in it (at `cur`).
    fn held_halted(&mut self, cur: u64) -> bool {
        let stopped = self.exec.as_ref().filter(|e| e.stopped).map(|e| e.stop_why);
        let Some(h) = self.held.as_mut() else { return false };
        if h.stop.is_some() {
            return true;
        }
        if let Some(why) = stopped {
            h.stop = Some((if why.is_empty() { "stopped" } else { why }, cur));
            return true;
        }
        false
    }

    /// Packets recording into held batches does not execute: what it cannot
    /// do before the work it comes after is done (a flip, a display buffer
    /// still shown, CP effects other than label writes, branches on memory).
    fn held_refuses(&self, opcode: u32, n: u64, p: &dyn Fn(u64) -> u32) -> Option<&'static str> {
        match opcode {
            op::NOP if n >= 2 && p(0) == DS_MAGIC => match p(1) {
                pop::FLIP => Some("next flip"),
                pop::WAIT_SAFE_FOR_RENDERING if n >= 4 => crate::hle::libs::videoout::buffer_busy(p(2) as i32, p(3) as i32).then_some("display buffer wait"),
                pop::LOD_STATS => Some("LOD statistics"),
                pop::BRANCH => Some("branch"),
                _ => None,
            },
            op::DMA_DATA => Some("DMA_DATA"),
            op::COPY_DATA => Some("COPY_DATA"),
            op::REWIND => Some("REWIND"),
            op::COND_EXEC => Some("COND_EXEC"),
            op::INDIRECT_BUFFER if self.past.is_some() || self.past_wait.is_some() => Some("indirect buffer"),
            _ => None,
        }
    }

    /// WAIT_REG_MEM recorded into held batches (`Cp::past_start`): a wait
    /// passes on its own label writes, else on memory or another queue's
    /// promise; otherwise the stretch stops there.
    fn held_wait(&mut self, control: u32, a: u64, wide: bool, reference: u64, mask: u64) -> Flow {
        let func = control & 7;
        if (control >> 4) & 3 != 1 || !valid(a, 8) {
            return Flow::Next;
        }
        // Past a label wait, a wait for the same value passes when that one
        // does (the stretch settles on it).
        if self.past_wait.as_deref().is_some_and(|w| w.a == a && w.wide == wide && w.control & 7 == func && w.reference == reference && w.mask == mask) {
            return Flow::Next;
        }
        let pass = |v: u64| compare(func, v & mask, reference & mask);
        let read = move || if wide { unsafe { rd64(a) } } else { (unsafe { rd32(a) }) as u64 };
        let at = self.cur_packet;
        // Its own write comes last.
        if let Some((v, bytes)) = self.exec.as_ref().and_then(|e| e.open_promise(a)) {
            let v = if wide { v } else { v & 0xFFFF_FFFF };
            if !(wide && bytes < 8) && pass(v) {
                if let Some(e) = self.exec.as_mut() {
                    e.order_after_open_writes();
                }
                label_event(&self.name, b'o', a, v, reference, at);
                return Flow::Next;
            }
            label_event(&self.name, b'x', a, v, reference, at);
            self.held_stop("label wait (its own value)", at);
            return Flow::Next;
        }
        if pass(read()) {
            if let Some(e) = self.exec.as_mut() {
                e.note_held_reads(&[(a, if wide { 8 } else { 4 })], 3);
            }
            if self.past_wait.is_some() {
                self.held_polls.push((a, wide, func, mask, reference));
            }
            label_event(&self.name, b'i', a, read(), reference, at);
            return Flow::Next;
        }
        if let Some((seq, by_cpu, _)) = promised(a, wide, func, mask, reference) {
            if let Some(e) = self.exec.as_mut() {
                e.depend_on(seq, by_cpu);
            }
            label_event(&self.name, b'p', a, read(), reference, at);
            return Flow::Next;
        }
        if self.submitted_at.is_some_and(|since| missed_write(a, wide, func, mask, reference, since)) || counter_passed(a, wide, func, mask, reference) {
            label_event(&self.name, b'm', a, read(), reference, at);
            return Flow::Next;
        }
        label_event(&self.name, b'x', a, read(), reference, at);
        self.held_stop("label wait", at);
        Flow::Next
    }

    pub fn chunk_start(&mut self) {
        if let Some(e) = self.exec.as_mut() {
            e.stopped = false;
            e.stop_why = "";
        }
    }

    /// Closes the batch a chunk recorded into.
    pub fn chunk_finish(&mut self) {
        if let Some(e) = self.exec.as_mut() {
            e.close();
        }
    }

    /// Executes the packets in [from, to) of a run (a parallel worker):
    /// returns the address of the packet where the executor stopped (it
    /// and the rest are left to the main command processor), or None.
    pub fn run_chunk(&mut self, from: u64, to: u64) -> Option<u64> {
        let f = Fetched::new(from, 0);
        let mut cur = from;
        let mut commands = 0u64;
        let mut last = unsafe { core::arch::x86_64::_rdtsc() };
        while cur < to {
            if super::shutting_down() || !valid(cur, 4) {
                return Some(cur);
            }
            let h = f.rd(cur);
            match h >> 30 {
                3 => {}
                2 => {
                    cur += 4;
                    continue;
                }
                _ => return Some(cur),
            }
            let count = (h >> 16) & 0x3FFF;
            let opcode = (h >> 8) & 0xFF;
            if count == 0x3FFF {
                cur += 4;
                continue;
            }
            let n = count as u64 + 1;
            let next = cur + 4 * (n + 1);
            let kept = if self.compute { matches!(opcode, op::WRITE_DATA | op::WAIT_REG_MEM | op::WAIT_REG_MEM64) } else { matches!(opcode, op::RELEASE_MEM | op::WAIT_REG_MEM | op::WAIT_REG_MEM64) };
            // (Multi-draws: in a run only where it holds them, `run_multi`.)
            let multi = matches!(opcode, op::DRAW_INDEX_INDIRECT_MULTI | op::DRAW_INDIRECT_MULTI) && !self.compute;
            if next > to || !valid(cur, 4 * (n + 1)) || !(is_command(opcode) || is_state(opcode) || opcode == op::NOP || par_event(opcode, self.compute) || kept || multi) {
                return Some(cur);
            }
            PACKETS.fetch_add(1, Ordering::Relaxed);
            self.cur_packet = cur;
            if super::exec::replay_census_on() {
                super::exec::CUR_PACKET.with(|c| c.set(cur));
            }
            self.cur_opcode = opcode;
            let p = |i: u64| f.rd(cur + 4 * (1 + i));
            // A timestamp or label wait the run holds (`run_pass`); else it
            // and the rest go to the main command processor.
            if kept {
                let ok = if opcode == op::WRITE_DATA { self.chunk_write_data(n, &p) } else { self.chunk_pass(opcode, cur, &p) };
                if !ok {
                    super::par::COMMANDS.fetch_add(commands, Ordering::Relaxed);
                    self.drop_ahead();
                    return Some(cur);
                }
                cur = next;
                continue;
            }
            // Stages prepared ahead within the chunk, as in `run`.
            let command = is_command(opcode) || multi;
            if is_command(opcode) && !self.ahead.is_empty() {
                if self.ahead.front().is_some_and(|c| c.at == cur) {
                    let c = self.ahead.pop_front().unwrap();
                    *self.cur_ahead.borrow_mut() = c.stages;
                } else {
                    self.drop_ahead();
                }
            }
            // A dispatch whose pipeline turns out missing scans the chunk's
            // commands after it for more first uses, as `run` does: without
            // it a chunk compiled its new programs one after another (a 5-3
            // first visit froze 11.1 s, ~15 particle programs of ~0.7 s on
            // one of queue 0x50's workers, 2026-11-01 night).
            if command && ((matches!(opcode, op::DISPATCH_DIRECT | op::DISPATCH_INDIRECT) && super::compile::at_first_uses()) || (matches!(opcode, op::DRAW_INDEX_2 | op::DRAW_INDEX_AUTO | op::DRAW_INDEX_OFFSET_2 | op::DRAW_INDIRECT | op::DRAW_INDEX_INDIRECT | op::DRAW_INDEX_INDIRECT_MULTI | op::DRAW_INDIRECT_MULTI) && super::compile::gfx_ahead())) {
                self.compile_ctx.set(Some((f.base, next, to)));
            }
            let flow = self.packet(opcode, h, cur, n, &p);
            self.compile_ctx.set(None);
            if command {
                for s in self.take_ahead().iter().flatten() {
                    s.cancel();
                }
            }
            if self.exec.as_ref().is_some_and(|e| e.stopped) {
                super::par::COMMANDS.fetch_add(commands, Ordering::Relaxed);
                self.drop_ahead();
                return Some(cur);
            }
            if !matches!(flow, Flow::Next) {
                self.drop_ahead();
                return Some(next);
            }
            commands += command as u64;
            if command {
                if let Some(v) = self.chunk_costs.as_mut() {
                    let now = unsafe { core::arch::x86_64::_rdtsc() };
                    v.push(now.wrapping_sub(last).min(u32::MAX as u64) as u32);
                    last = now;
                }
            }
            cur = next;
        }
        super::par::COMMANDS.fetch_add(commands, Ordering::Relaxed);
        self.drop_ahead();
        None
    }

    /// A packet the graphics queue's runs hold although it is neither state
    /// nor a command (this queue's main command processor, planning; since
    /// 2026-11-07 evening: the vista +3.7%): `RUN_PASS_TS` the title's GPU
    /// timestamp (RELEASE_MEM of the GPU clock without an interrupt: written
    /// by the planner, `cpu_timestamp`), `RUN_PASS_WAIT` a
    /// label wait that holds in guest memory now (the work that wrote the
    /// label is done; the planner gives the run the view the wait would, a
    /// worker orders its batch after the label's writer, `Exec::xq_label`).
    /// Neither orders the run's later commands after its earlier ones on the
    /// hardware, so no dependency inside the run comes with them. A census of
    /// a vista frame: these ended a third of its serial runs.
    fn run_pass(&mut self, opc: u32, p: &dyn Fn(u64) -> u32) -> Option<u8> {
        // (A run's timestamp is written now, by its planner, in stream
        // order: its values stay monotonic; its chunk passes it.)
        if opc == op::RELEASE_MEM && !self.compute && self.main_gfx {
            let c = p(1);
            if (c >> 29) & 7 == 3 && matches!((c >> 24) & 7, 0 | 3) {
                cpu_timestamp(addr64(p(2) & !3, p(3)));
            }
        }
        if matches!(opc, op::WAIT_REG_MEM | op::WAIT_REG_MEM64) && !self.compute && self.main_gfx {
            // (Read as `wait` reads it.)
            let a = addr64(p(1), p(2));
            if valid(a, 8) {
                self.before_fetch(a, 8);
            }
        }
        self.run_pass_here(opc, p)
    }

    /// Whether a graphics run holds a multi-draw packet (since 2026-11-07
    /// evening: the vista +6.4%, Doran +0.6%): a chunk's worker reads its
    /// count and records as the main command processor would; the planner
    /// waits for GPU work still writing them first, as for indirect draws'
    /// arguments. Inside a synchronization-free run no earlier command can
    /// have written them (the hardware's fetch needs a sync packet after
    /// such writes).
    fn run_multi(&self, opc: u32) -> bool {
        !self.compute && self.main_gfx && matches!(opc, op::DRAW_INDEX_INDIRECT_MULTI | op::DRAW_INDIRECT_MULTI)
    }

    /// `run_pass` without preparing the label's read (the scan ahead).
    fn run_pass_here(&self, opc: u32, p: &dyn Fn(u64) -> u32) -> Option<u8> {
        if self.compute || !self.main_gfx {
            return None;
        }
        match opc {
            op::RELEASE_MEM => {
                let c = p(1);
                ((c >> 29) & 7 == 3 && matches!((c >> 24) & 7, 0 | 3)).then_some(RUN_PASS_TS)
            }
            op::WAIT_REG_MEM | op::WAIT_REG_MEM64 => {
                let (c, a, wide) = (p(0), addr64(p(1), p(2)), opc == op::WAIT_REG_MEM64);
                if (c >> 4) & 3 != 1 || !valid(a, 8) {
                    return None;
                }
                let (reference, mask) = if wide { (addr64(p(3), p(4)), addr64(p(5), p(6))) } else { (p(3) as u64, p(4) as u64) };
                let v = if wide { unsafe { rd64(a) } } else { (unsafe { rd32(a) }) as u64 };
                compare(c & 7, v & mask, reference & mask).then_some(RUN_PASS_WAIT)
            }
            _ => None,
        }
    }

    /// A parallel worker meets a packet `run_pass` let into its run: records
    /// a timestamp in its batch, or orders its batch after a passed label's
    /// writer. False: the chunk stops there (the main command processor goes
    /// on from it).
    fn chunk_pass(&mut self, opcode: u32, at: u64, p: &dyn Fn(u64) -> u32) -> bool {
        if opcode == op::RELEASE_MEM {
            let c = p(1);
            if (c >> 29) & 7 != 3 || !matches!((c >> 24) & 7, 0 | 3) {
                return false;
            }
            // (Its planner wrote it: `run_pass`.)
            let Some(e) = self.exec.as_mut() else { return false };
            e.note_sync();
            self.fence_pending = false;
            RUN_PASSED[2].fetch_add(1, Ordering::Relaxed);
            return true;
        }
        if self.chunk_waits.binary_search(&at).is_err() {
            return false;
        }
        let (a, wide) = (addr64(p(1), p(2)), opcode == op::WAIT_REG_MEM64);
        let v = if wide { unsafe { rd64(a) } } else { (unsafe { rd32(a) }) as u64 };
        if let Some(e) = self.exec.as_mut() {
            e.xq_label(a, wide, v);
        }
        RUN_PASSED[3].fetch_add(1, Ordering::Relaxed);
        true
    }

    /// A worker's chunk: its run's segment in the graphics queue's
    /// handshake census (`gpu::hscensus`).
    pub fn set_hs_seg(&mut self, seg: u64) {
        if let Some(e) = self.exec.as_mut() {
            e.hs_seg = seg;
        }
    }

    /// The graphics queue's handshake census (`gpu::hscensus`; the main
    /// command processor at each packet): a packet a run through cache
    /// actions, partial flushes and label handshakes would still end at ends
    /// the stretch; a serial command is a segment of its own.
    fn hs_census_packet(&mut self, opcode: u32, p: &dyn Fn(u64) -> u32, n: u64) {
        let multi = matches!(opcode, op::DRAW_INDEX_INDIRECT_MULTI | op::DRAW_INDIRECT_MULTI);
        let pass = is_state(opcode)
            || is_command(opcode)
            || multi
            || matches!(
                opcode,
                op::SET_BASE
                    | op::INDEX_TYPE
                    | op::INDEX_BASE
                    | op::INDEX_BUFFER_SIZE
                    | op::NUM_INSTANCES
                    | op::CONTEXT_CONTROL
                    | op::INDIRECT_BUFFER
                    | op::ACQUIRE_MEM
                    | op::WRITE_DATA
                    | op::RELEASE_MEM
                    | op::WAIT_REG_MEM
                    | op::WAIT_REG_MEM64
            )
            || (opcode == op::EVENT_WRITE && matches!(p(0) & 0x3F, 7 | 0x10))
            || (opcode == op::NOP && !(n >= 2 && p(0) == DS_MAGIC && !matches!(p(1), pop::PUSH_MARKER | pop::SET_MARKER | pop::POP_MARKER)));
        if !pass {
            super::hscensus::boundary();
        }
        if (is_command(opcode) || multi) && self.par.is_none() {
            let seg = super::hscensus::next_segment();
            if let Some(e) = self.exec.as_mut() {
                e.hs_seg = seg;
            }
        }
    }

    /// A worker's chunk: the guest ranges CP writes of its run's earlier
    /// chunks write (`Exec::run_writes`).
    pub fn set_run_writes(&mut self, w: Vec<(u64, u64)>) {
        if let Some(e) = self.exec.as_mut() {
            e.run_writes = w;
        }
    }

    /// A compute queue's packet its run holds through a label handshake
    /// (planning; since 2026-11-07 night: queue 0x50's own work 6.7 -> 3.4
    /// ms a flip, the vista +6.5%, Doran +8.6%): `COMPUTE_PASS_WRITE` a WRITE_DATA
    /// to memory of 1-16 dwords (a worker performs it on the GPU, as this
    /// command processor does after work: `Exec::gpu_cp_write`, its label
    /// promised when its batch is submitted), `RUN_PASS_WAIT` a label wait
    /// that holds in guest memory now or on a value a WRITE_DATA earlier in
    /// the run writes (`writes`: label, value, bytes).
    fn compute_pass(&mut self, opc: u32, n: u64, p: &dyn Fn(u64) -> u32, writes: &[(u64, u64, u8)]) -> Option<u8> {
        if !self.compute || !super::fe2::on() {
            return None;
        }
        match opc {
            op::WRITE_DATA => {
                let c = p(0);
                let (dst, count) = ((c >> 8) & 0xF, n.saturating_sub(3));
                let a = addr64(p(1), p(2));
                (matches!(dst, 1 | 2 | 5) && (1..=16).contains(&count) && a % 4 == 0 && valid(a, 4 * count)).then_some(COMPUTE_PASS_WRITE)
            }
            op::WAIT_REG_MEM | op::WAIT_REG_MEM64 => {
                let (c, a, wide) = (p(0), addr64(p(1), p(2)), opc == op::WAIT_REG_MEM64);
                if (c >> 4) & 3 != 1 || !valid(a, 8) {
                    return None;
                }
                let (reference, mask) = if wide { (addr64(p(3), p(4)), addr64(p(5), p(6))) } else { (p(3) as u64, p(4) as u64) };
                let func = c & 7;
                // A write earlier in the run decides it (the last one there).
                if let Some(&(_, v, bytes)) = writes.iter().rev().find(|w| w.0 == a) {
                    let v = if wide && bytes < 8 { return None } else if wide { v } else { v & 0xFFFF_FFFF };
                    return compare(func, v & mask, reference & mask).then_some(RUN_PASS_WAIT_OWN);
                }
                self.before_fetch(a, 8);
                let v = if wide { unsafe { rd64(a) } } else { (unsafe { rd32(a) }) as u64 };
                compare(func, v & mask, reference & mask).then_some(RUN_PASS_WAIT)
            }
            _ => None,
        }
    }

    /// A compute queue's worker performs a WRITE_DATA its run holds
    /// (`compute_pass`) on the GPU in its batch; false: the chunk stops
    /// there.
    fn chunk_write_data(&mut self, n: u64, p: &dyn Fn(u64) -> u32) -> bool {
        let c = p(0);
        let (dst, no_incr, count) = ((c >> 8) & 0xF, (c >> 16) & 1 != 0, n.saturating_sub(3));
        let a = addr64(p(1), p(2));
        if !matches!(dst, 1 | 2 | 5) || count == 0 || !valid(a, 4 * count) {
            return false;
        }
        let vals: Vec<u32> = (0..count).map(|i| p(3 + i)).collect();
        let label = match (count, no_incr) {
            (1, _) => Some((vals[0] as u64, 4u8)),
            (2, false) => Some((vals[0] as u64 | (vals[1] as u64) << 32, 8u8)),
            _ => None,
        };
        let gpu_vals = if no_incr { vec![*vals.last().unwrap()] } else { vals.clone() };
        let Some(e) = self.exec.as_mut() else { return false };
        if !e.gpu_cp_write(a, &gpu_vals, label, false) {
            return false;
        }
        super::replay::note_cp_write(a, 4 * gpu_vals.len() as u64);
        self.fence_pending = false;
        // (Its queue's write, as `gpu_cp_write` notes the main command
        // processor's.)
        let queue = self.name.rsplit_once("-par").map_or(self.name.as_str(), |x| x.0).to_string();
        let mut g = LAST_WRITE.lock();
        let m = g.get_or_insert_with(Default::default);
        if m.len() > 1 << 16 {
            m.clear();
        }
        m.insert(a & !7, (queue, label.map(|l| l.0).unwrap_or(gpu_vals[0] as u64), self.cur_packet, a));
        RUN_PASSED[0].fetch_add(1, Ordering::Relaxed);
        true
    }

    /// Why this worker's chunk stopped.
    pub fn worker_stop_why(&self) -> &'static str {
        self.exec.as_ref().map_or("", |e| e.stop_why)
    }

    /// After a label wait of the graphics queue: the CPU may have written
    /// data before releasing it, so shared batch-start copies are dropped
    /// (`gpu::precache`), unless the label holds what a GPU queue's CP wrote
    /// last (the queues synchronize this way ~250 times per frame).
    fn end_copy_scope_after_wait(&mut self, a: u64, wide: bool) {
        if !valid(a, if wide { 8 } else { 4 }) || (self.held.is_some() && self.past.is_none() && self.past_wait.is_none()) {
            // (A compute queue's view: the epoch now, whatever the label.)
            self.fe2_sync(None);
            return;
        }
        // Front end 2: passed on another queue's promise or a latched CP
        // write, no CPU write released the wait: the CPU writes the title
        // ordered before it came before the writer's synchronization, which
        // the promise's epoch (or the epoch now) covers; no epoch ends.
        if self.last_wait_gpu && super::fe2::on() {
            // (A value not seen here as the CPU's: the counter's history
            // starts again, `passed_again`.)
            self.passed.remove(&(a & !7));
            EPOCH_GPU_WAITS.fetch_add(1, Ordering::Relaxed);
            self.fe2_adopt_epoch(if self.last_wait_kind == b'P' { self.last_wait_epoch } else { super::precache::epoch() as u32 });
            return;
        }
        let quiet = super::fe2::on();
        // Front end 2: gone past, the wait released nothing yet; the
        // stretch checks as of the epoch at each check and is checked again
        // at its label, where its view is taken.
        if quiet && self.last_wait_kind == b'G' && self.past_wait.is_some() && super::tune::past_check() {
            QUIET_WAITS[1].fetch_add(1, Ordering::Relaxed);
            return;
        }
        let now = unsafe { if wide { rd64(a) } else { rd32(a) as u64 } };
        let last = last_write(a);
        let by_gpu = last.as_ref().is_some_and(|(q, v, _)| q.starts_with("gpu-") && if wide { *v == now } else { *v as u32 == now as u32 });
        if by_gpu {
            self.passed.remove(&(a & !7));
            self.fe2_adopt(last.as_ref().map(|w| w.0.as_str()));
        } else if quiet && self.passed_again(a, now) {
            // The value this queue passed on last time at this counter
            // label: the CPU writes the title ordered before that write are
            // older than its view then (no wait can have been released by a
            // later write of the same value).
            QUIET_WAITS[0].fetch_add(1, Ordering::Relaxed);
        } else {
            super::precache::end_scope(1);
            self.fe2_sync(None);
            if super::exec::profiling_on() {
                let what = match last_write_at(a) {
                    None => "no CP write recorded",
                    Some((q, _, _, at)) if !q.starts_with("gpu-") => if at == a { "last written by a non-GPU queue" } else { "last written by a non-GPU queue (other half)" },
                    Some((_, _, _, at)) if !wide && at != a => if label_wrote(a, wide, now) { "a GPU queue's earlier value (slot's last write to the other half)" } else { "no GPU queue's value (slot's last write to the other half)" },
                    Some(_) => if label_wrote(a, wide, now) { "a GPU queue's earlier value" } else { "no GPU queue's value" },
                };
                let mut g = EPOCH_CLASSES.lock();
                let e = g.get_or_insert_with(HashMap::new).entry((self.name.clone(), self.last_wait_kind, what)).or_default();
                e.0 += 1;
                e.1.insert(a);
                drop(g);
                let mut g = EPOCH_REPEATS.lock();
                let r = g.get_or_insert_with(HashMap::new).entry((self.name.clone(), a)).or_insert((u64::MAX, 0, 0, self.last_wait_func));
                if r.0 == now {
                    r.1 += 1;
                } else {
                    r.2 += 1;
                    r.0 = now;
                }
                drop(g);
                let by = match last {
                    Some((q, v, _)) if (if wide { v == now } else { v as u32 == now as u32 }) => format!("written by {q}"),
                    Some((q, _, _)) => format!("{q} wrote another value"),
                    None => "no CP write recorded".into(),
                };
                *EPOCH_WAITS.lock().get_or_insert_with(HashMap::new).entry((self.name.clone(), a, by)).or_default() += 1;
            }
        }
    }

    /// The epoch the label at `a` that the last wait passed on orders before
    /// the work after it (front end 2), and what wrote
    /// it: a promise (its writer's view then), a latched CP write (the epoch
    /// now), a GPU queue's CP write in memory (that queue's view, the epoch
    /// now without one), else the CPU (later than any check made so far).
    fn label_needs(&self, a: u64, wide: bool) -> (u32, &'static str) {
        let now = super::precache::epoch() as u32;
        match self.last_wait_kind {
            b'P' => (self.last_wait_epoch, "checks older than its label's promise"),
            b'L' => (now, "checks older than its label's latched write"),
            _ if !valid(a, if wide { 8 } else { 4 }) => (now.saturating_add(1), "checks older than its label's CPU write"),
            _ => {
                let v = unsafe { if wide { rd64(a) } else { rd32(a) as u64 } };
                match last_write(a) {
                    Some((q, w, _)) if q.starts_with("gpu-") && (if wide { w == v } else { w as u32 == v as u32 }) => (super::fe2::view_of(&q).unwrap_or(now), "checks older than its label's GPU write"),
                    _ => (now.saturating_add(1), "checks older than its label's CPU write"),
                }
            }
        }
    }

    /// A title submission starts (its command buffers follow): CPU writes
    /// come between submissions, so shared batch-start copies of the
    /// graphics queue are dropped (`gpu::precache`).
    pub fn new_submission(&mut self) {
        CUR_SUB.with(|c| c.set(self.sub_index));
        if self.held.is_none() || self.past.is_some() || self.past_wait.is_some() {
            super::precache::end_scope(0);
            self.fe2_sync(None);
        }
    }

    /// Front end 2 (compute queues, `fe2::views_on`): this queue
    /// synchronized; its checks need pages current as of the epoch now, or
    /// as of the view `adopt`.
    fn fe2_sync(&self, adopt: Option<u32>) {
        if let Some(c) = self.fe2_view.as_ref() {
            if super::fe2::sync_view(c, adopt.unwrap_or_else(|| super::precache::epoch() as u32)) {
                super::fe2::note_raise(self.compute);
                if super::fe2census::on() {
                    super::fe2census::note_raise(&self.name);
                }
            }
        }
    }

    /// This command processor records part `chunk` of run `run` (`par::
    /// Run::id`; 0: none): uploads the run's parts share (`Exec::
    /// fe2_share`).
    pub fn fe2_set_run(&mut self, run: u64, chunk: usize) {
        if let Some(e) = self.exec.as_mut() {
            e.fe2_run = (run, chunk.min(super::fe2::RUN_CHUNKS as usize + 1) as u32);
        }
    }

    /// A parallel worker records a chunk of a run planned at view `v`
    /// (`par::Run::fe2_view`).
    pub fn fe2_set_view(&self, v: u32) {
        if let (Some(c), true) = (self.fe2_view.as_ref(), v != super::fe2::NO_VIEW) {
            let _ = super::fe2::sync_view(c, v);
        }
    }

    /// A wait passed on a value the CP of queue `writer` wrote: that
    /// queue's view (the epoch now for queues without one).
    fn fe2_adopt(&self, writer: Option<&str>) {
        if self.fe2_view.is_some() {
            self.fe2_sync(Some(writer.and_then(super::fe2::view_of).unwrap_or_else(|| super::precache::epoch() as u32)));
        }
    }

    /// A wait that passed at once on `now` at label `a` as released by the
    /// CPU: whether this queue passed on that same value last time there and
    /// the label is a counter (at least three values seen, each greater than
    /// the one before); notes the value either way (front end 2).
    fn passed_again(&mut self, a: u64, now: u64) -> bool {
        let at_once = self.last_wait_kind == b'I';
        if self.passed.len() > 4096 {
            self.passed.clear();
        }
        let e = self.passed.entry(a & !7).or_insert((now, 1, true));
        if e.0 == now {
            return at_once && e.2 && e.1 >= 3;
        }
        e.2 &= now > e.0;
        e.1 = e.1.saturating_add(1);
        e.0 = now;
        false
    }

    /// The view a label write recorded now is promised with (`promise`):
    /// this queue's, the epoch now without one.
    fn fe2_promise_view(&self) -> u32 {
        match self.fe2_view.as_ref().map(|c| c.load(Ordering::Acquire)) {
            Some(v) if v != super::fe2::NO_VIEW => v,
            _ => super::precache::epoch() as u32,
        }
    }

    /// This queue's checks need at least epoch `v` from here on (front
    /// end 2); one without a view yet takes the epoch now.
    fn fe2_adopt_epoch(&self, v: u32) {
        if let Some(c) = self.fe2_view.as_ref() {
            let v = if c.load(Ordering::Acquire) == super::fe2::NO_VIEW { super::precache::epoch() as u32 } else { v };
            self.fe2_sync(Some(v));
        }
    }

    /// What the batches this worker closed write (`DS_RUN_PROBE`).
    pub fn worker_writes(&self) -> super::ranges::Ranges {
        self.exec.as_ref().map(|e| e.closed_writes()).unwrap_or_default()
    }

    /// Submits the batches this worker closed (the main command processor,
    /// in program order); the last one's sequence.
    pub fn worker_submit(&mut self) -> Option<u64> {
        self.exec.as_mut().and_then(|e| e.submit_closed())
    }

    /// Drops the batches this worker closed (a cancelled parallel run).
    pub fn worker_discard(&mut self) {
        if let Some(e) = self.exec.as_mut() {
            e.discard_closed();
        }
    }

    /// A held run's chunk begins on this worker (`Exec::held_chunk_begin`).
    pub fn held_chunk_begin(&mut self, writes: Arc<super::ranges::Ranges>, keys: Arc<HashMap<super::images::ImageKey, u8>>) {
        if let Some(e) = self.exec.as_mut() {
            e.held_chunk_begin(writes, keys);
        }
    }

    /// The held chunk's end: its record (`Exec::held_chunk_end`).
    pub fn held_chunk_end(&mut self) -> super::exec::HeldChunk {
        self.exec.as_mut().map(|e| e.held_chunk_end()).unwrap_or_default()
    }

    /// Batches its executor closed and holds.
    pub fn closed_count(&self) -> usize {
        self.exec.as_ref().map_or(0, |e| e.closed_count())
    }

    /// Drops its executor's closed batches after the first `keep` (`Exec::discard_closed_tail`).
    pub fn discard_closed_tail(&mut self, keep: usize) {
        if let Some(e) = self.exec.as_mut() {
            e.discard_closed_tail(keep);
        }
    }

    /// Submits its executor's first `n` closed batches (`Exec::submit_closed_first`).
    pub fn submit_closed_first(&mut self, n: usize) -> Option<u64> {
        self.exec.as_mut().and_then(|e| e.submit_closed_first(n))
    }

    /// Applies a state packet of a run to `st` as `packet` would.
    /// Whether `apply_state` changes anything for packets of `opcode` (its match's arms; the
    /// planner's walk calls it for these only).
    #[inline]
    fn acts_on_state(opcode: u32) -> bool {
        matches!(
            opcode,
            op::SET_CONTEXT_REG
                | op::SET_CONTEXT_REG_INDEX
                | op::SET_SH_REG
                | op::SET_SH_REG_INDEX
                | op::SET_UCONFIG_REG
                | op::SET_UCONFIG_REG_INDEX
                | op::LOAD_CONTEXT_REG_INDEX
                | op::LOAD_SH_REG_INDEX
                | op::LOAD_UCONFIG_REG
                | op::INDEX_TYPE
                | op::INDEX_BASE
                | op::INDEX_BUFFER_SIZE
                | op::NUM_INSTANCES
                | op::SET_BASE
                | op::DRAW_INDIRECT
                | op::DRAW_INDEX_INDIRECT
                | op::DRAW_INDEX_INDIRECT_MULTI
                | op::DRAW_INDIRECT_MULTI
        )
    }

    fn apply_state(st: &mut ChunkState, opcode: u32, n: u64, p: &dyn Fn(u64) -> u32) {
        let set = |v: &mut Vec<u32>, space: Space, cap: usize, first: u32| {
            for i in 1..n {
                put_reg(v, space, cap, (first + (i - 1) as u32) as usize, p(i));
            }
        };
        let load = |v: &mut Vec<u32>, space: Space, cap: usize, a: u64, count: u32| {
            if count == 0 || !valid(a, 8 * count as u64) {
                return;
            }
            for i in 0..count as u64 {
                let (o, val) = unsafe { (rd32(a + 8 * i), rd32(a + 8 * i + 4)) };
                put_reg(v, space, cap, (o & 0xFFFF) as usize, val);
            }
        };
        match opcode {
            op::SET_CONTEXT_REG | op::SET_CONTEXT_REG_INDEX => set(&mut st.regs.ctx, Space::Ctx, CTX_DWORDS, p(0) & 0xFFFF),
            op::SET_SH_REG | op::SET_SH_REG_INDEX => set(&mut st.regs.sh, Space::Sh, SH_DWORDS, p(0) & 0xFFFF),
            op::SET_UCONFIG_REG | op::SET_UCONFIG_REG_INDEX => set(&mut st.regs.uc, Space::Uc, UC_DWORDS, p(0) & 0xFFFF),
            op::LOAD_CONTEXT_REG_INDEX => load(&mut st.regs.ctx, Space::Ctx, CTX_DWORDS, addr64(p(0), p(1)), p(2)),
            op::LOAD_SH_REG_INDEX => load(&mut st.regs.sh, Space::Sh, SH_DWORDS, addr64(p(0), p(1)), p(2)),
            op::LOAD_UCONFIG_REG => load(&mut st.regs.uc, Space::Uc, UC_DWORDS, addr64(p(0), p(1)), p(2)),
            op::INDEX_TYPE => put_reg(&mut st.regs.uc, Space::Uc, UC_DWORDS, crate::hle::libs::agc_cmd::REG_VGT_INDEX_TYPE as usize, p(0)),
            op::INDEX_BASE => st.index_base = addr64(p(0), p(1)),
            op::INDEX_BUFFER_SIZE => st.index_count = p(0),
            op::NUM_INSTANCES => st.num_instances = p(0).max(1),
            op::SET_BASE => {
                let base = addr64(p(1), p(2));
                if p(0) == 2 {
                    st.dispatch_indirect_base = base
                } else {
                    st.draw_indirect_base = base
                }
            }
            op::DRAW_INDIRECT | op::DRAW_INDEX_INDIRECT => {
                // The base vertex and start instance the packet hands to user SGPRs.
                let a = st.draw_indirect_base + p(0) as u64;
                let (bytes, k) = if opcode == op::DRAW_INDEX_INDIRECT { (20, 3) } else { (16, 2) };
                if !valid(a, bytes) {
                    return;
                }
                let index_ok = opcode == op::DRAW_INDIRECT || matches!(st.regs.uc.get(crate::hle::libs::agc_cmd::REG_VGT_INDEX_TYPE as usize).map(|t| t & 3), Some(0 | 1));
                if !index_ok {
                    return;
                }
                let (bv, si) = unsafe { (rd32(a + 4 * k), rd32(a + 4 * (k + 1))) };
                let (vtx_loc, inst_loc) = (p(1) & 0xFFFF, p(1) >> 16);
                let inst_loc = if inst_loc != 0 {
                    inst_loc
                } else {
                    let r = ScanRegs { sh: &st.regs.sh, ctx: &st.regs.ctx };
                    crate::hle::libs::agc::instance_offset_slot(super::shaders::ngg_addr_of(&r)).map_or(0, |s| 0x8C + s)
                };
                if vtx_loc != 0 {
                    put_reg(&mut st.regs.sh, Space::Sh, SH_DWORDS, vtx_loc as usize, bv);
                }
                if inst_loc != 0 {
                    put_reg(&mut st.regs.sh, Space::Sh, SH_DWORDS, inst_loc as usize, si);
                }
            }
            op::DRAW_INDEX_INDIRECT_MULTI | op::DRAW_INDIRECT_MULTI => {
                // As `draw_indirect_multi` leaves them: each record of the
                // count (from memory) sets the user SGPRs the packet names in
                // turn, up to the first unreadable one; 8-bit indices none.
                let indexed = opcode == op::DRAW_INDEX_INDIRECT_MULTI;
                if indexed && !matches!(st.regs.uc.get(crate::hle::libs::agc_cmd::REG_VGT_INDEX_TYPE as usize).map(|t| t & 3), Some(0 | 1)) {
                    return;
                }
                let (vtx_loc, inst_loc, id_loc) = (p(1) & 0xFFFF, p(2) & 0xFFFF, p(3) & 0xFFFF);
                let id_enable = p(3) & (1 << 31) != 0;
                let mut count = p(4);
                if p(3) & (1 << 30) != 0 {
                    let ca = addr64(p(5), p(6));
                    count = if valid(ca, 4) { count.min(unsafe { rd32(ca) }) } else { 0 };
                }
                let rec: u64 = if indexed { 20 } else { 16 };
                let stride = if p(7) != 0 { p(7) as u64 } else { rec };
                let first = st.draw_indirect_base + p(0) as u64;
                let mut last = None;
                for i in 0..count.min(1 << 16) {
                    if !valid(first + i as u64 * stride, rec) {
                        break;
                    }
                    last = Some(i);
                }
                let Some(i) = last else { return };
                let a = first + i as u64 * stride;
                let (bv, si) = unsafe { if indexed { (rd32(a + 12), rd32(a + 16)) } else { (rd32(a + 8), rd32(a + 12)) } };
                let inst_loc = if inst_loc != 0 {
                    inst_loc
                } else {
                    let r = ScanRegs { sh: &st.regs.sh, ctx: &st.regs.ctx };
                    crate::hle::libs::agc::instance_offset_slot(super::shaders::ngg_addr_of(&r)).map_or(0, |s| 0x8C + s)
                };
                if vtx_loc != 0 {
                    put_reg(&mut st.regs.sh, Space::Sh, SH_DWORDS, vtx_loc as usize, bv);
                }
                if inst_loc != 0 {
                    put_reg(&mut st.regs.sh, Space::Sh, SH_DWORDS, inst_loc as usize, si);
                }
                if id_enable && id_loc != 0 {
                    put_reg(&mut st.regs.sh, Space::Sh, SH_DWORDS, id_loc as usize, i);
                }
            }
            _ => {}
        }
    }

    /// Census (`HS_CENSUS`): from `from`, the stretch a compute queue's run
    /// would be if WRITE_DATA, label waits and flushes left it open; counted
    /// once per stretch (`hs_until`).
    fn handshake_census(&mut self, f: &Fetched, from: u64, end: u64) {
        if self.hs_until.0 == end && from < self.hs_until.1 {
            return;
        }
        let mut c = from;
        let mut n = 0u64;
        while c < end && valid(c, 4) {
            let h = f.rd(c);
            match h >> 30 {
                3 => {}
                2 => {
                    c += 4;
                    continue;
                }
                _ => break,
            }
            let count = (h >> 16) & 0x3FFF;
            if count == 0x3FFF {
                c += 4;
                continue;
            }
            if h & 1 != 0 {
                break;
            }
            let opc = (h >> 8) & 0xFF;
            let len = count as u64 + 1;
            let next = c + 4 * (len + 1);
            if next > end || !valid(c, 4 * (len + 1)) {
                break;
            }
            if is_command(opc) {
                n += 1;
            } else if opc == op::NOP {
                if len >= 2 && f.rd(c + 4) == DS_MAGIC && !matches!(f.rd(c + 8), pop::PUSH_MARKER | pop::SET_MARKER | pop::POP_MARKER) {
                    break;
                }
            } else if !(opc == op::SET_BASE || is_state(opc) || matches!(opc, op::EVENT_WRITE | op::ACQUIRE_MEM | op::WAIT_REG_MEM | op::WAIT_REG_MEM64) || (opc == op::WRITE_DATA && len <= 4)) {
                break;
            }
            c = next;
        }
        self.hs_until = (end, c);
        let par_min = (super::tune::par_min_compute() as usize).max(super::par::floor()) as u64;
        HS_CENSUS[0].fetch_add(1, Ordering::Relaxed);
        HS_CENSUS[1].fetch_add(n, Ordering::Relaxed);
        if n >= par_min {
            HS_CENSUS[2].fetch_add(n, Ordering::Relaxed);
        }
    }

    /// The planner's scan of the run at `from` (`par_plan`, `plan_ahead`): its commands up to
    /// the first packet a run cannot hold. Label waits' reads are prepared and runs'
    /// timestamps written as it passes them (`run_pass`).
    fn plan_scan(&mut self, f: &Fetched, from: u64, end: u64) -> PlanScan {
        // The run's commands, its end and the indirect arguments read in it.
        let mut ends: Vec<u64> = Vec::new();
        let mut args: Vec<(u64, u64)> = Vec::new();
        let (mut draw_base, mut disp_base) = (self.draw_indirect_base, self.dispatch_indirect_base);
        let mut c = from;
        let mut why = 0x130usize;
        let mut sync = false;
        // Whether the chunks' indirect dispatches read their arguments on
        // the GPU (`Exec::dispatch_gpu_indirect`; every queue's runs since
        // 2026-11-09, the compute queues' since 2026-10-19: the graphics
        // queue's planning waited ~1-3 ms twice a frame at the swamp for
        // the tessellation preparation's arguments, FINDINGS.md; captures
        // and interpreter checks read them here).
        let gpu_dispatch_args = super::tune::gpu_indirect() && !super::capture::capturing() && !super::verify::enabled() && !(super::exec::enabled() && super::emucheck::wanted(self));
        // The run's signature: its packet headers (opcode and size), the
        // same when the title records the same pass again.
        let mut sig = 0xcbf2_9ce4_8422_2325u64;
        // The label waits it holds (`run_pass`): packet, label, 64-bit; those
        // a write earlier in the run meets (`compute_pass`): packets; its
        // CP writes: label, value, bytes (labels) and packet, address, bytes.
        let mut passed: Vec<(u64, u64, bool)> = Vec::new();
        let mut own_waits: Vec<u64> = Vec::new();
        let mut cwrites: Vec<(u64, u64, u8)> = Vec::new();
        let mut cp_writes: Vec<(u64, u64, u64)> = Vec::new();
        let run_end = loop {
            if c >= end || !valid(c, 4) {
                break c;
            }
            let h = f.rd(c);
            sig = (sig ^ h as u64).wrapping_mul(0x9E37_79B9_7F4A_7C15);
            match h >> 30 {
                3 => {}
                2 => {
                    c += 4;
                    continue;
                }
                _ => break c,
            }
            let count = (h >> 16) & 0x3FFF;
            if count == 0x3FFF {
                c += 4;
                continue;
            }
            if h & 1 != 0 {
                why = 0x133;
                break c;
            }
            let opc = (h >> 8) & 0xFF;
            let n = count as u64 + 1;
            let next = c + 4 * (n + 1);
            if next > end || !valid(c, 4 * (n + 1)) {
                break c;
            }
            // (Register writes, most of a run's packets, are state that no check below takes.)
            if matches!(opc, op::SET_CONTEXT_REG | op::SET_CONTEXT_REG_INDEX | op::SET_SH_REG | op::SET_SH_REG_INDEX | op::SET_UCONFIG_REG | op::SET_UCONFIG_REG_INDEX) {
                c = next;
                continue;
            }
            let at = c;
            let p = |i: u64| f.rd(at + 4 * (1 + i));
            if is_command(opc) || self.run_multi(opc) {
                ends.push(next);
                match opc {
                    op::DRAW_INDIRECT => args.push((draw_base + p(0) as u64, 16)),
                    op::DRAW_INDEX_INDIRECT => args.push((draw_base + p(0) as u64, 20)),
                    op::DRAW_INDEX_INDIRECT_MULTI | op::DRAW_INDIRECT_MULTI => {
                        // Its count (from memory) and its records at their
                        // largest count.
                        if p(3) & (1 << 30) != 0 {
                            args.push((addr64(p(5), p(6)), 4));
                        }
                        let rec = if opc == op::DRAW_INDEX_INDIRECT_MULTI { 20 } else { 16 };
                        let stride = if p(7) != 0 { p(7) as u64 } else { rec };
                        let most = p(4).min(1 << 16) as u64;
                        if most > 0 {
                            args.push((draw_base + p(0) as u64, (most - 1) * stride + rec));
                        }
                    }
                    // Dispatch arguments the chunks let the GPU read
                    // (`Exec::dispatch_gpu_indirect`) need no wait here
                    // (one per run start cost queue 0x50's runs a GPU round
                    // trip of ~2.5 ms, twice per flip with parallel runs).
                    op::DISPATCH_INDIRECT if gpu_dispatch_args => {}
                    op::DISPATCH_INDIRECT => args.push((if n >= 3 { addr64(p(0), p(1)) } else { disp_base + p(0) as u64 }, 12)),
                    _ => {}
                }
                if ends.len() >= PAR_MAX {
                    why = 0x131;
                    break next;
                }
            } else if opc == op::NOP {
                if n >= 2 && p(0) == DS_MAGIC && !matches!(p(1), pop::PUSH_MARKER | pop::SET_MARKER | pop::POP_MARKER) {
                    why = 0x100 + (p(1) as usize).min(0x2F);
                    break c;
                }
            } else if opc == op::SET_BASE {
                if p(0) == 2 {
                    disp_base = addr64(p(1), p(2));
                } else {
                    draw_base = addr64(p(1), p(2));
                }
            } else if par_event(opc, self.compute) {
                sync = true;
            } else if let Some(k) = self.run_pass(opc, &p) {
                if k == RUN_PASS_WAIT {
                    passed.push((at, addr64(p(1), p(2)), opc == op::WAIT_REG_MEM64));
                }
            } else if let Some(k) = self.compute_pass(opc, n, &p, &cwrites) {
                match k {
                    COMPUTE_PASS_WRITE => {
                        let (c0, a) = (p(0), addr64(p(1), p(2)));
                        let (no_incr, count) = ((c0 >> 16) & 1 != 0, n - 3);
                        match (count, no_incr) {
                            (1, _) => cwrites.push((a, p(3) as u64, 4)),
                            (2, false) => cwrites.push((a, p(3) as u64 | (p(4) as u64) << 32, 8)),
                            _ => {}
                        }
                        cp_writes.push((at, a, if no_incr { 4 } else { 4 * count }));
                    }
                    RUN_PASS_WAIT => passed.push((at, addr64(p(1), p(2)), opc == op::WAIT_REG_MEM64)),
                    _ => own_waits.push(at),
                }
            } else if !is_state(opc) {
                // EVENT_WRITE by event type (statistics: partial flushes
                // and cache events order only GPU work).
                why = if opc == op::EVENT_WRITE { 0x200 + (p(0) & 0x3F) as usize } else { opc as usize };
                break c;
            }
            c = next;
        };
        PlanScan { ends, args, passed, own_waits, cp_writes, sync, sig, why, run_end }
    }

    /// A split run's chunks (`par_plan`, `plan_ahead`) and the walks of the command stream for
    /// their start states and the run's end state (`tune past_runs`; not for a held run).
    /// `timing`: `tune cp_classes`' counters.
    #[allow(clippy::too_many_arguments)]
    fn plan_chunks(&mut self, f: &Fetched, from: u64, ends: &[u64], run_end: u64, n: usize, chunks: usize, chunk_min: usize, sig: u64, held_run: bool, timing: bool) -> PlanChunks {
        // The main command processor's chunk first (`par::main_weight`
        // times a worker's), then equal ones for the workers' queue.
        let m = chunks - 1;
        let w = super::par::main_weight();
        let main_n = ((n as f64 * w / (w + m as f64)).round() as usize).clamp(chunk_min, n - m * chunk_min);
        let per = (n - main_n) / m;
        // At equal recording cost when the run was measured before, else at
        // equal command counts.
        let sig = (sig ^ n as u64).wrapping_mul(0x9E37_79B9_7F4A_7C15) | 1;
        let frame = super::FRAMES.load(Ordering::Relaxed);
        let costed = {
            if self.run_costs.len() > 512 && (self.run_costs_pruned != frame || !super::tune::plan_fast()) {
                self.run_costs_pruned = frame;
                self.run_costs.retain(|_, r| r.seen + 120 >= frame);
            }
            let rc = self.run_costs.entry(sig).or_insert_with(|| RunCost { costs: vec![f32::NAN; n], seen: frame });
            rc.seen = frame;
            if rc.costs.len() == n {
                cost_firsts(&rc.costs, main_n, n, m, chunk_min)
            } else {
                None
            }
        };
        if costed.is_some() {
            PAR_COSTED.fetch_add(1, Ordering::Relaxed);
        }
        let firsts: Vec<usize> = costed.unwrap_or_else(|| (0..m).map(|i| main_n + i * per).collect());
        let starts: Vec<u64> = firsts.iter().map(|&i| if i == 0 { from } else { ends[i - 1] }).collect();
        // The state at each chunk's start.
        let mut st = self.chunk_state();
        let mut states: Vec<Box<ChunkState>> = Vec::new();
        let mut c = from;
        // (`tune plan_check`: the same a register at a time.)
        let mut check = super::tune::plan_check().then(|| self.chunk_state());
        for &s in &starts {
            while c < s {
                let h = f.rd(c);
                if h >> 30 == 2 {
                    c += 4;
                    continue;
                }
                let count = (h >> 16) & 0x3FFF;
                if count == 0x3FFF {
                    c += 4;
                    continue;
                }
                let n = count as u64 + 1;
                let at = c;
                let opc = (h >> 8) & 0xFF;
                // Register ranges in one copy (most of a run's packets).
                let regs = match opc {
                    op::SET_CONTEXT_REG | op::SET_CONTEXT_REG_INDEX => Some((Space::Ctx, CTX_DWORDS)),
                    op::SET_SH_REG | op::SET_SH_REG_INDEX => Some((Space::Sh, SH_DWORDS)),
                    op::SET_UCONFIG_REG | op::SET_UCONFIG_REG_INDEX => Some((Space::Uc, UC_DWORDS)),
                    _ => None,
                };
                match regs {
                    Some((space, cap)) if n >= 2 => {
                        let v = match space {
                            Space::Ctx => &mut st.regs.ctx,
                            Space::Sh => &mut st.regs.sh,
                            Space::Uc => &mut st.regs.uc,
                        };
                        put_regs(v, space, cap, (f.rd(at + 4) & 0xFFFF) as usize, f, at + 8, (n - 1) as usize);
                    }
                    _ if timing && matches!(opc, op::DRAW_INDIRECT | op::DRAW_INDEX_INDIRECT | op::DRAW_INDEX_INDIRECT_MULTI | op::DRAW_INDIRECT_MULTI) => {
                        let t = Instant::now();
                        Self::apply_state(&mut st, opc, n, &|i: u64| f.rd(at + 4 * (1 + i)));
                        PLAN_WALK[1].fetch_add(1, Ordering::Relaxed);
                        PLAN_WALK[2].fetch_add(t.elapsed().as_nanos() as u64, Ordering::Relaxed);
                    }
                    // (Draws, dispatches, NOPs and events change no chunk state.)
                    _ if Self::acts_on_state(opc) => Self::apply_state(&mut st, opc, n, &|i: u64| f.rd(at + 4 * (1 + i))),
                    _ => {}
                }
                if timing {
                    PLAN_WALK[0].fetch_add(1, Ordering::Relaxed);
                }
                if let Some(st2) = check.as_mut() {
                    Self::apply_state(st2, opc, n, &|i: u64| f.rd(at + 4 * (1 + i)));
                }
                c += 4 * (n + 1);
            }
            if let Some(st2) = check.as_ref() {
                plan_check_states(&st, st2, s);
            }
            if timing {
                let tc = Instant::now();
                states.push(Box::new(st.clone()));
                PLAN_CLONE_NS.fetch_add(tc.elapsed().as_nanos() as u64, Ordering::Relaxed);
            } else {
                states.push(Box::new(st.clone()));
            }
        }
        // The state at the run's end (`tune past_runs`: the graphics command
        // processor goes on from there before collecting the chunks).
        let mut end_unknown: Vec<u32> = Vec::new();
        let end_state = (super::tune::past_runs() && self.main_gfx && !held_run).then(|| {
            let tc = timing.then(Instant::now);
            let mut st = st.clone();
            if let Some(tc) = tc {
                PLAN_CLONE_NS.fetch_add(tc.elapsed().as_nanos() as u64, Ordering::Relaxed);
            }
            let mut c = c;
            while c < run_end {
                let h = f.rd(c);
                if h >> 30 == 2 {
                    c += 4;
                    continue;
                }
                let count = (h >> 16) & 0x3FFF;
                if count == 0x3FFF {
                    c += 4;
                    continue;
                }
                let n = count as u64 + 1;
                let at = c;
                let opc = (h >> 8) & 0xFF;
                let p = |i: u64| f.rd(at + 4 * (1 + i));
                if matches!(opc, op::DRAW_INDIRECT | op::DRAW_INDEX_INDIRECT) {
                    let (vtx_loc, inst_loc) = (p(1) & 0xFFFF, p(1) >> 16);
                    let inst_loc = if inst_loc != 0 {
                        inst_loc
                    } else {
                        let r = ScanRegs { sh: &st.regs.sh, ctx: &st.regs.ctx };
                        crate::hle::libs::agc::instance_offset_slot(super::shaders::ngg_addr_of(&r)).map_or(0, |s| 0x8C + s)
                    };
                    for r in [vtx_loc, inst_loc] {
                        if r != 0 && !end_unknown.contains(&r) {
                            end_unknown.push(r);
                        }
                    }
                }
                if matches!(opc, op::DRAW_INDEX_INDIRECT_MULTI | op::DRAW_INDIRECT_MULTI) {
                    let (vtx_loc, inst_loc, id_loc) = (p(1) & 0xFFFF, p(2) & 0xFFFF, p(3) & 0xFFFF);
                    let inst_loc = if inst_loc != 0 {
                        inst_loc
                    } else {
                        let r = ScanRegs { sh: &st.regs.sh, ctx: &st.regs.ctx };
                        crate::hle::libs::agc::instance_offset_slot(super::shaders::ngg_addr_of(&r)).map_or(0, |s| 0x8C + s)
                    };
                    for r in [vtx_loc, inst_loc, if p(3) & (1 << 31) != 0 { id_loc } else { 0 }] {
                        if r != 0 && !end_unknown.contains(&r) {
                            end_unknown.push(r);
                        }
                    }
                }
                Self::apply_state(&mut st, opc, n, &p);
                c += 4 * (n + 1);
            }
            st
        });
        PlanChunks { main_n, firsts, starts, states, end_state, end_unknown, sig }
    }

    /// The stop "next run" of a stretch recorded past a run: the next run planned now, before
    /// the run before it is collected (`past_settle` waits for its chunks), as `par_plan`
    /// would plan it then: its scan, its split and its chunks' start states, which read only
    /// the command stream and this command processor's state (exact: a check mode made each
    /// plan again after the collect, 41,473 runs at the heavy Castlewall view, the same chunk
    /// starts and states in all; FINDINGS.md 2026-11-18). What depends on the run before
    /// being collected (label waits' effects on the view, fast clears, homes, indirect
    /// arguments' reads, the batch before the run, the workers) stays in `par_plan`. Returns
    /// whether the run splits (the main graphics command processor's rule).
    fn plan_ahead(&mut self, f: &Fetched, from: u64, end: u64) -> bool {
        let t = Instant::now();
        self.pre_plan = None;
        let scan = self.plan_scan(f, from, end);
        let n = scan.ends.len();
        let par_min = super::par::graphics_min();
        let chunk_min = (par_min / 3).clamp(16, PAR_CHUNK_MIN);
        let chunks = ((super::par::workers() + 1) * super::par::split()).min(n / chunk_min);
        if n < par_min || chunks < 2 {
            self.par_skip = (end, scan.run_end);
            self.par_skip_n = n;
            return false;
        }
        let start = self.chunk_state();
        let chunks = self.plan_chunks(f, from, &scan.ends, scan.run_end, n, chunks, chunk_min, scan.sig, false, false);
        self.pre_plan = Some(Box::new(PrePlan { from, end, buf: self.cur_buf, sub: self.sub_index, start, scan, chunks }));
        PLAN_AHEAD[0].fetch_add(1, Ordering::Relaxed);
        PLAN_AHEAD[3].fetch_add(t.elapsed().as_nanos() as u64, Ordering::Relaxed);
        true
    }

    /// The plan made ahead for the run at `from` (`plan_ahead`) if it is this run's: in the
    /// same command buffer and submission, not held, from the same state.
    fn take_pre_plan(&mut self, from: u64, end: u64, held_run: bool) -> Option<Box<PrePlan>> {
        let p = self.pre_plan.take()?;
        if p.from == from && p.end == end && p.buf == self.cur_buf && p.sub == self.sub_index && !held_run && chunk_states_equal(&p.start, &self.chunk_state(), &[]) {
            PLAN_AHEAD[1].fetch_add(1, Ordering::Relaxed);
            Some(p)
        } else {
            PLAN_AHEAD[2].fetch_add(1, Ordering::Relaxed);
            None
        }
    }

    /// Splits the synchronization-free run that starts with the command at
    /// `from` into chunks when it is long enough: the first stays here, the
    /// others go to the workers' queue now (`gpu::par`).
    fn par_plan(&mut self, f: &Fetched, from: u64, end: u64) {
        if self.compute && super::tune::cp_classes() && self.name.strip_prefix("gpu-acb").and_then(|h| u64::from_str_radix(h.trim_start_matches("0x"), 16).ok()) == Some(super::tune::cp_classes_acb()) {
            self.handshake_census(f, from, end);
        }
        // (Profile: `tune cp_classes`, `PLAN_PHASES`.)
        let mut lap = PlanLaps(super::tune::cp_classes().then(Instant::now));
        let k = if self.compute { compute_workers() } else { super::par::workers() };
        // Inside a stretch past a label wait (`held_runs_ok`): a held run (`par::Run::held`).
        let held_run = self.held.is_some();
        // A plan made ahead at the stop "next run" (`plan_ahead`), taken when it is this run's.
        let (scan, ahead) = match self.take_pre_plan(from, end, held_run) {
            Some(p) => {
                let p = *p;
                (p.scan, Some(p.chunks))
            }
            None => (self.plan_scan(f, from, end), None),
        };
        let PlanScan { ends, mut args, passed, own_waits, cp_writes, sync, sig, why, run_end } = scan;
        let n = ends.len();
        super::par::note_run(n, why, self.compute);
        let par_min = if self.compute { (super::tune::par_min_compute() as usize).max(super::par::floor()) } else { super::par::graphics_min() };
        let chunk_min = (par_min / 3).clamp(16, PAR_CHUNK_MIN);
        let chunks = ((k + 1) * super::par::split()).min(n / chunk_min);
        // (A held run's parts within its stretch's share: `fe2::RUN_CHUNKS`.)
        let chunks = if held_run { chunks.min((super::fe2::RUN_CHUNKS as usize).saturating_sub(self.past_wait.as_deref().map_or(0, |w| w.part as usize) + 2)) } else { chunks };
        if n < par_min || chunks < 2 || (held_run && n < super::tune::held_min()) {
            // Not again before execution passes this run's end.
            self.par_skip = (end, run_end);
            self.par_skip_n = n;
            if self.main_gfx && super::tune::cp_classes() {
                self.run_watch = Some((run_end, super::precache::epoch() as u32, Instant::now()));
                RUN_EPOCHS[0].fetch_add(1, Ordering::Relaxed);
            }
            lap.mark(0);
            PLAN_PHASES[1].fetch_add(1, Ordering::Relaxed);
            return;
        }
        // Held: not planned where planning would stop the stretch (it is recorded by this
        // command processor alone, as before, until its own stop): fast clears pending (a run
        // applies them at its start), indirect arguments its held work writes or that its
        // label's writer wrote before (`past_conflicts_hit`).
        if held_run {
            let a = self.past_wait.as_deref().map_or(0, |w| w.a);
            let skip = super::images::with(|im| !im.pending_clear_keys().is_empty())
                || (!args.is_empty() && (past_conflicts_hit(a, &args) || self.exec.as_ref().is_some_and(|e| e.held_writes_now().overlaps_any(&args))));
            if skip {
                self.par_skip = (end, run_end);
                self.par_skip_n = n;
                HELD_RUNS[10].fetch_add(1, Ordering::Relaxed);
                return;
            }
            // Its own fast clears whose values are known are applied now, before the run (as
            // the pending ones are at a run's start, below: its workers would stop at their
            // images' first uses).
            if let Some(e) = self.exec.as_mut() {
                let n = e.resolve_own_held_clears();
                HELD_RUNS[12].fetch_add(n as u64, Ordering::Relaxed);
            }
        }
        lap.mark(2);
        // The label waits the run holds take their effect on this queue's
        // view now, as each would when reached (`end_copy_scope_after_wait`):
        // the run's commands before one are checked as of a view no older
        // than theirs, and those after it see what the CPU wrote before the
        // label. In the main command processor's own chunk it passes again.
        for &(pk, a, wide) in &passed {
            self.last_wait_kind = b'I';
            self.last_wait_gpu = false;
            self.end_copy_scope_after_wait(a, wide);
            // (Held: read now, checked when the stretch settles, as `held_wait` notes it.)
            if held_run {
                if let Some(e) = self.exec.as_mut() {
                    e.note_held_reads(&[(a, if wide { 8 } else { 4 })], 3);
                }
                if self.past_wait.is_some() {
                    let w = |i: u64| f.rd(pk + 4 * (1 + i));
                    let (reference, mask) = if wide { (addr64(w(3), w(4)), addr64(w(5), w(6))) } else { (w(3) as u64, w(4) as u64) };
                    self.held_polls.push((a, wide, w(0) & 7, mask, reference));
                }
            }
            RUN_PASSED[1].fetch_add(1, Ordering::Relaxed);
        }
        // Pending fast clears are applied now (a worker cannot apply one at
        // the image's first use in program order, so each stopped its chunk:
        // ~4 times per frame in the courtyard). The run's start is a
        // synchronization point; the clear lands before every use in it.
        let clears = super::images::with(|im| im.pending_clear_keys());
        if !clears.is_empty() {
            if let Some(e) = self.exec.as_mut() {
                e.resolve_fast_clears(&clears);
            }
        }
        // Indirect arguments are read when the chunks run: GPU work still
        // writing them completes first.
        if super::homes::any_watched() {
            if let Some(e) = self.exec.as_mut() {
                e.evict_watched_homes();
            }
        }
        if !args.is_empty() && !super::tune::plan_fast() {
            args.sort_unstable();
            let mut merged: Vec<(u64, u64)> = Vec::new();
            for (a, n) in args {
                match merged.last_mut() {
                    Some(m) if a <= m.0 + m.1 + 4096 => m.1 = m.1.max(a + n - m.0),
                    _ => merged.push((a, n)),
                }
            }
            self.sync_reads(&merged);
        } else if !args.is_empty() {
            // The records come from a few arrays, interleaved (sorting
            // thousands per run was ~13% of the graphics CP's busy time at
            // a vista): each extends the open range it continues (one per
            // array), then the few ranges left are sorted and merged; the
            // same ranges as sorting them all (gaps of up to 4 KiB closed).
            let mut open: Vec<(u64, u64)> = Vec::new();
            for (a, n) in args {
                let e = a + n;
                match open.iter_mut().rev().take(8).find(|r| a <= r.1 + 4096 && e + 4096 >= r.0) {
                    Some(r) => {
                        r.0 = r.0.min(a);
                        r.1 = r.1.max(e);
                    }
                    None => open.push((a, e)),
                }
            }
            open.sort_unstable();
            let mut merged: Vec<(u64, u64)> = Vec::new();
            for (a, e) in open {
                match merged.last_mut() {
                    Some(m) if a <= m.0 + m.1 + 4096 => m.1 = m.1.max(e - m.0),
                    _ => merged.push((a, e - a)),
                }
            }
            self.sync_reads(&merged);
        }
        lap.mark(3);
        // (Held: a fast clear or a read of its own held work stopped the stretch here; it
        // settles before the command, `run`.)
        if held_run && (self.exec.as_ref().is_some_and(|e| e.stopped) || self.held.as_ref().is_some_and(|h| h.stop.is_some())) {
            HELD_RUNS[11].fetch_add(1, Ordering::Relaxed);
            return;
        }
        // The chunks: planned ahead, else now.
        let pc = match ahead {
            Some(pc) => pc,
            None => self.plan_chunks(f, from, &ends, run_end, n, chunks, chunk_min, sig, held_run, lap.0.is_some()),
        };
        let PlanChunks { main_n, firsts, starts, states, end_state, end_unknown, sig } = pc;
        lap.mark(4);
        let mut workers = Vec::new();
        for i in 0..k.min(starts.len()) {
            match self.par_pool.worker(i, &self.name, self.compute) {
                Some(w) => workers.push(w),
                None => break,
            }
        }
        if workers.is_empty() {
            return;
        }
        self.drop_ahead();
        // What this queue's batch being recorded writes: the workers cannot
        // see it unsubmitted, so a batch with writes is submitted before
        // they start and they wait for it like for any submitted work (the
        // run handed over first measured level: 2026-10-17, 2026-11-03).
        // (Held: nothing is submitted before the stretch settles; the workers stop on reading
        // what its held work writes, `par::Run::held_writes`.)
        let main_writes = if held_run { Default::default() } else { self.exec.as_ref().map(|e| e.unsubmitted_writes()).unwrap_or_default() };
        if lap.0.is_some() && !main_writes.is_empty() {
            // (Profile: how much of them the forwarded copies cover.)
            let ranges: Vec<(u64, u64)> = main_writes.iter().collect();
            let left = super::forward::uncovered(&ranges);
            PLAN_MAIN_WRITES[0].fetch_add(1, Ordering::Relaxed);
            PLAN_MAIN_WRITES[1].fetch_add(left.as_ref().is_some_and(|v| v.is_empty()) as u64, Ordering::Relaxed);
            PLAN_MAIN_WRITES[2].fetch_add(ranges.len() as u64, Ordering::Relaxed);
            PLAN_MAIN_WRITES[3].fetch_add(left.map_or(ranges.len(), |v| v.len()) as u64, Ordering::Relaxed);
        }
        lap.mark(7);
        if !main_writes.is_empty() {
            // (A batch with no command that writes only labels and stamps goes
            // on as this command processor's own part of the run: a batch,
            // its submission and a batch start less; since 2026-11-07 night:
            // the vista x1.006, Doran x1.008, FINDINGS.md.)
            if self.main_gfx && !self.compute && self.exec.as_ref().is_some_and(|e| e.only_effect_writes()) {
                PRE_RUN_KEPT.fetch_add(1, Ordering::Relaxed);
            } else {
                self.submit_gpu("parallel run");
            }
        }
        lap.mark(8);
        let queue = states.into_iter().enumerate().map(|(i, st)| (st, starts[i], if i + 1 < starts.len() { starts[i + 1] } else { run_end })).collect();
        let mut run = super::par::Run::new(queue);
        run.timed = true;
        let mut approved: Vec<u64> = passed.iter().map(|w| w.0).chain(own_waits.iter().copied()).collect();
        approved.sort_unstable();
        run.passed_waits = std::sync::Arc::from(approved);
        run.cp_writes = std::sync::Arc::from(cp_writes);
        if held_run {
            if let (Some(e), Some(w)) = (self.exec.as_ref(), self.past_wait.as_deref_mut()) {
                // (The stretch's parts share front end 2's uploads from its first held run on,
                // `Exec::fe2_share`: what it uploaded so far is its part's.)
                if !w.sharing {
                    e.claim_held_uploads(w.share, w.part);
                    w.sharing = true;
                }
                run.held = true;
                run.id = w.share;
                run.part_base = w.part;
                run.held_writes = Arc::new(e.held_writes_now());
                run.held_keys = Arc::new(e.held_note_keys());
                w.runs += 1;
            }
            let (share, part) = self.past_wait.as_deref().map_or((0, 0), |w| (w.share, w.part));
            self.fe2_set_run(share, part as usize);
            HELD_RUNS[0].fetch_add(1, Ordering::Relaxed);
            HELD_RUNS[1].fetch_add(starts.len() as u64, Ordering::Relaxed);
            HELD_RUNS[8].fetch_add((n - main_n) as u64, Ordering::Relaxed);
        }
        if self.main_gfx && !self.compute && super::hscensus::on() && !held_run {
            run.hs_seg = super::hscensus::next_segment();
            if let Some(e) = self.exec.as_mut() {
                e.hs_seg = run.hs_seg;
            }
        }
        // (Its workers check as of this command processor's view now.)
        run.fe2_view = self.fe2_view.as_ref().map_or(super::fe2::NO_VIEW, |c| c.load(Ordering::Acquire));
        let run = Arc::new(run);
        // Its own part, queued before the workers' chunks: part 0 of the run
        // (not past a run or held: then it has none before them).
        if self.past.is_none() && self.held.is_none() {
            self.fe2_set_run(run.id, 0);
        }
        lap.mark(9);
        for w in &workers {
            w.give(run.clone());
        }
        self.par_pool.wake();
        lap.mark(10);
        super::par::RUNS.fetch_add(1, Ordering::Relaxed);
        super::par::CHUNKS.fetch_add(starts.len() as u64, Ordering::Relaxed);
        let buf = self.cur_buf;
        self.par = Some(ParRun { starts, end: run_end, next: 0, run, waited_ns: 0, sync, sig, n, firsts, writes: Default::default(), end_state, end_unknown, buf, sub: self.sub_index, held: held_run, unheld: None });
        lap.mark(5);
        PLAN_PHASES[6].fetch_add(1, Ordering::Relaxed);
    }

    /// The main command processor reached the next chunk of the parallel
    /// run: its own work so far goes to the GPU, then the chunk's batches
    /// (its worker submits them once given the turn, or did when the chunk
    /// before ran to its end); execution continues with the chunk's end
    /// state from its end, or from the packet where the worker stopped.
    /// Returns where to continue.
    fn par_collect(&mut self) -> u64 {
        if self.par.as_ref().is_some_and(|p| p.held) {
            return self.held_collect();
        }
        // (Its own part of the run ends here.)
        self.fe2_set_run(0, 0);
        if self.past_ok() {
            return self.past_start();
        }
        let (run, i, to, last) = {
            let pr = self.par.as_mut().unwrap();
            let i = pr.next;
            pr.next += 1;
            if pr.sync {
                self.fence_pending = true;
            }
            (pr.run.clone(), i, if i + 1 < pr.starts.len() { pr.starts[i + 1] } else { pr.end }, pr.next == pr.starts.len())
        };
        self.submit_gpu("par");
        run.grant(i);
        let tw = super::timeline::wait_start();
        let (done, waited) = run.collect(i);
        tw.end(&self.name, || "wait: parallel chunk".into());
        // (Before the run ends below.)
        let cost_key = self.par.as_ref().filter(|p| p.sig != 0 && !done.costs.is_empty()).and_then(|p| Some((p.sig, p.n, *p.firsts.get(i)?)));
        match self.par.as_mut() {
            Some(pr) if last => {
                if self.main_gfx && super::runprobe::on() {
                    pr.writes.extend(&done.writes);
                    super::runprobe::run_end(pr.waited_ns + waited, std::mem::take(&mut pr.writes));
                }
                self.par = None;
            }
            Some(pr) => {
                pr.waited_ns += waited;
                if !done.writes.is_empty() {
                    pr.writes.extend(&done.writes);
                }
            }
            None => {}
        }
        if done.stopped.is_some() {
            super::par::note_stop(done.why);
        }
        if let Some((sig, n, first)) = cost_key {
            self.note_chunk_costs(sig, n, first, &done.costs);
        }
        // This queue's later work and CP effects follow the chunk's batches
        // (a label written after the run must wait for them).
        if let (Some(seq), Some(e)) = (done.last, self.exec.as_mut()) {
            e.depend_on(seq, false);
        }
        self.set_chunk_state(&done.state);
        match done.stopped {
            Some(s) => {
                super::par::STOPS.fetch_add(1, Ordering::Relaxed);
                s
            }
            None => to,
        }
    }

    /// The command processor reached the next chunk of a held run (`par::Run::held`): its own
    /// batch so far is closed (a segment of the stretch's commit, `held_commit`), the chunk's
    /// record awaited (its worker posts it once recorded) and merged into its held executor
    /// (`Exec::merge_held_chunk`); then it goes on as `par_collect` does, as the stretch's next
    /// part (`Exec::fe2_share`). A failed chunk (its batches dropped) is recorded here from its
    /// start, the rest of the run cancelled. Returns where to continue.
    fn held_collect(&mut self) -> u64 {
        let (run, i, from, to, last, unheld) = {
            let pr = self.par.as_mut().unwrap();
            let i = pr.next;
            pr.next += 1;
            (pr.run.clone(), i, pr.starts[i], if i + 1 < pr.starts.len() { pr.starts[i + 1] } else { pr.end }, pr.next == pr.starts.len(), pr.unheld.is_some())
        };
        // Its own work before the chunk: closed, a segment of the stretch's commit (or, the
        // stretch settled, to the GPU).
        let marker = if unheld {
            self.submit_gpu("par");
            0
        } else {
            self.exec.as_mut().map_or(0, |e| e.close_held())
        };
        let tw = super::timeline::wait_start();
        let (done, waited) = run.collect_held(i);
        tw.end(&self.name, || "wait: held chunk".into());
        HELD_RUNS[4].fetch_add(waited, Ordering::Relaxed);
        let cost_key = self.par.as_ref().filter(|p| p.sig != 0 && !done.costs.is_empty()).and_then(|p| Some((p.sig, p.n, *p.firsts.get(i)?)));
        let mut taken: Option<Box<Unheld>> = None;
        if last {
            taken = self.par.take().and_then(|p| p.unheld);
        } else if let Some(pr) = self.par.as_mut() {
            pr.waited_ns += waited;
        }
        if let Some((sig, n, first)) = cost_key {
            self.note_chunk_costs(sig, n, first, &done.costs);
        }
        if done.stopped.is_some() {
            super::par::note_stop(done.why);
            super::par::STOPS.fetch_add(1, Ordering::Relaxed);
            HELD_RUNS[2].fetch_add(1, Ordering::Relaxed);
        }
        if unheld {
            return self.unheld_collect(&run, i, from, to, done, taken);
        }
        let failed = done.held.is_none();
        if let Some(w) = self.past_wait.as_deref_mut() {
            w.segs.push(HeldSeg::Cp(marker));
            if let Some(h) = done.held.as_deref().filter(|h| h.batches > 0) {
                w.segs.push(HeldSeg::Chunk(done.worker, h.batches));
                if w.counts.len() <= done.worker {
                    w.counts.resize(done.worker + 1, 0);
                }
                w.counts[done.worker] += h.batches;
            }
            w.part = run.part_base + i as u32 + 1;
        }
        if let (Some(h), Some(e)) = (done.held.as_deref(), self.exec.as_mut()) {
            e.merge_held_chunk(h);
        }
        if failed {
            // (Later chunks may have left pages to its claims: they go too, `held_cancelled`
            // takes a new share.)
            if self.par.is_some() {
                self.par_cancel();
            } else {
                self.held_cancelled();
            }
        }
        let (share, part) = self.past_wait.as_deref().map_or((0, 0), |w| (w.share, w.part));
        self.fe2_set_run(share, part as usize);
        self.set_chunk_state(&done.state);
        match done.stopped {
            Some(s) => s,
            None => to,
        }
    }

    /// A chunk of a held run whose stretch settled (`Unheld`), checked as the stretch was: its
    /// CPU reads against what other queues wrote during the stretch, its images' pending clears
    /// and reloads, its placed ranges at the label's epoch (pages written since its checks
    /// uploaded first, `tune past_repair`). Then its batches go to the GPU (its image notes to
    /// the image cache); else they are dropped, the rest of the run cancelled (later chunks may
    /// have left pages to its uploads), and the chunk is recorded here from its start (`from`).
    /// `taken`: the run's state when this was its last chunk. Returns where to continue.
    fn unheld_collect(&mut self, run: &super::par::Run, i: usize, from: u64, to: u64, done: super::par::Done, taken: Option<Box<Unheld>>) -> u64 {
        HELD_RUNS[13].fetch_add(1, Ordering::Relaxed);
        let mut verdict: Result<Vec<(u64, u64)>, &'static str> = Err("failed");
        if let Some(h) = done.held.as_deref() {
            let u: Option<&Unheld> = match taken.as_deref() {
                Some(u) => Some(u),
                None => self.par.as_ref().and_then(|p| p.unheld.as_deref()),
            };
            if let (Some(u), Some(e)) = (u, self.exec.as_mut()) {
                verdict = if h.reads.iter().any(|&(a, n)| u.writes.overlaps(a, n)) {
                    Err("read what other queues wrote")
                } else if !h.keys.is_empty() && super::images::any_pending() && {
                    let keys: Vec<super::images::ImageKey> = h.keys.iter().copied().collect();
                    super::images::with(|im| im.clears_pending(&keys) || im.overwrites_pending(&keys))
                } {
                    Err("images with pending clears or reloads")
                } else if let Some(need) = u.need.filter(|&n| h.want_min < n) {
                    e.chunk_stale_for(&h.ranges, &h.up, &u.up, need).ok_or("a check failed")
                } else {
                    Ok(Vec::new())
                };
            }
        }
        // Pages written since its checks: uploaded first (`tune past_repair`).
        if let Ok(stale) = &verdict {
            if !stale.is_empty() && !(super::tune::past_repair() && self.exec.as_mut().is_some_and(|e| e.repair_now(stale))) {
                verdict = Err("checks older than its label");
            }
        }
        match (verdict, done.held.as_deref()) {
            (Ok(_), Some(h)) => {
                if !h.clears.is_empty() || !h.over.is_empty() {
                    super::images::with(|im| im.apply_held(&h.clears, &h.over));
                }
                let seq = if h.batches > 0 { self.par_pool.get(done.worker).and_then(|w| w.with_cp(|c| c.submit_closed_first(h.batches))) } else { None };
                if let (Some(s), Some(e)) = (seq, self.exec.as_mut()) {
                    e.depend_on(s, false);
                }
                let part = run.part_base + i as u32 + 1;
                self.fe2_set_run(if self.par.is_some() { run.id } else { 0 }, part as usize);
                self.set_chunk_state(&done.state);
                match done.stopped {
                    Some(s) => s,
                    None => to,
                }
            }
            (r, _) => {
                HELD_RUNS[14].fetch_add(1, Ordering::Relaxed);
                note_why(&HELD_DROPS, match r {
                    Err(w) => w,
                    Ok(_) => "failed",
                });
                // (Its batches: the first ones its worker's executor holds; the rest of the run's
                // follow them.)
                if self.par.is_some() {
                    self.par_cancel();
                } else {
                    self.held_truncate(&[]);
                }
                self.fe2_set_run(0, 0);
                from
            }
        }
    }

    /// Whether a parallel run may be planned while recording past a label wait (`tune
    /// held_runs`): its chunks are recorded held (`par::Run::held`).
    fn held_runs_ok(&self) -> bool {
        super::tune::held_runs()
            && self.past.is_none()
            && self.past_wait.as_deref().is_some_and(|w| w.share != 0 && (w.part as usize) + 4 < super::fe2::RUN_CHUNKS as usize)
            && self.held.as_ref().is_some_and(|h| h.stop.is_none())
            && !self.exec.as_ref().is_some_and(|e| e.stopped)
    }

    /// A held run was cancelled (its workers left it): its chunks posted but not collected
    /// drop their batches (after the collected ones in each worker's executor), and the stretch
    /// takes a new share from here (their claims stay under the old one).
    fn held_cancelled(&mut self) {
        HELD_RUNS[5].fetch_add(1, Ordering::Relaxed);
        let counts = self.past_wait.as_deref().map(|w| w.counts.clone()).unwrap_or_default();
        self.held_truncate(&counts);
        if let Some(w) = self.past_wait.as_deref_mut() {
            w.share = super::par::new_run_id();
            w.part = 0;
        }
        let share = self.past_wait.as_deref().map_or(0, |w| w.share);
        // (What the stretch uploaded so far is the new share's first part.)
        if let Some(e) = self.exec.as_ref() {
            e.claim_held_uploads(share, 0);
        }
        self.fe2_set_run(share, 0);
    }

    /// Drops the closed batches of each worker's executor after the first `counts[i]` (none
    /// kept past `counts`' end).
    fn held_truncate(&self, counts: &[usize]) {
        for i in 0..self.par_pool.count() {
            let keep = counts.get(i).copied().unwrap_or(0);
            if let Some(w) = self.par_pool.get(i) {
                w.with_cp(|c| {
                    if c.closed_count() > keep {
                        c.discard_closed_tail(keep);
                    }
                });
            }
        }
    }

    /// The stretch's held work goes to the GPU in program order (`Exec::past_commit` with held
    /// runs): the command processor's closed batches up to each marker, each collected chunk's
    /// batches from its worker's executor (this queue's later work follows them), then the
    /// rest of its own.
    fn held_commit(&mut self, segs: &[HeldSeg], counts: &[usize]) {
        let Some(e) = self.exec.as_mut() else { return };
        e.past_commit_begin();
        e.submit_repair_first();
        let mut own = 0usize;
        for s in segs {
            match *s {
                HeldSeg::Cp(marker) => {
                    if marker > own {
                        if let Some(e) = self.exec.as_mut() {
                            e.submit_closed_first(marker - own);
                        }
                        own = marker;
                    }
                }
                HeldSeg::Chunk(wi, k) => {
                    let seq = self.par_pool.get(wi).and_then(|w| w.with_cp(|c| c.submit_closed_first(k)));
                    HELD_RUNS[3].fetch_add(k as u64, Ordering::Relaxed);
                    if let (Some(s), Some(e)) = (seq, self.exec.as_mut()) {
                        e.depend_on(s, false);
                    }
                }
            }
        }
        if let Some(e) = self.exec.as_mut() {
            e.submit_closed();
        }
        // (Nothing should be left in the workers' executors: a cancelled run's batches were
        // dropped at its cancel. A held run being collected keeps its chunks' there.)
        for (i, &c) in counts.iter().enumerate().filter(|_| self.par.is_none()) {
            if c > 0 {
                if let Some(w) = self.par_pool.get(i) {
                    w.with_cp(|cp| {
                        let left = cp.closed_count();
                        if left > 0 {
                            HELD_RUNS[9].fetch_add(left as u64, Ordering::Relaxed);
                            cp.discard_closed_tail(0);
                        }
                    });
                }
            }
        }
    }

    /// Whether a wait on `a` that does not pass at once is gone past: the
    /// graphics command processor at command-buffer depth 1 with nothing
    /// held, the label last written by queue 0x50 (since 2026-10-27; any
    /// compute queue's, or none, were the switch `past_waits` removed on
    /// 2026-11-01).
    fn past_wait_ok(&self, a: u64) -> bool {
        self.main_gfx
           
            && !self.settling
            && self.held.is_none()
            && self.past.is_none()
            && self.past_wait.is_none()
            && self.par.is_none()
            && self.depth == 1
            && self.exec.is_some()
            && !super::capture::capturing()
            && last_write(a).is_some_and(|(q, _, _)| q == "gpu-acb0x50")
            && past_backoff_pass(a)
    }

    /// Why a wait on a label queue 0x50 wrote last is not gone past (`past_wait_ok`'s first
    /// condition that fails; None: not the graphics CP's, or not 0x50's label).
    fn past_wait_not(&self, a: u64) -> Option<&'static str> {
        if !self.main_gfx || !last_write(a).is_some_and(|(q, _, _)| q == "gpu-acb0x50") {
            return None;
        }
        Some(if self.settling {
            "settling a stretch"
        } else if self.past.is_some() {
            "recording past a run"
        } else if self.past_wait.is_some() {
            "in a stretch already"
        } else if self.held.is_some() {
            "held otherwise"
        } else if self.par.is_some() {
            "in a parallel run"
        } else if self.depth != 1 {
            "nested command buffer"
        } else if self.exec.is_none() || super::capture::capturing() {
            "no executor or capturing"
        } else {
            "backoff after drops"
        })
    }

    /// Goes past a label wait (`past_wait_ok`): this command processor's
    /// work so far to the GPU, its state noted, then on from the packet after
    /// the wait with its executor held (`Exec::past_begin`) until the label
    /// is promised or a stop (`past_wait_settle`).
    fn past_wait_start(&mut self, control: u32, a: u64, wide: bool, reference: u64, mask: u64, at: u64) -> Flow {
        self.submit_gpu("wait");
        self.held_polls.clear();
        self.past_wait = Some(Box::new(PastWait {
            control,
            a,
            wide,
            reference,
            mask,
            at,
            buf: self.cur_buf,
            sub: self.sub_index,
            state: self.chunk_state(),
            stack: self.ctx_stack.clone(),
            markers: self.markers.clone(),
            pred: self.pred.clone(),
            fence_pending: self.fence_pending,
            began: Instant::now(),
            ticks: 0,
            share: if super::tune::held_runs() && held_backoff_pass(a) { super::par::new_run_id() } else { 0 },
            part: 0,
            segs: Vec::new(),
            counts: Vec::new(),
            runs: 0,
            sharing: false,
        }));
        self.held = Some(Box::new(HeldCp { stop: None, commands: 0 }));
        label_event(&self.name, b'G', a, 0, reference, at);
        super::exec::journal_begin();
        if let Some(e) = self.exec.as_mut() {
            e.past_begin_growing();
        }
        PAST_WAITS[0].fetch_add(1, Ordering::Relaxed);
        Flow::Next
    }

    /// Whether the label gone past passes now (in memory or promised).
    fn past_wait_passes(&self) -> bool {
        let Some(w) = self.past_wait.as_deref() else { return false };
        let func = w.control & 7;
        let v = if w.wide { unsafe { rd64(w.a) } } else { (unsafe { rd32(w.a) }) as u64 };
        compare(func, v & w.mask, w.reference & w.mask) || promised(w.a, w.wide, func, w.mask, w.reference).is_some()
    }

    /// Recording past a label wait stopped, or the label was promised: the
    /// label passes (waited for here after a stop), then what was recorded
    /// past the wait is checked: no CPU read of what the batches submitted
    /// and the CP writes recorded meanwhile write (`exec::journal_end`), no
    /// image used with a fast clear or reload pending. Then it goes to the
    /// GPU after the label's writer (`Exec::past_commit`) and execution goes
    /// on at the stop; else it is dropped (`Exec::past_drop`) and execution
    /// goes back to the wait, which passes now. Returns where.
    fn past_wait_settle(&mut self) -> Option<u64> {
        // (A run held planning passed over, `tune held_runs`, may be planned once the stretch
        // settled: its label passed, its clears applied at its start.)
        if super::tune::held_runs() {
            self.par_skip = (0, 0);
        }
        let mut w = self.past_wait.take().unwrap();
        let held = self.held.take();
        let (why, at) = held.as_ref().and_then(|h| h.stop).unwrap_or(("stopped", w.at));
        let commands = held.as_ref().map_or(0, |h| h.commands);
        let promised_first = why == "promised";
        let t = Instant::now();
        self.settling = true;
        let flow = self.wait(w.control, w.a, w.wide, w.reference, w.mask, w.at);
        self.settling = false;
        let waited = t.elapsed();
        // The batch that promised the label, when it is not complete yet: the stretch's first
        // submission waits for it (`exec::commit_after`).
        let writer = promised(w.a, w.wide, w.control & 7, w.mask, w.reference).map_or(0, |p| p.0);
        // Front end 2: the stretch was checked before its label's writer
        // synchronized for the last time ahead of the label, and the CPU may
        // have written pages it bound in between (the title's per-frame
        // blocks, written after the graphics queue's submission and before
        // the compute queue's whose label it waits on): then it is checked
        // again as of the epoch its label needs (`tune past_check`), and
        // recorded again when a page was written since (`DS_FE2_STRETCH_AUDIT=1`:
        // counted, kept).
        let needs = (super::fe2::on() && (super::tune::past_check() || super::fe2::stretch_audit())).then(|| self.label_needs(w.a, w.wide));
        let audit = super::fe2::stretch_audit();
        let writes = super::exec::journal_end();
        PAST_WAITS[3].fetch_add(commands, Ordering::Relaxed);
        PAST_WAITS[4].fetch_add(w.began.elapsed().as_nanos() as u64, Ordering::Relaxed);
        if promised_first {
            PAST_WAITS[5].fetch_add(1, Ordering::Relaxed);
        } else {
            PAST_WAITS[6].fetch_add(1, Ordering::Relaxed);
            PAST_WAITS[7].fetch_add(waited.as_nanos() as u64, Ordering::Relaxed);
            note_why(&PAST_WAIT_STOPS, why);
            if PAST_WAITS[6].load(Ordering::Relaxed) <= 40 && valid(at, 4) {
                let op = (unsafe { rd32(at) } >> 8) & 0xFF;
                // The packets from the wait to the stop (same buffer only).
                let mut seq = Vec::new();
                if w.sub == self.sub_index && w.buf == self.cur_buf && w.at < at && at - w.at < 4096 {
                    let mut c = w.at;
                    while c < at && seq.len() < 24 && valid(c, 4) {
                        let h = unsafe { rd32(c) };
                        if h >> 30 != 3 {
                            break;
                        }
                        let n = ((h >> 16) & 0x3FFF) as u64 + 1;
                        let o = (h >> 8) & 0xFF;
                        seq.push(if o == op::WRITE_DATA && valid(c + 8, 8) { format!("WRITE_DATA->{:#x}", addr64(unsafe { rd32(c + 8) }, unsafe { rd32(c + 12) })) } else { op_name(o) });
                        c += 4 * (n + 1);
                    }
                }
                let writer = last_write(w.a).map(|x| x.0).unwrap_or_else(|| "cpu".into());
                crate::info!("gpu[{}]: past the wait at {:#x} on {:#x} (written by {writer}): stopped ({why}) at {at:#x} (opcode {op:#x}) after {commands} commands and {} us, {} submission(s) on; waited {} us for the label; packets: {}", self.name, w.at, w.a, w.began.elapsed().as_micros(), self.sub_index - w.sub, waited.as_micros(), seq.join(" "));
            }
        }
        // Label waits the stretch passed on memory (`held_polls`): a write of such a label by a
        // batch submitted while the stretch ran (the value read after it landed) is no conflict
        // while the wait passes on memory now and nothing unfinished writes the label: the
        // console's command processor, reaching the wait here, passes it the same.
        let polls = std::mem::take(&mut self.held_polls);
        let e = self.exec.as_mut().unwrap();
        let conflict = writes.overlaps_any(e.held_reads()) && {
            let mut kept = 0u64;
            let mut real = false;
            for r in e.held_reads().iter().filter(|r| writes.overlaps(r.0, r.1)) {
                let good = polls.iter().any(|&(a, wide, func, mask, reference)| {
                    a == r.0 && r.1 == if wide { 8 } else { 4 } && valid(a, 8) && {
                        let v = if wide { unsafe { rd64(a) } } else { (unsafe { rd32(a) }) as u64 };
                        compare(func, v & mask, reference & mask) && !super::exec::unfinished_write(a & !3, r.1)
                    }
                });
                if good {
                    kept += 1;
                } else {
                    real = true;
                    break;
                }
            }
            if !real {
                PAST_POLLS[0].fetch_add(kept, Ordering::Relaxed);
                PAST_POLLS[1].fetch_add(1, Ordering::Relaxed);
            }
            real
        };
        let dropped = if matches!(flow, Flow::Stop) {
            Some("shutdown")
        } else if conflict {
            Some("read what other queues wrote")
        } else if !e.held_keys().is_empty() && super::images::any_pending() && {
            let keys: Vec<super::images::ImageKey> = e.held_keys().iter().copied().collect();
            super::images::with(|im| im.clears_pending(&keys) || im.overwrites_pending(&keys))
        } {
            Some("images with pending clears or reloads")
        } else if let Some((_, why)) = needs.filter(|&(n, _)| e.past_checks_want() < n && !e.past_current_for(n) && !audit && !(super::tune::past_repair() && e.past_repair())) {
            Some(why)
        } else if super::tune::past_drop() {
            Some("dropped (past_drop)")
        } else {
            None
        };
        label_event(&self.name, if dropped.is_some() { b'D' } else { b'C' }, w.a, commands, w.reference, at);
        if let Some(d) = dropped {
            note_why(&PAST_WAIT_DROPS, d);
            if d == "read what other queues wrote" {
                // The next stretch on this label stops before these reads.
                let hits: Vec<(u64, u64)> = e.held_reads().iter().copied().filter(|r| writes.overlaps(r.0, r.1)).take(PAST_CONFLICTS_MAX).collect();
                past_conflicts_note(w.a, &hits);
            }
            if d == "read what other queues wrote" && super::exec::trace_past_drops() {
                static N: AtomicU64 = AtomicU64::new(0);
                if N.fetch_add(1, Ordering::Relaxed) < super::exec::trace_past_drops_max() {
                    let hits: Vec<String> = e
                        .held_reads()
                        .iter()
                        .enumerate()
                        .filter(|(_, r)| writes.overlaps(r.0, r.1))
                        .take(3)
                        .map(|(i, r)| format!("{} {:#x}+{:#x} written by {}", e.held_read_kind(i), r.0, r.1, super::exec::journal_writers(r.0, r.1).join(", ")))
                        .collect();
                    crate::info!("gpu[{}]: stretch past the wait at {:#x} on {:#x} dropped after {commands} commands ({} of its {} CPU reads written meanwhile): {}", self.name, w.at, w.a, e.held_reads().iter().filter(|r| writes.overlaps(r.0, r.1)).count(), e.held_reads().len(), hits.join("; "));
                }
            }
            e.past_drop();
            PAST_WAITS[2].fetch_add(1, Ordering::Relaxed);
            // (A held run being collected is cancelled: its chunks lie after the wait. Held
            // runs' chunks: their batches in the workers' executors.)
            if self.par.as_ref().is_some_and(|p| p.held) {
                self.par_cancel();
            }
            if w.counts.iter().any(|&c| c > 0) {
                self.held_truncate(&[]);
            }
            if w.share != 0 {
                self.fe2_set_run(0, 0);
            }
            if w.runs > 0 {
                HELD_RUNS[7].fetch_add(1, Ordering::Relaxed);
            }
            if d != "shutdown" {
                if w.runs > 0 {
                    held_backoff_note(w.a, true);
                } else {
                    past_backoff_note(w.a, true);
                }
            }
            // Back to the wait.
            self.set_chunk_state(&w.state);
            self.ctx_stack = std::mem::take(&mut w.stack);
            self.markers = std::mem::take(&mut w.markers);
            self.pred = w.pred.take();
            self.fence_pending = w.fence_pending;
            self.drop_ahead();
            if w.sub != self.sub_index {
                self.rewind_sub = Some((w.sub, w.buf, w.at));
                return None;
            }
            if w.buf != self.cur_buf {
                self.rewind = Some((w.buf, w.at));
                return None;
            }
            return Some(w.at);
        }
        // The label passed: front end 2's view as after any wait (the
        // writer's, or the epoch now; the stretch checked as of the epoch
        // at each check), taken before its held batches go to the GPU: the
        // label writes among them are promised with it. (The default front
        // end's scopes stay as before.)
        if super::fe2::on() {
            self.end_copy_scope_after_wait(w.a, w.wide);
        }
        // (A held run being collected goes on: its remaining chunks are checked as they are
        // collected, `unheld_collect`; the stretch's uploaded pages are current for them.)
        let up = if self.par.as_ref().is_some_and(|p| p.held) { self.exec.as_ref().map(|e| e.past_up().clone()) } else { None };
        super::exec::commit_after(writer);
        if w.segs.is_empty() && w.counts.iter().all(|&c| c == 0) {
            self.exec.as_mut().unwrap().past_commit();
        } else {
            self.held_commit(&w.segs, &w.counts);
        }
        super::exec::commit_after(0);
        match (self.par.as_mut(), up) {
            (Some(pr), Some(up)) => {
                pr.unheld = Some(Box::new(Unheld { writes, need: needs.map(|x| x.0), up }));
                self.fe2_set_run(w.share, w.part as usize);
            }
            _ => {
                if w.share != 0 {
                    self.fe2_set_run(0, 0);
                }
            }
        }
        if w.runs > 0 {
            HELD_RUNS[6].fetch_add(1, Ordering::Relaxed);
        }
        PAST_WAITS[1].fetch_add(1, Ordering::Relaxed);
        if w.runs > 0 {
            held_backoff_note(w.a, false);
        }
        past_backoff_note(w.a, false);
        if why == "label" {
            self.submit_gpu("label");
        }
        Some(at)
    }

    /// Settles what was recorded past a parallel run or a label wait.
    fn settle_held(&mut self) -> Option<u64> {
        if self.past_wait.is_some() {
            self.past_wait_settle()
        } else {
            self.past_settle()
        }
    }

    /// Whether to go past the parallel run at its first collect (`tune
    /// past_runs`): the graphics command processor, its end state walked.
    fn past_ok(&self) -> bool {
        super::tune::past_runs()
            && self.main_gfx
            && self.held.is_none()
            && self.exec.is_some()
            && !super::capture::capturing()
            && self.par.as_ref().is_some_and(|p| p.next == 0 && p.end_state.is_some())
    }

    /// Goes past the parallel run: this command processor's work so far to
    /// the GPU and the first chunk's turn given as at a collect, then on
    /// from the run's end with its walked end state, the executor held
    /// (`Exec::past_begin`) until a stop (`past_settle`). Returns the run's
    /// end.
    fn past_start(&mut self) -> u64 {
        let mut pr = self.par.take().unwrap();
        if pr.sync {
            self.fence_pending = true;
        }
        if self.shape() {
            crate::info!("cb shape: past a run of {} chunks from {:#x} to {:#x}, depth {}", pr.starts.len(), pr.starts.first().copied().unwrap_or(0), pr.end, self.depth);
        }
        self.submit_gpu("par");
        pr.run.grant(0);
        let end_state = pr.end_state.take().unwrap();
        let end_unknown = std::mem::take(&mut pr.end_unknown);
        let end = pr.end;
        self.set_chunk_state(&end_state);
        let across = self.depth == 1;
        self.past = Some(Box::new(Past { pr, end_state, end_unknown, across, stack: self.ctx_stack.clone(), markers: self.markers.clone(), pred: self.pred.clone(), fence_pending: self.fence_pending, began: Instant::now() }));
        self.held = Some(Box::new(HeldCp { stop: None, commands: 0 }));
        if let Some(e) = self.exec.as_mut() {
            e.past_begin();
        }
        super::par::PAST[0].fetch_add(1, Ordering::Relaxed);
        end
    }

    /// Recording past a run stopped (`HeldCp::stop`): the run's chunks are
    /// collected as `par_collect` does and checked: no chunk stopped, the
    /// end state the same as walked, no CPU read past the run of what the
    /// chunks wrote, no image used past it with a fast clear or reload
    /// pending (the chunks note theirs while they record). Then what was
    /// recorded past the run goes to the GPU after the chunks
    /// (`Exec::past_commit`) and execution goes on at the stop; else it is
    /// dropped (`Exec::past_drop`) and execution goes on from the run's end
    /// (or a stopped chunk's stop, as `par_collect` would). Returns where.
    fn past_settle(&mut self) -> Option<u64> {
        let past = self.past.take().unwrap();
        let held = self.held.take();
        let (why, at) = held.as_ref().and_then(|h| h.stop).unwrap_or(("stopped", past.pr.end));
        let t = Instant::now();
        let first_start = past.pr.starts.first().copied().unwrap_or(0);
        let mut pr = past.pr;
        let mut writes = super::ranges::Ranges::new();
        let mut end_state: Option<ChunkState> = None;
        let mut stopped: Option<(u64, ChunkState)> = None;
        while pr.next < pr.starts.len() {
            let i = pr.next;
            pr.next += 1;
            pr.run.grant(i);
            let tw = super::timeline::wait_start();
            let (done, waited) = pr.run.collect(i);
            tw.end(&self.name, || format!("wait: parallel chunk (settling at {why})"));
            pr.waited_ns += waited;
            if let (Some(seq), Some(e)) = (done.last, self.exec.as_mut()) {
                e.depend_on(seq, false);
            }
            if pr.sig != 0 && !done.costs.is_empty() {
                if let Some(&first) = pr.firsts.get(i) {
                    let (sig, n) = (pr.sig, pr.n);
                    self.note_chunk_costs(sig, n, first, &done.costs);
                }
            }
            writes.extend(&done.writes);
            if let Some(s) = done.stopped {
                super::par::note_stop(done.why);
                super::par::STOPS.fetch_add(1, Ordering::Relaxed);
                stopped = Some((s, done.state));
                break;
            }
            end_state = Some(done.state);
        }
        super::par::PAST[3].fetch_add(held.as_ref().map_or(0, |h| h.commands), Ordering::Relaxed);
        super::par::PAST[4].fetch_add(past.began.elapsed().saturating_sub(t.elapsed()).as_nanos() as u64, Ordering::Relaxed);
        super::par::PAST[5].fetch_add(t.elapsed().as_nanos() as u64, Ordering::Relaxed);
        let e = self.exec.as_mut().unwrap();
        let dropped = if stopped.is_some() {
            Some("a chunk stopped")
        } else if !end_state.as_ref().is_some_and(|s| chunk_states_equal(s, &past.end_state, &past.end_unknown)) {
            Some("end state")
        } else if writes.overlaps_any(e.held_reads()) {
            Some("read the run's writes")
        } else if !e.held_keys().is_empty() && super::images::any_pending() && {
            let keys: Vec<super::images::ImageKey> = e.held_keys().iter().copied().collect();
            super::images::with(|im| im.clears_pending(&keys) || im.overwrites_pending(&keys))
        } {
            Some("images with pending clears or reloads")
        } else if super::tune::past_drop() {
            Some("dropped (past_drop)")
        } else {
            None
        };
        super::par::note_past(why, dropped);
        if dropped.is_none() {
            e.past_commit();
            super::par::PAST[1].fetch_add(1, Ordering::Relaxed);
            if why == "label" {
                self.submit_gpu("label");
            }
            return Some(at);
        }
        e.past_drop();
        super::par::PAST[2].fetch_add(1, Ordering::Relaxed);
        // (Execution goes on from the run's end: the next run is planned again.)
        self.pre_plan = None;
        // Back to where collecting the run would have left it.
        self.ctx_stack = past.stack;
        self.markers = past.markers;
        self.pred = past.pred;
        self.fence_pending = past.fence_pending;
        self.drop_ahead();
        // (The command buffer and title submission the stop or the last
        // run's end is in.)
        let stopped_buf = stopped.as_ref().map(|_| pr.buf);
        let last_buf = pr.buf;
        let run_sub = pr.sub;
        let to = match stopped {
            Some((s, st)) => {
                self.set_chunk_state(&st);
                if pr.next < pr.starts.len() {
                    self.par = Some(pr);
                }
                s
            }
            None => {
                if let Some(st) = &end_state {
                    self.set_chunk_state(st);
                }
                pr.end
            }
        };
        // The run ended in an earlier command buffer: the worker goes back
        // (into an earlier title submission: a stretch past a run that went
        // on into later submissions, `tune past_subs`).
        let buf = if stopped_buf.is_some() { stopped_buf.unwrap() } else { last_buf };
        if run_sub != self.sub_index {
            static N: AtomicU64 = AtomicU64::new(0);
            if N.fetch_add(1, Ordering::Relaxed) < 40 {
                crate::info!("gpu[{}]: stretch past a run dropped ({}; stop: {why}) back into submission {run_sub} at {to:#x} from submission {} (the run gone past from {first_start:#x})", self.name, dropped.unwrap_or("?"), self.sub_index);
            }
            self.rewind_sub = Some((run_sub, buf, to));
            return None;
        }
        if buf != self.cur_buf {
            self.rewind = Some((buf, to));
            return None;
        }
        Some(to)
    }

    /// Folds the per-command costs a chunk measured (its first command:
    /// `first`) into its run's costs (`tune par_cost`): smoothed; a chunk's
    /// first command (it carries the chunk's setup) only when unknown, as
    /// the mean of the others.
    fn note_chunk_costs(&mut self, sig: u64, n: usize, first: usize, costs: &[u32]) {
        let Some(rc) = self.run_costs.get_mut(&sig) else { return };
        if rc.costs.len() != n || costs.len() < 2 {
            return;
        }
        let mut sum = 0.0f64;
        for (j, &c) in costs.iter().enumerate().skip(1) {
            let Some(slot) = rc.costs.get_mut(first + j) else { break };
            let v = c as f32;
            *slot = if slot.is_finite() { 0.5 * *slot + 0.5 * v } else { v };
            sum += c as f64;
        }
        if let Some(slot) = rc.costs.get_mut(first) {
            if !slot.is_finite() {
                *slot = (sum / (costs.len() - 1) as f64) as f32;
            }
        }
    }

    /// Abandons the parallel run in progress (execution left it): the
    /// workers drop the batches of chunks whose turn did not come (none had:
    /// execution leaves a run only from the main command processor's own
    /// chunk or the rest of a stopped one); their chunks are executed here.
    fn par_cancel(&mut self) {
        let Some(pr) = self.par.take() else { return };
        pr.run.cancel();
        super::par::CANCELLED.fetch_add(1, Ordering::Relaxed);
        if pr.held {
            self.held_cancelled();
        }
    }

    /// The executing command's stage preparations made ahead ([vertex,
    /// pixel] for draws, [compute, -] for dispatches), once.
    pub fn take_ahead(&self) -> [Option<Arc<super::ahead::Slot>>; 2] {
        std::mem::take(&mut *self.cur_ahead.borrow_mut())
    }

    /// Drops the preparations made ahead (execution took another path).
    fn drop_ahead(&mut self) {
        for c in self.ahead.drain(..) {
            for s in c.stages.iter().flatten() {
                s.cancel();
            }
        }
    }

    /// The commands after a label wait that blocked (`tune wait_ahead`):
    /// from `from` with this command processor's registers, through
    /// register writes, flushes, cache actions, end-of-pipe events, CP
    /// writes and further label waits (what a stage reads is checked again
    /// when it is used, `ahead::take`), up to any other control packet, the
    /// buffer's end or WAIT_SCAN_MAX commands. A synchronization-free run
    /// long enough for the parallel workers ends it before its first
    /// command (they prepare their own). The stages are prepared by this
    /// command processor while it waits (`wait`), not handed to workers.
    fn scan_through(&mut self, f: &Fetched, from: u64, end: u64) -> Vec<AheadCmd> {
        let (cmds, why, ns) = self.scan_commands(f, from, end, false, false);
        super::ahead::note_wait_scan(cmds.len(), why, ns);
        cmds
    }

    /// A dispatch's pipeline is missing (its program's first use with this
    /// state): the commands after it in its buffer are scanned as at a label
    /// wait (`scan_commands`, through synchronization and long runs) and
    /// their compute stages handed to the compile workers (`gpu::compile`).
    pub fn prefetch_compiles(&self) {
        let Some((base, from, end)) = self.compile_ctx.take() else { return };
        // Read in place (the scan does not copy the buffer).
        let f = Fetched { base, words: Vec::new(), live: false };
        let (cmds, why, ns) = self.scan_commands(&f, from, end, true, false);
        super::compile::note_scan_end(why, cmds.len());
        let slots: Vec<Arc<super::ahead::Slot>> = cmds.into_iter().filter_map(|c| c.stages[0].clone()).collect();
        super::compile::submit(slots, ns);
    }

    /// A draw waits for its missing pipeline (`exec::draw`, `compile::gfx_ahead`): the commands
    /// after it in its buffer (a parallel worker's: in its chunk) are scanned as for a
    /// dispatch's first use, and their draws handed to the compile workers, which prepare their
    /// stages from the registers the scan saw and queue the pipelines missing for the
    /// background graphics workers (`compile::submit_draws`), those of draws that always wait
    /// first: the draws after it find theirs made or in flight instead of compiling them one
    /// after another (2026-11-19: a new player's first walk waited 13.1 s at 125 draws, 73 of
    /// them on the graphics command processor in bursts of 10-21).
    pub fn prefetch_gfx(&self) {
        let Some((base, from, end)) = self.compile_ctx.take() else { return };
        let f = Fetched { base, words: Vec::new(), live: false };
        let (cmds, why, ns) = self.scan_commands(&f, from, end, true, true);
        let found = cmds.len();
        let draws: Vec<_> = cmds
            .into_iter()
            .filter_map(|c| {
                let [vs, ps] = c.stages;
                match (vs, c.gfx) {
                    (Some(vs), Some(g)) if vs.regs().kind == super::shaders::StageKind::Vertex => Some((vs, ps, g)),
                    _ => None,
                }
            })
            .collect();
        // (`DS_COMPILE_TRACE=1`: the first scans, what they found and why they ended.)
        if super::exec::compile_trace_on() {
            static N: std::sync::atomic::AtomicU32 = std::sync::atomic::AtomicU32::new(0);
            if N.fetch_add(1, Ordering::Relaxed) < 40 {
                crate::info!("compile ahead: scan at a waiting draw on {}: {found} commands, {} draws (ended by {why:#x}, {} bytes to the end)", self.name, draws.len(), end.saturating_sub(from));
            }
        }
        super::compile::submit_draws(draws, ns);
    }

    /// Direct dispatches of program `addr` of 1 to `small` workgroups right
    /// after the current command, up to `max`, before a dispatch of another
    /// program, a draw or a call (`exec`: the first
    /// dispatch of a run of them makes the program batchable, else its
    /// first pipeline is made for the run's first dispatches and again
    /// batched: cs 0x8026f4100 2.1 and 2.7 s in a first session).
    pub fn streak_after(&self, addr: u64, small: u64, max: u32) -> u32 {
        let (n, why) = self.streak_scan(addr, small, max);
        if super::exec::compile_trace_on() {
            crate::info!("streak lookahead: cs {addr:#x} {n} more (ended by {why})");
        }
        n
    }

    fn streak_scan(&self, addr: u64, small: u64, max: u32) -> (u32, String) {
        let Some((_, from, end)) = self.compile_ctx.get() else { return (0, "no buffer".into()) };
        let (mut lo, mut hi) = (self.reg(Space::Sh, 0x20C), self.reg(Space::Sh, 0x20D));
        let (mut cur, mut n) = (from, 0u32);
        for _ in 0..256 {
            if n >= max {
                return (n, "the count".into());
            }
            if cur >= end || !valid(cur, 4) {
                return (n, "the buffer's end".into());
            }
            let h = unsafe { rd32(cur) };
            match h >> 30 {
                3 => {}
                2 => {
                    cur += 4;
                    continue;
                }
                _ => return (n, "a header".into()),
            }
            let count = (h >> 16) & 0x3FFF;
            if count == 0x3FFF {
                cur += 4;
                continue;
            }
            let words = count as u64 + 1;
            let next = cur + 4 * (words + 1);
            if h & 1 != 0 || next > end || !valid(cur, 4 * (words + 1)) {
                return (n, "a predicated or cut packet".into());
            }
            let at = cur;
            let p = |i: u64| unsafe { rd32(at + 4 * (1 + i)) };
            let opcode = (h >> 8) & 0xFF;
            match opcode {
                op::SET_SH_REG | op::SET_SH_REG_INDEX => {
                    let first = p(0) & 0xFFFF;
                    for i in 1..words {
                        match first + (i - 1) as u32 {
                            0x20C => lo = p(i),
                            0x20D => hi = p(i),
                            _ => {}
                        }
                    }
                }
                op::DISPATCH_DIRECT | op::DISPATCH_INDIRECT => {
                    let a = ((lo as u64) | ((hi as u64 & 0xFF) << 32)) << 8;
                    if a != addr {
                        return (n, format!("a dispatch of cs {a:#x}"));
                    }
                    if opcode == op::DISPATCH_DIRECT {
                        let g = p(0) as u64 * p(1) as u64 * p(2) as u64;
                        if g > 0 && g <= small {
                            n += 1;
                        }
                    }
                }
                op::LOAD_SH_REG_INDEX => {
                    // As `load_pairs`: memory holds {offset, value} pairs
                    // (the title loads each dispatch's user data this way).
                    let (a, count) = (addr64(p(0), p(1)), p(2) as u64);
                    if count > 0 && !valid(a, 8 * count) {
                        return (n, "a register load's pairs".into());
                    }
                    for i in 0..count {
                        let (o, val) = unsafe { (rd32(a + 8 * i), rd32(a + 8 * i + 4)) };
                        match o & 0xFFFF {
                            0x20C => lo = val,
                            0x20D => hi = val,
                            _ => {}
                        }
                    }
                }
                op::DRAW_INDEX_2 | op::DRAW_INDEX_AUTO | op::DRAW_INDEX_OFFSET_2 | op::DRAW_INDIRECT | op::DRAW_INDEX_INDIRECT | op::INDIRECT_BUFFER => return (n, format!("opcode {opcode:#x}")),
                op::NOP if words >= 2 && p(0) == DS_MAGIC && !matches!(p(1), pop::PUSH_MARKER | pop::SET_MARKER | pop::POP_MARKER) => return (n, format!("private packet {:#x}", p(1))),
                _ => {}
            }
            cur = next;
        }
        (n, "the packet limit".into())
    }

    /// The scan of `scan_through` (`whole`: also through runs the parallel
    /// workers would record): the commands, why it ended, its nanoseconds.
    fn scan_commands(&self, f: &Fetched, from: u64, end: u64, whole: bool, want_gfx: bool) -> (Vec<AheadCmd>, usize, u64) {
        use super::ahead::{END_ARGS, END_BUFFER, END_HEADER, END_INVALID, END_MAX, END_PREDICATED};
        const WAIT_SCAN_MAX: usize = 1024;
        let t0 = Instant::now();
        // Runs this command processor would split among parallel workers.
        let par_min = if !whole && f.words.is_empty() && !super::capture::capturing() && (if self.compute { compute_workers() > 0 } else { super::par::workers() > 0 }) {
            if self.compute { (super::tune::par_min_compute() as usize).max(super::par::floor()) } else { super::par::graphics_min() }
        } else {
            usize::MAX
        };
        let mut why = END_BUFFER;
        let (mut sh, mut ctx) = std::mem::take(&mut *self.scan_regs.borrow_mut());
        sh.clear();
        sh.extend_from_slice(&self.regs.sh);
        ctx.clear();
        ctx.extend_from_slice(&self.regs.ctx);
        let mut draw_base = self.draw_indirect_base;
        // VGT_PRIMITIVE_TYPE, for the draws' baked-state registers (`want_gfx`), and the
        // context-state stack a graphics scan follows (`pop::CONTEXT_STATE`, as `walk_state`).
        let mut prim = self.reg(Space::Uc, 0x242);
        let mut stack: Vec<Vec<u32>> = if want_gfx { self.ctx_stack.clone() } else { Vec::new() };
        let mut cmds: Vec<AheadCmd> = Vec::new();
        // The synchronization-free run being scanned: its first command's
        // index and its commands.
        let (mut run_first, mut run_len) = (0usize, 0usize);
        // The program of the last small direct dispatches and how many in a
        // row: a run makes it batchable before the workers prepare it (as
        // execution's `cs_streak` would at the run's fourth dispatch; they
        // compiled it unbatched first, then it was made again batched: 11-25
        // first uses, 5.7-7.5 s in a first session's walk).
        let mut streak = (0u64, 0u32);
        let mut cur = from;
        while cur < end {
            if cmds.len() >= WAIT_SCAN_MAX {
                why = END_MAX;
                break;
            }
            if !valid(cur, 4) {
                why = END_INVALID;
                break;
            }
            let h = f.rd(cur);
            match h >> 30 {
                3 => {}
                2 => {
                    cur += 4;
                    continue;
                }
                _ => {
                    why = END_HEADER;
                    break;
                }
            }
            let count = (h >> 16) & 0x3FFF;
            if count == 0x3FFF {
                cur += 4;
                continue;
            }
            // Predicated packets may be skipped at execution.
            if h & 1 != 0 {
                why = END_PREDICATED;
                break;
            }
            let opcode = (h >> 8) & 0xFF;
            let n = count as u64 + 1;
            let next = cur + 4 * (n + 1);
            if next > end || !valid(cur, 4 * (n + 1)) {
                why = END_BUFFER;
                break;
            }
            let at = cur;
            let p = |i: u64| f.rd(at + 4 * (1 + i));
            let set = |v: &mut Vec<u32>, first: u32| {
                for i in 1..n {
                    if let Some(r) = v.get_mut((first + (i - 1) as u32) as usize) {
                        *r = p(i);
                    }
                }
            };
            let mut command: Option<[Option<super::shaders::StageRegs>; 2]> = None;
            let mut gfx: Option<super::exec::GfxRegs> = None;
            // (An indexed indirect draw drawn with its start instance as Vulkan's first instance,
            // as `cpu_args_base_ok` decides: its vertex stage's slot, `want_gfx`.)
            let mut base_inst: Option<u16> = None;
            let mut wave64 = true;
            let mut sync = false;
            match opcode {
                op::NOP => {
                    // Private packets (flips, branches, draws, state) end
                    // the scan; debug markers do not. A graphics scan
                    // (`want_gfx`) follows the context-state stack and passes
                    // the display-buffer wait and workload notes, as
                    // `walk_state` does (its scans ended at them after 9-33
                    // commands with 3-13 KiB of their buffer left, 2026-11-19).
                    if n >= 2 && p(0) == DS_MAGIC && !matches!(p(1), pop::PUSH_MARKER | pop::SET_MARKER | pop::POP_MARKER) {
                        let passed = want_gfx
                            && match p(1) {
                                pop::WAIT_SAFE_FOR_RENDERING | pop::WORKLOADS_ACTIVE | pop::WORKLOAD_COMPLETE | pop::LOD_STATS => true,
                                pop::CONTEXT_STATE if n >= 3 => {
                                    let clear = |c: &mut Vec<u32>| {
                                        c.fill(0);
                                        for &(i, val) in context_defaults() {
                                            if let Some(r) = c.get_mut(i as usize) {
                                                *r = val;
                                            }
                                        }
                                    };
                                    match p(2) {
                                        0 => {
                                            clear(&mut ctx);
                                            true
                                        }
                                        1 => {
                                            stack.push(ctx.clone());
                                            true
                                        }
                                        2 => match stack.pop() {
                                            Some(c) => {
                                                ctx = c;
                                                true
                                            }
                                            None => false,
                                        },
                                        3 => {
                                            stack.push(ctx.clone());
                                            clear(&mut ctx);
                                            true
                                        }
                                        _ => false,
                                    }
                                }
                                _ => false,
                            };
                        if !passed {
                            why = 0x100 + (p(1) as usize).min(0x2F);
                            break;
                        }
                    }
                }
                // GPU-side synchronization, CP writes and label waits:
                // passed (they end synchronization-free runs).
                // (Those a run holds, `run_pass`, `run_multi`, leave it open.)
                op::RELEASE_MEM | op::WAIT_REG_MEM | op::WAIT_REG_MEM64 if self.run_pass_here(opcode, &p).is_some() => {}
                op::DRAW_INDEX_INDIRECT_MULTI | op::DRAW_INDIRECT_MULTI if self.run_multi(opcode) => {}
                op::EVENT_WRITE | op::ACQUIRE_MEM | op::RELEASE_MEM | op::WRITE_DATA | op::WAIT_REG_MEM | op::WAIT_REG_MEM64 => sync = true,
                op::SET_UCONFIG_REG | op::SET_UCONFIG_REG_INDEX => {
                    // (VGT_PRIMITIVE_TYPE: the draws' baked state, `want_gfx`.)
                    let first = p(0) & 0xFFFF;
                    if n >= 2 && (first..first + (n as u32 - 1)).contains(&0x242) {
                        prim = p(1 + (0x242 - first) as u64);
                    }
                }
                op::LOAD_UCONFIG_REG | op::INDEX_TYPE | op::INDEX_BASE | op::INDEX_BUFFER_SIZE | op::NUM_INSTANCES | op::CONTEXT_CONTROL => {}
                op::SET_CONTEXT_REG | op::SET_CONTEXT_REG_INDEX => set(&mut ctx, p(0) & 0xFFFF),
                op::SET_SH_REG | op::SET_SH_REG_INDEX => set(&mut sh, p(0) & 0xFFFF),
                op::LOAD_CONTEXT_REG_INDEX | op::LOAD_SH_REG_INDEX => {
                    // As `load_pairs`: memory holds {offset, value} pairs.
                    let (a, count) = (addr64(p(0), p(1)), p(2) as u64);
                    if count > 0 && !valid(a, 8 * count) {
                        why = END_ARGS;
                        break;
                    }
                    let v = if opcode == op::LOAD_CONTEXT_REG_INDEX { &mut ctx } else { &mut sh };
                    for i in 0..count {
                        let (o, val) = unsafe { (rd32(a + 8 * i), rd32(a + 8 * i + 4)) };
                        if let Some(r) = v.get_mut((o & 0xFFFF) as usize) {
                            *r = val;
                        }
                    }
                }
                op::SET_BASE => {
                    if p(0) != 2 {
                        draw_base = addr64(p(1), p(2));
                    }
                }
                op::DRAW_INDEX_2 | op::DRAW_INDEX_AUTO | op::DRAW_INDEX_OFFSET_2 | op::DRAW_INDIRECT | op::DRAW_INDEX_INDIRECT => {
                    if matches!(opcode, op::DRAW_INDIRECT | op::DRAW_INDEX_INDIRECT) {
                        // As `packet` does: the base vertex and start instance
                        // go to the user SGPRs the packet names.
                        let a = draw_base + p(0) as u64;
                        let (bytes, k) = if opcode == op::DRAW_INDEX_INDIRECT { (20, 3) } else { (16, 2) };
                        if !valid(a, bytes) {
                            why = END_ARGS;
                            break;
                        }
                        let (bv, si) = unsafe { (rd32(a + 4 * k), rd32(a + 4 * (k + 1))) };
                        let (vtx_loc, inst_loc) = (p(1) & 0xFFFF, p(1) >> 16);
                        let ngg = super::shaders::ngg_addr_of(&ScanRegs { sh: &sh, ctx: &ctx });
                        let inst_loc = if inst_loc != 0 { inst_loc } else { crate::hle::libs::agc::instance_offset_slot(ngg).map_or(0, |s| 0x8C + s) };
                        if want_gfx && opcode == op::DRAW_INDEX_INDIRECT && vtx_loc == 0 && (0x8C..0x8C + 32).contains(&inst_loc) && super::tune::cpu_args_base() && super::vk::try_ctx().is_some_and(|c| c.draw_parameters) && !super::shaders::cpu_args_program(ngg) {
                            base_inst = Some((inst_loc - 0x8C) as u16);
                        }
                        if vtx_loc != 0 {
                            if let Some(r) = sh.get_mut(vtx_loc as usize) {
                                *r = bv;
                            }
                        }
                        if inst_loc != 0 && base_inst.is_none() {
                            if let Some(r) = sh.get_mut(inst_loc as usize) {
                                *r = si;
                            }
                        }
                    }
                    let view = ScanRegs { sh: &sh, ctx: &ctx };
                    let ps = super::shaders::ps_addr_of(&view);
                    command = Some([
                        Some(super::shaders::StageRegs::capture(super::shaders::StageKind::Vertex, &view)),
                        (ps != 0).then(|| super::shaders::StageRegs::capture(super::shaders::StageKind::Pixel, &view)),
                    ]);
                    if want_gfx {
                        gfx = Some(super::exec::GfxRegs::capture(&|a| ctx.get(((a - 0x28000) / 4) as usize).copied().unwrap_or(0), prim).with_base_inst(base_inst));
                    }
                }
                op::DISPATCH_DIRECT | op::DISPATCH_INDIRECT => {
                    let initiator = if opcode == op::DISPATCH_DIRECT { p(3) } else if n >= 3 { p(2) } else { p(1) };
                    wave64 = initiator & (1 << 15) == 0;
                    let groups = if opcode == op::DISPATCH_DIRECT { p(0) as u64 * p(1) as u64 * p(2) as u64 } else { 0 };
                    if super::tune::batch_cs() && groups > 0 && groups <= super::exec::BATCH_SMALL {
                        let a = ((sh.get(0x20C).copied().unwrap_or(0) as u64) | ((sh.get(0x20D).copied().unwrap_or(0) as u64 & 0xFF) << 32)) << 8;
                        streak = if streak.0 == a { (a, streak.1 + 1) } else { (a, 1) };
                        if streak.1 == super::exec::BATCH_STREAK && !super::shaders::batch_marked(a) {
                            if super::shaders::mark_batchable(a) {
                                let view = ScanRegs { sh: &sh, ctx: &ctx };
                                super::compile::submit_batched(super::shaders::StageRegs::capture(super::shaders::StageKind::Compute, &view), wave64);
                            }
                            super::exec::BATCHABLE_AHEAD.fetch_add(1, Ordering::Relaxed);
                        }
                    }
                    let view = ScanRegs { sh: &sh, ctx: &ctx };
                    command = Some([Some(super::shaders::StageRegs::capture(super::shaders::StageKind::Compute, &view)), None]);
                }
                _ => {
                    why = opcode as usize;
                    break;
                }
            }
            if sync {
                if run_len >= par_min {
                    break;
                }
                run_len = 0;
            }
            if let Some(stages) = command {
                if run_len == 0 {
                    run_first = cmds.len();
                }
                run_len += 1;
                let stages = stages.map(|r| r.map(|r| super::ahead::Slot::new(r, wave64)));
                cmds.push(AheadCmd { at, stages, gfx });
            }
            cur = next;
        }
        // A run the workers will record: its commands are theirs.
        if run_len >= par_min {
            cmds.truncate(run_first);
        }
        *self.scan_regs.borrow_mut() = (sh, ctx);
        (cmds, why, t0.elapsed().as_nanos() as u64)
    }

    /// A command buffer of `size` dwords at `addr` this command processor
    /// runs (`cb_census_on`).
    fn cb_note(&self, addr: u64, size: u32) {
        use std::hash::Hasher;
        if size == 0 || !valid(addr, 4 * size as u64) {
            return;
        }
        let words = unsafe { std::slice::from_raw_parts(addr as *const u32, size as usize) };
        let mut h = super::fx::FxHasher::default();
        hash_words(&mut h, words);
        let contents = h.finish();
        hash_words(&mut h, &self.regs.sh);
        hash_words(&mut h, &self.regs.ctx);
        hash_words(&mut h, &self.regs.uc);
        let with_regs = h.finish();
        h.write_u64(addr);
        let at = h.finish();
        let flip = super::FRAMES.load(Ordering::Relaxed);
        let (c, r, a) = CB_SEEN.with(|m| {
            let mut m = m.borrow_mut();
            if m.len() > 1 << 20 {
                m.clear();
            }
            let mut recent = |k: u64| m.insert(k, flip).is_some_and(|f| f < flip && flip - f <= CB_FLIPS);
            (recent(contents ^ 1), recent(with_regs ^ 2), recent(at ^ 3))
        });
        let s = &CB_CENSUS[self.compute as usize];
        let n = size as u64;
        s[0].fetch_add(1, Ordering::Relaxed);
        s[1].fetch_add(n, Ordering::Relaxed);
        for (k, hit) in [(2, c), (4, r), (6, a)] {
            if hit {
                s[k].fetch_add(1, Ordering::Relaxed);
                s[k + 1].fetch_add(n, Ordering::Relaxed);
            }
        }
    }

    /// Completes recorded GPU work (at synchronization points: the CP waits
    /// for shader work only where the packet stream asks it to).
    fn sync(&mut self) {
        self.fence_pending = false;
        if let Some(e) = self.exec.as_mut() {
            if e.busy() {
                let k = (self.cur_opcode as usize).min(255);
                SYNC_BY_OPCODE[k].fetch_add(1, Ordering::Relaxed);
            }
            e.flush();
        }
    }

    /// Runs a CP memory effect now, or after this queue's outstanding GPU
    /// work and earlier deferred effects. `end_of_pipe` effects (RELEASE_MEM)
    /// always follow the GPU work; other CP writes follow it only after a
    /// partial flush or acquire (otherwise the CP does not wait for shaders),
    /// and later GPU work then waits for them (it may read what they write).
    /// Returns the batch sequence a deferred effect runs after.
    /// A memory write that would be deferred behind earlier GPU work, done by
    /// the GPU in order with it (`Exec::gpu_cp_write`); false: take the CPU
    /// path (switch off, nothing outstanding, or earlier CPU-side effects
    /// still pending, which it must not overtake).
    fn gpu_cp_write(&mut self, a: u64, data: &[u32], label: Option<(u64, u8)>, end_of_pipe: bool) -> bool {
        // Writes recorded into held batches follow the work before them:
        // always recorded into its batch.
        let held = self.held.is_some();
        if !held {
            super::replay::note_cp_write(a, 4 * data.len() as u64);
        }
        let needs_prior = end_of_pipe || self.fence_pending;
        let Some(e) = self.exec.as_mut() else { return false };
        // Nothing outstanding to follow: the CPU performs the write now.
        let refused = if !held && !(e.busy() && (needs_prior || e.deferred_pending())) {
            Some(0)
        } else if !e.gpu_cp_write(a, data, label, end_of_pipe) {
            Some(2)
        } else {
            None
        };
        if let Some(k) = refused {
            if !held {
                super::exec::GPU_CP_REFUSED[k].fetch_add(1, Ordering::Relaxed);
            }
            return false;
        }
        if !end_of_pipe {
            super::exec::CPW_KINDS[self.compute as usize][(held as usize) << 1 | self.fence_pending as usize].fetch_add(1, Ordering::Relaxed);
            if self.fence_pending {
                super::exec::CPW_FENCES[self.compute as usize][self.fence_kind as usize & 127].fetch_add(1, Ordering::Relaxed);
            }
        }
        self.fence_pending = false;
        if held {
            return true;
        }
        // Diagnostics (no latching: the value is not in memory yet).
        let mut g = LAST_WRITE.lock();
        let m = g.get_or_insert_with(Default::default);
        if m.len() > 1 << 16 {
            m.clear();
        }
        m.insert(a & !7, (self.name.clone(), label.map(|l| l.0).unwrap_or(data[0] as u64), self.cur_packet, a));
        true
    }

    fn effect(&mut self, end_of_pipe: bool, write: Option<(u64, u64)>, f: Box<dyn FnOnce() + Send>) -> Option<u64> {
        if self.held.is_some() {
            // Held batches: nothing they do may happen before the work they
            // follow (CPU-side effects are not held).
            drop(f);
            self.held_stop("CP effect", self.cur_packet);
            return None;
        }
        if let Some((a, n)) = write {
            super::replay::note_cp_write(a, n);
            super::forward::written(a, n);
            if super::smalltrack::on() {
                super::smalltrack::cp_write(a, n, false);
            }
        }
        // Mirror mode: the pages it changes are copied into the mirror at
        // the next placement over them (`resident::cp_wrote`).
        let f = match write {
            Some((a, n)) if super::resident::mirror_mode() => Box::new(move || {
                f();
                super::resident::cp_wrote(a, n);
            }) as Box<dyn FnOnce() + Send>,
            // Front end 2: the mirror no longer holds what it changed.
            Some((a, n)) if super::fe2::on() => Box::new(move || {
                f();
                super::fe2::invalidate(a, n);
            }) as Box<dyn FnOnce() + Send>,
            _ => f,
        };
        let needs_prior = end_of_pipe || self.fence_pending;
        match self.exec.as_mut() {
            Some(e) if e.busy() && (needs_prior || e.deferred_pending()) => {
                let k = (self.cur_opcode as usize).min(255);
                let why = if end_of_pipe { 0 } else if self.fence_pending { 1 } else { 2 };
                DEFERRED_BY_OPCODE[k][why].fetch_add(1, Ordering::Relaxed);
                self.fence_pending = false;
                let f = match write {
                    Some((a, n)) if super::tune::race_checks() => {
                        let (q, at, op) = (self.name.clone(), self.cur_packet, self.cur_opcode);
                        Box::new(move || {
                            f();
                            cpw_note(&q, a, n, at, op);
                        }) as Box<dyn FnOnce() + Send>
                    }
                    _ => f,
                };
                e.after_gpu(f, write, !end_of_pipe)
            }
            _ => {
                if needs_prior {
                    self.fence_pending = false;
                }
                f();
                if let Some((a, n)) = write {
                    cpw_note(&self.name, a, n, self.cur_packet, self.cur_opcode);
                }
                None
            }
        }
    }

    /// Makes memory the CP reads now current: completes this queue's GPU
    /// work (and deferred CP writes) that writes into `reads`. Indirect
    /// arguments are typically written by a culling dispatch shortly before;
    /// the draws after the first then find nothing pending.
    fn sync_reads(&mut self, reads: &[(u64, u64)]) {
        // Past a label wait, a read that dropped a stretch on this label
        // before: the stretch stops before it (the packet runs again once
        // the label passed), keeping what it recorded.
        if let Some(w) = self.past_wait.as_deref() {
            if past_conflicts_hit(w.a, reads) {
                let at = self.cur_packet;
                self.held_stop("a read its label's writer wrote before", at);
                return;
            }
        }
        if let (Some(e), true) = (self.exec.as_mut(), self.held.is_some()) {
            e.note_held_reads(reads, 4);
        }
        if let Some(e) = self.exec.as_mut() {
            if e.flush_if_writes(reads) {
                let k = (self.cur_opcode as usize).min(255);
                SYNC_BY_OPCODE[k].fetch_add(1, Ordering::Relaxed);
            }
        }
    }

    /// Completes all GPU work recorded so far and its deferred effects.
    pub fn finish(&mut self) {
        self.sync();
    }

    /// Hands recorded GPU work to the device without waiting.
    fn submit_gpu(&mut self, why: &'static str) {
        if let Some(e) = self.exec.as_mut() {
            e.submit_for(why);
        }
    }

    /// Executes a dispatch (when GPU execution is enabled).
    /// An indirect dispatch with the GPU reading its arguments (see
    /// `Exec::dispatch_gpu_indirect`); false: not taken.
    fn dispatch_gpu_indirect(&mut self, args: u64, initiator: u32) -> bool {
        if !super::exec::enabled() || self.exec_failed {
            return false;
        }
        let Some(mut e) = self.exec.take() else { return false };
        let taken = e.dispatch_gpu_indirect(self, args, initiator);
        self.exec = Some(e);
        taken
    }

    fn dispatch(&mut self, groups: [u32; 3], initiator: u32) {
        if super::frameprobe::on() && self.name == "gpu-gfx" {
            super::frameprobe::command();
        }
        self.note_run_epoch();
        if !super::exec::enabled() || self.exec_failed {
            self.note("dispatches are decoded but not executed (set DS_GPU=1 to run them)");
            return;
        }
        if self.exec.is_none() {
            match super::exec::Exec::new(&self.name) {
                Ok(e) => self.exec = Some(Box::new(e)),
                Err(e) => {
                    crate::error!("gpu[{}]: no executor: {e}", self.name);
                    self.exec_failed = true;
                    return;
                }
            }
        }
        let mut e = self.exec.take().unwrap();
        super::xwait::note_command(&self.name);
        e.dispatch(self, groups, initiator);
        self.exec = Some(e);
        self.split_if_due();
    }

    /// Resets the context registers to their clear-state values: zero, except
    /// the registers with a non-zero Agc register default (assumed to be the
    /// hardware clear state; e.g. CB_COLOR_CONTROL = 0x00CC0010, scissor
    /// bottom-right corners 0x40004000: the title leaves these unprogrammed
    /// for ordinary draws).
    fn clear_context(&mut self) {
        self.regs.ctx.fill(0);
        self.ctx_dirty = true;
        for &(i, val) in context_defaults() {
            self.set_reg(Space::Ctx, i, val);
        }
    }

    /// The whole state now, as `walk_state` follows it (`gpu::frameprobe`).
    pub fn walk_start(&self) -> WalkState {
        WalkState { st: self.chunk_state(), stack: self.ctx_stack.clone(), unknown: Vec::new(), labels: Vec::new(), flip: None }
    }

    /// Applies the state effects of the packets in [addr, addr + 4 size) to
    /// `w` as `run` would, without executing anything: Ok(true) at a FLIP
    /// (the walk stops before it), Ok(false) at the buffer's end, Err at a
    /// packet whose effect cannot be known before it executes (COND_EXEC
    /// reads memory GPU work may write, an indirect buffer, a bad packet).
    /// The user SGPRs indirect draws set from their arguments (memory GPU
    /// work writes) are listed in `unknown`.
    pub fn walk_state(w: &mut WalkState, addr: u64, size: u32) -> Result<bool, &'static str> {
        let f = Fetched::new(addr, size);
        let end = addr + 4 * size as u64;
        let mut cur = addr;
        while cur < end {
            if !valid(cur, 4) {
                return Err("outside guest memory");
            }
            let h = f.rd(cur);
            match h >> 30 {
                3 => {}
                2 => {
                    cur += 4;
                    continue;
                }
                0 => {
                    cur += 4 * (((h >> 16) & 0x3FFF) as u64 + 2);
                    continue;
                }
                _ => return Err("bad packet header"),
            }
            let count = (h >> 16) & 0x3FFF;
            let opcode = (h >> 8) & 0xFF;
            if count == 0x3FFF {
                cur += 4;
                continue;
            }
            let n = count as u64 + 1;
            if !valid(cur, 4 * (n + 1)) {
                return Err("packet outside guest memory");
            }
            let at = cur;
            let p = |i: u64| f.rd(at + 4 * (1 + i));
            match opcode {
                op::NOP if n >= 2 && p(0) == DS_MAGIC => match p(1) {
                    pop::FLIP => {
                        w.flip = Some((at, 4 * (n + 1)));
                        return Ok(true);
                    }
                    pop::CONTEXT_STATE if n >= 3 => match p(2) {
                        0 => walk_clear(&mut w.st),
                        1 => w.stack.push(w.st.regs.ctx.clone()),
                        2 => match w.stack.pop() {
                            Some(c) => w.st.regs.ctx = c,
                            None => return Err("context state pop without push"),
                        },
                        3 => {
                            w.stack.push(w.st.regs.ctx.clone());
                            walk_clear(&mut w.st);
                        }
                        _ => return Err("unknown context state operation"),
                    },
                    _ => {}
                },
                op::CLEAR_STATE => walk_clear(&mut w.st),
                op::WRITE_DATA => {
                    let c = p(0);
                    let (dst, no_incr, count) = ((c >> 8) & 0xF, (c >> 16) & 1 != 0, n.saturating_sub(3));
                    let a = addr64(p(1), p(2));
                    match dst {
                        0 => return Err("WRITE_DATA to registers"),
                        1 | 2 | 5 if count > 0 => match (count, no_incr) {
                            (1, _) => w.labels.push((a, p(3) as u64, 4)),
                            (2, false) => w.labels.push((a, p(3) as u64 | (p(4) as u64) << 32, 8)),
                            (_, true) => w.labels.push((a, p(2 + count) as u64, 4)),
                            _ => w.labels.extend((0..count).map(|i| (a + 4 * i, p(3 + i) as u64, 4u8))),
                        },
                        _ => {}
                    }
                }
                op::RELEASE_MEM => {
                    let a = addr64(p(2) & !3, p(3));
                    match (p(1) >> 29) & 7 {
                        1 => w.labels.push((a, p(4) as u64, 4)),
                        2 => w.labels.push((a, addr64(p(4), p(5)), 8)),
                        3 | 4 => w.labels.push((a, 0, 0)),
                        _ => {}
                    }
                }
                op::COPY_DATA if (p(0) >> 8) & 0xF == 0 => return Err("COPY_DATA to a register"),
                op::COND_EXEC => return Err("COND_EXEC"),
                op::INDIRECT_BUFFER => return Err("indirect buffer"),
                op::DRAW_INDIRECT | op::DRAW_INDEX_INDIRECT => {
                    let (vtx_loc, inst_loc) = (p(1) & 0xFFFF, p(1) >> 16);
                    let inst_loc = if inst_loc != 0 {
                        inst_loc
                    } else {
                        let r = ScanRegs { sh: &w.st.regs.sh, ctx: &w.st.regs.ctx };
                        crate::hle::libs::agc::instance_offset_slot(super::shaders::ngg_addr_of(&r)).map_or(0, |s| 0x8C + s)
                    };
                    for r in [vtx_loc, inst_loc] {
                        if r != 0 && !w.unknown.contains(&r) {
                            w.unknown.push(r);
                        }
                    }
                    Self::apply_state(&mut w.st, opcode, n, &p);
                }
                _ => Self::apply_state(&mut w.st, opcode, n, &p),
            }
            cur = at + 4 * (n + 1);
        }
        Ok(false)
    }

    /// Diagnostics (`tune split_every`): ends the graphics queue's batch
    /// after every N commands recorded serially.
    fn split_if_due(&mut self) {
        let n = super::tune::split_every();
        if n == 0 || self.compute || self.par.is_some() {
            return;
        }
        if let Some(e) = self.exec.as_mut() {
            if e.serial_batch_len() as u64 >= n {
                SPLITS.fetch_add(1, Ordering::Relaxed);
                e.submit_for("split (diagnostics)");
            }
        }
    }

    /// Executes a draw (when GPU execution is enabled).
    /// A command of the serial run being watched (`RUN_EPOCHS`).
    fn note_run_epoch(&mut self) {
        if let Some((end, ep, t)) = self.run_watch {
            if self.cur_packet < end {
                RUN_EPOCHS[1].fetch_add(1, Ordering::Relaxed);
                if super::precache::epoch() as u32 == ep {
                    RUN_EPOCHS[2].fetch_add(1, Ordering::Relaxed);
                }
                RUN_EPOCHS[3].fetch_add(t.elapsed().as_nanos() as u64, Ordering::Relaxed);
            } else {
                self.run_watch = None;
            }
        }
    }

    fn draw(&mut self, args: super::exec::DrawArgs) {
        if super::frameprobe::on() && self.name == "gpu-gfx" {
            super::frameprobe::command();
        }
        let laps = (!self.compute && super::tune::cp_classes()).then(|| {
            IDRAW_GROUP.with(|g| g.set(if !self.main_gfx { 2 } else if self.past_wait.is_some() { 1 } else { 0 }));
            Instant::now()
        });
        let _every = EveryLap(laps);
        self.note_run_epoch();
        if !super::exec::enabled() || self.exec_failed {
            self.note("draws are decoded but not rendered (set DS_GPU=1 to render them)");
            return;
        }
        if self.exec.is_none() {
            match super::exec::Exec::new(&self.name) {
                Ok(e) => self.exec = Some(Box::new(e)),
                Err(e) => {
                    crate::error!("gpu[{}]: no executor: {e}", self.name);
                    self.exec_failed = true;
                    return;
                }
            }
        }
        // Measurement: draws with the last draw's context registers (and
        // programs and primitive type), as a draw that took its targets,
        // dynamic state and pipeline from the last one could.
        let progs = (super::shaders::ngg_addr(self), super::shaders::ps_addr(self), self.reg(Space::Uc, 0x242));
        DRAW_STATE[0].fetch_add(1, Ordering::Relaxed);
        // (The context registers and the primitive type alone: `Exec::draw_same_ctx`.)
        let same_ctx = !self.ctx_dirty && progs.2 == self.last_draw_progs.2;
        if self.ctx_dirty || std::mem::take(&mut self.ps_dirty) || self.ps_epoch == 0 {
            self.ps_epoch = PS_EPOCH.fetch_add(1, Ordering::Relaxed) + 1;
        }
        let same = same_ctx && progs == self.last_draw_progs;
        if !self.ctx_dirty {
            DRAW_STATE[1].fetch_add(1, Ordering::Relaxed);
            if same {
                DRAW_STATE[2].fetch_add(1, Ordering::Relaxed);
            }
        }
        // Runs of such draws (batching candidates), by length.
        if same {
            self.same_run += 1;
        } else {
            note_draw_run(self.same_run);
            self.same_run = 1;
        }
        self.ctx_dirty = false;
        self.last_draw_progs = progs;
        super::xwait::note_command(&self.name);
        let mut e = self.exec.take().unwrap();
        e.draw_same = same;
        e.draw_same_ctx = same_ctx;
        e.ps_epoch = self.ps_epoch;
        self.last_draw = e.draw(self, args);
        self.exec = Some(e);
        self.split_if_due();
    }

    /// Whether a packet leaves the draw state as the last draw had it (a
    /// repeat of that draw stays possible, `tune draw_repeat`): indexed
    /// indirect draws, plain NOPs, and register writes and loads and index
    /// or base settings that only write the values already there.
    fn keeps_repeat(&self, opcode: u32, n: u64, p: &dyn Fn(u64) -> u32) -> bool {
        let same = |space: Space| {
            let first = p(0) & 0xFFFF;
            (1..n).all(|i| self.reg(space, first + (i - 1) as u32) == p(i))
        };
        // {offset, value} pairs in memory, as `load_pairs` reads them.
        let same_pairs = |space: Space| {
            let (a, k) = (addr64(p(0), p(1)), p(2) as u64);
            k <= 4096 && valid(a, 8 * k.max(1)) && (0..k).all(|i| unsafe { self.reg(space, rd32(a + 8 * i) & 0xFFFF) == rd32(a + 8 * i + 4) })
        };
        match opcode {
            op::DRAW_INDEX_INDIRECT => true,
            op::NOP => !(n >= 2 && p(0) == DS_MAGIC),
            op::SET_CONTEXT_REG => same(Space::Ctx),
            op::SET_SH_REG => same(Space::Sh),
            op::SET_UCONFIG_REG => same(Space::Uc),
            op::LOAD_CONTEXT_REG_INDEX => same_pairs(Space::Ctx),
            op::LOAD_SH_REG_INDEX => same_pairs(Space::Sh),
            op::LOAD_UCONFIG_REG => same_pairs(Space::Uc),
            op::INDEX_BASE => self.index_base == addr64(p(0), p(1)),
            op::INDEX_BUFFER_SIZE => self.index_count == p(0),
            op::NUM_INSTANCES => self.num_instances == p(0).max(1),
            op::SET_BASE => (if p(0) == 2 { self.dispatch_indirect_base } else { self.draw_indirect_base }) == addr64(p(1), p(2)),
            _ => false,
        }
    }

    /// Whether an indexed indirect draw whose arguments the CPU reads hands
    /// the start instance over as Vulkan's first instance (`tune
    /// cpu_args_base`): no base-vertex word, a start-instance word among the
    /// vertex stage's user data, draw parameters, a vertex program not known
    /// to need the word on the CPU; not while a frame is captured.
    fn cpu_args_base_ok(&self, vtx_loc: u32, inst_loc: u32) -> bool {
        const GS_USER_DATA: u32 = 0x8C;
        super::tune::cpu_args_base()
            && vtx_loc == 0
            && (GS_USER_DATA..GS_USER_DATA + 32).contains(&inst_loc)
            && !self.exec_failed
            && !super::capture::capturing()
            && super::exec::enabled()
            && super::vk::try_ctx().is_some_and(|c| c.draw_parameters)
            && !super::shaders::cpu_args_program(super::shaders::ngg_addr(self))
    }

    /// An indexed indirect draw whose arguments the CPU read (`w`): the
    /// start instance as Vulkan's first instance, the vertex stage taking
    /// its user-data word from gl_BaseInstance (`tune cpu_args_base`); a
    /// repeat of the draw before when it can be (`tune draw_repeat`), else
    /// recorded in full; a vertex stage that needs the word on the CPU gets
    /// it in its register as before. The outcome.
    fn draw_cpu_args_base(&mut self, w: [u32; 5], sz: u32, inst_loc: u32) -> String {
        const GS_USER_DATA: u32 = 0x8C;
        let index = (self.index_base + w[2] as u64 * sz as u64, sz);
        let laps = (!self.compute && super::tune::cp_classes()).then(Instant::now);
        if std::mem::take(&mut self.repeat_ok) && super::tune::draw_repeat() {
            // (The pass census: a repeat's time by its pass, `exec::pass_note` kind 3.)
            let census = super::exec::prog_census().then(|| (super::exec::pass_key(self), Instant::now()));
            if let Some(e) = self.exec.as_mut() {
                if e.draw_repeat_cpu(index, w[0], w[1], w[3] as i32, w[4]) {
                    if let Some(((depth, key), t)) = census {
                        super::exec::pass_note(e.prog_class(), depth, key, 3, t.elapsed().as_nanos() as u64);
                    }
                    // (Stage preparations made ahead for it are not used.)
                    let _ = self.take_ahead();
                    self.repeat_ok = true;
                    self.split_if_due();
                    idraw_part(2, laps);
                    return "drawn (repeating the draw before)".into();
                }
            }
        }
        let _full = FullLap(laps);
        self.draw(super::exec::DrawArgs { count: w[0], instances: w[1], index: Some(index), base_vertex: Some(w[3] as i32), first_instance: w[4], base_inst: Some((inst_loc - GS_USER_DATA) as u16) });
        if self.last_draw == super::exec::NEEDS_CPU_ARGS {
            // The start instance in its register, as before.
            self.set_reg(Space::Sh, inst_loc, w[4]);
            self.draw(super::exec::DrawArgs { count: w[0], instances: w[1], index: Some(index), base_vertex: Some(w[3] as i32), first_instance: 0, base_inst: None });
            return std::mem::take(&mut self.last_draw).into_owned();
        }
        self.repeat_ok = super::tune::draw_repeat() && self.exec.as_ref().is_some_and(|e| e.repeat_ready());
        std::mem::take(&mut self.last_draw).into_owned()
    }

    /// Whether this (main) command processor is recording its own chunk of a
    /// parallel run (profile: draw kinds).
    pub(super) fn in_parallel_run(&self) -> bool {
        self.par.is_some()
    }

    /// The graphics CP's part of its stream for `tune ballast_at` bits 5-8
    /// (`exec::ballast_cp`): 5 past a label wait, 6 past a parallel run, 7
    /// its own chunk of a run, 8 the rest (serial); other executors none.
    pub(super) fn ballast_part(&self) -> u8 {
        if !self.main_gfx {
            0
        } else if self.past_wait.is_some() {
            5
        } else if self.past.is_some() {
            6
        } else if self.par.is_some() {
            7
        } else {
            8
        }
    }

    /// Whether this is the graphics queue's main command processor (not a
    /// parallel worker or a compute queue's).
    pub(super) fn is_main_gfx(&self) -> bool {
        self.main_gfx
    }

    /// The packet being executed (its address).
    pub(super) fn packet_addr(&self) -> u64 {
        self.cur_packet
    }

    /// START_INST_LOC for an indirect draw packet that names none (our Agc
    /// builders): the SH register of the bound primitive shader's
    /// instance-offset user SGPR (its shader object declares it, see
    /// `agc::instance_offset_slot`), else 0 (none: the start instance then
    /// stays part of the instance id, as in Vulkan).
    fn instance_offset_loc(&self) -> u32 {
        const GS_USER_DATA: u32 = 0x8C;
        crate::hle::libs::agc::instance_offset_slot(super::shaders::ngg_addr(self)).map_or(0, |s| GS_USER_DATA + s)
    }

    /// Whether an indexed indirect draw's arguments (`first` index, `count`
    /// indices) run past the bound index buffer (INDEX_BUFFER_SIZE, in
    /// indices; 0: not set, not checked). No draw the title makes does: past
    /// the buffer the hardware fetches index 0 (degenerate triangles at
    /// best).
    fn index_args_past(&self, w: &[u32; 5]) -> bool {
        self.index_count != 0 && w[2] as u64 + w[0] as u64 > self.index_count as u64
    }

    /// An indexed indirect draw's record at `a` as read (`w`), checked: such
    /// a record running past its index buffer (`index_args_past`) is words
    /// read before the title or the GPU wrote them (0.7-3.2 billion indices
    /// with float bit patterns for counts and base vertices, every time
    /// before a device loss of 2026-11-07 to 11-09: FINDINGS.md 2026-11-09;
    /// drawing them runs past the driver's timeout). It is read again for up
    /// to 2 ms (the title's CPU filling it meanwhile: drawn as it then
    /// reads), else the draw is dropped (None). Both counted, the first 16 of
    /// each logged.
    fn checked_index_args(&self, a: u64, w: [u32; 5]) -> Option<[u32; 5]> {
        if !self.index_args_past(&w) {
            return Some(w);
        }
        super::exec::watch_note(|| format!("{} (a guarded draw record read: {w:?})", self.name), a, 20);
        super::exec::watch_dump(&format!("the guarded record at {a:#x}"));
        let t0 = Instant::now();
        let mut now = w;
        while t0.elapsed() < std::time::Duration::from_millis(2) {
            std::hint::spin_loop();
            now = unsafe { [rd32(a), rd32(a + 4), rd32(a + 8), rd32(a + 12), rd32(a + 16)] };
            if !self.index_args_past(&now) {
                let n = ARGS_BEYOND[1].fetch_add(1, Ordering::Relaxed);
                if n < 16 {
                    crate::warn!(
                        "gpu[{}]: indexed indirect draw's arguments at {a:#x} (packet {:#x}) ran past its index buffer ({:#x}, {} indices) as {w:?} and read {now:?} after {} us: drawn so (the last wait passed: kind {}, function {}, on a GPU write {})",
                        self.name,
                        self.cur_packet,
                        self.index_base,
                        self.index_count,
                        t0.elapsed().as_micros(),
                        self.last_wait_kind as char,
                        self.last_wait_func,
                        self.last_wait_gpu
                    );
                }
                return Some(now);
            }
        }
        let n = ARGS_BEYOND[0].fetch_add(1, Ordering::Relaxed);
        if n < 16 {
            crate::warn!(
                "gpu[{}]: indexed indirect draw dropped at packet {:#x}: arguments at {a:#x} {w:?} (after 2 ms {now:?}) run past its index buffer ({:#x}, {} indices; the last wait passed: kind {}, function {}, on a GPU write {})",
                self.name,
                self.cur_packet,
                self.index_base,
                self.index_count,
                self.last_wait_kind as char,
                self.last_wait_func,
                self.last_wait_gpu
            );
        }
        None
    }

    /// DRAW_INDIRECT_MULTI / DRAW_INDEX_INDIRECT_MULTI: draws described by
    /// consecutive argument records. gfx9+ layout (Mesa radeonsi and the Agc
    /// builders emit it): data offset; BASE_VTX_LOC; START_INST_LOC;
    /// DRAW_INDEX_LOC [15:0] | bit 30 count from memory | bit 31 draw index
    /// enable; count; count address lo/hi; stride; initiator. Each draw gets
    /// its base vertex, start instance and draw index in the user SGPRs the
    /// packet names. Returns the outcome summary.
    fn draw_indirect_multi(&mut self, indexed: bool, p: &dyn Fn(u64) -> u32) -> String {
        let (vtx_loc, inst_loc, id_loc) = (p(1) & 0xFFFF, p(2) & 0xFFFF, p(3) & 0xFFFF);
        let inst_loc = if inst_loc != 0 { inst_loc } else { self.instance_offset_loc() };
        let id_enable = p(3) & (1 << 31) != 0;
        let mut count = p(4);
        let mut count_src = format!("max {count}");
        MDI_PACKETS.fetch_add(1, Ordering::Relaxed);
        if p(3) & (1 << 30) != 0 {
            MDI_COUNT_FROM_MEMORY.fetch_add(1, Ordering::Relaxed);
            // The count may have been written by earlier GPU work.
            let ca = addr64(p(5), p(6));
            self.sync_reads(&[(ca, 4)]);
            let mem = if valid(ca, 4) { Some(unsafe { rd32(ca) }) } else { None };
            count_src = format!("max {count}, memory {ca:#x} = {mem:?}");
            count = mem.map_or(0, |m| count.min(m));
        }
        let capturing = super::capture::capturing();
        let mut subs: Vec<String> = Vec::new();
        let rec = if indexed { 20 } else { 16 };
        let stride = if p(7) != 0 { p(7) as u64 } else { rec };
        {
            let first = self.draw_indirect_base + p(0) as u64;
            let ta = super::exec::draw::trace_args_addr();
            if ta != 0 && first & !0xFFF == ta & !0xFFF {
                let w: Vec<u32> = (0..(count.min(4) as u64 * stride / 4).max(5)).map(|k| first + 4 * k).filter(|&x| valid(x, 4)).map(|x| unsafe { rd32(x) }).collect();
                crate::info!("trace args {first:#x} [{}] for flip {}: multi ({count_src}, stride {stride}) {w:?} vs {:#x}", self.name, crate::hle::libs::videoout::QUEUED_FLIP.load(Ordering::Relaxed) + 1, super::shaders::ngg_addr(self));
            }
        }
        // So may the arguments.
        if count > 0 {
            let first = self.draw_indirect_base + p(0) as u64;
            self.sync_reads(&[(first, (count.min(1 << 16) as u64 - 1) * stride + rec)]);
        }
        let size = if indexed { self.index_size() } else { Some(0) };
        let Some(sz) = size else {
            self.note("draws with 8-bit indices are not rendered yet");
            return "not executed (8-bit indices)".into();
        };
        let (mut drawn, mut outcome) = (0u32, String::new());
        for i in 0..count.min(1 << 16) {
            let a = self.draw_indirect_base + p(0) as u64 + i as u64 * stride;
            if !valid(a, rec) {
                break;
            }
            let w = unsafe { [rd32(a), rd32(a + 4), rd32(a + 8), rd32(a + 12), if indexed { rd32(a + 16) } else { 0 }] };
            if capturing && subs.len() < 8 {
                subs.push(format!("{:?}", &w[..rec as usize / 4]));
            }
            let w = if indexed {
                match self.checked_index_args(a, w) {
                    Some(w) => w,
                    None => continue,
                }
            } else {
                w
            };
            let (args, bv, si) = if indexed {
                (super::exec::DrawArgs { count: w[0], instances: w[1], index: Some((self.index_base + w[2] as u64 * sz as u64, sz)), base_vertex: Some(w[3] as i32), first_instance: w[4], base_inst: None }, w[3], w[4])
            } else {
                (super::exec::DrawArgs { count: w[0], instances: w[1], index: None, base_vertex: Some(w[2] as i32), first_instance: w[3], base_inst: None }, w[2], w[3])
            };
            if vtx_loc != 0 {
                self.set_reg(Space::Sh, vtx_loc, bv);
            }
            let mut args = args;
            if inst_loc != 0 {
                self.set_reg(Space::Sh, inst_loc, si);
                args.first_instance = 0;
            }
            if id_enable && id_loc != 0 {
                self.set_reg(Space::Sh, id_loc, i);
            }
            if args.count == 0 || args.instances == 0 {
                continue;
            }
            self.draw(args);
            drawn += 1;
            MDI_DRAWS.fetch_add(1, Ordering::Relaxed);
            outcome = std::mem::take(&mut self.last_draw).into_owned();
        }
        let detail = if capturing { format!(" [count {count_src}, stride {stride}, args {}]", subs.join(" ")) } else { String::new() };
        format!("{drawn} of {count} draws ({}){detail}", if drawn > 0 { outcome } else { "none".into() })
    }

    /// Smallest and largest index of `count` indices from `start` of the
    /// bound index buffer, and the first ones (frame captures).
    fn index_range(&self, start: u32, count: u32) -> String {
        let Some(sz) = self.index_size() else { return String::new() };
        let a = self.index_base + start as u64 * sz as u64;
        let n = count.min(1 << 20) as u64;
        if n == 0 || !guest_readable(a, n * sz as u64) {
            return " indices unreadable".into();
        }
        let get = |i: u64| unsafe { if sz == 2 { (rd32((a + 2 * i) & !3) >> (((a + 2 * i) & 2) * 8)) & 0xFFFF } else { rd32(a + 4 * i) } };
        let (mut lo, mut hi) = (u32::MAX, 0);
        for i in 0..n {
            let v = get(i);
            lo = lo.min(v);
            hi = hi.max(v);
        }
        let head: Vec<String> = (0..n.min(6)).map(|i| get(i).to_string()).collect();
        format!(" indices {lo}..{hi} [{}]", head.join(" "))
    }

    /// Whether a packet with the predicate bit executes under the current
    /// SET_PREDICATION state, and what the predicate read (None: no
    /// predication). ZPASS / PRIMCOUNT: begin/end counter pairs (16 bytes per
    /// render backend, bit 63 valid); BOOL64 / BOOL32: a value in memory.
    fn pred_value(&self) -> Option<(bool, String)> {
        let pr = self.pred?;
        let (visible, what) = match pr.op {
            1 | 2 => {
                if !guest_readable(pr.addr, 16 * 16) {
                    return Some((true, format!("zpass {:#x} unreadable", pr.addr)));
                }
                let (mut sum, mut valid) = (0u64, 0);
                for rb in 0..16u64 {
                    let (b, e) = unsafe { (rd64(pr.addr + 16 * rb), rd64(pr.addr + 16 * rb + 8)) };
                    if b >> 63 != 0 && e >> 63 != 0 {
                        valid += 1;
                        sum += (e & !(1 << 63)).wrapping_sub(b & !(1 << 63));
                    }
                }
                (sum > 0, format!("zpass {sum} ({valid} valid pairs)"))
            }
            3 => {
                let v = if guest_readable(pr.addr, 8) { unsafe { rd64(pr.addr) } } else { 0 };
                (v != 0, format!("bool64 {v:#x}"))
            }
            4 => {
                let v = if guest_readable(pr.addr, 4) { unsafe { rd32(pr.addr) } } else { 0 };
                (v != 0, format!("bool32 {v:#x}"))
            }
            o => return Some((true, format!("op {o}"))),
        };
        Some((if pr.cond == 1 { visible } else { !visible }, what))
    }

    /// Index size in bytes (VGT_INDEX_TYPE), None for 8-bit indices.
    fn index_size(&self) -> Option<u32> {
        match self.reg(Space::Uc, crate::hle::libs::agc_cmd::REG_VGT_INDEX_TYPE) & 3 {
            0 => Some(2),
            1 => Some(4),
            _ => None,
        }
    }

    fn note(&mut self, what: &'static str) {
        if self.notes_seen.insert(what) {
            crate::info!("gpu[{}]: {what}", self.name);
        }
    }

    fn set_reg(&mut self, space: Space, index: u32, value: u32) {
        if super::frameprobe::on() && self.name == "gpu-gfx" {
            super::frameprobe::reg_written(match space {
                Space::Ctx => 0,
                Space::Sh => 1,
                Space::Uc => 2,
            }, index);
        }
        let (v, n) = match space {
            Space::Ctx => (&mut self.regs.ctx, CTX_DWORDS),
            Space::Sh => (&mut self.regs.sh, SH_DWORDS),
            Space::Uc => (&mut self.regs.uc, UC_DWORDS),
        };
        if (index as usize) < n {
            if space == Space::Ctx && v[index as usize] != value {
                self.ctx_dirty = true;
            }
            if space == Space::Sh && (0x08..0x2C).contains(&index) && v[index as usize] != value {
                self.ps_dirty = true;
            }
            v[index as usize] = value;
            note_high(space, index as usize);
        }
    }

    pub fn num_instances(&self) -> u32 {
        self.num_instances
    }

    pub fn reg(&self, space: Space, index: u32) -> u32 {
        let v = match space {
            Space::Ctx => &self.regs.ctx,
            Space::Sh => &self.regs.sh,
            Space::Uc => &self.regs.uc,
        };
        v.get(index as usize).copied().unwrap_or(0)
    }

    /// Register read by absolute byte address (COPY_DATA/WRITE_DATA register forms).
    fn reg_by_addr(&self, a: u32) -> u32 {
        match a {
            0x28000..=0x2FFFF => self.reg(Space::Ctx, (a - CTX_BASE) / 4),
            0xB000..=0xBFFF => self.reg(Space::Sh, (a - SH_BASE) / 4),
            0x30000..=0x3FFFF => self.reg(Space::Uc, (a - UC_BASE) / 4),
            _ => 0,
        }
    }

    /// Command buffer words or a waited label [a, a+bytes) the command
    /// processor is about to read from guest memory. With homes for reads
    /// (`tune home_reads`) their pages are noted, so that no such home is
    /// made over them (the title's command data shares pools that GPU work
    /// writes: queue 0x50's 4.9 MB particle pools at the 5-3 load, whose
    /// homes held the GPU's writes while the CP ran the guest memory's older
    /// packets and the queues waited on labels in a cycle, 2026-10-24);
    /// GPU-written parts of a home over them are written back first and
    /// waited for.
    fn before_fetch(&mut self, a: u64, bytes: u64) {
        if bytes == 0 {
            return;
        }
        super::homes::note_cp_code(a, bytes);
        if super::homes::any_stale(&[(a, bytes)]) {
            if let Some(e) = self.exec.as_mut() {
                e.flush_if_writes(&[(a, bytes)]);
            }
        }
    }

    /// Executes a command buffer (`size` dwords at `addr`). Small buffers (up
    /// to `FETCH_DWORDS`) are copied when execution starts, as the hardware's
    /// prefetch would have fetched them (a CPU rewrite of a started buffer is
    /// not seen; `DS_CP_FETCH=0` reads them lazily). Larger buffers are read
    /// as they execute: titles patch submitted frame command buffers until
    /// the GPU reaches the patched packets (copying them at submission broke
    /// rendering).
    pub fn run(&mut self, addr: u64, size: u32) {
        if self.depth > 8 {
            crate::error!("gpu[{}]: indirect buffer nesting too deep at {addr:#x}", self.name);
            return;
        }
        self.depth += 1;
        let mut cur = addr;
        let mut end = addr + 4 * size as u64;
        if self.shape() {
            crate::info!("cb shape: enter {addr:#x} ({size} dwords), depth {}, next in submission {}{}", self.depth, self.next_in_submission, self.held_tag());
        }
        self.before_fetch(addr, 4 * size as u64);
        let mut f = Fetched::new(addr, size);
        if cb_census_on() {
            self.cb_note(addr, size);
        }
        // Time by packet class (the graphics queue; `tune cp_classes`).
        let classes = !self.compute && super::tune::cp_classes();
        if classes && !f.words.is_empty() {
            SERIAL_WHY[8].fetch_add(1, Ordering::Relaxed);
        }
        let acb_ops = self.compute && super::tune::cp_classes() && self.name.strip_prefix("gpu-acb").and_then(|h| u64::from_str_radix(h.trim_start_matches("0x"), 16).ok()) == Some(super::tune::cp_classes_acb());
        loop {
            if cur >= end {
                // Past a run, the buffer's end is a stop (what was recorded
                // past it may be recorded again, from the run's end), unless
                // the submission's next buffer goes on with it.
                if (self.past.is_some() || self.past_wait.is_some()) && !self.past_goes_on() {
                    self.held_stop("buffer end", cur);
                    match self.settle_held() {
                        Some(c) if c < end => {
                            cur = c;
                            continue;
                        }
                        _ => {}
                    }
                }
                break;
            }
            // The next chunk of a parallel run: collected here.
            if let Some(pr) = &self.par {
                if pr.starts[pr.next] == cur {
                    let t = class_start(classes);
                    cur = self.par_collect();
                    class_end(8, t);
                    continue;
                }
                if cur > pr.starts[pr.next] {
                    // Execution left the run's packets.
                    self.par_cancel();
                }
            }
            if super::shutting_down() {
                // Process shutdown releases label waits: the work behind
                // them would run without its inputs. Nothing more runs.
                break;
            }
            if self.held.is_some() && self.held_halted(cur) {
                if self.past.is_some() || self.past_wait.is_some() {
                    match self.settle_held() {
                        Some(c) => {
                            cur = c;
                            continue;
                        }
                        None => break,
                    }
                }
                break;
            }
            if let Some(w) = self.past_wait.as_deref_mut() {
                w.ticks += 1;
                if w.ticks % 8 == 0 && self.past_wait_passes() {
                    self.held_stop("promised", cur);
                    match self.settle_held() {
                        Some(c) => {
                            cur = c;
                            continue;
                        }
                        None => break,
                    }
                }
            }
            if !valid(cur, 4) {
                crate::error!("gpu[{}]: command pointer {cur:#x} outside guest memory", self.name);
                break;
            }
            let h = f.rd(cur);
            match h >> 30 {
                3 => {}
                2 => {
                    cur += 4;
                    continue;
                }
                0 => {
                    // Type-0: base register + (count + 1) values.
                    let n = ((h >> 16) & 0x3FFF) as u64 + 1;
                    cur += 4 * (n + 1);
                    self.note("type-0 packet skipped");
                    continue;
                }
                _ => {
                    crate::error!("gpu[{}]: bad packet header {h:#010x} at {cur:#x}", self.name);
                    break;
                }
            }
            let count = (h >> 16) & 0x3FFF;
            let opcode = (h >> 8) & 0xFF;
            if count == 0x3FFF {
                cur += 4; // header-only NOP
                continue;
            }
            let n = count as u64 + 1; // payload dwords
            let next = cur + 4 * (n + 1);
            if next > end + 4 * 0x4000 || !valid(cur, 4 * (n + 1)) {
                crate::error!("gpu[{}]: packet {opcode:#x} at {cur:#x} overruns its buffer", self.name);
                break;
            }
            let skip = self.par_skip.0 == end && cur < self.par_skip.1;
            if self.past.is_some() && self.par.is_none() && !skip && is_command(opcode) && h & 1 == 0 && f.words.is_empty() && !f.live && super::par::workers() > 0 && self.plan_ahead(&f, cur, end) {
                // The next parallel run: planned once this one is collected.
                self.held_stop("next run", cur);
                match self.past_settle() {
                    Some(c) => {
                        cur = c;
                        continue;
                    }
                    None => break,
                }
            }
            if self.par.is_none() && !skip && is_command(opcode) && h & 1 == 0 && f.words.is_empty() && !f.live && self.exec.is_some() && (self.held.is_none() || self.held_runs_ok()) && (if self.compute { compute_workers() > 0 } else { super::par::workers() > 0 }) && !super::capture::capturing() {
                let t = class_start(classes);
                self.par_plan(&f, cur, end);
                class_end(7, t);
                // (Held: planning stopped the stretch before this command.)
                if self.held.is_some() && self.par.is_none() && self.held_halted(cur) {
                    match self.settle_held() {
                        Some(c) => {
                            cur = c;
                            continue;
                        }
                        None => break,
                    }
                }
            }
            PACKETS.fetch_add(1, Ordering::Relaxed);
            if self.main_gfx && !self.compute && self.par.is_none() && op_census_on() {
                let at = cur;
                let p = |i: u64| f.rd(at + 4 * (1 + i));
                op_census(opcode, self.depth, self.held_refuses(opcode, n, &p));
            }
            if self.depth == 1 && self.shape() {
                crate::info!("cb shape: packet {} at {cur:#x}{}", op_name(opcode), self.held_tag());
            }
            self.cur_packet = cur;
            if super::exec::replay_census_on() {
                super::exec::CUR_PACKET.with(|c| c.set(cur));
            }
            self.cur_opcode = opcode;
            if self.main_gfx && !self.compute && super::hscensus::on() {
                let q = |i: u64| f.rd(cur + 4 * (1 + i));
                self.hs_census_packet(opcode, &q, n);
            }
            // The packet as the CP fetches it on reaching it, before it
            // executes: handlers that wait (for GPU work, a label, a display
            // buffer) read their arguments as fetched. Read after the wait,
            // they came from what the title had written into the buffer
            // meanwhile: a FLIP (which waits for the work before it) took its
            // arguments from the rewritten buffer while its first dwords had
            // been read before, the flip failed, and the next frame waited for
            // its display buffer forever (runs 20261012-195508, -191738).
            let mut pk = [0u32; PACKET_FETCH];
            let fetched = if f.words.is_empty() { (n as usize).min(PACKET_FETCH) } else { 0 };
            for (i, w) in pk.iter_mut().enumerate().take(fetched) {
                *w = f.rd(cur + 4 * (1 + i as u64));
            }
            let p = |i: u64| if (i as usize) < fetched { pk[i as usize] } else { f.rd(cur + 4 * (1 + i)) };
            let command = matches!(
                opcode,
                op::DRAW_INDEX_2 | op::DRAW_INDEX_AUTO | op::DRAW_INDEX_OFFSET_2 | op::DRAW_INDIRECT | op::DRAW_INDEX_INDIRECT | op::DISPATCH_DIRECT | op::DISPATCH_INDIRECT
            );
            if command && !self.ahead.is_empty() {
                // The preparations made ahead for this command (another one
                // first: execution took another path than the scan).
                if self.ahead.front().is_some_and(|c| c.at == cur) {
                    let c = self.ahead.pop_front().unwrap();
                    *self.cur_ahead.borrow_mut() = c.stages;
                } else {
                    self.drop_ahead();
                }
            } else if !command
                && !self.ahead.is_empty()
                && !matches!(
                    opcode,
                    op::NOP
                        | op::SET_CONTEXT_REG
                        | op::SET_CONTEXT_REG_INDEX
                        | op::SET_SH_REG
                        | op::SET_SH_REG_INDEX
                        | op::SET_UCONFIG_REG
                        | op::SET_UCONFIG_REG_INDEX
                        | op::LOAD_UCONFIG_REG
                        | op::LOAD_CONTEXT_REG_INDEX
                        | op::LOAD_SH_REG_INDEX
                        | op::INDEX_TYPE
                        | op::INDEX_BASE
                        | op::INDEX_BUFFER_SIZE
                        | op::NUM_INSTANCES
                        | op::CONTEXT_CONTROL
                        | op::SET_BASE
                )
                // Preparations made past GPU-side synchronization, CP writes
                // and label waits (`scan_through`) are checked when used.
                && !matches!(opcode, op::EVENT_WRITE | op::ACQUIRE_MEM | op::RELEASE_MEM | op::WRITE_DATA | op::WAIT_REG_MEM | op::WAIT_REG_MEM64)
            {
                self.drop_ahead();
            }
            if super::capture::capturing() {
                // Synchronization and memory packets in frame captures.
                let name = match opcode {
                    op::WAIT_REG_MEM => Some("WAIT_REG_MEM"),
                    op::WAIT_REG_MEM64 => Some("WAIT_REG_MEM64"),
                    op::WRITE_DATA => Some("WRITE_DATA"),
                    op::RELEASE_MEM => Some("RELEASE_MEM"),
                    op::EVENT_WRITE => Some("EVENT_WRITE"),
                    op::ACQUIRE_MEM => Some("ACQUIRE_MEM"),
                    op::DMA_DATA => Some("DMA_DATA"),
                    op::COPY_DATA => Some("COPY_DATA"),
                    op::COND_EXEC => Some("COND_EXEC"),
                    op::INDIRECT_BUFFER => Some("INDIRECT_BUFFER"),
                    op::PFP_SYNC_ME => Some("PFP_SYNC_ME"),
                    op::DISPATCH_DIRECT | op::DISPATCH_INDIRECT | op::SET_SH_REG | op::SET_CONTEXT_REG | op::SET_UCONFIG_REG => None,
                    _ if std::env::var("DS_CAPTURE_ALL_PACKETS").ok().as_deref() == Some("1") => Some("PACKET"),
                    _ => None,
                };
                if let Some(name) = name {
                    let args: Vec<String> = (0..n.min(8)).map(|i| format!("{:08x}", p(i))).collect();
                    super::capture::packet(self, &format!("{name} {opcode:#x} @{cur:#x} [{}]", args.join(" ")));
                }
            }
            if acb_ops && command {
                let k = if self.par.is_some() {
                    14
                } else if !f.words.is_empty() {
                    0
                } else if f.live {
                    1
                } else if self.par_skip.0 == end && cur < self.par_skip.1 {
                    match self.par_skip_n {
                        0 | 1 => 2,
                        2..=7 => 3,
                        8..=31 => 4,
                        32..=127 => 5,
                        _ => 6,
                    }
                } else if self.held.is_some() {
                    12
                } else if !super::capture::capturing() && self.exec.is_some() {
                    13
                } else {
                    7
                };
                SERIAL_WHY_ACB[k].fetch_add(1, Ordering::Relaxed);
            }
            if classes && command && self.par.is_none() {
                let k = if !f.words.is_empty() {
                    0
                } else if f.live {
                    1
                } else if self.par_skip.0 == end && cur < self.par_skip.1 {
                    match self.par_skip_n {
                        0 | 1 => 2,
                        2..=7 => 3,
                        8..=31 => 4,
                        32..=127 => 5,
                        _ => 6,
                    }
                } else if self.past.is_some() {
                    9
                } else if self.past_wait.is_some() {
                    10
                } else if h & 1 != 0 {
                    11
                } else if self.held.is_some() {
                    12
                } else if !super::capture::capturing() && self.exec.is_some() {
                    13
                } else {
                    7
                };
                SERIAL_WHY[k].fetch_add(1, Ordering::Relaxed);
            }
            let dispatch = matches!(opcode, op::DISPATCH_DIRECT | op::DISPATCH_INDIRECT);
            let class = if command {
                (if self.par.is_some() { 0 } else { 2 }) + dispatch as usize
            } else if is_state(opcode) {
                5
            } else if opcode == op::NOP {
                6
            } else {
                4
            };
            let t = class_start(classes && opcode != op::INDIRECT_BUFFER);
            let ta = class_start(acb_ops && opcode != op::INDIRECT_BUFFER);
            let tw = class_start(classes && self.past_wait.is_some() && opcode != op::INDIRECT_BUFFER);
            // A label wait that blocks prepares the stages after it meanwhile
            // (`tune wait_ahead`).
            if matches!(opcode, op::WAIT_REG_MEM | op::WAIT_REG_MEM64) && !f.live && self.exec.is_some() && self.held.is_none() && super::tune::wait_ahead_for(self.compute) {
                self.wait_ctx = Some((f.clone(), next, end));
            }
            // A dispatch whose pipeline turns out missing scans the commands
            // after it for more first uses (`tune compile_ahead`).
            // (And a draw whose pipeline turns out missing and waits for it: `prefetch_gfx`.)
            if command && !f.live && ((dispatch && super::compile::at_first_uses()) || (matches!(opcode, op::DRAW_INDEX_2 | op::DRAW_INDEX_AUTO | op::DRAW_INDEX_OFFSET_2 | op::DRAW_INDIRECT | op::DRAW_INDEX_INDIRECT | op::DRAW_INDEX_INDIRECT_MULTI | op::DRAW_INDIRECT_MULTI) && super::compile::gfx_ahead())) {
                self.compile_ctx.set(Some((f.base, next, end)));
            }
            if self.held.is_some() {
                if let Some(why) = self.held_refuses(opcode, n, &p) {
                    self.held_stop(why, cur);
                    // Past a run or a wait: the packet runs once what was
                    // recorded past it is settled.
                    if self.past.is_some() || self.past_wait.is_some() {
                        match self.settle_held() {
                            Some(c) => {
                                cur = c;
                                continue;
                            }
                            None => break,
                        }
                    }
                    break;
                }
            }
            let flow = self.packet(opcode, h, cur, n, &p);
            self.wait_ctx = None;
            self.compile_ctx.set(None);
            class_end(class, t);
            acb_op_end(opcode, ta);
            stretch_op_end(opcode, tw);
            if self.held.is_some() {
                // Stopped inside it: it and what follows are left to the
                // main command processor; done: what follows.
                let stopped = self.exec.as_ref().is_some_and(|e| e.stopped) || self.held.as_ref().is_some_and(|h| h.stop.is_some());
                if command && !stopped {
                    if let Some(h) = self.held.as_mut() {
                        h.commands += 1;
                    }
                }
                if self.held_halted(cur) {
                    if self.past.is_some() || self.past_wait.is_some() {
                        match self.settle_held() {
                            Some(c) => {
                                cur = c;
                                continue;
                            }
                            None => break,
                        }
                    }
                    break;
                }
            }
            if command {
                // Preparations of this command that it did not take.
                for s in self.take_ahead().iter().flatten() {
                    s.cancel();
                }
            }
            // (Past a label wait, the queue asking waits for what follows the
            // wait in any case: it gets the label when the stretch settles.)
            if self.past.is_some() && self.exec.as_ref().is_some_and(|e| e.submit_requested() || e.held_label_requested()) {
                // Another queue waits for a label written past the run (in
                // the recording batch or one closed and held since).
                self.held_stop("label", next);
                match self.settle_held() {
                    Some(c) => {
                        cur = c;
                        continue;
                    }
                    None => break,
                }
            }
            if self.held.is_none() && self.exec.as_ref().is_some_and(|e| e.submit_requested()) {
                let ta = class_start(acb_ops);
                self.submit_gpu("label");
                acb_op_end(1, ta);
            }
            if opcode == op::REWIND {
                // The CPU streams packets behind a REWIND: read them live.
                f = Fetched::new(next, 0);
                f.live = true;
            }
            match flow {
                Flow::Next => cur = next,
                Flow::Skip(d) => cur = next + 4 * d as u64,
                Flow::Chain(a, s) => {
                    cur = a;
                    end = a + 4 * s as u64;
                    self.before_fetch(a, 4 * s as u64);
                    f = Fetched::new(a, s);
                    if cb_census_on() {
                        self.cb_note(a, s);
                    }
                }
                Flow::Stop => break,
            }
        }
        // Every stop past a run settles it within its buffer (a stop at the
        // buffer's end included); leaving the buffer with one unsettled
        // would skip the packets after it.
        if (self.past.is_some() || self.past_wait.is_some()) && !self.past_goes_on() && self.rewind.is_none() {
            // (Waits ended by the shutdown leave it as well.)
            if !crate::gpu::shutting_down() {
                crate::error!("gpu[{}]: left a command buffer at {cur:#x} while recording past a parallel run or label wait (settled here)", self.name);
            }
            let _ = self.settle_held();
            self.rewind = None;
        }
        // A parallel run planned in this buffer ends in it (unless execution
        // goes back into it: `take_rewind`, or into an earlier submission:
        // `take_rewind_sub`, a stretch past a run dropped there, the run kept
        // to collect from a chunk's stop; cancelled, the run was planned
        // again at the stop and its chunk stopped there again, endlessly).
        if self.par.is_some() && self.rewind.is_none() && self.rewind_sub.is_none() {
            self.par_cancel();
        }
        if self.shape() {
            crate::info!("cb shape: leave at {cur:#x}, depth {}{}", self.depth, self.held_tag());
        }
        self.depth -= 1;
        if self.depth == 0 {
            // The batch stays open into the next command buffer when that one
            // is already here (`tune cmdbuf_merge`): a batch less to begin.
            let mode = super::tune::cmdbuf_merge();
            let merge = (mode >= 2 || (mode == 1 && !self.compute)) && (self.next_in_submission || self.more_follows.as_ref().is_some_and(|f| f()));
            if merge {
                CMDBUF_MERGED.fetch_add(1, Ordering::Relaxed);
            } else {
                let t = class_start(classes);
                self.submit_gpu("cmdbuf");
                class_end(9, t);
            }
        }
    }

    fn packet(&mut self, opcode: u32, h: u32, at: u64, n: u64, p: &dyn Fn(u64) -> u32) -> Flow {
        crate::placement::note(&|| crate::log::thread_name());
        if self.repeat_ok && !self.keeps_repeat(opcode, n, p) {
            self.repeat_ok = false;
            REPEAT_ENDS[(opcode & 0xFF) as usize].fetch_add(1, Ordering::Relaxed);
        }
        match opcode {
            op::NOP => {
                if n >= 2 && p(0) == DS_MAGIC {
                    return self.private(p(1), at, n - 2, &|i| p(2 + i));
                }
            }
            op::SET_CONTEXT_REG | op::SET_CONTEXT_REG_INDEX => self.set_range(Space::Ctx, p(0) & 0xFFFF, n, p),
            op::SET_SH_REG | op::SET_SH_REG_INDEX => self.set_range(Space::Sh, p(0) & 0xFFFF, n, p),
            op::SET_UCONFIG_REG | op::SET_UCONFIG_REG_INDEX => self.set_range(Space::Uc, p(0) & 0xFFFF, n, p),
            op::SET_CONFIG_REG => self.note("SET_CONFIG_REG ignored"),
            op::SET_SH_REG_OFFSET => self.note("SET_SH_REG_OFFSET ignored"),
            op::LOAD_CONTEXT_REG_INDEX => self.load_pairs(Space::Ctx, addr64(p(0), p(1)), p(2)),
            op::LOAD_SH_REG_INDEX => self.load_pairs(Space::Sh, addr64(p(0), p(1)), p(2)),
            op::LOAD_UCONFIG_REG => self.load_pairs(Space::Uc, addr64(p(0), p(1)), p(2)),
            op::CLEAR_STATE => self.clear_context(),
            op::CONTEXT_CONTROL | op::PFP_SYNC_ME => {}
            // Partial flushes, cache actions and acquires order later work
            // after earlier work: GPU work is already ordered by barriers, so
            // only a following CP memory write has to wait (see `fence`).
            op::EVENT_WRITE | op::ACQUIRE_MEM => {
                self.fence_pending = true;
                self.fence_kind = if opcode == op::ACQUIRE_MEM { 64 } else { (p(0) & 0x3F) as u8 };
                if let Some(e) = self.exec.as_mut() {
                    e.note_sync();
                }
            }
            op::INDEX_TYPE => self.set_reg(Space::Uc, crate::hle::libs::agc_cmd::REG_VGT_INDEX_TYPE, p(0)),
            op::INDEX_BASE => self.index_base = addr64(p(0), p(1)),
            op::INDEX_BUFFER_SIZE => self.index_count = p(0),
            op::NUM_INSTANCES => self.num_instances = p(0).max(1),
            op::SET_BASE => {
                let base = addr64(p(1), p(2));
                if p(0) == 2 { self.dispatch_indirect_base = base } else { self.draw_indirect_base = base }
            }
            op::SET_PREDICATION => {
                let ctl = p(0);
                let pop = (ctl >> 16) & 7;
                let addr = addr64(p(1), p(2));
                self.pred = (pop != 0).then_some(Pred { op: pop, cond: (ctl >> 8) & 1, addr, count: if n >= 4 { p(3) } else { 0 } });
                self.note("SET_PREDICATION recorded but not applied");
                if super::capture::capturing() {
                    let what = self.pred_value().map(|(d, w)| format!("{w} -> {}", if d { "draw" } else { "skip" })).unwrap_or_else(|| "clear".into());
                    super::capture::packet(self, &format!("SET_PREDICATION op {pop} cond {} wait {} addr {addr:#x} count {}: {what}", (ctl >> 8) & 1, (ctl >> 12) & 1, if n >= 4 { p(3) } else { 0 }));
                }
            }
            op::DRAW_INDEX_2 | op::DRAW_INDEX_AUTO | op::DRAW_INDEX_OFFSET_2 | op::DRAW_INDIRECT | op::DRAW_INDEX_INDIRECT
            | op::DRAW_INDIRECT_MULTI | op::DRAW_INDEX_INDIRECT_MULTI | op::DRAW_INDEX_MULTI_AUTO => {
                DRAWS.fetch_add(1, Ordering::Relaxed);
                super::exec::draw::TRACE_DRAW.with(|c| c.set(false));
                let instances = self.num_instances;
                let mut multi: Option<String> = None;
                let mut indirect: Option<String> = None;
                let args = match opcode {
                    op::DRAW_INDEX_AUTO => Some(super::exec::DrawArgs { count: p(0), instances, index: None, base_vertex: None, first_instance: 0, base_inst: None }),
                    op::DRAW_INDEX_2 => self.index_size().map(|sz| super::exec::DrawArgs { count: p(3), instances, index: Some((addr64(p(1), p(2)), sz)), base_vertex: None, first_instance: 0, base_inst: None }),
                    op::DRAW_INDEX_OFFSET_2 => self.index_size().map(|sz| super::exec::DrawArgs { count: p(2), instances, index: Some((self.index_base + p(1) as u64 * sz as u64, sz)), base_vertex: None, first_instance: 0, base_inst: None }),
                    op::DRAW_INDIRECT | op::DRAW_INDEX_INDIRECT => {
                        // Arguments may have been written by earlier GPU work.
                        let a = self.draw_indirect_base + p(0) as u64;
                        let ta = super::exec::draw::trace_args_addr();
                        let traced = ta != 0 && a & !0xFFF == ta & !0xFFF;
                        super::exec::draw::TRACE_DRAW.with(|c| c.set(traced));
                        if traced && valid(a, 20) {
                            let w = unsafe { [rd32(a), rd32(a + 4), rd32(a + 8), rd32(a + 12), rd32(a + 16)] };
                            crate::info!("trace args {a:#x} [{}] for flip {}: {w:?} vs {:#x} ps {:#x} locs {:#x} repeat_ok {}", self.name, crate::hle::libs::videoout::QUEUED_FLIP.load(Ordering::Relaxed) + 1, super::shaders::ngg_addr(self), super::shaders::ps_addr(self), p(1), self.repeat_ok);
                        }
                        let laps = (!self.compute && super::tune::cp_classes()).then(|| {
                            IDRAW_GROUP.with(|g| g.set(if !self.main_gfx { 2 } else if self.past_wait.is_some() { 1 } else { 0 }));
                            Instant::now()
                        });
                        self.sync_reads(&[(a, if opcode == op::DRAW_INDEX_INDIRECT { 20 } else { 16 })]);
                        idraw_part(0, laps);
                        let laps = laps.map(|_| Instant::now());
                        // The CP hands the base vertex and start instance to the
                        // shader through the user SGPRs the packet names
                        // (BASE_VTX_LOC [15:0], START_INST_LOC [31:16]: SH
                        // register offsets; as Mesa radeonsi emits them), and
                        // the instance id VGPR then starts at 0. Our Agc
                        // packets name no locations: the start instance goes
                        // to the bound shader's instance-offset slot.
                        let (vtx_loc, inst_loc) = (p(1) & 0xFFFF, p(1) >> 16);
                        let inst_loc = if inst_loc != 0 { inst_loc } else { self.instance_offset_loc() };
                        let args = if opcode == op::DRAW_INDIRECT && valid(a, 16) {
                            let w = unsafe { [rd32(a), rd32(a + 4), rd32(a + 8), rd32(a + 12)] };
                            if super::capture::capturing() {
                                indirect = Some(format!("args {w:?} locs {vtx_loc:#x}/{inst_loc:#x}"));
                            }
                            Some((super::exec::DrawArgs { count: w[0], instances: w[1], index: None, base_vertex: Some(w[2] as i32), first_instance: w[3], base_inst: None }, w[2], w[3]))
                        } else if let Some(w) = (opcode == op::DRAW_INDEX_INDIRECT && valid(a, 20)).then(|| unsafe { [rd32(a), rd32(a + 4), rd32(a + 8), rd32(a + 12), rd32(a + 16)] }).and_then(|w| self.checked_index_args(a, w)) {
                            if super::verify::args_check_on() {
                                self.args_checked = Some((a, w));
                            }
                            if super::drawring::on() {
                                super::drawring::note(super::shaders::ngg_addr(self), super::shaders::ps_addr(self), a, w, self.index_base, self.index_count);
                            }
                            if super::capture::capturing() {
                                let ty = self.reg(Space::Uc, crate::hle::libs::agc_cmd::REG_VGT_INDEX_TYPE) & 3;
                                indirect = Some(format!("args {w:?} locs {vtx_loc:#x}/{inst_loc:#x} index {:#x} type {ty} size {}{}", self.index_base, self.index_count, self.index_range(w[2], w[0])));
                            }
                            let sz = self.index_size();
                            let ok = sz.is_some() && self.cpu_args_base_ok(vtx_loc, inst_loc);
                            idraw_part(1, laps);
                            match sz {
                                // The start instance as Vulkan's first instance
                                // (`tune cpu_args_base`): drawn here.
                                Some(sz) if ok => {
                                    multi = Some(self.draw_cpu_args_base(w, sz, inst_loc));
                                    None
                                }
                                sz => sz.map(|sz| (super::exec::DrawArgs { count: w[0], instances: w[1], index: Some((self.index_base + w[2] as u64 * sz as u64, sz)), base_vertex: Some(w[3] as i32), first_instance: w[4], base_inst: None }, w[3], w[4])),
                            }
                        } else {
                            None
                        };
                        if let Some((_, bv, si)) = args {
                            if vtx_loc != 0 {
                                self.set_reg(Space::Sh, vtx_loc, bv);
                            }
                            if inst_loc != 0 {
                                self.set_reg(Space::Sh, inst_loc, si);
                            }
                        }
                        args.map(|(mut a, _, _)| {
                            if inst_loc != 0 {
                                a.first_instance = 0;
                            }
                            a
                        })
                    }
                    op::DRAW_INDIRECT_MULTI | op::DRAW_INDEX_INDIRECT_MULTI => {
                        multi = Some(self.draw_indirect_multi(opcode == op::DRAW_INDEX_INDIRECT_MULTI, p));
                        None
                    }
                    _ => {
                        self.note("multi-draw packets are not rendered yet");
                        None
                    }
                };
                // (`DS_DRAW_RING`: direct draws too; the opcode stands in the arguments' column.)
                if super::drawring::on() && matches!(opcode, op::DRAW_INDEX_2 | op::DRAW_INDEX_OFFSET_2 | op::DRAW_INDEX_AUTO) {
                    if let Some(a) = args.as_ref() {
                        super::drawring::note(super::shaders::ngg_addr(self), super::shaders::ps_addr(self), opcode as u64, [a.count, a.instances, 0, 0, 0], a.index.map_or(0, |i| i.0), self.index_count);
                    }
                }
                self.last_draw = multi.take().map_or(std::borrow::Cow::Borrowed("not executed"), std::borrow::Cow::Owned);
                let args_checked = self.args_checked.take();
                match args {
                    Some(a) => {
                        self.draw(a);
                        if super::capture::draw_dumps() || super::capture::watching() || super::capture::listed_draw(self) {
                            // Per-draw target dumps: complete the draw first.
                            if let Some(e) = self.exec.as_mut() {
                                e.flush();
                            }
                        }
                    }
                    None if matches!(opcode, op::DRAW_INDEX_2 | op::DRAW_INDEX_OFFSET_2 | op::DRAW_INDEX_INDIRECT) && self.index_size().is_none() => {
                        self.note("draws with 8-bit indices are not rendered yet")
                    }
                    None => {}
                }
                // (`DS_ARGS_CHECK`: the arguments read, checked when the draw's batch completes.)
                if let Some((aa, w)) = args_checked {
                    let vs = super::shaders::ngg_addr(self);
                    if let Some(e) = self.exec.as_mut() {
                        e.add_args_check(aa, &w, vs);
                    }
                }
                if super::capture::capturing() {
                    let what = match opcode {
                        op::DRAW_INDEX_2 => format!(
                            "DRAW_INDEX_2 count {} index {:#x} type {} instances {}",
                            p(3),
                            addr64(p(1), p(2)),
                            self.reg(Space::Uc, crate::hle::libs::agc_cmd::REG_VGT_INDEX_TYPE) & 3,
                            self.num_instances
                        ),
                        op::DRAW_INDEX_AUTO => format!("DRAW_INDEX_AUTO count {} instances {}", p(0), self.num_instances),
                        op::DRAW_INDEX_OFFSET_2 => format!("DRAW_INDEX_OFFSET_2 count {} offset {} base {:#x}", p(2), p(1), self.index_base),
                        _ => format!("indirect draw opcode {opcode:#x} offset {:#x} base {:#x}{}", p(0), self.draw_indirect_base, indirect.take().map(|s| format!(" {s}")).unwrap_or_default()),
                    };
                    let what = match (h & 1 != 0).then(|| self.pred_value()) {
                        Some(Some((d, w))) => format!("{what} predicated ({w}: {})", if d { "draw" } else { "skip" }),
                        Some(None) => format!("{what} predicated (no predication set)"),
                        None => what,
                    };
                    super::capture::draw(self, &what);
                }
            }
            op::DISPATCH_DIRECT | op::DISPATCH_INDIRECT => {
                DISPATCHES.fetch_add(1, Ordering::Relaxed);
                let mut indirect_groups = None;
                let mut indirect_extra: Option<[u32; 2]> = None;
                let mut gpu_read = false;
                if opcode == op::DISPATCH_DIRECT {
                    // DS_DELAY_CS=<shader>:<ms>: wait before that shader's
                    // dispatches (diagnosis of CPU/GPU data races).
                    static DELAY: std::sync::OnceLock<Option<(Vec<u64>, u64)>> = std::sync::OnceLock::new();
                    if let Some((list, ms)) = DELAY.get_or_init(|| {
                        let v = std::env::var("DS_DELAY_CS").ok()?;
                        let (a, ms) = v.split_once(':')?;
                        Some((a.split(',').filter_map(|x| u64::from_str_radix(x.trim_start_matches("0x"), 16).ok()).collect(), ms.parse().ok()?))
                    }) {
                        let cs = ((self.reg(Space::Sh, 0x20C) as u64) | ((self.reg(Space::Sh, 0x20D) as u64 & 0xFF) << 32)) << 8;
                        if list.contains(&cs) {
                            std::thread::sleep(std::time::Duration::from_millis(*ms));
                        }
                    }
                    super::emucheck::race_check(self, p(3));
                    super::emucheck::race_watch(self, p(3));
                    // DS_EMU_CHECK: interpret a sample of the groups first.
                    let check = if super::exec::enabled() && super::emucheck::wanted(self) {
                        self.sync();
                        super::emucheck::prepare(self, [p(0), p(1), p(2)], p(3))
                    } else {
                        None
                    };
                    self.dispatch([p(0), p(1), p(2)], p(3));
                    if let Some(c) = check {
                        self.sync();
                        super::emucheck::compare(c);
                    }
                } else {
                    // Graphics ring: [offset from the SET_BASE base, initiator];
                    // compute queues: [address lo, address hi, initiator] (the
                    // form sceAgcAcbDispatchIndirect emits).
                    let (a, initiator) = if n >= 3 { (addr64(p(0), p(1)), p(2)) } else { (self.dispatch_indirect_base + p(0) as u64, p(1)) };
                    // Arguments that GPU work still writes: the GPU reads them
                    // (captures and interpreter checks need the group counts).
                    let gpu = valid(a, 12) && !super::capture::capturing() && !(super::exec::enabled() && super::emucheck::wanted(self)) && self.dispatch_gpu_indirect(a, initiator);
                    gpu_read = gpu;
                    if !gpu {
                        // Arguments may have been written by earlier GPU work.
                        self.sync_reads(&[(a, 12)]);
                        if !valid(a, 12) {
                            crate::warn!("gpu[{}]: DISPATCH_INDIRECT arguments at {a:#x} outside guest memory (dispatch skipped)", self.name);
                        }
                        if valid(a, 12) {
                            let g = unsafe { [rd32(a), rd32(a + 4), rd32(a + 8)] };
                            super::exec::note_cpu_args(self, g);
                            indirect_groups = Some(g);
                            // Captures: the two dwords after the group counts
                            // (titles keep item counts there).
                            if super::capture::capturing() && valid(a + 12, 8) {
                                indirect_extra = Some(unsafe { [rd32(a + 12), rd32(a + 16)] });
                            }
                            // DS_EMU_CHECK: interpret a sample of the groups first
                            // (after the earlier work has completed).
                            let check = if super::exec::enabled() && super::emucheck::wanted(self) {
                                self.sync();
                                super::emucheck::prepare(self, g, initiator)
                            } else {
                                None
                            };
                            super::exec::set_cpu_indirect(true);
                            self.dispatch(g, initiator);
                            super::exec::set_cpu_indirect(false);
                            if let Some(c) = check {
                                self.sync();
                                super::emucheck::compare(c);
                            }
                        }
                    }
                }
                // DS_TRACE_CS=<shader>[,<shader>..]: the first 8 dispatches of
                // each of those compute shaders log their resources and what
                // the image cache holds for each image (diagnostics).
                if !trace_cs_list().is_empty() {
                    let cs = ((self.reg(Space::Sh, 0x20C) as u64) | ((self.reg(Space::Sh, 0x20D) as u64 & 0xFF) << 32)) << 8;
                    if trace_cs_list().contains(&cs) {
                        static SEEN: parking_lot::Mutex<Vec<(u64, u32)>> = parking_lot::Mutex::new(Vec::new());
                        let n = {
                            let mut g = SEEN.lock();
                            match g.iter_mut().find(|e| e.0 == cs) {
                                Some(e) => {
                                    e.1 += 1;
                                    e.1
                                }
                                None => {
                                    g.push((cs, 1));
                                    1
                                }
                            }
                        };
                        if n <= 8 {
                            let mut lines = Vec::new();
                            if let Ok(p) = super::shaders::prepare_compute(self, true) {
                                lines.extend(super::shaders::describe_resources(&p));
                                for r in &p.resources {
                                    if let super::shaders::Resolved::Image(t) = r {
                                        lines.push(format!("image {:#x}: {}", t.base, super::images::with(|im| im.describe(t))));
                                    }
                                }
                            }
                            crate::info!(
                                "gpu[{}]: trace cs {cs:#x} dispatch #{n} {}x{}x{} at flip {}:\n    {}",
                                self.name,
                                p(0),
                                p(1),
                                p(2),
                                super::FRAMES.load(Ordering::Relaxed),
                                lines.join("\n    ")
                            );
                        }
                    }
                }
                if super::capture::dispatch_dumps() || super::capture::watching() {
                    // Per-dispatch image dumps and pixel watches: complete
                    // the dispatch first.
                    if let Some(e) = self.exec.as_mut() {
                        e.flush();
                    }
                }
                if super::capture::capturing() {
                    let what = if opcode == op::DISPATCH_DIRECT {
                        format!("DISPATCH_DIRECT {}x{}x{} initiator {:#x}", p(0), p(1), p(2), p(3))
                    } else {
                        let g = indirect_groups.map(|g| format!(" groups {}x{}x{}", g[0], g[1], g[2])).unwrap_or_else(|| if gpu_read { " (groups read by the GPU)".into() } else { " (skipped)".into() });
                        let g = match indirect_extra {
                            Some(x) => format!("{g} (then {:#x} {:#x})", x[0], x[1]),
                            None => g,
                        };
                        if n >= 3 {
                            format!("DISPATCH_INDIRECT args {:#x}{g}", addr64(p(0), p(1)))
                        } else {
                            format!("DISPATCH_INDIRECT {:#x} base {:#x}{g}", p(0), self.dispatch_indirect_base)
                        }
                    };
                    super::capture::dispatch(self, &what);
                    // DS_REPEAT_CS=<shader>: run that captured direct dispatch
                    // again on the same inputs (determinism checks with
                    // DS_CAPTURE_DISPATCHES).
                    let cs = ((self.reg(Space::Sh, 0x20C) as u64) | ((self.reg(Space::Sh, 0x20D) as u64 & 0xFF) << 32)) << 8;
                    if opcode == op::DISPATCH_DIRECT && std::env::var("DS_REPEAT_CS").ok().and_then(|v| u64::from_str_radix(v.trim_start_matches("0x"), 16).ok()) == Some(cs) {
                        self.dispatch([p(0), p(1), p(2)], p(3));
                        if let Some(e) = self.exec.as_mut() {
                            e.flush();
                        }
                        super::capture::dispatch(self, &format!("{what} (repeated)"));
                    }
                }
            }
            // CP memory writes wait for shaders only after a partial flush or
            // acquire (titles synchronize with those or RELEASE_MEM first).
            op::WRITE_DATA => self.write_data(n, p),
            op::COPY_DATA => self.copy_data(p),
            op::DMA_DATA => self.dma_data(p),
            op::RELEASE_MEM => self.release_mem(p),
            op::WAIT_REG_MEM => {
                if !super::tune::lazy_wait_submit() && self.held.is_none() {
                    self.submit_gpu("wait");
                }
                let c = p(0);
                let a = addr64(p(1), p(2));
                let flow = self.wait(c, a, false, p(3) as u64, p(4) as u64, at);
                self.end_copy_scope_after_wait(a, false);
                return flow;
            }
            op::WAIT_REG_MEM64 => {
                if !super::tune::lazy_wait_submit() && self.held.is_none() {
                    self.submit_gpu("wait");
                }
                let c = p(0);
                let a = addr64(p(1), p(2));
                let flow = self.wait(c, a, true, addr64(p(3), p(4)), addr64(p(5), p(6)), at);
                self.end_copy_scope_after_wait(a, true);
                return flow;
            }
            op::COND_EXEC => {
                self.sync();
                let a = addr64(p(0), p(1));
                // Another queue's unfinished batch may write it (queues run
                // ahead on promises).
                self.sync_reads(&[(a, 4)]);
                let skip = p(3) & 0x3FFF;
                // DS_TRACE_COND=1 (diagnosis): each COND_EXEC, its word and outcome.
                static TRACE: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
                if *TRACE.get_or_init(|| std::env::var("DS_TRACE_COND").is_ok_and(|v| v == "1")) {
                    let v = if valid(a, 4) { unsafe { rd32(a) } } else { 0xdead_beef };
                    crate::info!("trace cond [{}] for flip {}: COND_EXEC at {at:#x} word {a:#x} = {v:#x}, {} dwords {}", self.name, crate::hle::libs::videoout::QUEUED_FLIP.load(Ordering::Relaxed) + 1, skip, if v == 0 { "skipped" } else { "run" });
                }
                if valid(a, 4) && unsafe { rd32(a) } == 0 {
                    return Flow::Skip(skip);
                }
            }
            op::INDIRECT_BUFFER => {
                let a = addr64(p(0) & !3, p(1));
                let c = p(2);
                let size = c & 0xF_FFFF;
                if c & (1 << 20) != 0 {
                    return Flow::Chain(a, size);
                }
                self.run(a, size);
            }
            op::REWIND => {
                if !self.compute {
                    super::precache::end_scope(2);
                    self.fe2_sync(None);
                }
                // The parser stalls until the CPU marks the packet valid.
                if p(0) & (1 << 31) == 0 {
                    return self.poll_until(at, &format!("REWIND at {at:#x} (waiting for the valid bit)"), || unsafe { rd32(at + 4) } & (1 << 31) != 0);
                }
            }
            _ => {
                if self.unknown_seen.insert(opcode) {
                    let words: Vec<String> = (0..n.min(8)).map(|i| format!("{:08x}", p(i))).collect();
                    crate::warn!("gpu[{}]: unhandled PM4 opcode {opcode:#04x} (header {h:#010x}, {n} dwords: {}) at {at:#x}", self.name, words.join(" "));
                }
            }
        }
        Flow::Next
    }

    fn set_range(&mut self, space: Space, first: u32, n: u64, p: &dyn Fn(u64) -> u32) {
        for i in 1..n {
            self.set_reg(space, first + (i - 1) as u32, p(i));
        }
    }

    /// Our indirect register packets: memory holds `n` {offset, value} pairs.
    fn load_pairs(&mut self, space: Space, a: u64, n: u32) {
        if n == 0 {
            return;
        }
        if !valid(a, 8 * n as u64) {
            crate::error!("gpu[{}]: register pairs at {a:#x} ({n}) outside guest memory", self.name);
            return;
        }
        if let (Some(e), true) = (self.exec.as_mut(), self.held.is_some()) {
            e.note_held_reads(&[(a, 8 * n as u64)], 5);
        }
        for i in 0..n as u64 {
            let (o, v) = unsafe { (rd32(a + 8 * i), rd32(a + 8 * i + 4)) };
            self.set_reg(space, o & 0xFFFF, v);
        }
    }

    fn write_data(&mut self, n: u64, p: &dyn Fn(u64) -> u32) {
        let c = p(0);
        let dst = (c >> 8) & 0xF;
        let no_incr = (c >> 16) & 1 != 0;
        let a = addr64(p(1), p(2));
        let count = n.saturating_sub(3);
        match dst {
            1 | 2 | 5 => {
                if !valid(a, 4 * count.max(1)) {
                    crate::error!("gpu[{}]: WRITE_DATA to {a:#x} outside guest memory", self.name);
                    return;
                }
                if count == 0 {
                    return;
                }
                let vals: Vec<u32> = (0..count).map(|i| p(3 + i)).collect();
                let (name, at) = (self.name.clone(), self.cur_packet);
                let len = if no_incr { 4 } else { 4 * count };
                let label = match (count, no_incr) {
                    (1, _) => Some((vals[0] as u64, 4u8)),
                    (2, false) => Some((vals[0] as u64 | (vals[1] as u64) << 32, 8u8)),
                    _ => None,
                };
                // Without address increment only the last value stays.
                let gpu_vals = if no_incr { vec![*vals.last().unwrap()] } else { vals.clone() };
                if self.gpu_cp_write(a, &gpu_vals, label, false) {
                    return;
                }
                let seq = self.effect(
                    false,
                    Some((a, len)),
                    Box::new(move || {
                        for (i, v) in vals.iter().enumerate() {
                            let d = if no_incr { a } else { a + 4 * i as u64 };
                            unsafe { wr32(d, *v) };
                        }
                        if vals.len() <= 4 {
                            note_write(&name, a, vals[0] as u64 | (vals.get(1).copied().unwrap_or(0) as u64) << 32, at);
                        }
                    }),
                );
                if let (Some(s), Some((v, bytes))) = (seq, label) {
                    promise(s, a, v, bytes, true, self.fe2_promise_view());
                }
            }
            0 => {
                for i in 0..count {
                    let r = (a as u32).wrapping_add(if no_incr { 0 } else { 4 * i as u32 });
                    self.write_reg_addr(r, p(3 + i));
                }
            }
            _ => {
                if self.unknown_seen.insert(0x1_0000 | dst) {
                    let words: Vec<String> = (0..n.min(6)).map(|i| format!("{:08x}", p(i))).collect();
                    crate::warn!("gpu[{}]: WRITE_DATA with dst_sel {dst} ignored (addr {a:#x}; {})", self.name, words.join(" "));
                }
            }
        }
    }

    fn write_reg_addr(&mut self, r: u32, v: u32) {
        match r {
            0x28000..=0x2FFFF => self.set_reg(Space::Ctx, (r - CTX_BASE) / 4, v),
            0xB000..=0xBFFF => self.set_reg(Space::Sh, (r - SH_BASE) / 4, v),
            0x30000..=0x3FFFF => self.set_reg(Space::Uc, (r - UC_BASE) / 4, v),
            // Dword-index form (MMIO offsets).
            _ if r < 0x10000 => self.write_reg_addr(r * 4, v),
            _ => {}
        }
    }

    fn copy_data(&mut self, p: &dyn Fn(u64) -> u32) {
        let c = p(0);
        let src = c & 0xF;
        let dst = (c >> 8) & 0xF;
        let wide = (c >> 16) & 1 != 0;
        let src_v = addr64(p(1), p(2));
        let dst_a = addr64(p(3), p(4));
        if matches!(src, 1 | 2) && !valid(src_v, 8) {
            crate::error!("gpu[{}]: COPY_DATA from {src_v:#x} outside guest memory", self.name);
            return;
        }
        if !matches!(src, 0 | 1 | 2 | 5 | 9) {
            self.note("COPY_DATA from GDS/perf counters reads 0");
        }
        let reg = if src == 0 { self.reg_by_addr(src_v as u32) as u64 } else { 0 };
        let read = move || -> u64 {
            match src {
                0 => reg,
                1 | 2 => {
                    if wide { unsafe { rd64(src_v) } } else { (unsafe { rd32(src_v) }) as u64 }
                }
                5 => src_v,
                9 => gpu_clock(),
                _ => 0,
            }
        };
        match dst {
            1 | 2 | 5 => {
                if !valid(dst_a, 8) {
                    crate::error!("gpu[{}]: COPY_DATA to {dst_a:#x} outside guest memory", self.name);
                    return;
                }
                let (name, at) = (self.name.clone(), self.cur_packet);
                let seq = self.effect(
                    false,
                    Some((dst_a, if wide { 8 } else { 4 })),
                    Box::new(move || {
                        let value = read();
                        if wide { unsafe { wr64(dst_a, value) } } else { unsafe { wr32(dst_a, value as u32) } }
                        note_write(&name, dst_a, value, at);
                    }),
                );
                if let (Some(s), true) = (seq, matches!(src, 1 | 2)) {
                    super::exec::note_cp_source(s, (src_v, if wide { 8 } else { 4 }));
                }
            }
            0 => {
                // A register needs the value now: memory sources must be current.
                if matches!(src, 1 | 2) && self.exec.as_ref().is_some_and(|e| e.busy()) {
                    self.sync();
                }
                let v = read();
                self.write_reg_addr(dst_a as u32, v as u32);
            }
            _ => self.note("COPY_DATA to GDS/other ignored"),
        }
    }

    fn dma_data(&mut self, p: &dyn Fn(u64) -> u32) {
        let c1 = p(0);
        let c6 = p(5);
        let src_sel = (c1 >> 29) & 3;
        let dst_sel = (c1 >> 20) & 3;
        let bytes = (c6 & 0x03FF_FFFF) as u64;
        let sas = (c6 >> 26) & 1 != 0;
        let das = (c6 >> 27) & 1 != 0;
        let saic = (c6 >> 28) & 1 != 0;
        let daic = (c6 >> 29) & 1 != 0;
        let src = addr64(p(1), p(2));
        let dst = addr64(p(3), p(4));
        if bytes == 0 {
            return;
        }
        if dst_sel == 1 || das {
            self.note("DMA_DATA to GDS/registers ignored");
            return;
        }
        if !valid(dst, if daic { 4 } else { bytes }) {
            crate::error!("gpu[{}]: DMA_DATA to {dst:#x}+{bytes:#x} outside guest memory", self.name);
            return;
        }
        let (name, at) = (self.name.clone(), self.cur_packet);
        let range = Some((dst, if daic { 4 } else { bytes }));
        super::precache::invalidate(dst, if daic { 4 } else { bytes });
        super::exec::trace_writes(|| format!("{} DMA_DATA (src_sel {src_sel}, from {src:#x})", self.name), dst, bytes);
        if super::exec::write_journal_on() {
            super::exec::journal_write(dst, if daic { 4 } else { bytes as u64 }, &self.name, src, u32::MAX, "CP DMA", false);
        }
        if super::exec::enabled() {
            // (Held: the packet's effect is dropped and it runs again later, `effect`; its value
            // is noted then, a note now would let another queue's use apply it first.)
            let held = self.held.is_some();
            super::images::with(|im| {
                // A fill of memory with an immediate dword of one byte value (a clear code): a
                // fast clear over metadata it covers has that value, no read after the GPU
                // work; other dwords are other data, read at the image's use (2026-11-18 night).
                // (Its own range noted as the fill alone: `note_meta_writes_but_fill`.)
                let fill = !held && src_sel == 2 && dst_sel == 0 && !das && !daic && super::images::uniform_word(&[src as u32]).is_some();
                if fill {
                    im.note_meta_writes_but_fill(&[(dst, bytes as u64)], dst, bytes as u64);
                } else {
                    im.note_meta_writes(&[(dst, if daic { 4 } else { bytes as u64 })]);
                }
                im.note_overwrites(&[(dst, if daic { 4 } else { bytes as u64 })]);
                if fill {
                    im.note_meta_fill(dst, bytes, src as u32);
                }
            });
        }
        match src_sel {
            2 => {
                // Fill with the immediate dword.
                let v = src as u32;
                self.effect(
                    false,
                    range,
                    Box::new(move || {
                        if daic {
                            unsafe { wr32(dst, v) };
                            note_write(&name, dst, v as u64, at);
                        } else {
                            for i in 0..bytes / 4 {
                                unsafe { wr32(dst + 4 * i, v) };
                            }
                            note_block(&name, dst, bytes, at);
                        }
                    }),
                );
            }
            0 | 3 if !sas => {
                if !valid(src, if saic { 4 } else { bytes }) {
                    crate::error!("gpu[{}]: DMA_DATA from {src:#x}+{bytes:#x} outside guest memory", self.name);
                    return;
                }
                let seq = self.effect(
                    false,
                    range,
                    Box::new(move || {
                        if saic || daic {
                            for i in 0..bytes / 4 {
                                let s = if saic { src } else { src + 4 * i };
                                let d = if daic { dst } else { dst + 4 * i };
                                unsafe { wr32(d, rd32(s)) };
                            }
                        } else {
                            unsafe { std::ptr::copy(src as *const u8, dst as *mut u8, bytes as usize) };
                            // Shared copies checked in this epoch are stale now.
                            super::precache::invalidate(dst, bytes);
                        }
                        note_block(&name, dst, if daic { 4 } else { bytes }, at);
                    }),
                );
                if let Some(s) = seq {
                    super::exec::note_cp_source(s, (src, if saic { 4 } else { bytes }));
                }
            }
            _ => {
                // Register or GDS source: register values / counters.
                let v = if sas { self.reg_by_addr(src as u32) } else { 0 };
                self.note("DMA_DATA from registers/GDS: register value or 0");
                super::replay::note_cp_write(dst, if daic { 4 } else { bytes });
                super::forward::written(dst, if daic { 4 } else { bytes });
                for i in 0..(bytes / 4).max(1) {
                    unsafe { wr32(if daic { dst } else { dst + 4 * i }, v) };
                }
                // (Front end 2: the mirror no longer holds what it changed, as
                // `effect` says for the other forms.)
                if super::fe2::on() {
                    super::fe2::invalidate(dst, if daic { 4 } else { bytes });
                }
                cpw_note(&self.name, dst, if daic { 4 } else { bytes }, self.cur_packet, self.cur_opcode);
            }
        }
    }

    fn release_mem(&mut self, p: &dyn Fn(u64) -> u32) {
        if let Some(e) = self.exec.as_mut() {
            e.note_sync();
        }
        let data_cntl = p(1);
        let int_sel = (data_cntl >> 24) & 7;
        let data_sel = (data_cntl >> 29) & 7;
        RELEASE_KINDS[(data_sel * 8 + int_sel) as usize].fetch_add(1, Ordering::Relaxed);
        let a = addr64(p(2) & !3, p(3));
        let data = addr64(p(4), p(5));
        let ctx = p(6);
        if data_sel > 4 {
            self.note("RELEASE_MEM with GDS data ignored");
        }
        let write = matches!(data_sel, 1..=4);
        if write && !valid(a, 8) {
            crate::error!("gpu[{}]: RELEASE_MEM to {a:#x} outside guest memory", self.name);
            return;
        }
        let (name, at) = (self.name.clone(), self.cur_packet);
        if matches!(data_sel, 1 | 2) && (int_sel == 0 || int_sel == 3) {
            let (vals, label) = if data_sel == 1 { (vec![data as u32], (data & 0xFFFF_FFFF, 4u8)) } else { (vec![data as u32, (data >> 32) as u32], (data, 8u8)) };
            if self.gpu_cp_write(a, &vals, Some(label), true) {
                return;
            }
        }
        // Neither a write nor an interrupt: nothing the CPU would do after the
        // GPU work (deferring it ended the batch for an empty action).
        if !write && (int_sel == 0 || int_sel == 3) {
            return;
        }
        // The title's GPU timestamps (the GPU clock without an interrupt):
        // written now, in the clock `gpu_clock` gives, by the command
        // processor that meets them (FINDINGS.md 2026-11-10 late morning: the
        // swamp x1.037, the vista x1.026 against bottom-of-pipe queries stored
        // by a kernel; the title does not react to the values, 2026-11-04).
        if matches!(data_sel, 3 | 4) && (int_sel == 0 || int_sel == 3) {
            cpu_timestamp(a);
            self.fence_pending = false;
            return;
        }
        let seq = self.effect(
            true,
            write.then_some((a, 8)),
            Box::new(move || {
                if write {
                    match data_sel {
                        1 => unsafe { wr32(a, data as u32) },
                        2 => unsafe { wr64(a, data) },
                        _ => unsafe { wr64(a, gpu_clock()) },
                    }
                    note_write(&name, a, data, at);
                }
                // Not for a replayed frame (`gpu::replay`): the title is held.
                if int_sel != 0 && int_sel != 3 && !super::replay::replaying() {
                    EOP_INTERRUPTS.fetch_add(1, Ordering::Relaxed);
                    crate::hle::libs::agcdriver::raise_eop(ctx);
                }
            }),
        );
        if let Some(s) = seq {
            match data_sel {
                1 => promise(s, a, data & 0xFFFF_FFFF, 4, true, self.fe2_promise_view()),
                2 => promise(s, a, data, 8, true, self.fe2_promise_view()),
                _ => {}
            }
        }
    }

    /// Polls `addr` until the comparison holds (WAIT_REG_MEM).
    fn wait(&mut self, control: u32, a: u64, wide: bool, reference: u64, mask: u64, at: u64) -> Flow {
        if self.held.is_some() {
            return self.held_wait(control, a, wide, reference, mask);
        }
        let func = control & 7;
        let mem = (control >> 4) & 3 == 1;
        if !mem {
            let v = self.reg_by_addr(a as u32) as u64;
            if !compare(func, v & mask, reference & mask) {
                self.note("WAIT_REG_MEM on a register that does not match: continuing");
            }
            return Flow::Next;
        }
        if !valid(a, 8) {
            crate::error!("gpu[{}]: WAIT_REG_MEM on {a:#x} outside guest memory", self.name);
            return Flow::Next;
        }
        // The label is read from guest memory: as for command buffers
        // (`before_fetch`), no home for reads goes over it, and GPU-written
        // parts of one already there are written back first.
        self.before_fetch(a, 8);
        let read = move || if wide { unsafe { rd64(a) } } else { (unsafe { rd32(a) }) as u64 };
        WAITED.lock().get_or_insert_with(HashSet::new).insert(a & !7);
        self.last_wait_gpu = false;
        self.last_wait_kind = b'-';
        self.last_wait_func = func;
        let v0 = read();
        if compare(func, v0 & mask, reference & mask) {
            self.last_wait_kind = b'I';
            label_event(&self.name, b'I', a, v0, reference, at);
            if let Some(e) = self.exec.as_mut() {
                e.xq_label(a, wide, v0);
            }
            return Flow::Next;
        }
        if let Some((seq, by_cpu, epoch)) = promised(a, wide, func, mask, reference) {
            if let Some(e) = self.exec.as_mut() {
                e.depend_on(seq, by_cpu);
                PROMISED.fetch_add(1, Ordering::Relaxed);
                self.last_wait_kind = b'P';
                self.last_wait_epoch = epoch;
                label_event(&self.name, b'P', a, v0, reference, at);
                self.last_wait_gpu = true;
                return Flow::Next;
            }
        }
        // A matching write made after this command buffer was submitted:
        // hardware queues reach their waits at once and would have seen it.
        // Ours fall behind at times (pipeline compilation) and then waited
        // for a value already gone, e.g. a frame counter compared for
        // equality (run 20261011-062446: acb0x48 waited for 0x7d8 at 0x7dc,
        // and every queue stopped).
        if self.submitted_at.is_some_and(|since| missed_write(a, wide, func, mask, reference, since)) {
            MISSED.fetch_add(1, Ordering::Relaxed);
            self.last_wait_kind = b'M';
            label_event(&self.name, b'M', a, v0, reference, at);
            if let Some(e) = self.exec.as_mut() {
                e.xq_label(a, wide, v0);
            }
            return Flow::Next;
        }
        // A value this queue's recording batch writes on the GPU: its later
        // work follows that write in the batch (`tune gpu_cp_writes`).
        if let Some((v, bytes)) = self.exec.as_ref().and_then(|e| e.open_promise(a)) {
            let v = if wide { v } else { v & 0xFFFF_FFFF };
            if !(wide && bytes < 8) && compare(func, v & mask, reference & mask) {
                OPEN_PROMISED.fetch_add(1, Ordering::Relaxed);
                // Later work follows that write (end-of-pipe writes do not
                // order it by themselves: light end-of-pipe writes, `Exec::gpu_cp_write`).
                if let Some(e) = self.exec.as_mut() {
                    e.order_after_open_writes();
                }
                self.last_wait_kind = b'P';
                self.last_wait_epoch = 0;
                label_event(&self.name, b'P', a, v0, reference, at);
                self.last_wait_gpu = true;
                return Flow::Next;
            }
        }
        // An equality wait on a counter that has moved past the value (its
        // command buffer came after the write: run 20261011-082829, acb0x48
        // waited for 0x706 at 0x70a while pipelines compiled).
        if counter_passed(a, wide, func, mask, reference) {
            if COUNTER_PASSED.fetch_add(1, Ordering::Relaxed) < 20 {
                crate::info!("gpu[{}]: WAIT_REG_MEM at {at:#x} for [{a:#x}] == {reference:#x} passed: the counter is at {v0:#x}", self.name);
            }
            // (b'K': passed because its counter moved on; treated as b'M' and b'E' are, named
            // apart in the wait reports since 2026-11-24.)
            self.last_wait_kind = b'K';
            label_event(&self.name, b'K', a, v0, reference, at);
            if let Some(e) = self.exec.as_mut() {
                e.xq_label(a, wide, v0);
            }
            return Flow::Next;
        }
        // In a replayed frame, a wait the title's CPU released when the frame
        // ran passes (`gpu::replay`).
        if super::replay::pass_wait(at, a) {
            self.last_wait_kind = b'M';
            label_event(&self.name, b'M', a, v0, reference, at);
            return Flow::Next;
        }
        // Measurement only (`tune xwait_skip`, unsafe): the graphics CP does
        // not wait for labels compute queue 0x50 wrote last (an upper bound
        // for recording past them; what follows may read stale data). Before
        // the wait is registered (`ACTIVE_WAITS`): until 2026-10-24 it
        // returned after that, and the never-removed entries made every
        // later lookup slower, so its measurements understated the bound.
        if super::tune::xwait_skip() && self.name == "gpu-gfx" && last_write(a).is_some_and(|(q, _, _)| q == "gpu-acb0x50") {
            self.note("`tune xwait_skip`: label waits for gpu-acb0x50 skipped (measurement; pictures may be wrong)");
            return Flow::Next;
        }
        // Measurement only (`tune xwait_skip 2`, unsafe): queue 0x50 does not
        // wait for labels other queues' GPU work wrote last (an upper bound
        // for its recording past them).
        if super::tune::xwait_skip_acb() && self.name == "gpu-acb0x50" && last_write(a).is_some_and(|(q, _, _)| q.starts_with("gpu-") && q != self.name) {
            self.note("`tune xwait_skip 2`: gpu-acb0x50's waits for other queues' labels skipped (measurement; pictures may be wrong)");
            return Flow::Next;
        }
        // Recording goes on past it, held (`Cp::past_wait_ok`).
        if self.past_wait_ok(a) {
            self.last_wait_kind = b'G';
            return self.past_wait_start(control, a, wide, reference, mask, at);
        }
        if let Some(why) = self.past_wait_not(a) {
            note_why(&PAST_WAIT_NOT, why);
        }
        label_event(&self.name, b'S', a, v0, reference, at);
        // Let recorded work and deferred effects progress while polling; a
        // queue holding this label in an unsubmitted batch submits it.
        self.submit_gpu("wait");
        super::exec::request_label_submit(a);
        let what = format!("WAIT_REG_MEM at {at:#x}: [{a:#x}] & {mask:#x} func {func} ref {reference:#x}");
        self.waiting_on = a;
        // (A label our command processors poll is never held.)
        super::fe2::c4_cp_target(a, 8);
        // Submitters stop waiting for this queue only when the label is not
        // written by its own outstanding GPU work (it may come from them).
        let own = self.exec.as_mut().is_some_and(|e| e.will_write(a, if wide { 8 } else { 4 }));
        if let (Some(f), false) = (self.wait_flag, own) {
            f.store(true, Ordering::SeqCst);
        }
        let hit = std::sync::Arc::new(std::sync::atomic::AtomicBool::new(false));
        ACTIVE_WAITS.lock().push(ActiveWait { addr: a, wide, func, mask, reference, hit: hit.clone() });
        ACTIVE_COUNT.fetch_add(1, Ordering::AcqRel);
        let t0 = Instant::now();
        let tw = super::timeline::wait_start();
        let use_promise = self.exec.is_some();
        let promises_seen = std::cell::Cell::new(u64::MAX);
        let promises_moved = || {
            let g = PROMISE_GEN.load(Ordering::Acquire);
            promises_seen.replace(g) != g
        };
        // The stages of the commands after this wait are prepared meanwhile
        // (`tune wait_ahead`; checked when used as any made ahead), those
        // of an earlier wait's scan first.
        let mut pending: Vec<Arc<super::ahead::Slot>> = Vec::new();
        if let Some((f, from, end)) = self.wait_ctx.take() {
            if self.ahead.is_empty() {
                let cmds = self.scan_through(&f, from, end);
                self.ahead.extend(cmds);
            }
            pending = self.ahead.iter().flat_map(|c| c.stages.iter().flatten().cloned()).collect();
        }
        let mut next_slot = 0usize;
        let compile_ahead = super::compile::at_waits();
        let mut work = || {
            while next_slot < pending.len() {
                next_slot += 1;
                let slot = &pending[next_slot - 1];
                if super::ahead::prepare_here(slot) {
                    // A compute stage whose pipeline is missing: its compile
                    // starts now, on a compile worker (`tune compile_ahead`).
                    if compile_ahead {
                        if let Some(p) = super::ahead::prepared_compute(slot) {
                            if !super::exec::compute_pipe_known(&p) {
                                super::compile::submit_prepared(p);
                            }
                        }
                    }
                    return true;
                }
            }
            false
        };
        // A label a GPU queue wrote last: its next value comes with a
        // promise, a latched CP write or a GPU-performed write seen
        // (`wake_label`).
        let gpu_label = last_write(a).is_some_and(|(q, _, _)| q.starts_with("gpu-"));
        let probe = super::xwait::on() && self.name == "gpu-gfx";
        if probe {
            if let Some((q, _, _)) = last_write(a) {
                super::xwait::wait_start(&q);
            }
        }
        // An equality wait whose counter moves past its value while this queue waits passes too
        // (`counter_passed`, checked again whenever the label write history gains a value,
        // `LABEL_WRITES_GEN`; the first poll checks once, for history added since the check
        // above): checked only where the wait began until 2026-11-24, a queue that reached it one
        // step late, or whose preparing ahead here (`work`) kept it from polling while the counter
        // went by, waited forever, and every queue with it (the shader-prep branch's cold new
        // game, 2026-11-23: compute queue 0x48 waited for 0xa20 with the counter at 0xa84).
        let counter_gen = std::cell::Cell::new(u64::MAX);
        let counter_moved = std::cell::Cell::new(false);
        let r = self.poll_until_wake(
            at,
            &what,
            || {
                hit.load(Ordering::SeqCst)
                    || compare(func, read() & mask, reference & mask)
                    || (use_promise && promises_moved() && promised(a, wide, func, mask, reference).is_some())
                    || (func & 7 == 3 && {
                        let g = LABEL_WRITES_GEN.load(Ordering::Acquire);
                        counter_gen.replace(g) != g && counter_passed(a, wide, func, mask, reference) && {
                            counter_moved.set(true);
                            true
                        }
                    })
            },
            &mut work,
            (gpu_label && a % if wide { 8 } else { 4 } == 0).then_some((a, wide)),
        );
        if probe {
            super::xwait::wait_end();
        }
        {
            let by = last_write(a).map(|w| w.0).unwrap_or_else(|| "cpu".into());
            tw.end(&self.name, || format!("wait: label {a:#x} (written by {by})"));
            let mut g = WAIT_STATS.lock();
            let e = g.get_or_insert_with(HashMap::new).entry((self.name.clone(), by)).or_default();
            e.0 += 1;
            e.1 += t0.elapsed().as_nanos() as u64;
        }
        ACTIVE_WAITS.lock().retain(|w| !std::sync::Arc::ptr_eq(&w.hit, &hit));
        ACTIVE_COUNT.fetch_sub(1, Ordering::AcqRel);
        if let Some(f) = self.wait_flag {
            f.store(false, Ordering::SeqCst);
        }
        let in_memory = compare(func, read() & mask, reference & mask);
        let latched = hit.load(Ordering::SeqCst) && !in_memory;
        if latched {
            LATCHED.fetch_add(1, Ordering::Relaxed);
        }
        // Passed on a promise (its batch may have completed meanwhile).
        let mut kind = if latched { b'L' } else { b'E' };
        if !in_memory && !latched {
            // (A promise first, as the poll checks them: its dependency orders this queue after
            // the write; then the counter.)
            if let (Some((seq, by_cpu, epoch)), Some(e)) = (promised(a, wide, func, mask, reference), self.exec.as_mut()) {
                e.depend_on(seq, by_cpu);
                PROMISED.fetch_add(1, Ordering::Relaxed);
                kind = b'P';
                self.last_wait_epoch = epoch;
            } else if counter_moved.get() {
                if COUNTER_PASSED.fetch_add(1, Ordering::Relaxed) < 20 {
                    crate::info!("gpu[{}]: WAIT_REG_MEM at {at:#x} for [{a:#x}] == {reference:#x} passed while waiting: the counter is at {:#x}", self.name, read());
                }
                kind = b'K';
            }
        }
        self.last_wait_kind = kind;
        label_event(&self.name, kind, a, read(), reference, at);
        if kind != b'P' {
            if let Some(e) = self.exec.as_mut() {
                e.xq_label(a, wide, read());
            }
        }
        self.last_wait_gpu = kind == b'P' || kind == b'L';
        long_wait(&self.name, a, wide, func, mask, reference, at, t0, kind);
        match r {
            Flow::Stop => Flow::Stop,
            _ => Flow::Next,
        }
    }

    /// Spins (then sleeps) until `cond`; reports long waits once.
    fn poll_until(&mut self, at: u64, what: &str, cond: impl Fn() -> bool) -> Flow {
        self.poll_until_wake(at, what, cond, &mut || false, None)
    }

    /// `poll_until`, calling `work` between the checks while it finds
    /// something to do (it returns whether it did).
    /// `wake` (a label's address and width): after its spins the wait sleeps
    /// on the label's address (`wake_label`; at most 1 ms at a time) instead
    /// of yielding and sleeping 100 us between checks.
    fn poll_until_wake(&mut self, at: u64, what: &str, cond: impl Fn() -> bool, work: &mut dyn FnMut() -> bool, wake: Option<(u64, bool)>) -> Flow {
        let start = Instant::now();
        let mut reported = false;
        set_state(&self.name, what.to_string());
        let mut spins = 0u32;
        let yields = 200 + super::tune::wait_yields().min(100_000) as u32;
        while !cond() {
            if crate::gpu::shutting_down() {
                return Flow::Stop;
            }
            if super::replay::force_wait(at, start, what) {
                break;
            }
            if work() {
                continue;
            }
            spins += 1;
            if spins < 200 {
                std::hint::spin_loop();
            } else if let Some((a, wide)) = wake {
                // The value now: a write after this wakes the sleep too.
                let seen: u64 = unsafe { if wide { rd64(a) } else { rd32(a) as u64 } };
                if cond() {
                    break;
                }
                WAKE_SLEEPS.fetch_add(1, Ordering::Relaxed);
                unsafe { windows_sys::Win32::System::Threading::WaitOnAddress(a as *const core::ffi::c_void, &seen as *const u64 as *const core::ffi::c_void, if wide { 8 } else { 4 }, 1) };
            } else if spins < yields {
                std::thread::yield_now();
            } else {
                crate::thread::short_sleep(Duration::from_micros(100));
            }
            if !reported && start.elapsed() > Duration::from_secs(3) {
                reported = true;
                let marker = self.markers.last().cloned().unwrap_or_default();
                let a = self.waiting_on;
                let detail = if a != 0 {
                    let cur = unsafe { rd64(a) };
                    match last_write(a) {
                        Some((q, v, pk)) => format!("; value now {cur:#x}, last GPU write {v:#x} by {q} (packet {pk:#x})"),
                        None => format!("; value now {cur:#x}, never written by the GPU"),
                    }
                } else {
                    String::new()
                };
                let mut history = if a != 0 { format!("; recorded writes {}; label history:{}", label_writes(a), label_history(a, 16)) } else { String::new() };
                // (Every queue's state, and this queue's batch: cycles of
                // label waits, FINDINGS.md 2026-10-30 morning and
                // 2026-11-01.)
                let states: Vec<String> = STATE.lock().iter().map(|(n, s)| format!("{n}: {s}")).collect();
                let own = self.exec.as_ref().map(|e| e.wait_report()).unwrap_or_default();
                history.push_str(&format!("; completed through {}; this queue: {own}; queues: {}", super::exec::completed_seq(), states.join(" | ")));
                crate::warn!("gpu[{}]: waiting > 3 s: {what}{}{detail}{history}", self.name, if marker.is_empty() { String::new() } else { format!(" (marker '{marker}')") });
            }
        }
        set_state(&self.name, "running".into());
        self.waiting_on = 0;
        let _ = at;
        Flow::Next
    }

    fn private(&mut self, operation: u32, at: u64, n: u64, a: &dyn Fn(u64) -> u32) -> Flow {
        match operation {
            // A replayed frame (`gpu::replay`) is not presented (its work
            // completes here as for a flip).
            pop::FLIP if super::replay::replaying() => self.sync(),
            pop::FLIP if super::tune::flip_async() => {
                if super::itemcensus::on() && self.is_main_gfx() {
                    super::itemcensus::flip();
                }
                label_event(&self.name, b'F', 0, self.sub_index, 0, at);
                // Queued now, shown once this queue's GPU work before it
                // completed (as the hardware's CP queues it) instead of the
                // command processor waiting for that work here.
                let handle = a(0) as i32;
                let index = a(1) as i32;
                let arg = addr64(a(3), a(4)) as i64;
                let len = 4 * (n + 3);
                let mut r = crate::hle::libs::videoout::reserve_flip(handle, index, arg);
                if r == Err(crate::hle::libs::videoout::ERR_FLIP_QUEUE_FULL) {
                    // (The open batch submitted first: the flips filling the queue commit once
                    // the batches holding them complete, and one may be this one: see below.)
                    if let Some(e) = self.exec.as_mut() {
                        e.submit_for("flip slot wait");
                    }
                    set_state(&self.name, format!("flip of buffer {index}: waiting for a free flip slot"));
                    while r == Err(crate::hle::libs::videoout::ERR_FLIP_QUEUE_FULL) && !crate::gpu::shutting_down() {
                        std::thread::sleep(Duration::from_micros(500));
                        r = crate::hle::libs::videoout::reserve_flip(handle, index, arg);
                    }
                    set_state(&self.name, "running".into());
                }
                match r {
                    Ok(ticket) => {
                        self.effect(true, None, Box::new(move || crate::hle::libs::videoout::commit_flip(ticket)));
                        // Reflex: the frame's GPU work is all recorded (`gpu::latency`; the flip's
                        // number is the frame's index).
                        if !self.compute {
                            crate::gpu::latency::marker(crate::gpu::latency::Marker::RenderSubmitEnd, crate::hle::libs::videoout::QUEUED_FLIP.load(Ordering::Relaxed) as u32);
                        }
                    }
                    Err(e) => crate::warn!("gpu[{}]: flip of buffer {index} (handle {handle:#x}) failed ({e:#x})", self.name),
                }
                let n = crate::gpu::FRAMES.fetch_add(1, Ordering::Relaxed) + 1;
                super::capture::on_flip(n);
                if !self.compute {
                    super::note_run_ahead();
                    if super::frameprobe::on() && self.name == "gpu-gfx" {
                        super::frameprobe::flip([&self.regs.ctx, &self.regs.sh, &self.regs.uc]);
                        super::frameprobe::flip_walk(self.walk_start(), at, len);
                    }
                    if self.name == "gpu-gfx" && super::runprobe::on() {
                        super::runprobe::flip();
                    }
                }
            }
            pop::FLIP => {
                self.sync();
                label_event(&self.name, b'F', 0, self.sub_index, 0, at);
                let handle = a(0) as i32;
                let index = a(1) as i32;
                let arg = addr64(a(3), a(4)) as i64;
                // A full flip queue stalls the queue (as on hardware), which
                // paces the title through its label waits.
                let mut r = crate::hle::libs::videoout::submit_flip(handle, index, arg);
                if r == crate::hle::libs::videoout::ERR_FLIP_QUEUE_FULL {
                    set_state(&self.name, format!("flip of buffer {index}: waiting for a free flip slot"));
                    while r == crate::hle::libs::videoout::ERR_FLIP_QUEUE_FULL && !crate::gpu::shutting_down() {
                        std::thread::sleep(Duration::from_micros(500));
                        r = crate::hle::libs::videoout::submit_flip(handle, index, arg);
                    }
                    set_state(&self.name, "running".into());
                }
                if r != 0 {
                    // The packet as executed and as it is in memory now (a
                    // command buffer reused before execution shows here).
                    let words: Vec<String> = (0..n.min(9)).map(|k| format!("{:08x}", a(k))).collect();
                    let now: Vec<String> = (0..n.min(9) + 3).map(|k| at + 4 * k).filter(|&w| guest_readable(w, 4)).map(|w| format!("{:08x}", unsafe { rd32(w) })).collect();
                    crate::warn!("gpu[{}]: flip of buffer {index} (handle {handle:#x}) failed ({r:#x}); packet at {at:#x}: args {} (memory now: {}), submitted {} ms ago", self.name, words.join(" "), now.join(" "), self.submitted_at.map_or(0, |t| t.elapsed().as_millis()));
                }
                let n = crate::gpu::FRAMES.fetch_add(1, Ordering::Relaxed) + 1;
                super::capture::on_flip(n);
            }
            pop::WAIT_SAFE_FOR_RENDERING => {
                let (handle, index) = (a(0) as i32, a(1) as i32);
                if crate::hle::libs::videoout::buffer_busy(handle, index) && !super::replay::replaying() {
                    // (The open batch submitted first. An asynchronous flip commits once the batch
                    // holding it completes (`effect`), and that batch can be this queue's own, still
                    // being recorded: the wait then waited for itself. A self-play part was found
                    // hung so on 2026-11-22 small hours: the GPU done through the last submission,
                    // this queue recording with the flip that queued the buffer in its batch.)
                    if let Some(e) = self.exec.as_mut() {
                        e.submit_for("display buffer wait");
                    }
                    let what = format!("WaitUntilSafeForRendering: display buffer {index} is displayed or queued");
                    return self.poll_until(at, &what, || !crate::hle::libs::videoout::buffer_busy(handle, index));
                }
            }
            pop::LOD_STATS => {
                // Hardware LOD statistics (texture streaming feedback): the
                // title reads, per counter bank (T# word 6), the 8-byte entry
                // at 0x40 + 8 * bank: bits 56..59 the finest mip sampled (15:
                // none), bits 32..55 a count (eboot+0xac90b0).
                let (buf, size, flags) = (addr64(a(0), a(1)), a(2) as u64, a(3));
                let mode = super::tune::lod_stats();
                if buf != 0 {
                    static SEEN: std::sync::atomic::AtomicU32 = std::sync::atomic::AtomicU32::new(0);
                    let n = SEEN.fetch_add(1, Ordering::Relaxed);
                    let writable = size >= super::lodstats::BUFFER_BYTES && guest_readable(buf, super::lodstats::BUFFER_BYTES);
                    if matches!(mode, 3 | 4) && writable && self.exec.is_some() {
                        // Copied after the work recorded so far, written to the
                        // title's buffer once that completed (before the label
                        // writes behind it).
                        let e = self.exec.as_mut().unwrap();
                        let slot = super::lodstats::take_slot();
                        let reset = (flags >> 19) & 1 != 0;
                        e.lod_stats_copy(slot, reset);
                        // (The rotated draws' banks so far: `tune lod_late`.)
                        let carry = if self.name == "gpu-gfx" { super::lodstats::end_report(reset) } else { None };
                        let name = self.name.clone();
                        let log = n < 2 || n.is_power_of_two() || n % 1000 == 0;
                        let f: Box<dyn FnOnce() + Send> = Box::new(move || {
                            let words = super::exec::lod_stats_words(slot);
                            let (sampled, by_mip) = unsafe { super::lodstats::write_measured(buf, &words, slot, carry.as_ref()) };
                            if log {
                                let mips: Vec<String> = by_mip.iter().enumerate().filter(|&(_, &c)| c > 0).map(|(m, c)| format!("{m}:{c}")).collect();
                                crate::info!("gpu[{name}]: GetLodStats #{n} (buffer {buf:#x}, flags {flags:#x}): {sampled} of 256 counter banks sampled (finest mip:banks {}; measured on the GPU)", mips.join(" "));
                            }
                        });
                        e.after_gpu(f, Some((buf, super::lodstats::BUFFER_BYTES)), false);
                    } else if mode != 0 && writable {
                        // Before the label write behind it (a deferred effect).
                        let sampled = unsafe { super::lodstats::report(buf, mode, (flags >> 19) & 1 != 0) };
                        if n < 2 || n.is_power_of_two() {
                            crate::info!("gpu[{}]: GetLodStats #{n} (buffer {buf:#x}, flags {flags:#x}): {sampled} of 256 counter banks reported as sampled at mip 0 (mode {mode}; stand-in for the hardware counters)", self.name);
                        }
                    } else if n < 2 || n.is_power_of_two() {
                        crate::warn!("gpu[{}]: GetLodStats #{n} (buffer {buf:#x} size {size:#x}, flags {flags:#x}) not written (mode {mode}): the title streams no high mips", self.name);
                    }
                }
            }
            pop::WORKLOADS_ACTIVE | pop::WORKLOAD_COMPLETE => {}
            pop::CONTEXT_STATE => match a(0) {
                0 => self.clear_context(),
                1 => self.ctx_stack.push(self.regs.ctx.clone()),
                2 => match self.ctx_stack.pop() {
                    Some(c) => {
                        self.regs.ctx = c;
                        self.ctx_dirty = true;
                    }
                    None => crate::warn!("gpu[{}]: context state pop without push at {at:#x}", self.name),
                },
                3 => {
                    self.ctx_stack.push(self.regs.ctx.clone());
                    self.clear_context();
                }
                other => crate::warn!("gpu[{}]: unknown context state operation {other}", self.name),
            },
            pop::ACB_RESET => {
                self.regs.sh.fill(0);
            }
            pop::PUSH_MARKER | pop::SET_MARKER => {
                let mut bytes = Vec::new();
                for i in 1..n {
                    bytes.extend_from_slice(&a(i).to_le_bytes());
                }
                let s = String::from_utf8_lossy(&bytes[..bytes.iter().position(|&b| b == 0).unwrap_or(bytes.len())]).into_owned();
                if operation == pop::SET_MARKER {
                    self.markers.pop();
                }
                self.markers.push(s);
                if self.markers.len() > 64 {
                    self.markers.remove(0);
                }
            }
            pop::POP_MARKER => {
                self.markers.pop();
            }
            pop::BRANCH => {
                self.sync();
                let ctl = a(0);
                let cmp = addr64(a(1), a(2));
                self.sync_reads(&[(cmp, 8)]);
                let mask = addr64(a(3), a(4));
                let reference = addr64(a(5), a(6));
                let pass = valid(cmp, 8) && compare((ctl >> 8) & 7, unsafe { rd64(cmp) } & mask, reference & mask);
                let (lo, hi, s) = if pass { (a(7), a(8), a(9)) } else { (a(10), a(11), a(12)) };
                let target = addr64(lo, hi);
                let size = s & 0xF_FFFF;
                if target != 0 && size != 0 {
                    self.run(target, size);
                }
            }
            pop::DRAW_INDEX_MULTI_INSTANCED => {
                DRAWS.fetch_add(1, Ordering::Relaxed);
                self.note("multi-instanced draws are not rendered yet");
            }
            _ => {
                crate::warn!("gpu[{}]: unknown private packet op {operation} at {at:#x}", self.name);
            }
        }
        Flow::Next
    }
}

enum Flow {
    Next,
    Skip(u32),
    Chain(u64, u32),
    Stop,
}

#[cfg(test)]
mod tests {
    use super::*;

    fn history(addr: u64, values: &[u64]) {
        let mut g = LABEL_WRITES.lock();
        let h = g.get_or_insert_with(HashMap::new).entry(addr).or_default();
        h.clear();
        h.extend(values.iter().map(|&v| (Instant::now(), v)));
    }

    /// Equality waits on frame counters that moved on pass; waits on flags,
    /// on values not yet reached and on counters that jump do not.
    #[test]
    fn counter_passed_rule() {
        let m = 0xFFFF_FFFF;
        history(0x1000, &[0x7fe, 0x7ff, 0x800, 0x801, 0x802, 0x803, 0x804, 0x805]);
        assert!(counter_passed(0x1000, false, 3, m, 0x801));
        assert!(!counter_passed(0x1000, false, 3, m, 0x804), "one step behind may still come");
        assert!(!counter_passed(0x1000, false, 3, m, 0x806), "not reached yet");
        assert!(!counter_passed(0x1000, false, 2, m, 0x801), "only equality waits");
        // Counting by 2 (run 20261011-105452: 0x82c 0x82e 0x830 0x832).
        history(0x1008, &[0x82c, 0x82e, 0x830, 0x832, 0x834]);
        assert!(counter_passed(0x1008, false, 3, m, 0x82e));
        assert!(!counter_passed(0x1008, false, 3, m, 0x832));
        // The waited value before the recorded window.
        history(0x1010, &[0x802, 0x803, 0x803, 0x804, 0x805]);
        assert!(counter_passed(0x1010, false, 3, m, 0x801));
        assert!(!counter_passed(0x1010, false, 3, m, 0x700), "too far behind the window");
        // A flag (0 -> 1 -> 0) is not a counter.
        history(0x1018, &[0, 1, 0, 1]);
        assert!(!counter_passed(0x1018, false, 3, m, 0));
        // A counter that was reset is not trusted, while the reset is in the window (for the
        // values after it too).
        history(0x1020, &[0x10, 0x11, 0x0, 0x1, 0x2, 0x3]);
        assert!(!counter_passed(0x1020, false, 3, m, 0x11));
        assert!(!counter_passed(0x1020, false, 3, m, 0x1));
        // A ring (values counted modulo 4) is no counter: a queue waiting at 2 for its next 1 is
        // not passed at 3 (the review of 2026-11-24).
        history(0x1030, &[0, 1, 2, 3, 0, 1, 2, 3]);
        assert!(!counter_passed(0x1030, false, 3, m, 1));
        // A counter reset (at a load, say) is trusted again once it has counted 8 steps.
        history(0x1038, &[0x10, 0x11, 0x0, 0x1, 0x2, 0x3, 0x4, 0x5, 0x6, 0x7, 0x8, 0x9]);
        assert!(counter_passed(0x1038, false, 3, m, 0x2));
        assert!(!counter_passed(0x1038, false, 3, m, 0x11), "before the reset");
        // A queue that reaches its wait one step late is not passed then, but once the counter
        // moves on again, which the wait's polling now checks (2026-11-24: acb0x48 waited for
        // 0xa20 while the counter went on to 0xa84, and every queue stopped).
        history(0x1028, &[0xa1c, 0xa1e, 0xa20, 0xa22]);
        assert!(!counter_passed(0x1028, false, 3, m, 0xa20), "one step on: may still be a flag");
        history(0x1028, &[0xa1c, 0xa1e, 0xa20, 0xa22, 0xa24]);
        assert!(counter_passed(0x1028, false, 3, m, 0xa20));
    }

    #[test]
    fn register_snapshots_hold_the_written_part() {
        // A snapshot's registers, then zeros up to the current high mark
        // (written after the snapshot), the rest untouched (never written).
        let mut dst = vec![7u32; 8];
        copy_prefix(&mut dst, &[1, 2, 3], 6);
        assert_eq!(dst, [1, 2, 3, 0, 0, 0, 7, 7]);
        // Writes to a snapshot grow it and raise the mark; beyond the file
        // they are ignored.
        let mut v = vec![1u32, 2];
        put_reg(&mut v, Space::Sh, 4, 3, 9);
        assert_eq!(v, [1, 2, 0, 9]);
        assert!(high(Space::Sh, SH_DWORDS) >= 4);
        put_reg(&mut v, Space::Sh, 4, 4, 5);
        assert_eq!(v.len(), 4);
    }

    #[test]
    fn register_ranges_copy_like_single_writes() {
        // A SET_*_REG payload at 0x1000 (fetched): registers 3.. get 10, 11,
        // 12; a range past the file's end stops at it.
        let f = Fetched { base: 0x1000, words: vec![10, 11, 12, 13, 14], live: false };
        let mut a = vec![7u32; 2];
        put_regs(&mut a, Space::Ctx, 64, 3, &f, 0x1000, 3);
        let mut b = vec![7u32; 2];
        for j in 0..3 {
            put_reg(&mut b, Space::Ctx, 64, 3 + j, 10 + j as u32);
        }
        assert_eq!(a, b);
        assert_eq!(a, [7, 7, 0, 10, 11, 12]);
        let mut c = Vec::new();
        put_regs(&mut c, Space::Ctx, 5, 3, &f, 0x1004, 4);
        assert_eq!(c, [0, 0, 0, 11, 12]);
        assert_eq!(f.slice(0x1008, 3), Some(&[12u32, 13, 14][..]));
        assert_eq!(f.slice(0x1010, 2), None, "straddles the fetched words");
    }

    #[test]
    fn chunks_split_at_equal_cost() {
        // Commands 0..4 are the main command processor's; 4..16 go to three
        // workers. The last four cost three times the others.
        let mut c = vec![1.0f32; 16];
        for x in c.iter_mut().skip(12) {
            *x = 3.0;
        }
        // Total 8 + 12 = 20 over three chunks: cuts where ~6.7 and ~13.3
        // are reached.
        assert_eq!(cost_firsts(&c, 4, 16, 3, 2), Some(vec![4, 11, 14]));
        // Equal costs: equal counts.
        assert_eq!(cost_firsts(&[1.0; 16], 4, 16, 3, 2), Some(vec![4, 8, 12]));
        // Chunks keep their minimum size.
        let mut d = vec![1.0f32; 16];
        d[4] = 100.0;
        assert_eq!(cost_firsts(&d, 4, 16, 3, 3), Some(vec![4, 7, 10]));
        // Unmeasured commands: no split by cost.
        let mut e = vec![1.0f32; 16];
        e[9] = f32::NAN;
        assert_eq!(cost_firsts(&e, 4, 16, 3, 2), None);
        // Main's part may stay unmeasured.
        let mut g = vec![f32::NAN; 16];
        for x in g.iter_mut().skip(4) {
            *x = 1.0;
        }
        assert_eq!(cost_firsts(&g, 4, 16, 3, 2), Some(vec![4, 8, 12]));
    }
}
