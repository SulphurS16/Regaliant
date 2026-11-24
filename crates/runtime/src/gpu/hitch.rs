//! Hitches: presented frames far longer than the frames before them, logged
//! with the long pieces of work that overlapped them (pipeline creation,
//! shader translation, texture uploads, CPU waits for the GPU, long command
//! processor waits, file I/O, single long recordings), so a frame-time trace
//! says why a frame was long and not only that it was: the frame-rate goal is
//! judged by the worst frames (STATUS.md). Every run's log gets `hitch:`
//! lines; the window statistics (`frametimes`, the scripts' `dsport-frames:`
//! marks) count the hitches and their excess time by cause.
//!
//! Every batch's span on the GPU comes in too (`gpu_batch`, from its
//! timestamps once it completed): each frame's GPU busy time is taken once
//! its batches are in (a few flips late), the windows report it per frame,
//! and a hitch gets a second line, `hitch GPU:`, with the GPU's busy time in
//! that frame against the median frame's, the kinds of batches that took
//! longest above their usual time per frame and its longest batches (whether
//! the GPU had more work then, and which).

use parking_lot::Mutex;
use std::collections::VecDeque;
use std::sync::atomic::{AtomicU64, Ordering};

/// The GPU's busy share of each frame (busy time over the frame's interval), averaged over
/// ~16 frames, x 1e6 (`gpu_busy_share`; 1.0 until frames come in).
static BUSY_SHARE: AtomicU64 = AtomicU64::new(1_000_000);

/// The GPU's busy share of the recent frames (0..1): low where the command processors (or the
/// title) leave it waiting (`par::graphics_min`).
pub fn gpu_busy_share() -> f64 {
    BUSY_SHARE.load(Ordering::Relaxed) as f64 / 1e6
}

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum Kind {
    /// Pipeline creation (the driver's compile).
    Pipeline = 0,
    /// A shader's translation to SPIR-V.
    Translate,
    /// Texture uploads (detiling on the CPU, staging copies).
    Upload,
    /// A CPU thread waiting for GPU work to complete.
    GpuWait,
    /// The graphics command processor waiting for a label (the title's CPU
    /// or another queue); the compute queues' waits for the graphics queue
    /// are the frame's structure, not a cause, and not noted.
    CpWait,
    /// File reads and writes of the title's threads.
    File,
    /// One command's recording, beyond the parts above.
    Record,
    /// Keeping memory coherent: front end 2's placement checks (takes and
    /// uploads) and armings (`gpu::fe2`), a file read waiting for an
    /// arming, a mapping change waiting for the guest memory's readers.
    Mirror,
}

const KINDS: usize = 8;
const NAMES: [&str; KINDS + 1] = ["pipeline", "translation", "upload", "GPU wait", "CP wait", "file", "recording", "memory", "unexplained"];

/// Work shorter than this is not kept (us).
const MIN_US: u64 = 2_000;
/// Recent work kept for the next reports.
const KEEP: usize = 1024;
/// Batches' GPU spans kept (a few frames' worth).
const GPU_KEEP: usize = 16384;
/// Batches at least this long on the GPU (us) keep what they held.
const GPU_WHAT_US: u64 = 1_000;
/// Frames waiting for their batches' GPU spans; beyond this many the oldest
/// is taken as it is (its last batches may be missing: marked).
const PENDING_MAX: usize = 6;

/// A batch's execution on the GPU: [start, end) in us on `now_us`'s clock
/// (its prologue's copies included), its queue (`State::queues`), sequence
/// and commands, and what it held when long.
struct GpuSpan {
    start: u64,
    end: u64,
    queue: u8,
    seq: u64,
    cmds: u32,
    /// Its most frequent command (`exec::batch_key`).
    key: u64,
    what: Option<String>,
}

/// The weight of a frame in the per-kind running means of GPU time per
/// frame (`State::kind_mean`).
const KIND_ALPHA: f64 = 1.0 / 32.0;

/// A presented frame whose GPU time is taken once its batches are in.
struct Pending {
    n: u64,
    start: u64,
    end: u64,
    median: u64,
    hitch: bool,
}

struct Event {
    start: u64,
    end: u64,
    kind: Kind,
    thread: String,
    what: String,
}

struct State {
    events: VecDeque<Event>,
    /// The last presented intervals (us), for the median.
    recent: VecDeque<u32>,
    last_flip: u64,
    /// The window's hitches, their excess over the median (us) and that
    /// excess by the cause that overlapped it most.
    hitches: u64,
    excess: u64,
    by_kind: [u64; KINDS + 1],
    worst: Vec<(u64, u64)>,
    /// Batches' GPU spans, oldest first, and the latest end among them.
    gpu: VecDeque<GpuSpan>,
    gpu_upto: u64,
    queues: Vec<String>,
    /// Frames waiting for their GPU spans, oldest first.
    pending: VecDeque<Pending>,
    /// The GPU busy time of the last frames taken (us), for the median, and
    /// of the window's frames.
    gpu_recent: VecDeque<u32>,
    gpu_window: Vec<u32>,
    /// The window's GPU time per queue and per (queue, batch kind), summed
    /// over its frames (us; spans clipped to their frame).
    window_queues: Option<std::collections::HashMap<u8, u64>>,
    window_kinds: Option<std::collections::HashMap<(u8, u64), u64>>,
    /// Per queue and batch kind (`GpuSpan::key`), the running mean of its
    /// GPU time per frame (us).
    kind_mean: Option<std::collections::HashMap<(u8, u64), f64>>,
}

static STATE: Mutex<State> = Mutex::new(State {
    events: VecDeque::new(),
    recent: VecDeque::new(),
    last_flip: 0,
    hitches: 0,
    excess: 0,
    by_kind: [0; KINDS + 1],
    worst: Vec::new(),
    gpu: VecDeque::new(),
    gpu_upto: 0,
    queues: Vec::new(),
    pending: VecDeque::new(),
    gpu_recent: VecDeque::new(),
    gpu_window: Vec::new(),
    window_queues: None,
    window_kinds: None,
    kind_mean: None,
});

/// GPU timestamps on `now_us`'s clock: when the pair was taken, a GPU
/// timestamp and `now_us` at it, ns per tick; from a calibrated pair
/// (VK_EXT_calibrated_timestamps) renewed every second.
static GPU_CAL: Mutex<Option<(std::time::Instant, u64, u64, f64)>> = Mutex::new(None);

/// (GPU ticks, us, ns per tick) at a recent calibration (None without
/// calibrated timestamps).
fn gpu_clock() -> Option<(u64, u64, f64)> {
    let mut g = GPU_CAL.lock();
    if let Some((at, t, us, p)) = *g {
        if at.elapsed() < std::time::Duration::from_secs(1) {
            return Some((t, us, p));
        }
    }
    let c = super::vk::try_ctx()?;
    let (t, q) = c.calibrate()?;
    let (mut q_now, mut f) = (0i64, 0i64);
    unsafe {
        crate::platform::os::Win32::System::Performance::QueryPerformanceCounter(&mut q_now);
        crate::platform::os::Win32::System::Performance::QueryPerformanceFrequency(&mut f);
    }
    let us_now = now_us();
    let us = us_now.saturating_sub(((q_now - q).max(0) as u128 * 1_000_000 / f.max(1) as u128) as u64);
    let p = c.limits.timestamp_period as f64;
    *g = Some((std::time::Instant::now(), t, us, p));
    Some((t, us, p))
}

/// A completed batch of `queue` ran on the GPU from timestamp `start` to
/// `end` (its prologue's start, its own end) with `cmds` commands; `what`
/// (its contents) is called only for long ones.
pub fn gpu_batch(queue: &str, seq: u64, start: u64, end: u64, cmds: u32, key: u64, what: impl FnOnce() -> String) {
    if end <= start {
        return;
    }
    let Some((t0, us0, p)) = gpu_clock() else { return };
    let conv = |t: u64| (us0 as i64 + ((t as i64).wrapping_sub(t0 as i64) as f64 * p / 1000.0) as i64).max(0) as u64;
    let (a, b) = (conv(start), conv(end));
    let what = (b - a >= GPU_WHAT_US).then(what);
    let mut s = STATE.lock();
    let queue = match s.queues.iter().position(|q| q == queue) {
        Some(i) => i as u8,
        None if s.queues.len() < 255 => {
            s.queues.push(queue.to_string());
            (s.queues.len() - 1) as u8
        }
        None => 254,
    };
    if s.gpu.len() >= GPU_KEEP {
        s.gpu.pop_front();
    }
    s.gpu.push_back(GpuSpan { start: a, end: b, queue, seq, cmds, key, what });
    s.gpu_upto = s.gpu_upto.max(b);
}

/// The GPU's busy time in [a, b) (spans on several queues counted once).
fn gpu_busy(s: &State, a: u64, b: u64) -> u64 {
    let mut v: Vec<(u64, u64)> = Vec::new();
    for g in s.gpu.iter().rev() {
        if g.end.saturating_add(10_000_000) < a {
            break;
        }
        if g.end > a && g.start < b {
            v.push((g.start.max(a), g.end.min(b)));
        }
    }
    v.sort_unstable();
    let (mut sum, mut end) = (0u64, 0u64);
    for (x, y) in v {
        let x = x.max(end);
        if y > x {
            sum += y - x;
            end = y;
        }
    }
    sum
}

fn median_of(v: impl Iterator<Item = u32>) -> u64 {
    let mut v: Vec<u32> = v.collect();
    v.sort_unstable();
    v.get(v.len() / 2).copied().unwrap_or(0) as u64
}

/// Takes the frames whose batches are all in (or that waited too long):
/// their GPU busy time into the statistics, a `hitch GPU:` line for hitches.
fn take_pending(s: &mut State) {
    while let Some(p) = s.pending.front() {
        let complete = s.gpu_upto >= p.end;
        if !complete && s.pending.len() <= PENDING_MAX {
            break;
        }
        let p = s.pending.pop_front().unwrap();
        let busy = gpu_busy(s, p.start, p.end);
        let span = p.end.saturating_sub(p.start);
        if span > 0 {
            let x = (busy as f64 / span as f64).min(1.0);
            let old = BUSY_SHARE.load(Ordering::Relaxed) as f64 / 1e6;
            BUSY_SHARE.store(((old + (x - old) / 16.0) * 1e6) as u64, Ordering::Relaxed);
        }
        // GPU time per queue and batch kind in the frame (clipped to it).
        let mut kinds: std::collections::HashMap<(u8, u64), u64> = std::collections::HashMap::new();
        for g in s.gpu.iter().rev() {
            if g.end.saturating_add(10_000_000) < p.start {
                break;
            }
            if g.end > p.start && g.start < p.end {
                *kinds.entry((g.queue, g.key)).or_default() += g.end.min(p.end) - g.start.max(p.start);
            }
        }
        // The kinds that took longest above their running mean (before this
        // frame joins it).
        let grew: Vec<String> = if p.hitch {
            let means = s.kind_mean.as_ref();
            let mut d: Vec<((u8, u64), u64, f64)> = kinds.iter().map(|(&k, &t)| (k, t, means.and_then(|m| m.get(&k)).copied().unwrap_or(0.0))).filter(|x| x.1 as f64 > x.2 + 500.0).collect();
            d.sort_by(|x, y| (y.1 as f64 - y.2).total_cmp(&(x.1 as f64 - x.2)));
            d.iter()
                .take(4)
                .map(|&((q, k), t, m)| {
                    let qn = s.queues.get(q as usize).map_or("?", |x| x.as_str());
                    let kn = if qn == "uploads" { "texture uploads".to_string() } else { super::exec::batch_key_name(k) };
                    format!("{qn} {kn} +{:.1} ms ({:.1}, usually {:.1})", (t as f64 - m) / 1e3, t as f64 / 1e3, m / 1e3)
                })
                .collect()
        } else {
            Vec::new()
        };
        if s.gpu_window.len() < 1 << 20 {
            let wq = s.window_queues.get_or_insert_with(std::collections::HashMap::new);
            for (&(q, _), &t) in &kinds {
                *wq.entry(q).or_default() += t;
            }
            let wk = s.window_kinds.get_or_insert_with(std::collections::HashMap::new);
            for (&k, &t) in &kinds {
                *wk.entry(k).or_default() += t;
            }
        }
        let means = s.kind_mean.get_or_insert_with(std::collections::HashMap::new);
        for (k, m) in means.iter_mut() {
            *m *= 1.0 - KIND_ALPHA;
            if let Some(&t) = kinds.get(k) {
                *m += KIND_ALPHA * t as f64;
            }
        }
        for (k, t) in kinds {
            means.entry(k).or_insert(KIND_ALPHA * t as f64);
        }
        if p.hitch {
            let typical = median_of(s.gpu_recent.iter().copied());
            // The longest batches overlapping the frame and the median before it.
            let lo = p.start.saturating_sub(p.median);
            let mut long: Vec<&GpuSpan> = s.gpu.iter().rev().take_while(|g| g.end.saturating_add(10_000_000) >= lo).filter(|g| g.end > lo && g.start < p.end).collect();
            long.sort_by(|x, y| (y.end - y.start).cmp(&(x.end - x.start)));
            let parts: Vec<String> = long
                .iter()
                .take(4)
                .map(|g| format!("{} #{} {:.1} ms ({} commands{})", s.queues.get(g.queue as usize).map_or("?", |q| q.as_str()), g.seq, (g.end - g.start) as f64 / 1e3, g.cmds, g.what.as_deref().unwrap_or("")))
                .collect();
            crate::info!(
                "hitch GPU: flip {} took {:.0} ms: GPU busy {:.1} ms of it{} (median frame {:.0} ms, GPU busy {:.1}); above their usual GPU time per frame: {}; longest batches: {}",
                p.n,
                (p.end - p.start) as f64 / 1e3,
                busy as f64 / 1e3,
                if complete { "" } else { ", batches missing" },
                p.median as f64 / 1e3,
                typical as f64 / 1e3,
                if grew.is_empty() { "none".to_string() } else { grew.join("; ") },
                if parts.is_empty() { "none".to_string() } else { parts.join("; ") }
            );
        }
        let v = busy.min(u32::MAX as u64) as u32;
        if s.gpu_recent.len() >= 60 {
            s.gpu_recent.pop_front();
        }
        s.gpu_recent.push_back(v);
        if s.gpu_window.len() < 1 << 20 {
            s.gpu_window.push(v);
        }
    }
}

/// The window's GPU busy time per frame (a clause of the frame windows'
/// line, "" without GPU spans), and a new window.
pub fn gpu_window_clause() -> String {
    let mut s = STATE.lock();
    let mut v = std::mem::take(&mut s.gpu_window);
    if v.is_empty() || s.gpu_upto == 0 {
        return String::new();
    }
    v.sort_unstable();
    let n = v.len();
    let pct = |q: f64| v[((q / 100.0 * n as f64).ceil() as usize).clamp(1, n) - 1] as f64 / 1e3;
    // Mean GPU time per frame by queue, and the batch kinds that took most
    // (spans of the copy and upload queues overlap the others').
    let per = |t: u64| t as f64 / n as f64 / 1e3;
    let wq = s.window_queues.take().unwrap_or_default();
    let wk = s.window_kinds.take().unwrap_or_default();
    let names = s.queues.clone();
    let qname = |i: u8| names.get(i as usize).map_or("?".to_string(), |x| x.clone());
    // The parallel workers' chunk batches as one.
    let mut by_name: std::collections::HashMap<String, u64> = std::collections::HashMap::new();
    for (i, t) in wq {
        let n = qname(i);
        let n = if n.starts_with("gpu-gfx-par") { "gpu-gfx-par".to_string() } else { n };
        *by_name.entry(n).or_default() += t;
    }
    let mut q: Vec<(String, u64)> = by_name.into_iter().collect();
    q.sort_by(|a, b| b.1.cmp(&a.1));
    let queues: Vec<String> = q.iter().map(|(n, t)| format!("{n} {:.1}", per(*t))).collect();
    let mut k: Vec<((u8, u64), u64)> = wk.into_iter().collect();
    k.sort_by(|a, b| b.1.cmp(&a.1));
    let kinds: Vec<String> = k
        .iter()
        .take(8)
        .map(|&((i, key), t)| {
            let qn = qname(i);
            let kn = if qn == "uploads" { "texture uploads".to_string() } else { super::exec::batch_key_name(key) };
            format!("{qn} {kn} {:.2}", per(t))
        })
        .collect();
    format!(
        "; GPU busy per frame ms p50 {:.1}, p90 {:.1}, max {:.1} ({n} frames); GPU ms per frame by queue: {}; by batch kind: {}",
        pct(50.0),
        pct(90.0),
        v[n - 1] as f64 / 1e3,
        queues.join(", "),
        kinds.join(", ")
    )
}

/// The clock the presented flips are timed with (us since start).
pub fn now_us() -> u64 {
    crate::hle::kernel::time::process_time_us()
}

/// Notes work of `kind` from `start` to `end` (us, `now_us`) on this thread
/// for the hitch reports (kept if at least 2 ms long; `what` is called only
/// then).
pub fn note(kind: Kind, start: u64, end: u64, what: impl FnOnce() -> String) {
    if end.saturating_sub(start) < MIN_US {
        return;
    }
    let e = Event { start, end, kind, thread: crate::log::thread_name(), what: what() };
    let mut s = STATE.lock();
    if s.events.len() >= KEEP {
        s.events.pop_front();
    }
    s.events.push_back(e);
}

/// How many of a hitch's longest events its line names (`DS_HITCH_TOP=<n>`, diagnosis; 4).
fn shown_top() -> usize {
    static N: std::sync::OnceLock<usize> = std::sync::OnceLock::new();
    *N.get_or_init(|| std::env::var("DS_HITCH_TOP").ok().and_then(|v| v.parse().ok()).unwrap_or(4).clamp(1, 200))
}

/// `note` for work that started at `t0` and ends now.
pub fn note_since(kind: Kind, t0: std::time::Instant, what: impl FnOnce() -> String) {
    let us = t0.elapsed().as_micros() as u64;
    // (While the timeline records, parts of 100 us or more show on the
    // thread's track: what a recording thread was doing inside a command.)
    let tl = us >= 100 && super::timeline::on();
    if us < MIN_US && !tl {
        return;
    }
    let text = what();
    if tl {
        super::timeline::span_ago(us, || format!("{}: {text}", NAMES[kind as usize]));
    }
    if us < MIN_US {
        return;
    }
    let end = now_us();
    note(kind, end.saturating_sub(us), end, || text);
}

/// A presented flip (`pacing::note_flip`): flip `n` shown at `us`. A frame
/// half again as long as the median of the last 60 and at least a vblank
/// over it is a hitch: logged with the work that overlapped it (from a
/// median before its start, as work just before a frame delays it too).
pub fn flip(n: u64, us: u64) {
    let mut s = STATE.lock();
    let prev = s.last_flip;
    s.last_flip = us;
    if prev == 0 || us <= prev {
        return;
    }
    let d = us - prev;
    let median = {
        let mut v: Vec<u32> = s.recent.iter().copied().collect();
        v.sort_unstable();
        v.get(v.len() / 2).copied().unwrap_or(0) as u64
    };
    if s.recent.len() >= 60 {
        s.recent.pop_front();
    }
    s.recent.push_back(d.min(u32::MAX as u64) as u32);
    let hitch = !(s.recent.len() < 9 || d * 2 < median * 3 || d < median + 16_667);
    #[cfg(target_os = "linux")]
    let hitch = hitch || d >= 250_000;
    // Its GPU time once its batches are in (`take_pending`).
    s.pending.push_back(Pending { n, start: prev, end: us, median, hitch });
    take_pending(&mut s);
    if !hitch {
        return;
    }
    let lo = prev.saturating_sub(median);
    // Per kind the time any of its work ran (concurrent work on several
    // threads counted once), clipped to the frame and the median before it.
    let mut spans: [Vec<(u64, u64)>; KINDS] = Default::default();
    let mut over: Vec<(u64, &Event)> = Vec::new();
    for e in s.events.iter().filter(|e| e.end > lo && e.start < us) {
        let (a, b) = (e.start.max(lo), e.end.min(us));
        spans[e.kind as usize].push((a, b));
        over.push((b - a, e));
    }
    let mut sum = [0u64; KINDS];
    for (k, v) in spans.iter_mut().enumerate() {
        v.sort_unstable();
        let mut end = 0u64;
        for &(a, b) in v.iter() {
            let a = a.max(end);
            if b > a {
                sum[k] += b - a;
                end = b;
            }
        }
    }
    over.sort_by(|a, b| b.0.cmp(&a.0));
    #[cfg(target_os = "linux")]
    if d >= 250_000 && crate::platform::stalls::enabled() {
        crate::platform::stalls::record(n, prev, us, median, over.iter().map(|(_, e)| serde_json::json!({
            "start_us":e.start,"end_us":e.end,"kind":NAMES[e.kind as usize],"thread":e.thread,"reason":e.what
        })), over.len());
    }
    let excess = d.saturating_sub(median);
    let top = (0..KINDS).max_by_key(|&k| sum[k]).filter(|&k| sum[k] * 4 >= excess).unwrap_or(KINDS);
    let kinds: Vec<String> = (0..KINDS).filter(|&k| sum[k] > 0).map(|k| format!("{} {:.0} ms", NAMES[k], sum[k] as f64 / 1e3)).collect();
    // The four longest, and the longest on a command processor when none
    // of them is (background compiles at a level load crowd out what the
    // command processors met).
    let background = |e: &Event| e.thread.starts_with("gpu-compile") || e.thread.starts_with("gpu-async") || e.thread.starts_with("gpu-precompile");
    let n_shown = shown_top();
    let mut shown: Vec<&(u64, &Event)> = over.iter().take(n_shown).collect();
    if shown.iter().all(|x| background(x.1)) {
        shown.extend(over.iter().skip(n_shown).find(|x| !background(x.1)));
    }
    let longest: Vec<String> = shown.iter().map(|(o, e)| format!("{} {:.0} ms on {} ({})", NAMES[e.kind as usize], *o as f64 / 1e3, e.thread, e.what)).collect();
    crate::info!(
        "hitch: flip {n} took {:.0} ms (median {:.0}): {}; {}",
        d as f64 / 1e3,
        median as f64 / 1e3,
        if kinds.is_empty() { "no long work noted meanwhile".to_string() } else { kinds.join(", ") },
        if longest.is_empty() { String::new() } else { format!("longest: {}", longest.join("; ")) }
    );
    // The title's main thread meanwhile (`DS_HITCH_TRACE=1`).
    if let Some(t) = crate::guestprof::explain(prev, us) {
        crate::info!("hitch title: flip {n}: {t}");
    }
    if let Some(t) = crate::guestprof::explain_cp(prev, us) {
        crate::info!("hitch CP: flip {n}: {t}");
    }
    if let Some(t) = crate::guestprof::explain_par(prev, us) {
        crate::info!("hitch par: flip {n}: {t}");
    }
    if let Some(t) = crate::guestprof::explain_submit(prev, us) {
        crate::info!("hitch submit: flip {n}: {t}");
    }
    s.hitches += 1;
    s.excess += excess;
    s.by_kind[top] += excess;
    s.worst.push((d, n));
}

/// The window's hitches (one line), and a new window.
pub fn window_report_and_reset() -> String {
    let mut s = STATE.lock();
    let line = if s.hitches == 0 {
        "no hitches".to_string()
    } else {
        let by: Vec<String> = (0..=KINDS).filter(|&k| s.by_kind[k] > 0).map(|k| format!("{} {:.0}", NAMES[k], s.by_kind[k] as f64 / 1e3)).collect();
        s.worst.sort_by(|a, b| b.0.cmp(&a.0));
        let w: Vec<String> = s.worst.iter().take(3).map(|&(d, f)| format!("{:.0} ms (flip {f})", d as f64 / 1e3)).collect();
        format!("{} hitches, {:.0} ms over the median (by cause, ms: {}); longest {}", s.hitches, s.excess as f64 / 1e3, by.join(", "), w.join(", "))
    };
    s.hitches = 0;
    s.excess = 0;
    s.by_kind = [0; KINDS + 1];
    s.worst.clear();
    line
}
