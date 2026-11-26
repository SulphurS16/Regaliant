//! CPU time of the GPU front end's threads from their cycle counters, by
//! kind: the queues' command processors ("gfx", "acb"), the parallel
//! recording workers ("par"), stage preparation ("prep"), batch submission
//! ("submit"), batch completion ("complete") and texture scrubbing
//! ("scrub"). QueryThreadCycleTime counts
//! the time-stamp counter while the thread runs, exactly (thread times from
//! GetThreadTimes advance in clock ticks of ~15.6 ms), and does not include
//! time other processes take from the GPU: per-command costs compare across
//! windows with different outside load, where frame rates do not. The
//! counter's rate is measured once against the system clock.

use parking_lot::Mutex;
use std::collections::HashMap;
use std::sync::atomic::Ordering;
use std::time::Instant;

pub const KINDS: [&str; 8] = ["gfx", "acb", "par", "prep", "submit", "complete", "scrub", "ahead"];

/// Thread handles by host thread id, kept open (the threads live as long as
/// the process).
static HANDLES: Mutex<Option<HashMap<u32, usize>>> = Mutex::new(None);

fn thread_cycles(tid: u32) -> Option<u64> {
    use crate::platform::os::Win32::System::Threading::{OpenThread, THREAD_QUERY_LIMITED_INFORMATION};
    use crate::platform::os::Win32::System::WindowsProgramming::QueryThreadCycleTime;
    let mut g = HANDLES.lock();
    let m = g.get_or_insert_with(HashMap::new);
    let h = match m.get(&tid) {
        Some(&h) => h,
        None => {
            let h = unsafe { OpenThread(THREAD_QUERY_LIMITED_INFORMATION, 0, tid) };
            if h.is_null() {
                return None;
            }
            m.insert(tid, h as usize);
            h as usize
        }
    };
    let mut c = 0u64;
    (unsafe { QueryThreadCycleTime(h as _, &mut c) } != 0).then_some(c)
}

/// The calling thread's cycles so far (the timeline's recording spans: CPU
/// time beside wall time).
pub fn own_cycles() -> u64 {
    use crate::platform::os::Win32::System::Threading::GetCurrentThread;
    use crate::platform::os::Win32::System::WindowsProgramming::QueryThreadCycleTime;
    let mut c = 0u64;
    unsafe { QueryThreadCycleTime(GetCurrentThread(), &mut c) };
    c
}

/// Cycles so far of the threads of one sampler kind ("par", "cpar", ...): the
/// graphics queue's chunk workers apart from the compute queues' ones, which
/// `snapshot` counts together.
pub fn kind_cycles(kind: &str) -> u64 {
    super::cpu_threads().into_iter().filter(|(k, _)| *k == kind).filter_map(|(_, tid)| thread_cycles(tid)).sum()
}

/// Whether idle CPU is measured (`DS_IDLE_CENSUS=1`): the chunk workers' own
/// cycles in their runs and turn waits (`par`), the title's job workers' in
/// their streaks without work (`jobidle`). Two thread-cycle reads a run or a
/// streak.
pub fn idle_census() -> bool {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ON.get_or_init(|| std::env::var("DS_IDLE_CENSUS").ok().as_deref() == Some("1"))
}

/// Time-stamp counter cycles per second (measured over 20 ms at first use).
pub fn tsc_hz() -> f64 {
    static HZ: std::sync::OnceLock<f64> = std::sync::OnceLock::new();
    *HZ.get_or_init(|| {
        let (c0, t0) = (unsafe { core::arch::x86_64::_rdtsc() }, Instant::now());
        std::thread::sleep(std::time::Duration::from_millis(20));
        let c1 = unsafe { core::arch::x86_64::_rdtsc() };
        (c1.wrapping_sub(c0)) as f64 / t0.elapsed().as_secs_f64()
    })
}

/// Milliseconds of `cycles`.
pub fn ms(cycles: u64) -> f64 {
    cycles as f64 * 1000.0 / tsc_hz()
}

/// Cycles so far per kind (indexed like `KINDS`).
pub fn snapshot() -> [u64; KINDS.len()] {
    let mut out = [0u64; KINDS.len()];
    for (kind, tid) in super::cpu_threads() {
        // (The compute queues' workers, "cpar" in the sampler, count as "par".)
        let kind = if kind == "cpar" { "par" } else { kind };
        if let (Some(k), Some(c)) = (KINDS.iter().position(|&x| x == kind), thread_cycles(tid)) {
            out[k] += c;
        }
    }
    out
}

/// Commands (draws and dispatches) executed so far.
pub fn commands() -> u64 {
    super::cp::DRAWS.load(Ordering::Relaxed) + super::cp::DISPATCHES.load(Ordering::Relaxed)
}

struct Window {
    at: Instant,
    cycles: [u64; KINDS.len()],
    flips: u64,
    commands: u64,
    splits: u64,
    /// The title's threads: host thread id -> (name, cycles) at the start.
    title: HashMap<u32, (String, u64)>,
    /// Every thread's kernel time, user time and context switches at the
    /// start (`thread_times`).
    times: HashMap<u32, (u64, u64, u64)>,
}

use crate::platform::thread_times::thread_times;

/// The window's kernel share and context switches by thread kind (`profile
/// report`): tick-charged thread times, so shares, not exact ms.
fn times_line(w: &Window, flips: u64) -> String {
    let now = thread_times();
    if now.is_empty() || w.times.is_empty() {
        return String::new();
    }
    let delta = |tid: u32| {
        let (a, b) = (now.get(&tid).copied().unwrap_or_default(), w.times.get(&tid).copied().unwrap_or_default());
        (a.0.saturating_sub(b.0), a.1.saturating_sub(b.1), a.2.wrapping_sub(b.2) & 0xFFFF_FFFF)
    };
    let mut kinds: Vec<(String, [u64; 3])> = Vec::new();
    let mut add = |name: &str, d: (u64, u64, u64)| match kinds.iter_mut().find(|x| x.0 == name) {
        Some(x) => {
            x.1[0] += d.0;
            x.1[1] += d.1;
            x.1[2] += d.2;
        }
        None => kinds.push((name.to_string(), [d.0, d.1, d.2])),
    };
    for (kind, tid) in super::cpu_threads() {
        add(kind, delta(tid));
    }
    for (tid, (name, _)) in title_snapshot() {
        let kind = if name.starts_with("BPE JobWorkerThread") { "title jobs" } else { "title others" };
        add(kind, delta(tid));
    }
    let parts: Vec<String> = kinds
        .iter()
        .filter(|x| x.1[0] + x.1[1] > 0)
        .map(|(k, d)| format!("{k} {:.1} of {:.1} ({:.0} switches)", d[0] as f64 / 1e4 / flips as f64, (d[0] + d[1]) as f64 / 1e4 / flips as f64, d[2] as f64 / flips as f64))
        .collect();
    format!("    kernel time of thread time and context switches in the window, per flip (times charged by clock ticks: shares, not exact ms): {}\n", parts.join(", "))
}

/// The title's live threads' cycles so far, by host thread id.
fn title_snapshot() -> HashMap<u32, (String, u64)> {
    let list: Vec<(u32, String)> = crate::thread::THREADS
        .lock()
        .iter()
        .filter(|t| t.exited.try_lock().is_some_and(|e| e.is_none()))
        .map(|t| (t.host_tid.load(Ordering::Relaxed) as u32, t.name()))
        .filter(|x| x.0 != 0)
        .collect();
    list.into_iter().filter_map(|(tid, name)| thread_cycles(tid).map(|c| (tid, (name, c)))).collect()
}

static WINDOW: Mutex<Option<Window>> = Mutex::new(None);

/// The front end's cycles and commands at the start of the frame window
/// (`gpu::pacing`, its own windows beside the profile's).
static FRAMES_WINDOW: Mutex<Option<([u64; KINDS.len()], u64)>> = Mutex::new(None);

/// The frame window's front-end CPU per flip by kind (one clause of the
/// `frames at` line; cycle counts are not moved by other programs' load the
/// way frame rates are), and a new window.
pub fn frames_window_clause(flips: u64) -> String {
    let now = (snapshot(), commands());
    let Some((c0, k0)) = FRAMES_WINDOW.lock().replace(now) else { return String::new() };
    let flips = flips.max(1) as f64;
    let d: Vec<u64> = (0..KINDS.len()).map(|i| now.0[i].saturating_sub(c0[i])).collect();
    let all: u64 = d.iter().sum();
    let cmds = now.1.saturating_sub(k0).max(1);
    let kind = |k: &str| ms(d[KINDS.iter().position(|x| *x == k).unwrap()]) / flips;
    let main: Vec<String> = ["gfx", "acb", "par"].iter().map(|k| format!("{k} {:.1}", kind(k))).collect();
    // (The submission thread replays deferred batches, `tune record_defer`.)
    format!("; front-end CPU ms per flip: {} (all {:.1}; {:.2} us per command), submission {:.1}", main.join(", "), ms(all) / flips, ms(all) * 1000.0 / cmds as f64, kind("submit"))
}

/// Cycles at the frame window's start of each compute queue's command
/// processor (by queue id) and of the title's threads (by host thread id).
static FRAMES_DETAIL: Mutex<Option<(HashMap<u32, u64>, HashMap<u32, (String, u64)>)>> = Mutex::new(None);

/// The frame window's CPU per flip of the compute queues' command
/// processors (the four busiest) and of the title's threads (job workers,
/// main thread, the rest): one clause of the `frames at` line.
pub fn frames_detail_clause(flips: u64) -> String {
    let queues: HashMap<u32, u64> = super::compute_threads().into_iter().filter_map(|(id, tid)| thread_cycles(tid).map(|c| (id, c))).collect();
    let now = (queues, title_snapshot());
    let Some((q0, t0)) = FRAMES_DETAIL.lock().replace(now.clone()) else { return String::new() };
    let flips = flips.max(1) as f64;
    let mut qs: Vec<(u32, f64)> = now.0.iter().map(|(&id, &c)| (id, ms(c.saturating_sub(*q0.get(&id).unwrap_or(&c))) / flips)).collect();
    qs.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
    let (mut job, mut main, mut other) = (0.0, 0.0, 0.0);
    for (tid, (name, c)) in &now.1 {
        let d = ms(c.saturating_sub(t0.get(tid).map_or(*c, |x| x.1))) / flips;
        if name.starts_with("BPE JobWorkerThread") {
            job += d;
        } else if name == "GAME_MainThread" {
            main += d;
        } else {
            other += d;
        }
    }
    let q: Vec<String> = qs.iter().take(4).map(|(id, v)| format!("{id:#x} {v:.1}")).collect();
    format!("; compute queues ms per flip: {}; the title's CPU ms per flip: job workers {job:.1}, main thread {main:.1}, other {other:.1}", q.join(", "))
}

/// Starts a profile window (`profile reset`).
pub fn reset_window() {
    let w = Window { at: Instant::now(), cycles: snapshot(), flips: super::FRAMES.load(Ordering::Relaxed), commands: commands(), splits: super::cp::SPLITS.load(Ordering::Relaxed), title: title_snapshot(), times: thread_times() };
    *WINDOW.lock() = Some(w);
}

/// The window's thread CPU time per flip and per command (`profile report`).
pub fn window_line() -> String {
    let g = WINDOW.lock();
    let Some(w) = g.as_ref() else { return String::new() };
    let now = snapshot();
    let d: Vec<u64> = (0..KINDS.len()).map(|i| now[i].saturating_sub(w.cycles[i])).collect();
    let flips = super::FRAMES.load(Ordering::Relaxed).saturating_sub(w.flips).max(1);
    let cmds = commands().saturating_sub(w.commands).max(1);
    let all: u64 = d.iter().sum();
    let parts: Vec<String> = KINDS.iter().zip(&d).map(|(k, &c)| format!("{k} {:.1}", ms(c) / flips as f64)).collect();
    let splits = super::cp::SPLITS.load(Ordering::Relaxed).saturating_sub(w.splits);
    let split = if splits > 0 { format!("; {:.1} batch splits per flip (`split_every`)", splits as f64 / flips as f64) } else { String::new() };
    // The title's threads (its own frame's CPU side): all, the main thread,
    // the job workers, the busiest others.
    let mut per: Vec<(String, f64)> = title_snapshot()
        .into_iter()
        .map(|(tid, (name, c))| {
            let c0 = w.title.get(&tid).map_or(0, |x| x.1);
            (name, ms(c.saturating_sub(c0)) / flips as f64)
        })
        .collect();
    per.sort_by(|a, b| b.1.total_cmp(&a.1));
    let title_all: f64 = per.iter().map(|x| x.1).sum();
    let main: f64 = per.iter().filter(|x| x.0 == "GAME_MainThread").map(|x| x.1).sum();
    let jobs: Vec<f64> = per.iter().filter(|x| x.0.starts_with("BPE JobWorkerThread")).map(|x| x.1).collect();
    let others: Vec<String> = per.iter().filter(|x| x.0 != "GAME_MainThread" && !x.0.starts_with("BPE JobWorkerThread")).take(4).map(|x| format!("{} {:.1}", x.0, x.1)).collect();
    format!(
        "    thread CPU in the window, ms per flip: {} (all {:.1}); {:.2} us per command over all threads; {flips} flips, {cmds} commands in {:.1} s{split}\n    the title's threads' CPU in the window, ms per flip: all {title_all:.1}; main thread {main:.1}; {} job workers {:.1} (busiest {:.1}); others {}\n",
        parts.join(", "),
        ms(all) / flips as f64,
        ms(all) * 1000.0 / cmds as f64,
        w.at.elapsed().as_secs_f64(),
        jobs.len(),
        jobs.iter().sum::<f64>(),
        jobs.first().copied().unwrap_or(0.0),
        others.join(", ")
    ) + &times_line(w, flips)
}
