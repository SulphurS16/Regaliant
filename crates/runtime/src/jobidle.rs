//! The title's job workers when they find nothing to run (PPSA01341
//! 01.007.000). BPE's job system keeps jobs whose conditions do not hold yet
//! in its queues: a worker's take (eboot+0x83b880) claims such a job, tests
//! it and puts it back, which raises every eligible worker's job counters
//! again, so the workers' wait for work (eboot+0x83b640) returns at once and
//! they take again. On a console the title has its cores to itself; here its
//! 13 workers keep ~12.9 logical CPUs busy at the courtyard gate, 85% of it
//! in those takes and 1% in the wait (guest profile, FINDINGS.md
//! 2026-10-21), beside the GPU front end's threads and the title's own audio
//! jobs.
//!
//! With `tune job_idle <us>` (0: off, the title's own polling), a worker that
//! found nothing to run for that long (failed takes, or polling in the wait)
//! sleeps `tune job_sleep` us before each further look (~0.5 ms here: the
//! timer's granularity), except one such worker, the watcher, which keeps
//! looking without sleeps for up to `tune job_watch` us (0: no watcher; jobs
//! carry masks of the workers that may run them, so the watch rotates): work
//! that arrives is taken at once, and the sleepers add hands within a sleep.
//! Three hooks
//! (`titlefix`): `take_hook` at the end of a take's pass over one queue (the
//! last queue's end is a failed take), `wait_hook` where the wait's scan
//! found nothing, `run_hook` where the worker runs a job (it is busy again).

use std::cell::Cell;
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};

// Linux timers provide a real sleep here. No permanent watcher: a ready job is polled
// within a bounded sleep. Keep the Windows policy unchanged.
#[cfg(windows)]
pub const DEFAULT_IDLE: u64 = 50;
#[cfg(target_os = "linux")]
pub const DEFAULT_IDLE: u64 = 10;
#[cfg(windows)]
pub const DEFAULT_SLEEP: u64 = 200;
#[cfg(target_os = "linux")]
pub const DEFAULT_SLEEP: u64 = 500;
#[cfg(windows)]
pub const DEFAULT_WATCH: u64 = 2000;
#[cfg(target_os = "linux")]
pub const DEFAULT_WATCH: u64 = 0;

/// The three hooks are in place (`titlefix::apply`).
static READY: AtomicBool = AtomicBool::new(false);
/// A worker is the watcher.
static WATCHED: AtomicBool = AtomicBool::new(false);
/// [sleeps, ns asleep, watcher changes, failed takes, jobs run after a
/// sleep, jobs run, the workers' own cycles in streaks that ended in a job
/// (`DS_IDLE_CENSUS=1`: polling, watching and the sleeps' overhead)].
static STATS: [AtomicU64; 7] = [const { AtomicU64::new(0) }; 7];

/// A worker's streak without work.
#[derive(Clone, Copy, Default)]
struct Streak {
    /// Its start (TSC; 0: no streak).
    since: u64,
    /// When this worker became the watcher (TSC; 0: it is not).
    watching: u64,
    /// It slept in this streak.
    slept: bool,
    /// Its own cycles at the streak's start (`DS_IDLE_CENSUS=1`).
    cycles: u64,
}

thread_local! {
    static STREAK: Cell<Streak> = const { Cell::new(Streak { since: 0, watching: 0, slept: false, cycles: 0 }) };
}

/// Called once the hooks are installed.
pub fn enable() {
    READY.store(true, Ordering::Release);
}

fn ticks_per_us() -> u64 {
    static T: AtomicU64 = AtomicU64::new(0);
    let t = T.load(Ordering::Relaxed);
    if t != 0 {
        return t;
    }
    let t = (crate::hle::kernel::time::tsc_hz() / 1_000_000).max(1);
    T.store(t, Ordering::Relaxed);
    t
}

/// This worker found nothing to run: starts its streak, or once the streak
/// is `job_idle` us old sleeps (or watches).
fn nothing_found() {
    let budget = crate::gpu::tune::job_idle();
    if budget == 0 || !READY.load(Ordering::Relaxed) {
        return;
    }
    let now = unsafe { core::arch::x86_64::_rdtsc() };
    let mut st = STREAK.with(Cell::get);
    if st.since == 0 {
        st.since = now;
        if crate::gpu::threadcpu::idle_census() {
            st.cycles = crate::gpu::threadcpu::own_cycles();
        }
        STREAK.with(|c| c.set(st));
        return;
    }
    let tpu = ticks_per_us();
    let watch = crate::gpu::tune::job_watch();
    if st.watching != 0 {
        if now.wrapping_sub(st.watching) < watch * tpu {
            return;
        }
        // Its watch is over: it sleeps, and the next idle worker to look
        // takes the watch (this one too after its sleep).
        WATCHED.store(false, Ordering::Release);
        st.watching = 0;
        STREAK.with(|c| c.set(st));
    } else if now.wrapping_sub(st.since) < budget * tpu {
        return;
    } else if watch != 0 && WATCHED.compare_exchange(false, true, Ordering::AcqRel, Ordering::Relaxed).is_ok() {
        st.watching = now;
        STREAK.with(|c| c.set(st));
        STATS[2].fetch_add(1, Ordering::Relaxed);
        return;
    }
    let t = std::time::Instant::now();
    crate::thread::short_sleep(std::time::Duration::from_micros(crate::gpu::tune::job_sleep().max(1)));
    STATS[0].fetch_add(1, Ordering::Relaxed);
    STATS[1].fetch_add(t.elapsed().as_nanos() as u64, Ordering::Relaxed);
    if !st.slept {
        st.slept = true;
        STREAK.with(|c| c.set(st));
    }
}

/// The end of a take's pass over one queue (r15: the queue, [rbp-0A0h]: the
/// end of the queues): after the last one the take failed.
pub extern "sysv64" fn take_hook(regs: *const u64, _: u32) {
    if crate::gpu::tune::job_idle() == 0 {
        return;
    }
    let r = unsafe { std::slice::from_raw_parts(regs, 16) };
    let end = unsafe { ((r[5] - 0xA0) as *const u64).read() };
    if r[15].wrapping_add(0x9220) == end {
        STATS[3].fetch_add(1, Ordering::Relaxed);
        nothing_found();
    }
}

/// The wait's scan found no work for this worker.
pub extern "sysv64" fn wait_hook(_: *const u64, _: u32) {
    nothing_found();
}

/// The worker runs a job: its streak ends (a watcher stops watching).
pub extern "sysv64" fn run_hook(_: *const u64, _: u32) {
    crate::placement::note(&|| crate::log::thread_name());
    if crate::gpu::tune::job_idle() != 0 {
        STATS[5].fetch_add(1, Ordering::Relaxed);
    }
    let st = STREAK.with(Cell::get);
    if st.since == 0 && st.watching == 0 {
        return;
    }
    if st.watching != 0 {
        WATCHED.store(false, Ordering::Release);
    }
    if st.slept {
        STATS[4].fetch_add(1, Ordering::Relaxed);
    }
    if st.since != 0 && crate::gpu::threadcpu::idle_census() {
        STATS[6].fetch_add(crate::gpu::threadcpu::own_cycles().wrapping_sub(st.cycles), Ordering::Relaxed);
    }
    STREAK.with(|c| c.set(Streak::default()));
}

/// For the status report.
pub fn status() -> String {
    if !READY.load(Ordering::Relaxed) {
        return String::new();
    }
    let v: Vec<u64> = STATS.iter().map(|c| c.load(Ordering::Relaxed)).collect();
    let census = if crate::gpu::threadcpu::idle_census() { format!(", {:.1} s of their CPU in streaks without work", crate::gpu::threadcpu::ms(v[6]) / 1000.0) } else { String::new() };
    format!(
        "    idle job workers (`tune job_idle` {} us, `job_sleep` {} us, `job_watch` {} us): {} failed takes, {} sleeps ({:.1} s asleep), {} jobs run ({} after a sleep), {} watches{census}\n",
        crate::gpu::tune::job_idle(),
        crate::gpu::tune::job_sleep(),
        crate::gpu::tune::job_watch(),
        v[3],
        v[0],
        v[1] as f64 / 1e9,
        v[5],
        v[4],
        v[2]
    )
}

/// Relay-only snapshots: scheduled CPU seconds of BPE workers, not elapsed sleeps.
#[cfg(target_os = "linux")]
pub fn relay_snapshot() -> serde_json::Value {
    let ticks = unsafe { libc::sysconf(libc::_SC_CLK_TCK) }.max(1) as f64;
    let mut workers = serde_json::Map::new();
    if let Ok(tasks) = std::fs::read_dir("/proc/self/task") {
        for task in tasks.flatten() {
            if !std::fs::read_to_string(task.path().join("comm")).unwrap_or_default().starts_with("BPE JobWorker") { continue; }
            let Ok(stat) = std::fs::read_to_string(task.path().join("stat")) else { continue };
            let Some((_, rest)) = stat.rsplit_once(')') else { continue };
            let fields: Vec<_> = rest.split_whitespace().collect();
            let value = |i: usize| fields.get(i).and_then(|s| s.parse::<u64>().ok()).unwrap_or(0);
            workers.insert(format!("{}:{}", task.file_name().to_string_lossy(), value(19)), serde_json::json!((value(11) + value(12)) as f64 / ticks));
        }
    }
    serde_json::json!({"workers": workers, "failed_takes": STATS[3].load(Ordering::Relaxed),
        "sleeps": STATS[0].load(Ordering::Relaxed), "jobs": STATS[5].load(Ordering::Relaxed),
        "flips": crate::gpu::FRAMES.load(Ordering::Relaxed)})
}

#[cfg(target_os = "linux")]
pub fn relay_delta(before: &serde_json::Value, after: &serde_json::Value, seconds: f64, policy: &str) -> serde_json::Value {
    let workers = after["workers"].as_object().unwrap();
    let cpu: f64 = workers.iter().filter_map(|(id, v)| Some((v.as_f64()? - before["workers"][id].as_f64()?).max(0.0))).sum();
    let delta = |key: &str| after[key].as_u64().unwrap().saturating_sub(before[key].as_u64().unwrap());
    serde_json::json!({"policy": policy, "seconds": seconds, "workers": workers.len(),
        "worker_cpu_s": cpu, "worker_cpu_s_per_gameplay_s": cpu / seconds,
        "flips_per_second": delta("flips") as f64 / seconds, "failed_takes": delta("failed_takes"),
        "jobs": delta("jobs"), "sleeps": delta("sleeps")})
}
