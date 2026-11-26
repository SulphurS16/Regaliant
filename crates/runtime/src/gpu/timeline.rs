//! Timeline of command processing and GPU execution over a few flips
//! (`timeline <flips>` control command): spans of every command processor
//! (recording a batch and what ended it; waits for batch slots, for GPU work
//! writing what it reads, for labels and for parallel workers), the
//! submission thread's vkQueueSubmit calls, the completion thread's CP
//! effects and, with DS_PROFILE_GPU=1 (best with DS_PROFILE_GPU_STAMPS=0),
//! every batch's execution on the GPU, its timestamps put on the CPU clock
//! with VK_EXT_calibrated_timestamps, and with DS_PROFILE_GPU_STAMPS=seg
//! every barrier segment of a batch with its programs (track "GPU
//! segments"). Written to `timeline-<flip>.json` in the run directory (Chrome
//! trace format: ui.perfetto.dev, chrome://tracing; `tools/timeline.py`
//! summarizes it); `otherData` holds the performance counter at its zero and
//! the counter's frequency (`tools/gpuattrib.py` puts Nsight Systems' GPU
//! metric samples on it).

use parking_lot::Mutex;
use std::sync::OnceLock;
use std::sync::atomic::{AtomicBool, AtomicU32, Ordering};

static ON: AtomicBool = AtomicBool::new(false);
static FLIPS_LEFT: AtomicU32 = AtomicU32::new(0);
static STATE: Mutex<Option<State>> = Mutex::new(None);
/// Spans kept at most (a courtyard frame makes a few thousand).
const MAX_SPANS: usize = 1 << 20;

// Relay phase boundaries wait for the writer too: formatting a detailed trace
// must not spill CPU work into the following clean measurement.
static WRITERS: AtomicU32 = AtomicU32::new(0);
struct Writer;
impl Drop for Writer { fn drop(&mut self) { WRITERS.fetch_sub(1, Ordering::Release); } }
#[cfg(target_os = "linux")]
pub fn busy() -> bool { on() || WRITERS.load(Ordering::Acquire) != 0 }

struct Span {
    track: String,
    name: String,
    t0: i64,
    t1: i64,
}

struct State {
    spans: Vec<Span>,
    details: Vec<(String, Vec<DetailSpan>)>,
    /// A GPU timestamp and the performance counter read together,
    /// refreshed at every flip.
    cal: Option<(u64, i64)>,
    started: i64,
}

pub fn on() -> bool {
    ON.load(Ordering::Relaxed)
}

/// The timeline's clock (the performance counter).
pub fn now() -> i64 {
    let mut t = 0i64;
    unsafe { crate::platform::os::Win32::System::Performance::QueryPerformanceCounter(&mut t) };
    t
}

fn freq() -> i64 {
    static F: OnceLock<i64> = OnceLock::new();
    *F.get_or_init(|| {
        let mut f = 0i64;
        unsafe { crate::platform::os::Win32::System::Performance::QueryPerformanceFrequency(&mut f) };
        f.max(1)
    })
}

/// The start of a span while recording.
pub fn mark() -> Option<i64> {
    on().then(now)
}

/// Short, explicitly armed CPU recording detail trace. Normal gameplay and the
/// ordinary wait trace keep this off (per-packet/lap events can be numerous).
static CPU_DETAILS: AtomicBool = AtomicBool::new(false);
#[cfg(target_os = "linux")]
pub struct CpuDetails;
#[cfg(target_os = "linux")]
pub fn cpu_details() -> CpuDetails { CPU_DETAILS.store(true, Ordering::Relaxed); CpuDetails }
#[cfg(target_os = "linux")]
impl Drop for CpuDetails { fn drop(&mut self) { CPU_DETAILS.store(false, Ordering::Relaxed); } }
pub fn detail_mark() -> Option<i64> { if CPU_DETAILS.load(Ordering::Relaxed) { mark() } else { None } }
pub fn detail(name: &'static str) -> Section { Section { t0: detail_mark(), name, detailed: true } }

// Detailed packet/lap spans used to allocate two Strings and contend on STATE on
// every mark. Their cost fell outside the lap, into record_packet. Buffer fixed-size
// records per thread; expand names only on the trace writer after the last flip.
struct DetailSpan { name: &'static str, lap: bool, t0: i64, t1: i64 }
struct DetailBuffer { track: String, spans: Mutex<Vec<DetailSpan>> }
static DETAILS: Mutex<Vec<std::sync::Arc<DetailBuffer>>> = Mutex::new(Vec::new());
static DETAIL_START: std::sync::atomic::AtomicI64 = std::sync::atomic::AtomicI64::new(0);
thread_local! {
    static DETAIL_BUFFER: std::sync::Arc<DetailBuffer> = {
        let b = std::sync::Arc::new(DetailBuffer { track: crate::log::thread_name(), spans: Mutex::new(Vec::with_capacity(16384)) });
        DETAILS.lock().push(b.clone());
        b
    };
}
fn detail_end(t0: Option<i64>, name: &'static str, lap: bool) {
    let Some(t0) = t0 else { return };
    let t1 = now();
    DETAIL_BUFFER.with(|b| {
        let mut spans = b.spans.lock();
        if on() && t0 >= DETAIL_START.load(Ordering::Relaxed) && spans.len() < MAX_SPANS / 4 {
            spans.push(DetailSpan { name, lap, t0, t1 });
        }
    });
}
pub fn detail_lap(t0: Option<i64>, name: &'static str) { detail_end(t0, name, true); }

/// CPU-only phase on the current thread, including early returns. No clock or allocation
/// while the timeline is off; this never requests a GPU query or queue flush.
pub struct Section { t0: Option<i64>, name: &'static str, detailed: bool }
pub fn section(name: &'static str) -> Section { Section { t0: mark(), name, detailed: false } }
impl Drop for Section {
    fn drop(&mut self) {
        if self.detailed { detail_end(self.t0, self.name, false); }
        else if self.t0.is_some() { end(&crate::log::thread_name(), self.t0, || self.name.into()); }
    }
}

thread_local! {
    /// Nanoseconds the calling thread spent in waits (`Wait`).
    static WAITED: std::cell::Cell<u64> = const { std::cell::Cell::new(0) };
}

/// Nanoseconds the calling thread has spent in waits so far.
pub fn waited_ns() -> u64 {
    WAITED.with(|w| w.get())
}

/// A wait of the calling thread: always timed (`waited_ns`), a span when the
/// timeline records.
pub struct Wait {
    _cpu: super::cpu_profile::Scope,
    t0: std::time::Instant,
    tl: Option<i64>,
}

pub fn wait_start() -> Wait {
    Wait { _cpu: super::cpu_profile::scope(super::cpu_profile::Part::Wait), t0: std::time::Instant::now(), tl: mark() }
}

impl Wait {
    /// Ends the wait (a span named `name` on `track`); how long it took.
    pub fn end(self, track: &str, name: impl FnOnce() -> String) -> std::time::Duration {
        let d = self.t0.elapsed();
        WAITED.with(|w| w.set(w.get() + d.as_nanos() as u64));
        if super::cpu_profile::active() {
            // (Profiling only: the name is built once for both uses.)
            let n = name();
            super::cpu_profile::note_wait(&n, d.as_nanos() as u64);
            end(track, self.tl, || n);
            return d;
        }
        end(track, self.tl, name);
        d
    }
}

/// Ends a span begun with `mark` on `track`.
pub fn end(track: &str, t0: Option<i64>, name: impl FnOnce() -> String) {
    let Some(t0) = t0 else { return };
    let t1 = now();
    if let Some(s) = STATE.lock().as_mut() {
        if s.spans.len() < MAX_SPANS {
            s.spans.push(Span { track: track.to_string(), name: name(), t0, t1 });
        }
    }
}

/// A span of `us` microseconds ending now on the calling thread's track
/// (`gpu::hitch`'s long operations while recording).
pub fn span_ago(us: u64, name: impl FnOnce() -> String) {
    if !on() {
        return;
    }
    let t1 = now();
    let t0 = t1 - (us as i64).saturating_mul(freq()) / 1_000_000;
    if let Some(s) = STATE.lock().as_mut() {
        if s.spans.len() < MAX_SPANS {
            s.spans.push(Span { track: crate::log::thread_name(), name: name(), t0, t1 });
        }
    }
}

/// A span on `track` that began `ago` before now and ends now (a title
/// submission's time in its queue before its command processor took it).
pub fn span_back(track: &str, ago: std::time::Duration, name: impl FnOnce() -> String) {
    if !on() {
        return;
    }
    let t1 = now();
    let t0 = t1 - (ago.as_micros().min(i64::MAX as u128 / 16_000_000) as i64).saturating_mul(freq()) / 1_000_000;
    if let Some(s) = STATE.lock().as_mut() {
        if s.spans.len() < MAX_SPANS {
            s.spans.push(Span { track: track.to_string(), name: name(), t0, t1 });
        }
    }
}

/// Two adjacent wall-time phases ending now (a Vulkan queue-lock wait followed by the
/// driver call and preparation under that lock). One clock origin keeps their boundary exact.
pub fn split_back(track: &str, first: std::time::Duration, second: std::time::Duration, name: &str) {
    if !on() { return; }
    let t2 = now();
    let ticks = |d: std::time::Duration| (d.as_secs_f64() * freq() as f64) as i64;
    let t1 = t2 - ticks(second);
    let t0 = t1 - ticks(first);
    if let Some(s) = STATE.lock().as_mut() {
        if s.spans.len() + 2 <= MAX_SPANS {
            s.spans.push(Span { track: track.into(), name: format!("queue lock: {name}"), t0, t1 });
            s.spans.push(Span { track: track.into(), name: format!("call/preparation: {name}"), t0: t1, t1: t2 });
        }
    }
}

/// Ends a span begun with `mark` on the calling thread's track when it
/// lasted at least `min_us` (parts too small to list otherwise).
pub fn end_min(t0: Option<i64>, min_us: i64, name: impl FnOnce() -> String) {
    let Some(t) = t0 else { return };
    if (now() - t) * 1_000_000 >= min_us * freq() {
        end(&crate::log::thread_name(), t0, name);
    }
}

/// Starts recording for the next `flips` flips.
pub fn start(flips: u32) -> Result<(), String> {
    let c = super::vk::try_ctx().ok_or("no Vulkan device")?;
    let cal = super::exec::batch_stamps().then(|| c.calibrate()).flatten();
    if super::exec::batch_stamps() && cal.is_none() {
        crate::warn!("timeline: no calibrated timestamps (VK_EXT_calibrated_timestamps): GPU spans are left out");
    }
    let started = now();
    DETAIL_START.store(started, Ordering::Relaxed);
    for b in DETAILS.lock().iter() { b.spans.lock().clear(); }
    *STATE.lock() = Some(State { spans: Vec::with_capacity(1 << 16), details: Vec::new(), cal, started });
    FLIPS_LEFT.store(flips.max(1), Ordering::SeqCst);
    ON.store(true, Ordering::SeqCst);
    Ok(())
}

/// A batch's execution on the GPU between its timestamps `begin` and `end`.
pub fn gpu(queue: &str, seq: u64, begin: u64, end: u64, what: &str) {
    gpu_on("GPU", begin, end, || format!("{queue} #{seq}{what}"));
}

/// A barrier segment of a batch on the GPU (`seg_stamps`): the queue and its
/// programs (or what it held instead).
pub fn gpu_segment(begin: u64, end: u64, name: impl FnOnce() -> String) {
    gpu_on("GPU segments", begin, end, name);
}

fn gpu_on(track: &str, begin: u64, end: u64, name: impl FnOnce() -> String) {
    if !on() || end < begin {
        return;
    }
    let period = super::vk::ctx().limits.timestamp_period as f64;
    let f = freq() as f64;
    let mut g = STATE.lock();
    let Some(s) = g.as_mut() else { return };
    let Some((g0, q0)) = s.cal else { return };
    let conv = |t: u64| q0 + (t.wrapping_sub(g0) as i64 as f64 * period * f / 1e9) as i64;
    if s.spans.len() < MAX_SPANS {
        s.spans.push(Span { track: track.to_string(), name: name(), t0: conv(begin), t1: conv(end) });
    }
}

/// Flip `n` happened (vblank thread): after the requested number the
/// recording ends and the file is written on a thread of its own.
pub fn flipped(n: u64) {
    if !on() {
        return;
    }
    let t = now();
    let cal = super::exec::batch_stamps().then(|| super::vk::try_ctx().and_then(|c| c.calibrate())).flatten();
    if let Some(s) = STATE.lock().as_mut() {
        s.spans.push(Span { track: "flips".to_string(), name: format!("flip {n}"), t0: t, t1: t });
        if cal.is_some() {
            s.cal = cal;
        }
    }
    if FLIPS_LEFT.fetch_sub(1, Ordering::SeqCst) == 1 {
        WRITERS.fetch_add(1, Ordering::Release);
        let writer = Writer;
        ON.store(false, Ordering::SeqCst);
        if let Some(mut st) = STATE.lock().take() {
            for b in DETAILS.lock().iter() {
                st.details.push((b.track.clone(), std::mem::take(&mut *b.spans.lock())));
            }
            std::thread::spawn(move || { let _writer = writer; write(st, n); });
        }
    }
}

fn write(mut st: State, flip: u64) {
    for (track, spans) in std::mem::take(&mut st.details) {
        for d in spans {
            if st.spans.len() >= MAX_SPANS { break; }
            st.spans.push(Span { track: track.clone(), name: if d.lap { format!("record step: {}", d.name) } else { d.name.into() }, t0: d.t0, t1: d.t1 });
        }
    }
    let Some(dir) = crate::run_dir() else { return };
    let path = dir.join(format!("timeline-{flip}.json"));
    let us = |t: i64| (t - st.started) as f64 * 1e6 / freq() as f64;
    let mut tracks: Vec<&str> = Vec::new();
    for s in &st.spans {
        if !tracks.contains(&s.track.as_str()) {
            tracks.push(&s.track);
        }
    }
    tracks.sort();
    let tid = |t: &str| tracks.iter().position(|x| *x == t).unwrap_or(0) + 1;
    let esc = |s: &str| s.replace('\\', "\\\\").replace('"', "\\\"");
    let mut out = String::with_capacity(st.spans.len() * 100);
    out.push_str("{\"traceEvents\":[\n");
    for t in &tracks {
        out.push_str(&format!("{{\"name\":\"thread_name\",\"ph\":\"M\",\"pid\":1,\"tid\":{},\"args\":{{\"name\":\"{}\"}}}},\n", tid(t), esc(t)));
    }
    for (i, s) in st.spans.iter().enumerate() {
        let sep = if i + 1 < st.spans.len() { ",\n" } else { "\n" };
        if s.t1 == s.t0 && s.track == "flips" {
            out.push_str(&format!("{{\"name\":\"{}\",\"ph\":\"i\",\"s\":\"g\",\"pid\":1,\"tid\":{},\"ts\":{:.3}}}{sep}", esc(&s.name), tid(&s.track), us(s.t0)));
        } else {
            out.push_str(&format!("{{\"name\":\"{}\",\"ph\":\"X\",\"pid\":1,\"tid\":{},\"ts\":{:.3},\"dur\":{:.3}}}{sep}", esc(&s.name), tid(&s.track), us(s.t0), us(s.t1) - us(s.t0)));
        }
    }
    out.push_str(&format!("],\"otherData\":{{\"qpc_start\":{},\"qpc_freq\":{}}}}}\n", st.started, freq()));
    match std::fs::write(&path, out) {
        Ok(()) => crate::info!("timeline: {} spans on {} tracks written to {}", st.spans.len(), tracks.len(), path.display()),
        Err(e) => crate::warn!("timeline: writing {}: {e}", path.display()),
    }
}

#[cfg(all(test, target_os = "linux"))]
mod writer_tests {
    use super::*;
    #[test]
    fn writer_work_remains_busy_after_capture_stops() {
        assert!(!on());
        let before = WRITERS.fetch_add(1, Ordering::Release);
        let writer = Writer;
        assert!(busy());
        drop(writer);
        assert_eq!(WRITERS.load(Ordering::Acquire), before);
    }
}
