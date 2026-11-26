//! Bounded long-frame evidence from existing hitch events, without GPU queries or
//! sampling a thread from a signal handler. Written only at orderly shutdown.
use parking_lot::Mutex;
use serde_json::{Value, json};
use std::sync::OnceLock;
static FRAMES: Mutex<Vec<Value>> = Mutex::new(Vec::new());
static DROPPED: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);
pub fn enabled() -> bool {
    static ON: OnceLock<bool> = OnceLock::new();
    *ON.get_or_init(|| std::env::var("DS_FRAME_STALLS").ok().as_deref() == Some("1"))
}
pub fn record(frame: u64, start: u64, end: u64, median: u64, events: impl Iterator<Item=Value>, total: usize) {
    if !enabled() { return; }
    let mut frames = FRAMES.lock();
    if frames.len() >= 4096 { DROPPED.fetch_add(1, std::sync::atomic::Ordering::Relaxed); return; }
    frames.push(json!({"frame":frame, "start_us":start, "end_us":end, "duration_ms":(end-start) as f64/1000.,
        "median_ms":median as f64/1000., "events":events.take(128).collect::<Vec<_>>(), "overlapping_events":total, "omitted_events":total.saturating_sub(128),
        "note":"Longest retained operations; categories/threads overlap. Events below 2ms and unfinished operations are absent; this is not a stack sample."}));
}
pub fn flush() {
    if !enabled() { return; }
    let Some(dir) = crate::run_dir() else { return };
    let frames = std::mem::take(&mut *FRAMES.lock());
    let mut trace = Vec::new();
    for f in &frames {
        let a = f["start_us"].as_u64().unwrap();
        let b = f["end_us"].as_u64().unwrap();
        trace.push(json!({"ph":"X", "pid":1,"tid":"frames", "name":format!("flip {}",f["frame"]), "ts":a,"dur":b-a}));
        for e in f["events"].as_array().unwrap() {
            let lo = e["start_us"].as_u64().unwrap().max(a);
            let hi = e["end_us"].as_u64().unwrap().min(b);
            if hi > lo { trace.push(json!({"ph":"X","pid":1,"tid":e["thread"],"name":format!("{}: {}",e["kind"],e["reason"]),"ts":lo,"dur":hi-lo})); }
        }
    }
    let result = (|| -> std::io::Result<()> {
        use std::io::Write;
        let mut f = std::io::BufWriter::new(std::fs::File::create(dir.join("frame-stalls.jsonl"))?);
        for row in &frames { serde_json::to_writer(&mut f, row)?; writeln!(f)?; }
        let process_us = crate::hle::kernel::time::process_time_us();
        let mut clock: libc::timespec = unsafe { std::mem::zeroed() };
        unsafe { libc::clock_gettime(libc::CLOCK_MONOTONIC, &mut clock); }
        writeln!(f,"{}",json!({"dropped_frames":DROPPED.load(std::sync::atomic::Ordering::Relaxed),
            "clock_process_us":process_us,"clock_monotonic_ns":clock.tv_sec as u64*1_000_000_000+clock.tv_nsec as u64}))?;
        f.flush()?;
        let mut trace_file = std::io::BufWriter::new(std::fs::File::create(dir.join("frame-stalls.trace.json"))?);
        serde_json::to_writer(&mut trace_file, &json!({"traceEvents":trace,"displayTimeUnit":"ms"}))?;
        trace_file.flush()
    })();
    if let Err(e) = result { crate::warn!("frame stall output: {e}"); }
    else { crate::info!("frame stalls: {} intervals >=250ms; {}", frames.len(), dir.display()); }
}
