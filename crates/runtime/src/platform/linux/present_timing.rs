//! Bounded, in-memory gameplay presentation samples. KHR wait completion is a CPU
//! observation, not a scanout timestamp. It is the relay pacing measure.
use parking_lot::Mutex;
use std::time::Instant;
struct Row { kind: &'static str, id: u64, ns: u64, costs: [f64; 6], generated: bool, age: f64, compute: f64 }
struct State { begin: u64, rows: Vec<Row>, submitted: std::collections::HashMap<u64,bool>, reuse: serde_json::Value }
static STATE: Mutex<Option<State>> = Mutex::new(None);
fn now() -> u64 {
    let mut t: libc::timespec = unsafe { std::mem::zeroed() };
    unsafe { libc::clock_gettime(libc::CLOCK_MONOTONIC, &mut t); }
    t.tv_sec as u64*1_000_000_000 + t.tv_nsec as u64
}
pub fn begin() {
    let reuse = crate::gpu::pcache::reuse_report();
    *STATE.lock() = Some(State { begin: now(), rows: Vec::with_capacity(16000), submitted: std::collections::HashMap::with_capacity(4096), reuse });
}
fn record(kind: &'static str, id: u64, ns: u64, costs: [f64; 6]) {
    if let Some(s) = STATE.lock().as_mut() {
        // A completion for a pre-window enqueue must not pollute the new interval
        // after a capture or swapchain wait. Submission is recorded before the waiter
        // receives its present ID. Both series use the same KHR ID when supported.
        if ns >= s.begin && s.rows.len() < 100000 {
            let (generated, age, compute) = crate::gpu::framegen::timing();
            if kind == "submit" { s.submitted.insert(id, generated); }
            else if !s.submitted.contains_key(&id) { return; }
            s.rows.push(Row { kind, id, ns, costs, generated: s.submitted[&id], age: if kind=="submit" {age}else{0.0}, compute: if kind=="submit" {compute}else{0.0} });
        }
    }
}
pub fn submit(id: u64, times: [Instant; 7]) {
    let mut costs = [0.; 6];
    for i in 0..6 { costs[i] = times[i+1].duration_since(times[i]).as_secs_f64()*1000.; }
    record("submit", id, now(), costs);
}
pub fn completed(id: u64) { record("wait_complete", id, now(), [0.; 6]); }
pub fn end(phase: &str) {
    let Some(s) = STATE.lock().take() else { return };
    let Some(dir) = crate::run_dir() else { return };
    let reuse = serde_json::json!({"begin":s.reuse,"end":crate::gpu::pcache::reuse_report()});
    if let Err(e) = std::fs::write(dir.join(format!("{phase}-pipeline-reuse.json")), reuse.to_string()) { crate::warn!("pipeline reuse interval: {e}"); }
    use std::io::Write;
    let path = dir.join(format!("{phase}-presentation.csv"));
    let result = (|| -> std::io::Result<()> {
        let mut f = std::io::BufWriter::new(std::fs::File::create(&path)?);
        writeln!(f, "kind,id,monotonic_ns,slot_ms,acquire_ms,record_submit_ms,gpu_ready_ms,capture_queue_ms,present_ms,generated,fg_snapshot_age_ms,fg_compute_completion_ms")?;
        for r in s.rows {
            writeln!(f, "{},{},{},{:.6},{:.6},{:.6},{:.6},{:.6},{:.6},{},{:.6},{:.6}", r.kind, r.id, r.ns, r.costs[0], r.costs[1], r.costs[2], r.costs[3], r.costs[4], r.costs[5], r.generated, r.age, r.compute)?;
        }
        f.flush()
    })();
    match result { Ok(()) => crate::info!("gameplay presentation samples: {}", path.display()), Err(e) => crate::warn!("gameplay presentation samples: {e}") }
}
