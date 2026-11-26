//! Boundary-only accounting for the native-resolution relay. No GPU queries, readbacks,
//! per-draw counters or /proc reads in its timed loop.
use std::collections::BTreeMap;
use std::sync::atomic::Ordering::Relaxed;
use std::time::Instant;
use serde_json::{Value, json};

fn counters() -> BTreeMap<&'static str, u64> {
    let [upload, writeback, runs] = crate::gpu::fe2::traffic_counters();
    let ww = super::write_watch::counters();
    let fault = crate::gpu::resident::linux_fault_samples();
    BTreeMap::from([
        ("flips", crate::gpu::FRAMES.load(Relaxed)),
        ("draws", crate::gpu::cp::DRAWS.load(Relaxed)),
        ("dispatches", crate::gpu::cp::DISPATCHES.load(Relaxed)),
        ("faults", super::signals::RECOVERED.load(Relaxed)),
        ("direct_faults", super::signals::DIRECT.load(Relaxed)),
        ("fault_samples", fault[0]), ("fault_sample_wall_ns", fault[1]), ("fault_sample_cpu_ns", fault[2]),
        ("kernel_io_retries", crate::mem::KERNEL_IO[1].load(Relaxed)),
        ("upload_bytes", upload), ("writeback_bytes", writeback), ("upload_runs", runs),
        ("uffd_scans", ww[1]), ("uffd_ns", ww[3]), ("uffd_errors", ww[5]),
    ])
}

fn threads() -> BTreeMap<String, Value> {
    let mut result = BTreeMap::new();
    let ticks = unsafe { libc::sysconf(libc::_SC_CLK_TCK) }.max(1) as f64;
    let Ok(tasks) = std::fs::read_dir("/proc/self/task") else { return result };
    for task in tasks.flatten() {
        let p = task.path();
        let Ok(stat) = std::fs::read_to_string(p.join("stat")) else { continue };
        let Some((_, rest)) = stat.rsplit_once(')') else { continue };
        let f: Vec<_> = rest.split_whitespace().collect();
        let number = |i: usize| f.get(i).and_then(|s| s.parse::<u64>().ok()).unwrap_or(0);
        let name = std::fs::read_to_string(p.join("comm")).unwrap_or_default();
        let sched = std::fs::read_to_string(p.join("schedstat")).unwrap_or_default();
        let runqueue_ns = sched.split_whitespace().nth(1).and_then(|s| s.parse::<u64>().ok()).unwrap_or(0);
        result.insert(format!("{}:{}", task.file_name().to_string_lossy(), number(19)),
            json!({"name":name.trim(), "user_s":number(11) as f64/ticks, "system_s":number(12) as f64/ticks, "cpu_s":(number(11)+number(12)) as f64/ticks, "runqueue_s":runqueue_ns as f64/1e9}));
    }
    result
}

fn monotonic_ns() -> u64 {
    let mut t: libc::timespec = unsafe { std::mem::zeroed() };
    unsafe { libc::clock_gettime(libc::CLOCK_MONOTONIC, &mut t); }
    t.tv_sec as u64 * 1_000_000_000 + t.tv_nsec as u64
}

pub struct Phase {
    start: Instant,
    begin_ns: u64,
    counters: BTreeMap<&'static str, u64>,
    threads: BTreeMap<String, Value>,
    jobs: Value,
}
impl Phase {
    pub fn begin() -> Self {
        let threads = threads();
        let jobs = crate::jobidle::relay_snapshot();
        crate::platform::present_timing::begin();
        crate::gpu::pacing::take_intervals();
        Self { start:Instant::now(), begin_ns:monotonic_ns(), counters:counters(), threads, jobs }
    }
    pub fn end(self, label: &str) {
        let end_ns = monotonic_ns();
        let seconds = self.start.elapsed().as_secs_f64();
        let intervals = crate::gpu::pacing::take_intervals();
        let delta: BTreeMap<_, _> = counters().into_iter().map(|(k,v)| (k, v.saturating_sub(self.counters[k]))).collect();
        let threads: Vec<_> = threads().into_iter().filter_map(|(id, now)| {
            let before = self.threads.get(&id)?;
            Some(json!({"id":id, "name":now["name"],
                "user_s":(now["user_s"].as_f64()? - before["user_s"].as_f64()?).max(0.),
                "system_s":(now["system_s"].as_f64()? - before["system_s"].as_f64()?).max(0.),
                "cpu_s":(now["cpu_s"].as_f64()? - before["cpu_s"].as_f64()?).max(0.),
                "runqueue_s":(now["runqueue_s"].as_f64()? - before["runqueue_s"].as_f64()?).max(0.)}))
        }).collect();
        let jobs = crate::jobidle::relay_delta(&self.jobs, &crate::jobidle::relay_snapshot(), seconds, "unchanged");
        crate::platform::present_timing::end(label);
        let data = json!({"phase":label, "seconds":seconds, "begin_monotonic_ns":self.begin_ns, "end_monotonic_ns":end_ns, "counters":delta, "threads":threads,
            "job_workers":jobs, "intervals_us_and_flip":intervals,
            "upscaler":format!("{:?}", crate::gpu::upscale::choice()),
            "note":"CPU and runqueue are per-thread deltas, not additive frame wall time. Threads born/exited inside phase are absent. No GPU busy time inferred from CPU waits."});
        if let Some(dir) = crate::run_dir() {
            if let Err(e) = std::fs::write(dir.join(format!("{label}-steady.json")), data.to_string()) {
                crate::warn!("steady measurement: {e}");
            }
        }
        crate::info!("steady phase {label}: {:.3}s, {} flips; upscaler {:?}", seconds, delta["flips"], crate::gpu::upscale::choice());
    }
}


/// Saved-scene relay readiness, driven by observed UI state rather than old text logs.
/// The caller identifies the offline dialog after seeing its text; a change of menu or
/// HUD ends confirm retries permanently. Never confirm an unknown dialog after loading.
#[derive(Debug, PartialEq, Eq)]
pub enum ResumeAction { Wait, Confirm, Ready }

pub struct ResumeWait {
    dialog: Option<(u64, u64)>,
    left_dialog: bool,
    last_confirm: f64,
    attempts: u32,
    stable: Option<(f64, u64)>,
}
impl ResumeWait {
    pub fn new(dialog: Option<(u64, u64)>) -> Self {
        Self { dialog, left_dialog:false, last_confirm:-8.0, attempts:0, stable:None }
    }
    pub fn poll(&mut self, seconds: f64, flip: u64, menu: Option<u64>, hud: Option<u64>, scene: Option<u64>) -> ResumeAction {
        // Loading can have HUD 304/768 and briefly no menu. Outpost gameplay is 880.
        if menu == Some(0) && hud.is_some_and(|n| n >= 800) {
            let (since, first) = *self.stable.get_or_insert((seconds, flip));
            if seconds - since >= 2.0 && flip.saturating_sub(first) >= 8 { return ResumeAction::Ready; }
        } else { self.stable = None; }
        if let (Some((dialog, before)), Some(depth), Some(bytes)) = (self.dialog, menu, hud) {
            if depth != 1 || bytes != before || scene.is_some_and(|s| s != dialog) { self.left_dialog = true; }
            if !self.left_dialog && scene == Some(dialog) && self.attempts < 6 && seconds - self.last_confirm >= 8.0 {
                self.last_confirm = seconds;
                self.attempts += 1;
                return ResumeAction::Confirm;
            }
        }
        ResumeAction::Wait
    }
}

#[cfg(test)]
mod readiness_tests {
    use super::{ResumeAction::*, ResumeWait};
    #[test]
    fn offline_confirm_retries_are_bounded_and_stop_on_transition() {
        let mut w = ResumeWait::new(Some((123, 288)));
        assert_eq!(w.poll(0., 0, Some(1), Some(288), Some(123)), Confirm);
        assert_eq!(w.poll(1., 40, Some(1), Some(288), Some(123)), Wait);
        assert_eq!(w.poll(8., 320, Some(1), Some(288), Some(123)), Confirm);
        assert_eq!(w.poll(9., 360, Some(0), Some(304), None), Wait);
        assert_eq!(w.poll(20., 800, Some(1), Some(288), Some(123)), Wait);
        let mut w = ResumeWait::new(Some((123, 288)));
        for i in 0..6 { assert_eq!(w.poll(i as f64 * 8., i, Some(1), Some(288), Some(123)), Confirm); }
        assert_eq!(w.poll(60., 100, Some(1), Some(288), Some(123)), Wait);
    }
    #[test]
    fn different_or_unknown_dialog_is_never_confirmed() {
        for (menu, hud, scene) in [(Some(1),Some(288),Some(456)), (Some(2),Some(288),Some(123)), (Some(1),Some(304),Some(123))] {
            let mut w = ResumeWait::new(Some((123,288)));
            assert_eq!(w.poll(0.,0,menu,hud,scene), Wait);
            assert_eq!(w.poll(10.,400,Some(1),Some(288),Some(123)), Wait);
        }
        let mut w = ResumeWait::new(None);
        assert_eq!(w.poll(0.,0,Some(1),Some(288),Some(123)), Wait);
    }
    #[test]
    fn loading_hud_and_frozen_frames_are_not_gameplay() {
        let mut w = ResumeWait::new(None);
        for i in 0..20 { assert_eq!(w.poll(i as f64, i*40, Some(0), Some(768), None), Wait); }
        assert_eq!(w.poll(20.,800,Some(0),Some(880),None), Wait);
        assert_eq!(w.poll(25.,800,Some(0),Some(880),None), Wait);
        assert_eq!(w.poll(26.,801,Some(1),Some(880),Some(456)), Wait);
        assert_eq!(w.poll(27.,840,Some(0),Some(880),None), Wait);
        assert_eq!(w.poll(29.,848,Some(0),Some(880),None), Ready);
    }
}
