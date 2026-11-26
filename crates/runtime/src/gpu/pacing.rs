//! Frame pacing: the intervals between presented flips (host time at the
//! vblank that shows each frame), for the `frametimes` control command
//! (statistics since the last one; flips/s alone hide stalls and uneven
//! frames), with the window's hitches by cause (`gpu::hitch`) and other
//! processes' load on the machine meanwhile (`hostload`).

use parking_lot::Mutex;

/// The vblank period the intervals are counted in (60 Hz).
const VBLANK_US: f64 = 1_000_000.0 / 60.0;

struct State {
    /// The last flip's time (us) and number.
    last: Option<(u64, u64)>,
    /// Intervals since the window began: (us, number of the flip ending it).
    intervals: Vec<(u32, u64)>,
}

static STATE: Mutex<State> = Mutex::new(State { last: None, intervals: Vec::new() });

/// Flip `n` was presented at `us` (process time).
pub fn note_flip(n: u64, us: u64) {
    let mut s = STATE.lock();
    if let Some((t, _)) = s.last {
        let d = us.saturating_sub(t).min(u32::MAX as u64) as u32;
        if s.intervals.len() < 1 << 20 {
            s.intervals.push((d, n));
        }
    }
    s.last = Some((us, n));
    drop(s);
    super::hitch::flip(n, us);
    super::fe2::census_sample();
}

/// The window's statistics (one line), and a new window.
pub fn report_and_reset() -> String {
    let iv = std::mem::take(&mut STATE.lock().intervals);
    let hitches = super::hitch::window_report_and_reset();
    let gpu = super::hitch::gpu_window_clause();
    let host = crate::hostload::window_report_and_reset() + &crate::displayclock::window_clause();
    let cpu = super::threadcpu::frames_window_clause(iv.len() as u64) + &super::threadcpu::frames_detail_clause(iv.len() as u64) + &super::exec::title_barriers_clause(iv.len() as u64) + &super::cp::past_wait_window_clause(iv.len() as u64);
    if iv.is_empty() {
        return "no flips".into();
    }
    let mut ms: Vec<f64> = iv.iter().map(|&(d, _)| d as f64 / 1000.0).collect();
    ms.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let n = ms.len();
    let pct = |p: f64| ms[((p / 100.0 * n as f64).ceil() as usize).clamp(1, n) - 1];
    let total: f64 = ms.iter().sum();
    let median = pct(50.0);
    // Stalls: intervals over twice the median.
    let stalls: Vec<f64> = ms.iter().copied().filter(|&x| x > 2.0 * median).collect();
    // Intervals by vblanks shown (1 = 60 fps).
    let mut by = [0u64; 9];
    for &(d, _) in &iv {
        let v = ((d as f64 / VBLANK_US).round() as usize).clamp(1, 9);
        by[v - 1] += 1;
    }
    let by: Vec<String> = by.iter().enumerate().filter(|&(_, &c)| c > 0).map(|(i, c)| format!("{}{}:{c}", i + 1, if i == 8 { "+" } else { "" })).collect();
    let mut worst = iv.clone();
    worst.sort_by(|a, b| b.0.cmp(&a.0));
    let worst: Vec<String> = worst.iter().take(5).map(|&(d, f)| format!("{:.0} ms (flip {f})", d as f64 / 1000.0)).collect();
    format!(
        "{n} intervals, {:.2} flips/s; ms mean {:.1}, p50 {:.1}, p90 {:.1}, p99 {:.1}, max {:.1}; by vblanks {}; stalls (>2x median) {} ({:.0} ms); worst {}; {hitches}{gpu}{cpu}{host}",
        n as f64 * 1000.0 / total,
        total / n as f64,
        median,
        pct(90.0),
        pct(99.0),
        ms[n - 1],
        by.join(" "),
        stalls.len(),
        stalls.iter().sum::<f64>(),
        worst.join(", ")
    )
}

/// Start/end an isolated relay interval, excluding the gap before its first flip.
#[cfg(target_os = "linux")]
pub fn take_intervals() -> Vec<(u32, u64)> {
    let mut s = STATE.lock();
    s.last = None;
    std::mem::take(&mut s.intervals)
}
