//! Process-local thread accounting, sampled outside signal context.
use std::collections::HashMap;

pub fn thread_times() -> HashMap<u32, (u64, u64, u64)> {
    let mut out = HashMap::new();
    let ticks = unsafe { libc::sysconf(libc::_SC_CLK_TCK) };
    if ticks <= 0 {
        return out;
    }
    let Ok(tasks) = std::fs::read_dir("/proc/self/task") else {
        return out;
    };
    for task in tasks.flatten() {
        let Ok(tid) = task.file_name().to_string_lossy().parse() else {
            continue;
        };
        let Ok(stat) = std::fs::read_to_string(task.path().join("stat")) else {
            continue;
        };
        // The parenthesized comm can contain spaces and closing parentheses.
        let Some((_, rest)) = stat.rsplit_once(')') else {
            continue;
        };
        let fields: Vec<_> = rest.split_whitespace().collect();
        let number = |i: usize| {
            fields
                .get(i)
                .and_then(|s| s.parse::<u64>().ok())
                .unwrap_or(0)
        };
        let switches = std::fs::read_to_string(task.path().join("status"))
            .ok()
            .map(|s| {
                s.lines()
                    .filter(|l| {
                        l.starts_with("voluntary_ctxt_switches:")
                            || l.starts_with("nonvoluntary_ctxt_switches:")
                    })
                    .filter_map(|l| l.split_whitespace().last()?.parse::<u64>().ok())
                    .sum()
            })
            .unwrap_or(0);
        // Fields 14/15: user/system ticks, converted to the shared profiler's 100 ns units.
        out.insert(
            tid,
            (
                number(12) * 10_000_000 / ticks as u64,
                number(11) * 10_000_000 / ticks as u64,
                switches,
            ),
        );
    }
    out
}

/// Scheduled CPU time of this thread, for sampled driver-call diagnostics.
pub fn own_ns() -> u64 {
    let mut t: libc::timespec = unsafe { std::mem::zeroed() };
    if unsafe { libc::clock_gettime(libc::CLOCK_THREAD_CPUTIME_ID, &mut t) } != 0 { return 0; }
    t.tv_sec as u64 * 1_000_000_000 + t.tv_nsec as u64
}
