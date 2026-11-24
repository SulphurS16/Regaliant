//! Where threads run (`tune placement` 1, diagnostics): the logical CPU at
//! sampled moments of the command processors (every 16th packet) and of the
//! title's job workers (every 16th job they start), by physical core, and
//! how often each sampled thread's core was shared at that moment: another
//! sampled thread's latest sample, taken within SHARED_US, was on the same
//! physical core (SMT siblings) or the same logical CPU.
//!
//! The compute queues' command processors and the title's threads share the
//! title's CCD (`tune acb_cpus` 1, `guest_cpus`); from boot to boot the
//! compute queues ran at one of two speeds with the same masks (FINDINGS.md
//! 2026-10-26): this shows how the threads fell on the cores.

use parking_lot::Mutex;
use std::cell::Cell;
use std::sync::atomic::{AtomicU64, Ordering};

const MAX_THREADS: usize = 64;
const MAX_CPUS: usize = 64;
const SAMPLE_EVERY: u32 = 16;
/// A sample this recent counts as the thread running now.
const SHARED_US: u64 = 100;

/// Per registered thread: its name.
static NAMES: Mutex<Vec<String>> = Mutex::new(Vec::new());
/// Per thread and logical CPU: samples.
static COUNTS: [[AtomicU64; MAX_CPUS]; MAX_THREADS] = [const { [const { AtomicU64::new(0) }; MAX_CPUS] }; MAX_THREADS];
/// Per thread: its latest sample (logical CPU + 1 in the top byte, TSC in
/// the rest).
static LAST: [AtomicU64; MAX_THREADS] = [const { AtomicU64::new(0) }; MAX_THREADS];
/// Per thread: [samples, samples with its core shared].
static SHARED: [[AtomicU64; 2]; MAX_THREADS] = [const { [const { AtomicU64::new(0) }; 2] }; MAX_THREADS];
/// Registered threads.
static USED: AtomicU64 = AtomicU64::new(0);

thread_local! {
    static SLOT: Cell<u32> = const { Cell::new(u32::MAX) };
    static TICK: Cell<u32> = const { Cell::new(0) };
}

const TSC_MASK: u64 = (1 << 56) - 1;

fn register(name: String) -> u32 {
    let mut g = NAMES.lock();
    if g.len() >= MAX_THREADS {
        return MAX_THREADS as u32;
    }
    g.push(name);
    USED.store(g.len() as u64, Ordering::Release);
    (g.len() - 1) as u32
}

/// A sample point of the calling thread (`name` names it at its first).
#[inline]
pub fn note(name: &dyn Fn() -> String) {
    if !crate::gpu::tune::placement() {
        return;
    }
    let t = TICK.with(|c| {
        let v = c.get().wrapping_add(1);
        c.set(v);
        v
    });
    if t % SAMPLE_EVERY != 0 {
        return;
    }
    sample(name);
}

#[cold]
fn sample(name: &dyn Fn() -> String) {
    let slot = SLOT.with(|c| {
        if c.get() == u32::MAX {
            c.set(register(name()));
        }
        c.get()
    }) as usize;
    if slot >= MAX_THREADS {
        return;
    }
    let cpu = (unsafe { crate::platform::os::Win32::System::Threading::GetCurrentProcessorNumber() } as usize).min(MAX_CPUS - 1);
    let now = unsafe { core::arch::x86_64::_rdtsc() } & TSC_MASK;
    COUNTS[slot][cpu].fetch_add(1, Ordering::Relaxed);
    LAST[slot].store(((cpu as u64 + 1) << 56) | now, Ordering::Relaxed);
    let window = crate::hle::kernel::time::tsc_hz() / 1_000_000 * SHARED_US;
    let core = crate::gpu::prio::core_of(cpu);
    let n = (USED.load(Ordering::Acquire) as usize).min(MAX_THREADS);
    let shared = (0..n).any(|i| {
        if i == slot {
            return false;
        }
        let v = LAST[i].load(Ordering::Relaxed);
        let c = (v >> 56) as usize;
        c != 0 && now.wrapping_sub(v & TSC_MASK) & TSC_MASK < window && crate::gpu::prio::core_of(c - 1) == core
    });
    SHARED[slot][0].fetch_add(1, Ordering::Relaxed);
    if shared {
        SHARED[slot][1].fetch_add(1, Ordering::Relaxed);
    }
}

/// Status line ("" when off or nothing sampled).
pub fn status() -> String {
    if !crate::gpu::tune::placement() {
        return String::new();
    }
    let names = NAMES.lock().clone();
    if names.is_empty() {
        return String::new();
    }
    // Groups: each compute queue and the graphics queue's command processor
    // alone, the parallel workers together, the job workers together.
    let group = |n: &str| -> String {
        if n.starts_with("gpu-gfx-par") {
            "gpu-gfx-par*".into()
        } else if n.starts_with("BPE JobWorkerThread") {
            "job workers".into()
        } else {
            n.to_string()
        }
    };
    let mut groups: Vec<(String, Vec<usize>)> = Vec::new();
    for (i, n) in names.iter().enumerate() {
        let g = group(n);
        match groups.iter_mut().find(|x| x.0 == g) {
            Some(x) => x.1.push(i),
            None => groups.push((g, vec![i])),
        }
    }
    groups.sort_by(|a, b| a.0.cmp(&b.0));
    let mut out = String::from("    placement (`tune placement`): samples by physical core (its logical CPUs), and how often another sampled thread ran on the same core then:\n");
    for (g, slots) in groups {
        let mut by_core: std::collections::BTreeMap<usize, (u64, u64)> = std::collections::BTreeMap::new();
        let mut total = 0u64;
        for &s in &slots {
            for cpu in 0..MAX_CPUS {
                let k = COUNTS[s][cpu].load(Ordering::Relaxed);
                if k > 0 {
                    let core = crate::gpu::prio::core_of(cpu);
                    let e = by_core.entry(core).or_insert((0, crate::gpu::prio::core_mask(core)));
                    e.0 += k;
                    total += k;
                }
            }
        }
        if total == 0 {
            continue;
        }
        let (n, sh): (u64, u64) = slots.iter().fold((0, 0), |a, &s| (a.0 + SHARED[s][0].load(Ordering::Relaxed), a.1 + SHARED[s][1].load(Ordering::Relaxed)));
        let mut cores: Vec<(usize, u64, u64)> = by_core.into_iter().map(|(c, (k, m))| (c, k, m)).collect();
        cores.sort_by(|a, b| b.1.cmp(&a.1));
        let parts: Vec<String> = cores.iter().take(8).map(|&(c, k, m)| format!("core {c} ({m:#x}) {:.0}%", 100.0 * k as f64 / total as f64)).collect();
        out += &format!("      {g}: {} samples; shared {:.0}%; {}\n", total, 100.0 * sh as f64 / n.max(1) as f64, parts.join(", "));
    }
    out
}
