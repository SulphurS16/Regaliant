//! Other processes' load on this machine, sampled once a second from the
//! Windows performance counters: the GPU's 3D engine use and dedicated video
//! memory per process, CPU time per process. Every frame window
//! (`gpu::pacing`) carries the means of its samples, and a line is logged
//! when another program starts or stops loading the machine, so a
//! measurement taken while something else shared the GPU or the CPUs says so
//! (FINDINGS.md 2026-10-23: another port's sessions shared the GPU during
//! the frame-time walks). `DS_HOST_LOAD=0` leaves it off.

use parking_lot::Mutex;
use std::collections::HashMap;

/// Sums over a window's samples.
#[derive(Default)]
struct Window {
    samples: u32,
    /// 3D engine utilization (percent) of this process and of all others.
    gpu_ours: f64,
    gpu_other: f64,
    /// Dedicated video memory (bytes) of this process and of all others.
    vram_ours: f64,
    vram_other: f64,
    /// CPU time of all other processes (percent of one logical CPU).
    cpu_other: f64,
    /// Per other process name: 3D engine percent and CPU percent.
    gpu_by: HashMap<String, f64>,
    cpu_by: HashMap<String, f64>,
    /// Time spent collecting (us).
    cost_us: u64,
}

static WINDOW: Mutex<Option<Window>> = Mutex::new(None);

/// The last sample's GPU figures for the performance overlay (`perfhud`): the 3D engine use of
/// the card in use summed over every process (percent), this process's and the others' dedicated
/// video memory on it (bytes), and when it was taken.
#[derive(Clone, Copy)]
pub struct Latest {
    pub gpu_3d: f64,
    pub vram_ours: f64,
    pub vram_other: f64,
    pub at: std::time::Instant,
}

static LATEST: Mutex<Option<Latest>> = Mutex::new(None);

/// The last sample (once a second; None while the sampler is off, or before the card is known).
pub fn latest() -> Option<Latest> {
    *LATEST.lock()
}

/// Other processes' dedicated video memory at the last sample, bytes
/// (u64::MAX before the first).
static OTHERS_VRAM: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(u64::MAX);

/// The card in use as the GPU counters name it (`luid_0x<high>_0x<low>`, upper-case hex), set
/// when the device is chosen (`gpu::vk`). Video memory is counted on that card only: a PC with
/// an integrated GPU beside it lists both, and the sum over both read 40-68 GB of others' memory
/// on a 12 GB card (a Discord report, 2026-11-25; issue #5 had 38 GB). Until it is known, no
/// video memory is counted.
static ADAPTER: Mutex<Option<String>> = Mutex::new(None);

/// The card's adapter id (Vulkan's deviceLUID: the Windows LUID's low part, then its high part).
pub fn set_adapter_luid(luid: [u8; 8]) {
    let low = u32::from_le_bytes([luid[0], luid[1], luid[2], luid[3]]);
    let high = u32::from_le_bytes([luid[4], luid[5], luid[6], luid[7]]);
    let tag = format!("luid_0x{high:08X}_0x{low:08X}");
    crate::info!("host load: video memory counted on the card in use only ({tag})");
    *ADAPTER.lock() = Some(tag);
}

/// Other processes' dedicated video memory at the last sample (once a
/// second), bytes; None while the sampler is off or has not sampled yet.
pub fn others_vram() -> Option<u64> {
    // (`DS_SIM_OTHERS_MIB=<n>`: that count instead, a test of the image cache's limit under a
    // count like issue #5's 38 GB on a 12 GB card.)
    static SIM: std::sync::OnceLock<Option<u64>> = std::sync::OnceLock::new();
    if let Some(s) = *SIM.get_or_init(|| std::env::var("DS_SIM_OTHERS_MIB").ok().and_then(|v| v.trim().parse::<u64>().ok()).map(|m| m << 20)) {
        return Some(s);
    }
    let v = OTHERS_VRAM.load(std::sync::atomic::Ordering::Relaxed);
    (v != u64::MAX).then_some(v)
}

/// A process's 3D engine use from which it counts as loading the machine
/// (percent), and its CPU use (percent of one logical CPU).
const LOUD_GPU: f64 = 3.0;
const LOUD_CPU: f64 = 100.0;

/// Starts the sampler unless `DS_HOST_LOAD=0`.
pub fn start() {
    if std::env::var("DS_HOST_LOAD").ok().as_deref() == Some("0") {
        return;
    }
    *WINDOW.lock() = Some(Window::default());
    if let Err(e) = crate::thread::spawn_host("host-load", 1 << 20, Box::new(sampler)) {
        crate::warn!("host load: no sampler thread: {e}");
        *WINDOW.lock() = None;
    }
}

/// The window's means (one clause for the frame-window line), and a new
/// window; empty when the sampler is off.
pub fn window_report_and_reset() -> String {
    let w = match WINDOW.lock().as_mut() {
        Some(w) => std::mem::take(w),
        None => return String::new(),
    };
    if w.samples == 0 {
        return "; host: no samples".into();
    }
    let n = w.samples as f64;
    let top = |m: &HashMap<String, f64>, min: f64| {
        let mut v: Vec<(&String, f64)> = m.iter().map(|(k, &s)| (k, s / n)).filter(|&(_, x)| x >= min).collect();
        v.sort_by(|a, b| b.1.total_cmp(&a.1));
        let s: Vec<String> = v.iter().take(4).map(|(k, x)| format!("{k} {x:.0}")).collect();
        if s.is_empty() { String::new() } else { format!(" ({})", s.join(", ")) }
    };
    format!(
        "; host: others' GPU 3D {:.1}%{}, video memory {:.1} GB; others' CPU {:.0}% of a logical CPU{}; ours GPU 3D {:.0}%, video memory {:.1} GB ({} samples, {:.1} ms each)",
        w.gpu_other / n,
        top(&w.gpu_by, 1.0),
        w.vram_other / n / 1e9,
        w.cpu_other / n,
        top(&w.cpu_by, 25.0),
        w.gpu_ours / n,
        w.vram_ours / n / 1e9,
        w.samples,
        w.cost_us as f64 / n / 1e3,
    )
}

/// The formatted values of a wildcard counter: (instance name, value).
unsafe fn values(c: windows_sys::Win32::System::Performance::PDH_HCOUNTER, format: u32) -> Vec<(String, f64)> {
    use windows_sys::Win32::System::Performance::*;
    let (mut size, mut count) = (0u32, 0u32);
    let r = unsafe { PdhGetFormattedCounterArrayW(c, format, &mut size, &mut count, std::ptr::null_mut()) };
    if r != PDH_MORE_DATA || size == 0 {
        return Vec::new();
    }
    // u64 elements: the items hold doubles and pointers.
    let mut buf = vec![0u64; (size as usize).div_ceil(8)];
    let items = buf.as_mut_ptr() as *mut PDH_FMT_COUNTERVALUE_ITEM_W;
    if unsafe { PdhGetFormattedCounterArrayW(c, format, &mut size, &mut count, items) } != 0 {
        return Vec::new();
    }
    let mut out = Vec::with_capacity(count as usize);
    for i in 0..count as usize {
        let it = unsafe { &*items.add(i) };
        if it.FmtValue.CStatus != PDH_CSTATUS_VALID_DATA && it.FmtValue.CStatus != PDH_CSTATUS_NEW_DATA {
            continue;
        }
        let mut len = 0;
        while unsafe { *it.szName.add(len) } != 0 {
            len += 1;
        }
        let name = String::from_utf16_lossy(unsafe { std::slice::from_raw_parts(it.szName, len) });
        out.push((name, unsafe { it.FmtValue.Anonymous.doubleValue }));
    }
    out
}

/// The process id in a GPU counter instance name (`pid_1234_luid_...`).
fn pid_of(instance: &str) -> Option<u32> {
    instance.strip_prefix("pid_")?.split('_').next()?.parse().ok()
}

/// A process's executable name without its extension (from a process snapshot when it cannot be
/// opened, `pid <n>` when neither names it).
fn name_of(pid: u32) -> String {
    use windows_sys::Win32::Foundation::CloseHandle;
    use windows_sys::Win32::System::Threading::{OpenProcess, QueryFullProcessImageNameW, PROCESS_NAME_WIN32, PROCESS_QUERY_LIMITED_INFORMATION};
    let strip = |file: &str| file.strip_suffix(".exe").or_else(|| file.strip_suffix(".EXE")).unwrap_or(file).to_string();
    unsafe {
        let h = OpenProcess(PROCESS_QUERY_LIMITED_INFORMATION, 0, pid);
        if h.is_null() {
            return snapshot_name(pid).map(|n| strip(&n)).unwrap_or_else(|| format!("pid {pid}"));
        }
        let mut buf = [0u16; 520];
        let mut len = buf.len() as u32;
        let ok = QueryFullProcessImageNameW(h, PROCESS_NAME_WIN32, buf.as_mut_ptr(), &mut len) != 0;
        CloseHandle(h);
        if !ok {
            return snapshot_name(pid).map(|n| strip(&n)).unwrap_or_else(|| format!("pid {pid}"));
        }
        let path = String::from_utf16_lossy(&buf[..len as usize]);
        strip(path.rsplit(['\\', '/']).next().unwrap_or(&path))
    }
}

/// A process's executable file name from a process snapshot: protected processes (dwm.exe among
/// them, 4.3 GB of dedicated video memory here on 2026-11-24) cannot be opened for their image name.
fn snapshot_name(pid: u32) -> Option<String> {
    use windows_sys::Win32::Foundation::{CloseHandle, INVALID_HANDLE_VALUE};
    use windows_sys::Win32::System::Diagnostics::ToolHelp::{CreateToolhelp32Snapshot, Process32FirstW, Process32NextW, PROCESSENTRY32W, TH32CS_SNAPPROCESS};
    unsafe {
        let snap = CreateToolhelp32Snapshot(TH32CS_SNAPPROCESS, 0);
        if snap == INVALID_HANDLE_VALUE {
            return None;
        }
        let mut e: PROCESSENTRY32W = std::mem::zeroed();
        e.dwSize = std::mem::size_of::<PROCESSENTRY32W>() as u32;
        let mut found = None;
        let mut ok = Process32FirstW(snap, &mut e) != 0;
        while ok {
            if e.th32ProcessID == pid {
                let n = e.szExeFile.iter().position(|&c| c == 0).unwrap_or(e.szExeFile.len());
                found = Some(String::from_utf16_lossy(&e.szExeFile[..n]));
                break;
            }
            ok = Process32NextW(snap, &mut e) != 0;
        }
        CloseHandle(snap);
        found
    }
}

fn sampler() {
    use windows_sys::Win32::System::Performance::*;
    crate::log::set_thread_name("host-load");
    unsafe {
        windows_sys::Win32::System::Threading::SetThreadPriority(windows_sys::Win32::System::Threading::GetCurrentThread(), windows_sys::Win32::System::Threading::THREAD_PRIORITY_BELOW_NORMAL);
    }
    let wide = |s: &str| s.encode_utf16().chain(std::iter::once(0)).collect::<Vec<u16>>();
    let mut q: PDH_HQUERY = std::ptr::null_mut();
    if unsafe { PdhOpenQueryW(std::ptr::null(), 0, &mut q) } != 0 {
        crate::warn!("host load: no counter query");
        return;
    }
    let add = |path: &str| {
        let mut c: PDH_HCOUNTER = std::ptr::null_mut();
        let r = unsafe { PdhAddEnglishCounterW(q, wide(path).as_ptr(), 0, &mut c) };
        if r != 0 {
            crate::warn!("host load: counter {path} unavailable ({r:#x})");
            return None;
        }
        Some(c)
    };
    let gpu = add("\\GPU Engine(*engtype_3D)\\Utilization Percentage");
    let vram = add("\\GPU Process Memory(*)\\Dedicated Usage");
    let cpu = add("\\Process(*)\\% Processor Time");
    // Rates need two collections.
    unsafe { PdhCollectQueryData(q) };
    let me = std::process::id();
    let my_name = name_of(me);
    let mut names: HashMap<u32, String> = HashMap::new();
    // Other processes loading the machine, by name, over the last few
    // samples: (3D percent, CPU percent) sums and the count.
    let mut recent: HashMap<String, (f64, f64)> = HashMap::new();
    let mut recent_n = 0u32;
    let mut loud: Vec<String> = Vec::new();
    crate::info!("host load: other processes' GPU, video memory and CPU use sampled once a second (DS_HOST_LOAD=0: off)");
    loop {
        std::thread::sleep(std::time::Duration::from_secs(1));
        let t0 = std::time::Instant::now();
        if unsafe { PdhCollectQueryData(q) } != 0 {
            continue;
        }
        let (mut gpu_ours, mut gpu_other, mut vram_ours, mut vram_other, mut cpu_other) = (0.0, 0.0, 0.0, 0.0, 0.0);
        let mut gpu_by: HashMap<String, f64> = HashMap::new();
        let mut cpu_by: HashMap<String, f64> = HashMap::new();
        let adapter = ADAPTER.lock().clone();
        let want = adapter.as_deref().map(str::to_ascii_uppercase);
        // (The card in use's 3D engine over every process: the overlay's GPU use where NVIDIA's
        // NVML does not give it.)
        let mut card_3d = 0.0;
        if let Some(c) = gpu {
            for (inst, v) in unsafe { values(c, PDH_FMT_DOUBLE) } {
                if want.as_deref().is_some_and(|w| inst.to_ascii_uppercase().contains(w)) {
                    card_3d += v;
                }
                let Some(pid) = pid_of(&inst) else { continue };
                if pid == me {
                    gpu_ours += v;
                } else if v > 0.0 {
                    gpu_other += v;
                    let name = names.entry(pid).or_insert_with(|| name_of(pid)).clone();
                    *gpu_by.entry(name).or_default() += v;
                }
            }
        }
        if let (Some(c), Some(want)) = (vram, want.as_deref()) {
            let mut vram_by: HashMap<u32, f64> = HashMap::new();
            for (inst, v) in unsafe { values(c, PDH_FMT_DOUBLE) } {
                // (Only the card in use: `ADAPTER`.)
                if !inst.to_ascii_uppercase().contains(want) {
                    continue;
                }
                match pid_of(&inst) {
                    Some(pid) if pid == me => vram_ours += v,
                    Some(pid) => {
                        vram_other += v;
                        *vram_by.entry(pid).or_default() += v;
                    }
                    None => {}
                }
            }
            OTHERS_VRAM.store(vram_other as u64, std::sync::atomic::Ordering::Relaxed);
            // (Each time others' dedicated memory passes another 4 GiB: who holds it. A player's
            // 12 GB card had others at 11.7 GB, so the image cache kept to half its budget,
            // without the log naming them: issue #5, 2026-11-24.)
            // (`DS_HOST_VRAM_REPORT_MIB=<n>`: that step instead, for a test.)
            static STEP: std::sync::OnceLock<u64> = std::sync::OnceLock::new();
            let step = *STEP.get_or_init(|| {
                std::env::var("DS_HOST_VRAM_REPORT_MIB").ok().and_then(|v| v.trim().parse::<u64>().ok()).and_then(|m| m.checked_mul(1 << 20)).filter(|&s| s > 0).unwrap_or(4 << 30)
            });
            static NEXT_VRAM_REPORT: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);
            let other = vram_other as u64;
            if other >= NEXT_VRAM_REPORT.load(std::sync::atomic::Ordering::Relaxed).max(step) {
                NEXT_VRAM_REPORT.store((other / step).saturating_add(1).saturating_mul(step), std::sync::atomic::Ordering::Relaxed);
                let mut by_name: HashMap<String, f64> = HashMap::new();
                for (pid, v) in vram_by {
                    let name = names.entry(pid).or_insert_with(|| name_of(pid)).clone();
                    *by_name.entry(name).or_default() += v;
                }
                let mut top: Vec<(String, f64)> = by_name.into_iter().collect();
                top.sort_by(|a, b| b.1.total_cmp(&a.1));
                let list: Vec<String> = top.iter().take(5).map(|(n, v)| format!("{n} {:.1} GB", v / (1u64 << 30) as f64)).collect();
                crate::info!(
                    "host load: other processes' dedicated video memory {:.1} GB as Windows counts it (ours {:.1} GB); the most: {}",
                    other as f64 / (1u64 << 30) as f64,
                    vram_ours / (1u64 << 30) as f64,
                    list.join(", ")
                );
            }
        }
        if let Some(c) = cpu {
            // PDH_FMT_NOCAP100: a process's share over 100% of one CPU.
            for (inst, v) in unsafe { values(c, PDH_FMT_DOUBLE | 0x8000) } {
                let name = inst.split('#').next().unwrap_or(&inst);
                if name == "_Total" || name == "Idle" || name.eq_ignore_ascii_case(&my_name) {
                    continue;
                }
                cpu_other += v;
                *cpu_by.entry(name.to_string()).or_default() += v;
            }
        }
        if want.is_some() {
            *LATEST.lock() = Some(Latest { gpu_3d: card_3d.min(100.0), vram_ours, vram_other, at: std::time::Instant::now() });
        }
        let cost = t0.elapsed().as_micros() as u64;
        // Who loads the machine now (means over five samples).
        for (k, v) in &gpu_by {
            recent.entry(k.clone()).or_default().0 += v;
        }
        for (k, v) in &cpu_by {
            if *v >= 10.0 {
                recent.entry(k.clone()).or_default().1 += v;
            }
        }
        recent_n += 1;
        if recent_n == 5 {
            let n = recent_n as f64;
            let mut now: Vec<(String, f64, f64)> = recent.iter().map(|(k, &(g, c))| (k.clone(), g / n, c / n)).filter(|&(_, g, c)| g >= LOUD_GPU || c >= LOUD_CPU).collect();
            now.sort_by(|a, b| a.0.cmp(&b.0));
            let keys: Vec<String> = now.iter().map(|x| x.0.clone()).collect();
            if keys != loud {
                let s: Vec<String> = now.iter().map(|(k, g, c)| format!("{k} (GPU 3D {g:.0}%, CPU {c:.0}%)")).collect();
                crate::info!("host load: other processes loading the machine now: {}", if s.is_empty() { "none".to_string() } else { s.join(", ") });
                loud = keys;
            }
            recent.clear();
            recent_n = 0;
        }
        if names.len() > 4096 {
            names.clear();
        }
        let mut g = WINDOW.lock();
        let Some(w) = g.as_mut() else { return };
        w.samples += 1;
        w.gpu_ours += gpu_ours;
        w.gpu_other += gpu_other;
        w.vram_ours += vram_ours;
        w.vram_other += vram_other;
        w.cpu_other += cpu_other;
        for (k, v) in gpu_by {
            *w.gpu_by.entry(k).or_default() += v;
        }
        for (k, v) in cpu_by {
            *w.cpu_by.entry(k).or_default() += v;
        }
        w.cost_us += cost;
    }
}
