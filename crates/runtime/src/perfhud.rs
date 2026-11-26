//! The performance overlay (the PC setting Performance Overlay; players' suggestion of
//! 2026-11-26 evening), in the picture's bottom-right corner (the top-left one hid the health
//! bar, and the top-right the souls):
//! - the frame rate over the last second, and its 1% and 0.1% lows over the last 20 seconds: the
//!   mean of the slowest 1% and 0.1% of the flips' intervals, as a rate (a hitch stays in them
//!   for 20 s);
//! - the CPU use of the game's busiest thread: the share of the time it ran, 100% a whole core
//!   (the frame rate's limit when one thread holds it, which the total over all cores hides).
//!   Players asked for the main thread's, but the title's thread of that name only starts the
//!   game and waits (0.0 ms a flip in every profile): its frame runs in its job workers, and the
//!   port's GPU front end translates it on threads of its own, so the busiest of all of them;
//! - the GPU's use and clock from NVIDIA's NVML (the driver's nvml.dll, from System32); on other
//!   cards the Windows counters' 3D engine use of the card in use, without a clock;
//! - video memory: the game's (Vulkan's memory budget) and the card's whole use (NVML, else the
//!   Windows counters: `hostload`) of its size;
//! - memory: the game's working set and the PC's memory in use.
//!
//! "Frame Rate" shows the first line alone. A sampler thread gathers the figures twice a second
//! and draws the lines with GDI, and the presenter puts the picture over the frame
//! (`gpu::vk::present`). Off, a flip costs one atomic load.

use parking_lot::Mutex;
use std::collections::VecDeque;
use std::sync::atomic::{AtomicU32, AtomicU8, Ordering::Relaxed};
use std::sync::Arc;

/// 0 Off, 1 Frame Rate (the first line), 2 Full.
pub static MODE: AtomicU8 = AtomicU8::new(0);

/// The flips' times (process time, us) over the last `LOWS_US`.
static FLIPS: Mutex<VecDeque<u64>> = Mutex::new(VecDeque::new());
const LOWS_US: u64 = 20_000_000;
const RATE_US: u64 = 1_000_000;

/// The card Vulkan uses: its vendor and device id (NVML's devices are matched by it).
static DEVICE: Mutex<Option<(u32, [u8; 16])>> = Mutex::new(None);

/// The frame height and HDR the presenter last drew for (`height | hdr << 31`; 0 before any).
static FRAME: AtomicU32 = AtomicU32::new(0);

/// The overlay's picture: top-down BGRA rows (alpha 255) of `w` x `h`, drawn for `frame` (as
/// `FRAME`), `generation` new with each drawing.
pub struct Picture {
    pub generation: u64,
    pub frame: u32,
    pub w: u32,
    pub h: u32,
    pub px: Vec<u8>,
}

static PICTURE: Mutex<Option<Arc<Picture>>> = Mutex::new(None);

pub fn set_device(vendor: u32, uuid: [u8; 16]) {
    *DEVICE.lock() = Some((vendor, uuid));
}

/// The setting (`pcsettings::apply`): 0 Off, 1 Frame Rate, 2 Full. The sampler starts at the
/// first time it is on.
pub fn set_mode(mode: u8) {
    let was = MODE.swap(mode.min(2), Relaxed);
    if was == mode {
        return;
    }
    crate::info!("performance overlay: {}", ["off", "the frame rate", "full"][mode.min(2) as usize]);
    if mode == 0 {
        FLIPS.lock().clear();
        *PICTURE.lock() = None;
        return;
    }
    static STARTED: std::sync::Once = std::sync::Once::new();
    STARTED.call_once(|| {
        if let Err(e) = crate::thread::spawn_host("perf-overlay", 1 << 20, Box::new(sampler)) {
            crate::warn!("performance overlay: no sampler thread: {e}");
        }
    });
}

/// A flip shown at `us` (process time: `gpu::pacing::note_flip`).
pub fn note_flip(us: u64) {
    if MODE.load(Relaxed) == 0 {
        return;
    }
    let mut q = FLIPS.lock();
    q.push_back(us);
    while q.front().is_some_and(|&t| t + LOWS_US < us) || q.len() > 40_000 {
        q.pop_front();
    }
}

/// The picture to put over a frame `frame_h` lines high (HDR10 or not), None while the overlay
/// is off or not drawn for this frame yet (the next sample draws it, within half a second).
pub fn picture(frame_h: u32, hdr: bool) -> Option<Arc<Picture>> {
    if MODE.load(Relaxed) == 0 {
        return None;
    }
    let key = frame_h.min(0x7fff_ffff) | (hdr as u32) << 31;
    FRAME.store(key, Relaxed);
    PICTURE.lock().clone().filter(|p| p.frame == key)
}

/// The frame rate over the last second, and the 1% and 0.1% lows over the flips kept.
fn rates(now: u64) -> Option<(f64, f64, f64)> {
    let q = FLIPS.lock();
    if q.len() < 3 {
        return None;
    }
    let mut iv: Vec<u64> = q.iter().zip(q.iter().skip(1)).map(|(a, b)| b.saturating_sub(*a).max(1)).collect();
    // (The intervals ending in the last second; the newest at least.)
    let recent: Vec<u64> = q.iter().skip(1).zip(iv.iter()).filter(|(t, _)| **t + RATE_US >= now).map(|(_, d)| *d).collect();
    let recent = if recent.is_empty() { vec![*iv.last()?] } else { recent };
    let fps = recent.len() as f64 * 1e6 / recent.iter().sum::<u64>() as f64;
    iv.sort_unstable_by(|a, b| b.cmp(a));
    let low = |share: f64| {
        let k = ((iv.len() as f64 * share).ceil() as usize).clamp(1, iv.len());
        k as f64 * 1e6 / iv[..k].iter().sum::<u64>() as f64
    };
    Some((fps, low(0.01), low(0.001)))
}

/// NVIDIA's NVML (nvml.dll, the driver's): the GPU's use, clock and memory. Interface facts from
/// NVIDIA's NVML documentation.
struct Nvml {
    dev: *mut core::ffi::c_void,
    utilization: unsafe extern "C" fn(*mut core::ffi::c_void, *mut [u32; 2]) -> i32,
    clock: unsafe extern "C" fn(*mut core::ffi::c_void, u32, *mut u32) -> i32,
    memory: unsafe extern "C" fn(*mut core::ffi::c_void, *mut [u64; 3]) -> i32,
}

unsafe impl Send for Nvml {}

impl Nvml {
    /// NVML on the card Vulkan uses (matched by its device id, or the only NVIDIA card); None on
    /// other cards, or without the library.
    fn open() -> Option<Nvml> {
        use windows_sys::Win32::System::LibraryLoader::{GetProcAddress, LoadLibraryExW, LOAD_LIBRARY_SEARCH_SYSTEM32, LOAD_WITH_ALTERED_SEARCH_PATH};
        let (vendor, uuid) = (*DEVICE.lock())?;
        if vendor != 0x10de {
            return None;
        }
        let wide = |s: &str| s.encode_utf16().chain(std::iter::once(0)).collect::<Vec<u16>>();
        unsafe {
            // (System32 since NVIDIA's drivers of 2019; before them in NVSMI.)
            let mut lib = LoadLibraryExW(wide("nvml.dll").as_ptr(), std::ptr::null_mut(), LOAD_LIBRARY_SEARCH_SYSTEM32);
            if lib.is_null() {
                if let Some(pf) = std::env::var_os("ProgramW6432").or_else(|| std::env::var_os("ProgramFiles")) {
                    let path = std::path::Path::new(&pf).join("NVIDIA Corporation").join("NVSMI").join("nvml.dll");
                    lib = LoadLibraryExW(wide(&path.to_string_lossy()).as_ptr(), std::ptr::null_mut(), LOAD_WITH_ALTERED_SEARCH_PATH);
                }
            }
            if lib.is_null() {
                crate::info!("performance overlay: no NVML (nvml.dll): the GPU's use from Windows' counters, no clock");
                return None;
            }
            let sym = |name: &str| GetProcAddress(lib, std::ffi::CString::new(name).ok()?.as_ptr() as *const u8);
            let init: unsafe extern "C" fn() -> i32 = std::mem::transmute(sym("nvmlInit_v2")?);
            let count: unsafe extern "C" fn(*mut u32) -> i32 = std::mem::transmute(sym("nvmlDeviceGetCount_v2")?);
            let by_index: unsafe extern "C" fn(u32, *mut *mut core::ffi::c_void) -> i32 = std::mem::transmute(sym("nvmlDeviceGetHandleByIndex_v2")?);
            let get_uuid: unsafe extern "C" fn(*mut core::ffi::c_void, *mut u8, u32) -> i32 = std::mem::transmute(sym("nvmlDeviceGetUUID")?);
            let utilization = std::mem::transmute(sym("nvmlDeviceGetUtilizationRates")?);
            let clock = std::mem::transmute(sym("nvmlDeviceGetClockInfo")?);
            let memory = std::mem::transmute(sym("nvmlDeviceGetMemoryInfo")?);
            let r = init();
            if r != 0 {
                crate::info!("performance overlay: NVML did not start ({r}): the GPU's use from Windows' counters, no clock");
                return None;
            }
            let mut n = 0u32;
            if count(&mut n) != 0 || n == 0 {
                return None;
            }
            let mut devs = Vec::new();
            for i in 0..n {
                let mut d = std::ptr::null_mut();
                if by_index(i, &mut d) != 0 {
                    continue;
                }
                let mut buf = [0u8; 96];
                let id = if get_uuid(d, buf.as_mut_ptr(), buf.len() as u32) == 0 { parse_uuid(&buf) } else { None };
                devs.push((d, id));
            }
            // (The card by its id; one NVIDIA card without one is that card.)
            let dev = devs.iter().find(|(_, id)| *id == Some(uuid)).or_else(|| (devs.len() == 1).then(|| &devs[0])).map(|x| x.0)?;
            crate::info!("performance overlay: NVML on the card in use ({} NVIDIA card(s))", devs.len());
            Some(Nvml { dev, utilization, clock, memory })
        }
    }

    /// (Use in percent, the graphics clock in MHz, memory used and its total in bytes.)
    fn read(&self) -> (Option<u32>, Option<u32>, Option<(u64, u64)>) {
        unsafe {
            let mut u = [0u32; 2];
            let use_ = ((self.utilization)(self.dev, &mut u) == 0).then_some(u[0]);
            let mut mhz = 0u32;
            // (NVML_CLOCK_GRAPHICS.)
            let clock = ((self.clock)(self.dev, 0, &mut mhz) == 0).then_some(mhz);
            let mut m = [0u64; 3];
            let mem = ((self.memory)(self.dev, &mut m) == 0).then_some((m[2], m[0]));
            (use_, clock, mem)
        }
    }
}

/// NVML's device id text ("GPU-xxxxxxxx-xxxx-xxxx-xxxx-xxxxxxxxxxxx") as its 16 bytes.
fn parse_uuid(buf: &[u8]) -> Option<[u8; 16]> {
    let end = buf.iter().position(|&b| b == 0).unwrap_or(buf.len());
    let text = std::str::from_utf8(&buf[..end]).ok()?;
    let hex: Vec<u8> = text.strip_prefix("GPU-").unwrap_or(text).bytes().filter(|b| b.is_ascii_hexdigit()).collect();
    if hex.len() != 32 {
        return None;
    }
    let mut out = [0u8; 16];
    for (i, pair) in hex.chunks(2).enumerate() {
        out[i] = u8::from_str_radix(std::str::from_utf8(pair).ok()?, 16).ok()?;
    }
    Some(out)
}

/// The game's working set and the PC's memory in use and total, bytes.
fn memory() -> (Option<u64>, Option<(u64, u64)>) {
    use windows_sys::Win32::System::ProcessStatus::{GetProcessMemoryInfo, PROCESS_MEMORY_COUNTERS};
    use windows_sys::Win32::System::SystemInformation::{GlobalMemoryStatusEx, MEMORYSTATUSEX};
    use windows_sys::Win32::System::Threading::GetCurrentProcess;
    unsafe {
        let mut c: PROCESS_MEMORY_COUNTERS = std::mem::zeroed();
        c.cb = std::mem::size_of::<PROCESS_MEMORY_COUNTERS>() as u32;
        let ours = (GetProcessMemoryInfo(GetCurrentProcess(), &mut c, c.cb) != 0).then_some(c.WorkingSetSize as u64);
        let mut m: MEMORYSTATUSEX = std::mem::zeroed();
        m.dwLength = std::mem::size_of::<MEMORYSTATUSEX>() as u32;
        let all = (GlobalMemoryStatusEx(&mut m) != 0).then_some((m.ullTotalPhys.saturating_sub(m.ullAvailPhys), m.ullTotalPhys));
        (ours, all)
    }
}

/// A line's pieces: the text and whether it is a value (bright) or a label (dim).
type Line = Vec<(String, bool)>;

fn gb(bytes: u64) -> String {
    format!("{:.1} GB", bytes as f64 / (1u64 << 30) as f64)
}

/// The sampler's state between samples: NVML once opened, the threads' last cycle counts.
struct Sampler {
    nvml: Option<Option<Nvml>>,
    threads: Option<(std::collections::HashMap<u32, u64>, std::time::Instant)>,
}

impl Sampler {
    fn lines(&mut self, mode: u8) -> Vec<Line> {
        let now = crate::hle::kernel::time::process_time_us();
        let v = |s: String| (s, true);
        let l = |s: &str| (s.to_string(), false);
        let mut out = Vec::new();
        out.push(match rates(now) {
            Some((fps, low1, low01)) => vec![l("FPS "), v(format!("{fps:.0}")), l("   1% low "), v(format!("{low1:.0}")), l("   0.1% low "), v(format!("{low01:.0}"))],
            None => vec![l("FPS "), v("-".into())],
        });
        if mode < 2 {
            self.threads = None;
            return out;
        }
        // The busiest thread: the most cycles any thread ran since the last sample.
        let at = std::time::Instant::now();
        let now_cycles: std::collections::HashMap<u32, u64> = crate::gpu::threadcpu::known_thread_cycles().into_iter().collect();
        let busiest = self.threads.as_ref().and_then(|(before, t0)| {
            let secs = at.duration_since(*t0).as_secs_f64().max(1e-3);
            let most = now_cycles.iter().filter_map(|(tid, c)| Some(c.saturating_sub(*before.get(tid)?))).max()?;
            Some((crate::gpu::threadcpu::ms(most) / 1000.0 / secs * 100.0).min(100.0))
        });
        self.threads = Some((now_cycles, at));
        out.push(vec![l("CPU busiest thread "), v(busiest.map_or("-".into(), |p| format!("{p:.0}%")))]);
        let nv = self.nvml.get_or_insert_with(Nvml::open).as_ref().map(|n| n.read());
        // (Windows' counters, sampled once a second: a sample older than 3 s is not shown.)
        let host = crate::hostload::latest().filter(|h| h.at.elapsed() < std::time::Duration::from_secs(3));
        let gpu_use = nv.and_then(|n| n.0).map(|u| u as f64).or(host.map(|h| h.gpu_3d));
        let mut gpu = vec![l("GPU "), v(gpu_use.map_or("-".into(), |u| format!("{u:.0}%")))];
        if let Some(mhz) = nv.and_then(|n| n.1) {
            gpu.push(l("   "));
            gpu.push(v(format!("{mhz} MHz")));
        }
        out.push(gpu);
        let local = crate::gpu::vk::try_ctx().and_then(|c| c.local_heap_use());
        let ours = local.map(|x| x.0).or(host.map(|h| h.vram_ours as u64));
        let all = nv.and_then(|n| n.2).or_else(|| Some(((host?.vram_ours + host?.vram_other) as u64, local?.1)));
        let mut vram = vec![l("VRAM "), v(ours.map_or("-".into(), gb))];
        if let Some((used, total)) = all {
            vram.push(l("   all "));
            vram.push(v(format!("{} / {}", gb(used).trim_end_matches(" GB"), gb(total))));
        }
        out.push(vram);
        let (ours, all) = memory();
        let mut ram = vec![l("RAM "), v(ours.map_or("-".into(), gb))];
        if let Some((used, total)) = all {
            ram.push(l("   all "));
            ram.push(v(format!("{} / {}", gb(used).trim_end_matches(" GB"), gb(total))));
        }
        out.push(ram);
        out
    }
}

fn sampler() {
    crate::log::set_thread_name("perf-overlay");
    unsafe {
        use windows_sys::Win32::System::Threading::{GetCurrentThread, SetThreadPriority, THREAD_PRIORITY_BELOW_NORMAL};
        SetThreadPriority(GetCurrentThread(), THREAD_PRIORITY_BELOW_NORMAL);
    }
    let mut s = Sampler { nvml: None, threads: None };
    let mut generation = 0u64;
    let mut last: Option<(Vec<Line>, u32)> = None;
    loop {
        std::thread::sleep(std::time::Duration::from_millis(500));
        let mode = MODE.load(Relaxed);
        let frame = FRAME.load(Relaxed);
        if mode == 0 || frame == 0 {
            s.threads = None;
            last = None;
            continue;
        }
        let lines = s.lines(mode);
        // (Drawn again only when a figure or the frame changed.)
        if last.as_ref().is_some_and(|(l, f)| *l == lines && *f == frame) {
            continue;
        }
        let (h, hdr) = (frame & 0x7fff_ffff, frame >> 31 != 0);
        let (mut px, w, ph) = render(&lines, h);
        if hdr {
            crate::textbox::to_pq(&mut px, 203.0);
        }
        generation += 1;
        *PICTURE.lock() = Some(Arc::new(Picture { generation, frame, w, h: ph, px }));
        last = Some((lines, frame));
    }
}

/// The lines as top-down BGRA pixels for a frame `frame_h` lines high (designed at 1080).
fn render(lines: &[Line], frame_h: u32) -> (Vec<u8>, u32, u32) {
    use windows_sys::Win32::Foundation::{RECT, SIZE};
    use windows_sys::Win32::Graphics::Gdi::*;
    let s = (frame_h as f32 / 1080.0).max(0.6);
    let px = |v: f32| (v * s).round() as i32;
    let rgb = |r: u8, g: u8, b: u8| r as u32 | (g as u32) << 8 | (b as u32) << 16;
    unsafe {
        let dc = CreateCompatibleDC(std::ptr::null_mut());
        if dc.is_null() {
            return (vec![0; 4], 1, 1);
        }
        let face: Vec<u16> = "Segoe UI".encode_utf16().chain(std::iter::once(0)).collect();
        let font = CreateFontW(-px(19.0), 0, 0, 0, 600, 0, 0, 0, DEFAULT_CHARSET as _, OUT_TT_PRECIS as _, CLIP_DEFAULT_PRECIS as _, ANTIALIASED_QUALITY as _, (VARIABLE_PITCH as u32 | FF_SWISS as u32) as _, face.as_ptr());
        let old_font = SelectObject(dc, font as _);
        let width_of = |t: &str| {
            let w: Vec<u16> = t.encode_utf16().collect();
            let mut sz = SIZE { cx: 0, cy: 0 };
            if !w.is_empty() {
                GetTextExtentPoint32W(dc, w.as_ptr(), w.len() as i32, &mut sz);
            }
            sz.cx
        };
        let (pad, pitch) = (px(9.0), px(25.0));
        let widest = lines.iter().map(|l| l.iter().map(|(t, _)| width_of(t)).sum::<i32>()).max().unwrap_or(0);
        let (w, h) = ((widest + 2 * pad).max(1) as u32, (lines.len() as i32 * pitch + 2 * pad - px(4.0)).max(1) as u32);
        let mut out = vec![0u8; (w * h * 4) as usize];
        let mut bmi: BITMAPINFO = std::mem::zeroed();
        bmi.bmiHeader.biSize = std::mem::size_of::<BITMAPINFOHEADER>() as u32;
        bmi.bmiHeader.biWidth = w as i32;
        bmi.bmiHeader.biHeight = -(h as i32);
        bmi.bmiHeader.biPlanes = 1;
        bmi.bmiHeader.biBitCount = 32;
        bmi.bmiHeader.biCompression = BI_RGB;
        let mut bits: *mut core::ffi::c_void = std::ptr::null_mut();
        let bmp = CreateDIBSection(dc, &bmi, DIB_RGB_COLORS, &mut bits, std::ptr::null_mut(), 0);
        if bmp.is_null() || bits.is_null() {
            SelectObject(dc, old_font);
            DeleteObject(font as _);
            DeleteDC(dc);
            return (out, w, h);
        }
        let old_bmp = SelectObject(dc, bmp as _);
        // A near-black panel with a thin edge (the game's menus' colours).
        let back = CreateSolidBrush(rgb(11, 11, 13));
        FillRect(dc, &RECT { left: 0, top: 0, right: w as i32, bottom: h as i32 }, back);
        DeleteObject(back as _);
        let edge = CreateSolidBrush(rgb(58, 53, 44));
        FrameRect(dc, &RECT { left: 0, top: 0, right: w as i32, bottom: h as i32 }, edge);
        DeleteObject(edge as _);
        SetBkMode(dc, TRANSPARENT as _);
        for (i, line) in lines.iter().enumerate() {
            let mut x = pad;
            let y = pad - px(2.0) + i as i32 * pitch;
            for (text, value) in line {
                let t: Vec<u16> = text.encode_utf16().collect();
                SetTextColor(dc, if *value { rgb(240, 232, 212) } else { rgb(150, 144, 132) });
                if !t.is_empty() {
                    TextOutW(dc, x, y, t.as_ptr(), t.len() as i32);
                }
                x += width_of(text);
            }
        }
        GdiFlush();
        out.copy_from_slice(std::slice::from_raw_parts(bits as *const u8, (w * h * 4) as usize));
        for p in out.chunks_exact_mut(4) {
            p[3] = 255;
        }
        SelectObject(dc, old_bmp);
        SelectObject(dc, old_font);
        DeleteObject(bmp as _);
        DeleteObject(font as _);
        DeleteDC(dc);
        (out, w, h)
    }
}

/// The overlay's state for the status line (`pcsettings`' line).
pub fn status() -> String {
    let p = PICTURE.lock().clone();
    format!("overlay {} ({})", MODE.load(Relaxed), p.map_or("no picture".into(), |p| format!("picture {} of {}x{}", p.generation, p.w, p.h)))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn nvml_ids_parse() {
        let mut buf = [0u8; 96];
        let text = b"GPU-6d0a3c3e-1f2b-4c5d-8e9f-0a1b2c3d4e5f";
        buf[..text.len()].copy_from_slice(text);
        assert_eq!(parse_uuid(&buf), Some([0x6d, 0x0a, 0x3c, 0x3e, 0x1f, 0x2b, 0x4c, 0x5d, 0x8e, 0x9f, 0x0a, 0x1b, 0x2c, 0x3d, 0x4e, 0x5f]));
        assert_eq!(parse_uuid(b"GPU-12\0"), None);
    }

    #[test]
    fn lows_take_the_slowest_frames() {
        // 1000 flips 16.667 ms apart, one of them 100 ms late.
        let mut q = FLIPS.lock();
        q.clear();
        let mut t = 0u64;
        for i in 0..1000 {
            t += if i == 500 { 100_000 } else { 16_667 };
            q.push_back(t);
        }
        drop(q);
        let (fps, low1, low01) = rates(t).unwrap();
        assert!((fps - 60.0).abs() < 0.1, "{fps}");
        // 0.1% of 999 intervals: the slowest one (10 fps); 1%: the slowest ten.
        assert!((low01 - 10.0).abs() < 0.01, "{low01}");
        let want = 10.0 * 1e6 / (100_000.0 + 9.0 * 16_667.0);
        assert!((low1 - want).abs() < 0.01, "{low1} {want}");
        FLIPS.lock().clear();
    }

    #[test]
    fn the_picture_fits_its_lines() {
        let lines: Vec<Line> = vec![vec![("FPS ".into(), false), ("60".into(), true)], vec![("RAM ".into(), false), ("9.8 GB".into(), true)]];
        let (px, w, h) = render(&lines, 1080);
        assert_eq!(px.len(), (w * h * 4) as usize);
        assert!(w > 40 && h > 40 && h < 120, "{w}x{h}");
        // (Some text drawn: pixels brighter than the panel.)
        assert!(px.chunks_exact(4).any(|p| p[1] > 120));
    }
}
