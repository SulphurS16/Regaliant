//! libSceVideoOut (PS5 API): display buffer sets, flip queue, vblank pacing,
//! output status and color settings.
//!
//! A vblank thread executes queued flips: at each of the display's own vblanks where a
//! window is on a fixed-refresh screen (the frame limit a whole number of them), as soon as
//! the limit allows on a variable-refresh one, at 59.94 Hz on its own timer otherwise
//! (`vblank_thread`, PACING.md).
//! Until the GPU renders into display buffers, presentation shows black: the
//! guest display memory is not produced by the title's GPU work yet, so
//! nothing else would be truthful. Color settings are recorded, not applied.
//!
//! Structures (PS5 layouts; see docs/PROVENANCE.md): BufferAttribute2 (0x50),
//! FlipStatus, VblankStatus, OutputStatus {u32 resolution, u32 dynamicRange,
//! u64 refreshRate, u64 flags, u64 reserved[3]}, ColorSettings {f32 gamma, ...}.

use crate::hle::Registry;
use crate::{reg, sysv};
use parking_lot::{Condvar, Mutex};
use std::collections::VecDeque;
use std::time::{Duration, Instant};

const ERR_INVALID_VALUE: i32 = 0x8029_0001u32 as i32;
const ERR_INVALID_ADDRESS: i32 = 0x8029_0002u32 as i32;
const ERR_INVALID_INDEX: i32 = 0x8029_0004u32 as i32;
const ERR_RESOURCE_BUSY: i32 = 0x8029_0009u32 as i32;
const ERR_INVALID_HANDLE: i32 = 0x8029_000Bu32 as i32;
pub const ERR_FLIP_QUEUE_FULL: i32 = 0x8029_0012u32 as i32;

/// Output reported to the title: 1080p SDR at 59.94 Hz (resolution 1 = 1080p,
/// 2 = 2160p; dynamic range 1 = SDR). A lower output keeps GPU emulation cost
/// down during bring-up; it is a choice of this port, not a console default.
const OUTPUT_RESOLUTION: u32 = 1;
/// The vblank rate while the frame rate is uncapped (`tune uncapped`, the PC tab's Frame Rate
/// Limit): well above what the title renders at, so a frame is shown as soon as it is done.
pub const UNCAPPED_HZ: f64 = 240.0;
/// Dynamic range values of the output status: SDR, and HDR (HDR10).
const RANGE_SDR: u32 = 1;
const RANGE_HDR: u32 = 2;

/// The dynamic range reported to the title: HDR while the window's monitor is in Windows' HDR
/// mode (checked at most every 2 s), else SDR; `DS_HDR_STATUS=<n>` reports `n` (tests).
pub fn dynamic_range() -> u32 {
    use std::sync::atomic::{AtomicU32, AtomicU64, Ordering};
    static FORCED: std::sync::OnceLock<Option<u32>> = std::sync::OnceLock::new();
    if let Some(v) = *FORCED.get_or_init(|| std::env::var("DS_HDR_STATUS").ok().and_then(|v| v.trim().parse().ok())) {
        // (Tests: the PC tab's HDR Off still reports SDR.)
        return if crate::pcsettings::hdr_allowed() { v } else { RANGE_SDR };
    }
    static LAST: AtomicU32 = AtomicU32::new(RANGE_SDR);
    static CHECKED_MS: AtomicU64 = AtomicU64::new(0);
    let now = crate::log::elapsed_ms() as u64;
    let at = CHECKED_MS.load(Ordering::Relaxed);
    if at == 0 || now.saturating_sub(at) >= 2000 {
        CHECKED_MS.store(now.max(1), Ordering::Relaxed);
        let monitor = crate::gpu::vk::window::current_monitor();
        let hdr = crate::pcsettings::hdr_allowed() && monitor.as_ref().and_then(|(_, dev)| crate::display::hdr_now(dev)).is_some();
        let v = if hdr { RANGE_HDR } else { RANGE_SDR };
        // The monitor's signal as Windows drives it, logged when it changes (a washed-out
        // picture can be a YCbCr or limited signal the driver chose for a mode).
        if let Some((_, dev)) = monitor {
            static SIGNAL: parking_lot::Mutex<Option<(String, &'static str, u32)>> = parking_lot::Mutex::new(None);
            if let Some((enc, bits)) = dsconfig::display::signal(&dev) {
                let mut g = SIGNAL.lock();
                if g.as_ref() != Some(&(dev.clone(), enc, bits)) {
                    crate::info!("display: {dev} receives {enc} at {bits} bits per channel");
                    *g = Some((dev, enc, bits));
                }
            }
        }
        if LAST.swap(v, Ordering::Relaxed) != v || at == 0 {
            crate::info!("videoout: output reported as {}", if hdr { "HDR" } else { "SDR" });
        }
        return v;
    }
    LAST.load(Ordering::Relaxed)
}
const REFRESH_59_94HZ: u64 = 3;

#[derive(Clone, Copy, Debug, Default)]
pub struct BufferAttr {
    pub pixel_format: u64,
    pub tiling: u32,
    pub aspect: u32,
    pub width: u32,
    pub height: u32,
    pub pitch: u32,
    pub option: u64,
    pub dcc_control: u32,
}

#[derive(Clone, Copy, Debug)]
pub struct DisplayBuffer {
    pub addr: u64,
    pub metadata: u64,
    pub set: i32,
}

struct Flip {
    index: i32,
    arg: i64,
    submit_tsc: u64,
    /// The GPU work before it completed (`reserve_flip` queues it before,
    /// `commit_flip` marks it): only a ready flip at the front is shown.
    ready: bool,
    /// Unique per reserved flip (0: submitted ready).
    ticket: u64,
}

pub struct Port {
    pub buffers: [Option<DisplayBuffer>; 16],
    pub sets: [Option<BufferAttr>; 16],
    flip_rate: u32,
    pending: VecDeque<Flip>,
    count: u64,
    process_time: u64,
    tsc: u64,
    flip_arg: i64,
    submit_tsc: u64,
    current: i32,
    last_flip_vblank: u64,
    vblanks: u64,
    vblank_time: u64,
    vblank_tsc: u64,
    gamma: f32,
}

static PORT: Mutex<Option<Port>> = Mutex::new(None);

/// The flips shown lately, for the game's step (`gamestep`): each one's number and the
/// instant of the vblank it was shown at (the tick's own time on a grid: the display's
/// vblank or the timer's; the moment itself off the grid).
static FLIP_TIMES: Mutex<VecDeque<(u64, Instant)>> = Mutex::new(VecDeque::new());
/// Wakes the vblank thread when a flip is queued ready or becomes ready (off the grid it
/// shows a ready frame without waiting for a tick).
static FLIP_READY: std::sync::LazyLock<crate::thread::Waker> = std::sync::LazyLock::new(crate::thread::Waker::new);
/// The time between flips the frame limit asks for now (ns; 0: none, the frame rate uncapped).
pub static PACE_NS: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);

/// The last flip shown: its number and instant, and the time since the flip before it.
pub fn latest_flip() -> Option<(u64, Instant, Duration)> {
    let g = FLIP_TIMES.lock();
    let &(n, at) = g.back()?;
    let before = g.iter().rev().nth(1).map_or(Duration::ZERO, |&(_, b)| at.saturating_duration_since(b));
    Some((n, at, before))
}

/// When flip `n` was shown (the last 256 flips are kept).
pub fn flip_time(n: u64) -> Option<Instant> {
    let g = FLIP_TIMES.lock();
    let &(last, _) = g.back()?;
    let back = last.checked_sub(n)? as usize;
    g.iter().rev().nth(back).filter(|e| e.0 == n).map(|e| e.1)
}
static FLIP_CV: Condvar = Condvar::new();
static VBLANK_CV: Condvar = Condvar::new();

fn vblank_thread() {
    crate::log::set_thread_name("vblank");
    // (`DS_SIM_HANG_AT=<s>`, a test: this thread stops that long after it began, no flip from
    // then on: a hung session for the hang watches.)
    let hang_at = std::env::var("DS_SIM_HANG_AT").ok().and_then(|v| v.parse::<u64>().ok()).map(|s| Instant::now() + Duration::from_secs(s));
    // DS_VBLANK_HZ=<hz> (diagnosis, 30-1000): vblanks at another rate, so
    // flips are not quantized to 16.7 ms steps and A/Bs see frame cost below
    // a vblank (the title flips at every vblank, `sceVideoOutSetFlipRate(0)`;
    // it may pace or time its frames by them: not for behavior checks).
    let hz = std::env::var("DS_VBLANK_HZ").ok().and_then(|v| v.parse::<f64>().ok()).filter(|h| (30.0..=1000.0).contains(h));
    let base = match hz {
        Some(h) => {
            crate::info!("vblank at {h} Hz (DS_VBLANK_HZ, diagnosis)");
            Duration::from_nanos((1e9 / h) as u64)
        }
        None => Duration::from_nanos(16_683_333), // 59.94 Hz
    };
    // The uncapped frame rate (`tune uncapped`): on the thread's own timer the vblanks come at
    // UNCAPPED_HZ while it is on (an explicit DS_VBLANK_HZ stays the rate).
    let uncapped_period = Duration::from_nanos((1e9 / UNCAPPED_HZ) as u64);
    let period_now = || if hz.is_none() && crate::gpu::tune::uncapped() { uncapped_period } else { base };
    let mut period = period_now();
    if period != base {
        crate::info!("vblank at {:.2} Hz (the frame rate uncapped)", 1.0 / period.as_secs_f64());
    }
    // How the flips are paced (PACING.md):
    // - On the display's grid (a window on a fixed-refresh screen, `displayclock::lock`): the
    //   title's vblank ticks just after each of the display's (the frame flipped then is
    //   presented at once and shown at the next refresh), and flips stay a whole number of
    //   them apart (`displayclock::refreshes_for` the frame limit; uncapped: one). A frame
    //   done late shows at the next refresh after its due one.
    // - Off the grid (variable refresh, `vrr::active`; or `tune flip_grid` 0): a ready flip is
    //   shown as soon as the frame limit's time has passed since the last one, on a line of
    //   exact multiples of it while frames are on time; the title's vblanks keep the thread's
    //   own timer.
    // - On the thread's own timer (no window, or no steady second of the display's vblanks
    //   yet; a diagnosis rate): at its ticks, as the title's flip rate and `tune frame_cap` ask.
    const LOCK_DELAY: Duration = Duration::from_micros(300);
    let mut next = Instant::now() + period;
    // The last flip's own time: its tick's on a grid, its place on the limit's line off it.
    let mut last_flip = Instant::now();
    let mut last_cap = u64::MAX;
    // On the display's grid: the display's vblank the thread ticks at next (0: at the start)
    // and the display's period as last measured (None: not on that grid).
    let mut lock_target: u64 = 0;
    let mut locked: Option<Duration> = None;
    // The pacing as last logged: (kind, refreshes or 0, the limit's time in microseconds).
    let mut logged: (u8, u32, u64) = (0, 0, 0);
    let mut ticked = true;
    crate::displayclock::start();
    crate::vrr::start();
    loop {
        if hang_at.is_some_and(|t| Instant::now() >= t) {
            crate::warn!("DS_SIM_HANG_AT: the vblank thread stops (a test)");
            loop {
                std::thread::sleep(Duration::from_secs(3600));
            }
        }
        // Once a vblank: the PC settings' console variables (changed values, objects the title
        // builds later) and the other per-vblank work.
        if std::mem::take(&mut ticked) {
            crate::input::motion_tick();
            crate::pcsettings::mouse_blur_tick();
            crate::pcsettings::convar::tick();
            crate::pcsettings::popup::tick();
            crate::pcsettings::pointer::tick();
            crate::dispmode::tick();
        }
        // The rate changed (`tune uncapped`): the next vblank one new period from now.
        let p_now = period_now();
        if p_now != period {
            period = p_now;
            next = Instant::now() + period;
            crate::info!("vblank timer at {:.2} Hz ({})", 1.0 / period.as_secs_f64(), if period == uncapped_period { "the frame rate uncapped" } else { "the frame rate limited" });
        }
        let uncapped = period == uncapped_period;
        // (Frame generation counts as variable refresh: DLSS-G presents the frames on its own
        // pacing, so the display's vblanks follow its presents and not the title's flips; a lock
        // to them would feed back on itself.)
        let fg = crate::gpu::framegen::dlssg::loaded();
        let vrr = hz.is_none() && (crate::vrr::active() || fg);
        let lock = if hz.is_none() && !vrr { crate::displayclock::lock() } else { None };
        match lock {
            Some(s) => {
                if locked.is_none() {
                    crate::displayclock::LOCKS.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
                    lock_target = 0;
                }
                locked = Some(s.period);
                // (At the start, or behind by two or more: the display's next vblank.)
                if lock_target == 0 || lock_target + 2 <= s.count {
                    lock_target = s.count + 1;
                }
                let due = s.last + s.period * lock_target.saturating_sub(s.count) as u32;
                // (`tune tick_lead`: that long before the vblank instead, so the flip made at
                // the tick can be shown at it; PACING.md "Open" 3.)
                let lead = crate::gpu::tune::tick_lead();
                next = if lead == 0 { due + LOCK_DELAY } else { due.checked_sub(Duration::from_micros(lead)).unwrap_or(due) };
            }
            None if locked.is_some() => {
                locked = None;
                next = Instant::now() + period;
            }
            None => {}
        }
        let grid = crate::gpu::tune::flip_grid() && !vrr;
        // The frame limit off the grid: the least time between flips (the title's flip rate
        // and `tune frame_cap` count there as on the timer's grid).
        let limit_gap = |flip_rate: u32| -> Duration {
            if uncapped {
                return Duration::ZERO;
            }
            let cap = crate::gpu::tune::frame_cap();
            let own = base * (flip_rate + 1);
            if cap > 0 { own.max(Duration::from_secs_f64(1.0 / cap as f64)) } else { own }
        };
        let now = Instant::now();
        if grid {
            if next > now {
                crate::thread::short_sleep(next - now);
            }
        } else {
            // Off the grid: a flip that waits ready is shown at its place on the limit's line
            // (now, when that has passed); with none ready the thread sleeps to its next
            // vblank or until one becomes ready (`FLIP_READY`).
            let (ready, gap) = PORT.lock().as_ref().map_or((false, Duration::ZERO), |p| (p.pending.front().is_some_and(|f| f.ready), limit_gap(p.flip_rate)));
            let line = last_flip + gap;
            if ready {
                let wake = next.min(line);
                if wake > now {
                    crate::thread::short_sleep(wake - now);
                }
            } else if next > now {
                crate::thread::sleep_or_wake(next - now, &FLIP_READY);
            }
        }
        let now = Instant::now();
        let tick = now >= next;
        // The tick's own time: the flip shown at it counts from there, not from this wake. A
        // thread woken a grid step or more late (held up by the presenter) is at a later
        // tick's time: on the display's grid the one whose next vblank shows what is
        // presented now.
        let tick_at = {
            let step = locked.unwrap_or(period);
            let late = now.saturating_duration_since(next) + if locked.is_some() { LOCK_DELAY } else { Duration::ZERO };
            next + step * (late.as_secs_f64() / step.as_secs_f64()) as u32
        };
        if tick {
            ticked = true;
            if locked.is_some() {
                lock_target += 1;
                // (Recomputed from the display's vblanks at the next pass.)
                next = now + period;
            } else {
                next += period;
                // Uncapped, a thread held up (the presenter waiting for the display) skips
                // the vblanks it missed instead of ticking them in a burst.
                if uncapped && next < now {
                    next = now + period;
                }
            }
        }
        let mut show: Option<DisplayBuffer> = None;
        let mut flipped = None;
        {
            let mut g = PORT.lock();
            let Some(p) = g.as_mut() else { continue };
            if tick {
                p.vblanks += 1;
                p.vblank_time = crate::hle::kernel::time::process_time_us();
                p.vblank_tsc = unsafe { core::arch::x86_64::_rdtsc() };
            }
            let cap = crate::gpu::tune::frame_cap();
            // The frame limit in frames a second (None: uncapped): the title's own flip rate
            // (every vblank of its 59.94 Hz, or every second or third), `tune frame_cap` under it.
            let limit_fps: Option<f64> = (!uncapped).then(|| {
                let own = 1.0 / (base * (p.flip_rate + 1)).as_secs_f64();
                if cap > 0 { own.min(cap as f64) } else { own }
            });
            // Flips apart in ticks on a grid, and the time between flips the limit asks for.
            let (interval, pace): (u64, Duration) = match locked {
                Some(display) if grid => {
                    let n = crate::displayclock::forced_refreshes().unwrap_or_else(|| limit_fps.map_or(1, |f| crate::displayclock::refreshes_for(1.0 / display.as_secs_f64(), f))) as u64;
                    (n, display * n as u32)
                }
                _ if !grid => (1, limit_gap(p.flip_rate)),
                _ => {
                    // The thread's own timer: the title's flip rate in its vblanks, and under
                    // a frame cap the whole number of them that keeps under it.
                    let mut interval = p.flip_rate as u64 + 1;
                    let min_gap = if cap > 0 { Duration::from_secs_f64(1.0 / cap as f64) } else { Duration::ZERO };
                    if cap > 0 {
                        interval = interval.max((min_gap.as_secs_f64() / period.as_secs_f64() - 0.01).ceil().max(1.0) as u64);
                    }
                    (interval, if uncapped || hz.is_some() { Duration::ZERO } else { period * interval as u32 })
                }
            };
            // (The limit's time to the tenth of a millisecond: a display's measured period wanders.)
            let kind = (if !grid { 3 } else if locked.is_some() { 2 } else { 1 }, if locked.is_some() && grid { interval as u32 } else { 0 }, (pace.as_micros() as u64 + 50) / 100);
            if kind != logged || cap != last_cap {
                logged = kind;
                last_cap = cap;
                crate::displayclock::FLIPS_APART.store(kind.1, std::sync::atomic::Ordering::Relaxed);
                let limit = limit_fps.map_or("uncapped".to_string(), |f| format!("a limit of {f:.2} frames a second"));
                match (kind.0, locked) {
                    (2, Some(display)) => {
                        let rhz = 1.0 / display.as_secs_f64();
                        crate::info!("pacing: the display's grid ({rhz:.3} Hz measured): the title's vblank at each of its vblanks, flips at least {interval} apart ({:.2} frames a second at most; {limit})", rhz / interval as f64);
                    }
                    (3, _) => crate::info!("pacing: off the grid ({}): a ready frame is shown {}; the title's vblank on its own {:.2} Hz timer", if fg { "frame generation paces the display" } else if vrr { "variable refresh" } else { "`tune flip_grid` 0" }, if pace.is_zero() { "at once".to_string() } else { format!("{:.3} ms after the last at the earliest ({limit})", pace.as_secs_f64() * 1e3) }, 1.0 / period.as_secs_f64()),
                    _ => crate::info!("pacing: the vblank thread's own {:.2} Hz timer: flips at least {interval} vblank(s) apart ({limit}{})", 1.0 / period.as_secs_f64(), if cap > 0 { format!(", `tune frame_cap` {cap}") } else { String::new() }),
                }
            }
            // On a grid: at a tick, `interval` ticks after the last flip. Off it: once the
            // limit's time has passed since the last flip's place on the line.
            let due = if grid { tick && p.vblanks - p.last_flip_vblank >= interval } else { now + Duration::from_micros(50) >= last_flip + pace };
            if p.pending.front().is_some_and(|f| f.ready) && due {
                let f = p.pending.pop_front().unwrap();
                p.current = f.index;
                p.count += 1;
                p.flip_arg = f.arg;
                p.submit_tsc = f.submit_tsc;
                if tick {
                    p.process_time = p.vblank_time;
                    p.tsc = p.vblank_tsc;
                } else {
                    p.process_time = crate::hle::kernel::time::process_time_us();
                    p.tsc = unsafe { core::arch::x86_64::_rdtsc() };
                }
                p.last_flip_vblank = p.vblanks;
                // The flip's own time: the tick's on a grid; off it the next place on the
                // limit's line while the frame is on time (within a millisecond), else now.
                last_flip = if grid {
                    tick_at
                } else if !pace.is_zero() && now.saturating_duration_since(last_flip + pace) < Duration::from_millis(1) {
                    last_flip + pace
                } else {
                    now
                };
                {
                    let mut t = FLIP_TIMES.lock();
                    t.push_back((p.count, last_flip));
                    while t.len() > 256 {
                        t.pop_front();
                    }
                }
                PACE_NS.store(pace.as_nanos() as u64, std::sync::atomic::Ordering::Relaxed);
                if p.count <= 3 || p.count % 600 == 0 {
                    crate::info!("flip #{} -> buffer {} (arg {:#x})", p.count, f.index, f.arg);
                }
                if f.index >= 0 {
                    show = p.buffers[f.index as usize];
                }
                crate::gpu::FRAMES.store(p.count, std::sync::atomic::Ordering::Relaxed);
                crate::gpu::pacing::note_flip(p.count as u64, p.process_time);
                flipped = Some(p.count);
                FLIP_CV.notify_all();
            }
            if tick {
                VBLANK_CV.notify_all();
            }
        }
        if let Some(n) = flipped {
            crate::gpu::timeline::flipped(n as u64);
        }
        if let Some(b) = show {
            let (n, attr) = PORT.lock().as_ref().map(|p| (p.count, p.sets.get(b.set as usize).copied().flatten())).unwrap_or((0, None));
            if let Some(a) = attr {
                crate::gpu::capture::note_display_size(a.width, a.height);
            }
            maybe_screenshot(&b, n);
            crate::gpu::vk::framewatch::SHOWN_FLIP.store(n as u64, std::sync::atomic::Ordering::Relaxed);
            crate::displayclock::note_present();
            present_buffer(&b, n as u32);
        }
    }
}

fn port<R>(handle: i32, f: impl FnOnce(&mut Port) -> R) -> Result<R, i32> {
    if handle != 1 {
        return Err(ERR_INVALID_HANDLE);
    }
    let mut g = PORT.lock();
    match g.as_mut() {
        Some(p) => Ok(f(p)),
        None => Err(ERR_INVALID_HANDLE),
    }
}

/// The flip count the last queued flip will be shown as (`p.count` then; diagnosis traces name
/// the flip the work they record belongs to by it: the next one).
pub static QUEUED_FLIP: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);

/// Queues a flip of `index` (-1 = blank). Used by sceVideoOutSubmitFlip and by
/// GPU flips (sceAgcDcbSetFlip) when the command processor reaches them.
pub fn submit_flip(handle: i32, index: i32, arg: i64) -> i32 {
    port(handle, |p| {
        if index >= 16 || (index >= 0 && p.buffers[index as usize].is_none()) {
            return ERR_INVALID_INDEX;
        }
        if p.pending.len() >= 16 {
            return ERR_FLIP_QUEUE_FULL;
        }
        p.pending.push_back(Flip { index, arg, submit_tsc: unsafe { core::arch::x86_64::_rdtsc() }, ready: true, ticket: 0 });
        QUEUED_FLIP.store(p.count as u64 + p.pending.len() as u64, std::sync::atomic::Ordering::Relaxed);
        FLIP_READY.wake();
        0
    })
    .unwrap_or_else(|e| e)
}

/// A flip queued now and shown only once `commit_flip` marks it ready (a
/// GPU queue's FLIP packet, `tune flip_async`: the GPU work before it is
/// still running). It counts towards the queue's limit and keeps its buffer
/// busy at once. Ok(ticket), or the error `submit_flip` would give.
pub fn reserve_flip(handle: i32, index: i32, arg: i64) -> Result<u64, i32> {
    static NEXT: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(1);
    port(handle, |p| {
        if index >= 16 || (index >= 0 && p.buffers[index as usize].is_none()) {
            return Err(ERR_INVALID_INDEX);
        }
        if p.pending.len() >= 16 {
            return Err(ERR_FLIP_QUEUE_FULL);
        }
        let ticket = NEXT.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
        p.pending.push_back(Flip { index, arg, submit_tsc: unsafe { core::arch::x86_64::_rdtsc() }, ready: false, ticket });
        QUEUED_FLIP.store(p.count as u64 + p.pending.len() as u64, std::sync::atomic::Ordering::Relaxed);
        Ok(ticket)
    })
    .unwrap_or_else(Err)
}

/// The GPU work before a reserved flip completed: it may be shown.
pub fn commit_flip(ticket: u64) {
    let mut g = PORT.lock();
    if let Some(p) = g.as_mut() {
        if let Some(f) = p.pending.iter_mut().find(|f| f.ticket == ticket) {
            f.ready = true;
            FLIP_READY.wake();
        }
    }
}

/// Whether display buffer `index` is scanned out or waiting in the flip
/// queue (GPU `WaitUntilSafeForRendering` waits until it is neither).
pub fn buffer_busy(handle: i32, index: i32) -> bool {
    port(handle, |p| p.current == index || p.pending.iter().any(|f| f.index == index)).unwrap_or(false)
}

pub fn display_buffer(index: i32) -> Option<(DisplayBuffer, BufferAttr)> {
    let g = PORT.lock();
    let p = g.as_ref()?;
    let b = p.buffers.get(index as usize).copied().flatten()?;
    let a = p.sets.get(b.set as usize).copied().flatten()?;
    Some((b, a))
}

unsafe fn read_attr(attr: *const u8) -> BufferAttr {
    unsafe {
        let r32 = |o: usize| *(attr.add(o) as *const u32);
        let r64 = |o: usize| *(attr.add(o) as *const u64);
        BufferAttr {
            tiling: r32(0x04),
            aspect: r32(0x08),
            width: r32(0x0C),
            height: r32(0x10),
            pitch: r32(0x14),
            option: r64(0x18),
            pixel_format: r64(0x20),
            dcc_control: r32(0x30),
        }
    }
}

sysv! {
    fn sceVideoOutOpen(user: i32, bus: i32, index: i32, param: *const u8) -> i32 {
        if bus != 0 { return ERR_INVALID_VALUE; }
        let mut g = PORT.lock();
        if g.is_some() { return ERR_RESOURCE_BUSY; }
        *g = Some(Port {
            buffers: [None; 16],
            sets: [None; 16],
            flip_rate: 0,
            pending: VecDeque::new(),
            count: 0,
            process_time: 0,
            tsc: 0,
            flip_arg: 0,
            submit_tsc: 0,
            current: -1,
            last_flip_vblank: 0,
            vblanks: 0,
            vblank_time: 0,
            vblank_tsc: 0,
            gamma: 1.0,
        });
        drop(g);
        // (8 MiB: the presenter calls into Streamline, NGX and the driver on this thread, which
        // make DLSS Frame Generation's swapchain and run its first frames there.)
        crate::thread::spawn_host("vblank", 8 << 20, Box::new(vblank_thread)).expect("vblank thread");
        crate::info!("sceVideoOutOpen -> 1");
        1
    }
    fn sceVideoOutClose(h: i32) -> i32 { port(h, |_| 0).unwrap_or_else(|e| e) }
    fn sceVideoOutSetFlipRate(h: i32, rate: i32) -> i32 {
        port(h, |p| {
            p.flip_rate = rate.clamp(0, 2) as u32;
            crate::info!("sceVideoOutSetFlipRate({rate})");
            0
        }).unwrap_or_else(|e| e)
    }
    fn sceVideoOutColorSettingsSetGamma_(settings: *mut f32, gamma: f32, size: u64) -> i32 {
        if settings.is_null() { return ERR_INVALID_ADDRESS; }
        if !(0.1..=2.0).contains(&gamma) { return ERR_INVALID_VALUE; }
        *settings = gamma;
        0
    }
    fn sceVideoOutAdjustColor_(h: i32, settings: *const f32, size: u64) -> i32 {
        if settings.is_null() { return ERR_INVALID_ADDRESS; }
        let gamma = *settings;
        port(h, |p| {
            p.gamma = gamma;
            crate::info!("sceVideoOutAdjustColor: gamma {gamma} (recorded; not applied to presentation yet)");
            0
        }).unwrap_or_else(|e| e)
    }
    fn sceVideoOutGetOutputStatus(h: i32, st: *mut u8) -> i32 {
        if st.is_null() { return ERR_INVALID_ADDRESS; }
        port(h, |_| {
            std::ptr::write_bytes(st, 0, 0x30);
            *(st as *mut u32) = OUTPUT_RESOLUTION;
            *(st.add(4) as *mut u32) = dynamic_range();
            *(st.add(8) as *mut u64) = REFRESH_59_94HZ;
            0
        }).unwrap_or_else(|e| e)
    }
    fn sceVideoOutSetBufferAttribute2(attr: *mut u8, pixel_format: u64, tiling: u32, width: u32, height: u32, option: u64, dcc_control: u32, dcc_clear: u64) {
        if attr.is_null() { return; }
        std::ptr::write_bytes(attr, 0, 0x50);
        *(attr.add(0x04) as *mut u32) = tiling;
        *(attr.add(0x08) as *mut u32) = 0; // aspect ratio 16:9
        *(attr.add(0x0C) as *mut u32) = width;
        *(attr.add(0x10) as *mut u32) = height;
        *(attr.add(0x14) as *mut u32) = width; // pitch in pixels
        *(attr.add(0x18) as *mut u64) = option;
        *(attr.add(0x20) as *mut u64) = pixel_format;
        *(attr.add(0x28) as *mut u64) = dcc_clear;
        *(attr.add(0x30) as *mut u32) = dcc_control;
    }
    fn sceVideoOutRegisterBuffers2(h: i32, set: i32, start: i32, bufs: *const u64, num: i32, attr: *const u8, category: i32, option: u64) -> i32 {
        if bufs.is_null() || attr.is_null() { return ERR_INVALID_ADDRESS; }
        if !(0..16).contains(&set) || start < 0 || num <= 0 || start + num > 16 { return ERR_INVALID_INDEX; }
        let a = read_attr(attr);
        port(h, |p| {
            p.sets[set as usize] = Some(a);
            let mut addrs = Vec::new();
            for i in 0..num as usize {
                // SceVideoOutBuffers entries: {data, metadata, reserved[2]}
                // (32 bytes; the title fills them at eboot+0x90a99b/0x90a9c6).
                let data = *bufs.add(i * 4);
                let meta = *bufs.add(i * 4 + 1);
                p.buffers[start as usize + i] = Some(DisplayBuffer { addr: data, metadata: meta, set });
                addrs.push((data, meta));
            }
            crate::info!(
                "sceVideoOutRegisterBuffers2 set {set} [{start}..{}) {}x{} pitch {} format {:#x} tiling {} option {:#x} dcc {:#x} category {category} buffers {:x?}",
                start + num, a.width, a.height, a.pitch, a.pixel_format, a.tiling, a.option, a.dcc_control, addrs
            );
            0
        }).unwrap_or_else(|e| e)
    }
    fn sceVideoOutSubmitChangeBufferAttribute2(h: i32, set: i32, attr: *const u8, option: u64) -> i32 {
        if attr.is_null() { return ERR_INVALID_ADDRESS; }
        if !(0..16).contains(&set) { return ERR_INVALID_INDEX; }
        let a = read_attr(attr);
        port(h, |p| {
            p.sets[set as usize] = Some(a);
            crate::info!("sceVideoOutSubmitChangeBufferAttribute2 set {set}: {}x{} format {:#x}", a.width, a.height, a.pixel_format);
            0
        }).unwrap_or_else(|e| e)
    }
    fn sceVideoOutUnregisterBuffers(h: i32, set: i32) -> i32 {
        if !(0..16).contains(&set) { return ERR_INVALID_INDEX; }
        port(h, |p| {
            p.sets[set as usize] = None;
            for b in p.buffers.iter_mut() {
                if b.is_some_and(|x| x.set == set) { *b = None; }
            }
            0
        }).unwrap_or_else(|e| e)
    }
    fn sceVideoOutSubmitFlip(h: i32, index: i32, mode: u32, arg: i64) -> i32 {
        submit_flip(h, index, arg)
    }
    fn sceVideoOutIsFlipPending(h: i32) -> i32 {
        let _c = crate::hle::waitcensus::scope(|| "flip pending poll".into(), 0);
        port(h, |p| p.pending.len() as i32).unwrap_or_else(|e| e)
    }
    fn sceVideoOutWaitVblank(h: i32) -> i32 {
        if h != 1 { return ERR_INVALID_HANDLE; }
        let _c = crate::hle::waitcensus::scope(|| "wait vblank".into(), 0);
        let mut g = PORT.lock();
        let Some(start) = g.as_ref().map(|p| p.vblanks) else { return ERR_INVALID_HANDLE };
        while g.as_ref().is_some_and(|p| p.vblanks == start) {
            VBLANK_CV.wait_for(&mut g, Duration::from_millis(50));
        }
        0
    }
    fn sceVideoOutGetFlipStatus(h: i32, st: *mut u8) -> i32 {
        if st.is_null() { return ERR_INVALID_ADDRESS; }
        let _c = crate::hle::waitcensus::scope(|| "flip status poll".into(), 0);
        port(h, |p| {
            std::ptr::write_bytes(st, 0, 0x88);
            *(st as *mut u64) = p.count;
            *(st.add(0x08) as *mut u64) = p.process_time;
            *(st.add(0x18) as *mut i64) = p.flip_arg;
            *(st.add(0x28) as *mut u64) = p.tsc;
            *(st.add(0x30) as *mut i32) = 0; // gcQueueNum
            *(st.add(0x34) as *mut i32) = p.pending.len() as i32;
            *(st.add(0x38) as *mut i32) = p.current;
            *(st.add(0x40) as *mut u64) = p.submit_tsc;
            0
        }).unwrap_or_else(|e| e)
    }
    fn sceVideoOutSetWindowModeMargins(h: i32, top: i32, bottom: i32) -> i32 {
        port(h, |_| {
            crate::info!("sceVideoOutSetWindowModeMargins(top {top}, bottom {bottom}) (recorded only)");
            0
        }).unwrap_or_else(|e| e)
    }
}

pub fn register(r: &mut Registry) {
    reg!(r, "libSceVideoOut", [
        sceVideoOutOpen, sceVideoOutClose, sceVideoOutSetFlipRate, sceVideoOutColorSettingsSetGamma_,
        sceVideoOutAdjustColor_, sceVideoOutGetOutputStatus, sceVideoOutSetBufferAttribute2,
        sceVideoOutRegisterBuffers2, sceVideoOutSubmitChangeBufferAttribute2, sceVideoOutUnregisterBuffers,
        sceVideoOutSubmitFlip, sceVideoOutIsFlipPending, sceVideoOutWaitVblank, sceVideoOutGetFlipStatus,
        sceVideoOutSetWindowModeMargins,
    ]);
}

/// Whether a display buffer format holds HDR values (PQ-encoded BT.2020: color space bits set;
/// the title's SDR format 0x8100000022000000 has none, its HDR one 0x8100070422000000).
fn is_hdr_format(f: u64) -> bool {
    (f >> 32) & 0xFFFF != 0
}

/// The size display buffer `b` is registered with (its set's attribute).
fn registered_size(b: &DisplayBuffer) -> Option<(u32, u32)> {
    PORT.lock().as_ref().and_then(|p| p.sets.get(b.set as usize).copied().flatten()).map(|a| (a.width, a.height))
}

/// Presents a flipped display buffer: the GPU image the title rendered at
/// the buffer's address (`DS_GPU=1`), else a black frame of the buffer's size
/// (never raw guest memory, which is tiled).
/// Shows display buffer `b` as flip `frame` (the frame's index: `gpu::streamline`).
fn present_buffer(b: &DisplayBuffer, frame: u32) {
    if crate::gpu::vk::try_ctx().is_none() {
        return;
    }
    if crate::gpu::exec::enabled() {
        crate::gpu::framegen::capture::note_display_buffer(b.addr);
        let size = registered_size(b);
        if let Some((image, format, width, height)) = crate::gpu::images::with(|im| im.rendered_at(b.addr, size)) {
            // Diagnosis: the image each buffer presents, logged when it changes.
            static LAST: Mutex<Vec<(u64, u64)>> = Mutex::new(Vec::new());
            let id = ash::vk::Handle::as_raw(image);
            let changed = {
                let mut last = LAST.lock();
                match last.iter_mut().find(|(a, _)| *a == b.addr) {
                    Some(e) if e.1 == id => false,
                    Some(e) => {
                        e.1 = id;
                        true
                    }
                    None => {
                        last.push((b.addr, id));
                        true
                    }
                }
            };
            if changed {
                let all = crate::gpu::images::with(|im| im.at_base_report(b.addr));
                crate::info!("present: buffer {:#x} shows {format:?} {width}x{height} (registered {size:?}); images there: {all}", b.addr);
            }
            // HDR display buffers: the format's color space bits (0x0704: PQ-encoded BT.2020, what
            // the title registers while the output is reported as HDR; SDR has none).
            let hdr = PORT.lock().as_ref().and_then(|p| p.sets.get(b.set as usize).copied().flatten()).is_some_and(|a| is_hdr_format(a.pixel_format));
            crate::gpu::vk::present::present(crate::gpu::vk::present::Source::Image { image, format, layout: ash::vk::ImageLayout::GENERAL, width, height, hdr }, Some(frame));
            return;
        }
    }
    let (w, h) = PORT
        .lock()
        .as_ref()
        .and_then(|p| p.sets.get(b.set as usize).copied().flatten())
        .map(|a| (a.width.clamp(1, 8192), a.height.clamp(1, 8192)))
        .unwrap_or((1920, 1080));
    let data = vec![0u8; (w * h * 4) as usize];
    crate::gpu::vk::present::present(crate::gpu::vk::present::Source::Pixels { data, width: w, height: h }, None);
}

/// Screenshot requested through the control channel (file name stem).
pub static SHOT_REQUEST: Mutex<Option<String>> = Mutex::new(None);

/// A burst of shots of consecutive flips (`burst`: diagnosis of one-frame glitches): the name
/// stem, the shots left, the next index and a crop (x, y, width, height) of the frame.
pub struct Burst {
    pub name: String,
    pub left: u32,
    pub index: u32,
    pub crop: Option<(u32, u32, u32, u32)>,
    /// Saved at half the size (2x2 pixels averaged).
    pub half: bool,
}

/// F10 (the window): the next 60 presented frames saved at half size as `f10_<time>_NNN.png` in
/// the run directory, to show a glitch as it happens.
pub fn burst_hotkey() {
    let secs = std::time::SystemTime::now().duration_since(std::time::UNIX_EPOCH).map_or(0, |d| d.as_secs());
    let name = format!("f10_{secs}");
    crate::info!("F10: the next 60 frames saved at half size as {name}_NNN.png in the run directory");
    *BURST.lock() = Some(Burst { name, left: 60, index: 0, crop: None, half: true });
}

/// An RGBA8 frame at half its size (each 2x2 block averaged; an odd last row or column dropped).
fn half_rgba(rgba: &[u8], w: u32, h: u32) -> (Vec<u8>, u32, u32) {
    let (hw, hh) = (w / 2, h / 2);
    let mut out = Vec::with_capacity((hw * hh * 4) as usize);
    for y in 0..hh {
        let (r0, r1) = (((2 * y) * w * 4) as usize, ((2 * y + 1) * w * 4) as usize);
        for x in 0..hw as usize {
            for c in 0..4 {
                let s = rgba[r0 + 8 * x + c] as u32 + rgba[r0 + 8 * x + 4 + c] as u32 + rgba[r1 + 8 * x + c] as u32 + rgba[r1 + 8 * x + 4 + c] as u32;
                out.push(((s + 2) / 4) as u8);
            }
        }
    }
    (out, hw, hh)
}
pub static BURST: Mutex<Option<Burst>> = Mutex::new(None);

/// A crop of an RGBA8 frame (clamped to it).
fn crop_rgba(rgba: &[u8], w: u32, h: u32, c: (u32, u32, u32, u32)) -> (Vec<u8>, u32, u32) {
    let (x, y) = (c.0.min(w), c.1.min(h));
    let (cw, ch) = (c.2.min(w - x), c.3.min(h - y));
    let mut out = Vec::with_capacity((cw * ch * 4) as usize);
    for row in y..y + ch {
        let a = ((row * w + x) * 4) as usize;
        out.extend_from_slice(&rgba[a..a + (cw * 4) as usize]);
    }
    (out, cw, ch)
}

/// Screenshots being converted, encoded and written on threads of their own
/// (`maybe_screenshot`); `wait_shots` waits for them at shutdown.
static SHOTS_PENDING: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);

/// Waits (at most 10 s) for the screenshots still being written.
pub fn wait_shots() {
    let t = std::time::Instant::now();
    while SHOTS_PENDING.load(std::sync::atomic::Ordering::Acquire) > 0 && t.elapsed() < std::time::Duration::from_secs(10) {
        std::thread::sleep(std::time::Duration::from_millis(20));
    }
}

/// `imgprobe <flips> <u> <v>` (diagnosis): flips left, and where every GPU-written image is read
/// at each of them (`Images::probe_used`).
pub static PROBE: Mutex<Option<(u32, f32, f32)>> = Mutex::new(None);

fn maybe_screenshot(b: &DisplayBuffer, flip: u64) {
    let probe = {
        let mut g = PROBE.lock();
        let r = *g;
        if let Some(p) = g.as_mut() {
            p.0 = p.0.saturating_sub(1);
            if p.0 == 0 {
                *g = None;
            }
        }
        r
    };
    if let Some((_, u, v)) = probe {
        if crate::gpu::vk::try_ctx().is_some() && crate::gpu::exec::enabled() {
            let lines = crate::gpu::images::with(|im| im.probe_used(u, v));
            crate::info!("probe flip {flip}: {} image(s)\n    {}", lines.len(), lines.join("\n    "));
        }
    }
    let mut requested = SHOT_REQUEST.lock().take();
    let mut crop = None;
    let mut half = false;
    if requested.is_none() {
        let mut g = BURST.lock();
        if let Some(b) = g.as_mut() {
            requested = Some(format!("{}_{:03}", b.name, b.index));
            crop = b.crop;
            half = b.half;
            b.index += 1;
            b.left = b.left.saturating_sub(1);
            if b.left == 0 {
                *g = None;
            }
        }
    }
    let listed = std::env::var("DS_SCREENSHOT").is_ok_and(|list| list.split(',').any(|s| s.trim().parse::<u64>().ok() == Some(flip)));
    if requested.is_none() && !listed {
        return;
    }
    // The buffer is read back here (as flipped); its conversion, PNG encoding
    // and write run on a thread of their own: on this thread they held the
    // flips ~0.5 s per 4K shot (the boss fights' worst frames, 2026-11-01
    // night).
    let read = || -> Result<(Vec<u8>, ash::vk::Format, u32, u32, std::path::PathBuf), String> {
        if crate::gpu::vk::try_ctx().is_none() || !crate::gpu::exec::enabled() {
            return Err("GPU execution is off (DS_GPU=1 renders frames)".into());
        }
        let (raw, format, w, h) = crate::gpu::images::with(|im| -> Result<(Vec<u8>, ash::vk::Format, u32, u32), String> {
            let (image, format, w, h) = im.rendered_at(b.addr, registered_size(b)).ok_or("nothing was rendered to the buffer")?;
            Ok((im.read_back(image, format, w, h)?, format, w, h))
        })?;
        let dir = crate::run_dir().ok_or("no run directory")?;
        let name = match &requested {
            Some(n) if !n.is_empty() => format!("{n}.png"),
            _ => format!("screenshot-{flip}.png"),
        };
        Ok((raw, format, w, h, dir.join(name)))
    };
    let addr = b.addr;
    match read() {
        Ok((raw, format, w, h, path)) => {
            SHOTS_PENDING.fetch_add(1, std::sync::atomic::Ordering::AcqRel);
            let write = move || {
                let r = crate::gpu::images::to_rgba8(format, &raw).ok_or_else(|| format!("no RGBA8 conversion for {format:?}")).and_then(|rgba| {
                    let (rgba, w, h) = match crop {
                        Some(c) => crop_rgba(&rgba, w, h, c),
                        None => (rgba, w, h),
                    };
                    let (rgba, w, h) = if half { half_rgba(&rgba, w, h) } else { (rgba, w, h) };
                    std::fs::write(&path, crate::gpu::png::encode(w, h, &rgba)).map_err(|e| e.to_string())
                });
                match r {
                    Ok(()) => crate::info!("screenshot of flip {flip} (buffer {addr:#x}): {}", path.display()),
                    Err(e) => crate::warn!("screenshot of flip {flip} (buffer {addr:#x}) unavailable: {e}"),
                }
                SHOTS_PENDING.fetch_sub(1, std::sync::atomic::Ordering::AcqRel);
            };
            if let Err(e) = std::thread::Builder::new().name("shot-write".into()).spawn(write) {
                SHOTS_PENDING.fetch_sub(1, std::sync::atomic::Ordering::AcqRel);
                crate::warn!("screenshot of flip {flip} (buffer {addr:#x}) unavailable: no writer thread ({e})");
            }
        }
        Err(e) => crate::warn!("screenshot of flip {flip} (buffer {addr:#x}) unavailable: {e}"),
    }
}
