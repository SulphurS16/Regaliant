//! The title's frame step (PPSA01341 01.007.000; PACING.md, FINDINGS.md 2026-11-17 night): how
//! much game time each frame advances. The port sets it from the frames shown (`tune
//! pace_step`), so that the game runs at the same speed at every frame rate and each frame
//! moves by the time frames are shown for.
//!
//! The title's own step (eboot+0xb280e0, called once a frame from eboot+0xb27f77 with the
//! timer at [app+0x28F0]): an expected step = its vsync interval ([[eboot+0x2F662A8]+0x229800],
//! 1) over its display's refresh rate ([[eboot+0x2F0A988]+0x28], the constant 59.94 its
//! display object is built with), the measured time since the last call (rdtsc), and in its
//! fixed mode (a setting, on) the step = the expected step plus or minus half steps while the
//! accumulated error is half a step or more (`gameDeltaTimeCorrectionTolorance` 0.5), never
//! under the expected step (`gameMinDeltaTime` -1: the expected step is the floor) and never
//! over `gameMaxDeltaTime` (0.1 s). The step lands in the timer's +0x20 and +0x24, the error in
//! +0x28; the physics' fixed step (the physics manager's +0x2C) and the title's frame-counted
//! waits take the expected step (the global at eboot+0x2F7B4A0).
//!
//! So on its own the title runs the game fast above 60 frames a second (the floor: measured
//! x1.05 at 63, x1.62 at 97), and under 60 its steps come in halves of 16.68 ms whatever the
//! frames' real spacing.
//!
//! The port's step: every frame step takes the time between the next two flips in order (the
//! vblank thread's `videoout::flip_time`: on a grid the ticks' own times, so the times are
//! whole numbers of the display's refreshes), never one that has not been shown yet. The step
//! is their mean over the last `tune pace_smooth` frames (8): exact and constant while every
//! frame is shown as long as the last (the frame limit held), and a frame shown late gives
//! its extra time to the following frames evenly instead of one frame jumping. No time is
//! lost or made: the steps add up to the flips' times (but a single time over the title's
//! `gameMaxDeltaTime` is cut to it, as the title cuts its own). The mean of 8 against other
//! rules, measured with each frame paired to its flip: PACING.md.
//!
//! The title's refresh-rate value and its expected step stay its own: the motion blur's scale
//! reads the one, the developers' automation waits (frames of 1/59.94 s) the other.

use parking_lot::Mutex;
use std::collections::VecDeque;
use std::sync::atomic::{AtomicU32, AtomicU64, Ordering::Relaxed};
use std::time::{Duration, Instant};

/// The main executable's base (`init`).
static BASE: AtomicU64 = AtomicU64::new(0);

/// Module-relative addresses: the expected step the title's frame step computed last (f32), the
/// pointer to its display object (+0x28 the refresh rate, f32) and to its render state
/// (+0x229800 the vsync interval, i32), the pointers to the settings' values, the pointer to
/// the physics manager (+0x2C its fixed step, f32).
const EXPECTED: u64 = 0x2F7_B4A0;
const DISPLAY: u64 = 0x2F0_A988;
const RENDER: u64 = 0x2F6_62A8;
const RENDER_VSYNC: u64 = 0x22_9800;
const MIN_DT: u64 = 0x2F7_8E30;
const MAX_DT: u64 = 0x2F7_8F30;
const TOLERANCE: u64 = 0x2F7_9030;
const PHYSICS: u64 = 0x2F8_C3B0;
const PHYSICS_STEP: u64 = 0x2C;
/// The application object's timer and its fields.
const APP_TIMER: u64 = 0x28F0;
const T_STEP: u64 = 0x20;
const T_LAST: u64 = 0x24;
const T_ERROR: u64 = 0x28;
/// The title's own expected step (1/59.94 s).
const TITLE_STEP: f32 = 1.0 / 59.94;
/// The longest physics step the port asks for (the title's 30 fps mode's).
const PHYS_MAX: f32 = 2.0 / 59.94;

/// The step the per-frame simulation at eboot+0xe6f180 takes (f32 bits; `fixed_step_code`
/// loads it): the game's step, at most the title's 1/59.94 s.
static FIXED_STEP: AtomicU32 = AtomicU32::new(0x3C88_AB85);

/// The game's speed in hundredths (100: real time). Only the shader preparation's self-play sets
/// another (`prepplay`): its levels play at twice the speed, each frame's step twice the time
/// between its flips (the title's own 30 fps mode's step at 60 frames a second), so its walks and
/// fights take half the time.
pub static SPEED_X100: AtomicU32 = AtomicU32::new(100);

/// The shader preparation's levels as a fast-forward (`prepplay`): every frame steps by the
/// title's 1/59.94 s times the speed whatever the time between its flips, the physics by the
/// title's step, so that each frame does what a frame at 60 a second does (the automation's
/// waits count frames, and the self-play's presses and sticks too, `input::FLIP_TIMED`) while
/// the frames come as fast as the PC makes them (2026-11-20: the levels at ~30 frames a second
/// had stretched every frame-counted wait to twice its time).
pub static FIXED_FEED: std::sync::atomic::AtomicBool = std::sync::atomic::AtomicBool::new(false);

/// The code inserted at eboot+0xe6f1f9 (`titlefix`): the run counted (`lock inc qword ptr
/// [counter]`), then `mov rax, &FIXED_STEP; vmovss xmm0, [rax]` in place of the title's
/// 1/refresh rate in xmm0. rax is loaded by the displaced instruction next, and the flags the
/// count changes are set again by the function's next `test` before any use.
pub fn fixed_step_code(counter: u64) -> Vec<u8> {
    let mut c = vec![0x48, 0xb8];
    c.extend_from_slice(&counter.to_le_bytes());
    c.extend_from_slice(&[0xf0, 0x48, 0xff, 0x00]);
    c.extend_from_slice(&[0x48, 0xb8]);
    c.extend_from_slice(&(FIXED_STEP.as_ptr() as u64).to_le_bytes());
    c.extend_from_slice(&[0xc5, 0xfa, 0x10, 0x00]);
    c
}

/// One frame step.
#[derive(Clone, Copy)]
struct Rec {
    at: Instant,
    /// The step the title computed and the one it runs with (the same unless the port set it).
    title: f32,
    step: f32,
    /// The time between flips this frame took (s; the port's step is the mean of the last ones).
    interval: f32,
    /// The flip the frame's time came from, the newest flip shown then, and how long ago
    /// that one was shown.
    flip: u64,
    newest: u64,
    phase: f32,
    /// The physics' fixed step.
    phys: f32,
    /// The player's position (the gameplay camera's copy) and the camera's yaw, where known.
    pos: Option<[f32; 3]>,
    yaw: Option<f32>,
    /// `DS_PACE_PAIR=1`: the flip that shows this frame (0: not known) and when it was shown.
    own_flip: u64,
    shown: Option<Instant>,
}

/// The game's clock as the port keeps it: which flip the last frame step's time came from.
#[derive(Default)]
struct Clock {
    /// The flip whose time the last frame step took, and that time (the newest so far).
    flip: u64,
    at: Option<Instant>,
    /// Frame steps in a row that found no new flip.
    starved: u32,
    /// The last times between flips taken (s), newest last.
    intervals: VecDeque<f64>,
    /// The frame limit's time a frame as held (s; 0: none), for the exact step while the
    /// limit is held, and the time the exact steps left out or added (s), given back once it
    /// reaches half a millisecond.
    nominal: f64,
    drift: f64,
    /// Time given back, added to the next interval taken.
    pending: f64,
}

#[derive(Default)]
struct State {
    recs: VecDeque<Rec>,
    clock: Clock,
    /// Since the last report: frames, game time (s), the first and last frame's instants, the
    /// steps by value (in units of 10 us) with their counts, frames whose step was the exact
    /// frame limit's.
    frames: u64,
    game: f64,
    first: Option<Instant>,
    last: Option<Instant>,
    steps: Vec<(u32, u64)>,
    exact: u64,
    /// All frame steps and their game time since the start.
    total_frames: u64,
    total_game: f64,
    /// Frame steps the port set, frames that found no new flip, times the clock skipped to
    /// the newest flip or to the moment (status).
    fed: u64,
    starved: u64,
    resyncs: u64,
}

static STATE: Mutex<Option<State>> = Mutex::new(None);

pub fn init(base: u64) {
    BASE.store(base, Relaxed);
    *STATE.lock() = Some(State::default());
}

fn rdf(a: u64) -> f32 {
    unsafe { (a as *const f32).read_volatile() }
}

fn rdq(a: u64) -> u64 {
    unsafe { (a as *const u64).read_volatile() }
}

fn wrf(a: u64, v: f32) {
    unsafe { (a as *mut f32).write_volatile(v) }
}

/// A setting's float value through its pointer (NaN while the title has not set the pointer).
fn setting(base: u64, at: u64) -> f32 {
    let a = rdq(base + at);
    if a == 0 { f32::NAN } else { rdf(a) }
}

/// The title's refresh rate and vsync interval as its frame step reads them (None before the
/// title built its display object).
pub fn title_rate() -> Option<(f32, i32)> {
    let base = BASE.load(Relaxed);
    if base == 0 {
        return None;
    }
    let (display, render) = (rdq(base + DISPLAY), rdq(base + RENDER));
    if display == 0 || render == 0 {
        return None;
    }
    Some((rdf(display + 0x28), unsafe { ((render + RENDER_VSYNC) as *const i32).read_volatile() }))
}

impl Clock {
    /// The time this frame step takes (s): the time between the flip the last frame step took
    /// and the next one, when that one has been shown; nothing while it has not (the frame
    /// steps run ahead of the flips by a frame or two: the lag settles at the deepest seen);
    /// the whole time to the newest flip when the flips ran ahead by more than three (frames
    /// without a step: a load); the time to this moment when no flip came for three frame
    /// steps or a quarter second (frame steps without flips). Also the flip taken, the newest
    /// one, and what happened (0 a flip's time, 1 no new flip, 2 a skip).
    fn advance(&mut self, now: Instant) -> (f64, u64, u64, u8) {
        let Some((newest, newest_at, _)) = crate::hle::libs::videoout::latest_flip() else {
            // (No flips yet: real time.)
            let iv = self.at.map_or(0.0, |a| now.saturating_duration_since(a).as_secs_f64());
            self.at = Some(now);
            return (iv, 0, 0, 2);
        };
        let Some(at) = self.at else {
            self.flip = newest;
            self.at = Some(newest_at);
            return (0.0, newest, newest, 2);
        };
        let lag = || now.saturating_duration_since(at) > Duration::from_millis(250);
        let (to, flip, what) = if newest > self.flip + 3 {
            (newest_at, newest, 2)
        } else if newest > self.flip {
            match crate::hle::libs::videoout::flip_time(self.flip + 1) {
                Some(t) => (t, self.flip + 1, 0),
                None => (newest_at, newest, 2),
            }
        } else if self.starved >= 2 || lag() {
            // (Frame steps without flips: the moment itself, so that game time goes on.)
            (now, self.flip, 2)
        } else {
            self.starved += 1;
            return (0.0, self.flip, newest, 1);
        };
        if what != 2 || flip != self.flip {
            self.starved = 0;
        }
        self.flip = flip;
        let iv = to.saturating_duration_since(at).as_secs_f64();
        if to > at {
            self.at = Some(to);
        }
        (iv, flip, newest, what)
    }

    /// The step for a frame that took `interval` (s; cut to `max`): the mean of the last
    /// `window` intervals; while they all are the frame limit's time a frame (`pace`, within
    /// 1%), that time exactly, as held. Also whether the step is that exact one.
    fn step(&mut self, interval: f64, max: f64, window: usize, pace: f64) -> (f64, bool) {
        // The frame limit's time as held: taken anew when it moves by more than a thousandth
        // (a mode or limit change), else kept (the display's measured period wanders in its
        // sixth digit, and the steps should not).
        if pace <= 0.0 {
            self.nominal = 0.0;
        } else if self.nominal <= 0.0 || (pace / self.nominal - 1.0).abs() > 1e-3 {
            self.nominal = pace;
            self.give_back();
        }
        self.intervals.push_back((interval.min(max) + std::mem::take(&mut self.pending)).max(0.0));
        while self.intervals.len() > window {
            self.intervals.pop_front();
        }
        let mean = self.intervals.iter().sum::<f64>() / self.intervals.len() as f64;
        let n = self.nominal;
        let steady = n > 0.0 && self.intervals.len() == window && self.intervals.iter().all(|i| (i / n - 1.0).abs() < 0.01);
        if !steady {
            self.give_back();
            return (mean, false);
        }
        // Exact steps: what they leave out or add against the flips' times is kept and, at
        // half a millisecond, given back through the mean with the limit's time taken anew.
        self.drift += mean - n;
        if self.drift.abs() > 0.0005 {
            self.nominal = pace;
            self.give_back();
        }
        (n, true)
    }

    /// The time the exact steps left out or added goes into the next interval taken (the
    /// mean spreads it over the window).
    fn give_back(&mut self) {
        self.pending += std::mem::take(&mut self.drift);
    }
}

/// Diagnosis (`DS_PACE_PAIR=1`): lines left to log.
static PAIR_LEFT: AtomicU32 = AtomicU32::new(0);

fn pair_diag() -> bool {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ON.get_or_init(|| std::env::var_os("DS_PACE_PAIR").is_some())
}

/// Control `gamestep pair <n>`: the next n frame steps and frame submits are logged (`DS_PACE_PAIR=1`).
pub fn pair_log(n: u32) {
    PAIR_LEFT.store(n, Relaxed);
}

/// `DS_PACE_PAIR=1`: the last frame submitted with a flip: the title's number for it (esi at
/// CRenderContext::SubmitAndFlip, one up a frame) and the flip that will show it. The flips
/// follow the submits one for one, so a submit's flip is its place in their order plus a
/// constant, learned at a submit with no flip waiting (two blank flips come before the first).
static LAST_SUBMIT: Mutex<Option<(u32, u64)>> = Mutex::new(None);
/// Submits with a flip so far, and the constant (flip number minus the submit's place; 0: not
/// learned yet).
static SUBMITS: AtomicU64 = AtomicU64::new(0);
static SUBMIT_FLIP: AtomicU64 = AtomicU64::new(0);

/// The flip that will show the frame being stepped (`DS_PACE_PAIR=1`; 0: not known): the
/// frame's number is the last submitted one's, up by the steps of the title's ring of eight
/// frame packets between them, and frame numbers and flips run in step.
fn own_flip(ring: i32) -> u64 {
    let Some((frame, flip)) = *LAST_SUBMIT.lock() else { return 0 };
    if !(0..8).contains(&ring) {
        return 0;
    }
    let ahead = (ring as u32).wrapping_sub(frame) & 7;
    flip + if ahead == 0 { 8 } else { ahead as u64 }
}

/// Diagnosis hook at CRenderContext::SubmitAndFlip's entry (rdi the context, esi the frame's
/// number, edx the display buffer, rbx the caller's submit object).
pub extern "sysv64" fn submit_hook(regs: *const u64, _arg: u32) {
    {
        let frame = unsafe { *regs.add(6) } as u32;
        let seq = SUBMITS.fetch_add(1, Relaxed) + 1;
        let (queued, shown) = (crate::hle::libs::videoout::QUEUED_FLIP.load(Relaxed), crate::gpu::FRAMES.load(Relaxed));
        if queued == shown && shown > 0 {
            let k = (queued + 1).wrapping_sub(seq);
            let was = SUBMIT_FLIP.swap(k, Relaxed);
            if was != 0 && was != k {
                crate::info!("pace pair: flips minus submits now {k} (was {was}): flips came without a frame's submit, or a submit without a flip");
            }
        }
        let k = SUBMIT_FLIP.load(Relaxed);
        *LAST_SUBMIT.lock() = (k != 0).then_some((frame, seq.wrapping_add(k)));
        // Reflex: the frame's simulation ends and its render submission starts (`gpu::latency`),
        // named by the flip that will show it.
        if k != 0 {
            crate::gpu::latency::render_submit(seq.wrapping_add(k) as u32);
        }
    }
    if PAIR_LEFT.load(Relaxed) == 0 {
        return;
    }
    PAIR_LEFT.fetch_sub(1, Relaxed);
    let (rdi, rsi, rdx, rbx) = unsafe { (*regs.add(7), *regs.add(6), *regs.add(2), *regs.add(3)) };
    let q = |a: u64| if a > 0x10000 && crate::gpu::cp::guest_readable(a, 8) { rdq(a) } else { 0 };
    let words: Vec<String> = (0..8u64).map(|i| format!("{:x}", q(rbx + i * 8))).collect();
    crate::info!(
        "pace pair: submit ctx {rdi:#x} esi {:#x} edx {:#x} obj {rbx:#x} [{}] flips queued {} shown {}",
        rsi as u32,
        rdx as u32,
        words.join(" "),
        crate::hle::libs::videoout::QUEUED_FLIP.load(Relaxed),
        crate::gpu::FRAMES.load(Relaxed)
    );
}

/// After the title's frame step returned (eboot+0xb27f7c: `mov rax,[rbx+28E8h]`; rbx the
/// application object, r13b the title's own "one fixed step" request).
pub extern "sysv64" fn after_step(regs: *const u64, _arg: u32) {
    let app = unsafe { *regs.add(3) };
    // The flip that will show this frame, by the title's ring of frame packets, whose index was
    // just moved on for it (0: not known yet).
    let own = {
        let base = BASE.load(Relaxed);
        let render = if base != 0 { rdq(base + RENDER) } else { 0 };
        let ring = if render != 0 { unsafe { ((render + 8) as *const i32).read_volatile() } } else { -1 };
        (ring, own_flip(ring))
    };
    // Reflex: its sleep and the simulation's start, before the frame reads input and simulates
    // (`gpu::latency`; the frame named by the flip that will show it, two past the last queued
    // while that is not known).
    crate::gpu::latency::simulation_start(if own.1 != 0 { own.1 as u32 } else { (crate::hle::libs::videoout::QUEUED_FLIP.load(Relaxed) + 2) as u32 });
    // (`DS_PACE_PAIR=1`: logged beside the frame's flip.)
    let own_flip = if pair_diag() {
        let (ring, own) = own;
        if PAIR_LEFT.load(Relaxed) > 0 {
            PAIR_LEFT.fetch_sub(1, Relaxed);
            crate::info!(
                "pace pair: step packet {:#x} ring {ring} its flip {own}; flips queued {} shown {}",
                unsafe { *regs.add(15) },
                crate::hle::libs::videoout::QUEUED_FLIP.load(Relaxed),
                crate::gpu::FRAMES.load(Relaxed)
            );
        }
        own
    } else {
        0
    };
    let forced = unsafe { *regs.add(13) } as u8 != 0;
    let timer = rdq(app + APP_TIMER);
    let base = BASE.load(Relaxed);
    if timer == 0 || base == 0 {
        return;
    }
    let now = Instant::now();
    let title = rdf(timer + T_STEP);
    let mut g = STATE.lock();
    let Some(s) = g.as_mut() else { return };
    let (interval, flip, newest, what) = s.clock.advance(now);
    match what {
        1 => s.starved += 1,
        2 => s.resyncs += 1,
        _ => {}
    }
    let max = setting(base, MAX_DT);
    let max = if max.is_finite() && max > 0.001 { max as f64 } else { 0.1 };
    let pace = crate::hle::libs::videoout::PACE_NS.load(Relaxed) as f64 * 1e-9;
    let (fed, exact) = s.clock.step(interval, max, crate::gpu::tune::pace_smooth() as usize, pace);
    let fixed = FIXED_FEED.load(Relaxed);
    let fed = if fixed { TITLE_STEP as f64 } else { fed };
    let feed = crate::gpu::tune::pace_step() && !forced && s.total_frames >= 8;
    let (step, phys) = if feed {
        // (Never zero: the title divides by its step. The self-play's speed, `SPEED_X100`.)
        let speed = SPEED_X100.load(Relaxed).clamp(25, 400) as f64 / 100.0;
        let step = ((fed * speed) as f32).clamp(0.001, max as f32);
        wrf(timer + T_STEP, step);
        wrf(timer + T_LAST, step);
        wrf(timer + T_ERROR, 0.0);
        // (The expected step itself stays the title's: its only readers besides the physics
        // are the developers' automation waits and a timer list absent in play, and the
        // suite's scripts keep their timing with it.)
        // The physics' fixed step: the frame limit's time a frame, so that a frame at the
        // limit is one physics step (uncapped, and with `tune pace_phys` 2: the game's step).
        let phys = match crate::gpu::tune::pace_phys() {
            _ if fixed => TITLE_STEP,
            0 => TITLE_STEP,
            2 => step.min(PHYS_MAX),
            // The limit's time a frame while the limit is held, and for a limit of 60 frames
            // a second or fewer also while it is not (several steps a frame then, as the
            // title does); a time over the title's longest step, 1/30 s, in equal parts.
            _ if s.clock.nominal > 0.0 && (exact || s.clock.nominal as f32 >= TITLE_STEP * 0.99) => {
                let n = s.clock.nominal as f32;
                n / (n / (PHYS_MAX * 1.01)).ceil().max(1.0)
            }
            // (No limit, or a faster one that is not held: the title's step while frames take
            // longer, one step a frame above 60.)
            _ => step.min(TITLE_STEP),
        };
        let physics = rdq(base + PHYSICS);
        if physics != 0 {
            wrf(physics + PHYSICS_STEP, phys);
        }
        FIXED_STEP.store(step.min(TITLE_STEP).to_bits(), Relaxed);
        s.fed += 1;
        (step, phys)
    } else {
        FIXED_STEP.store(TITLE_STEP.to_bits(), Relaxed);
        (title, rdf(base + EXPECTED))
    };
    let phase = crate::hle::libs::videoout::latest_flip().map_or(0.0, |(_, at, _)| now.saturating_duration_since(at).as_secs_f32());
    let rec = Rec { at: now, title, step, interval: interval as f32, flip, newest, phase, phys, pos: crate::camhook::player_pos(), yaw: crate::camhook::camera_yaw(), own_flip, shown: None };
    // (The frames before it whose flips have been shown since: when.)
    if own_flip != 0 {
        for r in s.recs.iter_mut().rev().take(16) {
            if r.shown.is_none() && r.own_flip != 0 {
                r.shown = crate::hle::libs::videoout::flip_time(r.own_flip);
            }
        }
    }
    s.frames += 1;
    s.game += step as f64;
    s.total_frames += 1;
    s.total_game += step as f64;
    s.exact += (feed && exact) as u64;
    s.first.get_or_insert(now);
    s.last = Some(now);
    let key = (step * 1e5).round() as u32;
    match s.steps.iter().position(|e| e.0 == key) {
        Some(i) => s.steps[i].1 += 1,
        None if s.steps.len() < 64 => s.steps.push((key, 1)),
        None => {}
    }
    s.recs.push_back(rec);
    while s.recs.len() > 16384 {
        s.recs.pop_front();
    }
}

/// The frame steps since the last report (control `gamestep`), counted afresh: frames, game
/// time against real time, the steps by value.
pub fn report() -> String {
    let base = BASE.load(Relaxed);
    let mut g = STATE.lock();
    let Some(s) = g.as_mut() else { return "gamestep: no frame step hook".into() };
    let real = match (s.first, s.last) {
        (Some(a), Some(b)) => b.duration_since(a).as_secs_f64(),
        _ => 0.0,
    };
    // (The first frame's step belongs to the time before the window.)
    let first_step = s.recs.iter().rev().nth(s.frames.saturating_sub(1) as usize).map_or(0.0, |r| r.step as f64);
    let game = s.game - first_step;
    let mut steps = std::mem::take(&mut s.steps);
    steps.sort_by(|a, b| b.1.cmp(&a.1));
    let list: Vec<String> = steps.iter().take(8).map(|(k, n)| format!("{:.2} ms x{n}", *k as f64 / 100.0)).collect();
    // How far the picture's time strays from the flips' (0: every frame moves by exactly the
    // time it is shown after the one before).
    let window: Vec<Rec> = s.recs.iter().rev().take(s.frames as usize).rev().copied().collect();
    let stray = match own_stray(&window) {
        // (`DS_PACE_PAIR=1`: against the flip that showed each frame.)
        Some((frames, sd, per, mean_lag)) => format!("{:.2} ms from its own flips' (a frame's step {:.2} ms off the time it is shown after the frame before; {frames} frames, shown {:.1} ms after their step)", sd * 1e3, per * 1e3, mean_lag * 1e3),
        None => format!("{:.2} / {:.2} ms from the flips' (a frame shown one / two flips after its step)", stray(&window, 1) * 1e3, stray(&window, 2) * 1e3),
    };
    let rate = title_rate();
    let settings = if base != 0 { format!("gameMinDeltaTime {}, gameMaxDeltaTime {}, tolerance {}", setting(base, MIN_DT), setting(base, MAX_DT), setting(base, TOLERANCE)) } else { String::new() };
    let out = format!(
        "gamestep: {} frames in {:.3} s ({:.2} a second); game time {:.3} s: x{:.4} of real time; steps {}{}; {} at the frame limit's exact step ({:.3} ms held); the picture's time strays {stray}; the title's refresh rate {} and vsync interval {}; {settings}",
        s.frames,
        real,
        if real > 0.0 { (s.frames.saturating_sub(1)) as f64 / real } else { 0.0 },
        game,
        if real > 0.0 { game / real } else { 0.0 },
        list.join(", "),
        if steps.len() > 8 { format!(" and {} more values", steps.len() - 8) } else { String::new() },
        s.exact,
        s.clock.nominal * 1e3,
        rate.map_or("?".into(), |r| format!("{}", r.0)),
        rate.map_or("?".into(), |r| format!("{}", r.1)),
    );
    s.frames = 0;
    s.game = 0.0;
    s.exact = 0;
    s.first = None;
    s.last = None;
    out
}

/// How far the picture's time strays from the flips' over `recs` (s): the standard deviation
/// of the game time each frame holds minus the time of the flip that shows it, taken as `d`
/// flips after the one its step came from. Every frame moving by exactly the time it is shown
/// after the frame before is 0; steps that lag or lead the flips' spacing add to it.
fn stray(recs: &[Rec], d: usize) -> f64 {
    if recs.len() < d + 8 {
        return 0.0;
    }
    let (mut game, mut shown) = (Vec::with_capacity(recs.len()), Vec::with_capacity(recs.len()));
    let (mut a, mut b) = (0.0f64, 0.0f64);
    for r in recs {
        a += r.step as f64;
        b += r.interval as f64;
        game.push(a);
        shown.push(b);
    }
    let e: Vec<f64> = (0..recs.len() - d).map(|k| game[k] - shown[k + d]).collect();
    let mean = e.iter().sum::<f64>() / e.len() as f64;
    (e.iter().map(|v| (v - mean).powi(2)).sum::<f64>() / e.len() as f64).sqrt()
}

/// With each frame's own flip known (`DS_PACE_PAIR=1`): over the frames of `recs` that were
/// shown, how many, the standard deviation (s) of the game time a frame holds minus the time
/// it was shown, the standard deviation of a frame's step minus the time between its flip and
/// the frame before's (what the eye sees as uneven motion), and the mean time from a frame's
/// step to its flip. None without the pairing.
fn own_stray(recs: &[Rec]) -> Option<(usize, f64, f64, f64)> {
    let t0 = recs.first()?.at;
    let mut game = 0.0f64;
    let mut pts: Vec<(f64, f64, f64, f64)> = Vec::new(); // game time, shown, step, lag
    for r in recs {
        game += r.step as f64;
        if let Some(sh) = r.shown {
            let shown = sh.saturating_duration_since(t0).as_secs_f64() - t0.saturating_duration_since(sh).as_secs_f64();
            pts.push((game, shown, r.step as f64, sh.saturating_duration_since(r.at).as_secs_f64()));
        }
    }
    if pts.len() < 16 {
        return None;
    }
    let sd = |v: &[f64]| {
        let m = v.iter().sum::<f64>() / v.len() as f64;
        (v.iter().map(|x| (x - m).powi(2)).sum::<f64>() / v.len() as f64).sqrt()
    };
    let e: Vec<f64> = pts.iter().map(|p| p.0 - p.1).collect();
    let per: Vec<f64> = pts.windows(2).map(|w| (w[1].0 - w[0].0) - (w[1].1 - w[0].1)).collect();
    let lag = pts.iter().map(|p| p.3).sum::<f64>() / pts.len() as f64;
    Some((pts.len(), sd(&e), sd(&per), lag))
}

/// Status line: how the game's step is set.
pub fn status() -> String {
    let g = STATE.lock();
    let Some(s) = g.as_ref() else { return String::new() };
    if s.total_frames == 0 {
        return String::new();
    }
    format!(
        "    game step (`tune pace_step` {}): {} frame steps, {} set by the port from the flips' times (mean of {} frames); {} found no new flip, {} skips; game time {:.1} s\n",
        crate::gpu::tune::pace_step() as u8,
        s.total_frames,
        s.fed,
        crate::gpu::tune::pace_smooth(),
        s.starved,
        s.resyncs,
        s.total_game
    )
}

/// The last `n` frame steps, one a line (control `gamestep log <n>`): the real time since the
/// frame step before (ms), the title's step and the step run (ms), the time between flips
/// taken, the flip it came from against the newest and how long ago that one was shown, the
/// physics' step, the player's move over the step (m/s of game time).
pub fn log(n: usize) -> String {
    let g = STATE.lock();
    let Some(s) = g.as_ref() else { return String::new() };
    let recs: Vec<Rec> = s.recs.iter().rev().take(n + 1).rev().copied().collect();
    let mut out = String::new();
    for w in recs.windows(2) {
        let (a, b) = (w[0], w[1]);
        let speed = match (a.pos, b.pos) {
            (Some(p), Some(q)) if b.step > 0.0 => format!("{:.3}", dist(p, q) / b.step),
            _ => "-".into(),
        };
        out += &format!(
            "  real {:7.3} ms, title's step {:7.3}, step {:7.3}, flips' time {:7.3}, flip {} of {} ({:5.2} ms ago), physics {:7.3}, speed {speed}\n",
            b.at.duration_since(a.at).as_secs_f64() * 1e3,
            b.title as f64 * 1e3,
            b.step as f64 * 1e3,
            b.interval as f64 * 1e3,
            b.flip,
            b.newest,
            b.phase as f64 * 1e3,
            b.phys as f64 * 1e3,
        );
    }
    out
}

/// Control `gamestep dump <n>`: the last n frame steps as CSV in the run directory
/// (`gamestep-<flip>.csv`: the step's time, the title's and the port's step, the time between
/// flips taken, the flip it came from and the newest then, and with `DS_PACE_PAIR=1` the flip
/// that showed the frame and when, times in ms from the first step), for offline study of
/// step rules (PACING.md "Open" 2). Returns the file written.
pub fn dump(n: usize) -> String {
    let g = STATE.lock();
    let Some(s) = g.as_ref() else { return "gamestep dump: no frame step hook".into() };
    let recs: Vec<Rec> = s.recs.iter().rev().take(n).rev().copied().collect();
    drop(g);
    let Some(t0) = recs.first().map(|r| r.at) else { return "gamestep dump: no frame steps".into() };
    let ms = |t: Instant| t.saturating_duration_since(t0).as_secs_f64() * 1e3 - t0.saturating_duration_since(t).as_secs_f64() * 1e3;
    let mut out = String::from("at_ms,title_ms,step_ms,interval_ms,flip,newest,own_flip,shown_ms\n");
    for r in &recs {
        out += &format!(
            "{:.4},{:.4},{:.4},{:.4},{},{},{},{}\n",
            ms(r.at),
            r.title as f64 * 1e3,
            r.step as f64 * 1e3,
            r.interval as f64 * 1e3,
            r.flip,
            r.newest,
            r.own_flip,
            r.shown.map_or(String::new(), |t| format!("{:.4}", ms(t)))
        );
    }
    let Some(dir) = crate::run_dir() else { return "gamestep dump: no run directory".into() };
    let path = dir.join(format!("gamestep-{}.csv", crate::gpu::FRAMES.load(Relaxed)));
    match std::fs::write(&path, out) {
        Ok(()) => format!("gamestep dump: {} frame steps written to {}", recs.len(), path.display()),
        Err(e) => format!("gamestep dump: {e}"),
    }
}

fn dist(a: [f32; 3], b: [f32; 3]) -> f32 {
    ((a[0] - b[0]).powi(2) + (a[1] - b[1]).powi(2) + (a[2] - b[2]).powi(2)).sqrt()
}

/// The number of frame steps so far and their game time (s).
fn totals() -> (u64, f64) {
    STATE.lock().as_ref().map_or((0, 0.0), |s| (s.total_frames, s.total_game))
}

/// The newest frame step's record.
fn newest() -> Option<Rec> {
    STATE.lock().as_ref().and_then(|s| s.recs.back().copied())
}

/// Control `walktest <x> <y> <ms> [name]`: holds the left stick at the tilt for `ms` and
/// measures the player's speed over the hold's steady part (from 0.6 s in to 0.1 s before its
/// end): the distance, the real time and the game time between the two frames, and how evenly
/// the player moved per frame (each frame's move over its step). Runs on a thread of its own;
/// the result is logged as `walktest <name>: ...`.
pub fn walktest(x: i32, y: i32, ms: u64, name: String) {
    std::thread::Builder::new()
        .name("walktest".into())
        .spawn(move || {
            let hold = Duration::from_millis(ms);
            crate::input::hold_stick(false, x, y, hold);
            let t0 = Instant::now();
            std::thread::sleep(Duration::from_millis(600.min(ms / 3)));
            let (f0, g0) = totals();
            let Some(a) = newest() else {
                crate::warn!("walktest {name}: no frame steps seen");
                return;
            };
            let until = hold.saturating_sub(Duration::from_millis(100));
            // (One read of the clock: two could straddle `until` and underflow.)
            std::thread::sleep(until.saturating_sub(t0.elapsed()));
            let (f1, g1) = totals();
            let Some(b) = newest() else { return };
            let (Some(p), Some(q)) = (a.pos, b.pos) else {
                crate::warn!("walktest {name}: the player's position is not known (no gameplay camera update seen)");
                return;
            };
            let real = b.at.duration_since(a.at).as_secs_f64();
            let game = g1 - g0;
            let d = dist(p, q) as f64;
            // Each frame's move over its step, across the window.
            let per: Vec<f64> = {
                let g = STATE.lock();
                let recs: Vec<Rec> = g.as_ref().map(|s| s.recs.iter().rev().take((f1 - f0) as usize + 1).rev().copied().collect()).unwrap_or_default();
                recs.windows(2).filter_map(|w| Some(dist(w[0].pos?, w[1].pos?) as f64 / (w[1].step as f64).max(1e-6))).collect()
            };
            let mean = per.iter().sum::<f64>() / per.len().max(1) as f64;
            let sd = (per.iter().map(|v| (v - mean).powi(2)).sum::<f64>() / per.len().max(1) as f64).sqrt();
            crate::info!(
                "walktest {name}: {d:.3} m in {real:.3} s of real time ({:.3} m/s) and {game:.3} s of game time ({:.3} m/s); game time x{:.4} of real; {} frames ({:.1} a second); per frame {mean:.3} m/s of game time, spread {:.1}%",
                d / real.max(1e-9),
                d / game.max(1e-9),
                game / real.max(1e-9),
                f1 - f0,
                (f1 - f0) as f64 / real.max(1e-9),
                100.0 * sd / mean.max(1e-9)
            );
        })
        .ok();
}

/// Control `turntest <x> <ms> [name]`: holds the right stick at the horizontal tilt for `ms`
/// and measures the camera's turn over the hold's steady part (from 1.2 s in, past the
/// stick's ramp, to 0.1 s before its end): the angle, the real time and the game time between
/// the two frames. The camera turns by the stick times the frame's step, so the turn per real
/// second is the game's speed. Logged as `turntest <name>: ...`.
pub fn turntest(x: i32, ms: u64, name: String) {
    std::thread::Builder::new()
        .name("turntest".into())
        .spawn(move || {
            let hold = Duration::from_millis(ms);
            crate::input::hold_stick(true, x, 0, hold);
            let t0 = Instant::now();
            std::thread::sleep(Duration::from_millis(1200.min(ms / 2)));
            let (f0, g0) = totals();
            let until = hold.saturating_sub(Duration::from_millis(100));
            // (One read of the clock: two could straddle `until` and underflow.)
            std::thread::sleep(until.saturating_sub(t0.elapsed()));
            let (f1, g1) = totals();
            let recs: Vec<Rec> = STATE.lock().as_ref().map(|s| s.recs.iter().rev().take((f1 - f0) as usize + 1).rev().copied().collect()).unwrap_or_default();
            // The yaw's change frame to frame, unwrapped (it wraps at pi).
            let pi = std::f64::consts::PI;
            let mut turned = 0.0f64;
            let mut per = Vec::new();
            for w in recs.windows(2) {
                let (Some(a), Some(b)) = (w[0].yaw, w[1].yaw) else {
                    crate::warn!("turntest {name}: the camera's yaw is not known (no gameplay camera update seen)");
                    return;
                };
                let mut d = b as f64 - a as f64;
                d -= (d / (2.0 * pi)).round() * 2.0 * pi;
                turned += d;
                per.push(d / (w[1].step as f64).max(1e-6));
            }
            let (Some(a), Some(b)) = (recs.first(), recs.last()) else { return };
            let real = b.at.duration_since(a.at).as_secs_f64();
            let game = g1 - g0;
            let mean = per.iter().sum::<f64>() / per.len().max(1) as f64;
            let sd = (per.iter().map(|v| (v - mean).powi(2)).sum::<f64>() / per.len().max(1) as f64).sqrt();
            crate::info!(
                "turntest {name}: {:.4} rad in {real:.3} s of real time ({:.4} rad/s) and {game:.3} s of game time ({:.4} rad/s); game time x{:.4} of real; {} frames ({:.1} a second); per frame {:.4} rad/s of game time, spread {:.2}%",
                turned,
                turned / real.max(1e-9),
                turned / game.max(1e-9),
                game / real.max(1e-9),
                f1 - f0,
                (f1 - f0) as f64 / real.max(1e-9),
                mean,
                100.0 * sd / mean.abs().max(1e-9)
            );
        })
        .ok();
}

#[cfg(test)]
mod tests {
    use super::*;

    fn run(clock: &mut Clock, intervals: &[f64], pace: f64) -> Vec<(f64, bool)> {
        intervals.iter().map(|&i| clock.step(i, 0.1, 8, pace)).collect()
    }

    #[test]
    fn the_step_is_exact_while_the_limit_is_held() {
        // 120 Hz, frames two refreshes: the display's period measured with a wander in its
        // sixth digit.
        let t = 2.0 / 119.997;
        let mut c = Clock::default();
        let ivs: Vec<f64> = (0..600).map(|k| t * (1.0 + 2e-6 * ((k % 7) as f64 - 3.0))).collect();
        let out = run(&mut c, &ivs, t);
        // After the window filled, every step is the held time, bit for bit.
        assert!(out[8..].iter().all(|&(s, exact)| exact && s == t), "{:?}", &out[..12]);
    }

    #[test]
    fn a_late_frame_is_spread_and_no_time_is_lost() {
        let t = 1.0 / 60.0;
        let mut c = Clock::default();
        let mut ivs = vec![t; 40];
        ivs[20] = 2.0 * t;
        let out = run(&mut c, &ivs, t);
        let total: f64 = out.iter().map(|o| o.0).sum();
        // (The first frames' mean is over fewer intervals: the same values.)
        assert!((total - ivs.iter().sum::<f64>()).abs() < 1e-9, "{total}");
        // The late frame's extra time comes over eight frames, an eighth each.
        assert!(out[20..28].iter().all(|&(s, exact)| !exact && (s - t * 1.125).abs() < 1e-12));
        assert!(out[28..].iter().all(|&(s, exact)| exact && s == t));
    }

    #[test]
    fn the_steps_follow_real_time_at_any_rate() {
        for hz in [30.0, 48.0, 59.94, 97.0, 144.0, 165.0, 240.0] {
            let mut c = Clock::default();
            // Frames of uneven lengths around the rate, no frame limit.
            let ivs: Vec<f64> = (0..2000).map(|k| (1.0 + 0.3 * (((k * 7) % 5) as f64 - 2.0) / 2.0) / hz).collect();
            let out = run(&mut c, &ivs, 0.0);
            let (game, real): (f64, f64) = (out.iter().map(|o| o.0).sum(), ivs.iter().sum());
            // (The window's last frames are still owed: under eight frames' time.)
            assert!((game / real - 1.0).abs() < 0.005, "{hz}: {game} {real}");
        }
    }

    #[test]
    fn the_held_time_follows_a_slow_change_without_losing_time() {
        // The display's period 30 parts in a million off the held one: the exact steps'
        // drift comes back in small pieces.
        let t = 1.0 / 60.0;
        let mut c = Clock::default();
        let ivs = vec![t * (1.0 + 30e-6); 100_000];
        let out = run(&mut c, &ivs, t);
        let (game, real): (f64, f64) = (out.iter().map(|o| o.0).sum(), ivs.iter().sum());
        assert!((game - real).abs() < 0.001, "{game} {real}");
        // No step is off the held time by more than a hundredth.
        assert!(out[8..].iter().all(|o| (o.0 / t - 1.0).abs() < 0.01));
    }

    #[test]
    fn a_long_time_is_cut_as_the_title_cuts_it() {
        let mut c = Clock::default();
        let out = run(&mut c, &[0.016, 0.016, 2.5, 0.016], 0.0);
        assert!(out.iter().map(|o| o.0).sum::<f64>() < 0.2);
    }
}
