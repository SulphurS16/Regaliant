//! Host input aggregation for the PS5 pad: an XInput controller (each button
//! on the PS5 pad's button in its place unless the layout remaps it), the
//! output window's keyboard and mouse in a Dark Souls III-style PC layout, and
//! a scripted timeline for reproducible runs.
//!
//! Keyboard and mouse (`Layout`, window focused): the layout is read from
//! local-data/controls.txt, written with the defaults (`DEFAULT_LAYOUT`) on
//! first start; each line binds a key, mouse button or wheel direction
//! (optionally with Shift+, Ctrl+ or Alt+) to a pad button, a stick
//! direction or `walk`. A binding with modifiers wins over the plain one of
//! the same input while its modifiers are held (Shift+Mouse1 = R2 instead of
//! Mouse1 = R1). Wheel notches are d-pad taps. Mouse motion (raw counts,
//! `window`) turns the camera through the right stick: its speed over the
//! interval since the last pad read sets the deflection (`mouse_look`).
//!
//! Scripted input (`DS_INPUT`): `flip:buttons[:hold]` entries separated by
//! `;`, e.g. `DS_INPUT=700:cross;760:down:3;800:cross`. `buttons` joins names
//! with `+` (cross, circle, square, triangle, up, down, left, right, l1, r1,
//! l2, r2, l3, r3, options, touchpad, ls_up/down/left/right,
//! rs_up/down/left/right). Each press lasts `hold` GPU flips (default 6).

use parking_lot::Mutex;
use std::collections::VecDeque;
use std::sync::OnceLock;
use std::sync::atomic::Ordering::Relaxed;
use std::time::{Duration, Instant};
use crate::platform::controller::{XINPUT_GAMEPAD, XINPUT_STATE, XINPUT_VIBRATION, XInputGetState, XInputSetState};

#[path = "input_state.rs"]
mod state;
pub use state::*;

/// The pad's buttons as XInput reports them (an Xbox controller's names): the layout file's
/// name (`PadA = cross`), the menu's, and the XInput button bit (0: a trigger).
pub const PAD_BUTTONS: &[(&str, &str, u16)] = &[
    ("PadA", "A", 0x1000),
    ("PadB", "B", 0x2000),
    ("PadX", "X", 0x4000),
    ("PadY", "Y", 0x8000),
    ("PadLB", "LB", 0x0100),
    ("PadRB", "RB", 0x0200),
    ("PadLT", "LT", 0),
    ("PadRT", "RT", 0),
    ("PadLS", "LS", 0x0040),
    ("PadRS", "RS", 0x0080),
    ("PadView", "View", 0x0020),
    ("PadMenu", "Menu", 0x0010),
    ("PadUp", "D-pad Up", 0x0001),
    ("PadDown", "D-pad Down", 0x0002),
    ("PadLeft", "D-pad Left", 0x0004),
    ("PadRight", "D-pad Right", 0x0008),
];
pub const PAD_LT: usize = 6;
pub const PAD_RT: usize = 7;
/// Each pad button's target unless the layout remaps it: the PS5 pad's button in its place.
pub const PAD_DEFAULT: [u32; 16] = [BTN_CROSS, BTN_CIRCLE, BTN_SQUARE, BTN_TRIANGLE, BTN_L1, BTN_R1, BTN_L2, BTN_R2, BTN_L3, BTN_R3, BTN_TOUCHPAD, BTN_OPTIONS, BTN_UP, BTN_DOWN, BTN_LEFT, BTN_RIGHT];

/// A button target's bit (`cross`, `up`, ...; None for stick directions and unknown names).
pub fn button_bit(target: &str) -> Option<u32> {
    parse_name(target)?.ok()
}

/// A pad button's place in `PAD_BUTTONS` by its layout file name.
fn pad_index(name: &str) -> Option<usize> {
    PAD_BUTTONS.iter().position(|(n, _, _)| n.eq_ignore_ascii_case(name.trim()))
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Stick {
    LUp,
    LDown,
    LLeft,
    LRight,
    RUp,
    RDown,
    RLeft,
    RRight,
}

#[derive(Clone, Debug, PartialEq, Eq)]
struct Step {
    flip: u64,
    hold: u64,
    buttons: u32,
    sticks: Vec<Stick>,
}

fn parse_name(n: &str) -> Option<Result<u32, Stick>> {
    Some(match n {
        "cross" | "x" => Ok(BTN_CROSS),
        "circle" | "o" => Ok(BTN_CIRCLE),
        "square" => Ok(BTN_SQUARE),
        "triangle" => Ok(BTN_TRIANGLE),
        "up" => Ok(BTN_UP),
        "down" => Ok(BTN_DOWN),
        "left" => Ok(BTN_LEFT),
        "right" => Ok(BTN_RIGHT),
        "l1" => Ok(BTN_L1),
        "r1" => Ok(BTN_R1),
        "l2" => Ok(BTN_L2),
        "r2" => Ok(BTN_R2),
        "l3" => Ok(BTN_L3),
        "r3" => Ok(BTN_R3),
        "options" | "start" => Ok(BTN_OPTIONS),
        "touchpad" => Ok(BTN_TOUCHPAD),
        "ls_up" => Err(Stick::LUp),
        "ls_down" => Err(Stick::LDown),
        "ls_left" => Err(Stick::LLeft),
        "ls_right" => Err(Stick::LRight),
        "rs_up" => Err(Stick::RUp),
        "rs_down" => Err(Stick::RDown),
        "rs_left" => Err(Stick::RLeft),
        "rs_right" => Err(Stick::RRight),
        _ => return None,
    })
}

fn parse_script(s: &str) -> Result<Vec<Step>, String> {
    let mut steps = Vec::new();
    for entry in s.split(';').map(str::trim).filter(|e| !e.is_empty()) {
        let parts: Vec<&str> = entry.split(':').collect();
        if parts.len() < 2 || parts.len() > 3 {
            return Err(format!("bad entry '{entry}' (want flip:buttons[:hold])"));
        }
        let flip = parts[0].trim().parse::<u64>().map_err(|_| format!("bad flip in '{entry}'"))?;
        let hold = match parts.get(2) {
            Some(h) => h.trim().parse::<u64>().map_err(|_| format!("bad hold in '{entry}'"))?.max(1),
            None => 6,
        };
        let mut step = Step { flip, hold, buttons: 0, sticks: Vec::new() };
        for name in parts[1].split('+').map(|n| n.trim().to_ascii_lowercase()) {
            match parse_name(&name) {
                Some(Ok(b)) => step.buttons |= b,
                Some(Err(st)) => step.sticks.push(st),
                None => return Err(format!("unknown button '{name}' in '{entry}'")),
            }
        }
        steps.push(step);
    }
    Ok(steps)
}

fn script() -> &'static [Step] {
    static SCRIPT: OnceLock<Vec<Step>> = OnceLock::new();
    SCRIPT.get_or_init(|| match std::env::var("DS_INPUT") {
        Ok(s) => match parse_script(&s) {
            Ok(v) => {
                crate::info!("input script: {} steps", v.len());
                v
            }
            Err(e) => {
                crate::error!("DS_INPUT ignored: {e}");
                Vec::new()
            }
        },
        Err(_) => Vec::new(),
    })
}

fn apply_stick(st: &mut PadState, s: Stick) {
    match s {
        Stick::LUp => st.ly = 0,
        Stick::LDown => st.ly = 255,
        Stick::LLeft => st.lx = 0,
        Stick::LRight => st.lx = 255,
        Stick::RUp => st.ry = 0,
        Stick::RDown => st.ry = 255,
        Stick::RLeft => st.rx = 0,
        Stick::RRight => st.rx = 255,
    }
}

/// Presses scheduled at run time (control channel), relative to the flip at
/// which they were requested.
static LIVE: parking_lot::Mutex<Vec<Step>> = parking_lot::Mutex::new(Vec::new());

/// Schedules a press of `names` (joined with '+') for `hold` flips starting
/// at the next flip.
pub fn press_now(names: &str, hold: u64) -> Result<(), String> {
    note_device(DEVICE_PAD);
    let flip = crate::gpu::FRAMES.load(std::sync::atomic::Ordering::Relaxed) + 1;
    let mut steps = parse_script(&format!("{flip}:{names}:{}", hold.max(1)))?;
    let mut live = LIVE.lock();
    live.retain(|s| s.flip + s.hold > flip);
    live.append(&mut steps);
    Ok(())
}

/// Presses held for a wall-clock duration (control channel `press <b> 4s`):
/// frame rates differ about sixfold between runs with and without GPU
/// execution, so holds counted in flips do not transfer between them. Each
/// is held until its time is up and the game has read it in `MIN_FLIPS`
/// flips (the flip it was first read at, u64::MAX before): a game frozen
/// for the whole hold (a shader compiled in place on a slow PC, seconds)
/// still sees it.
/// (until, the step, the flip it was first read at, the flips it is held at least.)
static LIVE_TIMED: parking_lot::Mutex<Vec<(std::time::Instant, Step, u64, u64)>> = parking_lot::Mutex::new(Vec::new());

/// The flips a timed press stays held at least, from the first the game read it at.
const MIN_FLIPS: u64 = 3;

/// Timed presses and sticks held for their time's flips at the title's 59.94 frames a second
/// instead of the wall clock's time (the shader preparation's levels, a fast-forward:
/// `gamestep::FIXED_FEED`): each frame steps the game as one at 60 a second does, so a hold
/// covers the same game time at any frame rate.
pub static FLIP_TIMED: std::sync::atomic::AtomicBool = std::sync::atomic::AtomicBool::new(false);

/// The flips a hold of `dur` lasts under `FLIP_TIMED` (at least one).
fn flips_for(dur: std::time::Duration) -> u64 {
    ((dur.as_secs_f64() * 59.94).ceil() as u64).max(1)
}

/// Under `FLIP_TIMED`, a timed press or a stick the title's automation asked for
/// (`dsport-control: press <buttons> <n>ms` or `stick l|r <x> <y> <ms>`) held from the frame it
/// was printed in (true), instead of from the control file's next poll, up to 250 ms later (30
/// frames of the fast-forward: its fights' presses would have bunched). Other lines: false.
pub fn control_now(line: &str) -> bool {
    if !FLIP_TIMED.load(std::sync::atomic::Ordering::Relaxed) {
        return false;
    }
    let w: Vec<&str> = line.split_whitespace().collect();
    let r = match w.as_slice() {
        ["press", names, h] if h.ends_with("ms") => match h.trim_end_matches("ms").parse::<u64>() {
            Ok(ms) => press_for(names, std::time::Duration::from_millis(ms)),
            Err(_) => return false,
        },
        ["stick", side @ ("l" | "r"), x, y, ms] => match (x.parse::<i32>(), y.parse::<i32>(), ms.parse::<u64>()) {
            (Ok(x), Ok(y), Ok(ms)) => {
                hold_stick(*side == "r", x, y, std::time::Duration::from_millis(ms));
                Ok(())
            }
            _ => return false,
        },
        _ => return false,
    };
    match r {
        Ok(()) => crate::info!("control: '{line}' at flip {} (at once)", crate::gpu::FRAMES.load(std::sync::atomic::Ordering::Relaxed)),
        Err(e) => crate::warn!("control: '{line}': {e}"),
    }
    true
}

/// The menu pointer's presses (`press_quiet`): the game's alone (the system text box takes the
/// pad's and the control channel's, never these: a click or a hover step there would cancel
/// or move it).
static QUIET_TIMED: parking_lot::Mutex<Vec<(std::time::Instant, Step)>> = parking_lot::Mutex::new(Vec::new());

/// `press_for` without making the pad the device in use (the menu pointer's presses).
pub fn press_quiet(names: &str, dur: std::time::Duration) -> Result<(), String> {
    let steps = parse_script(&format!("0:{names}:1"))?;
    let now = std::time::Instant::now();
    let mut live = QUIET_TIMED.lock();
    live.retain(|x| x.0 > now);
    live.extend(steps.into_iter().map(|s| (now + dur, s)));
    Ok(())
}

/// Presses `names` from now for `dur`.
pub fn press_for(names: &str, dur: std::time::Duration) -> Result<(), String> {
    let steps = parse_script(&format!("0:{names}:1"))?;
    note_device(DEVICE_PAD);
    let now = std::time::Instant::now();
    let mut live = LIVE_TIMED.lock();
    let flip = crate::gpu::FRAMES.load(std::sync::atomic::Ordering::Relaxed);
    live.retain(|x| x.0 > now || x.2 == u64::MAX || flip < x.2 + x.3);
    let (until, flips) = if FLIP_TIMED.load(std::sync::atomic::Ordering::Relaxed) { (now, flips_for(dur).max(MIN_FLIPS)) } else { (now + dur, MIN_FLIPS) };
    live.extend(steps.into_iter().map(|s| (until, s, u64::MAX, flips)));
    Ok(())
}

/// A press held for `flips` of the game's frames from the first it reads it at, whatever their
/// time (the self-play's menu moves: a hold that spans a long frame lets a menu repeat the
/// direction, 2026-11-21).
pub fn press_flips(names: &str, flips: u64) -> Result<(), String> {
    let steps = parse_script(&format!("0:{names}:1"))?;
    note_device(DEVICE_PAD);
    let now = std::time::Instant::now();
    let mut live = LIVE_TIMED.lock();
    let flip = crate::gpu::FRAMES.load(std::sync::atomic::Ordering::Relaxed);
    live.retain(|x| x.0 > now || x.2 == u64::MAX || flip < x.2 + x.3);
    live.extend(steps.into_iter().map(|s| (now, s, u64::MAX, flips.max(1))));
    Ok(())
}

/// Sticks held at a tilt by the control channel (`hold_stick`): until when,
/// right stick, x, y, and under `FLIP_TIMED` the flip it was first read at and its flips (0:
/// held by the wall clock).
static STICKS: parking_lot::Mutex<Vec<(std::time::Instant, bool, i32, i32, u64, u64)>> = parking_lot::Mutex::new(Vec::new());

/// Diagnosis (control channel `stick`): holds the left or right stick at
/// (x, y) from the centre (-128..127, y down) for `dur`.
pub fn hold_stick(right: bool, x: i32, y: i32, dur: std::time::Duration) {
    // (A stick tilted past its dead zone makes the pad the device in use, as the pad's own does.)
    if x.abs() > 24 || y.abs() > 24 {
        note_device(DEVICE_PAD);
    }
    let now = std::time::Instant::now();
    let flip = crate::gpu::FRAMES.load(std::sync::atomic::Ordering::Relaxed);
    let mut g = STICKS.lock();
    g.retain(|s| s.0 > now || (s.5 > 0 && (s.4 == u64::MAX || flip < s.4 + s.5)));
    let flips = if FLIP_TIMED.load(std::sync::atomic::Ordering::Relaxed) { flips_for(dur) } else { 0 };
    g.push((if flips > 0 { now } else { now + dur }, right, x.clamp(-128, 127), y.clamp(-128, 127), u64::MAX, flips));
}

/// The scripted presses and sticks into `st`; `quiet`: the menu pointer's presses too.
fn scripted(st: &mut PadState, flip: u64, quiet: bool) {
    let now = std::time::Instant::now();
    if quiet {
        for (until, step) in QUIET_TIMED.lock().iter() {
            if now < *until {
                st.buttons |= step.buttons;
                for &s in &step.sticks {
                    apply_stick(st, s);
                }
            }
        }
    }
    for (until, right, x, y, first, flips) in STICKS.lock().iter_mut() {
        let on = if *flips > 0 {
            if *first == u64::MAX {
                *first = flip;
            }
            flip < *first + *flips
        } else {
            now < *until
        };
        if on {
            let (right, x, y) = (*right, *x, *y);
            let (ax, ay) = ((128 + x) as u8, (128 + y) as u8);
            if right {
                (st.rx, st.ry) = (ax, ay);
            } else {
                (st.lx, st.ly) = (ax, ay);
            }
        }
    }
    for (until, step, first, flips) in LIVE_TIMED.lock().iter_mut() {
        if now < *until || *first == u64::MAX || flip < *first + *flips {
            if *first == u64::MAX {
                *first = flip;
            }
            st.buttons |= step.buttons;
            for &s in &step.sticks {
                apply_stick(st, s);
            }
        }
    }
    for step in LIVE.lock().iter() {
        if flip >= step.flip && flip < step.flip + step.hold {
            st.buttons |= step.buttons;
            for &s in &step.sticks {
                apply_stick(st, s);
            }
        }
    }
    static LOGGED: std::sync::atomic::AtomicUsize = std::sync::atomic::AtomicUsize::new(0);
    for (i, step) in script().iter().enumerate() {
        if flip >= step.flip && flip < step.flip + step.hold {
            st.buttons |= step.buttons;
            for &s in &step.sticks {
                apply_stick(st, s);
            }
            if LOGGED.fetch_max(i + 1, std::sync::atomic::Ordering::Relaxed) <= i {
                crate::info!("input script: flip {flip} step {i} ({:#x} {:?}) for {} flips", step.buttons, step.sticks, step.hold);
            }
        }
    }
}

/// The system's controllers, keyboard and mouse are not read (the shader preparation's self-play,
/// `prepplay`: it plays by its own presses while the player may be using the PC); the presses
/// and sticks given through `press_for`, `hold_stick` and the control channel still reach the game.
pub static SYSTEM_OFF: std::sync::atomic::AtomicBool = std::sync::atomic::AtomicBool::new(false);

/// The XInput controllers seen: connected at the last ask, their report number when found,
/// whether they changed since (in use), and when a missing one is asked for again.
struct Xi {
    on: [bool; 4],
    first: [u32; 4],
    active: [bool; 4],
    retry: [Option<Instant>; 4],
    /// When an empty slot may be asked again (one at a time, 250 ms apart).
    probe: Option<Instant>,
}
static XI: Mutex<Xi> = Mutex::new(Xi { on: [false; 4], first: [0; 4], active: [false; 4], retry: [None; 4], probe: None });
/// The XInput controllers connected at the last poll (bit i: user index i): the vibration goes
/// to each.
static XI_ON: std::sync::atomic::AtomicU32 = std::sync::atomic::AtomicU32::new(0);
/// The pad in use is a PlayStation controller: the prompts and the bindings name its buttons.
static PAD_PS: std::sync::atomic::AtomicBool = std::sync::atomic::AtomicBool::new(false);

/// The pad in use is a PlayStation controller (read directly or behind Steam Input's pad).
pub fn pad_is_playstation() -> bool {
    #[cfg(windows)]
    { PAD_PS.load(Relaxed) }
    #[cfg(target_os = "linux")]
    { crate::platform::controller::prompt_style() == 1 }
}

/// A controller's input past the dead zones: a button, a trigger past 40 of 255, a stick.
fn pad_busy(g: &XINPUT_GAMEPAD) -> bool {
    let stick = |v: i16| (v as i32).abs() > 12000;
    g.wButtons != 0 || g.bLeftTrigger > 40 || g.bRightTrigger > 40 || stick(g.sThumbLX) || stick(g.sThumbLY) || stick(g.sThumbRX) || stick(g.sThumbRY)
}

/// `g` into the merged pad `m`: its buttons, each trigger's and each stick's strongest.
fn pad_merge(m: &mut XINPUT_GAMEPAD, g: &XINPUT_GAMEPAD) {
    m.wButtons |= g.wButtons;
    m.bLeftTrigger = m.bLeftTrigger.max(g.bLeftTrigger);
    m.bRightTrigger = m.bRightTrigger.max(g.bRightTrigger);
    let far = |x: i16, y: i16| (x as i32).abs() + (y as i32).abs();
    if far(g.sThumbLX, g.sThumbLY) > far(m.sThumbLX, m.sThumbLY) {
        (m.sThumbLX, m.sThumbLY) = (g.sThumbLX, g.sThumbLY);
    }
    if far(g.sThumbRX, g.sThumbRY) > far(m.sThumbRX, m.sThumbRY) {
        (m.sThumbRX, m.sThumbRY) = (g.sThumbRX, g.sThumbRY);
    }
}

/// Every controller as one pad (the game's one player): the XInput controllers (Xbox and
/// compatible, Steam Input's and DS4Windows' virtual pads too) and the PlayStation controllers
/// read directly (`pspad`, since 2026-11-23: no Steam Input needed), the buttons of all, each
/// trigger's and each stick's strongest. A controller counts once its state has changed since it
/// was found, so a device that only ever reports one state (a stuck trigger or stick) does not
/// drive the game, and a second device for the same controller (Steam Input's pad beside the
/// DualSense itself) presses nothing twice. None when no controller is connected.
pub fn xinput_state() -> Option<XINPUT_STATE> {
    if SYSTEM_OFF.load(Relaxed) {
        return None;
    }
    let mut x = XI.lock();
    let now = Instant::now();
    let mut any = false;
    let mut m: XINPUT_GAMEPAD = unsafe { std::mem::zeroed() };
    let mut packets = 0u32;
    let (mut used_ps, mut used_xbox) = (false, false);
    let mut bits = 0u32;
    for i in 0..4usize {
        if !crate::platform::controller::poll_disconnected() && !x.on[i] && x.retry[i].is_some_and(|t| now < t) {
            continue;
        }
        // (Asking for a controller that is not there can take long, and this runs on the game's
        // thread: the empty slots one at a time, 250 ms apart, each at most once a second.)
        if !crate::platform::controller::poll_disconnected() && !x.on[i] && x.retry[i].is_some() && x.probe.is_some_and(|t| now < t) {
            continue;
        }
        let mut s: XINPUT_STATE = unsafe { std::mem::zeroed() };
        let on = unsafe { XInputGetState(i as u32, &mut s) } == 0;
        if on {
            bits |= 1 << i;
            any = true;
            if !x.on[i] {
                x.first[i] = s.dwPacketNumber;
                x.active[i] = crate::platform::controller::initial_sample_active();
                crate::info!("controller: XInput controller {i} connected");
            } else if s.dwPacketNumber != x.first[i] {
                x.active[i] = true;
            }
            if x.active[i] {
                pad_merge(&mut m, &s.Gamepad);
                packets = packets.wrapping_add(s.dwPacketNumber);
                used_xbox |= pad_busy(&s.Gamepad);
            }
        } else {
            if x.on[i] {
                crate::info!("controller: XInput controller {i} disconnected");
            }
            x.retry[i] = Some(now + Duration::from_secs(1));
            x.probe = Some(now + Duration::from_millis(250));
        }
        x.on[i] = on;
    }
    drop(x);
    for p in crate::pspad::states() {
        any = true;
        if p.active {
            let g = XINPUT_GAMEPAD { wButtons: p.pad.buttons, bLeftTrigger: p.pad.lt, bRightTrigger: p.pad.rt, sThumbLX: p.pad.lx, sThumbLY: p.pad.ly, sThumbRX: p.pad.rx, sThumbRY: p.pad.ry };
            pad_merge(&mut m, &g);
            packets = packets.wrapping_add(p.packet);
            used_ps |= pad_busy(&g);
        }
    }
    XI_ON.store(bits, Relaxed);
    // The kind of pad in use: the one with input now, a PlayStation controller first (Steam
    // Input's pad for the same controller has the same input); with none yet, a PlayStation
    // controller alone counts as one.
    if used_ps || (!used_xbox && bits == 0 && any) {
        PAD_PS.store(true, Relaxed);
    } else if used_xbox {
        PAD_PS.store(false, Relaxed);
    }
    any.then_some(XINPUT_STATE { dwPacketNumber: packets, Gamepad: m })
}

/// The motor speeds set last (left << 16 | right).
static RUMBLE: std::sync::atomic::AtomicU32 = std::sync::atomic::AtomicU32::new(0);

/// The motors of every controller (XInput's speeds; left the strong motor): the XInput
/// controllers connected and the PlayStation controllers (`pspad`). Speeds already set are not
/// sent again.
pub fn rumble(left: u16, right: u16) {
    let packed = (left as u32) << 16 | right as u32;
    if RUMBLE.swap(packed, Relaxed) == packed {
        return;
    }
    let bits = XI_ON.load(Relaxed);
    for i in 0..4u32 {
        if bits & (1 << i) != 0 {
            let mut v = XINPUT_VIBRATION { wLeftMotorSpeed: left, wRightMotorSpeed: right };
            unsafe { XInputSetState(i, &mut v) };
        }
    }
    crate::pspad::set_motors((left >> 8) as u8, (right >> 8) as u8);
}

/// The pad buttons' names on a PlayStation controller (`PAD_BUTTONS` order).
pub const PS_LABELS: [&str; 16] =
    ["Cross", "Circle", "Square", "Triangle", "L1", "R1", "L2", "R2", "L3", "R3", "Touch Pad", "Options", "D-pad Up", "D-pad Down", "D-pad Left", "D-pad Right"];

/// Pad button `i`'s name on the controller in use (`PAD_BUTTONS` order).
pub fn pad_label(i: usize) -> &'static str {
    if pad_is_playstation() { PS_LABELS.get(i).copied().unwrap_or("?") } else { PAD_BUTTONS.get(i).map(|b| b.1).unwrap_or("?") }
}

/// The title's haptic waveform (the DualSense actuators' audio port, `audio2`), one grain's RMS per
/// side (0 to 1 full scale): an Xbox controller's motors at that strength (left the heavy
/// motor), so the game's Vibration Mode and Intensity settings carry over. Without a pad nothing
/// is sent.
pub fn haptics(left: f32, right: f32) {
    if !pad_connected() {
        return;
    }
    // A strong hit reaches ~0.18 RMS: about 85% of the motor; weak textures stay light.
    let speed = |rms: f32| -> u16 {
        let x = (rms * 4.5).clamp(0.0, 1.0).powf(0.8);
        if x < 0.06 { 0 } else { (x * 65535.0) as u16 }
    };
    rumble(speed(left), speed(right));
}

fn axis(v: i16, invert: bool) -> u8 {
    let f = (v as f32 + 32768.0) / 65535.0;
    let f = if invert { 1.0 - f } else { f };
    (f * 255.0).round().clamp(0.0, 255.0) as u8
}

/// An XInput pad answered the last poll.
static PAD_CONNECTED: std::sync::atomic::AtomicBool = std::sync::atomic::AtomicBool::new(false);
/// Pad buttons held back from the game until released (held when a controller binding was
/// captured: `swallow_pad`).
static PAD_SWALLOW: std::sync::atomic::AtomicU32 = std::sync::atomic::AtomicU32::new(0);

/// Holds the pad buttons `bits` (bit i: PAD_BUTTONS[i]) back from the game until each is released.
pub fn swallow_pad(bits: u32) {
    PAD_SWALLOW.fetch_or(bits, Relaxed);
}

pub fn pad_connected() -> bool {
    PAD_CONNECTED.load(Relaxed)
}

/// The XInput pad into `st`; returns its buttons held (bit i: PAD_BUTTONS[i]) and its left stick
/// (XInput's range, y up) for the system text box.
fn xinput(st: &mut PadState) -> (u32, i16, i16) {
    let s = xinput_state();
    PAD_CONNECTED.store(s.is_some(), Relaxed);
    let Some(s) = s else { return (0, 0, 0) };
    let g = s.Gamepad;
    // The pad is in use once a button, a trigger or a stick past a dead zone moves.
    let stick = |v: i16| (v as i32).abs() > 12000;
    if g.wButtons != 0 || g.bLeftTrigger > 40 || g.bRightTrigger > 40 || stick(g.sThumbLX) || stick(g.sThumbLY) || stick(g.sThumbRX) || stick(g.sThumbRY) {
        note_device(DEVICE_PAD);
    }
    // The buttons held (bit i: PAD_BUTTONS[i]; a trigger past 30 of 255).
    let mut phys = 0u32;
    for (i, &(_, _, bit)) in PAD_BUTTONS.iter().enumerate() {
        let down = match i {
            PAD_LT => g.bLeftTrigger > 30,
            PAD_RT => g.bRightTrigger > 30,
            _ => g.wButtons & bit != 0,
        };
        if down {
            phys |= 1 << i;
        }
    }
    // The system text box (the character's name): the pad is its alone while it is open
    // (`pad_state`).
    if crate::hle::libs::ime_dialog::active() {
        return (phys, g.sThumbLX, g.sThumbLY);
    }
    // A controller binding being captured: the pad is the menu's alone (`pcsettings::bindings`);
    // a button held when it ended reaches the game once released and pressed again.
    if crate::pcsettings::bindings::pad_input(phys) {
        return (phys, g.sThumbLX, g.sThumbLY);
    }
    // (The box sees the buttons as held, also those kept from the game: a box opened again while
    // the button that closed the last one is still down sees no new press.)
    let held = phys;
    let swallowed = PAD_SWALLOW.load(Relaxed) & phys;
    PAD_SWALLOW.store(swallowed, Relaxed);
    let phys = phys & !swallowed;
    let l = layout();
    for (i, &t) in l.pad.iter().enumerate() {
        // A trigger on its own target keeps its travel (below).
        if phys & (1 << i) != 0 && !(i == PAD_LT && t == BTN_L2) && !(i == PAD_RT && t == BTN_R2) {
            st.buttons |= t;
        }
    }
    if l.pad[PAD_LT] == BTN_L2 && swallowed & (1 << PAD_LT) == 0 {
        st.l2 = st.l2.max(g.bLeftTrigger);
    }
    if l.pad[PAD_RT] == BTN_R2 && swallowed & (1 << PAD_RT) == 0 {
        st.r2 = st.r2.max(g.bRightTrigger);
    }
    st.lx = axis(g.sThumbLX, false);
    st.ly = axis(g.sThumbLY, true);
    st.rx = axis(g.sThumbRX, false);
    st.ry = axis(g.sThumbRY, true);
    (held, g.sThumbLX, g.sThumbLY)
}

// ---------------------------------------------------------------------------
// Keyboard and mouse

/// The default keyboard and mouse layout, also written to controls.txt on
/// first start: Dark Souls III's PC defaults on the PS5 pad's buttons.
pub const DEFAULT_LAYOUT: &str = "\
# Keyboard and mouse layout of Regaliant (Dark Souls III-style PC
# controls), and the controller's buttons where remapped. The menu's Key
# Bindings and Controller Bindings write it; or edit it and restart; delete it
# to get the defaults back.
#
# Lines: <input> = <pad target>. Inputs: key names (A..Z, 0..9, Space, Enter,
# Esc, Tab, Backspace, Up, Down, Left, Right, Shift, LShift, RShift, Ctrl,
# LCtrl, RCtrl, Alt, LAlt, RAlt, F2..F12, Insert, Delete, Home, End, PageUp,
# PageDown, CapsLock), mouse buttons (Mouse1 left, Mouse2 right, Mouse3
# middle, Mouse4 and Mouse5 the side buttons), WheelUp, WheelDown; each may
# have Shift+, Ctrl+ or Alt+ in front (that binding then wins while the
# modifier is held). Targets: cross circle square triangle l1 r1 l2 r2 l3 r3
# options touchpad, up down left right (d-pad), ls_up ls_down ls_left
# ls_right (left stick), rs_up rs_down rs_left rs_right (right stick, the
# camera), walk (the left stick at walking tilt).
# Controller buttons (an Xbox controller's names: PadA PadB PadX PadY PadLB
# PadRB PadLT PadRT PadLS PadRS PadView PadMenu PadUp PadDown PadLeft
# PadRight) take a button target each: PadA = cross is the default.
# F1 releases the mouse cursor or takes it back.

# Mouse look: 1 on, 0 off; raw (1: the mouse turns the camera by its counts,
# without acceleration, smoothing or a speed cap; 0: through the right stick,
# with the game's response curve and top speed); sensitivity (1.0: 2000 mouse
# counts turn the camera as far as a second of full right stick at the game's
# base camera speed); the vertical axis's own sensitivity (optional, the same
# as mouse_sensitivity when left out); invert the vertical axis (1).
mouse_look = 1
mouse_look_raw = 1
mouse_sensitivity = 1.0
# mouse_sensitivity_y = 1.0
mouse_invert_y = 0

W = ls_up
S = ls_down
A = ls_left
D = ls_right
LAlt = walk
Space = circle          # dash (hold), backstep, roll; back in menus
Mouse1 = r1             # attack (right hand)
Shift+Mouse1 = r2       # strong attack
Mouse2 = l1             # guard (left hand)
LCtrl = l2              # parry / left-hand strong attack
F = triangle            # two-hand
R = square              # use item
E = cross               # interact; confirm in menus
Q = r3                  # lock on / reset camera
Mouse3 = r3
G = touchpad            # tool belt
Esc = options           # menu
Up = up                 # switch spell
Down = down             # switch item
Right = right           # switch right-hand weapon
Left = left             # switch left-hand weapon
WheelUp = up
WheelDown = down
Shift+WheelUp = right
Shift+WheelDown = left
Shift+Left = l1         # menus: previous page
Shift+Right = r1        # menus: next page
Enter = cross
Backspace = circle
I = rs_up               # camera without a mouse
K = rs_down
J = rs_left
L = rs_right
";

/// Modifier bits of a binding.
const MOD_SHIFT: u8 = 1;
const MOD_CTRL: u8 = 2;
const MOD_ALT: u8 = 4;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Src {
    Key(u8),
    /// Mouse button by bit (`window::MOUSE_BUTTONS`).
    Mouse(u8),
    WheelUp,
    WheelDown,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Dst {
    Button(u32),
    Stick(Stick),
    Walk,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
struct Bind {
    src: Src,
    mods: u8,
    dst: Dst,
}

#[derive(Clone, Debug)]
struct Layout {
    binds: Vec<Bind>,
    /// Each binding's comment in the file (`# ...`), kept when the file is written again.
    comments: Vec<Option<String>>,
    mouse_look: bool,
    sensitivity: f32,
    /// The vertical axis's own sensitivity (`mouse_sensitivity_y`; None: the
    /// same as `sensitivity`).
    sensitivity_y: Option<f32>,
    invert_y: bool,
    /// Mouse counts turn the gameplay camera directly (`camhook`), not
    /// through the stick (`mouse_look_raw`, default 1 since 2026-10-21, also
    /// for files written before; `DS_MOUSE_RAW` overrides).
    mouse_raw: bool,
    /// Each pad button's target (`PAD_BUTTONS` order; `PAD_DEFAULT` unless remapped).
    pad: [u32; 16],
    /// The file has the pad's lines (they are written again; else only once remapped).
    pad_lines: bool,
}

/// A key name's virtual-key code.
fn key_code(name: &str) -> Option<u8> {
    let n = name.trim().to_ascii_lowercase();
    if n.len() == 1 {
        let c = n.as_bytes()[0];
        if c.is_ascii_lowercase() {
            return Some(c.to_ascii_uppercase());
        }
        if c.is_ascii_digit() {
            return Some(c);
        }
    }
    if let Some(f) = n.strip_prefix('f').and_then(|d| d.parse::<u8>().ok()) {
        return (1..=24).contains(&f).then_some(0x6F + f);
    }
    Some(match n.as_str() {
        "space" => 0x20,
        "enter" | "return" => 0x0D,
        "esc" | "escape" => 0x1B,
        "tab" => 0x09,
        "backspace" => 0x08,
        "up" => 0x26,
        "down" => 0x28,
        "left" => 0x25,
        "right" => 0x27,
        "shift" => 0x10,
        "lshift" => 0xA0,
        "rshift" => 0xA1,
        "ctrl" | "control" => 0x11,
        "lctrl" => 0xA2,
        "rctrl" => 0xA3,
        "alt" => 0x12,
        "lalt" => 0xA4,
        "ralt" => 0xA5,
        "capslock" => 0x14,
        "insert" => 0x2D,
        "delete" => 0x2E,
        "home" => 0x24,
        "end" => 0x23,
        "pageup" => 0x21,
        "pagedown" => 0x22,
        _ => return None,
    })
}

fn parse_src(s: &str) -> Option<(Src, u8)> {
    let mut mods = 0u8;
    let mut rest = s.trim();
    loop {
        let lower = rest.to_ascii_lowercase();
        let (m, len) = if lower.starts_with("shift+") {
            (MOD_SHIFT, 6)
        } else if lower.starts_with("ctrl+") {
            (MOD_CTRL, 5)
        } else if lower.starts_with("alt+") {
            (MOD_ALT, 4)
        } else {
            break;
        };
        mods |= m;
        rest = rest[len..].trim_start();
    }
    let src = match rest.to_ascii_lowercase().as_str() {
        "mouse1" => Src::Mouse(0),
        "mouse2" => Src::Mouse(1),
        "mouse3" => Src::Mouse(2),
        "mouse4" => Src::Mouse(3),
        "mouse5" => Src::Mouse(4),
        "wheelup" => Src::WheelUp,
        "wheeldown" => Src::WheelDown,
        _ => Src::Key(key_code(rest)?),
    };
    Some((src, mods))
}

fn parse_dst(s: &str) -> Option<Dst> {
    let n = s.trim().to_ascii_lowercase();
    if n == "walk" {
        return Some(Dst::Walk);
    }
    match parse_name(&n)? {
        Ok(b) => Some(Dst::Button(b)),
        Err(st) => Some(Dst::Stick(st)),
    }
}

fn parse_layout(text: &str) -> Result<Layout, String> {
    let mut l = Layout { binds: Vec::new(), comments: Vec::new(), mouse_look: true, sensitivity: 1.0, sensitivity_y: None, invert_y: false, mouse_raw: true, pad: PAD_DEFAULT, pad_lines: false };
    for (i, full) in text.lines().enumerate() {
        let comment = full.find('#').map(|c| full[c..].trim_end().to_string());
        let line = full.split('#').next().unwrap_or("").trim();
        if line.is_empty() {
            continue;
        }
        let Some((a, b)) = line.split_once('=') else { return Err(format!("line {}: expected <input> = <target>", i + 1)) };
        let (a, b) = (a.trim(), b.trim());
        match a.to_ascii_lowercase().as_str() {
            "mouse_look" => l.mouse_look = b != "0",
            "mouse_invert_y" => l.invert_y = b != "0",
            "mouse_look_raw" => l.mouse_raw = b != "0",
            "mouse_sensitivity" => {
                l.sensitivity = b.parse::<f32>().ok().filter(|v| v.is_finite() && *v > 0.0).ok_or_else(|| format!("line {}: bad sensitivity '{b}'", i + 1))?;
            }
            "mouse_sensitivity_y" => {
                l.sensitivity_y = Some(b.parse::<f32>().ok().filter(|v| v.is_finite() && *v > 0.0).ok_or_else(|| format!("line {}: bad sensitivity '{b}'", i + 1))?);
            }
            _ if pad_index(a).is_some() => {
                let Some(Dst::Button(t)) = parse_dst(b) else { return Err(format!("line {}: a controller button takes a button target, not '{b}'", i + 1)) };
                l.pad[pad_index(a).unwrap()] = t;
                l.pad_lines = true;
            }
            _ => {
                let (src, mods) = parse_src(a).ok_or_else(|| format!("line {}: unknown input '{a}'", i + 1))?;
                let dst = parse_dst(b).ok_or_else(|| format!("line {}: unknown target '{b}'", i + 1))?;
                l.binds.push(Bind { src, mods, dst });
                l.comments.push(comment);
            }
        }
    }
    Ok(l)
}

/// The layout in use (replaced whole when an action is rebound in the menu, `rebind`) and its file.
static LAYOUT: Mutex<Option<std::sync::Arc<Layout>>> = Mutex::new(None);
static LAYOUT_PATH: Mutex<Option<std::path::PathBuf>> = Mutex::new(None);
/// Counts layout changes (the prompts' glyph map follows).
pub static LAYOUT_GEN: std::sync::atomic::AtomicU32 = std::sync::atomic::AtomicU32::new(0);

fn layout() -> std::sync::Arc<Layout> {
    LAYOUT.lock().get_or_insert_with(|| std::sync::Arc::new(parse_layout(DEFAULT_LAYOUT).expect("default layout"))).clone()
}

/// Reads the keyboard and mouse layout from `path` (written with the
/// defaults when missing; the defaults on errors, logged).
pub fn init(path: &std::path::Path) {
    if !path.exists() {
        if let Err(e) = std::fs::write(path, DEFAULT_LAYOUT) {
            crate::warn!("input: {} not written ({e})", path.display());
        }
    }
    let mut l = match std::fs::read_to_string(path) {
        Ok(text) => match parse_layout(&text) {
            Ok(l) => {
                crate::info!("input: keyboard and mouse layout from {} ({} bindings; mouse look {}{}, sensitivity {}{}, invert y {})", path.display(), l.binds.len(), l.mouse_look, if l.mouse_raw { " (raw)" } else { "" }, l.sensitivity, l.sensitivity_y.map(|v| format!(" (vertical {v})")).unwrap_or_default(), l.invert_y);
                l
            }
            Err(e) => {
                crate::warn!("input: {}: {e}; the default layout is used", path.display());
                parse_layout(DEFAULT_LAYOUT).expect("default layout")
            }
        },
        Err(_) => parse_layout(DEFAULT_LAYOUT).expect("default layout"),
    };
    if let Ok(v) = std::env::var("DS_MOUSE_RAW") {
        l.mouse_raw = v != "0";
        crate::info!("input: raw mouse look {} (DS_MOUSE_RAW)", if l.mouse_raw { "on" } else { "off" });
    }
    *LAYOUT.lock() = Some(std::sync::Arc::new(l));
    *LAYOUT_PATH.lock() = Some(path.to_path_buf());
}

/// Binds `target`'s first binding to `input` with `mods` (None: removes that binding). An input
/// another action had as its first binding goes to that action's place in exchange (the two
/// swap), so no input does two things. The file is written again (its other lines and comments
/// as they were; the changed bindings' lines replaced).
pub fn rebind(target: &str, input: Option<(PromptInput, u8)>) -> Result<(), String> {
    let dst = parse_dst(target).ok_or_else(|| format!("unknown target {target}"))?;
    let src_of = |i: PromptInput| match i {
        PromptInput::Key(k) => Src::Key(k),
        PromptInput::Mouse(m) => Src::Mouse(m),
        PromptInput::WheelUp => Src::WheelUp,
        PromptInput::WheelDown => Src::WheelDown,
    };
    let old = layout();
    let mut l = (*old).clone();
    let mine = l.binds.iter().position(|b| b.dst == dst);
    match input {
        Some((i, mods)) => {
            let (src, mods) = (src_of(i), mods);
            // Another action's binding of the same input: it takes this action's old input.
            if let Some(other) = l.binds.iter().position(|b| b.src == src && b.mods == mods && b.dst != dst) {
                match mine {
                    Some(m) => {
                        let (osrc, omods) = (l.binds[m].src, l.binds[m].mods);
                        l.binds[other].src = osrc;
                        l.binds[other].mods = omods;
                    }
                    None => {
                        l.binds.remove(other);
                        if other < l.comments.len() {
                            l.comments.remove(other);
                        }
                    }
                }
            }
            let mine = l.binds.iter().position(|b| b.dst == dst);
            match mine {
                Some(m) => {
                    l.binds[m].src = src;
                    l.binds[m].mods = mods;
                }
                None => {
                    l.binds.push(Bind { src, mods, dst });
                    l.comments.push(None);
                }
            }
        }
        None => {
            if let Some(m) = mine {
                l.binds.remove(m);
                if m < l.comments.len() {
                    l.comments.remove(m);
                }
            }
        }
    }
    *LAYOUT.lock() = Some(std::sync::Arc::new(l.clone()));
    LAYOUT_GEN.fetch_add(1, Relaxed);
    crate::info!("input: {target} bound to {}", input.map(|(i, m)| prompt_name(i, m)).unwrap_or_else(|| "nothing".into()));
    save(&l)
}

/// An input named as in the layout file (`Shift+Mouse1`, `G`): the prompts' form.
pub fn parse_prompt_input(name: &str) -> Option<(PromptInput, u8)> {
    let (src, mods) = parse_src(name)?;
    let i = match src {
        Src::Key(k) => PromptInput::Key(k),
        Src::Mouse(m) => PromptInput::Mouse(m),
        Src::WheelUp => PromptInput::WheelUp,
        Src::WheelDown => PromptInput::WheelDown,
    };
    Some((i, mods))
}

/// An input's name as the layout file writes it.
pub fn prompt_name(i: PromptInput, mods: u8) -> String {
    let mut s = String::new();
    for (bit, name) in [(MOD_SHIFT, "Shift+"), (MOD_CTRL, "Ctrl+"), (MOD_ALT, "Alt+")] {
        if mods & bit != 0 {
            s += name;
        }
    }
    s += &match i {
        PromptInput::Key(k) => key_name(k),
        PromptInput::Mouse(m) => format!("Mouse{}", m + 1),
        PromptInput::WheelUp => "WheelUp".into(),
        PromptInput::WheelDown => "WheelDown".into(),
    };
    s
}

/// A virtual-key code's name in the layout file.
fn key_name(k: u8) -> String {
    match k {
        b'A'..=b'Z' | b'0'..=b'9' => (k as char).to_string(),
        0x70..=0x87 => format!("F{}", k - 0x6F),
        _ => ["Space", "Enter", "Esc", "Tab", "Backspace", "Up", "Down", "Left", "Right", "Shift", "LShift", "RShift", "Ctrl", "LCtrl", "RCtrl", "Alt", "LAlt", "RAlt", "CapsLock", "Insert", "Delete", "Home", "End", "PageUp", "PageDown"]
            .into_iter()
            .find(|n| key_code(n) == Some(k))
            .map(str::to_string)
            .unwrap_or_else(|| format!("{k:#04x}")),
    }
}

/// The layout file written again (`render`).
fn save(l: &Layout) -> Result<(), String> {
    let Some(path) = LAYOUT_PATH.lock().clone() else { return Ok(()) };
    let text = std::fs::read_to_string(&path).unwrap_or_else(|_| DEFAULT_LAYOUT.to_string());
    std::fs::write(&path, render(l, &text)).map_err(|e| format!("{}: {e}", path.display()))
}

/// The layout file's text for `l`: the lines of `text` before its first binding kept (the
/// settings and the explanation), then the bindings in the layout's order, each with its comment.
fn render(l: &Layout, text: &str) -> String {
    let mut head = Vec::new();
    for line in text.lines() {
        let body = line.split('#').next().unwrap_or("").trim();
        let is_binding = body.split_once('=').is_some_and(|(a, _)| {
            let a = a.trim();
            (parse_src(a).is_some() || pad_index(a).is_some()) && !matches!(a.to_ascii_lowercase().as_str(), "mouse_look" | "mouse_invert_y" | "mouse_look_raw" | "mouse_sensitivity" | "mouse_sensitivity_y")
        });
        if is_binding {
            break;
        }
        head.push(line.to_string());
    }
    let target_name = |d: Dst| -> String {
        let names = ["cross", "circle", "square", "triangle", "up", "down", "left", "right", "l1", "r1", "l2", "r2", "l3", "r3", "options", "touchpad", "ls_up", "ls_down", "ls_left", "ls_right", "rs_up", "rs_down", "rs_left", "rs_right", "walk"];
        names.into_iter().find(|n| parse_dst(n) == Some(d)).unwrap_or("?").to_string()
    };
    let mut out = head.join("\n");
    if !out.is_empty() {
        out.push('\n');
    }
    for (i, b) in l.binds.iter().enumerate() {
        let input = match b.src {
            Src::Key(k) => PromptInput::Key(k),
            Src::Mouse(m) => PromptInput::Mouse(m),
            Src::WheelUp => PromptInput::WheelUp,
            Src::WheelDown => PromptInput::WheelDown,
        };
        let line = format!("{} = {}", prompt_name(input, b.mods), target_name(b.dst));
        match l.comments.get(i).cloned().flatten() {
            Some(c) => out += &format!("{line:<24}{c}\n"),
            None => out += &format!("{line}\n"),
        }
    }
    if l.pad_lines || l.pad != PAD_DEFAULT {
        out += "\n# The controller's buttons (the menu's Controller Bindings).\n";
        for (i, (name, _, _)) in PAD_BUTTONS.iter().enumerate() {
            out += &format!("{name} = {}\n", target_name(Dst::Button(l.pad[i])));
        }
    }
    out
}

/// The pad button (`PAD_BUTTONS` index) on target `target` (a `BTN_*` bit), the first of them.
pub fn pad_button_of(target: u32) -> Option<usize> {
    layout().pad.iter().position(|&t| t == target)
}

/// Pad button `button`'s target now.
pub fn pad_target(button: usize) -> u32 {
    layout().pad.get(button).copied().unwrap_or(0)
}

/// Puts target `target` (a button: `cross`, `r1`, `up`, ...) on pad button `button`; that
/// button's old target goes to the button `target` was on (they swap), so every target keeps a
/// button. The file is written again.
pub fn rebind_pad(target: &str, button: usize) -> Result<(), String> {
    let Some(Dst::Button(t)) = parse_dst(target) else { return Err(format!("{target} is not a button")) };
    if button >= PAD_BUTTONS.len() {
        return Err(format!("no pad button {button}"));
    }
    let mut l = (*layout()).clone();
    let prev = l.pad[button];
    if prev == t {
        return Ok(());
    }
    if let Some(mine) = l.pad.iter().position(|&x| x == t) {
        l.pad[mine] = prev;
    }
    l.pad[button] = t;
    l.pad_lines = true;
    *LAYOUT.lock() = Some(std::sync::Arc::new(l.clone()));
    LAYOUT_GEN.fetch_add(1, Relaxed);
    crate::info!("input: {target} on the controller's {}", PAD_BUTTONS[button].1);
    save(&l)
}

/// Keys and mouse buttons held by the control channel (`inject_hold`):
/// until when.
static INJECTED: Mutex<Vec<(Instant, Src)>> = Mutex::new(Vec::new());

/// Diagnosis (control channel `key`): holds a key or mouse button (a layout
/// input name without modifiers) for `dur`, as if pressed in the window.
pub fn inject_hold(name: &str, dur: Duration) -> Result<(), String> {
    let (src, mods) = parse_src(name).ok_or_else(|| format!("unknown input '{name}'"))?;
    if mods != 0 || matches!(src, Src::WheelUp | Src::WheelDown) {
        return Err(format!("'{name}': a key or mouse button without modifiers"));
    }
    note_device(DEVICE_KBM);
    let now = Instant::now();
    let mut g = INJECTED.lock();
    g.retain(|x| x.0 > now);
    g.push((now + dur, src));
    Ok(())
}

/// Diagnosis (control channel `mouse`, `wheel`): mouse motion in counts and
/// wheel notches (up positive), as if reported by the window.
pub fn inject_motion(dx: i64, dy: i64, wheel: i32) {
    use crate::gpu::vk::window::{MOUSE_DX, MOUSE_DY, MOUSE_WHEEL};
    note_device(DEVICE_KBM);
    MOUSE_DX.fetch_add(dx, Relaxed);
    MOUSE_DY.fetch_add(dy, Relaxed);
    MOUSE_WHEEL.fetch_add(wheel * 120, Relaxed);
}

/// Diagnosis (control channel `mousemove`): steady mouse motion, counts a second, for a while,
/// fed at every vblank (`motion_tick`) as a polled mouse delivers it.
static MOTION: Mutex<Option<(f64, f64, Instant, Instant, f64, f64)>> = Mutex::new(None);

pub fn inject_motion_for(dx_per_s: f64, dy_per_s: f64, dur: Duration) {
    let now = Instant::now();
    note_device(DEVICE_KBM);
    *MOTION.lock() = Some((dx_per_s, dy_per_s, now + dur, now, 0.0, 0.0));
}

/// The vblank thread's tick: the steady motion's counts since the last tick.
pub fn motion_tick() {
    use crate::gpu::vk::window::{MOUSE_DX, MOUSE_DY};
    let mut g = MOTION.lock();
    let Some((vx, vy, until, last, fx, fy)) = *g else { return };
    let now = Instant::now().min(until);
    let dt = now.duration_since(last).as_secs_f64();
    let (ax, ay) = (fx + vx * dt, fy + vy * dt);
    let (ix, iy) = (ax.trunc(), ay.trunc());
    MOUSE_DX.fetch_add(ix as i64, Relaxed);
    MOUSE_DY.fetch_add(iy as i64, Relaxed);
    if ix != 0.0 || iy != 0.0 {
        note_device(DEVICE_KBM);
    }
    *g = if now >= until { None } else { Some((vx, vy, until, now, ax - ix, ay - iy)) };
}

/// Whether `src` is held: in the window or by the control channel (a sided
/// modifier held there also holds the common one).
fn held(src: Src) -> bool {
    use crate::gpu::vk::window::{MOUSE_BUTTONS, key_down};
    let window = match src {
        Src::Key(k) => key_down(k),
        Src::Mouse(m) => MOUSE_BUTTONS.load(Relaxed) & (1 << m) != 0,
        Src::WheelUp | Src::WheelDown => false,
    };
    if window {
        return true;
    }
    let now = Instant::now();
    let g = INJECTED.lock();
    g.iter().any(|&(until, s)| {
        now < until
            && (s == src
                || matches!((s, src), (Src::Key(0xA0 | 0xA1), Src::Key(0x10)) | (Src::Key(0xA2 | 0xA3), Src::Key(0x11)) | (Src::Key(0xA4 | 0xA5), Src::Key(0x12))))
    })
}

/// The modifier bits held now.
fn held_mods() -> u8 {
    (if held(Src::Key(0x10)) { MOD_SHIFT } else { 0 }) | (if held(Src::Key(0x11)) { MOD_CTRL } else { 0 }) | (if held(Src::Key(0x12)) { MOD_ALT } else { 0 })
}

/// Of the bindings of `src`, those that apply with `held` modifiers: the
/// ones with the most modifiers, all held.
fn applying(l: &Layout, src: Src, held: u8) -> impl Iterator<Item = &Bind> {
    let best = l.binds.iter().filter(|b| b.src == src && b.mods & !held == 0).map(|b| b.mods.count_ones()).max();
    l.binds.iter().filter(move |b| b.src == src && b.mods & !held == 0 && Some(b.mods.count_ones()) == best)
}

// ---------------------------------------------------------------------------
// The device in use and what a pad target is bound to (the button prompts, `pcsettings::prompts`)

pub const DEVICE_NONE: u8 = 0;
pub const DEVICE_KBM: u8 = 1;
pub const DEVICE_PAD: u8 = 2;
/// The device the player used last: keyboard and mouse (keys, mouse buttons, the wheel and
/// motion in the window) or the pad (a button, a trigger, a stick past a dead zone).
static LAST_DEVICE: std::sync::atomic::AtomicU8 = std::sync::atomic::AtomicU8::new(DEVICE_NONE);

pub fn note_device(d: u8) {
    if LAST_DEVICE.swap(d, Relaxed) != d {
        crate::info!("input: {} in use", if d == DEVICE_PAD { "the pad" } else { "keyboard and mouse" });
    }
}

pub fn last_device() -> u8 {
    LAST_DEVICE.load(Relaxed)
}

/// An input a prompt names.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub enum PromptInput {
    /// A key by virtual-key code.
    Key(u8),
    /// A mouse button: 0 left, 1 right, 2 middle, 3 and 4 the side buttons.
    Mouse(u8),
    WheelUp,
    WheelDown,
}

pub const PROMPT_SHIFT: u8 = MOD_SHIFT;
pub const PROMPT_CTRL: u8 = MOD_CTRL;
pub const PROMPT_ALT: u8 = MOD_ALT;

/// The first binding of a layout target (`cross`, `l1`, `up`, `ls_up`, ...; file order) as
/// its input and modifier bits (`PROMPT_*`).
pub fn binding_of(target: &str) -> Option<(PromptInput, u8)> {
    let dst = parse_dst(target)?;
    layout().binds.iter().find(|b| b.dst == dst).map(|b| {
        let input = match b.src {
            Src::Key(k) => PromptInput::Key(k),
            Src::Mouse(m) => PromptInput::Mouse(m),
            Src::WheelUp => PromptInput::WheelUp,
            Src::WheelDown => PromptInput::WheelDown,
        };
        (input, b.mods)
    })
}

/// Every input of the layout's bindings with their modifiers (the prompts' glyph set).
pub fn bound_inputs() -> Vec<(PromptInput, u8)> {
    let mut v: Vec<(PromptInput, u8)> = layout()
        .binds
        .iter()
        .map(|b| {
            let input = match b.src {
                Src::Key(k) => PromptInput::Key(k),
                Src::Mouse(m) => PromptInput::Mouse(m),
                Src::WheelUp => PromptInput::WheelUp,
                Src::WheelDown => PromptInput::WheelDown,
            };
            (input, b.mods)
        })
        .collect();
    v.sort();
    v.dedup();
    v
}

/// Every key the layout file accepts (virtual-key codes): the prompts' glyphs for any binding.
pub fn bindable_keys() -> Vec<u8> {
    let mut v: Vec<u8> = (b'A'..=b'Z').chain(b'0'..=b'9').collect();
    for f in 1..=12u8 {
        v.push(0x6F + f);
    }
    for name in ["space", "enter", "esc", "tab", "backspace", "up", "down", "left", "right", "shift", "lshift", "rshift", "ctrl", "lctrl", "rctrl", "alt", "lalt", "ralt", "capslock", "insert", "delete", "home", "end", "pageup", "pagedown"] {
        v.push(key_code(name).expect("key name"));
    }
    v
}

/// Wheel taps not yet given to the title, and the one being given: its
/// button, whether it is down, pad reads and the flip of its phase's start.
/// A phase lasts at least two reads and one flip (the title may read the pad
/// more than once per frame).
static TAPS: Mutex<(VecDeque<u32>, Option<(u32, bool, u32, u64)>)> = Mutex::new((VecDeque::new(), None));

fn wheel_taps(st: &mut PadState, l: &Layout, held: u8) {
    use crate::gpu::vk::window::MOUSE_WHEEL;
    let mut g = TAPS.lock();
    let w = MOUSE_WHEEL.load(Relaxed);
    let notches = w / 120;
    if notches != 0 {
        MOUSE_WHEEL.fetch_sub(notches * 120, Relaxed);
        let src = if notches > 0 { Src::WheelUp } else { Src::WheelDown };
        let buttons: u32 = applying(l, src, held).filter_map(|b| if let Dst::Button(x) = b.dst { Some(x) } else { None }).fold(0, |a, x| a | x);
        for _ in 0..notches.unsigned_abs() {
            if buttons != 0 && g.0.len() < 8 {
                g.0.push_back(buttons);
            }
        }
    }
    let flip = crate::gpu::FRAMES.load(Relaxed);
    let (queue, cur) = &mut *g;
    loop {
        match cur {
            None => match queue.pop_front() {
                Some(b) => *cur = Some((b, true, 0, flip)),
                None => break,
            },
            Some((_, down, reads, since)) if *reads >= 2 && flip > *since => {
                if *down {
                    (*down, *reads, *since) = (false, 0, flip);
                    break;
                }
                *cur = None;
            }
            Some(_) => break,
        }
    }
    if let Some((b, down, reads, _)) = cur {
        if *down {
            st.buttons |= *b;
        }
        *reads += 1;
    }
}

/// Mouse look: the last deflection given and when the motion was last taken.
static LOOK: Mutex<(Option<Instant>, f32, f32)> = Mutex::new((None, 0.0, 0.0));

/// Mouse counts per second for a full right-stick deflection at sensitivity 1.
const FULL_SPEED: f32 = 2000.0;
/// Deflection added before mouse motion's own (a stick dead zone to step
/// over). None: the title's camera takes the raw stick, (rx - 128) / 127,
/// even at 1/127 (its pad layer applies the radial dead zone our
/// scePadGetControllerInformation reports, 13 of 127, only to values the
/// camera does not use; FINDINGS.md 2026-10-17).
const STICK_DEAD: f32 = 0.0;

/// Raw mouse look is chosen (`mouse_look_raw`, `DS_MOUSE_RAW`).
pub fn raw_mouse_look() -> bool {
    let l = layout();
    l.mouse_look && l.mouse_raw
}

/// The layout's change count (`LAYOUT_GEN`).
pub fn layout_gen() -> u32 {
    LAYOUT_GEN.load(Relaxed)
}

/// Raw mouse look with the mouse captured.
fn raw_active() -> bool {
    use crate::gpu::vk::window::{CAPTURED, CAPTURE_TEST};
    raw_mouse_look() && (CAPTURED.load(Relaxed) || CAPTURE_TEST.load(Relaxed))
}

/// Mouse motion waits for the gameplay camera (`camhook`).
pub fn camera_motion_pending() -> bool {
    use crate::gpu::vk::window::{MOUSE_DX, MOUSE_DY};
    raw_active() && (MOUSE_DX.load(Relaxed) != 0 || MOUSE_DY.load(Relaxed) != 0)
}

/// Mouse motion for the gameplay camera (`camhook`), taken: None when raw
/// mouse look is off or the mouse is not captured (then nothing is taken).
pub fn take_camera_motion() -> Option<(i64, i64)> {
    use crate::gpu::vk::window::{MOUSE_DX, MOUSE_DY};
    if !raw_active() {
        return None;
    }
    let (dx, dy) = (MOUSE_DX.swap(0, Relaxed), MOUSE_DY.swap(0, Relaxed));
    Some((dx, if invert_y(&layout()) { -dy } else { dy }))
}

/// The PC settings' mouse sensitivity (a factor on controls.txt's, f32 bits) and vertical
/// inversion (flipping controls.txt's), from the SETTINGS screen's PC tab: read at every use.
pub static PC_SENSITIVITY: std::sync::atomic::AtomicU32 = std::sync::atomic::AtomicU32::new(0x3f80_0000);
pub static PC_INVERT_Y: std::sync::atomic::AtomicBool = std::sync::atomic::AtomicBool::new(false);

fn pc_sensitivity() -> f32 {
    let v = f32::from_bits(PC_SENSITIVITY.load(Relaxed));
    if v.is_finite() && v > 0.0 { v } else { 1.0 }
}

fn invert_y(l: &Layout) -> bool {
    l.invert_y ^ PC_INVERT_Y.load(Relaxed)
}

/// The camera's base yaw speed when the title's is unusable (radians per
/// second at full tilt).
const RAW_FALLBACK_YAW_SPEED: f32 = 2.2;

/// Radians of yaw and of pitch per mouse count (raw mouse look), the same on
/// both axes unless `mouse_sensitivity_y` sets the vertical one: FULL_SPEED
/// counts at sensitivity 1 turn as far as a second of
/// full stick at the camera's base yaw speed (`yaw_speed`, radians per
/// second), as mouse look through the stick does below full tilt. Signs as
/// the camera's angles turn for right and downward motion.
pub fn camera_radians_per_count(yaw_speed: f32) -> (f32, f32) {
    let speed = if yaw_speed.is_finite() && yaw_speed > 0.0 && yaw_speed < 100.0 { yaw_speed } else { RAW_FALLBACK_YAW_SPEED };
    let l = layout();
    let k = speed / FULL_SPEED;
    let pc = pc_sensitivity();
    (k * l.sensitivity * pc * RAW_YAW_SIGN, k * l.sensitivity_y.unwrap_or(l.sensitivity) * pc * RAW_PITCH_SIGN)
}

/// The camera's yaw and pitch rise for rightward and downward motion, as for
/// the stick tilted right and down (FINDINGS.md 2026-10-21 evening).
const RAW_YAW_SIGN: f32 = 1.0;
const RAW_PITCH_SIGN: f32 = 1.0;

fn mouse_look(st: &mut PadState, l: &Layout) {
    use crate::gpu::vk::window::{CAPTURED, CAPTURE_TEST, MOUSE_DX, MOUSE_DY};
    let mut g = LOOK.lock();
    // Raw mouse look while the gameplay camera runs: the hook takes the
    // motion (`take_camera_motion`), the stick stays.
    if !l.mouse_look || !(CAPTURED.load(Relaxed) || CAPTURE_TEST.load(Relaxed)) || (l.mouse_raw && crate::camhook::live()) {
        *g = (None, 0.0, 0.0);
        return;
    }
    let now = Instant::now();
    // The title may read the pad several times a frame: motion is taken
    // over at least 8 ms, reads in between give the last deflection.
    let due = g.0.is_none_or(|t| now.duration_since(t) >= Duration::from_millis(8));
    if due {
        let secs = g.0.map_or(1.0 / 60.0, |t| now.duration_since(t).as_secs_f32()).clamp(1.0 / 240.0, 0.25);
        let (dx, dy) = (MOUSE_DX.swap(0, Relaxed) as f32, MOUSE_DY.swap(0, Relaxed) as f32);
        let pc = pc_sensitivity();
        let full = FULL_SPEED / (l.sensitivity * pc);
        let full_y = FULL_SPEED / (l.sensitivity_y.unwrap_or(l.sensitivity) * pc);
        let dy = if invert_y(l) { -dy } else { dy };
        *g = (Some(now), (dx / secs / full).clamp(-1.0, 1.0), (dy / secs / full_y).clamp(-1.0, 1.0));
    }
    // Radially, as the title removes the dead zone.
    let m = (g.1 * g.1 + g.2 * g.2).sqrt();
    if m > 1e-3 {
        let r = STICK_DEAD + (1.0 - STICK_DEAD) * m.min(1.0);
        let to_axis = |d: f32| (128.0 + d / m * r * 127.0).round().clamp(0.0, 255.0) as u8;
        st.rx = to_axis(g.1);
        st.ry = to_axis(g.2);
    }
}

/// Left-stick tilt while `walk` is held.
const WALK_TILT: f32 = 0.4;

fn keyboard_mouse(st: &mut PadState) {
    // A key binding being captured: the keyboard and mouse are the menu's alone; none at all in
    // the shader preparation's self-play (`SYSTEM_OFF`).
    if crate::pcsettings::bindings::capturing() || SYSTEM_OFF.load(Relaxed) {
        return;
    }
    let l = layout();
    let mods = held_mods();
    // Stick directions held: left x-, x+, y-, y+, right x-, x+, y-, y+.
    let mut dirs = [false; 8];
    let mut walk = false;
    for b in &l.binds {
        if b.mods & !mods != 0 || !held(b.src) || !applying(&l, b.src, mods).any(|x| x == b) {
            continue;
        }
        match b.dst {
            Dst::Button(x) => st.buttons |= x,
            Dst::Walk => walk = true,
            Dst::Stick(s) => {
                dirs[match s {
                    Stick::LLeft => 0,
                    Stick::LRight => 1,
                    Stick::LUp => 2,
                    Stick::LDown => 3,
                    Stick::RLeft => 4,
                    Stick::RRight => 5,
                    Stick::RUp => 6,
                    Stick::RDown => 7,
                }] = true
            }
        }
    }
    let stick = |neg: bool, pos: bool, cur: u8| -> u8 {
        match (neg, pos) {
            (true, false) => 0,
            (false, true) => 255,
            _ => cur,
        }
    };
    st.lx = stick(dirs[0], dirs[1], st.lx);
    st.ly = stick(dirs[2], dirs[3], st.ly);
    st.rx = stick(dirs[4], dirs[5], st.rx);
    st.ry = stick(dirs[6], dirs[7], st.ry);
    if walk {
        let tilt = |v: u8| (128.0 + (v as f32 - 128.0) * WALK_TILT).round() as u8;
        st.lx = tilt(st.lx);
        st.ly = tilt(st.ly);
    }
    wheel_taps(st, &l, mods);
    mouse_look(st, &l);
}

/// Cross presses (its rising edges in the combined state) since start: the key bindings' capture
/// starts on one (`pcsettings::bindings`).
static CROSS_PRESSES: std::sync::atomic::AtomicU32 = std::sync::atomic::AtomicU32::new(0);
static CROSS_HELD: std::sync::atomic::AtomicBool = std::sync::atomic::AtomicBool::new(false);

pub fn cross_presses() -> u32 {
    CROSS_PRESSES.load(Relaxed)
}

/// Current combined pad state.
pub fn pad_state() -> PadState {
    let mut st = PadState::default();
    let (phys, lx, ly) = xinput(&mut st);
    // The system text box (`ime_dialog`) gets the pad's buttons and left stick each poll, with
    // the control channel's presses (tests) as the pad's; while it is open the game sees no
    // input (as on the console while its keyboard shows) and the window's keys are the box's.
    let mut sc = PadState::default();
    scripted(&mut sc, crate::gpu::FRAMES.load(std::sync::atomic::Ordering::Relaxed), false);
    let mut script_bits = 0u32;
    for (i, &t) in PAD_DEFAULT.iter().enumerate() {
        if sc.buttons & t != 0 {
            script_bits |= 1 << i;
        }
    }
    let wide = |v: i32| (v * 256).clamp(-32768, 32767) as i16;
    let (lx, ly) = if (sc.lx, sc.ly) != (0x80, 0x80) { (wide(sc.lx as i32 - 128), wide(128 - sc.ly as i32)) } else { (lx, ly) };
    // (The control channel's presses held while the box was open stay its own until released,
    // as the pad's do: `swallow_pad`.)
    static SCRIPT_KEPT: std::sync::atomic::AtomicU32 = std::sync::atomic::AtomicU32::new(0);
    if crate::hle::libs::ime_dialog::pad_input(phys | script_bits, lx, ly) {
        SCRIPT_KEPT.store(script_bits, Relaxed);
        return PadState::default();
    }
    let kept = SCRIPT_KEPT.load(Relaxed) & script_bits;
    SCRIPT_KEPT.store(kept, Relaxed);
    keyboard_mouse(&mut st);
    scripted(&mut st, crate::gpu::FRAMES.load(std::sync::atomic::Ordering::Relaxed), true);
    for (i, &t) in PAD_DEFAULT.iter().enumerate() {
        if kept & (1 << i) != 0 {
            st.buttons &= !t;
        }
    }
    if st.l2 > 30 {
        st.buttons |= BTN_L2;
    }
    if st.r2 > 30 {
        st.buttons |= BTN_R2;
    }
    if st.buttons & BTN_L2 != 0 {
        st.l2 = st.l2.max(255);
    }
    if st.buttons & BTN_R2 != 0 {
        st.r2 = st.r2.max(255);
    }
    let cross = st.buttons & BTN_CROSS != 0;
    if CROSS_HELD.swap(cross, Relaxed) != cross && cross {
        CROSS_PRESSES.fetch_add(1, Relaxed);
    }
    st
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn script_parses_and_applies() {
        let s = parse_script("700:cross; 760:down+ls_left:3 ;800:Cross").unwrap();
        assert_eq!(s.len(), 3);
        assert_eq!(s[0], Step { flip: 700, hold: 6, buttons: BTN_CROSS, sticks: vec![] });
        assert_eq!(s[1].buttons, BTN_DOWN);
        assert_eq!(s[1].sticks, vec![Stick::LLeft]);
        assert_eq!(s[1].hold, 3);
        assert!(parse_script("700:jump").is_err());
        assert!(parse_script("x:cross").is_err());
    }

    #[test]
    fn default_layout_parses() {
        let l = parse_layout(DEFAULT_LAYOUT).unwrap();
        assert!(l.mouse_look && !l.invert_y && l.sensitivity == 1.0 && l.mouse_raw);
        // Files written before the key existed get raw mouse look too.
        assert!(parse_layout("mouse_look = 1\n").unwrap().mouse_raw);
        assert!(!parse_layout("mouse_look_raw = 0\n").unwrap().mouse_raw);
        let find = |src: Src, mods: u8| l.binds.iter().find(|b| b.src == src && b.mods == mods).map(|b| b.dst);
        assert_eq!(find(Src::Mouse(0), 0), Some(Dst::Button(BTN_R1)));
        assert_eq!(find(Src::Mouse(0), MOD_SHIFT), Some(Dst::Button(BTN_R2)));
        assert_eq!(find(Src::Key(0x20), 0), Some(Dst::Button(BTN_CIRCLE)));
        assert_eq!(find(Src::Key(0xA4), 0), Some(Dst::Walk));
        assert_eq!(find(Src::Key(0x1B), 0), Some(Dst::Button(BTN_OPTIONS)));
        assert_eq!(find(Src::Key(b'W'), 0), Some(Dst::Stick(Stick::LUp)));
        assert_eq!(find(Src::WheelDown, MOD_SHIFT), Some(Dst::Button(BTN_LEFT)));
    }

    #[test]
    fn pad_lines_parse_and_write_back() {
        let text = format!("{DEFAULT_LAYOUT}\nPadA = circle\nPadB = cross\n");
        let l = parse_layout(&text).unwrap();
        assert_eq!(l.pad[0], BTN_CIRCLE);
        assert_eq!(l.pad[1], BTN_CROSS);
        assert_eq!(l.pad[2], BTN_SQUARE);
        assert!(l.pad_lines);
        let out = render(&l, &text);
        assert!(out.contains("PadA = circle\nPadB = cross\nPadX = square\n"), "{out}");
        let again = parse_layout(&out).unwrap();
        assert_eq!(again.pad, l.pad);
        assert_eq!(render(&again, &out), out);
        assert!(parse_layout("PadA = ls_up").is_err());
    }

    #[test]
    fn layout_file_round_trips() {
        let l = parse_layout(DEFAULT_LAYOUT).unwrap();
        assert_eq!(render(&l, DEFAULT_LAYOUT), DEFAULT_LAYOUT);
        // A rebinding keeps the other lines and every comment where it was.
        let mut m = l.clone();
        let i = m.binds.iter().position(|b| b.dst == Dst::Button(BTN_R1)).unwrap();
        m.binds[i].src = Src::Mouse(1);
        let text = render(&m, DEFAULT_LAYOUT);
        let again = parse_layout(&text).unwrap();
        assert_eq!(again.binds.len(), l.binds.len());
        assert_eq!(again.binds[i].src, Src::Mouse(1));
        assert_eq!(again.comments, l.comments);
        assert!(text.contains("Mouse2 = r1             # attack (right hand)"));
        assert!(text.contains("LAlt = walk"));
    }

    #[test]
    fn modified_binding_wins() {
        let l = parse_layout("Mouse1 = r1\nShift+Mouse1 = r2\nCtrl+Shift+Mouse1 = l3\n").unwrap();
        let got = |held: u8| applying(&l, Src::Mouse(0), held).map(|b| b.dst).collect::<Vec<_>>();
        assert_eq!(got(0), vec![Dst::Button(BTN_R1)]);
        assert_eq!(got(MOD_SHIFT), vec![Dst::Button(BTN_R2)]);
        assert_eq!(got(MOD_SHIFT | MOD_CTRL), vec![Dst::Button(BTN_L3)]);
        assert_eq!(got(MOD_CTRL), vec![Dst::Button(BTN_R1)]);
    }

    #[test]
    fn layout_errors_name_the_line() {
        assert!(parse_layout("W = jump").unwrap_err().contains("line 1"));
        assert!(parse_layout("\nHyper+W = ls_up").unwrap_err().contains("line 2"));
        assert!(parse_layout("mouse_sensitivity = -1").is_err());
        assert!(parse_layout("mouse_sensitivity_y = 0").is_err());
        let l = parse_layout("mouse_sensitivity = 2\nmouse_sensitivity_y = 0.5").unwrap();
        assert!(l.sensitivity == 2.0 && l.sensitivity_y == Some(0.5));
        assert_eq!(parse_layout(DEFAULT_LAYOUT).unwrap().sensitivity_y, None);
        assert_eq!(key_code("f12"), Some(0x7B));
        assert_eq!(key_code("LAlt"), Some(0xA4));
        assert_eq!(parse_src("shift+wheelup"), Some((Src::WheelUp, MOD_SHIFT)));
    }
}
