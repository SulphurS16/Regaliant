//! Audio output (AudioOut), game pad (see `crate::input`), mouse, mic, voice.
//!
//! AudioOut currently paces output in real time without producing sound (the
//! host audio backend comes later); this keeps FMOD's mixer timing correct.

use crate::hle::Registry;
use crate::{fixed, reg, sysv};
use parking_lot::Mutex;
use std::time::{Duration, Instant};

// ---------------------------------------------------------------------------
// AudioOut

#[derive(Clone)]
struct Port {
    kind: i32,
    len: u32,
    freq: u32,
    format: u32,
    next: Instant,
    volume: [i32; 8],
    /// Host mixer stream (`crate::audio`).
    stream: usize,
}

/// Port types routed to the speakers (main, BGM, voice); the pad speaker and
/// personal (headphone) ports are not routed yet.
fn routed(kind: i32) -> bool {
    matches!(kind, 0 | 1 | 2)
}

/// Converts one output buffer to stereo f32 at 48 kHz: S16 or float, mono,
/// stereo or 8 channels (L R C LFE + surround pairs, downmixed at -3 dB),
/// scaled by the port volume (32768 = 0 dB).
unsafe fn to_stereo(p: &Port, ptr: *const u8) -> Vec<f32> {
    let frames = p.len as usize;
    let ch = port_channels(p.format) as usize;
    let float = matches!(p.format & 0xFF, 3 | 4 | 5 | 7);
    let sample = |i: usize| -> f32 {
        if float {
            unsafe { *(ptr as *const f32).add(i) }
        } else {
            (unsafe { *(ptr as *const i16).add(i) }) as f32 / 32768.0
        }
    };
    let vol = |c: usize| p.volume[c.min(7)] as f32 / 32768.0;
    let mut out = Vec::with_capacity(2 * frames);
    const K: f32 = 0.7071;
    for f in 0..frames {
        let b = f * ch;
        let (l, r) = match ch {
            1 => {
                let m = sample(b) * vol(0);
                (m, m)
            }
            2 => (sample(b) * vol(0), sample(b + 1) * vol(1)),
            _ => {
                let c = sample(b + 2) * vol(2) * K;
                let l = sample(b) * vol(0) + c + K * (sample(b + 4) * vol(4) + sample(b + 6) * vol(6));
                let r = sample(b + 1) * vol(1) + c + K * (sample(b + 5) * vol(5) + sample(b + 7) * vol(7));
                (l, r)
            }
        };
        out.push(l);
        out.push(r);
    }
    out
}

static PORTS: Mutex<Vec<Option<Port>>> = Mutex::new(Vec::new());

/// RAD's speaker-gain setter (`titlefix` "stereo movie sound on both front speakers",
/// eboot+0x1e4b340; rdi the voice (its channel count at +0x38), rsi a speaker-by-channel gain
/// matrix (speaker s, channel c at 2s + c for a stereo voice), edx the speaker count). The
/// title's movie player routes track k to speaker k alone, which for a stereo track is the
/// left channel at the front left and the right nowhere: the right channel then goes to the
/// front right with the left's gain. The first calls are logged.
pub extern "sysv64" fn rad_speaker_hook(regs: *const u64, _arg: u32) {
    let r = |i: usize| unsafe { *regs.add(i) };
    let (voice, matrix, count) = (r(7), r(6), r(2) as u32);
    if voice == 0 || matrix == 0 {
        return;
    }
    let ch = unsafe { std::ptr::read_unaligned((voice + 0x38) as *const u32) };
    let n = (count.min(8) * ch.clamp(1, 8)) as usize;
    let at = |i: usize| (matrix + 4 * i as u64) as *mut f32;
    let mut fixed = false;
    if ch == 2 && count >= 2 {
        let g0 = unsafe { std::ptr::read_unaligned(at(0)) };
        let rest_silent = (1..n).all(|i| unsafe { std::ptr::read_unaligned(at(i)) } == 0.0);
        if g0 != 0.0 && rest_silent {
            unsafe { std::ptr::write_unaligned(at(3), g0) };
            fixed = true;
        }
    }
    static N: std::sync::atomic::AtomicU32 = std::sync::atomic::AtomicU32::new(0);
    if N.fetch_add(1, std::sync::atomic::Ordering::Relaxed) < 8 {
        let gains: Vec<String> = (0..n.min(16)).map(|i| format!("{:.2}", unsafe { std::ptr::read_unaligned(at(i)) })).collect();
        crate::info!(
            "RAD speaker gains: voice {voice:#x} ({ch} channels), {count} speakers, matrix [{}]{} (from {})",
            gains.join(" "),
            if fixed { ": the right channel added to the front right" } else { "" },
            crate::platform::loader_hooks::current_guest_frames(3).join(" <- ")
        );
    }
}

/// A raw capture of the AudioOut ports (`start_raw_capture`): per port its
/// channels and samples, and the frames each port still takes.
struct RawCapture {
    dir: std::path::PathBuf,
    n: u32,
    frames: usize,
    ports: std::collections::BTreeMap<i32, (usize, Vec<f32>)>,
}

static RAW: Mutex<Option<RawCapture>> = Mutex::new(None);

/// `audio ports <seconds>` (control): the next seconds of every AudioOut
/// port's buffers as the title wrote them (all channels, as float, before
/// the port volume and the stereo fold) to `<run dir>/audioout-<port>-<n>.wav`.
pub fn start_raw_capture(seconds: f64) -> Result<(), String> {
    let dir = crate::run_dir().ok_or("no run directory")?;
    let mut g = RAW.lock();
    if g.is_some() {
        return Err("an AudioOut port capture is in progress".into());
    }
    static N: std::sync::atomic::AtomicU32 = std::sync::atomic::AtomicU32::new(0);
    *g = Some(RawCapture { dir: dir.to_path_buf(), n: N.fetch_add(1, std::sync::atomic::Ordering::Relaxed), frames: (seconds * 48000.0) as usize, ports: Default::default() });
    crate::info!("AudioOut: capturing {seconds} s of each port's buffers");
    Ok(())
}

/// Adds one output buffer of port `h` to the raw capture in progress.
unsafe fn raw_add(h: i32, p: &Port, ptr: *const u8) {
    let mut g = RAW.lock();
    let Some(rc) = g.as_mut() else { return };
    let ch = port_channels(p.format) as usize;
    let float = matches!(p.format & 0xFF, 3 | 4 | 5 | 7);
    let e = rc.ports.entry(h).or_insert_with(|| (ch, Vec::new()));
    for i in 0..p.len as usize * ch {
        e.1.push(if float { unsafe { *(ptr as *const f32).add(i) } } else { (unsafe { *(ptr as *const i16).add(i) }) as f32 / 32768.0 });
    }
    if rc.ports.values().all(|(c, v)| v.len() / c < rc.frames) {
        return;
    }
    let rc = g.take().unwrap();
    drop(g);
    let _ = std::thread::Builder::new().name("audioout-capture".into()).spawn(move || {
        for (port, (ch, v)) in &rc.ports {
            let path = rc.dir.join(format!("audioout-{port}-{}.wav", rc.n));
            let data = 4 * v.len() as u32;
            let mut b = Vec::with_capacity(46 + data as usize);
            b.extend_from_slice(b"RIFF");
            b.extend_from_slice(&(38 + data).to_le_bytes());
            b.extend_from_slice(b"WAVEfmt ");
            b.extend_from_slice(&18u32.to_le_bytes());
            b.extend_from_slice(&3u16.to_le_bytes());
            b.extend_from_slice(&(*ch as u16).to_le_bytes());
            b.extend_from_slice(&48000u32.to_le_bytes());
            b.extend_from_slice(&(48000 * 4 * *ch as u32).to_le_bytes());
            b.extend_from_slice(&(4 * *ch as u16).to_le_bytes());
            b.extend_from_slice(&32u16.to_le_bytes());
            b.extend_from_slice(&0u16.to_le_bytes());
            b.extend_from_slice(b"data");
            b.extend_from_slice(&data.to_le_bytes());
            for s in v {
                b.extend_from_slice(&s.to_le_bytes());
            }
            match std::fs::write(&path, &b) {
                Ok(()) => crate::info!("AudioOut: port {port} capture written: {} ({} channels, {:.1} s)", path.display(), ch, v.len() as f64 / *ch as f64 / 48000.0),
                Err(e) => crate::warn!("AudioOut: port {port} capture {}: {e}", path.display()),
            }
        }
    });
}

fn port_channels(format: u32) -> u32 {
    match format & 0xFF {
        0 | 3 => 1,
        1 | 4 => 2,
        _ => 8,
    }
}

// ---------------------------------------------------------------------------
// Pad

#[repr(C)]
#[derive(Clone, Copy)]
struct PadData {
    buttons: u32,
    lx: u8,
    ly: u8,
    rx: u8,
    ry: u8,
    l2: u8,
    r2: u8,
    pad0: [u8; 2],
    orientation: [f32; 4],
    acceleration: [f32; 3],
    angular_velocity: [f32; 3],
    touch: [u8; 24],
    connected: u8,
    pad1: [u8; 3],
    timestamp: u64,
    ext: [u8; 16],
    connected_count: u8,
    reserve: [u8; 2],
    unique_len: u8,
    unique: [u8; 12],
}

/// Logs the title's pad buffers and their readers (the first eight buffer
/// and call site pairs).
fn note_pad_buffer(data: *mut PadData, how: &str) {
    static SEEN: parking_lot::Mutex<Vec<(u64, String)>> = parking_lot::Mutex::new(Vec::new());
    // Every read at first, then every 64th (the guest unwind costs ~20 us).
    static N: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);
    let n = N.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
    if (n >= 256 && n % 64 != 0) || SEEN.lock().len() >= 8 {
        return;
    }
    let frames = crate::platform::loader_hooks::current_guest_frames(4);
    let key = (data as u64, frames.first().cloned().unwrap_or_default());
    let mut g = SEEN.lock();
    if g.len() < 8 && !g.contains(&key) {
        g.push(key);
        drop(g);
        crate::info!("pad: {how} into {:#x} [{}] from {}", data as u64, crate::log::thread_name(), frames.join(" <- "));
    }
}

fn read_pad() -> PadData {
    let mut d: PadData = unsafe { std::mem::zeroed() };
    let st = crate::input::pad_state();
    d.buttons = st.buttons;
    d.lx = st.lx;
    d.ly = st.ly;
    d.rx = st.rx;
    d.ry = st.ry;
    d.l2 = st.l2;
    d.r2 = st.r2;
    d.orientation = [0.0, 0.0, 0.0, 1.0];
    d.acceleration = [0.0, 9.8, 0.0];
    d.connected = 1;
    d.connected_count = 1;
    d.timestamp = crate::hle::kernel::time::process_time_us();
    d
}

sysv! {
    fn sceAudioOutInit() -> i32 { 0 }
    fn sceAudioOutOpen(user: i32, kind: i32, index: i32, len: u32, freq: u32, param: u32) -> i32 {
        if len == 0 || len % 256 != 0 || freq != 48000 { return 0x8026_0008u32 as i32; }
        let mut ports = PORTS.lock();
        let p = Port { kind, len, freq, format: param, next: Instant::now(), volume: [32768; 8], stream: crate::audio::new_stream() };
        let idx = match ports.iter().position(|p| p.is_none()) { Some(i) => { ports[i] = Some(p); i } None => { ports.push(Some(p)); ports.len() - 1 } };
        crate::info!("sceAudioOutOpen type {kind} len {len} freq {freq} format {:#x} ({} ch) -> port {}", param, port_channels(param), idx + 1);
        idx as i32 + 1
    }
    fn sceAudioOutClose(h: i32) -> i32 {
        let mut ports = PORTS.lock();
        match ports.get_mut((h - 1) as usize) { Some(p @ Some(_)) => { *p = None; 0 } _ => 0x8026_0004u32 as i32 }
    }
    /// Routed ports feed the host mixer (blocking while 3 buffers are queued,
    /// which paces the caller by the output clock); others are paced sinks.
    fn sceAudioOutOutput(h: i32, ptr: *const u8) -> i32 {
        let (port, wait) = {
            let mut ports = PORTS.lock();
            let Some(Some(p)) = ports.get_mut((h - 1) as usize) else { return 0x8026_0004u32 as i32 };
            let dur = Duration::from_secs_f64(p.len as f64 / p.freq as f64);
            let now = Instant::now();
            let start = if p.next > now { p.next } else { now };
            p.next = start + dur;
            (p.clone(), start.saturating_duration_since(now))
        };
        if !ptr.is_null() {
            raw_add(h, &port, ptr);
        }
        if routed(port.kind) {
            if !ptr.is_null() {
                crate::audio::push_stream(port.stream, to_stereo(&port, ptr), 3);
            }
            return 0;
        }
        if !wait.is_zero() { std::thread::sleep(wait); }
        0
    }
    fn sceAudioOutSetVolume(h: i32, flags: u32, vol: *const i32) -> i32 {
        let mut ports = PORTS.lock();
        let Some(Some(p)) = ports.get_mut((h - 1) as usize) else { return 0x8026_0004u32 as i32 };
        for i in 0..8 { if flags & (1 << i) != 0 && !vol.is_null() { p.volume[i] = *vol.add(i); } }
        static N: std::sync::atomic::AtomicU32 = std::sync::atomic::AtomicU32::new(0);
        if N.fetch_add(1, std::sync::atomic::Ordering::Relaxed) < 16 {
            crate::info!("sceAudioOutSetVolume port {h} flags {flags:#x}: volumes now {:?}", p.volume);
        }
        0
    }
    fn sceAudioOutGetPortState(h: i32, st: *mut u8) -> i32 {
        let ports = PORTS.lock();
        let Some(Some(p)) = ports.get((h - 1) as usize) else { return 0x8026_0004u32 as i32 };
        std::ptr::write_bytes(st, 0, 32);
        *(st as *mut u16) = 1; // connected primary
        *st.add(2) = port_channels(p.format).min(8) as u8;
        *(st.add(4) as *mut i16) = 127;
        0
    }

    // ---- Pad
    fn scePadOpen(user: i32, kind: i32, index: i32, param: *const u8) -> i32 {
        if user != super::app::USER_ID { return 0x80920005u32 as i32; }
        crate::info!("scePadOpen user {user} type {kind} index {index} -> 1 (a controller {}; keyboard via window)", if crate::input::xinput_state().is_some() { "connected" } else { "not connected yet" });
        1
    }
    fn scePadReadState(h: i32, data: *mut PadData) -> i32 {
        if h != 1 || data.is_null() { return 0x80920003u32 as i32; }
        note_pad_buffer(data, "scePadReadState");
        std::ptr::write_unaligned(data, read_pad());
        0
    }
    /// Buffered read: this runtime samples the current state, so one entry is
    /// returned per call (ScePadData is 120 bytes, as the title's buffer confirms).
    fn scePadRead(h: i32, data: *mut PadData, num: i32) -> i32 {
        if h != 1 || data.is_null() || num < 1 { return 0x80920003u32 as i32; }
        note_pad_buffer(data, "scePadRead");
        std::ptr::write_unaligned(data, read_pad());
        1
    }
    fn scePadGetHandle(user: i32, kind: i32, index: i32) -> i32 {
        if user != super::app::USER_ID { return 0x80920005u32 as i32; }
        1
    }
    /// The light bar's colour (ScePadLightBarParam: red, green, blue): a PlayStation controller's
    /// light bar (`pspad`); XInput controllers have none. Adaptive triggers and motion sensors:
    /// accepted, not mapped (logged once).
    fn scePadSetLightBar(h: i32, color: *const u8) -> i32 {
        if !color.is_null() && !crate::input::SYSTEM_OFF.load(std::sync::atomic::Ordering::Relaxed) {
            crate::pspad::set_lightbar(Some([*color, *color.add(1), *color.add(2)]));
        }
        0
    }
    fn scePadSetTriggerEffect(h: i32, param: *const u8) -> i32 { crate::hle::stub_log("scePadSetTriggerEffect (no adaptive triggers on the host pad)", 0); 0 }
    fn scePadSetMotionSensorState(h: i32, enable: i32) -> i32 { crate::hle::stub_log("scePadSetMotionSensorState (no motion sensors mapped)", 0); 0 }
    fn scePadGetControllerInformation(h: i32, info: *mut u8) -> i32 {
        if info.is_null() { return 0x80920001u32 as i32; }
        std::ptr::write_bytes(info, 0, 28);
        *(info as *mut f32) = 44.86;
        *(info.add(4) as *mut u16) = 1920;
        *(info.add(6) as *mut u16) = 943;
        *info.add(8) = 13;
        *info.add(9) = 13;
        *info.add(10) = 0; // local connection
        *info.add(11) = 1; // connected count
        *info.add(12) = 1; // connected
        0
    }
    fn scePadSetVibration(h: i32, param: *const u8) -> i32 {
        // (The shader preparation's self-play leaves the player's controller alone.)
        // (Every controller: the XInput ones and the PlayStation ones, `input::rumble`.)
        if !param.is_null() && !crate::input::SYSTEM_OFF.load(std::sync::atomic::Ordering::Relaxed) {
            crate::input::rumble((*param as u16) * 257, (*param.add(1) as u16) * 257);
        }
        0
    }
}

fixed! {
    scePadInit => 0,
    scePadClose => 0,
    scePadResetOrientation => 0,
    scePadSetAngularVelocityDeadbandState => 0,
    scePadSetTiltCorrectionState => 0,
    sceMouseInit => 0,
    sceMouseOpen => 0x80DF_0003u32,
    sceMouseClose => 0x80DF_0003u32,
    sceMouseRead => 0x80DF_0003u32,
    sceAudioInOpen => 0x8026_0007u32,
    sceAudioInClose => 0x8026_0007u32,
    sceAudioInInput => 0x8026_0007u32,
    sceVoiceInit => 0,
    sceVoiceEnd => 0,
    sceVoiceStart => 0,
    sceVoiceStop => 0,
    sceVoiceCreatePort => 0x8082_0001u32,
    sceVoiceDeletePort => 0x8082_0001u32,
    sceVoiceConnectIPortToOPort => 0x8082_0001u32,
    sceVoiceDisconnectIPortFromOPort => 0x8082_0001u32,
    sceVoiceGetPortInfo => 0x8082_0001u32,
    sceVoiceReadFromOPort => 0x8082_0001u32,
    sceVoiceWriteToIPort => 0x8082_0001u32,
}

pub fn register(r: &mut Registry) {
    reg!(r, "libSceAudioOut", [
        sceAudioOutInit, sceAudioOutOpen, sceAudioOutClose, sceAudioOutOutput, sceAudioOutSetVolume, sceAudioOutGetPortState,
    ]);
    reg!(r, "libScePad", [
        scePadInit, scePadOpen, scePadClose, scePadReadState, scePadGetControllerInformation, scePadSetVibration,
        scePadRead, scePadGetHandle, scePadSetLightBar, scePadSetTriggerEffect, scePadSetMotionSensorState,
        scePadResetOrientation, scePadSetAngularVelocityDeadbandState, scePadSetTiltCorrectionState,
    ]);
    reg!(r, "libSceMouse", [sceMouseInit, sceMouseOpen, sceMouseClose, sceMouseRead]);
    reg!(r, "libSceAudioIn", [sceAudioInOpen, sceAudioInClose, sceAudioInInput]);
    reg!(r, "libSceVoice", [
        sceVoiceInit, sceVoiceEnd, sceVoiceStart, sceVoiceStop, sceVoiceCreatePort, sceVoiceDeletePort,
        sceVoiceConnectIPortToOPort, sceVoiceDisconnectIPortFromOPort, sceVoiceGetPortInfo, sceVoiceReadFromOPort,
        sceVoiceWriteToIPort,
    ]);
}
