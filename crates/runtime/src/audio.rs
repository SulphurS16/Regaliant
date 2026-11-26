//! Host audio output: 48 kHz f32 streams mixed into a WASAPI shared-mode
//! render client (event driven) of 2, 6 or 8 channels (`OUT_CHANNELS`: the
//! AudioOut2 renderer's layout; FL FR FC LFE BL BR SL SR order). Producers
//! (AudioOut2 contexts, AudioOut ports) push fixed-size grains of their own
//! channel count into their stream (stereo grains go to FL and FR) and block
//! while the stream already holds `depth` grains, which paces the title's
//! audio threads by the real output clock. Without an output device the same
//! queues drain on a timer (null device, logged).

use parking_lot::{Condvar, Mutex};
use std::collections::VecDeque;
use std::sync::OnceLock;
use std::sync::atomic::{AtomicU64, Ordering};

pub const RATE: u32 = 48_000;

/// The output's channels (2, 6 or 8; the AudioOut2 renderer sets it, the output thread opens
/// its client with it), and the output device's own (its mix format; 0 before it opened).
pub static OUT_CHANNELS: std::sync::atomic::AtomicU32 = std::sync::atomic::AtomicU32::new(2);
pub static DEVICE_CHANNELS: std::sync::atomic::AtomicU32 = std::sync::atomic::AtomicU32::new(0);
/// The open device's form factor as Windows names it (`EndpointFormFactor`: 1 speakers,
/// 3 headphones, 5 headset, 9 a display's HDMI; u32::MAX unknown).
pub static DEVICE_FORM: std::sync::atomic::AtomicU32 = std::sync::atomic::AtomicU32::new(u32::MAX);

/// Windows' spatial sound (Windows Sonic, Dolby Atmos) on the open device: 0 not available,
/// 1 available (a 7.1.4 static stream opened), 2 not tried yet.
pub static SPATIAL: std::sync::atomic::AtomicU32 = std::sync::atomic::AtomicU32::new(2);

/// Windows' spatial sound streams are tried only with `DS_SPATIAL_AUDIO=1` (diagnosis): runs that
/// used one ended with heap corruption in Windows' heap (FINDINGS.md 2026-11-06 night).
pub fn spatial_allowed() -> bool {
    static ON: OnceLock<bool> = OnceLock::new();
    *ON.get_or_init(|| std::env::var("DS_SPATIAL_AUDIO").ok().as_deref() == Some("1"))
}

/// The output device chosen in the PC tab (its endpoint id; None: the system's default), and a
/// count of changes (the output thread opens the device again).
static DEVICE_ID: Mutex<Option<String>> = Mutex::new(None);
static DEVICE_GEN: std::sync::atomic::AtomicU32 = std::sync::atomic::AtomicU32::new(0);

/// The active render devices: endpoint id and name, sorted by name (listed once).
pub fn output_devices() -> &'static [(String, String)] {
    static LIST: OnceLock<Vec<(String, String)>> = OnceLock::new();
    LIST.get_or_init(|| match unsafe { host::devices() } {
        Ok(mut v) => {
            v.sort_by(|a, b| a.1.cmp(&b.1));
            crate::info!("audio: output devices: {}", v.iter().map(|(_, n)| n.as_str()).collect::<Vec<_>>().join(", "));
            v
        }
        Err(e) => {
            crate::warn!("audio: output devices not listed ({e})");
            Vec::new()
        }
    })
}

/// The output device to open (None: the system's default): applied at the output's next wake.
pub fn set_output_device(id: Option<String>) {
    let mut g = DEVICE_ID.lock();
    if *g != id {
        crate::info!("audio: output device {}", id.as_deref().and_then(|i| output_devices().iter().find(|d| d.0 == i)).map(|d| d.1.as_str()).unwrap_or("(the system's default)"));
        *g = id;
        DEVICE_GEN.fetch_add(1, Ordering::Relaxed);
    }
}

pub fn output_device() -> Option<String> {
    DEVICE_ID.lock().clone()
}

#[derive(Default)]
struct Stream {
    /// Grains: their channel count and interleaved samples.
    grains: VecDeque<(usize, Vec<f32>)>,
    /// Frames already consumed from the front grain.
    offset: usize,
    /// Frames pushed so far (diagnostics).
    pushed: u64,
    /// Peak absolute sample value pushed (diagnostics: 0 = silence).
    peak: f32,
    /// Whether the last mix took frames from it (a stream that did and
    /// has none now ran dry: an underrun heard as a gap).
    fed: bool,
    /// Underruns: mixes that found it dry while it was playing, the frames
    /// left silent, and the largest mix request then (diagnostics).
    dry: u64,
    dry_frames: u64,
    dry_request: usize,
}

struct Mixer {
    streams: Mutex<Vec<Stream>>,
    cv: Condvar,
}

static MIXER: OnceLock<Mixer> = OnceLock::new();
pub static UNDERRUN_FRAMES: AtomicU64 = AtomicU64::new(0);
pub static PLAYED_FRAMES: AtomicU64 = AtomicU64::new(0);
/// UNDERRUN_FRAMES and PLAYED_FRAMES at the last `profile reset`.
static AT_RESET: [AtomicU64; 2] = [AtomicU64::new(0), AtomicU64::new(0)];

/// Starts a profile window (`profile reset`).
pub fn reset_window() {
    AT_RESET[0].store(UNDERRUN_FRAMES.load(Ordering::Relaxed), Ordering::Relaxed);
    AT_RESET[1].store(PLAYED_FRAMES.load(Ordering::Relaxed), Ordering::Relaxed);
}

/// Played and underrun seconds since `profile reset` (profile report).
pub fn window_line() -> String {
    let u = UNDERRUN_FRAMES.load(Ordering::Relaxed).saturating_sub(AT_RESET[0].load(Ordering::Relaxed));
    let p = PLAYED_FRAMES.load(Ordering::Relaxed).saturating_sub(AT_RESET[1].load(Ordering::Relaxed));
    format!("    audio in the window: {:.1} s played, {:.2} s underrun\n", p as f64 / RATE as f64, u as f64 / RATE as f64)
}

/// A capture of the mix as played (`audio capture <seconds>`): 32-bit float
/// stereo WAV in the run directory.
struct Capture {
    path: std::path::PathBuf,
    frames_left: usize,
    samples: Vec<f32>,
    /// Samples the soft knee compressed (above `SOFT_KNEE` in the mix).
    knee: u64,
}

static CAPTURE: Mutex<Option<Capture>> = Mutex::new(None);
static CAPTURING: std::sync::atomic::AtomicBool = std::sync::atomic::AtomicBool::new(false);

/// Starts capturing the next `seconds` of the mix to `<run dir>/audio-<n>.wav`.
pub fn start_capture(seconds: f64) -> Result<(), String> {
    let dir = crate::run_dir().ok_or("no run directory")?;
    let mut c = CAPTURE.lock();
    if c.is_some() {
        return Err("a capture is in progress".into());
    }
    let path = (0..).map(|n| dir.join(format!("audio-{n}.wav"))).find(|p| !p.exists()).unwrap();
    let frames = (seconds * RATE as f64) as usize;
    crate::info!("audio: capturing {seconds} s of the mix to {}", path.display());
    *c = Some(Capture { path, frames_left: frames, samples: Vec::with_capacity(2 * frames), knee: 0 });
    CAPTURING.store(true, Ordering::Relaxed);
    Ok(())
}

/// Appends mixed frames (before the soft knee) to the capture in progress
/// and writes the file once it is complete.
fn capture(mix: &[f32]) {
    let mut g = CAPTURE.lock();
    let Some(c) = g.as_mut() else { return };
    let n = (mix.len() / 2).min(c.frames_left);
    for &v in &mix[..2 * n] {
        if v.abs() > SOFT_KNEE {
            c.knee += 1;
        }
        c.samples.push(soft_knee(v));
    }
    c.frames_left -= n;
    if c.frames_left > 0 {
        return;
    }
    let c = g.take().unwrap();
    CAPTURING.store(false, Ordering::Relaxed);
    drop(g);
    // Written on a thread of its own (the output thread must not wait for the file).
    let _ = std::thread::Builder::new().name("audio-capture".into()).spawn(move || {
        let peak = c.samples.iter().fold(0f32, |m, v| m.max(v.abs()));
        let rms = (c.samples.iter().map(|&v| v as f64 * v as f64).sum::<f64>() / c.samples.len().max(1) as f64).sqrt();
        match write_wav(&c.path, &c.samples) {
            Ok(()) => crate::info!(
                "audio: capture written: {} ({:.1} s, rms {rms:.4}, peak {peak:.3}, {} samples above the knee)",
                c.path.display(),
                c.samples.len() as f64 / 2.0 / RATE as f64,
                c.knee
            ),
            Err(e) => crate::warn!("audio: capture {}: {e}", c.path.display()),
        }
    });
}

/// A stereo f32 WAV at the output rate (diagnosis captures).
pub fn write_wav_file(path: &std::path::Path, samples: &[f32]) -> std::io::Result<()> {
    write_wav(path, samples)
}

fn write_wav(path: &std::path::Path, samples: &[f32]) -> std::io::Result<()> {
    let data = 4 * samples.len() as u32;
    let mut b = Vec::with_capacity(58 + data as usize);
    b.extend_from_slice(b"RIFF");
    b.extend_from_slice(&(50 + data).to_le_bytes());
    b.extend_from_slice(b"WAVEfmt ");
    b.extend_from_slice(&18u32.to_le_bytes());
    b.extend_from_slice(&3u16.to_le_bytes()); // IEEE float
    b.extend_from_slice(&2u16.to_le_bytes());
    b.extend_from_slice(&RATE.to_le_bytes());
    b.extend_from_slice(&(RATE * 8).to_le_bytes());
    b.extend_from_slice(&8u16.to_le_bytes());
    b.extend_from_slice(&32u16.to_le_bytes());
    b.extend_from_slice(&0u16.to_le_bytes());
    b.extend_from_slice(b"fact");
    b.extend_from_slice(&4u32.to_le_bytes());
    b.extend_from_slice(&(samples.len() as u32 / 2).to_le_bytes());
    b.extend_from_slice(b"data");
    b.extend_from_slice(&data.to_le_bytes());
    for v in samples {
        b.extend_from_slice(&v.to_le_bytes());
    }
    std::fs::write(path, b)
}


fn mixer() -> &'static Mixer {
    MIXER.get_or_init(|| {
        let m = Mixer { streams: Mutex::new(Vec::new()), cv: Condvar::new() };
        crate::thread::spawn_host("audio-out", 1 << 20, Box::new(output_thread)).expect("audio thread");
        m
    })
}

pub fn init() {
    mixer();
}

/// Allocates a producer stream id.
pub fn new_stream() -> usize {
    let mut s = mixer().streams.lock();
    s.push(Stream::default());
    s.len() - 1
}

fn ensure(streams: &mut Vec<Stream>, id: usize) {
    while streams.len() <= id {
        streams.push(Stream::default());
    }
}

// SDL pulls in device-sized bursts, unlike a steady one-grain consumer. V1 movie ports
// must hold a complete refill plus one mix chunk while their producer is waking up.
#[cfg(target_os = "linux")]
static LEGACY_BUFFER_FRAMES: std::sync::atomic::AtomicUsize = std::sync::atomic::AtomicUsize::new(2528);

pub fn legacy_queue_depth(grain_frames: usize) -> usize {
    #[cfg(windows)]
    { let _ = grain_frames; 3 }
    #[cfg(target_os = "linux")]
    { LEGACY_BUFFER_FRAMES.load(Ordering::Relaxed).div_ceil(grain_frames.max(1)).max(3) }
}

/// Queues one stereo interleaved grain on stream `id`, blocking while the
/// stream already holds `depth` grains.
pub fn push_stream(id: usize, grain: Vec<f32>, depth: usize) {
    push_stream_ch(id, grain, 2, depth)
}

/// Queues one grain of `channels` interleaved channels on stream `id` (as `push_stream`).
pub fn push_stream_ch(id: usize, grain: Vec<f32>, channels: usize, depth: usize) {
    // (The title mixes in its job system: any of its 13 job workers pushes
    // here, so raising producers raised the spinning workers to the GPU
    // front end's level and undid `gpu::prio`, run 20261013-123910.)
    let m = mixer();
    let mut s = m.streams.lock();
    ensure(&mut s, id);
    let _w = if s[id].grains.len() >= depth.max(1) { Some(crate::thread::waiting("audio output queue", id as u64, None)) } else { None };
    while s[id].grains.len() >= depth.max(1) {
        m.cv.wait(&mut s);
    }
    s[id].pushed += (grain.len() / channels.max(1)) as u64;
    let peak = grain.iter().fold(0f32, |m, v| m.max(v.abs()));
    if peak > s[id].peak {
        s[id].peak = peak;
    }
    s[id].grains.push_back((channels.max(1), grain));
}

/// One line per stream for status reports.
pub fn status() -> String {
    let Some(m) = MIXER.get() else { return String::new() };
    let Some(s) = m.streams.try_lock() else { return String::new() };
    let mut out = format!(
        "    audio: {:.1} s played, {:.2} s underrun\n",
        PLAYED_FRAMES.load(Ordering::Relaxed) as f64 / RATE as f64,
        UNDERRUN_FRAMES.load(Ordering::Relaxed) as f64 / RATE as f64
    );
    for (i, st) in s.iter().enumerate() {
        out.push_str(&format!(
            "    audio stream {i}: {:.1} s pushed, {} grain(s) queued, peak {:.3}; ran dry {} times while playing ({:.1} ms left silent, largest request {} frames)\n",
            st.pushed as f64 / RATE as f64,
            st.grains.len(),
            st.peak,
            st.dry,
            st.dry_frames as f64 * 1000.0 / RATE as f64,
            st.dry_request
        ));
    }
    out
}

/// Grains queued on stream `id`.
pub fn queued_grains(id: usize) -> usize {
    let s = mixer().streams.lock();
    s.get(id).map(|x| x.grains.len()).unwrap_or(0)
}

/// Where channel `k` of a `gch`-channel layout goes in an `och`-channel one (layouts: 1 mono, 2
/// FL FR, 6 FL FR FC LFE BL BR, 8 the same then SL SR, 12 the same then TFL TFR TBL TBR): its
/// targets and gains. A missing speaker is folded into its neighbours at 0.707 (LFE left out of
/// stereo).
fn fold(k: usize, gch: usize, och: usize) -> &'static [(usize, f32)] {
    const H: f32 = std::f32::consts::FRAC_1_SQRT_2;
    match (och, k) {
        // Stereo: fronts as they are, centre and everything left or right of the listener at 0.707.
        (2, 0) => &[(0, 1.0)],
        (2, 1) => &[(1, 1.0)],
        (2, 2) => &[(0, H), (1, H)],
        (2, 3) => &[],
        (2, 4 | 6 | 8 | 10) => &[(0, H)],
        (2, 5 | 7 | 9 | 11) => &[(1, H)],
        // 5.1 from 7.1 or 7.1.4: the sides into the backs, the heights into the fronts and backs.
        (6, 0..=5) => match k {
            0 => &[(0, 1.0)],
            1 => &[(1, 1.0)],
            2 => &[(2, 1.0)],
            3 => &[(3, 1.0)],
            4 => &[(4, 1.0)],
            _ => &[(5, 1.0)],
        },
        (6, 6) => &[(4, H)],
        (6, 7) => &[(5, H)],
        (6, 8) => &[(0, H)],
        (6, 9) => &[(1, H)],
        (6, 10) => &[(4, H)],
        (6, 11) => &[(5, H)],
        // 7.1 from 7.1.4: the heights into the fronts and backs; 5.1's surrounds to the backs.
        (8, 8) => &[(0, H)],
        (8, 9) => &[(1, H)],
        (8, 10) => &[(4, H)],
        (8, 11) => &[(5, H)],
        (8 | 12, _) if k < 8 || gch == 12 => match k {
            0 => &[(0, 1.0)],
            1 => &[(1, 1.0)],
            2 => &[(2, 1.0)],
            3 => &[(3, 1.0)],
            4 => &[(4, 1.0)],
            5 => &[(5, 1.0)],
            6 => &[(6, 1.0)],
            7 => &[(7, 1.0)],
            8 => &[(8, 1.0)],
            9 => &[(9, 1.0)],
            10 => &[(10, 1.0)],
            _ => &[(11, 1.0)],
        },
        _ => &[],
    }
}

/// Adds one frame `g` of `gch` channels into output frame `o` of `och` channels: equal layouts
/// directly, mono to the centre (both sides in stereo), stereo into FL and FR, the others
/// through `fold`.
fn add_frame(o: &mut [f32], och: usize, g: &[f32], gch: usize) {
    if gch == och {
        for (a, b) in o.iter_mut().zip(g) {
            *a += *b;
        }
    } else if gch == 1 {
        if och == 2 {
            o[0] += g[0] * 0.7071;
            o[1] += g[0] * 0.7071;
        } else {
            o[2] += g[0];
        }
    } else if gch == 2 {
        o[0] += g[0];
        o[1] += g[1];
    } else {
        for (k, &v) in g.iter().enumerate() {
            for &(t, w) in fold(k, gch, och) {
                if t < och {
                    o[t] += v * w;
                }
            }
        }
    }
}

/// Mixes `frames` frames of `och` channels from all streams into `out` (zero-filled first).
fn mix_into(out: &mut [f32], frames: usize, och: usize) {
    out[..och * frames].fill(0.0);
    let m = mixer();
    let mut s = m.streams.lock();
    let mut starved = false;
    for (id, st) in s.iter_mut().enumerate() {
        let mut done = 0usize;
        while done < frames {
            let Some((gch, front)) = st.grains.front() else {
                // Dry while playing: frames left silent in this mix.
                if done > 0 || st.fed {
                    starved = true;
                    st.dry += 1;
                    st.dry_frames += (frames - done) as u64;
                    st.dry_request = st.dry_request.max(frames);
                    static LOGGED: AtomicU64 = AtomicU64::new(0);
                    if LOGGED.fetch_add(1, Ordering::Relaxed) < 40 {
                        crate::info!("audio: stream {id} ran dry: {} of {frames} frames silent ({} pushed so far)", frames - done, st.pushed);
                    }
                }
                break;
            };
            let gch = *gch;
            let avail = front.len() / gch - st.offset;
            let n = avail.min(frames - done);
            for i in 0..n {
                let g = &front[gch * (st.offset + i)..gch * (st.offset + i + 1)];
                add_frame(&mut out[och * (done + i)..och * (done + i + 1)], och, g, gch);
            }
            done += n;
            st.offset += n;
            if st.offset * gch >= front.len() {
                st.grains.pop_front();
                st.offset = 0;
            }
        }
        st.fed = done > 0;
    }
    drop(s);
    m.cv.notify_all();
    if starved {
        UNDERRUN_FRAMES.fetch_add(frames as u64, Ordering::Relaxed);
    }
    if CAPTURING.load(Ordering::Relaxed) {
        // Captures are stereo (a surround mix folded down).
        if och == 2 {
            capture(&out[..2 * frames]);
        } else {
            let mut st = vec![0f32; 2 * frames];
            for f in 0..frames {
                add_frame(&mut st[2 * f..2 * f + 2], 2, &out[och * f..och * (f + 1)], och);
            }
            capture(&st);
        }
    }
    for v in out[..och * frames].iter_mut() {
        *v = soft_knee(*v);
    }
    PLAYED_FRAMES.fetch_add(frames as u64, Ordering::Relaxed);
}

/// A soft knee above 0.8 instead of clipping at full scale (loud transients
/// of the 3D mix reached ~1.2 in combat).
fn soft_knee(v: f32) -> f32 {
    let a = v.abs();
    if a > SOFT_KNEE { v.signum() * (SOFT_KNEE + (1.0 - SOFT_KNEE) * ((a - SOFT_KNEE) / (1.0 - SOFT_KNEE)).tanh()) } else { v }
}

/// Output level from which samples are compressed smoothly toward full
/// scale (`mix_into`).
const SOFT_KNEE: f32 = 0.8;

/// No sound at all (the shader preparation's self-play, `prepplay`): the streams drain on the
/// null device's clock and no output device is opened.
pub static SILENT: std::sync::atomic::AtomicBool = std::sync::atomic::AtomicBool::new(false);

fn output_thread() {
    crate::log::set_thread_name("audio-out");
    if SILENT.load(Ordering::Relaxed) {
        crate::info!("audio: silent (no output device opened)");
        null_device();
    }
    // Little work per wake-up, but late work is heard: above the title's
    // threads and the GPU front end's (`gpu::prio`).
    host::priority();
    match host::run() {
        Ok(()) => {}
        Err(e) => {
            crate::warn!("audio: no host output device ({e}); using a paced null device (silent)");
            null_device();
        }
    }
}

/// Drains the streams in real time without producing sound.
fn null_device() {
    let frames = 480usize; // 10 ms
    let mut buf = vec![0f32; 8 * frames];
    let start = std::time::Instant::now();
    let mut played = 0u64;
    loop {
        let och = (OUT_CHANNELS.load(Ordering::Relaxed) as usize).clamp(2, 8);
        mix_into(&mut buf, frames, och);
        played += frames as u64;
        let due = start + std::time::Duration::from_secs_f64(played as f64 / RATE as f64);
        let now = std::time::Instant::now();
        if due > now {
            std::thread::sleep(due - now);
        }
    }
}

#[cfg(windows)]
#[path = "platform/windows/audio.rs"]
mod host;
#[cfg(target_os = "linux")]
#[path = "platform/linux/audio.rs"]
mod host;
#[cfg(target_os = "linux")]
pub use host::{initialize, smoke_test};

#[cfg(all(test, target_os = "linux"))]
mod movie_tests {
    // The actual v1 queue and mixer, with no output device or sleeps. A 1024-frame device
    // consumption causes up to three 480-frame mixes before Bink's producer runs again.
    #[test]
    fn movie_grains_survive_device_refill_bursts() {
        // Initialize only the queue: a real/null output thread would race our synthetic clock.
        assert!(super::MIXER.set(super::Mixer { streams: parking_lot::Mutex::new(Vec::new()), cv: parking_lot::Condvar::new() }).is_ok());
        let id = super::new_stream();
        let depth = super::legacy_queue_depth(256);
        let mut out = [0.0; 960];
        for _ in 0..100 {
            while super::queued_grains(id) < depth {
                super::push_stream(id, vec![0.25; 512], depth);
            }
            for _ in 0..3 {
                super::mix_into(&mut out, 480, 2);
                assert!(out.iter().all(|v| *v == 0.25));
            }
        }
        let mut streams = super::mixer().streams.lock();
        assert_eq!(streams[id].dry_frames, 0);
        eprintln!("movie queue: 100 device refill bursts, 144000 frames, depth={depth}, gaps={} frames", streams[id].dry_frames);
        streams[id] = super::Stream::default();
    }
}
