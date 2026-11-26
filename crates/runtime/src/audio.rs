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
    LIST.get_or_init(|| match unsafe { wasapi::devices() } {
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

/// No sound at all (the shader preparation's self-play, `prepplay`; test runs, `DS_SILENT=1`,
/// which scripts/gpu-run.ps1 and the test drivers set unless a test asks for sound): the streams
/// drain on the null device's clock and no output device is opened. The mix is made as in play,
/// so `audio capture` and the stream statistics still work.
pub static SILENT: std::sync::atomic::AtomicBool = std::sync::atomic::AtomicBool::new(false);

fn output_thread() {
    crate::log::set_thread_name("audio-out");
    if std::env::var("DS_SILENT").is_ok_and(|v| v.trim() == "1") {
        SILENT.store(true, Ordering::Relaxed);
    }
    if SILENT.load(Ordering::Relaxed) {
        crate::info!("audio: silent (no output device opened)");
        null_device();
    }
    // Little work per wake-up, but late work is heard: above the title's
    // threads and the GPU front end's (`gpu::prio`).
    unsafe {
        use windows_sys::Win32::System::Threading::{GetCurrentThread, SetThreadPriority, THREAD_PRIORITY_HIGHEST};
        SetThreadPriority(GetCurrentThread(), THREAD_PRIORITY_HIGHEST);
    }
    match wasapi::run() {
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

mod wasapi {
    use windows::Win32::Foundation::{CloseHandle, WAIT_OBJECT_0};
    use windows::Win32::Media::Audio::*;
    use windows::Win32::Media::KernelStreaming::{
        SPEAKER_BACK_LEFT, SPEAKER_BACK_RIGHT, SPEAKER_FRONT_CENTER, SPEAKER_FRONT_LEFT, SPEAKER_FRONT_RIGHT, SPEAKER_LOW_FREQUENCY, SPEAKER_SIDE_LEFT, SPEAKER_SIDE_RIGHT, WAVE_FORMAT_EXTENSIBLE,
    };
    use windows::Win32::Media::Multimedia::KSDATAFORMAT_SUBTYPE_IEEE_FLOAT;
    use windows::Win32::System::Com::{CLSCTX_ALL, COINIT_MULTITHREADED, CoCreateInstance, CoInitializeEx};
    use windows::Win32::System::Threading::{CreateEventW, WaitForSingleObject};

    /// Runs the output, opening the client again whenever the wanted channel count changes
    /// (`OUT_CHANNELS`; 12: Windows' spatial sound, a 7.1.4 static stream); returns on a lost
    /// device.
    pub fn run() -> Result<(), String> {
        unsafe {
            CoInitializeEx(None, COINIT_MULTITHREADED).ok().map_err(|e| format!("CoInitializeEx: {e}"))?;
        }
        loop {
            let och = super::OUT_CHANNELS.load(std::sync::atomic::Ordering::Relaxed) as usize;
            if och == 12 && super::SPATIAL.load(std::sync::atomic::Ordering::Relaxed) != 0 && super::spatial_allowed() {
                match unsafe { spatial_session() } {
                    Ok(true) => continue,
                    Ok(false) => return Ok(()),
                    Err(e) => {
                        // Not available on this device (spatial sound off in Windows): the renderer
                        // picks another layout at its next grain; a stereo client meanwhile.
                        if super::SPATIAL.swap(0, std::sync::atomic::Ordering::Relaxed) != 0 {
                            crate::info!("audio: Windows spatial sound not available ({e}): the 3D sound is rendered here");
                        }
                    }
                }
            }
            match unsafe { session(och.max(2)) } {
                Ok(true) => continue,
                Ok(false) => return Ok(()),
                Err(e) => return Err(e),
            }
        }
    }

    /// The device the output opens (the PC tab's, else the system's default).
    unsafe fn open_device(en: &IMMDeviceEnumerator) -> Result<IMMDevice, String> {
        unsafe {
            let chosen = super::output_device().and_then(|id| {
                let w: Vec<u16> = id.encode_utf16().chain(std::iter::once(0)).collect();
                en.GetDevice(windows::core::PCWSTR(w.as_ptr())).ok()
            });
            match chosen {
                Some(d) => Ok(d),
                None => en.GetDefaultAudioEndpoint(eRender, eConsole).map_err(|e| format!("default endpoint: {e}")),
            }
        }
    }

    /// A spatial stream kept for the process's life: stopped when another output is wanted and
    /// started again when spatial sound is wanted on the same device. (Releasing one ended the
    /// process with heap corruption in Windows' heap, with or without our writes to its buffers:
    /// FINDINGS.md 2026-11-06 night.)
    struct Kept {
        generation: u32,
        _client: ISpatialAudioClient,
        _formats: IAudioFormatEnumerator,
        stream: ISpatialAudioObjectRenderStream,
        objects: Vec<ISpatialAudioObject>,
        event: windows::Win32::Foundation::HANDLE,
    }

    thread_local! {
        static KEPT: std::cell::RefCell<Vec<Kept>> = const { std::cell::RefCell::new(Vec::new()) };
    }

    /// Windows' spatial sound: a 7.1.4 static-object stream fed from the mixer's 12 channels
    /// (FL FR FC LFE BL BR SL SR TFL TFR TBL TBR). Ok(true) when the wanted output changed.
    unsafe fn spatial_session() -> Result<bool, String> {
        use std::sync::atomic::Ordering::Relaxed;
        let generation = super::DEVICE_GEN.load(Relaxed);
        let found = KEPT.with(|k| {
            let mut k = k.borrow_mut();
            k.iter().position(|x| x.generation == generation).map(|i| k.remove(i))
        });
        let kept = match found {
            Some(k) => {
                unsafe { k.stream.Start() }.map_err(|e| format!("restart: {e}"))?;
                crate::info!("audio: Windows spatial sound: the 7.1.4 stream started again");
                k
            }
            None => unsafe { spatial_open(generation)? },
        };
        let r = unsafe { spatial_run(&kept) };
        let _ = unsafe { kept.stream.Stop() };
        crate::info!("audio: Windows spatial sound stream stopped (kept)");
        KEPT.with(|k| k.borrow_mut().push(kept));
        r
    }

    /// A new 7.1.4 static-object stream on the output device, started.
    unsafe fn spatial_open(generation: u32) -> Result<Kept, String> {
        use std::sync::atomic::Ordering::Relaxed;
        use windows::Win32::System::Com::StructuredStorage::PROPVARIANT;
        use windows::Win32::System::Variant::VT_BLOB;
        unsafe {
            let en: IMMDeviceEnumerator = CoCreateInstance(&MMDeviceEnumerator, None, CLSCTX_ALL).map_err(|e| format!("device enumerator: {e}"))?;
            let dev = open_device(&en)?;
            super::DEVICE_FORM.store(form_factor(&dev).unwrap_or(u32::MAX), Relaxed);
            let client: ISpatialAudioClient = dev.Activate(CLSCTX_ALL, None).map_err(|e| format!("spatial client: {e}"))?;
            let formats = client.GetSupportedAudioObjectFormatEnumerator().map_err(|e| format!("formats: {e}"))?;
            let fmt = formats.GetFormat(0).map_err(|e| format!("format: {e}"))?;
            if fmt.is_null() || (*fmt).nSamplesPerSec != super::RATE || (*fmt).nChannels != 1 || (*fmt).wBitsPerSample != 32 {
                return Err("its object format is not 48 kHz mono float".into());
            }
            let types = [
                AudioObjectType_FrontLeft,
                AudioObjectType_FrontRight,
                AudioObjectType_FrontCenter,
                AudioObjectType_LowFrequency,
                AudioObjectType_BackLeft,
                AudioObjectType_BackRight,
                AudioObjectType_SideLeft,
                AudioObjectType_SideRight,
                AudioObjectType_TopFrontLeft,
                AudioObjectType_TopFrontRight,
                AudioObjectType_TopBackLeft,
                AudioObjectType_TopBackRight,
            ];
            let mask = AudioObjectType(types.iter().fold(0, |m, t| m | t.0));
            let event = CreateEventW(None, false, false, None).map_err(|e| format!("event: {e}"))?;
            let params = SpatialAudioObjectRenderStreamActivationParams {
                ObjectFormat: fmt,
                StaticObjectTypeMask: mask,
                MinDynamicObjectCount: 0,
                MaxDynamicObjectCount: 0,
                Category: AudioCategory_GameEffects,
                EventHandle: event,
                NotifyObject: std::mem::ManuallyDrop::new(None),
            };
            let mut pv = PROPVARIANT::default();
            (*pv.Anonymous.Anonymous).vt = VT_BLOB;
            (*pv.Anonymous.Anonymous).Anonymous.blob.cbSize = std::mem::size_of::<SpatialAudioObjectRenderStreamActivationParams>() as u32;
            (*pv.Anonymous.Anonymous).Anonymous.blob.pBlobData = &params as *const _ as *mut u8;
            // (A stream that failed to start is kept as well: nothing of it is released.)
            let stream: ISpatialAudioObjectRenderStream = client.ActivateSpatialAudioStream(&pv).map_err(|e| format!("stream: {e}"))?;
            let mut objects = Vec::new();
            for t in types {
                match stream.ActivateSpatialAudioObject(t) {
                    Ok(o) => objects.push(o),
                    Err(e) => {
                        std::mem::forget(stream);
                        return Err(format!("object {t:?}: {e}"));
                    }
                }
            }
            if let Err(e) = stream.Start() {
                std::mem::forget((stream, objects));
                return Err(format!("start: {e}"));
            }
            super::SPATIAL.store(1, Relaxed);
            crate::info!("audio: Windows spatial sound: a 7.1.4 static-object stream, 48 kHz");
            Ok(Kept { generation, _client: client, _formats: formats, stream, objects, event })
        }
    }

    /// Feeds a started spatial stream until the wanted output or device changes (Ok(true)).
    unsafe fn spatial_run(k: &Kept) -> Result<bool, String> {
        use std::sync::atomic::Ordering::Relaxed;
        unsafe {
            let mut scratch: Vec<f32> = Vec::new();
            loop {
                let w = WaitForSingleObject(k.event, 200);
                if w != WAIT_OBJECT_0 && w.0 != 0x102 {
                    return Err("spatial event wait failed".into());
                }
                if super::OUT_CHANNELS.load(Relaxed) != 12 || super::DEVICE_GEN.load(Relaxed) != k.generation {
                    return Ok(true);
                }
                // Updates only when the stream asked for one (its event).
                if w != WAIT_OBJECT_0 {
                    continue;
                }
                let (mut avail, mut frames) = (0u32, 0u32);
                if let Err(e) = k.stream.BeginUpdatingAudioObjects(&mut avail, &mut frames) {
                    crate::warn!("audio: spatial stream lost ({e})");
                    return Ok(true);
                }
                let n = frames as usize;
                scratch.resize(12 * n, 0.0);
                super::mix_into(&mut scratch, n, 12);
                for (ch, o) in k.objects.iter().enumerate() {
                    let (mut p, mut len) = (std::ptr::null_mut::<u8>(), 0u32);
                    if o.GetBuffer(&mut p, &mut len).is_ok() && !p.is_null() {
                        let dst = std::slice::from_raw_parts_mut(p as *mut f32, (len as usize / 4).min(n));
                        for (f, d) in dst.iter_mut().enumerate() {
                            *d = scratch[12 * f + ch];
                        }
                    }
                }
                let _ = k.stream.EndUpdatingAudioObjects();
            }
        }
    }

    /// A device's form factor (PKEY_AudioEndpoint_FormFactor).
    unsafe fn form_factor(d: &IMMDevice) -> Option<u32> {
        use windows::Win32::Foundation::PROPERTYKEY;
        use windows::Win32::System::Com::STGM_READ;
        use windows::Win32::System::Com::StructuredStorage::PropVariantClear;
        use windows::Win32::System::Variant::VT_UI4;
        const FORM_FACTOR: PROPERTYKEY = PROPERTYKEY { fmtid: windows::core::GUID::from_u128(0x1da5d803_d492_4edd_8c23_e0c0ffee7f0e), pid: 0 };
        unsafe {
            let store = d.OpenPropertyStore(STGM_READ).ok()?;
            let mut pv = store.GetValue(&FORM_FACTOR).ok()?;
            let v = (pv.Anonymous.Anonymous.vt == VT_UI4).then(|| pv.Anonymous.Anonymous.Anonymous.ulVal);
            let _ = PropVariantClear(&mut pv);
            v
        }
    }

    /// The active render endpoints: (id, friendly name).
    pub unsafe fn devices() -> Result<Vec<(String, String)>, String> {
        use windows::Win32::Foundation::PROPERTYKEY;
        use windows::Win32::System::Com::STGM_READ;
        use windows::Win32::System::Com::StructuredStorage::PropVariantClear;
        use windows::Win32::System::Variant::VT_LPWSTR;
        // PKEY_Device_FriendlyName.
        const FRIENDLY_NAME: PROPERTYKEY = PROPERTYKEY { fmtid: windows::core::GUID::from_u128(0xa45c254e_df1c_4efd_8020_67d146a850e0), pid: 14 };
        unsafe {
            let _ = CoInitializeEx(None, COINIT_MULTITHREADED);
            let en: IMMDeviceEnumerator = CoCreateInstance(&MMDeviceEnumerator, None, CLSCTX_ALL).map_err(|e| format!("device enumerator: {e}"))?;
            let col = en.EnumAudioEndpoints(eRender, DEVICE_STATE_ACTIVE).map_err(|e| format!("endpoints: {e}"))?;
            let n = col.GetCount().map_err(|e| format!("count: {e}"))?;
            let mut out = Vec::new();
            for i in 0..n {
                let Ok(d) = col.Item(i) else { continue };
                let Ok(id) = d.GetId() else { continue };
                let ids = id.to_string().unwrap_or_default();
                windows::Win32::System::Com::CoTaskMemFree(Some(id.0 as *const _));
                let mut name = String::new();
                if let Ok(store) = d.OpenPropertyStore(STGM_READ) {
                    if let Ok(mut pv) = store.GetValue(&FRIENDLY_NAME) {
                        if pv.Anonymous.Anonymous.vt == VT_LPWSTR {
                            name = pv.Anonymous.Anonymous.Anonymous.pwszVal.to_string().unwrap_or_default();
                        }
                        let _ = PropVariantClear(&mut pv);
                    }
                }
                if !ids.is_empty() {
                    out.push((ids, if name.is_empty() { format!("Device {}", i + 1) } else { name }));
                }
            }
            Ok(out)
        }
    }

    /// One client of `och` channels: Ok(true) when the wanted channels or device changed (open
    /// again).
    unsafe fn session(och: usize) -> Result<bool, String> {
        unsafe {
            crate::debug!("audio: session({och}): enumerator");
            let en: IMMDeviceEnumerator = CoCreateInstance(&MMDeviceEnumerator, None, CLSCTX_ALL).map_err(|e| format!("device enumerator: {e}"))?;
            let generation = super::DEVICE_GEN.load(std::sync::atomic::Ordering::Relaxed);
            let dev = open_device(&en)?;
            crate::info!("audio: session({och}): device open");
            let client: IAudioClient = dev.Activate(CLSCTX_ALL, None).map_err(|e| format!("activate: {e}"))?;
            crate::info!("audio: session({och}): client activated");
            super::DEVICE_FORM.store(form_factor(&dev).unwrap_or(u32::MAX), std::sync::atomic::Ordering::Relaxed);
            if let Ok(mix) = client.GetMixFormat() {
                if !mix.is_null() {
                    super::DEVICE_CHANNELS.store((*mix).nChannels as u32, std::sync::atomic::Ordering::Relaxed);
                    windows::Win32::System::Com::CoTaskMemFree(Some(mix as *const _));
                }
            }
            // The wanted channels (a surround bed on a stereo device: Windows downmixes it, or with
            // its spatial sound on (Windows Sonic, Dolby Atmos) renders it for the headphones);
            // 7.1.4 (diagnosis only) as stereo.
            let wanted = och;
            let dev_ch = super::DEVICE_CHANNELS.load(std::sync::atomic::Ordering::Relaxed) as usize;
            let och = if och > 8 { 2 } else { och };
            crate::info!("audio: opening a WASAPI client of {och} channels ({wanted} wanted, the device mixes {dev_ch})");
            let mut fmt: WAVEFORMATEXTENSIBLE = std::mem::zeroed();
            fmt.Format.wFormatTag = WAVE_FORMAT_EXTENSIBLE as u16;
            fmt.Format.nChannels = och as u16;
            fmt.Format.nSamplesPerSec = super::RATE;
            fmt.Format.wBitsPerSample = 32;
            fmt.Format.nBlockAlign = 4 * och as u16;
            fmt.Format.nAvgBytesPerSec = super::RATE * 4 * och as u32;
            fmt.Format.cbSize = 22;
            fmt.Samples.wValidBitsPerSample = 32;
            fmt.dwChannelMask = match och {
                6 => SPEAKER_FRONT_LEFT | SPEAKER_FRONT_RIGHT | SPEAKER_FRONT_CENTER | SPEAKER_LOW_FREQUENCY | SPEAKER_BACK_LEFT | SPEAKER_BACK_RIGHT,
                8 => SPEAKER_FRONT_LEFT | SPEAKER_FRONT_RIGHT | SPEAKER_FRONT_CENTER | SPEAKER_LOW_FREQUENCY | SPEAKER_BACK_LEFT | SPEAKER_BACK_RIGHT | SPEAKER_SIDE_LEFT | SPEAKER_SIDE_RIGHT,
                _ => SPEAKER_FRONT_LEFT | SPEAKER_FRONT_RIGHT,
            };
            fmt.SubFormat = KSDATAFORMAT_SUBTYPE_IEEE_FLOAT;
            let flags = AUDCLNT_STREAMFLAGS_EVENTCALLBACK | AUDCLNT_STREAMFLAGS_AUTOCONVERTPCM | AUDCLNT_STREAMFLAGS_SRC_DEFAULT_QUALITY;
            // 30 ms device buffer (100 ns units).
            client
                .Initialize(AUDCLNT_SHAREMODE_SHARED, flags, 300_000, 0, &fmt as *const _ as *const WAVEFORMATEX, None)
                .map_err(|e| format!("initialize: {e}"))?;
            let event = CreateEventW(None, false, false, None).map_err(|e| format!("event: {e}"))?;
            client.SetEventHandle(event).map_err(|e| format!("event handle: {e}"))?;
            let render: IAudioRenderClient = client.GetService().map_err(|e| format!("render client: {e}"))?;
            let total = client.GetBufferSize().map_err(|e| format!("buffer size: {e}"))?;
            client.Start().map_err(|e| format!("start: {e}"))?;
            crate::info!(
                "audio: WASAPI shared-mode output, 48 kHz float, {och} channels (the device mixes {}, form factor {}), device buffer {total} frames",
                super::DEVICE_CHANNELS.load(std::sync::atomic::Ordering::Relaxed),
                match super::DEVICE_FORM.load(std::sync::atomic::Ordering::Relaxed) {
                    1 => "speakers",
                    3 => "headphones",
                    5 => "headset",
                    9 => "a display (HDMI)",
                    8 => "S/PDIF",
                    u32::MAX => "unknown",
                    _ => "other",
                }
            );
            let mut scratch = vec![0f32; och * total as usize];
            loop {
                let w = WaitForSingleObject(event, 200);
                if w != WAIT_OBJECT_0 && w.0 != 0x102 {
                    break;
                }
                if super::OUT_CHANNELS.load(std::sync::atomic::Ordering::Relaxed) as usize != wanted || super::DEVICE_GEN.load(std::sync::atomic::Ordering::Relaxed) != generation {
                    let _ = client.Stop();
                    let _ = CloseHandle(event);
                    return Ok(true);
                }
                let padding = match client.GetCurrentPadding() {
                    Ok(p) => p,
                    Err(e) => {
                        crate::warn!("audio: device lost ({e}); switching to the null device");
                        let _ = CloseHandle(event);
                        return Err(e.to_string());
                    }
                };
                let avail = (total - padding) as usize;
                if avail == 0 {
                    continue;
                }
                super::mix_into(&mut scratch, avail, och);
                if let Ok(p) = render.GetBuffer(avail as u32) {
                    std::ptr::copy_nonoverlapping(scratch.as_ptr(), p as *mut f32, och * avail);
                    let _ = render.ReleaseBuffer(avail as u32, 0);
                }
            }
            let _ = CloseHandle(event);
            Err("event wait failed".into())
        }
    }
}
