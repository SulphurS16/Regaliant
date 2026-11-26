//! libSceAudioOut2: PS5 audio output contexts, users and ports.
//!
//! Model (interface facts, see docs/PROVENANCE.md): a context mixes "grains"
//! of `num_grains` samples (default 512 at 48 kHz). The title points each
//! port at one grain of PCM with `PortSetAttributes` (attribute 0 = PCM
//! pointer, 1 = per-channel gains), latches the grain with `ContextAdvance`
//! and submits it with `ContextPush`, which blocks while `queue_depth` grains
//! are queued: that blocking is the title's audio clock.
//!
//! Each pushed grain is mixed to stereo f32 here and handed to the host sink
//! (`crate::audio`).
//!
//! Port type 6 carries the DualSense's haptic waveform (two channels, 90% of its energy below
//! 200 Hz, bursts with hits and rolls: FINDINGS.md 2026-11-06 night): it is not sound, and played
//! through the speakers it was the loudest part of the mix (a 50 Hz rumble at RMS 0.17 against
//! the scene's 0.07). It drives an Xbox controller's motors instead (`input::haptics`).
//!
//! Object (3D) ports: the title renders its 3D sound to 5th-order ambisonics
//! and submits each of the 36 channels as an object port whose attribute 0x8
//! is 0x40 + the channel's ACN index (two such sets, attribute 0x9 = 0 or 1;
//! no position attribute is ever set). Measured, not documented: per-channel
//! energies follow ACN order with SN3D normalization (W strongest, the
//! channels that vanish for sources on the horizon near silent,
//! X^2 + Y^2 + Z^2 ~ 0.8 W^2; FINDINGS.md 2026-10-16). The field (orders 0-3)
//! goes to `crate::spatial`, the system renderer's stand-in: binaural for
//! headphones, panned for speakers (the game's Output Device Type), replacing
//! the first decode (two cardioids at +-90 degrees, which put a sound to one
//! side in one ear only). `DS_AUDIO_AMBI=2` mirrors left and right, `0` sums
//! every object port to the center. Other object ports (no channel index)
//! are still mixed to the center (logged once).

use crate::hle::Registry;
use crate::{reg, sysv};
use parking_lot::Mutex;
use std::collections::HashMap;
use std::sync::atomic::{AtomicU32, Ordering};

const ERR_INVALID_PARAM: i32 = 0x8026_8001u32 as i32;
const ERR_NOT_READY: i32 = 0x8026_8008u32 as i32;
const ERR_INVALID_PORT: i32 = 0x8026_8009u32 as i32;

const ATTR_PCM: u32 = 0;
const ATTR_GAIN: u32 = 1;

struct Context {
    num_grains: u32,
    queue_depth: u32,
    /// Its host stream (`crate::audio`): contexts are mixed together, each paced by its own
    /// queue. (Until 2026-11-13 every context pushed into one stream, so two contexts pushing at
    /// once had their grains played one after the other.)
    stream: usize,
    /// Grains pushed (status).
    pushes: u64,
    /// When the last push arrived, and per push the queue level it found (indices 0..=8, 8 for
    /// eight or more) and the gaps between pushes (status: [<2 ms, <6, <12, <25, <50, <100,
    /// longer]).
    last_push: Option<std::time::Instant>,
    levels: [u64; 9],
    gaps: [u64; 7],
}

/// Each context's 3D-field renderer (its filters' state is the context's own).
static RENDERERS: Mutex<Option<HashMap<u32, crate::spatial::Renderer>>> = Mutex::new(None);

struct Port {
    ctx: u32,
    port_type: u16,
    format: u32,
    pcm: u64,
    latched: u64,
    gains: [f32; 16],
    /// Object ports: the title's attribute 0x8 (0x40 + an index, one per
    /// port) and 0x9 (0 or 1), meaning not known yet (diagnostics).
    attr8: u32,
    attr9: u8,
}

static NEXT: AtomicU32 = AtomicU32::new(1);
static CONTEXTS: Mutex<Option<HashMap<u32, Context>>> = Mutex::new(None);
static PORTS: Mutex<Option<HashMap<u32, Port>>> = Mutex::new(None);

fn channels(format: u32) -> usize {
    match (format >> 8) & 0xFF {
        0 => 2,
        n => (n as usize).min(16),
    }
}

fn is_object(port_type: u16) -> bool {
    port_type & 0xFF00 == 0x0100
}

/// The DualSense haptics port type (not mixed: `haptics`).
const PORT_HAPTICS: u16 = 6;

/// The DualSense speaker's port type: AudioOut2's port types 0-6 are main, BGM, voice, pad
/// speaker, personal, aux and vibration (an interface fact, PROVENANCE.md); measured, a mono
/// layer of its own at 0.5-16 kHz with hits, attacks and rolls (FINDINGS.md 2026-11-06 night,
/// 2026-11-13). On the console it plays from the controller, never in the TV's or a headset's
/// sound, so it is left out of the output unless `PAD_SPEAKER` (the PC setting Controller
/// Speaker Sounds; it was mixed in, centred, until 2026-11-13).
const PORT_PAD_SPEAKER: u16 = 3;

/// Whether the pad speaker's sound is mixed into the output (`PORT_PAD_SPEAKER`).
pub static PAD_SPEAKER: std::sync::atomic::AtomicBool = std::sync::atomic::AtomicBool::new(false);

/// One grain of the haptics port: its RMS per side to the controller's motors.
unsafe fn haptics(p: &Port, n: usize) {
    if p.latched == 0 {
        return;
    }
    let ch = channels(p.format);
    let float = p.format & 0x7F == 0;
    let sample = |i: usize| -> f32 { unsafe { if float { *((p.latched as *const f32).add(i)) } else { *((p.latched as *const i16).add(i)) as f32 / 32768.0 } } };
    let (mut l, mut r) = (0f64, 0f64);
    for f in 0..n {
        let a = sample(f * ch) * p.gains[0];
        let b = if ch > 1 { sample(f * ch + 1) * p.gains[1] } else { a };
        l += (a * a) as f64;
        r += (b * b) as f64;
    }
    let k = 1.0 / n.max(1) as f64;
    crate::platform::loader_hooks::haptics((l * k).sqrt() as f32, (r * k).sqrt() as f32);
}

/// The ambisonic channel (ACN index 0..35) an object port carries (its
/// attribute 0x8 is 0x40 + the index).
fn ambi_channel(p: &Port) -> Option<u32> {
    (is_object(p.port_type) && (0x40..0x40 + 36).contains(&p.attr8)).then(|| p.attr8 - 0x40)
}

/// How ambisonic object ports are decoded (`DS_AUDIO_AMBI`): 1 (default)
/// AmbiX axes, +Y left; 2 mirrored, +Y right (if left and right sound
/// swapped); 0 not decoded (every object port to the center, as before).
fn ambi_mode() -> u8 {
    static MODE: std::sync::OnceLock<u8> = std::sync::OnceLock::new();
    *MODE.get_or_init(|| match std::env::var("DS_AUDIO_AMBI").as_deref() {
        Ok("0") => 0,
        Ok("2") => 2,
        _ => 1,
    })
}

fn ambi_decode() -> bool {
    ambi_mode() != 0
}

/// The PC tab's Audio Output: 0 automatic, 1 headphones, 2 stereo speakers, 3 5.1, 4 7.1, 5
/// Windows' spatial sound (Windows Sonic, Dolby Atmos; headphones where it is off). (The
/// game's own Output Device Type is the console's system output, shown, not chosen: FINDINGS.md
/// 2026-11-06 night.)
pub static OUTPUT_CHOICE: AtomicU32 = AtomicU32::new(0);

/// The layout the field is rendered for: the PC tab's Audio Output; automatic: Windows' spatial
/// sound when it is on for the device (its own head model), else headphones when Windows names
/// the device headphones or a headset, else the device's surround (7.1 or 5.1 by its mix
/// format's channels), else stereo speakers.
pub fn render_layout() -> crate::spatial::Layout {
    use crate::spatial::Layout;
    let dev = crate::audio::DEVICE_CHANNELS.load(Ordering::Relaxed);
    let spatial = crate::audio::spatial_allowed() && crate::audio::SPATIAL.load(Ordering::Relaxed) != 0;
    match OUTPUT_CHOICE.load(Ordering::Relaxed) {
        1 => Layout::Headphones,
        2 => Layout::Stereo,
        3 => Layout::Surround51,
        4 => Layout::Surround71,
        5 if spatial => Layout::Spatial714,
        // Windows Sonic / Atmos: a 7.1 bed the device's spatial sound renders for headphones.
        5 => Layout::Surround71,
        _ if spatial => Layout::Spatial714,
        _ => match crate::audio::DEVICE_FORM.load(Ordering::Relaxed) {
            3 | 5 => Layout::Headphones,
            _ if dev >= 8 => Layout::Surround71,
            _ if dev >= 6 => Layout::Surround51,
            _ => Layout::Stereo,
        },
    }
}

/// `DS_AUDIO_ATTRS` diagnostics: the ambisonic sound field's left-right
/// balance per second, sum(W * Y) / sum(W * W) (AmbiX: +1 all left, -1 all
/// right), and front-back, sum(W * X) / sum(W * W), to check the decode's
/// orientation against camera turns.
fn note_balance(w: &[f32], y: &[f32], x: &[f32]) {
    static ACC: Mutex<(f64, f64, u32, f64)> = Mutex::new((0.0, 0.0, 0, 0.0));
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    if !*ON.get_or_init(|| std::env::var("DS_AUDIO_ATTRS").is_ok_and(|v| v != "0")) {
        return;
    }
    let mut a = ACC.lock();
    for ((wv, yv), xv) in w.iter().zip(y).zip(x) {
        a.0 += (*wv * *yv) as f64;
        a.1 += (*wv * *wv) as f64;
        a.3 += (*wv * *xv) as f64;
    }
    a.2 += 1;
    if a.2 >= 187 {
        crate::info!("AudioOut2 ambisonic balance {:+.3} front {:+.3} (W energy {:.2e}; flip {})", a.0 / a.1.max(1e-12), a.3 / a.1.max(1e-12), a.1, crate::platform::loader_hooks::frame_count());
        *a = (0.0, 0.0, 0, 0.0);
    }
}

/// Logs each attribute id the title uses once (diagnostics).
fn note_attr(id: u32, value: u64, size: u64) {
    static SEEN: Mutex<Option<std::collections::HashSet<u32>>> = Mutex::new(None);
    if SEEN.lock().get_or_insert_with(Default::default).insert(id) {
        let first = if value != 0 && size >= 8 { format!(" first qword {:#x}", unsafe { *(value as *const u64) }) } else { String::new() };
        crate::info!("AudioOut2 port attribute id {id:#x} (value {value:#x}, size {size}){first}");
    }
}

/// `DS_AUDIO_ATTRS=<n>` (diagnostics): the first n attribute settings of
/// each (port type, attribute id) pair are logged with their bytes (as
/// words and floats), then every 500th, to tell what the title sends with
/// object (3D) ports.
fn trace_attr(port: u32, port_type: u16, id: u32, value: u64, size: u64) {
    static LIMIT: std::sync::OnceLock<u64> = std::sync::OnceLock::new();
    let limit = *LIMIT.get_or_init(|| std::env::var("DS_AUDIO_ATTRS").ok().and_then(|v| v.parse().ok()).unwrap_or(0));
    if limit == 0 {
        return;
    }
    static COUNTS: Mutex<Option<HashMap<(u16, u32), u64>>> = Mutex::new(None);
    let n = {
        let mut g = COUNTS.lock();
        let c = g.get_or_insert_with(HashMap::new).entry((port_type, id)).or_insert(0);
        *c += 1;
        *c
    };
    if n > limit && n % 500 != 0 {
        return;
    }
    let bytes: Vec<u8> = if value != 0 && size > 0 { (0..size.min(64) as usize).map(|i| unsafe { *(value as *const u8).add(i) }).collect() } else { Vec::new() };
    let words: Vec<String> = bytes.chunks(4).map(|c| {
        let mut w = [0u8; 4];
        w[..c.len()].copy_from_slice(c);
        let u = u32::from_le_bytes(w);
        format!("{u:#x}/{:.4}", f32::from_bits(u))
    }).collect();
    crate::info!("AudioOut2 attr #{n}: port {port} type {port_type:#x} id {id:#x} size {size}: {}", words.join(" "));
}

/// One-time per-port snapshot (diagnostics).
fn dump_ports(ctx: u32, n: usize) {
    let ports = PORTS.lock();
    let mut s = format!("AudioOut2 context {ctx} ports after 2000 pushes:");
    for (h, p) in ports.as_ref().into_iter().flat_map(|m| m.iter()) {
        let ch = channels(p.format);
        let peak = if p.latched != 0 {
            let float = p.format & 0x7F == 0;
            (0..n * ch).fold(0f32, |m, i| {
                let v = if float { unsafe { *(p.latched as *const f32).add(i) } } else { (unsafe { *(p.latched as *const i16).add(i) }) as f32 / 32768.0 };
                m.max(v.abs())
            })
        } else {
            0.0
        };
        s.push_str(&format!("
    port {h} ctx {} type {:#x} format {:#x} pcm {:#x} latched {:#x} peak {peak:.3} gains {:?}", p.ctx, p.port_type, p.format, p.pcm, p.latched, &p.gains[..ch.min(4)]));
    }
    crate::info!("{s}");
}

/// Signal mixed per port type (diagnostics, `status`): port type ->
/// (sum of squared samples after gains, samples, grains with signal).
static ENERGY: Mutex<Option<HashMap<u16, (f64, u64, u64)>>> = Mutex::new(None);
/// The same for object ports by (attribute 0x8, attribute 0x9).
static ENERGY_OBJ: Mutex<Option<HashMap<(u32, u8), (f64, u64, u64)>>> = Mutex::new(None);

/// Mixed signal by port type: RMS over the session and grains with signal
/// (object ports 0x1xx counted together).
pub fn status() -> String {
    let g = ENERGY.lock();
    let Some(m) = g.as_ref() else { return String::new() };
    let mut v: Vec<_> = m.iter().collect();
    v.sort_by_key(|e| *e.0);
    let parts: Vec<String> = v.iter().map(|(t, (e, n, gr))| format!("type {:#x}: rms {:.4} ({} grains with signal)", t, (e / (*n).max(1) as f64).sqrt(), gr)).collect();
    let mut out = format!("    AudioOut2 mixed signal by port type: {}\n", parts.join(", "));
    if let Some(cs) = CONTEXTS.try_lock() {
        for (h, c) in cs.iter().flat_map(|m| m.iter()).filter(|(_, c)| c.pushes > 0) {
            out.push_str(&format!(
                "    AudioOut2 context {h}: {} pushes ({} frames each, queue depth {}); the queue each found (grains 0..8+): {:?}; gaps between pushes (<2, <6, <12, <25, <50, <100 ms, longer): {:?}\n",
                c.pushes, c.num_grains, c.queue_depth, c.levels, c.gaps
            ));
        }
    }
    if let Some(o) = ENERGY_OBJ.lock().as_ref() {
        let mut v: Vec<_> = o.iter().collect();
        v.sort_by_key(|e| *e.0);
        let parts: Vec<String> = v.iter().map(|((a8, a9), (e, n, gr))| format!("{a8:#x}/{a9}: {:.4} ({gr})", (e / (*n).max(1) as f64).sqrt())).collect();
        out.push_str(&format!("    AudioOut2 object ports by attribute 0x8/0x9: rms (grains with signal) {}\n", parts.join(", ")));
    }
    out
}

/// `audio split <seconds>` (diagnosis): each port group's own stereo mix as `mix_port` adds it
/// (groups: port type, channel count and format; the ambisonic objects' decode), written to
/// `<run dir>/audio-split-<group>.wav` together, aligned to the grain.
struct Split {
    dir: std::path::PathBuf,
    /// The capture's number in the run (file names `audio-split<n>-<group>.wav`).
    n: u32,
    frames_left: usize,
    frames: usize,
    mixes: std::collections::BTreeMap<String, Vec<f32>>,
    /// The title's ambisonic field as it arrived (ACN 0-35 interleaved, after the ports' gains).
    field: Vec<f32>,
    /// 8-channel bed ports as they arrived (8 channels interleaved, after the ports' gains), by
    /// group.
    beds: std::collections::BTreeMap<String, Vec<f32>>,
}

/// Channels of the raw field the split capture keeps (the title's 5th order).
const FIELD_RAW: usize = 36;

/// One grain of each 8-channel bed port (raw channels after its gains) for the split capture.
unsafe fn beds_raw(ports: &HashMap<u32, Port>, ctx: u32, n: usize) -> Vec<(String, Vec<f32>)> {
    let mut out: Vec<(String, Vec<f32>)> = Vec::new();
    for p in ports.values().filter(|p| p.ctx == ctx && p.latched != 0 && !is_object(p.port_type) && channels(p.format) >= 8) {
        let ch = channels(p.format);
        let float = p.format & 0x7F == 0;
        let k = split_group(p);
        let i = match out.iter().position(|(g, _)| *g == k) {
            Some(i) => i,
            None => {
                out.push((k, vec![0f32; 8 * n]));
                out.len() - 1
            }
        };
        let v = &mut out[i].1;
        for f in 0..n {
            for c in 0..8 {
                let x = unsafe { if float { *((p.latched as *const f32).add(f * ch + c)) } else { *((p.latched as *const i16).add(f * ch + c)) as f32 / 32768.0 } };
                v[8 * f + c] += x * p.gains[c];
            }
        }
    }
    out
}

/// One grain of the raw field (36 channels) for the split capture.
unsafe fn field_raw(ports: &HashMap<u32, Port>, ctx: u32, n: usize) -> Vec<f32> {
    let mut out = vec![0f32; FIELD_RAW * n];
    for p in ports.values().filter(|p| p.ctx == ctx && p.latched != 0) {
        let Some(acn) = ambi_channel(p) else { continue };
        if channels(p.format) != 1 || acn as usize >= FIELD_RAW {
            continue;
        }
        let float = p.format & 0x7F == 0;
        let g = p.gains[0];
        for i in 0..n {
            let x = unsafe { if float { *((p.latched as *const f32).add(i)) } else { *((p.latched as *const i16).add(i)) as f32 / 32768.0 } };
            out[FIELD_RAW * i + acn as usize] += x * g;
        }
    }
    out
}

static SPLIT: Mutex<Option<Split>> = Mutex::new(None);

pub fn start_split(seconds: f64) -> Result<(), String> {
    let dir = crate::run_dir().ok_or("no run directory")?;
    let mut g = SPLIT.lock();
    if g.is_some() {
        return Err("a split capture is in progress".into());
    }
    static N: AtomicU32 = AtomicU32::new(0);
    // (Reserved whole: growing them in the title's audio job stalled it 40-117 ms at each
    // doubling, and the sound ran dry.)
    let frames = (seconds * 48000.0) as usize;
    *g = Some(Split { dir: dir.to_path_buf(), n: N.fetch_add(1, Ordering::Relaxed), frames_left: frames, frames: 0, mixes: Default::default(), field: Vec::with_capacity(FIELD_RAW * (frames + 4096)), beds: Default::default() });
    crate::info!("AudioOut2: capturing {seconds} s of each port group's mix");
    Ok(())
}

fn split_group(p: &Port) -> String {
    match ambi_channel(p) {
        Some(_) if ambi_decode() => format!("ambi{}", p.attr9),
        _ if is_object(p.port_type) => "object-other".into(),
        _ => format!("type{:x}-{}ch-f{:x}", p.port_type, channels(p.format), p.format),
    }
}

/// One grain's groups (`group` -> its stereo mix of `n` frames) and its raw field into the
/// split capture.
fn split_add(n: usize, grain: std::collections::BTreeMap<String, Vec<f32>>, field: Vec<f32>, beds: Vec<(String, Vec<f32>)>) {
    let mut g = SPLIT.lock();
    let Some(sp) = g.as_mut() else { return };
    sp.field.extend_from_slice(&field);
    let reserve = 8 * (sp.frames_left + sp.frames + 4096);
    for (k, v) in beds {
        let frames = sp.frames;
        let e = sp.beds.entry(k).or_insert_with(|| {
            let mut b = Vec::with_capacity(reserve);
            b.resize(8 * frames, 0.0);
            b
        });
        e.extend_from_slice(&v);
    }
    let frames_now = sp.frames + n;
    for b in sp.beds.values_mut() {
        b.resize(8 * frames_now, 0.0);
    }
    for (k, v) in grain {
        let reserve = 2 * (sp.frames_left + sp.frames + 4096);
        let e = sp.mixes.entry(k).or_insert_with(|| {
            let mut b = Vec::with_capacity(reserve);
            b.resize(2 * sp.frames, 0.0);
            b
        });
        e.extend_from_slice(&v);
    }
    sp.frames += n;
    for v in sp.mixes.values_mut() {
        v.resize(2 * sp.frames, 0.0);
    }
    sp.frames_left = sp.frames_left.saturating_sub(n);
    if sp.frames_left > 0 {
        return;
    }
    let sp = g.take().unwrap();
    drop(g);
    // Written on a thread of its own (the title's audio job must not wait for the files).
    let _ = std::thread::Builder::new().name("audio-split".into()).spawn(move || {
        // The raw beds: 8-channel float WAVs.
        for (k, v) in &sp.beds {
            let path = sp.dir.join(format!("audio-split{}-bed-{k}.wav", sp.n));
            let data = 4 * v.len() as u32;
            let mut b = Vec::with_capacity(46 + data as usize);
            b.extend_from_slice(b"RIFF");
            b.extend_from_slice(&(38 + data).to_le_bytes());
            b.extend_from_slice(b"WAVEfmt ");
            b.extend_from_slice(&18u32.to_le_bytes());
            b.extend_from_slice(&3u16.to_le_bytes());
            b.extend_from_slice(&8u16.to_le_bytes());
            b.extend_from_slice(&48000u32.to_le_bytes());
            b.extend_from_slice(&(48000 * 4 * 8u32).to_le_bytes());
            b.extend_from_slice(&32u16.to_le_bytes());
            b.extend_from_slice(&32u16.to_le_bytes());
            b.extend_from_slice(&0u16.to_le_bytes());
            b.extend_from_slice(b"data");
            b.extend_from_slice(&data.to_le_bytes());
            for s in v {
                b.extend_from_slice(&s.to_le_bytes());
            }
            match std::fs::write(&path, &b) {
                Ok(()) => crate::info!("AudioOut2: split capture {} (the raw bed, 8 channels)", path.display()),
                Err(e) => crate::warn!("AudioOut2: split capture {}: {e}", path.display()),
            }
        }
        // The raw field: a 36-channel float WAV.
        {
            let path = sp.dir.join(format!("audio-split{}-field36.wav", sp.n));
            let data = 4 * sp.field.len() as u32;
            let mut b = Vec::with_capacity(46 + data as usize);
            b.extend_from_slice(b"RIFF");
            b.extend_from_slice(&(38 + data).to_le_bytes());
            b.extend_from_slice(b"WAVEfmt ");
            b.extend_from_slice(&18u32.to_le_bytes());
            b.extend_from_slice(&3u16.to_le_bytes());
            b.extend_from_slice(&(FIELD_RAW as u16).to_le_bytes());
            b.extend_from_slice(&48000u32.to_le_bytes());
            b.extend_from_slice(&(48000 * 4 * FIELD_RAW as u32).to_le_bytes());
            b.extend_from_slice(&(4 * FIELD_RAW as u16).to_le_bytes());
            b.extend_from_slice(&32u16.to_le_bytes());
            b.extend_from_slice(&0u16.to_le_bytes());
            b.extend_from_slice(b"data");
            b.extend_from_slice(&data.to_le_bytes());
            for s in &sp.field {
                b.extend_from_slice(&s.to_le_bytes());
            }
            match std::fs::write(&path, &b) {
                Ok(()) => crate::info!("AudioOut2: split capture {} (the raw field, {} channels)", path.display(), FIELD_RAW),
                Err(e) => crate::warn!("AudioOut2: split capture {}: {e}", path.display()),
            }
        }
        for (k, v) in &sp.mixes {
            let path = sp.dir.join(format!("audio-split{}-{k}.wav", sp.n));
            let peak = v.iter().fold(0f32, |m, x| m.max(x.abs()));
            let rms = (v.iter().map(|&x| x as f64 * x as f64).sum::<f64>() / v.len().max(1) as f64).sqrt();
            match crate::audio::write_wav_file(&path, v) {
                Ok(()) => crate::info!("AudioOut2: split capture {} (rms {rms:.4}, peak {peak:.3})", path.display()),
                Err(e) => crate::warn!("AudioOut2: split capture {}: {e}", path.display()),
            }
        }
    });
}

/// Adds an ambisonic channel's grain (ACN 0-15, mono ports) to the field being gathered.
unsafe fn add_field(p: &Port, n: usize, field: &mut [Vec<f32>]) {
    let Some(acn) = ambi_channel(p) else { return };
    if p.latched == 0 || channels(p.format) != 1 || acn as usize >= field.len() {
        return;
    }
    let float = p.format & 0x7F == 0;
    let f = &mut field[acn as usize];
    if f.is_empty() {
        f.resize(n, 0.0);
    }
    let g = p.gains[0];
    for (i, v) in f.iter_mut().enumerate().take(n) {
        let x = unsafe { if float { *((p.latched as *const f32).add(i)) } else { *((p.latched as *const i16).add(i)) as f32 / 32768.0 } };
        *v += x * g;
    }
}

/// Headphones: an 8-channel bed port's speakers (the title's L R C LFE Ls Rs Lb Rb) added to
/// the context's field as plane waves from their places in the 7.1 layout the title mixes for
/// (L and R at +-30 degrees, C ahead, the sides at +-90, the backs at +-150; LFE left out, as in
/// the stereo fold-down), so the head model renders them as virtual speakers: each reaches both
/// ears with the head's level, shadow and delay (a console renders its bed so for headphones).
/// The stereo fold-down put the surrounds hard into one ear each. Returns whether it did.
unsafe fn add_bed_field(p: &Port, n: usize, field: &mut [Vec<f32>]) -> bool {
    use crate::spatial::{CHANNELS, Layout, sh};
    if p.latched == 0 || channels(p.format) < 8 || render_layout() != Layout::Headphones || field.len() < CHANNELS {
        return false;
    }
    static DIRS: std::sync::OnceLock<[[f32; CHANNELS]; 8]> = std::sync::OnceLock::new();
    let dirs = DIRS.get_or_init(|| {
        let az = [30.0f32, -30.0, 0.0, 0.0, 90.0, -90.0, 150.0, -150.0];
        std::array::from_fn(|c| {
            let a = az[c].to_radians();
            sh(a.cos(), a.sin(), 0.0)
        })
    });
    let ch = channels(p.format);
    let float = p.format & 0x7F == 0;
    for f in field.iter_mut().take(CHANNELS) {
        if f.len() < n {
            f.resize(n, 0.0);
        }
    }
    for i in 0..n {
        for c in [0usize, 1, 2, 4, 5, 6, 7] {
            let x = unsafe { if float { *((p.latched as *const f32).add(i * ch + c)) } else { *((p.latched as *const i16).add(i * ch + c)) as f32 / 32768.0 } } * p.gains[c];
            if x == 0.0 {
                continue;
            }
            for (k, f) in field.iter_mut().take(CHANNELS).enumerate() {
                f[i] += x * dirs[c][k];
            }
        }
    }
    true
}

/// Renders the gathered field of context `ctx` into its mix.
fn render_field(ctx: u32, field: &[Vec<f32>], n: usize, mix: &mut [f32]) {
    use crate::spatial::{CHANNELS, Renderer};
    if field.iter().all(|f| f.is_empty()) {
        return;
    }
    let layout = render_layout();
    let mut g = RENDERERS.lock();
    let m = g.get_or_insert_with(HashMap::new);
    if m.get(&ctx).is_none_or(|r| r.layout() != layout) {
        crate::info!("AudioOut2: context {ctx}'s 3D sound field rendered for {layout:?}");
        m.insert(ctx, Renderer::new(layout));
    }
    // Mirrored (`DS_AUDIO_AMBI=2`): the left-right channels (odd in y) negated.
    let mirror = ambi_mode() == 2;
    let flipped: Vec<Vec<f32>> = if mirror {
        field.iter().enumerate().map(|(k, f)| if matches!(k, 1 | 4 | 5 | 9 | 10 | 11) { f.iter().map(|v| -v).collect() } else { f.clone() }).collect()
    } else {
        Vec::new()
    };
    let src = if mirror { &flipped } else { field };
    let acn: [Option<&[f32]>; CHANNELS] = std::array::from_fn(|k| src.get(k).filter(|f| f.len() >= n).map(|f| &f[..n]));
    m.get_mut(&ctx).unwrap().render(&acn, n, mix);
}

/// The split capture's copy of the field's rendering: the mix minus everything else would need
/// a second renderer; this renders through a renderer of its own (its own delay state).
fn render_field_copy(field: &[Vec<f32>], n: usize, out: &mut [f32]) {
    use crate::spatial::{CHANNELS, Renderer};
    static COPY: Mutex<Option<Renderer>> = Mutex::new(None);
    // (Split captures are stereo: a surround layout's copy renders for stereo speakers.)
    let layout = match render_layout() {
        l @ (crate::spatial::Layout::Headphones | crate::spatial::Layout::Stereo) => l,
        _ => crate::spatial::Layout::Stereo,
    };
    let mut g = COPY.lock();
    if g.as_ref().is_none_or(|r| r.layout() != layout) {
        *g = Some(Renderer::new(layout));
    }
    let acn: [Option<&[f32]>; CHANNELS] = std::array::from_fn(|k| field.get(k).filter(|f| f.len() >= n).map(|f| &f[..n]));
    g.as_mut().unwrap().render(&acn, n, out);
}

/// Mixes one grain of a port into an accumulator of `och` channels (2, 6 or 8; FL FR FC LFE BL
/// BR SL SR): stereo ports into FL and FR, mono ones to the centre (both sides in stereo),
/// 8-channel beds (the title's L R C LFE Ls Rs Lb Rb) to the layout's speakers.
unsafe fn mix_port(p: &Port, n: usize, out: &mut [f32]) {
    unsafe { mix_port_ch(p, n, out, 2) }
}

unsafe fn mix_port_ch(p: &Port, n: usize, out: &mut [f32], och: usize) {
    if p.latched == 0 {
        return;
    }
    unsafe {
        port_energy(p, n);
        mix_port_body(p, n, out, och);
    }
}

/// Diagnostics: the signal a port adds, after its gains (the status's levels by port type).
unsafe fn port_energy(p: &Port, n: usize) {
    if p.latched == 0 {
        return;
    }
    {
        let ch = channels(p.format);
        let float = p.format & 0x7F == 0;
        let mut e = 0f64;
        for i in 0..n * ch {
            let v = if float { unsafe { *(p.latched as *const f32).add(i) } } else { (unsafe { *(p.latched as *const i16).add(i) }) as f32 / 32768.0 };
            let v = (v * p.gains[i % ch]) as f64;
            e += v * v;
        }
        let key = if is_object(p.port_type) { 0x100 } else { p.port_type };
        if is_object(p.port_type) {
            let mut g = ENERGY_OBJ.lock();
            let x = g.get_or_insert_with(HashMap::new).entry((p.attr8, p.attr9)).or_insert((0.0, 0, 0));
            x.0 += e;
            x.1 += (n * ch) as u64;
            x.2 += (e > 1e-9) as u64;
        }
        let mut g = ENERGY.lock();
        let x = g.get_or_insert_with(HashMap::new).entry(key).or_insert((0.0, 0, 0));
        x.0 += e;
        x.1 += (n * ch) as u64;
        x.2 += (e > 1e-9) as u64;
    }
}

unsafe fn mix_port_body(p: &Port, n: usize, out: &mut [f32], och: usize) {
    let ch = channels(p.format);
    let float = p.format & 0x7F == 0;
    let sample = |i: usize| -> f32 {
        unsafe {
            if float {
                *((p.latched as *const f32).add(i))
            } else {
                *((p.latched as *const i16).add(i)) as f32 / 32768.0
            }
        }
    };
    // An ambisonic channel: rendered with the field (`add_field`, `crate::spatial`).
    if let (Some(_), true, 1) = (ambi_channel(p), ambi_decode(), ch) {
        return;
    }
    for f in 0..n {
        let base = f * ch;
        let o = &mut out[f * och..(f + 1) * och];
        match ch {
            1 => {
                let v = sample(base) * p.gains[0];
                if och == 2 {
                    o[0] += v * 0.7071;
                    o[1] += v * 0.7071;
                } else {
                    o[2] += v;
                }
            }
            c if c >= 8 && och >= 6 => {
                // (7.1.4: the bed as in 7.1, nothing in the heights.)
                // The bed to its speakers: L R C LFE, Ls Rs to the sides (7.1) or surrounds
                // (5.1), Lb Rb to the backs (or the 5.1 surrounds at 0.707).
                let g = |k: usize| sample(base + k) * p.gains[k];
                for k in 0..4 {
                    o[k] += g(k);
                }
                if och == 8 {
                    o[6] += g(4);
                    o[7] += g(5);
                    o[4] += g(6);
                    o[5] += g(7);
                } else {
                    o[4] += g(4) + 0.7071 * g(6);
                    o[5] += g(5) + 0.7071 * g(7);
                }
            }
            _ => {
                // Front left/right; in stereo the other channels (center, surrounds) fold down
                // (LFE left out).
                let mut l = sample(base) * p.gains[0];
                let mut r = sample(base + 1) * p.gains[1];
                if ch >= 8 {
                    let c = sample(base + 2) * p.gains[2] * 0.7071;
                    let (sl, sr) = (sample(base + 4) * p.gains[4], sample(base + 5) * p.gains[5]);
                    let (bl, br) = (sample(base + 6) * p.gains[6], sample(base + 7) * p.gains[7]);
                    l += c + 0.7071 * (sl + bl);
                    r += c + 0.7071 * (sr + br);
                }
                o[0] += l;
                o[1] += r;
            }
        }
    }
}

sysv! {
    fn sceAudioOut2Initialize() -> i32 {
        crate::audio::init();
        0
    }
    /// SceAudioOut2ContextParam: {max_ports, max_object_ports, guarantee_object_ports,
    /// queue_depth, num_grains, flags, reserved[10]}.
    fn sceAudioOut2ContextResetParam(p: *mut u32) -> i32 {
        if p.is_null() { return ERR_INVALID_PARAM; }
        std::ptr::write_bytes(p as *mut u8, 0, 64);
        *p = 256;
        *p.add(1) = 256;
        *p.add(3) = 4;
        *p.add(4) = 512;
        *p.add(5) = 1;
        0
    }
    fn sceAudioOut2ContextQueryMemory(p: *const u32, size: *mut u64) -> i32 {
        if p.is_null() || size.is_null() { return ERR_INVALID_PARAM; }
        let depth = if *p.add(3) == 0 { 4 } else { *p.add(3) };
        *size = 0x10000 + depth as u64 * 0x590;
        0
    }
    fn sceAudioOut2ContextCreate(p: *const u32, mem: u64, mem_size: u64, out: *mut u32) -> i32 {
        if p.is_null() || out.is_null() { return ERR_INVALID_PARAM; }
        let depth = if *p.add(3) == 0 { 4 } else { *p.add(3) };
        let grains = if *p.add(4) == 0 { 512 } else { *p.add(4) };
        let h = NEXT.fetch_add(1, Ordering::Relaxed);
        CONTEXTS.lock().get_or_insert_with(HashMap::new).insert(h, Context { num_grains: grains, queue_depth: depth, stream: crate::audio::new_stream(), pushes: 0, last_push: None, levels: [0; 9], gaps: [0; 7] });
        crate::info!("sceAudioOut2ContextCreate -> {h} ({grains} samples/grain, queue depth {depth}, max ports {}, objects {})", *p, *p.add(1));
        *out = h;
        0
    }
    fn sceAudioOut2ContextDestroy(ctx: u32) -> i32 {
        CONTEXTS.lock().get_or_insert_with(HashMap::new).remove(&ctx);
        if let Some(m) = RENDERERS.lock().as_mut() {
            m.remove(&ctx);
        }
        PORTS.lock().get_or_insert_with(HashMap::new).retain(|_, p| p.ctx != ctx);
        0
    }
    /// (Context attributes are not applied; `DS_AUDIO_ATTRS` traces them as
    /// port type 0xffff.)
    fn sceAudioOut2ContextSetAttributes(ctx: u32, attrs: u64, num: u32) -> i32 {
        if attrs != 0 {
            for i in 0..num as u64 {
                let a = (attrs + i * 24) as *const u8;
                trace_attr(ctx, 0xffff, *(a as *const u32), *(a.add(8) as *const u64), *(a.add(16) as *const u64));
            }
        }
        0
    }
    /// Latches the PCM pointers set for this grain.
    fn sceAudioOut2ContextAdvance(ctx: u32) -> i32 {
        for p in PORTS.lock().get_or_insert_with(HashMap::new).values_mut().filter(|p| p.ctx == ctx) {
            p.latched = p.pcm;
        }
        0
    }
    fn sceAudioOut2ContextPush(ctx: u32, blocking: u32) -> i32 {
        let Some((n, depth, stream)) = CONTEXTS.lock().get_or_insert_with(HashMap::new).get_mut(&ctx).map(|c| {
            c.pushes += 1;
            // (Diagnostics: the queue this push found and the gap since the last one; a push
            // that found it empty came late.)
            let now = std::time::Instant::now();
            let level = crate::audio::queued_grains(c.stream);
            c.levels[level.min(8)] += 1;
            if let Some(t) = c.last_push {
                let gap = now.duration_since(t).as_secs_f64() * 1000.0;
                let k = [2.0, 6.0, 12.0, 25.0, 50.0, 100.0].iter().position(|&b| gap < b).unwrap_or(6);
                c.gaps[k] += 1;
                if level == 0 && c.pushes > 100 {
                    static LATE: AtomicU32 = AtomicU32::new(0);
                    if LATE.fetch_add(1, Ordering::Relaxed) < 40 {
                        crate::info!("AudioOut2: context {ctx} pushed into an empty queue {gap:.1} ms after its last push (thread {}, flip {})", crate::log::thread_name(), crate::platform::loader_hooks::frame_count());
                    }
                }
            }
            c.last_push = Some(now);
            (c.num_grains as usize, c.queue_depth, c.stream)
        }) else {
            return ERR_INVALID_PARAM;
        };
        if blocking == 0 && crate::audio::queued_grains(stream) >= depth as usize {
            return ERR_NOT_READY;
        }
        // The output's layout: the renderer's (the output thread opens its client to match).
        let och = render_layout().channels();
        crate::audio::OUT_CHANNELS.store(och as u32, Ordering::Relaxed);
        let mut mix = vec![0f32; och * n];
        static PUSHES: AtomicU32 = AtomicU32::new(0);
        if PUSHES.fetch_add(1, Ordering::Relaxed) == 2000 {
            dump_ports(ctx, n);
        }
        {
            let ports = PORTS.lock();
            // (Diagnostics: W and Y of the first ambisonic set.)
            let (mut w, mut y, mut x): (Vec<f32>, Vec<f32>, Vec<f32>) = (Vec::new(), Vec::new(), Vec::new());
            let splitting = SPLIT.lock().is_some();
            let mut groups: std::collections::BTreeMap<String, Vec<f32>> = Default::default();
            // The ambisonic field of this grain (both sets summed: the second has been silent).
            let mut field: Vec<Vec<f32>> = vec![Vec::new(); crate::spatial::CHANNELS];
            for p in ports.as_ref().into_iter().flat_map(|m| m.values()).filter(|p| p.ctx == ctx) {
                if ambi_decode() {
                    add_field(p, n, &mut field);
                }
                if splitting {
                    let g = groups.entry(split_group(p)).or_insert_with(|| vec![0f32; 2 * n]);
                    mix_port(p, n, g);
                }
                if p.port_type == PORT_HAPTICS {
                    haptics(p, n);
                    continue;
                }
                if p.port_type == PORT_PAD_SPEAKER && !PAD_SPEAKER.load(Ordering::Relaxed) {
                    continue;
                }
                if !(ambi_decode() && ambi_channel(p).is_some()) {
                    // (Headphones: a bed's speakers through the head model, `add_bed_field`.)
                    if !(ambi_decode() && add_bed_field(p, n, &mut field)) {
                        mix_port_ch(p, n, &mut mix, och);
                    } else {
                        port_energy(p, n);
                    }
                    continue;
                }
                match ambi_channel(p) {
                    Some(acn @ (0 | 1 | 3)) if p.attr9 == 0 && p.latched != 0 && channels(p.format) == 1 && p.format & 0x7F == 0 => {
                        let s: Vec<f32> = (0..n).map(|i| *(p.latched as *const f32).add(i) * p.gains[0]).collect();
                        match acn {
                            0 => w = s,
                            1 => y = s,
                            _ => x = s,
                        }
                    }
                    Some(_) => {}
                    None if is_object(p.port_type) => crate::hle::stub_log("AudioOut2 object ports without an ambisonic channel are mixed to the center", 0),
                    None => {}
                }
                mix_port(p, n, &mut mix);
            }
            if !w.is_empty() && w.len() == y.len() && w.len() == x.len() {
                note_balance(&w, &y, &x);
            }
            render_field(ctx, &field, n, &mut mix);
            if splitting {
                // The field's rendering as its own group (`ambi`).
                let mut r = vec![0f32; 2 * n];
                render_field_copy(&field, n, &mut r);
                groups.insert("ambi-rendered".into(), r);
                let raw = match ports.as_ref() {
                    Some(m) => unsafe { field_raw(m, ctx, n) },
                    None => vec![0f32; FIELD_RAW * n],
                };
                let beds = match ports.as_ref() {
                    Some(m) => unsafe { beds_raw(m, ctx, n) },
                    None => Vec::new(),
                };
                split_add(n, groups, raw, beds);
            }
        }
        // Blocks while `depth` grains are queued (the title's pacing).
        crate::audio::push_stream_ch(stream, mix, och, depth as usize);
        0
    }
    fn sceAudioOut2ContextGetQueueLevel(ctx: u32, level: *mut u32, available: *mut u32) -> i32 {
        let (depth, stream) = CONTEXTS.lock().get_or_insert_with(HashMap::new).get(&ctx).map(|c| (c.queue_depth, c.stream)).unwrap_or((4, usize::MAX));
        let q = (crate::audio::queued_grains(stream) as u32).min(depth);
        if !level.is_null() { *level = q; }
        if !available.is_null() { *available = depth - q; }
        0
    }
    fn sceAudioOut2UserCreate(user: u32, out: *mut u32) -> i32 {
        if out.is_null() { return ERR_INVALID_PARAM; }
        *out = 0x100 + user;
        0
    }
    fn sceAudioOut2UserDestroy(h: u32) -> i32 { 0 }
    /// SceAudioOut2PortParam: {u16 port_type, u16 pad, u32 data_format,
    /// u32 sampling_freq, u32 flags, u32 user_handle, ...}.
    fn sceAudioOut2PortCreate(ctx: u32, p: *const u8, out: *mut u32) -> i32 {
        if p.is_null() || out.is_null() { return ERR_INVALID_PARAM; }
        let port_type = *(p as *const u16);
        let format = *(p.add(4) as *const u32);
        let freq = *(p.add(8) as *const u32);
        if freq != 48000 {
            crate::warn!("sceAudioOut2PortCreate: unsupported sampling frequency {freq}");
            return ERR_INVALID_PARAM;
        }
        if !CONTEXTS.lock().get_or_insert_with(HashMap::new).contains_key(&ctx) { return ERR_INVALID_PARAM; }
        let h = NEXT.fetch_add(1, Ordering::Relaxed);
        PORTS.lock().get_or_insert_with(HashMap::new).insert(h, Port { ctx, port_type, format, pcm: 0, latched: 0, gains: [1.0; 16], attr8: 0, attr9: 0 });
        crate::debug!("sceAudioOut2PortCreate ctx {ctx} type {port_type:#x} format {format:#x} ({} ch) -> {h}", channels(format));
        *out = h;
        0
    }
    fn sceAudioOut2PortDestroy(port: u32) -> i32 {
        PORTS.lock().get_or_insert_with(HashMap::new).remove(&port);
        0
    }
    /// SceAudioOut2Attribute: {u32 id, i32 reserved, const void* value, u64 value_size}.
    fn sceAudioOut2PortSetAttributes(port: u32, attrs: *const u8, num: u32) -> i32 {
        if num != 0 && attrs.is_null() { return ERR_INVALID_PARAM; }
        let mut g = PORTS.lock();
        let Some(p) = g.get_or_insert_with(HashMap::new).get_mut(&port) else { return ERR_INVALID_PORT };
        for i in 0..num as usize {
            let a = attrs.add(i * 24);
            let id = *(a as *const u32);
            let value = *(a.add(8) as *const u64);
            let size = *(a.add(16) as *const u64);
            note_attr(id, value, size);
            trace_attr(port, p.port_type, id, value, size);
            match id {
                ATTR_PCM => p.pcm = if value == 0 { 0 } else { *(value as *const u64) },
                ATTR_GAIN if value != 0 => {
                    let n = (size as usize / 4).min(16);
                    for c in 0..n { p.gains[c] = *((value as *const f32).add(c)); }
                }
                0x8 if value != 0 && size >= 4 => p.attr8 = *(value as *const u32),
                0x9 if value != 0 && size >= 1 => p.attr9 = *(value as *const u8),
                _ => {}
            }
        }
        0
    }
    /// SceAudioOut2PortState: {u16 output, u8 channels, u8, i16 volume, u16 reroute, u32 flags, ...}.
    fn sceAudioOut2PortGetState(port: u32, st: *mut u8) -> i32 {
        if st.is_null() { return ERR_INVALID_PARAM; }
        let Some((own, ty)) = PORTS.lock().get_or_insert_with(HashMap::new).get(&port).map(|p| (channels(p.format), p.port_type)) else {
            return ERR_INVALID_PORT;
        };
        // The title polls its main, BGM and pad-speaker ports and sets its speaker layout and
        // Output Device Type from their state (eboot+0x1dda2a1: the channels through a jump
        // table, AudioOut's state bits 0x01 primary output, 0x04 controller speaker, 0x40
        // headphones). Each port reports its own data format's channels on the primary output:
        // 8 for its 7.1 beds, so the title mixes for 7.1 speakers (its ambisonic field and beds,
        // which `crate::spatial` renders). `DS_AUDIO_PORT_STATE=1` (diagnosis) reports the
        // output as the console would for this PC's (2 channels for stereo or headphones, the
        // headphone bit): the title then masters 6-8 dB louder, sends sound to object ports
        // outside the field (mixed to the centre here) and the mix reached the soft knee 2,974
        // times in 70 s of the cinematic (FINDINGS.md 2026-11-13).
        let (output, ch) = if std::env::var("DS_AUDIO_PORT_STATE").ok().as_deref() != Some("1") {
            (1u16, own as u8)
        } else if ty == PORT_PAD_SPEAKER {
            (4, 1)
        } else {
            match render_layout() {
                crate::spatial::Layout::Headphones => (0x40, 2),
                crate::spatial::Layout::Stereo => (1, 2),
                crate::spatial::Layout::Surround51 => (1, 6),
                _ => (1, 8),
            }
        };
        {
            static N: AtomicU32 = AtomicU32::new(0);
            if N.fetch_add(1, Ordering::Relaxed) < 8 {
                crate::info!("sceAudioOut2PortGetState port {port} (type {ty:#x}, {own} channels) -> output {output:#x}, {ch} channels");
            }
        }
        std::ptr::write_bytes(st, 0, 0x40);
        *(st as *mut u16) = output;
        *st.add(2) = ch;
        *(st.add(4) as *mut i16) = 127;
        0
    }
    /// Stereo speakers at +-30 degrees (the host output is mixed to stereo).
    fn sceAudioOut2GetSpeakerInfo(info: *mut u8, flags: u32) -> i32 {
        if info.is_null() { return ERR_INVALID_PARAM; }
        std::ptr::write_bytes(info, 0, 16 + 64);
        *(info.add(4) as *mut u32) = 0x3;
        *(info.add(16) as *mut i16) = -30;
        *(info.add(20) as *mut i16) = 30;
        0
    }
    fn sceAudioOut2GetSystemState(st: *mut u8) -> i32 {
        if st.is_null() { return ERR_INVALID_PARAM; }
        std::ptr::write_bytes(st, 0, 64);
        0
    }
}

pub fn register(r: &mut Registry) {
    reg!(r, "libSceAudioOut2", [
        sceAudioOut2Initialize, sceAudioOut2ContextResetParam, sceAudioOut2ContextQueryMemory,
        sceAudioOut2ContextCreate, sceAudioOut2ContextDestroy, sceAudioOut2ContextSetAttributes,
        sceAudioOut2ContextAdvance, sceAudioOut2ContextPush, sceAudioOut2ContextGetQueueLevel,
        sceAudioOut2UserCreate, sceAudioOut2UserDestroy, sceAudioOut2PortCreate, sceAudioOut2PortDestroy,
        sceAudioOut2PortSetAttributes, sceAudioOut2PortGetState, sceAudioOut2GetSpeakerInfo, sceAudioOut2GetSystemState,
    ]);
}
