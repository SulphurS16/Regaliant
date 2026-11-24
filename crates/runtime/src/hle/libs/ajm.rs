//! libSceAjm (PS5 batch API): audio decode jobs.
//!
//! The title builds batches with `sceAjmBatchJob*` into its own buffer and
//! starts them with `sceAjmBatchStart`. Jobs are encoded here in this
//! implementation's own format, at the record sizes the system library uses
//! (control 48 bytes, run 64 bytes; see docs/PROVENANCE.md), and executed at
//! BatchStart (the title may fill input buffers after building the batch).
//!
//! Codecs: ATRAC9 via LibAtrac9 (MIT, third_party/libatrac9), MP3 via minimp3
//! (CC0, third_party/minimp3). Other codecs fail their jobs with an explicit
//! "not supported" result (logged).
//!
//! Sideband output layout used by the title (eboot+0x1de2412):
//! SidebandResult{i32 result, i32 internal}, then SidebandStream{i32 consumed,
//! i32 produced, u64 total_decoded_samples} (flag bit 47), then
//! SidebandMFrame{u32 num_frames, u32} (RUN_MULTIPLE_FRAMES, bit 12).

use crate::hle::Registry;
use crate::{reg, sysv};
use parking_lot::Mutex;
use std::collections::HashMap;
use std::ffi::c_void;
use std::sync::atomic::{AtomicU32, Ordering};

unsafe extern "C" {
    fn Atrac9GetHandle() -> *mut c_void;
    fn Atrac9ReleaseHandle(h: *mut c_void);
    fn Atrac9InitDecoder(h: *mut c_void, config: *mut u8) -> i32;
    fn Atrac9Decode(h: *mut c_void, data: *const u8, pcm: *mut i16, used: *mut i32) -> i32;
    fn Atrac9GetCodecInfo(h: *mut c_void, info: *mut At9Info) -> i32;
    fn ds_mp3_new() -> *mut c_void;
    fn ds_mp3_free(d: *mut c_void);
    fn ds_mp3_decode(d: *mut c_void, data: *const u8, len: i32, pcm: *mut f32, info: *mut Mp3Info) -> i32;
}

#[repr(C)]
#[derive(Default, Clone, Copy)]
struct At9Info {
    channels: i32,
    channel_config_index: i32,
    sampling_rate: i32,
    superframe_size: i32,
    frames_in_superframe: i32,
    frame_samples: i32,
    wlength: i32,
    config: [u8; 4],
}

#[repr(C)]
#[derive(Default, Clone, Copy)]
struct Mp3Info {
    frame_bytes: i32,
    frame_offset: i32,
    channels: i32,
    hz: i32,
    layer: i32,
    bitrate_kbps: i32,
}

const ERR_INVALID_CONTEXT: i32 = 0x8093_0002u32 as i32;
const ERR_INVALID_INSTANCE: i32 = 0x8093_0003u32 as i32;
const ERR_INVALID_PARAMETER: i32 = 0x8093_0005u32 as i32;
const ERR_CODEC_NOT_SUPPORTED: i32 = 0x8093_0008u32 as i32;
const ERR_JOB_CREATION: i32 = 0x8093_0012u32 as i32;
/// Per-job result codes (SidebandResult.result).
/// Zeroed bytes kept after a job's gathered input (`run_job`): the
/// ATRAC9 bit reader reads past a frame's end (a superframe is at most a
/// few KiB; corrupt data can run further, still inside this slack in
/// practice).
const INPUT_SLACK: usize = 16 << 10;
const RESULT_INVALID_PARAMETER: i32 = 0x0000_0002;
const RESULT_NOT_INITIALIZED: i32 = 0x0000_0008;
const RESULT_PARTIAL_INPUT: i32 = 0x0000_0020;
const RESULT_CODEC_ERROR: i32 = 0x4000_0000;
const RESULT_INVALID_DATA: i32 = 0x0000_0004;

const CODEC_MP3: u32 = 0;
const CODEC_AT9: u32 = 1;
/// MPEG-4 AAC and Opus (codec 24; id per public sources, see docs/PROVENANCE.md):
/// the title's audio engine requires both modules to register
/// (eboot+0x1de17e0 fails its init otherwise), but these decoders are not
/// implemented yet: their jobs fail with a codec error, logged once each.
const CODEC_AAC: u32 = 2;
const CODEC_OPUS: u32 = 24;

fn codec_name(codec: u32) -> &'static str {
    match codec {
        CODEC_MP3 => "MP3",
        CODEC_AT9 => "ATRAC9",
        CODEC_AAC => "AAC",
        CODEC_OPUS => "Opus",
        _ => "unknown",
    }
}

fn codec_known(codec: u32) -> bool {
    matches!(codec, CODEC_MP3 | CODEC_AT9 | CODEC_AAC | CODEC_OPUS)
}

const FLAG_RUN_GET_CODEC_INFO: u64 = 1 << 11;
const FLAG_RUN_MULTIPLE_FRAMES: u64 = 1 << 12;
const FLAG_SIDEBAND_RESAMPLE_INFO: u64 = 1 << 44;
const FLAG_SIDEBAND_GAPLESS: u64 = 1 << 45;
const FLAG_SIDEBAND_FORMAT: u64 = 1 << 46;
const FLAG_SIDEBAND_STREAM: u64 = 1 << 47;

const CONTROL_SIZE: u64 = 48;
const RUN_SIZE: u64 = 64;
/// Most buffers per side of a split run job.
const MAX_SPLIT: u64 = 16;

/// Run record: 64-byte header + inline {address, size} descriptors.
fn run_size(inputs: u64, outputs: u64) -> u64 {
    RUN_SIZE + 16 * (inputs + outputs)
}
const TAG_CONTROL: u32 = 0x434D_4A41; // "AJMC"
const TAG_RUN: u32 = 0x524D_4A41; // "AJMR"
const OP_INIT: u32 = 1;
const OP_CLEAR: u32 = 2;
const OP_GAPLESS: u32 = 3;
const OP_STATISTICS: u32 = 4;

enum Decoder {
    At9 { handle: *mut c_void, info: At9Info },
    /// Multi-stream ATRAC9 (`at9_multi_config`): a mono decoder per stream,
    /// each stream's superframe `stream_sf` bytes; `info` describes the whole.
    At9Multi { handles: Vec<*mut c_void>, info: At9Info, stream_sf: usize },
    Mp3 { handle: *mut c_void },
    None,
}

unsafe impl Send for Decoder {}

struct Instance {
    codec: u32,
    /// ATRAC9 instance flag (bit 32): inputs may start with the .at9 RIFF
    /// header, which the decoder parses and consumes (fact per public sources; the
    /// title's sound engine feeds embedded .at9 files this way).
    parse_riff: bool,
    /// Output encoding: 0 s16, 1 s32, 2 f32.
    encoding: u32,
    max_channels: u32,
    decoder: Decoder,
    total_decoded: u64,
    gapless_total: u32,
    gapless_skip: u32,
    skipped: u32,
    /// Control operations received (diagnostics, first 8).
    ops: Vec<u32>,
    /// A decode on this instance was reported as not initialized.
    reported: bool,
}

/// A replaced or dropped decoder releases its handle.
impl Drop for Decoder {
    fn drop(&mut self) {
        unsafe {
            match self {
                Decoder::At9 { handle, .. } => Atrac9ReleaseHandle(*handle),
                Decoder::At9Multi { handles, .. } => handles.iter().for_each(|&h| Atrac9ReleaseHandle(h)),
                Decoder::Mp3 { handle } => ds_mp3_free(*handle),
                Decoder::None => {}
            }
        }
    }
}

static INSTANCES: Mutex<Option<HashMap<u32, Instance>>> = Mutex::new(None);
static NEXT_INSTANCE: AtomicU32 = AtomicU32::new(1);
static NEXT_BATCH: AtomicU32 = AtomicU32::new(1);

fn with_instances<R>(f: impl FnOnce(&mut HashMap<u32, Instance>) -> R) -> R {
    f(crate::platform::loader_hooks::ajm_lock(&INSTANCES).get_or_insert_with(HashMap::new))
}

// ---------------------------------------------------------------------------
// Batch building (SceAjmBatchInfo: {buffer, offset, size, last_good_job, last_good_job_ra})

unsafe fn reserve(info: *mut u64, len: u64) -> Option<u64> {
    unsafe {
        if info.is_null() {
            return None;
        }
        let (buf, off, size) = (*info, *info.add(1), *info.add(2));
        if buf == 0 || off + len > size {
            crate::warn!("ajm: batch buffer {buf:#x} full ({off:#x}+{len:#x} > {size:#x})");
            return None;
        }
        *info.add(1) = off + len;
        *info.add(3) = buf + off;
        Some(buf + off)
    }
}

unsafe fn control(info: *mut u64, op: u32, instance: u32, a: u64, b: u64, result: u64, extra: u64) -> i32 {
    unsafe {
        let Some(p) = reserve(info, CONTROL_SIZE) else { return ERR_JOB_CREATION };
        let w = p as *mut u32;
        *w = TAG_CONTROL;
        *w.add(1) = op;
        *w.add(2) = instance;
        *w.add(3) = 0;
        let q = (p + 16) as *mut u64;
        *q = a;
        *q.add(1) = b;
        *q.add(2) = result;
        *q.add(3) = extra;
        0
    }
}

// ---------------------------------------------------------------------------
// Execution

unsafe fn write_result(sb: u64, size: u64, result: i32) {
    if sb != 0 && size >= 8 {
        unsafe {
            *(sb as *mut i32) = result;
            *((sb + 4) as *mut i32) = 0;
        }
    }
}

fn bytes_per_sample(encoding: u32) -> usize {
    if encoding == 0 { 2 } else { 4 }
}

/// Writes interleaved PCM in the instance encoding; returns bytes written.
unsafe fn store_pcm(out: &mut Vec<u8>, encoding: u32, samples: impl Iterator<Item = f32>) {
    for s in samples {
        match encoding {
            0 => out.extend_from_slice(&((s.clamp(-1.0, 1.0) * 32767.0) as i16).to_le_bytes()),
            1 => out.extend_from_slice(&((s.clamp(-1.0, 1.0) as f64 * 2147483647.0) as i32).to_le_bytes()),
            _ => out.extend_from_slice(&s.to_le_bytes()),
        }
    }
}

/// (Re)initializes an ATRAC9 decoder from 4 config bytes: a fresh decoder
/// for each stream. LibAtrac9's initialization sets the configuration but
/// keeps the decoding state of the stream before (the superframe position,
/// a mono block's primary channel and stereo units, previous scale factors,
/// the transform's overlap); a rejected configuration leaves the instance's
/// decoder as it was.
unsafe fn init_at9(inst: &mut Instance, cfg: [u8; 4]) -> bool {
    let Some(decoder) = (unsafe { new_at9(cfg) }) else { return false };
    if let Decoder::At9Multi { handles, .. } = &decoder {
        if MULTI_INITS.fetch_add(1, Ordering::Relaxed) < 2 {
            crate::info!("ajm: multi-stream ATRAC9 decoder set up ({} streams, config {cfg:02x?})", handles.len());
        }
    }
    inst.decoder = decoder;
    inst.total_decoded = 0;
    inst.skipped = 0;
    INITS.fetch_add(1, Ordering::Relaxed);
    true
}

/// Multi-stream ATRAC9, configuration header byte 0x30 (the title's
/// ambisonic .at9 files: 16, 9 and 4 channels): 4 bits sample rate index,
/// 6 bits stream count - 1, 1 bit 0, 11 bits frame bytes - 1, 2 bits
/// superframe index, each stream a mono ATRAC9 stream of that frame size.
/// A superframe holds every stream's superframe: the streams' frames one
/// after another, frame by frame, each stream's last frame followed by its
/// superframe's padding. Read from the title's files (all of them decode
/// so, FINDINGS.md 2026-11-04). Returns the stream count and the streams'
/// standard configuration.
fn at9_multi_config(cfg: [u8; 4]) -> Option<(usize, [u8; 4])> {
    let v = u32::from_be_bytes(cfg);
    if cfg[0] != 0x30 || (v >> 13) & 1 != 0 {
        return None;
    }
    let (rate, streams, frame_m1, sf_index) = ((v >> 20) & 0xf, ((v >> 14) & 0x3f) as usize + 1, (v >> 2) & 0x7ff, v & 3);
    Some((streams, ((0xFE << 24) | (rate << 20) | (frame_m1 << 5) | (sf_index << 3)).to_be_bytes()))
}

/// A fresh ATRAC9 decoder for 4 config bytes, standard or multi-stream.
unsafe fn new_at9(cfg: [u8; 4]) -> Option<Decoder> {
    let open = |c: [u8; 4]| -> Option<(*mut c_void, At9Info)> {
        let mut c = c;
        let h = unsafe { Atrac9GetHandle() };
        if unsafe { Atrac9InitDecoder(h, c.as_mut_ptr()) } != 0 {
            unsafe { Atrac9ReleaseHandle(h) };
            return None;
        }
        let mut info = At9Info::default();
        unsafe { Atrac9GetCodecInfo(h, &mut info) };
        Some((h, info))
    };
    let Some((streams, mono)) = at9_multi_config(cfg) else {
        return open(cfg).map(|(handle, info)| Decoder::At9 { handle, info });
    };
    let mut handles = Vec::with_capacity(streams);
    let mut info = At9Info::default();
    for _ in 0..streams {
        let Some((h, i)) = open(mono) else {
            handles.iter().for_each(|&h| unsafe { Atrac9ReleaseHandle(h) });
            return None;
        };
        handles.push(h);
        info = i;
    }
    let stream_sf = info.superframe_size as usize;
    info.channels = streams as i32;
    info.superframe_size *= streams as i32;
    info.config = cfg;
    Some(Decoder::At9Multi { handles, info, stream_sf })
}

/// Time decoding under the instances' lock (status reports).
static DECODE_NS: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);
/// Decoders set up (multi-stream ones also apart) and contexts cleared
/// (status reports).
static INITS: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);
static MULTI_INITS: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);
static CLEARS: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);

/// .at9 RIFF header: 'fmt ' (WAVE_FORMAT_EXTENSIBLE with the ATRAC9 GUID,
/// config data at +44), optional 'fact' (total samples, skip samples), then
/// 'data'. Returns the payload offset of 'data', or a result code.
unsafe fn parse_at9_riff(inst: &mut Instance, input: &[u8]) -> Result<usize, i32> {
    const AT9_GUID: [u8; 16] = [0xd2, 0x42, 0xe1, 0x47, 0xba, 0x36, 0x8d, 0x4d, 0x88, 0xfc, 0x61, 0x65, 0x4f, 0x8c, 0x83, 0x6c];
    let le32 = |b: &[u8]| u32::from_le_bytes([b[0], b[1], b[2], b[3]]) as usize;
    if input.len() < 12 || &input[8..12] != b"WAVE" {
        return Err(RESULT_INVALID_DATA);
    }
    let mut off = 12usize;
    while off + 8 <= input.len() {
        let id = &input[off..off + 4];
        let size = le32(&input[off + 4..]);
        let payload = off + 8;
        if id == b"data" {
            return Ok(payload);
        }
        if payload + size > input.len() {
            return Err(RESULT_PARTIAL_INPUT);
        }
        let body = &input[payload..payload + size];
        if id == b"fmt " {
            if size < 48 || u16::from_le_bytes([body[0], body[1]]) != 0xFFFE || body[24..40] != AT9_GUID {
                return Err(RESULT_INVALID_DATA);
            }
            let cfg = [body[44], body[45], body[46], body[47]];
            if !unsafe { init_at9(inst, cfg) } {
                return Err(RESULT_INVALID_DATA);
            }
        } else if id == b"fact" && size >= 12 {
            inst.gapless_total = le32(body) as u32;
            inst.gapless_skip = le32(&body[4..]) as u32;
            inst.skipped = 0;
        }
        off = payload + size + (size & 1);
    }
    Err(RESULT_PARTIAL_INPUT)
}

struct Outcome {
    result: i32,
    consumed: usize,
    produced: usize,
    frames: u32,
}

/// This port's error for a multi-stream block past its superframe.
const ERR_AT9_LAYOUT: i32 = 0x82ff_0000u32 as i32;

/// Logs the first failure of each error and frame (8 at most), the whole
/// input saved beside the run's log (diagnosis with the at9_offline test).
fn note_at9_error(r: i32, input: &[u8], at: usize, info: &At9Info) {
    static REPORTED: Mutex<Vec<(i32, [u8; 16])>> = Mutex::new(Vec::new());
    let mut key = (r, [0u8; 16]);
    let hn = input.len().saturating_sub(at).min(16);
    key.1[..hn].copy_from_slice(&input[at..at + hn]);
    let k = {
        let mut seen = REPORTED.lock();
        if seen.len() >= 8 || seen.contains(&key) {
            return;
        }
        seen.push(key);
        seen.len() - 1
    };
    let head: Vec<String> = key.1[..hn].iter().map(|b| format!("{b:02x}")).collect();
    // (Development checkouts only: the input is the game's audio, and a player's log folder is
    // what they attach to a public report.)
    let dumped = crate::run_dir().filter(|_| crate::paths().layout == dsconfig::paths::Layout::Dev).map(|d| d.join(format!("ajm_fail_{k}.bin"))).filter(|f| std::fs::write(f, input).is_ok());
    crate::warn!(
        "ajm: ATRAC9 decode error {:#x} at input {at}/{} (superframe {} bytes, {} frame(s) x {} samples, {} ch, config {:02x?}); input {}{}",
        r as u32,
        input.len(),
        info.superframe_size,
        info.frames_in_superframe,
        info.frame_samples,
        info.channels,
        info.config,
        head.join(""),
        dumped.map(|f| format!("; input saved to {}", f.display())).unwrap_or_default()
    );
}

/// Decodes from `input` into `out_cap` bytes of PCM (`out`).
unsafe fn decode(inst: &mut Instance, input: &[u8], out_cap: usize, multiple: bool, out: &mut Vec<u8>) -> Outcome {
    let mut o = Outcome { result: 0, consumed: 0, produced: 0, frames: 0 };
    let bps = bytes_per_sample(inst.encoding);
    if inst.parse_riff && input.len() >= 4 && &input[..4] == b"RIFF" {
        match unsafe { parse_at9_riff(inst, input) } {
            Ok(data) => o.consumed = data,
            Err(r) => {
                o.result |= r;
                return o;
            }
        }
    }
    loop {
        let mut pcm_f: Vec<f32> = Vec::new();
        let mut channels = 0usize;
        let used: usize;
        match &mut inst.decoder {
            Decoder::At9 { handle, info } => {
                let sf = info.superframe_size as usize;
                let n = info.frames_in_superframe as usize;
                let fs = info.frame_samples as usize;
                channels = info.channels as usize;
                if input.len() - o.consumed < sf {
                    if o.frames == 0 {
                        o.result |= RESULT_PARTIAL_INPUT;
                    }
                    break;
                }
                let need = n * fs * channels * bps;
                if out_cap - o.produced < need {
                    break;
                }
                let mut pcm = vec![0i16; fs * channels];
                let mut pos = 0usize;
                for _ in 0..n {
                    let mut u = 0i32;
                    let r = unsafe { Atrac9Decode(*handle, input.as_ptr().add(o.consumed + pos), pcm.as_mut_ptr(), &mut u) };
                    if r != 0 {
                        o.result |= RESULT_CODEC_ERROR;
                        note_at9_error(r, input, o.consumed + pos, info);
                        return o;
                    }
                    pos += u.max(0) as usize;
                    pcm_f.extend(pcm.iter().map(|&s| s as f32 / 32768.0));
                }
                used = sf;
            }
            Decoder::At9Multi { handles, info, stream_sf } => {
                let sf = info.superframe_size as usize;
                let n = info.frames_in_superframe as usize;
                let fs = info.frame_samples as usize;
                channels = handles.len();
                if input.len() - o.consumed < sf {
                    if o.frames == 0 {
                        o.result |= RESULT_PARTIAL_INPUT;
                    }
                    break;
                }
                if out_cap - o.produced < n * fs * channels * bps {
                    break;
                }
                // Frame by frame, each stream's block after the one before;
                // a stream's last block takes the rest of its superframe.
                let mut pcm = vec![0i16; fs];
                let mut taken = vec![0usize; channels];
                let mut pos = 0usize;
                pcm_f.resize(n * fs * channels, 0.0);
                for f in 0..n {
                    for (c, &h) in handles.iter().enumerate() {
                        let mut u = 0i32;
                        let r = if pos < sf { unsafe { Atrac9Decode(h, input.as_ptr().add(o.consumed + pos), pcm.as_mut_ptr(), &mut u) } } else { ERR_AT9_LAYOUT };
                        if r != 0 {
                            o.result |= RESULT_CODEC_ERROR;
                            note_at9_error(r, input, o.consumed + pos.min(sf), info);
                            return o;
                        }
                        for (s, &v) in pcm.iter().enumerate() {
                            pcm_f[(f * fs + s) * channels + c] = v as f32 / 32768.0;
                        }
                        if f + 1 < n {
                            pos += u.max(0) as usize;
                            taken[c] += u.max(0) as usize;
                        } else {
                            pos += stream_sf.saturating_sub(taken[c]);
                        }
                    }
                }
                used = sf;
            }
            Decoder::Mp3 { handle } => {
                let rest = &input[o.consumed..];
                if rest.is_empty() {
                    break;
                }
                let mut pcm = vec![0f32; 1152 * 2];
                let mut info = Mp3Info::default();
                let samples = unsafe { ds_mp3_decode(*handle, rest.as_ptr(), rest.len() as i32, pcm.as_mut_ptr(), &mut info) };
                if info.frame_bytes == 0 {
                    if o.frames == 0 {
                        o.result |= RESULT_PARTIAL_INPUT;
                    }
                    break;
                }
                channels = info.channels.max(1) as usize;
                let need = samples as usize * channels * bps;
                if out_cap - o.produced < need {
                    break;
                }
                pcm_f.extend_from_slice(&pcm[..samples as usize * channels]);
                used = info.frame_bytes as usize;
            }
            Decoder::None => {
                if inst.codec == CODEC_AAC || inst.codec == CODEC_OPUS {
                    let msg = if inst.codec == CODEC_AAC { "ajm: AAC decode job failed (AAC decoding not implemented)" } else { "ajm: Opus decode job failed (Opus decoding not implemented)" };
                    crate::hle::stub_log(msg, RESULT_CODEC_ERROR as i64);
                    o.result |= RESULT_CODEC_ERROR;
                } else if inst.parse_riff && input.len() == o.consumed {
                    // Waiting for the RIFF header with no input: nothing to
                    // do. Failing these jobs (as "not initialized") made the
                    // title retry them ~370 times a second; reporting partial
                    // input stalled its mixer (no audio in Outpost Passage).
                } else {
                    o.result |= RESULT_NOT_INITIALIZED;
                    static NOTED: AtomicU32 = AtomicU32::new(0);
                    if !inst.reported && NOTED.fetch_add(1, Ordering::Relaxed) < 8 {
                        inst.reported = true;
                        let head: Vec<String> = input.iter().take(16).map(|b| format!("{b:02x}")).collect();
                        crate::warn!(
                            "ajm: decode on an uninitialized {} instance (riff {}, control ops {:?}, input {} bytes: {})",
                            codec_name(inst.codec), inst.parse_riff, inst.ops, input.len(), head.join("")
                        );
                    }
                }
                break;
            }
        }
        o.consumed += used;
        o.frames += 1;
        // Gapless: drop leading skip samples, stop at the declared total.
        let frames_in = pcm_f.len() / channels.max(1);
        let mut start = 0usize;
        if inst.skipped < inst.gapless_skip {
            let k = ((inst.gapless_skip - inst.skipped) as usize).min(frames_in);
            inst.skipped += k as u32;
            start = k;
        }
        let mut end = frames_in;
        if inst.gapless_total != 0 && inst.gapless_total != u32::MAX {
            let left = (inst.gapless_total as u64).saturating_sub(inst.total_decoded) as usize;
            end = end.min(start + left);
        }
        let keep = &pcm_f[start * channels..end.max(start) * channels];
        let before = out.len();
        unsafe { store_pcm(out, inst.encoding, keep.iter().copied()) };
        o.produced += out.len() - before;
        inst.total_decoded += (end.max(start) - start) as u64;
        if !multiple {
            break;
        }
    }
    o
}

struct Stats {
    jobs: std::sync::atomic::AtomicU64,
    frames: std::sync::atomic::AtomicU64,
    bytes: std::sync::atomic::AtomicU64,
    failed: std::sync::atomic::AtomicU64,
    last_error: std::sync::atomic::AtomicU64,
}

static STATS: Stats = Stats {
    jobs: std::sync::atomic::AtomicU64::new(0),
    frames: std::sync::atomic::AtomicU64::new(0),
    bytes: std::sync::atomic::AtomicU64::new(0),
    failed: std::sync::atomic::AtomicU64::new(0),
    last_error: std::sync::atomic::AtomicU64::new(0),
};

/// Decode statistics for status reports.
pub fn status() -> String {
    format!(
        "    ajm: {} run job(s), {} frame(s) decoded, {} PCM bytes, {} with errors (last result {:#x}); {} ATRAC9 decoder(s) set up ({} multi-stream), {} context clear(s); {} ms decoding under the lock\n",
        STATS.jobs.load(Ordering::Relaxed),
        STATS.frames.load(Ordering::Relaxed),
        STATS.bytes.load(Ordering::Relaxed),
        STATS.failed.load(Ordering::Relaxed),
        STATS.last_error.load(Ordering::Relaxed),
        INITS.load(Ordering::Relaxed),
        MULTI_INITS.load(Ordering::Relaxed),
        CLEARS.load(Ordering::Relaxed),
        DECODE_NS.load(Ordering::Relaxed) / 1_000_000
    )
}

unsafe fn run_job(p: u64) -> i32 {
    unsafe {
        let instance = *((p + 4) as *const u32);
        let flags = *((p + 8) as *const u64);
        let q = (p + 16) as *const u64;
        let (ins, nin, outs, nout, sb, sbsize) = (*q, *q.add(1), *q.add(2), *q.add(3), *q.add(4), *q.add(5));
        // Gather input and output buffers ({address, size} arrays). Sizes are
        // bounded: a job whose descriptors do not parse is rejected and logged.
        const MAX_BUF: u64 = 64 << 20;
        let sane = |base: u64, n: u64| -> bool {
            n <= 64 && (n == 0 || base != 0) && (0..n).all(|i| *((base + 16 * i + 8) as *const u64) <= MAX_BUF)
        };
        if !sane(ins, nin) || !sane(outs, nout) {
            crate::hle::stub_log("ajm: run job with implausible buffer descriptors rejected (see warning)", RESULT_INVALID_PARAMETER as i64);
            crate::warn!("ajm: rejected run job at {p:#x}: instance {instance} flags {flags:#x} inputs {ins:#x} x{nin} outputs {outs:#x} x{nout}");
            write_result(sb, sbsize, RESULT_INVALID_PARAMETER);
            return 0;
        }
        let mut input = Vec::new();
        for i in 0..nin {
            let a = *((ins + 16 * i) as *const u64);
            let n = *((ins + 16 * i + 8) as *const u64);
            if a != 0 && n != 0 {
                input.extend_from_slice(std::slice::from_raw_parts(a as *const u8, n as usize));
            }
        }
        // LibAtrac9's bit reader reads a few bytes past a frame (further
        // on corrupt data) without bounds: zeroed slack after the input, so
        // that its allocation never ends right behind the last frame (a job
        // worker faulted in `PeekInt` reading past a gathered input at an
        // unmapped page, FINDINGS.md 2026-10-19 afternoon). The decoder sees
        // the input's length; the slack stays in the allocation.
        let len = input.len();
        input.resize(len + INPUT_SLACK, 0);
        input.truncate(len);
        let out_cap: usize = (0..nout).map(|i| *((outs + 16 * i + 8) as *const u64) as usize).sum();
        let mut pcm = Vec::with_capacity(out_cap);
        let (outcome, fmt) = {
            let mut g = crate::platform::loader_hooks::ajm_lock(&INSTANCES);
            let m = g.get_or_insert_with(HashMap::new);
            let Some(inst) = m.get_mut(&instance) else {
                write_result(sb, sbsize, RESULT_INVALID_PARAMETER);
                return 0;
            };
            let t = std::time::Instant::now();
            let o = decode(inst, &input, out_cap, flags & FLAG_RUN_MULTIPLE_FRAMES != 0, &mut pcm);
            DECODE_NS.fetch_add(t.elapsed().as_nanos() as u64, Ordering::Relaxed);
            let fmt = match &inst.decoder {
                Decoder::At9 { info, .. } | Decoder::At9Multi { info, .. } => (info.channels as u32, info.sampling_rate as u32),
                _ => (2, 48000),
            };
            (o, (fmt, inst.encoding, inst.total_decoded, inst.gapless_total, inst.gapless_skip, inst.skipped))
        };
        // Scatter PCM into the output buffers.
        let mut off = 0usize;
        for i in 0..nout {
            let a = *((outs + 16 * i) as *const u64);
            let n = *((outs + 16 * i + 8) as *const u64) as usize;
            let k = n.min(pcm.len().saturating_sub(off));
            if a != 0 && k > 0 {
                std::ptr::copy_nonoverlapping(pcm.as_ptr().add(off), a as *mut u8, k);
            }
            off += k;
        }
        // Sideband: result, [resample], [gapless], [format], [stream], [mframe].
        STATS.jobs.fetch_add(1, Ordering::Relaxed);
        STATS.frames.fetch_add(outcome.frames as u64, Ordering::Relaxed);
        STATS.bytes.fetch_add(outcome.produced as u64, Ordering::Relaxed);
        if outcome.result != 0 {
            STATS.failed.fetch_add(1, Ordering::Relaxed);
            STATS.last_error.store(outcome.result as u32 as u64, Ordering::Relaxed);
        }
        write_result(sb, sbsize, outcome.result);
        let mut at = 8u64;
        let room = |at: u64, n: u64| sb != 0 && at + n <= sbsize;
        if flags & FLAG_SIDEBAND_RESAMPLE_INFO != 0 && room(at, 40) {
            std::ptr::write_bytes((sb + at) as *mut u8, 0, 40);
            *((sb + at) as *mut f32) = 1.0;
            at += 40;
        }
        if flags & FLAG_SIDEBAND_GAPLESS != 0 && room(at, 8) {
            *((sb + at) as *mut u32) = fmt.3;
            *((sb + at + 4) as *mut u16) = fmt.4 as u16;
            *((sb + at + 6) as *mut u16) = fmt.5 as u16;
            at += 8;
        }
        if flags & (FLAG_SIDEBAND_FORMAT | FLAG_RUN_GET_CODEC_INFO) != 0 && room(at, 24) {
            let w = (sb + at) as *mut u32;
            *w = (fmt.0).0;
            *w.add(1) = if (fmt.0).0 == 1 { 0x4 } else { 0x3 };
            *w.add(2) = (fmt.0).1;
            *w.add(3) = fmt.1;
            *w.add(4) = 0;
            *w.add(5) = 0;
            at += 24;
        }
        if flags & FLAG_SIDEBAND_STREAM != 0 && room(at, 16) {
            *((sb + at) as *mut i32) = outcome.consumed as i32;
            *((sb + at + 4) as *mut i32) = outcome.produced as i32;
            *((sb + at + 8) as *mut u64) = fmt.2;
            at += 16;
        }
        if flags & FLAG_RUN_MULTIPLE_FRAMES != 0 && room(at, 8) {
            *((sb + at) as *mut u32) = outcome.frames;
            *((sb + at + 4) as *mut u32) = 0;
        }
        0
    }
}

unsafe fn control_job(p: u64) {
    unsafe {
        let op = *((p + 4) as *const u32);
        let instance = *((p + 8) as *const u32);
        let q = (p + 16) as *const u64;
        let (a, b, result) = (*q, *q.add(1), *q.add(2));
        let r = with_instances(|m| {
            let Some(inst) = m.get_mut(&instance) else { return RESULT_INVALID_PARAMETER };
            if inst.ops.len() < 8 {
                inst.ops.push(op);
            }
            match op {
                OP_INIT => match inst.codec {
                    CODEC_AT9 => {
                        // SceAjmDecAt9InitializeParameters: {u8 config_data[4], u32 reserved}.
                        if a == 0 {
                            return RESULT_INVALID_PARAMETER;
                        }
                        let mut cfg = [0u8; 4];
                        std::ptr::copy_nonoverlapping(a as *const u8, cfg.as_mut_ptr(), 4);
                        if init_at9(inst, cfg) {
                            0
                        } else if inst.parse_riff && cfg[0] != 0xFE {
                            // RIFF-fed instances get their configuration from
                            // the .at9 header of the first input: with
                            // parameters the decoder cannot take, reset and
                            // wait for the header. (The {u32 0x7ec17330} seen
                            // in Outpost Passage is a multi-stream
                            // configuration, `at9_multi_config`.)
                            inst.decoder = Decoder::None;
                            inst.total_decoded = 0;
                            inst.skipped = 0;
                            0
                        } else {
                            static BAD: AtomicU32 = AtomicU32::new(0);
                            if BAD.fetch_add(1, Ordering::Relaxed) < 4 {
                                let raw: Vec<String> = (0..32).map(|i| format!("{:02x}", *((a + i) as *const u8))).collect();
                                crate::warn!("ajm: instance {instance}: ATRAC9 init rejected config {:02x?} (parameters at {a:#x}: {}; b {b:#x})", cfg, raw.join(""));
                            }
                            RESULT_INVALID_PARAMETER
                        }
                    }
                    CODEC_MP3 => {
                        if matches!(inst.decoder, Decoder::None) {
                            inst.decoder = Decoder::Mp3 { handle: ds_mp3_new() };
                        }
                        inst.total_decoded = 0;
                        inst.skipped = 0;
                        0
                    }
                    CODEC_AAC | CODEC_OPUS => 0,
                    _ => RESULT_INVALID_PARAMETER,
                },
                OP_CLEAR => {
                    // A fresh decoder with the same configuration.
                    inst.total_decoded = 0;
                    inst.skipped = 0;
                    CLEARS.fetch_add(1, Ordering::Relaxed);
                    match inst.decoder {
                        Decoder::At9 { info, .. } | Decoder::At9Multi { info, .. } => {
                            init_at9(inst, info.config);
                        }
                        Decoder::Mp3 { .. } => inst.decoder = Decoder::Mp3 { handle: ds_mp3_new() },
                        Decoder::None => {}
                    }
                    0
                }
                OP_GAPLESS => {
                    // SceAjmSidebandGaplessDecode {u32 total_samples, u16 skip, u16 skipped}; b = reset.
                    if a != 0 {
                        inst.gapless_total = *(a as *const u32);
                        inst.gapless_skip = *((a + 4) as *const u16) as u32;
                    }
                    if b != 0 {
                        inst.skipped = 0;
                        inst.total_decoded = 0;
                    }
                    0
                }
                _ => 0,
            }
        });
        if op == OP_STATISTICS && result != 0 {
            // SceAjmGetStatisticsResult: result, engine usage (batch + 3 intervals), memory[6].
            std::ptr::write_bytes(result as *mut u8, 0, 8 + 16 + 24);
            return;
        }
        write_result(result, 8, r);
    }
}

sysv! {
    fn sceAjmInitialize(reserved: i64, ctx: *mut u32) -> i32 {
        if ctx.is_null() { return ERR_INVALID_PARAMETER; }
        *ctx = 1;
        0
    }
    fn sceAjmFinalize(ctx: u32) -> i32 { 0 }
    fn sceAjmMemoryRegister(ctx: u32, ptr: u64, pages: u64) -> i32 { if ctx != 1 { ERR_INVALID_CONTEXT } else { 0 } }
    fn sceAjmMemoryUnregister(ctx: u32, ptr: u64) -> i32 { 0 }
    fn sceAjmModuleRegister(ctx: u32, codec: u32, reserved: i64) -> i32 {
        if ctx != 1 { return ERR_INVALID_CONTEXT; }
        match codec {
            CODEC_MP3 | CODEC_AT9 => 0,
            CODEC_AAC | CODEC_OPUS => {
                crate::warn!("ajm: {} module registered, but {} decoding is not implemented yet (its jobs will fail)", codec_name(codec), codec_name(codec));
                0
            }
            other => {
                crate::warn!("ajm: codec {other} is not supported");
                ERR_CODEC_NOT_SUPPORTED
            }
        }
    }
    fn sceAjmModuleUnregister(ctx: u32, codec: u32) -> i32 { 0 }
    /// Instance flags: max channels [6:0], output encoding [9:7] (0 s16, 1 s32, 2 f32).
    fn sceAjmInstanceCreate(ctx: u32, codec: u32, flags: u64, out: *mut u32) -> i32 {
        if ctx != 1 { return ERR_INVALID_CONTEXT; }
        if out.is_null() { return ERR_INVALID_PARAMETER; }
        if !codec_known(codec) { return ERR_CODEC_NOT_SUPPORTED; }
        let id = NEXT_INSTANCE.fetch_add(1, Ordering::Relaxed);
        let inst = Instance {
            codec,
            parse_riff: codec == CODEC_AT9 && flags & (1 << 32) != 0,
            encoding: ((flags >> 7) & 7) as u32,
            max_channels: (flags & 0x7F) as u32,
            decoder: Decoder::None,
            total_decoded: 0,
            gapless_total: 0,
            gapless_skip: 0,
            skipped: 0,
            ops: Vec::new(),
            reported: false,
        };
        with_instances(|m| m.insert(id, inst));
        *out = id;
        0
    }
    fn sceAjmInstanceDestroy(ctx: u32, instance: u32) -> i32 {
        if with_instances(|m| m.remove(&instance)).is_none() { return ERR_INVALID_INSTANCE; }
        0
    }
    fn sceAjmBatchInitialize(buffer: u64, size: u64, info: *mut u64) -> i32 {
        if info.is_null() { return ERR_INVALID_PARAMETER; }
        *info = buffer;
        *info.add(1) = 0;
        *info.add(2) = size;
        *info.add(3) = 0;
        *info.add(4) = 0;
        0
    }
    fn sceAjmBatchJobInitialize(info: *mut u64, instance: u32, param: u64, param_size: u64, result: u64) -> i32 {
        control(info, OP_INIT, instance, param, param_size, result, 0)
    }
    fn sceAjmBatchJobClearContext(info: *mut u64, instance: u32, result: u64) -> i32 {
        control(info, OP_CLEAR, instance, 0, 0, result, 0)
    }
    fn sceAjmBatchJobSetGaplessDecode(info: *mut u64, instance: u32, gapless: u64, reset: i32, result: u64) -> i32 {
        control(info, OP_GAPLESS, instance, gapless, reset as u64, result, 0)
    }
    fn sceAjmBatchJobGetStatistics(info: *mut u64, interval: f32, result: u64) -> i32 {
        control(info, OP_STATISTICS, 0, 0, 0, result, 0)
    }
    /// The {address, size} descriptor arrays are copied into the batch (titles
    /// build them on the stack and start the batch later); the buffers they
    /// describe and the sideband are used when the batch runs.
    fn sceAjmBatchJobRunSplit(info: *mut u64, instance: u32, flags: u64, inputs: u64, num_inputs: u64, outputs: u64, num_outputs: u64, sideband: u64, sideband_size: u64) -> i32 {
        if num_inputs > MAX_SPLIT || num_outputs > MAX_SPLIT || (num_inputs != 0 && inputs == 0) || (num_outputs != 0 && outputs == 0) {
            return ERR_INVALID_PARAMETER;
        }
        let Some(p) = reserve(info, run_size(num_inputs, num_outputs)) else { return ERR_JOB_CREATION };
        *(p as *mut u32) = TAG_RUN;
        *((p + 4) as *mut u32) = instance;
        *((p + 8) as *mut u64) = flags;
        let inline_in = p + RUN_SIZE;
        let inline_out = inline_in + 16 * num_inputs;
        std::ptr::copy_nonoverlapping(inputs as *const u8, inline_in as *mut u8, 16 * num_inputs as usize);
        std::ptr::copy_nonoverlapping(outputs as *const u8, inline_out as *mut u8, 16 * num_outputs as usize);
        let q = (p + 16) as *mut u64;
        *q = inline_in;
        *q.add(1) = num_inputs;
        *q.add(2) = inline_out;
        *q.add(3) = num_outputs;
        *q.add(4) = sideband;
        *q.add(5) = sideband_size;
        0
    }
    /// Executes the batch synchronously; BatchWait then returns at once.
    fn sceAjmBatchStart(ctx: u32, info: *const u64, prio: i32, error: *mut u8, batch: *mut u32) -> i32 {
        if ctx != 1 { return ERR_INVALID_CONTEXT; }
        if info.is_null() { return ERR_INVALID_PARAMETER; }
        let (buf, len) = (*info, *info.add(1));
        let mut off = 0u64;
        while off < len {
            let p = buf + off;
            match *(p as *const u32) {
                TAG_CONTROL => { control_job(p); off += CONTROL_SIZE; }
                TAG_RUN => {
                    run_job(p);
                    let q = (p + 16) as *const u64;
                    off += run_size((*q.add(1)).min(MAX_SPLIT), (*q.add(3)).min(MAX_SPLIT));
                }
                other => {
                    crate::error!("ajm: unknown job tag {other:#x} at batch offset {off:#x}");
                    return ERR_INVALID_PARAMETER;
                }
            }
        }
        if !error.is_null() { std::ptr::write_bytes(error, 0, 32); }
        if !batch.is_null() { *batch = NEXT_BATCH.fetch_add(1, Ordering::Relaxed); }
        0
    }
    fn sceAjmBatchWait(ctx: u32, batch: u32, timeout: u32, error: *mut u8) -> i32 {
        if !error.is_null() { std::ptr::write_bytes(error, 0, 32); }
        0
    }
    fn sceAjmBatchErrorDump(info: *const u64, error: *mut u8) -> i32 { 0 }
    /// SceAjmDecAt9ConfigDataInfo: {channels, sample_rate, frame_samples_per_channel,
    /// superframe_samples_per_channel, superframe_size}.
    fn sceAjmDecAt9ParseConfigData(config: *const u8, out: *mut u32) -> i32 {
        if config.is_null() || out.is_null() { return ERR_INVALID_PARAMETER; }
        let mut cfg = [0u8; 4];
        std::ptr::copy_nonoverlapping(config, cfg.as_mut_ptr(), 4);
        // Decoder setup writes libatrac9's shared tables: under the lock the
        // decoders run under.
        let info = {
            let _g = crate::platform::loader_hooks::ajm_lock(&INSTANCES);
            match new_at9(cfg) {
                Some(Decoder::At9 { info, .. }) | Some(Decoder::At9Multi { info, .. }) => info,
                _ => return ERR_INVALID_PARAMETER,
            }
        };
        if info.channels <= 0 { return ERR_INVALID_PARAMETER; }
        *out = info.channels as u32;
        *out.add(1) = info.sampling_rate as u32;
        *out.add(2) = info.frame_samples as u32;
        *out.add(3) = (info.frame_samples * info.frames_in_superframe) as u32;
        *out.add(4) = info.superframe_size as u32;
        0
    }
}

pub fn register(r: &mut Registry) {
    reg!(r, "libSceAjm", [
        sceAjmInitialize, sceAjmFinalize, sceAjmMemoryRegister, sceAjmMemoryUnregister, sceAjmModuleRegister,
        sceAjmModuleUnregister, sceAjmInstanceCreate, sceAjmInstanceDestroy, sceAjmBatchInitialize,
        sceAjmBatchJobInitialize, sceAjmBatchJobClearContext, sceAjmBatchJobSetGaplessDecode,
        sceAjmBatchJobGetStatistics, sceAjmBatchJobRunSplit, sceAjmBatchStart, sceAjmBatchWait,
        sceAjmBatchErrorDump, sceAjmDecAt9ParseConfigData,
    ]);
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Diagnosis: decodes an input an ATRAC9 decode error saved
    /// (`ajm_fail_<k>.bin`) with the 4 configuration bytes it was decoded
    /// with, or with the configuration of its own .at9 header when it has one,
    /// and reports where libatrac9 fails, trying each byte offset of the first
    /// frames too. `DS_AT9_INPUT=<file> DS_AT9_CONFIG=<8 hex digits> cargo
    /// test --release -p dsport at9_offline -- --ignored --nocapture`.
    #[test]
    #[ignore]
    fn at9_offline() {
        let path = std::env::var("DS_AT9_INPUT").expect("DS_AT9_INPUT");
        let input = std::fs::read(&path).expect("input");
        let hex = std::env::var("DS_AT9_CONFIG").unwrap_or_default();
        let mut cfg = [0u8; 4];
        for (k, b) in cfg.iter_mut().enumerate() {
            *b = u8::from_str_radix(hex.get(2 * k..2 * k + 2).unwrap_or("00"), 16).unwrap_or(0);
        }
        let mut start = 0usize;
        if input.len() > 12 && &input[..4] == b"RIFF" {
            let mut inst = Instance { codec: CODEC_AT9, parse_riff: true, encoding: 0, max_channels: 2, decoder: Decoder::None, total_decoded: 0, gapless_total: 0, gapless_skip: 0, skipped: 0, ops: Vec::new(), reported: false };
            match unsafe { parse_at9_riff(&mut inst, &input) } {
                Ok(data) => {
                    start = data;
                    if let Decoder::At9 { info, .. } = &inst.decoder {
                        println!("RIFF header: data at {data}, config {:02x?}", info.config);
                        if hex.is_empty() {
                            cfg = info.config;
                        }
                    }
                }
                Err(r) => println!("RIFF header not parsed: {r:#x}"),
            }
        }
        unsafe {
            let h = Atrac9GetHandle();
            // DS_AT9_PRE: an .at9 file decoded first on the same handle (state
            // a reused instance carries into the next stream).
            if let Ok(pre) = std::env::var("DS_AT9_PRE") {
                let pre = std::fs::read(pre).expect("DS_AT9_PRE");
                let mut inst = Instance { codec: CODEC_AT9, parse_riff: true, encoding: 0, max_channels: 2, decoder: Decoder::None, total_decoded: 0, gapless_total: 0, gapless_skip: 0, skipped: 0, ops: Vec::new(), reported: false };
                let data = parse_at9_riff(&mut inst, &pre).expect("DS_AT9_PRE's RIFF header");
                if let Decoder::At9 { info, .. } = &inst.decoder {
                    let mut c = info.config;
                    assert_eq!(Atrac9InitDecoder(h, c.as_mut_ptr()), 0);
                    let mut pi = At9Info::default();
                    Atrac9GetCodecInfo(h, &mut pi);
                    let mut pcm = vec![0i16; pi.frame_samples as usize * pi.channels.max(1) as usize];
                    let (mut at, mut ok, mut bad) = (data, 0usize, 0usize);
                    // DS_AT9_PRE_SUPERFRAMES: only its first n superframes.
                    let n: usize = std::env::var("DS_AT9_PRE_SUPERFRAMES").ok().and_then(|v| v.parse().ok()).unwrap_or(usize::MAX);
                    while at + pi.superframe_size as usize <= pre.len().min(data.saturating_add(n.saturating_mul(pi.superframe_size as usize))) {
                        let mut pos = 0usize;
                        for _ in 0..pi.frames_in_superframe {
                            let mut used = 0i32;
                            if Atrac9Decode(h, pre.as_ptr().add(at + pos), pcm.as_mut_ptr(), &mut used) == 0 { ok += 1 } else { bad += 1 }
                            pos += used.max(0) as usize;
                        }
                        at += pi.superframe_size as usize;
                    }
                    println!("pre-stream config {:02x?}: {ok} frame(s) decoded, {bad} failed", info.config);
                }
            }
            assert_eq!(Atrac9InitDecoder(h, cfg.as_mut_ptr()), 0, "config {cfg:02x?} rejected");
            let mut info = At9Info::default();
            Atrac9GetCodecInfo(h, &mut info);
            println!("config {cfg:02x?}: {} ch, {} Hz, superframe {} bytes, {} frame(s) x {} samples", info.channels, info.sampling_rate, info.superframe_size, info.frames_in_superframe, info.frame_samples);
            let mut pcm = vec![0i16; info.frame_samples as usize * info.channels.max(1) as usize];
            // As decode() does: a superframe's frames one after another, the
            // next superframe at the superframe size.
            let (mut at, mut frames) = (start, 0usize);
            'sf: while at + info.superframe_size as usize <= input.len() {
                let mut pos = 0usize;
                for _ in 0..info.frames_in_superframe {
                    let mut used = 0i32;
                    let r = Atrac9Decode(h, input.as_ptr().add(at + pos), pcm.as_mut_ptr(), &mut used);
                    if r != 0 {
                        println!("error {r:#x} at {} (superframe at {at}) after {frames} frame(s)", at + pos);
                        break 'sf;
                    }
                    frames += 1;
                    pos += used.max(0) as usize;
                }
                at += info.superframe_size as usize;
            }
            println!("{frames} frame(s) decoded of {}", (input.len() - start) / info.superframe_size.max(1) as usize);
            // Which byte offsets of the first 256 decode a first frame.
            let ok: Vec<usize> = (start..(start + 256).min(input.len())).filter(|&o| {
                let h2 = Atrac9GetHandle();
                let mut c2 = cfg;
                Atrac9InitDecoder(h2, c2.as_mut_ptr());
                let mut used = 0i32;
                let r = if o + info.superframe_size as usize <= input.len() { Atrac9Decode(h2, input.as_ptr().add(o), pcm.as_mut_ptr(), &mut used) } else { -1 };
                Atrac9ReleaseHandle(h2);
                r == 0
            }).collect();
            println!("offsets from {start} that decode a first frame: {:?}", &ok[..ok.len().min(24)]);
            Atrac9ReleaseHandle(h);
        }
    }

    /// The title's multi-stream configurations (16-, 9- and 4-channel
    /// ambisonic files) give mono streams of 96-byte frames, 4 a superframe.
    #[test]
    fn at9_multi_configs() {
        for (cfg, streams) in [([0x30, 0x73, 0xc1, 0x7e], 16), ([0x30, 0x72, 0x01, 0x7e], 9), ([0x30, 0x70, 0xc1, 0x7e], 4)] {
            assert_eq!(at9_multi_config(cfg), Some((streams, [0xfe, 0x70, 0x0b, 0xf0])));
        }
        assert_eq!(at9_multi_config([0xfe, 0x72, 0x17, 0xe0]), None);
    }

    /// Survey: every .at9 file under DS_AT9_DIR and every one embedded in a
    /// sound bank (.bnk) or an .xvag stream (after its header's padding)
    /// decoded as the title's jobs feed one (from its header, then what each
    /// job leaves, 1 MiB of PCM a job) through decode(); failures by result.
    /// `DS_AT9_DIR=<dir> dsport-<hash>.exe at9_survey --ignored --nocapture`
    #[test]
    #[ignore]
    fn at9_survey() {
        let dir = std::env::var("DS_AT9_DIR").expect("DS_AT9_DIR");
        let mut files = Vec::new();
        let mut stack = vec![std::path::PathBuf::from(dir)];
        while let Some(d) = stack.pop() {
            for e in std::fs::read_dir(&d).expect("directory").flatten() {
                let p = e.path();
                if p.is_dir() {
                    stack.push(p);
                } else if p.extension().is_some_and(|x| x == "at9" || x == "bnk" || x == "xvag") {
                    files.push(p);
                }
            }
        }
        files.sort();
        // Decoder setup writes libatrac9's shared tables: one at a time.
        let init = Mutex::new(());
        let next = std::sync::atomic::AtomicUsize::new(0);
        let superframes = std::sync::atomic::AtomicU64::new(0);
        let multi = std::sync::atomic::AtomicUsize::new(0);
        let streams = std::sync::atomic::AtomicUsize::new(0);
        let failed: Mutex<Vec<(u32, String, usize, u64)>> = Mutex::new(Vec::new());
        std::thread::scope(|s| {
            for _ in 0..std::thread::available_parallelism().map_or(4, |n| n.get()) {
                s.spawn(|| loop {
                    let i = next.fetch_add(1, Ordering::Relaxed);
                    let Some(path) = files.get(i) else { break };
                    let file = std::fs::read(path).expect("read");
                    let ext = path.extension().and_then(|x| x.to_str()).unwrap_or("");
                    let le32 = |b: &[u8], at: usize| u32::from_le_bytes([b[at], b[at + 1], b[at + 2], b[at + 3]]) as usize;
                    // The .at9 files in it: (start, end).
                    let mut parts: Vec<(usize, usize)> = Vec::new();
                    match ext {
                        "bnk" | "xvag" => {
                            let mut at = 0usize;
                            while let Some(k) = file[at..].windows(4).position(|w| w == b"RIFF") {
                                let r = at + k;
                                if r + 12 <= file.len() && &file[r + 8..r + 12] == b"WAVE" {
                                    parts.push((r, (r + 8 + le32(&file, r + 4)).min(file.len())));
                                }
                                at = r + 4;
                            }
                        }
                        _ => parts.push((0, file.len())),
                    }
                    for (start, end) in parts {
                    let mut input = file[start..end].to_vec();
                    // Zeroed slack after the input, as run_job leaves it.
                    let len = input.len();
                    input.resize(len + INPUT_SLACK, 0);
                    input.truncate(len);
                    let mut inst = Instance { codec: CODEC_AT9, parse_riff: true, encoding: 0, max_channels: 16, decoder: Decoder::None, total_decoded: 0, gapless_total: 0, gapless_skip: 0, skipped: 0, ops: Vec::new(), reported: false };
                    let mut out = Vec::with_capacity(1 << 20);
                    let (mut off, mut n) = (0usize, 0u64);
                    while off < len {
                        out.clear();
                        let o = {
                            let _g = (off == 0).then(|| init.lock());
                            unsafe { decode(&mut inst, &input[off..], 1 << 20, true, &mut out) }
                        };
                        if o.result & !RESULT_PARTIAL_INPUT != 0 {
                            failed.lock().push((o.result as u32, path.display().to_string(), start + off + o.consumed, n + o.frames as u64));
                            break;
                        }
                        n += o.frames as u64;
                        if o.consumed == 0 {
                            break;
                        }
                        off += o.consumed;
                    }
                    if matches!(inst.decoder, Decoder::At9Multi { .. }) {
                        multi.fetch_add(1, Ordering::Relaxed);
                    }
                    superframes.fetch_add(n, Ordering::Relaxed);
                    streams.fetch_add(1, Ordering::Relaxed);
                    }
                });
            }
        });
        let mut failed = failed.into_inner();
        failed.sort();
        println!("{} files, {} streams ({} multi-stream), {} superframes decoded, {} failed", files.len(), streams.load(Ordering::Relaxed), multi.load(Ordering::Relaxed), superframes.load(Ordering::Relaxed), failed.len());
        let mut codes: Vec<u32> = failed.iter().map(|f| f.0).collect();
        codes.dedup();
        for c in codes {
            let these: Vec<_> = failed.iter().filter(|f| f.0 == c).collect();
            println!("result {c:#x}: {} file(s)", these.len());
            for (_, p, at, n) in these.iter().take(6) {
                println!("  {p} at {at} after {n} superframe(s)");
            }
        }
    }
}
