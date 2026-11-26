//! The convolution reverb job of libSceAcm (`NCA::ACMConvolver1`), computed
//! on the CPU at its submission (FINDINGS.md 2026-10-23: the structures, from
//! the title's reverb module around eboot+0x1dfa640 / 0x1dfc350 and its data).
//!
//! The title accumulates 1024 samples of each reverb send (the convolver's
//! dry buffers) and submits one job per full block; during the next block it
//! plays the output buffers. The impulse response comes per output channel
//! as P partitions, each the packed 2048-point spectrum (`acm_fft`) of 1024
//! samples followed by 1024 zeros, in half floats, with a per-partition
//! byte size (shorter partitions keep only their low bins) and a gain per
//! partition (the title fades the tail with it). So the output is the
//! uniformly partitioned convolution (overlap-save) of the input with the
//! impulse response, times the job's gain. The DSP's own state buffers (a
//! spectrum history per input) are not used: the history is kept here in
//! f32; the title only clears them at a reset, which a marker word in the
//! first one shows.
//!
//! Layouts (offsets in bytes):
//! - convolver: u32 block, u32 partitions (its capacity: 248 where seen,
//!   impulse responses of up to that many), u32 inputs, u32 ?, u64 0,
//!   +0x18 list of {u64 state buffer, u64 0} per input, +0x20 list of dry
//!   buffers (block f32) per input.
//! - impulse response set: u32 block, u32 partitions (135 where seen),
//!   u32 channels, u32 ?,
//!   +0x10 partition gains (partitions f32), +0x18 list of {u64 spectra
//!   (partitions x 2048 f16), u64 sizes (partitions x 2 u16: bytes)} per
//!   channel.
//! - output: u32 block, u32 channels, +8 list of state buffers (block f32),
//!   +0x10 list of output buffers (4 + block f32: the block before's last 4
//!   samples, then the block).

use super::acm_fft::{C, Fft, N};
use parking_lot::Mutex;
use std::collections::HashMap;
use std::sync::atomic::{AtomicU64, Ordering};

const BLOCK: usize = N / 2;
/// Written into the first state buffer after each job: missing at the next
/// one when the title reset the reverb.
const MARKER: u64 = 0x5245_5642_4453_4143; // "CASDBVER"

struct State {
    parts: usize,
    channels: usize,
    /// Per input: the last block, and the spectra of the last `parts`
    /// frames (ring, newest at `pos`).
    prev: Vec<Vec<f32>>,
    hist: Vec<Vec<Vec<C>>>,
    pos: usize,
    /// Per input: whether the last block was all zeros, and per ring slot
    /// whether its frame was (its spectrum is zero: skipped, not stored).
    prev_zero: Vec<bool>,
    zero: Vec<Vec<bool>>,
    /// Per output channel: the last 4 samples written.
    last4: Vec<[f32; 4]>,
    /// The job count when last used (states of convolvers the title no
    /// longer submits are dropped).
    used: u64,
}

impl State {
    fn new(parts: usize, channels: usize, inputs: usize) -> State {
        State {
            parts,
            channels,
            prev: vec![vec![0.0; BLOCK]; inputs],
            hist: vec![vec![vec![C::default(); BLOCK]; parts]; inputs],
            pos: 0,
            prev_zero: vec![true; inputs],
            zero: vec![vec![true; parts]; inputs],
            last4: vec![[0.0; 4]; channels],
            used: 0,
        }
    }
}

static STATES: Mutex<Option<HashMap<u64, State>>> = Mutex::new(None);
static FFT: std::sync::OnceLock<Fft> = std::sync::OnceLock::new();
/// [jobs computed, jobs refused, resets seen, ns computing, jobs whose
/// outputs were silent without computing]
pub static STATS: [AtomicU64; 5] = [const { AtomicU64::new(0) }; 5];
/// Levels (diagnostics): dry sum of squares and samples, wet sum of squares
/// and samples, wet peak, non-finite wet samples replaced by 0.
static LEVELS: Mutex<[f64; 6]> = Mutex::new([0.0; 6]);

fn half_table() -> &'static [f32] {
    static T: std::sync::OnceLock<Vec<f32>> = std::sync::OnceLock::new();
    T.get_or_init(|| (0..=u16::MAX).map(half).collect())
}

fn half(h: u16) -> f32 {
    let s = if h >> 15 != 0 { -1.0 } else { 1.0 };
    let e = ((h >> 10) & 0x1F) as i32;
    let m = (h & 0x3FF) as f32;
    s * match e {
        0 => m * 2f32.powi(-24),
        31 => {
            if m == 0.0 {
                f32::INFINITY
            } else {
                f32::NAN
            }
        }
        _ => (1.0 + m / 1024.0) * 2f32.powi(e - 15),
    }
}

fn readable(a: u64, n: u64) -> bool {
    a >= 0x10000 && crate::platform::loader_hooks::guest_readable(a, n)
}

unsafe fn rd64(a: u64) -> u64 {
    unsafe { (a as *const u64).read_unaligned() }
}

unsafe fn rd32(a: u64) -> u32 {
    unsafe { (a as *const u32).read_unaligned() }
}

/// Computes one job: convolver `conv`, impulse response set `irs`, gain
/// `gain`, output `out`. Err: a structure this does not handle (the job is
/// left undone).
pub fn run(conv: u64, irs: u64, gain: f32, out: u64) -> Result<(), String> {
    let t0 = std::time::Instant::now();
    let r = unsafe { run_inner(conv, irs, gain, out) };
    STATS[if r.is_ok() { 0 } else { 1 }].fetch_add(1, Ordering::Relaxed);
    STATS[3].fetch_add(t0.elapsed().as_nanos() as u64, Ordering::Relaxed);
    r
}

unsafe fn run_inner(conv: u64, irs: u64, gain: f32, out: u64) -> Result<(), String> {
    if !readable(conv, 0x28) || !readable(irs, 0x20) || !readable(out, 0x18) {
        return Err("structures not readable".into());
    }
    let (block, parts, inputs) = unsafe { (rd32(conv) as usize, rd32(conv + 4) as usize, rd32(conv + 8) as usize) };
    let (iblock, iparts, channels) = unsafe { (rd32(irs) as usize, rd32(irs + 4) as usize, rd32(irs + 8) as usize) };
    let (oblock, ochannels) = unsafe { (rd32(out) as usize, rd32(out + 4) as usize) };
    if block != BLOCK || iblock != BLOCK || oblock != BLOCK {
        return Err(format!("block sizes {block}/{iblock}/{oblock}"));
    }
    if parts == 0 || parts > 4096 || iparts == 0 || iparts > parts || inputs == 0 || inputs > 8 || channels == 0 || channels > 16 || ochannels < channels {
        return Err(format!("partitions {parts}/{iparts}, inputs {inputs}, channels {channels}/{ochannels}"));
    }
    let (state_list, dry_list) = unsafe { (rd64(conv + 0x18), rd64(conv + 0x20)) };
    let (gains_at, ch_list) = unsafe { (rd64(irs + 0x10), rd64(irs + 0x18)) };
    let (_ostate_list, out_list) = unsafe { (rd64(out + 8), rd64(out + 0x10)) };
    if !readable(state_list, 16 * inputs as u64) || !readable(dry_list, 8 * inputs as u64) || !readable(gains_at, 4 * iparts as u64) || !readable(ch_list, 8 * channels as u64) || !readable(out_list, 8 * channels as u64) {
        return Err("lists not readable".into());
    }
    let pgain: &[f32] = unsafe { std::slice::from_raw_parts(gains_at as *const f32, iparts) };
    // Per channel: the impulse response (spectra, partition sizes), output.
    let mut ir = Vec::with_capacity(channels);
    let mut outs = Vec::with_capacity(channels);
    for c in 0..channels {
        let pair = unsafe { rd64(ch_list + 8 * c as u64) };
        if !readable(pair, 16) {
            return Err(format!("channel {c} not readable"));
        }
        let (spec_at, sizes_at) = unsafe { (rd64(pair), rd64(pair + 8)) };
        let ob = unsafe { rd64(out_list + 8 * c as u64) };
        if !readable(spec_at, 4096 * iparts as u64) || !readable(sizes_at, 4 * iparts as u64) || !readable(ob, 4 * (BLOCK + 4) as u64) {
            return Err(format!("channel {c}'s impulse response or output not readable"));
        }
        ir.push(unsafe { (std::slice::from_raw_parts(spec_at as *const u16, 2048 * iparts), std::slice::from_raw_parts(sizes_at as *const u16, 2 * iparts)) });
        outs.push(unsafe { std::slice::from_raw_parts_mut(ob as *mut f32, BLOCK + 4) });
    }
    let mut dry = Vec::with_capacity(inputs);
    for i in 0..inputs {
        let d = unsafe { rd64(dry_list + 8 * i as u64) };
        if !readable(d, 4 * BLOCK as u64) {
            return Err(format!("dry buffer {i} not readable"));
        }
        dry.push(unsafe { std::slice::from_raw_parts(d as *const f32, BLOCK) });
    }
    // The state buffer of input 0 (the title clears it at a reset).
    let marker_at = unsafe { rd64(rd64(state_list)) };
    let reset = !readable(marker_at, 8) || unsafe { rd64(marker_at) } != MARKER;
    // The convolver's state, out of the map while it is computed (the
    // title's job workers submit the convolvers' jobs concurrently).
    let job = STATS[0].load(Ordering::Relaxed) + STATS[1].load(Ordering::Relaxed);
    let mut st = {
        let mut g = STATES.lock();
        let states = g.get_or_insert_with(HashMap::new);
        if job % 1024 == 0 {
            states.retain(|_, s| s.used + 8192 > job);
        }
        states.remove(&conv)
    };
    match &st {
        Some(s) if s.parts == parts && s.channels == channels && s.prev.len() == inputs => {
            if reset {
                STATS[2].fetch_add(1, Ordering::Relaxed);
                st = Some(State::new(parts, channels, inputs));
            }
        }
        _ => {
            crate::info!("acm: convolver {conv:#x}: {inputs} input(s), {channels} channel(s), impulse responses of {iparts} of its {parts} partitions of {BLOCK}");
            st = Some(State::new(parts, channels, inputs));
        }
    }
    let mut st = st.unwrap();
    st.used = job;
    let mut lv = [0f64; 6];
    process(&mut st, &dry, &ir, pgain, gain, &mut outs, &mut lv);
    if readable(marker_at, 8) {
        unsafe { (marker_at as *mut u64).write_unaligned(MARKER) };
    }
    STATES.lock().get_or_insert_with(HashMap::new).insert(conv, st);
    let mut l = LEVELS.lock();
    for k in [0, 1, 2, 3, 5] {
        l[k] += lv[k];
    }
    l[4] = l[4].max(lv[4]);
    Ok(())
}

/// One block: the inputs' `dry` samples in, each channel's output buffer
/// (`out`: the last block's last 4 samples, then this block) from its
/// impulse response `ir` (packed half-float spectra of 2048 values per
/// partition, sizes in bytes per partition), partition gains `pgain` and
/// gain. `lv`: levels (`LEVELS`).
fn process(st: &mut State, dry: &[&[f32]], ir: &[(&[u16], &[u16])], pgain: &[f32], gain: f32, out: &mut [&mut [f32]], lv: &mut [f64; 6]) {
    let fft = FFT.get_or_init(Fft::new);
    let parts = st.parts;
    // The inputs' new frames.
    st.pos = (st.pos + 1) % parts;
    let mut frame = vec![0f32; N];
    for (i, cur) in dry.iter().enumerate() {
        let energy = cur.iter().map(|&v| v as f64 * v as f64).sum::<f64>();
        lv[0] += energy;
        lv[1] += BLOCK as f64;
        // (Reverb sends are often silent: a frame of zeros has a zero
        // spectrum, so its partitions' products are skipped.)
        let cur_zero = energy == 0.0 && cur.iter().all(|&v| v == 0.0);
        let pos = st.pos;
        st.zero[i][pos] = cur_zero && st.prev_zero[i];
        if !st.zero[i][pos] {
            frame[..BLOCK].copy_from_slice(&st.prev[i]);
            frame[BLOCK..].copy_from_slice(cur);
            fft.forward(&frame, &mut st.hist[i][pos]);
        }
        st.prev[i].copy_from_slice(cur);
        st.prev_zero[i] = cur_zero;
    }
    // Each channel: the sum over partitions of the history times the
    // impulse response's partition (partition by partition, every channel
    // from the same history spectrum).
    let mut acc = vec![vec![C::default(); BLOCK]; ir.len()];
    let mut summed = vec![false; ir.len()];
    for (p, &pg) in pgain.iter().enumerate() {
        let slot = (st.pos + parts - p) % parts;
        if pg == 0.0 || !pg.is_finite() || st.zero.iter().all(|z| z[slot]) {
            continue;
        }
        for (c, &(spec, sizes)) in ir.iter().enumerate() {
            let bins = ((sizes[2 * p] as usize) / 4).min(BLOCK);
            if bins == 0 {
                continue;
            }
            let h = &spec[2048 * p..2048 * p + 2 * bins];
            for (x, z) in st.hist.iter().zip(&st.zero) {
                if !z[slot] {
                    mac(&mut acc[c], &x[slot], h, pg);
                    summed[c] = true;
                }
            }
        }
    }
    if !summed.contains(&true) {
        STATS[4].fetch_add(1, Ordering::Relaxed);
    }
    let mut y = vec![0f32; N];
    let mut scratch = vec![C::default(); BLOCK];
    for (c, o) in out.iter_mut().enumerate() {
        if summed[c] {
            fft.inverse(&acc[c], &mut y, &mut scratch);
        } else {
            y.fill(0.0);
        }
        o[..4].copy_from_slice(&st.last4[c]);
        for k in 0..BLOCK {
            let v = y[BLOCK + k] * gain;
            if v.is_finite() {
                o[4 + k] = v;
                lv[2] += v as f64 * v as f64;
                lv[4] = lv[4].max(v.abs() as f64);
            } else {
                o[4 + k] = 0.0;
                lv[5] += 1.0;
            }
        }
        lv[3] += BLOCK as f64;
        st.last4[c].copy_from_slice(&o[BLOCK..BLOCK + 4]);
    }
}

/// acc += pg * x h over the bins of `h` (half floats, re and im), bin 0
/// holding two real products (DC, Nyquist).
fn mac(acc: &mut [C], x: &[C], h: &[u16], pg: f32) {
    #[cfg(target_arch = "x86_64")]
    {
        static AVX: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
        if *AVX.get_or_init(|| is_x86_feature_detected!("avx2") && is_x86_feature_detected!("fma") && is_x86_feature_detected!("f16c")) {
            return unsafe { mac_avx(acc, x, h, pg) };
        }
    }
    mac_scalar(acc, x, h, pg, 0)
}

/// `mac` from bin `from` on.
fn mac_scalar(acc: &mut [C], x: &[C], h: &[u16], pg: f32, from: usize) {
    let t = half_table();
    if from == 0 {
        acc[0].re += x[0].re * t[h[0] as usize] * pg;
        acc[0].im += x[0].im * t[h[1] as usize] * pg;
    }
    for k in from.max(1)..h.len() / 2 {
        let (hr, hi) = (t[h[2 * k] as usize] * pg, t[h[2 * k + 1] as usize] * pg);
        let a = x[k];
        acc[k].re += a.re * hr - a.im * hi;
        acc[k].im += a.re * hi + a.im * hr;
    }
}

/// `mac` four bins at a time (F16C conversion, FMA).
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2,fma,f16c")]
unsafe fn mac_avx(acc: &mut [C], x: &[C], h: &[u16], pg: f32) {
    use std::arch::x86_64::*;
    let bins = h.len() / 2;
    let n4 = bins / 4 * 4;
    let a0 = acc[0];
    let (ap, xp, hp) = (acc.as_mut_ptr() as *mut f32, x.as_ptr() as *const f32, h.as_ptr());
    let pgv = _mm256_set1_ps(pg);
    let mut k = 0;
    while k < n4 {
        unsafe {
            // Four complex bins: h as [re0 im0 re1 im1 ..], x likewise.
            let hv = _mm256_cvtph_ps(_mm_loadu_si128(hp.add(2 * k) as *const __m128i));
            let xv = _mm256_loadu_ps(xp.add(2 * k));
            let t = _mm256_mul_ps(_mm256_permute_ps::<0b10_11_00_01>(xv), _mm256_movehdup_ps(hv));
            // Even lanes x.re h.re - x.im h.im, odd lanes x.im h.re + x.re h.im.
            let prod = _mm256_fmaddsub_ps(xv, _mm256_moveldup_ps(hv), t);
            _mm256_storeu_ps(ap.add(2 * k), _mm256_fmadd_ps(prod, pgv, _mm256_loadu_ps(ap.add(2 * k))));
        }
        k += 4;
    }
    // Bin 0 again as its two real products, then the bins after the last four.
    let t = half_table();
    acc[0] = C { re: a0.re + x[0].re * t[h[0] as usize] * pg, im: a0.im + x[0].im * t[h[1] as usize] * pg };
    mac_scalar(acc, x, h, pg, n4.max(1));
}

/// One status line ("" before any job).
pub fn status() -> String {
    let n = |i: usize| STATS[i].load(Ordering::Relaxed);
    if n(0) + n(1) == 0 {
        return String::new();
    }
    let l = *LEVELS.lock();
    let rms = |s: f64, n: f64| (s / n.max(1.0)).sqrt();
    let jobs = format!(
        "    acm: {} convolution jobs computed ({} silent, {} refused, {} resets), {:.0} us each; dry rms {:.4}, wet rms {:.4} (peak {:.3}, {} non-finite replaced)\n",
        n(0),
        n(4),
        n(1),
        n(2),
        n(3) as f64 / 1e3 / (n(0) + n(1)).max(1) as f64,
        rms(l[0], l[1]),
        rms(l[2], l[3]),
        l[4],
        l[5]
    );
    super::acm::wait_line() + &jobs
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Nearest half float of `v` (normal range and zero; enough here).
    fn to_half(v: f32) -> u16 {
        if v == 0.0 {
            return 0;
        }
        let s = if v < 0.0 { 0x8000u16 } else { 0 };
        let a = v.abs();
        let e = a.log2().floor() as i32;
        if e < -14 {
            return s | ((a / 2f32.powi(-24)).round() as u16).min(0x3FF);
        }
        let m = ((a / 2f32.powi(e) - 1.0) * 1024.0).round() as u32;
        let (e, m) = if m == 1024 { (e + 1, 0) } else { (e, m) };
        s | (((e + 15) as u16) << 10) | m as u16
    }

    fn noise(n: usize, seed: u64) -> Vec<f32> {
        let mut x = seed;
        (0..n)
            .map(|_| {
                x ^= x << 13;
                x ^= x >> 7;
                x ^= x << 17;
                (x % 20001) as f32 / 10000.0 - 1.0
            })
            .collect()
    }

    /// Several blocks through `process` against the direct convolution of
    /// the same (half-float rounded) impulse response, partition gains and
    /// gain included: checks the overlap-save arrangement, the partitions'
    /// order in the history and the packed bin 0.
    #[test]
    fn matches_direct_convolution() {
        let (parts, iparts, channels, blocks) = (6usize, 4usize, 2usize, 16usize);
        let fft = Fft::new();
        let pgain = [1.0f32, 0.5, 0.0, 0.25];
        let gain = 0.75f32;
        let mut spectra = vec![vec![0u16; 2048 * iparts]; channels];
        let sizes = vec![0x1000u16; 2 * iparts];
        // The impulse responses as the half-float spectra give them back.
        let mut h = vec![vec![0f64; BLOCK * iparts]; channels];
        for c in 0..channels {
            let taps = noise(BLOCK * iparts, 7 + c as u64);
            for p in 0..iparts {
                let mut frame = vec![0f32; N];
                for k in 0..BLOCK {
                    frame[k] = taps[p * BLOCK + k] * 0.01;
                }
                let mut spec = vec![C::default(); BLOCK];
                fft.forward(&frame, &mut spec);
                for k in 0..BLOCK {
                    spectra[c][2048 * p + 2 * k] = to_half(spec[k].re);
                    spectra[c][2048 * p + 2 * k + 1] = to_half(spec[k].im);
                }
                let back: Vec<C> = (0..BLOCK).map(|k| C { re: half(spectra[c][2048 * p + 2 * k]), im: half(spectra[c][2048 * p + 2 * k + 1]) }).collect();
                let mut t = vec![0f32; N];
                let mut scratch = vec![C::default(); BLOCK];
                fft.inverse(&back, &mut t, &mut scratch);
                for k in 0..BLOCK {
                    h[c][p * BLOCK + k] = t[k] as f64 * pgain[p] as f64;
                }
            }
        }
        // Sound, a silent stretch longer than the response (its tail ends,
        // then whole jobs are silent), sound again.
        let mut input = noise(BLOCK * blocks, 3);
        input[3 * BLOCK + 100..10 * BLOCK].fill(0.0);
        let mut st = State::new(parts, channels, 1);
        let mut got = vec![Vec::new(); channels];
        for b in 0..blocks {
            let dry = [&input[b * BLOCK..(b + 1) * BLOCK]];
            let ir: Vec<(&[u16], &[u16])> = spectra.iter().map(|s| (&s[..], &sizes[..])).collect();
            let mut bufs = vec![vec![0f32; BLOCK + 4]; channels];
            let mut outs: Vec<&mut [f32]> = bufs.iter_mut().map(|v| &mut v[..]).collect();
            let mut lv = [0f64; 6];
            process(&mut st, &dry, &ir, &pgain, gain, &mut outs, &mut lv);
            for c in 0..channels {
                if b > 0 {
                    // The last block's last 4 samples come first.
                    assert_eq!(&bufs[c][..4], &got[c][got[c].len() - 4..]);
                }
                got[c].extend_from_slice(&bufs[c][4..]);
            }
        }
        for c in 0..channels {
            let mut worst = 0f64;
            for n in 0..input.len() {
                let mut y = 0f64;
                for (k, &hk) in h[c].iter().enumerate().take(n + 1) {
                    y += hk * input[n - k] as f64;
                }
                worst = worst.max((got[c][n] as f64 - gain as f64 * y).abs());
            }
            assert!(worst < 1e-3, "channel {c}: worst difference {worst}");
        }
    }

    /// Cost of one job at the gate's largest shape (4 channels, 183 of 248
    /// partitions): `cargo test --release bench_process -- --ignored --nocapture`.
    #[test]
    #[ignore]
    fn bench_process() {
        let (parts, iparts, channels) = (248usize, 183usize, 4usize);
        let spectra: Vec<Vec<u16>> = (0..channels).map(|c| noise(2048 * iparts, 11 + c as u64).iter().map(|&v| to_half(v * 0.01)).collect()).collect();
        let sizes = vec![0x1000u16; 2 * iparts];
        let pgain = vec![1.0f32; iparts];
        let input = noise(BLOCK, 4);
        let mut st = State::new(parts, channels, 1);
        let ir: Vec<(&[u16], &[u16])> = spectra.iter().map(|s| (&s[..], &sizes[..])).collect();
        let mut bufs = vec![vec![0f32; BLOCK + 4]; channels];
        let mut lv = [0f64; 6];
        let fft = Fft::new();
        let t = std::time::Instant::now();
        let mut spec = vec![C::default(); BLOCK];
        let mut y = vec![0f32; N];
        let mut scratch = vec![C::default(); BLOCK];
        let frame = noise(N, 6);
        for _ in 0..1000 {
            fft.forward(&frame, &mut spec);
            fft.inverse(&spec, &mut y, &mut scratch);
        }
        assert!(y[0].is_finite());
        println!("forward + inverse FFT: {:.1} us", t.elapsed().as_secs_f64() * 1e6 / 1000.0);
        let t = std::time::Instant::now();
        for _ in 0..200 {
            let mut outs: Vec<&mut [f32]> = bufs.iter_mut().map(|v| &mut v[..]).collect();
            process(&mut st, &[&input], &ir, &pgain, 1.0, &mut outs, &mut lv);
        }
        println!("process ({channels} channels, {iparts} partitions): {:.0} us per job", t.elapsed().as_secs_f64() * 1e6 / 200.0);
    }

    #[test]
    fn mac_paths_agree() {
        let x: Vec<C> = noise(2 * BLOCK, 5).chunks(2).map(|p| C { re: p[0], im: p[1] }).collect();
        let h: Vec<u16> = noise(2 * BLOCK, 9).iter().map(|&v| to_half(v)).collect();
        for bins in [1usize, 3, 4, 5, 511, 1024] {
            let mut a = vec![C { re: 0.5, im: -0.25 }; BLOCK];
            let mut b = a.clone();
            mac(&mut a, &x, &h[..2 * bins], 0.7);
            mac_scalar(&mut b, &x, &h[..2 * bins], 0.7, 0);
            for k in 0..BLOCK {
                assert!((a[k].re - b[k].re).abs() < 1e-5 && (a[k].im - b[k].im).abs() < 1e-5, "bins {bins}, bin {k}: {:?} {:?}", a[k], b[k]);
            }
        }
    }
}
