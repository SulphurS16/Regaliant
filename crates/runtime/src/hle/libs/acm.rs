//! libSceAcm: the PS5 audio compute manager (DSP jobs run on the audio
//! coprocessor). The title's audio engine uses it for one DSP module,
//! `NCA::ACMConvolver1` (a partitioned convolution reverb), and requires the
//! context to exist (its engine init fails otherwise, eboot+0x1ddea9b).
//!
//! Batch buffers are descriptors {u64 data, u64 used, u64 capacity} holding
//! records that start with a u64 opcode. The title writes one record kind
//! itself: opcode 0x0B {u64 0x0B, u64 status*}, a status/statistics request
//! whose u32 at status+8 it reads back as the batch's processor load
//! (value * 100 / 2^31 percent, eboot+0x1de477d). Job records from the
//! builder below use this runtime's own private opcode (the builder is a
//! library function, so the encoding is ours).
//!
//! Batches are consumed at submission: the title builds its per-tick batch in
//! a stack buffer and returns right after `sceAcmBatchStartBuffers`, so the
//! records cannot be read later. Waits therefore always find the batch done.
//! Convolution jobs are computed then, on the submitting thread
//! (`acm_conv`; `tune acm_reverb 0` / `DS_ACM_REVERB=0`: left undone, the
//! dry stand-in of before).

use crate::hle::Registry;
use crate::{reg, sysv};
use std::sync::atomic::{AtomicU32, AtomicU64, Ordering};

const ERR_INVALID_PARAMETER: i32 = 0x8194_0002u32 as i32;
const ERR_BUFFER_FULL: i32 = 0x8194_0004u32 as i32;

/// Title-written status record (see module docs).
const OP_STATUS: u64 = 0x0B;
/// Private opcode of the convolution job records written by the builder:
/// {u64 OP_DS_CONVOLVE, u64 record bytes, 7 x u64 builder arguments}.
const OP_DS_CONVOLVE: u64 = 0x4453_4143_0000_0001; // "DSAC" | 1
const CONVOLVE_RECORD: u64 = 9 * 8;

static NEXT_BATCH: AtomicU32 = AtomicU32::new(1);
/// Diagnostics: when the recent batches were started (ns, by id modulo 64)
/// and the waits for them: [waits, polls (timeout 0), ns from the start to
/// the wait summed, the longest].
static STARTED: [AtomicU64; 64] = [const { AtomicU64::new(0) }; 64];
static WAITS: [AtomicU64; 4] = [const { AtomicU64::new(0) }; 4];

fn now_ns() -> u64 {
    static EPOCH: std::sync::OnceLock<std::time::Instant> = std::sync::OnceLock::new();
    EPOCH.get_or_init(std::time::Instant::now).elapsed().as_nanos() as u64 + 1
}

/// How the title waits for its batches (profile report; "" before any wait).
pub fn wait_line() -> String {
    let w = |i: usize| WAITS[i].load(Ordering::Relaxed);
    if w(0) == 0 {
        return String::new();
    }
    format!("    acm: {} batch waits ({} polls), {:.0} us from the start on average, at most {:.0} us\n", w(0), w(1), w(2) as f64 / 1e3 / w(0) as f64, w(3) as f64 / 1e3)
}
static JOBS: AtomicU64 = AtomicU64::new(0);
static DUMPED: AtomicU32 = AtomicU32::new(0);

fn hexdump(addr: u64, len: usize) -> String {
    if addr < 0x10000 {
        return format!("{addr:#x} (not a pointer)");
    }
    let mut s = format!("{addr:#x}:");
    for i in (0..len).step_by(8) {
        if i % 32 == 0 {
            s.push_str("\n          ");
        }
        let v = unsafe { ((addr + i as u64) as *const u64).read_unaligned() };
        s.push_str(&format!(" {v:016x}"));
    }
    s
}

/// Logs the builder arguments and the structures they point to (first jobs only).
unsafe fn dump_job(args: &[u64; 7]) {
    if DUMPED.fetch_add(1, Ordering::Relaxed) >= 2 {
        return;
    }
    let [batch, n_conv, conv, n_in, inputs, gains, outputs] = *args;
    let mut s = format!("acm: convolution job builder u70oWo92SYQ(batch {batch:#x}, {n_conv}, {conv:#x}, {n_in}, {inputs:#x}, {gains:#x}, {outputs:#x})");
    s.push_str(&format!("\n    convolver[0] {}", hexdump(conv, 0x60)));
    unsafe {
        if inputs >= 0x10000 {
            let p = *(inputs as *const u64);
            s.push_str(&format!("\n    input[0] -> {}", hexdump(p, 0x40)));
        }
        if gains >= 0x10000 {
            s.push_str(&format!("\n    gain[0] = {}", *(gains as *const f32)));
        }
        if outputs >= 0x10000 {
            let p = *(outputs as *const u64);
            s.push_str(&format!("\n    output[0] -> {}", hexdump(p, 0x40)));
        }
    }
    crate::info!("{s}");
}

/// Diagnosis (`DS_ACM_DUMP=1`): the buffers a convolution job's structures
/// point to, at a few jobs (statistics as f32, f16 and i16 words), to learn
/// which ones the title fills and in what format.
fn dump_buffers(args: &[u64; 7], job: u64) {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    if !*ON.get_or_init(|| std::env::var("DS_ACM_DUMP").ok().as_deref() == Some("1")) {
        return;
    }
    if !matches!(job, 50 | 2000 | 6000 | 9000) {
        return;
    }
    let readable = |a: u64, n: u64| a >= 0x10000 && crate::platform::loader_hooks::guest_readable(a, n);
    let stats = |a: u64, bytes: u64| -> String {
        if !readable(a, bytes) {
            return format!("{a:#x} (not readable)");
        }
        let n32 = (bytes / 4) as usize;
        let f: &[f32] = unsafe { std::slice::from_raw_parts(a as *const f32, n32) };
        let w: &[u16] = unsafe { std::slice::from_raw_parts(a as *const u16, (bytes / 2) as usize) };
        let nz32 = f.iter().filter(|v| v.to_bits() != 0).count();
        let fin: Vec<f32> = f.iter().copied().filter(|v| v.is_finite()).collect();
        let max32 = fin.iter().fold(0f32, |m, v| m.max(v.abs()));
        let rms32 = (fin.iter().map(|v| (v * v) as f64).sum::<f64>() / fin.len().max(1) as f64).sqrt();
        let h = |x: u16| -> f32 { half(x) };
        let maxh = w.iter().map(|&x| h(x)).filter(|v| v.is_finite()).fold(0f32, |m, v| m.max(v.abs()));
        let maxi = w.iter().map(|&x| (x as i16).unsigned_abs()).max().unwrap_or(0);
        let head: Vec<String> = f.iter().take(6).map(|v| format!("{v:.4e}")).collect();
        let headw: Vec<String> = w.iter().take(8).map(|x| format!("{x:04x}")).collect();
        format!("{a:#x}+{bytes:#x}: nonzero dwords {nz32}/{n32}, f32 max {max32:.4e} rms {rms32:.3e} [{}], as f16 max {maxh:.4e}, as i16 max {maxi} [{}]", head.join(" "), headw.join(" "))
    };
    let rd = |a: u64| -> u64 { if readable(a, 8) { unsafe { *(a as *const u64) } } else { 0 } };
    let rd32 = |a: u64| -> u32 { if readable(a, 4) { unsafe { *(a as *const u32) } } else { 0 } };
    let [_, _, conv, _, inputs, gains, outputs] = *args;
    let mut s = format!("acm dump at job {job}:");
    // The convolver: {block, partitions, n_ir, ?, 0, ptr ir_list, ptr out_list}.
    let (block, parts, n_ir) = (rd32(conv) as u64, rd32(conv + 4) as u64, rd32(conv + 8) as u64);
    s.push_str(&format!("\n  convolver {conv:#x}: block {block}, partitions {parts}, irs {n_ir}, word3 {}", rd32(conv + 12)));
    let irs = rd(conv + 0x18);
    for k in 0..n_ir.min(2) {
        let pair = rd(irs + 8 * k);
        let ir = rd(pair);
        s.push_str(&format!("\n    ir[{k}] (pair {pair:#x}, second word {:#x}): {}", rd(pair + 8), stats(ir, 2 * block * parts * 2)));
        s.push_str(&format!("\n      first partition: {}", stats(ir, 4 * block)));
    }
    let outs = rd(conv + 0x20);
    for k in 0..n_ir.min(2) {
        s.push_str(&format!("\n    out[{k}]: {}", stats(rd(outs + 8 * k), 4 * block)));
    }
    // The input descriptor: {block, partitions, channels, ?, ptr, ptr list}.
    let ind = rd(inputs);
    let (ib, ip, ich) = (rd32(ind) as u64, rd32(ind + 4) as u64, rd32(ind + 8) as u64);
    s.push_str(&format!("\n  input {ind:#x}: block {ib}, partitions {ip}, channels {ich}, word3 {}, gain {}", rd32(ind + 12), if readable(gains, 4) { unsafe { *(gains as *const f32) } } else { f32::NAN }));
    s.push_str(&format!("\n    buffer: {}", stats(rd(ind + 16), 4 * ip.max(1))));
    let chl = rd(ind + 24);
    for k in 0..ich.min(2) {
        let c = rd(chl + 8 * k);
        s.push_str(&format!("\n    channel[{k}] {c:#x}: a {}", stats(rd(c), 4 * ib * ip)));
        s.push_str(&format!("\n      b {}", stats(rd(c + 8), 4 * ip)));
        // Partitions as packed spectra of 2048-point frames, transformed
        // back: where their energy lies in the frame.
        let ir = rd(c);
        if readable(ir, 4 * ib * ip) && ib == 1024 {
            let fft = super::acm_fft::Fft::new();
            let mut spec = vec![super::acm_fft::C::default(); 1024];
            let mut t = vec![0f32; 2048];
            let mut scratch = vec![super::acm_fft::C::default(); 1024];
            for p in [0u64, 1, ip / 4, ip / 2, ip.saturating_sub(1)] {
                let w: &[u16] = unsafe { std::slice::from_raw_parts((ir + 4096 * p) as *const u16, 2048) };
                for (j, v) in spec.iter_mut().enumerate() {
                    *v = super::acm_fft::C { re: half(w[2 * j]), im: half(w[2 * j + 1]) };
                }
                fft.inverse(&spec, &mut t, &mut scratch);
                let e1: f64 = t[..1024].iter().map(|v| (v * v) as f64).sum();
                let e2: f64 = t[1024..].iter().map(|v| (v * v) as f64).sum();
                let (pk, pv) = t.iter().enumerate().fold((0, 0f32), |m, (i, v)| if v.abs() > m.1.abs() { (i, *v) } else { m });
                s.push_str(&format!("\n      partition {p}: energy first half {e1:.3e}, second half {e2:.3e}, peak {pv:.4} at {pk}, head [{:.4} {:.4} {:.4} {:.4}]", t[0], t[1], t[2], t[3]));
            }
        }
    }
    // The output descriptor: {block, count, ptr list a, ptr list b}.
    let od = rd(outputs);
    let (ob, oc) = (rd32(od) as u64, rd32(od + 4) as u64);
    s.push_str(&format!("\n  output {od:#x}: block {ob}, count {oc}"));
    for k in 0..oc.min(2) {
        s.push_str(&format!("\n    a[{k}]: {}", stats(rd(rd(od + 8) + 8 * k), 4 * ob)));
        s.push_str(&format!("\n    b[{k}]: {}", stats(rd(rd(od + 16) + 8 * k), 4 * (ob + 4))));
    }
    crate::info!("{s}");
}

/// IEEE half to f32.
fn half(h: u16) -> f32 {
    let s = ((h >> 15) & 1) as u32;
    let e = ((h >> 10) & 0x1F) as i32;
    let m = (h & 0x3FF) as u32;
    let v = if e == 0 {
        (m as f32) * 2f32.powi(-24)
    } else if e == 31 {
        if m == 0 { f32::INFINITY } else { f32::NAN }
    } else {
        (1.0 + m as f32 / 1024.0) * 2f32.powi(e - 15)
    };
    if s == 1 { -v } else { v }
}

/// The reverb computed (`tune acm_reverb`, default on).
fn reverb_on() -> bool {
    crate::platform::loader_hooks::acm_reverb()
}

/// Computes a convolution job (builder arguments after the batch: convolver
/// count, convolvers, input count, impulse response set pointers, gains,
/// output pointers). One convolver and one set per job is what the title
/// builds; others are reported once and left undone.
fn convolve(_job: u64, a: [u64; 6]) {
    let [n_conv, conv, n_in, inputs, gains, outputs] = a;
    if !reverb_on() {
        static OFF: std::sync::atomic::AtomicBool = std::sync::atomic::AtomicBool::new(false);
        if !OFF.swap(true, Ordering::Relaxed) {
            crate::warn!("acm: convolution jobs left undone (`tune acm_reverb 0`: no reverb)");
        }
        return;
    }
    static WARNED: AtomicU32 = AtomicU32::new(0);
    if n_conv != 1 || n_in != 1 || inputs < 0x10000 || gains < 0x10000 || outputs < 0x10000 {
        if WARNED.fetch_add(1, Ordering::Relaxed) == 0 {
            crate::warn!("acm: convolution job with {n_conv} convolvers and {n_in} inputs not computed (one of each handled)");
        }
        return;
    }
    let (irs, gain, out) = unsafe { (*(inputs as *const u64), *(gains as *const f32), *(outputs as *const u64)) };
    if let Err(e) = super::acm_conv::run(conv, irs, gain, out) {
        if WARNED.fetch_add(1, Ordering::Relaxed) < 4 {
            crate::warn!("acm: convolution job not computed: {e}");
        }
    }
}

/// Executes the records of one batch buffer descriptor.
unsafe fn run_buffer(desc: *const u64) -> Result<(), String> {
    let (data, used) = unsafe { (*desc, *desc.add(1)) };
    let mut off = 0u64;
    while off < used {
        let rec = data + off;
        let op = unsafe { *(rec as *const u64) };
        match op {
            OP_STATUS => {
                // status+8: processor load of the batch (no coprocessor time is spent).
                let status = unsafe { *((rec + 8) as *const u64) };
                if status >= 0x10000 {
                    unsafe { *((status + 8) as *mut u32) = 0 };
                }
                off += 16;
            }
            OP_DS_CONVOLVE => {
                let size = unsafe { *((rec + 8) as *const u64) };
                if size < 16 {
                    return Err(format!("bad record size {size:#x} at +{off:#x}"));
                }
                let job = JOBS.fetch_add(1, Ordering::Relaxed);
                let w = |i: u64| unsafe { *((rec + 8 * i) as *const u64) };
                dump_buffers(&[0, w(2), w(3), w(4), w(5), w(6), w(7)], job);
                convolve(job, [w(2), w(3), w(4), w(5), w(6), w(7)]);
                off += size;
            }
            other => return Err(format!("unknown record opcode {other:#x} at +{off:#x} (buffer {data:#x}, {used:#x} bytes)")),
        }
    }
    Ok(())
}

sysv! {
    fn sceAcmContextCreate(out: *mut u32) -> i32 {
        if out.is_null() { return ERR_INVALID_PARAMETER; }
        *out = 1;
        crate::info!("sceAcmContextCreate -> 1");
        0
    }
    fn sceAcmContextDestroy(ctx: u32) -> i32 { 0 }
    /// (ctx, buffer count, buffer descriptor pointers, flags, out batch id)
    fn sceAcmBatchStartBuffers(ctx: u32, count: u32, descs: *const u64, flags: u32, out_id: *mut u32) -> i32 {
        if descs.is_null() { return ERR_INVALID_PARAMETER; }
        for i in 0..count as usize {
            let d = *descs.add(i) as *const u64;
            if d.is_null() { return ERR_INVALID_PARAMETER; }
            if let Err(e) = run_buffer(d) {
                crate::error!("acm: batch buffer {i}: {e}");
                return ERR_INVALID_PARAMETER;
            }
        }
        let id = NEXT_BATCH.fetch_add(1, Ordering::Relaxed);
        STARTED[id as usize % 64].store(now_ns(), Ordering::Relaxed);
        if !out_id.is_null() { *out_id = id; }
        0
    }
    /// (ctx, batch id, timeout in microseconds; 0 polls, !0 waits forever).
    /// Batches complete at submission, so waits always succeed.
    fn sceAcmBatchWait(ctx: u32, batch: u32, timeout: u32) -> i32 {
        let started = STARTED[batch as usize % 64].load(Ordering::Relaxed);
        if started != 0 {
            let gap = now_ns().saturating_sub(started);
            WAITS[0].fetch_add(1, Ordering::Relaxed);
            WAITS[1].fetch_add((timeout == 0) as u64, Ordering::Relaxed);
            WAITS[2].fetch_add(gap, Ordering::Relaxed);
            WAITS[3].fetch_max(gap, Ordering::Relaxed);
        }
        0
    }
    /// Unnamed libSceAcm export: appends one convolution job.
    /// (batch descriptor, convolver count, convolvers, input count, input
    /// pointers, input gains, output pointers) -- argument roles inferred from
    /// the call at eboot+0x1dfcfa1.
    fn ds_acm_batch_convolution(batch: *mut u64, n_conv: u64, conv: u64, n_in: u64, inputs: u64, gains: u64, outputs: u64) -> i32 {
        if batch.is_null() { return ERR_INVALID_PARAMETER; }
        let args = [batch as u64, n_conv, conv, n_in, inputs, gains, outputs];
        dump_job(&args);
        let (data, used, cap) = (*batch, *batch.add(1), *batch.add(2));
        if used + CONVOLVE_RECORD > cap { return ERR_BUFFER_FULL; }
        let rec = (data + used) as *mut u64;
        *rec = OP_DS_CONVOLVE;
        *rec.add(1) = CONVOLVE_RECORD;
        for (i, a) in args.iter().enumerate().skip(1) {
            *rec.add(1 + i) = *a;
        }
        *batch.add(1) = used + CONVOLVE_RECORD;
        0
    }
}

pub fn register(r: &mut Registry) {
    reg!(r, "libSceAcm", [sceAcmContextCreate, sceAcmContextDestroy, sceAcmBatchStartBuffers, sceAcmBatchWait]);
    r.func_nid("libSceAcm", "u70oWo92SYQ", "u70oWo92SYQ (unnamed: convolution job builder)", ds_acm_batch_convolution as *const () as u64);
}
