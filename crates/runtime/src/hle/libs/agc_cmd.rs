//! libSceAgc command-buffer writers: `sceAgcDcb*` (graphics), `sceAgcAcb*`
//! (asynchronous compute), `sceAgcCb*` (either queue), their `*GetSize`
//! companions and the `*Patch*` helpers that edit written packets.
//!
//! Packets are genuine gfx10 PM4 type-3 packets wherever a hardware packet
//! expresses the operation (register writes, draws, dispatches, events,
//! RELEASE_MEM/ACQUIRE_MEM, WAIT_REG_MEM, WRITE_DATA, COPY_DATA, DMA_DATA,
//! COND_EXEC, INDIRECT_BUFFER, REWIND, ...): the title also writes PM4 inline
//! and the command processor (`crate::gpu::cp`) parses one format. Driver
//! level operations without a hardware packet (flips, workload tracking,
//! debug markers, the conditional branch) are PM4 NOPs carrying a private
//! payload: `[NOP header, DS_MAGIC, op, args...]` (see `private`).
//!
//! Register "indirect" writes (`Set*RegistersIndirect`) reference an array of
//! ShaderRegister {u32 offset, u32 value} pairs that the command processor
//! reads when it executes the packet (LOAD_*_REG-style opcodes, pair format).
//!
//! `*GetSize` functions return sizes in bytes (signatures and units per
//! public sources, see docs/PROVENANCE.md); every writer here emits exactly that
//! many bytes, and returns the address of its first packet (0 on failure).

use super::agc::{alloc_dw, pm4_hdr};
use crate::hle::{Registry, cstr};
use crate::{reg, sysv};

// PM4 type-3 opcodes (gfx10; AMD's open-source headers, see docs/PROVENANCE.md).
pub mod op {
    pub const NOP: u32 = 0x10;
    pub const SET_BASE: u32 = 0x11;
    pub const CLEAR_STATE: u32 = 0x12;
    pub const INDEX_BUFFER_SIZE: u32 = 0x13;
    pub const DISPATCH_DIRECT: u32 = 0x15;
    pub const DISPATCH_INDIRECT: u32 = 0x16;
    pub const SET_PREDICATION: u32 = 0x20;
    pub const COND_EXEC: u32 = 0x22;
    pub const DRAW_INDIRECT: u32 = 0x24;
    pub const DRAW_INDEX_INDIRECT: u32 = 0x25;
    pub const INDEX_BASE: u32 = 0x26;
    pub const DRAW_INDEX_2: u32 = 0x27;
    pub const CONTEXT_CONTROL: u32 = 0x28;
    pub const INDEX_TYPE: u32 = 0x2A;
    pub const DRAW_INDIRECT_MULTI: u32 = 0x2C;
    pub const DRAW_INDEX_AUTO: u32 = 0x2D;
    pub const NUM_INSTANCES: u32 = 0x2F;
    pub const DRAW_INDEX_MULTI_AUTO: u32 = 0x30;
    pub const DRAW_INDEX_OFFSET_2: u32 = 0x35;
    pub const WRITE_DATA: u32 = 0x37;
    pub const DRAW_INDEX_INDIRECT_MULTI: u32 = 0x38;
    pub const WAIT_REG_MEM: u32 = 0x3C;
    pub const INDIRECT_BUFFER: u32 = 0x3F;
    pub const COPY_DATA: u32 = 0x40;
    pub const PFP_SYNC_ME: u32 = 0x42;
    pub const COND_WRITE: u32 = 0x45;
    pub const EVENT_WRITE: u32 = 0x46;
    pub const RELEASE_MEM: u32 = 0x49;
    pub const DMA_DATA: u32 = 0x50;
    pub const ACQUIRE_MEM: u32 = 0x58;
    pub const REWIND: u32 = 0x59;
    pub const LOAD_UCONFIG_REG: u32 = 0x5E;
    pub const LOAD_SH_REG: u32 = 0x5F;
    pub const LOAD_CONTEXT_REG: u32 = 0x61;
    pub const LOAD_SH_REG_INDEX: u32 = 0x63;
    pub const SET_CONFIG_REG: u32 = 0x68;
    pub const SET_CONTEXT_REG: u32 = 0x69;
    pub const SET_CONTEXT_REG_INDEX: u32 = 0x6A;
    pub const SET_SH_REG: u32 = 0x76;
    pub const SET_SH_REG_OFFSET: u32 = 0x77;
    pub const SET_UCONFIG_REG: u32 = 0x79;
    pub const SET_UCONFIG_REG_INDEX: u32 = 0x7A;
    pub const WAIT_REG_MEM64: u32 = 0x93;
    pub const SET_SH_REG_INDEX: u32 = 0x9B;
    pub const LOAD_CONTEXT_REG_INDEX: u32 = 0x9F;
}

/// VGT_INDEX_TYPE (uconfig 0x3090C) as a uconfig dword index.
pub const REG_VGT_INDEX_TYPE: u32 = 0x243;

/// First payload dword of private NOP packets ("DSHP").
pub const DS_MAGIC: u32 = 0x5048_5344;

/// Private NOP payload operations (`[hdr, DS_MAGIC, op, args...]`).
pub mod pop {
    pub const FLIP: u32 = 1; // handle, index, mode, arg lo, arg hi
    pub const WAIT_SAFE_FOR_RENDERING: u32 = 2; // handle, index
    pub const WORKLOADS_ACTIVE: u32 = 3; // stream, count, ids...
    pub const WORKLOAD_COMPLETE: u32 = 4; // stream, id
    pub const PUSH_MARKER: u32 = 5; // color, chars...
    pub const SET_MARKER: u32 = 6; // color, chars...
    pub const POP_MARKER: u32 = 7;
    pub const BRANCH: u32 = 8; // mode|func<<8, cmp lo, hi, mask lo, hi, ref lo, hi, b1 lo, hi, size1|cp<<28, b2 lo, hi, size2|cp<<28
    pub const DRAW_INDEX_MULTI_INSTANCED: u32 = 9; // count, addr lo, hi, ids lo, hi, instances, initiator
    pub const LOD_STATS: u32 = 10; // buffer lo, hi, size, flags
    pub const ACB_RESET: u32 = 11; // op
    pub const CONTEXT_STATE: u32 = 12; // operation
}

const ACB: bool = true;
const DCB: bool = false;

fn hdr(opcode: u32, payload: u32, compute: bool) -> u32 {
    pm4_hdr(opcode, payload, compute)
}

/// Writes `dw` at the command buffer's cursor; returns its address or 0.
/// `DS_EMIT_CENSUS=1` (diagnosis): packets the title wrote through `emit`, and the direct
/// dispatches among them (`sceAgcCbDispatch`), for the status report (`emit_status`). Off by
/// default: the counters are shared by every thread of the title that writes packets.
static EMITS: [std::sync::atomic::AtomicU64; 2] = [const { std::sync::atomic::AtomicU64::new(0) }; 2];

fn emit_census() -> bool {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ON.get_or_init(|| std::env::var("DS_EMIT_CENSUS").ok().as_deref() == Some("1"))
}

/// For the status report: how much of the title's stream is written through these functions.
pub fn emit_status() -> String {
    if !emit_census() {
        return String::new();
    }
    format!(
        "    agc: {} packets written by the title through the command functions' `emit`, {} of them direct dispatches (`sceAgcCbDispatch`)
",
        EMITS[0].load(std::sync::atomic::Ordering::Relaxed),
        EMITS[1].load(std::sync::atomic::Ordering::Relaxed)
    )
}

fn emit(cb: u64, dw: &[u32]) -> u64 {
    if emit_census() {
        EMITS[0].fetch_add(1, std::sync::atomic::Ordering::Relaxed);
    }
    let p = alloc_dw(cb, dw.len() as u32);
    if p != 0 {
        unsafe { std::ptr::copy_nonoverlapping(dw.as_ptr(), p as *mut u32, dw.len()) };
    }
    p
}

fn private(cb: u64, compute: bool, operation: u32, args: &[u32]) -> u64 {
    let mut v = Vec::with_capacity(3 + args.len());
    v.push(hdr(op::NOP, 2 + args.len() as u32, compute));
    v.push(DS_MAGIC);
    v.push(operation);
    v.extend_from_slice(args);
    emit(cb, &v)
}

fn lo(v: u64) -> u32 {
    v as u32
}
fn hi(v: u64) -> u32 {
    (v >> 32) as u32
}

/// ShaderRegister passed by value: {u32 offset (dword index in its space), u32 value}.
fn split_reg(r: u64) -> (u32, u32) {
    (r as u32, (r >> 32) as u32)
}

/// Draw initiator from an Agc draw modifier (bit 8 -> USE_OPAQUE; bit 32
/// selects a variant without it). Source select is added by each draw.
fn draw_initiator(modifier: u64, source_select: u32) -> u32 {
    // Each distinct modifier once (its layout is not fully known).
    static SEEN: parking_lot::Mutex<Option<std::collections::HashSet<(u64, u32)>>> = parking_lot::Mutex::new(None);
    if SEEN.lock().get_or_insert_with(Default::default).insert((modifier, source_select)) {
        crate::info!("agc: draw modifier {modifier:#018x} (source select {source_select})");
    }
    let extra = if modifier & (1 << 32) != 0 { 0 } else { ((modifier as u32) >> 3) & 0x20 };
    source_select | extra
}

/// Dispatch initiator: COMPUTE_SHADER_EN | FORCE_START_AT_000 | Agc flag bits
/// (ORDER_MODE, USE_THREAD_DIMENSIONS, ...).
fn dispatch_initiator(modifier: u32) -> u32 {
    (modifier & 0xA038) | 0x41
}

fn marker_payload(color: u32, s: &str) -> Vec<u32> {
    let mut bytes = s.as_bytes().to_vec();
    bytes.push(0);
    while bytes.len() % 4 != 0 {
        bytes.push(0);
    }
    let mut v = vec![color];
    v.extend(bytes.chunks(4).map(|c| u32::from_le_bytes([c[0], c[1], c[2], c[3]])));
    v
}

fn marker(cb: u64, compute: bool, operation: u32, s: *const u8, color: u32) -> u64 {
    let text: String = if s.is_null() { String::new() } else { unsafe { cstr(s) }.chars().take(255).collect() };
    if text == "BinkGPU" && std::env::var("DS_TRACE_SUBMIT").ok().as_deref() == Some("1") {
        // The cursor relative to the buffer start: the title resets it just
        // before pushing this marker.
        let rd = |o: u64| unsafe { std::ptr::read_volatile((cb + o) as *const u64) };
        let (bottom, top, up, down) = (rd(0), rd(8), rd(0x10), rd(0x18));
        crate::info!(
            "bink build: cb {cb:#x} (start {bottom:#x}, cursor +{} dw, top-down {} dw) by {}",
            (up.wrapping_sub(bottom) as i64) / 4,
            (top.wrapping_sub(down) as i64) / 4,
            crate::platform::loader_hooks::current_guest_frames(10).join(" <- ")
        );
    }
    private(cb, compute, operation, &marker_payload(color, &text))
}

// ---- WAIT_REG_MEM: dw1 = function | mem_space(1)<<4 | operation<<6 |
// engine<<8 | cache_policy<<25; 32-bit: addr lo/hi, reference, mask,
// poll interval; 64-bit (WAIT_REG_MEM64): reference and mask lo/hi.
fn wait_reg_mem(cb: u64, compute: bool, size: u32, func: u32, operation: u32, cache_policy: u32, addr: u64, reference: u64, mask: u64, poll: u32) -> u64 {
    let control = (func & 7) | (1 << 4) | ((operation & 3) << 6) | ((cache_policy & 3) << 25);
    let poll = poll.clamp(1, 0xFFFF);
    if size & 0xFF == 1 {
        emit(cb, &[hdr(op::WAIT_REG_MEM64, 8, compute), control, lo(addr) & !7, hi(addr), lo(reference), hi(reference), lo(mask), hi(mask), poll])
    } else {
        emit(cb, &[hdr(op::WAIT_REG_MEM, 6, compute), control, lo(addr) & !3, hi(addr), lo(reference), lo(mask), poll])
    }
}

fn wait_size_bytes(size: u32) -> u64 {
    if size & 0xFF == 1 { 9 * 4 } else { 7 * 4 }
}

/// Agc data selectors (write/copy source and destination) encode
/// `(PM4 select << 1) | engine` (layout per public sources; the title's labels use
/// destination 4 = TC L2 memory).
fn agc_sel(v: u32) -> (u32, u32) {
    ((v >> 1) & 0xF, v & 1)
}

// ---- WRITE_DATA: dw1 = dst_sel<<8 | addr_incr<<16 (1 = no increment) |
// wr_confirm<<20 | cache_policy<<25 | engine<<30. The Agc `increment`
// argument is that bit (0 = increment), as the title's 64-bit label writes
// [value lo, value hi] require.
unsafe fn write_data(cb: u64, compute: bool, dst: u32, cache_policy: u32, addr: u64, data: *const u32, n: u32, increment: u32, confirm: u32) -> u64 {
    if n == 0 || n > 0x3FFC || data.is_null() {
        return 0;
    }
    let (sel, engine) = agc_sel(dst);
    let confirm = if sel == 0 { 0 } else { confirm & 1 };
    let control = (sel << 8) | ((increment & 1) << 16) | (confirm << 20) | ((cache_policy & 3) << 25) | (engine << 30);
    let mut v = vec![hdr(op::WRITE_DATA, 3 + n, compute), control, lo(addr), hi(addr)];
    v.extend_from_slice(unsafe { std::slice::from_raw_parts(data, n as usize) });
    emit(cb, &v)
}

// ---- COPY_DATA: dw1 = src_sel | dst_sel<<8 | src_cp<<13 | count_sel<<16 |
// wr_confirm<<20 | dst_cp<<25; then src lo/hi, dst lo/hi.
fn copy_data(cb: u64, compute: bool, dst: u32, dst_cp: u32, dst_addr: u64, src: u32, src_cp: u32, src_v: u64, item_size: u32, confirm: u32) -> u64 {
    let (ssel, engine) = agc_sel(src);
    let (dsel, _) = agc_sel(dst);
    let control = ssel | (dsel << 8) | ((src_cp & 3) << 13) | ((item_size & 1) << 16) | ((confirm & 1) << 20) | ((dst_cp & 3) << 25) | (engine << 30);
    emit(cb, &[hdr(op::COPY_DATA, 5, compute), control, lo(src_v), hi(src_v), lo(dst_addr), hi(dst_addr)])
}

// ---- DMA_DATA: dw1 = engine | src_cp<<13 | dst_sel<<20 | dst_cp<<25 |
// src_sel<<29 | cp_sync<<31; src lo/hi, dst lo/hi; dw6 = byte_count |
// sas<<26 | das<<27 | saic<<28 | daic<<29 | raw_wait<<30 | dis_wc<<31.
// Agc src/dst: bits[1:0] select, bit 2 register space, bit 3 no increment.
#[allow(clippy::too_many_arguments)]
fn dma_data(cb: u64, compute: bool, engine: u32, dst: u32, dst_cp: u32, dst_v: u64, src: u32, src_cp: u32, src_v: u64, bytes: u32, wait_prev: u32, confirm: u32, block: u32) -> u64 {
    let c1 = (engine & 1) | ((src_cp & 3) << 13) | ((dst & 3) << 20) | ((dst_cp & 3) << 25) | ((src & 3) << 29) | ((block & 1) << 31);
    let c6 = (bytes & 0x03FF_FFFF)
        | (((src >> 2) & 1) << 26)
        | (((dst >> 2) & 1) << 27)
        | (((src >> 3) & 1) << 28)
        | (((dst >> 3) & 1) << 29)
        | ((wait_prev & 1) << 30)
        | ((confirm & 1) << 31);
    let p = emit(cb, &[hdr(op::DMA_DATA, 6, compute), c1, lo(src_v), hi(src_v), lo(dst_v), hi(dst_v), c6]);
    if p != 0 && (src >> 4) != 0 {
        crate::hle::stub_log("agc: DMA_DATA from a special Agc source (register/counter) seen", src as i64);
    }
    p
}

// ---- ACQUIRE_MEM (gfx10, 8 dwords): coher_cntl (cb/db actions), size lo/hi
// (256-byte units), base lo/hi (256-byte units), poll interval, gcr_cntl.
fn acquire_mem(cb: u64, compute: bool, engine: u32, cb_db_op: u32, gcr_cntl: u32, base: u64, size: u64, poll: u32) -> u64 {
    let (b, s) = if size == 0 || size == u64::MAX { (0, u64::MAX >> 8) } else { (base >> 8, size.div_ceil(256)) };
    emit(
        cb,
        &[hdr(op::ACQUIRE_MEM, 7, compute), cb_db_op | ((engine & 1) << 31), lo(s), hi(s) & 0xFF, lo(b), hi(b) & 0xFF, (poll / 40).min(0xFFFF), gcr_cntl & 0x7FFFF],
    )
}

fn cond_exec(cb: u64, compute: bool, addr: u64, n: u32) -> u64 {
    if n > 0x3FFF {
        return 0;
    }
    emit(cb, &[hdr(op::COND_EXEC, 4, compute), lo(addr) & !3, hi(addr), 0, n])
}

fn jump(cb: u64, compute: bool, chain: u32, cache_policy: u32, target: u64, size: u32) -> u64 {
    let control = (size & 0xF_FFFF) | ((chain & 1) << 20) | (1 << 23) | ((cache_policy & 3) << 28);
    emit(cb, &[hdr(op::INDIRECT_BUFFER, 3, compute), lo(target) & !3, hi(target), control])
}

/// `DS_DRAW_TRACE=<index count>` (diagnosis): the guest call chains of the
/// first 16 draw-builder calls with that index count.
fn draw_trace(what: &str, count: u32) {
    use std::sync::atomic::{AtomicU32, Ordering};
    static WANT: std::sync::OnceLock<Option<u32>> = std::sync::OnceLock::new();
    static LEFT: AtomicU32 = AtomicU32::new(16);
    let Some(want) = *WANT.get_or_init(|| std::env::var("DS_DRAW_TRACE").ok().and_then(|v| v.trim().parse().ok())) else { return };
    if count == want && LEFT.load(Ordering::Relaxed) > 0 && LEFT.fetch_sub(1, Ordering::Relaxed) > 0 {
        // On a fiber, also the guest stack above the call: the callers'
        // saved registers (the render item, geometry, ...).
        let stack = match crate::hle::libs::fiber::guest_stack_words(96) {
            Some(w) => format!("; guest stack: {}", w.iter().map(|v| format!("{v:x}")).collect::<Vec<_>>().join(" ")),
            None => String::new(),
        };
        crate::info!("agc: {what} count {count}: {}{stack}", crate::platform::loader_hooks::current_guest_frames(16).join(" <- "));
    }
}

/// `DS_INDIRECT_TRACE=1`: each distinct guest call chain of the indirect
/// draw builders, once (up to 32), with the guest stack above the call;
/// only the first 20000 calls are unwound (the unwinding is slow).
fn indirect_trace(what: &str, offset: u32) {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    static SEEN: parking_lot::Mutex<Vec<String>> = parking_lot::Mutex::new(Vec::new());
    static CALLS: std::sync::atomic::AtomicU32 = std::sync::atomic::AtomicU32::new(0);
    if !*ON.get_or_init(|| std::env::var_os("DS_INDIRECT_TRACE").is_some())
        || CALLS.fetch_add(1, std::sync::atomic::Ordering::Relaxed) >= 20_000
        || SEEN.lock().len() >= 32
    {
        return;
    }
    let frames = crate::platform::loader_hooks::current_guest_frames(12);
    let key = frames.iter().take(6).cloned().collect::<Vec<_>>().join(" <- ");
    {
        let mut g = SEEN.lock();
        if g.len() >= 32 || g.contains(&key) {
            return;
        }
        g.push(key);
    }
    let stack = match crate::hle::libs::fiber::guest_stack_words(64) {
        Some(w) => format!("; guest stack: {}", w.iter().map(|v| format!("{v:x}")).collect::<Vec<_>>().join(" ")),
        None => String::new(),
    };
    crate::info!("agc: {what} offset {offset:#x}: {}{stack}", frames.join(" <- "));
}

// ---- EVENT_WRITE: dw1 = event_type | event_index<<8. Event types 0x38/0x39
// (with an address, per the Agc size rule) carry addr lo/hi.
fn event_index(t: u32) -> u32 {
    match t {
        0x07 | 0x10 => 4,        // CS/PS partial flush style events
        0x14 | 0x15 => 1,        // ZPASS_DONE family
        0x1E | 0x1F => 2,        // SAMPLE_PIPELINESTAT
        0x04 | 0x0F => 4,        // CACHE_FLUSH / VS/PS partial flushes
        0x28 | 0x2F | 0x30 => 5, // timestamp events
        _ => 0,
    }
}

fn event_write(cb: u64, compute: bool, t: u32, addr: u64) -> u64 {
    let t = t & 0x3F;
    let cntl = t | (event_index(t) << 8);
    if t & 0x3E == 0x38 {
        emit(cb, &[hdr(op::EVENT_WRITE, 3, compute), cntl, lo(addr) & !7, hi(addr)])
    } else {
        emit(cb, &[hdr(op::EVENT_WRITE, 1, compute), cntl])
    }
}

fn event_write_size(t: u32) -> u64 {
    if t & 0x3E == 0x38 { 16 } else { 8 }
}

// ---- RELEASE_MEM (gfx10, 8 dwords): event_cntl = event_type | index<<8 |
// gcr_cntl<<12 | cache_policy<<25; data_cntl = dst_sel<<16 | int_sel<<24 |
// data_sel<<29; addr lo/hi, data lo/hi, int ctx id.
#[allow(clippy::too_many_arguments)]
fn release_mem(cb: u64, compute: bool, action: u32, gcr: u32, dst: u32, cache_policy: u32, addr: u64, data_sel: u32, data: u64, gds_off: u32, gds_size: u32, int_sel: u32, ctx: u32) -> u64 {
    let action = action & 0x3F;
    let index = if action >= 0x2F { 6 } else { 5 };
    let data = if data_sel & 7 == 5 { (gds_off as u64 & 0xFFFF) | ((gds_size as u64 & 0xFFFF) << 16) } else { data };
    emit(
        cb,
        &[
            hdr(op::RELEASE_MEM, 7, compute),
            action | (index << 8) | ((gcr & 0xFFF) << 12) | ((cache_policy & 3) << 25),
            ((dst & 3) << 16) | ((int_sel & 7) << 24) | ((data_sel & 7) << 29),
            lo(addr) & !3,
            hi(addr),
            lo(data),
            hi(data),
            ctx & 0x07FF_FFFF,
        ],
    )
}

fn set_reg(cb: u64, compute: bool, opcode: u32, r: u64) -> u64 {
    let (off, val) = split_reg(r);
    emit(cb, &[hdr(opcode, 2, compute), off & 0xFFFF, val])
}

fn set_regs_indirect(cb: u64, compute: bool, opcode: u32, regs: u64, n: u32) -> u64 {
    emit(cb, &[hdr(opcode, 3, compute), lo(regs), hi(regs), n])
}

/// One SET_*_REG packet per register (ShaderRegister array).
unsafe fn set_regs_direct(cb: u64, compute: bool, opcode: u32, regs: *const u32, n: u32) -> u64 {
    if n == 0 || regs.is_null() {
        return 0;
    }
    let mut v = Vec::with_capacity(3 * n as usize);
    for i in 0..n as usize {
        let (off, val) = unsafe { (*regs.add(2 * i), *regs.add(2 * i + 1)) };
        v.extend_from_slice(&[hdr(opcode, 2, compute), off & 0xFFFF, val]);
    }
    emit(cb, &v)
}

/// With `values` NULL the packet is reserved (values zeroed) for the title to
/// fill through its payload address (eboot+0x1d0a6e0 does this for user data).
unsafe fn set_reg_range(cb: u64, compute: bool, opcode: u32, offset: u32, values: *const u32, n: u32) -> u64 {
    if n == 0 || n > 0x3FF0 {
        return 0;
    }
    let mut v = vec![hdr(opcode, 1 + n, compute), offset & 0xFFFF];
    if values.is_null() {
        v.resize(2 + n as usize, 0);
    } else {
        v.extend_from_slice(unsafe { std::slice::from_raw_parts(values, n as usize) });
    }
    emit(cb, &v)
}

/// Patch helpers receive the packet address returned by a writer.
unsafe fn dw(cmd: u64, i: u64) -> *mut u32 {
    (cmd + 4 * i) as *mut u32
}

fn payload_dwords(header: u32) -> u64 {
    ((header >> 16) & 0x3FFF) as u64 + 1
}

/// Packet size in dwords from a type-3 (or type-2 filler) header.
fn packet_dwords(header: u32) -> u64 {
    match header >> 30 {
        3 if (header >> 16) & 0x3FFF == 0x3FFF => 1, // header-only NOP
        3 => 1 + payload_dwords(header),
        _ => 1,
    }
}

/// Context state operation sizes in dwords (Clear 5, Push/Pop 27, PushClear
/// 32; per public sources).
fn context_state_dwords(operation: u32) -> u32 {
    match operation {
        0 => 5,
        1 | 2 => 27,
        3 => 32,
        _ => 0,
    }
}

const OK: i32 = 0;
const ERR_INVALID: i32 = 0x8A6C_0001u32 as i32;

sysv! {
    // ---- queue reset / pipeline sync
    fn sceAgcDcbResetQueue(cb: u64, operation: u32, state: u32) -> u64 {
        emit(cb, &[hdr(op::CLEAR_STATE, 1, DCB), state & 0xF])
    }
    fn sceAgcAcbResetQueue(cb: u64, operation: u32) -> u64 {
        private(cb, ACB, pop::ACB_RESET, &[operation])
    }
    fn sceAgcDcbStallCommandBufferParser(cb: u64) -> u64 { emit(cb, &[hdr(op::PFP_SYNC_ME, 1, DCB), 0]) }
    fn sceAgcDcbStallCommandBufferParserGetSize() -> u64 { 8 }
    fn sceAgcDcbContextStateOpGetSize(operation: u32) -> u64 { 4 * context_state_dwords(operation) as u64 }
    /// Unnamed export qj7QZpgr9Uw (context state operation; role per public sources,
    /// consistent with the title's call at eboot+0x1d225b8): one private
    /// packet of the documented size. 0 clear, 1 push, 2 pop, 3 push + clear.
    fn ds_agc_dcb_context_state_op(cb: u64, operation: u32) -> u64 {
        let n = context_state_dwords(operation);
        if n == 0 { return 0; }
        let mut v = vec![0u32; n as usize];
        v[0] = hdr(op::NOP, n - 1, DCB);
        v[1] = DS_MAGIC;
        v[2] = pop::CONTEXT_STATE;
        v[3] = operation;
        emit(cb, &v)
    }

    // ---- display synchronisation and workloads
    fn sceAgcDcbWaitUntilSafeForRendering(cb: u64, handle: u32, index: u32) -> u64 {
        private(cb, DCB, pop::WAIT_SAFE_FOR_RENDERING, &[handle, index])
    }
    fn sceAgcAcbWaitUntilSafeForRendering(cb: u64, handle: u32, index: u32) -> u64 {
        private(cb, ACB, pop::WAIT_SAFE_FOR_RENDERING, &[handle, index])
    }
    fn sceAgcDcbSetFlip(cb: u64, handle: u32, index: i32, mode: u32, arg: i64) -> u64 {
        private(cb, DCB, pop::FLIP, &[handle, index as u32, mode, lo(arg as u64), hi(arg as u64)])
    }
    fn sceAgcAcbSetFlip(cb: u64, handle: u32, index: i32, mode: u32, arg: i64) -> u64 {
        private(cb, ACB, pop::FLIP, &[handle, index as u32, mode, lo(arg as u64), hi(arg as u64)])
    }
    fn sceAgcDcbSetWorkloadsActive(cb: u64, stream: u32, ids: *const u32, count: u32) -> u64 {
        let mut a = vec![stream, count];
        if !ids.is_null() { a.extend_from_slice(std::slice::from_raw_parts(ids, count.min(64) as usize)); }
        private(cb, DCB, pop::WORKLOADS_ACTIVE, &a)
    }
    fn sceAgcAcbSetWorkloadsActive(cb: u64, stream: u32, ids: *const u32, count: u32) -> u64 {
        let mut a = vec![stream, count];
        if !ids.is_null() { a.extend_from_slice(std::slice::from_raw_parts(ids, count.min(64) as usize)); }
        private(cb, ACB, pop::WORKLOADS_ACTIVE, &a)
    }
    fn sceAgcDcbSetWorkloadComplete(cb: u64, stream: u32, id: u32) -> u64 {
        private(cb, DCB, pop::WORKLOAD_COMPLETE, &[stream, id])
    }
    fn sceAgcAcbSetWorkloadComplete(cb: u64, stream: u32, id: u32) -> u64 {
        private(cb, ACB, pop::WORKLOAD_COMPLETE, &[stream, id])
    }

    // ---- debug markers
    fn sceAgcDcbPushMarker(cb: u64, s: *const u8, color: u32) -> u64 { marker(cb, DCB, pop::PUSH_MARKER, s, color) }
    fn sceAgcDcbSetMarker(cb: u64, s: *const u8, color: u32) -> u64 { marker(cb, DCB, pop::SET_MARKER, s, color) }
    fn sceAgcDcbPopMarker(cb: u64) -> u64 { private(cb, DCB, pop::POP_MARKER, &[]) }
    fn sceAgcAcbPushMarker(cb: u64, s: *const u8, color: u32) -> u64 { marker(cb, ACB, pop::PUSH_MARKER, s, color) }
    fn sceAgcAcbSetMarker(cb: u64, s: *const u8, color: u32) -> u64 { marker(cb, ACB, pop::SET_MARKER, s, color) }
    fn sceAgcAcbPopMarker(cb: u64) -> u64 { private(cb, ACB, pop::POP_MARKER, &[]) }

    // ---- registers
    fn sceAgcDcbSetCxRegisterDirect(cb: u64, r: u64) -> u64 { set_reg(cb, DCB, op::SET_CONTEXT_REG, r) }
    fn sceAgcDcbSetCxRegisterDirectGetSize() -> u64 { 12 }
    fn sceAgcDcbSetShRegisterDirect(cb: u64, r: u64) -> u64 { set_reg(cb, DCB, op::SET_SH_REG, r) }
    fn sceAgcDcbSetShRegisterDirectGetSize() -> u64 { 12 }
    fn sceAgcDcbSetUcRegisterDirect(cb: u64, r: u64) -> u64 { set_reg(cb, DCB, op::SET_UCONFIG_REG, r) }
    fn sceAgcDcbSetUcRegisterDirectGetSize() -> u64 { 12 }
    fn sceAgcDcbSetCxRegistersIndirect(cb: u64, regs: u64, n: u32) -> u64 { set_regs_indirect(cb, DCB, op::LOAD_CONTEXT_REG_INDEX, regs, n) }
    fn sceAgcDcbSetCxRegistersIndirectGetSize() -> u64 { 16 }
    fn sceAgcDcbSetShRegistersIndirect(cb: u64, regs: u64, n: u32) -> u64 { set_regs_indirect(cb, DCB, op::LOAD_SH_REG_INDEX, regs, n) }
    fn sceAgcDcbSetShRegistersIndirectGetSize() -> u64 { 16 }
    fn sceAgcDcbSetUcRegistersIndirect(cb: u64, regs: u64, n: u32) -> u64 { set_regs_indirect(cb, DCB, op::LOAD_UCONFIG_REG, regs, n) }
    fn sceAgcDcbSetUcRegistersIndirectGetSize() -> u64 { 16 }
    fn sceAgcCbSetShRegisterRangeDirect(cb: u64, offset: u32, values: *const u32, n: u32) -> u64 { set_reg_range(cb, DCB, op::SET_SH_REG, offset, values, n) }
    fn sceAgcCbSetShRegisterRangeDirectGetSize(n: u32) -> u64 { 8 + 4 * n as u64 }
    fn sceAgcCbSetUcRegisterRangeDirect(cb: u64, offset: u32, values: *const u32, n: u32) -> u64 { set_reg_range(cb, DCB, op::SET_UCONFIG_REG, offset, values, n) }
    fn sceAgcCbSetUcRegisterRangeDirectGetSize(n: u32) -> u64 { 8 + 4 * n as u64 }
    fn sceAgcCbSetShRegistersDirect(cb: u64, regs: *const u32, n: u32) -> u64 { set_regs_direct(cb, DCB, op::SET_SH_REG, regs, n) }
    fn sceAgcCbSetShRegistersDirectGetSize(n: u32) -> u64 { 12 * n as u64 }
    fn sceAgcCbSetUcRegistersDirect(cb: u64, regs: *const u32, n: u32) -> u64 { set_regs_direct(cb, DCB, op::SET_UCONFIG_REG, regs, n) }
    fn sceAgcCbSetUcRegistersDirectGetSize(n: u32) -> u64 { 12 * n as u64 }

    fn sceAgcSetCxRegIndirectPatchSetAddress(cmd: u64, regs: u64) -> i32 { if cmd == 0 { return ERR_INVALID; } *dw(cmd, 1) = lo(regs); *dw(cmd, 2) = hi(regs); OK }
    fn sceAgcSetShRegIndirectPatchSetAddress(cmd: u64, regs: u64) -> i32 { if cmd == 0 { return ERR_INVALID; } *dw(cmd, 1) = lo(regs); *dw(cmd, 2) = hi(regs); OK }
    fn sceAgcSetUcRegIndirectPatchSetAddress(cmd: u64, regs: u64) -> i32 { if cmd == 0 { return ERR_INVALID; } *dw(cmd, 1) = lo(regs); *dw(cmd, 2) = hi(regs); OK }
    fn sceAgcSetCxRegIndirectPatchSetNumRegisters(cmd: u64, n: u32) -> i32 { if cmd == 0 { return ERR_INVALID; } *dw(cmd, 3) = n; OK }
    fn sceAgcSetShRegIndirectPatchSetNumRegisters(cmd: u64, n: u32) -> i32 { if cmd == 0 { return ERR_INVALID; } *dw(cmd, 3) = n; OK }
    fn sceAgcSetUcRegIndirectPatchSetNumRegisters(cmd: u64, n: u32) -> i32 { if cmd == 0 { return ERR_INVALID; } *dw(cmd, 3) = n; OK }
    fn sceAgcSetCxRegIndirectPatchAddRegisters(cmd: u64, n: u32) -> i32 { if cmd == 0 { return ERR_INVALID; } *dw(cmd, 3) += n; OK }
    fn sceAgcSetShRegIndirectPatchAddRegisters(cmd: u64, n: u32) -> i32 { if cmd == 0 { return ERR_INVALID; } *dw(cmd, 3) += n; OK }
    fn sceAgcSetUcRegIndirectPatchAddRegisters(cmd: u64, n: u32) -> i32 { if cmd == 0 { return ERR_INVALID; } *dw(cmd, 3) += n; OK }

    // ---- index state and draws
    /// gfx9+ form: SET_UCONFIG_REG_INDEX (index 2) on VGT_INDEX_TYPE.
    fn sceAgcDcbSetIndexSize(cb: u64, size: u32, cache_policy: u32) -> u64 {
        emit(cb, &[hdr(op::SET_UCONFIG_REG_INDEX, 2, DCB), (2 << 28) | REG_VGT_INDEX_TYPE, (size & 3) | ((cache_policy & 3) << 6)])
    }
    fn sceAgcDcbSetIndexSizeGetSize() -> u64 { 12 }
    /// Unnamed export -KRzWekV120: VGT_INDEX_TYPE with index type, buffer
    /// access policy and instance packing (field placement per public sources;
    /// called at eboot+0x9145c0).
    fn ds_agc_dcb_set_index_type(cb: u64, index_type: u32, policy: u32, disable_packing: u32) -> u64 {
        let v = (index_type & 3) | ((policy & 3) << 6) | ((disable_packing & 1) << 14);
        emit(cb, &[hdr(op::SET_UCONFIG_REG_INDEX, 2, DCB), (2 << 28) | REG_VGT_INDEX_TYPE, v])
    }
    fn sceAgcDcbSetIndexBuffer(cb: u64, addr: u64) -> u64 { emit(cb, &[hdr(op::INDEX_BASE, 2, DCB), lo(addr) & !1, hi(addr)]) }
    fn sceAgcDcbSetIndexBufferGetSize() -> u64 { 12 }
    fn sceAgcDcbSetIndexCount(cb: u64, n: u32) -> u64 { emit(cb, &[hdr(op::INDEX_BUFFER_SIZE, 1, DCB), n]) }
    fn sceAgcDcbSetIndexCountGetSize() -> u64 { 8 }
    fn sceAgcDcbSetNumInstances(cb: u64, n: u32) -> u64 { emit(cb, &[hdr(op::NUM_INSTANCES, 1, DCB), n]) }
    fn sceAgcDcbSetNumInstancesGetSize() -> u64 { 8 }
    fn sceAgcDcbDrawIndex(cb: u64, count: u32, addr: u64, modifier: u64) -> u64 {
        draw_trace("DrawIndex", count);
        emit(cb, &[hdr(op::DRAW_INDEX_2, 5, DCB), count.max(1), lo(addr), hi(addr), count, draw_initiator(modifier, 0)])
    }
    fn sceAgcDcbDrawIndexGetSize() -> u64 { 24 }
    fn sceAgcDcbDrawIndexAuto(cb: u64, count: u32, modifier: u64) -> u64 {
        emit(cb, &[hdr(op::DRAW_INDEX_AUTO, 2, DCB), count, draw_initiator(modifier, 2)])
    }
    fn sceAgcDcbDrawIndexAutoGetSize() -> u64 { 12 }
    fn sceAgcDcbDrawIndexOffset(cb: u64, offset: u32, count: u32, modifier: u64) -> u64 {
        draw_trace("DrawIndexOffset", count);
        emit(cb, &[hdr(op::DRAW_INDEX_OFFSET_2, 4, DCB), offset.wrapping_add(count), offset, count, draw_initiator(modifier, 0)])
    }
    fn sceAgcDcbDrawIndexOffsetGetSize() -> u64 { 20 }
    fn sceAgcDcbDrawIndexMultiInstanced(cb: u64, count: u32, addr: u64, ids: u64, instances: u32, modifier: u64) -> u64 {
        private(cb, DCB, pop::DRAW_INDEX_MULTI_INSTANCED, &[count, lo(addr), hi(addr), lo(ids), hi(ids), instances])
    }
    fn sceAgcDcbDrawIndexMultiInstancedGetSize() -> u64 { 36 }
    /// SET_BASE base_index 1 = draw-indirect arguments (shader_type selects
    /// graphics/compute; compute uses index 2 here).
    fn sceAgcDcbSetBaseIndirectArgs(cb: u64, shader_type: u32, addr: u64) -> u64 {
        let compute = shader_type == 1;
        emit(cb, &[hdr(op::SET_BASE, 3, compute), if compute { 2 } else { 1 }, lo(addr), hi(addr)])
    }
    fn sceAgcDcbDrawIndexIndirect(cb: u64, offset: u32, modifier: u64) -> u64 {
        static SEEN: parking_lot::Mutex<Option<std::collections::HashSet<u64>>> = parking_lot::Mutex::new(None);
        if SEEN.lock().get_or_insert_with(Default::default).insert(modifier) {
            crate::info!("agc: DrawIndexIndirect modifier {modifier:#018x}");
        }
        indirect_trace("DrawIndexIndirect", offset);
        emit(cb, &[hdr(op::DRAW_INDEX_INDIRECT, 4, DCB), offset, 0, 0, draw_initiator(modifier, 0)])
    }
    fn sceAgcDcbDrawIndexIndirectGetSize() -> u64 { 20 }
    fn sceAgcDcbDrawIndirect(cb: u64, offset: u32, modifier: u64) -> u64 {
        emit(cb, &[hdr(op::DRAW_INDIRECT, 4, DCB), offset, 0, 0, draw_initiator(modifier, 2)])
    }
    fn sceAgcDcbDrawIndirectGetSize() -> u64 { 20 }
    /// DRAW_INDIRECT_MULTI: offset, base vtx loc, start inst loc, flags
    /// (count_indirect_enable bit 30), count, count addr lo/hi, stride, initiator.
    fn sceAgcDcbDrawIndirectMulti(cb: u64, offset: u32, count_indirect: u32, max_count: u32, count_addr: u64, stride: u32, modifier: u64) -> u64 {
        emit(cb, &[hdr(op::DRAW_INDIRECT_MULTI, 9, DCB), offset, 0, 0, (count_indirect & 1) << 30, max_count, lo(count_addr), hi(count_addr), stride, draw_initiator(modifier, 2)])
    }
    fn sceAgcDcbDrawIndirectMultiGetSize() -> u64 { 40 }
    fn sceAgcDcbDrawIndexIndirectMulti(cb: u64, offset: u32, count_indirect: u32, max_count: u32, count_addr: u64, stride: u32, modifier: u64) -> u64 {
        indirect_trace("DrawIndexIndirectMulti", offset);
        emit(cb, &[hdr(op::DRAW_INDEX_INDIRECT_MULTI, 9, DCB), offset, 0, 0, (count_indirect & 1) << 30, max_count, lo(count_addr), hi(count_addr), stride, draw_initiator(modifier, 0)])
    }
    fn sceAgcDcbDrawIndexIndirectMultiGetSize() -> u64 { 40 }

    // ---- dispatches
    fn sceAgcCbDispatch(cb: u64, x: u32, y: u32, z: u32, modifier: u32) -> u64 {
        if emit_census() {
            EMITS[1].fetch_add(1, std::sync::atomic::Ordering::Relaxed);
        }
        emit(cb, &[hdr(op::DISPATCH_DIRECT, 4, ACB), x, y, z, dispatch_initiator(modifier)])
    }
    fn sceAgcCbDispatchGetSize() -> u64 { 20 }
    fn sceAgcDcbDispatchIndirect(cb: u64, offset: u32, flags: u32) -> u64 {
        emit(cb, &[hdr(op::DISPATCH_INDIRECT, 2, ACB), offset, dispatch_initiator(flags)])
    }
    fn sceAgcDcbDispatchIndirectGetSize() -> u64 { 12 }
    /// Compute queues take the argument address in the packet.
    fn sceAgcAcbDispatchIndirect(cb: u64, args: u64, modifier: u32) -> u64 {
        emit(cb, &[hdr(op::DISPATCH_INDIRECT, 3, ACB), lo(args), hi(args), dispatch_initiator(modifier)])
    }
    fn sceAgcAcbDispatchIndirectGetSize() -> u64 { 16 }

    // ---- events, caches and synchronisation
    fn sceAgcDcbEventWrite(cb: u64, t: u32, addr: u64) -> u64 { event_write(cb, DCB, t, addr) }
    fn sceAgcDcbEventWriteGetSize(t: u32) -> u64 { event_write_size(t) }
    fn sceAgcAcbEventWrite(cb: u64, t: u32, addr: u64) -> u64 { event_write(cb, ACB, t, addr) }
    fn sceAgcAcbEventWriteGetSize(t: u32) -> u64 { event_write_size(t) }
    fn sceAgcDcbAcquireMem(cb: u64, engine: u32, cb_db_op: u32, gcr_cntl: u32, base: u64, size: u64, poll: u32) -> u64 {
        acquire_mem(cb, DCB, engine, cb_db_op, gcr_cntl, base, size, poll)
    }
    fn sceAgcDcbAcquireMemGetSize() -> u64 { 32 }
    fn sceAgcAcbAcquireMem(cb: u64, gcr_cntl: u32, base: u64, size: u64, poll: u32) -> u64 {
        acquire_mem(cb, ACB, 0, 0, gcr_cntl, base, size, poll)
    }
    fn sceAgcAcbAcquireMemGetSize() -> u64 { 32 }
    #[allow(clippy::too_many_arguments)]
    fn sceAgcCbReleaseMem(cb: u64, action: u32, gcr: u32, dst: u32, cache_policy: u32, addr: u64, data_sel: u32, data: u64, gds_off: u32, gds_size: u32, int_sel: u32, ctx: u32) -> u64 {
        release_mem(cb, DCB, action, gcr, dst, cache_policy, addr, data_sel, data, gds_off, gds_size, int_sel, ctx)
    }
    fn sceAgcCbReleaseMemGetSize() -> u64 { 32 }
    fn sceAgcCbQueueEndOfPipeActionGetSize() -> u64 { 32 }
    fn sceAgcQueueEndOfPipeActionPatchAddress(cmd: u64, label: u64) -> i32 { if cmd == 0 { return ERR_INVALID; } *dw(cmd, 3) = lo(label) & !3; *dw(cmd, 4) = hi(label); OK }
    fn sceAgcQueueEndOfPipeActionPatchData(cmd: u64, data: u64) -> i32 { if cmd == 0 { return ERR_INVALID; } *dw(cmd, 5) = lo(data); *dw(cmd, 6) = hi(data); OK }

    fn sceAgcDcbWaitRegMem(cb: u64, size: u32, func: u32, operation: u32, cache_policy: u32, addr: u64, reference: u64, mask: u64, poll: u32) -> u64 {
        wait_reg_mem(cb, DCB, size, func, operation, cache_policy, addr, reference, mask, poll)
    }
    fn sceAgcDcbWaitOnAddressGetSize(size: u32) -> u64 { wait_size_bytes(size) }
    fn sceAgcAcbWaitRegMem(cb: u64, size: u32, func: u32, cache_policy: u32, addr: u64, reference: u64, mask: u64, poll: u32) -> u64 {
        wait_reg_mem(cb, ACB, size, func, 0, cache_policy, addr, reference, mask, poll)
    }
    fn sceAgcAcbWaitOnAddressGetSize(size: u32) -> u64 { wait_size_bytes(size) }
    fn sceAgcWaitRegMemPatchAddress(cmd: u64, addr: u64) -> i32 { if cmd == 0 { return ERR_INVALID; } *dw(cmd, 2) = lo(addr) & !3; *dw(cmd, 3) = hi(addr); OK }
    fn sceAgcWaitRegMemPatchReference(cmd: u64, reference: u64) -> i32 {
        if cmd == 0 { return ERR_INVALID; }
        *dw(cmd, 4) = lo(reference);
        if (*dw(cmd, 0) >> 8) & 0xFF == op::WAIT_REG_MEM64 { *dw(cmd, 5) = hi(reference); }
        OK
    }

    fn sceAgcDcbCondExec(cb: u64, addr: u64, n: u32) -> u64 { cond_exec(cb, DCB, addr, n) }
    fn sceAgcDcbCondExecGetSize() -> u64 { 20 }
    fn sceAgcAcbCondExec(cb: u64, addr: u64, n: u32) -> u64 { cond_exec(cb, ACB, addr, n) }
    fn sceAgcAcbCondExecGetSize() -> u64 { 20 }
    /// End = first dword after the conditional block.
    fn sceAgcCondExecPatchSetEnd(cmd: u64, end: u64) -> i32 {
        if cmd == 0 || end < cmd + 20 { return ERR_INVALID; }
        *dw(cmd, 4) = ((end - (cmd + 20)) / 4) as u32;
        OK
    }
    fn sceAgcCondExecPatchSetCommandAddress(cmd: u64, addr: u64) -> i32 { if cmd == 0 { return ERR_INVALID; } *dw(cmd, 1) = lo(addr) & !3; *dw(cmd, 2) = hi(addr); OK }

    // ---- predication
    /// SET_PREDICATION: control (pred_op<<16 | wait<<12 | condition<<8),
    /// addr lo/hi, count (dwords covered; 0 = until changed).
    fn sceAgcDcbSetPredication(cb: u64, condition: u32, operation: u32, wait: u32, addr: u64, count: u32) -> u64 {
        emit(cb, &[hdr(op::SET_PREDICATION, 4, DCB), ((condition & 1) << 8) | ((wait & 1) << 12) | ((operation & 7) << 16), lo(addr) & !0xF, hi(addr), count])
    }
    fn sceAgcSetPacketPredication(packet: u64, predicate: u32) -> i32 {
        if packet == 0 { return ERR_INVALID; }
        let h = dw(packet, 0);
        if *h >> 30 == 3 { *h = (*h & !1) | (predicate & 1); }
        OK
    }
    fn sceAgcSetRangePredication(start: u64, end: u64, predicate: u32) -> i32 {
        if start == 0 || end < start { return ERR_INVALID; }
        let mut p = start;
        while p < end {
            let h = dw(p, 0);
            if *h >> 30 == 3 { *h = (*h & !1) | (predicate & 1); }
            p += 4 * packet_dwords(*h);
        }
        OK
    }

    // ---- memory writes and copies
    fn sceAgcDcbWriteData(cb: u64, dst: u32, cache_policy: u32, addr: u64, data: *const u32, n: u32, increment: u32, confirm: u32) -> u64 {
        write_data(cb, DCB, dst, cache_policy, addr, data, n, increment, confirm)
    }
    fn sceAgcDcbWriteDataGetSize(n: u32) -> u64 { 16 + 4 * n as u64 }
    fn sceAgcAcbWriteData(cb: u64, dst: u32, cache_policy: u32, addr: u64, data: *const u32, n: u32, increment: u32, confirm: u32) -> u64 {
        write_data(cb, ACB, dst, cache_policy, addr, data, n, increment, confirm)
    }
    fn sceAgcAcbWriteDataGetSize(n: u32) -> u64 { 16 + 4 * n as u64 }
    fn sceAgcWriteDataPatchSetAddressOrOffset(cmd: u64, addr: u64) -> i32 { if cmd == 0 { return ERR_INVALID; } *dw(cmd, 2) = lo(addr); *dw(cmd, 3) = hi(addr); OK }
    fn sceAgcWriteDataPatchSetDst(cmd: u64, dst: u32) -> i32 {
        if cmd == 0 { return ERR_INVALID; }
        let (sel, engine) = agc_sel(dst);
        let c = dw(cmd, 1);
        *c = (*c & !(0xF00 | (1 << 30))) | (sel << 8) | (engine << 30);
        OK
    }

    fn sceAgcDcbCopyData(cb: u64, dst: u32, dst_cp: u32, dst_addr: u64, src: u32, src_cp: u32, src_v: u64, item_size: u32, confirm: u32) -> u64 {
        copy_data(cb, DCB, dst, dst_cp, dst_addr, src, src_cp, src_v, item_size, confirm)
    }
    fn sceAgcDcbCopyDataGetSize() -> u64 { 24 }
    fn sceAgcAcbCopyData(cb: u64, dst: u32, dst_cp: u32, dst_addr: u64, src: u32, src_cp: u32, src_v: u64, item_size: u32, confirm: u32) -> u64 {
        copy_data(cb, ACB, dst, dst_cp, dst_addr, src, src_cp, src_v, item_size, confirm)
    }
    fn sceAgcAcbCopyDataGetSize() -> u64 { 24 }

    #[allow(clippy::too_many_arguments)]
    fn sceAgcDcbDmaData(cb: u64, engine: u32, dst: u32, dst_cp: u32, dst_v: u64, src: u32, src_cp: u32, src_v: u64, bytes: u32, wait_prev: u32, confirm: u32, block: u32) -> u64 {
        dma_data(cb, DCB, engine, dst, dst_cp, dst_v, src, src_cp, src_v, bytes, wait_prev, confirm, block)
    }
    fn sceAgcDcbDmaDataGetSize() -> u64 { 28 }
    #[allow(clippy::too_many_arguments)]
    fn sceAgcAcbDmaData(cb: u64, dst: u32, dst_cp: u32, dst_v: u64, src: u32, src_cp: u32, src_v: u64, bytes: u32, wait_prev: u32, confirm: u32) -> u64 {
        dma_data(cb, ACB, 0, dst, dst_cp, dst_v, src, src_cp, src_v, bytes, wait_prev, confirm, 0)
    }
    fn sceAgcAcbDmaDataGetSize() -> u64 { 28 }
    fn sceAgcDmaDataPatchSetDstAddressOrOffset(cmd: u64, v: u64) -> i32 { if cmd == 0 { return ERR_INVALID; } *dw(cmd, 4) = lo(v); *dw(cmd, 5) = hi(v); OK }
    fn sceAgcDmaDataPatchSetSrcAddressOrOffsetOrImmediate(cmd: u64, v: u64) -> i32 { if cmd == 0 { return ERR_INVALID; } *dw(cmd, 2) = lo(v); *dw(cmd, 3) = hi(v); OK }

    // ---- control flow
    /// mode 1 = chain (continue in the target), 0 = call and return.
    fn sceAgcDcbJump(cb: u64, mode: u32, cache_policy: u32, target: u64, size: u32) -> u64 { jump(cb, DCB, mode, cache_policy, target, size) }
    fn sceAgcDcbJumpGetSize() -> u64 { 16 }
    fn sceAgcAcbJump(cb: u64, cache_policy: u32, target: u64, size: u32) -> u64 { jump(cb, ACB, 1, cache_policy, target, size) }
    fn sceAgcAcbJumpGetSize() -> u64 { 16 }
    /// REWIND: the parser waits until bit 31 (valid) is set, then refetches.
    fn sceAgcDcbRewind(cb: u64, state: u32) -> u64 { emit(cb, &[hdr(op::REWIND, 1, DCB), (state & 1) << 31]) }
    fn sceAgcDcbRewindGetSize() -> u64 { 8 }
    fn sceAgcAcbRewind(cb: u64, state: u32) -> u64 { emit(cb, &[hdr(op::REWIND, 1, ACB), (state & 1) << 31]) }
    fn sceAgcAcbRewindGetSize() -> u64 { 8 }
    fn sceAgcRewindPatchSetRewindState(cmd: u64, state: u32) -> i32 { if cmd == 0 { return ERR_INVALID; } *dw(cmd, 1) = (state & 1) << 31; OK }
    #[allow(clippy::too_many_arguments)]
    fn sceAgcCbBranch(cb: u64, mode: u32, func: u32, cmp: u64, mask: u64, reference: u64, cp1: u32, b1: u64, s1: u32, cp2: u32, b2: u64, s2: u32) -> u64 {
        private(cb, DCB, pop::BRANCH, &[
            (mode & 3) | ((func & 7) << 8), lo(cmp) & !7, hi(cmp), lo(mask), hi(mask), lo(reference), hi(reference),
            lo(b1) & !3, hi(b1), (s1 & 0xF_FFFF) | ((cp1 & 3) << 28), lo(b2) & !3, hi(b2), (s2 & 0xF_FFFF) | ((cp2 & 3) << 28),
        ])
    }
    fn sceAgcCbBranchGetSize() -> u64 { 16 * 4 }
    fn sceAgcCbNop(cb: u64, n: u32) -> u64 {
        match n {
            0 => 0,
            1 => emit(cb, &[hdr(op::NOP, 1, DCB) | 0x3FFF_0000]),
            _ => {
                let mut v = vec![0u32; n as usize];
                v[0] = hdr(op::NOP, n - 1, DCB);
                emit(cb, &v)
            }
        }
    }
    fn sceAgcCbNopGetSize(n: u32) -> u64 { 4 * n as u64 }

    // ---- statistics
    fn sceAgcDcbGetLodStats(cb: u64, cache_policy: u32, buffer: u64, size: u32, reset_count: u32, force_reset: u32, report_and_reset: u32, interval: u32) -> u64 {
        let flags = ((cache_policy & 3) << 28) | ((report_and_reset & 1) << 19) | ((force_reset & 1) << 18) | ((reset_count & 0xFF) << 10) | ((interval & 0xFF) << 2);
        private(cb, DCB, pop::LOD_STATS, &[lo(buffer), hi(buffer), size, flags])
    }
    fn sceAgcDcbGetLodStatsGetSize() -> u64 { 7 * 4 }

    // ---- packet inspection
    /// Size of the packet at `packet`, in bytes.
    fn sceAgcGetPacketSize(packet: u64) -> u32 { if packet == 0 { 0 } else { 4 * packet_dwords(*dw(packet, 0)) as u32 } }
    /// Data packets are NOPs carrying a payload: type 0 = whole payload,
    /// otherwise the payload after its first dword. range = {base, size}.
    fn sceAgcGetDataPacketPayloadRange(range: *mut u64, cmd: u64, kind: i32) -> i32 {
        if range.is_null() || cmd == 0 { return ERR_INVALID; }
        let h = *dw(cmd, 0);
        let count4 = (((h >> 16) & 0x3FFF) * 4) as u64;
        if kind != 0 {
            *range = cmd + 8;
            *range.add(1) = count4;
        } else if (h >> 16) & 0x3FFF == 0x3FFF {
            *range = 0;
            *range.add(1) = 0;
        } else {
            *range = cmd + 4;
            *range.add(1) = count4 + 4;
        }
        OK
    }
    /// Unnamed export V++UgBtQhn0: payload address of a data packet (type 0 =
    /// first payload dword, otherwise the dword after it). Role inferred from
    /// the title (eboot+0x1d0c64e fills reserved user-data packets through it).
    fn ds_agc_get_data_packet_payload_address(out: *mut u64, cmd: u64, kind: i32) -> i32 {
        if out.is_null() || cmd == 0 { return ERR_INVALID; }
        *out = if kind != 0 { cmd + 8 } else { cmd + 4 };
        OK
    }
    fn sceAgcSuspendPoint() -> i32 { 0 }
    fn sceAgcDebugRaiseException(id: u32) -> i32 {
        crate::warn!("sceAgcDebugRaiseException({id:#x}) (ignored)");
        0
    }
}

pub fn register(r: &mut Registry) {
    reg!(r, "libSceAgc", [
        sceAgcDcbResetQueue, sceAgcAcbResetQueue, sceAgcDcbStallCommandBufferParser, sceAgcDcbStallCommandBufferParserGetSize,
        sceAgcDcbContextStateOpGetSize,
        sceAgcDcbWaitUntilSafeForRendering, sceAgcAcbWaitUntilSafeForRendering, sceAgcDcbSetFlip, sceAgcAcbSetFlip,
        sceAgcDcbSetWorkloadsActive, sceAgcAcbSetWorkloadsActive, sceAgcDcbSetWorkloadComplete, sceAgcAcbSetWorkloadComplete,
        sceAgcDcbPushMarker, sceAgcDcbSetMarker, sceAgcDcbPopMarker, sceAgcAcbPushMarker, sceAgcAcbSetMarker, sceAgcAcbPopMarker,
        sceAgcDcbSetCxRegisterDirect, sceAgcDcbSetCxRegisterDirectGetSize, sceAgcDcbSetShRegisterDirect, sceAgcDcbSetShRegisterDirectGetSize,
        sceAgcDcbSetUcRegisterDirect, sceAgcDcbSetUcRegisterDirectGetSize, sceAgcDcbSetCxRegistersIndirect, sceAgcDcbSetCxRegistersIndirectGetSize,
        sceAgcDcbSetShRegistersIndirect, sceAgcDcbSetShRegistersIndirectGetSize, sceAgcDcbSetUcRegistersIndirect, sceAgcDcbSetUcRegistersIndirectGetSize,
        sceAgcCbSetShRegisterRangeDirect, sceAgcCbSetShRegisterRangeDirectGetSize, sceAgcCbSetUcRegisterRangeDirect, sceAgcCbSetUcRegisterRangeDirectGetSize,
        sceAgcCbSetShRegistersDirect, sceAgcCbSetShRegistersDirectGetSize, sceAgcCbSetUcRegistersDirect, sceAgcCbSetUcRegistersDirectGetSize,
        sceAgcSetCxRegIndirectPatchSetAddress, sceAgcSetShRegIndirectPatchSetAddress, sceAgcSetUcRegIndirectPatchSetAddress,
        sceAgcSetCxRegIndirectPatchSetNumRegisters, sceAgcSetShRegIndirectPatchSetNumRegisters, sceAgcSetUcRegIndirectPatchSetNumRegisters,
        sceAgcSetCxRegIndirectPatchAddRegisters, sceAgcSetShRegIndirectPatchAddRegisters, sceAgcSetUcRegIndirectPatchAddRegisters,
        sceAgcDcbSetIndexSize, sceAgcDcbSetIndexSizeGetSize, sceAgcDcbSetIndexBuffer, sceAgcDcbSetIndexBufferGetSize, sceAgcDcbSetIndexCount,
        sceAgcDcbSetIndexCountGetSize, sceAgcDcbSetNumInstances, sceAgcDcbSetNumInstancesGetSize, sceAgcDcbDrawIndex, sceAgcDcbDrawIndexGetSize,
        sceAgcDcbDrawIndexAuto, sceAgcDcbDrawIndexAutoGetSize, sceAgcDcbDrawIndexOffset, sceAgcDcbDrawIndexOffsetGetSize,
        sceAgcDcbDrawIndexMultiInstanced, sceAgcDcbDrawIndexMultiInstancedGetSize, sceAgcDcbSetBaseIndirectArgs, sceAgcDcbDrawIndexIndirect,
        sceAgcDcbDrawIndexIndirectGetSize, sceAgcDcbDrawIndirect, sceAgcDcbDrawIndirectGetSize, sceAgcDcbDrawIndirectMulti,
        sceAgcDcbDrawIndirectMultiGetSize, sceAgcDcbDrawIndexIndirectMulti, sceAgcDcbDrawIndexIndirectMultiGetSize,
        sceAgcCbDispatch, sceAgcCbDispatchGetSize, sceAgcDcbDispatchIndirect, sceAgcDcbDispatchIndirectGetSize, sceAgcAcbDispatchIndirect,
        sceAgcAcbDispatchIndirectGetSize,
        sceAgcDcbEventWrite, sceAgcDcbEventWriteGetSize, sceAgcAcbEventWrite, sceAgcAcbEventWriteGetSize, sceAgcDcbAcquireMem,
        sceAgcDcbAcquireMemGetSize, sceAgcAcbAcquireMem, sceAgcAcbAcquireMemGetSize, sceAgcCbReleaseMem, sceAgcCbReleaseMemGetSize,
        sceAgcCbQueueEndOfPipeActionGetSize, sceAgcQueueEndOfPipeActionPatchAddress, sceAgcQueueEndOfPipeActionPatchData,
        sceAgcDcbWaitRegMem, sceAgcDcbWaitOnAddressGetSize, sceAgcAcbWaitRegMem, sceAgcAcbWaitOnAddressGetSize,
        sceAgcWaitRegMemPatchAddress, sceAgcWaitRegMemPatchReference,
        sceAgcDcbCondExec, sceAgcDcbCondExecGetSize, sceAgcAcbCondExec, sceAgcAcbCondExecGetSize, sceAgcCondExecPatchSetEnd,
        sceAgcCondExecPatchSetCommandAddress,
        sceAgcDcbSetPredication, sceAgcSetPacketPredication, sceAgcSetRangePredication,
        sceAgcDcbWriteData, sceAgcDcbWriteDataGetSize, sceAgcAcbWriteData, sceAgcAcbWriteDataGetSize, sceAgcWriteDataPatchSetAddressOrOffset,
        sceAgcWriteDataPatchSetDst, sceAgcDcbCopyData, sceAgcDcbCopyDataGetSize, sceAgcAcbCopyData, sceAgcAcbCopyDataGetSize,
        sceAgcDcbDmaData, sceAgcDcbDmaDataGetSize, sceAgcAcbDmaData, sceAgcAcbDmaDataGetSize, sceAgcDmaDataPatchSetDstAddressOrOffset,
        sceAgcDmaDataPatchSetSrcAddressOrOffsetOrImmediate,
        sceAgcDcbJump, sceAgcDcbJumpGetSize, sceAgcAcbJump, sceAgcAcbJumpGetSize, sceAgcDcbRewind, sceAgcDcbRewindGetSize, sceAgcAcbRewind,
        sceAgcAcbRewindGetSize, sceAgcRewindPatchSetRewindState, sceAgcCbBranch, sceAgcCbBranchGetSize, sceAgcCbNop, sceAgcCbNopGetSize,
        sceAgcDcbGetLodStats, sceAgcDcbGetLodStatsGetSize, sceAgcGetPacketSize, sceAgcGetDataPacketPayloadRange, sceAgcSuspendPoint,
        sceAgcDebugRaiseException,
    ]);
    r.func_nid("libSceAgc", "-KRzWekV120", "-KRzWekV120 (unnamed: set VGT_INDEX_TYPE)", ds_agc_dcb_set_index_type as *const () as u64);
    r.func_nid("libSceAgc", "qj7QZpgr9Uw", "qj7QZpgr9Uw (unnamed: context state operation)", ds_agc_dcb_context_state_op as *const () as u64);
    r.func_nid("libSceAgc", "V++UgBtQhn0", "V++UgBtQhn0 (unnamed: data packet payload address)", ds_agc_get_data_packet_payload_address as *const () as u64);
}
