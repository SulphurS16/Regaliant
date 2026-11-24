//! libSceAmpr command buffers + libkernel APR (asynchronous file reads).
//!
//! The title builds command buffers in its own memory (`sceAmpr*CommandBuffer*`)
//! and submits them to the APR engine (`sceKernelAprSubmitCommandBuffer*`),
//! which reads files into memory, maps memory, writes completion values,
//! waits on memory/counters and signals event queues, strictly in order.
//!
//! Guest-visible contract (interface facts, see docs/PROVENANCE.md): the 0x18-byte
//! header {u32 type, u32 offset, u32 num_commands, u32 size, u64 buffer}, the APR
//! state words passed by pointer, and the per-command record sizes reported by
//! `sceAmprMeasureCommandSize*`. The record *contents* are this implementation's
//! own packed encoding (the title never decodes them); records are
//! self-describing so copies/reuse of buffers by the title stay correct.
//!
//! Execution: one in-order worker per submission priority. Interpretations that
//! are not confirmed against the title yet are marked "unconfirmed" and their
//! arguments are logged on first use.

use crate::hle::kernel::{errno::*, sce};
use crate::hle::{Registry, cstr};
use crate::mem::{self, MEM, PAGE, VmaKind};
use crate::{reg, sysv};
use parking_lot::{Condvar, Mutex};
use std::collections::{HashMap, VecDeque};
use std::fs::File;
use crate::platform::files::FileExt;
use std::sync::atomic::{AtomicU32, Ordering};
use std::sync::{Arc, OnceLock};

// ---------------------------------------------------------------------------
// Header layout and record sizes

const OFF_TYPE: u64 = 0x00;
const OFF_OFFSET: u64 = 0x04;
const OFF_NUM: u64 = 0x08;
const OFF_SIZE: u64 = 0x0C;
const OFF_DATA: u64 = 0x10;
const HEADER_SIZE: u64 = 0x18;
const APR_STATE0: u64 = 0x18;
const APR_STATE1: u64 = 0x20;
const SIZE_MAX: u32 = 64 << 20;
const TYPE_GS_VALID: u32 = 0x1_0000;
const TYPE_MAP_ACTIVE: u32 = 0x2_0000;

const REC_READ: u64 = 0x14;
const REC_READ_EXT: u64 = 0x18;
const REC_GATHER: u64 = 0x08;
const REC_GATHER_EXT: u64 = 0x0C;
const REC_SCATTER: u64 = 0x0C;
const REC_GS: u64 = 0x10;
const REC_GS_EXT: u64 = 0x14;
const REC_RESET_GS: u64 = 0x04;
const REC_KEVENT: u64 = 0x20;
const REC_FIXED: u64 = 0x20;
const REC_MAP_BEGIN: u64 = 0x0C;
const REC_MAP_DIRECT_BEGIN: u64 = 0x10;
const REC_MAP_END: u64 = 0x04;

const MAX_READ: u64 = 1 << 32;
const MAX_FILE_OFFSET: u64 = 1 << 40;
const MAX_ADDRESS: u64 = 0xF000_0000_0000;

/// Error codes returned as negative values through u64 "measure" results.
fn err64(e: i32) -> u64 {
    sce(e) as i64 as u64
}

mod op {
    pub const NOP: u8 = 1;
    pub const MARKER: u8 = 2;
    pub const READ: u8 = 3;
    pub const GATHER: u8 = 4;
    pub const SCATTER: u8 = 5;
    pub const GATHER_SCATTER: u8 = 6;
    pub const RESET_GS: u8 = 7;
    pub const MAP_BEGIN: u8 = 8;
    pub const MAP_DIRECT_BEGIN: u8 = 9;
    pub const MAP_END: u8 = 10;
    pub const WRITE_ADDR: u8 = 11;
    pub const WRITE_ADDR_TIME: u8 = 12;
    pub const WRITE_ADDR_COUNTER: u8 = 13;
    pub const WRITE_ADDR_COUNTER_PAIR: u8 = 14;
    pub const WRITE_COUNTER: u8 = 15;
    pub const WAIT_ADDR: u8 = 16;
    pub const WAIT_COUNTER: u8 = 17;
    pub const KEVENT: u8 = 18;
}

/// Little-endian bit packer for one record.
struct Bits {
    buf: Vec<u8>,
    pos: usize,
}

impl Bits {
    fn new(len: u64, opcode: u8) -> Bits {
        let mut b = Bits { buf: vec![0; len as usize], pos: 0 };
        b.put(8, opcode as u64);
        b
    }
    fn put(&mut self, n: usize, v: u64) {
        debug_assert!(self.pos + n <= self.buf.len() * 8, "record overflow");
        for i in 0..n {
            if (v >> i) & 1 != 0 {
                self.buf[(self.pos + i) / 8] |= 1 << ((self.pos + i) % 8);
            }
        }
        self.pos += n;
    }
}

struct BitReader<'a> {
    buf: &'a [u8],
    pos: usize,
}

impl BitReader<'_> {
    fn get(&mut self, n: usize) -> u64 {
        let mut v = 0u64;
        for i in 0..n {
            let p = self.pos + i;
            if p / 8 < self.buf.len() && (self.buf[p / 8] >> (p % 8)) & 1 != 0 {
                v |= 1 << i;
            }
        }
        self.pos += n;
        v
    }
}

// ---------------------------------------------------------------------------
// Guest memory helpers

unsafe fn rd32(a: u64) -> u32 {
    unsafe { std::ptr::read_volatile(a as *const u32) }
}
unsafe fn wr32(a: u64, v: u32) {
    unsafe { std::ptr::write_volatile(a as *mut u32, v) }
}
unsafe fn rd64(a: u64) -> u64 {
    unsafe { std::ptr::read_volatile(a as *const u64) }
}
unsafe fn wr64(a: u64, v: u64) {
    unsafe { std::ptr::write_volatile(a as *mut u64, v) }
}

/// Appends a record to the command buffer at `cb` (space-checked).
fn append(cb: u64, rec: &[u8]) -> i32 {
    if cb == 0 {
        return sce(EINVAL);
    }
    unsafe {
        let data = rd64(cb + OFF_DATA);
        let size = rd32(cb + OFF_SIZE) as u64;
        let off = rd32(cb + OFF_OFFSET) as u64;
        if data == 0 {
            return sce(EPERM);
        }
        if off + rec.len() as u64 > size {
            crate::warn!("ampr: command buffer {cb:#x} full ({off:#x}+{:#x} > {size:#x})", rec.len());
            return sce(EBUSY);
        }
        std::ptr::copy_nonoverlapping(rec.as_ptr(), (data + off) as *mut u8, rec.len());
        wr32(cb + OFF_OFFSET, (off + rec.len() as u64) as u32);
        wr32(cb + OFF_NUM, rd32(cb + OFF_NUM) + 1);
    }
    0
}

fn set_type_bits(cb: u64, set: u32, clear: u32) {
    unsafe { wr32(cb + OFF_TYPE, (rd32(cb + OFF_TYPE) | set) & !clear) }
}

fn once_log(key: &'static str, text: String) {
    static SEEN: Mutex<Option<std::collections::HashSet<&'static str>>> = Mutex::new(None);
    if SEEN.lock().get_or_insert_with(Default::default).insert(key) {
        crate::info!("ampr: {key}: {text}");
    }
}

fn valid_range(dest: u64, size: u64) -> bool {
    size != 0 && size <= MAX_READ && dest != 0 && dest.checked_add(size).is_some_and(|e| e <= MAX_ADDRESS)
}

// ---------------------------------------------------------------------------
// File ids (APR resolves paths up front; reads refer to ids)

struct AprFile {
    path: String,
    /// Its host file, or its node in the game's archive.
    at: crate::hle::kernel::fs::Found,
    size: u64,
    handle: Mutex<Option<Arc<File>>>,
    /// Content the port serves from memory (`crate::pcsettings`: title files it extends).
    mem: Option<Arc<Vec<u8>>>,
}

/// Registered files: ids index `files` (id - 1). One lock guards both maps.
#[derive(Default)]
struct FileTable {
    files: Vec<Arc<AprFile>>,
    by_path: HashMap<String, u32>,
}

static FILES: Mutex<Option<FileTable>> = Mutex::new(None);

/// File ids start at 1 (0 is never valid).
fn resolve_path(guest: &str) -> Result<(u32, u64), i32> {
    {
        let mut g = FILES.lock();
        let t = g.get_or_insert_with(Default::default);
        if let Some(&id) = t.by_path.get(guest) {
            return Ok((id, t.files[id as usize - 1].size));
        }
    }
    // Filesystem queries run without the lock (cached listings for the
    // read-only game tree).
    let Some((at, md)) = crate::hle::kernel::fs::guest_meta(guest) else {
        return Err(ENOENT);
    };
    if md.is_dir {
        return Err(EISDIR);
    }
    let mem = crate::platform::file_hooks::generated(guest);
    let mut g = FILES.lock();
    let t = g.get_or_insert_with(Default::default);
    // Another thread may have registered the path meanwhile.
    if let Some(&id) = t.by_path.get(guest) {
        return Ok((id, t.files[id as usize - 1].size));
    }
    if let Some(d) = &mem {
        crate::info!("apr: '{guest}' served from the port's extended copy ({} bytes)", d.len());
    }
    t.files.push(Arc::new(AprFile { path: guest.to_string(), at, size: md.size, handle: Mutex::new(None), mem }));
    let id = t.files.len() as u32;
    t.by_path.insert(guest.to_string(), id);
    Ok((id, md.size))
}

fn file(id: u32) -> Option<Arc<AprFile>> {
    FILES.lock().as_ref()?.files.get((id as usize).wrapping_sub(1)).cloned()
}

fn open_handle(f: &AprFile) -> Result<Arc<File>, i32> {
    let mut h = f.handle.lock();
    if let Some(x) = h.as_ref() {
        return Ok(x.clone());
    }
    let crate::hle::kernel::fs::Found::Host(host) = &f.at else { return Err(EIO) };
    let file = Arc::new(File::open(host).map_err(|e| {
        crate::warn!("apr: open {} ({}) failed: {e}", f.path, host.display());
        EIO
    })?);
    *h = Some(file.clone());
    Ok(file)
}

/// Common worker of the sceKernelAprResolveFilepaths* family.
/// Returns the number of resolved paths, or a negative SCE error with
/// `*error_index` set to the failing path.
unsafe fn resolve_list(prefix: Option<&str>, paths: *const u64, count: u32, ids: *mut u32, sizes: *mut u64, error_index: *mut u32, for_each: Option<*mut i32>) -> i32 {
    if paths.is_null() || ids.is_null() {
        return sce(EINVAL);
    }
    let mut first_err = 0;
    unsafe {
        for i in 0..count as usize {
            let p = cstr(*paths.add(i) as *const u8);
            let full = match prefix {
                Some(pre) => format!("{pre}{p}"),
                None => p.to_string(),
            };
            match resolve_path(&full) {
                Ok((id, size)) => {
                    if crate::hle::iostat::note("apr", &full, crate::hle::iostat::What::Open, 0) {
                        crate::info!("apr: resolved '{full}' -> id {id} ({size:#x} bytes)");
                    }
                    *ids.add(i) = id;
                    if !sizes.is_null() {
                        *sizes.add(i) = size;
                    }
                    if let Some(r) = for_each {
                        *r.add(i) = 0;
                    }
                }
                Err(e) => {
                    if crate::hle::iostat::note("apr", &full, crate::hle::iostat::What::Fail, 0) {
                        crate::info!("apr: resolve '{full}' failed ({e})");
                    } else {
                        crate::debug!("apr: resolve '{full}' failed ({e})");
                    }
                    *ids.add(i) = 0;
                    if !sizes.is_null() {
                        *sizes.add(i) = 0;
                    }
                    match for_each {
                        Some(r) => {
                            *r.add(i) = sce(e);
                            if first_err == 0 {
                                first_err = sce(e);
                            }
                        }
                        None => {
                            if !error_index.is_null() {
                                *error_index = i as u32;
                            }
                            return sce(e);
                        }
                    }
                }
            }
        }
    }
    // The *ForEach variants report per-path results and succeed overall.
    let _ = first_err;
    0
}

// ---------------------------------------------------------------------------
// Execution engine

static COUNTERS: [AtomicU32; 256] = [const { AtomicU32::new(0) }; 256];

struct Submission {
    id: u32,
    cb: u64,
    records: Vec<u8>,
    result: u64,
}

struct Queue {
    pending: Mutex<VecDeque<Submission>>,
    cv: Condvar,
}

static QUEUES: Mutex<Vec<&'static Queue>> = Mutex::new(Vec::new());
static NEXT_ID: AtomicU32 = AtomicU32::new(1);
static DONE: Mutex<Option<HashMap<u32, (i32, u32)>>> = Mutex::new(None);
static DONE_CV: Condvar = Condvar::new();
pub static STATS: [std::sync::atomic::AtomicU64; 3] = [const { std::sync::atomic::AtomicU64::new(0) }; 3]; // submissions, reads, bytes

fn queue(prio: u32) -> &'static Queue {
    let prio = prio.min(15) as usize;
    let mut qs = QUEUES.lock();
    while qs.len() <= prio {
        let q: &'static Queue = Box::leak(Box::new(Queue { pending: Mutex::new(VecDeque::new()), cv: Condvar::new() }));
        let n = qs.len();
        crate::thread::spawn_host(&format!("apr{n}"), 1 << 20, Box::new(move || worker(n, q))).expect("apr worker thread");
        qs.push(q);
    }
    qs[prio]
}

fn submit(cb: u64, prio: u32, result: u64) -> Result<u32, i32> {
    if cb == 0 {
        return Err(EINVAL);
    }
    let (data, len) = unsafe { (rd64(cb + OFF_DATA), rd32(cb + OFF_OFFSET) as usize) };
    if data == 0 {
        return Err(EINVAL);
    }
    // Snapshot: the title may reset and refill the buffer once submitted.
    let records = unsafe { std::slice::from_raw_parts(data as *const u8, len).to_vec() };
    let mut id = NEXT_ID.fetch_add(1, Ordering::Relaxed);
    if id == 0 {
        id = NEXT_ID.fetch_add(1, Ordering::Relaxed);
    }
    STATS[0].fetch_add(1, Ordering::Relaxed);
    let q = queue(prio);
    q.pending.lock().push_back(Submission { id, cb, records, result });
    q.cv.notify_one();
    Ok(id)
}

fn worker(n: usize, q: &'static Queue) {
    crate::log::set_thread_name(&format!("apr{n}"));
    crate::platform::loader_hooks::arm_watch_current_thread(false);
    loop {
        let s = {
            let mut g = q.pending.lock();
            loop {
                if let Some(s) = g.pop_front() {
                    break s;
                }
                q.cv.wait(&mut g);
            }
        };
        let (res, err_off) = execute(&s);
        if s.result != 0 {
            unsafe {
                wr32(s.result, res as u32);
                wr32(s.result + 4, err_off);
            }
        }
        if res != 0 {
            crate::warn!("apr: submission {} (cb {:#x}) failed {res:#x} at record offset {err_off:#x}", s.id, s.cb);
        }
        DONE.lock().get_or_insert_with(HashMap::new).insert(s.id, (res, err_off));
        DONE_CV.notify_all();
    }
}

#[derive(Default)]
struct GsState {
    file: u32,
    dest: u64,
    offset: u64,
}

fn read_into(file_id: u32, dest: u64, size: u64, offset: u64) -> Result<(), i32> {
    let f = file(file_id).ok_or(EBADF)?;
    if offset.checked_add(size).is_none_or(|e| e > f.size) {
        crate::warn!("apr: read of {} [{offset:#x}+{size:#x}] past end of file ({:#x})", f.path, f.size);
        return Err(EINVAL);
    }
    if let Some(d) = &f.mem {
        let _w = crate::platform::file_hooks::host_write(dest, size);
        let what = || format!("apr: read {} [{offset:#x}] into {dest:#x} (memory)", f.path);
        let src = &d[offset as usize..(offset + size) as usize];
        crate::mem::kernel_io(&what, dest, size, true, || {
            unsafe { std::ptr::copy_nonoverlapping(src.as_ptr(), dest as *mut u8, src.len()) };
            Ok(())
        })
        .map_err(|_| EIO)?;
        STATS[1].fetch_add(1, Ordering::Relaxed);
        STATS[2].fetch_add(size, Ordering::Relaxed);
        return Ok(());
    }
    if let crate::hle::kernel::fs::Found::Zar(a, node) = &f.at {
        if crate::hle::iostat::note("apr", &f.path, crate::hle::iostat::What::Read, size) {
            crate::info!("apr: read {} [{offset:#x}+{size:#x}] -> {dest:#x} (archive)", f.path);
        }
        let buf = unsafe { std::slice::from_raw_parts_mut(dest as *mut u8, size as usize) };
        match crate::hle::kernel::fs::read_archive_into(&f.path, a, *node, buf, offset) {
            Ok(n) if n as u64 == size => {}
            _ => return Err(EIO),
        }
        STATS[1].fetch_add(1, Ordering::Relaxed);
        STATS[2].fetch_add(size, Ordering::Relaxed);
        return Ok(());
    }
    let h = open_handle(&f)?;
    if crate::hle::iostat::note("apr", &f.path, crate::hle::iostat::What::Read, size) {
        crate::info!("apr: read {} [{offset:#x}+{size:#x}] -> {dest:#x}", f.path);
    }
    // The kernel writes the buffer: write-tracked pages cannot fault.
    let _w = crate::platform::file_hooks::host_write(dest, size);
    crate::platform::file_hooks::trace_writes(|| format!("apr read of {} [{offset:#x}]", f.path), dest, size);
    let buf = unsafe { std::slice::from_raw_parts_mut(dest as *mut u8, size as usize) };
    let mut done = 0usize;
    while done < buf.len() {
        let what = || format!("apr: read {} [{:#x}] into {:#x}", f.path, offset + done as u64, dest + done as u64);
        match crate::mem::kernel_io(&what, dest + done as u64, (buf.len() - done) as u64, true, || h.seek_read(&mut buf[done..], offset + done as u64)) {
            Ok(0) => return Err(EIO),
            Ok(n) => done += n,
            Err(e) => {
                crate::warn!("apr: read {} failed: {e}", f.path);
                return Err(EIO);
            }
        }
    }
    STATS[1].fetch_add(1, Ordering::Relaxed);
    STATS[2].fetch_add(size, Ordering::Relaxed);
    Ok(())
}

fn compare(value: u64, reference: u64, cmp: u8, bits: u32) -> bool {
    // Unconfirmed encoding: 0 ==, 1 >, 2 <, 3 !=, 4 wrap-around >=, 5 signed >, 6 signed <.
    let sign = 1u64 << (bits - 1);
    match cmp {
        0 => value == reference,
        1 => value > reference,
        2 => value < reference,
        3 => value != reference,
        4 => value.wrapping_sub(reference) & sign == 0,
        5 => (value ^ sign) > (reference ^ sign),
        _ => (value ^ sign) < (reference ^ sign),
    }
}

/// Runs one submission. Returns (result, failing record offset).
fn execute(s: &Submission) -> (i32, u32) {
    let r = &s.records;
    let mut pos = 0usize;
    let mut gs = GsState::default();
    while pos < r.len() {
        let opc = r[pos];
        let at = pos as u32;
        let take = |len: u64| r.get(pos..pos + len as usize).ok_or(EINVAL);
        let out: Result<u64, i32> = (|| {
            match opc {
                op::NOP | op::MARKER => {
                    let w = u32::from_le_bytes(take(4)?.try_into().unwrap());
                    Ok(((w >> 8) as u64).max(1) * 4)
                }
                op::READ => {
                    let ext = r[pos + 1] & 1 != 0;
                    let len = if ext { REC_READ_EXT } else { REC_READ };
                    let mut b = BitReader { buf: take(len)?, pos: 16 };
                    let size = b.get(33);
                    let dest = b.get(48);
                    let file = b.get(32) as u32;
                    let offset = b.get(if ext { 40 } else { 32 });
                    read_into(file, dest, size, offset)?;
                    gs = GsState { file, dest: dest + size, offset: offset + size };
                    Ok(len)
                }
                op::GATHER => {
                    let ext = r[pos + 1] & 1 != 0;
                    let len = if ext { REC_GATHER_EXT } else { REC_GATHER };
                    let mut b = BitReader { buf: take(len)?, pos: 16 };
                    let size = b.get(33);
                    let offset = b.get(if ext { 40 } else { 18 });
                    read_into(gs.file, gs.dest, size, offset)?;
                    gs.dest += size;
                    gs.offset = offset + size;
                    Ok(len)
                }
                op::SCATTER => {
                    let mut b = BitReader { buf: take(REC_SCATTER)?, pos: 16 };
                    let dest = b.get(48);
                    let size = b.get(33);
                    read_into(gs.file, dest, size, gs.offset)?;
                    gs.dest = dest + size;
                    gs.offset += size;
                    Ok(REC_SCATTER)
                }
                op::GATHER_SCATTER => {
                    let ext = r[pos + 1] & 1 != 0;
                    let len = if ext { REC_GS_EXT } else { REC_GS };
                    let mut b = BitReader { buf: take(len)?, pos: 16 };
                    let dest = b.get(48);
                    let size = b.get(33);
                    let offset = b.get(if ext { 40 } else { 32 });
                    read_into(gs.file, dest, size, offset)?;
                    gs.dest = dest + size;
                    gs.offset = offset + size;
                    Ok(len)
                }
                op::RESET_GS => {
                    gs = GsState::default();
                    Ok(REC_RESET_GS)
                }
                op::MAP_BEGIN | op::MAP_DIRECT_BEGIN => {
                    let direct = opc == op::MAP_DIRECT_BEGIN;
                    let len = if direct { REC_MAP_DIRECT_BEGIN } else { REC_MAP_BEGIN };
                    let mut b = BitReader { buf: take(len)?, pos: 8 };
                    let va = b.get(34) << 14;
                    let size = b.get(30) << 14;
                    let mtype = b.get(8) as i32;
                    let prot = b.get(10) as u32;
                    let dmem = if direct { b.get(30) << 14 } else { 0 };
                    let (kind, name) = if direct { (VmaKind::Direct, "apr-map-direct") } else { (VmaKind::Flexible, "apr-map") };
                    MEM.lock().map(va, size, kind, prot | mem::PROT_CPU_READ | mem::PROT_CPU_WRITE, dmem, mtype, name).map_err(|e| {
                        crate::warn!("apr: map {va:#x}+{size:#x} failed: {e:?}");
                        ENOMEM
                    })?;
                    Ok(len)
                }
                op::MAP_END => Ok(REC_MAP_END),
                op::WRITE_ADDR | op::WRITE_ADDR_TIME | op::WRITE_ADDR_COUNTER | op::WRITE_ADDR_COUNTER_PAIR => {
                    let rec = take(REC_FIXED)?;
                    let a = u64::from_le_bytes(rec[8..16].try_into().unwrap());
                    let v = u64::from_le_bytes(rec[16..24].try_into().unwrap());
                    let value = match opc {
                        op::WRITE_ADDR => v,
                        op::WRITE_ADDR_TIME => unsafe { core::arch::x86_64::_rdtsc() },
                        op::WRITE_ADDR_COUNTER => COUNTERS[rec[1] as usize].load(Ordering::Acquire) as u64,
                        _ => {
                            let lo = COUNTERS[rec[1] as usize].load(Ordering::Acquire) as u64;
                            let hi = COUNTERS[rec[1].wrapping_add(1) as usize].load(Ordering::Acquire) as u64;
                            lo | (hi << 32)
                        }
                    };
                    unsafe { wr64(a, value) };
                    Ok(REC_FIXED)
                }
                op::WRITE_COUNTER => {
                    let rec = take(REC_FIXED)?;
                    let idx = rec[1] as usize;
                    let v = u64::from_le_bytes(rec[16..24].try_into().unwrap()) as u32;
                    COUNTERS[idx].store(v, Ordering::Release);
                    Ok(REC_FIXED)
                }
                op::WAIT_ADDR | op::WAIT_COUNTER => {
                    let rec = take(REC_FIXED)?;
                    let cmp = rec[2];
                    let a = u64::from_le_bytes(rec[8..16].try_into().unwrap());
                    let v = u64::from_le_bytes(rec[16..24].try_into().unwrap());
                    let mut spins = 0u64;
                    let start = std::time::Instant::now();
                    let mut warned = false;
                    loop {
                        let cur = if opc == op::WAIT_ADDR { unsafe { rd64(a) } } else { COUNTERS[rec[1] as usize].load(Ordering::Acquire) as u64 };
                        let bits = if opc == op::WAIT_ADDR { 64 } else { 32 };
                        if compare(cur, v, cmp, bits) {
                            break;
                        }
                        spins += 1;
                        if spins < 64 {
                            std::hint::spin_loop();
                        } else {
                            std::thread::sleep(std::time::Duration::from_micros(200));
                        }
                        if !warned && start.elapsed() > std::time::Duration::from_secs(5) {
                            warned = true;
                            crate::warn!("apr: submission {} waiting > 5 s on {} {a:#x} (cmp {cmp}, want {v:#x}, have {cur:#x})", s.id, if opc == op::WAIT_ADDR { "address" } else { "counter" });
                        }
                    }
                    Ok(REC_FIXED)
                }
                op::KEVENT => {
                    let rec = take(REC_KEVENT)?;
                    let eq = u64::from_le_bytes(rec[8..16].try_into().unwrap());
                    let id = i32::from_le_bytes(rec[16..20].try_into().unwrap());
                    let data = u64::from_le_bytes(rec[24..32].try_into().unwrap());
                    match unsafe { crate::hle::kernel::equeue::from_handle(eq) } {
                        Some(q) => {
                            if !q.trigger(id as u64, crate::hle::kernel::equeue::EVFILT_USER, Some(data), None) {
                                crate::warn!("apr: event queue {eq:#x} has no user event {id}");
                            }
                        }
                        None => crate::warn!("apr: invalid event queue {eq:#x}"),
                    }
                    Ok(REC_KEVENT)
                }
                other => {
                    crate::error!("apr: unknown record opcode {other} at offset {pos:#x} in submission {} (cb {:#x})", s.id, s.cb);
                    Err(EINVAL)
                }
            }
        })();
        match out {
            Ok(n) => pos += n as usize,
            Err(e) => return (sce(e), at),
        }
    }
    (0, 0)
}

fn wait_done(id: u32) -> Result<(i32, u32), i32> {
    let mut g = DONE.lock();
    let _w = crate::thread::waiting("apr completion", id as u64, None);
    loop {
        if let Some(r) = g.get_or_insert_with(HashMap::new).remove(&id) {
            return Ok(r);
        }
        if id == 0 || id >= NEXT_ID.load(Ordering::Relaxed) {
            return Err(ESRCH);
        }
        DONE_CV.wait(&mut g);
    }
}

// ---------------------------------------------------------------------------
// Record builders

fn fixed(cb: u64, opc: u8, b1: u8, b2: u8, b3: u8, addr: u64, value: u64, extra: u64) -> i32 {
    let mut rec = [0u8; REC_FIXED as usize];
    rec[0] = opc;
    rec[1] = b1;
    rec[2] = b2;
    rec[3] = b3;
    rec[8..16].copy_from_slice(&addr.to_le_bytes());
    rec[16..24].copy_from_slice(&value.to_le_bytes());
    rec[24..32].copy_from_slice(&extra.to_le_bytes());
    append(cb, &rec)
}

fn read_rec(file: u32, dest: u64, size: u64, offset: u64) -> Vec<u8> {
    let ext = offset >> 32 != 0;
    let mut b = Bits::new(if ext { REC_READ_EXT } else { REC_READ }, op::READ);
    b.put(8, ext as u64);
    b.put(33, size);
    b.put(48, dest);
    b.put(32, file as u64);
    b.put(if ext { 40 } else { 32 }, offset);
    b.buf
}

fn noop_len(cb: u64, opc: u8, len: u64) -> i32 {
    if len > 1 << 20 {
        return sce(EINVAL);
    }
    let len = (len + 3) & !3;
    let mut rec = vec![0u8; len as usize];
    rec[..4].copy_from_slice(&((opc as u32) | (((len / 4) as u32) << 8)).to_le_bytes());
    append(cb, &rec)
}

fn marker_size(msg: *const u8, color: bool) -> u64 {
    let n = if msg.is_null() { 0 } else { unsafe { cstr(msg).len() as u64 + 1 } };
    ((if color { 8 } else { 4 }) + n + 3) & !3
}

fn gs_state_ok(cb: u64) -> bool {
    unsafe { rd32(cb + OFF_TYPE) & TYPE_GS_VALID != 0 }
}

fn map_rec(cb: u64, direct: bool, va: u64, dmem: u64, size: u64, mtype: i32, prot: i32) -> i32 {
    if va == 0 || size == 0 || va % PAGE != 0 || size % PAGE != 0 || dmem % PAGE != 0 {
        return sce(EINVAL);
    }
    let mut b = Bits::new(if direct { REC_MAP_DIRECT_BEGIN } else { REC_MAP_BEGIN }, if direct { op::MAP_DIRECT_BEGIN } else { op::MAP_BEGIN });
    b.put(34, va >> 14);
    b.put(30, size >> 14);
    b.put(8, mtype as u64 & 0xFF);
    b.put(10, prot as u64 & 0x3FF);
    if direct {
        b.put(30, dmem >> 14);
    }
    let r = append(cb, &b.buf);
    if r == 0 {
        set_type_bits(cb, TYPE_MAP_ACTIVE, 0);
    }
    r
}

sysv! {
    // ---- object lifetime / buffer management
    fn sceAmprCommandBufferConstructor(cb: u64) -> i32 {
        if cb != 0 { std::ptr::write_bytes(cb as *mut u8, 0, HEADER_SIZE as usize); }
        0
    }
    fn sceAmprCommandBufferDestructor(cb: u64) -> i32 { 0 }
    fn sceAmprAprCommandBufferConstructor(cb: u64, s0: u64, s1: u64) -> i32 {
        if cb == 0 { return 0; }
        let s0 = if s0 == 0 { cb + APR_STATE0 } else { s0 };
        let s1 = if s1 == 0 { cb + APR_STATE1 } else { s1 };
        wr64(s0, 0);
        wr64(s1, 0);
        0
    }
    fn sceAmprAprCommandBufferDestructor(cb: u64) -> i32 { 0 }
    fn sceAmprCommandBufferSetBuffer(cb: u64, buffer: u64, size: u32) -> i32 {
        if cb == 0 || buffer == 0 || buffer % 4 != 0 || size == 0 || size % 4 != 0 || size > SIZE_MAX { return sce(EINVAL); }
        if rd64(cb + OFF_DATA) != 0 { return sce(EBUSY); }
        wr64(cb + OFF_DATA, buffer);
        wr32(cb + OFF_SIZE, size);
        wr32(cb + OFF_OFFSET, 0);
        wr32(cb + OFF_NUM, 0);
        0
    }
    fn sceAmprCommandBufferReset(cb: u64) -> i32 {
        if cb == 0 || rd64(cb + OFF_DATA) == 0 { return sce(EPERM); }
        wr32(cb + OFF_OFFSET, 0);
        wr32(cb + OFF_NUM, 0);
        set_type_bits(cb, 0, TYPE_GS_VALID | TYPE_MAP_ACTIVE);
        0
    }
    fn sceAmprCommandBufferClearBuffer(cb: u64) -> u64 {
        if cb == 0 { return 0; }
        let b = rd64(cb + OFF_DATA);
        wr64(cb + OFF_DATA, 0);
        wr32(cb + OFF_SIZE, 0);
        wr32(cb + OFF_OFFSET, 0);
        wr32(cb + OFF_NUM, 0);
        b
    }
    fn sceAmprCommandBufferGetType(cb: u64) -> i32 { if cb == 0 { 0 } else { rd32(cb + OFF_TYPE) as i32 } }
    fn sceAmprCommandBufferGetSize(cb: u64) -> u64 { if cb == 0 { 0 } else { rd32(cb + OFF_SIZE) as u64 } }
    fn sceAmprCommandBufferGetBufferBaseAddress(cb: u64) -> u64 { if cb == 0 { 0 } else { rd64(cb + OFF_DATA) } }
    fn sceAmprCommandBufferGetNumCommands(cb: u64) -> u32 { if cb == 0 { 0 } else { rd32(cb + OFF_NUM) } }
    fn sceAmprCommandBufferGetCurrentOffset(cb: u64) -> u64 { if cb == 0 { 0 } else { rd32(cb + OFF_OFFSET) as u64 } }

    // ---- generic commands
    fn sceAmprCommandBufferNop(cb: u64, num_u32: u32) -> i32 { noop_len(cb, op::NOP, (num_u32 as u64 * 4).max(4)) }
    fn sceAmprCommandBufferNopWithData(cb: u64, num_u32: u32, data: *const u32) -> i32 { noop_len(cb, op::NOP, (num_u32 as u64 + 1) * 4) }
    fn sceAmprCommandBufferConstructNop(cb: u64, num_u32: u32, data: *const u8, bytes: u32) -> i32 {
        once_log("ConstructNop", format!("cb {cb:#x} num {num_u32} data {data:p} bytes {bytes}"));
        noop_len(cb, op::NOP, 4 + bytes as u64)
    }
    fn sceAmprCommandBufferConstructMarker(cb: u64, kind: u32, msg: *const u8, color: *const u32) -> i32 {
        noop_len(cb, op::MARKER, marker_size(msg, !color.is_null()))
    }
    fn sceAmprCommandBufferSetMarker(cb: u64, msg: *const u8) -> i32 { noop_len(cb, op::MARKER, marker_size(msg, false)) }
    fn sceAmprCommandBufferSetMarkerWithColor(cb: u64, msg: *const u8, color: u32) -> i32 { noop_len(cb, op::MARKER, marker_size(msg, true)) }
    fn sceAmprCommandBufferPushMarker(cb: u64, msg: *const u8) -> i32 { noop_len(cb, op::MARKER, marker_size(msg, false)) }
    fn sceAmprCommandBufferPushMarkerWithColor(cb: u64, msg: *const u8, color: u32) -> i32 { noop_len(cb, op::MARKER, marker_size(msg, true)) }
    fn sceAmprCommandBufferPopMarker(cb: u64) -> i32 { noop_len(cb, op::MARKER, 4) }

    fn sceAmprCommandBufferWriteAddress_04_00(cb: u64, addr: u64, value: u64, on_completion: u32) -> i32 {
        if addr == 0 || addr % 8 != 0 { return sce(EINVAL); }
        fixed(cb, op::WRITE_ADDR, on_completion as u8, 0, 0, addr, value, 0)
    }
    fn sceAmprCommandBufferWriteAddressFromTimeCounter_04_00(cb: u64, addr: u64, on_completion: u32) -> i32 {
        if addr == 0 || addr % 8 != 0 { return sce(EINVAL); }
        once_log("WriteAddressFromTimeCounter", format!("addr {addr:#x} flag {on_completion} (writes the host TSC; unconfirmed time base)"));
        fixed(cb, op::WRITE_ADDR_TIME, 0, 0, 0, addr, 0, 0)
    }
    fn sceAmprCommandBufferWriteAddressFromCounter_04_00(cb: u64, addr: u64, index: u32, on_completion: u32) -> i32 {
        if addr == 0 || addr % 8 != 0 || index > 255 { return sce(EINVAL); }
        once_log("WriteAddressFromCounter", format!("addr {addr:#x} counter {index} flag {on_completion} (unconfirmed)"));
        fixed(cb, op::WRITE_ADDR_COUNTER, index as u8, 0, 0, addr, 0, 0)
    }
    fn sceAmprCommandBufferWriteAddressFromCounterPair_04_00(cb: u64, addr: u64, index: u32, on_completion: u32) -> i32 {
        if addr == 0 || addr % 8 != 0 || index > 254 { return sce(EINVAL); }
        once_log("WriteAddressFromCounterPair", format!("addr {addr:#x} counter {index} flag {on_completion} (unconfirmed)"));
        fixed(cb, op::WRITE_ADDR_COUNTER_PAIR, index as u8, 0, 0, addr, 0, 0)
    }
    fn sceAmprCommandBufferWriteCounter_04_00(cb: u64, index: u32, access: u32, value: u64, op_: u32, on_completion: u32) -> i32 {
        if index > 255 { return sce(EINVAL); }
        once_log("WriteCounter", format!("counter {index} access {access} value {value:#x} op {op_} flag {on_completion} (unconfirmed: stores value)"));
        fixed(cb, op::WRITE_COUNTER, index as u8, access as u8, op_ as u8, 0, value, 0)
    }
    fn sceAmprCommandBufferWaitOnAddress_04_00(cb: u64, addr: u64, value: u64, cmp: u32, flags: u32) -> i32 {
        if addr == 0 || addr % 8 != 0 || cmp > 6 { return sce(EINVAL); }
        once_log("WaitOnAddress", format!("addr {addr:#x} value {value:#x} cmp {cmp} flags {flags} (unconfirmed compare encoding)"));
        fixed(cb, op::WAIT_ADDR, 0, cmp as u8, flags as u8, addr, value, 0)
    }
    fn sceAmprCommandBufferWaitOnCounter_04_00(cb: u64, index: u32, access: u32, value: u64, cmp: u32, a5: u32, a6: u32, a7: u32) -> i32 {
        if index > 255 || cmp > 6 { return sce(EINVAL); }
        once_log("WaitOnCounter", format!("counter {index} access {access} value {value:#x} cmp {cmp} args {a5} {a6} {a7} (unconfirmed)"));
        fixed(cb, op::WAIT_COUNTER, index as u8, cmp as u8, access as u8, 0, value, 0)
    }
    fn sceAmprCommandBufferWriteKernelEventQueue_04_00(cb: u64, eq: u64, id: u64, data: u64, a4: u64, a5: u64) -> i32 {
        once_log("WriteKernelEventQueue", format!("eq {eq:#x} id {id:#x} data {data:#x} args {a4:#x} {a5:#x} (triggers EVFILT_USER id)"));
        let mut rec = [0u8; REC_KEVENT as usize];
        rec[0] = op::KEVENT;
        rec[8..16].copy_from_slice(&eq.to_le_bytes());
        rec[16..20].copy_from_slice(&(id as i32).to_le_bytes());
        rec[24..32].copy_from_slice(&data.to_le_bytes());
        append(cb, &rec)
    }

    // ---- APR reads and maps
    fn sceAmprAprCommandBufferReadFile(cb: u64, s0: u64, s1: u64, file: u32, dest: u64, size: u64, offset: u64) -> i32 {
        if !valid_range(dest, size) || offset >= MAX_FILE_OFFSET { return sce(EINVAL); }
        let r = append(cb, &read_rec(file, dest, size, offset));
        if r == 0 { set_type_bits(cb, TYPE_GS_VALID, 0); }
        r
    }
    fn sceAmprAprCommandBufferReadFileGather(cb: u64, s0: u64, s1: u64, size: u64, offset: u64) -> i32 {
        if size == 0 || size > MAX_READ || offset >= MAX_FILE_OFFSET { return sce(EINVAL); }
        if !gs_state_ok(cb) { return sce(EPERM); }
        let ext = offset > 0x3_FFFF;
        let mut b = Bits::new(if ext { REC_GATHER_EXT } else { REC_GATHER }, op::GATHER);
        b.put(8, ext as u64);
        b.put(33, size);
        b.put(if ext { 40 } else { 18 }, offset);
        append(cb, &b.buf)
    }
    fn sceAmprAprCommandBufferReadFileScatter(cb: u64, s0: u64, s1: u64, dest: u64, size: u64) -> i32 {
        if !valid_range(dest, size) { return sce(EINVAL); }
        if !gs_state_ok(cb) { return sce(EPERM); }
        let mut b = Bits::new(REC_SCATTER, op::SCATTER);
        b.put(8, 0);
        b.put(48, dest);
        b.put(33, size);
        append(cb, &b.buf)
    }
    fn sceAmprAprCommandBufferReadFileGatherScatter(cb: u64, s0: u64, s1: u64, dest: u64, size: u64, offset: u64) -> i32 {
        if !valid_range(dest, size) || offset >= MAX_FILE_OFFSET { return sce(EINVAL); }
        if !gs_state_ok(cb) { return sce(EPERM); }
        let ext = offset >> 32 != 0;
        let mut b = Bits::new(if ext { REC_GS_EXT } else { REC_GS }, op::GATHER_SCATTER);
        b.put(8, ext as u64);
        b.put(48, dest);
        b.put(33, size);
        b.put(if ext { 40 } else { 32 }, offset);
        append(cb, &b.buf)
    }
    fn sceAmprAprCommandBufferResetGatherScatterState(cb: u64, s0: u64, s1: u64) -> i32 {
        let r = noop_len(cb, op::RESET_GS, REC_RESET_GS);
        if r == 0 { set_type_bits(cb, 0, TYPE_GS_VALID); }
        r
    }
    fn sceAmprAprCommandBufferMapBegin(cb: u64, s0: u64, s1: u64, va: u64, size: u64, mtype: i32, prot: i32) -> i32 {
        once_log("MapBegin", format!("va {va:#x} size {size:#x} type {mtype} prot {prot:#x} (maps flexible memory, unconfirmed)"));
        map_rec(cb, false, va, 0, size, mtype, prot)
    }
    fn sceAmprAprCommandBufferMapDirectBegin(cb: u64, s0: u64, s1: u64, va: u64, dmem: u64, size: u64, mtype: i32, prot: i32) -> i32 {
        once_log("MapDirectBegin", format!("va {va:#x} dmem {dmem:#x} size {size:#x} type {mtype} prot {prot:#x} (unconfirmed)"));
        map_rec(cb, true, va, dmem, size, mtype, prot)
    }
    fn sceAmprAprCommandBufferMapEnd(cb: u64, s0: u64, s1: u64) -> i32 {
        if cb == 0 || rd32(cb + OFF_TYPE) & TYPE_MAP_ACTIVE == 0 { return sce(EPERM); }
        let r = noop_len(cb, op::MAP_END, REC_MAP_END);
        if r == 0 { set_type_bits(cb, 0, TYPE_MAP_ACTIVE); }
        r
    }

    // ---- sizes
    fn sceAmprMeasureCommandSizeNop(num_u32: u32) -> u64 { if num_u32 == 0 { 4 } else { num_u32 as u64 * 4 } }
    fn sceAmprMeasureCommandSizeNopWithData(num_u32: u32, data: *const u32) -> u64 { (num_u32 as u64 + 1) * 4 }
    fn sceAmprMeasureCommandSizeSetMarker(msg: *const u8) -> u64 { marker_size(msg, false) }
    fn sceAmprMeasureCommandSizeSetMarkerWithColor(msg: *const u8, color: u32) -> u64 { marker_size(msg, true) }
    fn sceAmprMeasureCommandSizePushMarker(msg: *const u8) -> u64 { marker_size(msg, false) }
    fn sceAmprMeasureCommandSizePushMarkerWithColor(msg: *const u8, color: u32) -> u64 { marker_size(msg, true) }
    fn sceAmprMeasureCommandSizePopMarker() -> u64 { 4 }
    fn sceAmprMeasureCommandSizeReadFile(file: u32, dest: u64, size: u64, offset: u64) -> u64 {
        if !valid_range(dest, size) || offset >= MAX_FILE_OFFSET { return err64(EINVAL); }
        if offset >> 32 != 0 { REC_READ_EXT } else { REC_READ }
    }
    fn sceAmprMeasureCommandSizeReadFileGather(size: u64, offset: u64) -> u64 {
        if size == 0 || size > MAX_READ || offset >= MAX_FILE_OFFSET { return err64(EINVAL); }
        if offset > 0x3_FFFF { REC_GATHER_EXT } else { REC_GATHER }
    }
    fn sceAmprMeasureCommandSizeReadFileScatter(dest: u64, size: u64) -> u64 {
        if !valid_range(dest, size) { return err64(EINVAL); }
        REC_SCATTER
    }
    fn sceAmprMeasureCommandSizeReadFileGatherScatter(dest: u64, size: u64, offset: u64) -> u64 {
        if !valid_range(dest, size) || offset >= MAX_FILE_OFFSET { return err64(EINVAL); }
        if offset >> 32 != 0 { REC_GS_EXT } else { REC_GS }
    }
    fn sceAmprMeasureCommandSizeResetGatherScatterState() -> u64 { REC_RESET_GS }
    fn sceAmprMeasureCommandSizeMapBegin(va: u64, size: u64, mtype: i32, prot: i32) -> u64 {
        if va == 0 || size == 0 || va % PAGE != 0 || size % PAGE != 0 { return err64(EINVAL); }
        REC_MAP_BEGIN
    }
    fn sceAmprMeasureCommandSizeMapDirectBegin(va: u64, dmem: u64, size: u64, mtype: i32, prot: i32) -> u64 {
        if va == 0 || size == 0 || va % PAGE != 0 || size % PAGE != 0 || dmem % PAGE != 0 { return err64(EINVAL); }
        REC_MAP_DIRECT_BEGIN
    }
    fn sceAmprMeasureCommandSizeMapEnd() -> u64 { REC_MAP_END }
    fn sceAmprMeasureCommandSizeWriteAddress_04_00() -> u64 { REC_FIXED }
    fn sceAmprMeasureCommandSizeWriteAddressFromTimeCounter_04_00() -> u64 { REC_FIXED }
    fn sceAmprMeasureCommandSizeWriteAddressFromCounter_04_00() -> u64 { REC_FIXED }
    fn sceAmprMeasureCommandSizeWriteAddressFromCounterPair_04_00() -> u64 { REC_FIXED }
    fn sceAmprMeasureCommandSizeWriteCounter_04_00() -> u64 { REC_FIXED }
    fn sceAmprMeasureCommandSizeWaitOnAddress_04_00() -> u64 { REC_FIXED }
    fn sceAmprMeasureCommandSizeWaitOnCounter_04_00() -> u64 { REC_FIXED }
    fn sceAmprMeasureCommandSizeWriteKernelEventQueue_04_00() -> u64 { REC_KEVENT }

    // ---- libkernel APR
    fn sceKernelAprResolveFilepathsToIds(paths: *const u64, count: u32, ids: *mut u32, err_index: *mut u32) -> i32 {
        crate::hle::kernel::posix_ret(resolve_list(None, paths, count, ids, std::ptr::null_mut(), err_index, None))
    }
    fn sceKernelAprResolveFilepathsToIdsAndFileSizes(paths: *const u64, count: u32, ids: *mut u32, sizes: *mut u64, err_index: *mut u32) -> i32 {
        crate::hle::kernel::posix_ret(resolve_list(None, paths, count, ids, sizes, err_index, None))
    }
    fn sceKernelAprResolveFilepathsWithPrefixToIds(prefix: *const u8, paths: *const u64, count: u32, ids: *mut u32, err_index: *mut u32) -> i32 {
        crate::hle::kernel::posix_ret(resolve_list(Some(cstr(prefix)), paths, count, ids, std::ptr::null_mut(), err_index, None))
    }
    fn sceKernelAprResolveFilepathsWithPrefixToIdsAndFileSizes(prefix: *const u8, paths: *const u64, count: u32, ids: *mut u32, sizes: *mut u64, err_index: *mut u32) -> i32 {
        crate::hle::kernel::posix_ret(resolve_list(Some(cstr(prefix)), paths, count, ids, sizes, err_index, None))
    }
    fn sceKernelAprResolveFilepathsToIdsForEach(paths: *const u64, count: u32, ids: *mut u32, results: *mut i32) -> i32 {
        crate::hle::kernel::posix_ret(resolve_list(None, paths, count, ids, std::ptr::null_mut(), std::ptr::null_mut(), Some(results)))
    }
    fn sceKernelAprResolveFilepathsToIdsAndFileSizesForEach(paths: *const u64, count: u32, ids: *mut u32, sizes: *mut u64, results: *mut i32) -> i32 {
        crate::hle::kernel::posix_ret(resolve_list(None, paths, count, ids, sizes, std::ptr::null_mut(), Some(results)))
    }
    fn sceKernelAprResolveFilepathsWithPrefixToIdsForEach(prefix: *const u8, paths: *const u64, count: u32, ids: *mut u32, results: *mut i32) -> i32 {
        crate::hle::kernel::posix_ret(resolve_list(Some(cstr(prefix)), paths, count, ids, std::ptr::null_mut(), std::ptr::null_mut(), Some(results)))
    }
    fn sceKernelAprResolveFilepathsWithPrefixToIdsAndFileSizesForEach(prefix: *const u8, paths: *const u64, count: u32, ids: *mut u32, sizes: *mut u64, results: *mut i32) -> i32 {
        crate::hle::kernel::posix_ret(resolve_list(Some(cstr(prefix)), paths, count, ids, sizes, std::ptr::null_mut(), Some(results)))
    }
    fn sceKernelAprGetFileSize(id: u32, size: *mut u64) -> i32 {
        match file(id) {
            Some(f) if !size.is_null() => { *size = f.size; 0 }
            Some(_) => crate::hle::kernel::posix_ret(sce(EINVAL)),
            None => crate::hle::kernel::posix_ret(sce(EBADF)),
        }
    }
    fn sceKernelAprGetFileStat(id: u32, st: *mut crate::hle::kernel::fs::Stat) -> i32 {
        let Some(f) = file(id) else { return crate::hle::kernel::posix_ret(sce(EBADF)) };
        // (The file as the game tree has it: the host file's metadata, or the archive's entry as
        // the folder's would be.)
        let stat = match &f.at {
            crate::hle::kernel::fs::Found::Host(host) => std::fs::metadata(host).ok().map(|md| crate::hle::kernel::fs::fill_stat(&md)),
            crate::hle::kernel::fs::Found::Zar(a, n) => Some(crate::hle::kernel::fs::archive_stat(a, *n)),
        };
        match stat {
            Some(s) if !st.is_null() => { std::ptr::write_unaligned(st, s); 0 }
            _ => crate::hle::kernel::posix_ret(sce(EIO)),
        }
    }
    fn sceKernelAprSubmitCommandBuffer(cb: u64, prio: u32) -> i32 {
        match submit(cb, prio, 0) { Ok(_) => 0, Err(e) => crate::hle::kernel::posix_ret(sce(e)) }
    }
    fn sceKernelAprSubmitCommandBufferAndGetId(cb: u64, prio: u32, id_out: *mut u32) -> i32 {
        if id_out.is_null() { return crate::hle::kernel::posix_ret(sce(EINVAL)); }
        match submit(cb, prio, 0) { Ok(id) => { *id_out = id; 0 } Err(e) => crate::hle::kernel::posix_ret(sce(e)) }
    }
    fn sceKernelAprSubmitCommandBufferAndGetResult(cb: u64, prio: u32, result: u64, id_out: *mut u32) -> i32 {
        if result != 0 { wr32(result, 0); wr32(result + 4, 0); }
        match submit(cb, prio, result) {
            Ok(id) => { if !id_out.is_null() { *id_out = id; } 0 }
            Err(e) => crate::hle::kernel::posix_ret(sce(e)),
        }
    }
    fn sceKernelAprWaitCommandBuffer(id: u32) -> i32 {
        match wait_done(id) { Ok(_) => 0, Err(e) => crate::hle::kernel::posix_ret(sce(e)) }
    }
}

pub fn register(r: &mut Registry) {
    reg!(r, "libSceAmpr", [
        sceAmprCommandBufferConstructor, sceAmprCommandBufferDestructor, sceAmprAprCommandBufferConstructor,
        sceAmprAprCommandBufferDestructor, sceAmprCommandBufferSetBuffer, sceAmprCommandBufferReset,
        sceAmprCommandBufferClearBuffer, sceAmprCommandBufferGetType, sceAmprCommandBufferGetSize,
        sceAmprCommandBufferGetBufferBaseAddress, sceAmprCommandBufferGetNumCommands, sceAmprCommandBufferGetCurrentOffset,
        sceAmprCommandBufferNop, sceAmprCommandBufferNopWithData, sceAmprCommandBufferConstructNop,
        sceAmprCommandBufferConstructMarker, sceAmprCommandBufferSetMarker, sceAmprCommandBufferSetMarkerWithColor,
        sceAmprCommandBufferPushMarker, sceAmprCommandBufferPushMarkerWithColor, sceAmprCommandBufferPopMarker,
        sceAmprCommandBufferWriteAddress_04_00, sceAmprCommandBufferWriteAddressFromTimeCounter_04_00,
        sceAmprCommandBufferWriteAddressFromCounter_04_00, sceAmprCommandBufferWriteAddressFromCounterPair_04_00,
        sceAmprCommandBufferWriteCounter_04_00, sceAmprCommandBufferWaitOnAddress_04_00,
        sceAmprCommandBufferWaitOnCounter_04_00, sceAmprCommandBufferWriteKernelEventQueue_04_00,
        sceAmprAprCommandBufferReadFile, sceAmprAprCommandBufferReadFileGather, sceAmprAprCommandBufferReadFileScatter,
        sceAmprAprCommandBufferReadFileGatherScatter, sceAmprAprCommandBufferResetGatherScatterState,
        sceAmprAprCommandBufferMapBegin, sceAmprAprCommandBufferMapDirectBegin, sceAmprAprCommandBufferMapEnd,
        sceAmprMeasureCommandSizeNop, sceAmprMeasureCommandSizeNopWithData, sceAmprMeasureCommandSizeSetMarker,
        sceAmprMeasureCommandSizeSetMarkerWithColor, sceAmprMeasureCommandSizePushMarker,
        sceAmprMeasureCommandSizePushMarkerWithColor, sceAmprMeasureCommandSizePopMarker,
        sceAmprMeasureCommandSizeReadFile, sceAmprMeasureCommandSizeReadFileGather,
        sceAmprMeasureCommandSizeReadFileScatter, sceAmprMeasureCommandSizeReadFileGatherScatter,
        sceAmprMeasureCommandSizeResetGatherScatterState, sceAmprMeasureCommandSizeMapBegin,
        sceAmprMeasureCommandSizeMapDirectBegin, sceAmprMeasureCommandSizeMapEnd,
        sceAmprMeasureCommandSizeWriteAddress_04_00, sceAmprMeasureCommandSizeWriteAddressFromTimeCounter_04_00,
        sceAmprMeasureCommandSizeWriteAddressFromCounter_04_00, sceAmprMeasureCommandSizeWriteAddressFromCounterPair_04_00,
        sceAmprMeasureCommandSizeWriteCounter_04_00, sceAmprMeasureCommandSizeWaitOnAddress_04_00,
        sceAmprMeasureCommandSizeWaitOnCounter_04_00, sceAmprMeasureCommandSizeWriteKernelEventQueue_04_00,
    ]);
    reg!(r, "libkernel", [
        sceKernelAprResolveFilepathsToIds, sceKernelAprResolveFilepathsToIdsAndFileSizes,
        sceKernelAprResolveFilepathsWithPrefixToIds, sceKernelAprResolveFilepathsWithPrefixToIdsAndFileSizes,
        sceKernelAprResolveFilepathsToIdsForEach, sceKernelAprResolveFilepathsToIdsAndFileSizesForEach,
        sceKernelAprResolveFilepathsWithPrefixToIdsForEach, sceKernelAprResolveFilepathsWithPrefixToIdsAndFileSizesForEach,
        sceKernelAprGetFileSize, sceKernelAprGetFileStat, sceKernelAprSubmitCommandBuffer,
        sceKernelAprSubmitCommandBufferAndGetId, sceKernelAprSubmitCommandBufferAndGetResult, sceKernelAprWaitCommandBuffer,
    ]);
}

/// Unused-but-kept guard: the stats are reported by the main loop.
pub fn stats_line() -> String {
    format!(
        "apr: {} submissions, {} reads, {:.1} MiB",
        STATS[0].load(Ordering::Relaxed),
        STATS[1].load(Ordering::Relaxed),
        STATS[2].load(Ordering::Relaxed) as f64 / (1 << 20) as f64
    )
}

#[allow(dead_code)]
fn _assert_send(_: OnceLock<()>) {}
