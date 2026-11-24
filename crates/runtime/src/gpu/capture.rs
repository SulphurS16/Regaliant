//! Frame capture: the GPU state of every draw and dispatch submitted between
//! two flips, written to `<run dir>/capture-<flip>.txt`, plus each distinct
//! shader program to `<run dir>/shaders/<address>.bin`. Armed with the
//! `capture` control command or `DS_CAPTURE=<flip number>`.

use super::cp::{Cp, Space};
use parking_lot::Mutex;
use std::collections::BTreeSet;
use std::fmt::Write;
use std::sync::atomic::{AtomicU64, Ordering};

/// Flip number at which to start capturing (u64::MAX = not armed).
static ARMED: AtomicU64 = AtomicU64::new(u64::MAX);
static COMPLETED: AtomicU64 = AtomicU64::new(0);

pub fn completed_flip() -> u64 { COMPLETED.load(Ordering::Acquire) }

/// Async flips ordinarily leave batches queued. A diagnostic frame's boundaries must
/// complete before target readbacks, otherwise the snapshot can precede its GPU writes.
pub fn needs_boundary_sync(flip: u64) -> bool {
    CAPTURE_ON.load(Ordering::Acquire) || flip >= ARMED.load(Ordering::Acquire)
}

struct Capture {
    flip: u64,
    text: String,
    shaders: BTreeSet<u64>,
    events: u64,
    /// Textures referenced in the frame (dumped as PNG when viewable).
    textures: std::collections::BTreeMap<u64, super::desc::Tsharp>,
    targets: BTreeSet<u64>,
}

static ACTIVE: Mutex<Option<Capture>> = Mutex::new(None);

pub fn init_from_env() {
    if let Some(n) = std::env::var("DS_CAPTURE").ok().and_then(|v| v.trim().parse::<u64>().ok()) {
        ARMED.store(n, Ordering::SeqCst);
        crate::info!("capture armed for flip {n}");
    }
}

/// Arms a capture of the frame after the next flip.
pub fn arm_next() {
    ARMED.store(super::FRAMES.load(Ordering::SeqCst) + 1, Ordering::SeqCst);
}

pub fn capturing() -> bool {
    ACTIVE.try_lock().is_some_and(|g| g.is_some())
}

/// A frame is being captured (set and cleared with `ACTIVE`'s capture):
/// the lock-free form for translation-time checks, where `capturing`'s
/// try_lock fails whenever another thread writes the capture's record.
static CAPTURE_ON: std::sync::atomic::AtomicBool = std::sync::atomic::AtomicBool::new(false);

/// Called by the graphics queue for each flip packet (`flip` = flips so far).
/// Frames still to capture after the current one (DS_CAPTURE_COUNT).
static MORE: AtomicU64 = AtomicU64::new(0);

/// Flips so far (graphics queue).
static FLIPS: AtomicU64 = AtomicU64::new(0);

pub fn flips() -> u64 {
    FLIPS.load(Ordering::Relaxed)
}

pub fn on_flip(flip: u64) {
    FLIPS.store(flip, Ordering::Relaxed);
    let mut g = ACTIVE.lock();
    if let Some(c) = g.take() {
        // Consecutive captures: text only (no texture, target or draw dumps).
        let more = MORE.load(Ordering::SeqCst);
        let last = more == 0;
        CAPTURE_ON.store(!last, Ordering::SeqCst);
        finish(c, last);
        if !last {
            MORE.store(more - 1, Ordering::SeqCst);
            *g = Some(Capture { flip, text: format!("# frame after flip {flip}\n"), shaders: BTreeSet::new(), events: 0, textures: Default::default(), targets: Default::default() });
        }
        return;
    }
    let armed = ARMED.load(Ordering::SeqCst);
    if armed != u64::MAX && flip >= armed {
        ARMED.store(u64::MAX, Ordering::SeqCst);
        MORE.store(std::env::var("DS_CAPTURE_COUNT").ok().and_then(|v| v.parse::<u64>().ok()).unwrap_or(1).saturating_sub(1), Ordering::SeqCst);
        *g = Some(Capture { flip, text: format!("# frame after flip {flip}\n"), shaders: BTreeSet::new(), events: 0, textures: Default::default(), targets: Default::default() });
        CAPTURE_ON.store(true, Ordering::SeqCst);
        crate::info!("capture: recording the frame after flip {flip}");
    }
}

fn finish(c: Capture, dumps: bool) {
    let Some(dir) = crate::run_dir() else { return };
    let path = dir.join(format!("capture-{}.txt", c.flip));
    let _ = std::fs::write(&path, &c.text);
    let dumped = c.shaders.iter().filter(|&&a| save_program(a).is_some()).count();
    let tex = if dumps { dump_textures(&dir, &c.textures) } else { 0 };
    // Render targets as rendered (GPU execution): the graphics queue's work
    // of the frame completed before its flip.
    let frame_targets = std::env::var("DS_CAPTURE_TARGETS_FRAME").ok().as_deref() == Some("1");
    let target_dir = if frame_targets { dir.join(format!("targets-{}", c.flip)) } else { dir.join("targets") };
    let rts = if dumps && super::exec::enabled() {
        super::images::with(|im| if frame_targets { im.dump_frame_targets(&target_dir, &c.targets) } else { im.dump_targets(&target_dir) })
    } else { 0 };
    crate::info!("capture: {} event(s) -> {} ({dumped} shader binaries in shaders/, {tex} texture(s) in textures/, {rts} color render target(s) plus depth/stencil in {})", c.events, path.display(), target_dir.display());
    COMPLETED.store(c.flip, Ordering::Release);
}

/// Writes viewable textures (RGBA8/R8/RG8 in standard swizzles) of the frame
/// as PNG files, detiled with `tiling` (mip 0, slice 0): a visual check of
/// the swizzle implementation.
fn dump_textures(dir: &std::path::Path, textures: &std::collections::BTreeMap<u64, super::desc::Tsharp>) -> usize {
    let tdir = dir.join("textures");
    let mut n = 0;
    for (base, t) in textures {
        if n >= 48 || t.width < 16 || t.height < 16 || !matches!(t.sw_mode, 1 | 5 | 9) {
            continue;
        }
        let (elem_log2, expand): (u32, fn(&[u8]) -> [u8; 4]) = match t.format {
            56 | 130 => (2, |p| [p[0], p[1], p[2], p[3]]),
            1 => (0, |p| [p[0], p[0], p[0], 255]),
            14 => (1, |p| [p[0], p[1], 0, 255]),
            _ => continue,
        };
        let levels = if t.max_mip > 0 { t.max_mip + 1 } else { (t.last_level + 1).max(1) };
        let Some(l) = super::tiling::layout(t.sw_mode, elem_log2, t.width, t.height, levels, false) else { continue };
        let size = l.size(1);
        if !crate::gpu::cp::guest_readable(*base, size) {
            continue;
        }
        let src = unsafe { std::slice::from_raw_parts(*base as *const u8, size as usize) };
        let mut lin = Vec::new();
        l.detile(src, 0, 0, &mut lin);
        let e = 1usize << elem_log2;
        let mut rgba = Vec::with_capacity(lin.len() / e * 4);
        for px in lin.chunks_exact(e) {
            rgba.extend_from_slice(&expand(px));
        }
        let _ = std::fs::create_dir_all(&tdir);
        let name = format!("{base:010x}_{}x{}_f{}_sw{}_l{levels}.png", t.width, t.height, t.format, t.sw_mode);
        if std::fs::write(tdir.join(name), super::png::encode(t.width, t.height, &rgba)).is_ok() {
            n += 1;
        }
    }
    n
}

/// Fallback size: up to the first s_endpgm (gfx10 0xBF810000), at most 64 KiB.
/// Writes the shader program at `a` to `<run dir>/shaders/<address>.bin`.
pub fn save_program(a: u64) -> Option<std::path::PathBuf> {
    // (Development checkouts only: a player's log folder is what they attach to a public report,
    // and the programs are the game's own code.)
    if crate::paths().layout != dsconfig::paths::Layout::Dev && std::env::var("DS_CAPTURE_CODE").ok().as_deref() != Some("1") {
        return None;
    }
    let sdir = crate::run_dir()?.join("shaders");
    let size = crate::hle::libs::agc::shader_code_size(a).unwrap_or_else(|| scan_to_endpgm(a));
    if size == 0 || !crate::gpu::cp::guest_readable(a, size as u64) {
        return None;
    }
    let _ = std::fs::create_dir_all(&sdir);
    let bytes = unsafe { std::slice::from_raw_parts(a as *const u8, size) };
    let path = sdir.join(format!("{a:012x}.bin"));
    std::fs::write(&path, bytes).ok().map(|_| path)
}

fn scan_to_endpgm(a: u64) -> usize {
    for i in 0..0x4000u64 {
        if !crate::gpu::cp::guest_readable(a + 4 * i, 4) {
            return 0;
        }
        if unsafe { *((a + 4 * i) as *const u32) } == 0xBF81_0000 {
            return 4 * (i as usize + 1);
        }
    }
    0x10000
}

fn sh_addr(cp: &Cp, lo: u32) -> u64 {
    ((cp.reg(Space::Sh, lo) as u64) << 8) | (((cp.reg(Space::Sh, lo + 1) & 0xFF) as u64) << 40)
}

/// Program, resources and user data of one hardware stage.
fn stage(cp: &Cp, out: &mut String, shaders: &mut BTreeSet<u64>, name: &str, lo: u32, rsrc: u32, user: u32) {
    let a = sh_addr(cp, lo);
    if a == 0 {
        return;
    }
    shaders.insert(a);
    let rsrc1 = cp.reg(Space::Sh, rsrc);
    let rsrc2 = cp.reg(Space::Sh, rsrc + 1);
    let n = (((rsrc2 >> 1) & 0x1F) | (((rsrc2 >> 27) & 1) << 5)).min(32);
    let ud: Vec<String> = (0..n).map(|i| format!("{:08x}", cp.reg(Space::Sh, user + i))).collect();
    let _ = writeln!(out, "    {name} {a:#x} rsrc1 {rsrc1:#010x} rsrc2 {rsrc2:#010x} user[{n}] {}", ud.join(" "));
}

/// First records of the vertex stage's formatted buffers (vertex data).
fn vertex_data(out: &mut String, p: &super::shaders::Prepared) {
    for (i, r) in p.resources.iter().enumerate() {
        let super::shaders::Resolved::Buffer(v) = r else { continue };
        if v.stride == 0 || v.stride > 64 || v.format == 0 {
            continue;
        }
        let n = v.num_records.min(6) as u64;
        if !crate::gpu::cp::guest_readable(v.base, n * v.stride as u64) {
            continue;
        }
        let mut recs = Vec::new();
        for k in 0..n {
            let a = v.base + k * v.stride as u64;
            let words: Vec<u32> = (0..(v.stride as u64 / 4).min(4)).map(|j| unsafe { std::ptr::read_volatile((a + 4 * j) as *const u32) }).collect();
            recs.push(match v.format {
                // 32-bit float components.
                64 | 74 | 77 | 22 => format!("({})", words.iter().map(|w| format!("{}", f32::from_bits(*w))).collect::<Vec<_>>().join(", ")),
                _ => format!("[{}]", words.iter().map(|w| format!("{w:08x}")).collect::<Vec<_>>().join(" ")),
            });
        }
        let _ = writeln!(out, "      es {i} data: {}", recs.join(" "));
    }
}

fn note_textures(map: &mut std::collections::BTreeMap<u64, super::desc::Tsharp>, p: &super::shaders::Prepared) {
    for r in p.resources.iter() {
        if let super::shaders::Resolved::Image(t) = r {
            map.entry(t.base).or_insert(*t);
        }
        if let super::shaders::Resolved::ImageTable(e) = r {
            for t in e.iter().flatten() {
                map.entry(t.base).or_insert(*t);
            }
        }
    }
}

fn f(v: u32) -> f32 {
    f32::from_bits(v)
}

/// `DS_CAPTURE_DRAWS=1` (every draw) or `=<from>-<to>` (capture events):
/// after each such draw, the pixels of its first color target and of its
/// stencil that changed (and the nonzero stencil count) are reported, and
/// the color target is written to draws/<event>_<address>.png.
pub fn draw_dumps() -> bool {
    let Some((lo, hi)) = dumps_env() else { return false };
    let next = ACTIVE.try_lock().and_then(|g| g.as_ref().map(|c| c.events + 1));
    next.is_some_and(|e| e >= lo && e <= hi)
}

/// A frame capture with a watched pixel (`capture pixel=x,y`): each draw
/// completes before its texel is read back.
pub fn watching() -> bool {
    WATCH_PIXEL.lock().is_some() && capturing()
}

/// `DS_CAPTURE_DISPATCHES=1` or `capture dispatches` (control command, from
/// then on): also dump the images each captured dispatch writes, after it
/// (dispatches/<flip>_<event>_<shader>_<image>.png, with statistics).
pub fn dispatch_dumps() -> bool {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    (*ON.get_or_init(|| std::env::var("DS_CAPTURE_DISPATCHES").ok().as_deref() == Some("1")) || DISPATCH_DUMPS.load(Ordering::Relaxed)) && capturing()
}

static DISPATCH_DUMPS: std::sync::atomic::AtomicBool = std::sync::atomic::AtomicBool::new(false);

pub fn set_dispatch_dumps(on: bool) {
    DISPATCH_DUMPS.store(on, Ordering::Relaxed);
}

/// Compute shaders whose bound buffers a capture with dispatch dumps writes
/// to dispatches/ after each of their dispatches (`capture buffers=<shader>[,..]`;
/// buffers up to 64 MiB for one shader, 4 MiB each for several).
static BUFFER_DUMPS: Mutex<Vec<u64>> = Mutex::new(Vec::new());

pub fn set_buffer_dumps(cs: Vec<u64>) {
    *BUFFER_DUMPS.lock() = cs;
}

fn buffer_dump_limit(cs: u64) -> Option<u64> {
    let g = BUFFER_DUMPS.lock();
    g.contains(&cs).then(|| if g.len() > 1 { 4 << 20 } else { 64 << 20 })
}

/// Images whose contents a capture with dispatch dumps writes (PNG and raw
/// texels) after each dispatch that writes them (`capture images=<hex
/// base>[,..]`); empty: every image dispatches write (PNG only).
static IMAGE_DUMPS: Mutex<Vec<u64>> = Mutex::new(Vec::new());

pub fn set_image_dumps(bases: Vec<u64>) {
    *IMAGE_DUMPS.lock() = bases;
}

/// Per-draw report range set by the `capture <from>-<to>` control command
/// (overrides `DS_CAPTURE_DRAWS`).
static DRAW_RANGE: Mutex<Option<(u64, u64)>> = Mutex::new(None);

pub fn set_draw_range(r: (u64, u64)) {
    *DRAW_RANGE.lock() = Some(r);
}

/// Pixel shaders whose draws get target reports in every captured frame
/// (`capture ps=<hex>[,<hex>...]`), wherever they fall in the frame.
static DRAW_PS: Mutex<Option<Vec<u64>>> = Mutex::new(None);

pub fn set_draw_ps(list: Vec<u64>) {
    *DRAW_PS.lock() = Some(list);
}

fn ps_listed(cp: &Cp) -> bool {
    DRAW_PS.lock().as_ref().is_some_and(|l| l.contains(&super::shaders::ps_addr(cp)))
}

/// Vertex programs whose draws get target reports and, with `DS_VS_DUMP`,
/// register dumps in every captured frame (`capture vs=<hex>[,<hex>...]`),
/// wherever they fall in the frame (event numbers move from run to run).
static DRAW_VS: Mutex<Option<Vec<u64>>> = Mutex::new(None);

pub fn set_draw_vs(list: Vec<u64>) {
    *DRAW_VS.lock() = Some(list);
}

fn vs_listed(cp: &Cp) -> bool {
    DRAW_VS.lock().as_ref().is_some_and(|l| l.contains(&super::shaders::ngg_addr(cp)))
}

/// A draw on the `capture ps=` / `vs=` lists while a frame is captured: its
/// report reads its targets (and vertex dumps) after it, so it completes
/// first.
pub fn listed_draw(cp: &Cp) -> bool {
    CAPTURE_ON.load(Ordering::Relaxed) && (ps_listed(cp) || vs_listed(cp))
}

/// `vs_dump` for the vertex program at `addr`: also while a frame is
/// captured when the program is on the `capture vs=` list.
pub fn vs_dump_for(addr: u64) -> Option<(u32, u32, u32)> {
    let v = vs_dump_env()?;
    (draw_dumps() || (CAPTURE_ON.load(Ordering::Relaxed) && DRAW_VS.lock().as_ref().is_some_and(|l| l.contains(&addr)))).then_some(v)
}

fn dumps_env() -> Option<(u64, u64)> {
    if let Some(r) = *DRAW_RANGE.lock() {
        return Some(r);
    }
    static ON: std::sync::OnceLock<Option<(u64, u64)>> = std::sync::OnceLock::new();
    *ON.get_or_init(|| {
        let v = std::env::var("DS_CAPTURE_DRAWS").ok()?;
        if v == "1" {
            return Some((0, u64::MAX));
        }
        let (a, b) = v.split_once('-')?;
        Some((a.trim().parse().ok()?, b.trim().parse().ok()?))
    })
}

/// `DS_VS_DUMP=<first vertex index>[:<instance>[:<instances>]]` with
/// per-draw reports (`DS_CAPTURE_DRAWS=<from>-<to>`): the vertex shaders of
/// those draws store their registers at their position export for 8192
/// vertex indices from the first one (instance 0 unless given); with an
/// instance count n above 1 the 8192 slots go to n instances from the given
/// one, 8192 / n vertex indices each (`0:0:8192`: vertex 0 of 8192
/// instances). `DS_VS_DUMP_PC=<hex>`: after that instruction instead (any
/// vertex program with an instruction there). The report writes each dumped
/// vertex's exported position to draws/<event>_vs.txt (at the export) and
/// all registers to draws/<event>_vs.bin (per vertex: instance, vertex
/// index, then `DUMP_STRIDE` dwords: VGPRs, SGPRs at 256, SCC at 384).
pub fn vs_dump() -> Option<(u32, u32, u32)> {
    let v = vs_dump_env()?;
    draw_dumps().then_some(v)
}

/// `DS_VS_DUMP` parsed (whether or not a draw is being dumped now: the
/// report runs while the capture's lock is held, which `draw_dumps` tries).
fn vs_dump_env() -> Option<(u32, u32, u32)> {
    static V: std::sync::OnceLock<Option<(u32, u32, u32)>> = std::sync::OnceLock::new();
    *V.get_or_init(|| {
        let s = std::env::var("DS_VS_DUMP").ok()?;
        let mut it = s.split(':');
        let first = it.next()?.trim().parse().ok()?;
        let inst = it.next().and_then(|x| x.trim().parse().ok()).unwrap_or(0);
        let count = it.next().and_then(|x| x.trim().parse::<u32>().ok()).unwrap_or(1);
        // `:pos`: positions only (64 vertex slots per instance, up to 16384
        // instances; draws/<event>_pos.txt).
        let pos = it.next().is_some_and(|x| x.trim() == "pos");
        let count = if pos {
            count.clamp(1, rdna::translate::DUMP_DWORDS / 4 / rdna::translate::DUMP_POS_VERTICES) | rdna::translate::DUMP_POS
        } else {
            count.clamp(1, rdna::translate::DUMP_DWORDS / rdna::translate::DUMP_STRIDE)
        };
        Some((first, inst, count))
    })
}

/// `DS_VS_DUMP_PC`, when set: the instruction vertex dumps follow.
pub fn vs_dump_pc() -> Option<u32> {
    static V: std::sync::OnceLock<Option<u32>> = std::sync::OnceLock::new();
    *V.get_or_init(|| u32::from_str_radix(std::env::var("DS_VS_DUMP_PC").ok()?.trim().trim_start_matches("0x"), 16).ok())
}

fn vs_dump_report(cp: &Cp, event: u64) -> Option<String> {
    let (first, inst0, count) = vs_dump_env()?;
    if count & rdna::translate::DUMP_POS != 0 {
        // Positions only: slots of four dwords, 64 per instance; empty ones zero.
        let regs = super::exec::debug_dump(true);
        let per = rdna::translate::DUMP_POS_VERTICES as usize;
        let mut out = String::new();
        let mut n = 0;
        for (slot, p) in regs.chunks_exact(4).enumerate() {
            if p.iter().all(|&w| w == 0) {
                continue;
            }
            n += 1;
            let f: Vec<f32> = p.iter().map(|&w| f32::from_bits(w)).collect();
            let _ = writeln!(out, "{} {} {} {} {} {}", inst0 as usize + slot / per, first as usize + slot % per, f[0], f[1], f[2], f[3]);
        }
        if n == 0 {
            return None;
        }
        let dir = crate::run_dir()?.join("draws");
        let _ = std::fs::create_dir_all(&dir);
        let _ = std::fs::write(dir.join(format!("{event:04}_pos.txt")), out);
        return Some(format!("{n} vertex positions dumped"));
    }
    let per = (rdna::translate::DUMP_DWORDS / rdna::translate::DUMP_STRIDE / count) as usize;
    let addr = super::shaders::ngg_addr(cp);
    let (insts, _) = super::shaders::program(addr).ok()?;
    let exp = super::shaders::pos_export(&insts)?;
    let at_export = vs_dump_pc().is_none_or(|pc| pc == exp.pc || !insts.iter().any(|i| i.pc == pc));
    let regs = super::exec::debug_dump(true);
    let stride = rdna::translate::DUMP_STRIDE as usize;
    let mut out = String::new();
    let mut bin: Vec<u8> = Vec::new();
    let mut n = 0;
    for (slot, r) in regs.chunks_exact(stride).enumerate() {
        if r[385] == 0 {
            continue;
        }
        n += 1;
        let (inst, vtx) = (inst0 as usize + slot / per, first as usize + slot % per);
        if at_export {
            let pos: Vec<String> = (0..4).map(|k| format!("{}", f32::from_bits(r[exp.exp_src(k) as usize]))).collect();
            let _ = writeln!(out, "{inst} {vtx} {}", pos.join(" "));
        }
        bin.extend_from_slice(&(inst as u32).to_le_bytes());
        bin.extend_from_slice(&(vtx as u32).to_le_bytes());
        for w in r {
            bin.extend_from_slice(&w.to_le_bytes());
        }
    }
    let dir = crate::run_dir()?.join("draws");
    let _ = std::fs::create_dir_all(&dir);
    if at_export {
        let _ = std::fs::write(dir.join(format!("{event:04}_vs.txt")), out);
    }
    let _ = std::fs::write(dir.join(format!("{event:04}_vs.bin")), bin);
    let pc = if at_export { exp.pc } else { vs_dump_pc().unwrap_or(exp.pc) };
    Some(format!("vs {addr:#x}: {n} vertices dumped (registers after {pc:#x}{})", if at_export { ", the position export" } else { "" }))
}

/// Pixel whose cb0 value per-draw reports follow (`capture ... pixel=x,y`),
/// in the presented image's coordinates (`DISPLAY`).
static WATCH_PIXEL: Mutex<Option<(u32, u32)>> = Mutex::new(None);

/// Size of the presented image (width << 32 | height), noted at each flip:
/// a watched pixel is scaled to images of other sizes.
static DISPLAY: AtomicU64 = AtomicU64::new(0);

pub fn note_display_size(w: u32, h: u32) {
    DISPLAY.store((w as u64) << 32 | h as u64, Ordering::Relaxed);
}

/// The watched pixel's position in a `w` x `h` image.
fn scaled(x: u32, y: u32, w: u32, h: u32) -> (u32, u32) {
    let d = DISPLAY.load(Ordering::Relaxed);
    let (dw, dh) = ((d >> 32) as u32, d as u32);
    if dw == 0 || dh == 0 || (w == dw && h == dh) {
        return (x, y);
    }
    ((x as u64 * w as u64 / dw as u64) as u32, (y as u64 * h as u64 / dh as u64) as u32)
}

/// Hex of a texel, with the channels of 8-byte (four f16) and 16-byte (four
/// f32) texels decoded.
fn texel_hex(b: &[u8]) -> String {
    let h = b.iter().rev().map(|x| format!("{x:02x}")).collect::<String>();
    let dec: Option<Vec<f32>> = match b.len() {
        8 => Some(b.chunks_exact(2).map(|c| super::images::half_to_f32(u16::from_le_bytes([c[0], c[1]]))).collect()),
        16 => Some(b.chunks_exact(4).map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]])).collect()),
        _ => None,
    };
    match dec {
        Some(v) => format!("{h} ({})", v.iter().map(|x| format!("{x:.4}")).collect::<Vec<_>>().join(", ")),
        None => h,
    }
}

/// The watched pixel of an image a dispatch wrote, against its value after
/// the previous dispatch that wrote it: " px(x,y) before -> after" (None
/// when unchanged and `only_changes`).
fn watch_image(t: &super::desc::Tsharp, x: u32, y: u32, only_changes: bool) -> Option<String> {
    let (sx, sy) = scaled(x, y, t.width, t.height);
    let (now, w, h, fmt) = super::images::with(|im| im.texel_of(t, sx, sy)).ok()?;
    let mut g = LAST_PIXEL.lock();
    let before = g.get_or_insert_with(Default::default).insert(t.base, now.clone());
    match before {
        Some(b) if b != now => Some(format!(" px({sx},{sy}) of {w}x{h} {fmt:?} {} -> {}", texel_hex(&b), texel_hex(&now))),
        Some(_) if only_changes => None,
        Some(_) => Some(format!(" px({sx},{sy}) unchanged {}", texel_hex(&now))),
        None => Some(format!(" px({sx},{sy}) of {w}x{h} {fmt:?} {}", texel_hex(&now))),
    }
}

pub fn set_watch_pixel(p: Option<(u32, u32)>) {
    *WATCH_PIXEL.lock() = p;
    *WATCH_TEXELS.lock() = None;
}

/// Last watched-pixel value per image written by dispatches.
static LAST_PIXEL: Mutex<Option<std::collections::HashMap<u64, Vec<u8>>>> = Mutex::new(None);

/// Previous contents of dumped images (changed-pixel counts).
static LAST_DUMP: Mutex<Option<std::collections::HashMap<(u64, bool), Vec<u8>>>> = Mutex::new(None);

fn changed(key: (u64, bool), raw: &[u8], texel: usize) -> Option<usize> {
    let mut g = LAST_DUMP.lock();
    let m = g.get_or_insert_with(Default::default);
    let n = m.get(&key).filter(|p| p.len() == raw.len()).map(|p| p.chunks_exact(texel).zip(raw.chunks_exact(texel)).filter(|(a, b)| a != b).count());
    m.insert(key, raw.to_vec());
    n
}

/// The watched pixel's color-target-0 texel and depth after the previous
/// draws that wrote them (watch-only captures).
static WATCH_TEXELS: Mutex<Option<std::collections::HashMap<(u64, bool), Vec<u8>>>> = Mutex::new(None);

/// `capture pixel=<x>,<y>` without a draw range: the draws that change the
/// watched pixel of their color target 0 or depth, with the values, from a
/// one-texel read back after each draw (whole-target dumps of a frame's
/// thousands of draws take far too long).
fn watch_texels(cp: &Cp, x: u32, y: u32) -> Option<String> {
    let hex = |b: &[u8]| b.iter().rev().map(|v| format!("{v:02x}")).collect::<String>();
    let mut out = Vec::new();
    let mut note = |key: (u64, bool), now: Vec<u8>, what: String| {
        let mut g = WATCH_TEXELS.lock();
        let before = g.get_or_insert_with(Default::default).insert(key, now.clone());
        if before.as_ref() != Some(&now) {
            out.push(match before {
                Some(b) => format!("{what} {} -> {}", hex(&b), hex(&now)),
                None => format!("{what} {}", hex(&now)),
            });
        }
    };
    if let Some(s) = super::exec::color_surface(cp, 0).ok().flatten() {
        let (x, y) = scaled(x, y, s.width, s.height);
        if x < s.width && y < s.height {
            if let Ok((raw, fmt)) = super::images::with(|im| -> Result<(Vec<u8>, ash::vk::Format), String> {
                let t = im.target(&s)?;
                Ok((im.read_back_texel(t.image, t.key.format, ash::vk::ImageAspectFlags::COLOR, x, y)?, t.key.format))
            }) {
                note((s.base, false), raw, format!("cb0 {:#x} {}x{} {fmt:?} px({x},{y})", s.base, s.width, s.height));
            }
        }
    }
    if let Some(d) = super::exec::depth_surface(cp).filter(|d| super::images::aspects(d.format).contains(ash::vk::ImageAspectFlags::DEPTH)) {
        let (x, y) = scaled(x, y, d.width, d.height);
        if x < d.width && y < d.height {
            if let Ok(raw) = super::images::with(|im| -> Result<Vec<u8>, String> {
                let t = im.target(&d)?;
                im.read_back_texel(t.image, t.key.format, ash::vk::ImageAspectFlags::DEPTH, x, y)
            }) {
                let z = (raw.len() == 4).then(|| f32::from_le_bytes([raw[0], raw[1], raw[2], raw[3]]));
                note((d.base, true), raw, format!("depth {:#x} z({x},{y}){}", d.base, z.map(|z| format!(" = {z}")).unwrap_or_default()));
            }
        }
    }
    (!out.is_empty()).then(|| out.join("; "))
}

fn dump_draw_target(cp: &Cp, event: u64) -> Option<String> {
    let listed = ps_listed(cp);
    if dumps_env().is_none() && !listed {
        if let Some((x, y)) = *WATCH_PIXEL.lock() {
            return watch_texels(cp, x, y);
        }
    }
    let (lo, hi) = match dumps_env() {
        Some(r) => r,
        None if listed => (u64::MAX, u64::MAX),
        None => return None,
    };
    if !listed && (event < lo || event > hi) {
        return None;
    }
    let mut out = Vec::new();
    if let Some(s) = super::exec::color_surface(cp, 0).ok().flatten() {
        let dir = crate::run_dir()?.join("draws");
        let _ = std::fs::create_dir_all(&dir);
        let r = super::images::with(|im| -> Result<(Vec<u8>, u32, u32, ash::vk::Format), String> {
            let t = im.target(&s)?;
            let raw = im.read_back(t.image, t.key.format, t.key.width, t.key.height)?;
            Ok((raw, t.key.width, t.key.height, t.key.format))
        });
        match r {
            Ok((raw, w, h, format)) => {
                let texel = (raw.len() / (w as usize * h as usize).max(1)).max(1);
                // The watched pixel before and after this draw.
                let watch = (*WATCH_PIXEL.lock()).filter(|&(x, y)| x < w && y < h).map(|(x, y)| {
                    let o = (y as usize * w as usize + x as usize) * texel;
                    let before = LAST_DUMP.lock().as_ref().and_then(|m| m.get(&(s.base, false)).and_then(|p| p.get(o..o + texel).map(|b| b.to_vec())));
                    (x, y, before, raw[o..o + texel].to_vec())
                });
                let ch = changed((s.base, false), &raw, texel).map(|n| format!("{n} px changed")).unwrap_or_else(|| "first dump".into());
                let hex = |b: &[u8]| b.iter().rev().map(|x| format!("{x:02x}")).collect::<String>();
                let ch = match watch {
                    Some((x, y, Some(b), a)) if b != a => format!("{ch}; px({x},{y}) {} -> {}", hex(&b), hex(&a)),
                    Some((x, y, None, a)) => format!("{ch}; px({x},{y}) {}", hex(&a)),
                    _ => ch,
                };
                let ch = match (listed, super::images::float_stats(format, &raw)) {
                    (true, Some(st)) => format!("{ch}; {st}"),
                    _ => ch,
                };
                let name = format!("{event:04}_{:010x}.png", s.base);
                // PNGs only on request (DS_CAPTURE_DRAWS_PNG=1): a range of
                // full-size targets is gigabytes.
                static PNG: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
                let png = *PNG.get_or_init(|| std::env::var("DS_CAPTURE_DRAWS_PNG").ok().as_deref() == Some("1")) || lo == 0;
                if png {
                    if let Some(rgba) = super::images::to_rgba8(format, &raw) {
                        let _ = std::fs::write(dir.join(&name), super::png::encode(w, h, &rgba));
                    }
                }
                out.push(format!("cb0 {name} ({ch})"));
            }
            Err(e) => out.push(format!("cb0 dump failed: {e}")),
        }
    }
    if let Some(d) = super::exec::depth_surface(cp).filter(|d| super::images::aspects(d.format).contains(ash::vk::ImageAspectFlags::DEPTH)) {
        let r = super::images::with(|im| -> Result<Vec<u8>, String> {
            let t = im.target(&d)?;
            im.read_back_aspect(t.image, t.key.format, ash::vk::ImageAspectFlags::DEPTH, t.key.width, t.key.height)
        });
        match r {
            Ok(raw) => {
                let texel = (raw.len() / (d.width as usize * d.height as usize).max(1)).max(1);
                // The watched pixel's depth before and after this draw.
                let watch = (*WATCH_PIXEL.lock()).filter(|&(x, y)| x < d.width && y < d.height && texel == 4).map(|(x, y)| {
                    let o = (y as usize * d.width as usize + x as usize) * 4;
                    let f = |b: &[u8]| f32::from_le_bytes([b[0], b[1], b[2], b[3]]);
                    let before = LAST_DUMP.lock().as_ref().and_then(|m| m.get(&(d.base, false)).and_then(|p| p.get(o..o + 4).map(f)));
                    (x, y, before, f(&raw[o..o + 4]))
                });
                let ch = changed((d.base, false), &raw, texel).map(|n| format!("{n} changed")).unwrap_or_else(|| "first dump".into());
                let ch = match watch {
                    Some((x, y, Some(b), a)) if b != a => format!("{ch}; z({x},{y}) {b} -> {a}"),
                    Some((x, y, None, a)) => format!("{ch}; z({x},{y}) {a}"),
                    _ => ch,
                };
                out.push(format!("depth {ch}"));
            }
            Err(e) => out.push(format!("depth dump failed: {e}")),
        }
    }
    if let Some(d) = super::exec::depth_surface(cp).filter(|d| super::images::has_stencil(d.format)) {
        let r = super::images::with(|im| -> Result<Vec<u8>, String> {
            let t = im.target(&d)?;
            im.read_back_aspect(t.image, t.key.format, ash::vk::ImageAspectFlags::STENCIL, t.key.width, t.key.height)
        });
        match r {
            Ok(raw) => {
                let nz = raw.iter().filter(|&&v| v != 0).count();
                let ch = changed((d.base, true), &raw, 1).map(|n| format!(", {n} changed")).unwrap_or_default();
                out.push(format!("stencil {nz} nonzero{ch}"));
                if std::env::var("DS_CAPTURE_DRAWS_PNG").ok().as_deref() == Some("1") {
                    if let Some(dir) = crate::run_dir().map(|r| r.join("draws")) {
                        let gray: Vec<u8> = raw.iter().flat_map(|&v| [v, v, v, 255]).collect();
                        let _ = std::fs::write(dir.join(format!("{event:04}_{:010x}_stencil.png", d.base)), super::png::encode(d.width, d.height, &gray));
                    }
                }
            }
            Err(e) => out.push(format!("stencil dump failed: {e}")),
        }
    }
    (!out.is_empty()).then(|| out.join("; "))
}

/// Records one draw (`what` names the packet and its arguments).
pub fn draw(cp: &Cp, what: &str) {
    let mut g = ACTIVE.lock();
    let Some(c) = g.as_mut() else { return };
    c.events += 1;
    let mut s = String::new();
    let ctx = |a: u32| cp.reg(Space::Ctx, (a - 0x28000) / 4);
    let _ = writeln!(
        s,
        "draw #{} [{}] {what} prim {} stages {:#x} ge_cntl {:#x} -> {}",
        c.events,
        cp.name,
        cp.reg(Space::Uc, 0x242),
        ctx(0x28B54),
        cp.reg(Space::Uc, 0x25B),
        cp.last_draw
    );
    let _ = writeln!(s, "  packet address {:#x}", cp.packet_address());
    if dumps_env().is_some() || ps_listed(cp) || vs_listed(cp) || WATCH_PIXEL.lock().is_some() {
        if let Some(n) = dump_draw_target(cp, c.events) {
            let _ = writeln!(s, "    target after draw: {n}");
        }
        if std::env::var("DS_VS_DUMP").is_ok() {
            if let Some(n) = vs_dump_report(cp, c.events) {
                let _ = writeln!(s, "    {n}");
            }
        }
    }
    // NGG primitive shaders: program in the ES registers, resources and user
    // data in the GS ones (gfx10.3 register database: RSRC1/2_GS 0x8A/0x8B,
    // USER_DATA_GS_0 0x8C).
    for (name, lo, rsrc, user) in [("ps", 0x08, 0x0A, 0x0C), ("vs", 0x48, 0x4A, 0x4C), ("gs", 0x88, 0x8A, 0x8C), ("es", 0xC8, 0x8A, 0x8C), ("hs", 0x108, 0x10A, 0x10C), ("ls", 0x148, 0x14A, 0x14C)] {
        stage(cp, &mut s, &mut c.shaders, name, lo, rsrc, user);
    }
    if let Some(d) = super::exec::depth_surface(cp) { c.targets.insert(d.base); }
    let mask = ctx(0x28238);
    for i in 0..8u32 {
        let info = ctx(0x28C70 + 0x3C * i);
        if (info >> 2) & 0x1F == 0 || (mask >> (4 * i)) & 0xF == 0 {
            continue;
        }
        let base = ((ctx(0x28C60 + 0x3C * i) as u64) << 8) | ((ctx(0x28E40 + 4 * i) as u64 & 0xFF) << 40);
        c.targets.insert(base);
        let a2 = ctx(0x28EC0 + 4 * i);
        let a3 = ctx(0x28EE0 + 4 * i);
        // Metadata: DCC (INFO bit 28) and CMASK surfaces, fast-clear color.
        let dcc = ((ctx(0x28C94 + 0x3C * i) as u64) << 8) | ((ctx(0x28EA0 + 4 * i) as u64 & 0xFF) << 40);
        let cmask = ((ctx(0x28C7C + 0x3C * i) as u64) << 8) | ((ctx(0x28E60 + 4 * i) as u64 & 0xFF) << 40);
        let _ = writeln!(
            s,
            "    cb{i} {base:#x} info {info:#010x} view {:#x} attrib {:#x} {}x{} sw_mode {} type {} mask {:x} dcc {dcc:#x} ctl {:#x} cmask {cmask:#x} clear {:#x} {:#x}",
            ctx(0x28C6C + 0x3C * i),
            ctx(0x28C74 + 0x3C * i),
            ((a2 >> 14) & 0x3FFF) + 1,
            (a2 & 0x3FFF) + 1,
            (a3 >> 14) & 0x1F,
            (a3 >> 24) & 3,
            (mask >> (4 * i)) & 0xF,
            ctx(0x28C78 + 0x3C * i),
            ctx(0x28C8C + 0x3C * i),
            ctx(0x28C90 + 0x3C * i)
        );
    }
    // Depth state: DB_DEPTH_CONTROL (stencil, z test, z write, zfunc), the
    // depth clear value, PA_SU_SC_MODE_CNTL polygon offset enables and the
    // PA_SU_POLY_OFFSET registers (DB_FMT_CNTL, clamp, front/back scale and
    // offset).
    let f = |a: u32| f32::from_bits(ctx(a));
    let dctl = ctx(0x28800);
    let _ = writeln!(
        s,
        "    depth control {dctl:#x} (z test {} write {} func {}, stencil {}), clear {}, poly offset {}{} fmt {:#x} clamp {} front {} x16 + {} back {} x16 + {}",
        (dctl >> 1) & 1,
        (dctl >> 2) & 1,
        (dctl >> 4) & 7,
        dctl & 1,
        f(0x2802C),
        if ctx(0x28814) & (1 << 11) != 0 { "front" } else { "" },
        if ctx(0x28814) & (1 << 12) != 0 { " back" } else { "" },
        ctx(0x28B78),
        f(0x28B7C),
        f(0x28B80),
        f(0x28B84),
        f(0x28B88),
        f(0x28B8C)
    );
    for (name, r) in [("es", super::shaders::prepare_ngg(cp)), ("ps", super::shaders::prepare_ps(cp))] {
        match r {
            Ok(p) => {
                for l in super::shaders::describe_resources(&p) {
                    let _ = writeln!(s, "      {name} {l}");
                }
                if super::exec::enabled() {
                    for r in p.resources.iter() {
                        if let super::shaders::Resolved::Image(t) = r {
                            let d = super::images::with(|im| {
                                let d = im.describe(t);
                                match im.gpu_zero_stats(t) {
                                    Some(z) => format!("{d}; {z}"),
                                    None => d,
                                }
                            });
                            let _ = writeln!(s, "      {name} image {:#x}: {d}", t.base);
                        }
                    }
                }
                // Pixel watch: images the stage writes through image stores
                // (e.g. decals updating the G-buffer), where they changed.
                if let (Some((x, y)), true) = (*WATCH_PIXEL.lock(), super::exec::enabled()) {
                    for (i, r) in p.resources.iter().enumerate() {
                        let super::shaders::Resolved::Image(t) = r else { continue };
                        if !p.tr.plan.resources.get(i).is_some_and(|x| x.written) {
                            continue;
                        }
                        if let Some(w) = watch_image(t, x, y, true) {
                            let _ = writeln!(s, "      {name} image after draw: {:#x}{w}", t.base);
                        }
                    }
                }
                // Small constant buffers (per-draw data such as a decal's
                // transform), as floats.
                for (i, r) in p.resources.iter().enumerate() {
                    let super::shaders::Resolved::Buffer(v) = r else { continue };
                    let Some(n) = v.byte_size().filter(|&n| n > 0 && n <= 256) else { continue };
                    if !crate::gpu::cp::guest_readable(v.base, n) {
                        continue;
                    }
                    let words: Vec<String> = (0..n / 4).map(|k| f32::from_bits(unsafe { std::ptr::read_volatile((v.base + 4 * k) as *const u32) })).map(|x| format!("{x}")).collect();
                    let _ = writeln!(s, "      {name} {i} constants @{:#x}: {}", v.base, words.join(" "));
                }
                if name == "es" {
                    vertex_data(&mut s, &p);
                }
                for l in super::shaders::describe_loads(&p) {
                    let _ = writeln!(s, "      {name} {l}");
                }
                // The SPIR-V used at run time (specialized to this draw).
                if let Some(dir) = crate::run_dir() {
                    let sdir = dir.join("shaders");
                    let _ = std::fs::create_dir_all(&sdir);
                    let path = sdir.join(format!("{:012x}-{:016x}.spv", p.addr, p.tr.key));
                    if !path.exists() {
                        let bytes: Vec<u8> = p.tr.out.spirv.iter().flat_map(|w| w.to_le_bytes()).collect();
                        let _ = std::fs::write(&path, bytes);
                    }
                    let _ = writeln!(s, "      {name} spirv {}", path.file_name().unwrap().to_string_lossy());
                }
                note_textures(&mut c.textures, &p);
            }
            Err(e) => {
                let _ = writeln!(s, "      {name} resources unresolved: {e}");
            }
        }
    }
    let zinfo = ctx(0x28040);
    if zinfo & 3 != 0 {
        // HTILE (Z_INFO bit 29), fast-clear values.
        let htile = ((ctx(0x28014) as u64) << 8) | ((ctx(0x28078) as u64 & 0xFF) << 40);
        let _ = writeln!(
            s,
            "    depth {:#x} z_info {zinfo:#010x} stencil {:#x} s_info {:#x} size {:#x} control {:#x} htile {htile:#x} clear z {} s {:#x}",
            (ctx(0x28048) as u64) << 8,
            ((ctx(0x2804C) as u64) << 8) | ((ctx(0x2806C) as u64 & 0xFF) << 40),
            ctx(0x28044),
            ctx(0x2801C),
            ctx(0x28800),
            f32::from_bits(ctx(0x2802C)),
            ctx(0x28028)
        );
    }
    let _ = writeln!(
        s,
        "    viewport scale ({}, {}, {}) offset ({}, {}, {}) scissor {:#x}..{:#x} col_format {:#x} shader_mask {:#x} cb_control {:#x} blend0 {:#x} ps_input {:#x}/{:#x}",
        f(ctx(0x2843C)),
        f(ctx(0x28444)),
        f(ctx(0x2844C)),
        f(ctx(0x28440)),
        f(ctx(0x28448)),
        f(ctx(0x28450)),
        ctx(0x28204),
        ctx(0x28208),
        ctx(0x28714),
        ctx(0x2823C),
        ctx(0x28808),
        ctx(0x28780),
        ctx(0x286CC),
        ctx(0x286D0)
    );
    let inputs: Vec<String> = (0..(ctx(0x286D8) & 0x3F).min(8)).map(|n| format!("{:#x}", ctx(0x28644 + 4 * n))).collect();
    let _ = writeln!(s, "    ps_in_control {:#x} input_cntl [{}] z_format {:#x}", ctx(0x286D8), inputs.join(" "), ctx(0x28710));
    // Raster, depth/stencil and blend state.
    let blends: Vec<String> = (0..8u32).filter(|i| (mask >> (4 * i)) & 0xF != 0).map(|i| format!("{i}:{:#x}", ctx(0x28780 + 4 * i))).collect();
    let _ = writeln!(
        s,
        "    state vte {:#x} clip {:#x} su_sc_mode {:#x} sc_mode0 {:#x} aa {:#x} win_off {:#x} | screen_sc {:#x}..{:#x} generic_sc {:#x}..{:#x} vport_sc {:#x}..{:#x} zrange {}..{} | db_render {:#x} db_shader {:#x} alpha_to_mask {:#x} depth_ctl {:#x} stencil_ctl {:#x} ref {:#x}/{:#x} z_read {:#x} s_info {:#x} | target_mask {mask:#x} blend [{}] blend_color ({}, {}, {}, {}) | index_type {:#x} instances {} reset_en {:#x} gb_clip {}/{}",
        ctx(0x28818),
        ctx(0x28810),
        ctx(0x28814),
        ctx(0x28A48),
        ctx(0x28BE0),
        ctx(0x28200),
        ctx(0x28030),
        ctx(0x28034),
        ctx(0x28240),
        ctx(0x28244),
        ctx(0x28250),
        ctx(0x28254),
        f(ctx(0x282D0)),
        f(ctx(0x282D4)),
        ctx(0x28000),
        ctx(0x2880C),
        ctx(0x28B70),
        ctx(0x28800),
        ctx(0x2842C),
        ctx(0x28430),
        ctx(0x28434),
        ((ctx(0x28048) as u64) << 8) | ((ctx(0x28068) as u64 & 0xFF) << 40),
        ctx(0x28044),
        blends.join(" "),
        f(ctx(0x28414)),
        f(ctx(0x28418)),
        f(ctx(0x2841C)),
        f(ctx(0x28420)),
        cp.reg(Space::Uc, 0x243),
        cp.num_instances(),
        ctx(0x28A94),
        f(ctx(0x28BE8)),
        f(ctx(0x28BF0))
    );
    c.text.push_str(&s);
}

/// Records a synchronization or memory packet.
pub fn packet(cp: &Cp, what: &str) {
    let mut g = ACTIVE.lock();
    let Some(c) = g.as_mut() else { return };
    let _ = writeln!(c.text, "packet [{}] {what}", cp.name);
}

/// Records one dispatch.
pub fn dispatch(cp: &Cp, what: &str) {
    let mut g = ACTIVE.lock();
    let Some(c) = g.as_mut() else { return };
    c.events += 1;
    let mut s = String::new();
    let _ = writeln!(
        s,
        "dispatch #{} [{}] {what} threads {}x{}x{}",
        c.events,
        cp.name,
        cp.reg(Space::Sh, 0x207) & 0xFFFF,
        cp.reg(Space::Sh, 0x208) & 0xFFFF,
        cp.reg(Space::Sh, 0x209) & 0xFFFF
    );
    let _ = writeln!(s, "  packet address {:#x}", cp.packet_address());
    stage(cp, &mut s, &mut c.shaders, "cs", 0x20C, 0x212, 0x240);
    let dumps = DISPATCH_DUMPS.load(Ordering::Relaxed) || std::env::var("DS_CAPTURE_DISPATCHES").ok().as_deref() == Some("1");
    match super::shaders::prepare_compute(cp, true) {
        Ok(p) => {
            for l in super::shaders::describe_resources(&p) {
                let _ = writeln!(s, "      {l}");
            }
            note_textures(&mut c.textures, &p);
            for (i, r) in p.resources.iter().enumerate() {
                if p.tr.plan.resources[i].written {
                    if let super::shaders::Resolved::Image(t) = r { c.targets.insert(t.base); }
                }
            }
            if let Some(limit) = buffer_dump_limit(p.addr).filter(|_| dumps) {
                // The buffers as the completed dispatch left them.
                let dir = crate::run_dir().map(|d| d.join("dispatches"));
                for (i, r) in p.resources.iter().enumerate() {
                    let (super::shaders::Resolved::Buffer(v), Some(dir)) = (r, &dir) else { continue };
                    let Some(n) = v.byte_size().filter(|&n| n > 0 && n <= limit) else { continue };
                    if !super::cp::guest_readable(v.base, n) {
                        continue;
                    }
                    let _ = std::fs::create_dir_all(dir);
                    let name = format!("{}_{:04}_{:010x}_{i}_{:010x}.bin", c.flip, c.events, p.addr, v.base);
                    let bytes = unsafe { std::slice::from_raw_parts(v.base as *const u8, n as usize) };
                    let nz = bytes.iter().filter(|&&b| b != 0).count();
                    let _ = std::fs::write(dir.join(&name), bytes);
                    let _ = writeln!(s, "      buffer after dispatch: {name} ({n} bytes, {nz} non-zero)");
                }
            }
            let watched = *WATCH_PIXEL.lock();
            if !dumps && super::exec::enabled() {
                // Watch-only capture: the watched pixel of each image the
                // dispatch writes or reads, where it changed since the last
                // dispatch that bound the image (one texel read back).
                if let Some((x, y)) = watched {
                    for (i, r) in p.resources.iter().enumerate() {
                        let super::shaders::Resolved::Image(t) = r else { continue };
                        let written = p.tr.plan.resources[i].written;
                        if let Some(w) = watch_image(t, x, y, true) {
                            let _ = writeln!(s, "      image {} dispatch: {:#x}{w}", if written { "after" } else { "read by" }, t.base);
                        }
                    }
                }
            }
            if dumps && super::exec::enabled() {
                let only = IMAGE_DUMPS.lock().clone();
                for (i, r) in p.resources.iter().enumerate() {
                    let super::shaders::Resolved::Image(t) = r else { continue };
                    if !p.tr.plan.resources[i].written {
                        continue;
                    }
                    if !only.is_empty() && !only.contains(&t.base) {
                        if let Some(w) = watched.and_then(|(x, y)| watch_image(t, x, y, true)) {
                            let _ = writeln!(s, "      image after dispatch: {:#x}{w}", t.base);
                        }
                        continue;
                    }
                    let Some(dir) = crate::run_dir().map(|d| d.join("dispatches")) else { continue };
                    let _ = std::fs::create_dir_all(&dir);
                    let name = format!("{}_{:04}_{:010x}_{:010x}.png", c.flip, c.events, p.addr, t.base);
                    if buffer_dump_limit(p.addr).is_some() || !only.is_empty() {
                        // Raw texels as well (mip 0, layer 0; e.g. probed values).
                        if let Ok((raw, w, h, texel)) = super::images::with(|im| im.snapshot_raw(t)) {
                            let _ = std::fs::write(dir.join(name.replace(".png", &format!("_{w}x{h}x{texel}.raw"))), raw);
                        }
                    }
                    // The watched pixel (scaled from the presented image's size).
                    let watch = watched.and_then(|(x, y)| watch_image(t, x, y, false));
                    let r = super::images::with(|im| im.snapshot(t));
                    match r {
                        Ok((rgba, w, h, stats)) => {
                            let _ = std::fs::write(dir.join(&name), super::png::encode(w, h, &rgba));
                            let _ = writeln!(s, "      image after dispatch: {name}{}{}", stats.map(|x| format!(" ({x})")).unwrap_or_default(), watch.unwrap_or_default());
                        }
                        Err(e) => {
                            let _ = writeln!(s, "      image after dispatch: {name} failed: {e}");
                        }
                    }
                }
            }
        }
        Err(e) => {
            let _ = writeln!(s, "      resources unresolved: {e}");
        }
    }
    c.text.push_str(&s);
}
