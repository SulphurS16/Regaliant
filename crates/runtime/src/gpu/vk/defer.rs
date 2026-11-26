//! Deferred command recording (`tune record_defer`): a command processor's
//! batch is recorded into lists of commands instead of through the driver,
//! and the submission thread replays the lists into the batch's command
//! buffers just before it submits the batch. The command processor's thread
//! then copies each command's arguments instead of paying for the driver's
//! recording (the driver was ~15% of the graphics command processor's
//! working time at the tower vista; FINDINGS.md 2026-10-31).
//!
//! The device everything records with (`Ctx::device`) has its command
//! functions replaced by the ones here (`install`): a command for a command
//! buffer this thread claimed (`claim`, from `Exec::begin`) goes to that
//! buffer's list, any other to the driver. Push descriptors, labels and
//! checkpoints come here through `Ctx::push_descriptors`, `label_begin`,
//! `label_end` and `checkpoint`. A command recorded here in a form the lists
//! do not hold (a pNext chain, another descriptor kind) first replays this
//! thread's lists in place (`flush`): the batch goes on through the driver.
//! `DS_RECORD_SHIMS=0` leaves the device as loaded (nothing is deferred);
//! `DS_RECORD_CHECK=1` reports commands other threads send to the driver for
//! a buffer some thread claimed.
//!
//! The design (an enum of commands with argument arenas, replayed on another
//! thread before submission) follows the earlier native port's deferred
//! recording (PROVENANCE.md); claims and replaced device functions are this
//! port's.

use ash::vk;
use ash::vk::Handle;
use parking_lot::Mutex;
use std::cell::{Cell, RefCell};
use std::ffi::{CStr, c_char, c_void};
use std::ptr::null;
use std::sync::OnceLock;
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};

// Diagnostic only: sample 1/64 actual driver calls, rotating the sample across lists.
// Clean relay measurements leave this off. Counts include all replayed commands; wall
// and scheduled CPU times describe only sampled calls (no scaling of rare commands).
const RECORD_CLASSES: [&str; 10] = ["begin/end", "pipeline", "descriptors", "draw", "dispatch", "transfer/clear", "barrier", "rendering", "dynamic-state", "other"];
static RECORD_STATS: [[AtomicU64; 4]; 10] = [const { [const { AtomicU64::new(0) }; 4] }; 10];
fn record_profile() -> bool {
    static ON: OnceLock<bool> = OnceLock::new();
    *ON.get_or_init(|| std::env::var("DS_RECORD_PROFILE").ok().as_deref() == Some("1"))
}
fn record_class(op: &Op) -> usize {
    match op {
        Op::Begin(_) | Op::End => 0,
        Op::BindPipeline(..) => 1,
        Op::PushDescriptors(..) | Op::BindSets(..) | Op::PushConstants(..) => 2,
        Op::Draw(..) | Op::DrawIndexed(..) | Op::DrawIndirect(..) | Op::DrawIndexedIndirect(..) | Op::DrawIndirectCount(..) | Op::DrawIndexedIndirectCount(..) => 3,
        Op::Dispatch(..) | Op::DispatchBase(..) | Op::DispatchIndirect(..) => 4,
        Op::CopyBuffer(..) | Op::CopyImage(..) | Op::BlitImage(..) | Op::CopyBufferToImage(..) | Op::CopyImageToBuffer(..) | Op::UpdateBuffer(..) | Op::FillBuffer(..) | Op::ClearColorImage(..) | Op::ClearDepthStencilImage(..) | Op::ClearAttachments(..) => 5,
        Op::Barrier(..) => 6,
        Op::BeginRendering(..) | Op::EndRendering => 7,
        Op::Viewports(..) | Op::Scissors(..) | Op::LineWidth(..) | Op::DepthBias(..) | Op::DepthBounds(..) | Op::BlendConstants(..) | Op::StencilCompareMask(..) | Op::StencilWriteMask(..) | Op::StencilReference(..) | Op::CullMode(..) | Op::FrontFace(..) | Op::Topology(..) | Op::DepthTestEnable(..) | Op::DepthWriteEnable(..) | Op::DepthCompareOp(..) | Op::DepthBoundsTestEnable(..) | Op::StencilTestEnable(..) | Op::StencilOp(..) | Op::RasterizerDiscardEnable(..) | Op::DepthBiasEnable(..) | Op::PrimitiveRestartEnable(..) => 8,
        _ => 9,
    }
}
fn record_report() -> String {
    if !record_profile() { return String::new(); }
    let mut out = String::new();
    for (i, name) in RECORD_CLASSES.iter().enumerate() {
        let [calls, samples, wall, cycles] = RECORD_STATS[i].each_ref().map(|n| n.load(Ordering::Relaxed));
        out += &format!("    driver-call sample {name}: {calls} calls, {samples} samples, {wall} wall ns, {cycles} CPU ns\n");
    }
    out
}

/// A command's array in its list's arena: offset in 8-byte words, items
/// (bytes for byte data).
#[derive(Clone, Copy)]
struct Span {
    at: u32,
    n: u32,
}

/// No item (`Op::BeginRendering`'s depth and stencil attachments).
const NONE: u32 = u32::MAX;

/// A recorded command: arguments by value, arrays in the arena.
#[derive(Clone, Copy)]
enum Op {
    Begin(vk::CommandBufferUsageFlags),
    End,
    BindPipeline(vk::PipelineBindPoint, vk::Pipeline),
    Viewports(u32, Span),
    Scissors(u32, Span),
    LineWidth(f32),
    DepthBias(f32, f32, f32),
    DepthBounds(f32, f32),
    BlendConstants([f32; 4]),
    StencilCompareMask(vk::StencilFaceFlags, u32),
    StencilWriteMask(vk::StencilFaceFlags, u32),
    StencilReference(vk::StencilFaceFlags, u32),
    BindSets(vk::PipelineBindPoint, vk::PipelineLayout, u32, Span, Span),
    BindIndexBuffer(vk::Buffer, vk::DeviceSize, vk::IndexType),
    BindVertexBuffers(u32, Span, Span),
    Draw(u32, u32, u32, u32),
    DrawIndexed(u32, u32, u32, i32, u32),
    DrawIndirect(vk::Buffer, vk::DeviceSize, u32, u32),
    DrawIndexedIndirect(vk::Buffer, vk::DeviceSize, u32, u32),
    DrawIndirectCount(vk::Buffer, vk::DeviceSize, vk::Buffer, vk::DeviceSize, u32, u32),
    DrawIndexedIndirectCount(vk::Buffer, vk::DeviceSize, vk::Buffer, vk::DeviceSize, u32, u32),
    Dispatch(u32, u32, u32),
    DispatchBase([u32; 6]),
    DispatchIndirect(vk::Buffer, vk::DeviceSize),
    CopyBuffer(vk::Buffer, vk::Buffer, Span),
    CopyImage(vk::Image, vk::ImageLayout, vk::Image, vk::ImageLayout, Span),
    BlitImage(vk::Image, vk::ImageLayout, vk::Image, vk::ImageLayout, Span, vk::Filter),
    CopyBufferToImage(vk::Buffer, vk::Image, vk::ImageLayout, Span),
    CopyImageToBuffer(vk::Image, vk::ImageLayout, vk::Buffer, Span),
    UpdateBuffer(vk::Buffer, vk::DeviceSize, Span),
    FillBuffer(vk::Buffer, vk::DeviceSize, vk::DeviceSize, u32),
    ClearColorImage(vk::Image, vk::ImageLayout, vk::ClearColorValue, Span),
    ClearDepthStencilImage(vk::Image, vk::ImageLayout, vk::ClearDepthStencilValue, Span),
    ClearAttachments(Span, Span),
    Barrier(vk::PipelineStageFlags, vk::PipelineStageFlags, vk::DependencyFlags, Span, Span, Span),
    BeginQuery(vk::QueryPool, u32, vk::QueryControlFlags),
    EndQuery(vk::QueryPool, u32),
    ResetQueryPool(vk::QueryPool, u32, u32),
    WriteTimestamp(vk::PipelineStageFlags, vk::QueryPool, u32),
    CopyQueryPoolResults(vk::QueryPool, u32, u32, vk::Buffer, vk::DeviceSize, vk::DeviceSize, vk::QueryResultFlags),
    PushConstants(vk::PipelineLayout, vk::ShaderStageFlags, u32, Span),
    /// Flags, area, layers, view mask, color attachments, depth and stencil
    /// attachment (arena word or `NONE`).
    BeginRendering(vk::RenderingFlags, vk::Rect2D, u32, u32, Span, u32, u32),
    EndRendering,
    CullMode(vk::CullModeFlags),
    FrontFace(vk::FrontFace),
    Topology(vk::PrimitiveTopology),
    DepthTestEnable(vk::Bool32),
    DepthWriteEnable(vk::Bool32),
    DepthCompareOp(vk::CompareOp),
    DepthBoundsTestEnable(vk::Bool32),
    StencilTestEnable(vk::Bool32),
    StencilOp(vk::StencilFaceFlags, vk::StencilOp, vk::StencilOp, vk::StencilOp, vk::CompareOp),
    RasterizerDiscardEnable(vk::Bool32),
    DepthBiasEnable(vk::Bool32),
    PrimitiveRestartEnable(vk::Bool32),
    /// The writes' array pointers hold arena words + 1 (0: none) until the
    /// replay points them into the arena.
    PushDescriptors(vk::PipelineBindPoint, vk::PipelineLayout, u32, Span),
    LabelBegin(Span),
    LabelEnd,
    Checkpoint(u64),
    /// Host code that records into the command buffer itself (`external`: the upscalers'
    /// libraries), by its index in `List::externals`.
    External(u32),
}

/// Host code run with the real command buffer at replay (`external`).
pub type ExternalFn = Box<dyn FnOnce(vk::CommandBuffer) + Send>;

/// A claimed command buffer's commands, in order.
#[derive(Default)]
pub struct List {
    ops: Vec<Op>,
    /// The commands' arrays (8-byte words: no structure here needs more
    /// alignment).
    arena: Vec<u64>,
    /// Host code recorded as commands (`Op::External`), taken at replay.
    externals: Vec<Option<ExternalFn>>,
    /// Seamless chunks (`seam_mark`): the ops [begin, end) of the batch
    /// close's write-back (SEAM_WB), its leading full barrier (SEAM_LEAD) and
    /// its host barrier (SEAM_TAIL).
    seams: [(u32, u32); 3],
    /// The parallel run the batch is a pure part of (`set_part`; 0: not one).
    pure_run: u64,
    /// The parallel run the batch is a part of, pure or not (0: none).
    run: u64,
}

impl List {
    /// Copies `n` items at `p` into the arena.
    fn put<T: Copy>(&mut self, p: *const T, n: u32) -> Span {
        const { assert!(std::mem::align_of::<T>() <= 8) };
        let at = self.arena.len();
        let bytes = n as usize * std::mem::size_of::<T>();
        if bytes == 0 || p.is_null() {
            return Span { at: at as u32, n: 0 };
        }
        let words = bytes.div_ceil(8);
        self.arena.reserve(words);
        unsafe {
            let dst = self.arena.as_mut_ptr().add(at);
            dst.add(words - 1).write(0);
            std::ptr::copy_nonoverlapping(p.cast::<u8>(), dst.cast::<u8>(), bytes);
            self.arena.set_len(at + words);
        }
        Span { at: at as u32, n }
    }

    /// A string's bytes and a NUL.
    fn put_str(&mut self, s: &str) -> Span {
        let at = self.arena.len();
        let words = (s.len() + 1).div_ceil(8);
        self.arena.resize(at + words, 0);
        unsafe { std::ptr::copy_nonoverlapping(s.as_ptr(), self.arena.as_mut_ptr().add(at).cast::<u8>(), s.len()) };
        Span { at: at as u32, n: s.len() as u32 + 1 }
    }

    /// The arena's items of `s`.
    fn items<T>(&mut self, s: Span) -> &mut [T] {
        if s.n == 0 {
            return &mut [];
        }
        unsafe { std::slice::from_raw_parts_mut(self.arena.as_mut_ptr().add(s.at as usize).cast::<T>(), s.n as usize) }
    }

    /// An arena word + 1 for `n` items at `p` (0 for none: `Op::PushDescriptors`).
    fn put_ref<T: Copy>(&mut self, p: *const T, n: u32) -> usize {
        if p.is_null() || n == 0 { 0 } else { self.put(p, n).at as usize + 1 }
    }
}

/// Command buffers one thread claims at once: a batch's commands and its
/// prologue.
const CLAIMS: usize = 2;

thread_local! {
    /// The command buffers this thread records into lists (raw handles; 0
    /// for none).
    static CLAIMED: Cell<[u64; CLAIMS]> = const { Cell::new([0; CLAIMS]) };
    static LISTS: RefCell<[Option<Box<List>>; CLAIMS]> = const { RefCell::new([None, None]) };
}

/// The device as loaded: replays, and commands for buffers this thread did
/// not claim.
static REAL: OnceLock<ash::Device> = OnceLock::new();
static INSTALLED: AtomicBool = AtomicBool::new(false);
/// `DS_RECORD_CHECK=1`: claims of every thread (`ANY`) checked against
/// the commands sent to the driver.
static CHECK: AtomicBool = AtomicBool::new(false);
static ANY: [AtomicU64; 64] = [const { AtomicU64::new(0) }; 64];

static STATS: [AtomicU64; 10] = [const { AtomicU64::new(0) }; 10];
/// Batches replayed, their commands, arena words, replay ns, most commands
/// in one batch, batches discarded, claims replayed in place (`flush`),
/// commands other threads sent to the driver for claimed buffers, claims
/// begun over unreleased ones, commands of a form the lists do not hold.
const S_BATCHES: usize = 0;
const S_OPS: usize = 1;
const S_WORDS: usize = 2;
const S_NS: usize = 3;
const S_MAX_OPS: usize = 4;
const S_DISCARDED: usize = 5;
const S_FLUSHES: usize = 6;
const S_FOREIGN: usize = 7;
const S_STALE: usize = 8;
const S_UNHELD: usize = 9;

#[inline(always)]
fn real() -> &'static ash::Device {
    REAL.get().expect("deferred recording without the device")
}

#[inline(always)]
fn claimed(cb: vk::CommandBuffer) -> Option<usize> {
    let raw = cb.as_raw();
    CLAIMED.with(Cell::get).iter().position(|&h| h == raw && raw != 0)
}

/// Appends a command to `cb`'s list if this thread claimed it.
#[inline(always)]
fn record(cb: vk::CommandBuffer, f: impl FnOnce(&mut List)) -> bool {
    let Some(i) = claimed(cb) else {
        return false;
    };
    LISTS.with(|l| {
        if let Some(l) = l.borrow_mut()[i].as_deref_mut() {
            f(l);
        }
    });
    true
}

/// A command for `cb` goes to the driver (`DS_RECORD_CHECK=1`: counted if
/// another thread claimed `cb`).
#[inline(always)]
pub fn direct(cb: vk::CommandBuffer) {
    if CHECK.load(Ordering::Relaxed) {
        check_direct(cb);
    }
}

#[cold]
fn check_direct(cb: vk::CommandBuffer) {
    let raw = cb.as_raw();
    if raw != 0 && ANY.iter().any(|h| h.load(Ordering::Relaxed) == raw) && STATS[S_FOREIGN].fetch_add(1, Ordering::Relaxed) == 0 {
        crate::error!("record_defer: a command for a command buffer another thread claimed went to the driver (thread {})", crate::log::thread_name());
    }
}

/// A command for a claimed buffer in a form the lists do not hold: this
/// thread's lists are replayed in place (`flush`) and it goes to the driver.
#[cold]
fn unheld(what: &str) {
    if STATS[S_UNHELD].fetch_add(1, Ordering::Relaxed) == 0 {
        crate::warn!("record_defer: {what} (the batch goes on through the driver)");
    }
    flush();
}

/// The device to record with: `device` with its command functions replaced
/// by the ones here (`DS_RECORD_SHIMS=0`: `device` as loaded).
pub fn install(device: ash::Device) -> ash::Device {
    if std::env::var("DS_RECORD_SHIMS").is_ok_and(|v| v.trim() == "0") {
        return device;
    }
    CHECK.store(std::env::var("DS_RECORD_CHECK").is_ok_and(|v| v.trim() == "1"), Ordering::Relaxed);
    let (mut f0, mut f1, mut f2, mut f3) = (device.fp_v1_0().clone(), device.fp_v1_1().clone(), device.fp_v1_2().clone(), device.fp_v1_3().clone());
    f0.begin_command_buffer = begin_command_buffer;
    f0.end_command_buffer = end_command_buffer;
    f0.cmd_bind_pipeline = bind_pipeline;
    f0.cmd_set_viewport = set_viewport;
    f0.cmd_set_scissor = set_scissor;
    f0.cmd_set_line_width = set_line_width;
    f0.cmd_set_depth_bias = set_depth_bias;
    f0.cmd_set_blend_constants = set_blend_constants;
    f0.cmd_set_depth_bounds = set_depth_bounds;
    f0.cmd_set_stencil_compare_mask = set_stencil_compare_mask;
    f0.cmd_set_stencil_write_mask = set_stencil_write_mask;
    f0.cmd_set_stencil_reference = set_stencil_reference;
    f0.cmd_bind_descriptor_sets = bind_descriptor_sets;
    f0.cmd_bind_index_buffer = bind_index_buffer;
    f0.cmd_bind_vertex_buffers = bind_vertex_buffers;
    f0.cmd_draw = draw;
    f0.cmd_draw_indexed = draw_indexed;
    f0.cmd_draw_indirect = draw_indirect;
    f0.cmd_draw_indexed_indirect = draw_indexed_indirect;
    f0.cmd_dispatch = dispatch;
    f0.cmd_dispatch_indirect = dispatch_indirect;
    f0.cmd_copy_buffer = copy_buffer;
    f0.cmd_copy_image = copy_image;
    f0.cmd_blit_image = blit_image;
    f0.cmd_copy_buffer_to_image = copy_buffer_to_image;
    f0.cmd_copy_image_to_buffer = copy_image_to_buffer;
    f0.cmd_update_buffer = update_buffer;
    f0.cmd_fill_buffer = fill_buffer;
    f0.cmd_clear_color_image = clear_color_image;
    f0.cmd_clear_depth_stencil_image = clear_depth_stencil_image;
    f0.cmd_clear_attachments = clear_attachments;
    f0.cmd_pipeline_barrier = pipeline_barrier;
    f0.cmd_begin_query = begin_query;
    f0.cmd_end_query = end_query;
    f0.cmd_reset_query_pool = reset_query_pool;
    f0.cmd_write_timestamp = write_timestamp;
    f0.cmd_copy_query_pool_results = copy_query_pool_results;
    f0.cmd_push_constants = push_constants;
    f1.cmd_dispatch_base = dispatch_base;
    f2.cmd_draw_indirect_count = draw_indirect_count;
    f2.cmd_draw_indexed_indirect_count = draw_indexed_indirect_count;
    f3.cmd_begin_rendering = begin_rendering;
    f3.cmd_end_rendering = end_rendering;
    f3.cmd_set_cull_mode = set_cull_mode;
    f3.cmd_set_front_face = set_front_face;
    f3.cmd_set_primitive_topology = set_primitive_topology;
    f3.cmd_set_depth_test_enable = set_depth_test_enable;
    f3.cmd_set_depth_write_enable = set_depth_write_enable;
    f3.cmd_set_depth_compare_op = set_depth_compare_op;
    f3.cmd_set_depth_bounds_test_enable = set_depth_bounds_test_enable;
    f3.cmd_set_stencil_test_enable = set_stencil_test_enable;
    f3.cmd_set_stencil_op = set_stencil_op;
    f3.cmd_set_rasterizer_discard_enable = set_rasterizer_discard_enable;
    f3.cmd_set_depth_bias_enable = set_depth_bias_enable;
    f3.cmd_set_primitive_restart_enable = set_primitive_restart_enable;
    let shimmed = ash::Device::from_parts_1_3(device.handle(), f0, f1, f2, f3);
    REAL.set(device).ok();
    INSTALLED.store(true, Ordering::Relaxed);
    shimmed
}

/// Whether an executor's batch beginning now is recorded into lists (`tune
/// record_defer`): `bit` its class's (1 the graphics command processor, 2
/// the compute ones, 4 the graphics queue's parallel workers, 8 the compute
/// queues').
pub fn wanted(bit: u64) -> bool {
    INSTALLED.load(Ordering::Relaxed) && crate::gpu::tune::record_defer() & bit != 0
}

static FREE: Mutex<Vec<Box<List>>> = Mutex::new(Vec::new());

fn new_list() -> Box<List> {
    FREE.lock().pop().unwrap_or_default()
}

/// Keeps a replayed list's memory for the next claim (not a huge one's).
fn recycle(mut l: Box<List>) {
    if l.arena.capacity() > 1 << 24 || l.ops.capacity() > 1 << 20 {
        return;
    }
    l.ops.clear();
    l.arena.clear();
    l.externals.clear();
    l.seams = [(0, 0); 3];
    l.pure_run = 0;
    l.run = 0;
    let mut f = FREE.lock();
    if f.len() < 32 {
        f.push(l);
    }
}

/// This thread records the batch's command buffers (its commands, its
/// prologue) into lists from here (`Exec::begin`), until `release`,
/// `abandon` or `flush`.
pub fn claim(cbs: [vk::CommandBuffer; CLAIMS]) {
    let raw = cbs.map(|c| c.as_raw());
    let old = CLAIMED.with(|c| c.replace(raw));
    if old != [0; CLAIMS] && STATS[S_STALE].fetch_add(1, Ordering::Relaxed) == 0 {
        crate::error!("record_defer: a claim over an unreleased one (its commands are dropped)");
    }
    LISTS.with(|l| {
        for x in l.borrow_mut().iter_mut() {
            if let Some(o) = x.take() {
                recycle(o);
            }
            *x = Some(new_list());
        }
    });
    if CHECK.load(Ordering::Relaxed) {
        for h in old.into_iter().filter(|&h| h != 0) {
            let _ = ANY.iter().find(|s| s.compare_exchange(h, 0, Ordering::Relaxed, Ordering::Relaxed).is_ok());
        }
        for h in raw.into_iter().filter(|&h| h != 0) {
            let _ = ANY.iter().find(|s| s.compare_exchange(0, h, Ordering::Relaxed, Ordering::Relaxed).is_ok());
        }
    }
}

/// A batch's lists for the submission thread (`Lists::replay`).
pub struct Lists([(u64, Option<Box<List>>); CLAIMS]);

/// The lists of this thread's claimed command buffers; the claims end.
pub fn release() -> Lists {
    let raw = CLAIMED.with(|c| c.replace([0; CLAIMS]));
    if CHECK.load(Ordering::Relaxed) {
        for h in raw.into_iter().filter(|&h| h != 0) {
            let _ = ANY.iter().find(|s| s.compare_exchange(h, 0, Ordering::Relaxed, Ordering::Relaxed).is_ok());
        }
    }
    let [a, b] = LISTS.with(|l| std::mem::take(&mut *l.borrow_mut()));
    Lists([(raw[0], a), (raw[1], b)])
}

/// Ends this thread's claims and drops their lists (`Exec::discard`).
pub fn abandon() {
    for (_, l) in release().0 {
        if let Some(l) = l {
            recycle(l);
        }
    }
    STATS[S_DISCARDED].fetch_add(1, Ordering::Relaxed);
}

/// Replays this thread's lists into their command buffers now and ends the
/// claims: the rest of the batch is recorded through the driver.
pub fn flush() {
    STATS[S_FLUSHES].fetch_add(1, Ordering::Relaxed);
    release().play();
}

impl Lists {
    /// Records the lists into their command buffers through the driver (the
    /// submission thread, just before the batch is submitted).
    pub fn replay(self) {
        let _trace = super::super::timeline::section("replay commands");
        let t = std::time::Instant::now();
        let (ops, words) = self.play();
        STATS[S_BATCHES].fetch_add(1, Ordering::Relaxed);
        STATS[S_OPS].fetch_add(ops, Ordering::Relaxed);
        STATS[S_WORDS].fetch_add(words, Ordering::Relaxed);
        STATS[S_NS].fetch_add(t.elapsed().as_nanos() as u64, Ordering::Relaxed);
        STATS[S_MAX_OPS].fetch_max(ops, Ordering::Relaxed);
    }

    /// `replay` with the commands' list played without the marked parts
    /// `cuts` (seamless chunks).
    pub fn replay_cut(mut self, cuts: &[Cut]) {
        self.play_part(1, None, 0, &[]);
        self.play_part(0, None, 0, cuts);
    }

    /// Plays each list into its buffer: commands and arena words played.
    fn play(self) -> (u64, u64) {
        let (mut ops, mut words) = (0, 0);
        for (raw, l) in self.0 {
            let Some(mut l) = l else { continue };
            if raw != 0 && !l.ops.is_empty() {
                ops += l.ops.len() as u64;
                words += l.arena.len() as u64;
                unsafe { play_list(vk::CommandBuffer::from_raw(raw), &mut l, 0) };
            }
            recycle(l);
        }
        (ops, words)
    }

    /// Whether the batch's lists hold its commands (fusion needs them).
    pub fn has_main(&self) -> bool {
        self.0[0].0 != 0 && self.0[0].1.is_some()
    }
}

/// Seamless chunks: a batch's close write-back, leading full barrier and host
/// barrier (`List::seams`).
pub const SEAM_WB: usize = 0;
pub const SEAM_LEAD: usize = 1;
pub const SEAM_TAIL: usize = 2;

/// The ops of `cb`'s batch's part `kind` begin (`begin`) or end here in its
/// command list (its commands' or its prologue's; nothing when this thread
/// records `cb` through the driver).
pub fn seam_mark(cb: vk::CommandBuffer, kind: usize, begin: bool) {
    let Some(i) = claimed(cb) else { return };
    LISTS.with(|l| {
        if let Some(list) = l.borrow_mut()[i].as_mut() {
            let n = list.ops.len() as u32;
            if begin {
                list.seams[kind] = (n, n);
            } else {
                list.seams[kind].1 = n;
            }
        }
    });
}

/// This thread's batch is a part of parallel run `run`, pure or not
/// (seamless chunks: `Exec::submit_for`).
pub fn set_part(run: u64, pure: bool) {
    LISTS.with(|l| {
        if let Some(main) = l.borrow_mut()[0].as_mut() {
            main.run = run;
            main.pure_run = if pure { run } else { 0 };
        }
    });
}

/// A batch's seams for the fused replay: whether its write-back, leading
/// barrier and host barrier were marked, and the run it is a pure part of.
#[derive(Clone, Copy, Default)]
pub struct Seams {
    pub has: [bool; 3],
    pub pure_run: u64,
    pub run: u64,
}

/// A cut of a list's ops at replay: the ops' range of a part (`SEAM_*`), and
/// whether an attachment-only barrier (by region) stands in for them.
#[derive(Clone, Copy)]
pub struct Cut {
    pub kind: usize,
    pub light: bool,
}

impl Lists {
    /// The batch's prologue buffer when its list holds commands (0: none).
    pub fn pre_raw(&self) -> u64 {
        if self.0[1].1.as_ref().is_some_and(|l| !l.ops.is_empty()) { self.0[1].0 } else { 0 }
    }

    /// The batch's prologue's marked barriers (`seam_mark`: SEAM_LEAD its
    /// leading one, SEAM_TAIL its trailing one).
    pub fn pre_seams(&self) -> [bool; 3] {
        self.0[1].1.as_ref().map_or([false; 3], |l| l.seams.map(|s| s.1 > s.0))
    }

    /// The batch's seams (`seam_mark`, `set_pure`).
    pub fn seams(&self) -> Seams {
        self.0[0].1.as_ref().map_or(Seams::default(), |l| Seams { has: l.seams.map(|s| s.1 > s.0), pure_run: l.pure_run, run: l.run })
    }

    /// Plays one of the batch's lists (0 its commands, 1 its prologue) into
    /// `into` (None: its own buffer) with `skip` (`SKIP_BEGIN`,
    /// `SKIP_END`), without the marked parts `cuts` name.
    pub fn play_part(&mut self, which: usize, into: Option<vk::CommandBuffer>, skip: u8, cuts: &[Cut]) {
        let _trace = super::super::timeline::section("replay commands");
        let (raw, slot) = &mut self.0[which];
        let Some(mut l) = slot.take() else { return };
        let t = std::time::Instant::now();
        if *raw != 0 && !l.ops.is_empty() {
            let cb = into.unwrap_or(vk::CommandBuffer::from_raw(*raw));
            let mut ranges: [(u32, u32, bool); 3] = [(0, 0, false); 3];
            for (k, c) in cuts.iter().enumerate().take(3) {
                let s = l.seams[c.kind];
                ranges[k] = (s.0, s.1, c.light);
            }
            STATS[S_OPS].fetch_add(l.ops.len() as u64, Ordering::Relaxed);
            STATS[S_WORDS].fetch_add(l.arena.len() as u64, Ordering::Relaxed);
            unsafe { play_list_cut(cb, &mut l, skip, &ranges[..cuts.len().min(3)]) };
        }
        if which == 0 {
            STATS[S_BATCHES].fetch_add(1, Ordering::Relaxed);
        }
        STATS[S_NS].fetch_add(t.elapsed().as_nanos() as u64, Ordering::Relaxed);
        recycle(l);
    }
}

/// Seamless chunks' attachment-only barrier (`Cut::light`): earlier
/// attachment writes before later attachment accesses, by region.
unsafe fn light_barrier(cb: vk::CommandBuffer) {
    let st = vk::PipelineStageFlags::COLOR_ATTACHMENT_OUTPUT | vk::PipelineStageFlags::EARLY_FRAGMENT_TESTS | vk::PipelineStageFlags::LATE_FRAGMENT_TESTS;
    let mb = vk::MemoryBarrier::default()
        .src_access_mask(vk::AccessFlags::COLOR_ATTACHMENT_WRITE | vk::AccessFlags::DEPTH_STENCIL_ATTACHMENT_WRITE)
        .dst_access_mask(vk::AccessFlags::COLOR_ATTACHMENT_READ | vk::AccessFlags::COLOR_ATTACHMENT_WRITE | vk::AccessFlags::DEPTH_STENCIL_ATTACHMENT_READ | vk::AccessFlags::DEPTH_STENCIL_ATTACHMENT_WRITE);
    unsafe { (real().fp_v1_0().cmd_pipeline_barrier)(cb, st, st, vk::DependencyFlags::BY_REGION, 1, &mb, 0, null(), 0, null()) };
}

/// Fusion's end: the first batch's buffer ended after the last batch's
/// commands.
pub fn end_fused(cb: vk::CommandBuffer) {
    unsafe {
        let _ = (real().fp_v1_0().end_command_buffer)(cb);
    }
}

/// `play_list`'s skips.
pub const SKIP_BEGIN: u8 = 1;
pub const SKIP_END: u8 = 2;

/// An arena word + 1 (0: none) as a pointer into the arena at `base`.
unsafe fn rebase<T>(v: usize, base: *const u64) -> *const T {
    if v == 0 { null() } else { unsafe { base.add(v - 1).cast::<T>() } }
}

/// Records `l` into `cb` through the driver (`skip`: SKIP_BEGIN and
/// SKIP_END leave out its begin and end, fusion).
unsafe fn play_list(cb: vk::CommandBuffer, l: &mut List, skip: u8) {
    unsafe { play_list_cut(cb, l, skip, &[]) }
}

/// `play_list` without the ops of `cuts` ([begin, end), an attachment-only
/// barrier in their place when marked so: seamless chunks).
unsafe fn play_list_cut(cb: vk::CommandBuffer, l: &mut List, skip: u8, cuts: &[(u32, u32, bool)]) {
    let d = real();
    let (f0, f1, f2, f3) = (d.fp_v1_0(), d.fp_v1_1(), d.fp_v1_2(), d.fp_v1_3());
    let c = super::ctx();
    let base = l.arena.as_mut_ptr();
    let at = move |s: Span| -> *const u64 { if s.n == 0 { null() } else { unsafe { base.add(s.at as usize).cast_const() } } };
    let one = move |i: u32| -> *const u64 { if i == NONE { null() } else { unsafe { base.add(i as usize).cast_const() } } };
    let profile = record_profile();
    let mut counts = [[0u64; 4]; 10];
    thread_local! { static SAMPLE: Cell<usize> = const { Cell::new(0) }; }
    let phase = if profile { SAMPLE.with(|n| { let v = n.get(); n.set((v+1)&63); v }) } else { 0 };
    for (k, op) in l.ops.iter().enumerate() {
        if !cuts.is_empty() {
            let k = k as u32;
            if let Some(c) = cuts.iter().find(|c| k >= c.0 && k < c.1) {
                if c.2 && k == c.0 {
                    unsafe { light_barrier(cb) };
                }
                continue;
            }
        }
        let class = if profile { record_class(op) } else { 0 };
        let sampled = profile && k & 63 == phase;
        let start = sampled.then(|| (std::time::Instant::now(), crate::platform::thread_times::own_ns()));
        if profile { counts[class][0] += 1; }
        unsafe {
            match *op {
                Op::Begin(flags) => {
                    if skip & SKIP_BEGIN == 0 {
                        let _ = (f0.begin_command_buffer)(cb, &vk::CommandBufferBeginInfo::default().flags(flags));
                    }
                }
                Op::End => {
                    if skip & SKIP_END == 0 {
                        let _ = (f0.end_command_buffer)(cb);
                    }
                }
                Op::BindPipeline(bp, p) => (f0.cmd_bind_pipeline)(cb, bp, p),
                Op::Viewports(first, s) => (f0.cmd_set_viewport)(cb, first, s.n, at(s).cast()),
                Op::Scissors(first, s) => (f0.cmd_set_scissor)(cb, first, s.n, at(s).cast()),
                Op::LineWidth(w) => (f0.cmd_set_line_width)(cb, w),
                Op::DepthBias(a, b, s) => (f0.cmd_set_depth_bias)(cb, a, b, s),
                Op::DepthBounds(a, b) => (f0.cmd_set_depth_bounds)(cb, a, b),
                Op::BlendConstants(k) => (f0.cmd_set_blend_constants)(cb, &k),
                Op::StencilCompareMask(f, v) => (f0.cmd_set_stencil_compare_mask)(cb, f, v),
                Op::StencilWriteMask(f, v) => (f0.cmd_set_stencil_write_mask)(cb, f, v),
                Op::StencilReference(f, v) => (f0.cmd_set_stencil_reference)(cb, f, v),
                Op::BindSets(bp, layout, first, s, o) => (f0.cmd_bind_descriptor_sets)(cb, bp, layout, first, s.n, at(s).cast(), o.n, at(o).cast()),
                Op::BindIndexBuffer(b, o, t) => (f0.cmd_bind_index_buffer)(cb, b, o, t),
                Op::BindVertexBuffers(first, b, o) => (f0.cmd_bind_vertex_buffers)(cb, first, b.n, at(b).cast(), at(o).cast()),
                Op::Draw(a, b, x, y) => (f0.cmd_draw)(cb, a, b, x, y),
                Op::DrawIndexed(a, b, x, y, z) => (f0.cmd_draw_indexed)(cb, a, b, x, y, z),
                Op::DrawIndirect(b, o, n, s) => (f0.cmd_draw_indirect)(cb, b, o, n, s),
                Op::DrawIndexedIndirect(b, o, n, s) => (f0.cmd_draw_indexed_indirect)(cb, b, o, n, s),
                Op::DrawIndirectCount(b, o, k, ko, m, s) => (f2.cmd_draw_indirect_count)(cb, b, o, k, ko, m, s),
                Op::DrawIndexedIndirectCount(b, o, k, ko, m, s) => (f2.cmd_draw_indexed_indirect_count)(cb, b, o, k, ko, m, s),
                Op::Dispatch(x, y, z) => (f0.cmd_dispatch)(cb, x, y, z),
                Op::DispatchBase(v) => (f1.cmd_dispatch_base)(cb, v[0], v[1], v[2], v[3], v[4], v[5]),
                Op::DispatchIndirect(b, o) => (f0.cmd_dispatch_indirect)(cb, b, o),
                Op::CopyBuffer(s, d, r) => (f0.cmd_copy_buffer)(cb, s, d, r.n, at(r).cast()),
                Op::CopyImage(s, sl, d, dl, r) => (f0.cmd_copy_image)(cb, s, sl, d, dl, r.n, at(r).cast()),
                Op::BlitImage(s, sl, d, dl, r, f) => (f0.cmd_blit_image)(cb, s, sl, d, dl, r.n, at(r).cast(), f),
                Op::CopyBufferToImage(s, d, dl, r) => (f0.cmd_copy_buffer_to_image)(cb, s, d, dl, r.n, at(r).cast()),
                Op::CopyImageToBuffer(s, sl, d, r) => (f0.cmd_copy_image_to_buffer)(cb, s, sl, d, r.n, at(r).cast()),
                Op::UpdateBuffer(b, o, s) => (f0.cmd_update_buffer)(cb, b, o, s.n as vk::DeviceSize, at(s).cast()),
                Op::FillBuffer(b, o, n, v) => (f0.cmd_fill_buffer)(cb, b, o, n, v),
                Op::ClearColorImage(i, lay, v, r) => (f0.cmd_clear_color_image)(cb, i, lay, &v, r.n, at(r).cast()),
                Op::ClearDepthStencilImage(i, lay, v, r) => (f0.cmd_clear_depth_stencil_image)(cb, i, lay, &v, r.n, at(r).cast()),
                Op::ClearAttachments(a, r) => (f0.cmd_clear_attachments)(cb, a.n, at(a).cast(), r.n, at(r).cast()),
                Op::Barrier(src, dst, dep, m, b, i) => (f0.cmd_pipeline_barrier)(cb, src, dst, dep, m.n, at(m).cast(), b.n, at(b).cast(), i.n, at(i).cast()),
                Op::BeginQuery(p, q, f) => (f0.cmd_begin_query)(cb, p, q, f),
                Op::EndQuery(p, q) => (f0.cmd_end_query)(cb, p, q),
                Op::ResetQueryPool(p, a, n) => (f0.cmd_reset_query_pool)(cb, p, a, n),
                Op::WriteTimestamp(s, p, q) => (f0.cmd_write_timestamp)(cb, s, p, q),
                Op::CopyQueryPoolResults(p, a, n, b, o, s, f) => (f0.cmd_copy_query_pool_results)(cb, p, a, n, b, o, s, f),
                Op::PushConstants(layout, st, o, s) => (f0.cmd_push_constants)(cb, layout, st, o, s.n, at(s).cast()),
                Op::BeginRendering(flags, area, layers, view_mask, colors, depth, stencil) => {
                    let mut ri = vk::RenderingInfo::default();
                    ri.flags = flags;
                    ri.render_area = area;
                    ri.layer_count = layers;
                    ri.view_mask = view_mask;
                    ri.color_attachment_count = colors.n;
                    ri.p_color_attachments = at(colors).cast();
                    ri.p_depth_attachment = one(depth).cast();
                    ri.p_stencil_attachment = one(stencil).cast();
                    (f3.cmd_begin_rendering)(cb, &ri);
                }
                Op::EndRendering => (f3.cmd_end_rendering)(cb),
                Op::CullMode(m) => (f3.cmd_set_cull_mode)(cb, m),
                Op::FrontFace(f) => (f3.cmd_set_front_face)(cb, f),
                Op::Topology(t) => (f3.cmd_set_primitive_topology)(cb, t),
                Op::DepthTestEnable(e) => (f3.cmd_set_depth_test_enable)(cb, e),
                Op::DepthWriteEnable(e) => (f3.cmd_set_depth_write_enable)(cb, e),
                Op::DepthCompareOp(o) => (f3.cmd_set_depth_compare_op)(cb, o),
                Op::DepthBoundsTestEnable(e) => (f3.cmd_set_depth_bounds_test_enable)(cb, e),
                Op::StencilTestEnable(e) => (f3.cmd_set_stencil_test_enable)(cb, e),
                Op::StencilOp(f, a, b, x, o) => (f3.cmd_set_stencil_op)(cb, f, a, b, x, o),
                Op::RasterizerDiscardEnable(e) => (f3.cmd_set_rasterizer_discard_enable)(cb, e),
                Op::DepthBiasEnable(e) => (f3.cmd_set_depth_bias_enable)(cb, e),
                Op::PrimitiveRestartEnable(e) => (f3.cmd_set_primitive_restart_enable)(cb, e),
                Op::PushDescriptors(bp, layout, set, w) => {
                    let writes = at(w).cast::<vk::WriteDescriptorSet<'static>>().cast_mut();
                    for i in 0..w.n as usize {
                        let x = &mut *writes.add(i);
                        x.p_image_info = rebase(x.p_image_info as usize, base);
                        x.p_buffer_info = rebase(x.p_buffer_info as usize, base);
                        x.p_texel_buffer_view = rebase(x.p_texel_buffer_view as usize, base);
                    }
                    (c.push_desc.fp().cmd_push_descriptor_set_khr)(cb, bp, layout, set, w.n, writes);
                }
                Op::LabelBegin(s) => {
                    if let Some(lb) = &c.labels {
                        lb.cmd_begin_debug_utils_label(cb, &vk::DebugUtilsLabelEXT::default().label_name(CStr::from_ptr(at(s).cast::<c_char>())));
                    }
                }
                Op::LabelEnd => {
                    if let Some(lb) = &c.labels {
                        lb.cmd_end_debug_utils_label(cb);
                    }
                }
                Op::Checkpoint(m) => {
                    if let Some(cp) = &c.checkpoints {
                        cp.cmd_set_checkpoint(cb, m as *const c_void);
                    }
                }
                Op::External(i) => {
                    if let Some(f) = l.externals.get_mut(i as usize).and_then(Option::take) {
                        f(cb);
                    }
                }
            }
        }
        if let Some((time, cpu)) = start {
            let end = crate::platform::thread_times::own_ns();
            counts[class][1] += 1;
            counts[class][2] += time.elapsed().as_nanos() as u64;
            counts[class][3] += end.saturating_sub(cpu);
        }
    }
    if profile {
        for (i, values) in counts.iter().enumerate() { for (j, &v) in values.iter().enumerate() { RECORD_STATS[i][j].fetch_add(v, Ordering::Relaxed); } }
    }
}

/// Host code that records into `cb` itself (a library given the command buffer): kept in
/// `cb`'s list when this thread claimed it, run at replay in the commands' order; else run now.
pub fn external(cb: vk::CommandBuffer, f: ExternalFn) {
    let mut f = Some(f);
    let kept = record(cb, |l| {
        let i = l.externals.len() as u32;
        l.externals.push(f.take());
        l.ops.push(Op::External(i));
    });
    if !kept {
        if let Some(f) = f.take() {
            f(cb);
        }
    }
}

/// vkCmdPushDescriptorSetKHR into `cb`'s list if this thread claimed it
/// (`Ctx::push_descriptors`).
pub fn push_descriptors(cb: vk::CommandBuffer, bp: vk::PipelineBindPoint, layout: vk::PipelineLayout, set: u32, writes: &[vk::WriteDescriptorSet<'_>]) -> bool {
    use vk::DescriptorType as T;
    if claimed(cb).is_none() {
        return false;
    }
    let kind = |t: T| match t {
        T::SAMPLER | T::COMBINED_IMAGE_SAMPLER | T::SAMPLED_IMAGE | T::STORAGE_IMAGE | T::INPUT_ATTACHMENT => 1,
        T::UNIFORM_BUFFER | T::STORAGE_BUFFER | T::UNIFORM_BUFFER_DYNAMIC | T::STORAGE_BUFFER_DYNAMIC => 2,
        T::UNIFORM_TEXEL_BUFFER | T::STORAGE_TEXEL_BUFFER => 3,
        _ => 0,
    };
    if writes.iter().any(|w| !w.p_next.is_null() || kind(w.descriptor_type) == 0) {
        unheld("push descriptors of a kind the lists do not hold");
        return false;
    }
    record(cb, |l| {
        let w = l.put(writes.as_ptr(), writes.len() as u32);
        for (i, x) in writes.iter().enumerate() {
            let n = x.descriptor_count;
            let k = kind(x.descriptor_type);
            let img = if k == 1 { l.put_ref(x.p_image_info, n) } else { 0 };
            let buf = if k == 2 { l.put_ref(x.p_buffer_info, n) } else { 0 };
            let tex = if k == 3 { l.put_ref(x.p_texel_buffer_view, n) } else { 0 };
            let y = &mut l.items::<vk::WriteDescriptorSet<'static>>(w)[i];
            y.p_image_info = img as *const _;
            y.p_buffer_info = buf as *const _;
            y.p_texel_buffer_view = tex as *const _;
        }
        l.ops.push(Op::PushDescriptors(bp, layout, set, w));
    })
}

/// A label range's begin into `cb`'s list if this thread claimed it.
pub fn label_begin(cb: vk::CommandBuffer, name: &str) -> bool {
    record(cb, |l| {
        let s = l.put_str(name);
        l.ops.push(Op::LabelBegin(s));
    })
}

pub fn label_end(cb: vk::CommandBuffer) -> bool {
    record(cb, |l| l.ops.push(Op::LabelEnd))
}

pub fn checkpoint(cb: vk::CommandBuffer, marker: u64) -> bool {
    record(cb, |l| l.ops.push(Op::Checkpoint(marker)))
}

/// A command whose arguments are values: `$op` into the list, else to the
/// driver's `$fp().$f`.
macro_rules! shim {
    ($name:ident, $fp:ident . $f:ident, ($($a:ident: $t:ty),*), $op:expr) => {
        unsafe extern "system" fn $name(cb: vk::CommandBuffer, $($a: $t),*) {
            if !record(cb, |l| l.ops.push($op)) {
                direct(cb);
                unsafe { (real().$fp().$f)(cb, $($a),*) }
            }
        }
    };
}

shim!(bind_pipeline, fp_v1_0.cmd_bind_pipeline, (bp: vk::PipelineBindPoint, p: vk::Pipeline), Op::BindPipeline(bp, p));
shim!(set_line_width, fp_v1_0.cmd_set_line_width, (w: f32), Op::LineWidth(w));
shim!(set_depth_bias, fp_v1_0.cmd_set_depth_bias, (a: f32, b: f32, s: f32), Op::DepthBias(a, b, s));
shim!(set_depth_bounds, fp_v1_0.cmd_set_depth_bounds, (a: f32, b: f32), Op::DepthBounds(a, b));
shim!(set_stencil_compare_mask, fp_v1_0.cmd_set_stencil_compare_mask, (f: vk::StencilFaceFlags, v: u32), Op::StencilCompareMask(f, v));
shim!(set_stencil_write_mask, fp_v1_0.cmd_set_stencil_write_mask, (f: vk::StencilFaceFlags, v: u32), Op::StencilWriteMask(f, v));
shim!(set_stencil_reference, fp_v1_0.cmd_set_stencil_reference, (f: vk::StencilFaceFlags, v: u32), Op::StencilReference(f, v));
shim!(bind_index_buffer, fp_v1_0.cmd_bind_index_buffer, (b: vk::Buffer, o: vk::DeviceSize, t: vk::IndexType), Op::BindIndexBuffer(b, o, t));
shim!(draw, fp_v1_0.cmd_draw, (a: u32, b: u32, x: u32, y: u32), Op::Draw(a, b, x, y));
shim!(draw_indexed, fp_v1_0.cmd_draw_indexed, (a: u32, b: u32, x: u32, y: i32, z: u32), Op::DrawIndexed(a, b, x, y, z));
shim!(draw_indirect, fp_v1_0.cmd_draw_indirect, (b: vk::Buffer, o: vk::DeviceSize, n: u32, s: u32), Op::DrawIndirect(b, o, n, s));
shim!(draw_indexed_indirect, fp_v1_0.cmd_draw_indexed_indirect, (b: vk::Buffer, o: vk::DeviceSize, n: u32, s: u32), Op::DrawIndexedIndirect(b, o, n, s));
shim!(dispatch, fp_v1_0.cmd_dispatch, (x: u32, y: u32, z: u32), Op::Dispatch(x, y, z));
shim!(dispatch_indirect, fp_v1_0.cmd_dispatch_indirect, (b: vk::Buffer, o: vk::DeviceSize), Op::DispatchIndirect(b, o));
shim!(fill_buffer, fp_v1_0.cmd_fill_buffer, (b: vk::Buffer, o: vk::DeviceSize, n: vk::DeviceSize, v: u32), Op::FillBuffer(b, o, n, v));
shim!(begin_query, fp_v1_0.cmd_begin_query, (p: vk::QueryPool, q: u32, f: vk::QueryControlFlags), Op::BeginQuery(p, q, f));
shim!(end_query, fp_v1_0.cmd_end_query, (p: vk::QueryPool, q: u32), Op::EndQuery(p, q));
shim!(reset_query_pool, fp_v1_0.cmd_reset_query_pool, (p: vk::QueryPool, a: u32, n: u32), Op::ResetQueryPool(p, a, n));
shim!(write_timestamp, fp_v1_0.cmd_write_timestamp, (s: vk::PipelineStageFlags, p: vk::QueryPool, q: u32), Op::WriteTimestamp(s, p, q));
shim!(
    copy_query_pool_results,
    fp_v1_0.cmd_copy_query_pool_results,
    (p: vk::QueryPool, a: u32, n: u32, b: vk::Buffer, o: vk::DeviceSize, s: vk::DeviceSize, f: vk::QueryResultFlags),
    Op::CopyQueryPoolResults(p, a, n, b, o, s, f)
);
shim!(dispatch_base, fp_v1_1.cmd_dispatch_base, (a: u32, b: u32, x: u32, y: u32, z: u32, w: u32), Op::DispatchBase([a, b, x, y, z, w]));
shim!(
    draw_indirect_count,
    fp_v1_2.cmd_draw_indirect_count,
    (b: vk::Buffer, o: vk::DeviceSize, k: vk::Buffer, ko: vk::DeviceSize, m: u32, s: u32),
    Op::DrawIndirectCount(b, o, k, ko, m, s)
);
shim!(
    draw_indexed_indirect_count,
    fp_v1_2.cmd_draw_indexed_indirect_count,
    (b: vk::Buffer, o: vk::DeviceSize, k: vk::Buffer, ko: vk::DeviceSize, m: u32, s: u32),
    Op::DrawIndexedIndirectCount(b, o, k, ko, m, s)
);
shim!(end_rendering, fp_v1_3.cmd_end_rendering, (), Op::EndRendering);
shim!(set_cull_mode, fp_v1_3.cmd_set_cull_mode, (m: vk::CullModeFlags), Op::CullMode(m));
shim!(set_front_face, fp_v1_3.cmd_set_front_face, (f: vk::FrontFace), Op::FrontFace(f));
shim!(set_primitive_topology, fp_v1_3.cmd_set_primitive_topology, (t: vk::PrimitiveTopology), Op::Topology(t));
shim!(set_depth_test_enable, fp_v1_3.cmd_set_depth_test_enable, (e: vk::Bool32), Op::DepthTestEnable(e));
shim!(set_depth_write_enable, fp_v1_3.cmd_set_depth_write_enable, (e: vk::Bool32), Op::DepthWriteEnable(e));
shim!(set_depth_compare_op, fp_v1_3.cmd_set_depth_compare_op, (o: vk::CompareOp), Op::DepthCompareOp(o));
shim!(set_depth_bounds_test_enable, fp_v1_3.cmd_set_depth_bounds_test_enable, (e: vk::Bool32), Op::DepthBoundsTestEnable(e));
shim!(set_stencil_test_enable, fp_v1_3.cmd_set_stencil_test_enable, (e: vk::Bool32), Op::StencilTestEnable(e));
shim!(
    set_stencil_op,
    fp_v1_3.cmd_set_stencil_op,
    (f: vk::StencilFaceFlags, a: vk::StencilOp, b: vk::StencilOp, x: vk::StencilOp, o: vk::CompareOp),
    Op::StencilOp(f, a, b, x, o)
);
shim!(set_rasterizer_discard_enable, fp_v1_3.cmd_set_rasterizer_discard_enable, (e: vk::Bool32), Op::RasterizerDiscardEnable(e));
shim!(set_depth_bias_enable, fp_v1_3.cmd_set_depth_bias_enable, (e: vk::Bool32), Op::DepthBiasEnable(e));
shim!(set_primitive_restart_enable, fp_v1_3.cmd_set_primitive_restart_enable, (e: vk::Bool32), Op::PrimitiveRestartEnable(e));

unsafe extern "system" fn begin_command_buffer(cb: vk::CommandBuffer, info: *const vk::CommandBufferBeginInfo<'_>) -> vk::Result {
    let flags = unsafe { (*info).flags };
    if record(cb, |l| l.ops.push(Op::Begin(flags))) {
        return vk::Result::SUCCESS;
    }
    direct(cb);
    unsafe { (real().fp_v1_0().begin_command_buffer)(cb, info) }
}

unsafe extern "system" fn end_command_buffer(cb: vk::CommandBuffer) -> vk::Result {
    if record(cb, |l| l.ops.push(Op::End)) {
        return vk::Result::SUCCESS;
    }
    direct(cb);
    unsafe { (real().fp_v1_0().end_command_buffer)(cb) }
}

unsafe extern "system" fn set_viewport(cb: vk::CommandBuffer, first: u32, n: u32, p: *const vk::Viewport) {
    if !record(cb, |l| {
        let s = l.put(p, n);
        l.ops.push(Op::Viewports(first, s));
    }) {
        direct(cb);
        unsafe { (real().fp_v1_0().cmd_set_viewport)(cb, first, n, p) }
    }
}

unsafe extern "system" fn set_scissor(cb: vk::CommandBuffer, first: u32, n: u32, p: *const vk::Rect2D) {
    if !record(cb, |l| {
        let s = l.put(p, n);
        l.ops.push(Op::Scissors(first, s));
    }) {
        direct(cb);
        unsafe { (real().fp_v1_0().cmd_set_scissor)(cb, first, n, p) }
    }
}

unsafe extern "system" fn set_blend_constants(cb: vk::CommandBuffer, k: *const [f32; 4]) {
    let v = unsafe { *k };
    if !record(cb, |l| l.ops.push(Op::BlendConstants(v))) {
        direct(cb);
        unsafe { (real().fp_v1_0().cmd_set_blend_constants)(cb, k) }
    }
}

#[allow(clippy::too_many_arguments)]
unsafe extern "system" fn bind_descriptor_sets(cb: vk::CommandBuffer, bp: vk::PipelineBindPoint, layout: vk::PipelineLayout, first: u32, n: u32, sets: *const vk::DescriptorSet, dn: u32, offsets: *const u32) {
    if !record(cb, |l| {
        let s = l.put(sets, n);
        let o = l.put(offsets, dn);
        l.ops.push(Op::BindSets(bp, layout, first, s, o));
    }) {
        direct(cb);
        unsafe { (real().fp_v1_0().cmd_bind_descriptor_sets)(cb, bp, layout, first, n, sets, dn, offsets) }
    }
}

unsafe extern "system" fn bind_vertex_buffers(cb: vk::CommandBuffer, first: u32, n: u32, bufs: *const vk::Buffer, offsets: *const vk::DeviceSize) {
    if !record(cb, |l| {
        let b = l.put(bufs, n);
        let o = l.put(offsets, n);
        l.ops.push(Op::BindVertexBuffers(first, b, o));
    }) {
        direct(cb);
        unsafe { (real().fp_v1_0().cmd_bind_vertex_buffers)(cb, first, n, bufs, offsets) }
    }
}

unsafe extern "system" fn copy_buffer(cb: vk::CommandBuffer, src: vk::Buffer, dst: vk::Buffer, n: u32, p: *const vk::BufferCopy) {
    if !record(cb, |l| {
        let r = l.put(p, n);
        l.ops.push(Op::CopyBuffer(src, dst, r));
    }) {
        direct(cb);
        unsafe { (real().fp_v1_0().cmd_copy_buffer)(cb, src, dst, n, p) }
    }
}

unsafe extern "system" fn copy_image(cb: vk::CommandBuffer, src: vk::Image, sl: vk::ImageLayout, dst: vk::Image, dl: vk::ImageLayout, n: u32, p: *const vk::ImageCopy) {
    if !record(cb, |l| {
        let r = l.put(p, n);
        l.ops.push(Op::CopyImage(src, sl, dst, dl, r));
    }) {
        direct(cb);
        unsafe { (real().fp_v1_0().cmd_copy_image)(cb, src, sl, dst, dl, n, p) }
    }
}

#[allow(clippy::too_many_arguments)]
unsafe extern "system" fn blit_image(cb: vk::CommandBuffer, src: vk::Image, sl: vk::ImageLayout, dst: vk::Image, dl: vk::ImageLayout, n: u32, p: *const vk::ImageBlit, f: vk::Filter) {
    if !record(cb, |l| {
        let r = l.put(p, n);
        l.ops.push(Op::BlitImage(src, sl, dst, dl, r, f));
    }) {
        direct(cb);
        unsafe { (real().fp_v1_0().cmd_blit_image)(cb, src, sl, dst, dl, n, p, f) }
    }
}

unsafe extern "system" fn copy_buffer_to_image(cb: vk::CommandBuffer, src: vk::Buffer, dst: vk::Image, dl: vk::ImageLayout, n: u32, p: *const vk::BufferImageCopy) {
    if !record(cb, |l| {
        let r = l.put(p, n);
        l.ops.push(Op::CopyBufferToImage(src, dst, dl, r));
    }) {
        direct(cb);
        unsafe { (real().fp_v1_0().cmd_copy_buffer_to_image)(cb, src, dst, dl, n, p) }
    }
}

unsafe extern "system" fn copy_image_to_buffer(cb: vk::CommandBuffer, src: vk::Image, sl: vk::ImageLayout, dst: vk::Buffer, n: u32, p: *const vk::BufferImageCopy) {
    if !record(cb, |l| {
        let r = l.put(p, n);
        l.ops.push(Op::CopyImageToBuffer(src, sl, dst, r));
    }) {
        direct(cb);
        unsafe { (real().fp_v1_0().cmd_copy_image_to_buffer)(cb, src, sl, dst, n, p) }
    }
}

unsafe extern "system" fn update_buffer(cb: vk::CommandBuffer, dst: vk::Buffer, off: vk::DeviceSize, size: vk::DeviceSize, data: *const c_void) {
    if !record(cb, |l| {
        let s = l.put(data.cast::<u8>(), size as u32);
        l.ops.push(Op::UpdateBuffer(dst, off, s));
    }) {
        direct(cb);
        unsafe { (real().fp_v1_0().cmd_update_buffer)(cb, dst, off, size, data) }
    }
}

unsafe extern "system" fn clear_color_image(cb: vk::CommandBuffer, img: vk::Image, layout: vk::ImageLayout, color: *const vk::ClearColorValue, n: u32, ranges: *const vk::ImageSubresourceRange) {
    let v = unsafe { *color };
    if !record(cb, |l| {
        let r = l.put(ranges, n);
        l.ops.push(Op::ClearColorImage(img, layout, v, r));
    }) {
        direct(cb);
        unsafe { (real().fp_v1_0().cmd_clear_color_image)(cb, img, layout, color, n, ranges) }
    }
}

unsafe extern "system" fn clear_depth_stencil_image(cb: vk::CommandBuffer, img: vk::Image, layout: vk::ImageLayout, value: *const vk::ClearDepthStencilValue, n: u32, ranges: *const vk::ImageSubresourceRange) {
    let v = unsafe { *value };
    if !record(cb, |l| {
        let r = l.put(ranges, n);
        l.ops.push(Op::ClearDepthStencilImage(img, layout, v, r));
    }) {
        direct(cb);
        unsafe { (real().fp_v1_0().cmd_clear_depth_stencil_image)(cb, img, layout, value, n, ranges) }
    }
}

unsafe extern "system" fn clear_attachments(cb: vk::CommandBuffer, n: u32, att: *const vk::ClearAttachment, rn: u32, rects: *const vk::ClearRect) {
    if !record(cb, |l| {
        let a = l.put(att, n);
        let r = l.put(rects, rn);
        l.ops.push(Op::ClearAttachments(a, r));
    }) {
        direct(cb);
        unsafe { (real().fp_v1_0().cmd_clear_attachments)(cb, n, att, rn, rects) }
    }
}

#[allow(clippy::too_many_arguments)]
/// Barriers merged into the barrier before them (`pipeline_barrier`).
pub static MERGED_BARRIERS: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);

unsafe extern "system" fn pipeline_barrier(
    cb: vk::CommandBuffer,
    src: vk::PipelineStageFlags,
    dst: vk::PipelineStageFlags,
    dep: vk::DependencyFlags,
    mn: u32,
    mb: *const vk::MemoryBarrier<'_>,
    bn: u32,
    bb: *const vk::BufferMemoryBarrier<'_>,
    inn: u32,
    ib: *const vk::ImageMemoryBarrier<'_>,
) {
    if !record(cb, |l| {
        // Right after another barrier (nothing recorded between), both of
        // global memory barriers only and the same dependency flags: one
        // barrier with both scopes, as strong as the two (+0.3-1.2% at 1-1's
        // walk places, FINDINGS.md 2026-11-04).
        if bn == 0 && inn == 0 && mn <= 1 {
            if let Some(&Op::Barrier(s0, d0, dep0, m0, b0, i0)) = l.ops.last() {
                if b0.n == 0 && i0.n == 0 && m0.n <= 1 && dep0 == dep {
                    let (mut sa, mut da) = (vk::AccessFlags::empty(), vk::AccessFlags::empty());
                    if m0.n == 1 {
                        let x = &l.items::<vk::MemoryBarrier<'static>>(m0)[0];
                        (sa, da) = (x.src_access_mask, x.dst_access_mask);
                    }
                    if mn == 1 {
                        let x = unsafe { &*mb };
                        (sa, da) = (sa | x.src_access_mask, da | x.dst_access_mask);
                    }
                    let one = vk::MemoryBarrier::default().src_access_mask(sa).dst_access_mask(da);
                    let m = if m0.n == 1 {
                        l.items::<vk::MemoryBarrier<'static>>(m0)[0] = one;
                        m0
                    } else {
                        l.put(&one, 1)
                    };
                    if let Some(last) = l.ops.last_mut() {
                        *last = Op::Barrier(s0 | src, d0 | dst, dep, m, b0, i0);
                    }
                    MERGED_BARRIERS.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
                    return;
                }
            }
        }
        let m = l.put(mb, mn);
        let b = l.put(bb, bn);
        let i = l.put(ib, inn);
        // (No pNext chains here: the port records none.)
        for x in l.items::<vk::MemoryBarrier<'static>>(m) {
            x.p_next = null();
        }
        for x in l.items::<vk::BufferMemoryBarrier<'static>>(b) {
            x.p_next = null();
        }
        for x in l.items::<vk::ImageMemoryBarrier<'static>>(i) {
            x.p_next = null();
        }
        l.ops.push(Op::Barrier(src, dst, dep, m, b, i));
    }) {
        direct(cb);
        unsafe { (real().fp_v1_0().cmd_pipeline_barrier)(cb, src, dst, dep, mn, mb, bn, bb, inn, ib) }
    }
}

unsafe extern "system" fn push_constants(cb: vk::CommandBuffer, layout: vk::PipelineLayout, stages: vk::ShaderStageFlags, off: u32, size: u32, p: *const c_void) {
    if !record(cb, |l| {
        let s = l.put(p.cast::<u8>(), size);
        l.ops.push(Op::PushConstants(layout, stages, off, s));
    }) {
        direct(cb);
        unsafe { (real().fp_v1_0().cmd_push_constants)(cb, layout, stages, off, size, p) }
    }
}

unsafe extern "system" fn begin_rendering(cb: vk::CommandBuffer, ri: *const vk::RenderingInfo<'_>) {
    let r = unsafe { &*ri };
    if claimed(cb).is_some() && !r.p_next.is_null() {
        unheld("a rendering pass with a pNext chain");
    }
    if !record(cb, |l| {
        let colors = l.put(r.p_color_attachments, r.color_attachment_count);
        let depth = if r.p_depth_attachment.is_null() { NONE } else { l.put(r.p_depth_attachment, 1).at };
        let stencil = if r.p_stencil_attachment.is_null() {
            NONE
        } else if r.p_stencil_attachment == r.p_depth_attachment {
            depth
        } else {
            l.put(r.p_stencil_attachment, 1).at
        };
        for x in l.items::<vk::RenderingAttachmentInfo<'static>>(colors) {
            x.p_next = null();
        }
        for at in [depth, stencil].into_iter().filter(|&a| a != NONE) {
            l.items::<vk::RenderingAttachmentInfo<'static>>(Span { at, n: 1 })[0].p_next = null();
        }
        l.ops.push(Op::BeginRendering(r.flags, r.render_area, r.layer_count, r.view_mask, colors, depth, stencil));
    }) {
        direct(cb);
        unsafe { (real().fp_v1_3().cmd_begin_rendering)(cb, ri) }
    }
}

/// The status line (`exec::status`): empty until a batch was deferred.
pub fn status() -> String {
    let s = |i: usize| STATS[i].load(Ordering::Relaxed);
    let b = s(S_BATCHES);
    if b == 0 && s(S_FLUSHES) == 0 && s(S_DISCARDED) == 0 && s(S_FOREIGN) == 0 {
        return String::new();
    }
    let ops = s(S_OPS);
    format!(
        "    deferred recording: {b} batches replayed by the submission thread ({ops} commands, {:.0} per batch, at most {}; {:.1} MiB of arrays) in {:.1} ms ({:.1} us per batch, {:.0} ns per command); {} discarded, {} replayed in place ({} commands of forms the lists do not hold), {} claims over unreleased ones, {} commands of other threads for claimed buffers, {} barriers merged into the one before{}\n",
        ops as f64 / b.max(1) as f64,
        s(S_MAX_OPS),
        s(S_WORDS) as f64 * 8.0 / (1 << 20) as f64,
        s(S_NS) as f64 / 1e6,
        s(S_NS) as f64 / 1e3 / b.max(1) as f64,
        s(S_NS) as f64 / ops.max(1) as f64,
        s(S_DISCARDED),
        s(S_FLUSHES),
        s(S_UNHELD),
        s(S_STALE),
        s(S_FOREIGN),
        MERGED_BARRIERS.load(Ordering::Relaxed),
        if CHECK.load(Ordering::Relaxed) { " (checked)" } else { "" }
    ) + &record_report()
}
