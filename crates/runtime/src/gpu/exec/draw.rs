//! Draws (docs/RENDERER.md milestone 3): the NGG primitive shader runs as a
//! Vulkan vertex shader, the pixel shader as the fragment shader, rect lists
//! are expanded by a generated geometry shader. Color and depth targets come
//! from the CB/DB registers and live in the image cache (shared with textures
//! by guest address); draws record into the queue's command buffer inside
//! dynamic-rendering instances that last while the attachments stay the same.

use super::*;
use crate::gpu::images::{self, Surface, Target};
use std::hash::{Hash, Hasher};
use std::sync::atomic::AtomicU32;

pub static DRAWN: AtomicU64 = AtomicU64::new(0);

/// `DS_RUN_CENSUS`: the run a GPU-resolved pair's draws on this thread would be appended to:
/// its pair, run roots' values, run user words and per-item buffers' formats as of its first
/// draw (`Exec::run_census_note`).
struct RcRun {
    pair: (u64, u64),
    roots: Vec<u64>,
    user: Vec<u32>,
    formats: Vec<(u8, u16)>,
}

thread_local! {
    static RC_RUN: std::cell::RefCell<Option<RcRun>> = const { std::cell::RefCell::new(None) };
}

/// Draws without visible effect here (metadata passes, all writes disabled).
pub static DRAW_NOOP: AtomicU64 = AtomicU64::new(0);
pub static DRAW_SKIPPED: AtomicU64 = AtomicU64::new(0);
/// Indexed indirect draws recorded with the GPU reading their arguments.
pub static GPU_ARG_DRAWS: AtomicU64 = AtomicU64::new(0);
/// Of them, those recorded as a repeat of the draw before (`draw_repeat`).
pub static REPEAT_DRAWS: AtomicU64 = AtomicU64::new(0);
/// Buffer-delta draws (`tune buffer_delta`): [candidates (right after a
/// full draw kept for them, with its programs and context state), taken,
/// refused: sync or diagnosis modes; other translations or index use;
/// other images or samplers; image lookups changed; a fast clear or reload
/// pending for its images (a held executor's own included); textures not
/// known unwritten; another rendering instance].
pub static DELTA: [AtomicU64; 9] = [const { AtomicU64::new(0) }; 9];
/// `tune buffer_delta 2`: [draws verified, differences from the full path:
/// a texture wait, targets or pipeline, an image hazard, the rendering
/// instance ended, dynamic state recorded, other bound images or samplers].
pub static DELTA_VERIFY: [AtomicU64; 7] = [const { AtomicU64::new(0) }; 7];
/// Buffer-delta draws' stages: [binding the last draw's images and samplers,
/// looking them up (their memos' generations or the frame moved, or the
/// last draw's came from the image cache)].
pub static DELTA_BIND: [AtomicU64; 2] = [const { AtomicU64::new(0) }; 2];
/// DS_STEP_LAPS: draws step time in all by kind (DRAW_KINDS), other / buffer-delta:
/// [ns, draws] (a batch first draws apart).
static DELTA_COST: [[[AtomicU64; 2]; 2]; 3] = [const { [const { [const { AtomicU64::new(0) }; 2] }; 2] }; 3];
/// DS_STEP_LAPS: a draw's steps (`DRAW_STEP_NAMES`, ns; slot 18 the draws) by class (a batch's
/// first draws apart): 4 for the graphics CP's workers + 2 for a draw without colour targets
/// + 1 for a buffer-delta draw.
static STEPS_SPLIT: [[AtomicU64; 19]; 8] = [const { [const { AtomicU64::new(0) }; 19] }; 8];
/// Pixel stages that read no memory, their preparation taken from the draw before
/// (`Exec::empty_ps`): [taken (counted with `DS_STEP_LAPS` only), compared with a fresh
/// preparation (`tune buffer_delta` 2), differing].
pub static EMPTY_PS: [AtomicU64; 3] = [const { AtomicU64::new(0) }; 3];

/// Whether a prepared stage depends on no guest memory: nothing read, no resources.
fn reads_nothing(p: &Prepared) -> bool {
    p.found.reads.is_empty() && p.found.resources.is_empty() && p.found.unread.is_empty() && p.found.data_loads.is_empty() && p.cmp_images == 0
}

/// Whether two preparations of a stage are the same (`tune buffer_delta` 2).
fn same_prepared(a: &Prepared, b: &Prepared) -> bool {
    a.addr == b.addr && std::sync::Arc::ptr_eq(&a.found.tr, &b.found.tr) && a.sgpr_data == b.sgpr_data && a.user == b.user && a.code_hash == b.code_hash && reads_nothing(b)
}

/// Status part: `STEPS_SPLIT`.
pub(super) fn steps_split_line() -> String {
    use std::sync::atomic::Ordering::Relaxed;
    let mut out = String::new();
    for (c, steps) in STEPS_SPLIT.iter().enumerate() {
        let n = steps[18].load(Relaxed);
        if n == 0 {
            continue;
        }
        let all: u64 = steps[..18].iter().map(|s| s.load(Relaxed)).sum();
        let parts: Vec<String> = super::DRAW_STEP_NAMES.iter().zip(steps.iter()).map(|(name, s)| format!("{name} {:.2}", s.load(Relaxed) as f64 / n as f64 / 1e3)).collect();
        out.push_str(&format!(
            "    a draw by step ({}, {}, {}: {n}, {:.2} us each): {}\n",
            if c & 4 != 0 { "workers" } else { "the graphics CP" },
            if c & 2 != 0 { "depth only" } else { "colour targets" },
            if c & 1 != 0 { "buffer-delta" } else { "full" },
            all as f64 / n as f64 / 1e3,
            parts.join(", ")
        ));
    }
    out
}
/// DS_STEP_LAPS: draws that were not buffer-delta draws (a batch first draws apart): [all, the
/// pair of the draw before (refused as a delta), of the one before that, of one of the two
/// before those, of one of four more, none (the first draw since a dispatch or of its
/// instance), none of the last eight pairs].
static PAIR_CENSUS: [AtomicU64; 7] = [const { AtomicU64::new(0) }; 7];

/// What a full draw's targets, pipeline and hazard check gave, that a
/// buffer-delta draw right after it takes over (`DrawDelta`).
#[derive(Clone, Copy)]
struct Fixed {
    colors: [Option<Target>; MAX_RT],
    depth: Option<Target>,
    area: (u32, u32),
    scissor: (i32, i32, i32, i32),
    z_test: bool,
    z_write: bool,
    s_test: bool,
    s_write: bool,
    clear_z: bool,
    clear_s: bool,
    dctl: u32,
    sctl: u32,
    refm: u32,
    refm_bf: u32,
    pipe: Pipe,
    /// Its stages write memory.
    writes: bool,
    /// Its image hazards were pending.
    hazard: bool,
    /// `images::LOOKUP_GEN` before its targets were looked up.
    lookup_gen: u64,
}

impl Fixed {
    /// Whether the full path's results are these (`tune buffer_delta 2`).
    fn same_as(&self, o: &Fixed) -> bool {
        let t = |x: &Option<Target>| x.map(|t| (t.image, t.view, t.format, t.width, t.height));
        self.colors.iter().map(t).eq(o.colors.iter().map(t))
            && t(&self.depth) == t(&o.depth)
            && (self.area, self.scissor) == (o.area, o.scissor)
            && [self.z_test, self.z_write, self.s_test, self.s_write, self.clear_z, self.clear_s, self.writes] == [o.z_test, o.z_write, o.s_test, o.s_write, o.clear_z, o.clear_s, o.writes]
            && [self.dctl, self.sctl, self.refm, self.refm_bf] == [o.dctl, o.sctl, o.refm, o.refm_bf]
            && (self.pipe.pipeline, self.pipe.layout, self.pipe.set_layout, self.pipe.push) == (o.pipe.pipeline, o.pipe.layout, o.pipe.set_layout, o.pipe.push)
    }
}

/// The last full draw of an executor that a draw right after it can take
/// over (a buffer-delta draw, `tune buffer_delta`, RENDERER.md): such a
/// draw records its buffers, SGPR data, indices and the draw; texture sync,
/// targets, pipeline, image hazards, the rendering instance and dynamic
/// state are this draw's, and its bound images and samplers while their
/// memos hold.
#[derive(Default)]
pub(super) struct DrawDelta {
    /// Valid for a draw at this (batch, recorded-command count) only (taken
    /// by every draw: one that returns early leaves none).
    pub(super) at: Option<(u64, u64)>,
    /// The stages' translations.
    keys: [u64; 2],
    /// Indexed.
    index: bool,
    /// The stages' images and samplers (`image_sig`).
    images: Vec<(u32, [u32; 8])>,
    /// The textures' ranges (texture sync's).
    textures: Vec<(u64, u64)>,
    /// The attachments' and images' keys (the hazard check's).
    used: Vec<images::ImageKey>,
    fixed: Option<Fixed>,
    /// This executor's counts not yet added to `DELTA` and `DELTA_BIND`
    /// (every `DELTA_FLUSH` candidates: nine recording threads adding to
    /// shared counters at every draw contended for their cache lines).
    counts: [u64; 11],
    /// The last draw's stages' bound images and samplers, when both came
    /// from the binding memos, and the generations they hold for
    /// (`images::LOOKUP_GEN`, `track::GEN`, the frame: `exec::bind_gens`).
    bind: [BoundImages; 2],
    bind_gens: Option<[u64; 3]>,
}

/// A stage's bound images and samplers (`Bound`'s).
#[derive(Default)]
struct BoundImages {
    imgs: Vec<(u32, vk::DescriptorType, vk::DescriptorImageInfo)>,
    images: Vec<vk::Image>,
    stencil_views: Vec<vk::Image>,
}

/// Candidates between additions of an executor's counts to the shared ones.
const DELTA_FLUSH: u64 = 1024;

impl DrawDelta {
    /// Counts `k` (`DELTA`'s index; `DELTA_BIND`'s from 9).
    fn count(&mut self, k: usize) {
        self.counts[k] += 1;
        if k == 0 && self.counts[0] >= DELTA_FLUSH {
            self.flush();
        }
    }

    fn flush(&mut self) {
        for (k, n) in self.counts.iter_mut().enumerate() {
            if *n != 0 {
                let c = if k < 9 { &DELTA[k] } else { &DELTA_BIND[k - 9] };
                c.fetch_add(std::mem::take(n), Ordering::Relaxed);
            }
        }
    }

    /// The generations the last draw's bound images and samplers hold for.
    pub(super) fn bind_gens(&self) -> Option<[u64; 3]> {
        self.bind_gens
    }

    /// Stage `k`'s images and samplers into `b` (`bind_stage`).
    pub(super) fn bind_into(&self, k: usize, b: &mut Bound) {
        let s = &self.bind[k];
        b.imgs.extend_from_slice(&s.imgs);
        b.images.extend_from_slice(&s.images);
        b.stencil_views.extend_from_slice(&s.stencil_views);
    }

    /// Whether `b` binds stage `k`'s images and samplers (`tune buffer_delta 2`).
    fn bind_same(&self, k: usize, b: &Bound) -> bool {
        let s = &self.bind[k];
        let info = |x: &(u32, vk::DescriptorType, vk::DescriptorImageInfo)| (x.0, x.1, x.2.sampler, x.2.image_view, x.2.image_layout);
        s.imgs.iter().map(info).eq(b.imgs.iter().map(info)) && s.images == b.images && s.stencil_views == b.stencil_views
    }
}

/// The stages' image and sampler resources (stage << 16 | resource, the
/// descriptor's words), into `out`: false with an image table.
fn image_sig(stages: [Option<&Prepared>; 2], out: &mut Vec<(u32, [u32; 8])>) -> bool {
    out.clear();
    for (k, p) in stages.iter().enumerate() {
        let Some(p) = p else { continue };
        for (i, r) in p.resources.iter().enumerate() {
            let at = (k as u32) << 16 | i as u32;
            match r {
                Resolved::Image(t) => out.push((at, t.raw)),
                Resolved::Sampler(w) => out.push((at | 1 << 31, [w[0], w[1], w[2], w[3], 0, 0, 0, 0])),
                Resolved::ImageTable(_) => return false,
                _ => {}
            }
        }
    }
    true
}

/// A difference `tune buffer_delta 2` found (kind as `DELTA_VERIFY`'s,
/// less one), the first ones logged.
fn delta_mismatch(kind: usize, vs: u64, ps: u64) {
    let n = DELTA_VERIFY[1 + kind].fetch_add(1, Ordering::Relaxed);
    static LOGGED: AtomicU64 = AtomicU64::new(0);
    if n < 4 && LOGGED.fetch_add(1, Ordering::Relaxed) < 24 {
        const WHAT: [&str; 6] = ["a texture wait", "other targets or pipeline", "an image hazard", "the rendering instance ended", "dynamic state recorded", "other bound images or samplers"];
        crate::warn!("gpu: buffer-delta verify: {} on the full path (vs {vs:#x} ps {ps:#x})", WHAT[kind]);
    }
}

/// Status part: `DELTA`, `DELTA_VERIFY`.
pub(super) fn delta_line() -> String {
    use std::sync::atomic::Ordering::Relaxed;
    let d: Vec<u64> = DELTA.iter().map(|c| c.load(Relaxed)).collect();
    if d[0] == 0 {
        return String::new();
    }
    let v: Vec<u64> = DELTA_VERIFY.iter().map(|c| c.load(Relaxed)).collect();
    let b: Vec<u64> = DELTA_BIND.iter().map(|c| c.load(Relaxed)).collect();
    let verified = if v[0] > 0 {
        format!("; verified {} (differences: texture waits {}, targets or pipeline {}, image hazards {}, instance ended {}, dynamic state {}, bound images or samplers {})", v[0], v[1], v[2], v[3], v[4], v[5], v[6])
    } else {
        String::new()
    };
    let costs = if super::laps() {
        let us = |k: usize, d: usize| DELTA_COST[k][d][0].load(Ordering::Relaxed) as f64 / DELTA_COST[k][d][1].load(Ordering::Relaxed).max(1) as f64 / 1e3;
        let n = |k: usize, d: usize| DELTA_COST[k][d][1].load(Ordering::Relaxed);
        format!(
            "
    a draw's steps in all (`DS_STEP_LAPS`, not a batch's first; buffer-delta / other): graphics serial {:.2} us ({}) / {:.2} ({}), own chunks {:.2} ({}) / {:.2} ({}), workers {:.2} ({}) / {:.2} ({})",
            us(0, 1), n(0, 1), us(0, 0), n(0, 0), us(1, 1), n(1, 1), us(1, 0), n(1, 0), us(2, 1), n(2, 1), us(2, 0), n(2, 0)
        )
        + &{
            let p: Vec<u64> = PAIR_CENSUS.iter().map(|c| c.load(Ordering::Relaxed)).collect();
            format!(
                "\n    draws that were not buffer-delta draws (a batch's first apart): {}; with the pair of the draw before {}, of the one before that {}, of one of the two before those {}, of one of four more {}, the first since a dispatch or of its instance {}, none of the last eight pairs {}",
                p[0], p[1], p[2], p[3], p[4], p[5], p[6]
            )
        }
    } else {
        String::new()
    };
    let e: Vec<u64> = EMPTY_PS.iter().map(|c| c.load(Relaxed)).collect();
    let empty = if e.iter().any(|&x| x != 0) {
        format!("    pixel stages that read no memory: {} preparations taken from the draw before (counted with `DS_STEP_LAPS`), {} compared with a fresh one ({} differing; `tune buffer_delta` 2)\n", e[0], e[1], e[2])
    } else {
        String::new()
    };
    empty + &format!(
        "    buffer-delta draws: {} of {} candidates (refused: modes {}, translations or indices {}, images or samplers {}, image lookups {}, clears or reloads pending {}, textures {}, rendering instance {}); stages binding the draw before's images and samplers {}, looking them up {}{verified}{costs}\n",
        d[1], d[0], d[2], d[3], d[4], d[5], d[6], d[7], d[8], b[0], b[1]
    )
}

/// An indexed indirect draw whose arguments the CPU read, drawn in
/// `cpu_args_base` mode with its indices in a home, that the next can
/// repeat (`draw_repeat_cpu`): the batch and `cmd_uses` right after it was
/// recorded, its index size and the home's buffer and offset bound last.
#[derive(Clone, Copy)]
pub(crate) struct Repeat {
    batch: u64,
    uses: u64,
    index: (u64, u32, u64),
    index_buf: vk::Buffer,
    index_off: u64,
}

const MAX_RT: usize = 8;

/// Active dynamic-rendering instance of an executor.
/// Dynamic graphics state recorded in the current command buffer: per
/// state its value plus one (0: not recorded yet), the bound pipeline, the
/// viewport, scissor, depth bias and blend constants (all of them as bits).
/// A new command buffer starts with nothing recorded; only draws record
/// graphics state, and every graphics pipeline makes the same states
/// dynamic. Re-setting unchanged state cost ~1.2 us per draw (~25 calls, run
/// 20261011-133502).
#[derive(Default)]
pub(crate) struct DynCache {
    words: [u64; DYN_SLOTS],
    pipeline: vk::Pipeline,
    /// The compute pipeline bound last in this command buffer.
    pub(crate) cs_pipeline: vk::Pipeline,
    /// The push descriptors in this command buffer, per bind point
    /// (graphics, compute; `bind_descriptors`).
    pub(crate) pushed: [super::PushedSet; 2],
    viewport: Option<[u32; 6]>,
    scissor: Option<[i32; 4]>,
    bias: Option<[u32; 3]>,
    blend: Option<[u32; 4]>,
}

const DYN_SLOTS: usize = 22;

impl DynCache {
    /// What it holds apart from the push descriptors (`tune buffer_delta 2`).
    fn sig(&self) -> ([u64; DYN_SLOTS], vk::Pipeline, Option<[u32; 6]>, Option<[i32; 4]>, Option<[u32; 3]>, Option<[u32; 4]>) {
        (self.words, self.pipeline, self.viewport, self.scissor, self.bias, self.blend)
    }

    /// Whether state `slot` must be recorded with value `v` (and notes it).
    fn set(&mut self, slot: usize, v: u64) -> bool {
        let w = v.wrapping_add(1);
        if self.words[slot] == w {
            return false;
        }
        self.words[slot] = w;
        true
    }
}

pub(super) struct Rendering {
    pub colors: [Option<Target>; MAX_RT],
    pub depth: Option<Target>,
    pub area: (u32, u32),
    /// Color attachments its draws wrote, and those they sampled; the depth
    /// attachment's depth and stencil aspects written and sampled (one
    /// instance never does both to one aspect of an attachment: `tune
    /// feedback_exact`).
    pub written: Vec<vk::Image>,
    pub sampled: Vec<vk::Image>,
    pub ds_written: [bool; 2],
    pub ds_sampled: [bool; 2],
}

/// Arguments of one draw packet.
pub struct DrawArgs {
    /// Vertices (non-indexed) or indices.
    pub count: u32,
    pub instances: u32,
    /// Guest address of the first index and the index size in bytes.
    pub index: Option<(u64, u32)>,
    /// First vertex (non-indexed) or vertex offset (indexed); None: VGT_INDX_OFFSET.
    pub base_vertex: Option<i32>,
    /// Added to the instance id VGPR (indirect draws whose packet names no
    /// start-instance SGPR; see the CP).
    pub first_instance: u32,
    /// The vertex stage's user-data word that takes the start instance from
    /// `gl_BaseInstance`: an indexed indirect draw whose arguments the CPU
    /// read, drawn with `first_instance` as Vulkan's first instance (`tune
    /// cpu_args_base`); None: any start instance is in that word's register.
    pub base_inst: Option<u16>,
}


/// Vertex invocations (count x instances) from which a draw is logged as
/// absurd (`Exec::note_huge_draw`).
const HUGE_DRAW: u64 = 1 << 26;
static HUGE_DRAWS: AtomicU64 = AtomicU64::new(0);

impl Exec {
    /// Logs a draw of `HUGE_DRAW` vertex invocations or more (its executor,
    /// packet, programs and arguments; the first 16 a boot).
    #[cold]
    fn note_huge_draw(&self, cp: &Cp, a: &DrawArgs) {
        if HUGE_DRAWS.fetch_add(1, Ordering::Relaxed) >= 16 {
            return;
        }
        let ps = ((cp.reg(Space::Sh, 0x08) as u64) | ((cp.reg(Space::Sh, 0x09) as u64 & 0xFF) << 32)) << 8;
        crate::warn!(
            "gpu[{}]: a draw of {} x {} vertex invocations at packet {:#x} (vs {:#x} ps {ps:#x}, index {:?}, base vertex {:?}, first instance {}){}",
            self.name,
            a.count,
            a.instances,
            cp.packet_addr(),
            shaders::ngg_addr(cp),
            a.index,
            a.base_vertex,
            a.first_instance,
            if a.count as u64 * a.instances as u64 >= ABSURD_DRAW { ": dropped" } else { "" }
        );
    }
}

/// Vertex invocations (count x instances) from which a draw is dropped (`ABSURD_DROPPED`): no frame
/// of the title draws a billion vertices, and such a draw, its count read from stale data, held
/// the GPU until the device was lost (2026-11-23, a failing 16 GB first session: 1,056,800,768 x
/// 1,029 at a level load).
const ABSURD_DRAW: u64 = 1 << 30;
pub static ABSURD_DROPPED: AtomicU64 = AtomicU64::new(0);

/// `draw` outcome when the vertex stage needs the start instance on the CPU
/// (`shaders::uses_user_word`): the command processor reads the arguments
/// and draws again.
pub const NEEDS_CPU_ARGS: &str = "indirect arguments needed on the CPU";

fn ctx_reg(cp: &Cp, a: u32) -> u32 {
    cp.reg(Space::Ctx, (a - 0x28000) / 4)
}

fn f32r(v: u32) -> f32 {
    f32::from_bits(v)
}

/// CB_BLENDn_CONTROL blend factor.
fn blend_factor(f: u32) -> vk::BlendFactor {
    use vk::BlendFactor as B;
    match f {
        0 => B::ZERO,
        1 => B::ONE,
        2 => B::SRC_COLOR,
        3 => B::ONE_MINUS_SRC_COLOR,
        4 => B::SRC_ALPHA,
        5 => B::ONE_MINUS_SRC_ALPHA,
        6 => B::DST_ALPHA,
        7 => B::ONE_MINUS_DST_ALPHA,
        8 => B::DST_COLOR,
        9 => B::ONE_MINUS_DST_COLOR,
        10 => B::SRC_ALPHA_SATURATE,
        13 => B::CONSTANT_COLOR,
        14 => B::ONE_MINUS_CONSTANT_COLOR,
        15 => B::SRC1_COLOR,
        16 => B::ONE_MINUS_SRC1_COLOR,
        17 => B::SRC1_ALPHA,
        18 => B::ONE_MINUS_SRC1_ALPHA,
        19 => B::CONSTANT_ALPHA,
        20 => B::ONE_MINUS_CONSTANT_ALPHA,
        _ => B::ONE,
    }
}

/// CB_BLENDn_CONTROL combine function.
fn blend_op(f: u32) -> vk::BlendOp {
    match f {
        1 => vk::BlendOp::SUBTRACT,
        2 => vk::BlendOp::MIN,
        3 => vk::BlendOp::MAX,
        4 => vk::BlendOp::REVERSE_SUBTRACT,
        _ => vk::BlendOp::ADD,
    }
}

/// DB_DEPTH_CONTROL ZFUNC / STENCILFUNC.
fn compare_op(f: u32) -> vk::CompareOp {
    match f & 7 {
        0 => vk::CompareOp::NEVER,
        1 => vk::CompareOp::LESS,
        2 => vk::CompareOp::EQUAL,
        3 => vk::CompareOp::LESS_OR_EQUAL,
        4 => vk::CompareOp::GREATER,
        5 => vk::CompareOp::NOT_EQUAL,
        6 => vk::CompareOp::GREATER_OR_EQUAL,
        _ => vk::CompareOp::ALWAYS,
    }
}

/// DB_STENCIL_CONTROL operation (ONES/logic ops approximated).
fn stencil_op(op: u32) -> vk::StencilOp {
    match op & 0xF {
        1 => vk::StencilOp::ZERO,
        3 | 4 => vk::StencilOp::REPLACE,
        5 => vk::StencilOp::INCREMENT_AND_CLAMP,
        6 => vk::StencilOp::DECREMENT_AND_CLAMP,
        7 => vk::StencilOp::INVERT,
        8 => vk::StencilOp::INCREMENT_AND_WRAP,
        9 => vk::StencilOp::DECREMENT_AND_WRAP,
        _ => vk::StencilOp::KEEP,
    }
}

/// Draws with a stencil state Vulkan's cannot reproduce, by kind: [the
/// pixel shader exports the test value (DB_SHADER_CONTROL bit 1) and a
/// compare function or REPLACE_TEST op uses it, it exports the op value
/// (bit 2) and a REPLACE_OP op writes it, a REPLACE_OP op writes an op
/// value other than the test value (Vulkan replaces with the reference),
/// an op without a Vulkan equivalent (ONES, logic ops: KEEP)]. Each
/// distinct (kind, pixel shader, registers) is logged once.
pub static STENCIL_GAPS: [AtomicU64; 4] = [const { AtomicU64::new(0) }; 4];

fn note_stencil_gaps(ps: u64, dctl: u32, sctl: u32, refm: u32, refm_bf: u32, dbsc: u32) {
    let back = dctl & (1 << 7) != 0;
    let funcs = [(dctl >> 8) & 7, if back { (dctl >> 20) & 7 } else { (dctl >> 8) & 7 }];
    let bs = if back { sctl >> 12 } else { sctl };
    let ops = [sctl & 0xF, (sctl >> 4) & 0xF, (sctl >> 8) & 0xF, bs & 0xF, (bs >> 4) & 0xF, (bs >> 8) & 0xF];
    let uses_ref = funcs.iter().any(|&f| f != 0 && f != 7);
    let kinds = [
        dbsc & 2 != 0 && (uses_ref || ops.contains(&3)),
        dbsc & 4 != 0 && ops.contains(&4),
        ops.contains(&4) && ((refm >> 24) & 0xFF != refm & 0xFF || (back && (refm_bf >> 24) & 0xFF != refm_bf & 0xFF)),
        ops.iter().any(|&o| o == 2 || o >= 10),
    ];
    if !kinds.iter().any(|&k| k) {
        return;
    }
    static SEEN: parking_lot::Mutex<Option<std::collections::HashSet<(usize, u64, u32, u32, u32, u32)>>> = parking_lot::Mutex::new(None);
    for (k, &on) in kinds.iter().enumerate() {
        if !on {
            continue;
        }
        STENCIL_GAPS[k].fetch_add(1, Ordering::Relaxed);
        let mut g = SEEN.lock();
        let seen = g.get_or_insert_with(Default::default);
        if seen.len() < 200 && seen.insert((k, ps, dctl, sctl, refm, dbsc)) {
            const NAMES: [&str; 4] = ["test value exported and used", "op value exported and replaced", "REPLACE_OP with an op value other than the test value", "an op without a Vulkan equivalent"];
            crate::info!("stencil gap: {} (ps {ps:#x}, depth_ctl {dctl:#x}, stencil_ctl {sctl:#x}, ref {refm:#x}/{refm_bf:#x}, db_shader {dbsc:#x})", NAMES[k]);
        }
    }
}

/// EQUAL depth test diagnosis (`DS_DEPTH_EQUAL_BIAS` or the `eqbias`
/// control command): f32 bits, NaN = off, infinity = always pass.
pub static EQUAL_BIAS: AtomicU32 = AtomicU32::new(0x7FC0_0000);

pub fn set_equal_bias(v: Option<f32>) {
    EQUAL_BIAS.store(v.unwrap_or(f32::NAN).to_bits(), Ordering::Relaxed);
}

/// Whether a draw trace is on (`DS_TRACE_PS`, `DS_TRACE_PS_BINS`, `DS_TRACE_ARGS`,
/// `DS_TRACE_DEPTH`): the draw path's one check while they are off.
#[inline]
fn draw_traces() -> bool {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ON.get_or_init(|| ["DS_TRACE_PS", "DS_TRACE_PS_BINS", "DS_TRACE_ARGS", "DS_TRACE_DEPTH"].iter().any(|v| std::env::var_os(v).is_some()))
}

/// `DS_DECAL_CHECK=<hex image base>` (diagnosis, 2026-11-20): see `note_decal_bufs`.
pub fn decal_check() -> u64 {
    static D: std::sync::OnceLock<u64> = std::sync::OnceLock::new();
    *D.get_or_init(|| std::env::var("DS_DECAL_CHECK").ok().and_then(|v| u64::from_str_radix(v.trim().trim_start_matches("0x"), 16).ok()).unwrap_or(0))
}

/// The read-only buffers bound for draws whose pixel stage stores to `decal_check`'s image:
/// (flip, program, guest base, the bytes as recorded (at most 256), buffer, offset of the base in
/// it), compared at the next `atprobe` with what that buffer holds then (`decal check` lines).
pub static DECAL_BUFS: parking_lot::Mutex<Vec<(u64, u64, u64, Vec<u8>, vk::Buffer, u64)>> = parking_lot::Mutex::new(Vec::new());

fn note_decal_bufs(p: &Prepared, b: &Bound) {
    let img = decal_check();
    if !p.found.resources.iter().enumerate().any(|(i, r)| matches!(r, Resolved::Image(t) if t.base == img) && p.tr.plan.resources.get(i).is_some_and(|x| x.written)) {
        return;
    }
    let flip = crate::hle::libs::videoout::QUEUED_FLIP.load(Ordering::Relaxed) + 1;
    let mirror = super::super::resident::mirror_buffer().map(|m| m.0);
    let mut g = DECAL_BUFS.lock();
    if g.len() > 4096 {
        return;
    }
    for (i, r) in p.found.resources.iter().enumerate() {
        let Resolved::Buffer(v) = r else { continue };
        if p.tr.plan.resources.get(i).is_some_and(|x| x.written) {
            continue;
        }
        let len = (v.stride.max(16) as u64 * v.num_records.max(1) as u64).min(256) & !3;
        let Some(bn) = p.tr.out.bindings.iter().find(|bd| matches!(bd.kind, BindingKind::Buffer(j) if j == i)).map(|bd| bd.binding) else { continue };
        let Some(inf) = b.bufs.iter().find(|(x, _)| *x == bn).map(|(_, inf)| *inf) else { continue };
        // Only the mirror's placements: their offset is the guest address's.
        if Some(inf.buffer) != mirror || v.base < super::super::resident::MIRROR_BASE || len == 0 || !super::super::cp::guest_readable(v.base, len) {
            continue;
        }
        let bytes: Vec<u8> = (0..len).map(|k| unsafe { std::ptr::read_volatile((v.base + k) as *const u8) }).collect();
        g.push((flip, p.addr, v.base, bytes, inf.buffer, v.base - super::super::resident::MIRROR_BASE));
    }
}

/// `skiprw <hex base>` (diagnosis): draws whose pixel stage writes the image at that base are
/// left out (0: none), and how many were.
pub static SKIP_RW: AtomicU64 = AtomicU64::new(0);
pub static SKIP_RW_N: AtomicU64 = AtomicU64::new(0);
/// The ordinals among a flip's such draws left out ([lo, hi); all by default), the current
/// flip's count, and the (ordinal, program)s logged.
pub static SKIP_RW_RANGE: parking_lot::Mutex<(u64, u64)> = parking_lot::Mutex::new((0, u64::MAX));
static SKIP_RW_ORD: parking_lot::Mutex<(u64, u64)> = parking_lot::Mutex::new((0, 0));
static SKIP_RW_SEEN: parking_lot::Mutex<std::collections::BTreeSet<(u64, u64)>> = parking_lot::Mutex::new(std::collections::BTreeSet::new());

/// Draws whose vertex shader is on this list are skipped (`skipvs` control
/// command; diagnosis).
static SKIP_VS: parking_lot::Mutex<Option<std::collections::HashSet<u64>>> = parking_lot::Mutex::new(None);
/// Whether the list has entries (read by every draw without the lock).
static SKIP_VS_ANY: std::sync::atomic::AtomicBool = std::sync::atomic::AtomicBool::new(false);

/// Adds or removes a vertex shader address; returns whether it is listed now.
pub fn toggle_skip_vs(addr: u64) -> bool {
    let mut g = SKIP_VS.lock();
    let s = g.get_or_insert_with(Default::default);
    let listed = if s.remove(&addr) {
        false
    } else {
        s.insert(addr);
        true
    };
    SKIP_VS_ANY.store(!s.is_empty(), Ordering::Relaxed);
    listed
}

/// `DS_TRACE_PS=<hex>[,<hex>...]` (diagnosis): each draw of those pixel programs logs, at the
/// flip it is recorded in, its image resources (T# base, type, size) and what they were bound
/// to (view and image handles, null views of a 2D/3D binding left unused), and whether the
/// bindings came from the thread's memo (`trace ps` lines).
fn trace_ps_addrs() -> &'static [u64] {
    static ADDRS: std::sync::OnceLock<Vec<u64>> = std::sync::OnceLock::new();
    ADDRS.get_or_init(|| {
        std::env::var("DS_TRACE_PS").unwrap_or_default().split([',', ';']).filter_map(|v| u64::from_str_radix(v.trim().trim_start_matches("0x"), 16).ok()).collect()
    })
}

/// `DS_TRACE_DEPTH=<hex base>` (diagnosis): each flip's draws with a depth attachment at that
/// address, by attachment image, key and depth write (one line per flip and combination, with
/// the count of draws and the first draw's programs).
fn trace_depth_attachment(vs: u64, ps: u64, depth: &Option<Target>, z_write: bool) {
    static BASE: std::sync::OnceLock<u64> = std::sync::OnceLock::new();
    let base = *BASE.get_or_init(|| std::env::var("DS_TRACE_DEPTH").ok().and_then(|v| u64::from_str_radix(v.trim().trim_start_matches("0x"), 16).ok()).unwrap_or(0));
    let Some(d) = depth.as_ref().filter(|d| base != 0 && d.key.base == base) else { return };
    use ash::vk::Handle;
    static SEEN: parking_lot::Mutex<(u64, Vec<(u64, u32, u32, bool, u64, u64, u32)>)> = parking_lot::Mutex::new((0, Vec::new()));
    let flip = crate::hle::libs::videoout::QUEUED_FLIP.load(Ordering::Relaxed) + 1;
    let mut g = SEEN.lock();
    if g.0 != flip {
        for &(img, w, h, zw, v, p, n) in &g.1 {
            crate::info!("trace depth {base:#x} for flip {}: image {img:x} {w}x{h} z write {zw}: {n} draws, first vs {v:#x} ps {p:#x}", g.0);
        }
        g.0 = flip;
        g.1.clear();
    }
    let img = d.image.as_raw();
    match g.1.iter_mut().find(|e| e.0 == img && e.3 == z_write) {
        Some(e) => e.6 += 1,
        None => g.1.push((img, d.key.width, d.key.height, z_write, vs, ps, 1)),
    }
}

/// `DS_TRACE_PS`: the traced draw's attachments (key base and image handle).
fn trace_ps_targets(ps: u64, colors: &[Option<Target>; MAX_RT], depth: &Option<Target>) {
    let addrs = trace_ps_addrs();
    if addrs.is_empty() || !addrs.contains(&ps) {
        return;
    }
    use ash::vk::Handle;
    let t: Vec<String> = colors.iter().enumerate().filter_map(|(i, c)| c.map(|c| format!("c{i}:{:#x}={:x}", c.key.base, c.image.as_raw()))).chain(depth.iter().map(|d| format!("z:{:#x}={:x}", d.key.base, d.image.as_raw()))).collect();
    crate::info!("trace ps {ps:#x} targets at flip {}: {}", crate::hle::libs::videoout::QUEUED_FLIP.load(Ordering::Relaxed) + 1, t.join(" "));
}

/// `DS_TRACE_PS_BINS=<path>[;<path>...]` (diagnosis): programs whose code up to s_endpgm equals
/// those of a saved shader binary (a capture's `shaders/<address>.bin`) are traced as
/// `DS_TRACE_PS` addresses are, wherever this boot loaded them (level programs move).
fn trace_bin_match(addr: u64) -> bool {
    static BINS: std::sync::OnceLock<Vec<Vec<u8>>> = std::sync::OnceLock::new();
    let bins = BINS.get_or_init(|| {
        std::env::var("DS_TRACE_PS_BINS").unwrap_or_default().split(';').filter(|s| !s.is_empty()).filter_map(|p| std::fs::read(p.trim()).ok()).map(|mut b| {
            // The program up to its first s_endpgm (0xbf810000).
            let end = b.chunks_exact(4).position(|w| w == [0x00, 0x00, 0x81, 0xbf]).map_or(b.len(), |i| 4 * i + 4);
            b.truncate(end);
            b
        }).collect()
    });
    if bins.is_empty() {
        return false;
    }
    static SEEN: parking_lot::Mutex<Option<std::collections::HashMap<u64, bool>>> = parking_lot::Mutex::new(None);
    let mut g = SEEN.lock();
    let m = g.get_or_insert_with(Default::default);
    *m.entry(addr).or_insert_with(|| {
        let hit = bins.iter().any(|b| super::super::cp::guest_readable(addr, b.len() as u64) && unsafe { std::slice::from_raw_parts(addr as *const u8, b.len()) } == &b[..]);
        if hit {
            crate::info!("trace ps: program {addr:#x} matches a DS_TRACE_PS_BINS binary");
        }
        hit
    })
}

thread_local! {
    /// The command processor is recording a draw whose indirect arguments lie at
    /// `DS_TRACE_ARGS` (diagnosis): its stages are traced as `DS_TRACE_PS` ones are.
    pub static TRACE_DRAW: std::cell::Cell<bool> = const { std::cell::Cell::new(false) };
}

/// `DS_TRACE_ARGS=<hex>` (diagnosis): the indirect argument address whose draws are traced.
pub fn trace_args_addr() -> u64 {
    static A: std::sync::OnceLock<u64> = std::sync::OnceLock::new();
    *A.get_or_init(|| std::env::var("DS_TRACE_ARGS").ok().and_then(|v| u64::from_str_radix(v.trim().trim_start_matches("0x"), 16).ok()).unwrap_or(0))
}

/// `DS_TRACE_PS_REC=<first word>:<count>` (diagnosis, 2026-11-20): with `DS_TRACE_PS` /
/// `DS_TRACE_PS_BINS`, each traced draw logs one short line in place of the long ones (`trace
/// rec`): the flip, the indices (guest address and count) and instances, the user words, and
/// each buffer resource's base, stride and records with its guest words [first, first + count)
/// as they are when the draw is recorded (the remaining flicker's wall: the per-object record's
/// blend weight at word 28).
fn trace_rec_words() -> Option<(u64, u64)> {
    static W: std::sync::OnceLock<Option<(u64, u64)>> = std::sync::OnceLock::new();
    *W.get_or_init(|| {
        let v = std::env::var("DS_TRACE_PS_REC").ok()?;
        let (a, b) = v.split_once(':')?;
        Some((a.trim().parse().ok()?, b.trim().parse().ok()?))
    })
}

fn trace_ps_binding(p: &Prepared, b: &Bound, note: Option<([u64; 3], bool)>, a: &DrawArgs) {
    let addrs = trace_ps_addrs();
    if !addrs.contains(&p.addr) && !trace_bin_match(p.addr) && !TRACE_DRAW.with(|c| c.get()) {
        return;
    }
    if let Some((first, n)) = trace_rec_words() {
        // (`DS_TRACE_PS_REC_PROBE=1`: only while `atprobe` runs.)
        if std::env::var("DS_TRACE_PS_REC_PROBE").is_ok_and(|v| v == "1") && super::AT_PROBE.lock().is_none() {
            return;
        }
        let bufs: Vec<String> = p.found.resources.iter().enumerate().filter_map(|(i, r)| match r {
            Resolved::Buffer(v) => {
                let words: Vec<String> = (first..first + n).map(|k| v.base + 4 * k).map(|w| if super::super::cp::guest_readable(w, 4) { format!("{:08x}", unsafe { std::ptr::read_volatile(w as *const u32) }) } else { "-".into() }).collect();
                Some(format!("{i}:{:#x}/{}x{} [{}]", v.base, v.stride, v.num_records, words.join(" ")))
            }
            _ => None,
        }).collect();
        // (`DS_ATPROBE_BOUND`: the first buffer resource's binding, for `atprobe bound`.)
        if let Some((i, v)) = p.found.resources.iter().enumerate().find_map(|(i, r)| match r {
            Resolved::Buffer(v) => Some((i, v)),
            _ => None,
        }) {
            let bn = p.tr.out.bindings.iter().find(|bd| matches!(bd.kind, BindingKind::Buffer(j) if j == i)).map(|bd| bd.binding);
            if let Some(inf) = bn.and_then(|bn| b.bufs.iter().find(|(x, _)| *x == bn).map(|(_, inf)| *inf)) {
                *super::PROBE_BIND.lock() = Some((v.base, (v.stride.max(16) as u64 * v.num_records.max(1) as u64).min(4096), inf.buffer, crate::hle::libs::videoout::QUEUED_FLIP.load(Ordering::Relaxed) + 1));
            }
        }
        use ash::vk::Handle;
        let places: Vec<String> = b.placements.iter().map(|(pl, n, w)| format!("{pl:?}/{n}{}", if *w { "w" } else { "" })).collect();
        let bound: Vec<String> = b.bufs.iter().map(|(bn, inf)| format!("b{bn}={:x}+{:#x}/{:#x}", inf.buffer.as_raw(), inf.offset, inf.range)).collect();
        crate::info!(
            "trace rec {:#x} flip {}: idx {} count {} inst {} first {} user {:x?} {} places [{}] bound [{}]",
            p.addr,
            crate::hle::libs::videoout::QUEUED_FLIP.load(Ordering::Relaxed) + 1,
            a.index.map_or("-".into(), |(ib, sz)| format!("{ib:#x}/{sz}")),
            a.count,
            a.instances,
            a.first_instance,
            p.user,
            bufs.join(" | "),
            places.join(" "),
            bound.join(" ")
        );
        return;
    }
    use ash::vk::Handle;
    let res: Vec<String> = p.found.resources.iter().enumerate().filter_map(|(i, r)| match r {
        Resolved::Image(t) => Some(format!("{i}:{:#x} t{} {}x{}x{} f{}", t.base, t.ty, t.width, t.height, t.depth, t.format)),
        _ => None,
    }).collect();
    // Buffer resources: their guest bytes now (first 12 words) and the bound buffer/offset.
    let bufs: Vec<String> = p.found.resources.iter().enumerate().filter_map(|(i, r)| match r {
        Resolved::Buffer(v) => {
            let words: Vec<String> = (0..12u64).map(|k| v.base + 4 * k).map(|a| if super::super::cp::guest_readable(a, 4) { format!("{:08x}", unsafe { std::ptr::read_volatile(a as *const u32) }) } else { "-".into() }).collect();
            let bn = p.tr.out.bindings.iter().find(|bd| matches!(bd.kind, BindingKind::Buffer(j) if j == i)).map(|bd| bd.binding);
            let info = bn.and_then(|bn| b.bufs.iter().find(|(x, _)| *x == bn).map(|(_, inf)| *inf));
            Some(format!("{i}:{:#x} [{}] bound {}", v.base, words.join(" "), info.map_or("-".into(), |inf| format!("{:x}+{:#x}", inf.buffer.as_raw(), inf.offset))))
        }
        _ => None,
    }).collect();
    if !bufs.is_empty() {
        crate::info!("trace ps {:#x} buffers at flip {}: {}", p.addr, crate::hle::libs::videoout::QUEUED_FLIP.load(Ordering::Relaxed) + 1, bufs.join(" | "));
    }
    let imgs: Vec<String> = b.imgs.iter().filter(|e| e.1 != vk::DescriptorType::SAMPLER).map(|e| format!("b{}={:x}", e.0, e.2.image_view.as_raw())).collect();
    let images: Vec<String> = b.images.iter().map(|i| format!("{:x}", i.as_raw())).collect();
    crate::info!(
        "trace ps {:#x} at flip {}: user {:x?} base inst {:?} images [{}] views [{}] vk images [{}] memo {:?} loads {}",
        p.addr,
        crate::hle::libs::videoout::QUEUED_FLIP.load(Ordering::Relaxed) + 1,
        p.user,
        p.tr.spec.base_instance_sgpr,
        res.join(", "),
        imgs.join(" "),
        images.join(" "),
        note.map(|n| n.1),
        super::super::shaders::describe_loads(p).join(" | ")
    );
}

fn equal_bias() -> Option<f32> {
    static INIT: std::sync::Once = std::sync::Once::new();
    INIT.call_once(|| {
        if let Ok(v) = std::env::var("DS_DEPTH_EQUAL_BIAS") {
            set_equal_bias(if v == "always" { Some(f32::INFINITY) } else { v.parse().ok() });
        }
    });
    Some(f32::from_bits(EQUAL_BIAS.load(Ordering::Relaxed))).filter(|v| !v.is_nan())
}

/// Color target n from CB_COLORn_* (None: no target bound).
pub(crate) fn color_surface(cp: &Cp, n: u32) -> Result<Option<Surface>, String> {
    let r = |o: u32| ctx_reg(cp, 0x28C60 + 0x3C * n + o);
    let info = r(0x10);
    let fmt = (info >> 2) & 0x1F;
    if fmt == 0 {
        return Ok(None);
    }
    let format = images::cb_format(fmt, (info >> 8) & 7, (info >> 11) & 3).ok_or_else(|| format!("color target format {fmt} number type {} swap {} unsupported", (info >> 8) & 7, (info >> 11) & 3))?;
    let base = ((r(0) as u64) << 8) | ((ctx_reg(cp, 0x28E40 + 4 * n) as u64 & 0xFF) << 40);
    let a2 = ctx_reg(cp, 0x28EC0 + 4 * n);
    let a3 = ctx_reg(cp, 0x28EE0 + 4 * n);
    let view = r(0xC);
    // DCC (INFO.DCC_ENABLE): fast clears write only its metadata.
    let meta = if info & (1 << 28) != 0 { ((r(0x34) as u64) << 8) | ((ctx_reg(cp, 0x28EA0 + 4 * n) as u64 & 0xFF) << 40) } else { 0 };
    Ok(Some(Surface {
        base,
        format,
        width: ((a2 >> 14) & 0x3FFF) + 1,
        height: (a2 & 0x3FFF) + 1,
        layers: (a3 & 0x1FFF) + 1,
        levels: (a2 >> 28) + 1,
        ty3d: (a3 >> 24) & 3 == 2,
        mip: (view >> 26) & 0xF,
        slice: view & 0x1FFF,
        stencil_base: 0,
        meta,
        clear: [r(0x2C), r(0x30)],
    }))
}

/// A draw's pass for the pass census (`exec::pass_note`): its depth target's base (0: none)
/// and, packed, the depth target's size (bits 0-27), the first bound colour slot's size (bits
/// 28-55) and the colour slots bound (from bit 56; a slot counts by its format, as
/// `color_surface` reads it).
pub(crate) fn pass_key(cp: &Cp) -> (u64, u64) {
    let (depth, d) = depth_surface(cp).map_or((0, 0), |s| (s.base, (s.width as u64) << 14 | s.height as u64));
    let (mut n, mut c0) = (0u64, 0u64);
    for i in 0..MAX_RT as u32 {
        if (ctx_reg(cp, 0x28C60 + 0x3C * i + 0x10) >> 2) & 0x1F != 0 {
            if n == 0 {
                let a2 = ctx_reg(cp, 0x28EC0 + 4 * i);
                c0 = (((a2 >> 14) & 0x3FFF) as u64 + 1) << 14 | ((a2 & 0x3FFF) as u64 + 1);
            }
            n += 1;
        }
    }
    (depth, d | c0 << 28 | n << 56)
}

/// Depth/stencil surface from DB_* (None: none bound).
pub(crate) fn depth_surface(cp: &Cp) -> Option<Surface> {
    let zinfo = ctx_reg(cp, 0x28040);
    let sinfo = ctx_reg(cp, 0x28044);
    let format = match (zinfo & 3, sinfo & 1) {
        (0, 0) => return None,
        (1, 0) => vk::Format::D16_UNORM,
        (3, 0) => vk::Format::D32_SFLOAT,
        (_, 1) => vk::Format::D32_SFLOAT_S8_UINT,
        _ => return None,
    };
    // DB_STENCIL_READ_BASE: the stencil plane lives at its own address.
    let stencil_base = ((ctx_reg(cp, 0x2804C) as u64) << 8) | ((ctx_reg(cp, 0x2806C) as u64 & 0xFF) << 40);
    let base = if zinfo & 3 != 0 { ((ctx_reg(cp, 0x28048) as u64) << 8) | ((ctx_reg(cp, 0x28068) as u64 & 0xFF) << 40) } else { stencil_base };
    let size = ctx_reg(cp, 0x2801C);
    let view = ctx_reg(cp, 0x28008);
    let slice = (view & 0x7FF) | (((view >> 11) & 3) << 11);
    let slice_max = ((view >> 13) & 0x7FF) | (((view >> 30) & 3) << 11);
    Some(Surface {
        base,
        format,
        width: (size & 0x3FFF) + 1,
        height: ((size >> 16) & 0x3FFF) + 1,
        layers: slice_max.max(slice) + 1,
        levels: ((zinfo >> 16) & 0xF) + 1,
        ty3d: false,
        mip: (view >> 26) & 0xF,
        slice,
        stencil_base: if sinfo & 1 != 0 { stencil_base } else { 0 },
        // HTILE (Z_INFO.TILE_SURFACE_ENABLE); DB_DEPTH_CLEAR, DB_STENCIL_CLEAR.
        meta: if zinfo & (1 << 29) != 0 { ((ctx_reg(cp, 0x28014) as u64) << 8) | ((ctx_reg(cp, 0x28078) as u64 & 0xFF) << 40) } else { 0 },
        clear: [ctx_reg(cp, 0x2802C), ctx_reg(cp, 0x28028)],
    })
}

/// Intersection of the enabled scissors (screen, window, generic, viewport 0),
/// clipped to the render area. Empty (unprogrammed) scissors are ignored.
fn scissor(cp: &Cp, area: (u32, u32)) -> (i32, i32, i32, i32) {
    let mut s = (0i32, 0i32, area.0 as i32, area.1 as i32);
    let vport = ctx_reg(cp, 0x28A48) & 2 != 0;
    for (tl, br, bits) in [(0x28030, 0x28034, 0xFFFF), (0x28204, 0x28208, 0x7FFF), (0x28240, 0x28244, 0x7FFF), (0x28250, 0x28254, 0x7FFF)] {
        if tl == 0x28250 && !vport {
            continue;
        }
        let (a, b) = (ctx_reg(cp, tl), ctx_reg(cp, br));
        let (x0, y0, x1, y1) = ((a & bits) as i32, ((a >> 16) & bits) as i32, (b & bits) as i32, ((b >> 16) & bits) as i32);
        if x1 > x0 && y1 > y0 {
            s = (s.0.max(x0), s.1.max(y0), s.2.min(x1), s.3.min(y1));
        }
    }
    s
}

/// Pipeline state that is baked into a graphics pipeline.
#[derive(Hash, Clone, Copy, PartialEq, Eq)]
struct GfxKey {
    vs: u64,
    ps: u64,
    rect_gs: Option<u32>,
    colors: [i32; MAX_RT],
    blend: [u32; MAX_RT],
    masks: [u8; MAX_RT],
    depth: i32,
    topology_class: u8,
    depth_clamp: bool,
}

thread_local! {
    /// The last graphics pipeline key this thread hashed, and its hash.
    static LAST_GFX_KEY: std::cell::Cell<Option<(GfxKey, u64)>> = const { std::cell::Cell::new(None) };
}

/// A graphics pipeline's state apart from its stages (`GfxKey` without the
/// translation keys; `gpu::precompile` keeps it with the stages' specs).
#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug)]
pub(crate) struct GfxState {
    pub rect: bool,
    pub colors: [i32; MAX_RT],
    pub blend: [u32; MAX_RT],
    pub masks: [u8; MAX_RT],
    pub depth: i32,
    pub topology_class: u8,
    pub depth_clamp: bool,
}

/// Creates the graphics pipeline of translations made ahead from shader
/// objects (`gpu::precompile`) with `state`, under the key its draws look
/// up: Some(true) when created here, Some(false) when it existed or was in
/// flight, None when creating it failed.
pub(crate) fn precompile_gfx_translations(vs: StageTr, ps: Option<StageTr>, state: &GfxState, keep_ok: bool) -> Option<bool> {
    let key = GfxKey {
        vs: vs.0.key,
        ps: ps.map_or(0, |p| p.0.key),
        rect_gs: state.rect.then(|| rect_gs_mask(vs.0.out.param_mask, ps.map_or(0, |p| p.0.out.in_mask))),
        colors: state.colors,
        blend: state.blend,
        masks: state.masks,
        depth: state.depth,
        topology_class: state.topology_class,
        depth_clamp: state.depth_clamp,
    };
    let bindings: u32 = super::descriptor_bindings(&vs.0.out).chain(ps.iter().flat_map(|p| super::descriptor_bindings(&p.0.out))).map(|b| super::binding_count(b.kind)).sum();
    let push_mode = super::super::tune::push_desc() && bindings <= ctx().max_push;
    let hkey = super::push_key(hash_gfx_key(&key), push_mode);
    if gfx_known(hkey) {
        return Some(false);
    }
    // (One whose compile crashed this driver in an earlier session is left to the draws that
    // need it: `pcache::crashed_skip`.)
    if crate::gpu::pcache::crashed_skip('g', hkey) {
        return None;
    }
    // (Kept alive only when a recent session used it; else made through the
    // store and let go when the store lacks it: `precompile_or_warm`.)
    if !(keep_ok && super::super::pipeuse::recent(hkey)) {
        if crate::gpu::pcache::stored_cheap('g', hkey) {
            super::PRECOMPILE_LIVE[5].fetch_add(1, Ordering::Relaxed);
            return Some(false);
        }
        super::PRECOMPILE_LIVE[4].fetch_add(1, Ordering::Relaxed);
        return match graphics_pipeline(&key, hkey, vs, ps, push_mode) {
            Ok(p) => {
                unsafe {
                    let d = &ctx().device;
                    d.destroy_pipeline(p.pipeline, None);
                    d.destroy_pipeline_layout(p.layout, None);
                    d.destroy_descriptor_set_layout(p.set_layout, None);
                }
                Some(true)
            }
            Err(_) => None,
        };
    }
    super::PRECOMPILE_LIVE[3].fetch_add(1, Ordering::Relaxed);
    let created = std::cell::Cell::new(false);
    gfx_pipe_with(&key, hkey, vs, ps, push_mode, &created).ok().map(|_| created.get())
}

/// A key's hash (the persistent cache file's name, so its hasher stays).
fn hash_gfx_key(key: &GfxKey) -> u64 {
    let mut h = std::collections::hash_map::DefaultHasher::new();
    key.hash(&mut h);
    h.finish()
}

/// Graphics pipelines being created, by cache hash: a thread that needs
/// one waits for it instead of compiling it again (compiles ahead of use).
static GFX_FLIGHTS: parking_lot::Mutex<Option<HashMap<u64, std::sync::Arc<PipeFlight>>>> = parking_lot::Mutex::new(None);
/// Draws that waited for a pipeline in flight: count, nanoseconds, and
/// compiles that saved.
pub static GFX_FLIGHT_WAITS: [AtomicU64; 3] = [const { AtomicU64::new(0) }; 3];
/// Draws whose graphics pipeline was missing and the nanoseconds their
/// command processors spent creating it or waiting for it (first uses).
pub static GFX_FIRST_USES: [AtomicU64; 2] = [const { AtomicU64::new(0) }; 2];

fn gfx_known(hkey: u64) -> bool {
    gfx_pipe(hkey).is_some() || GFX_FLIGHTS.lock().as_ref().is_some_and(|m| m.contains_key(&hkey))
}

/// The registers a draw's baked pipeline state is made of, as a scan of the commands ahead
/// saw them (`gfx_key_ahead`): context registers and the primitive type (a uconfig register).
/// (The October compile-ahead's, removed 2026-11-01 and revived 2026-11-19 for the draws
/// that wait for their pipeline: `gpu::compile::submit_draws`.)
#[derive(Clone, Copy, PartialEq, Eq, Hash)]
pub(crate) struct GfxRegs {
    cb_color_control: u32,
    clip: u32,
    col_format: u32,
    target_mask: u32,
    zinfo: u32,
    sinfo: u32,
    dctl: u32,
    db_render: u32,
    sctl: u32,
    refm: u32,
    refm_bf: u32,
    cb_info: [u32; MAX_RT],
    blend: [u32; MAX_RT],
    prim: u32,
    /// An indexed indirect draw drawn with its start instance as Vulkan's first instance
    /// (`Cp::cpu_args_base_ok`): its vertex stage's start-instance slot, read from
    /// gl_BaseInstance (`shaders::with_base_instance`; 17 of 21 keys made ahead differed from
    /// their draws' without it, 2026-11-19).
    base_inst: Option<u16>,
}

impl GfxRegs {
    /// From a context register reader (`ctx(0x28xxx)`) and VGT_PRIMITIVE_TYPE.
    pub(crate) fn capture(ctx: &dyn Fn(u32) -> u32, prim: u32) -> GfxRegs {
        GfxRegs {
            cb_color_control: ctx(0x28808),
            clip: ctx(0x28810),
            col_format: ctx(0x28714),
            target_mask: ctx(0x28238) & ctx(0x2823C),
            zinfo: ctx(0x28040),
            sinfo: ctx(0x28044),
            dctl: ctx(0x28800),
            db_render: ctx(0x28000),
            sctl: ctx(0x2842C),
            refm: ctx(0x28430),
            refm_bf: ctx(0x28434),
            cb_info: std::array::from_fn(|n| ctx(0x28C60 + 0x3C * n as u32 + 0x10)),
            blend: std::array::from_fn(|n| ctx(0x28780 + 4 * n as u32)),
            prim,
            base_inst: None,
        }
    }

    /// With the vertex stage's start-instance slot of an indirect draw (`base_inst`).
    pub(crate) fn with_base_inst(mut self, slot: Option<u16>) -> GfxRegs {
        self.base_inst = slot;
        self
    }

    pub(crate) fn base_inst(&self) -> Option<u16> {
        self.base_inst
    }

    pub(crate) fn hash64(&self) -> u64 {
        let mut h = std::collections::hash_map::DefaultHasher::new();
        self.hash(&mut h);
        h.finish()
    }
}

/// The pipeline key, its cache hash (with the push-descriptor mode) and the mode of a draw with
/// these registers and stages as `draw_inner` makes them, and whether such a draw waits for
/// its pipeline whatever its targets (memory writes, a depth or stencil clear); None when such
/// a draw records nothing or is not supported. A key that differs at execution only costs a
/// compile (the October check: 0 differences).
fn gfx_key_ahead(g: &GfxRegs, vs: &Prepared, ps: Option<&Prepared>) -> Option<(GfxKey, u64, bool, bool)> {
    let cb_mode = (g.cb_color_control >> 4) & 7;
    if cb_mode > 1 || g.clip & (1 << 22) != 0 {
        return None;
    }
    let (class, rect) = match g.prim & 0x3F {
        1 => (0u8, false),
        2 | 3 => (1, false),
        4..=6 => (2, false),
        7 | 0x11 => (2, true),
        _ => return None,
    };
    let mrt_mask = ps.map(|p| p.tr.out.mrt_mask).unwrap_or(0);
    let mut colors = [vk::Format::UNDEFINED.as_raw(); MAX_RT];
    let mut blend = [0u32; MAX_RT];
    let mut masks = [0u8; MAX_RT];
    let mut any_color = false;
    if cb_mode == 1 {
        for i in 0..MAX_RT {
            let m = (g.target_mask >> (4 * i)) & 0xF;
            if m == 0 || mrt_mask & (1 << i) == 0 || (g.col_format >> (4 * i)) & 0xF == 0 {
                continue;
            }
            let info = g.cb_info[i];
            let fmt = (info >> 2) & 0x1F;
            if fmt == 0 {
                continue;
            }
            colors[i] = images::cb_format(fmt, (info >> 8) & 7, (info >> 11) & 3)?.as_raw();
            blend[i] = g.blend[i];
            masks[i] = m as u8;
            any_color = true;
        }
    }
    let dformat = match (g.zinfo & 3, g.sinfo & 1) {
        (0, 0) => None,
        (1, 0) => Some(vk::Format::D16_UNORM),
        (3, 0) => Some(vk::Format::D32_SFLOAT),
        (_, 1) => Some(vk::Format::D32_SFLOAT_S8_UINT),
        _ => None,
    };
    let (has_z, has_s) = match dformat {
        Some(f) => (images::aspects(f).contains(vk::ImageAspectFlags::DEPTH) && g.zinfo & 3 != 0, images::has_stencil(f)),
        None => (false, false),
    };
    let z_test = has_z && g.dctl & 2 != 0;
    let z_write = z_test && g.dctl & 4 != 0;
    let s_test = has_s && g.dctl & 1 != 0;
    let s_write = s_test && (g.refm | g.refm_bf) & 0x00FF_0000 != 0 && g.sctl & 0xFF_FFFF != 0;
    let clear_z = has_z && g.db_render & 1 != 0;
    let clear_s = has_s && g.db_render & 2 != 0;
    let depth = if z_test || s_test || clear_z || clear_s { dformat } else { None };
    let mem_writes = vs.tr.plan.resources.iter().any(|r| r.written) || ps.is_some_and(|p| p.tr.plan.resources.iter().any(|r| r.written));
    if !any_color && !z_write && !s_write && !clear_z && !clear_s && !mem_writes {
        return None;
    }
    if !any_color && depth.is_none() && !super::super::tune::attachless_draws() {
        return None;
    }
    let key = GfxKey {
        vs: vs.tr.key,
        ps: ps.map(|p| p.tr.key).unwrap_or(0),
        rect_gs: rect.then(|| rect_gs_mask(vs.tr.out.param_mask, ps.map_or(0, |p| p.tr.out.in_mask))),
        colors,
        blend,
        masks,
        depth: depth.map(|d| d.as_raw()).unwrap_or(vk::Format::UNDEFINED.as_raw()),
        topology_class: class,
        depth_clamp: g.clip & (3 << 26) != 0,
    };
    let push_mode = match ps {
        Some(p) => super::push_fits(&[vs, p]),
        None => super::push_fits(&[vs]),
    };
    Some((key, super::push_key(hash_gfx_key(&key), push_mode), push_mode, mem_writes || clear_z || clear_s))
}

/// A draw a scan ahead saw (`gpu::compile` workers, its stages prepared from the scan's
/// registers): its pipeline queued for the background graphics workers when missing, after the
/// pipelines the store holds and ahead of the left-out draws' compiles (`async_gfx_worker`).
/// Some(true): queued; Some(false): made, in flight or queued already; None: such a draw
/// records nothing.
pub(crate) fn queue_gfx_ahead(g: &GfxRegs, vs: &Prepared, ps: Option<&Prepared>) -> Option<bool> {
    let (key, hkey, push_mode, waits) = gfx_key_ahead(g, vs, ps)?;
    // (`DS_COMPILE_TRACE=1`: the keys made ahead by program pair, with their stages' specs, for
    // a first use under another key: `ahead_gfx_diff`.)
    if compile_trace() {
        let mut m = AHEAD_GFX.lock();
        let v = m.get_or_insert_with(HashMap::new).entry((vs.addr, ps.map_or(0, |p| p.addr))).or_default();
        if v.len() < 8 {
            v.push((key, vs.tr.spec.clone(), ps.map(|p| p.tr.spec.clone())));
        }
    }
    if gfx_known(hkey) {
        return Some(false);
    }
    Some(async_gfx_queue(&key, hkey, vs, ps, push_mode, Some(waits)))
}

/// `DS_COMPILE_TRACE`: the pipeline keys scans ahead made, by (vertex, pixel program address),
/// with the stages' specs (at most 8 a pair).
#[allow(clippy::type_complexity)]
static AHEAD_GFX: parking_lot::Mutex<Option<HashMap<(u64, u64), Vec<(GfxKey, rdna::translate::Spec, Option<rdna::translate::Spec>)>>>> = parking_lot::Mutex::new(None);

/// What differs between a first use's key and the closest key a scan made ahead for the same
/// programs (None: none made for them, or one equal).
fn ahead_gfx_diff(key: &GfxKey, vs: &Prepared, ps: Option<&Prepared>) -> Option<String> {
    let g = AHEAD_GFX.lock();
    let v = g.as_ref()?.get(&(vs.addr, ps.map_or(0, |p| p.addr)))?;
    if v.iter().any(|(k, _, _)| k == key) {
        return None;
    }
    let diff = |k: &GfxKey, s: &rdna::translate::Spec, p: &Option<rdna::translate::Spec>| {
        let mut d: Vec<String> = Vec::new();
        if k.vs != key.vs {
            d.push(format!("vs spec: {}", super::spec_diff(s, &vs.tr.spec)));
        }
        if k.ps != key.ps {
            match (p, ps) {
                (Some(p), Some(q)) => d.push(format!("ps spec: {}", super::spec_diff(p, &q.tr.spec))),
                _ => d.push("ps presence".into()),
            }
        }
        if k.rect_gs != key.rect_gs {
            d.push(format!("rect_gs {:?} -> {:?}", k.rect_gs, key.rect_gs));
        }
        if k.colors != key.colors {
            d.push(format!("colors {:?} -> {:?}", k.colors, key.colors));
        }
        if k.blend != key.blend {
            d.push(format!("blend {:x?} -> {:x?}", k.blend, key.blend));
        }
        if k.masks != key.masks {
            d.push(format!("masks {:?} -> {:?}", k.masks, key.masks));
        }
        if k.depth != key.depth {
            d.push(format!("depth {} -> {}", k.depth, key.depth));
        }
        if k.topology_class != key.topology_class {
            d.push(format!("topology {} -> {}", k.topology_class, key.topology_class));
        }
        if k.depth_clamp != key.depth_clamp {
            d.push(format!("depth clamp {} -> {}", k.depth_clamp, key.depth_clamp));
        }
        d
    };
    let best = v.iter().map(|(k, s, p)| diff(k, s, p)).min_by_key(|d| d.len())?;
    Some(format!("{} made ahead for these programs, the closest differs in: {}", v.len(), best.join("; ")))
}

/// The graphics pipeline of `key` (created on first use; one in flight on
/// another thread is waited for).
/// A stage of a graphics pipeline: its translation and program address.
type StageTr<'a> = (&'a super::super::shaders::Translated, u64);

/// The barycentric geometry shader's generator version (`rdna::translate::gs::bary_gs`, in
/// the pipelines' fingerprints).
const BARY_GS_VERSION: u32 = 1;

/// The barycentric geometry shader a pipeline needs (`rdna::translate::Spec::bary_gs`): a
/// triangle draw whose pixel stage reads per vertex without fragment barycentrics; with
/// whether it passes a batched draw's varying.
fn bary_gs_of<'a>(key: &GfxKey, vs: StageTr, ps: Option<StageTr<'a>>) -> Option<(&'a rdna::translate::gs::BaryIo, bool)> {
    if key.rect_gs.is_some() || key.topology_class != 2 {
        return None;
    }
    let p = ps?;
    let io = p.0.out.bary_gs.as_ref()?;
    Some((io, vs.0.out.batched && p.0.out.in_mask & (1 << rdna::translate::BATCH_VARYING) != 0))
}

/// A rect-list draw's geometry shader key: the vertex stage's outputs and
/// the pixel stage's inputs (`rdna::translate::gs::rect_list_gs` passes the
/// first through and writes zeros at the rest of the second: a pixel input
/// the vertex stage does not write had no output before it).
fn rect_gs_mask(vs_mask: u32, ps_inputs: u32) -> u32 {
    vs_mask | ps_inputs
}

/// Graphics pipelines compiled in the background (`tune async_gfx`): the
/// keys queued or being made, those whose creation failed there (their draws
/// wait as before and report the error), and the jobs (pipelines the store
/// holds first: made in milliseconds), after those of scans ahead (`queue_gfx_ahead`: served
/// first, in the order the scans met their draws).
#[derive(Default)]
struct AsyncGfx {
    pending: std::collections::HashSet<u64>,
    failed: std::collections::HashSet<u64>,
    jobs: std::collections::VecDeque<GfxJob>,
    ahead: std::collections::VecDeque<GfxJob>,
}

struct GfxJob {
    key: GfxKey,
    hkey: u64,
    vs: (std::sync::Arc<super::super::shaders::Translated>, u64),
    ps: Option<(std::sync::Arc<super::super::shaders::Translated>, u64)>,
    push_mode: bool,
    stored: bool,
    queued: std::time::Instant,
}

static ASYNC_GFX: parking_lot::Mutex<Option<AsyncGfx>> = parking_lot::Mutex::new(None);

/// No pipeline queued for the background graphics workers or being made by them (the
/// self-play's end waits for it: its statistics leave out jobs dropped as made meanwhile).
pub fn async_gfx_idle() -> bool {
    ASYNC_GFX.lock().as_ref().is_none_or(|s| s.pending.is_empty() && s.jobs.is_empty() && s.ahead.is_empty())
}
static ASYNC_GFX_CV: parking_lot::Condvar = parking_lot::Condvar::new();
static ASYNC_GFX_WORKERS: std::sync::atomic::AtomicUsize = std::sync::atomic::AtomicUsize::new(0);
/// [draws left out, pipelines queued, of them from the store, made, their
/// creation ns, ns from queued to made, the longest of those ns, failed].
pub static ASYNC_GFX_STATS: [AtomicU64; 8] = [const { AtomicU64::new(0) }; 8];
/// Draws that waited for their pipeline although `async_gfx` is on, by why:
/// [memory writes, depth or stencil clear, a target not drawn into in 3 of
/// the 4 frames before, (measuring LOD statistics: left out, its textures
/// carried), a GPU-resolved pair's draw, a mode that records every draw
/// (sync, capture, verify, replay)].
pub static ASYNC_GFX_WAITED: [AtomicU64; 6] = [const { AtomicU64::new(0) }; 6];

/// The time first draws spent making stored graphics pipelines themselves this frame
/// (`tune async_gfx_inline`): (frame, ns); and [pipelines made so, their ns, stored ones left
/// out once the frame's budget was spent].
static INLINE_SPENT: parking_lot::Mutex<(u64, u64)> = parking_lot::Mutex::new((u64::MAX, 0));
pub static INLINE_STATS: [AtomicU64; 3] = [const { AtomicU64::new(0) }; 3];

/// Whether a draw makes a stored pipeline itself (within the frame's budget).
fn inline_ok(frame: u64) -> bool {
    let budget_ms = super::super::tune::async_gfx_inline();
    if budget_ms == 0 {
        return false;
    }
    let g = INLINE_SPENT.lock();
    g.0 != frame || g.1 < budget_ms * 1_000_000
}

fn inline_spent(frame: u64, ns: u64) {
    let mut g = INLINE_SPENT.lock();
    if g.0 != frame {
        *g = (frame, 0);
    }
    g.1 += ns;
    INLINE_STATS[0].fetch_add(1, Ordering::Relaxed);
    INLINE_STATS[1].fetch_add(ns, Ordering::Relaxed);
}

/// Leaves a draw out while its pipeline is made in the background: true when
/// it is (queued now or before), false when the draw must make or wait for
/// it itself (its creation failed in the background, or it is made now).
fn async_gfx_submit(key: &GfxKey, hkey: u64, vs: &Prepared, ps: Option<&Prepared>, push_mode: bool) -> bool {
    async_gfx_queue(key, hkey, vs, ps, push_mode, None)
}

/// `async_gfx_submit` for a draw left out (`ahead` None), or for one a scan ahead saw
/// (`queue_gfx_ahead`: Some(whether such a draw always waits for it), logged; its job goes to
/// the scans' queue). Returns whether the key is queued (now or before); for a scan's draw,
/// whether it was queued now.
fn async_gfx_queue(key: &GfxKey, hkey: u64, vs: &Prepared, ps: Option<&Prepared>, push_mode: bool, ahead: Option<bool>) -> bool {
    let stored = crate::gpu::pcache::stored_cheap('g', hkey);
    let mut queued_now = false;
    {
        let mut g = ASYNC_GFX.lock();
        let st = g.get_or_insert_with(AsyncGfx::default);
        if st.failed.contains(&hkey) {
            return false;
        }
        if !st.pending.contains(&hkey) {
            // (Made since the draw looked.)
            if gfx_pipe(hkey).is_some() {
                return false;
            }
            st.pending.insert(hkey);
            let job = GfxJob { key: *key, hkey, vs: (vs.tr.0.clone(), vs.addr), ps: ps.map(|p| (p.tr.0.clone(), p.addr)), push_mode, stored, queued: std::time::Instant::now() };
            if ahead.is_some() {
                st.ahead.push_back(job);
            } else if stored {
                st.jobs.push_front(job);
            } else {
                st.jobs.push_back(job);
            }
            queued_now = true;
            ASYNC_GFX_STATS[1].fetch_add(1, Ordering::Relaxed);
            if stored {
                ASYNC_GFX_STATS[2].fetch_add(1, Ordering::Relaxed);
            }
            if compile_trace() {
                match ahead {
                    None => crate::info!("first use: vs {:#x} ps {:#x} key {hkey:#x} left out while its pipeline is made in the background{}", vs.addr, ps.map_or(0, |p| p.addr), if stored { " (from the store)" } else { "" }),
                    Some(w) => crate::info!("compile ahead: vs {:#x} ps {:#x} key {hkey:#x} queued{}{}", vs.addr, ps.map_or(0, |p| p.addr), if w { " first (its draws wait)" } else { "" }, if stored { " (from the store)" } else { "" }),
                }
            }
        }
    }
    ensure_async_gfx_workers();
    ASYNC_GFX_CV.notify_one();
    if ahead.is_some() {
        return queued_now;
    }
    ASYNC_GFX_STATS[0].fetch_add(1, Ordering::Relaxed);
    true
}

thread_local! {
    /// This draw makes a stored pipeline itself (`inline_ok`): its time goes to the budget.
    static INLINE_MARK: std::cell::Cell<bool> = const { std::cell::Cell::new(false) };
}

/// Sets this thread's inline mark (`on`), or takes it (false: returns and clears it).
fn inline_mark(on: bool) -> bool {
    INLINE_MARK.with(|m| if on { m.set(true); true } else { m.replace(false) })
}

fn ensure_async_gfx_workers() {
    let want = super::super::tune::async_gfx_workers();
    while ASYNC_GFX_WORKERS.load(Ordering::Acquire) < want {
        let n = ASYNC_GFX_WORKERS.fetch_add(1, Ordering::AcqRel);
        if n >= want {
            ASYNC_GFX_WORKERS.fetch_sub(1, Ordering::AcqRel);
            break;
        }
        let name = format!("gpu-gfxasync{n}");
        let r = crate::thread::spawn_host(
            &name.clone(),
            8 << 20,
            Box::new(move || {
                crate::log::set_thread_name(&name);
                crate::exception::arm_watch_current_thread(false);
                async_gfx_worker();
            }),
        );
        if let Err(e) = r {
            crate::warn!("gpu: no background graphics compile worker ({e})");
            ASYNC_GFX_WORKERS.fetch_sub(1, Ordering::AcqRel);
            break;
        }
    }
}

fn async_gfx_worker() {
    loop {
        let job = {
            let mut g = ASYNC_GFX.lock();
            loop {
                // (Pipelines the store holds first (milliseconds each: left-out draws come back
                // at once), then the scans' jobs, then the left-out draws' compiles.)
                if let Some(j) = g.as_mut().and_then(|s| if s.jobs.front().is_some_and(|j| j.stored) { s.jobs.pop_front() } else { s.ahead.pop_front().or_else(|| s.jobs.pop_front()) }) {
                    break j;
                }
                ASYNC_GFX_CV.wait(&mut g);
            }
        };
        // Made meanwhile, or being made by a draw that waits for it: not waited for here.
        if gfx_known(job.hkey) {
            if let Some(st) = ASYNC_GFX.lock().as_mut() {
                st.pending.remove(&job.hkey);
            }
            continue;
        }
        let t = std::time::Instant::now();
        let created = std::cell::Cell::new(false);
        // A panic must not leave the key pending (its draws left out forever).
        let r = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| gfx_pipe_with(&job.key, job.hkey, (&*job.vs.0, job.vs.1), job.ps.as_ref().map(|p| (&*p.0, p.1)), job.push_mode, &created))).unwrap_or_else(|_| Err("pipeline creation panicked".to_string()));
        let mut g = ASYNC_GFX.lock();
        let st = g.get_or_insert_with(AsyncGfx::default);
        st.pending.remove(&job.hkey);
        match r {
            Ok(_) => {
                ASYNC_GFX_STATS[3].fetch_add(1, Ordering::Relaxed);
                ASYNC_GFX_STATS[4].fetch_add(t.elapsed().as_nanos() as u64, Ordering::Relaxed);
                let wait = job.queued.elapsed().as_nanos() as u64;
                ASYNC_GFX_STATS[5].fetch_add(wait, Ordering::Relaxed);
                ASYNC_GFX_STATS[6].fetch_max(wait, Ordering::Relaxed);
            }
            Err(e) => {
                st.failed.insert(job.hkey);
                ASYNC_GFX_STATS[7].fetch_add(1, Ordering::Relaxed);
                crate::warn!("gpu: background graphics pipeline for vs {:#x} ps {:#x} failed ({e}); its draws make it themselves", job.vs.1, job.ps.as_ref().map_or(0, |p| p.1));
            }
        }
        let _ = job.stored;
    }
}

fn gfx_pipe_with(key: &GfxKey, hkey: u64, vs: StageTr, ps: Option<StageTr>, push_mode: bool, created: &std::cell::Cell<bool>) -> Result<Pipe, String> {
    if let Some(x) = gfx_pipe(hkey) {
        return Ok(x);
    }
    let (mine, flight) = {
        let mut g = GFX_FLIGHTS.lock();
        let m = g.get_or_insert_with(HashMap::new);
        match m.get(&hkey) {
            Some(f) => (false, f.clone()),
            None => {
                let f = std::sync::Arc::new(PipeFlight::new());
                m.insert(hkey, f.clone());
                (true, f)
            }
        }
    };
    if !mine {
        let t = std::time::Instant::now();
        flight.waited();
        let r = {
            let mut d = flight.done.lock();
            while d.is_none() {
                flight.cv.wait(&mut d);
            }
            d.clone().unwrap()
        };
        GFX_FLIGHT_WAITS[0].fetch_add(1, Ordering::Relaxed);
        GFX_FLIGHT_WAITS[1].fetch_add(t.elapsed().as_nanos() as u64, Ordering::Relaxed);
        if r.is_ok() {
            GFX_FLIGHT_WAITS[2].fetch_add(1, Ordering::Relaxed);
            return r;
        }
        return gfx_pipe_create(key, hkey, vs, ps, push_mode, created);
    }
    let stored = gfx_pipe(hkey);
    // A panic must not leave the threads waiting for it blocked.
    let r = match stored {
        Some(x) => Ok(x),
        None => std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| gfx_pipe_create(key, hkey, vs, ps, push_mode, created))).unwrap_or_else(|_| Err("pipeline creation panicked".to_string())),
    };
    *flight.done.lock() = Some(r.clone());
    flight.cv.notify_all();
    if let Some(m) = GFX_FLIGHTS.lock().as_mut() {
        m.remove(&hkey);
    }
    r
}

fn gfx_pipe_create(key: &GfxKey, hkey: u64, vs: StageTr, ps: Option<StageTr>, push_mode: bool, created: &std::cell::Cell<bool>) -> Result<Pipe, String> {
    created.set(true);
    let t0 = std::time::Instant::now();
    let p = graphics_pipeline(key, hkey, vs, ps, push_mode)?;
    // What a later session makes ahead when the title creates these
    // shaders' objects (`gpu::precompile`).
    if super::super::tune::precompile() > 0 {
        let state = GfxState { rect: key.rect_gs.is_some(), colors: key.colors, blend: key.blend, masks: key.masks, depth: key.depth, topology_class: key.topology_class, depth_clamp: key.depth_clamp };
        super::super::precompile::learn_gfx(vs, ps, &state);
    }
    super::PIPES_CREATED[1].fetch_add(1, Ordering::Relaxed);
    super::PIPES_NS[1].fetch_add(t0.elapsed().as_nanos() as u64, Ordering::Relaxed);
    // Pipelines whose shader pair an earlier pipeline had (other render
    // state only: what pipeline libraries would link without compiling the
    // shaders again).
    if !GFX_PAIRS.lock().insert((key.vs, key.ps, key.rect_gs, key.depth_clamp)) {
        GFX_REPEAT_PAIRS[0].fetch_add(1, Ordering::Relaxed);
        GFX_REPEAT_PAIRS[1].fetch_add(t0.elapsed().as_nanos() as u64, Ordering::Relaxed);
    }
    if p.push {
        super::PIPES_PUSH[1].fetch_add(1, Ordering::Relaxed);
    }
    super::super::hitch::note_since(super::super::hitch::Kind::Pipeline, t0, || format!("vs {:#x} ps {:#x}", vs.1, ps.map_or(0, |p| p.1)));
    let ms = t0.elapsed().as_millis();
    if ms >= 200 {
        crate::info!("gpu: graphics pipeline for vs {:#x} ps {:#x} took {ms} ms", vs.1, ps.map_or(0, |p| p.1));
    }
    Ok(keep_gfx_pipe(hkey, p))
}

fn set_layout_bindings(stages: &[(&super::super::shaders::Translated, vk::ShaderStageFlags)]) -> Vec<vk::DescriptorSetLayoutBinding<'static>> {
    let mut v = Vec::new();
    for (tr, flags) in stages {
        for b in super::descriptor_bindings(&tr.out) {
            v.push(vk::DescriptorSetLayoutBinding::default().binding(b.binding).descriptor_type(descriptor_type(b.kind)).descriptor_count(super::binding_count(b.kind)).stage_flags(*flags));
        }
    }
    v
}

/// Creates a graphics pipeline for dynamic rendering (`push_mode`: a push
/// descriptor layout if it fits, `push_layout`), through the pipeline cache
/// file of `hkey`.
/// Shader pairs graphics pipelines were created for (`GFX_REPEAT_PAIRS`).
static GFX_PAIRS: parking_lot::Mutex<std::collections::BTreeSet<(u64, u64, Option<u32>, bool)>> = parking_lot::Mutex::new(std::collections::BTreeSet::new());
/// Graphics pipelines created for a shader pair already seen with other
/// render state, and their nanoseconds (status).
pub static GFX_REPEAT_PAIRS: [std::sync::atomic::AtomicU64; 2] = [const { std::sync::atomic::AtomicU64::new(0) }; 2];

/// Render area of instances without attachments (`tune attachless_draws`).
const ATTACHLESS_AREA: u32 = 16384;
/// Draws recorded without attachments (status).
pub static ATTACHLESS: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);

fn graphics_pipeline(key: &GfxKey, hkey: u64, vs: StageTr, ps: Option<StageTr>, push_mode: bool) -> Result<Pipe, String> {
    let c = ctx();
    // (A pipeline whose compile crashed this driver: refused before its layouts and modules are
    // made, which a refusal inside `pcache::create_with` made and leaked at every draw.)
    if crate::gpu::pcache::crashed_skip('g', hkey) {
        return Err(format!("g{hkey:016x} left out: its compile crashed this graphics driver before (crashed.txt)"));
    }
    unsafe {
        let mut stages_p = vec![(vs.0, vk::ShaderStageFlags::VERTEX)];
        if let Some(p) = ps {
            stages_p.push((p.0, vk::ShaderStageFlags::FRAGMENT));
        }
        let bindings = set_layout_bindings(&stages_p);
        let (push, flags) = super::push_layout(&bindings, push_mode);
        let set_layout = c.device.create_descriptor_set_layout(&vk::DescriptorSetLayoutCreateInfo::default().flags(flags).bindings(&bindings), None).map_err(|e| format!("set layout: {e}"))?;
        let layouts = [set_layout];
        let ranges = super::sgpr_pc_ranges(vk::ShaderStageFlags::VERTEX, vs.0.out.sgpr_pc.is_some() || ps.is_some_and(|p| p.0.out.sgpr_pc.is_some()));
        let layout = c.device.create_pipeline_layout(&vk::PipelineLayoutCreateInfo::default().set_layouts(&layouts).push_constant_ranges(ranges), None).map_err(|e| format!("pipeline layout: {e}"))?;
        let module = |code: &[u32]| {
            let t = std::time::Instant::now();
            let m = c.device.create_shader_module(&vk::ShaderModuleCreateInfo::default().code(&super::module_code(code)), None).map_err(|e| format!("shader module: {e}"));
            crate::gpu::pcache::note_module(t.elapsed().as_nanos() as u64);
            m
        };
        let g = super::super::vk::creating(|| format!("vs {:#x}", vs.1));
        let mut modules = vec![(vk::ShaderStageFlags::VERTEX, module(&vs.0.out.spirv)?)];
        drop(g);
        super::super::vk::name_object(modules[0].1, &format!("vs {:#x}", vs.1));
        if let Some(mask) = key.rect_gs {
            modules.push((vk::ShaderStageFlags::GEOMETRY, module(&rdna::translate::gs::rect_list_gs(vs.0.out.param_mask, mask))?));
            super::super::vk::name_object(modules[modules.len() - 1].1, &format!("rect-list gs {mask:#x}"));
        } else if let Some((io, batch)) = bary_gs_of(key, vs, ps) {
            let p = ps.expect("a pixel stage");
            modules.push((vk::ShaderStageFlags::GEOMETRY, module(&rdna::translate::gs::bary_gs(vs.0.out.param_mask, p.0.out.in_mask, io, batch))?));
            super::super::vk::name_object(modules[modules.len() - 1].1, &format!("barycentric gs for ps {:#x}", p.1));
        }
        if let Some(p) = ps {
            let from_gs = if key.rect_gs.is_none() { bary_gs_of(key, vs, ps).map_or(0, |(io, _)| io.added()) } else { 0 };
            // (A batched vertex stage writes the batch varying besides its parameters, which
            // `param_mask` leaves out by design.)
            let batch = if vs.0.out.batched { 1u32 << rdna::translate::BATCH_VARYING } else { 0 };
            let missing = p.0.out.in_mask & !vs.0.out.param_mask & !from_gs & !batch;
            if key.rect_gs.is_none() && missing != 0 {
                static N: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);
                if N.fetch_add(1, std::sync::atomic::Ordering::Relaxed) < 40 {
                    crate::info!("gfx pipeline: ps {:#x} reads locations {missing:#x} vs {:#x} does not write (vs writes {:#x}, ps reads {:#x})", p.1, vs.1, vs.0.out.param_mask, p.0.out.in_mask);
                }
            }
            let g = super::super::vk::creating(|| format!("ps {:#x}", p.1));
            modules.push((vk::ShaderStageFlags::FRAGMENT, module(&p.0.out.spirv)?));
            drop(g);
            super::super::vk::name_object(modules[modules.len() - 1].1, &format!("ps {:#x}", p.1));
        }
        let c32 = ctx();
        let mut req32: Vec<vk::PipelineShaderStageRequiredSubgroupSizeCreateInfo> = modules.iter().map(|_| vk::PipelineShaderStageRequiredSubgroupSizeCreateInfo::default().required_subgroup_size(32)).collect();
        let stages: Vec<vk::PipelineShaderStageCreateInfo> = modules
            .iter()
            .zip(req32.iter_mut())
            .map(|((s, m), r)| {
                let st = vk::PipelineShaderStageCreateInfo::default().stage(*s).module(*m).name(c"main");
                if c32.wants_subgroup32(*s) { st.push_next(r) } else { st }
            })
            .collect();
        let vi = vk::PipelineVertexInputStateCreateInfo::default();
        let topo = match key.topology_class {
            0 => vk::PrimitiveTopology::POINT_LIST,
            1 => vk::PrimitiveTopology::LINE_LIST,
            _ => vk::PrimitiveTopology::TRIANGLE_LIST,
        };
        let ia = vk::PipelineInputAssemblyStateCreateInfo::default().topology(topo);
        let vps = vk::PipelineViewportStateCreateInfo::default().viewport_count(1).scissor_count(1);
        let rs = vk::PipelineRasterizationStateCreateInfo::default().depth_clamp_enable(key.depth_clamp).polygon_mode(vk::PolygonMode::FILL).line_width(1.0);
        let ms = vk::PipelineMultisampleStateCreateInfo::default().rasterization_samples(vk::SampleCountFlags::TYPE_1);
        let ds = vk::PipelineDepthStencilStateCreateInfo::default();
        let n = key.colors.iter().rposition(|&f| f != vk::Format::UNDEFINED.as_raw()).map(|i| i + 1).unwrap_or(0);
        let formats: Vec<vk::Format> = key.colors[..n].iter().map(|&f| vk::Format::from_raw(f)).collect();
        let atts: Vec<vk::PipelineColorBlendAttachmentState> = (0..n)
            .map(|i| {
                let b = key.blend[i];
                let enable = b & (1 << 30) != 0 && formats[i] != vk::Format::UNDEFINED;
                let sep = b & (1 << 29) != 0;
                let (cs, cf, cd) = (b & 0x1F, (b >> 5) & 7, (b >> 8) & 0x1F);
                let (asrc, af, ad) = if sep { ((b >> 16) & 0x1F, (b >> 21) & 7, (b >> 24) & 0x1F) } else { (cs, cf, cd) };
                vk::PipelineColorBlendAttachmentState::default()
                    .blend_enable(enable)
                    .src_color_blend_factor(blend_factor(cs))
                    .dst_color_blend_factor(blend_factor(cd))
                    .color_blend_op(blend_op(cf))
                    .src_alpha_blend_factor(blend_factor(asrc))
                    .dst_alpha_blend_factor(blend_factor(ad))
                    .alpha_blend_op(blend_op(af))
                    .color_write_mask(vk::ColorComponentFlags::from_raw(key.masks[i] as u32))
            })
            .collect();
        let cb = vk::PipelineColorBlendStateCreateInfo::default().attachments(&atts);
        let dyns = [
            vk::DynamicState::VIEWPORT,
            vk::DynamicState::SCISSOR,
            vk::DynamicState::LINE_WIDTH,
            vk::DynamicState::DEPTH_BIAS,
            vk::DynamicState::BLEND_CONSTANTS,
            vk::DynamicState::STENCIL_COMPARE_MASK,
            vk::DynamicState::STENCIL_WRITE_MASK,
            vk::DynamicState::STENCIL_REFERENCE,
            vk::DynamicState::CULL_MODE,
            vk::DynamicState::FRONT_FACE,
            vk::DynamicState::PRIMITIVE_TOPOLOGY,
            vk::DynamicState::DEPTH_TEST_ENABLE,
            vk::DynamicState::DEPTH_WRITE_ENABLE,
            vk::DynamicState::DEPTH_COMPARE_OP,
            vk::DynamicState::DEPTH_BOUNDS_TEST_ENABLE,
            vk::DynamicState::STENCIL_TEST_ENABLE,
            vk::DynamicState::STENCIL_OP,
            vk::DynamicState::DEPTH_BIAS_ENABLE,
            vk::DynamicState::PRIMITIVE_RESTART_ENABLE,
            vk::DynamicState::RASTERIZER_DISCARD_ENABLE,
        ];
        let dy = vk::PipelineDynamicStateCreateInfo::default().dynamic_states(&dyns);
        let depth = vk::Format::from_raw(key.depth);
        let mut rinfo = vk::PipelineRenderingCreateInfo::default().color_attachment_formats(&formats).depth_attachment_format(depth);
        if images::has_stencil(depth) {
            rinfo = rinfo.stencil_attachment_format(depth);
        }
        let gpci = vk::GraphicsPipelineCreateInfo::default()
            .stages(&stages)
            .vertex_input_state(&vi)
            .input_assembly_state(&ia)
            .viewport_state(&vps)
            .rasterization_state(&rs)
            .multisample_state(&ms)
            .depth_stencil_state(&ds)
            .color_blend_state(&cb)
            .dynamic_state(&dy)
            .layout(layout)
            .flags(crate::gpu::pstats::flags())
            .push_next(&mut rinfo);
        // (Through the pipeline store: its cache file, or its binaries,
        // `pcache::create_with`.)
        // (The code the binaries were made from: the stages' SPIR-V and what
        // the rect-list geometry shader is made from.)
        let fp = {
            let gs = [key.rect_gs.unwrap_or(u32::MAX), vs.0.out.param_mask];
            let ps_code: &[u32] = ps.map_or(&[], |p| p.0.out.spirv.as_slice());
            // (The barycentric geometry shader's version and inputs: pipelines
            // without one keep their fingerprints.)
            match bary_gs_of(key, vs, ps) {
                Some((_, batch)) => crate::gpu::pcache::translated_fingerprint(&[&vs.0.out.spirv, &gs, ps_code, &[BARY_GS_VERSION, batch as u32]]),
                None => crate::gpu::pcache::translated_fingerprint(&[&vs.0.out.spirv, &gs, ps_code]),
            }
        };
        let _crumb = crate::gpu::pcache::Crumb::set(
            || format!("graphics pipeline g{hkey:016x}: vs {:#x} (translation {:016x}), ps {}, rect-list gs {:?}, barycentric gs {}", vs.1, vs.0.key, ps.map_or("none".to_string(), |p| format!("{:#x} (translation {:016x})", p.1, p.0.key)), key.rect_gs, bary_gs_of(key, vs, ps).is_some()),
            || {
                // (The modules as the driver gets them: `module_code`'s declaration included.)
                let mut v = vec![("vs", super::module_code(&vs.0.out.spirv).into_owned())];
                if let Some(p) = ps {
                    v.push(("ps", super::module_code(&p.0.out.spirv).into_owned()));
                }
                v
            },
        );
        let r = crate::gpu::pcache::create_with('g', hkey, fp, |via| {
            let mut gpci = gpci;
            let mut f2 = vk::PipelineCreateFlags2CreateInfoKHR::default().flags(vk::PipelineCreateFlags2KHR::from_raw(gpci.flags.as_raw() as u64 | crate::gpu::vk::pbin::CAPTURE_DATA));
            if via.capture {
                gpci = gpci.push_next(&mut f2);
            }
            let bi;
            if let Some(b) = via.binaries {
                bi = crate::gpu::vk::pbin::BinaryInfo { s_type: crate::gpu::vk::pbin::ST_BINARY_INFO, p_next: gpci.p_next, binary_count: b.len() as u32, p_binaries: b.as_ptr() };
                gpci.p_next = &bi as *const crate::gpu::vk::pbin::BinaryInfo as *const std::ffi::c_void;
            }
            c.device.create_graphics_pipelines(via.cache, &[gpci], None).map(|v| v[0]).map_err(|(_, e)| format!("graphics pipeline: {e}"))
        });
        for (_, m) in modules {
            c.device.destroy_shader_module(m, None);
        }
        let pipeline = match r {
            Ok(p) => p,
            Err(e) => {
                c.device.destroy_pipeline_layout(layout, None);
                c.device.destroy_descriptor_set_layout(set_layout, None);
                return Err(e);
            }
        };
        crate::gpu::pstats::report(pipeline, &format!("vs {:#x} ps {:#x}", vs.1, ps.map_or(0, |p| p.1)));
        Ok(Pipe { pipeline, layout, set_layout, push })
    }
}

/// Pixel shaders whose draws are limited to one pixel (control `tiny`), and
/// whether there are any (read by every draw without the lock).
pub static TINY_SCISSOR: parking_lot::Mutex<Vec<u64>> = parking_lot::Mutex::new(Vec::new());
pub static TINY_ANY: std::sync::atomic::AtomicBool = std::sync::atomic::AtomicBool::new(false);

impl Exec {
    /// Executes a draw with the graphics state of `cp`; returns what
    /// happened (frame captures).
    pub fn draw(&mut self, cp: &Cp, args: DrawArgs) -> std::borrow::Cow<'static, str> {
        if super::ballast_progs().is_empty() {
            super::ballast_cp(self.ballast_kind, cp.ballast_part());
        }
        if DEVICE_LOST.load(Ordering::Relaxed) {
            DRAW_SKIPPED.fetch_add(1, Ordering::Relaxed);
            return "skipped (device lost)".into();
        }
        if super::super::tune::skip_cmds() & 1 != 0 {
            DRAW_SKIPPED.fetch_add(1, Ordering::Relaxed);
            return "skipped (`tune skip_cmds`)".into();
        }
        // Diagnosis (`DS_SKIP_DEPTH=<hex depth base>`): draws into that depth target left out
        // (Forlorn Outpost's odd-flip culls: the occluders' 1024x512 depth, 0x10f6470000).
        if let Some(d) = skip_depth() {
            if depth_surface(cp).is_some_and(|s| s.base == d) {
                DRAW_SKIPPED.fetch_add(1, Ordering::Relaxed);
                return "skipped (`DS_SKIP_DEPTH`)".into();
            }
        }
        // Diagnosis (`DS_SKIP_NGG` / `DS_SKIP_PS=<hex lo>-<hex hi>[;...]`, 2026-11-19): draws
        // whose NGG or pixel program's code hash lies in a listed range left out (the Nexus
        // triangles' bisection by program: code hashes are the same in every boot, the
        // programs' addresses are not).
        if skip_progs_on() && skip_by_program(cp) {
            DRAW_SKIPPED.fetch_add(1, Ordering::Relaxed);
            return "skipped (`DS_SKIP_NGG` / `DS_SKIP_PS`)".into();
        }
        if let Some(t) = imm_check() {
            if shaders::ngg_addr(cp) == t {
                imm_check_draw(cp, &args, t);
            }
        }
        let t_prog = super::prog_census().then(std::time::Instant::now);
        self.census_kind = 0;
        let composite = super::super::uiscale::is_composite(((cp.reg(Space::Sh, 0x08) as u64) | ((cp.reg(Space::Sh, 0x09) as u64 & 0xFF) << 32)) << 8);
        // Menu & HUD Size: the interface layer scaled before the title composites it.
        if composite && super::super::uiscale::scale() < 1.0 {
            self.scale_ui_layer(cp);
        }
        // Frame generation: the composite drawn first without the interface (`framegen::capture`).
        if composite && super::super::framegen::capture::wanted() {
            self.fg_capture_composite(cp, &args);
        }
        // (The vertices a non-indexed draw of one instance reads: `snap_prog`'s snapshots.)
        set_snap_verts(if args.index.is_none() && args.instances == 1 { Some((args.base_vertex.unwrap_or_else(|| ctx_reg(cp, 0x28408) as i32).max(0) as u64, args.count as u64)) } else { None });
        let r = self.draw_inner(cp, &args);
        // (Frame generation's diagnosis: the pixel stages that draw into a display buffer, when
        // they are not the composite it captures: `capture::note_display_draw`.)
        if !composite && super::super::framegen::capture::wanted() && matches!(r, Ok(true)) {
            if let Some(rt) = self.rendering.as_ref().and_then(|r| r.colors[0]) {
                let ps = ((cp.reg(Space::Sh, 0x08) as u64) | ((cp.reg(Space::Sh, 0x09) as u64 & 0xFF) << 32)) << 8;
                super::super::framegen::capture::note_display_draw(rt.key.base, ps, rt.key.format, rt.width, rt.height);
            }
        }
        set_snap_verts(None);
        self.ui_sub = None;
        self.ui_job = None;
        if let Some(t) = t_prog {
            let ps = ((cp.reg(Space::Sh, 0x08) as u64) | ((cp.reg(Space::Sh, 0x09) as u64 & 0xFF) << 32)) << 8;
            let ns = t.elapsed().as_nanos() as u64;
            super::prog_note(self.prog_class(), shaders::ngg_addr(cp), ps.max(2), ns);
            let (depth, key) = pass_key(cp);
            super::pass_note(self.prog_class(), depth, key, self.census_kind, ns);
        }
        let outcome = match r {
            Ok(true) => {
                DRAWN.fetch_add(1, Ordering::Relaxed);
                "drawn".into()
            }
            Ok(false) => {
                DRAW_NOOP.fetch_add(1, Ordering::Relaxed);
                // (Nothing recorded: the last recorded draw is no longer this state's, so the
                // next draw is not its repeat (`draw_repeat_cpu`); a dropped absurd draw's
                // followers went that way unchecked: the review of 2026-11-23.)
                self.repeat = None;
                "no effect".into()
            }
            Err(e) if e == super::STOPPED => "stopped (parallel run)".into(),
            Err(e) if e == NEEDS_CPU_ARGS => NEEDS_CPU_ARGS.into(),
            Err(e) => {
                DRAW_SKIPPED.fetch_add(1, Ordering::Relaxed);
                let s = format!("skipped: {e}");
                self.report(e);
                s.into()
            }
        };
        if sync_mode() {
            self.flush();
            if DEVICE_LOST.load(Ordering::Relaxed) {
                let vs = ((cp.reg(Space::Sh, 0xC8) as u64) | ((cp.reg(Space::Sh, 0xC9) as u64 & 0xFF) << 32)) << 8;
                let ps = ((cp.reg(Space::Sh, 0x08) as u64) | ((cp.reg(Space::Sh, 0x09) as u64 & 0xFF) << 32)) << 8;
                crate::error!("gpu[{}]: device lost by a draw (vs {vs:#x}, ps {ps:#x})", self.name);
            }
        }
        outcome
    }

    /// Menu & HUD Size (`gpu::uiscale`): the interface layer the title's composite samples (its
    /// pixel stage's resource 0) scaled around its centre into an image of ours; the composite
    /// binds that image in the layer's place (`ui_sub`, read where the draw binds its pixel
    /// stage). Nothing is recorded here: the copy goes in where the composite begins rendering
    /// (`ui_job`), once the draw's batch is open (a draw may be the first command of a batch, and
    /// a parallel chunk may stop at it).
    fn scale_ui_layer(&mut self, cp: &Cp) {
        let Ok(p) = shaders::prepare_ps(cp) else { return };
        let Some(super::Resolved::Image(t)) = p.resources.first() else { return };
        let frame = super::super::FRAMES.load(Ordering::Relaxed);
        let Ok(tex) = super::super::images::with(|im| im.upscale_tex(t, false, frame)) else { return };
        let s = super::super::uiscale::scale();
        if let Some(job) = super::super::uiscale::prepare(tex.image, tex.format, tex.size.0, tex.size.1, s) {
            self.ui_sub = Some((tex.view, job.view));
            self.ui_job = Some(job);
            // (Rendering begins again at the composite: the copy goes in before it.)
            self.end_rendering();
        }
    }

    /// Frame generation's HUD-less picture (`gpu::framegen::capture`): the title's composite drawn
    /// once with a transparent image in its interface layer's place, the display buffer it drew
    /// into copied, and the layer (or its scaled copy, what the composite samples) copied beside
    /// it; the composite then draws as the title made it (`draw` goes on). Whether it writes every
    /// pixel or blends the layer over the picture already there, the first draw leaves the picture
    /// without the interface and the second the title's own.
    fn fg_capture_composite(&mut self, cp: &Cp, args: &DrawArgs) {
        use super::super::framegen::capture::skipped;
        skipped("");
        let p = match shaders::prepare_ps(cp) {
            Ok(p) => p,
            Err(e) => return skipped(&format!("its pixel stage not prepared ({e})")),
        };
        let Some(super::Resolved::Image(t)) = p.resources.first() else { return skipped("its resource 0 not an image") };
        let frame = super::super::FRAMES.load(Ordering::Relaxed);
        let layer = match super::super::images::with(|im| im.upscale_tex(t, false, frame)) {
            Ok(l) => l,
            Err(e) => return skipped(&format!("the interface layer not resolved ({e})")),
        };
        let Some(zero) = super::super::framegen::capture::transparent_view() else { return skipped("no transparent image") };
        let id = super::super::streamline_frame();
        let saved = self.ui_sub;
        self.ui_sub = Some((layer.view, zero));
        let r = self.draw_inner(cp, args);
        self.ui_sub = saved;
        // (The title's own draw comes next: no buffer-delta or repeated draw of this one.)
        self.delta.at = None;
        self.repeat = None;
        match r {
            Ok(true) => {}
            Ok(false) => return skipped("the hidden-interface draw drew nothing"),
            Err(e) => return skipped(&format!("the hidden-interface draw failed ({e})")),
        }
        let Some(rt) = self.rendering.as_ref().and_then(|r| r.colors[0]) else { return skipped("no colour target after the draw") };
        self.end_rendering();
        // What the composite samples in the layer's place: the scaled copy under Menu & HUD Size.
        let ui = if saved.is_some() { super::super::uiscale::spare_image() } else { Some((layer.image, layer.size.0, layer.size.1)) };
        super::super::framegen::capture::record(self.cmd(), id, rt.image, rt.key.format, rt.width, rt.height, ui);
    }

    /// The same for an indexed indirect draw whose arguments the CPU read
    /// (`tune cpu_args_base`): `index` is its first index's address and the
    /// index size, the others its arguments. Its indices must lie in the
    /// home the draw before bound (bound again at their offset).
    pub fn draw_repeat_cpu(&mut self, index: (u64, u32), count: u32, instances: u32, base_vertex: i32, first_instance: u32) -> bool {
        let Some(mut r) = self.repeat.take() else { return false };
        if DEVICE_LOST.load(Ordering::Relaxed) || !self.recording || self.stopped || self.batch_count != r.batch || self.cmd_uses.get() != r.uses || self.rendering.is_none() || index.1 != r.index.1 || sync_mode() {
            return false;
        }
        if count == 0 || instances == 0 {
            // Nothing to draw (the full path's "no effect").
            DRAW_NOOP.fetch_add(1, Ordering::Relaxed);
            self.repeat = Some(r);
            return true;
        }
        // (A draw of absurd size dropped here too, as on the full path: `ABSURD_DRAW`.)
        if count as u64 * instances as u64 >= ABSURD_DRAW {
            if ABSURD_DROPPED.fetch_add(1, Ordering::Relaxed) < 16 {
                crate::warn!("gpu[{}]: a repeated draw of {count} x {instances} vertex invocations: dropped", self.name);
            }
            self.repeat = Some(r);
            return true;
        }
        let bytes = count as u64 * index.1 as u64;
        let laps = (!self.compute_cp && super::super::tune::cp_classes()).then(std::time::Instant::now);
        let ih = self.home_bind(index.0, bytes, false, HomeUse::Index);
        super::super::cp::idraw_part(3, laps);
        let Some(ih) = ih else { return false };
        if self.stopped || self.batch_count != r.batch || self.cmd_uses.get() != r.uses || ih.0 != r.index_buf {
            return false;
        }
        let cmd = self.slots[self.cur].cmd;
        let c = ctx();
        unsafe {
            if ih.1 != r.index_off {
                let ty = if index.1 == 4 { vk::IndexType::UINT32 } else { vk::IndexType::UINT16 };
                c.device.cmd_bind_index_buffer(cmd, ih.0, ih.1, ty);
                r.index_off = ih.1;
            }
            c.device.cmd_draw_indexed(cmd, count, instances, 0, base_vertex, first_instance);
        }
        REPEAT_DRAWS.fetch_add(1, Ordering::Relaxed);
        DRAWN.fetch_add(1, Ordering::Relaxed);
        self.repeat = Some(r);
        true
    }

    /// Whether the draw just recorded can be repeated (`draw_repeat`).
    pub fn repeat_ready(&self) -> bool {
        self.repeat.is_some_and(|r| r.uses == self.cmd_uses.get() && r.batch == self.batch_count)
    }

    /// Ends the active rendering instance; its attachment writes become
    /// visible to later commands.
    #[track_caller]
    pub(super) fn end_rendering(&mut self) {
        if self.rendering.take().is_some() {
            self.pair_ring_n = 0;
            unsafe { ctx().device.cmd_end_rendering(self.cmd()) };
            self.ops[super::OP_PASS_ENDS] += 1;
            if super::super::tune::skip_barriers() & 1 == 0 {
                self.barrier();
            }
            self.barrier_wanted = false;
            self.dispatched = false;
        }
    }

    /// Ends the active rendering instance for a caller that records a full
    /// barrier next, before anything else touches memory: that barrier
    /// makes the attachment writes visible (`tune merge_barriers`; ending
    /// with a barrier of its own drained the GPU twice in a row).
    #[track_caller]
    pub(super) fn end_rendering_before_barrier(&mut self) {
        if !super::super::tune::merge_barriers() {
            self.end_rendering();
        } else if self.rendering.take().is_some() {
            unsafe { ctx().device.cmd_end_rendering(self.cmd()) };
            self.ops[super::OP_PASS_ENDS] += 1;
            MERGED_PASS_BARRIERS.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
        }
    }

    /// `end_rendering_before_barrier`, then that barrier.
    #[track_caller]
    pub(super) fn end_rendering_then_barrier(&mut self) {
        self.end_rendering_before_barrier();
        self.barrier();
    }

    fn begin_rendering(&mut self, colors: [Option<Target>; MAX_RT], depth: Option<Target>, area: (u32, u32)) {
        self.since_draws = true;
        let n = colors.iter().rposition(Option::is_some).map(|i| i + 1).unwrap_or(0);
        let color_infos: Vec<vk::RenderingAttachmentInfo> = colors[..n]
            .iter()
            .map(|t| match t {
                Some(t) => vk::RenderingAttachmentInfo::default().image_view(t.view).image_layout(vk::ImageLayout::GENERAL).load_op(vk::AttachmentLoadOp::LOAD).store_op(vk::AttachmentStoreOp::STORE),
                None => vk::RenderingAttachmentInfo::default().image_view(vk::ImageView::null()),
            })
            .collect();
        let depth_info = depth.map(|d| vk::RenderingAttachmentInfo::default().image_view(d.view).image_layout(vk::ImageLayout::GENERAL).load_op(vk::AttachmentLoadOp::LOAD).store_op(vk::AttachmentStoreOp::STORE));
        let mut ri = vk::RenderingInfo::default()
            .render_area(vk::Rect2D { offset: vk::Offset2D::default(), extent: vk::Extent2D { width: area.0, height: area.1 } })
            .layer_count(1)
            .color_attachments(&color_infos);
        if let (Some(d), Some(t)) = (&depth_info, &depth) {
            if images::aspects(t.format).contains(vk::ImageAspectFlags::DEPTH) {
                ri = ri.depth_attachment(d);
            }
            if images::has_stencil(t.format) {
                ri = ri.stencil_attachment(d);
            }
        }
        super::note_vk(super::VK_RENDER_PASSES);
        unsafe { ctx().device.cmd_begin_rendering(self.cmd(), &ri) };
        self.rendering = Some(Rendering { colors, depth, area, written: Vec::new(), sampled: Vec::new(), ds_written: [false; 2], ds_sampled: [false; 2] });
    }

    /// The last full draw's results when this draw can take them over (a
    /// buffer-delta draw): right after it (nothing recorded between, the
    /// same batch) with its context state, translations, images and
    /// samplers, its targets still the rendering instance's, the image
    /// cache's lookups as then, no fast clear or reload pending for its
    /// images and its textures known unwritten. Err: why not (0: no such
    /// draw, `DELTA`'s refusals from 1).
    /// `DS_RUN_CENSUS` (`gpures::RUN_CENSUS`): this prepared draw against the run of its pair a
    /// GPU-resolved run on the mirror would make, and the chain walk an append needs, timed
    /// (with the reads the preparation just made warm in the caches: a floor).
    fn run_census_note(&mut self, vs: &Prepared, ps: &Prepared, continues: bool) {
        use super::super::gpures as g;
        use super::super::resident::PAGE;
        let c = &g::RUN_CENSUS;
        c[0].fetch_add(1, Ordering::Relaxed);
        let Some(d) = g::draw_decided(vs.addr, ps.addr) else {
            c[1].fetch_add(1, Ordering::Relaxed);
            RC_RUN.with(|r| *r.borrow_mut() = None);
            return;
        };
        c[2].fetch_add(1, Ordering::Relaxed);
        let st = [vs, ps];
        let mut roots = Vec::with_capacity(8);
        let mut user = Vec::with_capacity(8);
        let mut formats = Vec::with_capacity(8);
        for k in 0..2 {
            let dk = &d.stages[k];
            for &r in &dk.run_roots {
                roots.push(g::root_value(r, &st[k].user));
            }
            for j in 0..32 {
                if dk.run_user & (1u32 << j) != 0 {
                    user.push(st[k].user.get(j).copied().unwrap_or(0));
                }
            }
            for (i, res) in st[k].resources.iter().enumerate().take(64) {
                if dk.gpu_res & (1u64 << i) != 0 {
                    if let shaders::Resolved::Buffer(v) = res {
                        formats.push((v.format, v.dst_sel));
                    }
                }
            }
        }
        let pair = (vs.addr, ps.addr);
        let verdict = RC_RUN.with(|r| match r.borrow().as_ref() {
            Some(x) if continues && x.pair == pair && x.roots == roots && x.user == user => Some(x.formats == formats),
            _ => None,
        });
        let Some(same_formats) = verdict else {
            c[5].fetch_add(1, Ordering::Relaxed);
            RC_RUN.with(|r| *r.borrow_mut() = Some(RcRun { pair, roots, user, formats }));
            return;
        };
        c[if same_formats { 3 } else { 4 }].fetch_add(1, Ordering::Relaxed);
        // The walk of the per-item chains an append makes (`shaders::gpu_chain_ranges`).
        let specs: [rdna::translate::Spec; 2] = [0, 1].map(|k| {
            let mut s = st[k].tr.spec.clone();
            s.gpu_loads = d.stages[k].gpu_loads;
            s.gpu_res = d.stages[k].gpu_res;
            s
        });
        let (mut loads, mut bufs, mut reads) = (Vec::new(), Vec::new(), Vec::new());
        let t = unsafe { core::arch::x86_64::_rdtsc() };
        let mut ok = true;
        for k in 0..2 {
            ok &= shaders::gpu_chain_ranges(st[k].addr, &st[k].tr.plan, &specs[k], &st[k].user, &mut loads, &mut bufs, &mut reads).is_ok();
        }
        let cycles = unsafe { core::arch::x86_64::_rdtsc() }.wrapping_sub(t);
        c[6].fetch_add(1, Ordering::Relaxed);
        c[7].fetch_add(cycles, Ordering::Relaxed);
        if !ok {
            c[11].fetch_add(1, Ordering::Relaxed);
        }
        c[8].fetch_add((loads.len() + bufs.len()) as u64, Ordering::Relaxed);
        c[10].fetch_add(loads.len() as u64, Ordering::Relaxed);
        // Which of them front end 2's mirror holds current as it stands (at most 16 pages each).
        let want = self.fe2_want(super::super::precache::epoch() as u32);
        let mut fine = 0u64;
        for &(a, n) in loads.iter().chain(bufs.iter()) {
            if a == 0 || n == 0 {
                continue;
            }
            let n = n.min(16 * PAGE);
            let (lo, hi) = (a & !(PAGE - 1), (a + n).next_multiple_of(PAGE));
            let in_arena = crate::engine::arena_allocated(a);
            if super::super::fe2::in_span(a, n) && super::super::fe2::quick_fine(lo, hi, want) {
                fine += 1;
                if in_arena {
                    c[15].fetch_add(1, Ordering::Relaxed);
                }
            } else if in_arena {
                // (Not current: inside the GPU arena's live frame allocations, which a pass
                // per title submission confined to them would make current, FINDINGS.md
                // 2026-11-09 night; above a MiB's high-water mark; elsewhere.)
                c[12].fetch_add(1, Ordering::Relaxed);
                g::census_stable(a, n, true);
            } else if crate::engine::alloc_entry_of(a >> 20) != 0 {
                c[13].fetch_add(1, Ordering::Relaxed);
            } else {
                c[14].fetch_add(1, Ordering::Relaxed);
                g::census_stable(a, n, false);
                // (By the engine allocator owning it, and whether front end 2 found a CPU
                // write to its page within the last 4 flips.)
                let o = crate::engine::owner(a) as usize;
                let w = super::super::fe2::wrote_flip(a) as u64;
                let hot = w != 0 && super::super::FRAMES.load(Ordering::Relaxed).saturating_sub(w - 1) < 4;
                g::RUN_CENSUS_OWNER[o][0].fetch_add(1, Ordering::Relaxed);
                g::RUN_CENSUS_OWNER[o][1].fetch_add(hot as u64, Ordering::Relaxed);
            }
        }
        c[9].fetch_add(fine, Ordering::Relaxed);
    }

    fn delta_holds(&self, at: Option<(u64, u64)>, keys: [u64; 2], sig: &[(u32, [u32; 8])], indexed: bool) -> Result<Fixed, usize> {
        let d = &self.delta;
        // (Programs at other addresses with the same translations count as the same: the
        // keys below are the translations'.)
        let same = self.draw_same || self.draw_same_ctx;
        let Some(f) = d.fixed.filter(|_| same && at == Some((self.batch_count, self.cmd_uses.get()))) else { return Err(0) };
        if !self.recording || sync_mode() || super::super::tune::race_checks() || TINY_ANY.load(Ordering::Relaxed) || !self.cpw_open.is_empty() {
            return Err(1);
        }
        if d.keys != keys || d.index != indexed {
            return Err(2);
        }
        if d.images[..] != *sig {
            return Err(3);
        }
        if images::LOOKUP_GEN.load(Ordering::Acquire) != f.lookup_gen {
            return Err(4);
        }
        if images::pending_hit(&d.used) {
            return Err(5);
        }
        if !d.textures.is_empty() {
            // Texture sync's memo (`sync_textures`) as it would find them.
            let tag = [self.own_gen, super::INFLIGHT.notes.load(Ordering::Acquire)];
            if super::super::tune::tex_waits() == 0 || tag != self.tex_clean_tag || !d.textures.iter().all(|r| self.tex_clean.contains(r)) {
                return Err(6);
            }
        }
        if !self.rendering.as_ref().is_some_and(|r| r.colors == f.colors && r.depth == f.depth) {
            return Err(7);
        }
        Ok(f)
    }

    /// `delta_holds`, counted (`DELTA`).
    fn delta_check(&mut self, at: Option<(u64, u64)>, keys: [u64; 2], sig: &[(u32, [u32; 8])], indexed: bool) -> Option<Fixed> {
        let r = self.delta_holds(at, keys, sig, indexed);
        match r {
            Ok(_) => self.delta.count(1),
            Err(0) => return None,
            Err(k) => self.delta.count(1 + k),
        }
        self.delta.count(0);
        r.ok()
    }

    /// Records the draw; Ok(false) when it has no effect here.
    fn draw_inner(&mut self, cp: &Cp, a: &DrawArgs) -> Result<bool, String> {
        self.fe2_min_want = u32::MAX;
        let delta_at = self.delta.at.take();
        if a.count == 0 || a.instances == 0 {
            return Ok(false);
        }
        // A draw of absurd size is logged (the first 16 a boot): 2026-11-06,
        // a 6 s frame at Phalanx's door had the GPU busy throughout on
        // ~20-draw worker batches of 385-487 ms each, unexplained.
        if a.count as u64 * a.instances as u64 >= HUGE_DRAW {
            self.note_huge_draw(cp, a);
            if a.count as u64 * a.instances as u64 >= ABSURD_DRAW {
                ABSURD_DROPPED.fetch_add(1, Ordering::Relaxed);
                return Ok(false);
            }
        }
        // Phase times (DS_PROFILE_GPU only: a clock read costs ~20 ns).
        let t0 = super::profiling().then(std::time::Instant::now);
        let since = |t0: Option<std::time::Instant>| t0.map_or(std::time::Duration::ZERO, |t| t.elapsed());
        let kind = if self.deferred { 2 } else if cp.in_parallel_run() { 1 } else { 0 };
        let first = self.batch_shaders.is_empty();
        super::super::fe2::set_first_command(first);
        let mut lap = super::Laps::for_draw(kind, first);
        let reg = |r: u32| ctx_reg(cp, r);
        // CB_COLOR_CONTROL.MODE: 0 color disabled, 1 normal; others are
        // metadata passes (fast-clear eliminate, FMASK/DCC decompress, resolve)
        // that have nothing to do on uncompressed single-sample images.
        let cb_mode = (reg(0x28808) >> 4) & 7;
        if cb_mode > 1 {
            if cb_mode == 3 {
                self.report("MSAA resolve pass not implemented (skipped)".into());
            }
            return Ok(false);
        }
        let clip = reg(0x28810);
        if clip & (1 << 22) != 0 {
            return Ok(false); // DX_RASTERIZATION_KILL
        }
        if SKIP_VS_ANY.load(Ordering::Relaxed) && SKIP_VS.lock().as_ref().is_some_and(|s| s.contains(&super::super::shaders::ngg_addr(cp))) {
            return Err("vertex shader on the skip list (diagnosis)".into());
        }
        let prim = cp.reg(Space::Uc, 0x242) & 0x3F;
        let (topology, class, rect) = match prim {
            1 => (vk::PrimitiveTopology::POINT_LIST, 0u8, false),
            2 => (vk::PrimitiveTopology::LINE_LIST, 1, false),
            3 => (vk::PrimitiveTopology::LINE_STRIP, 1, false),
            4 => (vk::PrimitiveTopology::TRIANGLE_LIST, 2, false),
            5 => (vk::PrimitiveTopology::TRIANGLE_FAN, 2, false),
            6 => (vk::PrimitiveTopology::TRIANGLE_STRIP, 2, false),
            // Agc rect list (7, see PROVENANCE) and the hardware RECTLIST.
            7 | 0x11 => (vk::PrimitiveTopology::TRIANGLE_LIST, 2, true),
            p => return Err(format!("primitive type {p:#x} not supported (draw skipped)")),
        };
        let aa = reg(0x28BE0);
        if aa & 7 != 0 {
            self.report(format!("MSAA ({} samples) rendered single-sampled", 1 << (aa & 7)));
        }
        // Shader stages.
        let ps_addr = ((cp.reg(Space::Sh, 0x08) as u64) | ((cp.reg(Space::Sh, 0x09) as u64 & 0xFF) << 32)) << 8;
        // Stages prepared ahead by the workers (`gpu::ahead`) when they still
        // hold, else here. A draw whose arguments the GPU reads takes the
        // start instance on the GPU (prepared here).
        // The start instance from gl_BaseInstance: arguments the GPU reads,
        // or CPU-read ones drawn with it as Vulkan's first instance.
        let base_slot = a.base_inst;
        // Batched draws: a GPU-resolved pair's stages are translated batched
        // and its draws recorded as runs of a multi-draw (`DrawRun`). Rect
        // lists (a geometry stage) and draws whose arguments the GPU reads
        // stay as they are.
        let vs_addr = shaders::ngg_addr(cp);
        // GPU-resolved draws (`gpures`, stage G4): a pair whose per-item
        // chains the shaders resolve runs batched, its draws after a run's
        // first appended with their user data alone (no preparation).
        let direct = ps_addr != 0 && !rect && self.recording && !sync_mode();
        let mut gpu_pair = if direct && matches!(super::super::tune::gpu_chains(), 1 | 3) && ctx().draw_indirect_count && ctx().device_addresses { super::super::gpures::draw_decided(vs_addr, ps_addr) } else { None };
        if gpu_pair.is_some() && self.draw_same {
            if let Some(drawn) = self.draw_gpu_append(cp, a, vs_addr, ps_addr)? {
                return Ok(drawn);
            }
        }
        let mut batched = direct && gpu_pair.is_some();
        let mut batched_guard = shaders::batched_draw(batched);
        let gpu_masks = gpu_pair.as_ref().filter(|_| batched).map(|d| [(d.stages[0].gpu_loads, d.stages[0].gpu_res), (d.stages[1].gpu_loads, d.stages[1].gpu_res)]);
        let masks_guard = shaders::gpu_draw_masks(gpu_masks);
        let prep_vs = |cp: &Cp| match base_slot {
            Some(s) => shaders::with_base_instance(s, || shaders::prepare_ngg(cp)),
            None => shaders::prepare_ngg(cp),
        };
        let [pre_vs, pre_ps] = cp.take_ahead();
        let pre_vs = if base_slot.is_some() || batched { None } else { pre_vs };
        let pre_ps = if batched { None } else { pre_ps };
        // (Diagnosis: the replay memo, `tune prep_memo`.)
        let memo = super::super::tune::prep_memo();
        let memo = memo != 0 && super::super::replay::replaying() && (memo == 2 || cp.is_main_gfx());
        let memo_extra = base_slot.map_or(0, |s| s as u64 + 1) | (batched as u64) << 32;
        let mut vs = match pre_vs.and_then(|s| crate::gpu::ahead::take(&s, cp, true)) {
            Some(p) => p,
            None if memo => shaders::prep_memo(shaders::prep_memo_key(shaders::StageKind::Vertex, cp.packet_addr(), cp, memo_extra), || prep_vs(cp))?,
            None => prep_vs(cp)?,
        };
        if let Some(slot) = base_slot {
            if shaders::uses_user_word(&vs, slot) {
                shaders::note_cpu_args_program(vs.addr);
                return Err(NEEDS_CPU_ARGS.into());
            }
        }
        lap.mark(0);
        // A pixel stage that reads no memory (the depth-only items' usual one: no user data,
        // resources or loads): the last one prepared under these context registers and
        // pixel program registers (`Cp::ps_epoch`) and tunables is this draw's too. (`tune
        // buffer_delta` 2: prepared anyway and compared.)
        let tune_gen = super::super::tune::generation();
        let taken = if ps_addr != 0 && !batched && !memo {
            self.empty_ps.as_ref().filter(|(e, g, p)| *e == self.ps_epoch && *g == tune_gen && p.addr == ps_addr).map(|(_, _, p)| p.clone())
        } else {
            None
        };
        let (taken, compare) = if super::super::tune::buffer_delta() == 2 { (None, taken) } else { (taken, None) };
        let ps_taken = taken.is_some();
        let mut ps = if ps_addr != 0 {
            match taken {
                Some(p) => {
                    if super::laps() {
                        EMPTY_PS[0].fetch_add(1, Ordering::Relaxed);
                    }
                    Some(p)
                }
                None => match pre_ps.and_then(|s| crate::gpu::ahead::take(&s, cp, true)) {
                    Some(p) => Some(p),
                    None if memo => Some(shaders::prep_memo(shaders::prep_memo_key(shaders::StageKind::Pixel, cp.packet_addr(), cp, memo_extra), || shaders::prepare_ps(cp))?),
                    None => Some(shaders::prepare_ps(cp)?),
                },
            }
        } else {
            None
        };
        if let (Some(old), Some(p)) = (&compare, &ps) {
            EMPTY_PS[1].fetch_add(1, Ordering::Relaxed);
            if !same_prepared(old, p) && EMPTY_PS[2].fetch_add(1, Ordering::Relaxed) < 8 {
                crate::warn!("gpu[{}]: a pixel stage taken from the draw before would differ from its preparation (ps {ps_addr:#x}, `tune buffer_delta` 2)", self.name);
            }
        }
        lap.mark(1);
        // Diagnosis (`skiprw <hex base>|off`, 2026-11-20): draws whose pixel stage writes the
        // image at that base left out (the G-buffer's in-place decals).
        let skip_rw = SKIP_RW.load(Ordering::Relaxed);
        if skip_rw != 0 {
            if let Some(p) = ps.as_ref() {
                if p.found.resources.iter().enumerate().any(|(i, r)| matches!(r, Resolved::Image(t) if t.base == skip_rw) && p.tr.plan.resources.get(i).is_some_and(|x| x.written)) {
                    // (The draw's ordinal among this flip's such draws: `skiprw <base> <lo> <hi>`
                    // leaves out only those in [lo, hi).)
                    let flip = crate::hle::libs::videoout::QUEUED_FLIP.load(Ordering::Relaxed);
                    let n = {
                        let mut g = SKIP_RW_ORD.lock();
                        if g.0 != flip {
                            *g = (flip, 0);
                        }
                        g.1 += 1;
                        g.1 - 1
                    };
                    let (lo, hi) = *SKIP_RW_RANGE.lock();
                    if n >= lo && n < hi {
                        SKIP_RW_N.fetch_add(1, Ordering::Relaxed);
                        if hi - lo == 1 && SKIP_RW_SEEN.lock().insert((lo, p.addr)) {
                            let rec = p.found.resources.iter().find_map(|r| match r {
                                Resolved::Buffer(v) => Some(v.base),
                                _ => None,
                            });
                            crate::info!("skiprw: draw {n} of flip {}: ps {:#x} record {:x?}", flip + 1, p.addr, rec);
                        }
                        return Ok(false);
                    }
                }
            }
        }
        // Batched stages that cannot run batched (memory writes, the varying's
        // location taken): the pair is refused and the stages prepared again.
        if batched {
            let writes = vs.tr.plan.resources.iter().any(|r| r.written) || ps.as_ref().is_some_and(|p| p.tr.plan.resources.iter().any(|r| r.written));
            let why = if !vs.tr.out.batched || !ps.as_ref().is_some_and(|p| p.tr.out.batched) {
                Some("not translated batched")
            } else if writes {
                Some("a stage writes memory")
            } else if vs.tr.out.param_mask & (1 << rdna::translate::BATCH_VARYING) != 0 {
                Some("the vertex stage exports the varying's location")
            } else {
                None
            };
            if let Some(why) = why {
                if gpu_pair.take().is_some() {
                    super::super::gpures::refuse_draw((vs_addr, ps_addr), why);
                }
                batched = false;
                drop(batched_guard);
                batched_guard = shaders::batched_draw(false);
                vs = prep_vs(cp)?;
                ps = Some(shaders::prepare_ps(cp)?);
            }
        }
        // A GPU-resolved pair's draw recorded in full (a run's first): what
        // its per-item chains start from and reach goes into the page table,
        // and their data must be guest memory as the CPU path would bind it
        // (else the pair keeps CPU-resolved chains, this draw included).
        if let (true, Some(d)) = (batched, gpu_pair.as_ref()) {
            let p = ps.as_ref().unwrap();
            let mut why = None;
            // Front end 2 (stage F3): what the per-item chains read placed in
            // the mirror; else this draw resolves them on the CPU (the pair
            // stays GPU-resolved).
            let mut cpu_this = false;
            if super::super::fe2::on() {
                cpu_this = self.fe2_gpu_ranges(&vs.found, vs.addr, &vs.user).is_err() || self.fe2_gpu_ranges(&p.found, p.addr, &p.user).is_err();
                if cpu_this {
                    super::F3_PLACED[2].fetch_add(1, Ordering::Relaxed);
                }
            } else {
                for (k, st) in [&vs, p].into_iter().enumerate() {
                    if !super::gpu_pages_mapped(&d.stages[k].item_roots, &st.user) || !super::gpu_res_imported(&st.found, st.addr, st.tr.spec.gpu_res, &st.user) {
                        why = Some("its per-item chains reach memory that cannot be imported".to_string());
                    }
                }
                why = why.or_else(|| self.gpu_draw_check(&vs, p));
            }
            if why.is_some() || cpu_this {
                if let Some(why) = why {
                    super::super::gpures::refuse_draw((vs_addr, ps_addr), &why);
                }
                gpu_pair = None;
                batched = false;
                drop(batched_guard);
                batched_guard = shaders::batched_draw(false);
                vs = prep_vs(cp)?;
                ps = Some(shaders::prepare_ps(cp)?);
            }
        }
        // LOD statistics in rotation (`tune lod_late` >= 2): a pixel program
        // whose depth tests stay after the shader, prepared without them,
        // measures in one of its draws in N (prepared again with them); the
        // others carry their textures' recent measurements (`lodstats`).
        let lod_rotated = ps.as_ref().is_some_and(shaders::lod_rotated);
        let lod_measure = lod_rotated && super::super::lodstats::rotate(&ps.as_ref().unwrap().resources);
        if lod_measure {
            let p = shaders::with_lod_stats(|| shaders::prepare_ps(cp))?;
            super::super::lodstats::bound(&p.resources);
            ps = Some(p);
        }
        self.sync_readers = [vs.addr, ps.as_ref().map_or(0, |p| p.addr)];
        // Each stage's reads in place (no combined copy per draw).
        let waited = self.sync_for_stage(&vs.reads, SYNC_DRAW_READS) | ps.as_ref().is_some_and(|p| self.sync_for_stage(&p.reads, SYNC_DRAW_READS));
        if self.stopped {
            return Err(super::STOPPED.into());
        }
        if waited {
            if let Some(c) = self.sync_culprit.take() {
                let mut st: Vec<&Prepared> = vec![&vs];
                if let Some(p) = &ps {
                    st.push(p);
                }
                super::explain_wait(&self.name, c, &st);
            }
            let same = shaders::words_unchanged(&vs) && ps.as_ref().is_none_or(shaders::words_unchanged);
            super::READ_WAIT_WORDS[if same { 0 } else { 1 }].fetch_add(1, Ordering::Relaxed);
            vs = prep_vs(cp)?;
            ps = match ps_addr {
                0 => None,
                _ if lod_measure => Some(shaders::with_lod_stats(|| shaders::prepare_ps(cp))?),
                _ => Some(shaders::prepare_ps(cp)?),
            };
            if let Some(p) = ps.as_ref().filter(|_| lod_rotated) {
                match lod_measure {
                    true => super::super::lodstats::bound(&p.resources),
                    false => super::super::lodstats::carry(&p.resources),
                }
            }
        }
        drop(batched_guard);
        drop(masks_guard);
        if !ps_taken && !batched {
            self.empty_ps = match &ps {
                Some(p) if reads_nothing(p) && !lod_rotated => Some((self.ps_epoch, tune_gen, p.clone())),
                _ => None,
            };
        }
        lap.mark(2);
        // (`DS_ITEM_CENSUS=1`: the draw's content for the lookup in the frame before.)
        if super::super::itemcensus::on() {
            super::super::itemcensus::note(pass_key(cp), &vs, ps.as_ref(), a.index.map(|i| i.0), a.count);
        }
        // (A GPU-resolved run's first draw: its steps apart, `DS_STEP_LAPS`.)
        let gpu_open = batched && gpu_pair.is_some();
        // Pairs not decided yet: learned from the draws the CPU resolves (also for the census of
        // GPU-resolved runs, `DS_RUN_CENSUS`, which only counts and prices).
        let run_census = super::super::gpures::census_on();
        if direct && gpu_pair.is_none() && (matches!(super::super::tune::gpu_chains(), 1 | 3) || run_census) {
            if let Some(p) = &ps {
                let continues = self.draw_same && (self.batch_count, self.cmd_uses.get()) == self.last_draw_end;
                super::super::gpures::observe_draw((vs.addr, p.addr), [(&vs.tr.plan, &vs.user[..]), (&p.tr.plan, &p.user[..])], continues);
                if run_census {
                    self.run_census_note(&vs, p, continues);
                }
            }
        }
        // Stage G4's census of draw runs (`DS_DRAW_CENSUS=1`): a sampled
        // draw's per-item buffers checked for data GPU work wrote.
        if super::super::drawcensus::on() {
            let pair = (vs.addr, ps.as_ref().map_or(0, |p| p.addr));
            let consecutive = self.draw_same && (self.batch_count, self.cmd_uses.get()) == self.last_draw_end;
            let mut stages: Vec<(&std::sync::Arc<rdna::analysis::Plan>, &[u32])> = vec![(&vs.tr.plan, &vs.user[..])];
            if let Some(p) = &ps {
                stages.push((&p.tr.plan, &p.user[..]));
            }
            if let Some(check) = super::super::drawcensus::observe(pair, &stages, consecutive, false) {
                let mut hit = false;
                for (st, idx) in [Some(&vs), ps.as_ref()].into_iter().flatten().zip(check) {
                    for i in idx {
                        if let Some(shaders::Resolved::Buffer(v)) = st.resources.get(i) {
                            let r = [(v.base, v.extent().max(4))];
                            hit |= super::super::homes::overlaps(v.base, v.extent().max(4)) || self.overlaps_pending(&r) || super::INFLIGHT.latest(&r).is_some() || super::CP_WRITES.latest(&r).is_some();
                        }
                    }
                }
                super::super::drawcensus::checked(pair, hit);
            }
        }
        // Measurement: what differs from the last draw of the same programs
        // on this executor (batching candidates; profiling only).
        if draw_stats() {
            let sig = draw_sig(&vs, ps.as_ref());
            // Consecutive: nothing recorded since the last full draw (repeats
            // record on its state) and the same context state.
            let consecutive = self.draw_same && (self.batch_count, self.cmd_uses.get()) == self.last_draw_end;
            let k = note_draw_diff(&mut self.last_draw_sig, sig, consecutive);
            // Batched-draw candidates (buffers differ): where their arguments
            // and indices are against the draw before.
            let here: (Option<u64>, Option<u64>) = (None, a.index.map(|i| i.0));
            if consecutive && k == 2 {
                let (args, index) = self.last_draw_where;
                let c = &BATCH_CANDS;
                match (here.0, args) {
                    (Some(x), Some(y)) if x == y + 20 => c[1].fetch_add(1, Ordering::Relaxed),
                    (Some(x), Some(y)) if x == y => c[2].fetch_add(1, Ordering::Relaxed),
                    (Some(_), _) => c[3].fetch_add(1, Ordering::Relaxed),
                    (None, _) => c[0].fetch_add(1, Ordering::Relaxed),
                };
                match (here.1, index) {
                    (Some(x), Some(y)) if x == y => c[4].fetch_add(1, Ordering::Relaxed),
                    (Some(x), Some(y)) if x.abs_diff(y) < 64 << 20 => c[5].fetch_add(1, Ordering::Relaxed),
                    (Some(_), _) => c[6].fetch_add(1, Ordering::Relaxed),
                    (None, _) => c[7].fetch_add(1, Ordering::Relaxed),
                };
            }
            self.last_draw_where = here;
            note_prep_repeat(&vs, ps.as_ref());
        }
        if let Some((lo, hi)) = view_census_range() {
            note_view_census(cp, lo, hi, &vs, ps.as_ref());
        }
        let t_prep = since(t0);
        // Buffer-delta draws (`tune buffer_delta`, RENDERER.md): a draw right
        // after a full draw of the same programs, images and samplers with
        // the same context state takes over that draw's texture sync,
        // targets, pipeline, image hazards, rendering instance and dynamic
        // state (`delta_holds`); 2 runs the full path and compares.
        let delta_mode = super::super::tune::buffer_delta();
        let mut sig = std::mem::take(&mut self.delta_sig);
        let keys = [vs.tr.key, ps.as_ref().map_or(0, |p| p.tr.key)];
        let has_sig = delta_mode != 0 && !batched && gpu_pair.is_none() && image_sig([Some(&vs), ps.as_ref()], &mut sig);
        let mut delta = if has_sig { self.delta_check(delta_at, keys, &sig, a.index.is_some()) } else { None };
        if delta.is_some() && self.held {
            // A held executor's reads and its own clears and reloads, as the
            // full path notes and checks them (`sync_textures`, `held_hit`).
            let used = std::mem::take(&mut self.delta.used);
            let hit = self.held_hit(&used);
            self.delta.used = used;
            if hit {
                delta = None;
                // (Taken back; a flush since added it: the shared count wraps back.)
                self.delta.counts[1] = self.delta.counts[1].wrapping_sub(1);
                self.delta.count(6);
            } else {
                let n = self.delta.textures.len();
                self.held_reads.extend_from_slice(&self.delta.textures);
                self.note_kinds(1, n);
            }
        }
        let expect = if delta_mode == 2 { delta.take() } else { None };
        if delta.is_none() {
            let mut textures = std::mem::take(&mut self.tex_scratch);
            textures.clear();
            super::texture_ranges_into(&vs, &mut textures);
            if let Some(p) = &ps {
                super::texture_ranges_into(p, &mut textures);
            }
            let waited = self.sync_textures(&textures, SYNC_DRAW_TEXTURES);
            self.tex_scratch = textures;
            if waited && expect.is_some() {
                delta_mismatch(0, vs.addr, ps_addr);
            }
        }
        if self.stopped {
            return Err(super::STOPPED.into());
        }
        lap.mark(3);
        // Indices in a video-memory home (`homes`; one is made for indices
        // that GPU work wrote through a large binding) are bound from it and
        // read in order on the GPU; others are waited for if GPU work writes
        // them.
        let index_bytes = |size: u32| a.count as u64 * size as u64;
        let index_home = match a.index {
            Some((addr, size)) => {
                let bytes = index_bytes(size);
                let h = self.home_bind(addr, bytes, false, super::HomeUse::Index);
                if h.is_none() {
                    self.sync_for(&[(addr, bytes)], SYNC_DRAW_INDICES);
                }
                if self.stopped {
                    return Err(super::STOPPED.into());
                }
                h
            }
            None => None,
        };
        lap.mark(4);
        // A buffer-delta draw takes the last full draw's targets, pipeline
        // and hazard check (`delta_holds`).
        let was_delta = delta.is_some();
        self.census_kind = 1 + was_delta as u8;
        let (fixed, t_pipe) = match delta {
            Some(f) => {
                // (The targets' use stamps, as their lookups make.)
                let frame = crate::gpu::FRAMES.load(Ordering::Relaxed);
                for t in f.colors.iter().flatten().chain(f.depth.iter()) {
                    t.used.stamp(frame);
                }
                // (And the frames they were drawn into, `async_gfx`.)
                for t in f.colors.iter().flatten() {
                    t.used.stamp_drawn(frame);
                }
                if f.z_write || f.s_write || f.clear_z || f.clear_s {
                    if let Some(d) = &f.depth {
                        d.used.stamp_drawn(frame);
                    }
                }
                for k in 5..9 {
                    lap.mark(k);
                }
                (f, since(t0))
            }
            None => {
                let lookup_gen = images::LOOKUP_GEN.load(Ordering::Acquire);
                // Color targets: enabled in CB_TARGET_MASK and CB_SHADER_MASK,
                // exported by the pixel shader with a non-zero export format.
                let mrt_mask = ps.as_ref().map(|p| p.tr.out.mrt_mask).unwrap_or(0);
                let col_format = reg(0x28714);
                let target_mask = reg(0x28238) & reg(0x2823C);
                let mut colors: [Option<Target>; MAX_RT] = [None; MAX_RT];
                // Each colour target's metadata and clear words as bound now (`claim_meta`).
                let mut color_metas: [(u64, [u32; 2]); MAX_RT] = [(0, [0; 2]); MAX_RT];
                let mut blend = [0u32; MAX_RT];
                let mut masks = [0u8; MAX_RT];
                if cb_mode == 1 {
                    // The surfaces first: slots that bind slices of one array (the
                    // G-buffer: CB_COLOR_VIEW selects the slice, CB_COLOR_ATTRIB3
                    // gives no depth) need it with all their slices, or an image made
                    // for the first slot alone has one layer and the others clamp to
                    // it (the G-buffer made again after the title changed its display
                    // mode lost three of its four layers, FINDINGS.md 2026-10-21).
                    let mut surfs: [Option<Surface>; MAX_RT] = [None; MAX_RT];
                    for i in 0..MAX_RT as u32 {
                        let m = (target_mask >> (4 * i)) & 0xF;
                        if m == 0 || mrt_mask & (1 << i) == 0 || (col_format >> (4 * i)) & 0xF == 0 {
                            continue;
                        }
                        surfs[i as usize] = color_surface(cp, i)?;
                    }
                    for i in 0..MAX_RT {
                        let Some(s) = surfs[i] else { continue };
                        if s.ty3d {
                            continue;
                        }
                        let need = surfs.iter().flatten().filter(|o| o.base == s.base && !o.ty3d && (o.width, o.height) == (s.width, s.height)).map(|o| o.slice + 1).max().unwrap_or(1);
                        if let Some(x) = surfs[i].as_mut() {
                            x.layers = x.layers.max(need);
                        }
                    }
                    for i in 0..MAX_RT as u32 {
                        let m = (target_mask >> (4 * i)) & 0xF;
                        let Some(s) = surfs[i as usize] else { continue };
                        let t = images::target(&s)?;
                        if colors.iter().flatten().any(|c| c.view == t.view) {
                            self.report(format!("color target {:#x} bound to two MRT slots (second ignored)", s.base));
                            continue;
                        }
                        colors[i as usize] = Some(t);
                        color_metas[i as usize] = (s.meta, s.clear);
                        blend[i as usize] = reg(0x28780 + 4 * i);
                        masks[i as usize] = m as u8;
                    }
                }
                lap.mark(5);
                // Depth/stencil.
                let dctl = reg(0x28800);
                let db_render = reg(0x28000);
                let dsurf = depth_surface(cp);
                let (has_z, has_s) = match &dsurf {
                    Some(s) => (images::aspects(s.format).contains(vk::ImageAspectFlags::DEPTH) && reg(0x28040) & 3 != 0, images::has_stencil(s.format)),
                    None => (false, false),
                };
                let z_test = has_z && dctl & 2 != 0;
                let z_write = z_test && dctl & 4 != 0;
                let s_test = has_s && dctl & 1 != 0;
                let sctl = reg(0x2842C);
                let refm = reg(0x28430);
                let refm_bf = reg(0x28434);
                // Diagnosis (`tune stencil_export 1`): draws whose pixel shader exports
                // the stencil test value (not translated: no stencil export on the
                // device) replace with 0 instead of the register's test value.
                let (refm, refm_bf) = if super::super::tune::stencil_export() == 1 && reg(0x2880C) & 2 != 0 { (refm & !0xFF, refm_bf & !0xFF) } else { (refm, refm_bf) };
                let s_write = s_test && (refm | refm_bf) & 0x00FF_0000 != 0 && sctl & 0xFF_FFFF != 0;
                if s_test {
                    note_stencil_gaps(ps.as_ref().map_or(0, |p| p.addr), dctl, sctl, refm, refm_bf, reg(0x2880C));
                }
                let clear_z = has_z && db_render & 1 != 0;
                let clear_s = has_s && db_render & 2 != 0;
                if z_test || s_test || clear_z || clear_s {
                    // A view of a later slice than the depth image has: grow it
                    // (after the work that used it) instead of a separate image.
                    let s = dsurf.as_ref().unwrap();
                    if let Some(old) = images::depth_to_grow(s.base, s.width, s.height, s.format, s.layers) {
                        self.end_rendering();
                        self.flush_for("depth growth");
                        if self.stopped {
                            return Err(super::STOPPED.into());
                        }
                        images::with(|im| im.grow_depth(old, s.layers))?;
                    }
                }
                let depth = if z_test || s_test || clear_z || clear_s { Some(images::target(dsurf.as_ref().unwrap())?) } else { None };
                if draw_traces() {
                    trace_depth_attachment(vs.addr, ps.as_ref().map_or(0, |p| p.addr), &depth, z_write);
                }
                lap.mark(6);
                let mem_writes = vs.tr.plan.resources.iter().any(|r| r.written) || ps.as_ref().is_some_and(|p| p.tr.plan.resources.iter().any(|r| r.written));
                if colors.iter().all(Option::is_none) && !z_write && !s_write && !clear_z && !clear_s && !mem_writes {
                    return Ok(false);
                }
                let attachless = colors.iter().all(Option::is_none) && depth.is_none();
                if attachless && !super::super::tune::attachless_draws() {
                    return Err("draw with only memory writes (no attachments) not supported yet (skipped)".into());
                }
                let mut area = (u32::MAX, u32::MAX);
                for t in colors.iter().flatten().chain(depth.iter()) {
                    area = (area.0.min(t.width), area.1.min(t.height));
                }
                if attachless {
                    // Memory side effects only: a rendering instance without
                    // attachments, shared by such draws; each draw's scissors (from
                    // its registers) limit what it rasterizes.
                    area = (ATTACHLESS_AREA, ATTACHLESS_AREA);
                    ATTACHLESS.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
                }
                let (x0, y0, x1, y1) = scissor(cp, area);
                if x1 <= x0 || y1 <= y0 {
                    return Ok(false);
                }
                // Pipeline.
                let depth_clamp = clip & (3 << 26) != 0;
                let key = GfxKey {
                    vs: vs.tr.key,
                    ps: ps.as_ref().map(|p| p.tr.key).unwrap_or(0),
                    rect_gs: rect.then(|| rect_gs_mask(vs.tr.out.param_mask, ps.as_ref().map_or(0, |p| p.tr.out.in_mask))),
                    colors: colors.map(|c| c.map(|t| t.format.as_raw()).unwrap_or(vk::Format::UNDEFINED.as_raw())),
                    blend,
                    masks,
                    depth: depth.map(|d| d.format.as_raw()).unwrap_or(vk::Format::UNDEFINED.as_raw()),
                    topology_class: class,
                    depth_clamp,
                };
                let push_mode = match &ps {
                    Some(p) => super::push_fits(&[&vs, p]),
                    None => super::push_fits(&[&vs]),
                };
                // Consecutive draws mostly share their pipeline state: the key's
                // hash (also the persistent cache file's name, so its hasher stays)
                // is computed again only when the key changed on this thread.
                let hkey = {
                    let last = LAST_GFX_KEY.with(|c| c.get()).filter(|(k, _)| *k == key).map(|(_, h)| h);
                    let h = last.unwrap_or_else(|| {
                        let h = hash_gfx_key(&key);
                        LAST_GFX_KEY.with(|c| c.set(Some((key, h))));
                        h
                    });
                    super::push_key(h, push_mode)
                };
                let cached = gfx_pipe_use(hkey);
                let frame = crate::gpu::FRAMES.load(Ordering::Relaxed);
                let Pipe { pipeline, layout, set_layout, push } = match cached {
                    Some(x) => x,
                    None => {
                        // A first use. A draw that changes nothing lasting is
                        // left out while its pipeline is made in the
                        // background (`tune async_gfx`): no memory writes, no
                        // depth or stencil clear, and every target it writes
                        // one that draws wrote in 3 of the 4 frames before (a
                        // target drawn once, as a texture the title renders
                        // for later frames, waits).
                        if super::super::tune::async_gfx() != 0 {
                            let per_frame = |t: &Target| t.used.drawn_before(frame) >= 3;
                            let why = if mem_writes {
                                Some(0)
                            } else if clear_z || clear_s {
                                Some(1)
                            } else if batched || gpu_pair.is_some() {
                                Some(4)
                            } else if !self.recording || sync_mode() || super::super::capture::capturing() || super::super::verify::enabled() || super::super::replay::replaying() {
                                Some(5)
                            } else if !colors.iter().flatten().all(per_frame) || ((z_write || s_write) && !depth.as_ref().is_some_and(per_frame)) {
                                Some(2)
                            } else {
                                None
                            };
                            // A pipeline the store holds is made here within the frame's
                            // budget (milliseconds; an object left out is missing from the
                            // frame).
                            let stored_inline = why.is_none() && crate::gpu::pcache::stored_cheap('g', hkey) && {
                                let ok = inline_ok(frame as u64);
                                if !ok {
                                    INLINE_STATS[2].fetch_add(1, Ordering::Relaxed);
                                }
                                ok
                            };
                            match why {
                                None if stored_inline => {
                                    inline_mark(true);
                                }
                                None => {
                                    if async_gfx_submit(&key, hkey, &vs, ps.as_ref(), push_mode) {
                                        // A draw that was to measure LOD statistics: its
                                        // textures report their recent measurements instead,
                                        // as the rotation's other draws do (`lodstats`).
                                        if lod_measure {
                                            if let Some(p) = &ps {
                                                super::super::lodstats::carry(&p.resources);
                                            }
                                            ASYNC_GFX_WAITED[3].fetch_add(1, Ordering::Relaxed);
                                        }
                                        return Ok(false);
                                    }
                                }
                                Some(w) => {
                                    ASYNC_GFX_WAITED[w].fetch_add(1, Ordering::Relaxed);
                                    // The draws after this one: their pipelines queued
                                    // meanwhile (`Cp::prefetch_gfx`).
                                    if crate::gpu::compile::gfx_ahead() {
                                        cp.prefetch_gfx();
                                    }
                                }
                            }
                        }
                        let t = std::time::Instant::now();
                        let created = std::cell::Cell::new(false);
                        let p = gfx_pipe_with(&key, hkey, (&**vs.tr, vs.addr), ps.as_ref().map(|p| (&**p.tr, p.addr)), push_mode, &created)?;
                        // (On an error the mark stays set: this thread's next first use is charged
                        // to the budget. Left as trained: the profile of this function, 2026-11-14.)
                        if inline_mark(false) {
                            inline_spent(frame as u64, t.elapsed().as_nanos() as u64);
                        }
                        if compile_trace() {
                            crate::info!("first use: vs {:#x} ps {ps_addr:#x} key {hkey:#x} after {} ms{}", vs.addr, t.elapsed().as_millis(), if created.get() { "" } else { " (made by another thread)" });
                            if let Some(d) = ahead_gfx_diff(&key, &vs, ps.as_ref()) {
                                crate::info!("first use: vs {:#x} ps {ps_addr:#x} key {hkey:#x}: {d}", vs.addr);
                            }
                        }
                        GFX_FIRST_USES[0].fetch_add(1, Ordering::Relaxed);
                        GFX_FIRST_USES[1].fetch_add(t.elapsed().as_nanos() as u64, Ordering::Relaxed);
                        p
                    }
                };
                super::super::pipeuse::note(hkey);
                // The frames each target was drawn into (`async_gfx`).
                for t in colors.iter().flatten() {
                    t.used.stamp_drawn(frame);
                }
                if z_write || s_write || clear_z || clear_s {
                    if let Some(d) = &depth {
                        d.used.stamp_drawn(frame);
                    }
                }
                let t_pipe = since(t0);
                lap.mark(7);
                // Profile: the step's parts by draw kind ([draws, grow ns, check
                // ns, pending, resolve ns]).
                let hz_t = profiling().then(std::time::Instant::now);
                // Depth arrays sampled with more layers than rendered so far.
                match &ps {
                    Some(p) => self.grow_sampled_depth(&[&vs, p]),
                    None => self.grow_sampled_depth(&[&vs]),
                }
                let hz_t1 = hz_t.map(|_| std::time::Instant::now());
                // A fast clear pending on a bound target's metadata under another image is this
                // target's (`Exec::claim_meta`: the title reuses metadata blocks between targets).
                for (i, c) in colors.iter().enumerate() {
                    if let Some(t) = c {
                        self.claim_meta(t.key, color_metas[i].0, color_metas[i].1);
                    }
                }
                if let (Some(t), Some(s)) = (depth.as_ref(), dsurf.as_ref())
                    && t.key.layers == 1
                {
                    self.claim_meta(t.key, s.meta, s.clear);
                }
                if self.stopped {
                    return Err(super::STOPPED.into());
                }
                // Fast clears (metadata writes), reloads and block copies of the
                // attachments and textures.
                let mut used = std::mem::take(&mut self.keys_scratch);
                used.clear();
                used.extend(colors.iter().flatten().chain(depth.iter()).map(|t| t.key));
                let pending = super::super::tune::skip_steps() & 2 == 0
                    && match &ps {
                        Some(p) => super::image_hazards(&[&vs, p], &mut used),
                        None => super::image_hazards(&[&vs], &mut used),
                    };
                let pending = self.held_hit(&used) || pending;
                if pending && expect.is_some() {
                    delta_mismatch(2, vs.addr, ps_addr);
                }
                let hz_t2 = hz_t.map(|_| std::time::Instant::now());
                if self.stopped {
                    return Err(super::STOPPED.into());
                }
                if pending {
                    self.resolve_fast_clears(&used);
                    self.resolve_overwrites(&used);
                    self.sync_block_textures(&used);
                    self.sync_depth_views(&used);
                    if self.stopped {
                        return Err(super::STOPPED.into());
                    }
                }
                if let (Some(t0), Some(t1), Some(t2)) = (hz_t, hz_t1, hz_t2) {
                    let h = &super::HAZARD_STEPS[kind];
                    h[0].fetch_add(1, Ordering::Relaxed);
                    h[1].fetch_add((t1 - t0).as_nanos() as u64, Ordering::Relaxed);
                    h[2].fetch_add((t2 - t1).as_nanos() as u64, Ordering::Relaxed);
                    h[3].fetch_add(pending as u64, Ordering::Relaxed);
                    h[4].fetch_add(t2.elapsed().as_nanos() as u64, Ordering::Relaxed);
                }
                self.keys_scratch = used;
                lap.mark(8);
                let f = Fixed { colors, depth, area, scissor: (x0, y0, x1, y1), z_test, z_write, s_test, s_write, clear_z, clear_s, dctl, sctl, refm, refm_bf, pipe: Pipe { pipeline, layout, set_layout, push }, writes: mem_writes, hazard: pending, lookup_gen };
                if expect.is_some_and(|e| !f.same_as(&e)) {
                    delta_mismatch(1, vs.addr, ps_addr);
                }
                (f, t_pipe)
            }
        };
        let Fixed { colors, depth, area, scissor: (x0, y0, x1, y1), z_test, z_write, s_test, s_write, clear_z, clear_s, dctl, sctl, refm, refm_bf, pipe: Pipe { pipeline, layout, set_layout, push }, .. } = fixed;
        // Resources.
        match &ps {
            Some(p) => self.reserve(&[&vs, p], push),
            None => self.reserve(&[&vs], push),
        }
        if self.stopped {
            return Err(super::STOPPED.into());
        }
        if crate::gpu::vk::labels_on() {
            let l = format!("{} draw ps {:#x}", self.name, ps.as_ref().map_or(0, |p| p.addr));
            self.label(&l);
        }
        if let Some(h) = &index_home {
            // Kept until the batch that draws completes (once per batch).
            if !self.keep_home(h.0, h.3.as_ref()) {
                return Err("the index buffer's video-memory home was dropped during the draw (draw skipped)".into());
            }
        }
        // Indices in guest memory that a resident shadow holds newer data for.
        if let Some((addr, size)) = a.index {
            let t = profiling().then(std::time::Instant::now);
            self.flush_resident(Some(&[(addr, index_bytes(size))]))?;
            if let Some(t) = t {
                super::BEGIN_STATS[2].fetch_add(1, Ordering::Relaxed);
                super::BEGIN_STATS[3].fetch_add(t.elapsed().as_nanos() as u64, Ordering::Relaxed);
            }
        }
        lap.mark(9);
        // The vector is kept for the next draw (empty: it drains at the end).
        let mut bounds = std::mem::take(&mut self.draw_bounds);
        bounds.clear();
        let mut sgpr_at = [0u64; 2];
        let mut sgpr_ring = vk::Buffer::null();
        let mut notes: [Option<([u64; 3], bool)>; 2] = [None; 2];
        // A buffer-delta draw's images and samplers are the last draw's
        // while their memos' generations and the frame hold.
        let gens_hold = (delta.is_some() || expect.is_some()) && self.delta.bind_gens.is_some_and(|g| g == super::bind_gens());
        let mut reused = 0;
        for k in 0..2 {
            let p: &mut Prepared = match k {
                0 => &mut vs,
                _ => match ps.as_mut() {
                    Some(p) => p,
                    None => break,
                },
            };
            if delta.is_some() || expect.is_some() {
                self.delta.count(if gens_hold { 9 } else { 10 });
            }
            if delta.is_some() && gens_hold {
                self.bind_reuse = Some(k);
                reused += 1;
            }
            let mut b = self.bind_stage(p, false)?;
            notes[k] = self.bind_note.take();
            if draw_traces() {
                trace_ps_binding(p, &b, notes[k], a);
            }
            // (`DS_DECAL_CHECK=<hex image base>`: the read-only buffers of draws whose pixel stage
            // stores to that image, as the CPU holds them now and where they are bound, for
            // `atprobe`'s comparison with what the GPU read.)
            if k == 1 && decal_check() != 0 {
                note_decal_bufs(p, &b);
            }
            // Menu & HUD Size: the composite samples our scaled copy of the interface layer.
            if k == 1 {
                if let Some((from, to)) = self.ui_sub {
                    for e in b.imgs.iter_mut().filter(|e| e.2.image_view == from) {
                        e.2.image_view = to;
                    }
                }
            }
            if self.stopped {
                return Err(super::STOPPED.into());
            }
            if gens_hold && expect.is_some() && !self.delta.bind_same(k, &b) {
                delta_mismatch(5, vs_addr, ps_addr);
            }
            let (ring, at) = self.bind_sgpr(p, &mut b)?;
            sgpr_at[k] = at;
            sgpr_ring = ring;
            bounds.push(b);
            lap.mark(10 + k.min(1));
        }
        if profiling() {
            // Buffer placements of the draw's stages (vertex stage first), under
            // the draw's profile key.
            let all: Vec<_> = bounds.iter().flat_map(|b| b.placements.iter().copied()).collect();
            super::note_placements(super::DRAW_KEY | if ps_addr != 0 { ps_addr } else { vs.addr }, [0; 3], false, &all);
        }
        let set = if push { None } else { Some(self.alloc_set(set_layout)?) };
        lap.mark(12);
        if let Some(set) = set {
            self.write_set(set, &bounds);
        }
        lap.mark(13);
        // A CP write left open (`tune cpw_lazy`) that this draw's buffers or
        // indices touch: its barrier first (ending the rendering instance).
        if !self.cpw_open.is_empty() && super::cpw_lazy_here() {
            let index = a.index.map(|(addr, size)| (addr, a.count as u64 * size as u64));
            let hit = self.cpw_hazard(bounds.iter().flat_map(|b| b.reads.iter().chain(b.writes.iter())).chain(index.iter()));
            if hit {
                if self.rendering.is_some() {
                    self.end_rendering();
                } else {
                    self.barrier();
                }
            }
        }
        let t_bind = since(t0);
        // (A buffer-delta draw keeps the instance: the draw before it
        // neither sampled nor stored an attachment of it.)
        let mut touches = true;
        if delta.is_none() {
            // Rendering instance: reuse it while the attachments stay the same
            // and no attachment is both written and sampled in it (Vulkan's
            // rule for an image that is an attachment and a descriptor at once;
            // `tune feedback_exact`, else any sampled attachment ends it: ~266
            // instances per gate frame, each with a full barrier).
            let writes_ds = z_write || s_write || clear_z || clear_s;
            // This draw's writes of the depth attachment's [depth, stencil].
            let ds_writes = [z_write || clear_z, s_write || clear_s];
            let draw_writes = |img: vk::Image| colors.iter().flatten().any(|t| t.image == img) || (writes_ds && depth.as_ref().is_some_and(|d| d.image == img));
            let sampled_now = |img: vk::Image| bounds.iter().any(|b| b.images.contains(&img));
            let stored_now = |img: vk::Image| bounds.iter().any(|b| b.stored.contains(&img));
            // Its sampling of the depth attachment's [depth, stencil] (any view
            // that is not R8_UINT reads depth).
            let ds_sampled_now = |img: vk::Image| [bounds.iter().any(|b| b.images.contains(&img) && !b.stencil_views.contains(&img)), bounds.iter().any(|b| b.stencil_views.contains(&img))];
            let exact = super::super::tune::feedback_exact();
            let mut samples_attachment = false;
            let feedback = self.rendering.as_ref().is_some_and(|r| {
                let color = r.colors.iter().flatten().any(|t| {
                    let a = t.image;
                    let sampled = sampled_now(a);
                    samples_attachment |= sampled;
                    if !exact {
                        return sampled;
                    }
                    stored_now(a) || (sampled && (draw_writes(a) || r.written.contains(&a))) || (draw_writes(a) && r.sampled.contains(&a))
                });
                let ds = r.depth.as_ref().is_some_and(|t| {
                    let a = t.image;
                    let sampled = sampled_now(a);
                    samples_attachment |= sampled;
                    if !exact {
                        return sampled;
                    }
                    let now = ds_sampled_now(a);
                    stored_now(a) || (0..2).any(|k| (now[k] && (ds_writes[k] || r.ds_written[k])) || (ds_writes[k] && r.ds_sampled[k]))
                });
                color || ds
            });
            let same = self.rendering.as_ref().is_some_and(|r| r.colors == colors && r.depth == depth);
            if draw_traces() {
                trace_ps_targets(ps_addr, &colors, &depth);
            }
            if (!same || feedback) && expect.is_some() {
                delta_mismatch(3, vs.addr, ps_addr);
            }
            if samples_attachment && same {
                super::FEEDBACK[feedback as usize].fetch_add(1, std::sync::atomic::Ordering::Relaxed);
                // What the sampled attachment is and how this draw uses it
                // (statistics): [color written by the draw, color not written,
                // depth with depth writes, with stencil writes only, with a
                // clear only, depth not written, stored to].
                let r = self.rendering.as_ref().unwrap();
                let mut kinds = [false; 7];
                for t in r.colors.iter().flatten() {
                    if sampled_now(t.image) {
                        kinds[if draw_writes(t.image) { 0 } else { 1 }] = true;
                    }
                    if stored_now(t.image) {
                        kinds[6] = true;
                    }
                }
                if r.depth.as_ref().is_some_and(|d| sampled_now(d.image)) {
                    kinds[if z_write { 2 } else if s_write { 3 } else if clear_z || clear_s { 4 } else { 5 }] = true;
                }
                for (k, &on) in kinds.iter().enumerate() {
                    if on {
                        super::FEEDBACK_KINDS[k].fetch_add(1, std::sync::atomic::Ordering::Relaxed);
                    }
                }
                if super::profiling_on() {
                    let ps_addr = ps.as_ref().map_or(0, |p| p.addr);
                    *super::FEEDBACK_PS.lock().get_or_insert_with(std::collections::HashMap::new).entry(ps_addr).or_default() += 1;
                }
            }
            if !same || feedback {
                if feedback {
                    self.end_rendering();
                } else {
                    self.end_rendering();
                }
                self.barrier_if_wanted(true);
                // Menu & HUD Size: the interface layer's scaled copy before the composite.
                if let Some(job) = self.ui_job.take() {
                    super::super::uiscale::record(self.cmd(), job);
                }
                self.begin_rendering(colors, depth, area);
            } else if super::super::tune::race_checks() && self.barrier_wanted {
                super::UNORDERED[4].fetch_add(1, std::sync::atomic::Ordering::Relaxed);
            }
            if super::super::tune::race_checks() {
                self.note_unordered(&bounds, &[vs.addr, ps_addr], true);
            }
            if super::unordered_diag() {
                let ib = a.index.map(|(ad, sz)| (ad, a.count as u64 * sz as u64));
                self.note_draw_reads(&bounds, ib);
            }
            if let Some(r) = self.rendering.as_mut() {
                for t in r.colors.iter().flatten() {
                    let a = t.image;
                    if draw_writes(a) && !r.written.contains(&a) {
                        r.written.push(a);
                    }
                    if sampled_now(a) && !r.sampled.contains(&a) {
                        r.sampled.push(a);
                    }
                }
                if let Some(a) = r.depth.as_ref().map(|t| t.image) {
                    let now = ds_sampled_now(a);
                    for k in 0..2 {
                        r.ds_written[k] |= ds_writes[k];
                        r.ds_sampled[k] |= now[k];
                    }
                }
            }
            touches = self.rendering.as_ref().is_none_or(|r| r.colors.iter().flatten().chain(r.depth.iter()).any(|t| sampled_now(t.image) || stored_now(t.image)));
        }
        lap.mark(14);
        let c = ctx();
        let cmd = self.cmd();
        crate::gpu::vk::checkpoint(cmd, crate::gpu::vk::mark::DRAW, ps_addr);
        // Viewport 0 (PA_CL_VPORT_*): the hardware window transform maps
        // directly onto a Vulkan viewport (negative height for a flipped Y).
        let vte = reg(0x28818);
        let (xs, xo, ys, yo, zs, zo) = (f32r(reg(0x2843C)), f32r(reg(0x28440)), f32r(reg(0x28444)), f32r(reg(0x28448)), f32r(reg(0x2844C)), f32r(reg(0x28450)));
        let vp = if vte & 0x3F == 0x3F && xs != 0.0 && ys != 0.0 {
            vk::Viewport { x: xo - xs, y: yo - ys, width: 2.0 * xs, height: 2.0 * ys, min_depth: zo.clamp(0.0, 1.0), max_depth: (zo + zs).clamp(0.0, 1.0) }
        } else {
            self.report(format!("PA_CL_VTE_CNTL {vte:#x} (viewport transform disabled) approximated"));
            vk::Viewport { x: 0.0, y: 0.0, width: area.0 as f32, height: area.1 as f32, min_depth: 0.0, max_depth: 1.0 }
        };
        if draw_traces() && (trace_ps_addrs().contains(&vs.addr) || trace_bin_match(vs.addr)) {
            crate::info!(
                "trace vs {:#x} viewport at flip {}: {:?} xs {xs} xo {xo} ys {ys} yo {yo} zs {zs} zo {zo} vte {vte:#x} clip {:#x} zmin/zmax {}/{} depth ctl {:#x}",
                vs.addr,
                crate::hle::libs::videoout::QUEUED_FLIP.load(Ordering::Relaxed) + 1,
                vp,
                reg(0x28810),
                f32r(reg(0x282D0)),
                f32r(reg(0x282D4)),
                reg(0x28800)
            );
        }
        let dyn_before = expect.map(|_| self.dyncache.sig());
        let dc = &mut self.dyncache;
        let untimed;
        unsafe {
            let rebound = dc.pipeline != pipeline;
            if rebound {
                super::note_vk(super::VK_GFX_PIPELINES);
                c.device.cmd_bind_pipeline(cmd, vk::PipelineBindPoint::GRAPHICS, pipeline);
                dc.pipeline = pipeline;
            }
            super::bind_descriptors(cmd, vk::PipelineBindPoint::GRAPHICS, layout, set, &bounds, &mut dc.pushed[0], rebound);
            super::push_sgpr_bases(cmd, layout, false, &bounds);
            if delta.is_none() {
                let vpw = [vp.x.to_bits(), vp.y.to_bits(), vp.width.to_bits(), vp.height.to_bits(), vp.min_depth.to_bits(), vp.max_depth.to_bits()];
                if dc.viewport != Some(vpw) {
                    c.device.cmd_set_viewport(cmd, 0, &[vp]);
                    dc.viewport = Some(vpw);
                }
                let mut sc = vk::Rect2D { offset: vk::Offset2D { x: x0, y: y0 }, extent: vk::Extent2D { width: (x1 - x0) as u32, height: (y1 - y0) as u32 } };
                // Diagnosis (control `tiny <ps>`): those draws cover one pixel
                // (their vertex work stays, pixel work goes).
                if TINY_ANY.load(Ordering::Relaxed) && TINY_SCISSOR.lock().contains(&ps_addr) {
                    sc.extent = vk::Extent2D { width: 1, height: 1 };
                }
                let scw = [sc.offset.x, sc.offset.y, sc.extent.width as i32, sc.extent.height as i32];
                if dc.scissor != Some(scw) {
                    c.device.cmd_set_scissor(cmd, 0, &[sc]);
                    dc.scissor = Some(scw);
                }
                // Depth/stencil clears requested through DB_RENDER_CONTROL.
                if clear_z || clear_s {
                    let mut aspect = vk::ImageAspectFlags::empty();
                    if clear_z {
                        aspect |= vk::ImageAspectFlags::DEPTH;
                    }
                    if clear_s {
                        aspect |= vk::ImageAspectFlags::STENCIL;
                    }
                    let ca = vk::ClearAttachment {
                        aspect_mask: aspect,
                        color_attachment: 0,
                        clear_value: vk::ClearValue { depth_stencil: vk::ClearDepthStencilValue { depth: f32r(reg(0x2802C)), stencil: reg(0x28028) & 0xFF } },
                    };
                    c.device.cmd_clear_attachments(cmd, &[ca], &[vk::ClearRect { rect: sc, base_array_layer: 0, layer_count: 1 }]);
                    if colors.iter().all(Option::is_none) {
                        return Ok(true);
                    }
                }
                let mode = reg(0x28814);
                let mut cull = vk::CullModeFlags::NONE;
                if mode & 1 != 0 && !rect {
                    cull |= vk::CullModeFlags::FRONT;
                }
                if mode & 2 != 0 && !rect {
                    cull |= vk::CullModeFlags::BACK;
                }
                if dc.set(0, cull.as_raw() as u64) {
                    c.device.cmd_set_cull_mode(cmd, cull);
                }
                let front = if mode & 4 != 0 { vk::FrontFace::CLOCKWISE } else { vk::FrontFace::COUNTER_CLOCKWISE };
                if dc.set(1, front.as_raw() as u64) {
                    c.device.cmd_set_front_face(cmd, front);
                }
                if dc.set(2, topology.as_raw() as u64) {
                    c.device.cmd_set_primitive_topology(cmd, topology);
                }
                let restart = reg(0x28A94) & 1 != 0 && matches!(prim, 3 | 5 | 6) && a.index.is_some();
                if dc.set(3, restart as u64) {
                    c.device.cmd_set_primitive_restart_enable(cmd, restart);
                }
                if dc.set(4, 0) {
                    c.device.cmd_set_rasterizer_discard_enable(cmd, false);
                }
                if dc.set(5, z_test as u64) {
                    c.device.cmd_set_depth_test_enable(cmd, z_test);
                }
                if dc.set(6, z_write as u64) {
                    c.device.cmd_set_depth_write_enable(cmd, z_write);
                }
                // DS_DEPTH_EQUAL_BIAS=<units> (diagnosis): EQUAL tests without
                // depth writes pass a little in front of the stored depth instead
                // (reversed Z when the depth clear value is 0); `always`: they
                // always pass (do the draw's fragments exist at all?).
                let eq_bias = equal_bias().filter(|_| z_test && !z_write && (dctl >> 4) & 7 == 2);
                let reversed = f32r(reg(0x2802C)) == 0.0;
                let zop = match eq_bias {
                    Some(b) if b.is_infinite() => vk::CompareOp::ALWAYS,
                    Some(_) => {
                        if reversed {
                            vk::CompareOp::GREATER_OR_EQUAL
                        } else {
                            vk::CompareOp::LESS_OR_EQUAL
                        }
                    }
                    None => compare_op(dctl >> 4),
                };
                if dc.set(7, zop.as_raw() as u64) {
                    c.device.cmd_set_depth_compare_op(cmd, zop);
                }
                let eq_bias = eq_bias.filter(|b| b.is_finite());
                if dc.set(8, 0) {
                    c.device.cmd_set_depth_bounds_test_enable(cmd, false);
                }
                if dc.set(9, s_test as u64) {
                    c.device.cmd_set_stencil_test_enable(cmd, s_test);
                }
                let back = dctl & (1 << 7) != 0;
                let ffunc = compare_op(dctl >> 8);
                let bfunc = if back { compare_op(dctl >> 20) } else { ffunc };
                let sop = |w: u32, f: vk::CompareOp| (stencil_op(w), stencil_op(w >> 4), stencil_op(w >> 8), f);
                let pack = |o: (vk::StencilOp, vk::StencilOp, vk::StencilOp, vk::CompareOp)| (o.0.as_raw() as u64) | (o.1.as_raw() as u64) << 16 | (o.2.as_raw() as u64) << 32 | (o.3.as_raw() as u64) << 48;
                let fo = sop(sctl, ffunc);
                if dc.set(10, pack(fo)) {
                    c.device.cmd_set_stencil_op(cmd, vk::StencilFaceFlags::FRONT, fo.0, fo.1, fo.2, fo.3);
                }
                let bs = if back { sctl >> 12 } else { sctl };
                let bo = sop(bs, bfunc);
                if dc.set(11, pack(bo)) {
                    c.device.cmd_set_stencil_op(cmd, vk::StencilFaceFlags::BACK, bo.0, bo.1, bo.2, bo.3);
                }
                let rb = if back { refm_bf } else { refm };
                for (slot, face, v) in [
                    (12, vk::StencilFaceFlags::FRONT, refm & 0xFF),
                    (13, vk::StencilFaceFlags::BACK, rb & 0xFF),
                ] {
                    if dc.set(slot, v as u64) {
                        c.device.cmd_set_stencil_reference(cmd, face, v);
                    }
                }
                for (slot, face, v) in [
                    (14, vk::StencilFaceFlags::FRONT, (refm >> 8) & 0xFF),
                    (15, vk::StencilFaceFlags::BACK, (rb >> 8) & 0xFF),
                ] {
                    if dc.set(slot, v as u64) {
                        c.device.cmd_set_stencil_compare_mask(cmd, face, v);
                    }
                }
                for (slot, face, v) in [
                    (16, vk::StencilFaceFlags::FRONT, (refm >> 16) & 0xFF),
                    (17, vk::StencilFaceFlags::BACK, (rb >> 16) & 0xFF),
                ] {
                    if dc.set(slot, v as u64) {
                        c.device.cmd_set_stencil_write_mask(cmd, face, v);
                    }
                }
                // PA_SU_SC_MODE_CNTL.POLY_OFFSET_FRONT/BACK_ENABLE with
                // PA_SU_POLY_OFFSET_* (scale in 1/16 units).
                let poly_offset = mode & (3 << 11) != 0;
                let bias_on = poly_offset || eq_bias.is_some();
                if dc.set(18, bias_on as u64) {
                    c.device.cmd_set_depth_bias_enable(cmd, bias_on);
                }
                let bias = if let (Some(units), false) = (eq_bias, poly_offset) {
                    Some((if reversed { units } else { -units }, 0.0, 0.0))
                } else if poly_offset {
                    let (sr, or) = if mode & (1 << 11) != 0 { (0x28B80, 0x28B84) } else { (0x28B88, 0x28B8C) };
                    Some((f32r(reg(or)), f32r(reg(0x28B7C)), f32r(reg(sr)) / 16.0))
                } else {
                    None
                };
                if let Some((k, clamp, slope)) = bias {
                    let bw = [k.to_bits(), clamp.to_bits(), slope.to_bits()];
                    if dc.bias != Some(bw) {
                        c.device.cmd_set_depth_bias(cmd, k, clamp, slope);
                        dc.bias = Some(bw);
                    }
                }
                // DS_TRACE_DEPTH_DRAWS=<hex depth base> (diagnosis): the first 400 draws into that
                // depth surface, with their depth, cull, polygon offset, clip, clamp and
                // conservative rasterization registers.
                {
                    // (`<hex base>[;<hex base>..]:<flip>`: from that flip on.)
                    static WANT: std::sync::OnceLock<Option<(Vec<u64>, u64)>> = std::sync::OnceLock::new();
                    static SEEN: std::sync::atomic::AtomicU32 = std::sync::atomic::AtomicU32::new(0);
                    let want = WANT.get_or_init(|| {
                        let v = std::env::var("DS_TRACE_DEPTH_DRAWS").ok()?;
                        let (b, f) = v.split_once(':').unwrap_or((v.as_str(), "0"));
                        let bases: Vec<u64> = b.split([';', ',']).filter_map(|x| u64::from_str_radix(x.trim().trim_start_matches("0x"), 16).ok()).collect();
                        (!bases.is_empty()).then(|| (bases, f.trim().parse().unwrap_or(0)))
                    });
                    let flip_now = || crate::hle::libs::videoout::QUEUED_FLIP.load(std::sync::atomic::Ordering::Relaxed) + 1;
                    if let Some((ws, s)) = want.as_ref().filter(|(_, f)| flip_now() >= *f).and_then(|(ws, _)| Some((ws, depth_surface(cp)?))) {
                        let db_render = reg(0x28000);
                        if ws.contains(&s.base) && SEEN.fetch_add(1, std::sync::atomic::Ordering::Relaxed) < 400 {
                            crate::info!(
                                "depth draw {:#x} flip {}: vs {vs_addr:#x} count {} instances {} indexed {} | depth control {dctl:#x} (test {z_test} write {z_write} func {}) clear {} | mode {mode:#x} (cull {:?} front {}) poly offset db fmt {:#x} clamp {} front {} x16 + {} back {} x16 + {} | clip {:#x} | db render {db_render:#x} override {:#x} override2 {:#x} | vport z [{}, {}] x {} {} y {} {} z scale {} offset {} vte {:#x} | cons raster {:#x} | aa config {:#x} | sc mode {:#x}",
                                s.base,
                                crate::hle::libs::videoout::QUEUED_FLIP.load(std::sync::atomic::Ordering::Relaxed) + 1,
                                a.count,
                                a.instances,
                                a.index.is_some(),
                                (dctl >> 4) & 7,
                                f32r(reg(0x2802C)),
                                cull,
                                mode & 4 != 0,
                                reg(0x28B78),
                                f32r(reg(0x28B7C)),
                                f32r(reg(0x28B80)),
                                f32r(reg(0x28B84)),
                                f32r(reg(0x28B88)),
                                f32r(reg(0x28B8C)),
                                reg(0x28810),
                                reg(0x2800C),
                                reg(0x28010),
                                f32r(reg(0x282D0)),
                                f32r(reg(0x282D4)),
                                f32r(reg(0x2843C)),
                                f32r(reg(0x28440)),
                                f32r(reg(0x28444)),
                                f32r(reg(0x28448)),
                                f32r(reg(0x2844C)),
                                f32r(reg(0x28450)),
                                reg(0x28818),
                                reg(0x28C54),
                                reg(0x28BE0),
                                reg(0x28A48),
                            );
                        }
                    }
                }
                if dc.set(20, 1) {
                    c.device.cmd_set_line_width(cmd, 1.0);
                }
                let bc = [reg(0x28414), reg(0x28418), reg(0x2841C), reg(0x28420)];
                if dc.blend != Some(bc) {
                    c.device.cmd_set_blend_constants(cmd, &[f32r(bc[0]), f32r(bc[1]), f32r(bc[2]), f32r(bc[3])]);
                    dc.blend = Some(bc);
                }
            }
            if dyn_before.is_some_and(|b| b != dc.sig()) {
                delta_mismatch(4, vs.addr, ps_addr);
            }
            let base_vertex = a.base_vertex.unwrap_or(reg(0x28408) as i32);
            lap.mark(15);
            // DS_PROFILE_GPU: timestamps around the draw, keyed by its shader.
            let qp = self.slots[self.cur].queries;
            let ti = self.batch_shaders.len() as u32;
            // Full-screen passes (a triangle or quad, one instance) count as
            // large: their cost is the pixels they shade.
            let work = if a.count <= 4 && a.instances == 1 {
                u64::MAX
            } else {
                a.count as u64 * a.instances as u64
            };
            let timed = qp != vk::QueryPool::null() && ti < super::MAX_TIMED && super::stamp_worth(Some(work), 16384);
            untimed = !timed;
            if timed {
                for k in 0..2 {
                    c.device.cmd_write_timestamp(cmd, vk::PipelineStageFlags::BOTTOM_OF_PIPE, qp, super::STAMPS * ti + k);
                }
            }
            // Index buffers: a copy in video memory (the input assembler
            // fetches indices again for every instance; from system memory
            // across PCIe an instanced foliage draw took 4-8 ms, 0.2-0.3 ms
            // from video memory), made by the GPU at the batch's start, else
            // by the CPU now.
            if super::super::census::on() {
                if let Some((addr, size)) = a.index {
                    super::super::census::note(addr, a.count as u64 * size as u64, super::super::census::INDEX);
                }
            }
            let snap = index_home.as_ref().map(|h| (h.0, h.1, 0)).or_else(|| {
                a.index.and_then(|(addr, size)| {
                    let bytes = a.count as u64 * size as u64;
                    self.pre_copy(addr, bytes).or_else(|| self.snapshot(addr, bytes, crate::gpu::tune::snap_index()))
                })
            });
            // A batched pair's draw opens a run (`DrawRun`): one multi-draw
            // whose count, arguments and table the CPU appends to.
            let run_index = if batched {
                match a.index {
                    Some((_, size)) if snap.is_some() => {
                        let (sbuf, soff, sbias) = snap.unwrap();
                        Some(Some((sbuf, soff + sbias, size)))
                    }
                    Some((addr, size)) => {
                        let bytes = a.count as u64 * size as u64;
                        let Some((buffer, start, _)) = super::span(addr, bytes) else {
                            return Err(format!("index buffer {addr:#x}+{bytes:#x} is not mapped GPU memory (draw skipped)"));
                        };
                        if let Some(s) = super::CP_WRITES.latest(&[(addr, bytes)]) {
                            self.submit_after = self.submit_after.max(s);
                        }
                        Some(Some((buffer, addr - start, size)))
                    }
                    None => Some(None),
                }
            } else {
                None
            };
            if let Some(index) = run_index {
                let appendable = !(clear_z || clear_s);
                // A GPU-resolved pair's run: its later draws are appended from
                // this draw's bound SGPR data.
                let gpu_run = gpu_pair.as_ref().map(|d| Box::new(GpuDrawRun::new((vs_addr, ps_addr), d, &vs, ps.as_ref().unwrap(), [cp.reg(Space::Sh, 0x8B), cp.reg(Space::Sh, 0x0B)], base_slot, self.fe2_run_want())));
                self.draw_run_open(cmd, layout, a, base_vertex, index, sgpr_ring, sgpr_at, appendable, gpu_run)?;
            } else {
            match a.index {
                Some((_, size)) if snap.is_some() => {
                    let (sbuf, soff, sbias) = snap.unwrap();
                    super::INDEX_SNAPS.fetch_add(1, Ordering::Relaxed);
                    super::INDEX_SNAP_BYTES.fetch_add(a.count as u64 * size as u64, Ordering::Relaxed);
                    let ty = if size == 4 { vk::IndexType::UINT32 } else { vk::IndexType::UINT16 };
                    c.device.cmd_bind_index_buffer(cmd, sbuf, soff + sbias, ty);
                    c.device.cmd_draw_indexed(cmd, a.count, a.instances, 0, base_vertex, a.first_instance);
                }
                Some((addr, size)) => {
                    let bytes = a.count as u64 * size as u64;
                    let span = super::span(addr, bytes);
                    let Some((buffer, start, _)) = span else {
                        return Err(format!("index buffer {addr:#x}+{bytes:#x} is not mapped GPU memory (draw skipped)"));
                    };
                    let ty = if size == 4 { vk::IndexType::UINT32 } else { vk::IndexType::UINT16 };
                    if let Some(s) = super::CP_WRITES.latest(&[(addr, bytes)]) {
                        self.submit_after = self.submit_after.max(s);
                    }
                    c.device.cmd_bind_index_buffer(cmd, buffer, addr - start, ty);
                    c.device.cmd_draw_indexed(cmd, a.count, a.instances, 0, base_vertex, a.first_instance);
                }
                None => c.device.cmd_draw(cmd, a.count, a.instances, base_vertex as u32, a.first_instance),
            }
            }
            if timed {
                for k in 2..4 {
                    c.device.cmd_write_timestamp(cmd, vk::PipelineStageFlags::BOTTOM_OF_PIPE, qp, super::STAMPS * ti + k);
                }
                self.batch_shaders.push(super::DRAW_KEY | if ps_addr != 0 { ps_addr } else { vs.addr });
            } else {
                // The batch's commands (its GPU time by kind: `note_batch_by`,
                // the hitch reports; slow and lost batches name them).
                self.batch_shaders.push(super::DRAW_KEY | if ps_addr != 0 { ps_addr } else { vs.addr });
            }
        }
        lap.mark(16);
        // Another draw like it can be recorded on the state this one left
        // (`draw_repeat`): GPU-read arguments, indices and arguments in
        // homes, no buffer writes, no timestamps around it.
        let no_writes = bounds.iter().all(|b| b.writes.is_empty());
        // (Not after a batched draw: its pipeline's vertex stage reads its
        // SGPR data from the run's table by draw index.)
        let repeat = match (&index_home, a.index) {
            _ if batched => None,
            (Some(ih), Some((_, size))) if a.base_inst.is_some() && untimed && no_writes && self.recording && super::super::tune::draw_repeat() && !sync_mode() => {
                Some(Repeat { batch: self.batch_count, uses: 0, index: (0, size, 0), index_buf: ih.0, index_off: ih.1 })
            }
            _ => None,
        };
        // The stages' images and samplers for a buffer-delta draw after this
        // one: when both came from the binding memos for the same generations.
        if has_sig {
            let g = match notes {
                [Some((g0, true)), Some((g1, true))] if g0 == g1 => Some(g0),
                [Some((g0, true)), None] if ps.is_none() => Some(g0),
                _ => None,
            };
            // (Not again when every stage took them over: they are those.)
            if g.is_some() && reused < bounds.len() {
                for (k, b) in bounds.iter().enumerate() {
                    let s = &mut self.delta.bind[k];
                    s.imgs.clear();
                    s.imgs.extend_from_slice(&b.imgs);
                    s.images.clear();
                    s.images.extend_from_slice(&b.images);
                    s.stencil_views.clear();
                    s.stencil_views.extend_from_slice(&b.stencil_views);
                }
            }
            self.delta.bind_gens = g;
        }
        for b in bounds.drain(..) {
            if !b.sized_writes.is_empty() {
                // A held executor notes them for itself.
                if self.held {
                    self.held_note(&[], &b.sized_writes, None);
                } else {
                    crate::gpu::images::with(|im| im.note_overwrites(&b.sized_writes));
                }
            }
            for &(w, n) in &b.writes {
                self.note_write(w, n);
            }
            self.recycle(b);
        }
        self.draw_bounds = bounds;
        // (After everything this draw recorded.)
        self.repeat = repeat.map(|r| Repeat { uses: self.cmd_uses.get(), ..r });
        self.last_draw_end = (self.batch_count, self.cmd_uses.get());
        // What a buffer-delta draw after this one takes over.
        if expect.is_some() {
            DELTA_VERIFY[0].fetch_add(1, Ordering::Relaxed);
        }
        if delta.is_some() {
            self.delta.at = Some(self.last_draw_end);
        } else if has_sig && !fixed.writes && !fixed.hazard && !clear_z && !clear_s && !touches {
            let d = &mut self.delta;
            d.textures.clear();
            d.textures.extend(self.tex_scratch.iter().map(|x| x.0));
            d.used.clear();
            d.used.extend_from_slice(&self.keys_scratch);
            std::mem::swap(&mut d.images, &mut sig);
            (d.keys, d.index, d.fixed) = (keys, a.index.is_some(), Some(fixed));
            d.at = Some(self.last_draw_end);
        }
        self.delta_sig = sig;
        if let Some(r) = self.draw_run.as_mut().filter(|r| r.batch == self.batch_count && r.fresh) {
            r.uses = self.cmd_uses.get();
            r.fresh = false;
        }
        lap.mark(17);
        if super::laps() && !first {
            let c = &DELTA_COST[kind][was_delta as usize];
            c[0].fetch_add(lap.local.iter().sum::<u64>(), Ordering::Relaxed);
            c[1].fetch_add(1, Ordering::Relaxed);
            let depth_only = fixed.colors.iter().all(Option::is_none);
            let s = &STEPS_SPLIT[(kind == 2) as usize * 4 + depth_only as usize * 2 + was_delta as usize];
            for (k, v) in lap.local[..18].iter().enumerate() {
                s[k].fetch_add(*v, Ordering::Relaxed);
            }
            s[18].fetch_add(1, Ordering::Relaxed);
        }
        if super::laps() {
            // The pair census: a draw that was not a buffer-delta draw whose pair one of the
            // last draws of this rendering instance had (since the last dispatch).
            let pair = (vs_addr, ps_addr);
            let n = self.pair_ring_n as usize;
            if !was_delta && !first {
                let pos = (0..n.min(8)).find(|&k| self.pair_ring[(n - 1 - k) % 8] == pair);
                PAIR_CENSUS[0].fetch_add(1, Ordering::Relaxed);
                let slot = match pos {
                    Some(0) => 1,
                    Some(k) if k < 2 => 2,
                    Some(k) if k < 4 => 3,
                    Some(_) => 4,
                    None if n == 0 => 5,
                    None => 6,
                };
                PAIR_CENSUS[slot].fetch_add(1, Ordering::Relaxed);
            }
            if n == 0 || self.pair_ring[(n - 1) % 8] != pair {
                self.pair_ring[n % 8] = pair;
                self.pair_ring_n += 1;
            }
        }
        if super::profiling() {
            super::note_phases(&super::DRAW_PHASES, [t_prep, t_pipe, t_bind, since(t0)]);
            super::note_place(self.deferred, self.batch_shaders.len().saturating_sub(1), [t_prep, t_pipe, t_bind, since(t0)]);
        }
        if super::laps() {
            if first {
                super::DRAW_STEPS_FIRST[kind][18].fetch_add(1, Ordering::Relaxed);
            } else {
                super::DRAW_STEPS[kind][18].fetch_add(1, Ordering::Relaxed);
            }
            if gpu_open {
                let g = &super::DRAW_STEPS_GPU[kind];
                for (c, &ns) in g.iter().zip(lap.local.iter().take(18)) {
                    c.fetch_add(ns, Ordering::Relaxed);
                }
                g[18].fetch_add(1, Ordering::Relaxed);
            }
            lap.slow(&self.name, || format!("vs {vs_addr:#x} ps {ps_addr:#x}{}", if first { ", first of its batch" } else { "" }));
        }
        Ok(true)
    }
}

/// `DS_GPU_VERIFY=1`: GPU-resolved draws appended are also prepared in
/// full and compared (`Exec::gpu_draw_verify`; diagnosis).
fn gpu_verify() -> bool {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ON.get_or_init(|| std::env::var("DS_GPU_VERIFY").ok().as_deref() == Some("1"))
}

/// Draws one run holds at most.
const DRAW_RUN_MAX: u32 = 64;

/// An open batched draw run (a GPU-resolved pair's): one multi-draw
/// (vkCmdDraw[Indexed]IndirectCount) reading its count, arguments and table
/// from the SGPR data ring, which draws right after it with the same
/// context state, translations, images and index buffer append to.
pub(crate) struct DrawRun {
    batch: u64,
    /// `cmd_uses` right after it (or its last append) was recorded; `fresh`
    /// until the full draw that opened it was recorded to its end.
    uses: u64,
    fresh: bool,
    ring: vk::Buffer,
    count: *mut u32,
    args: *mut u32,
    table: *mut u32,
    n: u32,
    max: u32,
    /// Its index buffer, index size and the offset the buffer is bound at
    /// (`draw_run_open`); None: not indexed.
    index: Option<(vk::Buffer, u32, u64)>,
    appendable: bool,
    /// A GPU-resolved pair's run (`draw_gpu_append`).
    gpu: Option<Box<GpuDrawRun>>,
    /// Its number in the run log (`DS_GPU_CHECKPOINTS=1`, `RUN_LOG`); 0: none.
    seq: u64,
}

/// `DS_GPU_CHECKPOINTS=1`: a GPU-resolved run as recorded (`RUN_LOG`): its draws' sizes and
/// where their indices and vertices start against the run's index binding.
struct RunRec {
    seq: u64,
    exec: String,
    batch: u64,
    pair: (u64, u64),
    n: u32,
    max: u32,
    /// Index size and the binding's base offset (None: not indexed).
    index: Option<(u32, u64)>,
    /// The largest draw (count x instances) and its place in the run.
    big: u64,
    big_at: u32,
    /// First indices: the lowest, and the highest first + count.
    first_lo: u32,
    end_hi: u64,
    vertex_lo: i32,
    vertex_hi: i32,
    inst_hi: u32,
}

/// Numbers GPU-resolved runs for the checkpoints (`mark::RUN`).
static RUN_SEQ: AtomicU64 = AtomicU64::new(0);
/// The last `RUN_LOG_MAX` GPU-resolved runs (`DS_GPU_CHECKPOINTS=1`), printed when GPU work
/// stops (`run_log_report`): with the checkpoints' run numbers they name what a hung run held.
static RUN_LOG: Mutex<std::collections::VecDeque<RunRec>> = Mutex::new(std::collections::VecDeque::new());
const RUN_LOG_MAX: usize = 512;
/// Appended draws refused for their size (count x instances `HUGE_DRAW` or more): the full
/// path logs them (`note_huge_draw`).
static APPEND_HUGE: AtomicU64 = AtomicU64::new(0);

/// A draw appended to run `seq` (draw `at` of it): its size and starts in its record.
fn run_log_note(seq: u64, a: &DrawArgs, first: u32, base_vertex: i32, at: u32) {
    let mut g = RUN_LOG.lock();
    if let Some(r) = g.iter_mut().rev().find(|r| r.seq == seq) {
        let size = a.count as u64 * a.instances.max(1) as u64;
        if size > r.big {
            (r.big, r.big_at) = (size, at);
        }
        r.n = r.n.max(at + 1);
        r.first_lo = r.first_lo.min(first);
        r.end_hi = r.end_hi.max(first as u64 + a.count as u64);
        r.vertex_lo = r.vertex_lo.min(base_vertex);
        r.vertex_hi = r.vertex_hi.max(base_vertex);
        r.inst_hi = r.inst_hi.max(a.instances);
    }
}

/// The last GPU-resolved runs recorded (`DS_GPU_CHECKPOINTS=1`), oldest first ("" when none):
/// all `RUN_LOG_MAX`, so the checkpoints' run numbers find theirs.
pub(super) fn run_log_report() -> String {
    let g = RUN_LOG.lock();
    g.iter()
        .map(|r| {
            let index = r.index.map_or("not indexed".to_string(), |(s, base)| format!("{s}-byte indices bound at {base:#x}, first {}..{} (end; bytes {:#x})", r.first_lo, r.end_hi, r.end_hi * s as u64));
            format!(
                "      run #{} on {} batch {}: vs {:#x} ps {:#x}, {} of {} draws, the largest {} vertex invocations (draw {}), {index}, vertex offsets {}..{}, instances up to {}",
                r.seq, r.exec, r.batch, r.pair.0, r.pair.1, r.n, r.max, r.big, r.big_at, r.vertex_lo, r.vertex_hi, r.inst_hi
            )
        })
        .collect::<Vec<_>>()
        .join("\n")
}

/// A GPU-resolved pair's open run (`gpures`, stage G4): per stage (vertex,
/// pixel) the SGPR data its first draw bound (the run roots' descriptors and
/// data in it), its user-data dword count and SH register RSRC2, the run
/// roots with their values, the per-item roots, the preparation (for the
/// ranges its per-item buffers reach) and the first draw's stages (their
/// images' pending work is checked per appended draw).
pub(crate) struct GpuDrawRun {
    pair: (u64, u64),
    templates: [Vec<u32>; 2],
    users: [usize; 2],
    rsrc2: [u32; 2],
    roots: [Vec<(super::super::gpures::Root, u64)>; 2],
    items: [Vec<super::super::gpures::Root>; 2],
    stages: [Prepared; 2],
    /// The user-data words the run's own descriptors take as they are
    /// (`Decided::run_user`), and the first draw's user data.
    run_user: [u32; 2],
    first_user: [[u32; 32]; 2],
    /// The vertex stage's user word that takes the start instance from
    /// `gl_BaseInstance` (`DrawArgs::base_inst`): its translation reads it
    /// there, so every draw of the run must.
    base_inst: Option<u16>,
    /// Front end 2: the epoch its first draw's checks needed
    /// (`Exec::fe2_run_want`); later draws append while no check needs a
    /// later one.
    fe2_want: u32,
}

impl GpuDrawRun {
    fn new(pair: (u64, u64), d: &super::super::gpures::DrawDecided, vs: &Prepared, ps: &Prepared, rsrc2: [u32; 2], base_inst: Option<u16>, fe2_want: u32) -> GpuDrawRun {
        let roots = |k: usize, p: &Prepared| d.stages[k].run_roots.iter().map(|&r| (r, super::super::gpures::root_value(r, &p.user))).collect();
        GpuDrawRun {
            pair,
            templates: [vs.sgpr_data.clone(), ps.sgpr_data.clone()],
            users: [vs.tr.spec.user_sgprs as usize, ps.tr.spec.user_sgprs as usize],
            rsrc2,
            roots: [roots(0, vs), roots(1, ps)],
            items: [d.stages[0].item_roots.clone(), d.stages[1].item_roots.clone()],
            stages: [vs.clone(), ps.clone()],
            run_user: [d.stages[0].run_user, d.stages[1].run_user],
            first_user: [std::array::from_fn(|i| vs.user.get(i).copied().unwrap_or(0)), std::array::from_fn(|i| ps.user.get(i).copied().unwrap_or(0))],
            base_inst,
            fe2_want,
        }
    }
}

/// [runs opened, draws appended, runs that could not append (opened
/// without room), appends refused (another index buffer, images, a
/// recording meanwhile...); runs that could not append by why: a clear,
/// no room in the ring, indices at an offset not a multiple of their size].
static BATCH_DRAWS: [AtomicU64; 7] = [const { AtomicU64::new(0) }; 7];

/// The alignment of a draw run's index binding when first indices from
/// offset 0 would pass u32 (`draw_run_open`).
const INDEX_BASE_ALIGN: u64 = 1 << 30;

/// The first index of a draw whose indices are at byte `o` of buffer `b`
/// (`size` bytes each, `count` of them) in a run whose index binding is
/// `run` (`DrawRun::index`); None: it cannot join the run.
fn run_first_index(run: Option<(vk::Buffer, u32, u64)>, b: vk::Buffer, o: u64, size: u32, count: u32) -> Option<u32> {
    let (rb, rsize, base) = run?;
    if rb != b || rsize != size || o < base || (o - base) % size as u64 != 0 {
        return None;
    }
    let first = (o - base) / size as u64;
    (first + count as u64 <= u32::MAX as u64).then_some(first as u32)
}

/// Status part: `BATCH_DRAWS`.
pub(super) fn batch_line() -> String {
    let v: Vec<u64> = BATCH_DRAWS.iter().map(|c| c.load(Ordering::Relaxed)).collect();
    if v[0] == 0 {
        return String::new();
    }
    format!(
        "    batched draws (GPU-resolved pairs): {} runs, {} draws appended ({:.1} per run), {} runs without room to append (a clear {}, no room in the ring {}, indices at an offset not a multiple of their size {}), {} appends refused ({} of them for their size)\n",
        v[0],
        v[1],
        (v[0] + v[1]) as f64 / v[0] as f64,
        v[2],
        v[4],
        v[5],
        v[6],
        v[3],
        APPEND_HUGE.load(Ordering::Relaxed)
    )
}

impl Exec {
    /// Records a batched pair's draw as a new run: its count, arguments and
    /// table in the SGPR data ring (`ring`, which the stages bind whole), the
    /// table's offset as push constant byte 0 (the vertex stage reads its
    /// draw's entry by DrawIndex), the index buffer bound at a base, offset 0
    /// or the 1 GiB boundary below the draw's indices when first indices
    /// (u32) do not reach them from 0 (front end 2's mirror spans 32 GiB),
    /// each draw's first index its byte offset from there / size (at its own
    /// offset and not appendable if not a multiple), one indirect-count draw.
    #[allow(clippy::too_many_arguments)]
    fn draw_run_open(&mut self, cmd: vk::CommandBuffer, layout: vk::PipelineLayout, a: &DrawArgs, base_vertex: i32, index: Option<(vk::Buffer, u64, u32)>, ring: vk::Buffer, at: [u64; 2], appendable: bool, gpu: Option<Box<GpuDrawRun>>) -> Result<(), String> {
        let which = if self.slots[self.cur].rings[1].is_some_and(|r| r.0 == ring) { 1 } else { 0 };
        let rec = if index.is_some() { 5u64 } else { 4 };
        let mut max = DRAW_RUN_MAX as u64;
        let off = loop {
            let bytes = 16 + 4 * (rec + rdna::translate::BATCH_DRAW_ENTRY_DWORDS as u64) * max;
            if let Some(o) = self.ring_alloc(which, bytes) {
                break o;
            }
            if max == 1 {
                return Err("SGPR data ring exhausted (batched draw run)".into());
            }
            max = 1;
        };
        let Some((_, ring_ptr)) = self.slots[self.cur].rings[which] else { return Err("no SGPR data ring".into()) };
        let args_off = off + 16;
        let table_off = args_off + 4 * rec * max;
        let (index, first, at_base) = match index {
            Some((b, o, size)) if o % size as u64 == 0 => {
                let base = if o / size as u64 + a.count as u64 <= u32::MAX as u64 { 0 } else { o & !(INDEX_BASE_ALIGN - 1) };
                (Some((b, base, size)), ((o - base) / size as u64) as u32, true)
            }
            Some((b, o, size)) => (Some((b, o, size)), 0, false),
            None => (None, 0, true),
        };
        if !appendable {
            BATCH_DRAWS[4].fetch_add(1, Ordering::Relaxed);
        } else if max == 1 {
            BATCH_DRAWS[5].fetch_add(1, Ordering::Relaxed);
        } else if !at_base {
            BATCH_DRAWS[6].fetch_add(1, Ordering::Relaxed);
        }
        let appendable = appendable && max > 1 && at_base;
        // (`DS_GPU_CHECKPOINTS=1`: a GPU-resolved run gets a number, its checkpoint before its
        // multi-draw, and a record; a hang then names the run and what it held.)
        let seq = if gpu.is_some() && crate::gpu::vk::checkpoints_on() {
            let s = RUN_SEQ.fetch_add(1, Ordering::Relaxed) + 1;
            crate::gpu::vk::checkpoint(cmd, crate::gpu::vk::mark::RUN, s);
            let mut g = RUN_LOG.lock();
            if g.len() >= RUN_LOG_MAX {
                g.pop_front();
            }
            g.push_back(RunRec {
                seq: s,
                exec: self.name.clone(),
                batch: self.batch_count,
                pair: gpu.as_ref().map_or((0, 0), |g| g.pair),
                n: 1,
                max: max as u32,
                index: index.map(|(_, base, size)| (size, base)),
                big: a.count as u64 * a.instances.max(1) as u64,
                big_at: 0,
                first_lo: first,
                end_hi: first as u64 + a.count as u64,
                vertex_lo: base_vertex,
                vertex_hi: base_vertex,
                inst_hi: a.instances,
            });
            s
        } else {
            0
        };
        unsafe {
            let count = ring_ptr.add(off as usize) as *mut u32;
            let args = ring_ptr.add(args_off as usize) as *mut u32;
            let table = ring_ptr.add(table_off as usize) as *mut u32;
            if index.is_some() {
                for (k, v) in [a.count, a.instances, first, base_vertex as u32, a.first_instance].into_iter().enumerate() {
                    args.add(k).write(v);
                }
            } else {
                for (k, v) in [a.count, a.instances, base_vertex as u32, a.first_instance].into_iter().enumerate() {
                    args.add(k).write(v);
                }
            }
            table.write((at[0] / 4) as u32);
            table.add(1).write((at[1] / 4) as u32);
            count.write(1);
            let c = ctx();
            let t = (table_off / 4) as u32;
            c.device.cmd_push_constants(cmd, layout, super::sgpr_pc_ranges(vk::ShaderStageFlags::VERTEX, true)[0].stage_flags, 0, &t.to_le_bytes());
            match index {
                Some((b, o, size)) => {
                    let ty = if size == 4 { vk::IndexType::UINT32 } else { vk::IndexType::UINT16 };
                    c.device.cmd_bind_index_buffer(cmd, b, o, ty);
                    c.device.cmd_draw_indexed_indirect_count(cmd, ring, args_off, ring, off, max as u32, 20);
                }
                None => c.device.cmd_draw_indirect_count(cmd, ring, args_off, ring, off, max as u32, 16),
            }
            if index.is_some() {
                // The index binding moved off the draw before's.
                self.repeat = None;
            }
            self.draw_run = Some(DrawRun {
                batch: self.batch_count,
                uses: 0,
                fresh: true,
                ring,
                count,
                args,
                table,
                n: 1,
                max: max as u32,
                index: index.map(|(b, base, size)| (b, size, base)),
                appendable,
                gpu,
                seq,
            });
        }
        BATCH_DRAWS[0].fetch_add(1, Ordering::Relaxed);
        if !appendable {
            BATCH_DRAWS[2].fetch_add(1, Ordering::Relaxed);
        }
        Ok(())
    }

    /// A GPU-resolved pair's draw right after the last draw of its open run
    /// (`GpuDrawRun`) with the run's context state: when its run roots hold
    /// the run's values, its per-item chains' pages and ranges are in the
    /// page table, the run's images have no pending work and its indices lie
    /// in the run's index buffer, its arguments, table entry and SGPR data
    /// (the run's first draw's, with its own user data) are appended to the
    /// run, before any preparation: no stage is resolved, synced or bound on
    /// the CPU. None: the full path records it (a new run).
    pub(super) fn draw_gpu_append(&mut self, cp: &Cp, a: &DrawArgs, vs_addr: u64, ps_addr: u64) -> Result<Option<bool>, String> {
        let Some(mut run) = self.draw_run.take() else { return Ok(None) };
        let r = self.draw_gpu_append_to(&mut run, cp, a, vs_addr, ps_addr);
        self.draw_run = Some(run);
        r
    }

    fn draw_gpu_append_to(&mut self, r: &mut DrawRun, cp: &Cp, a: &DrawArgs, vs_addr: u64, ps_addr: u64) -> Result<Option<bool>, String> {
        let stats = &super::GPU_DRAW_APPENDS;
        // Step times of appends (`DS_STEP_LAPS`): checks, imports, image
        // hazards, indices, SGPR data and arguments.
        let mut t = super::laps().then(std::time::Instant::now);
        let mut laps = [0u64; 5];
        let mut mark = |k: usize| {
            if let Some(t0) = t.as_mut() {
                let now = std::time::Instant::now();
                laps[k] = (now - *t0).as_nanos() as u64;
                *t0 = now;
            }
        };
        let at = (self.batch_count, self.cmd_uses.get());
        let Some(g) = r.gpu.as_ref() else { return Ok(None) };
        if g.pair != (vs_addr, ps_addr) || !r.appendable || r.fresh || (r.batch, r.uses) != at || r.n >= r.max || r.index.is_some() != a.index.is_some() || a.base_inst != g.base_inst || self.rendering.is_none() || self.stopped || [cp.reg(Space::Sh, 0x8B), cp.reg(Space::Sh, 0x0B)] != g.rsrc2 {
            stats[3].fetch_add(1, Ordering::Relaxed);
            return Ok(None);
        }
        let vs_user: [u32; 32] = std::array::from_fn(|i| cp.reg(Space::Sh, 0x8C + i as u32));
        let ps_user: [u32; 32] = std::array::from_fn(|i| cp.reg(Space::Sh, 0x0C + i as u32));
        let users: [&[u32]; 2] = [&vs_user[..g.users[0].min(32)], &ps_user[..g.users[1].min(32)]];
        if (0..2).any(|k| g.roots[k].iter().any(|&(root, v)| super::super::gpures::root_value(root, users[k]) != v) || (0..32).any(|j| g.run_user[k] & (1 << j) != 0 && users[k].get(j).copied().unwrap_or(0) != g.first_user[k][j])) {
            stats[1].fetch_add(1, Ordering::Relaxed);
            return Ok(None);
        }
        // The guest memory the run's first draw was resolved from, unchanged:
        // a CP write between draws (no command recorded) rewrites a run root's
        // block in place (a 4x4 transform at the same pointer, `DS_GPU_VERIFY`),
        // which the copied SGPR data would miss.
        if !shaders::words_unchanged(&g.stages[0]) || !shaders::words_unchanged(&g.stages[1]) {
            stats[1].fetch_add(1, Ordering::Relaxed);
            return Ok(None);
        }
        // (A draw of absurd size goes to the full path, which logs it: `APPEND_HUGE`.)
        if a.count as u64 * a.instances.max(1) as u64 >= HUGE_DRAW {
            APPEND_HUGE.fetch_add(1, Ordering::Relaxed);
            stats[3].fetch_add(1, Ordering::Relaxed);
            return Ok(None);
        }
        mark(0);
        if super::super::fe2::on() {
            // Front end 2 (stage F3): the run's bindings were checked for the
            // view its first draw had; this draw's per-item chains are placed
            // now.
            if self.fe2_want(crate::gpu::precache::epoch() as u32) > g.fe2_want {
                super::F3_REFUSED[super::F3_VIEW].fetch_add(1, Ordering::Relaxed);
                stats[3].fetch_add(1, Ordering::Relaxed);
                return Ok(None);
            }
            for k in 0..2 {
                let st = &g.stages[k];
                if self.fe2_gpu_ranges(&st.found, st.addr, users[k]).is_err() {
                    stats[3].fetch_add(1, Ordering::Relaxed);
                    return Ok(None);
                }
            }
        } else {
            for k in 0..2 {
                let st = &g.stages[k];
                if !super::gpu_pages_mapped(&g.items[k], users[k]) || !super::gpu_res_imported(&st.found, st.addr, st.tr.spec.gpu_res, users[k]) {
                    stats[3].fetch_add(1, Ordering::Relaxed);
                    return Ok(None);
                }
            }
        }
        if gpu_verify() {
            self.gpu_draw_verify(cp, g);
        }
        mark(1);
        // Pending work on the run's images, as for a batched draw (the
        // textures' CPU writes were checked by the run's first draw).
        let mut used = std::mem::take(&mut self.keys_scratch);
        used.clear();
        if let Some(rd) = &self.rendering {
            used.extend(rd.colors.iter().flatten().chain(rd.depth.iter()).map(|t| t.key));
        }
        let pending = super::image_hazards(&[&g.stages[0], &g.stages[1]], &mut used);
        let pending = self.held_hit(&used) || pending;
        self.keys_scratch = used;
        if pending {
            stats[3].fetch_add(1, Ordering::Relaxed);
            return Ok(None);
        }
        mark(2);
        // Indices, placed as the full path places them: in the run's buffer.
        let base_vertex = a.base_vertex.unwrap_or(ctx_reg(cp, 0x28408) as i32);
        let first = match a.index {
            Some((addr, size)) => {
                let bytes = a.count as u64 * size as u64;
                let h = self.home_bind(addr, bytes, false, super::HomeUse::Index);
                if h.is_none() {
                    self.sync_for(&[(addr, bytes)], SYNC_DRAW_INDICES);
                }
                if self.stopped {
                    return Err(super::STOPPED.into());
                }
                if let Some(h) = &h {
                    if !self.keep_home(h.0, h.3.as_ref()) {
                        stats[2].fetch_add(1, Ordering::Relaxed);
                        return Ok(None);
                    }
                }
                self.flush_resident(Some(&[(addr, bytes)]))?;
                let snap = h.as_ref().map(|h| (h.0, h.1, 0)).or_else(|| self.pre_copy(addr, bytes).or_else(|| self.snapshot(addr, bytes, crate::gpu::tune::snap_index())));
                let (b, o) = match snap {
                    Some((b, off, bias)) => (b, off + bias),
                    None => {
                        let Some((buffer, start, _)) = super::span(addr, bytes) else {
                            stats[2].fetch_add(1, Ordering::Relaxed);
                            return Ok(None);
                        };
                        if let Some(s) = super::CP_WRITES.latest(&[(addr, bytes)]) {
                            self.submit_after = self.submit_after.max(s);
                        }
                        (buffer, addr - start)
                    }
                };
                let Some(first) = run_first_index(r.index, b, o, size, a.count) else {
                    stats[2].fetch_add(1, Ordering::Relaxed);
                    return Ok(None);
                };
                first
            }
            None => 0,
        };
        if (self.batch_count, self.cmd_uses.get()) != at || self.rendering.is_none() || self.stopped {
            stats[3].fetch_add(1, Ordering::Relaxed);
            return Ok(None);
        }
        mark(3);
        // Both stages' SGPR data: the run's first draw's with this draw's
        // user data, in the run's ring (the stages bind it whole).
        let which = if self.slots[self.cur].rings[1].is_some_and(|x| x.0 == r.ring) { 1 } else { 0 };
        let mut items = [0u64; 2];
        for k in 0..2 {
            let bytes = 4 * g.templates[k].len() as u64;
            let Some(off) = self.ring_alloc(which, bytes) else {
                stats[3].fetch_add(1, Ordering::Relaxed);
                return Ok(None);
            };
            let Some((ring, rp)) = self.slots[self.cur].rings[which] else { return Ok(None) };
            if ring != r.ring {
                stats[3].fetch_add(1, Ordering::Relaxed);
                return Ok(None);
            }
            unsafe {
                let dst = rp.add(off as usize) as *mut u32;
                std::ptr::copy_nonoverlapping(g.templates[k].as_ptr(), dst, g.templates[k].len());
                std::ptr::copy_nonoverlapping(users[k].as_ptr(), dst, users[k].len().min(g.templates[k].len()));
            }
            items[k] = off;
        }
        unsafe {
            let n = r.n as usize;
            if r.index.is_some() {
                let e = r.args.add(5 * n);
                for (k, v) in [a.count, a.instances, first, base_vertex as u32, a.first_instance].into_iter().enumerate() {
                    e.add(k).write(v);
                }
            } else {
                let e = r.args.add(4 * n);
                for (k, v) in [a.count, a.instances, base_vertex as u32, a.first_instance].into_iter().enumerate() {
                    e.add(k).write(v);
                }
            }
            let t = r.table.add(rdna::translate::BATCH_DRAW_ENTRY_DWORDS as usize * n);
            t.write((items[0] / 4) as u32);
            t.add(1).write((items[1] / 4) as u32);
            r.n += 1;
            r.count.write(r.n);
        }
        if r.seq != 0 {
            run_log_note(r.seq, a, first, base_vertex, r.n - 1);
        }
        self.batch_shaders.push(super::DRAW_KEY | ps_addr);
        stats[0].fetch_add(1, Ordering::Relaxed);
        DRAWN.fetch_add(1, Ordering::Relaxed);
        mark(4);
        if t.is_some() {
            let l = &super::GPU_APPEND_LAPS;
            l[0].fetch_add(1, Ordering::Relaxed);
            for (k, ns) in laps.iter().enumerate() {
                l[1 + k].fetch_add(*ns, Ordering::Relaxed);
            }
        }
        Ok(Some(true))
    }

    /// `DS_GPU_VERIFY=1` (diagnosis): an appended draw's stages prepared as
    /// the full path would prepare them, compared with what the append takes
    /// from the run's first draw: the data of the loads the CPU resolves for
    /// the run, the run's descriptors, and per item a formatted buffer's
    /// format and whether its data is GPU-written. The first differences are
    /// logged with the pair (counted in `GPU_VERIFY`).
    fn gpu_draw_verify(&mut self, cp: &Cp, g: &GpuDrawRun) {
        let masks = [(g.stages[0].tr.spec.gpu_loads, g.stages[0].tr.spec.gpu_res), (g.stages[1].tr.spec.gpu_loads, g.stages[1].tr.spec.gpu_res)];
        let fresh = {
            let _b = shaders::batched_draw(true);
            let _m = shaders::gpu_draw_masks(Some(masks));
            let vs = match g.base_inst {
                Some(s) => shaders::with_base_instance(s, || shaders::prepare_ngg(cp)),
                None => shaders::prepare_ngg(cp),
            };
            match (vs, shaders::prepare_ps(cp)) {
                (Ok(v), Ok(p)) => [v, p],
                _ => return,
            }
        };
        let v = &super::GPU_VERIFY;
        v[0].fetch_add(1, Ordering::Relaxed);
        let mut found: Vec<String> = Vec::new();
        for k in 0..2 {
            let (first, now) = (&g.stages[k], &fresh[k]);
            if !std::sync::Arc::ptr_eq(&first.found.tr, &now.found.tr) && first.tr.key != now.tr.key {
                // Which part of the specialization differs.
                let (a, b) = (&first.tr.spec, &now.tr.spec);
                let mut what: Vec<String> = Vec::new();
                for (i, (x, y)) in a.resources.iter().zip(b.resources.iter()).enumerate() {
                    if x != y {
                        what.push(format!("resource {i} {x:?} -> {y:?}{}", if i < 64 && a.gpu_res & (1u64 << i) != 0 { " (per item)" } else { "" }));
                    }
                }
                if a.resources.len() != b.resources.len() {
                    what.push("resource count".into());
                }
                let mut a2 = a.clone();
                a2.resources = b.resources.clone();
                if a2 != *b {
                    what.push(format!("other fields: {a2:?} -> {b:?}"));
                }
                found.push(format!("stage {k} ({:#x}) translated differently: {}", first.addr, what.join(", ")));
                continue;
            }
            // Load data the CPU resolves for the run.
            let mut at = rdna::translate::SGPR_DATA_USER as usize;
            for (j, l) in first.tr.plan.loads.iter().enumerate() {
                let n = l.dwords as usize;
                if j < 64 && first.tr.spec.gpu_loads & (1u64 << j) == 0 {
                    let (a, b) = (g.templates[k].get(at..at + n), now.sgpr_data.get(at..at + n));
                    if a != b {
                        found.push(format!("stage {k} ({:#x}) run load {j} ({:?}+{:#x} x{n}) differs: run {:x?}, now {:x?}", first.addr, l.ptr, l.offset, a, b));
                    }
                }
                at += n;
            }
            // The run's descriptors; per-item buffers' formats and data.
            for (i, (r0, r1)) in first.resources.iter().zip(now.resources.iter()).enumerate() {
                let item = i < 64 && first.tr.spec.gpu_res & (1u64 << i) != 0;
                if !item && r0 != r1 {
                    found.push(format!("stage {k} ({:#x}) run resource {i} differs: run {r0:?}, now {r1:?}", first.addr));
                }
                if let (true, shaders::Resolved::Buffer(vb)) = (item, r1) {
                    if first.tr.plan.resources[i].formatted {
                        if let Some(rdna::translate::ResSpec::Buffer { format, dst_sel }) = first.tr.spec.resources.get(i) {
                            if vb.format != *format || vb.dst_sel != *dst_sel {
                                found.push(format!("stage {k} ({:#x}) item buffer {i} format {}/{:#x} against the translation's {format}/{dst_sel:#x}", first.addr, vb.format, vb.dst_sel));
                            }
                        }
                    }
                    let (a, n) = (vb.base, vb.extent().max(4));
                    let rr = [(a, n)];
                    if super::super::homes::overlaps(a, n) || self.overlaps_pending(&rr) || super::INFLIGHT.latest(&rr).is_some() || super::CP_WRITES.latest(&rr).is_some() {
                        found.push(format!("stage {k} ({:#x}) item buffer {i} at {a:#x}+{n:#x} is GPU-written data", first.addr));
                    }
                }
            }
        }
        if !found.is_empty() {
            let n = v[1].fetch_add(1, Ordering::Relaxed);
            if n < 40 {
                crate::info!("gpu[{}]: GPU-resolved draw of vs {:#x} ps {:#x} (verify): {}", self.name, g.pair.0, g.pair.1, found.join("; "));
            }
        }
    }

    /// A GPU-resolved pair's draw recorded in full: whether a per-item
    /// buffer's data is where the CPU path would not read guest memory (a
    /// video-memory home, pending GPU or CP writes, a resident shadow) or
    /// not imported; the first such finding.
    pub(super) fn gpu_draw_check(&mut self, vs: &Prepared, ps: &Prepared) -> Option<String> {
        for p in [vs, ps] {
            for (i, r) in p.resources.iter().enumerate().take(64) {
                if p.tr.spec.gpu_res & (1u64 << i) == 0 {
                    continue;
                }
                let Resolved::Buffer(v) = r else { continue };
                let (a, n) = (v.base, v.extent().max(4));
                let rr = [(a, n)];
                let what = if super::super::homes::overlaps(a, n) {
                    "in a video-memory home"
                } else if self.overlaps_pending(&rr) || super::INFLIGHT.latest(&rr).is_some() || super::CP_WRITES.latest(&rr).is_some() {
                    "written by pending GPU or CP work"
                } else if self.resident.iter().any(|x| x.guest < a + n && a < x.guest + x.range) {
                    "in a resident shadow"
                } else if super::span(a, n).is_none() {
                    "not imported"
                } else {
                    continue;
                };
                return Some(format!("{:#x}'s buffer {i} at {a:#x}+{n:#x} is {what}", p.addr));
            }
        }
        None
    }

}

/// `profile reset`: the draw statistics of this module.
pub(super) fn reset_draw_stats() {
    for c in DRAW_DIFFS.iter().chain(CONSEC_DIFFS.iter()).chain(BATCH_CANDS.iter()).chain(PREP_REPEAT.iter()).chain(DELTA.iter()).chain(DELTA_VERIFY.iter()).chain(DELTA_BIND.iter()).chain(STEPS_SPLIT.iter().flatten()) {
        c.store(0, Ordering::Relaxed);
    }
}

/// The draw-difference statistics (`DRAW_DIFFS`, `CONSEC_DIFFS`): with
/// `DS_PROFILE_GPU=1` or `DS_DRAW_STATS=1` (without the timestamps of the
/// former, which keep timed draws from `draw_repeat`).
fn draw_stats() -> bool {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ON.get_or_init(|| super::profiling() || std::env::var("DS_DRAW_STATS").ok().as_deref() == Some("1"))
}

/// Draws after a draw of the same programs on the same executor, by what
/// differs: [nothing, the user data only, buffer resources (images the
/// same), image resources]; and draws of other programs (profiling).
static DRAW_DIFFS: [std::sync::atomic::AtomicU64; 5] = [const { std::sync::atomic::AtomicU64::new(0) }; 5];
/// The same for draws right after the draw before them (nothing recorded
/// between, the same context state): [nothing different, the user data
/// only, buffer resources, image resources].
static CONSEC_DIFFS: [std::sync::atomic::AtomicU64; 4] = [const { std::sync::atomic::AtomicU64::new(0) }; 4];

/// A draw's programs and hashes of its buffer resources, image and sampler
/// resources and user data.
fn draw_sig(vs: &shaders::Prepared, ps: Option<&shaders::Prepared>) -> [u64; 5] {
    use std::hash::{Hash, Hasher};
    let (mut hb, mut hi, mut hu) = (std::collections::hash_map::DefaultHasher::new(), std::collections::hash_map::DefaultHasher::new(), std::collections::hash_map::DefaultHasher::new());
    for p in std::iter::once(vs).chain(ps) {
        for r in p.resources.iter() {
            match r {
                shaders::Resolved::Buffer(v) => v.hash(&mut hb),
                shaders::Resolved::Global(a, n) => (a, n).hash(&mut hb),
                shaders::Resolved::Image(t) => t.hash(&mut hi),
                shaders::Resolved::ImageTable(e) => e.hash(&mut hi),
                shaders::Resolved::Sampler(sm) => sm.hash(&mut hi),
            }
        }
        let n = p.sgpr_data.len().min(rdna::translate::SGPR_DATA_USER as usize);
        p.sgpr_data[..n].hash(&mut hu);
    }
    [vs.addr, ps.map_or(0, |p| p.addr), hb.finish(), hi.finish(), hu.finish()]
}

/// Batched-draw candidates (right after a like draw, buffers differing):
/// [arguments read by the CPU or none, GPU-read arguments 20 bytes after the
/// draw before's, at the same address, elsewhere; the same index base, one
/// within 64 MiB, farther, not indexed].
static BATCH_CANDS: [AtomicU64; 8] = [const { AtomicU64::new(0) }; 8];

fn note_draw_diff(last: &mut [u64; 5], sig: [u64; 5], consecutive: bool) -> usize {
    use std::sync::atomic::Ordering::Relaxed;
    let k = if sig[0] != last[0] || sig[1] != last[1] {
        4
    } else if sig[2] == last[2] && sig[3] == last[3] {
        if sig[4] == last[4] { 0 } else { 1 }
    } else if sig[3] == last[3] {
        2
    } else {
        3
    };
    DRAW_DIFFS[k].fetch_add(1, Relaxed);
    if consecutive && k < 4 {
        CONSEC_DIFFS[k].fetch_add(1, Relaxed);
    }
    *last = sig;
    k
}

/// Diagnosis (`DS_IMM_CHECK=<hex vertex program address>`, 2026-11-19): that program's draws (a
/// pass-through vertex program: position xyz through the buffer descriptor in user data 0-3,
/// colour xyzw through 4-7) have their positions read from guest memory as they are recorded;
/// non-finite ones and ones beyond 1e3 are logged (`imm check:`), with the first draws' as
/// samples.
fn imm_check() -> Option<u64> {
    static D: std::sync::OnceLock<Option<u64>> = std::sync::OnceLock::new();
    *D.get_or_init(|| std::env::var("DS_IMM_CHECK").ok().and_then(|v| u64::from_str_radix(v.trim().trim_start_matches("0x"), 16).ok()))
}

/// `DS_IMM_CHECK`: [draws checked, draws with a bad position, lines logged, rechecks, data
/// changed at a recheck, changes logged].
pub static IMM_CHECKS: [AtomicU64; 6] = [const { AtomicU64::new(0) }; 6];

/// `DS_IMM_CHECK`'s recheck: the checked draws' positions as recorded (draw number, ms, base,
/// the first four vertices' 48 bytes), looked at again 4-40 ms later.
static IMM_SNAPS: parking_lot::Mutex<std::collections::VecDeque<(u64, f64, u64, [u32; 12])>> = parking_lot::Mutex::new(std::collections::VecDeque::new());

fn imm_recheck(now: f64) {
    use std::sync::atomic::Ordering::Relaxed;
    let mut g = IMM_SNAPS.lock();
    while let Some(&(k, at, base, then)) = g.front() {
        if now - at < 4.0 {
            break;
        }
        g.pop_front();
        if now - at > 40.0 || !super::super::cp::guest_readable(base, 48) {
            continue;
        }
        let cur: [u32; 12] = std::array::from_fn(|i| unsafe { std::ptr::read_volatile((base + 4 * i as u64) as *const u32) });
        IMM_CHECKS[3].fetch_add(1, Relaxed);
        if cur != then {
            IMM_CHECKS[4].fetch_add(1, Relaxed);
            if IMM_CHECKS[5].fetch_add(1, Relaxed) < 300 {
                let f = |w: &[u32; 12]| w.chunks(3).map(|c| format!("{:.3},{:.3},{:.3}", f32::from_bits(c[0]), f32::from_bits(c[1]), f32::from_bits(c[2]))).collect::<Vec<_>>().join(" ");
                crate::info!("imm recheck: draw {k} at {base:#x} changed {:.1} ms after it was recorded: then {}; now {}", now - at, f(&then), f(&cur));
            }
        }
    }
}

fn imm_check_draw(cp: &Cp, args: &DrawArgs, prog: u64) {
    use std::sync::atomic::Ordering::Relaxed;
    let ud = |i: u32| cp.reg(Space::Sh, 0x8C + i);
    let (d0, d1, d2, d3) = (ud(0), ud(1), ud(2), ud(3));
    let base = d0 as u64 | ((d1 & 0xFFFF) as u64) << 32;
    let stride = ((d1 >> 16) & 0x3FFF) as u64;
    let n = args.count.min(16) as u64;
    let first = args.base_vertex.unwrap_or(0).max(0) as u64;
    let k = IMM_CHECKS[0].fetch_add(1, Relaxed);
    let mut bad = Vec::new();
    let mut all = Vec::new();
    for i in 0..n {
        let a = base + (first + i) * stride;
        if !super::super::cp::guest_readable(a, 12) {
            bad.push(format!("v{i} unreadable at {a:#x}"));
            continue;
        }
        let p: [f32; 3] = unsafe { [std::ptr::read_volatile(a as *const f32), std::ptr::read_volatile((a + 4) as *const f32), std::ptr::read_volatile((a + 8) as *const f32)] };
        if p.iter().any(|x| !x.is_finite() || x.abs() > 1e3) {
            bad.push(format!("v{i} {p:?} at {a:#x}"));
        }
        all.push(format!("{:.3},{:.3},{:.3}", p[0], p[1], p[2]));
    }
    let cbase = ud(4) as u64 | ((ud(5) & 0xFFFF) as u64) << 32;
    let now = crate::log::elapsed_ms();
    imm_recheck(now);
    if stride == 12 && n >= 4 && super::super::cp::guest_readable(base + first * 12, 48) {
        let a = base + first * 12;
        let snap: [u32; 12] = std::array::from_fn(|i| unsafe { std::ptr::read_volatile((a + 4 * i as u64) as *const u32) });
        let mut g = IMM_SNAPS.lock();
        if g.len() < 4096 {
            g.push_back((k, now, a, snap));
        }
    }
    if !bad.is_empty() {
        IMM_CHECKS[1].fetch_add(1, Relaxed);
        if IMM_CHECKS[2].fetch_add(1, Relaxed) < 300 {
            crate::info!("imm check: draw {k} of {prog:#x}: V# {d0:08x} {d1:08x} {d2:08x} {d3:08x} (base {base:#x}, stride {stride}, colours at {cbase:#x}), count {} first {first}: bad {}; all {}", args.count, bad.join("; "), all.join(" "));
        }
    } else if k < 6 || k % 20000 == 0 {
        crate::info!("imm check: draw {k} of {prog:#x} (sample): V# {d0:08x} {d1:08x} {d2:08x} {d3:08x} (base {base:#x}, stride {stride}, colours at {cbase:#x}), count {} first {first}: {}", args.count, all.join(" "));
    }
}

/// `DS_SKIP_NGG` / `DS_SKIP_PS`: code-hash ranges `[lo, hi)` (a single hash stands for itself).
fn skip_ranges() -> &'static (Vec<(u64, u64)>, Vec<(u64, u64)>) {
    static S: std::sync::OnceLock<(Vec<(u64, u64)>, Vec<(u64, u64)>)> = std::sync::OnceLock::new();
    S.get_or_init(|| {
        let p = |s: &str| u64::from_str_radix(s.trim().trim_start_matches("0x"), 16).ok();
        let ranges = |var: &str| -> Vec<(u64, u64)> {
            std::env::var(var).map_or(Vec::new(), |v| {
                v.split(';')
                    .filter(|e| !e.trim().is_empty())
                    .filter_map(|e| match e.split_once('-') {
                        Some((a, b)) => Some((p(a)?, p(b)?)),
                        None => p(e).map(|a| (a, a.saturating_add(1))),
                    })
                    .collect()
            })
        };
        let r = (ranges("DS_SKIP_NGG"), ranges("DS_SKIP_PS"));
        if !r.0.is_empty() || !r.1.is_empty() {
            crate::info!("gpu: draws left out by program code hash (`DS_SKIP_NGG` {:x?}, `DS_SKIP_PS` {:x?})", r.0, r.1);
        }
        r
    })
}

fn skip_progs_on() -> bool {
    let r = skip_ranges();
    !r.0.is_empty() || !r.1.is_empty()
}

thread_local! {
    /// Programs this thread has left out (each logged once a thread).
    static SKIPPED_PROGS: std::cell::RefCell<std::collections::HashSet<u64>> = std::cell::RefCell::new(std::collections::HashSet::new());
}

fn skip_by_program(cp: &Cp) -> bool {
    let (ngg, ps) = skip_ranges();
    let inside = |r: &[(u64, u64)], addr: u64| -> Option<u64> {
        if r.is_empty() || addr == 0 {
            return None;
        }
        let h = super::super::shaders::program_hash(addr).ok()?;
        r.iter().any(|&(lo, hi)| h >= lo && h < hi).then_some(h)
    };
    let (na, pa) = (shaders::ngg_addr(cp), shaders::ps_addr(cp));
    let hit = inside(ngg, na).map(|h| ("NGG", h, na)).or_else(|| inside(ps, pa).map(|h| ("pixel", h, pa)));
    match hit {
        Some((k, h, a)) => {
            if SKIPPED_PROGS.with(|s| s.borrow_mut().insert(h)) {
                crate::info!("gpu: draws of the {k} program {h:#018x} (at {a:#x}) left out");
            }
            true
        }
        None => false,
    }
}

fn skip_depth() -> Option<u64> {
    static D: std::sync::OnceLock<Option<u64>> = std::sync::OnceLock::new();
    *D.get_or_init(|| std::env::var("DS_SKIP_DEPTH").ok().and_then(|v| u64::from_str_radix(v.trim().trim_start_matches("0x"), 16).ok()))
}

/// Diagnosis (`DS_VIEW_CENSUS=<hex lo>-<hex hi>`): the words a draw's stages read inside that
/// range (the title's per-view block: one version per pass), kept per (frame: the graphics CP's
/// flips executed, depth target, colour target 0) by address; from frame 300 on, a frame's
/// target pair whose draws read two values at one address is logged once the frame is 3 behind
/// (`view census: ...`: the addresses that differ, each value's draws, first vertex program and
/// recording threads; at most 400 lines).
fn view_census_range() -> Option<(u64, u64)> {
    static R: std::sync::OnceLock<Option<(u64, u64)>> = std::sync::OnceLock::new();
    *R.get_or_init(|| {
        let v = std::env::var("DS_VIEW_CENSUS").ok()?;
        let (a, b) = v.split_once('-')?;
        let p = |x: &str| u64::from_str_radix(x.trim().trim_start_matches("0x"), 16).ok();
        Some((p(a)?, p(b)?))
    })
}

/// A value's draws: [count, first vertex program], the recording threads seen.
type ViewValue = (u64, u64, Vec<String>);
#[allow(clippy::type_complexity)]
static VIEW_CENSUS: parking_lot::Mutex<Option<std::collections::BTreeMap<u64, HashMap<(u64, u64, [u32; 4]), std::collections::BTreeMap<u64, HashMap<u32, ViewValue>>>>>> = parking_lot::Mutex::new(None);
static VIEW_CENSUS_LOGGED: AtomicU64 = AtomicU64::new(0);
/// Minority signatures logged so far (pass, value, vertex program) -> times: each at most 30.
#[allow(clippy::type_complexity)]
static VIEW_CENSUS_SIGS: parking_lot::Mutex<Option<HashMap<(u64, u64, [u32; 4], u32, u64), u32>>> = parking_lot::Mutex::new(None);

fn note_view_census(cp: &Cp, lo: u64, hi: u64, vs: &shaders::Prepared, ps: Option<&shaders::Prepared>) {
    let frame = super::super::FRAMES.load(Ordering::Relaxed);
    if frame < 300 {
        return;
    }
    let mut seen: Vec<(u64, u32)> = Vec::new();
    for p in std::iter::once(vs).chain(ps) {
        let mut k = 0usize;
        for &(a, len) in &p.found.reads {
            let m = (len / 4) as usize;
            for j in 0..m {
                let addr = a + 4 * j as u64;
                if addr >= lo && addr < hi {
                    if let Some(&w) = p.found.words.get(k + j) {
                        seen.push((addr, w));
                    }
                }
            }
            k += m;
        }
    }
    if seen.is_empty() {
        return;
    }
    let depth = depth_surface(cp).map_or(0, |s| s.base);
    let cb0 = (ctx_reg(cp, 0x28C60) as u64) << 8 | ((ctx_reg(cp, 0x28E40) as u64 & 0xFF) << 40);
    // (The viewport too: a shadow atlas's cascades share a depth target, each with its own view.)
    let vport = [ctx_reg(cp, 0x2843C), ctx_reg(cp, 0x28440), ctx_reg(cp, 0x28444), ctx_reg(cp, 0x28448)];
    let thread = crate::log::thread_name();
    let mut g = VIEW_CENSUS.lock();
    let m = g.get_or_insert_with(std::collections::BTreeMap::new);
    let t = m.entry(frame).or_default().entry((depth, cb0, vport)).or_default();
    for (addr, w) in seen {
        let e = t.entry(addr).or_default().entry(w).or_insert_with(|| (0, vs.addr, Vec::new()));
        e.0 += 1;
        if e.2.len() < 4 && !e.2.contains(&thread) {
            e.2.push(thread.clone());
        }
    }
    // Frames 3 behind: logged where an address read two values, then dropped.
    while let Some((&f, _)) = m.iter().next() {
        if f + 3 > frame {
            break;
        }
        let targets = m.remove(&f).unwrap();
        for ((d, c, vp), addrs) in targets {
            let differing: Vec<(&u64, &HashMap<u32, ViewValue>)> = addrs.iter().filter(|(_, vals)| vals.len() > 1).collect();
            if differing.is_empty() || VIEW_CENSUS_LOGGED.load(Ordering::Relaxed) >= 2000 {
                continue;
            }
            // (Logged where one value holds most of the group's draws and others are few: the
            // shape of draws that took another pass's view.)
            let (_, vals0) = differing[0];
            let total: u64 = vals0.values().map(|x| x.0).sum();
            let top = vals0.values().map(|x| x.0).max().unwrap_or(0);
            if top * 10 < total * 7 {
                continue;
            }
            VIEW_CENSUS_LOGGED.fetch_add(1, Ordering::Relaxed);
            let (a0, vals) = differing[0];
            let mut v: Vec<(&u32, &ViewValue)> = vals.iter().collect();
            v.sort_by(|a, b| b.1 .0.cmp(&a.1 .0));
            // (Each minority signature at most 30 times: steady ones stop, rare ones stay.)
            {
                let mut sg = VIEW_CENSUS_SIGS.lock();
                let sm = sg.get_or_insert_with(HashMap::new);
                let mut fresh = false;
                for (w, x) in v.iter().skip(1) {
                    let n = sm.entry((d, c, vp, **w, x.1)).or_insert(0);
                    *n += 1;
                    if *n <= 30 {
                        fresh = true;
                    }
                }
                if !fresh {
                    continue;
                }
            }
            let parts: Vec<String> = v.iter().map(|(w, x)| format!("{w:08x}: {} draws (vs {:#x}; {})", x.0, x.1, x.2.join(" "))).collect();
            crate::info!("view census: frame {f} depth {d:#x} cb0 {c:#x} viewport {:08x} {:08x} {:08x} {:08x}: {} of {} addresses read two or more values; at {a0:#x}: {}", vp[0], vp[1], vp[2], vp[3], differing.len(), addrs.len(), parts.join("; "));
        }
    }
}

/// Draws with the programs and user data of a draw of a recent flip:
/// [draws, then per distance 1-4 flips back: such repeats, of them with
/// unchanged words read by their preparation] (`DS_DRAW_STATS`: a
/// cross-flip preparation cache's hits).
static PREP_REPEAT: [AtomicU64; 9] = [const { AtomicU64::new(0) }; 9];
/// [flip, the keys -> words hash of this flip and the four before].
#[allow(clippy::type_complexity)]
static PREP_SEEN: parking_lot::Mutex<(u64, [Option<HashMap<u64, u64>>; 5])> = parking_lot::Mutex::new((0, [None, None, None, None, None]));

fn note_prep_repeat(vs: &shaders::Prepared, ps: Option<&shaders::Prepared>) {
    let mut hk = super::super::fx::FxHasher::default();
    let mut hw = super::super::fx::FxHasher::default();
    for p in std::iter::once(vs).chain(ps) {
        p.addr.hash(&mut hk);
        p.user.hash(&mut hk);
        p.found.words.hash(&mut hw);
    }
    let (key, words) = (hk.finish(), hw.finish());
    let frame = super::super::FRAMES.load(Ordering::Relaxed);
    let mut g = PREP_SEEN.lock();
    if g.0 != frame {
        // One map per flip: the oldest dropped.
        g.1.rotate_right(1);
        g.1[0] = Some(HashMap::new());
        g.0 = frame;
    }
    PREP_REPEAT[0].fetch_add(1, Ordering::Relaxed);
    for d in 1..5 {
        if let Some(&w) = g.1[d].as_ref().and_then(|m| m.get(&key)) {
            PREP_REPEAT[2 * d - 1].fetch_add(1, Ordering::Relaxed);
            if w == words {
                PREP_REPEAT[2 * d].fetch_add(1, Ordering::Relaxed);
            }
        }
    }
    g.1[0].get_or_insert_with(HashMap::new).insert(key, words);
}

/// The status line of `PREP_REPEAT`.
pub(super) fn prep_repeat_line() -> String {
    let r: Vec<u64> = PREP_REPEAT.iter().map(|x| x.load(Ordering::Relaxed)).collect();
    if r[0] == 0 {
        return String::new();
    }
    let parts: Vec<String> = (1..5).map(|d| format!("{d} back {:.1}% ({:.1}% with unchanged words)", 100.0 * r[2 * d - 1] as f64 / r[0] as f64, 100.0 * r[2 * d] as f64 / r[0] as f64)).collect();
    format!("    draws with the programs and user data of a draw of a recent flip (of {}): {}\n", r[0], parts.join(", "))
}

/// Status part: `DRAW_DIFFS`.
pub(super) fn draw_diff_line() -> String {
    use std::sync::atomic::Ordering::Relaxed;
    let v: Vec<u64> = DRAW_DIFFS.iter().map(|c| c.load(Relaxed)).collect();
    if v.iter().sum::<u64>() == 0 {
        return String::new();
    }
    let c: Vec<u64> = CONSEC_DIFFS.iter().map(|x| x.load(Relaxed)).collect();
    let b: Vec<u64> = BATCH_CANDS.iter().map(|x| x.load(Relaxed)).collect();
    format!(
        "    draws after a draw of the same programs (same executor): nothing different {}, the user data only {}, buffer resources (same images) {}, image resources {}; after other programs {}; right after it with the same context state (nothing recorded between): nothing different {}, the user data only {}, buffer resources {}, image resources {}; of the last, arguments by the CPU or none {}, GPU-read arguments next to the last's {}, at the same address {}, elsewhere {}; index base the same {}, within 64 MiB {}, farther {}, not indexed {}\n",
        v[0], v[1], v[2], v[3], v[4], c[0], c[1], c[2], c[3], b[0], b[1], b[2], b[3], b[4], b[5], b[6], b[7]
    )
}
