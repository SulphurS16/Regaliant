//! RDNA (gfx10) -> SPIR-V translation.
//!
//! Ported from the earlier native port's GCN translator and re-keyed by gfx10
//! instruction names. Execution model: each SPIR-V invocation emulates one
//! hardware lane as a wave of one. Scalar registers are per-invocation copies
//! (scalar inputs are uniform, so the copies agree across lanes); EXEC/VCC and
//! compare masks keep this lane's bit in bit 0. VALU results are written only
//! when the lane is active. Branchy programs run in a loop + switch
//! dispatcher over basic blocks. Cross-lane operations within 32 lanes (DPP
//! rows, permlane, swizzles) use subgroup operations; wave64 operations that
//! cross 32-lane halves are approximated and reported as warnings.
//!
//! Hardware lanes of a wave execute in lockstep, and shaders rely on it:
//! lanes exchange data through LDS without barriers. Invocations do not run
//! in lockstep, so LDS accesses get workgroup barriers wherever another
//! lane's earlier access could conflict (in blocks every invocation executes
//! alike).
//!
//! Programs whose loops nest (each loop a contiguous address range from its
//! header to its last back edge, entered only at the header) need no
//! dispatcher: their blocks are emitted in address order, each run when the
//! next address is its start, and each loop range as a structured loop that
//! repeats while the next address is its header (an invocation's path visits
//! the blocks of one iteration in increasing order; exits, early ends and
//! jumps to an outer header fall through the remaining guards). The driver
//! then sees structured control flow instead of one loop over a switch.
//! Others (overlapping or irreducible loops) keep the dispatcher. Compute programs that use LDS run in *lockstep mode* when their
//! workgroup is a single wave, or when no scalar value can differ between
//! its waves (no v_readlane / v_readfirstlane, no wave id): lane-dependent
//! branches are decided for the whole workgroup by a vote (as the hardware
//! decides them for each wave), so every block is uniform and all LDS
//! accesses are ordered. With several waves a wave whose lanes are all
//! inactive runs a region with EXEC empty instead of skipping it.
//! Single-wave compute programs that read compare masks as values (ballots
//! they store or count: [`crate::wavemask::reads_mask_values`]) run in
//! lockstep mode as well, with or without LDS.
//!
//! Outside lockstep mode invocations that took different paths reach a
//! uniform block after different numbers of dispatcher steps, so the
//! workgroup barriers placed there (LDS ordering, wave64 lane exchanges,
//! s_barrier) would be different dynamic instances: invalid in Vulkan, and on
//! NVIDIA a warp whose threads arrive at different steps passes the barrier
//! early (a 64-lane luminance reduction raced and the exposure came out
//! wrong, depending on timing). Such blocks are *sync blocks*: the dispatcher
//! runs in rounds, and an invocation that reaches a sync block after other
//! blocks of its round ends the round there; every invocation then starts
//! the next round at that block together (see `emit_dispatch_loop`).
//!
//! Single-wave lockstep programs run in *wave-mask mode*: lane masks (EXEC,
//! VCC, compare results) have the hardware layout, bit i for lane i, so
//! masks used as values (ballots kept in LDS, counted, scanned with s_ff1)
//! and the scalar state derived from them agree across invocations, and
//! branches need no votes. A compare computes its own lane's bit; a ballot
//! through workgroup memory completes the mask wherever it may be read as a
//! whole (see [`crate::wavemask`]); elsewhere an invocation keeps a partial
//! mask whose own bit is right.
//!
//! *Subgroup waves* (`Spec::sg_waves`) run wave-mask mode per subgroup, for
//! wave64 programs without LDS whose results do not depend on how many lanes
//! a wave has (the runtime lists them, each checked by reading its code):
//! each subgroup is a wave of its own, its invocations lanes 0 to its size
//! - 1 (`SubgroupLocalInvocationId`), the wave's other lanes absent (their
//! mask bits 0, lane reads of them 0). Subgroup ballots complete the masks,
//! and no workgroup barrier is needed.

mod active;
mod alu;
pub mod gs;
mod mem;

use crate::analysis::{Plan, ResKind};
use crate::{Enc, Inst};
use rspirv::binary::Assemble;
use rspirv::dr::{Builder, Operand};
use rspirv::spirv::{self, BuiltIn, Capability, Decoration, ExecutionMode, ExecutionModel, StorageClass, Word};
use std::collections::{BTreeSet, VecDeque};
use crate::HashMap;

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum Stage {
    /// NGG primitive shader (merged ES/GS) run as a Vulkan vertex shader:
    /// per-vertex work only; primitive exports are dropped.
    Vertex,
    Fragment,
    Compute,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum ImageDim {
    D1,
    D2,
    D3,
    Cube,
    D1Array,
    D2Array,
    D2Msaa,
    D2MsaaArray,
    CubeArray,
    /// Compute: a sampled resource its instructions address as 3D, bound
    /// with a 3D or a 2D texture (the title binds 2D textures where an
    /// effect's program samples a 3D one: its dispatches alternated between
    /// two pipelines). The shader holds a 3D and a 2D variable
    /// (`BindingKind::Image`, `BindingKind::ImageAlt`) and picks by the T#'s
    /// TYPE when it runs; a 2D texture is addressed at (x, y), as the
    /// hardware's address unit does. The runtime binds the other one null.
    D3OrD2,
}

impl ImageDim {
    /// From a T# TYPE field (word3[31:28]) and its layer count (cube
    /// descriptors count faces: more than 6 make a cube array).
    pub fn from_type(t: u32, layers: u32) -> Option<ImageDim> {
        Some(match t {
            8 => ImageDim::D1,
            9 => ImageDim::D2,
            10 => ImageDim::D3,
            11 if layers > 6 => ImageDim::CubeArray,
            11 => ImageDim::Cube,
            12 => ImageDim::D1Array,
            13 => ImageDim::D2Array,
            14 => ImageDim::D2Msaa,
            15 => ImageDim::D2MsaaArray,
            _ => return None,
        })
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum NumClass {
    Float,
    Sint,
    Uint,
}

/// Fragment output / export conversion for one MRT.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, Default)]
pub struct MrtSpec {
    /// SPI_SHADER_COL_FORMAT value for this target (0 = none).
    pub export_format: u8,
    /// 0 float, 1 sint, 2 uint (render target number type).
    pub class: u8,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct PsInput {
    /// Vertex parameter index feeding this attribute (SPI_PS_INPUT_CNTL.OFFSET).
    pub param: u8,
    pub flat: bool,
    /// Use a constant default instead of a vertex output (0..3 per DEFAULT_VAL).
    pub default: Option<u8>,
}

/// `ResSpec::Buffer` format: the V#'s FORMAT and DST_SEL are read from its
/// word 3 when the program runs and decoded through [`rt_format_entry`]'s
/// table (`dst_sel` unused), so one translation serves every typed buffer
/// format the title binds there (the runtime's compute stages, buffers the
/// program only loads formatted).
pub const FORMAT_RT: u8 = 0xFF;

/// [`FORMAT_RT`]'s table entry of a unified format: the component widths
/// (6 bits each from bit 0, component X first) and the number class (bits
/// 24-26: 0 UNORM, 1 SNORM, 2 USCALED, 3 SSCALED, 4 UINT, 5 SINT, 6 FLOAT),
/// bit 31 set; 0 for a format the decoder does not read (block-compressed,
/// sRGB, floats other than 16 and 32 bits, components across dwords, sizes
/// other than 1, 2, 4, 8, 12 and 16 bytes): such a binding keeps its format
/// in the spec.
pub fn rt_format_entry(id: u16) -> u32 {
    use crate::format::Num;
    let Some(f) = crate::format::gfx10(id) else { return 0 };
    if f.block || f.components() == 0 {
        return 0;
    }
    let class = match f.num {
        Num::Unorm => 0,
        Num::Snorm => 1,
        Num::Uscaled => 2,
        Num::Sscaled => 3,
        Num::Uint => 4,
        Num::Sint => 5,
        Num::Float => 6,
        _ => return 0,
    };
    let (mut e, mut off) = (0u32, 0u32);
    for (c, &w) in f.bits.iter().enumerate() {
        let w = w as u32;
        if w == 0 {
            break;
        }
        if class == 6 && w != 16 && w != 32 || off / 32 != (off + w - 1) / 32 {
            return 0;
        }
        e |= w << (6 * c);
        off += w;
    }
    if off % 8 != 0 || !matches!(off / 8, 1 | 2 | 4 | 8 | 12 | 16) {
        return 0;
    }
    e | class << 24 | 1 << 31
}

/// Per-resource specialization resolved from descriptors at draw time.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum ResSpec {
    /// V#: unified FORMAT (word3[18:12]) and DST_SEL_XYZW (word3[11:0]).
    Buffer { format: u8, dst_sel: u16 },
    /// `depth`: bound as a depth/stencil view (depth-comparison sampling
    /// works; on color views it is emulated, see `mem::compare_texels`).
    Image { dim: ImageDim, class: NumClass, depth: bool },
    Sampler,
    /// Base address of GLOBAL instructions (bound like a raw buffer).
    Global,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Spec {
    pub stage: Stage,
    /// Number of user-data dwords.
    pub user_sgprs: u16,
    /// First SGPR receiving user data (8 for NGG shaders, 0 otherwise).
    pub user_base: u16,
    /// The program's address. Translations do not depend on it (the
    /// shader reads it from the SGPR data, `code_base_offset`), except for
    /// the DS_WAVEMASK diagnosis filter.
    pub code_base: u64,
    /// Wave64 (else wave32): lane numbering for cross-lane operations.
    pub wave64: bool,
    pub ps_input_addr: u32,
    pub ps_inputs: Vec<PsInput>,
    /// Fragment barycentrics and per-vertex inputs are available
    /// (VK_KHR_fragment_shader_barycentric): the barycentric VGPRs hold the
    /// real (i, j) and v_interp_mov reads each vertex's value.
    pub barycentric: bool,
    /// Without them: a generated geometry shader hands a pixel stage that
    /// reads per vertex each vertex's values and the barycentrics as inputs
    /// (`gs::bary_gs`; `Output::bary_gs` names their locations).
    pub bary_gs: bool,
    pub mrt: [MrtSpec; 8],
    pub cs_local_size: [u32; 3],
    /// COMPUTE_PGM_RSRC2 (TGID enables, TG size, TIDIG count, LDS size).
    pub cs_rsrc2: u32,
    pub resources: Vec<ResSpec>,
    /// Binding index of the first binding (stages share one descriptor set).
    pub binding_base: u32,
    /// Debugging: (pc, register): the value of the register after the
    /// instruction at `pc` replaces the color of every MRT export ((v, v, v,
    /// 1) as float), or (compute) the data of every image and buffer store.
    /// The register is a VGPR number, or an SGPR with PROBE_SGPR; with
    /// PROBE_UINT / PROBE_SINT its integer value is converted to float.
    pub probe: Option<(u32, u32)>,
    /// Debugging (compute): (pc, workgroup id): after each execution of the
    /// instruction at pc, the invocations of that workgroup write their
    /// registers to the debug binding (see [`DUMP_STRIDE`]). Vertex stage:
    /// (pc, [first vertex index, instance index, instances n]): the
    /// invocations of the n instances from that one with vertex indices from
    /// the first one do (8192 / n slots per instance; n 0 counts as 1). With
    /// [`DUMP_POS`] in n, at a position export: only the exported position,
    /// four dwords per slot, 64 vertex slots per instance (n & !DUMP_POS
    /// instances, up to 16384).
    pub dump: Option<(u32, [u32; 3])>,
    /// The loop-and-switch dispatcher instead of structured control flow:
    /// for every program, or for programs with loops only (A/B comparisons).
    pub force_dispatcher: bool,
    pub dispatcher_for_loops: bool,
    /// Structured control flow with each block inside the guarded region of
    /// its immediate dominator (`emit_nested`): its guard is tested only in
    /// passes where the dominator ran, instead of every block's guard on
    /// every pass.
    pub dom_nest: bool,
    /// One variable per register (programs without relative addressing),
    /// else the register files as arrays (the earlier form, for A/B).
    pub reg_vars: bool,
    /// Runs of vector instructions that only touch their own lane's
    /// registers (`lane_local`) are translated inside one lane guard (`if`
    /// this lane is active) with plain register writes, instead of a select
    /// against the old value for every write (smaller programs: compile time
    /// grows with their size).
    pub lane_guards: bool,
    /// Storage images whose instructions select mip levels (image_store_mip,
    /// image_load_mip) are bound one view per level
    /// (`BindingKind::StorageImageMips`); else as one view of the T#'s base
    /// level that every level is written to (the earlier form, for A/B).
    pub mip_views: bool,
    /// Hardware LOD statistics (`sceAgcDcbGetLodStats`, the title's texture
    /// streaming feedback): sampling instructions on textures whose T#
    /// enables counting record the finest mip they sample and a count in
    /// the T#'s counter bank (`BindingKind::LodStats`, [`LOD_INFO_ENABLED`]).
    pub lod_stats: bool,
    /// With `lod_stats`: a bank's mip is stored only when it is finer than
    /// the one the bank holds (read first: banks only get finer between
    /// resets, so a stale read costs an atomic at most), and counts go to
    /// the count slot the subgroup's position selects (`LOD_COUNT_SLOTS`):
    /// the statistics cost the vista 4% of its GPU time as one min and one
    /// add on one address per bank and subgroup (FINDINGS.md 2026-10-30).
    pub lod_lean: bool,
    /// Pixel stage: the depth and stencil tests run before the shader
    /// (DB_SHADER_CONTROL.DEPTH_BEFORE_SHADER, D3D's [earlydepthstencil]), so
    /// fragments that fail them do not run it and its image and buffer stores
    /// do not happen (`EarlyFragmentTests`; not with a depth export). Without
    /// it a Vulkan fragment shader with stores runs for every rasterized
    /// fragment: the courtyard's decal volumes (stencil-marked boxes whose
    /// shader rewrites the G-buffer through image stores) darkened the whole
    /// screen rectangle of a box instead of the ground inside it.
    pub early_tests: bool,
    /// Pixel stage: the title's state lets the depth and stencil tests run
    /// before a program that writes no memory and exports no depth without
    /// changing what it draws (the hardware tests early there): the program
    /// cannot kill (DB_SHADER_CONTROL.KILL_ENABLE clear, no coverage or
    /// depth export) or the draw writes neither depth nor stencil (a kill then
    /// undoes nothing a test wrote), and no alpha to coverage while it does.
    /// Such programs declare EarlyFragmentTests when they write no memory
    /// themselves (`writes_memory`): the LOD statistics' atomics (and any
    /// side effect of ours) otherwise make the driver run them for every
    /// rasterized fragment, hidden ones too (the Arcade's G-buffer passes
    /// through its ivy: 1.2 ms a draw, FINDINGS.md 2026-10-30). A kill at a
    /// valid-mask export then drops the fragment's colors only, as the
    /// hardware's export without KILL_ENABLE does.
    pub early_ok: bool,
    /// Compute: translate per lane, each invocation a wave of one, even when
    /// the program would run in lockstep or wave-mask mode: for programs
    /// whose cross-lane operations only scalarize per-lane work (their
    /// lane masks OR-reduced across the wave and walked with s_ff1, values
    /// broadcast with v_readlane / v_readfirstlane in waterfall loops, LDS
    /// slots addressed by lane): walking only this lane's part gives each
    /// lane the same result. The runtime decides per program (its list of
    /// verified programs).
    pub per_lane: bool,
    /// Per lane: EXEC and VCC branches are decided for the 32-lane subgroup
    /// (`branch_cond`), for programs whose cross-lane reads (after saving
    /// EXEC and enabling all lanes) must find lanes EXEC masked off earlier.
    /// It costs time where lanes would have skipped a region alone.
    pub lane_votes: bool,
    /// Compute: no lockstep (and wave-mask) mode although cross-lane
    /// operations sit in code the invocations may not all run, for programs
    /// whose such reads are waterfall loops (v_readfirstlane: each 32-lane
    /// subgroup walks its own lanes' values), wave-wide OR reductions of
    /// per-lane masks walked with s_ff1 under per-lane tests (each subgroup
    /// walks its own half) and DPP within quads and rows, and that use no
    /// LDS: each subgroup then gives its lanes the wave's results. The
    /// runtime decides per program (its list of verified programs).
    pub lockstep_off: bool,
    /// Compute, wave-mask mode: subgroup waves (see the module
    /// documentation): each subgroup a wave of its own, no workgroup
    /// barriers. For wave64 programs without LDS whose lane-0 elections,
    /// appends and reductions are per wave and that use no lane index as
    /// data, so that two waves of 32 lanes give what one of 64 gives. The
    /// runtime decides per program (its list of verified programs).
    pub sg_waves: bool,
    /// Compute: wave-aggregated appends done per subgroup ([`Appends`]),
    /// when all of the program's are of that form (the runtime sets it for
    /// programs with one, `has_append`).
    pub sg_appends: bool,
    /// Per-invocation translations (not lockstep, wave-mask mode or lane
    /// votes): where this invocation's EXEC bit is known set
    /// ([`active::known_active`]), register writes are plain (no select, no
    /// lane guard) and stores unguarded.
    pub known_active: bool,
    /// Scalar buffer loads (`s_buffer_load`, a dynamic `s_load` through a
    /// binding) of buffers that only scalar loads read, without a range check
    /// of each dword (`Output::unchecked_res`): the device reads 0 past a
    /// binding's range (robustness2's robustBufferAccess2), and the runtime
    /// binds such a buffer to end where the checks did, and an empty or
    /// unmapped resource as a null descriptor.
    pub hw_bounds: bool,
    /// Vertex stage of a draw whose arguments the GPU reads (indexed
    /// indirect draws, see the runtime's `DrawArgs::gpu_args`): this SGPR
    /// receives the start instance (`gl_BaseInstance`), as the command
    /// processor writes it into the user SGPR the packet names, and the
    /// instance id VGPR counts from 0.
    pub base_instance_sgpr: Option<u16>,
    /// The SGPR data binding covers many commands' data and this stage reads
    /// its own from a dword offset in a push constant at this byte offset
    /// ([`sgpr_pc_offset`]): the runtime binds its ring once per batch and
    /// pushes each command's offset, so a command whose resources equal the
    /// one before pushes no descriptors. None: the binding starts at the
    /// command's data.
    pub sgpr_pc: Option<u32>,
    /// Buffer resources are read and written through device addresses
    /// (`PhysicalStorageBuffer64`): each is listed as `BindingKind::Address`
    /// without a descriptor, and the runtime writes its range's address and
    /// size into the SGPR data's address block ([`addr_info_offset`]).
    /// Accesses outside the range read 0 (the block's zero page) and do not
    /// write, as robust buffer access did for descriptors.
    pub addresses: bool,
    /// Compute, with `addresses`: one dispatch runs consecutive dispatches of
    /// the program (`BATCH_ENTRY_DWORDS` per workgroup): the push constant
    /// at byte 0 holds the dword offset of a table in the SGPR data binding
    /// (the whole ring), and workgroup x's entry gives its command's SGPR
    /// data offset and its workgroup x within that command.
    /// Vertex and pixel stages, with `addresses`: one multi-draw runs
    /// consecutive draws of the programs (`BATCH_DRAW_ENTRY_DWORDS` per
    /// draw): the vertex stage reads the table's offset from push constant
    /// byte 0 and its draw's entry by `DrawIndex` (its own SGPR data offset
    /// and the pixel stage's), and hands the pixel stage its offset in a
    /// flat varying at `BATCH_VARYING`.
    pub batched: bool,
    /// Mirror mode (the runtime's `DS_RESIDENT=2`): every buffer store and
    /// atomic also sets the bits of the dwords it writes in the written-bit
    /// map (`BindingKind::Marks`: one bit per dword of the mirror), from its
    /// resource's first map word in dword 3 of the resource's address-block
    /// entry (u32::MAX: not marked; a marked binding starts at a multiple of
    /// 128 bytes of the mirror). Only dwords inside the bound range count.
    pub write_marks: bool,
    /// With `write_marks`, timing only (the runtime's `tune mark_touch 0`):
    /// stores set their dwords' bits but do not enter their pages in the
    /// touched-page list (`touch_page`), so write-backs find nothing to
    /// write and GPU writes never reach guest memory.
    pub marks_untouched: bool,
    /// With `write_marks` (the runtime's front end 2): a store reads its
    /// page's touched bit first and sets it (the atomic whose result decides
    /// the list append) only when it finds it clear, and the mark words one
    /// store sets in one page touch that page once (`touch_page`).
    pub touch_test: bool,
    /// With `write_marks`, an exact ballast (the runtime's `tune tr_ballast`
    /// bit 0): every store's marking is done twice. The second sets the
    /// same bits and finds its pages touched, so what reaches guest memory
    /// is the same and only the marking's cost doubles: what it takes from
    /// live frames prices a leaner form.
    pub mark_twice: bool,
    /// Exact ballast (diagnosis, the runtime's `tune tr_ballast` bit 2):
    /// each store's page-touch test (`touch_test`: the load of the page's
    /// touched bit and the branch on it) done twice, both loads volatile.
    pub touch_twice: bool,
    /// Exact ballast (diagnosis, `tune tr_ballast` bit 3): each store's
    /// partitioned reduction and its leader's OR into the map done twice
    /// (`marks_part`; not the page touch).
    pub part_twice: bool,
    /// Exact ballast (diagnosis): two dependent loads of the SGPR data
    /// before the user SGPRs, whose loads then depend on them (zero offsets
    /// at run time): what fetching a stage's descriptors from memory at its
    /// start would cost the GPU.
    pub dep_loads: bool,
    /// With `write_marks`, the lanes of a store that mark one map word OR
    /// their bits by a partitioned reduction (OpGroupNonUniformPartitionEXT:
    /// the device offers VK_NV_shader_subgroup_partitioned), the
    /// partition's lowest lane storing them: the same bits and pages as the
    /// subgroup ORs over up to five words and the per-lane path, in one
    /// reduction per word.
    pub marks_part: bool,
    /// With `write_marks`, stores into pages whose bit is set in the map's
    /// held region (`kernels::MARK_HELD_AT`: front end 4's held pages) set
    /// no marks and touch no page (C5: a held page's release copies it
    /// whole from the mirror, so its words need no marks).
    pub marks_held: bool,
    /// How the SGPR data binding is declared: 0 a storage buffer (u32
    /// array), 1 a uniform buffer (`BindingKind::SgprUniform`: a std140
    /// array of uvec4 sized to the plan's SGPR data) where the binding is the
    /// command's own range (not with `sgpr_pc` or `batched`, whose binding is
    /// the whole ring; at most [`SGPR_UBO_MAX_DWORDS`]), else a storage
    /// buffer decorated NonWritable; 2 a storage buffer decorated
    /// NonWritable; 3 as 1, and batched compute translations read a uniform
    /// buffer of [`SGPR_WINDOW_DWORDS`] (the runtime binds a window of the
    /// ring holding the run's table and its commands' SGPR data, the table
    /// offset and the entries relative to it). NVIDIA reads uniform buffers
    /// through the constant cache (operands and uniform registers instead of
    /// per-lane loads).
    pub sgpr_ubo: u8,
    /// GPU-resolved chains (RENDERER.md "GPU-resolved item passes"; needs
    /// `addresses`): bit k set, the plan's load k is read by the shader
    /// itself from the address its base SGPRs hold when it runs, through the
    /// guest page table (the address block's header dwords 2-3 hold the
    /// table's device address; [`PT_BASE`]); its words are not taken from
    /// the SGPR data. Pages the table does not map read 0.
    pub gpu_loads: u64,
    /// Bit r set: buffer resource r (read-only) is addressed through the V#
    /// its instructions' SGPRs hold when they run and the guest page table:
    /// base, stride and range from the V# (as `Vsharp::extent` and
    /// `index_limit`), bytes outside the range read 0; its info block and
    /// address entry are unused.
    pub gpu_res: u64,
    /// GPU-resolved buffers (`gpu_res`) read one buffer that holds
    /// [PT_BASE, PT_END) at guest offsets (front end 2's mirror) instead of
    /// going through the page table: the device address of guest PT_BASE in
    /// it is in the resource's address entry (dwords 0-1, unused otherwise),
    /// addresses outside the span read 0, and their reads are not counted.
    /// GPU-resolved scalar loads read the mirror too (its address in the
    /// address block's header, dwords 2-3), outside the span the zero page,
    /// uncounted (`report_miss`).
    pub pt_linear: bool,
    /// Diagnosis (vertex stage): (a threshold as f32 bits, the program's code hash): every
    /// vertex whose exported position is finite with |x|, |y| or |w| above the threshold counts
    /// in the debug binding's slot of the hash ([`outlier_slot`], [`OUTLIER_STRIDE`]): the
    /// hash, a count, the largest magnitude and a sample position.
    pub outlier: Option<(u32, u64)>,
}

/// The guest page table of GPU-resolved chains (`Spec::gpu_loads`,
/// `gpu_res`): per 4 KiB page of [PT_BASE, PT_END) a u64, the device address
/// where the page's contents are for the GPU (0: none).
pub const PT_BASE: u64 = 0x10_0000_0000;
pub const PT_END: u64 = 0x18_0000_0000;
pub const PT_PAGE_SHIFT: u32 = 12;
/// Byte offset, from the table's address, of the u64 device address of
/// its feedback words (u32: reads through unmapped pages, then the last
/// such guest address, low and high).
pub const PT_FEEDBACK: u64 = ((PT_END - PT_BASE) >> PT_PAGE_SHIFT) * 8;

/// The largest SGPR data a uniform-buffer binding holds (`Spec::sgpr_ubo`;
/// Vulkan guarantees maxUniformBufferRange >= 16 KiB).
pub const SGPR_UBO_MAX_DWORDS: u32 = 4096;
/// A batched compute translation's uniform window of the ring
/// (`Spec::sgpr_ubo` 3; needs maxUniformBufferRange >= 64 KiB).
pub const SGPR_WINDOW_DWORDS: u32 = 16384;

/// Dwords per workgroup in a batched dispatch's table (`Spec::batched`):
/// the command's SGPR data offset, the workgroup's x in its command.
pub const BATCH_ENTRY_DWORDS: u32 = 2;
/// A batched translation's push constant with this bit set holds the
/// command's own SGPR data offset (a dispatch of its own, no table).
pub const BATCH_SINGLE: u32 = 1 << 31;
/// Dwords per draw in a batched draw's table (`Spec::batched`): the vertex
/// and the pixel stage's SGPR data offsets.
pub const BATCH_DRAW_ENTRY_DWORDS: u32 = 2;
/// The location of the flat varying that carries a batched draw's pixel
/// stage's SGPR data offset (the vertex stage's parameters must leave it
/// free: `Output::param_mask` bit 28; a pixel stage reading a parameter
/// there is not translated batched). 28 keeps the interface within the
/// 128-component limits with the built-ins (a pixel stage's up to 12:
/// 28 * 4 + 1 + 12 = 125); at 31 the validation layer reported the batched
/// stages over maxVertexOutputComponents and maxFragmentInputComponents
/// (2026-11-18, while the GPU-resolved draws lost the device).
pub const BATCH_VARYING: u32 = 28;
/// The locations a stage's interface may use (`bary_gs_io`'s free ones lie
/// below it).
pub const VARYING_LOCATIONS: u32 = 31;

/// The byte offset of a stage's SGPR data offset in the push constants
/// (`Spec::sgpr_pc`): the vertex and pixel stages of a draw push both.
pub fn sgpr_pc_offset(stage: Stage) -> u32 {
    match stage {
        Stage::Fragment => 4,
        Stage::Vertex | Stage::Compute => 0,
    }
}

/// As derived, except that `base_instance_sgpr`, `sgpr_pc` and the later
/// switches are hashed only when set: translations and pipelines keyed by
/// the hash of a spec without them keep their keys (pipeline cache files
/// are named by them).
impl std::hash::Hash for Spec {
    fn hash<H: std::hash::Hasher>(&self, state: &mut H) {
        self.stage.hash(state);
        self.user_sgprs.hash(state);
        self.user_base.hash(state);
        self.code_base.hash(state);
        self.wave64.hash(state);
        self.ps_input_addr.hash(state);
        self.ps_inputs.hash(state);
        self.barycentric.hash(state);
        self.mrt.hash(state);
        self.cs_local_size.hash(state);
        self.cs_rsrc2.hash(state);
        self.resources.hash(state);
        self.binding_base.hash(state);
        self.probe.hash(state);
        self.dump.hash(state);
        self.force_dispatcher.hash(state);
        self.dispatcher_for_loops.hash(state);
        self.reg_vars.hash(state);
        self.lane_guards.hash(state);
        self.mip_views.hash(state);
        self.lod_stats.hash(state);
        self.lod_lean.hash(state);
        self.early_tests.hash(state);
        if self.early_ok {
            0x4541_524Cu32.hash(state);
        }
        self.per_lane.hash(state);
        self.lane_votes.hash(state);
        if self.lockstep_off {
            0x4C4F_4646u32.hash(state);
        }
        if self.sg_waves {
            0x5347_5756u32.hash(state);
        }
        if self.hw_bounds {
            0x4842_4E44u32.hash(state);
        }
        if self.dom_nest {
            0x444F_4D4Eu32.hash(state);
        }
        if let Some(s) = self.base_instance_sgpr {
            s.hash(state);
        }
        if let Some(o) = self.sgpr_pc {
            0x5350_4330u32.hash(state);
            o.hash(state);
        }
        if self.addresses {
            0x4144_4452u32.hash(state);
        }
        if self.batched {
            0x4241_5443u32.hash(state);
        }
        if self.write_marks {
            0x4D41_524Bu32.hash(state);
        }
        if self.marks_untouched {
            0x4D55_4E54u32.hash(state);
        }
        if self.mark_twice {
            0x4D32_5458u32.hash(state);
        }
        if self.touch_twice {
            0x5454_5754u32.hash(state);
        }
        if self.part_twice {
            0x5054_5754u32.hash(state);
        }

        if self.dep_loads {
            0x4445_504Cu32.hash(state);
        }
        if self.marks_part {
            0x4D50_5254u32.hash(state);
        }
        if self.marks_held {
            0x4D48_4C44u32.hash(state);
        }
        if self.touch_test {
            0x5454_5354u32.hash(state);
        }
        if self.sgpr_ubo != 0 {
            (0x5542_4F00u32 | self.sgpr_ubo as u32).hash(state);
        }
        if self.gpu_loads != 0 || self.gpu_res != 0 {
            0x4750_5552u32.hash(state);
            self.gpu_loads.hash(state);
            self.gpu_res.hash(state);
            if self.pt_linear {
                0x4C49_4E52u32.hash(state);
            }
        }
        if self.sg_appends {
            0x5347_4150u32.hash(state);
        }
        if self.known_active {
            0x4B41_4354u32.hash(state);
        }
        if self.bary_gs {
            0x4247_5348u32.hash(state);
        }
        if let Some((t, tag)) = self.outlier {
            0x4F55_544Cu32.hash(state);
            t.hash(state);
            tag.hash(state);
        }
    }
}

/// Spill slots of a stage without LDS (`T::spill_var`): one dword per
/// 128 bytes of LDS address (a spill slot holds a dword per lane, 128 bytes
/// for wave32 and 256 for wave64), the first 8 KiB.
pub const SPILL_SLOTS: u32 = 64;

/// Entries of an image table bound as one array of sampled images
/// (`ResKind::ImageTable`); later entries are not reachable.
pub const IMAGE_TABLE_LEN: u32 = 64;

/// Views bound for a storage image whose program selects mip levels
/// (`BindingKind::StorageImageMips`).
pub const MIP_VIEWS: u32 = 16;

/// Count slots per counter bank of the LOD-statistics binding: with
/// `Spec::lod_lean` a subgroup adds its lanes to the slot its position
/// selects (one address per bank held every subgroup's atomic in turn), the
/// CPU sums them.
pub const LOD_COUNT_SLOTS: u32 = 8;
/// The LOD-statistics binding (`Spec::lod_stats`): per counter bank the
/// finest mip sampled (dwords 0..256; 15: none), then LOD_COUNT_SLOTS
/// counts of sampling lanes (dwords 256 + 256 * slot + bank; slot 0 alone
/// without `Spec::lod_lean`).
pub const LOD_STATS_DWORDS: u32 = 256 + 256 * LOD_COUNT_SLOTS;
/// LOD information of a sampled image, filled from its T# when binding
/// (resource info dword 0 of an image; one dword per entry of an image
/// table from [`lod_info_offset`]): counting enabled (T# word 5 bit 25,
/// LOD_HDW_CNT_EN) in bit 31, the T#'s base level in bits 8..11, the
/// counter bank (word 6 bits 0..7) in bits 0..7.
pub const LOD_INFO_ENABLED: u32 = 1 << 31;

/// Probe register flags (see `Spec::probe`).
pub const PROBE_SGPR: u32 = 1 << 12;
pub const PROBE_UINT: u32 = 1 << 13;
pub const PROBE_SINT: u32 = 1 << 14;

/// Dwords per invocation in the debug binding: VGPRs 0..256, SGPRs at 256
/// (128), SCC at 384, execution count at 385.
pub const DUMP_STRIDE: u32 = 512;
/// Size of the debug binding in dwords (up to 8192 invocations: a vertex
/// of each of thousands of instances).
pub const DUMP_DWORDS: u32 = 8192 * DUMP_STRIDE;
/// Flag in a vertex dump's instance count: positions only (see `Spec::dump`).
pub const DUMP_POS: u32 = 1 << 31;
/// Vertex slots per instance of a positions-only dump.
pub const DUMP_POS_VERTICES: u32 = 64;

/// The vertex outlier census (`Spec::outlier`): slots of the debug binding, `OUTLIER_STRIDE`
/// dwords each: [code hash low, high, vertices counted, the largest magnitude (f32 bits), a
/// sample position x, y, z, w (f32 bits), the sample's first two user-data SGPRs, 0, 0].
pub const OUTLIER_SLOTS: u32 = 32768;
pub const OUTLIER_STRIDE: u32 = 12;

/// The census slot of a program's code hash (`Spec::outlier`).
pub fn outlier_slot(tag: u64) -> u32 {
    (tag.wrapping_mul(0x9E37_79B9_7F4A_7C15) >> 49) as u32 & (OUTLIER_SLOTS - 1)
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum BindingKind {
    /// u32 array: user data, resolved scalar loads, resource info blocks
    /// (see [`SGPR_DATA_USER`], [`load_offsets`], [`res_info_offset`]).
    SgprData,
    /// The same data as a uniform buffer (`Spec::sgpr_ubo`): uvec4 array.
    SgprUniform,
    /// Raw u32 storage buffer: a V# range or a global base address.
    Buffer(usize),
    Image(usize),
    /// The 2D variable of an `ImageDim::D3OrD2` resource (`Image` holds
    /// its 3D one).
    ImageAlt(usize),
    StorageImage(usize),
    /// MIP_VIEWS storage images, one view per mip level from the T#'s base
    /// level (views past its last level repeat the last): image_store_mip
    /// and image_load_mip select one by their mip operand, the other
    /// instructions use the first.
    StorageImageMips(usize),
    /// IMAGE_TABLE_LEN sampled images: the entries of an image table.
    ImageArray(usize),
    /// A sampler resource's S# for its instructions without depth comparison.
    Sampler(usize),
    /// The same S# with depth comparison enabled, for its depth-compare
    /// instructions (Vulkan samplers compare only for Dref instructions and
    /// must not be used otherwise; the title uses some samplers both ways).
    CompareSampler(usize),
    /// u32 array: the global data share (GDS_BYTES), shared by all shaders.
    Gds,
    /// u32 array of DUMP_DWORDS: register dumps (`Spec::dump`).
    Debug,
    /// u32 array of WATCHDOG_DWORDS: the loop watchdog's record (see [`LOOP_LIMIT`]).
    Watchdog,
    /// u32 array of LOD_STATS_DWORDS: the LOD statistics (`Spec::lod_stats`).
    LodStats,
    /// A buffer resource read through its device address (`Spec::addresses`):
    /// no descriptor (its `binding` is unused); the runtime places it as a
    /// `Buffer` and writes the address block.
    Address(usize),
    /// u32 array: mirror mode's written-bit map (`Spec::write_marks`).
    Marks,
}

/// Size of the global data share.
pub const GDS_BYTES: u32 = 64 << 10;

/// Loop watchdog: an invocation of a program with backward branches leaves
/// after this many blocks. A lane that never leaves a loop would otherwise
/// run until the driver declares the device lost (seen in Outpost Passage),
/// which also stalls other applications' GPU work. A stopped invocation adds
/// 1 to dword 0 of the watchdog binding and stores its program address
/// (from the SGPR data, `code_base_offset`) in dwords 1 (low) and 2 (high)
/// and its SGPRs s0-s31 in dwords 4-35 (the last stopped invocation's).
/// 2^16 since 2026-11-11 (2^20 before): a culling program's plane loop
/// running on a wrong count took ~450 ms of GPU to reach 2^20 blocks (~430
/// ns a block), a walk's 584 ms frame; level loads (2026-10-12/13) and a
/// 9-level tour (2026-11-11) at 2^14 stopped nothing.
pub const LOOP_LIMIT: u32 = 1 << 16;
/// The limit translations use: LOOP_LIMIT unless set (`set_loop_limit`;
/// diagnosis of device losses, `DS_LOOP_LIMIT`: a lower limit names a
/// shader whose every invocation loops long instead of losing the device).
static LOOP_LIMIT_SET: std::sync::atomic::AtomicU32 = std::sync::atomic::AtomicU32::new(0);

pub fn set_loop_limit(n: u32) {
    LOOP_LIMIT_SET.store(n, std::sync::atomic::Ordering::Relaxed);
}

pub fn loop_limit() -> u32 {
    match LOOP_LIMIT_SET.load(std::sync::atomic::Ordering::Relaxed) {
        0 => LOOP_LIMIT,
        n => n,
    }
}
/// Size of the watchdog binding in dwords (`LOOP_LIMIT`'s record).
pub const WATCHDOG_DWORDS: u32 = 36;
/// The first of the watchdog record's SGPR dwords (s0-s31).
pub const WATCHDOG_SGPRS_AT: u32 = 4;

#[derive(Clone, Debug)]
pub struct Binding {
    pub binding: u32,
    pub kind: BindingKind,
}

impl Binding {
    /// Plan resource index of a resource binding.
    pub fn kind_resource(&self) -> Option<usize> {
        match self.kind {
            BindingKind::Buffer(i) | BindingKind::Address(i) | BindingKind::Image(i) | BindingKind::ImageAlt(i) | BindingKind::StorageImage(i) | BindingKind::StorageImageMips(i) | BindingKind::ImageArray(i) | BindingKind::Sampler(i) | BindingKind::CompareSampler(i) => Some(i),
            _ => None,
        }
    }
}

#[derive(Clone)]
pub struct Output {
    pub spirv: Vec<u32>,
    pub bindings: Vec<Binding>,
    /// Number of vertex parameters exported (vertex) / highest attribute read (fragment).
    pub params: u32,
    /// Bit mask of MRTs written (fragment).
    pub mrt_mask: u32,
    /// Bit mask of parameter locations exported (vertex).
    pub param_mask: u32,
    pub writes_depth: bool,
    pub warnings: Vec<String>,
    /// The stage reads its SGPR data offset from the push constants at this
    /// byte offset (`Spec::sgpr_pc`).
    pub sgpr_pc: Option<u32>,
    /// `Spec::batched`: the SGPR data binding is the whole ring and the push
    /// constant at byte 0 the dword offset of the workgroups' table.
    pub batched: bool,
    /// Lane-dependent branches lockstep mode decided without a vote
    /// (`masked_skip`).
    pub masked_skips: u32,
    /// The lane model translated (statistics): lockstep, and wave-mask mode
    /// within it.
    pub lockstep: bool,
    pub wave_masks: bool,
    /// Wave-mask mode per subgroup (`Spec::sg_waves`).
    pub sg_waves: bool,
    /// Wave-aggregated appends done per subgroup ([`Appends`]).
    pub sg_appends: u32,
    /// Instructions known active (`Spec::known_active`).
    pub known_active: u32,
    /// Buffer resources (bit n: resource n) whose scalar loads have no range
    /// check (`Spec::hw_bounds`): bound with a range ending where the checks
    /// did, dword (bias + o) / 4 in range exactly when o < size, for any bias
    /// (the binding's bias rounded down to a dword plus the size rounded up).
    pub unchecked_res: u64,
    /// The input Locations the stage declares (bit n = Location n; a pixel
    /// stage's: a rect-list draw's geometry shader outputs each of them).
    pub in_mask: u32,
    /// A pixel stage translated with `Spec::bary_gs` that reads per vertex:
    /// the locations its geometry shader writes (`gs::bary_gs`).
    pub bary_gs: Option<gs::BaryIo>,
}

/// Dwords of user data at the start of the SGPR data buffer.
pub const SGPR_DATA_USER: u32 = 32;
/// Dwords per resource info block: [byte bias of the resource base within
/// the bound range, stride, bound bytes, records an index may address
/// (structured range check; u32::MAX: none)].
pub const RES_INFO_DWORDS: u32 = 4;

/// Offsets (in dwords) of each scalar load in the SGPR data buffer.
pub fn load_offsets(plan: &Plan) -> Vec<u32> {
    let mut v = Vec::with_capacity(plan.loads.len());
    let mut off = SGPR_DATA_USER;
    for l in &plan.loads {
        v.push(off);
        off += l.dwords;
    }
    v
}

/// Dword offset of the first resource info block in the SGPR data buffer.
pub fn res_info_offset(plan: &Plan) -> u32 {
    SGPR_DATA_USER + plan.loads.iter().map(|l| l.dwords).sum::<u32>()
}

/// Dword offset of the program's address (low, high dword) in the SGPR
/// data buffer, after the resource info blocks: s_getpc_b64 and the loop
/// watchdog read it, so one translation serves every copy of a program
/// (the title loads the same code at several addresses, and streamed
/// shaders land at different addresses in every run).
pub fn code_base_offset(plan: &Plan) -> u32 {
    res_info_offset(plan) + RES_INFO_DWORDS * plan.resources.len() as u32
}

/// Dword offset of the image tables' LOD information (IMAGE_TABLE_LEN
/// dwords per image table resource, in plan order; [`LOD_INFO_ENABLED`]),
/// after the program's address.
pub fn lod_info_offset(plan: &Plan) -> u32 {
    code_base_offset(plan) + 2
}

/// Position of image table resource `r` among the plan's image tables.
pub fn table_slot(plan: &Plan, r: usize) -> u32 {
    plan.resources[..r].iter().filter(|x| x.kind == ResKind::ImageTable).count() as u32
}

/// Dwords per resource in the address block (`Spec::addresses`): the
/// address of the bound range (low, high dword), its size in bytes, and its
/// first word in the written-bit map (`Spec::write_marks`).
pub const ADDR_INFO_DWORDS: u32 = 4;
/// Dwords before the resources' entries in the address block: the zero
/// page's address (low, high; out-of-range reads load from it), the page
/// table's.
pub const ADDR_HEADER_DWORDS: u32 = 4;

/// Dword offset of the address block in the SGPR data buffer, after the
/// image tables' LOD information (the earlier layout ended there).
pub fn addr_info_offset(plan: &Plan) -> u32 {
    lod_info_offset(plan) + IMAGE_TABLE_LEN * plan.resources.iter().filter(|x| x.kind == ResKind::ImageTable).count() as u32
}

/// Total dwords of the SGPR data buffer.
pub fn sgpr_data_dwords(plan: &Plan) -> u32 {
    addr_info_offset(plan) + ADDR_HEADER_DWORDS + ADDR_INFO_DWORDS * plan.resources.len() as u32
}

pub(crate) struct Types {
    pub bool_: Word,
    pub u32: Word,
    pub i32: Word,
    pub f32: Word,
    pub u64: Word,
    pub v2f: Word,
    pub v3f: Word,
    pub v4f: Word,
    pub v2u: Word,
    pub v3u: Word,
    pub v4u: Word,
    pub v4i: Word,
    pub v2i: Word,
    pub v3i: Word,
    pub fn_u32: Word,
    pub fn_bool: Word,
    pub sb_u32: Word,
}

#[derive(Clone, Copy)]
pub(crate) struct ImageRes {
    pub var: Word,
    pub img_ty: Word,
    pub dim: ImageDim,
    pub class: NumClass,
    pub storage: bool,
    /// Bound as a depth/stencil view (`ResSpec::Image::depth`).
    pub depth: bool,
    /// Bound as MIP_VIEWS views, one per mip level (`var` is their array).
    pub mips: bool,
}

pub(crate) struct T<'a> {
    pub b: Builder,
    pub t: Types,
    pub glsl: Word,
    pub spec: &'a Spec,
    pub plan: &'a Plan,
    /// Register files as arrays (programs with relative addressing,
    /// s_movrel* / v_movrel*), else 0 ...
    pub sgpr: Word,
    pub vgpr: Word,
    /// ... and one variable per register instead (no access chain per use:
    /// the arrays made about a third of the translated modules' words).
    pub sregs: Vec<Word>,
    pub vregs: Vec<Word>,
    /// This lane's EXEC bit as computed in a block (function, block): reused
    /// there until EXEC is written (it was recomputed for every VGPR write).
    active_cache: Option<((usize, usize), Word)>,
    pub scc: Word,
    pub consts: HashMap<(Word, u32), Word>,
    pub interface: Vec<Word>,
    pub sgpr_data: Word,
    /// The SGPR data is a uniform buffer (`Spec::sgpr_ubo`): the pointer
    /// type of its dwords (Uniform u32).
    pub sgpr_ub: Option<Word>,
    /// The command's offset into the SGPR data binding (`Spec::sgpr_pc`),
    /// loaded in the entry block.
    pub sgpr_base: Option<Word>,
    pub load_offsets: Vec<u32>,
    pub res_info: u32,
    /// Offset of the program's address in the SGPR data.
    pub code_base_at: u32,
    pub buffers: HashMap<usize, Word>,
    pub images: HashMap<usize, ImageRes>,
    /// `ImageDim::D3OrD2` resources' 2D variables (`images` holds the 3D
    /// one).
    pub image_alts: HashMap<usize, ImageRes>,
    /// Image tables: resource -> pc of the load that reads its T#, and the
    /// byte offset each such load used (Function variables).
    pub image_tables: HashMap<usize, u32>,
    pub table_offsets: HashMap<u32, Word>,
    /// Sampler resources, and their variables (declared at first use) for
    /// plain and for depth-compare instructions.
    pub sampler_res: BTreeSet<usize>,
    pub samplers: HashMap<usize, Word>,
    pub compare_sampler_vars: HashMap<usize, Word>,
    pub sampler_ty: Word,
    pub bindings: Vec<Binding>,
    pub next_binding: u32,
    // stage interface
    pub out_pos: Option<Word>,
    pub out_params: HashMap<u32, Word>,
    pub out_mrt: HashMap<u32, (Word, Word)>,
    pub out_depth: Option<Word>,
    /// Fragment inputs by location (vertex parameter): one variable per
    /// location, shared by the attributes that read it.
    pub in_attrs: HashMap<u32, Word>,
    /// Locations read per vertex (`per_vertex_locations`, with
    /// `Spec::barycentric`), and their per-vertex input arrays.
    pub per_vertex_locs: BTreeSet<u32>,
    pub per_vertex_in: HashMap<u32, Word>,
    /// `Spec::bary_gs`: where the geometry shader puts each vertex's values
    /// and the barycentrics, and the three flat inputs of each per-vertex
    /// location.
    pub bary_io: Option<gs::BaryIo>,
    pub per_vertex_gs: HashMap<u32, [Word; 3]>,
    /// BaryCoordKHR / BaryCoordNoPerspKHR inputs (loaded at entry).
    pub bary: Option<(Word, Word)>,
    pub frag_coord: Option<Word>,
    pub lds: Option<Word>,
    /// Register spills of stages without LDS (`spill_var`).
    pub spill: Option<Word>,
    /// GDS storage buffer (when the program uses GDS instructions).
    pub gds: Option<Word>,
    /// The written-bit map binding (`Spec::write_marks`, programs with
    /// written buffers).
    pub marks: Option<Word>,
    /// Hardware lane index within the wave (compute: from the local index).
    pub lane_id: Option<Word>,
    /// LocalInvocationIndex (compute).
    pub local_index: Option<Word>,
    /// Workgroup array for wave64 lane exchanges (created on first use).
    pub xchg: Option<Word>,
    /// Workgroup array for wave64 first-lane reads: values, then flags.
    pub xchg_first: Option<Word>,
    /// Blocks every invocation of the workgroup executes alike (barriers allowed).
    pub uniformity: crate::uniform::Uniformity,
    /// Lockstep mode (see the module documentation).
    pub lockstep: bool,
    /// Lockstep ballots: workgroup u32[6] (three rotating (lo, hi) slots)
    /// and this invocation's ballot counter (mod 3).
    pub vote: Option<Word>,
    pub vote_ctr: Option<Word>,
    /// Sync blocks (see the module documentation) and the dispatcher steps
    /// of the current round.
    pub sync_starts: Vec<u32>,
    pub sync_steps: Option<Word>,
    /// LDS accesses since the last workgroup barrier: bit 0 reads, bit 1
    /// writes (3 at block entry: unknown).
    pub lds_state: u8,
    /// Inside a lane guard (`emit_block_body`): this lane is active, so
    /// register writes need no select and `active` is true.
    pub lane_guard: bool,
    /// Lane-dependent branches decided without a vote (`masked_skip`).
    pub masked_skips: u32,
    /// Inside a lane guard, for the current block only (values made in a
    /// nested construct do not dominate the code after it): VGPR values
    /// written or loaded there, and bitcast pairs (f32 <-> u32) made there,
    /// reused instead of another load or a bitcast back.
    vcache: HashMap<u32, Word>,
    bitcasts: HashMap<Word, Word>,
    vcache_block: Option<(usize, usize)>,
    /// SubgroupLocalInvocationId input (created on first use).
    pub sg_lane_var: Option<Word>,
    /// (vgpr, lane) slots written by v_writelane with constant lanes (SGPR
    /// spills): per-invocation copies of those lanes' values.
    pub lane_slots: HashMap<(u32, u32), Word>,
    /// Lane reads that take the invocation's own value (see
    /// [`atomic_result_reads`]).
    pub own_lane_reads: BTreeSet<u32>,
    /// Lane-0 elections: compare pc -> source operand (0/1) holding the lane
    /// index, compared as lane 0 (see [`lane0_elections`]).
    pub elections: HashMap<u32, usize>,
    /// Wave-aggregated appends done per subgroup ([`Appends`]), and the
    /// subgroup index of the invocation adding the count (`subgroup_leader`).
    pub sg: Appends,
    pub leader: Option<Word>,
    /// Pcs at whose start this invocation's EXEC bit is known set
    /// (`Spec::known_active`), and the instruction being translated.
    pub known_active: crate::HashSet<u32>,
    pub cur_pc: u32,
    /// Buffers only scalar loads read (`Spec::hw_bounds`), and those whose
    /// loads went unchecked (`Output::unchecked_res`).
    pub scalar_only: u64,
    pub unchecked_res: u64,
    /// Input Locations declared (`Output::in_mask`).
    pub in_locs: u32,
    /// Wave-mask mode (see the module documentation).
    pub wave_masks: bool,
    /// Subgroup waves (`Spec::sg_waves`).
    pub sg_waves: bool,
    /// [`FORMAT_RT`] buffers: [`rt_format_entry`]'s table (a private array
    /// of 128 entries, made when the spec has such a buffer), and the
    /// instruction's V# word 3 while a typed access is translated.
    pub fmt_table: Option<Word>,
    pub rt_word3: Option<Word>,
    /// Wave-mask mode: pcs of the mask writes that produce complete masks.
    pub complete: crate::HashSet<u32>,
    /// Wave-mask mode: whether this lane's bit lies in the high dword of a
    /// mask, and its position in that dword.
    pub own_bit: Option<(Word, Word)>,
    pub warnings: Vec<String>,
    pub params: u32,
    pub mrt_mask: u32,
    pub pc_var: Word,
    /// Probe value (see `Spec::probe`).
    pub probe_var: Option<Word>,
    /// Loop watchdog binding and this invocation's block counter (programs
    /// with backward branches; see [`LOOP_LIMIT`]).
    pub watchdog: Option<Word>,
    pub loop_ctr: Option<Word>,
    /// Debug binding (see `Spec::dump`) and WorkgroupId (compute).
    pub dump_buf: Option<Word>,
    /// The LOD-statistics binding (`Spec::lod_stats`).
    pub lod_stats_var: Option<Word>,
    /// gl_HelperInvocation (fragment LOD statistics skip helper lanes).
    pub helper_var: Option<Word>,
    pub wg_id: Option<Word>,
    /// VertexIndex and InstanceIndex (vertex stage; register dumps).
    pub vertex_ids: Option<(Word, Word)>,
    pub sb_struct: Option<Word>,
    /// Binding of each raw buffer variable, and its uvec2/uvec4 views
    /// (aliases of the same binding for aligned 8/16-byte accesses).
    pub buffer_binding: HashMap<Word, u32>,
    pub buffer_views: HashMap<(Word, u32), Word>,
    /// Block struct types of the vector views (created once: types are deduplicated).
    pub view_structs: HashMap<u32, Word>,
    /// `Spec::addresses`: the resource index of each buffer resource's
    /// placeholder id (stands for the resource where a variable would).
    pub buffer_res: HashMap<Word, usize>,
    /// Dword offset of the address block in the SGPR data.
    pub addr_info_at: u32,
    /// Address-mode resources' base address (u64) and range, and the zero
    /// page's address, loaded once in the entry block (`mem::preload_addresses`):
    /// loaded at every access they put three dependent reads before each
    /// load and store (the gate ran 35% slower, GPU-bound).
    pub addr_vals: HashMap<usize, (Word, Word)>,
    pub zero_addr: Option<Word>,
    /// `Spec::batched`: this workgroup's x within its command, and its
    /// WorkgroupId as loaded (loaded once in the entry block).
    pub batch_wg: Option<(Word, Word)>,
    /// GPU-resolved chains: the guest page table's device address (loaded
    /// in the entry block), the V# of the buffer instruction being
    /// translated when its resource is GPU-resolved (resource, base u64,
    /// stride, index limit, range in bytes), and u64 constants.
    pub pt_addr: Option<Word>,
    pub rt_vsharp: Option<RtVsharp>,
    pub consts64: HashMap<u64, Word>,
}

/// A GPU-resolved buffer resource's V# as its instruction reads it.
#[derive(Clone, Copy)]
pub(crate) struct RtVsharp {
    pub r: usize,
    /// The V#'s base aligned down to 16 bytes, and the bytes it was moved
    /// (offsets are from the aligned base, as the CPU path's placements).
    pub base: Word,
    pub bias: Word,
    pub stride: Word,
    pub limit: Word,
    pub range: Word,
}

pub type R<T> = Result<T, String>;

pub(crate) fn e<E: std::fmt::Debug>(r: Result<Word, E>) -> R<Word> {
    r.map_err(|x| format!("spirv builder: {x:?}"))
}

pub(crate) fn ok<E: std::fmt::Debug>(r: Result<(), E>) -> R<()> {
    r.map_err(|x| format!("spirv builder: {x:?}"))
}

impl<'a> T<'a> {
    pub fn u(&mut self, v: u32) -> Word {
        let ty = self.t.u32;
        *self.consts.entry((ty, v)).or_insert_with(|| self.b.constant_bit32(ty, v))
    }
    pub fn u64c(&mut self, v: u64) -> Word {
        let ty = self.t.u64;
        *self.consts64.entry(v).or_insert_with(|| self.b.constant_bit64(ty, v))
    }
    pub fn i(&mut self, v: i32) -> Word {
        let ty = self.t.i32;
        *self.consts.entry((ty, v as u32)).or_insert_with(|| self.b.constant_bit32(ty, v as u32))
    }
    pub fn f(&mut self, v: f32) -> Word {
        let ty = self.t.f32;
        *self.consts.entry((ty, v.to_bits())).or_insert_with(|| self.b.constant_bit32(ty, v.to_bits()))
    }
    pub fn bool_c(&mut self, v: bool) -> Word {
        let ty = self.t.bool_;
        *self.consts.entry((ty, v as u32)).or_insert_with(|| if v { self.b.constant_true(ty) } else { self.b.constant_false(ty) })
    }

    fn reg_ptr(&mut self, file: Word, idx: u32) -> R<Word> {
        let i = self.u(idx);
        let pt = self.t.fn_u32;
        e(self.b.access_chain(pt, None, file, [i]))
    }

    fn sreg_ptr(&mut self, idx: u32) -> R<Word> {
        match self.sregs.get(idx as usize) {
            Some(&v) => Ok(v),
            None => self.reg_ptr(self.sgpr, idx),
        }
    }
    fn vreg_ptr(&mut self, idx: u32) -> R<Word> {
        match self.vregs.get(idx as usize) {
            Some(&v) => Ok(v),
            None => self.reg_ptr(self.vgpr, idx),
        }
    }

    /// Raw 32-bit value of a scalar register slot (0..127; NULL reads 0).
    pub fn sreg(&mut self, idx: u32) -> R<Word> {
        if idx == crate::OP_NULL as u32 || idx >= 128 {
            return Ok(self.u(0));
        }
        let p = self.sreg_ptr(idx)?;
        e(self.b.load(self.t.u32, None, p, None, []))
    }
    pub fn set_sreg(&mut self, idx: u32, v: Word) -> R<()> {
        if idx >= 128 || idx == crate::OP_NULL as u32 {
            return Ok(()); // writes to constants / null are ignored
        }
        if idx == crate::OP_EXEC_LO as u32 || idx == crate::OP_EXEC_HI as u32 {
            self.active_cache = None;
        }
        let p = self.sreg_ptr(idx)?;
        ok(self.b.store(p, v, None, []))
    }
    pub fn vreg(&mut self, idx: u32) -> R<Word> {
        if self.lane_guard && self.vcache_here() {
            if let Some(&v) = self.vcache.get(&(idx & 255)) {
                return Ok(v);
            }
        }
        let p = self.vreg_ptr(idx & 255)?;
        let v = e(self.b.load(self.t.u32, None, p, None, []))?;
        if self.lane_guard {
            self.vcache.insert(idx & 255, v);
        }
        Ok(v)
    }
    /// Unconditional VGPR write (initialization, lane-exact writes).
    pub fn set_vreg_raw(&mut self, idx: u32, v: Word) -> R<()> {
        let p = self.vreg_ptr(idx & 255)?;
        ok(self.b.store(p, v, None, []))?;
        if self.lane_guard && self.vcache_here() {
            self.vcache.insert(idx & 255, v);
        }
        Ok(())
    }
    /// Whether the lane guard's caches hold for the block being emitted
    /// (else they are emptied and tied to it).
    fn vcache_here(&mut self) -> bool {
        let here = self.b.selected_function().zip(self.b.selected_block());
        if here.is_some() && here == self.vcache_block {
            return true;
        }
        self.vcache.clear();
        self.bitcasts.clear();
        self.vcache_block = here;
        false
    }
    /// The label of the block being emitted (a phi's predecessor after code
    /// that may have opened blocks).
    pub fn current_label(&self) -> R<Word> {
        let (f, bi) = self.b.selected_function().zip(self.b.selected_block()).ok_or("no block being emitted")?;
        self.b.module_ref().functions[f].blocks[bi].label.as_ref().and_then(|l| l.result_id).ok_or_else(|| "a block without its label".into())
    }

    /// Empties the lane guard's caches (a guard begins or ends).
    pub fn reset_vcache(&mut self) {
        self.vcache.clear();
        self.bitcasts.clear();
        self.vcache_block = None;
    }
    /// VGPR write predicated on this lane being active.
    pub fn set_vreg(&mut self, idx: u32, v: Word) -> R<()> {
        if self.lane_guard {
            return self.set_vreg_raw(idx, v);
        }
        let act = self.active()?;
        if act == self.bool_c(true) {
            return self.set_vreg_raw(idx, v);
        }
        let old = self.vreg(idx)?;
        let nv = e(self.b.select(self.t.u32, None, act, v, old))?;
        self.set_vreg_raw(idx, nv)
    }
    pub fn active(&mut self) -> R<Word> {
        if self.lane_guard || self.known_active.contains(&self.cur_pc) {
            return Ok(self.bool_c(true));
        }
        let here = self.b.selected_function().zip(self.b.selected_block());
        if let (Some((at, v)), Some(h)) = (self.active_cache, here) {
            if at == h {
                return Ok(v);
            }
        }
        let lo = self.sreg(crate::OP_EXEC_LO as u32)?;
        let hi = if self.own_bit.is_some() { self.sreg(crate::OP_EXEC_HI as u32)? } else { self.u(0) };
        let v = self.mask_bit(lo, hi)?;
        // Not with relative addressing: a v/s_movreld could write EXEC
        // through the array.
        if self.sregs.is_empty() {
            self.active_cache = None;
        } else if let Some(h) = here {
            self.active_cache = Some((h, v));
        }
        Ok(v)
    }

    /// This lane's bit of a lane mask (lo, hi): bit 0, or in wave-mask mode
    /// the bit of the lane's index.
    pub fn mask_bit(&mut self, lo: Word, hi: Word) -> R<Word> {
        let u = self.t.u32;
        let w = match self.own_bit {
            Some((in_hi, sh)) => {
                let w = e(self.b.select(u, None, in_hi, hi, lo))?;
                e(self.b.shift_right_logical(u, None, w, sh))?
            }
            None => lo,
        };
        let one = self.u(1);
        let zero = self.u(0);
        let b = e(self.b.bitwise_and(u, None, w, one))?;
        e(self.b.i_not_equal(self.t.bool_, None, b, zero))
    }

    /// Lane mask (lo, hi) of a per-lane condition written at `pc` (lanes
    /// outside EXEC 0): this lane's bit alone (bit 0; in wave-mask mode the
    /// bit of its index), or in wave-mask mode the complete mask where the
    /// value may be read as a whole (a ballot).
    pub fn lane_mask(&mut self, pc: u32, c: Word) -> R<(Word, Word)> {
        let u = self.t.u32;
        let z = self.u(0);
        if c == self.bool_c(false) {
            return Ok((z, z));
        }
        let act = self.active()?;
        let c = e(self.b.logical_and(self.t.bool_, None, c, act))?;
        match self.own_bit {
            Some(_) if self.complete.contains(&pc) => {
                if self.sg_waves {
                    self.subgroup_ballot(c)
                } else {
                    self.wave_ballot(c)
                }
            }
            Some((in_hi, sh)) => {
                let one = self.u(1);
                let bit = e(self.b.shift_left_logical(u, None, one, sh))?;
                let m = e(self.b.select(u, None, c, bit, z))?;
                let lo = e(self.b.select(u, None, in_hi, z, m))?;
                let hi = e(self.b.select(u, None, in_hi, m, z))?;
                Ok((lo, hi))
            }
            None if self.sg.ballots.contains(&pc) => self.subgroup_ballot(c),
            None => Ok((self.bool_to_u(c)?, z)),
        }
    }

    /// The mask of the subgroup's invocations with `c` in the hardware
    /// layout (bit i for lane i of the wave; other subgroups' lanes 0): an
    /// append's ballot done per subgroup ([`Appends`]), or a complete mask
    /// of subgroup waves.
    pub fn subgroup_ballot(&mut self, c: Word) -> R<(Word, Word)> {
        let u = self.t.u32;
        let lane = self.lane()?;
        let one = self.u(1);
        let k31 = self.u(31);
        let pos = e(self.b.bitwise_and(u, None, lane, k31))?;
        let bit = e(self.b.shift_left_logical(u, None, one, pos))?;
        let z = self.u(0);
        let mine = e(self.b.select(u, None, c, bit, z))?;
        self.b.capability(Capability::GroupNonUniform);
        self.b.capability(Capability::GroupNonUniformArithmetic);
        let sg = self.u(spirv::Scope::Subgroup as u32);
        if !self.spec.wave64 {
            let lo = e(self.b.group_non_uniform_bitwise_or(u, None, sg, spirv::GroupOperation::Reduce, mine, None))?;
            return Ok((lo, z));
        }
        let k32 = self.u(32);
        let lower = e(self.b.u_less_than(self.t.bool_, None, lane, k32))?;
        let lo_part = e(self.b.select(u, None, lower, mine, z))?;
        let hi_part = e(self.b.select(u, None, lower, z, mine))?;
        let lo = e(self.b.group_non_uniform_bitwise_or(u, None, sg, spirv::GroupOperation::Reduce, lo_part, None))?;
        let hi = e(self.b.group_non_uniform_bitwise_or(u, None, sg, spirv::GroupOperation::Reduce, hi_part, None))?;
        Ok((lo, hi))
    }

    /// Whether this invocation is the first of its subgroup with its lane
    /// active (EXEC): the one that adds an append's count ([`Appends`]). Its
    /// subgroup index is kept for `leader_read`.
    pub fn subgroup_leader(&mut self) -> R<Word> {
        let u = self.t.u32;
        let bt = self.t.bool_;
        let act = self.active()?;
        self.b.capability(Capability::GroupNonUniform);
        self.b.capability(Capability::GroupNonUniformBallot);
        self.b.capability(Capability::GroupNonUniformVote);
        let sg = self.u(spirv::Scope::Subgroup as u32);
        let ballot = e(self.b.group_non_uniform_ballot(self.t.v4u, None, sg, act))?;
        let first = e(self.b.group_non_uniform_ballot_find_lsb(u, None, sg, ballot))?;
        let id = self.subgroup_lane()?;
        let me = e(self.b.i_equal(bt, None, id, first))?;
        let lead = e(self.b.logical_and(bt, None, act, me))?;
        // (None active: no leader; reads then take invocation 0's value.)
        let any = e(self.b.group_non_uniform_any(bt, None, sg, act))?;
        let z = self.u(0);
        let first = e(self.b.select(u, None, any, first, z))?;
        let Some(var) = self.leader else { return Err("append leader without its variable".into()) };
        ok(self.b.store(var, first, None, []))?;
        Ok(lead)
    }

    /// `v`'s value in the invocation `subgroup_leader` chose last: a lane
    /// read of an append's result ([`Appends`]).
    pub fn leader_read(&mut self, v: Word) -> R<Word> {
        let Some(var) = self.leader else { return Err("append result read without a leader".into()) };
        let id = e(self.b.load(self.t.u32, None, var, None, []))?;
        let (x, _) = self.lane_fetch(v, id)?;
        Ok(x)
    }
    pub fn get_scc(&mut self) -> R<Word> {
        e(self.b.load(self.t.bool_, None, self.scc, None, []))
    }
    pub fn set_scc(&mut self, v: Word) -> R<()> {
        ok(self.b.store(self.scc, v, None, []))
    }

    /// 32-bit value of an operand (SGPR, VGPR, inline constant, literal, special).
    pub fn operand(&mut self, ins: &Inst, op: u16) -> R<Word> {
        Ok(match op {
            0..=127 => self.sreg(op as u32)?,
            128..=192 => self.u((op - 128) as u32),
            193..=208 => self.u((-((op - 192) as i32)) as u32),
            240 => self.u(0.5f32.to_bits()),
            241 => self.u((-0.5f32).to_bits()),
            242 => self.u(1.0f32.to_bits()),
            243 => self.u((-1.0f32).to_bits()),
            244 => self.u(2.0f32.to_bits()),
            245 => self.u((-2.0f32).to_bits()),
            246 => self.u(4.0f32.to_bits()),
            247 => self.u((-4.0f32).to_bits()),
            248 => self.u(0x3E22_F983), // 1 / (2 * pi)
            crate::OP_VCCZ | crate::OP_EXECZ => {
                let r = if op == crate::OP_VCCZ { crate::OP_VCC_LO } else { crate::OP_EXEC_LO } as u32;
                let lo = self.sreg(r)?;
                let hi = if self.spec.wave64 { self.sreg(r + 1)? } else { self.u(0) };
                let o = e(self.b.bitwise_or(self.t.u32, None, lo, hi))?;
                let z = self.u(0);
                let c = e(self.b.i_equal(self.t.bool_, None, o, z))?;
                self.bool_to_u(c)?
            }
            crate::OP_SCC => {
                let s = self.get_scc()?;
                self.bool_to_u(s)?
            }
            crate::OP_LITERAL => self.u(ins.lit.unwrap_or(0)),
            256..=511 => self.vreg((op - 256) as u32)?,
            _ => {
                self.warn(format!("{:#x}: unsupported operand {}", ins.pc, crate::operand_name(op)));
                self.u(0)
            }
        })
    }

    pub fn to_f(&mut self, v: Word) -> R<Word> {
        self.bitcast_cached(self.t.f32, v)
    }
    pub fn to_u(&mut self, v: Word) -> R<Word> {
        self.bitcast_cached(self.t.u32, v)
    }
    /// A bitcast between f32 and u32; inside a lane guard the value it was
    /// made from when that is the other half of an earlier pair.
    fn bitcast_cached(&mut self, ty: Word, v: Word) -> R<Word> {
        if self.lane_guard && self.vcache_here() {
            if let Some(&x) = self.bitcasts.get(&v) {
                return Ok(x);
            }
            let x = e(self.b.bitcast(ty, None, v))?;
            self.bitcasts.insert(v, x);
            self.bitcasts.insert(x, v);
            return Ok(x);
        }
        e(self.b.bitcast(ty, None, v))
    }
    pub fn to_i(&mut self, v: Word) -> R<Word> {
        e(self.b.bitcast(self.t.i32, None, v))
    }
    pub fn glsl1(&mut self, ty: Word, op: spirv::GlslStd450Op, a: Word) -> R<Word> {
        e(self.b.ext_inst(ty, None, self.glsl, op as u32, [Operand::IdRef(a)]))
    }
    pub fn glsl2(&mut self, ty: Word, op: spirv::GlslStd450Op, a: Word, c: Word) -> R<Word> {
        e(self.b.ext_inst(ty, None, self.glsl, op as u32, [Operand::IdRef(a), Operand::IdRef(c)]))
    }
    pub fn glsl3(&mut self, ty: Word, op: spirv::GlslStd450Op, a: Word, c: Word, d: Word) -> R<Word> {
        e(self.b.ext_inst(ty, None, self.glsl, op as u32, [Operand::IdRef(a), Operand::IdRef(c), Operand::IdRef(d)]))
    }
    pub fn bool_to_u(&mut self, c: Word) -> R<Word> {
        let one = self.u(1);
        let z = self.u(0);
        e(self.b.select(self.t.u32, None, c, one, z))
    }
    pub fn warn(&mut self, s: String) {
        if !self.warnings.contains(&s) {
            self.warnings.push(s);
        }
    }

    /// The per-invocation private array holding a stage's register spills
    /// through the lane-addressed LDS forms when the stage has no LDS
    /// (vertex, pixel; `mem::ds`), made at its first use.
    pub fn spill_var(&mut self) -> Word {
        if let Some(v) = self.spill {
            return v;
        }
        let n = self.u(SPILL_SLOTS);
        let arr = self.b.type_array(self.t.u32, n);
        let pt = self.b.type_pointer(None, StorageClass::Private, arr);
        let v = self.global_var(pt, StorageClass::Private);
        self.spill = Some(v);
        v
    }

    /// Reads dword `idx` of the SGPR data buffer.
    pub fn sgpr_data(&mut self, idx: u32) -> R<Word> {
        if let (Some(pt), false) = (self.sgpr_ub, self.spec.batched) {
            // A command's own uniform buffer (no command base): constant
            // element and component.
            let z = self.u(0);
            let q = self.u(idx / 4);
            let r = self.u(idx % 4);
            let p = e(self.b.access_chain(pt, None, self.sgpr_data, [z, q, r]))?;
            return e(self.b.load(self.t.u32, None, p, None, []));
        }
        let i = self.u(idx);
        let i = self.sgpr_index(i)?;
        self.sgpr_raw(i)
    }

    /// Reads dword `i` of the SGPR data binding itself (no command offset).
    pub fn sgpr_raw(&mut self, i: Word) -> R<Word> {
        let z = self.u(0);
        if let Some(pt) = self.sgpr_ub {
            let two = self.u(2);
            let three = self.u(3);
            let q = e(self.b.shift_right_logical(self.t.u32, None, i, two))?;
            let r = e(self.b.bitwise_and(self.t.u32, None, i, three))?;
            let p = e(self.b.access_chain(pt, None, self.sgpr_data, [z, q, r]))?;
            return e(self.b.load(self.t.u32, None, p, None, []));
        }
        let p = e(self.b.access_chain(self.t.sb_u32, None, self.sgpr_data, [z, i]))?;
        e(self.b.load(self.t.u32, None, p, None, []))
    }

    /// The SGPR data binding's index of dword `i` of the command's data:
    /// offset by the command's base with `Spec::sgpr_pc`.
    pub fn sgpr_index(&mut self, i: Word) -> R<Word> {
        if self.spec.sgpr_pc.is_none() && !self.spec.batched {
            return Ok(i);
        }
        let base = self.sgpr_base.ok_or("SGPR data read before its offset was loaded")?;
        e(self.b.i_add(self.t.u32, None, base, i))
    }

    /// The program's address (low, high dword) from the SGPR data.
    pub fn code_base(&mut self) -> R<(Word, Word)> {
        let i = self.code_base_at;
        Ok((self.sgpr_data(i)?, self.sgpr_data(i + 1)?))
    }

    /// Dword `k` of resource `r`'s info block (0 bias, 1 stride, 2 num_records).
    pub fn res_info(&mut self, r: usize, k: u32) -> R<Word> {
        let idx = self.res_info + RES_INFO_DWORDS * r as u32 + k;
        self.sgpr_data(idx)
    }

    /// Dword `k` of resource `r`'s address entry (`Spec::addresses`: 0, 1
    /// the address of its bound range, 2 the range's size in bytes).
    pub fn addr_info(&mut self, r: usize, k: u32) -> R<Word> {
        let idx = self.addr_info_at + ADDR_HEADER_DWORDS + ADDR_INFO_DWORDS * r as u32 + k;
        self.sgpr_data(idx)
    }

    /// Dword `k` of the address block's header (0, 1: the zero page).
    pub fn addr_header(&mut self, k: u32) -> R<Word> {
        let idx = self.addr_info_at + k;
        self.sgpr_data(idx)
    }

    /// Creates a module-scope OpVariable (never inside the current block).
    pub fn global_var(&mut self, ptr_ty: Word, sc: StorageClass) -> Word {
        let id = self.b.id();
        let inst = rspirv::dr::Instruction::new(spirv::Op::Variable, Some(ptr_ty), Some(id), vec![Operand::StorageClass(sc)]);
        self.b.module_mut().types_global_values.push(inst);
        id
    }

    /// The variable of sampler resource `i` for instructions with (`compare`)
    /// or without depth comparison, declared and bound at its first use.
    pub fn sampler_var(&mut self, i: usize, compare: bool) -> R<Word> {
        let known = if compare { self.compare_sampler_vars.get(&i) } else { self.samplers.get(&i) };
        if let Some(&v) = known {
            return Ok(v);
        }
        if !self.sampler_res.contains(&i) {
            return Err(format!("resource {i} is not a sampler"));
        }
        let pt = self.b.type_pointer(None, StorageClass::UniformConstant, self.sampler_ty);
        let v = self.global_var(pt, StorageClass::UniformConstant);
        let bnd = self.alloc_binding(if compare { BindingKind::CompareSampler(i) } else { BindingKind::Sampler(i) });
        self.decorate_binding(v, bnd);
        if compare {
            self.compare_sampler_vars.insert(i, v);
        } else {
            self.samplers.insert(i, v);
        }
        Ok(v)
    }

    fn alloc_binding(&mut self, kind: BindingKind) -> u32 {
        let n = self.spec.binding_base + self.next_binding;
        self.next_binding += 1;
        self.bindings.push(Binding { binding: n, kind });
        n
    }

    fn decorate_binding(&mut self, var: Word, binding: u32) {
        self.b.decorate(var, Decoration::DescriptorSet, [Operand::LiteralBit32(0)]);
        self.b.decorate(var, Decoration::Binding, [Operand::LiteralBit32(binding)]);
    }

    /// Uniform buffer `struct { uvec4 data[n]; }` variable holding `dwords`
    /// (`Spec::sgpr_ubo`; std140: 16-byte array stride). Its array and
    /// struct types are its own (layout decorations on shared types would
    /// reach other uses).
    fn sgpr_ubo_var(&mut self, dwords: u32) -> Word {
        let n = self.u(dwords.div_ceil(4).max(1));
        let arr = self.b.id();
        self.b.type_array_id(Some(arr), self.t.v4u, n);
        self.b.decorate(arr, Decoration::ArrayStride, [Operand::LiteralBit32(16)]);
        let st = self.b.id();
        self.b.type_struct_id(Some(st), [arr]);
        self.b.decorate(st, Decoration::Block, []);
        self.b.member_decorate(st, 0, Decoration::Offset, [Operand::LiteralBit32(0)]);
        let pt = self.b.type_pointer(None, StorageClass::Uniform, st);
        self.global_var(pt, StorageClass::Uniform)
    }

    /// Storage buffer `struct { uint data[]; }` variable.
    fn sb_var(&mut self) -> Word {
        // rspirv deduplicates identical types: create and decorate the block type once.
        let st = match self.sb_struct {
            Some(s) => s,
            None => {
                let rta = self.b.type_runtime_array(self.t.u32);
                self.b.decorate(rta, Decoration::ArrayStride, [Operand::LiteralBit32(4)]);
                let st = self.b.type_struct([rta]);
                self.b.decorate(st, Decoration::Block, []);
                self.b.member_decorate(st, 0, Decoration::Offset, [Operand::LiteralBit32(0)]);
                self.sb_struct = Some(st);
                st
            }
        };
        let pt = self.b.type_pointer(None, StorageClass::StorageBuffer, st);
        self.global_var(pt, StorageClass::StorageBuffer)
    }

    /// `struct { uvecN data[]; }` view (N = 2 or 4) of the binding of raw
    /// buffer variable `var`, for aligned 8/16-byte accesses.
    pub fn vec_view(&mut self, var: Word, n: u32) -> Option<Word> {
        if let Some(&v) = self.buffer_views.get(&(var, n)) {
            return Some(v);
        }
        let binding = *self.buffer_binding.get(&var)?;
        let st = match self.view_structs.get(&n) {
            Some(&s) => s,
            None => {
                let vt = if n == 2 { self.t.v2u } else { self.t.v4u };
                let rta = self.b.type_runtime_array(vt);
                self.b.decorate(rta, Decoration::ArrayStride, [Operand::LiteralBit32(4 * n)]);
                let st = self.b.type_struct([rta]);
                self.b.decorate(st, Decoration::Block, []);
                self.b.member_decorate(st, 0, Decoration::Offset, [Operand::LiteralBit32(0)]);
                self.view_structs.insert(n, st);
                st
            }
        };
        let pt = self.b.type_pointer(None, StorageClass::StorageBuffer, st);
        let v = self.global_var(pt, StorageClass::StorageBuffer);
        self.decorate_binding(v, binding);
        self.b.decorate(v, Decoration::Aliased, []);
        self.buffer_views.insert((var, n), v);
        Some(v)
    }

    /// This invocation's index in its subgroup (for cross-lane operations).
    pub fn subgroup_lane(&mut self) -> R<Word> {
        self.b.capability(Capability::GroupNonUniform);
        let var = match self.sg_lane_var {
            Some(v) => v,
            None => {
                let v = self.input(self.t.u32, Some(BuiltIn::SubgroupLocalInvocationId), None);
                if self.spec.stage == Stage::Fragment {
                    self.b.decorate(v, Decoration::Flat, []);
                }
                self.sg_lane_var = Some(v);
                v
            }
        };
        e(self.b.load(self.t.u32, None, var, None, []))
    }

    /// Hardware lane index of this invocation within its wave.
    pub fn lane(&mut self) -> R<Word> {
        match self.lane_id {
            Some(l) => Ok(l),
            None => self.subgroup_lane(),
        }
    }

    /// Value of `v` in subgroup lane `src` (subgroup shuffle).
    /// Value of `v` in subgroup lane `src` and whether that invocation is
    /// executing here. Invocations that diverged away read as 0 (shuffles
    /// from them are undefined), like hardware lanes outside EXEC under
    /// whole-wave mode.
    pub fn lane_fetch(&mut self, v: Word, src: Word) -> R<(Word, Word)> {
        self.b.capability(Capability::GroupNonUniformShuffle);
        self.b.capability(Capability::GroupNonUniformBallot);
        let sg = self.u(spirv::Scope::Subgroup as u32);
        let t = self.bool_c(true);
        let ballot = e(self.b.group_non_uniform_ballot(self.t.v4u, None, sg, t))?;
        let live = e(self.b.group_non_uniform_ballot_bit_extract(self.t.bool_, None, sg, ballot, src))?;
        let x = e(self.b.group_non_uniform_shuffle(self.t.u32, None, sg, v, src))?;
        let z = self.u(0);
        let x = e(self.b.select(self.t.u32, None, live, x, z))?;
        Ok((x, live))
    }

    /// Lane fetch that also requires the source lane's EXEC bit.
    pub fn lane_fetch_active(&mut self, v: Word, src: Word) -> R<(Word, Word)> {
        let u = self.t.u32;
        let (x, live) = self.lane_fetch(v, src)?;
        let act = self.active()?;
        let one = self.u(1);
        let z = self.u(0);
        let exb = e(self.b.select(u, None, act, one, z))?;
        let (sex, _) = self.lane_fetch(exb, src)?;
        let act = e(self.b.i_not_equal(self.t.bool_, None, sex, z))?;
        let ok_ = e(self.b.logical_and(self.t.bool_, None, live, act))?;
        let x = e(self.b.select(u, None, ok_, x, z))?;
        Ok((x, ok_))
    }

    /// Value of `v` in hardware lane `lane` of this invocation's wave. Wave32
    /// waves are subgroups. Wave64 compute code in uniform blocks (see
    /// [`crate::uniform`]) exchanges values through workgroup memory between
    /// barriers; elsewhere wave64 reads stay within the 32-lane subgroup
    /// (reported).
    pub fn wave_read(&mut self, ins: &Inst, v: Word, lane: Word) -> R<Word> {
        let u = self.t.u32;
        let exchange = !self.sg_waves && self.spec.wave64 && self.spec.stage == Stage::Compute && self.uniformity.is_uniform(ins.pc) && self.local_index.is_some();
        if exchange {
            self.lds_state = 0;
        }
        if !exchange {
            if self.spec.wave64 && !self.sg_waves {
                self.warn(format!("{:#x}: wave64 lane read within the 32-lane subgroup", ins.pc));
            }
            // (Subgroup waves: lanes past the subgroup are absent, read 0.)
            let k = self.u(if self.sg_waves { 63 } else { 31 });
            let l = e(self.b.bitwise_and(u, None, lane, k))?;
            return Ok(self.lane_fetch(v, l)?.0);
        }
        let li = self.local_index.unwrap();
        let xchg = match self.xchg {
            Some(x) => x,
            None => {
                let n = self.spec.cs_local_size.iter().map(|&x| x.max(1)).product::<u32>().max(64);
                let len = self.u(n);
                let arr = self.b.type_array(u, len);
                let pt = self.b.type_pointer(None, StorageClass::Workgroup, arr);
                let x = self.global_var(pt, StorageClass::Workgroup);
                self.xchg = Some(x);
                x
            }
        };
        let p_wg = self.b.type_pointer(None, StorageClass::Workgroup, u);
        let wg = self.u(spirv::Scope::Workgroup as u32);
        let sem = self.u((spirv::MemorySemantics::ACQUIRE_RELEASE | spirv::MemorySemantics::WORKGROUP_MEMORY).bits());
        ok(self.b.control_barrier(wg, wg, sem))?;
        let mine = e(self.b.access_chain(p_wg, None, xchg, [li]))?;
        ok(self.b.store(mine, v, None, []))?;
        ok(self.b.control_barrier(wg, wg, sem))?;
        let m = self.u(!63);
        let base = e(self.b.bitwise_and(u, None, li, m))?;
        let k63 = self.u(63);
        let l = e(self.b.bitwise_and(u, None, lane, k63))?;
        let idx = e(self.b.i_add(u, None, base, l))?;
        let src = e(self.b.access_chain(p_wg, None, xchg, [idx]))?;
        e(self.b.load(u, None, src, None, []))
    }

    /// v_readfirstlane: the value of `v` in the first active lane of this
    /// invocation's wave (lane 0 when EXEC is empty), the same in every lane.
    /// Exact within a subgroup; wave64 compute code in uniform blocks
    /// combines the two 32-lane halves through workgroup memory; elsewhere a
    /// wave64 read stays within the 32-lane subgroup (reported).
    pub fn wave_read_first(&mut self, ins: &Inst, v: Word) -> R<Word> {
        let u = self.t.u32;
        let bt = self.t.bool_;
        self.b.capability(Capability::GroupNonUniformBallot);
        let sg = self.u(spirv::Scope::Subgroup as u32);
        let one = self.u(1);
        let z = self.u(0);
        let act = self.active()?;
        let ballot = e(self.b.group_non_uniform_ballot(self.t.v4u, None, sg, act))?;
        if self.sg_waves {
            // Subgroup waves: the subgroup's first active invocation.
            self.b.capability(Capability::GroupNonUniformVote);
            let any = e(self.b.group_non_uniform_any(bt, None, sg, act))?;
            let lsb = e(self.b.group_non_uniform_ballot_find_lsb(u, None, sg, ballot))?;
            let first = e(self.b.select(u, None, any, lsb, z))?;
            return Ok(self.lane_fetch(v, first)?.0);
        }
        let bits = e(self.b.composite_extract(u, None, ballot, [0]))?;
        let any = e(self.b.i_not_equal(bt, None, bits, z))?;
        let lsb = self.glsl1(self.t.i32, spirv::GlslStd450Op::FindILsb, bits)?;
        let lsb = self.to_u(lsb)?;
        let first = e(self.b.select(u, None, any, lsb, z))?;
        let (val, _) = self.lane_fetch(v, first)?;
        let threads: u32 = self.spec.cs_local_size.iter().map(|&x| x.max(1)).product();
        let compute = self.spec.stage == Stage::Compute;
        if !self.spec.wave64 || (compute && threads <= 32) {
            return Ok(val);
        }
        if !(compute && self.uniformity.is_uniform(ins.pc) && self.local_index.is_some()) {
            self.warn(format!("{:#x}: wave64 first-lane read within the 32-lane subgroup", ins.pc));
            return Ok(val);
        }
        // Each lane publishes its half's candidate and whether that half has
        // an active lane; the wave takes the lower half's unless it has none.
        self.lds_state = 0;
        let li = self.local_index.unwrap();
        let n = threads.next_multiple_of(64);
        let xf = match self.xchg_first {
            Some(x) => x,
            None => {
                let len = self.u(2 * n);
                let arr = self.b.type_array(u, len);
                let pt = self.b.type_pointer(None, StorageClass::Workgroup, arr);
                let x = self.global_var(pt, StorageClass::Workgroup);
                self.xchg_first = Some(x);
                x
            }
        };
        let p_wg = self.b.type_pointer(None, StorageClass::Workgroup, u);
        let wg = self.u(spirv::Scope::Workgroup as u32);
        let sem = self.u((spirv::MemorySemantics::ACQUIRE_RELEASE | spirv::MemorySemantics::WORKGROUP_MEMORY).bits());
        let kn = self.u(n);
        ok(self.b.control_barrier(wg, wg, sem))?;
        let p = e(self.b.access_chain(p_wg, None, xf, [li]))?;
        ok(self.b.store(p, val, None, []))?;
        let anyu = e(self.b.select(u, None, any, one, z))?;
        let i = e(self.b.i_add(u, None, kn, li))?;
        let p = e(self.b.access_chain(p_wg, None, xf, [i]))?;
        ok(self.b.store(p, anyu, None, []))?;
        ok(self.b.control_barrier(wg, wg, sem))?;
        let m = self.u(!63);
        let base = e(self.b.bitwise_and(u, None, li, m))?;
        let k32 = self.u(32);
        let upper = e(self.b.i_add(u, None, base, k32))?;
        let load = |tr: &mut Self, idx: Word| -> R<Word> {
            let p = e(tr.b.access_chain(p_wg, None, xf, [idx]))?;
            e(tr.b.load(u, None, p, None, []))
        };
        let lower_val = load(self, base)?;
        let upper_val = load(self, upper)?;
        let i = e(self.b.i_add(u, None, kn, base))?;
        let lower_any = load(self, i)?;
        let i = e(self.b.i_add(u, None, kn, upper))?;
        let upper_any = load(self, i)?;
        // Lanes past the end of a partial last wave do not exist.
        let kt = self.u(threads);
        let exists = e(self.b.u_less_than(bt, None, upper, kt))?;
        let lower_none = e(self.b.i_equal(bt, None, lower_any, z))?;
        let upper_some = e(self.b.i_not_equal(bt, None, upper_any, z))?;
        let use_upper = e(self.b.logical_and(bt, None, lower_none, upper_some))?;
        let use_upper = e(self.b.logical_and(bt, None, use_upper, exists))?;
        e(self.b.select(u, None, use_upper, upper_val, lower_val))
    }

    fn input(&mut self, ty: Word, builtin: Option<BuiltIn>, location: Option<u32>) -> Word {
        let pt = self.b.type_pointer(None, StorageClass::Input, ty);
        let v = self.global_var(pt, StorageClass::Input);
        if let Some(bi) = builtin {
            self.b.decorate(v, Decoration::BuiltIn, [Operand::BuiltIn(bi)]);
        }
        if let Some(l) = location {
            self.b.decorate(v, Decoration::Location, [Operand::LiteralBit32(l)]);
            if l < 32 {
                self.in_locs |= 1 << l;
            }
        }
        self.interface.push(v);
        v
    }

    fn output(&mut self, ty: Word, builtin: Option<BuiltIn>, location: Option<u32>) -> Word {
        let pt = self.b.type_pointer(None, StorageClass::Output, ty);
        let v = self.global_var(pt, StorageClass::Output);
        if let Some(bi) = builtin {
            self.b.decorate(v, Decoration::BuiltIn, [Operand::BuiltIn(bi)]);
        }
        if let Some(l) = location {
            self.b.decorate(v, Decoration::Location, [Operand::LiteralBit32(l)]);
        }
        self.interface.push(v);
        v
    }

    /// Data VGPR of a store; a compute probe replaces it (see `Spec::probe`).
    pub fn store_vreg(&mut self, idx: u32) -> R<Word> {
        match (self.spec.stage, self.probe_var) {
            (Stage::Compute, Some(pv)) => e(self.b.load(self.t.u32, None, pv, None, [])),
            _ => self.vreg(idx),
        }
    }

    /// Workgroup barrier ordering workgroup memory (uniform blocks only).
    pub fn workgroup_barrier(&mut self) -> R<()> {
        let wg = self.u(spirv::Scope::Workgroup as u32);
        let sem = self.u((spirv::MemorySemantics::ACQUIRE_RELEASE | spirv::MemorySemantics::WORKGROUP_MEMORY).bits());
        ok(self.b.control_barrier(wg, wg, sem))?;
        self.lds_state = 0;
        Ok(())
    }

    /// Orders an LDS access after other lanes' earlier accesses, as lockstep
    /// hardware lanes are: a barrier before a read that follows writes, or a
    /// write that follows any access, since the last barrier.
    pub fn lds_order(&mut self, pc: u32, write: bool) -> R<()> {
        if self.spec.stage != Stage::Compute {
            return Ok(());
        }
        if !self.uniformity.is_uniform(pc) {
            self.warn(format!("{pc:#x}: LDS access in divergent code is not ordered across lanes"));
            return Ok(());
        }
        let need = if write { self.lds_state != 0 } else { self.lds_state & 2 != 0 };
        if need {
            self.workgroup_barrier()?;
        }
        self.lds_state |= if write { 2 } else { 1 };
        Ok(())
    }

    /// Ballot of a per-lane condition over the workgroup (lockstep mode: the
    /// workgroup is one wave): (lo, hi) lane masks. Each subgroup ORs its
    /// lanes' bits of either half and one of its invocations adds them to a
    /// slot. Three rotating slots of workgroup memory need one barrier per
    /// ballot: a slot is cleared (by lane 0) after the barrier of the ballot
    /// following its use, when every lane has read it, and reused after the
    /// next barrier.
    pub fn wave_ballot(&mut self, c: Word) -> R<(Word, Word)> {
        let (Some(vote), Some(ctr), Some(li)) = (self.vote, self.vote_ctr, self.local_index) else {
            return Err("wave ballot outside lockstep mode".into());
        };
        let u = self.t.u32;
        let k = e(self.b.load(u, None, ctr, None, []))?;
        let one = self.u(1);
        let two = self.u(2);
        let three = self.u(3);
        let kn = e(self.b.i_add(u, None, k, one))?;
        let kn = e(self.b.u_mod(u, None, kn, three))?;
        ok(self.b.store(ctr, kn, None, []))?;
        let base = e(self.b.i_mul(u, None, k, two))?;
        let k31 = self.u(31);
        let pos = e(self.b.bitwise_and(u, None, li, k31))?;
        let bit = e(self.b.shift_left_logical(u, None, one, pos))?;
        let z = self.u(0);
        let mine = e(self.b.select(u, None, c, bit, z))?;
        let k32 = self.u(32);
        let lower = e(self.b.u_less_than(self.t.bool_, None, li, k32))?;
        let lo_part = e(self.b.select(u, None, lower, mine, z))?;
        let hi_part = e(self.b.select(u, None, lower, z, mine))?;
        self.b.capability(Capability::GroupNonUniform);
        self.b.capability(Capability::GroupNonUniformArithmetic);
        let sg = self.u(spirv::Scope::Subgroup as u32);
        let lo_sum = e(self.b.group_non_uniform_bitwise_or(u, None, sg, spirv::GroupOperation::Reduce, lo_part, None))?;
        let hi_sum = e(self.b.group_non_uniform_bitwise_or(u, None, sg, spirv::GroupOperation::Reduce, hi_part, None))?;
        let elected = e(self.b.group_non_uniform_elect(self.t.bool_, None, sg))?;
        let p_wg = self.b.type_pointer(None, StorageClass::Workgroup, u);
        self.guarded(elected, &mut |tr| {
            let scope = tr.u(spirv::Scope::Workgroup as u32);
            let sem = tr.u(0);
            let p = e(tr.b.access_chain(p_wg, None, vote, [base]))?;
            e(tr.b.atomic_or(u, None, p, scope, sem, lo_sum))?;
            let base1 = e(tr.b.i_add(u, None, base, one))?;
            let p = e(tr.b.access_chain(p_wg, None, vote, [base1]))?;
            e(tr.b.atomic_or(u, None, p, scope, sem, hi_sum))?;
            Ok(())
        })?;
        self.workgroup_barrier()?;
        let p = e(self.b.access_chain(p_wg, None, vote, [base]))?;
        let lo = e(self.b.load(u, None, p, None, []))?;
        let base1 = e(self.b.i_add(u, None, base, one))?;
        let p = e(self.b.access_chain(p_wg, None, vote, [base1]))?;
        let hi = e(self.b.load(u, None, p, None, []))?;
        let old = e(self.b.i_add(u, None, k, two))?;
        let old = e(self.b.u_mod(u, None, old, three))?;
        let old = e(self.b.i_mul(u, None, old, two))?;
        let first = e(self.b.i_equal(self.t.bool_, None, li, z))?;
        self.guarded(first, &mut |tr| {
            let p = e(tr.b.access_chain(p_wg, None, vote, [old]))?;
            ok(tr.b.store(p, z, None, []))?;
            let old1 = e(tr.b.i_add(u, None, old, one))?;
            let p = e(tr.b.access_chain(p_wg, None, vote, [old1]))?;
            ok(tr.b.store(p, z, None, []))
        })?;
        Ok((lo, hi))
    }

    /// Whether any invocation of the workgroup has `c` (lockstep mode; the
    /// slot protocol of `wave_ballot` with one dword).
    pub fn wave_any(&mut self, c: Word) -> R<Word> {
        let (Some(vote), Some(ctr), Some(li)) = (self.vote, self.vote_ctr, self.local_index) else {
            return Err("vote outside lockstep mode".into());
        };
        let u = self.t.u32;
        let k = e(self.b.load(u, None, ctr, None, []))?;
        let one = self.u(1);
        let two = self.u(2);
        let three = self.u(3);
        let kn = e(self.b.i_add(u, None, k, one))?;
        let kn = e(self.b.u_mod(u, None, kn, three))?;
        ok(self.b.store(ctr, kn, None, []))?;
        let idx = e(self.b.i_mul(u, None, k, two))?;
        let p_wg = self.b.type_pointer(None, StorageClass::Workgroup, u);
        self.guarded(c, &mut |tr| {
            let p = e(tr.b.access_chain(p_wg, None, vote, [idx]))?;
            let scope = tr.u(spirv::Scope::Workgroup as u32);
            let sem = tr.u(0);
            e(tr.b.atomic_or(u, None, p, scope, sem, one))?;
            Ok(())
        })?;
        self.workgroup_barrier()?;
        let p = e(self.b.access_chain(p_wg, None, vote, [idx]))?;
        let x = e(self.b.load(u, None, p, None, []))?;
        let old = e(self.b.i_add(u, None, k, two))?;
        let old = e(self.b.u_mod(u, None, old, three))?;
        let old = e(self.b.i_mul(u, None, old, two))?;
        let z = self.u(0);
        let first = e(self.b.i_equal(self.t.bool_, None, li, z))?;
        self.guarded(first, &mut |tr| {
            let p = e(tr.b.access_chain(p_wg, None, vote, [old]))?;
            ok(tr.b.store(p, z, None, []))
        })?;
        e(self.b.i_not_equal(self.t.bool_, None, x, z))
    }

    /// Whether `c` holds in every active invocation of the subgroup.
    pub fn subgroup_all(&mut self, c: Word) -> R<Word> {
        self.b.capability(Capability::GroupNonUniformVote);
        let sg = self.u(spirv::Scope::Subgroup as u32);
        e(self.b.group_non_uniform_all(self.t.bool_, None, sg, c))
    }

    /// Emits `f` in a block executed only when `cond` holds.
    pub fn guarded(&mut self, cond: Word, f: &mut dyn FnMut(&mut T) -> R<()>) -> R<()> {
        let merge = self.b.id();
        let body = self.b.id();
        ok(self.b.selection_merge(merge, spirv::SelectionControl::NONE))?;
        ok(self.b.branch_conditional(cond, body, merge, []))?;
        e(self.b.begin_block(Some(body)))?;
        f(self)?;
        ok(self.b.branch(merge))?;
        e(self.b.begin_block(Some(merge)))?;
        Ok(())
    }
}

/// Lane reads (v_readlane / v_readfirstlane) of a value an atomic has just
/// returned: the wave-aggregated atomic (a ballot counts the lanes, one lane
/// adds the count, every lane takes the old value from that lane and adds
/// its v_mbcnt prefix). Each invocation is a wave of one here (its ballot
/// holds only its own bit, its prefix is 0), so it ran the atomic for itself
/// and the result to take is its own; another lane's would make all lanes
/// share one slot.
fn atomic_result_reads(code: &[Inst]) -> BTreeSet<u32> {
    let mut out = BTreeSet::new();
    for (i, ins) in code.iter().enumerate() {
        if !matches!(ins.base_name(), Some("v_readlane_b32" | "v_readfirstlane_b32")) {
            continue;
        }
        let s = ins.src(0);
        if s < crate::OP_VGPR0 {
            continue;
        }
        let v = s - crate::OP_VGPR0;
        for p in code[..i].iter().rev().take(32) {
            match vgpr_def(p, v) {
                Some(atomic) => {
                    if atomic {
                        out.insert(ins.pc);
                    }
                    break;
                }
                None => {}
            }
        }
    }
    out
}

/// Lane-0 elections of wave-aggregated atomics that count with popcounts of
/// lane masks. Pattern (the deferred lighting's tile classification appends
/// each 8x8 block of tiles this way):
///
///   v_cmp_lg_u32 vcc, 0, vL            ; vL: the lane's index in the group
///   s_not_b64 vcc, vcc                 ; (or v_cmp_eq_u32 directly)
///   s_mov_b64 exec, vcc                ; lane 0 only
///   v_bcnt_u32_b32 vN, sMaskHi, 0      ; lanes taking part
///   v_bcnt_u32_b32 vN, sMaskLo, vN
///   buffer_atomic_add vN, ... glc      ; lane 0 reserves their slots
///   v_readlane_b32 sBase, vN, 0        ; every lane takes the base ...
///   ... v_mbcnt (mask) + sBase         ; ... plus its prefix
///
/// The workgroup is a single wave, so vL is its lane index. Here every
/// invocation is a wave of one: its masks hold only its own bit and its
/// prefix is 0, so it must reserve its own slot (with its own count) and read
/// its own result (`atomic_result_reads`). The compare therefore takes lane
/// 0's result in every lane. Lane-0 work that adds a count gathered across
/// lanes (DPP sums read with v_readlane) is left alone.
/// Returns compare pc -> index of the source operand holding vL.
fn lane0_elections(code: &[Inst], spec: &Spec) -> HashMap<u32, usize> {
    use crate::{OP_EXEC_LO, OP_VCC_LO, OP_VGPR0};
    let mut out = HashMap::default();
    let wave = if spec.wave64 { 64 } else { 32 };
    let [sx, sy, sz] = spec.cs_local_size.map(|x| x.max(1));
    if spec.stage != Stage::Compute || sx * sy * sz != wave {
        return out;
    }
    // Writes of VGPR v anywhere in the program (pcs).
    let writes = |v: u16| -> Vec<usize> { code.iter().enumerate().filter(|(_, p)| vgpr_def(p, v).is_some()).map(|(i, _)| i).collect() };
    // Whether VGPR v holds the lane index (the linear local invocation index
    // of a single-wave group) at instruction `at`.
    let lane_index = |v: u16, at: usize| -> bool {
        let w = writes(v);
        if w.iter().any(|&i| i >= at) && w.len() != 1 {
            return false;
        }
        match w.as_slice() {
            [] => v == 0 && sy == 1 && sz == 1,
            [i] if *i < at => {
                let p = &code[*i];
                // v_lshl_add_u32 vL, v1, log2(sx), v0 with v0/v1 still the
                // local invocation id.
                let k = p.src(1);
                p.base_name() == Some("v_lshl_add_u32")
                    && p.src(0) == OP_VGPR0 + 1
                    && p.src(2) == OP_VGPR0
                    && (129..=192).contains(&k)
                    && 1u32 << (k - 128) == sx
                    && sz == 1
                    && writes(0).iter().chain(writes(1).iter()).all(|&j| j >= *i)
            }
            _ => false,
        }
    };
    // Compare operands and destination: (cmp, src0, src1, sdst).
    let cmp_parts = |p: &Inst| -> Option<(&'static str, u16, u16, u16)> {
        let name = p.base_name()?;
        let rest = name.strip_prefix("v_cmp_")?;
        let cmp = match rest {
            "eq_u32" | "eq_i32" => "eq",
            "lg_u32" | "lg_i32" | "ne_u32" | "ne_i32" => "ne",
            _ => return None,
        };
        match (p.enc, p.ext) {
            (crate::Enc::Vopc, crate::Ext::Sdwa(d)) => {
                let s0 = if d >> 23 & 1 != 0 { (d & 0xFF) as u16 } else { OP_VGPR0 + (d & 0xFF) as u16 };
                let s1 = if d >> 31 & 1 != 0 { (p.w0 >> 9 & 0xFF) as u16 } else { OP_VGPR0 + (p.w0 >> 9 & 0xFF) as u16 };
                Some((cmp, s0, s1, p.sdwa_sdst().unwrap_or(OP_VCC_LO)))
            }
            (crate::Enc::Vopc, _) => Some((cmp, p.src(0), p.src(1), OP_VCC_LO)),
            (crate::Enc::Vop3, _) => Some((cmp, p.src(0), p.src(1), p.vdst())),
            _ => None,
        }
    };
    for (ia, a) in code.iter().enumerate() {
        let name = a.base_name().unwrap_or("");
        if !(name.starts_with("buffer_atomic_add") || name.starts_with("global_atomic_add")) || !a.glc() {
            continue;
        }
        let d = a.vdst();
        // The data: a popcount of SGPR lane masks.
        let Some(id) = (0..ia).rev().take(24).find(|&i| vgpr_def(&code[i], d).is_some()) else { continue };
        let def = &code[id];
        if def.base_name() != Some("v_bcnt_u32_b32") || def.src(0) >= 106 {
            continue;
        }
        // The result is read with v_readlane / v_readfirstlane.
        let read = code[ia + 1..].iter().take(8).any(|p| matches!(p.base_name(), Some("v_readlane_b32" | "v_readfirstlane_b32")) && p.src(0) == OP_VGPR0 + d);
        if !read {
            continue;
        }
        // EXEC at the atomic: s_mov_b64 exec, X.
        let Some(ie) = (0..ia).rev().take(24).find(|&i| writes_sgpr(&code[i], OP_EXEC_LO)) else { continue };
        let e = &code[ie];
        if e.base_name() != Some("s_mov_b64") || e.sdst() != OP_EXEC_LO {
            continue;
        }
        // X from a compare, possibly inverted by s_not_b64.
        let mut want = e.ssrc0();
        let mut from = ie;
        let mut negated = false;
        let mut found = None;
        for _ in 0..3 {
            let Some(ix) = (0..from).rev().take(24).find(|&i| writes_sgpr(&code[i], want)) else { break };
            let x = &code[ix];
            if x.base_name() == Some("s_not_b64") && x.sdst() == want {
                negated = !negated;
                want = x.ssrc0();
                from = ix;
                continue;
            }
            if let Some((cmp, s0, s1, sd)) = cmp_parts(x) {
                if sd == want {
                    found = Some((ix, cmp, s0, s1));
                }
            }
            break;
        }
        let Some((ix, cmp, s0, s1)) = found else { continue };
        // Lane 0 must end up active: (0 == vL) or not(0 != vL).
        if (cmp == "eq") == negated {
            continue;
        }
        let zero = 128u16;
        let lane_op = if s0 == zero && s1 >= OP_VGPR0 && lane_index(s1 - OP_VGPR0, ix) {
            1
        } else if s1 == zero && s0 >= OP_VGPR0 && lane_index(s0 - OP_VGPR0, ix) {
            0
        } else {
            continue;
        };
        out.insert(code[ix].pc, lane_op);
    }
    out
}

/// Whether `p` may write SGPR `s` (pairs count both halves).
fn writes_sgpr(p: &Inst, s: u16) -> bool {
    use crate::{OP_EXEC_LO, OP_VCC_LO};
    let name = p.base_name().unwrap_or("");
    let pair = |d: u16| s == d || s == d + 1;
    match p.enc {
        crate::Enc::Sop1 | crate::Enc::Sop2 | crate::Enc::Sopk => {
            let d = p.sdst();
            d == s || (name.ends_with("_b64") || name.ends_with("_u64") || name.ends_with("_i64")) && d + 1 == s || name.contains("saveexec") && pair(OP_EXEC_LO)
        }
        crate::Enc::Smem => p.sdst() <= s && s < p.sdst() + 16,
        crate::Enc::Vopc => {
            if name.starts_with("v_cmpx") {
                pair(OP_EXEC_LO)
            } else {
                pair(p.sdwa_sdst().unwrap_or(OP_VCC_LO))
            }
        }
        crate::Enc::Vop3 if name.starts_with("v_cmpx") => pair(OP_EXEC_LO),
        crate::Enc::Vop3 if name.starts_with("v_cmp") => pair(p.vdst()),
        crate::Enc::Vop3 if p.is_vop3b() => pair(p.vop3_sdst()),
        crate::Enc::Vop3 if matches!(name, "v_readlane_b32" | "v_readfirstlane_b32") => p.vdst() == s,
        crate::Enc::Vop1 if name == "v_readfirstlane_b32" => p.vdst() == s,
        crate::Enc::Vop2 => (name.contains("_co_") || name.contains("_ci_")) && pair(OP_VCC_LO),
        _ => false,
    }
}

/// Wave-aggregated appends done per subgroup (`Spec::sg_appends`). On the
/// hardware a ballot of the lanes taking part is counted (v_bcnt), one lane
/// adds the count to a counter with an atomic returning the old value, every
/// lane reads that lane's result (v_readlane / v_readfirstlane) and adds its
/// rank among the lanes taking part (v_mbcnt of the ballot):
///
///   v_cmp_lg_u32 s[0:1], 0, v15        ; the lanes taking part
///   s_and_b64 s16, exec, 1             ; lane 0 (or a compare with the lane index)
///   s_mov_b64 exec, s16
///   v_bcnt_u32_b32 v1, s1, 0           ; the count ...
///   v_bcnt_u32_b32 v1, s0, v1
///   v_mul_u32_u24 v1, 3, v1            ; (scaled)
///   buffer_atomic_add v1, ... glc      ; ... added by lane 0
///   v_readlane_b32 vcc_lo, v1, 0       ; the old value for every lane
///   s_mov_b64 exec, -1
///   v_cmpx_lg_u32 0, v15
///   v_mbcnt_hi_u32_b32 v0, s1, 0       ; plus the lane's rank
///   v_mbcnt_lo_u32_b32 v0, s0, v0
///
/// (the title's triangle culls into index buffers, cs 0x1008994d00 and its
/// siblings; the deferred lighting's tile classification cs 0x10088ab000).
/// Exact, such ballots need wave-mask mode: lockstep, with a workgroup
/// barrier for each. Instead each
/// subgroup does the same for its own lanes: the ballot is the mask of the
/// subgroup's lanes in the hardware layout (other subgroups' lanes 0), so
/// v_bcnt counts the subgroup's lanes and v_mbcnt ranks a lane among them;
/// the EXEC write before the atomic leaves the subgroup's first active
/// invocation alone, which adds the subgroup's count; the result's lane
/// reads take that invocation's value. Every lane gets a slot of its own and
/// the counter the total; a subgroup's slots are contiguous and the
/// subgroups' ranges come in the order of their atomics (a wave's on the
/// hardware, in the order of the waves'). The election itself (s16 above)
/// stays exact for its other uses (cs 0x1008994d00's lane 0 of group 0
/// writes the draw's arguments with it).
///
/// Found when every append of the program has that form and nothing else
/// reads the ballots (`wavemask::ballots_for`); each append's ballots, its
/// election and its EXEC write in one uniform block with no other EXEC
/// write between them; the result's reads in uniform blocks, read only lane
/// by lane (`wavemask::read_by_lanes_only`: the subgroups' values differ).
/// The translation must be structured (no dispatcher), so that a subgroup
/// runs those blocks together.
#[derive(Default)]
pub struct Appends {
    /// The ballots' compares.
    pub ballots: crate::HashSet<u32>,
    /// v_mbcnt of the ballots (the mask taken as it is).
    pub ranks: crate::HashSet<u32>,
    /// `s_and_b64 X, exec, 1` elections (made exact: lane 0's bit).
    pub lane0: crate::HashSet<u32>,
    /// Compares with the lane index that elect lane 0 (left exact, not
    /// [`lane0_elections`]).
    pub lane0_compares: crate::HashSet<u32>,
    /// The EXEC writes before the atomics.
    pub leaders: crate::HashSet<u32>,
    /// Lane reads of the atomics' results.
    pub results: crate::HashSet<u32>,
}

/// An atomic of a wave-aggregated append: `buffer_atomic_add` /
/// `global_atomic_add` returning the old value whose data is a popcount of
/// SGPR masks (v_bcnt_u32_b32, accumulated, perhaps then scaled in place by
/// a constant): (the v_bcnt pcs, their SGPRs).
fn append_count(code: &[Inst], ia: usize) -> Option<(Vec<u32>, Vec<u16>)> {
    let a = &code[ia];
    if !matches!(a.base_name(), Some("buffer_atomic_add" | "global_atomic_add")) || !a.glc() {
        return None;
    }
    let d = a.vdst();
    let def_of = |before: usize| (0..before).rev().take(24).find(|&i| vgpr_def(&code[i], d).is_some());
    let mut at = def_of(ia)?;
    let p = &code[at];
    let constant = |op: u16| (128..=208).contains(&op) || op == crate::OP_LITERAL;
    if matches!(p.base_name(), Some("v_mul_u32_u24" | "v_mul_lo_u32")) && p.vdst() == d {
        let (s0, s1) = (p.src(0), p.src(1));
        if !(constant(s0) && s1 == crate::OP_VGPR0 + d || constant(s1) && s0 == crate::OP_VGPR0 + d) {
            return None;
        }
        at = def_of(at)?;
    }
    let (mut pcs, mut masks) = (Vec::new(), Vec::new());
    loop {
        let p = &code[at];
        if p.base_name() != Some("v_bcnt_u32_b32") || p.vdst() != d || vgpr_def(p, d) != Some(false) {
            return None;
        }
        let s = p.src(0);
        if s > crate::OP_VCC_LO + 1 {
            return None;
        }
        pcs.push(p.pc);
        masks.push(s);
        match p.src(1) {
            128 => break,
            x if x == crate::OP_VGPR0 + d => at = def_of(at)?,
            _ => return None,
        }
    }
    Some((pcs, masks))
}

/// Lane reads (v_readlane of lane 0, v_readfirstlane) of the result of the
/// atomic `code[ia]` within the next 8 instructions.
fn append_result_reads(code: &[Inst], ia: usize) -> Vec<usize> {
    let d = code[ia].vdst();
    let mut out = Vec::new();
    for (j, p) in code.iter().enumerate().skip(ia + 1).take(8) {
        match p.base_name() {
            Some("v_readlane_b32") if p.src(0) == crate::OP_VGPR0 + d && p.src(1) == 128 => out.push(j),
            Some("v_readfirstlane_b32") if p.src(0) == crate::OP_VGPR0 + d => out.push(j),
            _ => {}
        }
        if vgpr_def(p, d).is_some() {
            break;
        }
    }
    out
}

/// Whether the program has an atomic of a wave-aggregated append whose
/// result is read back (the runtime's cheap test before `Spec::sg_appends`).
pub fn has_append(code: &[Inst]) -> bool {
    (0..code.len()).any(|ia| append_count(code, ia).is_some() && !append_result_reads(code, ia).is_empty())
}

/// The program's wave-aggregated appends when all of them can be done per
/// subgroup (see [`Appends`]), or why not.
pub fn subgroup_appends(code: &[Inst], spec: &Spec) -> Result<Appends, &'static str> {
    use crate::{OP_EXEC_LO, OP_VCC_LO};
    if spec.stage != Stage::Compute || code.is_empty() {
        return Err("not a compute program");
    }
    let wave64 = spec.wave64;
    let u = crate::uniform::analyze(code);
    let starts = block_starts(code);
    // Whether code[i..j] lies in one block.
    let one_block = |i: usize, j: usize| code[i + 1..=j].iter().all(|p| !starts.contains(&p.pc));
    let writes_exec = |p: &Inst| writes_sgpr(p, OP_EXEC_LO);
    let targets: crate::HashSet<u32> = code.iter().filter_map(|p| p.branch_target()).collect();
    let le = lane0_elections(code, spec);
    let mut out = Appends::default();
    let mut reads = crate::HashSet::default();
    // Each append's EXEC write (index), for the ballots' check below.
    let mut execs = Vec::new();
    for ia in 0..code.len() {
        let Some((bcnts, mut masks)) = append_count(code, ia) else { continue };
        masks.sort();
        let pair = if wave64 { masks.len() == 2 && masks[0] % 2 == 0 && masks[1] == masks[0] + 1 } else { masks.len() == 1 };
        let results = append_result_reads(code, ia);
        if !pair {
            return Err("a count of other than the ballot's two halves");
        }
        if results.is_empty() || results.iter().any(|&j| !u.is_uniform(code[j].pc)) {
            return Err("its result not read back in a uniform block");
        }
        // EXEC at the atomic: s_mov X, from an election in its block.
        let Some(ie) = (0..ia).rev().take(24).find(|&i| writes_exec(&code[i])) else { return Err("no EXEC write before the atomic") };
        let e = &code[ie];
        let x = e.ssrc0();
        let mov = if wave64 { "s_mov_b64" } else { "s_mov_b32" };
        // (Reached from it alone: no branch lands between.)
        if e.base_name() != Some(mov) || e.sdst() != OP_EXEC_LO || x > OP_VCC_LO || !u.is_uniform(e.pc) || code[ie + 1..=ia].iter().any(|p| targets.contains(&p.pc)) {
            return Err("the atomic's EXEC not a uniform copy reaching it alone");
        }
        let Some(ix) = (0..ie).rev().take(24).find(|&i| writes_sgpr(&code[i], x)) else { return Err("no election before the atomic's EXEC") };
        let xd = &code[ix];
        let and = if wave64 { "s_and_b64" } else { "s_and_b32" };
        let (s0, s1) = (xd.ssrc0(), xd.ssrc1());
        if xd.base_name() == Some(and) && xd.sdst() == x && (s0 == OP_EXEC_LO && s1 == 129 || s1 == OP_EXEC_LO && s0 == 129) {
            if !one_block(ix, ie) || code[ix + 1..ie].iter().any(writes_exec) {
                return Err("the election apart from the atomic's EXEC");
            }
            out.lane0.insert(xd.pc);
        } else {
            // A compare with the lane index, perhaps inverted.
            let (mut want, mut from, mut found) = (x, ie, None);
            for _ in 0..3 {
                let Some(k) = (0..from).rev().take(24).find(|&i| writes_sgpr(&code[i], want)) else { break };
                let p = &code[k];
                if matches!(p.base_name(), Some("s_not_b64" | "s_not_b32")) && p.sdst() == want {
                    want = p.ssrc0();
                    from = k;
                    continue;
                }
                if le.contains_key(&p.pc) {
                    found = Some(k);
                }
                break;
            }
            let Some(k) = found else { return Err("an election of another form") };
            if !one_block(k, ie) || code[k + 1..ie].iter().any(writes_exec) {
                return Err("the election apart from the atomic's EXEC");
            }
            out.lane0_compares.insert(code[k].pc);
        }
        out.leaders.insert(e.pc);
        out.results.extend(results.iter().map(|&j| code[j].pc));
        reads.extend(bcnts);
        // The ranks: v_mbcnt of the ballot's SGPRs after the atomic.
        for p in code.iter().skip(ia + 1).take(48) {
            if matches!(p.base_name(), Some("v_mbcnt_lo_u32_b32" | "v_mbcnt_hi_u32_b32")) && masks.contains(&p.src(0)) {
                out.ranks.insert(p.pc);
                reads.insert(p.pc);
            }
        }
        execs.push(ie);
    }
    if out.leaders.is_empty() {
        return Err("no append");
    }
    let Some(ballots) = crate::wavemask::ballots_for(code, wave64, &reads) else { return Err("a ballot read otherwise, or a count of other values") };
    // Each ballot in an append's block before its EXEC write, no other EXEC
    // write between.
    for (ib, b) in code.iter().enumerate().filter(|(_, b)| ballots.contains(&b.pc)) {
        if !execs.iter().any(|&ie| ib < ie && one_block(ib, ie) && !code[ib + 1..ie].iter().any(writes_exec)) {
            return Err("a ballot apart from its atomic's EXEC");
        }
        if !u.is_uniform(b.pc) {
            return Err("a ballot in a divergent block");
        }
    }
    if !crate::wavemask::read_by_lanes_only(code, wave64, &out.results) {
        return Err("a result read by scalar code");
    }
    out.ballots = ballots;
    Ok(out)
}

/// Attributes that v_interp_mov_f32 reads as P10 or P20: the title
/// interpolates packed 16-bit attributes itself (flat-shaded inputs whose
/// parameter cache holds each vertex's value, combined with the barycentric
/// VGPRs through v_fma_mix_f32), e.g. the UVs of alpha-tested foliage.
fn per_vertex_attrs(code: &[Inst]) -> BTreeSet<u32> {
    code.iter().filter(|i| i.enc == Enc::Vintrp && i.base_name() == Some("v_interp_mov_f32") && i.intrp_vsrc() & 3 != 2).map(|i| i.intrp_attr()).collect()
}

/// Vertex parameter (location) of attribute `attr` and whether it is
/// flat-shaded; None for an attribute with a constant default.
pub(crate) fn attr_location(spec: &Spec, attr: u32) -> Option<(u32, bool)> {
    match spec.ps_inputs.get(attr as usize) {
        Some(pi) if pi.default.is_some() => None,
        Some(pi) => Some((pi.param as u32, pi.flat)),
        None => Some((attr, false)),
    }
}

/// Locations a fragment shader reads through per-vertex input arrays: those
/// of attributes read per vertex (`per_vertex_attrs`), of interpolated
/// attributes read as P0 (vertex 0's value), and locations that attributes
/// read both flat and interpolated. Each location is a single input
/// variable: Vulkan has no aliasing of a location's flat, interpolated and
/// per-vertex inputs (the title maps two attributes to one parameter, one
/// read per vertex, one flat: the G-buffer shaders of World 4-1's cliffs,
/// whose UVs became constant per triangle).
fn per_vertex_locations(code: &[Inst], spec: &Spec) -> BTreeSet<u32> {
    let mut out: BTreeSet<u32> = per_vertex_attrs(code).into_iter().filter_map(|a| attr_location(spec, a).map(|(loc, _)| loc)).collect();
    let mut kinds: HashMap<u32, (bool, bool)> = HashMap::default();
    for i in code.iter().filter(|i| i.enc == Enc::Vintrp) {
        let Some((loc, flat)) = attr_location(spec, i.intrp_attr()) else { continue };
        if !flat && i.base_name() == Some("v_interp_mov_f32") {
            out.insert(loc);
        }
        let k = kinds.entry(loc).or_default();
        if flat {
            k.0 = true;
        } else {
            k.1 = true;
        }
    }
    out.extend(kinds.into_iter().filter(|(_, (f, s))| *f && *s).map(|(l, _)| l));
    out
}

/// Where a pixel stage reading the per-vertex locations `pv` gets them
/// without fragment barycentrics (`Spec::bary_gs`): each location keeps
/// vertex 0's value, vertices 1 and 2 and the two barycentrics take free
/// locations below `VARYING_LOCATIONS` other than `BATCH_VARYING` (None
/// when too few are free: the stage keeps vertex 0's values, faceted).
fn bary_gs_io(code: &[Inst], spec: &Spec, pv: &BTreeSet<u32>) -> Option<gs::BaryIo> {
    let mut used: u32 = if spec.batched { 1 << BATCH_VARYING } else { 0 };
    for i in code.iter().filter(|i| i.enc == Enc::Vintrp) {
        if let Some((loc, _)) = attr_location(spec, i.intrp_attr()) {
            used |= 1u32.checked_shl(loc).unwrap_or(0);
        }
    }
    for &l in pv {
        used |= 1u32.checked_shl(l).unwrap_or(0);
    }
    let mut free = (0..VARYING_LOCATIONS).filter(|l| used & (1 << l) == 0);
    let mut per_vertex = Vec::new();
    for &l in pv {
        per_vertex.push((l, [free.next()?, free.next()?]));
    }
    let bary = [free.next()?, free.next()?];
    Some(gs::BaryIo { per_vertex, bary })
}

/// Whether `p` may write VGPR `v`: Some(true) for an atomic returning the
/// previous value into it, Some(false) for any other (possible) write.
fn vgpr_def(p: &Inst, v: u16) -> Option<bool> {
    let name = p.base_name().unwrap_or("");
    let d = p.vdst();
    match p.enc {
        Enc::Mubuf | Enc::Mtbuf | Enc::Mimg | Enc::Global | Enc::Flat | Enc::Scratch | Enc::Ds => {
            let atomic = name.contains("atomic") && (p.enc == Enc::Ds || p.glc()) || name.contains("_rtn");
            let returns = atomic && (p.enc != Enc::Ds || name.contains("_rtn"));
            let store = name.contains("store") || name.contains("write") || name.contains("atomic") && !returns;
            if store || !(d..d + 4).contains(&v) {
                return None;
            }
            Some(returns && d == v)
        }
        Enc::Vop1 | Enc::Vop2 | Enc::Vop3 | Enc::Vop3p | Enc::Vintrp => {
            // Compares and lane reads write SGPRs; 64-bit results two
            // registers (v_mqsad_u32_u8 four), others one (a v_mov_b32 v0
            // counted as a write of v1 hid cs 0x1008994d00's append count).
            if name.starts_with("v_cmp") || matches!(name, "v_readlane_b32" | "v_readfirstlane_b32") {
                return None;
            }
            let n = if name.starts_with("v_mqsad_u32") {
                4
            } else if name.contains("64") || name.contains("qsad") {
                2
            } else {
                1
            };
            (d <= v && v < d + n).then_some(false)
        }
        _ => None,
    }
}

/// Splits the program into basic blocks: returns block start pcs.
fn block_starts(code: &[Inst]) -> BTreeSet<u32> {
    let mut s = BTreeSet::new();
    s.insert(code[0].pc);
    for (i, ins) in code.iter().enumerate() {
        if let Some(t) = ins.branch_target() {
            s.insert(t);
            if let Some(n) = code.get(i + 1) {
                s.insert(n.pc);
            }
        }
        if ins.is_endpgm() {
            if let Some(n) = code.get(i + 1) {
                s.insert(n.pc);
            }
        }
    }
    s
}

pub fn translate(code: &[Inst], plan: &Plan, spec: &Spec) -> R<Output> {
    if code.is_empty() {
        return Err("empty program".into());
    }
    if (spec.gpu_loads != 0 || spec.gpu_res != 0) && !spec.addresses {
        return Err("GPU-resolved chains need device addresses (`Spec::addresses`)".into());
    }
    if let Some(r) = (0..plan.resources.len().min(64)).find(|&r| spec.gpu_res & (1u64 << r) != 0 && (plan.resources[r].written || plan.resources[r].kind != ResKind::Buffer)) {
        return Err(format!("GPU-resolved resource {r} is not a read-only buffer"));
    }
    let mut b = Builder::new();
    b.set_version(1, 3);
    b.capability(Capability::Shader);
    b.capability(Capability::ImageQuery);
    b.capability(Capability::StorageImageWriteWithoutFormat);
    b.capability(Capability::StorageImageReadWithoutFormat);
    b.capability(Capability::Int64);
    b.capability(Capability::Sampled1D);
    b.capability(Capability::Image1D);
    b.capability(Capability::SampledCubeArray);
    b.capability(Capability::ImageCubeArray);
    b.capability(Capability::ImageGatherExtended);
    b.extension("SPV_KHR_storage_buffer_storage_class");
    if spec.addresses {
        b.capability(Capability::PhysicalStorageBufferAddresses);
        b.extension("SPV_KHR_physical_storage_buffer");
    }
    let glsl = b.ext_inst_import("GLSL.std.450");
    let addressing = if spec.addresses { spirv::AddressingModel::PhysicalStorageBuffer64 } else { spirv::AddressingModel::Logical };
    b.memory_model(addressing, spirv::MemoryModel::GLSL450);
    let void = b.type_void();
    let bool_ = b.type_bool();
    let u32t = b.type_int(32, 0);
    let i32t = b.type_int(32, 1);
    let f32t = b.type_float(32, None);
    let u64t = b.type_int(64, 0);
    let v2f = b.type_vector(f32t, 2);
    let v3f = b.type_vector(f32t, 3);
    let v4f = b.type_vector(f32t, 4);
    let v2u = b.type_vector(u32t, 2);
    let v3u = b.type_vector(u32t, 3);
    let v4u = b.type_vector(u32t, 4);
    let v2i = b.type_vector(i32t, 2);
    let v3i = b.type_vector(i32t, 3);
    let v4i = b.type_vector(i32t, 4);
    let t = Types {
        bool_,
        u32: u32t,
        i32: i32t,
        f32: f32t,
        u64: u64t,
        v2f,
        v3f,
        v4f,
        v2u,
        v3u,
        v4u,
        v2i,
        v3i,
        v4i,
        fn_u32: b.type_pointer(None, StorageClass::Function, u32t),
        fn_bool: b.type_pointer(None, StorageClass::Function, bool_),
        sb_u32: b.type_pointer(None, StorageClass::StorageBuffer, u32t),
    };
    let sampler_ty = b.type_sampler();
    // Wave-aggregated appends per subgroup: structured translations only
    // (each subgroup runs the appends' blocks together).
    let structured = {
        let starts = block_starts(code);
        starts.len() == 1 || matches!(loop_ranges(code, &starts), Some(loops) if !spec.force_dispatcher && (loops.is_empty() || !spec.dispatcher_for_loops))
    };
    let (sg, sg_why) = match spec.sg_appends {
        false => (Appends::default(), None),
        true if !structured => (Appends::default(), Some("a dispatcher translation")),
        true => match subgroup_appends(code, spec) {
            Ok(a) => (a, None),
            Err(why) => (Appends::default(), Some(why)),
        },
    };
    let (lockstep, one_wave) = lockstep_mode(spec, code, &sg.ballots);
    // Wave-mask mode for a single wave that reads a lane mask as a value
    // (a ballot it stores or counts); one that reads none runs on votes, as
    // several waves do: each vote one workgroup barrier at a lane-dependent
    // branch, `masked_skip` sparing most, against a ballot at every compare
    // whose mask reaches EXEC or a branch (single-wave passes 2.2x faster at
    // most, the same shots: FINDINGS.md 2026-11-03 early morning).
    let wave_masks = lockstep && one_wave && wave_masks_wanted(spec.code_base) && crate::wavemask::reads_mask_values_except(code, spec.wave64, &sg.ballots);
    // (In wave-mask mode the appends are exact.)
    let sg_why = if wave_masks && !sg.leaders.is_empty() { Some("other masks read as values") } else { sg_why };
    let sg = if wave_masks { Appends::default() } else { sg };
    // Subgroup waves for the programs the runtime lists (`DS_SGWAVE`: others
    // too, by address), where the translation allows.
    let sg_waves_asked = spec.sg_waves || sg_waves_wanted(spec.code_base);
    let sg_waves_why = if !sg_waves_asked {
        None
    } else if !wave_masks {
        Some("not in wave-mask mode")
    } else {
        sg_waves_refusal(spec, code, structured)
    };
    let sg_waves = sg_waves_asked && sg_waves_why.is_none();
    let pv_locs = if (spec.barycentric || spec.bary_gs) && spec.stage == Stage::Fragment { per_vertex_locations(code, spec) } else { BTreeSet::new() };
    let bary_io = if spec.bary_gs && !spec.barycentric && !pv_locs.is_empty() { bary_gs_io(code, spec, &pv_locs) } else { None };
    // (Too few free locations for the geometry shader: vertex 0's values.)
    let pv_locs = if spec.barycentric || bary_io.is_some() { pv_locs } else { BTreeSet::new() };
    let mut tr = T {
        b,
        t,
        glsl,
        spec,
        plan,
        sgpr: 0,
        vgpr: 0,
        sregs: Vec::new(),
        vregs: Vec::new(),
        active_cache: None,
        scc: 0,
        consts: HashMap::default(),
        interface: Vec::new(),
        sgpr_data: 0,
        sgpr_ub: None,
        sgpr_base: None,
        load_offsets: load_offsets(plan),
        res_info: res_info_offset(plan),
        code_base_at: code_base_offset(plan),
        buffers: HashMap::default(),
        images: HashMap::default(),
        image_alts: HashMap::default(),
        sampler_res: BTreeSet::new(),
        samplers: HashMap::default(),
        compare_sampler_vars: HashMap::default(),
        sampler_ty,
        bindings: Vec::new(),
        next_binding: 0,
        out_pos: None,
        out_params: HashMap::default(),
        out_mrt: HashMap::default(),
        out_depth: None,
        in_attrs: HashMap::default(),
        per_vertex_locs: pv_locs,
        per_vertex_in: HashMap::default(),
        bary_io,
        per_vertex_gs: HashMap::default(),
        bary: None,
        frag_coord: None,
        lds: None,
        spill: None,
        gds: None,
        marks: None,
        lane_id: None,
        local_index: None,
        xchg: None,
        xchg_first: None,
        uniformity: crate::uniform::analyze(code),
        lockstep: false,
        fmt_table: None,
        rt_word3: None,
        vote: None,
        vote_ctr: None,
        sync_starts: Vec::new(),
        sync_steps: None,
        lds_state: 0,
        lane_guard: false,
        masked_skips: 0,
        vcache: HashMap::default(),
        bitcasts: HashMap::default(),
        vcache_block: None,
        sg_lane_var: None,
        lane_slots: HashMap::default(),
        image_tables: HashMap::default(),
        table_offsets: HashMap::default(),
        // In wave-mask mode lane reads and elections are exact.
        own_lane_reads: if wave_masks { BTreeSet::new() } else { atomic_result_reads(code).into_iter().filter(|pc| !sg.results.contains(pc)).collect() },
        elections: if wave_masks { HashMap::default() } else { lane0_elections(code, spec).into_iter().filter(|(pc, _)| !sg.lane0_compares.contains(pc)).collect() },
        sg,
        leader: None,
        known_active: crate::HashSet::default(),
        scalar_only: 0,
        unchecked_res: 0,
        cur_pc: 0,
        in_locs: 0,
        wave_masks,
        sg_waves,
        complete: if wave_masks { crate::wavemask::complete_masks(code, spec.wave64) } else { Default::default() },
        own_bit: None,
        warnings: Vec::new(),
        params: 0,
        mrt_mask: 0,
        pc_var: 0,
        probe_var: None,
        watchdog: None,
        loop_ctr: None,
        dump_buf: None,
        lod_stats_var: None,
        helper_var: None,
        wg_id: None,
        vertex_ids: None,
        sb_struct: None,
        buffer_binding: HashMap::default(),
        buffer_views: HashMap::default(),
        view_structs: HashMap::default(),
        buffer_res: HashMap::default(),
        addr_info_at: addr_info_offset(plan),
        addr_vals: HashMap::default(),
        zero_addr: None,
        batch_wg: None,
        pt_addr: None,
        rt_vsharp: None,
        consts64: HashMap::default(),
    };
    if let Some(why) = sg_why {
        tr.warn(format!("wave-aggregated appends left as they were ({why})"));
    }
    if let Some(why) = sg_waves_why {
        tr.warn(format!("subgroup waves left out ({why})"));
    }
    let mut elected: Vec<u32> = tr.elections.keys().copied().collect();
    elected.sort();
    for pc in elected {
        tr.warn(format!("{pc:#x}: lane-0 election of a wave-aggregated atomic taken by every lane (each appends itself)"));
    }
    // SGPR data buffer (binding base + 0).
    let sgpr_dwords = sgpr_data_dwords(plan);
    let window = spec.sgpr_ubo == 3 && spec.batched && spec.stage == Stage::Compute;
    let ubo = window || (spec.sgpr_ubo & 1 == 1 && !spec.batched && spec.sgpr_pc.is_none() && sgpr_dwords <= SGPR_UBO_MAX_DWORDS);
    let sd = if ubo { tr.sgpr_ubo_var(if window { SGPR_WINDOW_DWORDS } else { sgpr_dwords }) } else { tr.sb_var() };
    let bnd = tr.alloc_binding(if ubo { BindingKind::SgprUniform } else { BindingKind::SgprData });
    tr.decorate_binding(sd, bnd);
    if !ubo && (spec.addresses || spec.sgpr_ubo != 0) {
        // Read-only: its loads need not wait for device-pointer stores.
        tr.b.decorate(sd, Decoration::NonWritable, []);
    }
    tr.sgpr_data = sd;
    if ubo {
        tr.sgpr_ub = Some(tr.b.type_pointer(None, StorageClass::Uniform, tr.t.u32));
    }
    if code.iter().any(|i| i.enc == Enc::Ds && i.ds_gds()) {
        let v = tr.sb_var();
        let bnd = tr.alloc_binding(BindingKind::Gds);
        tr.decorate_binding(v, bnd);
        tr.gds = Some(v);
    }
    if spec.write_marks && plan.resources.iter().any(|r| r.written) {
        let v = tr.sb_var();
        let bnd = tr.alloc_binding(BindingKind::Marks);
        tr.decorate_binding(v, bnd);
        tr.marks = Some(v);
    }
    if code.iter().any(|i| i.branch_target().is_some_and(|t| t <= i.pc)) {
        let v = tr.sb_var();
        let bnd = tr.alloc_binding(BindingKind::Watchdog);
        tr.decorate_binding(v, bnd);
        tr.watchdog = Some(v);
    }
    if (spec.dump.is_some() && matches!(spec.stage, Stage::Compute | Stage::Vertex)) || (spec.outlier.is_some() && matches!(spec.stage, Stage::Vertex)) {
        let v = tr.sb_var();
        let bnd = tr.alloc_binding(BindingKind::Debug);
        tr.decorate_binding(v, bnd);
        tr.dump_buf = Some(v);
    }
    if spec.lod_stats && code.iter().any(|i| i.name().is_some_and(|n| n.starts_with("image_sample"))) {
        let v = tr.sb_var();
        let bnd = tr.alloc_binding(BindingKind::LodStats);
        tr.decorate_binding(v, bnd);
        tr.lod_stats_var = Some(v);
    }
    // Images whose instructions select a mip level (bound one view per level
    // when they are storage images).
    let mip_res: BTreeSet<usize> = code
        .iter()
        .filter(|i| matches!(i.name(), Some("image_store_mip" | "image_load_mip")))
        .filter_map(|i| plan.inst_res.get(&i.pc).map(|&(ri, _)| ri))
        .collect();
    // Resource bindings in plan order.
    for (i, r) in plan.resources.iter().enumerate() {
        match (r.kind, spec.resources.get(i)) {
            (ResKind::Buffer | ResKind::Global, _) if spec.addresses => {
                // A placeholder id in place of a variable (`T::buffer_res`).
                let v = tr.b.id();
                tr.bindings.push(Binding { binding: u32::MAX, kind: BindingKind::Address(i) });
                tr.buffer_res.insert(v, i);
                tr.buffers.insert(i, v);
            }
            (ResKind::Buffer | ResKind::Global, _) => {
                let v = tr.sb_var();
                let bnd = tr.alloc_binding(BindingKind::Buffer(i));
                tr.decorate_binding(v, bnd);
                // Vector views of the same binding alias it.
                tr.b.decorate(v, Decoration::Aliased, []);
                tr.buffer_binding.insert(v, bnd);
                tr.buffers.insert(i, v);
            }
            (ResKind::Image, Some(ResSpec::Image { dim: ImageDim::D3OrD2, class, depth })) if !r.written => {
                let (class, depth) = (*class, *depth);
                let sampled_ty = match class {
                    NumClass::Float => tr.t.f32,
                    NumClass::Sint => tr.t.i32,
                    NumClass::Uint => tr.t.u32,
                };
                for (alt, sd, dim) in [(false, spirv::Dim::Dim3D, ImageDim::D3), (true, spirv::Dim::Dim2D, ImageDim::D2)] {
                    let img_ty = tr.b.type_image(sampled_ty, sd, 0, 0, 0, 1, spirv::ImageFormat::Unknown, None);
                    let pt = tr.b.type_pointer(None, StorageClass::UniformConstant, img_ty);
                    let v = tr.global_var(pt, StorageClass::UniformConstant);
                    let bnd = tr.alloc_binding(if alt { BindingKind::ImageAlt(i) } else { BindingKind::Image(i) });
                    tr.decorate_binding(v, bnd);
                    let res = ImageRes { var: v, img_ty, dim, class, storage: false, depth, mips: false };
                    if alt {
                        tr.image_alts.insert(i, res);
                    } else {
                        tr.images.insert(i, res);
                    }
                }
            }
            (ResKind::Image, Some(ResSpec::Image { dim, class, depth })) => {
                let (dim, class, depth) = (*dim, *class, *depth);
                let sampled_ty = match class {
                    NumClass::Float => tr.t.f32,
                    NumClass::Sint => tr.t.i32,
                    NumClass::Uint => tr.t.u32,
                };
                // 1D images are 2D images of height 1 on the host (so they can
                // alias other images); coordinates get a zero y component.
                let (sd, arrayed, ms) = match dim {
                    ImageDim::D1 | ImageDim::D2 => (spirv::Dim::Dim2D, 0, 0),
                    ImageDim::D3 | ImageDim::D3OrD2 => (spirv::Dim::Dim3D, 0, 0),
                    ImageDim::Cube => (spirv::Dim::DimCube, 0, 0),
                    ImageDim::CubeArray => (spirv::Dim::DimCube, 1, 0),
                    ImageDim::D1Array | ImageDim::D2Array => (spirv::Dim::Dim2D, 1, 0),
                    ImageDim::D2Msaa => (spirv::Dim::Dim2D, 0, 1),
                    ImageDim::D2MsaaArray => (spirv::Dim::Dim2D, 1, 1),
                };
                let storage = r.written;
                let mips = storage && spec.mip_views && mip_res.contains(&i);
                let img_ty = tr.b.type_image(sampled_ty, sd, 0, arrayed, ms, if storage { 2 } else { 1 }, spirv::ImageFormat::Unknown, None);
                let var_ty = if mips {
                    // Indexed by each lane's mip operand.
                    tr.b.extension("SPV_EXT_descriptor_indexing");
                    tr.b.capability(Capability::ShaderNonUniform);
                    tr.b.capability(Capability::StorageImageArrayNonUniformIndexing);
                    let n = tr.u(MIP_VIEWS);
                    tr.b.type_array(img_ty, n)
                } else {
                    img_ty
                };
                let pt = tr.b.type_pointer(None, StorageClass::UniformConstant, var_ty);
                let v = tr.global_var(pt, StorageClass::UniformConstant);
                let bnd = tr.alloc_binding(if mips {
                    BindingKind::StorageImageMips(i)
                } else if storage {
                    BindingKind::StorageImage(i)
                } else {
                    BindingKind::Image(i)
                });
                tr.decorate_binding(v, bnd);
                tr.images.insert(i, ImageRes { var: v, img_ty, dim, class, storage, depth, mips });
            }
            (ResKind::Sampler, _) => {
                // Bound at first use (`sampler_var`), plain or comparing.
                tr.sampler_res.insert(i);
            }
            (ResKind::ImageTable, Some(ResSpec::Image { dim, class, .. })) => {
                if r.written {
                    return Err(format!("image table {i} is written (storage tables unsupported)"));
                }
                let (dim, class) = (*dim, *class);
                let sampled_ty = match class {
                    NumClass::Float => tr.t.f32,
                    NumClass::Sint => tr.t.i32,
                    NumClass::Uint => tr.t.u32,
                };
                let (sd, arrayed) = match dim {
                    ImageDim::D1 | ImageDim::D2 | ImageDim::D2Msaa => (spirv::Dim::Dim2D, 0),
                    ImageDim::D3 | ImageDim::D3OrD2 => (spirv::Dim::Dim3D, 0),
                    ImageDim::Cube => (spirv::Dim::DimCube, 0),
                    ImageDim::CubeArray => (spirv::Dim::DimCube, 1),
                    ImageDim::D1Array | ImageDim::D2Array | ImageDim::D2MsaaArray => (spirv::Dim::Dim2D, 1),
                };
                tr.b.extension("SPV_EXT_descriptor_indexing");
                tr.b.capability(Capability::ShaderNonUniform);
                tr.b.capability(Capability::SampledImageArrayNonUniformIndexing);
                let img_ty = tr.b.type_image(sampled_ty, sd, 0, arrayed, 0, 1, spirv::ImageFormat::Unknown, None);
                let n = tr.u(IMAGE_TABLE_LEN);
                let arr = tr.b.type_array(img_ty, n);
                let pt = tr.b.type_pointer(None, StorageClass::UniformConstant, arr);
                let v = tr.global_var(pt, StorageClass::UniformConstant);
                let bnd = tr.alloc_binding(BindingKind::ImageArray(i));
                tr.decorate_binding(v, bnd);
                tr.images.insert(i, ImageRes { var: v, img_ty, dim, class, storage: false, depth: false, mips: false });
                tr.image_tables.insert(i, r.table.ok_or("image table without its load")?);
            }
            (ResKind::Image | ResKind::ImageTable, _) => return Err(format!("image resource {i} has no specialization")),
        }
    }
    // Function.
    let fn_ty = tr.b.type_function(void, []);
    let func = e(tr.b.begin_function(void, None, spirv::FunctionControl::NONE, fn_ty))?;
    e(tr.b.begin_block(None))?;
    let arr_s = {
        let n = tr.u(128);
        tr.b.type_array(tr.t.u32, n)
    };
    let arr_v = {
        let n = tr.u(256);
        tr.b.type_array(tr.t.u32, n)
    };
    let p_arr_s = tr.b.type_pointer(None, StorageClass::Function, arr_s);
    let p_arr_v = tr.b.type_pointer(None, StorageClass::Function, arr_v);
    // Relative register addressing needs the files as arrays; otherwise
    // one variable per register (zero).
    let relative = code.iter().any(|i| i.name().is_some_and(|n| n.starts_with("s_movrel") || n.starts_with("v_movrel")));
    if relative || !spec.reg_vars {
        tr.sgpr = tr.b.variable(p_arr_s, None, StorageClass::Function, None);
        tr.vgpr = tr.b.variable(p_arr_v, None, StorageClass::Function, None);
    } else {
        let z = tr.u(0);
        tr.sregs = (0..128).map(|_| tr.b.variable(tr.t.fn_u32, None, StorageClass::Function, Some(z))).collect();
        tr.vregs = (0..256).map(|_| tr.b.variable(tr.t.fn_u32, None, StorageClass::Function, Some(z))).collect();
    }
    tr.scc = tr.b.variable(tr.t.fn_bool, None, StorageClass::Function, None);
    tr.pc_var = tr.b.variable(tr.t.fn_u32, None, StorageClass::Function, None);
    if tr.watchdog.is_some() {
        let z = tr.u(0);
        tr.loop_ctr = Some(tr.b.variable(tr.t.fn_u32, None, StorageClass::Function, Some(z)));
    }
    if tr.spec.probe.is_some() {
        let z = tr.u(0);
        tr.probe_var = Some(tr.b.variable(tr.t.fn_u32, None, StorageClass::Function, Some(z)));
    }
    let mut table_pcs: Vec<u32> = tr.image_tables.values().copied().collect();
    table_pcs.sort();
    table_pcs.dedup();
    for pc in table_pcs {
        let z = tr.u(0);
        let v = tr.b.variable(tr.t.fn_u32, None, StorageClass::Function, Some(z));
        tr.table_offsets.insert(pc, v);
    }
    // Lockstep mode (see the module documentation).
    tr.lockstep = lockstep;
    if spec.known_active && !tr.lockstep && !tr.wave_masks && !spec.lane_votes {
        tr.known_active = active::known_active(code, spec.wave64, &tr.sg);
    }
    if tr.lockstep {
        // (Subgroup waves vote nowhere: no workgroup memory.)
        if !tr.sg_waves {
            let z = tr.u(0);
            tr.vote_ctr = Some(tr.b.variable(tr.t.fn_u32, None, StorageClass::Function, Some(z)));
            let len = tr.u(6);
            let arr = tr.b.type_array(tr.t.u32, len);
            let pt = tr.b.type_pointer(None, StorageClass::Workgroup, arr);
            tr.vote = Some(tr.global_var(pt, StorageClass::Workgroup));
        }
        tr.uniformity.set_lockstep();
    }
    if tr.spec.resources.iter().any(|r| matches!(r, ResSpec::Buffer { format: FORMAT_RT, .. })) {
        let n = tr.u(128);
        let arr = tr.b.type_array(tr.t.u32, n);
        let entries: Vec<Word> = (0..128u16).map(|id| tr.u(rt_format_entry(id))).collect();
        let init = tr.b.constant_composite(arr, entries);
        let pt = tr.b.type_pointer(None, StorageClass::Private, arr);
        // (Module scope, with its initializer.)
        let id = tr.b.id();
        let inst = rspirv::dr::Instruction::new(spirv::Op::Variable, Some(pt), Some(id), vec![Operand::StorageClass(StorageClass::Private), Operand::IdRef(init)]);
        tr.b.module_mut().types_global_values.push(inst);
        tr.fmt_table = Some(id);
    }
    for ins in code {
        if let Some((true, vgpr, lane)) = ins.const_lane_op() {
            if !tr.lane_slots.contains_key(&(vgpr as u32, lane)) {
                let v = tr.b.variable(tr.t.fn_u32, None, StorageClass::Function, None);
                tr.lane_slots.insert((vgpr as u32, lane), v);
            }
        }
    }
    if !tr.sg.leaders.is_empty() {
        let z = tr.u(0);
        tr.leader = Some(tr.b.variable(tr.t.fn_u32, None, StorageClass::Function, Some(z)));
    }
    tr.sync_starts = sync_starts(&tr, code);
    if !tr.sync_starts.is_empty() {
        let z = tr.u(0);
        tr.sync_steps = Some(tr.b.variable(tr.t.fn_u32, None, StorageClass::Function, Some(z)));
    }
    // The command's SGPR data offset (`Spec::sgpr_pc`), loaded once in the
    // entry block (after its variables), which dominates every read.
    if spec.batched && spec.stage == Stage::Compute {
        // The table's offset (push constant byte 0), this workgroup's entry:
        // its command's SGPR data offset and its x within that command.
        let st = tr.b.id();
        tr.b.type_struct_id(Some(st), [tr.t.u32]);
        tr.b.decorate(st, Decoration::Block, []);
        tr.b.member_decorate(st, 0, Decoration::Offset, [Operand::LiteralBit32(0)]);
        let pt = tr.b.type_pointer(None, StorageClass::PushConstant, st);
        let var = tr.global_var(pt, StorageClass::PushConstant);
        let pu = tr.b.type_pointer(None, StorageClass::PushConstant, tr.t.u32);
        let z = tr.u(0);
        let p = e(tr.b.access_chain(pu, None, var, [z]))?;
        let pc = e(tr.b.load(tr.t.u32, None, p, None, []))?;
        let wid = tr.input(tr.t.v3u, Some(BuiltIn::WorkgroupId), None);
        let w = e(tr.b.load(tr.t.v3u, None, wid, None, []))?;
        let gx = e(tr.b.composite_extract(tr.t.u32, None, w, [0]))?;
        // [`BATCH_SINGLE`] set: a dispatch of its own, the value its SGPR
        // data offset (indirect dispatches, ones the runtime does not batch).
        let flag = tr.u(BATCH_SINGLE);
        let low = tr.u(!BATCH_SINGLE);
        let fl = e(tr.b.bitwise_and(tr.t.u32, None, pc, flag))?;
        let single = e(tr.b.i_not_equal(tr.t.bool_, None, fl, z))?;
        let own = e(tr.b.bitwise_and(tr.t.u32, None, pc, low))?;
        let k = tr.u(BATCH_ENTRY_DWORDS);
        let at = e(tr.b.i_mul(tr.t.u32, None, gx, k))?;
        let at = e(tr.b.i_add(tr.t.u32, None, at, pc))?;
        let at = e(tr.b.select(tr.t.u32, None, single, z, at))?;
        let one = tr.u(1);
        let at1 = e(tr.b.i_add(tr.t.u32, None, at, one))?;
        let base = tr.sgpr_raw(at)?;
        let local = tr.sgpr_raw(at1)?;
        let base = e(tr.b.select(tr.t.u32, None, single, own, base))?;
        let local = e(tr.b.select(tr.t.u32, None, single, gx, local))?;
        tr.sgpr_base = Some(base);
        tr.batch_wg = Some((local, w));
    } else if spec.batched && spec.stage == Stage::Vertex {
        // A batched draw: the table's offset (push constant byte 0), this
        // draw's entry by its draw index (DrawParameters is core in SPIR-V
        // 1.3): its SGPR data offset, and the pixel stage's for the varying.
        tr.b.capability(Capability::DrawParameters);
        let st = tr.b.id();
        tr.b.type_struct_id(Some(st), [tr.t.u32]);
        tr.b.decorate(st, Decoration::Block, []);
        tr.b.member_decorate(st, 0, Decoration::Offset, [Operand::LiteralBit32(0)]);
        let pt = tr.b.type_pointer(None, StorageClass::PushConstant, st);
        let var = tr.global_var(pt, StorageClass::PushConstant);
        let pu = tr.b.type_pointer(None, StorageClass::PushConstant, tr.t.u32);
        let z = tr.u(0);
        let p = e(tr.b.access_chain(pu, None, var, [z]))?;
        let pc = e(tr.b.load(tr.t.u32, None, p, None, []))?;
        let dv = tr.input(tr.t.u32, Some(BuiltIn::DrawIndex), None);
        let di = e(tr.b.load(tr.t.u32, None, dv, None, []))?;
        let k = tr.u(BATCH_DRAW_ENTRY_DWORDS);
        let at = e(tr.b.i_mul(tr.t.u32, None, di, k))?;
        let at = e(tr.b.i_add(tr.t.u32, None, at, pc))?;
        let one = tr.u(1);
        let at1 = e(tr.b.i_add(tr.t.u32, None, at, one))?;
        let own = tr.sgpr_raw(at)?;
        let ps = tr.sgpr_raw(at1)?;
        tr.sgpr_base = Some(own);
        let out = tr.output(tr.t.u32, None, Some(BATCH_VARYING));
        ok(tr.b.store(out, ps, None, []))?;
    } else if spec.batched && spec.stage == Stage::Fragment {
        // A batched draw's pixel stage: its SGPR data offset from the
        // vertex stage (not when one of its own inputs is that location:
        // the runtime then prepares the pair unbatched).
        if spec.ps_inputs.iter().any(|i| i.default.is_none() && i.param as u32 == BATCH_VARYING) {
            return Err(format!("a batched pixel stage reads parameter {BATCH_VARYING}, the batched draw's varying"));
        }
        let v = tr.input(tr.t.u32, None, Some(BATCH_VARYING));
        tr.b.decorate(v, Decoration::Flat, []);
        tr.sgpr_base = Some(e(tr.b.load(tr.t.u32, None, v, None, []))?);
    } else if let Some(off) = spec.sgpr_pc {
        // A struct type of its own (rspirv merges identical ones, and
        // this one's member offset is the stage's).
        let st = tr.b.id();
        tr.b.type_struct_id(Some(st), [tr.t.u32]);
        tr.b.decorate(st, Decoration::Block, []);
        tr.b.member_decorate(st, 0, Decoration::Offset, [Operand::LiteralBit32(off)]);
        let pt = tr.b.type_pointer(None, StorageClass::PushConstant, st);
        let var = tr.global_var(pt, StorageClass::PushConstant);
        let pu = tr.b.type_pointer(None, StorageClass::PushConstant, tr.t.u32);
        let z = tr.u(0);
        let p = e(tr.b.access_chain(pu, None, var, [z]))?;
        tr.sgpr_base = Some(e(tr.b.load(tr.t.u32, None, p, None, []))?);
    }
    // Initial state: all registers defined (zero), this lane active.
    let f = tr.bool_c(false);
    tr.set_scc(f)?;
    let zero = tr.u(0);
    if tr.sregs.is_empty() {
        let zs = tr.b.constant_null(arr_s);
        ok(tr.b.store(tr.sgpr, zs, None, []))?;
        let zv = tr.b.constant_null(arr_v);
        ok(tr.b.store(tr.vgpr, zv, None, []))?;
    }
    // EXEC holds this lane's bit (bit 0) only; in wave-mask mode the lanes
    // of the wave (subgroup waves: of the subgroup).
    let (exec_lo, exec_hi) = if tr.sg_waves {
        let t = tr.bool_c(true);
        tr.subgroup_ballot(t)?
    } else if tr.wave_masks {
        let threads: u32 = spec.cs_local_size.iter().map(|&x| x.max(1)).product();
        let lo = if threads >= 32 { u32::MAX } else { (1u32 << threads) - 1 };
        let hi = if !spec.wave64 || threads <= 32 { 0 } else if threads >= 64 { u32::MAX } else { (1u32 << (threads - 32)) - 1 };
        (tr.u(lo), tr.u(hi))
    } else {
        (tr.u(1), zero)
    };
    tr.set_sreg(crate::OP_EXEC_LO as u32, exec_lo)?;
    tr.set_sreg(crate::OP_EXEC_HI as u32, exec_hi)?;
    // (`Spec::dep_loads`: the program address's high dword shifted right
    // by 16, zero for every user-space address (below 2^47), read twice, the
    // second at an index the first gives, offsets every user SGPR's index.)
    let dep = if spec.dep_loads {
        let hi = tr.code_base_at + 1;
        let eight = tr.u(16);
        let k1 = tr.u(hi);
        let k1 = tr.sgpr_index(k1)?;
        let w1 = tr.sgpr_raw(k1)?;
        let d1 = e(tr.b.shift_right_logical(tr.t.u32, None, w1, eight))?;
        let k2 = tr.u(hi);
        let k2 = e(tr.b.i_add(tr.t.u32, None, k2, d1))?;
        let k2 = tr.sgpr_index(k2)?;
        let w2 = tr.sgpr_raw(k2)?;
        Some(e(tr.b.shift_right_logical(tr.t.u32, None, w2, eight))?)
    } else {
        None
    };
    for i in 0..spec.user_sgprs.min(SGPR_DATA_USER as u16) as u32 {
        let v = match dep {
            Some(d) => {
                let k = tr.u(i);
                let k = e(tr.b.i_add(tr.t.u32, None, k, d))?;
                let k = tr.sgpr_index(k)?;
                tr.sgpr_raw(k)?
            }
            None => tr.sgpr_data(i)?,
        };
        tr.set_sreg(spec.user_base as u32 + i, v)?;
    }
    if spec.addresses {
        mem::preload_addresses(&mut tr)?;
    }
    mem::stage_inputs(&mut tr)?;
    if tr.wave_masks {
        let lane = tr.lane()?;
        let k32 = tr.u(32);
        let k31 = tr.u(31);
        let in_hi = e(tr.b.u_greater_than_equal(tr.t.bool_, None, lane, k32))?;
        let sh = e(tr.b.bitwise_and(tr.t.u32, None, lane, k31))?;
        tr.own_bit = Some((in_hi, sh));
    }
    if let (Some(vote), Some(li)) = (tr.vote, tr.local_index) {
        // Ballot slots start cleared.
        let z = tr.u(0);
        let first = e(tr.b.i_equal(tr.t.bool_, None, li, z))?;
        let p_wg = tr.b.type_pointer(None, StorageClass::Workgroup, tr.t.u32);
        tr.guarded(first, &mut |tr| {
            for k in 0..6 {
                let i = tr.u(k);
                let p = e(tr.b.access_chain(p_wg, None, vote, [i]))?;
                ok(tr.b.store(p, z, None, []))?;
            }
            Ok(())
        })?;
        tr.workgroup_barrier()?;
    }

    // Buffers only scalar loads read (`Spec::hw_bounds`: their loads go
    // unchecked; a buffer vector instructions also access keeps its checks,
    // as one binding's range cannot end where both kinds' checks do when
    // its base is off a dword boundary).
    if spec.hw_bounds {
        let (mut scalar, mut other) = (0u64, 0u64);
        for ins in code {
            if let Some(&(r, _)) = plan.inst_res.get(&ins.pc) {
                if r >= 64 {
                    continue;
                }
                let load = ins.enc == Enc::Smem && ins.name().is_some_and(|n| n.starts_with("s_buffer_load_dword") || n.starts_with("s_load_dword"));
                if load {
                    scalar |= 1u64 << r;
                } else {
                    other |= 1u64 << r;
                }
            }
        }
        tr.scalar_only = scalar & !other;
    }

    // Body.
    let starts = block_starts(code);
    if starts.len() == 1 {
        for ins in code {
            if ins.is_endpgm() {
                break;
            }
            translate_inst(&mut tr, ins)?;
        }
        ok(tr.b.ret())?;
    } else {
        match loop_ranges(code, &starts) {
            Some(loops) if !spec.force_dispatcher && (loops.is_empty() || !spec.dispatcher_for_loops) => emit_structured(&mut tr, code, &starts, &loops)?,
            _ => emit_dispatch_loop(&mut tr, code, &starts)?,
        }
    }
    ok(tr.b.end_function())?;

    let model = match spec.stage {
        Stage::Vertex => ExecutionModel::Vertex,
        Stage::Fragment => ExecutionModel::Fragment,
        Stage::Compute => ExecutionModel::GLCompute,
    };
    let iface = tr.interface.clone();
    tr.b.entry_point(model, func, "main", iface);
    // Hardware rounds every float multiply and add (v_mad_f32 / v_mac_f32
    // are not fused): keep the driver from contracting them into FMAs, which
    // changes results in the last bit (texture coordinates on texel
    // boundaries select other texels).
    let ids: Vec<Word> = tr
        .b
        .module_ref()
        .functions
        .iter()
        .flat_map(|f| f.blocks.iter())
        .flat_map(|b| b.instructions.iter())
        .filter(|i| matches!(i.class.opcode, spirv::Op::FAdd | spirv::Op::FSub | spirv::Op::FMul))
        .filter_map(|i| i.result_id)
        .collect();
    for id in ids {
        tr.b.decorate(id, Decoration::NoContraction, []);
    }
    match spec.stage {
        Stage::Fragment => {
            tr.b.execution_mode(func, ExecutionMode::OriginUpperLeft, []);
            if tr.out_depth.is_some() {
                tr.b.execution_mode(func, ExecutionMode::DepthReplacing, []);
            } else if spec.early_tests || (spec.early_ok && !writes_memory(code)) {
                tr.b.execution_mode(func, ExecutionMode::EarlyFragmentTests, []);
            }
        }
        Stage::Compute => {
            let l = spec.cs_local_size;
            tr.b.execution_mode(func, ExecutionMode::LocalSize, [l[0].max(1), l[1].max(1), l[2].max(1)]);
        }
        Stage::Vertex => {}
    }
    let spirv = tr.b.module().assemble();
    let param_mask = tr.out_params.keys().filter(|&&p| p < 32).fold(0u32, |m, &p| m | 1 << p);
    // Batched translations bind the whole ring as their SGPR data (their
    // offsets come from a table): the runtime takes that from `sgpr_pc`.
    let batched = spec.batched;
    let sgpr_pc = if batched { Some(sgpr_pc_offset(spec.stage)) } else { spec.sgpr_pc };
    Ok(Output { spirv, bindings: tr.bindings, params: tr.params, mrt_mask: tr.mrt_mask, param_mask, writes_depth: tr.out_depth.is_some(), warnings: tr.warnings, sgpr_pc, batched, masked_skips: tr.masked_skips, lockstep: tr.lockstep, wave_masks: tr.wave_masks, sg_waves: tr.sg_waves, sg_appends: tr.sg.leaders.len() as u32, known_active: tr.known_active.len() as u32, unchecked_res: tr.unchecked_res, in_mask: tr.in_locs, bary_gs: tr.bary_io })
}

/// Wave-mask mode for the shader at `addr` (diagnosis: `DS_WAVEMASK=0`
/// disables it, `DS_WAVEMASK=<hex address>[,...]` keeps it for those
/// shaders only).
fn wave_masks_wanted(addr: u64) -> bool {
    match std::env::var("DS_WAVEMASK") {
        Err(_) => true,
        Ok(v) => v.split(',').any(|a| u64::from_str_radix(a.trim().trim_start_matches("0x"), 16).is_ok_and(|x| x == addr && x != 0)),
    }
}

/// `DS_SGWAVE=<hex,...>`: subgroup waves for these programs too (by code
/// address; diagnosis, e.g. with `rdnadis --translate`).
fn sg_waves_wanted(addr: u64) -> bool {
    match std::env::var("DS_SGWAVE") {
        Err(_) => false,
        Ok(v) => v.split([',', ';']).any(|a| u64::from_str_radix(a.trim().trim_start_matches("0x"), 16).is_ok_and(|x| x == addr && x != 0)),
    }
}

/// Why a wave-mask program cannot run as subgroup waves, if it cannot:
/// they are wave64 compute programs translated structured (each subgroup's
/// invocations take its branches together), with no LDS (the waves would
/// address the workgroup's alike) or lane exchange through it, and no
/// s_barrier.
fn sg_waves_refusal(spec: &Spec, code: &[Inst], structured: bool) -> Option<&'static str> {
    if spec.stage != Stage::Compute || !spec.wave64 {
        return Some("not a wave64 compute program");
    }
    if !structured {
        return Some("a dispatcher translation");
    }
    if code.iter().any(|i| i.enc == Enc::Ds) {
        return Some("LDS or a lane exchange through it");
    }
    if code.iter().any(|i| i.base_name() == Some("s_barrier")) {
        return Some("s_barrier");
    }
    None
}

/// Lockstep mode, and whether the workgroup is a single wave (see the
/// module documentation).
fn lockstep_mode(spec: &Spec, code: &[Inst], sg_ballots: &crate::HashSet<u32>) -> (bool, bool) {
    let threads: u32 = spec.cs_local_size.iter().map(|&x| x.max(1)).product();
    let uses_lds = code.iter().any(|i| i.enc == Enc::Ds && !i.ds_gds() && !matches!(i.name(), Some("ds_swizzle_b32" | "ds_bpermute_b32" | "ds_permute_b32")));
    let one_wave = threads <= if spec.wave64 { 64 } else { 32 };
    let wave_scalars = spec.cs_rsrc2 & (1 << 10) != 0 || code.iter().any(|i| matches!(i.base_name(), Some("v_readlane_b32" | "v_readfirstlane_b32")));
    // A single wave that reads lane masks as values (ballots it stores or
    // counts) needs wave-mask mode, which runs in lockstep, with or without
    // LDS (the title's batch culling stores one compare ballot per 32 items);
    // the ballots of appends done per subgroup do not count.
    // (`DS_MASK_VALUES=0`: not for such programs, diagnosis.)
    let mask_values = spec.stage == Stage::Compute
        && one_wave
        && !uses_lds
        && std::env::var("DS_MASK_VALUES").ok().as_deref() != Some("0")
        && crate::wavemask::reads_mask_values_except(code, spec.wave64, sg_ballots);
    // A single wave whose cross-lane operations sit in code its invocations
    // may not all run (since 2026-11-02, when the mask analysis stopped
    // reading unused source fields: such programs left lockstep then).
    let divergent_lanes = spec.stage == Stage::Compute && one_wave && !uses_lds && !mask_values && !divergent_lane_ops(code).is_empty();
    let lockstep = spec.stage == Stage::Compute
        && (uses_lds || mask_values || divergent_lanes)
        && (one_wave || !wave_scalars)
        && std::env::var("DS_LOCKSTEP").ok().as_deref() != Some("0")
        && !spec.per_lane
        && !spec.lockstep_off
        && !lockstep_off(spec.code_base);
    (lockstep, one_wave)
}

/// Pcs of the cross-lane operations ([`crate::cross_lane`]) in blocks a
/// program's invocations may not all run, lane reads of values every lane
/// holds alike left out: outside lockstep a wave64 reads only its 32-lane
/// subgroup there, and other lanes' values from invocations that may be
/// elsewhere (such single waves run in lockstep).
pub fn divergent_lane_ops(code: &[Inst]) -> Vec<u32> {
    if code.is_empty() {
        return Vec::new();
    }
    let u = crate::uniform::analyze(code);
    let starts = block_starts(code);
    code.iter().enumerate().filter(|(k, i)| crate::cross_lane(i) && !u.is_uniform(i.pc) && !uniform_lane_read(code, &starts, *k)).map(|(_, i)| i.pc).collect()
}

/// Whether the lane read at `code[k]` (v_readlane_b32, v_readfirstlane_b32)
/// reads a VGPR whose value every lane holds alike, so that any lane's copy
/// is the one the hardware reads ([`uniform_vgpr`]).
fn uniform_lane_read(code: &[Inst], starts: &BTreeSet<u32>, k: usize) -> bool {
    let i = &code[k];
    if !matches!(i.base_name(), Some("v_readlane_b32" | "v_readfirstlane_b32")) || !matches!(i.ext, crate::Ext::None) {
        return false;
    }
    let src = i.src(0);
    src >= crate::OP_VGPR0 && uniform_vgpr(code, starts, k, src - crate::OP_VGPR0, 12)
}

/// Whether VGPR `v` holds the same value in every lane where `code[k]`
/// reads it: its last write before `k` within the block computes it from
/// SGPRs, constants and such VGPRs only (a lane-local VALU instruction), or
/// loads it with a buffer load whose address VGPRs are such. Conservative:
/// a write outside the block, any other writer, a 64-bit write whose
/// extent is unclear, or `depth` used up count as lane-dependent.
fn uniform_vgpr(code: &[Inst], starts: &BTreeSet<u32>, k: usize, v: u16, depth: u8) -> bool {
    if depth == 0 {
        return false;
    }
    // Scalar operands are one value for the wave, but lane masks (VCC,
    // EXEC, SCC and their tests) are each invocation's own outside wave-mask
    // mode: SGPRs s0-s105, M0, inline constants and literals count.
    let src_ok = |op: u16, at: usize| match op {
        0..=105 | crate::OP_M0 | 128..=248 | 255 => true,
        _ if op >= crate::OP_VGPR0 => uniform_vgpr(code, starts, at, op - crate::OP_VGPR0, depth - 1),
        _ => false,
    };
    let mut j = k;
    while j > 0 && !starts.contains(&code[j].pc) {
        j -= 1;
        let w = &code[j];
        let Some(name) = w.base_name() else { return false };
        let valu = matches!(w.enc, Enc::Vop1 | Enc::Vop2 | Enc::Vop3);
        let load = matches!(w.enc, Enc::Mubuf) && name.starts_with("buffer_load_") && !name.contains("lds");
        if !valu && !load {
            // Scalar instructions, compares, exports and waits write no
            // VGPR; other vector writes (and returning atomics) are not
            // followed: lane-dependent where they may write `v`.
            let writes = matches!(w.enc, Enc::Mtbuf | Enc::Mimg | Enc::Ds | Enc::Flat | Enc::Global | Enc::Scratch | Enc::Vintrp | Enc::Vop3p) || w.enc == Enc::Mubuf && name.contains("atomic");
            if writes && w.vdst() <= v && v < w.vdst().saturating_add(4) {
                return false;
            }
            continue;
        }
        if valu && w.enc == Enc::Vop3 && crate::vop3_kind(w.op) == crate::Vop3Kind::FromVopc {
            continue;
        }
        // (Lane reads write an SGPR.)
        if name.starts_with("v_readlane") || name.starts_with("v_readfirstlane") {
            continue;
        }
        let d = w.vdst();
        let n: u16 = if load {
            match ["x4", "xyzw", "x3", "xyz", "x2", "xy"].iter().position(|x| name.ends_with(x)) {
                Some(0 | 1) => 4,
                Some(2 | 3) => 3,
                Some(_) => 2,
                None => 1,
            }
        } else {
            1
        };
        let wide = valu && name.contains("64");
        if wide && (v == d + 1 || v == d) {
            return false;
        }
        if !(d <= v && v < d + n) {
            continue;
        }
        if load {
            let a = crate::OP_VGPR0 + w.buf_vaddr();
            return !(w.buf_idxen() || bits_offen(w)) || src_ok(a, j) && (!(w.buf_idxen() && bits_offen(w)) || src_ok(a + 1, j));
        }
        // A lane-local VALU instruction: its result depends on its sources.
        if !matches!(w.ext, crate::Ext::None) || (w.enc == Enc::Vop3 && w.is_vop3b()) {
            return false;
        }
        if name.starts_with("v_cmp") || name.starts_with("v_writelane") || name.starts_with("v_permlane") || name.starts_with("v_movrel") || name.starts_with("v_swap") || name.starts_with("v_mbcnt") || name.starts_with("v_cndmask") || name.starts_with("v_interp") || name.contains("_co_") || name.contains("_ci_") {
            return false;
        }
        let n_src = match w.enc {
            Enc::Vop1 => 1,
            Enc::Vop2 => 2,
            _ => match crate::vop3_kind(w.op) {
                crate::Vop3Kind::FromVop1 => 1,
                crate::Vop3Kind::FromVop2 => 2,
                _ => 3,
            },
        };
        // VOP2's mac/fma forms also read their destination.
        let acc = name.starts_with("v_mac_") || name.starts_with("v_fmac_");
        return (0..n_src).all(|s| src_ok(w.src(s), j)) && (!acc || uniform_vgpr(code, starts, j, d, depth - 1));
    }
    false
}

/// MUBUF OFFEN (word 0 bit 12).
fn bits_offen(i: &Inst) -> bool {
    (i.w0 >> 12) & 1 != 0
}

/// Diagnosis: `DS_LOCKSTEP_OFF=<hex address>[,...]` translates those
/// shaders without lockstep (and wave-mask) mode, each invocation a wave
/// of one (translations are then keyed by address, as with DS_WAVEMASK;
/// `;` separates too, for run drivers that split their variables at commas).
fn lockstep_off(addr: u64) -> bool {
    match std::env::var("DS_LOCKSTEP_OFF") {
        Err(_) => false,
        Ok(v) => v.split([',', ';']).any(|a| u64::from_str_radix(a.trim().trim_start_matches("0x"), 16).is_ok_and(|x| x == addr && x != 0)),
    }
}

const PC_EXIT: u32 = 0xFFFF_FFFF;

/// Branch condition of a conditional SOPP branch (None = always taken).
/// `before` holds the block's instructions before the branch.
fn branch_cond(tr: &mut T, ins: &Inst, before: &[Inst]) -> R<Option<Word>> {
    let name = ins.name().unwrap_or("");
    let bt = tr.t.bool_;
    if tr.lockstep && !tr.wave_masks && tr.uniformity.divergent_branches.contains(&ins.pc) {
        // The wave's decision: EXEC/VCC zero when no lane has its bit; SCC
        // from lane masks is the OR of the lanes' SCC (results != 0), or the
        // AND for equality compares.
        let (lane, all) = match name {
            "s_cbranch_vccz" | "s_cbranch_vccnz" | "s_cbranch_execz" | "s_cbranch_execnz" => {
                let src = if name.contains("vcc") { crate::OP_VCC_LO } else { crate::OP_EXEC_LO } as u32;
                let lo = tr.sreg(src)?;
                let hi = tr.sreg(src + 1)?;
                let o = e(tr.b.bitwise_or(tr.t.u32, None, lo, hi))?;
                let z = tr.u(0);
                (e(tr.b.i_not_equal(bt, None, o, z))?, false)
            }
            "s_cbranch_scc0" | "s_cbranch_scc1" => {
                let all = match before.iter().rev().find(|i| crate::uniform::writes_scc(i)) {
                    Some(w) => {
                        let n = w.name().unwrap_or("");
                        n.starts_with("s_cmp_eq_") || n.starts_with("s_cmpk_eq_")
                    }
                    None => {
                        tr.warn(format!("{:#x}: lane-dependent SCC from another block decided as any-lane", ins.pc));
                        false
                    }
                };
                (tr.get_scc()?, all)
            }
            _ => return Ok(Some(tr.bool_c(false))),
        };
        let wave = if all {
            let not = e(tr.b.logical_not(bt, None, lane))?;
            let any_not = tr.wave_any(not)?;
            e(tr.b.logical_not(bt, None, any_not))?
        } else {
            tr.wave_any(lane)?
        };
        return Ok(Some(match name {
            "s_cbranch_scc1" | "s_cbranch_vccnz" | "s_cbranch_execnz" => wave,
            _ => e(tr.b.logical_not(bt, None, wave))?,
        }));
    }
    Ok(match name {
        "s_branch" => None,
        "s_cbranch_scc0" => {
            let s = tr.get_scc()?;
            Some(e(tr.b.logical_not(bt, None, s))?)
        }
        "s_cbranch_scc1" => Some(tr.get_scc()?),
        "s_cbranch_vccz" | "s_cbranch_vccnz" | "s_cbranch_execz" | "s_cbranch_execnz" => {
            let src = if name.contains("vcc") { crate::OP_VCC_LO } else { crate::OP_EXEC_LO } as u32;
            let lo = tr.sreg(src)?;
            let hi = if tr.spec.wave64 { tr.sreg(src + 1)? } else { tr.u(0) };
            let o = e(tr.b.bitwise_or(tr.t.u32, None, lo, hi))?;
            let z = tr.u(0);
            let zero = e(tr.b.i_equal(bt, None, o, z))?;
            // `Spec::lane_votes` (per lane): decided for the 32-lane subgroup
            // as the hardware decides for the wave, so a lane whose EXEC (or
            // VCC: compares leave inactive lanes' bits clear) is empty stays
            // with its subgroup through the region, writing nothing, where
            // cross-lane operations (DPP, permlanes, lane reads after an
            // EXEC-all-lanes save) read it as hardware reads a lane EXEC
            // masked off. Decided per invocation, it left the region and the
            // reads found no value (the froxel fog's light-mask unions took
            // garbage at the volume's right edge, whose lanes skip the work).
            let zero = if tr.spec.lane_votes { tr.subgroup_all(zero)? } else { zero };
            Some(if name.ends_with('z') && !name.ends_with("nz") { zero } else { e(tr.b.logical_not(bt, None, zero))? })
        }
        _ => {
            tr.warn(format!("{:#x}: {name} never taken", ins.pc));
            Some(tr.bool_c(false))
        }
    })
}

/// Starts of the sync blocks of a compute program outside lockstep mode:
/// uniform blocks with an instruction that may place a workgroup barrier
/// (s_barrier, an LDS access, a wave64 v_readlane / v_readfirstlane).
fn sync_starts(tr: &T, code: &[Inst]) -> Vec<u32> {
    if tr.spec.stage != Stage::Compute || tr.lockstep {
        return Vec::new();
    }
    let starts: Vec<u32> = block_starts(code).into_iter().collect();
    if starts.len() <= 1 {
        return Vec::new();
    }
    let barrier = |i: &Inst| match i.base_name() {
        Some("s_barrier") => true,
        Some("v_readlane_b32" | "v_readfirstlane_b32") => tr.spec.wave64,
        _ => i.enc == Enc::Ds && !i.ds_gds() && !matches!(i.name(), Some("ds_swizzle_b32" | "ds_bpermute_b32" | "ds_permute_b32")),
    };
    let mut out = Vec::new();
    for (bi, &s) in starts.iter().enumerate() {
        let end = starts.get(bi + 1).copied().unwrap_or(u32::MAX);
        if code.iter().any(|i| i.pc >= s && i.pc < end && tr.uniformity.is_uniform(i.pc) && barrier(i)) {
            out.push(s);
        }
    }
    out
}

/// Translates block [start, end) and stores the address of the block to
/// run next (PC_EXIT after s_endpgm) in `pc_var`.
/// A vector instruction that reads and writes only its own lane's registers
/// (and reads SGPRs): no SGPR, VCC or EXEC write (compares, carries, VOP3b,
/// lane reads), no other lane's register (DPP, permlane, writelane). Runs of
/// them share one lane guard (`Spec::lane_guards`).
fn lane_local(ins: &Inst) -> bool {
    if !matches!(ins.enc, Enc::Vop1 | Enc::Vop2 | Enc::Vop3 | Enc::Vop3p) || !matches!(ins.ext, crate::Ext::None | crate::Ext::Sdwa(_)) {
        return false;
    }
    if ins.enc == Enc::Vop3 && ins.is_vop3b() {
        return false;
    }
    let Some(name) = ins.base_name() else { return false };
    !(name.starts_with("v_cmp")
        || name.starts_with("v_readlane")
        || name.starts_with("v_readfirstlane")
        || name.starts_with("v_writelane")
        || name.starts_with("v_permlane")
        || name.starts_with("v_movrel")
        || name.starts_with("v_swap")
        || name.contains("_co_")
        || name.contains("_ci_"))
}

/// Shortest run of lane-local instructions given a lane guard of its own
/// (the guard's branch and labels cost about one select's words).
const LANE_GUARD_MIN: usize = 3;

/// Whether `ins` has no effect in a lane EXEC masks off: a lane-local VALU
/// op, a vector memory or LDS access (not GDS) or a wait (`masked_skip`).
/// Whether a program writes memory (buffer, image, flat and global stores
/// and atomics; LDS and scratch stay private): side effects that depth and
/// stencil tests before the shader would leave out for hidden fragments
/// (`Spec::early_ok`).
pub fn writes_memory(code: &[Inst]) -> bool {
    code.iter().any(|ins| {
        let Some(n) = ins.name() else { return false };
        match ins.enc {
            Enc::Mubuf | Enc::Mtbuf | Enc::Flat | Enc::Global => n.contains("_store") || n.contains("_atomic"),
            Enc::Mimg => n.starts_with("image_store") || n.starts_with("image_atomic"),
            Enc::Ds => ins.ds_gds(),
            _ => false,
        }
    })
}

fn exec_masked(ins: &Inst) -> bool {
    match ins.enc {
        Enc::Mubuf | Enc::Mtbuf | Enc::Mimg | Enc::Flat | Enc::Global | Enc::Scratch => true,
        Enc::Ds => !ins.ds_gds(),
        Enc::Sopp => matches!(ins.name(), Some("s_waitcnt" | "s_nop" | "s_inst_prefetch")),
        _ => lane_local(ins),
    }
}

/// A forward s_cbranch_execz over (or s_cbranch_execnz past) instructions
/// that all have no effect in lanes EXEC masks off: whether a wave takes it
/// changes nothing when it has no active lane, so lockstep mode need not
/// decide it for the workgroup (a vote: a workgroup barrier each time).
/// Some(false): execz never taken (the region runs, each lane by its EXEC
/// bit, as it does already for a wave with no active lane beside an active
/// one); Some(true): execnz always taken (the region it skips runs only
/// with no lane active). The bitonic sort steps of the title's particle
/// sorting cast two such votes per step (FINDINGS.md 2026-10-30).
/// `DS_MASKED_SKIP=0` (diagnosis): every lane-dependent branch of lockstep
/// mode decided by a vote, as before 2026-10-30.
pub(crate) fn masked_skip_on() -> bool {
    static W: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *W.get_or_init(|| std::env::var("DS_MASKED_SKIP").ok().as_deref() != Some("0"))
}

pub(crate) fn masked_skip(code: &[Inst], k: usize) -> Option<bool> {
    let ins = &code[k];
    let taken = match ins.name()? {
        "s_cbranch_execz" => false,
        "s_cbranch_execnz" => true,
        _ => return None,
    };
    let target = ins.branch_target()?;
    if target <= ins.pc {
        return None;
    }
    code[k + 1..].iter().take_while(|i| i.pc < target).all(exec_masked).then_some(taken)
}

fn emit_block_body(tr: &mut T, code: &[Inst], start: u32, end: u32) -> R<()> {
    // LDS accesses of other blocks are unknown here.
    tr.lds_state = 3;
    let mut next_pc = PC_EXIT;
    let mut terminated = false;
    let mut first = None;
    // The end (exclusive index) of the lane-guarded run being translated.
    let mut guarded_until = 0usize;
    for (k, ins) in code.iter().enumerate() {
        if ins.pc < start || ins.pc >= end {
            continue;
        }
        if k < guarded_until {
            continue;
        }
        let first = *first.get_or_insert(k);
        tr.cur_pc = ins.pc;
        if tr.spec.lane_guards && lane_local(ins) {
            let mut j = k;
            while j < code.len() && code[j].pc < end && lane_local(&code[j]) && !code[j].is_endpgm() && code[j].branch_target().is_none() {
                j += 1;
            }
            if j - k >= LANE_GUARD_MIN {
                let act = tr.active()?;
                let run = &code[k..j];
                let mut body = |tr: &mut T| {
                    tr.reset_vcache();
                    tr.lane_guard = true;
                    let r = run.iter().try_for_each(|i| translate_inst(tr, i));
                    tr.lane_guard = false;
                    tr.reset_vcache();
                    r
                };
                if act == tr.bool_c(true) {
                    body(tr)?;
                } else {
                    tr.guarded(act, &mut body)?;
                }
                guarded_until = j;
                next_pc = code.get(j).map(|n| n.pc).unwrap_or(PC_EXIT);
                continue;
            }
        }
        next_pc = code.get(k + 1).map(|n| n.pc).unwrap_or(PC_EXIT);
        if ins.is_endpgm() {
            let x = tr.u(PC_EXIT);
            ok(tr.b.store(tr.pc_var, x, None, []))?;
            terminated = true;
            break;
        }
        if let Some(target) = ins.branch_target() {
            // (Lockstep mode, `masked_skip`: no vote. `DS_MASKED_SKIP=0`:
            // every lane-dependent branch voted on, as before 2026-10-30.)
            // (Wave-mask mode with `DS_EXEC_PARTIAL=1` too: EXEC may then be
            // partial there.)
            let decide = masked_skip_on() && tr.lockstep && if tr.wave_masks { crate::wavemask::exec_partial() } else { tr.uniformity.divergent_branches.contains(&ins.pc) };
            let skip = decide.then(|| masked_skip(code, k)).flatten();
            let cond = match skip {
                Some(taken) => {
                    tr.masked_skips += 1;
                    Some(tr.bool_c(taken))
                }
                None => branch_cond(tr, ins, &code[first..k])?,
            };
            let tv = tr.u(target);
            let val = match cond {
                None => tv,
                Some(c) => {
                    let fv = tr.u(next_pc);
                    e(tr.b.select(tr.t.u32, None, c, tv, fv))?
                }
            };
            ok(tr.b.store(tr.pc_var, val, None, []))?;
            terminated = true;
            break;
        }
        translate_inst(tr, ins)?;
    }
    if !terminated {
        let x = tr.u(if end == u32::MAX { PC_EXIT } else { next_pc });
        ok(tr.b.store(tr.pc_var, x, None, []))?;
    }
    Ok(())
}

/// Next-address value after the loop watchdog stopped an invocation (no
/// block starts there).
const PC_STOPPED: u32 = 0xFFFF_FFFE;

/// The loops of a program as address ranges [header, end) (end: the end of
/// the block of its last back edge), outermost first for equal headers; None
/// when two ranges overlap without nesting or a branch enters a range
/// elsewhere than at its header (the dispatcher handles those).
fn loop_ranges(code: &[Inst], starts: &BTreeSet<u32>) -> Option<Vec<(u32, u32)>> {
    let block_end = |pc: u32| starts.range(pc + 1..).next().copied().unwrap_or(u32::MAX);
    let mut loops: std::collections::BTreeMap<u32, u32> = std::collections::BTreeMap::new();
    for ins in code {
        if let Some(t) = ins.branch_target() {
            if t <= ins.pc {
                let e = block_end(ins.pc);
                let x = loops.entry(t).or_insert(e);
                *x = (*x).max(e);
            }
        }
    }
    let list: Vec<(u32, u32)> = loops.into_iter().collect();
    for (i, &(_, e1)) in list.iter().enumerate() {
        for &(h2, e2) in &list[i + 1..] {
            // h1 < h2: nested (e2 <= e1) or disjoint (h2 >= e1).
            if h2 < e1 && e2 > e1 {
                return None;
            }
        }
    }
    for ins in code {
        if let Some(t) = ins.branch_target() {
            for &(h, e) in &list {
                let inside = |a: u32| a >= h && a < e;
                if inside(t) && t != h && !inside(ins.pc) {
                    return None;
                }
            }
        }
    }
    Some(list)
}

/// Structured control flow (see the module documentation): the blocks of
/// [lo, hi) in address order, loops as structured loops.
fn emit_structured(tr: &mut T, code: &[Inst], starts: &BTreeSet<u32>, loops: &[(u32, u32)]) -> R<()> {
    let first = code[0].pc;
    let s0 = tr.u(first);
    ok(tr.b.store(tr.pc_var, s0, None, []))?;
    let starts_vec: Vec<u32> = starts.iter().copied().collect();
    emit_range(tr, code, &starts_vec, loops, 0, starts_vec.len(), None)?;
    emit_watchdog_record(tr)?;
    ok(tr.b.ret())?;
    Ok(())
}

/// Blocks `lo..hi` of `starts`; `inside`: the header of the loop being
/// emitted (its own range is not a nested loop).
fn emit_range(tr: &mut T, code: &[Inst], starts: &[u32], loops: &[(u32, u32)], lo: usize, hi: usize, inside: Option<u32>) -> R<()> {
    if tr.spec.dom_nest {
        return emit_nested(tr, code, starts, loops, lo, hi, inside);
    }
    let mut i = lo;
    while i < hi {
        let s = starts[i];
        if let Some(&(h, end)) = loops.iter().find(|&&(h, _)| h == s && Some(h) != inside) {
            let j = starts.iter().position(|&x| x >= end).unwrap_or(starts.len()).min(hi);
            emit_loop(tr, code, starts, loops, i, j, h)?;
            i = j;
            continue;
        }
        let end = starts.get(i + 1).copied().unwrap_or(u32::MAX);
        let pc = e(tr.b.load(tr.t.u32, None, tr.pc_var, None, []))?;
        let k = tr.u(s);
        let here = e(tr.b.i_equal(tr.t.bool_, None, pc, k))?;
        let (body, merge) = (tr.b.id(), tr.b.id());
        ok(tr.b.selection_merge(merge, spirv::SelectionControl::NONE))?;
        ok(tr.b.branch_conditional(here, body, merge, []))?;
        e(tr.b.begin_block(Some(body)))?;
        emit_block_body(tr, code, s, end)?;
        ok(tr.b.branch(merge))?;
        e(tr.b.begin_block(Some(merge)))?;
        i += 1;
    }
    Ok(())
}

/// The addresses block [s, e) may store as the next one, as
/// `emit_block_body` stores them: its branch's target and, for a
/// conditional branch, the next instruction's (PC_EXIT past the program);
/// PC_EXIT after s_endpgm; else the next block (PC_EXIT for the last).
fn block_succs(code: &[Inst], s: u32, e: u32) -> Vec<u32> {
    let first = code.partition_point(|i| i.pc < s);
    for (k, ins) in code.iter().enumerate().skip(first) {
        if ins.pc >= e {
            break;
        }
        if ins.is_endpgm() {
            return vec![PC_EXIT];
        }
        if let Some(t) = ins.branch_target() {
            let mut out = vec![t];
            if ins.name() != Some("s_branch") {
                out.push(code.get(k + 1).map_or(PC_EXIT, |n| n.pc));
            }
            return out;
        }
    }
    vec![if e == u32::MAX { PC_EXIT } else { e }]
}

/// A node of a range's forward graph (`emit_nested`): one block (`lo`), or
/// a nested loop over blocks `lo..hi` entered only at its header.
struct Node {
    lo: usize,
    hi: usize,
    start: u32,
    header: Option<u32>,
}

/// A range's forward graph and dominator tree (`emit_nested`).
struct Graph {
    nodes: Vec<Node>,
    /// Dominated nodes of each node in address order; `kids[n]`: the
    /// virtual root's.
    kids: Vec<Vec<usize>>,
    /// Forward edges to the range's nodes.
    succ: Vec<Vec<usize>>,
    /// The node may leave the pass otherwise: a back edge, a target outside
    /// the range, the end of the program, or (a loop) the loop watchdog.
    exits: Vec<bool>,
    /// Pre-order position in the dominator tree and the subtree's size.
    pre: Vec<usize>,
    size: Vec<usize>,
    /// Nodes in pre-order (a subtree is `order[pre[k]..pre[k] + size[k]]`).
    order: Vec<usize>,
}

/// Nesting depth of guarded regions beyond which a block's dominated
/// blocks follow its region instead of nesting in it (a long chain of
/// joins would nest as deep as it is long).
const MAX_NEST: u32 = 24;

/// Blocks `lo..hi` of `starts` (`Spec::dom_nest`), each emitted inside the
/// guarded region of its immediate dominator over the range's forward edges
/// (back edges end a pass; exits leave it): a block can run in a pass only
/// after its dominator ran in it, so its guard is tested only then. Address
/// order is a topological order of forward edges, and emitting siblings in
/// it, each with its dominated blocks, keeps every block after all its
/// predecessors (they are all dominated by its dominator). A block every
/// path from its dominator's region reaches (an if's join) needs no guard
/// (`emit_tree`). Blocks no forward edge reaches hang off a virtual root and
/// stay at the range's level.
fn emit_nested(tr: &mut T, code: &[Inst], starts: &[u32], loops: &[(u32, u32)], lo: usize, hi: usize, inside: Option<u32>) -> R<()> {
    let mut nodes: Vec<Node> = Vec::new();
    let mut i = lo;
    while i < hi {
        let s = starts[i];
        if let Some(&(h, end)) = loops.iter().find(|&&(h, _)| h == s && Some(h) != inside) {
            let j = starts.iter().position(|&x| x >= end).unwrap_or(starts.len()).min(hi);
            nodes.push(Node { lo: i, hi: j, start: s, header: Some(h) });
            i = j;
        } else {
            nodes.push(Node { lo: i, hi: i + 1, start: s, header: None });
            i += 1;
        }
    }
    let n = nodes.len();
    let at: HashMap<u32, usize> = nodes.iter().enumerate().map(|(k, x)| (x.start, k)).collect();
    let mut succ: Vec<Vec<usize>> = vec![Vec::new(); n];
    let mut exits = vec![false; n];
    let mut preds: Vec<Vec<usize>> = vec![Vec::new(); n];
    for (k, node) in nodes.iter().enumerate() {
        let span_end = starts.get(node.hi).copied().unwrap_or(u32::MAX);
        exits[k] = node.header.is_some();
        for b in node.lo..node.hi {
            let (s, e) = (starts[b], starts.get(b + 1).copied().unwrap_or(u32::MAX));
            for t in block_succs(code, s, e) {
                // A nested loop's own edges stay inside it.
                if node.header.is_some() && t >= node.start && t < span_end {
                    continue;
                }
                match at.get(&t) {
                    Some(&m) if m > k => {
                        if !succ[k].contains(&m) {
                            succ[k].push(m);
                            preds[m].push(k);
                        }
                    }
                    _ => exits[k] = true,
                }
            }
        }
    }
    // Immediate dominators (Cooper, Harvey and Kennedy over the topological
    // order; `n` is the virtual root).
    let root = n;
    let mut idom = vec![root; n];
    for k in 1..n {
        let mut d: Option<usize> = None;
        for &p in &preds[k] {
            d = Some(match d {
                None => p,
                Some(mut a) => {
                    let mut b = p;
                    while a != b {
                        if a == root || b == root {
                            a = root;
                            break;
                        }
                        if a > b {
                            a = idom[a];
                        } else {
                            b = idom[b];
                        }
                    }
                    a
                }
            });
        }
        idom[k] = d.unwrap_or(root);
    }
    let mut kids: Vec<Vec<usize>> = vec![Vec::new(); n + 1];
    for k in 0..n {
        kids[idom[k]].push(k);
    }
    // Pre-order numbering (iterative).
    let (mut pre, mut size, mut order) = (vec![0; n], vec![1; n], Vec::with_capacity(n));
    let mut stack: Vec<(usize, bool)> = kids[root].iter().rev().map(|&k| (k, false)).collect();
    while let Some((k, done)) = stack.pop() {
        if done {
            size[k] = 1 + kids[k].iter().map(|&c| size[c]).sum::<usize>();
            continue;
        }
        pre[k] = order.len();
        order.push(k);
        stack.push((k, true));
        for &c in kids[k].iter().rev() {
            stack.push((c, false));
        }
    }
    let g = Graph { nodes, kids, succ, exits, pre, size, order };
    let top = g.kids[root].clone();
    emit_tree(tr, code, starts, loops, &g, top, None, 0)
}

/// Nodes `list` (siblings in address order) of `emit_nested`, each with
/// the nodes it dominates: inside its guarded region, or after it (a loop,
/// or at `MAX_NEST`). With `parent` (the list is its dominated nodes),
/// a node whose every way in from the parent's region so far leads to it
/// (no exits, no loops, every edge of the parent and of the earlier
/// siblings' subtrees into those or to it) runs without a guard: the next
/// address is certainly its own there.
fn emit_tree(tr: &mut T, code: &[Inst], starts: &[u32], loops: &[(u32, u32)], g: &Graph, list: Vec<usize>, parent: Option<usize>, depth: u32) -> R<()> {
    // The parent's region so far: its members, their edges leaving it, and
    // whether one may exit.
    let mut members: crate::HashSet<usize> = crate::HashSet::default();
    let mut out: crate::HashSet<usize> = crate::HashSet::default();
    let mut bad = true;
    if let Some(p) = parent {
        members.insert(p);
        out.extend(g.succ[p].iter().copied());
        bad = g.exits[p];
    }
    let direct: crate::HashSet<usize> = list.iter().copied().collect();
    let mut queue: VecDeque<usize> = list.into();
    while let Some(k) = queue.pop_front() {
        let node = &g.nodes[k];
        let certain = parent.is_some() && direct.contains(&k) && !bad && out.iter().all(|&t| t == k);
        if parent.is_some() && direct.contains(&k) {
            // The parent's region grows by this node's subtree (emitted
            // before the next sibling, from the queue or nested).
            for &m in &g.order[g.pre[k]..g.pre[k] + g.size[k]] {
                members.insert(m);
                bad |= g.exits[m];
                out.extend(g.succ[m].iter().copied());
            }
            out.retain(|t| !members.contains(t));
        }
        if let Some(h) = node.header {
            emit_loop(tr, code, starts, loops, node.lo, node.hi, h)?;
            for &c in g.kids[k].iter().rev() {
                queue.push_front(c);
            }
            continue;
        }
        let s = node.start;
        let end = starts.get(node.lo + 1).copied().unwrap_or(u32::MAX);
        let nested = depth < MAX_NEST;
        if certain {
            emit_block_body(tr, code, s, end)?;
            if nested && !g.kids[k].is_empty() {
                emit_tree(tr, code, starts, loops, g, g.kids[k].clone(), Some(k), depth + 1)?;
            } else {
                for &c in g.kids[k].iter().rev() {
                    queue.push_front(c);
                }
            }
            continue;
        }
        let pc = e(tr.b.load(tr.t.u32, None, tr.pc_var, None, []))?;
        let ks = tr.u(s);
        let here = e(tr.b.i_equal(tr.t.bool_, None, pc, ks))?;
        let (body, merge) = (tr.b.id(), tr.b.id());
        ok(tr.b.selection_merge(merge, spirv::SelectionControl::NONE))?;
        ok(tr.b.branch_conditional(here, body, merge, []))?;
        e(tr.b.begin_block(Some(body)))?;
        emit_block_body(tr, code, s, end)?;
        if nested && !g.kids[k].is_empty() {
            emit_tree(tr, code, starts, loops, g, g.kids[k].clone(), Some(k), depth + 1)?;
        }
        ok(tr.b.branch(merge))?;
        e(tr.b.begin_block(Some(merge)))?;
        if !nested {
            for &c in g.kids[k].iter().rev() {
                queue.push_front(c);
            }
        }
    }
    Ok(())
}

/// The loop with header `h` over blocks `lo..hi`: repeats while the next
/// address is `h` (counted by the loop watchdog).
fn emit_loop(tr: &mut T, code: &[Inst], starts: &[u32], loops: &[(u32, u32)], lo: usize, hi: usize, h: u32) -> R<()> {
    let (head, body, cont, merge) = (tr.b.id(), tr.b.id(), tr.b.id(), tr.b.id());
    ok(tr.b.branch(head))?;
    e(tr.b.begin_block(Some(head)))?;
    ok(tr.b.loop_merge(merge, cont, spirv::LoopControl::NONE, []))?;
    ok(tr.b.branch(body))?;
    e(tr.b.begin_block(Some(body)))?;
    emit_range(tr, code, starts, loops, lo, hi, Some(h))?;
    ok(tr.b.branch(cont))?;
    e(tr.b.begin_block(Some(cont)))?;
    let bt = tr.t.bool_;
    let pc = e(tr.b.load(tr.t.u32, None, tr.pc_var, None, []))?;
    let kh = tr.u(h);
    let mut again = e(tr.b.i_equal(bt, None, pc, kh))?;
    if let Some(ctr) = tr.loop_ctr {
        let n = e(tr.b.load(tr.t.u32, None, ctr, None, []))?;
        let one = tr.u(1);
        let n = e(tr.b.i_add(tr.t.u32, None, n, one))?;
        ok(tr.b.store(ctr, n, None, []))?;
        let lim = tr.u(loop_limit());
        let over = e(tr.b.u_greater_than_equal(bt, None, n, lim))?;
        // Stopped: no block runs any more (the record follows at the end).
        let stop = e(tr.b.logical_and(bt, None, again, over))?;
        let ks = tr.u(PC_STOPPED);
        let npc = e(tr.b.select(tr.t.u32, None, stop, ks, pc))?;
        ok(tr.b.store(tr.pc_var, npc, None, []))?;
        let go = e(tr.b.logical_not(bt, None, over))?;
        again = e(tr.b.logical_and(bt, None, again, go))?;
    }
    ok(tr.b.branch_conditional(again, head, merge, []))?;
    e(tr.b.begin_block(Some(merge)))?;
    Ok(())
}

/// After the program: an invocation left with a block still to run was
/// stopped by the loop watchdog; it records that and the program address.
fn emit_watchdog_record(tr: &mut T) -> R<()> {
    let Some(wd) = tr.watchdog else { return Ok(()) };
    let ex = tr.u(PC_EXIT);
    let pc3 = e(tr.b.load(tr.t.u32, None, tr.pc_var, None, []))?;
    let stopped = e(tr.b.i_not_equal(tr.t.bool_, None, pc3, ex))?;
    let rec = tr.b.id();
    let end = tr.b.id();
    ok(tr.b.selection_merge(end, spirv::SelectionControl::NONE))?;
    ok(tr.b.branch_conditional(stopped, rec, end, []))?;
    e(tr.b.begin_block(Some(rec)))?;
    let z = tr.u(0);
    let p = e(tr.b.access_chain(tr.t.sb_u32, None, wd, [z, z]))?;
    let scope = tr.u(spirv::Scope::Device as u32);
    let one = tr.u(1);
    e(tr.b.atomic_i_add(tr.t.u32, None, p, scope, z, one))?;
    let (bl, bh) = tr.code_base()?;
    for (k, c) in [(1, bl), (2, bh)] {
        let i = tr.u(k);
        let p = e(tr.b.access_chain(tr.t.sb_u32, None, wd, [z, i]))?;
        ok(tr.b.store(p, c, None, []))?;
    }
    // Its SGPRs as they were when it stopped (a loop's bound read from
    // memory, the user data's pointers).
    for k in 0..32u32 {
        let v = tr.sreg(k)?;
        let i = tr.u(WATCHDOG_SGPRS_AT + k);
        let p = e(tr.b.access_chain(tr.t.sb_u32, None, wd, [z, i]))?;
        ok(tr.b.store(p, v, None, []))?;
    }
    ok(tr.b.branch(end))?;
    e(tr.b.begin_block(Some(end)))?;
    Ok(())
}

/// The dispatcher: a loop over a switch on the next block's address. With
/// sync blocks it runs in rounds (an outer loop): an invocation reaching a
/// sync block after other blocks of the round leaves the inner loop, whose
/// merge reconverges each subgroup, and the next round starts at that block
/// for every invocation, so its barriers are the same dynamic instances.
fn emit_dispatch_loop(tr: &mut T, code: &[Inst], starts: &BTreeSet<u32>) -> R<()> {
    let first = code[0].pc;
    let s0 = tr.u(first);
    ok(tr.b.store(tr.pc_var, s0, None, []))?;
    let rounds = match tr.sync_steps {
        Some(steps) if !tr.sync_starts.is_empty() => {
            let (oh, ob, oc, ox) = (tr.b.id(), tr.b.id(), tr.b.id(), tr.b.id());
            ok(tr.b.branch(oh))?;
            e(tr.b.begin_block(Some(oh)))?;
            ok(tr.b.loop_merge(ox, oc, spirv::LoopControl::NONE, []))?;
            ok(tr.b.branch(ob))?;
            e(tr.b.begin_block(Some(ob)))?;
            let z = tr.u(0);
            ok(tr.b.store(steps, z, None, []))?;
            Some((steps, oh, oc, ox))
        }
        _ => None,
    };
    let header = tr.b.id();
    let dispatch = tr.b.id();
    let sw_merge = tr.b.id();
    let cont = tr.b.id();
    let exit = tr.b.id();
    ok(tr.b.branch(header))?;
    e(tr.b.begin_block(Some(header)))?;
    ok(tr.b.loop_merge(exit, cont, spirv::LoopControl::NONE, []))?;
    ok(tr.b.branch(dispatch))?;
    e(tr.b.begin_block(Some(dispatch)))?;
    let pc = e(tr.b.load(tr.t.u32, None, tr.pc_var, None, []))?;
    if let Some((steps, ..)) = rounds {
        let bt = tr.t.bool_;
        let n = e(tr.b.load(tr.t.u32, None, steps, None, []))?;
        let mut at_sync = tr.bool_c(false);
        for s in tr.sync_starts.clone() {
            let k = tr.u(s);
            let eq = e(tr.b.i_equal(bt, None, pc, k))?;
            at_sync = e(tr.b.logical_or(bt, None, at_sync, eq))?;
        }
        let z = tr.u(0);
        let started = e(tr.b.i_not_equal(bt, None, n, z))?;
        let leave = e(tr.b.logical_and(bt, None, at_sync, started))?;
        let (brk, go) = (tr.b.id(), tr.b.id());
        ok(tr.b.selection_merge(go, spirv::SelectionControl::NONE))?;
        ok(tr.b.branch_conditional(leave, brk, go, []))?;
        e(tr.b.begin_block(Some(brk)))?;
        ok(tr.b.branch(exit))?;
        e(tr.b.begin_block(Some(go)))?;
        let one = tr.u(1);
        let n1 = e(tr.b.i_add(tr.t.u32, None, n, one))?;
        ok(tr.b.store(steps, n1, None, []))?;
    }
    let block_ids: Vec<(u32, Word)> = starts.iter().map(|&s| (s, tr.b.id())).collect();
    ok(tr.b.selection_merge(sw_merge, spirv::SelectionControl::NONE))?;
    let targets: Vec<(Operand, Word)> = block_ids.iter().map(|&(s, id)| (Operand::LiteralBit32(s), id)).collect();
    ok(tr.b.switch(pc, sw_merge, targets))?;
    let starts_vec: Vec<u32> = starts.iter().copied().collect();
    for (bi, &(start, id)) in block_ids.iter().enumerate() {
        e(tr.b.begin_block(Some(id)))?;
        let end = starts_vec.get(bi + 1).copied().unwrap_or(u32::MAX);
        emit_block_body(tr, code, start, end)?;
        ok(tr.b.branch(sw_merge))?;
    }
    e(tr.b.begin_block(Some(sw_merge)))?;
    ok(tr.b.branch(cont))?;
    e(tr.b.begin_block(Some(cont)))?;
    let pc2 = e(tr.b.load(tr.t.u32, None, tr.pc_var, None, []))?;
    let ex = tr.u(PC_EXIT);
    let mut done = e(tr.b.i_equal(tr.t.bool_, None, pc2, ex))?;
    if let Some(ctr) = tr.loop_ctr {
        let n = e(tr.b.load(tr.t.u32, None, ctr, None, []))?;
        let one = tr.u(1);
        let n = e(tr.b.i_add(tr.t.u32, None, n, one))?;
        ok(tr.b.store(ctr, n, None, []))?;
        let lim = tr.u(loop_limit());
        let over = e(tr.b.u_greater_than_equal(tr.t.bool_, None, n, lim))?;
        done = e(tr.b.logical_or(tr.t.bool_, None, done, over))?;
    }
    ok(tr.b.branch_conditional(done, exit, header, []))?;
    e(tr.b.begin_block(Some(exit)))?;
    if let Some((_, oh, oc, ox)) = rounds {
        // Next round unless finished (or stopped by the watchdog).
        ok(tr.b.branch(oc))?;
        e(tr.b.begin_block(Some(oc)))?;
        let pc3 = e(tr.b.load(tr.t.u32, None, tr.pc_var, None, []))?;
        let mut fin = e(tr.b.i_equal(tr.t.bool_, None, pc3, ex))?;
        if let Some(ctr) = tr.loop_ctr {
            let n = e(tr.b.load(tr.t.u32, None, ctr, None, []))?;
            let lim = tr.u(loop_limit());
            let over = e(tr.b.u_greater_than_equal(tr.t.bool_, None, n, lim))?;
            fin = e(tr.b.logical_or(tr.t.bool_, None, fin, over))?;
        }
        ok(tr.b.branch_conditional(fin, ox, oh, []))?;
        e(tr.b.begin_block(Some(ox)))?;
    }
    emit_watchdog_record(tr)?;
    ok(tr.b.ret())?;
    Ok(())
}

fn translate_inst(tr: &mut T, ins: &Inst) -> R<()> {
    tr.cur_pc = ins.pc;
    translate_one(tr, ins)?;
    if let (Some((pc, reg)), Some(var)) = (tr.spec.probe, tr.probe_var) {
        if ins.pc == pc {
            let n = reg & 0xFFF;
            let v = if reg & PROBE_SGPR != 0 { tr.sreg(n)? } else { tr.vreg(n)? };
            let v = if reg & (PROBE_UINT | PROBE_SINT) != 0 {
                let (u, f) = (tr.t.u32, tr.t.f32);
                let x = if reg & PROBE_SINT != 0 {
                    let i = e(tr.b.bitcast(tr.t.i32, None, v))?;
                    e(tr.b.convert_s_to_f(f, None, i))?
                } else {
                    e(tr.b.convert_u_to_f(f, None, v))?
                };
                e(tr.b.bitcast(u, None, x))?
            } else {
                v
            };
            ok(tr.b.store(var, v, None, []))?;
        }
    }
    if let Some((pc, g)) = tr.spec.dump {
        if ins.pc == pc {
            if g[2] & DUMP_POS != 0 && ins.enc == Enc::Exp {
                emit_pos_dump(tr, ins, g)?;
            } else {
                emit_dump(tr, g)?;
            }
        }
    }
    Ok(())
}

/// Positions-only vertex dump at a position export (see `Spec::dump`): the
/// four exported registers of each vertex of the selected instances.
fn emit_pos_dump(tr: &mut T, ins: &Inst, g: [u32; 3]) -> R<()> {
    let u = tr.t.u32;
    let bt = tr.t.bool_;
    let (Some(buf), Some((vi, ii))) = (tr.dump_buf, tr.vertex_ids) else { return Ok(()) };
    let count = (g[2] & !DUMP_POS).clamp(1, DUMP_DWORDS / 4 / DUMP_POS_VERTICES);
    let first = tr.u(g[0]);
    let v = e(tr.b.i_sub(u, None, vi, first))?;
    let pn = tr.u(DUMP_POS_VERTICES);
    let inside = e(tr.b.u_less_than(bt, None, v, pn))?;
    let inst = tr.u(g[1]);
    let k = e(tr.b.i_sub(u, None, ii, inst))?;
    let cn = tr.u(count);
    let ours = e(tr.b.u_less_than(bt, None, k, cn))?;
    let cond = e(tr.b.logical_and(bt, None, inside, ours))?;
    tr.guarded(cond, &mut |tr| {
        let kp = e(tr.b.i_mul(u, None, k, pn))?;
        let slot = e(tr.b.i_add(u, None, kp, v))?;
        let four = tr.u(4);
        let base = e(tr.b.i_mul(u, None, slot, four))?;
        let z = tr.u(0);
        for c in 0..4u32 {
            let val = tr.vreg(ins.exp_src(c as usize) as u32)?;
            let kk = tr.u(c);
            let i = e(tr.b.i_add(u, None, base, kk))?;
            let p = e(tr.b.access_chain(tr.t.sb_u32, None, buf, [z, i]))?;
            ok(tr.b.store(p, val, None, []))?;
        }
        Ok(())
    })
}

/// Register dump of the invocations of workgroup `g`, or of a range of
/// vertices (see `Spec::dump`).
fn emit_dump(tr: &mut T, g: [u32; 3]) -> R<()> {
    let u = tr.t.u32;
    let bt = tr.t.bool_;
    let (buf, li, cond) = match (tr.dump_buf, tr.wg_id, tr.local_index, tr.vertex_ids) {
        (Some(buf), Some(wg), Some(li), _) => {
            let mut cond = tr.bool_c(true);
            for (k, gk) in g.iter().enumerate() {
                let c = e(tr.b.composite_extract(u, None, wg, [k as u32]))?;
                let kk = tr.u(*gk);
                let eq = e(tr.b.i_equal(bt, None, c, kk))?;
                cond = e(tr.b.logical_and(bt, None, cond, eq))?;
            }
            (buf, li, cond)
        }
        (Some(buf), _, _, Some((vi, ii))) => {
            let count = g[2].clamp(1, DUMP_DWORDS / DUMP_STRIDE);
            let per = DUMP_DWORDS / DUMP_STRIDE / count;
            let first = tr.u(g[0]);
            let v = e(tr.b.i_sub(u, None, vi, first))?;
            let pn = tr.u(per);
            let inside = e(tr.b.u_less_than(bt, None, v, pn))?;
            let inst = tr.u(g[1]);
            let k = e(tr.b.i_sub(u, None, ii, inst))?;
            let cn = tr.u(count);
            let ours = e(tr.b.u_less_than(bt, None, k, cn))?;
            let cond = e(tr.b.logical_and(bt, None, inside, ours))?;
            let kp = e(tr.b.i_mul(u, None, k, pn))?;
            let slot = e(tr.b.i_add(u, None, kp, v))?;
            (buf, slot, cond)
        }
        _ => return Ok(()),
    };
    tr.guarded(cond, &mut |tr| {
        let stride = tr.u(DUMP_STRIDE);
        let base = e(tr.b.i_mul(u, None, li, stride))?;
        let z = tr.u(0);
        let put = |tr: &mut T, k: u32, v: Word| -> R<()> {
            let kk = tr.u(k);
            let i = e(tr.b.i_add(u, None, base, kk))?;
            let p = e(tr.b.access_chain(tr.t.sb_u32, None, buf, [z, i]))?;
            ok(tr.b.store(p, v, None, []))
        };
        for k in 0..256 {
            let v = tr.vreg(k)?;
            put(tr, k, v)?;
        }
        for k in 0..128 {
            let v = tr.sreg(k)?;
            put(tr, 256 + k, v)?;
        }
        let s = tr.get_scc()?;
        let s = tr.bool_to_u(s)?;
        put(tr, 384, s)?;
        let k385 = tr.u(385);
        let i = e(tr.b.i_add(u, None, base, k385))?;
        let p = e(tr.b.access_chain(tr.t.sb_u32, None, buf, [z, i]))?;
        let n = e(tr.b.load(u, None, p, None, []))?;
        let one = tr.u(1);
        let n1 = e(tr.b.i_add(u, None, n, one))?;
        ok(tr.b.store(p, n1, None, []))
    })
}

fn translate_one(tr: &mut T, ins: &Inst) -> R<()> {
    match ins.enc {
        Enc::Sop2 | Enc::Sop1 | Enc::Sopc | Enc::Sopk | Enc::Sopp => alu::scalar(tr, ins),
        Enc::Vop1 | Enc::Vop2 | Enc::Vopc | Enc::Vop3 | Enc::Vop3p => alu::vector(tr, ins),
        Enc::Smem => mem::smem(tr, ins),
        Enc::Vintrp => mem::interp(tr, ins),
        Enc::Mubuf | Enc::Mtbuf => mem::buffer(tr, ins),
        Enc::Mimg => mem::image(tr, ins),
        Enc::Exp => mem::export(tr, ins),
        Enc::Ds => mem::ds(tr, ins),
        Enc::Global => mem::global(tr, ins),
        Enc::Flat | Enc::Scratch => {
            tr.warn(format!("{:#x}: {} unsupported", ins.pc, ins.name().unwrap_or("flat/scratch")));
            Ok(())
        }
    }
}

/// A spec as one line of words and back (`to_words`, `from_words`): every field in order, the
/// flags as one mask. The runtime's learned store keeps the exact key spec of each compute
/// pipeline a session made this way, so that its compute step can make all of them again after a
/// driver update or a refused store (2026-11-21: the learned specs, five fields, could not name
/// batched programs' address variants, LOD-statistics variants or every buffer-format mix, and a
/// store the compute step made again left ~330 compute pipelines to first sessions). Both take
/// every field by name: a field added to `Spec` fails to compile here until it is written too.
impl Spec {
    pub fn to_words(&self) -> String {
        let Spec {
            stage,
            user_sgprs,
            user_base,
            code_base,
            wave64,
            ps_input_addr,
            ps_inputs,
            barycentric,
            bary_gs,
            mrt,
            cs_local_size,
            cs_rsrc2,
            resources,
            binding_base,
            probe,
            dump,
            force_dispatcher,
            dispatcher_for_loops,
            dom_nest,
            reg_vars,
            lane_guards,
            mip_views,
            lod_stats,
            lod_lean,
            early_tests,
            early_ok,
            per_lane,
            lane_votes,
            lockstep_off,
            sg_waves,
            sg_appends,
            known_active,
            hw_bounds,
            base_instance_sgpr,
            sgpr_pc,
            addresses,
            batched,
            write_marks,
            marks_untouched,
            touch_test,
            mark_twice,
            touch_twice,
            part_twice,
            dep_loads,
            marks_part,
            marks_held,
            sgpr_ubo,
            gpu_loads,
            gpu_res,
            pt_linear,
            outlier,
        } = self;
        let flags = [
            *wave64,
            *barycentric,
            *bary_gs,
            *force_dispatcher,
            *dispatcher_for_loops,
            *dom_nest,
            *reg_vars,
            *lane_guards,
            *mip_views,
            *lod_stats,
            *lod_lean,
            *early_tests,
            *early_ok,
            *per_lane,
            *lane_votes,
            *lockstep_off,
            *sg_waves,
            *sg_appends,
            *known_active,
            *hw_bounds,
            *addresses,
            *batched,
            *write_marks,
            *marks_untouched,
            *touch_test,
            *mark_twice,
            *touch_twice,
            *part_twice,
            *dep_loads,
            *marks_part,
            *marks_held,
            *pt_linear,
        ];
        let mask: u64 = flags.iter().enumerate().fold(0, |m, (i, &f)| m | (f as u64) << i);
        let opt = |o: Option<String>| o.unwrap_or_else(|| "-".into());
        let mut w: Vec<String> = vec![
            "s1".into(),
            match stage {
                Stage::Vertex => "v",
                Stage::Fragment => "f",
                Stage::Compute => "c",
            }
            .into(),
            user_sgprs.to_string(),
            user_base.to_string(),
            format!("{code_base:x}"),
            ps_input_addr.to_string(),
            format!("{mask:x}"),
            format!("{},{},{}", cs_local_size[0], cs_local_size[1], cs_local_size[2]),
            cs_rsrc2.to_string(),
            binding_base.to_string(),
            opt(probe.map(|(a, b)| format!("{a},{b}"))),
            opt(dump.map(|(a, b)| format!("{a},{},{},{}", b[0], b[1], b[2]))),
            opt(base_instance_sgpr.map(|v| v.to_string())),
            opt(sgpr_pc.map(|v| v.to_string())),
            sgpr_ubo.to_string(),
            format!("{gpu_loads:x}"),
            format!("{gpu_res:x}"),
            opt(outlier.map(|(a, b)| format!("{a},{b:x}"))),
        ];
        w.push(mrt.iter().map(|m| format!("{}.{}", m.export_format, m.class)).collect::<Vec<_>>().join(","));
        w.push(if ps_inputs.is_empty() {
            "-".into()
        } else {
            ps_inputs.iter().map(|i| format!("{}.{}.{}", i.param, i.flat as u8, i.default.map_or("-".into(), |d| d.to_string()))).collect::<Vec<_>>().join(",")
        });
        let dim = |d: ImageDim| match d {
            ImageDim::D1 => 0,
            ImageDim::D2 => 1,
            ImageDim::D3 => 2,
            ImageDim::Cube => 3,
            ImageDim::D1Array => 4,
            ImageDim::D2Array => 5,
            ImageDim::D2Msaa => 6,
            ImageDim::D2MsaaArray => 7,
            ImageDim::CubeArray => 8,
            ImageDim::D3OrD2 => 9,
        };
        let class = |c: NumClass| match c {
            NumClass::Float => 0,
            NumClass::Sint => 1,
            NumClass::Uint => 2,
        };
        w.push(if resources.is_empty() {
            "-".into()
        } else {
            resources
                .iter()
                .map(|r| match r {
                    ResSpec::Buffer { format, dst_sel } => format!("b{format}.{dst_sel}"),
                    ResSpec::Image { dim: d, class: c, depth } => format!("i{}.{}.{}", dim(*d), class(*c), *depth as u8),
                    ResSpec::Sampler => "s".into(),
                    ResSpec::Global => "g".into(),
                })
                .collect::<Vec<_>>()
                .join(",")
        });
        w.join(" ")
    }

    pub fn from_words(line: &str) -> Option<Spec> {
        let mut t = line.split_whitespace();
        if t.next()? != "s1" {
            return None;
        }
        let stage = match t.next()? {
            "v" => Stage::Vertex,
            "f" => Stage::Fragment,
            "c" => Stage::Compute,
            _ => return None,
        };
        let user_sgprs: u16 = t.next()?.parse().ok()?;
        let user_base: u16 = t.next()?.parse().ok()?;
        let code_base = u64::from_str_radix(t.next()?, 16).ok()?;
        let ps_input_addr: u32 = t.next()?.parse().ok()?;
        let mask = u64::from_str_radix(t.next()?, 16).ok()?;
        let f = |i: u32| mask >> i & 1 == 1;
        let ls: Vec<u32> = t.next()?.split(',').map(|v| v.parse().ok()).collect::<Option<Vec<u32>>>()?;
        if ls.len() != 3 {
            return None;
        }
        let cs_rsrc2: u32 = t.next()?.parse().ok()?;
        let binding_base: u32 = t.next()?.parse().ok()?;
        let opt = |w: &str| if w == "-" { None } else { Some(w.to_string()) };
        let nums = |w: &str| w.split(',').map(|v| v.parse::<u32>().ok()).collect::<Option<Vec<u32>>>();
        let probe = match opt(t.next()?) {
            None => None,
            Some(v) => {
                let n = nums(&v)?;
                if n.len() != 2 {
                    return None;
                }
                Some((n[0], n[1]))
            }
        };
        let dump = match opt(t.next()?) {
            None => None,
            Some(v) => {
                let n = nums(&v)?;
                if n.len() != 4 {
                    return None;
                }
                Some((n[0], [n[1], n[2], n[3]]))
            }
        };
        let base_instance_sgpr = match opt(t.next()?) {
            None => None,
            Some(v) => Some(v.parse::<u16>().ok()?),
        };
        let sgpr_pc = match opt(t.next()?) {
            None => None,
            Some(v) => Some(v.parse::<u32>().ok()?),
        };
        let sgpr_ubo: u8 = t.next()?.parse().ok()?;
        let gpu_loads = u64::from_str_radix(t.next()?, 16).ok()?;
        let gpu_res = u64::from_str_radix(t.next()?, 16).ok()?;
        let outlier = match opt(t.next()?) {
            None => None,
            Some(v) => {
                let (a, b) = v.split_once(',')?;
                Some((a.parse::<u32>().ok()?, u64::from_str_radix(b, 16).ok()?))
            }
        };
        let mrt_w: Vec<&str> = t.next()?.split(',').collect();
        if mrt_w.len() != 8 {
            return None;
        }
        let mut mrt = [MrtSpec::default(); 8];
        for (m, w) in mrt.iter_mut().zip(mrt_w) {
            let (a, b) = w.split_once('.')?;
            *m = MrtSpec { export_format: a.parse().ok()?, class: b.parse().ok()? };
        }
        let ps_w = t.next()?;
        let ps_inputs = if ps_w == "-" {
            Vec::new()
        } else {
            ps_w.split(',')
                .map(|w| {
                    let mut p = w.split('.');
                    let param: u8 = p.next()?.parse().ok()?;
                    let flat = p.next()? == "1";
                    let d = p.next()?;
                    let default = if d == "-" { None } else { Some(d.parse::<u8>().ok()?) };
                    Some(PsInput { param, flat, default })
                })
                .collect::<Option<Vec<_>>>()?
        };
        let res_w = t.next()?;
        let dim = |v: u32| {
            Some(match v {
                0 => ImageDim::D1,
                1 => ImageDim::D2,
                2 => ImageDim::D3,
                3 => ImageDim::Cube,
                4 => ImageDim::D1Array,
                5 => ImageDim::D2Array,
                6 => ImageDim::D2Msaa,
                7 => ImageDim::D2MsaaArray,
                8 => ImageDim::CubeArray,
                9 => ImageDim::D3OrD2,
                _ => return None,
            })
        };
        let class = |v: u32| {
            Some(match v {
                0 => NumClass::Float,
                1 => NumClass::Sint,
                2 => NumClass::Uint,
                _ => return None,
            })
        };
        let resources = if res_w == "-" {
            Vec::new()
        } else {
            res_w
                .split(',')
                .map(|w| {
                    if w == "s" {
                        return Some(ResSpec::Sampler);
                    }
                    if w == "g" {
                        return Some(ResSpec::Global);
                    }
                    if let Some(b) = w.strip_prefix('b') {
                        let (a, d) = b.split_once('.')?;
                        return Some(ResSpec::Buffer { format: a.parse().ok()?, dst_sel: d.parse().ok()? });
                    }
                    let i = w.strip_prefix('i')?;
                    let n = i.split('.').map(|v| v.parse::<u32>().ok()).collect::<Option<Vec<u32>>>()?;
                    if n.len() != 3 {
                        return None;
                    }
                    Some(ResSpec::Image { dim: dim(n[0])?, class: class(n[1])?, depth: n[2] == 1 })
                })
                .collect::<Option<Vec<_>>>()?
        };
        if t.next().is_some() {
            return None;
        }
        Some(Spec {
            stage,
            user_sgprs,
            user_base,
            code_base,
            wave64: f(0),
            ps_input_addr,
            ps_inputs,
            barycentric: f(1),
            bary_gs: f(2),
            mrt,
            cs_local_size: [ls[0], ls[1], ls[2]],
            cs_rsrc2,
            resources,
            binding_base,
            probe,
            dump,
            force_dispatcher: f(3),
            dispatcher_for_loops: f(4),
            dom_nest: f(5),
            reg_vars: f(6),
            lane_guards: f(7),
            mip_views: f(8),
            lod_stats: f(9),
            lod_lean: f(10),
            early_tests: f(11),
            early_ok: f(12),
            per_lane: f(13),
            lane_votes: f(14),
            lockstep_off: f(15),
            sg_waves: f(16),
            sg_appends: f(17),
            known_active: f(18),
            hw_bounds: f(19),
            base_instance_sgpr,
            sgpr_pc,
            addresses: f(20),
            batched: f(21),
            write_marks: f(22),
            marks_untouched: f(23),
            touch_test: f(24),
            mark_twice: f(25),
            touch_twice: f(26),
            part_twice: f(27),
            dep_loads: f(28),
            marks_part: f(29),
            marks_held: f(30),
            sgpr_ubo,
            gpu_loads,
            gpu_res,
            pt_linear: f(31),
            outlier,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// `Spec::to_words` and `from_words`: every field away from its default comes back.
    #[test]
    fn spec_words_round_trip() {
        let mut s = cs_spec([16, 4, 2]);
        let mut flip = 0u32;
        let mut b = || {
            flip += 1;
            flip % 3 != 0
        };
        s.stage = Stage::Compute;
        s.user_sgprs = 12;
        s.user_base = 3;
        s.code_base = 0x10_0889_a200;
        s.wave64 = b();
        s.ps_input_addr = 0x7f;
        s.ps_inputs = vec![PsInput { param: 3, flat: true, default: None }, PsInput { param: 7, flat: false, default: Some(2) }];
        s.barycentric = b();
        s.bary_gs = b();
        s.mrt[0] = MrtSpec { export_format: 4, class: 2 };
        s.mrt[7] = MrtSpec { export_format: 9, class: 1 };
        s.cs_rsrc2 = 0x8984;
        s.resources = vec![
            ResSpec::Buffer { format: 20, dst_sel: 4 },
            ResSpec::Image { dim: ImageDim::D2Array, class: NumClass::Uint, depth: true },
            ResSpec::Image { dim: ImageDim::D3OrD2, class: NumClass::Sint, depth: false },
            ResSpec::Sampler,
            ResSpec::Global,
        ];
        s.binding_base = 5;
        s.probe = Some((3, 9));
        s.dump = Some((1, [2, 3, 4]));
        s.force_dispatcher = b();
        s.dispatcher_for_loops = b();
        s.dom_nest = b();
        s.reg_vars = b();
        s.lane_guards = b();
        s.mip_views = b();
        s.lod_stats = b();
        s.lod_lean = b();
        s.early_tests = b();
        s.early_ok = b();
        s.per_lane = b();
        s.lane_votes = b();
        s.lockstep_off = b();
        s.sg_waves = b();
        s.sg_appends = b();
        s.known_active = b();
        s.hw_bounds = b();
        s.base_instance_sgpr = Some(6);
        s.sgpr_pc = Some(0x140);
        s.addresses = b();
        s.batched = b();
        s.write_marks = b();
        s.marks_untouched = b();
        s.touch_test = b();
        s.mark_twice = b();
        s.touch_twice = b();
        s.part_twice = b();
        s.dep_loads = b();
        s.marks_part = b();
        s.marks_held = b();
        s.sgpr_ubo = 7;
        s.gpu_loads = 0xdead_beef_0000_1234;
        s.gpu_res = 0x55;
        s.pt_linear = b();
        s.outlier = Some((11, 0xabc));
        let w = s.to_words();
        assert_eq!(Spec::from_words(&w).as_ref(), Some(&s), "{w}");
        // The defaults too, and every flag on in turn (the mask's bits stay apart).
        let d = cs_spec([8, 8, 1]);
        assert_eq!(Spec::from_words(&d.to_words()).as_ref(), Some(&d));
        assert!(Spec::from_words("s1 c 2").is_none());
        assert!(Spec::from_words(&format!("{w} extra")).is_none());
    }

    /// V_MED3_F32 with the fog's operands, med3(71.0, v0, 0) (assembled here: VOP3 0x157, a
    /// literal, v0, the inline 0): translated with the hardware's NaN rule, bit tests and a
    /// select, no `OpIsNan`. `RDNA_GS_DUMP=<dir>` writes the module there for spirv-val.
    #[test]
    fn med3_takes_the_hardware_nan_rule() {
        let words = [0xd557_0000, 0x0202_00ff, 0x428e_0000, 0xbf81_0000];
        let code = crate::decode_program(&words).unwrap();
        let spec = cs_spec([64, 1, 1]);
        let plan = crate::analysis::analyze(&code, 0, 0);
        let out = translate(&code, &plan, &spec).unwrap();
        assert!(out.warnings.is_empty(), "{:?}", out.warnings);
        let mut ops = Vec::new();
        let mut i = 5;
        while i < out.spirv.len() {
            ops.push(out.spirv[i] & 0xFFFF);
            i += ((out.spirv[i] >> 16) as usize).max(1);
        }
        let count = |op: u32| ops.iter().filter(|&&o| o == op).count();
        assert_eq!(count(156), 0, "OpIsNan");
        assert!(count(172) >= 3, "OpUGreaterThan: the three NaN tests");
        assert!(count(169) >= 1, "OpSelect");
        if let Some(dir) = std::env::var_os("RDNA_GS_DUMP") {
            let dir = std::path::PathBuf::from(dir);
            std::fs::write(dir.join("med3.spv"), out.spirv.iter().flat_map(|x| x.to_le_bytes()).collect::<Vec<u8>>()).expect("written");
        }
    }

    fn cs_spec(local: [u32; 3]) -> Spec {
        Spec {
            stage: Stage::Compute,
            user_sgprs: 2,
            user_base: 0,
            code_base: 0x1000,
            wave64: true,
            ps_input_addr: 0,
            ps_inputs: Vec::new(),
            barycentric: false,
            bary_gs: false,
            mrt: [MrtSpec::default(); 8],
            cs_local_size: local,
            cs_rsrc2: 0,
            resources: Vec::new(),
            binding_base: 0,
            probe: None,
            dump: None,
            force_dispatcher: false,
            dispatcher_for_loops: false,
            dom_nest: false,
            reg_vars: true,
            lane_guards: true,
            mip_views: true,
            lod_stats: false,
            lod_lean: false,
            early_tests: false,
            early_ok: false,
            per_lane: false,
            lane_votes: false,
            lockstep_off: false,
            sg_waves: false,
            sg_appends: false,
            known_active: false,
            hw_bounds: false,
            base_instance_sgpr: None,
            sgpr_pc: None,
            addresses: false,
            batched: false,
            write_marks: false,
            marks_untouched: false,
            touch_test: false,
            mark_twice: false,
            touch_twice: false,
            part_twice: false,
            dep_loads: false,
            marks_part: false,
            marks_held: false,
            sgpr_ubo: 0,
            gpu_loads: 0,
            gpu_res: 0,
            pt_linear: false,
            outlier: None,
        }
    }

    /// The tile classification's append kernel (cs 0x10088ab000 of PPSA01341
    /// 01.007.000; one invocation per tile, 8x8 per wave): lane 0 of each
    /// wave reserves the slots counted with v_bcnt of the class masks.
    #[test]
    fn lane0_election_of_mask_counted_append() {
        let Some(words) = crate::title_words("append_kernel") else { return };
        let code = crate::decode_program(&words).unwrap();
        let e = lane0_elections(&code, &cs_spec([8, 8, 1]));
        // v_cmp_lg_u32 vcc, 0, v0 (v0 = ly * 8 + lx): operand 1 is the lane.
        assert_eq!(e.into_iter().collect::<Vec<_>>(), vec![(0x9c, 1)]);
        // Not for groups of several waves.
        assert!(lane0_elections(&code, &cs_spec([16, 8, 1])).is_empty());
    }

    /// The append kernel above done per subgroup (`Spec::sg_appends`): its
    /// two class ballots (ORed) only counted and ranked, lane 0 elected by
    /// the lane-index compare (left exact), one EXEC write before the
    /// atomic, one read of its result; translated without wave-mask mode.
    #[test]
    fn append_per_subgroup() {
        let Some(words) = crate::title_words("append_kernel") else { return };
        let code = crate::decode_program(&words).unwrap();
        let mut spec = cs_spec([8, 8, 1]);
        assert!(has_append(&code));
        let a = subgroup_appends(&code, &spec).unwrap();
        let sorted = |s: &crate::HashSet<u32>| {
            let mut v: Vec<u32> = s.iter().copied().collect();
            v.sort();
            v
        };
        assert_eq!(sorted(&a.ballots), vec![0xa0, 0xa8]);
        assert_eq!(sorted(&a.ranks), vec![0x118, 0x120]);
        assert_eq!(sorted(&a.lane0_compares), vec![0x9c]);
        assert!(a.lane0.is_empty());
        assert_eq!(sorted(&a.leaders), vec![0xc4]);
        assert_eq!(sorted(&a.results), vec![0xf8]);
        let plan = crate::analysis::analyze(&code, 2, 0);
        spec.resources = plan.resources.iter().map(|_| ResSpec::Buffer { format: 0, dst_sel: 0xFAC }).collect();
        let before = translate(&code, &plan, &spec).unwrap();
        assert!(before.wave_masks && before.sg_appends == 0);
        spec.sg_appends = true;
        let after = translate(&code, &plan, &spec).unwrap();
        assert!(!after.wave_masks && !after.lockstep && after.sg_appends == 1);
        // Not for groups of several waves: then the compare elects no lane.
        assert!(subgroup_appends(&code, &cs_spec([16, 8, 1])).is_err());
    }

    /// Subgroup waves (`Spec::sg_waves`): the append kernel above in
    /// wave-mask mode per subgroup, its masks completed by subgroup ballots:
    /// no workgroup memory, no barrier.
    #[test]
    fn subgroup_waves() {
        let Some(words) = crate::title_words("append_kernel") else { return };
        let code = crate::decode_program(&words).unwrap();
        let mut spec = cs_spec([8, 8, 1]);
        let plan = crate::analysis::analyze(&code, 2, 0);
        spec.resources = plan.resources.iter().map(|_| ResSpec::Buffer { format: 0, dst_sel: 0xFAC }).collect();
        let ops = |w: &[u32]| {
            let mut v = Vec::new();
            let mut i = 5;
            while i < w.len() {
                v.push(w[i] & 0xFFFF);
                i += (w[i] >> 16).max(1) as usize;
            }
            v
        };
        let barrier = rspirv::spirv::Op::ControlBarrier as u32;
        let before = translate(&code, &plan, &spec).unwrap();
        assert!(before.wave_masks && !before.sg_waves);
        assert!(ops(&before.spirv).contains(&barrier));
        spec.sg_waves = true;
        let after = translate(&code, &plan, &spec).unwrap();
        assert!(after.wave_masks && after.sg_waves, "{:?}", after.warnings);
        assert!(!after.warnings.iter().any(|w| w.contains("subgroup") || w.contains("wave64")), "{:?}", after.warnings);
        assert!(!ops(&after.spirv).contains(&barrier));
        // Not for groups of several waves (no wave-mask mode there).
        let mut spec = cs_spec([16, 8, 1]);
        spec.resources = plan.resources.iter().map(|_| ResSpec::Buffer { format: 0, dst_sel: 0xFAC }).collect();
        spec.sg_waves = true;
        let several = translate(&code, &plan, &spec).unwrap();
        assert!(!several.sg_waves && several.warnings.iter().any(|w| w.contains("subgroup waves left out")));
    }

    /// `Spec::addresses`: the append kernel above without buffer
    /// descriptors - its buffers listed as address entries, the module with
    /// the PhysicalStorageBufferAddresses capability (validated with spvval
    /// over a frame capture: `rdnadis --translate` with RDNA_ADDRESSES).
    #[test]
    fn buffers_by_device_address() {
        let Some(words) = crate::title_words("append_kernel") else { return };
        let code = crate::decode_program(&words).unwrap();
        let plan = crate::analysis::analyze(&code, 2, 0);
        let mut spec = cs_spec([8, 8, 1]);
        spec.resources = plan
            .resources
            .iter()
            .map(|r| match r.kind {
                ResKind::Global => ResSpec::Global,
                _ => ResSpec::Buffer { format: 0, dst_sel: 0xFAC },
            })
            .collect();
        let a = translate(&code, &plan, &spec).unwrap();
        spec.addresses = true;
        let b = translate(&code, &plan, &spec).unwrap();
        let count = |o: &Output, addr: bool| o.bindings.iter().filter(|x| if addr { matches!(x.kind, BindingKind::Address(_)) } else { matches!(x.kind, BindingKind::Buffer(_)) }).count();
        assert!(count(&a, false) > 0);
        assert_eq!(count(&a, true), 0);
        assert_eq!(count(&b, false), 0);
        assert_eq!(count(&b, true), count(&a, false));
        // OpCapability PhysicalStorageBufferAddresses (5347).
        let cap = |w: &[u32]| w.windows(2).any(|x| x[0] == (2 << 16 | 17) && x[1] == 5347);
        assert!(cap(&b.spirv) && !cap(&a.spirv));
        // (The test's buffer specs have no format: its formatted store warns in both.)
        assert_eq!(a.warnings, b.warnings);
    }

    /// `Spec::write_marks`: a program with buffer stores gets the marks
    /// binding (and nothing else changes: the same bindings otherwise).
    /// RDNA_MARKS_DUMP=<dir> writes the module for spvval.
    #[test]
    fn write_marks_binding() {
        let Some(words) = crate::title_words("append_kernel") else { return };
        let code = crate::decode_program(&words).unwrap();
        let plan = crate::analysis::analyze(&code, 2, 0);
        assert!(plan.resources.iter().any(|r| r.written));
        let mut spec = cs_spec([8, 8, 1]);
        spec.resources = plan
            .resources
            .iter()
            .map(|r| match r.kind {
                ResKind::Global => ResSpec::Global,
                _ => ResSpec::Buffer { format: 0, dst_sel: 0xFAC },
            })
            .collect();
        let a = translate(&code, &plan, &spec).unwrap();
        spec.write_marks = true;
        let b = translate(&code, &plan, &spec).unwrap();
        let marks = |o: &Output| o.bindings.iter().filter(|x| x.kind == BindingKind::Marks).count();
        assert_eq!(marks(&a), 0);
        assert_eq!(marks(&b), 1);
        assert_eq!(a.bindings.len() + 1, b.bindings.len());
        assert_eq!(a.warnings, b.warnings);
        if let Ok(dir) = std::env::var("RDNA_MARKS_DUMP") {
            let bytes: Vec<u8> = b.spirv.iter().flat_map(|x| x.to_le_bytes()).collect();
            std::fs::write(std::path::Path::new(&dir).join("write_marks.spv"), bytes).unwrap();
        }
        // C5 (`Spec::marks_held`), with the ballot and the partitioned forms:
        // the same bindings and warnings, a longer module.
        for (part, name) in [(false, "write_marks_held.spv"), (true, "write_marks_held_part.spv")] {
            spec.marks_part = part;
            spec.marks_held = false;
            let plain = translate(&code, &plan, &spec).unwrap();
            spec.marks_held = true;
            let held = translate(&code, &plan, &spec).unwrap();
            assert_eq!(plain.bindings.len(), held.bindings.len());
            assert_eq!(marks(&held), 1);
            assert_eq!(plain.warnings, held.warnings);
            assert!(held.spirv.len() > plain.spirv.len());
            if let Ok(dir) = std::env::var("RDNA_MARKS_DUMP") {
                let bytes: Vec<u8> = held.spirv.iter().flat_map(|x| x.to_le_bytes()).collect();
                std::fs::write(std::path::Path::new(&dir).join(name), bytes).unwrap();
            }
        }
    }

    /// `rt_format_entry`: component widths from the least significant
    /// (2_10_10_10: X, Y and Z 10 bits, W 2) and classes; 0 for what the
    /// run-time decoder does not read.
    #[test]
    fn run_time_format_table() {
        let w = |e: u32| (0..4).map(|c| (e >> (6 * c)) & 63).collect::<Vec<_>>();
        let e = rt_format_entry(56);
        assert_eq!((w(e), (e >> 24) & 7, e >> 31), (vec![8, 8, 8, 8], 0, 1));
        let e = rt_format_entry(74);
        assert_eq!((w(e), (e >> 24) & 7), (vec![32, 32, 32, 0], 6));
        let e = rt_format_entry(50);
        assert_eq!((w(e), (e >> 24) & 7), (vec![10, 10, 10, 2], 0));
        // 10_11_11 floats, BC1, an id without a format.
        assert_eq!((rt_format_entry(36), rt_format_entry(169), rt_format_entry(0)), (0, 0, 0));
    }

    /// `FORMAT_RT`: a formatted load reads its format from the V# when the
    /// program runs (the table a private array, no warning, the same
    /// bindings as with the format in the spec). RDNA_RT_FORMAT_DUMP=<dir>
    /// writes both modules for spvval.
    #[test]
    fn run_time_buffer_formats() {
        use crate::tables::*;
        let op = |t: &[(u16, &str)], n: &str| t.iter().find(|(_, x)| x.split('/').next() == Some(n)).unwrap().0 as u32;
        // MUBUF, IDXEN, SOFFSET 0 (inline constant).
        let mubuf = |opc: u32, vaddr: u32, vdata: u32, srsrc: u32| [(0x38 << 26) | (opc << 18) | (1 << 13), vaddr | vdata << 8 | (srsrc / 4) << 16 | 0x80 << 24];
        // buffer_load_format_xyzw v[4:7], v0, s[0:3] ; buffer_store_dwordx4 v[4:7], v0, s[4:7] ; s_endpgm
        let mut words = Vec::new();
        words.extend(mubuf(op(MUBUF, "buffer_load_format_xyzw"), 0, 4, 0));
        words.extend(mubuf(op(MUBUF, "buffer_store_dwordx4"), 0, 4, 4));
        words.push(0xBF81_0000);
        let code = crate::decode_program(&words).unwrap();
        let plan = crate::analysis::analyze(&code, 8, 0);
        assert_eq!(plan.resources.iter().filter(|r| r.formatted).count(), 1);
        let mut spec = cs_spec([64, 1, 1]);
        spec.user_sgprs = 8;
        let specs = |rt: bool| -> Vec<ResSpec> {
            plan.resources
                .iter()
                .map(|r| match r.kind {
                    ResKind::Global => ResSpec::Global,
                    _ if r.formatted && rt => ResSpec::Buffer { format: FORMAT_RT, dst_sel: 0 },
                    _ if r.formatted => ResSpec::Buffer { format: 56, dst_sel: 0xFAC },
                    _ => ResSpec::Buffer { format: 0, dst_sel: 0 },
                })
                .collect()
        };
        spec.resources = specs(false);
        let a = translate(&code, &plan, &spec).unwrap();
        spec.resources = specs(true);
        let b = translate(&code, &plan, &spec).unwrap();
        assert!(a.warnings.is_empty() && b.warnings.is_empty(), "{:?} {:?}", a.warnings, b.warnings);
        assert_eq!(a.bindings.len(), b.bindings.len());
        // OpVariable (59) in Private (6): the table, only with FORMAT_RT.
        let private = |w: &[u32]| {
            let mut i = 5;
            let mut n = 0;
            while i < w.len() && w[i] >> 16 != 0 {
                n += (w[i] & 0xFFFF == 59 && w.get(i + 3) == Some(&6)) as usize;
                i += (w[i] >> 16) as usize;
            }
            n
        };
        assert_eq!((private(&a.spirv), private(&b.spirv)), (0, 1));
        if let Ok(dir) = std::env::var("RDNA_RT_FORMAT_DUMP") {
            for (name, o) in [("rt_formats_static.spv", &a), ("rt_formats.spv", &b)] {
                let bytes: Vec<u8> = o.spirv.iter().flat_map(|x| x.to_le_bytes()).collect();
                std::fs::write(std::path::Path::new(&dir).join(name), bytes).unwrap();
            }
        }
    }

    /// `FORMAT_RT` stores (2026-11-18): a formatted store packs per the
    /// format the V# names when the program runs, translated without a
    /// warning beside the same program with the format in the spec, the
    /// table only in the run-time one. RDNA_RT_FORMAT_DUMP=<dir> writes both
    /// modules for spvval.
    #[test]
    fn run_time_buffer_store_formats() {
        use crate::tables::*;
        let op = |t: &[(u16, &str)], n: &str| t.iter().find(|(_, x)| x.split('/').next() == Some(n)).unwrap().0 as u32;
        let mubuf = |opc: u32, vaddr: u32, vdata: u32, srsrc: u32| [(0x38 << 26) | (opc << 18) | (1 << 13), vaddr | vdata << 8 | (srsrc / 4) << 16 | 0x80 << 24];
        // buffer_load_dwordx4 v[4:7], v0, s[0:3] ; buffer_store_format_xyzw v[4:7], v0, s[4:7] ;
        // buffer_store_format_x v4, v0, s[4:7] ; s_endpgm
        let mut words = Vec::new();
        words.extend(mubuf(op(MUBUF, "buffer_load_dwordx4"), 0, 4, 0));
        words.extend(mubuf(op(MUBUF, "buffer_store_format_xyzw"), 0, 4, 4));
        words.extend(mubuf(op(MUBUF, "buffer_store_format_x"), 0, 4, 4));
        words.push(0xBF81_0000);
        let code = crate::decode_program(&words).unwrap();
        let plan = crate::analysis::analyze(&code, 8, 0);
        assert_eq!(plan.resources.iter().filter(|r| r.formatted).count(), 1);
        let mut spec = cs_spec([64, 1, 1]);
        spec.user_sgprs = 8;
        let specs = |rt: bool| -> Vec<ResSpec> {
            plan.resources
                .iter()
                .map(|r| match r.kind {
                    ResKind::Global => ResSpec::Global,
                    _ if r.formatted && rt => ResSpec::Buffer { format: FORMAT_RT, dst_sel: 0 },
                    _ if r.formatted => ResSpec::Buffer { format: 56, dst_sel: 0xFAC },
                    _ => ResSpec::Buffer { format: 0, dst_sel: 0 },
                })
                .collect()
        };
        spec.resources = specs(false);
        let a = translate(&code, &plan, &spec).unwrap();
        spec.resources = specs(true);
        let b = translate(&code, &plan, &spec).unwrap();
        assert!(a.warnings.is_empty() && b.warnings.is_empty(), "{:?} {:?}", a.warnings, b.warnings);
        assert_eq!(a.bindings.len(), b.bindings.len());
        let private = |w: &[u32]| {
            let mut i = 5;
            let mut n = 0;
            while i < w.len() && w[i] >> 16 != 0 {
                n += (w[i] & 0xFFFF == 59 && w.get(i + 3) == Some(&6)) as usize;
                i += (w[i] >> 16) as usize;
            }
            n
        };
        assert_eq!((private(&a.spirv), private(&b.spirv)), (0, 1));
        if let Ok(dir) = std::env::var("RDNA_RT_FORMAT_DUMP") {
            for (name, o) in [("rt_store_static.spv", &a), ("rt_store.spv", &b)] {
                let bytes: Vec<u8> = o.spirv.iter().flat_map(|x| x.to_le_bytes()).collect();
                std::fs::write(std::path::Path::new(&dir).join(name), bytes).unwrap();
            }
        }
    }

    /// `ImageDim::D3OrD2`: an image_load addressed as 3D on a resource that
    /// may be bound with a 2D texture gets a 3D and a 2D variable (`Image`,
    /// `ImageAlt`) and no warning. RDNA_D3OR2D_DUMP=<dir> writes the module
    /// for spvval.
    #[test]
    fn three_d_or_two_d_images() {
        use crate::tables::*;
        let op = |t: &[(u16, &str)], n: &str| t.iter().find(|(_, x)| x.split('/').next() == Some(n)).unwrap().0 as u32;
        // image_load v[4:7], v[0:2], s[0:7] dmask 0xf dim 3D ;
        // buffer_store_dwordx4 v[4:7], v0, s[8:11] idxen ; s_endpgm
        let words = [
            (0x3C << 26) | (op(MIMG, "image_load") << 18) | (0xF << 8) | (2 << 3),
            4 << 8,
            (0x38 << 26) | (op(MUBUF, "buffer_store_dwordx4") << 18) | (1 << 13),
            4 << 8 | 2 << 16 | 0x80 << 24,
            0xBF81_0000,
        ];
        let code = crate::decode_program(&words).unwrap();
        let plan = crate::analysis::analyze(&code, 12, 0);
        let mut spec = cs_spec([64, 1, 1]);
        spec.user_sgprs = 12;
        spec.resources = plan
            .resources
            .iter()
            .map(|r| match r.kind {
                ResKind::Image => ResSpec::Image { dim: ImageDim::D3OrD2, class: NumClass::Float, depth: false },
                _ => ResSpec::Buffer { format: 0, dst_sel: 0 },
            })
            .collect();
        let o = translate(&code, &plan, &spec).unwrap();
        assert!(o.warnings.is_empty(), "{:?}", o.warnings);
        let kinds: Vec<BindingKind> = o.bindings.iter().map(|b| b.kind).collect();
        assert!(kinds.iter().any(|k| matches!(k, BindingKind::Image(_))) && kinds.iter().any(|k| matches!(k, BindingKind::ImageAlt(_))), "{kinds:?}");
        if let Ok(dir) = std::env::var("RDNA_D3OR2D_DUMP") {
            let bytes: Vec<u8> = o.spirv.iter().flat_map(|x| x.to_le_bytes()).collect();
            std::fs::write(std::path::Path::new(&dir).join("d3or2d.spv"), bytes).unwrap();
        }
    }

    /// Resources of another kind than the instruction names (learned specs):
    /// a 2D-array fetch from a 3D resource keeps its slice as the depth, and a
    /// sample of a multisampled resource becomes a fetch of sample 0 (SPIR-V
    /// samples no multisampled image).
    #[test]
    fn instruction_and_resource_dims_differ() {
        use crate::tables::*;
        use rspirv::spirv::Op;
        let op = |t: &[(u16, &str)], n: &str| t.iter().find(|(_, x)| x.split('/').next() == Some(n)).unwrap().0 as u32;
        // (MIMG word 0: dmask [11:8], dim [5:3]; word 1: vaddr [7:0], vdata
        // [15:8], srsrc/4 [20:16], ssamp/4 [25:21].)
        let translate_one = |name: &str, dim: u32, res: ImageDim| {
            // <name> v[4:7], v[0:2], s[0:7] (, s[8:11]) dmask 0xf ;
            // buffer_store_dwordx4 v[4:7], v0, s[12:15] idxen ; s_endpgm
            let words = [
                (0x3C << 26) | (op(MIMG, name) << 18) | (0xF << 8) | (dim << 3),
                4 << 8 | 2 << 21,
                (0x38 << 26) | (op(MUBUF, "buffer_store_dwordx4") << 18) | (1 << 13),
                4 << 8 | 3 << 16 | 0x80 << 24,
                0xBF81_0000,
            ];
            let code = crate::decode_program(&words).unwrap();
            let plan = crate::analysis::analyze(&code, 16, 0);
            let mut spec = cs_spec([64, 1, 1]);
            spec.user_sgprs = 16;
            spec.resources = plan
                .resources
                .iter()
                .map(|r| match r.kind {
                    ResKind::Image => ResSpec::Image { dim: res, class: NumClass::Float, depth: false },
                    ResKind::Sampler => ResSpec::Sampler,
                    _ => ResSpec::Buffer { format: 0, dst_sel: 0 },
                })
                .collect();
            let o = translate(&code, &plan, &spec).unwrap();
            rspirv::dr::load_words(&o.spirv).unwrap()
        };
        // The component count of the coordinate operand of each instruction `of`.
        let coords = |m: &rspirv::dr::Module, of: Op| -> Vec<u32> {
            let insts: Vec<&rspirv::dr::Instruction> = m.types_global_values.iter().chain(m.functions.iter().flat_map(|f| f.blocks.iter().flat_map(|b| b.instructions.iter()))).collect();
            let def = |id: u32| insts.iter().find(|i| i.result_id == Some(id)).copied();
            insts
                .iter()
                .filter(|i| i.class.opcode == of)
                .map(|i| {
                    let c = def(i.operands[1].unwrap_id_ref()).unwrap();
                    let t = def(c.result_type.unwrap()).unwrap();
                    if t.class.opcode == Op::TypeVector { t.operands[1].unwrap_literal_bit32() } else { 1 }
                })
                .collect()
        };
        // image_load, dim 2D_ARRAY (5), on a 3D resource: (x, y, slice).
        let m = translate_one("image_load", 5, ImageDim::D3);
        assert_eq!(coords(&m, Op::ImageFetch), vec![3]);
        // image_sample_lz, dim 2D (1), on a multisampled resource.
        let m = translate_one("image_sample_lz", 1, ImageDim::D2Msaa);
        assert!(coords(&m, Op::ImageSampleExplicitLod).is_empty() && coords(&m, Op::ImageSampleImplicitLod).is_empty());
        assert_eq!(coords(&m, Op::ImageFetch), vec![2]);
    }

    /// A program's translation does not depend on its address (s_getpc_b64,
    /// the loop watchdog's record): copies at other addresses share it.
    #[test]
    fn translation_is_position_independent() {
        // s_mov_b32 s0, 0; s_getpc_b64 s[4:5]; loop: s_add_u32 s0, s0, 1;
        // s_cmp_lt_u32 s0, 10; s_cbranch_scc1 loop; s_endpgm
        let words = [0xbe800380, 0xbe841f00, 0x80008100, 0xbf0a8a00, 0xbf85fffd, 0xbf810000];
        let code = crate::decode_program(&words).unwrap();
        assert_eq!(code[1].name(), Some("s_getpc_b64"));
        let plan = crate::analysis::analyze(&code, 2, 0);
        let a = cs_spec([64, 1, 1]);
        let mut b = cs_spec([64, 1, 1]);
        b.code_base = 0x7_1234_5000;
        let (ta, tb) = (translate(&code, &plan, &a).unwrap(), translate(&code, &plan, &b).unwrap());
        assert_eq!(ta.spirv, tb.spirv);
    }

    fn ps_spec(inputs: &[(u8, bool)]) -> Spec {
        let mut s = cs_spec([1, 1, 1]);
        s.stage = Stage::Fragment;
        s.barycentric = true;
        s.ps_input_addr = 0x2;
        s.ps_inputs = inputs.iter().map(|&(param, flat)| PsInput { param, flat, default: None }).collect();
        s
    }

    /// Input variables (Location, PerVertexKHR, Flat) of a SPIR-V module.
    fn input_locations(spirv: &[u32]) -> Vec<(u32, bool, bool)> {
        let mut decs: HashMap<u32, (Option<u32>, bool, bool)> = HashMap::default();
        let mut inputs = Vec::new();
        let mut i = 5;
        while i < spirv.len() {
            let (op, n) = (spirv[i] & 0xFFFF, (spirv[i] >> 16) as usize);
            let a = &spirv[i + 1..i + n];
            match op {
                71 => {
                    let d = decs.entry(a[0]).or_default();
                    match a[1] {
                        30 => d.0 = Some(a[2]),
                        14 => d.2 = true,
                        5285 => d.1 = true,
                        _ => {}
                    }
                }
                59 if a[2] == 1 => inputs.push(a[1]),
                _ => {}
            }
            i += n.max(1);
        }
        let mut out: Vec<_> = inputs.iter().filter_map(|v| decs.get(v).and_then(|d| d.0.map(|l| (l, d.1, d.2)))).collect();
        out.sort();
        out
    }

    /// Two attributes on one parameter, one read per vertex and one flat (ps
    /// 0x139d9a6500, World 4-1's cliffs: input_cntl [0x0 0x401 0x401
    /// 0x402]): one per-vertex input at that location, none flat.
    #[test]
    fn shared_parameter_has_one_input() {
        // v_interp_mov_f32 v5, p0, attr1.w; v_interp_mov_f32 v8, p10, attr1.w;
        // v_interp_mov_f32 v11, p20, attr1.w; v_interp_mov_f32 v22, p0, attr2.x;
        // v_interp_p1_f32 v10, v0, attr0.z; v_interp_p2_f32 v10, v1, attr0.z;
        // v_interp_mov_f32 v9, p0, attr3.x; v_interp_mov_f32 v12, p10, attr3.x;
        // exp mrt0 v5, v8, v11, v22 done vm; s_endpgm
        let words = [
            0xc8160702, 0xc8220700, 0xc82e0701, 0xc85a0802, 0xc8280200, 0xc8290201, 0xc8260c02, 0xc8320c00, 0xf800180f, 0x160b0805, 0xbf810000,
        ];
        let code = crate::decode_program(&words).unwrap();
        let spec = ps_spec(&[(0, false), (1, true), (1, true), (2, true)]);
        assert_eq!(per_vertex_locations(&code, &spec).into_iter().collect::<Vec<_>>(), vec![1, 2]);
        let plan = crate::analysis::analyze(&code, 0, 0);
        let out = translate(&code, &plan, &spec).unwrap();
        assert!(out.warnings.is_empty(), "{:?}", out.warnings);
        assert_eq!(input_locations(&out.spirv), vec![(0, false, false), (1, true, false), (2, true, false)]);
    }

    /// The same program without fragment barycentrics (`Spec::bary_gs`): each
    /// per-vertex location's three values as flat inputs (vertex 0's at the
    /// location, 1's and 2's at free ones), then the barycentrics.
    #[test]
    fn per_vertex_from_geometry_shader() {
        let words = [
            0xc8160702, 0xc8220700, 0xc82e0701, 0xc85a0802, 0xc8280200, 0xc8290201, 0xc8260c02, 0xc8320c00, 0xf800180f, 0x160b0805, 0xbf810000,
        ];
        let code = crate::decode_program(&words).unwrap();
        let mut spec = ps_spec(&[(0, false), (1, true), (1, true), (2, true)]);
        spec.barycentric = false;
        spec.bary_gs = true;
        let plan = crate::analysis::analyze(&code, 0, 0);
        let out = translate(&code, &plan, &spec).unwrap();
        assert!(out.warnings.is_empty(), "{:?}", out.warnings);
        let io = out.bary_gs.clone().expect("geometry shader inputs");
        assert_eq!(io, gs::BaryIo { per_vertex: vec![(1, [3, 4]), (2, [5, 6])], bary: [7, 8] });
        let flat = |l| (l, false, true);
        assert_eq!(input_locations(&out.spirv), vec![(0, false, false), flat(1), flat(2), flat(3), flat(4), flat(5), flat(6), (7, false, false), (8, false, false)]);
        assert!(!out.spirv.windows(2).any(|w| w[0] & 0xFFFF == 17 && w[1] == 5284), "no FragmentBarycentricKHR capability");
        // The geometry shader for it, with a vertex stage writing 0..=2.
        let g = gs::bary_gs(0b111, out.in_mask, &io, false);
        assert_eq!(rspirv::dr::load_words(&g).expect("parses").entry_points[0].operands.len() - 3, 2 + 3 + 1 + 6 + 2);
        // `RDNA_GS_DUMP=<dir>`: both stages written there for spirv-val.
        if let Some(dir) = std::env::var_os("RDNA_GS_DUMP") {
            let dir = std::path::PathBuf::from(dir);
            for (name, w) in [("bary_ps.spv", &out.spirv), ("bary_ps_gs.spv", &g)] {
                std::fs::write(dir.join(name), w.iter().flat_map(|x| x.to_le_bytes()).collect::<Vec<u8>>()).expect("written");
            }
        }
    }

    /// Execution modes (OpExecutionMode operands after the entry point) of a
    /// SPIR-V module.
    fn execution_modes(spirv: &[u32]) -> Vec<u32> {
        let mut out = Vec::new();
        let mut i = 5;
        while i < spirv.len() {
            let (n, op) = ((spirv[i] >> 16) as usize, spirv[i] & 0xFFFF);
            if op == 16 && n >= 3 {
                out.push(spirv[i + 2]);
            }
            i += n.max(1);
        }
        out
    }

    /// DEPTH_BEFORE_SHADER (`Spec::early_tests`) makes the fragment shader
    /// declare EarlyFragmentTests (mode 9), only when asked.
    #[test]
    fn early_fragment_tests() {
        // The pixel shader of shared_parameter_has_one_input (mrt0 export).
        let words = [
            0xc8160702, 0xc8220700, 0xc82e0701, 0xc85a0802, 0xc8280200, 0xc8290201, 0xc8260c02, 0xc8320c00, 0xf800180f, 0x160b0805, 0xbf810000,
        ];
        let code = crate::decode_program(&words).unwrap();
        let plan = crate::analysis::analyze(&code, 0, 0);
        let mut spec = ps_spec(&[(0, false), (1, true), (1, true), (2, true)]);
        let late = translate(&code, &plan, &spec).unwrap();
        assert!(!execution_modes(&late.spirv).contains(&9));
        spec.early_tests = true;
        let early = translate(&code, &plan, &spec).unwrap();
        assert!(execution_modes(&early.spirv).contains(&9));
        // Allowed by the draw's state (`early_ok`): a program that writes no
        // memory tests early; one with a store does not.
        spec.early_tests = false;
        spec.early_ok = true;
        let allowed = translate(&code, &plan, &spec).unwrap();
        assert!(execution_modes(&allowed.spirv).contains(&9));
        assert!(!writes_memory(&code));
        // buffer_store_dword v1, v0, s[0:3], 0 idxen; s_endpgm
        let store = crate::decode_program(&[0xE0702000, 0x80000100, 0xbf810000]).unwrap();
        assert!(writes_memory(&store));
    }

    /// 16-bit instructions of the title's compute shaders (cs 0x100898d500's
    /// `v_add_u16_e64 v5, 0xffff, v7`, cs 0x10089e3a00's v_cmpx_gt_f16)
    /// translate without warnings.
    #[test]
    fn sixteen_bit_ops_translate() {
        // v_add_u16_e64 v5, 0xffff, v7; v_cmpx_gt_f16 1.0, v1;
        // v_max_i16 v2, v3, v4 op_sel:[0,0,0,1]; s_endpgm
        let words = [0xd7030005, 0x00020eff, 0x0000ffff, 0x7db802f2, 0xd70a4002, 0x00020903, 0xbf810000];
        let code = crate::decode_program(&words).unwrap();
        assert_eq!(code[0].base_name(), Some("v_add_u16"));
        assert_eq!(code[1].name(), Some("v_cmpx_gt_f16"));
        assert_eq!(code[2].base_name(), Some("v_max_i16"));
        let plan = crate::analysis::analyze(&code, 2, 0);
        let out = translate(&code, &plan, &cs_spec([64, 1, 1])).unwrap();
        assert!(out.warnings.is_empty(), "{:?}", out.warnings);
    }

    /// v_interp_mov_f32 of P10 / P20 (cs 0x100cc3d100's foliage alpha test:
    /// P10, P0, P20 of attribute 0): that attribute is read per vertex; a
    /// lone P0 read (a flat value) is not.
    #[test]
    fn per_vertex_attributes() {
        let code = crate::decode_program(&[0xc80e0000, 0xc80a0002, 0xc8120001, 0xbf810000]).unwrap();
        assert_eq!(per_vertex_attrs(&code).into_iter().collect::<Vec<_>>(), vec![0]);
        let code = crate::decode_program(&[0xc80a0002, 0xbf810000]).unwrap();
        assert!(per_vertex_attrs(&code).is_empty());
    }

    /// Wave-mask mode (one wave, LDS): the compare whose mask is counted
    /// gets a ballot (two subgroup ORs), the one only selected from does
    /// not; the module parses.
    #[test]
    fn wave_mask_ballots_only_whole_reads() {
        use crate::tables::*;
        let op = |t: Table, n: &str| t.iter().find(|(_, x)| x.split('/').next() == Some(n)).unwrap().0 as u32;
        let vopc = |o: u32, s0: u32, v1: u32| (0x3E << 25) | (o << 17) | (v1 << 9) | s0;
        let words = [
            vopc(op(VOPC, "v_cmp_gt_u32"), 0xA0, 0),                              // vcc = 32 > v0
            (0x17D << 23) | (4 << 16) | (op(SOP1, "s_bcnt1_i32_b64") << 8) | 106, // s4 = popcount(vcc)
            vopc(op(VOPC, "v_cmp_lt_f32"), 256, 1),                               // vcc = v0 < v1
            (op(VOP2, "v_cndmask_b32") << 25) | (2 << 17) | (1 << 9) | 256,       // v2 = vcc ? v1 : v0
            (0x3F << 25) | (3 << 17) | (op(VOP1, "v_mov_b32") << 9) | 4,          // v3 = s4
            (0x36 << 26) | (op(DS, "ds_write_b32") << 18),                        // ds_write_b32 v0, v3
            3 << 8,
            0xBF81_0000, // s_endpgm
        ];
        let code = crate::decode_program(&words).unwrap();
        assert_eq!(crate::wavemask::complete_masks(&code, true).into_iter().collect::<Vec<_>>(), vec![code[0].pc]);
        let plan = crate::analysis::analyze(&code, 2, 0);
        let mut spec = cs_spec([64, 1, 1]);
        spec.cs_rsrc2 = 1 << 15;
        let out = translate(&code, &plan, &spec).unwrap();
        let mut loader = rspirv::dr::Loader::new();
        rspirv::binary::parse_words(&out.spirv, &mut loader).unwrap();
        let ors = loader.module().functions.iter().flat_map(|f| f.blocks.iter()).flat_map(|b| b.instructions.iter()).filter(|i| i.class.opcode == spirv::Op::GroupNonUniformBitwiseOr).count();
        assert_eq!(ors, 2);
    }

    /// The counting kernel (cs 0x10088aa500) sums the class masks with DPP
    /// and v_readlane; its lane 0 adds the wave's total: no election.
    #[test]
    fn lane0_election_not_for_wave_sums() {
        let Some(words) = crate::title_words("count_kernel") else { return };
        let code = crate::decode_program(&words).unwrap();
        assert!(lane0_elections(&code, &cs_spec([8, 8, 1])).is_empty());
    }
}
