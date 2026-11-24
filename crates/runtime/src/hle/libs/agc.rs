//! libSceAgc: PS5 graphics command-buffer construction (HLE).
//!
//! The title builds gfx10 PM4 command buffers through `sceAgcDcb*`/`Acb*`/`Cb*`
//! writers into title-owned memory, sizes them with `*GetSize`, and edits
//! written packets with `*Patch*` functions. This module emits genuine gfx10
//! PM4 packets so that the title's own inline packet code, these writers and
//! the command processor (`crate::gpu`) all agree on one format.
//!
//! Register-default tables (`sceAgcGetRegisterDefaults2*`) normally come from
//! the firmware library, which is not supplied; see `agc_defaults.rs` for the
//! entries served and their provenance. The title looks up register types by
//! 32-bit IDs and caches the `{register, value}` it finds (-1 when absent).

use crate::hle::Registry;
use crate::{reg, sysv};
use parking_lot::Mutex;

/// AGC CommandBuffer (title-owned): {bottom, top, cursor_up, cursor_down,
/// grow_callback, user_data, reserved_dw}.
pub mod cb {
    pub const BOTTOM: u64 = 0x00;
    pub const TOP: u64 = 0x08;
    pub const CURSOR_UP: u64 = 0x10;
    pub const CURSOR_DOWN: u64 = 0x18;
    pub const CALLBACK: u64 = 0x20;
    pub const USER_DATA: u64 = 0x28;
    pub const RESERVED_DW: u64 = 0x30;
}

unsafe fn rd64(a: u64) -> u64 {
    unsafe { std::ptr::read_volatile(a as *const u64) }
}
unsafe fn rd32(a: u64) -> u32 {
    unsafe { std::ptr::read_volatile(a as *const u32) }
}
unsafe fn wr64(a: u64, v: u64) {
    unsafe { std::ptr::write_volatile(a as *mut u64, v) }
}

/// Reserves `n` dwords at the command buffer's upward cursor (calling the
/// title's grow callback when needed) and returns their address, or 0.
pub fn alloc_dw(buf: u64, n: u32) -> u64 {
    if buf == 0 || n == 0 {
        return 0;
    }
    unsafe {
        let avail = |b: u64| -> u64 {
            let up = rd64(b + cb::CURSOR_UP) as i64;
            let down = rd64(b + cb::CURSOR_DOWN) as i64;
            let a = ((down - up) >> 2) as u64;
            a.saturating_sub(rd32(b + cb::RESERVED_DW) as u64)
        };
        if (n as u64) > avail(buf) {
            let callback = rd64(buf + cb::CALLBACK);
            if callback == 0 {
                crate::error!("agc: command buffer {buf:#x} full ({} dwords requested) and no grow callback", n);
                return 0;
            }
            let reserved = rd32(buf + cb::RESERVED_DW) as u64;
            let ok = crate::hle::call_guest(callback, &[buf, n as u64 + reserved, rd64(buf + cb::USER_DATA)]) & 0xFF != 0;
            if !ok || (n as u64) > avail(buf) {
                crate::error!("agc: command buffer {buf:#x} grow callback failed for {n} dwords");
                return 0;
            }
        }
        let p = rd64(buf + cb::CURSOR_UP);
        wr64(buf + cb::CURSOR_UP, p + n as u64 * 4);
        p
    }
}

/// PM4 type-3 header: `count` = dwords following the header.
pub const fn pm4_hdr(opcode: u32, count: u32, compute: bool) -> u32 {
    (3 << 30) | (((count - 1) & 0x3FFF) << 16) | ((opcode & 0xFF) << 8) | ((compute as u32) << 1)
}

// ---------------------------------------------------------------------------
// Register defaults

/// ShaderRegister offset convention: dword index within the register's
/// space (context 0x28000, SH 0xB000, uconfig 0x30000).
pub fn reg_index(addr: u32) -> u32 {
    match addr {
        0x28000..=0x2FFFF => (addr - 0x28000) / 4,
        0xB000..=0xBFFF => (addr - 0xB000) / 4,
        0x30000..=0x3FFFF => (addr - 0x30000) / 4,
        _ => addr / 4,
    }
}

/// Builds a RegisterDefaults structure in guest memory mirroring the system
/// library's shape: {tbl0..tbl3: ShaderRegister**, u32 counts[4],
/// types: {u32 id, u32 loc, u32 0}*, u32 count}, loc = (pointer index << 2) | table
/// plus run-shape bits (served verbatim).
/// Pointer slots without a known entry are NULL (the title treats them as absent).
fn build_defaults(defs: &[super::agc_defaults::Def], what: &str) -> u64 {
    let s = crate::mem::arena_alloc(0x40, 8);
    unsafe {
        std::ptr::write_bytes(s as *mut u8, 0, 0x40);
        for table in 0..4u32 {
            let ds: Vec<&super::agc_defaults::Def> = defs.iter().filter(|d| d.table() == table).collect();
            let slots = ds.iter().map(|d| d.index() + 1).max().unwrap_or(0);
            if slots == 0 {
                continue;
            }
            let ptrs = crate::mem::arena_alloc(8 * slots as u64, 8);
            std::ptr::write_bytes(ptrs as *mut u8, 0, 8 * slots as usize);
            for d in &ds {
                let regs = crate::mem::arena_alloc(8 * d.regs.len() as u64, 8);
                for (k, &(off, val)) in d.regs.iter().enumerate() {
                    *((regs + 8 * k as u64) as *mut u32) = off;
                    *((regs + 8 * k as u64 + 4) as *mut u32) = val;
                }
                *((ptrs + 8 * d.index() as u64) as *mut u64) = regs;
            }
            *((s + 8 * table as u64) as *mut u64) = ptrs;
            *((s + 0x20 + 4 * table as u64) as *mut u32) = slots;
        }
        let types = crate::mem::arena_alloc(12 * defs.len().max(1) as u64, 8);
        for (i, d) in defs.iter().enumerate() {
            let t = types + 12 * i as u64;
            *(t as *mut u32) = d.id;
            *((t + 4) as *mut u32) = d.loc;
            *((t + 8) as *mut u32) = 0;
        }
        *((s + 0x30) as *mut u64) = types;
        *((s + 0x38) as *mut u32) = defs.len() as u32;
    }
    crate::info!("agc: {what} register defaults: {} register types", defs.len());
    s
}

static DEFAULTS_PTR: Mutex<(u64, u64)> = Mutex::new((0, 0));

// ---------------------------------------------------------------------------
// Shader objects
//
// Agc shader object (version 0x18), found inside the title's `.csdr` files and
// embedded in the eboot: {u32 magic "1234", u32 version, user_data*, code*,
// cx_registers*, sh_registers*, specials*, input_semantics*, output_semantics*,
// u32 header_size, u32 shader_size, u32 ecb_size_dqw, u32 target,
// u32 num_input_semantics, u16 scratch_dw, u16 num_output_semantics,
// u16 special_sizes, u8 type, u8 num_cx_registers, u8 num_sh_registers}.
// Pointers are stored self-relative until sceAgcCreateShader relocates them.

/// Shader programs created by the title: program address -> (stage, code
/// bytes, instance-offset user-data slot, registration number).
static SHADERS: Mutex<std::collections::BTreeMap<u64, (u8, usize, Option<u8>, u64)>> = Mutex::new(std::collections::BTreeMap::new());

/// Shader objects created so far (`sceAgcCreateShader`): caches of decoded
/// programs stay valid while it is unchanged.
static REGISTRATIONS: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);
/// Bumped after each insert into SHADERS: the recording threads' copies of
/// instance-offset slots (`instance_offset_slot`) hold while it is unchanged.
static SHADERS_GEN: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);

/// Code size of the program at `addr` (from its shader object), if known.
pub fn shader_code_size(addr: u64) -> Option<usize> {
    SHADERS.lock().get(&addr).map(|&(_, n, _, _)| n).filter(|&n| n > 0 && n < 1 << 20)
}

/// User-data slot (counted from the stage's first user SGPR) in which the
/// program at `addr` expects a draw's start instance, if it has one.
pub fn instance_offset_slot(addr: u64) -> Option<u32> {
    // From this thread's copy: indirect draws ask for every draw, and the
    // registry's lock is shared by all recording threads. Consecutive draws
    // mostly keep their shader: the last answer is checked first (the map
    // lookup was ~10% of the graphics CP's busy time at a vista).
    thread_local! {
        static LAST: std::cell::Cell<(u64, u64, Option<u32>)> = const { std::cell::Cell::new((u64::MAX, u64::MAX, None)) };
        static MEMO: std::cell::RefCell<(u64, crate::platform::graphics::Map<u64, Option<u32>>)> = std::cell::RefCell::new((u64::MAX, Default::default()));
    }
    let g = SHADERS_GEN.load(std::sync::atomic::Ordering::Acquire);
    let (lg, la, lv) = LAST.get();
    if lg == g && la == addr && crate::platform::graphics::plan_fast() {
        return lv;
    }
    let v = MEMO.with(|m| {
        let mut m = m.borrow_mut();
        if m.0 != g {
            m.0 = g;
            m.1.clear();
        }
        if let Some(&v) = m.1.get(&addr) {
            return v;
        }
        let v = SHADERS.lock().get(&addr).and_then(|&(_, _, s, _)| s).map(u32::from);
        m.1.insert(addr, v);
        v
    });
    LAST.set((g, addr, v));
    v
}

/// How many shader objects have been created (changes with every creation).
pub fn shader_registrations() -> u64 {
    REGISTRATIONS.load(std::sync::atomic::Ordering::Acquire)
}

/// The registration number of the shader object whose program is at `addr`.
pub fn shader_registration(addr: u64) -> Option<u64> {
    SHADERS.lock().get(&addr).map(|&(_, _, _, r)| r)
}

/// Draw-modifier bit of primitive shaders that take an instance offset in
/// a user SGPR (the modifier's layout is otherwise unknown; in this title
/// it comes with bit 2 set as well: 0x40100004 against 0x40000000).
const DRAW_MODIFIER_INSTANCE_OFFSET: u64 = 1 << 20;

/// The instance-offset slot of a primitive shader object: the last dword of
/// its user-data range. In PPSA01341 every shader with the flag has a range
/// one dword longer than its declared resources (ShaderUserData +0x2A), and
/// its code adds the low half of that SGPR to the instance id (s10 in the
/// GPU-driven static meshes' vertex shaders). On hardware an indirect draw
/// has the command processor write its start instance there
/// (START_INST_LOC), and the instance id VGPR counts from 0.
unsafe fn instance_offset_slot_of(h: u64, stage: u8) -> Option<u8> {
    use shader::*;
    let sp = unsafe { rd64(h + SPECIALS) };
    if stage != 2 || sp == 0 {
        return None;
    }
    let modifier = unsafe { rd64(sp + specials::DRAW_MODIFIER) };
    let end = unsafe { *((sp + specials::USER_DATA_RANGE + 2) as *const u16) };
    let slot = (modifier & DRAW_MODIFIER_INSTANCE_OFFSET != 0 && (1..=32).contains(&end)).then(|| (end - 1) as u8);
    static SEEN: Mutex<Option<std::collections::HashSet<(u64, Option<u8>)>>> = Mutex::new(None);
    if SEEN.lock().get_or_insert_with(Default::default).insert((modifier, slot)) {
        crate::info!("agc: primitive shaders with draw modifier {modifier:#x}: start instance in user-data slot {slot:?} (shader {h:#x})");
    }
    slot
}

pub mod shader {
    pub const MAGIC: u32 = 0x3433_3231; // "1234"
    pub const VERSION: u32 = 0x18;
    pub const USER_DATA: u64 = 0x08;
    pub const CODE: u64 = 0x10;
    pub const CX_REGS: u64 = 0x18;
    pub const SH_REGS: u64 = 0x20;
    pub const SPECIALS: u64 = 0x28;
    pub const INPUT_SEMANTICS: u64 = 0x30;
    pub const OUTPUT_SEMANTICS: u64 = 0x38;
    pub const SHADER_SIZE: u64 = 0x44;
    pub const TYPE: u64 = 0x5A;
    pub const NUM_CX: u64 = 0x5B;
    pub const NUM_SH: u64 = 0x5C;
    /// ShaderUserData: {direct_resource_offset*, sharp_resource_offset*[4], ...}.
    pub const UD_DIRECT: u64 = 0x00;
    pub const UD_SHARP: u64 = 0x08;
}

/// SH register (dword index from 0xB000) holding a stage's program address.
fn program_register(stage: u8) -> Option<u32> {
    Some(match stage {
        0 => 0x20C, // COMPUTE_PGM_LO
        1 => 0x008, // SPI_SHADER_PGM_LO_PS
        2 => 0x0C8, // SPI_SHADER_PGM_LO_ES (NGG primitive shader)
        3 => 0x148, // SPI_SHADER_PGM_LO_LS (merged LS/HS)
        6 => 0x088, // SPI_SHADER_PGM_LO_GS
        7 => 0x108, // SPI_SHADER_PGM_LO_HS
        _ => return None,
    })
}

unsafe fn relocate(field: u64) {
    unsafe {
        let v = rd64(field);
        if v != 0 {
            wr64(field, field.wrapping_add(v));
        }
    }
}

const ERR_INVALID_SHADER_PROGRAM: i32 = 0x8A6C_0005u32 as i32;

/// `DS_SHADER_INFO=1`: logs each created shader object's header, special
/// registers, user-data layout and SH registers (interface research).
unsafe fn log_shader_info(h: u64) {
    use shader::*;
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    if !*ON.get_or_init(|| std::env::var_os("DS_SHADER_INFO").is_some()) {
        return;
    }
    let readable = |a: u64, n: u64| a != 0 && crate::platform::loader_hooks::guest_readable(a, n);
    let words = |a: u64, n: u64| (0..n).map(|i| format!("{:08x}", unsafe { rd32(a + 4 * i) })).collect::<Vec<_>>().join(" ");
    unsafe {
        let stage = *((h + TYPE) as *const u8);
        let mut s = format!("agc: shader info {h:#x} stage {stage} header [{}]", words(h, 0x60 / 4));
        let sp = rd64(h + SPECIALS);
        if readable(sp, 0x30) {
            s += &format!(" specials [{}]", words(sp, 0x30 / 4));
        }
        let ud = rd64(h + USER_DATA);
        if readable(ud, 0x40) {
            s += &format!(" userdata [{}]", words(ud, 0x40 / 4));
            let d = rd64(ud + UD_DIRECT);
            if readable(d, 0x40) {
                let v: Vec<String> = (0..32).map(|i| format!("{:04x}", *((d + 2 * i) as *const u16))).collect();
                s += &format!(" direct [{}]", v.join(" "));
            }
        }
        let regs = rd64(h + SH_REGS);
        let n = *((h + NUM_SH) as *const u8) as u64;
        if readable(regs, 8 * n) {
            let v: Vec<String> = (0..n).map(|i| format!("{:x}={:08x}", rd32(regs + 8 * i), rd32(regs + 8 * i + 4))).collect();
            s += &format!(" sh [{}]", v.join(" "));
        }
        crate::info!("{s}");
    }
}

/// ShaderSpecialRegs (0x30 bytes): {ge_cntl, vgt_shader_stages_en: ShaderRegister,
/// u32 dispatch_modifier, u16 user_data start/end, u64 draw_modifier,
/// vgt_gs_out_prim_type, ge_user_vgpr_en: ShaderRegister}.
pub mod specials {
    pub const GE_CNTL: u64 = 0x00;
    pub const VGT_SHADER_STAGES_EN: u64 = 0x08;
    pub const DISPATCH_MODIFIER: u64 = 0x10;
    pub const USER_DATA_RANGE: u64 = 0x14;
    pub const DRAW_MODIFIER: u64 = 0x18;
    pub const VGT_GS_OUT_PRIM_TYPE: u64 = 0x20;
    pub const GE_USER_VGPR_EN: u64 = 0x28;
}

const REG_SPI_PS_INPUT_CNTL_0: u32 = 0x191; // context 0x28644
const REG_VGT_GS_OUT_PRIM_TYPE: u32 = 0x29B; // context 0x28A6C
const REG_VGT_PRIMITIVE_TYPE: u32 = 0x242; // uconfig 0x30908

/// VGT_GS_OUT_PRIM_TYPE for a VGT_DI_PRIM_TYPE (DI_PT_*) draw primitive.
fn gs_out_prim(di_pt: u32) -> u32 {
    match di_pt {
        1 => 0,                       // points
        2 | 3 | 10 | 11 | 18 => 1,    // line types -> LINESTRIP
        7 | 17 => 3,                  // 2D rectangle / rect list -> RECTLIST
        _ => 2,                       // triangle types -> TRISTRIP
    }
}

unsafe fn copy_reg(dst: u64, src: u64) {
    unsafe { wr64(dst, rd64(src)) }
}

/// One SPI_PS_INPUT_CNTL value from a PS input semantic word and the matching
/// GS/VS output semantic word (if any). Semantic word: semantic[7:0],
/// hardware_mapping[15:8], size[19:16], f16 halves mask[21:20], flat[22],
/// linear[23], custom[24], default_value[29:28], default_value_hi[31:30].
/// Register fields: OFFSET[5:0] (bit 5 = use DEFAULT_VAL), DEFAULT_VAL[9:8],
/// FLAT_SHADE[10], FP16_INTERP_MODE[19], USE_DEFAULT_ATTR1[20],
/// DEFAULT_VAL_ATTR1[22:21], ATTR0_VALID[24], ATTR1_VALID[25].
fn ps_input_cntl(ps: u32, gs: Option<u32>) -> u32 {
    let flat = (ps >> 22) & 1 != 0 || (ps >> 24) & 1 != 0;
    let f16 = (ps >> 20) & 3;
    let mut v = ((ps >> 28) & 3) << 8; // DEFAULT_VAL
    match gs {
        Some(g) => v |= (g >> 8) & 0x1F,
        None => v |= 0x20,
    }
    if flat {
        v |= 1 << 10;
    }
    if f16 != 0 {
        v |= 1 << 19;
        v |= (f16 & 1) << 24 | ((f16 >> 1) & 1) << 25;
        v |= ((ps >> 30) & 3) << 21;
        // The second half falls back to its default when the producer does not
        // write it.
        let gs_f16 = gs.map(|g| (g >> 20) & 3).unwrap_or(0);
        if f16 & 2 != 0 && gs_f16 & 2 == 0 {
            v |= 1 << 20;
        }
    }
    v
}

/// Writes SPI_PS_INPUT_CNTL_0..31 for a GS/VS -> PS pair (identity for unused slots).
unsafe fn interpolant_mapping(regs: u64, gs: u64, ps: u64) -> i32 {
    use shader::*;
    unsafe {
        let n_in = if ps != 0 { rd32(ps + 0x50) } else { 0 };
        let ins = if ps != 0 { rd64(ps + INPUT_SEMANTICS) } else { 0 };
        let n_out = if gs != 0 { *((gs + 0x56) as *const u16) as u64 } else { 0 };
        let outs = if gs != 0 { rd64(gs + OUTPUT_SEMANTICS) } else { 0 };
        for i in 0..32u32 {
            let value = if i < n_in && ins != 0 {
                let p = rd32(ins + 4 * i as u64);
                let g = (0..n_out).map(|j| rd32(outs + 4 * j)).find(|&g| g & 0xFF == p & 0xFF);
                ps_input_cntl(p, g)
            } else {
                i
            };
            *((regs + 8 * i as u64) as *mut u32) = REG_SPI_PS_INPUT_CNTL_0 + i;
            *((regs + 8 * i as u64 + 4) as *mut u32) = value;
        }
    }
    0
}

sysv! {
    /// NID 23LRUSvYu1M (label "AgcInit" in public research; the plain
    /// `sceAgcInit` name does not hash to it). Called with (state, 13).
    fn ds_agc_init(state: u64, version: u32) -> i32 {
        crate::info!("agc: init(state {state:#x}, version {version})");
        0
    }
    fn sceAgcGetRegisterDefaults2(version: u32) -> u64 {
        let mut g = DEFAULTS_PTR.lock();
        if g.0 == 0 { g.0 = build_defaults(super::agc_defaults::PUBLIC, &format!("public (v{version})")); }
        g.0
    }
    fn sceAgcGetRegisterDefaults2Internal(version: u32) -> u64 {
        let mut g = DEFAULTS_PTR.lock();
        if g.1 == 0 { g.1 = build_defaults(super::agc_defaults::INTERNAL, &format!("internal (v{version})")); }
        g.1
    }
    /// Base PS5 (see sceKernelIsTrinityMode).
    fn sceAgcGetIsTrinityMode(out: *mut u8) {
        if !out.is_null() { *out = 0; }
    }

    /// Relocates a shader object in place, records its code address and rebases
    /// the program-address registers; returns the object through `out`.
    fn sceAgcCreateShader(out: *mut u64, h: u64, code: u64) -> i32 {
        use shader::*;
        if out.is_null() || h == 0 || code == 0 { return ERR_INVALID_SHADER_PROGRAM; }
        if rd32(h) != MAGIC || rd32(h + 4) != VERSION {
            crate::error!("agc: shader object {h:#x} has magic {:#x} version {:#x}", rd32(h), rd32(h + 4));
            return ERR_INVALID_SHADER_PROGRAM;
        }
        if code & 0xFF != 0 {
            crate::error!("agc: shader code {code:#x} is not 256-byte aligned");
            return ERR_INVALID_SHADER_PROGRAM;
        }
        for f in [USER_DATA, CX_REGS, SH_REGS, SPECIALS, INPUT_SEMANTICS, OUTPUT_SEMANTICS] {
            relocate(h + f);
        }
        let ud = rd64(h + USER_DATA);
        if ud != 0 {
            relocate(ud + UD_DIRECT);
            for i in 0..4 { relocate(ud + UD_SHARP + 8 * i); }
        }
        wr64(h + CODE, code);
        let stage = *((h + TYPE) as *const u8);
        if let Some(lo_reg) = program_register(stage) {
            let regs = rd64(h + SH_REGS);
            let n = *((h + NUM_SH) as *const u8) as u64;
            let mut patched = false;
            for i in 0..n {
                let e = regs + 8 * i;
                if rd32(e) == lo_reg && i + 1 < n && rd32(e + 8) == lo_reg + 1 {
                    let lo = rd32(e + 4) as u64;
                    let hi = rd32(e + 12);
                    let addr = code + ((lo << 8) | (((hi & 0xFF) as u64) << 40));
                    *((e + 4) as *mut u32) = (addr >> 8) as u32;
                    *((e + 12) as *mut u32) = (hi & !0xFF) | ((addr >> 40) as u32 & 0xFF);
                    patched = true;
                    break;
                }
            }
            if !patched {
                crate::error!("agc: shader {h:#x} (stage {stage}) has no program address register pair {lo_reg:#x}");
                return ERR_INVALID_SHADER_PROGRAM;
            }
        }
        crate::debug!("agc: shader {h:#x} stage {stage} code {code:#x} size {:#x}", rd32(h + SHADER_SIZE));
        log_shader_info(h);
        if let Some(lo_reg) = program_register(stage) {
            // Program address as programmed (code may hold data before it).
            let regs = rd64(h + SH_REGS);
            let n = *((h + NUM_SH) as *const u8) as u64;
            for i in 0..n.saturating_sub(1) {
                let e = regs + 8 * i;
                if rd32(e) == lo_reg {
                    let addr = ((rd32(e + 4) as u64) << 8) | (((rd32(e + 12) & 0xFF) as u64) << 40);
                    let size = (code + rd32(h + SHADER_SIZE) as u64).saturating_sub(addr);
                    let r = REGISTRATIONS.fetch_add(1, std::sync::atomic::Ordering::AcqRel) + 1;
                    SHADERS.lock().insert(addr, (stage, size as usize, instance_offset_slot_of(h, stage), r));
                    SHADERS_GEN.fetch_add(1, std::sync::atomic::Ordering::AcqRel);
                    if matches!(stage, 1 | 2) && crate::platform::graphics::precompile() > 0 {
                        crate::platform::graphics::created_graphics(addr);
                    }
                    if stage == 0 && crate::platform::graphics::precompile() > 0 {
                        // The registers a compute dispatch takes from it
                        // (`gpu::precompile`).
                        let reg = |r: u32| (0..n).map(|k| regs + 8 * k).find(|&a| rd32(a) == r).map(|a| rd32(a + 4));
                        let sp = rd64(h + SPECIALS);
                        let modifier = if sp != 0 { rd32(sp + specials::DISPATCH_MODIFIER) } else { 0 };
                        let threads = |r: u32| reg(r).unwrap_or(1) & 0xFFFF;
                        crate::platform::graphics::created(addr, reg(0x213).unwrap_or(0), [threads(0x207), threads(0x208), threads(0x209)], modifier);
                    }
                    break;
                }
            }
        }
        *out = h;
        0
    }

    fn sceAgcCreateInterpolantMapping(regs: u64, gs: u64, ps: u64) -> i32 {
        if regs == 0 { return ERR_INVALID_SHADER_PROGRAM; }
        interpolant_mapping(regs, gs, ps)
    }
    /// NID dbOlWdppb4o ("CreateInterpolantMapping2" in public research); the title
    /// passes (regs, gs, ps, flag) — the flag's meaning is not known yet.
    fn ds_agc_create_interpolant_mapping2(regs: u64, gs: u64, ps: u64, flag: u64) -> i32 {
        if regs == 0 { return ERR_INVALID_SHADER_PROGRAM; }
        interpolant_mapping(regs, gs, ps)
    }
    /// cx: {VGT_SHADER_STAGES_EN, VGT_GS_OUT_PRIM_TYPE}; uc: {GE_CNTL,
    /// GE_USER_VGPR_EN, VGT_PRIMITIVE_TYPE}, from the shaders' special registers.
    fn sceAgcCreatePrimState(cx: u64, uc: u64, hs: u64, gs: u64, prim: u32) -> i32 {
        use shader::SPECIALS;
        if cx == 0 && uc == 0 { return 0; }
        if gs == 0 { return ERR_INVALID_SHADER_PROGRAM; }
        let gsp = rd64(gs + SPECIALS);
        let hsp = if hs != 0 { rd64(hs + SPECIALS) } else { 0 };
        if gsp == 0 || (hs != 0 && hsp == 0) { return ERR_INVALID_SHADER_PROGRAM; }
        if cx != 0 {
            copy_reg(cx, gsp + specials::VGT_SHADER_STAGES_EN);
            let mut stages = rd32(cx + 4);
            if hsp != 0 { stages |= rd32(hsp + specials::VGT_SHADER_STAGES_EN + 4); }
            *((cx + 4) as *mut u32) = stages;
            if stages & 0x20 != 0 {
                copy_reg(cx + 8, gsp + specials::VGT_GS_OUT_PRIM_TYPE); // real GS: its own output type
            } else if hsp != 0 {
                copy_reg(cx + 8, hsp + specials::VGT_GS_OUT_PRIM_TYPE);
            } else {
                *((cx + 8) as *mut u32) = REG_VGT_GS_OUT_PRIM_TYPE;
                *((cx + 12) as *mut u32) = gs_out_prim(prim);
            }
        }
        if uc != 0 {
            copy_reg(uc, gsp + specials::GE_CNTL);
            copy_reg(uc + 8, if hsp != 0 { hsp } else { gsp } + specials::GE_USER_VGPR_EN);
            *((uc + 16) as *mut u32) = REG_VGT_PRIMITIVE_TYPE;
            *((uc + 20) as *mut u32) = prim;
        }
        0
    }
    fn sceAgcUpdatePrimState(cx: u64, uc: u64, prim: u32) -> i32 {
        if cx != 0 && rd32(cx + 4) & 0x24 == 0 {
            let v = rd32(cx + 12);
            *((cx + 12) as *mut u32) = (v & !7) | gs_out_prim(prim);
        }
        if uc != 0 { *((uc + 20) as *mut u32) = prim; }
        0
    }
}

pub fn register(r: &mut Registry) {
    r.func_nid("libSceAgc", "23LRUSvYu1M", "agc_init (23LRUSvYu1M)", ds_agc_init as *const () as u64);
    r.func_nid("libSceAgc", "dbOlWdppb4o", "agc_create_interpolant_mapping2 (dbOlWdppb4o)", ds_agc_create_interpolant_mapping2 as *const () as u64);
    reg!(r, "libSceAgc", [
        sceAgcGetRegisterDefaults2, sceAgcGetRegisterDefaults2Internal, sceAgcGetIsTrinityMode, sceAgcCreateShader,
        sceAgcCreateInterpolantMapping, sceAgcCreatePrimState, sceAgcUpdatePrimState,
    ]);
}
