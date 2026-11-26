//! AMD RDNA (gfx10) shader instruction decoder and disassembler.
//!
//! Encodings follow AMD's RDNA ISA; opcode names and numbers come from the
//! generated `tables` (Mesa ACO's opcode table, gfx10 column).

/// Hash containers with a fixed hasher (SipHash with fixed keys): iteration
/// order, and with it the emitted SPIR-V, is the same in every run, so the
/// driver's shader cache (keyed on the SPIR-V bytes) hits across runs.
pub type HashMap<K, V> = std::collections::HashMap<K, V, std::hash::BuildHasherDefault<std::collections::hash_map::DefaultHasher>>;
pub type HashSet<T> = std::collections::HashSet<T, std::hash::BuildHasherDefault<std::collections::hash_map::DefaultHasher>>;

/// A hash of this crate's source (`build.rs`): translations kept across
/// runs are kept under it.
pub const SOURCE_HASH: &str = env!("RDNA_SOURCE_HASH");

pub mod analysis;
pub mod capture;
pub mod emu;
pub mod format;
pub mod kernels;
pub mod translate;
pub mod tables;
pub mod uniform;
pub mod wavemask;

use std::fmt::Write;

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum Enc {
    Sop2,
    Sopk,
    Sop1,
    Sopc,
    Sopp,
    Smem,
    Vop2,
    Vop1,
    Vopc,
    Vop3,
    Vop3p,
    Vintrp,
    Ds,
    Mubuf,
    Mtbuf,
    Mimg,
    Exp,
    Flat,
    Global,
    Scratch,
}

/// Extra source modifier dword of VOP1/VOP2/VOPC.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Ext {
    None,
    /// DPP16 control dword (src0 = 250).
    Dpp16(u32),
    /// DPP8 lane selects (src0 = 233, or 234 with fetch-inactive).
    Dpp8(u32, bool),
    /// SDWA control dword (src0 = 249).
    Sdwa(u32),
}

#[derive(Clone, Copy, Debug)]
pub struct Inst {
    pub enc: Enc,
    /// Encoding-local opcode (VOP3: the full 10-bit VOP3 opcode).
    pub op: u16,
    /// Byte offset from the start of the program.
    pub pc: u32,
    /// Length in dwords including literal / DPP / SDWA / NSA dwords.
    pub len: u8,
    pub w0: u32,
    pub w1: u32,
    pub lit: Option<u32>,
    pub ext: Ext,
    /// MIMG NSA extra address VGPRs (up to 12).
    pub nsa: [u8; 12],
}

pub const OP_VCC_LO: u16 = 106;
pub const OP_VCC_HI: u16 = 107;
pub const OP_M0: u16 = 124;
pub const OP_NULL: u16 = 125;
pub const OP_EXEC_LO: u16 = 126;
pub const OP_EXEC_HI: u16 = 127;
pub const OP_SDWA: u16 = 249;
pub const OP_DPP16: u16 = 250;
pub const OP_DPP8: u16 = 233;
pub const OP_DPP8FI: u16 = 234;
pub const OP_VCCZ: u16 = 251;
pub const OP_EXECZ: u16 = 252;
pub const OP_SCC: u16 = 253;
pub const OP_LITERAL: u16 = 255;
pub const OP_VGPR0: u16 = 256;

#[derive(Debug)]
pub struct DecodeError(pub String);

pub fn bits(v: u32, lo: u32, n: u32) -> u32 {
    (v >> lo) & ((1u32 << n) - 1)
}

fn lookup(t: tables::Table, op: u16) -> Option<&'static str> {
    t.binary_search_by_key(&op, |(o, _)| *o).ok().map(|i| t[i].1)
}

/// Every encoding, in `Enc`'s order (`NAMES`).
const ENCS: [Enc; 20] = [
    Enc::Sop2,
    Enc::Sopk,
    Enc::Sop1,
    Enc::Sopc,
    Enc::Sopp,
    Enc::Smem,
    Enc::Vop2,
    Enc::Vop1,
    Enc::Vopc,
    Enc::Vop3,
    Enc::Vop3p,
    Enc::Vintrp,
    Enc::Ds,
    Enc::Mubuf,
    Enc::Mtbuf,
    Enc::Mimg,
    Enc::Exp,
    Enc::Flat,
    Enc::Global,
    Enc::Scratch,
];

/// Opcodes below this have their names in `NAMES` (VOP3's are 10 bits).
const NAME_OPS: usize = 1024;

/// `Inst::name` by encoding and opcode, made once from the tables (a binary
/// search and an alias split per call cost the recording threads ~4% of
/// their busy samples at an arrival, in decoding and per-program scans:
/// FINDINGS.md 2026-10-31).
static NAMES: std::sync::OnceLock<Vec<[Option<&'static str>; NAME_OPS]>> = std::sync::OnceLock::new();

fn names() -> &'static Vec<[Option<&'static str>; NAME_OPS]> {
    NAMES.get_or_init(|| ENCS.iter().map(|&enc| std::array::from_fn(|op| name_searched(enc, op as u16))).collect())
}

/// An instruction's name (first alias) from the tables themselves.
fn name_searched(enc: Enc, op: u16) -> Option<&'static str> {
    use tables::*;
    let n = match enc {
        Enc::Sop2 => lookup(SOP2, op),
        Enc::Sopk => lookup(SOPK, op),
        Enc::Sop1 => lookup(SOP1, op),
        Enc::Sopc => lookup(SOPC, op),
        Enc::Sopp => lookup(SOPP, op),
        Enc::Smem => lookup(SMEM, op),
        Enc::Vop2 => lookup(VOP2, op),
        Enc::Vop1 => lookup(VOP1, op),
        Enc::Vopc => lookup(VOPC, op),
        Enc::Vop3 => vop3_name(op),
        Enc::Vop3p => lookup(VOP3P, op),
        Enc::Vintrp => lookup(VINTRP, op),
        Enc::Ds => lookup(DS, op),
        Enc::Mubuf => lookup(MUBUF, op),
        Enc::Mtbuf => lookup(MTBUF, op),
        Enc::Mimg => lookup(MIMG, op),
        Enc::Exp => Some("exp"),
        Enc::Flat => lookup(FLAT, op),
        Enc::Global => lookup(GLOBAL, op),
        Enc::Scratch => lookup(SCRATCH, op),
    }?;
    Some(n.split('/').next().unwrap_or(n))
}

impl Inst {
    /// Instruction name (first alias), or None if the opcode is unknown.
    #[inline]
    pub fn name(&self) -> Option<&'static str> {
        match names()[self.enc as usize].get(self.op as usize) {
            Some(n) => *n,
            None => name_searched(self.enc, self.op),
        }
    }

    /// Name without ACO's `_e64` suffix (VOP3-only variants of VOP1/VOP2 ops):
    /// the key used for semantics.
    pub fn base_name(&self) -> Option<&'static str> {
        self.name().map(|n| n.strip_suffix("_e64").unwrap_or(n))
    }

    pub fn is_endpgm(&self) -> bool {
        self.enc == Enc::Sopp && self.op == 1
    }

    /// Branch target byte offset (SOPP branches): pc + 4 + simm16 * 4.
    pub fn branch_target(&self) -> Option<u32> {
        if self.enc != Enc::Sopp {
            return None;
        }
        let n = self.name()?;
        if n == "s_branch" || n.starts_with("s_cbranch_") {
            Some((self.pc as i64 + 4 + 4 * (self.simm16() as i16) as i64) as u32)
        } else {
            None
        }
    }

    /// VOP3B: carry/flag output SGPR in w0[14:8] instead of abs bits.
    pub fn is_vop3b(&self) -> bool {
        self.enc == Enc::Vop3
            && matches!(
                self.base_name(),
                Some(
                    "v_add_co_u32"
                        | "v_sub_co_u32"
                        | "v_subrev_co_u32"
                        | "v_addc_co_u32"
                        | "v_subb_co_u32"
                        | "v_subbrev_co_u32"
                        | "v_div_scale_f32"
                        | "v_div_scale_f64"
                        | "v_mad_u64_u32"
                        | "v_mad_i64_i32"
                )
            )
    }

    /// SGPR spills through VGPR lanes: `(is_write, vgpr, lane)` for
    /// v_writelane_b32 / v_readlane_b32 with an inline-constant lane.
    pub fn const_lane_op(&self) -> Option<(bool, u16, u32)> {
        let n = self.base_name()?;
        let write = match n {
            "v_writelane_b32" => true,
            "v_readlane_b32" => false,
            _ => return None,
        };
        let lane = self.src(1);
        if !(128..=192).contains(&lane) {
            return None;
        }
        let lane = (lane - 128) as u32;
        if write {
            Some((true, self.vdst(), lane))
        } else {
            let v = self.src(0);
            if v < OP_VGPR0 {
                return None;
            }
            Some((false, v - OP_VGPR0, lane))
        }
    }

    /// SDWA compare (VOPC) with an explicit SGPR destination (SD bit).
    pub fn sdwa_sdst(&self) -> Option<u16> {
        match (self.enc, self.ext) {
            (Enc::Vopc, Ext::Sdwa(d)) if bits(d, 15, 1) != 0 => Some(bits(d, 8, 7) as u16),
            _ => None,
        }
    }

    /// Resource descriptor base SGPR of buffer/image instructions.
    pub fn mem_srsrc(&self) -> u16 {
        match self.enc {
            Enc::Mimg => self.mimg_srsrc(),
            _ => self.buf_srsrc(),
        }
    }

    // Scalar fields
    pub fn sdst(&self) -> u16 {
        match self.enc {
            Enc::Smem => bits(self.w0, 6, 7) as u16,
            Enc::Vop3 => bits(self.w0, 8, 7) as u16,
            _ => bits(self.w0, 16, 7) as u16,
        }
    }
    pub fn ssrc0(&self) -> u16 {
        bits(self.w0, 0, 8) as u16
    }
    pub fn ssrc1(&self) -> u16 {
        bits(self.w0, 8, 8) as u16
    }
    pub fn simm16(&self) -> u16 {
        (self.w0 & 0xFFFF) as u16
    }
    /// SMEM base SGPR (pair index * 2).
    pub fn smem_sbase(&self) -> u16 {
        (bits(self.w0, 0, 6) * 2) as u16
    }
    /// SMEM signed 21-bit byte offset.
    pub fn smem_offset(&self) -> i32 {
        ((bits(self.w1, 0, 21) << 11) as i32) >> 11
    }
    pub fn smem_soffset(&self) -> u16 {
        bits(self.w1, 25, 7) as u16
    }
    pub fn glc(&self) -> bool {
        match self.enc {
            Enc::Smem => bits(self.w0, 16, 1) != 0,
            Enc::Flat | Enc::Global | Enc::Scratch => bits(self.w0, 16, 1) != 0,
            _ => bits(self.w0, 13, 1) != 0 && self.enc == Enc::Mimg || bits(self.w0, 14, 1) != 0 && matches!(self.enc, Enc::Mubuf | Enc::Mtbuf),
        }
    }
    // Vector ALU fields
    pub fn vdst(&self) -> u16 {
        match self.enc {
            Enc::Vop3 | Enc::Vop3p => bits(self.w0, 0, 8) as u16,
            Enc::Vintrp => bits(self.w0, 18, 8) as u16,
            Enc::Ds => bits(self.w1, 24, 8) as u16,
            Enc::Flat | Enc::Global | Enc::Scratch => bits(self.w1, 24, 8) as u16,
            Enc::Mubuf | Enc::Mtbuf | Enc::Mimg => bits(self.w1, 8, 8) as u16,
            _ => bits(self.w0, 17, 8) as u16,
        }
    }
    /// 9-bit source operand `i` (VOP1/VOP2/VOPC/VOP3/VOP3P).
    pub fn src(&self, i: usize) -> u16 {
        match self.enc {
            Enc::Vop3 | Enc::Vop3p => bits(self.w1, 9 * i as u32, 9) as u16,
            _ => {
                if i == 0 {
                    bits(self.w0, 0, 9) as u16
                } else {
                    OP_VGPR0 + bits(self.w0, 9, 8) as u16
                }
            }
        }
    }
    pub fn vop3_abs(&self) -> u32 {
        bits(self.w0, 8, 3)
    }
    pub fn vop3_neg(&self) -> u32 {
        bits(self.w1, 29, 3)
    }
    pub fn vop3_clamp(&self) -> bool {
        bits(self.w0, 15, 1) != 0
    }
    pub fn vop3_omod(&self) -> u32 {
        bits(self.w1, 27, 2)
    }
    pub fn vop3_opsel(&self) -> u32 {
        bits(self.w0, 11, 4)
    }
    /// VOP3B (carry-out) scalar destination.
    pub fn vop3_sdst(&self) -> u16 {
        bits(self.w0, 8, 7) as u16
    }
    /// MIMG fields.
    pub fn mimg_dmask(&self) -> u32 {
        bits(self.w0, 8, 4)
    }
    pub fn mimg_dim(&self) -> u32 {
        bits(self.w0, 3, 3)
    }
    pub fn mimg_unorm(&self) -> bool {
        bits(self.w0, 12, 1) != 0
    }
    pub fn mimg_srsrc(&self) -> u16 {
        (bits(self.w1, 16, 5) * 4) as u16
    }
    pub fn mimg_ssamp(&self) -> u16 {
        (bits(self.w1, 21, 5) * 4) as u16
    }
    /// TFE (bit 16) and LWE (bit 17): a residency status written after the data.
    pub fn mimg_tfe_lwe(&self) -> u32 {
        bits(self.w0, 16, 2)
    }
    pub fn mimg_a16(&self) -> bool {
        bits(self.w1, 30, 1) != 0
    }
    pub fn mimg_d16(&self) -> bool {
        bits(self.w1, 31, 1) != 0
    }
    /// Buffer fields (MUBUF/MTBUF).
    pub fn buf_offset(&self) -> u32 {
        bits(self.w0, 0, 12)
    }
    pub fn buf_offen(&self) -> bool {
        bits(self.w0, 12, 1) != 0
    }
    pub fn buf_idxen(&self) -> bool {
        bits(self.w0, 13, 1) != 0
    }
    pub fn buf_srsrc(&self) -> u16 {
        (bits(self.w1, 16, 5) * 4) as u16
    }
    pub fn buf_soffset(&self) -> u16 {
        bits(self.w1, 24, 8) as u16
    }
    pub fn buf_vaddr(&self) -> u16 {
        bits(self.w1, 0, 8) as u16
    }
    /// MTBUF unified format.
    pub fn mtbuf_format(&self) -> u32 {
        bits(self.w0, 19, 7)
    }
    /// Export fields.
    pub fn exp_en(&self) -> u32 {
        bits(self.w0, 0, 4)
    }
    pub fn exp_target(&self) -> u32 {
        bits(self.w0, 4, 6)
    }
    pub fn exp_compr(&self) -> bool {
        bits(self.w0, 10, 1) != 0
    }
    pub fn exp_done(&self) -> bool {
        bits(self.w0, 11, 1) != 0
    }
    pub fn exp_vm(&self) -> bool {
        bits(self.w0, 12, 1) != 0
    }
    pub fn exp_src(&self, i: usize) -> u16 {
        bits(self.w1, 8 * i as u32, 8) as u16
    }
    /// DS fields.
    pub fn ds_offset0(&self) -> u32 {
        bits(self.w0, 0, 8)
    }
    pub fn ds_offset1(&self) -> u32 {
        bits(self.w0, 8, 8)
    }
    pub fn ds_gds(&self) -> bool {
        bits(self.w0, 17, 1) != 0
    }
    pub fn ds_addr(&self) -> u16 {
        bits(self.w1, 0, 8) as u16
    }
    pub fn ds_data0(&self) -> u16 {
        bits(self.w1, 8, 8) as u16
    }
    pub fn ds_data1(&self) -> u16 {
        bits(self.w1, 16, 8) as u16
    }
    /// FLAT/GLOBAL/SCRATCH fields.
    pub fn flat_offset(&self) -> i32 {
        ((bits(self.w0, 0, 12) << 20) as i32) >> 20
    }
    pub fn flat_addr(&self) -> u16 {
        bits(self.w1, 0, 8) as u16
    }
    pub fn flat_data(&self) -> u16 {
        bits(self.w1, 8, 8) as u16
    }
    pub fn flat_saddr(&self) -> u16 {
        bits(self.w1, 16, 7) as u16
    }
    /// VINTRP fields.
    pub fn intrp_attr(&self) -> u32 {
        bits(self.w0, 10, 6)
    }
    pub fn intrp_chan(&self) -> u32 {
        bits(self.w0, 8, 2)
    }
    pub fn intrp_vsrc(&self) -> u16 {
        bits(self.w0, 0, 8) as u16
    }
}

/// VOP3 opcode space: VOP3-only ops, VOPC (0x000..), VOP2 (0x100..),
/// VOP1 (0x180..).
pub fn vop3_name(op: u16) -> Option<&'static str> {
    use tables::*;
    if let Some(n) = lookup(VOP3, op) {
        return Some(n);
    }
    match op {
        0x000..=0x0FF => lookup(VOPC, op),
        0x100..=0x13F => lookup(VOP2, op - 0x100),
        0x180..=0x1FF => lookup(VOP1, op - 0x180),
        _ => None,
    }
}

/// Which basic encoding a VOP3 opcode came from.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Vop3Kind {
    Native,
    FromVopc,
    FromVop2,
    FromVop1,
}

/// Whether an instruction reads other lanes' values: lane reads
/// (v_readlane, v_readfirstlane), DPP-modified VALU, permlanes, and the
/// lane-exchange LDS instructions (ds_swizzle, ds_permute, ds_bpermute).
pub fn cross_lane(i: &Inst) -> bool {
    matches!(i.ext, Ext::Dpp16(_) | Ext::Dpp8(..))
        || matches!(i.base_name(), Some("v_readlane_b32" | "v_readfirstlane_b32" | "v_permlane16_b32" | "v_permlanex16_b32" | "ds_swizzle_b32" | "ds_permute_b32" | "ds_bpermute_b32"))
}

pub fn vop3_kind(op: u16) -> Vop3Kind {
    if lookup(tables::VOP3, op).is_some() {
        return Vop3Kind::Native;
    }
    match op {
        0x000..=0x0FF => Vop3Kind::FromVopc,
        0x100..=0x13F => Vop3Kind::FromVop2,
        _ => Vop3Kind::FromVop1,
    }
}

fn uses_literal(srcs: &[u16]) -> bool {
    srcs.contains(&OP_LITERAL)
}

/// Decodes the instruction at dword index `i` of `code`.
pub fn decode(code: &[u32], i: usize) -> Result<Inst, DecodeError> {
    let w0 = *code.get(i).ok_or_else(|| DecodeError(format!("pc {:#x} past end", i * 4)))?;
    let next = |k: usize| code.get(i + k).copied().ok_or_else(|| DecodeError(format!("truncated instruction at {:#x}", i * 4)));
    let pc = (i * 4) as u32;
    let mut inst = Inst { enc: Enc::Sopp, op: 0, pc, len: 1, w0, w1: 0, lit: None, ext: Ext::None, nsa: [0; 12] };
    let top9 = w0 >> 23;
    let top6 = w0 >> 26;
    if top9 == 0x17D || top9 == 0x17E || top9 == 0x17F {
        inst.enc = match top9 {
            0x17D => Enc::Sop1,
            0x17E => Enc::Sopc,
            _ => Enc::Sopp,
        };
        inst.op = match inst.enc {
            Enc::Sop1 => bits(w0, 8, 8),
            Enc::Sopc => bits(w0, 16, 7),
            _ => bits(w0, 16, 7),
        } as u16;
        let srcs: Vec<u16> = match inst.enc {
            Enc::Sop1 => vec![inst.ssrc0()],
            Enc::Sopc => vec![inst.ssrc0(), inst.ssrc1()],
            _ => vec![],
        };
        if uses_literal(&srcs) {
            inst.lit = Some(next(1)?);
            inst.len = 2;
        }
    } else if w0 >> 28 == 0xB {
        inst.enc = Enc::Sopk;
        inst.op = bits(w0, 23, 5) as u16;
        // s_setreg_imm32_b32 carries a 32-bit immediate.
        if inst.name() == Some("s_setreg_imm32_b32") {
            inst.lit = Some(next(1)?);
            inst.len = 2;
        }
    } else if w0 >> 30 == 0b10 {
        inst.enc = Enc::Sop2;
        inst.op = bits(w0, 23, 7) as u16;
        if uses_literal(&[inst.ssrc0(), inst.ssrc1()]) {
            inst.lit = Some(next(1)?);
            inst.len = 2;
        }
    } else if w0 >> 31 == 0 {
        let top7 = w0 >> 25;
        inst.enc = match top7 {
            0x3F => Enc::Vop1,
            0x3E => Enc::Vopc,
            _ => Enc::Vop2,
        };
        inst.op = match inst.enc {
            Enc::Vop1 => bits(w0, 9, 8),
            Enc::Vopc => bits(w0, 17, 8),
            _ => bits(w0, 25, 6),
        } as u16;
        let s0 = inst.src(0);
        match s0 {
            OP_DPP16 => {
                inst.ext = Ext::Dpp16(next(1)?);
                inst.len = 2;
            }
            OP_DPP8 | OP_DPP8FI => {
                inst.ext = Ext::Dpp8(next(1)?, s0 == OP_DPP8FI);
                inst.len = 2;
            }
            OP_SDWA => {
                inst.ext = Ext::Sdwa(next(1)?);
                inst.len = 2;
            }
            _ => {
                // VOP2 madmk/madak/fmamk/fmaak always carry a literal.
                let always = matches!(inst.enc, Enc::Vop2) && inst.name().is_some_and(|n| n.ends_with("mk_f32") || n.ends_with("ak_f32") || n.ends_with("mk_f16") || n.ends_with("ak_f16"));
                if s0 == OP_LITERAL || always {
                    inst.lit = Some(next(1)?);
                    inst.len = 2;
                }
            }
        }
    } else {
        match top6 {
            0x3D => {
                inst.enc = Enc::Smem;
                inst.op = bits(w0, 18, 8) as u16;
                inst.w1 = next(1)?;
                inst.len = 2;
            }
            0x35 => {
                inst.enc = Enc::Vop3;
                inst.op = bits(w0, 16, 10) as u16;
                inst.w1 = next(1)?;
                inst.len = 2;
                if uses_literal(&[inst.src(0), inst.src(1), inst.src(2)]) {
                    inst.lit = Some(next(2)?);
                    inst.len = 3;
                }
            }
            0x33 if w0 >> 24 == 0xCC => {
                inst.enc = Enc::Vop3p;
                inst.op = bits(w0, 16, 7) as u16;
                inst.w1 = next(1)?;
                inst.len = 2;
                if uses_literal(&[inst.src(0), inst.src(1), inst.src(2)]) {
                    inst.lit = Some(next(2)?);
                    inst.len = 3;
                }
            }
            0x32 => {
                inst.enc = Enc::Vintrp;
                inst.op = bits(w0, 16, 2) as u16;
            }
            0x36 => {
                inst.enc = Enc::Ds;
                inst.op = bits(w0, 18, 8) as u16;
                inst.w1 = next(1)?;
                inst.len = 2;
            }
            0x38 => {
                inst.enc = Enc::Mubuf;
                inst.op = bits(w0, 18, 8) as u16;
                inst.w1 = next(1)?;
                inst.len = 2;
            }
            0x3A => {
                inst.enc = Enc::Mtbuf;
                inst.w1 = next(1)?;
                inst.op = (bits(w0, 16, 3) | (bits(inst.w1, 21, 1) << 3)) as u16;
                inst.len = 2;
            }
            0x3C => {
                inst.enc = Enc::Mimg;
                inst.op = (bits(w0, 18, 7) | (bits(w0, 0, 1) << 7)) as u16;
                inst.w1 = next(1)?;
                let nsa = bits(w0, 1, 2) as usize;
                inst.len = 2 + nsa as u8;
                for k in 0..nsa {
                    let w = next(2 + k)?;
                    for b in 0..4 {
                        inst.nsa[4 * k + b] = (w >> (8 * b)) as u8;
                    }
                }
            }
            0x3E => {
                inst.enc = Enc::Exp;
                inst.w1 = next(1)?;
                inst.len = 2;
            }
            0x37 => {
                inst.enc = match bits(w0, 14, 2) {
                    0 => Enc::Flat,
                    1 => Enc::Scratch,
                    _ => Enc::Global,
                };
                inst.op = bits(w0, 18, 7) as u16;
                inst.w1 = next(1)?;
                inst.len = 2;
            }
            _ => return Err(DecodeError(format!("unknown encoding {w0:#010x} at {pc:#x}"))),
        }
    }
    Ok(inst)
}

/// Decodes a whole program (stops after s_endpgm at top level of straight-line
/// code is reached and no branch targets lie beyond it).
pub fn decode_program(code: &[u32]) -> Result<Vec<Inst>, DecodeError> {
    let mut out = Vec::new();
    let mut i = 0usize;
    let mut max_target = 0usize;
    while i < code.len() {
        let inst = decode(code, i)?;
        i += inst.len as usize;
        if inst.enc == Enc::Sopp {
            if let Some(n) = inst.name() {
                if n.starts_with("s_branch") || n.starts_with("s_cbranch") {
                    let t = (i as i64 + (inst.simm16() as i16) as i64) as usize;
                    max_target = max_target.max(t);
                }
                let end = n == "s_endpgm";
                out.push(inst);
                if end && i > max_target {
                    break;
                }
                continue;
            }
        }
        out.push(inst);
    }
    Ok(out)
}

pub fn operand_name(o: u16) -> String {
    match o {
        0..=105 => format!("s{o}"),
        OP_VCC_LO => "vcc_lo".into(),
        OP_VCC_HI => "vcc_hi".into(),
        108..=123 => format!("ttmp{}", o - 108),
        OP_M0 => "m0".into(),
        OP_NULL => "null".into(),
        OP_EXEC_LO => "exec_lo".into(),
        OP_EXEC_HI => "exec_hi".into(),
        128 => "0".into(),
        129..=192 => format!("{}", o - 128),
        193..=208 => format!("-{}", o - 192),
        235 => "src_shared_base".into(),
        236 => "src_shared_limit".into(),
        237 => "src_private_base".into(),
        238 => "src_private_limit".into(),
        239 => "src_pops_exiting_wave_id".into(),
        240 => "0.5".into(),
        241 => "-0.5".into(),
        242 => "1.0".into(),
        243 => "-1.0".into(),
        244 => "2.0".into(),
        245 => "-2.0".into(),
        246 => "4.0".into(),
        247 => "-4.0".into(),
        248 => "0.15915494".into(),
        OP_SDWA => "sdwa".into(),
        OP_DPP16 => "dpp".into(),
        OP_DPP8 | OP_DPP8FI => "dpp8".into(),
        OP_VCCZ => "vccz".into(),
        OP_EXECZ => "execz".into(),
        OP_SCC => "scc".into(),
        OP_LITERAL => "lit".into(),
        256..=511 => format!("v{}", o - 256),
        _ => format!("?{o}"),
    }
}

/// Human-readable disassembly of one instruction.
pub fn disasm(inst: &Inst) -> String {
    let name = inst.name().unwrap_or("<unknown>");
    let mut s = format!("{:06x}: {name}", inst.pc);
    let lit = |o: u16| if o == OP_LITERAL { inst.lit.map(|l| format!("{l:#x}")).unwrap_or_else(|| "lit".into()) } else { operand_name(o) };
    match inst.enc {
        Enc::Sop2 => {
            let _ = write!(s, " {}, {}, {}", operand_name(inst.sdst()), lit(inst.ssrc0()), lit(inst.ssrc1()));
        }
        Enc::Sop1 => {
            let _ = write!(s, " {}, {}", operand_name(inst.sdst()), lit(inst.ssrc0()));
        }
        Enc::Sopc => {
            let _ = write!(s, " {}, {}", lit(inst.ssrc0()), lit(inst.ssrc1()));
        }
        Enc::Sopk => {
            let _ = write!(s, " {}, {:#x}", operand_name(inst.sdst()), inst.simm16());
            if let Some(l) = inst.lit {
                let _ = write!(s, ", {l:#x}");
            }
        }
        Enc::Sopp => {
            if name.contains("branch") {
                let t = inst.pc as i64 + 4 + 4 * (inst.simm16() as i16) as i64;
                let _ = write!(s, " {t:#x}");
            } else if inst.simm16() != 0 {
                let _ = write!(s, " {:#x}", inst.simm16());
            }
        }
        Enc::Smem => {
            let _ = write!(s, " {}, s[{}:{}], {:#x}", operand_name(inst.sdst()), inst.smem_sbase(), inst.smem_sbase() + 1, inst.smem_offset());
            if inst.smem_soffset() != OP_NULL {
                let _ = write!(s, " + {}", operand_name(inst.smem_soffset()));
            }
            if inst.glc() {
                s.push_str(" glc");
            }
        }
        Enc::Vop1 | Enc::Vop2 | Enc::Vopc => {
            let mut parts = Vec::new();
            if inst.enc != Enc::Vopc {
                parts.push(format!("v{}", inst.vdst()));
            }
            parts.push(lit(inst.src(0)));
            if inst.enc != Enc::Vop1 {
                parts.push(operand_name(inst.src(1)));
            }
            if inst.enc == Enc::Vop2 && name.contains("mk_") {
                parts.insert(2, inst.lit.map(|l| format!("{l:#x}")).unwrap_or_default());
            } else if inst.enc == Enc::Vop2 && name.contains("ak_") {
                parts.push(inst.lit.map(|l| format!("{l:#x}")).unwrap_or_default());
            }
            let _ = write!(s, " {}", parts.join(", "));
            match inst.ext {
                Ext::Dpp16(d) => {
                    let _ = write!(s, " dpp(src v{}, ctrl {:#x}, row_mask {:x}, bank_mask {:x})", bits(d, 0, 8), bits(d, 8, 9), bits(d, 28, 4), bits(d, 24, 4));
                }
                Ext::Dpp8(d, fi) => {
                    let _ = write!(s, " dpp8(src v{}, sel {:#x}{})", bits(d, 0, 8), d >> 8, if fi { ", fi" } else { "" });
                }
                Ext::Sdwa(d) => {
                    let _ = write!(s, " sdwa(src0 {}, {:#x})", operand_name(bits(d, 0, 8) as u16 + if bits(d, 23, 1) != 0 { 0 } else { OP_VGPR0 }), d);
                }
                Ext::None => {}
            }
        }
        Enc::Vop3 | Enc::Vop3p => {
            let kind = if inst.enc == Enc::Vop3 { vop3_kind(inst.op) } else { Vop3Kind::Native };
            let mut parts = Vec::new();
            // VOP3A: compares, v_readlane and v_readfirstlane write the SGPR
            // in the VDST field; VOP3B adds a carry-out SGPR in [14:8].
            let sgpr_dst = kind == Vop3Kind::FromVopc || matches!(inst.base_name(), Some("v_readlane_b32" | "v_readfirstlane_b32"));
            if sgpr_dst {
                parts.push(operand_name(inst.vdst()));
            } else {
                parts.push(format!("v{}", inst.vdst()));
                if inst.is_vop3b() {
                    parts.push(operand_name(inst.vop3_sdst()));
                }
            }
            for k in 0..3 {
                let o = inst.src(k);
                if k > 0 && o == 0 && inst.w1 >> (9 * k) & 0x1FF == 0 && k == 2 {
                    continue;
                }
                let mut t = lit(o);
                if inst.enc == Enc::Vop3 && kind != Vop3Kind::FromVopc && (inst.vop3_abs() >> k) & 1 != 0 {
                    t = format!("|{t}|");
                }
                if (inst.vop3_neg() >> k) & 1 != 0 {
                    t = format!("-{t}");
                }
                parts.push(t);
            }
            let _ = write!(s, " {}", parts.join(", "));
            if inst.vop3_clamp() {
                s.push_str(" clamp");
            }
            if inst.enc == Enc::Vop3 && inst.vop3_omod() != 0 {
                let _ = write!(s, " omod:{}", inst.vop3_omod());
            }
            if inst.vop3_opsel() != 0 {
                let _ = write!(s, " opsel:{:#x}", inst.vop3_opsel());
            }
        }
        Enc::Vintrp => {
            let _ = write!(s, " v{}, v{}, attr{}.{}", inst.vdst(), inst.intrp_vsrc(), inst.intrp_attr(), ["x", "y", "z", "w"][inst.intrp_chan() as usize]);
        }
        Enc::Ds => {
            let _ = write!(s, " v{}, v{}, v{}, v{} offset0:{} offset1:{}", inst.vdst(), inst.ds_addr(), inst.ds_data0(), inst.ds_data1(), inst.ds_offset0(), inst.ds_offset1());
            if inst.ds_gds() {
                s.push_str(" gds");
            }
        }
        Enc::Mubuf | Enc::Mtbuf => {
            let _ = write!(
                s,
                " v{}, v{}, s[{}:{}], {} offset:{}{}{}",
                inst.vdst(),
                inst.buf_vaddr(),
                inst.buf_srsrc(),
                inst.buf_srsrc() + 3,
                operand_name(inst.buf_soffset()),
                inst.buf_offset(),
                if inst.buf_offen() { " offen" } else { "" },
                if inst.buf_idxen() { " idxen" } else { "" }
            );
            if inst.enc == Enc::Mtbuf {
                let _ = write!(s, " format:{}", inst.mtbuf_format());
            }
            if inst.glc() {
                s.push_str(" glc");
            }
        }
        Enc::Mimg => {
            let mut addrs = vec![format!("v{}", bits(inst.w1, 0, 8))];
            let n = bits(inst.w0, 1, 2) as usize * 4;
            for k in 0..n {
                if inst.nsa[k] != 0 || k == 0 {
                    addrs.push(format!("v{}", inst.nsa[k]));
                }
            }
            let _ = write!(
                s,
                " v{}, [{}], s[{}:{}], s[{}:{}] dmask:{:#x} dim:{}{}{}{}",
                inst.vdst(),
                addrs.join(","),
                inst.mimg_srsrc(),
                inst.mimg_srsrc() + 7,
                inst.mimg_ssamp(),
                inst.mimg_ssamp() + 3,
                inst.mimg_dmask(),
                inst.mimg_dim(),
                if inst.mimg_unorm() { " unorm" } else { "" },
                if inst.mimg_a16() { " a16" } else { "" },
                if inst.mimg_d16() { " d16" } else { "" }
            );
        }
        Enc::Exp => {
            let t = inst.exp_target();
            let target = match t {
                0..=7 => format!("mrt{t}"),
                8 => "mrtz".into(),
                9 => "null".into(),
                12..=15 => format!("pos{}", t - 12),
                20 => "prim".into(),
                32..=63 => format!("param{}", t - 32),
                _ => format!("target{t}"),
            };
            let srcs: Vec<String> = (0..4).map(|k| if inst.exp_en() >> k & 1 != 0 { format!("v{}", inst.exp_src(k)) } else { "off".into() }).collect();
            let _ = write!(s, " {target}, {}{}{}{}", srcs.join(", "), if inst.exp_compr() { " compr" } else { "" }, if inst.exp_done() { " done" } else { "" }, if inst.exp_vm() { " vm" } else { "" });
        }
        Enc::Flat | Enc::Global | Enc::Scratch => {
            let _ = write!(
                s,
                " v{}, v{}, v{}, {} offset:{}",
                inst.vdst(),
                inst.flat_addr(),
                inst.flat_data(),
                operand_name(inst.flat_saddr()),
                inst.flat_offset()
            );
            if inst.glc() {
                s.push_str(" glc");
            }
        }
    }
    s
}

/// The title's own programs used as test inputs (its tile classification's kernels, a program
/// addressing a table in its code): game code, kept out of this repository. A test that uses
/// one reads it from local-data/research/rdna-tests/<name>.txt (hex words) and checks nothing
/// when the file is not there.
#[cfg(test)]
pub(crate) fn title_words(name: &str) -> Option<Vec<u32>> {
    let dir = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("../../local-data/research/rdna-tests");
    let Ok(t) = std::fs::read_to_string(dir.join(format!("{name}.txt"))) else {
        eprintln!("{name}: the title's program is not in local-data/research/rdna-tests: not checked");
        return None;
    };
    let words: Option<Vec<u32>> = t.split(|c: char| c.is_whitespace() || c == ',').filter(|s| !s.is_empty()).map(|s| u32::from_str_radix(s.trim_start_matches("0x").trim_start_matches("0X"), 16).ok()).collect();
    if words.is_none() {
        eprintln!("{name}: local-data/research/rdna-tests/{name}.txt holds something not a hex word: not checked");
    }
    words
}

#[cfg(test)]
mod name_tests {
    use super::*;

    /// `NAMES` is indexed by `Enc`'s discriminants and agrees with the
    /// tables for every opcode it holds.
    #[test]
    fn names_follow_the_tables() {
        for (i, &enc) in ENCS.iter().enumerate() {
            assert_eq!(enc as usize, i, "{enc:?}");
            for op in 0..NAME_OPS as u16 {
                assert_eq!(names()[i][op as usize], name_searched(enc, op), "{enc:?} {op:#x}");
            }
        }
        assert_eq!(names()[Enc::Sopp as usize][1], name_searched(Enc::Sopp, 1));
    }
}
