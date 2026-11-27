//! Scalar and vector ALU semantics, keyed by gfx10 instruction names.

use super::{e, ok, Stage, R, T};
use crate::{bits, Enc, Ext, Inst, OP_EXEC_HI, OP_EXEC_LO, OP_VCC_LO};
use rspirv::spirv::{self, GlslStd450Op as GLOp, Word};

// ---------------------------------------------------------------------------
// Shared helpers

/// 64-bit scalar operand: SGPR pair or a constant.
fn s64(tr: &mut T, ins: &Inst, op: u16) -> R<(Word, Word)> {
    if op < 128 {
        return Ok((tr.sreg(op as u32)?, tr.sreg(op as u32 + 1)?));
    }
    let lo = tr.operand(ins, op)?;
    // Inline integer constants are sign-extended; literals and float constants zero-extended.
    let hi = if (193..=208).contains(&op) { tr.u(0xFFFF_FFFF) } else { tr.u(0) };
    Ok((lo, hi))
}

fn set_s64(tr: &mut T, d: u16, lo: Word, hi: Word) -> R<()> {
    tr.set_sreg(d as u32, lo)?;
    tr.set_sreg(d as u32 + 1, hi)
}

fn to64(tr: &mut T, lo: Word, hi: Word) -> R<Word> {
    let v = e(tr.b.composite_construct(tr.t.v2u, None, [lo, hi]))?;
    e(tr.b.bitcast(tr.t.u64, None, v))
}

fn from64(tr: &mut T, v: Word) -> R<(Word, Word)> {
    let p = e(tr.b.bitcast(tr.t.v2u, None, v))?;
    Ok((e(tr.b.composite_extract(tr.t.u32, None, p, [0]))?, e(tr.b.composite_extract(tr.t.u32, None, p, [1]))?))
}

fn nonzero(tr: &mut T, v: Word) -> R<Word> {
    let z = tr.u(0);
    e(tr.b.i_not_equal(tr.t.bool_, None, v, z))
}

fn nonzero64(tr: &mut T, lo: Word, hi: Word) -> R<Word> {
    let o = e(tr.b.bitwise_or(tr.t.u32, None, lo, hi))?;
    nonzero(tr, o)
}

/// The f32 `x`'s bits without the sign, and the exponent field all ones, for the bit tests below.
fn abs_bits(tr: &mut T, x: Word) -> R<(Word, Word)> {
    let bits = tr.to_u(x)?;
    let m = tr.u(0x7fff_ffff);
    let abs = e(tr.b.bitwise_and(tr.t.u32, None, bits, m))?;
    Ok((abs, tr.u(0x7f80_0000)))
}

/// Whether the f32 `x` is a NaN (exponent all ones, fraction nonzero), read from its bits: a
/// driver free to assume no NaNs (a module without `SignedZeroInfNanPreserve`, the RTX 40
/// series and older here) may fold `OpIsNan` to false, never an integer test.
pub(crate) fn nan_bits(tr: &mut T, x: Word) -> R<Word> {
    let (abs, inf) = abs_bits(tr, x)?;
    e(tr.b.u_greater_than(tr.t.bool_, None, abs, inf))
}

/// Whether the f32 `x` is a NaN or an infinity (exponent all ones), from its bits as `nan_bits`.
pub(crate) fn nonfinite_bits(tr: &mut T, x: Word) -> R<Word> {
    let (abs, inf) = abs_bits(tr, x)?;
    e(tr.b.u_greater_than_equal(tr.t.bool_, None, abs, inf))
}

/// Whole-quad mode: each 4-bit group becomes 0xF if any bit is set.
fn wqm32(tr: &mut T, v: Word) -> R<Word> {
    let mut acc = tr.u(0);
    for q in 0..8 {
        let m = tr.u(0xF << (q * 4));
        let g = e(tr.b.bitwise_and(tr.t.u32, None, v, m))?;
        let nz = nonzero(tr, g)?;
        let z = tr.u(0);
        let sel = e(tr.b.select(tr.t.u32, None, nz, m, z))?;
        acc = e(tr.b.bitwise_or(tr.t.u32, None, acc, sel))?;
    }
    Ok(acc)
}

/// Quad mask: bit i set if any bit of 4-bit group i is set (8 groups per dword).
fn quadmask32(tr: &mut T, v: Word) -> R<Word> {
    let mut acc = tr.u(0);
    for q in 0..8 {
        let m = tr.u(0xF << (q * 4));
        let g = e(tr.b.bitwise_and(tr.t.u32, None, v, m))?;
        let nz = nonzero(tr, g)?;
        let bit = tr.u(1 << q);
        let z = tr.u(0);
        let sel = e(tr.b.select(tr.t.u32, None, nz, bit, z))?;
        acc = e(tr.b.bitwise_or(tr.t.u32, None, acc, sel))?;
    }
    Ok(acc)
}

/// Integer compare by mnemonic ("eq", "lg"/"ne", "gt", "ge", "lt", "le", "f", "t"/"tru").
fn int_cmp(tr: &mut T, a: Word, b: Word, cmp: &str, signed: bool) -> R<Option<Word>> {
    let bt = tr.t.bool_;
    Ok(Some(match (cmp, signed) {
        ("eq", _) => e(tr.b.i_equal(bt, None, a, b))?,
        ("lg" | "ne", _) => e(tr.b.i_not_equal(bt, None, a, b))?,
        ("gt", true) => e(tr.b.s_greater_than(bt, None, a, b))?,
        ("ge", true) => e(tr.b.s_greater_than_equal(bt, None, a, b))?,
        ("lt", true) => e(tr.b.s_less_than(bt, None, a, b))?,
        ("le", true) => e(tr.b.s_less_than_equal(bt, None, a, b))?,
        ("gt", false) => e(tr.b.u_greater_than(bt, None, a, b))?,
        ("ge", false) => e(tr.b.u_greater_than_equal(bt, None, a, b))?,
        ("lt", false) => e(tr.b.u_less_than(bt, None, a, b))?,
        ("le", false) => e(tr.b.u_less_than_equal(bt, None, a, b))?,
        ("f", _) => tr.bool_c(false),
        ("t" | "tru", _) => tr.bool_c(true),
        _ => return Ok(None),
    }))
}

/// Signed overflow of r = a + b.
fn add_overflow(tr: &mut T, a: Word, b: Word, r: Word) -> R<Word> {
    let u = tr.t.u32;
    let x1 = e(tr.b.bitwise_xor(u, None, a, r))?;
    let x2 = e(tr.b.bitwise_xor(u, None, b, r))?;
    let m = e(tr.b.bitwise_and(u, None, x1, x2))?;
    let s = tr.u(31);
    let top = e(tr.b.shift_right_logical(u, None, m, s))?;
    nonzero(tr, top)
}

/// Signed overflow of r = a - b.
fn sub_overflow(tr: &mut T, a: Word, b: Word, r: Word) -> R<Word> {
    let u = tr.t.u32;
    let x1 = e(tr.b.bitwise_xor(u, None, a, b))?;
    let x2 = e(tr.b.bitwise_xor(u, None, a, r))?;
    let m = e(tr.b.bitwise_and(u, None, x1, x2))?;
    let s = tr.u(31);
    let top = e(tr.b.shift_right_logical(u, None, m, s))?;
    nonzero(tr, top)
}

fn mul_hi(tr: &mut T, a: Word, b: Word, signed: bool) -> R<Word> {
    let u = tr.t.u32;
    if signed {
        let i = tr.t.i32;
        let st = tr.b.type_struct([i, i]);
        let (ai, bi) = (tr.to_i(a)?, tr.to_i(b)?);
        let r = e(tr.b.s_mul_extended(st, None, ai, bi))?;
        let hi = e(tr.b.composite_extract(i, None, r, [1]))?;
        tr.to_u(hi)
    } else {
        let st = tr.b.type_struct([u, u]);
        let r = e(tr.b.u_mul_extended(st, None, a, b))?;
        e(tr.b.composite_extract(u, None, r, [1]))
    }
}

/// ((1 << size) - 1) << offset, both masked to 5 bits.
fn bfm32(tr: &mut T, size: Word, offset: Word) -> R<Word> {
    let u = tr.t.u32;
    let k31 = tr.u(31);
    let one = tr.u(1);
    let sa = e(tr.b.bitwise_and(u, None, size, k31))?;
    let sb = e(tr.b.bitwise_and(u, None, offset, k31))?;
    let m = e(tr.b.shift_left_logical(u, None, one, sa))?;
    let m = e(tr.b.i_sub(u, None, m, one))?;
    e(tr.b.shift_left_logical(u, None, m, sb))
}

/// Bit-field extract with a 32-bit width that may be 32 or more (whole value).
fn bfe(tr: &mut T, v: Word, off: Word, width: Word, signed: bool) -> R<Word> {
    let u = tr.t.u32;
    let bt = tr.t.bool_;
    // OpBitField*Extract is undefined when offset + count > 32: clamp count.
    let k32 = tr.u(32);
    let room = e(tr.b.i_sub(u, None, k32, off))?;
    let w = tr.glsl2(u, GLOp::UMin, width, room)?;
    let r = if signed { e(tr.b.bit_field_s_extract(u, None, v, off, w))? } else { e(tr.b.bit_field_u_extract(u, None, v, off, w))? };
    let z = tr.u(0);
    let wz = e(tr.b.i_equal(bt, None, width, z))?;
    e(tr.b.select(u, None, wz, z, r))
}

// ---------------------------------------------------------------------------
// Scalar

pub fn scalar(tr: &mut T, ins: &Inst) -> R<()> {
    let name = ins.name().unwrap_or("");
    match ins.enc {
        Enc::Sopp => sopp(tr, ins, name),
        Enc::Sopk => sopk(tr, ins, name),
        Enc::Sopc => sopc(tr, ins, name),
        Enc::Sop1 => sop1(tr, ins, name),
        Enc::Sop2 => sop2(tr, ins, name),
        _ => Ok(()),
    }
}

fn unsupported(tr: &mut T, ins: &Inst) {
    tr.warn(format!("{:#x}: unsupported {}", ins.pc, ins.name().unwrap_or("<unknown>")));
}

fn sopp(tr: &mut T, ins: &Inst, name: &str) -> R<()> {
    match name {
        "s_nop" | "s_waitcnt" | "s_inst_prefetch" | "s_clause" | "s_sendmsg" | "s_sendmsghalt" | "s_sleep" | "s_setprio" | "s_icache_inv" | "s_incperflevel" | "s_decperflevel" | "s_ttracedata" | "s_ttracedata_imm" | "s_waitcnt_depctr" | "s_round_mode" | "s_denorm_mode" | "s_wait_idle" | "s_code_end" | "s_wakeup" => {}
        "s_barrier" => {
            if tr.spec.stage == Stage::Compute {
                let wg = tr.u(spirv::Scope::Workgroup as u32);
                let sem = tr.u((spirv::MemorySemantics::ACQUIRE_RELEASE | spirv::MemorySemantics::WORKGROUP_MEMORY | spirv::MemorySemantics::UNIFORM_MEMORY).bits());
                ok(tr.b.control_barrier(wg, wg, sem))?;
                tr.lds_state = 0;
            }
        }
        _ => unsupported(tr, ins),
    }
    Ok(())
}

fn sopk(tr: &mut T, ins: &Inst, name: &str) -> R<()> {
    let u = tr.t.u32;
    let d = ins.sdst() as u32;
    let imm_s = ins.simm16() as i16 as i32 as u32;
    let imm_u = ins.simm16() as u32;
    match name {
        "s_movk_i32" => {
            let v = tr.u(imm_s);
            tr.set_sreg(d, v)?;
        }
        "s_cmovk_i32" => {
            let c = tr.get_scc()?;
            let old = tr.sreg(d)?;
            let v = tr.u(imm_s);
            let r = e(tr.b.select(u, None, c, v, old))?;
            tr.set_sreg(d, r)?;
        }
        n if n.starts_with("s_cmpk_") => {
            let (cmp, ty) = n["s_cmpk_".len()..].rsplit_once('_').unwrap_or(("", ""));
            let signed = ty == "i32";
            let a = tr.sreg(d)?;
            let k = if signed { tr.u(imm_s) } else { tr.u(imm_u) };
            match int_cmp(tr, a, k, cmp, signed)? {
                Some(c) => tr.set_scc(c)?,
                None => unsupported(tr, ins),
            }
        }
        "s_addk_i32" => {
            let a = tr.sreg(d)?;
            let k = tr.u(imm_s);
            let r = e(tr.b.i_add(u, None, a, k))?;
            let ov = add_overflow(tr, a, k, r)?;
            tr.set_scc(ov)?;
            tr.set_sreg(d, r)?;
        }
        "s_mulk_i32" => {
            let a = tr.sreg(d)?;
            let k = tr.u(imm_s);
            let r = e(tr.b.i_mul(u, None, a, k))?;
            tr.set_sreg(d, r)?;
        }
        "s_getreg_b32" => {
            tr.warn(format!("{:#x}: s_getreg_b32 {:#x} returns 0", ins.pc, ins.simm16()));
            let z = tr.u(0);
            tr.set_sreg(d, z)?;
        }
        // Mode/trap registers are not modeled; waits are implicit.
        "s_setreg_b32" | "s_setreg_imm32_b32" | "s_waitcnt_vscnt" | "s_waitcnt_vmcnt" | "s_waitcnt_expcnt" | "s_waitcnt_lgkmcnt" | "s_version" => {}
        _ => unsupported(tr, ins),
    }
    Ok(())
}

fn sopc(tr: &mut T, ins: &Inst, name: &str) -> R<()> {
    let u = tr.t.u32;
    let bt = tr.t.bool_;
    let c = if let Some(rest) = name.strip_prefix("s_bitcmp") {
        // s_bitcmp{0,1}_b{32,64}: SCC = bit S1 of S0 is 0/1.
        let want_one = rest.starts_with('1');
        let wide = rest.ends_with("b64");
        let b = tr.operand(ins, ins.ssrc1())?;
        let set = if wide {
            let (lo, hi) = s64(tr, ins, ins.ssrc0())?;
            let a = to64(tr, lo, hi)?;
            let k = tr.u(63);
            let sh = e(tr.b.bitwise_and(u, None, b, k))?;
            let x = e(tr.b.shift_right_logical(tr.t.u64, None, a, sh))?;
            let x = e(tr.b.u_convert(u, None, x))?;
            let one = tr.u(1);
            let m = e(tr.b.bitwise_and(u, None, x, one))?;
            nonzero(tr, m)?
        } else {
            let a = tr.operand(ins, ins.ssrc0())?;
            let k = tr.u(31);
            let sh = e(tr.b.bitwise_and(u, None, b, k))?;
            let x = e(tr.b.shift_right_logical(u, None, a, sh))?;
            let one = tr.u(1);
            let m = e(tr.b.bitwise_and(u, None, x, one))?;
            nonzero(tr, m)?
        };
        if want_one { set } else { e(tr.b.logical_not(bt, None, set))? }
    } else if let Some(rest) = name.strip_prefix("s_cmp_") {
        let (cmp, ty) = rest.rsplit_once('_').unwrap_or(("", ""));
        if ty == "u64" {
            let (alo, ahi) = s64(tr, ins, ins.ssrc0())?;
            let (blo, bhi) = s64(tr, ins, ins.ssrc1())?;
            let a = to64(tr, alo, ahi)?;
            let b = to64(tr, blo, bhi)?;
            match cmp {
                "eq" => e(tr.b.i_equal(bt, None, a, b))?,
                _ => e(tr.b.i_not_equal(bt, None, a, b))?,
            }
        } else {
            let a = tr.operand(ins, ins.ssrc0())?;
            let b = tr.operand(ins, ins.ssrc1())?;
            match int_cmp(tr, a, b, cmp, ty == "i32")? {
                Some(c) => c,
                None => {
                    unsupported(tr, ins);
                    tr.bool_c(false)
                }
            }
        }
    } else {
        unsupported(tr, ins);
        return Ok(());
    };
    tr.set_scc(c)
}

/// Applies a 32-bit bitwise op by name stem ("and", "or", "xor", "andn2",
/// "orn2", "nand", "nor", "xnor", "andn1", "orn1").
fn bitop(tr: &mut T, stem: &str, a: Word, b: Word) -> R<Option<Word>> {
    let u = tr.t.u32;
    Ok(Some(match stem {
        "and" => e(tr.b.bitwise_and(u, None, a, b))?,
        "or" => e(tr.b.bitwise_or(u, None, a, b))?,
        "xor" => e(tr.b.bitwise_xor(u, None, a, b))?,
        "andn2" => {
            let nb = e(tr.b.not(u, None, b))?;
            e(tr.b.bitwise_and(u, None, a, nb))?
        }
        "orn2" => {
            let nb = e(tr.b.not(u, None, b))?;
            e(tr.b.bitwise_or(u, None, a, nb))?
        }
        "andn1" => {
            let na = e(tr.b.not(u, None, a))?;
            e(tr.b.bitwise_and(u, None, na, b))?
        }
        "orn1" => {
            let na = e(tr.b.not(u, None, a))?;
            e(tr.b.bitwise_or(u, None, na, b))?
        }
        "nand" => {
            let t = e(tr.b.bitwise_and(u, None, a, b))?;
            e(tr.b.not(u, None, t))?
        }
        "nor" => {
            let t = e(tr.b.bitwise_or(u, None, a, b))?;
            e(tr.b.not(u, None, t))?
        }
        "xnor" => {
            let t = e(tr.b.bitwise_xor(u, None, a, b))?;
            e(tr.b.not(u, None, t))?
        }
        _ => return Ok(None),
    }))
}

fn sop1(tr: &mut T, ins: &Inst, name: &str) -> R<()> {
    let u = tr.t.u32;
    let bt = tr.t.bool_;
    let d = ins.sdst();
    let s = ins.ssrc0();
    // s_*_saveexec_b{32,64} / s_andn{1,2}_wrexec_b{32,64}
    if let Some(stem) = name.strip_suffix("_saveexec_b64").or_else(|| name.strip_suffix("_saveexec_b32")).or_else(|| name.strip_suffix("_wrexec_b64")).or_else(|| name.strip_suffix("_wrexec_b32")) {
        let wide = name.ends_with("_b64");
        let wrexec = name.contains("_wrexec_");
        let stem = stem.trim_start_matches("s_");
        let (slo, shi) = if wide { s64(tr, ins, s)? } else { (tr.operand(ins, s)?, tr.u(0)) };
        let elo = tr.sreg(OP_EXEC_LO as u32)?;
        let ehi = tr.sreg(OP_EXEC_HI as u32)?;
        let (Some(nlo), Some(nhi)) = (bitop(tr, stem, slo, elo)?, bitop(tr, stem, shi, ehi)?) else {
            unsupported(tr, ins);
            return Ok(());
        };
        let nhi = if wide { nhi } else { ehi };
        let (dlo, dhi) = if wrexec { (nlo, nhi) } else { (elo, ehi) };
        if wide {
            set_s64(tr, d, dlo, dhi)?;
        } else {
            tr.set_sreg(d as u32, dlo)?;
        }
        tr.set_sreg(OP_EXEC_LO as u32, nlo)?;
        tr.set_sreg(OP_EXEC_HI as u32, nhi)?;
        let nz = if wide { nonzero64(tr, nlo, nhi)? } else { nonzero(tr, nlo)? };
        return tr.set_scc(nz);
    }
    if tr.sg.leaders.contains(&ins.pc) {
        // EXEC for an append's atomic (`Appends`): the subgroup's first
        // active invocation alone.
        let lead = tr.subgroup_leader()?;
        let lo = tr.bool_to_u(lead)?;
        tr.set_sreg(OP_EXEC_LO as u32, lo)?;
        if name == "s_mov_b64" {
            let z = tr.u(0);
            tr.set_sreg(OP_EXEC_HI as u32, z)?;
        }
        return Ok(());
    }
    // (Per lane, a constant lane mask moved into EXEC: this lane's bit of it, `native::exec_constant_slice`.)
    if super::native::exec_constant_slice(tr, ins, name)? {
        return Ok(());
    }
    match name {
        "s_mov_b32" => {
            let v = tr.operand(ins, s)?;
            tr.set_sreg(d as u32, v)?;
        }
        "s_mov_b64" => {
            let (lo, hi) = s64(tr, ins, s)?;
            set_s64(tr, d, lo, hi)?;
        }
        "s_cmov_b32" => {
            let c = tr.get_scc()?;
            let v = tr.operand(ins, s)?;
            let old = tr.sreg(d as u32)?;
            let r = e(tr.b.select(u, None, c, v, old))?;
            tr.set_sreg(d as u32, r)?;
        }
        "s_cmov_b64" => {
            let c = tr.get_scc()?;
            let (lo, hi) = s64(tr, ins, s)?;
            let (olo, ohi) = (tr.sreg(d as u32)?, tr.sreg(d as u32 + 1)?);
            let rl = e(tr.b.select(u, None, c, lo, olo))?;
            let rh = e(tr.b.select(u, None, c, hi, ohi))?;
            set_s64(tr, d, rl, rh)?;
        }
        "s_not_b32" => {
            let v = tr.operand(ins, s)?;
            let r = e(tr.b.not(u, None, v))?;
            let nz = nonzero(tr, r)?;
            tr.set_scc(nz)?;
            tr.set_sreg(d as u32, r)?;
        }
        "s_not_b64" => {
            let (lo, hi) = s64(tr, ins, s)?;
            let rl = e(tr.b.not(u, None, lo))?;
            let rh = e(tr.b.not(u, None, hi))?;
            let nz = nonzero64(tr, rl, rh)?;
            tr.set_scc(nz)?;
            set_s64(tr, d, rl, rh)?;
        }
        "s_wqm_b32" | "s_wqm_b64" => {
            let (lo, hi) = s64(tr, ins, s)?;
            let rl = wqm32(tr, lo)?;
            let wide = name.ends_with("b64");
            let rh = if wide { wqm32(tr, hi)? } else { tr.u(0) };
            let nz = nonzero64(tr, rl, rh)?;
            tr.set_scc(nz)?;
            if wide { set_s64(tr, d, rl, rh)? } else { tr.set_sreg(d as u32, rl)? }
        }
        "s_quadmask_b32" | "s_quadmask_b64" => {
            let (lo, hi) = s64(tr, ins, s)?;
            let ql = quadmask32(tr, lo)?;
            let r = if name.ends_with("b64") {
                let qh = quadmask32(tr, hi)?;
                let k8 = tr.u(8);
                let qh = e(tr.b.shift_left_logical(u, None, qh, k8))?;
                e(tr.b.bitwise_or(u, None, ql, qh))?
            } else {
                ql
            };
            let nz = nonzero(tr, r)?;
            tr.set_scc(nz)?;
            if name.ends_with("b64") {
                let z = tr.u(0);
                set_s64(tr, d, r, z)?;
            } else {
                tr.set_sreg(d as u32, r)?;
            }
        }
        "s_brev_b32" => {
            let v = tr.operand(ins, s)?;
            let r = e(tr.b.bit_reverse(u, None, v))?;
            tr.set_sreg(d as u32, r)?;
        }
        "s_brev_b64" => {
            let (lo, hi) = s64(tr, ins, s)?;
            let rl = e(tr.b.bit_reverse(u, None, hi))?;
            let rh = e(tr.b.bit_reverse(u, None, lo))?;
            set_s64(tr, d, rl, rh)?;
        }
        "s_bcnt0_i32_b32" | "s_bcnt1_i32_b32" => {
            let v = tr.operand(ins, s)?;
            let x = if name.starts_with("s_bcnt0") { e(tr.b.not(u, None, v))? } else { v };
            let r = e(tr.b.bit_count(u, None, x))?;
            let nz = nonzero(tr, r)?;
            tr.set_scc(nz)?;
            tr.set_sreg(d as u32, r)?;
        }
        "s_bcnt0_i32_b64" | "s_bcnt1_i32_b64" => {
            let (lo, hi) = s64(tr, ins, s)?;
            let (lo, hi) = if name.starts_with("s_bcnt0") { (e(tr.b.not(u, None, lo))?, e(tr.b.not(u, None, hi))?) } else { (lo, hi) };
            let a = e(tr.b.bit_count(u, None, lo))?;
            let b = e(tr.b.bit_count(u, None, hi))?;
            let r = e(tr.b.i_add(u, None, a, b))?;
            let nz = nonzero(tr, r)?;
            tr.set_scc(nz)?;
            tr.set_sreg(d as u32, r)?;
        }
        "s_ff0_i32_b32" | "s_ff1_i32_b32" => {
            let v = tr.operand(ins, s)?;
            let x = if name.starts_with("s_ff0") { e(tr.b.not(u, None, v))? } else { v };
            let r = tr.glsl1(tr.t.i32, GLOp::FindILsb, x)?;
            let r = tr.to_u(r)?;
            tr.set_sreg(d as u32, r)?;
        }
        "s_ff0_i32_b64" | "s_ff1_i32_b64" => {
            let (lo, hi) = s64(tr, ins, s)?;
            let (lo, hi) = if name.starts_with("s_ff0") { (e(tr.b.not(u, None, lo))?, e(tr.b.not(u, None, hi))?) } else { (lo, hi) };
            let a = tr.glsl1(tr.t.i32, GLOp::FindILsb, lo)?;
            let b = tr.glsl1(tr.t.i32, GLOp::FindILsb, hi)?;
            let a = tr.to_u(a)?;
            let b = tr.to_u(b)?;
            let k32 = tr.u(32);
            let b32 = e(tr.b.i_add(u, None, b, k32))?;
            let neg1 = tr.u(u32::MAX);
            let bvalid = e(tr.b.i_not_equal(bt, None, b, neg1))?;
            let bres = e(tr.b.select(u, None, bvalid, b32, neg1))?;
            let avalid = e(tr.b.i_not_equal(bt, None, a, neg1))?;
            let r = e(tr.b.select(u, None, avalid, a, bres))?;
            tr.set_sreg(d as u32, r)?;
        }
        "s_flbit_i32_b32" | "s_flbit_i32" => {
            // Bit index counted from the MSB of the first 1 (b32) or the first
            // bit differing from the sign (i32); -1 if none.
            let v = tr.operand(ins, s)?;
            let m = if name == "s_flbit_i32" {
                let vi = tr.to_i(v)?;
                tr.glsl1(tr.t.i32, GLOp::FindSMsb, vi)?
            } else {
                tr.glsl1(tr.t.i32, GLOp::FindUMsb, v)?
            };
            let m = tr.to_u(m)?;
            let k31 = tr.u(31);
            let r = e(tr.b.i_sub(u, None, k31, m))?;
            let neg1 = tr.u(u32::MAX);
            let none = e(tr.b.i_equal(bt, None, m, neg1))?;
            let r = e(tr.b.select(u, None, none, neg1, r))?;
            tr.set_sreg(d as u32, r)?;
        }
        "s_sext_i32_i8" | "s_sext_i32_i16" => {
            let v = tr.operand(ins, s)?;
            let z = tr.u(0);
            let w = tr.u(if name.ends_with("i8") { 8 } else { 16 });
            let r = e(tr.b.bit_field_s_extract(u, None, v, z, w))?;
            tr.set_sreg(d as u32, r)?;
        }
        "s_bitset0_b32" | "s_bitset1_b32" => {
            let v = tr.operand(ins, s)?;
            let one = tr.u(1);
            let k = tr.u(31);
            let idx = e(tr.b.bitwise_and(u, None, v, k))?;
            let bit = e(tr.b.shift_left_logical(u, None, one, idx))?;
            let old = tr.sreg(d as u32)?;
            let r = if name == "s_bitset0_b32" {
                let nb = e(tr.b.not(u, None, bit))?;
                e(tr.b.bitwise_and(u, None, old, nb))?
            } else {
                e(tr.b.bitwise_or(u, None, old, bit))?
            };
            tr.set_sreg(d as u32, r)?;
        }
        "s_bitset0_b64" | "s_bitset1_b64" => {
            let v = tr.operand(ins, s)?;
            let k = tr.u(63);
            let idx = e(tr.b.bitwise_and(u, None, v, k))?;
            let one = tr.u(1);
            let z = tr.u(0);
            let one64 = to64(tr, one, z)?;
            let bit = e(tr.b.shift_left_logical(tr.t.u64, None, one64, idx))?;
            let (olo, ohi) = (tr.sreg(d as u32)?, tr.sreg(d as u32 + 1)?);
            let old = to64(tr, olo, ohi)?;
            let r = if name == "s_bitset0_b64" {
                let nb = e(tr.b.not(tr.t.u64, None, bit))?;
                e(tr.b.bitwise_and(tr.t.u64, None, old, nb))?
            } else {
                e(tr.b.bitwise_or(tr.t.u64, None, old, bit))?
            };
            let (l, h) = from64(tr, r)?;
            set_s64(tr, d, l, h)?;
        }
        "s_getpc_b64" => {
            // Address of the next instruction: the program's address (from
            // the SGPR data) plus its offset.
            let (bl, bh) = tr.code_base()?;
            let base = to64(tr, bl, bh)?;
            let off_lo = tr.u(ins.pc as u32 + 4);
            let off_hi = tr.u(0);
            let off = to64(tr, off_lo, off_hi)?;
            let pc = e(tr.b.i_add(tr.t.u64, None, base, off))?;
            let (lo, hi) = from64(tr, pc)?;
            set_s64(tr, d, lo, hi)?;
        }
        "s_abs_i32" => {
            let v = tr.operand(ins, s)?;
            let i = tr.to_i(v)?;
            let a = tr.glsl1(tr.t.i32, GLOp::SAbs, i)?;
            let r = tr.to_u(a)?;
            let nz = nonzero(tr, r)?;
            tr.set_scc(nz)?;
            tr.set_sreg(d as u32, r)?;
        }
        "s_movrels_b32" | "s_movrels_b64" | "s_movreld_b32" | "s_movreld_b64" => {
            // Relative SGPR addressing through M0.
            let m0 = tr.sreg(crate::OP_M0 as u32)?;
            let n = if name.ends_with("b64") { 2 } else { 1 };
            let pt = tr.t.fn_u32;
            let k127 = tr.u(127);
            for k in 0..n {
                if name.starts_with("s_movrels") {
                    let base = tr.u(s as u32 + k);
                    let idx = e(tr.b.i_add(u, None, base, m0))?;
                    let idx = e(tr.b.bitwise_and(u, None, idx, k127))?;
                    let p = e(tr.b.access_chain(pt, None, tr.sgpr, [idx]))?;
                    let v = e(tr.b.load(u, None, p, None, []))?;
                    tr.set_sreg(d as u32 + k, v)?;
                } else {
                    let v = if s < 128 { tr.sreg(s as u32 + k)? } else { tr.operand(ins, s)? };
                    let base = tr.u(d as u32 + k);
                    let idx = e(tr.b.i_add(u, None, base, m0))?;
                    let idx = e(tr.b.bitwise_and(u, None, idx, k127))?;
                    let p = e(tr.b.access_chain(pt, None, tr.sgpr, [idx]))?;
                    ok(tr.b.store(p, v, None, []))?;
                }
            }
        }
        "s_bitreplicate_b64_b32" => {
            let v = tr.operand(ins, s)?;
            let mut lo = tr.u(0);
            let mut hi = tr.u(0);
            for i in 0..32u32 {
                let ki = tr.u(i);
                let one = tr.u(1);
                let b = e(tr.b.shift_right_logical(u, None, v, ki))?;
                let b = e(tr.b.bitwise_and(u, None, b, one))?;
                let three = tr.u(3);
                let pair = e(tr.b.i_mul(u, None, b, three))?;
                let sh = tr.u((2 * i) % 32);
                let p = e(tr.b.shift_left_logical(u, None, pair, sh))?;
                if i < 16 {
                    lo = e(tr.b.bitwise_or(u, None, lo, p))?;
                } else {
                    hi = e(tr.b.bitwise_or(u, None, hi, p))?;
                }
            }
            set_s64(tr, d, lo, hi)?;
        }
        _ => unsupported(tr, ins),
    }
    Ok(())
}

fn sop2(tr: &mut T, ins: &Inst, name: &str) -> R<()> {
    let u = tr.t.u32;
    let bt = tr.t.bool_;
    let d = ins.sdst();
    if tr.sg.lane0.contains(&ins.pc) {
        // An append's election (`s_and X, exec, 1`, `Appends`): lane 0's
        // bit alone, for the election's other uses (each invocation is not
        // lane 0 of a wave of its own here); SCC as before.
        let act = tr.active()?;
        let lane = tr.lane()?;
        let z = tr.u(0);
        let first = e(tr.b.i_equal(bt, None, lane, z))?;
        let c = e(tr.b.logical_and(bt, None, act, first))?;
        let lo = tr.bool_to_u(c)?;
        tr.set_scc(act)?;
        return if name.ends_with("_b64") { set_s64(tr, d, lo, z) } else { tr.set_sreg(d as u32, lo) };
    }
    // 64-bit bitwise ops.
    if let Some(stem) = name.strip_prefix("s_").and_then(|n| n.strip_suffix("_b64")) {
        if matches!(stem, "and" | "or" | "xor" | "andn2" | "orn2" | "nand" | "nor" | "xnor") {
            let (alo, ahi) = s64(tr, ins, ins.ssrc0())?;
            let (blo, bhi) = s64(tr, ins, ins.ssrc1())?;
            let rl = bitop(tr, stem, alo, blo)?.unwrap();
            let rh = bitop(tr, stem, ahi, bhi)?.unwrap();
            let nz = nonzero64(tr, rl, rh)?;
            tr.set_scc(nz)?;
            return set_s64(tr, d, rl, rh);
        }
    }
    match name {
        "s_cselect_b64" => {
            let (alo, ahi) = s64(tr, ins, ins.ssrc0())?;
            let (blo, bhi) = s64(tr, ins, ins.ssrc1())?;
            let c = tr.get_scc()?;
            let rl = e(tr.b.select(u, None, c, alo, blo))?;
            let rh = e(tr.b.select(u, None, c, ahi, bhi))?;
            return set_s64(tr, d, rl, rh);
        }
        "s_lshl_b64" | "s_lshr_b64" | "s_ashr_i64" => {
            let (alo, ahi) = s64(tr, ins, ins.ssrc0())?;
            let b = tr.operand(ins, ins.ssrc1())?;
            let a = to64(tr, alo, ahi)?;
            let k = tr.u(63);
            let sh = e(tr.b.bitwise_and(u, None, b, k))?;
            let r = match name {
                "s_lshl_b64" => e(tr.b.shift_left_logical(tr.t.u64, None, a, sh))?,
                "s_lshr_b64" => e(tr.b.shift_right_logical(tr.t.u64, None, a, sh))?,
                _ => e(tr.b.shift_right_arithmetic(tr.t.u64, None, a, sh))?,
            };
            let (l, h) = from64(tr, r)?;
            let nz = nonzero64(tr, l, h)?;
            tr.set_scc(nz)?;
            return set_s64(tr, d, l, h);
        }
        "s_bfm_b64" => {
            let a = tr.operand(ins, ins.ssrc0())?;
            let b = tr.operand(ins, ins.ssrc1())?;
            let one = tr.u(1);
            let z = tr.u(0);
            let one64 = to64(tr, one, z)?;
            let k = tr.u(63);
            let size = e(tr.b.bitwise_and(u, None, a, k))?;
            let off = e(tr.b.bitwise_and(u, None, b, k))?;
            let m = e(tr.b.shift_left_logical(tr.t.u64, None, one64, size))?;
            let m = e(tr.b.i_sub(tr.t.u64, None, m, one64))?;
            let r = e(tr.b.shift_left_logical(tr.t.u64, None, m, off))?;
            let (l, h) = from64(tr, r)?;
            return set_s64(tr, d, l, h);
        }
        "s_bfe_u64" | "s_bfe_i64" => {
            let (alo, ahi) = s64(tr, ins, ins.ssrc0())?;
            let b = tr.operand(ins, ins.ssrc1())?;
            let a = to64(tr, alo, ahi)?;
            let k63 = tr.u(63);
            let off = e(tr.b.bitwise_and(u, None, b, k63))?;
            let s16 = tr.u(16);
            let wsh = e(tr.b.shift_right_logical(u, None, b, s16))?;
            let k127 = tr.u(0x7F);
            let w = e(tr.b.bitwise_and(u, None, wsh, k127))?;
            let k64 = tr.u(64);
            let room = e(tr.b.i_sub(u, None, k64, off))?;
            let w = tr.glsl2(u, GLOp::UMin, w, room)?;
            let r = if name == "s_bfe_u64" { e(tr.b.bit_field_u_extract(tr.t.u64, None, a, off, w))? } else { e(tr.b.bit_field_s_extract(tr.t.u64, None, a, off, w))? };
            let (l, h) = from64(tr, r)?;
            let nz = nonzero64(tr, l, h)?;
            tr.set_scc(nz)?;
            return set_s64(tr, d, l, h);
        }
        _ => {}
    }
    let a = tr.operand(ins, ins.ssrc0())?;
    let b = tr.operand(ins, ins.ssrc1())?;
    let k31 = tr.u(31);
    let r: Word = match name {
        "s_add_u32" => {
            let r = e(tr.b.i_add(u, None, a, b))?;
            let c = e(tr.b.u_less_than(bt, None, r, a))?;
            tr.set_scc(c)?;
            r
        }
        "s_sub_u32" => {
            let r = e(tr.b.i_sub(u, None, a, b))?;
            let c = e(tr.b.u_less_than(bt, None, a, b))?;
            tr.set_scc(c)?;
            r
        }
        "s_add_i32" => {
            let r = e(tr.b.i_add(u, None, a, b))?;
            let ov = add_overflow(tr, a, b, r)?;
            tr.set_scc(ov)?;
            r
        }
        "s_sub_i32" => {
            let r = e(tr.b.i_sub(u, None, a, b))?;
            let ov = sub_overflow(tr, a, b, r)?;
            tr.set_scc(ov)?;
            r
        }
        "s_addc_u32" | "s_subb_u32" => {
            let c = tr.get_scc()?;
            let ci = tr.bool_to_u(c)?;
            if name == "s_addc_u32" {
                let t = e(tr.b.i_add(u, None, a, b))?;
                let r = e(tr.b.i_add(u, None, t, ci))?;
                let c1 = e(tr.b.u_less_than(bt, None, t, a))?;
                let c2 = e(tr.b.u_less_than(bt, None, r, t))?;
                let co = e(tr.b.logical_or(bt, None, c1, c2))?;
                tr.set_scc(co)?;
                r
            } else {
                let t = e(tr.b.i_sub(u, None, a, b))?;
                let r = e(tr.b.i_sub(u, None, t, ci))?;
                let c1 = e(tr.b.u_less_than(bt, None, a, b))?;
                let c2 = e(tr.b.u_less_than(bt, None, t, ci))?;
                let co = e(tr.b.logical_or(bt, None, c1, c2))?;
                tr.set_scc(co)?;
                r
            }
        }
        "s_min_i32" | "s_min_u32" | "s_max_i32" | "s_max_u32" => {
            let signed = name.ends_with("i32");
            let pick_a = if name.starts_with("s_min") { int_cmp(tr, a, b, "lt", signed)?.unwrap() } else { int_cmp(tr, a, b, "gt", signed)?.unwrap() };
            tr.set_scc(pick_a)?;
            e(tr.b.select(u, None, pick_a, a, b))?
        }
        "s_cselect_b32" => {
            let c = tr.get_scc()?;
            e(tr.b.select(u, None, c, a, b))?
        }
        "s_and_b32" | "s_or_b32" | "s_xor_b32" | "s_andn2_b32" | "s_orn2_b32" | "s_nand_b32" | "s_nor_b32" | "s_xnor_b32" => {
            let stem = &name[2..name.len() - 4];
            let r = bitop(tr, stem, a, b)?.unwrap();
            let nz = nonzero(tr, r)?;
            tr.set_scc(nz)?;
            r
        }
        "s_lshl_b32" | "s_lshr_b32" | "s_ashr_i32" => {
            let sh = e(tr.b.bitwise_and(u, None, b, k31))?;
            let r = match name {
                "s_lshl_b32" => e(tr.b.shift_left_logical(u, None, a, sh))?,
                "s_lshr_b32" => e(tr.b.shift_right_logical(u, None, a, sh))?,
                _ => e(tr.b.shift_right_arithmetic(u, None, a, sh))?,
            };
            let nz = nonzero(tr, r)?;
            tr.set_scc(nz)?;
            r
        }
        "s_bfm_b32" => bfm32(tr, a, b)?,
        "s_mul_i32" => e(tr.b.i_mul(u, None, a, b))?,
        "s_mul_hi_u32" => mul_hi(tr, a, b, false)?,
        "s_mul_hi_i32" => mul_hi(tr, a, b, true)?,
        "s_bfe_u32" | "s_bfe_i32" => {
            let off = e(tr.b.bitwise_and(u, None, b, k31))?;
            let s16 = tr.u(16);
            let wsh = e(tr.b.shift_right_logical(u, None, b, s16))?;
            let k7f = tr.u(0x7F);
            let w = e(tr.b.bitwise_and(u, None, wsh, k7f))?;
            let r = bfe(tr, a, off, w, name == "s_bfe_i32")?;
            let nz = nonzero(tr, r)?;
            tr.set_scc(nz)?;
            r
        }
        "s_absdiff_i32" => {
            let ai = tr.to_i(a)?;
            let bi = tr.to_i(b)?;
            let df = e(tr.b.i_sub(tr.t.i32, None, ai, bi))?;
            let ab = tr.glsl1(tr.t.i32, GLOp::SAbs, df)?;
            let r = tr.to_u(ab)?;
            let nz = nonzero(tr, r)?;
            tr.set_scc(nz)?;
            r
        }
        "s_lshl1_add_u32" | "s_lshl2_add_u32" | "s_lshl3_add_u32" | "s_lshl4_add_u32" => {
            // D = (S0 << n) + S1; SCC = unsigned carry out of the 64-bit sum.
            let n = (name.as_bytes()[6] - b'0') as u32;
            let z = tr.u(0);
            let a64 = to64(tr, a, z)?;
            let b64 = to64(tr, b, z)?;
            let kn = tr.u(n);
            let sh = e(tr.b.shift_left_logical(tr.t.u64, None, a64, kn))?;
            let sum = e(tr.b.i_add(tr.t.u64, None, sh, b64))?;
            let (l, h) = from64(tr, sum)?;
            let c = nonzero(tr, h)?;
            tr.set_scc(c)?;
            l
        }
        "s_pack_ll_b32_b16" | "s_pack_lh_b32_b16" | "s_pack_hh_b32_b16" => {
            let k16 = tr.u(16);
            let m = tr.u(0xFFFF);
            let lo = if name == "s_pack_hh_b32_b16" { e(tr.b.shift_right_logical(u, None, a, k16))? } else { e(tr.b.bitwise_and(u, None, a, m))? };
            let hi = if name == "s_pack_ll_b32_b16" {
                e(tr.b.shift_left_logical(u, None, b, k16))?
            } else {
                let mh = tr.u(0xFFFF_0000);
                e(tr.b.bitwise_and(u, None, b, mh))?
            };
            e(tr.b.bitwise_or(u, None, lo, hi))?
        }
        _ => {
            unsupported(tr, ins);
            return Ok(());
        }
    };
    tr.set_sreg(d as u32, r)
}

// ---------------------------------------------------------------------------
// Vector operand model

#[derive(Clone, Copy)]
struct Sdwa {
    dst_sel: u32,
    dst_unused: u32,
    src_sel: [u32; 2],
    sext: [bool; 2],
}

/// Normalized vector instruction.
struct V {
    vop3: bool,
    vop3p: bool,
    /// VGPR destination, or the SGPR destination of compares / readlane.
    dst: u16,
    src: [u16; 3],
    abs: u32,
    neg: u32,
    clamp: bool,
    omod: u32,
    opsel: u32,
    /// VOP3P op_sel_hi (3 bits) and neg_hi.
    opsel_hi: u32,
    neg_hi: u32,
    sdwa: Option<Sdwa>,
    /// DPP: source-0 value fetched from another lane, and the write enable.
    dpp: Option<(Word, Word)>,
    /// Compare / carry-out SGPR destination.
    sdst: u16,
    lit: Option<u32>,
}

/// Operand codes of SDWA sources: VGPR unless the S bit selects the scalar space.
fn sdwa_src(code: u32, scalar: bool) -> u16 {
    if scalar { code as u16 } else { crate::OP_VGPR0 + code as u16 }
}

fn prep(tr: &mut T, ins: &Inst) -> R<V> {
    let mut v = V {
        vop3: false,
        vop3p: false,
        dst: ins.vdst(),
        src: [ins.src(0), ins.src(1), 0],
        abs: 0,
        neg: 0,
        clamp: false,
        omod: 0,
        opsel: 0,
        opsel_hi: 0,
        neg_hi: 0,
        sdwa: None,
        dpp: None,
        sdst: OP_VCC_LO,
        lit: ins.lit,
    };
    match ins.enc {
        Enc::Vop3 => {
            v.vop3 = true;
            v.src = [ins.src(0), ins.src(1), ins.src(2)];
            v.neg = ins.vop3_neg();
            v.omod = ins.vop3_omod();
            if ins.is_vop3b() {
                v.sdst = ins.vop3_sdst();
            } else {
                v.abs = ins.vop3_abs();
                v.clamp = ins.vop3_clamp();
                v.opsel = ins.vop3_opsel();
                // Compares (and v_readlane) write the SGPR in the VDST field.
                v.sdst = ins.vdst();
            }
        }
        Enc::Vop3p => {
            v.vop3 = true;
            v.vop3p = true;
            v.src = [ins.src(0), ins.src(1), ins.src(2)];
            v.neg = bits(ins.w1, 29, 3);
            v.neg_hi = bits(ins.w0, 8, 3);
            v.opsel = bits(ins.w0, 11, 3);
            v.opsel_hi = bits(ins.w1, 27, 2) | (bits(ins.w0, 14, 1) << 2);
            v.clamp = bits(ins.w0, 15, 1) != 0;
        }
        _ => match ins.ext {
            Ext::Sdwa(d) => {
                v.src[0] = sdwa_src(bits(d, 0, 8), bits(d, 23, 1) != 0);
                if ins.enc != Enc::Vop1 {
                    v.src[1] = sdwa_src(bits(ins.w0, 9, 8), bits(d, 31, 1) != 0);
                }
                v.neg = bits(d, 20, 1) | (bits(d, 28, 1) << 1);
                v.abs = bits(d, 21, 1) | (bits(d, 29, 1) << 1);
                let (dst_sel, dst_unused) = if ins.enc == Enc::Vopc {
                    if let Some(sd) = ins.sdwa_sdst() {
                        v.sdst = sd;
                    }
                    (6, 0)
                } else {
                    v.clamp = bits(d, 13, 1) != 0;
                    v.omod = bits(d, 14, 2);
                    (bits(d, 8, 3), bits(d, 11, 2))
                };
                v.sdwa = Some(Sdwa { dst_sel, dst_unused, src_sel: [bits(d, 16, 3), bits(d, 24, 3)], sext: [bits(d, 19, 1) != 0, bits(d, 27, 1) != 0] });
            }
            Ext::Dpp16(d) => {
                v.src[0] = crate::OP_VGPR0 + bits(d, 0, 8) as u16;
                v.neg = bits(d, 20, 1) | (bits(d, 22, 1) << 1);
                v.abs = bits(d, 21, 1) | (bits(d, 23, 1) << 1);
                v.dpp = Some(dpp16(tr, ins, d)?);
            }
            Ext::Dpp8(d, fi) => {
                v.src[0] = crate::OP_VGPR0 + bits(d, 0, 8) as u16;
                v.dpp = Some(dpp8(tr, d, fi)?);
            }
            Ext::None => {}
        },
    }
    Ok(v)
}

/// Lane of the source for DPP16 controls within rows of 16 lanes, and
/// whether the row offset stays in range.
fn dpp16(tr: &mut T, ins: &Inst, d: u32) -> R<(Word, Word)> {
    let u = tr.t.u32;
    let bt = tr.t.bool_;
    let ctrl = bits(d, 8, 9);
    let fi = bits(d, 18, 1) != 0;
    let bc = bits(d, 19, 1) != 0;
    let row_mask = bits(d, 28, 4);
    let bank_mask = bits(d, 24, 4);
    let lane = tr.subgroup_lane()?;
    let k15 = tr.u(15);
    let l = e(tr.b.bitwise_and(u, None, lane, k15))?;
    let nk15 = tr.u(!15);
    let rowbase = e(tr.b.bitwise_and(u, None, lane, nk15))?;
    let t = tr.bool_c(true);
    let (src_l, in_range) = match ctrl {
        0x000..=0x0FF => {
            // quad_perm
            let k3 = tr.u(3);
            let q = e(tr.b.bitwise_and(u, None, l, k3))?;
            let qbase = e(tr.b.i_sub(u, None, l, q))?;
            let two = tr.u(2);
            let sh = e(tr.b.i_mul(u, None, q, two))?;
            let c = tr.u(ctrl);
            let sel = e(tr.b.shift_right_logical(u, None, c, sh))?;
            let sel = e(tr.b.bitwise_and(u, None, sel, k3))?;
            (e(tr.b.bitwise_or(u, None, qbase, sel))?, t)
        }
        0x101..=0x10F => {
            let n = tr.u(ctrl & 15);
            let s = e(tr.b.i_add(u, None, l, n))?;
            let ok_ = e(tr.b.u_less_than_equal(bt, None, s, k15))?;
            (e(tr.b.bitwise_and(u, None, s, k15))?, ok_)
        }
        0x111..=0x11F => {
            let n = tr.u(ctrl & 15);
            let ok_ = e(tr.b.u_greater_than_equal(bt, None, l, n))?;
            let s = e(tr.b.i_sub(u, None, l, n))?;
            (e(tr.b.bitwise_and(u, None, s, k15))?, ok_)
        }
        0x121..=0x12F => {
            let n = tr.u(ctrl & 15);
            let s = e(tr.b.i_sub(u, None, l, n))?;
            (e(tr.b.bitwise_and(u, None, s, k15))?, t)
        }
        0x140 => (e(tr.b.i_sub(u, None, k15, l))?, t),
        0x141 => {
            let k8 = tr.u(8);
            let k7 = tr.u(7);
            let hi = e(tr.b.bitwise_and(u, None, l, k8))?;
            let lo = e(tr.b.bitwise_and(u, None, l, k7))?;
            let m = e(tr.b.i_sub(u, None, k7, lo))?;
            (e(tr.b.bitwise_or(u, None, hi, m))?, t)
        }
        0x150..=0x15F => (tr.u(ctrl & 15), t),
        0x160..=0x16F => {
            let x = tr.u(ctrl & 15);
            (e(tr.b.bitwise_xor(u, None, l, x))?, t)
        }
        _ => {
            tr.warn(format!("{:#x}: unsupported DPP control {ctrl:#x}", ins.pc));
            (l, t)
        }
    };
    let src = e(tr.b.bitwise_or(u, None, rowbase, src_l))?;
    let raw = tr.vreg(bits(d, 0, 8))?;
    // Source lane enabled (EXEC bit of the emulated lane) unless fetch-inactive.
    let (val, live) = if fi { tr.lane_fetch(raw, src)? } else { tr.lane_fetch_active(raw, src)? };
    let valid = e(tr.b.logical_and(bt, None, in_range, live))?;
    let z = tr.u(0);
    let val = e(tr.b.select(u, None, valid, val, z))?;
    // Invalid sources disable the lane unless bound_ctrl substitutes zero.
    let mut enable = if bc { tr.bool_c(true) } else { valid };
    if row_mask != 0xF || bank_mask != 0xF {
        let hw = tr.lane()?;
        let k4 = tr.u(4);
        let row = e(tr.b.shift_right_logical(u, None, hw, k4))?;
        let k3 = tr.u(3);
        let row = e(tr.b.bitwise_and(u, None, row, k3))?;
        let rm = tr.u(row_mask);
        let rb = e(tr.b.shift_right_logical(u, None, rm, row))?;
        let two = tr.u(2);
        let bank = e(tr.b.shift_right_logical(u, None, l, two))?;
        let bm = tr.u(bank_mask);
        let bb = e(tr.b.shift_right_logical(u, None, bm, bank))?;
        let both = e(tr.b.bitwise_and(u, None, rb, bb))?;
        let one = tr.u(1);
        let both = e(tr.b.bitwise_and(u, None, both, one))?;
        let en = nonzero(tr, both)?;
        enable = e(tr.b.logical_and(bt, None, enable, en))?;
    }
    Ok((val, enable))
}

fn dpp8(tr: &mut T, d: u32, fi: bool) -> R<(Word, Word)> {
    let u = tr.t.u32;
    let lane = tr.subgroup_lane()?;
    let k7 = tr.u(7);
    let l = e(tr.b.bitwise_and(u, None, lane, k7))?;
    let base = e(tr.b.i_sub(u, None, lane, l))?;
    let three = tr.u(3);
    let sh = e(tr.b.i_mul(u, None, l, three))?;
    let sel = tr.u(d >> 8);
    let s = e(tr.b.shift_right_logical(u, None, sel, sh))?;
    let s = e(tr.b.bitwise_and(u, None, s, k7))?;
    let src = e(tr.b.bitwise_or(u, None, base, s))?;
    let raw = tr.vreg(bits(d, 0, 8))?;
    let (val, _) = if fi { tr.lane_fetch(raw, src)? } else { tr.lane_fetch_active(raw, src)? };
    let t = tr.bool_c(true);
    Ok((val, t))
}

/// Raw 32-bit value of source `k` (SDWA selection and DPP applied).
fn usrc(tr: &mut T, ins: &Inst, v: &V, k: usize) -> R<Word> {
    let raw = match (k, v.dpp) {
        (0, Some((val, _))) => val,
        _ => tr.operand(ins, v.src[k])?,
    };
    match v.sdwa {
        Some(s) if k < 2 && s.src_sel[k] != 6 => {
            let u = tr.t.u32;
            let sel = s.src_sel[k];
            let (off, w) = if sel < 4 { (sel * 8, 8) } else { ((sel - 4) * 16, 16) };
            let o = tr.u(off);
            let wk = tr.u(w);
            Ok(if s.sext[k] { e(tr.b.bit_field_s_extract(u, None, raw, o, wk))? } else { e(tr.b.bit_field_u_extract(u, None, raw, o, wk))? })
        }
        _ => Ok(raw),
    }
}

/// Float value of source `k` with abs/neg modifiers.
fn fsrc(tr: &mut T, ins: &Inst, v: &V, k: usize) -> R<Word> {
    let raw = usrc(tr, ins, v, k)?;
    let mut x = tr.to_f(raw)?;
    if v.abs >> k & 1 != 0 {
        x = tr.glsl1(tr.t.f32, GLOp::FAbs, x)?;
    }
    if v.neg >> k & 1 != 0 {
        x = e(tr.b.f_negate(tr.t.f32, None, x))?;
    }
    Ok(x)
}

/// Source `k` with abs/neg applied as sign-bit operations (moves, selects).
fn bsrc(tr: &mut T, ins: &Inst, v: &V, k: usize) -> R<Word> {
    let mut x = usrc(tr, ins, v, k)?;
    let u = tr.t.u32;
    if v.abs >> k & 1 != 0 {
        let m = tr.u(0x7FFF_FFFF);
        x = e(tr.b.bitwise_and(u, None, x, m))?;
    }
    if v.neg >> k & 1 != 0 {
        let m = tr.u(0x8000_0000);
        x = e(tr.b.bitwise_xor(u, None, x, m))?;
    }
    Ok(x)
}

/// 16-bit compares and integer arithmetic are translated (`DS_TR_OPS16=0`:
/// left untranslated as before 2026-10-12, for A/B runs).
fn ops16() -> bool {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ON.get_or_init(|| std::env::var("DS_TR_OPS16").ok().as_deref() != Some("0"))
}

/// 16-bit source `k` in the low bits (zero-extended): the low half, or the
/// high half with VOP3 op_sel (SDWA selections are applied by `usrc`);
/// inline float constants are their f16 encodings.
fn src16(tr: &mut T, ins: &Inst, v: &V, k: usize) -> R<Word> {
    const HALF_CONSTS: [u32; 9] = [0x3800, 0xB800, 0x3C00, 0xBC00, 0x4000, 0xC000, 0x4400, 0xC400, 0x3118];
    let u = tr.t.u32;
    let op = v.src[k];
    let raw = match op {
        240..=248 if !(k == 0 && v.dpp.is_some()) => tr.u(HALF_CONSTS[(op - 240) as usize]),
        _ => usrc(tr, ins, v, k)?,
    };
    let raw = if v.vop3 && v.opsel >> k & 1 != 0 {
        let k16 = tr.u(16);
        e(tr.b.shift_right_logical(u, None, raw, k16))?
    } else {
        raw
    };
    let m = tr.u(0xFFFF);
    e(tr.b.bitwise_and(u, None, raw, m))
}

/// 16-bit integer source `k` as a 32-bit value (sign-extended for i16).
fn isrc16(tr: &mut T, ins: &Inst, v: &V, k: usize, signed: bool) -> R<Word> {
    let x = src16(tr, ins, v, k)?;
    if !signed {
        return Ok(x);
    }
    let z = tr.u(0);
    let w = tr.u(16);
    e(tr.b.bit_field_s_extract(tr.t.u32, None, x, z, w))
}

/// f16 source `k` as f32 with abs/neg modifiers.
fn fsrc16(tr: &mut T, ins: &Inst, v: &V, k: usize) -> R<Word> {
    let h = src16(tr, ins, v, k)?;
    let mut x = half_to_f(tr, h, false)?;
    if v.abs >> k & 1 != 0 {
        x = tr.glsl1(tr.t.f32, GLOp::FAbs, x)?;
    }
    if v.neg >> k & 1 != 0 {
        x = e(tr.b.f_negate(tr.t.f32, None, x))?;
    }
    Ok(x)
}

/// Writes a 16-bit result: the low half of the destination, or the high half
/// with VOP3 op_sel[3]; gfx10 16-bit instructions keep the other half.
fn out16(tr: &mut T, v: &V, x: Word) -> R<()> {
    let u = tr.t.u32;
    let m = tr.u(0xFFFF);
    let x = e(tr.b.bitwise_and(u, None, x, m))?;
    if v.sdwa.is_some() {
        return uout(tr, v, x);
    }
    let old = tr.vreg(v.dst as u32)?;
    let r = if v.opsel & 8 != 0 {
        let k16 = tr.u(16);
        let hi = e(tr.b.shift_left_logical(u, None, x, k16))?;
        let lo = e(tr.b.bitwise_and(u, None, old, m))?;
        e(tr.b.bitwise_or(u, None, lo, hi))?
    } else {
        let mh = tr.u(0xFFFF_0000);
        let hi = e(tr.b.bitwise_and(u, None, old, mh))?;
        e(tr.b.bitwise_or(u, None, hi, x))?
    };
    uout(tr, v, r)
}

/// 64-bit vector source (VGPR pair, SGPR pair or constant).
fn vsrc64(tr: &mut T, ins: &Inst, op: u16) -> R<(Word, Word)> {
    if op >= 256 {
        let r = (op - 256) as u32;
        return Ok((tr.vreg(r)?, tr.vreg(r + 1)?));
    }
    s64(tr, ins, op)
}

/// Lane write enable: EXEC bit and DPP validity.
fn write_enable(tr: &mut T, v: &V) -> R<Word> {
    let act = tr.active()?;
    match v.dpp {
        Some((_, en)) => e(tr.b.logical_and(tr.t.bool_, None, act, en)),
        None => Ok(act),
    }
}

/// Writes a 32-bit result to the VGPR destination (SDWA destination
/// selection, lane predicate).
fn uout(tr: &mut T, v: &V, x: Word) -> R<()> {
    let u = tr.t.u32;
    let mut x = x;
    if let Some(s) = v.sdwa {
        if s.dst_sel != 6 {
            let (off, w) = if s.dst_sel < 4 { (s.dst_sel * 8, 8) } else { ((s.dst_sel - 4) * 16, 16) };
            let mask = ((1u64 << w) - 1) as u32;
            let km = tr.u(mask);
            let ko = tr.u(off);
            let field = e(tr.b.bitwise_and(u, None, x, km))?;
            let placed = e(tr.b.shift_left_logical(u, None, field, ko))?;
            x = match s.dst_unused {
                2 => {
                    // preserve the other bits of the destination
                    let old = tr.vreg(v.dst as u32)?;
                    let hole = tr.u(!(mask << off));
                    let kept = e(tr.b.bitwise_and(u, None, old, hole))?;
                    e(tr.b.bitwise_or(u, None, kept, placed))?
                }
                1 => {
                    // sign-extend the field into the bits above it
                    let z = tr.u(0);
                    let kw = tr.u(w);
                    let sx = e(tr.b.bit_field_s_extract(u, None, x, z, kw))?;
                    let sxp = e(tr.b.shift_left_logical(u, None, sx, ko))?;
                    let low = tr.u(((1u64 << off) - 1) as u32);
                    let sxp_hi = {
                        let nl = e(tr.b.not(u, None, low))?;
                        e(tr.b.bitwise_and(u, None, sxp, nl))?
                    };
                    sxp_hi
                }
                _ => placed,
            };
        }
    }
    // Inside a lane guard the lane is active (and DPP never is).
    if tr.lane_guard && v.dpp.is_none() {
        return tr.set_vreg_raw(v.dst as u32, x);
    }
    let en = write_enable(tr, v)?;
    let old = tr.vreg(v.dst as u32)?;
    let nv = e(tr.b.select(u, None, en, x, old))?;
    tr.set_vreg_raw(v.dst as u32, nv)
}

/// Writes a float result with output modifiers.
fn fout(tr: &mut T, v: &V, x: Word) -> R<()> {
    let mut x = x;
    let k = match v.omod {
        1 => Some(2.0),
        2 => Some(4.0),
        3 => Some(0.5),
        _ => None,
    };
    if let Some(k) = k {
        let kk = tr.f(k);
        x = e(tr.b.f_mul(tr.t.f32, None, x, kk))?;
    }
    if v.clamp {
        let z = tr.f(0.0);
        let o = tr.f(1.0);
        x = tr.glsl3(tr.t.f32, GLOp::NClamp, x, z, o)?;
    }
    let u = tr.to_u(x)?;
    uout(tr, v, u)
}

/// Writes a per-lane boolean as a lane mask (see `T::lane_mask`) to an SGPR
/// pair (VCC or an explicit destination; one SGPR in wave32).
fn write_mask(tr: &mut T, ins: &Inst, sdst: u16, c: Word) -> R<()> {
    let (lo, hi) = tr.lane_mask(ins.pc, c)?;
    tr.set_sreg(sdst as u32, lo)?;
    if tr.spec.wave64 {
        tr.set_sreg(sdst as u32 + 1, hi)?;
    }
    Ok(())
}

/// Lane bit of a mask operand (VCC or an SGPR pair).
fn read_mask(tr: &mut T, ins: &Inst, op: u16) -> R<Word> {
    let (lo, hi) = if tr.spec.wave64 { s64(tr, ins, op)? } else { (tr.operand(ins, op)?, tr.u(0)) };
    tr.mask_bit(lo, hi)
}

fn fcmp(tr: &mut T, a: Word, b: Word, c: &str) -> R<Option<Word>> {
    let bt = tr.t.bool_;
    Ok(Some(match c {
        "f" => tr.bool_c(false),
        "lt" => e(tr.b.f_ord_less_than(bt, None, a, b))?,
        "eq" => e(tr.b.f_ord_equal(bt, None, a, b))?,
        "le" => e(tr.b.f_ord_less_than_equal(bt, None, a, b))?,
        "gt" => e(tr.b.f_ord_greater_than(bt, None, a, b))?,
        "lg" => e(tr.b.f_ord_not_equal(bt, None, a, b))?,
        "ge" => e(tr.b.f_ord_greater_than_equal(bt, None, a, b))?,
        // (Ordered / unordered from the bits, `nan_bits`: v_cmp_u x, x is the title's NaN test,
        // and `OpUnordered` may be folded as `OpIsNan` may.)
        "o" | "u" => {
            let na = nan_bits(tr, a)?;
            let nb = nan_bits(tr, b)?;
            let un = e(tr.b.logical_or(bt, None, na, nb))?;
            if c == "u" {
                un
            } else {
                e(tr.b.logical_not(bt, None, un))?
            }
        }
        "nge" => e(tr.b.f_unord_less_than(bt, None, a, b))?,
        "nlg" => e(tr.b.f_unord_equal(bt, None, a, b))?,
        "ngt" => e(tr.b.f_unord_less_than_equal(bt, None, a, b))?,
        "nle" => e(tr.b.f_unord_greater_than(bt, None, a, b))?,
        "neq" => e(tr.b.f_unord_not_equal(bt, None, a, b))?,
        "nlt" => e(tr.b.f_unord_greater_than_equal(bt, None, a, b))?,
        "tru" => tr.bool_c(true),
        _ => return Ok(None),
    }))
}

/// v_cmp_class_f32: tests floating-point class bits.
fn fclass(tr: &mut T, x: Word, mask: Word) -> R<Word> {
    let bt = tr.t.bool_;
    let u = tr.t.u32;
    let bits_ = tr.to_u(x)?;
    let s31 = tr.u(31);
    let sign_b = e(tr.b.shift_right_logical(u, None, bits_, s31))?;
    let one = tr.u(1);
    let neg = e(tr.b.i_equal(bt, None, sign_b, one))?;
    let pos = e(tr.b.logical_not(bt, None, neg))?;
    // Every class from the bits, as the hardware reads them: the exponent field 0, 1-254 or 255
    // and the fraction zero or not. Neither a driver free to assume no NaNs nor one that
    // flushes denormals (an `FAbs` of one read as zero) changes the answer.
    let s23 = tr.u(23);
    let sh = e(tr.b.shift_right_logical(u, None, bits_, s23))?;
    let kff = tr.u(0xff);
    let ex = e(tr.b.bitwise_and(u, None, sh, kff))?;
    let kfrac = tr.u(0x7f_ffff);
    let fr = e(tr.b.bitwise_and(u, None, bits_, kfrac))?;
    let z = tr.u(0);
    let ex0 = e(tr.b.i_equal(bt, None, ex, z))?;
    let exmax = e(tr.b.i_equal(bt, None, ex, kff))?;
    let fr0 = e(tr.b.i_equal(bt, None, fr, z))?;
    let frn = e(tr.b.logical_not(bt, None, fr0))?;
    let nan = e(tr.b.logical_and(bt, None, exmax, frn))?;
    let qbit = tr.u(1 << 22);
    let qm = e(tr.b.bitwise_and(u, None, bits_, qbit))?;
    let quiet = nonzero(tr, qm)?;
    let snan = {
        let nq = e(tr.b.logical_not(bt, None, quiet))?;
        e(tr.b.logical_and(bt, None, nan, nq))?
    };
    let qnan = e(tr.b.logical_and(bt, None, nan, quiet))?;
    let isinf = e(tr.b.logical_and(bt, None, exmax, fr0))?;
    let iszero = e(tr.b.logical_and(bt, None, ex0, fr0))?;
    let normal = {
        let a = e(tr.b.logical_not(bt, None, ex0))?;
        let b = e(tr.b.logical_not(bt, None, exmax))?;
        e(tr.b.logical_and(bt, None, a, b))?
    };
    let denorm = e(tr.b.logical_and(bt, None, ex0, frn))?;
    let classes = [
        snan,
        qnan,
        e(tr.b.logical_and(bt, None, neg, isinf))?,
        e(tr.b.logical_and(bt, None, neg, normal))?,
        e(tr.b.logical_and(bt, None, neg, denorm))?,
        e(tr.b.logical_and(bt, None, neg, iszero))?,
        e(tr.b.logical_and(bt, None, pos, iszero))?,
        e(tr.b.logical_and(bt, None, pos, denorm))?,
        e(tr.b.logical_and(bt, None, pos, normal))?,
        e(tr.b.logical_and(bt, None, pos, isinf))?,
    ];
    let mut acc = tr.bool_c(false);
    for (i, c) in classes.iter().enumerate() {
        let bit = tr.u(1 << i);
        let mb = e(tr.b.bitwise_and(u, None, mask, bit))?;
        let on = nonzero(tr, mb)?;
        let t = e(tr.b.logical_and(bt, None, on, *c))?;
        acc = e(tr.b.logical_or(bt, None, acc, t))?;
    }
    Ok(acc)
}

/// Masks of the lanes below this one (lo, hi) for v_mbcnt.
fn lane_masks(tr: &mut T) -> R<(Word, Word)> {
    let u = tr.t.u32;
    let bt = tr.t.bool_;
    let lane = tr.lane()?;
    let one = tr.u(1);
    let k32 = tr.u(32);
    let k31 = tr.u(31);
    let lo_full = e(tr.b.u_greater_than_equal(bt, None, lane, k32))?;
    let shl = e(tr.b.bitwise_and(u, None, lane, k31))?;
    let m = e(tr.b.shift_left_logical(u, None, one, shl))?;
    let m = e(tr.b.i_sub(u, None, m, one))?;
    let all = tr.u(u32::MAX);
    let lo = e(tr.b.select(u, None, lo_full, all, m))?;
    let z = tr.u(0);
    let hi = e(tr.b.select(u, None, lo_full, m, z))?;
    Ok((lo, hi))
}

/// DX9-style multiply: 0 * anything = 0.
fn legacy_mul(tr: &mut T, a: Word, b: Word) -> R<Word> {
    let f = tr.t.f32;
    let bt = tr.t.bool_;
    let z = tr.f(0.0);
    let az = e(tr.b.f_ord_equal(bt, None, a, z))?;
    let bz = e(tr.b.f_ord_equal(bt, None, b, z))?;
    let any = e(tr.b.logical_or(bt, None, az, bz))?;
    let m = e(tr.b.f_mul(f, None, a, b))?;
    e(tr.b.select(f, None, any, z, m))
}

/// v_cube{id,sc,tc,ma}_f32.
fn cube_op(tr: &mut T, which: &str, x: Word, y: Word, z: Word) -> R<Word> {
    let f = tr.t.f32;
    let bt = tr.t.bool_;
    let ax = tr.glsl1(f, GLOp::FAbs, x)?;
    let ay = tr.glsl1(f, GLOp::FAbs, y)?;
    let az = tr.glsl1(f, GLOp::FAbs, z)?;
    let zero = tr.f(0.0);
    let zmaj = {
        let a = e(tr.b.f_ord_greater_than_equal(bt, None, az, ax))?;
        let b = e(tr.b.f_ord_greater_than_equal(bt, None, az, ay))?;
        e(tr.b.logical_and(bt, None, a, b))?
    };
    let ymaj = {
        let a = e(tr.b.f_ord_greater_than_equal(bt, None, ay, ax))?;
        let nz = e(tr.b.logical_not(bt, None, zmaj))?;
        e(tr.b.logical_and(bt, None, a, nz))?
    };
    let zneg = e(tr.b.f_ord_less_than(bt, None, z, zero))?;
    let yneg = e(tr.b.f_ord_less_than(bt, None, y, zero))?;
    let xneg = e(tr.b.f_ord_less_than(bt, None, x, zero))?;
    let nx = e(tr.b.f_negate(f, None, x))?;
    let ny = e(tr.b.f_negate(f, None, y))?;
    let nz = e(tr.b.f_negate(f, None, z))?;
    match which {
        "id" => {
            let (c0, c1, c2, c3, c4, c5) = (tr.f(0.0), tr.f(1.0), tr.f(2.0), tr.f(3.0), tr.f(4.0), tr.f(5.0));
            let zid = e(tr.b.select(f, None, zneg, c5, c4))?;
            let yid = e(tr.b.select(f, None, yneg, c3, c2))?;
            let xid = e(tr.b.select(f, None, xneg, c1, c0))?;
            let t = e(tr.b.select(f, None, ymaj, yid, xid))?;
            e(tr.b.select(f, None, zmaj, zid, t))
        }
        "sc" => {
            let zsc = e(tr.b.select(f, None, zneg, nx, x))?;
            let xsc = e(tr.b.select(f, None, xneg, z, nz))?;
            let t = e(tr.b.select(f, None, ymaj, x, xsc))?;
            e(tr.b.select(f, None, zmaj, zsc, t))
        }
        "tc" => {
            let ytc = e(tr.b.select(f, None, yneg, nz, z))?;
            let t = e(tr.b.select(f, None, ymaj, ytc, ny))?;
            e(tr.b.select(f, None, zmaj, ny, t))
        }
        _ => {
            let two = tr.f(2.0);
            let m = e(tr.b.select(f, None, ymaj, ay, ax))?;
            let m = e(tr.b.select(f, None, zmaj, az, m))?;
            e(tr.b.f_mul(f, None, m, two))
        }
    }
}

/// f16 half (0 low, 1 high) of a 32-bit value as f32.
pub(crate) fn half_to_f(tr: &mut T, raw: Word, hi: bool) -> R<Word> {
    let v = tr.glsl1(tr.t.v2f, GLOp::UnpackHalf2x16, raw)?;
    e(tr.b.composite_extract(tr.t.f32, None, v, [hi as u32]))
}

/// f32 -> f16 bits in the low 16 bits (high bits zero).
pub(crate) fn f_to_half(tr: &mut T, x: Word) -> R<Word> {
    let z = tr.f(0.0);
    let vv = e(tr.b.composite_construct(tr.t.v2f, None, [x, z]))?;
    tr.glsl1(tr.t.u32, GLOp::PackHalf2x16, vv)
}

// ---------------------------------------------------------------------------
// Vector

pub fn vector(tr: &mut T, ins: &Inst) -> R<()> {
    let name = ins.base_name().unwrap_or("");
    let v = prep(tr, ins)?;
    let f = tr.t.f32;
    let u = tr.t.u32;
    let i = tr.t.i32;
    let bt = tr.t.bool_;
    // Compares.
    if let Some(rest) = name.strip_prefix("v_cmpx_").or_else(|| name.strip_prefix("v_cmp_")) {
        let is_x = name.starts_with("v_cmpx_");
        let (cmp, ty) = rest.rsplit_once('_').unwrap_or(("", ""));
        let c = match (cmp, ty) {
            ("class", "f32") => {
                let a = fsrc(tr, ins, &v, 0)?;
                let m = usrc(tr, ins, &v, 1)?;
                Some(fclass(tr, a, m)?)
            }
            (_, "f32") => {
                let a = fsrc(tr, ins, &v, 0)?;
                let b = fsrc(tr, ins, &v, 1)?;
                fcmp(tr, a, b, cmp)?
            }
            (_, "i32" | "u32") => {
                let mut a = usrc(tr, ins, &v, 0)?;
                let mut b = usrc(tr, ins, &v, 1)?;
                // A lane-0 election: every invocation is lane 0 of its wave.
                match tr.elections.get(&ins.pc) {
                    Some(0) => a = tr.u(0),
                    Some(_) => b = tr.u(0),
                    None => {}
                }
                int_cmp(tr, a, b, cmp, ty == "i32")?
            }
            (_, "i64" | "u64") => {
                let (alo, ahi) = vsrc64(tr, ins, v.src[0])?;
                let (blo, bhi) = vsrc64(tr, ins, v.src[1])?;
                let a = to64(tr, alo, ahi)?;
                let b = to64(tr, blo, bhi)?;
                int_cmp(tr, a, b, cmp, ty == "i64")?
            }
            // 16-bit compares (f16 halves widen exactly to f32).
            (_, "f16") if cmp != "class" && ops16() => {
                let a = fsrc16(tr, ins, &v, 0)?;
                let b = fsrc16(tr, ins, &v, 1)?;
                fcmp(tr, a, b, cmp)?
            }
            (_, "i16" | "u16") if ops16() => {
                let a = isrc16(tr, ins, &v, 0, ty == "i16")?;
                let b = isrc16(tr, ins, &v, 1, ty == "i16")?;
                int_cmp(tr, a, b, cmp, ty == "i16")?
            }
            _ => None,
        };
        let Some(c) = c else {
            unsupported(tr, ins);
            return Ok(());
        };
        if is_x {
            // gfx10: v_cmpx writes EXEC only.
            let (lo, hi) = tr.lane_mask(ins.pc, c)?;
            tr.set_sreg(OP_EXEC_LO as u32, lo)?;
            tr.set_sreg(OP_EXEC_HI as u32, hi)?;
        } else {
            write_mask(tr, ins, v.sdst, c)?;
        }
        return Ok(());
    }
    match name {
        "v_nop" => {}
        "v_mov_b32" => {
            let a = bsrc(tr, ins, &v, 0)?;
            uout(tr, &v, a)?;
        }
        "v_movreld_b32" | "v_movrels_b32" | "v_movrelsd_b32" | "v_movrelsd_2_b32" => {
            // Relative VGPR addressing: index offsets from M0 (movrelsd_2:
            // source offset M0[9:0], destination offset M0[25:16]).
            let m0 = tr.sreg(crate::OP_M0 as u32)?;
            let (soff, doff) = match name {
                "v_movrelsd_2_b32" => {
                    let k = tr.u(0x3FF);
                    let k16 = tr.u(16);
                    let so = e(tr.b.bitwise_and(u, None, m0, k))?;
                    let d = e(tr.b.shift_right_logical(u, None, m0, k16))?;
                    let dof = e(tr.b.bitwise_and(u, None, d, k))?;
                    (Some(so), Some(dof))
                }
                "v_movrels_b32" => (Some(m0), None),
                "v_movreld_b32" => (None, Some(m0)),
                _ => (Some(m0), Some(m0)),
            };
            let pt = tr.t.fn_u32;
            let k255 = tr.u(255);
            let val = match (soff, v.src[0]) {
                (Some(off), sv) if sv >= 256 => {
                    let base = tr.u((sv - 256) as u32);
                    let idx = e(tr.b.i_add(u, None, base, off))?;
                    let idx = e(tr.b.bitwise_and(u, None, idx, k255))?;
                    let p = e(tr.b.access_chain(pt, None, tr.vgpr, [idx]))?;
                    e(tr.b.load(u, None, p, None, []))?
                }
                _ => bsrc(tr, ins, &v, 0)?,
            };
            match doff {
                Some(off) => {
                    let base = tr.u(v.dst as u32);
                    let idx = e(tr.b.i_add(u, None, base, off))?;
                    let idx = e(tr.b.bitwise_and(u, None, idx, k255))?;
                    let p = e(tr.b.access_chain(pt, None, tr.vgpr, [idx]))?;
                    let act = tr.active()?;
                    let old = e(tr.b.load(u, None, p, None, []))?;
                    let nv = e(tr.b.select(u, None, act, val, old))?;
                    ok(tr.b.store(p, nv, None, []))?;
                }
                None => uout(tr, &v, val)?,
            }
        }
        "v_cndmask_b32" => {
            let a = bsrc(tr, ins, &v, 0)?;
            let b = bsrc(tr, ins, &v, 1)?;
            let c = if v.vop3 { read_mask(tr, ins, v.src[2])? } else { read_mask(tr, ins, OP_VCC_LO)? };
            let r = e(tr.b.select(u, None, c, b, a))?;
            uout(tr, &v, r)?;
        }
        "v_readfirstlane_b32" => {
            // Where all lanes run the block together, the first active lane's
            // value, as hardware: loops that take a bit of the first lane's
            // mask and clear it in every lane end only then (a lane reading
            // its own empty mask toggles bit 31 forever while a lockstep vote
            // keeps the loop going). In divergent code each lane emulates its
            // own wave, so it reads its own value.
            // An atomic's result is the invocation's own (see
            // `atomic_result_reads`).
            // An append's result done per subgroup: its leader's.
            let a = usrc(tr, ins, &v, 0)?;
            let r = if tr.sg.results.contains(&ins.pc) {
                tr.leader_read(a)?
            } else if tr.uniformity.is_uniform(ins.pc) && !tr.own_lane_reads.contains(&ins.pc) {
                tr.wave_read_first(ins, a)?
            } else {
                a
            };
            tr.set_sreg(v.dst as u32, r)?;
        }
        "v_readlane_b32" => {
            // A lane written by v_writelane (SGPR spill) reads its slot;
            // otherwise the lane's value via a subgroup shuffle.
            let slot = ins.const_lane_op().and_then(|(_, g, l)| tr.lane_slots.get(&(g as u32, l)).copied());
            let a = match slot {
                Some(var) => e(tr.b.load(u, None, var, None, []))?,
                // An atomic's result: the invocation's own (see
                // `atomic_result_reads`), or its subgroup leader's
                // (`Appends`).
                None if tr.sg.results.contains(&ins.pc) => {
                    let x = tr.operand(ins, v.src[0])?;
                    tr.leader_read(x)?
                }
                None if tr.own_lane_reads.contains(&ins.pc) => tr.operand(ins, v.src[0])?,
                None => {
                    let x = tr.operand(ins, v.src[0])?;
                    let l = tr.operand(ins, v.src[1])?;
                    tr.wave_read(ins, x, l)?
                }
            };
            tr.set_sreg(v.dst as u32, a)?;
        }
        "v_writelane_b32" => {
            // Writes only that lane: the VGPR changes only in the invocation
            // emulating it (spill slots keep constant-lane values for readback).
            let a = tr.operand(ins, v.src[0])?;
            match ins.const_lane_op() {
                Some((_, g, l)) => {
                    if let Some(&var) = tr.lane_slots.get(&(g as u32, l)) {
                        ok(tr.b.store(var, a, None, []))?;
                    }
                    let lid = tr.lane()?;
                    let kl = tr.u(l);
                    let here = e(tr.b.i_equal(bt, None, lid, kl))?;
                    let old = tr.vreg(v.dst as u32)?;
                    let nv = e(tr.b.select(u, None, here, a, old))?;
                    tr.set_vreg_raw(v.dst as u32, nv)?;
                }
                None => {
                    let l = tr.operand(ins, v.src[1])?;
                    let lid = tr.lane()?;
                    let k63 = tr.u(63);
                    let l = e(tr.b.bitwise_and(u, None, l, k63))?;
                    let here = e(tr.b.i_equal(bt, None, lid, l))?;
                    let old = tr.vreg(v.dst as u32)?;
                    let nv = e(tr.b.select(u, None, here, a, old))?;
                    tr.set_vreg_raw(v.dst as u32, nv)?;
                }
            }
        }
        "v_permlane16_b32" | "v_permlanex16_b32" => {
            // Lane i of each row of 16 reads lane sel[i] of the same row
            // (permlane16) or of the other row of its 32-lane half (x16).
            let x = tr.vreg((v.src[0].saturating_sub(256)) as u32)?;
            let s1 = tr.operand(ins, v.src[1])?;
            let s2 = tr.operand(ins, v.src[2])?;
            let lane = tr.subgroup_lane()?;
            let k15 = tr.u(15);
            let l = e(tr.b.bitwise_and(u, None, lane, k15))?;
            let k8 = tr.u(8);
            let hi_half = e(tr.b.u_greater_than_equal(bt, None, l, k8))?;
            let selw = e(tr.b.select(u, None, hi_half, s2, s1))?;
            let k7 = tr.u(7);
            let li = e(tr.b.bitwise_and(u, None, l, k7))?;
            let k4 = tr.u(4);
            let sh = e(tr.b.i_mul(u, None, li, k4))?;
            let sel = e(tr.b.shift_right_logical(u, None, selw, sh))?;
            let sel = e(tr.b.bitwise_and(u, None, sel, k15))?;
            let nk15 = tr.u(!15);
            let rowbase = e(tr.b.bitwise_and(u, None, lane, nk15))?;
            let rowbase = if name == "v_permlanex16_b32" {
                let k16 = tr.u(16);
                e(tr.b.bitwise_xor(u, None, rowbase, k16))?
            } else {
                rowbase
            };
            let src = e(tr.b.bitwise_or(u, None, rowbase, sel))?;
            // op_sel[0] = fetch inactive, op_sel[1] = bound control (0 for
            // invalid sources; otherwise the destination is kept).
            let (fi, bc) = (v.opsel & 1 != 0, v.opsel & 2 != 0);
            let (r, valid) = if fi { tr.lane_fetch(x, src)? } else { tr.lane_fetch_active(x, src)? };
            let r = if bc {
                r
            } else {
                let old = tr.vreg(v.dst as u32)?;
                e(tr.b.select(u, None, valid, r, old))?
            };
            uout(tr, &v, r)?;
        }
        "v_swap_b32" => {
            let a = tr.vreg(v.dst as u32)?;
            let b = tr.operand(ins, v.src[0])?;
            uout(tr, &v, b)?;
            if v.src[0] >= 256 {
                let act = tr.active()?;
                let old = tr.vreg((v.src[0] - 256) as u32)?;
                let nv = e(tr.b.select(u, None, act, a, old))?;
                tr.set_vreg_raw((v.src[0] - 256) as u32, nv)?;
            }
        }
        // ---- float arithmetic
        "v_add_f32" | "v_sub_f32" | "v_subrev_f32" | "v_mul_f32" | "v_mul_legacy_f32" | "v_min_f32" | "v_max_f32" | "v_ldexp_f32" => {
            let a = fsrc(tr, ins, &v, 0)?;
            let r = if name == "v_ldexp_f32" {
                let b = usrc(tr, ins, &v, 1)?;
                let bi = tr.to_i(b)?;
                tr.glsl2(f, GLOp::Ldexp, a, bi)?
            } else {
                let b = fsrc(tr, ins, &v, 1)?;
                match name {
                    "v_add_f32" => e(tr.b.f_add(f, None, a, b))?,
                    "v_sub_f32" => e(tr.b.f_sub(f, None, a, b))?,
                    "v_subrev_f32" => e(tr.b.f_sub(f, None, b, a))?,
                    "v_mul_f32" => e(tr.b.f_mul(f, None, a, b))?,
                    "v_mul_legacy_f32" => legacy_mul(tr, a, b)?,
                    "v_min_f32" => tr.glsl2(f, GLOp::NMin, a, b)?,
                    _ => tr.glsl2(f, GLOp::NMax, a, b)?,
                }
            };
            fout(tr, &v, r)?;
        }
        "v_mac_f32" | "v_fmac_f32" | "v_fmac_legacy_f32" | "v_mac_legacy_f32" => {
            let a = fsrc(tr, ins, &v, 0)?;
            let b = fsrc(tr, ins, &v, 1)?;
            let dv = tr.vreg(v.dst as u32)?;
            let d = tr.to_f(dv)?;
            let r = match name {
                "v_fmac_f32" => tr.glsl3(f, GLOp::Fma, a, b, d)?,
                "v_mac_f32" => {
                    let m = e(tr.b.f_mul(f, None, a, b))?;
                    e(tr.b.f_add(f, None, m, d))?
                }
                _ => {
                    let m = legacy_mul(tr, a, b)?;
                    e(tr.b.f_add(f, None, m, d))?
                }
            };
            fout(tr, &v, r)?;
        }
        "v_madmk_f32" | "v_madak_f32" | "v_fmamk_f32" | "v_fmaak_f32" => {
            let a = fsrc(tr, ins, &v, 0)?;
            let b = fsrc(tr, ins, &v, 1)?;
            let k = tr.f(f32::from_bits(v.lit.unwrap_or(0)));
            let (x, y, z) = if name.ends_with("mk_f32") { (a, k, b) } else { (a, b, k) };
            let r = if name.starts_with("v_fma") {
                tr.glsl3(f, GLOp::Fma, x, y, z)?
            } else {
                let m = e(tr.b.f_mul(f, None, x, y))?;
                e(tr.b.f_add(f, None, m, z))?
            };
            fout(tr, &v, r)?;
        }
        "v_mad_f32" | "v_fma_f32" | "v_fma_legacy_f32" | "v_mad_legacy_f32" | "v_div_fmas_f32" => {
            let a = fsrc(tr, ins, &v, 0)?;
            let b = fsrc(tr, ins, &v, 1)?;
            let c = fsrc(tr, ins, &v, 2)?;
            let r = match name {
                "v_mad_f32" => {
                    let m = e(tr.b.f_mul(f, None, a, b))?;
                    e(tr.b.f_add(f, None, m, c))?
                }
                "v_fma_f32" | "v_div_fmas_f32" => tr.glsl3(f, GLOp::Fma, a, b, c)?,
                _ => {
                    let m = legacy_mul(tr, a, b)?;
                    e(tr.b.f_add(f, None, m, c))?
                }
            };
            fout(tr, &v, r)?;
        }
        "v_min3_f32" | "v_max3_f32" | "v_med3_f32" => {
            let a = fsrc(tr, ins, &v, 0)?;
            let b = fsrc(tr, ins, &v, 1)?;
            let c = fsrc(tr, ins, &v, 2)?;
            let r = match name {
                "v_min3_f32" => {
                    let m = tr.glsl2(f, GLOp::NMin, a, b)?;
                    tr.glsl2(f, GLOp::NMin, m, c)?
                }
                "v_max3_f32" => {
                    let m = tr.glsl2(f, GLOp::NMax, a, b)?;
                    tr.glsl2(f, GLOp::NMax, m, c)?
                }
                _ => {
                    // As the hardware does it (AMD's RDNA2 ISA reference, V_MED3_F32): with a
                    // NaN among the inputs the result is V_MIN3_F32 of them, the smallest of
                    // the others; else the median. The median of NMin/NMax alone gave the
                    // first input's partner (a NaN first or second) or the larger of the first
                    // two (a NaN third): the fog's slice clamp med3(71.0, x, 0) made a NaN x
                    // slice 71 where the console makes it 0.
                    let mn = tr.glsl2(f, GLOp::NMin, a, b)?;
                    let mx = tr.glsl2(f, GLOp::NMax, a, b)?;
                    let m2 = tr.glsl2(f, GLOp::NMin, mx, c)?;
                    let med = tr.glsl2(f, GLOp::NMax, mn, m2)?;
                    let min3 = tr.glsl2(f, GLOp::NMin, mn, c)?;
                    let bt = tr.t.bool_;
                    let na = nan_bits(tr, a)?;
                    let nb = nan_bits(tr, b)?;
                    let nc = nan_bits(tr, c)?;
                    let n1 = e(tr.b.logical_or(bt, None, na, nb))?;
                    let n = e(tr.b.logical_or(bt, None, n1, nc))?;
                    e(tr.b.select(f, None, n, min3, med))?
                }
            };
            fout(tr, &v, r)?;
        }
        "v_fract_f32" | "v_trunc_f32" | "v_ceil_f32" | "v_rndne_f32" | "v_floor_f32" | "v_exp_f32" | "v_log_f32" | "v_rcp_f32" | "v_rcp_iflag_f32" | "v_rsq_f32" | "v_sqrt_f32" | "v_sin_f32" | "v_cos_f32" | "v_frexp_mant_f32" => {
            let a = fsrc(tr, ins, &v, 0)?;
            let r = match name {
                "v_fract_f32" => tr.glsl1(f, GLOp::Fract, a)?,
                "v_trunc_f32" => tr.glsl1(f, GLOp::Trunc, a)?,
                "v_ceil_f32" => tr.glsl1(f, GLOp::Ceil, a)?,
                "v_rndne_f32" => tr.glsl1(f, GLOp::RoundEven, a)?,
                "v_floor_f32" => tr.glsl1(f, GLOp::Floor, a)?,
                "v_exp_f32" => tr.glsl1(f, GLOp::Exp2, a)?,
                "v_log_f32" => tr.glsl1(f, GLOp::Log2, a)?,
                "v_rcp_f32" | "v_rcp_iflag_f32" => {
                    let one = tr.f(1.0);
                    e(tr.b.f_div(f, None, one, a))?
                }
                "v_rsq_f32" => tr.glsl1(f, GLOp::InverseSqrt, a)?,
                "v_sqrt_f32" => tr.glsl1(f, GLOp::Sqrt, a)?,
                "v_sin_f32" | "v_cos_f32" => {
                    // Input in revolutions.
                    let two_pi = tr.f(std::f32::consts::TAU);
                    let x = e(tr.b.f_mul(f, None, a, two_pi))?;
                    tr.glsl1(f, if name == "v_sin_f32" { GLOp::Sin } else { GLOp::Cos }, x)?
                }
                _ => {
                    let st = tr.b.type_struct([f, i]);
                    let fr = e(tr.b.ext_inst(st, None, tr.glsl, GLOp::FrexpStruct as u32, [rspirv::dr::Operand::IdRef(a)]))?;
                    e(tr.b.composite_extract(f, None, fr, [0]))?
                }
            };
            fout(tr, &v, r)?;
        }
        "v_frexp_exp_i32_f32" => {
            let a = fsrc(tr, ins, &v, 0)?;
            let st = tr.b.type_struct([f, i]);
            let fr = e(tr.b.ext_inst(st, None, tr.glsl, GLOp::FrexpStruct as u32, [rspirv::dr::Operand::IdRef(a)]))?;
            let ex = e(tr.b.composite_extract(i, None, fr, [1]))?;
            let r = tr.to_u(ex)?;
            uout(tr, &v, r)?;
        }
        "v_cubeid_f32" | "v_cubesc_f32" | "v_cubetc_f32" | "v_cubema_f32" => {
            let x = fsrc(tr, ins, &v, 0)?;
            let y = fsrc(tr, ins, &v, 1)?;
            let z = fsrc(tr, ins, &v, 2)?;
            let r = cube_op(tr, &name[6..8], x, y, z)?;
            fout(tr, &v, r)?;
        }
        "v_div_scale_f32" => {
            // No scaling needed: pass src0 through; VCC = 0.
            let a = fsrc(tr, ins, &v, 0)?;
            fout(tr, &v, a)?;
            let fl = tr.bool_c(false);
            write_mask(tr, ins, v.sdst, fl)?;
        }
        "v_div_fixup_f32" => {
            // src0 already holds the quotient computed by the preceding sequence.
            let a = fsrc(tr, ins, &v, 0)?;
            fout(tr, &v, a)?;
        }
        // ---- conversions
        "v_cvt_f32_i32" | "v_cvt_f32_u32" => {
            let a = usrc(tr, ins, &v, 0)?;
            let r = if name == "v_cvt_f32_i32" {
                let ai = tr.to_i(a)?;
                e(tr.b.convert_s_to_f(f, None, ai))?
            } else {
                e(tr.b.convert_u_to_f(f, None, a))?
            };
            fout(tr, &v, r)?;
        }
        "v_cvt_u32_f32" | "v_cvt_i32_f32" | "v_cvt_rpi_i32_f32" | "v_cvt_flr_i32_f32" => {
            let a = fsrc(tr, ins, &v, 0)?;
            let r = match name {
                "v_cvt_u32_f32" => {
                    let z = tr.f(0.0);
                    let mx = tr.f(4294967040.0);
                    let c = tr.glsl3(f, GLOp::NClamp, a, z, mx)?;
                    let cv = e(tr.b.convert_f_to_u(u, None, c))?;
                    // Saturated as the hardware does it: 2^32 and above (infinity too) give
                    // 0xffffffff (NClamp's bound gave 0xffffff00); a NaN gives 0, from the bits.
                    let bt = tr.t.bool_;
                    let top = tr.f(4294967296.0);
                    let over = e(tr.b.f_ord_greater_than_equal(bt, None, a, top))?;
                    let all = tr.u(u32::MAX);
                    let cv = e(tr.b.select(u, None, over, all, cv))?;
                    let n = nan_bits(tr, a)?;
                    let zu = tr.u(0);
                    e(tr.b.select(u, None, n, zu, cv))?
                }
                _ => {
                    let x = match name {
                        "v_cvt_rpi_i32_f32" => {
                            let h = tr.f(0.5);
                            let s = e(tr.b.f_add(f, None, a, h))?;
                            tr.glsl1(f, GLOp::Floor, s)?
                        }
                        "v_cvt_flr_i32_f32" => tr.glsl1(f, GLOp::Floor, a)?,
                        _ => a,
                    };
                    let lo = tr.f(-2147483648.0);
                    let hi = tr.f(2147483520.0);
                    let c = tr.glsl3(f, GLOp::NClamp, x, lo, hi)?;
                    let cv = e(tr.b.convert_f_to_s(i, None, c))?;
                    let cv = tr.to_u(cv)?;
                    // Saturated as the hardware does it: 2^31 and above (infinity too) give
                    // 0x7fffffff, where NClamp's bound gave 0x7fffff80; a NaN converts to 0 (the
                    // D3D rule; NClamp made it -2147483648), tested on the source's bits (floor
                    // and +0.5 keep a NaN, a driver free to assume none need not).
                    let bt = tr.t.bool_;
                    let top = tr.f(2147483648.0);
                    let over = e(tr.b.f_ord_greater_than_equal(bt, None, x, top))?;
                    let imax = tr.u(0x7fff_ffff);
                    let cv = e(tr.b.select(u, None, over, imax, cv))?;
                    let n = nan_bits(tr, a)?;
                    let z = tr.u(0);
                    e(tr.b.select(u, None, n, z, cv))?
                }
            };
            uout(tr, &v, r)?;
        }
        "v_cvt_off_f32_i4" => {
            let a = usrc(tr, ins, &v, 0)?;
            let z = tr.u(0);
            let k4 = tr.u(4);
            let s = e(tr.b.bit_field_s_extract(u, None, a, z, k4))?;
            let si = tr.to_i(s)?;
            let x = e(tr.b.convert_s_to_f(f, None, si))?;
            let k = tr.f(1.0 / 16.0);
            let r = e(tr.b.f_mul(f, None, x, k))?;
            fout(tr, &v, r)?;
        }
        "v_cvt_f32_ubyte0" | "v_cvt_f32_ubyte1" | "v_cvt_f32_ubyte2" | "v_cvt_f32_ubyte3" => {
            let a = usrc(tr, ins, &v, 0)?;
            let k = (name.as_bytes()[name.len() - 1] - b'0') as u32;
            let off = tr.u(k * 8);
            let w = tr.u(8);
            let x = e(tr.b.bit_field_u_extract(u, None, a, off, w))?;
            let r = e(tr.b.convert_u_to_f(f, None, x))?;
            fout(tr, &v, r)?;
        }
        "v_cvt_f16_f32" => {
            let a = fsrc(tr, ins, &v, 0)?;
            let h = f_to_half(tr, a)?;
            // VOP3 op_sel[3]: write the high half, keep the low half.
            let r = if v.opsel & 8 != 0 {
                let k16 = tr.u(16);
                let hs = e(tr.b.shift_left_logical(u, None, h, k16))?;
                let old = tr.vreg(v.dst as u32)?;
                let m = tr.u(0xFFFF);
                let lo = e(tr.b.bitwise_and(u, None, old, m))?;
                e(tr.b.bitwise_or(u, None, lo, hs))?
            } else {
                h
            };
            uout(tr, &v, r)?;
        }
        "v_cvt_f32_f16" => {
            let a = usrc(tr, ins, &v, 0)?;
            let mut x = half_to_f(tr, a, v.opsel & 1 != 0)?;
            if v.abs & 1 != 0 {
                x = tr.glsl1(f, GLOp::FAbs, x)?;
            }
            if v.neg & 1 != 0 {
                x = e(tr.b.f_negate(f, None, x))?;
            }
            fout(tr, &v, x)?;
        }
        "v_cvt_pkrtz_f16_f32" | "v_cvt_pknorm_i16_f32" | "v_cvt_pknorm_u16_f32" => {
            let a = fsrc(tr, ins, &v, 0)?;
            let b = fsrc(tr, ins, &v, 1)?;
            // (The norm packs: a NaN packs as 0, as the hardware; GLSL's packs clamp with
            // FClamp, undefined for a NaN.)
            let (a, b) = if name == "v_cvt_pkrtz_f16_f32" {
                (a, b)
            } else {
                let z = tr.f(0.0);
                let na = nan_bits(tr, a)?;
                let a = e(tr.b.select(f, None, na, z, a))?;
                let nb = nan_bits(tr, b)?;
                let b = e(tr.b.select(f, None, nb, z, b))?;
                (a, b)
            };
            let vv = e(tr.b.composite_construct(tr.t.v2f, None, [a, b]))?;
            let op = match name {
                "v_cvt_pknorm_i16_f32" => GLOp::PackSnorm2x16,
                "v_cvt_pknorm_u16_f32" => GLOp::PackUnorm2x16,
                _ => GLOp::PackHalf2x16,
            };
            let r = tr.glsl1(u, op, vv)?;
            uout(tr, &v, r)?;
        }
        "v_cvt_pk_u16_u32" | "v_cvt_pk_i16_i32" => {
            let a = usrc(tr, ins, &v, 0)?;
            let b = usrc(tr, ins, &v, 1)?;
            let m = tr.u(0xFFFF);
            let s16 = tr.u(16);
            let (a, b) = if name == "v_cvt_pk_u16_u32" {
                (tr.glsl2(u, GLOp::UMin, a, m)?, tr.glsl2(u, GLOp::UMin, b, m)?)
            } else {
                let lo = tr.i(-32768);
                let hi = tr.i(32767);
                let (ai, bi) = (tr.to_i(a)?, tr.to_i(b)?);
                let ca = tr.glsl3(i, GLOp::SClamp, ai, lo, hi)?;
                let cb = tr.glsl3(i, GLOp::SClamp, bi, lo, hi)?;
                (tr.to_u(ca)?, tr.to_u(cb)?)
            };
            let al = e(tr.b.bitwise_and(u, None, a, m))?;
            let bl = e(tr.b.bitwise_and(u, None, b, m))?;
            let bh = e(tr.b.shift_left_logical(u, None, bl, s16))?;
            let r = e(tr.b.bitwise_or(u, None, al, bh))?;
            uout(tr, &v, r)?;
        }
        "v_cvt_pk_u8_f32" => {
            // d = (src2 & ~(0xff << 8*byte)) | (u8(src0) << 8*byte)
            let x = fsrc(tr, ins, &v, 0)?;
            let by = usrc(tr, ins, &v, 1)?;
            let base = usrc(tr, ins, &v, 2)?;
            let z = tr.f(0.0);
            let m255 = tr.f(255.0);
            let c = tr.glsl3(f, GLOp::NClamp, x, z, m255)?;
            let rn = tr.glsl1(f, GLOp::RoundEven, c)?;
            let q = e(tr.b.convert_f_to_u(u, None, rn))?;
            let k3 = tr.u(3);
            let k8 = tr.u(8);
            let bi = e(tr.b.bitwise_and(u, None, by, k3))?;
            let sh = e(tr.b.i_mul(u, None, bi, k8))?;
            let kff = tr.u(0xFF);
            let mask = e(tr.b.shift_left_logical(u, None, kff, sh))?;
            let nm = e(tr.b.not(u, None, mask))?;
            let kept = e(tr.b.bitwise_and(u, None, base, nm))?;
            let ins_ = e(tr.b.shift_left_logical(u, None, q, sh))?;
            let r = e(tr.b.bitwise_or(u, None, kept, ins_))?;
            uout(tr, &v, r)?;
        }
        "v_pack_b32_f16" => {
            let a = usrc(tr, ins, &v, 0)?;
            let b = usrc(tr, ins, &v, 1)?;
            let k16 = tr.u(16);
            let m = tr.u(0xFFFF);
            let a = if v.opsel & 1 != 0 { e(tr.b.shift_right_logical(u, None, a, k16))? } else { a };
            let b = if v.opsel & 2 != 0 { e(tr.b.shift_right_logical(u, None, b, k16))? } else { b };
            let al = e(tr.b.bitwise_and(u, None, a, m))?;
            let bl = e(tr.b.bitwise_and(u, None, b, m))?;
            let bh = e(tr.b.shift_left_logical(u, None, bl, k16))?;
            let r = e(tr.b.bitwise_or(u, None, al, bh))?;
            uout(tr, &v, r)?;
        }
        "v_fma_mix_f32" | "v_fma_mixlo_f16" | "v_fma_mixhi_f16" => {
            // Sources are f32, or f16 halves when op_sel_hi is set (op_sel
            // picks the half); neg_hi acts as abs.
            let mut s = [0; 3];
            for (k, sk) in s.iter_mut().enumerate() {
                let raw = tr.operand(ins, v.src[k])?;
                let mut x = if v.opsel_hi >> k & 1 != 0 { half_to_f(tr, raw, v.opsel >> k & 1 != 0)? } else { tr.to_f(raw)? };
                if v.neg_hi >> k & 1 != 0 {
                    x = tr.glsl1(f, GLOp::FAbs, x)?;
                }
                if v.neg >> k & 1 != 0 {
                    x = e(tr.b.f_negate(f, None, x))?;
                }
                *sk = x;
            }
            let mut r = tr.glsl3(f, GLOp::Fma, s[0], s[1], s[2])?;
            if v.clamp {
                let z = tr.f(0.0);
                let o = tr.f(1.0);
                r = tr.glsl3(f, GLOp::NClamp, r, z, o)?;
            }
            if name == "v_fma_mix_f32" {
                let ru = tr.to_u(r)?;
                uout(tr, &v, ru)?;
            } else {
                let h = f_to_half(tr, r)?;
                let old = tr.vreg(v.dst as u32)?;
                let res = if name == "v_fma_mixlo_f16" {
                    let mh = tr.u(0xFFFF_0000);
                    let keep = e(tr.b.bitwise_and(u, None, old, mh))?;
                    e(tr.b.bitwise_or(u, None, keep, h))?
                } else {
                    let ml = tr.u(0xFFFF);
                    let keep = e(tr.b.bitwise_and(u, None, old, ml))?;
                    let k16 = tr.u(16);
                    let hs = e(tr.b.shift_left_logical(u, None, h, k16))?;
                    e(tr.b.bitwise_or(u, None, keep, hs))?
                };
                uout(tr, &v, res)?;
            }
        }
        // ---- 16-bit integer arithmetic (VOP3 on gfx10: v_add_nc_u16 ...)
        "v_add_u16" | "v_sub_u16" | "v_add_i16" | "v_sub_i16" | "v_mul_lo_u16" | "v_max_u16" | "v_max_i16" | "v_min_u16" | "v_min_i16" | "v_lshlrev_b16" | "v_lshrrev_b16" | "v_ashrrev_i16" | "v_mad_u16" | "v_mad_i16" if ops16() => {
            let signed = name.ends_with("i16");
            let a = isrc16(tr, ins, &v, 0, signed)?;
            let b = isrc16(tr, ins, &v, 1, signed || name == "v_ashrrev_i16")?;
            let k15 = tr.u(15);
            let r = match name {
                "v_add_u16" | "v_sub_u16" | "v_add_i16" | "v_sub_i16" => {
                    let r = if name.starts_with("v_add") { e(tr.b.i_add(u, None, a, b))? } else { e(tr.b.i_sub(u, None, a, b))? };
                    if v.clamp {
                        // Saturating forms (the exact result fits in 32 bits).
                        let (lo, hi) = if signed { (tr.i(-32768), tr.i(32767)) } else { (tr.i(0), tr.i(65535)) };
                        let ri = tr.to_i(r)?;
                        let c = tr.glsl3(i, GLOp::SClamp, ri, lo, hi)?;
                        tr.to_u(c)?
                    } else {
                        r
                    }
                }
                "v_mul_lo_u16" => e(tr.b.i_mul(u, None, a, b))?,
                "v_max_u16" => tr.glsl2(u, GLOp::UMax, a, b)?,
                "v_min_u16" => tr.glsl2(u, GLOp::UMin, a, b)?,
                "v_max_i16" | "v_min_i16" => {
                    let (ai, bi) = (tr.to_i(a)?, tr.to_i(b)?);
                    let r = tr.glsl2(i, if name == "v_max_i16" { GLOp::SMax } else { GLOp::SMin }, ai, bi)?;
                    tr.to_u(r)?
                }
                "v_lshlrev_b16" | "v_lshrrev_b16" | "v_ashrrev_i16" => {
                    let sh = e(tr.b.bitwise_and(u, None, a, k15))?;
                    match name {
                        "v_lshlrev_b16" => e(tr.b.shift_left_logical(u, None, b, sh))?,
                        "v_lshrrev_b16" => e(tr.b.shift_right_logical(u, None, b, sh))?,
                        _ => e(tr.b.shift_right_arithmetic(u, None, b, sh))?,
                    }
                }
                _ => {
                    // v_mad_u16 / v_mad_i16: a * b + c.
                    let c = isrc16(tr, ins, &v, 2, signed)?;
                    let p = e(tr.b.i_mul(u, None, a, b))?;
                    let r = e(tr.b.i_add(u, None, p, c))?;
                    if v.clamp {
                        if signed {
                            let (lo, hi) = (tr.i(-32768), tr.i(32767));
                            let ri = tr.to_i(r)?;
                            let c = tr.glsl3(i, GLOp::SClamp, ri, lo, hi)?;
                            tr.to_u(c)?
                        } else {
                            let m = tr.u(0xFFFF);
                            tr.glsl2(u, GLOp::UMin, r, m)?
                        }
                    } else {
                        r
                    }
                }
            };
            out16(tr, &v, r)?;
        }
        // ---- integer arithmetic
        "v_add_u32" | "v_sub_u32" | "v_subrev_u32" | "v_add_i32" | "v_sub_i32" => {
            let a = usrc(tr, ins, &v, 0)?;
            let b = usrc(tr, ins, &v, 1)?;
            let (x, y) = if name == "v_subrev_u32" { (b, a) } else { (a, b) };
            let add = name.starts_with("v_add");
            let mut r = if add { e(tr.b.i_add(u, None, x, y))? } else { e(tr.b.i_sub(u, None, x, y))? };
            if v.clamp {
                // Saturating forms.
                if name.ends_with("u32") {
                    let lim = if add { tr.u(u32::MAX) } else { tr.u(0) };
                    let ovf = if add { e(tr.b.u_less_than(bt, None, r, x))? } else { e(tr.b.u_less_than(bt, None, x, y))? };
                    r = e(tr.b.select(u, None, ovf, lim, r))?;
                } else {
                    let ovf = if add { add_overflow(tr, x, y, r)? } else { sub_overflow(tr, x, y, r)? };
                    let k31 = tr.u(31);
                    let sx = e(tr.b.shift_right_arithmetic(u, None, x, k31))?;
                    let maxi = tr.u(0x7FFF_FFFF);
                    let lim = e(tr.b.bitwise_xor(u, None, sx, maxi))?;
                    r = e(tr.b.select(u, None, ovf, lim, r))?;
                }
            }
            uout(tr, &v, r)?;
        }
        "v_add_co_u32" | "v_sub_co_u32" | "v_subrev_co_u32" | "v_addc_co_u32" | "v_subb_co_u32" | "v_subbrev_co_u32" => {
            // VOP2 forms use VCC for carry in/out; VOP3B uses src2 / sdst.
            let a = usrc(tr, ins, &v, 0)?;
            let b = usrc(tr, ins, &v, 1)?;
            let carry_out = if v.vop3 { v.sdst } else { OP_VCC_LO };
            let cin = if name.contains("addc") || name.contains("subb") {
                let c = if v.vop3 { read_mask(tr, ins, v.src[2])? } else { read_mask(tr, ins, OP_VCC_LO)? };
                Some(tr.bool_to_u(c)?)
            } else {
                None
            };
            let (x, y) = if name.contains("rev") { (b, a) } else { (a, b) };
            let (r, co) = if name.starts_with("v_add") {
                let t = e(tr.b.i_add(u, None, x, y))?;
                let c1 = e(tr.b.u_less_than(bt, None, t, x))?;
                match cin {
                    Some(ci) => {
                        let r = e(tr.b.i_add(u, None, t, ci))?;
                        let c2 = e(tr.b.u_less_than(bt, None, r, t))?;
                        (r, e(tr.b.logical_or(bt, None, c1, c2))?)
                    }
                    None => (t, c1),
                }
            } else {
                let t = e(tr.b.i_sub(u, None, x, y))?;
                let c1 = e(tr.b.u_less_than(bt, None, x, y))?;
                match cin {
                    Some(ci) => {
                        let r = e(tr.b.i_sub(u, None, t, ci))?;
                        let c2 = e(tr.b.u_less_than(bt, None, t, ci))?;
                        (r, e(tr.b.logical_or(bt, None, c1, c2))?)
                    }
                    None => (t, c1),
                }
            };
            uout(tr, &v, r)?;
            write_mask(tr, ins, carry_out, co)?;
        }
        "v_add3_u32" | "v_xad_u32" | "v_lshl_add_u32" | "v_add_lshl_u32" | "v_lshl_or_b32" | "v_and_or_b32" | "v_or3_b32" | "v_xor3_b32" => {
            let a = usrc(tr, ins, &v, 0)?;
            let b = usrc(tr, ins, &v, 1)?;
            let c = usrc(tr, ins, &v, 2)?;
            let k31 = tr.u(31);
            let r = match name {
                "v_add3_u32" => {
                    let t = e(tr.b.i_add(u, None, a, b))?;
                    e(tr.b.i_add(u, None, t, c))?
                }
                "v_xad_u32" => {
                    let t = e(tr.b.bitwise_xor(u, None, a, b))?;
                    e(tr.b.i_add(u, None, t, c))?
                }
                "v_lshl_add_u32" => {
                    let s = e(tr.b.bitwise_and(u, None, b, k31))?;
                    let t = e(tr.b.shift_left_logical(u, None, a, s))?;
                    e(tr.b.i_add(u, None, t, c))?
                }
                "v_add_lshl_u32" => {
                    let t = e(tr.b.i_add(u, None, a, b))?;
                    let s = e(tr.b.bitwise_and(u, None, c, k31))?;
                    e(tr.b.shift_left_logical(u, None, t, s))?
                }
                "v_lshl_or_b32" => {
                    let s = e(tr.b.bitwise_and(u, None, b, k31))?;
                    let t = e(tr.b.shift_left_logical(u, None, a, s))?;
                    e(tr.b.bitwise_or(u, None, t, c))?
                }
                "v_and_or_b32" => {
                    let t = e(tr.b.bitwise_and(u, None, a, b))?;
                    e(tr.b.bitwise_or(u, None, t, c))?
                }
                "v_or3_b32" => {
                    let t = e(tr.b.bitwise_or(u, None, a, b))?;
                    e(tr.b.bitwise_or(u, None, t, c))?
                }
                _ => {
                    let t = e(tr.b.bitwise_xor(u, None, a, b))?;
                    e(tr.b.bitwise_xor(u, None, t, c))?
                }
            };
            uout(tr, &v, r)?;
        }
        "v_mul_lo_u32" | "v_mul_lo_i32" | "v_mul_hi_u32" | "v_mul_hi_i32" => {
            let a = usrc(tr, ins, &v, 0)?;
            let b = usrc(tr, ins, &v, 1)?;
            let r = match name {
                "v_mul_hi_u32" => mul_hi(tr, a, b, false)?,
                "v_mul_hi_i32" => mul_hi(tr, a, b, true)?,
                _ => e(tr.b.i_mul(u, None, a, b))?,
            };
            uout(tr, &v, r)?;
        }
        "v_mul_u32_u24" | "v_mul_hi_u32_u24" | "v_mul_i32_i24" | "v_mul_hi_i32_i24" | "v_mad_u32_u24" | "v_mad_i32_i24" | "v_mad_u32_u16" | "v_mad_i32_i16" => {
            let a = usrc(tr, ins, &v, 0)?;
            let b = usrc(tr, ins, &v, 1)?;
            let signed = name.contains("_i32_");
            let w = tr.u(if name.ends_with("16") { 16 } else { 24 });
            let z = tr.u(0);
            let (ea, eb) = if signed {
                (e(tr.b.bit_field_s_extract(u, None, a, z, w))?, e(tr.b.bit_field_s_extract(u, None, b, z, w))?)
            } else {
                (e(tr.b.bit_field_u_extract(u, None, a, z, w))?, e(tr.b.bit_field_u_extract(u, None, b, z, w))?)
            };
            let r = if name.contains("_hi_") { mul_hi(tr, ea, eb, signed)? } else { e(tr.b.i_mul(u, None, ea, eb))? };
            let r = if name.starts_with("v_mad") {
                let c = usrc(tr, ins, &v, 2)?;
                e(tr.b.i_add(u, None, r, c))?
            } else {
                r
            };
            uout(tr, &v, r)?;
        }
        "v_mad_u64_u32" | "v_mad_i64_i32" => {
            let a = usrc(tr, ins, &v, 0)?;
            let b = usrc(tr, ins, &v, 1)?;
            let (clo, chi) = vsrc64(tr, ins, v.src[2])?;
            let c = to64(tr, clo, chi)?;
            let u64t = tr.t.u64;
            let (a64, b64) = if name == "v_mad_i64_i32" {
                let i64t = tr.b.type_int(64, 1);
                let (ai, bi) = (tr.to_i(a)?, tr.to_i(b)?);
                let a = e(tr.b.s_convert(i64t, None, ai))?;
                let b = e(tr.b.s_convert(i64t, None, bi))?;
                (e(tr.b.bitcast(u64t, None, a))?, e(tr.b.bitcast(u64t, None, b))?)
            } else {
                (e(tr.b.u_convert(u64t, None, a))?, e(tr.b.u_convert(u64t, None, b))?)
            };
            let m = e(tr.b.i_mul(u64t, None, a64, b64))?;
            let r = e(tr.b.i_add(u64t, None, m, c))?;
            let (l, h) = from64(tr, r)?;
            let en = tr.active()?;
            for (k, x) in [(0u32, l), (1, h)] {
                let old = tr.vreg(v.dst as u32 + k)?;
                let nv = e(tr.b.select(u, None, en, x, old))?;
                tr.set_vreg_raw(v.dst as u32 + k, nv)?;
            }
            let fl = tr.bool_c(false);
            write_mask(tr, ins, v.sdst, fl)?;
        }
        "v_min_i32" | "v_max_i32" | "v_min_u32" | "v_max_u32" => {
            let a = usrc(tr, ins, &v, 0)?;
            let b = usrc(tr, ins, &v, 1)?;
            let r = match name {
                "v_min_i32" | "v_max_i32" => {
                    let (x, y) = (tr.to_i(a)?, tr.to_i(b)?);
                    let m = tr.glsl2(i, if name == "v_min_i32" { GLOp::SMin } else { GLOp::SMax }, x, y)?;
                    tr.to_u(m)?
                }
                "v_min_u32" => tr.glsl2(u, GLOp::UMin, a, b)?,
                _ => tr.glsl2(u, GLOp::UMax, a, b)?,
            };
            uout(tr, &v, r)?;
        }
        "v_min3_i32" | "v_max3_i32" | "v_med3_i32" | "v_min3_u32" | "v_max3_u32" | "v_med3_u32" => {
            let a = usrc(tr, ins, &v, 0)?;
            let b = usrc(tr, ins, &v, 1)?;
            let c = usrc(tr, ins, &v, 2)?;
            let signed = name.ends_with("i32");
            let (mn, mx, ty) = if signed { (GLOp::SMin, GLOp::SMax, i) } else { (GLOp::UMin, GLOp::UMax, u) };
            let (a, b, c) = if signed { (tr.to_i(a)?, tr.to_i(b)?, tr.to_i(c)?) } else { (a, b, c) };
            let r = match &name[..5] {
                "v_min" => {
                    let m = tr.glsl2(ty, mn, a, b)?;
                    tr.glsl2(ty, mn, m, c)?
                }
                "v_max" => {
                    let m = tr.glsl2(ty, mx, a, b)?;
                    tr.glsl2(ty, mx, m, c)?
                }
                _ => {
                    let lo = tr.glsl2(ty, mn, a, b)?;
                    let hi = tr.glsl2(ty, mx, a, b)?;
                    let m2 = tr.glsl2(ty, mn, hi, c)?;
                    tr.glsl2(ty, mx, lo, m2)?
                }
            };
            let r = if signed { tr.to_u(r)? } else { r };
            uout(tr, &v, r)?;
        }
        "v_lshrrev_b32" | "v_ashrrev_i32" | "v_lshlrev_b32" | "v_and_b32" | "v_or_b32" | "v_xor_b32" | "v_xnor_b32" | "v_bfm_b32" | "v_bcnt_u32_b32" => {
            let a = usrc(tr, ins, &v, 0)?;
            let b = usrc(tr, ins, &v, 1)?;
            let k31 = tr.u(31);
            let r = match name {
                "v_lshrrev_b32" => {
                    let s = e(tr.b.bitwise_and(u, None, a, k31))?;
                    e(tr.b.shift_right_logical(u, None, b, s))?
                }
                "v_ashrrev_i32" => {
                    let s = e(tr.b.bitwise_and(u, None, a, k31))?;
                    e(tr.b.shift_right_arithmetic(u, None, b, s))?
                }
                "v_lshlrev_b32" => {
                    let s = e(tr.b.bitwise_and(u, None, a, k31))?;
                    e(tr.b.shift_left_logical(u, None, b, s))?
                }
                "v_and_b32" => e(tr.b.bitwise_and(u, None, a, b))?,
                "v_or_b32" => e(tr.b.bitwise_or(u, None, a, b))?,
                "v_xor_b32" => e(tr.b.bitwise_xor(u, None, a, b))?,
                "v_xnor_b32" => {
                    let t = e(tr.b.bitwise_xor(u, None, a, b))?;
                    e(tr.b.not(u, None, t))?
                }
                "v_bfm_b32" => bfm32(tr, a, b)?,
                _ => {
                    let c = e(tr.b.bit_count(u, None, a))?;
                    e(tr.b.i_add(u, None, c, b))?
                }
            };
            uout(tr, &v, r)?;
        }
        "v_mbcnt_lo_u32_b32" | "v_mbcnt_hi_u32_b32" => {
            // popcount(mask & lanes_below) + src1. Lane masks (EXEC/VCC and
            // compare results) hold only this lane's bit: in a wave of one no
            // active lane is below, so they count 0. Other masks (-1 for the
            // lane id) use the hardware lane index. Wave-mask mode has the
            // wave's masks, an append done per subgroup its subgroup's.
            let m = usrc(tr, ins, &v, 0)?;
            let b = usrc(tr, ins, &v, 1)?;
            let m = if tr.wave_masks || tr.sg.ranks.contains(&ins.pc) {
                m
            } else {
                let one = tr.u(1);
                let lane_mask = e(tr.b.u_less_than_equal(bt, None, m, one))?;
                let zero = tr.u(0);
                e(tr.b.select(u, None, lane_mask, zero, m))?
            };
            let (lo, hi) = lane_masks(tr)?;
            let lm = if name == "v_mbcnt_lo_u32_b32" { lo } else { hi };
            let x = e(tr.b.bitwise_and(u, None, m, lm))?;
            let c = e(tr.b.bit_count(u, None, x))?;
            let r = e(tr.b.i_add(u, None, c, b))?;
            uout(tr, &v, r)?;
        }
        "v_not_b32" | "v_bfrev_b32" | "v_ffbh_u32" | "v_ffbh_i32" | "v_ffbl_b32" => {
            let a = usrc(tr, ins, &v, 0)?;
            let r = match name {
                "v_not_b32" => e(tr.b.not(u, None, a))?,
                "v_bfrev_b32" => e(tr.b.bit_reverse(u, None, a))?,
                "v_ffbl_b32" => {
                    let r = tr.glsl1(i, GLOp::FindILsb, a)?;
                    tr.to_u(r)?
                }
                _ => {
                    // Count from the MSB; -1 if none.
                    let m = if name == "v_ffbh_u32" {
                        tr.glsl1(i, GLOp::FindUMsb, a)?
                    } else {
                        let ai = tr.to_i(a)?;
                        tr.glsl1(i, GLOp::FindSMsb, ai)?
                    };
                    let m = tr.to_u(m)?;
                    let k31 = tr.u(31);
                    let r = e(tr.b.i_sub(u, None, k31, m))?;
                    let neg1 = tr.u(u32::MAX);
                    let none = e(tr.b.i_equal(bt, None, m, neg1))?;
                    e(tr.b.select(u, None, none, neg1, r))?
                }
            };
            uout(tr, &v, r)?;
        }
        "v_bfe_u32" | "v_bfe_i32" => {
            let a = usrc(tr, ins, &v, 0)?;
            let b = usrc(tr, ins, &v, 1)?;
            let c = usrc(tr, ins, &v, 2)?;
            let k31 = tr.u(31);
            let off = e(tr.b.bitwise_and(u, None, b, k31))?;
            let w = e(tr.b.bitwise_and(u, None, c, k31))?;
            let r = bfe(tr, a, off, w, name == "v_bfe_i32")?;
            uout(tr, &v, r)?;
        }
        "v_bfi_b32" => {
            let m = usrc(tr, ins, &v, 0)?;
            let a = usrc(tr, ins, &v, 1)?;
            let b = usrc(tr, ins, &v, 2)?;
            let x = e(tr.b.bitwise_and(u, None, m, a))?;
            let nm = e(tr.b.not(u, None, m))?;
            let y = e(tr.b.bitwise_and(u, None, nm, b))?;
            let r = e(tr.b.bitwise_or(u, None, x, y))?;
            uout(tr, &v, r)?;
        }
        "v_alignbit_b32" | "v_alignbyte_b32" => {
            // ({a,b} >> shift)[31:0]
            let a = usrc(tr, ins, &v, 0)?;
            let b = usrc(tr, ins, &v, 1)?;
            let c = usrc(tr, ins, &v, 2)?;
            let wide = to64(tr, b, a)?;
            let k = tr.u(if name == "v_alignbit_b32" { 31 } else { 3 });
            let mut sh = e(tr.b.bitwise_and(u, None, c, k))?;
            if name == "v_alignbyte_b32" {
                let k3 = tr.u(3);
                sh = e(tr.b.shift_left_logical(u, None, sh, k3))?;
            }
            let r64 = e(tr.b.shift_right_logical(tr.t.u64, None, wide, sh))?;
            let r = e(tr.b.u_convert(u, None, r64))?;
            uout(tr, &v, r)?;
        }
        "v_perm_b32" => {
            // Byte permute of {src0, src1} by selector src2.
            let a = usrc(tr, ins, &v, 0)?;
            let b = usrc(tr, ins, &v, 1)?;
            let s = usrc(tr, ins, &v, 2)?;
            let wide = to64(tr, b, a)?;
            let mut r = tr.u(0);
            for k in 0..4u32 {
                let ko = tr.u(k * 8);
                let k8 = tr.u(8);
                let sel = e(tr.b.bit_field_u_extract(u, None, s, ko, k8))?;
                let k7 = tr.u(7);
                let idx = e(tr.b.bitwise_and(u, None, sel, k7))?;
                let k3 = tr.u(3);
                let sh = e(tr.b.shift_left_logical(u, None, idx, k3))?;
                let byte64 = e(tr.b.shift_right_logical(tr.t.u64, None, wide, sh))?;
                let byte = e(tr.b.u_convert(u, None, byte64))?;
                let kff = tr.u(0xFF);
                let byte = e(tr.b.bitwise_and(u, None, byte, kff))?;
                // Selectors >= 12 produce constants (0x00 for 12, 0xff for >= 13).
                let k12 = tr.u(12);
                let k13 = tr.u(13);
                let is12 = e(tr.b.i_equal(bt, None, sel, k12))?;
                let ge13 = e(tr.b.u_greater_than_equal(bt, None, sel, k13))?;
                let k8b = tr.u(8);
                let lt8 = e(tr.b.u_less_than(bt, None, sel, k8b))?;
                let z = tr.u(0);
                let x = e(tr.b.select(u, None, lt8, byte, z))?;
                let x = e(tr.b.select(u, None, is12, z, x))?;
                let x = e(tr.b.select(u, None, ge13, kff, x))?;
                let xs = e(tr.b.shift_left_logical(u, None, x, ko))?;
                r = e(tr.b.bitwise_or(u, None, r, xs))?;
            }
            uout(tr, &v, r)?;
        }
        "v_lshlrev_b64" | "v_lshrrev_b64" | "v_ashrrev_i64" => {
            let s = usrc(tr, ins, &v, 0)?;
            let (blo, bhi) = vsrc64(tr, ins, v.src[1])?;
            let b = to64(tr, blo, bhi)?;
            let k = tr.u(63);
            let sh = e(tr.b.bitwise_and(u, None, s, k))?;
            let r = match name {
                "v_lshlrev_b64" => e(tr.b.shift_left_logical(tr.t.u64, None, b, sh))?,
                "v_lshrrev_b64" => e(tr.b.shift_right_logical(tr.t.u64, None, b, sh))?,
                _ => e(tr.b.shift_right_arithmetic(tr.t.u64, None, b, sh))?,
            };
            let (l, h) = from64(tr, r)?;
            let en = tr.active()?;
            for (k, x) in [(0u32, l), (1, h)] {
                let old = tr.vreg(v.dst as u32 + k)?;
                let nv = e(tr.b.select(u, None, en, x, old))?;
                tr.set_vreg_raw(v.dst as u32 + k, nv)?;
            }
        }
        _ => unsupported(tr, ins),
    }
    Ok(())
}
