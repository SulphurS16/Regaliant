//! Fixes of the shared translator found by the native renderer's line of work (branch `native`,
//! its NATIVE-FINDINGS.md), taken into main as that line wrote them. The native line's own
//! `native.rs` holds these beside its binding model.

use super::{e, R, T};
use crate::Inst;
use rspirv::spirv::Word;

fn exec_consts_off() -> bool {
    static OFF: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *OFF.get_or_init(|| std::env::var("DS_EXEC_CONSTS").ok().as_deref() == Some("0"))
}

/// The constant [`exec_constant_slice`] slices at each lane, if `ins` is a
/// move of one into EXEC: (low, high dword, 64-bit move). 0 and all lanes
/// are the same in every lane and are left to the plain move.
fn sliced_constant(ins: &Inst) -> Option<(u32, u32, bool)> {
    if exec_consts_off() || ins.enc != crate::Enc::Sop1 {
        return None;
    }
    let name = ins.name()?;
    if !matches!(name, "s_mov_b64" | "s_mov_b32") || !matches!(ins.sdst(), crate::OP_EXEC_LO | crate::OP_EXEC_HI) {
        return None;
    }
    let s = ins.ssrc0();
    // (As the translator widens 64-bit operands: inline integers sign-extended, literals zero-extended.)
    let (lo, hi): (u32, u32) = match s {
        129..=192 => ((s - 128) as u32, 0),
        193..=208 => ((-((s - 192) as i32)) as u32, u32::MAX),
        crate::OP_LITERAL => (ins.lit?, 0),
        _ => return None,
    };
    let wide = name == "s_mov_b64";
    if (wide && ((lo == 0 && hi == 0) || (lo == u32::MAX && hi == u32::MAX))) || (!wide && (lo == 0 || lo == u32::MAX)) {
        return None;
    }
    Some((lo, hi, wide))
}

/// For the per-invocation analyses (`active::known_active`): `ins` moves a
/// constant into EXEC that [`exec_constant_slice`] slices, so this lane's
/// bit is not known after it.
pub(crate) fn slices_exec_constant(ins: &Inst) -> bool {
    sliced_constant(ins).is_some()
}

/// Per lane (EXEC holding this lane's bit in bit 0, outside wave-mask mode),
/// `s_mov_b64 exec, <constant>` gives each lane its own bit of the
/// hardware-layout constant (bit L for lane L) and `s_mov_b32 exec_lo|hi,
/// <constant>` the lanes of that half theirs, the other half keeping its
/// own: the lanes the constant leaves out stop executing, as on the
/// hardware. Before, the constant's bit 0 stood for every lane: 7 of the
/// game's compute programs translated per lane (its auto-exposure, three of
/// its GI probes' and the virtual geometry's instance counter) ran their
/// lane-0 step on every lane, the instance counter adding its wave's sum 64
/// times (the native line's finding of 2026-11-26 night). True when it
/// translated the instruction. (`DS_EXEC_CONSTS=0`: off, as before.)
pub(crate) fn exec_constant_slice(tr: &mut T, ins: &Inst, _name: &str) -> R<bool> {
    if tr.own_bit.is_some() {
        return Ok(false);
    }
    let Some((lo, hi, wide)) = sliced_constant(ins) else { return Ok(false) };
    let d = ins.sdst();
    let (u, bt) = (tr.t.u32, tr.t.bool_);
    let lane = tr.lane()?;
    let (k32, k31, one) = (tr.u(32), tr.u(31), tr.u(1));
    let in_hi = e(tr.b.u_greater_than_equal(bt, None, lane, k32))?;
    let sh = e(tr.b.bitwise_and(u, None, lane, k31))?;
    let bit_of = |tr: &mut T, word: u32| -> R<Word> {
        let w = tr.u(word);
        let x = e(tr.b.shift_right_logical(u, None, w, sh))?;
        e(tr.b.bitwise_and(u, None, x, one))
    };
    let new_bit = if wide {
        let bl = bit_of(tr, lo)?;
        let bh = bit_of(tr, hi)?;
        e(tr.b.select(u, None, in_hi, bh, bl))?
    } else {
        let b = bit_of(tr, lo)?;
        let old = tr.sreg(crate::OP_EXEC_LO as u32)?;
        let old_bit = e(tr.b.bitwise_and(u, None, old, one))?;
        let mine = if d == crate::OP_EXEC_LO { e(tr.b.logical_not(bt, None, in_hi))? } else { in_hi };
        e(tr.b.select(u, None, mine, b, old_bit))?
    };
    tr.set_sreg(crate::OP_EXEC_LO as u32, new_bit)?;
    let z = tr.u(0);
    tr.set_sreg(crate::OP_EXEC_HI as u32, z)?;
    Ok(true)
}
