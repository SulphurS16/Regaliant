//! Known-active instructions of per-invocation translations.
//!
//! Outside lockstep each invocation is a wave of one: EXEC holds its own
//! bit in bit 0 (set at entry), lane masks from compares hold only that bit,
//! and `s_cbranch_execz` is decided by the invocation's own EXEC. Where its
//! EXEC bit is known set, [`super::T::active`] is the constant true: register
//! writes need no select against the old value and no lane guard, stores no
//! guard. The analysis follows EXEC, VCC and the SGPRs through the program
//! as two three-valued parts per 32-bit register (its bit 0, its other 31
//! bits): the title's EXEC saves and restores, the regions after a
//! per-lane `s_cbranch_execz` and the all-lanes idiom (`s_orn2_saveexec`
//! with EXEC) keep the bit known.

use super::{block_starts, writes_sgpr, Appends};
use crate::{Enc, Inst, OP_EXEC_HI, OP_EXEC_LO, OP_NULL, OP_VCC_LO};
use std::collections::VecDeque;

/// Bits known all zero, all one, or not known.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
enum Tri {
    Z,
    O,
    U,
}

impl Tri {
    fn and(self, b: Tri) -> Tri {
        match (self, b) {
            (Tri::Z, _) | (_, Tri::Z) => Tri::Z,
            (Tri::O, x) | (x, Tri::O) => x,
            _ => Tri::U,
        }
    }
    fn or(self, b: Tri) -> Tri {
        match (self, b) {
            (Tri::O, _) | (_, Tri::O) => Tri::O,
            (Tri::Z, x) | (x, Tri::Z) => x,
            _ => Tri::U,
        }
    }
    fn not(self) -> Tri {
        match self {
            Tri::Z => Tri::O,
            Tri::O => Tri::Z,
            Tri::U => Tri::U,
        }
    }
    fn xor(self, b: Tri) -> Tri {
        match (self, b) {
            (Tri::Z, x) | (x, Tri::Z) => x,
            (Tri::O, x) | (x, Tri::O) => x.not(),
            _ => Tri::U,
        }
    }
    fn join(self, b: Tri) -> Tri {
        if self == b {
            self
        } else {
            Tri::U
        }
    }
}

/// A 32-bit register: its bit 0 and its bits 1-31.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
struct V {
    b0: Tri,
    rest: Tri,
}

const UNKNOWN: V = V { b0: Tri::U, rest: Tri::U };
const ZERO: V = V { b0: Tri::Z, rest: Tri::Z };
/// A lane mask of this invocation alone (a compare's result).
const OWN: V = V { b0: Tri::U, rest: Tri::Z };

impl V {
    fn konst(c: u32) -> V {
        let rest = match c >> 1 {
            0 => Tri::Z,
            0x7FFF_FFFF => Tri::O,
            _ => Tri::U,
        };
        V { b0: if c & 1 != 0 { Tri::O } else { Tri::Z }, rest }
    }
    fn map2(self, b: V, f: fn(Tri, Tri) -> Tri) -> V {
        V { b0: f(self.b0, b.b0), rest: f(self.rest, b.rest) }
    }
    fn not(self) -> V {
        V { b0: self.b0.not(), rest: self.rest.not() }
    }
    fn join(self, b: V) -> V {
        self.map2(b, Tri::join)
    }
    /// A bitwise op by the translator's stem names (`alu::bitop`: a, b).
    fn op(stem: &str, a: V, b: V) -> Option<V> {
        Some(match stem {
            "and" => a.map2(b, Tri::and),
            "or" => a.map2(b, Tri::or),
            "xor" => a.map2(b, Tri::xor),
            "andn2" => a.map2(b.not(), Tri::and),
            "orn2" => a.map2(b.not(), Tri::or),
            "andn1" => a.not().map2(b, Tri::and),
            "orn1" => a.not().map2(b, Tri::or),
            "nand" => a.map2(b, Tri::and).not(),
            "nor" => a.map2(b, Tri::or).not(),
            "xnor" => a.map2(b, Tri::xor).not(),
            _ => return None,
        })
    }
}

/// A bitwise op of a register with itself (`s_orn2_saveexec_b64 vcc, exec`
/// enables every lane: EXEC | ~EXEC).
fn op_same(stem: &str, x: V) -> Option<V> {
    Some(match stem {
        "and" | "or" => x,
        "xor" | "andn2" | "andn1" => ZERO,
        "orn2" | "orn1" | "xnor" => V { b0: Tri::O, rest: Tri::O },
        "nand" | "nor" => x.not(),
        _ => return None,
    })
}

/// SGPRs 0-127 (VCC 106/107, EXEC 126/127).
type State = [V; 128];

/// A scalar source operand's (lo, hi) halves (hi: the next register, or a
/// 64-bit constant's upper half).
fn src(st: &State, ins: &Inst, s: u16) -> (V, V) {
    match s {
        0..=105 | 106 | 107 | 126 | 127 => (st[s as usize], st.get(s as usize + 1).copied().unwrap_or(UNKNOWN)),
        OP_NULL => (ZERO, ZERO),
        128..=192 => (V::konst((s - 128) as u32), ZERO),
        193..=208 => (V::konst((-((s - 192) as i32)) as u32), V::konst(u32::MAX)),
        // (Literals and float constants zero-extended, as `alu::s64`.)
        255 => (ins.lit.map_or(UNKNOWN, V::konst), ZERO),
        240..=248 => (UNKNOWN, ZERO),
        _ => (UNKNOWN, UNKNOWN),
    }
}

fn set(st: &mut State, d: u16, v: V) {
    if (d as usize) < st.len() && d != OP_NULL {
        st[d as usize] = v;
    }
}

/// Every SGPR the instruction writes made unknown.
fn clobber(st: &mut State, ins: &Inst) {
    for s in 0..128u16 {
        if writes_sgpr(ins, s) {
            st[s as usize] = UNKNOWN;
        }
    }
}

/// One instruction's effect; false where the analysis cannot follow the
/// program (relative SGPR addressing, calls and computed jumps).
fn step(st: &mut State, ins: &Inst, wide_waves: bool, sg: &Appends) -> bool {
    let name = ins.base_name().unwrap_or("");
    if name.starts_with("s_movrel") || name.starts_with("s_subvector_loop") || matches!(name, "s_setpc_b64" | "s_swappc_b64" | "s_rfe_b64" | "s_call_b64" | "s_cbranch_join" | "s_cbranch_g_fork" | "s_cbranch_i_fork") {
        return false;
    }
    let wide = name.ends_with("_b64");
    match ins.enc {
        Enc::Sop1 => {
            let d = ins.sdst();
            let (slo, shi) = src(st, ins, ins.ssrc0());
            if let Some(stem) = name.strip_suffix("_saveexec_b64").or_else(|| name.strip_suffix("_saveexec_b32")).or_else(|| name.strip_suffix("_wrexec_b64")).or_else(|| name.strip_suffix("_wrexec_b32")) {
                let stem = stem.trim_start_matches("s_");
                let (elo, ehi) = (st[OP_EXEC_LO as usize], st[OP_EXEC_HI as usize]);
                let same = ins.ssrc0() == OP_EXEC_LO;
                let (nlo, nhi) = if same { (op_same(stem, elo), op_same(stem, ehi)) } else { (V::op(stem, slo, elo), V::op(stem, shi, ehi)) };
                let (Some(nlo), Some(nhi)) = (nlo, nhi) else {
                    clobber(st, ins);
                    return true;
                };
                let nhi = if wide { nhi } else { ehi };
                let (dlo, dhi) = if name.contains("_wrexec_") { (nlo, nhi) } else { (elo, ehi) };
                set(st, d, dlo);
                if wide {
                    set(st, d + 1, dhi);
                }
                st[OP_EXEC_LO as usize] = nlo;
                st[OP_EXEC_HI as usize] = nhi;
                return true;
            }
            if sg.leaders.contains(&ins.pc) {
                // EXEC := the subgroup's first active invocation (alu::sop1).
                clobber(st, ins);
                st[OP_EXEC_LO as usize] = OWN;
                st[OP_EXEC_HI as usize] = ZERO;
                return true;
            }
            // (A constant lane mask sliced at each lane, `native::exec_constant_slice`: this
            // lane's bit not known after it.)
            if super::native::slices_exec_constant(ins) {
                st[OP_EXEC_LO as usize] = V { b0: Tri::U, rest: Tri::Z };
                st[OP_EXEC_HI as usize] = ZERO;
                return true;
            }
            match name {
                "s_mov_b32" => set(st, d, slo),
                "s_mov_b64" => {
                    set(st, d, slo);
                    set(st, d + 1, shi);
                }
                "s_not_b32" => set(st, d, slo.not()),
                "s_not_b64" => {
                    set(st, d, slo.not());
                    set(st, d + 1, shi.not());
                }
                "s_cmov_b32" => set(st, d, st[d.min(127) as usize].join(slo)),
                "s_cmov_b64" => {
                    let (olo, ohi) = (st[d.min(127) as usize], st[(d + 1).min(127) as usize]);
                    set(st, d, olo.join(slo));
                    set(st, d + 1, ohi.join(shi));
                }
                _ => clobber(st, ins),
            }
        }
        Enc::Sop2 => {
            let d = ins.sdst();
            if sg.lane0.contains(&ins.pc) {
                // `s_and_b64 X, exec, 1` made lane 0's bit (alu::sop2).
                clobber(st, ins);
                set(st, d, OWN);
                if wide {
                    set(st, d + 1, ZERO);
                }
                return true;
            }
            let (alo, ahi) = src(st, ins, ins.ssrc0());
            let (blo, bhi) = src(st, ins, ins.ssrc1());
            let stem = name.strip_suffix("_b64").or_else(|| name.strip_suffix("_b32")).map(|s| s.trim_start_matches("s_"));
            match stem {
                Some("cselect") => {
                    set(st, d, alo.join(blo));
                    if wide {
                        set(st, d + 1, ahi.join(bhi));
                    }
                }
                Some(stem) => match if ins.ssrc0() == ins.ssrc1() && ins.ssrc0() < 128 { (op_same(stem, alo), op_same(stem, ahi)) } else { (V::op(stem, alo, blo), V::op(stem, ahi, bhi)) } {
                    (Some(lo), Some(hi)) => {
                        set(st, d, lo);
                        if wide {
                            set(st, d + 1, hi);
                        }
                    }
                    _ => clobber(st, ins),
                },
                None => clobber(st, ins),
            }
        }
        Enc::Sopk => {
            let d = ins.sdst();
            let k = V::konst(ins.simm16() as i16 as i32 as u32);
            match name {
                "s_movk_i32" => set(st, d, k),
                "s_cmovk_i32" => set(st, d, st[d.min(127) as usize].join(k)),
                _ => clobber(st, ins),
            }
        }
        Enc::Vopc | Enc::Vop3 if name.starts_with("v_cmp") => {
            let whole = sg.ballots.contains(&ins.pc);
            let (lo, hi) = if whole { (UNKNOWN, UNKNOWN) } else { (OWN, ZERO) };
            let d = if name.starts_with("v_cmpx") {
                OP_EXEC_LO
            } else if ins.enc == Enc::Vop3 {
                ins.vdst()
            } else {
                ins.sdwa_sdst().unwrap_or(OP_VCC_LO)
            };
            set(st, d, lo);
            if wide_waves {
                set(st, d + 1, hi);
            }
        }
        _ => clobber(st, ins),
    }
    true
}

/// EXEC != 0 known (the not-taken edge of `s_cbranch_execz`): with its
/// other bits known clear, this invocation's bit is set.
fn exec_nonzero(st: &mut State, wide_waves: bool) {
    let lo = st[OP_EXEC_LO as usize];
    if lo.rest == Tri::Z && (!wide_waves || st[OP_EXEC_HI as usize] == ZERO) {
        st[OP_EXEC_LO as usize].b0 = Tri::O;
    }
}

fn exec_zero(st: &mut State) {
    st[OP_EXEC_LO as usize] = ZERO;
    st[OP_EXEC_HI as usize] = ZERO;
}

/// Pcs of the instructions at whose start this invocation's EXEC bit is
/// known set (per-invocation translations only: not lockstep, not
/// wave-mask mode, not lane votes). Empty where the program uses relative
/// SGPR addressing or computed jumps.
pub(crate) fn known_active(code: &[Inst], wave64: bool, sg: &Appends) -> crate::HashSet<u32> {
    let mut out = crate::HashSet::default();
    if code.is_empty() {
        return out;
    }
    let starts = block_starts(code);
    let index: crate::HashMap<u32, usize> = code.iter().enumerate().map(|(i, x)| (x.pc, i)).collect();
    // Block start pc -> its instructions' index range.
    let mut blocks: crate::HashMap<u32, (usize, usize)> = crate::HashMap::default();
    let st_list: Vec<u32> = starts.iter().copied().collect();
    for (n, &s) in st_list.iter().enumerate() {
        let Some(&a) = index.get(&s) else { return out };
        let b = st_list.get(n + 1).and_then(|e| index.get(e).copied()).unwrap_or(code.len());
        if a < b {
            blocks.insert(s, (a, b));
        }
    }
    let mut entry = [UNKNOWN; 128];
    entry[OP_EXEC_LO as usize] = V { b0: Tri::O, rest: Tri::Z };
    entry[OP_EXEC_HI as usize] = ZERO;
    let mut at: crate::HashMap<u32, State> = crate::HashMap::default();
    at.insert(code[0].pc, entry);
    let mut work: VecDeque<u32> = VecDeque::from([code[0].pc]);
    let mut queued: crate::HashSet<u32> = crate::HashSet::from_iter([code[0].pc]);
    let mut steps = 0usize;
    while let Some(b) = work.pop_front() {
        queued.remove(&b);
        steps += 1;
        if steps > 64 * blocks.len() + 64 {
            return crate::HashSet::default();
        }
        let Some(&(a, e)) = blocks.get(&b) else { return crate::HashSet::default() };
        let mut st = at[&b];
        let mut edges: Vec<(u32, State)> = Vec::new();
        let mut falls = true;
        for ins in &code[a..e] {
            let name = ins.name().unwrap_or("");
            if ins.is_endpgm() {
                falls = false;
                break;
            }
            if let Some(t) = ins.branch_target() {
                let (mut taken, mut not) = (st, st);
                match name {
                    "s_branch" => falls = false,
                    "s_cbranch_execz" => {
                        exec_zero(&mut taken);
                        exec_nonzero(&mut not, wave64);
                    }
                    "s_cbranch_execnz" => {
                        exec_nonzero(&mut taken, wave64);
                        exec_zero(&mut not);
                    }
                    _ => {}
                }
                edges.push((t, taken));
                st = not;
                continue;
            }
            if !step(&mut st, ins, wave64, sg) {
                return crate::HashSet::default();
            }
        }
        if falls {
            if let Some(n) = code.get(e) {
                edges.push((n.pc, st));
            }
        }
        for (t, s) in edges {
            if !blocks.contains_key(&t) {
                return crate::HashSet::default();
            }
            let merged = match at.get(&t) {
                Some(old) => {
                    let mut m = *old;
                    for (x, y) in m.iter_mut().zip(s.iter()) {
                        *x = x.join(*y);
                    }
                    if m == *old {
                        continue;
                    }
                    m
                }
                None => s,
            };
            at.insert(t, merged);
            if queued.insert(t) {
                work.push_back(t);
            }
        }
    }
    // The bit at each instruction's start.
    for (&b, &(a, e)) in &blocks {
        let Some(mut st) = at.get(&b).copied() else { continue };
        for ins in &code[a..e] {
            if st[OP_EXEC_LO as usize].b0 == Tri::O {
                out.insert(ins.pc);
            }
            if ins.branch_target().is_some() || ins.is_endpgm() {
                // (The not-taken edge's state continues in the block.)
                if ins.name() == Some("s_cbranch_execz") {
                    exec_nonzero(&mut st, wave64);
                } else if ins.name() == Some("s_cbranch_execnz") {
                    exec_zero(&mut st);
                }
                continue;
            }
            if !step(&mut st, ins, wave64, sg) {
                return crate::HashSet::default();
            }
        }
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    fn known(words: &[u32]) -> Vec<u32> {
        let code = crate::decode_program(words).unwrap();
        let mut v: Vec<u32> = known_active(&code, true, &Appends::default()).into_iter().collect();
        v.sort();
        v
    }

    /// A per-lane region (`s_and_saveexec` + `s_cbranch_execz`) and the
    /// restore after it.
    #[test]
    fn region_and_restore() {
        let words = [
            0x7C020280, // 0x00 v_cmp_lt_f32 vcc, 0, v1
            0xBE88246A, // 0x04 s_and_saveexec_b64 s[8:9], vcc
            0xBF880001, // 0x08 s_cbranch_execz 0x10
            0x7E0402F2, // 0x0c v_mov_b32 v2, 1.0      (known: past execz)
            0x88FE087E, // 0x10 s_or_b64 exec, exec, s[8:9] (merge: not known)
            0x7E060280, // 0x14 v_mov_b32 v3, 0        (known: restored)
            0xBF810000, // 0x18 s_endpgm
        ];
        assert_eq!(known(&words), vec![0x00, 0x04, 0x0c, 0x14, 0x18]);
    }

    /// EXEC from a lane read is not known, nor after an execz past it; the
    /// all-lanes idiom (EXEC | ~EXEC) is.
    #[test]
    fn unknown_mask_and_all_lanes() {
        let words = [
            0x7E100500, // 0x00 v_readfirstlane_b32 s8, v0
            0xBEFE0408, // 0x04 s_mov_b64 exec, s[8:9]
            0xBF880001, // 0x08 s_cbranch_execz 0x10
            0x7E0402F2, // 0x0c v_mov_b32 v2, 1.0      (not known: EXEC's other bits)
            0xBE8A287E, // 0x10 s_orn2_saveexec_b64 s[10:11], exec
            0x7E060280, // 0x14 v_mov_b32 v3, 0        (known: every lane)
            0xBF810000, // 0x18 s_endpgm
        ];
        assert_eq!(known(&words), vec![0x00, 0x04, 0x14, 0x18]);
    }

    /// A loop whose body narrows EXEC: the header's state is the join of
    /// entry and back edge.
    #[test]
    fn loop_join() {
        let words = [
            0x7E060280, // 0x00 v_mov_b32 v3, 0        (known)
            0x7C020280, // 0x04 v_cmp_lt_f32 vcc, 0, v1 (header: entry and back edge)
            0x87FE6A7E, // 0x08 s_and_b64 exec, exec, vcc
            0xBF880001, // 0x0c s_cbranch_execz 0x14
            0xBF82FFFC, // 0x10 s_branch 0x04          (known: past execz)
            0xBF810000, // 0x14 s_endpgm               (EXEC zero: not known)
        ];
        assert_eq!(known(&words), vec![0x00, 0x04, 0x08, 0x10]);
    }
}
