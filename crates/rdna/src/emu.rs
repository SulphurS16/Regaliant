//! Reference interpreter of gfx10 compute programs, for validating the
//! SPIR-V translation (`translate`).
//!
//! It runs one workgroup with hardware semantics: the lanes of a wave execute
//! in lockstep under a real EXEC mask, scalar registers exist once per wave,
//! compare results are lane masks, LDS is shared by the waves and s_barrier
//! synchronizes them (waves run in turn between barriers, which is one valid
//! hardware interleaving). Memory and images go through [`Env`]; nothing here
//! touches guest memory directly. Unsupported instructions stop the group
//! with an error rather than guessing.
//!
//! Semantics follow the "RDNA 1.0" Instruction Set Architecture reference
//! guide (AMD, 2020); they are implemented independently of the translator so
//! that the two can be compared.

use crate::format::{self, Num};
use crate::{bits, Enc, Ext, Inst};
use crate::HashMap;

/// Memory, image and logging services for an emulated workgroup.
pub trait Env {
    /// Guest memory read; false when unmapped (the bytes are then zero).
    fn read(&mut self, addr: u64, out: &mut [u8]) -> bool;
    fn write(&mut self, addr: u64, data: &[u8]);
    /// Raw texel bits (little-endian dwords) of image `t` at (x, y, z/layer)
    /// of mip `level`; None outside the image or when unavailable.
    fn texel(&mut self, t: &[u32; 8], x: u32, y: u32, z: u32, level: u32) -> Option<[u32; 4]>;
    fn store_texel(&mut self, t: &[u32; 8], x: u32, y: u32, z: u32, level: u32, raw: [u32; 4]);
}

/// Dispatch parameters of a workgroup.
#[derive(Clone, Debug)]
pub struct Launch {
    pub wave64: bool,
    pub local_size: [u32; 3],
    /// COMPUTE_PGM_RSRC2 (user SGPR count, TGID enables, TG_SIZE, TIDIG count).
    pub rsrc2: u32,
    pub user: Vec<u32>,
    pub group: [u32; 3],
    /// LDS bytes.
    pub lds_bytes: u32,
    /// Address of the program (s_getpc_b64).
    pub code_base: u64,
    /// Debugging, as `translate::Spec::probe`: (pc, vgpr): the VGPR's value
    /// after the instruction at pc replaces the data of every store.
    pub probe: Option<(u32, u32)>,
}

const VCC: usize = 106;
const M0: usize = 124;
const EXEC: usize = 126;

struct Wave {
    s: [u32; 128],
    v: Vec<[u32; 64]>,
    probe: [u32; 64],
    scc: bool,
    pc: usize,
    done: bool,
    at_barrier: bool,
    id: u32,
}

impl Wave {
    fn exec(&self) -> u64 {
        self.s[EXEC] as u64 | (self.s[EXEC + 1] as u64) << 32
    }
    fn set_exec(&mut self, m: u64) {
        self.s[EXEC] = m as u32;
        self.s[EXEC + 1] = (m >> 32) as u32;
    }
    fn s64(&self, r: usize) -> u64 {
        self.s[r] as u64 | (self.s[(r + 1).min(127)] as u64) << 32
    }
    fn set_s64(&mut self, r: usize, x: u64) {
        if r < 128 && r != 125 {
            self.s[r] = x as u32;
            if r + 1 < 128 {
                self.s[r + 1] = (x >> 32) as u32;
            }
        }
    }
    fn set_s(&mut self, r: usize, x: u32) {
        if r < 128 && r != 125 {
            self.s[r] = x;
        }
    }
}

/// Decoded vector operands (VOP1/2/C/3, SDWA/DPP extensions).
struct Vop {
    dst: u16,
    src: [u16; 3],
    abs: u32,
    neg: u32,
    clamp: bool,
    omod: u32,
    opsel: u32,
    sdst: u16,
    vop3: bool,
    /// SDWA: (dst_sel, dst_unused, src_sel[2], sext[2]).
    sdwa: Option<(u32, u32, [u32; 2], [bool; 2])>,
    /// DPP16 control dword / DPP8 selects.
    dpp16: Option<u32>,
    dpp8: Option<(u32, bool)>,
}

pub struct Group<'a, E: Env> {
    code: &'a [Inst],
    index: HashMap<u32, usize>,
    env: &'a mut E,
    lds: Vec<u8>,
    waves: Vec<Wave>,
    lanes: usize,
    launch: Launch,
    /// Executed instructions (all waves), for limits and statistics.
    pub steps: u64,
    /// Optional per-instruction trace callback input: (wave, pc).
    pub trace: Option<Box<dyn FnMut(u32, &Inst, &[u32; 128], &[[u32; 64]]) + 'a>>,
}

type R<T> = Result<T, String>;

fn f(x: u32) -> f32 {
    f32::from_bits(x)
}
fn u(x: f32) -> u32 {
    x.to_bits()
}

fn sext(x: u32, w: u32) -> u32 {
    if w >= 32 {
        x
    } else {
        (((x << (32 - w)) as i32) >> (32 - w)) as u32
    }
}

fn bfe(x: u32, off: u32, w: u32, signed: bool) -> u32 {
    let off = off & 31;
    let w = w & 31;
    if w == 0 {
        return 0;
    }
    if off + w < 32 {
        let v = (x >> off) & ((1u32 << w) - 1);
        if signed { sext(v, w) } else { v }
    } else {
        let v = x >> off;
        if signed { ((x as i32) >> off) as u32 } else { v }
    }
}

fn half_to_f32(h: u16) -> f32 {
    let s = ((h >> 15) & 1) as u32;
    let e = ((h >> 10) & 0x1F) as i32;
    let m = (h & 0x3FF) as u32;
    let bits_ = if e == 0 {
        if m == 0 {
            s << 31
        } else {
            // subnormal
            let mut e2 = -14i32;
            let mut m2 = m;
            while m2 & 0x400 == 0 {
                m2 <<= 1;
                e2 -= 1;
            }
            (s << 31) | (((e2 + 127) as u32) << 23) | ((m2 & 0x3FF) << 13)
        }
    } else if e == 31 {
        (s << 31) | 0x7F80_0000 | (m << 13)
    } else {
        (s << 31) | (((e - 15 + 127) as u32) << 23) | (m << 13)
    };
    f32::from_bits(bits_)
}

fn f32_to_half(x: f32) -> u16 {
    // Round to nearest even.
    let b = x.to_bits();
    let s = ((b >> 16) & 0x8000) as u16;
    let e = ((b >> 23) & 0xFF) as i32;
    let m = b & 0x7F_FFFF;
    if e == 0xFF {
        return s | 0x7C00 | if m != 0 { 0x200 } else { 0 };
    }
    let e2 = e - 127 + 15;
    if e2 >= 31 {
        return s | 0x7C00;
    }
    if e2 <= 0 {
        if e2 < -10 {
            return s;
        }
        let m = m | 0x80_0000;
        let shift = (14 - e2) as u32;
        let hm = m >> shift;
        let rem = m & ((1 << shift) - 1);
        let half = 1 << (shift - 1);
        let hm = if rem > half || (rem == half && hm & 1 != 0) { hm + 1 } else { hm };
        return s | hm as u16;
    }
    let hm = m >> 13;
    let rem = m & 0x1FFF;
    let mut r = ((e2 as u32) << 10) | hm;
    if rem > 0x1000 || (rem == 0x1000 && hm & 1 != 0) {
        r += 1;
    }
    s | r as u16
}

/// Register value of a raw format component.
fn from_component(raw: u32, w: u32, num: Num) -> u32 {
    let max = if w >= 32 { u32::MAX as f64 } else { ((1u64 << w) - 1) as f64 };
    match num {
        Num::Uint => raw,
        Num::Sint => sext(raw, w),
        Num::Unorm | Num::Srgb => {
            let x = (raw as f64 / max) as f32;
            if num == Num::Srgb && w == 8 {
                u(if x <= 0.04045 { x / 12.92 } else { ((x + 0.055) / 1.055).powf(2.4) })
            } else {
                u(x)
            }
        }
        Num::Snorm => {
            let m = ((1u64 << (w - 1)) - 1) as f32;
            u((sext(raw, w) as i32 as f32 / m).max(-1.0))
        }
        Num::Uscaled => u(raw as f32),
        Num::Sscaled => u(sext(raw, w) as i32 as f32),
        Num::Float => match w {
            32 => raw,
            16 => u(half_to_f32(raw as u16)),
            11 | 10 => {
                let mb = w - 5;
                let man = raw & ((1 << mb) - 1);
                let ex = raw >> mb;
                u(half_to_f32(((ex << 10) | (man << (10 - mb))) as u16))
            }
            _ => raw,
        },
    }
}

/// Raw format component of a register value.
fn to_component(v: u32, w: u32, num: Num) -> u32 {
    let mask = if w >= 32 { u32::MAX } else { (1u32 << w) - 1 };
    let r = match num {
        Num::Uint | Num::Sint => v,
        Num::Unorm | Num::Srgb => {
            let k = ((1u64 << w) - 1) as f32;
            (f(v).clamp(0.0, 1.0) * k).round_ties_even() as u32
        }
        Num::Snorm => {
            let k = ((1u64 << (w - 1)) - 1) as f32;
            (f(v).clamp(-1.0, 1.0) * k).round_ties_even() as i32 as u32
        }
        Num::Uscaled => f(v) as u32,
        Num::Sscaled => f(v) as i32 as u32,
        Num::Float => match w {
            32 => v,
            16 => f32_to_half(f(v)) as u32,
            _ => v,
        },
    };
    r & mask
}

/// Splits a raw element (dwords) into register values per format: (values,
/// count of format components).
fn unpack(fmt: u16, raw: &[u32; 4]) -> Option<([u32; 4], usize)> {
    let fm = format::gfx10(fmt).filter(|f| !f.block && f.components() > 0)?;
    let mut out = [0u32, 0, 0, if fm.num.is_float() { u(1.0) } else { 1 }];
    let mut bo = 0u32;
    for (c, &w) in fm.bits.iter().enumerate() {
        let w = w as u32;
        if w == 0 {
            break;
        }
        let dw = raw[(bo / 32) as usize];
        let x = if w == 32 { dw } else { (dw >> (bo % 32)) & ((1u32 << w) - 1) };
        out[c] = from_component(x, w, fm.num);
        bo += w;
    }
    Some((out, fm.components()))
}

fn pack(fmt: u16, vals: &[u32; 4]) -> Option<([u32; 4], u32)> {
    let fm = format::gfx10(fmt).filter(|f| !f.block && f.components() > 0)?;
    let mut raw = [0u32; 4];
    let mut bo = 0u32;
    for (c, &w) in fm.bits.iter().enumerate() {
        let w = w as u32;
        if w == 0 {
            break;
        }
        let x = to_component(vals[c], w, fm.num);
        raw[(bo / 32) as usize] |= x << (bo % 32);
        bo += w;
    }
    Some((raw, fm.bytes()))
}

/// Applies a DST_SEL swizzle (3 bits per channel: 0 zero, 1 one, 4..7 x..w).
fn swizzle(vals: &[u32; 4], dst_sel: u32, float: bool) -> [u32; 4] {
    let mut r = [0u32; 4];
    for (k, o) in r.iter_mut().enumerate() {
        *o = match (dst_sel >> (3 * k)) & 7 {
            0 => 0,
            1 => {
                if float {
                    u(1.0)
                } else {
                    1
                }
            }
            s @ 4..=7 => vals[(s - 4) as usize],
            _ => 0,
        };
    }
    r
}

struct Vsharp {
    base: u64,
    stride: u32,
    num_records: u32,
    dst_sel: u32,
    format: u16,
    add_tid: bool,
    oob: u32,
}

fn vsharp(w: &[u32]) -> Vsharp {
    Vsharp {
        base: w[0] as u64 | ((bits(w[1], 0, 16) as u64) << 32),
        stride: bits(w[1], 16, 14),
        num_records: w[2],
        dst_sel: bits(w[3], 0, 12),
        format: bits(w[3], 12, 7) as u16,
        add_tid: bits(w[3], 23, 1) != 0,
        oob: bits(w[3], 28, 2),
    }
}

struct Tsharp {
    format: u16,
    width: u32,
    height: u32,
    depth: u32,
    dst_sel: u32,
    base_level: u32,
    last_level: u32,
    ty: u32,
    base_array: u32,
}

fn tsharp(w: &[u32; 8]) -> Tsharp {
    Tsharp {
        format: bits(w[1], 20, 9) as u16,
        width: (bits(w[1], 30, 2) | (bits(w[2], 0, 12) << 2)) + 1,
        height: bits(w[2], 14, 14) + 1,
        depth: bits(w[4], 0, 13) + 1,
        dst_sel: bits(w[3], 0, 12),
        base_level: bits(w[3], 12, 4),
        last_level: bits(w[3], 16, 4),
        ty: bits(w[3], 28, 4),
        base_array: bits(w[4], 16, 13),
    }
}

impl<'a, E: Env> Group<'a, E> {
    pub fn new(code: &'a [Inst], env: &'a mut E, launch: Launch) -> Group<'a, E> {
        let index = code.iter().enumerate().map(|(i, x)| (x.pc, i)).collect();
        let lanes = if launch.wave64 { 64 } else { 32 };
        let threads: u32 = launch.local_size.iter().map(|&x| x.max(1)).product();
        let nw = threads.div_ceil(lanes as u32);
        let mut waves = Vec::new();
        for w in 0..nw {
            let mut wave = Wave { s: [0; 128], v: vec![[0u32; 64]; 256], probe: [0; 64], scc: false, pc: 0, done: false, at_barrier: false, id: w };
            let nuser = ((launch.rsrc2 >> 1) & 0x1F) as usize;
            for (i, &x) in launch.user.iter().take(nuser.min(16)).enumerate() {
                wave.s[i] = x;
            }
            let mut sp = nuser;
            for k in 0..3 {
                if launch.rsrc2 & (1 << (7 + k)) != 0 {
                    wave.s[sp] = launch.group[k];
                    sp += 1;
                }
            }
            if launch.rsrc2 & (1 << 10) != 0 {
                // TG_SIZE: wave id [5:0], wave count [11:6].
                wave.s[sp] = w | (nw << 6);
            }
            let [sx, sy, _] = launch.local_size.map(|x| x.max(1));
            let mut exec = 0u64;
            for l in 0..lanes {
                let t = w * lanes as u32 + l as u32;
                if t < threads {
                    exec |= 1 << l;
                }
                wave.v[0][l] = t % sx;
                wave.v[1][l] = (t / sx) % sy;
                wave.v[2][l] = t / (sx * sy);
            }
            wave.set_exec(exec);
            waves.push(wave);
        }
        let lds = vec![0u8; launch.lds_bytes.max(4) as usize];
        Group { code, index, env, lds, waves, lanes, launch, steps: 0, trace: None }
    }

    /// Runs the group to completion (or until `max_steps` instructions).
    /// Waves take turns one instruction at a time (one hardware interleaving).
    pub fn run(&mut self, max_steps: u64) -> R<()> {
        loop {
            let mut progressed = false;
            loop {
                let mut any = false;
                for w in 0..self.waves.len() {
                    if !self.waves[w].done && !self.waves[w].at_barrier {
                        self.step(w)?;
                        any = true;
                        progressed = true;
                    }
                }
                if self.steps > max_steps {
                    return Err(format!("step limit {max_steps} reached"));
                }
                if !any {
                    break;
                }
            }
            if self.waves.iter().all(|w| w.done) {
                return Ok(());
            }
            if self.waves.iter().all(|w| w.done || w.at_barrier) {
                for w in self.waves.iter_mut() {
                    w.at_barrier = false;
                }
                continue;
            }
            if !progressed {
                return Err("no wave can make progress".into());
            }
        }
    }

    fn lit(&self, ins: &Inst) -> u32 {
        ins.lit.unwrap_or(0)
    }

    /// 32-bit scalar operand.
    fn sop(&self, w: usize, ins: &Inst, op: u16) -> R<u32> {
        let wv = &self.waves[w];
        Ok(match op {
            0..=127 => {
                if op == 125 {
                    0
                } else {
                    wv.s[op as usize]
                }
            }
            128..=192 => (op - 128) as u32,
            193..=208 => (-((op - 192) as i32)) as u32,
            240 => u(0.5),
            241 => u(-0.5),
            242 => u(1.0),
            243 => u(-1.0),
            244 => u(2.0),
            245 => u(-2.0),
            246 => u(4.0),
            247 => u(-4.0),
            248 => 0x3E22_F983,
            251 => (wv.s64(VCC) == 0) as u32,
            252 => (wv.exec() == 0) as u32,
            253 => wv.scc as u32,
            255 => self.lit(ins),
            _ => return Err(format!("{:#x}: unsupported scalar operand {op}", ins.pc)),
        })
    }

    /// 64-bit scalar operand (register pair, sign-extended inline integer).
    fn sop64(&self, w: usize, ins: &Inst, op: u16) -> R<u64> {
        let wv = &self.waves[w];
        Ok(match op {
            0..=127 => {
                if op == 125 {
                    0
                } else {
                    wv.s64(op as usize)
                }
            }
            128..=192 => (op - 128) as u64,
            193..=208 => (-((op - 192) as i64)) as u64,
            240..=248 => {
                let x = f32::from_bits(self.sop(w, ins, op)?) as f64;
                x.to_bits()
            }
            251..=253 => self.sop(w, ins, op)? as u64,
            255 => self.lit(ins) as u64,
            _ => return Err(format!("{:#x}: unsupported 64-bit operand {op}", ins.pc)),
        })
    }

    fn step(&mut self, w: usize) -> R<()> {
        let pc = self.waves[w].pc;
        let Some(ins) = self.code.get(pc).copied() else {
            self.waves[w].done = true;
            return Ok(());
        };
        self.steps += 1;
        let name = ins.base_name().ok_or_else(|| format!("{:#x}: unknown instruction {:?} op {:#x}", ins.pc, ins.enc, ins.op))?;
        self.waves[w].pc = pc + 1;
        match ins.enc {
            Enc::Sopp => self.sopp(w, &ins, name)?,
            Enc::Sop1 => self.sop1(w, &ins, name)?,
            Enc::Sop2 => self.sop2(w, &ins, name)?,
            Enc::Sopc => self.sopc(w, &ins, name)?,
            Enc::Sopk => self.sopk(w, &ins, name)?,
            Enc::Smem => self.smem(w, &ins, name)?,
            Enc::Vop1 | Enc::Vop2 | Enc::Vopc | Enc::Vop3 => self.valu(w, &ins, name)?,
            Enc::Vop3p => self.vop3p(w, &ins, name)?,
            Enc::Ds => self.ds(w, &ins, name)?,
            Enc::Mubuf | Enc::Mtbuf => self.buffer(w, &ins, name)?,
            Enc::Global => self.global(w, &ins, name)?,
            Enc::Mimg => self.image(w, &ins, name)?,
            _ => return Err(format!("{:#x}: {name} ({:?}) not emulated", ins.pc, ins.enc)),
        }
        if let Some((ppc, pv)) = self.launch.probe {
            if ins.pc == ppc {
                let wv = &mut self.waves[w];
                wv.probe = wv.v[pv as usize & 255];
            }
        }
        if let Some(t) = self.trace.as_mut() {
            let wv = &self.waves[w];
            t(wv.id, &ins, &wv.s, &wv.v);
        }
        Ok(())
    }

    /// Store data: VGPR `r` of lane `l`, or the probe value.
    fn sdata(&self, w: usize, r: usize, l: usize) -> u32 {
        if self.launch.probe.is_some() {
            self.waves[w].probe[l]
        } else {
            self.waves[w].v[r & 255][l]
        }
    }

    fn jump(&mut self, w: usize, ins: &Inst) -> R<()> {
        let t = ins.branch_target().ok_or("branch without target")?;
        let i = *self.index.get(&t).ok_or_else(|| format!("{:#x}: branch to {t:#x} outside the program", ins.pc))?;
        self.waves[w].pc = i;
        Ok(())
    }

    fn sopp(&mut self, w: usize, ins: &Inst, name: &str) -> R<()> {
        let wv = &self.waves[w];
        let take = match name {
            "s_endpgm" | "s_endpgm_saved" => {
                self.waves[w].done = true;
                return Ok(());
            }
            "s_branch" => true,
            "s_cbranch_scc0" => !wv.scc,
            "s_cbranch_scc1" => wv.scc,
            "s_cbranch_vccz" => wv.s64(VCC) == 0,
            "s_cbranch_vccnz" => wv.s64(VCC) != 0,
            "s_cbranch_execz" => wv.exec() == 0,
            "s_cbranch_execnz" => wv.exec() != 0,
            "s_barrier" => {
                if self.waves.len() > 1 {
                    self.waves[w].at_barrier = true;
                }
                return Ok(());
            }
            "s_nop" | "s_waitcnt" | "s_sleep" | "s_setprio" | "s_sendmsg" | "s_sendmsghalt" | "s_icache_inv" | "s_inst_prefetch" | "s_clause" | "s_code_end" | "s_waitcnt_depctr" | "s_round_mode" | "s_denorm_mode" | "s_ttracedata" | "s_ttracedata_imm" | "s_incperflevel" | "s_decperflevel" | "s_wait_idle" => return Ok(()),
            _ => return Err(format!("{:#x}: {name} not emulated", ins.pc)),
        };
        if take {
            self.jump(w, ins)?;
        }
        Ok(())
    }

    fn sop1(&mut self, w: usize, ins: &Inst, name: &str) -> R<()> {
        let d = ins.sdst() as usize;
        let s = ins.ssrc0();
        if let Some(stem) = name.strip_suffix("_saveexec_b64").or_else(|| name.strip_suffix("_saveexec_b32")).or_else(|| name.strip_suffix("_wrexec_b64")).or_else(|| name.strip_suffix("_wrexec_b32")) {
            let wide = name.ends_with("_b64");
            let a = if wide { self.sop64(w, ins, s)? } else { self.sop(w, ins, s)? as u64 };
            let old = self.waves[w].exec();
            let e = if wide { old } else { old & 0xFFFF_FFFF };
            let n = match stem {
                "s_and" => a & e,
                "s_or" => a | e,
                "s_xor" => a ^ e,
                "s_andn2" => a & !e,
                "s_orn2" => a | !e,
                "s_nand" => !(a & e),
                "s_nor" => !(a | e),
                "s_xnor" => !(a ^ e),
                "s_andn1" => !a & e,
                "s_orn1" => !a | e,
                _ => return Err(format!("{:#x}: {name} not emulated", ins.pc)),
            };
            let n = if wide { n } else { (n & 0xFFFF_FFFF) | (old & !0xFFFF_FFFF) };
            let wv = &mut self.waves[w];
            if name.contains("wrexec") {
                if wide { wv.set_s64(d, n) } else { wv.set_s(d, n as u32) }
            } else if wide {
                wv.set_s64(d, old)
            } else {
                wv.set_s(d, old as u32)
            }
            wv.set_exec(n);
            wv.scc = if wide { n != 0 } else { n as u32 != 0 };
            return Ok(());
        }
        let a32 = || self.sop(w, ins, s);
        match name {
            "s_mov_b32" => {
                let x = a32()?;
                self.waves[w].set_s(d, x);
            }
            "s_mov_b64" => {
                let x = self.sop64(w, ins, s)?;
                self.waves[w].set_s64(d, x);
            }
            "s_cmov_b32" => {
                let x = a32()?;
                if self.waves[w].scc {
                    self.waves[w].set_s(d, x);
                }
            }
            "s_cmov_b64" => {
                let x = self.sop64(w, ins, s)?;
                if self.waves[w].scc {
                    self.waves[w].set_s64(d, x);
                }
            }
            "s_not_b32" => {
                let x = !a32()?;
                let wv = &mut self.waves[w];
                wv.set_s(d, x);
                wv.scc = x != 0;
            }
            "s_not_b64" => {
                let x = !self.sop64(w, ins, s)?;
                let wv = &mut self.waves[w];
                wv.set_s64(d, x);
                wv.scc = x != 0;
            }
            "s_brev_b32" => {
                let x = a32()?.reverse_bits();
                self.waves[w].set_s(d, x);
            }
            "s_brev_b64" => {
                let x = self.sop64(w, ins, s)?.reverse_bits();
                self.waves[w].set_s64(d, x);
            }
            "s_bcnt0_i32_b32" | "s_bcnt1_i32_b32" => {
                let x = a32()?;
                let r = if name.starts_with("s_bcnt0") { (!x).count_ones() } else { x.count_ones() };
                let wv = &mut self.waves[w];
                wv.set_s(d, r);
                wv.scc = r != 0;
            }
            "s_bcnt0_i32_b64" | "s_bcnt1_i32_b64" => {
                let x = self.sop64(w, ins, s)?;
                let r = if name.starts_with("s_bcnt0") { (!x).count_ones() } else { x.count_ones() };
                let wv = &mut self.waves[w];
                wv.set_s(d, r);
                wv.scc = r != 0;
            }
            "s_ff0_i32_b32" | "s_ff1_i32_b32" => {
                let x = a32()?;
                let x = if name.starts_with("s_ff0") { !x } else { x };
                let r = if x == 0 { u32::MAX } else { x.trailing_zeros() };
                self.waves[w].set_s(d, r);
            }
            "s_ff0_i32_b64" | "s_ff1_i32_b64" => {
                let x = self.sop64(w, ins, s)?;
                let x = if name.starts_with("s_ff0") { !x } else { x };
                let r = if x == 0 { u32::MAX } else { x.trailing_zeros() };
                self.waves[w].set_s(d, r);
            }
            "s_flbit_i32_b32" => {
                let x = a32()?;
                self.waves[w].set_s(d, if x == 0 { u32::MAX } else { x.leading_zeros() });
            }
            "s_flbit_i32" => {
                let x = a32()? as i32;
                let y = if x < 0 { !x } else { x } as u32;
                self.waves[w].set_s(d, if x == 0 || x == -1 { u32::MAX } else { y.leading_zeros() });
            }
            "s_sext_i32_i8" => {
                let x = sext(a32()?, 8);
                self.waves[w].set_s(d, x);
            }
            "s_sext_i32_i16" => {
                let x = sext(a32()?, 16);
                self.waves[w].set_s(d, x);
            }
            "s_bitset0_b32" | "s_bitset1_b32" => {
                let b = a32()? & 31;
                let wv = &mut self.waves[w];
                let old = wv.s[d.min(127)];
                wv.set_s(d, if name.ends_with("0_b32") { old & !(1 << b) } else { old | (1 << b) });
            }
            "s_abs_i32" => {
                let x = (a32()? as i32).unsigned_abs();
                let wv = &mut self.waves[w];
                wv.set_s(d, x);
                wv.scc = x != 0;
            }
            "s_getpc_b64" => {
                let next = self.code.get(self.waves[w].pc).map(|i| i.pc).unwrap_or(ins.pc + 4);
                let a = self.launch.code_base + next as u64;
                self.waves[w].set_s64(d, a);
            }
            "s_movrels_b32" => {
                let m0 = self.waves[w].s[M0];
                let r = (s as u32 + m0) as usize;
                let x = self.waves[w].s[r.min(127)];
                self.waves[w].set_s(d, x);
            }
            "s_movreld_b32" => {
                let x = a32()?;
                let m0 = self.waves[w].s[M0] as usize;
                self.waves[w].set_s(d + m0, x);
            }
            _ => return Err(format!("{:#x}: {name} not emulated", ins.pc)),
        }
        Ok(())
    }

    fn sop2(&mut self, w: usize, ins: &Inst, name: &str) -> R<()> {
        let d = ins.sdst() as usize;
        let (s0, s1) = (ins.ssrc0(), ins.ssrc1());
        let wide = name.ends_with("_b64") || name.ends_with("_u64") || name.ends_with("_i64");
        if wide && !name.starts_with("s_lsh") && !name.starts_with("s_ashr") && !name.starts_with("s_bfe") && !name.starts_with("s_bfm") {
            let a = self.sop64(w, ins, s0)?;
            let b = self.sop64(w, ins, s1)?;
            let scc = self.waves[w].scc;
            let r = match name {
                "s_and_b64" => a & b,
                "s_or_b64" => a | b,
                "s_xor_b64" => a ^ b,
                "s_andn2_b64" => a & !b,
                "s_orn2_b64" => a | !b,
                "s_nand_b64" => !(a & b),
                "s_nor_b64" => !(a | b),
                "s_xnor_b64" => !(a ^ b),
                "s_cselect_b64" => {
                    self.waves[w].set_s64(d, if scc { a } else { b });
                    return Ok(());
                }
                _ => return Err(format!("{:#x}: {name} not emulated", ins.pc)),
            };
            let wv = &mut self.waves[w];
            wv.set_s64(d, r);
            wv.scc = r != 0;
            return Ok(());
        }
        if wide {
            let a = self.sop64(w, ins, s0)?;
            let b = self.sop(w, ins, s1)?;
            let r = match name {
                "s_lshl_b64" => a << (b & 63),
                "s_lshr_b64" => a >> (b & 63),
                "s_ashr_i64" => ((a as i64) >> (b & 63)) as u64,
                "s_bfm_b64" => {
                    let a = self.sop(w, ins, s0)?;
                    let r = (1u64.checked_shl(a & 63).unwrap_or(0).wrapping_sub(1)) << (b & 63);
                    self.waves[w].set_s64(d, r);
                    return Ok(());
                }
                "s_bfe_u64" | "s_bfe_i64" => {
                    let off = b & 63;
                    let wd = (b >> 16) & 0x7F;
                    let x = a >> off;
                    let r = if wd == 0 {
                        0
                    } else if wd >= 64 {
                        x
                    } else {
                        let v = x & ((1u64 << wd) - 1);
                        if name == "s_bfe_i64" { (((v << (64 - wd)) as i64) >> (64 - wd)) as u64 } else { v }
                    };
                    r
                }
                _ => return Err(format!("{:#x}: {name} not emulated", ins.pc)),
            };
            let wv = &mut self.waves[w];
            wv.set_s64(d, r);
            wv.scc = r != 0;
            return Ok(());
        }
        let a = self.sop(w, ins, s0)?;
        let b = self.sop(w, ins, s1)?;
        let scc = self.waves[w].scc;
        let (r, nscc): (u32, Option<bool>) = match name {
            "s_add_u32" => {
                let (r, c) = a.overflowing_add(b);
                (r, Some(c))
            }
            "s_sub_u32" => {
                let (r, c) = a.overflowing_sub(b);
                (r, Some(c))
            }
            "s_add_i32" => {
                let (r, o) = (a as i32).overflowing_add(b as i32);
                (r as u32, Some(o))
            }
            "s_sub_i32" => {
                let (r, o) = (a as i32).overflowing_sub(b as i32);
                (r as u32, Some(o))
            }
            "s_addc_u32" => {
                let r = a as u64 + b as u64 + scc as u64;
                (r as u32, Some(r >> 32 != 0))
            }
            "s_subb_u32" => {
                let r = (a as u64).wrapping_sub(b as u64).wrapping_sub(scc as u64);
                (r as u32, Some((b as u64 + scc as u64) > a as u64))
            }
            "s_min_i32" => ((a as i32).min(b as i32) as u32, Some((a as i32) < (b as i32))),
            "s_max_i32" => ((a as i32).max(b as i32) as u32, Some((a as i32) > (b as i32))),
            "s_min_u32" => (a.min(b), Some(a < b)),
            "s_max_u32" => (a.max(b), Some(a > b)),
            "s_cselect_b32" => (if scc { a } else { b }, None),
            "s_and_b32" => (a & b, Some(a & b != 0)),
            "s_or_b32" => (a | b, Some(a | b != 0)),
            "s_xor_b32" => (a ^ b, Some(a ^ b != 0)),
            "s_andn2_b32" => (a & !b, Some(a & !b != 0)),
            "s_orn2_b32" => (a | !b, Some(a | !b != 0)),
            "s_nand_b32" => (!(a & b), Some(!(a & b) != 0)),
            "s_nor_b32" => (!(a | b), Some(!(a | b) != 0)),
            "s_xnor_b32" => (!(a ^ b), Some(!(a ^ b) != 0)),
            "s_lshl_b32" => {
                let r = a << (b & 31);
                (r, Some(r != 0))
            }
            "s_lshr_b32" => {
                let r = a >> (b & 31);
                (r, Some(r != 0))
            }
            "s_ashr_i32" => {
                let r = ((a as i32) >> (b & 31)) as u32;
                (r, Some(r != 0))
            }
            "s_bfm_b32" => ((1u32.checked_shl(a & 31).unwrap_or(0).wrapping_sub(1)) << (b & 31), None),
            "s_mul_i32" => (a.wrapping_mul(b), None),
            "s_mul_hi_u32" => (((a as u64 * b as u64) >> 32) as u32, None),
            "s_mul_hi_i32" => (((a as i32 as i64 * b as i32 as i64) >> 32) as u32, None),
            "s_bfe_u32" | "s_bfe_i32" => {
                let r = bfe(a, b & 31, (b >> 16) & 0x7F, name == "s_bfe_i32");
                let r = if (b >> 16) & 0x7F >= 32 { if name == "s_bfe_i32" { ((a as i32) >> (b & 31)) as u32 } else { a >> (b & 31) } } else { r };
                (r, Some(r != 0))
            }
            "s_absdiff_i32" => {
                let r = (a as i32).wrapping_sub(b as i32).unsigned_abs();
                (r, Some(r != 0))
            }
            "s_lshl1_add_u32" | "s_lshl2_add_u32" | "s_lshl3_add_u32" | "s_lshl4_add_u32" => {
                let n = name.as_bytes()[6] - b'0';
                let r = ((a as u64) << n) + b as u64;
                (r as u32, Some(r >> 32 != 0))
            }
            "s_pack_ll_b32_b16" => ((a & 0xFFFF) | (b << 16), None),
            "s_pack_lh_b32_b16" => ((a & 0xFFFF) | (b & 0xFFFF_0000), None),
            "s_pack_hh_b32_b16" => ((a >> 16) | (b & 0xFFFF_0000), None),
            _ => return Err(format!("{:#x}: {name} not emulated", ins.pc)),
        };
        let wv = &mut self.waves[w];
        wv.set_s(d, r);
        if let Some(c) = nscc {
            wv.scc = c;
        }
        Ok(())
    }

    fn sopc(&mut self, w: usize, ins: &Inst, name: &str) -> R<()> {
        let (s0, s1) = (ins.ssrc0(), ins.ssrc1());
        let c = if name.ends_with("_u64") {
            let a = self.sop64(w, ins, s0)?;
            let b = self.sop64(w, ins, s1)?;
            match name {
                "s_cmp_eq_u64" => a == b,
                "s_cmp_lg_u64" | "s_cmp_ne_u64" => a != b,
                _ => return Err(format!("{:#x}: {name} not emulated", ins.pc)),
            }
        } else if name.starts_with("s_bitcmp") {
            if name.ends_with("_b64") {
                let a = self.sop64(w, ins, s0)?;
                let b = self.sop(w, ins, s1)? & 63;
                let bit = (a >> b) & 1;
                if name.starts_with("s_bitcmp0") { bit == 0 } else { bit == 1 }
            } else {
                let a = self.sop(w, ins, s0)?;
                let b = self.sop(w, ins, s1)? & 31;
                let bit = (a >> b) & 1;
                if name.starts_with("s_bitcmp0") { bit == 0 } else { bit == 1 }
            }
        } else {
            let a = self.sop(w, ins, s0)?;
            let b = self.sop(w, ins, s1)?;
            let (cmp, ty) = name["s_cmp_".len()..].rsplit_once('_').unwrap_or(("", ""));
            int_cmp(cmp, a, b, ty == "i32").ok_or_else(|| format!("{:#x}: {name} not emulated", ins.pc))?
        };
        self.waves[w].scc = c;
        Ok(())
    }

    fn sopk(&mut self, w: usize, ins: &Inst, name: &str) -> R<()> {
        let d = ins.sdst() as usize;
        let imm_s = ins.simm16() as i16 as i32 as u32;
        let imm_u = ins.simm16() as u32;
        match name {
            "s_movk_i32" => self.waves[w].set_s(d, imm_s),
            "s_cmovk_i32" => {
                if self.waves[w].scc {
                    self.waves[w].set_s(d, imm_s);
                }
            }
            "s_addk_i32" => {
                let a = self.waves[w].s[d.min(127)];
                let (r, o) = (a as i32).overflowing_add(imm_s as i32);
                let wv = &mut self.waves[w];
                wv.set_s(d, r as u32);
                wv.scc = o;
            }
            "s_mulk_i32" => {
                let a = self.waves[w].s[d.min(127)];
                self.waves[w].set_s(d, a.wrapping_mul(imm_s));
            }
            "s_getreg_b32" => self.waves[w].set_s(d, 0),
            "s_setreg_b32" | "s_setreg_imm32_b32" | "s_waitcnt_vscnt" | "s_waitcnt_vmcnt" | "s_waitcnt_expcnt" | "s_waitcnt_lgkmcnt" | "s_version" | "s_subvector_loop_begin" | "s_subvector_loop_end" => {}
            n if n.starts_with("s_cmpk_") => {
                let (cmp, ty) = n["s_cmpk_".len()..].rsplit_once('_').unwrap_or(("", ""));
                let signed = ty == "i32";
                let a = self.waves[w].s[d.min(127)];
                let c = int_cmp(cmp, a, if signed { imm_s } else { imm_u }, signed).ok_or_else(|| format!("{:#x}: {name} not emulated", ins.pc))?;
                self.waves[w].scc = c;
            }
            _ => return Err(format!("{:#x}: {name} not emulated", ins.pc)),
        }
        Ok(())
    }

    fn smem(&mut self, w: usize, ins: &Inst, name: &str) -> R<()> {
        let n = [("x16", 16), ("x8", 8), ("x4", 4), ("x2", 2)].iter().find(|(s, _)| name.ends_with(s)).map(|x| x.1).unwrap_or(1usize);
        let d = ins.sdst() as usize;
        let sb = ins.smem_sbase() as usize;
        let soff = ins.smem_soffset();
        let off = ins.smem_offset() as i64 + if soff != 125 { self.sop(w, ins, soff)? as i64 } else { 0 };
        if name.starts_with("s_load_dword") {
            let base = self.waves[w].s64(sb) & 0xFFFF_FFFF_FFFF;
            let a = (base as i64 + off) as u64 & !3;
            let mut buf = vec![0u8; 4 * n];
            if !self.env.read(a, &mut buf) {
                return Err(format!("{:#x}: s_load from unmapped {a:#x}", ins.pc));
            }
            for k in 0..n {
                let x = u32::from_le_bytes(buf[4 * k..4 * k + 4].try_into().unwrap());
                self.waves[w].set_s(d + k, x);
            }
        } else if name.starts_with("s_buffer_load_dword") {
            let v = vsharp(&self.waves[w].s[sb..sb + 4]);
            // Bytes the descriptor covers (records x stride when strided;
            // OOB_SELECT 2 checks nothing), as the runtime binds it.
            let limit = if v.oob == 2 { u64::MAX } else if v.stride == 0 { v.num_records as u64 } else { v.num_records as u64 * v.stride as u64 };
            for k in 0..n {
                let o = (off as u64 & !3) + 4 * k as u64;
                let x = if o + 4 <= limit {
                    let mut b = [0u8; 4];
                    self.env.read(v.base + o, &mut b);
                    u32::from_le_bytes(b)
                } else {
                    0
                };
                self.waves[w].set_s(d + k, x);
            }
        } else if matches!(name, "s_dcache_inv" | "s_gl1_inv" | "s_dcache_wb") {
        } else if name.starts_with("s_memtime") || name.starts_with("s_memrealtime") {
            self.waves[w].set_s64(d, self.steps);
        } else {
            return Err(format!("{:#x}: {name} not emulated", ins.pc));
        }
        Ok(())
    }

    // -----------------------------------------------------------------------
    // Vector ALU

    /// VOP3P: the mixed-precision FMAs (f32 sources, or f16 halves picked by
    /// op_sel when op_sel_hi is set; neg_hi is abs, as in the translator).
    fn vop3p(&mut self, w: usize, ins: &Inst, name: &str) -> R<()> {
        if !matches!(name, "v_fma_mix_f32" | "v_fma_mixlo_f16" | "v_fma_mixhi_f16") {
            return Err(format!("{:#x}: {name} (Vop3p) not emulated", ins.pc));
        }
        let v = Vop { dst: ins.vdst(), src: [ins.src(0), ins.src(1), ins.src(2)], abs: 0, neg: 0, clamp: false, omod: 0, opsel: 0, sdst: VCC as u16, vop3: true, sdwa: None, dpp16: None, dpp8: None };
        let neg = bits(ins.w1, 29, 3);
        let neg_hi = bits(ins.w0, 8, 3);
        let opsel = bits(ins.w0, 11, 3);
        let opsel_hi = bits(ins.w1, 27, 2) | (bits(ins.w0, 14, 1) << 2);
        let clamp = bits(ins.w0, 15, 1) != 0;
        for l in 0..self.lanes {
            if !self.lane_writes(w, &v, l) {
                continue;
            }
            let mut s = [0f32; 3];
            for (k, sk) in s.iter_mut().enumerate() {
                let raw = self.vsrc(w, ins, &v, k, l)?;
                let mut x = if opsel_hi >> k & 1 != 0 { half_to_f32(if opsel >> k & 1 != 0 { (raw >> 16) as u16 } else { raw as u16 }) } else { f32::from_bits(raw) };
                if neg_hi >> k & 1 != 0 {
                    x = x.abs();
                }
                if neg >> k & 1 != 0 {
                    x = -x;
                }
                *sk = x;
            }
            let mut r = s[0].mul_add(s[1], s[2]);
            if clamp {
                r = if r.is_nan() { 0.0 } else { r.clamp(0.0, 1.0) };
            }
            let d = (v.dst & 255) as usize;
            let old = self.waves[w].v[d][l];
            self.waves[w].v[d][l] = match name {
                "v_fma_mix_f32" => r.to_bits(),
                "v_fma_mixlo_f16" => (old & 0xFFFF_0000) | f32_to_half(r) as u32,
                _ => (old & 0xFFFF) | ((f32_to_half(r) as u32) << 16),
            };
        }
        Ok(())
    }

    /// v_permlane16_b32 / v_permlanex16_b32: lane i of each row of 16 reads
    /// lane sel[i] (4 bits from src1 for i < 8, src2 above) of the same row
    /// or (x16) of the other row of its 32-lane half; op_sel[0] fetches from
    /// inactive lanes, op_sel[1] (bound control) writes the fetched value
    /// even when the source lane is invalid, which otherwise keeps the
    /// destination.
    fn permlane(&mut self, w: usize, ins: &Inst, v: &Vop, name: &str) -> R<()> {
        let s1 = self.sop(w, ins, v.src[1])?;
        let s2 = self.sop(w, ins, v.src[2])?;
        let (fi, bc) = (v.opsel & 1 != 0, v.opsel & 2 != 0);
        let exec = self.waves[w].exec();
        let x = (v.src[0].saturating_sub(256) & 255) as usize;
        let before: Vec<u32> = self.waves[w].v[x].to_vec();
        let d = (v.dst & 255) as usize;
        for l in 0..self.lanes {
            if exec >> l & 1 == 0 {
                continue;
            }
            let i = l & 15;
            let selw = if i >= 8 { s2 } else { s1 };
            let sel = ((selw >> (4 * (i & 7))) & 15) as usize;
            let mut row = l & !15;
            if name == "v_permlanex16_b32" {
                row ^= 16;
            }
            let src = row | sel;
            let valid = fi || exec >> src & 1 != 0;
            if valid {
                self.waves[w].v[d][l] = before[src];
            } else if bc {
                self.waves[w].v[d][l] = 0;
            }
        }
        Ok(())
    }

    fn vop(&self, ins: &Inst) -> Vop {
        let mut v = Vop { dst: ins.vdst(), src: [ins.src(0), ins.src(1), 0], abs: 0, neg: 0, clamp: false, omod: 0, opsel: 0, sdst: VCC as u16, vop3: false, sdwa: None, dpp16: None, dpp8: None };
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
                    v.sdst = ins.vdst();
                }
            }
            _ => match ins.ext {
                Ext::Sdwa(d) => {
                    v.src[0] = if bits(d, 23, 1) != 0 { bits(d, 0, 8) as u16 } else { 256 + bits(d, 0, 8) as u16 };
                    if ins.enc != Enc::Vop1 {
                        let s1 = bits(ins.w0, 9, 8) as u16;
                        v.src[1] = if bits(d, 31, 1) != 0 { s1 } else { 256 + s1 };
                    }
                    v.neg = bits(d, 20, 1) | (bits(d, 28, 1) << 1);
                    v.abs = bits(d, 21, 1) | (bits(d, 29, 1) << 1);
                    let (ds, du) = if ins.enc == Enc::Vopc {
                        if bits(d, 15, 1) != 0 {
                            v.sdst = bits(d, 8, 7) as u16;
                        }
                        (6, 0)
                    } else {
                        v.clamp = bits(d, 13, 1) != 0;
                        v.omod = bits(d, 14, 2);
                        (bits(d, 8, 3), bits(d, 11, 2))
                    };
                    v.sdwa = Some((ds, du, [bits(d, 16, 3), bits(d, 24, 3)], [bits(d, 19, 1) != 0, bits(d, 27, 1) != 0]));
                }
                Ext::Dpp16(d) => {
                    v.src[0] = 256 + bits(d, 0, 8) as u16;
                    v.neg = bits(d, 20, 1) | (bits(d, 22, 1) << 1);
                    v.abs = bits(d, 21, 1) | (bits(d, 23, 1) << 1);
                    v.dpp16 = Some(d);
                }
                Ext::Dpp8(d, fi) => {
                    v.src[0] = 256 + bits(d, 0, 8) as u16;
                    v.dpp8 = Some((d, fi));
                }
                Ext::None => {}
            },
        }
        v
    }

    /// Raw source `k` of lane `l` (DPP, SDWA selection applied).
    fn vsrc(&self, w: usize, ins: &Inst, v: &Vop, k: usize, l: usize) -> R<u32> {
        let op = v.src[k];
        let raw = if k == 0 && (v.dpp16.is_some() || v.dpp8.is_some()) {
            let src_lane = self.dpp_lane(w, v, l);
            match src_lane {
                Some(sl) => self.waves[w].v[(op - 256) as usize][sl],
                None => 0,
            }
        } else if op >= 256 {
            self.waves[w].v[(op - 256) as usize][l]
        } else {
            self.sop(w, ins, op)?
        };
        Ok(match v.sdwa {
            Some((_, _, sel, sx)) if k < 2 && sel[k] != 6 => {
                let (off, wd) = if sel[k] < 4 { (sel[k] * 8, 8) } else { ((sel[k] - 4) * 16, 16) };
                let x = (raw >> off) & ((1u32 << wd) - 1);
                if sx[k] { sext(x, wd) } else { x }
            }
            _ => raw,
        })
    }

    /// DPP source lane (None: invalid source, reads 0 / disables the lane).
    fn dpp_lane(&self, w: usize, v: &Vop, l: usize) -> Option<usize> {
        let exec = self.waves[w].exec();
        if let Some((d, fi)) = v.dpp8 {
            let s = (l & !7) | ((d >> 8 >> (3 * (l & 7))) & 7) as usize;
            return if fi || exec >> s & 1 != 0 { Some(s) } else { None };
        }
        let d = v.dpp16?;
        let ctrl = bits(d, 8, 9);
        let fi = bits(d, 18, 1) != 0;
        let row = l & !15;
        let i = l & 15;
        let s = match ctrl {
            0x000..=0x0FF => Some((i & !3) | ((ctrl >> (2 * (i & 3))) & 3) as usize),
            0x101..=0x10F => {
                let n = (ctrl & 15) as usize;
                (i + n <= 15).then_some(i + n)
            }
            0x111..=0x11F => {
                let n = (ctrl & 15) as usize;
                (i >= n).then(|| i - n)
            }
            0x121..=0x12F => Some((i + 16 - (ctrl & 15) as usize) & 15),
            0x140 => Some(15 - i),
            0x141 => Some((i & 8) | (7 - (i & 7))),
            0x150..=0x15F => Some((ctrl & 15) as usize),
            0x160..=0x16F => Some(i ^ (ctrl & 15) as usize),
            _ => None,
        }?;
        let s = row + s;
        if fi || exec >> s & 1 != 0 {
            Some(s)
        } else {
            None
        }
    }

    /// Whether lane `l` writes its result (EXEC, DPP row/bank masks and validity).
    fn lane_writes(&self, w: usize, v: &Vop, l: usize) -> bool {
        if self.waves[w].exec() >> l & 1 == 0 {
            return false;
        }
        if let Some(d) = v.dpp16 {
            let row_mask = bits(d, 28, 4);
            let bank_mask = bits(d, 24, 4);
            if row_mask >> ((l >> 4) & 3) & 1 == 0 || bank_mask >> ((l & 15) >> 2) & 1 == 0 {
                return false;
            }
            let bc = bits(d, 19, 1) != 0;
            if !bc && self.dpp_lane(w, v, l).is_none() {
                return false;
            }
        }
        true
    }

    /// 16-bit source `k` of lane `l` in the low bits: the low half, or the
    /// high half with VOP3 op_sel; inline float constants are f16 encodings.
    fn src16(&self, w: usize, ins: &Inst, v: &Vop, k: usize, l: usize) -> R<u32> {
        const HALF_CONSTS: [u32; 9] = [0x3800, 0xB800, 0x3C00, 0xBC00, 0x4000, 0xC000, 0x4400, 0xC400, 0x3118];
        let op = v.src[k];
        let raw = match op {
            240..=248 if !(k == 0 && (v.dpp16.is_some() || v.dpp8.is_some())) => HALF_CONSTS[(op - 240) as usize],
            _ => self.vsrc(w, ins, v, k, l)?,
        };
        let raw = if v.vop3 && v.opsel >> k & 1 != 0 { raw >> 16 } else { raw };
        Ok(raw & 0xFFFF)
    }

    /// f16 source `k` of lane `l` with abs/neg.
    fn fsrc16(&self, w: usize, ins: &Inst, v: &Vop, k: usize, l: usize) -> R<f32> {
        let mut x = half_to_f32(self.src16(w, ins, v, k, l)? as u16);
        if v.abs >> k & 1 != 0 {
            x = x.abs();
        }
        if v.neg >> k & 1 != 0 {
            x = -x;
        }
        Ok(x)
    }

    fn fsrc(&self, w: usize, ins: &Inst, v: &Vop, k: usize, l: usize) -> R<f32> {
        let mut x = f(self.vsrc(w, ins, v, k, l)?);
        if v.abs >> k & 1 != 0 {
            x = x.abs();
        }
        if v.neg >> k & 1 != 0 {
            x = -x;
        }
        Ok(x)
    }

    fn fres(v: &Vop, x: f32) -> u32 {
        let mut x = match v.omod {
            1 => x * 2.0,
            2 => x * 4.0,
            3 => x * 0.5,
            _ => x,
        };
        if v.clamp {
            x = if x.is_nan() { 0.0 } else { x.clamp(0.0, 1.0) };
        }
        u(x)
    }

    /// Writes a 32-bit result to lane `l` of the VGPR destination (SDWA dst).
    fn vwrite(&mut self, w: usize, v: &Vop, l: usize, x: u32) {
        let d = (v.dst & 255) as usize;
        let x = match v.sdwa {
            Some((ds, du, _, _)) if ds != 6 => {
                let (off, wd) = if ds < 4 { (ds * 8, 8) } else { ((ds - 4) * 16, 16) };
                let mask = ((1u64 << wd) - 1) as u32;
                let field = (x & mask) << off;
                match du {
                    2 => (self.waves[w].v[d][l] & !(mask << off)) | field,
                    1 => (sext(x & mask, wd) << off) & !((1u32 << off).wrapping_sub(1)),
                    _ => field,
                }
            }
            _ => x,
        };
        self.waves[w].v[d][l] = x;
    }

    fn valu(&mut self, w: usize, ins: &Inst, name: &str) -> R<()> {
        let v = self.vop(ins);
        let lanes = self.lanes;
        // Compares.
        if let Some(rest) = name.strip_prefix("v_cmpx_").or_else(|| name.strip_prefix("v_cmp_")) {
            let x = name.starts_with("v_cmpx_");
            let (cmp, ty) = rest.rsplit_once('_').ok_or("compare name")?;
            let mut mask = 0u64;
            for l in 0..lanes {
                if self.waves[w].exec() >> l & 1 == 0 {
                    continue;
                }
                let c = match ty {
                    "f32" => {
                        let a = self.fsrc(w, ins, &v, 0, l)?;
                        let b = self.fsrc(w, ins, &v, 1, l)?;
                        if cmp == "class" {
                            class_f32(a, self.vsrc(w, ins, &v, 1, l)?)
                        } else {
                            fcmp(cmp, a, b).ok_or_else(|| format!("{:#x}: {name} not emulated", ins.pc))?
                        }
                    }
                    "i32" | "u32" => {
                        let a = self.vsrc(w, ins, &v, 0, l)?;
                        let b = self.vsrc(w, ins, &v, 1, l)?;
                        int_cmp(cmp, a, b, ty == "i32").ok_or_else(|| format!("{:#x}: {name} not emulated", ins.pc))?
                    }
                    "i16" | "u16" => {
                        let a = self.src16(w, ins, &v, 0, l)?;
                        let b = self.src16(w, ins, &v, 1, l)?;
                        let (a, b) = if ty == "i16" { (sext(a, 16), sext(b, 16)) } else { (a, b) };
                        int_cmp(cmp, a, b, ty == "i16").ok_or_else(|| format!("{:#x}: {name} not emulated", ins.pc))?
                    }
                    "f16" if cmp != "class" => {
                        let a = self.fsrc16(w, ins, &v, 0, l)?;
                        let b = self.fsrc16(w, ins, &v, 1, l)?;
                        fcmp(cmp, a, b).ok_or_else(|| format!("{:#x}: {name} not emulated", ins.pc))?
                    }
                    _ => return Err(format!("{:#x}: {name} not emulated", ins.pc)),
                };
                if c {
                    mask |= 1 << l;
                }
            }
            let wv = &mut self.waves[w];
            if x {
                wv.set_exec(mask);
            } else {
                wv.set_s64(v.sdst as usize, mask);
            }
            return Ok(());
        }
        match name {
            "v_readfirstlane_b32" => {
                let exec = self.waves[w].exec();
                let l = if exec == 0 { 0 } else { exec.trailing_zeros() as usize };
                let x = self.vsrc(w, ins, &v, 0, l)?;
                self.waves[w].set_s(v.dst as usize, x);
                return Ok(());
            }
            "v_readlane_b32" => {
                let l = (self.sop(w, ins, v.src[1])? as usize) & (lanes - 1);
                let x = self.vsrc(w, ins, &v, 0, l)?;
                self.waves[w].set_s(v.dst as usize, x);
                return Ok(());
            }
            "v_writelane_b32" => {
                let x = self.sop(w, ins, v.src[0])?;
                let l = (self.sop(w, ins, v.src[1])? as usize) & (lanes - 1);
                self.waves[w].v[(v.dst & 255) as usize][l] = x;
                return Ok(());
            }
            "v_nop" | "v_pipeflush" | "v_clrexcp" => return Ok(()),
            "v_permlane16_b32" | "v_permlanex16_b32" => return self.permlane(w, ins, &v, name),
            _ => {}
        }
        // Carry-out / carry-in arithmetic (VOP2 writes VCC, VOP3B an SGPR pair).
        if matches!(name, "v_add_co_u32" | "v_sub_co_u32" | "v_subrev_co_u32" | "v_addc_co_u32" | "v_subb_co_u32" | "v_subbrev_co_u32") {
            let cin_src = if v.vop3 { v.src[2] as usize } else { VCC };
            let cin = self.waves[w].s64(cin_src);
            let mut cout = 0u64;
            for l in 0..lanes {
                if !self.lane_writes(w, &v, l) {
                    continue;
                }
                let a = self.vsrc(w, ins, &v, 0, l)? as u64;
                let b = self.vsrc(w, ins, &v, 1, l)? as u64;
                let c = (cin >> l) & 1;
                let (r, carry) = match name {
                    "v_add_co_u32" => (a + b, (a + b) >> 32 != 0),
                    "v_sub_co_u32" => (a.wrapping_sub(b), b > a),
                    "v_subrev_co_u32" => (b.wrapping_sub(a), a > b),
                    "v_addc_co_u32" => (a + b + c, (a + b + c) >> 32 != 0),
                    "v_subb_co_u32" => (a.wrapping_sub(b).wrapping_sub(c), b + c > a),
                    _ => (b.wrapping_sub(a).wrapping_sub(c), a + c > b),
                };
                let r = if v.clamp && carry { if name.contains("sub") { 0 } else { u32::MAX as u64 } } else { r };
                self.vwrite(w, &v, l, r as u32);
                if carry {
                    cout |= 1 << l;
                }
            }
            self.waves[w].set_s64(v.sdst as usize, cout);
            return Ok(());
        }
        let exec = self.waves[w].exec();
        let mut results: Vec<(usize, u32)> = Vec::with_capacity(lanes);
        for l in 0..lanes {
            if !self.lane_writes(w, &v, l) {
                continue;
            }
            let us = |k: usize| self.vsrc(w, ins, &v, k, l);
            let fs = |k: usize| self.fsrc(w, ins, &v, k, l);
            let r: u32 = match name {
                // Source modifiers act on the sign bit (as translated).
                "v_mov_b32" => self.bsrc(w, ins, &v, 0, l)?,
                "v_cndmask_b32" => {
                    let m = if v.vop3 { self.waves[w].s64(v.src[2] as usize) } else { self.waves[w].s64(VCC) };
                    let a = self.bsrc(w, ins, &v, 0, l)?;
                    let b = self.bsrc(w, ins, &v, 1, l)?;
                    if m >> l & 1 != 0 { b } else { a }
                }
                "v_add_f32" => Self::fres(&v, fs(0)? + fs(1)?),
                "v_sub_f32" => Self::fres(&v, fs(0)? - fs(1)?),
                "v_subrev_f32" => Self::fres(&v, fs(1)? - fs(0)?),
                "v_mul_f32" => Self::fres(&v, fs(0)? * fs(1)?),
                "v_mul_legacy_f32" => {
                    let (a, b) = (fs(0)?, fs(1)?);
                    Self::fres(&v, if a == 0.0 || b == 0.0 { 0.0 } else { a * b })
                }
                "v_mac_f32" | "v_fmac_f32" => {
                    let c = f(self.waves[w].v[(v.dst & 255) as usize][l]);
                    let (a, b) = (fs(0)?, fs(1)?);
                    Self::fres(&v, if name == "v_fmac_f32" { a.mul_add(b, c) } else { a * b + c })
                }
                "v_mad_f32" | "v_fma_f32" => {
                    let (a, b, c) = (fs(0)?, fs(1)?, fs(2)?);
                    Self::fres(&v, if name == "v_fma_f32" { a.mul_add(b, c) } else { a * b + c })
                }
                "v_madmk_f32" | "v_fmamk_f32" => {
                    let k = f(self.lit(ins));
                    let (a, b) = (fs(0)?, fs(1)?);
                    Self::fres(&v, if name == "v_fmamk_f32" { a.mul_add(k, b) } else { a * k + b })
                }
                "v_madak_f32" | "v_fmaak_f32" => {
                    let k = f(self.lit(ins));
                    let (a, b) = (fs(0)?, fs(1)?);
                    Self::fres(&v, if name == "v_fmaak_f32" { a.mul_add(b, k) } else { a * b + k })
                }
                "v_min_f32" => Self::fres(&v, fmin(fs(0)?, fs(1)?)),
                "v_max_f32" => Self::fres(&v, fmax(fs(0)?, fs(1)?)),
                "v_min3_f32" => Self::fres(&v, fmin(fmin(fs(0)?, fs(1)?), fs(2)?)),
                "v_max3_f32" => Self::fres(&v, fmax(fmax(fs(0)?, fs(1)?), fs(2)?)),
                "v_med3_f32" => {
                    let (a, b, c) = (fs(0)?, fs(1)?, fs(2)?);
                    Self::fres(&v, if a.is_nan() || b.is_nan() || c.is_nan() { fmin(fmin(a, b), c) } else { fmax(fmin(a, b), fmin(fmax(a, b), c)) })
                }
                "v_mul_i32_i24" => (sext(us(0)?, 24) as i32).wrapping_mul(sext(us(1)?, 24) as i32) as u32,
                "v_mul_hi_i32_i24" => ((sext(us(0)?, 24) as i32 as i64 * sext(us(1)?, 24) as i32 as i64) >> 32) as u32,
                "v_mul_u32_u24" => (us(0)? & 0xFF_FFFF).wrapping_mul(us(1)? & 0xFF_FFFF),
                "v_mul_hi_u32_u24" => (((us(0)? & 0xFF_FFFF) as u64 * (us(1)? & 0xFF_FFFF) as u64) >> 32) as u32,
                "v_mad_i32_i24" => (sext(us(0)?, 24) as i32).wrapping_mul(sext(us(1)?, 24) as i32).wrapping_add(us(2)? as i32) as u32,
                "v_mad_u32_u24" => (us(0)? & 0xFF_FFFF).wrapping_mul(us(1)? & 0xFF_FFFF).wrapping_add(us(2)?),
                "v_mul_lo_u32" | "v_mul_lo_i32" => us(0)?.wrapping_mul(us(1)?),
                "v_mul_hi_u32" => ((us(0)? as u64 * us(1)? as u64) >> 32) as u32,
                "v_mul_hi_i32" => ((us(0)? as i32 as i64 * us(1)? as i32 as i64) >> 32) as u32,
                "v_min_i32" => (us(0)? as i32).min(us(1)? as i32) as u32,
                "v_max_i32" => (us(0)? as i32).max(us(1)? as i32) as u32,
                "v_min_u32" => us(0)?.min(us(1)?),
                "v_max_u32" => us(0)?.max(us(1)?),
                "v_min3_i32" => (us(0)? as i32).min(us(1)? as i32).min(us(2)? as i32) as u32,
                "v_max3_i32" => (us(0)? as i32).max(us(1)? as i32).max(us(2)? as i32) as u32,
                "v_min3_u32" => us(0)?.min(us(1)?).min(us(2)?),
                "v_max3_u32" => us(0)?.max(us(1)?).max(us(2)?),
                "v_med3_i32" => {
                    let (a, b, c) = (us(0)? as i32, us(1)? as i32, us(2)? as i32);
                    a.min(b).max(a.max(b).min(c)) as u32
                }
                "v_med3_u32" => {
                    let (a, b, c) = (us(0)?, us(1)?, us(2)?);
                    a.min(b).max(a.max(b).min(c))
                }
                "v_lshrrev_b32" => us(1)? >> (us(0)? & 31),
                "v_ashrrev_i32" => ((us(1)? as i32) >> (us(0)? & 31)) as u32,
                "v_lshlrev_b32" => us(1)? << (us(0)? & 31),
                "v_and_b32" => us(0)? & us(1)?,
                "v_or_b32" => us(0)? | us(1)?,
                "v_xor_b32" => us(0)? ^ us(1)?,
                "v_xnor_b32" => !(us(0)? ^ us(1)?),
                "v_not_b32" => !us(0)?,
                "v_bfrev_b32" => us(0)?.reverse_bits(),
                "v_ffbh_u32" => {
                    let x = us(0)?;
                    if x == 0 { u32::MAX } else { x.leading_zeros() }
                }
                "v_ffbl_b32" => {
                    let x = us(0)?;
                    if x == 0 { u32::MAX } else { x.trailing_zeros() }
                }
                "v_ffbh_i32" => {
                    let x = us(0)? as i32;
                    if x == 0 || x == -1 { u32::MAX } else { (if x < 0 { !x } else { x } as u32).leading_zeros() }
                }
                "v_add_u32" | "v_add_nc_u32" => {
                    let (r, c) = us(0)?.overflowing_add(us(1)?);
                    if v.clamp && c { u32::MAX } else { r }
                }
                "v_sub_u32" | "v_sub_nc_u32" => {
                    let (a, b) = (us(0)?, us(1)?);
                    if v.clamp && b > a { 0 } else { a.wrapping_sub(b) }
                }
                "v_subrev_u32" | "v_subrev_nc_u32" => {
                    let (a, b) = (us(0)?, us(1)?);
                    if v.clamp && a > b { 0 } else { b.wrapping_sub(a) }
                }
                "v_add_i32" | "v_add_nc_i32" => {
                    let (a, b) = (us(0)? as i32, us(1)? as i32);
                    if v.clamp { a.saturating_add(b) as u32 } else { a.wrapping_add(b) as u32 }
                }
                "v_sub_i32" | "v_sub_nc_i32" => {
                    let (a, b) = (us(0)? as i32, us(1)? as i32);
                    if v.clamp { a.saturating_sub(b) as u32 } else { a.wrapping_sub(b) as u32 }
                }
                "v_add3_u32" => us(0)?.wrapping_add(us(1)?).wrapping_add(us(2)?),
                "v_xad_u32" => (us(0)? ^ us(1)?).wrapping_add(us(2)?),
                "v_lshl_add_u32" => (us(0)? << (us(1)? & 31)).wrapping_add(us(2)?),
                "v_add_lshl_u32" => us(0)?.wrapping_add(us(1)?) << (us(2)? & 31),
                "v_lshl_or_b32" => (us(0)? << (us(1)? & 31)) | us(2)?,
                "v_and_or_b32" => (us(0)? & us(1)?) | us(2)?,
                "v_or3_b32" => us(0)? | us(1)? | us(2)?,
                "v_xor3_b32" => us(0)? ^ us(1)? ^ us(2)?,
                "v_bfe_u32" => bfe(us(0)?, us(1)?, us(2)?, false),
                "v_bfe_i32" => bfe(us(0)?, us(1)?, us(2)?, true),
                "v_bfi_b32" => {
                    let m = us(0)?;
                    (m & us(1)?) | (!m & us(2)?)
                }
                "v_bfm_b32" => (1u32.checked_shl(us(0)? & 31).unwrap_or(0).wrapping_sub(1)) << (us(1)? & 31),
                "v_bcnt_u32_b32" => us(0)?.count_ones().wrapping_add(us(1)?),
                "v_mbcnt_lo_u32_b32" => {
                    let below = if l >= 32 { u32::MAX } else { (1u32 << l).wrapping_sub(1) };
                    (us(0)? & below).count_ones().wrapping_add(us(1)?)
                }
                "v_mbcnt_hi_u32_b32" => {
                    let below = if l < 32 { 0 } else { (1u32 << (l - 32)).wrapping_sub(1) };
                    (us(0)? & below).count_ones().wrapping_add(us(1)?)
                }
                "v_alignbit_b32" => ((((us(0)? as u64) << 32) | us(1)? as u64) >> (us(2)? & 31)) as u32,
                "v_alignbyte_b32" => ((((us(0)? as u64) << 32) | us(1)? as u64) >> (8 * (us(2)? & 3))) as u32,
                "v_perm_b32" => {
                    let (a, b, sel) = (us(0)?, us(1)?, us(2)?);
                    let bytes = ((a as u64) << 32) | b as u64;
                    let mut r = 0u32;
                    for k in 0..4 {
                        let s = (sel >> (8 * k)) & 0xFF;
                        let byte = match s {
                            0..=7 => ((bytes >> (8 * s)) & 0xFF) as u32,
                            8..=11 => {
                                let src = (bytes >> (8 * ((s - 8) * 2 + 1))) & 0x80;
                                if src != 0 { 0xFF } else { 0 }
                            }
                            12 => 0,
                            _ => 0xFF,
                        };
                        r |= byte << (8 * k);
                    }
                    r
                }
                "v_cvt_f32_i32" => Self::fres(&v, us(0)? as i32 as f32),
                "v_cvt_f32_u32" => Self::fres(&v, us(0)? as f32),
                "v_cvt_u32_f32" => {
                    let x = fs(0)?;
                    if x.is_nan() { 0 } else { x as u32 }
                }
                "v_cvt_i32_f32" => {
                    let x = fs(0)?;
                    if x.is_nan() { 0 } else { x as i32 as u32 }
                }
                "v_cvt_rpi_i32_f32" => (fs(0)? + 0.5).floor() as i32 as u32,
                "v_cvt_flr_i32_f32" => fs(0)?.floor() as i32 as u32,
                "v_cvt_f32_ubyte0" => u((us(0)? & 0xFF) as f32),
                "v_cvt_f32_ubyte1" => u(((us(0)? >> 8) & 0xFF) as f32),
                "v_cvt_f32_ubyte2" => u(((us(0)? >> 16) & 0xFF) as f32),
                "v_cvt_f32_ubyte3" => u((us(0)? >> 24) as f32),
                "v_cvt_f16_f32" => f32_to_half(fs(0)?) as u32,
                "v_cvt_f32_f16" => Self::fres(&v, half_to_f32(us(0)? as u16)),
                "v_cvt_pkrtz_f16_f32" => {
                    // Round toward zero.
                    let h = |x: f32| -> u32 {
                        let r = f32_to_half(x);
                        if (half_to_f32(r) as f64).abs() > (x as f64).abs() && r & 0x3FF != 0 { (r - 1) as u32 } else { r as u32 }
                    };
                    h(fs(0)?) | (h(fs(1)?) << 16)
                }
                "v_cvt_pk_i16_i32" => {
                    let s = |x: u32| (x as i32).clamp(-32768, 32767) as u32 & 0xFFFF;
                    s(us(0)?) | (s(us(1)?) << 16)
                }
                "v_cvt_pk_u16_u32" => us(0)?.min(0xFFFF) | (us(1)?.min(0xFFFF) << 16),
                "v_cvt_pk_u8_f32" => {
                    let x = fs(0)?;
                    let byte = (us(1)? & 3) * 8;
                    let q = (x.round_ties_even().clamp(0.0, 255.0)) as u32;
                    (us(2)? & !(0xFF << byte)) | (q << byte)
                }
                "v_fract_f32" => {
                    let x = fs(0)?;
                    Self::fres(&v, (x - x.floor()).min(f32::from_bits(0x3F7F_FFFF)))
                }
                "v_trunc_f32" => Self::fres(&v, fs(0)?.trunc()),
                "v_ceil_f32" => Self::fres(&v, fs(0)?.ceil()),
                "v_rndne_f32" => Self::fres(&v, fs(0)?.round_ties_even()),
                "v_floor_f32" => Self::fres(&v, fs(0)?.floor()),
                "v_exp_f32" => Self::fres(&v, fs(0)?.exp2()),
                "v_log_f32" => Self::fres(&v, fs(0)?.log2()),
                "v_rcp_f32" | "v_rcp_iflag_f32" => Self::fres(&v, 1.0 / fs(0)?),
                "v_rsq_f32" => Self::fres(&v, 1.0 / fs(0)?.sqrt()),
                "v_sqrt_f32" => Self::fres(&v, fs(0)?.sqrt()),
                "v_sin_f32" => Self::fres(&v, (fs(0)? * std::f32::consts::TAU).sin()),
                "v_cos_f32" => Self::fres(&v, (fs(0)? * std::f32::consts::TAU).cos()),
                "v_ldexp_f32" => Self::fres(&v, fs(0)? * 2f32.powi(us(1)? as i32)),
                "v_frexp_exp_i32_f32" => {
                    let x = fs(0)?;
                    if x == 0.0 || !x.is_finite() { 0 } else { ((x.to_bits() >> 23 & 0xFF) as i32 - 126) as u32 }
                }
                "v_frexp_mant_f32" => {
                    let x = fs(0)?;
                    if x == 0.0 || !x.is_finite() { u(x) } else { (x.to_bits() & 0x807F_FFFF) | (126 << 23) }
                }
                "v_lerp_u8" => {
                    let (a, b, c) = (us(0)?, us(1)?, us(2)?);
                    let mut r = 0;
                    for k in 0..4 {
                        let x = (a >> (8 * k)) & 0xFF;
                        let y = (b >> (8 * k)) & 0xFF;
                        let rr = (c >> (8 * k)) & 1;
                        r |= ((x + y + rr) >> 1) << (8 * k);
                    }
                    r
                }
                "v_sad_u32" => us(0)?.abs_diff(us(1)?).wrapping_add(us(2)?),
                "v_movrels_b32" => {
                    let m0 = self.waves[w].s[M0];
                    let src = ((v.src[0] - 256) as u32 + m0) as usize & 255;
                    self.waves[w].v[src][l]
                }
                "v_movreld_b32" => {
                    let m0 = self.waves[w].s[M0] as usize;
                    let x = us(0)?;
                    let d = ((v.dst & 255) as usize + m0) & 255;
                    results.push((l | (d << 8) | (1 << 20), x));
                    continue;
                }
                "v_cubeid_f32" | "v_cubesc_f32" | "v_cubetc_f32" | "v_cubema_f32" => {
                    let (x, y, z) = (fs(0)?, fs(1)?, fs(2)?);
                    u(cube(name, x, y, z))
                }
                // 16-bit integer arithmetic (VOP3 on gfx10): the result goes to
                // the low half (the high half with op_sel[3]); the other half
                // is kept.
                "v_add_u16" | "v_sub_u16" | "v_add_i16" | "v_sub_i16" | "v_mul_lo_u16" | "v_max_u16" | "v_max_i16" | "v_min_u16" | "v_min_i16" | "v_lshlrev_b16" | "v_lshrrev_b16" | "v_ashrrev_i16" | "v_mad_u16" | "v_mad_i16" => {
                    let signed = name.ends_with("i16");
                    let s16 = |k: usize| -> R<u32> {
                        let x = self.src16(w, ins, &v, k, l)?;
                        Ok(if signed { sext(x, 16) } else { x })
                    };
                    let (a, b) = (s16(0)?, s16(1)?);
                    let (ai, bi) = (a as i32, b as i32);
                    let r = match name {
                        "v_add_u16" => {
                            let r = a + b;
                            if v.clamp { r.min(0xFFFF) } else { r }
                        }
                        "v_sub_u16" => {
                            if v.clamp && b > a { 0 } else { a.wrapping_sub(b) }
                        }
                        "v_add_i16" | "v_sub_i16" => {
                            let r = if name == "v_add_i16" { ai + bi } else { ai - bi };
                            (if v.clamp { r.clamp(-32768, 32767) } else { r }) as u32
                        }
                        "v_mul_lo_u16" => a.wrapping_mul(b),
                        "v_max_u16" => a.max(b),
                        "v_min_u16" => a.min(b),
                        "v_max_i16" => ai.max(bi) as u32,
                        "v_min_i16" => ai.min(bi) as u32,
                        "v_lshlrev_b16" => b << (a & 15),
                        "v_lshrrev_b16" => b >> (a & 15),
                        "v_ashrrev_i16" => (sext(b, 16) as i32 >> (a & 15)) as u32,
                        _ => {
                            let c = s16(2)?;
                            let r = a.wrapping_mul(b).wrapping_add(c);
                            match (v.clamp, signed) {
                                (false, _) => r,
                                (true, true) => (r as i32).clamp(-32768, 32767) as u32,
                                (true, false) => r.min(0xFFFF),
                            }
                        }
                    } & 0xFFFF;
                    let old = self.waves[w].v[(v.dst & 255) as usize][l];
                    if v.sdwa.is_some() {
                        r
                    } else if v.opsel & 8 != 0 {
                        (old & 0xFFFF) | (r << 16)
                    } else {
                        (old & 0xFFFF_0000) | r
                    }
                }
                _ => return Err(format!("{:#x}: {name} not emulated", ins.pc)),
            };
            results.push((l, r));
        }
        let _ = exec;
        for (l, r) in results {
            if l & (1 << 20) != 0 {
                let d = (l >> 8) & 255;
                self.waves[w].v[d][l & 255] = r;
            } else {
                self.vwrite(w, &v, l, r);
            }
        }
        Ok(())
    }

    /// Source `k` with abs/neg as sign-bit operations (moves, selects).
    fn bsrc(&self, w: usize, ins: &Inst, v: &Vop, k: usize, l: usize) -> R<u32> {
        let mut x = self.vsrc(w, ins, v, k, l)?;
        if v.abs >> k & 1 != 0 {
            x &= 0x7FFF_FFFF;
        }
        if v.neg >> k & 1 != 0 {
            x ^= 0x8000_0000;
        }
        Ok(x)
    }

    // -----------------------------------------------------------------------
    // LDS

    fn lds_rd(&self, a: u32, n: usize) -> u64 {
        let mut x = 0u64;
        for k in 0..n {
            let b = self.lds.get(a as usize + k).copied().unwrap_or(0);
            x |= (b as u64) << (8 * k);
        }
        x
    }

    fn lds_wr(&mut self, a: u32, n: usize, x: u64) {
        for k in 0..n {
            if let Some(b) = self.lds.get_mut(a as usize + k) {
                *b = (x >> (8 * k)) as u8;
            }
        }
    }

    fn ds(&mut self, w: usize, ins: &Inst, name: &str) -> R<()> {
        if ins.ds_gds() {
            return Err(format!("{:#x}: GDS {name} not emulated", ins.pc));
        }
        let lanes = self.lanes;
        let exec = self.waves[w].exec();
        let off16 = ins.ds_offset0() | (ins.ds_offset1() << 8);
        let (o0, o1) = (ins.ds_offset0(), ins.ds_offset1());
        let addr = ins.ds_addr() as usize;
        let (d0, d1, vd) = (ins.ds_data0() as usize, ins.ds_data1() as usize, ins.vdst() as usize);
        let active: Vec<usize> = (0..lanes).filter(|&l| exec >> l & 1 != 0).collect();
        match name {
            "ds_swizzle_b32" => {
                let src: Vec<u32> = (0..64).map(|l| self.waves[w].v[d0][l]).collect();
                for &l in &active {
                    let s = if off16 & 0x8000 != 0 {
                        (l & !3) | ((off16 >> (2 * (l & 3))) & 3) as usize
                    } else {
                        let and = (off16 & 31) as usize;
                        let or = ((off16 >> 5) & 31) as usize;
                        let xor = ((off16 >> 10) & 31) as usize;
                        (l & !31) | ((((l & 31) & and) | or) ^ xor)
                    };
                    self.waves[w].v[vd][l] = if exec >> s & 1 != 0 { src[s] } else { 0 };
                }
                return Ok(());
            }
            "ds_bpermute_b32" | "ds_permute_b32" => {
                let src: Vec<u32> = (0..64).map(|l| self.waves[w].v[d0][l]).collect();
                let mut out = self.waves[w].v[vd];
                if name == "ds_permute_b32" {
                    for &l in &active {
                        let t = ((self.waves[w].v[addr][l].wrapping_add(off16)) / 4) as usize & (lanes - 1);
                        out[t] = src[l];
                    }
                } else {
                    for &l in &active {
                        let s = ((self.waves[w].v[addr][l].wrapping_add(off16)) / 4) as usize & (lanes - 1);
                        out[l] = if exec >> s & 1 != 0 { src[s] } else { 0 };
                    }
                }
                self.waves[w].v[vd] = out;
                return Ok(());
            }
            _ => {}
        }
        let nb = |n: &str| -> usize {
            if n.ends_with("b128") {
                16
            } else if n.ends_with("b96") {
                12
            } else if n.ends_with("b64") {
                8
            } else {
                4
            }
        };
        for &l in &active {
            let a = if name.ends_with("_addtid_b32") { self.waves[w].s[M0].wrapping_add(l as u32 * 4) } else { self.waves[w].v[addr][l] };
            match name {
                "ds_write_b32" | "ds_write_b64" | "ds_write_b96" | "ds_write_b128" | "ds_write_addtid_b32" => {
                    let n = nb(name) / 4;
                    for k in 0..n {
                        let x = self.waves[w].v[d0 + k][l];
                        self.lds_wr(a.wrapping_add(off16) + 4 * k as u32, 4, x as u64);
                    }
                }
                "ds_write2_b32" | "ds_write2st64_b32" | "ds_write2_b64" | "ds_write2st64_b64" => {
                    let esz = if name.ends_with("b64") { 8 } else { 4 };
                    let st = if name.contains("st64") { 64 } else { 1 };
                    for (o, src) in [(o0, d0), (o1, d1)] {
                        let base = a.wrapping_add(o * esz * st);
                        for k in 0..(esz / 4) as usize {
                            let x = self.waves[w].v[src + k][l];
                            self.lds_wr(base + 4 * k as u32, 4, x as u64);
                        }
                    }
                }
                "ds_write_b8" | "ds_write_b16" | "ds_write_b8_d16_hi" | "ds_write_b16_d16_hi" => {
                    let n = if name.contains("b8") { 1 } else { 2 };
                    let mut x = self.waves[w].v[d0][l];
                    if name.ends_with("_hi") {
                        x >>= 16;
                    }
                    self.lds_wr(a.wrapping_add(off16), n, x as u64);
                }
                "ds_read_b32" | "ds_read_b64" | "ds_read_b96" | "ds_read_b128" | "ds_read_addtid_b32" => {
                    let n = nb(name) / 4;
                    for k in 0..n {
                        let x = self.lds_rd(a.wrapping_add(off16) + 4 * k as u32, 4) as u32;
                        self.waves[w].v[vd + k][l] = x;
                    }
                }
                "ds_read2_b32" | "ds_read2st64_b32" | "ds_read2_b64" | "ds_read2st64_b64" => {
                    let esz = if name.ends_with("b64") { 8 } else { 4 };
                    let st = if name.contains("st64") { 64 } else { 1 };
                    let mut dst = vd;
                    for o in [o0, o1] {
                        let base = a.wrapping_add(o * esz * st);
                        for k in 0..(esz / 4) as usize {
                            let x = self.lds_rd(base + 4 * k as u32, 4) as u32;
                            self.waves[w].v[dst][l] = x;
                            dst += 1;
                        }
                    }
                }
                "ds_read_u8" | "ds_read_i8" | "ds_read_u16" | "ds_read_i16" => {
                    let n = if name.ends_with('8') { 1 } else { 2 };
                    let x = self.lds_rd(a.wrapping_add(off16), n) as u32;
                    self.waves[w].v[vd][l] = if name.contains("_i") { sext(x, 8 * n as u32) } else { x };
                }
                n if n.ends_with("_u32") || n.ends_with("_i32") || n.ends_with("_b32") => {
                    let rtn = n.contains("_rtn_");
                    let op = n.trim_start_matches("ds_").replace("_rtn", "");
                    let ea = a.wrapping_add(off16);
                    let old = self.lds_rd(ea, 4) as u32;
                    let x = self.waves[w].v[d0][l];
                    let y = self.waves[w].v[d1][l];
                    let new = match op.as_str() {
                        "add_u32" => old.wrapping_add(x),
                        "sub_u32" => old.wrapping_sub(x),
                        "rsub_u32" => x.wrapping_sub(old),
                        "inc_u32" => if old >= x { 0 } else { old + 1 },
                        "dec_u32" => if old == 0 || old > x { x } else { old - 1 },
                        "min_i32" => (old as i32).min(x as i32) as u32,
                        "max_i32" => (old as i32).max(x as i32) as u32,
                        "min_u32" => old.min(x),
                        "max_u32" => old.max(x),
                        "and_b32" => old & x,
                        "or_b32" => old | x,
                        "xor_b32" => old ^ x,
                        "mskor_b32" => (old & !x) | y,
                        "wrxchg_b32" => x,
                        "cmpst_b32" => if old == x { y } else { old },
                        _ => return Err(format!("{:#x}: {name} not emulated", ins.pc)),
                    };
                    self.lds_wr(ea, 4, new as u64);
                    if rtn {
                        self.waves[w].v[vd][l] = old;
                    }
                }
                _ => return Err(format!("{:#x}: {name} not emulated", ins.pc)),
            }
        }
        Ok(())
    }

    // -----------------------------------------------------------------------
    // Buffers

    fn rd(&mut self, a: u64, n: usize) -> Vec<u8> {
        let mut b = vec![0u8; n];
        if !self.env.read(a, &mut b) {
            b.iter_mut().for_each(|x| *x = 0);
        }
        b
    }

    fn buffer(&mut self, w: usize, ins: &Inst, name: &str) -> R<()> {
        let lanes = self.lanes;
        let exec = self.waves[w].exec();
        let r = ins.buf_srsrc() as usize;
        let vs = vsharp(&self.waves[w].s[r..r + 4]);
        let soff = self.sop(w, ins, ins.buf_soffset())?;
        let va = ins.buf_vaddr() as usize;
        let vd = ins.vdst() as usize;
        let (idxen, offen) = (ins.buf_idxen(), ins.buf_offen());
        let typed = ins.enc == Enc::Mtbuf || name.contains("_format_");
        let fmt = if ins.enc == Enc::Mtbuf { ins.mtbuf_format() as u16 } else { vs.format };
        let elem = |n: &str| -> (usize, bool) {
            // (bytes, signed) of untyped accesses
            if n.ends_with("dwordx4") {
                (16, false)
            } else if n.ends_with("dwordx3") {
                (12, false)
            } else if n.ends_with("dwordx2") {
                (8, false)
            } else if n.ends_with("dword") {
                (4, false)
            } else if n.ends_with("sbyte") {
                (1, true)
            } else if n.ends_with("ubyte") || n.ends_with("_byte") {
                (1, false)
            } else if n.ends_with("sshort") {
                (2, true)
            } else if n.ends_with("ushort") || n.ends_with("_short") {
                (2, false)
            } else {
                (4, false)
            }
        };
        let ncomp = if name.ends_with("_xyzw") {
            4
        } else if name.ends_with("_xyz") {
            3
        } else if name.ends_with("_xy") {
            2
        } else {
            1
        };
        let fm = format::gfx10(fmt);
        let esize = if typed { fm.map(|f| f.bytes()).unwrap_or(4) as u64 } else { elem(name).0 as u64 };
        for l in 0..lanes {
            if exec >> l & 1 == 0 {
                continue;
            }
            let mut index = if idxen { self.waves[w].v[va][l] } else { 0 };
            if vs.add_tid {
                index = index.wrapping_add(l as u32);
            }
            let voff = if offen { self.waves[w].v[va + idxen as usize][l] } else { 0 };
            let offset = ins.buf_offset() as u64 + soff as u64 + voff as u64;
            let oob = match vs.oob {
                3 => false,
                _ if idxen && vs.stride != 0 => index >= vs.num_records || (vs.oob == 0 && offset + esize > vs.stride as u64),
                _ => {
                    let size = if vs.stride == 0 { vs.num_records as u64 } else { vs.num_records as u64 * vs.stride as u64 };
                    index as u64 * vs.stride as u64 + offset + esize > size
                }
            };
            let a = vs.base + index as u64 * vs.stride as u64 + offset;
            if name.starts_with("buffer_load") || name.starts_with("tbuffer_load") {
                if typed {
                    let mut raw = [0u32; 4];
                    if !oob {
                        let b = self.rd(a, esize as usize);
                        for (k, c) in b.chunks(4).enumerate() {
                            let mut x = [0u8; 4];
                            x[..c.len()].copy_from_slice(c);
                            raw[k] = u32::from_le_bytes(x);
                        }
                    }
                    let (vals, _) = unpack(fmt, &raw).ok_or_else(|| format!("{:#x}: buffer format {fmt} not emulated", ins.pc))?;
                    let float = fm.is_some_and(|f| f.num.is_float());
                    let vals = if oob { [0; 4] } else if ins.enc == Enc::Mtbuf { vals } else { swizzle(&vals, vs.dst_sel, float) };
                    for k in 0..ncomp {
                        self.waves[w].v[vd + k][l] = vals[k];
                    }
                } else {
                    let (n, signed) = elem(name);
                    if n >= 4 {
                        let b = if oob { vec![0u8; n] } else { self.rd(a, n) };
                        for k in 0..n / 4 {
                            self.waves[w].v[vd + k][l] = u32::from_le_bytes(b[4 * k..4 * k + 4].try_into().unwrap());
                        }
                    } else {
                        let b = if oob { vec![0u8; n] } else { self.rd(a, n) };
                        let x = if n == 1 { b[0] as u32 } else { u16::from_le_bytes([b[0], b[1]]) as u32 };
                        self.waves[w].v[vd][l] = if signed { sext(x, 8 * n as u32) } else { x };
                    }
                }
            } else if name.starts_with("buffer_store") || name.starts_with("tbuffer_store") {
                if oob {
                    continue;
                }
                if typed {
                    let mut vals = [0u32; 4];
                    for (k, x) in vals.iter_mut().enumerate().take(ncomp) {
                        *x = self.sdata(w, vd + k, l);
                    }
                    let (raw, nbytes) = pack(fmt, &vals).ok_or_else(|| format!("{:#x}: buffer format {fmt} not emulated", ins.pc))?;
                    let bytes: Vec<u8> = raw.iter().flat_map(|x| x.to_le_bytes()).take(nbytes as usize).collect();
                    self.env.write(a, &bytes);
                } else {
                    let (n, _) = elem(name);
                    let mut bytes = Vec::new();
                    for k in 0..n.div_ceil(4) {
                        bytes.extend_from_slice(&self.sdata(w, vd + k, l).to_le_bytes());
                    }
                    bytes.truncate(n);
                    self.env.write(a, &bytes);
                }
            } else if let Some(op) = name.strip_prefix("buffer_atomic_") {
                if oob {
                    continue;
                }
                let glc = ins.glc();
                let x64 = op.ends_with("_x2");
                if x64 {
                    return Err(format!("{:#x}: {name} not emulated", ins.pc));
                }
                let old = u32::from_le_bytes(self.rd(a, 4)[..4].try_into().unwrap());
                let x = self.waves[w].v[vd][l];
                let new = match op {
                    "add" => old.wrapping_add(x),
                    "sub" => old.wrapping_sub(x),
                    "smin" => (old as i32).min(x as i32) as u32,
                    "umin" => old.min(x),
                    "smax" => (old as i32).max(x as i32) as u32,
                    "umax" => old.max(x),
                    "and" => old & x,
                    "or" => old | x,
                    "xor" => old ^ x,
                    "inc" => if old >= x { 0 } else { old + 1 },
                    "dec" => if old == 0 || old > x { x } else { old - 1 },
                    "swap" => x,
                    "cmpswap" => {
                        let cmp = self.waves[w].v[vd + 1][l];
                        if old == cmp { x } else { old }
                    }
                    _ => return Err(format!("{:#x}: {name} not emulated", ins.pc)),
                };
                self.env.write(a, &new.to_le_bytes());
                if glc {
                    self.waves[w].v[vd][l] = old;
                }
            } else {
                return Err(format!("{:#x}: {name} not emulated", ins.pc));
            }
        }
        Ok(())
    }

    fn global(&mut self, w: usize, ins: &Inst, name: &str) -> R<()> {
        let lanes = self.lanes;
        let exec = self.waves[w].exec();
        let saddr = ins.flat_saddr();
        let va = ins.flat_addr() as usize;
        let vd = ins.vdst() as usize;
        let data = ins.flat_data() as usize;
        let off = ins.flat_offset() as i64;
        let n = if name.ends_with("dwordx4") {
            16
        } else if name.ends_with("dwordx3") {
            12
        } else if name.ends_with("dwordx2") {
            8
        } else if name.ends_with("dword") {
            4
        } else {
            return Err(format!("{:#x}: {name} not emulated", ins.pc));
        };
        for l in 0..lanes {
            if exec >> l & 1 == 0 {
                continue;
            }
            let base = if saddr != 0x7D && saddr != 0x7F && saddr < 106 { self.waves[w].s64(saddr as usize).wrapping_add(self.waves[w].v[va][l] as u64) } else { self.waves[w].v[va][l] as u64 | (self.waves[w].v[va + 1][l] as u64) << 32 };
            let a = (base as i64 + off) as u64;
            if name.starts_with("global_load") {
                let b = self.rd(a, n);
                for k in 0..n / 4 {
                    self.waves[w].v[vd + k][l] = u32::from_le_bytes(b[4 * k..4 * k + 4].try_into().unwrap());
                }
            } else if name.starts_with("global_store") {
                let mut bytes = Vec::new();
                for k in 0..n / 4 {
                    bytes.extend_from_slice(&self.waves[w].v[data + k][l].to_le_bytes());
                }
                self.env.write(a, &bytes);
            } else {
                return Err(format!("{:#x}: {name} not emulated", ins.pc));
            }
        }
        Ok(())
    }

    // -----------------------------------------------------------------------
    // Images

    fn image(&mut self, w: usize, ins: &Inst, name: &str) -> R<()> {
        let lanes = self.lanes;
        let exec = self.waves[w].exec();
        let r = ins.mimg_srsrc() as usize;
        let mut tw = [0u32; 8];
        tw.copy_from_slice(&self.waves[w].s[r..r + 8]);
        let t = tsharp(&tw);
        let sr = ins.mimg_ssamp() as usize;
        let samp = [self.waves[w].s[sr], self.waves[w].s[sr + 1], self.waves[w].s[sr + 2], self.waves[w].s[sr + 3]];
        let dmask = ins.mimg_dmask();
        let vd = ins.vdst() as usize;
        let dim = ins.mimg_dim();
        let fm = format::gfx10(t.format);
        let float = fm.is_some_and(|f| f.num.is_float());
        if ins.mimg_d16() || ins.mimg_a16() {
            return Err(format!("{:#x}: 16-bit image data/addresses not emulated", ins.pc));
        }
        // Address VGPRs (NSA or consecutive).
        let vaddr0 = bits(ins.w1, 0, 8) as usize;
        let addr_reg = |k: usize| -> usize {
            if k == 0 {
                vaddr0
            } else if ins.nsa.iter().any(|&x| x != 0) && k <= ins.nsa.len() {
                ins.nsa[k - 1] as usize
            } else {
                vaddr0 + k
            }
        };
        let coords = match dim {
            0 => 1,
            1 | 6 => 2,
            2 | 3 | 4 | 5 | 7 => {
                if dim == 4 {
                    2
                } else {
                    3
                }
            }
            _ => 2,
        };
        let ncomp = (dmask.count_ones() as usize).max(1);
        for l in 0..lanes {
            if exec >> l & 1 == 0 {
                continue;
            }
            let ar = |k: usize| self.waves[w].v[addr_reg(k)][l];
            match name {
                "image_load" | "image_load_mip" => {
                    let x = ar(0);
                    let y = if coords > 1 { ar(1) } else { 0 };
                    let z = if coords > 2 { ar(2) } else { 0 };
                    let mip = if name == "image_load_mip" { ar(coords) } else { 0 };
                    let raw = self.env.texel(&tw, x, y, z, t.base_level + mip);
                    let vals = match raw {
                        Some(r) => {
                            let (v, _) = unpack(t.format, &r).ok_or_else(|| format!("{:#x}: image format {} not emulated", ins.pc, t.format))?;
                            swizzle(&v, t.dst_sel, float)
                        }
                        None => [0; 4],
                    };
                    let mut k = 0;
                    for c in 0..4 {
                        if dmask >> c & 1 != 0 {
                            self.waves[w].v[vd + k][l] = vals[c];
                            k += 1;
                        }
                    }
                }
                "image_store" | "image_store_mip" => {
                    let x = ar(0);
                    let y = if coords > 1 { ar(1) } else { 0 };
                    let z = if coords > 2 { ar(2) } else { 0 };
                    let mip = if name == "image_store_mip" { ar(coords) } else { 0 };
                    let mut vals = [0u32; 4];
                    let mut k = 0;
                    for (c, v) in vals.iter_mut().enumerate() {
                        if dmask >> c & 1 != 0 {
                            *v = self.sdata(w, vd + k, l);
                            k += 1;
                        }
                    }
                    if x >= t.width || y >= t.height {
                        continue;
                    }
                    let (raw, _) = pack(t.format, &vals).ok_or_else(|| format!("{:#x}: image format {} not emulated", ins.pc, t.format))?;
                    self.env.store_texel(&tw, x, y, z, t.base_level + mip, raw);
                }
                n if n.starts_with("image_gather4") => {
                    let mods: Vec<&str> = n.trim_start_matches("image_gather4").split('_').filter(|s| !s.is_empty()).collect();
                    if mods.iter().any(|m| !matches!(*m, "lz" | "o")) || dim != 1 {
                        return Err(format!("{:#x}: {name} (dim {dim}) not emulated", ins.pc));
                    }
                    let mut k = 0;
                    let (ox, oy) = if mods.contains(&"o") {
                        let o = ar(0);
                        k = 1;
                        (sext(o & 0x3F, 6) as i32, sext((o >> 8) & 0x3F, 6) as i32)
                    } else {
                        (0, 0)
                    };
                    let (uu, vv) = (f(ar(k)), f(ar(k + 1)));
                    let unnorm = ins.mimg_unorm() || bits(samp[0], 15, 1) != 0;
                    let (fx, fy) = if unnorm { (uu - 0.5, vv - 0.5) } else { (uu * t.width as f32 - 0.5, vv * t.height as f32 - 0.5) };
                    // Fixed-point texel coordinates (8 fractional bits).
                    let q = |x: f32| ((x * 256.0).round() / 256.0).floor() as i32;
                    let (i0, j0) = (q(fx) + ox, q(fy) + oy);
                    let clamp_x = bits(samp[0], 0, 3);
                    let clamp_y = bits(samp[0], 3, 3);
                    let wrap = |i: i32, n: u32, mode: u32| -> Option<u32> {
                        match mode {
                            0 => Some(i.rem_euclid(n as i32) as u32),
                            2 => Some(i.clamp(0, n as i32 - 1) as u32),
                            _ => {
                                if i < 0 || i >= n as i32 {
                                    None
                                } else {
                                    Some(i as u32)
                                }
                            }
                        }
                    };
                    let comp = match dmask {
                        2 => 1,
                        4 => 2,
                        8 => 3,
                        _ => 0,
                    };
                    let mut out = [0u32; 4];
                    for (o, (di, dj)) in out.iter_mut().zip([(0, 1), (1, 1), (1, 0), (0, 0)]) {
                        let (xi, yi) = (wrap(i0 + di, t.width, clamp_x), wrap(j0 + dj, t.height, clamp_y));
                        let raw = match (xi, yi) {
                            (Some(x), Some(y)) => self.env.texel(&tw, x, y, 0, t.base_level),
                            _ => None,
                        };
                        *o = match raw {
                            Some(r) => {
                                let (v, _) = unpack(t.format, &r).ok_or_else(|| format!("{:#x}: image format {} not emulated", ins.pc, t.format))?;
                                swizzle(&v, t.dst_sel, float)[comp]
                            }
                            None => 0,
                        };
                    }
                    for (k, x) in out.iter().enumerate() {
                        self.waves[w].v[vd + k][l] = *x;
                    }
                }
                "image_sample_l" | "image_sample_lz" | "image_sample" => {
                    // 2D, 3D and 2D-array sampling at an explicit LOD (`_l`) or LOD 0 (`_lz`, and
                    // plain sampling: no derivatives in compute), as the sampler filters: point
                    // or bilinear texels, no, nearest or linear mips, its LOD clamp and wrap
                    // modes (the check of the spawn programs, 2026-11-14).
                    if dim != 1 && dim != 2 && dim != 5 {
                        return Err(format!("{:#x}: {name} (dim {dim}) not emulated", ins.pc));
                    }
                    if !float {
                        return Err(format!("{:#x}: {name} of an integer format not emulated", ins.pc));
                    }
                    let (uu, vv) = (f(ar(0)), f(ar(1)));
                    let slice = if dim == 5 { f(ar(2)).round().max(0.0) as u32 } else { 0 };
                    let lod_in = if name == "image_sample_l" { f(ar(if dim == 5 || dim == 2 { 3 } else { 2 })) } else { 0.0 };
                    let span = t.last_level.saturating_sub(t.base_level);
                    let (min_lod, max_lod) = (bits(samp[1], 0, 12) as f32 / 256.0, bits(samp[1], 12, 12) as f32 / 256.0);
                    let lod = lod_in.max(min_lod).min(max_lod).clamp(0.0, span as f32);
                    let xy_filter = bits(samp[2], if lod > 0.0 { 22 } else { 20 }, 2);
                    let mips: [(u32, f32); 2] = match bits(samp[2], 26, 2) {
                        0 => [(0, 1.0), (0, 0.0)],
                        2 => {
                            let m0 = lod.floor();
                            [(m0 as u32, 1.0 - (lod - m0)), ((m0 as u32 + 1).min(span), lod - m0)]
                        }
                        _ => [(((lod + 0.5).floor() as u32).min(span), 1.0), (0, 0.0)],
                    };
                    let unnorm = ins.mimg_unorm() || bits(samp[0], 15, 1) != 0;
                    let (clamp_x, clamp_y) = (bits(samp[0], 0, 3), bits(samp[0], 3, 3));
                    let wrap = |i: i32, n: u32, mode: u32| -> Option<u32> {
                        let n = n as i32;
                        match mode {
                            0 => Some(i.rem_euclid(n) as u32),
                            1 => {
                                let k = i.rem_euclid(2 * n);
                                Some(if k < n { k } else { 2 * n - 1 - k } as u32)
                            }
                            2 | 3 => Some(i.clamp(0, n - 1) as u32),
                            _ => (i >= 0 && i < n).then_some(i as u32),
                        }
                    };
                    let mut acc = [0f32; 4];
                    for &(m, weight) in &mips {
                        if weight == 0.0 {
                            continue;
                        }
                        let level = t.base_level + m;
                        let (wl, hl) = ((t.width >> level).max(1), (t.height >> level).max(1));
                        let (fx, fy) = if unnorm { (uu, vv) } else { (uu * wl as f32, vv * hl as f32) };
                        let taps: Vec<(i32, i32, f32)> = if xy_filter & 1 == 0 {
                            vec![(fx.floor() as i32, fy.floor() as i32, 1.0)]
                        } else {
                            let (gx, gy) = (fx - 0.5, fy - 0.5);
                            let (x0, y0) = (gx.floor(), gy.floor());
                            let (ax, ay) = (gx - x0, gy - y0);
                            let (x0, y0) = (x0 as i32, y0 as i32);
                            vec![(x0, y0, (1.0 - ax) * (1.0 - ay)), (x0 + 1, y0, ax * (1.0 - ay)), (x0, y0 + 1, (1.0 - ax) * ay), (x0 + 1, y0 + 1, ax * ay)]
                        };
                        let dl = (t.depth >> level).max(1);
                        let ztaps = if dim == 2 {
                            let fz = f(ar(2)) * if unnorm { 1.0 } else { dl as f32 };
                            if xy_filter & 1 == 0 {
                                vec![(fz.floor() as i32, 1.0)]
                            } else {
                                let gz = fz - 0.5;
                                let z = gz.floor();
                                vec![(z as i32, 1.0 - (gz - z)), (z as i32 + 1, gz - z)]
                            }
                        } else {
                            vec![(slice as i32, 1.0)]
                        };
                        for (i, j, tap) in taps {
                            for &(zi, zw) in &ztaps {
                                if dim == 2 && tap * zw == 0.0 { continue; }
                                let z = if dim == 2 { wrap(zi, dl, bits(samp[0], 6, 3)) } else { Some(slice) };
                                let raw = match (wrap(i, wl, clamp_x), wrap(j, hl, clamp_y), z) {
                                    (Some(x), Some(y), Some(z)) => self.env.texel(&tw, x, y, z, level),
                                    _ => None,
                                };
                                let vals = match raw {
                                    Some(r) => {
                                        let (v, _) = unpack(t.format, &r).ok_or_else(|| format!("{:#x}: image format {} not emulated", ins.pc, t.format))?;
                                        swizzle(&v, t.dst_sel, float)
                                    }
                                    None => [0; 4],
                                };
                                for (a, v) in acc.iter_mut().zip(vals) {
                                    *a += weight * tap * zw * f(v);
                                }
                            }
                        }
                    }
                    let mut k = 0;
                    for (c, a) in acc.iter().enumerate() {
                        if dmask >> c & 1 != 0 {
                            self.waves[w].v[vd + k][l] = u(*a);
                            k += 1;
                        }
                    }
                }
                "image_get_resinfo" => {
                    let mip = ar(0);
                    let vals = [(t.width >> mip).max(1), (t.height >> mip).max(1), if t.ty == 13 || t.ty == 12 { t.depth } else { t.depth }, t.last_level - t.base_level + 1];
                    let mut k = 0;
                    for c in 0..4 {
                        if dmask >> c & 1 != 0 {
                            self.waves[w].v[vd + k][l] = vals[c];
                            k += 1;
                        }
                    }
                }
                _ => return Err(format!("{:#x}: {name} not emulated", ins.pc)),
            }
        }
        let _ = (ncomp, t.base_array);
        Ok(())
    }
}

fn int_cmp(cmp: &str, a: u32, b: u32, signed: bool) -> Option<bool> {
    Some(if signed {
        let (a, b) = (a as i32, b as i32);
        match cmp {
            "f" => false,
            "lt" => a < b,
            "eq" => a == b,
            "le" => a <= b,
            "gt" => a > b,
            "lg" | "ne" => a != b,
            "ge" => a >= b,
            "t" | "tru" => true,
            _ => return None,
        }
    } else {
        match cmp {
            "f" => false,
            "lt" => a < b,
            "eq" => a == b,
            "le" => a <= b,
            "gt" => a > b,
            "lg" | "ne" => a != b,
            "ge" => a >= b,
            "t" | "tru" => true,
            _ => return None,
        }
    })
}

fn fcmp(c: &str, a: f32, b: f32) -> Option<bool> {
    let un = a.is_nan() || b.is_nan();
    Some(match c {
        "f" => false,
        "lt" => a < b,
        "eq" => a == b,
        "le" => a <= b,
        "gt" => a > b,
        "lg" => !un && a != b,
        "ge" => a >= b,
        "o" => !un,
        "u" => un,
        "nge" => !(a >= b),
        "nlg" => !(!un && a != b),
        "ngt" => !(a > b),
        "nle" => !(a <= b),
        "neq" => a != b,
        "nlt" => !(a < b),
        "tru" => true,
        _ => return None,
    })
}

fn class_f32(a: f32, mask: u32) -> bool {
    let b = a.to_bits();
    let neg = b >> 31 != 0;
    let cls = if a.is_nan() {
        if b & 0x40_0000 != 0 { 1 } else { 0 }
    } else if a.is_infinite() {
        if neg { 2 } else { 9 }
    } else if a == 0.0 {
        if neg { 5 } else { 6 }
    } else if a.is_subnormal() {
        if neg { 4 } else { 7 }
    } else if neg {
        3
    } else {
        8
    };
    mask >> cls & 1 != 0
}

fn fmin(a: f32, b: f32) -> f32 {
    if a.is_nan() {
        b
    } else if b.is_nan() {
        a
    } else {
        a.min(b)
    }
}

fn fmax(a: f32, b: f32) -> f32 {
    if a.is_nan() {
        b
    } else if b.is_nan() {
        a
    } else {
        a.max(b)
    }
}

fn cube(name: &str, x: f32, y: f32, z: f32) -> f32 {
    let (ax, ay, az) = (x.abs(), y.abs(), z.abs());
    if az >= ax && az >= ay {
        match name {
            "v_cubeid_f32" => if z < 0.0 { 5.0 } else { 4.0 },
            "v_cubesc_f32" => if z < 0.0 { -x } else { x },
            "v_cubetc_f32" => -y,
            _ => 2.0 * z,
        }
    } else if ay >= ax {
        match name {
            "v_cubeid_f32" => if y < 0.0 { 3.0 } else { 2.0 },
            "v_cubesc_f32" => x,
            "v_cubetc_f32" => if y < 0.0 { -z } else { z },
            _ => 2.0 * y,
        }
    } else {
        match name {
            "v_cubeid_f32" => if x < 0.0 { 1.0 } else { 0.0 },
            "v_cubesc_f32" => if x < 0.0 { z } else { -z },
            "v_cubetc_f32" => -y,
            _ => 2.0 * x,
        }
    }
}

#[cfg(test)]
mod volume_tests {
    use super::*;
    struct Volume {
        nan: bool,
    }
    impl Env for Volume {
        fn read(&mut self, _: u64, _: &mut [u8]) -> bool {
            false
        }
        fn write(&mut self, _: u64, _: &[u8]) {
            panic!("unexpected write")
        }
        fn texel(&mut self, _: &[u32; 8], _: u32, _: u32, z: u32, _: u32) -> Option<[u32; 4]> {
            Some([if self.nan { f32::NAN } else { 2.0 + z as f32 * 4.0 }.to_bits(), 0, 0, 0])
        }
        fn store_texel(&mut self, _: &[u32; 8], _: u32, _: u32, _: u32, _: u32, _: [u32; 4]) {
            panic!("unexpected image write")
        }
    }
    fn launch() -> Launch {
        Launch {
            wave64: true,
            local_size: [8, 8, 1],
            rsrc2: 0,
            user: vec![],
            group: [0; 3],
            lds_bytes: 1024,
            code_base: 0,
            probe: None,
        }
    }
    fn ins(enc: Enc, w0: u32, w1: u32) -> Inst {
        Inst {
            enc,
            op: 0,
            pc: 0,
            len: 2,
            w0,
            w1,
            lit: None,
            ext: Ext::None,
            nsa: [0; 12],
        }
    }
    #[test]
    fn volume_sampling_interpolates_depth_and_rejects_nan_history() {
        for nan in [false, true] {
            let mut e = Volume { nan };
            let mut g = Group::new(&[], &mut e, launch());
            // Synthetic 1x1x2 R32_FLOAT, identity R swizzle, trilinear clamp sampler.
            g.waves[0].s[..8].copy_from_slice(&[0, 22 << 20, 0, (10 << 28) | 0x924, 1, 0, 0, 0]);
            g.waves[0].s[12] = 2 | (2 << 3) | (2 << 6);
            g.waves[0].s[14] = 1 << 20;
            for v in &mut g.waves[0].v[..3] {
                *v = [0.5f32.to_bits(); 64];
            }
            g.image(0, &ins(Enc::Mimg, (2 << 3) | (1 << 8), (3 << 21) | (4 << 8)), "image_sample_lz").unwrap();
            let value = f32::from_bits(g.waves[0].v[4][0]);
            if nan {
                assert!(value.is_nan());
                assert!(!class_f32(value, 56));
            } else {
                assert_eq!(value, 4.0);
            }
        }
    }
    #[test]
    fn lane_addressed_spill_keeps_both_wave64_halves_and_exec_mask() {
        let mut e = Volume { nan: false };
        let mut g = Group::new(&[], &mut e, launch());
        g.waves[0].s[M0] = 128;
        for l in 0..64 {
            g.waves[0].v[3][l] = l as u32 + 100;
        }
        g.waves[0].set_exec(0xffff_ffff_ffff_fffe);
        g.ds(0, &ins(Enc::Ds, 0, 3 << 8), "ds_write_addtid_b32").unwrap();
        g.waves[0].set_exec(u64::MAX);
        g.ds(0, &ins(Enc::Ds, 0, 4 << 24), "ds_read_addtid_b32").unwrap();
        assert_eq!(g.waves[0].v[4][0], 0);
        for l in 1..64 {
            assert_eq!(g.waves[0].v[4][l], l as u32 + 100);
        }
    }
}
