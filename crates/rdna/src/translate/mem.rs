//! Memory, image, interpolation, export and stage-interface semantics.

use super::{e, ok, ImageDim, NumClass, ResSpec, Stage, R, T};
use crate::format::{self, Num};
use crate::{bits, Enc, Inst};
use rspirv::dr::Operand;
use rspirv::spirv::{self, BuiltIn, Decoration, GlslStd450Op as GLOp, ImageOperands, StorageClass, Word};

// ---------------------------------------------------------------------------
// Stage inputs

/// PS input VGPR counts per SPI_PS_INPUT_ADDR bit.
const PS_INPUT_VGPRS: [u32; 16] = [2, 2, 2, 3, 2, 2, 2, 1, 1, 1, 1, 1, 1, 1, 1, 1];

pub fn stage_inputs(tr: &mut T) -> R<()> {
    match tr.spec.stage {
        Stage::Vertex => {
            // NGG (merged ES/GS) wave: s3 = merged wave info (ES threads in
            // [7:0], GS threads in [15:8]); v0 = primitive export data,
            // v5 = vertex id, v8 = instance id.
            let k = tr.u(0x4040);
            tr.set_sreg(3, k)?;
            let vid = tr.input(tr.t.i32, Some(BuiltIn::VertexIndex), None);
            let iid = tr.input(tr.t.i32, Some(BuiltIn::InstanceIndex), None);
            let v = e(tr.b.load(tr.t.i32, None, vid, None, []))?;
            let v = tr.to_u(v)?;
            tr.set_vreg_raw(5, v)?;
            let i = e(tr.b.load(tr.t.i32, None, iid, None, []))?;
            let mut i = tr.to_u(i)?;
            if let Some(k) = tr.spec.base_instance_sgpr {
                // A draw whose arguments the GPU reads: the start instance
                // reaches the shader as the command processor hands it over
                // (in a user SGPR), the instance id counts from 0
                // (DrawParameters is core in SPIR-V 1.3).
                tr.b.capability(rspirv::spirv::Capability::DrawParameters);
                let bid = tr.input(tr.t.i32, Some(BuiltIn::BaseInstance), None);
                let b = e(tr.b.load(tr.t.i32, None, bid, None, []))?;
                let b = tr.to_u(b)?;
                tr.set_sreg(k as u32, b)?;
                i = e(tr.b.i_sub(tr.t.u32, None, i, b))?;
            }
            tr.set_vreg_raw(8, i)?;
            tr.vertex_ids = Some((v, i));
        }
        Stage::Fragment => {
            let fc = tr.input(tr.t.v4f, Some(BuiltIn::FragCoord), None);
            tr.frag_coord = Some(fc);
            let ff = tr.input(tr.t.bool_, Some(BuiltIn::FrontFacing), None);
            let fcv = e(tr.b.load(tr.t.v4f, None, fc, None, []))?;
            // Barycentrics (perspective-correct and linear): vertex 0's
            // weight is 1 - i - j, as in the hardware's (i, j) VGPRs.
            let bary = if tr.spec.barycentric {
                tr.b.capability(rspirv::spirv::Capability::FragmentBarycentricKHR);
                tr.b.extension("SPV_KHR_fragment_shader_barycentric");
                let p = tr.input(tr.t.v3f, Some(BuiltIn::BaryCoordKHR), None);
                let l = tr.input(tr.t.v3f, Some(BuiltIn::BaryCoordNoPerspKHR), None);
                let pv = e(tr.b.load(tr.t.v3f, None, p, None, []))?;
                let lv = e(tr.b.load(tr.t.v3f, None, l, None, []))?;
                tr.bary = Some((pv, lv));
                Some((pv, lv))
            } else if let Some(io) = tr.bary_io.clone() {
                // From the geometry shader (`gs::bary_gs`): (1, 0, 0),
                // (0, 1, 0), (0, 0, 1) at the vertices, interpolated
                // perspective-correct and linear.
                let p = tr.input(tr.t.v4f, None, Some(io.bary[0]));
                let l = tr.input(tr.t.v4f, None, Some(io.bary[1]));
                tr.b.decorate(l, Decoration::NoPerspective, []);
                let p4 = e(tr.b.load(tr.t.v4f, None, p, None, []))?;
                let l4 = e(tr.b.load(tr.t.v4f, None, l, None, []))?;
                let pv = e(tr.b.vector_shuffle(tr.t.v3f, None, p4, p4, [0, 1, 2]))?;
                let lv = e(tr.b.vector_shuffle(tr.t.v3f, None, l4, l4, [0, 1, 2]))?;
                tr.bary = Some((pv, lv));
                Some((pv, lv))
            } else {
                None
            };
            let mut vg = 0u32;
            let addr = if tr.spec.ps_input_addr == 0 { 0x2 } else { tr.spec.ps_input_addr };
            for bit in 0..16 {
                if addr & (1 << bit) == 0 {
                    continue;
                }
                let n = PS_INPUT_VGPRS[bit];
                for k in 0..n {
                    let val = match bit {
                        // PERSP_SAMPLE/CENTER/CENTROID and LINEAR_* (i, j):
                        // single-sampled rendering, all at the center.
                        0..=2 | 4..=6 if bary.is_some() => {
                            let (pv, lv) = bary.unwrap();
                            let src = if bit < 3 { pv } else { lv };
                            let c = e(tr.b.composite_extract(tr.t.f32, None, src, [k + 1]))?;
                            tr.to_u(c)?
                        }
                        8..=10 => {
                            let c = e(tr.b.composite_extract(tr.t.f32, None, fcv, [(bit - 8) as u32]))?;
                            tr.to_u(c)?
                        }
                        // POS_W_FLOAT holds the clip-space W (the view depth
                        // the title's soft particles and fog lookups read; AMD's
                        // own drivers take its reciprocal for gl_FragCoord.w),
                        // while FragCoord.w is 1/W: with 1/W the particles'
                        // depth fade never hid them behind walls (2026-11-11).
                        11 => {
                            let c = e(tr.b.composite_extract(tr.t.f32, None, fcv, [3]))?;
                            let one = tr.f(1.0);
                            let w = e(tr.b.f_div(tr.t.f32, None, one, c))?;
                            tr.to_u(w)?
                        }
                        12 => {
                            let f = e(tr.b.load(tr.t.bool_, None, ff, None, []))?;
                            let one = tr.f(1.0);
                            let m1 = tr.f(-1.0);
                            let s = e(tr.b.select(tr.t.f32, None, f, one, m1))?;
                            tr.to_u(s)?
                        }
                        14 => tr.u(1),
                        _ => tr.u(0),
                    };
                    tr.set_vreg_raw(vg + k, val)?;
                }
                vg += n;
            }
        }
        Stage::Compute => {
            let lid = tr.input(tr.t.v3u, Some(BuiltIn::LocalInvocationId), None);
            // Batched dispatches loaded WorkgroupId in the entry block (one
            // variable per built-in).
            let wid = if tr.batch_wg.is_none() { Some(tr.input(tr.t.v3u, Some(BuiltIn::WorkgroupId), None)) } else { None };
            let lidx = tr.input(tr.t.u32, Some(BuiltIn::LocalInvocationIndex), None);
            let l = e(tr.b.load(tr.t.v3u, None, lid, None, []))?;
            let tid_cnt = (tr.spec.cs_rsrc2 >> 11) & 3;
            for k in 0..=tid_cnt.min(2) {
                let c = e(tr.b.composite_extract(tr.t.u32, None, l, [k]))?;
                tr.set_vreg_raw(k, c)?;
            }
            // Batched dispatches: x within the workgroup's own command.
            let w = match (tr.batch_wg, wid) {
                (Some((local, w0)), _) => e(tr.b.composite_insert(tr.t.v3u, None, local, w0, [0]))?,
                (None, Some(wid)) => e(tr.b.load(tr.t.v3u, None, wid, None, []))?,
                (None, None) => return Err("no workgroup id".into()),
            };
            tr.wg_id = Some(w);
            let mut s = tr.spec.user_base as u32 + tr.spec.user_sgprs as u32;
            for k in 0..3 {
                if tr.spec.cs_rsrc2 & (1 << (7 + k)) != 0 {
                    let c = e(tr.b.composite_extract(tr.t.u32, None, w, [k]))?;
                    tr.set_sreg(s, c)?;
                    s += 1;
                }
            }
            if tr.spec.cs_rsrc2 & (1 << 10) != 0 {
                // TG_SIZE: wave id in group [5:0], wave count [11:6]
                // (single-lane waves: report one wave).
                let v = tr.u(1 << 6);
                tr.set_sreg(s, v)?;
            }
            let idx = e(tr.b.load(tr.t.u32, None, lidx, None, []))?;
            tr.local_index = Some(idx);
            // (Subgroup waves: lanes numbered within the subgroup, `T::lane`.)
            if !tr.sg_waves {
                let km = tr.u(if tr.spec.wave64 { 63 } else { 31 });
                tr.lane_id = Some(e(tr.b.bitwise_and(tr.t.u32, None, idx, km))?);
            }
            let lds_dwords = ((tr.spec.cs_rsrc2 >> 15) & 0x1FF) * 128;
            if lds_dwords > 0 {
                let n = lds_dwords.min(12288);
                if n < lds_dwords {
                    tr.warn(format!("LDS of {lds_dwords} dwords clamped to {n}"));
                }
                let len = tr.u(n);
                let arr = tr.b.type_array(tr.t.u32, len);
                let pt = tr.b.type_pointer(None, StorageClass::Workgroup, arr);
                let v = tr.global_var(pt, StorageClass::Workgroup);
                tr.lds = Some(v);
            }
        }
    }
    Ok(())
}

// ---------------------------------------------------------------------------
// Raw buffer access

fn dword_ptr(tr: &mut T, var: Word, idx: Word) -> R<Word> {
    let z = tr.u(0);
    e(tr.b.access_chain(tr.t.sb_u32, None, var, [z, idx]))
}

/// `Spec::addresses`: the resource a buffer placeholder stands for.
fn addressed(tr: &T, var: Word) -> Option<usize> {
    tr.buffer_res.get(&var).copied()
}

/// The u64 of two dwords (low, high).
fn u64_of(tr: &mut T, lo: Word, hi: Word) -> R<Word> {
    let v = e(tr.b.composite_construct(tr.t.v2u, None, [lo, hi]))?;
    e(tr.b.bitcast(tr.t.u64, None, v))
}

/// A device pointer to `ty` (`bytes` wide) at byte `off` of the bound range
/// of address-mode resource `r`, and whether the access lies inside the
/// range; outside it the pointer is the zero page's (loads read 0; stores
/// and atomics are skipped by their callers).
fn addr_ref(tr: &mut T, r: usize, off: Word, ty: Word, bytes: u32) -> R<(Word, Word)> {
    let (u, u64t, b) = (tr.t.u32, tr.t.u64, tr.t.bool_);
    if let Some(v) = tr.rt_vsharp.filter(|v| v.r == r) {
        return rt_ref(tr, v, off, ty, bytes);
    }
    let (base, range) = match tr.addr_vals.get(&r) {
        Some(&v) => v,
        None => {
            let lo = tr.addr_info(r, 0)?;
            let hi = tr.addr_info(r, 1)?;
            let range = tr.addr_info(r, 2)?;
            (u64_of(tr, lo, hi)?, range)
        }
    };
    // off + bytes <= range without wrapping: range >= bytes, off <= range - bytes.
    let nb = tr.u(bytes);
    let fits = e(tr.b.u_greater_than_equal(b, None, range, nb))?;
    let lim = e(tr.b.i_sub(u, None, range, nb))?;
    let le = e(tr.b.u_less_than_equal(b, None, off, lim))?;
    let inr = e(tr.b.logical_and(b, None, fits, le))?;
    let o = e(tr.b.u_convert(u64t, None, off))?;
    let a = e(tr.b.i_add(u64t, None, base, o))?;
    let z = match tr.zero_addr {
        Some(z) => z,
        None => {
            let zlo = tr.addr_header(0)?;
            let zhi = tr.addr_header(1)?;
            u64_of(tr, zlo, zhi)?
        }
    };
    let a = e(tr.b.select(u64t, None, inr, a, z))?;
    let pt = tr.b.type_pointer(None, StorageClass::PhysicalStorageBuffer, ty);
    let p = e(tr.b.convert_u_to_ptr(pt, None, a))?;
    Ok((p, inr))
}

/// The zero page's device address (loaded once in the entry block).
fn zero_page(tr: &mut T) -> R<Word> {
    match tr.zero_addr {
        Some(z) => Ok(z),
        None => {
            let zlo = tr.addr_header(0)?;
            let zhi = tr.addr_header(1)?;
            u64_of(tr, zlo, zhi)
        }
    }
}

/// GPU-resolved chains: the device address of guest address `a` (u64)
/// through the guest page table, and whether the table maps it (else the
/// address is meaningless: callers use the zero page). `linear`: the
/// GPU-resolved buffer it is read for; with `Spec::pt_linear` every such
/// read goes to one buffer at guest offsets instead (front end 2's mirror),
/// inside the span only: a buffer's through the device address in its
/// address entry, a scalar load's (`linear` None) through the address
/// block header's dwords 2-3, which then hold the mirror's device address
/// in place of the page table's (since 2026-11-18: before, loads read
/// through the page table of host import windows, across PCIe).
fn pt_translate(tr: &mut T, a: Word, linear: Option<usize>) -> R<(Word, Word)> {
    let (u64t, b) = (tr.t.u64, tr.t.bool_);
    let pt = match tr.pt_addr {
        Some(p) => p,
        None => {
            let lo = tr.addr_header(2)?;
            let hi = tr.addr_header(3)?;
            u64_of(tr, lo, hi)?
        }
    };
    let lo_c = tr.u64c(super::PT_BASE);
    let hi_c = tr.u64c(super::PT_END);
    let ge = e(tr.b.u_greater_than_equal(b, None, a, lo_c))?;
    let lt = e(tr.b.u_less_than(b, None, a, hi_c))?;
    let inside = e(tr.b.logical_and(b, None, ge, lt))?;
    let rel = e(tr.b.i_sub(u64t, None, a, lo_c))?;
    let z64 = tr.u64c(0);
    let rel = e(tr.b.select(u64t, None, inside, rel, z64))?;
    if tr.spec.pt_linear {
        let base = match linear {
            Some(r) => {
                let lo = tr.addr_info(r, 0)?;
                let hi = tr.addr_info(r, 1)?;
                u64_of(tr, lo, hi)?
            }
            None => pt,
        };
        let dev = e(tr.b.i_add(u64t, None, base, rel))?;
        return Ok((dev, inside));
    }
    let sh = tr.u64c(super::PT_PAGE_SHIFT as u64);
    let page = e(tr.b.shift_right_logical(u64t, None, rel, sh))?;
    let three = tr.u64c(3);
    let eoff = e(tr.b.shift_left_logical(u64t, None, page, three))?;
    let ea = e(tr.b.i_add(u64t, None, pt, eoff))?;
    let pty = tr.b.type_pointer(None, StorageClass::PhysicalStorageBuffer, u64t);
    let ep = e(tr.b.convert_u_to_ptr(pty, None, ea))?;
    let (ma, ops) = aligned(8);
    let entry = e(tr.b.load(u64t, None, ep, ma, ops))?;
    let mapped = e(tr.b.i_not_equal(b, None, entry, z64))?;
    let ok_ = e(tr.b.logical_and(b, None, inside, mapped))?;
    let m = tr.u64c((1u64 << super::PT_PAGE_SHIFT) - 1);
    let low = e(tr.b.bitwise_and(u64t, None, a, m))?;
    let dev = e(tr.b.i_add(u64t, None, entry, low))?;
    Ok((dev, ok_))
}

/// `addr_ref` for a GPU-resolved resource (`Spec::gpu_res`): byte `off` of
/// the V#'s range at guest address base + off, through the page table.
fn rt_ref(tr: &mut T, v: super::RtVsharp, off: Word, ty: Word, bytes: u32) -> R<(Word, Word)> {
    let (u, u64t, b) = (tr.t.u32, tr.t.u64, tr.t.bool_);
    let nb = tr.u(bytes);
    let fits = e(tr.b.u_greater_than_equal(b, None, v.range, nb))?;
    let lim = e(tr.b.i_sub(u, None, v.range, nb))?;
    let le = e(tr.b.u_less_than_equal(b, None, off, lim))?;
    let inr = e(tr.b.logical_and(b, None, fits, le))?;
    let o = e(tr.b.u_convert(u64t, None, off))?;
    let ga = e(tr.b.i_add(u64t, None, v.base, o))?;
    let (dev, mapped) = pt_translate(tr, ga, Some(v.r))?;
    let inr = e(tr.b.logical_and(b, None, inr, mapped))?;
    let z = zero_page(tr)?;
    let a = e(tr.b.select(u64t, None, inr, dev, z))?;
    let pt = tr.b.type_pointer(None, StorageClass::PhysicalStorageBuffer, ty);
    let p = e(tr.b.convert_u_to_ptr(pt, None, a))?;
    Ok((p, inr))
}

/// A GPU-resolved resource's V# from the four SGPRs at `sr` (`Spec::gpu_res`).
fn rt_vsharp(tr: &mut T, r: usize, sr: u32) -> R<super::RtVsharp> {
    let (u, b) = (tr.t.u32, tr.t.bool_);
    let w0 = tr.sreg(sr)?;
    let w1 = tr.sreg(sr + 1)?;
    let w2 = tr.sreg(sr + 2)?;
    let w3 = tr.sreg(sr + 3)?;
    let m16 = tr.u(0xFFFF);
    let hi = e(tr.b.bitwise_and(u, None, w1, m16))?;
    // Aligned down to 16 bytes (vector accesses check their alignment
    // against the offset, as for the CPU path's placements).
    let k15 = tr.u(15);
    let bias = e(tr.b.bitwise_and(u, None, w0, k15))?;
    let nk15 = tr.u(!15);
    let lo_al = e(tr.b.bitwise_and(u, None, w0, nk15))?;
    let base = u64_of(tr, lo_al, hi)?;
    let sixteen = tr.u(16);
    let st = e(tr.b.shift_right_logical(u, None, w1, sixteen))?;
    let m14 = tr.u(0x3FFF);
    let stride = e(tr.b.bitwise_and(u, None, st, m14))?;
    // OOB_SELECT (word 3 bits 28-29): 2 checks nothing.
    let k28 = tr.u(28);
    let os = e(tr.b.shift_right_logical(u, None, w3, k28))?;
    let three = tr.u(3);
    let os = e(tr.b.bitwise_and(u, None, os, three))?;
    let two = tr.u(2);
    let one = tr.u(1);
    let unchecked = e(tr.b.i_equal(b, None, os, two))?;
    let z = tr.u(0);
    let structured = e(tr.b.i_not_equal(b, None, stride, z))?;
    let idx_checked = e(tr.b.u_less_than_equal(b, None, os, one))?;
    let idx_checked = e(tr.b.logical_and(b, None, idx_checked, structured))?;
    let all = tr.u(u32::MAX);
    let limit = e(tr.b.select(u, None, idx_checked, w2, all))?;
    let bytes = e(tr.b.i_mul(u, None, w2, stride))?;
    let extent = e(tr.b.select(u, None, structured, bytes, w2))?;
    let extent = e(tr.b.i_add(u, None, extent, bias))?;
    let range = e(tr.b.select(u, None, unchecked, all, extent))?;
    Ok(super::RtVsharp { r, base, bias, stride, limit, range })
}

/// GPU-resolved chains: counts a read through a page the table does not
/// map when `miss` holds (the table's feedback words: count, the guest
/// address). Call at an instruction's top level (it opens a block).
/// Nothing under `Spec::pt_linear`: the header's address is the mirror's
/// there, and the u64 at its feedback offset is guest data (a command
/// chunk's first words, used as an atomic's address, lost the device on
/// 2026-11-18); reads outside the span read zeros, uncounted.
fn report_miss(tr: &mut T, miss: Word, guest: Word) -> R<()> {
    if tr.spec.pt_linear {
        return Ok(());
    }
    let (u, u64t) = (tr.t.u32, tr.t.u64);
    let pt = match tr.pt_addr {
        Some(p) => p,
        None => {
            let lo = tr.addr_header(2)?;
            let hi = tr.addr_header(3)?;
            u64_of(tr, lo, hi)?
        }
    };
    tr.guarded(miss, &mut |tr| {
        let off = tr.u64c(super::PT_FEEDBACK);
        let fa = e(tr.b.i_add(u64t, None, pt, off))?;
        let p64 = tr.b.type_pointer(None, StorageClass::PhysicalStorageBuffer, u64t);
        let fp = e(tr.b.convert_u_to_ptr(p64, None, fa))?;
        let (ma, ops) = aligned(8);
        let fb = e(tr.b.load(u64t, None, fp, ma, ops))?;
        let p32 = tr.b.type_pointer(None, StorageClass::PhysicalStorageBuffer, u);
        let cp = e(tr.b.convert_u_to_ptr(p32, None, fb))?;
        let scope = tr.u(spirv::Scope::Device as u32);
        let sem = tr.u(0);
        let one = tr.u(1);
        e(tr.b.atomic_i_add(u, None, cp, scope, sem, one))?;
        let eight = tr.u64c(8);
        let la = e(tr.b.i_add(u64t, None, fb, eight))?;
        let lp = e(tr.b.convert_u_to_ptr(p64, None, la))?;
        let (ma, ops) = aligned(8);
        ok(tr.b.store(lp, guest, ma, ops))
    })
}

/// A GPU-resolved resource's first byte through the page table: a read of
/// its range would miss when the table does not map the page (counted).
fn check_rt_base(tr: &mut T, v: super::RtVsharp) -> R<()> {
    // (Read from the mirror: no table entry to miss, `Spec::pt_linear`.)
    if tr.spec.pt_linear {
        return Ok(());
    }
    let (u, b) = (tr.t.u32, tr.t.bool_);
    let (_, mapped) = pt_translate(tr, v.base, None)?;
    let z = tr.u(0);
    let some = e(tr.b.u_greater_than(b, None, v.range, z))?;
    let not_mapped = e(tr.b.logical_not(b, None, mapped))?;
    let miss = e(tr.b.logical_and(b, None, some, not_mapped))?;
    let _ = u;
    report_miss(tr, miss, v.base)
}

/// Whether resource `r` is GPU-resolved (`Spec::gpu_res`).
fn gpu_resolved(tr: &T, r: usize) -> bool {
    r < 64 && tr.spec.gpu_res & (1u64 << r) != 0
}

/// A GPU-resolved s_load (`Spec::gpu_loads`): `n` dwords from the address
/// its base SGPRs hold (plus its offsets) through the guest page table;
/// dwords on pages the table does not map read 0. A load spans at most two
/// pages.
fn gpu_smem(tr: &mut T, ins: &Inst, d: u32, n: u32) -> R<()> {
    let (u, u64t, b) = (tr.t.u32, tr.t.u64, tr.t.bool_);
    let sb = ins.smem_sbase() as u32;
    let lo = tr.sreg(sb)?;
    let hi = tr.sreg(sb + 1)?;
    let m16 = tr.u(0xFFFF);
    let hi = e(tr.b.bitwise_and(u, None, hi, m16))?;
    let base = u64_of(tr, lo, hi)?;
    let off = tr.u64c(ins.smem_offset() as i64 as u64);
    let mut a = e(tr.b.i_add(u64t, None, base, off))?;
    if ins.smem_soffset() != crate::OP_NULL {
        let so = tr.operand(ins, ins.smem_soffset())?;
        let so = e(tr.b.u_convert(u64t, None, so))?;
        a = e(tr.b.i_add(u64t, None, a, so))?;
    }
    let am = tr.u64c(!3u64);
    let a = e(tr.b.bitwise_and(u64t, None, a, am))?;
    let (dev0, ok0) = pt_translate(tr, a, None)?;
    let (dev1, ok1) = if n > 1 {
        let last = tr.u64c(4 * (n as u64 - 1));
        let al = e(tr.b.i_add(u64t, None, a, last))?;
        pt_translate(tr, al, None)?
    } else {
        (dev0, ok0)
    };
    let both = e(tr.b.logical_and(b, None, ok0, ok1))?;
    let miss = e(tr.b.logical_not(b, None, both))?;
    report_miss(tr, miss, a)?;
    let pm = tr.u64c((1u64 << super::PT_PAGE_SHIFT) - 1);
    let low = e(tr.b.bitwise_and(u64t, None, a, pm))?;
    let low = e(tr.b.u_convert(u, None, low))?;
    let z = zero_page(tr)?;
    let zero = tr.u(0);
    for j in 0..n {
        // Dword j: on the first page unless low + 4j reaches the next.
        let (dev, ok_) = if n > 1 {
            let k = tr.u(4 * j);
            let at = e(tr.b.i_add(u, None, low, k))?;
            let page = tr.u(1 << super::PT_PAGE_SHIFT);
            let first = e(tr.b.u_less_than(b, None, at, page))?;
            let o0 = tr.u64c(4 * j as u64);
            let d0 = e(tr.b.i_add(u64t, None, dev0, o0))?;
            let o1 = tr.u64c(4 * (n - 1 - j) as u64);
            let d1 = e(tr.b.i_sub(u64t, None, dev1, o1))?;
            let dev = e(tr.b.select(u64t, None, first, d0, d1))?;
            let ok_ = e(tr.b.select(b, None, first, ok0, ok1))?;
            (dev, ok_)
        } else {
            (dev0, ok0)
        };
        let dev = e(tr.b.select(u64t, None, ok_, dev, z))?;
        let pty = tr.b.type_pointer(None, StorageClass::PhysicalStorageBuffer, u);
        let p = e(tr.b.convert_u_to_ptr(pty, None, dev))?;
        let (ma, ops) = aligned(4);
        let v = e(tr.b.load(u, None, p, ma, ops))?;
        let v = e(tr.b.select(u, None, ok_, v, zero))?;
        tr.set_sreg(d + j, v)?;
    }
    Ok(())
}

/// Loads the address-mode resources' addresses and ranges and the zero
/// page's address (`T::addr_vals`); call in the entry block, after the SGPR
/// data's base is known.
pub fn preload_addresses(tr: &mut T) -> R<()> {
    let mut rs: Vec<usize> = tr.buffer_res.values().copied().collect();
    rs.sort_unstable();
    for r in rs {
        let lo = tr.addr_info(r, 0)?;
        let hi = tr.addr_info(r, 1)?;
        let range = tr.addr_info(r, 2)?;
        let base = u64_of(tr, lo, hi)?;
        tr.addr_vals.insert(r, (base, range));
    }
    let zlo = tr.addr_header(0)?;
    let zhi = tr.addr_header(1)?;
    tr.zero_addr = Some(u64_of(tr, zlo, zhi)?);
    if tr.spec.gpu_loads != 0 || tr.spec.gpu_res != 0 {
        let lo = tr.addr_header(2)?;
        let hi = tr.addr_header(3)?;
        tr.pt_addr = Some(u64_of(tr, lo, hi)?);
    }
    Ok(())
}

/// A pointer to dword `idx` of buffer `var`, and for an address-mode
/// resource whether the dword lies inside its range (writes must check).
fn dword_ref(tr: &mut T, var: Word, idx: Word) -> R<(Word, Option<Word>)> {
    if let Some(r) = addressed(tr, var) {
        let two = tr.u(2);
        let off = e(tr.b.shift_left_logical(tr.t.u32, None, idx, two))?;
        let u = tr.t.u32;
        let (p, inr) = addr_ref(tr, r, off, u, 4)?;
        return Ok((p, Some(inr)));
    }
    Ok((dword_ptr(tr, var, idx)?, None))
}

/// The Aligned memory operand of a device-pointer access of `n` bytes.
fn aligned(n: u32) -> (Option<spirv::MemoryAccess>, [Operand; 1]) {
    (Some(spirv::MemoryAccess::ALIGNED), [Operand::LiteralBit32(n)])
}

fn load_dword(tr: &mut T, var: Word, idx: Word) -> R<Word> {
    let (p, inr) = dword_ref(tr, var, idx)?;
    if inr.is_some() {
        let (ma, ops) = aligned(4);
        return e(tr.b.load(tr.t.u32, None, p, ma, ops));
    }
    e(tr.b.load(tr.t.u32, None, p, None, []))
}

fn store_dword(tr: &mut T, var: Word, idx: Word, v: Word) -> R<()> {
    let (p, inr) = dword_ref(tr, var, idx)?;
    match inr {
        Some(c) => tr.guarded(c, &mut |tr| {
            let (ma, ops) = aligned(4);
            ok(tr.b.store(p, v, ma, ops))
        }),
        None => ok(tr.b.store(p, v, None, [])),
    }
}

/// How an aligned 8/16-byte access of a buffer is made: through a vector
/// view of its binding, or (address mode) a device pointer and its range
/// check.
#[derive(Clone, Copy)]
enum VecAccess {
    View(Word),
    Addr(Word, Word),
}

fn vec_access(tr: &mut T, var: Word, addr: Word, n: u32) -> R<Option<VecAccess>> {
    if n != 2 && n != 4 {
        return Ok(None);
    }
    if let Some(r) = addressed(tr, var) {
        let vt = if n == 2 { tr.t.v2u } else { tr.t.v4u };
        let (p, inr) = addr_ref(tr, r, addr, vt, 4 * n)?;
        return Ok(Some(VecAccess::Addr(p, inr)));
    }
    Ok(tr.vec_view(var, n).map(VecAccess::View))
}

/// Loads `n` dwords at byte address `addr`: one 8/16-byte access when the
/// address is aligned to it (n = 2 or 4), else dword by dword (`bytes`:
/// unaligned-safe byte loads instead of dword-truncated addresses).
fn load_dwords(tr: &mut T, var: Word, addr: Word, n: u32, bytes: bool) -> R<Vec<Word>> {
    let u = tr.t.u32;
    let scalar = |tr: &mut T| -> R<Vec<Word>> {
        let mut v = Vec::new();
        if bytes {
            for k in 0..n {
                let o = tr.u(4 * k);
                let a = e(tr.b.i_add(u, None, addr, o))?;
                v.push(load_bytes(tr, var, a, 4)?);
            }
        } else {
            let two = tr.u(2);
            let idx = e(tr.b.shift_right_logical(u, None, addr, two))?;
            for k in 0..n {
                let kk = tr.u(k);
                let i = e(tr.b.i_add(u, None, idx, kk))?;
                v.push(load_dword(tr, var, i)?);
            }
        }
        Ok(v)
    };
    let Some(how) = vec_access(tr, var, addr, n)? else { return scalar(tr) };
    let mask = tr.u(4 * n - 1);
    let low = e(tr.b.bitwise_and(u, None, addr, mask))?;
    let z = tr.u(0);
    let mut aligned_ = e(tr.b.i_equal(tr.t.bool_, None, low, z))?;
    if let VecAccess::Addr(_, inr) = how {
        aligned_ = e(tr.b.logical_and(tr.t.bool_, None, aligned_, inr))?;
    }
    let (merge, then_b, else_b) = (tr.b.id(), tr.b.id(), tr.b.id());
    ok(tr.b.selection_merge(merge, spirv::SelectionControl::NONE))?;
    ok(tr.b.branch_conditional(aligned_, then_b, else_b, []))?;
    e(tr.b.begin_block(Some(then_b)))?;
    let vt = if n == 2 { tr.t.v2u } else { tr.t.v4u };
    let vv = match how {
        VecAccess::View(view) => {
            let sh = tr.u(if n == 2 { 3 } else { 4 });
            let idx = e(tr.b.shift_right_logical(u, None, addr, sh))?;
            let pt = tr.b.type_pointer(None, StorageClass::StorageBuffer, vt);
            let p = e(tr.b.access_chain(pt, None, view, [z, idx]))?;
            e(tr.b.load(vt, None, p, None, []))?
        }
        VecAccess::Addr(p, _) => {
            let (ma, ops) = aligned(4 * n);
            e(tr.b.load(vt, None, p, ma, ops))?
        }
    };
    let mut va = Vec::new();
    for k in 0..n {
        va.push(e(tr.b.composite_extract(u, None, vv, [k]))?);
    }
    ok(tr.b.branch(merge))?;
    e(tr.b.begin_block(Some(else_b)))?;
    let vs = scalar(tr)?;
    // (The scalar path may open blocks: it ends in the block being emitted
    // now, which the phi names; an invalid phi crashed the driver's
    // compiler, 2026-11-10.)
    let else_end = tr.current_label()?;
    ok(tr.b.branch(merge))?;
    e(tr.b.begin_block(Some(merge)))?;
    let mut out = Vec::new();
    for k in 0..n as usize {
        out.push(e(tr.b.phi(u, None, [(va[k], then_b), (vs[k], else_end)]))?);
    }
    Ok(out)
}

/// Stores dwords at byte address `addr` (one 8/16-byte store when aligned).
fn store_dwords(tr: &mut T, var: Word, addr: Word, vals: &[Word]) -> R<()> {
    let u = tr.t.u32;
    let n = vals.len() as u32;
    let scalar = |tr: &mut T| -> R<()> {
        let two = tr.u(2);
        let idx = e(tr.b.shift_right_logical(u, None, addr, two))?;
        for (k, v) in vals.iter().enumerate() {
            let kk = tr.u(k as u32);
            let i = e(tr.b.i_add(u, None, idx, kk))?;
            store_dword(tr, var, i, *v)?;
        }
        Ok(())
    };
    let Some(how) = vec_access(tr, var, addr, n)? else { return scalar(tr) };
    let mask = tr.u(4 * n - 1);
    let low = e(tr.b.bitwise_and(u, None, addr, mask))?;
    let z = tr.u(0);
    let mut aligned_ = e(tr.b.i_equal(tr.t.bool_, None, low, z))?;
    if let VecAccess::Addr(_, inr) = how {
        aligned_ = e(tr.b.logical_and(tr.t.bool_, None, aligned_, inr))?;
    }
    let (merge, then_b, else_b) = (tr.b.id(), tr.b.id(), tr.b.id());
    ok(tr.b.selection_merge(merge, spirv::SelectionControl::NONE))?;
    ok(tr.b.branch_conditional(aligned_, then_b, else_b, []))?;
    e(tr.b.begin_block(Some(then_b)))?;
    let vt = if n == 2 { tr.t.v2u } else { tr.t.v4u };
    let vv = e(tr.b.composite_construct(vt, None, vals.to_vec()))?;
    match how {
        VecAccess::View(view) => {
            let sh = tr.u(if n == 2 { 3 } else { 4 });
            let idx = e(tr.b.shift_right_logical(u, None, addr, sh))?;
            let pt = tr.b.type_pointer(None, StorageClass::StorageBuffer, vt);
            let p = e(tr.b.access_chain(pt, None, view, [z, idx]))?;
            ok(tr.b.store(p, vv, None, []))?;
        }
        VecAccess::Addr(p, _) => {
            let (ma, ops) = aligned(4 * n);
            ok(tr.b.store(p, vv, ma, ops))?;
        }
    }
    ok(tr.b.branch(merge))?;
    e(tr.b.begin_block(Some(else_b)))?;
    scalar(tr)?;
    ok(tr.b.branch(merge))?;
    e(tr.b.begin_block(Some(merge)))?;
    Ok(())
}

/// Loads `nbytes` (1..=4) little-endian bytes at an arbitrary byte address.
fn load_bytes(tr: &mut T, var: Word, addr: Word, nbytes: u32) -> R<Word> {
    let u = tr.t.u32;
    let two = tr.u(2);
    let idx = e(tr.b.shift_right_logical(u, None, addr, two))?;
    let lo = load_dword(tr, var, idx)?;
    let one = tr.u(1);
    let idx1 = e(tr.b.i_add(u, None, idx, one))?;
    let hi = load_dword(tr, var, idx1)?;
    let wide = {
        let v = e(tr.b.composite_construct(tr.t.v2u, None, [lo, hi]))?;
        e(tr.b.bitcast(tr.t.u64, None, v))?
    };
    let k3 = tr.u(3);
    let sub = e(tr.b.bitwise_and(u, None, addr, k3))?;
    let sh = e(tr.b.shift_left_logical(u, None, sub, k3))?;
    let shifted = e(tr.b.shift_right_logical(tr.t.u64, None, wide, sh))?;
    let r = e(tr.b.u_convert(u, None, shifted))?;
    if nbytes >= 4 {
        return Ok(r);
    }
    let m = tr.u((1u32 << (nbytes * 8)) - 1);
    e(tr.b.bitwise_and(u, None, r, m))
}

/// Stores the low `nbytes` (1 or 2) bytes of `v` at a byte address
/// (atomic read-modify-write of the containing dword).
fn store_bytes(tr: &mut T, var: Word, addr: Word, v: Word, nbytes: u32) -> R<()> {
    let u = tr.t.u32;
    let two = tr.u(2);
    let idx = e(tr.b.shift_right_logical(u, None, addr, two))?;
    let k3 = tr.u(3);
    let sub = e(tr.b.bitwise_and(u, None, addr, k3))?;
    let sh = e(tr.b.shift_left_logical(u, None, sub, k3))?;
    let m = tr.u((1u32 << (nbytes * 8)) - 1);
    let mask = e(tr.b.shift_left_logical(u, None, m, sh))?;
    let vm = e(tr.b.bitwise_and(u, None, v, m))?;
    let vs = e(tr.b.shift_left_logical(u, None, vm, sh))?;
    let (p, inr) = dword_ref(tr, var, idx)?;
    let scope = tr.u(spirv::Scope::Device as u32);
    let sem = tr.u(0);
    let nm = e(tr.b.not(u, None, mask))?;
    let mut rmw = |tr: &mut T| -> R<()> {
        e(tr.b.atomic_and(u, None, p, scope, sem, nm))?;
        e(tr.b.atomic_or(u, None, p, scope, sem, vs))?;
        Ok(())
    };
    match inr {
        Some(c) => tr.guarded(c, &mut rmw),
        None => rmw(tr),
    }
}

// ---------------------------------------------------------------------------
// Scalar memory

fn smem_dwords(name: &str) -> u32 {
    for (suf, n) in [("x16", 16), ("x8", 8), ("x4", 4), ("x2", 2)] {
        if name.ends_with(suf) {
            return n;
        }
    }
    1
}

pub fn smem(tr: &mut T, ins: &Inst) -> R<()> {
    let name = ins.name().unwrap_or("");
    let u = tr.t.u32;
    let d = ins.sdst() as u32;
    // Dynamic-offset s_load from a known base: read through its binding
    // (same path as s_buffer_load).
    let dynamic_load = name.starts_with("s_load_dword") && !tr.plan.inst_load.contains_key(&ins.pc) && tr.plan.inst_res.contains_key(&ins.pc);
    if name.starts_with("s_load_dword") && !dynamic_load {
        let n = smem_dwords(name);
        let Some(&k) = tr.plan.inst_load.get(&ins.pc) else {
            tr.warn(format!("{:#x}: s_load without resolved address (zero)", ins.pc));
            for j in 0..n {
                let z = tr.u(0);
                tr.set_sreg(d + j, z)?;
            }
            return Ok(());
        };
        if k < 64 && tr.spec.gpu_loads & (1u64 << k) != 0 {
            return gpu_smem(tr, ins, d, n);
        }
        let base = tr.load_offsets[k];
        for j in 0..n {
            let v = tr.sgpr_data(base + j)?;
            tr.set_sreg(d + j, v)?;
        }
        return Ok(());
    }
    if name.starts_with("s_buffer_load_dword") || dynamic_load {
        let n = smem_dwords(name);
        let Some(&(r, _)) = tr.plan.inst_res.get(&ins.pc) else { return Err(format!("{:#x}: s_buffer_load without resource", ins.pc)) };
        let var = tr.buffers[&r];
        let mut off = tr.u(ins.smem_offset() as u32);
        let mut dynamic = None;
        if ins.smem_soffset() != crate::OP_NULL {
            let so = tr.operand(ins, ins.smem_soffset())?;
            dynamic = Some(so);
            off = e(tr.b.i_add(u, None, off, so))?;
        }
        if let Some(&v) = tr.table_offsets.get(&ins.pc) {
            // The table entry this load reads (an image table's index): the
            // offset into a V#'s range, or for a T#-sized s_load the offset
            // register alone (its table starts at the immediate offset).
            let pointer_table = tr.plan.table_loads.get(&ins.pc).is_some_and(|d| d.len() == 2);
            let x = match (pointer_table, dynamic) {
                (true, Some(so)) => so,
                _ => off,
            };
            ok(tr.b.store(v, x, None, []))?;
        }
        let km = tr.u(!3);
        let off = e(tr.b.bitwise_and(u, None, off, km))?;
        if gpu_resolved(tr, r) {
            // The V# in the base SGPRs, as the instruction reads it.
            let v = rt_vsharp(tr, r, ins.smem_sbase() as u32)?;
            check_rt_base(tr, v)?;
            let ut = tr.t.u32;
            for j in 0..n {
                let kb = tr.u(4 * j);
                let o = e(tr.b.i_add(u, None, off, kb))?;
                let o = e(tr.b.i_add(u, None, o, v.bias))?;
                let (p, inr) = rt_ref(tr, v, o, ut, 4)?;
                let (ma, ops) = aligned(4);
                let x = e(tr.b.load(u, None, p, ma, ops))?;
                let z = tr.u(0);
                let x = e(tr.b.select(u, None, inr, x, z))?;
                tr.set_sreg(d + j, x)?;
            }
            return Ok(());
        }
        let bias = tr.res_info(r, 0)?;
        // Out-of-range dwords read 0: checked here, or by the device past the
        // binding's range (`Spec::hw_bounds`, a buffer only scalar loads
        // read: the runtime binds it to end where these checks would,
        // `Output::unchecked_res`).
        let unchecked = tr.spec.hw_bounds && r < 64 && tr.scalar_only & (1u64 << r) != 0;
        if unchecked {
            tr.unchecked_res |= 1u64 << r;
        }
        let size = if unchecked { None } else { Some(tr.res_info(r, 2)?) };
        let addr = e(tr.b.i_add(u, None, off, bias))?;
        let two = tr.u(2);
        let idx0 = e(tr.b.shift_right_logical(u, None, addr, two))?;
        for j in 0..n {
            let kj = tr.u(j);
            let idx = e(tr.b.i_add(u, None, idx0, kj))?;
            let v = load_dword(tr, var, idx)?;
            let v = match size {
                Some(size) => {
                    let kb = tr.u(4 * j);
                    let o = e(tr.b.i_add(u, None, off, kb))?;
                    let inr = e(tr.b.u_less_than(tr.t.bool_, None, o, size))?;
                    let z = tr.u(0);
                    e(tr.b.select(u, None, inr, v, z))?
                }
                None => v,
            };
            tr.set_sreg(d + j, v)?;
        }
        return Ok(());
    }
    match name {
        "s_dcache_inv" | "s_gl1_inv" | "s_dcache_wb" | "s_dcache_discard" | "s_dcache_discard_x2" | "s_atc_probe" | "s_atc_probe_buffer" => {}
        "s_memtime" | "s_memrealtime" | "s_get_waveid_in_workgroup" => {
            let z = tr.u(0);
            tr.set_sreg(d, z)?;
            if name != "s_get_waveid_in_workgroup" {
                tr.set_sreg(d + 1, z)?;
            }
        }
        _ => tr.warn(format!("{:#x}: unsupported {name}", ins.pc)),
    }
    Ok(())
}

// ---------------------------------------------------------------------------
// Attribute interpolation

fn attr_input(tr: &mut T, attr: u32) -> R<Option<Word>> {
    let Some((loc, flat)) = super::attr_location(tr.spec, attr) else { return Ok(None) };
    tr.params = tr.params.max(attr + 1);
    if let Some(&v) = tr.in_attrs.get(&loc) {
        return Ok(Some(v));
    }
    let v = tr.input(tr.t.v4f, None, Some(loc));
    if flat {
        tr.b.decorate(v, Decoration::Flat, []);
    }
    tr.in_attrs.insert(loc, v);
    Ok(Some(v))
}

/// v_interp_p1/p2/mov: the interpolated (or flat) attribute value is
/// written by each step; the barycentric arithmetic is done by the host.
pub fn interp(tr: &mut T, ins: &Inst) -> R<()> {
    let attr = ins.intrp_attr();
    let chan = ins.intrp_chan();
    if super::attr_location(tr.spec, attr).is_some_and(|(loc, _)| tr.per_vertex_locs.contains(&loc)) {
        return interp_per_vertex(tr, ins, attr, chan);
    }
    let val = match attr_input(tr, attr)? {
        Some(var) => {
            let v = e(tr.b.load(tr.t.v4f, None, var, None, []))?;
            e(tr.b.composite_extract(tr.t.f32, None, v, [chan]))?
        }
        None => {
            let d = tr.spec.ps_inputs[attr as usize].default.unwrap_or(0);
            let c = [[0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 1.0], [1.0, 1.0, 1.0, 0.0], [1.0, 1.0, 1.0, 1.0]][d as usize & 3][chan as usize];
            tr.f(c)
        }
    };
    let u = tr.to_u(val)?;
    tr.set_vreg(ins.vdst() as u32, u)
}

/// Interpolation of an attribute whose location is read per vertex (see
/// `per_vertex_locations`): v_interp_mov returns P0 = vertex 0's value and
/// P10/P20 = vertex 1's / 2's (flat-shaded inputs keep raw values) or their
/// differences from P0 (interpolated inputs); v_interp_p1/p2 give the
/// interpolated value (the provoking vertex's of a flat-shaded input).
fn interp_per_vertex(tr: &mut T, ins: &Inst, attr: u32, chan: u32) -> R<()> {
    let f = tr.t.f32;
    let pi = tr.spec.ps_inputs.get(attr as usize).copied();
    if let Some(d) = pi.and_then(|p| p.default) {
        let c = [[0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 1.0], [1.0, 1.0, 1.0, 0.0], [1.0, 1.0, 1.0, 1.0]][d as usize & 3][chan as usize];
        let v = tr.f(c);
        let u = tr.to_u(v)?;
        return tr.set_vreg(ins.vdst() as u32, u);
    }
    let flat = pi.is_some_and(|p| p.flat);
    let loc = pi.map(|p| p.param as u32).unwrap_or(attr);
    tr.params = tr.params.max(attr + 1);
    // Each vertex's value: a per-vertex input array, or without fragment
    // barycentrics three flat inputs the geometry shader writes
    // (`gs::bary_gs`: vertex 0's at the location itself).
    let gs_at = tr.bary_io.as_ref().and_then(|io| io.per_vertex.iter().find(|(l, _)| *l == loc).map(|(_, m)| *m));
    let vars: Result<Word, [Word; 3]> = match gs_at {
        Some(more) => Err(match tr.per_vertex_gs.get(&loc) {
            Some(&v) => v,
            None => {
                let mut v = [0; 3];
                for (k, at) in [loc, more[0], more[1]].into_iter().enumerate() {
                    v[k] = tr.input(tr.t.v4f, None, Some(at));
                    tr.b.decorate(v[k], Decoration::Flat, []);
                }
                tr.per_vertex_gs.insert(loc, v);
                v
            }
        }),
        None => Ok(match tr.per_vertex_in.get(&loc) {
            Some(&v) => v,
            None => {
                let k3 = tr.u(3);
                let arr = tr.b.type_array(tr.t.v4f, k3);
                let v = tr.input(arr, None, Some(loc));
                tr.b.decorate(v, Decoration::PerVertexKHR, []);
                tr.per_vertex_in.insert(loc, v);
                v
            }
        }),
    };
    let pf = tr.b.type_pointer(None, StorageClass::Input, f);
    // (The per-vertex array's constants in their old order: translations
    // with fragment barycentrics keep their SPIR-V, and their stored
    // pipelines their fingerprints.)
    let vert = |tr: &mut T, k: u32| -> R<Word> {
        let p = match vars {
            Ok(arr) => {
                let kk = tr.u(k);
                let kc = tr.u(chan);
                e(tr.b.access_chain(pf, None, arr, [kk, kc]))?
            }
            Err(v) => {
                let kc = tr.u(chan);
                e(tr.b.access_chain(pf, None, v[k as usize], [kc]))?
            }
        };
        e(tr.b.load(f, None, p, None, []))
    };
    let val = if ins.base_name() == Some("v_interp_mov_f32") {
        match ins.intrp_vsrc() & 3 {
            2 => vert(tr, 0)?,
            sel => {
                let x = vert(tr, sel as u32 + 1)?;
                if flat {
                    x
                } else {
                    let p0 = vert(tr, 0)?;
                    e(tr.b.f_sub(f, None, x, p0))?
                }
            }
        }
    } else if flat {
        vert(tr, 0)?
    } else {
        let (pv, _) = tr.bary.ok_or("per-vertex attribute without barycentrics")?;
        let mut acc = tr.f(0.0);
        for k in 0..3 {
            let w = e(tr.b.composite_extract(f, None, pv, [k]))?;
            let x = vert(tr, k)?;
            acc = tr.glsl3(f, GLOp::Fma, w, x, acc)?;
        }
        acc
    };
    let u = tr.to_u(val)?;
    tr.set_vreg(ins.vdst() as u32, u)
}

// ---------------------------------------------------------------------------
// Buffer formats

/// Converts a raw component (`bits` wide) to a 32-bit register value.
fn convert_component(tr: &mut T, raw: Word, bits_: u32, num: Num) -> R<Word> {
    let u = tr.t.u32;
    let f = tr.t.f32;
    let z = tr.u(0);
    let w = tr.u(bits_);
    let sext = |tr: &mut T| -> R<Word> { if bits_ == 32 { Ok(raw) } else { e(tr.b.bit_field_s_extract(u, None, raw, z, w)) } };
    Ok(match num {
        Num::Unorm | Num::Srgb => {
            let x = e(tr.b.convert_u_to_f(f, None, raw))?;
            let k = tr.f(((1u64 << bits_) - 1) as f32);
            let r = e(tr.b.f_div(f, None, x, k))?;
            let r = if num == Num::Srgb && bits_ == 8 { srgb_to_linear(tr, r)? } else { r };
            tr.to_u(r)?
        }
        Num::Snorm => {
            let s = sext(tr)?;
            let si = tr.to_i(s)?;
            let x = e(tr.b.convert_s_to_f(f, None, si))?;
            let k = tr.f(((1u64 << (bits_ - 1)) - 1) as f32);
            let r = e(tr.b.f_div(f, None, x, k))?;
            let m1 = tr.f(-1.0);
            let r = tr.glsl2(f, GLOp::FMax, r, m1)?;
            tr.to_u(r)?
        }
        Num::Uscaled => {
            let x = e(tr.b.convert_u_to_f(f, None, raw))?;
            tr.to_u(x)?
        }
        Num::Sscaled => {
            let s = sext(tr)?;
            let si = tr.to_i(s)?;
            let x = e(tr.b.convert_s_to_f(f, None, si))?;
            tr.to_u(x)?
        }
        Num::Uint => raw,
        Num::Sint => sext(tr)?,
        Num::Float => match bits_ {
            32 => raw,
            16 => {
                let v = tr.glsl1(tr.t.v2f, GLOp::UnpackHalf2x16, raw)?;
                let x = e(tr.b.composite_extract(f, None, v, [0]))?;
                tr.to_u(x)?
            }
            11 | 10 => {
                // Unsigned small float (5-bit exponent, 6/5-bit mantissa):
                // rebias into an f16 (same exponent bias) and convert.
                let mbits = bits_ - 5;
                let mmask = tr.u((1 << mbits) - 1);
                let man = e(tr.b.bitwise_and(u, None, raw, mmask))?;
                let ms = tr.u(mbits);
                let ex = e(tr.b.shift_right_logical(u, None, raw, ms))?;
                let sh = tr.u(10 - mbits);
                let man16 = e(tr.b.shift_left_logical(u, None, man, sh))?;
                let k10 = tr.u(10);
                let ex16 = e(tr.b.shift_left_logical(u, None, ex, k10))?;
                let h = e(tr.b.bitwise_or(u, None, ex16, man16))?;
                let v = tr.glsl1(tr.t.v2f, GLOp::UnpackHalf2x16, h)?;
                let x = e(tr.b.composite_extract(f, None, v, [0]))?;
                tr.to_u(x)?
            }
            _ => {
                tr.warn(format!("float buffer component of {bits_} bits unsupported"));
                raw
            }
        },
    })
}

fn srgb_to_linear(tr: &mut T, c: Word) -> R<Word> {
    let f = tr.t.f32;
    let bt = tr.t.bool_;
    let k = tr.f(0.04045);
    let lin = {
        let d = tr.f(12.92);
        e(tr.b.f_div(f, None, c, d))?
    };
    let pw = {
        let a = tr.f(0.055);
        let s = e(tr.b.f_add(f, None, c, a))?;
        let d = tr.f(1.055);
        let q = e(tr.b.f_div(f, None, s, d))?;
        let g = tr.f(2.4);
        tr.glsl2(f, GLOp::Pow, q, g)?
    };
    let small = e(tr.b.f_ord_less_than_equal(bt, None, c, k))?;
    e(tr.b.select(f, None, small, lin, pw))
}

/// Loads and converts a formatted element: returns 4 register values.
fn load_formatted(tr: &mut T, var: Word, addr: Word, fmt: u16) -> R<[Word; 4]> {
    let Some(fm) = format::gfx10(fmt).filter(|f| !f.block && f.components() > 0) else {
        if fmt != 0 {
            tr.warn(format!("buffer format {fmt} unsupported"));
        }
        // FORMAT_INVALID reads return zero.
        let z = tr.u(0);
        return Ok([z, z, z, z]);
    };
    let u = tr.t.u32;
    let size = fm.bytes();
    let ndw = size.div_ceil(4);
    let dws = if size == 8 || size == 16 {
        load_dwords(tr, var, addr, ndw, true)?
    } else {
        let mut dws = Vec::new();
        for k in 0..ndw {
            let off = tr.u(k * 4);
            let a = e(tr.b.i_add(u, None, addr, off))?;
            dws.push(load_bytes(tr, var, a, size.min(4))?);
        }
        dws
    };
    let one_f = tr.u(1.0f32.to_bits());
    let one_i = tr.u(1);
    let z = tr.u(0);
    let mut out = [z, z, z, if fm.num.is_float() { one_f } else { one_i }];
    let mut bo = 0u32;
    for (c, &bw) in fm.bits.iter().enumerate() {
        let bw = bw as u32;
        if bw == 0 {
            break;
        }
        let dw = dws[(bo / 32) as usize];
        let raw = if bw == 32 {
            dw
        } else {
            let o = tr.u(bo % 32);
            let w = tr.u(bw);
            e(tr.b.bit_field_u_extract(u, None, dw, o, w))?
        };
        out[c] = convert_component(tr, raw, bw, fm.num)?;
        bo += bw;
    }
    Ok(out)
}

/// Converts a register value to a raw component (`bits` wide).
fn pack_component(tr: &mut T, v: Word, bits_: u32, num: Num) -> R<Word> {
    let u = tr.t.u32;
    let f = tr.t.f32;
    let mask = tr.u(if bits_ >= 32 { u32::MAX } else { (1u32 << bits_) - 1 });
    let r = match num {
        Num::Unorm | Num::Srgb | Num::Snorm => {
            let x = tr.to_f(v)?;
            let (lo, k) = if num == Num::Snorm { (-1.0, ((1u64 << (bits_ - 1)) - 1) as f32) } else { (0.0, ((1u64 << bits_) - 1) as f32) };
            let lo = tr.f(lo);
            let one = tr.f(1.0);
            let c = tr.glsl3(f, GLOp::NClamp, x, lo, one)?;
            // A NaN converts to 0 on the hardware (the D3D rule); NClamp gives the lower
            // bound, which is that 0 for UNORM but -1 for SNORM.
            let c = if num == Num::Snorm {
                let n = super::alu::nan_bits(tr, x)?;
                let z = tr.f(0.0);
                e(tr.b.select(f, None, n, z, c))?
            } else {
                c
            };
            let kk = tr.f(k);
            let s = e(tr.b.f_mul(f, None, c, kk))?;
            let rn = tr.glsl1(f, GLOp::RoundEven, s)?;
            let i = e(tr.b.convert_f_to_s(tr.t.i32, None, rn))?;
            tr.to_u(i)?
        }
        Num::Uscaled => {
            let x = tr.to_f(v)?;
            e(tr.b.convert_f_to_u(u, None, x))?
        }
        Num::Sscaled => {
            let x = tr.to_f(v)?;
            let i = e(tr.b.convert_f_to_s(tr.t.i32, None, x))?;
            tr.to_u(i)?
        }
        Num::Uint | Num::Sint => v,
        Num::Float => match bits_ {
            32 => v,
            16 => {
                let x = tr.to_f(v)?;
                let z = tr.f(0.0);
                let vv = e(tr.b.composite_construct(tr.t.v2f, None, [x, z]))?;
                tr.glsl1(u, GLOp::PackHalf2x16, vv)?
            }
            _ => {
                tr.warn(format!("float buffer store component of {bits_} bits unsupported"));
                v
            }
        },
    };
    e(tr.b.bitwise_and(u, None, r, mask))
}

/// Packs `n` register values per format and stores the element.
fn store_formatted(tr: &mut T, var: Word, addr: Word, vals: &[Word], fmt: u16) -> R<()> {
    let Some(fm) = format::gfx10(fmt).filter(|f| !f.block && f.components() > 0) else {
        tr.warn(format!("formatted store of format {fmt} unsupported"));
        return Ok(());
    };
    let u = tr.t.u32;
    let size = fm.bytes();
    let ndw = size.div_ceil(4) as usize;
    let mut dws: Vec<Word> = (0..ndw).map(|_| tr.u(0)).collect();
    let mut bo = 0u32;
    for (c, &bw) in fm.bits.iter().enumerate() {
        let bw = bw as u32;
        if bw == 0 || c >= vals.len() {
            break;
        }
        let raw = pack_component(tr, vals[c], bw, fm.num)?;
        let sh = tr.u(bo % 32);
        let placed = e(tr.b.shift_left_logical(u, None, raw, sh))?;
        let k = (bo / 32) as usize;
        dws[k] = e(tr.b.bitwise_or(u, None, dws[k], placed))?;
        bo += bw;
    }
    if size >= 4 {
        store_dwords(tr, var, addr, &dws)
    } else {
        store_bytes(tr, var, addr, dws[0], size)
    }
}

/// `FORMAT_RT`: packs `vals` per the format the V# names when the program
/// runs (`rt_format_entry`'s table, as `load_formatted_rt` reads it: each
/// component converted by the format's class to its width, placed at its
/// offset) and stores the element in the size the table gives (1, 2, 4, 8,
/// 12 or 16 bytes), marking those bytes for resource `r`. A format outside
/// the table stores nothing, as FORMAT_INVALID does. Call inside the
/// store's active-lanes block.
fn store_formatted_rt(tr: &mut T, var: Word, addr: Word, vals: &[Word], r: usize) -> R<()> {
    let (Some(w3), Some(table)) = (tr.rt_word3, tr.fmt_table) else { return Err("a run-time format store without its V# or table".into()) };
    let (u, f, bt, i32t) = (tr.t.u32, tr.t.f32, tr.t.bool_, tr.t.i32);
    let c = |tr: &mut T, v: u32| tr.u(v);
    let (k12, k7f) = (c(tr, 12), c(tr, 0x7F));
    let id = e(tr.b.shift_right_logical(u, None, w3, k12))?;
    let id = e(tr.b.bitwise_and(u, None, id, k7f))?;
    let pt = tr.b.type_pointer(None, StorageClass::Private, u);
    let p = e(tr.b.access_chain(pt, None, table, [id]))?;
    let ent = e(tr.b.load(u, None, p, None, []))?;
    let (k63, k7, k24, k31, k1, k3, z) = (c(tr, 63), c(tr, 7), c(tr, 24), c(tr, 31), c(tr, 1), c(tr, 3), c(tr, 0));
    let class = e(tr.b.shift_right_logical(u, None, ent, k24))?;
    let class = e(tr.b.bitwise_and(u, None, class, k7))?;
    let valid = e(tr.b.shift_right_logical(u, None, ent, k31))?;
    let valid = e(tr.b.i_not_equal(bt, None, valid, z))?;
    let is = |tr: &mut T, k: u32| -> R<Word> {
        let kk = tr.u(k);
        e(tr.b.i_equal(bt, None, class, kk))
    };
    let cl: Vec<Word> = (0..7).map(|k| is(tr, k)).collect::<R<_>>()?;
    // Widths and offsets (bits), the element's bytes.
    let mut w = [z; 4];
    let mut o = [z; 4];
    for k in 0..4 {
        let sh = c(tr, 6 * k as u32);
        let x = e(tr.b.shift_right_logical(u, None, ent, sh))?;
        w[k] = e(tr.b.bitwise_and(u, None, x, k63))?;
        if k > 0 {
            o[k] = e(tr.b.i_add(u, None, o[k - 1], w[k - 1]))?;
        }
    }
    let bits_ = e(tr.b.i_add(u, None, o[3], w[3]))?;
    let bytes = e(tr.b.shift_right_logical(u, None, bits_, k3))?;
    let (f0, fm1, f1) = (tr.f(0.0), tr.f(-1.0), tr.f(1.0));
    let (k32, all, k16, k5) = (c(tr, 32), c(tr, u32::MAX), c(tr, 16), c(tr, 5));
    let kj: Vec<Word> = (0..4).map(|j| c(tr, j)).collect();
    let mut dws = [z; 4];
    for (k, &v) in vals.iter().enumerate().take(4) {
        let wk = w[k];
        // The component's mask: (1 << w) - 1, all bits at 32 (no shift by
        // 32), none at 0.
        let wide = e(tr.b.u_greater_than_equal(bt, None, wk, k32))?;
        let ws = e(tr.b.bitwise_and(u, None, wk, k31))?;
        let one_sh = e(tr.b.shift_left_logical(u, None, k1, ws))?;
        let m = e(tr.b.i_sub(u, None, one_sh, k1))?;
        let mask = e(tr.b.select(u, None, wide, all, m))?;
        // UNORM: clamp(x, 0, 1) * (2^w - 1); SNORM: clamp(x, -1, 1) *
        // (2^(w-1) - 1); rounded to even, as `pack_component`.
        let x = tr.to_f(v)?;
        let umax = e(tr.b.convert_u_to_f(f, None, mask))?;
        let wm1 = e(tr.b.i_sub(u, None, wk, k1))?;
        let wm1 = e(tr.b.bitwise_and(u, None, wm1, k31))?;
        let half = e(tr.b.shift_left_logical(u, None, k1, wm1))?;
        let half = e(tr.b.i_sub(u, None, half, k1))?;
        let smax = e(tr.b.convert_u_to_f(f, None, half))?;
        let un = tr.glsl3(f, GLOp::NClamp, x, f0, f1)?;
        let un = e(tr.b.f_mul(f, None, un, umax))?;
        let un = tr.glsl1(f, GLOp::RoundEven, un)?;
        let un = e(tr.b.convert_f_to_s(i32t, None, un))?;
        let un = tr.to_u(un)?;
        let sn = tr.glsl3(f, GLOp::NClamp, x, fm1, f1)?;
        // (A NaN to 0, not NClamp's -1: the hardware's rule, as `pack_component`.)
        let nx = super::alu::nan_bits(tr, x)?;
        let sn = e(tr.b.select(f, None, nx, f0, sn))?;
        let sn = e(tr.b.f_mul(f, None, sn, smax))?;
        let sn = tr.glsl1(f, GLOp::RoundEven, sn)?;
        let sn = e(tr.b.convert_f_to_s(i32t, None, sn))?;
        let sn = tr.to_u(sn)?;
        let us = e(tr.b.convert_f_to_u(u, None, x))?;
        let ss = e(tr.b.convert_f_to_s(i32t, None, x))?;
        let ss = tr.to_u(ss)?;
        let hv = e(tr.b.composite_construct(tr.t.v2f, None, [x, f0]))?;
        let h = tr.glsl1(u, GLOp::PackHalf2x16, hv)?;
        let is16 = e(tr.b.i_equal(bt, None, wk, k16))?;
        let fl = e(tr.b.select(u, None, is16, h, v))?;
        // By the class (0 UNORM, 1 SNORM, 2 USCALED, 3 SSCALED, 4 UINT, 5
        // SINT, 6 FLOAT).
        let mut raw = v;
        for (kc, val) in [(0usize, un), (1, sn), (2, us), (3, ss), (6, fl)] {
            raw = e(tr.b.select(u, None, cl[kc], val, raw))?;
        }
        let raw = e(tr.b.bitwise_and(u, None, raw, mask))?;
        // Placed at its offset in its dword (the table keeps components
        // inside one dword).
        let bit = e(tr.b.bitwise_and(u, None, o[k], k31))?;
        let di = e(tr.b.shift_right_logical(u, None, o[k], k5))?;
        let placed = e(tr.b.shift_left_logical(u, None, raw, bit))?;
        for (j, dw) in dws.iter_mut().enumerate() {
            let at = e(tr.b.i_equal(bt, None, di, kj[j]))?;
            let add = e(tr.b.select(u, None, at, placed, z))?;
            *dw = e(tr.b.bitwise_or(u, None, *dw, add))?;
        }
    }
    let two = tr.u(2);
    let d0 = e(tr.b.shift_right_logical(u, None, addr, two))?;
    for size in [1u32, 2, 4, 8, 12, 16] {
        let ks = tr.u(size);
        let at = e(tr.b.i_equal(bt, None, bytes, ks))?;
        let at = e(tr.b.logical_and(bt, None, valid, at))?;
        let dw = dws;
        tr.guarded(at, &mut |tr| {
            if size >= 4 {
                store_dwords(tr, var, addr, &dw[..(size / 4) as usize])?;
                mark_dwords(tr, r, d0, size / 4)
            } else {
                store_bytes(tr, var, addr, dw[0], size)?;
                mark_bytes(tr, r, addr, size)
            }
        })?;
    }
    Ok(())
}

/// Applies V# DST_SEL to formatted-load results.
/// `FORMAT_RT`: the `n` outputs of a formatted load of the format the V#
/// names when the program runs, decoded through `rt_format_entry`'s table:
/// each output takes the component its DST_SEL names (word 3 bits 0-11: 0
/// zero, 1 one, 4-7 component X-W, else its own), loads the dword holding
/// it and converts it by the format's class. A format outside the table
/// reads zero, as FORMAT_INVALID does.
fn load_formatted_rt(tr: &mut T, var: Word, addr: Word, n: usize) -> R<[Word; 4]> {
    let (Some(w3), Some(table)) = (tr.rt_word3, tr.fmt_table) else { return Err("a run-time format access without its V# or table".into()) };
    let (u, f32t, bt) = (tr.t.u32, tr.t.f32, tr.t.bool_);
    let c = |tr: &mut T, v: u32| tr.u(v);
    let (k12, k7f) = (c(tr, 12), c(tr, 0x7F));
    let id = e(tr.b.shift_right_logical(u, None, w3, k12))?;
    let id = e(tr.b.bitwise_and(u, None, id, k7f))?;
    let pt = tr.b.type_pointer(None, StorageClass::Private, u);
    let p = e(tr.b.access_chain(pt, None, table, [id]))?;
    let ent = e(tr.b.load(u, None, p, None, []))?;
    let (k63, k7, k24, k31, k5, k3, k4, z) = (c(tr, 63), c(tr, 7), c(tr, 24), c(tr, 31), c(tr, 5), c(tr, 3), c(tr, 4), c(tr, 0));
    let class = e(tr.b.shift_right_logical(u, None, ent, k24))?;
    let class = e(tr.b.bitwise_and(u, None, class, k7))?;
    let valid = e(tr.b.shift_right_logical(u, None, ent, k31))?;
    let valid = e(tr.b.i_not_equal(bt, None, valid, z))?;
    let is = |tr: &mut T, k: u32| -> R<Word> {
        let kk = tr.u(k);
        e(tr.b.i_equal(bt, None, class, kk))
    };
    let (one_f, one_i) = (c(tr, 1.0f32.to_bits()), c(tr, 1));
    // The class's traits: signed (SNORM, SSCALED, SINT), normalized (UNORM,
    // SNORM), integer (UINT, SINT), float.
    let cl: Vec<Word> = (0..7).map(|k| is(tr, k)).collect::<R<_>>()?;
    let any = |tr: &mut T, ks: &[usize]| -> R<Word> {
        let mut x = cl[ks[0]];
        for &k in &ks[1..] {
            x = e(tr.b.logical_or(bt, None, x, cl[k]))?;
        }
        Ok(x)
    };
    let signed = any(tr, &[1, 3, 5])?;
    let norm = any(tr, &[0, 1])?;
    let int_class = any(tr, &[4, 5])?;
    let float_class = cl[6];
    let one = e(tr.b.select(u, None, int_class, one_i, one_f))?;
    // Widths and their offsets.
    let mut w = [z; 4];
    let mut o = [z; 4];
    for k in 0..4 {
        let sh = c(tr, 6 * k as u32);
        let x = e(tr.b.shift_right_logical(u, None, ent, sh))?;
        w[k] = e(tr.b.bitwise_and(u, None, x, k63))?;
        if k > 0 {
            o[k] = e(tr.b.i_add(u, None, o[k - 1], w[k - 1]))?;
        }
    }
    let mut out = [z; 4];
    for (k, slot) in out.iter_mut().enumerate().take(n) {
        let sh = c(tr, 3 * k as u32);
        let sel = e(tr.b.shift_right_logical(u, None, w3, sh))?;
        let sel = e(tr.b.bitwise_and(u, None, sel, k7))?;
        // The source component: sel 4-7 names X-W, other values this one.
        let named = e(tr.b.u_greater_than_equal(bt, None, sel, k4))?;
        let m4 = e(tr.b.i_sub(u, None, sel, k4))?;
        let kk = c(tr, k as u32);
        let comp = e(tr.b.select(u, None, named, m4, kk))?;
        let (mut wc, mut oc) = (w[0], o[0]);
        for j in 1..4 {
            let kj = c(tr, j as u32);
            let at = e(tr.b.i_equal(bt, None, comp, kj))?;
            wc = e(tr.b.select(u, None, at, w[j], wc))?;
            oc = e(tr.b.select(u, None, at, o[j], oc))?;
        }
        // Its dword (the table keeps components inside one; the first
        // from the element's byte address, a 1- or 2-byte element in its
        // low bits) and bits.
        let di = e(tr.b.shift_right_logical(u, None, oc, k5))?;
        let four = c(tr, 4);
        let dbytes = e(tr.b.i_mul(u, None, di, four))?;
        let a = e(tr.b.i_add(u, None, addr, dbytes))?;
        let dw = load_bytes(tr, var, a, 4)?;
        let bit = e(tr.b.bitwise_and(u, None, oc, k31))?;
        let raw = e(tr.b.bit_field_u_extract(u, None, dw, bit, wc))?;
        // (Sign-extended in a u32: the base's type; ConvertSToF reads it
        // as signed.)
        let sxu = e(tr.b.bit_field_s_extract(u, None, dw, bit, wc))?;
        let xi = e(tr.b.select(u, None, signed, sxu, raw))?;
        let fu = e(tr.b.convert_u_to_f(f32t, None, raw))?;
        let fs = e(tr.b.convert_s_to_f(f32t, None, sxu))?;
        let xf = e(tr.b.select(f32t, None, signed, fs, fu))?;
        // Normalized: over 2^w - 1 (2^(w-1) - 1 signed; w 32 kept in
        // range), SNORM's -2^(w-1) clamped to -1.
        let k32 = c(tr, 32);
        let full = e(tr.b.u_greater_than_equal(bt, None, wc, k32))?;
        let one_u = c(tr, 1);
        let p2 = e(tr.b.shift_left_logical(u, None, one_u, wc))?;
        let maxu = e(tr.b.i_sub(u, None, p2, one_u))?;
        let all = c(tr, u32::MAX);
        let maxu = e(tr.b.select(u, None, full, all, maxu))?;
        let maxs = e(tr.b.shift_right_logical(u, None, maxu, one_u))?;
        let scale = e(tr.b.select(u, None, signed, maxs, maxu))?;
        let scale = e(tr.b.convert_u_to_f(f32t, None, scale))?;
        let nf = e(tr.b.f_div(f32t, None, xf, scale))?;
        let m1 = tr.f(-1.0);
        let nf = tr.glsl2(f32t, GLOp::FMax, nf, m1)?;
        // FLOAT: 16 or 32 bits (the table has no other width).
        let k16 = c(tr, 16);
        let half = e(tr.b.i_equal(bt, None, wc, k16))?;
        let h = tr.glsl1(tr.t.v2f, GLOp::UnpackHalf2x16, raw)?;
        let h = e(tr.b.composite_extract(f32t, None, h, [0]))?;
        let hu = tr.to_u(h)?;
        let flt = e(tr.b.select(u, None, half, hu, raw))?;
        let nf = tr.to_u(nf)?;
        let xf = tr.to_u(xf)?;
        let v = e(tr.b.select(u, None, norm, nf, xf))?;
        let v = e(tr.b.select(u, None, int_class, xi, v))?;
        let v = e(tr.b.select(u, None, float_class, flt, v))?;
        // An absent component: 0, the fourth one.
        let absent = e(tr.b.i_equal(bt, None, wc, z))?;
        let w_comp = e(tr.b.i_equal(bt, None, comp, k3))?;
        let dflt = e(tr.b.select(u, None, w_comp, one, z))?;
        let v = e(tr.b.select(u, None, absent, dflt, v))?;
        let v = e(tr.b.select(u, None, valid, v, z))?;
        // DST_SEL 0 and 1.
        let is1 = e(tr.b.i_equal(bt, None, sel, one_u))?;
        let v = e(tr.b.select(u, None, is1, one, v))?;
        let is0 = e(tr.b.i_equal(bt, None, sel, z))?;
        *slot = e(tr.b.select(u, None, is0, z, v))?;
    }
    Ok(out)
}

fn apply_dst_sel(tr: &mut T, v: [Word; 4], dst_sel: u16, float: bool) -> [Word; 4] {
    let one = if float { tr.u(1.0f32.to_bits()) } else { tr.u(1) };
    let z = tr.u(0);
    let mut out = v;
    for (c, o) in out.iter_mut().enumerate() {
        let sel = (dst_sel >> (c * 3)) & 7;
        *o = match sel {
            0 => z,
            1 => one,
            4..=7 => v[(sel - 4) as usize],
            _ => v[c],
        };
    }
    out
}

fn atomic_op(tr: &mut T, op: &str, p: Word, scope: spirv::Scope, v: Word, cmp: Option<Word>) -> R<Option<Word>> {
    let u = tr.t.u32;
    let scope = tr.u(scope as u32);
    let sem = tr.u(0);
    Ok(Some(match op {
        "swap" => e(tr.b.atomic_exchange(u, None, p, scope, sem, v))?,
        "cmpswap" => {
            let c = cmp.unwrap_or(v);
            e(tr.b.atomic_compare_exchange(u, None, p, scope, sem, sem, v, c))?
        }
        "add" => e(tr.b.atomic_i_add(u, None, p, scope, sem, v))?,
        "sub" => e(tr.b.atomic_i_sub(u, None, p, scope, sem, v))?,
        "smin" | "min_i32" => e(tr.b.atomic_s_min(u, None, p, scope, sem, v))?,
        "umin" | "min_u32" => e(tr.b.atomic_u_min(u, None, p, scope, sem, v))?,
        "smax" | "max_i32" => e(tr.b.atomic_s_max(u, None, p, scope, sem, v))?,
        "umax" | "max_u32" => e(tr.b.atomic_u_max(u, None, p, scope, sem, v))?,
        "and" => e(tr.b.atomic_and(u, None, p, scope, sem, v))?,
        "or" => e(tr.b.atomic_or(u, None, p, scope, sem, v))?,
        "xor" => e(tr.b.atomic_xor(u, None, p, scope, sem, v))?,
        _ => return Ok(None),
    }))
}

/// Byte address (relative to the binding) of a MUBUF/MTBUF access.
fn buffer_addr(tr: &mut T, ins: &Inst, r: usize) -> R<Word> {
    let u = tr.t.u32;
    let mut addr = tr.u(ins.buf_offset());
    let mut vreg = ins.buf_vaddr() as u32;
    let mut in_range = None;
    let rt = tr.rt_vsharp.filter(|v| v.r == r);
    if ins.buf_idxen() {
        let idx = tr.vreg(vreg)?;
        vreg += 1;
        let st = match rt {
            Some(v) => v.stride,
            None => tr.res_info(r, 1)?,
        };
        let m = e(tr.b.i_mul(u, None, idx, st))?;
        addr = e(tr.b.i_add(u, None, addr, m))?;
        // Structured buffers check the index before scaling it (an index at
        // or beyond NUM_RECORDS reads 0): titles store flags in the high
        // bits of indices, and the 32-bit product could wrap back into the
        // bound range.
        let limit = match rt {
            Some(v) => v.limit,
            None => tr.res_info(r, 3)?,
        };
        in_range = Some(e(tr.b.u_less_than(tr.t.bool_, None, idx, limit))?);
    }
    if ins.buf_offen() {
        let off = tr.vreg(vreg)?;
        addr = e(tr.b.i_add(u, None, addr, off))?;
    }
    let so = tr.operand(ins, ins.buf_soffset())?;
    addr = e(tr.b.i_add(u, None, addr, so))?;
    // A GPU-resolved resource's offsets are from its V#'s base.
    let bias = match rt {
        Some(v) => v.bias,
        None => tr.res_info(r, 0)?,
    };
    let addr = e(tr.b.i_add(u, None, addr, bias))?;
    match in_range {
        // Beyond every bound range: robust access reads 0, drops writes.
        Some(ok) => {
            let out = tr.u(0xFFFF_FFF0);
            e(tr.b.select(u, None, ok, addr, out))
        }
        None => Ok(addr),
    }
}

pub fn buffer(tr: &mut T, ins: &Inst) -> R<()> {
    let name = ins.name().unwrap_or("");
    let Some(&(r, _)) = tr.plan.inst_res.get(&ins.pc) else { return Err(format!("{:#x}: buffer op without resource", ins.pc)) };
    if bits(ins.w0, 16, 1) != 0 || bits(ins.w1, 23, 1) != 0 {
        tr.warn(format!("{:#x}: buffer lds/tfe unsupported", ins.pc));
    }
    if name.starts_with("buffer_gl") {
        return Ok(()); // cache invalidations
    }
    let var = tr.buffers[&r];
    if gpu_resolved(tr, r) {
        let v = rt_vsharp(tr, r, ins.buf_srsrc() as u32)?;
        check_rt_base(tr, v)?;
        tr.rt_vsharp = Some(v);
    }
    let addr = buffer_addr(tr, ins, r)?;
    let (vfmt, dst_sel) = match tr.spec.resources.get(r) {
        Some(ResSpec::Buffer { format, dst_sel }) => (*format as u16, *dst_sel),
        _ => (0, 0xFAC), // identity swizzle
    };
    let fmt = if ins.enc == Enc::Mtbuf { ins.mtbuf_format() as u16 } else { vfmt };
    // (`FORMAT_RT`: the V#'s word 3 names the format when the program runs.)
    if fmt == super::FORMAT_RT as u16 {
        tr.rt_word3 = Some(tr.sreg(ins.buf_srsrc() as u32 + 3)?);
    }
    let res = mem_access(tr, ins, name.trim_start_matches('t'), r, var, addr, fmt, dst_sel);
    tr.rt_vsharp = None;
    tr.rt_word3 = None;
    res
}

/// GLOBAL instructions with an SGPR base: the base is a bound resource.
pub fn global(tr: &mut T, ins: &Inst) -> R<()> {
    let name = ins.name().unwrap_or("");
    let Some(&(r, _)) = tr.plan.inst_res.get(&ins.pc) else {
        tr.warn(format!("{:#x}: {name} with a 64-bit VGPR address unsupported", ins.pc));
        return Ok(());
    };
    let u = tr.t.u32;
    let var = tr.buffers[&r];
    let off = tr.vreg(ins.flat_addr() as u32)?;
    let k = tr.u(ins.flat_offset() as u32);
    let a = e(tr.b.i_add(u, None, off, k))?;
    let bias = tr.res_info(r, 0)?;
    let addr = e(tr.b.i_add(u, None, a, bias))?;
    let name = name.replacen("global_", "buffer_", 1);
    mem_access(tr, ins, &name, r, var, addr, 0, 0xFAC)
}

/// `Spec::write_marks`: sets the bits of `n` dwords (1..=4, a constant)
/// from dword `d0` of resource `r`'s binding in the written-bit map, when
/// the resource is marked and the dwords lie inside its bound bytes. Run it
/// where the store happens (active lanes). (`Spec::mark_twice`: twice.)
fn mark_dwords(tr: &mut T, r: usize, d0: Word, n: u32) -> R<()> {
    mark_dwords_once(tr, r, d0, n)?;
    if tr.spec.mark_twice {
        mark_dwords_once(tr, r, d0, n)?;
    }
    Ok(())
}

fn mark_dwords_once(tr: &mut T, r: usize, d0: Word, n: u32) -> R<()> {
    let Some(marks) = tr.marks else { return Ok(()) };
    let (u, b) = (tr.t.u32, tr.t.bool_);
    let base = tr.addr_info(r, 3)?;
    let none = tr.u(u32::MAX);
    let marked = e(tr.b.i_not_equal(b, None, base, none))?;
    // The binding's dwords end at bias + bound bytes (rounded up: robust
    // access takes the last dword whole): d0 + n <= that, without wrapping.
    let bias = tr.res_info(r, 0)?;
    let bound = tr.res_info(r, 2)?;
    let k3 = tr.u(3);
    let end = e(tr.b.i_add(u, None, bound, bias))?;
    let end = e(tr.b.i_add(u, None, end, k3))?;
    let two = tr.u(2);
    let bdw = e(tr.b.shift_right_logical(u, None, end, two))?;
    let nn = tr.u(n);
    let fits = e(tr.b.u_greater_than_equal(b, None, bdw, nn))?;
    let lim = e(tr.b.i_sub(u, None, bdw, nn))?;
    let le = e(tr.b.u_less_than_equal(b, None, d0, lim))?;
    let inr = e(tr.b.logical_and(b, None, fits, le))?;
    let cond = e(tr.b.logical_and(b, None, marked, inr))?;
    tr.guarded(cond, &mut |tr| {
        let (five, k31, k32) = (tr.u(5), tr.u(31), tr.u(32));
        let w = e(tr.b.shift_right_logical(u, None, d0, five))?;
        let w = e(tr.b.i_add(u, None, base, w))?;
        let lo = e(tr.b.bitwise_and(u, None, d0, k31))?;
        let bits = tr.u((1u32 << n) - 1);
        let z = tr.u(0);
        // The bits in its word, and those past the word's end in the next.
        let m0 = e(tr.b.shift_left_logical(u, None, bits, lo))?;
        let m1 = if n > 1 {
            let end = e(tr.b.i_add(u, None, lo, nn))?;
            let spill = e(tr.b.u_greater_than(b, None, end, k32))?;
            let sh = e(tr.b.i_sub(u, None, k32, lo))?;
            let hi = e(tr.b.shift_right_logical(u, None, bits, sh))?;
            e(tr.b.select(u, None, spill, hi, z))?
        } else {
            z
        };
        // C5 (`Spec::marks_held`): no bits for words in held pages (the
        // spill's word is in the next page when `w` is its page's last).
        let (m0, m1) = if tr.spec.marks_held {
            let m0 = held_cleared(tr, marks, w, m0)?;
            let m1 = if n > 1 {
                let one = tr.u(1);
                let w1 = e(tr.b.i_add(u, None, w, one))?;
                held_cleared(tr, marks, w1, m1)?
            } else {
                m1
            };
            (m0, m1)
        } else {
            (m0, m1)
        };
        if tr.spec.marks_part {
            return mark_partitioned(tr, marks, w, m0, m1, n);
        }
        // Lanes whose words lie within four of the first active lane's
        // (consecutive stores: usually all) combine their bits: one lane
        // stores at most five words for them. The others store their own.
        tr.b.capability(spirv::Capability::GroupNonUniform);
        tr.b.capability(spirv::Capability::GroupNonUniformBallot);
        tr.b.capability(spirv::Capability::GroupNonUniformArithmetic);
        let sg = tr.u(spirv::Scope::Subgroup as u32);
        let first = e(tr.b.group_non_uniform_broadcast_first(u, None, sg, w))?;
        let dl = e(tr.b.i_sub(u, None, w, first))?;
        let k4 = tr.u(4);
        let near = e(tr.b.u_less_than(b, None, dl, k4))?;
        let (merge, near_b, far_b) = (tr.b.id(), tr.b.id(), tr.b.id());
        ok(tr.b.selection_merge(merge, spirv::SelectionControl::NONE))?;
        ok(tr.b.branch_conditional(near, near_b, far_b, []))?;
        e(tr.b.begin_block(Some(near_b)))?;
        let words = if n > 1 { 5 } else { 4 };
        let mut ors = Vec::with_capacity(words);
        for k in 0..words as u32 {
            let kk = tr.u(k);
            let at0 = e(tr.b.i_equal(b, None, dl, kk))?;
            let mut c = e(tr.b.select(u, None, at0, m0, z))?;
            if n > 1 && k > 0 {
                let km = tr.u(k - 1);
                let at1 = e(tr.b.i_equal(b, None, dl, km))?;
                let c1 = e(tr.b.select(u, None, at1, m1, z))?;
                c = e(tr.b.bitwise_or(u, None, c, c1))?;
            }
            ors.push(e(tr.b.group_non_uniform_bitwise_or(u, None, sg, spirv::GroupOperation::Reduce, c, None))?);
        }
        let el = e(tr.b.group_non_uniform_elect(b, None, sg))?;
        let once = tr.spec.touch_test;
        tr.guarded(el, &mut |tr| {
            let scope = tr.u(spirv::Scope::Device as u32);
            let sem = tr.u(0);
            for (k, &o) in ors.iter().enumerate() {
                let z = tr.u(0);
                let any = e(tr.b.i_not_equal(tr.t.bool_, None, o, z))?;
                tr.guarded(any, &mut |tr| {
                    let kk = tr.u(k as u32);
                    let wk = e(tr.b.i_add(u, None, first, kk))?;
                    let z = tr.u(0);
                    let p = e(tr.b.access_chain(tr.t.sb_u32, None, marks, [z, wk]))?;
                    let scope = tr.u(spirv::Scope::Device as u32);
                    let sem = tr.u(0);
                    e(tr.b.atomic_or(u, None, p, scope, sem, o))?;
                    if once {
                        return Ok(());
                    }
                    touch_page(tr, marks, wk)
                })?;
            }
            if once {
                // (`Spec::touch_test`: the words lie in the first word's page
                // or the next; each page with a set word touched once.)
                let (z, k31, k32) = (tr.u(0), tr.u(31), tr.u(32));
                let lo = e(tr.b.bitwise_and(u, None, first, k31))?;
                let (mut in_lo, mut in_hi) = (z, z);
                for (k, &o) in ors.iter().enumerate() {
                    let kk = tr.u(k as u32);
                    let at = e(tr.b.i_add(u, None, lo, kk))?;
                    let same = e(tr.b.u_less_than(tr.t.bool_, None, at, k32))?;
                    let a = e(tr.b.select(u, None, same, o, z))?;
                    let c = e(tr.b.select(u, None, same, z, o))?;
                    in_lo = e(tr.b.bitwise_or(u, None, in_lo, a))?;
                    in_hi = e(tr.b.bitwise_or(u, None, in_hi, c))?;
                }
                let any_lo = e(tr.b.i_not_equal(tr.t.bool_, None, in_lo, z))?;
                tr.guarded(any_lo, &mut |tr| touch_page(tr, marks, first))?;
                let any_hi = e(tr.b.i_not_equal(tr.t.bool_, None, in_hi, z))?;
                tr.guarded(any_hi, &mut |tr| {
                    let last = tr.u(ors.len() as u32 - 1);
                    let wl = e(tr.b.i_add(tr.t.u32, None, first, last))?;
                    touch_page(tr, marks, wl)
                })?;
            }
            let _ = (scope, sem);
            Ok(())
        })?;
        ok(tr.b.branch(merge))?;
        e(tr.b.begin_block(Some(far_b)))?;
        if tr.spec.marks_held {
            let any0 = e(tr.b.i_not_equal(b, None, m0, z))?;
            tr.guarded(any0, &mut |tr| {
                let (u, z) = (tr.t.u32, tr.u(0));
                let scope = tr.u(spirv::Scope::Device as u32);
                let sem = tr.u(0);
                let p0 = e(tr.b.access_chain(tr.t.sb_u32, None, marks, [z, w]))?;
                e(tr.b.atomic_or(u, None, p0, scope, sem, m0))?;
                touch_page(tr, marks, w)
            })?;
        } else {
            let scope = tr.u(spirv::Scope::Device as u32);
            let sem = tr.u(0);
            let p0 = e(tr.b.access_chain(tr.t.sb_u32, None, marks, [z, w]))?;
            e(tr.b.atomic_or(u, None, p0, scope, sem, m0))?;
            touch_page(tr, marks, w)?;
        }
        if n > 1 {
            let any = e(tr.b.i_not_equal(b, None, m1, z))?;
            tr.guarded(any, &mut |tr| {
                let one = tr.u(1);
                let w1 = e(tr.b.i_add(u, None, w, one))?;
                let z = tr.u(0);
                let p1 = e(tr.b.access_chain(tr.t.sb_u32, None, marks, [z, w1]))?;
                let scope = tr.u(spirv::Scope::Device as u32);
                let sem = tr.u(0);
                e(tr.b.atomic_or(u, None, p1, scope, sem, m1))?;
                if !tr.spec.touch_test {
                    return touch_page(tr, marks, w1);
                }
                // (`Spec::touch_test`: its page only when it is the next.)
                let k31 = tr.u(31);
                let lo = e(tr.b.bitwise_and(u, None, w, k31))?;
                let next = e(tr.b.i_equal(tr.t.bool_, None, lo, k31))?;
                tr.guarded(next, &mut |tr| touch_page(tr, marks, w1))
            })?;
        }
        ok(tr.b.branch(merge))?;
        e(tr.b.begin_block(Some(merge)))?;
        Ok(())
    })
}

/// C5 (`Spec::marks_held`): `bits` for map word `w`, or 0 when the word's
/// page is held (bit `w / 32` of the map's held region, MARK_HELD_AT). The
/// region changes only between batches (the runtime's copies), so a plain
/// load.
fn held_cleared(tr: &mut T, marks: Word, w: Word, bits: Word) -> R<Word> {
    use crate::kernels::MARK_HELD_AT;
    let (u, b) = (tr.t.u32, tr.t.bool_);
    let (five, k10, k31, one, z) = (tr.u(5), tr.u(10), tr.u(31), tr.u(1), tr.u(0));
    let hw = e(tr.b.shift_right_logical(u, None, w, k10))?;
    let at = tr.u(MARK_HELD_AT);
    let hw = e(tr.b.i_add(u, None, at, hw))?;
    let hp = e(tr.b.access_chain(tr.t.sb_u32, None, marks, [z, hw]))?;
    let hv = e(tr.b.load(u, None, hp, None, []))?;
    let page = e(tr.b.shift_right_logical(u, None, w, five))?;
    let pbit = e(tr.b.bitwise_and(u, None, page, k31))?;
    let hb = e(tr.b.shift_right_logical(u, None, hv, pbit))?;
    let hb = e(tr.b.bitwise_and(u, None, hb, one))?;
    let held = e(tr.b.i_not_equal(b, None, hb, z))?;
    e(tr.b.select(u, None, held, z, bits))
}

/// `Spec::marks_part`: the active lanes marking map word `w` (bits `m0`,
/// and `m1` in the next word for a store spilling past it; `n` dwords) form
/// a partition by `w`; its lowest lane ORs the partition's bits into the
/// word (and the next) and touches their pages as the other paths do.
fn mark_partitioned(tr: &mut T, marks: Word, w: Word, m0: Word, m1: Word, n: u32) -> R<()> {
    tr.b.capability(spirv::Capability::GroupNonUniform);
    tr.b.capability(spirv::Capability::GroupNonUniformBallot);
    tr.b.capability(spirv::Capability::GroupNonUniformArithmetic);
    tr.b.capability(spirv::Capability::GroupNonUniformPartitionedEXT);
    tr.b.extension("SPV_NV_shader_subgroup_partitioned");
    if tr.spec.part_twice {
        // (Ballast: the same reduction and OR once more, the page left
        // alone; the bits are set already, the map ends the same.)
        mark_partitioned_or(tr, marks, w, m0, m1, n)?;
    }
    mark_part_lanes(tr, marks, w, m0, m1, n)
}

/// `mark_partitioned` without the page touch (`Spec::part_twice`).
fn mark_partitioned_or(tr: &mut T, marks: Word, w: Word, m0: Word, m1: Word, n: u32) -> R<()> {
    let (u, b) = (tr.t.u32, tr.t.bool_);
    let sg = tr.u(spirv::Scope::Subgroup as u32);
    let part = e(tr.b.group_non_uniform_partition_ext(tr.t.v4u, None, w))?;
    let or0 = e(tr.b.group_non_uniform_bitwise_or(u, None, sg, spirv::GroupOperation::PartitionedReduceEXT, m0, Some(part)))?;
    let or1 = if n > 1 { Some(e(tr.b.group_non_uniform_bitwise_or(u, None, sg, spirv::GroupOperation::PartitionedReduceEXT, m1, Some(part)))?) } else { None };
    let lead = e(tr.b.group_non_uniform_ballot_find_lsb(u, None, sg, part))?;
    let me = tr.subgroup_lane()?;
    let is_lead = e(tr.b.i_equal(b, None, me, lead))?;
    tr.guarded(is_lead, &mut |tr| {
        let u = tr.t.u32;
        let z = tr.u(0);
        let scope = tr.u(spirv::Scope::Device as u32);
        let sem = tr.u(0);
        let p0 = e(tr.b.access_chain(tr.t.sb_u32, None, marks, [z, w]))?;
        e(tr.b.atomic_or(u, None, p0, scope, sem, or0))?;
        if let Some(o1) = or1 {
            let one = tr.u(1);
            let w1 = e(tr.b.i_add(u, None, w, one))?;
            let p1 = e(tr.b.access_chain(tr.t.sb_u32, None, marks, [z, w1]))?;
            e(tr.b.atomic_or(u, None, p1, scope, sem, o1))?;
        }
        Ok(())
    })
}

/// The partitioned reduction over the active lanes by map word `w`, its
/// lowest lane the leader (`mark_lead`).
fn mark_part_lanes(tr: &mut T, marks: Word, w: Word, m0: Word, m1: Word, n: u32) -> R<()> {
    let (u, b) = (tr.t.u32, tr.t.bool_);
    let sg = tr.u(spirv::Scope::Subgroup as u32);
    let part = e(tr.b.group_non_uniform_partition_ext(tr.t.v4u, None, w))?;
    let or0 = e(tr.b.group_non_uniform_bitwise_or(u, None, sg, spirv::GroupOperation::PartitionedReduceEXT, m0, Some(part)))?;
    let or1 = if n > 1 { Some(e(tr.b.group_non_uniform_bitwise_or(u, None, sg, spirv::GroupOperation::PartitionedReduceEXT, m1, Some(part)))?) } else { None };
    let lead = e(tr.b.group_non_uniform_ballot_find_lsb(u, None, sg, part))?;
    let me = tr.subgroup_lane()?;
    let is_lead = e(tr.b.i_equal(b, None, me, lead))?;
    tr.guarded(is_lead, &mut |tr| mark_lead(tr, marks, w, or0, or1))
}

/// The leader's part of a mark: the OR of the lanes' bits into map word
/// `w` (and the next for `or1`), and the pages touched.
fn mark_lead(tr: &mut T, marks: Word, w: Word, or0: Word, or1: Option<Word>) -> R<()> {
    let (u, b) = (tr.t.u32, tr.t.bool_);
    let z = tr.u(0);
    if tr.spec.marks_held {
        // (C5: a held page's word has no bits; nothing stored or touched.)
        let any0 = e(tr.b.i_not_equal(b, None, or0, z))?;
        tr.guarded(any0, &mut |tr| {
            let (u, z) = (tr.t.u32, tr.u(0));
            let scope = tr.u(spirv::Scope::Device as u32);
            let sem = tr.u(0);
            let p0 = e(tr.b.access_chain(tr.t.sb_u32, None, marks, [z, w]))?;
            e(tr.b.atomic_or(u, None, p0, scope, sem, or0))?;
            touch_page(tr, marks, w)
        })?;
    } else {
        let scope = tr.u(spirv::Scope::Device as u32);
        let sem = tr.u(0);
        let p0 = e(tr.b.access_chain(tr.t.sb_u32, None, marks, [z, w]))?;
        e(tr.b.atomic_or(u, None, p0, scope, sem, or0))?;
        touch_page(tr, marks, w)?;
    }
    if let Some(o1) = or1 {
        let any = e(tr.b.i_not_equal(b, None, o1, z))?;
        tr.guarded(any, &mut |tr| {
            let u = tr.t.u32;
            let one = tr.u(1);
            let w1 = e(tr.b.i_add(u, None, w, one))?;
            let z = tr.u(0);
            let p1 = e(tr.b.access_chain(tr.t.sb_u32, None, marks, [z, w1]))?;
            let scope = tr.u(spirv::Scope::Device as u32);
            let sem = tr.u(0);
            e(tr.b.atomic_or(u, None, p1, scope, sem, o1))?;
            if !tr.spec.touch_test {
                return touch_page(tr, marks, w1);
            }
            // (`Spec::touch_test`: its page only when it is the next.)
            let k31 = tr.u(31);
            let lo = e(tr.b.bitwise_and(u, None, w, k31))?;
            let next = e(tr.b.i_equal(tr.t.bool_, None, lo, k31))?;
            tr.guarded(next, &mut |tr| touch_page(tr, marks, w1))
        })?;
    }
    Ok(())
}

/// After marking map word `w`: its page's touched bit, and on the page's
/// first touch in the batch its index appended to the touched-page list
/// (`kernels::MARK_LIST_AT`; past MARK_LIST_CAP entries only counted: a
/// range gather takes the rest).
fn touch_page(tr: &mut T, marks: Word, w: Word) -> R<()> {
    use crate::kernels::MARK_PAGE_BITS_AT;
    if tr.spec.marks_untouched {
        return Ok(());
    }
    let (u, b) = (tr.t.u32, tr.t.bool_);
    let (five, k31, one, z) = (tr.u(5), tr.u(31), tr.u(1), tr.u(0));
    let page = e(tr.b.shift_right_logical(u, None, w, five))?;
    let pw = e(tr.b.shift_right_logical(u, None, page, five))?;
    let at = tr.u(MARK_PAGE_BITS_AT);
    let pw = e(tr.b.i_add(u, None, at, pw))?;
    let pbit = e(tr.b.bitwise_and(u, None, page, k31))?;
    let pb = e(tr.b.shift_left_logical(u, None, one, pbit))?;
    let pp = e(tr.b.access_chain(tr.t.sb_u32, None, marks, [z, pw]))?;
    if tr.spec.touch_test {
        // (`Spec::touch_test`: a bit found set was set in this batch, its
        // page listed by the store whose atomic found it clear; bits are
        // cleared only by the write-back, between barriers. A stale clear
        // read only costs the atomic. `Spec::touch_twice`, ballast: the
        // test twice, its loads volatile so neither is merged away.)
        let access = tr.spec.touch_twice.then_some(spirv::MemoryAccess::VOLATILE);
        for _ in 0..(1 + tr.spec.touch_twice as u32) {
            let cur = e(tr.b.load(u, None, pp, access, []))?;
            let set = e(tr.b.bitwise_and(u, None, cur, pb))?;
            let clear = e(tr.b.i_equal(b, None, set, z))?;
            tr.guarded(clear, &mut |tr| touch_set(tr, marks, pp, page, pb))?;
        }
        return Ok(());
    }
    touch_set(tr, marks, pp, page, pb)
}

/// `touch_page`'s atomic on the page-bits word `pp` setting bit `pb` of page
/// `page`, and the list append when it found the bit clear.
fn touch_set(tr: &mut T, marks: Word, pp: Word, page: Word, pb: Word) -> R<()> {
    use crate::kernels::{MARK_LIST_AT, MARK_LIST_CAP};
    let (u, b) = (tr.t.u32, tr.t.bool_);
    let z = tr.u(0);
    let scope = tr.u(spirv::Scope::Device as u32);
    let sem = tr.u(0);
    let old = e(tr.b.atomic_or(u, None, pp, scope, sem, pb))?;
    let had = e(tr.b.bitwise_and(u, None, old, pb))?;
    let first = e(tr.b.i_equal(b, None, had, z))?;
    tr.guarded(first, &mut |tr| {
        let (u, b) = (tr.t.u32, tr.t.bool_);
        let z = tr.u(0);
        let one = tr.u(1);
        let la = tr.u(MARK_LIST_AT);
        let pc = e(tr.b.access_chain(tr.t.sb_u32, None, marks, [z, la]))?;
        let scope = tr.u(spirv::Scope::Device as u32);
        let sem = tr.u(0);
        let idx = e(tr.b.atomic_i_add(u, None, pc, scope, sem, one))?;
        let capc = tr.u(MARK_LIST_CAP);
        let room = e(tr.b.u_less_than(b, None, idx, capc))?;
        tr.guarded(room, &mut |tr| {
            let u = tr.t.u32;
            let z = tr.u(0);
            let l0 = tr.u(MARK_LIST_AT + 4);
            let li = e(tr.b.i_add(u, None, l0, idx))?;
            let pl = e(tr.b.access_chain(tr.t.sb_u32, None, marks, [z, li]))?;
            ok(tr.b.store(pl, page, None, []))
        })
    })
}

/// `mark_dwords` for a store of `nbytes` (1 or 2) at byte `addr`: the one or
/// two dwords holding them.
fn mark_bytes(tr: &mut T, r: usize, addr: Word, nbytes: u32) -> R<()> {
    if tr.marks.is_none() {
        return Ok(());
    }
    let u = tr.t.u32;
    let two = tr.u(2);
    let d0 = e(tr.b.shift_right_logical(u, None, addr, two))?;
    if nbytes == 1 {
        return mark_dwords(tr, r, d0, 1);
    }
    // A short at byte 3 of a dword spans two.
    let k3 = tr.u(3);
    let sub = e(tr.b.bitwise_and(u, None, addr, k3))?;
    let spans = e(tr.b.i_equal(tr.t.bool_, None, sub, k3))?;
    let (merge, two_b, one_b) = (tr.b.id(), tr.b.id(), tr.b.id());
    ok(tr.b.selection_merge(merge, spirv::SelectionControl::NONE))?;
    ok(tr.b.branch_conditional(spans, two_b, one_b, []))?;
    e(tr.b.begin_block(Some(two_b)))?;
    mark_dwords(tr, r, d0, 2)?;
    ok(tr.b.branch(merge))?;
    e(tr.b.begin_block(Some(one_b)))?;
    mark_dwords(tr, r, d0, 1)?;
    ok(tr.b.branch(merge))?;
    e(tr.b.begin_block(Some(merge)))?;
    Ok(())
}

/// Shared MUBUF/GLOBAL semantics on a raw storage buffer at byte `addr`
/// (resource `r`).
#[allow(clippy::too_many_arguments)]
fn mem_access(tr: &mut T, ins: &Inst, name: &str, r: usize, var: Word, addr: Word, fmt: u16, dst_sel: u16) -> R<()> {
    let u = tr.t.u32;
    let vd = match ins.enc {
        Enc::Global | Enc::Flat | Enc::Scratch => {
            if name.contains("_store") || name.contains("_atomic") {
                ins.flat_data() as u32
            } else {
                ins.vdst() as u32
            }
        }
        _ => ins.vdst() as u32,
    };
    let rdst = ins.vdst() as u32;
    let two = tr.u(2);
    if let Some(comps) = name.strip_prefix("buffer_load_format_") {
        if comps.starts_with("d16") {
            tr.warn(format!("{:#x}: {name} unsupported", ins.pc));
            return Ok(());
        }
        let n = comps.len();
        let v = if fmt == super::FORMAT_RT as u16 {
            load_formatted_rt(tr, var, addr, n)?
        } else {
            let v = load_formatted(tr, var, addr, fmt)?;
            let float = format::gfx10(fmt).is_none_or(|f| f.num.is_float());
            apply_dst_sel(tr, v, dst_sel, float)
        };
        for (k, x) in v.iter().enumerate().take(n) {
            tr.set_vreg(vd + k as u32, *x)?;
        }
        return Ok(());
    }
    if let Some(comps) = name.strip_prefix("buffer_store_format_") {
        if comps.starts_with("d16") {
            tr.warn(format!("{:#x}: {name} unsupported", ins.pc));
            return Ok(());
        }
        let n = comps.len() as u32;
        let mut vals = Vec::new();
        for k in 0..n {
            vals.push(tr.store_vreg(vd + k)?);
        }
        let act = tr.active()?;
        // (`FORMAT_RT`: packed per the format the V# names when the program
        // runs, since 2026-11-18; before, the runtime kept stored buffers'
        // formats in the spec and every format a dispatch bound took a
        // translation of its own.)
        if fmt == super::FORMAT_RT as u16 {
            return tr.guarded(act, &mut |tr| store_formatted_rt(tr, var, addr, &vals, r));
        }
        let size = format::gfx10(fmt).filter(|f| !f.block && f.components() > 0).map(|f| f.bytes());
        return tr.guarded(act, &mut |tr| {
            store_formatted(tr, var, addr, &vals, fmt)?;
            match size {
                Some(sz) if sz >= 4 => {
                    let two = tr.u(2);
                    let d0 = e(tr.b.shift_right_logical(tr.t.u32, None, addr, two))?;
                    mark_dwords(tr, r, d0, sz.div_ceil(4))
                }
                Some(sz) => mark_bytes(tr, r, addr, sz),
                None => Ok(()),
            }
        });
    }
    if let Some(kind) = name.strip_prefix("buffer_load_") {
        match kind {
            "ubyte" | "sbyte" | "ushort" | "sshort" => {
                let nb = if kind.ends_with("byte") { 1 } else { 2 };
                let raw = load_bytes(tr, var, addr, nb)?;
                let r = if kind.starts_with('s') {
                    let z = tr.u(0);
                    let w = tr.u(nb * 8);
                    e(tr.b.bit_field_s_extract(u, None, raw, z, w))?
                } else {
                    raw
                };
                tr.set_vreg(rdst, r)?;
            }
            "dword" | "dwordx2" | "dwordx3" | "dwordx4" => {
                let n = match kind {
                    "dword" => 1,
                    "dwordx2" => 2,
                    "dwordx3" => 3,
                    _ => 4,
                };
                let vals = load_dwords(tr, var, addr, n, false)?;
                for (k, v) in vals.into_iter().enumerate() {
                    tr.set_vreg(rdst + k as u32, v)?;
                }
            }
            "ubyte_d16" | "ubyte_d16_hi" | "sbyte_d16" | "sbyte_d16_hi" | "short_d16" | "short_d16_hi" => {
                let nb = if kind.contains("byte") { 1 } else { 2 };
                let raw = load_bytes(tr, var, addr, nb)?;
                let val = if kind.starts_with('s') && nb == 1 {
                    let z = tr.u(0);
                    let w = tr.u(8);
                    let s = e(tr.b.bit_field_s_extract(u, None, raw, z, w))?;
                    let m = tr.u(0xFFFF);
                    e(tr.b.bitwise_and(u, None, s, m))?
                } else {
                    raw
                };
                let old = tr.vreg(rdst)?;
                let r = if kind.ends_with("_hi") {
                    let k16 = tr.u(16);
                    let hs = e(tr.b.shift_left_logical(u, None, val, k16))?;
                    let m = tr.u(0xFFFF);
                    let lo = e(tr.b.bitwise_and(u, None, old, m))?;
                    e(tr.b.bitwise_or(u, None, lo, hs))?
                } else {
                    let m = tr.u(0xFFFF_0000);
                    let hi = e(tr.b.bitwise_and(u, None, old, m))?;
                    e(tr.b.bitwise_or(u, None, hi, val))?
                };
                tr.set_vreg(rdst, r)?;
            }
            _ => tr.warn(format!("{:#x}: unsupported {name}", ins.pc)),
        }
        return Ok(());
    }
    if let Some(kind) = name.strip_prefix("buffer_store_") {
        let act = tr.active()?;
        let mut unsupported = false;
        tr.guarded(act, &mut |tr| {
            match kind {
                "dword" | "dwordx2" | "dwordx3" | "dwordx4" => {
                    let n = match kind {
                        "dword" => 1,
                        "dwordx2" => 2,
                        "dwordx3" => 3,
                        _ => 4,
                    };
                    let mut vals = Vec::new();
                    for k in 0..n {
                        vals.push(tr.store_vreg(vd + k)?);
                    }
                    // (The dword index before the store: the order of the
                    // modules the pipeline store holds since 2026-11-04.)
                    let two = tr.u(2);
                    let d0 = e(tr.b.shift_right_logical(u, None, addr, two))?;
                    store_dwords(tr, var, addr, &vals)?;
                    mark_dwords(tr, r, d0, n)?;
                }
                "byte" | "short" | "byte_d16_hi" | "short_d16_hi" => {
                    let nb = if kind.starts_with("byte") { 1 } else { 2 };
                    let mut v = tr.vreg(vd)?;
                    if kind.ends_with("_hi") {
                        let k16 = tr.u(16);
                        v = e(tr.b.shift_right_logical(u, None, v, k16))?;
                    }
                    store_bytes(tr, var, addr, v, nb)?;
                    mark_bytes(tr, r, addr, nb)?;
                }
                _ => unsupported = true,
            }
            Ok(())
        })?;
        if unsupported {
            tr.warn(format!("{:#x}: unsupported {name}", ins.pc));
        }
        return Ok(());
    }
    if let Some(op) = name.strip_prefix("buffer_atomic_") {
        if op.ends_with("_x2") || op.starts_with('f') || op == "csub" || op == "inc" || op == "dec" {
            tr.warn(format!("{:#x}: unsupported {name}", ins.pc));
            return Ok(());
        }
        let idx = e(tr.b.shift_right_logical(u, None, addr, two))?;
        let (p, inr) = dword_ref(tr, var, idx)?;
        let v = tr.vreg(vd)?;
        let cmp = if op == "cmpswap" { Some(tr.vreg(vd + 1)?) } else { None };
        let glc = match ins.enc {
            Enc::Global => bits(ins.w0, 16, 1) != 0,
            _ => bits(ins.w0, 14, 1) != 0,
        };
        let act = tr.active()?;
        // Address mode: an atomic outside the bound range does nothing.
        let act = match inr {
            Some(c) => e(tr.b.logical_and(tr.t.bool_, None, act, c))?,
            None => act,
        };
        let ret = if ins.enc == Enc::Global { rdst } else { vd };
        tr.guarded(act, &mut |tr| {
            if let Some(old) = atomic_op(tr, op, p, spirv::Scope::Device, v, cmp)? {
                if glc {
                    tr.set_vreg_raw(ret, old)?;
                }
            }
            mark_dwords(tr, r, idx, 1)
        })?;
        return Ok(());
    }
    tr.warn(format!("{:#x}: unsupported {name}", ins.pc));
    Ok(())
}

// ---------------------------------------------------------------------------
// Images

/// Coordinate count of an instruction DIM (MIMG dim field).
fn inst_dim(d: u32) -> ImageDim {
    match d {
        0 => ImageDim::D1,
        1 => ImageDim::D2,
        2 => ImageDim::D3,
        3 => ImageDim::Cube,
        4 => ImageDim::D1Array,
        5 => ImageDim::D2Array,
        6 => ImageDim::D2Msaa,
        _ => ImageDim::D2MsaaArray,
    }
}

fn coord_count(dim: ImageDim) -> u32 {
    match dim {
        ImageDim::D1 => 1,
        ImageDim::D2 | ImageDim::D1Array => 2,
        ImageDim::D3 | ImageDim::D3OrD2 | ImageDim::Cube | ImageDim::D2Array | ImageDim::D2Msaa | ImageDim::CubeArray => 3,
        ImageDim::D2MsaaArray => 4,
    }
}

fn is_array(dim: ImageDim) -> bool {
    matches!(dim, ImageDim::D1Array | ImageDim::D2Array | ImageDim::D2MsaaArray | ImageDim::CubeArray)
}

/// Address VGPR `k` of an image instruction (NSA or contiguous).
fn addr_vgpr(ins: &Inst, k: u32) -> u32 {
    let nsa = bits(ins.w0, 1, 2);
    if nsa == 0 || k == 0 { bits(ins.w1, 0, 8) + if nsa == 0 { k } else { 0 } } else { ins.nsa[(k - 1) as usize] as u32 }
}

struct Addr<'i> {
    ins: &'i Inst,
    next: u32,
}

impl Addr<'_> {
    fn take(&mut self, tr: &mut T) -> R<Word> {
        let r = addr_vgpr(self.ins, self.next);
        self.next += 1;
        tr.vreg(r)
    }

    /// The next integer address component of an A16 instruction: 16-bit
    /// values packed two per address VGPR (low half first), zero-extended.
    fn take_u16(&mut self, tr: &mut T) -> R<Word> {
        let u = tr.t.u32;
        let r = addr_vgpr(self.ins, self.next / 2);
        let hi = self.next % 2 == 1;
        self.next += 1;
        let v = tr.vreg(r)?;
        if hi {
            let k16 = tr.u(16);
            e(tr.b.shift_right_logical(u, None, v, k16))
        } else {
            let m = tr.u(0xFFFF);
            e(tr.b.bitwise_and(u, None, v, m))
        }
    }

    /// The next integer address component (A16 or 32-bit).
    fn take_int(&mut self, tr: &mut T, a16: bool) -> R<Word> {
        if a16 { self.take_u16(tr) } else { self.take(tr) }
    }
}

/// D16 store data: component `k` (of the enabled ones) is the 16-bit half
/// `k % 2` of data VGPR `vd + k / 2`, converted for the image class (f16 to
/// f32, zero- or sign-extended integers).
fn d16_component(tr: &mut T, vd: u32, k: u32, class: NumClass) -> R<Word> {
    let u = tr.t.u32;
    let raw = tr.store_vreg(vd + k / 2)?;
    let hi = k % 2 == 1;
    Ok(match class {
        NumClass::Float => super::alu::half_to_f(tr, raw, hi)?,
        NumClass::Uint => {
            if hi {
                let k16 = tr.u(16);
                e(tr.b.shift_right_logical(u, None, raw, k16))?
            } else {
                let m = tr.u(0xFFFF);
                e(tr.b.bitwise_and(u, None, raw, m))?
            }
        }
        NumClass::Sint => {
            let i = tr.to_i(raw)?;
            let off = tr.u(if hi { 16 } else { 0 });
            let w16 = tr.u(16);
            e(tr.b.bit_field_s_extract(tr.t.i32, None, i, off, w16))?
        }
    })
}

/// Integer coordinates (instruction layout) adapted to the host image type:
/// 1D images are 2D on the host (y = 0); array layers are dropped or added.
fn int_coord(tr: &mut T, idim: ImageDim, rdim: ImageDim, cs: Vec<Word>) -> R<Word> {
    let mut xy: Vec<Word> = Vec::new();
    let n_space = match idim {
        ImageDim::D1 | ImageDim::D1Array => 1,
        ImageDim::D3 | ImageDim::Cube => 3,
        _ => 2,
    };
    for c in cs.iter().take(n_space) {
        xy.push(*c);
    }
    let layer = if is_array(idim) { cs.get(n_space).copied() } else { None };
    if matches!(rdim, ImageDim::D1 | ImageDim::D1Array | ImageDim::D2 | ImageDim::D2Array | ImageDim::D2Msaa | ImageDim::D2MsaaArray) {
        while xy.len() < 2 {
            xy.push(tr.i(0));
        }
        xy.truncate(2);
    }
    // A 3D resource addressed by a 1D or 2D instruction (a 2D-array one in a
    // learned spec of cs 0xe890e8e2b31bf06f): its slice is the depth, else 0.
    if rdim == ImageDim::D3 && xy.len() < 3 {
        if let Some(l) = layer.filter(|_| xy.len() == 2) {
            xy.push(l);
        }
        while xy.len() < 3 {
            xy.push(tr.i(0));
        }
    }
    if is_array(rdim) {
        let l = match layer {
            Some(l) => l,
            None => tr.i(0),
        };
        xy.push(l);
    }
    let cty = match xy.len() {
        1 => tr.t.i32,
        2 => tr.t.v2i,
        3 => tr.t.v3i,
        _ => tr.t.v4i,
    };
    if xy.len() == 1 { Ok(xy[0]) } else { e(tr.b.composite_construct(cty, None, xy)) }
}

/// Sample 0 of a multisampled image at the texel normalized coordinates
/// (u, v[, layer]) fall in (clamped to the image), for sampling
/// instructions on multisampled resources.
fn ms_fetch(tr: &mut T, image: Word, coord: Word, arrayed: bool, res_ty: Word) -> R<Word> {
    let (f, i) = (tr.t.f32, tr.t.i32);
    let sty = if arrayed { tr.t.v3i } else { tr.t.v2i };
    let size = e(tr.b.image_query_size(sty, None, image))?;
    let (zero, one) = (tr.i(0), tr.i(1));
    let mut parts = Vec::new();
    for k in 0..2 {
        let n = e(tr.b.composite_extract(i, None, size, [k]))?;
        let nf = e(tr.b.convert_s_to_f(f, None, n))?;
        let c = e(tr.b.composite_extract(f, None, coord, [k]))?;
        let s = e(tr.b.f_mul(f, None, c, nf))?;
        let fl = tr.glsl1(f, GLOp::Floor, s)?;
        let v = e(tr.b.convert_f_to_s(i, None, fl))?;
        let hi = e(tr.b.i_sub(i, None, n, one))?;
        parts.push(tr.glsl3(i, GLOp::SClamp, v, zero, hi)?);
    }
    if arrayed {
        let c = e(tr.b.composite_extract(f, None, coord, [2]))?;
        let r = tr.glsl1(f, GLOp::RoundEven, c)?;
        parts.push(e(tr.b.convert_f_to_s(i, None, r))?);
    }
    let ic = e(tr.b.composite_construct(sty, None, parts))?;
    e(tr.b.image_fetch(res_ty, None, image, ic, Some(ImageOperands::SAMPLE), [Operand::IdRef(zero)]))
}

/// Cube coordinates (s+1.5, t+1.5, face) -> direction vector.
fn cube_dir(tr: &mut T, s: Word, t: Word, face: Word) -> R<Word> {
    let f = tr.t.f32;
    let bt = tr.t.bool_;
    let k15 = tr.f(1.5);
    let two = tr.f(2.0);
    let sc0 = e(tr.b.f_sub(f, None, s, k15))?;
    let tc0 = e(tr.b.f_sub(f, None, t, k15))?;
    let sc = e(tr.b.f_mul(f, None, sc0, two))?;
    let tc = e(tr.b.f_mul(f, None, tc0, two))?;
    let one = tr.f(1.0);
    let m1 = tr.f(-1.0);
    let nsc = e(tr.b.f_negate(f, None, sc))?;
    let ntc = e(tr.b.f_negate(f, None, tc))?;
    let fr = tr.glsl1(f, GLOp::RoundEven, face)?;
    // face: 0 +X, 1 -X, 2 +Y, 3 -Y, 4 +Z, 5 -Z
    let faces = [[one, ntc, nsc], [m1, ntc, sc], [sc, one, tc], [sc, m1, ntc], [sc, ntc, one], [nsc, ntc, m1]];
    let mut out = Vec::new();
    for c in 0..3 {
        let mut v = faces[5][c];
        for k in (0..5).rev() {
            let kf = tr.f(k as f32);
            let is = e(tr.b.f_ord_equal(bt, None, fr, kf))?;
            v = e(tr.b.select(f, None, is, faces[k][c], v))?;
        }
        out.push(v);
    }
    e(tr.b.composite_construct(tr.t.v3f, None, out))
}

pub fn image(tr: &mut T, ins: &Inst) -> R<()> {
    // An `ImageDim::D3OrD2` resource: the 3D variable when the T#'s TYPE
    // (word 3 bits 28-31) is 3D (10), else its 2D one.
    if let Some(&(ri, _)) = tr.plan.inst_res.get(&ins.pc) {
        if let Some(alt) = tr.image_alts.get(&ri).copied() {
            let main = tr.images[&ri];
            let (u, bt) = (tr.t.u32, tr.t.bool_);
            let w3 = tr.sreg(ins.mimg_srsrc() as u32 + 3)?;
            let k28 = tr.u(28);
            let ty = e(tr.b.shift_right_logical(u, None, w3, k28))?;
            let k10 = tr.u(10);
            let is3d = e(tr.b.i_equal(bt, None, ty, k10))?;
            let (merge, then_b, else_b) = (tr.b.id(), tr.b.id(), tr.b.id());
            ok(tr.b.selection_merge(merge, spirv::SelectionControl::NONE))?;
            ok(tr.b.branch_conditional(is3d, then_b, else_b, []))?;
            e(tr.b.begin_block(Some(then_b)))?;
            image_with(tr, ins)?;
            ok(tr.b.branch(merge))?;
            e(tr.b.begin_block(Some(else_b)))?;
            tr.images.insert(ri, alt);
            let r = image_with(tr, ins);
            tr.images.insert(ri, main);
            r?;
            ok(tr.b.branch(merge))?;
            e(tr.b.begin_block(Some(merge)))?;
            return Ok(());
        }
    }
    image_with(tr, ins)
}

/// An image instruction on the variable `tr.images` holds for its resource.
fn image_with(tr: &mut T, ins: &Inst) -> R<()> {
    let name = ins.name().unwrap_or("");
    let u = tr.t.u32;
    let f = tr.t.f32;
    let Some(&(ri, si)) = tr.plan.inst_res.get(&ins.pc) else { return Err(format!("{:#x}: image op without resource", ins.pc)) };
    let (var, img_ty, rdim, class, storage, depth, mips) = {
        let ir = &tr.images[&ri];
        (ir.var, ir.img_ty, ir.dim, ir.class, ir.storage, ir.depth, ir.mips)
    };
    // 16-bit addresses (A16) and data (D16): image_load / image_store only.
    let (a16, d16) = (ins.mimg_a16(), ins.mimg_d16());
    let packed_ok = matches!(name, "image_load" | "image_load_mip" | "image_store" | "image_store_mip");
    if (a16 || d16) && !packed_ok {
        tr.warn(format!("{:#x}: {name} with a16/d16 unsupported", ins.pc));
        return Ok(());
    }
    let idim = inst_dim(ins.mimg_dim());
    let vd = ins.vdst() as u32;
    let dmask = ins.mimg_dmask();
    let res_ty = match class {
        NumClass::Float => tr.t.v4f,
        NumClass::Sint => tr.t.v4i,
        NumClass::Uint => tr.t.v4u,
    };
    let comp_ty = match class {
        NumClass::Float => f,
        NumClass::Sint => tr.t.i32,
        NumClass::Uint => u,
    };
    let table = tr.image_tables.get(&ri).copied();
    let image = match table {
        Some(pc) => {
            // The T# came from a table at a run-time offset: index the array
            // of its entries (32 bytes each), per invocation.
            let idx = table_index(tr, ins, pc)?;
            let pt = tr.b.type_pointer(None, StorageClass::UniformConstant, img_ty);
            let p = e(tr.b.access_chain(pt, None, var, [idx]))?;
            tr.b.decorate(p, Decoration::NonUniform, []);
            let img = e(tr.b.load(img_ty, None, p, None, []))?;
            tr.b.decorate(img, Decoration::NonUniform, []);
            img
        }
        // A storage image bound per mip level: the first view (the T#'s
        // base level); image_store_mip and image_load_mip select their own.
        None if mips => {
            let z = tr.u(0);
            let pt = tr.b.type_pointer(None, StorageClass::UniformConstant, img_ty);
            let p = e(tr.b.access_chain(pt, None, var, [z]))?;
            e(tr.b.load(img_ty, None, p, None, []))?
        }
        None => e(tr.b.load(img_ty, None, var, None, []))?,
    };
    // Writes dmask-selected components of a vec4 result to consecutive VGPRs.
    // Gathers return four texels (four VGPRs) of the component dmask selects.
    let out_mask = if name.starts_with("image_gather4") { 0xF } else { dmask };
    let write_result = |tr: &mut T, res: Word| -> R<()> {
        let mut k = 0;
        for c in 0..4 {
            if out_mask & (1 << c) != 0 {
                let x = e(tr.b.composite_extract(comp_ty, None, res, [c]))?;
                let xu = tr.to_u(x)?;
                tr.set_vreg(vd + k, xu)?;
                k += 1;
            }
        }
        Ok(())
    };
    let mut ad = Addr { ins, next: 0 };
    match name {
        "image_load" | "image_load_mip" => {
            let mut cs = Vec::new();
            for _ in 0..coord_count(idim) {
                let v = ad.take_int(tr, a16)?;
                cs.push(tr.to_i(v)?);
            }
            let coord = int_coord(tr, idim, rdim, cs)?;
            let res = if storage {
                // image_load_mip reads the level its operand selects.
                let img = if mips && name == "image_load_mip" {
                    let lod = ad.take_int(tr, a16)?;
                    mip_view(tr, var, img_ty, lod)?
                } else {
                    image
                };
                e(tr.b.image_read(res_ty, None, img, coord, None, []))?
            } else if matches!(rdim, ImageDim::D2Msaa | ImageDim::D2MsaaArray) {
                let s = tr.i(0);
                e(tr.b.image_fetch(res_ty, None, image, coord, Some(ImageOperands::SAMPLE), [Operand::IdRef(s)]))?
            } else {
                let lod = if name == "image_load_mip" {
                    let v = ad.take_int(tr, a16)?;
                    tr.to_i(v)?
                } else {
                    tr.i(0)
                };
                e(tr.b.image_fetch(res_ty, None, image, coord, Some(ImageOperands::LOD), [Operand::IdRef(lod)]))?
            };
            if d16 {
                // Enabled components packed as 16-bit halves, two per VGPR.
                let mut halves = Vec::new();
                for c in 0..4 {
                    if dmask & (1 << c) == 0 {
                        continue;
                    }
                    let x = e(tr.b.composite_extract(comp_ty, None, res, [c]))?;
                    let h = match class {
                        NumClass::Float => super::alu::f_to_half(tr, x)?,
                        _ => {
                            let xu = tr.to_u(x)?;
                            let m = tr.u(0xFFFF);
                            e(tr.b.bitwise_and(u, None, xu, m))?
                        }
                    };
                    halves.push(h);
                }
                for (k, pair) in halves.chunks(2).enumerate() {
                    let v = if pair.len() == 2 {
                        let k16 = tr.u(16);
                        let hi = e(tr.b.shift_left_logical(u, None, pair[1], k16))?;
                        e(tr.b.bitwise_or(u, None, pair[0], hi))?
                    } else {
                        pair[0]
                    };
                    tr.set_vreg(vd + k as u32, v)?;
                }
                return Ok(());
            }
            write_result(tr, res)
        }
        "image_store" | "image_store_mip" => {
            if !storage {
                return Err(format!("{:#x}: image_store to a non-storage image", ins.pc));
            }
            let mut cs = Vec::new();
            for _ in 0..coord_count(idim) {
                let v = ad.take_int(tr, a16)?;
                cs.push(tr.to_i(v)?);
            }
            // image_store_mip writes the level its operand selects, counted
            // from the T#'s base level; past its last level nothing is
            // written.
            let mut target = image;
            let mut in_range = None;
            if name == "image_store_mip" {
                let lod = ad.take_int(tr, a16)?;
                if mips {
                    target = mip_view(tr, var, img_ty, lod)?;
                    in_range = Some(mip_in_range(tr, ins, lod)?);
                } else {
                    tr.warn(format!("{:#x}: image_store_mip writes the bound mip level", ins.pc));
                }
            }
            let coord = int_coord(tr, idim, rdim, cs)?;
            let mut comps = Vec::new();
            let mut k = 0;
            for c in 0..4 {
                if d16 {
                    let v = if dmask & (1 << c) != 0 {
                        let x = d16_component(tr, vd, k, class)?;
                        k += 1;
                        x
                    } else {
                        match class {
                            NumClass::Float => tr.f(0.0),
                            NumClass::Sint => tr.i(0),
                            NumClass::Uint => tr.u(0),
                        }
                    };
                    comps.push(v);
                    continue;
                }
                let v = if dmask & (1 << c) != 0 {
                    let x = tr.store_vreg(vd + k)?;
                    k += 1;
                    x
                } else {
                    tr.u(0)
                };
                comps.push(match class {
                    NumClass::Float => tr.to_f(v)?,
                    NumClass::Sint => tr.to_i(v)?,
                    NumClass::Uint => v,
                });
            }
            let texel = e(tr.b.composite_construct(res_ty, None, comps))?;
            let mut act = tr.active()?;
            if let Some(k) = in_range {
                act = e(tr.b.logical_and(tr.t.bool_, None, act, k))?;
            }
            tr.guarded(act, &mut |tr| ok(tr.b.image_write(target, coord, texel, None, [])))
        }
        "image_get_resinfo" => {
            // [width, height, depth or layers, levels]
            let lodv = ad.take(tr)?;
            let lod = tr.to_i(lodv)?;
            let (n, sty) = match rdim {
                ImageDim::D1 | ImageDim::D2 | ImageDim::Cube | ImageDim::D2Msaa => (2, tr.t.v2i),
                // Cube arrays report (w, h, layers) where layers count cubes.
                _ => (3, tr.t.v3i),
            };
            let ms = matches!(rdim, ImageDim::D2Msaa | ImageDim::D2MsaaArray);
            let size = if storage || ms { e(tr.b.image_query_size(sty, None, image))? } else { e(tr.b.image_query_size_lod(sty, None, image, lod))? };
            let levels = if storage || ms { tr.i(1) } else { e(tr.b.image_query_levels(tr.t.i32, None, image))? };
            // 1D images are 2D on the host: 1D arrays report (width, layers).
            let order: [u32; 3] = if rdim == ImageDim::D1Array { [0, 2, 1] } else { [0, 1, 2] };
            let mut vals = Vec::new();
            for (pos, c) in order.into_iter().enumerate() {
                let one_d = matches!(rdim, ImageDim::D1) && pos == 1;
                if c < n && !one_d {
                    vals.push(e(tr.b.composite_extract(tr.t.i32, None, size, [c]))?);
                } else {
                    vals.push(tr.i(1));
                }
            }
            vals.push(levels);
            let mut k = 0;
            for (c, v) in vals.iter().enumerate() {
                if dmask & (1 << c) != 0 {
                    let x = tr.to_u(*v)?;
                    tr.set_vreg(vd + k, x)?;
                    k += 1;
                }
            }
            Ok(())
        }
        n if n.starts_with("image_sample") || n.starts_with("image_gather4") || n == "image_get_lod" => sample(tr, ins, name, image, img_ty, si, idim, rdim, depth, res_ty, &write_result),
        _ => {
            tr.warn(format!("{:#x}: unsupported {name}", ins.pc));
            Ok(())
        }
    }
}

/// View `lod` (clamped to the views bound) of a storage image bound as one
/// view per mip level (`BindingKind::StorageImageMips`), per invocation.
fn mip_view(tr: &mut T, var: Word, img_ty: Word, lod: Word) -> R<Word> {
    let u = tr.t.u32;
    let max = tr.u(super::MIP_VIEWS - 1);
    let idx = tr.glsl2(u, GLOp::UMin, lod, max)?;
    tr.b.decorate(idx, Decoration::NonUniform, []);
    let pt = tr.b.type_pointer(None, StorageClass::UniformConstant, img_ty);
    let p = e(tr.b.access_chain(pt, None, var, [idx]))?;
    tr.b.decorate(p, Decoration::NonUniform, []);
    let img = e(tr.b.load(img_ty, None, p, None, []))?;
    tr.b.decorate(img, Decoration::NonUniform, []);
    Ok(img)
}

/// Whether mip `lod` (from BASE_LEVEL) of an image instruction's T# is at
/// most its LAST_LEVEL (T# dword 3 bits 12..15 and 16..19).
fn mip_in_range(tr: &mut T, ins: &Inst, lod: Word) -> R<Word> {
    let u = tr.t.u32;
    let w3 = tr.sreg(ins.mimg_srsrc() as u32 + 3)?;
    let (k12, k16, k4) = (tr.u(12), tr.u(16), tr.u(4));
    let base = e(tr.b.bit_field_u_extract(u, None, w3, k12, k4))?;
    let last = e(tr.b.bit_field_u_extract(u, None, w3, k16, k4))?;
    // Clamped first: base + lod must not wrap.
    let l = tr.glsl2(u, GLOp::UMin, lod, k16)?;
    let level = e(tr.b.i_add(u, None, base, l))?;
    e(tr.b.u_less_than_equal(tr.t.bool_, None, level, last))
}

/// Index of the image table entry the T# of table load `pc` selects (per
/// invocation; clamped to the bound entries).
fn table_index(tr: &mut T, ins: &Inst, pc: u32) -> R<Word> {
    let u = tr.t.u32;
    let ov = *tr.table_offsets.get(&pc).ok_or_else(|| format!("{:#x}: image table load {pc:#x} not translated", ins.pc))?;
    let off = e(tr.b.load(u, None, ov, None, []))?;
    let k5 = tr.u(5);
    let idx = e(tr.b.shift_right_logical(u, None, off, k5))?;
    let max = tr.u(super::IMAGE_TABLE_LEN - 1);
    let idx = tr.glsl2(u, GLOp::UMin, idx, max)?;
    tr.b.decorate(idx, Decoration::NonUniform, []);
    Ok(idx)
}

/// How a sampling instruction selects its level of detail (LOD statistics).
enum LodHow {
    /// From the implicit derivatives (fragment), plus the bias operand.
    Implicit(Option<Word>),
    /// The LOD operand.
    Explicit(Word),
    /// From the gradient operands (x and y per dimension).
    Grad(Vec<Word>, Vec<Word>),
    /// The view's first mip (LOD zero).
    Base,
}

/// The coordinate of a level-of-detail query: without the array layer.
fn lod_coord(tr: &mut T, coord: Word, rdim: ImageDim) -> R<Word> {
    let f = tr.t.f32;
    let n = match rdim {
        ImageDim::D1Array | ImageDim::D2Array => 2,
        ImageDim::CubeArray => 3,
        _ => return Ok(coord),
    };
    let mut c = Vec::new();
    for k in 0..n {
        c.push(e(tr.b.composite_extract(f, None, coord, [k]))?);
    }
    e(tr.b.composite_construct(if n == 2 { tr.t.v2f } else { tr.t.v3f }, None, c))
}

/// Level of detail from explicit gradients, relative to the view's first
/// mip: log2 of the longer footprint axis in texels.
fn grad_lod(tr: &mut T, image: Word, rdim: ImageDim, dx: &[Word], dy: &[Word]) -> R<Word> {
    let f = tr.t.f32;
    let sty = match rdim {
        ImageDim::D1Array | ImageDim::D2Array | ImageDim::D3 | ImageDim::CubeArray => tr.t.v3i,
        _ => tr.t.v2i,
    };
    let z = tr.i(0);
    let size = e(tr.b.image_query_size_lod(sty, None, image, z))?;
    let mut sx = tr.f(0.0);
    let mut sy = tr.f(0.0);
    for (j, (&gx, &gy)) in dx.iter().zip(dy.iter()).enumerate() {
        let s = e(tr.b.composite_extract(tr.t.i32, None, size, [j as u32]))?;
        let s = e(tr.b.convert_s_to_f(f, None, s))?;
        let a = e(tr.b.f_mul(f, None, gx, s))?;
        let a2 = e(tr.b.f_mul(f, None, a, a))?;
        sx = e(tr.b.f_add(f, None, sx, a2))?;
        let b = e(tr.b.f_mul(f, None, gy, s))?;
        let b2 = e(tr.b.f_mul(f, None, b, b))?;
        sy = e(tr.b.f_add(f, None, sy, b2))?;
    }
    let m = tr.glsl2(f, GLOp::FMax, sx, sy)?;
    let l = tr.glsl1(f, GLOp::Log2, m)?;
    let half = tr.f(0.5);
    e(tr.b.f_mul(f, None, l, half))
}

/// Hardware LOD statistics (`Spec::lod_stats`): when the sampled T# enables
/// counting ([`super::LOD_INFO_ENABLED`]), adds the active, non-helper lanes
/// to its counter bank: the finest mip they sample (the level of detail
/// relative to the texture's mip 0, floored, 0..14) and their number. The
/// lanes of a subgroup sampling one bank add one atomic pair.
#[allow(clippy::too_many_arguments)]
fn record_lod(tr: &mut T, ins: &Inst, image: Word, simg: Word, coord: Word, rdim: ImageDim, how: LodHow) -> R<()> {
    let Some(stats) = tr.lod_stats_var else { return Ok(()) };
    let Some(&(r, _)) = tr.plan.inst_res.get(&ins.pc) else { return Ok(()) };
    let u = tr.t.u32;
    let f = tr.t.f32;
    let bt = tr.t.bool_;
    let info = match tr.image_tables.get(&r).copied() {
        Some(pc) => {
            let idx = table_index(tr, ins, pc)?;
            let base = tr.u(super::lod_info_offset(&tr.plan) + super::IMAGE_TABLE_LEN * super::table_slot(&tr.plan, r));
            let i = e(tr.b.i_add(u, None, base, idx))?;
            let i = tr.sgpr_index(i)?;
            tr.sgpr_raw(i)?
        }
        None => tr.res_info(r, 0)?,
    };
    // In uniform control flow (implicit derivatives).
    let rel = match how {
        LodHow::Implicit(bias) => {
            let c = lod_coord(tr, coord, rdim)?;
            let q = e(tr.b.image_query_lod(tr.t.v2f, None, simg, c))?;
            let y = e(tr.b.composite_extract(f, None, q, [1]))?;
            match bias {
                Some(b) => {
                    let bf = tr.to_f(b)?;
                    e(tr.b.f_add(f, None, y, bf))?
                }
                None => y,
            }
        }
        LodHow::Explicit(l) => tr.to_f(l)?,
        LodHow::Grad(dx, dy) => grad_lod(tr, image, rdim, &dx, &dy)?,
        LodHow::Base => tr.f(0.0),
    };
    let k8 = tr.u(8);
    let k15 = tr.u(15);
    let sh = e(tr.b.shift_right_logical(u, None, info, k8))?;
    let base = e(tr.b.bitwise_and(u, None, sh, k15))?;
    let basef = e(tr.b.convert_u_to_f(f, None, base))?;
    let abs = e(tr.b.f_add(f, None, rel, basef))?;
    let z = tr.f(0.0);
    let hi = tr.f(14.0);
    let lo = tr.glsl2(f, GLOp::NMax, abs, z)?;
    let cl = tr.glsl2(f, GLOp::NMin, lo, hi)?;
    let fl = tr.glsl1(f, GLOp::Floor, cl)?;
    let mip = e(tr.b.convert_f_to_u(u, None, fl))?;
    let kff = tr.u(0xFF);
    let bank = e(tr.b.bitwise_and(u, None, info, kff))?;
    let ken = tr.u(super::LOD_INFO_ENABLED);
    let enabled = e(tr.b.u_greater_than_equal(bt, None, info, ken))?;
    let act = tr.active()?;
    let mut cond = e(tr.b.logical_and(bt, None, enabled, act))?;
    if tr.spec.stage == Stage::Fragment {
        let hv = match tr.helper_var {
            Some(v) => v,
            None => {
                let v = tr.input(bt, Some(BuiltIn::HelperInvocation), None);
                tr.helper_var = Some(v);
                v
            }
        };
        let h = e(tr.b.load(bt, None, hv, None, []))?;
        let nh = e(tr.b.logical_not(bt, None, h))?;
        cond = e(tr.b.logical_and(bt, None, cond, nh))?;
    }
    // `Spec::lod_lean`: this invocation's count slot, by its position.
    let slot = if tr.spec.lod_lean { Some(lod_count_slot(tr)?) } else { None };
    tr.guarded(cond, &mut |tr| {
        tr.b.capability(spirv::Capability::GroupNonUniform);
        tr.b.capability(spirv::Capability::GroupNonUniformBallot);
        tr.b.capability(spirv::Capability::GroupNonUniformArithmetic);
        let sg = tr.u(spirv::Scope::Subgroup as u32);
        let scope = tr.u(spirv::Scope::Device as u32);
        let sem = tr.u(0);
        let zero = tr.u(0);
        let k256 = tr.u(256);
        let first = e(tr.b.group_non_uniform_broadcast_first(u, None, sg, bank))?;
        let same = e(tr.b.i_equal(bt, None, bank, first))?;
        let count_at = e(tr.b.i_add(u, None, bank, k256))?;
        let count_at = match slot {
            Some(s) => {
                let off = e(tr.b.i_mul(u, None, s, k256))?;
                e(tr.b.i_add(u, None, count_at, off))?
            }
            None => count_at,
        };
        // The bank's finest mip; under `lod_lean` only when finer than the
        // one it holds.
        let store_min = |tr: &mut T, p: Word, m: Word| -> R<()> {
            if slot.is_none() {
                e(tr.b.atomic_u_min(u, None, p, scope, sem, m))?;
                return Ok(());
            }
            let held = e(tr.b.load(u, None, p, None, []))?;
            let finer = e(tr.b.u_greater_than(bt, None, held, m))?;
            tr.guarded(finer, &mut |tr| {
                e(tr.b.atomic_u_min(u, None, p, scope, sem, m))?;
                Ok(())
            })
        };
        // Lanes sampling the subgroup's first bank (usually all): one lane
        // adds for them.
        tr.guarded(same, &mut |tr| {
            let one = tr.u(1);
            let mm = e(tr.b.group_non_uniform_u_min(u, None, sg, spirv::GroupOperation::Reduce, mip, None))?;
            let n = e(tr.b.group_non_uniform_i_add(u, None, sg, spirv::GroupOperation::Reduce, one, None))?;
            let el = e(tr.b.group_non_uniform_elect(bt, None, sg))?;
            tr.guarded(el, &mut |tr| {
                let p = e(tr.b.access_chain(tr.t.sb_u32, None, stats, [zero, bank]))?;
                store_min(tr, p, mm)?;
                let p = e(tr.b.access_chain(tr.t.sb_u32, None, stats, [zero, count_at]))?;
                e(tr.b.atomic_i_add(u, None, p, scope, sem, n))?;
                Ok(())
            })
        })?;
        let other = e(tr.b.logical_not(bt, None, same))?;
        tr.guarded(other, &mut |tr| {
            let one = tr.u(1);
            let p = e(tr.b.access_chain(tr.t.sb_u32, None, stats, [zero, bank]))?;
            store_min(tr, p, mip)?;
            let p = e(tr.b.access_chain(tr.t.sb_u32, None, stats, [zero, count_at]))?;
            e(tr.b.atomic_i_add(u, None, p, scope, sem, one))?;
            Ok(())
        })
    })
}

/// The LOD-statistics count slot of this invocation (`Spec::lod_lean`):
/// from its position, 8-pixel tiles (pixel stage), its workgroup and wave
/// (compute), its vertex (vertex stage), so that neighbouring subgroups add
/// to different slots.
fn lod_count_slot(tr: &mut T) -> R<Word> {
    let u = tr.t.u32;
    let k3 = tr.u(3);
    let k5 = tr.u(5);
    let km = tr.u(super::LOD_COUNT_SLOTS - 1);
    let h = match tr.spec.stage {
        Stage::Fragment if tr.frag_coord.is_some() => {
            let fc = e(tr.b.load(tr.t.v4f, None, tr.frag_coord.unwrap(), None, []))?;
            let x = e(tr.b.composite_extract(tr.t.f32, None, fc, [0]))?;
            let y = e(tr.b.composite_extract(tr.t.f32, None, fc, [1]))?;
            let x = e(tr.b.convert_f_to_u(u, None, x))?;
            let y = e(tr.b.convert_f_to_u(u, None, y))?;
            let x = e(tr.b.shift_right_logical(u, None, x, k3))?;
            let y = e(tr.b.shift_right_logical(u, None, y, k3))?;
            let y = e(tr.b.shift_left_logical(u, None, y, k3))?;
            e(tr.b.bitwise_xor(u, None, x, y))?
        }
        Stage::Compute if tr.wg_id.is_some() => {
            let w = tr.wg_id.unwrap();
            let x = e(tr.b.composite_extract(u, None, w, [0]))?;
            let y = e(tr.b.composite_extract(u, None, w, [1]))?;
            let y = e(tr.b.shift_left_logical(u, None, y, k3))?;
            let h = e(tr.b.bitwise_xor(u, None, x, y))?;
            match tr.local_index {
                Some(li) => {
                    let wave = e(tr.b.shift_right_logical(u, None, li, k5))?;
                    e(tr.b.i_add(u, None, h, wave))?
                }
                None => h,
            }
        }
        _ => match tr.vertex_ids {
            Some((v, _)) => e(tr.b.shift_right_logical(u, None, v, k5))?,
            None => tr.u(0),
        },
    };
    e(tr.b.bitwise_and(u, None, h, km))
}

#[allow(clippy::too_many_arguments)]
fn sample(tr: &mut T, ins: &Inst, name: &str, image: Word, img_ty: Word, si: Option<usize>, idim: ImageDim, rdim: ImageDim, depth: bool, res_ty: Word, write_result: &dyn Fn(&mut T, Word) -> R<()>) -> R<()> {
    let f = tr.t.f32;
    let gather = name.starts_with("image_gather4");
    let get_lod = name == "image_get_lod";
    let mods: Vec<&str> = name.trim_start_matches("image_sample").trim_start_matches("image_gather4").split('_').filter(|s| !s.is_empty()).collect();
    let has = |m: &str| mods.contains(&m);
    let has_o = has("o");
    let has_c = has("c");
    let has_d = has("d") || has("cd");
    let has_l = has("l");
    let has_b = has("b");
    let has_cl = has("cl");
    let has_lz = has("lz");
    let Some(sidx) = si else { return Err(format!("{:#x}: sample without sampler", ins.pc)) };
    // Vulkan compares only on depth formats (on the RTX 4090 not R16_UNORM,
    // R16/R32_SFLOAT): comparisons on color views are emulated with a plain
    // sampler (`compare_texels`).
    let manual_cmp = has_c && !depth && !get_lod && matches!(rdim, ImageDim::D1 | ImageDim::D1Array | ImageDim::D2 | ImageDim::D2Array);
    if has_c && !depth && !manual_cmp && !get_lod {
        tr.warn(format!("{:#x}: depth comparison on a color {rdim:?} image (not emulated)", ins.pc));
    }
    let svar = tr.sampler_var(sidx, has_c && !manual_cmp && !get_lod)?;
    let sampler = e(tr.b.load(tr.sampler_ty, None, svar, None, []))?;
    let si_ty = tr.b.type_sampled_image(img_ty);
    let simg = e(tr.b.sampled_image(si_ty, None, image, sampler))?;
    if tr.plan.inst_res.get(&ins.pc).is_some_and(|&(r, _)| tr.image_tables.contains_key(&r)) {
        tr.b.decorate(simg, Decoration::NonUniform, []);
    }
    let mut ad = Addr { ins, next: 0 };
    let offset = if has_o { Some(ad.take(tr)?) } else { None };
    let bias = if has_b { Some(ad.take(tr)?) } else { None };
    let zcmp = if has_c { Some(ad.take(tr)?) } else { None };
    let gdims = match idim {
        ImageDim::D1 | ImageDim::D1Array => 1,
        ImageDim::D3 => 3,
        _ => 2,
    };
    let derivs = if has_d {
        let mut dx = Vec::new();
        let mut dy = Vec::new();
        for _ in 0..gdims {
            let v = ad.take(tr)?;
            dx.push(tr.to_f(v)?);
        }
        for _ in 0..gdims {
            let v = ad.take(tr)?;
            dy.push(tr.to_f(v)?);
        }
        Some((dx, dy))
    } else {
        None
    };
    let mut cs = Vec::new();
    for _ in 0..coord_count(idim) {
        let v = ad.take(tr)?;
        cs.push(tr.to_f(v)?);
    }
    let lod_or_clamp = if has_l || has_cl { Some(ad.take(tr)?) } else { None };
    let cube = matches!(rdim, ImageDim::Cube | ImageDim::CubeArray);
    let coord = if cube {
        if idim != ImageDim::Cube {
            tr.warn(format!("{:#x}: cube image sampled with dim {:?}", ins.pc, idim));
        }
        let face = cs.get(2).copied().unwrap_or_else(|| tr.f(0.0));
        if rdim == ImageDim::CubeArray {
            // Face coordinate = slice * 8 + face.
            let k8 = tr.f(8.0);
            let q = e(tr.b.f_div(f, None, face, k8))?;
            let slice = tr.glsl1(f, GLOp::Floor, q)?;
            let s8 = e(tr.b.f_mul(f, None, slice, k8))?;
            let fc = e(tr.b.f_sub(f, None, face, s8))?;
            let d = cube_dir(tr, cs[0], cs[1], fc)?;
            let x = e(tr.b.composite_extract(f, None, d, [0]))?;
            let y = e(tr.b.composite_extract(f, None, d, [1]))?;
            let z = e(tr.b.composite_extract(f, None, d, [2]))?;
            e(tr.b.composite_construct(tr.t.v4f, None, [x, y, z, slice]))?
        } else {
            cube_dir(tr, cs[0], cs[1], face)?
        }
    } else {
        let n_space = match idim {
            ImageDim::D1 | ImageDim::D1Array => 1,
            ImageDim::D3 | ImageDim::Cube => 3,
            _ => 2,
        };
        let mut v: Vec<Word> = cs.iter().take(n_space).copied().collect();
        let layer = if is_array(idim) { cs.get(n_space).copied() } else { None };
        if matches!(rdim, ImageDim::D1 | ImageDim::D1Array | ImageDim::D2 | ImageDim::D2Array) {
            if v.len() == 1 {
                // Row centre of the height-1 host image.
                v.push(tr.f(0.5));
            }
            v.truncate(2);
        }
        // A 3D resource sampled by a 1D or 2D instruction (ps 0x1008895d00):
        // the address register after them is its r coordinate, else 0.
        if rdim == ImageDim::D3 && v.len() < 3 {
            if let Some(l) = layer.filter(|_| v.len() == 2) {
                v.push(l);
            }
            while v.len() < 3 {
                v.push(tr.f(0.0));
            }
        }
        if is_array(rdim) {
            let l = match layer {
                Some(l) => l,
                None => tr.f(0.0),
            };
            v.push(l);
        }
        match v.len() {
            1 => v[0],
            2 => e(tr.b.composite_construct(tr.t.v2f, None, v))?,
            3 => e(tr.b.composite_construct(tr.t.v3f, None, v))?,
            _ => e(tr.b.composite_construct(tr.t.v4f, None, v))?,
        }
    };
    if get_lod {
        if tr.spec.stage != Stage::Fragment {
            tr.warn(format!("{:#x}: image_get_lod outside fragment shaders returns 0", ins.pc));
            let z = tr.f(0.0);
            let r = e(tr.b.composite_construct(tr.t.v4f, None, [z, z, z, z]))?;
            return write_result(tr, r);
        }
        let l = e(tr.b.image_query_lod(tr.t.v2f, None, simg, coord))?;
        let x = e(tr.b.composite_extract(f, None, l, [0]))?;
        let y = e(tr.b.composite_extract(f, None, l, [1]))?;
        let z = tr.f(0.0);
        let r = e(tr.b.composite_construct(tr.t.v4f, None, [x, y, z, z]))?;
        return write_result(tr, r);
    }
    let implicit_ok = tr.spec.stage == Stage::Fragment;
    let mut operands: Vec<Operand> = Vec::new();
    let mut mask = ImageOperands::NONE;
    let explicit;
    if let (Some((dx, dy)), false) = (&derivs, cube) {
        let (vty, dx, dy) = if gdims == 1 {
            // 1D images are 2D on the host: zero y gradient.
            let z = tr.f(0.0);
            (tr.t.v2f, vec![dx[0], z], vec![dy[0], z])
        } else {
            (if gdims == 2 { tr.t.v2f } else { tr.t.v3f }, dx.clone(), dy.clone())
        };
        let gx = e(tr.b.composite_construct(vty, None, dx))?;
        let gy = e(tr.b.composite_construct(vty, None, dy))?;
        mask |= ImageOperands::GRAD;
        operands.push(Operand::IdRef(gx));
        operands.push(Operand::IdRef(gy));
        explicit = true;
    } else if has_l || has_lz || !implicit_ok || (derivs.is_some() && cube) {
        if derivs.is_some() {
            tr.warn(format!("{:#x}: cube derivatives replaced by lod 0", ins.pc));
        }
        let lod = if has_l {
            let v = lod_or_clamp.unwrap();
            tr.to_f(v)?
        } else {
            tr.f(0.0)
        };
        mask |= ImageOperands::LOD;
        operands.push(Operand::IdRef(lod));
        explicit = true;
    } else {
        if let Some(bv) = bias {
            let bf = tr.to_f(bv)?;
            mask |= ImageOperands::BIAS;
            operands.insert(0, Operand::IdRef(bf));
        }
        explicit = false;
    }
    let mut coord = coord;
    if let (Some(ov), false) = (offset, gather) {
        if matches!(rdim, ImageDim::D2 | ImageDim::D2Array) {
            // Vulkan permits the Offset operand only on gathers: shift the
            // coordinates by offset / level size instead.
            let u = tr.t.u32;
            let z = tr.u(0);
            let six = tr.u(6);
            let k8 = tr.u(8);
            let x = e(tr.b.bit_field_s_extract(u, None, ov, z, six))?;
            let y = e(tr.b.bit_field_s_extract(u, None, ov, k8, six))?;
            let xi = tr.to_i(x)?;
            let yi = tr.to_i(y)?;
            let xf = e(tr.b.convert_s_to_f(f, None, xi))?;
            let yf = e(tr.b.convert_s_to_f(f, None, yi))?;
            let level = if has_l {
                let lv = tr.to_f(lod_or_clamp.unwrap())?;
                let r = tr.glsl1(f, GLOp::RoundEven, lv)?;
                e(tr.b.convert_f_to_s(tr.t.i32, None, r))?
            } else if !has_lz && implicit_ok && derivs.is_none() {
                let q = e(tr.b.image_query_lod(tr.t.v2f, None, simg, coord))?;
                let lv = e(tr.b.composite_extract(f, None, q, [0]))?;
                let r = tr.glsl1(f, GLOp::RoundEven, lv)?;
                e(tr.b.convert_f_to_s(tr.t.i32, None, r))?
            } else {
                tr.i(0)
            };
            let sty = if rdim == ImageDim::D2Array { tr.t.v3i } else { tr.t.v2i };
            let size = e(tr.b.image_query_size_lod(sty, None, image, level))?;
            let w = e(tr.b.composite_extract(tr.t.i32, None, size, [0]))?;
            let h = e(tr.b.composite_extract(tr.t.i32, None, size, [1]))?;
            let wf = e(tr.b.convert_s_to_f(f, None, w))?;
            let hf = e(tr.b.convert_s_to_f(f, None, h))?;
            let dx = e(tr.b.f_div(f, None, xf, wf))?;
            let dy = e(tr.b.f_div(f, None, yf, hf))?;
            let cx = e(tr.b.composite_extract(f, None, coord, [0]))?;
            let cy = e(tr.b.composite_extract(f, None, coord, [1]))?;
            let nx = e(tr.b.f_add(f, None, cx, dx))?;
            let ny = e(tr.b.f_add(f, None, cy, dy))?;
            let cty = if rdim == ImageDim::D2Array { tr.t.v3f } else { tr.t.v2f };
            coord = e(tr.b.composite_insert(cty, None, nx, coord, [0]))?;
            coord = e(tr.b.composite_insert(cty, None, ny, coord, [1]))?;
        } else {
            tr.warn(format!("{:#x}: sample offset on non-2D image ignored", ins.pc));
        }
    }
    if let (Some(ov), true) = (offset, gather) {
        if matches!(rdim, ImageDim::D2 | ImageDim::D2Array) {
            let u = tr.t.u32;
            let z = tr.u(0);
            let six = tr.u(6);
            let k8 = tr.u(8);
            let x = e(tr.b.bit_field_s_extract(u, None, ov, z, six))?;
            let y = e(tr.b.bit_field_s_extract(u, None, ov, k8, six))?;
            let xi = tr.to_i(x)?;
            let yi = tr.to_i(y)?;
            let off = e(tr.b.composite_construct(tr.t.v2i, None, [xi, yi]))?;
            mask |= ImageOperands::OFFSET;
            operands.push(Operand::IdRef(off));
        } else {
            tr.warn(format!("{:#x}: sample offset on non-2D image ignored", ins.pc));
        }
    }
    if has_cl && !has_l {
        tr.warn(format!("{:#x}: min-lod clamp ignored", ins.pc));
    }
    let m = if mask == ImageOperands::NONE { None } else { Some(mask) };
    let res = if let (true, Some(zc)) = (manual_cmp, zcmp) {
        let zf = tr.to_f(zc)?;
        // A gather's offset operand (samples had theirs folded into coord).
        let goff = if gather && mask.contains(ImageOperands::OFFSET) { operands.last().cloned() } else { None };
        let arrayed = matches!(rdim, ImageDim::D1Array | ImageDim::D2Array);
        compare_texels(tr, ins, image, simg, coord, zf, arrayed, gather, goff)?
    } else if gather {
        let comp = tr.u(match ins.mimg_dmask() {
            2 => 1,
            4 => 2,
            8 => 3,
            _ => 0,
        });
        // Gather has no LOD/gradient operands (it samples level 0).
        let (m, operands) = if explicit {
            let keep: Vec<Operand> = if mask.contains(ImageOperands::OFFSET) { vec![operands.last().unwrap().clone()] } else { vec![] };
            (if keep.is_empty() { None } else { Some(ImageOperands::OFFSET) }, keep)
        } else {
            (m, operands)
        };
        if let Some(zc) = zcmp {
            let zf = tr.to_f(zc)?;
            e(tr.b.image_dref_gather(res_ty, None, simg, coord, zf, m, operands))?
        } else {
            e(tr.b.image_gather(res_ty, None, simg, coord, comp, m, operands))?
        }
    } else if let Some(zc) = zcmp {
        let zf = tr.to_f(zc)?;
        let r = if explicit { e(tr.b.image_sample_dref_explicit_lod(f, None, simg, coord, zf, mask, operands))? } else { e(tr.b.image_sample_dref_implicit_lod(f, None, simg, coord, zf, m, operands))? };
        let z = tr.f(0.0);
        let one = tr.f(1.0);
        e(tr.b.composite_construct(tr.t.v4f, None, [r, z, z, one]))?
    } else if matches!(rdim, ImageDim::D2Msaa | ImageDim::D2MsaaArray) {
        // A sample of a multisampled resource (learned specs of four programs,
        // e.g. cs 0xffafd65f8762677f): SPIR-V samples none, so sample 0 of
        // the texel the coordinates fall in is fetched.
        ms_fetch(tr, image, coord, rdim == ImageDim::D2MsaaArray, res_ty)?
    } else if explicit {
        e(tr.b.image_sample_explicit_lod(res_ty, None, simg, coord, mask, operands))?
    } else {
        e(tr.b.image_sample_implicit_lod(res_ty, None, simg, coord, m, operands))?
    };
    // After the sample (its sampled image is consumed in this block; the
    // statistics open blocks of their own).
    if tr.lod_stats_var.is_some() && !gather && !has_c {
        let how = match (&derivs, cube) {
            (Some((dx, dy)), false) => LodHow::Grad(dx.clone(), dy.clone()),
            _ if has_l => LodHow::Explicit(lod_or_clamp.unwrap()),
            _ if explicit => LodHow::Base,
            _ => LodHow::Implicit(bias),
        };
        record_lod(tr, ins, image, simg, coord, rdim, how)?;
    }
    write_result(tr, res)
}

/// Depth-comparison sampling of a color image (emulated: Vulkan defines
/// comparison only for depth formats): gathers the 2x2 footprint with a plain
/// sampler, compares each texel with the reference by the S# DEPTH_COMPARE_FUNC
/// (word0[14:12], `ref func texel` as the hardware and Vulkan define it) and,
/// for samples, filters the four results bilinearly (LINEAR mag filter,
/// word2[20]; PCF as the hardware does) or takes the nearest one. Gathers
/// return the four comparisons. The result is (r, 0, 0, 1) for samples.
#[allow(clippy::too_many_arguments)]
fn compare_texels(tr: &mut T, ins: &Inst, image: Word, simg: Word, coord: Word, zref: Word, arrayed: bool, gather: bool, goff: Option<Operand>) -> R<Word> {
    let f = tr.t.f32;
    let u = tr.t.u32;
    let bt = tr.t.bool_;
    let comp0 = tr.u(0);
    let (gm, gops) = match goff {
        Some(o) => (Some(ImageOperands::OFFSET), vec![o]),
        None => (None, vec![]),
    };
    let g = e(tr.b.image_gather(tr.t.v4f, None, simg, coord, comp0, gm, gops))?;
    let sb = ins.mimg_ssamp() as u32;
    let w0 = tr.sreg(sb)?;
    let w2 = tr.sreg(sb + 2)?;
    let k12 = tr.u(12);
    let k7 = tr.u(7);
    let func = e(tr.b.shift_right_logical(u, None, w0, k12))?;
    let func = e(tr.b.bitwise_and(u, None, func, k7))?;
    let one = tr.f(1.0);
    let zero = tr.f(0.0);
    let t_true = tr.b.constant_true(bt);
    let t_false = tr.b.constant_false(bt);
    let mut c = [0; 4];
    for (k, ck) in c.iter_mut().enumerate() {
        let t = e(tr.b.composite_extract(f, None, g, [k as u32]))?;
        let lt = e(tr.b.f_ord_less_than(bt, None, zref, t))?;
        let eq = e(tr.b.f_ord_equal(bt, None, zref, t))?;
        let gt = e(tr.b.f_ord_greater_than(bt, None, zref, t))?;
        let le = e(tr.b.logical_or(bt, None, lt, eq))?;
        let ge = e(tr.b.logical_or(bt, None, gt, eq))?;
        let ne = e(tr.b.f_unord_not_equal(bt, None, zref, t))?;
        let mut r = t_false;
        for (v, cond) in [(1u32, lt), (2, eq), (3, le), (4, gt), (5, ne), (6, ge), (7, t_true)] {
            let kv = tr.u(v);
            let is = e(tr.b.i_equal(bt, None, func, kv))?;
            r = e(tr.b.select(bt, None, is, cond, r))?;
        }
        *ck = e(tr.b.select(f, None, r, one, zero))?;
    }
    if gather {
        return Ok(e(tr.b.composite_construct(tr.t.v4f, None, c))?);
    }
    // Weights of the footprint: gather order x = (i0,j1), y = (i1,j1),
    // z = (i1,j0), w = (i0,j0).
    let lod0 = tr.i(0);
    let szty = if arrayed { tr.t.v3i } else { tr.t.v2i };
    let size = e(tr.b.image_query_size_lod(szty, None, image, lod0))?;
    let half = tr.f(0.5);
    let mut fr = [0; 2];
    for (k, fk) in fr.iter_mut().enumerate() {
        let n = e(tr.b.composite_extract(tr.t.i32, None, size, [k as u32]))?;
        let nf = e(tr.b.convert_s_to_f(f, None, n))?;
        let x = e(tr.b.composite_extract(f, None, coord, [k as u32]))?;
        let tx = e(tr.b.f_mul(f, None, x, nf))?;
        let tx = e(tr.b.f_sub(f, None, tx, half))?;
        *fk = tr.glsl1(f, GLOp::Fract, tx)?;
    }
    let bottom = tr.glsl3(f, GLOp::FMix, c[3], c[2], fr[0])?;
    let top = tr.glsl3(f, GLOp::FMix, c[0], c[1], fr[0])?;
    let bil = tr.glsl3(f, GLOp::FMix, bottom, top, fr[1])?;
    let ihi = e(tr.b.f_ord_greater_than_equal(bt, None, fr[0], half))?;
    let jhi = e(tr.b.f_ord_greater_than_equal(bt, None, fr[1], half))?;
    let nb = e(tr.b.select(f, None, ihi, c[2], c[3]))?;
    let nt = e(tr.b.select(f, None, ihi, c[1], c[0]))?;
    let near = e(tr.b.select(f, None, jhi, nt, nb))?;
    let k20 = tr.u(20);
    let k1 = tr.u(1);
    let mag = e(tr.b.shift_right_logical(u, None, w2, k20))?;
    let mag = e(tr.b.bitwise_and(u, None, mag, k1))?;
    let k0 = tr.u(0);
    let lin = e(tr.b.i_not_equal(bt, None, mag, k0))?;
    let r = e(tr.b.select(f, None, lin, bil, near))?;
    let z = tr.f(0.0);
    let one = tr.f(1.0);
    Ok(e(tr.b.composite_construct(tr.t.v4f, None, [r, z, z, one]))?)
}

// ---------------------------------------------------------------------------
// Exports

pub fn export(tr: &mut T, ins: &Inst) -> R<()> {
    let tgt = ins.exp_target();
    let en = ins.exp_en();
    let f = tr.t.f32;
    let mut vals = Vec::new();
    for k in 0..4 {
        let v = tr.vreg(ins.exp_src(k) as u32)?;
        vals.push(v);
    }
    let act = tr.active()?;
    // Kill inactive fragments at the valid-mask export.
    if tr.spec.stage == Stage::Fragment && ins.exp_vm() {
        let nact = e(tr.b.logical_not(tr.t.bool_, None, act))?;
        let merge = tr.b.id();
        let kill = tr.b.id();
        ok(tr.b.selection_merge(merge, spirv::SelectionControl::NONE))?;
        ok(tr.b.branch_conditional(nact, kill, merge, []))?;
        e(tr.b.begin_block(Some(kill)))?;
        ok(tr.b.kill())?;
        e(tr.b.begin_block(Some(merge)))?;
    }
    match tgt {
        0..=7 => {
            let mrt = tr.spec.mrt[tgt as usize];
            let comps: Vec<Word> = if ins.exp_compr() {
                let mut c = Vec::new();
                for &w in vals.iter().take(2) {
                    let op = match mrt.export_format {
                        5 => Some(GLOp::UnpackUnorm2x16),
                        6 => Some(GLOp::UnpackSnorm2x16),
                        7 | 8 => None,
                        _ => Some(GLOp::UnpackHalf2x16),
                    };
                    if let Some(op) = op {
                        let v = tr.glsl1(tr.t.v2f, op, w)?;
                        c.push(e(tr.b.composite_extract(f, None, v, [0]))?);
                        c.push(e(tr.b.composite_extract(f, None, v, [1]))?);
                    } else {
                        let u = tr.t.u32;
                        let z = tr.u(0);
                        let k16 = tr.u(16);
                        let (lo, hi) = if mrt.export_format == 8 {
                            (e(tr.b.bit_field_s_extract(u, None, w, z, k16))?, e(tr.b.bit_field_s_extract(u, None, w, k16, k16))?)
                        } else {
                            (e(tr.b.bit_field_u_extract(u, None, w, z, k16))?, e(tr.b.bit_field_u_extract(u, None, w, k16, k16))?)
                        };
                        c.push(tr.to_f(lo)?);
                        c.push(tr.to_f(hi)?);
                    }
                }
                c
            } else {
                let mut c = Vec::new();
                for v in vals.iter().take(4) {
                    c.push(tr.to_f(*v)?);
                }
                c
            };
            let (ty, var) = match tr.out_mrt.get(&tgt) {
                Some(&x) => x,
                None => {
                    let ty = match mrt.class {
                        1 => tr.t.v4i,
                        2 => tr.t.v4u,
                        _ => tr.t.v4f,
                    };
                    let v = tr.output(ty, None, Some(tgt));
                    tr.out_mrt.insert(tgt, (ty, v));
                    (ty, v)
                }
            };
            tr.mrt_mask |= 1 << tgt;
            let mut cs = Vec::new();
            for c in comps {
                cs.push(if ty == tr.t.v4f {
                    c
                } else {
                    let u = tr.to_u(c)?;
                    if ty == tr.t.v4i { tr.to_i(u)? } else { u }
                });
            }
            while cs.len() < 4 {
                cs.push(if ty == tr.t.v4f { tr.f(0.0) } else if ty == tr.t.v4i { tr.i(0) } else { tr.u(0) });
            }
            let mut v = e(tr.b.composite_construct(ty, None, cs))?;
            if let Some(pv) = tr.probe_var {
                let pu = e(tr.b.load(tr.t.u32, None, pv, None, []))?;
                v = if ty == tr.t.v4f {
                    let pf = tr.to_f(pu)?;
                    let one = tr.f(1.0);
                    e(tr.b.composite_construct(ty, None, [pf, pf, pf, one]))?
                } else {
                    // Integer targets: the raw bits in every channel.
                    let pc = if ty == tr.t.v4i { tr.to_i(pu)? } else { pu };
                    e(tr.b.composite_construct(ty, None, [pc, pc, pc, pc]))?
                };
            }
            tr.guarded(act, &mut |tr| ok(tr.b.store(var, v, None, [])))?;
        }
        8 => {
            if en & 1 != 0 {
                let var = match tr.out_depth {
                    Some(v) => v,
                    None => {
                        let v = tr.output(f, Some(BuiltIn::FragDepth), None);
                        tr.out_depth = Some(v);
                        v
                    }
                };
                let d = tr.to_f(vals[0])?;
                tr.guarded(act, &mut |tr| ok(tr.b.store(var, d, None, [])))?;
            }
            if en & 0xE != 0 {
                tr.warn("mrtz stencil/sample-mask export ignored".into());
            }
        }
        9 | 20 => {} // null; NGG primitive export
        12 => {
            let var = match tr.out_pos {
                Some(v) => v,
                None => {
                    let v = tr.output(tr.t.v4f, Some(BuiltIn::Position), None);
                    // Passes that test depth EQUAL against a prepass need
                    // bit-identical positions from separately built pipelines.
                    tr.b.decorate(v, spirv::Decoration::Invariant, []);
                    tr.out_pos = Some(v);
                    v
                }
            };
            let mut cs = Vec::new();
            for v in vals.iter() {
                cs.push(tr.to_f(*v)?);
            }
            // A position with a NaN or infinite component: the console's
            // clipper discards the primitive (the D3D rule), which the
            // title's instancing relies on: a draw covers a fixed number of
            // instance slots, the unused ones zero records whose vertices
            // come out NaN (a zero-length normal normalized). Vulkan leaves
            // such primitives undefined, and the RTX driver drew them as
            // plant-shaped streaks across the view (the tutorial's opening
            // tunnel). Such a vertex goes to one point outside the clip
            // volume (z < 0) instead: a primitive with all three there is
            // dropped.
            // The test reads the bits (`alu::nonfinite_bits`): `OpIsNan` / `OpIsInf` may be
            // folded to false by a driver free to assume no NaNs, as the RTX 50 series did
            // before the modules declared NaN preservation there (0.1.4).
            let bt = tr.t.bool_;
            let mut bad = tr.bool_c(false);
            for &c in &cs {
                let ni = super::alu::nonfinite_bits(tr, c)?;
                bad = e(tr.b.logical_or(bt, None, bad, ni))?;
            }
            // Diagnosis (`Spec::outlier`): finite positions counted in the program's census
            // slots: beyond the threshold (|x|, |y| or |w|) under the code hash, and with |w|
            // under 1e-4 (points at or near infinity, all-zero ones too) under the hash with its
            // lowest bit flipped.
            if let (Some((tbits, tag)), Some(buf)) = (tr.spec.outlier, tr.dump_buf) {
                let f = tr.t.f32;
                let u = tr.t.u32;
                let ax = tr.glsl1(f, GLOp::FAbs, cs[0])?;
                let ay = tr.glsl1(f, GLOp::FAbs, cs[1])?;
                let aw = tr.glsl1(f, GLOp::FAbs, cs[3])?;
                let m1 = tr.glsl2(f, GLOp::FMax, ax, ay)?;
                let m = tr.glsl2(f, GLOp::FMax, m1, aw)?;
                let t = tr.f(f32::from_bits(tbits));
                let over = e(tr.b.f_ord_greater_than(bt, None, m, t))?;
                let tiny = tr.f(1.0e-4);
                let wz = e(tr.b.f_ord_less_than(bt, None, aw, tiny))?;
                let fin = e(tr.b.logical_not(bt, None, bad))?;
                let live = e(tr.b.logical_and(bt, None, fin, act))?;
                let mu = tr.to_u(m)?;
                let mut raw = Vec::with_capacity(6);
                for &c in &cs {
                    raw.push(tr.to_u(c)?);
                }
                // (The first two user-data SGPRs: which draw, e.g. its vertex buffer's address.)
                let ub = tr.spec.user_base as u32;
                raw.push(tr.sreg(ub)?);
                raw.push(tr.sreg(ub + 1)?);
                for (class_tag, c) in [(tag, over), (tag ^ 1, wz)] {
                    let cond = e(tr.b.logical_and(bt, None, c, live))?;
                    let base = super::outlier_slot(class_tag) * super::OUTLIER_STRIDE;
                    tr.guarded(cond, &mut |tr| {
                        let z = tr.u(0);
                        let at = |tr: &mut T, k: u32| -> R<Word> {
                            let i = tr.u(base + k);
                            e(tr.b.access_chain(tr.t.sb_u32, None, buf, [z, i]))
                        };
                        let p0 = at(tr, 0)?;
                        let lo = tr.u(class_tag as u32);
                        ok(tr.b.store(p0, lo, None, []))?;
                        let p1 = at(tr, 1)?;
                        let hi = tr.u((class_tag >> 32) as u32);
                        ok(tr.b.store(p1, hi, None, []))?;
                        let scope = tr.u(spirv::Scope::Device as u32);
                        let sem = tr.u(0);
                        let one = tr.u(1);
                        let p2 = at(tr, 2)?;
                        e(tr.b.atomic_i_add(u, None, p2, scope, sem, one))?;
                        let p3 = at(tr, 3)?;
                        e(tr.b.atomic_u_max(u, None, p3, scope, sem, mu))?;
                        for (k, &v) in raw.iter().enumerate() {
                            let p = at(tr, 4 + k as u32)?;
                            ok(tr.b.store(p, v, None, []))?;
                        }
                        Ok(())
                    })?;
                }
            }
            for (k, c) in cs.iter_mut().enumerate() {
                let o = tr.f([0.0, 0.0, -1.0, 1.0][k]);
                *c = e(tr.b.select(tr.t.f32, None, bad, o, *c))?;
            }
            let p = e(tr.b.composite_construct(tr.t.v4f, None, cs))?;
            ok(tr.b.store(var, p, None, []))?;
        }
        13..=15 => tr.warn(format!("export to pos{} ignored", tgt - 12)),
        32..=63 => {
            let p = tgt - 32;
            let var = match tr.out_params.get(&p) {
                Some(&v) => v,
                None => {
                    let v = tr.output(tr.t.v4f, None, Some(p));
                    tr.out_params.insert(p, v);
                    v
                }
            };
            tr.params = tr.params.max(p + 1);
            let mut cs = Vec::new();
            for (k, v) in vals.iter().enumerate() {
                // Disabled channels export 0.
                cs.push(if en >> k & 1 != 0 { tr.to_f(*v)? } else { tr.f(0.0) });
            }
            let pv = e(tr.b.composite_construct(tr.t.v4f, None, cs))?;
            ok(tr.b.store(var, pv, None, []))?;
        }
        _ => tr.warn(format!("export target {tgt} ignored")),
    }
    Ok(())
}

// ---------------------------------------------------------------------------
// LDS / GDS

/// GDS instructions on the shared GDS storage buffer. Address: M0[15:0]
/// (allocation base) + instruction offset (+ VGPR address except for
/// append/consume/ordered count). Lanes are waves of one, so ds_append /
/// ds_consume add / subtract 1 per active lane and return the pre-op value.
fn gds(tr: &mut T, ins: &Inst, name: &str) -> R<()> {
    let u = tr.t.u32;
    let Some(gds) = tr.gds else { return Err(format!("{:#x}: GDS binding missing", ins.pc)) };
    let append = matches!(name, "ds_append" | "ds_consume" | "ds_ordered_count");
    if name.starts_with("ds_gws_") {
        tr.warn(format!("{:#x}: {name} (global wave sync) ignored", ins.pc));
        return Ok(());
    }
    if name == "ds_ordered_count" {
        tr.warn(format!("{:#x}: ds_ordered_count approximated by an unordered atomic add", ins.pc));
    }
    let off16 = ins.ds_offset0() | (ins.ds_offset1() << 8);
    let m0 = tr.sreg(crate::OP_M0 as u32)?;
    let lo16 = tr.u(0xFFFF);
    let base = e(tr.b.bitwise_and(u, None, m0, lo16))?;
    let k = tr.u(off16);
    let mut addr = e(tr.b.i_add(u, None, base, k))?;
    if !append {
        let va = tr.vreg(ins.ds_addr() as u32)?;
        addr = e(tr.b.i_add(u, None, addr, va))?;
    }
    let two = tr.u(2);
    let dw = e(tr.b.shift_right_logical(u, None, addr, two))?;
    let lim = tr.u(super::GDS_BYTES / 4 - 1);
    let dw = e(tr.b.bitwise_and(u, None, dw, lim))?;
    let p = dword_ptr(tr, gds, dw)?;
    let vd = ins.vdst() as u32;
    let d0 = ins.ds_data0() as u32;
    let act = tr.active()?;
    let mut unsupported = false;
    let name = name.to_string();
    tr.guarded(act, &mut |tr| {
        let one = tr.u(1);
        let ret = match name.as_str() {
            "ds_append" => atomic_op(tr, "add", p, spirv::Scope::Device, one, None)?,
            "ds_consume" => atomic_op(tr, "sub", p, spirv::Scope::Device, one, None)?,
            "ds_ordered_count" => {
                let v = tr.vreg(d0)?;
                atomic_op(tr, "add", p, spirv::Scope::Device, v, None)?
            }
            "ds_write_b32" => {
                let v = tr.vreg(d0)?;
                ok(tr.b.store(p, v, None, []))?;
                None
            }
            "ds_read_b32" => Some(e(tr.b.load(u, None, p, None, []))?),
            n => {
                let rtn = n.contains("_rtn_");
                let op = n.trim_start_matches("ds_").replace("_rtn", "");
                let op = op.trim_end_matches("_u32").trim_end_matches("_i32").trim_end_matches("_b32");
                let op = match op {
                    "min" if n.ends_with("i32") => "smin",
                    "max" if n.ends_with("i32") => "smax",
                    "min" => "umin",
                    "max" => "umax",
                    "wrxchg" => "swap",
                    o => o,
                };
                let v = tr.vreg(d0)?;
                match atomic_op(tr, op, p, spirv::Scope::Device, v, None)? {
                    Some(old) => rtn.then_some(old),
                    None => {
                        unsupported = true;
                        None
                    }
                }
            }
        };
        if let Some(r) = ret {
            tr.set_vreg_raw(vd, r)?;
        }
        Ok(())
    })?;
    if unsupported {
        tr.warn(format!("{:#x}: unsupported GDS {}", ins.pc, ins.name().unwrap_or("")));
    }
    Ok(())
}

/// ds_swizzle_b32 (no LDS access): offset bit 15 selects quad permute mode
/// (lane i of each quad reads lane offset[2i+1:2i]); otherwise lanes read
/// lane ((id & and) | or) ^ xor within groups of 32.
fn swizzle(tr: &mut T, ins: &Inst) -> R<()> {
    let u = tr.t.u32;
    let off = ins.ds_offset0() | (ins.ds_offset1() << 8);
    let v = tr.vreg(ins.ds_data0() as u32)?;
    let lane = tr.subgroup_lane()?;
    let src = if off & 0x8000 != 0 {
        let three = tr.u(3);
        let q = e(tr.b.bitwise_and(u, None, lane, three))?;
        let qbase = e(tr.b.i_sub(u, None, lane, q))?;
        let two = tr.u(2);
        let sh = e(tr.b.i_mul(u, None, q, two))?;
        let c = tr.u(off & 0xFF);
        let sel = e(tr.b.shift_right_logical(u, None, c, sh))?;
        let sel = e(tr.b.bitwise_and(u, None, sel, three))?;
        e(tr.b.bitwise_or(u, None, qbase, sel))?
    } else {
        let (and, or, xor) = (off & 0x1F, (off >> 5) & 0x1F, (off >> 10) & 0x1F);
        let m31 = tr.u(31);
        let l = e(tr.b.bitwise_and(u, None, lane, m31))?;
        let hi_mask = tr.u(!31);
        let hi = e(tr.b.bitwise_and(u, None, lane, hi_mask))?;
        let ka = tr.u(and);
        let ko = tr.u(or);
        let kx = tr.u(xor);
        let s = e(tr.b.bitwise_and(u, None, l, ka))?;
        let s = e(tr.b.bitwise_or(u, None, s, ko))?;
        let s = e(tr.b.bitwise_xor(u, None, s, kx))?;
        e(tr.b.bitwise_or(u, None, hi, s))?
    };
    let (r, _) = tr.lane_fetch_active(v, src)?;
    tr.set_vreg(ins.vdst() as u32, r)
}

pub fn ds(tr: &mut T, ins: &Inst) -> R<()> {
    let name = ins.name().unwrap_or("");
    if ins.ds_gds() {
        return gds(tr, ins, name);
    }
    if name == "ds_swizzle_b32" {
        return swizzle(tr, ins);
    }
    let u = tr.t.u32;
    let off16 = ins.ds_offset0() | (ins.ds_offset1() << 8);
    let vd = ins.vdst() as u32;
    let d0 = ins.ds_data0() as u32;
    let d1 = ins.ds_data1() as u32;
    if name == "ds_bpermute_b32" || name == "ds_permute_b32" {
        // Lane-addressed exchange (address / 4 = lane) within 32 lanes.
        let a = tr.vreg(ins.ds_addr() as u32)?;
        let k = tr.u(off16);
        let a = e(tr.b.i_add(u, None, a, k))?;
        let two = tr.u(2);
        let l = e(tr.b.shift_right_logical(u, None, a, two))?;
        let k31 = tr.u(31);
        let l = e(tr.b.bitwise_and(u, None, l, k31))?;
        let v = tr.vreg(d0)?;
        if name == "ds_permute_b32" {
            tr.warn(format!("{:#x}: ds_permute_b32 approximated as a backward permute", ins.pc));
        }
        let (r, _) = tr.lane_fetch_active(v, l)?;
        return tr.set_vreg(vd, r);
    }
    if name.ends_with("_addtid_b32") && tr.lds.is_none() {
        // A stage without LDS (vertex, pixel): the lane-addressed forms are
        // the compiler's register spills (address M0 + offset + lane * 4,
        // each lane its own dword), held in a private array per invocation.
        // Dropped before, the spilled values read back as stale registers:
        // pixel shaders that spill (many) computed garbage after the spill
        // (1-2's crystal creatures wrote negative colors and bloomed into
        // vertical streaks).
        let arr = tr.spill_var();
        let m0 = tr.sreg(crate::OP_M0 as u32)?;
        let k = tr.u(off16);
        let a = e(tr.b.i_add(u, None, m0, k))?;
        let k7 = tr.u(7);
        let slot = e(tr.b.shift_right_logical(u, None, a, k7))?;
        let km = tr.u(super::SPILL_SLOTS - 1);
        let slot = e(tr.b.bitwise_and(u, None, slot, km))?;
        let pp = tr.b.type_pointer(None, StorageClass::Private, u);
        let p = e(tr.b.access_chain(pp, None, arr, [slot]))?;
        if name.starts_with("ds_read") {
            let v = e(tr.b.load(u, None, p, None, []))?;
            return tr.set_vreg(vd, v);
        }
        let act = tr.active()?;
        let v = tr.vreg(d0)?;
        return tr.guarded(act, &mut |tr| ok(tr.b.store(p, v, None, [])));
    }
    let Some(lds) = tr.lds else {
        tr.warn(format!("{:#x}: {name} without LDS", ins.pc));
        return Ok(());
    };
    tr.lds_order(ins.pc, !name.starts_with("ds_read"))?;
    let p_wg = tr.b.type_pointer(None, StorageClass::Workgroup, u);
    let mut addr = tr.vreg(ins.ds_addr() as u32)?;
    if name.ends_with("_addtid_b32") {
        // Address = M0 base + offset + lane id * 4.
        let m0 = tr.sreg(crate::OP_M0 as u32)?;
        let lane = tr.lane()?;
        let k4 = tr.u(4);
        let t = e(tr.b.i_mul(u, None, lane, k4))?;
        addr = e(tr.b.i_add(u, None, m0, t))?;
    }
    let two = tr.u(2);
    let elem = |tr: &mut T, byte: Word| -> R<Word> {
        let i = e(tr.b.shift_right_logical(u, None, byte, two))?;
        e(tr.b.access_chain(p_wg, None, lds, [i]))
    };
    let add = |tr: &mut T, a: Word, k: u32| -> R<Word> {
        let kk = tr.u(k);
        e(tr.b.i_add(u, None, a, kk))
    };
    let act = tr.active()?;
    let nwords = |n: &str| -> u32 {
        if n.ends_with("b128") {
            4
        } else if n.ends_with("b96") {
            3
        } else if n.ends_with("b64") {
            2
        } else {
            1
        }
    };
    match name {
        "ds_write_b32" | "ds_write_b64" | "ds_write_b96" | "ds_write_b128" | "ds_write_addtid_b32" => {
            let n = nwords(name);
            tr.guarded(act, &mut |tr| {
                let a = add(tr, addr, off16)?;
                for k in 0..n {
                    let ak = add(tr, a, k * 4)?;
                    let p = elem(tr, ak)?;
                    let v = tr.vreg(d0 + k)?;
                    ok(tr.b.store(p, v, None, []))?;
                }
                Ok(())
            })?;
        }
        "ds_write2_b32" | "ds_write2st64_b32" | "ds_write2_b64" | "ds_write2st64_b64" => {
            let st = if name.contains("st64") { 64 } else { 1 };
            let esz = if name.ends_with("b64") { 8 } else { 4 };
            tr.guarded(act, &mut |tr| {
                for (off, src) in [(ins.ds_offset0(), d0), (ins.ds_offset1(), d1)] {
                    let a = add(tr, addr, off * esz * st)?;
                    for w in 0..(esz / 4) {
                        let aw = add(tr, a, w * 4)?;
                        let p = elem(tr, aw)?;
                        let v = tr.vreg(src + w)?;
                        ok(tr.b.store(p, v, None, []))?;
                    }
                }
                Ok(())
            })?;
        }
        "ds_write_b8" | "ds_write_b16" | "ds_write_b8_d16_hi" | "ds_write_b16_d16_hi" => {
            let nb = if name.contains("b8") { 8 } else { 16 };
            tr.guarded(act, &mut |tr| {
                let a = add(tr, addr, off16)?;
                let p = elem(tr, a)?;
                let mut v = tr.vreg(d0)?;
                if name.ends_with("_hi") {
                    let k16 = tr.u(16);
                    v = e(tr.b.shift_right_logical(u, None, v, k16))?;
                }
                let k3 = tr.u(3);
                let sub = e(tr.b.bitwise_and(u, None, a, k3))?;
                let sh = e(tr.b.shift_left_logical(u, None, sub, k3))?;
                let m = tr.u((1u32 << nb) - 1);
                let mask = e(tr.b.shift_left_logical(u, None, m, sh))?;
                let vm = e(tr.b.bitwise_and(u, None, v, m))?;
                let vs = e(tr.b.shift_left_logical(u, None, vm, sh))?;
                let scope = tr.u(spirv::Scope::Workgroup as u32);
                let sem = tr.u(0);
                let nm = e(tr.b.not(u, None, mask))?;
                e(tr.b.atomic_and(u, None, p, scope, sem, nm))?;
                e(tr.b.atomic_or(u, None, p, scope, sem, vs))?;
                Ok(())
            })?;
        }
        "ds_read_b32" | "ds_read_b64" | "ds_read_b96" | "ds_read_b128" | "ds_read_addtid_b32" => {
            let n = nwords(name);
            let a = add(tr, addr, off16)?;
            for k in 0..n {
                let ak = add(tr, a, k * 4)?;
                let p = elem(tr, ak)?;
                let v = e(tr.b.load(u, None, p, None, []))?;
                tr.set_vreg(vd + k, v)?;
            }
        }
        "ds_read2_b32" | "ds_read2st64_b32" | "ds_read2_b64" | "ds_read2st64_b64" => {
            let st = if name.contains("st64") { 64 } else { 1 };
            let esz = if name.ends_with("b64") { 8 } else { 4 };
            let mut dst = vd;
            for off in [ins.ds_offset0(), ins.ds_offset1()] {
                let a = add(tr, addr, off * esz * st)?;
                for w in 0..(esz / 4) {
                    let aw = add(tr, a, w * 4)?;
                    let p = elem(tr, aw)?;
                    let v = e(tr.b.load(u, None, p, None, []))?;
                    tr.set_vreg(dst, v)?;
                    dst += 1;
                }
            }
        }
        "ds_read_u8" | "ds_read_i8" | "ds_read_u16" | "ds_read_i16" => {
            let nb = if name.ends_with('8') { 8 } else { 16 };
            let a = add(tr, addr, off16)?;
            let p = elem(tr, a)?;
            let w = e(tr.b.load(u, None, p, None, []))?;
            let k3 = tr.u(3);
            let sub = e(tr.b.bitwise_and(u, None, a, k3))?;
            let sh = e(tr.b.shift_left_logical(u, None, sub, k3))?;
            let kw = tr.u(nb);
            let r = if name.contains("_i") { e(tr.b.bit_field_s_extract(u, None, w, sh, kw))? } else { e(tr.b.bit_field_u_extract(u, None, w, sh, kw))? };
            tr.set_vreg(vd, r)?;
        }
        n if n.ends_with("_u32") || n.ends_with("_i32") || n.ends_with("_b32") => {
            // Atomics; only active lanes perform them.
            let rtn = n.contains("_rtn_");
            let op = n.trim_start_matches("ds_").replace("_rtn", "");
            let signed = op.ends_with("_i32");
            let op = op.trim_end_matches("_u32").trim_end_matches("_i32").trim_end_matches("_b32").to_string();
            let op = match op.as_str() {
                "min" if signed => "smin".to_string(),
                "max" if signed => "smax".to_string(),
                "min" => "umin".to_string(),
                "max" => "umax".to_string(),
                "wrxchg" => "swap".to_string(),
                "cmpst" => "cmpswap".to_string(),
                o => o.to_string(),
            };
            let mut unsupported = false;
            tr.guarded(act, &mut |tr| {
                let a = add(tr, addr, off16)?;
                let p = elem(tr, a)?;
                let v = tr.vreg(d0)?;
                // ds_cmpst: data0 = compare value, data1 = new value.
                let (v, cmp) = if op == "cmpswap" { (tr.vreg(d1)?, Some(v)) } else { (v, None) };
                match atomic_op(tr, &op, p, spirv::Scope::Workgroup, v, cmp)? {
                    Some(old) if rtn => tr.set_vreg_raw(vd, old)?,
                    Some(_) => {}
                    None => unsupported = true,
                }
                Ok(())
            })?;
            if unsupported {
                tr.warn(format!("{:#x}: unsupported {name}", ins.pc));
            }
        }
        _ => tr.warn(format!("{:#x}: unsupported {name}", ins.pc)),
    }
    Ok(())
}
