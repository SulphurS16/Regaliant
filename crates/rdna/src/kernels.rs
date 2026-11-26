//! Compute kernels the runtime runs on its own behalf (not translated from a
//! title's shaders), built here because this crate owns the SPIR-V builder.

use rspirv::binary::Assemble;
use rspirv::dr::{Builder, Instruction, Operand};
use rspirv::spirv::{self, BuiltIn, Capability, Decoration, ExecutionMode, ExecutionModel, FunctionControl, Op, SelectionControl, StorageClass, Word};

/// Workgroup size of `shadow_merge`.
pub const MERGE_LOCAL_SIZE: u32 = 64;

fn global(b: &mut Builder, ptr_ty: Word, sc: StorageClass) -> Word {
    let id = b.id();
    b.module_mut().types_global_values.push(Instruction::new(Op::Variable, Some(ptr_ty), Some(id), vec![Operand::StorageClass(sc)]));
    id
}

/// Write-back of a dispatch's device-local buffer copy (a "shadow"): every
/// dword of binding 0 (the copy after the dispatch) that differs from
/// binding 1 (the copy taken before it) is stored to binding 2 (guest
/// memory), for dword indices `first..end` (push constants `{first, end}`).
/// Dwords the dispatch did not change are left alone: the CPU may have
/// written them meanwhile, and a whole-range copy would revert that.
pub fn shadow_merge() -> Vec<u32> {
    build_merge().expect("shadow merge kernel")
}

fn build_merge() -> Result<Vec<u32>, rspirv::dr::Error> {
    let lit = Operand::LiteralBit32;
    let mut b = Builder::new();
    b.set_version(1, 3);
    b.capability(Capability::Shader);
    b.extension("SPV_KHR_storage_buffer_storage_class");
    b.memory_model(spirv::AddressingModel::Logical, spirv::MemoryModel::GLSL450);
    let void = b.type_void();
    let bool_ = b.type_bool();
    let u32t = b.type_int(32, 0);
    let v3 = b.type_vector(u32t, 3);
    let rta = b.type_runtime_array(u32t);
    b.decorate(rta, Decoration::ArrayStride, [lit(4)]);
    let st = b.type_struct([rta]);
    b.decorate(st, Decoration::Block, []);
    b.member_decorate(st, 0, Decoration::Offset, [lit(0)]);
    let p_st = b.type_pointer(None, StorageClass::StorageBuffer, st);
    let p_u = b.type_pointer(None, StorageClass::StorageBuffer, u32t);
    let pc = b.type_struct([u32t, u32t]);
    b.decorate(pc, Decoration::Block, []);
    b.member_decorate(pc, 0, Decoration::Offset, [lit(0)]);
    b.member_decorate(pc, 1, Decoration::Offset, [lit(4)]);
    let p_pc = b.type_pointer(None, StorageClass::PushConstant, pc);
    let p_pcu = b.type_pointer(None, StorageClass::PushConstant, u32t);
    let p_in = b.type_pointer(None, StorageClass::Input, v3);
    let gid = global(&mut b, p_in, StorageClass::Input);
    b.decorate(gid, Decoration::BuiltIn, [Operand::BuiltIn(BuiltIn::GlobalInvocationId)]);
    let mut bufs = [0; 3];
    for (i, v) in bufs.iter_mut().enumerate() {
        *v = global(&mut b, p_st, StorageClass::StorageBuffer);
        b.decorate(*v, Decoration::DescriptorSet, [lit(0)]);
        b.decorate(*v, Decoration::Binding, [lit(i as u32)]);
    }
    b.decorate(bufs[0], Decoration::NonWritable, []);
    b.decorate(bufs[1], Decoration::NonWritable, []);
    let pcv = global(&mut b, p_pc, StorageClass::PushConstant);
    let c0 = b.constant_bit32(u32t, 0);
    let c1 = b.constant_bit32(u32t, 1);
    let fty = b.type_function(void, []);
    let f = b.begin_function(void, None, FunctionControl::NONE, fty)?;
    b.begin_block(None)?;
    let g = b.load(v3, None, gid, None, [])?;
    let x = b.composite_extract(u32t, None, g, [0])?;
    let pf = b.access_chain(p_pcu, None, pcv, [c0])?;
    let first = b.load(u32t, None, pf, None, [])?;
    let pe = b.access_chain(p_pcu, None, pcv, [c1])?;
    let end = b.load(u32t, None, pe, None, [])?;
    let i = b.i_add(u32t, None, x, first)?;
    let inside = b.u_less_than(bool_, None, i, end)?;
    let (l_body, l_store, l_join, l_end) = (b.id(), b.id(), b.id(), b.id());
    b.selection_merge(l_end, SelectionControl::NONE)?;
    b.branch_conditional(inside, l_body, l_end, [])?;
    b.begin_block(Some(l_body))?;
    let pw = b.access_chain(p_u, None, bufs[0], [c0, i])?;
    let w = b.load(u32t, None, pw, None, [])?;
    let pp = b.access_chain(p_u, None, bufs[1], [c0, i])?;
    let old = b.load(u32t, None, pp, None, [])?;
    let changed = b.i_not_equal(bool_, None, w, old)?;
    b.selection_merge(l_join, SelectionControl::NONE)?;
    b.branch_conditional(changed, l_store, l_join, [])?;
    b.begin_block(Some(l_store))?;
    let ph = b.access_chain(p_u, None, bufs[2], [c0, i])?;
    b.store(ph, w, None, [])?;
    b.branch(l_join)?;
    b.begin_block(Some(l_join))?;
    b.branch(l_end)?;
    b.begin_block(Some(l_end))?;
    b.ret()?;
    b.end_function()?;
    b.entry_point(ExecutionModel::GLCompute, f, "main", [gid]);
    b.execution_mode(f, ExecutionMode::LocalSize, [MERGE_LOCAL_SIZE, 1, 1]);
    Ok(b.module().assemble())
}

/// Dwords per entry of `shadow_merge_many`'s table.
pub const MERGE_ENTRY_DWORDS: u32 = 8;
/// Entries one `shadow_merge_many` dispatch takes (its search's reach).
pub const MERGE_MANY_MAX: u32 = 256;

/// The write-back of several shadows in one dispatch (`shadow_merge` for
/// each): binding 0 holds their working and pristine copies, binding 1 the
/// table: `n` entries (push constant 1, at most MERGE_MANY_MAX) from dword
/// `at` (push constant 0), MERGE_ENTRY_DWORDS each: [first workgroup, the
/// working copy's dword offset in binding 0, the pristine copy's, first
/// dword, end dword, the destination's device address (low, high: guest
/// memory at the copy's start), 0], in increasing first-workgroup order from
/// 0. Workgroup g takes the last entry whose first workgroup is at most g
/// (an unrolled binary search) and its dwords first + (g - first workgroup)
/// * MERGE_LOCAL_SIZE + local x below end: one that differs from its
/// pristine copy is stored to the destination + 4 * index.
pub fn shadow_merge_many() -> Vec<u32> {
    build_merge_many().expect("shadow merge (many) kernel")
}

fn build_merge_many() -> Result<Vec<u32>, rspirv::dr::Error> {
    let lit = Operand::LiteralBit32;
    let mut b = Builder::new();
    b.set_version(1, 3);
    b.capability(Capability::Shader);
    b.capability(Capability::Int64);
    b.capability(Capability::PhysicalStorageBufferAddresses);
    b.extension("SPV_KHR_storage_buffer_storage_class");
    b.extension("SPV_KHR_physical_storage_buffer");
    b.memory_model(spirv::AddressingModel::PhysicalStorageBuffer64, spirv::MemoryModel::GLSL450);
    let void = b.type_void();
    let bool_ = b.type_bool();
    let u32t = b.type_int(32, 0);
    let u64t = b.type_int(64, 0);
    let v3 = b.type_vector(u32t, 3);
    let rta = b.type_runtime_array(u32t);
    b.decorate(rta, Decoration::ArrayStride, [lit(4)]);
    let st = b.type_struct([rta]);
    b.decorate(st, Decoration::Block, []);
    b.member_decorate(st, 0, Decoration::Offset, [lit(0)]);
    let p_st = b.type_pointer(None, StorageClass::StorageBuffer, st);
    let p_u = b.type_pointer(None, StorageClass::StorageBuffer, u32t);
    let p_dev = b.type_pointer(None, StorageClass::PhysicalStorageBuffer, u32t);
    let pc = b.type_struct([u32t, u32t]);
    b.decorate(pc, Decoration::Block, []);
    b.member_decorate(pc, 0, Decoration::Offset, [lit(0)]);
    b.member_decorate(pc, 1, Decoration::Offset, [lit(4)]);
    let p_pc = b.type_pointer(None, StorageClass::PushConstant, pc);
    let p_pcu = b.type_pointer(None, StorageClass::PushConstant, u32t);
    let p_in = b.type_pointer(None, StorageClass::Input, v3);
    let lid_v = global(&mut b, p_in, StorageClass::Input);
    b.decorate(lid_v, Decoration::BuiltIn, [Operand::BuiltIn(BuiltIn::LocalInvocationId)]);
    let wid_v = global(&mut b, p_in, StorageClass::Input);
    b.decorate(wid_v, Decoration::BuiltIn, [Operand::BuiltIn(BuiltIn::WorkgroupId)]);
    let mut bufs = [0; 2];
    for (i, v) in bufs.iter_mut().enumerate() {
        *v = global(&mut b, p_st, StorageClass::StorageBuffer);
        b.decorate(*v, Decoration::DescriptorSet, [lit(0)]);
        b.decorate(*v, Decoration::Binding, [lit(i as u32)]);
        b.decorate(*v, Decoration::NonWritable, []);
    }
    let (copies, table) = (bufs[0], bufs[1]);
    let pcv = global(&mut b, p_pc, StorageClass::PushConstant);
    let c = |b: &mut Builder, v: u32| b.constant_bit32(u32t, v);
    let (c0, c1, c4) = (c(&mut b, 0), c(&mut b, 1), c(&mut b, 4));
    let c32_64 = b.constant_bit64(u64t, 32);
    let ced = c(&mut b, MERGE_ENTRY_DWORDS);
    let cls = c(&mut b, MERGE_LOCAL_SIZE);
    let fty = b.type_function(void, []);
    let f = b.begin_function(void, None, FunctionControl::NONE, fty)?;
    b.begin_block(None)?;
    let l = b.load(v3, None, lid_v, None, [])?;
    let lx = b.composite_extract(u32t, None, l, [0])?;
    let w = b.load(v3, None, wid_v, None, [])?;
    let g = b.composite_extract(u32t, None, w, [0])?;
    let pa = b.access_chain(p_pcu, None, pcv, [c0])?;
    let at = b.load(u32t, None, pa, None, [])?;
    let pn = b.access_chain(p_pcu, None, pcv, [c1])?;
    let n = b.load(u32t, None, pn, None, [])?;
    let word = |b: &mut Builder, e: Word, k: u32| -> Result<Word, rspirv::dr::Error> {
        let ck = b.constant_bit32(u32t, k);
        let base = b.i_mul(u32t, None, e, ced)?;
        let i = b.i_add(u32t, None, at, base)?;
        let i = b.i_add(u32t, None, i, ck)?;
        let p = b.access_chain(p_u, None, table, [c0, i])?;
        b.load(u32t, None, p, None, [])
    };
    // The last entry whose first workgroup is at most g.
    let mut lo = c0;
    let mut step = MERGE_MANY_MAX / 2;
    while step > 0 {
        let cs = b.constant_bit32(u32t, step);
        let cand = b.i_add(u32t, None, lo, cs)?;
        let inr = b.u_less_than(bool_, None, cand, n)?;
        // Entries past n are not read (their index is clamped to 0).
        let safe = b.select(u32t, None, inr, cand, c0)?;
        let wg = word(&mut b, safe, 0)?;
        let le = b.u_less_than_equal(bool_, None, wg, g)?;
        let take = b.logical_and(bool_, None, inr, le)?;
        lo = b.select(u32t, None, take, cand, lo)?;
        step /= 2;
    }
    let wg0 = word(&mut b, lo, 0)?;
    let wdw = word(&mut b, lo, 1)?;
    let pdw = word(&mut b, lo, 2)?;
    let first = word(&mut b, lo, 3)?;
    let end = word(&mut b, lo, 4)?;
    let dlo = word(&mut b, lo, 5)?;
    let dhi = word(&mut b, lo, 6)?;
    let rel = b.i_sub(u32t, None, g, wg0)?;
    let rel = b.i_mul(u32t, None, rel, cls)?;
    let i = b.i_add(u32t, None, first, rel)?;
    let i = b.i_add(u32t, None, i, lx)?;
    let inside = b.u_less_than(bool_, None, i, end)?;
    let (l_body, l_store, l_join, l_end) = (b.id(), b.id(), b.id(), b.id());
    b.selection_merge(l_end, SelectionControl::NONE)?;
    b.branch_conditional(inside, l_body, l_end, [])?;
    b.begin_block(Some(l_body))?;
    let wi = b.i_add(u32t, None, wdw, i)?;
    let pw = b.access_chain(p_u, None, copies, [c0, wi])?;
    let wv = b.load(u32t, None, pw, None, [])?;
    let pi = b.i_add(u32t, None, pdw, i)?;
    let pp = b.access_chain(p_u, None, copies, [c0, pi])?;
    let old = b.load(u32t, None, pp, None, [])?;
    let changed = b.i_not_equal(bool_, None, wv, old)?;
    b.selection_merge(l_join, SelectionControl::NONE)?;
    b.branch_conditional(changed, l_store, l_join, [])?;
    b.begin_block(Some(l_store))?;
    let lo64 = b.u_convert(u64t, None, dlo)?;
    let hi64 = b.u_convert(u64t, None, dhi)?;
    let hi64 = b.shift_left_logical(u64t, None, hi64, c32_64)?;
    let base = b.bitwise_or(u64t, None, lo64, hi64)?;
    let i4 = b.i_mul(u32t, None, i, c4)?;
    let i64_ = b.u_convert(u64t, None, i4)?;
    let a = b.i_add(u64t, None, base, i64_)?;
    let ptr = b.convert_u_to_ptr(p_dev, None, a)?;
    b.store(ptr, wv, Some(spirv::MemoryAccess::ALIGNED), [lit(4)])?;
    b.branch(l_join)?;
    b.begin_block(Some(l_join))?;
    b.branch(l_end)?;
    b.begin_block(Some(l_end))?;
    b.ret()?;
    b.end_function()?;
    b.entry_point(ExecutionModel::GLCompute, f, "main", [lid_v, wid_v]);
    b.execution_mode(f, ExecutionMode::LocalSize, [MERGE_LOCAL_SIZE, 1, 1]);
    Ok(b.module().assemble())
}

/// The written-bit map's layout (`Spec::write_marks`), in u32 words of its
/// buffer: one bit per dword of the mirror's span (MARK_SPAN_BYTES), then
/// one bit per 4 KiB page of it (pages with marks in the batch running),
/// then the batch's touched-page list: [count, 1, 1, 0] (an indirect
/// dispatch's arguments) and up to MARK_LIST_CAP page indices.
pub const MARK_SPAN_BYTES: u64 = 32 << 30;
pub const MARK_WORDS: u32 = (MARK_SPAN_BYTES / 128) as u32;
pub const MARK_PAGE_BITS_AT: u32 = MARK_WORDS;
pub const MARK_LIST_AT: u32 = MARK_PAGE_BITS_AT + (MARK_SPAN_BYTES / 4096 / 32) as u32;
pub const MARK_LIST_CAP: u32 = 1 << 20;
/// Then one bit per page: the counting write-back's buckets
/// (`mirror_scatter_counting`: WB_BUCKETS dword counts, then WB_BUCKETS
/// page counts), else unused.
pub const MARK_AUX_AT: u32 = MARK_LIST_AT + 4 + MARK_LIST_CAP;
pub const MARK_AUX_WORDS: u32 = (MARK_SPAN_BYTES / 4096 / 32) as u32;
/// Then one bit per page: front end 4's held pages as the GPU sees them
/// (C5, `Spec::marks_held`: stores into them set no marks; the runtime
/// copies its held map here, a held bit set only after the host holds the
/// page and cleared before a batch that may write it again is pushed).
pub const MARK_HELD_AT: u32 = MARK_AUX_AT + MARK_AUX_WORDS;
pub const MARK_HELD_WORDS: u32 = (MARK_SPAN_BYTES / 4096 / 32) as u32;
pub const MARK_TOTAL_WORDS: u32 = MARK_HELD_AT + MARK_HELD_WORDS;
/// `mirror_scatter_counting`'s buckets: 2^WB_BUCKET_SHIFT pages (1 MiB).
pub const WB_BUCKET_SHIFT: u32 = 8;
pub const WB_BUCKETS: u32 = (MARK_SPAN_BYTES / 4096 >> WB_BUCKET_SHIFT) as u32;
/// Then a bit per bucket: its pages' dwords are left out of the counting
/// write-back's stores (diagnosis, the runtime's `wbb lazy`).
pub const WB_LAZY_AT: u32 = MARK_AUX_AT + 2 * WB_BUCKETS;
const _: () = assert!(2 * WB_BUCKETS + WB_BUCKETS / 32 <= MARK_AUX_WORDS);

/// Workgroup size of `mirror_gather`: one invocation per mark word of a
/// page.
pub const GATHER_LOCAL_SIZE: u32 = 32;
/// Dwords per record of `mirror_gather`: the page's index and its 32 mask
/// words.
pub const GATHER_MASK_RECORD_DWORDS: u32 = 1 + 32;
/// Dwords before the first record: the record count, the touched-page
/// list's count (list mode), two unused.
pub const GATHER_HEADER_DWORDS: u32 = 4;

/// Mirror mode's write-back gather: workgroup g takes a page of the mirror,
/// with flags bit 0 the g-th of the touched-page list (MARK_LIST_AT; past
/// its count or MARK_LIST_CAP it does nothing; workgroup 0 copies the
/// list's count to output dword 1), else page `first_page + g` for g below
/// `pages`. Binding 0 is the whole written-bit map, binding 1 the host
/// output: dword 0 counts records, records start at dword
/// GATHER_HEADER_DWORDS. A page with marked dwords takes a record (while
/// fewer than `cap` were taken): its index and its 32 mask words, whose
/// bits it clears; a page over the capacity keeps its bits (a later range
/// gather takes them). Every page taken gets its touched bit cleared. Push
/// constants `{pages, cap, first_page, flags}`.
pub fn mirror_gather() -> Vec<u32> {
    build_gather().expect("mirror gather kernel")
}

fn build_gather() -> Result<Vec<u32>, rspirv::dr::Error> {
    let lit = Operand::LiteralBit32;
    let mut b = Builder::new();
    b.set_version(1, 3);
    b.capability(Capability::Shader);
    b.capability(Capability::GroupNonUniform);
    b.capability(Capability::GroupNonUniformVote);
    b.capability(Capability::GroupNonUniformBallot);
    b.extension("SPV_KHR_storage_buffer_storage_class");
    b.memory_model(spirv::AddressingModel::Logical, spirv::MemoryModel::GLSL450);
    let void = b.type_void();
    let bool_ = b.type_bool();
    let u32t = b.type_int(32, 0);
    let v3 = b.type_vector(u32t, 3);
    let rta = b.type_runtime_array(u32t);
    b.decorate(rta, Decoration::ArrayStride, [lit(4)]);
    let st = b.type_struct([rta]);
    b.decorate(st, Decoration::Block, []);
    b.member_decorate(st, 0, Decoration::Offset, [lit(0)]);
    let p_st = b.type_pointer(None, StorageClass::StorageBuffer, st);
    let p_u = b.type_pointer(None, StorageClass::StorageBuffer, u32t);
    let pc = b.type_struct([u32t, u32t, u32t, u32t]);
    b.decorate(pc, Decoration::Block, []);
    for (k, off) in [0u32, 4, 8, 12].into_iter().enumerate() {
        b.member_decorate(pc, k as u32, Decoration::Offset, [lit(off)]);
    }
    let p_pc = b.type_pointer(None, StorageClass::PushConstant, pc);
    let p_pcu = b.type_pointer(None, StorageClass::PushConstant, u32t);
    let p_in = b.type_pointer(None, StorageClass::Input, v3);
    let lid_v = global(&mut b, p_in, StorageClass::Input);
    b.decorate(lid_v, Decoration::BuiltIn, [Operand::BuiltIn(BuiltIn::LocalInvocationId)]);
    let wid_v = global(&mut b, p_in, StorageClass::Input);
    b.decorate(wid_v, Decoration::BuiltIn, [Operand::BuiltIn(BuiltIn::WorkgroupId)]);
    let mut bufs = [0; 2];
    for (i, v) in bufs.iter_mut().enumerate() {
        *v = global(&mut b, p_st, StorageClass::StorageBuffer);
        b.decorate(*v, Decoration::DescriptorSet, [lit(0)]);
        b.decorate(*v, Decoration::Binding, [lit(i as u32)]);
    }
    let (marks, out) = (bufs[0], bufs[1]);
    let pcv = global(&mut b, p_pc, StorageClass::PushConstant);
    let c = |b: &mut Builder, v: u32| b.constant_bit32(u32t, v);
    let (c0, c1, c2, c3, c5, c31, c32) = (c(&mut b, 0), c(&mut b, 1), c(&mut b, 2), c(&mut b, 3), c(&mut b, 5), c(&mut b, 31), c(&mut b, 32));
    let list_at = c(&mut b, MARK_LIST_AT);
    let list_first = c(&mut b, MARK_LIST_AT + 4);
    let list_cap = c(&mut b, MARK_LIST_CAP);
    let bits_at = c(&mut b, MARK_PAGE_BITS_AT);
    let rec_dw = c(&mut b, GATHER_MASK_RECORD_DWORDS);
    let hdr = c(&mut b, GATHER_HEADER_DWORDS);
    let scope_sg = c(&mut b, 3);
    let scope_dev = c(&mut b, 1);
    let sem_none = c0;
    let fty = b.type_function(void, []);
    let f = b.begin_function(void, None, FunctionControl::NONE, fty)?;
    b.begin_block(None)?;
    let l = b.load(v3, None, lid_v, None, [])?;
    let lid = b.composite_extract(u32t, None, l, [0])?;
    let w = b.load(v3, None, wid_v, None, [])?;
    let g = b.composite_extract(u32t, None, w, [0])?;
    let pc_at = |b: &mut Builder, k: Word| -> Result<Word, rspirv::dr::Error> {
        let p = b.access_chain(p_pcu, None, pcv, [k])?;
        b.load(u32t, None, p, None, [])
    };
    let pages = pc_at(&mut b, c0)?;
    let cap = pc_at(&mut b, c1)?;
    let first_page = pc_at(&mut b, c2)?;
    let flags = pc_at(&mut b, c3)?;
    let lb = b.bitwise_and(u32t, None, flags, c1)?;
    let list = b.i_not_equal(bool_, None, lb, c0)?;
    let is0 = b.i_equal(bool_, None, lid, c0)?;
    // List mode: the list's count (workgroup 0 copies it out).
    let pcount = b.access_chain(p_u, None, marks, [c0, list_at])?;
    let lcount = b.load(u32t, None, pcount, None, [])?;
    let g0 = b.i_equal(bool_, None, g, c0)?;
    let copy_count = b.logical_and(bool_, None, list, g0)?;
    let copy_count = b.logical_and(bool_, None, copy_count, is0)?;
    let (l_cc, l_ccd) = (b.id(), b.id());
    b.selection_merge(l_ccd, SelectionControl::NONE)?;
    b.branch_conditional(copy_count, l_cc, l_ccd, [])?;
    b.begin_block(Some(l_cc))?;
    let po1 = b.access_chain(p_u, None, out, [c0, c1])?;
    b.store(po1, lcount, None, [])?;
    b.branch(l_ccd)?;
    b.begin_block(Some(l_ccd))?;
    let under = b.u_less_than(bool_, None, lcount, list_cap)?;
    let lim_list = b.select(u32t, None, under, lcount, list_cap)?;
    let limit = b.select(u32t, None, list, lim_list, pages)?;
    let inside = b.u_less_than(bool_, None, g, limit)?;
    let (l_in, l_end) = (b.id(), b.id());
    b.selection_merge(l_end, SelectionControl::NONE)?;
    b.branch_conditional(inside, l_in, l_end, [])?;
    b.begin_block(Some(l_in))?;
    let li = b.i_add(u32t, None, list_first, g)?;
    let pli = b.access_chain(p_u, None, marks, [c0, li])?;
    let from_list = b.load(u32t, None, pli, None, [])?;
    let from_range = b.i_add(u32t, None, first_page, g)?;
    let page = b.select(u32t, None, list, from_list, from_range)?;
    let p32 = b.i_mul(u32t, None, page, c32)?;
    let wi = b.i_add(u32t, None, p32, lid)?;
    let pm = b.access_chain(p_u, None, marks, [c0, wi])?;
    let mw = b.load(u32t, None, pm, None, [])?;
    let nz = b.i_not_equal(bool_, None, mw, c0)?;
    let any = b.group_non_uniform_any(bool_, None, scope_sg, nz)?;
    // The page's touched bit, cleared.
    let (l_pb, l_pbd) = (b.id(), b.id());
    b.selection_merge(l_pbd, SelectionControl::NONE)?;
    b.branch_conditional(is0, l_pb, l_pbd, [])?;
    b.begin_block(Some(l_pb))?;
    let pw = b.shift_right_logical(u32t, None, page, c5)?;
    let pwi = b.i_add(u32t, None, bits_at, pw)?;
    let pbit = b.bitwise_and(u32t, None, page, c31)?;
    let pmask = b.shift_left_logical(u32t, None, c1, pbit)?;
    let pinv = b.not(u32t, None, pmask)?;
    let ppb = b.access_chain(p_u, None, marks, [c0, pwi])?;
    b.atomic_and(u32t, None, ppb, scope_dev, sem_none, pinv)?;
    b.branch(l_pbd)?;
    b.begin_block(Some(l_pbd))?;
    let (l_rec, l_recd) = (b.id(), b.id());
    b.selection_merge(l_recd, SelectionControl::NONE)?;
    b.branch_conditional(any, l_rec, l_recd, [])?;
    b.begin_block(Some(l_rec))?;
    // Lane 0 takes a record; the others learn its slot.
    let (l_t, l_td) = (b.id(), b.id());
    b.selection_merge(l_td, SelectionControl::NONE)?;
    b.branch_conditional(is0, l_t, l_td, [])?;
    b.begin_block(Some(l_t))?;
    let pcnt = b.access_chain(p_u, None, out, [c0, c0])?;
    let s_taken = b.atomic_i_add(u32t, None, pcnt, scope_dev, sem_none, c1)?;
    b.branch(l_td)?;
    b.begin_block(Some(l_td))?;
    let s_phi = b.phi(u32t, None, [(s_taken, l_t), (c0, l_rec)])?;
    let slot = b.group_non_uniform_broadcast_first(u32t, None, scope_sg, s_phi)?;
    let room = b.u_less_than(bool_, None, slot, cap)?;
    let (l_r, l_rd) = (b.id(), b.id());
    b.selection_merge(l_rd, SelectionControl::NONE)?;
    b.branch_conditional(room, l_r, l_rd, [])?;
    b.begin_block(Some(l_r))?;
    let sr = b.i_mul(u32t, None, slot, rec_dw)?;
    let rec = b.i_add(u32t, None, hdr, sr)?;
    let (l_h, l_hd) = (b.id(), b.id());
    b.selection_merge(l_hd, SelectionControl::NONE)?;
    b.branch_conditional(is0, l_h, l_hd, [])?;
    b.begin_block(Some(l_h))?;
    let ph = b.access_chain(p_u, None, out, [c0, rec])?;
    b.store(ph, page, None, [])?;
    b.branch(l_hd)?;
    b.begin_block(Some(l_hd))?;
    let r1 = b.i_add(u32t, None, rec, c1)?;
    let ri = b.i_add(u32t, None, r1, lid)?;
    let pr = b.access_chain(p_u, None, out, [c0, ri])?;
    b.store(pr, mw, None, [])?;
    let inv = b.not(u32t, None, mw)?;
    let pm2 = b.access_chain(p_u, None, marks, [c0, wi])?;
    b.atomic_and(u32t, None, pm2, scope_dev, sem_none, inv)?;
    b.branch(l_rd)?;
    b.begin_block(Some(l_rd))?;
    b.branch(l_recd)?;
    b.begin_block(Some(l_recd))?;
    b.branch(l_end)?;
    b.begin_block(Some(l_end))?;
    b.ret()?;
    b.end_function()?;
    b.entry_point(ExecutionModel::GLCompute, f, "main", [lid_v, wid_v]);
    b.execution_mode(f, ExecutionMode::LocalSize, [GATHER_LOCAL_SIZE, 1, 1]);
    Ok(b.module().assemble())
}

/// Counters of `mirror_scatter` (binding 1): pages written back, dwords,
/// pages with marks but no guest page-table entry (their dwords are lost to
/// guest memory), scatters that found the touched-page list past
/// MARK_LIST_CAP; four more unused (the lazy write-back study's census,
/// FINDINGS.md 2026-10-31 night).
pub const SCATTER_COUNTERS: u32 = 8;

/// Workgroup size of `mirror_scatter`: invocation t copies dwords t, t +
/// 256, t + 512 and t + 768 of its page, so that a subgroup's stores are
/// consecutive dwords (one transfer across PCIe rather than one per dword).
pub const SCATTER_LOCAL_SIZE: u32 = 256;

/// Front end 2's write-back (docs/FRONTEND2.md "Coherence"): workgroup g
/// takes the g-th page of the touched-page list (MARK_LIST_AT; past its
/// count or MARK_LIST_CAP it does nothing) and stores every dword its
/// written-bit map marks from the mirror into guest memory, then clears the
/// page's mask words and its touched bit. The mirror's span starts at
/// device address `mirror`; guest memory is where the guest page table
/// (`table`: one u64 device address per 4 KiB page of the span, 0 where
/// nothing is mapped) puts the page. Binding 0 is the whole written-bit
/// map, binding 1 the SCATTER_COUNTERS counters. Push constants `{mirror
/// (low, high), table (low, high)}`.
pub fn mirror_scatter() -> Vec<u32> {
    build_scatter(false, false, false, 0, false, false).expect("mirror scatter kernel")
}

/// `mirror_scatter` emptying the touched-page list itself: workgroup 0's
/// first invocation takes the list's count and stores 0 there (an atomic
/// exchange; the indirect dispatch read the count already), and the other
/// workgroups never read it (they are bounded by MARK_LIST_CAP alone: the
/// dispatch made as many as the count), so the write-back needs no fill of
/// the count and no barrier before one.
pub fn mirror_scatter_reset() -> Vec<u32> {
    build_scatter(true, false, false, 0, false, false).expect("mirror scatter kernel (emptying the list)")
}

/// `mirror_scatter_reset` that also adds, per bucket of 2^WB_BUCKET_SHIFT
/// pages of the span, the dwords it writes back and the pages with any into
/// the map's aux region ([dwords by bucket, then pages by bucket];
/// diagnosis, the runtime's `DS_WB_BUCKETS`), and stores nothing for pages
/// of buckets whose bit at WB_LAZY_AT is set (their marks cleared, their
/// dwords counted as if stored).
pub fn mirror_scatter_counting() -> Vec<u32> {
    build_scatter(true, true, false, 0, false, false).expect("mirror scatter kernel (counting buckets)")
}

/// `mirror_scatter_reset` that also counts what holding pages would leave
/// out (diagnosis, the runtime's `DS_FE2_HOLD=1`; FRONTEND2.md "GPU-held
/// pages"): a bitmap of held pages (a bit per page of the span, bit i of
/// word i / 32) at the device address in push constants 4-5 (non-zero), and
/// for each listed page whose bit is set its marked dwords are added to
/// counter 4 and the page to counter 5 (as stored or not); stores unchanged.
/// A page with marked dwords also gets its bit in the map's aux region (bit
/// i of word MARK_AUX_AT + i / 32: written back since the runtime last read
/// and cleared them).
pub fn mirror_scatter_hold_census() -> Vec<u32> {
    build_scatter(true, false, true, 0, false, false).expect("mirror scatter kernel (hold census)")
}

/// `mirror_scatter_reset` that also counts the pages it stores unchanged
/// (diagnosis, the runtime's `DS_WB_SAME=1`): before each store the dword's
/// copy in guest memory is read and compared; a page whose marked dwords
/// all equal it adds them to counter 4 and itself to counter 5 (stores
/// unchanged).
pub fn mirror_scatter_same_census() -> Vec<u32> {
    build_scatter(true, false, false, 1, false, false).expect("mirror scatter kernel (same-page census)")
}

/// `mirror_scatter_reset` that leaves out pages it would store unchanged (the
/// runtime's default write-back since 2026-11-05): a 64-bit hash of the page's whole mirror copy
/// (video memory alone) is compared with the one kept at its last store in
/// a table at the device address in push constants 4-5 (8 bytes a page of
/// the span; 0: none, as the runtime clears it wherever an upload or a
/// mapping change may have changed guest memory); equal: the page is left
/// out (guest memory holds those words), else its marked dwords are stored
/// and the hash kept. Marks and touched bit are cleared either way. A false
/// match needs a 64-bit hash collision. Counters 4 and 5 add the dwords and
/// pages left out (counter 1 counts those stored).
pub fn mirror_scatter_same_skip() -> Vec<u32> {
    build_scatter(true, false, false, 2, false, false).expect("mirror scatter kernel (skipping unchanged pages)")
}

/// `mirror_scatter_reset` storing the dwords of 32-byte sectors whose marks
/// form two runs or more one dword per line in each store instruction
/// (invocation l of a subgroup takes line l, the subgroups the line's dwords
/// in turn), sectors with one run (full ones too) as before: a store instruction whose marked dwords leave holes in
/// one line runs ~100x slower into host memory on the RTX 4090 than one
/// that writes single dwords of different lines (FINDINGS.md 2026-11-09:
/// 2 of every 3 dwords marked 66 ms for 2,048 pages, 32 isolated dwords a
/// page 0.42 ms). The same dwords are stored.
pub fn mirror_scatter_split() -> Vec<u32> {
    build_scatter(true, false, false, 0, true, false).expect("mirror scatter kernel (split lines)")
}

/// `mirror_scatter_split` leaving held pages in the mirror (the runtime's
/// front end 4 holds, FRONTEND4.md C4): a bitmap of held pages (a bit per
/// page of the span, bit i of word i / 32) at the device address in push
/// constants 4-5; a listed page whose bit is set stores nothing and keeps
/// its mask words (its marked dwords stay pending in the mirror until the
/// runtime takes them), its touched bit cleared as ever (a later mark lists
/// it again); its marked dwords add to counter 4 and the page to counter 5.
/// A page stored with marked dwords gets its bit in the map's aux region
/// (bit i of word MARK_AUX_AT + i / 32: written back since the runtime last
/// read and cleared them).
pub fn mirror_scatter_held() -> Vec<u32> {
    build_scatter(true, false, false, 0, true, true).expect("mirror scatter kernel (held pages left out)")
}

fn build_scatter(reset: bool, count: bool, hold: bool, same: u8, split: bool, skip_held: bool) -> Result<Vec<u32>, rspirv::dr::Error> {
    let lit = Operand::LiteralBit32;
    let mut b = Builder::new();
    b.set_version(1, 3);
    b.capability(Capability::Shader);
    b.capability(Capability::Int64);
    b.capability(Capability::PhysicalStorageBufferAddresses);
    b.extension("SPV_KHR_storage_buffer_storage_class");
    b.extension("SPV_KHR_physical_storage_buffer");
    b.memory_model(spirv::AddressingModel::PhysicalStorageBuffer64, spirv::MemoryModel::GLSL450);
    let void = b.type_void();
    let bool_ = b.type_bool();
    let u32t = b.type_int(32, 0);
    let u64t = b.type_int(64, 0);
    let v3 = b.type_vector(u32t, 3);
    let rta = b.type_runtime_array(u32t);
    b.decorate(rta, Decoration::ArrayStride, [lit(4)]);
    let st = b.type_struct([rta]);
    b.decorate(st, Decoration::Block, []);
    b.member_decorate(st, 0, Decoration::Offset, [lit(0)]);
    let p_st = b.type_pointer(None, StorageClass::StorageBuffer, st);
    let p_u = b.type_pointer(None, StorageClass::StorageBuffer, u32t);
    let p_dev = b.type_pointer(None, StorageClass::PhysicalStorageBuffer, u32t);
    let p_dev64 = b.type_pointer(None, StorageClass::PhysicalStorageBuffer, u64t);
    // (The skipping write-back: 6-7 the other family's table, whose entry
    // of a page this one stores is cleared. 8-9: the zero-known bitmap,
    // every variant.)
    let pc_members = 10;
    let pc = b.type_struct(vec![u32t; pc_members]);
    b.decorate(pc, Decoration::Block, []);
    for k in 0..pc_members as u32 {
        b.member_decorate(pc, k, Decoration::Offset, [lit(4 * k)]);
    }
    let p_pc = b.type_pointer(None, StorageClass::PushConstant, pc);
    let p_pcu = b.type_pointer(None, StorageClass::PushConstant, u32t);
    let p_in = b.type_pointer(None, StorageClass::Input, v3);
    let lid_v = global(&mut b, p_in, StorageClass::Input);
    b.decorate(lid_v, Decoration::BuiltIn, [Operand::BuiltIn(BuiltIn::LocalInvocationId)]);
    let wid_v = global(&mut b, p_in, StorageClass::Input);
    b.decorate(wid_v, Decoration::BuiltIn, [Operand::BuiltIn(BuiltIn::WorkgroupId)]);
    let mut bufs = [0; 2];
    for (i, v) in bufs.iter_mut().enumerate() {
        *v = global(&mut b, p_st, StorageClass::StorageBuffer);
        b.decorate(*v, Decoration::DescriptorSet, [lit(0)]);
        b.decorate(*v, Decoration::Binding, [lit(i as u32)]);
    }
    let (marks, counters) = (bufs[0], bufs[1]);
    let pcv = global(&mut b, p_pc, StorageClass::PushConstant);
    let c = |b: &mut Builder, v: u32| b.constant_bit32(u32t, v);
    let (c0, c1, c2, c3, c5, c31, c32) = (c(&mut b, 0), c(&mut b, 1), c(&mut b, 2), c(&mut b, 3), c(&mut b, 5), c(&mut b, 31), c(&mut b, 32));
    let c32_64 = b.constant_bit64(u64t, 32);
    let c0_64 = b.constant_bit64(u64t, 0);
    let c8_64 = b.constant_bit64(u64t, 8);
    let c4096_64 = b.constant_bit64(u64t, 4096);
    let list_at = c(&mut b, MARK_LIST_AT);
    let list_first = c(&mut b, MARK_LIST_AT + 4);
    let list_cap = c(&mut b, MARK_LIST_CAP);
    let bits_at = c(&mut b, MARK_PAGE_BITS_AT);
    let scope_dev = c(&mut b, 1);
    let scope_wg = c(&mut b, 2);
    // AcquireRelease | UniformMemory: the mask words read before any is cleared.
    let sem_wg = c(&mut b, 0x8 | 0x40);
    let sem_none = c0;
    let fty = b.type_function(void, []);
    let f = b.begin_function(void, None, FunctionControl::NONE, fty)?;
    b.begin_block(None)?;
    let l = b.load(v3, None, lid_v, None, [])?;
    let lid = b.composite_extract(u32t, None, l, [0])?;
    let w = b.load(v3, None, wid_v, None, [])?;
    let g = b.composite_extract(u32t, None, w, [0])?;
    let pc_at = |b: &mut Builder, k: Word| -> Result<Word, rspirv::dr::Error> {
        let p = b.access_chain(p_pcu, None, pcv, [k])?;
        b.load(u32t, None, p, None, [])
    };
    let addr64 = |b: &mut Builder, lo: Word, hi: Word| -> Result<Word, rspirv::dr::Error> {
        let lo64 = b.u_convert(u64t, None, lo)?;
        let hi64 = b.u_convert(u64t, None, hi)?;
        let hi64 = b.shift_left_logical(u64t, None, hi64, c32_64)?;
        b.bitwise_or(u64t, None, lo64, hi64)
    };
    let mlo = pc_at(&mut b, c0)?;
    let mhi = pc_at(&mut b, c1)?;
    let tlo = pc_at(&mut b, c2)?;
    let thi = pc_at(&mut b, c3)?;
    let mirror = addr64(&mut b, mlo, mhi)?;
    let table = addr64(&mut b, tlo, thi)?;
    let is0 = b.i_equal(bool_, None, lid, c0)?;
    let pcount = b.access_chain(p_u, None, marks, [c0, list_at])?;
    let inside = if reset {
        // Workgroup 0's first invocation takes the count, empties the list
        // and counts a list past its capacity.
        let g0 = b.i_equal(bool_, None, g, c0)?;
        let first = b.logical_and(bool_, None, g0, is0)?;
        let (l_rs, l_rsd) = (b.id(), b.id());
        b.selection_merge(l_rsd, SelectionControl::NONE)?;
        b.branch_conditional(first, l_rs, l_rsd, [])?;
        b.begin_block(Some(l_rs))?;
        let lcount = b.atomic_exchange(u32t, None, pcount, scope_dev, sem_none, c0)?;
        let over = b.u_greater_than(bool_, None, lcount, list_cap)?;
        let (l_ov, l_ovd) = (b.id(), b.id());
        b.selection_merge(l_ovd, SelectionControl::NONE)?;
        b.branch_conditional(over, l_ov, l_ovd, [])?;
        b.begin_block(Some(l_ov))?;
        let pov = b.access_chain(p_u, None, counters, [c0, c3])?;
        b.atomic_i_add(u32t, None, pov, scope_dev, sem_none, c1)?;
        b.branch(l_ovd)?;
        b.begin_block(Some(l_ovd))?;
        b.branch(l_rsd)?;
        b.begin_block(Some(l_rsd))?;
        // (Uniform in the workgroup: the barrier below is inside it.)
        b.u_less_than(bool_, None, g, list_cap)?
    } else {
        let lcount = b.load(u32t, None, pcount, None, [])?;
        // Workgroup 0 counts a list past its capacity.
        let g0 = b.i_equal(bool_, None, g, c0)?;
        let over = b.u_greater_than(bool_, None, lcount, list_cap)?;
        let note_over = b.logical_and(bool_, None, g0, is0)?;
        let note_over = b.logical_and(bool_, None, note_over, over)?;
        let (l_ov, l_ovd) = (b.id(), b.id());
        b.selection_merge(l_ovd, SelectionControl::NONE)?;
        b.branch_conditional(note_over, l_ov, l_ovd, [])?;
        b.begin_block(Some(l_ov))?;
        let pov = b.access_chain(p_u, None, counters, [c0, c3])?;
        b.atomic_i_add(u32t, None, pov, scope_dev, sem_none, c1)?;
        b.branch(l_ovd)?;
        b.begin_block(Some(l_ovd))?;
        let under = b.u_less_than(bool_, None, lcount, list_cap)?;
        let limit = b.select(u32t, None, under, lcount, list_cap)?;
        // (Uniform in the workgroup: the barrier below is inside it.)
        b.u_less_than(bool_, None, g, limit)?
    };
    let (l_in, l_end) = (b.id(), b.id());
    b.selection_merge(l_end, SelectionControl::NONE)?;
    b.branch_conditional(inside, l_in, l_end, [])?;
    b.begin_block(Some(l_in))?;
    let page = {
        let li = b.i_add(u32t, None, list_first, g)?;
        let pli = b.access_chain(p_u, None, marks, [c0, li])?;
        b.load(u32t, None, pli, None, [])?
    };
    // (A page past the guest page table is never read there or in the map:
    // stored as unmapped, counted lost. Front end 3's C6 met one: a page
    // number no store's mark gives.)
    let c_pages = c(&mut b, ((crate::translate::PT_END - crate::translate::PT_BASE) >> crate::translate::PT_PAGE_SHIFT) as u32);
    let page_in = b.u_less_than(bool_, None, page, c_pages)?;
    let page = b.select(u32t, None, page_in, page, c0)?;
    // The page's bit in the zero-known bitmap (push constants 8-9: a bit per
    // page of the span, bit i of word i / 32; 0: none) cleared by the first
    // invocation: once the write-back stores into the page, guest memory's
    // copy may hold other words than zeros (`zero_store`; the runtime's
    // `gpu::zk`). Every listed page, stored now or held: a held page's words
    // reach guest memory later.
    {
        let c8 = c(&mut b, 8);
        let c9 = c(&mut b, 9);
        let zlo = pc_at(&mut b, c8)?;
        let zhi = pc_at(&mut b, c9)?;
        let zmap = addr64(&mut b, zlo, zhi)?;
        let has = b.i_not_equal(bool_, None, zmap, c0_64)?;
        let go = b.logical_and(bool_, None, is0, page_in)?;
        let go = b.logical_and(bool_, None, go, has)?;
        let (l_z, l_zd) = (b.id(), b.id());
        b.selection_merge(l_zd, SelectionControl::NONE)?;
        b.branch_conditional(go, l_z, l_zd, [])?;
        b.begin_block(Some(l_z))?;
        let zw = b.shift_right_logical(u32t, None, page, c5)?;
        let zw64 = b.u_convert(u64t, None, zw)?;
        let c4_64z = b.constant_bit64(u64t, 4);
        let zoff = b.i_mul(u64t, None, zw64, c4_64z)?;
        let za = b.i_add(u64t, None, zmap, zoff)?;
        let zp = b.convert_u_to_ptr(p_dev, None, za)?;
        let zbit = b.bitwise_and(u32t, None, page, c31)?;
        let one = b.shift_left_logical(u32t, None, c1, zbit)?;
        let mask = b.not(u32t, None, one)?;
        b.atomic_and(u32t, None, zp, scope_dev, sem_none, mask)?;
        b.branch(l_zd)?;
        b.begin_block(Some(l_zd))?;
    }
    // (`same`: whether any stored dword differs from guest memory's copy,
    // workgroup memory set to 0 by invocation 0 first.)
    let shared2 = if same != 0 {
        let p_wg_u = b.type_pointer(None, StorageClass::Workgroup, u32t);
        let v = global(&mut b, p_wg_u, StorageClass::Workgroup);
        let v2 = global(&mut b, p_wg_u, StorageClass::Workgroup);
        let (l_z, l_zd) = (b.id(), b.id());
        b.selection_merge(l_zd, SelectionControl::NONE)?;
        b.branch_conditional(is0, l_z, l_zd, [])?;
        b.begin_block(Some(l_z))?;
        b.store(v, c0, None, [])?;
        b.store(v2, c0, None, [])?;
        b.branch(l_zd)?;
        b.begin_block(Some(l_zd))?;
        let sem_sh = c(&mut b, 0x8 | 0x100);
        b.control_barrier(scope_wg, scope_wg, sem_sh)?;
        Some((v, v2))
    } else {
        None
    };
    let diff_sh = shared2.map(|(v, _)| v);
    let first_word = b.i_mul(u32t, None, page, c32)?;
    // The page's k-th mask word.
    let mask_word = |b: &mut Builder, k: Word| -> Result<Word, rspirv::dr::Error> {
        let wi = b.i_add(u32t, None, first_word, k)?;
        let pm = b.access_chain(p_u, None, marks, [c0, wi])?;
        b.load(u32t, None, pm, None, [])
    };
    // The page's entry in the guest page table, its first byte in the mirror.
    let page64 = b.u_convert(u64t, None, page)?;
    let toff = b.i_mul(u64t, None, page64, c8_64)?;
    let ta = b.i_add(u64t, None, table, toff)?;
    let tp = b.convert_u_to_ptr(p_dev64, None, ta)?;
    let dest = b.load(u64t, None, tp, Some(spirv::MemoryAccess::ALIGNED), [lit(8)])?;
    let mapped = b.i_not_equal(bool_, None, dest, c0_64)?;
    let mapped = b.logical_and(bool_, None, mapped, page_in)?;
    let moff = b.i_mul(u64t, None, page64, c4096_64)?;
    let msrc = b.i_add(u64t, None, mirror, moff)?;
    // (`count`: a page of a lazy bucket stores nothing; its dwords are
    // counted as stored.)
    let store_ok = if count {
        let shift = c(&mut b, WB_BUCKET_SHIFT);
        let bucket = b.shift_right_logical(u32t, None, page, shift)?;
        let lw = b.shift_right_logical(u32t, None, bucket, c5)?;
        let lazy_at = c(&mut b, WB_LAZY_AT);
        let lwi = b.i_add(u32t, None, lazy_at, lw)?;
        let plw = b.access_chain(p_u, None, marks, [c0, lwi])?;
        let lword = b.load(u32t, None, plw, None, [])?;
        let lbit = b.bitwise_and(u32t, None, bucket, c31)?;
        let lsh = b.shift_right_logical(u32t, None, lword, lbit)?;
        let lone = b.bitwise_and(u32t, None, lsh, c1)?;
        let eager = b.i_equal(bool_, None, lone, c0)?;
        b.logical_and(bool_, None, mapped, eager)?
    } else {
        mapped
    };
    // (`skip_held`: the page's bit in the held-page bitmap; a held page
    // stores nothing.)
    let held_one = if skip_held {
        let c4 = c(&mut b, 4);
        let hlo = pc_at(&mut b, c4)?;
        let hhi = pc_at(&mut b, c5)?;
        let hmap = addr64(&mut b, hlo, hhi)?;
        let hw = b.shift_right_logical(u32t, None, page, c5)?;
        let hw64 = b.u_convert(u64t, None, hw)?;
        let c4_64 = b.constant_bit64(u64t, 4);
        let hoff = b.i_mul(u64t, None, hw64, c4_64)?;
        let ha = b.i_add(u64t, None, hmap, hoff)?;
        let hp = b.convert_u_to_ptr(p_dev, None, ha)?;
        let hword = b.load(u32t, None, hp, Some(spirv::MemoryAccess::ALIGNED), [lit(4)])?;
        let hbit = b.bitwise_and(u32t, None, page, c31)?;
        let hsh = b.shift_right_logical(u32t, None, hword, hbit)?;
        Some(b.bitwise_and(u32t, None, hsh, c1)?)
    } else {
        None
    };
    let held = match held_one {
        Some(h) => Some(b.i_not_equal(bool_, None, h, c0)?),
        None => None,
    };
    let store_ok = match held {
        Some(h) => {
            let not_held = b.logical_not(bool_, None, h)?;
            b.logical_and(bool_, None, store_ok, not_held)?
        }
        None => store_ok,
    };
    let bit_of = b.bitwise_and(u32t, None, lid, c31)?;
    // (`same` 2: the page's hash over its whole mirror copy against the one
    // stored at its last store; equal and non-zero: left out.)
    let mut hashed: Option<(Word, Word, Word, Word, Word, Word, Word, Word)> = None;
    let store_ok = if same == 2 {
        let (sh_lo, sh_hi) = shared2.expect("same");
        let k1 = c(&mut b, 0x9E37_79B1);
        let k2 = c(&mut b, 0x85EB_CA77);
        let k3 = c(&mut b, 0xC2B2_AE3D);
        let k4 = c(&mut b, 0x27D4_EB2F);
        let k5 = c(&mut b, 0x1656_67B1);
        let c13 = c(&mut b, 13);
        let c15 = c(&mut b, 15);
        let c16 = c(&mut b, 16);
        let mut acc_lo = c0;
        let mut acc_hi = c0;
        for r in 0..4u32 {
            let cr = b.constant_bit32(u32t, r * SCATTER_LOCAL_SIZE);
            let d = b.i_add(u32t, None, lid, cr)?;
            let d4 = b.shift_left_logical(u32t, None, d, c2)?;
            let d64 = b.u_convert(u64t, None, d4)?;
            let sa = b.i_add(u64t, None, msrc, d64)?;
            let sp = b.convert_u_to_ptr(p_dev, None, sa)?;
            let v = b.load(u32t, None, sp, Some(spirv::MemoryAccess::ALIGNED), [lit(4)])?;
            // (Two mixes of the dword and its place, XORed over the page.)
            let x = b.i_mul(u32t, None, v, k1)?;
            let y = b.i_mul(u32t, None, d, k2)?;
            let x = b.i_add(u32t, None, x, y)?;
            let s = b.shift_right_logical(u32t, None, x, c15)?;
            let x = b.bitwise_xor(u32t, None, x, s)?;
            let x = b.i_mul(u32t, None, x, k3)?;
            let s = b.shift_right_logical(u32t, None, x, c13)?;
            let x = b.bitwise_xor(u32t, None, x, s)?;
            acc_lo = b.bitwise_xor(u32t, None, acc_lo, x)?;
            let z = b.i_mul(u32t, None, v, k4)?;
            let w = b.i_mul(u32t, None, d, k5)?;
            let z = b.bitwise_xor(u32t, None, z, w)?;
            let z = b.i_add(u32t, None, z, k1)?;
            let s = b.shift_right_logical(u32t, None, z, c16)?;
            let z = b.bitwise_xor(u32t, None, z, s)?;
            let z = b.i_mul(u32t, None, z, k2)?;
            let s = b.shift_right_logical(u32t, None, z, c13)?;
            let z = b.bitwise_xor(u32t, None, z, s)?;
            acc_hi = b.bitwise_xor(u32t, None, acc_hi, z)?;
        }
        b.atomic_xor(u32t, None, sh_lo, scope_wg, sem_none, acc_lo)?;
        b.atomic_xor(u32t, None, sh_hi, scope_wg, sem_none, acc_hi)?;
        let sem_sh = c(&mut b, 0x8 | 0x100);
        b.control_barrier(scope_wg, scope_wg, sem_sh)?;
        let h_lo = b.load(u32t, None, sh_lo, None, [])?;
        let h_hi = b.load(u32t, None, sh_hi, None, [])?;
        // The table's entry (push constants 4-5: the table; 8 bytes a page).
        let c4 = c(&mut b, 4);
        let tlo_ = pc_at(&mut b, c4)?;
        let thi_ = pc_at(&mut b, c5)?;
        let tab = addr64(&mut b, tlo_, thi_)?;
        let c8_64b = b.constant_bit64(u64t, 8);
        let toff2 = b.i_mul(u64t, None, page64, c8_64b)?;
        let te = b.i_add(u64t, None, tab, toff2)?;
        let te_lo = b.convert_u_to_ptr(p_dev, None, te)?;
        let c4_64 = b.constant_bit64(u64t, 4);
        let te4 = b.i_add(u64t, None, te, c4_64)?;
        let te_hi = b.convert_u_to_ptr(p_dev, None, te4)?;
        let t_lo = b.load(u32t, None, te_lo, Some(spirv::MemoryAccess::ALIGNED), [lit(4)])?;
        let t_hi = b.load(u32t, None, te_hi, Some(spirv::MemoryAccess::ALIGNED), [lit(4)])?;
        let eq_lo = b.i_equal(bool_, None, h_lo, t_lo)?;
        let eq_hi = b.i_equal(bool_, None, h_hi, t_hi)?;
        let tor = b.bitwise_or(u32t, None, t_lo, t_hi)?;
        let nz = b.i_not_equal(bool_, None, tor, c0)?;
        let eq = b.logical_and(bool_, None, eq_lo, eq_hi)?;
        let unchanged = b.logical_and(bool_, None, eq, nz)?;
        // The other family's entry of the page (push constants 6-7; 0: no
        // other table): cleared when this family stores the page, so that the
        // other's next write-back of it cannot be left out on a hash its store
        // made before this one changed guest memory's copy (a dword both
        // families write).
        let c6 = c(&mut b, 6);
        let c7 = c(&mut b, 7);
        let olo_ = pc_at(&mut b, c6)?;
        let ohi_ = pc_at(&mut b, c7)?;
        let otab = addr64(&mut b, olo_, ohi_)?;
        let c0_64 = b.constant_bit64(u64t, 0);
        let has_other = b.i_not_equal(bool_, None, otab, c0_64)?;
        let oe = b.i_add(u64t, None, otab, toff2)?;
        let oe_lo = b.convert_u_to_ptr(p_dev, None, oe)?;
        let oe4 = b.i_add(u64t, None, oe, c4_64)?;
        let oe_hi = b.convert_u_to_ptr(p_dev, None, oe4)?;
        hashed = Some((unchanged, te_lo, te_hi, h_lo, h_hi, has_other, oe_lo, oe_hi));
        let changed = b.logical_not(bool_, None, unchanged)?;
        b.logical_and(bool_, None, store_ok, changed)?
    } else {
        store_ok
    };
    // Dwords lid + 256 r, r = 0..3: marked ones copied (`split`: those of
    // lines whose marks form at most one run; the rest below).
    // Whether the 32-byte sector `sec` (0-3) of a line's mask word holds
    // two runs of marked dwords or more (a hole between marked ones): its
    // run starts within the sector (bits set whose lower neighbour in the
    // sector is clear) number two or more. Such sectors go out of one store
    // instruction slowly; one run (a full sector too) fast.
    let c_fe = c(&mut b, 0xFEFE_FEFE);
    let c_ff = c(&mut b, 0xFF);
    let c2b = c(&mut b, 2);
    let c3b = c(&mut b, 3);
    let multi = |b: &mut Builder, mw: Word, sec: Word| -> Result<Word, rspirv::dr::Error> {
        let sh = b.shift_left_logical(u32t, None, mw, c1)?;
        let shm = b.bitwise_and(u32t, None, sh, c_fe)?;
        let nsh = b.not(u32t, None, shm)?;
        let starts = b.bitwise_and(u32t, None, mw, nsh)?;
        let s8 = b.shift_left_logical(u32t, None, sec, c3b)?;
        let x = b.shift_right_logical(u32t, None, starts, s8)?;
        let x = b.bitwise_and(u32t, None, x, c_ff)?;
        let n = b.bit_count(u32t, None, x)?;
        b.u_greater_than_equal(bool_, None, n, c2b)
    };
    for r in 0..4u32 {
        let cr = b.constant_bit32(u32t, r * SCATTER_LOCAL_SIZE);
        let d = b.i_add(u32t, None, lid, cr)?;
        let wd = b.shift_right_logical(u32t, None, d, c5)?;
        let mw = mask_word(&mut b, wd)?;
        let sh = b.shift_right_logical(u32t, None, mw, bit_of)?;
        let bit = b.bitwise_and(u32t, None, sh, c1)?;
        let set = b.i_not_equal(bool_, None, bit, c0)?;
        let go = b.logical_and(bool_, None, set, store_ok)?;
        let go = if split {
            // (A dword of a sector with one run: as before, fast; of one
            // with two or more: below.)
            let sec = b.shift_right_logical(u32t, None, bit_of, c3b)?;
            let m = multi(&mut b, mw, sec)?;
            let single = b.logical_not(bool_, None, m)?;
            b.logical_and(bool_, None, go, single)?
        } else {
            go
        };
        let (l_cp, l_cpd) = (b.id(), b.id());
        b.selection_merge(l_cpd, SelectionControl::NONE)?;
        b.branch_conditional(go, l_cp, l_cpd, [])?;
        b.begin_block(Some(l_cp))?;
        let d4 = b.shift_left_logical(u32t, None, d, c2)?;
        let d64 = b.u_convert(u64t, None, d4)?;
        let sa = b.i_add(u64t, None, msrc, d64)?;
        let sp = b.convert_u_to_ptr(p_dev, None, sa)?;
        let v = b.load(u32t, None, sp, Some(spirv::MemoryAccess::ALIGNED), [lit(4)])?;
        let da = b.i_add(u64t, None, dest, d64)?;
        let dp = b.convert_u_to_ptr(p_dev, None, da)?;
        if let Some(sh_) = diff_sh.filter(|_| same == 1) {
            // (Guest memory's copy first: differs?)
            let old = b.load(u32t, None, dp, Some(spirv::MemoryAccess::ALIGNED), [lit(4)])?;
            let ne = b.i_not_equal(bool_, None, v, old)?;
            let one = b.select(u32t, None, ne, c1, c0)?;
            b.atomic_or(u32t, None, sh_, scope_wg, sem_none, one)?;
        }
        b.store(dp, v, Some(spirv::MemoryAccess::ALIGNED), [lit(4)])?;
        b.branch(l_cpd)?;
        b.begin_block(Some(l_cpd))?;
    }
    if split {
        // Partly marked lines: invocation l of each subgroup takes line
        // l % 32 (its mask word), subgroup w its dwords w, w + 8, w + 16,
        // w + 24: each store instruction writes single dwords of different
        // lines.
        let line = bit_of;
        let mwl = mask_word(&mut b, line)?;
        let warp = b.shift_right_logical(u32t, None, lid, c5)?;
        let line32 = b.shift_left_logical(u32t, None, line, c5)?;
        for k in 0..4u32 {
            let ck = b.constant_bit32(u32t, 8 * k);
            let j = b.i_add(u32t, None, warp, ck)?;
            let sh = b.shift_right_logical(u32t, None, mwl, j)?;
            let bit = b.bitwise_and(u32t, None, sh, c1)?;
            let set = b.i_not_equal(bool_, None, bit, c0)?;
            let sec = b.shift_right_logical(u32t, None, j, c3b)?;
            let m = multi(&mut b, mwl, sec)?;
            let go = b.logical_and(bool_, None, set, m)?;
            let go = b.logical_and(bool_, None, go, store_ok)?;
            let (l_sp, l_spd) = (b.id(), b.id());
            b.selection_merge(l_spd, SelectionControl::NONE)?;
            b.branch_conditional(go, l_sp, l_spd, [])?;
            b.begin_block(Some(l_sp))?;
            let d = b.i_add(u32t, None, line32, j)?;
            let d4 = b.shift_left_logical(u32t, None, d, c2)?;
            let d64 = b.u_convert(u64t, None, d4)?;
            let sa = b.i_add(u64t, None, msrc, d64)?;
            let sp = b.convert_u_to_ptr(p_dev, None, sa)?;
            let v = b.load(u32t, None, sp, Some(spirv::MemoryAccess::ALIGNED), [lit(4)])?;
            let da = b.i_add(u64t, None, dest, d64)?;
            let dp = b.convert_u_to_ptr(p_dev, None, da)?;
            b.store(dp, v, Some(spirv::MemoryAccess::ALIGNED), [lit(4)])?;
            b.branch(l_spd)?;
            b.begin_block(Some(l_spd))?;
        }
    }
    if diff_sh.is_some() && same == 1 {
        // (Every invocation's comparisons in before invocation 0 counts.)
        let sem_sh = c(&mut b, 0x8 | 0x100);
        b.control_barrier(scope_wg, scope_wg, sem_sh)?;
    }
    // Invocation 0 counts the page's marked dwords (before they are cleared).
    let (l_cnt, l_cntd) = (b.id(), b.id());
    b.selection_merge(l_cntd, SelectionControl::NONE)?;
    b.branch_conditional(is0, l_cnt, l_cntd, [])?;
    b.begin_block(Some(l_cnt))?;
    let mut total = c0;
    for k in 0..32u32 {
        let ck = b.constant_bit32(u32t, k);
        let mw = mask_word(&mut b, ck)?;
        let n = b.bit_count(u32t, None, mw)?;
        total = b.i_add(u32t, None, total, n)?;
    }
    let which_dw = b.select(u32t, None, mapped, c1, c0)?;
    let which_dw = match held {
        Some(h) => b.select(u32t, None, h, c0, which_dw)?,
        None => which_dw,
    };
    let pdw = b.access_chain(p_u, None, counters, [c0, c1])?;
    let add = b.i_mul(u32t, None, total, which_dw)?;
    let add = match hashed {
        Some((unchanged, te_lo, te_hi, h_lo, h_hi, has_other, oe_lo, oe_hi)) => {
            // (A page stored keeps its hash for the next time, and the other
            // family's entry of it is cleared.)
            let changed = b.logical_not(bool_, None, unchanged)?;
            let keep = b.logical_and(bool_, None, changed, mapped)?;
            let (l_hk, l_hkd) = (b.id(), b.id());
            b.selection_merge(l_hkd, SelectionControl::NONE)?;
            b.branch_conditional(keep, l_hk, l_hkd, [])?;
            b.begin_block(Some(l_hk))?;
            b.store(te_lo, h_lo, Some(spirv::MemoryAccess::ALIGNED), [lit(4)])?;
            b.store(te_hi, h_hi, Some(spirv::MemoryAccess::ALIGNED), [lit(4)])?;
            let (l_oc, l_ocd) = (b.id(), b.id());
            b.selection_merge(l_ocd, SelectionControl::NONE)?;
            b.branch_conditional(has_other, l_oc, l_ocd, [])?;
            b.begin_block(Some(l_oc))?;
            b.store(oe_lo, c0, Some(spirv::MemoryAccess::ALIGNED), [lit(4)])?;
            b.store(oe_hi, c0, Some(spirv::MemoryAccess::ALIGNED), [lit(4)])?;
            b.branch(l_ocd)?;
            b.begin_block(Some(l_ocd))?;
            b.branch(l_hkd)?;
            b.begin_block(Some(l_hkd))?;
            // (A page left out: its dwords and the page in counters 4 and 5.)
            let c4 = c(&mut b, 4);
            let ud = b.select(u32t, None, unchanged, add, c0)?;
            let pd4 = b.access_chain(p_u, None, counters, [c0, c4])?;
            b.atomic_i_add(u32t, None, pd4, scope_dev, sem_none, ud)?;
            let any = b.u_greater_than(bool_, None, add, c0)?;
            let both = b.logical_and(bool_, None, unchanged, any)?;
            let up = b.select(u32t, None, both, c1, c0)?;
            let pd5 = b.access_chain(p_u, None, counters, [c0, c5])?;
            b.atomic_i_add(u32t, None, pd5, scope_dev, sem_none, up)?;
            b.select(u32t, None, unchanged, c0, add)?
        }
        None => add,
    };
    b.atomic_i_add(u32t, None, pdw, scope_dev, sem_none, add)?;
    if let Some(sh_) = diff_sh.filter(|_| same == 1) {
        // (A page stored unchanged: its dwords and the page counted.)
        let c4 = c(&mut b, 4);
        let sv = b.load(u32t, None, sh_, None, [])?;
        let unchanged = b.i_equal(bool_, None, sv, c0)?;
        let ud = b.select(u32t, None, unchanged, add, c0)?;
        let pd4 = b.access_chain(p_u, None, counters, [c0, c4])?;
        b.atomic_i_add(u32t, None, pd4, scope_dev, sem_none, ud)?;
        let any = b.u_greater_than(bool_, None, add, c0)?;
        let both = b.logical_and(bool_, None, unchanged, any)?;
        let up = b.select(u32t, None, both, c1, c0)?;
        let pd5 = b.access_chain(p_u, None, counters, [c0, c5])?;
        b.atomic_i_add(u32t, None, pd5, scope_dev, sem_none, up)?;
    }
    if hold {
        // (The page's bit in the held-page bitmap: its dwords and the page
        // counted when set.)
        let c4 = c(&mut b, 4);
        let hlo = pc_at(&mut b, c4)?;
        let hhi = pc_at(&mut b, c5)?;
        let hmap = addr64(&mut b, hlo, hhi)?;
        let hw = b.shift_right_logical(u32t, None, page, c5)?;
        let hw64 = b.u_convert(u64t, None, hw)?;
        let c4_64 = b.constant_bit64(u64t, 4);
        let hoff = b.i_mul(u64t, None, hw64, c4_64)?;
        let ha = b.i_add(u64t, None, hmap, hoff)?;
        let hp = b.convert_u_to_ptr(p_dev, None, ha)?;
        let hword = b.load(u32t, None, hp, Some(spirv::MemoryAccess::ALIGNED), [lit(4)])?;
        let hbit = b.bitwise_and(u32t, None, page, c31)?;
        let hsh = b.shift_right_logical(u32t, None, hword, hbit)?;
        let hone = b.bitwise_and(u32t, None, hsh, c1)?;
        let held_dw = b.i_mul(u32t, None, total, hone)?;
        let phd = b.access_chain(p_u, None, counters, [c0, c4])?;
        b.atomic_i_add(u32t, None, phd, scope_dev, sem_none, held_dw)?;
        let php = b.access_chain(p_u, None, counters, [c0, c5])?;
        b.atomic_i_add(u32t, None, php, scope_dev, sem_none, hone)?;
        // (Written back: the page's bit in the aux region.)
        let any = b.u_greater_than(bool_, None, total, c0)?;
        let one_at = b.shift_left_logical(u32t, None, c1, hbit)?;
        let wbit = b.select(u32t, None, any, one_at, c0)?;
        let aux_at = c(&mut b, MARK_AUX_AT);
        let awi = b.i_add(u32t, None, aux_at, hw)?;
        let paw = b.access_chain(p_u, None, marks, [c0, awi])?;
        b.atomic_or(u32t, None, paw, scope_dev, sem_none, wbit)?;
    }
    if let Some(hone) = held_one {
        // (A held page: its pending dwords and the page in counters 4 and 5;
        // a page stored with marked dwords: its bit in the aux region.)
        let c4 = c(&mut b, 4);
        let held_dw = b.i_mul(u32t, None, total, hone)?;
        let phd = b.access_chain(p_u, None, counters, [c0, c4])?;
        b.atomic_i_add(u32t, None, phd, scope_dev, sem_none, held_dw)?;
        let php = b.access_chain(p_u, None, counters, [c0, c5])?;
        b.atomic_i_add(u32t, None, php, scope_dev, sem_none, hone)?;
        let any = b.u_greater_than(bool_, None, total, c0)?;
        let not_held = b.i_equal(bool_, None, hone, c0)?;
        let stored = b.logical_and(bool_, None, any, not_held)?;
        let stored = b.logical_and(bool_, None, stored, mapped)?;
        let hw = b.shift_right_logical(u32t, None, page, c5)?;
        let hbit = b.bitwise_and(u32t, None, page, c31)?;
        let one_at = b.shift_left_logical(u32t, None, c1, hbit)?;
        let wbit = b.select(u32t, None, stored, one_at, c0)?;
        let aux_at = c(&mut b, MARK_AUX_AT);
        let awi = b.i_add(u32t, None, aux_at, hw)?;
        let paw = b.access_chain(p_u, None, marks, [c0, awi])?;
        b.atomic_or(u32t, None, paw, scope_dev, sem_none, wbit)?;
    }
    if count {
        // (The page's bucket: its dwords, and the page once when any.)
        let shift = c(&mut b, WB_BUCKET_SHIFT);
        let bucket = b.shift_right_logical(u32t, None, page, shift)?;
        let dw_at = c(&mut b, MARK_AUX_AT);
        let bi = b.i_add(u32t, None, dw_at, bucket)?;
        let pb = b.access_chain(p_u, None, marks, [c0, bi])?;
        b.atomic_i_add(u32t, None, pb, scope_dev, sem_none, add)?;
        let any = b.u_greater_than(bool_, None, add, c0)?;
        let one = b.select(u32t, None, any, c1, c0)?;
        let pg_at = c(&mut b, MARK_AUX_AT + WB_BUCKETS);
        let pi = b.i_add(u32t, None, pg_at, bucket)?;
        let pp = b.access_chain(p_u, None, marks, [c0, pi])?;
        b.atomic_i_add(u32t, None, pp, scope_dev, sem_none, one)?;
    }
    b.branch(l_cntd)?;
    b.begin_block(Some(l_cntd))?;
    // Every mask word read before any is cleared.
    b.control_barrier(scope_wg, scope_wg, sem_wg)?;
    let first32 = b.u_less_than(bool_, None, lid, c32)?;
    // (A page with no table entry keeps its mask words too: its dwords are pending, stored by
    // the CPU after the batch (the runtime's `fe2::release_unimported`) or by a later write-back
    // once the page is mapped; until 2026-11-22 night they were cleared, the dwords lost.)
    let first32 = b.logical_and(bool_, None, first32, mapped)?;
    // (A held page keeps its mask words: its dwords are pending.)
    let first32 = match held {
        Some(h) => {
            let not_held = b.logical_not(bool_, None, h)?;
            b.logical_and(bool_, None, first32, not_held)?
        }
        None => first32,
    };
    let (l_clr, l_clrd) = (b.id(), b.id());
    b.selection_merge(l_clrd, SelectionControl::NONE)?;
    b.branch_conditional(first32, l_clr, l_clrd, [])?;
    b.begin_block(Some(l_clr))?;
    let wi = b.i_add(u32t, None, first_word, lid)?;
    let pm = b.access_chain(p_u, None, marks, [c0, wi])?;
    b.store(pm, c0, None, [])?;
    b.branch(l_clrd)?;
    b.begin_block(Some(l_clrd))?;
    // Invocation 0: the page's touched bit cleared, the page counted.
    let (l_pb, l_pbd) = (b.id(), b.id());
    b.selection_merge(l_pbd, SelectionControl::NONE)?;
    b.branch_conditional(is0, l_pb, l_pbd, [])?;
    b.begin_block(Some(l_pb))?;
    {
        let pw = b.shift_right_logical(u32t, None, page, c5)?;
        let pwi = b.i_add(u32t, None, bits_at, pw)?;
        let pbit = b.bitwise_and(u32t, None, page, c31)?;
        let pmask = b.shift_left_logical(u32t, None, c1, pbit)?;
        let pinv = b.not(u32t, None, pmask)?;
        let ppb = b.access_chain(p_u, None, marks, [c0, pwi])?;
        b.atomic_and(u32t, None, ppb, scope_dev, sem_none, pinv)?;
    }
    let which = b.select(u32t, None, mapped, c0, c2)?;
    let pcnt = b.access_chain(p_u, None, counters, [c0, which])?;
    let one_page = match held {
        Some(h) => b.select(u32t, None, h, c0, c1)?,
        None => c1,
    };
    b.atomic_i_add(u32t, None, pcnt, scope_dev, sem_none, one_page)?;
    // (The last page counted lost, its number + 1 in counter 7: the runtime's status names its
    // address. Counters 6 and 7 are written by no variant otherwise.)
    {
        let lost = b.logical_not(bool_, None, mapped)?;
        let lost = match held {
            Some(h) => {
                let not_held = b.logical_not(bool_, None, h)?;
                b.logical_and(bool_, None, lost, not_held)?
            }
            None => lost,
        };
        let (l_lost, l_lostd) = (b.id(), b.id());
        b.selection_merge(l_lostd, SelectionControl::NONE)?;
        b.branch_conditional(lost, l_lost, l_lostd, [])?;
        b.begin_block(Some(l_lost))?;
        let c7l = c(&mut b, 7);
        let p7 = b.access_chain(p_u, None, counters, [c0, c7l])?;
        let v = b.i_add(u32t, None, page, c1)?;
        b.atomic_exchange(u32t, None, p7, scope_dev, sem_none, v)?;
        b.branch(l_lostd)?;
        b.begin_block(Some(l_lostd))?;
    }
    b.branch(l_pbd)?;
    b.begin_block(Some(l_pbd))?;
    b.branch(l_end)?;
    b.begin_block(Some(l_end))?;
    b.ret()?;
    b.end_function()?;
    b.entry_point(ExecutionModel::GLCompute, f, "main", [lid_v, wid_v]);
    b.execution_mode(f, ExecutionMode::LocalSize, [SCATTER_LOCAL_SIZE, 1, 1]);
    Ok(b.module().assemble())
}

/// Workgroup size of `zero_store`: invocation t stores dwords t, t + 256,
/// t + 512 and t + 768 of its page (a subgroup's stores consecutive dwords:
/// one transfer across PCIe).
pub const ZERO_STORE_LOCAL_SIZE: u32 = 256;

/// Counters of `zero_store` (at the device address in push constants 4-5):
/// pages stored, pages left out as known zero, dwords found non-zero in pages
/// left out (`flags` bit 0, audit), one unused.
pub const ZERO_STORE_COUNTERS: u32 = 4;

/// A zero fill's stores into guest memory (the runtime's `gpu::zk`):
/// workgroup g takes the g-th 4 KiB page of the fill (its first page at
/// index `first` of the span), whose bytes [lead, lead + len) relative to
/// that first page's start are zeroed; guest memory's copy of the fill's
/// first byte is at device address `dst`. A page whose bit in the
/// zero-known bitmap (`zk`: a bit per page of the span, bit i of word
/// i / 32) is set holds zeros in guest memory already (stored by an earlier
/// zero store, nothing stored into it since: every other writer of guest
/// memory clears the bit first, the write-back in its kernel) and is left
/// out; any other page has its part of the fill stored, and a page the
/// fill covers whole gets its bit. With `flags` bit 0 (audit) the dwords of
/// pages left out are read and those found non-zero counted (counter 2): a
/// writer that left the bit set. Push constants `{dst (low, high), zk (low,
/// high), counters (low, high), first, lead, len, flags}`; no bindings.
pub fn zero_store() -> Vec<u32> {
    build_zero_store().expect("zero store kernel")
}

fn build_zero_store() -> Result<Vec<u32>, rspirv::dr::Error> {
    let lit = Operand::LiteralBit32;
    let mut b = Builder::new();
    b.set_version(1, 3);
    b.capability(Capability::Shader);
    b.capability(Capability::Int64);
    b.capability(Capability::PhysicalStorageBufferAddresses);
    b.extension("SPV_KHR_physical_storage_buffer");
    b.memory_model(spirv::AddressingModel::PhysicalStorageBuffer64, spirv::MemoryModel::GLSL450);
    let void = b.type_void();
    let bool_ = b.type_bool();
    let u32t = b.type_int(32, 0);
    let u64t = b.type_int(64, 0);
    let v3 = b.type_vector(u32t, 3);
    let p_dev = b.type_pointer(None, StorageClass::PhysicalStorageBuffer, u32t);
    let pc = b.type_struct(vec![u32t; 10]);
    b.decorate(pc, Decoration::Block, []);
    for k in 0..10u32 {
        b.member_decorate(pc, k, Decoration::Offset, [lit(4 * k)]);
    }
    let p_pc = b.type_pointer(None, StorageClass::PushConstant, pc);
    let p_pcu = b.type_pointer(None, StorageClass::PushConstant, u32t);
    let p_in = b.type_pointer(None, StorageClass::Input, v3);
    let lid_v = global(&mut b, p_in, StorageClass::Input);
    b.decorate(lid_v, Decoration::BuiltIn, [Operand::BuiltIn(BuiltIn::LocalInvocationId)]);
    let wid_v = global(&mut b, p_in, StorageClass::Input);
    b.decorate(wid_v, Decoration::BuiltIn, [Operand::BuiltIn(BuiltIn::WorkgroupId)]);
    let p_wg = b.type_pointer(None, StorageClass::Workgroup, u32t);
    let known_sh = global(&mut b, p_wg, StorageClass::Workgroup);
    let pcv = global(&mut b, p_pc, StorageClass::PushConstant);
    let c = |b: &mut Builder, v: u32| b.constant_bit32(u32t, v);
    let (c0, c1, c2, c5, c31) = (c(&mut b, 0), c(&mut b, 1), c(&mut b, 2), c(&mut b, 5), c(&mut b, 31));
    let c4096 = c(&mut b, 4096);
    let c12 = c(&mut b, 12);
    let c32_64 = b.constant_bit64(u64t, 32);
    let c4_64 = b.constant_bit64(u64t, 4);
    let c8_64 = b.constant_bit64(u64t, 8);
    let scope_dev = c1;
    let scope_wg = c2;
    let sem_none = c0;
    // AcquireRelease | WorkgroupMemory: invocation 0's store seen by all.
    let sem_sh = c(&mut b, 0x8 | 0x100);
    let fty = b.type_function(void, []);
    let f = b.begin_function(void, None, FunctionControl::NONE, fty)?;
    b.begin_block(None)?;
    let l = b.load(v3, None, lid_v, None, [])?;
    let lid = b.composite_extract(u32t, None, l, [0])?;
    let w = b.load(v3, None, wid_v, None, [])?;
    let g = b.composite_extract(u32t, None, w, [0])?;
    let pc_at = |b: &mut Builder, k: u32| -> Result<Word, rspirv::dr::Error> {
        let kc = b.constant_bit32(u32t, k);
        let p = b.access_chain(p_pcu, None, pcv, [kc])?;
        b.load(u32t, None, p, None, [])
    };
    let addr64 = |b: &mut Builder, lo: Word, hi: Word| -> Result<Word, rspirv::dr::Error> {
        let lo64 = b.u_convert(u64t, None, lo)?;
        let hi64 = b.u_convert(u64t, None, hi)?;
        let hi64 = b.shift_left_logical(u64t, None, hi64, c32_64)?;
        b.bitwise_or(u64t, None, lo64, hi64)
    };
    let (dlo, dhi) = (pc_at(&mut b, 0)?, pc_at(&mut b, 1)?);
    let dst = addr64(&mut b, dlo, dhi)?;
    let (zlo, zhi) = (pc_at(&mut b, 2)?, pc_at(&mut b, 3)?);
    let zk = addr64(&mut b, zlo, zhi)?;
    let (klo, khi) = (pc_at(&mut b, 4)?, pc_at(&mut b, 5)?);
    let cnt = addr64(&mut b, klo, khi)?;
    let first = pc_at(&mut b, 6)?;
    let lead = pc_at(&mut b, 7)?;
    let len = pc_at(&mut b, 8)?;
    let flags = pc_at(&mut b, 9)?;
    // The page and the fill's bytes in it, relative to the first page.
    let page = b.i_add(u32t, None, first, g)?;
    let pstart = b.shift_left_logical(u32t, None, g, c12)?;
    let pend = b.i_add(u32t, None, pstart, c4096)?;
    let end = b.i_add(u32t, None, lead, len)?;
    let lo_ge = b.u_greater_than_equal(bool_, None, lead, pstart)?;
    let lo = b.select(u32t, None, lo_ge, lead, pstart)?;
    let hi_le = b.u_less_than_equal(bool_, None, end, pend)?;
    let hi = b.select(u32t, None, hi_le, end, pend)?;
    let full_lo = b.i_equal(bool_, None, lo, pstart)?;
    let full_hi = b.i_equal(bool_, None, hi, pend)?;
    let full = b.logical_and(bool_, None, full_lo, full_hi)?;
    let is0 = b.i_equal(bool_, None, lid, c0)?;
    // The page's bitmap word and bit.
    let zw = b.shift_right_logical(u32t, None, page, c5)?;
    let zw64 = b.u_convert(u64t, None, zw)?;
    let zoff = b.i_mul(u64t, None, zw64, c4_64)?;
    let za = b.i_add(u64t, None, zk, zoff)?;
    let zbit = b.bitwise_and(u32t, None, page, c31)?;
    let one = b.shift_left_logical(u32t, None, c1, zbit)?;
    // Invocation 0 reads the bit for the workgroup.
    let (l_r, l_rd) = (b.id(), b.id());
    b.selection_merge(l_rd, SelectionControl::NONE)?;
    b.branch_conditional(is0, l_r, l_rd, [])?;
    b.begin_block(Some(l_r))?;
    let zp = b.convert_u_to_ptr(p_dev, None, za)?;
    let word = b.load(u32t, None, zp, Some(spirv::MemoryAccess::ALIGNED), [lit(4)])?;
    let kb = b.bitwise_and(u32t, None, word, one)?;
    b.store(known_sh, kb, None, [])?;
    b.branch(l_rd)?;
    b.begin_block(Some(l_rd))?;
    b.control_barrier(scope_wg, scope_wg, sem_sh)?;
    let kv = b.load(u32t, None, known_sh, None, [])?;
    let known = b.i_not_equal(bool_, None, kv, c0)?;
    let fa = b.bitwise_and(u32t, None, flags, c1)?;
    let audit = b.i_not_equal(bool_, None, fa, c0)?;
    // A dword of the page this invocation takes (r: its round): whether the
    // fill covers it, its address in guest memory.
    let dword = |b: &mut Builder, r: u32| -> Result<(Word, Word), rspirv::dr::Error> {
        let cr = b.constant_bit32(u32t, r * ZERO_STORE_LOCAL_SIZE);
        let d = b.i_add(u32t, None, lid, cr)?;
        let d4 = b.shift_left_logical(u32t, None, d, c2)?;
        let byte = b.i_add(u32t, None, pstart, d4)?;
        let ge = b.u_greater_than_equal(bool_, None, byte, lo)?;
        let lt = b.u_less_than(bool_, None, byte, hi)?;
        let inside = b.logical_and(bool_, None, ge, lt)?;
        let rel = b.i_sub(u32t, None, byte, lead)?;
        let rel64 = b.u_convert(u64t, None, rel)?;
        let a = b.i_add(u64t, None, dst, rel64)?;
        Ok((inside, a))
    };
    let (l_known, l_store, l_done) = (b.id(), b.id(), b.id());
    b.selection_merge(l_done, SelectionControl::NONE)?;
    b.branch_conditional(known, l_known, l_store, [])?;
    // Known zero: left out (counted; the audit reads the page).
    b.begin_block(Some(l_known))?;
    {
        let (l_a, l_ad) = (b.id(), b.id());
        b.selection_merge(l_ad, SelectionControl::NONE)?;
        b.branch_conditional(audit, l_a, l_ad, [])?;
        b.begin_block(Some(l_a))?;
        for r in 0..4u32 {
            let (inside, a) = dword(&mut b, r)?;
            let (l_i, l_id) = (b.id(), b.id());
            b.selection_merge(l_id, SelectionControl::NONE)?;
            b.branch_conditional(inside, l_i, l_id, [])?;
            b.begin_block(Some(l_i))?;
            let p = b.convert_u_to_ptr(p_dev, None, a)?;
            let v = b.load(u32t, None, p, Some(spirv::MemoryAccess::ALIGNED), [lit(4)])?;
            let nz = b.i_not_equal(bool_, None, v, c0)?;
            let (l_n, l_nd) = (b.id(), b.id());
            b.selection_merge(l_nd, SelectionControl::NONE)?;
            b.branch_conditional(nz, l_n, l_nd, [])?;
            b.begin_block(Some(l_n))?;
            let ca = b.i_add(u64t, None, cnt, c8_64)?;
            let cp = b.convert_u_to_ptr(p_dev, None, ca)?;
            b.atomic_i_add(u32t, None, cp, scope_dev, sem_none, c1)?;
            b.branch(l_nd)?;
            b.begin_block(Some(l_nd))?;
            b.branch(l_id)?;
            b.begin_block(Some(l_id))?;
        }
        b.branch(l_ad)?;
        b.begin_block(Some(l_ad))?;
        let (l_c, l_cd) = (b.id(), b.id());
        b.selection_merge(l_cd, SelectionControl::NONE)?;
        b.branch_conditional(is0, l_c, l_cd, [])?;
        b.begin_block(Some(l_c))?;
        let ca = b.i_add(u64t, None, cnt, c4_64)?;
        let cp = b.convert_u_to_ptr(p_dev, None, ca)?;
        b.atomic_i_add(u32t, None, cp, scope_dev, sem_none, c1)?;
        b.branch(l_cd)?;
        b.begin_block(Some(l_cd))?;
        b.branch(l_done)?;
    }
    // Not known zero: the page's part of the fill stored; a whole page gets
    // its bit.
    b.begin_block(Some(l_store))?;
    {
        for r in 0..4u32 {
            let (inside, a) = dword(&mut b, r)?;
            let (l_i, l_id) = (b.id(), b.id());
            b.selection_merge(l_id, SelectionControl::NONE)?;
            b.branch_conditional(inside, l_i, l_id, [])?;
            b.begin_block(Some(l_i))?;
            let p = b.convert_u_to_ptr(p_dev, None, a)?;
            b.store(p, c0, Some(spirv::MemoryAccess::ALIGNED), [lit(4)])?;
            b.branch(l_id)?;
            b.begin_block(Some(l_id))?;
        }
        let (l_c, l_cd) = (b.id(), b.id());
        b.selection_merge(l_cd, SelectionControl::NONE)?;
        b.branch_conditional(is0, l_c, l_cd, [])?;
        b.begin_block(Some(l_c))?;
        let cp = b.convert_u_to_ptr(p_dev, None, cnt)?;
        b.atomic_i_add(u32t, None, cp, scope_dev, sem_none, c1)?;
        let (l_f, l_fd) = (b.id(), b.id());
        b.selection_merge(l_fd, SelectionControl::NONE)?;
        b.branch_conditional(full, l_f, l_fd, [])?;
        b.begin_block(Some(l_f))?;
        let zp2 = b.convert_u_to_ptr(p_dev, None, za)?;
        b.atomic_or(u32t, None, zp2, scope_dev, sem_none, one)?;
        b.branch(l_fd)?;
        b.begin_block(Some(l_fd))?;
        b.branch(l_cd)?;
        b.begin_block(Some(l_cd))?;
        b.branch(l_done)?;
    }
    b.begin_block(Some(l_done))?;
    b.ret()?;
    b.end_function()?;
    b.entry_point(ExecutionModel::GLCompute, f, "main", [lid_v, wid_v]);
    b.execution_mode(f, ExecutionMode::LocalSize, [ZERO_STORE_LOCAL_SIZE, 1, 1]);
    Ok(b.module().assemble())
}

/// A GPU timestamp stored in the title's GPU clock (an end-of-pipe
/// RELEASE_MEM timestamp performed by the GPU): one invocation reads the
/// 64-bit timestamp at index `query` of binding 0 (copied from a query pool,
/// nanoseconds) and stores (t + k) / 10 (100 MHz ticks) as two dwords at
/// dword `dst` of binding 1 (push constants `{query, dst, k_lo, k_hi}`; `k`
/// wraps, so it may stand for a negative offset).
pub fn timestamp_store() -> Vec<u32> {
    build_timestamp().expect("timestamp kernel")
}

fn build_timestamp() -> Result<Vec<u32>, rspirv::dr::Error> {
    let lit = Operand::LiteralBit32;
    let mut b = Builder::new();
    b.set_version(1, 3);
    b.capability(Capability::Shader);
    b.capability(Capability::Int64);
    b.extension("SPV_KHR_storage_buffer_storage_class");
    b.memory_model(spirv::AddressingModel::Logical, spirv::MemoryModel::GLSL450);
    let void = b.type_void();
    let u32t = b.type_int(32, 0);
    let u64t = b.type_int(64, 0);
    let rta64 = b.type_runtime_array(u64t);
    b.decorate(rta64, Decoration::ArrayStride, [lit(8)]);
    let st64 = b.type_struct([rta64]);
    b.decorate(st64, Decoration::Block, []);
    b.member_decorate(st64, 0, Decoration::Offset, [lit(0)]);
    let p_st64 = b.type_pointer(None, StorageClass::StorageBuffer, st64);
    let p_u64 = b.type_pointer(None, StorageClass::StorageBuffer, u64t);
    let rta = b.type_runtime_array(u32t);
    b.decorate(rta, Decoration::ArrayStride, [lit(4)]);
    let st = b.type_struct([rta]);
    b.decorate(st, Decoration::Block, []);
    b.member_decorate(st, 0, Decoration::Offset, [lit(0)]);
    let p_st = b.type_pointer(None, StorageClass::StorageBuffer, st);
    let p_u = b.type_pointer(None, StorageClass::StorageBuffer, u32t);
    let pc = b.type_struct([u32t, u32t, u32t, u32t]);
    b.decorate(pc, Decoration::Block, []);
    for i in 0..4 {
        b.member_decorate(pc, i, Decoration::Offset, [lit(4 * i)]);
    }
    let p_pc = b.type_pointer(None, StorageClass::PushConstant, pc);
    let p_pcu = b.type_pointer(None, StorageClass::PushConstant, u32t);
    let src = global(&mut b, p_st64, StorageClass::StorageBuffer);
    b.decorate(src, Decoration::DescriptorSet, [lit(0)]);
    b.decorate(src, Decoration::Binding, [lit(0)]);
    b.decorate(src, Decoration::NonWritable, []);
    let dst = global(&mut b, p_st, StorageClass::StorageBuffer);
    b.decorate(dst, Decoration::DescriptorSet, [lit(0)]);
    b.decorate(dst, Decoration::Binding, [lit(1)]);
    let pcv = global(&mut b, p_pc, StorageClass::PushConstant);
    let c: Vec<Word> = (0..4).map(|i| b.constant_bit32(u32t, i)).collect();
    let c32 = b.constant_bit64(u64t, 32);
    let c10 = b.constant_bit64(u64t, 10);
    let fty = b.type_function(void, []);
    let f = b.begin_function(void, None, FunctionControl::NONE, fty)?;
    b.begin_block(None)?;
    let mut pcs = [0; 4];
    for (i, v) in pcs.iter_mut().enumerate() {
        let p = b.access_chain(p_pcu, None, pcv, [c[i]])?;
        *v = b.load(u32t, None, p, None, [])?;
    }
    let pt = b.access_chain(p_u64, None, src, [c[0], pcs[0]])?;
    let t = b.load(u64t, None, pt, None, [])?;
    let klo = b.u_convert(u64t, None, pcs[2])?;
    let khi = b.u_convert(u64t, None, pcs[3])?;
    let khi = b.shift_left_logical(u64t, None, khi, c32)?;
    let k = b.bitwise_or(u64t, None, khi, klo)?;
    let sum = b.i_add(u64t, None, t, k)?;
    let v = b.u_div(u64t, None, sum, c10)?;
    let lo = b.u_convert(u32t, None, v)?;
    let vh = b.shift_right_logical(u64t, None, v, c32)?;
    let hi = b.u_convert(u32t, None, vh)?;
    let p0 = b.access_chain(p_u, None, dst, [c[0], pcs[1]])?;
    b.store(p0, lo, None, [])?;
    let d1 = b.i_add(u32t, None, pcs[1], c[1])?;
    let p1 = b.access_chain(p_u, None, dst, [c[0], d1])?;
    b.store(p1, hi, None, [])?;
    b.ret()?;
    b.end_function()?;
    b.entry_point(ExecutionModel::GLCompute, f, "main", []);
    b.execution_mode(f, ExecutionMode::LocalSize, [1, 1, 1]);
    Ok(b.module().assemble())
}

/// Timestamps `timestamp_store_many` stores per dispatch (its push
/// constants: 128 bytes, the guaranteed minimum).
pub const TS_MANY: u32 = 14;

/// Up to `TS_MANY` GPU timestamps stored in the title's GPU clock by one
/// dispatch (`timestamp_store` stores one): invocation i < count reads the
/// 64-bit timestamp at index q_i of binding 0 and stores (t + k) / 10 as two
/// dwords at dword d_i of binding 1 (push constants
/// `{k_lo, k_hi, count, 0, q_0, d_0, ..., q_13, d_13}`).
pub fn timestamp_store_many() -> Vec<u32> {
    build_timestamp_many().expect("timestamp kernel (many)")
}

fn build_timestamp_many() -> Result<Vec<u32>, rspirv::dr::Error> {
    let lit = Operand::LiteralBit32;
    let mut b = Builder::new();
    b.set_version(1, 3);
    b.capability(Capability::Shader);
    b.capability(Capability::Int64);
    b.extension("SPV_KHR_storage_buffer_storage_class");
    b.memory_model(spirv::AddressingModel::Logical, spirv::MemoryModel::GLSL450);
    let void = b.type_void();
    let bool_ = b.type_bool();
    let u32t = b.type_int(32, 0);
    let u64t = b.type_int(64, 0);
    let v3 = b.type_vector(u32t, 3);
    let rta64 = b.type_runtime_array(u64t);
    b.decorate(rta64, Decoration::ArrayStride, [lit(8)]);
    let st64 = b.type_struct([rta64]);
    b.decorate(st64, Decoration::Block, []);
    b.member_decorate(st64, 0, Decoration::Offset, [lit(0)]);
    let p_st64 = b.type_pointer(None, StorageClass::StorageBuffer, st64);
    let p_u64 = b.type_pointer(None, StorageClass::StorageBuffer, u64t);
    let rta = b.type_runtime_array(u32t);
    b.decorate(rta, Decoration::ArrayStride, [lit(4)]);
    let st = b.type_struct([rta]);
    b.decorate(st, Decoration::Block, []);
    b.member_decorate(st, 0, Decoration::Offset, [lit(0)]);
    let p_st = b.type_pointer(None, StorageClass::StorageBuffer, st);
    let p_u = b.type_pointer(None, StorageClass::StorageBuffer, u32t);
    let n_words = b.constant_bit32(u32t, 4 + 2 * TS_MANY);
    let arr = b.type_array(u32t, n_words);
    b.decorate(arr, Decoration::ArrayStride, [lit(4)]);
    let pc = b.type_struct([arr]);
    b.decorate(pc, Decoration::Block, []);
    b.member_decorate(pc, 0, Decoration::Offset, [lit(0)]);
    let p_pc = b.type_pointer(None, StorageClass::PushConstant, pc);
    let p_pcu = b.type_pointer(None, StorageClass::PushConstant, u32t);
    let p_in = b.type_pointer(None, StorageClass::Input, v3);
    let lid = global(&mut b, p_in, StorageClass::Input);
    b.decorate(lid, Decoration::BuiltIn, [Operand::BuiltIn(BuiltIn::LocalInvocationId)]);
    let src = global(&mut b, p_st64, StorageClass::StorageBuffer);
    b.decorate(src, Decoration::DescriptorSet, [lit(0)]);
    b.decorate(src, Decoration::Binding, [lit(0)]);
    b.decorate(src, Decoration::NonWritable, []);
    let dst = global(&mut b, p_st, StorageClass::StorageBuffer);
    b.decorate(dst, Decoration::DescriptorSet, [lit(0)]);
    b.decorate(dst, Decoration::Binding, [lit(1)]);
    let pcv = global(&mut b, p_pc, StorageClass::PushConstant);
    let c: Vec<Word> = (0..6).map(|i| b.constant_bit32(u32t, i)).collect();
    let c32 = b.constant_bit64(u64t, 32);
    let c10 = b.constant_bit64(u64t, 10);
    let fty = b.type_function(void, []);
    let f = b.begin_function(void, None, FunctionControl::NONE, fty)?;
    b.begin_block(None)?;
    let l = b.load(v3, None, lid, None, [])?;
    let x = b.composite_extract(u32t, None, l, [0])?;
    let pn = b.access_chain(p_pcu, None, pcv, [c[0], c[2]])?;
    let count = b.load(u32t, None, pn, None, [])?;
    let inside = b.u_less_than(bool_, None, x, count)?;
    let (l_body, l_end) = (b.id(), b.id());
    b.selection_merge(l_end, SelectionControl::NONE)?;
    b.branch_conditional(inside, l_body, l_end, [])?;
    b.begin_block(Some(l_body))?;
    let x2 = b.i_mul(u32t, None, x, c[2])?;
    let iq = b.i_add(u32t, None, x2, c[4])?;
    let id_ = b.i_add(u32t, None, x2, c[5])?;
    let pq = b.access_chain(p_pcu, None, pcv, [c[0], iq])?;
    let q = b.load(u32t, None, pq, None, [])?;
    let pd = b.access_chain(p_pcu, None, pcv, [c[0], id_])?;
    let d = b.load(u32t, None, pd, None, [])?;
    let pklo = b.access_chain(p_pcu, None, pcv, [c[0], c[0]])?;
    let klo = b.load(u32t, None, pklo, None, [])?;
    let pkhi = b.access_chain(p_pcu, None, pcv, [c[0], c[1]])?;
    let khi = b.load(u32t, None, pkhi, None, [])?;
    let klo = b.u_convert(u64t, None, klo)?;
    let khi = b.u_convert(u64t, None, khi)?;
    let khi = b.shift_left_logical(u64t, None, khi, c32)?;
    let k = b.bitwise_or(u64t, None, khi, klo)?;
    let pt = b.access_chain(p_u64, None, src, [c[0], q])?;
    let t = b.load(u64t, None, pt, None, [])?;
    let sum = b.i_add(u64t, None, t, k)?;
    let v = b.u_div(u64t, None, sum, c10)?;
    let lo = b.u_convert(u32t, None, v)?;
    let vh = b.shift_right_logical(u64t, None, v, c32)?;
    let hi = b.u_convert(u32t, None, vh)?;
    let p0 = b.access_chain(p_u, None, dst, [c[0], d])?;
    b.store(p0, lo, None, [])?;
    let d1 = b.i_add(u32t, None, d, c[1])?;
    let p1 = b.access_chain(p_u, None, dst, [c[0], d1])?;
    b.store(p1, hi, None, [])?;
    b.branch(l_end)?;
    b.begin_block(Some(l_end))?;
    b.ret()?;
    b.end_function()?;
    b.entry_point(ExecutionModel::GLCompute, f, "main", [lid]);
    b.execution_mode(f, ExecutionMode::LocalSize, [16, 1, 1]);
    Ok(b.module().assemble())
}

#[cfg(test)]
mod tests {
    #[test]
    fn ui_alpha_kernel_assembles() {
        let w = super::ui_alpha();
        assert_eq!(w[0], 0x0723_0203);
        if let Ok(dir) = std::env::var("RDNA_KERNEL_DUMP") {
            let bytes: Vec<u8> = w.iter().flat_map(|x| x.to_le_bytes()).collect();
            std::fs::write(std::path::Path::new(&dir).join("ui_alpha.spv"), bytes).unwrap();
        }
    }

    #[test]
    fn sharpen_kernel_assembles() {
        let w = super::sharpen();
        assert_eq!(w[0], 0x0723_0203);
        if let Ok(dir) = std::env::var("RDNA_KERNEL_DUMP") {
            let bytes: Vec<u8> = w.iter().flat_map(|x| x.to_le_bytes()).collect();
            std::fs::write(std::path::Path::new(&dir).join("sharpen.spv"), bytes).unwrap();
        }
    }

    #[test]
    fn detile_kernel_assembles() {
        let w = super::detile();
        assert_eq!(w[0], 0x0723_0203);
        if let Ok(dir) = std::env::var("RDNA_KERNEL_DUMP") {
            let bytes: Vec<u8> = w.iter().flat_map(|x| x.to_le_bytes()).collect();
            std::fs::write(std::path::Path::new(&dir).join("detile.spv"), bytes).unwrap();
        }
    }

    #[test]
    fn timestamp_many_kernel_assembles() {
        let w = super::timestamp_store_many();
        assert_eq!(w[0], 0x0723_0203);
        if let Ok(dir) = std::env::var("RDNA_KERNEL_DUMP") {
            let bytes: Vec<u8> = w.iter().flat_map(|x| x.to_le_bytes()).collect();
            std::fs::write(std::path::Path::new(&dir).join("timestamp_store_many.spv"), bytes).unwrap();
        }
    }

    #[test]
    fn timestamp_kernel_assembles() {
        let w = super::timestamp_store();
        assert_eq!(w[0], 0x0723_0203);
        if let Ok(dir) = std::env::var("RDNA_KERNEL_DUMP") {
            let bytes: Vec<u8> = w.iter().flat_map(|x| x.to_le_bytes()).collect();
            std::fs::write(std::path::Path::new(&dir).join("timestamp_store.spv"), bytes).unwrap();
        }
    }

    #[test]
    fn gather_kernel_assembles() {
        let w = super::mirror_gather();
        assert_eq!(w[0], 0x0723_0203);
        if let Ok(dir) = std::env::var("RDNA_KERNEL_DUMP") {
            let bytes: Vec<u8> = w.iter().flat_map(|x| x.to_le_bytes()).collect();
            std::fs::write(std::path::Path::new(&dir).join("mirror_gather.spv"), bytes).unwrap();
        }
    }

    #[test]
    fn zero_store_kernel_assembles() {
        let w = super::zero_store();
        assert_eq!(w[0], 0x0723_0203);
        if let Ok(dir) = std::env::var("RDNA_KERNEL_DUMP") {
            let bytes: Vec<u8> = w.iter().flat_map(|x| x.to_le_bytes()).collect();
            std::fs::write(std::path::Path::new(&dir).join("zero_store.spv"), bytes).unwrap();
        }
    }

    #[test]
    fn scatter_kernel_assembles() {
        let w = super::mirror_scatter();
        assert_eq!(w[0], 0x0723_0203);
        if let Ok(dir) = std::env::var("RDNA_KERNEL_DUMP") {
            let bytes: Vec<u8> = w.iter().flat_map(|x| x.to_le_bytes()).collect();
            std::fs::write(std::path::Path::new(&dir).join("mirror_scatter.spv"), bytes).unwrap();
        }
    }

    #[test]
    fn scatter_variants_assemble() {
        let all = [
            ("mirror_scatter_reset", super::mirror_scatter_reset()),
            ("mirror_scatter_counting", super::mirror_scatter_counting()),
            ("mirror_scatter_hold_census", super::mirror_scatter_hold_census()),
            ("mirror_scatter_split", super::mirror_scatter_split()),
            ("mirror_scatter_held", super::mirror_scatter_held()),
        ];
        for (name, w) in all {
            assert_eq!(w[0], 0x0723_0203);
            if let Ok(dir) = std::env::var("RDNA_KERNEL_DUMP") {
                let bytes: Vec<u8> = w.iter().flat_map(|x| x.to_le_bytes()).collect();
                std::fs::write(std::path::Path::new(&dir).join(format!("{name}.spv")), bytes).unwrap();
            }
        }
    }

    #[test]
    fn scatter_same_skip_kernel_assembles() {
        let w = super::mirror_scatter_same_skip();
        assert_eq!(w[0], 0x0723_0203);
        assert_ne!(w, super::mirror_scatter_same_census());
        if let Ok(dir) = std::env::var("RDNA_KERNEL_DUMP") {
            let bytes: Vec<u8> = w.iter().flat_map(|x| x.to_le_bytes()).collect();
            std::fs::write(std::path::Path::new(&dir).join("mirror_scatter_same_skip.spv"), bytes).unwrap();
        }
    }

    #[test]
    fn scatter_same_census_kernel_assembles() {
        let w = super::mirror_scatter_same_census();
        assert_eq!(w[0], 0x0723_0203);
        assert_ne!(w, super::mirror_scatter_reset());
        if let Ok(dir) = std::env::var("RDNA_KERNEL_DUMP") {
            let bytes: Vec<u8> = w.iter().flat_map(|x| x.to_le_bytes()).collect();
            std::fs::write(std::path::Path::new(&dir).join("mirror_scatter_same_census.spv"), bytes).unwrap();
        }
    }

    #[test]
    fn scatter_held_kernel_assembles() {
        let w = super::mirror_scatter_held();
        assert_eq!(w[0], 0x0723_0203);
        assert_ne!(w, super::mirror_scatter_split());
        if let Ok(dir) = std::env::var("RDNA_KERNEL_DUMP") {
            let bytes: Vec<u8> = w.iter().flat_map(|x| x.to_le_bytes()).collect();
            std::fs::write(std::path::Path::new(&dir).join("mirror_scatter_held.spv"), bytes).unwrap();
        }
    }

    #[test]
    fn scatter_hold_census_kernel_assembles() {
        let w = super::mirror_scatter_hold_census();
        assert_eq!(w[0], 0x0723_0203);
        assert_ne!(w, super::mirror_scatter_reset());
        if let Ok(dir) = std::env::var("RDNA_KERNEL_DUMP") {
            let bytes: Vec<u8> = w.iter().flat_map(|x| x.to_le_bytes()).collect();
            std::fs::write(std::path::Path::new(&dir).join("mirror_scatter_hold_census.spv"), bytes).unwrap();
        }
    }

    #[test]
    fn scatter_reset_kernel_assembles() {
        let w = super::mirror_scatter_reset();
        assert_eq!(w[0], 0x0723_0203);
        assert_ne!(w, super::mirror_scatter());
        if let Ok(dir) = std::env::var("RDNA_KERNEL_DUMP") {
            let bytes: Vec<u8> = w.iter().flat_map(|x| x.to_le_bytes()).collect();
            std::fs::write(std::path::Path::new(&dir).join("mirror_scatter_reset.spv"), bytes).unwrap();
        }
    }

    #[test]
    fn scatter_split_kernel_assembles() {
        let w = super::mirror_scatter_split();
        assert_eq!(w[0], 0x0723_0203);
        assert_ne!(w, super::mirror_scatter_reset());
        if let Ok(dir) = std::env::var("RDNA_KERNEL_DUMP") {
            let bytes: Vec<u8> = w.iter().flat_map(|x| x.to_le_bytes()).collect();
            std::fs::write(std::path::Path::new(&dir).join("mirror_scatter_split.spv"), bytes).unwrap();
        }
    }

    #[test]
    fn scatter_counting_kernel_assembles() {
        let w = super::mirror_scatter_counting();
        assert_eq!(w[0], 0x0723_0203);
        assert_ne!(w, super::mirror_scatter_reset());
        if let Ok(dir) = std::env::var("RDNA_KERNEL_DUMP") {
            let bytes: Vec<u8> = w.iter().flat_map(|x| x.to_le_bytes()).collect();
            std::fs::write(std::path::Path::new(&dir).join("mirror_scatter_counting.spv"), bytes).unwrap();
        }
    }

    #[test]
    fn merge_kernel_assembles() {
        let w = super::shadow_merge();
        assert_eq!(w[0], 0x0723_0203);
        if let Ok(dir) = std::env::var("RDNA_KERNEL_DUMP") {
            let bytes: Vec<u8> = w.iter().flat_map(|x| x.to_le_bytes()).collect();
            std::fs::write(std::path::Path::new(&dir).join("shadow_merge.spv"), bytes).unwrap();
        }
    }
    #[test]
    fn merge_many_kernel_assembles() {
        let w = super::shadow_merge_many();
        assert_eq!(w[0], 0x0723_0203);
        if let Ok(dir) = std::env::var("RDNA_KERNEL_DUMP") {
            let bytes: Vec<u8> = w.iter().flat_map(|x| x.to_le_bytes()).collect();
            std::fs::write(std::path::Path::new(&dir).join("shadow_merge_many.spv"), bytes).unwrap();
        }
    }
}

/// Workgroup size of `detile`.
pub const DETILE_LOCAL_SIZE: u32 = 64;
/// Push-constant words of `detile` (112 bytes, under the guaranteed 128).
pub const DETILE_PC_WORDS: usize = 28;

/// Texture detiling on the GPU (front end 2 stage F4): one (mip, layer) of a
/// gfx10 thin surface, tiled in binding 0 (u32 words from the part's start
/// at byte `src_off`), written to binding 1 as tightly packed rows of
/// elements from dword `dst_off`. Invocation o (row `gid.y` of `stride`
/// invocations) writes output dword o < `count`. An element's byte offset
/// is `src_off` + its block's ((y >> lh) * pitch_blocks + (x >> lw)) <<
/// block_log2 + the XOR over address bits i of the parities of x mask_i &
/// (x % bw) and y mask_i & (y % bh) (the CPU detiler's `xs[x] ^ ys[y]`),
/// with (x, y) the element's coordinates plus the mip-tail offsets (tx,
/// ty); bytes from `src_len` (a multiple of 4) on read as zero. Push constants: {src_off,
/// dst_off, w, h, tx, ty, pitch_blocks, block_log2 | lw << 8 | lh << 16 |
/// elem_log2 << 24, count, stride, src_len, a word XORed into every element's
/// in-block offset (0 for a 2D part; a thick 3D slice's depth term, its slab's
/// offset in `src_off`), then per address bit i < 16 x mask | y mask << 16}.
pub fn detile() -> Vec<u32> {
    build_detile().expect("detile kernel")
}

fn build_detile() -> Result<Vec<u32>, rspirv::dr::Error> {
    let lit = Operand::LiteralBit32;
    let mut b = Builder::new();
    b.set_version(1, 3);
    b.capability(Capability::Shader);
    b.extension("SPV_KHR_storage_buffer_storage_class");
    b.memory_model(spirv::AddressingModel::Logical, spirv::MemoryModel::GLSL450);
    let void = b.type_void();
    let bool_ = b.type_bool();
    let u32t = b.type_int(32, 0);
    let v3 = b.type_vector(u32t, 3);
    let rta = b.type_runtime_array(u32t);
    b.decorate(rta, Decoration::ArrayStride, [lit(4)]);
    let st = b.type_struct([rta]);
    b.decorate(st, Decoration::Block, []);
    b.member_decorate(st, 0, Decoration::Offset, [lit(0)]);
    let p_st = b.type_pointer(None, StorageClass::StorageBuffer, st);
    let p_u = b.type_pointer(None, StorageClass::StorageBuffer, u32t);
    let pc = b.type_struct(vec![u32t; DETILE_PC_WORDS]);
    b.decorate(pc, Decoration::Block, []);
    for i in 0..DETILE_PC_WORDS as u32 {
        b.member_decorate(pc, i, Decoration::Offset, [lit(4 * i)]);
    }
    let p_pc = b.type_pointer(None, StorageClass::PushConstant, pc);
    let p_pcu = b.type_pointer(None, StorageClass::PushConstant, u32t);
    let p_in = b.type_pointer(None, StorageClass::Input, v3);
    let gid = global(&mut b, p_in, StorageClass::Input);
    b.decorate(gid, Decoration::BuiltIn, [Operand::BuiltIn(BuiltIn::GlobalInvocationId)]);
    let src = global(&mut b, p_st, StorageClass::StorageBuffer);
    b.decorate(src, Decoration::DescriptorSet, [lit(0)]);
    b.decorate(src, Decoration::Binding, [lit(0)]);
    b.decorate(src, Decoration::NonWritable, []);
    let dst = global(&mut b, p_st, StorageClass::StorageBuffer);
    b.decorate(dst, Decoration::DescriptorSet, [lit(0)]);
    b.decorate(dst, Decoration::Binding, [lit(1)]);
    let pcv = global(&mut b, p_pc, StorageClass::PushConstant);
    let k: Vec<Word> = (0..=32).map(|i| b.constant_bit32(u32t, i)).collect();
    let kff = b.constant_bit32(u32t, 0xFF);
    let kffff = b.constant_bit32(u32t, 0xFFFF);
    let fty = b.type_function(void, []);
    let f = b.begin_function(void, None, FunctionControl::NONE, fty)?;
    b.begin_block(None)?;
    let mut pcw = Vec::with_capacity(DETILE_PC_WORDS);
    for i in 0..DETILE_PC_WORDS {
        let p = b.access_chain(p_pcu, None, pcv, [k[i]])?;
        pcw.push(b.load(u32t, None, p, None, [])?);
    }
    let (src_off, dst_off, w, h, tx, ty, pitch, shifts, count, stride, src_len) = (pcw[0], pcw[1], pcw[2], pcw[3], pcw[4], pcw[5], pcw[6], pcw[7], pcw[8], pcw[9], pcw[10]);
    let g = b.load(v3, None, gid, None, [])?;
    let gx = b.composite_extract(u32t, None, g, [0])?;
    let gy = b.composite_extract(u32t, None, g, [1])?;
    let row = b.i_mul(u32t, None, gy, stride)?;
    let o = b.i_add(u32t, None, row, gx)?;
    let inside = b.u_less_than(bool_, None, o, count)?;
    let (l_body, l_end) = (b.id(), b.id());
    b.selection_merge(l_end, SelectionControl::NONE)?;
    b.branch_conditional(inside, l_body, l_end, [])?;
    b.begin_block(Some(l_body))?;
    let bl = b.bitwise_and(u32t, None, shifts, kff)?;
    let s8 = b.shift_right_logical(u32t, None, shifts, k[8])?;
    let lw = b.bitwise_and(u32t, None, s8, kff)?;
    let s16 = b.shift_right_logical(u32t, None, shifts, k[16])?;
    let lh = b.bitwise_and(u32t, None, s16, kff)?;
    let e = b.shift_right_logical(u32t, None, shifts, k[24])?;
    let bw = b.shift_left_logical(u32t, None, k[1], lw)?;
    let mw = b.i_sub(u32t, None, bw, k[1])?;
    let bh = b.shift_left_logical(u32t, None, k[1], lh)?;
    let mh = b.i_sub(u32t, None, bh, k[1])?;
    let total = b.i_mul(u32t, None, w, h)?;
    // Byte offset of element `el` in binding 0.
    let offset = |b: &mut Builder, el: Word| -> Result<Word, rspirv::dr::Error> {
        let x = b.u_mod(u32t, None, el, w)?;
        let y = b.u_div(u32t, None, el, w)?;
        let x = b.i_add(u32t, None, x, tx)?;
        let y = b.i_add(u32t, None, y, ty)?;
        let by = b.shift_right_logical(u32t, None, y, lh)?;
        let bx = b.shift_right_logical(u32t, None, x, lw)?;
        let r = b.i_mul(u32t, None, by, pitch)?;
        let blk = b.i_add(u32t, None, r, bx)?;
        let blk = b.shift_left_logical(u32t, None, blk, bl)?;
        let xm = b.bitwise_and(u32t, None, x, mw)?;
        let ym = b.bitwise_and(u32t, None, y, mh)?;
        let mut inner = k[0];
        for i in 0..16 {
            let m = pcw[12 + i];
            let mx = b.bitwise_and(u32t, None, m, kffff)?;
            let my = b.shift_right_logical(u32t, None, m, k[16])?;
            let ax = b.bitwise_and(u32t, None, mx, xm)?;
            let ay = b.bitwise_and(u32t, None, my, ym)?;
            let cx = b.bit_count(u32t, None, ax)?;
            let cy = b.bit_count(u32t, None, ay)?;
            let c = b.i_add(u32t, None, cx, cy)?;
            let p = b.bitwise_and(u32t, None, c, k[1])?;
            let p = b.shift_left_logical(u32t, None, p, k[i])?;
            inner = b.bitwise_or(u32t, None, inner, p)?;
        }
        let inner = b.bitwise_xor(u32t, None, inner, pcw[11])?;
        let off = b.i_add(u32t, None, src_off, blk)?;
        b.i_add(u32t, None, off, inner)
    };
    // The dword holding byte `off` (0 when that dword is not below
    // `src_len`, a multiple of 4, or when `ok` is false).
    let src_words = b.shift_right_logical(u32t, None, src_len, k[2])?;
    let word_at = |b: &mut Builder, off: Word, ok: Word| -> Result<Word, rspirv::dr::Error> {
        let idx = b.shift_right_logical(u32t, None, off, k[2])?;
        let inb = b.u_less_than(bool_, None, idx, src_words)?;
        let inb = b.logical_and(bool_, None, inb, ok)?;
        let idx = b.select(u32t, None, inb, idx, k[0])?;
        let p = b.access_chain(p_u, None, src, [k[0], idx])?;
        let v = b.load(u32t, None, p, None, [])?;
        b.select(u32t, None, inb, v, k[0])
    };
    let wide = b.u_greater_than_equal(bool_, None, e, k[2])?;
    let (l_wide, l_narrow, l_join) = (b.id(), b.id(), b.id());
    b.selection_merge(l_join, SelectionControl::NONE)?;
    b.branch_conditional(wide, l_wide, l_narrow, [])?;
    // Elements of 4, 8 or 16 bytes: dword (o mod n) of element o / n, n
    // their dwords.
    b.begin_block(Some(l_wide))?;
    let q = b.i_sub(u32t, None, e, k[2])?;
    let el = b.shift_right_logical(u32t, None, o, q)?;
    let n = b.shift_left_logical(u32t, None, k[1], q)?;
    let nm = b.i_sub(u32t, None, n, k[1])?;
    let kd = b.bitwise_and(u32t, None, o, nm)?;
    let kb = b.shift_left_logical(u32t, None, kd, k[2])?;
    let base = offset(&mut b, el)?;
    let off = b.i_add(u32t, None, base, kb)?;
    let ok = b.u_less_than(bool_, None, el, total)?;
    let v_wide = word_at(&mut b, off, ok)?;
    b.branch(l_join)?;
    // Elements of 1 or 2 bytes: the 4 >> e elements from o << (2 - e).
    b.begin_block(Some(l_narrow))?;
    let per = b.shift_right_logical(u32t, None, k[4], e)?;
    let sh = b.i_sub(u32t, None, k[2], e)?;
    let first = b.shift_left_logical(u32t, None, o, sh)?;
    let bits = b.shift_left_logical(u32t, None, k[8], e)?;
    let one_bits = b.shift_left_logical(u32t, None, k[1], bits)?;
    let emask = b.i_sub(u32t, None, one_bits, k[1])?;
    let e3 = b.i_add(u32t, None, e, k[3])?;
    let mut acc = k[0];
    for j in 0..4 {
        let el = b.i_add(u32t, None, first, k[j])?;
        let in_word = b.u_less_than(bool_, None, k[j], per)?;
        let in_part = b.u_less_than(bool_, None, el, total)?;
        let ok = b.logical_and(bool_, None, in_word, in_part)?;
        let off = offset(&mut b, el)?;
        let wd = word_at(&mut b, off, ok)?;
        let lo = b.bitwise_and(u32t, None, off, k[3])?;
        let lo = b.shift_left_logical(u32t, None, lo, k[3])?;
        let v = b.shift_right_logical(u32t, None, wd, lo)?;
        let v = b.bitwise_and(u32t, None, v, emask)?;
        // (An element past the word's `per` shifts by 0: 2-byte elements 2 and 3 would shift by
        // 32 and 48, which SPIR-V leaves undefined; their value is 0 either way.)
        let at = b.shift_left_logical(u32t, None, k[j], e3)?;
        let at = b.select(u32t, None, in_word, at, k[0])?;
        let v = b.shift_left_logical(u32t, None, v, at)?;
        acc = b.bitwise_or(u32t, None, acc, v)?;
    }
    let v_narrow = acc;
    b.branch(l_join)?;
    b.begin_block(Some(l_join))?;
    let v = b.phi(u32t, None, [(v_wide, l_wide), (v_narrow, l_narrow)])?;
    let di = b.i_add(u32t, None, dst_off, o)?;
    let pd = b.access_chain(p_u, None, dst, [k[0], di])?;
    b.store(pd, v, None, [])?;
    b.branch(l_end)?;
    b.begin_block(Some(l_end))?;
    b.ret()?;
    b.end_function()?;
    b.entry_point(ExecutionModel::GLCompute, f, "main", [gid]);
    b.execution_mode(f, ExecutionMode::LocalSize, [DETILE_LOCAL_SIZE, 1, 1]);
    Ok(b.module().assemble())
}

/// Workgroup size of `latency_test`.
pub const LATENCY_LOCAL_SIZE: u32 = 64;

/// A test kernel (`dsport --dispatch-bench`): each invocation runs `iters`
/// (push constant 0) steps of a dependent integer chain seeded by its index
/// and stores the result at dword `base + index` of binding 0 (push constant
/// 1). `salt` is baked into the code: kernels with different salts are
/// different programs (and pipelines).
pub fn latency_test(salt: u32) -> Vec<u32> {
    build_latency(salt).expect("latency test kernel")
}

fn build_latency(salt: u32) -> Result<Vec<u32>, rspirv::dr::Error> {
    let lit = Operand::LiteralBit32;
    let mut b = Builder::new();
    b.set_version(1, 3);
    b.capability(Capability::Shader);
    b.extension("SPV_KHR_storage_buffer_storage_class");
    b.memory_model(spirv::AddressingModel::Logical, spirv::MemoryModel::GLSL450);
    let void = b.type_void();
    let bool_ = b.type_bool();
    let u32t = b.type_int(32, 0);
    let v3 = b.type_vector(u32t, 3);
    let rta = b.type_runtime_array(u32t);
    b.decorate(rta, Decoration::ArrayStride, [lit(4)]);
    let st = b.type_struct([rta]);
    b.decorate(st, Decoration::Block, []);
    b.member_decorate(st, 0, Decoration::Offset, [lit(0)]);
    let p_st = b.type_pointer(None, StorageClass::StorageBuffer, st);
    let p_u = b.type_pointer(None, StorageClass::StorageBuffer, u32t);
    let pc = b.type_struct([u32t, u32t]);
    b.decorate(pc, Decoration::Block, []);
    b.member_decorate(pc, 0, Decoration::Offset, [lit(0)]);
    b.member_decorate(pc, 1, Decoration::Offset, [lit(4)]);
    let p_pc = b.type_pointer(None, StorageClass::PushConstant, pc);
    let p_pcu = b.type_pointer(None, StorageClass::PushConstant, u32t);
    let p_in = b.type_pointer(None, StorageClass::Input, v3);
    let gid = global(&mut b, p_in, StorageClass::Input);
    b.decorate(gid, Decoration::BuiltIn, [Operand::BuiltIn(BuiltIn::GlobalInvocationId)]);
    let buf = global(&mut b, p_st, StorageClass::StorageBuffer);
    b.decorate(buf, Decoration::DescriptorSet, [lit(0)]);
    b.decorate(buf, Decoration::Binding, [lit(0)]);
    let pcv = global(&mut b, p_pc, StorageClass::PushConstant);
    let c0 = b.constant_bit32(u32t, 0);
    let c1 = b.constant_bit32(u32t, 1);
    let cm = b.constant_bit32(u32t, 1664525);
    let ca = b.constant_bit32(u32t, 1013904223u32.wrapping_add(salt.wrapping_mul(2)));
    let fty = b.type_function(void, []);
    let f = b.begin_function(void, None, FunctionControl::NONE, fty)?;
    let l_entry = b.begin_block(None)?;
    let g = b.load(v3, None, gid, None, [])?;
    let x0 = b.composite_extract(u32t, None, g, [0])?;
    let p_it = b.access_chain(p_pcu, None, pcv, [c0])?;
    let iters = b.load(u32t, None, p_it, None, [])?;
    let p_base = b.access_chain(p_pcu, None, pcv, [c1])?;
    let base = b.load(u32t, None, p_base, None, [])?;
    let (l_head, l_body, l_cont, l_merge) = (b.id(), b.id(), b.id(), b.id());
    let (x_next, i_next) = (b.id(), b.id());
    b.branch(l_head)?;
    b.begin_block(Some(l_head))?;
    let i = b.phi(u32t, None, [(c0, l_entry), (i_next, l_cont)])?;
    let x = b.phi(u32t, None, [(x0, l_entry), (x_next, l_cont)])?;
    b.loop_merge(l_merge, l_cont, spirv::LoopControl::NONE, [])?;
    let more = b.u_less_than(bool_, None, i, iters)?;
    b.branch_conditional(more, l_body, l_merge, [])?;
    b.begin_block(Some(l_body))?;
    let m = b.i_mul(u32t, None, x, cm)?;
    b.i_add(u32t, Some(x_next), m, ca)?;
    b.branch(l_cont)?;
    b.begin_block(Some(l_cont))?;
    b.i_add(u32t, Some(i_next), i, c1)?;
    b.branch(l_head)?;
    b.begin_block(Some(l_merge))?;
    let idx = b.i_add(u32t, None, base, x0)?;
    let p = b.access_chain(p_u, None, buf, [c0, idx])?;
    b.store(p, x, None, [])?;
    b.ret()?;
    b.end_function()?;
    b.entry_point(ExecutionModel::GLCompute, f, "main", [gid]);
    b.execution_mode(f, ExecutionMode::LocalSize, [LATENCY_LOCAL_SIZE, 1, 1]);
    Ok(b.module().assemble())
}

/// Workgroup size (each axis) of `sharpen`.
pub const SHARPEN_LOCAL: u32 = 8;

/// The PC settings' sharpening of the presented image: binding 0 the image to sharpen, binding
/// 1 the result (storage images, rgba16f), push constants {width, height, amount}. Each pixel
/// moves away from the mean of its four neighbours by `amount` times the difference, kept within
/// the range of its 3x3 neighbourhood (so edges get no halo); alpha 1.
/// The upscalers' merge (`gpu::upscale`): each pixel of the title's TAA output (binding 1,
/// RGBA16F storage image) takes the upscaler's colour (binding 0) times the exposure the TAA
/// applies to its input (the first float of binding 2, a storage buffer: the TAA's output is
/// pre-exposed) and keeps its own alpha, which the TAA packs with per-pixel data later passes
/// read. Push constants: width, height.
pub fn rgb_merge() -> Vec<u32> {
    build_rgb_merge().expect("rgb merge kernel")
}

/// The merge kernel's workgroup side.
pub const RGB_MERGE_LOCAL: u32 = 8;

fn build_rgb_merge() -> Result<Vec<u32>, rspirv::dr::Error> {
    use rspirv::spirv::{Dim, ImageFormat};
    let lit = Operand::LiteralBit32;
    let mut b = Builder::new();
    b.set_version(1, 3);
    b.capability(Capability::Shader);
    b.memory_model(spirv::AddressingModel::Logical, spirv::MemoryModel::GLSL450);
    let void = b.type_void();
    let bool_ = b.type_bool();
    let u32t = b.type_int(32, 0);
    let i32t = b.type_int(32, 1);
    let f32t = b.type_float(32, None);
    let v3u = b.type_vector(u32t, 3);
    let v2i = b.type_vector(i32t, 2);
    let v4f = b.type_vector(f32t, 4);
    let img = b.type_image(f32t, Dim::Dim2D, 0, 0, 0, 2, ImageFormat::Rgba16f, None);
    let p_img = b.type_pointer(None, StorageClass::UniformConstant, img);
    let pc = b.type_struct([u32t, u32t]);
    b.decorate(pc, Decoration::Block, []);
    b.member_decorate(pc, 0, Decoration::Offset, [lit(0)]);
    b.member_decorate(pc, 1, Decoration::Offset, [lit(4)]);
    let rta = b.type_runtime_array(f32t);
    b.decorate(rta, Decoration::ArrayStride, [lit(4)]);
    let sb = b.type_struct([rta]);
    b.decorate(sb, Decoration::Block, []);
    b.member_decorate(sb, 0, Decoration::Offset, [lit(0)]);
    let p_sb = b.type_pointer(None, StorageClass::StorageBuffer, sb);
    let p_sbf = b.type_pointer(None, StorageClass::StorageBuffer, f32t);
    let p_pc = b.type_pointer(None, StorageClass::PushConstant, pc);
    let p_pcu = b.type_pointer(None, StorageClass::PushConstant, u32t);
    let p_in = b.type_pointer(None, StorageClass::Input, v3u);
    let gid = global(&mut b, p_in, StorageClass::Input);
    b.decorate(gid, Decoration::BuiltIn, [Operand::BuiltIn(BuiltIn::GlobalInvocationId)]);
    let src = global(&mut b, p_img, StorageClass::UniformConstant);
    b.decorate(src, Decoration::DescriptorSet, [lit(0)]);
    b.decorate(src, Decoration::Binding, [lit(0)]);
    b.decorate(src, Decoration::NonWritable, []);
    let dst = global(&mut b, p_img, StorageClass::UniformConstant);
    b.decorate(dst, Decoration::DescriptorSet, [lit(0)]);
    b.decorate(dst, Decoration::Binding, [lit(1)]);
    let expo = global(&mut b, p_sb, StorageClass::StorageBuffer);
    b.decorate(expo, Decoration::DescriptorSet, [lit(0)]);
    b.decorate(expo, Decoration::Binding, [lit(2)]);
    b.decorate(expo, Decoration::NonWritable, []);
    let pcv = global(&mut b, p_pc, StorageClass::PushConstant);
    let (u0, u1) = (b.constant_bit32(u32t, 0), b.constant_bit32(u32t, 1));
    let fty = b.type_function(void, []);
    let f = b.begin_function(void, None, FunctionControl::NONE, fty)?;
    b.begin_block(None)?;
    let g = b.load(v3u, None, gid, None, [])?;
    let gx = b.composite_extract(u32t, None, g, [0])?;
    let gy = b.composite_extract(u32t, None, g, [1])?;
    let p_w = b.access_chain(p_pcu, None, pcv, [u0])?;
    let w = b.load(u32t, None, p_w, None, [])?;
    let p_h = b.access_chain(p_pcu, None, pcv, [u1])?;
    let h = b.load(u32t, None, p_h, None, [])?;
    let in_x = b.u_less_than(bool_, None, gx, w)?;
    let in_y = b.u_less_than(bool_, None, gy, h)?;
    let inside = b.logical_and(bool_, None, in_x, in_y)?;
    let (l_body, l_end) = (b.id(), b.id());
    b.selection_merge(l_end, SelectionControl::NONE)?;
    b.branch_conditional(inside, l_body, l_end, [])?;
    b.begin_block(Some(l_body))?;
    let x = b.bitcast(i32t, None, gx)?;
    let y = b.bitcast(i32t, None, gy)?;
    let at = b.composite_construct(v2i, None, [x, y])?;
    let simg = b.load(img, None, src, None, [])?;
    let s = b.image_read(v4f, None, simg, at, None, [])?;
    let dimg = b.load(img, None, dst, None, [])?;
    let d = b.image_read(v4f, None, dimg, at, None, [])?;
    let p_e = b.access_chain(p_sbf, None, expo, [u0, u0])?;
    let e = b.load(f32t, None, p_e, None, [])?;
    let r = b.composite_extract(f32t, None, s, [0])?;
    let gg = b.composite_extract(f32t, None, s, [1])?;
    let bb = b.composite_extract(f32t, None, s, [2])?;
    let r = b.f_mul(f32t, None, r, e)?;
    let gg = b.f_mul(f32t, None, gg, e)?;
    let bb = b.f_mul(f32t, None, bb, e)?;
    let a = b.composite_extract(f32t, None, d, [3])?;
    let out = b.composite_construct(v4f, None, [r, gg, bb, a])?;
    b.image_write(dimg, at, out, None, [])?;
    b.branch(l_end)?;
    b.begin_block(Some(l_end))?;
    b.ret()?;
    b.end_function()?;
    b.entry_point(ExecutionModel::GLCompute, f, "main", [gid]);
    b.execution_mode(f, ExecutionMode::LocalSize, [RGB_MERGE_LOCAL, RGB_MERGE_LOCAL, 1]);
    Ok(b.module().assemble())
}

pub fn sharpen() -> Vec<u32> {
    build_sharpen().expect("sharpen kernel")
}

fn build_sharpen() -> Result<Vec<u32>, rspirv::dr::Error> {
    use rspirv::spirv::{Dim, GlslStd450Op as G, ImageFormat};
    let lit = Operand::LiteralBit32;
    let mut b = Builder::new();
    b.set_version(1, 3);
    b.capability(Capability::Shader);
    let glsl = b.ext_inst_import("GLSL.std.450");
    b.memory_model(spirv::AddressingModel::Logical, spirv::MemoryModel::GLSL450);
    let void = b.type_void();
    let bool_ = b.type_bool();
    let u32t = b.type_int(32, 0);
    let i32t = b.type_int(32, 1);
    let f32t = b.type_float(32, None);
    let v3u = b.type_vector(u32t, 3);
    let v2i = b.type_vector(i32t, 2);
    let v4f = b.type_vector(f32t, 4);
    let img = b.type_image(f32t, Dim::Dim2D, 0, 0, 0, 2, ImageFormat::Rgba16f, None);
    let p_img = b.type_pointer(None, StorageClass::UniformConstant, img);
    let pc = b.type_struct([u32t, u32t, f32t]);
    b.decorate(pc, Decoration::Block, []);
    b.member_decorate(pc, 0, Decoration::Offset, [lit(0)]);
    b.member_decorate(pc, 1, Decoration::Offset, [lit(4)]);
    b.member_decorate(pc, 2, Decoration::Offset, [lit(8)]);
    let p_pc = b.type_pointer(None, StorageClass::PushConstant, pc);
    let p_pcu = b.type_pointer(None, StorageClass::PushConstant, u32t);
    let p_pcf = b.type_pointer(None, StorageClass::PushConstant, f32t);
    let p_in = b.type_pointer(None, StorageClass::Input, v3u);
    let gid = global(&mut b, p_in, StorageClass::Input);
    b.decorate(gid, Decoration::BuiltIn, [Operand::BuiltIn(BuiltIn::GlobalInvocationId)]);
    let src = global(&mut b, p_img, StorageClass::UniformConstant);
    b.decorate(src, Decoration::DescriptorSet, [lit(0)]);
    b.decorate(src, Decoration::Binding, [lit(0)]);
    b.decorate(src, Decoration::NonWritable, []);
    let dst = global(&mut b, p_img, StorageClass::UniformConstant);
    b.decorate(dst, Decoration::DescriptorSet, [lit(0)]);
    b.decorate(dst, Decoration::Binding, [lit(1)]);
    b.decorate(dst, Decoration::NonReadable, []);
    let pcv = global(&mut b, p_pc, StorageClass::PushConstant);
    let (u0, u1, u2) = (b.constant_bit32(u32t, 0), b.constant_bit32(u32t, 1), b.constant_bit32(u32t, 2));
    let (i0, i1) = (b.constant_bit32(i32t, 0), b.constant_bit32(i32t, 1));
    let f_quarter = b.constant_bit32(f32t, 0.25f32.to_bits());
    let f_one = b.constant_bit32(f32t, 1.0f32.to_bits());
    let fty = b.type_function(void, []);
    let f = b.begin_function(void, None, FunctionControl::NONE, fty)?;
    b.begin_block(None)?;
    let g = b.load(v3u, None, gid, None, [])?;
    let gx = b.composite_extract(u32t, None, g, [0])?;
    let gy = b.composite_extract(u32t, None, g, [1])?;
    let p_w = b.access_chain(p_pcu, None, pcv, [u0])?;
    let w = b.load(u32t, None, p_w, None, [])?;
    let p_h = b.access_chain(p_pcu, None, pcv, [u1])?;
    let h = b.load(u32t, None, p_h, None, [])?;
    let p_k = b.access_chain(p_pcf, None, pcv, [u2])?;
    let k = b.load(f32t, None, p_k, None, [])?;
    let in_x = b.u_less_than(bool_, None, gx, w)?;
    let in_y = b.u_less_than(bool_, None, gy, h)?;
    let inside = b.logical_and(bool_, None, in_x, in_y)?;
    let (l_body, l_end) = (b.id(), b.id());
    b.selection_merge(l_end, SelectionControl::NONE)?;
    b.branch_conditional(inside, l_body, l_end, [])?;
    b.begin_block(Some(l_body))?;
    let x = b.bitcast(i32t, None, gx)?;
    let y = b.bitcast(i32t, None, gy)?;
    let wi = b.bitcast(i32t, None, w)?;
    let hi = b.bitcast(i32t, None, h)?;
    let xmax = b.i_sub(i32t, None, wi, i1)?;
    let ymax = b.i_sub(i32t, None, hi, i1)?;
    let xm = b.i_sub(i32t, None, x, i1)?;
    let xm = b.ext_inst(i32t, None, glsl, G::SMax as u32, [Operand::IdRef(xm), Operand::IdRef(i0)])?;
    let xp = b.i_add(i32t, None, x, i1)?;
    let xp = b.ext_inst(i32t, None, glsl, G::SMin as u32, [Operand::IdRef(xp), Operand::IdRef(xmax)])?;
    let ym = b.i_sub(i32t, None, y, i1)?;
    let ym = b.ext_inst(i32t, None, glsl, G::SMax as u32, [Operand::IdRef(ym), Operand::IdRef(i0)])?;
    let yp = b.i_add(i32t, None, y, i1)?;
    let yp = b.ext_inst(i32t, None, glsl, G::SMin as u32, [Operand::IdRef(yp), Operand::IdRef(ymax)])?;
    let simg = b.load(img, None, src, None, [])?;
    let mut fetch = |b: &mut Builder, px: Word, py: Word| -> Result<Word, rspirv::dr::Error> {
        let at = b.composite_construct(v2i, None, [px, py])?;
        b.image_read(v4f, None, simg, at, None, [])
    };
    let c = fetch(&mut b, x, y)?;
    let n = fetch(&mut b, x, ym)?;
    let s = fetch(&mut b, x, yp)?;
    let e = fetch(&mut b, xp, y)?;
    let wv = fetch(&mut b, xm, y)?;
    let nw = fetch(&mut b, xm, ym)?;
    let ne = fetch(&mut b, xp, ym)?;
    let sw = fetch(&mut b, xm, yp)?;
    let se = fetch(&mut b, xp, yp)?;
    let mut lo = c;
    let mut hi4 = c;
    for v in [n, s, e, wv, nw, ne, sw, se] {
        lo = b.ext_inst(v4f, None, glsl, G::FMin as u32, [Operand::IdRef(lo), Operand::IdRef(v)])?;
        hi4 = b.ext_inst(v4f, None, glsl, G::FMax as u32, [Operand::IdRef(hi4), Operand::IdRef(v)])?;
    }
    let sum = b.f_add(v4f, None, n, s)?;
    let sum = b.f_add(v4f, None, sum, e)?;
    let sum = b.f_add(v4f, None, sum, wv)?;
    let quarter = b.composite_construct(v4f, None, [f_quarter, f_quarter, f_quarter, f_quarter])?;
    let mean = b.f_mul(v4f, None, sum, quarter)?;
    let detail = b.f_sub(v4f, None, c, mean)?;
    let kv = b.composite_construct(v4f, None, [k, k, k, k])?;
    let add = b.f_mul(v4f, None, detail, kv)?;
    let out = b.f_add(v4f, None, c, add)?;
    let out = b.ext_inst(v4f, None, glsl, G::FClamp as u32, [Operand::IdRef(out), Operand::IdRef(lo), Operand::IdRef(hi4)])?;
    let r = b.composite_extract(f32t, None, out, [0])?;
    let gg = b.composite_extract(f32t, None, out, [1])?;
    let bb = b.composite_extract(f32t, None, out, [2])?;
    let out = b.composite_construct(v4f, None, [r, gg, bb, f_one])?;
    let at = b.composite_construct(v2i, None, [x, y])?;
    let dimg = b.load(img, None, dst, None, [])?;
    b.image_write(dimg, at, out, None, [])?;
    b.branch(l_end)?;
    b.begin_block(Some(l_end))?;
    b.ret()?;
    b.end_function()?;
    b.entry_point(ExecutionModel::GLCompute, f, "main", [gid]);
    b.execution_mode(f, ExecutionMode::LocalSize, [SHARPEN_LOCAL, SHARPEN_LOCAL, 1]);
    Ok(b.module().assemble())
}

/// Workgroup size of `ui_alpha` (each way).
pub const UI_ALPHA_LOCAL: u32 = 8;

/// Frame generation's interface coverage (`gpu::framegen::capture`): binding 1 (RGBA8, the
/// swapchain's size) gets, for each pixel inside the picture's rectangle, the alpha of the pixel
/// of binding 0 (RGBA8, the interface layer as the title's composite maps it over its picture)
/// that it shows, and 0 outside; inside a second rectangle (the port's own text box) 1. The
/// value fills all four channels. Push constants (u32): the picture's x, y, width, height in the
/// output, the source's width and height, the output's width and height, the box's x0, y0, x1, y1.
pub fn ui_alpha() -> Vec<u32> {
    build_ui_alpha().expect("ui alpha kernel")
}

fn build_ui_alpha() -> Result<Vec<u32>, rspirv::dr::Error> {
    use rspirv::spirv::{Dim, GlslStd450Op as G, ImageFormat};
    let lit = Operand::LiteralBit32;
    let mut b = Builder::new();
    b.set_version(1, 3);
    b.capability(Capability::Shader);
    let glsl = b.ext_inst_import("GLSL.std.450");
    b.memory_model(spirv::AddressingModel::Logical, spirv::MemoryModel::GLSL450);
    let void = b.type_void();
    let bool_ = b.type_bool();
    let u32t = b.type_int(32, 0);
    let i32t = b.type_int(32, 1);
    let f32t = b.type_float(32, None);
    let v3u = b.type_vector(u32t, 3);
    let v2i = b.type_vector(i32t, 2);
    let v4f = b.type_vector(f32t, 4);
    let img = b.type_image(f32t, Dim::Dim2D, 0, 0, 0, 2, ImageFormat::Rgba8, None);
    let p_img = b.type_pointer(None, StorageClass::UniformConstant, img);
    let pc = b.type_struct([u32t; 12]);
    b.decorate(pc, Decoration::Block, []);
    for i in 0..12 {
        b.member_decorate(pc, i, Decoration::Offset, [lit(4 * i)]);
    }
    let p_pc = b.type_pointer(None, StorageClass::PushConstant, pc);
    let p_pcu = b.type_pointer(None, StorageClass::PushConstant, u32t);
    let p_in = b.type_pointer(None, StorageClass::Input, v3u);
    let gid = global(&mut b, p_in, StorageClass::Input);
    b.decorate(gid, Decoration::BuiltIn, [Operand::BuiltIn(BuiltIn::GlobalInvocationId)]);
    let src = global(&mut b, p_img, StorageClass::UniformConstant);
    b.decorate(src, Decoration::DescriptorSet, [lit(0)]);
    b.decorate(src, Decoration::Binding, [lit(0)]);
    b.decorate(src, Decoration::NonWritable, []);
    let dst = global(&mut b, p_img, StorageClass::UniformConstant);
    b.decorate(dst, Decoration::DescriptorSet, [lit(0)]);
    b.decorate(dst, Decoration::Binding, [lit(1)]);
    b.decorate(dst, Decoration::NonReadable, []);
    let pcv = global(&mut b, p_pc, StorageClass::PushConstant);
    let idx: Vec<Word> = (0..12u32).map(|i| b.constant_bit32(u32t, i)).collect();
    let u1 = b.constant_bit32(u32t, 1);
    let f0 = b.constant_bit32(f32t, 0.0f32.to_bits());
    let f1 = b.constant_bit32(f32t, 1.0f32.to_bits());
    let fty = b.type_function(void, []);
    let f = b.begin_function(void, None, FunctionControl::NONE, fty)?;
    b.begin_block(None)?;
    let g = b.load(v3u, None, gid, None, [])?;
    let gx = b.composite_extract(u32t, None, g, [0])?;
    let gy = b.composite_extract(u32t, None, g, [1])?;
    let mut pcs = Vec::with_capacity(12);
    for &i in &idx {
        let p = b.access_chain(p_pcu, None, pcv, [i])?;
        pcs.push(b.load(u32t, None, p, None, [])?);
    }
    let (rx, ry, rw, rh, sw, sh, dw, dh, tx0, ty0, tx1, ty1) = (pcs[0], pcs[1], pcs[2], pcs[3], pcs[4], pcs[5], pcs[6], pcs[7], pcs[8], pcs[9], pcs[10], pcs[11]);
    let in_w = b.u_less_than(bool_, None, gx, dw)?;
    let in_h = b.u_less_than(bool_, None, gy, dh)?;
    let in_out = b.logical_and(bool_, None, in_w, in_h)?;
    let (l_body, l_end) = (b.id(), b.id());
    b.selection_merge(l_end, SelectionControl::NONE)?;
    b.branch_conditional(in_out, l_body, l_end, [])?;
    b.begin_block(Some(l_body))?;
    // Inside the picture: [rx, rx + rw) x [ry, ry + rh).
    let within = |b: &mut Builder, v: Word, lo: Word, hi: Word| -> Result<Word, rspirv::dr::Error> {
        let a = b.u_greater_than_equal(bool_, None, v, lo)?;
        let c = b.u_less_than(bool_, None, v, hi)?;
        b.logical_and(bool_, None, a, c)
    };
    let rx1 = b.i_add(u32t, None, rx, rw)?;
    let ry1 = b.i_add(u32t, None, ry, rh)?;
    let ix = within(&mut b, gx, rx, rx1)?;
    let iy = within(&mut b, gy, ry, ry1)?;
    let inside = b.logical_and(bool_, None, ix, iy)?;
    // The source pixel: (g - r) * s / r, clamped into the source.
    let coord = |b: &mut Builder, v: Word, r0: Word, rlen: Word, slen: Word| -> Result<Word, rspirv::dr::Error> {
        let d = b.i_sub(u32t, None, v, r0)?;
        let m = b.i_mul(u32t, None, d, slen)?;
        let den = b.ext_inst(u32t, None, glsl, G::UMax as u32, [Operand::IdRef(rlen), Operand::IdRef(u1)])?;
        let q = b.u_div(u32t, None, m, den)?;
        let smax = b.ext_inst(u32t, None, glsl, G::UMax as u32, [Operand::IdRef(slen), Operand::IdRef(u1)])?;
        let last = b.i_sub(u32t, None, smax, u1)?;
        let q = b.ext_inst(u32t, None, glsl, G::UMin as u32, [Operand::IdRef(q), Operand::IdRef(last)])?;
        b.bitcast(i32t, None, q)
    };
    let sx = coord(&mut b, gx, rx, rw, sw)?;
    let sy = coord(&mut b, gy, ry, rh, sh)?;
    let simg = b.load(img, None, src, None, [])?;
    let at = b.composite_construct(v2i, None, [sx, sy])?;
    let s = b.image_read(v4f, None, simg, at, None, [])?;
    let a_src = b.composite_extract(f32t, None, s, [3])?;
    let a = b.select(f32t, None, inside, a_src, f0)?;
    let bx = within(&mut b, gx, tx0, tx1)?;
    let by = within(&mut b, gy, ty0, ty1)?;
    let in_box = b.logical_and(bool_, None, bx, by)?;
    let a = b.select(f32t, None, in_box, f1, a)?;
    let out = b.composite_construct(v4f, None, [a, a, a, a])?;
    let ox = b.bitcast(i32t, None, gx)?;
    let oy = b.bitcast(i32t, None, gy)?;
    let oat = b.composite_construct(v2i, None, [ox, oy])?;
    let dimg = b.load(img, None, dst, None, [])?;
    b.image_write(dimg, oat, out, None, [])?;
    b.branch(l_end)?;
    b.begin_block(Some(l_end))?;
    b.ret()?;
    b.end_function()?;
    b.entry_point(ExecutionModel::GLCompute, f, "main", [gid]);
    b.execution_mode(f, ExecutionMode::LocalSize, [UI_ALPHA_LOCAL, UI_ALPHA_LOCAL, 1]);
    Ok(b.module().assemble())
}
