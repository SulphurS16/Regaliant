//! Shader front end: program decoding (cached), resolution of descriptors and
//! scalar loads against guest state, specialization and translation to SPIR-V
//! (cached by program contents and specialization).

use super::cp::{Cp, Space};
use super::desc::{Tsharp, Vsharp};
use parking_lot::Mutex;
use rdna::analysis::{Plan, Prov, Ptr, ResKind};
use rdna::translate::{self, ImageDim, MrtSpec, NumClass, Output, PsInput, ResSpec, Spec, Stage};
use std::collections::HashMap;
use std::hash::{Hash, Hasher};
use std::sync::Arc;

struct Program {
    words: usize,
    hash: u64,
    insts: Arc<Vec<rdna::Inst>>,
}

static PROGRAMS: Mutex<Option<HashMap<u64, Program>>> = Mutex::new(None);
static PLANS: Mutex<Option<HashMap<u64, Arc<Plan>>>> = Mutex::new(None);
static TRANSLATIONS: Mutex<Option<HashMap<(u64, u64), Arc<Translated>>>> = Mutex::new(None);
/// Translations at least this large are logged with their program saved
/// (`prepare`), once per program address.
const BIG_SPIRV_WORDS: usize = 40_000;
static BIG_SPIRV_SEEN: Mutex<Option<std::collections::HashSet<u64>>> = Mutex::new(None);
thread_local! {
    /// Vertex stages prepared inside `with_base_instance`: the user-data
    /// word that receives the draw's start instance on the GPU.
    static BASE_INSTANCE: std::cell::Cell<Option<u16>> = const { std::cell::Cell::new(None) };
    /// Draw stages prepared while a `BatchedDraw` guard is held: translated
    /// batched (`Spec::batched`, buffers by device address).
    static BATCHED_DRAW: std::cell::Cell<bool> = const { std::cell::Cell::new(false) };
    /// Batched draw stages prepared while a `GpuDrawMasks` guard is held:
    /// their GPU-resolved loads and buffers (`Spec::gpu_loads`, `gpu_res`;
    /// vertex stage, pixel stage).
    static GPU_DRAW_MASKS: std::cell::Cell<Option<[(u64, u64); 2]>> = const { std::cell::Cell::new(None) };
}

/// While held, the batched draw stages this thread prepares resolve the
/// given loads and buffers on the GPU (a GPU-resolved pair, `gpures`).
pub struct GpuDrawMasks(bool);

pub fn gpu_draw_masks(m: Option<[(u64, u64); 2]>) -> GpuDrawMasks {
    if m.is_some() {
        GPU_DRAW_MASKS.with(|c| c.set(m));
    }
    GpuDrawMasks(m.is_some())
}

impl Drop for GpuDrawMasks {
    fn drop(&mut self) {
        if self.0 {
            GPU_DRAW_MASKS.with(|c| c.set(None));
        }
    }
}

/// The GPU-resolved masks of a batched draw stage being prepared (0:
/// vertex, 1: pixel), when the page table exists.
fn gpu_draw_masks_now(stage: usize) -> Option<(u64, u64)> {
    let m = GPU_DRAW_MASKS.with(|c| c.get())?;
    (super::pagetable::address() != 0).then_some(m[stage])
}

/// While held, the draw stages this thread prepares are translated batched
/// (a batched draw run, `exec::draw`); cleared when dropped.
pub struct BatchedDraw(bool);

pub fn batched_draw(on: bool) -> BatchedDraw {
    if on {
        BATCHED_DRAW.with(|c| c.set(true));
    }
    BatchedDraw(on)
}

impl Drop for BatchedDraw {
    fn drop(&mut self) {
        if self.0 {
            BATCHED_DRAW.with(|c| c.set(false));
        }
    }
}

fn batched_draw_now() -> bool {
    BATCHED_DRAW.with(|c| c.get())
}

/// Runs `f` with the vertex stages it prepares taking user-data word `slot`
/// from the draw's start instance on the GPU (`Spec::base_instance_sgpr`:
/// indexed indirect draws whose arguments the GPU reads).
pub fn with_base_instance<T>(slot: u16, f: impl FnOnce() -> T) -> T {
    BASE_INSTANCE.with(|c| c.set(Some(slot)));
    let r = f();
    BASE_INSTANCE.with(|c| c.set(None));
    r
}

/// Whether anything of the stage resolved on the CPU (descriptors, scalar
/// loads and their pointers) depends on user-data word `slot`: then its
/// value must be known when the draw is recorded.
pub fn uses_user_word(p: &Prepared, slot: u16) -> bool {
    user_deps(&p.tr.plan) & (1u32 << slot.min(31)) != 0
}

/// NGG programs whose CPU-resolved inputs depend on the start-instance
/// word: their indirect draws read their arguments on the CPU.
static CPU_ARGS_PROGRAMS: Mutex<Option<std::collections::HashSet<u64>>> = Mutex::new(None);
static CPU_ARGS_ANY: std::sync::atomic::AtomicBool = std::sync::atomic::AtomicBool::new(false);

pub fn note_cpu_args_program(addr: u64) {
    if CPU_ARGS_PROGRAMS.lock().get_or_insert_with(Default::default).insert(addr) {
        crate::info!("gpu: vertex program {addr:#x} resolves inputs from its start-instance word on the CPU: its indirect draws read their arguments there");
    }
    CPU_ARGS_ANY.store(true, std::sync::atomic::Ordering::Release);
}

pub fn cpu_args_program(addr: u64) -> bool {
    CPU_ARGS_ANY.load(std::sync::atomic::Ordering::Acquire) && CPU_ARGS_PROGRAMS.lock().as_ref().is_some_and(|s| s.contains(&addr))
}

thread_local! {
    /// Set while `without_gpu_chains` runs.
    static NO_GPU_CHAINS: std::cell::Cell<bool> = const { std::cell::Cell::new(false) };
}

/// Runs `f` with the compute preparations it makes resolving every chain on
/// the CPU, a GPU-resolved program's included (front end 2: a command whose
/// per-item chains it cannot take on the GPU, `exec::fe2_gpu_ranges`).
pub fn without_gpu_chains<T>(f: impl FnOnce() -> T) -> T {
    NO_GPU_CHAINS.with(|c| c.set(true));
    let r = f();
    NO_GPU_CHAINS.with(|c| c.set(false));
    r
}

thread_local! {
    /// Set while `with_lod_stats` runs.
    static LOD_MEASURE: std::cell::Cell<bool> = const { std::cell::Cell::new(false) };
}

/// Runs `f` with the pixel stages it prepares keeping their LOD statistics
/// where the depth tests stay after the shader (a draw measuring in
/// rotation, `tune lod_late` >= 2: `lodstats::rotate`).
pub fn with_lod_stats<T>(f: impl FnOnce() -> T) -> T {
    LOD_MEASURE.with(|c| c.set(true));
    let r = f();
    LOD_MEASURE.with(|c| c.set(false));
    r
}

/// Whether the LOD statistics of pixel programs whose depth tests stay
/// after the shader are decided per draw (`tune lod_late` N >= 2, or 1 with
/// `lod_skip`): their stages are prepared without them, and a draw that
/// measures prepares its pixel stage again with them (`lodstats::rotate`).
pub fn lod_per_draw() -> bool {
    matches!(super::tune::lod_stats(), 3 | 4) && match super::tune::lod_late() {
        0 => false,
        1 => super::tune::lod_skip(),
        _ => true,
    }
}

/// Whether a pixel program whose depth tests stay after the shader keeps
/// its LOD statistics (`tune lod_late`; `measure`: batched, or a draw
/// measuring, `lod_per_draw`).
fn late_stats(measure: bool) -> bool {
    match super::tune::lod_late() {
        0 => false,
        _ if lod_per_draw() => measure,
        _ => true,
    }
}

/// Whether `p` is a pixel stage prepared without the LOD statistics its
/// draws measure when they bind counted textures (`lod_per_draw`): a
/// program whose depth tests stay after the shader, not batched (a batched
/// run's draws' textures are not all known to the CPU: its stages keep
/// them).
pub fn lod_rotated(p: &Prepared) -> bool {
    let s = &p.tr.spec;
    s.stage == Stage::Fragment && !s.lod_stats && !s.batched && !s.early_tests && !s.early_ok && lod_per_draw()
}

/// Whether compute programs' LOD statistics are decided per dispatch
/// (`tune lod_cs` >= 2 with the measured statistics on): their stages are
/// prepared without them, and a dispatch that measures prepares its stage
/// again with them (`lodstats::rotate`).
pub fn cs_lod_rotation() -> bool {
    matches!(super::tune::lod_stats(), 3 | 4) && super::tune::lod_cs() >= 2
}

/// Whether `p` is a compute stage prepared without the LOD statistics its
/// dispatches measure when they bind counted textures (`cs_lod_rotation`).
pub fn cs_lod_rotated(p: &Prepared) -> bool {
    let s = &p.tr.spec;
    s.stage == Stage::Compute && !s.lod_stats && !s.batched && cs_lod_rotation()
}

thread_local! {
    /// This thread's view of the shared caches (entries never change once
    /// made), so that every command does not take their locks: programs by
    /// address with the shader registrations count at which they were last
    /// confirmed and the address's registration number (validating them by
    /// hashing their code took a few microseconds per command), plans, and
    /// translations.
    /// Per-thread handles (`Rc`) on the shared programs and plans: a
    /// preparation clones the handle, not the shared reference count the
    /// recording threads all touch (only a new preparation keeps a shared
    /// clone).
    static LOCAL_PROGRAMS: std::cell::RefCell<super::fx::Map<u64, (u64, u64, Arc<Local<Vec<rdna::Inst>>>, u64)>> = std::cell::RefCell::new(super::fx::Map::default());
    static LOCAL_PLANS: std::cell::RefCell<super::fx::Map<u64, std::rc::Rc<Arc<Plan>>>> = std::cell::RefCell::new(super::fx::Map::default());
    /// The reader's sizes in the last full preparation of each plan (ranges
    /// read, words read, pointers loaded): its vectors start at that
    /// capacity instead of doubling from empty (their growth was ~4% of the
    /// graphics command processor's busy samples, run 20261013-143720).
    static READ_HINTS: std::cell::RefCell<super::fx::Map<u64, [u32; 3]>> = std::cell::RefCell::new(super::fx::Map::default());
    static LOCAL_TRANSLATIONS: std::cell::RefCell<HashMap<(u64, u64), Arc<Translated>>> = std::cell::RefCell::new(HashMap::new());
    /// Translations by (plan key, fast hash of the specialization), checked
    /// by comparing the specialization: SipHashing the whole Spec for every
    /// command took about a microsecond per stage.
    static FAST_TRANSLATIONS: std::cell::RefCell<super::fx::Map<(u64, u64), Arc<Local<Translated>>>> = std::cell::RefCell::new(super::fx::Map::default());
    /// Each plan's latest translation on this thread: a preparation whose
    /// specialization equals it skips hashing the specialization (runs of
    /// commands repeat their shaders' specializations).
    static LAST_TRANSLATIONS: std::cell::RefCell<super::fx::Map<u64, Arc<Local<Translated>>>> = std::cell::RefCell::new(super::fx::Map::default());
    /// The vectors of the specializations being built (`spec_vecs`), kept
    /// for the next preparation.
    static SPEC_VECS: std::cell::RefCell<(Vec<PsInput>, Vec<ResSpec>)> = const { std::cell::RefCell::new((Vec::new(), Vec::new())) };
}

/// Empty vectors for a specialization's pixel inputs and resources, with
/// the capacity of earlier ones (`spec_vecs_back`).
fn spec_vecs() -> (Vec<PsInput>, Vec<ResSpec>) {
    SPEC_VECS.with(|v| {
        let (mut a, mut b) = std::mem::take(&mut *v.borrow_mut());
        a.clear();
        b.clear();
        (a, b)
    })
}

fn spec_vecs_back(spec: Spec) {
    SPEC_VECS.with(|v| *v.borrow_mut() = (spec.ps_inputs, spec.resources));
}

/// A recording thread's own handle on a shared program or translation:
/// preparations and their copies clone the handle, whose reference count
/// only its thread touches. (The shared counts are contended: a clone and
/// drop took up to ~100 ns on the development machine's Ryzen 9 5900X with
/// several recording threads, 3 ns uncontended.)
pub struct Local<T>(pub Arc<T>);

impl<T> std::ops::Deref for Local<T> {
    type Target = T;
    fn deref(&self) -> &T {
        &self.0
    }
}

#[derive(Clone)]
pub struct Translated {
    pub out: Output,
    pub plan: Arc<Plan>,
    /// Cache key (program hash ^ specialization hash).
    pub key: u64,
    /// Buffer resources targeted by atomic instructions.
    pub atomic_res: Vec<usize>,
    /// The specialization it was made for (program address zero unless
    /// DS_WAVEMASK keys translations by it): the per-thread lookup by a fast
    /// hash compares it.
    pub spec: Spec,
}

pub(crate) fn hash_words(w: &[u32]) -> u64 {
    let mut h = std::collections::hash_map::DefaultHasher::new();
    w.hash(&mut h);
    h.finish()
}

/// Program words at `addr`: the Agc shader object's code size when known,
/// else up to the first s_endpgm (at most 256 KiB).
fn code_slice(addr: u64) -> Option<&'static [u32]> {
    let size = match crate::hle::libs::agc::shader_code_size(addr) {
        Some(n) => n,
        None => {
            let mut n = 0usize;
            while n < 0x40000 && super::cp::guest_readable(addr + n as u64, 4) {
                let w = unsafe { std::ptr::read_volatile((addr + n as u64) as *const u32) };
                n += 4;
                if w == 0xBF81_0000 {
                    break;
                }
            }
            n
        }
    };
    if size < 4 || !super::cp::guest_readable(addr, size as u64) {
        return None;
    }
    Some(unsafe { std::slice::from_raw_parts(addr as *const u32, size / 4) })
}

/// Decodes (and caches) the program at `addr`; returns it and its hash.
pub(crate) fn program(addr: u64) -> Result<(Arc<Vec<rdna::Inst>>, u64), String> {
    program_local(addr).map(|(p, h)| (Arc::clone(&p.0), h))
}

/// `program` as this thread's handle on it.
fn program_local(addr: u64) -> Result<(Arc<Local<Vec<rdna::Inst>>>, u64), String> {
    use crate::hle::libs::agc;
    // A program from a shader object stays as decoded until a shader object
    // is created again at its address (or anywhere: then it is checked).
    let regs = agc::shader_registrations();
    let hit = LOCAL_PROGRAMS.with(|c| {
        let mut c = c.borrow_mut();
        let e = c.get_mut(&addr)?;
        if e.0 != regs {
            if agc::shader_registration(addr) != Some(e.1) {
                return None;
            }
            e.0 = regs;
        }
        Some((e.2.clone(), e.3))
    });
    if let Some(h) = hit {
        return Ok(h);
    }
    let reg = agc::shader_registration(addr);
    let (insts, hash) = program_checked(addr)?;
    let insts = Arc::new(Local(insts));
    if let Some(r) = reg {
        LOCAL_PROGRAMS.with(|c| c.borrow_mut().insert(addr, (regs, r, insts.clone(), hash)));
    }
    Ok((insts, hash))
}

/// The program at `addr`, decoded (cached and checked by hashing its code).
fn program_checked(addr: u64) -> Result<(Arc<Vec<rdna::Inst>>, u64), String> {
    let code = code_slice(addr).ok_or_else(|| format!("shader {addr:#x} not mapped"))?;
    let mut g = PROGRAMS.lock();
    let map = g.get_or_insert_with(HashMap::new);
    if let Some(p) = map.get(&addr) {
        if p.words <= code.len() && hash_words(&code[..p.words]) == p.hash {
            return Ok((p.insts.clone(), p.hash));
        }
    }
    let insts = rdna::decode_program(code).map_err(|e| format!("decode {addr:#x}: {}", e.0))?;
    let last = insts.last().ok_or("empty program")?;
    let words = (last.pc / 4 + last.len as u32) as usize;
    let hash = hash_words(&code[..words.min(code.len())]);
    note_per_lane(addr, &code[..words.min(code.len())], hash);
    note_append(addr, &insts, hash);
    let insts = Arc::new(insts);
    map.insert(addr, Program { words, hash, insts: insts.clone() });
    Ok((insts, hash))
}

/// A pointer chain's structural hash (`Reader::ptrs`).
fn ptr_hash(p: &Ptr) -> u64 {
    match p {
        Ptr::User(i) => 0x9E37_79B9_7F4A_7C15 ^ *i as u64,
        Ptr::Mem(inner, off) => ptr_hash(inner).rotate_left(17) ^ (*off as u64 + 1).wrapping_mul(0xBF58_476D_1CE4_E5B9),
        Ptr::Pc(o) => (*o ^ 0x94D0_49BB_1331_11EB).wrapping_mul(0x2545_F491_4F6C_DD1D),
    }
}

/// Guest memory reader that records what it read (hazard checks; adjacent
/// reads merged). It lives within one preparation of a stage whose plan
/// (`'a`) holds the pointer chains it resolves.
pub struct Reader<'a> {
    pub reads: Vec<(u64, u64)>,
    /// The words read, in the order of `reads` (a preparation made ahead of
    /// execution is checked against them, see `words_unchanged`).
    pub words: Vec<u32>,
    /// Ranges it could not read (not mapped or not readable): what was
    /// resolved without them holds while they stay so.
    pub unread: Vec<(u64, u64)>,
    /// The mappings of recent reads (no memory-map lookup inside them; a
    /// stage reads its user data, descriptor tables and constants from a few
    /// mappings in turn), replaced round robin. (R2: eight, kept per thread
    /// across preparations while the mappings are unchanged: `load_spans`.)
    spans: [(u64, u64); 8],
    next_span: usize,
    /// Pointers loaded from memory so far, by the plan's chain (each resource
    /// dword names its pointer chain again; kept by reference, not cloned:
    /// the clones' allocations and deep comparisons were ~6% of the graphics
    /// command processor's time, run 20261012-202512), with the chain's
    /// structural hash (equal chains elsewhere in the plan are found by it
    /// before one deep comparison: comparing every entry deeply was up to 7%
    /// of the compute queues' busy samples, run 20261013-115322).
    ptrs: Vec<(&'a Ptr, u64, u64)>,
    /// Address of the program (program-relative pointers).
    code_base: u64,
    /// Pointers resolved so far by the plan's chain number
    /// (`Plan::chain_ids`): loads and resources naming equal chains find
    /// them without hashing or comparing chains (`ptr_hash` was 7.8% of the
    /// compute queues' busy samples, run 20261014-145828).
    chain_vals: Vec<Option<u64>>,
    /// `mem::MAP_GEN` when the spans were loaded (`load_spans`; u64::MAX:
    /// not kept).
    span_gen: u64,
}

thread_local! {
    /// R2: this thread's reader spans (`Reader::load_spans`) and the
    /// mapping generation they hold for.
    static R2_SPANS: std::cell::Cell<(u64, [(u64, u64); 8], usize)> = const { std::cell::Cell::new((u64::MAX, [(0, 0); 8], 0)) };
    /// R2: the next stage template's number (`CsTmpl::id` etc.; 0 is none).
    static TMPL_IDS: std::cell::Cell<u64> = const { std::cell::Cell::new(1) };
}

/// A new stage template's number on this thread (never 0).
fn next_tmpl_id() -> u64 {
    TMPL_IDS.with(|c| {
        let v = c.get();
        c.set(v + 1);
        v
    })
}

impl<'a> Reader<'a> {
    pub fn new(code_base: u64) -> Reader<'a> {
        Self::with_sizes(code_base, [0; 3])
    }

    /// A reader whose vectors start with room for `sizes` (ranges read,
    /// words read, pointers loaded).
    fn with_sizes(code_base: u64, sizes: [u32; 3]) -> Reader<'a> {
        Reader {
            reads: pool::pairs(sizes[0] as usize),
            words: pool::words(sizes[1] as usize),
            unread: pool::pairs(0),
            spans: [(0, 0); 8],
            next_span: 0,
            ptrs: Vec::with_capacity(sizes[2] as usize),
            code_base,
            chain_vals: pool::addrs(0),
            span_gen: u64::MAX,
        }
    }

    /// The lean lane's reader (`lean::prepare`): no pointer list or chain values (its plans
    /// keep node values of their own).
    fn lean(code_base: u64, sizes: [u32; 3]) -> Reader<'a> {
        Reader {
            reads: pool::pairs(sizes[0] as usize),
            words: pool::words(sizes[1] as usize),
            unread: pool::pairs(0),
            spans: [(0, 0); 8],
            next_span: 0,
            ptrs: Vec::new(),
            code_base,
            chain_vals: Vec::new(),
            span_gen: u64::MAX,
        }
    }

    /// R2: this thread's spans from its earlier preparations, while the
    /// mappings are unchanged (`mem::MAP_GEN`; `readable_span` answers the
    /// same then).
    fn load_spans(&mut self) {
        let generation = crate::mem::MAP_GEN.load(std::sync::atomic::Ordering::Acquire);
        R2_SPANS.with(|c| {
            let (g, spans, next) = c.get();
            if g == generation {
                self.spans = spans;
                self.next_span = next;
            }
        });
        self.span_gen = generation;
    }

    /// R2: the spans kept for this thread's next preparation.
    fn save_spans(&self) {
        R2_SPANS.with(|c| c.set((self.span_gen, self.spans, self.next_span)));
    }

    /// `ptr` of the chain numbered `id` in its plan (u32::MAX: unnumbered).
    fn chain(&mut self, id: u32, p: &'a Ptr, user: &[u32]) -> Option<u64> {
        if id == u32::MAX || !super::tune::chain_memo() {
            return self.ptr(p, user);
        }
        let i = id as usize;
        if let Some(Some(v)) = self.chain_vals.get(i) {
            return Some(*v);
        }
        let v = self.ptr(p, user)?;
        if self.chain_vals.len() <= i {
            self.chain_vals.resize(i + 1, None);
        }
        self.chain_vals[i] = Some(v);
        Some(v)
    }

    /// Whether guest memory [addr, addr + n) is readable: within a recent
    /// read's mapping, else one memory-map lookup (`readable_span` was ~7% of
    /// the recording threads' time, about 50 lookups per command, mostly
    /// from here: run 20261012-202512).
    fn readable(&mut self, addr: u64, n: u64) -> bool {
        let end = addr.saturating_add(n);
        if !super::tune::span_cache() {
            // The earlier behavior: the last read's mapping, two lookups.
            if addr >= self.spans[0].0 && end <= self.spans[0].1 {
                return true;
            }
            if !super::cp::guest_readable(addr, n) {
                return false;
            }
            self.spans[0] = crate::mem::readable_span(addr).unwrap_or((addr, end));
            return true;
        }
        if self.spans.iter().any(|&(s, e)| addr >= s && end <= e) {
            return true;
        }
        if !super::cp::valid(addr, n) {
            return false;
        }
        // R2: the readability map's "readable" stands without the table's
        // lock (other answers go the table's way: a mapping in progress is
        // waited for there).
        if self.span_gen != u64::MAX && crate::mem::rg_readable(addr, n) == Some(true) {
            return true;
        }
        match crate::mem::readable_span(addr) {
            Some((s, e)) if end <= e => {
                self.spans[self.next_span] = (s, e);
                self.next_span = (self.next_span + 1) % self.spans.len();
                true
            }
            // Across mappings with different protections: the slow check.
            Some(_) => crate::mem::readable(addr, n),
            None => false,
        }
    }

    fn u32(&mut self, addr: u64) -> Option<u32> {
        if addr % 4 != 0 {
            return None;
        }
        if !self.readable(addr, 4) {
            self.unread.push((addr, 4));
            return None;
        }
        match self.reads.last_mut() {
            Some(r) if r.0 + r.1 == addr => r.1 += 4,
            _ => self.reads.push((addr, 4)),
        }
        // A copy recorded earlier leaves other words there (`gpu::forward`).
        let w = super::forward::word_or(addr, unsafe { std::ptr::read_volatile(addr as *const u32) });
        self.words.push(w);
        Some(w)
    }

    /// Reads `out.len()` consecutive dwords at `addr` with one readability
    /// check; false (nothing read) if any of them is not readable.
    fn read_into(&mut self, addr: u64, out: &mut [u32]) -> bool {
        let n = 4 * out.len() as u64;
        if addr % 4 != 0 || n == 0 {
            return false;
        }
        if !self.readable(addr, n) {
            self.unread.push((addr, n));
            return false;
        }
        match self.reads.last_mut() {
            Some(r) if r.0 + r.1 == addr => r.1 += n,
            _ => self.reads.push((addr, n)),
        }
        for (k, o) in out.iter_mut().enumerate() {
            *o = unsafe { std::ptr::read_volatile((addr + 4 * k as u64) as *const u32) };
        }
        super::forward::overlay(addr, out);
        self.words.extend_from_slice(out);
        true
    }

    fn ptr(&mut self, p: &'a Ptr, user: &[u32]) -> Option<u64> {
        match p {
            Ptr::User(i) => Some((*user.get(*i as usize)? as u64) | ((*user.get(*i as usize + 1)? as u64) << 32)),
            Ptr::Mem(inner, off) => {
                // The same chain object first, then an equal one elsewhere in
                // the plan.
                if let Some(&(_, _, v)) = self.ptrs.iter().find(|e| std::ptr::eq(e.0, p)) {
                    return Some(v);
                }
                let h = ptr_hash(p);
                if let Some(&(_, _, v)) = self.ptrs.iter().find(|e| e.1 == h && e.0 == p) {
                    return Some(v);
                }
                let base = self.ptr(inner, user)? + *off as u64;
                let v = (self.u32(base)? as u64) | ((self.u32(base + 4)? as u64) << 32);
                self.ptrs.push((p, h, v));
                Some(v)
            }
            Ptr::Pc(o) => Some(self.code_base + *o),
        }
    }

    fn eval(&mut self, p: &'a Prov, user: &[u32]) -> Option<u32> {
        match p {
            Prov::User(i) => user.get(*i as usize).copied(),
            Prov::Const(c) => Some(*c),
            Prov::Mem(ptr, off) => {
                let b = self.ptr(ptr, user)?;
                self.u32(b + *off as u64)
            }
            Prov::PtrLo(ptr, b) => Some((self.ptr(ptr, user)? + *b as u64) as u32),
            Prov::PtrHi(ptr, b) => Some(((self.ptr(ptr, user)? + *b as u64) >> 32) as u32),
            Prov::Unknown | Prov::Table(..) => None,
        }
    }
}

#[derive(Clone, Debug, PartialEq)]
pub enum Resolved {
    Buffer(Vsharp),
    /// A base address and, for data loads at constant offsets, the bytes they
    /// read from it.
    Global(u64, Option<u64>),
    Image(Tsharp),
    /// The entries of an image table (None: unusable entry, bound as a
    /// dummy image).
    ImageTable(Vec<Option<Tsharp>>),
    Sampler([u32; 4]),
}

/// What preparing a stage found in guest memory, shared by the
/// preparation's copies (one allocation: a copy costs no deep copy).
/// `Prepared` derefs to it.
/// The program and its translation are held here too: the recording
/// threads share their reference counts, so a copy of a preparation
/// touches only this block's (contended clones cost up to ~100 ns each on
/// the development machine's Ryzen 9 5900X).
pub struct Found {
    pub insts: Arc<Local<Vec<rdna::Inst>>>,
    pub tr: Arc<Local<Translated>>,
    pub resources: Vec<Resolved>,
    /// Guest address of each descriptor's first dword when read from memory.
    pub res_addrs: Vec<Option<u64>>,
    /// Guest memory read while resolving (for write hazards).
    pub reads: Vec<(u64, u64)>,
    /// The words read, in the order of `reads`.
    pub words: Vec<u32>,
    /// Guest ranges it could not read (`Reader::unread`).
    pub unread: Vec<(u64, u64)>,
    /// The data-only scalar loads (`Plan::load_data`) as read: where their
    /// dwords sit in `sgpr_data`, the guest address, and where their words
    /// start in `words`.
    pub data_loads: Vec<DataLoad>,
}

impl Drop for Found {
    fn drop(&mut self) {
        pool::put_resolved(std::mem::take(&mut self.resources));
        pool::put_addrs(std::mem::take(&mut self.res_addrs));
        pool::put_pairs(std::mem::take(&mut self.reads));
        pool::put_words(std::mem::take(&mut self.words));
        pool::put_pairs(std::mem::take(&mut self.unread));
        pool::put_loads(std::mem::take(&mut self.data_loads));
    }
}

/// The vectors of finished preparations, kept per thread for the next ones
/// (`Found` and `Prepared` give theirs back when dropped): a preparation
/// made ~10 allocations, ~5% of the recording workers' busy samples in
/// `mi_malloc_aligned` (2026-10-15).
mod pool {
    use super::{DataLoad, Resolved};
    use std::cell::RefCell;

    /// Vectors kept per kind at most, and the largest capacity kept.
    const MAX: usize = 64;
    const CAP_MAX: usize = 1 << 16;

    #[derive(Default)]
    struct Pool {
        pairs: Vec<Vec<(u64, u64)>>,
        words: Vec<Vec<u32>>,
        resolved: Vec<Vec<Resolved>>,
        addrs: Vec<Vec<Option<u64>>>,
        loads: Vec<Vec<DataLoad>>,
    }

    thread_local! {
        static POOL: RefCell<Pool> = RefCell::new(Pool::default());
    }

    macro_rules! kind {
        ($take:ident, $put:ident, $field:ident, $t:ty) => {
            /// An empty vector with room for `cap` (from the pool if one is left).
            pub fn $take(cap: usize) -> Vec<$t> {
                if !crate::gpu::tune::prep_pool() {
                    return Vec::with_capacity(cap);
                }
                let v = POOL.try_with(|p| p.try_borrow_mut().ok().and_then(|mut p| p.$field.pop())).ok().flatten();
                match v {
                    Some(mut v) => {
                        v.reserve(cap);
                        v
                    }
                    None => Vec::with_capacity(cap),
                }
            }

            /// Gives `v` back (emptied) for a later preparation.
            pub fn $put(mut v: Vec<$t>) {
                if v.capacity() == 0 || v.capacity() > CAP_MAX || !crate::gpu::tune::prep_pool() {
                    return;
                }
                v.clear();
                let _ = POOL.try_with(|p| {
                    if let Ok(mut p) = p.try_borrow_mut() {
                        if p.$field.len() < MAX {
                            p.$field.push(v);
                        }
                    }
                });
            }
        };
    }

    kind!(pairs, put_pairs, pairs, (u64, u64));
    kind!(words, put_words, words, u32);
    kind!(resolved, put_resolved, resolved, Resolved);
    kind!(addrs, put_addrs, addrs, Option<u64>);
    kind!(loads, put_loads, loads, DataLoad);
}

/// Everything needed to bind one shader stage for a draw/dispatch: what was
/// found (`Found`, shared by copies; fields reached through `Deref`), and
/// the SGPR data and user data, a copy's own.
pub struct Prepared {
    pub stage: Stage,
    pub addr: u64,
    pub found: Arc<Found>,
    /// SGPR data buffer contents (resource info blocks filled at binding).
    pub sgpr_data: Vec<u32>,
    pub user: UserData,
    /// `mem::MAP_GEN` and `images::GPU_IMAGES_GEN` when it was made (a
    /// preparation made ahead of execution is checked against them).
    pub map_gen: u64,
    pub images_gen: u64,
    /// Image resources that depth-comparison instructions sample (bit per
    /// resource index) and whether each was a depth view (the only part of a
    /// preparation that depends on the image cache).
    pub cmp_images: u64,
    pub cmp_depth: u64,
    /// The program code's hash (`program_checked`).
    pub code_hash: u64,
    /// Where it was made, for `DS_FE2_CALLERS` (`fe2::StageSource`): 1 by
    /// the thread binding it, 2 ahead by a worker (`gpu::ahead`).
    pub ahead: u8,
}

impl Clone for Prepared {
    fn clone(&self) -> Prepared {
        let mut sgpr_data = pool::words(self.sgpr_data.len());
        sgpr_data.extend_from_slice(&self.sgpr_data);
        Prepared {
            stage: self.stage,
            addr: self.addr,
            found: self.found.clone(),
            sgpr_data,
            user: self.user,
            map_gen: self.map_gen,
            images_gen: self.images_gen,
            cmp_images: self.cmp_images,
            cmp_depth: self.cmp_depth,
            code_hash: self.code_hash,
            ahead: self.ahead,
        }
    }
}

impl Drop for Prepared {
    fn drop(&mut self) {
        pool::put_words(std::mem::take(&mut self.sgpr_data));
    }
}

/// A stage's user-data dwords (at most 32), held inline: a preparation
/// and each copy of one would otherwise allocate them.
#[derive(Clone, Copy, PartialEq, Eq, Hash)]
pub struct UserData {
    n: u8,
    w: [u32; 32],
}

impl UserData {
    /// Its words.
    pub fn words(&self) -> &[u32] {
        &self.w[..self.n as usize]
    }

    pub fn new(s: &[u32]) -> UserData {
        let n = s.len().min(32);
        let mut w = [0u32; 32];
        w[..n].copy_from_slice(&s[..n]);
        UserData { n: n as u8, w }
    }
}

impl std::ops::Deref for UserData {
    type Target = [u32];
    fn deref(&self) -> &[u32] {
        &self.w[..self.n as usize]
    }
}

impl std::fmt::Debug for UserData {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        (**self).fmt(f)
    }
}

impl std::ops::Deref for Prepared {
    type Target = Found;
    fn deref(&self) -> &Found {
        &self.found
    }
}

/// A data-only scalar load of a prepared stage (`Prepared::data_loads`).
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct DataLoad {
    pub sgpr: u32,
    pub addr: u64,
    pub dwords: u32,
    pub word: u32,
}

/// Per-plan instruction scans (`compare_images`, `format_resources`) shared
/// by every thread behind their per-thread caches: a new plan is scanned
/// once, not on each recording thread (FINDINGS.md 2026-10-31: 4.9% of the
/// graphics CP's busy samples over Castlewall's arrival went to the scans'
/// instruction names). Keyed by (plan key, scan).
static SCANS: [parking_lot::Mutex<Option<super::fx::Map<(u64, u8), u64>>>; 16] = [const { parking_lot::Mutex::new(None) }; 16];

fn shared_scan(kind: u8, key: u64, f: impl FnOnce() -> u64) -> u64 {
    let shard = &SCANS[((key ^ (key >> 29)) % 16) as usize];
    if let Some(v) = shard.lock().as_ref().and_then(|m| m.get(&(key, kind)).copied()) {
        return v;
    }
    let v = f();
    shard.lock().get_or_insert_with(Default::default).insert((key, kind), v);
    v
}

/// Image resources (bit per index) that depth-comparison sampling
/// instructions (`image_sample_c*`, `image_gather4_c*`) use: only for them
/// does the translation depend on whether the image is a depth view.
fn compare_images(plan_key: u64, insts: &[rdna::Inst], plan: &Plan) -> u64 {
    thread_local! {
        static MASKS: std::cell::RefCell<super::fx::Map<u64, u64>> = std::cell::RefCell::new(super::fx::Map::default());
    }
    MASKS.with(|m| {
        *m.borrow_mut().entry(plan_key).or_insert_with(|| shared_scan(0, plan_key, || {
            let mut mask = 0u64;
            for i in insts {
                let Some(name) = i.name() else { continue };
                if name == "image_get_lod" || !(name.starts_with("image_sample") || name.starts_with("image_gather4")) {
                    continue;
                }
                let has_c = name.trim_start_matches("image_sample").trim_start_matches("image_gather4").split('_').any(|m| m == "c");
                if let (true, Some(&(ri, _))) = (has_c, plan.inst_res.get(&i.pc)) {
                    if ri < 64 {
                        mask |= 1 << ri;
                    }
                }
            }
            mask
        }))
    })
}

/// Whether the depth-view decisions a preparation made for its compared
/// images still hold (`images::Images::depth_view`).
pub fn depth_views_hold(p: &Prepared) -> bool {
    if p.cmp_images == 0 || p.images_gen == super::images::GPU_IMAGES_GEN.load(std::sync::atomic::Ordering::Acquire) {
        return true;
    }
    super::images::with(|im| {
        p.resources.iter().enumerate().all(|(i, r)| match r {
            Resolved::Image(t) if i < 64 && p.cmp_images & (1 << i) != 0 => im.depth_view(t) == (p.cmp_depth & (1 << i) != 0),
            _ => true,
        })
    })
}

/// The registers a stage preparation reads: the live command processor, or
/// a snapshot taken ahead of execution (`StageRegs`).
pub trait RegRead {
    fn sh(&self, i: u32) -> u32;
    /// Context register by byte address (0x28000..).
    fn ctx(&self, a: u32) -> u32;
}

impl RegRead for Cp {
    fn sh(&self, i: u32) -> u32 {
        self.reg(Space::Sh, i)
    }
    fn ctx(&self, a: u32) -> u32 {
        self.reg(Space::Ctx, (a - 0x28000) / 4)
    }
}

/// The kind of stage a preparation is for.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum StageKind {
    Vertex,
    Pixel,
    Compute,
}

/// SH registers (dword indices) and context registers (byte addresses) the
/// preparation of a stage kind reads (`prepare_ngg`, `prepare_ps`,
/// `prepare_compute`; keep in step with them).
fn reg_lists(kind: StageKind) -> (&'static [u32], &'static [u32]) {
    static LISTS: std::sync::OnceLock<[(Vec<u32>, Vec<u32>); 3]> = std::sync::OnceLock::new();
    let l = LISTS.get_or_init(|| {
        let vs_sh: Vec<u32> = [ES_PGM_LO, ES_PGM_LO + 1, GS_RSRC2].into_iter().chain(GS_USER_DATA..GS_USER_DATA + 32).collect();
        let ps_sh: Vec<u32> = [PS_PGM_LO, PS_PGM_LO + 1, PS_RSRC2].into_iter().chain(PS_USER_DATA..PS_USER_DATA + 32).collect();
        // (The depth and stencil state `early_ok` and `ps_key` read: without them every pixel
        // stage prepared from a snapshot missed, 2026-11-19.)
        let mut ps_ctx: Vec<u32> = vec![SPI_PS_IN_CONTROL, SPI_PS_INPUT_ADDR, SPI_SHADER_COL_FORMAT, DB_SHADER_CONTROL, DB_DEPTH_CONTROL, DB_STENCIL_CONTROL, DB_STENCILREFMASK, DB_STENCILREFMASK_BF, DB_ALPHA_TO_MASK];
        ps_ctx.extend((0..32).map(|n| SPI_PS_INPUT_CNTL_0 + 4 * n));
        ps_ctx.extend((0..8).map(|m| CB_COLOR0_INFO + 0x3C * m));
        let cs_sh: Vec<u32> = [CS_PGM_LO, CS_PGM_LO + 1, CS_RSRC2, CS_NUM_THREAD, CS_NUM_THREAD + 1, CS_NUM_THREAD + 2].into_iter().chain(CS_USER_DATA..CS_USER_DATA + 16).collect();
        let mut v = [(vs_sh, vec![VGT_SHADER_STAGES_EN]), (ps_sh, ps_ctx), (cs_sh, Vec::new())];
        for (sh, ctx) in v.iter_mut() {
            sh.sort_unstable();
            ctx.sort_unstable();
        }
        v
    });
    let e = &l[kind as usize];
    (&e.0, &e.1)
}

/// The registers a stage preparation reads, captured ahead of execution
/// (values in the order of `reg_lists`).
#[derive(Debug)]
pub struct StageRegs {
    pub kind: StageKind,
    pub sh: Vec<u32>,
    pub ctx: Vec<u32>,
    /// A register outside the lists was read (the preparation is not used).
    pub missed: std::sync::atomic::AtomicBool,
    /// The first such register (SH: dword index; context: byte address).
    pub missed_reg: std::sync::atomic::AtomicU32,
}

impl StageRegs {
    pub fn capture<R: RegRead + ?Sized>(kind: StageKind, r: &R) -> StageRegs {
        let (sh, ctx) = reg_lists(kind);
        StageRegs {
            kind,
            sh: sh.iter().map(|&i| r.sh(i)).collect(),
            ctx: ctx.iter().map(|&a| r.ctx(a)).collect(),
            missed: std::sync::atomic::AtomicBool::new(false),
            missed_reg: std::sync::atomic::AtomicU32::new(0),
        }
    }

    /// Whether `r` holds the same values.
    pub fn matches<R: RegRead + ?Sized>(&self, r: &R) -> bool {
        let (sh, ctx) = reg_lists(self.kind);
        sh.iter().zip(&self.sh).all(|(&i, &v)| r.sh(i) == v) && ctx.iter().zip(&self.ctx).all(|(&a, &v)| r.ctx(a) == v)
    }
}

impl RegRead for StageRegs {
    fn sh(&self, i: u32) -> u32 {
        let (sh, _) = reg_lists(self.kind);
        match sh.binary_search(&i) {
            Ok(k) => self.sh[k],
            Err(_) => {
                if !self.missed.swap(true, std::sync::atomic::Ordering::Relaxed) {
                    self.missed_reg.store(i, std::sync::atomic::Ordering::Relaxed);
                }
                0
            }
        }
    }
    fn ctx(&self, a: u32) -> u32 {
        let (_, ctx) = reg_lists(self.kind);
        match ctx.binary_search(&a) {
            Ok(k) => self.ctx[k],
            Err(_) => {
                if !self.missed.swap(true, std::sync::atomic::Ordering::Relaxed) {
                    self.missed_reg.store(a, std::sync::atomic::Ordering::Relaxed);
                }
                0
            }
        }
    }
}

/// Whether the guest words a preparation read are still in memory and what
/// it could not read still cannot be read (a preparation made ahead of
/// execution, or a reused one, is used only then): a data load from pages
/// the title's streaming had not mapped yet resolved to zeros, and a cached
/// preparation kept them after the pages were mapped (DS_PREP_CHECK, run
/// 20261013-064118).
pub fn words_unchanged(p: &Prepared) -> bool {
    let checked = crate::mem::MAP_GEN.load(std::sync::atomic::Ordering::Acquire) != p.map_gen;
    if checked && p.unread.iter().any(|&(a, n)| super::cp::guest_readable(a, n)) {
        return false;
    }
    let mut k = 0usize;
    for &(a, n) in p.reads.iter() {
        let m = (n / 4) as usize;
        if checked && !super::cp::guest_readable(a, n) {
            return false;
        }
        let Some(want) = p.words.get(k..k + m) else { return false };
        if super::forward::active() {
            // As the reader sees them now (`gpu::forward`).
            let mut now = vec![0u32; m];
            for (j, o) in now.iter_mut().enumerate() {
                *o = unsafe { std::ptr::read_volatile((a + 4 * j as u64) as *const u32) };
            }
            super::forward::overlay(a, &mut now);
            if now != want {
                return false;
            }
            k += m;
            continue;
        }
        for (j, &w) in want.iter().enumerate() {
            if unsafe { std::ptr::read_volatile((a + 4 * j as u64) as *const u32) } != w {
                return false;
            }
        }
        k += m;
    }
    k == p.words.len()
}

fn num_class(format: u16) -> NumClass {
    match rdna::format::gfx10(format).map(|f| f.num) {
        Some(rdna::format::Num::Uint) => NumClass::Uint,
        Some(rdna::format::Num::Sint) => NumClass::Sint,
        _ => NumClass::Float,
    }
}

/// Compute-stage registers (SH dword indices).
const CS_PGM_LO: u32 = 0x20C;
const CS_RSRC2: u32 = 0x213;
const CS_NUM_THREAD: u32 = 0x207;
const CS_USER_DATA: u32 = 0x240;

/// Resolves and translates the compute shader bound on `cp`. `wave64` comes
/// from the dispatch initiator (CS_W32_EN clear).
/// Programs dispatched in runs of consecutive direct dispatches (`tune
/// batch_cs`, `Exec::note_cs_streak`): translated batched (`Spec::batched`).
static BATCHABLE: [std::sync::atomic::AtomicU64; 64] = [const { std::sync::atomic::AtomicU64::new(0) }; 64];

/// Programs marked batchable while they had a pipeline already: batched
/// once their batched pipeline exists, made by the compile workers
/// (`compile::submit_batched`). Batched at once, the form was translated and
/// compiled at its next dispatch: 1-2's Gatehouse froze 684 ms on 2026-11-01
/// night (cs 0x138a5ab900, 661 ms on queue 0x28, the graphics CP waiting on
/// its label).
static PENDING_BATCHED: [std::sync::atomic::AtomicU64; 64] = [const { std::sync::atomic::AtomicU64::new(0) }; 64];

thread_local! {
    /// The program a compile worker prepares batched (`compile::submit_batched`).
    static FORCE_BATCHED: std::cell::Cell<u64> = const { std::cell::Cell::new(0) };
}

/// Whether program `addr` is batched (never address 0: the free slots hold 0,
/// and the precompile's specs from kept code have no address; until
/// 2026-10-30 every one of them came out batched, so no non-batched dispatch
/// found its translation made ahead).
pub fn batchable(addr: u64) -> bool {
    addr != 0 && (BATCHABLE.iter().any(|a| a.load(std::sync::atomic::Ordering::Relaxed) == addr) || FORCE_BATCHED.with(|f| f.get()) == addr)
}

/// Runs `f` on this thread with program `addr` prepared batched.
pub fn with_batched<T>(addr: u64, f: impl FnOnce() -> T) -> T {
    FORCE_BATCHED.with(|c| c.set(addr));
    let r = f();
    FORCE_BATCHED.with(|c| c.set(0));
    r
}

/// Whether program `addr` was marked batchable (batched, or waiting for its
/// batched pipeline).
pub fn batch_marked(addr: u64) -> bool {
    batchable(addr) || PENDING_BATCHED.iter().any(|a| a.load(std::sync::atomic::Ordering::Relaxed) == addr)
}

/// Marks program `addr` batchable (at most 64 programs). A program with a
/// pipeline already (`compile::has_pipeline`), or whose learned specs are all
/// unbatched (made ahead from its kept code, `precompile::learned_unbatched_only`),
/// stays unbatched until its batched pipeline is made: true when the caller
/// is to hand its stage to `compile::submit_batched` (once per program).
/// (Before 2026-11-08 a program first dispatched in a run went batched at
/// once and compiled that form at its dispatch: the suite's 284 ms frame.)
pub fn mark_batchable(addr: u64) -> bool {
    use std::sync::atomic::Ordering;
    if batch_marked(addr) {
        return false;
    }
    // (Every program since 2026-11-18, not only those with a pipeline or unbatched learned
    // specs: a new player's prepared store holds the unbatched forms, and batching a program at
    // once compiled its batched form where the dispatch stood (the follicle update 1.4 s, index
    // culling ~0.2 s in a first 1-1 walk); unbatched until its batched pipeline is made, a
    // dispatch takes the stored form.)
    if super::compile::enabled() {
        for a in &PENDING_BATCHED {
            if a.compare_exchange(0, addr, Ordering::AcqRel, Ordering::Acquire).is_ok() {
                crate::info!("gpu: cs {addr:#x} batched once its batched pipeline is made (runs of consecutive dispatches)");
                return true;
            }
            if a.load(Ordering::Acquire) == addr {
                return false;
            }
        }
    }
    mark_now(addr);
    false
}

/// The batched pipeline of a program `mark_batchable` left pending is made
/// (`ok`), or could not be (the program stays unbatched).
pub fn batched_ready(addr: u64, ok: bool) {
    if ok {
        mark_now(addr);
    } else {
        crate::info!("gpu: cs {addr:#x} stays unbatched (its batched pipeline was not made)");
    }
}

fn mark_now(addr: u64) {
    use std::sync::atomic::Ordering;
    for a in &BATCHABLE {
        if a.compare_exchange(0, addr, Ordering::AcqRel, Ordering::Acquire).is_ok() {
            BATCH_GEN.fetch_add(1, Ordering::AcqRel);
            // (Its code hash: the learned store's and the compile bench's
            // name for it, `DS_CB_HASHES`.)
            let code = program_hash(addr).map(|h| format!(", code {h:#x}")).unwrap_or_default();
            crate::info!("gpu: cs {addr:#x} batched from now on (runs of consecutive dispatches{code})");
            return;
        }
        if a.load(Ordering::Acquire) == addr {
            return;
        }
    }
}

/// Stage preparation by part (`DS_STEP_LAPS=1` or the profiler): TSC
/// cycles of each part of `PREP_PART_NAMES` per stage kind (vertex, pixel,
/// compute, compute of indirect dispatches), then preparations, those the
/// stage cache served, and the words, read ranges, loaded pointers and
/// resources of the others (`prep_lap_volume`).
static PREP_PARTS: [[std::sync::atomic::AtomicU64; PREP_PARTS_N + 6]; 4] = [const { [const { std::sync::atomic::AtomicU64::new(0) }; PREP_PARTS_N + 6] }; 4];
const PREP_PARTS_N: usize = 10;
const PREP_PART_NAMES: [&str; PREP_PARTS_N] = ["registers", "program", "spec", "plan", "cache check", "descriptors", "depth views", "translation", "scalar loads", "result and cache"];

/// A thread's preparation laps: the preparation being timed (its kind, the
/// last lap's TSC) and its sums not yet added to `PREP_PARTS`.
struct PrepLaps {
    on: bool,
    kind: usize,
    last: u64,
    acc: [[u64; PREP_PARTS_N + 6]; 4],
    n: u32,
}

thread_local! {
    static PREP_LAPS: std::cell::RefCell<PrepLaps> = const { std::cell::RefCell::new(PrepLaps { on: false, kind: 0, last: 0, acc: [[0; PREP_PARTS_N + 6]; 4], n: 0 }) };
}

thread_local! {
    /// The compute stage being prepared is an indirect dispatch's (its laps
    /// go to the fourth kind: `IndirectLaps`).
    static PREP_INDIRECT: std::cell::Cell<bool> = const { std::cell::Cell::new(false) };
}

/// While it lives, this thread's compute preparations are timed as an
/// indirect dispatch's (`DS_STEP_LAPS`).
pub struct IndirectLaps(bool);

impl IndirectLaps {
    pub fn new(indirect: bool) -> IndirectLaps {
        if indirect && prep_laps_on() {
            PREP_INDIRECT.with(|c| c.set(true));
        }
        IndirectLaps(indirect)
    }
}

impl Drop for IndirectLaps {
    fn drop(&mut self) {
        if self.0 {
            PREP_INDIRECT.with(|c| c.set(false));
        }
    }
}

fn prep_laps_on() -> bool {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ON.get_or_init(|| super::exec::profiling_on() || std::env::var("DS_STEP_LAPS").ok().as_deref() == Some("1"))
}

thread_local! {
    /// Set during the untimed first preparation of `DS_PREP_TWICE=1`.
    static PREP_WARM: std::cell::Cell<bool> = const { std::cell::Cell::new(false) };
}

/// `DS_PREP_TWICE=1` (diagnosis, with the step laps): every stage is
/// prepared once untimed before its timed preparation, whose guest reads
/// then find the caches the first one warmed (the laps' parts without
/// cache misses; frames are slower).
fn prep_warm<T>(f: impl FnOnce() -> T) {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    if *ON.get_or_init(|| std::env::var("DS_PREP_TWICE").ok().as_deref() == Some("1")) && !PREP_WARM.with(|c| c.get()) {
        PREP_WARM.with(|c| c.set(true));
        let _ = f();
        PREP_WARM.with(|c| c.set(false));
    }
}

/// Starts timing a preparation of stage kind `kind` (0 vertex, 1 pixel,
/// 2 compute).
#[inline]
fn prep_lap_begin(kind: usize) {
    if prep_laps_on() && !PREP_WARM.with(|c| c.get()) {
        PREP_LAPS.with(|l| {
            let mut l = l.borrow_mut();
            l.on = true;
            l.kind = kind;
            l.last = unsafe { core::arch::x86_64::_rdtsc() };
        });
    }
}

/// Adds the cycles since the last lap to part `part`.
#[inline]
fn prep_lap(part: usize) {
    if prep_laps_on() {
        PREP_LAPS.with(|l| {
            let mut l = l.borrow_mut();
            if l.on {
                let now = unsafe { core::arch::x86_64::_rdtsc() };
                let (k, d) = (l.kind, now.wrapping_sub(l.last));
                l.acc[k][part] += d;
                l.last = now;
            }
        });
    }
}

/// Ends a timed preparation; the sums go to `PREP_PARTS` every 512.
fn prep_lap_end() {
    if prep_laps_on() {
        PREP_LAPS.with(|l| {
            let mut l = l.borrow_mut();
            if !l.on {
                return;
            }
            l.on = false;
            let k = l.kind;
            l.acc[k][PREP_PARTS_N] += 1;
            l.n += 1;
            if l.n >= 512 {
                l.n = 0;
                let acc = std::mem::replace(&mut l.acc, [[0; PREP_PARTS_N + 6]; 4]);
                for (g, a) in PREP_PARTS.iter().zip(acc.iter()) {
                    for (c, &x) in g.iter().zip(a.iter()) {
                        c.fetch_add(x, std::sync::atomic::Ordering::Relaxed);
                    }
                }
            }
        });
    }
}

/// What a full preparation read: words, read ranges, pointers loaded, and
/// its resources.
#[inline]
fn prep_lap_volume(words: usize, ranges: usize, ptrs: usize, resources: usize) {
    if prep_laps_on() {
        PREP_LAPS.with(|l| {
            let mut l = l.borrow_mut();
            if l.on {
                let k = l.kind;
                l.acc[k][PREP_PARTS_N + 2] += words as u64;
                l.acc[k][PREP_PARTS_N + 3] += ranges as u64;
                l.acc[k][PREP_PARTS_N + 4] += ptrs as u64;
                l.acc[k][PREP_PARTS_N + 5] += resources as u64;
            }
        });
    }
}

/// `PREP_PARTS` emptied (`profile reset`).
pub fn reset_prep_parts() {
    for c in PREP_PARTS.iter().flatten().chain(TMPL_CENSUS.iter().flatten()) {
        c.store(0, std::sync::atomic::Ordering::Relaxed);
    }
    lean::reset_cycles();
}

/// The profile lines of `PREP_PARTS` ("" before any).
/// The format and swizzle each (program code, resource) read through the
/// run-time decode has seen (`tune fmt_single`): `FMT_MULTI` once it saw two.
static FMT_SEEN: parking_lot::RwLock<Option<HashMap<(u64, usize), u32>>> = parking_lot::RwLock::new(None);
const FMT_MULTI: u32 = u32::MAX;

thread_local! {
    /// This thread's view of `FMT_SEEN` (a pair it saw two formats for, or
    /// one the shared map names so).
    static FMT_LOCAL: std::cell::RefCell<HashMap<(u64, usize), u32>> = std::cell::RefCell::new(HashMap::new());
    /// Moves whenever this thread's `FMT_LOCAL` changes: the lean lane's per-resource
    /// memos of `single_format` hold while it stands (`lean::prepare`).
    static FMT_GEN: std::cell::Cell<u64> = const { std::cell::Cell::new(0) };
}

#[path = "shaders_lean.rs"]
mod lean;
pub(super) use lean::{join_content, join_items, join_keys_on, JoinFirst};

/// What a compute stage's template is checked against besides its registers (`tune join`:
/// a joined run's later commands are prepared from its first while these stand): the
/// switches' generation, the batching generation and the title's shader registrations; and
/// whether the lean preparation runs now.
pub(super) fn join_gens() -> Option<[u64; 3]> {
    lean_now().then(|| [super::tune::generation(), BATCH_GEN.load(std::sync::atomic::Ordering::Acquire), crate::hle::libs::agc::shader_registrations()])
}

/// Template lookups by stage [vertex, pixel, compute] and outcome [hit, empty slot, another
/// program in the slot, its shader registrations changed, other keys changed] (`DS_STEP_LAPS`).
static TMPL_CENSUS: [[std::sync::atomic::AtomicU64; 5]; 3] = [const { [const { std::sync::atomic::AtomicU64::new(0) }; 5] }; 3];

fn tmpl_census(stage: usize, outcome: usize) {
    if prep_laps_on() {
        TMPL_CENSUS[stage][outcome].fetch_add(1, std::sync::atomic::Ordering::Relaxed);
    }
}

fn tmpl_census_line() -> String {
    let c: Vec<Vec<u64>> = TMPL_CENSUS.iter().map(|k| k.iter().map(|x| x.load(std::sync::atomic::Ordering::Relaxed)).collect()).collect();
    if c.iter().all(|k| k[0] == 0) {
        return String::new();
    }
    let part = |k: &Vec<u64>| format!("{} hits, {} empty, {} other program, {} registrations moved, {} other keys", k[0], k[1], k[2], k[3], k[4]);
    format!("    stage templates (`DS_STEP_LAPS`): vertex {}; pixel {}; compute {}
", part(&c[0]), part(&c[1]), part(&c[2]))
}

/// `DS_PROFILE_REPEATS=1` with the profiler (the full path's repeat statistics).
fn repeat_stats_on() -> bool {
    static REPEAT_STATS: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *REPEAT_STATS.get_or_init(|| std::env::var("DS_PROFILE_REPEATS").ok().as_deref() == Some("1")) && super::exec::profiling_on()
}

/// A stage's preparation through the lean lane when it is on (`tune lean`, `lean::prepare`),
/// else (or for what it leaves) through `prepare`; `DS_LEAN_CHECK=1` makes both and compares.
#[allow(clippy::too_many_arguments)]
fn prepare_stage(addr: u64, insts: Arc<Local<Vec<rdna::Inst>>>, code_hash: u64, user: &[u32], spec: &mut Spec, tmpl: u64, tkey: u64) -> Result<Prepared, String> {
    if tkey == 0 || !super::tune::lean() || repeat_stats_on() {
        let t0 = unsafe { core::arch::x86_64::_rdtsc() };
        let r = prepare(addr, insts, code_hash, user, spec, tmpl);
        lean::time(1, unsafe { core::arch::x86_64::_rdtsc() }.wrapping_sub(t0));
        return r;
    }
    if lean::check_on() {
        let a = lean::prepare(addr, &insts, code_hash, user, spec.stage, spec.user_sgprs, spec.user_base, tkey);
        let b = prepare(addr, insts.clone(), code_hash, user, spec, tmpl);
        if let (Some(Ok(a)), Ok(b)) = (&a, &b) {
            // (A difference in the words alone is checked against a second lean read: the
            // title's data read twice a few microseconds apart.)
            if !lean::compare(a, b, false) {
                if let Some(Ok(a2)) = lean::prepare(addr, &insts, code_hash, user, spec.stage, spec.user_sgprs, spec.user_base, tkey) {
                    lean::compare(&a2, b, true);
                }
            }
        }
        if let Ok(p) = &b {
            lean::learn(p, spec, tkey);
        }
        return b;
    }
    let t0 = unsafe { core::arch::x86_64::_rdtsc() };
    let r = prepare(addr, insts, code_hash, user, spec, tmpl);
    lean::time(2, unsafe { core::arch::x86_64::_rdtsc() }.wrapping_sub(t0));
    if let Ok(p) = &r {
        lean::learn(p, spec, tkey);
    }
    r
}

/// Whether a template hit tries the lean lane first (`tune lean`, not in its check mode).
fn lean_now() -> bool {
    super::tune::lean() && !lean::check_on() && !repeat_stats_on()
}

/// A template's content key for the lean lane: its specialization without resources and
/// program address, hashed (0, no lean preparations, in the diagnosis modes that key
/// translations by address).
fn tmpl_key(spec: &Spec) -> u64 {
    static BY_ADDRESS: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    if *BY_ADDRESS.get_or_init(|| std::env::var_os("DS_WAVEMASK").is_some() || std::env::var_os("DS_LOCKSTEP_OFF").is_some()) {
        return 0;
    }
    let mut s = spec.clone();
    s.resources = Vec::new();
    s.code_base = 0;
    let mut h = super::fx::FxHasher::default();
    s.hash(&mut h);
    h.finish() | 1
}

/// Seeds the formats a program's learned specs (`specs`, each a resource
/// list) give its formatted buffers: one format and swizzle in all of them a
/// single format, two different ones several (run-time); specs keyed with the
/// run-time decode say nothing. Pairs already seen are left as they are.
pub(crate) fn seed_formats(code: u64, specs: &[&[ResSpec]]) {
    // `DS_FMT_SEED=0` (diagnosis): nothing seeded, so precompiles key
    // format-loaded buffers run-time as specs learned before 2026-11-06 do,
    // and dispatches meet their specialized keys as first uses
    // (`fmt_run_time`).
    static SEED: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    if !*SEED.get_or_init(|| std::env::var("DS_FMT_SEED").ok().as_deref() != Some("0")) {
        return;
    }
    let n = specs.iter().map(|s| s.len()).max().unwrap_or(0);
    for i in 0..n {
        let mut one: Option<u32> = None;
        let mut multi = false;
        for s in specs {
            if let Some(ResSpec::Buffer { format, dst_sel }) = s.get(i) {
                if *format != 0 && *format != translate::FORMAT_RT {
                    let v = (*format as u32) << 16 | *dst_sel as u32;
                    match one {
                        None => one = Some(v),
                        Some(o) if o != v => multi = true,
                        _ => {}
                    }
                }
            }
        }
        let Some(v) = one else { continue };
        let v = if multi { FMT_MULTI } else { v };
        FMT_SEEN.write().get_or_insert_with(HashMap::new).entry((code, i)).or_insert(v);
    }
}

/// Whether `(code, res)` holds `format` and `dst_sel` as its one format
/// (`seed_formats`, or the dispatches seen so far).
fn seeded_single(code: u64, res: usize, format: u8, dst_sel: u16) -> bool {
    let want = (format as u32) << 16 | dst_sel as u32;
    FMT_SEEN.read().as_ref().and_then(|m| m.get(&(code, res)).copied()) == Some(want)
}

thread_local! {
    /// Set while `with_fmt_rt` runs.
    static FMT_FORCE_RT: std::cell::Cell<bool> = const { std::cell::Cell::new(false) };
}

/// Runs `f` with the compute preparations it makes taking every formatted buffer at run time
/// (`translate::FORMAT_RT`), as the run-time variant a missing specialized key falls back to
/// (`fmt_run_time`), without noting formats (`single_format`): a batched program's run-time
/// variant is made beside its specialized one in the background (`gpu::compile`; 2026-11-19:
/// 30 of a first walk's 48 run-time variant misses were batched dispatches, whose run-time
/// variant the launcher's preparation never makes).
pub fn with_fmt_rt<T>(f: impl FnOnce() -> T) -> T {
    FMT_FORCE_RT.with(|c| c.set(true));
    let r = f();
    FMT_FORCE_RT.with(|c| c.set(false));
    r
}

/// Whether resource `res` of the program with code `code` can be specialized
/// to `format` and `dst_sel`: the first pair seen for it, or the same as
/// before; a different one makes it run-time from now on (every thread). A
/// specialized translation is used only for V#s of its own format.
fn single_format(code: u64, res: usize, format: u8, dst_sel: u16) -> bool {
    if FMT_FORCE_RT.with(|c| c.get()) {
        return false;
    }
    // (Diagnosis, `DS_FMT_RT_ONLY=1`: every typed buffer through the run-time
    // decode, never specialized: checks of `FORMAT_RT` loads and stores.)
    static RT_ONLY: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    if *RT_ONLY.get_or_init(|| std::env::var("DS_FMT_RT_ONLY").ok().as_deref() == Some("1")) {
        return false;
    }
    let want = (format as u32) << 16 | dst_sel as u32;
    let key = (code, res);
    let local = FMT_LOCAL.with(|l| l.borrow().get(&key).copied());
    let seen = match local {
        Some(v) => v,
        None => {
            let shared = FMT_SEEN.read().as_ref().and_then(|m| m.get(&key).copied());
            let v = match shared {
                Some(v) => v,
                None => *FMT_SEEN.write().get_or_insert_with(HashMap::new).entry(key).or_insert(want),
            };
            FMT_LOCAL.with(|l| l.borrow_mut().insert(key, v));
            FMT_GEN.with(|g| g.set(g.get() + 1));
            v
        }
    };
    if seen == want {
        return true;
    }
    if seen != FMT_MULTI {
        // Another format: run-time for every thread from now on.
        let shared = *FMT_SEEN.write().get_or_insert_with(HashMap::new).entry(key).and_modify(|v| {
            if *v != want {
                *v = FMT_MULTI;
            }
        }).or_insert(FMT_MULTI);
        FMT_LOCAL.with(|l| l.borrow_mut().insert(key, shared));
        FMT_GEN.with(|g| g.set(g.get() + 1));
        return shared == want;
    }
    false
}

/// Specialized compute keys whose translation and pipeline are being made
/// in the background (`fmt_later`), by (plan key, spec hash); their count.
/// Until then their preparations take the run-time variant (`fmt_run_time`).
static FMT_PENDING: Mutex<Option<std::collections::HashSet<(u64, u64)>>> = Mutex::new(None);
static FMT_PENDING_N: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);
/// Keys whose background making failed: made in place from then on (which
/// reports the error).
static FMT_FAILED: Mutex<Option<std::collections::HashSet<(u64, u64)>>> = Mutex::new(None);
/// The background jobs, their workers, and [preparations given the run-time
/// variant, specialized keys queued, made, ns making them, failed, first
/// uses made in place (no run-time variant ready), run-time variants still
/// compiling ahead, run-time variants made from the store].
struct FmtJob {
    addr: u64,
    insts: Arc<Vec<rdna::Inst>>,
    plan: Arc<Plan>,
    plan_key: u64,
    spec_hash: u64,
    spec: Spec,
    key_spec: Spec,
}
static FMT_JOBS: Mutex<std::collections::VecDeque<FmtJob>> = Mutex::new(std::collections::VecDeque::new());
static FMT_CV: parking_lot::Condvar = parking_lot::Condvar::new();
static FMT_WORKERS: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);
static FMT_STATS: [std::sync::atomic::AtomicU64; 8] = [const { std::sync::atomic::AtomicU64::new(0) }; 8];

/// Whether a specialized key is being made in the background.
fn fmt_pending(plan_key: u64, spec_hash: u64) -> bool {
    FMT_PENDING_N.load(std::sync::atomic::Ordering::Acquire) != 0 && FMT_PENDING.lock().as_ref().is_some_and(|s| s.contains(&(plan_key, spec_hash)))
}

/// Whether making a specialized key in the background failed.
fn fmt_failed(plan_key: u64, spec_hash: u64) -> bool {
    FMT_STATS[4].load(std::sync::atomic::Ordering::Relaxed) != 0 && FMT_FAILED.lock().as_ref().is_some_and(|s| s.contains(&(plan_key, spec_hash)))
}

/// The run-time variant of a compute stage whose buffers `specialized` (a
/// bit per resource) were specialized to their one format and whose
/// specialized translation is missing: its translation when it and its
/// pipeline are made, or when the store holds its pipeline (else None: the
/// specialized one is made in place). `key_spec` is the specialized key
/// spec, `code_base` the dispatch's.
#[allow(clippy::too_many_arguments)]
fn fmt_run_time(key_spec: &Spec, specialized: u64, plan_key: u64, addr: u64, insts: &[rdna::Inst], plan: &Arc<Plan>, code_base: u64) -> Option<Arc<Translated>> {
    let mut rt = key_spec.clone();
    for (i, r) in rt.resources.iter_mut().enumerate().take(64) {
        if specialized & (1 << i) != 0 {
            *r = ResSpec::Buffer { format: translate::FORMAT_RT, dst_sel: 0 };
        }
    }
    let h = {
        let mut h = std::collections::hash_map::DefaultHasher::new();
        rt.hash(&mut h);
        h.finish()
    };
    let t = LOCAL_TRANSLATIONS.with(|c| c.borrow().get(&(plan_key, h)).cloned()).or_else(|| TRANSLATIONS.lock().as_ref().and_then(|m| m.get(&(plan_key, h)).cloned()));
    if let Some(t) = &t {
        if t.spec != rt {
            return None;
        }
        if super::exec::compute_pipe_ready(t) {
            return Some(t.clone());
        }
        // (Its pipeline still compiling ahead of use, as after a level load:
        // the dispatch waits for that compile, begun earlier, instead of
        // compiling the specialized one where it stands.)
        if super::exec::compute_pipe_in_flight(t) {
            FMT_STATS[6].fetch_add(1, std::sync::atomic::Ordering::Relaxed);
            return Some(t.clone());
        }
    }
    // Its pipeline in the store, as the launcher's shader preparation leaves
    // the run-time variants (it cannot know the formats dispatches bind):
    // made from its binaries in milliseconds instead of a compile of the
    // specialized key where the dispatch stands (2026-11-11: 1,032 of a
    // prepared session's 1,080 compute first uses). Both push modes are
    // looked up before translating.
    if !super::exec::compute_key_stored(plan_key ^ h.rotate_left(1)) {
        // (Diagnosis, DS_FMT_MISS_TRACE=1: the preparation's guess for the same program against
        // the run-time variant looked up.)
        static LIMIT: std::sync::OnceLock<u32> = std::sync::OnceLock::new();
        let limit = *LIMIT.get_or_init(|| match std::env::var("DS_FMT_MISS_TRACE").ok().and_then(|v| v.parse::<u32>().ok()) {
            Some(1) => 24,
            Some(n) => n,
            None => 0,
        });
        if limit > 0 {
            static N: std::sync::atomic::AtomicU32 = std::sync::atomic::AtomicU32::new(0);
            if N.fetch_add(1, std::sync::atomic::Ordering::Relaxed) < limit {
                let code_hash = plan_key ^ (rt.user_sgprs as u64) << 48 ^ (rt.user_base as u64) << 56;
                let mut g = guess_compute_spec(0, insts, plan, code_hash, rt.cs_rsrc2, rt.cs_local_size);
                if g.lod_stats && !g.batched && cs_lod_rotation() && !rt.lod_stats {
                    g.lod_stats = false;
                    g.lod_lean = false;
                }
                let gh = {
                    let mut x = std::collections::hash_map::DefaultHasher::new();
                    g.hash(&mut x);
                    x.finish()
                };
                let stored = super::exec::compute_key_stored(plan_key ^ gh.rotate_left(1));
                crate::info!("fmt miss: cs {addr:#x} code {code_hash:#x} rsrc2 {:#x} local {:?} user_base {}: guess stored {stored}, guess == run-time {}; differs: {}", rt.cs_rsrc2, rt.cs_local_size, rt.user_base, g == rt, super::exec::spec_diff(&g, &rt));
                // (`DS_FMT_MISS_SPEC=1`: the run-time variant's whole spec, to set beside the
                // preparation's `prep obj:` line, `DS_CB_OBJ`.)
                if std::env::var("DS_FMT_MISS_SPEC").ok().as_deref() == Some("1") {
                    crate::info!("fmt miss spec: cs {addr:#x} code {code_hash:#x}: {rt:?}");
                }
            }
        }
        return None;
    }
    let t = match t {
        Some(t) => t,
        None => {
            let mut spec = rt.clone();
            spec.code_base = code_base;
            let t = translate_kept(addr, insts, plan, plan_key, h, &spec, rt).ok()?;
            LOCAL_TRANSLATIONS.with(|c| c.borrow_mut().insert((plan_key, h), t.clone()));
            t
        }
    };
    if !super::exec::compute_pipe_stored(&t) {
        return None;
    }
    FMT_STATS[7].fetch_add(1, std::sync::atomic::Ordering::Relaxed);
    Some(t)
}

/// Queues a specialized key's translation and pipeline (false when it is
/// already queued); its preparations take the run-time variant until both
/// are made.
fn fmt_later(job: FmtJob) -> bool {
    use std::sync::atomic::Ordering;
    {
        let mut g = FMT_PENDING.lock();
        if !g.get_or_insert_with(std::collections::HashSet::new).insert((job.plan_key, job.spec_hash)) {
            return false;
        }
        FMT_PENDING_N.fetch_add(1, Ordering::AcqRel);
    }
    FMT_STATS[1].fetch_add(1, Ordering::Relaxed);
    FMT_JOBS.lock().push_back(job);
    // Two workers below normal priority (an area's first uses come
    // together; the largest programs take seconds).
    while FMT_WORKERS.load(Ordering::Relaxed) < 2 {
        let i = FMT_WORKERS.fetch_add(1, Ordering::Relaxed);
        if i >= 2 {
            break;
        }
        let name = format!("gpu-fmt{i}");
        let _ = crate::thread::spawn_host(&name.clone(), 8 << 20, Box::new(move || fmt_worker(name)));
    }
    FMT_CV.notify_one();
    true
}

fn fmt_worker(name: String) {
    use std::sync::atomic::Ordering;
    crate::log::set_thread_name(&name);
    unsafe {
        windows_sys::Win32::System::Threading::SetThreadPriority(windows_sys::Win32::System::Threading::GetCurrentThread(), windows_sys::Win32::System::Threading::THREAD_PRIORITY_BELOW_NORMAL);
    }
    loop {
        let j = {
            let mut g = FMT_JOBS.lock();
            loop {
                if let Some(j) = g.pop_front() {
                    break j;
                }
                FMT_CV.wait(&mut g);
            }
        };
        let t = std::time::Instant::now();
        let r = translate_kept(j.addr, &j.insts, &j.plan, j.plan_key, j.spec_hash, &j.spec, j.key_spec).and_then(|tr| super::exec::precompile_translation(&tr, j.addr).ok_or_else(|| "pipeline creation failed".to_string()));
        match r {
            Ok(_) => {
                FMT_STATS[2].fetch_add(1, Ordering::Relaxed);
                FMT_STATS[3].fetch_add(t.elapsed().as_nanos() as u64, Ordering::Relaxed);
            }
            // (Its next preparation makes it in place and reports why.)
            Err(e) => {
                FMT_FAILED.lock().get_or_insert_with(std::collections::HashSet::new).insert((j.plan_key, j.spec_hash));
                if FMT_STATS[4].fetch_add(1, Ordering::Relaxed) < 8 {
                    crate::warn!("gpu: making cs {:#x}'s specialized translation in the background failed: {e}", j.addr);
                }
            }
        }
        if FMT_PENDING.lock().as_mut().is_some_and(|s| s.remove(&(j.plan_key, j.spec_hash))) {
            FMT_PENDING_N.fetch_sub(1, Ordering::AcqRel);
        }
    }
}

/// Status line: first uses of specialized compute keys.
pub fn fmt_line() -> String {
    let v: Vec<u64> = FMT_STATS.iter().map(|c| c.load(std::sync::atomic::Ordering::Relaxed)).collect();
    if v.iter().all(|&x| x == 0) {
        return String::new();
    }
    format!(
        "    compute keys specialized to one format, first uses: {} made in the background ({} made, {:.1} s; {} failed; {} pending now), {} preparations ran the run-time variant meanwhile ({} of them with its pipeline still compiling ahead; {} first uses made it from the store); {} made in place (no run-time variant)\n",
        v[1],
        v[2],
        v[3] as f64 / 1e9,
        v[4],
        FMT_PENDING_N.load(std::sync::atomic::Ordering::Relaxed),
        v[0],
        v[6],
        v[7],
        v[5]
    )
}

/// `tune prep_memo` (diagnosis): stage preparations memoized by key during
/// replay iterations (`prep_memo_key`), in shards; [copies served, kept].
static PREP_MEMO: [Mutex<Option<super::fx::Map<u64, Prepared>>>; 64] = [const { Mutex::new(None) }; 64];
static PREP_MEMO_STATS: [std::sync::atomic::AtomicU64; 2] = [const { std::sync::atomic::AtomicU64::new(0) }; 2];

/// Empties the memo (a replay starts).
pub fn prep_memo_clear() {
    for s in &PREP_MEMO {
        *s.lock() = None;
    }
}

/// The memo key of a stage preparation at `packet`: the registers it reads
/// and `extra` (how it is prepared).
pub fn prep_memo_key<R: RegRead + ?Sized>(kind: StageKind, packet: u64, r: &R, extra: u64) -> u64 {
    let (sh, ctx) = reg_lists(kind);
    let mut h = super::fx::FxHasher::default();
    h.write_u64(packet);
    h.write_u64(kind as u64);
    h.write_u64(extra);
    for &i in sh {
        h.write_u32(r.sh(i));
    }
    for &a in ctx {
        h.write_u32(r.ctx(a));
    }
    h.finish()
}

/// The memoized preparation of `key` (a copy), else `f`'s, kept.
pub fn prep_memo(key: u64, f: impl FnOnce() -> Result<Prepared, String>) -> Result<Prepared, String> {
    use std::sync::atomic::Ordering;
    let shard = &PREP_MEMO[((key ^ (key >> 32)) & 63) as usize];
    if let Some(p) = shard.lock().as_ref().and_then(|m| m.get(&key).cloned()) {
        PREP_MEMO_STATS[0].fetch_add(1, Ordering::Relaxed);
        return Ok(p);
    }
    let p = f()?;
    shard.lock().get_or_insert_with(Default::default).insert(key, p.clone());
    PREP_MEMO_STATS[1].fetch_add(1, Ordering::Relaxed);
    Ok(p)
}

/// Profile line of the memo ("" when unused).
pub fn prep_memo_line() -> String {
    let v: Vec<u64> = PREP_MEMO_STATS.iter().map(|c| c.load(std::sync::atomic::Ordering::Relaxed)).collect();
    if v[0] == 0 && v[1] == 0 {
        return String::new();
    }
    format!("    replay preparation memo (`tune prep_memo`): {} copies served, {} kept
", v[0], v[1])
}

/// `DS_FMT_CENSUS=1` (diagnosis): the formats and swizzles compute stages
/// read through `FORMAT_RT`, per (program code, resource): [(format,
/// dst_sel) -> preparations], with the program's address.
static FMT_CENSUS: parking_lot::Mutex<Option<HashMap<(u64, usize), (u64, HashMap<(u16, u16), u64>)>>> = parking_lot::Mutex::new(None);

fn fmt_census_on() -> bool {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ON.get_or_init(|| std::env::var("DS_FMT_CENSUS").ok().as_deref() == Some("1"))
}

fn fmt_census_note(code: u64, addr: u64, res: usize, format: u16, dst_sel: u16) {
    let mut g = FMT_CENSUS.lock();
    let e = g.get_or_insert_with(HashMap::new).entry((code, res)).or_insert_with(|| (addr, HashMap::new()));
    *e.1.entry((format, dst_sel)).or_default() += 1;
}

/// The census's lines (`DS_FMT_CENSUS=1`; in the step laps' report): per
/// program, its run-time-format resources by preparations, the formats seen.
pub fn fmt_census_lines() -> String {
    let g = FMT_CENSUS.lock();
    let Some(m) = g.as_ref() else { return String::new() };
    // Per program: preparations (the most counted resource's), resources.
    let mut progs: HashMap<u64, (u64, u64, Vec<(usize, Vec<((u16, u16), u64)>)>)> = HashMap::new();
    for (&(code, res), (addr, f)) in m.iter() {
        let n: u64 = f.values().sum();
        let e = progs.entry(code).or_insert((*addr, 0, Vec::new()));
        e.1 = e.1.max(n);
        let mut fs: Vec<((u16, u16), u64)> = f.iter().map(|(k, v)| (*k, *v)).collect();
        fs.sort_by(|a, b| b.1.cmp(&a.1));
        e.2.push((res, fs));
    }
    let mut v: Vec<_> = progs.into_iter().collect();
    v.sort_by(|a, b| b.1 .1.cmp(&a.1 .1));
    let (single, multi) = v.iter().fold((0u64, 0u64), |(s, m), (_, (_, n, rs))| if rs.iter().all(|(_, fs)| fs.len() == 1) { (s + n, m) } else { (s, m + n) });
    let mut out = format!("    run-time formats (`DS_FMT_CENSUS`): {} programs; preparations of programs whose every such resource saw one format {}, of the others {}; the most prepared:\n", v.len(), single, multi);
    for (code, (addr, n, mut rs)) in v.into_iter().take(40) {
        rs.sort_by_key(|r| r.0);
        let parts: Vec<String> = rs.iter().map(|(r, fs)| format!("#{r} {}", fs.iter().map(|((f, d), c)| format!("fmt {f} sel {d:#05x} x{c}")).collect::<Vec<_>>().join(" / "))).collect();
        out.push_str(&format!("      cs {addr:#x} (code {code:#x}) {n} preparations: {}\n", parts.join("; ")));
    }
    out
}

pub fn prep_parts_line() -> String {
    let hz = super::threadcpu::tsc_hz();
    let mut s = lean::status() + &tmpl_census_line();
    for (k, name) in ["vertex", "pixel", "compute", "compute, indirect dispatches"].iter().enumerate() {
        let v: Vec<u64> = PREP_PARTS[k].iter().map(|c| c.load(std::sync::atomic::Ordering::Relaxed)).collect();
        let n = v[PREP_PARTS_N];
        if n == 0 {
            continue;
        }
        let ns = |c: u64| c as f64 * 1e9 / hz / n as f64;
        let total: u64 = v[..PREP_PARTS_N].iter().sum();
        let parts: Vec<String> = PREP_PART_NAMES.iter().zip(v.iter()).map(|(p, &c)| format!("{p} {:.0}", ns(c))).collect();
        let full = n.max(1) as f64;
        s.push_str(&format!(
            "    stage preparation by part, {name} ({n} preparations; {:.0} ns each): {}; one read {:.1} words in {:.1} ranges, {:.1} pointers, {:.1} resources\n",
            ns(total),
            parts.join(", "),
            v[PREP_PARTS_N + 2] as f64 / full,
            v[PREP_PARTS_N + 3] as f64 / full,
            v[PREP_PARTS_N + 4] as f64 / full,
            v[PREP_PARTS_N + 5] as f64 / full
        ));
    }
    s
}

pub fn prepare_compute<R: RegRead + ?Sized>(cp: &R, wave64: bool) -> Result<Prepared, String> {
    if super::tune::r2() && matches!(super::tune::gpu_chains(), 0 | 3) && !PROBE_ON.load(std::sync::atomic::Ordering::Relaxed) && FORCE_BATCHED.with(|f| f.get()) == 0 {
        return prepare_compute_r2(cp, wave64);
    }
    prep_warm(|| prepare_compute(cp, wave64));
    prep_lap_begin(if prep_laps_on() && PREP_INDIRECT.with(|c| c.get()) { 3 } else { 2 });
    let sh = |i: u32| cp.sh(i);
    let addr = ((sh(CS_PGM_LO) as u64) | ((sh(CS_PGM_LO + 1) as u64 & 0xFF) << 32)) << 8;
    let rsrc2 = sh(CS_RSRC2);
    let user: [u32; 16] = std::array::from_fn(|i| sh(CS_USER_DATA + i as u32));
    let local = [sh(CS_NUM_THREAD) & 0xFFFF, sh(CS_NUM_THREAD + 1) & 0xFFFF, sh(CS_NUM_THREAD + 2) & 0xFFFF];
    prep_lap(0);
    let (insts, code_hash) = program_local(addr)?;
    prep_lap(1);
    let mut spec = cs_spec(addr, rsrc2, local, wave64, code_hash);
    prep_lap(2);
    let r = prepare(addr, insts, code_hash, &user, &mut spec, 0);
    spec_vecs_back(spec);
    r
}

/// `DS_R2_CHECK=1` (diagnosis): R2's templates checked at every hit against
/// the spec built anew (`r2_check_spec`; mismatches logged, the first 40,
/// and counted in `R2_CHECKS`).
pub fn r2_check() -> bool {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ON.get_or_init(|| std::env::var("DS_R2_CHECK").ok().as_deref() == Some("1"))
}

/// `DS_R2_ARMED_CENSUS=1` (diagnosis): the ranges stage preparations read,
/// by whether front end 2 and the host have their pages armed now (a CPU
/// write faults first): [vertex, pixel, compute] x [descriptors, scalar
/// loads] x [ranges, of them armed, bytes, of them armed].
pub fn armed_census_on() -> bool {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ON.get_or_init(|| std::env::var("DS_R2_ARMED_CENSUS").ok().as_deref() == Some("1"))
}

pub static ARMED_CENSUS: [[[std::sync::atomic::AtomicU64; 4]; 2]; 3] = [const { [const { [const { std::sync::atomic::AtomicU64::new(0) }; 4] }; 2] }; 3];

/// Counts a preparation's reads (`ARMED_CENSUS`): the first `res` were the
/// resources' descriptors, the rest its scalar loads.
fn armed_census(stage: Stage, reads: &[(u64, u64)], res: usize) {
    use std::sync::atomic::Ordering::Relaxed;
    let st = match stage {
        Stage::Vertex => 0,
        Stage::Fragment => 1,
        Stage::Compute => 2,
    };
    for (k, &(a, n)) in reads.iter().enumerate() {
        let c = &ARMED_CENSUS[st][if k < res { 0 } else { 1 }];
        let armed = super::fe2::range_armed(a, n);
        c[0].fetch_add(1, Relaxed);
        c[2].fetch_add(n, Relaxed);
        if armed {
            c[1].fetch_add(1, Relaxed);
            c[3].fetch_add(n, Relaxed);
        }
    }
}

/// [template hits checked, mismatches] (`DS_R2_CHECK=1`).
pub static R2_CHECKS: [std::sync::atomic::AtomicU64; 2] = [const { std::sync::atomic::AtomicU64::new(0) }; 2];

/// Compares a template's spec (`got`, its pixel inputs filled, no
/// resources) with the one built anew (`fresh`), whose vectors go back to
/// the pool.
fn r2_check_spec(kind: &str, addr: u64, got: &Spec, fresh: Spec) {
    use std::sync::atomic::Ordering;
    R2_CHECKS[0].fetch_add(1, Ordering::Relaxed);
    let mut a = got.clone();
    a.resources.clear();
    let same = a == fresh;
    if !same {
        let n = R2_CHECKS[1].fetch_add(1, Ordering::Relaxed);
        if n < 40 {
            crate::info!("r2 check: {kind} {addr:#x}: the template's spec differs from the spec built now: template {a:?}; now {fresh:?}");
        }
    }
    spec_vecs_back(fresh);
}

/// R2 (`tune r2`): a compute stage's program and specialization template
/// (`cs_spec` without resources), kept per thread by what they derive from:
/// the program's address and registers, the shader registrations, the
/// switches' generation, the batchable programs' generation and the LOD
/// measuring flag (probes, forced batching and GPU-resolved chains take the
/// full path). The spec is then cloned instead of built (~25 switch reads,
/// two 64-entry batchable scans and four program-list scans a stage).
struct CsTmpl {
    id: u64,
    /// The specialization's content key for the lean lane (`tmpl_key`).
    tkey: u64,
    addr: u64,
    rsrc2: u32,
    local: [u32; 3],
    wave64: bool,
    lod_measure: bool,
    tune_gen: u64,
    batch_gen: u64,
    regs: u64,
    insts: Option<Arc<Local<Vec<rdna::Inst>>>>,
    code_hash: u64,
    spec: Option<Box<Spec>>,
    /// `tmpl_tick` at its last use (`tmpl_way`).
    used: std::cell::Cell<u32>,
}

const CS_TMPL_SLOTS: usize = 2048;

thread_local! {
    static CS_TMPLS: std::cell::RefCell<Vec<CsTmpl>> = std::cell::RefCell::new(Vec::new());
}

/// Entries per set of the stage template tables (`CsTmpl`, `PsTmpl`, `NggTmpl`): a program's
/// set by its address, the least recently used entry replaced. (Direct-mapped tables of 256-512
/// slots missed ~1,260 times a flip at Castlewall turning against ~1,700 programs, 84% of the
/// stage preparations left to the full path at ~4.4 us each: FINDINGS.md 2026-11-16 afternoon.)
const TMPL_WAYS: usize = 4;

thread_local! {
    /// This thread's template use counter (`tmpl_way`).
    static TMPL_TICK: std::cell::Cell<u32> = const { std::cell::Cell::new(0) };
}

/// The next use tick of this thread's templates.
fn tmpl_tick() -> u32 {
    TMPL_TICK.with(|c| {
        let v = c.get().wrapping_add(1);
        c.set(v);
        v
    })
}

/// The entry of set `set` holding program `addr`, else the one to replace (an empty one or the
/// least recently used), in a table of `len` entries (`at(i)`: entry i's program and last use);
/// an empty table's first entry of the set (filled when the table is made).
fn tmpl_way(len: usize, at: impl Fn(usize) -> (u64, u32), addr: u64, set: usize) -> usize {
    let base = set * TMPL_WAYS;
    if base + TMPL_WAYS > len {
        return base;
    }
    let now = TMPL_TICK.with(|c| c.get());
    let (mut victim, mut age) = (base, 0u32);
    for i in base..base + TMPL_WAYS {
        let (a, used) = at(i);
        if a == addr {
            return i;
        }
        let ag = if a == 0 { u32::MAX } else { now.wrapping_sub(used) };
        if ag >= age {
            (victim, age) = (i, ag);
        }
    }
    victim
}

/// Bumped when a program is marked batchable (`mark_now`): R2's compute
/// templates made before are made again (`CsTmpl`).
static BATCH_GEN: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);

fn prepare_compute_r2<R: RegRead + ?Sized>(cp: &R, wave64: bool) -> Result<Prepared, String> {
    use std::sync::atomic::Ordering;
    prep_warm(|| prepare_compute(cp, wave64));
    prep_lap_begin(if prep_laps_on() && PREP_INDIRECT.with(|c| c.get()) { 3 } else { 2 });
    let sh = |i: u32| cp.sh(i);
    let addr = ((sh(CS_PGM_LO) as u64) | ((sh(CS_PGM_LO + 1) as u64 & 0xFF) << 32)) << 8;
    let rsrc2 = sh(CS_RSRC2);
    let user: [u32; 16] = std::array::from_fn(|i| sh(CS_USER_DATA + i as u32));
    let local = [sh(CS_NUM_THREAD) & 0xFFFF, sh(CS_NUM_THREAD + 1) & 0xFFFF, sh(CS_NUM_THREAD + 2) & 0xFFFF];
    prep_lap(0);
    let lod_measure = LOD_MEASURE.with(|c| c.get());
    let tune_gen = super::tune::generation();
    let batch_gen = BATCH_GEN.load(Ordering::Acquire);
    let regs = crate::hle::libs::agc::shader_registrations();
    let set = ((addr >> 8) ^ (addr >> 19)) as usize & (CS_TMPL_SLOTS / TMPL_WAYS - 1);
    let slot = CS_TMPLS.with(|t| {
        let t = t.borrow();
        tmpl_way(t.len(), |i| (t[i].addr, t[i].used.get()), addr, set)
    });
    // The lean lane first, from the template without cloning its specialization.
    if lean_now() {
        let t = CS_TMPLS.with(|t| {
            let t = t.borrow();
            let e = t.get(slot)?;
            let s = e.spec.as_ref()?;
            (e.addr == addr && e.rsrc2 == rsrc2 && e.local == local && e.wave64 == wave64 && e.lod_measure == lod_measure && e.tune_gen == tune_gen && e.batch_gen == batch_gen && e.regs == regs && e.tkey != 0).then(|| {
                e.used.set(tmpl_tick());
                (e.insts.clone(), e.code_hash, e.tkey, s.user_sgprs, s.user_base)
            })
        });
        if let Some((Some(insts), code_hash, tkey, us, ub)) = t {
            if let Some(r) = lean::prepare(addr, &insts, code_hash, &user, Stage::Compute, us, ub, tkey) {
                prep_lap_end();
                return r;
            }
        }
    }
    let hit = CS_TMPLS.with(|t| {
        let t = t.borrow();
        let Some(e) = t.get(slot) else {
            tmpl_census(2, 1);
            return None;
        };
        tmpl_census(2, if e.spec.is_none() { 1 } else if e.addr != addr { 2 } else if e.regs != regs { 3 } else if e.rsrc2 == rsrc2 && e.local == local && e.wave64 == wave64 && e.lod_measure == lod_measure && e.tune_gen == tune_gen && e.batch_gen == batch_gen { 0 } else { 4 });
        if e.addr == addr && e.rsrc2 == rsrc2 && e.local == local && e.wave64 == wave64 && e.lod_measure == lod_measure && e.tune_gen == tune_gen && e.batch_gen == batch_gen && e.regs == regs {
            e.used.set(tmpl_tick());
            Some((e.insts.clone()?, e.code_hash, (**e.spec.as_ref()?).clone(), (e.id, e.tkey)))
        } else {
            None
        }
    });
    let (insts, code_hash, mut spec, (id, tkey)) = match hit {
        Some((insts, code_hash, mut spec, id)) => {
            prep_lap(1);
            if r2_check() {
                r2_check_spec("cs", addr, &spec, cs_spec(addr, rsrc2, local, wave64, code_hash));
            }
            let (ps_inputs, resources) = spec_vecs();
            spec.ps_inputs = ps_inputs;
            spec.resources = resources;
            (insts, code_hash, spec, id)
        }
        None => {
            let (insts, code_hash) = program_local(addr)?;
            prep_lap(1);
            let spec = cs_spec(addr, rsrc2, local, wave64, code_hash);
            let id = next_tmpl_id();
            let tkey = if LOCAL_PROGRAMS.with(|c| c.borrow().contains_key(&addr)) { tmpl_key(&spec) } else { 0 };
            // (The template: the switches, batchable programs and shader
            // registrations as they were before the spec read them. Only for
            // a program from a shader object, which `program_local` keeps
            // until another registration: others are checked by hashing
            // their code at every preparation.)
            let registered = LOCAL_PROGRAMS.with(|c| c.borrow().contains_key(&addr));
            CS_TMPLS.with(|t| {
                if !registered {
                    return;
                }
                let mut t = t.borrow_mut();
                if t.is_empty() {
                    t.resize_with(CS_TMPL_SLOTS, || CsTmpl { id: 0, tkey: 0, addr: 0, rsrc2: 0, local: [0; 3], wave64: false, lod_measure: false, tune_gen: 0, batch_gen: 0, regs: 0, insts: None, code_hash: 0, spec: None, used: std::cell::Cell::new(0) });
                }
                let mut tmpl = spec.clone();
                tmpl.ps_inputs = Vec::new();
                tmpl.resources = Vec::new();
                t[slot] = CsTmpl { id, tkey, addr, rsrc2, local, wave64, lod_measure, tune_gen, batch_gen, regs, insts: Some(insts.clone()), code_hash, spec: Some(Box::new(tmpl)), used: std::cell::Cell::new(tmpl_tick()) };
            });
            (insts, code_hash, spec, (id, tkey))
        }
    };
    prep_lap(2);
    let r = prepare_stage(addr, insts, code_hash, &user, &mut spec, id, tkey);
    spec_vecs_back(spec);
    r
}

/// The specialization of compute program `addr` (code hash `code_hash`)
/// for its registers, before its resources (`prepare` adds them).
fn cs_spec(addr: u64, rsrc2: u32, local: [u32; 3], wave64: bool, code_hash: u64) -> Spec {
    let user_sgprs = ((rsrc2 >> 1) & 0x1F).min(16) as u16;
    let (ps_inputs, resources) = spec_vecs();
    let mut spec = Spec {
        stage: Stage::Compute,
        user_sgprs,
        user_base: 0,
        code_base: addr,
        wave64,
        ps_input_addr: 0,
        ps_inputs,
        barycentric: false,
        bary_gs: false,
        mrt: [MrtSpec::default(); 8],
        cs_local_size: local,
        cs_rsrc2: rsrc2,
        resources,
        binding_base: 0,
        probe: probe_for(addr),
        dump: dump_for(addr),
        force_dispatcher: super::tune::block_chain() == 0,
        dispatcher_for_loops: super::tune::block_chain() == 2,
        dom_nest: super::tune::block_chain() == 3,
        reg_vars: super::tune::reg_vars(),
        lane_guards: true,
        mip_views: super::tune::mip_views(),
        lod_stats: matches!(super::tune::lod_stats(), 3 | 4),
        lod_lean: super::tune::lod_stats() == 4,
        early_tests: false,
        early_ok: false,
        per_lane: false,
        lane_votes: false,
        lockstep_off: false,
        sg_waves: false,
        sg_appends: false,
        known_active: true,
        hw_bounds: super::tune::hw_bounds_on(),
        base_instance_sgpr: None,
        sgpr_pc: super::tune::sgpr_pc().then_some(translate::sgpr_pc_offset(Stage::Compute)),
        addresses: super::tune::batch_cs() && batchable(addr),
        batched: super::tune::batch_cs() && batchable(addr),
        write_marks: marks_wanted(),
        marks_untouched: !super::tune::mark_touch(),
        touch_test: touch_test_wanted(),
        mark_twice: marks_wanted() && super::tune::tr_ballast() & 1 != 0,
        touch_twice: marks_wanted() && super::tune::tr_ballast() & 4 != 0,
        part_twice: marks_wanted() && super::tune::tr_ballast() & 8 != 0,
        dep_loads: super::tune::tr_ballast() & 2 != 0,
        marks_part: marks_part_wanted(),
        marks_held: marks_held_wanted(),
        sgpr_ubo: super::tune::sgpr_ubo(),
        gpu_loads: 0,
        gpu_res: 0,
        pt_linear: false,
        outlier: None,
    };
    apply_modes(&mut spec, code_hash);
    // LOD statistics in rotation (`tune lod_cs` >= 2): a dispatch measures
    // only when prepared within `with_lod_stats` (`lodstats::rotate`);
    // batched dispatches keep them (their textures are not all known to the
    // CPU).
    if spec.lod_stats && !spec.batched && cs_lod_rotation() && !LOD_MEASURE.with(|c| c.get()) {
        spec.lod_stats = false;
        spec.lod_lean = false;
    }
    // GPU-resolved per-item chains (`gpures`, `tune gpu_chains`; batched
    // programs, whose translations use device addresses).
    if spec.batched && matches!(super::tune::gpu_chains(), 1 | 2) && super::pagetable::address() != 0 && !NO_GPU_CHAINS.with(|c| c.get()) {
        if let Some(d) = super::gpures::decided(addr) {
            spec.gpu_loads = d.gpu_loads;
            spec.gpu_res = d.gpu_res;
            // Front end 2: its buffers and scalar loads from the mirror at guest offsets.
            spec.pt_linear = super::fe2::on();
        }
    }
    spec
}

/// Compute programs translated per lane (`Spec::per_lane`), by FNV-1a of
/// their words up to the first s_endpgm, each checked by reading its code:
/// its cross-lane operations only scalarize per-lane work (FINDINGS.md
/// 2026-10-13: their translations then take 128 registers instead of 168
/// and run 35-40% faster); and whether its EXEC branches are decided per
/// subgroup (`Spec::lane_votes`).
const PER_LANE_PROGRAMS: &[(u64, &str, bool)] = &[
    // Tiled lighting passes (~40k workgroups of one wave64): per-cluster
    // light masks OR-reduced across the wave (DPP, v_permlanex16,
    // v_readlane of lanes 31 and 63) and walked with s_ff1; light data read
    // with scalar loads, per-lane tests inside.
    (0xb65e_e39a_31c6_fa92, "tiled pass cs 0x10088ab600", false),
    (0xbf04_60d4_7979_f8d3, "tiled pass cs 0x10088b8200", false),
    // The froxel fog (the same light-mask walks, two waterfall loops, LDS
    // slots per lane): per lane in half the time. Its light-mask unions
    // enable all lanes and read the ones EXEC had masked off as zero; with
    // EXEC branches decided per invocation such lanes had left the region
    // and the reads found no value: a haze band along the right screen edge
    // (the volume is 214 froxels wide in 27 groups of 8; lanes past it read
    // no depth). Its EXEC branches are decided per 32-lane subgroup
    // (`Spec::lane_votes`; the tiled passes' would cost them ~20%).
    (0xcb83_739f_ea04_1817, "froxel fog cs 0x100892b200", true),
    // Two more passes of the tiled kind in the deferred lighting batch (cs
    // 0x1008a08e00, 0x1008a0f200 at InsideL Village and the swamp, lockstep
    // until 2026-11-12 night): two cluster masks each OR-reduced across the
    // wave (DPP rows, v_permlanex16, v_readlane of lanes 31 and 63) and
    // walked with s_ff1, no LDS; their two first-lane reads take uniform
    // values (an SGPR's reciprocal, a load at a fixed index). Per lane: the
    // swamp x1.011, GPU busy 12.7 -> 11.7 ms a frame (FINDINGS.md).
    (0x61c6_38df_4c69_edb5, "tiled pass cs 0x1008a08e00", false),
    (0xa91a_ab21_188b_3a2a, "tiled pass cs 0x1008a0f200", false),
];


/// Compute programs translated without lockstep mode (`Spec::lockstep_off`),
/// by FNV-1a of their words up to the first s_endpgm (as PER_LANE_PROGRAMS),
/// each checked by reading its code (its divergent cross-lane reads are
/// waterfall loops and OR-reduced light masks, its DPP within quads and
/// rows, no LDS) and by shots of one view against lockstep (FINDINGS.md
/// 2026-11-03 early morning).
const NO_LOCKSTEP_PROGRAMS: &[(u64, &str)] = &[
    // The GI probe update (from giupda; one 8x8 workgroup per probe tile):
    // screen-space differences by DPP (quad_perm, row_ror), a waterfall
    // loop of v_readfirstlane, the light-mask walk of the tiled passes.
    // 881 -> 408-448 us a frame at 1-1's vista, on a compute queue beside
    // the graphics work: frames level on the RTX 4090 (in-run A/B).
    (0x7761_ec89_6045_ec47, "GI probe update cs 0x1008a05b00"),
];

/// Compute programs translated as subgroup waves (`Spec::sg_waves`), by
/// FNV-1a of their words up to the first s_endpgm (as PER_LANE_PROGRAMS),
/// each checked by reading its code: a wave of 32 lanes gives what one of
/// 64 gives (its lane-0 elections, appends and reductions per wave, no lane
/// index as data, no LDS), and by the interpreter check (`emucheck`).
const SG_WAVE_PROGRAMS: &[(u64, &str)] = &[
    // The GI sweep (giubertree_sweep; one wave64 per group, 64 entries):
    // per-entry tests and loads, two OR reductions and an add reduction
    // (DPP rows, v_permlanex16 and v_readlane of lanes 31 and 63), an
    // append (a ballot tested by halves, v_bcnt, lane 0 elected by its
    // compare's bit 0 and its result read from lane 0, v_mbcnt ranks), lane
    // 0's atomic subtract of the overflow's sum. Its arrival spikes about
    // halve (wave-mask mode ~6.3 ms an arrival; FINDINGS.md 2026-11-04
    // morning, later).
    (0xb406_602d_7ad8_fe58, "GI sweep cs 0x10089f3f00"),
    // The GI reinsert (one wave64 per group of 64 entries): per-entry tests
    // and loads, two appends of that form (halves tested by scalar
    // compares, lane 0's atomic add or subtract of the count, its result
    // read from lane 0, v_mbcnt ranks), per-lane atomic ands.
    (0x7f89_1d6a_4856_4598, "GI reinsert cs 0x10089f4d00"),
];

/// A compute program's translation modes as this device takes them (`plain_modes`): per lane
/// (with its lane votes), without lockstep, as subgroup waves, with per-subgroup appends. Also for
/// the specs earlier sessions kept (`precompile::exact_work`): a preparation on a device whose modes
/// changed then makes the variants its dispatches look up (the review of 2026-11-23 night).
pub(crate) fn apply_modes(spec: &mut rdna::translate::Spec, code_hash: u64) {
    spec.per_lane = false;
    spec.lane_votes = false;
    if super::tune::per_lane() != 0 && per_lane_program(code_hash) && !plain_modes() {
        let votes = lane_votes_program(code_hash);
        spec.per_lane = super::tune::per_lane() == 1 || !votes;
        spec.lane_votes = spec.per_lane && votes;
    }
    spec.lockstep_off = no_lockstep_program(code_hash) && !plain_modes();
    spec.sg_waves = sg_wave_program(code_hash) && !plain_modes();
    spec.sg_appends = append_program(code_hash) && !plain_modes();
}

/// Whether every compute program runs in the mode the translator picks from its code alone, the
/// portable one: none translated per lane, without lockstep, as subgroup waves or with
/// per-subgroup appends (the lists above and `append_program`). Those modes rely on how a subgroup
/// treats lanes that are off (reads of them, votes) and on its lanes coming back together, which
/// Vulkan leaves to the driver; they were checked on NVIDIA's RTX 40 series only. On an RTX 5090
/// and an RTX 5080 players saw the tunnel at the game's start drawn black, the deferred lighting
/// and the froxel fog (both in these modes) the likely part (issue #4, 2026-11-23): on a design
/// they were not checked on (`vk::subgroup_modes_checked`: the RTX 50 series, AMD, Intel) the
/// portable mode is the default since then. `DS_PLAIN_MODES=1` / `0` forces it on or off (an
/// A/B: the stored specs are keyed with the modes, so their programs then compile at their first
/// dispatch).
fn plain_modes() -> bool {
    static FORCED: std::sync::OnceLock<Option<bool>> = std::sync::OnceLock::new();
    let forced = *FORCED.get_or_init(|| match std::env::var("DS_PLAIN_MODES").ok().as_deref() {
        Some("1") => Some(true),
        Some("0") => Some(false),
        _ => None,
    });
    forced.unwrap_or_else(|| !super::vk::subgroup_modes_checked())
}

/// Program hashes found in SG_WAVE_PROGRAMS.
static SG_WAVE_HASHES: [std::sync::atomic::AtomicU64; 8] = [const { std::sync::atomic::AtomicU64::new(0) }; 8];

fn sg_wave_program(code_hash: u64) -> bool {
    code_hash != 0 && SG_WAVE_HASHES.iter().any(|h| h.load(std::sync::atomic::Ordering::Relaxed) == code_hash)
}

/// Program hashes found in NO_LOCKSTEP_PROGRAMS.
static NO_LOCKSTEP_HASHES: [std::sync::atomic::AtomicU64; 8] = [const { std::sync::atomic::AtomicU64::new(0) }; 8];

fn no_lockstep_program(code_hash: u64) -> bool {
    code_hash != 0 && NO_LOCKSTEP_HASHES.iter().any(|h| h.load(std::sync::atomic::Ordering::Relaxed) == code_hash)
}

/// Program hashes (`hash_words`) found in PER_LANE_PROGRAMS (read without a
/// lock by every compute preparation).
static PER_LANE_HASHES: [std::sync::atomic::AtomicU64; 8] = [const { std::sync::atomic::AtomicU64::new(0) }; 8];

fn per_lane_program(code_hash: u64) -> bool {
    code_hash != 0 && PER_LANE_HASHES.iter().any(|h| h.load(std::sync::atomic::Ordering::Relaxed) == code_hash)
}

/// Program hashes of PER_LANE_PROGRAMS entries with subgroup-decided EXEC
/// branches.
static LANE_VOTE_HASHES: [std::sync::atomic::AtomicU64; 8] = [const { std::sync::atomic::AtomicU64::new(0) }; 8];

fn lane_votes_program(code_hash: u64) -> bool {
    code_hash != 0 && LANE_VOTE_HASHES.iter().any(|h| h.load(std::sync::atomic::Ordering::Relaxed) == code_hash)
}

/// FNV-1a of a program's bytes up to and including its first s_endpgm.
fn program_id(words: &[u32]) -> u64 {
    let n = words.iter().position(|&w| w == 0xBF81_0000).map_or(words.len(), |i| i + 1);
    let mut h = 0xcbf2_9ce4_8422_2325u64;
    for &w in &words[..n] {
        for b in w.to_le_bytes() {
            h ^= b as u64;
            h = h.wrapping_mul(0x0000_0100_0000_01b3);
        }
    }
    h
}

/// Hashes of programs with a wave-aggregated append (`translate::has_append`,
/// noted at decode): their specs ask for appends done per subgroup
/// (`Spec::sg_appends`; FINDINGS.md 2026-11-03 afternoon), which the
/// translator takes when all of the program's are of that form. Other
/// programs' specs and keys stay as they were.
static APPEND_HASHES: [std::sync::atomic::AtomicU64; 64] = [const { std::sync::atomic::AtomicU64::new(0) }; 64];

fn append_program(code_hash: u64) -> bool {
    use std::sync::atomic::Ordering::Relaxed;
    code_hash != 0 && sg_appends_wanted() && APPEND_HASHES.iter().take_while(|h| h.load(Relaxed) != 0).any(|h| h.load(Relaxed) == code_hash)
}

/// `DS_SG_APPENDS=0` (diagnosis, boot; a spec field, so translations are
/// keyed apart): wave-aggregated appends translated in wave-mask mode as
/// before 2026-11-03 (the mirror audit's lasting pages by store path).
fn sg_appends_wanted() -> bool {
    static OFF: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    !*OFF.get_or_init(|| std::env::var("DS_SG_APPENDS").ok().as_deref() == Some("0"))
}

fn note_append(addr: u64, insts: &[rdna::Inst], hash: u64) {
    use std::sync::atomic::Ordering::Relaxed;
    if hash == 0 || append_program(hash) || !rdna::translate::has_append(insts) {
        return;
    }
    for slot in &APPEND_HASHES {
        match slot.compare_exchange(0, hash, Relaxed, Relaxed) {
            Ok(_) => {
                crate::info!("shader {addr:#x}: a wave-aggregated append (done per subgroup where the translator can)");
                return;
            }
            Err(h) if h == hash => return,
            Err(_) => {}
        }
    }
}

/// A program known by its code alone (the preparation's shader files, the learned store's kept
/// code): noted in the program lists as the title's own loads note theirs (`program_checked`),
/// so the specs made for it (per lane, subgroup waves, without lockstep, appends) are the ones
/// its dispatches key. Until 2026-11-15 night the preparation's guesses for those programs (the
/// fog scatter per lane, the GI reinsert as subgroup waves: the largest, 1-5 s compiles) never
/// matched a dispatch.
pub(crate) fn note_program(code: &[u32], insts: &[rdna::Inst], hash: u64) {
    note_per_lane(0, code, hash);
    note_append(0, insts, hash);
}

/// Notes a decoded program found in PER_LANE_PROGRAMS, NO_LOCKSTEP_PROGRAMS
/// or SG_WAVE_PROGRAMS (by its hash).
fn note_per_lane(addr: u64, words: &[u32], hash: u64) {
    let id = program_id(words);
    if let Some(&(_, what)) = SG_WAVE_PROGRAMS.iter().find(|p| p.0 == id) {
        if !sg_wave_program(hash) {
            for slot in &SG_WAVE_HASHES {
                if slot.compare_exchange(0, hash, std::sync::atomic::Ordering::Relaxed, std::sync::atomic::Ordering::Relaxed).is_ok() {
                    crate::info!("shader {addr:#x}: translated as subgroup waves ({what})");
                    break;
                }
            }
        }
    }
    if let Some(&(_, what)) = NO_LOCKSTEP_PROGRAMS.iter().find(|p| p.0 == id) {
        if !no_lockstep_program(hash) {
            for slot in &NO_LOCKSTEP_HASHES {
                if slot.compare_exchange(0, hash, std::sync::atomic::Ordering::Relaxed, std::sync::atomic::Ordering::Relaxed).is_ok() {
                    crate::info!("shader {addr:#x}: translated without lockstep ({what})");
                    break;
                }
            }
        }
    }
    let Some(&(_, what, votes)) = PER_LANE_PROGRAMS.iter().find(|p| p.0 == id) else { return };
    if per_lane_program(hash) {
        return;
    }
    // (The vote list first: a preparation that finds the program per lane
    // finds its votes too.)
    if votes {
        for slot in &LANE_VOTE_HASHES {
            if slot.compare_exchange(0, hash, std::sync::atomic::Ordering::Relaxed, std::sync::atomic::Ordering::Relaxed).is_ok() {
                break;
            }
        }
    }
    for slot in &PER_LANE_HASHES {
        if slot.compare_exchange(0, hash, std::sync::atomic::Ordering::Relaxed, std::sync::atomic::Ordering::Relaxed).is_ok() {
            crate::info!("shader {addr:#x}: translated per lane ({what}{})", if votes { "; EXEC branches per subgroup" } else { "" });
            return;
        }
    }
}

/// User-data dword count of a graphics stage: RSRC2 USER_SGPR[5:1] and USER_SGPR_MSB[27].
fn user_count(rsrc2: u32) -> u16 {
    (((rsrc2 >> 1) & 0x1F) | (((rsrc2 >> 27) & 1) << 5)).min(32) as u16
}

/// Pixel-shader registers (SH dword indices) and context registers (byte addresses).
const PS_PGM_LO: u32 = 0x08;
const PS_RSRC2: u32 = 0x0B;
const PS_USER_DATA: u32 = 0x0C;
const SPI_PS_INPUT_CNTL_0: u32 = 0x28644;
const SPI_PS_INPUT_ADDR: u32 = 0x286D0;
const SPI_PS_IN_CONTROL: u32 = 0x286D8;
/// DB_SHADER_CONTROL (bit 12 DEPTH_BEFORE_SHADER: `Spec::early_tests`;
/// bits 0-2 depth and stencil exports, 6 KILL_ENABLE, 7-8 coverage
/// exports: `Spec::early_ok`).
const DB_SHADER_CONTROL: u32 = 0x2880C;
/// Depth and stencil state (`Spec::early_ok`): DB_DEPTH_CONTROL (bit 0
/// stencil test, 1 depth test, 2 depth writes), DB_STENCIL_CONTROL (the
/// stencil operations), DB_STENCILREFMASK and its back-face twin (bits
/// 16-23 the write mask), DB_ALPHA_TO_MASK (bit 0 enable).
const DB_DEPTH_CONTROL: u32 = 0x28800;
const DB_STENCIL_CONTROL: u32 = 0x2842C;
const DB_STENCILREFMASK: u32 = 0x28430;
const DB_STENCILREFMASK_BF: u32 = 0x28434;
const DB_ALPHA_TO_MASK: u32 = 0x28B70;
const SPI_SHADER_COL_FORMAT: u32 = 0x28714;
const CB_COLOR0_INFO: u32 = 0x28C70;

/// Whether the draw's state lets the depth and stencil tests run before a
/// pixel program that writes no memory (`Spec::early_ok`): it exports no
/// depth, stencil or coverage, and it cannot kill (KILL_ENABLE clear, no
/// alpha to coverage) or the draw writes neither depth nor stencil (the
/// tests as the draw sets them up: `exec::draw`).
fn early_ok(ctx: &dyn Fn(u32) -> u32) -> bool {
    let dsc = ctx(DB_SHADER_CONTROL);
    let dctl = ctx(DB_DEPTH_CONTROL);
    let z_write = dctl & 6 == 6;
    let s_write = dctl & 1 != 0 && (ctx(DB_STENCILREFMASK) | ctx(DB_STENCILREFMASK_BF)) & 0x00FF_0000 != 0 && ctx(DB_STENCIL_CONTROL) & 0xFF_FFFF != 0;
    let exports = dsc & (0x7 | 0x3 << 7) != 0;
    let may_kill = dsc & (1 << 6) != 0 || ctx(DB_ALPHA_TO_MASK) & 1 != 0;
    super::tune::early_auto() && !exports && (!may_kill || !(z_write || s_write))
}

/// Translations mark the dwords their stores write for front end 2's
/// write-back (and the mirror's); `tune write_marks 0` (timing only):
/// unmarked, so GPU writes never reach guest memory.
fn marks_wanted() -> bool {
    (super::resident::mirror_wanted() || super::fe2::wanted()) && super::tune::write_marks()
}
// (`tune mark_touch 0`, timing only: `Spec::marks_untouched`.)

/// Marks combined over the lanes that write the same map word through NVIDIA's
/// partitioned subgroups (`Spec::marks_part`) where the device has them;
/// `DS_MARKS_PART=0` (diagnosis, boot; a spec field, so translations are
/// keyed apart): the ballot form for every store (the mirror audit's
/// lasting pages by store path).
fn marks_part_wanted() -> bool {
    static OFF: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    marks_wanted() && super::vk::subgroup_partitioned() && !*OFF.get_or_init(|| std::env::var("DS_MARKS_PART").ok().as_deref() == Some("0"))
}

/// C5: stores into pages front end 4 holds set no marks (`Spec::marks_held`,
/// with C4; `DS_C5=0` off, a spec field, so translations are keyed apart).
fn marks_held_wanted() -> bool {
    marks_wanted() && super::fe2::c5_wanted()
}

/// Front end 2's stores test their pages' touched bits before setting them
/// and touch a page once a store (`Spec::touch_test`; in-run A/B: FINDINGS.md
/// 2026-11-03 morning).
fn touch_test_wanted() -> bool {
    marks_wanted() && super::fe2::wanted()
}

/// Resolves and translates the pixel shader bound on `cp`.
/// `DS_MRTZ_CENSUS=1` (diagnosis; `prepare_ps`).
fn mrtz_census() -> bool {
    static MRTZ_CENSUS: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *MRTZ_CENSUS.get_or_init(|| std::env::var("DS_MRTZ_CENSUS").ok().as_deref() == Some("1"))
}

pub fn prepare_ps<R: RegRead + ?Sized>(cp: &R) -> Result<Prepared, String> {
    if super::tune::r2() && !PROBE_ON.load(std::sync::atomic::Ordering::Relaxed) && !super::exec::profiling_on() && !mrtz_census() && GPU_DRAW_MASKS.with(|c| c.get()).is_none() {
        return prepare_ps_r2(cp);
    }
    prep_warm(|| prepare_ps(cp));
    prep_lap_begin(1);
    let sh = |i: u32| cp.sh(i);
    let ctx = |a: u32| cp.ctx(a);
    let addr = ((sh(PS_PGM_LO) as u64) | ((sh(PS_PGM_LO + 1) as u64 & 0xFF) << 32)) << 8;
    let rsrc2 = sh(PS_RSRC2);
    let user: [u32; 32] = std::array::from_fn(|i| sh(PS_USER_DATA + i as u32));
    prep_lap(0);
    let (insts, code_hash) = program_local(addr)?;
    prep_lap(1);
    // `DS_MRTZ_CENSUS=1` (diagnosis): pixel programs exporting stencil, a sample mask or alpha
    // through MRTZ, channels the translator leaves out, each logged once with the draw's state.
    if mrtz_census() {
        let mz = insts.iter().filter(|i| i.enc == rdna::Enc::Exp && i.exp_target() == 8).fold(0, |m, i| m | i.exp_en());
        static SEEN: Mutex<Option<std::collections::HashSet<u64>>> = Mutex::new(None);
        if mz & 0xE != 0 && SEEN.lock().get_or_insert_with(Default::default).insert(addr) {
            crate::info!(
                "mrtz census: ps {addr:#x} exports channels {mz:#x} (1 depth, 2 stencil, 4 sample mask, 8 alpha); DB_SHADER_CONTROL {:#x}, DB_ALPHA_TO_MASK {:#x}, DB_DEPTH_CONTROL {:#x}, SPI_SHADER_Z_FORMAT {:#x}, PA_SC_AA_CONFIG {:#x}, color formats {:#x}",
                ctx(DB_SHADER_CONTROL),
                ctx(DB_ALPHA_TO_MASK),
                ctx(DB_DEPTH_CONTROL),
                ctx(0x28710),
                ctx(0x28BE0),
                ctx(SPI_SHADER_COL_FORMAT)
            );
            let _ = super::capture::save_program(addr);
        }
    }
    let mut spec = ps_spec(addr, rsrc2, &ctx);
    // Profile (`DS_PROFILE_GPU=1`): when the draw's depth tests run.
    if super::exec::profiling_on() {
        let class = if spec.early_tests {
            0
        } else if translate::writes_memory(&insts) {
            3
        } else if spec.early_ok {
            1
        } else if spec.lod_stats {
            // After the shader with the LOD statistics' atomics: no early
            // rejection either (the driver runs it for every fragment).
            4
        } else {
            2
        };
        super::exec::note_ps_tests(addr, class);
    }
    prep_lap(2);
    let r = prepare(addr, insts, code_hash, &user, &mut spec, 0);
    spec_vecs_back(spec);
    r
}

/// The pixel stage's registers its specialization depends on (`ps_spec`), in
/// a fixed order (R2's template key): SPI_PS_IN_CONTROL, its inputs'
/// SPI_PS_INPUT_CNTL_n, SPI_SHADER_COL_FORMAT, the CB_COLORn_INFO number
/// types, SPI_PS_INPUT_ADDR and `early_ok`'s depth and stencil state.
const PS_KEY_MAX: usize = 52;

fn ps_key<R: RegRead + ?Sized>(cp: &R, k: &mut [u32; PS_KEY_MAX]) -> usize {
    let in_control = cp.ctx(SPI_PS_IN_CONTROL);
    k[0] = in_control;
    let ni = (in_control & 0x3F).min(32) as usize;
    for i in 0..ni {
        k[1 + i] = cp.ctx(SPI_PS_INPUT_CNTL_0 + 4 * i as u32);
    }
    let mut n = 1 + ni;
    k[n] = cp.ctx(SPI_SHADER_COL_FORMAT);
    n += 1;
    for m in 0..8u32 {
        k[n] = cp.ctx(CB_COLOR0_INFO + 0x3C * m) & (7 << 8);
        n += 1;
    }
    for a in [SPI_PS_INPUT_ADDR, DB_SHADER_CONTROL, DB_DEPTH_CONTROL, DB_STENCILREFMASK, DB_STENCILREFMASK_BF, DB_STENCIL_CONTROL, DB_ALPHA_TO_MASK] {
        k[n] = cp.ctx(a);
        n += 1;
    }
    n
}

/// R2: a pixel stage's program and specialization template (`ps_spec`, its
/// inputs included, no resources), kept per thread by the program's
/// address and RSRC2, the registers `ps_key` lists, the switches'
/// generation, the shader registrations and the batched and LOD-measuring
/// flags (probes, profiles, the MRTZ census and GPU-resolved draws take the
/// full path).
struct PsTmpl {
    id: u64,
    /// The specialization's content key for the lean lane (`tmpl_key`).
    tkey: u64,
    addr: u64,
    rsrc2: u32,
    n: u8,
    batched: bool,
    lod_measure: bool,
    key: [u32; PS_KEY_MAX],
    tune_gen: u64,
    regs: u64,
    insts: Option<Arc<Local<Vec<rdna::Inst>>>>,
    code_hash: u64,
    /// The spec's pixel inputs (the template spec's vectors stay empty).
    inputs: Vec<PsInput>,
    spec: Option<Box<Spec>>,
    /// `tmpl_tick` at its last use (`tmpl_way`).
    used: std::cell::Cell<u32>,
}

const PS_TMPL_SLOTS: usize = 2048;

thread_local! {
    static PS_TMPLS: std::cell::RefCell<Vec<PsTmpl>> = std::cell::RefCell::new(Vec::new());
}

fn prepare_ps_r2<R: RegRead + ?Sized>(cp: &R) -> Result<Prepared, String> {
    prep_warm(|| prepare_ps(cp));
    prep_lap_begin(1);
    let sh = |i: u32| cp.sh(i);
    let addr = ((sh(PS_PGM_LO) as u64) | ((sh(PS_PGM_LO + 1) as u64 & 0xFF) << 32)) << 8;
    let rsrc2 = sh(PS_RSRC2);
    let user: [u32; 32] = std::array::from_fn(|i| sh(PS_USER_DATA + i as u32));
    prep_lap(0);
    let mut key = [0u32; PS_KEY_MAX];
    let n = ps_key(cp, &mut key);
    let batched = batched_draw_now();
    let lod_measure = LOD_MEASURE.with(|c| c.get());
    let tune_gen = super::tune::generation();
    let regs = crate::hle::libs::agc::shader_registrations();
    let set = ((addr >> 8) ^ (addr >> 19)) as usize & (PS_TMPL_SLOTS / TMPL_WAYS - 1);
    let slot = PS_TMPLS.with(|t| {
        let t = t.borrow();
        tmpl_way(t.len(), |i| (t[i].addr, t[i].used.get()), addr, set)
    });
    // The lean lane first, from the template without cloning its specialization.
    if lean_now() {
        let t = PS_TMPLS.with(|t| {
            let t = t.borrow();
            let e = t.get(slot)?;
            let s = e.spec.as_ref()?;
            (e.addr == addr && e.rsrc2 == rsrc2 && e.n as usize == n && e.batched == batched && e.lod_measure == lod_measure && e.tune_gen == tune_gen && e.regs == regs && e.key[..n] == key[..n] && e.tkey != 0).then(|| {
                e.used.set(tmpl_tick());
                (e.insts.clone(), e.code_hash, e.tkey, s.user_sgprs, s.user_base)
            })
        });
        if let Some((Some(insts), code_hash, tkey, us, ub)) = t {
            if let Some(r) = lean::prepare(addr, &insts, code_hash, &user, Stage::Fragment, us, ub, tkey) {
                prep_lap_end();
                return r;
            }
        }
    }
    let hit = PS_TMPLS.with(|t| {
        let t = t.borrow();
        let Some(e) = t.get(slot) else {
            tmpl_census(1, 1);
            return None;
        };
        tmpl_census(1, if e.spec.is_none() { 1 } else if e.addr != addr { 2 } else if e.regs != regs { 3 } else if e.rsrc2 == rsrc2 && e.n as usize == n && e.batched == batched && e.lod_measure == lod_measure && e.tune_gen == tune_gen && e.key[..n] == key[..n] { 0 } else { 4 });
        if e.addr == addr && e.rsrc2 == rsrc2 && e.n as usize == n && e.batched == batched && e.lod_measure == lod_measure && e.tune_gen == tune_gen && e.regs == regs && e.key[..n] == key[..n] {
            // (The template's vectors are empty: its clone allocates nothing.)
            e.used.set(tmpl_tick());
            let mut spec = (**e.spec.as_ref()?).clone();
            let (mut ps_inputs, resources) = spec_vecs();
            ps_inputs.extend_from_slice(&e.inputs);
            spec.ps_inputs = ps_inputs;
            spec.resources = resources;
            Some((e.insts.clone()?, e.code_hash, spec, (e.id, e.tkey)))
        } else {
            None
        }
    });
    let (insts, code_hash, mut spec, (id, tkey)) = match hit {
        Some(h) => {
            prep_lap(1);
            if r2_check() {
                r2_check_spec("ps", addr, &h.2, ps_spec(addr, rsrc2, &|a: u32| cp.ctx(a)));
            }
            h
        }
        None => {
            let (insts, code_hash) = program_local(addr)?;
            prep_lap(1);
            let spec = ps_spec(addr, rsrc2, &|a: u32| cp.ctx(a));
            let id = next_tmpl_id();
            let registered = LOCAL_PROGRAMS.with(|c| c.borrow().contains_key(&addr));
            let tkey = if registered { tmpl_key(&spec) } else { 0 };
            if registered {
                PS_TMPLS.with(|t| {
                    let mut t = t.borrow_mut();
                    if t.is_empty() {
                        t.resize_with(PS_TMPL_SLOTS, || PsTmpl { id: 0, tkey: 0, addr: 0, rsrc2: 0, n: 0, batched: false, lod_measure: false, key: [0; PS_KEY_MAX], tune_gen: 0, regs: 0, insts: None, code_hash: 0, inputs: Vec::new(), spec: None, used: std::cell::Cell::new(0) });
                    }
                    let mut tmpl = spec.clone();
                    let inputs = std::mem::take(&mut tmpl.ps_inputs);
                    tmpl.resources = Vec::new();
                    t[slot] = PsTmpl { id, tkey, addr, rsrc2, n: n as u8, batched, lod_measure, key, tune_gen, regs, insts: Some(insts.clone()), code_hash, inputs, spec: Some(Box::new(tmpl)), used: std::cell::Cell::new(tmpl_tick()) };
                });
            }
            (insts, code_hash, spec, (id, tkey))
        }
    };
    prep_lap(2);
    let r = prepare_stage(addr, insts, code_hash, &user, &mut spec, id, tkey);
    spec_vecs_back(spec);
    r
}

/// The specialization of pixel program `addr` for the draw's registers
/// (`ctx` by byte address), before its resources (`prepare` adds them).
fn ps_spec(addr: u64, rsrc2: u32, ctx: &dyn Fn(u32) -> u32) -> Spec {
    let user_sgprs = user_count(rsrc2);
    let in_control = ctx(SPI_PS_IN_CONTROL);
    let (mut ps_inputs, resources) = spec_vecs();
    for n in 0..(in_control & 0x3F).min(32) {
        let c = ctx(SPI_PS_INPUT_CNTL_0 + 4 * n);
        let off = c & 0x3F;
        ps_inputs.push(PsInput { param: (off & 0x1F) as u8, flat: (c >> 10) & 1 != 0, default: (off & 0x20 != 0).then_some(((c >> 8) & 3) as u8) });
    }
    let col = ctx(SPI_SHADER_COL_FORMAT);
    let mut mrt = [MrtSpec::default(); 8];
    for (m, slot) in mrt.iter_mut().enumerate() {
        let info = ctx(CB_COLOR0_INFO + 0x3C * m as u32);
        let class = match (info >> 8) & 7 {
            4 => 2,
            5 => 1,
            _ => 0,
        };
        *slot = MrtSpec { export_format: ((col >> (4 * m)) & 0xF) as u8, class };
    }
    // A batched draw's pixel stage takes its SGPR data offset at the
    // `BATCH_VARYING` location: not batched when one of its own inputs is
    // there (the draw path then refuses the pair, "not translated batched").
    let batched = batched_draw_now() && !ps_inputs.iter().any(|i| i.default.is_none() && i.param as u32 == translate::BATCH_VARYING);
    let mut spec = Spec {
        stage: Stage::Fragment,
        user_sgprs,
        user_base: 0,
        code_base: addr,
        wave64: (in_control >> 15) & 1 == 0,
        ps_input_addr: ctx(SPI_PS_INPUT_ADDR),
        barycentric: super::vk::ctx().barycentric,
        bary_gs: super::vk::ctx().bary_gs,
        ps_inputs,
        mrt,
        cs_local_size: [1, 1, 1],
        cs_rsrc2: 0,
        resources,
        binding_base: PS_BINDING_BASE,
        probe: probe_for(addr),
        dump: None,
        force_dispatcher: super::tune::block_chain() == 0,
        dispatcher_for_loops: super::tune::block_chain() == 2,
        dom_nest: super::tune::block_chain() == 3,
        reg_vars: super::tune::reg_vars(),
        lane_guards: true,
        mip_views: super::tune::mip_views(),
        lod_stats: matches!(super::tune::lod_stats(), 3 | 4),
        lod_lean: super::tune::lod_stats() == 4,
        early_tests: ctx(DB_SHADER_CONTROL) & (1 << 12) != 0,
        early_ok: early_ok(ctx),
        per_lane: false,
        lane_votes: false,
        lockstep_off: false,
        sg_waves: false,
        sg_appends: false,
        known_active: true,
        hw_bounds: super::tune::hw_bounds_on(),
        base_instance_sgpr: None,
        sgpr_pc: super::tune::sgpr_pc().then_some(translate::sgpr_pc_offset(Stage::Fragment)),
        addresses: batched,
        batched,
        write_marks: marks_wanted(),
        marks_untouched: !super::tune::mark_touch(),
        touch_test: touch_test_wanted(),
        mark_twice: marks_wanted() && super::tune::tr_ballast() & 1 != 0,
        touch_twice: marks_wanted() && super::tune::tr_ballast() & 4 != 0,
        part_twice: marks_wanted() && super::tune::tr_ballast() & 8 != 0,
        dep_loads: super::tune::tr_ballast() & 2 != 0,
        marks_part: marks_part_wanted(),
        marks_held: marks_held_wanted(),
        sgpr_ubo: super::tune::sgpr_ubo(),
        gpu_loads: 0,
        gpu_res: 0,
        pt_linear: false,
        outlier: None,
    };
    if spec.batched {
        if let Some((l, r)) = gpu_draw_masks_now(1) {
            spec.gpu_loads = l;
            spec.gpu_res = r;
            spec.pt_linear = super::fe2::on();
        }
    }
    // LOD statistics where the depth tests stay after the shader (`tune
    // lod_late`): in rotation only for a draw measuring (`with_lod_stats`)
    // or batched; none at 0 (diagnosis).
    if spec.lod_stats && !spec.early_tests && !spec.early_ok && !late_stats(spec.batched || LOD_MEASURE.with(|c| c.get())) {
        spec.lod_stats = false;
        spec.lod_lean = false;
    }
    spec
}

/// `DS_PROBE=<shader address>:<pc>:<vgpr>` (hex address and pc, decimal
/// VGPR) or the `probe` control command: that pixel shader exports the
/// VGPR's value after the instruction at pc as its color (float targets) or
/// its raw bits (integer targets), or that compute shader stores it as the
/// data of its image stores (debugging). An optional fourth field `:<n>`
/// starts probing in the n-th frame (0-based) that uses the shader, e.g. to
/// keep a reference frame intact.
pub(crate) fn probe_for(addr: u64) -> Option<(u32, u32)> {
    static USES: std::sync::Mutex<(u64, u32)> = std::sync::Mutex::new((u64::MAX, 0));
    // Every stage preparation asks: the shared lock only while a probe is
    // set (it was ~1% of the recording threads' busy samples).
    PROBE_FROM_ENV.call_once(|| {
        let mut g = PROBE.lock();
        let p = *g.get_or_insert_with(|| std::env::var("DS_PROBE").ok().and_then(|v| parse_probe(&v)));
        PROBE_ON.store(p.is_some(), std::sync::atomic::Ordering::Relaxed);
    });
    if !PROBE_ON.load(std::sync::atomic::Ordering::Relaxed) {
        return None;
    }
    let p = *PROBE.lock().get_or_insert_with(|| std::env::var("DS_PROBE").ok().and_then(|v| parse_probe(&v)));
    let (a, pc, vg, from) = p?;
    if a != addr {
        return None;
    }
    let mut u = USES.lock().unwrap();
    let flip = super::capture::flips();
    if u.0 != flip {
        *u = (flip, u.1 + 1);
    }
    (u.1 > from).then_some((pc, vg))
}

/// Probe setting (see `probe_for`): None until read from `DS_PROBE`.
static PROBE: parking_lot::Mutex<Option<Option<(u64, u32, u32, u32)>>> = parking_lot::Mutex::new(None);
/// A probe is set (PROBE holds one); `DS_PROBE` read once.
static PROBE_ON: std::sync::atomic::AtomicBool = std::sync::atomic::AtomicBool::new(false);
static PROBE_FROM_ENV: std::sync::Once = std::sync::Once::new();

/// `<shader>:<pc>:<register>[:<n>]`; the register is `v<k>` (or `<k>`) or
/// `s<k>`, with a suffix `u` / `i` to show an unsigned / signed integer as
/// a float.
fn parse_probe(v: &str) -> Option<(u64, u32, u32, u32)> {
    use rdna::translate::{PROBE_SGPR, PROBE_SINT, PROBE_UINT};
    let mut it = v.split(':');
    let a = u64::from_str_radix(it.next()?.trim().trim_start_matches("0x"), 16).ok()?;
    let pc = u32::from_str_radix(it.next()?.trim().trim_start_matches("0x"), 16).ok()?;
    let r = it.next()?.trim();
    let (r, conv) = match r.strip_suffix('u') {
        Some(x) => (x, PROBE_UINT),
        None => match r.strip_suffix('i') {
            Some(x) => (x, PROBE_SINT),
            None => (r, 0),
        },
    };
    let (r, file) = match r.strip_prefix('s') {
        Some(x) => (x, PROBE_SGPR),
        None => (r.strip_prefix('v').unwrap_or(r), 0),
    };
    let n: u32 = r.parse().ok()?;
    if n >= if file != 0 { 128 } else { 256 } {
        return None;
    }
    let vg = n | file | conv;
    let from = it.next().and_then(|x| x.trim().parse().ok()).unwrap_or(0);
    Some((a, pc, vg, from))
}

/// `probe <shader>:<pc>:<vgpr>[:<n>]` / `probe off` (control command).
pub fn set_probe(v: &str) -> Result<(), String> {
    let p = if v.trim() == "off" { None } else { Some(parse_probe(v).ok_or("usage: probe <hex shader>:<hex pc>:<v<k>|s<k>>[u|i][:<frames>] | off")?) };
    // The environment's setting is replaced (read first so that it cannot
    // come back).
    PROBE_FROM_ENV.call_once(|| {});
    *PROBE.lock() = Some(p);
    PROBE_ON.store(p.is_some(), std::sync::atomic::Ordering::Relaxed);
    Ok(())
}

/// Diagnosis (`DS_VS_OUTLIER=<threshold>`, 2026-11-19): every vertex program counts the vertices
/// it exports with |x|, |y| or |w| of the clip position above the threshold, in its census slot
/// (`rdna::translate::OUTLIER_SLOTS`; `exec::outlier_note` writes the counts that moved each
/// frame to outliers.txt in the run directory). The f32 bits.
pub fn vs_outlier() -> Option<u32> {
    static T: std::sync::OnceLock<Option<u32>> = std::sync::OnceLock::new();
    *T.get_or_init(|| std::env::var("DS_VS_OUTLIER").ok().and_then(|v| v.trim().parse::<f32>().ok()).filter(|t| *t > 0.0).map(f32::to_bits))
}

/// `DS_VS_OUTLIER_LOW=<hex code hash>:<threshold>[;...]`: those programs' census thresholds
/// (2D programs whose positions stay within a few units).
fn outlier_low(h: u64) -> Option<u32> {
    static L: std::sync::OnceLock<Vec<(u64, u32)>> = std::sync::OnceLock::new();
    L.get_or_init(|| {
        std::env::var("DS_VS_OUTLIER_LOW").map_or(Vec::new(), |v| {
            v.split(';')
                .filter_map(|e| {
                    let (a, t) = e.split_once(':')?;
                    Some((u64::from_str_radix(a.trim().trim_start_matches("0x"), 16).ok()?, t.trim().parse::<f32>().ok()?.to_bits()))
                })
                .collect()
        })
    })
    .iter()
    .find(|e| e.0 == h)
    .map(|e| e.1)
}

/// The census's code hash `h`, its program's address logged at its first use (outliers.txt
/// names programs by hash).
fn outlier_named(h: u64, addr: u64) -> u64 {
    static SEEN: parking_lot::Mutex<Option<std::collections::HashSet<u64>>> = parking_lot::Mutex::new(None);
    if SEEN.lock().get_or_insert_with(std::collections::HashSet::new).insert(h) {
        crate::info!("vertex outlier census: program {h:#018x} at {addr:#x}");
    }
    h
}

/// `DS_DUMP=<shader>:<pc>:<x>,<y>,<z>` (hex shader address and pc, decimal
/// workgroup id): that compute shader's workgroup dumps its registers after
/// the instruction at pc (`DS_EMU_CHECK` compares them with the interpreter).
pub(crate) fn dump_for(addr: u64) -> Option<(u32, [u32; 3])> {
    static D: std::sync::OnceLock<Option<(u64, u32, [u32; 3])>> = std::sync::OnceLock::new();
    let d = D.get_or_init(|| {
        let v = std::env::var("DS_DUMP").ok()?;
        let mut it = v.split(':');
        let a = u64::from_str_radix(it.next()?.trim_start_matches("0x"), 16).ok()?;
        let pc = u32::from_str_radix(it.next()?.trim_start_matches("0x"), 16).ok()?;
        let g: Vec<u32> = it.next()?.split(',').filter_map(|x| x.parse().ok()).collect();
        Some((a, pc, [g.first().copied().unwrap_or(0), g.get(1).copied().unwrap_or(0), g.get(2).copied().unwrap_or(0)]))
    });
    d.and_then(|(a, pc, g)| (a == addr).then_some((pc, g)))
}

/// Binding numbers of the pixel stage start here (vertex stage from 0).
pub const PS_BINDING_BASE: u32 = 32;

/// NGG primitive shader: program in the ES registers, resources and user
/// data in the GS ones (gfx10.3 register database).
const ES_PGM_LO: u32 = 0xC8;
const GS_RSRC2: u32 = 0x8B;
const GS_USER_DATA: u32 = 0x8C;
const VGT_SHADER_STAGES_EN: u32 = 0x28B54;

/// Resolves and translates the NGG (vertex) shader bound on `cp`.
pub fn prepare_ngg<R: RegRead + ?Sized>(cp: &R) -> Result<Prepared, String> {
    if super::tune::r2() && GPU_DRAW_MASKS.with(|c| c.get()).is_none() {
        let addr = ngg_addr_of(cp);
        if super::capture::vs_dump_for(addr).is_none() {
            return prepare_ngg_r2(cp);
        }
    }
    prep_warm(|| prepare_ngg(cp));
    prep_lap_begin(0);
    let sh = |i: u32| cp.sh(i);
    let stages = cp.ctx(VGT_SHADER_STAGES_EN);
    let addr = ((sh(ES_PGM_LO) as u64) | ((sh(ES_PGM_LO + 1) as u64 & 0xFF) << 32)) << 8;
    let rsrc2 = sh(GS_RSRC2);
    let user: [u32; 32] = std::array::from_fn(|i| sh(GS_USER_DATA + i as u32));
    prep_lap(0);
    let (insts, code_hash) = program_local(addr)?;
    prep_lap(1);
    let mut spec = ngg_spec(addr, rsrc2, stages, &insts);
    prep_lap(2);
    let r = prepare(addr, insts, code_hash, &user, &mut spec, 0);
    spec_vecs_back(spec);
    r
}

/// R2: a vertex (NGG) stage's program and specialization template
/// (`ngg_spec` without resources), kept per thread by the program's address,
/// RSRC2, VGT_SHADER_STAGES_EN's wave size, the start-instance slot, the
/// batched flag, the switches' generation and the shader registrations
/// (GPU-resolved draws and vertex dumps take the full path).
struct NggTmpl {
    id: u64,
    /// The specialization's content key for the lean lane (`tmpl_key`).
    tkey: u64,
    addr: u64,
    rsrc2: u32,
    wave: u32,
    base_instance: Option<u16>,
    batched: bool,
    tune_gen: u64,
    regs: u64,
    insts: Option<Arc<Local<Vec<rdna::Inst>>>>,
    code_hash: u64,
    spec: Option<Box<Spec>>,
    /// `tmpl_tick` at its last use (`tmpl_way`).
    used: std::cell::Cell<u32>,
}

const NGG_TMPL_SLOTS: usize = 2048;

thread_local! {
    static NGG_TMPLS: std::cell::RefCell<Vec<NggTmpl>> = std::cell::RefCell::new(Vec::new());
}

fn prepare_ngg_r2<R: RegRead + ?Sized>(cp: &R) -> Result<Prepared, String> {
    prep_warm(|| prepare_ngg(cp));
    prep_lap_begin(0);
    let sh = |i: u32| cp.sh(i);
    let stages = cp.ctx(VGT_SHADER_STAGES_EN);
    let addr = ((sh(ES_PGM_LO) as u64) | ((sh(ES_PGM_LO + 1) as u64 & 0xFF) << 32)) << 8;
    let rsrc2 = sh(GS_RSRC2);
    let user: [u32; 32] = std::array::from_fn(|i| sh(GS_USER_DATA + i as u32));
    prep_lap(0);
    let wave = (stages >> 22) & 1;
    let base_instance = BASE_INSTANCE.with(|c| c.get());
    let batched = batched_draw_now();
    let tune_gen = super::tune::generation();
    let regs = crate::hle::libs::agc::shader_registrations();
    let set = ((addr >> 8) ^ (addr >> 19)) as usize & (NGG_TMPL_SLOTS / TMPL_WAYS - 1);
    let slot = NGG_TMPLS.with(|t| {
        let t = t.borrow();
        tmpl_way(t.len(), |i| (t[i].addr, t[i].used.get()), addr, set)
    });
    // The lean lane first, from the template without cloning its specialization.
    if lean_now() {
        let t = NGG_TMPLS.with(|t| {
            let t = t.borrow();
            let e = t.get(slot)?;
            let s = e.spec.as_ref()?;
            (e.addr == addr && e.rsrc2 == rsrc2 && e.wave == wave && e.base_instance == base_instance && e.batched == batched && e.tune_gen == tune_gen && e.regs == regs && e.tkey != 0).then(|| {
                e.used.set(tmpl_tick());
                (e.insts.clone(), e.code_hash, e.tkey, s.user_sgprs, s.user_base)
            })
        });
        if let Some((Some(insts), code_hash, tkey, us, ub)) = t {
            if let Some(r) = lean::prepare(addr, &insts, code_hash, &user, Stage::Vertex, us, ub, tkey) {
                prep_lap_end();
                return r;
            }
        }
    }
    let hit = NGG_TMPLS.with(|t| {
        let t = t.borrow();
        let Some(e) = t.get(slot) else {
            tmpl_census(0, 1);
            return None;
        };
        tmpl_census(0, if e.spec.is_none() { 1 } else if e.addr != addr { 2 } else if e.regs != regs { 3 } else if e.rsrc2 == rsrc2 && e.wave == wave && e.base_instance == base_instance && e.batched == batched && e.tune_gen == tune_gen { 0 } else { 4 });
        if e.addr == addr && e.rsrc2 == rsrc2 && e.wave == wave && e.base_instance == base_instance && e.batched == batched && e.tune_gen == tune_gen && e.regs == regs {
            e.used.set(tmpl_tick());
            let mut spec = (**e.spec.as_ref()?).clone();
            let (ps_inputs, resources) = spec_vecs();
            spec.ps_inputs = ps_inputs;
            spec.resources = resources;
            Some((e.insts.clone()?, e.code_hash, spec, (e.id, e.tkey)))
        } else {
            None
        }
    });
    let (insts, code_hash, mut spec, (id, tkey)) = match hit {
        Some(h) => {
            prep_lap(1);
            if r2_check() {
                r2_check_spec("vs", addr, &h.2, ngg_spec(addr, rsrc2, stages, &h.0));
            }
            h
        }
        None => {
            let (insts, code_hash) = program_local(addr)?;
            prep_lap(1);
            let spec = ngg_spec(addr, rsrc2, stages, &insts);
            let id = next_tmpl_id();
            let registered = LOCAL_PROGRAMS.with(|c| c.borrow().contains_key(&addr));
            let tkey = if registered { tmpl_key(&spec) } else { 0 };
            if registered {
                NGG_TMPLS.with(|t| {
                    let mut t = t.borrow_mut();
                    if t.is_empty() {
                        t.resize_with(NGG_TMPL_SLOTS, || NggTmpl { id: 0, tkey: 0, addr: 0, rsrc2: 0, wave: 0, base_instance: None, batched: false, tune_gen: 0, regs: 0, insts: None, code_hash: 0, spec: None, used: std::cell::Cell::new(0) });
                    }
                    let mut tmpl = spec.clone();
                    tmpl.ps_inputs = Vec::new();
                    tmpl.resources = Vec::new();
                    t[slot] = NggTmpl { id, tkey, addr, rsrc2, wave, base_instance, batched, tune_gen, regs, insts: Some(insts.clone()), code_hash, spec: Some(Box::new(tmpl)), used: std::cell::Cell::new(tmpl_tick()) };
                });
            }
            (insts, code_hash, spec, (id, tkey))
        }
    };
    prep_lap(2);
    let r = prepare_stage(addr, insts, code_hash, &user, &mut spec, id, tkey);
    spec_vecs_back(spec);
    r
}

/// The specialization of vertex (NGG) program `addr` for the draw's
/// registers, before its resources (`prepare` adds them).
fn ngg_spec(addr: u64, rsrc2: u32, stages: u32, insts: &[rdna::Inst]) -> Spec {
    let user_sgprs = user_count(rsrc2);
    let (ps_inputs, resources) = spec_vecs();
    let mut spec = Spec {
        stage: Stage::Vertex,
        user_sgprs,
        user_base: 8,
        code_base: addr,
        wave64: (stages >> 22) & 1 == 0,
        ps_input_addr: 0,
        ps_inputs,
        barycentric: false,
        bary_gs: false,
        mrt: [MrtSpec::default(); 8],
        cs_local_size: [1, 1, 1],
        cs_rsrc2: 0,
        resources,
        binding_base: 0,
        probe: None,
        dump: super::capture::vs_dump_for(addr).and_then(|(first, inst, count)| {
            let at = super::capture::vs_dump_pc().filter(|pc| insts.iter().any(|i| i.pc == *pc));
            at.or_else(|| pos_export(insts).map(|i| i.pc)).map(|pc| (pc, [first, inst, count]))
        }),
        force_dispatcher: super::tune::block_chain() == 0,
        dispatcher_for_loops: super::tune::block_chain() == 2,
        dom_nest: super::tune::block_chain() == 3,
        reg_vars: super::tune::reg_vars(),
        lane_guards: true,
        mip_views: super::tune::mip_views(),
        lod_stats: matches!(super::tune::lod_stats(), 3 | 4),
        lod_lean: super::tune::lod_stats() == 4,
        early_tests: false,
        early_ok: false,
        per_lane: false,
        lane_votes: false,
        lockstep_off: false,
        sg_waves: false,
        sg_appends: false,
        known_active: true,
        hw_bounds: super::tune::hw_bounds_on(),
        base_instance_sgpr: BASE_INSTANCE.with(|c| c.get()).map(|k| 8 + k),
        sgpr_pc: super::tune::sgpr_pc().then_some(translate::sgpr_pc_offset(Stage::Vertex)),
        addresses: batched_draw_now(),
        batched: batched_draw_now(),
        write_marks: marks_wanted(),
        marks_untouched: !super::tune::mark_touch(),
        touch_test: touch_test_wanted(),
        mark_twice: marks_wanted() && super::tune::tr_ballast() & 1 != 0,
        touch_twice: marks_wanted() && super::tune::tr_ballast() & 4 != 0,
        part_twice: marks_wanted() && super::tune::tr_ballast() & 8 != 0,
        dep_loads: super::tune::tr_ballast() & 2 != 0,
        marks_part: marks_part_wanted(),
        marks_held: marks_held_wanted(),
        sgpr_ubo: super::tune::sgpr_ubo(),
        gpu_loads: 0,
        gpu_res: 0,
        pt_linear: false,
        outlier: vs_outlier().and_then(|t| program_hash(addr).ok().map(|h| (outlier_low(h).unwrap_or(t), outlier_named(h, addr)))),
    };
    if spec.batched {
        if let Some((l, r)) = gpu_draw_masks_now(0) {
            spec.gpu_loads = l;
            spec.gpu_res = r;
            spec.pt_linear = super::fe2::on();
        }
    }
    spec
}

/// Program address of the pixel shader in a register view.
pub fn ps_addr_of<R: RegRead + ?Sized>(r: &R) -> u64 {
    ((r.sh(PS_PGM_LO) as u64) | ((r.sh(PS_PGM_LO + 1) as u64 & 0xFF) << 32)) << 8
}

/// Program address of the NGG (vertex) shader in a register view.
pub fn ngg_addr_of<R: RegRead + ?Sized>(r: &R) -> u64 {
    ((r.sh(ES_PGM_LO) as u64) | ((r.sh(ES_PGM_LO + 1) as u64 & 0xFF) << 32)) << 8
}

/// Program address of the pixel shader bound on `cp`.
pub fn ps_addr(cp: &Cp) -> u64 {
    let sh = |i: u32| cp.reg(Space::Sh, i);
    ((sh(PS_PGM_LO) as u64) | ((sh(PS_PGM_LO + 1) as u64 & 0xFF) << 32)) << 8
}

/// Program address of the NGG (vertex) shader bound on `cp`.
pub fn ngg_addr(cp: &Cp) -> u64 {
    let sh = |i: u32| cp.reg(Space::Sh, i);
    ((sh(ES_PGM_LO) as u64) | ((sh(ES_PGM_LO + 1) as u64 & 0xFF) << 32)) << 8
}

/// The first position export (`exp pos0`) of a vertex program.
pub fn pos_export(insts: &[rdna::Inst]) -> Option<&rdna::Inst> {
    insts.iter().find(|i| i.enc == rdna::Enc::Exp && i.exp_target() == 12)
}

/// One line per resolved resource (frame captures).
/// Where a prepared stage's CPU-side reads come from (wait traces): each
/// scalar load and each descriptor read from memory, with its guest range.
/// The V# of resource `r` of `plan` for user data `user`, read from guest
/// memory now (GPU-resolved chains, `gpures`: the range such a resource
/// reaches, imported before the shader reads it). None: unreadable.
pub fn resource_vsharp(code_base: u64, plan: &Plan, r: usize, user: &[u32]) -> Option<Vsharp> {
    let res = plan.resources.get(r)?;
    let mut rd = Reader::new(code_base);
    let mut w = [0u32; 4];
    for (k, p) in res.dwords.iter().take(4).enumerate() {
        w[k] = rd.eval(p, user)?;
    }
    Some(Vsharp::decode(&w))
}

/// The guest range scalar load `k` of `plan` reads for user data `user`,
/// its pointer chain read from guest memory now (GPU-resolved loads,
/// `gpures`: imported before the shader reads it). None: unreadable.
pub fn load_range(code_base: u64, plan: &Plan, k: usize, user: &[u32]) -> Option<(u64, u64)> {
    let l = plan.loads.get(k)?;
    let mut rd = Reader::new(code_base);
    let base = rd.ptr(&l.ptr, user)?;
    Some((base.wrapping_add(l.offset as u64), 4 * l.dwords as u64))
}

/// Front end 2's GPU-resolved chains (stage F3, `exec::fe2_gpu_ranges`):
/// into `loads` the guest ranges the shader-resolved loads of `spec`
/// (`gpu_loads`) read, into `bufs` those its shader-resolved buffers
/// (`gpu_res`) reach (u64::MAX bytes: to the end of the mapping), for user
/// data `user`, their pointer chains and V#s read from guest memory now
/// (forwarded words as `Reader` reads them); into `reads` the words read for
/// that. Err(0): a chain is unreadable; Err(1): a formatted buffer's V# has
/// another format or swizzle than the translation's (its loads decode with
/// that one).
pub fn gpu_chain_ranges(code_base: u64, plan: &Plan, spec: &Spec, user: &[u32], loads: &mut Vec<(u64, u64)>, bufs: &mut Vec<(u64, u64)>, reads: &mut Vec<(u64, u64)>) -> Result<(), u8> {
    let mut rd = Reader::new(code_base);
    for (k, l) in plan.loads.iter().enumerate().take(64) {
        if spec.gpu_loads & (1u64 << k) == 0 {
            continue;
        }
        let base = rd.ptr(&l.ptr, user).ok_or(0u8)?;
        loads.push((base.wrapping_add(l.offset as u64), 4 * l.dwords as u64));
    }
    for (r, res) in plan.resources.iter().enumerate().take(64) {
        if spec.gpu_res & (1u64 << r) == 0 {
            continue;
        }
        let mut w = [0u32; 4];
        for (k, p) in res.dwords.iter().take(4).enumerate() {
            w[k] = rd.eval(p, user).ok_or(0u8)?;
        }
        let v = Vsharp::decode(&w);
        if res.formatted {
            if let Some(translate::ResSpec::Buffer { format, dst_sel }) = spec.resources.get(r) {
                if v.format != *format || v.dst_sel != *dst_sel {
                    return Err(1);
                }
            }
        }
        // Without range checks (OOB_SELECT 2) the shader reaches past the
        // records: to the end of the mapping, as the CPU path binds it
        // (u64::MAX here; `exec::fe2_gpu_ranges` clips it).
        let n = v.byte_size().unwrap_or(u64::MAX);
        if n > 0 && v.base != 0 {
            bufs.push((v.base, n));
        }
    }
    reads.extend_from_slice(&rd.reads);
    Ok(())
}

pub fn read_sources(p: &Prepared) -> Vec<(String, u64, u64)> {
    let mut rd = Reader::new(p.addr);
    let mut v = Vec::new();
    for (k, l) in p.tr.plan.loads.iter().enumerate() {
        if let Some(b) = rd.ptr(&l.ptr, &p.user) {
            v.push((format!("load {k} {:?}+{:#x} x{}", l.ptr, l.offset, l.dwords), b + l.offset as u64, 4 * l.dwords as u64));
        }
    }
    for (i, a) in p.res_addrs.iter().enumerate() {
        if let Some(a) = a {
            let res = &p.tr.plan.resources[i];
            v.push((format!("resource {i} {:?} {:?}", res.kind, res.dwords.first()), *a, 4 * res.dwords.len() as u64));
        }
    }
    v
}

pub fn describe_resources(p: &Prepared) -> Vec<String> {
    p.resources
        .iter()
        .enumerate()
        .map(|(i, r)| {
            let used = &p.tr.plan.resources[i];
            let from = match p.res_addrs.get(i).copied().flatten() {
                Some(a) => format!(" @{a:#x}"),
                None => String::new(),
            };
            let rw = format!("{}{}{from}", if used.read { "r" } else { "" }, if used.written { "w" } else { "" });
            match r {
                Resolved::Buffer(v) => format!("{i}: V# {:#x} stride {} records {} format {} dst_sel {:#x} oob {} {rw}", v.base, v.stride, v.num_records, v.format, v.dst_sel, v.oob_select),
                Resolved::Global(a, n) => format!("{i}: global {a:#x}{} {rw}", n.map(|n| format!("+{n:#x}")).unwrap_or_default()),
                Resolved::ImageTable(e) => {
                    let list: Vec<String> = e.iter().enumerate().filter_map(|(k, t)| t.map(|t| format!("{k}:{:#x} {}x{}x{} f{}", t.base, t.width, t.height, t.depth, t.format))).collect();
                    format!("{i}: image table of {} entries {rw} [{}]", e.len(), list.join(", "))
                }
                Resolved::Image(t) => format!(
                    "{i}: T# {:#x} format {} {}x{}x{} type {} sw_mode {} levels {}..{} base_array {} dst_sel {:#x} {rw} [{}]",
                    t.base,
                    t.format,
                    t.width,
                    t.height,
                    t.depth,
                    t.ty,
                    t.sw_mode,
                    t.base_level,
                    t.last_level,
                    t.base_array,
                    t.dst_sel,
                    t.raw.iter().map(|w| format!("{w:08x}")).collect::<Vec<_>>().join(" ")
                ),
                Resolved::Sampler(w) => format!("{i}: S# {:08x} {:08x} {:08x} {:08x} {rw}", w[0], w[1], w[2], w[3]),
            }
        })
        .collect()
}

/// One line per resolved scalar load: pointer provenance, offset and the
/// loaded dwords (hex, and as floats) (frame captures).
pub fn describe_loads(p: &Prepared) -> Vec<String> {
    let offs = translate::load_offsets(&p.tr.plan);
    p.tr
        .plan
        .loads
        .iter()
        .zip(offs)
        .map(|(l, o)| {
            let vals: Vec<String> = (0..l.dwords as usize)
                .map(|k| {
                    let w = p.sgpr_data.get(o as usize + k).copied().unwrap_or(0);
                    let f = f32::from_bits(w);
                    if f.is_finite() && f != 0.0 && (f.abs() < 1e6 && f.abs() > 1e-6) { format!("{w:08x}({f})") } else { format!("{w:08x}") }
                })
                .collect();
            format!("load {:?} + {:#x}: {}", l.ptr, l.offset, vals.join(" "))
        })
        .collect()
}

/// How often a stage's preparation repeats the previous one on its thread
/// (profile): [calls, same program / user data / register state, and the
/// same bytes read] per stage (vertex, pixel, compute).
pub static REPEATS: [[std::sync::atomic::AtomicU64; 3]; 4] = [const { [const { std::sync::atomic::AtomicU64::new(0) }; 3] }; 4];
/// The same with only the user-data dwords that descriptors and scalar
/// loads depend on in the key: [same deps, also same bytes] per stage.
pub static REPEATS_DEPS: [[std::sync::atomic::AtomicU64; 2]; 4] = [const { [const { std::sync::atomic::AtomicU64::new(0) }; 2] }; 4];

thread_local! {
    static LAST_PREP: std::cell::RefCell<[(u64, u64); 4]> = const { std::cell::RefCell::new([(0, 0); 4]) };
    static LAST_DEPS: std::cell::RefCell<[(u64, u64); 4]> = const { std::cell::RefCell::new([(0, 0); 4]) };
    /// This thread is a compute queue's command processor (statistics).
    static COMPUTE_QUEUE: std::cell::Cell<bool> = const { std::cell::Cell::new(false) };
}

/// User-data dwords that a plan's descriptors and scalar loads depend on.
fn user_deps(plan: &rdna::analysis::Plan) -> u32 {
    fn walk_ptr(p: &Ptr, m: &mut u32) {
        match p {
            Ptr::User(i) => *m |= 3 << (*i).min(30),
            Ptr::Mem(inner, _) => walk_ptr(inner, m),
            Ptr::Pc(_) => {}
        }
    }
    fn walk(p: &Prov, m: &mut u32) {
        match p {
            Prov::User(i) => *m |= 1 << (*i).min(31),
            Prov::Mem(ptr, _) | Prov::PtrLo(ptr, _) | Prov::PtrHi(ptr, _) => walk_ptr(ptr, m),
            _ => {}
        }
    }
    let mut m = 0u32;
    for r in &plan.resources {
        for d in &r.dwords {
            walk(d, &mut m);
        }
    }
    for l in &plan.loads {
        walk_ptr(&l.ptr, &mut m);
    }
    m
}

/// Preparations whose inputs and bytes read match one of the previous frame
/// (profile; how much a cache across frames could serve): per stage
/// [calls, same full inputs, same inputs counting only the user data the
/// descriptors depend on].
pub static CROSS_FRAME: [[std::sync::atomic::AtomicU64; 3]; 4] = [const { [const { std::sync::atomic::AtomicU64::new(0) }; 3] }; 4];

/// Marks the calling thread as a compute queue's command processor: its
/// compute preparations are counted apart from the graphics queue's
/// (`repeats_line`).
pub fn set_compute_queue(on: bool) {
    COMPUTE_QUEUE.with(|c| c.set(on));
}

/// The repeat statistics' class of a stage: vertex, pixel, compute on the
/// graphics queue (and its workers), compute on a compute queue.
fn stat_slot(stage: Stage) -> usize {
    match stage {
        Stage::Vertex => 0,
        Stage::Fragment => 1,
        _ if COMPUTE_QUEUE.with(|c| c.get()) => 3,
        _ => 2,
    }
}

/// (frame, this frame's (kind, stage, key, bytes) set, the previous frame's).
type FrameSets = (u64, super::fx::Set<(u8, u8, u64, u64)>, super::fx::Set<(u8, u8, u64, u64)>);
static FRAME_SETS: parking_lot::Mutex<Option<FrameSets>> = parking_lot::Mutex::new(None);

/// Notes a preparation's (kind, key, bytes) for the cross-frame statistic;
/// true when the previous frame had the same.
fn seen_last_frame(kind: u8, stage: usize, key: u64, bytes: u64) -> bool {
    let frame = super::FRAMES.load(std::sync::atomic::Ordering::Relaxed);
    let mut g = FRAME_SETS.lock();
    let st = g.get_or_insert_with(|| (frame, Default::default(), Default::default()));
    if st.0 != frame {
        st.2 = std::mem::take(&mut st.1);
        st.0 = frame;
    }
    let k = (kind, stage as u8, key, bytes);
    let hit = st.2.contains(&k);
    st.1.insert(k);
    hit
}

fn note_repeat_deps(stage: Stage, key: u64, bytes: u64) {
    use std::sync::atomic::Ordering;
    let s = stat_slot(stage);
    if seen_last_frame(1, s, key, bytes) {
        CROSS_FRAME[s][2].fetch_add(1, Ordering::Relaxed);
    }
    LAST_DEPS.with(|c| {
        let mut c = c.borrow_mut();
        if c[s].0 == key {
            REPEATS_DEPS[s][0].fetch_add(1, Ordering::Relaxed);
            if c[s].1 == bytes {
                REPEATS_DEPS[s][1].fetch_add(1, Ordering::Relaxed);
            }
        }
        c[s] = (key, bytes);
    });
}

fn note_repeat(stage: Stage, key: u64, reads: &[(u64, u64)]) -> u64 {
    use std::sync::atomic::Ordering;
    let s = stat_slot(stage);
    let mut h = super::fx::FxHasher::default();
    for &(a, n) in reads {
        h.write_u64(a);
        // One check per range: per-word checks made this profile-only
        // statistic most of the readable-range checks (~7.6 million per
        // second, run 20261011-115142) and inflated their profile share.
        if !super::cp::guest_readable(a, n) {
            continue;
        }
        let mut k = 0;
        while k + 4 <= n {
            h.write_u32(unsafe { std::ptr::read_volatile((a + k) as *const u32) });
            k += 4;
        }
    }
    let bytes = h.finish();
    REPEATS[s][0].fetch_add(1, Ordering::Relaxed);
    CROSS_FRAME[s][0].fetch_add(1, Ordering::Relaxed);
    if seen_last_frame(0, s, key, bytes) {
        CROSS_FRAME[s][1].fetch_add(1, Ordering::Relaxed);
    }
    LAST_PREP.with(|c| {
        let mut c = c.borrow_mut();
        if c[s].0 == key {
            REPEATS[s][1].fetch_add(1, Ordering::Relaxed);
            if c[s].1 == bytes {
                REPEATS[s][2].fetch_add(1, Ordering::Relaxed);
            }
        }
        c[s] = (key, bytes);
    });
    bytes
}

/// One line on how often preparations repeat (profile report).
pub fn repeats_line() -> String {
    use std::sync::atomic::Ordering;
    let names = ["vertex", "pixel", "compute (graphics queue)", "compute (compute queues)"];
    let parts: Vec<String> = (0..4)
        .map(|s| {
            let r = &REPEATS[s];
            let n = r[0].load(Ordering::Relaxed).max(1);
            let d = &REPEATS_DEPS[s];
            format!(
                "{} {} ({}% same inputs, {}% also same bytes; with only the user data descriptors depend on {}%, {}%)",
                names[s],
                r[0].load(Ordering::Relaxed),
                r[1].load(Ordering::Relaxed) * 100 / n,
                r[2].load(Ordering::Relaxed) * 100 / n,
                d[0].load(Ordering::Relaxed) * 100 / n,
                d[1].load(Ordering::Relaxed) * 100 / n
            )
        })
        .collect();
    let cross: Vec<String> = (0..4)
        .map(|s| {
            let c = &CROSS_FRAME[s];
            let n = c[0].load(Ordering::Relaxed).max(1);
            format!("{} {}% ({}% with only the user data descriptors depend on)", names[s], c[1].load(Ordering::Relaxed) * 100 / n, c[2].load(Ordering::Relaxed) * 100 / n)
        })
        .collect();
    format!(
        "    preparations repeating the previous one: {}\n    preparations identical to one of the previous frame (inputs and bytes read): {}\n",
        parts.join(", "),
        cross.join(", ")
    )
}

pub fn reset_repeats() {
    for r in CROSS_FRAME.iter() {
        for c in r.iter() {
            c.store(0, std::sync::atomic::Ordering::Relaxed);
        }
    }
    for r in REPEATS.iter() {
        for c in r.iter() {
            c.store(0, std::sync::atomic::Ordering::Relaxed);
        }
    }
    for r in REPEATS_DEPS.iter() {
        for c in r.iter() {
            c.store(0, std::sync::atomic::Ordering::Relaxed);
        }
    }
}

thread_local! {
    /// `format_resources` per plan.
    static LOCAL_FORMATS: std::cell::RefCell<super::fx::Map<u64, u64>> = std::cell::RefCell::new(super::fx::Map::default());
    /// `d3_resources` per plan.
    static LOCAL_DIM3: std::cell::RefCell<super::fx::Map<u64, u64>> = std::cell::RefCell::new(super::fx::Map::default());
    /// `d2_resources` per plan.
    static LOCAL_DIM2: std::cell::RefCell<super::fx::Map<u64, (u64, u64)>> = std::cell::RefCell::new(super::fx::Map::default());
}

/// The resources of `plan` whose V# format and swizzle the translation
/// reads (format loads and stores, typed buffer ones included; a bit per
/// resource, all past 64). Other buffers keep them out of their spec: one
/// translation whatever the title's descriptors hold there (and a spec a
/// shader object alone can give, `guess_compute_spec`).
fn format_resources(insts: &[rdna::Inst], plan: &Plan) -> u64 {
    if plan.resources.len() > 64 {
        return u64::MAX;
    }
    let mut m = 0u64;
    for i in insts {
        let Some(n) = i.name() else { continue };
        let n = n.strip_prefix('t').unwrap_or(n);
        if n.starts_with("buffer_load_format_") || n.starts_with("buffer_store_format_") {
            if let Some(&(r, _)) = plan.inst_res.get(&i.pc) {
                m |= 1 << r.min(63);
            }
        }
    }
    m
}

/// Learned resources keyed as a preparation keys them now (the store keeps
/// what dispatches had: typed buffers' formats, images' 2D or 3D): typed
/// buffers' formats of the table read at run time (`FORMAT_RT`; stored ones
/// too since 2026-11-18), and
/// 3D-addressed sampled images bound 2D or 3D `ImageDim::D3OrD2`. The
/// precompile, the compile bench and the guess report key them so.
pub(crate) fn key_learned(code: u64, insts: &[rdna::Inst], plan: &Plan, resources: &mut [ResSpec]) {
    let d3 = d3_resources(insts, plan);
    let (only2d, only2da) = if dim_by_instructions() { d2_resources(insts, plan) } else { (0, 0) };
    for (i, r) in resources.iter_mut().enumerate() {
        match *r {
            // (A program's one format kept, as preparations keep it:
            // `seed_formats` from its learned specs.)
            ResSpec::Buffer { format, dst_sel } if format != 0 && format != translate::FORMAT_RT && translate::rt_format_entry(format as u16) != 0 => {
                if !seeded_single(code, i, format, dst_sel) {
                    *r = ResSpec::Buffer { format: translate::FORMAT_RT, dst_sel: 0 };
                }
            }
            ResSpec::Image { dim: ImageDim::D2 | ImageDim::D3, class, depth } if i < 64 && d3 & (1 << i) != 0 && plan.resources.get(i).is_some_and(|x| x.kind == ResKind::Image && !x.written) => {
                *r = ResSpec::Image { dim: ImageDim::D3OrD2, class, depth };
            }
            // (2D and 2D arrays by the instructions' dimension, `d2_resources`.)
            ResSpec::Image { dim: ImageDim::D2 | ImageDim::D2Array, class, depth } if i < 64 && (only2d | only2da) & (1 << i) != 0 && plan.resources.get(i).is_some_and(|x| x.kind == ResKind::Image && !x.written) => {
                let dim = if only2da & (1 << i) != 0 { ImageDim::D2Array } else { ImageDim::D2 };
                *r = ResSpec::Image { dim, class, depth };
            }
            _ => {}
        }
    }
}

/// The resources of `plan` its image instructions address only as 2D, and those they address
/// only as 2D arrays (their DIM fields; a bit per resource, none past 64). A compute program's
/// sampled or loaded image is keyed by that dimension whatever the T# binds of the two
/// (`dim_by_instructions`): the hardware reads a 2D texture's only slice for a 2D-array
/// address and an array's base slice for a 2D one, as a 1-layer array view and a 2D view of the
/// base slice do, so one translation serves both (2026-11-15 night: the fog scatter's resource
/// 11, 2D-array-addressed, came bound as 2D and as a 2D array: two keys, 1.1-2.0 s compiles).
pub(crate) fn d2_resources(insts: &[rdna::Inst], plan: &Plan) -> (u64, u64) {
    let (mut d2, mut d2a, mut other) = (0u64, 0u64, 0u64);
    for i in insts {
        if i.enc != rdna::Enc::Mimg {
            continue;
        }
        if let Some(&(r, _)) = plan.inst_res.get(&i.pc) {
            if r < 64 {
                match i.mimg_dim() {
                    1 => d2 |= 1 << r,
                    5 => d2a |= 1 << r,
                    _ => other |= 1 << r,
                }
            }
        }
    }
    (d2 & !d2a & !other, d2a & !d2 & !other)
}

/// Whether compute images bound as 2D or 2D arrays are keyed by their instructions'
/// dimension (`d2_resources`; `DS_IMG_DIM_INST=0`: by the T#'s type, as before 2026-11-15).
pub(crate) fn dim_by_instructions() -> bool {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ON.get_or_init(|| std::env::var("DS_IMG_DIM_INST").ok().as_deref() != Some("0"))
}

/// The resources of `plan` its image instructions address as 3D (their DIM
/// field; a bit per resource, none past 64): sampled ones bound with a 2D or
/// 3D texture are `ImageDim::D3OrD2`.
pub(crate) fn d3_resources(insts: &[rdna::Inst], plan: &Plan) -> u64 {
    let mut m = 0u64;
    for i in insts {
        if i.enc == rdna::Enc::Mimg && i.mimg_dim() == 2 {
            if let Some(&(r, _)) = plan.inst_res.get(&i.pc) {
                if r < 64 {
                    m |= 1 << r;
                }
            }
        }
    }
    m
}

/// Translates `insts` (planned as `plan`) for `spec` and keeps the
/// translation under (plan_key, spec_hash) for `key_spec` (the spec it is
/// kept for: the code base left out).
fn translate_kept(addr: u64, insts: &[rdna::Inst], plan: &Arc<Plan>, plan_key: u64, spec_hash: u64, spec: &Spec, key_spec: Spec) -> Result<Arc<Translated>, String> {
    let t0 = std::time::Instant::now();
    // (Kept across runs: `gpu::trcache`.)
    let out = match super::trcache::load(plan_key, spec_hash) {
        Some(o) => {
            if super::trcache::verify() {
                match translate::translate(insts, plan, spec) {
                    Ok(fresh) => super::trcache::note_verify(&o, &fresh, || format!("{:?} shader {addr:#x} (spec {spec_hash:016x})", spec.stage)),
                    Err(e) => crate::warn!("translation store: {:?} shader {addr:#x} no longer translates ({e})", spec.stage),
                }
            }
            o
        }
        None => {
            let o = translate::translate(insts, plan, spec).map_err(|e| format!("{:?} shader {addr:#x}: {e}", spec.stage))?;
            super::trcache::store(plan_key, spec_hash, &o);
            o
        }
    };
    super::hitch::note_since(super::hitch::Kind::Translate, t0, || format!("{:?} {addr:#x}", spec.stage));
    let ms = t0.elapsed().as_millis();
    if ms >= 100 {
        crate::info!("{:?} shader {addr:#x}: translated in {ms} ms ({} instructions, {} SPIR-V words)", spec.stage, insts.len(), out.spirv.len());
    }
    // Diagnosis (`DS_TRANSLATE_STATS=1`): every translation's size and the
    // spec's lane model, for a census of the costliest programs; 2 also
    // saves each program and its SPIR-V into the run directory's shaders/.
    static STATS: std::sync::OnceLock<u8> = std::sync::OnceLock::new();
    let stats = *STATS.get_or_init(|| std::env::var("DS_TRANSLATE_STATS").ok().and_then(|v| v.parse().ok()).unwrap_or(0));
    if stats >= 2 {
        if super::capture::save_program(addr).is_some() {
            if let Some(dir) = crate::run_dir() {
                let bytes: Vec<u8> = out.spirv.iter().flat_map(|w| w.to_le_bytes()).collect();
                let _ = std::fs::write(dir.join("shaders").join(format!("{addr:012x}-{spec_hash:016x}.spv")), bytes);
            }
        }
    }
    // (A program's further specs: which fields split its pipelines.)
    if stats >= 1 && spec.stage == Stage::Compute {
        static LAST: Mutex<Option<HashMap<u64, Spec>>> = Mutex::new(None);
        let mut g = LAST.lock();
        let m = g.get_or_insert_with(HashMap::new);
        if let Some(prev) = m.get(&plan_key) {
            crate::info!("translate stats: Compute shader {addr:#x}: another spec ({})", super::exec::spec_diff(prev, &key_spec));
        }
        m.insert(plan_key, key_spec.clone());
    }
    if stats >= 1 {
        crate::info!(
            "translate stats: {:?} shader {addr:#x}: {} instructions -> {} SPIR-V words ({:.1} per instruction; wave64 {}, per lane {}, lane votes {}, lockstep {}, wave masks {}, subgroup waves {}, local size {:?}, {} resources, {} branches decided without a vote)",
            spec.stage,
            insts.len(),
            out.spirv.len(),
            out.spirv.len() as f64 / insts.len().max(1) as f64,
            spec.wave64,
            spec.per_lane,
            spec.lane_votes,
            out.lockstep,
            out.wave_masks,
            out.sg_waves,
            spec.cs_local_size,
            plan.resources.len(),
            out.masked_skips
        );
    }
    // Large translations compile for seconds on first use (play stalled
    // 0.5-3.4 s per effect shader, FINDINGS.md 2026-10-14): each logged
    // once, its program saved.
    if out.spirv.len() >= BIG_SPIRV_WORDS && BIG_SPIRV_SEEN.lock().get_or_insert_with(std::collections::HashSet::new).insert(addr) {
        let saved = super::capture::save_program(addr).map(|f| format!("; program saved to {}", f.display())).unwrap_or_default();
        let from = program_hash(addr).ok().and_then(super::particles::origin).map(|d| format!("; from {d}")).unwrap_or_default();
        crate::info!(
            "{:?} shader {addr:#x}: {} instructions -> {} SPIR-V words ({:.0} per instruction; wave64 {}, per lane {}, lane votes {}, local size {:?}){from}{saved}",
            spec.stage,
            insts.len(),
            out.spirv.len(),
            out.spirv.len() as f64 / insts.len().max(1) as f64,
            spec.wave64,
            spec.per_lane,
            spec.lane_votes,
            spec.cs_local_size
        );
    }
    for w in &out.warnings {
        crate::warn!("{:?} shader {addr:#x}: {w}", spec.stage);
    }
    let t = Arc::new(translated(insts, out, plan, plan_key, spec_hash, key_spec));
    TRANSLATIONS.lock().get_or_insert_with(HashMap::new).insert((plan_key, spec_hash), t.clone());
    Ok(t)
}

/// A translation's record: its output, plan, key and the resources its
/// atomics use.
fn translated(insts: &[rdna::Inst], out: Output, plan: &Arc<Plan>, plan_key: u64, spec_hash: u64, key_spec: Spec) -> Translated {
    let mut atomic_res: Vec<usize> = insts
        .iter()
        .filter(|i| i.name().is_some_and(|n| n.starts_with("buffer_atomic_") || n.starts_with("global_atomic_")))
        .filter_map(|i| plan.inst_res.get(&i.pc).map(|r| r.0))
        .collect();
    atomic_res.sort_unstable();
    atomic_res.dedup();
    Translated { out, plan: Arc::clone(plan), key: plan_key ^ spec_hash.rotate_left(1), atomic_res, spec: key_spec }
}

/// `translate_code_for_spec` keeping neither the translation nor its plan,
/// and quiet (no hitch notes or logs): `--precompile-all` makes thousands,
/// of which only the pipelines are kept (all translations of the game's
/// files are ~566 million SPIR-V words).
pub(crate) fn translate_code_unkept(code: &[u32], code_hash: u64, key_spec: Spec) -> Result<Translated, String> {
    if hash_words(code) != code_hash {
        return Err("stored code does not match its hash".into());
    }
    let plan_key = code_hash ^ (key_spec.user_sgprs as u64) << 48 ^ (key_spec.user_base as u64) << 56;
    let insts = rdna::decode_program(code).map_err(|e| format!("decode stored code: {}", e.0))?;
    let plan = Arc::new(rdna::analysis::analyze_opts(&insts, key_spec.user_sgprs, key_spec.user_base, false));
    if !plan.errors.is_empty() {
        return Err(plan.errors.join("; "));
    }
    let spec_hash = {
        let mut h = std::collections::hash_map::DefaultHasher::new();
        key_spec.hash(&mut h);
        h.finish()
    };
    let out = translate::translate(&insts, &plan, &key_spec).map_err(|e| format!("{:?} shader {code_hash:#x}: {e}", key_spec.stage))?;
    Ok(translated(&insts, out, &plan, plan_key, spec_hash, key_spec))
}

/// The code hash of the program at `addr` (as `prepare` keys it).
pub(crate) fn program_hash(addr: u64) -> Result<u64, String> {
    program_local(addr).map(|(_, h)| h)
}

/// The words of the program at `addr` its code hash covers (up to its last
/// instruction), and that hash (`gpu::precompile` keeps them).
pub(crate) fn program_code(addr: u64) -> Result<(Vec<u32>, u64), String> {
    let code = code_slice(addr).ok_or_else(|| format!("shader {addr:#x} not mapped"))?;
    let insts = rdna::decode_program(code).map_err(|e| format!("decode {addr:#x}: {}", e.0))?;
    let last = insts.last().ok_or("empty program")?;
    let words = ((last.pc / 4 + last.len as u32) as usize).min(code.len());
    Ok((code[..words].to_vec(), hash_words(&code[..words])))
}

/// The translation of stored program words (`gpu::precompile`, before the
/// title has loaded the program) for `key_spec`, kept where `prepare`
/// looks for it (translations do not depend on the program's address).
pub(crate) fn translate_code_for_spec(code: &[u32], code_hash: u64, key_spec: Spec) -> Result<Arc<Translated>, String> {
    if hash_words(code) != code_hash {
        return Err("stored code does not match its hash".into());
    }
    let plan_key = code_hash ^ (key_spec.user_sgprs as u64) << 48 ^ (key_spec.user_base as u64) << 56;
    let cached = PLANS.lock().as_ref().and_then(|m| m.get(&plan_key).cloned());
    // (Decoded only when needed: the plan is cached, the program here.)
    let mut insts = None;
    let plan = match cached {
        Some(p) => p,
        None => {
            let i = rdna::decode_program(code).map_err(|e| format!("decode stored code: {}", e.0))?;
            let p = Arc::new(rdna::analysis::analyze_opts(&i, key_spec.user_sgprs, key_spec.user_base, false));
            PLANS.lock().get_or_insert_with(HashMap::new).insert(plan_key, p.clone());
            insts = Some(i);
            p
        }
    };
    let spec_hash = {
        let mut h = std::collections::hash_map::DefaultHasher::new();
        key_spec.hash(&mut h);
        h.finish()
    };
    if let Some(t) = TRANSLATIONS.lock().as_ref().and_then(|m| m.get(&(plan_key, spec_hash)).cloned()) {
        return Ok(t);
    }
    let insts = match insts {
        Some(i) => i,
        None => rdna::decode_program(code).map_err(|e| format!("decode stored code: {}", e.0))?,
    };
    if !plan.errors.is_empty() {
        return Err(plan.errors.join("; "));
    }
    translate_kept(0, &insts, &plan, plan_key, spec_hash, &key_spec.clone(), key_spec)
}

/// `program_hash` without keeping the decoded program (`gpu::precompile`
/// looks at every shader object the title creates, ~40,000 in a session,
/// and makes pipelines for few).
pub(crate) fn program_hash_uncached(addr: u64) -> Result<u64, String> {
    if let Some(p) = PROGRAMS.lock().as_ref().and_then(|m| m.get(&addr).map(|p| (p.words, p.hash))) {
        if let Some(code) = code_slice(addr) {
            if p.0 <= code.len() && hash_words(&code[..p.0]) == p.1 {
                return Ok(p.1);
            }
        }
    }
    let code = code_slice(addr).ok_or_else(|| format!("shader {addr:#x} not mapped"))?;
    let insts = rdna::decode_program(code).map_err(|e| format!("decode {addr:#x}: {}", e.0))?;
    let last = insts.last().ok_or("empty program")?;
    let words = (last.pc / 4 + last.len as u32) as usize;
    Ok(hash_words(&code[..words.min(code.len())]))
}

/// The decoded program at `addr` and its plan for `user_sgprs` user-data
/// dwords from SGPR `user_base` (as `prepare` makes them without GPU-side
/// data loads).
pub(crate) fn stage_plan(addr: u64, user_sgprs: u16, user_base: u16) -> Result<(Arc<Local<Vec<rdna::Inst>>>, Arc<Plan>, u64), String> {
    let (insts, code_hash) = program_local(addr)?;
    let plan_key = code_hash ^ (user_sgprs as u64) << 48 ^ (user_base as u64) << 56;
    let cached = PLANS.lock().as_ref().and_then(|m| m.get(&plan_key).cloned());
    let plan = match cached {
        Some(p) => p,
        None => {
            let p = Arc::new(rdna::analysis::analyze_opts(&insts, user_sgprs, user_base, false));
            PLANS.lock().get_or_insert_with(HashMap::new).insert(plan_key, p.clone());
            p
        }
    };
    if !plan.errors.is_empty() {
        return Err(plan.errors.join("; "));
    }
    Ok((insts, plan, code_hash))
}

/// The translation of the program at `addr` for `key_spec` (a key spec as
/// `compute_spec`, `vertex_spec` or `pixel_spec` make it; `gpu::precompile`),
/// kept where `prepare` looks for it.
pub(crate) fn translate_for_spec(addr: u64, key_spec: Spec) -> Result<Arc<Translated>, String> {
    let (insts, plan, code_hash) = stage_plan(addr, key_spec.user_sgprs, key_spec.user_base)?;
    let plan_key = code_hash ^ (key_spec.user_sgprs as u64) << 48 ^ (key_spec.user_base as u64) << 56;
    let spec_hash = {
        let mut h = std::collections::hash_map::DefaultHasher::new();
        key_spec.hash(&mut h);
        h.finish()
    };
    if let Some(t) = TRANSLATIONS.lock().as_ref().and_then(|m| m.get(&(plan_key, spec_hash)).cloned()) {
        return Ok(t);
    }
    let mut spec = key_spec.clone();
    spec.code_base = addr;
    translate_kept(addr, &insts, &plan, plan_key, spec_hash, &spec, key_spec)
}

/// The translation spec of a compute program at `addr` guessed from its
/// shader object (`gpu::precompile`): the user-data count, RSRC2 and thread
/// group size it carries; wave64 unless the code saves EXEC in 32 bits;
/// images by their instructions' dimension as float views, those sampled
/// with a depth comparison as depth views; buffers as `prepare` keys them
/// (unknown for format accesses).
pub(crate) fn guess_compute_spec(addr: u64, insts: &[rdna::Inst], plan: &Plan, code_hash: u64, rsrc2: u32, local: [u32; 3]) -> Spec {
    let wave64 = !insts.iter().any(|i| i.name().is_some_and(|n| n.ends_with("_saveexec_b32") || n.ends_with("_wrexec_b32")));
    let fmt = format_resources(insts, plan);
    let user_sgprs = ((rsrc2 >> 1) & 0x1F).min(16) as u16;
    let plan_key = code_hash ^ (user_sgprs as u64) << 48;
    let cmp = super::tune::cmp_emulate().then(|| compare_images(plan_key, insts, plan)).unwrap_or(0);
    let resources = plan
        .resources
        .iter()
        .enumerate()
        .map(|(r, res)| match res.kind {
            // (Stored ones too since 2026-11-18: `FORMAT_RT` packs stores, so the
            // guess no longer picks the commonest format of the stores' width.)
            ResKind::Buffer if fmt & (1 << r.min(63)) != 0 => ResSpec::Buffer { format: translate::FORMAT_RT, dst_sel: 0 },
            ResKind::Buffer => ResSpec::Buffer { format: 0, dst_sel: 0 },
            ResKind::Global => ResSpec::Global,
            ResKind::Sampler => ResSpec::Sampler,
            ResKind::Image | ResKind::ImageTable => {
                let dim = insts
                    .iter()
                    .find(|i| i.enc == rdna::Enc::Mimg && plan.inst_res.get(&i.pc).is_some_and(|x| x.0 == r))
                    .map(|i| match i.mimg_dim() {
                        0 => ImageDim::D1,
                        1 => ImageDim::D2,
                        2 if !res.written && res.kind == ResKind::Image => ImageDim::D3OrD2,
                        2 => ImageDim::D3,
                        3 => ImageDim::Cube,
                        4 => ImageDim::D1Array,
                        5 => ImageDim::D2Array,
                        6 => ImageDim::D2Msaa,
                        _ => ImageDim::D2MsaaArray,
                    })
                    .unwrap_or(ImageDim::D2);
                // As `prepare` decides: images keep depth views without
                // comparison emulation, else only the compared ones.
                let depth = match res.kind {
                    ResKind::ImageTable => false,
                    _ if !super::tune::cmp_emulate() => true,
                    _ => r < 64 && cmp & (1 << r) != 0,
                };
                ResSpec::Image { dim, class: guess_image_class(insts, plan, r), depth }
            }
        })
        .collect();
    compute_spec(addr, code_hash, rsrc2, local, wave64, resources, batchable(addr))
}

/// The preparation's other guesses of an engine program's spec (`compbench::precompile_all`):
/// image dimensions and depth views its instructions do not settle, which the descriptor decides
/// at run time (the guess report of 2026-11-13: the engine's cube instructions on cube arrays in
/// 341 learned specs, its 2D-array instructions on 2D textures in 309, compared images bound
/// with colour textures). The guess with every cube image as a cube array, with every 2D array as
/// 2D, and with both; each of those and the guess also with every depth view a colour view.
pub(crate) fn guess_dim_variants(guess: &Spec, keyed2d: u64) -> Vec<Spec> {
    // (2D arrays the dispatches key by their instructions, `keyed2d`, are not flipped.)
    let flippable = |i: usize| i >= 64 || keyed2d & (1 << i) == 0;
    let has = |d: ImageDim| guess.resources.iter().enumerate().any(|(i, r)| matches!(r, ResSpec::Image { dim, .. } if *dim == d && (d != ImageDim::D2Array || flippable(i))));
    let flip = |cube: bool, array: bool| {
        let mut s = guess.clone();
        for (i, r) in s.resources.iter_mut().enumerate() {
            if let ResSpec::Image { dim, .. } = r {
                if cube && *dim == ImageDim::Cube {
                    *dim = ImageDim::CubeArray;
                } else if array && *dim == ImageDim::D2Array && flippable(i) {
                    *dim = ImageDim::D2;
                }
            }
        }
        s
    };
    let (cube, array) = (has(ImageDim::Cube), has(ImageDim::D2Array));
    let mut v = Vec::new();
    if cube {
        v.push(flip(true, false));
    }
    if array {
        v.push(flip(false, true));
    }
    if cube && array {
        v.push(flip(true, true));
    }
    // Each also with its compared images' depth views as colour views (a compared image the
    // title binds with a colour texture: the report's depth-view differences, 141 more specs):
    // every combination of up to three of them (the fog scatter's three shadow textures came in
    // three of their eight, 2026-11-15 night), else all of them at once.
    let compared: Vec<usize> = guess.resources.iter().enumerate().filter(|(_, r)| matches!(r, ResSpec::Image { depth: true, .. })).map(|(i, _)| i).collect();
    if !compared.is_empty() {
        let mut all = vec![guess.clone()];
        all.extend(v.iter().cloned());
        let masks: Vec<u32> = if compared.len() <= 3 { (1..(1u32 << compared.len())).collect() } else { vec![(1u32 << compared.len().min(31)) - 1] };
        for s in all {
            for &m in &masks {
                let mut s = s.clone();
                for (k, &i) in compared.iter().enumerate() {
                    if m & (1 << k.min(31)) != 0 {
                        if let Some(ResSpec::Image { depth, .. }) = s.resources.get_mut(i) {
                            *depth = false;
                        }
                    }
                }
                v.push(s);
            }
        }
    }
    v
}

/// The preparation's other guess of any program's spec (`compbench::precompile_all`): images its
/// instructions address as 3D that it also writes (no `ImageDim::D3OrD2` for them), as 2D (the
/// title binds 2D textures there: 29 misses of 26 particle and material programs in a prepared
/// first session, 2026-11-13). None when it has no such image.
pub(crate) fn guess_d3_variant(guess: &Spec) -> Option<Spec> {
    if !guess.resources.iter().any(|r| matches!(r, ResSpec::Image { dim: ImageDim::D3, .. })) {
        return None;
    }
    let mut s = guess.clone();
    for r in s.resources.iter_mut() {
        if let ResSpec::Image { dim, .. } = r {
            if *dim == ImageDim::D3 {
                *dim = ImageDim::D2;
            }
        }
    }
    Some(s)
}

/// The class a stored value's last writer computes it in (`guess_image_class`): a conversion by
/// its result (`v_cvt_f32_u32` makes a float), else by its operands' type as for loads.
fn store_class_votes(n: &str) -> (u32, u32) {
    if n.starts_with("v_cvt_f32") || n.starts_with("v_cvt_f16") || n.starts_with("v_cvt_pk") && n.contains("_f") {
        return (0, 1);
    }
    if n.starts_with("v_cvt_u32") || n.starts_with("v_cvt_i32") || n.starts_with("v_cvt_u16") || n.starts_with("v_cvt_i16") || n.starts_with("v_cvt_flr_i32") || n.starts_with("v_cvt_rpi_i32") {
        return (1, 0);
    }
    let int = n.contains("_u32") || n.contains("_i32") || n.contains("_b32") || n.contains("_u16") || n.contains("_i16") || n.contains("bfe") || n.contains("bfi") || n.contains("lshl") || n.contains("lshr") || n.contains("ashr") || n.contains("and_") || n.contains("or_") || n.contains("xor") || n.contains("perm");
    let float = !int && (n.contains("_f32") || n.contains("_f16") || n.contains("_f64"));
    (int as u32, float as u32)
}

/// The integer and float votes of what a move or select `j` (at `at`) passes on: its value
/// sources' last writers before it (within 64 instructions), followed through further moves and
/// selects up to `depth` steps (`store_class_votes`); an inline integer constant votes integer,
/// an inline float constant float.
fn passed_on_votes(insts: &[rdna::Inst], at: usize, j: &rdna::Inst, depth: u32) -> (u32, u32) {
    let n = j.name().unwrap_or("");
    let srcs: &[usize] = if n.starts_with("v_cndmask") { &[0, 1] } else { &[0] };
    let (mut iv, mut fv) = (0u32, 0u32);
    for &s in srcs {
        let o = j.src(s) as u32;
        match o {
            128..=208 => iv += 1,
            240..=248 => fv += 1,
            256..=511 => {
                let v = o - 256;
                for (back, w) in insts[..at].iter().rev().take(64).enumerate() {
                    if !matches!(w.enc, rdna::Enc::Vop1 | rdna::Enc::Vop2 | rdna::Enc::Vop3 | rdna::Enc::Vop3p) || w.vdst() as u32 != v {
                        continue;
                    }
                    let Some(wn) = w.name() else { break };
                    if wn.starts_with("v_cmp") || wn.starts_with("v_readlane") || wn.starts_with("v_readfirstlane") {
                        continue;
                    }
                    let (a, b) = if wn.starts_with("v_mov") || wn.starts_with("v_cndmask") {
                        if depth > 0 {
                            passed_on_votes(insts, at - 1 - back, w, depth - 1)
                        } else {
                            (0, 0)
                        }
                    } else {
                        store_class_votes(wn)
                    };
                    iv += a;
                    fv += b;
                    break;
                }
            }
            _ => {}
        }
    }
    (iv, fv)
}

/// An image's number class as its program uses it (`guess_compute_spec`; the class itself comes
/// from the descriptor's format at run time): an image the program only loads from or stores to,
/// whose loaded words feed integer operations first, or whose stored words integer operations
/// computed last (bit operations, integer arithmetic and compares, a conversion from an
/// integer), is guessed Uint; anything else Float. (The guess report:
/// the engine's images read with image_load alone were Uint in 952 specs and Float in 1,050.)
fn guess_image_class(insts: &[rdna::Inst], plan: &Plan, r: usize) -> NumClass {
    let on_r = |i: &rdna::Inst| i.enc == rdna::Enc::Mimg && plan.inst_res.get(&i.pc).is_some_and(|x| x.0 == r);
    let mut loads = insts.iter().enumerate().filter(|(_, i)| on_r(i)).peekable();
    if loads.peek().is_none() {
        return NumClass::Float;
    }
    let (mut int_votes, mut float_votes) = (0u32, 0u32);
    // (What stored words' moves and selects pass on: decides only when nothing else voted.)
    let (mut passed_int, mut passed_float) = (0u32, 0u32);
    // (The class an instruction computes in, by its name: integer, float, or neither.)
    let class_of = |n: &str| -> (bool, bool) {
        let int = n.contains("_u32") || n.contains("_i32") || n.contains("_b32") || n.contains("_u16") || n.contains("_i16") || n.contains("bfe") || n.contains("bfi") || n.contains("lshl") || n.contains("lshr") || n.contains("ashr") || n.contains("and_") || n.contains("or_") || n.contains("xor") || n.contains("perm") || n.starts_with("v_cvt_f32_u") || n.starts_with("v_cvt_f32_i") || n.starts_with("v_cvt_f32_ubyte");
        let float = !int && (n.contains("_f32") || n.contains("_f16") || n.contains("_f64"));
        (int, float)
    };
    for (k, i) in insts.iter().enumerate().filter(|(_, i)| on_r(i)) {
        let Some(name) = i.name() else { return NumClass::Float };
        if !name.starts_with("image_load") || name.contains("d16") {
            if !name.starts_with("image_store") || name.contains("d16") {
                // Sampled or anything else: as before.
                return NumClass::Float;
            }
            // A stored image (2026-11-18): the class its stored words' last writers compute in
            // (cs 0x8026f4100 packs its four words with shifts and ors into a Uint image; guessed
            // Float, its 192,061-word program compiled at a new player's start, 2.3 s).
            let first = i.vdst() as u32;
            let count = i.mimg_dmask().count_ones().max(1);
            let mut found = 0u32;
            for (back, j) in insts[..k].iter().rev().take(64).enumerate() {
                if !matches!(j.enc, rdna::Enc::Vop1 | rdna::Enc::Vop2 | rdna::Enc::Vop3 | rdna::Enc::Vop3p) {
                    continue;
                }
                let Some(n) = j.name() else { continue };
                if n.starts_with("v_cmp") || n.starts_with("v_readlane") || n.starts_with("v_readfirstlane") {
                    continue;
                }
                let d = j.vdst() as u32;
                if d < first || d >= first + count || found & (1 << (d - first)) != 0 {
                    continue;
                }
                found |= 1 << (d - first);
                // (Moves and selects pass a value on: the values they pass vote, followed two
                // steps back (2026-11-19: the BC3 face encoder cs 0x8026ec300 stores words its
                // selects pick from bit operations; with no vote it was guessed Float, and its
                // first dispatch compiled in place).)
                if n.starts_with("v_mov") || n.starts_with("v_cndmask") {
                    let (int, float) = passed_on_votes(insts, k - 1 - back, j, 2);
                    passed_int += int;
                    passed_float += float;
                } else {
                    let (int, float) = store_class_votes(n);
                    int_votes += int;
                    float_votes += float;
                }
                if found == (1u32 << count) - 1 {
                    break;
                }
            }
            continue;
        }
        // The loaded VGPRs and the first instructions after the load that read one of them.
        let first = i.vdst() as u32;
        let count = i.mimg_dmask().count_ones().max(1);
        let reads = |j: &rdna::Inst| -> bool {
            let srcs: &[usize] = match j.enc {
                rdna::Enc::Vop1 => &[0],
                rdna::Enc::Vop2 | rdna::Enc::Vopc => &[0, 1],
                rdna::Enc::Vop3 | rdna::Enc::Vop3p => &[0, 1, 2],
                _ => &[],
            };
            srcs.iter().any(|&s| {
                let o = j.src(s) as u32;
                o >= 256 && (o - 256) >= first && (o - 256) < first + count
            })
        };
        let mut seen = 0;
        for j in insts[k + 1..].iter().take(64) {
            if !reads(j) {
                continue;
            }
            let Some(n) = j.name() else { break };
            let (int, float) = class_of(n);
            if int {
                int_votes += 1;
            } else if float {
                float_votes += 1;
            }
            seen += 1;
            if seen >= 3 {
                break;
            }
        }
    }
    if int_votes == 0 && float_votes == 0 {
        (int_votes, float_votes) = (passed_int, passed_float);
    }
    if int_votes > float_votes { NumClass::Uint } else { NumClass::Float }
}

/// The key spec (code base left out) `prepare_ngg` makes for the vertex
/// program at `addr` drawn with these registers and descriptors, under the
/// current switches (None while vertex dumps are on: they depend on the
/// code).
pub(crate) fn vertex_spec(addr: u64, user_sgprs: u16, wave64: bool, base_instance_sgpr: Option<u16>, resources: Vec<ResSpec>) -> Option<Spec> {
    if super::capture::vs_dump().is_some() || super::capture::vs_dump_for(addr).is_some() {
        return None;
    }
    let _ = addr;
    Some(Spec {
        stage: Stage::Vertex,
        user_sgprs,
        user_base: 8,
        code_base: 0,
        wave64,
        ps_input_addr: 0,
        ps_inputs: Vec::new(),
        barycentric: false,
        bary_gs: false,
        mrt: [MrtSpec::default(); 8],
        cs_local_size: [1, 1, 1],
        cs_rsrc2: 0,
        resources,
        binding_base: 0,
        probe: None,
        dump: None,
        force_dispatcher: super::tune::block_chain() == 0,
        dispatcher_for_loops: super::tune::block_chain() == 2,
        dom_nest: super::tune::block_chain() == 3,
        reg_vars: super::tune::reg_vars(),
        lane_guards: true,
        mip_views: super::tune::mip_views(),
        lod_stats: matches!(super::tune::lod_stats(), 3 | 4),
        lod_lean: super::tune::lod_stats() == 4,
        early_tests: false,
        early_ok: false,
        per_lane: false,
        lane_votes: false,
        lockstep_off: false,
        sg_waves: false,
        sg_appends: false,
        known_active: true,
        hw_bounds: super::tune::hw_bounds_on(),
        base_instance_sgpr,
        sgpr_pc: super::tune::sgpr_pc().then_some(translate::sgpr_pc_offset(Stage::Vertex)),
        addresses: false,
        batched: false,
        write_marks: marks_wanted(),
        marks_untouched: !super::tune::mark_touch(),
        touch_test: touch_test_wanted(),
        mark_twice: marks_wanted() && super::tune::tr_ballast() & 1 != 0,
        touch_twice: marks_wanted() && super::tune::tr_ballast() & 4 != 0,
        part_twice: marks_wanted() && super::tune::tr_ballast() & 8 != 0,
        dep_loads: super::tune::tr_ballast() & 2 != 0,
        marks_part: marks_part_wanted(),
        marks_held: marks_held_wanted(),
        sgpr_ubo: super::tune::sgpr_ubo(),
        gpu_loads: 0,
        gpu_res: 0,
        pt_linear: false,
        outlier: vs_outlier().and_then(|t| program_hash(addr).ok().map(|h| (outlier_low(h).unwrap_or(t), outlier_named(h, addr)))),
    })
}

/// The key spec `prepare_ps` makes for the pixel program at `addr` drawn
/// with these registers and descriptors, under the current switches
/// (`measure`: a draw measuring LOD statistics in rotation).
#[allow(clippy::too_many_arguments)]
pub(crate) fn pixel_spec(addr: u64, user_sgprs: u16, wave64: bool, ps_input_addr: u32, ps_inputs: Vec<PsInput>, mrt: [MrtSpec; 8], early_tests: bool, early_ok: bool, resources: Vec<ResSpec>, measure: bool) -> Spec {
    let lod = matches!(super::tune::lod_stats(), 3 | 4) && (early_tests || early_ok || late_stats(measure));
    Spec {
        stage: Stage::Fragment,
        user_sgprs,
        user_base: 0,
        code_base: 0,
        wave64,
        ps_input_addr,
        ps_inputs,
        barycentric: super::vk::ctx().barycentric,
        bary_gs: super::vk::ctx().bary_gs,
        mrt,
        cs_local_size: [1, 1, 1],
        cs_rsrc2: 0,
        resources,
        binding_base: PS_BINDING_BASE,
        probe: probe_for(addr),
        dump: None,
        force_dispatcher: super::tune::block_chain() == 0,
        dispatcher_for_loops: super::tune::block_chain() == 2,
        dom_nest: super::tune::block_chain() == 3,
        reg_vars: super::tune::reg_vars(),
        lane_guards: true,
        mip_views: super::tune::mip_views(),
        lod_stats: lod,
        lod_lean: lod && super::tune::lod_stats() == 4,
        early_tests,
        early_ok,
        per_lane: false,
        lane_votes: false,
        lockstep_off: false,
        sg_waves: false,
        sg_appends: false,
        known_active: true,
        hw_bounds: super::tune::hw_bounds_on(),
        base_instance_sgpr: None,
        sgpr_pc: super::tune::sgpr_pc().then_some(translate::sgpr_pc_offset(Stage::Fragment)),
        addresses: false,
        batched: false,
        write_marks: marks_wanted(),
        marks_untouched: !super::tune::mark_touch(),
        touch_test: touch_test_wanted(),
        mark_twice: marks_wanted() && super::tune::tr_ballast() & 1 != 0,
        touch_twice: marks_wanted() && super::tune::tr_ballast() & 4 != 0,
        part_twice: marks_wanted() && super::tune::tr_ballast() & 8 != 0,
        dep_loads: super::tune::tr_ballast() & 2 != 0,
        marks_part: marks_part_wanted(),
        marks_held: marks_held_wanted(),
        sgpr_ubo: super::tune::sgpr_ubo(),
        gpu_loads: 0,
        gpu_res: 0,
        pt_linear: false,
        outlier: None,
    }
}

/// The key spec (code base left out) `prepare_compute` makes for the
/// compute program at `addr` (code hash `code_hash`) dispatched with these
/// registers and descriptors, under the current switches.
/// (`batched`: translated for runs of consecutive dispatches, as a program
/// `batchable` is: learned with its spec, or the program's state now.)
/// The specs a learned compute spec (`compute_spec`) stands for at
/// dispatch: with compute LOD statistics in rotation (`cs_lod_rotation`)
/// most dispatches are prepared without them and one in `tune lod_cs` with
/// them (`lodstats::rotate`), so the precompile makes both, the one without
/// first (it made only the one with them from 2026-11-05 on: a variant of
/// most programs compiled at first use unless the pipeline store held it,
/// FINDINGS.md 2026-11-06); else the spec alone.
pub(crate) fn dispatch_variants(spec: Spec) -> Vec<Spec> {
    if spec.stage == Stage::Compute && spec.lod_stats && !spec.batched && cs_lod_rotation() {
        let mut rotated = spec.clone();
        rotated.lod_stats = false;
        rotated.lod_lean = false;
        vec![rotated, spec]
    } else {
        vec![spec]
    }
}

pub(crate) fn compute_spec(addr: u64, code_hash: u64, rsrc2: u32, local: [u32; 3], wave64: bool, resources: Vec<ResSpec>, batched: bool) -> Spec {
    let mut spec = Spec {
        stage: Stage::Compute,
        user_sgprs: ((rsrc2 >> 1) & 0x1F).min(16) as u16,
        user_base: 0,
        code_base: 0,
        wave64,
        ps_input_addr: 0,
        ps_inputs: Vec::new(),
        barycentric: false,
        bary_gs: false,
        mrt: [MrtSpec::default(); 8],
        cs_local_size: local,
        cs_rsrc2: rsrc2,
        resources,
        binding_base: 0,
        probe: probe_for(addr),
        dump: dump_for(addr),
        force_dispatcher: super::tune::block_chain() == 0,
        dispatcher_for_loops: super::tune::block_chain() == 2,
        dom_nest: super::tune::block_chain() == 3,
        reg_vars: super::tune::reg_vars(),
        lane_guards: true,
        mip_views: super::tune::mip_views(),
        lod_stats: matches!(super::tune::lod_stats(), 3 | 4),
        lod_lean: super::tune::lod_stats() == 4,
        early_tests: false,
        early_ok: false,
        per_lane: false,
        lane_votes: false,
        lockstep_off: false,
        sg_waves: false,
        sg_appends: false,
        known_active: true,
        hw_bounds: super::tune::hw_bounds_on(),
        base_instance_sgpr: None,
        sgpr_pc: super::tune::sgpr_pc().then_some(translate::sgpr_pc_offset(Stage::Compute)),
        addresses: super::tune::batch_cs() && batched,
        batched: super::tune::batch_cs() && batched,
        write_marks: marks_wanted(),
        marks_untouched: !super::tune::mark_touch(),
        touch_test: touch_test_wanted(),
        mark_twice: marks_wanted() && super::tune::tr_ballast() & 1 != 0,
        touch_twice: marks_wanted() && super::tune::tr_ballast() & 4 != 0,
        part_twice: marks_wanted() && super::tune::tr_ballast() & 8 != 0,
        dep_loads: super::tune::tr_ballast() & 2 != 0,
        marks_part: marks_part_wanted(),
        marks_held: marks_held_wanted(),
        sgpr_ubo: super::tune::sgpr_ubo(),
        gpu_loads: 0,
        gpu_res: 0,
        pt_linear: false,
        outlier: None,
    };
    apply_modes(&mut spec, code_hash);
    spec
}
/// R2 (`tune r2`): what `prepare` looks up per plan (its analysis, the
/// reader's sizes, the format, store-format, 3D-or-2D and compared-image
/// masks, the latest translation), kept per thread in one direct-mapped
/// record (seven map lookups a stage before). Every field is fixed per plan
/// key except the latest translation and the sizes, which only this
/// thread's preparations change; a record evicted by another plan is made
/// again from the maps.
#[derive(Default)]
struct PlanRec {
    key: u64,
    plan: Option<std::rc::Rc<Arc<Plan>>>,
    hints: [u32; 3],
    hints_known: bool,
    fmt: Option<u64>,
    dim3: Option<u64>,
    /// `d2_resources` (2D-addressed, 2D-array-addressed).
    dim2: Option<(u64, u64)>,
    cmp: Option<u64>,
    last_tr: Option<Arc<Local<Translated>>>,
    /// The stage template number `last_tr`'s spec came from (0: unknown).
    last_tmpl: u64,
}

const PLAN_REC_SLOTS: usize = 2048;

thread_local! {
    static PLAN_RECS: std::cell::RefCell<Vec<PlanRec>> = std::cell::RefCell::new(Vec::new());
}

fn plan_rec_slot(key: u64) -> usize {
    ((key ^ (key >> 23) ^ (key >> 41)) as usize) & (PLAN_REC_SLOTS - 1)
}

/// A plan's record taken out of its slot for one preparation and put back
/// when dropped (early returns included).
struct PlanRecGuard(Option<PlanRec>);

impl PlanRecGuard {
    fn take(on: bool, key: u64) -> PlanRecGuard {
        if !on {
            return PlanRecGuard(None);
        }
        let r = PLAN_RECS.with(|t| {
            let mut t = t.borrow_mut();
            if t.is_empty() {
                t.resize_with(PLAN_REC_SLOTS, PlanRec::default);
            }
            let slot = &mut t[plan_rec_slot(key)];
            if slot.key == key && slot.plan.is_some() {
                std::mem::take(slot)
            } else {
                PlanRec { key, ..PlanRec::default() }
            }
        });
        PlanRecGuard(Some(r))
    }

    fn get(&self) -> Option<&PlanRec> {
        self.0.as_ref()
    }

    fn get_mut(&mut self) -> Option<&mut PlanRec> {
        self.0.as_mut()
    }
}

impl Drop for PlanRecGuard {
    fn drop(&mut self) {
        if let Some(r) = self.0.take() {
            if r.plan.is_some() {
                PLAN_RECS.with(|t| {
                    if let Ok(mut t) = t.try_borrow_mut() {
                        if !t.is_empty() {
                            let k = plan_rec_slot(r.key);
                            t[k] = r;
                        }
                    }
                });
            }
        }
    }
}

/// `tmpl`: R2's number of the stage template `spec` comes from (0: none),
/// which the plan's latest translation is checked against (`PlanRec`).
fn prepare(addr: u64, insts: Arc<Local<Vec<rdna::Inst>>>, code_hash: u64, user: &[u32], spec: &mut Spec, tmpl: u64) -> Result<Prepared, String> {
    // Diagnostics (DS_PROFILE_GPU with DS_PROFILE_REPEATS=1: hashing every
    // word read and a global set per preparation inflate the prepare times
    // the profile reports): the inputs that would key a cached preparation.
    static REPEAT_STATS: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    let repeat_stats = *REPEAT_STATS.get_or_init(|| std::env::var("DS_PROFILE_REPEATS").ok().as_deref() == Some("1"));
    let repeat_key = (repeat_stats && super::exec::profiling_on()).then(|| {
        let mut h = super::fx::FxHasher::default();
        h.write_u64(addr);
        h.write_u64(code_hash);
        for &w in &user[..(spec.user_sgprs as usize).min(user.len())] {
            h.write_u32(w);
        }
        spec.hash(&mut h);
        h.finish()
    });
    // The plan depends only on the program and the user-data layout (data
    // loads read by the CPU).
    let plan_key = code_hash ^ (spec.user_sgprs as u64) << 48 ^ (spec.user_base as u64) << 56;
    // R2: this plan's per-thread record (`PlanRec`), put back on return.
    let mut rec = PlanRecGuard::take(super::tune::r2(), plan_key);
    let cached = rec.get().and_then(|r| r.plan.clone()).or_else(|| LOCAL_PLANS.with(|c| c.borrow().get(&plan_key).cloned())).or_else(|| {
        let p = std::rc::Rc::new(PLANS.lock().as_ref().and_then(|m| m.get(&plan_key).cloned())?);
        LOCAL_PLANS.with(|c| c.borrow_mut().insert(plan_key, p.clone()));
        Some(p)
    });
    let plan = match cached {
        Some(p) => p,
        None => {
            let t0 = std::time::Instant::now();
            let p = Arc::new(rdna::analysis::analyze_opts(&insts, spec.user_sgprs, spec.user_base, false));
            let ms = t0.elapsed().as_millis();
            if ms >= 100 {
                crate::info!("{:?} shader {addr:#x}: analysis took {ms} ms ({} instructions)", spec.stage, insts.len());
            }
            PLANS.lock().get_or_insert_with(HashMap::new).insert(plan_key, p.clone());
            let p = std::rc::Rc::new(p);
            LOCAL_PLANS.with(|c| c.borrow_mut().insert(plan_key, p.clone()));
            p
        }
    };
    if let Some(r) = rec.get_mut() {
        if r.plan.is_none() {
            r.plan = Some(plan.clone());
        }
    }
    if !plan.errors.is_empty() {
        return Err(format!("{:?} shader {addr:#x}: {}", spec.stage, plan.errors.join("; ")));
    }
    prep_lap(3);
    prep_lap(4);
    let map_gen = crate::mem::MAP_GEN.load(std::sync::atomic::Ordering::Acquire);
    let images_gen = super::images::GPU_IMAGES_GEN.load(std::sync::atomic::Ordering::Acquire);
    let sizes = match rec.get() {
        Some(r) if r.hints_known => r.hints,
        _ => READ_HINTS.with(|h| h.borrow().get(&plan_key).copied().unwrap_or([0; 3])),
    };
    let mut rd = Reader::with_sizes(addr, sizes);
    if rec.get().is_some() {
        rd.load_spans();
    }
    let mut resources = pool::resolved(plan.resources.len());
    let mut res_addrs = pool::addrs(plan.resources.len());
    spec.resources.clear();
    spec.resources.reserve(plan.resources.len());
    let fmt_mask = if plan.resources.iter().any(|r| r.kind == ResKind::Buffer) {
        match rec.get_mut() {
            Some(r) => *r.fmt.get_or_insert_with(|| LOCAL_FORMATS.with(|c| *c.borrow_mut().entry(plan_key).or_insert_with(|| shared_scan(1, plan_key, || format_resources(&insts, &plan))))),
            None => LOCAL_FORMATS.with(|c| *c.borrow_mut().entry(plan_key).or_insert_with(|| shared_scan(1, plan_key, || format_resources(&insts, &plan)))),
        }
    } else {
        0
    };
    // Buffers specialized to their one format (a bit per resource):
    // `fmt_run_time` while their translation is missing.
    let mut specialized = 0u64;
    for (i, res) in plan.resources.iter().enumerate() {
        let mut w = [0u32; 8];
        let src = match res.dwords.first() {
            Some(Prov::Mem(ptr, off)) => rd.chain(plan.chain_ids.get(plan.loads.len() + i).copied().unwrap_or(u32::MAX), ptr, user).map(|b| b + *off as u64),
            _ => None,
        };
        res_addrs.push(src);
        // A descriptor in consecutive words: one block read.
        let n = res.dwords.len().min(8);
        let block = plan.blocks.get(i).copied().unwrap_or(false) && src.is_some_and(|b| rd.read_into(b, &mut w[..n]));
        if !block {
            for (k, p) in res.dwords.iter().enumerate() {
                w[k] = rd.eval(p, user).ok_or_else(|| format!("{:?} shader {addr:#x}: resource {i} dword {k} unresolvable ({p:?})", spec.stage))?;
            }
        }
        match res.kind {
            ResKind::Buffer => {
                let v = Vsharp::decode(&w);
                // Raw accesses do not read the format and swizzle (see
                // `format_resources`).
                // (Compute stages: read from the V# when the program runs
                // where the decoder can; +4.7% at four walk places,
                // FINDINGS.md 2026-11-02 late afternoon. Stored buffers too
                // since 2026-11-18: `rdna::translate::FORMAT_RT` packs
                // stores.)
                let (format, dst_sel) = if fmt_mask & (1 << i.min(63)) == 0 {
                    (0, 0)
                } else if spec.stage == Stage::Compute && translate::rt_format_entry(v.format as u16) != 0 {
                    if fmt_census_on() {
                        fmt_census_note(code_hash, addr, i, v.format as u16, v.dst_sel as u16);
                    }
                    // (Specialized while its program and resource see one
                    // format: GPU busy -1.5% at the vista and -1.9% at Doran,
                    // frames +0.6-0.7%, FINDINGS.md 2026-11-06.)
                    if single_format(code_hash, i, v.format, v.dst_sel) {
                        if i < 64 {
                            specialized |= 1 << i;
                        }
                        (v.format, v.dst_sel)
                    } else {
                        (translate::FORMAT_RT, 0)
                    }
                } else {
                    (v.format, v.dst_sel)
                };
                spec.resources.push(ResSpec::Buffer { format, dst_sel });
                resources.push(Resolved::Buffer(v));
            }
            ResKind::Global => {
                spec.resources.push(ResSpec::Global);
                resources.push(Resolved::Global(w[0] as u64 | ((w[1] as u64) << 32), res.extent.map(|e| e as u64)));
            }
            ResKind::Image => {
                let t = Tsharp::decode(&w);
                let dim = ImageDim::from_type(t.ty, t.depth).ok_or_else(|| format!("{:?} shader {addr:#x}: image resource {i} has type {}", spec.stage, t.ty))?;
                // (Compute: addressed as 3D by its instructions, sampled, a
                // 2D or 3D texture: one translation for both, FINDINGS.md
                // 2026-11-02 evening.)
                let dim = if spec.stage == Stage::Compute
                    && matches!(t.ty, 9 | 10)
                    && !res.written
                    && match rec.get_mut() {
                        Some(r) => *r.dim3.get_or_insert_with(|| LOCAL_DIM3.with(|c| *c.borrow_mut().entry(plan_key).or_insert_with(|| d3_resources(&insts, &plan)))),
                        None => LOCAL_DIM3.with(|c| *c.borrow_mut().entry(plan_key).or_insert_with(|| d3_resources(&insts, &plan))),
                    } & (1 << i.min(63))
                        != 0
                {
                    ImageDim::D3OrD2
                } else {
                    dim
                };
                // (Compute: a sampled or loaded 2D or 2D-array texture keyed by the dimension
                // its instructions address, `d2_resources`; the binding's view follows the key.)
                let dim = if spec.stage == Stage::Compute && matches!(dim, ImageDim::D2 | ImageDim::D2Array) && !res.written && dim_by_instructions() {
                    let (only2d, only2da) = match rec.get_mut() {
                        Some(r) => *r.dim2.get_or_insert_with(|| LOCAL_DIM2.with(|c| *c.borrow_mut().entry(plan_key).or_insert_with(|| d2_resources(&insts, &plan)))),
                        None => LOCAL_DIM2.with(|c| *c.borrow_mut().entry(plan_key).or_insert_with(|| d2_resources(&insts, &plan))),
                    };
                    let bit = 1u64 << i.min(63);
                    if only2da & bit != 0 {
                        ImageDim::D2Array
                    } else if only2d & bit != 0 {
                        ImageDim::D2
                    } else {
                        dim
                    }
                } else {
                    dim
                };
                super::lodstats::note(&t.raw);
                // `depth` is decided below for all images at once.
                spec.resources.push(ResSpec::Image { dim, class: num_class(t.format), depth: true });
                resources.push(Resolved::Image(t));
            }
            ResKind::Sampler => {
                spec.resources.push(ResSpec::Sampler);
                resources.push(Resolved::Sampler([w[0], w[1], w[2], w[3]]));
            }
            ResKind::ImageTable => {
                // Entries of 32 bytes (T#s) from the table's base: a V#'s
                // range, or from a pointer (a T#-sized s_load at a run-time
                // offset) as many as the analysis saw loaded at constant
                // offsets, else up to the first entry that is no usable T#
                // (at most 32: the title's waterfall loops select entries by
                // a 32-bit mask).
                let len = rdna::translate::IMAGE_TABLE_LEN as u64;
                let (tbase, n, stop) = if res.dwords.len() == 2 {
                    let b = w[0] as u64 | ((w[1] as u64) << 32);
                    match res.extent {
                        Some(e) => (b, (e as u64 / 32).min(len), false),
                        None => (b, 32.min(len), true),
                    }
                } else {
                    let v = Vsharp::decode(&w);
                    (v.base, (v.extent() / 32).min(len), false)
                };
                let mut entries = Vec::with_capacity(n as usize);
                let mut first: Option<(ImageDim, u16)> = None;
                for k in 0..n {
                    let a = tbase + 32 * k;
                    let mut d = [0u32; 8];
                    let ok = rd.read_into(a, &mut d);
                    let t = Tsharp::decode(&d);
                    if ok {
                        super::lodstats::note(&t.raw);
                    }
                    let dim = ImageDim::from_type(t.ty, t.depth);
                    let usable = ok && t.base != 0 && super::images::vk_format(t.format).is_some() && dim.is_some() && first.is_none_or(|(fd, _)| Some(fd) == dim);
                    if !usable && stop {
                        break;
                    }
                    if usable && first.is_none() {
                        first = Some((dim.unwrap(), t.format));
                    }
                    entries.push(usable.then_some(t));
                }
                let (dim, format) = first.ok_or_else(|| format!("{:?} shader {addr:#x}: image table {i} ({tbase:#x}, {n} entries) has no usable T#", spec.stage))?;
                spec.resources.push(ResSpec::Image { dim, class: num_class(format), depth: false });
                resources.push(Resolved::ImageTable(entries));
            }
        }
    }
    prep_lap(5);
    let res_reads = rd.reads.len();
    // Depth views keep the hardware comparison, color views emulate it: the
    // image cache decides, for the images comparison instructions sample
    // (the others' translation does not depend on it).
    let mut cmp_images = 0u64;
    let mut cmp_depth = 0u64;
    if super::tune::cmp_emulate() && resources.iter().any(|r| matches!(r, Resolved::Image(_))) {
        cmp_images = match rec.get_mut() {
            Some(r) => *r.cmp.get_or_insert_with(|| compare_images(plan_key, &insts, &plan)),
            None => compare_images(plan_key, &insts, &plan),
        };
        for (r, s) in resources.iter().zip(spec.resources.iter_mut()) {
            if let (Resolved::Image(_), ResSpec::Image { depth, .. }) = (r, s) {
                *depth = false;
            }
        }
        if cmp_images != 0 {
            super::images::with(|im| {
                for (i, (r, s)) in resources.iter().zip(spec.resources.iter_mut()).enumerate() {
                    if let (Resolved::Image(t), ResSpec::Image { depth, .. }) = (r, s) {
                        if i < 64 && cmp_images & (1 << i) != 0 {
                            *depth = im.depth_view(t);
                            if *depth {
                                cmp_depth |= 1 << i;
                            }
                        }
                    }
                }
            });
        }
    }
    prep_lap(6);
    // Translations do not depend on the program's address (see
    // translate::code_base_offset): copies of a program share one, and its
    // pipeline (the Vulkan pipeline cache finds it again in the next run).
    static BY_ADDRESS: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    let by_address = *BY_ADDRESS.get_or_init(|| std::env::var_os("DS_WAVEMASK").is_some() || std::env::var_os("DS_LOCKSTEP_OFF").is_some());
    let base = spec.code_base;
    if !by_address {
        spec.code_base = 0;
    }
    // (Under R2 a spec from the template the latest translation's spec came
    // from equals it but for its resources: those alone are compared.)
    let last_hit = match rec.get() {
        Some(r) => r
            .last_tr
            .as_ref()
            .filter(|t| {
                if tmpl != 0 && r.last_tmpl == tmpl {
                    let same = t.spec.resources == spec.resources;
                    if r2_check() {
                        R2_CHECKS[0].fetch_add(1, std::sync::atomic::Ordering::Relaxed);
                        if same != (t.spec == *spec) && R2_CHECKS[1].fetch_add(1, std::sync::atomic::Ordering::Relaxed) < 40 {
                            crate::info!("r2 check: {:?} {addr:#x}: the translation found by the template's number ({same}) differs from the full comparison", spec.stage);
                        }
                    }
                    same
                } else {
                    t.spec == *spec
                }
            })
            .cloned(),
        None => LAST_TRANSLATIONS.with(|c| c.borrow().get(&plan_key).filter(|t| t.spec == *spec).cloned()),
    };
    let from_last = last_hit.is_some();
    let (fast, fast_hit) = match last_hit {
        Some(t) => (0, Some(t)),
        None => {
            let fast = {
                let mut h = super::fx::FxHasher::default();
                spec.hash(&mut h);
                h.finish()
            };
            (fast, FAST_TRANSLATIONS.with(|c| c.borrow().get(&(plan_key, fast)).filter(|t| t.spec == *spec).cloned()))
        }
    };
    let key_spec = if fast_hit.is_none() { Some(spec.clone()) } else { None };
    let spec_hash = match &key_spec {
        Some(ks) => {
            let mut h = std::collections::hash_map::DefaultHasher::new();
            ks.hash(&mut h);
            h.finish()
        }
        None => 0,
    };
    spec.code_base = base;
    let tr = if let Some(t) = fast_hit {
        t
    } else {
        // (A specialized key being made in the background is missing until
        // its pipeline is made too.)
        let pending = specialized != 0 && fmt_pending(plan_key, spec_hash);
        let cached = if pending {
            None
        } else {
            LOCAL_TRANSLATIONS.with(|c| c.borrow().get(&(plan_key, spec_hash)).cloned()).or_else(|| {
                let t = TRANSLATIONS.lock().as_ref().and_then(|m| m.get(&(plan_key, spec_hash)).cloned())?;
                LOCAL_TRANSLATIONS.with(|c| c.borrow_mut().insert((plan_key, spec_hash), t.clone()));
                Some(t)
            })
        };
        // A first use of a specialized key: the run-time variant runs while
        // the specialized translation and pipeline are made in the
        // background, when its own pipeline is made (instead of the command
        // processor stopping for them: 370-570 ms per particle program,
        // FINDINGS.md 2026-11-06 morning).
        let run_time = match (&cached, &key_spec) {
            (None, Some(ks)) if specialized != 0 && spec.stage == Stage::Compute && !fmt_failed(plan_key, spec_hash) => fmt_run_time(ks, specialized, plan_key, addr, &insts, &plan, spec.code_base),
            _ => None,
        };
        if let (Some(_), Some(ks)) = (&run_time, &key_spec) {
            FMT_STATS[0].fetch_add(1, std::sync::atomic::Ordering::Relaxed);
            if !pending {
                let queued = fmt_later(FmtJob { addr, insts: insts.0.clone(), plan: (*plan).clone(), plan_key, spec_hash, spec: spec.clone(), key_spec: ks.clone() });
                if queued && super::tune::precompile() > 0 {
                    super::precompile::first_translation(addr, &insts, &plan, code_hash, ks);
                }
            }
        } else if cached.is_none() && specialized != 0 && spec.stage == Stage::Compute {
            FMT_STATS[5].fetch_add(1, std::sync::atomic::Ordering::Relaxed);
        }
        let t = match (cached, run_time) {
            (Some(t), _) | (None, Some(t)) => t,
            (None, None) => {
                if spec.stage == Stage::Compute && super::tune::precompile() > 0 {
                    if let Some(ks) = &key_spec {
                        super::precompile::first_translation(addr, &insts, &plan, code_hash, ks);
                    }
                }
                let t = translate_kept(addr, &insts, &plan, plan_key, spec_hash, spec, key_spec.clone().unwrap_or_else(|| spec.clone()))?;
                LOCAL_TRANSLATIONS.with(|c| c.borrow_mut().insert((plan_key, spec_hash), t.clone()));
                t
            }
        };
        // This thread's handle on it.
        let t = Arc::new(Local(t));
        if key_spec.as_ref().is_some_and(|ks| *ks == t.spec) {
            FAST_TRANSLATIONS.with(|c| c.borrow_mut().insert((plan_key, fast), t.clone()));
        }
        t
    };
    if let Some(r) = rec.get_mut() {
        // (Whole specs compared equal: the template's number holds for it.)
        r.last_tmpl = tmpl;
    }
    if !from_last {
        match rec.get_mut() {
            Some(r) => r.last_tr = Some(tr.clone()),
            None => {
                LAST_TRANSLATIONS.with(|c| c.borrow_mut().insert(plan_key, tr.clone()));
            }
        }
    }
    prep_lap(7);
    // Scalar-load data, resolved now.
    let mut sgpr_data = pool::words((translate::sgpr_data_dwords(&tr.plan) as usize).max(translate::SGPR_DATA_USER as usize));
    sgpr_data.extend_from_slice(user);
    sgpr_data.resize(translate::SGPR_DATA_USER as usize, 0);
    let mut data_loads = pool::loads(tr.plan.load_data.iter().filter(|&&d| d).count());
    for (j, l) in tr.plan.loads.iter().enumerate() {
        let base = rd.chain(tr.plan.chain_ids.get(j).copied().unwrap_or(u32::MAX), &l.ptr, user);
        let start = sgpr_data.len();
        sgpr_data.resize(start + l.dwords as usize, 0);
        let Some(b) = base else { continue };
        let word = rd.words.len();
        if !rd.read_into(b + l.offset as u64, &mut sgpr_data[start..]) {
            for k in 0..l.dwords as usize {
                sgpr_data[start + k] = rd.u32(b + l.offset as u64 + 4 * k as u64).unwrap_or(0);
            }
        }
        if tr.plan.load_data.get(j).copied().unwrap_or(false) && rd.words.len() == word + l.dwords as usize {
            data_loads.push(DataLoad { sgpr: start as u32, addr: b + l.offset as u64, dwords: l.dwords, word: word as u32 });
        }
    }
    sgpr_data.resize(translate::sgpr_data_dwords(&tr.plan) as usize, 0);
    let cb = translate::code_base_offset(&tr.plan) as usize;
    sgpr_data[cb] = addr as u32;
    sgpr_data[cb + 1] = (addr >> 32) as u32;
    // Resource info blocks are filled when binding; until then no index
    // range check applies.
    let ri = translate::res_info_offset(&tr.plan) as usize;
    for k in 0..tr.plan.resources.len() {
        sgpr_data[ri + translate::RES_INFO_DWORDS as usize * k + 3] = u32::MAX;
    }
    // LOD information of the sampled images (hardware LOD statistics).
    for (k, r) in resources.iter().enumerate() {
        match r {
            Resolved::Image(t) => sgpr_data[ri + translate::RES_INFO_DWORDS as usize * k] = super::lodstats::info(t),
            Resolved::ImageTable(entries) => {
                let at = (translate::lod_info_offset(&tr.plan) + translate::IMAGE_TABLE_LEN * translate::table_slot(&tr.plan, k)) as usize;
                for (j, t) in entries.iter().enumerate().take(translate::IMAGE_TABLE_LEN as usize) {
                    sgpr_data[at + j] = t.as_ref().map(super::lodstats::info).unwrap_or(0);
                }
            }
            _ => {}
        }
    }
    prep_lap(8);
    prep_lap_volume(rd.words.len(), rd.reads.len(), rd.ptrs.len(), resources.len());
    if super::slotcensus::on() {
        super::slotcensus::note_reads(&rd.reads);
    }
    if armed_census_on() {
        armed_census(spec.stage, &rd.reads, res_reads);
    }
    // The reader's references into the translation's plan end here.
    let used = [rd.reads.len() as u32, rd.words.len() as u32, rd.ptrs.len() as u32];
    match rec.get_mut() {
        Some(r) => {
            r.hints = used;
            r.hints_known = true;
        }
        None if used != sizes => {
            READ_HINTS.with(|h| h.borrow_mut().insert(plan_key, used));
        }
        None => {}
    }
    if rd.span_gen != u64::MAX {
        rd.save_spans();
    }
    let Reader { reads, words, unread, chain_vals, .. } = rd;
    pool::put_addrs(chain_vals);
    let prepared = Prepared {
        stage: spec.stage,
        addr,
        found: Arc::new(Found { insts, tr, resources, res_addrs, reads, words, unread, data_loads }),
        sgpr_data,
        user: UserData::new(user),
        map_gen,
        images_gen,
        cmp_images,
        cmp_depth,
        code_hash,
        ahead: 1,
    };
    if let Some(k) = repeat_key {
        let tr = &prepared.tr;
        let bytes = note_repeat(spec.stage, k, &prepared.reads);
        let deps = user_deps(&tr.plan);
        let mut h = super::fx::FxHasher::default();
        h.write_u64(addr);
        h.write_u64(code_hash);
        for (i, &w) in user.iter().enumerate().take(32) {
            if deps & (1 << i) != 0 {
                h.write_u32(w);
            }
        }
        spec.hash(&mut h);
        note_repeat_deps(spec.stage, h.finish(), bytes);
    }
    prep_lap(9);
    prep_lap_end();
    Ok(prepared)
}
