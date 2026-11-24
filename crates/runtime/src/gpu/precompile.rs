//! Compute pipelines made when the title creates their shader objects
//! (`tune precompile`).
//!
//! The title creates a shader object (`sceAgcCreateShader`) for each program
//! of a shader file it loads (.csdr: the engine's at boot, a level's materials
//! and particle effects as they load), well before the programs first run,
//! and many more than run (5,646 compute objects against ~340 programs
//! dispatched in a first session to the courtyard). A first dispatch
//! otherwise compiles its pipeline on the command processor (0.2-2.9 s for
//! the large programs; `gpu::compile` scans ahead from there), in every
//! first session and again whenever the translations change (a new build).
//!
//! A program's translation spec comes from its dispatch: the registers the
//! object carries (RSRC2, the thread-group size; wave64 follows from the
//! code) and the descriptors bound then (image dimensions and number
//! classes, typed buffers' formats), which no object holds. So every first
//! translation of a compute program records its spec by code hash
//! (`learn`), in a file that outlives the pipeline store (`DS_SPECS`, else
//! local-data/shader-specs.txt). With `precompile` 2 (the default) a created object
//! whose program was seen before has its pipelines made from those specs on
//! `tune precompile_workers` threads at the lowest priority, under the keys
//! its dispatches look up (`shaders::translate_for_spec`,
//! `exec::precompile_translation`); 3 also makes the guess from the object
//! alone (`shaders::guess_compute_spec`; equal to 86% of a first session's
//! first translations, 2026-11-02 night, and most objects never run). With 1, and above, each
//! first translation is also compared with that guess (status line).
//!
//! Graphics pipelines the same way: each one created records both stages'
//! specs and its state (`learn_gfx`; target and depth formats, blend, write
//! masks, primitive class, depth clamp), and once the title has created
//! objects for both of a learned pipeline's programs it is made
//! (`exec::precompile_gfx_translations`).
//!
//! The learned programs' code is kept too (`<store>.code`: code hash, word
//! count, words; local data like the pipeline store), so every learned
//! pipeline whose programs' code is there is made from the start of a
//! session (`early`), before the title loads them: a first session's ~500
//! compute pipelines take ~330 s of compiles, more than the boot leaves if
//! they start only when the objects are created.

use parking_lot::{Condvar, Mutex};
use rdna::translate::{ImageDim, MrtSpec, NumClass, PsInput, ResSpec, Spec};
use std::collections::HashMap;
use std::io::Write;
use std::sync::atomic::{AtomicU64, Ordering};

/// A compute shader object's registers: RSRC2, the thread-group size and
/// the dispatch modifier (`sceAgcCreateShader`).
#[derive(Clone, Copy)]
struct Created {
    rsrc2: u32,
    local: [u32; 3],
    modifier: u32,
}

/// Compute programs by address, as their shader objects describe them.
static CREATED: Mutex<Option<HashMap<u64, Created>>> = Mutex::new(None);
/// Programs whose first translation was compared (addresses).
static COMPARED: Mutex<Option<std::collections::HashSet<u64>>> = Mutex::new(None);

/// What a dispatch adds to a program's spec: RSRC2, the thread-group size,
/// wave64, the resources' specs and whether it ran batched (runs of
/// consecutive dispatches, `shaders::batchable`: its translation differs).
#[derive(Clone, PartialEq, Eq, Hash)]
struct Seen {
    rsrc2: u32,
    local: [u32; 3],
    wave64: bool,
    resources: Vec<ResSpec>,
    batched: bool,
}

/// The specs dispatches used, by code hash, with the program's rank (the
/// order in which programs were first dispatched, loaded from the store,
/// then appended to it), and the store's path.
static LEARNED: Mutex<Option<HashMap<u64, (u32, Vec<Seen>)>>> = Mutex::new(None);
static STORE: std::sync::OnceLock<std::path::PathBuf> = std::sync::OnceLock::new();

/// A stage of a learned graphics pipeline: what its draws' registers and
/// descriptors put in its spec (`shaders::vertex_spec`, `pixel_spec`).
#[derive(Clone, PartialEq, Eq, Hash)]
struct StageSeen {
    user_sgprs: u16,
    wave64: bool,
    base_instance: Option<u16>,
    ps_input_addr: u32,
    ps_inputs: Vec<PsInput>,
    mrt: [MrtSpec; 8],
    early_tests: bool,
    early_ok: bool,
    /// A pixel stage of a draw measuring LOD statistics where they are
    /// decided per draw (`shaders::lod_per_draw`: its depth tests stay after
    /// the shader and it keeps them; the other draws' variant has none).
    lod_measure: bool,
    resources: Vec<ResSpec>,
}

impl StageSeen {
    fn of(s: &Spec) -> StageSeen {
        let lod_measure = s.stage == rdna::translate::Stage::Fragment && s.lod_stats && !s.batched && !s.early_tests && !s.early_ok && super::shaders::lod_per_draw();
        StageSeen { user_sgprs: s.user_sgprs, wave64: s.wave64, base_instance: s.base_instance_sgpr, ps_input_addr: s.ps_input_addr, ps_inputs: s.ps_inputs.clone(), mrt: s.mrt, early_tests: s.early_tests, early_ok: s.early_ok, lod_measure, resources: s.resources.clone() }
    }
}

/// A learned graphics pipeline: its programs' code hashes (pixel 0: none),
/// their stages and the state.
#[derive(Clone, PartialEq, Eq, Hash)]
struct GfxRecipe {
    vs: u64,
    vs_seen: StageSeen,
    ps: u64,
    ps_seen: Option<StageSeen>,
    state: super::exec::GfxState,
}

/// Learned graphics pipelines with their ranks, by program hash, and those
/// queued already.
#[derive(Default)]
struct GfxLearned {
    recipes: Vec<(u32, GfxRecipe)>,
    set: std::collections::HashSet<GfxRecipe>,
    by_hash: HashMap<u64, Vec<usize>>,
    queued: std::collections::HashSet<usize>,
}

static GFX: Mutex<Option<GfxLearned>> = Mutex::new(None);
/// Learned programs' code by code hash (`<store>.code`).
static CODES: Mutex<Option<HashMap<u64, std::sync::Arc<Vec<u32>>>>> = Mutex::new(None);
/// Compute programs whose learned specs were queued (by code hash): made
/// once per session, early or when their object was created.
static QUEUED_HASHES: Mutex<Option<std::collections::HashSet<u64>>> = Mutex::new(None);
/// Store lines read or written so far (ranks of new entries).
static LINES: AtomicU64 = AtomicU64::new(0);
/// Programs of created objects by code hash (their addresses).
static HASH_ADDR: Mutex<Option<HashMap<u64, u64>>> = Mutex::new(None);

/// [shader objects, first translations, with an object, guesses equal,
/// equal but for buffers of unknown format, then the first field that
/// differed: wave64, thread-group size, RSRC2, user data, resource count,
/// a buffer, an image's dimension, class or depth view, another resource,
/// another field; translations after the first; specs learned, specs
/// loaded].
static STATS: [AtomicU64; 21] = [const { AtomicU64::new(0) }; 21];
const OBJECTS: usize = 0;
const FIRSTS: usize = 1;
const WITH_OBJECT: usize = 2;
const EQUAL: usize = 3;
const UNKNOWN_FORMAT: usize = 4;
const DIFF: usize = 5;
const DIFF_NAMES: [&str; 10] = ["wave64", "thread-group size", "RSRC2", "user data", "resource count", "a buffer", "an image's dimension", "an image's class", "an image's depth view", "another field"];
const LATER: usize = 15;
const LEARNT: usize = 16;
const LOADED: usize = 17;
/// Programs' code loaded, kept this session, pipelines queued early.
const CODES_LOADED: usize = 18;
const CODES_KEPT: usize = 19;
const EARLY: usize = 20;
/// Dispatch modifier bit 15 against the wave64 of first translations:
/// [set and wave64, set and wave32] (the rest: clear).
static MODIFIER: [AtomicU64; 4] = [const { AtomicU64::new(0) }; 4];

/// Translations the dispatch path made itself for programs with learned
/// specs (`first_translation`): [any, one of whose learned specs gives the
/// same key spec (precompile not done with it yet, or it failed), none
/// does (the key spec differs)].
static MISSES: [AtomicU64; 3] = [const { AtomicU64::new(0) }; 3];

/// The fields in which two specs' Debug forms differ (diagnostics).
fn spec_diff(a: &Spec, b: &Spec) -> String {
    let (fa, fb) = (format!("{a:?}"), format!("{b:?}"));
    // Fields at top level: split on ", " outside brackets.
    let split = |s: &str| -> Vec<String> {
        let (mut out, mut cur, mut depth) = (Vec::new(), String::new(), 0i32);
        let body = s.trim_start_matches("Spec {").trim_end_matches('}');
        let chars: Vec<char> = body.chars().collect();
        let mut i = 0;
        while i < chars.len() {
            let c = chars[i];
            match c {
                '(' | '[' | '{' => depth += 1,
                ')' | ']' | '}' => depth -= 1,
                _ => {}
            }
            if c == ',' && depth == 0 {
                out.push(cur.trim().to_string());
                cur.clear();
            } else {
                cur.push(c);
            }
            i += 1;
        }
        if !cur.trim().is_empty() {
            out.push(cur.trim().to_string());
        }
        out
    };
    let (xa, xb) = (split(&fa), split(&fb));
    let diffs: Vec<String> = xa.iter().zip(xb.iter()).filter(|(x, y)| x != y).map(|(x, y)| format!("{x} | {}", y.split_once(':').map_or(y.as_str(), |(_, v)| v.trim()))).collect();
    if xa.len() != xb.len() {
        return format!("field counts {} and {}", xa.len(), xb.len());
    }
    diffs.join("; ")
}

/// Where learned specs are kept (`main`: DS_SPECS, else under local-data).
pub fn set_store(path: std::path::PathBuf) {
    if let Some(dir) = path.parent() {
        let _ = std::fs::create_dir_all(dir);
    }
    super::pipeuse::set_path(&path);
    let _ = STORE.set(path);
}

/// Reads the store once (compute specs and graphics pipelines; a line's
/// index is its rank).
fn load() {
    static DONE_LOADING: std::sync::Once = std::sync::Once::new();
    DONE_LOADING.call_once(|| {
        let mut m: HashMap<u64, (u32, Vec<Seen>)> = HashMap::new();
        let mut g = GfxLearned::default();
        let mut exact: HashMap<(u64, u64), Spec> = HashMap::new();
        // (Read as text even with bytes that are not: a write cut by a power loss; such lines are
        // passed, the rest read.)
        if let Some(text) = STORE.get().and_then(|p| std::fs::read(p).ok()).map(|b| String::from_utf8_lossy(&b).into_owned()) {
            for (k, line) in text.lines().enumerate() {
                let rank = k as u32;
                if let Some(r) = line.strip_prefix("G ").and_then(parse_gfx) {
                    if g.set.insert(r.clone()) {
                        add_recipe(&mut g, rank, r);
                        STATS[LOADED].fetch_add(1, Ordering::Relaxed);
                    }
                } else if let Some(rest) = line.strip_prefix("K ") {
                    // (A compute pipeline's exact key spec, `note_compute_pipeline`: taken whole only
                    // (its checksum: a line cut short by a power loss could read as another spec),
                    // for this build's preparation rules (others' key specs are not this one's: a
                    // change of how key specs are formed changes the rules, and the whole
                    // preparation is made again), kept under the hash its words give; one
                    // recorded under another counted, a sign that the hasher changed.)
                    match parse_exact(rest) {
                        Some(Ok((h, sh, spec))) => {
                            let got = spec_hash_of(&spec);
                            if got != sh {
                                EXACT_STATS[2].fetch_add(1, Ordering::Relaxed);
                            }
                            if exact.insert((h, got), spec).is_none() {
                                EXACT_STATS[0].fetch_add(1, Ordering::Relaxed);
                            }
                        }
                        Some(Err(())) => {
                            EXACT_STATS[4].fetch_add(1, Ordering::Relaxed);
                        }
                        None => {
                            EXACT_STATS[3].fetch_add(1, Ordering::Relaxed);
                        }
                    }
                } else if let Some((h, s)) = parse(line).filter(|(_, s)| local_ok(s.local)) {
                    let v = &mut m.entry(h).or_insert_with(|| (rank, Vec::new())).1;
                    if !v.contains(&s) {
                        v.push(s);
                        STATS[LOADED].fetch_add(1, Ordering::Relaxed);
                    }
                }
                LINES.store(k as u64 + 1, Ordering::Relaxed);
            }
        }
        let mut codes: HashMap<u64, std::sync::Arc<Vec<u32>>> = HashMap::new();
        if let Some(b) = code_path().and_then(|p| std::fs::read(p).ok()) {
            let (passed, bad);
            (codes, passed, bad) = parse_codes(&b);
            if passed > 0 {
                crate::warn!("gpu: the learned programs' code file has a damaged part ({passed} bytes passed, {bad} records not matching their hash; a write cut short, e.g. by a power loss): the records after it read");
            }
        }
        STATS[CODES_LOADED].fetch_add(codes.len() as u64, Ordering::Relaxed);
        *LEARNED.lock() = Some(m);
        *GFX.lock() = Some(g);
        *CODES.lock() = Some(codes);
        *EXACT.lock() = Some(exact);
    });
}

/// The exact key specs of the compute pipelines sessions made (`note_compute_pipeline`), by (code
/// hash, spec hash): the store's `K` lines. [loaded, kept this session, loaded under another hash,
/// lines not read (cut short, damaged, unknown words), lines of other preparation rules].
static EXACT: Mutex<Option<HashMap<(u64, u64), Spec>>> = Mutex::new(None);
pub(crate) static EXACT_STATS: [AtomicU64; 5] = [const { AtomicU64::new(0) }; 5];

/// FNV-1a over `b` (a `K` line's checksum).
fn fnv64(b: &[u8]) -> u64 {
    b.iter().fold(0xcbf2_9ce4_8422_2325u64, |a, &x| (a ^ x as u64).wrapping_mul(0x100_0000_01b3))
}

/// A `K` line (without its `K `): `<code hash> <spec hash> r<rules> <spec words> <checksum>`.
/// Some(Ok): its code hash, spec hash and spec; Some(Err): whole, of other preparation rules;
/// None: not read.
fn parse_exact(rest: &str) -> Option<Result<(u64, u64, Spec), ()>> {
    let (body, sum) = rest.rsplit_once(' ')?;
    if u64::from_str_radix(sum, 16).ok()? != fnv64(body.as_bytes()) {
        return None;
    }
    let mut w = body.splitn(4, ' ');
    let h = u64::from_str_radix(w.next()?, 16).ok()?;
    let sh = u64::from_str_radix(w.next()?, 16).ok()?;
    let rules: u32 = w.next()?.strip_prefix('r')?.parse().ok()?;
    if rules != super::compbench::PREPARATION_RULES {
        return Some(Err(()));
    }
    Some(Ok((h, sh, Spec::from_words(w.next()?)?)))
}

/// A `K` line for `parse_exact`.
fn exact_line(h: u64, sh: u64, spec: &Spec) -> String {
    let body = format!("{h:016x} {sh:016x} r{} {}", super::compbench::PREPARATION_RULES, spec.to_words());
    format!("K {body} {:016x}\n", fnv64(body.as_bytes()))
}

/// A spec's hash as pipeline keys take it.
fn spec_hash_of(s: &Spec) -> u64 {
    use std::hash::{Hash, Hasher};
    let mut h = std::collections::hash_map::DefaultHasher::new();
    s.hash(&mut h);
    h.finish()
}

/// A compute pipeline this session made (`exec::compute_pipe_create`, at a dispatch, in the
/// background or ahead): its exact key spec kept once, with its program's code, so that the
/// compute step makes it again after a driver update, a version that translates differently or a
/// store the driver refuses (`exact_work`). Until 2026-11-21 only the learned specs were kept, and
/// they cannot name batched programs' address variants, LOD-statistics variants or every
/// buffer-format mix: a store the compute step made again left ~330 compute pipelines to the first
/// session. The compute step's own pipelines (no program address) are not kept. Only the check
/// and a queue's push here, on the creating thread (a command processor's, often): the code and
/// the store's line are written by a thread of their own (`exact_writer`; done inline at first,
/// the character creation's frames hitched 5 times as much and the self-play's menu presses
/// repeated).
pub(crate) fn note_compute_pipeline(addr: u64, tr: &super::shaders::Translated) {
    // (`DS_EXACT_SPECS=0`, a test: none kept.)
    static OFF: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    if *OFF.get_or_init(|| std::env::var("DS_EXACT_SPECS").ok().as_deref() == Some("0")) {
        return;
    }
    if addr == 0 || tr.spec.stage != rdna::translate::Stage::Compute || !local_ok(tr.spec.cs_local_size) {
        return;
    }
    // (Not a diagnosis's variants: probes, dumps, outlier counts, exact ballasts; the review of
    // 2026-11-21.)
    let d = &tr.spec;
    if d.probe.is_some() || d.dump.is_some() || d.outlier.is_some() || d.mark_twice || d.touch_twice || d.part_twice || d.pt_linear || d.dep_loads || d.marks_untouched {
        return;
    }
    load();
    let sh = spec_hash_of(&tr.spec);
    let plan_key = tr.key ^ sh.rotate_left(1);
    let h = plan_key ^ (tr.spec.user_sgprs as u64) << 48 ^ (tr.spec.user_base as u64) << 56;
    {
        let mut g = EXACT.lock();
        let m = g.get_or_insert_with(HashMap::new);
        if m.contains_key(&(h, sh)) {
            return;
        }
        m.insert((h, sh), tr.spec.clone());
    }
    EXACT_JOBS.lock().push_back((addr, h, sh, tr.spec.clone()));
    if !EXACT_WRITER.swap(true, Ordering::Relaxed) {
        let _ = std::thread::Builder::new().name("exact-specs".into()).spawn(exact_writer);
    }
    EXACT_CV.notify_one();
}

/// The exact specs waiting for their code and line (`note_compute_pipeline`).
static EXACT_JOBS: Mutex<std::collections::VecDeque<(u64, u64, u64, Spec)>> = Mutex::new(std::collections::VecDeque::new());
static EXACT_CV: Condvar = Condvar::new();
static EXACT_WRITER: std::sync::atomic::AtomicBool = std::sync::atomic::AtomicBool::new(false);

/// Keeps each queued spec's program code and writes their lines, those queued together in one
/// append, below normal priority.
fn exact_writer() {
    crate::log::set_thread_name("exact-specs");
    unsafe {
        crate::platform::os::Win32::System::Threading::SetThreadPriority(crate::platform::os::Win32::System::Threading::GetCurrentThread(), crate::platform::os::Win32::System::Threading::THREAD_PRIORITY_BELOW_NORMAL);
    }
    loop {
        let busy = EXACT_BUSY.lock();
        let jobs: Vec<(u64, u64, u64, Spec)> = {
            let mut g = EXACT_JOBS.lock();
            if g.is_empty() {
                drop(busy);
                while g.is_empty() {
                    EXACT_CV.wait(&mut g);
                }
                continue;
            }
            g.drain(..).collect()
        };
        exact_write(jobs);
        drop(busy);
    }
}

/// Held while queued specs are written (`exact_flush` waits for a batch in hand).
static EXACT_BUSY: Mutex<()> = Mutex::new(());

/// The queued specs' code kept and their lines written. A spec whose code could not be kept (its
/// program unmapped or replaced meanwhile) leaves the map, so that its next creation queues it
/// again (the review of 2026-11-21).
fn exact_write(jobs: Vec<(u64, u64, u64, Spec)>) {
    let mut lines = String::new();
    for (addr, h, sh, spec) in jobs {
        keep_code(addr, h);
        if CODES.lock().as_ref().is_some_and(|c| c.contains_key(&h)) {
            EXACT_STATS[1].fetch_add(1, Ordering::Relaxed);
            lines.push_str(&exact_line(h, sh, &spec));
        } else if let Some(m) = EXACT.lock().as_mut() {
            m.remove(&(h, sh));
        }
    }
    if !lines.is_empty() {
        store_line(&lines);
    }
}

/// At the session's end: the specs still queued written (the writer's batch in hand first, 2 s at
/// most: a writer held up does not hold the end).
pub(crate) fn exact_flush() {
    let Some(_busy) = EXACT_BUSY.try_lock_for(std::time::Duration::from_secs(2)) else { return };
    let jobs: Vec<(u64, u64, u64, Spec)> = EXACT_JOBS.lock().drain(..).collect();
    if !jobs.is_empty() {
        exact_write(jobs);
    }
}

/// The exact key specs sessions made whose program's code is kept (`--precompile-all`'s list).
pub(crate) fn exact_work() -> Vec<(u64, std::sync::Arc<Vec<u32>>, Spec)> {
    load();
    let codes = CODES.lock().clone().unwrap_or_default();
    EXACT.lock().as_ref().map(|m| m.iter().filter_map(|((h, _), s)| Some((*h, codes.get(h)?.clone(), s.clone()))).collect()).unwrap_or_default()
}

/// The code file's records ([code hash, word count, words], appended): each taken when its words
/// match its hash. Anything else, e.g. the zeros a write cut short by a power loss leaves (the
/// file's length kept, not its data), is passed in 4-byte steps, the records' alignment, to the
/// next record: the records after it are read, and a program whose record was lost is kept again
/// when next met (until 2026-11-20 the reading stopped there, and what later sessions kept was
/// never read again). With the bytes passed and the records that did not match their hash.
fn parse_codes(b: &[u8]) -> (HashMap<u64, std::sync::Arc<Vec<u32>>>, usize, usize) {
    const MOST_WORDS: usize = 1 << 18;
    let mut codes = HashMap::new();
    let (mut at, mut passed, mut bad) = (0usize, 0usize, 0usize);
    while at + 12 <= b.len() {
        let h = u64::from_le_bytes(b[at..at + 8].try_into().unwrap());
        let n = u32::from_le_bytes(b[at + 8..at + 12].try_into().unwrap()) as usize;
        let end = at + 12 + 4 * n;
        if h == 0 || n == 0 || n > MOST_WORDS || end > b.len() {
            at += 4;
            passed += 4;
            continue;
        }
        let words: Vec<u32> = b[at + 12..end].chunks_exact(4).map(|c| u32::from_le_bytes(c.try_into().unwrap())).collect();
        if super::shaders::hash_words(&words) != h {
            at += 4;
            passed += 4;
            bad += 1;
            continue;
        }
        codes.insert(h, std::sync::Arc::new(words));
        at = end;
    }
    (codes, passed, bad)
}

/// The code file beside the store.
fn code_path() -> Option<std::path::PathBuf> {
    STORE.get().map(|p| p.with_extension("code"))
}

/// Keeps the code of the program at `addr` (code hash `h`) unless kept.
fn keep_code(addr: u64, h: u64) {
    load();
    if CODES.lock().as_ref().is_some_and(|m| m.contains_key(&h)) {
        return;
    }
    let Ok((words, hash)) = super::shaders::program_code(addr) else { return };
    if hash != h {
        return;
    }
    let mut rec = Vec::with_capacity(12 + 4 * words.len());
    rec.extend_from_slice(&h.to_le_bytes());
    rec.extend_from_slice(&(words.len() as u32).to_le_bytes());
    for w in &words {
        rec.extend_from_slice(&w.to_le_bytes());
    }
    {
        let mut g = CODES.lock();
        let m = g.get_or_insert_with(HashMap::new);
        if m.contains_key(&h) {
            return;
        }
        m.insert(h, std::sync::Arc::new(words));
    }
    STATS[CODES_KEPT].fetch_add(1, Ordering::Relaxed);
    if let Some(p) = code_path() {
        let _ = std::fs::OpenOptions::new().create(true).append(true).open(p).and_then(|mut f| f.write_all(&rec));
    }
}

/// The learned compute programs whose code is kept, with their specs
/// (`--compile-bench`).
pub(crate) fn learned_compute() -> Vec<(u64, std::sync::Arc<Vec<u32>>, Vec<Spec>)> {
    load();
    let codes = CODES.lock().clone().unwrap_or_default();
    learned(|m| {
        m.iter()
            .filter_map(|(h, (_, seen))| {
                let code = codes.get(h)?.clone();
                let specs = seen.iter().map(|s| super::shaders::compute_spec(0, *h, s.rsrc2, s.local, s.wave64, s.resources.clone(), s.batched)).collect();
                Some((*h, code, specs))
            })
            .collect()
    })
}

/// Whether program `code_hash` has learned specs, none of them batched, and
/// its code kept: the session's start makes those pipelines ahead (`early`),
/// so a run of its dispatches keeps it unbatched while its batched pipeline
/// is made (`shaders::mark_batchable`).
pub(crate) fn learned_unbatched_only(code_hash: u64) -> bool {
    learned(|m| m.get(&code_hash).is_some_and(|(_, v)| !v.is_empty() && v.iter().all(|s| !s.batched))) && code_of(code_hash).is_some()
}

fn code_of(h: u64) -> Option<std::sync::Arc<Vec<u32>>> {
    load();
    CODES.lock().as_ref().and_then(|m| m.get(&h).cloned())
}

/// Most pipelines queued at the start of a session (`early`).
const EARLY_MAX: usize = 2500;

/// Queues the learned pipelines whose programs' code is kept, by rank, up
/// to EARLY_MAX (once, when the workers start).
fn early() {
    load();
    let codes: std::collections::HashSet<u64> = CODES.lock().as_ref().map(|m| m.keys().copied().collect()).unwrap_or_default();
    let mut jobs: Vec<(u32, Job)> = Vec::new();
    learned(|m| {
        for (h, (rank, _)) in m.iter() {
            if codes.contains(h) {
                jobs.push((*rank, Job::Code(*h)));
            }
        }
    });
    gfx_learned(|g| {
        for (i, (rank, r)) in g.recipes.iter().enumerate() {
            if codes.contains(&r.vs) && (r.ps == 0 || codes.contains(&r.ps)) {
                jobs.push((*rank, Job::Gfx(i)));
            }
        }
    });
    // The earliest used first; the rest (later areas, a growing store) when
    // their objects are created, so a session holds at most EARLY_MAX
    // pipelines it may not need.
    jobs.sort_unstable();
    jobs.truncate(EARLY_MAX);
    {
        let mut q = QUEUED_HASHES.lock();
        let q = q.get_or_insert_with(Default::default);
        jobs.retain(|(_, j)| match j {
            Job::Code(h) => q.insert(*h),
            _ => true,
        });
    }
    gfx_learned(|g| {
        jobs.retain(|(_, j)| match j {
            Job::Gfx(i) => g.queued.insert(*i),
            _ => true,
        })
    });
    STATS[EARLY].fetch_add(jobs.len() as u64, Ordering::Relaxed);
    // (After every job of an object the title created, which a level being
    // loaded is about to use: FINDINGS.md 2026-11-06 late morning.)
    let mut q = QUEUE.lock();
    for (r, j) in jobs {
        q.ready.push(std::cmp::Reverse((r | STARTUP, j)));
    }
    CV.notify_all();
}

/// The rank bit of jobs queued at the session's start (`early`): the jobs
/// of objects the title creates go first.
const STARTUP: u32 = 1 << 31;
/// Programs and graphics pipelines queued again for an object the title
/// created after their start-up job.
static URGENT_HASHES: Mutex<Option<std::collections::HashSet<u64>>> = Mutex::new(None);
static URGENT_GFX: Mutex<Option<std::collections::HashSet<usize>>> = Mutex::new(None);
/// Jobs queued again that way: [compute, graphics].
static URGENT: [AtomicU64; 2] = [const { AtomicU64::new(0) }; 2];

fn add_recipe(g: &mut GfxLearned, rank: u32, r: GfxRecipe) {
    let i = g.recipes.len();
    g.by_hash.entry(r.vs).or_default().push(i);
    if r.ps != 0 && r.ps != r.vs {
        g.by_hash.entry(r.ps).or_default().push(i);
    }
    g.recipes.push((rank, r));
}

/// The learned compute specs.
fn learned<R>(f: impl FnOnce(&mut HashMap<u64, (u32, Vec<Seen>)>) -> R) -> R {
    load();
    let mut g = LEARNED.lock();
    f(g.get_or_insert_with(HashMap::new))
}

/// The learned graphics pipelines.
fn gfx_learned<R>(f: impl FnOnce(&mut GfxLearned) -> R) -> R {
    load();
    let mut g = GFX.lock();
    f(g.get_or_insert_with(GfxLearned::default))
}

/// Appends a line to the store.
fn store_line(line: &str) {
    // (Lines written together count each: the next learned spec's rank.)
    LINES.fetch_add(line.matches('\n').count().max(1) as u64, Ordering::Relaxed);
    if let Some(p) = STORE.get() {
        // (A store whose last line was cut short, by a power loss: this session's first line on a
        // line of its own, not glued to it. Until the first line is written the appends go one
        // at a time, so the first written is the one that gets the newline, spent only once
        // written; then they go as they come. The reviews of 2026-11-21.)
        static APPEND: Mutex<Option<bool>> = Mutex::new(None);
        static SETTLED: std::sync::atomic::AtomicBool = std::sync::atomic::AtomicBool::new(false);
        let append = |text: &str| std::fs::OpenOptions::new().create(true).append(true).open(p).and_then(|mut f| f.write_all(text.as_bytes()));
        let r = if SETTLED.load(Ordering::Acquire) {
            append(line)
        } else {
            let mut cut = APPEND.lock();
            if cut.is_none() {
                use std::io::{Read, Seek};
                *cut = Some(
                    std::fs::File::open(p)
                        .ok()
                        .and_then(|mut f| {
                            let len = f.metadata().ok()?.len();
                            if len == 0 {
                                return Some(false);
                            }
                            f.seek(std::io::SeekFrom::Start(len - 1)).ok()?;
                            let mut b = [0u8; 1];
                            f.read_exact(&mut b).ok()?;
                            Some(b[0] != b'\n')
                        })
                        .unwrap_or(false),
                );
            }
            let first = *cut == Some(true);
            let r = if first { append(&format!("\n{line}")) } else { append(line) };
            if !first || r.is_ok() {
                *cut = Some(false);
                SETTLED.store(true, Ordering::Release);
            }
            r
        };
        if let Err(e) = r {
            static WARNED: std::sync::atomic::AtomicBool = std::sync::atomic::AtomicBool::new(false);
            if !WARNED.swap(true, Ordering::Relaxed) {
                crate::warn!("gpu: learned specs not stored in {} ({e})", p.display());
            }
        }
    }
}

/// The title created a compute shader object for the program at `addr`.
pub fn created(addr: u64, rsrc2: u32, local: [u32; 3], modifier: u32) {
    STATS[OBJECTS].fetch_add(1, Ordering::Relaxed);
    // (`DS_COMPILE_TRACE=1`: when, against the program's first dispatch,
    // the lead a pipeline made from the object would have.)
    if super::exec::compile_trace_on() {
        crate::info!("compile trace: compute object created for cs {addr:#x} at flip {}", super::FRAMES.load(Ordering::Relaxed));
    }
    CREATED.lock().get_or_insert_with(HashMap::new).insert(addr, Created { rsrc2, local, modifier });
    if super::tune::precompile() >= 2 {
        // (A program in a loaded module's image first: its guess is made at once, `run`; the
        // title creates ~6,600 objects at its start, looked at in ~4 s.)
        if in_module(addr) {
            QUEUE.lock().fresh.push_front(addr);
        } else {
            QUEUE.lock().fresh.push_back(addr);
        }
        CV.notify_one();
        ensure_workers();
    }
}

/// The title created a vertex (NGG) or pixel shader object for the program
/// at `addr`: learned graphics pipelines of its program are made once the
/// other stage's program has an object too.
pub fn created_graphics(addr: u64) {
    if super::tune::precompile() >= 2 {
        QUEUE.lock().fresh.push_back(addr);
        CV.notify_one();
        ensure_workers();
    }
}

/// Objects created and not looked at yet (in creation order), and programs
/// with specs to make, the earliest first dispatched first (by rank; the
/// guesses of `precompile` 3 after all learned ones).
struct Queues {
    fresh: std::collections::VecDeque<u64>,
    ready: std::collections::BinaryHeap<std::cmp::Reverse<(u32, Job)>>,
}

/// A pipeline job: a compute program by its object's address or by its
/// kept code (code hash), a learned graphics pipeline (index).
#[derive(Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
enum Job {
    Addr(u64),
    Code(u64),
    Gfx(usize),
}

static QUEUE: Mutex<Queues> = Mutex::new(Queues { fresh: std::collections::VecDeque::new(), ready: std::collections::BinaryHeap::new() });
static CV: Condvar = Condvar::new();
static WORKERS: std::sync::atomic::AtomicUsize = std::sync::atomic::AtomicUsize::new(0);
/// [pipelines created, their ns (translation included), present already,
/// objects with no spec to make, failed, objects done].
static DONE: [AtomicU64; 6] = [const { AtomicU64::new(0) }; 6];

thread_local! {
    /// This precompile thread's own handle (`boostable_thread`).
    static HANDLE: std::cell::Cell<Option<isize>> = const { std::cell::Cell::new(None) };
}

/// Raised priorities (`boost`), for the status line.
static BOOSTS: AtomicU64 = AtomicU64::new(0);

/// The calling thread's handle if it is a precompile thread (`exec`'s
/// pipeline flights record it).
pub(crate) fn boostable_thread() -> Option<isize> {
    HANDLE.with(|h| h.get())
}

/// A command processor waits for a pipeline a precompile thread is making:
/// that thread runs above normal until its job ends (`worker`).
pub(crate) fn boost(h: isize) {
    use crate::platform::os::Win32::System::Threading::{SetThreadPriority, THREAD_PRIORITY_ABOVE_NORMAL};
    unsafe { SetThreadPriority(h as _, THREAD_PRIORITY_ABOVE_NORMAL) };
    BOOSTS.fetch_add(1, Ordering::Relaxed);
}

/// The lowest priority again (after each job).
fn lower() {
    use crate::platform::os::Win32::System::Threading::{GetCurrentThread, SetThreadPriority, THREAD_PRIORITY_LOWEST};
    unsafe { SetThreadPriority(GetCurrentThread(), THREAD_PRIORITY_LOWEST) };
}

fn ensure_workers() {
    let want = super::tune::precompile_workers().min(16) as usize;
    while WORKERS.load(Ordering::Acquire) < want {
        let n = WORKERS.fetch_add(1, Ordering::AcqRel);
        if n >= want {
            WORKERS.fetch_sub(1, Ordering::AcqRel);
            break;
        }
        let name = format!("gpu-precompile{n}");
        let r = crate::thread::spawn_host(
            &name.clone(),
            8 << 20,
            Box::new(move || {
                crate::log::set_thread_name(&name);
                crate::exception::arm_watch_current_thread(false);
                // Idle time only: the command processors and the title first.
                // A real handle (the pseudo-handle names the caller) lets a
                // waiting thread raise this one (`boost`).
                unsafe {
                    use crate::platform::os::Win32::System::Threading::{GetCurrentThreadId, OpenThread, THREAD_QUERY_LIMITED_INFORMATION, THREAD_SET_LIMITED_INFORMATION};
                    let h = OpenThread(THREAD_SET_LIMITED_INFORMATION | THREAD_QUERY_LIMITED_INFORMATION, 0, GetCurrentThreadId());
                    if !h.is_null() {
                        HANDLE.with(|c| c.set(Some(h as isize)));
                    }
                }
                lower();
                worker();
            }),
        );
        if let Err(e) = r {
            crate::warn!("gpu: no precompile worker ({e})");
            WORKERS.fetch_sub(1, Ordering::AcqRel);
            break;
        }
    }
}

fn worker() {
    // The title creates its first shader objects before the device exists.
    while super::vk::try_ctx().is_none() {
        std::thread::sleep(std::time::Duration::from_millis(50));
    }
    static EARLY_ONCE: std::sync::Once = std::sync::Once::new();
    EARLY_ONCE.call_once(early);
    loop {
        // Looking at a created object is cheap (its code hash), making a
        // pipeline is not: every object is looked at first.
        let (fresh, job) = {
            let mut q = QUEUE.lock();
            loop {
                // (A job of rank 0 first: a module's guess, `classify`.)
                if matches!(q.ready.peek(), Some(std::cmp::Reverse((0, _)))) {
                    if let Some(std::cmp::Reverse((r, j))) = q.ready.pop() {
                        break (None, Some((r, j)));
                    }
                }
                if let Some(a) = q.fresh.pop_front() {
                    break (Some(a), None);
                } else if let Some(std::cmp::Reverse((r, j))) = q.ready.pop() {
                    break (None, Some((r, j)));
                }
                CV.wait(&mut q);
            }
        };
        if let Some(a) = fresh {
            classify(a);
            continue;
        }
        // (Jobs queued at the session's start make the store's binaries
        // only: what an area needs alive comes with its objects, `pipeuse`.)
        match job {
            None => {}
            Some((_, Job::Addr(a))) => {
                run(a);
                DONE[5].fetch_add(1, Ordering::Relaxed);
            }
            Some((_, Job::Code(h))) => run_code(h),
            Some((r, Job::Gfx(i))) => run_gfx(i, r & STARTUP == 0),
        }
        lower();
    }
}

/// Looks at a created object: its program's code hash, a compute program
/// with learned specs queued by its rank, graphics pipelines it completes
/// queued by theirs.
fn classify(addr: u64) {
    let Ok(code_hash) = super::shaders::program_hash_uncached(addr) else {
        DONE[4].fetch_add(1, Ordering::Relaxed);
        return;
    };
    HASH_ADDR.lock().get_or_insert_with(HashMap::new).insert(code_hash, addr);
    let compute = CREATED.lock().as_ref().is_some_and(|m| m.contains_key(&addr));
    if compute {
        // (A program in a loaded module's image without learned specs: its guess, first.)
        let learned_rank = rank(code_hash);
        match learned_rank.or_else(|| in_module(addr).then_some(0)) {
            Some(r) => {
                // Learned: its code kept for the next session's start.
                if learned_rank.is_some() {
                    keep_code(addr, code_hash);
                }
                if QUEUED_HASHES.lock().get_or_insert_with(Default::default).insert(code_hash) || super::tune::precompile() >= 3 {
                    QUEUE.lock().ready.push(std::cmp::Reverse((r, Job::Addr(addr))));
                    CV.notify_one();
                } else if URGENT_HASHES.lock().get_or_insert_with(Default::default).insert(code_hash) {
                    // Queued at the start behind the earlier-learned areas:
                    // queued again ahead of them (the later job finds its
                    // pipelines made).
                    URGENT[0].fetch_add(1, Ordering::Relaxed);
                    QUEUE.lock().ready.push(std::cmp::Reverse((r, Job::Addr(addr))));
                    CV.notify_one();
                } else {
                    DONE[5].fetch_add(1, Ordering::Relaxed);
                }
            }
            None => {
                DONE[3].fetch_add(1, Ordering::Relaxed);
                DONE[5].fetch_add(1, Ordering::Relaxed);
            }
        }
        return;
    }
    if gfx_learned(|g| g.by_hash.contains_key(&code_hash)) {
        keep_code(addr, code_hash);
    }
    let jobs: Vec<(u32, usize)> = gfx_learned(|g| {
        let mut v = Vec::new();
        let Some(ids) = g.by_hash.get(&code_hash) else { return v };
        // (Lock order: the learned pipelines, then the addresses.)
        let present = HASH_ADDR.lock();
        let has = |h: u64| present.as_ref().is_some_and(|m| m.contains_key(&h));
        for &i in ids {
            let (rank, r) = &g.recipes[i];
            if has(r.vs) && (r.ps == 0 || has(r.ps)) {
                if !g.queued.contains(&i) {
                    v.push((*rank, i));
                } else if URGENT_GFX.lock().get_or_insert_with(Default::default).insert(i) {
                    URGENT[1].fetch_add(1, Ordering::Relaxed);
                    v.push((*rank, i));
                }
            }
        }
        drop(present);
        for &(_, i) in &v {
            g.queued.insert(i);
        }
        v
    });
    if !jobs.is_empty() {
        let mut q = QUEUE.lock();
        for (r, i) in jobs {
            q.ready.push(std::cmp::Reverse((r, Job::Gfx(i))));
        }
        CV.notify_all();
    }
}

/// Makes the learned graphics pipeline `i` from the programs of the
/// objects created for its stages.
fn run_gfx(i: usize, keep_ok: bool) {
    let Some(r) = gfx_learned(|g| g.recipes.get(i).map(|e| e.1.clone())) else { return };
    let (vs_addr, ps_addr) = {
        let m = HASH_ADDR.lock();
        let m = m.as_ref();
        (m.and_then(|m| m.get(&r.vs).copied()).unwrap_or(0), m.and_then(|m| m.get(&r.ps).copied()).unwrap_or(0))
    };
    // A stage's translation: from the kept code, else the object's program.
    let stage = |h: u64, addr: u64, spec: Spec| -> Result<std::sync::Arc<super::shaders::Translated>, String> {
        match code_of(h) {
            Some(code) => super::shaders::translate_code_for_spec(&code, h, spec),
            None if addr != 0 => super::shaders::translate_for_spec(addr, spec),
            None => Err("no code".into()),
        }
    };
    let t = std::time::Instant::now();
    let made = (|| -> Result<Option<bool>, String> {
        let v = &r.vs_seen;
        let Some(vs_spec) = super::shaders::vertex_spec(vs_addr, v.user_sgprs, v.wave64, v.base_instance, v.resources.clone()) else { return Ok(None) };
        let vs_tr = stage(r.vs, vs_addr, vs_spec)?;
        let ps_spec = |p: &StageSeen, measure: bool| super::shaders::pixel_spec(ps_addr, p.user_sgprs, p.wave64, p.ps_input_addr, p.ps_inputs.clone(), p.mrt, p.early_tests, p.early_ok, p.resources.clone(), measure);
        let ps_tr = match &r.ps_seen {
            Some(p) if r.ps != 0 => Some((stage(r.ps, ps_addr, ps_spec(p, p.lod_measure))?, ps_addr)),
            _ => None,
        };
        let made = super::exec::precompile_gfx_translations((&vs_tr, vs_addr), ps_tr.as_ref().map(|(t, a)| (&**t, *a)), &r.state, keep_ok);
        if made == Some(true) && r.ps_seen.as_ref().is_some_and(|p| p.lod_measure) {
            GFX_DONE[4].fetch_add(1, Ordering::Relaxed);
        }
        Ok(made)
    })();
    match made {
        Ok(Some(true)) => {
            GFX_DONE[0].fetch_add(1, Ordering::Relaxed);
            GFX_DONE[1].fetch_add(t.elapsed().as_nanos() as u64, Ordering::Relaxed);
        }
        Ok(Some(false)) => {
            GFX_DONE[2].fetch_add(1, Ordering::Relaxed);
        }
        Ok(None) | Err(_) => {
            if GFX_DONE[3].fetch_add(1, Ordering::Relaxed) < 8 {
                crate::info!("gpu: precompile of graphics pipeline vs {:#x} ps {:#x} failed: {}", r.vs, r.ps, made.err().unwrap_or_default());
            }
        }
    }
}

/// The player's learned work for the launcher's preparation (`compbench::precompile_all`):
/// every learned compute program whose code the store keeps, with its specs keyed as a session
/// makes them (`keyed_learned_specs`), and the learned graphics recipes whose stages' code it
/// keeps (indices for `prepare_gfx`, with their code words). After an update whose translations
/// differ the store's pipelines no longer serve, but its specs and recipes still name what the
/// player's sessions used: prepared again, a first session after the update finds them made.
pub(crate) fn learned_work() -> (Vec<(u64, std::sync::Arc<Vec<u32>>, Vec<Spec>)>, Vec<(usize, usize)>) {
    load();
    let codes = CODES.lock().clone().unwrap_or_default();
    let hashes: Vec<u64> = learned(|m| m.keys().copied().collect());
    let compute = hashes
        .into_iter()
        .filter_map(|h| {
            let code = codes.get(&h)?.clone();
            let specs = keyed_learned_specs(h, &code);
            (!specs.is_empty()).then_some((h, code, specs))
        })
        .collect();
    let gfx = gfx_learned(|g| {
        g.recipes
            .iter()
            .enumerate()
            .filter_map(|(i, (_, r))| {
                let vs = codes.get(&r.vs)?.len();
                let ps = if r.ps == 0 { 0 } else { codes.get(&r.ps)?.len() };
                Some((i, vs + ps))
            })
            .collect()
    });
    (compute, gfx)
}

/// Makes learned graphics recipe `i`'s pipeline into the store (the preparation's; as a
/// session's start does, `run_gfx`): whether it was made or found.
pub(crate) fn prepare_gfx(i: usize) -> bool {
    let before = GFX_DONE[3].load(Ordering::Relaxed);
    run_gfx(i, false);
    GFX_DONE[3].load(Ordering::Relaxed) == before
}

/// Graphics pipelines made from objects: [created, their ns, present
/// already, failed, of the created those of draws measuring LOD statistics
/// in rotation].
static GFX_DONE: [AtomicU64; 5] = [const { AtomicU64::new(0) }; 5];

/// A graphics pipeline was created on a draw's path (`exec::draw`): its
/// stages' specs and state are learned (once).
pub(crate) fn learn_gfx(vs: (&super::shaders::Translated, u64), ps: Option<(&super::shaders::Translated, u64)>, state: &super::exec::GfxState) {
    let Ok(vs_hash) = super::shaders::program_hash(vs.1) else { return };
    let ps_hash = match ps {
        Some(p) => match super::shaders::program_hash(p.1) {
            Ok(h) => h,
            Err(_) => return,
        },
        None => 0,
    };
    let r = GfxRecipe { vs: vs_hash, vs_seen: StageSeen::of(&vs.0.spec), ps: ps_hash, ps_seen: ps.map(|p| StageSeen::of(&p.0.spec)), state: *state };
    let new = gfx_learned(|g| {
        if g.set.contains(&r) {
            return false;
        }
        g.set.insert(r.clone());
        let rank = LINES.load(Ordering::Relaxed) as u32;
        add_recipe(g, rank, r.clone());
        // Made (or being made) now: not queued again this session.
        let i = g.recipes.len() - 1;
        g.queued.insert(i);
        true
    });
    if new {
        STATS[LEARNT].fetch_add(1, Ordering::Relaxed);
        store_line(&format_gfx(&r));
        keep_code(vs.1, vs_hash);
        if let Some(p) = ps {
            keep_code(p.1, ps_hash);
        }
    }
}

/// The rank of the compute program `code_hash` among those to make: its
/// learned rank, after all of them with `precompile` 3; None: nothing to
/// make.
fn rank(code_hash: u64) -> Option<u32> {
    match learned(|m| m.get(&code_hash).map(|e| e.0)) {
        Some(r) => Some(r),
        None if super::tune::precompile() >= 3 => Some(u32::MAX),
        None => None,
    }
}

/// Whether a compute object's program lies in a loaded module's image (the executable's own
/// programs: outside the game's shader files, so never in the launcher's preparation; 58 of a
/// first 1-1 walk's 6,614 compute objects, 2026-11-19): its object's guess is made at its
/// creation when its program has no learned specs (`run`).
fn in_module(addr: u64) -> bool {
    crate::loader::module_at(addr).is_some()
}

/// Makes the pipelines of the program at `addr`: for the specs its
/// dispatches used before, and for its object's guess (`precompile` 3, or a
/// program in a loaded module's image without learned specs: `in_module`).
fn run(addr: u64) {
    let Some(c) = CREATED.lock().as_ref().and_then(|m| m.get(&addr).copied()) else { return };
    let Ok(code_hash) = super::shaders::program_hash(addr) else {
        DONE[4].fetch_add(1, Ordering::Relaxed);
        return;
    };
    let seen = learned(|m| m.get(&code_hash).map(|e| e.1.clone())).unwrap_or_default();
    super::shaders::seed_formats(code_hash, &seen.iter().map(|s| s.resources.as_slice()).collect::<Vec<_>>());
    let module_guess = seen.is_empty() && in_module(addr);
    // (A module guess's first variant, the one its dispatches use: kept alive, so that a
    // dispatch or a scan's compile meeting it in flight waits for it instead of compiling it
    // again; 2026-11-19: made through the store and let go, the face encoder was compiled
    // twice at once and its dispatch still waited 2.3 s.)
    let mut keep_alive: Option<Spec> = None;
    let mut specs: Vec<Spec> = Vec::new();
    for mut s in seen {
        // (Keyed as a preparation keys them now.)
        let user_sgprs = ((s.rsrc2 >> 1) & 0x1F).min(16) as u16;
        if let Ok((insts, plan, _)) = super::shaders::stage_plan(addr, user_sgprs, 0) {
            super::shaders::key_learned(code_hash, &insts, &plan, &mut s.resources);
        }
        let spec = super::shaders::compute_spec(addr, code_hash, s.rsrc2, s.local, s.wave64, s.resources, s.batched);
        for spec in super::shaders::dispatch_variants(spec) {
            if !specs.contains(&spec) {
                specs.push(spec);
            }
        }
    }
    // The guess's dispatch variants, the LOD-rotated one first, as for learned specs (the guess
    // alone was the measuring variant: the face encoder cs 0x8026f4100 waited 2.8 s at a new
    // player's first load for its plain one, compiled by a scan 280 ms before its dispatch,
    // its object made 6.9 s before; 2026-11-19).
    if super::tune::precompile() >= 3 || module_guess {
        let user_sgprs = ((c.rsrc2 >> 1) & 0x1F).min(16) as u16;
        if let Ok((insts, plan, _)) = super::shaders::stage_plan(addr, user_sgprs, 0) {
            let g = super::shaders::guess_compute_spec(addr, &insts, &plan, code_hash, c.rsrc2, c.local);
            // (A module's guess with a multisampled image is left out: its translation declares
            // multisampled storage images, which spirv-val refuses without the
            // StorageImageMultisample capability (8 such modules under the validation layer,
            // 2026-11-19); no dispatch of such a program was seen.)
            let ms = g.resources.iter().any(|r| matches!(r, ResSpec::Image { dim: rdna::translate::ImageDim::D2Msaa | rdna::translate::ImageDim::D2MsaaArray, .. }));
            // (A module's guess: its first variant only, the one its first dispatches use; both
            // cost 15 compiles and 44 s of these threads at a first session's start, 2026-11-19.)
            let variants = if module_guess && super::tune::precompile() < 3 {
                if ms {
                    Vec::new()
                } else {
                    super::shaders::dispatch_variants(g).into_iter().take(1).collect()
                }
            } else {
                super::shaders::dispatch_variants(g)
            };
            if module_guess && !variants.is_empty() {
                MODULE_GUESSES.fetch_add(1, Ordering::Relaxed);
                keep_alive = variants.first().cloned();
            }
            for g in variants {
                if !specs.contains(&g) {
                    specs.push(g);
                }
            }
        }
    }
    if specs.is_empty() {
        DONE[3].fetch_add(1, Ordering::Relaxed);
        return;
    }
    for spec in specs {
        let t = std::time::Instant::now();
        // (Alive only while the store lacks it: later sessions find it there, and most of these
        // programs never run.)
        let alive = keep_alive.as_ref() == Some(&spec);
        let r = super::shaders::translate_for_spec(addr, spec).and_then(|tr| if alive && !super::exec::compute_pipe_stored(&tr) { super::exec::precompile_translation(&tr, addr) } else { super::exec::precompile_or_warm(&tr, addr, true) }.ok_or_else(|| "pipeline creation failed".to_string()));
        match r {
            Ok(true) => {
                DONE[0].fetch_add(1, Ordering::Relaxed);
                DONE[1].fetch_add(t.elapsed().as_nanos() as u64, Ordering::Relaxed);
            }
            Ok(false) => {
                DONE[2].fetch_add(1, Ordering::Relaxed);
            }
            Err(e) => {
                if DONE[4].fetch_add(1, Ordering::Relaxed) < 8 {
                    crate::info!("gpu: precompile of cs {addr:#x} failed: {e}");
                }
            }
        }
    }
}

/// A program's learned compute specs keyed as `run_code` keys them
/// (`--compile-bench` with `DS_CB_MEMTEST`).
pub(crate) fn keyed_learned_specs(h: u64, code: &[u32]) -> Vec<Spec> {
    let seen = learned(|m| m.get(&h).map(|e| e.1.clone())).unwrap_or_default();
    let Ok(insts) = rdna::decode_program(code) else { return Vec::new() };
    super::shaders::note_program(code, &insts, h);
    let mut plans: Vec<(u16, rdna::analysis::Plan)> = Vec::new();
    super::shaders::seed_formats(h, &seen.iter().map(|s| s.resources.as_slice()).collect::<Vec<_>>());
    let mut done: Vec<Spec> = Vec::new();
    for mut s in seen {
        let user_sgprs = ((s.rsrc2 >> 1) & 0x1F).min(16) as u16;
        let k = match plans.iter().position(|(u, _)| *u == user_sgprs) {
            Some(k) => k,
            None => {
                plans.push((user_sgprs, rdna::analysis::analyze_opts(&insts, user_sgprs, 0, false)));
                plans.len() - 1
            }
        };
        super::shaders::key_learned(h, &insts, &plans[k].1, &mut s.resources);
        let spec = super::shaders::compute_spec(0, h, s.rsrc2, s.local, s.wave64, s.resources, s.batched);
        for spec in super::shaders::dispatch_variants(spec) {
            if !done.contains(&spec) {
                done.push(spec);
            }
        }
    }
    done
}

/// Makes the learned compute pipelines of a program from its kept code.
fn run_code(h: u64) {
    let Some(code) = code_of(h) else { return };
    let seen = learned(|m| m.get(&h).map(|e| e.1.clone())).unwrap_or_default();
    let insts = rdna::decode_program(&code).ok();
    if let Some(i) = &insts {
        super::shaders::note_program(&code, i, h);
    }
    let mut plans: Vec<(u16, rdna::analysis::Plan)> = Vec::new();
    super::shaders::seed_formats(h, &seen.iter().map(|s| s.resources.as_slice()).collect::<Vec<_>>());
    let mut done: Vec<Spec> = Vec::new();
    for (n, mut s) in seen.into_iter().enumerate() {
        let t = std::time::Instant::now();
        // (Keyed as a preparation keys them now, a plan per user SGPR count;
        // equal ones made once.)
        if let Some(insts) = &insts {
            let user_sgprs = ((s.rsrc2 >> 1) & 0x1F).min(16) as u16;
            let k = match plans.iter().position(|(u, _)| *u == user_sgprs) {
                Some(k) => k,
                None => {
                    plans.push((user_sgprs, rdna::analysis::analyze_opts(insts, user_sgprs, 0, false)));
                    plans.len() - 1
                }
            };
            super::shaders::key_learned(h, insts, &plans[k].1, &mut s.resources);
        }
        let spec = super::shaders::compute_spec(0, h, s.rsrc2, s.local, s.wave64, s.resources, s.batched);
        for spec in super::shaders::dispatch_variants(spec) {
            if done.contains(&spec) {
                continue;
            }
            done.push(spec.clone());
            // (Under the validation layer its messages name the program and
            // its learned spec: `DS_CB_HASHES` / `DS_CB_SPEC` of the compile
            // bench.)
            let _g = super::vk::creating(|| format!("cs {h:#018x} learned spec {n}"));
            let r = super::shaders::translate_code_for_spec(&code, h, spec).and_then(|tr| super::exec::precompile_or_warm(&tr, 0, false).ok_or_else(|| "pipeline creation failed".to_string()));
            match r {
                Ok(true) => {
                    DONE[0].fetch_add(1, Ordering::Relaxed);
                    DONE[1].fetch_add(t.elapsed().as_nanos() as u64, Ordering::Relaxed);
                }
                Ok(false) => {
                    DONE[2].fetch_add(1, Ordering::Relaxed);
                }
                Err(e) => {
                    if DONE[4].fetch_add(1, Ordering::Relaxed) < 8 {
                        crate::info!("gpu: precompile of cs {h:#018x} from its kept code failed: {e}");
                    }
                }
            }
        }
    }
}

/// Compute objects in a loaded module's image whose guess was made (`run`, `in_module`).
static MODULE_GUESSES: AtomicU64 = AtomicU64::new(0);

/// Format-accessed buffers of first translations by (V# format, dst_sel).
static FORMATS: Mutex<Option<HashMap<(u8, u16), u64>>> = Mutex::new(None);

/// A compute program's translation is missing (`shaders::prepare`, before
/// translating with `actual`, the code base left out): its spec is learned,
/// and the first time per program it is compared with the guess from its
/// shader object.
pub fn first_translation(addr: u64, insts: &[rdna::Inst], plan: &rdna::analysis::Plan, code_hash: u64, actual: &Spec) {
    // A program whose specs were learned: why its translation was not made
    // ahead (the same key spec from a learned one, or none).
    if let Some(seen) = learned(|m| m.get(&code_hash).map(|e| e.1.clone())) {
        MISSES[0].fetch_add(1, Ordering::Relaxed);
        let mut key = actual.clone();
        key.code_base = 0;
        let made: Vec<Spec> = seen.into_iter().flat_map(|s| super::shaders::dispatch_variants(super::shaders::compute_spec(0, code_hash, s.rsrc2, s.local, s.wave64, s.resources, s.batched))).collect();
        if made.iter().any(|m| *m == key) {
            MISSES[1].fetch_add(1, Ordering::Relaxed);
        } else {
            let n = MISSES[2].fetch_add(1, Ordering::Relaxed);
            if n < 12 {
                let best = made.iter().map(|m| spec_diff(m, &key)).min_by_key(|d| d.len()).unwrap_or_default();
                crate::info!("gpu: cs {addr:#x} (code hash {code_hash:#018x}) translated at its dispatch: none of its {} learned specs gives its key spec (closest differs in: {best})", made.len());
            }
        }
    }
    LEARN_ADDR.with(|c| c.set(Some(addr)));
    learn(code_hash, actual);
    LEARN_ADDR.with(|c| c.set(None));
    if !COMPARED.lock().get_or_insert_with(Default::default).insert(addr) {
        STATS[LATER].fetch_add(1, Ordering::Relaxed);
        return;
    }
    STATS[FIRSTS].fetch_add(1, Ordering::Relaxed);
    let Some(c) = CREATED.lock().as_ref().and_then(|m| m.get(&addr).copied()) else { return };
    STATS[WITH_OBJECT].fetch_add(1, Ordering::Relaxed);
    let k = if c.modifier & (1 << 15) != 0 { 0 } else { 2 } + if actual.wave64 { 0 } else { 1 };
    MODIFIER[k].fetch_add(1, Ordering::Relaxed);
    let guess = super::shaders::guess_compute_spec(addr, insts, plan, code_hash, c.rsrc2, c.local);
    if guess == *actual {
        STATS[EQUAL].fetch_add(1, Ordering::Relaxed);
        return;
    }
    for (g, a) in guess.resources.iter().zip(actual.resources.iter()) {
        if let (ResSpec::Buffer { format: 0, dst_sel: 0xFAC }, ResSpec::Buffer { format, dst_sel }) = (g, a) {
            *FORMATS.lock().get_or_insert_with(HashMap::new).entry((*format, *dst_sel)).or_insert(0) += 1;
        }
    }
    let why = differs(&guess, actual);
    if why == usize::MAX {
        STATS[UNKNOWN_FORMAT].fetch_add(1, Ordering::Relaxed);
        return;
    }
    let n = STATS[DIFF + why].fetch_add(1, Ordering::Relaxed);
    // (`DS_GUESS_TRACE=1`, diagnosis: every program's first translation against the guess its
    // preparation made, field by field, with its code hash.)
    static TRACE: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    if *TRACE.get_or_init(|| std::env::var("DS_GUESS_TRACE").ok().as_deref() == Some("1")) {
        crate::info!("guess trace: cs {addr:#x} code {code_hash:#018x} first differs in {}; fields: {}", DIFF_NAMES[why], super::exec::spec_diff(&guess, actual));
    } else if n < 2 {
        crate::info!("gpu: precompile guess for cs {addr:#x} differs in {}: guess resources {:?}; actual {:?}", DIFF_NAMES[why], guess.resources, actual.resources);
    }
}

/// Whether a thread-group size can be a dispatch's (1 to 1,024 threads, the hardware's and
/// Vulkan's common limit). The store never takes another: it is made again at every start
/// (2026-11-15: specs with 512x8 and 0x1x1 threads, learned from the scouts' recording with
/// stale registers, were precompiled at each start; the validation layer reported one, and the
/// driver crashed with a division by zero once).
fn local_ok(l: [u32; 3]) -> bool {
    let n = l[0] as u64 * l[1] as u64 * l[2] as u64;
    n > 0 && n <= 1024
}

/// Records the spec a dispatch of the program `code_hash` used (once per
/// distinct spec; appended to the store).
fn learn(code_hash: u64, spec: &Spec) {
    if !local_ok(spec.cs_local_size) {
        return;
    }
    let s = Seen { rsrc2: spec.cs_rsrc2, local: spec.cs_local_size, wave64: spec.wave64, resources: spec.resources.clone(), batched: spec.batched };
    let new = learned(|m| {
        let rank = LINES.load(Ordering::Relaxed) as u32;
        let v = &mut m.entry(code_hash).or_insert_with(|| (rank, Vec::new())).1;
        if v.contains(&s) {
            false
        } else {
            v.push(s.clone());
            true
        }
    });
    if !new {
        return;
    }
    STATS[LEARNT].fetch_add(1, Ordering::Relaxed);
    store_line(&format_line(code_hash, &s));
    if let Some(a) = LEARN_ADDR.with(|c| c.get()) {
        keep_code(a, code_hash);
    }
}

thread_local! {
    /// The program being learned (`first_translation`): its code is kept.
    static LEARN_ADDR: std::cell::Cell<Option<u64>> = const { std::cell::Cell::new(None) };
}

const DIMS: [ImageDim; 10] = [ImageDim::D1, ImageDim::D2, ImageDim::D3, ImageDim::Cube, ImageDim::D1Array, ImageDim::D2Array, ImageDim::D2Msaa, ImageDim::D2MsaaArray, ImageDim::CubeArray, ImageDim::D3OrD2];
const CLASSES: [NumClass; 3] = [NumClass::Float, NumClass::Sint, NumClass::Uint];

/// One store line: code hash, RSRC2, thread-group size, wave64, then the
/// resources (`b<format>/<dst_sel>`, `g`, `s`, `i<dim>/<class>/<depth>`).
fn format_line(code_hash: u64, s: &Seen) -> String {
    let res: Vec<String> = s
        .resources
        .iter()
        .map(|r| match r {
            ResSpec::Buffer { format, dst_sel } => format!("b{format}/{dst_sel}"),
            ResSpec::Global => "g".into(),
            ResSpec::Sampler => "s".into(),
            ResSpec::Image { dim, class, depth } => format!("i{}/{}/{}", DIMS.iter().position(|d| d == dim).unwrap_or(1), CLASSES.iter().position(|c| c == class).unwrap_or(0), *depth as u8),
        })
        .collect();
    // (A final `B`: batched.)
    format!("{code_hash:016x} {:x} {} {} {} {} {}{}\n", s.rsrc2, s.local[0], s.local[1], s.local[2], s.wave64 as u8, res.join(" "), if s.batched { " B" } else { "" })
}

fn parse(line: &str) -> Option<(u64, Seen)> {
    let mut it = line.split_ascii_whitespace();
    let h = u64::from_str_radix(it.next()?, 16).ok()?;
    let rsrc2 = u32::from_str_radix(it.next()?, 16).ok()?;
    let local = [it.next()?.parse().ok()?, it.next()?.parse().ok()?, it.next()?.parse().ok()?];
    let wave64 = match it.next()? {
        "1" => true,
        "0" => false,
        _ => return None,
    };
    let mut resources = Vec::new();
    let mut batched = false;
    for t in it {
        if t == "B" {
            batched = true;
            continue;
        }
        let (k, rest) = t.split_at(1);
        let nums = rest.split('/').filter(|x| !x.is_empty()).map(|x| x.parse::<u32>().ok()).collect::<Option<Vec<u32>>>()?;
        resources.push(match (k, nums.as_slice()) {
            ("b", [f, d]) => ResSpec::Buffer { format: u8::try_from(*f).ok()?, dst_sel: u16::try_from(*d).ok()? },
            ("g", []) => ResSpec::Global,
            ("s", []) => ResSpec::Sampler,
            ("i", [d, c, z]) => ResSpec::Image { dim: *DIMS.get(*d as usize)?, class: *CLASSES.get(*c as usize)?, depth: *z != 0 },
            _ => return None,
        });
    }
    Some((h, Seen { rsrc2, local, wave64, resources, batched }))
}

fn res_tokens(res: &[ResSpec]) -> Vec<String> {
    res.iter()
        .map(|r| match r {
            ResSpec::Buffer { format, dst_sel } => format!("b{format}/{dst_sel}"),
            ResSpec::Global => "g".into(),
            ResSpec::Sampler => "s".into(),
            ResSpec::Image { dim, class, depth } => format!("i{}/{}/{}", DIMS.iter().position(|d| d == dim).unwrap_or(1), CLASSES.iter().position(|c| c == class).unwrap_or(0), *depth as u8),
        })
        .collect()
}

fn res_token(t: &str) -> Option<ResSpec> {
    let (k, rest) = t.split_at(1);
    let nums = rest.split('/').filter(|x| !x.is_empty()).map(|x| x.parse::<u32>().ok()).collect::<Option<Vec<u32>>>()?;
    Some(match (k, nums.as_slice()) {
        ("b", [f, d]) => ResSpec::Buffer { format: u8::try_from(*f).ok()?, dst_sel: u16::try_from(*d).ok()? },
        ("g", []) => ResSpec::Global,
        ("s", []) => ResSpec::Sampler,
        ("i", [d, c, z]) => ResSpec::Image { dim: *DIMS.get(*d as usize)?, class: *CLASSES.get(*c as usize)?, depth: *z != 0 },
        _ => return None,
    })
}

/// A stage: user data, wave64, base-instance SGPR (- none), PS_INPUT_ADDR,
/// early tests (bit 0 asked for, bit 1 allowed: `Spec::early_ok`; bit 2 a
/// draw measuring LOD statistics in rotation), `m` the targets' (export format, class) pairs, then `p`
/// pixel inputs (param/flat/default, 9 none) and the resources.
fn format_stage(s: &StageSeen) -> String {
    let mrt: Vec<String> = s.mrt.iter().flat_map(|m| [m.export_format.to_string(), m.class.to_string()]).collect();
    let inputs = s.ps_inputs.iter().map(|i| format!("p{}/{}/{}", i.param, i.flat as u8, i.default.unwrap_or(9)));
    let mut t = vec![s.user_sgprs.to_string(), (s.wave64 as u8).to_string(), s.base_instance.map_or("-".to_string(), |b| b.to_string()), format!("{:x}", s.ps_input_addr), (s.early_tests as u8 | (s.early_ok as u8) << 1 | (s.lod_measure as u8) << 2).to_string(), format!("m{}", mrt.join("/"))];
    t.extend(inputs);
    t.extend(res_tokens(&s.resources));
    t.join(" ")
}

fn parse_stage(text: &str) -> Option<StageSeen> {
    let mut it = text.split_ascii_whitespace();
    let user_sgprs = it.next()?.parse().ok()?;
    let wave64 = it.next()? == "1";
    let base_instance = match it.next()? {
        "-" => None,
        b => Some(b.parse().ok()?),
    };
    let ps_input_addr = u32::from_str_radix(it.next()?, 16).ok()?;
    let early: u8 = it.next()?.parse().ok()?;
    let (early_tests, early_ok, lod_measure) = (early & 1 != 0, early & 2 != 0, early & 4 != 0);
    let m: Vec<u8> = it.next()?.strip_prefix('m')?.split('/').map(|x| x.parse().ok()).collect::<Option<Vec<u8>>>()?;
    if m.len() != 16 {
        return None;
    }
    let mrt: [MrtSpec; 8] = std::array::from_fn(|k| MrtSpec { export_format: m[2 * k], class: m[2 * k + 1] });
    let (mut ps_inputs, mut resources) = (Vec::new(), Vec::new());
    for t in it {
        if let Some(p) = t.strip_prefix('p') {
            let v: Vec<u8> = p.split('/').map(|x| x.parse().ok()).collect::<Option<Vec<u8>>>()?;
            let [param, flat, default] = v.as_slice() else { return None };
            ps_inputs.push(PsInput { param: *param, flat: *flat != 0, default: (*default != 9).then_some(*default) });
        } else {
            resources.push(res_token(t)?);
        }
    }
    Some(StageSeen { user_sgprs, wave64, base_instance, ps_input_addr, ps_inputs, mrt, early_tests, early_ok, lod_measure, resources })
}

/// `G <vs hash> <ps hash> ; <vertex stage> ; <pixel stage | -> ; <state>`;
/// the state: rect, colors, blend (hex), write masks, depth format,
/// primitive class, depth clamp.
fn format_gfx(r: &GfxRecipe) -> String {
    let st = &r.state;
    let join = |v: Vec<String>| v.join("/");
    format!(
        "G {:016x} {:016x} ; {} ; {} ; {} {} {} {} {} {} {}\n",
        r.vs,
        r.ps,
        format_stage(&r.vs_seen),
        r.ps_seen.as_ref().map_or("-".to_string(), format_stage),
        st.rect as u8,
        join(st.colors.iter().map(|c| c.to_string()).collect()),
        join(st.blend.iter().map(|b| format!("{b:x}")).collect()),
        join(st.masks.iter().map(|m| m.to_string()).collect()),
        st.depth,
        st.topology_class,
        st.depth_clamp as u8
    )
}

fn parse_gfx(text: &str) -> Option<GfxRecipe> {
    let parts: Vec<&str> = text.split(" ; ").collect();
    let [head, vs, ps, state] = parts.as_slice() else { return None };
    let mut h = head.split_ascii_whitespace();
    let vs_hash = u64::from_str_radix(h.next()?, 16).ok()?;
    let ps_hash = u64::from_str_radix(h.next()?, 16).ok()?;
    let vs_seen = parse_stage(vs)?;
    let ps_seen = if ps.trim() == "-" { None } else { Some(parse_stage(ps)?) };
    let mut it = state.split_ascii_whitespace();
    let rect = it.next()? == "1";
    let colors: Vec<i32> = it.next()?.split('/').map(|x| x.parse().ok()).collect::<Option<Vec<i32>>>()?;
    let blend: Vec<u32> = it.next()?.split('/').map(|x| u32::from_str_radix(x, 16).ok()).collect::<Option<Vec<u32>>>()?;
    let masks: Vec<u8> = it.next()?.split('/').map(|x| x.parse().ok()).collect::<Option<Vec<u8>>>()?;
    let depth = it.next()?.parse().ok()?;
    let topology_class = it.next()?.parse().ok()?;
    let depth_clamp = it.next()? == "1";
    let state = super::exec::GfxState { rect, colors: colors.try_into().ok()?, blend: blend.try_into().ok()?, masks: masks.try_into().ok()?, depth, topology_class, depth_clamp };
    Some(GfxRecipe { vs: vs_hash, vs_seen, ps: ps_hash, ps_seen, state })
}

/// The first field (`DIFF_NAMES` index) in which `g` and `a` differ;
/// usize::MAX when only buffers the guess left unknown differ.
fn differs(g: &Spec, a: &Spec) -> usize {
    if g.wave64 != a.wave64 {
        return 0;
    }
    if g.cs_local_size != a.cs_local_size {
        return 1;
    }
    if g.cs_rsrc2 != a.cs_rsrc2 {
        return 2;
    }
    if g.user_sgprs != a.user_sgprs {
        return 3;
    }
    if g.resources.len() != a.resources.len() {
        return 4;
    }
    let mut unknown_only = false;
    for (x, y) in g.resources.iter().zip(a.resources.iter()) {
        if x == y {
            continue;
        }
        match (x, y) {
            // The guess's marker for a format-accessed buffer.
            (ResSpec::Buffer { format: 0, dst_sel: 0xFAC }, ResSpec::Buffer { .. }) => unknown_only = true,
            (ResSpec::Buffer { .. }, _) => return 5,
            (ResSpec::Image { dim: d1, class: c1, .. }, ResSpec::Image { dim: d2, class: c2, .. }) => {
                return if d1 != d2 {
                    6
                } else if c1 != c2 {
                    7
                } else {
                    8
                };
            }
            _ => return 9,
        }
    }
    let mut g2 = g.clone();
    let mut a2 = a.clone();
    g2.resources.clear();
    a2.resources.clear();
    if g2 != a2 {
        return 9;
    }
    if unknown_only {
        usize::MAX
    } else {
        9
    }
}

/// One status line ("" when off).
pub fn status() -> String {
    let s: Vec<u64> = STATS.iter().map(|c| c.load(Ordering::Relaxed)).collect();
    if s[OBJECTS] == 0 || super::tune::precompile() == 0 {
        return String::new();
    }
    let diffs: Vec<String> = DIFF_NAMES.iter().enumerate().filter(|(k, _)| s[DIFF + k] > 0).map(|(k, n)| format!("{n} {}", s[DIFF + k])).collect();
    let m: Vec<u64> = MODIFIER.iter().map(|c| c.load(Ordering::Relaxed)).collect();
    let d: Vec<u64> = DONE.iter().map(|c| c.load(Ordering::Relaxed)).collect();
    let made = if super::tune::precompile() >= 2 {
        format!(
            "; made from objects ({} of them done, {} waiting, {} guessed in a module's image): {} pipelines ({:.1} s), {} present already, {} objects with no spec to make, {} failed, {} raised for a waiting thread",
            d[5],
            {
                let q = QUEUE.lock();
                q.fresh.len() + q.ready.len()
            },
            MODULE_GUESSES.load(Ordering::Relaxed),
            d[0],
            d[1] as f64 / 1e9,
            d[2],
            d[3],
            d[4],
            BOOSTS.load(Ordering::Relaxed)
        )
    } else {
        String::new()
    };
    let made = if super::tune::precompile() >= 2 {
        let g: Vec<u64> = GFX_DONE.iter().map(|c| c.load(Ordering::Relaxed)).collect();
        format!(
            "{made}; graphics pipelines made from objects: {} ({:.1} s), {} present already, {} failed ({} created for draws measuring LOD statistics in rotation); objects' jobs queued again ahead of the start-up ones: {} compute, {} graphics",
            g[0],
            g[1] as f64 / 1e9,
            g[2],
            g[3],
            g[4],
            URGENT[0].load(Ordering::Relaxed),
            URGENT[1].load(Ordering::Relaxed)
        )
    } else {
        made
    };
    let made = if super::tune::precompile() >= 2 {
        let l = super::exec::precompile_live();
        format!(
            "{made}; kept alive (used in a recent session) {} compute, {} graphics; made for the store only {} compute, {} graphics; left to their first use (stored) {} compute, {} graphics; {}; {}",
            l[0],
            l[3],
            l[1],
            l[4],
            l[2],
            l[5],
            super::pipeuse::line(),
            super::exec::pipe_evict_line()
        )
    } else {
        made
    };
    let mut f: Vec<((u8, u16), u64)> = FORMATS.lock().as_ref().map(|m| m.iter().map(|(k, v)| (*k, *v)).collect()).unwrap_or_default();
    f.sort_by(|a, b| b.1.cmp(&a.1));
    let f: Vec<String> = f.iter().take(12).map(|((fm, ds), n)| format!("{fm}/{ds:#x} {n}")).collect();
    format!(
        "    compute shader objects (`precompile`): {} created; specs {} loaded, {} learned; translated at their dispatch with specs learned {} ({} of them with a learned spec giving the same key spec, {} with none); programs' code {} loaded, {} kept; exact key specs {} loaded ({} under another hash, {} of other preparation rules, {} lines not read), {} kept; {} jobs queued at the start; {} programs translated first ({} later translations with another spec), {} of them from an object: guess equal {}, equal but for format-accessed buffers {} ({}), different {} ({}); dispatch modifier bit 15 set: wave64 {}, wave32 {}; clear: wave64 {}, wave32 {}{made}\n",
        s[OBJECTS],
        s[LOADED],
        s[LEARNT],
        MISSES[0].load(Ordering::Relaxed),
        MISSES[1].load(Ordering::Relaxed),
        MISSES[2].load(Ordering::Relaxed),
        s[CODES_LOADED],
        s[CODES_KEPT],
        EXACT_STATS[0].load(Ordering::Relaxed),
        EXACT_STATS[2].load(Ordering::Relaxed),
        EXACT_STATS[4].load(Ordering::Relaxed),
        EXACT_STATS[3].load(Ordering::Relaxed),
        EXACT_STATS[1].load(Ordering::Relaxed),
        s[EARLY],
        s[FIRSTS],
        s[LATER],
        s[WITH_OBJECT],
        s[EQUAL],
        s[UNKNOWN_FORMAT],
        f.join(", "),
        (DIFF..DIFF + 10).map(|k| s[k]).sum::<u64>(),
        diffs.join(", "),
        m[0],
        m[1],
        m[2],
        m[3]
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A code record as `keep_code` appends it.
    fn code_record(words: &[u32]) -> Vec<u8> {
        let mut r = super::super::shaders::hash_words(words).to_le_bytes().to_vec();
        r.extend_from_slice(&(words.len() as u32).to_le_bytes());
        for w in words {
            r.extend_from_slice(&w.to_le_bytes());
        }
        r
    }

    #[test]
    fn code_file_read_through_damage() {
        let (a, b, c) = (vec![0xBF80_0000u32, 7, 9], vec![0x1234_5678u32; 5], vec![0xBF81_0000u32]);
        // Whole: all three.
        let mut f = [code_record(&a), code_record(&b), code_record(&c)].concat();
        let (codes, passed, bad) = parse_codes(&f);
        assert_eq!((codes.len(), passed, bad), (3, 0, 0));
        // Zeros after the first (a write cut by a power loss, 4 KB), then the others appended later.
        f = [code_record(&a), vec![0u8; 4096], code_record(&b), code_record(&c)].concat();
        let (codes, passed, bad) = parse_codes(&f);
        assert_eq!((codes.len(), passed, bad), (3, 4096, 0));
        // A record whose words were lost (zeros after its header), then the next.
        let mut cut = code_record(&b);
        for x in &mut cut[12..] {
            *x = 0;
        }
        f = [code_record(&a), cut.clone(), code_record(&c)].concat();
        let (codes, _, bad) = parse_codes(&f);
        assert!(codes.contains_key(&super::super::shaders::hash_words(&a)) && codes.contains_key(&super::super::shaders::hash_words(&c)));
        assert!(!codes.contains_key(&super::super::shaders::hash_words(&b)) && bad >= 1);
        // A record cut short at the file's end: the ones before it.
        f = [code_record(&a), code_record(&b)[..10].to_vec()].concat();
        let (codes, _, _) = parse_codes(&f);
        assert_eq!(codes.len(), 1);
    }

    #[test]
    fn store_lines_round_trip() {
        let s = Seen {
            rsrc2: 0x8984,
            local: [8, 8, 1],
            wave64: true,
            resources: vec![
                ResSpec::Image { dim: ImageDim::D2Array, class: NumClass::Uint, depth: false },
                ResSpec::Buffer { format: 20, dst_sel: 4 },
                ResSpec::Global,
                ResSpec::Sampler,
                ResSpec::Image { dim: ImageDim::CubeArray, class: NumClass::Float, depth: true },
            ],
            batched: true,
        };
        let line = format_line(0x1234_5678_9abc_def0, &s);
        let (h, t) = parse(line.trim_end()).unwrap();
        assert_eq!(h, 0x1234_5678_9abc_def0);
        assert!(t == s);
        assert!(parse("zz 1 2 3 4 1").is_none());
        assert!(parse("12 84 64 1 1 1 q7").is_none());
    }

    #[test]
    fn exact_lines_round_trip() {
        // (Words as a session of 2026-11-21 recorded them.)
        let words = "s1 c 12 0 0 0 614c01e1 64,1,1 152 0 - - - - 1 0 0 - 0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0 - b0.0,b20.4,b20.4";
        let spec = Spec::from_words(words).expect("words read");
        let (h, sh) = (0x6825_ea61_efbc_0222u64, 0x475a_b4e0_6b9f_07cfu64);
        let line = exact_line(h, sh, &spec);
        assert!(line.ends_with('\n'));
        let rest = line.trim_end().strip_prefix("K ").unwrap();
        let (h2, sh2, back) = parse_exact(rest).unwrap().unwrap();
        assert_eq!((h2, sh2), (h, sh));
        assert_eq!(back.to_words(), spec.to_words());
        // (Cut short, or a word changed: not read.)
        assert!(parse_exact(&rest[..rest.len() - 20]).is_none());
        assert!(parse_exact(&rest.replace("b20.4,b20.4", "b20.4,b20.1")).is_none());
        // (Whole, but of other preparation rules.)
        let body = format!("{:016x} {:016x} r{} {words}", 1u64, 2u64, crate::gpu::compbench::PREPARATION_RULES + 1);
        assert!(matches!(parse_exact(&format!("{body} {:016x}", fnv64(body.as_bytes()))), Some(Err(()))));
    }

    #[test]
    fn graphics_lines_round_trip() {
        let stage = |ps: bool| StageSeen {
            user_sgprs: 12,
            wave64: !ps,
            base_instance: (!ps).then_some(11),
            ps_input_addr: if ps { 0x2 } else { 0 },
            ps_inputs: if ps { vec![PsInput { param: 0, flat: false, default: None }, PsInput { param: 3, flat: true, default: Some(2) }] } else { Vec::new() },
            mrt: std::array::from_fn(|k| MrtSpec { export_format: if ps && k < 2 { 9 } else { 0 }, class: if k == 1 { 2 } else { 0 } }),
            early_tests: ps,
            early_ok: ps,
            lod_measure: ps,
            resources: vec![ResSpec::Buffer { format: 0, dst_sel: 0 }, ResSpec::Image { dim: ImageDim::D2, class: NumClass::Float, depth: false }, ResSpec::Sampler],
        };
        let r = GfxRecipe {
            vs: 0x1111,
            vs_seen: stage(false),
            ps: 0x2222,
            ps_seen: Some(stage(true)),
            state: super::super::exec::GfxState { rect: true, colors: [44, 37, 0, 0, 0, 0, 0, 0], blend: [0x4000_0123, 0, 0, 0, 0, 0, 0, 0], masks: [15, 7, 0, 0, 0, 0, 0, 0], depth: 130, topology_class: 2, depth_clamp: true },
        };
        let line = format_gfx(&r);
        let back = parse_gfx(line.trim_end().strip_prefix("G ").unwrap()).unwrap();
        assert!(back == r);
        let mut nops = r.clone();
        nops.ps = 0;
        nops.ps_seen = None;
        let line = format_gfx(&nops);
        assert!(parse_gfx(line.trim_end().strip_prefix("G ").unwrap()).unwrap() == nops);
    }
}
