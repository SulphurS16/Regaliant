//! Translations kept across runs (`<cache>/translations/<translator hash>/
//! <plan key>-<spec hash>.tr`, deflated). A warm boot translated every
//! program the title loads again: ~75 s of the precompile threads' CPU a
//! boot besides their pipelines' creation, and an area's new programs at
//! its arrival (FINDINGS.md 2026-11-04 afternoon). A translation's output
//! depends only on the program's code and analysis (the plan key holds its
//! code hash and user-data layout), the spec (its hash; the program's
//! address is not in it: `Spec::code_base`) and the translator's code
//! (`rdna::SOURCE_HASH` names the store; stores of other hashes are removed
//! at startup). Off while a switch the translator reads from the
//! environment is set (`TRANSLATOR_ENV`); `DS_TRCACHE=0`: off.
//! `DS_TRCACHE_VERIFY=1`: each translation loaded is made again and compared
//! (status line).

use parking_lot::{Condvar, Mutex};
use rdna::translate::{Binding, BindingKind, Output};
use std::path::PathBuf;
use std::sync::OnceLock;
use std::sync::atomic::{AtomicU64, Ordering};

static DIR: OnceLock<PathBuf> = OnceLock::new();

const MAGIC: &[u8; 8] = b"DSTRANS4";

/// Environment switches the translator reads (`rdna`; the dumps write files
/// while translating): any of them set keeps the store off.
const TRANSLATOR_ENV: [&str; 17] = [
    "DS_EXEC_PARTIAL",
    "DS_LOOP_LIMIT",
    "DS_LOCKSTEP",
    "DS_LOCKSTEP_OFF",
    "DS_MASKED_SKIP",
    "DS_MASK_VALUES",
    "DS_SGWAVE",
    "DS_WAVEMASK",
    "DS_TR_OPS16",
    "RDNA_KNOWN_ACTIVE",
    "RDNA_LANE_GUARDS",
    "RDNA_SGPR_UBO",
    "RDNA_SG_APPENDS",
    "RDNA_KERNEL_DUMP",
    "RDNA_MARKS_DUMP",
    "RDNA_RT_FORMAT_DUMP",
    "RDNA_D3OR2D_DUMP",
];

/// [loaded, their ns, stored, bytes stored, not in the store, unreadable,
/// verified equal, verified different].
static STATS: [AtomicU64; 8] = [const { AtomicU64::new(0) }; 8];

/// Translations waiting for the `trcache-write` thread: (file, encoded).
static QUEUE: Mutex<Vec<(PathBuf, Vec<u8>)>> = Mutex::new(Vec::new());
static QUEUED: Condvar = Condvar::new();
static PENDING: AtomicU64 = AtomicU64::new(0);

/// The first translator switch set in the environment (`TRANSLATOR_ENV`):
/// translations then differ from the stores' under the same keys, so the
/// translation store is off and the pipeline store (keyed by program and
/// spec, not by the SPIR-V) is off too unless `DS_PIPELINE_DIR` names one.
pub fn translator_switch() -> Option<&'static str> {
    TRANSLATOR_ENV.iter().copied().find(|v| std::env::var_os(v).is_some())
}

/// Where the store lives (set once at startup; none: translations are not
/// kept). Removes the stores of other translator hashes in the background.
pub fn set_dir(base: PathBuf) {
    if std::env::var("DS_TRCACHE").ok().as_deref() == Some("0") {
        crate::info!("translation store: off (DS_TRCACHE=0)");
        return;
    }
    if let Some(v) = translator_switch() {
        crate::info!("translation store: off while {v} is set");
        return;
    }
    let dir = base.join(rdna::SOURCE_HASH);
    if std::fs::create_dir_all(&dir).is_err() || DIR.set(dir.clone()).is_err() {
        return;
    }
    let _ = std::thread::Builder::new().name("trcache-prune".into()).spawn(move || {
        let mut removed = 0;
        if let Ok(rd) = std::fs::read_dir(&base) {
            for e in rd.flatten() {
                let p = e.path();
                if p.is_dir() && p != dir && std::fs::remove_dir_all(&p).is_ok() {
                    removed += 1;
                }
            }
        }
        let (mut n, mut bytes) = (0u64, 0u64);
        if let Ok(rd) = std::fs::read_dir(&dir) {
            for e in rd.flatten() {
                // (A write a run did not finish: its temporary file.)
                if e.path().extension().is_some_and(|x| x == "tmp") {
                    let _ = std::fs::remove_file(e.path());
                    continue;
                }
                if let Ok(m) = e.metadata() {
                    n += 1;
                    bytes += m.len();
                }
            }
        }
        crate::info!("translation store {}: {n} translations, {} MiB ({removed} stores of other translator versions removed)", rdna::SOURCE_HASH, bytes >> 20);
    });
}

fn path(plan_key: u64, spec_hash: u64) -> Option<PathBuf> {
    DIR.get().map(|d| d.join(format!("{plan_key:016x}-{spec_hash:016x}.tr")))
}

/// `DS_TRCACHE_VERIFY=1`: translations loaded are made again and compared.
pub fn verify() -> bool {
    static V: OnceLock<bool> = OnceLock::new();
    *V.get_or_init(|| std::env::var("DS_TRCACHE_VERIFY").ok().as_deref() == Some("1"))
}

/// The stored translation of (`plan_key`, `spec_hash`), if any.
pub fn load(plan_key: u64, spec_hash: u64) -> Option<Output> {
    let p = path(plan_key, spec_hash)?;
    let t = std::time::Instant::now();
    let Ok(z) = std::fs::read(&p) else {
        STATS[4].fetch_add(1, Ordering::Relaxed);
        return None;
    };
    let out = miniz_oxide::inflate::decompress_to_vec(&z).ok().and_then(|b| decode(&b, plan_key, spec_hash));
    if out.is_some() {
        STATS[0].fetch_add(1, Ordering::Relaxed);
        STATS[1].fetch_add(t.elapsed().as_nanos() as u64, Ordering::Relaxed);
    } else {
        STATS[5].fetch_add(1, Ordering::Relaxed);
    }
    out
}

/// Keeps `out` as the translation of (`plan_key`, `spec_hash`): deflated and
/// written by the `trcache-write` thread.
pub fn store(plan_key: u64, spec_hash: u64, out: &Output) {
    let Some(p) = path(plan_key, spec_hash) else { return };
    static STARTED: std::sync::Once = std::sync::Once::new();
    STARTED.call_once(|| {
        let _ = std::thread::Builder::new().name("trcache-write".into()).spawn(writer);
    });
    let bytes = encode(out, plan_key, spec_hash);
    PENDING.fetch_add(1, Ordering::Release);
    QUEUE.lock().push((p, bytes));
    QUEUED.notify_one();
}

fn writer() {
    loop {
        let items = {
            let mut q = QUEUE.lock();
            while q.is_empty() {
                QUEUED.wait(&mut q);
            }
            std::mem::take(&mut *q)
        };
        for (p, bytes) in items {
            let z = miniz_oxide::deflate::compress_to_vec(&bytes, 1);
            // (A temporary name first: two threads may make the same one.)
            let tmp = p.with_extension(format!("{}.tmp", std::process::id()));
            if std::fs::write(&tmp, &z).and_then(|_| std::fs::rename(&tmp, &p)).is_ok() {
                STATS[2].fetch_add(1, Ordering::Relaxed);
                STATS[3].fetch_add(z.len() as u64, Ordering::Relaxed);
            } else {
                let _ = std::fs::remove_file(&tmp);
            }
            PENDING.fetch_sub(1, Ordering::Release);
        }
    }
}

/// Drain queued AND in-flight writes before Linux's process-lifetime workers end.
/// Work still translating may enqueue later; a bounded shutdown never waits forever.
#[cfg(target_os = "linux")]
pub fn flush() {
    let start = std::time::Instant::now();
    while PENDING.load(Ordering::Acquire) != 0 && start.elapsed() < std::time::Duration::from_secs(5) {
        QUEUED.notify_one();
        std::thread::sleep(std::time::Duration::from_millis(10));
    }
    crate::info!("translation store shutdown: {} writes pending after {}ms", PENDING.load(Ordering::Acquire), start.elapsed().as_millis());
}

/// `verify`: the translation loaded against the one made again.
pub fn note_verify(loaded: &Output, fresh: &Output, what: impl FnOnce() -> String) {
    if encode(loaded, 0, 0) == encode(fresh, 0, 0) {
        STATS[6].fetch_add(1, Ordering::Relaxed);
    } else if STATS[7].fetch_add(1, Ordering::Relaxed) < 8 {
        crate::warn!("translation store: {} differs from its translation made again", what());
    }
}

/// One status line ("" when the store is off).
pub fn status() -> String {
    if DIR.get().is_none() {
        return String::new();
    }
    let s: Vec<u64> = STATS.iter().map(|c| c.load(Ordering::Relaxed)).collect();
    let verified = if verify() { format!("; verified {} equal, {} different", s[6], s[7]) } else { String::new() };
    format!(
        "    translation store: {} loaded ({:.0} us each), {} stored ({} MiB), {} not stored yet, {} unreadable{verified}\n",
        s[0],
        s[1] as f64 / s[0].max(1) as f64 / 1000.0,
        s[2],
        s[3] >> 20,
        s[4],
        s[5]
    )
}

fn kind_code(k: BindingKind) -> (u8, usize) {
    match k {
        BindingKind::SgprData => (0, 0),
        BindingKind::SgprUniform => (1, 0),
        BindingKind::Buffer(i) => (2, i),
        BindingKind::Image(i) => (3, i),
        BindingKind::ImageAlt(i) => (4, i),
        BindingKind::StorageImage(i) => (5, i),
        BindingKind::StorageImageMips(i) => (6, i),
        BindingKind::ImageArray(i) => (7, i),
        BindingKind::Sampler(i) => (8, i),
        BindingKind::CompareSampler(i) => (9, i),
        BindingKind::Gds => (10, 0),
        BindingKind::Debug => (11, 0),
        BindingKind::Watchdog => (12, 0),
        BindingKind::LodStats => (13, 0),
        BindingKind::Address(i) => (14, i),
        BindingKind::Marks => (15, 0),
    }
}

fn kind_of(tag: u8, i: usize) -> Option<BindingKind> {
    Some(match tag {
        0 => BindingKind::SgprData,
        1 => BindingKind::SgprUniform,
        2 => BindingKind::Buffer(i),
        3 => BindingKind::Image(i),
        4 => BindingKind::ImageAlt(i),
        5 => BindingKind::StorageImage(i),
        6 => BindingKind::StorageImageMips(i),
        7 => BindingKind::ImageArray(i),
        8 => BindingKind::Sampler(i),
        9 => BindingKind::CompareSampler(i),
        10 => BindingKind::Gds,
        11 => BindingKind::Debug,
        12 => BindingKind::Watchdog,
        13 => BindingKind::LodStats,
        14 => BindingKind::Address(i),
        15 => BindingKind::Marks,
        _ => return None,
    })
}

fn fnv(b: &[u8]) -> u64 {
    b.iter().fold(0xcbf2_9ce4_8422_2325u64, |h, &x| (h ^ x as u64).wrapping_mul(0x100_0000_01b3))
}

/// An output as bytes: the keys, every field, then a hash of them.
fn encode(o: &Output, plan_key: u64, spec_hash: u64) -> Vec<u8> {
    let mut w = Vec::with_capacity(96 + 4 * o.spirv.len() + 16 * o.bindings.len());
    let u32s = |w: &mut Vec<u8>, v: u32| w.extend_from_slice(&v.to_le_bytes());
    w.extend_from_slice(MAGIC);
    w.extend_from_slice(&plan_key.to_le_bytes());
    w.extend_from_slice(&spec_hash.to_le_bytes());
    u32s(&mut w, o.spirv.len() as u32);
    for &x in &o.spirv {
        u32s(&mut w, x);
    }
    u32s(&mut w, o.bindings.len() as u32);
    for b in &o.bindings {
        let (tag, i) = kind_code(b.kind);
        u32s(&mut w, b.binding);
        w.push(tag);
        w.extend_from_slice(&(i as u64).to_le_bytes());
    }
    u32s(&mut w, o.params);
    u32s(&mut w, o.mrt_mask);
    u32s(&mut w, o.param_mask);
    w.push(o.writes_depth as u8);
    u32s(&mut w, o.warnings.len() as u32);
    for s in &o.warnings {
        u32s(&mut w, s.len() as u32);
        w.extend_from_slice(s.as_bytes());
    }
    w.push(o.sgpr_pc.is_some() as u8);
    u32s(&mut w, o.sgpr_pc.unwrap_or(0));
    w.push(o.batched as u8);
    u32s(&mut w, o.masked_skips);
    w.push(o.lockstep as u8);
    w.push(o.wave_masks as u8);
    w.push(o.sg_waves as u8);
    u32s(&mut w, o.sg_appends);
    u32s(&mut w, o.known_active);
    w.extend_from_slice(&o.unchecked_res.to_le_bytes());
    u32s(&mut w, o.in_mask);
    // (`Spec::bary_gs`: its per-vertex locations and barycentrics.)
    match &o.bary_gs {
        Some(io) => {
            w.push(1);
            u32s(&mut w, io.per_vertex.len() as u32);
            for (l, [a, b]) in &io.per_vertex {
                u32s(&mut w, *l);
                u32s(&mut w, *a);
                u32s(&mut w, *b);
            }
            u32s(&mut w, io.bary[0]);
            u32s(&mut w, io.bary[1]);
        }
        None => w.push(0),
    }
    let h = fnv(&w);
    w.extend_from_slice(&h.to_le_bytes());
    w
}

/// `encode`'s inverse for the keys asked (None: another key, a short or
/// damaged file).
fn decode(b: &[u8], plan_key: u64, spec_hash: u64) -> Option<Output> {
    if b.len() < MAGIC.len() + 24 || &b[..8] != MAGIC {
        return None;
    }
    let (body, tail) = b.split_at(b.len() - 8);
    if fnv(body) != u64::from_le_bytes(tail.try_into().ok()?) {
        return None;
    }
    struct R<'a>(&'a [u8]);
    impl<'a> R<'a> {
        fn take(&mut self, n: usize) -> Option<&'a [u8]> {
            if self.0.len() < n {
                return None;
            }
            let (a, rest) = self.0.split_at(n);
            self.0 = rest;
            Some(a)
        }
        fn u8(&mut self) -> Option<u8> {
            self.take(1).map(|a| a[0])
        }
        fn u32(&mut self) -> Option<u32> {
            self.take(4).map(|a| u32::from_le_bytes(a.try_into().unwrap()))
        }
        fn u64(&mut self) -> Option<u64> {
            self.take(8).map(|a| u64::from_le_bytes(a.try_into().unwrap()))
        }
    }
    let mut r = R(&body[8..]);
    if r.u64()? != plan_key || r.u64()? != spec_hash {
        return None;
    }
    let n = r.u32()? as usize;
    let words = r.take(4 * n)?;
    let spirv: Vec<u32> = words.chunks_exact(4).map(|c| u32::from_le_bytes(c.try_into().unwrap())).collect();
    let nb = r.u32()? as usize;
    let mut bindings = Vec::with_capacity(nb.min(4096));
    for _ in 0..nb {
        let binding = r.u32()?;
        let tag = r.u8()?;
        let i = r.u64()? as usize;
        bindings.push(Binding { binding, kind: kind_of(tag, i)? });
    }
    let params = r.u32()?;
    let mrt_mask = r.u32()?;
    let param_mask = r.u32()?;
    let writes_depth = r.u8()? != 0;
    let nw = r.u32()? as usize;
    let mut warnings = Vec::with_capacity(nw.min(256));
    for _ in 0..nw {
        let len = r.u32()? as usize;
        warnings.push(String::from_utf8(r.take(len)?.to_vec()).ok()?);
    }
    let has_pc = r.u8()? != 0;
    let pc = r.u32()?;
    let batched = r.u8()? != 0;
    let masked_skips = r.u32()?;
    let lockstep = r.u8()? != 0;
    let wave_masks = r.u8()? != 0;
    let sg_waves = r.u8()? != 0;
    let sg_appends = r.u32()?;
    let known_active = r.u32()?;
    let unchecked_res = r.u64()?;
    let in_mask = r.u32()?;
    let bary_gs = if r.u8()? != 0 {
        let n = r.u32()? as usize;
        let mut per_vertex = Vec::with_capacity(n.min(32));
        for _ in 0..n {
            per_vertex.push((r.u32()?, [r.u32()?, r.u32()?]));
        }
        Some(rdna::translate::gs::BaryIo { per_vertex, bary: [r.u32()?, r.u32()?] })
    } else {
        None
    };
    if !r.0.is_empty() {
        return None;
    }
    Some(Output {
        spirv,
        bindings,
        params,
        mrt_mask,
        param_mask,
        writes_depth,
        warnings,
        sgpr_pc: has_pc.then_some(pc),
        batched,
        masked_skips,
        lockstep,
        wave_masks,
        sg_waves,
        sg_appends,
        known_active,
        unchecked_res,
        in_mask,
        bary_gs,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn round_trip() {
        let o = Output {
            spirv: vec![0x0723_0203, 1, 2, 3, 0xFFFF_FFFF],
            bindings: vec![Binding { binding: 0, kind: BindingKind::SgprData }, Binding { binding: 3, kind: BindingKind::Buffer(7) }, Binding { binding: 9, kind: BindingKind::CompareSampler(2) }],
            params: 4,
            mrt_mask: 0b101,
            param_mask: 0xF0,
            writes_depth: true,
            warnings: vec!["a warning".into()],
            sgpr_pc: Some(12),
            batched: true,
            masked_skips: 3,
            lockstep: true,
            wave_masks: false,
            sg_waves: true,
            sg_appends: 2,
            known_active: 99,
            unchecked_res: 0b101,
            in_mask: 0b1011,
            bary_gs: Some(rdna::translate::gs::BaryIo { per_vertex: vec![(1, [4, 5])], bary: [6, 7] }),
        };
        let b = encode(&o, 0x1234, 0x5678);
        let d = decode(&b, 0x1234, 0x5678).expect("decodes");
        assert_eq!(encode(&d, 0x1234, 0x5678), b);
        assert_eq!(d.in_mask, 0b1011);
        assert_eq!(d.bary_gs, o.bary_gs);
        assert!(decode(&b, 0x1234, 0x5679).is_none());
        let mut bad = b.clone();
        bad[20] ^= 1;
        assert!(decode(&bad, 0x1234, 0x5678).is_none());
        let z = miniz_oxide::deflate::compress_to_vec(&b, 1);
        assert_eq!(miniz_oxide::inflate::decompress_to_vec(&z).unwrap(), b);
    }
}
