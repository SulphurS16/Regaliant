//! Opt-in serialized capture diagnostic. Never repairs or sanitizes the title's pixels.
//! Before/after reads select the actual cache key and subresource, not the largest image at
//! an address. A first observed NaN is distinguished from a finite -> NaN transition.
use super::{cp::{Cp, Space}, images::{ImageKey, Images}, shaders::{Prepared, Resolved}};
use ash::vk;
use parking_lot::Mutex;
use std::collections::{HashMap, HashSet};
use std::io::Write;
use std::sync::{OnceLock, atomic::{AtomicU64, Ordering}};

pub fn active() -> bool {
    static ON: OnceLock<bool> = OnceLock::new();
    *ON.get_or_init(|| std::env::var("DS_NAN_PROBE").ok().as_deref() == Some("1")) && super::capture::capturing()
}

// Title 01.007.000 depth pyramid -> reduced/dilated fog depth -> scatter/raymarch.
// Start at first dispatch, not one flip later: a next-frame trigger loses the producer.
const EARLY_CS: &[u64] = &[0x100893f100, 0x1008940e00, 0x1008941300, 0x100893e100,
    0x100893e700, 0x100893ec00, 0x100892a100, 0x100892b200, 0x1008935200, 0x1008936600];
static PHASE: AtomicU64 = AtomicU64::new(0);
pub(super) fn early_on() -> bool {
    static ON: OnceLock<bool> = OnceLock::new();
    *ON.get_or_init(|| std::env::var("DS_NAN_EARLY").ok().as_deref() == Some("1"))
}

pub fn early_phase(name: &str) {
    if early_on() {
        let phase = PHASE.fetch_add(1, Ordering::Relaxed) + 1;
        crate::info!("early NaN probe phase {phase}: {name} (first two uses per program/output layout, at most 64 packets)");
    }
}

#[derive(Default)]
struct Budget { used: HashMap<(u64, u64, Vec<(ImageKey, u32, u32)>), u32>, totals: HashMap<u64, u32> }
impl Budget {
    fn take(&mut self, phase: u64, shader: u64, keys: Vec<(ImageKey, u32, u32)>) -> bool {
        if keys.is_empty() { return false; }
        let total = self.totals.entry(phase).or_default();
        let seen = self.used.entry((phase, shader, keys)).or_default();
        if *total >= 64 || *seen >= 2 { return false; }
        *total += 1; *seen += 1; true
    }
}

pub fn early(cp: &Cp, dispatch: bool) -> bool {
    if !early_on() { return false; }
    let lo = if dispatch { 0x20c } else { 0x008 };
    let addr = ((cp.reg(Space::Sh, lo) as u64) << 8) | ((cp.reg(Space::Sh, lo+1) as u64 & 255) << 40);
    if if dispatch { !EARLY_CS.contains(&addr) } else { addr != 0x1008937e00 } { return false; }
    // Resolve only the descriptors here. Readbacks and input snapshots happen after the
    // caller flushes the queue. No queue flush for a program/layout whose budget is spent.
    let Ok(p) = (if dispatch { super::shaders::prepare_compute(cp, true) } else { super::shaders::prepare_ps(cp) }) else { return false; };
    let mut keys = Vec::new();
    for (i, r) in p.resources.iter().enumerate() {
        if p.tr.plan.resources.get(i).is_some_and(|r| r.written) {
            if let Resolved::Image(t) = r { if let Some(r) = image_ref(t, true) { keys.push((r.key, r.level, r.slice)); } }
        }
    }
    if !dispatch {
        let mask = cp.reg(Space::Ctx, (0x28238 - 0x28000) / 4);
        for i in 0..8 {
            if mask >> (4*i) & 15 == 0 { continue; }
            if let Ok(Some(s)) = super::exec::color_surface(cp, i) {
                let r = ImageRef::surface(s); keys.push((r.key, r.level, r.slice));
            }
        }
    }
    static BUDGET: Mutex<Option<Budget>> = Mutex::new(None);
    BUDGET.lock().get_or_insert_with(Default::default).take(PHASE.load(Ordering::Relaxed), addr, keys)
}

/// IEEE encodings only. Integer targets may contain arbitrary packed bits, not float NaNs.
fn fields(format: vk::Format) -> Option<(usize, Vec<(u32, u32, u32)>)> {
    use vk::Format as F;
    let (bytes, channels) = match format {
        F::R16_SFLOAT => (2, 1), F::R16G16_SFLOAT => (2, 2), F::R16G16B16A16_SFLOAT => (2, 4),
        F::R32_SFLOAT | F::D32_SFLOAT | F::D32_SFLOAT_S8_UINT => (4, 1),
        F::R32G32_SFLOAT => (4, 2), F::R32G32B32_SFLOAT => (4, 3), F::R32G32B32A32_SFLOAT => (4, 4),
        F::B10G11R11_UFLOAT_PACK32 => return Some((4, vec![(0, 0x7c0, 0x3f), (11, 0x7c0, 0x3f), (22, 0x3e0, 0x1f)])),
        _ => return None,
    };
    Some((bytes * channels, (0..channels).map(|c| (c as u32 * bytes as u32 * 8, if bytes == 2 { 0x7c00 } else { 0x7f800000 }, if bytes == 2 { 0x3ff } else { 0x7fffff })).collect()))
}

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct Counts { pub nan: u64, pub inf: u64, pub zero: u64, pub first: Option<(usize, usize, u32)> }

pub fn count(format: vk::Format, raw: &[u8]) -> Option<Counts> { scan(format, raw, None) }

fn scan(format: vk::Format, raw: &[u8], mut nan_mask: Option<&mut Vec<u64>>) -> Option<Counts> {
    let (texel, fields) = fields(format)?;
    let packed = format == vk::Format::B10G11R11_UFLOAT_PACK32;
    let mut result = Counts::default();
    if let Some(mask) = nan_mask.as_mut() { mask.clear(); mask.resize((raw.len() / texel * fields.len()).div_ceil(64), 0); }
    for (pixel, p) in raw.chunks_exact(texel).enumerate() {
        for (channel, &(shift, exponent, mantissa)) in fields.iter().enumerate() {
            let bits = if packed { u32::from_le_bytes(p[..4].try_into().unwrap()) >> shift }
                else if exponent == 0x7c00 { u16::from_le_bytes(p[shift as usize / 8..shift as usize / 8 + 2].try_into().unwrap()) as u32 }
                else { u32::from_le_bytes(p[shift as usize / 8..shift as usize / 8 + 4].try_into().unwrap()) };
            if bits & (exponent | mantissa) == 0 { result.zero += 1; }
            if bits & exponent == exponent {
                if bits & mantissa != 0 {
                    result.nan += 1;
                    if let Some(mask) = nan_mask.as_mut() {
                        let component = pixel * fields.len() + channel;
                        mask[component / 64] |= 1 << (component % 64);
                    }
                    result.first.get_or_insert((pixel, channel, bits));
                } else { result.inf += 1; }
            }
        }
    }
    Some(result)
}

#[derive(Clone, Debug)]
struct ImageRef { key: ImageKey, level: u32, slice: u32, attachment: bool, storage: bool }
impl ImageRef {
    fn surface(s: super::images::Surface) -> Self {
        Self { key: ImageKey { base: s.base, format: s.format, width: s.width, height: s.height,
            layers: s.layers.max(1), levels: s.levels.max(1), ty3d: s.ty3d, cube: false },
            level: s.mip, slice: s.slice, attachment: true, storage: false }
    }
    fn read(&self) -> Result<(ImageKey, Vec<u8>), String> {
        super::images::with(|im| im.diagnostic_read(self.key, self.level, self.slice, self.attachment, self.storage))
    }
}
struct Sample { target: ImageRef, before: Result<(ImageKey, Counts, Vec<u64>), String> }
struct Buffer { name: String, base: u64, total: u64, bytes: Vec<u8> }
pub struct Probe {
    id: u64, flip: u64, packet: u64, opcode: u32, early: bool, phase: u64, kind: &'static str,
    targets: Vec<Sample>, prepared: Vec<Prepared>, buffers: Vec<Buffer>, details: Vec<String>,
}

fn image_ref(t: &super::desc::Tsharp, storage: bool) -> Option<ImageRef> {
    Some(ImageRef { key: Images::key_of(t)?, level: t.base_level, slice: t.base_array, attachment: false, storage })
}

pub fn before(cp: &Cp, dispatch: bool, packet: u64, opcode: u32, index: (u64, u64), early: bool) -> Probe {
    static IDS: AtomicU64 = AtomicU64::new(0);
    let mut probe = Probe { id: IDS.fetch_add(1, Ordering::Relaxed), flip: super::capture::flips(), packet, opcode, early, phase: PHASE.load(Ordering::Relaxed),
        kind: if dispatch { "dispatch" } else { "draw" }, targets: Vec::new(), prepared: Vec::new(), buffers: Vec::new(), details: Vec::new() };
    let mut outputs = Vec::new();
    if !dispatch {
        let mask = cp.reg(Space::Ctx, (0x28238 - 0x28000) / 4);
        for i in 0..8 {
            if mask >> (4*i) & 15 == 0 { continue; }
            if let Ok(Some(s)) = super::exec::color_surface(cp, i) {
                probe.details.push(format!("cb{i}: {s:?}"));
                outputs.push(ImageRef::surface(s));
            }
        }
    }
    if !dispatch && (cp.reg(Space::Ctx, (0x28800 - 0x28000) / 4) & 6 == 6 || cp.reg(Space::Ctx, 0) & 1 != 0) {
        if let Some(s) = super::exec::depth_surface(cp) {
            probe.details.push(format!("depth: {s:?}"));
            outputs.push(ImageRef::surface(s));
        }
    }
    let stages = if dispatch { vec![super::shaders::prepare_compute(cp, true)] }
        else { vec![super::shaders::prepare_ngg(cp), super::shaders::prepare_ps(cp)] };
    for stage in stages {
        match stage {
            Ok(p) => {
                for (i, r) in p.resources.iter().enumerate() {
                    if p.tr.plan.resources[i].written {
                        if let Resolved::Image(t) = r { if let Some(r) = image_ref(t, true) { outputs.push(r); } }
                    }
                }
                probe.prepared.push(p);
            }
            Err(e) => probe.details.push(format!("prepare error: {e}")),
        }
    }
    let mut seen = HashSet::new();
    for target in outputs {
        if fields(target.key.format).is_none() || !seen.insert((target.key, target.level, target.slice)) { continue; }
        let before = target.read().and_then(|(key, raw)| {
            let mut mask = Vec::new();
            scan(target.key.format, &raw, Some(&mut mask)).map(|c| (key, c, mask)).ok_or("requested view format is not float".into())
        });
        probe.targets.push(Sample { target, before });
    }
    // Constants and vertex streams as the CPU saw them BEFORE this operation, up to 64 KiB
    // per resource. Keep them only until the after-check decides whether to write a report.
    if !probe.targets.is_empty() {
        let mut requests = vec![("indices".to_string(), index.0, index.1.min(64 << 10))];
        for p in &probe.prepared {
            for (i, r) in p.resources.iter().enumerate() {
                match r {
                    Resolved::Buffer(v) => requests.push((format!("shader-{:x}-resource-{i}", p.addr), v.base, v.byte_size().unwrap_or(0))),
                    Resolved::Global(base, n) => requests.push((format!("shader-{:x}-global-{i}", p.addr), *base, n.unwrap_or(64 << 10))),
                    _ => {}
                }
            }
        }
        for (name, base, total) in requests.into_iter().take(128) {
            let n = total.min(64 << 10);
            if n > 0 && super::cp::guest_readable(base, n) {
                let bytes = unsafe { std::slice::from_raw_parts(base as *const u8, n as usize) }.to_vec();
                probe.buffers.push(Buffer { name, base, total, bytes });
            }
        }
    }
    if early {
        if let Some(dir) = crate::run_dir().map(|p| p.join(format!("nan-{}/event-{}/before", probe.flip, probe.id))) {
            crate::info!("early NaN probe phase {} event {} {} packet {:#x} at flip {}", probe.phase, probe.id, probe.kind, probe.packet, probe.flip);
            save_inputs(&dir, &probe, "PRE");
        }
    }
    probe
}

fn nan_changes(before: &[u64], after: &[u64]) -> (u64, u64, Option<usize>) {
    let (mut new, mut cleared, mut first) = (0, 0, None);
    for (i, (&a, &b)) in before.iter().zip(after).enumerate() {
        let added = b & !a;
        new += added.count_ones() as u64;
        cleared += (a & !b).count_ones() as u64;
        if added != 0 { first.get_or_insert(i * 64 + added.trailing_zeros() as usize); }
    }
    (new, cleared, first)
}

fn counts_json(c: Counts) -> serde_json::Value {
    serde_json::json!({"nan_components": c.nan, "infinite_components": c.inf, "zero_components": c.zero, "first_nan_pixel_channel_bits": c.first})
}

pub fn after(cp: &Cp, probe: Probe) {
    let Some(dir) = crate::run_dir().map(|p| p.join(format!("nan-{}", probe.flip))) else { return };
    if probe.targets.is_empty() { return; }
    let _ = std::fs::create_dir_all(&dir);
    static REPORTS: Mutex<Option<HashSet<(u64, ImageKey, u32, u32, bool)>>> = Mutex::new(None);
    static LOG: Mutex<()> = Mutex::new(());
    let mut records = Vec::new();
    let mut save = probe.early;
    for sample in &probe.targets {
        let result = sample.target.read();
        match result {
            Ok((actual, raw)) => {
                let mut mask = Vec::new();
                let now = scan(sample.target.key.format, &raw, Some(&mut mask));
                let delta = sample.before.as_ref().ok().filter(|(key, _, before)| *key == actual && before.len() == mask.len())
                    .map(|(_, _, before)| nan_changes(before, &mask));
                let onset = delta.is_some_and(|(new, _, _)| new > 0);
                let classification = if onset && sample.before.as_ref().is_ok_and(|(_, c, _)| c.nan == 0) { "FINITE_TO_NAN" }
                    else if onset { "NEW_NAN_COMPONENTS" }
                    else if now.is_some_and(|v| v.nan > 0) { "ALREADY_NAN_OR_UNKNOWN_BEFORE" } else { "finite" };
                let alias = super::images::with(|im| im.at_base_report(actual.base));
                records.push(serde_json::json!({"id": probe.id, "flip": probe.flip, "early": probe.early, "phase": probe.phase, "packet": format!("{:#x}", probe.packet), "opcode": probe.opcode,
                    "queue": cp.name, "kind": probe.kind, "counts_format": format!("{:?}", sample.target.key.format), "requested": format!("{:?}", sample.target), "actual": format!("{actual:?}"),
                    "before": sample.before.as_ref().ok().map(|(_, c, _)| counts_json(*c)), "before_key": sample.before.as_ref().ok().map(|(key, _, _)| format!("{key:?}")),
                    "new_nan_components": delta.map(|v| v.0), "cleared_nan_components": delta.map(|v| v.1), "first_new_nan_component": delta.and_then(|v| v.2),
                    "before_error": sample.before.as_ref().err(), "after": now.map(counts_json), "classification": classification, "cache_incarnations": alias,
                    "shaders": probe.prepared.iter().map(|p| format!("{:#x}", p.addr)).collect::<Vec<_>>(), "draw_result": cp.last_draw}));
                if probe.early || now.is_some_and(|v| v.nan > 0) {
                    let first = {
                        let mut g = REPORTS.lock();
                        let seen = g.get_or_insert_with(Default::default);
                        seen.iter().filter(|k| k.0 == probe.flip && k.4 == onset).count() < 8 && seen.insert((probe.flip, actual, sample.target.level, sample.target.slice, onset))
                    };
                    if probe.early || first {
                        save = true;
                        let name = format!("event-{}-{:x}-{}x{}-{:?}-layers{}-m{}-slice{}", probe.id, actual.base, actual.width, actual.height, actual.format, actual.layers, sample.target.level, sample.target.slice);
                        let _ = std::fs::write(dir.join(format!("{name}.raw")), &raw);
                        if !actual.ty3d && (sample.target.attachment || actual.layers == 1) { if let Some(rgba) = super::images::to_rgba8(actual.format, &raw) {
                            let _ = std::fs::write(dir.join(format!("{name}.png")), super::png::encode((actual.width >> sample.target.level).max(1), (actual.height >> sample.target.level).max(1), &rgba));
                        } }
                        crate::info!("NaN probe {classification}: event {} {} packet {:#x}, {actual:?}, before {:?}, after {now:?}", probe.id, probe.kind, probe.packet, sample.before.as_ref().map(|(key, c, _)| (key, c)));
                    }
                }
            }
            Err(e) => records.push(serde_json::json!({"id":probe.id,"flip":probe.flip,"early":probe.early,"phase":probe.phase,"queue":cp.name,"kind":probe.kind,"packet":format!("{:#x}",probe.packet),"shaders":probe.prepared.iter().map(|p| format!("{:#x}", p.addr)).collect::<Vec<_>>(),"target":format!("{:?}",sample.target),"error":e})),
        }
    }
    if save { save_inputs(&dir.join(format!("event-{}", probe.id)), &probe, "POST"); }
    let _lock = LOG.lock();
    if let Ok(mut file) = std::fs::OpenOptions::new().create(true).append(true).open(dir.join("passes.jsonl")) {
        for record in records { let _ = writeln!(file, "{record}"); }
    }
}

fn save_inputs(dir: &std::path::Path, probe: &Probe, when: &str) {
    let _ = std::fs::create_dir_all(dir);
    let mut report = format!("{} event {} opcode {:#x} packet {:#x}, frame {}. Raw files contain native bytes without header. Output geometry/format in passes.jsonl.\nCPU buffers are PRE-operation prefixes (max 64KiB); they are not guaranteed to equal GPU mirror/snapshots. GPU mirror and image inputs in THIS directory are {when}-operation snapshots: read/write aliases may already be modified. Other queues and guest CPU remain live.\n", probe.kind, probe.id, probe.opcode, probe.packet, probe.flip);
    report += &probe.details.join("\n");
    report += "\n";
    for b in &probe.buffers {
        let name = format!("{}-{:x}.bin", b.name, b.base);
        let _ = std::fs::write(dir.join(&name), &b.bytes);
        report += &format!("{name}: {} of {} bytes, f32-bit classification {:?} (vertex/integer formats may not be float)\n", b.bytes.len(), b.total, count(vk::Format::R32_SFLOAT, &b.bytes));
        // Observe only already backed mirror ranges: backing/uploading here would repair
        // the stale-data bug this diagnostic is meant to detect. Ring snapshots may differ.
        if super::resident::backed_now(b.base, b.bytes.len() as u64) {
            match super::resident::readback_now(b.base, b.bytes.len() as u64) {
                Ok(raw) => {
                    let different = b.bytes.iter().zip(&raw).filter(|(a, b)| a != b).count();
                    let _ = std::fs::write(dir.join(format!("mirror-{when}-{name}")), &raw);
                    report += &format!("  {when}-operation GPU mirror: {different} bytes differ from PRE-operation CPU prefix. This is not necessarily the actual bound ring/snapshot; writes by the operation or other threads can explain differences.\n");
                }
                Err(e) => report += &format!("  GPU mirror read failed: {e}\n"),
            }
        }
    }
    let mut images = HashSet::new();
    let mut image_bytes = 0usize;
    for p in &probe.prepared {
        let saved = super::capture::save_program(p.addr);
        let assembly = p.insts.iter().map(|i| rdna::disasm(i)).collect::<Vec<_>>().join("\n");
        let _ = std::fs::write(dir.join(format!("{:x}.rdna.txt", p.addr)), assembly);
        report += &format!("program {:#x} hash {:#x}, source {:?}, original {:?}; data loads {:?}; read ranges {:?}; unread {:?}\n", p.addr, p.code_hash,
            super::particles::origin(p.code_hash), saved, p.data_loads, p.reads, p.unread);
        for (name, words) in [("sgpr-before-binding", p.sgpr_data.as_slice()), ("user", p.user.words()), ("descriptor-loads", p.words.as_slice())] {
            let bytes: Vec<u8> = words.iter().flat_map(|w| w.to_le_bytes()).collect();
            let _ = std::fs::write(dir.join(format!("{:x}-{name}.bin", p.addr)), bytes);
        }
        let _ = std::fs::write(dir.join(format!("{:x}-spec.txt", p.addr)), p.tr.spec.to_words());
        let spirv: Vec<u8> = p.tr.out.spirv.iter().flat_map(|w| w.to_le_bytes()).collect();
        let _ = std::fs::write(dir.join(format!("{:x}-{:x}.spv", p.addr, p.tr.key)), spirv);
        report += &format!("shader {:#x}, translation {:#x}\n{}\n", p.addr, p.tr.key, super::shaders::describe_resources(p).join("\n"));
        let mut textures = Vec::new();
        for (i, r) in p.resources.iter().enumerate() {
            let storage = p.tr.plan.resources.get(i).is_some_and(|r| r.written);
            match r { Resolved::Image(t) => textures.push((*t, storage)), Resolved::ImageTable(ts) => textures.extend(ts.iter().flatten().map(|t| (*t, false))), _ => {} }
        }
        for (t, storage) in textures {
            let Some(r) = image_ref(&t, storage) else { continue };
            if !images.insert((r.key, r.level, r.slice)) { continue; }
            // Bound total output for bindless tables while explicitly reporting omitted data.
            if images.len() > 32 { report += &format!("image {:#x}: omitted, 32-image report limit\n", t.base); continue; }
            let desc = super::images::with(|im| im.describe(&t));
            match r.read() {
                Ok((key, raw)) => {
                    if image_bytes + raw.len() > 128 << 20 { report += &format!("image {:#x}: omitted, 128 MiB input-image report limit\n", t.base); continue; }
                    image_bytes += raw.len();
                    let name = format!("input-{:x}-{}x{}-{:?}-m{}-slice{}.raw", t.base, key.width, key.height, key.format, r.level, r.slice);
                    let _ = std::fs::write(dir.join(&name), &raw);
                    report += &format!("{name}: requested {:?}; selected {key:?}; {} bytes; {:?}; {desc}\n", r.key, raw.len(), count(r.key.format, &raw));
                }
                Err(e) => report += &format!("image {:#x}: {e}; {desc}\n", t.base),
            }
        }
    }
    let _ = std::fs::write(dir.join("inputs.txt"), report);
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn first_use_budget_rearms_for_level_transition_and_layout_changes() {
        let mut budget = Budget::default();
        let key = ImageKey { base: 0x1000, width: 214, height: 120, layers: 72, levels: 1,
            format: vk::Format::R16G16B16A16_SFLOAT, ty3d: true, cube: false };
        assert!(!budget.take(0, 1, vec![]));
        assert!(budget.take(0, 1, vec![(key, 0, 0)]));
        assert!(budget.take(0, 1, vec![(key, 0, 0)]));
        assert!(!budget.take(0, 1, vec![(key, 0, 0)]));
        assert!(budget.take(1, 1, vec![(key, 0, 0)]));
        assert!(budget.take(0, 1, vec![(ImageKey { width: 320, ..key }, 0, 0)]));
        for shader in 2..63 { assert!(budget.take(0, shader, vec![(key, 0, 0)])); }
        assert!(!budget.take(0, 100, vec![(key, 0, 0)]));
        assert!(budget.take(2, 100, vec![(key, 0, 0)]));
    }

    #[test]
    fn zeros_include_both_signs_but_not_subnormals_or_nan_sentinels() {
        let raw: Vec<u8> = [0u16, 0x8000, 1, 0x8001, 0x7fff].iter().flat_map(|x| x.to_le_bytes()).collect();
        let c = count(vk::Format::R16_SFLOAT, &raw).unwrap();
        assert_eq!((c.zero, c.nan, c.inf), (2, 1, 0));
    }

    #[test]
    fn detects_new_nan_locations_even_when_total_count_is_unchanged() {
        let mut before = Vec::new();
        let mut after = Vec::new();
        let a: Vec<u8> = [f32::NAN, 0.0].iter().flat_map(|x| x.to_le_bytes()).collect();
        let b: Vec<u8> = [0.0, f32::NAN].iter().flat_map(|x| x.to_le_bytes()).collect();
        assert_eq!(scan(vk::Format::R32_SFLOAT, &a, Some(&mut before)).unwrap().nan, 1);
        assert_eq!(scan(vk::Format::R32_SFLOAT, &b, Some(&mut after)).unwrap().nan, 1);
        assert_eq!(nan_changes(&before, &after), (1, 1, Some(1)));
        assert_eq!(nan_changes(&after, &after), (0, 0, None));
    }

    #[test]
    fn nan_counts_are_format_aware_and_preserve_infinity() {
        let raw: Vec<u8> = [0x0000u16, 0x7c00, 0x7fff, 0xfe00].iter().flat_map(|x| x.to_le_bytes()).collect();
        assert_eq!(count(vk::Format::R16G16_SFLOAT, &raw), Some(Counts { nan: 2, inf: 1, zero: 1, first: Some((1, 0, 0x7fff)) }));
        assert_eq!(count(vk::Format::R32_UINT, &[255; 4]), None);
        let raw: Vec<u8> = [0f32, f32::NAN, f32::NEG_INFINITY, 1.0].iter().flat_map(|x| x.to_le_bytes()).collect();
        let c = count(vk::Format::R32G32B32A32_SFLOAT, &raw).unwrap();
        assert_eq!((c.nan, c.inf, c.first.unwrap().1), (1, 1, 1));
        let packed = (0x7c1u32 | (0x7c0 << 11) | (0x3e1 << 22)).to_le_bytes();
        let c = count(vk::Format::B10G11R11_UFLOAT_PACK32, &packed).unwrap();
        assert_eq!((c.nan, c.inf), (2, 1));
    }
}
