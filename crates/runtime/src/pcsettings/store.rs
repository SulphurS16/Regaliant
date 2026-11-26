//! The PC settings' values, in the title's own settings paths (PPSA01341 01.007.000;
//! FINDINGS.md 2026-11-06 afternoon). The title's settings store (`[[ctx+0x10]+0xA155F0]`, built
//! at eboot+0xbd0120) keeps vectors of 12-byte {key hash, sub-key, value} entries: bools at
//! +0x18, ints at +0x40, floats at +0x90 (begin, end); a key is FNV-1a of the setting's name in
//! upper case. The menu's bindings (`UISetting_<type>::Evaluate`: int eboot+0xc241f0, float
//! +0xc24cc0) scan them and fall back to the item's default; the menu's setters (int
//! eboot+0xfa5cb3, float +0xfa5ede) post {key, sub-key, value} to the store, which keeps only
//! the settings it registers. For the PC keys the port answers the scans (the hooks hand the
//! scan a one-entry store with the port's value, so it takes the title's own "found" path) and
//! takes the setters' values, keeps them in local-data/pcsettings.txt and applies them at once.
//! PC toggles are int choices (Off, On): a bool item also scans the store itself
//! (eboot+0xfa5990) and through further paths, and shows no value without an entry.

use parking_lot::Mutex;
use std::collections::HashMap;
use std::path::PathBuf;
use std::sync::atomic::{AtomicU64, Ordering};

#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord)]
pub enum Kind {
    Int,
    Float,
}

/// A PC setting: its key name, kind and default (the value's 32 bits).
#[derive(Clone, Debug)]
pub struct Def {
    pub name: String,
    pub kind: Kind,
    pub default: u32,
    /// A row that shows the achievements (answered live from them, never stored).
    pub live: Option<Live>,
}

/// What a live row shows.
#[derive(Clone, Copy, Debug)]
pub enum Live {
    /// A trophy unlocked (1) or not.
    Trophy(u32),
    /// The unlocked trophies of a grade (P, G, S, B).
    Count(char),
    /// The achievement pop-up shows (bool).
    Popup,
    /// The text hash of the pop-up's trophy name.
    PopupName,
    /// The text hash of the pop-up's title line.
    PopupTitle,
    /// Whether a set of button prompts is shown (`prompts::shown_setting`).
    Prompts(u8),
    /// A key binding's row: the option of an action's input (`bindings`); a change rebinds.
    Binding(u16),
    /// A controller binding's row: the option of the pad button on an action (`bindings`).
    PadBinding(u16),
    /// The audio device (0: the system's default, k: `audio::output_devices()[k - 1]`; the
    /// choice kept as its endpoint id in local-data/audio-device.txt).
    AudioDevice,
    /// The display choices (`crate::display`): 0 the monitor, 1 the fullscreen resolution, 2
    /// the refresh rate (kept in local-data/display.txt).
    Display(u8),
    /// Anti-Aliasing: the menu shows the session list's index of the code kept
    /// (`super::aa_list`); a change keeps the code.
    AntiAliasing,
    /// The Quality Preset: the preset the settings match (`presets::current`); a change sets
    /// the preset's settings.
    Preset,
    /// Frame Generation: the menu shows the session list's index of the code kept
    /// (`super::fg_list`); a change keeps the code.
    FrameGen,
    /// Frame Generation Mode: the menu shows the session list's index of the multiplier kept
    /// (`super::fg_multipliers`); a change keeps the multiplier.
    FgMultiplier,
}

struct Store {
    path: Option<PathBuf>,
    defs: HashMap<u32, Def>,
    values: HashMap<u32, u32>,
}

static STORE: Mutex<Option<Store>> = Mutex::new(None);
/// What the hooks saw: [-, int scans answered, float, PC values taken, title values seen].
static STATS: [AtomicU64; 5] = [const { AtomicU64::new(0) }; 5];

fn fmt(kind: Kind, bits: u32) -> String {
    match kind {
        Kind::Int => (bits as i32).to_string(),
        Kind::Float => {
            let t = format!("{:.4}", f32::from_bits(bits));
            t.trim_end_matches('0').trim_end_matches('.').to_string()
        }
    }
}

fn parse(kind: Kind, s: &str) -> Option<u32> {
    match kind {
        Kind::Int => match s {
            "true" | "on" => Some(1),
            "false" | "off" => Some(0),
            _ => s.parse::<i32>().ok().map(|v| v as u32),
        },
        Kind::Float => s.parse::<f32>().ok().filter(|v| v.is_finite()).map(f32::to_bits),
    }
}

/// Loads the values (`name = value` lines; unknown names and bad values are logged and left
/// out) for these definitions.
pub fn init(path: PathBuf, defs: Vec<Def>) {
    let mut values = HashMap::new();
    let by_name: HashMap<String, (u32, Kind)> = defs.iter().map(|d| (d.name.clone(), (super::setting_hash(&d.name), d.kind))).collect();
    // Renamed settings: their values carried over unless the new name is set too.
    let mut carried: Vec<(u32, u32)> = Vec::new();
    if let Ok(text) = std::fs::read_to_string(&path) {
        for line in text.lines().map(str::trim).filter(|l| !l.is_empty() && !l.starts_with('#')) {
            let Some((k, v)) = line.split_once('=') else { continue };
            // Sharpening (until 2026-11-14 an amount 0..1): the Sharpness option nearest to it.
            if k.trim() == "Setting_PC_Sharpening" && by_name.contains_key("Setting_PC_Sharpness") {
                if let Some(x) = v.trim().parse::<f32>().ok().filter(|x| x.is_finite()) {
                    let i = super::SHARPNESS.iter().enumerate().min_by(|a, b| (a.1 - x).abs().total_cmp(&(b.1 - x).abs())).map_or(0, |(i, _)| i as u32);
                    carried.push((super::setting_hash("Setting_PC_Sharpness"), i));
                    crate::info!("pc settings: Sharpening {x} carried over as Sharpness option {i}");
                }
                continue;
            }
            match by_name.get(k.trim()) {
                Some(&(h, kind)) => match parse(kind, v.trim()) {
                    Some(bits) => {
                        values.insert(h, bits);
                    }
                    None => crate::warn!("pc settings: {}: bad value '{}' for {k}", path.display(), v.trim()),
                },
                None => crate::warn!("pc settings: {}: unknown setting '{}'", path.display(), k.trim()),
            }
        }
    }
    for (h, v) in carried {
        values.entry(h).or_insert(v);
    }
    let n = values.len();
    let defs: HashMap<u32, Def> = defs.into_iter().map(|d| (super::setting_hash(&d.name), d)).collect();
    crate::info!("pc settings: {} settings, {n} set in {}", defs.len(), path.display());
    *STORE.lock() = Some(Store { path: Some(path), defs, values });
}

/// Safe mode's store (the launcher's, for one start): every setting at its default but those in
/// `set`, nothing read from or written to the settings file.
pub fn init_session(defs: Vec<Def>, set: &[(&str, u32)]) {
    let values: HashMap<u32, u32> = set.iter().map(|(n, v)| (super::setting_hash(n), *v)).collect();
    let defs: HashMap<u32, Def> = defs.into_iter().map(|d| (super::setting_hash(&d.name), d)).collect();
    crate::info!("pc settings: safe mode: {} settings at their defaults for this session ({} set), the settings file left as it is", defs.len(), values.len());
    *STORE.lock() = Some(Store { path: None, defs, values });
}

/// The value of a PC key (its default when unset); `None` for other keys.
pub fn get(hash: u32) -> Option<u32> {
    let g = STORE.lock();
    let s = g.as_ref()?;
    let d = s.defs.get(&hash)?;
    if let Some(Live::Preset) = d.live {
        // Reads other settings (the store's lock released first).
        drop(g);
        return Some(super::presets::current());
    }
    match d.live {
        Some(Live::Trophy(id)) => return Some(crate::achievements::unlocked(id) as u32),
        Some(Live::Count(g)) => return Some(crate::achievements::count(g).0),
        Some(Live::Popup) => return Some(super::popup::visible() as u32),
        Some(Live::PopupName) => return Some(super::popup::name_hash()),
        Some(Live::PopupTitle) => return Some(super::cgpr::fnv1a(super::popup::TITLE_KEY.as_bytes())),
        Some(Live::Prompts(w)) => return Some(super::prompts::shown_setting(w)),
        Some(Live::Binding(a)) => return Some(super::bindings::index_of(a as usize)),
        Some(Live::PadBinding(a)) => return Some(super::bindings::pad_index_of(a as usize)),
        Some(Live::Preset) => unreachable!(),
        Some(Live::AudioDevice) => {
            let id = crate::audio::output_device();
            return Some(id.and_then(|id| crate::audio::output_devices().iter().position(|d| d.0 == id)).map(|k| k as u32 + 1).unwrap_or(0));
        }
        Some(Live::AntiAliasing) => {
            let code = s.values.get(&hash).copied().unwrap_or(d.default);
            drop(g);
            return Some(super::aa_list().iter().position(|(c, _)| *c == code).unwrap_or(0) as u32);
        }
        Some(Live::FrameGen) => {
            let code = s.values.get(&hash).copied().unwrap_or(d.default);
            drop(g);
            return Some(super::fg_list().iter().position(|(c, _)| *c == code).unwrap_or(0) as u32);
        }
        Some(Live::FgMultiplier) => {
            let m = s.values.get(&hash).copied().unwrap_or(d.default);
            drop(g);
            // (A multiplier past this GPU's most shows as its most.)
            let l = super::fg_multipliers();
            return Some(l.iter().position(|x| *x == m).unwrap_or_else(|| l.iter().rposition(|x| *x <= m).unwrap_or(0)) as u32);
        }
        Some(Live::Display(k)) => {
            return Some(match k {
                0 => crate::display::monitor_index(),
                1 => crate::display::resolution_index(),
                _ => crate::display::refresh_index(),
            });
        }
        None => {}
    }
    Some(s.values.get(&hash).copied().unwrap_or(d.default))
}

/// The value kept for a PC setting (its code, not what a live row shows).
pub fn raw_value(name: &str) -> Option<u32> {
    let hash = super::setting_hash(name);
    let g = STORE.lock();
    let s = g.as_ref()?;
    let d = s.defs.get(&hash)?;
    Some(s.values.get(&hash).copied().unwrap_or(d.default))
}

/// The value of a PC setting by name.
pub fn value(name: &str) -> Option<u32> {
    get(super::setting_hash(name))
}

/// Sets a PC setting by name (as the menu does: kept in pcsettings.txt); false when unknown.
pub fn set_value(name: &str, bits: u32) -> bool {
    set(super::setting_hash(name), bits)
}

fn set(hash: u32, bits: u32) -> bool {
    let mut g = STORE.lock();
    let Some(s) = g.as_mut() else { return false };
    let Some(d) = s.defs.get(&hash).cloned() else { return false };
    if let Some(Live::Binding(a)) = d.live {
        // A key binding: the layout changes (the value is the layout's).
        drop(g);
        super::bindings::set(a as usize, bits);
        return true;
    }
    if let Some(Live::PadBinding(a)) = d.live {
        drop(g);
        super::bindings::pad_set(a as usize, bits);
        return true;
    }
    if let Some(Live::Preset) = d.live {
        drop(g);
        super::presets::apply(bits);
        return true;
    }
    if let Some(Live::AudioDevice) = d.live {
        let path = s.path.as_ref().map(|p| p.with_file_name("audio-device.txt"));
        drop(g);
        let id = if bits == 0 { None } else { crate::audio::output_devices().get(bits as usize - 1).map(|d| d.0.clone()) };
        if let Some(p) = path {
            let text = id.clone().unwrap_or_default();
            if let Err(e) = std::fs::write(&p, format!("{text}\n")) {
                crate::warn!("pc settings: write {}: {e}", p.display());
            }
        }
        crate::audio::set_output_device(id);
        return true;
    }
    // Anti-Aliasing: the menu's index becomes the code kept (the normal path below stores it).
    let bits = match d.live {
        Some(Live::AntiAliasing) => super::aa_list().get(bits as usize).map_or(super::AA_TAA, |(c, _)| *c),
        Some(Live::FrameGen) => super::fg_list().get(bits as usize).map_or(crate::gpu::framegen::FG_OFF, |(c, _)| *c),
        Some(Live::FgMultiplier) => super::fg_multipliers().get(bits as usize).copied().unwrap_or(2),
        _ => bits,
    };
    if let Some(Live::Display(k)) = d.live {
        drop(g);
        match k {
            0 => crate::display::set_monitor_index(bits),
            1 => crate::display::set_resolution_index(bits),
            _ => crate::display::set_refresh_index(bits),
        }
        return true;
    }
    if d.live.is_some() && !matches!(d.live, Some(Live::AntiAliasing | Live::FrameGen | Live::FgMultiplier)) {
        // A row showing the achievements; the menu cannot change it.
        return true;
    }
    if s.values.get(&hash) == Some(&bits) {
        return true;
    }
    s.values.insert(hash, bits);
    crate::info!("pc settings: {} = {}", d.name, fmt(d.kind, bits));
    if let Some(p) = &s.path {
        let mut names: Vec<(&String, Kind, u32)> = s.values.iter().filter_map(|(h, v)| s.defs.get(h).map(|d| (&d.name, d.kind, *v))).collect();
        names.sort();
        let text: String = std::iter::once("# PC settings (the game's SETTINGS screen, PC tab); written by the port\n".to_string())
            .chain(names.iter().map(|(n, k, v)| format!("{n} = {}\n", fmt(*k, *v))))
            .collect();
        if let Err(e) = std::fs::write(p, text) {
            crate::warn!("pc settings: write {}: {e}", p.display());
        }
    }
    true
}

fn reg(regs: *const u64, i: usize) -> u64 {
    unsafe { *regs.add(i) }
}

fn set_reg(regs: *const u64, i: usize, v: u64) {
    unsafe { *(regs as *mut u64).add(i) = v };
}

/// A store of one entry for a scan the port answers: each vector (bools +0x18, ints +0x40,
/// floats +0x90) holds the same {key, sub-key, value}, so the scan takes the title's own
/// "found" path with the PC value. One a thread (the scan follows its hook at once).
#[repr(C, align(16))]
struct OneEntry {
    store: [u64; 0x28],
    entry: [u32; 4],
}

thread_local! {
    static ONE: std::cell::UnsafeCell<Box<OneEntry>> = std::cell::UnsafeCell::new(Box::new(OneEntry { store: [0; 0x28], entry: [0; 4] }));
}

/// The address of this thread's one-entry store holding {key, sub-key, value}.
fn one_entry(key: u32, sub: u32, value: u32) -> u64 {
    ONE.with(|c| {
        let o = unsafe { &mut **c.get() };
        o.entry = [key, sub, value, 0];
        let e = o.entry.as_ptr() as u64;
        for (begin, end) in [(0x18, 0x20), (0x40, 0x48), (0x90, 0x98)] {
            o.store[begin / 8] = e;
            o.store[end / 8] = e + 12;
        }
        o.store.as_ptr() as u64
    })
}

/// The title's settings store as its `Evaluate` hooks last saw it (0: not yet).
static TITLE_STORE: AtomicU64 = AtomicU64::new(0);

/// A title setting's int value from the title's own store (its first entry of the key; None
/// before a menu or the HUD evaluated a setting, or when the title has no entry for it).
#[allow(dead_code)]
pub fn title_int(name: &str) -> Option<u32> {
    let store = TITLE_STORE.load(Ordering::Relaxed);
    if store < 0x10000 {
        return None;
    }
    let key = super::setting_hash(name);
    let (begin, end) = unsafe { (((store + 0x40) as *const u64).read_volatile(), ((store + 0x48) as *const u64).read_volatile()) };
    if begin < 0x10000 || end < begin || end - begin > 12 * 4096 || (end - begin) % 12 != 0 {
        return None;
    }
    (begin..end).step_by(12).find(|&e| unsafe { (e as *const u32).read_volatile() } == key).map(|e| unsafe { ((e + 8) as *const u32).read_volatile() })
}

/// A scan of the store for `key` (with `sub` its sub-key) in register `store_reg`: for a PC key
/// the register is pointed at a one-entry store with the port's value. (Before that, the
/// register holds the title's own store: noted for `title_int`.)
fn answer(regs: *const u64, key: u32, sub: u32, store_reg: usize, stat: usize) {
    let store = reg(regs, store_reg);
    if store >= 0x10000 {
        TITLE_STORE.store(store, Ordering::Relaxed);
    }
    if let Some(v) = get(key) {
        STATS[stat].fetch_add(1, Ordering::Relaxed);
        set_reg(regs, store_reg, one_entry(key, sub, v));
    }
}

/// `UISetting_bool::Evaluate` before its scan (eboot+0xc3a8e5: r12d the key, ebx the sub-key,
/// r13 the store): the achievement pop-up's visibility (`popup`).
pub extern "sysv64" fn eval_bool_hook(regs: *const u64, _arg: u32) {
    answer(regs, reg(regs, 12) as u32, reg(regs, 3) as u32, 13, 0);
}

/// `UISetting_int::Evaluate` before its scan (eboot+0xc24245; the same registers).
pub extern "sysv64" fn eval_int_hook(regs: *const u64, _arg: u32) {
    answer(regs, reg(regs, 12) as u32, reg(regs, 3) as u32, 13, 1);
}

/// `UISetting_float::Evaluate` before its scan (eboot+0xc24d15; the same registers).
pub extern "sysv64" fn eval_float_hook(regs: *const u64, _arg: u32) {
    answer(regs, reg(regs, 12) as u32, reg(regs, 3) as u32, 13, 2);
}

/// A PC setting by name (`pcset`; the value as in pcsettings.txt).
pub fn set_by_name(name: &str, value: &str) -> Result<(), String> {
    let h = super::setting_hash(name);
    let kind = STORE.lock().as_ref().and_then(|s| s.defs.get(&h).map(|d| d.kind)).ok_or_else(|| format!("no PC setting {name}"))?;
    let bits = parse(kind, value).ok_or_else(|| format!("bad value '{value}' for {name}"))?;
    set(h, bits);
    super::apply();
    Ok(())
}

fn taken(key: u32, bits: u32, kind: &str) {
    if set(key, bits) {
        super::apply();
        STATS[3].fetch_add(1, Ordering::Relaxed);
    } else if STATS[4].fetch_add(1, Ordering::Relaxed) < 64 {
        let name = super::setting_name(key).unwrap_or_else(|| format!("#{key:08x}"));
        crate::info!("pc settings: the menu set {name} ({kind}) = {bits:#x}");
    }
}

/// The int setter (eboot+0xfa5cba: r12d the key, r14d the value).
pub extern "sysv64" fn set_int_hook(regs: *const u64, _arg: u32) {
    taken(reg(regs, 12) as u32, reg(regs, 14) as u32, "int");
}

/// The float setter (eboot+0xfa5ee5: r13d the key, the value at [r15+0x10]).
pub extern "sysv64" fn set_float_hook(regs: *const u64, _arg: u32) {
    let v = unsafe { ((reg(regs, 15) + 0x10) as *const u32).read_volatile() };
    taken(reg(regs, 13) as u32, v, "float");
}

pub fn status() -> String {
    let s: Vec<u64> = STATS.iter().map(|a| a.load(Ordering::Relaxed)).collect();
    if s.iter().all(|&v| v == 0) {
        return String::new();
    }
    format!("    pc settings: scans answered (int/float) {}/{}, PC values taken {}, title values seen {}\n", s[1], s[2], s[3], s[4])
}
