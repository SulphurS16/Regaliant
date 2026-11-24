//! Button prompts for the keyboard and mouse and for Xbox pads, drawn by the title's own glyph
//! system. The title's text shows a button as `#e<hash>#`; the text renderer (eboot+0x150e380)
//! maps that hash to the name hash (FNV-1a of the upper-cased name) of one of the `ButtonIcon_*`
//! entities under the global UI scene (`ui/scripts/_cmn/uiglobal.cgpr`, `GlobalHandler`) and
//! draws that entity's Quad texture inline (`ui/textures/buttonicons/_ps5/ps4_*.ctxr`).
//!
//! The port adds an entity per glyph of its own to that scene (copies of `ButtonIcon_X` with
//! their Quad on a texture the port serves: `pc_<id>.ctxr` for keys, mouse buttons and the
//! wheel, `xb_<id>.ctxr` for an Xbox pad; `glyphs` has the file layout) and, in the renderer
//! after its map (`glyph_hook`), turns the PlayStation entity into the one of the device in use:
//! a pad button's first keyboard or mouse binding (controls.txt), or the Xbox button the pad
//! layer maps to it. The PC tab's Button Prompts picks the set (Automatic: the device used
//! last). Sticks and the d-pad keep the title's glyphs on an Xbox pad (they show no
//! PlayStation marks).
//!
//! The art is the port's own, in the title's glyph style (measured from its glyphs: white
//! faces with dark symbols, dark shoulder shapes with white outlines and labels, the touch
//! pad's teal for a pressed part): key caps with the key's name, a mouse with the button
//! lit, Xbox face buttons with their letters. Labels are drawn with the system's Arial (GDI).

use super::{cgpr, glyphs};
use crate::input::{self, PromptInput};
use std::collections::{BTreeMap, HashMap};
use std::sync::OnceLock;
use std::sync::atomic::{AtomicU8, Ordering::Relaxed};

pub const DIR: &str = "/app0/ui/textures/buttonicons/_ps5/";
const TEMPLATE: &str = "ps4_button_cross.ctxr";
const TEMPLATE_PATH: &str = "$/ui/textures/buttonicons/****/ps4_button_cross.ctxr";

/// The title's glyph entities and the layout target each stands for.
const PS: &[(&str, &str)] = &[
    ("ButtonIcon_X", "cross"),
    ("ButtonIcon_Circle", "circle"),
    ("ButtonIcon_Square", "square"),
    ("ButtonIcon_Triangle", "triangle"),
    ("ButtonIcon_L1", "l1"),
    ("ButtonIcon_R1", "r1"),
    ("ButtonIcon_L2", "l2"),
    ("ButtonIcon_R2", "r2"),
    ("ButtonIcon_L3", "l3"),
    ("ButtonIcon_R3", "r3"),
    ("ButtonIcon_LS", "ls"),
    ("ButtonIcon_RS", "rs"),
    ("ButtonIcon_Dpad_U", "up"),
    ("ButtonIcon_Dpad_D", "down"),
    ("ButtonIcon_Dpad_L", "left"),
    ("ButtonIcon_Dpad_R", "right"),
    ("ButtonIcon_Dpad_UD", "updown"),
    ("ButtonIcon_Dpad_LR", "leftright"),
    ("ButtonIcon_Options", "options"),
    ("ButtonIcon_Share", "share"),
    ("ButtonIcon_TouchPad", "touchpad"),
    ("ButtonIcon_TouchPad_Left", "touchpad"),
    ("ButtonIcon_TouchPad_Right", "touchpad"),
    ("ButtonIcon_TouchPad_SwipeRight", "touchpad"),
];

/// Per pad button (`input::PAD_BUTTONS`): its Xbox glyph (None: the d-pad, drawn with the
/// title's own) and the title's glyph of the PS5 pad's button in its place.
const PAD_GLYPHS: [(Option<&str>, &str); 16] = [
    (Some("xb_a"), "ButtonIcon_X"),
    (Some("xb_b"), "ButtonIcon_Circle"),
    (Some("xb_x"), "ButtonIcon_Square"),
    (Some("xb_y"), "ButtonIcon_Triangle"),
    (Some("xb_lb"), "ButtonIcon_L1"),
    (Some("xb_rb"), "ButtonIcon_R1"),
    (Some("xb_lt"), "ButtonIcon_L2"),
    (Some("xb_rt"), "ButtonIcon_R2"),
    (Some("xb_ls"), "ButtonIcon_L3"),
    (Some("xb_rs"), "ButtonIcon_R3"),
    (Some("xb_view"), "ButtonIcon_TouchPad"),
    (Some("xb_menu"), "ButtonIcon_Options"),
    (None, "ButtonIcon_Dpad_U"),
    (None, "ButtonIcon_Dpad_D"),
    (None, "ButtonIcon_Dpad_L"),
    (None, "ButtonIcon_Dpad_R"),
];

/// The pad button on a target (the layout's controller map: A on cross, B on circle, ...,
/// unless remapped); None for sticks, d-pad pairs and share.
fn pad_button(target: &str) -> Option<usize> {
    input::pad_button_of(input::button_bit(target)?)
}

/// The Xbox glyph of a target: the pad button on it (None: the title's own glyph: sticks, a
/// d-pad button, share).
fn xbox_id(target: &str) -> Option<&'static str> {
    PAD_GLYPHS[pad_button(target)?].0
}

/// What a glyph shows.
#[derive(Clone, Debug, PartialEq)]
enum Art {
    /// A key cap with a label.
    Key(String),
    /// An arrow key (0 up, 1 down, 2 left, 3 right).
    Arrow(u8),
    /// Two arrow keys side by side.
    ArrowPair(u8, u8),
    /// Two key caps side by side.
    KeyPair(String, String),
    /// A mouse with a button lit (0 left, 1 right, 2 middle, 3 and 4 the side buttons).
    Mouse(u8),
    /// The wheel turned up (true) or down.
    Wheel(bool),
    /// The mouse moved (camera).
    MouseMove,
    /// Four keys (up, left, down, right): the left stick on the keyboard.
    Cluster([String; 4]),
    /// A modifier key in front of a glyph.
    Combo(String, Box<Art>),
    XFace(char),
    XBumper(&'static str),
    XTrigger(&'static str),
    XStick(&'static str),
    XMenu,
    XView,
}

fn key_label(vk: u8) -> String {
    match vk {
        b'A'..=b'Z' | b'0'..=b'9' => (vk as char).to_string(),
        0x70..=0x7B => format!("F{}", vk - 0x6F),
        0x20 => "SPACE".into(),
        0x0D => "ENTER".into(),
        0x1B => "ESC".into(),
        0x09 => "TAB".into(),
        0x08 => "BKSP".into(),
        0x10 | 0xA0 | 0xA1 => "SHIFT".into(),
        0x11 | 0xA2 | 0xA3 => "CTRL".into(),
        0x12 | 0xA4 | 0xA5 => "ALT".into(),
        0x14 => "CAPS".into(),
        0x2D => "INS".into(),
        0x2E => "DEL".into(),
        0x24 => "HOME".into(),
        0x23 => "END".into(),
        0x21 => "PG UP".into(),
        0x22 => "PG DN".into(),
        _ => format!("{vk:02X}"),
    }
}

fn arrow_of(vk: u8) -> Option<u8> {
    match vk {
        0x26 => Some(0),
        0x28 => Some(1),
        0x25 => Some(2),
        0x27 => Some(3),
        _ => None,
    }
}

fn input_art(i: PromptInput) -> Art {
    match i {
        PromptInput::Key(k) => arrow_of(k).map(Art::Arrow).unwrap_or_else(|| Art::Key(key_label(k))),
        PromptInput::Mouse(m) => Art::Mouse(m),
        PromptInput::WheelUp => Art::Wheel(true),
        PromptInput::WheelDown => Art::Wheel(false),
    }
}

fn input_id(i: PromptInput, mods: u8) -> String {
    let base = match i {
        PromptInput::Key(k) => format!("k{k:02x}"),
        PromptInput::Mouse(m) => format!("m{m}"),
        PromptInput::WheelUp => "wu".into(),
        PromptInput::WheelDown => "wd".into(),
    };
    let mut pre = String::new();
    for (bit, c) in [(input::PROMPT_SHIFT, 's'), (input::PROMPT_CTRL, 'c'), (input::PROMPT_ALT, 'a')] {
        if mods & bit != 0 {
            pre.push(c);
        }
    }
    if pre.is_empty() { base } else { format!("{pre}_{base}") }
}

fn mod_label(mods: u8) -> String {
    let mut v = Vec::new();
    if mods & input::PROMPT_CTRL != 0 {
        v.push("CTRL");
    }
    if mods & input::PROMPT_ALT != 0 {
        v.push("ALT");
    }
    if mods & input::PROMPT_SHIFT != 0 {
        v.push("SHIFT");
    }
    v.join("+")
}

fn binding_art(i: PromptInput, mods: u8) -> Art {
    if mods == 0 { input_art(i) } else { Art::Combo(mod_label(mods), Box::new(input_art(i))) }
}

/// The glyphs the port adds (file id -> art) and, per layout target, the keyboard glyph that
/// stands for it.
struct Set {
    arts: BTreeMap<String, Art>,
    keyboard: HashMap<&'static str, String>,
}

static SET: OnceLock<Set> = OnceLock::new();

fn set() -> &'static Set {
    SET.get_or_init(|| {
        let mut arts = BTreeMap::new();
        // Every key, mouse button and wheel direction a binding may use (a rebound action shows
        // its new input), and the layout's bindings with modifiers.
        for k in input::bindable_keys() {
            arts.insert(format!("pc_{}", input_id(PromptInput::Key(k), 0)), input_art(PromptInput::Key(k)));
        }
        for i in (0..5).map(PromptInput::Mouse).chain([PromptInput::WheelUp, PromptInput::WheelDown]) {
            arts.insert(format!("pc_{}", input_id(i, 0)), input_art(i));
        }
        for (i, mods) in input::bound_inputs() {
            arts.insert(format!("pc_{}", input_id(i, mods)), binding_art(i, mods));
        }
        let keyboard = keyboard_map(Some(&mut arts));
        for (id, art) in [
            ("xb_a", Art::XFace('A')),
            ("xb_b", Art::XFace('B')),
            ("xb_x", Art::XFace('X')),
            ("xb_y", Art::XFace('Y')),
            ("xb_lb", Art::XBumper("LB")),
            ("xb_rb", Art::XBumper("RB")),
            ("xb_lt", Art::XTrigger("LT")),
            ("xb_rt", Art::XTrigger("RT")),
            ("xb_ls", Art::XStick("LS")),
            ("xb_rs", Art::XStick("RS")),
            ("xb_menu", Art::XMenu),
            ("xb_view", Art::XView),
        ] {
            arts.insert(id.into(), art);
        }
        crate::info!("prompts: {} glyphs of the port's ({} targets on the keyboard and mouse)", arts.len(), keyboard.len());
        Set { arts, keyboard }
    })
}

/// Per layout target, the keyboard glyph standing for it (its first binding's; WASD-like
/// clusters and arrow pairs as one glyph). With `arts` (at start) glyphs the layout needs are
/// added; without (after a rebinding) only glyphs that exist are named: a cluster or pair the
/// layout did not have at start falls back to its first key's glyph.
fn keyboard_map(mut arts: Option<&mut BTreeMap<String, Art>>) -> HashMap<&'static str, String> {
    let mut keyboard = HashMap::new();
    let first = |t: &str| input::binding_of(t);
    let exists = |arts: &Option<&mut BTreeMap<String, Art>>, id: &str| arts.as_ref().is_some_and(|a| a.contains_key(id)) || set_arts_has(id);
    for &(_, target) in PS {
        let id = match target {
            "ls" | "rs" => {
                let side = &target[..1];
                let dirs = ["up", "left", "down", "right"].map(|d| first(&format!("{side}s_{d}")));
                if target == "rs" && input::raw_mouse_look() {
                    if let Some(a) = arts.as_deref_mut() {
                        a.insert("pc_look".into(), Art::MouseMove);
                    }
                    "pc_look".to_string()
                } else if dirs.iter().all(|d| matches!(d, Some((PromptInput::Key(_), 0)))) {
                    let labels = dirs.clone().map(|d| match d {
                        Some((PromptInput::Key(k), _)) => key_label(k),
                        _ => String::new(),
                    });
                    let id = format!("pc_mv_{}", dirs.iter().map(|d| input_id(d.unwrap().0, 0)).collect::<Vec<_>>().join("_"));
                    if let Some(a) = arts.as_deref_mut() {
                        a.insert(id.clone(), Art::Cluster(labels));
                        id
                    } else if exists(&arts, &id) {
                        id
                    } else {
                        format!("pc_{}", input_id(dirs[0].unwrap().0, 0))
                    }
                } else {
                    continue;
                }
            }
            "updown" | "leftright" => {
                let (a, b) = if target == "updown" { ("up", "down") } else { ("left", "right") };
                match (first(a), first(b)) {
                    (Some((PromptInput::Key(x), 0)), Some((PromptInput::Key(y), 0))) => {
                        let id = format!("pc_p_{}_{}", input_id(PromptInput::Key(x), 0), input_id(PromptInput::Key(y), 0));
                        if let Some(arts) = arts.as_deref_mut() {
                            let art = match (arrow_of(x), arrow_of(y)) {
                                (Some(p), Some(q)) => Art::ArrowPair(p, q),
                                _ => Art::KeyPair(key_label(x), key_label(y)),
                            };
                            arts.insert(id.clone(), art);
                            id
                        } else if exists(&arts, &id) {
                            id
                        } else {
                            format!("pc_{}", input_id(PromptInput::Key(x), 0))
                        }
                    }
                    (Some((i, m)), _) => format!("pc_{}", input_id(i, m)),
                    _ => continue,
                }
            }
            t => match first(t) {
                Some((i, m)) => format!("pc_{}", input_id(i, m)),
                None => continue,
            },
        };
        if arts.is_some() || exists(&arts, &id) {
            keyboard.insert(target, id);
        }
    }
    keyboard
}

/// Whether the port's glyph set (once made) has `id`.
fn set_arts_has(id: &str) -> bool {
    SET.get().is_some_and(|s| s.arts.contains_key(id))
}

/// The entity name of a glyph of the port's.
fn entity_name(id: &str) -> String {
    format!("ButtonIcon_PC_{id}")
}

fn name_hash(name: &str) -> u32 {
    cgpr::fnv1a(name.to_ascii_uppercase().as_bytes())
}

// ---------------------------------------------------------------------------
// The device and the remap

/// The PC tab's Button Prompts: 0 automatic, 1 PlayStation, 2 Xbox, 3 keyboard and mouse.
pub static MODE: AtomicU8 = AtomicU8::new(0);

/// The glyph set shown now: 1 PlayStation (the title's), 2 Xbox, 3 keyboard and mouse.
fn shown() -> u8 {
    match MODE.load(Relaxed) {
        m @ 1..=3 => m,
        // (A PlayStation controller in use shows the title's own glyphs, since 2026-11-23.)
        _ => match input::last_device() {
            input::DEVICE_KBM => 3,
            input::DEVICE_PAD => {
                if input::pad_is_playstation() {
                    1
                } else {
                    2
                }
            }
            _ => {
                if !input::pad_connected() {
                    3
                } else if input::pad_is_playstation() {
                    1
                } else {
                    2
                }
            }
        },
    }
}

/// The glyph set shown now (1 PlayStation, 2 Xbox, 3 keyboard and mouse): the name entry's
/// controller keyboard names its buttons by it.
pub fn shown_set() -> u8 {
    shown()
}

/// Per title glyph hash: the hashes of its keyboard, Xbox and PlayStation stand-ins (0: keep;
/// the PlayStation one where the controller's buttons are remapped: the PS5 pad's button in the
/// place of the one on the target); made again when the layout changes (a rebinding in the menu).
fn remaps() -> std::sync::Arc<Vec<(u32, u32, u32, u32)>> {
    static R: parking_lot::Mutex<Option<(u32, std::sync::Arc<Vec<(u32, u32, u32, u32)>>)>> = parking_lot::Mutex::new(None);
    let generation = input::layout_gen();
    let mut g = R.lock();
    if let Some((at, v)) = g.as_ref() {
        if *at == generation {
            return v.clone();
        }
    }
    let s = set();
    let keyboard = if generation == 0 { s.keyboard.clone() } else { keyboard_map(None) };
    let v: Vec<(u32, u32, u32, u32)> = PS
        .iter()
        .map(|&(name, target)| {
            let kb = keyboard.get(target).map(|id| name_hash(&entity_name(id))).unwrap_or(0);
            let (xb, ps) = match pad_button(target) {
                Some(p) => {
                    let (xb_id, ps_name) = PAD_GLYPHS[p];
                    // The title's own glyph where it already is the one (its d-pad, its button).
                    let own = |n: &str| if n == name || (name.starts_with("ButtonIcon_TouchPad") && n == "ButtonIcon_TouchPad") { 0 } else { name_hash(n) };
                    (xb_id.map(|id| name_hash(&entity_name(id))).unwrap_or_else(|| own(ps_name)), own(ps_name))
                }
                None => (0, 0),
            };
            (name_hash(name), kb, xb, ps)
        })
        .collect();
    let v = std::sync::Arc::new(v);
    *g = Some((generation, v.clone()));
    v
}

/// The title's inline-image draw after its map from the text's hash to a glyph entity's
/// (eboot+0x150e645, `lea rcx,[rbp-90h]` before the entity lookup; r15d the entity's name
/// hash): the device's glyph instead.
pub extern "sysv64" fn glyph_hook(regs: *const u64, _arg: u32) {
    let regs = regs as *mut u64;
    let shown = shown();
    let h = unsafe { *regs.add(15) } as u32;
    if let Some(&(_, kb, xb, ps)) = remaps().iter().find(|r| r.0 == h) {
        let to = match shown {
            1 => ps,
            3 => kb,
            _ => xb,
        };
        if to != 0 {
            unsafe { *regs.add(15) = (*regs.add(15) & !0xFFFF_FFFF) | to as u64 };
        }
    }
}

/// The settings the copies of directly drawn icons are shown by (`variants`): each true while
/// its condition on the shown set holds.
pub const SHOWN_PS: &str = "Setting_PC_PromptsPS";
pub const SHOWN_NOT_KB: &str = "Setting_PC_PromptsNotKB";
pub const SHOWN_NOT_XB: &str = "Setting_PC_PromptsNotXB";
pub const SHOWN_KB: &str = "Setting_PC_PromptsKB";
pub const SHOWN_XB: &str = "Setting_PC_PromptsXB";

/// The value of one of the SHOWN_* settings now.
pub fn shown_setting(which: u8) -> u32 {
    let s = shown();
    (match which {
        0 => s == 1,
        1 => s != 3,
        2 => s != 2,
        3 => s == 3,
        _ => s == 2,
    }) as u32
}

// ---------------------------------------------------------------------------
// The UI data and the textures

/// The global UI scene with an entity per glyph of the port's (copies of `ButtonIcon_X`). The
/// scene's `GlobalHandlerCP11` component names each glyph entity in a link (its event the
/// entity name's hash, its target the entity's component): the renderer finds glyphs there,
/// so each copy gets a copy of `ButtonIcon_X`'s link.
pub fn extend_global(orig: &[u8]) -> Result<Vec<u8>, String> {
    use super::value::{Payload, h};
    let mut f = cgpr::parse(orig).ok_or("unexpected layout")?;
    if cgpr::serialize(&f) != orig {
        return Err("does not serialize back to the same bytes".into());
    }
    let root = f.roots.iter().position(|r| r.name == b"GlobalHandler").ok_or("no GlobalHandler")?;
    let template = f.roots[root].kids.iter().find(|k| k.name == b"ButtonIcon_X").ok_or("no ButtonIcon_X")?.clone();
    let handler = f.roots[root]
        .comps
        .iter()
        .map(|(_, i)| *i as usize)
        .find(|&i| Payload::parse(&f.comps[i].payload).is_some_and(|p| p.value.hash == h("GlobalHandlerCP11")))
        .ok_or("no GlobalHandlerCP11")?;
    let mut links = Payload::parse(&f.comps[handler].payload).ok_or("GlobalHandlerCP11 unreadable")?;
    let x_hash = name_hash("ButtonIcon_X");
    let (list, x_link) = links
        .links
        .iter()
        .enumerate()
        .find_map(|(n, l)| l.iter().find(|k| k.event == x_hash).map(|k| (n, k.clone())))
        .ok_or("no link naming ButtonIcon_X")?;
    let s = set();
    let mut made = Vec::new();
    for id in s.arts.keys() {
        let repl = vec![(TEMPLATE_PATH.as_bytes().to_vec(), format!("$/ui/textures/buttonicons/****/{id}.ctxr").into_bytes())];
        let salt = cgpr::salt(id);
        let mut e = cgpr::clone_entity(&mut f.comps, &template, &repl, &salt);
        e.name = entity_name(id).into_bytes();
        let mut link = x_link.clone();
        link.event = name_hash(&entity_name(id));
        link.target = cgpr::salted(&x_link.target, &salt);
        links.links[list].push(link);
        made.push(e);
    }
    f.comps[handler].payload = links.ser();
    let n = made.len();
    f.roots[root].kids.extend(made);
    let out = cgpr::serialize(&f);
    cgpr::parse(&out).ok_or("the extended file does not parse")?;
    crate::info!("prompts: {n} glyph entities added to the global UI scene");
    Ok(out)
}

/// The target a directly drawn icon stands for: the title's glyph textures, and the docks whose
/// frame carries a button (the inventory's Square and Triangle docks).
fn direct_target(s: &str) -> Option<&'static str> {
    if let Some(name) = s.strip_prefix("$/ui/textures/buttonicons/****/") {
        return PS.iter().find(|(e, _)| name == texture_of(e)).map(|(_, t)| *t);
    }
    match s {
        "Tex_Frame_Clean_NavButton" => Some("triangle"),
        "Tex_InvFrame_ButtonDock" => Some("square"),
        _ => None,
    }
}

/// The title's texture file of a glyph entity (uiglobal.cgpr's names).
fn texture_of(entity: &str) -> &'static str {
    match entity {
        "ButtonIcon_X" => "ps4_button_cross.ctxr",
        "ButtonIcon_Circle" => "ps4_button_circle.ctxr",
        "ButtonIcon_Square" => "ps4_button_square.ctxr",
        "ButtonIcon_Triangle" => "ps4_button_triangle.ctxr",
        _ => "",
    }
}

/// The subtitles' condition in the HUD data (a `UISetting_bool` on `Setting_ShowSubtitles`: shown
/// while the setting is on), the template of the variants' conditions: its category and payload.
fn hide_template() -> Result<(u32, super::value::Payload), String> {
    use super::value::{Payload, h};
    let d = crate::hle::kernel::fs::read_file("/app0/ui/scripts/_cmn/uihud.cgpr").map_err(|e| if e.kind() == std::io::ErrorKind::NotFound { "no HUD data" } else { "HUD data unreadable" })?;
    let f = cgpr::parse(&d).ok_or("HUD data of another layout")?;
    let sub = f.roots.iter().find(|r| r.name.starts_with(b"ConversationSubtitle")).ok_or("no subtitles")?;
    let idx = sub
        .comps
        .iter()
        .map(|(_, i)| *i as usize)
        .find(|&i| Payload::parse(&f.comps[i].payload).is_some_and(|p| p.value.hash == h("HideCondition")))
        .ok_or("no subtitle condition")?;
    let p = Payload::parse(&f.comps[idx].payload).ok_or("condition unreadable")?;
    if p.links.iter().any(|l| !l.is_empty()) {
        return Err("the subtitle condition has links".into());
    }
    Ok((f.comps[idx].cat, p))
}

/// The Quad field holding a quad's texture source: a file (`#0ec58c80 {Texture: <path>}`, the
/// glyphs') or a sprite of a texture page (`#502ba93a`, the docks').
const QUAD_SOURCE: u32 = 0xd307_6083;
const FILE_SOURCE: u32 = 0x0ec5_8c80;

/// Points an entity's Quad at a texture file (a new component: the original's other users keep
/// theirs).
fn set_quad_file(f: &mut cgpr::File, e: &mut cgpr::Entity, path: &str) -> Result<(), String> {
    use super::value::{Node, Payload, h};
    let slot = e
        .comps
        .iter()
        .position(|(_, i)| Payload::parse(&f.comps[*i as usize].payload).is_some_and(|p| p.value.hash == h("Quad")))
        .ok_or("no Quad")?;
    let idx = e.comps[slot].1 as usize;
    let mut p = Payload::parse(&f.comps[idx].payload).ok_or("Quad unreadable")?;
    let src = p.value.field_mut(QUAD_SOURCE).ok_or("Quad without its texture")?;
    *src = Node::Obj { hash: FILE_SOURCE, fields: vec![(h("Texture"), Node::Str(path.as_bytes().to_vec()))] };
    let cat = f.comps[idx].cat;
    f.comps.push(cgpr::Component { cat, payload: p.ser() });
    e.comps[slot].1 = (f.comps.len() - 1) as u32;
    Ok(())
}

/// Adds a condition on `setting` to an entity (a new component).
fn add_condition(f: &mut cgpr::File, e: &mut cgpr::Entity, template: &(u32, super::value::Payload), setting: &str) -> Result<(), String> {
    let mut p = template.1.clone();
    if p.value.replace_str(b"Setting_ShowSubtitles", setting.as_bytes()) != 1 {
        return Err("condition template without its setting".into());
    }
    f.comps.push(cgpr::Component { cat: template.0, payload: p.ser() });
    let guid = cgpr::salted(&e.guid, &cgpr::salt(setting));
    e.comps.push((guid, (f.comps.len() - 1) as u32));
    Ok(())
}

/// Entities that draw a PlayStation button themselves (a Quad on a glyph texture or a dock
/// whose frame shows a button) get a copy per device after them, on that device's glyph and
/// shown while its set is shown; the original then shows only while no copy stands in. An
/// entity with a condition of its own is left as it is.
pub fn variants(f: &mut cgpr::File) -> Result<usize, String> {
    use super::value::{Node, Payload, h};
    let template = hide_template()?;
    let quad = h("Quad");
    let hide = h("HideCondition");
    // The entities to change, by path (child indices from a root).
    fn walk(f: &cgpr::File, e: &cgpr::Entity, path: &mut Vec<usize>, quad: u32, hide: u32, out: &mut Vec<(Vec<usize>, String, &'static str)>) {
        let mut has_hide = false;
        let mut found = None;
        for (_, i) in &e.comps {
            let Some(p) = Payload::parse(&f.comps[*i as usize].payload) else { continue };
            if p.value.hash == hide {
                has_hide = true;
            }
            if p.value.hash == quad {
                let mut strs = Vec::new();
                fn strings(n: &Node, out: &mut Vec<String>) {
                    match n {
                        Node::Str(b) => out.push(String::from_utf8_lossy(b).into_owned()),
                        Node::Obj { fields, .. } => fields.iter().for_each(|(_, v)| strings(v, out)),
                        _ => {}
                    }
                }
                if let super::value::Body::Fields(fields) = &p.value.body {
                    fields.iter().for_each(|(_, v)| strings(v, &mut strs));
                }
                found = strs.into_iter().find_map(|s| direct_target(&s).map(|t| (s, t)));
            }
        }
        if let (Some((s, t)), false) = (found, has_hide) {
            out.push((path.clone(), s, t));
        }
        for (k, kid) in e.kids.iter().enumerate() {
            path.push(k);
            walk(f, kid, path, quad, hide, out);
            path.pop();
        }
    }
    let mut todo = Vec::new();
    for r in 0..f.roots.len() {
        let root = f.roots[r].clone();
        walk(f, &root, &mut vec![r], quad, hide, &mut todo);
    }
    let set = set();
    let mut n = 0;
    // Deepest and last first: inserting siblings keeps the other paths valid.
    todo.sort_by(|a, b| b.0.cmp(&a.0));
    for (path, _src, target) in todo {
        let kb = set.keyboard.get(target).cloned();
        let xb = xbox_id(target);
        if kb.is_none() && xb.is_none() {
            continue;
        }
        let (r, rest) = path.split_first().unwrap();
        let orig = if rest.is_empty() { f.roots[*r].clone() } else { f.roots[*r].at(rest).clone() };
        let mut copies = Vec::new();
        for (id, setting, tag) in [(kb.clone(), SHOWN_KB, "PC"), (xb.map(String::from), SHOWN_XB, "XB")] {
            let Some(id) = id else { continue };
            let mut salt_src = format!("PCPrompt{tag}");
            for p in &path {
                salt_src.push_str(&format!("/{p}"));
            }
            let mut c = cgpr::clone_entity(&mut f.comps, &orig, &[], &cgpr::salt(&salt_src));
            c.name = [orig.name.as_slice(), b"_", tag.as_bytes()].concat();
            // The copy's quad on the device's glyph file (a dock's frame sprite as well).
            set_quad_file(f, &mut c, &format!("$/ui/textures/buttonicons/****/{id}.ctxr"))?;
            add_condition(f, &mut c, &template, setting)?;
            copies.push(c);
        }
        let shown_by = match (kb.is_some(), xb.is_some()) {
            (true, true) => SHOWN_PS,
            (true, false) => SHOWN_NOT_KB,
            _ => SHOWN_NOT_XB,
        };
        if rest.is_empty() {
            let mut e = f.roots[*r].clone();
            add_condition(f, &mut e, &template, shown_by)?;
            f.roots[*r] = e;
            for (k, c) in copies.into_iter().enumerate() {
                f.roots.insert(*r + 1 + k, c);
            }
        } else {
            let (last, parent_path) = rest.split_last().unwrap();
            let mut e = f.roots[*r].at(rest).clone();
            add_condition(f, &mut e, &template, shown_by)?;
            let parent = if parent_path.is_empty() { &mut f.roots[*r] } else { f.roots[*r].at_mut(parent_path) };
            parent.kids[*last] = e;
            for (k, c) in copies.into_iter().enumerate() {
                parent.kids.insert(*last + 1 + k, c);
            }
        }
        n += 1;
    }
    Ok(n)
}

/// A UI file with the variants of its directly drawn icons (the movies', the character
/// creator's, the shops' and storage's).
pub fn extend_icons(orig: &[u8]) -> Result<Vec<u8>, String> {
    let mut f = cgpr::parse(orig).ok_or("unexpected layout")?;
    if cgpr::serialize(&f) != orig {
        return Err("does not serialize back to the same bytes".into());
    }
    let n = variants(&mut f)?;
    let out = cgpr::serialize(&f);
    cgpr::parse(&out).ok_or("the extended file does not parse")?;
    crate::info!("prompts: {n} directly drawn icons with keyboard and Xbox copies");
    Ok(out)
}

/// Whether `guest` is a glyph texture of the port's.
pub fn serves(guest: &str) -> bool {
    guest.strip_prefix(DIR).is_some_and(|n| (n.starts_with("pc_") || n.starts_with("xb_")) && n.ends_with(".ctxr"))
}

/// A glyph texture of the port's: the title's glyph file layout around the art's pixels.
pub fn texture(guest: &str) -> Result<Vec<u8>, String> {
    let id = guest.strip_prefix(DIR).and_then(|n| n.strip_suffix(".ctxr")).ok_or("not a glyph")?;
    let art = set().arts.get(id).ok_or("no such glyph")?.clone();
    let template = {
        crate::hle::kernel::fs::read_file(&format!("{DIR}{TEMPLATE}")).map_err(|e| if e.kind() == std::io::ErrorKind::NotFound { "no template glyph" } else { "template unreadable" })?
    };
    if glyphs::decode(&template).is_none() {
        return Err("the template glyph has another layout".into());
    }
    let px = draw(&art);
    Ok(glyphs::file(&template, &glyphs::mips(&px)))
}

// ---------------------------------------------------------------------------
// The art: drawn at 4x (512x512) with exact coverage, then averaged to 128x128.

const S: usize = 512;
const U: f32 = 4.0; // canvas pixels per glyph texel

const WHITE: [f32; 3] = [0.94, 0.94, 0.94];
const KEY_SIDE: [f32; 3] = [0.62, 0.62, 0.64];
const DARK: [f32; 3] = [0.05, 0.05, 0.05];
const BODY: [f32; 3] = [0.07, 0.08, 0.08];
const GAP: [f32; 3] = [0.30, 0.30, 0.30];
const TEAL: [f32; 3] = [0.13, 0.82, 0.63];
const GREY: [f32; 3] = [0.55, 0.55, 0.57];

struct Canvas {
    /// Premultiplied RGBA.
    px: Vec<[f32; 4]>,
}

impl Canvas {
    fn new() -> Self {
        Canvas { px: vec![[0.0; 4]; S * S] }
    }

    /// Paints `c` (opaque) where `inside(x, y)` holds; coordinates in glyph texels (0..128).
    fn fill(&mut self, c: [f32; 3], inside: impl Fn(f32, f32) -> bool) {
        for y in 0..S {
            let fy = (y as f32 + 0.5) / U;
            for x in 0..S {
                let fx = (x as f32 + 0.5) / U;
                if inside(fx, fy) {
                    self.px[y * S + x] = [c[0], c[1], c[2], 1.0];
                }
            }
        }
    }

    /// Paints `c` with a coverage mask (0..255 per canvas pixel).
    fn fill_mask(&mut self, c: [f32; 3], mask: &[u8]) {
        for (p, &m) in self.px.iter_mut().zip(mask) {
            if m != 0 {
                let a = m as f32 / 255.0;
                for k in 0..3 {
                    p[k] = c[k] * a + p[k] * (1.0 - a);
                }
                p[3] = a + p[3] * (1.0 - a);
            }
        }
    }

    /// The glyph texture's pixels (128x128 premultiplied RGBA8, sRGB-encoded colours).
    fn finish(&self) -> Vec<u8> {
        let n = glyphs::SIZE as usize;
        let k = S / n;
        let mut out = vec![0u8; n * n * 4];
        for y in 0..n {
            for x in 0..n {
                let mut acc = [0f32; 4];
                for dy in 0..k {
                    for dx in 0..k {
                        let p = self.px[(y * k + dy) * S + x * k + dx];
                        for c in 0..4 {
                            acc[c] += p[c];
                        }
                    }
                }
                let o = (y * n + x) * 4;
                let cnt = (k * k) as f32;
                for c in 0..4 {
                    out[o + c] = ((acc[c] / cnt).clamp(0.0, 1.0) * 255.0 + 0.5) as u8;
                }
            }
        }
        out
    }
}

/// Inside a rounded rectangle (x0, y0)-(x1, y1) with corner radius r.
fn rrect(x0: f32, y0: f32, x1: f32, y1: f32, r: f32) -> impl Fn(f32, f32) -> bool {
    rrect2(x0, y0, x1, y1, r, r)
}

/// v kept within lo..hi (their middle when the range is empty).
fn within(v: f32, lo: f32, hi: f32) -> f32 {
    if lo > hi { (lo + hi) / 2.0 } else { v.clamp(lo, hi) }
}

/// Inside a rounded rectangle with its own top and bottom corner radii.
fn rrect2(x0: f32, y0: f32, x1: f32, y1: f32, rt: f32, rb: f32) -> impl Fn(f32, f32) -> bool {
    move |x, y| {
        if x < x0 || x > x1 || y < y0 || y > y1 {
            return false;
        }
        let r = if y < (y0 + y1) / 2.0 { rt } else { rb };
        let cx = within(x, x0 + r, x1 - r);
        let cy = within(y, y0 + r, y1 - r);
        (x - cx).powi(2) + (y - cy).powi(2) <= r * r
    }
}

fn disc(cx: f32, cy: f32, r: f32) -> impl Fn(f32, f32) -> bool {
    move |x, y| (x - cx).powi(2) + (y - cy).powi(2) <= r * r
}

/// Inside a convex polygon (points in order).
fn poly(pts: Vec<(f32, f32)>) -> impl Fn(f32, f32) -> bool {
    move |x, y| {
        let n = pts.len();
        let mut sign = 0f32;
        for i in 0..n {
            let (ax, ay) = pts[i];
            let (bx, by) = pts[(i + 1) % n];
            let c = (bx - ax) * (y - ay) - (by - ay) * (x - ax);
            if c != 0.0 {
                if sign == 0.0 {
                    sign = c.signum();
                } else if c.signum() != sign {
                    return false;
                }
            }
        }
        true
    }
}

/// An arrow pointing `dir` (0 up, 1 down, 2 left, 3 right) in the box centred at (cx, cy) of
/// half-size h: a head and a stem.
fn arrow(c: &mut Canvas, col: [f32; 3], dir: u8, cx: f32, cy: f32, h: f32) {
    // Drawn pointing up, then turned.
    let turn = move |x: f32, y: f32| -> (f32, f32) {
        let (dx, dy) = (x - cx, y - cy);
        match dir {
            0 => (dx, dy),
            1 => (dx, -dy),
            2 => (dy, dx),
            _ => (dy, -dx),
        }
    };
    let head = poly(vec![(0.0, -h), (h * 0.85, h * 0.05), (-h * 0.85, h * 0.05)]);
    c.fill(col, move |x, y| {
        let (u, v) = turn(x, y);
        head(u, v) || (u.abs() <= h * 0.28 && v >= -0.1 && v <= h)
    });
}

/// Inside a stroke from (ax, ay) to (bx, by) of width w (round ends).
fn seg(ax: f32, ay: f32, bx: f32, by: f32, w: f32) -> impl Fn(f32, f32) -> bool {
    move |x, y| {
        let (dx, dy) = (bx - ax, by - ay);
        let t = (((x - ax) * dx + (y - ay) * dy) / (dx * dx + dy * dy).max(1e-6)).clamp(0.0, 1.0);
        let (px, py) = (ax + t * dx - x, ay + t * dy - y);
        px * px + py * py <= w * w / 4.0
    }
}

/// The symbol a wide key's name gets on a key cap (its name would be too small to read in a
/// prompt): space, enter, shift, backspace, tab; drawn in the face (fx0, fy0)-(fx1, fy1).
fn key_symbol(c: &mut Canvas, label: &str, fx0: f32, fy0: f32, fx1: f32, fy1: f32) -> bool {
    let (w, h) = (fx1 - fx0, fy1 - fy0);
    let (cx, cy) = ((fx0 + fx1) / 2.0, (fy0 + fy1) / 2.0);
    let lw = w * 0.085;
    match label {
        "SPACE" => {
            // A bracket open at the top.
            let (x0, x1, y) = (cx - w * 0.30, cx + w * 0.30, cy + h * 0.12);
            c.fill(DARK, seg(x0, y, x1, y, lw));
            c.fill(DARK, seg(x0, y, x0, y - h * 0.22, lw));
            c.fill(DARK, seg(x1, y, x1, y - h * 0.22, lw));
        }
        "ENTER" => {
            // Down from the top right, then left to an arrow head.
            let (xr, yt, yb, xl) = (cx + w * 0.26, cy - h * 0.26, cy + h * 0.10, cx - w * 0.14);
            c.fill(DARK, seg(xr, yt, xr, yb, lw));
            c.fill(DARK, seg(xr, yb, xl, yb, lw));
            arrow(c, DARK, 2, cx - w * 0.20, yb, h * 0.17);
        }
        "SHIFT" => {
            // A hollow arrow pointing up.
            let (top, mid, bot) = (cy - h * 0.32, cy - h * 0.02, cy + h * 0.30);
            let (hw, sw) = (w * 0.30, w * 0.13);
            for (a, b) in [((cx, top), (cx + hw, mid)), ((cx + hw, mid), (cx + sw, mid)), ((cx + sw, mid), (cx + sw, bot)), ((cx + sw, bot), (cx - sw, bot)), ((cx - sw, bot), (cx - sw, mid)), ((cx - sw, mid), (cx - hw, mid)), ((cx - hw, mid), (cx, top))] {
                c.fill(DARK, seg(a.0, a.1, b.0, b.1, lw));
            }
        }
        "BKSP" => {
            // A long arrow pointing left.
            c.fill(DARK, seg(cx - w * 0.12, cy, cx + w * 0.30, cy, lw * 1.2));
            arrow(c, DARK, 2, cx - w * 0.16, cy, h * 0.18);
        }
        "TAB" => {
            // An arrow pointing right to a bar.
            let xb = cx + w * 0.28;
            c.fill(DARK, seg(cx - w * 0.30, cy, xb - w * 0.12, cy, lw * 1.2));
            arrow(c, DARK, 3, xb - w * 0.08, cy, h * 0.18);
            c.fill(DARK, seg(xb, cy - h * 0.20, xb, cy + h * 0.20, lw));
        }
        _ => return false,
    }
    true
}

/// A key cap in the box, its label (or arrow) dark on its face.
fn key_cap(c: &mut Canvas, x0: f32, y0: f32, x1: f32, y1: f32, label: Option<&str>, arrow_dir: Option<u8>) {
    let w = x1 - x0;
    let r = w * 0.17;
    c.fill(KEY_SIDE, rrect(x0, y0, x1, y1, r));
    let inset = w * 0.06;
    let (fx0, fy0, fx1, fy1) = (x0 + inset, y0 + inset, x1 - inset, y1 - inset - w * 0.08);
    c.fill(WHITE, rrect(fx0, fy0, fx1, fy1, r * 0.75));
    let (cx, cy) = ((fx0 + fx1) / 2.0, (fy0 + fy1) / 2.0);
    if let Some(d) = arrow_dir {
        arrow(c, DARK, d, cx, cy, (fy1 - fy0) * 0.32);
    }
    if let Some(t) = label {
        if key_symbol(c, t, fx0, fy0, fx1, fy1) {
            return;
        }
        let n = t.chars().count();
        let max_w = (fx1 - fx0) * if n == 1 { 0.82 } else { 0.92 };
        let max_h = (fy1 - fy0) * match n {
            1 => 0.66,
            2 => 0.52,
            _ => 0.46,
        };
        if let Some(m) = text_mask(t, cx, cy, max_w, max_h) {
            c.fill_mask(DARK, &m);
        }
    }
}

/// The mouse body in the box with button `lit` (or the wheel: 2) in teal.
fn mouse(c: &mut Canvas, x0: f32, y0: f32, x1: f32, y1: f32, lit: Option<u8>) {
    let w = x1 - x0;
    let h = y1 - y0;
    let r = w * 0.48;
    let body = rrect2(x0, y0, x1, y1, r, w * 0.42);
    c.fill(WHITE, &body);
    let split = y0 + h * 0.40;
    let mid = (x0 + x1) / 2.0;
    let line = w * 0.045;
    if let Some(b) = lit.filter(|b| *b < 2) {
        let left = b == 0;
        c.fill(TEAL, |x, y| body(x, y) && y < split && if left { x < mid } else { x > mid });
    }
    // The buttons' seams.
    c.fill(DARK, |x, y| body(x, y) && ((y < split && (x - mid).abs() < line) || (y - split).abs() < line));
    // The wheel.
    let (wy0, wy1) = (y0 + h * 0.10, y0 + h * 0.30);
    c.fill(if lit == Some(2) { TEAL } else { DARK }, rrect(mid - w * 0.09, wy0, mid + w * 0.09, wy1, w * 0.09));
    if lit == Some(2) {
        c.fill(DARK, |x, y| (y - (wy0 + wy1) / 2.0).abs() < line * 0.6 && (x - mid).abs() < w * 0.09);
    }
    // Side buttons (the fourth and fifth) on the left flank.
    if let Some(b @ 3..=4) = lit {
        let (sy0, sy1) = if b == 3 { (y0 + h * 0.48, y0 + h * 0.60) } else { (y0 + h * 0.64, y0 + h * 0.76) };
        c.fill(TEAL, rrect(x0 - w * 0.10, sy0, x0 + w * 0.16, sy1, w * 0.06));
        c.fill(DARK, |x, y| body(x, y) && (x - (x0 + w * 0.16)).abs() < line * 0.8 && y > sy0 && y < sy1);
    }
}

fn face_disc(c: &mut Canvas) {
    c.fill(WHITE, disc(64.0, 64.0, 56.0));
    c.fill(GAP, disc(64.0, 64.0, 50.5));
    c.fill(WHITE, disc(64.0, 64.0, 47.5));
}

fn draw(art: &Art) -> Vec<u8> {
    let mut c = Canvas::new();
    draw_into(&mut c, art, (0.0, 0.0, 128.0));
    c.finish()
}

/// Draws `art` into the square at (ox, oy) of side `size` (glyph texels): the glyph's 128
/// units scaled into it.
fn draw_into(c: &mut Canvas, art: &Art, (ox, oy, size): (f32, f32, f32)) {
    let k = size / 128.0;
    let p = |v: f32| v * k;
    let (x, y) = (|v: f32| ox + v * k, |v: f32| oy + v * k);
    match art {
        Art::Key(label) => key_cap(c, x(14.0), y(16.0), x(114.0), y(114.0), Some(label), None),
        Art::Arrow(d) => key_cap(c, x(14.0), y(16.0), x(114.0), y(114.0), None, Some(*d)),
        Art::ArrowPair(a, b) => {
            key_cap(c, x(4.0), y(30.0), x(62.0), y(98.0), None, Some(*a));
            key_cap(c, x(66.0), y(30.0), x(124.0), y(98.0), None, Some(*b));
        }
        Art::KeyPair(a, b) => {
            key_cap(c, x(4.0), y(30.0), x(62.0), y(98.0), Some(a), None);
            key_cap(c, x(66.0), y(30.0), x(124.0), y(98.0), Some(b), None);
        }
        Art::Mouse(b) => mouse(c, x(36.0), y(10.0), x(92.0), y(118.0), Some(*b)),
        Art::Wheel(up) => {
            mouse(c, x(22.0), y(12.0), x(78.0), y(118.0), Some(2));
            arrow(c, WHITE, if *up { 0 } else { 1 }, x(104.0), y(if *up { 34.0 } else { 52.0 }), p(18.0));
        }
        Art::MouseMove => {
            mouse(c, x(46.0), y(30.0), x(82.0), y(98.0), None);
            arrow(c, WHITE, 0, x(64.0), y(14.0), p(12.0));
            arrow(c, WHITE, 1, x(64.0), y(114.0), p(12.0));
            arrow(c, WHITE, 2, x(24.0), y(64.0), p(12.0));
            arrow(c, WHITE, 3, x(104.0), y(64.0), p(12.0));
        }
        Art::Cluster(k4) => {
            let s = 40.0;
            key_cap(c, x(44.0), y(16.0), x(44.0 + s), y(16.0 + s), Some(&k4[0]), None);
            key_cap(c, x(2.0), y(62.0), x(2.0 + s), y(62.0 + s), Some(&k4[1]), None);
            key_cap(c, x(44.0), y(62.0), x(44.0 + s), y(62.0 + s), Some(&k4[2]), None);
            key_cap(c, x(86.0), y(62.0), x(86.0 + s), y(62.0 + s), Some(&k4[3]), None);
        }
        Art::Combo(m, inner) => {
            draw_into(c, inner, (x(36.0), y(0.0), p(92.0)));
            key_cap(c, x(0.0), y(72.0), x(54.0), y(124.0), Some(m), None);
        }
        Art::XFace(ch) => {
            face_disc(c);
            if let Some(m) = text_mask(&ch.to_string(), x(64.0), y(64.0), p(60.0), p(52.0)) {
                c.fill_mask(DARK, &m);
            }
        }
        Art::XBumper(label) => {
            c.fill(WHITE, rrect2(x(14.0), y(36.0), x(114.0), y(92.0), p(22.0), p(12.0)));
            c.fill(BODY, rrect2(x(19.0), y(41.0), x(109.0), y(87.0), p(18.0), p(8.0)));
            if let Some(m) = text_mask(label, x(64.0), y(64.0), p(70.0), p(34.0)) {
                c.fill_mask(WHITE, &m);
            }
        }
        Art::XTrigger(label) => {
            c.fill(WHITE, rrect2(x(18.0), y(16.0), x(110.0), y(106.0), p(36.0), p(12.0)));
            c.fill(BODY, rrect2(x(23.0), y(21.0), x(105.0), y(101.0), p(32.0), p(8.0)));
            if let Some(m) = text_mask(label, x(64.0), y(68.0), p(66.0), p(36.0)) {
                c.fill_mask(WHITE, &m);
            }
        }
        Art::XStick(label) => {
            c.fill(WHITE, disc(x(64.0), y(64.0), p(50.0)));
            c.fill(BODY, disc(x(64.0), y(64.0), p(45.0)));
            c.fill(GREY, |a, b| {
                let d = ((a - x(64.0)).powi(2) + (b - y(64.0)).powi(2)).sqrt();
                (p(36.0)..=p(39.0)).contains(&d)
            });
            if let Some(m) = text_mask(label, x(64.0), y(64.0), p(54.0), p(32.0)) {
                c.fill_mask(WHITE, &m);
            }
        }
        Art::XMenu => {
            face_disc(c);
            for row in [-14.0f32, 0.0, 14.0] {
                c.fill(DARK, rrect(x(42.0), y(64.0 + row - 4.0), x(86.0), y(64.0 + row + 4.0), p(2.0)));
            }
        }
        Art::XView => {
            face_disc(c);
            let ring = |x0: f32, y0: f32, x1: f32, y1: f32| {
                let outer = rrect(x0, y0, x1, y1, 4.0);
                let inner = rrect(x0 + 5.0, y0 + 5.0, x1 - 5.0, y1 - 5.0, 2.0);
                move |a: f32, b: f32| outer(a, b) && !inner(a, b)
            };
            c.fill(DARK, ring(x(36.0), y(36.0), x(76.0), y(72.0)));
            c.fill(WHITE, rrect(x(52.0), y(56.0), x(92.0), y(92.0), p(4.0)));
            c.fill(DARK, ring(x(52.0), y(56.0), x(92.0), y(92.0)));
        }
    }
}

/// A label's coverage on the canvas (S x S, 0..255), centred at (cx, cy) in glyph texels and
/// fitted into max_w x max_h: Arial Bold through GDI, drawn without smoothing at the canvas's
/// 4x (the average to the glyph's size smooths it).
#[cfg(windows)]
fn text_mask(text: &str, cx: f32, cy: f32, max_w: f32, max_h: f32) -> Option<Vec<u8>> {
    use windows_sys::Win32::Foundation::SIZE;
    use windows_sys::Win32::Graphics::Gdi::*;
    let wide: Vec<u16> = text.encode_utf16().collect();
    let face: Vec<u16> = "Arial\0".encode_utf16().collect();
    unsafe {
        let dc = CreateCompatibleDC(std::ptr::null_mut());
        if dc.is_null() {
            return None;
        }
        let mut bmi: BITMAPINFO = std::mem::zeroed();
        bmi.bmiHeader.biSize = std::mem::size_of::<BITMAPINFOHEADER>() as u32;
        bmi.bmiHeader.biWidth = S as i32;
        bmi.bmiHeader.biHeight = -(S as i32);
        bmi.bmiHeader.biPlanes = 1;
        bmi.bmiHeader.biBitCount = 32;
        bmi.bmiHeader.biCompression = BI_RGB;
        let mut bits: *mut std::ffi::c_void = std::ptr::null_mut();
        let bmp = CreateDIBSection(dc, &bmi, DIB_RGB_COLORS, &mut bits, std::ptr::null_mut(), 0);
        if bmp.is_null() || bits.is_null() {
            DeleteDC(dc);
            return None;
        }
        let old_bmp = SelectObject(dc, bmp as HGDIOBJ);
        // Cell height from the box, shrunk until the label fits.
        let mut height = (max_h * U * 1.3) as i32;
        let mut result = None;
        for _ in 0..12 {
            let font = CreateFontW(-height, 0, 0, 0, 700, 0, 0, 0, DEFAULT_CHARSET as u32, OUT_TT_PRECIS as u32, CLIP_DEFAULT_PRECIS as u32, NONANTIALIASED_QUALITY as u32, 0, face.as_ptr());
            if font.is_null() {
                break;
            }
            let old_font = SelectObject(dc, font as HGDIOBJ);
            let mut sz: SIZE = std::mem::zeroed();
            GetTextExtentPoint32W(dc, wide.as_ptr(), wide.len() as i32, &mut sz);
            let mut tm: TEXTMETRICW = std::mem::zeroed();
            GetTextMetricsW(dc, &mut tm);
            // The label's visible height is about its cap height (ascent less internal leading).
            let cap = (tm.tmAscent - tm.tmInternalLeading) as f32;
            let fits = (sz.cx as f32) <= max_w * U && cap <= max_h * U;
            if fits || height < 8 {
                std::ptr::write_bytes(bits as *mut u8, 0, S * S * 4);
                SetBkMode(dc, TRANSPARENT as _);
                SetTextColor(dc, 0x00FF_FFFF);
                // Centre the cap height's box on (cx, cy).
                let x = (cx * U - sz.cx as f32 / 2.0).round() as i32;
                let top = (cy * U - cap / 2.0 - tm.tmInternalLeading as f32).round() as i32;
                TextOutW(dc, x, top, wide.as_ptr(), wide.len() as i32);
                GdiFlush();
                let px = std::slice::from_raw_parts(bits as *const u8, S * S * 4);
                result = Some(px.chunks_exact(4).map(|p| p[2].max(p[1]).max(p[0])).collect());
                SelectObject(dc, old_font);
                DeleteObject(font as HGDIOBJ);
                break;
            }
            SelectObject(dc, old_font);
            DeleteObject(font as HGDIOBJ);
            let scale = ((max_w * U) / sz.cx.max(1) as f32).min((max_h * U) / cap.max(1.0)).min(0.95);
            height = ((height as f32) * scale).floor() as i32;
        }
        SelectObject(dc, old_bmp);
        DeleteObject(bmp as HGDIOBJ);
        DeleteDC(dc);
        result
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Diagnosis: the entities `variants` gives copies in a UI file (`DS_TEST_CGPR=<file>`):
    /// their path, names from the root down, and the quad string that made them a target.
    #[test]
    #[ignore]
    fn variant_targets_dump() {
        use super::super::value::{Body, Node, Payload, h};
        let Ok(path) = std::env::var("DS_TEST_CGPR") else { return };
        let f = cgpr::parse(&std::fs::read(path).unwrap()).unwrap();
        fn walk(f: &cgpr::File, e: &cgpr::Entity, names: &mut Vec<String>, path: &mut Vec<usize>) {
            names.push(String::from_utf8_lossy(&e.name).into_owned());
            for (_, i) in &e.comps {
                let Some(p) = Payload::parse(&f.comps[*i as usize].payload) else { continue };
                if p.value.hash != h("Quad") {
                    continue;
                }
                let mut strs = Vec::new();
                fn strings(n: &Node, out: &mut Vec<String>) {
                    match n {
                        Node::Str(b) => out.push(String::from_utf8_lossy(b).into_owned()),
                        Node::Obj { fields, .. } => fields.iter().for_each(|(_, v)| strings(v, out)),
                        _ => {}
                    }
                }
                if let Body::Fields(fields) = &p.value.body {
                    fields.iter().for_each(|(_, v)| strings(v, &mut strs));
                }
                for s in &strs {
                    if let Some(t) = direct_target(s) {
                        println!("{path:?} {} -> {s} ({t})", names.join(" / "));
                    }
                }
            }
            for (k, kid) in e.kids.iter().enumerate() {
                path.push(k);
                walk(f, kid, names, path);
                path.pop();
            }
            names.pop();
        }
        for (r, root) in f.roots.iter().enumerate() {
            walk(&f, root, &mut Vec::new(), &mut vec![r]);
        }
    }

    #[test]
    fn glyphs_draw() {
        // Every kind draws something inside the glyph, nothing at its corners' edge pixels.
        for art in [
            Art::Key("E".into()),
            Art::Key("SPACE".into()),
            Art::Arrow(0),
            Art::Mouse(0),
            Art::Wheel(true),
            Art::MouseMove,
            Art::Cluster(["W".into(), "A".into(), "S".into(), "D".into()]),
            Art::Combo("SHIFT".into(), Box::new(Art::Mouse(0))),
            Art::XFace('A'),
            Art::XBumper("LB"),
            Art::XTrigger("RT"),
            Art::XStick("LS"),
            Art::XMenu,
            Art::XView,
        ] {
            let px = draw(&art);
            let covered = px.chunks_exact(4).filter(|p| p[3] > 128).count();
            assert!(covered > 1500, "{art:?}: {covered}");
        }
        if let Ok(dir) = std::env::var("DS_DUMP_DIR") {
            for (name, art) in [
                ("e", Art::Key("E".into())),
                ("space", Art::Key("SPACE".into())),
                ("enter", Art::Key("ENTER".into())),
                ("shift", Art::Key("SHIFT".into())),
                ("bksp", Art::Key("BKSP".into())),
                ("tab", Art::Key("TAB".into())),
                ("ctrl", Art::Key("CTRL".into())),
                ("esc", Art::Key("ESC".into())),
                ("f12", Art::Key("F12".into())),
                ("pgup", Art::Key("PG UP".into())),
                ("up", Art::Arrow(0)),
                ("pair", Art::ArrowPair(0, 1)),
                ("lmb", Art::Mouse(0)),
                ("rmb", Art::Mouse(1)),
                ("mmb", Art::Mouse(2)),
                ("m4", Art::Mouse(3)),
                ("wu", Art::Wheel(true)),
                ("look", Art::MouseMove),
                ("wasd", Art::Cluster(["W".into(), "A".into(), "S".into(), "D".into()])),
                ("shift_lmb", Art::Combo("SHIFT".into(), Box::new(Art::Mouse(0)))),
                ("xa", Art::XFace('A')),
                ("xlb", Art::XBumper("LB")),
                ("xrt", Art::XTrigger("RT")),
                ("xls", Art::XStick("LS")),
                ("xmenu", Art::XMenu),
                ("xview", Art::XView),
            ] {
                let px = draw(&art);
                std::fs::write(std::path::Path::new(&dir).join(format!("pc_{name}.png")), crate::gpu::png::encode(128, 128, &px)).unwrap();
            }
        }
    }
}

#[cfg(target_os = "linux")]
fn text_mask(text: &str, cx: f32, cy: f32, max_w: f32, max_h: f32) -> Option<Vec<u8>> { crate::platform::text::mask(text, S, S, (cx * U, cy * U, max_w * U, max_h * U)) }
