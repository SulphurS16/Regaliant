//! The title's UI manager, read from guest memory (PPSA01341 01.007.000; nothing is called or
//! written). From the disassembly of `UIManager::PrintSceneState` (eboot+0xde5040..0xde5990)
//! and the console commands that queue work on the manager (eboot+0xdeba60):
//! - the manager at [eboot+0x329B050];
//! - its active menu stack (top first when read from the end): a vector at +0x1260 / +0x1268
//!   of 0x18-byte entries, guarded by a spin lock at +0x1220 (a reader count at +0x1228);
//! - its active HUD scenes: a vector at +0x1288 / +0x1290;
//! - scenes in transition: menu stack +0x910 / +0x918, HUD +0x11F0 / +0x11F8.
//!
//! The menu stack tells the port whether a menu is open (the menu pointer, `pcsettings`).
//! The offsets are checked against the code bytes they were read from first; another
//! executable reports nothing.

const MANAGER: u64 = 0x329B050;
/// eboot+0xde55bd: `mov r14,[rbx+1260h]; mov r15,[rbx+1268h]` (the active menu stack).
const STACK_CODE_AT: u64 = 0xde55bd;
const STACK_CODE: [u8; 14] = [0x4c, 0x8b, 0xb3, 0x60, 0x12, 0x00, 0x00, 0x4c, 0x8b, 0xbb, 0x68, 0x12, 0x00, 0x00];
/// eboot+0xde5696: `mov rax,[rbx+1290h]; cmp rax,[rbx+1288h]` (the HUD scenes).
const HUD_CODE_AT: u64 = 0xde5696;
const HUD_CODE: [u8; 14] = [0x48, 0x8b, 0x83, 0x90, 0x12, 0x00, 0x00, 0x48, 0x3b, 0x83, 0x88, 0x12, 0x00, 0x00];
const ENTRY: u64 = 0x18;

fn base() -> Option<u64> {
    crate::loader::main_module().map(|m| m.base)
}

fn rd64(a: u64) -> Option<u64> {
    if a < 0x1000 || !crate::mem::readable(a, 8) {
        return None;
    }
    Some(unsafe { std::ptr::read_volatile(a as *const u64) })
}

/// The layout holds for this executable (its code bytes as read).
fn checked() -> Option<u64> {
    static OK: std::sync::OnceLock<Option<u64>> = std::sync::OnceLock::new();
    *OK.get_or_init(|| {
        let b = base()?;
        let same = |at: u64, code: &[u8]| crate::mem::readable(b + at, code.len() as u64) && unsafe { std::slice::from_raw_parts((b + at) as *const u8, code.len()) } == code;
        if same(STACK_CODE_AT, &STACK_CODE) && same(HUD_CODE_AT, &HUD_CODE) {
            Some(b)
        } else {
            crate::warn!("uistate: the UI manager's layout does not match this executable");
            None
        }
    })
}

/// The UI manager (None before the title made it).
fn manager() -> Option<u64> {
    let b = checked()?;
    rd64(b + MANAGER).filter(|&m| m >= 0x1000)
}

fn vec_len(m: u64, at: u64) -> Option<u64> {
    let (begin, end) = (rd64(m + at)?, rd64(m + at + 8)?);
    (end >= begin && (end - begin) % ENTRY == 0 && end - begin < 0x10000).then(|| (end - begin) / ENTRY)
}

/// Scenes on the active menu stack (None: no UI manager yet, or another executable).
pub fn menu_depth() -> Option<u64> {
    vec_len(manager()?, 0x1260)
}

/// Identity of the current menu, for retrying a relay's known offline dialog only.
#[cfg(target_os = "linux")]
pub fn top_menu() -> Option<u64> {
    let m = manager()?;
    let n = vec_len(m, 0x1260)?;
    if n == 0 { return None; }
    rd64(rd64(m + 0x1260)? + (n - 1) * ENTRY).filter(|&p| p >= 0x1000)
}

/// Active HUD scenes.
pub fn hud_count() -> Option<u64> {
    let m = manager()?;
    let (begin, end) = (rd64(m + 0x1288)?, rd64(m + 0x1290)?);
    (end >= begin && end - begin < 0x10000).then(|| end - begin)
}

/// Diagnosis (`uistack`): the manager, the stack's entries (raw words) and the HUD vector.
pub fn report() -> String {
    let Some(m) = manager() else { return "uistack: no UI manager (not made yet, or another executable)".into() };
    let mut s = format!("uistack: manager {m:#x}, menu stack {:?} scene(s), HUD bytes {:?}", menu_depth(), hud_count());
    if let (Some(begin), Some(n)) = (rd64(m + 0x1260), menu_depth()) {
        for i in 0..n.min(8) {
            let e = begin + i * ENTRY;
            let w: Vec<String> = (0..3).map(|k| rd64(e + k * 8).map(|v| format!("{v:#x}")).unwrap_or("?".into())).collect();
            s += &format!("; [{i}] {}", w.join(" "));
        }
    }
    s
}

/// The UIElement component's vtable (eboot+0x25d83a8; its loader, eboot+0xe09210, reads X +0xa4,
/// Y +0xa8, Width +0xb0, Height +0xb4).
const ELEMENT_VTABLE: u64 = 0x25d83a8;

/// Diagnosis (`uielems <tag>`): every UIElement component in guest memory (found by its vtable)
/// with its first 0x100 bytes, one line each, in `<run dir>/uielems-<tag>.txt`.
pub fn dump_elements(dir: &std::path::Path, tag: &str) -> Result<String, String> {
    dump_by_vtable(dir, &format!("uielems-{tag}"), ELEMENT_VTABLE, 0x40)
}

/// The runtime UI node's vtable (eboot+0x25d8430: its definition, a UIElement, at +0x110).
const NODE_VTABLE: u64 = 0x25d8430;

/// Diagnosis (`uinodes <tag>`): every runtime UI node with its first 0x300 bytes.
pub fn dump_nodes(dir: &std::path::Path, tag: &str) -> Result<String, String> {
    dump_by_vtable(dir, &format!("uinodes-{tag}"), NODE_VTABLE, 0xD0)
}

fn dump_by_vtable(dir: &std::path::Path, name: &str, vtable: u64, words_n: u64) -> Result<String, String> {
    let b = checked().ok_or("no UI manager layout for this executable")?;
    let vt = b + vtable;
    let t0 = std::time::Instant::now();
    let mut out = String::new();
    let mut n = 0;
    sweep(vt, words_n * 4, |a, _| {
        let words: Vec<u32> = (0..words_n).map(|i| unsafe { std::ptr::read_volatile((a + i * 4) as *const u32) }).collect();
        out += &format!("{a:#x}");
        for w in &words {
            out += &format!(" {w:08x}");
        }
        out += "\n";
        n += 1;
    });
    let path = dir.join(format!("{name}.txt"));
    std::fs::write(&path, out).map_err(|e| e.to_string())?;
    Ok(format!("{name}: {n} objects in {} ms -> {}", t0.elapsed().as_millis(), path.display()))
}

// ---------------------------------------------------------------------------
// The runtime UI node tree (the menu pointer, `pcsettings::pointer`). From the disassembly of
// the node's size and position update (eboot+0xe0b8f0) and dumps of every node in a boot
// (2026-11-06): a node's definition (a UIElement: X +0xa4, Y +0xa8, W +0xb0, H +0xb4) at +0x110,
// its parent at +0x2c8, flags at +0x2dc (0x41 set on the focused item), its position relative
// to the parent at +0x2fc/+0x300 and its size at +0x304/+0x308, in units of a 1920x1080 canvas
// (FLT_MAX: not placed; a scene's root is drawn at the origin then). Flag 0x1 marks the selected
// item of each list (68 at once in a SETTINGS page), 0x40 the input focus's path (the containers
// from the screen down to the list the player moves in: entering a page moves it from the tab
// list to the page's list, leaving moves it back). The node pools
// do not change while the game runs (36458 nodes in every state of a boot), so one scan finds
// them all.

use parking_lot::Mutex;
use std::collections::HashMap;
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};

pub struct Tree {
    pub nodes: Vec<u64>,
    pub children: HashMap<u64, Vec<u64>>,
    /// The guest ranges the nodes and their ancestors lie in (their objects merged, within the
    /// mapping each was found in; sorted): checked mapped before nodes are read (`guarded`).
    pub spans: Vec<(u64, u64)>,
}

impl Tree {
    /// Whether a node object at `a` lies in the checked ranges: only such nodes are read (a
    /// parent pointer read out of a node may lead anywhere, unmapped memory too).
    pub fn holds(&self, a: u64) -> bool {
        let i = self.spans.partition_point(|&(s, _)| s <= a);
        i > 0 && a.saturating_add(NODE_BYTES) <= self.spans[i - 1].1
    }
}

/// A node object's bytes the port reads (its fields up to the size at +0x308).
const NODE_BYTES: u64 = 0x310;

static TREE: Mutex<Option<Arc<Tree>>> = Mutex::new(None);
static SCANNING: AtomicBool = AtomicBool::new(false);
static LAST_SCAN: Mutex<Option<std::time::Instant>> = Mutex::new(None);

/// The node tree (None while the first scan runs: it starts on the first call, on a thread of
/// its own).
pub fn tree() -> Option<Arc<Tree>> {
    let t = TREE.lock().clone();
    match t {
        // Scanned before the title made its UI's pools: again (at most every 20 s).
        Some(t) if t.nodes.is_empty() => {
            rescan();
            None
        }
        Some(t) => Some(t),
        None => {
            start_scan();
            None
        }
    }
}

fn start_scan() -> bool {
    if checked().is_some() && manager().is_some() && !SCANNING.swap(true, Ordering::SeqCst) {
        *LAST_SCAN.lock() = Some(std::time::Instant::now());
        std::thread::Builder::new()
            .name("ui-tree".into())
            .spawn(|| {
                let t0 = std::time::Instant::now();
                let t = scan();
                crate::info!("uistate: {} UI nodes ({} with children) in {} ms", t.nodes.len(), t.children.len(), t0.elapsed().as_millis());
                *TREE.lock() = Some(Arc::new(t));
                SCANNING.store(false, Ordering::SeqCst);
            })
            .ok();
        return true;
    }
    false
}

/// A new scan (the old tree serves meanwhile), at most every 20 s: nodes the index misses.
pub fn rescan() {
    if LAST_SCAN.lock().is_some_and(|t| t.elapsed() < std::time::Duration::from_secs(20)) {
        return;
    }
    crate::info!("uistate: the UI nodes are scanned again (a focused item outside the index)");
    start_scan();
}

/// A screen's scan asked for and not started yet (`rescan_screen`).
static SCREEN_PENDING: AtomicBool = AtomicBool::new(false);

/// A new scan for a menu screen just opened (its rows may be new nodes: the lists a data
/// provider fills, as equipment and inventory): it starts once no scan runs and 3 s have passed
/// since the last one started (`screen_tick`), so a screen opened soon after another is not
/// missed.
pub fn rescan_screen() {
    SCREEN_PENDING.store(true, Ordering::SeqCst);
    screen_tick();
}

/// Each vblank while a menu is open: starts a screen's scan that waits for its turn.
pub fn screen_tick() {
    if !SCREEN_PENDING.load(Ordering::SeqCst) || SCANNING.load(Ordering::SeqCst) {
        return;
    }
    if LAST_SCAN.lock().is_some_and(|t| t.elapsed() < std::time::Duration::from_secs(3)) {
        return;
    }
    if start_scan() {
        SCREEN_PENDING.store(false, Ordering::SeqCst);
        crate::info!("uistate: the UI nodes are scanned again (another menu screen)");
    }
}

/// The top of the active menu stack (its entry's first word: the scene), None without a menu.
pub fn menu_top() -> Option<u64> {
    let m = manager()?;
    let (begin, end) = (rd64(m + 0x1260)?, rd64(m + 0x1268)?);
    if end <= begin || (end - begin) % ENTRY != 0 || end - begin >= 0x10000 {
        return None;
    }
    rd64(end - ENTRY)
}

/// Calls `f(a, mapping)` for each 8-byte aligned guest address `a` of the readable mappings whose
/// first word is `word` and whose `len` bytes lie in its mapping (`mapping`: that mapping's
/// [start, end) when the sweep began). Memory is read in pieces of 1 MiB, each checked in the
/// mapping table and read under the guest-reader lock with no mapping change in between: the
/// title unmaps memory while the sweep runs (two of the user's sessions ended on a read of a page
/// unmapped 0.2-0.6 s before, 2026-11-13). A piece no longer mapped is passed over. `f` runs
/// under the lock: it must not take the mapping table's (`mem::readable`, `mem::MEM`).
fn sweep(word: u64, len: u64, mut f: impl FnMut(u64, (u64, u64))) {
    use std::sync::atomic::Ordering::SeqCst;
    const PIECE: u64 = 1 << 20;
    let maps: Vec<(u64, u64)> = crate::mem::MEM.lock().vmas.values().filter(|m| m.kind != crate::mem::VmaKind::Reserved && m.prot & 1 != 0).map(|m| (m.start, m.end())).collect();
    for (lo, hi) in maps {
        let mut c = (lo + 7) & !7;
        let mut tries = 0;
        while c + len <= hi {
            let end = (c + PIECE).min(hi);
            // The piece and the bytes its last candidates reach.
            let reach = (end + len).min(hi);
            let g0 = crate::mem::MAP_GEN.load(SeqCst);
            if !crate::mem::host_readable(c, reach - c) {
                c = end;
                tries = 0;
                continue;
            }
            let readers = crate::mem::GUEST_READERS.read();
            if !crate::mem::table_view_held(g0) {
                drop(readers);
                tries += 1;
                if tries > 64 {
                    c = end;
                    tries = 0;
                }
                continue;
            }
            let mut a = c;
            while a < end && a + len <= hi {
                // (A page the GPU holds, no-access while its newer words are elsewhere: a read
                // would force its release, about 3,650 a sweep; UI nodes are never there.)
                if a & 0xFFF == 0 || a == c {
                    let page = a & !0xFFF;
                    if crate::hosted::page_held(page) {
                        a = page + 0x1000;
                        continue;
                    }
                }
                if unsafe { std::ptr::read_volatile(a as *const u64) } == word {
                    f(a, (lo, hi));
                }
                a += 8;
            }
            drop(readers);
            c = end;
            tries = 0;
        }
    }
}

fn scan() -> Tree {
    let Some(b) = checked() else { return Tree { nodes: Vec::new(), children: HashMap::new(), spans: Vec::new() } };
    let vt = b + NODE_VTABLE;
    let mut found: Vec<(u64, (u64, u64))> = Vec::new();
    sweep(vt, NODE_BYTES, |a, m| found.push((a, m)));
    // The pools: node objects merged within the mapping each was found in (a page's gap at most).
    let mut spans: Vec<(u64, u64)> = Vec::new();
    let mut last_map = (0, 0);
    for &(a, m) in &found {
        match spans.last_mut() {
            Some(s) if m == last_map && a <= s.1 + 0x1000 => s.1 = s.1.max(a + NODE_BYTES),
            _ => spans.push((a, a + NODE_BYTES)),
        }
        last_map = m;
    }
    let nodes: Vec<u64> = found.into_iter().map(|(a, _)| a).collect();
    let mut t = Tree { nodes, children: HashMap::new(), spans };
    let children = guarded(&t, || {
        let mut children: HashMap<u64, Vec<u64>> = HashMap::new();
        // Ancestors outside the pools (another class's containers, scene roots): readable ones
        // get spans of their own, each checked once with the operating system (the table's lock
        // is not taken under the guest-reader lock).
        let mut extra: Vec<(u64, u64)> = Vec::new();
        let mut seen: std::collections::HashSet<u64> = std::collections::HashSet::new();
        for &n in &t.nodes {
            let p = parent(n);
            if p != 0 {
                children.entry(p).or_default().push(n);
            }
            let mut a = p;
            for _ in 0..64 {
                if a == 0 || !seen.insert(a) {
                    break;
                }
                if !t.holds(a) {
                    if a & 7 != 0 || !crate::mem::os_readable(a, NODE_BYTES) {
                        break;
                    }
                    extra.push((a, a + NODE_BYTES));
                }
                a = parent(a);
            }
        }
        (children, extra)
    });
    match children {
        Some((c, extra)) => {
            t.children = c;
            if !extra.is_empty() {
                t.spans.extend(extra);
                t.spans.sort_unstable();
                let mut merged: Vec<(u64, u64)> = Vec::with_capacity(t.spans.len());
                for (s, e) in t.spans.drain(..) {
                    match merged.last_mut() {
                        Some(m) if s <= m.1 => m.1 = m.1.max(e),
                        _ => merged.push((s, e)),
                    }
                }
                t.spans = merged;
            }
        }
        // The pools changed while the sweep ran: an empty index (scanned again on the next use).
        None => {
            crate::warn!("uistate: the UI's node pools changed during the scan: scanned again later");
            t.nodes.clear();
            t.spans.clear();
        }
    }
    t
}

/// Runs `f` with the node pools held mapped: each pool checked in the mapping table, then `f`
/// under the guest-reader lock when no mapping changed in between (`mem::table_view_held`;
/// the title unmaps memory at any time). None: a pool is no longer mapped (a new scan starts)
/// or the mappings kept changing; the caller tries again later. `f` must not take the mapping
/// table's lock (`mem::readable`, `mem::MEM`, `rescan`): decommits hold it while they wait here.
pub fn guarded<R>(t: &Tree, f: impl FnOnce() -> R) -> Option<R> {
    use std::sync::atomic::Ordering::SeqCst;
    for _ in 0..4 {
        let g0 = crate::mem::MAP_GEN.load(SeqCst);
        if let Some(&(s, e)) = t.spans.iter().find(|&&(s, e)| !crate::mem::host_readable(s, e - s)) {
            static GONE: std::sync::atomic::AtomicU32 = std::sync::atomic::AtomicU32::new(0);
            if GONE.fetch_add(1, Ordering::Relaxed) < 8 {
                crate::info!("uistate: a node pool [{s:#x}, {e:#x}) is no longer mapped: the nodes are scanned again");
            }
            *TREE.lock() = None;
            return None;
        }
        let readers = crate::mem::GUEST_READERS.read_recursive();
        if crate::mem::table_view_held(g0) {
            let r = f();
            drop(readers);
            return Some(r);
        }
    }
    None
}

fn rdf(a: u64) -> f32 {
    unsafe { std::ptr::read_volatile(a as *const f32) }
}

/// A node's parent (0: none). Read only for nodes in the index's checked ranges (`Tree::holds`)
/// under `guarded`.
pub fn parent(n: u64) -> u64 {
    unsafe { std::ptr::read_volatile((n + 0x2C8) as *const u64) }
}

pub fn flags(n: u64) -> u32 {
    unsafe { std::ptr::read_volatile((n + 0x2DC) as *const u32) }
}

/// A list's selected item.
pub const SELECTED: u32 = 0x1;
/// On the input focus's path: the screen's containers down to the list the player moves in, and
/// that list's item when the focus arrived with the screen or the list.
pub const FOCUS_PATH: u32 = 0x40;
/// Not shown: 0x2 on items the title hides (the main menu's last row), 0x4 on a scrolling list's
/// rows outside its window (a SETTINGS page's rows below the box: placed, but clipped).
pub const NOT_SHOWN: u32 = 0x6;

/// A node's definition (its UIElement) and the definition's design width and height. Under
/// `guarded` (its page asked of the operating system: the table's lock is not taken there).
pub fn definition_size(n: u64) -> Option<(f32, f32)> {
    let d = unsafe { std::ptr::read_volatile((n + 0x110) as *const u64) };
    if d < 0x1000 || !crate::mem::os_readable(d + 0xB0, 8) {
        return None;
    }
    Some((rdf(d + 0xB0), rdf(d + 0xB4)))
}

/// A node's rectangle relative to its parent (x, y, w, h).
pub fn local(n: u64) -> (f32, f32, f32, f32) {
    (rdf(n + 0x2FC), rdf(n + 0x300), rdf(n + 0x304), rdf(n + 0x308))
}

fn placed(v: f32) -> bool {
    v.is_finite() && v.abs() < 1.0e6
}

/// A node's position on the canvas (its and its ancestors' positions summed); None when one of
/// them is not placed, or lies outside the index's checked ranges (made since the scan).
pub fn absolute(t: &Tree, n: u64) -> Option<(f32, f32)> {
    let (mut x, mut y) = (0.0f32, 0.0f32);
    let mut a = n;
    for _ in 0..64 {
        if a == 0 {
            return Some((x, y));
        }
        if !t.holds(a) {
            return None;
        }
        let (lx, ly, _, _) = local(a);
        let p = parent(a);
        if !placed(lx) || !placed(ly) {
            // A scene's root without a placement is at the origin; elsewhere: not shown.
            if p == 0 && a != n {
                return Some((x, y));
            }
            return None;
        }
        x += lx;
        y += ly;
        a = p;
    }
    None
}

/// The canvas rectangle (x0, y0, x1, y1) covering a node and its placed descendants of some size.
pub fn bounds(t: &Tree, n: u64) -> Option<(f32, f32, f32, f32)> {
    let (ax, ay) = absolute(t, n)?;
    let mut acc: Option<(f32, f32, f32, f32)> = None;
    fn walk(t: &Tree, n: u64, ox: f32, oy: f32, depth: u32, acc: &mut Option<(f32, f32, f32, f32)>) {
        let (_, _, w, h) = local(n);
        if placed(w) && placed(h) && w > 1.0 && h > 1.0 {
            let r = (ox, oy, ox + w, oy + h);
            *acc = Some(match *acc {
                None => r,
                Some(a) => (a.0.min(r.0), a.1.min(r.1), a.2.max(r.2), a.3.max(r.3)),
            });
        }
        if depth > 12 {
            return;
        }
        for &k in t.children.get(&n).map(|v| v.as_slice()).unwrap_or(&[]) {
            let (kx, ky, _, _) = local(k);
            if placed(kx) && placed(ky) {
                walk(t, k, ox + kx, oy + ky, depth + 1, acc);
            }
        }
    }
    walk(t, n, ax, ay, 0, &mut acc);
    acc
}
