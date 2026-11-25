//! The menu pointer: while a menu of the title is open (its UI manager's menu stack, `uistate`)
//! and the keyboard and mouse are the device in use, the window frees the cursor and shows the
//! port's pointer; moving it over an item of the focused list moves the title's selection there
//! at once: one d-pad press toward it makes the title's list navigation run, and a hook at that
//! navigation's entry (`select_hook`) turns the move into the list's own jump to the item (its
//! highlight, scrolling and one navigation sound). Lists the hook does not serve (grids) step
//! toward the item with d-pad presses as before. A left click on an item selects it (Cross,
//! after the selection reached it), a right click goes back (Circle), the wheel moves the
//! selection (Up/Down).
//!
//! The items: the focused node and its siblings (the children of its parent: a list's rows, a
//! grid's slots, the SETTINGS tabs), each covering its subtree's placed rectangles on the
//! title's canvas (1080 units high; 1920 wide at 16:9, `uistate::bounds`); the pointer is mapped from the window to the
//! canvas as the presenter fits the frame. Every list keeps a selected item (flag 0x1); the
//! focused one is the selected item of the list on the input focus's path (0x40).

use crate::input;
use crate::uistate;
use parking_lot::Mutex;
use std::sync::atomic::{AtomicBool, AtomicI32, AtomicU32, Ordering::Relaxed};
use std::time::{Duration, Instant};

/// Pointer mode now (the window frees the cursor and draws the pointer).
pub static ACTIVE: AtomicBool = AtomicBool::new(false);
/// The pointer in client pixels, and a count of moves.
pub static CLIENT_X: AtomicI32 = AtomicI32::new(-1);
pub static CLIENT_Y: AtomicI32 = AtomicI32::new(-1);
pub static MOVES: AtomicU32 = AtomicU32::new(0);

#[derive(Clone, Copy, Debug)]
pub enum Event {
    Left,
    Right,
    Wheel(i32),
}

static EVENTS: Mutex<Vec<Event>> = Mutex::new(Vec::new());

/// A pointer event from the window (or the control channel).
pub fn event(e: Event) {
    let mut g = EVENTS.lock();
    if g.len() < 32 {
        g.push(e);
    }
}

/// The pointer at canvas coordinates for tests (`pointer <x> <y>`): as if moved there.
static TEST_AT: Mutex<Option<(f32, f32)>> = Mutex::new(None);

pub fn test_move(x: f32, y: f32) {
    *TEST_AT.lock() = Some((x, y));
    MOVES.fetch_add(1, Relaxed);
    input::note_device(input::DEVICE_KBM);
}

/// Tests: the pointer at `(u, v)` of the picture as shown (1080 units high), placed in the
/// window's client area as a real mouse there would be (through Menu & HUD Size's mapping).
pub fn test_screen(u: f32, v: f32) -> Result<(), String> {
    let cw = crate::gpu::vk::window::CLIENT_W.load(Relaxed);
    let ch = crate::gpu::vk::window::CLIENT_H.load(Relaxed);
    let (ow, oh) = crate::dispmode::output_size();
    let (rx, ry, rw, rh) = crate::gpu::vk::present::fit(ow, oh, cw, ch);
    if rw == 0 || rh == 0 {
        return Err("no picture in the window".into());
    }
    let width = 1080.0 * ow as f32 / oh as f32;
    let x = rx as f32 + u / width * rw as f32;
    let y = ry as f32 + v / 1080.0 * rh as f32;
    *TEST_AT.lock() = None;
    CLIENT_X.store(x.round() as i32, Relaxed);
    CLIENT_Y.store(y.round() as i32, Relaxed);
    MOVES.fetch_add(1, Relaxed);
    input::note_device(input::DEVICE_KBM);
    crate::info!("pointer: test at client ({:.0}, {:.0}) of {cw}x{ch} (picture {u:.0},{v:.0}; canvas {:?})", x, y, canvas_pos());
    Ok(())
}

struct State {
    /// The item the pointer last moved onto (the focus is moved there) and its list.
    target: Option<u64>,
    target_group: u64,
    /// A click waits for the focus to reach the target.
    click: bool,
    /// The focused items last seen.
    focused: Vec<u64>,
    /// Lists whose horizontal moves take L1/R1 (a tab bar: Left/Right did not move it), and the
    /// last horizontal press (its list, its selected item then, when).
    shoulder: std::collections::HashSet<u64>,
    tried: Option<(u64, u64, Instant)>,
    next_press: Instant,
    /// Presses waiting their turn (spaced so the title sees each).
    queue: std::collections::VecDeque<&'static str>,
    moves_seen: u32,
    logged: u32,
    /// The target a direct jump was asked for (once per target).
    direct_tried: Option<u64>,
    /// A click on an item of another list of the screen (a page beside the focused list): the
    /// lists in focus then and until when the focus may take to move (after Cross into a page on
    /// the right, Circle back to a list on the left); the item under the pointer in the list the
    /// focus moved to is then the click's (an item found beforehand may be a hidden page's lying
    /// under the shown one).
    across: Option<(Vec<u64>, Instant)>,
    /// The items the selection passed while stepping toward the target: back on one of them, the
    /// target cannot be selected (a greyed row the list steps over) and the chase ends.
    chase: Vec<u64>,
    /// No press before then (a page coming in after a press that moved the focus into it).
    hold_until: Instant,
    /// A jump to an item of another group in the focused scene (the equipment grid's other
    /// parts: the grid's own jump, `grid_hook`, takes any of its cells): the item, until when,
    /// and whether a click follows.
    jump: Option<(u64, Instant, bool)>,
    /// Presses toward the target in a row that did not move the selection.
    stuck: u32,
}

static STATE: Mutex<Option<State>> = Mutex::new(None);

/// The item the pointer asks the title's next list move to jump to (`select_hook`), and since
/// when (a request the title did not take within `DIRECT_WAIT` falls back to stepping).
static DIRECT: Mutex<Option<(u64, Instant)>> = Mutex::new(None);
const DIRECT_WAIT: Duration = Duration::from_millis(250);
/// Direct jumps made, lists the hook did not find the item in, and jumps in lists a data
/// provider fills.
static DIRECT_STATS: [std::sync::atomic::AtomicU64; 3] = [const { std::sync::atomic::AtomicU64::new(0) }; 3];
/// For a list a data provider fills: the rows from the selected row to the target in reading
/// order (top to bottom, left to right), with the target they were counted for.
static DIRECT_STEP: Mutex<Option<(u64, i64)>> = Mutex::new(None);

/// The rows from `from` to `to` among `group`'s shown items in reading order (lines by their
/// centres' heights, left to right within a line).
fn reading_step(t: &uistate::Tree, group: u64, from: u64, to: u64) -> Option<i64> {
    let mut rows: Vec<(u64, (f32, f32), f32)> = t
        .children
        .get(&group)?
        .iter()
        .copied()
        .filter(|&k| uistate::flags(k) & uistate::NOT_SHOWN == 0)
        .filter_map(|k| uistate::bounds(t, k).map(|r| (k, center(r), r.3 - r.1)))
        .filter(|&(_, _, h)| h > 1.0)
        .collect();
    rows.sort_by(|a, b| a.1 .1.total_cmp(&b.1 .1));
    // Lines: a row more than half a row's height below the line's first starts the next.
    let mut keyed: Vec<(usize, f32, u64)> = Vec::new();
    let (mut line, mut top) = (0usize, f32::NAN);
    for &(k, (cx, cy), h) in &rows {
        if top.is_nan() || cy - top > h / 2.0 {
            if !top.is_nan() {
                line += 1;
            }
            top = cy;
        }
        keyed.push((line, cx, k));
    }
    keyed.sort_by(|a, b| a.0.cmp(&b.0).then(a.1.total_cmp(&b.1)));
    let a = keyed.iter().position(|e| e.2 == from)?;
    let b = keyed.iter().position(|e| e.2 == to)?;
    Some(b as i64 - a as i64)
}

/// The title's menu list selection change (eboot+0xe2a8d0, `titlefix`), at its entry with the
/// call's registers (`patch::Patcher::hook`: rax, rcx, rdx, rbx, rsp, rbp, rsi, rdi, r8 .. r15,
/// written back after the handler; the vector registers in the FXSAVE area below them, restored
/// from there). When the pointer waits for a jump to an item of this list (rdi) the move becomes
/// the list's own jump to that item: no direction (xmm0 zero), its index in esi, the other
/// arguments as the list's focus handler passes them (0).
pub extern "sysv64" fn select_hook(regs: *const u64, _arg: u32) {
    let mut g = DIRECT.lock();
    let Some((target, since)) = *g else { return };
    if since.elapsed() > DIRECT_WAIT {
        *g = None;
        return;
    }
    let regs = regs as *mut u64;
    unsafe {
        let list = *regs.add(7);
        if list < 0x10000 {
            return;
        }
        // The list's items: entries of three words, the item's node first (`items` at +0x158,
        // their end at +0x160).
        let (begin, end) = (*((list + 0x158) as *const u64), *((list + 0x160) as *const u64));
        if begin == 0 || end < begin || (end - begin) % 0x18 != 0 || end - begin > 0x18 * 4096 {
            return;
        }
        // A list a data provider fills (its pointer at +0x2a0): its rows are recycled, a row's
        // place is not its item's index. The item's index is the selected one's (+0x2b0, where
        // this function stores it) moved by the rows from the selected row to the item in reading
        // order, counted by the pointer on screen (`DIRECT_STEP`).
        if *((list + 0x2a0) as *const u64) != 0 {
            let sel = *((list + 0x2b0) as *const i32);
            let step = *DIRECT_STEP.lock();
            // (Only the target's own list: a press meant for another list (the storage screen's
            // panels) runs the focused list's move, which must not take the step.)
            let n = (end - begin) / 0x18;
            let here = (0..n).any(|i| *((begin + i * 0x18) as *const u64) == target);
            static REFUSED: AtomicU32 = AtomicU32::new(0);
            let Some(d) = step.filter(|&(t, d)| here && t == target && sel >= 0 && d != 0).map(|s| s.1) else {
                if REFUSED.fetch_add(1, Relaxed) < 8 {
                    crate::info!("pointer: a data list jump left to the title's step: the item in this list {here}, the step {:?} for {target:#x}, selected index {sel}", step);
                }
                DIRECT_STATS[1].fetch_add(1, Relaxed);
                return;
            };
            let idx = sel as i64 + d;
            // The provider's item count, as this function reads it right after (an index past it
            // would select the first item: hovering an empty row below a short list): its
            // cached handle when the scene's generation still matches (eboot+0xd21350 reads
            // [[list+0x18]+0x10]; the generation at +0x18 of that, the handle's at +0x2a8), the
            // count from eboot+0xf30540 (a virtual call on the title's thread, DIRECT released).
            drop(g);
            let Some(base) = crate::loader::main_module().map(|m| m.base) else { return };
            let scene = *((list + 0x18) as *const u64);
            let ctx = if scene > 0x10000 { *((scene + 0x10) as *const u64) } else { 0 };
            let prov = *((list + 0x2a0) as *const u64);
            // (The generation is the frame's: the handle is current only when the list was
            // already moved this frame. Otherwise the item's being one of the list's rows in use
            // bounds the index: an empty row below a short list is none of them.)
            let (gen_now, gen_held) = if ctx > 0x10000 { (*((ctx + 0x18) as *const i32), *((list + 0x2a8) as *const i32)) } else { (0, -1) };
            let count: extern "sysv64" fn(u64, u32) -> u32 = std::mem::transmute(base + 0xf30540);
            let n_items = if gen_now == gen_held && prov > 0x10000 { count(prov, 0) as i64 } else { i64::MAX };
            if idx < 0 || idx >= n_items {
                if REFUSED.fetch_add(1, Relaxed) < 8 {
                    crate::info!("pointer: a data list jump left to the title's step: index {idx} outside its {n_items} items");
                }
                DIRECT_STATS[1].fetch_add(1, Relaxed);
                return;
            }
            {
                let mut g = DIRECT.lock();
                if g.is_some_and(|(t, _)| t == target) {
                    *g = None;
                }
            }
            *regs.add(6) = idx as u64;
            *regs.add(2) = 0;
            *regs.add(1) = 0;
            *regs.add(8) = 0;
            let fx = ((regs as u64 - 8) & !15) - 512;
            std::ptr::write_bytes((fx + 160) as *mut u8, 0, 16);
            if DIRECT_STATS[2].fetch_add(1, Relaxed) < 4 {
                crate::info!("pointer: the title's data list jumps {d:+} rows to the item under the pointer (index {sel} -> {idx})");
            }
            return;
        }
        let n = (end - begin) / 0x18;
        let Some(i) = (0..n).find(|&i| *((begin + i * 0x18) as *const u64) == target) else {
            DIRECT_STATS[1].fetch_add(1, Relaxed);
            return;
        };
        *g = None;
        *regs.add(6) = i;
        *regs.add(2) = 0;
        *regs.add(1) = 0;
        *regs.add(8) = 0;
        // xmm0: the trampoline saved the flags at regs - 8, aligned the stack down to 16 and put
        // the FXSAVE area 512 bytes below; XMM0 is at byte 160 of it.
        let fx = ((regs as u64 - 8) & !15) - 512;
        std::ptr::write_bytes((fx + 160) as *mut u8, 0, 16);
    }
    if DIRECT_STATS[0].fetch_add(1, Relaxed) < 4 {
        crate::info!("pointer: the title's list jumps to the item under the pointer");
    }
}

/// The title's input grid cell selection (eboot+0xf648b0, `titlefix`), at its entry: when the
/// pointer waits for a jump to a cell of this grid (rdi), the move selects that cell: its column
/// (esi) and row (edx) found with the grid's own lookups, called here on the title's thread
/// (the row count of a column, eboot+0xf640c0; the cell at a column and row, eboot+0xf63db0;
/// the columns are the definition's entries of 40 bytes at +0xa0..+0xa8).
pub extern "sysv64" fn grid_hook(regs: *const u64, _arg: u32) {
    // (The request is copied out: DIRECT is not held across the title's own calls below, as the
    // vblank thread takes it under the guest-reader lock.)
    let target = {
        let mut g = DIRECT.lock();
        match *g {
            Some((_, since)) if since.elapsed() > DIRECT_WAIT => {
                *g = None;
                0
            }
            Some((target, _)) => target,
            None => 0,
        }
    };
    let Some(base) = crate::loader::main_module().map(|m| m.base) else { return };
    let regs = regs as *mut u64;
    unsafe {
        let grid = *regs.add(7);
        if grid < 0x10000 {
            return;
        }
        let def = *((grid + 0x110) as *const u64);
        if def < 0x10000 {
            return;
        }
        let (cb, ce) = (*((def + 0xa0) as *const u64), *((def + 0xa8) as *const u64));
        if ce < cb || (ce - cb) % 40 != 0 || ce - cb > 40 * 64 {
            return;
        }
        let rows: extern "sysv64" fn(u64, u32) -> u64 = std::mem::transmute(base + 0xf640c0);
        let cell: extern "sysv64" fn(u64, u32, u32) -> u64 = std::mem::transmute(base + 0xf63db0);
        // Every cell of the grid, for the pointer to find (`GRID_CELLS`).
        let mut all = Vec::new();
        for c in 0..((ce - cb) / 40) as u32 {
            let nr = rows(grid, c) as u32;
            if nr == u32::MAX || nr > 512 {
                continue;
            }
            all.extend((0..nr).map(|r| cell(grid, c, r)).filter(|&n| n > 0x10000));
        }
        {
            static LOGGED: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);
            static COUNT: AtomicU32 = AtomicU32::new(0);
            if LOGGED.swap(grid, Relaxed) != grid && COUNT.fetch_add(1, Relaxed) < 20 {
                crate::info!("pointer: input grid {grid:#x}: {} columns, {} cells", (ce - cb) / 40, all.len());
            }
        }
        *GRID_CELLS.lock() = Some((Instant::now(), all));
        if target == 0 {
            return;
        }
        for c in 0..((ce - cb) / 40) as u32 {
            let nr = rows(grid, c) as u32;
            if nr == u32::MAX || nr > 512 {
                continue;
            }
            for r in 0..nr {
                if cell(grid, c, r) == target {
                    {
                        let mut g = DIRECT.lock();
                        if g.is_some_and(|(t, _)| t == target) {
                            *g = None;
                        }
                    }
                    *regs.add(6) = c as u64;
                    *regs.add(2) = r as u64;
                    if DIRECT_STATS[0].fetch_add(1, Relaxed) < 4 {
                        crate::info!("pointer: the title's grid jumps to the cell under the pointer (column {c}, row {r})");
                    }
                    return;
                }
            }
        }
        DIRECT_STATS[1].fetch_add(1, Relaxed);
    }
}

/// The cells of the input grid the title moved in last (`grid_hook`, on its thread): the pointer
/// finds the grid's other parts there (the equipment screen's hands, armour, rings, items).
static GRID_CELLS: Mutex<Option<(Instant, Vec<u64>)>> = Mutex::new(None);

/// The cell of the last grid under the pointer (not one of the focused lists' items).
fn grid_cell_at(t: &uistate::Tree, found: &[u64], (x, y): (f32, f32)) -> Option<u64> {
    let g = GRID_CELLS.lock();
    let (when, cells) = g.as_ref()?;
    // (Only the grid of the focused cell: one of the focused items is its.)
    if when.elapsed() > Duration::from_secs(600) || !found.iter().any(|f| cells.contains(f)) {
        return None;
    }
    cells
        .iter()
        .copied()
        // (Shown with every ancestor, not disabled (0x10): the title's own navigation skips
        // such cells, and its cell selection does not check.)
        .filter(|&c| t.holds(c) && uistate::flags(c) & (uistate::NOT_SHOWN | 0x10) == 0 && root_shown(t, c).1)
        .filter_map(|c| uistate::bounds(t, c).map(|r| (c, r)))
        .filter(|(_, r)| x >= r.0 && x < r.2 && y >= r.1 && y < r.3)
        .min_by(|a, b| ((a.1 .2 - a.1 .0) * (a.1 .3 - a.1 .1)).total_cmp(&((b.1 .2 - b.1 .0) * (b.1 .3 - b.1 .1))))
        .map(|(c, _)| c)
}

/// Status: direct jumps made and moves in lists the hook did not serve.
pub fn direct_line() -> String {
    let (a, b, c) = (DIRECT_STATS[0].load(Relaxed), DIRECT_STATS[1].load(Relaxed), DIRECT_STATS[2].load(Relaxed));
    if a == 0 && b == 0 && c == 0 {
        return String::new();
    }
    format!("    menu pointer: {a} direct jumps to the item under the pointer, {c} in lists a data provider fills, {b} list moves without the item\n")
}

/// The pointer on the canvas (None: outside the frame, or unknown).
fn canvas_pos() -> Option<(f32, f32)> {
    if let Some(p) = *TEST_AT.lock() {
        return Some(p);
    }
    let (x, y) = (CLIENT_X.load(Relaxed), CLIENT_Y.load(Relaxed));
    if x < 0 || y < 0 {
        return None;
    }
    let cw = crate::gpu::vk::window::CLIENT_W.load(Relaxed);
    let ch = crate::gpu::vk::window::CLIENT_H.load(Relaxed);
    // The frame as the presenter fits it; the canvas 1080 units high, as wide as the output's
    // shape (wider than 1920 with another Aspect Ratio).
    let (ow, oh) = crate::dispmode::output_size();
    let (rx, ry, rw, rh) = crate::gpu::vk::present::fit(ow, oh, cw, ch);
    if rw == 0 || rh == 0 {
        return None;
    }
    let width = 1080.0 * ow as f32 / oh as f32;
    let u = (x as f32 - rx as f32) / rw as f32 * width;
    let v = (y as f32 - ry as f32) / rh as f32 * 1080.0;
    // Menu & HUD Size: the interface is drawn scaled around the screen's centre.
    let s = super::ui_size();
    let (u, v) = if s < 1.0 { ((u - width / 2.0) / s + width / 2.0, (v - 540.0) / s + 540.0) } else { (u, v) };
    ((0.0..width).contains(&u) && (0.0..1080.0).contains(&v)).then_some((u, v))
}

/// The focused items: the selected item (0x1) of each list on the input focus's path (its
/// container marked 0x40: the path runs from the screen down to the lists the player moves in,
/// e.g. the start menu's tab bar (L1/R1) and the SETTINGS tab list (Up/Down); it moves to a
/// page's list as the page is entered and back as it is left).
fn focused(t: &uistate::Tree) -> Vec<u64> {
    t.nodes
        .iter()
        .copied()
        .filter(|&n| uistate::flags(n) & uistate::SELECTED != 0)
        .filter(|&n| {
            // The path runs through the list (its parent), or, in the lists a data provider fills
            // (equipment, inventory), through a node inside the selected row (its child; on the
            // equipment screen's Items row two levels down), or ends at the item itself (the main
            // menu: its rows sit in a layout container off the path).
            let p = uistate::parent(n);
            uistate::flags(n) & uistate::FOCUS_PATH != 0 || (p != 0 && t.holds(p) && uistate::flags(p) & uistate::FOCUS_PATH != 0) || path_below(t, n, 3)
        })
        .filter(|&n| uistate::absolute(t, n).is_some())
        .collect()
}

/// Whether a node within `depth` levels below `n` is on the input focus's path.
fn path_below(t: &uistate::Tree, n: u64, depth: u32) -> bool {
    depth > 0 && t.children.get(&n).is_some_and(|k| k.iter().any(|&c| uistate::flags(c) & uistate::FOCUS_PATH != 0 || path_below(t, c, depth - 1)))
}

fn center(r: (f32, f32, f32, f32)) -> (f32, f32) {
    ((r.0 + r.2) / 2.0, (r.1 + r.3) / 2.0)
}

/// A click on an item's value widget: Left or Right by the side of the widget it lands on; None
/// for an item without one (a click there is Cross). The widgets, in the item's right part (the
/// nodes' own rectangles, shown ones): a choice's two arrows (the only two small parts of one
/// size, side by side at one height: an inventory row's icons are not a pair) with the value
/// between them, the side taken from their midpoint; a slider (a thin wide bar with a small knob
/// on it), the side taken from the knob.
fn value_side(t: &uistate::Tree, item: u64, (x, y): (f32, f32)) -> Option<&'static str> {
    let ib = uistate::bounds(t, item)?;
    let split = ib.0 + (ib.2 - ib.0) * 0.55;
    // (x0, y0, w, h) of the shown descendants whose centre lies right of the split.
    let mut parts: Vec<(f32, f32, f32, f32)> = Vec::new();
    let mut stack: Vec<(u64, u32)> = t.children.get(&item).map(|v| v.iter().map(|&k| (k, 0)).collect()).unwrap_or_default();
    while let Some((n, depth)) = stack.pop() {
        if uistate::flags(n) & uistate::NOT_SHOWN != 0 {
            continue;
        }
        if let Some((ax, ay)) = uistate::absolute(t, n) {
            let (_, _, w, h) = uistate::local(n);
            if w.is_finite() && h.is_finite() && w > 0.0 && h > 0.0 && w < 2000.0 && h < 1080.0 && ax + w / 2.0 > split {
                parts.push((ax, ay, w, h));
            }
        }
        if depth < 8 {
            if let Some(k) = t.children.get(&n) {
                stack.extend(k.iter().map(|&c| (c, depth + 1)));
            }
        }
    }
    // (Nested nodes of one rectangle count once: an arrow and its glyph.)
    parts.sort_by(|a, b| a.0.total_cmp(&b.0).then(a.1.total_cmp(&b.1)).then(a.2.total_cmp(&b.2)).then(a.3.total_cmp(&b.3)));
    parts.dedup_by(|a, b| (a.0 - b.0).abs() < 1.0 && (a.1 - b.1).abs() < 1.0 && (a.2 - b.2).abs() < 1.0 && (a.3 - b.3).abs() < 1.0);
    let small = |r: &(f32, f32, f32, f32)| r.2 > 4.0 && r.2 <= 64.0 && r.3 > 4.0 && r.3 <= 64.0;
    let cx = |r: &(f32, f32, f32, f32)| r.0 + r.2 / 2.0;
    let cy = |r: &(f32, f32, f32, f32)| r.1 + r.3 / 2.0;
    // A choice's arrows.
    for a in parts.iter().filter(|r| small(r)) {
        let same: Vec<&(f32, f32, f32, f32)> = parts.iter().filter(|b| small(b) && (b.2 - a.2).abs() < 1.0 && (b.3 - a.3).abs() < 1.0 && (cy(b) - cy(a)).abs() < 6.0).collect();
        if same.len() == 2 {
            let (l, r) = if cx(same[0]) < cx(same[1]) { (same[0], same[1]) } else { (same[1], same[0]) };
            // (A value between them: the choice's text or a count, as the deposit dialog's.)
            let between = parts.iter().any(|v| !same.contains(&v) && cx(v) > cx(l) + l.2 / 2.0 && cx(v) < cx(r) - r.2 / 2.0 && (cy(v) - cy(l)).abs() < 24.0);
            if cx(r) - cx(l) >= 40.0 && between && x >= l.0 - 24.0 && x < r.0 + r.2 + 24.0 && (y - cy(l)).abs() < 48.0 {
                return Some(if x < (cx(l) + cx(r)) / 2.0 { "left" } else { "right" });
            }
        }
    }
    // A slider.
    for bar in parts.iter().filter(|r| r.2 >= 150.0 && r.3 >= 2.0 && r.3 <= 24.0) {
        if x < bar.0 - 24.0 || x >= bar.0 + bar.2 + 24.0 || (y - cy(bar)).abs() > 40.0 {
            continue;
        }
        let knob = parts.iter().find(|k| small(k) && k.2 <= 48.0 && cx(k) >= bar.0 - 8.0 && cx(k) <= bar.0 + bar.2 + 8.0 && (cy(k) - cy(bar)).abs() < 24.0);
        if let Some(k) = knob {
            return Some(if x < cx(k) { "left" } else { "right" });
        }
    }
    None
}

/// The vblank thread, each vblank.
pub fn tick() {
    // The node index is built once the title shows its first menu (the UI's node pools exist
    // by then; a scan of some seconds, usually done before the player uses the pointer).
    static STARTED: AtomicBool = AtomicBool::new(false);
    if !STARTED.load(Relaxed) && uistate::menu_depth().unwrap_or(0) > 0 && uistate::tree().is_some() {
        STARTED.store(true, Relaxed);
    }
    // (Not while the system text box is open over a menu: the character's name. The box takes
    // the keys; the pointer has nothing to point at there.)
    let menu = uistate::menu_depth().unwrap_or(0) > 0 && !crate::hle::libs::ime_dialog::active() && !crate::hle::libs::savedata::list_active();
    // Only where there is something to point at: a menu with a list the index finds (not a
    // cutscene's skip prompt, a loading screen or a title card; `POINTABLE`, kept a moment over a
    // screen's change so the pointer does not blink between menus).
    let pointable = POINTABLE.lock().is_some_and(|t| t.elapsed() < POINTABLE_HOLD);
    let want = menu && pointable && input::last_device() == input::DEVICE_KBM;
    if ACTIVE.swap(want, Relaxed) != want {
        crate::info!("pointer: {}", if want { "on (a menu with items is open, keyboard and mouse in use)" } else { "off" });
        crate::gpu::vk::window::pointer_mode_changed();
        EVENTS.lock().clear();
        if !want {
            *TEST_AT.lock() = None;
        }
    }
    // The focus is followed while any menu is open (the pad's moves too: the pointer then
    // starts from the item the player left), forgotten when the menus close.
    if !menu {
        *STATE.lock() = None;
        return;
    }
    // Another menu screen: its rows may be nodes made for it (lists a data provider fills): the
    // index is made again (in the background; the old one serves meanwhile).
    {
        static TOP: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);
        let top = uistate::menu_top().unwrap_or(0);
        if TOP.swap(top, Relaxed) != top {
            // (Another screen: an earlier grid's cells and a step counted for an earlier list
            // no longer apply; node pools are reused across screens.)
            *GRID_CELLS.lock() = None;
            *DIRECT_STEP.lock() = None;
            if top != 0 && STARTED.load(Relaxed) {
                uistate::rescan_screen();
            }
        }
        uistate::screen_tick();
    }
    let Some(tree) = uistate::tree() else { return };
    // The nodes are read with their pools held mapped (`uistate::guarded`; a skipped vblank
    // when the title's mappings change meanwhile); a new scan is asked for after it.
    let kbm = input::last_device() == input::DEVICE_KBM;
    if uistate::guarded(&tree, || tick_nodes(&tree, want, kbm)) == Some(true) {
        uistate::rescan();
    }
}

/// When the last screen with items to point at was seen (`tick`'s condition for the pointer).
static POINTABLE: Mutex<Option<Instant>> = Mutex::new(None);
const POINTABLE_HOLD: Duration = Duration::from_millis(400);
/// Items at least this wide and this tall (canvas units) are panels, not the pointer's targets
/// (the storage screen's are 505x700 and 625x700; a wide list's rows are ~1200x132).
const PANEL_SIDE: f32 = 400.0;

/// Since when the screen has had items without a break, and how long before the pointer shows.
static POINTABLE_SINCE: Mutex<Option<Instant>> = Mutex::new(None);
const POINTABLE_SETTLE: Duration = Duration::from_millis(120);

/// A row of tabs: items side by side across the top of the screen (the start menu's tab bar)
/// or above another focused list (the inventory's categories over its items). The pointer does
/// not switch them as it passes over; a click switches without pressing Cross. The row's top.
fn tab_row(t: &uistate::Tree, group: u64, found: &[u64]) -> Option<f32> {
    let kids = t.children.get(&group)?;
    // (The items' own rectangles, not their subtrees': the selected tab's glow reaches lower.)
    let rows: Vec<(f32, f32)> = kids
        .iter()
        .filter(|&&k| uistate::flags(k) & uistate::NOT_SHOWN == 0)
        .filter_map(|&k| {
            let (_, y) = uistate::absolute(t, k)?;
            let (_, _, _, h) = uistate::local(k);
            (h.is_finite() && h > 0.0 && h < 1080.0).then_some((y, y + h))
        })
        .collect();
    if rows.len() < 2 || !rows.iter().all(|r| (r.0 - rows[0].0).abs() < 4.0) {
        return None;
    }
    let (top, bottom) = (rows[0].0, rows.iter().map(|r| r.1).fold(f32::MIN, f32::max));
    let above_list = found.iter().any(|&f| uistate::parent(f) != group && uistate::absolute(t, f).is_some_and(|(_, y)| y >= bottom - 1.0));
    ((top + bottom) / 2.0 < 200.0 || above_list).then_some(top)
}

/// A node's scene root (its topmost ancestor in the index) and whether the node and every
/// ancestor are shown (no `uistate::NOT_SHOWN` flag).
fn root_shown(t: &uistate::Tree, n: u64) -> (u64, bool) {
    let (mut a, mut shown) = (n, true);
    for _ in 0..64 {
        shown &= uistate::flags(a) & uistate::NOT_SHOWN == 0;
        let p = uistate::parent(a);
        if p == 0 || !t.holds(p) {
            break;
        }
        a = p;
    }
    (a, shown)
}

/// The screen's other lists (a container with a selected item: every list keeps one, in the
/// scenes of the focused items, shown) and the item of theirs under the pointer: (list, item).
fn other_item_at(t: &uistate::Tree, found: &[u64], (x, y): (f32, f32)) -> Option<(u64, u64, bool)> {
    let roots: Vec<u64> = found.iter().map(|&f| root_shown(t, f).0).collect();
    let groups: Vec<u64> = found.iter().map(|&n| uistate::parent(n)).collect();
    let mut lists: Vec<u64> = Vec::new();
    for &n in &t.nodes {
        if uistate::flags(n) & uistate::SELECTED == 0 {
            continue;
        }
        let p = uistate::parent(n);
        if p == 0 || !t.holds(p) || groups.contains(&p) || lists.contains(&p) {
            continue;
        }
        // (Any scene: a SETTINGS category's page is a scene of its own beside the list's. Shown:
        // no hidden or clipped node on the item's path.)
        let (_, shown) = root_shown(t, n);
        if shown {
            lists.push(p);
        }
    }
    let mut best: Option<(f32, u64, u64)> = None;
    for &list in &lists {
        for &k in t.children.get(&list).map(|v| v.as_slice()).unwrap_or(&[]) {
            if uistate::flags(k) & uistate::NOT_SHOWN != 0 {
                continue;
            }
            if let Some(r) = uistate::bounds(t, k) {
                let area = (r.2 - r.0) * (r.3 - r.1);
                if x >= r.0 && x < r.2 && y >= r.1 && y < r.3 && area > 16.0 && best.is_none_or(|b| area < b.0) {
                    best = Some((area, list, k));
                }
            }
        }
    }
    static LOGGED: std::sync::atomic::AtomicU32 = std::sync::atomic::AtomicU32::new(0);
    if LOGGED.fetch_add(1, Relaxed) < 20 {
        // (Diagnosis: the selected items under the point, whatever their scene: root, shown.)
        let under: Vec<String> = t
            .nodes
            .iter()
            .copied()
            .filter(|&n| uistate::flags(n) & uistate::SELECTED != 0)
            .filter(|&n| uistate::bounds(t, n).is_some_and(|r| x >= r.0 && x < r.2 && y >= r.1 && y < r.3))
            .take(6)
            .map(|n| {
                let (root, shown) = root_shown(t, n);
                format!("{n:#x} (root {root:#x}, shown {shown}, flags {:#x})", uistate::flags(n))
            })
            .collect();
        crate::info!("pointer: a click off the focused lists at ({x:.0}, {y:.0}): {} other lists in the scene, item {:?}; the focused items' roots {:x?}; selected items there: {}", lists.len(), best.map(|b| b.2), roots, under.join(", "));
    }
    best.map(|(_, list, item)| (list, item, roots.contains(&root_shown(t, item).0)))
}

/// A jump to `item` of another group in the focused scene: the title's next list or grid move
/// is asked to land there (`DIRECT`), with one press from the nearest focused item toward it.
fn jump_to(s: &mut State, t: &uistate::Tree, found: &[u64], item: u64, click: bool) {
    let Some(tb) = uistate::bounds(t, item) else { return };
    let (tx, ty) = center(tb);
    let near = found.iter().filter_map(|&f| uistate::bounds(t, f).map(|r| center(r))).min_by(|a, b| ((a.0 - tx).powi(2) + (a.1 - ty).powi(2)).total_cmp(&((b.0 - tx).powi(2) + (b.1 - ty).powi(2))));
    let Some((fx, fy)) = near else { return };
    let dir = if (ty - fy).abs() >= (tx - fx).abs() { if ty > fy { "down" } else { "up" } } else if tx > fx { "right" } else { "left" };
    s.target = None;
    s.click = false;
    *DIRECT_STEP.lock() = None;
    *DIRECT.lock() = Some((item, Instant::now()));
    s.queue.push_back(dir);
    s.jump = Some((item, Instant::now() + Duration::from_millis(500), click));
}

/// `tick`'s part that reads the UI nodes (under `uistate::guarded`): true when the index
/// misses the focused item (a new scan wanted).
fn tick_nodes(tree: &std::sync::Arc<uistate::Tree>, want: bool, kbm: bool) -> bool {
    let mut g = STATE.lock();
    let s = g.get_or_insert_with(|| State {
        target: None,
        target_group: 0,
        click: false,
        focused: Vec::new(),
        shoulder: Default::default(),
        tried: None,
        next_press: Instant::now(),
        queue: Default::default(),
        moves_seen: MOVES.load(Relaxed),
        logged: 0,
        direct_tried: None,
        across: None,
        chase: Vec::new(),
        hold_until: Instant::now(),
        jump: None,
        stuck: 0,
    });
    let found = focused(tree);
    {
        // Shown once a screen has had items for a moment, kept while it has them: a cutscene's
        // first frames can carry the closing menu's items for a frame or two (the pointer
        // blinked on for 0.2 s as the opening cutscene began, -141417).
        let now = Instant::now();
        let mut since = POINTABLE_SINCE.lock();
        if found.is_empty() {
            *since = None;
        } else if ACTIVE.load(Relaxed) || now.duration_since(*since.get_or_insert(now)) >= POINTABLE_SETTLE {
            *POINTABLE.lock() = Some(now);
        }
    }
    // (The pointer turns on at the next vblank once the screen has items: `tick`.)
    let _ = kbm;
    if found != s.focused {
        for &n in found.iter().filter(|n| !s.focused.contains(n)) {
            if s.logged < 60 {
                s.logged += 1;
                crate::info!("pointer: focus on node {n:#x} at {:?}{}", uistate::bounds(tree, n), if want { "" } else { " (followed, the pointer off)" });
            }
        }
        s.focused = found.clone();
    }
    super::bindings::tick(tree, &found);
    // A horizontal press that did not move its list: that list takes L1/R1.
    if let Some((group, before, when)) = s.tried {
        let now_sel = found.iter().copied().find(|&n| uistate::parent(n) == group);
        if now_sel != Some(before) {
            s.tried = None;
        } else if when.elapsed() > Duration::from_millis(400) {
            s.tried = None;
            if s.shoulder.insert(group) {
                crate::info!("pointer: the list {group:#x} moves with L1/R1");
            }
        }
    }
    if !want {
        s.target = None;
        s.click = false;
        s.queue.clear();
        s.moves_seen = MOVES.load(Relaxed);
        return false;
    }
    let events: Vec<Event> = std::mem::take(&mut *EVENTS.lock());
    let moves = MOVES.load(Relaxed);
    let moved = moves != s.moves_seen;
    s.moves_seen = moves;
    if found.is_empty() {
        // No focused item (a screen without a list, or one the index misses: scanned again).
        for e in events {
            match e {
                Event::Right => s.queue.push_back("circle"),
                Event::Wheel(n) => (0..n.unsigned_abs().min(4)).for_each(|_| s.queue.push_back(if n > 0 { "up" } else { "down" })),
                Event::Left => s.queue.push_back("cross"),
            }
        }
        pump(s);
        return true;
    }
    // The items: each focused item's list (its parent's children), where they cover the pointer:
    // the one whose centre is nearest (an item's glow or frame may reach into its neighbours').
    let pos = canvas_pos();
    let hovered: Option<(u64, u64)> = pos.and_then(|(x, y)| {
        let dist = |r: (f32, f32, f32, f32)| {
            let (cx, cy) = center(r);
            (cx - x).powi(2) + (cy - y).powi(2)
        };
        found
            .iter()
            .flat_map(|&f| {
                let group = uistate::parent(f);
                let siblings = tree.children.get(&group).map(|v| v.as_slice()).unwrap_or(&[]).iter().map(move |&n| (group, n));
                // A grid's other columns: the cells under the group's siblings of the focused
                // cell's size (as placed), taken as the focused group's (the move toward one runs
                // the grid's own selection, `grid_hook`).
                let (_, _, fw, fh) = uistate::local(f);
                let same = move |n: u64| {
                    let (_, _, w, h) = uistate::local(n);
                    fw > 1.0 && fh > 1.0 && (w - fw).abs() < 0.5 && (h - fh).abs() < 0.5
                };
                let cousins: Vec<(u64, u64)> = match fw.is_finite() {
                    true => tree
                        .children
                        .get(&if tree.holds(group) { uistate::parent(group) } else { 0 })
                        .map(|v| v.as_slice())
                        .unwrap_or(&[])
                        .iter()
                        .filter(|&&col| col != group)
                        .flat_map(|col| tree.children.get(col).map(|v| v.as_slice()).unwrap_or(&[]).iter())
                        .filter(|&&n| same(n))
                        .map(|&n| (group, n))
                        .collect(),
                    false => Vec::new(),
                };
                siblings.chain(cousins)
            })
            .filter(|&(_, n)| uistate::flags(n) & uistate::NOT_SHOWN == 0)
            .filter_map(|(group, n)| uistate::bounds(tree, n).map(|r| (group, n, r)))
            .filter(|(_, _, r)| x >= r.0 && x < r.2 && y >= r.1 && y < r.3)
            // (Not whole panels: the storage screen's Inventory and Storage Box are the two items
            // of a list that a screen's own button switches, not a pointer's target.)
            .filter(|(_, _, r)| !(r.2 - r.0 >= PANEL_SIDE && r.3 - r.1 >= PANEL_SIDE))
            .min_by(|a, b| dist(a.2).total_cmp(&dist(b.2)))
            .map(|(group, n, _)| (group, n))
    });
    let focused_in = |group: u64| found.iter().copied().find(|&n| uistate::parent(n) == group);
    if moved {
        if let Some((group, h)) = hovered {
            if tab_row(tree, group, &found).is_none() && s.target != Some(h) {
                s.target = Some(h);
                s.target_group = group;
                s.chase.clear();
                // (A click goes to the item it was made on: moved on to another, it is dropped.)
                s.click = false;
            }
        }
    }
    // A jump into another group of the focused scene: done once the item is focused (then the
    // click, if any), given up after its time.
    if let Some((item, until, click)) = s.jump {
        if found.contains(&item) {
            s.jump = None;
            if click {
                s.queue.push_back(pos.and_then(|p| value_side(tree, item, p)).unwrap_or("cross"));
            }
        } else if Instant::now() > until {
            s.jump = None;
        }
    }
    // Another part of the focused input grid (the equipment screen's groups): its own jump.
    // (Other lists' items take no press on a hover: a press runs the focused list's move.)
    if moved && hovered.is_none() && s.jump.is_none() {
        if let Some(c) = pos.and_then(|p| grid_cell_at(tree, &found, p)) {
            if !found.contains(&c) {
                jump_to(s, tree, &found, c, false);
            }
        }
    }
    // A click on another list's item: once the focus arrived in that list, its item is the
    // target and the click goes on there.
    if let Some((before, until)) = s.across.clone() {
        match hovered {
            Some((group, h)) if !before.contains(&group) => {
                s.across = None;
                s.chase.clear();
                s.target = Some(h);
                s.target_group = group;
                s.click = true;
                // (The page is still coming in: a press at once is dropped.)
                s.hold_until = Instant::now() + Duration::from_millis(350);
            }
            _ if Instant::now() > until => s.across = None,
            _ => {}
        }
    }
    for e in events {
        match e {
            Event::Left => match hovered {
                // A tab: switched to (the presses that move its row), no Cross; the row's
                // shown tab: nothing to do.
                Some((group, h)) if tab_row(tree, group, &found).is_some() => {
                    if focused_in(group) != Some(h) {
                        if s.target != Some(h) {
                            s.chase.clear();
                        }
                        s.target = Some(h);
                        s.target_group = group;
                        s.click = false;
                        // (The top bar moves with L1/R1: no Left/Right first, which would move
                        // the page's list.)
                        if tab_row(tree, group, &found).is_some_and(|top| top < 200.0) {
                            s.shoulder.insert(group);
                        }
                    }
                }
                Some((group, h)) if focused_in(group) == Some(h) => {
                    let press = pos.and_then(|p| value_side(tree, h, p)).unwrap_or("cross");
                    if s.logged < 60 {
                        s.logged += 1;
                        crate::info!("pointer: click on the selected item {h:#x}: {press}");
                    }
                    s.queue.push_back(press);
                }
                Some((group, h)) => {
                    if s.target != Some(h) {
                        s.chase.clear();
                    }
                    s.target = Some(h);
                    s.target_group = group;
                    s.click = true;
                }
                // Another list's item (a page's button beside the list in focus): into that list
                // as a pad does (Cross into a page on the right, Circle back to the left), then
                // the click goes on there.
                None => {
                    if let Some(c) = pos.and_then(|p| grid_cell_at(tree, &found, p)) {
                        if found.contains(&c) {
                            s.queue.push_back("cross");
                        } else {
                            jump_to(s, tree, &found, c, true);
                        }
                        continue;
                    }
                    match pos.and_then(|p| other_item_at(tree, &found, p).map(|o| (o, p))) {
                        // A page on the right of the lists in focus: into it with
                        // Cross as a pad does, then the click there. (Nothing to the left: Circle
                        // would close a top-level menu.)
                        // Only from a narrow column of items at the left (the SETTINGS
                        // categories, whose Cross opens the page) into a page of another scene:
                        // from a grid Cross acts on the item (the storage screen's Deposit), and
                        // beside a wide list a passive detail panel keeps selected rows too (a
                        // shop's or the equipment's: Cross there would buy or equip; a review).
                        Some(((_, _, same_scene), (px, _))) if !same_scene => {
                            let column = |g: u64| {
                                let xs: Vec<f32> = tree.children.get(&g).map(|v| v.as_slice()).unwrap_or(&[]).iter().filter(|&&k| uistate::flags(k) & uistate::NOT_SHOWN == 0).filter_map(|&k| uistate::absolute(tree, k).map(|p| p.0)).collect();
                                xs.len() >= 2 && xs.iter().all(|x| (x - xs[0]).abs() < 4.0)
                            };
                            let fx = found.iter().map(|&f| uistate::parent(f)).filter(|&g| column(g)).filter_map(|g| uistate::bounds(tree, g)).filter(|r| r.2 < 768.0).map(|r| r.2).fold(f32::NAN, f32::max);
                            if !fx.is_nan() && px >= fx {
                                s.queue.push_back("cross");
                                s.across = Some((found.iter().map(|&f| uistate::parent(f)).collect(), Instant::now() + Duration::from_millis(900)));
                                s.target = None;
                            }
                        }
                        _ => {}
                    }
                }
            },
            Event::Right => {
                s.target = None;
                s.queue.push_back("circle");
            }
            Event::Wheel(n) => {
                s.target = None;
                (0..n.unsigned_abs().min(4)).for_each(|_| s.queue.push_back(if n > 0 { "up" } else { "down" }));
            }
        }
    }
    if !s.queue.is_empty() {
        pump(s);
        return false;
    }
    // Toward the target, one press at a time, from its list's selected item.
    if let Some(t) = s.target {
        let group = s.target_group;
        match focused_in(group) {
            Some(f) if f == t => {
                s.target = None;
                s.direct_tried = None;
                *DIRECT.lock() = None;
                s.next_press = s.next_press.min(Instant::now() + Duration::from_millis(40));
                if std::mem::take(&mut s.click) {
                    s.queue.push_back(pos.and_then(|p| value_side(tree, t, p)).unwrap_or("cross"));
                    pump(s);
                }
            }
            // Back on an item passed while stepping toward this target: the list steps over it
            // (a greyed row): the chase (and its click) ends here.
            Some(f) if s.chase.last() != Some(&f) && s.chase.contains(&f) => {
                if s.logged < 60 {
                    s.logged += 1;
                    crate::info!("pointer: the item under the pointer cannot be selected (the list steps over it)");
                }
                s.target = None;
                s.click = false;
                s.chase.clear();
            }
            // Presses that left the selection where it was (an empty row below a short list):
            // after three the chase ends.
            Some(f) if Instant::now() >= s.next_press.max(s.hold_until) && s.tried.is_none() && s.chase.last() == Some(&f) && s.stuck >= 2 => {
                if s.logged < 60 {
                    s.logged += 1;
                    crate::info!("pointer: the item under the pointer cannot be reached (the selection does not move)");
                }
                s.target = None;
                s.click = false;
                s.chase.clear();
                s.stuck = 0;
            }
            // (Not before `hold_until`: a page coming in drops presses.)
            Some(f) if Instant::now() >= s.next_press.max(s.hold_until) && s.tried.is_none() => {
                if s.chase.last() != Some(&f) {
                    s.chase.push(f);
                    s.stuck = 0;
                } else {
                    s.stuck += 1;
                }
                // A jump the title's list takes in one move (`select_hook`): asked once per
                // target; a request it did not take within DIRECT_WAIT steps as before.
                let direct = crate::titlefix::applied("menu pointer selects directly") && s.direct_tried != Some(t);
                if direct {
                    s.direct_tried = Some(t);
                    *DIRECT_STEP.lock() = reading_step(tree, group, f, t).map(|d| (t, d));
                    *DIRECT.lock() = Some((t, Instant::now()));
                }
                if let (Some(a), Some(b)) = (uistate::bounds(tree, f), uistate::bounds(tree, t)) {
                    let ((ax, ay), (bx, by)) = (center(a), center(b));
                    let dir = if (by - ay).abs() >= (bx - ax).abs() {
                        if by > ay { "down" } else { "up" }
                    } else {
                        let shoulder = s.shoulder.contains(&group);
                        if !shoulder {
                            s.tried = Some((group, f, Instant::now()));
                        }
                        match (bx > ax, shoulder) {
                            (true, false) => "right",
                            (false, false) => "left",
                            (true, true) => "r1",
                            (false, true) => "l1",
                        }
                    };
                    press(dir);
                    s.next_press = Instant::now() + if direct { DIRECT_WAIT } else { Duration::from_millis(110) };
                } else {
                    s.target = None;
                }
            }
            Some(_) => {}
            // The target's list lost the focus (another screen): forget it.
            None => {
                s.target = None;
                s.click = false;
            }
        }
    }
    false
}

/// The next queued press, when its turn has come.
fn pump(s: &mut State) {
    if Instant::now() >= s.next_press && Instant::now() >= s.hold_until {
        if let Some(name) = s.queue.pop_front() {
            press(name);
            s.next_press = Instant::now() + Duration::from_millis(110);
        }
    }
}

fn press(name: &str) {
    // A pad press that does not make the pad the device in use.
    let _ = input::press_quiet(name, Duration::from_millis(60));
}
