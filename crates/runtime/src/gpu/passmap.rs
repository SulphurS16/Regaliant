//! `passmap <flips>` (diagnosis, 2026-11-26 evening): for the next flips (at most 4), each
//! queue's compute dispatches and render passes in its recording order, with the program, the
//! images it reads and writes (T# base, size, format) and its attachments: where the title's
//! frame goes from the scene through the TAA and the post passes to the output and the HUD
//! (`passmap <queue> flip <n>:` lines; consecutive equal lines on a thread collapse into a
//! count). `tune par 0` keeps the graphics queue on one recorder, in order; the compute queues
//! record on their own threads. On the display-sized target every change of program is a line
//! (the output's passes and the HUD's), elsewhere only a change of attachments.

use super::images::Target;
use super::shaders::{Prepared, Resolved};
use std::cell::RefCell;
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering::Relaxed};

static ARMED: AtomicBool = AtomicBool::new(false);
static FROM: AtomicU64 = AtomicU64::new(0);
static TO: AtomicU64 = AtomicU64::new(0);

fn flip() -> u64 {
    crate::hle::libs::videoout::QUEUED_FLIP.load(Relaxed) + 1
}

/// `passmap <flips>`: the flips from the second next on.
pub fn arm(flips: u64) -> Result<(), String> {
    let n = flips.clamp(1, 4);
    let f = flip() + 2;
    FROM.store(f, Relaxed);
    TO.store(f + n, Relaxed);
    ARMED.store(true, Relaxed);
    crate::info!("passmap: flips {f} to {} (the queues' passes in order)", f + n - 1);
    Ok(())
}

/// Whether a map is armed (the draw and dispatch paths' one check).
#[inline]
pub fn armed() -> bool {
    ARMED.load(Relaxed)
}

/// The flip now when it is in the window (a flip past it disarms).
fn in_window() -> Option<u64> {
    let f = flip();
    if f >= TO.load(Relaxed) {
        ARMED.store(false, Relaxed);
        return None;
    }
    (f >= FROM.load(Relaxed)).then_some(f)
}

/// The images a stage reads and writes (T# base, size, format).
fn images(p: &Prepared) -> (Vec<String>, Vec<String>) {
    let (mut r, mut w) = (Vec::new(), Vec::new());
    for (i, res) in p.resources.iter().enumerate() {
        let Resolved::Image(t) = res else { continue };
        let s = format!("{:#x} {}x{} f{}", t.base, t.width, t.height, t.format);
        let list = if p.tr.plan.resources.get(i).is_some_and(|x| x.written) { &mut w } else { &mut r };
        if !list.contains(&s) {
            list.push(s);
        }
    }
    (r, w)
}

thread_local! {
    /// The thread's last line and how many times it repeated since.
    static LAST: RefCell<(String, u64)> = const { RefCell::new((String::new(), 0)) };
}

fn emit(queue: &str, flip: u64, line: String) {
    LAST.with(|l| {
        let mut l = l.borrow_mut();
        if l.0 == line {
            l.1 += 1;
            return;
        }
        if l.1 > 0 {
            crate::info!("passmap {queue} flip {flip}: (the line before x{} more)", l.1);
        }
        crate::info!("passmap {queue} flip {flip}: {line}");
        *l = (line, 0);
    });
}

/// A compute dispatch.
pub fn dispatch(queue: &str, p: &Prepared, groups: [u32; 3]) {
    let Some(f) = in_window() else { return };
    let (r, w) = images(p);
    emit(queue, f, format!("cs {:#x} {}x{}x{} writes [{}] reads [{}]", p.addr, groups[0], groups[1], groups[2], w.join(", "), r.join(", ")));
}

/// A draw that begins other attachments (`new_pass`), or any draw into a display-sized target.
pub fn draw(queue: &str, vs: u64, ps: Option<&Prepared>, colors: &[Option<Target>], depth: &Option<Target>, new_pass: bool) {
    let display = colors.iter().flatten().any(|c| c.width >= 3840 || (c.width, c.height) == crate::dispmode::output_size());
    if !new_pass && !display {
        return;
    }
    let Some(f) = in_window() else { return };
    let t: Vec<String> = colors
        .iter()
        .enumerate()
        .filter_map(|(i, c)| c.as_ref().map(|c| format!("c{i} {:#x} {}x{} {:?}", c.key.base, c.width, c.height, c.format)))
        .chain(depth.iter().map(|d| format!("z {:#x} {}x{}", d.key.base, d.width, d.height)))
        .collect();
    let (r, w) = ps.map(images).unwrap_or_default();
    emit(queue, f, format!("draw vs {vs:#x} ps {:#x} targets [{}] reads [{}] stores [{}]", ps.map_or(0, |p| p.addr), t.join(", "), r.join(", "), w.join(", ")));
}
