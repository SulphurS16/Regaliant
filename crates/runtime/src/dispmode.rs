//! The title's display mode as a render-resolution choice (PPSA01341
//! 01.007.000; FINDINGS.md 2026-10-21 evening).
//!
//! The title's video-out object picks its render and display targets by a
//! mode (+0x3C), set each frame from eboot+0x9fe6b2 through eboot+0x90a600:
//! 3 unless the settings (0x2F13B40) say otherwise (+0xD38, then +0xD30, the
//! options' value: 3 at the gate). eboot+0x90a600 takes the render and
//! output sizes per mode: 1 renders 1600x900 for a 1920x1080 output, 2
//! 1920x1080 for 1920x1080, 3 2560x1440 for 3840x2160, 4 3840x2160 for
//! 3840x2160, 5 2688x1512 for 3840x2160 (with a flag); others 1920x1080. The
//! title starts in mode 2 and switches to 3 when the options load.
//!
//! Two more readers make the same choice each frame: eboot+0xb26030 keeps the
//! effective mode at the settings' +0xD28 and calls eboot+0x934260 when it
//! changes, and eboot+0xde88c0 derives "a 4K output mode" (3-5), the UI's
//! scale (mode 2 held at the display call alone drew the HUD at twice its
//! size, cut off at the right).
//!
//! The output size (experiment, `DS_OUTPUT_SIZE=<w>x<h>`): modes 3-5 output
//! 3840x2160, a size that appears at five places of the video-out object
//! and one of the UI: the sizes handed to the render resolution manager
//! (+0x14 of the size block built by eboot+0x90a600 and by the constructor
//! eboot+0x90ad80; +0x0C, 3840x2160 in every mode, stays: the largest size),
//! the two display targets the constructor makes (eboot+0x90af74,
//! eboot+0x90b061; their 1920x1080 views for modes 1-2 stay), the current
//! mode's sizes as the getter eboot+0x90ab90 reports them from its per-mode
//! tables (the UI makes its target from them, eboot+0xdedc59), and the UI's
//! canvas (eboot+0xde85a7: 3840x2160 for modes 3-5, else 1920x1080, unless
//! an override is set). Each is replaced where it is 3840x2160.
//!
//! The render scale (the PC tab's Render Scale, FINDINGS.md 2026-11-06 night): the render size
//! follows the window's 16:9 area times the setting (100%: true resolution), at most the
//! largest size the title makes targets for (3840x2160), applied in the 4K-output modes 3-5
//! only. A new size is applied by a mode pulse: the display call is held at another 4K-output
//! mode for two of its calls, so the title runs its set-display-mode twice (as when the player
//! changes Visual Mode) and makes its targets at the new size each time (`tick`).
//!
//! The aspect ratio (the PC tab's Aspect Ratio, from the start; FINDINGS.md 2026-11-06 night):
//! a shape other than 16:9 makes the output W x 2160 (the UI's canvas with it: its 4K sprites
//! keep their size, the HUD reaches the screen's edges), the render size the same shape, and the
//! video-out object's aspect (+4, 16/9 as the title makes it) the output's, which its cameras
//! hand their projection: the view widens (Hor+). The display targets are made once, at the
//! video-out constructor: a change applies at the next start.
//!
//! `DS_DISPLAY_MODE=<1-5>` holds the mode from the first frame: at all three
//! the register holding the options' value (+0xD30, which has the last say)
//! becomes the chosen mode just before its `test`/`cmovne` (`hook_rax` at
//! eboot+0x9fe6a8 and eboot+0xb2606e, `hook_r15` at eboot+0xde88c0). The
//! option itself is left alone (the title saves its options).

use std::sync::atomic::{AtomicU32, AtomicU64, Ordering};

/// The chosen mode (0: the title's own).
static MODE: AtomicU32 = AtomicU32::new(0);
/// The chosen render size (width << 32 | height; 0: the mode's own).
static RENDER: AtomicU64 = AtomicU64::new(0);
/// Times the render size was set.
static RENDER_SET: AtomicU64 = AtomicU64::new(0);
/// Frames the mode was held.
static HELD: AtomicU64 = AtomicU64::new(0);
/// The PC tab's render scale (percent of the window's 16:9 area; 0: the title's own size).
pub static SCALE_PCT: AtomicU32 = AtomicU32::new(0);
/// `DS_RENDER_SIZE` set the render size (the setting then leaves it alone).
static RENDER_FROM_ENV: std::sync::atomic::AtomicBool = std::sync::atomic::AtomicBool::new(false);
/// The options' mode at the title's display call (the pulse's other mode is chosen from it).
static TITLE_MODE: AtomicU32 = AtomicU32::new(0);
/// A mode pulse: display calls left that see PULSE_MODE.
static PULSE_LEFT: AtomicU32 = AtomicU32::new(0);
static PULSE_MODE: AtomicU32 = AtomicU32::new(0);
/// Pulses made (render size changes applied).
static PULSES: AtomicU64 = AtomicU64::new(0);
/// The render size the setting asks for, as last seen, and for how many ticks (a resized
/// window settles before the targets are made again).
static WANT: AtomicU64 = AtomicU64::new(0);
static WANT_TICKS: AtomicU32 = AtomicU32::new(0);
/// A render size asked for in a run without a window by the control command
/// `rsize` (width << 32 | height; 0: the title's own), for in-run
/// comparisons: applied by a mode pulse as the Render Scale is.
static HEADLESS_WANT: AtomicU64 = AtomicU64::new(0);

/// `rsize <w>x<h>` (0x0: the title's own size again; no window only).
pub fn set_headless_size(w: u32, h: u32) -> Result<(), String> {
    if crate::gpu::vk::present::area().is_some() {
        return Err("rsize: a window's size follows the Render Scale".into());
    }
    if (w, h) != (0, 0) && !((320..=3840).contains(&w) && (180..=2160).contains(&h)) {
        return Err(format!("rsize: {w}x{h} is not a size (320x180 to 3840x2160, or 0x0)"));
    }
    HEADLESS_WANT.store((w as u64) << 32 | h as u64, Ordering::Relaxed);
    crate::info!("rsize: render size {w}x{h} asked for (applied by a mode pulse)");
    Ok(())
}

/// The chosen output size for modes 3-5 (width << 32 | height; 0: the
/// title's 3840x2160).
static OUTPUT: AtomicU64 = AtomicU64::new(0);
/// Times the output size was set: mode change, constructor, targets, UI
/// canvas, size getter.
static OUTPUT_SET: [AtomicU64; 5] = [const { AtomicU64::new(0) }; 5];

/// The aspect the title's cameras project with (f32 bits; 0: the title's 16:9): the video-out
/// object's +4 (eboot+0x2F0A988 holds the object; its constructor stores 16/9 at eboot+0x90ae7f),
/// which the gameplay camera (eboot+0xcdae49) and another (eboot+0xdeabf5) hand the projection
/// function eboot+0x8e71c0 with their field of view.
pub static ASPECT: AtomicU32 = AtomicU32::new(0);
static ASPECT_SET: AtomicU64 = AtomicU64::new(0);

fn apply_aspect() {
    let a = ASPECT.load(Ordering::Relaxed);
    let base = crate::pcsettings::convar::base();
    if a == 0 || base == 0 {
        return;
    }
    let vo = unsafe { ((base + 0x2F0A988) as *const u64).read_volatile() };
    if vo == 0 {
        return;
    }
    let p = (vo + 4) as *mut u32;
    let cur = unsafe { p.read_volatile() };
    if cur != a {
        unsafe { p.write_volatile(a) };
        if ASPECT_SET.fetch_add(1, Ordering::Relaxed) < 4 {
            crate::info!("aspect {:.4} set for the title's cameras (was {:.4})", f32::from_bits(a), f32::from_bits(cur));
        }
    }
}

/// Render and output sizes of a mode (eboot+0x90a600's table).
pub fn sizes(mode: u32) -> ((u32, u32), (u32, u32)) {
    match mode {
        1 => ((1600, 900), (1920, 1080)),
        3 => ((2560, 1440), (3840, 2160)),
        4 => ((3840, 2160), (3840, 2160)),
        5 => ((2688, 1512), (3840, 2160)),
        _ => ((1920, 1080), (1920, 1080)),
    }
}

/// The Aspect Ratio setting's shapes (options 1-4): 16:9, 16:10, 21:9 (64:27), 32:9.
const ASPECTS: [f32; 4] = [16.0 / 9.0, 16.0 / 10.0, 64.0 / 27.0, 32.0 / 9.0];

/// The output size (the title's 3840x2160 unless an aspect or `DS_OUTPUT_SIZE` changed it).
pub fn output_size() -> (u32, u32) {
    let o = OUTPUT.load(Ordering::Relaxed);
    if o == 0 { (3840, 2160) } else { ((o >> 32) as u32, o as u32) }
}

/// The render size for the output's shape at the title's own size (no window, or the Render
/// Scale's Visual Mode): the mode's lines (1440 in mode 3) at the output's shape; 0 (the
/// title's own) at 16:9.
fn headless_render() -> u64 {
    let (ow, oh) = output_size();
    if (ow, oh) == (3840, 2160) {
        return 0;
    }
    let lines = match TITLE_MODE.load(Ordering::Relaxed) {
        m @ 3..=5 => sizes(m).0 .1,
        _ => 1440,
    } as u64;
    let w = ((lines as f32 * ow as f32 / oh as f32 / 8.0).round() as u64 * 8).min(7680);
    w << 32 | lines
}

/// The Aspect Ratio setting at start: Automatic takes the primary screen's shape when the
/// window opens fullscreen (16:9 in a window and in headless runs); a shape within 1% of 16:9
/// leaves the title's own output.
fn start_aspect() {
    if OUTPUT.load(Ordering::Relaxed) != 0 {
        return;
    }
    let choice = crate::pcsettings::store::value("Setting_PC_AspectRatio").unwrap_or(0);
    let a = match choice {
        1..=4 => ASPECTS[choice as usize - 1],
        _ => {
            let fullscreen = match std::env::var("DS_FULLSCREEN").ok().as_deref() {
                Some("1") | Some("2") => true,
                Some("0") => false,
                _ => crate::pcsettings::window_mode().is_some_and(|m| m != 0),
            };
            if !crate::gpu::vk::present::active() || !fullscreen {
                return;
            }
            let (w, h) = crate::platform::loader_hooks::desktop_size();
            if w <= 0 || h <= 0 {
                return;
            }
            w as f32 / h as f32
        }
    };
    if (a / (16.0 / 9.0) - 1.0).abs() < 0.01 {
        return;
    }
    let a = a.clamp(1.5, 3.6);
    let ow = (2160.0 * a / 8.0).round() as u32 * 8;
    OUTPUT.store((ow as u64) << 32 | 2160, Ordering::Relaxed);
    ASPECT.store((ow as f32 / 2160.0).to_bits(), Ordering::Relaxed);
    if !RENDER_FROM_ENV.load(Ordering::Relaxed) && !crate::gpu::vk::present::active() {
        RENDER.store(headless_render(), Ordering::Relaxed);
    }
    crate::info!("aspect ratio {a:.3} (Aspect Ratio option {choice}): output {ow}x2160, the cameras' view widened, the UI centred where it is 16:9");
}

/// Reads `DS_DISPLAY_MODE` and `DS_RENDER_SIZE` (called before the hooks are
/// installed).
pub fn init() {
    if let Ok(v) = std::env::var("DS_RENDER_SIZE") {
        let size = v.trim().split_once(['x', 'X']).and_then(|(w, h)| Some((w.trim().parse::<u32>().ok()?, h.trim().parse::<u32>().ok()?)));
        match size {
            Some((w, h)) if (320..=7680).contains(&w) && (180..=4320).contains(&h) => {
                RENDER.store((w as u64) << 32 | h as u64, Ordering::Relaxed);
                RENDER_FROM_ENV.store(true, Ordering::Relaxed);
                crate::info!("render size {w}x{h} (DS_RENDER_SIZE): the title's render targets at that size, its output as the mode has it");
            }
            _ => crate::warn!("DS_RENDER_SIZE={v}: not a size (WxH, 320x180 to 7680x4320); the mode's own is used"),
        }
    }
    if let Ok(v) = std::env::var("DS_OUTPUT_SIZE") {
        let size = v.trim().split_once(['x', 'X']).and_then(|(w, h)| Some((w.trim().parse::<u32>().ok()?, h.trim().parse::<u32>().ok()?)));
        match size {
            Some((w, h)) if (1920..=7680).contains(&w) && (1080..=2160).contains(&h) => {
                OUTPUT.store((w as u64) << 32 | h as u64, Ordering::Relaxed);
                if std::env::var("DS_OUTPUT_ASPECT").ok().as_deref() != Some("0") {
                    ASPECT.store((w as f32 / h as f32).to_bits(), Ordering::Relaxed);
                }
                crate::info!("output size {w}x{h} (DS_OUTPUT_SIZE, experiment): the title's 3840x2160 output, display targets and UI canvas at that size");
            }
            _ => crate::warn!("DS_OUTPUT_SIZE={v}: not a size (WxH, 1920x1080 to 7680x2160); the title's own is used"),
        }
    }
    start_aspect();
    let Ok(v) = std::env::var("DS_DISPLAY_MODE") else { return };
    match v.trim().parse::<u32>() {
        Ok(m @ 1..=5) => {
            let ((rw, rh), (ow, oh)) = sizes(m);
            MODE.store(m, Ordering::Relaxed);
            crate::info!("display mode {m} held (DS_DISPLAY_MODE): renders {rw}x{rh} for a {ow}x{oh} output");
        }
        Ok(0) => {}
        _ => crate::warn!("DS_DISPLAY_MODE={v}: not a mode (1-5); the title's own is used"),
    }
}

/// The mode held now: a pulse's, else `DS_DISPLAY_MODE`'s (0: none).
fn held_mode() -> u32 {
    if PULSE_LEFT.load(Ordering::Relaxed) > 0 { PULSE_MODE.load(Ordering::Relaxed) } else { MODE.load(Ordering::Relaxed) }
}

/// Before a mode choice's last `cmovne`: the register holding the options'
/// value (`regs` index: rax 0, r15 15) becomes the chosen mode.
fn hold(regs: *const u64, reg: usize) {
    let m = held_mode();
    if m == 0 {
        return;
    }
    unsafe { *(regs as *mut u64).add(reg) = m as u64 };
    HELD.fetch_add(1, Ordering::Relaxed);
}

/// eboot+0xb2606e (the effective mode, +0xD28).
pub extern "sysv64" fn hook_rax(regs: *const u64, _arg: u32) {
    hold(regs, 0);
}

/// eboot+0x9fe6a8 (the display mode call; eax the options' mode): the title's mode noted, the
/// held one given, a pulse counted down.
pub extern "sysv64" fn hook_display(regs: *const u64, _arg: u32) {
    let m = unsafe { *regs } as u32;
    if m != 0 {
        TITLE_MODE.store(m, Ordering::Relaxed);
    }
    hold(regs, 0);
    let left = PULSE_LEFT.load(Ordering::Relaxed);
    if left > 0 {
        PULSE_LEFT.store(left - 1, Ordering::Relaxed);
    }
}

/// The render size for a window area of `w`x`h` at `pct` percent: 16:9, the height a multiple
/// of 4 from 360 to 2160, the width the height's 16:9 rounded to a multiple of 4.
pub fn scaled_size(w: u32, h: u32, pct: u32) -> (u32, u32) {
    let (ow, oh) = output_size();
    // The output's shape fitted to the area, scaled; at most 2160 lines and the 3840x2160
    // frame's pixels (the most the title's targets were made for).
    let fit_h = (h as u64).min(w as u64 * oh as u64 / ow as u64);
    let mut fh = (fit_h * pct as u64 / 100).clamp(360, 2160);
    let max_px = 3840u64 * 2160;
    while fh > 360 && fh * fh * ow as u64 / oh as u64 > max_px {
        fh -= 4;
    }
    let fh = ((fh + 2) / 4 * 4) as u32;
    let fw = ((fh as u64 * ow as u64 / oh as u64 + 2) / 4 * 4).min(7680) as u32;
    (fw, fh)
}

/// Each vblank: the render size the setting and the window ask for; a settled change is applied
/// by a mode pulse (in the 4K-output modes; none in headless runs, which keep the title's size).
pub fn tick() {
    apply_aspect();
    if RENDER_FROM_ENV.load(Ordering::Relaxed) || PULSE_LEFT.load(Ordering::Relaxed) > 0 {
        return;
    }
    let pct = SCALE_PCT.load(Ordering::Relaxed);
    let want = match crate::gpu::vk::present::area() {
        Some((w, h)) if pct != 0 => {
            let (rw, rh) = scaled_size(w, h, pct);
            (rw as u64) << 32 | rh as u64
        }
        // No window: the title's own size, or its shape's for another aspect
        // (`rsize`: the size asked for).
        _ => match HEADLESS_WANT.load(Ordering::Relaxed) {
            0 => headless_render(),
            v => v,
        },
    };
    if WANT.swap(want, Ordering::Relaxed) != want {
        WANT_TICKS.store(0, Ordering::Relaxed);
        return;
    }
    if WANT_TICKS.fetch_add(1, Ordering::Relaxed) < 20 || want == RENDER.load(Ordering::Relaxed) {
        return;
    }
    let title = TITLE_MODE.load(Ordering::Relaxed);
    if !(3..=5).contains(&title) || MODE.load(Ordering::Relaxed) != 0 {
        return;
    }
    RENDER.store(want, Ordering::Relaxed);
    PULSE_MODE.store(if title == 4 { 3 } else { 4 }, Ordering::Relaxed);
    PULSE_LEFT.store(2, Ordering::Relaxed);
    let n = PULSES.fetch_add(1, Ordering::Relaxed);
    if n < 32 {
        if want == 0 {
            crate::info!("render size: the title's own again (mode {title})");
        } else {
            crate::info!("render size {}x{} ({pct}% of the window's {:?}): applied by a mode pulse ({title} -> {} -> {title})", want >> 32, want as u32, crate::gpu::vk::present::area(), PULSE_MODE.load(Ordering::Relaxed));
        }
    }
}

/// eboot+0xde88c0 (the UI's 4K output test).
pub extern "sysv64" fn hook_r15(regs: *const u64, _arg: u32) {
    hold(regs, 15);
}

/// The packed size at `at` (two u32) becomes the chosen output size where it
/// is 3840x2160 (`site`: the OUTPUT_SET counter).
fn swap_output(at: u64, site: usize) {
    let o = OUTPUT.load(Ordering::Relaxed);
    if o == 0 || at == 0 {
        return;
    }
    let p = at as *mut u32;
    unsafe {
        if (p.read_volatile(), p.add(1).read_volatile()) == (3840, 2160) {
            p.write_volatile((o >> 32) as u32);
            p.add(1).write_volatile(o as u32);
            if OUTPUT_SET[site].fetch_add(1, Ordering::Relaxed) < 4 {
                crate::info!("output size {}x{} set ({})", o >> 32, o as u32, ["mode change", "constructor", "display target", "UI canvas"][site]);
            }
        }
    }
}

/// eboot+0x90a6fc, in set display mode after it built the sizes it hands the
/// render resolution manager (eboot+0x90c430): the render size at
/// [rbp-0x4C] (width) and [rbp-0x48] (height) becomes the chosen one; the
/// output size ([rbp-0x3C]) the chosen one where it is 3840x2160.
pub extern "sysv64" fn hook_render_size(regs: *const u64, _arg: u32) {
    let rbp = unsafe { *regs.add(5) };
    swap_output(rbp.wrapping_sub(0x3C), 0);
    let r = RENDER.load(Ordering::Relaxed);
    if r == 0 {
        return;
    }
    // Only for the 4K-output modes (the boot's 1080p mode keeps its own).
    let out = rbp.wrapping_sub(0x3C);
    let (ow, oh) = unsafe { ((out as *const u32).read_volatile(), ((out + 4) as *const u32).read_volatile()) };
    let o = OUTPUT.load(Ordering::Relaxed);
    if (ow, oh) != (3840, 2160) && (o == 0 || (ow as u64, oh as u64) != (o >> 32, o & 0xFFFF_FFFF)) {
        return;
    }
    let (w, h) = ((r >> 32) as u32, r as u32);
    let at = rbp.wrapping_sub(0x4C);
    let old = unsafe { ((at as *const u32).read_volatile(), ((at + 4) as *const u32).read_volatile()) };
    unsafe {
        (at as *mut u32).write_volatile(w);
        ((at + 4) as *mut u32).write_volatile(h);
    }
    if RENDER_SET.fetch_add(1, Ordering::Relaxed) < 8 {
        crate::info!("render size {w}x{h} set (the mode's was {}x{})", old.0, old.1);
    }
}

/// eboot+0x90b278, in the video-out constructor before it hands its size
/// block (r15) to eboot+0x91c680 and the render resolution manager: the
/// output size (+0x14).
pub extern "sysv64" fn hook_ctor_sizes(regs: *const u64, _arg: u32) {
    let r15 = unsafe { *regs.add(15) };
    swap_output(r15.wrapping_add(0x14), 1);
}

/// eboot+0x90af7e and eboot+0x90b06b, the constructor's display targets
/// (edi width, esi height): the chosen output size.
pub extern "sysv64" fn hook_targets(regs: *const u64, _arg: u32) {
    let o = OUTPUT.load(Ordering::Relaxed);
    let (w, h) = unsafe { (*regs.add(7) as u32, *regs.add(6) as u32) };
    if o == 0 || (w, h) != (3840, 2160) {
        return;
    }
    unsafe {
        *(regs as *mut u64).add(7) = o >> 32;
        *(regs as *mut u64).add(6) = o & 0xFFFF_FFFF;
    }
    if OUTPUT_SET[2].fetch_add(1, Ordering::Relaxed) < 4 {
        crate::info!("output size {}x{} set (display target)", o >> 32, o as u32);
    }
}

/// eboot+0x90ac57, in the current mode's size getter (eboot+0x90ab90)
/// before it stores the output size (r9d width, r10d height).
pub extern "sysv64" fn hook_size_getter(regs: *const u64, _arg: u32) {
    let o = OUTPUT.load(Ordering::Relaxed);
    let (w, h) = unsafe { (*regs.add(9) as u32, *regs.add(10) as u32) };
    if o == 0 || (w, h) != (3840, 2160) {
        return;
    }
    unsafe {
        *(regs as *mut u64).add(9) = o >> 32;
        *(regs as *mut u64).add(10) = o & 0xFFFF_FFFF;
    }
    if OUTPUT_SET[4].fetch_add(1, Ordering::Relaxed) < 4 {
        crate::info!("output size {}x{} set (size getter)", o >> 32, o as u32);
    }
}

/// eboot+0xde85db, the UI's canvas size (two f32 at [rbp-0x30]) before its
/// layout pass: the chosen output size where it is 3840x2160.
pub extern "sysv64" fn hook_ui_canvas(regs: *const u64, _arg: u32) {
    let o = OUTPUT.load(Ordering::Relaxed);
    if o == 0 {
        return;
    }
    let p = unsafe { *regs.add(5) }.wrapping_sub(0x30) as *mut f32;
    unsafe {
        if (p.read_volatile(), p.add(1).read_volatile()) == (3840.0, 2160.0) {
            p.write_volatile((o >> 32) as f32);
            p.add(1).write_volatile(o as u32 as f32);
            if OUTPUT_SET[3].fetch_add(1, Ordering::Relaxed) < 4 {
                crate::info!("output size {}x{} set (UI canvas)", o >> 32, o as u32);
            }
        }
    }
}

/// eboot+0xa4f2e9, after the frame's copy of the settings (memcpy of 0xD8A
/// bytes to r14+0x390): the copy's option (+0xD30) becomes the chosen mode.
pub extern "sysv64" fn hook_snapshot(regs: *const u64, _arg: u32) {
    let m = MODE.load(Ordering::Relaxed);
    if m == 0 {
        return;
    }
    let r14 = unsafe { *regs.add(14) };
    if r14 != 0 {
        unsafe { ((r14 + 0x390 + 0xD30) as *mut u32).write_volatile(m) };
    }
}

/// Status line.
pub fn status() -> String {
    let (m, r) = (MODE.load(Ordering::Relaxed), RENDER.load(Ordering::Relaxed));
    let mut s = String::new();
    if m != 0 {
        s += &format!("    display mode {m} held ({} choices)\n", HELD.load(Ordering::Relaxed));
    }
    if r != 0 {
        s += &format!("    render size {}x{} set {} times ({}; {} mode pulses)\n", r >> 32, r as u32, RENDER_SET.load(Ordering::Relaxed), if RENDER_FROM_ENV.load(Ordering::Relaxed) { "DS_RENDER_SIZE".to_string() } else { format!("Render Scale {}%", SCALE_PCT.load(Ordering::Relaxed)) }, PULSES.load(Ordering::Relaxed));
    }
    let o = OUTPUT.load(Ordering::Relaxed);
    if o != 0 {
        let n: Vec<u64> = OUTPUT_SET.iter().map(|c| c.load(Ordering::Relaxed)).collect();
        s += &format!("    output size {}x{} set: mode changes {}, constructor {}, display targets {}, UI canvas {}, size getter {}\n", o >> 32, o as u32, n[0], n[1], n[2], n[3], n[4]);
    }
    s
}
