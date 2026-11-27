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
/// Times the mode's size getter was given the port's render size (`hook_size_getter`), for its
/// first log lines.
static GETTER_SET: AtomicU64 = AtomicU64::new(0);
/// Frames the mode was held.
static HELD: AtomicU64 = AtomicU64::new(0);
/// The PC tab's render scale (percent of the window's 16:9 area; 0: the title's own size).
pub static SCALE_PCT: AtomicU32 = AtomicU32::new(0);
/// The Upscaling mode's factor, used while DLSS, FSR or XeSS replaces the TAA
/// (`upscale::active`): the screen's size over the render size per side, x100
/// (`pcsettings::UPSCALE_FACTORS`; 150 Quality).
pub static UPSCALE_FACTOR: AtomicU32 = AtomicU32::new(150);
/// The upscaler's output size while an upscaler runs (width << 32 | height; 0: none): the screen's
/// size at the output's shape, at most the title's output. Above the render size the title's
/// composite is drawn again at that size from it (`gpu::upcomp`); at the render size (Native AA)
/// its colour goes into the TAA's output (`gpu::upmerge`).
static UPSCALE_OUT: AtomicU64 = AtomicU64::new(0);
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

/// The upscaler's output size while an upscaler runs (the screen's, at most the title's output).
pub fn upscale_output() -> Option<(u32, u32)> {
    let v = UPSCALE_OUT.load(Ordering::Relaxed);
    (v != 0).then_some(((v >> 32) as u32, v as u32))
}

/// Tests: `DS_UPSCALE_SCREEN=<w>x<h>` stands for the screen in a run without a window (the
/// upscalers' render and output sizes as at that screen).
fn test_screen() -> Option<(u32, u32)> {
    static V: std::sync::OnceLock<Option<(u32, u32)>> = std::sync::OnceLock::new();
    *V.get_or_init(|| {
        let v = std::env::var("DS_UPSCALE_SCREEN").ok()?;
        let size = v.trim().split_once(['x', 'X']).and_then(|(w, h)| Some((w.trim().parse::<u32>().ok()?, h.trim().parse::<u32>().ok()?)));
        match size {
            Some((w, h)) if (640..=7680).contains(&w) && (360..=4320).contains(&h) => {
                crate::info!("upscaler sizes as at a {w}x{h} screen (DS_UPSCALE_SCREEN)");
                Some((w, h))
            }
            _ => {
                crate::warn!("DS_UPSCALE_SCREEN={v}: not a size (WxH, 640x360 to 7680x4320)");
                None
            }
        }
    })
}

/// The render size and the upscaler's output size at the Upscaling mode's factor: the output the
/// screen's area at the output's shape (at most the title's output; the output itself without a
/// screen), the render that over the factor, rounded up to a multiple of 4 lines (at least 360),
/// so that each upscaler runs at its own mode's ratio.
fn upscaling_sizes(area: Option<(u32, u32)>) -> ((u32, u32), (u32, u32)) {
    let (ow, oh) = output_size();
    let (uw, uh) = match area {
        Some((w, h)) => scaled_for(w, h, 100, ow, oh),
        None => (ow, oh),
    };
    let f = UPSCALE_FACTOR.load(Ordering::Relaxed).clamp(100, 400) as u64;
    let lines = (uh as u64 * 100).div_ceil(f).div_ceil(4) * 4;
    let rh = lines.clamp(360, uh.max(360) as u64) as u32;
    let rw = ((rh as u64 * ow as u64 / oh as u64 + 2) / 4 * 4).min(7680) as u32;
    ((rw, rh), (uw, uh))
}

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

/// The render size now: the one applied, or the title's own for its mode.
pub fn render_size() -> (u32, u32) {
    let r = RENDER.load(Ordering::Relaxed);
    if r == 0 { sizes(TITLE_MODE.load(Ordering::Relaxed)).0 } else { ((r >> 32) as u32, r as u32) }
}

/// The title's own render size for its Visual Mode (2560x1440 in Performance), whatever size the
/// port holds.
pub fn title_render_size() -> (u32, u32) {
    sizes(TITLE_MODE.load(Ordering::Relaxed)).0
}

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
    };
    let (w, h) = shaped_lines(lines, ow, oh);
    (w as u64) << 32 | h as u64
}

/// `lines` (at most `max_lines`) at the output's shape `ow`x`oh`, the width a multiple of 8.
fn shaped_lines(lines: u32, ow: u32, oh: u32) -> (u32, u32) {
    let lines = (lines as u64).min(max_lines(ow, oh));
    let w = ((lines as f32 * ow as f32 / oh as f32 / 8.0).round() as u64 * 8).min(7680);
    (w as u32, lines as u32)
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
            use windows_sys::Win32::UI::WindowsAndMessaging::{GetSystemMetrics, SM_CXSCREEN, SM_CYSCREEN};
            let (w, h) = unsafe { (GetSystemMetrics(SM_CXSCREEN), GetSystemMetrics(SM_CYSCREEN)) };
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

/// The render size for a window area of `w`x`h` at `pct` percent: the output's shape, the height
/// a multiple of 4 from 360 to `max_lines` (2160, or 1512 at shapes wider than 16:9), the width
/// the height's at that shape rounded to a multiple of 4.
pub fn scaled_size(w: u32, h: u32, pct: u32) -> (u32, u32) {
    let (ow, oh) = output_size();
    scaled_for(w, h, pct, ow, oh)
}

/// The most lines the port renders at the output's shape: the title's own 2160, and at shapes
/// wider than 16:9 1512 (issue #20: the title's render-target pool, RTSharedAllocator, is one
/// 687 MiB block made after the targets at the render and output sizes, from what they leave; at
/// 21:9 and 32:9 a render of 1868 or 2160 lines left less than that and the title stopped on its
/// out-of-memory assertion, eboot+0x7f9cab, while 1440 to 1524 lines played in: Visual Mode
/// Cinematic's 2160 lines at those shapes now render as Performance's largest, 1512). 0.1.8's
/// package of 2026-11-26 03:42 held 2048 lines (issue #6: above 2048 lines two of the title's
/// targets share a compression block and the port's fast clear went to one of them by the binds'
/// timing, washed out and ghosting); since the fix (`Images::meta_reach`: a fill reaches every
/// image whose metadata starts at its address) the title's 2160 is right again.
pub const MAX_LINES: u64 = 2160;
fn max_lines(ow: u32, oh: u32) -> u64 {
    if ow as u64 * 9 > oh as u64 * 16 { 1512 } else { MAX_LINES }
}

fn scaled_for(w: u32, h: u32, pct: u32, ow: u32, oh: u32) -> (u32, u32) {
    // The output's shape fitted to the area, scaled; at most `max_lines` and the 3840x2160
    // frame's pixels (the most the title's targets were made for).
    let fit_h = (h as u64).min(w as u64 * oh as u64 / ow as u64);
    let mut fh = (fit_h * pct as u64 / 100).clamp(360, max_lines(ow, oh));
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
/// The render size in effect at the last tick (`tick`): the port's (`RENDER`) or, when it holds
/// none, the title's own for its Visual Mode; 0 before the title chose a mode.
static EFFECTIVE: AtomicU64 = AtomicU64::new(0);

fn effective_size() -> u64 {
    let r = RENDER.load(Ordering::Relaxed);
    if r != 0 {
        return r;
    }
    let title = TITLE_MODE.load(Ordering::Relaxed);
    if !(3..=5).contains(&title) {
        return 0;
    }
    let (w, h) = sizes(title).0;
    (w as u64) << 32 | h as u64
}

/// The upscaler's output size asked for and the ticks it has held (`tick`).
static OUT_PENDING: AtomicU64 = AtomicU64::new(0);
static OUT_TICKS: std::sync::atomic::AtomicU32 = std::sync::atomic::AtomicU32::new(0);

pub fn tick() {
    apply_aspect();
    // (The textures' bias for the upscaler's ratio, from the sizes in effect.)
    crate::gpu::upscale::note_sizes(render_size(), upscale_output().unwrap_or_else(output_size));
    if RENDER_FROM_ENV.load(Ordering::Relaxed) || PULSE_LEFT.load(Ordering::Relaxed) > 0 {
        return;
    }
    // (A size change the port did not make: the game's own Visual Mode changed in its menu while
    // the port holds no size (Render Scale Automatic). That size's targets are left as a pulse's
    // are, `images::note_left_size`; until 2026-11-26 only the port's own changes were noted.)
    let eff = effective_size();
    let last = EFFECTIVE.swap(eff, Ordering::Relaxed);
    if last != 0 && eff != 0 && last != eff && RENDER.load(Ordering::Relaxed) == 0 {
        crate::gpu::images::note_left_size((last >> 32) as u32, last as u32);
    }
    let pct = SCALE_PCT.load(Ordering::Relaxed);
    let area = crate::gpu::vk::present::area().or_else(test_screen);
    let headless_want = HEADLESS_WANT.load(Ordering::Relaxed);
    // An upscaler in the TAA's place: its Upscaling mode's size below the screen's, its output
    // at the screen's (`rsize` decides in a run without a window, for comparisons).
    let upscaling = crate::gpu::upscale::active() && !(crate::gpu::vk::present::area().is_none() && headless_want != 0);
    let want = if upscaling {
        let ((rw, rh), (uw, uh)) = upscaling_sizes(area);
        let (ow, oh) = output_size();
        let (uw, uh) = (uw.min(ow), uh.min(oh));
        let out = (uw as u64) << 32 | uh as u64;
        // (A new output size once it held 20 ticks, as the render size: a window's edge dragged
        // gave every frame another one, each a new upscaler context and new images; the review
        // of 2026-11-27.)
        let cur = UPSCALE_OUT.load(Ordering::Relaxed);
        if out != cur {
            if OUT_PENDING.swap(out, Ordering::Relaxed) != out {
                OUT_TICKS.store(0, Ordering::Relaxed);
            }
            if cur == 0 || OUT_TICKS.fetch_add(1, Ordering::Relaxed) >= 20 {
                UPSCALE_OUT.store(out, Ordering::Relaxed);
                crate::info!("upscaler: output {uw}x{uh} for the screen, render {rw}x{rh} (Upscaling at {:.2}x)", UPSCALE_FACTOR.load(Ordering::Relaxed) as f32 / 100.0);
            }
        }
        // (The title's own size needs no pulse while the port holds none.)
        let title = TITLE_MODE.load(Ordering::Relaxed);
        if RENDER.load(Ordering::Relaxed) == 0 && (3..=5).contains(&title) && (rw, rh) == sizes(title).0 && (ow, oh) == (3840, 2160) {
            0
        } else {
            (rw as u64) << 32 | rh as u64
        }
    } else {
        UPSCALE_OUT.store(0, Ordering::Relaxed);
        match area {
            Some((w, h)) if pct != 0 && crate::gpu::vk::present::area().is_some() => {
                let (rw, rh) = scaled_size(w, h, pct);
                (rw as u64) << 32 | rh as u64
            }
            // No window: the title's own size, or its shape's for another aspect
            // (`rsize`: the size asked for).
            _ => match headless_want {
                0 => headless_render(),
                v => v,
            },
        }
    };
    // (Back to the title's own size by a pulse: in Cinematic, 2160 lines at 16:9 (`MAX_LINES`);
    // at wider shapes the capped size.)
    let want = if want == 0 && RENDER.load(Ordering::Relaxed) != 0 && TITLE_MODE.load(Ordering::Relaxed) == 4 {
        let (ow, oh) = output_size();
        let (w, h) = shaped_lines(2160, ow, oh);
        (w as u64) << 32 | h as u64
    } else {
        want
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
    let old = RENDER.swap(want, Ordering::Relaxed);
    // (The size left: its render targets are freed once the change settled, `images::note_left_size`.)
    let size_of = |r: u64| if r == 0 { sizes(title).0 } else { ((r >> 32) as u32, r as u32) };
    if size_of(old) != size_of(want) {
        let (w, h) = size_of(old);
        crate::gpu::images::note_left_size(w, h);
    }
    // (The size in effect is the new one: the check above must not note the old one again.)
    let (nw, nh) = size_of(want);
    EFFECTIVE.store((nw as u64) << 32 | nh as u64, Ordering::Relaxed);
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
    // The mode's render size, stored already at [rsi+4] (width) and [rsi+8] (height) from the
    // mode's table: the port's render size when it holds one, as the render resolution manager
    // gets it (`hook_render_size`). Code that asks this getter (the menus' and character
    // creation's 3D previews) had the mode's own size beside targets made at the port's: the
    // preview drawn into a corner of its space and dark (the known issue of Render Scale
    // percentages; issue #1's dark shop preview at a 4K render). Only for the 4K-output modes.
    let r = RENDER.load(Ordering::Relaxed);
    if r != 0 && ((w, h) == (3840, 2160) || (o != 0 && (w as u64, h as u64) == (o >> 32, o & 0xFFFF_FFFF))) {
        let rsi = unsafe { *regs.add(6) };
        let at = rsi.wrapping_add(4) as *mut u32;
        let (rw, rh) = ((r >> 32) as u32, r as u32);
        let old = unsafe { (at.read_volatile(), at.add(1).read_volatile()) };
        if old != (rw, rh) {
            unsafe {
                at.write_volatile(rw);
                at.add(1).write_volatile(rh);
            }
            if GETTER_SET.fetch_add(1, Ordering::Relaxed) < 8 {
                // The caller: the getter has no frame, so the guest's [rsp] is its return
                // address; the trampoline's saved rsp is 0x80 (red zone) and 11 pushes below it.
                let ret = unsafe { ((*regs.add(4)).wrapping_add(0xd8) as *const u64).read_unaligned() };
                crate::info!(
                    "render size {rw}x{rh} given by the mode's size getter to {} (the mode's was {}x{})",
                    crate::loader::describe_addr(ret.wrapping_sub(5)),
                    old.0,
                    old.1
                );
            }
        }
    }
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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn wide_shapes_lines() {
        // Visual Mode at the output's shape: 16:9 and 16:10 keep their lines, wider shapes take
        // at most 1512 (issue #20: 21:9 and 32:9 at 1440-1524 lines played in, 1868 and 2160
        // stopped the title on its render-target pool).
        // (16:9 and 16:10 at most 2048 lines from 2026-11-25 night until issue #6's fix.)
        assert_eq!(shaped_lines(2160, 3840, 2160), (3840, 2160));
        assert_eq!(shaped_lines(2048, 3840, 2160), (3640, 2048));
        assert_eq!(shaped_lines(1440, 3840, 2160), (2560, 1440));
        assert_eq!(shaped_lines(2160, 3456, 2160), (3456, 2160));
        assert_eq!(shaped_lines(1440, 7680, 2160), (5120, 1440));
        assert_eq!(shaped_lines(1512, 5120, 2160), (3584, 1512));
        assert_eq!(shaped_lines(2160, 7680, 2160), (5376, 1512));
        assert_eq!(shaped_lines(2160, 5120, 2160), (3584, 1512));
        // Render Scale: 150% of a 3440x1440 window at 21:9, and of a 5120x1440 one at 32:9.
        assert_eq!(scaled_for(3440, 1440, 150, 5120, 2160), (3584, 1512));
        assert_eq!(scaled_for(5120, 1440, 150, 7680, 2160), (5376, 1512));
        // 100% of those windows as before; 16:9 at 150% of 2560x1440 and 100% of a 4K window
        // the title's 2160 lines.
        assert_eq!(scaled_for(3440, 1440, 100, 5120, 2160), (3412, 1440));
        assert_eq!(scaled_for(2560, 1440, 150, 3840, 2160), (3840, 2160));
        assert_eq!(scaled_for(3840, 2160, 100, 3840, 2160), (3840, 2160));
        assert_eq!(scaled_for(3840, 2160, 90, 3840, 2160), (3456, 1944));
    }
}
