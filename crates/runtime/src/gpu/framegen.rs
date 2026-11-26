//! Frame generation: frames generated between the game's own, shown in between by the presenter.
//! The backend today is NVIDIA DLSS Frame Generation through Streamline (`dlssg`,
//! ProgrammingGuideDLSS_G.md); what the backends share lives here, so that another (AMD's FSR
//! frame generation, say) plugs in beside it: the setting and its multiplier, the game's inputs
//! each frame (`record_render_inputs`: the depth and motion vectors the upscalers read at the
//! title's TAA, the same images DLSS Super Resolution gets), the HUD-less picture and the
//! interface's coverage (`capture`), and the presenter's side (`present_frame`).
//!
//! The inputs, all the game's own (nothing estimated): the title's TAA reads the scene's depth
//! (reversed) and motion vectors (R16G16, screen fractions, camera motion included) at the render
//! size; they are tagged for the frame as they are when the TAA runs (Streamline copies them for
//! the present). The final picture is the presenter's swapchain image. The HUD-less picture is the
//! title's own final composite drawn once more with its interface layer replaced by a transparent
//! image (`capture`), and the interface's coverage is that layer's alpha; both go through the same
//! scaling, sharpening and placement in the window as the final picture, so that they line up
//! with it pixel for pixel in the swapchain's own format and colour space (HDR10's PQ-encoded
//! BT.2020 in 10:10:10:2 when the game shows HDR, the SDR picture's 8 bits otherwise: what DLSS-G
//! takes, without a lossy step between).
//!
//! A frame is generated only after frames of the game's that have all their inputs: menus and
//! loading screens without the title's TAA turn it off for their frames ("deferred off"), and it
//! comes back on with the next game frame. Changing the setting makes a new swapchain (DLSS-G's
//! guide 18.0: no overhead while it is off); resizing or switching the window mode turns it off
//! first (12.0).
//!
//! `DS_FRAMEGEN=0`: no frame generation (Streamline does not load its plugin). `DS_FG_HUDLESS=0`:
//! no HUD-less picture or interface coverage (the HUD is interpolated with the scene).

pub mod capture;
pub mod dlssg;

use std::sync::atomic::{AtomicU32, Ordering::Relaxed};

/// The Frame Generation setting's codes (pcsettings.txt keeps the code).
pub const FG_OFF: u32 = 0;
pub const FG_DLSS: u32 = 1;

/// The setting and its multiplier (2 to 6: frames shown per frame rendered).
static CHOICE: AtomicU32 = AtomicU32::new(FG_OFF);
static MULTIPLIER: AtomicU32 = AtomicU32::new(2);

pub fn backend_name(code: u32) -> &'static str {
    match code {
        FG_DLSS => "DLSS Frame Generation",
        _ => "Off",
    }
}

/// The setting changed (`pcsettings::apply`; `DS_FG=0|1` and `DS_FG_MULT=2..6` decide in
/// development runs). Takes effect at the next present (a new swapchain).
pub fn set_choice(code: u32, multiplier: u32) {
    let code = std::env::var("DS_FG").ok().and_then(|v| v.trim().parse::<u32>().ok()).unwrap_or(code).min(FG_DLSS);
    let multiplier = std::env::var("DS_FG_MULT").ok().and_then(|v| v.trim().parse::<u32>().ok()).unwrap_or(multiplier).clamp(2, 6);
    let was = (CHOICE.swap(code, Relaxed), MULTIPLIER.swap(multiplier, Relaxed));
    if was != (code, multiplier) {
        let note = match code {
            FG_DLSS if dlssg::usable() => String::new(),
            FG_DLSS => match dlssg::unavailable_reason() {
                Some(why) => format!(" (not available: {why})"),
                None => " (once Streamline is ready)".to_string(),
            },
            _ => String::new(),
        };
        crate::info!("frame generation: {} chosen, {multiplier}x{note}", backend_name(code));
        // (Reflex is at least On while it is chosen.)
        super::latency::apply();
    }
}

pub fn choice() -> u32 {
    CHOICE.load(Relaxed)
}

/// The multiplier chosen (2 to 6).
pub fn multiplier() -> u32 {
    MULTIPLIER.load(Relaxed)
}

/// The Frame Generation list of this session: Off, then each backend that can run here or may
/// still become ready (DLSS-G: once Streamline started) (codes and labels).
pub fn offered() -> Vec<(u32, &'static str)> {
    let mut v = vec![(FG_OFF, "Off")];
    if dlssg::offered() {
        v.push((FG_DLSS, "DLSS Frame Generation"));
    }
    v
}

/// The multipliers of this session (2x up to what the GPU can: DLSS-G's
/// numFramesToGenerateMax + 1; all of 2x to 6x while that is not known yet).
pub fn multipliers() -> Vec<u32> {
    let max = dlssg::max_multiplier().unwrap_or(6).clamp(2, 6);
    (2..=max).collect()
}

/// Frame generation is chosen and its backend can run (Reflex is then at least On).
pub fn requires_reflex() -> bool {
    choice() == FG_DLSS && dlssg::usable()
}

/// Frame generation runs on the presenter's swapchain now (the game's inputs are tagged each
/// frame: `record_render_inputs`, `capture`).
pub fn render_inputs_wanted() -> bool {
    dlssg::loaded()
}

/// Streamline is ready (`gpu::streamline::device_created`).
pub(super) fn streamline_ready() {
    dlssg::streamline_ready();
}

/// Status line.
pub fn status_line() -> String {
    dlssg::status_line() + &capture::status_line()
}
