//! NVIDIA DLSS Frame Generation through Streamline's sl.dlss_g (ProgrammingGuideDLSS_G.md, SL
//! 2.14.16), in manual hooking mode: the presenter makes its swapchain through Streamline's
//! entry points with the plugin loaded (`gpu::vk::present`), so that DLSS-G puts its own
//! swapchain and pacing behind it; each frame's common constants, depth and motion vectors are
//! tagged when the game's GPU work is recorded (`record_render_inputs`), and the HUD-less picture,
//! the interface's coverage and the picture's place in the window when the frame is presented
//! (`tag_present`), all under the frame's index (`gpu::streamline`).
//!
//! The options (`present_frame`, on the presenting thread as the guide asks): on with the chosen
//! multiplier (clamped to what the GPU can: numFramesToGenerateMax) while frames come with their
//! inputs, off ("deferred off") on frames without them; resources kept while off
//! (eRetainResourcesWhenOff) and freed when the setting turns it off. Its state is read every few
//! seconds: a status other than OK turns it off for the session, logged.
//!
//! The model: DLSS-G runs the newest full model of the player's nvngx_dlssg.dll. Streamline and
//! NGX give an application no choice of frame generation model (the NGX library's "render
//! presets" and its lighter "FG Lite" variant are only reached through the NVIDIA driver's own
//! overrides, the NVIDIA app's), so the port asks for none: what runs is the library's default,
//! its latest full model, which the log names by the library's version.

use super::super::streamline::{self as sl, SlDlssgOptions, SlDlssgState, SlExtent, SlResource, SlResourceTag, SlViewport};
use super::super::upscale::Tex;
use parking_lot::Mutex;
use std::sync::OnceLock;
use std::sync::atomic::{AtomicBool, AtomicU32, AtomicU64, Ordering::Relaxed, Ordering::SeqCst};

type FnSetOptions = unsafe extern "C" fn(*const SlViewport, *const SlDlssgOptions) -> i32;
type FnGetState = unsafe extern "C" fn(*const SlViewport, *mut SlDlssgState, *const SlDlssgOptions) -> i32;

struct Fns {
    set_options: FnSetOptions,
    get_state: FnGetState,
}

static FNS: OnceLock<Fns> = OnceLock::new();
/// The most frames the GPU generates per rendered frame (numFramesToGenerateMax; 0: unknown).
static MAX_GENERATED: AtomicU32 = AtomicU32::new(0);
/// The presenter's swapchain was made with DLSS-G loaded.
static LOADED: AtomicBool = AtomicBool::new(false);
/// Why DLSS-G stopped for this session (its state reported a failure).
static FAILED: Mutex<Option<String>> = Mutex::new(None);
/// The last frame whose constants, depth and motion vectors were tagged (u32::MAX: none).
static INPUTS_FRAME: AtomicU32 = AtomicU32::new(u32::MAX);
/// The motion vectors' and depth's size (width << 32 | height) as last tagged.
static INPUT_SIZE: AtomicU64 = AtomicU64::new(0);
/// Frames generated (numFramesActuallyPresented summed) and game frames presented with it on.
static SHOWN: AtomicU64 = AtomicU64::new(0);
static FRAMES_ON: AtomicU64 = AtomicU64::new(0);

fn fns() -> Option<&'static Fns> {
    FNS.get()
}

/// DLSS-G can run in this session (supported on the device, Streamline ready, not failed).
pub fn usable() -> bool {
    sl::supported(sl::FEATURE_DLSS_G) && fns().is_some() && FAILED.lock().is_none()
}

/// Offered in the menu: usable, or loaded and supported while Streamline is still starting.
pub fn offered() -> bool {
    usable() || (sl::loaded(sl::FEATURE_DLSS_G) && sl::unsupported_reason(sl::FEATURE_DLSS_G).is_none() && !sl::ready() && FAILED.lock().is_none())
}

/// Why DLSS-G cannot run here (None: it can, or Streamline is still starting).
pub fn unavailable_reason() -> Option<String> {
    if let Some(f) = FAILED.lock().clone() {
        return Some(f);
    }
    if !sl::loaded(sl::FEATURE_DLSS_G) {
        return Some("not installed (sl.dlss_g.dll and your own nvngx_dlssg.dll in the upscalers folder, an NVIDIA GeForce RTX 40 or newer)".into());
    }
    sl::unsupported_reason(sl::FEATURE_DLSS_G)
}

/// The largest multiplier the GPU can (None: not known yet).
pub fn max_multiplier() -> Option<u32> {
    match MAX_GENERATED.load(Relaxed) {
        0 => None,
        n => Some(n + 1),
    }
}

/// The presenter's swapchain has DLSS-G loaded.
pub fn loaded() -> bool {
    LOADED.load(SeqCst)
}

/// The presenter should make its swapchain with DLSS-G loaded.
pub fn wants_swapchain() -> bool {
    super::choice() == super::FG_DLSS && usable()
}

fn get_state(f: &Fns) -> Option<SlDlssgState> {
    let vp = sl::viewport();
    let mut st = SlDlssgState::default();
    let r = unsafe { (f.get_state)(&vp, &mut st, std::ptr::null()) };
    (r == sl::SL_OK).then_some(st)
}

/// Streamline is ready: DLSS-G's functions and what the GPU can.
pub(super) fn streamline_ready() {
    if !sl::supported(sl::FEATURE_DLSS_G) {
        return;
    }
    let set = sl::feature_fn(sl::FEATURE_DLSS_G, c"slDLSSGSetOptions");
    let get = sl::feature_fn(sl::FEATURE_DLSS_G, c"slDLSSGGetState");
    let (Some(set), Some(get)) = (set, get) else {
        *FAILED.lock() = Some("Streamline gave no slDLSSGSetOptions or slDLSSGGetState".into());
        return;
    };
    // SAFETY: sl_dlss_g.h's PFun_slDLSSGSetOptions and PFun_slDLSSGGetState (references are pointers).
    let f = unsafe { Fns { set_options: std::mem::transmute(set), get_state: std::mem::transmute(get) } };
    if let Some(st) = get_state(&f) {
        MAX_GENERATED.store(st.num_frames_to_generate_max.max(1), Relaxed);
        crate::info!(
            "frame generation: DLSS Frame Generation ready: up to {}x on this GPU (DLSS-G status {:#x}, smallest size {}, VSync with it {}, Dynamic {}); the model is nvngx_dlssg.dll's default, its newest full model",
            st.num_frames_to_generate_max.max(1) + 1,
            st.status,
            st.min_width_or_height,
            if st.b_is_vsync_support_available == sl::SL_TRUE { "supported" } else { "not supported (Vulkan)" },
            if st.b_is_dynamic_mfg_supported == sl::SL_TRUE { "supported" } else { "not supported" },
        );
    } else {
        crate::info!("frame generation: DLSS Frame Generation ready (its state not readable yet)");
    }
    let _ = FNS.set(f);
    if super::choice() == super::FG_DLSS {
        crate::info!("frame generation: DLSS Frame Generation chosen: on from the next swapchain");
        super::super::latency::apply();
    }
}

// ---------------------------------------------------------------------------
// The presenter's side (its own thread, the vblank thread).

struct Present {
    /// The options as last set: on, frames generated per frame.
    on: bool,
    generated: u32,
    /// Game frames in a row presented with their inputs.
    good: u32,
    /// Presents since the state was last read.
    since_state: u32,
    extent: (u32, u32),
    format: vk::Format,
    images: u32,
    /// Frames turned off for lack of inputs (logged in the status line).
    deferred: u64,
}

use ash::vk;

static PRESENT: Mutex<Present> = Mutex::new(Present { on: false, generated: 0, good: 0, since_state: 0, extent: (0, 0), format: vk::Format::UNDEFINED, images: 0, deferred: 0 });

fn options(on: bool, generated: u32, p: &Present) -> SlDlssgOptions {
    let (mw, mh) = {
        let s = INPUT_SIZE.load(Relaxed);
        ((s >> 32) as u32, s as u32)
    };
    SlDlssgOptions {
        mode: if on { sl::DLSSG_ON } else { sl::DLSSG_OFF },
        num_frames_to_generate: generated.max(1),
        flags: sl::DLSSG_RETAIN_RESOURCES_WHEN_OFF,
        num_back_buffers: p.images,
        mvec_depth_width: mw,
        mvec_depth_height: mh,
        color_width: p.extent.0,
        color_height: p.extent.1,
        color_buffer_format: p.format.as_raw() as u32,
        ..SlDlssgOptions::default()
    }
}

fn set_options(f: &Fns, o: &SlDlssgOptions) -> bool {
    let vp = sl::viewport();
    let r = unsafe { (f.set_options)(&vp, o) };
    if r != sl::SL_OK {
        static N: AtomicU32 = AtomicU32::new(0);
        if N.fetch_add(1, Relaxed) < 6 {
            crate::warn!("frame generation: slDLSSGSetOptions(mode {}) failed ({r}: {})", o.mode, sl::result_name(r));
        }
        return false;
    }
    true
}

/// Before the presenter makes a swapchain: DLSS-G loaded (`on`) or unhooked (and its resources
/// freed when it was in use).
pub fn swapchain_loading(on: bool) {
    if !sl::loaded(sl::FEATURE_DLSS_G) {
        return;
    }
    let was = LOADED.swap(false, SeqCst);
    if was && !on {
        let r = sl::free_resources(sl::FEATURE_DLSS_G);
        if r != sl::SL_OK {
            crate::debug!("frame generation: slFreeResources ({r}: {})", sl::result_name(r));
        }
    }
    // (Unsupported here: left unloaded since the device was made.)
    if sl::supported(sl::FEATURE_DLSS_G) {
        let r = sl::set_feature_loaded(sl::FEATURE_DLSS_G, on);
        if r != sl::SL_OK {
            crate::warn!("frame generation: slSetFeatureLoaded({on}) failed ({r}: {})", sl::result_name(r));
        }
    }
}

/// The presenter made its swapchain (`loaded`: with DLSS-G): its size, format and images.
pub fn swapchain_created(loaded: bool, extent: (u32, u32), format: vk::Format, images: u32) {
    let mut p = PRESENT.lock();
    *p = Present { on: false, generated: 0, good: 0, since_state: 0, extent, format, images, deferred: p.deferred };
    if !loaded {
        return;
    }
    let Some(f) = fns() else { return };
    // (Set off with the swapchain's description: on once game frames come with their inputs.)
    set_options(f, &options(false, 1, &p));
    LOADED.store(true, SeqCst);
    let min = get_state(f).map_or(0, |s| s.min_width_or_height);
    crate::info!("frame generation: swapchain {}x{} {format:?} with DLSS Frame Generation ({images} images{})", extent.0, extent.1, if min > 0 && extent.0.min(extent.1) < min { format!("; smaller than its {min}: it stays off") } else { String::new() });
}

/// Before the presenter destroys its swapchain or changes the window: DLSS-G off first (the
/// guide's 12.0).
pub fn swapchain_destroying() {
    let Some(f) = fns() else { return };
    let mut p = PRESENT.lock();
    if p.on {
        set_options(f, &options(false, p.generated, &p));
        p.on = false;
        p.good = 0;
    }
}

/// Frame `frame` is about to be presented: DLSS-G on for it when it and the frame before came
/// with their inputs, else off (menus, loading). True when frames are generated for it.
pub fn present_frame(frame: u32) -> bool {
    if !loaded() {
        return false;
    }
    let Some(f) = fns() else { return false };
    let mut p = PRESENT.lock();
    let inputs = INPUTS_FRAME.load(SeqCst) == frame;
    p.good = if inputs { p.good.saturating_add(1) } else { 0 };
    let max = MAX_GENERATED.load(Relaxed).max(1);
    let generated = (super::multiplier() - 1).clamp(1, max);
    let want = p.good >= 2 && FAILED.lock().is_none();
    if !inputs && p.on {
        p.deferred += 1;
    }
    if (want, if want { generated } else { p.generated }) != (p.on, p.generated) {
        if set_options(f, &options(want, generated, &p)) {
            static LOGGED: AtomicU32 = AtomicU32::new(0);
            if LOGGED.fetch_add(1, Relaxed) < 40 {
                if want {
                    crate::info!("frame generation: on at flip {frame}, {}x ({} generated a frame{})", generated + 1, generated, if generated + 1 < super::multiplier() { format!("; {}x chosen, this GPU's most is {}x", super::multiplier(), max + 1) } else { String::new() });
                } else {
                    crate::info!("frame generation: off from flip {frame} (a frame without the game's depth and motion vectors: a menu, a loading screen or a video)");
                }
            }
            p.on = want;
            if want {
                p.generated = generated;
            }
        }
    }
    if p.on {
        FRAMES_ON.fetch_add(1, Relaxed);
        p.since_state += 1;
        // Its state now and then: what it presented, and whether it runs.
        if p.since_state >= 240 {
            p.since_state = 0;
            if let Some(st) = get_state(f) {
                SHOWN.fetch_add(st.num_frames_actually_presented as u64, Relaxed);
                if st.status != 0 {
                    let why = status_text(st.status);
                    crate::warn!("frame generation: DLSS Frame Generation reports {why} (status {:#x}): off for this session", st.status);
                    *FAILED.lock() = Some(why);
                    set_options(f, &options(false, p.generated, &p));
                    p.on = false;
                }
            }
        }
    }
    p.on
}

/// DLSSGStatus's flags in words.
fn status_text(s: u32) -> String {
    let mut v = Vec::new();
    for (bit, what) in [
        (1, "the window is too small for it"),
        (2, "Reflex not running"),
        (4, "an unsupported HDR format"),
        (8, "invalid common constants"),
        (16, "the back buffer index not queried"),
    ] {
        if s & bit != 0 {
            v.push(what);
        }
    }
    if v.is_empty() { format!("an unknown failure {s:#x}") } else { v.join(", ") }
}

/// The present's inputs for frame `frame`, recorded into the presenter's command buffer `cmd`:
/// the picture's place in the swapchain image (`rect`, of `full`; the back buffer tagged with it
/// only when it does not fill the window), the HUD-less picture and the interface's coverage at
/// the swapchain's size (None: not this frame; their tags cleared).
pub fn tag_present(frame: u32, cmd: vk::CommandBuffer, rect: (u32, u32, u32, u32), full: (u32, u32), hudless: Option<SlResource>, ui_alpha: Option<SlResource>) {
    let Some(tok) = sl::token(frame) else { return };
    let (x, y, w, h) = rect;
    let extent = SlExtent { top: y, left: x, width: w, height: h };
    let mut tags: Vec<SlResourceTag> = Vec::with_capacity(3);
    let res = [hudless, ui_alpha];
    for (i, ty) in [sl::SL_HUDLESS, sl::SL_UI_ALPHA].into_iter().enumerate() {
        match &res[i] {
            Some(r) => tags.push(SlResourceTag::new(r, ty, sl::SL_VALID_UNTIL_PRESENT, extent)),
            None => tags.push(SlResourceTag::new(std::ptr::null(), ty, sl::SL_VALID_UNTIL_PRESENT, SlExtent::default())),
        }
    }
    // (The back buffer itself: Streamline knows it; only its subrect is given.)
    let sub = (x, y, w, h) != (0, 0, full.0, full.1);
    tags.push(SlResourceTag::new(std::ptr::null(), sl::SL_BACKBUFFER, sl::SL_VALID_UNTIL_PRESENT, if sub { extent } else { SlExtent::default() }));
    let r = sl::set_tags(tok, &tags, cmd);
    if r != sl::SL_OK {
        static N: AtomicU32 = AtomicU32::new(0);
        if N.fetch_add(1, Relaxed) < 4 {
            crate::warn!("frame generation: the present's tags for flip {frame} failed ({r}: {})", sl::result_name(r));
        }
    }
}

// ---------------------------------------------------------------------------
// The game's side (where the title's TAA is recorded, played into the batch's command buffer).

/// The frame tagged before the last (the reset of a frame that does not follow it).
static LAST_FRAME: AtomicU32 = AtomicU32::new(u32::MAX);

/// Frame `frame`'s common constants (when DLSS has not set them already) and its depth and
/// motion vectors (`extent`: the render size), recorded into `cb` right after the title's TAA
/// read them: Streamline copies them there for the present (eOnlyValidNow), as the title reuses
/// both images later in the frame.
pub fn record_render_inputs(cb: vk::CommandBuffer, frame: u32, depth: &Tex, mvec: &Tex, extent: (u32, u32), consts: &sl::SlConstants) {
    if !loaded() {
        return;
    }
    let Some(tok) = sl::token(frame) else { return };
    let last = LAST_FRAME.swap(frame, Relaxed);
    let mut c = *consts;
    // (A frame that does not follow the last tagged one starts DLSS-G's history anew.)
    if last == u32::MAX || frame != last.wrapping_add(1) {
        c.reset = sl::SL_TRUE;
    }
    if !sl::set_constants(frame, tok, &c) {
        return;
    }
    let res = [super::super::upscale::sl_resource(depth), super::super::upscale::sl_resource(mvec)];
    let e = SlExtent { top: 0, left: 0, width: extent.0, height: extent.1 };
    let tags = [SlResourceTag::new(&res[0], sl::SL_DEPTH, sl::SL_ONLY_VALID_NOW, e), SlResourceTag::new(&res[1], sl::SL_MOTION_VECTORS, sl::SL_ONLY_VALID_NOW, e)];
    let r = sl::set_tags(tok, &tags, cb);
    if r != sl::SL_OK {
        static N: AtomicU32 = AtomicU32::new(0);
        if N.fetch_add(1, Relaxed) < 4 {
            crate::warn!("frame generation: depth and motion vectors of flip {frame} not tagged ({r}: {})", sl::result_name(r));
        }
        return;
    }
    INPUT_SIZE.store((extent.0 as u64) << 32 | extent.1 as u64, Relaxed);
    INPUTS_FRAME.store(frame, SeqCst);
}

/// Status line.
pub fn status_line() -> String {
    if !sl::loaded(sl::FEATURE_DLSS_G) {
        return String::new();
    }
    let p = PRESENT.lock();
    format!(
        "    frame generation (DLSS-G): {}; chosen {} {}x (most {}x); {} game frames presented with it, {} frames presented by it at the last reads; {} off for frames without inputs{}\n",
        if p.on { "on" } else if loaded() { "loaded, off now" } else { "not loaded" },
        super::backend_name(super::choice()),
        super::multiplier(),
        MAX_GENERATED.load(Relaxed) + 1,
        FRAMES_ON.load(Relaxed),
        SHOWN.load(Relaxed),
        p.deferred,
        FAILED.lock().as_ref().map_or(String::new(), |f| format!("; stopped: {f}"))
    )
}
