//! The upscalers' output through the title's composite at the screen's size: real upscaling,
//! stage B (2026-11-26 evening; FINDINGS.md 2026-11-26 ~18:50, the pass map). The title's TAA,
//! depth of field, bloom and composite run at the render size, and its output pass scales the
//! composite to the 3840x2160 output with a bicubic of its own. While an upscaler runs above the
//! render size (`dispmode::upscale_output`), its output at the screen's size (the TAA's alpha laid
//! in, `upmerge::record_up`) is composited at that size too: right after the title's composite
//! draw the same draw is recorded again into an image of ours (`Exec::composite_up`), with the
//! upscaler's output in the TAA output's place and the depth of field's tiles (read at the pixel's
//! position / 16) enlarged to match. The output pass then reads that image in the composite's
//! place, its size constants given for it (`Exec::output_up`). The title's own composite at the
//! render size is left as it drew it, for whatever else reads it. While this runs, a parallel
//! worker stops at the output pass, so the command processor records it after the chunk that
//! holds the composite (`par`): the handover (`note_done`, `take_done`) follows program order.

use super::vk::ctx;
use ash::vk;
use parking_lot::Mutex;
use std::sync::atomic::{AtomicU64, Ordering::Relaxed};

/// The title's composite at the render size (01.007.000): the TAA output, the depth of field's
/// layers and bloom into the scene image its output pass (`uiscale::COMPOSITE_PS`) scales.
pub const SCENE_COMPOSITE_PS: u64 = 0x1008890d00;

/// The TAA output's T# base as the TAA's replacement last saw it (0: none yet).
static TAA_BASE: AtomicU64 = AtomicU64::new(0);

pub fn note_taa_base(base: u64) {
    TAA_BASE.store(base, Relaxed);
}

pub fn taa_base() -> u64 {
    TAA_BASE.load(Relaxed)
}

/// The upscaler's output size when it is larger than the render size `r` (the screen's), while
/// an upscaler runs and stage B has not failed: the TAA's replacement writes that size.
pub fn size_for(r: (u32, u32)) -> Option<(u32, u32)> {
    if !super::upscale::active() || FAILED.load(Relaxed) != 0 {
        return None;
    }
    let o = crate::dispmode::upscale_output()?;
    (o.0 >= r.0 && o.1 >= r.1 && o != r).then_some(o)
}

/// Whether this frame's composite is drawn again at the upscaler's size: `size_for`, and the
/// upscaler's last evaluation went through (else its image holds an older frame).
pub fn wanted(r: (u32, u32)) -> Option<(u32, u32)> {
    if !super::upscale::last_eval_ok() {
        return None;
    }
    size_for(r)
}

/// The composite drawn again: what the output pass after it reads.
#[derive(Clone, Copy, Debug)]
pub struct Done {
    /// The title's composite target (its T# base where the output pass reads it).
    pub base: u64,
    pub view: vk::ImageView,
    pub r: (u32, u32),
    pub o: (u32, u32),
}

static DONE: Mutex<Option<Done>> = Mutex::new(None);

pub fn note_done(d: Done) {
    *DONE.lock() = Some(d);
}

/// The composite drawn again before this output pass (taken once).
pub fn take_done() -> Option<Done> {
    DONE.lock().take()
}

/// Stage B stopped for the session (a reason's number, 0: running) after a frame it could not
/// draw: the output pass then goes on reading the title's own composite.
static FAILED: AtomicU64 = AtomicU64::new(0);

pub fn fail(why: &str) {
    if FAILED.swap(1, Relaxed) == 0 {
        crate::warn!("upscaler: the composite at the upscaler's size is off for this session ({why}): the title's own bicubic scales its composite");
    }
}

struct Img {
    image: vk::Image,
    memory: vk::DeviceMemory,
    view: vk::ImageView,
    format: vk::Format,
    w: u32,
    h: u32,
}

unsafe impl Send for Img {}

/// An image slot: the current one and those of earlier sizes, freed once no frame in flight can
/// use them (sizes change rarely: a window resized, the Upscaling mode changed).
struct Slot {
    cur: Option<Img>,
    retired: Vec<(Img, u64)>,
}

const RETIRE_FLIPS: u64 = 240;

static COMPOSITE: Mutex<Slot> = Mutex::new(Slot { cur: None, retired: Vec::new() });
static TILES: Mutex<Slot> = Mutex::new(Slot { cur: None, retired: Vec::new() });
/// Images made (the status line).
pub static MADE: AtomicU64 = AtomicU64::new(0);

unsafe fn destroy(i: Img) {
    let c = ctx();
    unsafe {
        c.device.destroy_image_view(i.view, None);
        c.device.destroy_image(i.image, None);
        c.device.free_memory(i.memory, None);
    }
}

unsafe fn make(format: vk::Format, w: u32, h: u32, usage: vk::ImageUsageFlags) -> Result<Img, String> {
    let c = ctx();
    unsafe {
        let ici = vk::ImageCreateInfo::default()
            .image_type(vk::ImageType::TYPE_2D)
            .format(format)
            .extent(vk::Extent3D { width: w, height: h, depth: 1 })
            .mip_levels(1)
            .array_layers(1)
            .samples(vk::SampleCountFlags::TYPE_1)
            .tiling(vk::ImageTiling::OPTIMAL)
            .usage(usage);
        let image = c.device.create_image(&ici, None).map_err(|e| e.to_string())?;
        let memory = match c.alloc(c.device.get_image_memory_requirements(image), vk::MemoryPropertyFlags::DEVICE_LOCAL) {
            Ok(m) => m,
            Err(e) => {
                c.device.destroy_image(image, None);
                return Err(e);
            }
        };
        if let Err(e) = c.device.bind_image_memory(image, memory, 0) {
            c.device.destroy_image(image, None);
            c.device.free_memory(memory, None);
            return Err(e.to_string());
        }
        let range = vk::ImageSubresourceRange { aspect_mask: vk::ImageAspectFlags::COLOR, base_mip_level: 0, level_count: 1, base_array_layer: 0, layer_count: 1 };
        let view = match c.device.create_image_view(&vk::ImageViewCreateInfo::default().image(image).view_type(vk::ImageViewType::TYPE_2D).format(format).subresource_range(range), None) {
            Ok(v) => v,
            Err(e) => {
                c.device.destroy_image(image, None);
                c.device.free_memory(memory, None);
                return Err(e.to_string());
            }
        };
        Ok(Img { image, memory, view, format, w, h })
    }
}

fn get(slot: &Mutex<Slot>, what: &str, format: vk::Format, w: u32, h: u32, usage: vk::ImageUsageFlags) -> Result<(vk::Image, vk::ImageView), String> {
    let frame = super::FRAMES.load(Relaxed);
    let mut g = slot.lock();
    let mut keep = Vec::with_capacity(g.retired.len());
    for (img, at) in g.retired.drain(..) {
        if frame > at + RETIRE_FLIPS {
            unsafe { destroy(img) };
        } else {
            keep.push((img, at));
        }
    }
    g.retired = keep;
    if let Some(i) = g.cur.as_ref().filter(|i| (i.format, i.w, i.h) == (format, w, h)) {
        return Ok((i.image, i.view));
    }
    let img = unsafe { make(format, w, h, usage) }?;
    MADE.fetch_add(1, Relaxed);
    crate::info!("upscaler: {what} image {w}x{h} {format:?} made (the composite at the upscaler's size)");
    let r = (img.image, img.view);
    if let Some(old) = g.cur.replace(img) {
        g.retired.push((old, frame));
    }
    Ok(r)
}

/// The composite drawn at the upscaler's size (`format`, a colour target the output pass samples).
pub fn composite_image(format: vk::Format, w: u32, h: u32) -> Result<(vk::Image, vk::ImageView), String> {
    let usage = vk::ImageUsageFlags::COLOR_ATTACHMENT | vk::ImageUsageFlags::SAMPLED | vk::ImageUsageFlags::TRANSFER_SRC | vk::ImageUsageFlags::TRANSFER_DST;
    get(&COMPOSITE, "composite", format, w, h, usage)
}

/// The depth of field's tiles at the upscaler's size / 16 (blitted from the title's, nearest), as
/// a sampled image and, where the format allows, a storage one.
pub fn tile_image(format: vk::Format, w: u32, h: u32) -> Result<(vk::Image, vk::ImageView), String> {
    let c = ctx();
    let f = unsafe { c.instance.get_physical_device_format_properties(c.pdev, format) }.optimal_tiling_features;
    if !f.contains(vk::FormatFeatureFlags::BLIT_SRC | vk::FormatFeatureFlags::BLIT_DST) {
        return Err(format!("{format:?} cannot be blitted here"));
    }
    let mut usage = vk::ImageUsageFlags::TRANSFER_DST | vk::ImageUsageFlags::SAMPLED;
    if f.contains(vk::FormatFeatureFlags::STORAGE_IMAGE) {
        usage |= vk::ImageUsageFlags::STORAGE;
    }
    get(&TILES, "tile", format, w, h, usage)
}

/// The status line (none before stage B ran).
pub fn status_line() -> String {
    let made = MADE.load(Relaxed);
    let drawn = DRAWN.load(Relaxed);
    if made == 0 && drawn == 0 {
        return String::new();
    }
    format!("    upscaler composites at the upscaler's size: {drawn} drawn, {} read by the output pass, {made} images made{}\n", READ.load(Relaxed), if FAILED.load(Relaxed) != 0 { " (off: see the warning)" } else { "" })
}

/// Composites drawn again, and read by the output pass.
pub static DRAWN: AtomicU64 = AtomicU64::new(0);
pub static READ: AtomicU64 = AtomicU64::new(0);
