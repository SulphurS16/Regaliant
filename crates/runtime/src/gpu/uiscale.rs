//! Menu and HUD size (the PC tab's): the title draws its interface into a layer of its own
//! (3840x2160 RGBA16F, mostly transparent) that its last draw of a frame composites over the
//! scaled-up scene (ps 0x100888d100, the layer its resource 0). Before that draw the layer is
//! blitted smaller around its centre into an image of ours, and the composite samples that image
//! in the layer's place (`Exec::scale_ui_layer`), so the whole interface (menus and HUD alike)
//! shrinks toward the middle. The layer itself is left as the title drew it: recording this twice
//! (a parallel chunk that stops at the composite, a held batch dropped and recorded again) gives
//! the same picture. The pointer maps through the same scale (`pcsettings::pointer`).

use super::vk::ctx;
use ash::vk;
use parking_lot::Mutex;

/// The title's interface composite (01.007.000), SDR.
pub const COMPOSITE_PS: u64 = 0x100888d100;
/// The same composite when the title outputs HDR10 (its HDR variant beside it, the interface
/// layer its resource 0 as well; seen drawing the display buffer in HDR sessions on 2026-11-26,
/// where the SDR one never ran: frame generation's HUD-less capture met no composite).
pub const COMPOSITE_PS_HDR: u64 = 0x100888d900;

/// Whether pixel stage `ps` is the interface composite (SDR or HDR).
pub fn is_composite(ps: u64) -> bool {
    ps == COMPOSITE_PS || ps == COMPOSITE_PS_HDR
}

struct Spare {
    image: vk::Image,
    memory: vk::DeviceMemory,
    view: vk::ImageView,
    format: vk::Format,
    w: u32,
    h: u32,
    filter: vk::Filter,
}

unsafe impl Send for Spare {}

static SPARE: Mutex<Option<Spare>> = Mutex::new(None);
/// Images of earlier sizes (frames in flight may still use them; sizes change rarely).
static RETIRED: Mutex<Vec<Spare>> = Mutex::new(Vec::new());

fn spare(format: vk::Format, w: u32, h: u32) -> Result<(vk::Image, vk::ImageView, vk::Filter), String> {
    let mut g = SPARE.lock();
    if let Some(s) = g.as_ref() {
        if (s.format, s.w, s.h) == (format, w, h) {
            return Ok((s.image, s.view, s.filter));
        }
    }
    let c = ctx();
    unsafe {
        let f = c.instance.get_physical_device_format_properties(c.pdev, format).optimal_tiling_features;
        if !f.contains(vk::FormatFeatureFlags::BLIT_SRC | vk::FormatFeatureFlags::BLIT_DST | vk::FormatFeatureFlags::SAMPLED_IMAGE) {
            return Err(format!("{format:?} cannot be blitted here"));
        }
        let filter = if f.contains(vk::FormatFeatureFlags::SAMPLED_IMAGE_FILTER_LINEAR) { vk::Filter::LINEAR } else { vk::Filter::NEAREST };
        let ici = vk::ImageCreateInfo::default()
            .image_type(vk::ImageType::TYPE_2D)
            .format(format)
            .extent(vk::Extent3D { width: w, height: h, depth: 1 })
            .mip_levels(1)
            .array_layers(1)
            .samples(vk::SampleCountFlags::TYPE_1)
            .tiling(vk::ImageTiling::OPTIMAL)
            .usage(vk::ImageUsageFlags::TRANSFER_DST | vk::ImageUsageFlags::TRANSFER_SRC | vk::ImageUsageFlags::SAMPLED);
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
        if let Some(old) = g.replace(Spare { image, memory, view, format, w, h, filter }) {
            RETIRED.lock().push(old);
        }
        crate::info!("ui scale: image {w}x{h} {format:?} made ({filter:?} filter)");
        Ok((image, view, filter))
    }
}

/// A scaled copy to record (`prepare`, then `record`).
#[derive(Clone, Copy)]
pub struct Job {
    layer: vk::Image,
    tmp: vk::Image,
    /// Our image's view, which the composite binds in the layer's place.
    pub view: vk::ImageView,
    w: u32,
    h: u32,
    s: f32,
    filter: vk::Filter,
}

/// The copy of `layer` (GENERAL, `w`x`h`) scaled by `s` into our image (made when needed);
/// nothing recorded yet. None when no image could be made (the interface keeps its size).
pub fn prepare(layer: vk::Image, format: vk::Format, w: u32, h: u32, s: f32) -> Option<Job> {
    match spare(format, w, h) {
        Ok((tmp, view, filter)) => Some(Job { layer, tmp, view, w, h, s, filter }),
        Err(e) => {
            static ONCE: std::sync::Once = std::sync::Once::new();
            ONCE.call_once(|| crate::warn!("ui scale: no image ({e}): the interface keeps its size"));
            None
        }
    }
}

/// Records the copy into `cb` (recording, outside rendering): the layer scaled around its centre
/// into our image, its edges stretched across the rest.
pub fn record(cb: vk::CommandBuffer, j: Job) {
    super::vk::defer::external(cb, Box::new(move |cb| unsafe { scale_into(cb, j.layer, j.tmp, j.w, j.h, j.s, j.filter) }));
}

unsafe fn scale_into(cb: vk::CommandBuffer, layer: vk::Image, tmp: vk::Image, w: u32, h: u32, s: f32, filter: vk::Filter) {
    let c = ctx();
    let range = vk::ImageSubresourceRange { aspect_mask: vk::ImageAspectFlags::COLOR, base_mip_level: 0, level_count: 1, base_array_layer: 0, layer_count: 1 };
    let layers = vk::ImageSubresourceLayers { aspect_mask: vk::ImageAspectFlags::COLOR, mip_level: 0, base_array_layer: 0, layer_count: 1 };
    let mb = vk::MemoryBarrier::default().src_access_mask(vk::AccessFlags::MEMORY_WRITE).dst_access_mask(vk::AccessFlags::MEMORY_READ | vk::AccessFlags::MEMORY_WRITE);
    // (Its contents are written whole each time: the earlier ones are not kept.)
    let tmp_general = vk::ImageMemoryBarrier::default()
        .src_access_mask(vk::AccessFlags::empty())
        .dst_access_mask(vk::AccessFlags::TRANSFER_WRITE | vk::AccessFlags::TRANSFER_READ)
        .old_layout(vk::ImageLayout::UNDEFINED)
        .new_layout(vk::ImageLayout::GENERAL)
        .src_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
        .dst_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
        .image(tmp)
        .subresource_range(range);
    let (dw, dh) = (((w as f32) * s).round().max(1.0) as i32, ((h as f32) * s).round().max(1.0) as i32);
    let (dx, dy) = ((w as i32 - dw) / 2, (h as i32 - dh) / 2);
    // The layer smaller in the middle, its edges stretched across the margins: a full-screen
    // fade or dimming in the layer still covers the screen, a full-width bar still reaches its
    // sides, and transparent edges (the HUD's) leave the margins transparent.
    let xs = [(0, 1, 0, dx), (0, w as i32, dx, dx + dw), (w as i32 - 1, w as i32, dx + dw, w as i32)];
    let ys = [(0, 1, 0, dy), (0, h as i32, dy, dy + dh), (h as i32 - 1, h as i32, dy + dh, h as i32)];
    let mut blits = Vec::with_capacity(9);
    for &(sx0, sx1, dx0, dx1) in &xs {
        for &(sy0, sy1, dy0, dy1) in &ys {
            if dx1 > dx0 && dy1 > dy0 {
                blits.push(vk::ImageBlit {
                    src_subresource: layers,
                    src_offsets: [vk::Offset3D { x: sx0, y: sy0, z: 0 }, vk::Offset3D { x: sx1, y: sy1, z: 1 }],
                    dst_subresource: layers,
                    dst_offsets: [vk::Offset3D { x: dx0, y: dy0, z: 0 }, vk::Offset3D { x: dx1, y: dy1, z: 1 }],
                });
            }
        }
    }
    unsafe {
        // The layer's draws (and the last frame's reads of our image) done.
        c.device.cmd_pipeline_barrier(cb, vk::PipelineStageFlags::ALL_COMMANDS, vk::PipelineStageFlags::TRANSFER, vk::DependencyFlags::empty(), &[mb], &[], &[tmp_general]);
        c.device.cmd_blit_image(cb, layer, vk::ImageLayout::GENERAL, tmp, vk::ImageLayout::GENERAL, &blits, filter);
        // The composite samples it next.
        c.device.cmd_pipeline_barrier(cb, vk::PipelineStageFlags::TRANSFER, vk::PipelineStageFlags::ALL_COMMANDS, vk::DependencyFlags::empty(), &[mb], &[], &[]);
    }
}

/// The scale now (1.0: the original's).
/// Our scaled copy of the interface layer now (image, width, height; frame generation's
/// interface coverage reads what the composite samples: `framegen::capture`).
pub fn spare_image() -> Option<(vk::Image, u32, u32)> {
    SPARE.lock().as_ref().map(|s| (s.image, s.w, s.h))
}

pub fn scale() -> f32 {
    crate::pcsettings::ui_size()
}
