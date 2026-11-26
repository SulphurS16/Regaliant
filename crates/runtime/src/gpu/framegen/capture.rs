//! Frame generation's HUD-less picture and interface coverage (`gpu::framegen`).
//!
//! The title draws its interface into a layer of its own (3840x2160 RGBA16F, mostly transparent:
//! `gpu::uiscale`) that the last draw of its frame, the composite (ps 0x100888d100, the layer its
//! resource 0), puts over the picture in the display buffer. While frame generation runs, that
//! draw is made twice (`Exec::fg_capture_composite`): first with a transparent 1x1 image in the
//! layer's place, after which the display buffer holds the frame without its interface and is
//! copied here (the HUD-less picture, in the display buffer's own format and size: the title's
//! output encoding, PQ for HDR, untouched), then as the title made it. Whether the composite
//! writes every pixel or blends the layer over the picture already there, the first draw leaves
//! the picture without the interface and the second the title's own. The layer (or its scaled
//! copy under Menu & HUD Size: what the composite samples) is copied beside it at the display
//! buffer's size (RGBA8: its alpha, the interface's coverage).
//!
//! The presenter takes them for the frame it shows (`for_present`: the same frame, from the same
//! display buffer), scales and places the HUD-less picture exactly as the frame itself (the same
//! halvings, sharpening and blit into an image of the swapchain's format and size) and turns the
//! layer's alpha into the interface coverage at the swapchain's size (`Present::record_ui`,
//! `rdna::kernels::ui_alpha`; the port's own text box counts as interface). Both are DLSS-G's
//! "HUD-less" and "UI alpha" inputs (`dlssg::tag_present`).
//!
//! Four slots by frame index: a frame's slot is written again four frames later, by which time
//! its present has long been recorded (the title renders into at most three display buffers).

use super::super::vk::ctx;
use ash::vk;
use parking_lot::Mutex;
use std::sync::atomic::{AtomicBool, AtomicU32, AtomicU64, Ordering::Relaxed, Ordering::SeqCst};

const SLOTS: usize = 4;
/// The interface layer's copy and the coverage made from it.
const UI_FORMAT: vk::Format = vk::Format::R8G8B8A8_UNORM;
/// The transparent image the composite samples in the layer's place.
const ZERO_FORMAT: vk::Format = vk::Format::R16G16B16A16_SFLOAT;

struct Img {
    image: vk::Image,
    memory: vk::DeviceMemory,
    view: vk::ImageView,
    format: vk::Format,
    w: u32,
    h: u32,
}

unsafe impl Send for Img {}

impl Img {
    unsafe fn new(format: vk::Format, w: u32, h: u32, usage: vk::ImageUsageFlags) -> Result<Img, String> {
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
            let image = c.device.create_image(&ici, None).map_err(|e| format!("vkCreateImage: {e}"))?;
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

    unsafe fn destroy(self) {
        let c = ctx();
        unsafe {
            c.device.destroy_image_view(self.view, None);
            c.device.destroy_image(self.image, None);
            c.device.free_memory(self.memory, None);
        }
    }
}

const RANGE: vk::ImageSubresourceRange = vk::ImageSubresourceRange { aspect_mask: vk::ImageAspectFlags::COLOR, base_mip_level: 0, level_count: 1, base_array_layer: 0, layer_count: 1 };

fn barrier(cb: vk::CommandBuffer, img: vk::Image, old: vk::ImageLayout, new: vk::ImageLayout, src: (vk::PipelineStageFlags, vk::AccessFlags), dst: (vk::PipelineStageFlags, vk::AccessFlags)) {
    let b = vk::ImageMemoryBarrier::default()
        .image(img)
        .old_layout(old)
        .new_layout(new)
        .src_access_mask(src.1)
        .dst_access_mask(dst.1)
        .src_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
        .dst_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
        .subresource_range(RANGE);
    unsafe { ctx().device.cmd_pipeline_barrier(cb, src.0, dst.0, vk::DependencyFlags::empty(), &[], &[], &[b]) };
}

const ANY: (vk::PipelineStageFlags, vk::AccessFlags) = (vk::PipelineStageFlags::ALL_COMMANDS, vk::AccessFlags::from_raw(vk::AccessFlags::MEMORY_READ.as_raw() | vk::AccessFlags::MEMORY_WRITE.as_raw()));
const XFER_W: (vk::PipelineStageFlags, vk::AccessFlags) = (vk::PipelineStageFlags::TRANSFER, vk::AccessFlags::TRANSFER_WRITE);
const XFER_R: (vk::PipelineStageFlags, vk::AccessFlags) = (vk::PipelineStageFlags::TRANSFER, vk::AccessFlags::TRANSFER_READ);

struct Slot {
    /// The frame recorded into it, the display buffer it came from, whether the layer was copied.
    frame: u32,
    rt: vk::Image,
    has_ui: bool,
    hudless: Option<Img>,
    ui: Option<Img>,
}

struct Capture {
    slots: Vec<Slot>,
    /// Images of an earlier size, freed once `RETIRE_FRAMES` frames later.
    retired: Vec<(Img, u32)>,
    zero: Option<Img>,
}

const RETIRE_FRAMES: u32 = 240;

static CAP: Mutex<Option<Capture>> = Mutex::new(None);
/// Per slot: the frame whose copies were played into a command buffer (u32::MAX: none).
static READY: [AtomicU32; SLOTS] = [const { AtomicU32::new(u32::MAX) }; SLOTS];
/// The transparent image was cleared (in a command buffer played).
static ZERO_READY: AtomicBool = AtomicBool::new(false);
/// Frames captured, frames whose present took them, capture failures (status line).
static CAPTURED: AtomicU64 = AtomicU64::new(0);
static USED: AtomicU64 = AtomicU64::new(0);
static MISSED: AtomicU64 = AtomicU64::new(0);

/// Frame generation runs and the HUD-less picture is wanted (`DS_FG_HUDLESS=0`: not).
pub fn wanted() -> bool {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    super::render_inputs_wanted() && *ON.get_or_init(|| std::env::var("DS_FG_HUDLESS").map_or(true, |v| v.trim() != "0"))
}

fn with_cap<R>(f: impl FnOnce(&mut Capture) -> R) -> R {
    let mut g = CAP.lock();
    let c = g.get_or_insert_with(|| Capture { slots: (0..SLOTS).map(|_| Slot { frame: u32::MAX, rt: vk::Image::null(), has_ui: false, hudless: None, ui: None }).collect(), retired: Vec::new(), zero: None });
    f(c)
}

/// The transparent image's view (made on first use; None when it cannot be).
pub fn transparent_view() -> Option<vk::ImageView> {
    with_cap(|c| {
        if c.zero.is_none() {
            match unsafe { Img::new(ZERO_FORMAT, 1, 1, vk::ImageUsageFlags::SAMPLED | vk::ImageUsageFlags::TRANSFER_DST) }.and_then(|i| match unsafe { clear_now(i.image) } {
                Ok(()) => Ok(i),
                Err(e) => {
                    unsafe { i.destroy() };
                    Err(e)
                }
            }) {
                Ok(i) => {
                    c.zero = Some(i);
                    ZERO_READY.store(true, Relaxed);
                }
                Err(e) => {
                    static ONCE: std::sync::Once = std::sync::Once::new();
                    ONCE.call_once(|| crate::warn!("frame generation: no transparent image ({e}): no HUD-less picture"));
                    return None;
                }
            }
        }
        c.zero.as_ref().map(|z| z.view)
    })
}

/// Clears `image` (1x1) to transparent and leaves it GENERAL (as the title's images are bound),
/// in a command buffer of its own, waited for: once, when the transparent image is made. (Not in
/// the executor's batch: the composite's draw may be the first command of a batch not begun yet,
/// whose command buffer is then still the GPU's. Recorded there, the first test of frame
/// generation on 2026-11-26 crashed in the driver's vkCmdPipelineBarrier.)
unsafe fn clear_now(image: vk::Image) -> Result<(), String> {
    let c = ctx();
    unsafe {
        let pool = c.device.create_command_pool(&vk::CommandPoolCreateInfo::default().queue_family_index(c.queue_family).flags(vk::CommandPoolCreateFlags::TRANSIENT), None).map_err(|e| format!("pool: {e}"))?;
        let r = (|| -> Result<(), String> {
            let cb = c.device.allocate_command_buffers(&vk::CommandBufferAllocateInfo::default().command_pool(pool).command_buffer_count(1)).map_err(|e| format!("command buffer: {e}"))?[0];
            c.device.begin_command_buffer(cb, &vk::CommandBufferBeginInfo::default().flags(vk::CommandBufferUsageFlags::ONE_TIME_SUBMIT)).map_err(|e| e.to_string())?;
            barrier(cb, image, vk::ImageLayout::UNDEFINED, vk::ImageLayout::GENERAL, ANY, XFER_W);
            let zero = vk::ClearColorValue { float32: [0.0; 4] };
            c.device.cmd_clear_color_image(cb, image, vk::ImageLayout::GENERAL, &zero, &[RANGE]);
            barrier(cb, image, vk::ImageLayout::GENERAL, vk::ImageLayout::GENERAL, XFER_W, ANY);
            c.device.end_command_buffer(cb).map_err(|e| e.to_string())?;
            let fence = c.device.create_fence(&vk::FenceCreateInfo::default(), None).map_err(|e| e.to_string())?;
            // (On the presenter's queue where there is one: the main queue's earlier batches may
            // wait for work this thread has yet to submit.)
            let cbs = [cb];
            let si = vk::SubmitInfo::default().command_buffers(&cbs);
            let submitted = match c.present_queue.as_ref() {
                Some(q) => {
                    let q = q.lock();
                    c.device.queue_submit(*q, &[si], fence).map_err(|e| format!("submit: {e}"))
                }
                None => c.submit(cb, &[], &[], fence),
            };
            submitted?;
            match c.device.wait_for_fences(&[fence], true, 5_000_000_000) {
                Ok(()) => {
                    c.device.destroy_fence(fence, None);
                    Ok(())
                }
                // (Still the GPU's: its pool and fence are left, not freed under it.)
                Err(e) => Err(format!("wait: {e}")),
            }
        })();
        if !matches!(&r, Err(e) if e.starts_with("wait")) {
            c.device.destroy_command_pool(pool, None);
        }
        r
    }
}

/// Makes or keeps `slot`'s image of this format and size (an old one retired).
fn ensure(img: &mut Option<Img>, retired: &mut Vec<(Img, u32)>, frame: u32, format: vk::Format, w: u32, h: u32, usage: vk::ImageUsageFlags) -> Result<(), String> {
    if img.as_ref().is_some_and(|i| (i.format, i.w, i.h) == (format, w, h)) {
        return Ok(());
    }
    if let Some(old) = img.take() {
        retired.push((old, frame));
    }
    *img = Some(unsafe { Img::new(format, w, h, usage)? });
    Ok(())
}

/// After the composite's first draw (the interface hidden): the display buffer `rt` (GENERAL,
/// `format`, `w`x`h`) copied as frame `frame`'s HUD-less picture, and the layer `ui` (GENERAL:
/// image, its width and height) at the display buffer's size beside it. Recorded into `cmd`.
pub fn record(cmd: vk::CommandBuffer, frame: u32, rt: vk::Image, format: vk::Format, w: u32, h: u32, ui: Option<(vk::Image, u32, u32)>) {
    let k = frame as usize % SLOTS;
    let made = with_cap(|c| -> Result<(vk::Image, Option<vk::Image>), String> {
        // (Retired images: freed once their frames are long done.)
        let mut keep = Vec::new();
        for (img, at) in c.retired.drain(..) {
            if frame.wrapping_sub(at) > RETIRE_FRAMES && frame > at {
                unsafe { img.destroy() };
            } else {
                keep.push((img, at));
            }
        }
        c.retired = keep;
        let s = &mut c.slots[k];
        ensure(&mut s.hudless, &mut c.retired, frame, format, w, h, vk::ImageUsageFlags::TRANSFER_DST | vk::ImageUsageFlags::TRANSFER_SRC)?;
        if ui.is_some() {
            ensure(&mut s.ui, &mut c.retired, frame, UI_FORMAT, w, h, vk::ImageUsageFlags::TRANSFER_DST | vk::ImageUsageFlags::STORAGE)?;
        }
        s.frame = frame;
        s.rt = rt;
        s.has_ui = ui.is_some() && s.ui.is_some();
        Ok((s.hudless.as_ref().unwrap().image, s.ui.as_ref().filter(|_| ui.is_some()).map(|i| i.image)))
    });
    let (hudless, ui_img) = match made {
        Ok(v) => v,
        Err(e) => {
            if MISSED.fetch_add(1, Relaxed) < 4 {
                crate::warn!("frame generation: no HUD-less image of {w}x{h} {format:?} ({e})");
            }
            return;
        }
    };
    READY[k].store(u32::MAX, SeqCst);
    super::super::vk::defer::external(cmd, Box::new(move |cb| {
        let d = &ctx().device;
        // The display buffer as the hidden-interface draw left it.
        barrier(cb, rt, vk::ImageLayout::GENERAL, vk::ImageLayout::GENERAL, ANY, XFER_R);
        barrier(cb, hudless, vk::ImageLayout::UNDEFINED, vk::ImageLayout::TRANSFER_DST_OPTIMAL, ANY, XFER_W);
        let layers = vk::ImageSubresourceLayers { aspect_mask: vk::ImageAspectFlags::COLOR, mip_level: 0, base_array_layer: 0, layer_count: 1 };
        let copy = vk::ImageCopy { src_subresource: layers, src_offset: vk::Offset3D::default(), dst_subresource: layers, dst_offset: vk::Offset3D::default(), extent: vk::Extent3D { width: w, height: h, depth: 1 } };
        unsafe { d.cmd_copy_image(cb, rt, vk::ImageLayout::GENERAL, hudless, vk::ImageLayout::TRANSFER_DST_OPTIMAL, &[copy]) };
        barrier(cb, hudless, vk::ImageLayout::TRANSFER_DST_OPTIMAL, vk::ImageLayout::TRANSFER_SRC_OPTIMAL, XFER_W, ANY);
        if let (Some((layer, lw, lh)), Some(ui_img)) = (ui, ui_img) {
            barrier(cb, layer, vk::ImageLayout::GENERAL, vk::ImageLayout::GENERAL, ANY, XFER_R);
            barrier(cb, ui_img, vk::ImageLayout::UNDEFINED, vk::ImageLayout::TRANSFER_DST_OPTIMAL, ANY, XFER_W);
            let corners = |w: u32, h: u32| [vk::Offset3D { x: 0, y: 0, z: 0 }, vk::Offset3D { x: w as i32, y: h as i32, z: 1 }];
            let region = vk::ImageBlit { src_subresource: layers, src_offsets: corners(lw, lh), dst_subresource: layers, dst_offsets: corners(w, h) };
            unsafe { d.cmd_blit_image(cb, layer, vk::ImageLayout::GENERAL, ui_img, vk::ImageLayout::TRANSFER_DST_OPTIMAL, &[region], vk::Filter::LINEAR) };
            barrier(cb, ui_img, vk::ImageLayout::TRANSFER_DST_OPTIMAL, vk::ImageLayout::GENERAL, XFER_W, ANY);
        }
        // (The display buffer and the layer go on as the title's.)
        barrier(cb, rt, vk::ImageLayout::GENERAL, vk::ImageLayout::GENERAL, XFER_R, ANY);
        READY[k].store(frame, SeqCst);
        CAPTURED.fetch_add(1, Relaxed);
    }));
}

/// What the presenter takes for frame `frame` shown from display buffer `src`.
pub struct ForPresent {
    pub hudless: vk::Image,
    pub format: vk::Format,
    pub w: u32,
    pub h: u32,
    /// The layer's copy (its view for the coverage kernel).
    pub ui: Option<vk::ImageView>,
}

/// The HUD-less picture and the layer's copy of frame `frame`, when they are there and come
/// from the display buffer `src` the presenter shows.
pub fn for_present(frame: u32, src: vk::Image) -> Option<ForPresent> {
    let k = frame as usize % SLOTS;
    if READY[k].load(SeqCst) != frame {
        MISSED.fetch_add(1, Relaxed);
        return None;
    }
    let g = CAP.lock();
    let s = &g.as_ref()?.slots[k];
    if s.frame != frame || s.rt != src {
        static ONCE: std::sync::Once = std::sync::Once::new();
        if s.rt != src {
            ONCE.call_once(|| crate::warn!("frame generation: the title's composite drew into another image than the one presented: no HUD-less picture"));
        }
        MISSED.fetch_add(1, Relaxed);
        return None;
    }
    let h = s.hudless.as_ref()?;
    USED.fetch_add(1, Relaxed);
    Some(ForPresent { hudless: h.image, format: h.format, w: h.w, h: h.h, ui: s.ui.as_ref().filter(|_| s.has_ui).map(|u| u.view) })
}

// ---------------------------------------------------------------------------
// The presenter's side: the HUD-less picture and the coverage at the swapchain's size.

/// The presenter's frame generation images (two per presenter slot) and the coverage kernel.
pub struct Present {
    dsl: vk::DescriptorSetLayout,
    layout: vk::PipelineLayout,
    pipeline: vk::Pipeline,
    pool: vk::DescriptorPool,
    sets: Vec<vk::DescriptorSet>,
    /// Per presenter slot: the HUD-less picture (the swapchain's format) and the coverage.
    imgs: Vec<(Img, Img)>,
}

unsafe impl Send for Present {}

impl Present {
    pub unsafe fn new(slots: usize) -> Result<Present, String> {
        let c = ctx();
        let d = &c.device;
        unsafe {
            let bindings = [0, 1].map(|b| vk::DescriptorSetLayoutBinding::default().binding(b).descriptor_type(vk::DescriptorType::STORAGE_IMAGE).descriptor_count(1).stage_flags(vk::ShaderStageFlags::COMPUTE));
            let dsl = d.create_descriptor_set_layout(&vk::DescriptorSetLayoutCreateInfo::default().bindings(&bindings), None).map_err(|e| format!("set layout: {e}"))?;
            let pcr = [vk::PushConstantRange { stage_flags: vk::ShaderStageFlags::COMPUTE, offset: 0, size: 48 }];
            let dsls = [dsl];
            let layout = d.create_pipeline_layout(&vk::PipelineLayoutCreateInfo::default().set_layouts(&dsls).push_constant_ranges(&pcr), None).map_err(|e| format!("layout: {e}"))?;
            let code = rdna::kernels::ui_alpha();
            let module = d.create_shader_module(&vk::ShaderModuleCreateInfo::default().code(&code), None).map_err(|e| format!("module: {e}"))?;
            let stage = vk::PipelineShaderStageCreateInfo::default().stage(vk::ShaderStageFlags::COMPUTE).module(module).name(c"main");
            let pipeline = d.create_compute_pipelines(vk::PipelineCache::null(), &[vk::ComputePipelineCreateInfo::default().stage(stage).layout(layout)], None).map_err(|(_, e)| format!("pipeline: {e}"));
            d.destroy_shader_module(module, None);
            let pipeline = pipeline?[0];
            let sizes = [vk::DescriptorPoolSize { ty: vk::DescriptorType::STORAGE_IMAGE, descriptor_count: 2 * slots as u32 }];
            let pool = d.create_descriptor_pool(&vk::DescriptorPoolCreateInfo::default().max_sets(slots as u32).pool_sizes(&sizes), None).map_err(|e| format!("pool: {e}"))?;
            let layouts = vec![dsl; slots];
            let sets = d.allocate_descriptor_sets(&vk::DescriptorSetAllocateInfo::default().descriptor_pool(pool).set_layouts(&layouts)).map_err(|e| format!("sets: {e}"))?;
            Ok(Present { dsl, layout, pipeline, pool, sets, imgs: Vec::new() })
        }
    }

    /// The images at the swapchain's size and format (`idle`: waits for frames still using old
    /// ones).
    pub unsafe fn ensure(&mut self, slots: usize, format: vk::Format, w: u32, h: u32, idle: impl FnOnce()) -> Result<(), String> {
        if self.imgs.len() == slots && self.imgs.iter().all(|(a, _)| (a.format, a.w, a.h) == (format, w, h)) {
            return Ok(());
        }
        idle();
        unsafe {
            for (a, b) in self.imgs.drain(..) {
                a.destroy();
                b.destroy();
            }
            for _ in 0..slots {
                let a = Img::new(format, w, h, vk::ImageUsageFlags::TRANSFER_DST | vk::ImageUsageFlags::TRANSFER_SRC | vk::ImageUsageFlags::SAMPLED)?;
                let b = match Img::new(UI_FORMAT, w, h, vk::ImageUsageFlags::STORAGE | vk::ImageUsageFlags::SAMPLED | vk::ImageUsageFlags::TRANSFER_SRC) {
                    Ok(b) => b,
                    Err(e) => {
                        a.destroy();
                        return Err(e);
                    }
                };
                self.imgs.push((a, b));
            }
        }
        crate::info!("frame generation: HUD-less and interface images at {w}x{h} {format:?}");
        Ok(())
    }

    /// The HUD-less image of presenter slot `slot` (the presenter scales the HUD-less picture
    /// into it as it does the frame).
    pub fn hudless(&self, slot: usize) -> vk::Image {
        self.imgs[slot].0.image
    }

    /// After the HUD-less picture went into `slot`'s image (TRANSFER_DST_OPTIMAL): GENERAL, for
    /// DLSS-G at the present.
    pub fn hudless_done(&self, cmd: vk::CommandBuffer, slot: usize) {
        barrier(cmd, self.imgs[slot].0.image, vk::ImageLayout::TRANSFER_DST_OPTIMAL, vk::ImageLayout::GENERAL, XFER_W, ANY);
    }

    /// The coverage into `slot`'s image from the layer's copy `src` (`sw`x`sh`): its alpha over the
    /// picture's place `rect`, the port's text box (`text_box`) fully covered, 0 elsewhere.
    pub unsafe fn record_ui(&mut self, cmd: vk::CommandBuffer, slot: usize, src: vk::ImageView, sw: u32, sh: u32, rect: (u32, u32, u32, u32), text_box: Option<(u32, u32, u32, u32)>) {
        let d = &ctx().device;
        let (dst, w, h) = {
            let b = &self.imgs[slot].1;
            (b, b.w, b.h)
        };
        let infos = [[vk::DescriptorImageInfo { sampler: vk::Sampler::null(), image_view: src, image_layout: vk::ImageLayout::GENERAL }], [vk::DescriptorImageInfo { sampler: vk::Sampler::null(), image_view: dst.view, image_layout: vk::ImageLayout::GENERAL }]];
        let writes = [0usize, 1].map(|k| vk::WriteDescriptorSet::default().dst_set(self.sets[slot]).dst_binding(k as u32).descriptor_type(vk::DescriptorType::STORAGE_IMAGE).image_info(&infos[k]));
        let (tx0, ty0, tx1, ty1) = text_box.map_or((0, 0, 0, 0), |(x, y, bw, bh)| (x, y, x + bw, y + bh));
        let pc: [u32; 12] = [rect.0, rect.1, rect.2, rect.3, sw, sh, w, h, tx0, ty0, tx1, ty1];
        let bytes: Vec<u8> = pc.iter().flat_map(|v| v.to_le_bytes()).collect();
        let dst_image = dst.image;
        unsafe {
            d.update_descriptor_sets(&writes, &[]);
            barrier(cmd, dst_image, vk::ImageLayout::UNDEFINED, vk::ImageLayout::GENERAL, ANY, (vk::PipelineStageFlags::COMPUTE_SHADER, vk::AccessFlags::SHADER_WRITE));
            d.cmd_bind_pipeline(cmd, vk::PipelineBindPoint::COMPUTE, self.pipeline);
            d.cmd_bind_descriptor_sets(cmd, vk::PipelineBindPoint::COMPUTE, self.layout, 0, &[self.sets[slot]], &[]);
            d.cmd_push_constants(cmd, self.layout, vk::ShaderStageFlags::COMPUTE, 0, &bytes);
            let n = rdna::kernels::UI_ALPHA_LOCAL;
            d.cmd_dispatch(cmd, w.div_ceil(n), h.div_ceil(n), 1);
        }
        barrier(cmd, dst_image, vk::ImageLayout::GENERAL, vk::ImageLayout::GENERAL, (vk::PipelineStageFlags::COMPUTE_SHADER, vk::AccessFlags::SHADER_WRITE), ANY);
    }

    /// `slot`'s images as Streamline resources (GENERAL), for DLSS-G's present tags.
    pub fn resources(&self, slot: usize) -> (super::super::streamline::SlResource, super::super::streamline::SlResource) {
        let (a, b) = &self.imgs[slot];
        let r = |i: &Img, usage: vk::ImageUsageFlags| super::super::streamline::SlResource::image(i.image, i.view, vk::ImageLayout::GENERAL, i.format, (i.w, i.h), usage);
        (r(a, vk::ImageUsageFlags::TRANSFER_DST | vk::ImageUsageFlags::TRANSFER_SRC | vk::ImageUsageFlags::SAMPLED), r(b, vk::ImageUsageFlags::STORAGE | vk::ImageUsageFlags::SAMPLED | vk::ImageUsageFlags::TRANSFER_SRC))
    }
}

impl Drop for Present {
    fn drop(&mut self) {
        unsafe {
            for (a, b) in self.imgs.drain(..) {
                a.destroy();
                b.destroy();
            }
            let d = &ctx().device;
            d.destroy_pipeline(self.pipeline, None);
            d.destroy_pipeline_layout(self.layout, None);
            d.destroy_descriptor_pool(self.pool, None);
            d.destroy_descriptor_set_layout(self.dsl, None);
        }
    }
}

/// Status line.
/// The composite's capture: `""` when it is tried (counted), else why it was not taken; each
/// reason logged the first time and counted (status line).
pub fn skipped(why: &str) {
    if why.is_empty() {
        TRIED.fetch_add(1, Relaxed);
        return;
    }
    let mut g = SKIPS.lock();
    let key: String = why.chars().take_while(|c| *c != '(').collect::<String>().trim().to_string();
    match g.iter_mut().find(|(k, _)| *k == key) {
        Some((_, n)) => *n += 1,
        None => {
            crate::warn!("frame generation: the HUD-less picture not taken: {why}");
            g.push((key, 1));
        }
    }
}

/// The display buffers' addresses as presented (`note_display_buffer`).
static DISPLAY: [AtomicU64; 4] = [const { AtomicU64::new(0) }; 4];
/// Pixel stages logged drawing into a display buffer (`note_display_draw`).
static DISPLAY_PS: Mutex<Vec<u64>> = Mutex::new(Vec::new());

/// A display buffer the presenter shows (its address).
pub fn note_display_buffer(addr: u64) {
    if DISPLAY.iter().any(|a| a.load(Relaxed) == addr) {
        return;
    }
    if let Some(a) = DISPLAY.iter().find(|a| a.load(Relaxed) == 0) {
        a.store(addr, Relaxed);
    }
}

/// A draw other than the composite drew into an image at `base`: when that is a display buffer,
/// its pixel stage is logged (once each, eight at most): what draws the final picture when the
/// composite is not met (HDR's, another version of the title's).
pub fn note_display_draw(base: u64, ps: u64, format: vk::Format, w: u32, h: u32) {
    if base == 0 || !DISPLAY.iter().any(|a| a.load(Relaxed) == base) {
        return;
    }
    let mut g = DISPLAY_PS.lock();
    if g.len() >= 8 || g.contains(&ps) {
        return;
    }
    g.push(ps);
    crate::info!("frame generation: ps {ps:#x} draws into display buffer {base:#x} ({format:?} {w}x{h}); the composites captured are ps {:#x} and {:#x}", super::super::uiscale::COMPOSITE_PS, super::super::uiscale::COMPOSITE_PS_HDR);
}

/// Composite draws met while capturing, and why some were not taken (reason, count).
static TRIED: AtomicU64 = AtomicU64::new(0);
static SKIPS: Mutex<Vec<(String, u64)>> = Mutex::new(Vec::new());

pub fn status_line() -> String {
    let n = CAPTURED.load(Relaxed);
    let tried = TRIED.load(Relaxed);
    if n == 0 && MISSED.load(Relaxed) == 0 && tried == 0 && !super::super::streamline::loaded(super::super::streamline::FEATURE_DLSS_G) {
        return String::new();
    }
    let skips: Vec<String> = SKIPS.lock().iter().map(|(k, c)| format!("{k} {c}")).collect();
    format!(
        "    frame generation's HUD-less picture: {tried} composite draws met, {n} captured, {} presented with it, {} frames without{}\n",
        USED.load(Relaxed),
        MISSED.load(Relaxed),
        if skips.is_empty() { String::new() } else { format!("; not taken: {}", skips.join(", ")) }
    )
}
