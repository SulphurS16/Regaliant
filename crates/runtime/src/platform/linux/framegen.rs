//! Optional FSR3 interpolation. Snapshots are made in guest command order; WSI stays ours.
//! No guest flip is synthesized. Off is free of GPU copies and vendor loading.
use crate::gpu::{
    upscale::Tex,
    vk::{self, ctx},
};
use ash::vk::{self as v, Handle};
use parking_lot::Mutex;
use std::{
    collections::{BTreeMap, HashMap},
    ffi::{CStr, CString},
    sync::{
        atomic::{AtomicBool, AtomicU32, AtomicU64, Ordering::Relaxed},
        Arc, OnceLock,
    },
    time::{Duration, Instant},
};

#[path = "framegen_constants.rs"]
mod constants;
use constants::FrameConstants;

static EPOCH: AtomicU64 = AtomicU64::new(0);
static CHOICE: AtomicU32 = AtomicU32::new(0);
static FAILED: AtomicBool = AtomicBool::new(false);
static PRESENTING: AtomicBool = AtomicBool::new(false);
static DUMP_REQUEST: AtomicBool = AtomicBool::new(false);
static GENERATED: AtomicU64 = AtomicU64::new(0);
static INPUT_AGE: AtomicU64 = AtomicU64::new(0);
static COMPUTE_MS: AtomicU64 = AtomicU64::new(0);
static SKIPPED: AtomicU64 = AtomicU64::new(0);
// Bounded, stage-specific diagnostics: no per-draw logging when FG is off.
static NOTES: Mutex<BTreeMap<&'static str, u64>> = Mutex::new(BTreeMap::new());
pub fn note(stage: &'static str, detail: impl std::fmt::Display) {
    let n = {
        let mut notes = NOTES.lock();
        let n = notes.entry(stage).or_default();
        *n += 1;
        *n
    };
    if n <= 4 || n % 600 == 0 {
        crate::info!("FG {stage} [{n}]: {detail}");
    }
}
fn diagnostic_state() {
    if let Some(dir) = crate::run_dir() {
        let value = serde_json::json!({"choice":CHOICE.load(Relaxed), "failed":FAILED.load(Relaxed),
            "present_active":vk::present::active(), "generated":GENERATED.load(Relaxed),
            "skipped":SKIPPED.load(Relaxed), "stages":*NOTES.lock()});
        let _ = std::fs::write(dir.join("framegen-state.json"), value.to_string());
    }
}
// shaderuicompose SDR variants: native sampling and filtered/upscaled sampling.
// Program VAs change with allocation order and the selected resolution. These are the
// decoded-code hashes used by the shader store, not a pipeline/translation fingerprint.
fn composite_hash(hash: u64) -> bool {
    matches!(hash, 0xb72a4daf3c99cae3 | 0x4eca1d1844c8d0c1)
}
pub fn composite(addr: u64) -> bool {
    if !enabled() || addr == 0 { return false; }
    match crate::gpu::shaders::program_hash(addr) {
        Ok(hash) if composite_hash(hash) => {
            note("composite recognized", format_args!("ps {addr:#x}, code {hash:#x}"));
            true
        }
        Ok(hash) => {
            if crate::gpu::particles::origin(hash).is_some_and(|s| s.as_ref() == "shaderuicompose") {
                note("unsupported composite", format_args!("ps {addr:#x}, code {hash:#x}; no HUDless capture"));
            }
            false
        }
        Err(e) => { note("composite identification failed", e); false }
    }
}
static DEVICE_INPUTS_SUPPORTED: AtomicBool = AtomicBool::new(false);
/// The SDK's R16/RG16/RGBA16 storage-image shaders declare this capability.
/// Enable it at device creation even if the user enables FG later in PC settings.
/// The pipeline store includes the enabled feature set in its compatibility key.
pub fn apply_features(supported: &v::PhysicalDeviceFeatures, enabled: &mut v::PhysicalDeviceFeatures) {
    let supported = supported.shader_storage_image_extended_formats != 0;
    enabled.shader_storage_image_extended_formats = u32::from(supported);
    DEVICE_INPUTS_SUPPORTED.store(supported, Relaxed);
    note("device features", format_args!("shaderStorageImageExtendedFormats={supported}; required by FSR3 FI/OF"));
}
pub fn set_choice(value: u32) {
    let mut value = std::env::var("DS_FRAME_GENERATION")
        .ok()
        .map_or(value, |v| u32::from(v == "fsr3" || v == "1"));
    if value != 0 && crate::gpu::par::whole_run() {
        crate::warn!("FSR3 disabled: experimental whole-run recording failed visual correctness; unset DS_PAR_WHOLE_RUN");
        value = 0;
    }
    note("setting", format_args!("requested {value}, env {:?}, failed {}, presenter {}", std::env::var("DS_FRAME_GENERATION").ok(), FAILED.load(Relaxed), vk::present::active()));
    if CHOICE.swap(value.min(1), Relaxed) != value.min(1) {
        EPOCH.fetch_add(1, Relaxed);
        if let Some(s) = INPUTS.lock().as_mut() { s.current = None; s.ready.clear(); }
        crate::info!("Linux frame generation: {} (experimental, native resolution; whole-run recording must be off)",if value==0 {"off"} else {"FSR3"});
    }
}
pub fn enabled() -> bool {
    static LAST: AtomicU32 = AtomicU32::new(u32::MAX);
    let gate = if CHOICE.load(Relaxed) != 1 { 0 }
        else if FAILED.load(Relaxed) { 1 }
        else if !vk::present::active() { 2 } else { 3 };
    if LAST.swap(gate, Relaxed) != gate {
        note("gate", match gate {
            0 => "setting is off; no context requested",
            1 => "disabled after an error; restart to retry; real frames continue",
            2 => "waiting for visible presenter; offscreen/headless cannot generate",
            _ => "enabled; waiting for paired TAA and SDR composite inputs",
        });
    }
    gate == 3
}
pub fn timing() -> (bool, f64, f64) {
    (
        presenting(),
        f64::from_bits(INPUT_AGE.load(Relaxed)),
        f64::from_bits(COMPUTE_MS.load(Relaxed)),
    )
}
pub fn presenting() -> bool {
    PRESENTING.load(Relaxed)
}
fn fail(e: impl std::fmt::Display) {
    if !FAILED.swap(true, Relaxed) {
        crate::error!("FSR3 frame generation disabled for this session: {e}; real frames continue");
        diagnostic_state();
    }
}
fn path() -> std::path::PathBuf {
    std::env::var_os("DS_FSR_FG_LIB")
        .map(Into::into)
        .unwrap_or_else(|| {
            std::env::current_exe()
                .unwrap()
                .parent()
                .unwrap()
                .join("upscalers/libregaliant_fsrfg.so")
        })
}
#[repr(C)]
#[derive(Clone, Copy, Default)]
struct ImageDesc {
    image: u64,
    format: u32,
    width: u32,
    height: u32,
}
#[repr(C)]
struct Params {
    color: ImageDesc,
    hudless: ImageDesc,
    depth: ImageDesc,
    motion: ImageDesc,
    output: ImageDesc,
    prepared_motion: ImageDesc,
    motion_status: ImageDesc,
    reprojection: [f32; 16],
    jitter_x: f32,
    jitter_y: f32,
    near_plane: f32,
    far_plane: f32,
    fov: f32,
    milliseconds: f32,
    reset: u32,
    linear: u32,
}
type Create = unsafe extern "C" fn(
    v::Instance,
    v::PhysicalDevice,
    v::Device,
    u32,
    u32,
    u32,
    u32,
    u32,
    *mut usize,
) -> i32;
type Dispatch = unsafe extern "C" fn(usize, v::CommandBuffer, *const Params) -> i32;
type Destroy = unsafe extern "C" fn(usize);
struct Api {
    create: Create,
    dispatch: Dispatch,
    destroy: Destroy,
}
static API: OnceLock<Result<Api, String>> = OnceLock::new();
fn api() -> Result<&'static Api, String> {
    API.get_or_init(|| unsafe {
        let p = CString::new(path().as_os_str().as_encoded_bytes()).map_err(|e| e.to_string())?;
        let h = libc::dlopen(p.as_ptr(), libc::RTLD_NOW | libc::RTLD_LOCAL);
        if h.is_null() {
            return Err(format!(
                "{}: {}; run scripts/linux-build-framegen.py",
                path().display(),
                {
                    let e = libc::dlerror();
                    if e.is_null() {
                        std::borrow::Cow::Borrowed("unknown loader error")
                    } else {
                        CStr::from_ptr(e).to_string_lossy()
                    }
                }
            ));
        }
        // Process lifetime like the Vulkan loader. Do not dlclose code under runtime workers.
        let sym = |name: &CStr| {
            let p = libc::dlsym(h, name.as_ptr());
            if p.is_null() {
                Err(format!("missing {}", name.to_string_lossy()))
            } else {
                Ok(p)
            }
        };
        let abi: unsafe extern "C" fn() -> u32 = std::mem::transmute(sym(c"regaliant_fsrfg_abi")?);
        if abi() != 2 {
            return Err("FSR3 bridge ABI mismatch".into());
        }
        crate::info!("FSR3 native bridge loaded: {}", path().display());
        Ok(Api {
            create: std::mem::transmute(sym(c"regaliant_fsrfg_create")?),
            dispatch: std::mem::transmute(sym(c"regaliant_fsrfg_dispatch")?),
            destroy: std::mem::transmute(sym(c"regaliant_fsrfg_destroy")?),
        })
    })
    .as_ref()
    .map_err(Clone::clone)
}
struct Image {
    desc: ImageDesc,
    memory: v::DeviceMemory,
}
impl Image {
    fn image(&self) -> v::Image {
        v::Image::from_raw(self.desc.image)
    }
    fn new(w: u32, h: u32, format: v::Format) -> Result<Self, String> {
        unsafe {
            let c = ctx();
            let info = v::ImageCreateInfo::default()
                .image_type(v::ImageType::TYPE_2D)
                .format(format)
                .extent(v::Extent3D {
                    width: w,
                    height: h,
                    depth: 1,
                })
                .mip_levels(1)
                .array_layers(1)
                .samples(v::SampleCountFlags::TYPE_1)
                .tiling(v::ImageTiling::OPTIMAL)
                .usage(
                    v::ImageUsageFlags::TRANSFER_SRC
                        | v::ImageUsageFlags::TRANSFER_DST
                        | v::ImageUsageFlags::SAMPLED
                        | v::ImageUsageFlags::STORAGE,
                );
            // TAA may execute on the async-compute family; the presenter uses graphics.
            let info = if c.families.len() > 1 {
                info.sharing_mode(v::SharingMode::CONCURRENT)
                    .queue_family_indices(&c.families)
            } else {
                info
            };
            let image = c
                .device
                .create_image(&info, None)
                .map_err(|e| e.to_string())?;
            let memory = match c.alloc(
                c.device.get_image_memory_requirements(image),
                v::MemoryPropertyFlags::DEVICE_LOCAL,
            ) {
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
            Ok(Self {
                desc: ImageDesc {
                    image: image.as_raw(),
                    format: format.as_raw() as u32,
                    width: w,
                    height: h,
                },
                memory,
            })
        }
    }
    unsafe fn destroy(&self) {
        unsafe {
            let d = &ctx().device;
            d.destroy_image(self.image(), None);
            d.free_memory(self.memory, None);
        }
    }
}
const COLOR: v::Format = v::Format::R16G16B16A16_SFLOAT;
fn range(aspect: v::ImageAspectFlags) -> v::ImageSubresourceRange {
    v::ImageSubresourceRange {
        aspect_mask: aspect,
        base_mip_level: 0,
        level_count: 1,
        base_array_layer: 0,
        layer_count: 1,
    }
}
fn barrier(
    cmd: v::CommandBuffer,
    image: v::Image,
    aspect: v::ImageAspectFlags,
    old: v::ImageLayout,
    new: v::ImageLayout,
) {
    unsafe {
        let b = v::ImageMemoryBarrier::default()
            .src_access_mask(if old == v::ImageLayout::UNDEFINED {
                v::AccessFlags::empty()
            } else {
                v::AccessFlags::MEMORY_READ | v::AccessFlags::MEMORY_WRITE
            })
            .dst_access_mask(v::AccessFlags::MEMORY_READ | v::AccessFlags::MEMORY_WRITE)
            .old_layout(old)
            .new_layout(new)
            .src_queue_family_index(v::QUEUE_FAMILY_IGNORED)
            .dst_queue_family_index(v::QUEUE_FAMILY_IGNORED)
            .image(image)
            .subresource_range(range(aspect));
        ctx().device.cmd_pipeline_barrier(
            cmd,
            v::PipelineStageFlags::ALL_COMMANDS,
            v::PipelineStageFlags::ALL_COMMANDS,
            v::DependencyFlags::empty(),
            &[],
            &[],
            &[b],
        );
    }
}
fn general(cmd: v::CommandBuffer, image: &Image, discard: bool) {
    barrier(
        cmd,
        image.image(),
        v::ImageAspectFlags::COLOR,
        if discard {
            v::ImageLayout::UNDEFINED
        } else {
            v::ImageLayout::GENERAL
        },
        v::ImageLayout::GENERAL,
    )
}
fn blit(cmd: v::CommandBuffer, src: v::Image, size: (u32, u32), dst: &Image) {
    unsafe {
        barrier(
            cmd,
            src,
            v::ImageAspectFlags::COLOR,
            v::ImageLayout::GENERAL,
            v::ImageLayout::GENERAL,
        );
        general(cmd, dst, true);
        let layer = v::ImageSubresourceLayers {
            aspect_mask: v::ImageAspectFlags::COLOR,
            mip_level: 0,
            base_array_layer: 0,
            layer_count: 1,
        };
        let region = v::ImageBlit::default()
            .src_subresource(layer)
            .dst_subresource(layer)
            .src_offsets([
                v::Offset3D::default(),
                v::Offset3D {
                    x: size.0 as i32,
                    y: size.1 as i32,
                    z: 1,
                },
            ])
            .dst_offsets([
                v::Offset3D::default(),
                v::Offset3D {
                    x: dst.desc.width as i32,
                    y: dst.desc.height as i32,
                    z: 1,
                },
            ]);
        ctx().device.cmd_blit_image(
            cmd,
            src,
            v::ImageLayout::GENERAL,
            dst.image(),
            v::ImageLayout::GENERAL,
            &[region],
            v::Filter::NEAREST,
        );
        general(cmd, dst, false);
        barrier(
            cmd,
            src,
            v::ImageAspectFlags::COLOR,
            v::ImageLayout::GENERAL,
            v::ImageLayout::GENERAL,
        );
    }
}
/// A transparent UI input for a second invocation of the title's exact composite.
/// The composite samples scene resource 2, filters it, then applies its SDR transfer
/// function. The old target contents are not a HUDless frame. Do not copy them.
pub fn transparent_ui(cmd: v::CommandBuffer) -> Result<v::ImageView, String> {
    static UI: Mutex<Option<(Image, v::ImageView)>> = Mutex::new(None);
    let mut g = UI.lock();
    if g.is_none() {
        let image = Image::new(1, 1, COLOR)?;
        let view = unsafe { ctx().device.create_image_view(&v::ImageViewCreateInfo::default()
            .image(image.image()).view_type(v::ImageViewType::TYPE_2D)
            .format(COLOR).subresource_range(range(v::ImageAspectFlags::COLOR)), None) }
            .map_err(|e| e.to_string())?;
        *g = Some((image, view));
    }
    let (image, view) = g.as_ref().unwrap();
    let raw = image.image();
    vk::defer::external(cmd, Box::new(move |cmd| unsafe {
        barrier(cmd, raw, v::ImageAspectFlags::COLOR, v::ImageLayout::UNDEFINED, v::ImageLayout::GENERAL);
        ctx().device.cmd_clear_color_image(cmd, raw, v::ImageLayout::GENERAL,
            &v::ClearColorValue { float32: [0.0; 4] }, &[range(v::ImageAspectFlags::COLOR)]);
        barrier(cmd, raw, v::ImageAspectFlags::COLOR, v::ImageLayout::GENERAL, v::ImageLayout::GENERAL);
    }));
    Ok(*view)
}
// Diagnostic readback only. Each allocation stays alive through the generation fence.
struct Readback {
    name: &'static str,
    desc: ImageDesc,
    buffer: v::Buffer,
    memory: v::DeviceMemory,
    ptr: *mut u8,
    bytes: usize,
}
fn readback(cmd: v::CommandBuffer, name: &'static str, image: &Image) -> Result<Readback, String> {
    unsafe {
        let d = image.desc;
        let stride = match v::Format::from_raw(d.format as i32) {
            COLOR => 8,
            v::Format::R16_UNORM => 2,
            _ => 4,
        };
        let bytes = d.width as usize * d.height as usize * stride;
        let (buffer, memory, ptr) =
            ctx().host_buffer(bytes as u64, v::BufferUsageFlags::TRANSFER_DST)?;
        general(cmd, image, false);
        let r = v::BufferImageCopy::default()
            .image_subresource(v::ImageSubresourceLayers {
                aspect_mask: v::ImageAspectFlags::COLOR,
                mip_level: 0,
                base_array_layer: 0,
                layer_count: 1,
            })
            .image_extent(v::Extent3D {
                width: d.width,
                height: d.height,
                depth: 1,
            });
        ctx().device.cmd_copy_image_to_buffer(
            cmd,
            image.image(),
            v::ImageLayout::GENERAL,
            buffer,
            &[r],
        );
        general(cmd, image, false);
        Ok(Readback {
            name,
            desc: d,
            buffer,
            memory,
            ptr,
            bytes,
        })
    }
}
fn save_readbacks(rows: Vec<Readback>, n: u64, params: &Params, current_id: u64, previous_id: u64, constants: &FrameConstants) {
    unsafe {
        let dir = crate::run_dir().map(|p| p.join(format!("fg-inputs-{n}")));
        if let Some(p) = &dir {
            let _ = std::fs::create_dir_all(p);
        }
        let mut manifest = Vec::new();
        for r in rows {
            let raw = std::slice::from_raw_parts(r.ptr, r.bytes);
            let format = v::Format::from_raw(r.desc.format as i32);
            if let Some(p) = &dir {
                if let Err(e) = std::fs::write(p.join(format!("{}.raw", r.name)), raw) {
                    crate::warn!("FG input dump: {e}");
                }
                if let Some(rgba) = crate::gpu::images::to_rgba8(format, raw) {
                    let _ = std::fs::write(
                        p.join(format!("{}.png", r.name)),
                        crate::gpu::png::encode_opaque(r.desc.width, r.desc.height, &rgba),
                    );
                }
            }
            manifest.push(serde_json::json!({"name":r.name,"width":r.desc.width,"height":r.desc.height,"format":format.as_raw(),"format_name":format!("{format:?}"),"bytes":r.bytes,"stats":crate::gpu::images::float_stats(format,raw),"image_handle":r.desc.image}));
            let d = &ctx().device;
            d.destroy_buffer(r.buffer, None);
            d.unmap_memory(r.memory);
            d.free_memory(r.memory, None);
        }
        if let Some(p) = dir {
            let value = serde_json::json!({"schema":2,"camera_source":"TAA per-view globals inverse projection +0x40", "depth_inverted":true,"depth_infinite":constants.infinite,"title_history_enabled":constants.history,"globals_words":constants.globals,"taa_words":constants.taa,"reprojection":params.reprojection,"jitter_units":"render pixels, screen x-right/y-down from globals +0x1c0", "motion_status_bits":"low byte: 0 unchanged, 1 depth-matched neighbour, 2 camera reprojection, 3 zero last resort; bit 8 nonfinite source, bit 9 out-of-range source", "images":manifest,"raw":"native Vulkan texels, row-major, no header", "jitter":[params.jitter_x,params.jitter_y],"motion_scale":[-(params.motion.width as f32),-(params.motion.height as f32)],"camera_near":params.near_plane,"camera_far":params.far_plane,"camera_fov":params.fov,"reset":params.reset,"linear":params.linear,"current_taa_id":current_id,"previous_taa_id":previous_id,"transfer_function":if params.linear==0 {"SRGB encoded SDR"} else {"linear scRGB"},"preview_alpha":"opaque, as the swapchain; raw alpha retains SDK inpainting weight","history":"previous-hudless is the exact retained input copied by SDK to its previous interpolation source; previous-real is the last submitted full backbuffer"});
            let _ = std::fs::write(p.join("inputs.json"), value.to_string());
            crate::info!("FG diagnostic inputs/output: {}", p.display());
        }
    }
}
struct Snapshot {
    depth: Image,
    motion: Image,
    buffer: v::Buffer,
    _memory: v::DeviceMemory,
}
#[derive(Clone)]
struct Captured {
    snap: Arc<Snapshot>,
    hudless: Option<Arc<Image>>,
    id: u64,
    constants: FrameConstants,
    at: Instant,
    linear: bool,
}
#[derive(Default)]
struct Inputs {
    pool: Vec<Arc<Snapshot>>,
    huds: Vec<Arc<Image>>,
    current: Option<Captured>,
    ready: HashMap<u64, Captured>,
    id: u64,
}
static INPUTS: Mutex<Option<Inputs>> = Mutex::new(None);
/// Resolve the same pointer table used by TAA. Copy now: deferred replay must never
/// reread a guest per-frame allocation after the CPU has recycled it.
pub fn read_constants(user: u64, size: (u32,u32)) -> Result<FrameConstants, String> {
    let words = |addr: u64, bytes: u64| -> Result<Vec<u32>,String> {
        if addr == 0 || addr & 3 != 0 || !crate::gpu::cp::guest_readable(addr, bytes) {
            return Err(format!("unreadable FG constants {addr:#x}+{bytes:#x}"));
        }
        Ok((0..bytes/4).map(|i| unsafe { ((addr+4*i) as *const u32).read_volatile() }).collect())
    };
    let table = words(user,16)?;
    let ptr = |i| table[i] as u64 | (table[i+1] as u64) << 32;
    let globals = words(ptr(0),0x2b0)?;
    let taa = words(ptr(2),0x64)?;
    let result = FrameConstants::decode(globals.clone(),taa.clone(),size);
    if let Ok(c) = &result {
        note("title camera",format_args!("near {}, far {}, infinite {}, vertical FOV {} rad, jitter {:?} render pixels, history {}",
            c.near,c.far,c.infinite,c.fov,c.jitter,c.history));
    } else {
        static SAVED: AtomicBool = AtomicBool::new(false);
        if !SAVED.swap(true,Relaxed) {
            if let Some(dir) = crate::run_dir() {
                let value=serde_json::json!({"user":user,"global_address":ptr(0),"taa_address":ptr(2),
                    "render_size":size,"global_words":globals,"taa_words":taa,"error":result.as_ref().err()});
                let _=std::fs::write(dir.join("fg-rejected-constants.json"),value.to_string());
            }
        }
    }
    result
}
pub fn reject_taa(reason: &str) {
    if let Some(s) = INPUTS.lock().as_mut() { s.current = None; s.ready.clear(); }
    EPOCH.fetch_add(1, Relaxed);
    note("TAA constants rejected", reason);
}
/// Called at deferred replay, after TAA read these exact depth/MV images. Never in a render pass.
pub fn capture_taa(cmd: v::CommandBuffer, depth: Tex, motion: Tex, constants: FrameConstants) {
    if !enabled() {
        return;
    }
    if !DEVICE_INPUTS_SUPPORTED.load(Relaxed) {
        fail("FSR3 requires shaderStorageImageExtendedFormats; unavailable on this device");
        return;
    }
    if let Err(e) = api() {
        fail(e);
        return;
    }
    let result = (|| -> Result<(), String> {
        if depth.size != motion.size || motion.format != v::Format::R16G16_SFLOAT {
            return Err(format!(
                "unsupported TAA inputs {:?}/{:?} {:?}",
                depth.size, motion.size, motion.format
            ));
        }
        let (format, bytes) = match depth.format {
            v::Format::D32_SFLOAT | v::Format::D32_SFLOAT_S8_UINT | v::Format::R32_SFLOAT => {
                (v::Format::R32_SFLOAT, 4)
            }
            v::Format::D16_UNORM | v::Format::R16_UNORM => (v::Format::R16_UNORM, 2),
            _ => return Err(format!("unsupported depth {:?}", depth.format)),
        };
        let mut g = INPUTS.lock();
        let s = g.get_or_insert_with(Inputs::default);
        s.current = None;
        let found = s
            .pool
            .iter()
            .find(|p| {
                Arc::strong_count(p) == 1
                    && (p.depth.desc.width, p.depth.desc.height) == depth.size
                    && p.depth.desc.format == format.as_raw() as u32
            })
            .cloned();
        let snap = match found {
            Some(p) => p,
            None => {
                // Bounded process-lifetime pool: never free an image referenced by deferred work.
                if s.pool.len() >= 8 {
                    return Err(
                        "snapshot pool full; disable FG before changing render size repeatedly"
                            .into(),
                    );
                }
                let d = Image::new(depth.size.0, depth.size.1, format)?;
                let m = Image::new(motion.size.0, motion.size.1, motion.format)?;
                let (buffer, memory) = ctx().device_buffer(
                    depth.size.0 as u64 * depth.size.1 as u64 * bytes,
                    v::BufferUsageFlags::TRANSFER_SRC | v::BufferUsageFlags::TRANSFER_DST,
                )?;
                let p = Arc::new(Snapshot {
                    depth: d,
                    motion: m,
                    buffer,
                    _memory: memory,
                });
                s.pool.push(p.clone());
                p
            }
        };
        unsafe {
            // Order reuse of the intermediate transfer buffer as well as the images.
            let reuse = v::MemoryBarrier::default()
                .src_access_mask(v::AccessFlags::MEMORY_READ | v::AccessFlags::MEMORY_WRITE)
                .dst_access_mask(v::AccessFlags::MEMORY_READ | v::AccessFlags::MEMORY_WRITE);
            ctx().device.cmd_pipeline_barrier(
                cmd,
                v::PipelineStageFlags::ALL_COMMANDS,
                v::PipelineStageFlags::ALL_COMMANDS,
                v::DependencyFlags::empty(),
                &[reuse],
                &[],
                &[],
            );
            // VUID-03320: a combined depth/stencil image needs both aspects in barriers;
            // the copy itself selects depth alone. No separate-layout feature is assumed.
            let depth_sync = if depth.format == v::Format::D32_SFLOAT_S8_UINT {
                v::ImageAspectFlags::DEPTH | v::ImageAspectFlags::STENCIL
            } else {
                depth.range.aspect_mask
            };
            // Buffer transfer preserves depth bits without illegal depth-to-colour image copies.
            barrier(
                cmd,
                depth.image,
                depth_sync,
                v::ImageLayout::GENERAL,
                v::ImageLayout::GENERAL,
            );
            general(cmd, &snap.depth, true);
            let layers = |a| v::ImageSubresourceLayers {
                aspect_mask: a,
                mip_level: 0,
                base_array_layer: 0,
                layer_count: 1,
            };
            let r = v::BufferImageCopy::default()
                .image_subresource(layers(depth.range.aspect_mask))
                .image_extent(v::Extent3D {
                    width: depth.size.0,
                    height: depth.size.1,
                    depth: 1,
                });
            ctx().device.cmd_copy_image_to_buffer(
                cmd,
                depth.image,
                v::ImageLayout::GENERAL,
                snap.buffer,
                &[r],
            );
            let b = v::MemoryBarrier::default()
                .src_access_mask(v::AccessFlags::TRANSFER_WRITE)
                .dst_access_mask(v::AccessFlags::TRANSFER_READ);
            ctx().device.cmd_pipeline_barrier(
                cmd,
                v::PipelineStageFlags::TRANSFER,
                v::PipelineStageFlags::TRANSFER,
                v::DependencyFlags::empty(),
                &[b],
                &[],
                &[],
            );
            let r = r.image_subresource(layers(v::ImageAspectFlags::COLOR));
            ctx().device.cmd_copy_buffer_to_image(
                cmd,
                snap.buffer,
                snap.depth.image(),
                v::ImageLayout::GENERAL,
                &[r],
            );
            general(cmd, &snap.depth, false);
            barrier(
                cmd,
                depth.image,
                depth_sync,
                v::ImageLayout::GENERAL,
                v::ImageLayout::GENERAL,
            );
            blit(cmd, motion.image, motion.size, &snap.motion);
        }
        s.id += 1;
        if s.id <= 4 {
            crate::info!(
                "FG TAA snapshot {}: depth {:?} {:?}, MV {:?} {:?}, jitter {:?}",
                s.id,
                depth.format,
                depth.size,
                motion.format,
                motion.size,
                constants.jitter
            );
        }
        s.current = Some(Captured {
            snap,
            hudless: None,
            id: s.id,
            constants,
            at: Instant::now(),
            linear: false,
        });
        Ok(())
    })();
    if let Err(e) = result {
        fail(e);
    }
}
/// After the extra transparent-UI composite, before the real composite overwrites it.
/// Key by display image handle, not recycled guest VA.
pub fn capture_hudless(cmd: v::CommandBuffer, target: crate::gpu::images::Target) {
    if !enabled() {
        return;
    }
    let mut g = INPUTS.lock();
    let Some(s) = g.as_mut() else { note("HUDless deferred", "no TAA inputs yet; retry next composite"); return };
    let Some(mut captured) = s.current.take() else {
        s.ready.remove(&target.image.as_raw());
        note("HUDless deferred", "TAA snapshot unavailable; retry next composite");
        return;
    };
    note("HUDless callback", format_args!("image {:#x}, TAA {}", target.image.as_raw(), captured.id));
    if s.ready
        .get(&target.image.as_raw())
        .is_some_and(|v| v.id == captured.id)
    {
        note("HUDless duplicate", format_args!("image {:#x}, TAA {}", target.image.as_raw(), captured.id));
        return;
    }
    let hud = s
        .huds
        .iter()
        .find(|p| {
            Arc::strong_count(p) == 1
                && (p.desc.width, p.desc.height) == (target.width, target.height)
        })
        .cloned();
    let hud = match hud {
        Some(h) => h,
        None => {
            if s.huds.len() >= 8 {
                fail("pre-UI snapshot pool full");
                return;
            }
            match Image::new(target.width, target.height, COLOR) {
                Ok(h) => {
                    let h = Arc::new(h);
                    s.huds.push(h.clone());
                    h
                }
                Err(e) => {
                    fail(e);
                    return;
                }
            }
        }
    };
    blit(cmd, target.image, (target.width, target.height), &hud);
    captured.hudless = Some(hud);
    captured.linear = matches!(
        target.format,
        v::Format::R8G8B8A8_SRGB | v::Format::B8G8R8A8_SRGB
    );
    if captured.id <= 4 {
        crate::info!(
            "FG snapshot {}: pre-UI target {:#x} {:?} {}x{}, depth/MV {}x{} jitter {:?}",
            captured.id,
            target.key.base,
            target.format,
            target.width,
            target.height,
            captured.snap.depth.desc.width,
            captured.snap.depth.desc.height,
            captured.constants.jitter
        );
    }
    note("paired inputs ready", format_args!("image {:#x}, TAA {}, output {}x{}", target.image.as_raw(), captured.id, target.width, target.height));
    s.ready.insert(target.image.as_raw(), captured);
}
type ContextKey = (u32, u32, u32, u32);
fn context_key(display: (u32, u32), hudless: (u32, u32), depth: (u32, u32), motion: (u32, u32)) -> Result<ContextKey, String> {
    if display.0 == 0 || display.1 == 0 || depth.0 == 0 || depth.1 == 0 || display != hudless || depth != motion {
        return Err(format!("input resolution mismatch: display {display:?}, HUDless {hudless:?}, depth {depth:?}, MV {motion:?}"));
    }
    Ok((display.0, display.1, depth.0, depth.1))
}
fn ensure_context<T>(slot: &mut Option<T>, key: ContextKey, key_of: impl Fn(&T) -> ContextKey,
    create: impl FnOnce(ContextKey) -> Result<T, String>, destroy: impl FnOnce(T)) -> Result<bool, String> {
    if slot.as_ref().is_some_and(|g| key_of(g) == key) { return Ok(false); }
    if let Some(old) = slot.take() { destroy(old); }
    *slot = Some(create(key)?);
    Ok(true)
}
// Diagnostic history keeps exactly the previous dispatch's inputs, not a newly sampled
// guest target. Rotate colour allocations after the fence; no extra per-frame GPU copy.
struct History {
    color: Image,
    hudless: Option<Arc<Image>>,
    id: u64,
}
struct Generator {
    context: usize,
    infinite: bool,
    epoch: u64,
    prepared_motion: Image,
    motion_status: Image,
    key: (u32, u32, u32, u32),
    color: Image,
    output: Image,
    history: Option<History>,
    pool: v::CommandPool,
    cmd: v::CommandBuffer,
    fence: v::Fence,
    last_id: u64,
    last: Instant,
    next: Option<Instant>,
}
static GENERATOR: Mutex<Option<Generator>> = Mutex::new(None);
fn generator(w: u32, h: u32, rw: u32, rh: u32, infinite: bool) -> Result<Generator, String> {
    unsafe {
        let api = api()?;
        let c = ctx();
        let mut context = 0;
        crate::info!("FSR3 context creation attempt: depth/MV {rw}x{rh}, output {w}x{h}, bridge {}", path().display());
        let started = Instant::now();
        let e = (api.create)(
            c.instance.handle(),
            c.pdev,
            c.device.handle(),
            w,
            h,
            rw,
            rh,
            u32::from(infinite),
            &mut context,
        );
        crate::info!("FSR3 context creation returned: stage/code {e}, handle {context:#x}, {:.3}s", started.elapsed().as_secs_f64());
        if e != 0 || context == 0 {
            return Err(format!("native context creation stage/code {e}, handle {context:#x}"));
        }
        let pool = c
            .device
            .create_command_pool(
                &v::CommandPoolCreateInfo::default()
                    .queue_family_index(c.queue_family)
                    .flags(v::CommandPoolCreateFlags::RESET_COMMAND_BUFFER),
                None,
            )
            .map_err(|e| e.to_string())?;
        let cmd = c
            .device
            .allocate_command_buffers(
                &v::CommandBufferAllocateInfo::default()
                    .command_pool(pool)
                    .command_buffer_count(1),
            )
            .map_err(|e| e.to_string())?[0];
        let fence = c
            .device
            .create_fence(&v::FenceCreateInfo::default(), None)
            .map_err(|e| e.to_string())?;
        crate::info!("FSR3 FG context: {rw}x{rh} depth/MV -> {w}x{h}, native/no upscaling, reverse depth. Camera planes/FOV/jitter decoded from per-frame constants; infinite far={infinite}.");
        Ok(Generator {
            context,
            infinite,
            epoch: EPOCH.load(Relaxed),
            prepared_motion: Image::new(rw, rh, v::Format::R16G16_SFLOAT)?,
            motion_status: Image::new(rw, rh, v::Format::R32_UINT)?,
            key: (w, h, rw, rh),
            color: Image::new(w, h, COLOR)?,
            output: Image::new(w, h, COLOR)?,
            history: if input_dumps() { Some(History { color: Image::new(w,h,COLOR)?, hudless: None, id: 0 }) } else { None },
            pool,
            cmd,
            fence,
            last_id: 0,
            last: Instant::now(),
            next: None,
        })
    }
}
impl Generator {
    unsafe fn destroy(self) {
        unsafe {
            let c = ctx();
            (api().unwrap().destroy)(self.context);
            self.prepared_motion.destroy();
            self.motion_status.destroy();
            self.color.destroy();
            self.output.destroy();
            if let Some(history) = self.history { history.color.destroy(); }
            c.device.destroy_command_pool(self.pool, None);
            c.device.destroy_fence(self.fence, None);
        }
    }
}
/// Fixed 60 presents on 60Hz; on high-refresh displays at most twice the real arrival rate.
/// Only presentation sleeps: guest simulation/flip IDs and its clocks are unchanged.
fn period(refresh: f64, real_ms: f64) -> Duration {
    let hz = if refresh < 90.0 {
        refresh.clamp(30.0, 60.0)
    } else {
        (2000.0 / real_ms.clamp(8.0, 100.0)).min(refresh)
    };
    Duration::from_secs_f64(1.0 / hz)
}
fn pace(next: &mut Option<Instant>, period: Duration) {
    let now = Instant::now();
    let due = next.unwrap_or(now);
    if due > now {
        crate::thread::short_sleep(due - now);
    }
    *next = Some(if now.saturating_duration_since(due) > period {
        now + period
    } else {
        due + period
    });
}
fn reset_history(last: u64, current: u64, title_history: bool, epoch_changed: bool,
    real_ms: f64, age: Duration) -> bool {
    !title_history || last == 0 || current != last.saturating_add(1)
        || epoch_changed || real_ms > 150.0 || age > Duration::from_millis(500)
}
/// Before ordinary present, interpolate and present an extra image. Source remains protected
/// by videoout's current-buffer ownership until both presentations finish.
pub fn present_pair(src: &vk::present::Source) -> bool {
    INPUT_AGE.store(0, Relaxed);
    COMPUTE_MS.store(0, Relaxed);
    if !enabled() {
        return false;
    }
    let vk::present::Source::Image {
        image,
        format,
        layout,
        width,
        height,
        hdr,
    } = *src
    else {
        note("present deferred", "buffer source; requires a rendered image; retry next present");
        return false;
    };
    if hdr || layout != v::ImageLayout::GENERAL {
        EPOCH.fetch_add(1, Relaxed);
        note("present unsupported", format_args!("HDR={hdr}, layout {layout:?}, format {format:?}, {width}x{height}"));
        return false;
    }
    let cap = INPUTS
        .lock()
        .as_mut()
        .and_then(|s| s.ready.remove(&image.as_raw()));
    let Some(cap) = cap else {
        EPOCH.fetch_add(1, Relaxed);
        let n = SKIPPED.fetch_add(1, Relaxed) + 1;
        if n <= 4 || n % 600 == 0 {
            let g = INPUTS.lock();
            let pending = g
                .as_ref()
                .map(|s| s.ready.keys().copied().collect::<Vec<_>>())
                .unwrap_or_default();
            drop(g);
            diagnostic_state();
            crate::info!("FG waiting for paired TAA/pre-UI snapshot: display image {:#x}, pending {pending:x?}, skipped {n}",image.as_raw());
        }
        return false;
    };
    let Some(hudless) = cap.hudless.as_ref() else {
        EPOCH.fetch_add(1, Relaxed);
        note("present deferred", "paired snapshot has no HUDless colour; retry next present");
        return false;
    };
    let result = (|| -> Result<bool, String> {
        unsafe {
            let c = ctx();
            let mut guard = GENERATOR.lock();
            let key = context_key((width, height), (hudless.desc.width, hudless.desc.height),
                (cap.snap.depth.desc.width, cap.snap.depth.desc.height),
                (cap.snap.motion.desc.width, cap.snap.motion.desc.height))?;
            note("context inputs accepted", format_args!("image {:#x}, format {format:?}, TAA {}, dimensions {key:?}, existing {:?}", image.as_raw(), cap.id, guard.as_ref().map(|g| g.key)));
            if guard.as_ref().is_some_and(|g| g.infinite != cap.constants.infinite) {
                guard.take().unwrap().destroy();
            }
            if ensure_context(&mut *guard, key, |g| g.key,
                |k| generator(k.0, k.1, k.2, k.3, cap.constants.infinite), |old| old.destroy())? {
                diagnostic_state();
            }
            let g = guard.as_mut().unwrap();
            let now = Instant::now();
            let real_ms = now.duration_since(g.last).as_secs_f64() * 1000.0;
            let epoch = EPOCH.load(Relaxed);
            let reset = reset_history(g.last_id, cap.id, cap.constants.history,
                g.epoch != epoch, real_ms, cap.at.elapsed());
            c.device
                .reset_command_pool(g.pool, v::CommandPoolResetFlags::empty())
                .map_err(|e| e.to_string())?;
            c.device
                .begin_command_buffer(g.cmd, &v::CommandBufferBeginInfo::default())
                .map_err(|e| e.to_string())?;
            blit(g.cmd, image, (width, height), &g.color);
            general(g.cmd, &g.output, true);
            general(g.cmd, &g.prepared_motion, true);
            general(g.cmd, &g.motion_status, true);
            general(g.cmd, &cap.snap.depth, false);
            general(g.cmd, &cap.snap.motion, false);
            general(g.cmd, hudless, false);
            let params = Params {
                color: g.color.desc,
                hudless: hudless.desc,
                depth: cap.snap.depth.desc,
                motion: cap.snap.motion.desc,
                output: g.output.desc,
                prepared_motion: g.prepared_motion.desc,
                motion_status: g.motion_status.desc,
                reprojection: cap.constants.reprojection,
                jitter_x: cap.constants.jitter[0],
                jitter_y: cap.constants.jitter[1],
                near_plane: cap.constants.near,
                far_plane: cap.constants.far,
                fov: cap.constants.fov,
                milliseconds: real_ms.clamp(1.0, 100.0) as f32,
                reset: u32::from(reset),
                linear: u32::from(cap.linear),
            };
            let e = (api()?.dispatch)(g.context, g.cmd, &params);
            let next_n = GENERATED.load(Relaxed) + 1;
            let mut dumps = Vec::new();
            let dump_now = e == 0 && !reset && input_dumps() && DUMP_REQUEST.swap(false, Relaxed);
            if dump_now {
                for (name, img) in [
                    ("depth", &cap.snap.depth),
                    ("motion-raw", &cap.snap.motion),
                    ("motion", &g.prepared_motion),
                    ("motion-status", &g.motion_status),
                    ("hudless", hudless.as_ref()),
                    ("real", &g.color),
                    ("generated", &g.output),
                ] {
                    dumps.push(readback(g.cmd, name, img)?);
                }
                if let Some(history) = g.history.as_ref().filter(|h| h.id != 0) {
                    dumps.push(readback(g.cmd, "previous-real", &history.color)?);
                    if let Some(hudless) = &history.hudless {
                        dumps.push(readback(g.cmd, "previous-hudless", hudless)?);
                    }
                }
            }
            // SDK error paths may already record work. Always submit/wait it before teardown/reuse.
            general(g.cmd, &g.output, false);
            c.device
                .end_command_buffer(g.cmd)
                .map_err(|e| e.to_string())?;
            c.device
                .reset_fences(&[g.fence])
                .map_err(|e| e.to_string())?;
            c.submit(g.cmd, &[], &[], g.fence)?;
            c.device
                .wait_for_fences(&[g.fence], true, 5_000_000_000)
                .map_err(|e| e.to_string())?;
            if e != 0 {
                return Err(format!("native dispatch stage/code {e}"));
            }
            let gen_ms = now.elapsed().as_secs_f64() * 1000.0;
            if !dumps.is_empty() {
                save_readbacks(dumps, next_n, &params, cap.id, g.history.as_ref().map_or(0, |h| h.id), &cap.constants);
            }
            let refresh = crate::displayclock::MODE_MILLIHZ.load(Relaxed) as f64 / 1000.0;
            let tick = period(if refresh > 1.0 { refresh } else { 60.0 }, real_ms);
            if let Some(history) = g.history.as_mut() {
                std::mem::swap(&mut g.color, &mut history.color);
                history.hudless = Some(hudless.clone());
                history.id = cap.id;
            }
            g.last = now;
            g.last_id = cap.id;
            g.epoch = epoch;
            if reset {
                note("history reset", format_args!("TAA {}, arrival {real_ms:.3}ms, input age {:.3}ms; SDK dispatched, real frame only", cap.id, cap.at.elapsed().as_secs_f64()*1000.0));
                g.next = None;
                return Ok(false);
            }
            pace(&mut g.next, tick);
            COMPUTE_MS.store(gen_ms.to_bits(), Relaxed);
            INPUT_AGE.store((cap.at.elapsed().as_secs_f64() * 1000.0).to_bits(), Relaxed);
            PRESENTING.store(true, Relaxed);
            let n = GENERATED.fetch_add(1, Relaxed) + 1;
            if shots() && (n <= 3 || n % 300 == 0 || dump_now) {
                vk::present::shot_next();
            }
            let ok = vk::present::present(vk::present::Source::Image {
                image: g.output.image(),
                format: COLOR,
                layout: v::ImageLayout::GENERAL,
                width,
                height,
                hdr: false,
            });
            PRESENTING.store(false, Relaxed);
            if shots() && (n <= 3 || n % 300 == 0 || dump_now) {
                vk::present::shot_next();
            }
            if n <= 4 || n % 120 == 0 {
                diagnostic_state();
                crate::info!("FG pair #{n}: input {}, compute+completion {gen_ms:.3}ms, input age {:.3}ms, target {:.3}ms, real arrival {real_ms:.3}ms, source {format:?}, generated presented={ok}, missing-input skips {}",cap.id,cap.at.elapsed().as_secs_f64()*1000.0,tick.as_secs_f64()*1000.0,SKIPPED.load(Relaxed));
            }
            pace(&mut g.next, tick);
            INPUT_AGE.store((cap.at.elapsed().as_secs_f64() * 1000.0).to_bits(), Relaxed);
            Ok(true)
        }
    })();
    match result {
        Ok(v) => v,
        Err(e) => {
            fail(e);
            false
        }
    }
}
pub fn request_dump() {
    if enabled() && input_dumps() {
        DUMP_REQUEST.store(true, Relaxed);
    }
}
fn input_dumps() -> bool {
    static ON: OnceLock<bool> = OnceLock::new();
    *ON.get_or_init(|| std::env::var("DS_FG_INPUT_DUMP").ok().as_deref() == Some("1"))
}
fn shots() -> bool {
    static ON: OnceLock<bool> = OnceLock::new();
    *ON.get_or_init(|| std::env::var("DS_FG_SHOTS").ok().as_deref() == Some("1"))
}
#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn bridge_layout() {
        assert_eq!(std::mem::size_of::<ImageDesc>(), 24);
        assert_eq!(std::mem::size_of::<Params>(), 264);
        assert_eq!(std::mem::offset_of!(Params, prepared_motion), 120);
        assert_eq!(std::mem::offset_of!(Params, reprojection), 168);
        assert_eq!(std::mem::offset_of!(Params, jitter_x), 232);
    }
    #[test]
    fn cuts_skips_repeats_and_reenable_reset_history() {
        let age = Duration::from_millis(30);
        assert!(!reset_history(10,11,true,false,33.0,age));
        for (last,current,history,epoch,ms,age) in [
            (0,1,true,false,33.0,age), (10,11,false,false,33.0,age),
            (10,12,true,false,33.0,age), (10,10,true,false,33.0,age),
            (10,9,true,false,33.0,age), (10,11,true,true,33.0,age),
            (10,11,true,false,151.0,age), (10,11,true,false,33.0,Duration::from_millis(501))
        ] { assert!(reset_history(last,current,history,epoch,ms,age)); }
    }
    #[test]
    fn native_and_filtered_composites_reach_context_creation() {
        // The reviewer frame uses 3840x2160 D32S8 + RG16F TAA inputs and the
        // native SDR composite, not the old filtered composite's recorded VA.
        for hash in [0xb72a4daf3c99cae3, 0x4eca1d1844c8d0c1] {
            assert!(composite_hash(hash));
            let key = context_key((3840, 2160), (3840, 2160), (3840, 2160), (3840, 2160)).unwrap();
            let mut context = None;
            let calls = std::cell::Cell::new(0);
            let create = |k| { calls.set(calls.get() + 1); Ok(k) };
            assert!(ensure_context(&mut context, key, |k| *k, create, |_| panic!("first create must not destroy")).unwrap());
            assert!(!ensure_context(&mut context, key, |k| *k, create, |_| panic!("reuse must not destroy")).unwrap());
            assert_eq!(calls.get(), 1);
            // A different render scale retains the display extent but recreates FI.
            let scaled = context_key((3840,2160), (3840,2160), (2560,1440), (2560,1440)).unwrap();
            assert!(ensure_context(&mut context, scaled, |k| *k, create, |old| assert_eq!(old,key)).unwrap());
            assert_eq!(calls.get(), 2);
        }
        assert!(!composite_hash(0x100888d100)); // a VA is not a code identity
        assert!(!composite_hash(0));
    }
    #[test]
    fn ultrawide_render_cap_keeps_display_sized_colour() {
        for (display, expected) in [((7680,2160),(5376,1512)), ((5120,2160),(3584,1512)),
            ((3840,2160),(3840,2160)), ((3456,2160),(3456,2160))] {
            let render = crate::dispmode::shaped_lines(2160, display.0, display.1);
            assert_eq!(render, expected);
            let key = context_key(display, display, render, render).unwrap();
            let mut slot = None;
            assert!(ensure_context(&mut slot, key, |k| *k, Ok, |_| {}).unwrap());
            assert!(!ensure_context(&mut slot, key, |k| *k, Ok, |_| {}).unwrap());
            // A Visual Mode/scale change keeps the colour/output size but changes depth/MV.
            let lower = crate::dispmode::shaped_lines(1080, display.0, display.1);
            let changed = context_key(display, display, lower, lower).unwrap();
            assert_ne!(key, changed);
            assert!(ensure_context(&mut slot, changed, |k| *k, Ok, |old| assert_eq!(old,key)).unwrap());
            assert!(context_key(display, display, lower, render).is_err());
        }
    }

    #[test]
    fn incomplete_inputs_do_not_create_and_failed_creation_is_not_cached() {
        assert!(context_key((3840,2160), (2400,1350), (3840,2160), (3840,2160)).is_err());
        assert!(context_key((3840,2160), (3840,2160), (3840,2160), (2560,1440)).is_err());
        assert!(context_key((0,0), (0,0), (0,0), (0,0)).is_err());
        let key = context_key((3840,2160), (3840,2160), (3840,2160), (3840,2160)).unwrap();
        let mut slot = None;
        assert!(ensure_context(&mut slot, key, |k| *k, |_| Err("native stage/code 401".into()), |_| {}).is_err());
        assert!(slot.is_none());
        assert!(ensure_context(&mut slot, key, |k| *k, Ok, |_| {}).unwrap());
    }
    #[test]
    fn pacing_caps_output_not_simulation() {
        assert!((period(60.0, 21.0).as_secs_f64() * 60.0 - 1.0).abs() < 1e-6);
        assert!((period(120.0, 22.0).as_secs_f64() - 0.011).abs() < 1e-6);
        assert_eq!(period(120.0, 8.0), Duration::from_secs_f64(1.0 / 120.0));
    }
}
