//! NVIDIA Streamline (SL 2.14.16), shared by everything the port takes from it: DLSS Super
//! Resolution (`gpu::upscale`), DLSS Frame Generation (`gpu::framegen`), and Reflex with its PC
//! Latency markers (`gpu::latency`). One `slInit` loads every plugin that is installed in the
//! upscalers folder (sl.interposer.dll and sl.common.dll always; sl.dlss.dll with the player's own
//! nvngx_dlss.dll; sl.dlss_g.dll with the player's own nvngx_dlssg.dll; sl.reflex.dll; sl.pcl.dll),
//! each library checked for NVIDIA's signature first (`upscale::load_lib`).
//!
//! The integration is Streamline's "manual hooking" for Vulkan (ProgrammingGuideManualHooking.md):
//! the port makes its instance and device itself with the plugins' extensions, features and extra
//! queues (`instance_extensions`, `device_needs`), hands them over with slSetVulkanInfo
//! (`device_created`, on a thread of its own: NGX starts there), and from then on calls the few
//! functions Streamline intercepts through Streamline's own entry points (`proxies`: the window
//! surface, the swapchain, its images, acquire and present), so that its plugins see the
//! presentation they hook (DLSS-G's own swapchain and pacing, Reflex's latency-mode swapchain,
//! sl.common's per-frame bookkeeping).
//!
//! Frames are named by the port's own index (`token`, slGetNewFrameToken with an explicit index):
//! the number of the flip that shows the frame (`hle::libs::videoout`), the same for the
//! common constants and resource tags set while the frame's GPU work is recorded, for the
//! Reflex markers and for the present, as DLSS-G requires (ProgrammingGuideDLSS_G.md 8.0).
//!
//! A start loads Streamline only after a separate process loaded these files without dying
//! (`probe_streamline`, `load`): a player's runtime that ended every start ends only that check.
//!
//! The structures below are sl_struct.h, sl_core_types.h, sl_consts.h, sl_helpers_vk.h,
//! sl_dlss.h, sl_dlss_g.h, sl_reflex.h and sl_pcl.h of SDK 2.14.16 in MSVC x64 layout, which
//! repr(C) reproduces (the sizes and offsets are checked in the tests).

use super::upscale::{LoadError, WideNull, check_signature, export, lib_dir, load_lib, names};
use ash::vk;
use parking_lot::Mutex;
use std::ffi::{CStr, CString, c_char, c_void};
use std::path::{Path, PathBuf};
use std::sync::OnceLock;
use std::sync::atomic::{AtomicBool, AtomicU32, Ordering::Relaxed, Ordering::SeqCst};

// ---------------------------------------------------------------------------
// The C interface.

/// sl::StructType (a GUID).
#[repr(C)]
#[derive(Clone, Copy)]
pub struct SlType(pub u32, pub u16, pub u16, pub [u8; 8]);

/// sl::BaseStructure.
#[repr(C)]
#[derive(Clone, Copy)]
pub struct SlBase {
    pub next: *const SlBase,
    pub ty: SlType,
    pub version: usize,
}

impl SlBase {
    pub const fn new(ty: SlType, version: usize) -> Self {
        Self { next: std::ptr::null(), ty, version }
    }
}

// SAFETY: `next` is always null here (no chained structures): the per-frame structures move to
// the recorder thread as plain data.
unsafe impl Send for SlBase {}

pub const SL_PREFERENCES: SlType = SlType(0x1ca10965, 0xbf8e, 0x432b, [0x8d, 0xa1, 0x67, 0x16, 0xd8, 0x79, 0xfb, 0x14]);
pub const SL_RESOURCE: SlType = SlType(0x3a9d70cf, 0x2418, 0x4b72, [0x83, 0x91, 0x13, 0xf8, 0x72, 0x1c, 0x72, 0x61]);
pub const SL_RESOURCE_TAG: SlType = SlType(0x4c6a5aad, 0xb445, 0x496c, [0x87, 0xff, 0x1a, 0xf3, 0x84, 0x5b, 0xe6, 0x53]);
pub const SL_VIEWPORT: SlType = SlType(0x171b6435, 0x9b3c, 0x4fc8, [0x99, 0x94, 0xfb, 0xe5, 0x25, 0x69, 0xaa, 0xa4]);
pub const SL_FEATURE_REQUIREMENTS: SlType = SlType(0x66714097, 0xac6d, 0x4bc6, [0x89, 0x15, 0x1e, 0x0f, 0x55, 0xa6, 0xb6, 0x1f]);
pub const SL_FEATURE_VERSION: SlType = SlType(0x6d5b51f0, 0x076b, 0x486d, [0x99, 0x95, 0x5a, 0x56, 0x10, 0x43, 0xf5, 0xc1]);
pub const SL_ADAPTER_INFO: SlType = SlType(0x0677315f, 0xa746, 0x4492, [0x9f, 0x42, 0xcb, 0x61, 0x42, 0xc9, 0xc3, 0xd4]);
pub const SL_VULKAN_INFO: SlType = SlType(0x0eed6fd5, 0x82cd, 0x43a9, [0xbd, 0xb5, 0x47, 0xa5, 0xba, 0x2f, 0x45, 0xd6]);
pub const SL_CONSTANTS: SlType = SlType(0xdcd35ad7, 0x4e4a, 0x4bad, [0xa9, 0x0c, 0xe0, 0xc4, 0x9e, 0xb2, 0x3a, 0xfe]);
pub const SL_DLSS_OPTIONS: SlType = SlType(0x6ac826e4, 0x4c61, 0x4101, [0xa9, 0x2d, 0x63, 0x8d, 0x42, 0x10, 0x57, 0xb8]);
pub const SL_DLSSG_OPTIONS: SlType = SlType(0xfac5f1cb, 0x2dfd, 0x4f36, [0xa1, 0xe6, 0x3a, 0x9e, 0x86, 0x52, 0x56, 0xc5]);
pub const SL_DLSSG_STATE: SlType = SlType(0xcc8ac8e1, 0xa179, 0x44f5, [0x97, 0xfa, 0xe7, 0x41, 0x12, 0xf9, 0xbc, 0x61]);
pub const SL_REFLEX_OPTIONS: SlType = SlType(0xf03af81a, 0x6d0b, 0x4902, [0xa6, 0x51, 0xc4, 0x96, 0x5e, 0x21, 0x54, 0x34]);
pub const SL_REFLEX_REPORT: SlType = SlType(0x0d569b37, 0xa1c8, 0x4453, [0xbe, 0x4d, 0x40, 0xf4, 0xde, 0x57, 0x95, 0x2b]);
pub const SL_REFLEX_REPORT2: SlType = SlType(0x68bb0632, 0x5e1c, 0x402b, [0x89, 0x9d, 0xb4, 0x9f, 0x63, 0x3c, 0x56, 0xc2]);
pub const SL_REFLEX_STATE: SlType = SlType(0xf0bb5985, 0xdaf9, 0x4728, [0xb2, 0xfd, 0xae, 0x80, 0xa2, 0xbd, 0x79, 0x89]);
pub const SL_PCL_STATE: SlType = SlType(0xcfa32f9b, 0x023c, 0x420e, [0x90, 0x56, 0x68, 0x32, 0xb7, 0x4f, 0x89, 0xb5]);

/// sl::kSDKVersion of 2.14.16.
pub const SL_SDK_VERSION: u64 = (2u64 << 48) | (14u64 << 32) | (16u64 << 16) | 0xfedc;
/// sl::Feature ids.
pub const FEATURE_DLSS: u32 = 0;
pub const FEATURE_REFLEX: u32 = 3;
pub const FEATURE_PCL: u32 = 4;
pub const FEATURE_DLSS_G: u32 = 1000;
pub const SL_OK: i32 = 0;
// PreferenceFlags.
const SL_DISABLE_CL_STATE_TRACKING: u64 = 1 << 0;
const SL_USE_MANUAL_HOOKING: u64 = 1 << 2;
const SL_FRAME_BASED_TAGGING: u64 = 1 << 7;
// BufferType.
pub const SL_DEPTH: u32 = 0;
pub const SL_MOTION_VECTORS: u32 = 1;
pub const SL_HUDLESS: u32 = 2;
pub const SL_SCALING_INPUT_COLOR: u32 = 3;
pub const SL_SCALING_OUTPUT_COLOR: u32 = 4;
pub const SL_BACKBUFFER: u32 = 53;
pub const SL_UI_ALPHA: u32 = 69;
// ResourceLifecycle.
pub const SL_ONLY_VALID_NOW: u32 = 0;
pub const SL_VALID_UNTIL_PRESENT: u32 = 1;
pub const SL_VALID_UNTIL_EVALUATE: u32 = 2;
/// RenderAPI::eVulkan.
const SL_VULKAN: u32 = 2;
// sl::Boolean.
pub const SL_FALSE: u8 = 0;
pub const SL_TRUE: u8 = 1;
pub const SL_INVALID: u8 = 2;
/// FeatureRequirementFlags.
const REQ_VULKAN: u32 = 1 << 2;
const REQ_HWS: u32 = 1 << 4;

/// sl::Preferences.
#[repr(C)]
struct SlPreferences {
    base: SlBase,
    show_console: bool,
    log_level: u32,
    paths_to_plugins: *const *const u16,
    num_paths_to_plugins: u32,
    path_to_logs_and_data: *const u16,
    allocate_callback: *const c_void,
    release_callback: *const c_void,
    log_message_callback: Option<unsafe extern "C" fn(u32, *const c_char)>,
    flags: u64,
    features_to_load: *const u32,
    num_features_to_load: u32,
    application_id: u32,
    engine: u32,
    engine_version: *const c_char,
    project_id: *const c_char,
    render_api: u32,
}

/// sl::Resource (Vulkan: the image, its view, layout and description).
#[repr(C)]
#[derive(Clone, Copy)]
pub struct SlResource {
    pub base: SlBase,
    pub ty: u8,
    pub native: u64,
    pub memory: u64,
    pub view: u64,
    pub state: u32,
    pub width: u32,
    pub height: u32,
    pub native_format: u32,
    pub mip_levels: u32,
    pub array_layers: u32,
    pub gpu_virtual_address: u64,
    pub flags: u32,
    pub usage: u32,
    pub internal_flags: u16,
    pub reserved: u16,
}

impl SlResource {
    /// A 2D image of one level and layer in `layout` while Streamline uses it.
    pub fn image(image: vk::Image, view: vk::ImageView, layout: vk::ImageLayout, format: vk::Format, size: (u32, u32), usage: vk::ImageUsageFlags) -> SlResource {
        use ash::vk::Handle;
        SlResource {
            base: SlBase::new(SL_RESOURCE, 1),
            ty: 0,
            native: image.as_raw(),
            memory: 0,
            view: view.as_raw(),
            state: layout.as_raw() as u32,
            width: size.0,
            height: size.1,
            native_format: format.as_raw() as u32,
            mip_levels: 1,
            array_layers: 1,
            gpu_virtual_address: 0,
            flags: 0,
            usage: usage.as_raw(),
            internal_flags: 0,
            reserved: 0,
        }
    }
}

#[repr(C)]
#[derive(Clone, Copy, Default, PartialEq, Eq, Debug)]
pub struct SlExtent {
    pub top: u32,
    pub left: u32,
    pub width: u32,
    pub height: u32,
}

/// sl::ResourceTag.
#[repr(C)]
#[derive(Clone, Copy)]
pub struct SlResourceTag {
    pub base: SlBase,
    pub resource: *const SlResource,
    pub ty: u32,
    pub lifecycle: u32,
    pub extent: SlExtent,
}

impl SlResourceTag {
    pub fn new(resource: *const SlResource, ty: u32, lifecycle: u32, extent: SlExtent) -> SlResourceTag {
        SlResourceTag { base: SlBase::new(SL_RESOURCE_TAG, 1), resource, ty, lifecycle, extent }
    }
}

/// sl::ViewportHandle.
#[repr(C)]
#[derive(Clone, Copy)]
pub struct SlViewport {
    pub base: SlBase,
    pub value: u32,
}

/// The port's one viewport (DLSS, DLSS-G and the tags share it).
pub fn viewport() -> SlViewport {
    SlViewport { base: SlBase::new(SL_VIEWPORT, 1), value: 0 }
}

/// sl::FeatureRequirements (version 2).
#[repr(C)]
pub struct SlFeatureRequirements {
    pub base: SlBase,
    pub flags: u32,
    pub max_cpu_threads: u32,
    pub max_viewports: u32,
    pub num_required_tags: u32,
    pub required_tags: *const u32,
    pub os_detected: [u32; 3],
    pub os_required: [u32; 3],
    pub driver_detected: [u32; 3],
    pub driver_required: [u32; 3],
    pub vk_compute_queues: u32,
    pub vk_graphics_queues: u32,
    pub vk_num_device_extensions: u32,
    pub vk_device_extensions: *const *const c_char,
    pub vk_num_instance_extensions: u32,
    pub vk_instance_extensions: *const *const c_char,
    pub vk_num_features12: u32,
    pub vk_features12: *const *const c_char,
    pub vk_num_features13: u32,
    pub vk_features13: *const *const c_char,
    pub vk_optical_flow_queues: u32,
}

/// sl::FeatureVersion.
#[repr(C)]
pub struct SlFeatureVersion {
    pub base: SlBase,
    pub sl: [u32; 3],
    pub ngx: [u32; 3],
}

/// sl::AdapterInfo.
#[repr(C)]
pub struct SlAdapterInfo {
    pub base: SlBase,
    pub luid: *const u8,
    pub luid_size: u32,
    pub vk_physical_device: vk::PhysicalDevice,
}

/// sl::VulkanInfo (version 3).
#[repr(C)]
pub struct SlVulkanInfo {
    pub base: SlBase,
    pub device: vk::Device,
    pub instance: vk::Instance,
    pub physical_device: vk::PhysicalDevice,
    pub compute_queue_index: u32,
    pub compute_queue_family: u32,
    pub graphics_queue_index: u32,
    pub graphics_queue_family: u32,
    pub optical_flow_queue_index: u32,
    pub optical_flow_queue_family: u32,
    pub use_native_optical_flow: bool,
    pub compute_queue_create_flags: u32,
    pub graphics_queue_create_flags: u32,
    pub optical_flow_queue_create_flags: u32,
}

/// A row-major 4x4 matrix (sl::float4x4).
pub type SlMatrix = [[f32; 4]; 4];

pub const IDENTITY: SlMatrix = [[1.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0], [0.0, 0.0, 1.0, 0.0], [0.0, 0.0, 0.0, 1.0]];

/// sl::Constants (version 2).
#[repr(C)]
#[derive(Clone, Copy)]
pub struct SlConstants {
    pub base: SlBase,
    pub camera_view_to_clip: SlMatrix,
    pub clip_to_camera_view: SlMatrix,
    pub clip_to_lens_clip: SlMatrix,
    pub clip_to_prev_clip: SlMatrix,
    pub prev_clip_to_clip: SlMatrix,
    pub jitter_offset: [f32; 2],
    pub mvec_scale: [f32; 2],
    pub camera_pinhole_offset: [f32; 2],
    pub camera_pos: [f32; 3],
    pub camera_up: [f32; 3],
    pub camera_right: [f32; 3],
    pub camera_fwd: [f32; 3],
    pub camera_near: f32,
    pub camera_far: f32,
    pub camera_fov: f32,
    pub camera_aspect_ratio: f32,
    pub motion_vectors_invalid_value: f32,
    pub depth_inverted: u8,
    pub camera_motion_included: u8,
    pub motion_vectors_3d: u8,
    pub reset: u8,
    pub orthographic_projection: u8,
    pub motion_vectors_dilated: u8,
    pub motion_vectors_jittered: u8,
    pub min_relative_linear_depth_object_separation: f32,
}

/// sl::DLSSOptions (version 3).
#[repr(C)]
#[derive(Clone, Copy)]
pub struct SlDlssOptions {
    pub base: SlBase,
    pub mode: u32,
    pub output_width: u32,
    pub output_height: u32,
    pub sharpness: f32,
    pub pre_exposure: f32,
    pub exposure_scale: f32,
    pub color_buffers_hdr: u8,
    pub indicator_invert_x: u8,
    pub indicator_invert_y: u8,
    pub dlaa_preset: u32,
    pub quality_preset: u32,
    pub balanced_preset: u32,
    pub performance_preset: u32,
    pub ultra_performance_preset: u32,
    pub ultra_quality_preset: u32,
    pub use_auto_exposure: u8,
    pub alpha_upscaling_enabled: u8,
}

/// sl::DLSSGMode.
pub const DLSSG_OFF: u32 = 0;
pub const DLSSG_ON: u32 = 1;
/// sl::DLSSGFlags::eRetainResourcesWhenOff.
pub const DLSSG_RETAIN_RESOURCES_WHEN_OFF: u32 = 1 << 3;

/// sl::DLSSGOptions (version 5).
#[repr(C)]
#[derive(Clone, Copy)]
pub struct SlDlssgOptions {
    pub base: SlBase,
    pub mode: u32,
    pub num_frames_to_generate: u32,
    pub flags: u32,
    pub dynamic_res_width: u32,
    pub dynamic_res_height: u32,
    pub num_back_buffers: u32,
    pub mvec_depth_width: u32,
    pub mvec_depth_height: u32,
    pub color_width: u32,
    pub color_height: u32,
    pub color_buffer_format: u32,
    pub mvec_buffer_format: u32,
    pub depth_buffer_format: u32,
    pub hud_less_buffer_format: u32,
    pub ui_buffer_format: u32,
    pub on_error_callback: *const c_void,
    pub b_reserved15: u8,
    pub queue_parallelism_mode: u32,
    pub enable_user_interface_recomposition: u8,
    pub dynamic_target_frame_rate: f32,
}

impl Default for SlDlssgOptions {
    fn default() -> Self {
        SlDlssgOptions {
            base: SlBase::new(SL_DLSSG_OPTIONS, 5),
            mode: DLSSG_OFF,
            num_frames_to_generate: 1,
            flags: 0,
            dynamic_res_width: 0,
            dynamic_res_height: 0,
            num_back_buffers: 0,
            mvec_depth_width: 0,
            mvec_depth_height: 0,
            color_width: 0,
            color_height: 0,
            color_buffer_format: 0,
            mvec_buffer_format: 0,
            depth_buffer_format: 0,
            hud_less_buffer_format: 0,
            ui_buffer_format: 0,
            on_error_callback: std::ptr::null(),
            b_reserved15: SL_INVALID,
            queue_parallelism_mode: 0,
            enable_user_interface_recomposition: SL_FALSE,
            dynamic_target_frame_rate: 0.0,
        }
    }
}

// SAFETY: the error callback pointer is always null here.
unsafe impl Send for SlDlssgOptions {}

/// sl::DLSSGState (version 4).
#[repr(C)]
#[derive(Clone, Copy)]
pub struct SlDlssgState {
    pub base: SlBase,
    pub estimated_vram_usage_in_bytes: u64,
    pub status: u32,
    pub min_width_or_height: u32,
    pub num_frames_actually_presented: u32,
    pub num_frames_to_generate_max: u32,
    pub b_reserved4: u8,
    pub b_is_vsync_support_available: u8,
    pub inputs_processing_completion_fence: *const c_void,
    pub last_present_inputs_processing_completion_fence_value: u64,
    pub b_is_dynamic_mfg_supported: u8,
}

impl Default for SlDlssgState {
    fn default() -> Self {
        SlDlssgState {
            base: SlBase::new(SL_DLSSG_STATE, 4),
            estimated_vram_usage_in_bytes: 0,
            status: 0,
            min_width_or_height: 0,
            num_frames_actually_presented: 0,
            num_frames_to_generate_max: 0,
            b_reserved4: SL_INVALID,
            b_is_vsync_support_available: SL_INVALID,
            inputs_processing_completion_fence: std::ptr::null(),
            last_present_inputs_processing_completion_fence_value: 0,
            b_is_dynamic_mfg_supported: SL_INVALID,
        }
    }
}

/// sl::ReflexOptions (version 1).
#[repr(C)]
#[derive(Clone, Copy)]
pub struct SlReflexOptions {
    pub base: SlBase,
    pub mode: u32,
    pub frame_limit_us: u32,
    pub use_markers_to_optimize: bool,
    pub virtual_key: u16,
    pub id_thread: u32,
}

/// sl::ReflexReport.
#[repr(C)]
#[derive(Clone, Copy)]
pub struct SlReflexReport {
    pub base: SlBase,
    pub frame_id: u64,
    pub input_sample_time: u64,
    pub sim_start_time: u64,
    pub sim_end_time: u64,
    pub render_submit_start_time: u64,
    pub render_submit_end_time: u64,
    pub present_start_time: u64,
    pub present_end_time: u64,
    pub driver_start_time: u64,
    pub driver_end_time: u64,
    pub os_render_queue_start_time: u64,
    pub os_render_queue_end_time: u64,
    pub gpu_render_start_time: u64,
    pub gpu_render_end_time: u64,
    pub gpu_active_render_time_us: u32,
    pub gpu_frame_time_us: u32,
}

/// sl::ReflexReport2.
#[repr(C)]
#[derive(Clone, Copy)]
pub struct SlReflexReport2 {
    pub base: SlBase,
    pub camera_constructed_time: u64,
    pub cross_adapter_copy_time_us: u32,
}

/// sl::ReflexState (version 2).
#[repr(C)]
pub struct SlReflexState {
    pub base: SlBase,
    pub low_latency_available: bool,
    pub latency_report_available: bool,
    pub stats_window_message: u32,
    pub frame_report: [SlReflexReport; 64],
    pub flash_indicator_driver_controlled: bool,
    pub frame_report2: [SlReflexReport2; 64],
}

impl SlReflexState {
    pub fn boxed() -> Box<SlReflexState> {
        // SAFETY: plain data; zero is a valid value of every field.
        let mut s: Box<SlReflexState> = unsafe { Box::new(std::mem::zeroed()) };
        s.base = SlBase::new(SL_REFLEX_STATE, 2);
        for r in s.frame_report.iter_mut() {
            r.base = SlBase::new(SL_REFLEX_REPORT, 1);
        }
        for r in s.frame_report2.iter_mut() {
            r.base = SlBase::new(SL_REFLEX_REPORT2, 1);
        }
        s
    }
}

/// sl::PCLState (version 1).
#[repr(C)]
pub struct SlPclState {
    pub base: SlBase,
    pub stats_window_message: u32,
}

/// sl::Result's names (sl_result.h), for the logs.
pub fn result_name(r: i32) -> &'static str {
    const NAMES: [&str; 41] = [
        "ok",
        "IO error",
        "driver out of date",
        "Windows out of date",
        "Windows' hardware-accelerated GPU scheduling is off",
        "device not created",
        "no supported adapter",
        "adapter not supported",
        "no plugins",
        "Vulkan API error",
        "DXGI API error",
        "D3D API error",
        "NRD API error",
        "NVAPI error",
        "Reflex API error",
        "NGX failed",
        "JSON parsing error",
        "missing proxy",
        "missing resource state",
        "invalid integration",
        "missing input parameter",
        "not initialized",
        "compute failed",
        "init not called",
        "exception handler",
        "invalid parameter",
        "missing constants",
        "duplicated constants",
        "missing or invalid API",
        "common constants missing",
        "unsupported interface",
        "feature missing",
        "feature not supported",
        "feature missing hooks",
        "feature failed to load",
        "feature wrong priority",
        "feature missing dependency",
        "feature manager invalid state",
        "invalid state",
        "out of video memory",
        "unknown",
    ];
    NAMES.get(r.max(0) as usize).copied().unwrap_or("unknown")
}

pub fn feature_name(f: u32) -> &'static str {
    match f {
        FEATURE_DLSS => "DLSS",
        FEATURE_REFLEX => "Reflex",
        FEATURE_PCL => "PC Latency",
        FEATURE_DLSS_G => "DLSS Frame Generation",
        _ => "?",
    }
}

// ---------------------------------------------------------------------------
// The library.

/// A feature's Vulkan needs (slGetFeatureRequirements).
#[derive(Default, Clone, Debug)]
pub struct Reqs {
    pub flags: u32,
    pub instance_ext: Vec<CString>,
    pub device_ext: Vec<CString>,
    pub features12: Vec<String>,
    pub features13: Vec<String>,
    pub graphics_queues: u32,
    pub compute_queues: u32,
    pub optical_flow_queues: u32,
}

pub type FnIsSupported = unsafe extern "C" fn(u32, *const SlAdapterInfo) -> i32;
pub type FnSetVulkanInfo = unsafe extern "C" fn(*const SlVulkanInfo) -> i32;
pub type FnFrameToken = unsafe extern "C" fn(*mut *const c_void, *const u32) -> i32;
pub type FnSetConstants = unsafe extern "C" fn(*const SlConstants, *const c_void, *const SlViewport) -> i32;
pub type FnEvaluate = unsafe extern "C" fn(u32, *const c_void, *const *const SlBase, u32, vk::CommandBuffer) -> i32;
pub type FnFeatureFunction = unsafe extern "C" fn(u32, *const c_char, *mut *const c_void) -> i32;
pub type FnVersion = unsafe extern "C" fn(u32, *mut SlFeatureVersion) -> i32;
pub type FnSetTagForFrame = unsafe extern "C" fn(*const c_void, *const SlViewport, *const SlResourceTag, u32, vk::CommandBuffer) -> i32;
pub type FnSetFeatureLoaded = unsafe extern "C" fn(u32, bool) -> i32;
pub type FnFreeResources = unsafe extern "C" fn(u32, *const SlViewport) -> i32;

pub struct Lib {
    pub is_supported: FnIsSupported,
    pub set_vulkan_info: FnSetVulkanInfo,
    frame_token: FnFrameToken,
    set_constants: FnSetConstants,
    pub evaluate: FnEvaluate,
    feature_function: FnFeatureFunction,
    pub version: FnVersion,
    set_tag_for_frame: FnSetTagForFrame,
    set_feature_loaded: FnSetFeatureLoaded,
    free_resources: FnFreeResources,
    gipa: vk::PFN_vkGetInstanceProcAddr,
    /// The features slInit loaded, with their needs.
    pub features: Vec<(u32, Reqs)>,
}

// SAFETY: plain function pointers into a library that stays loaded.
unsafe impl Send for Lib {}
unsafe impl Sync for Lib {}

impl Lib {
    pub fn reqs(&self, f: u32) -> Option<&Reqs> {
        self.features.iter().find(|(x, _)| *x == f).map(|(_, r)| r)
    }
}

/// Streamline, initialised with the features installed (None: not installed, not on an NVIDIA
/// driver, or unusable: logged once).
pub fn lib() -> Option<&'static Lib> {
    static L: OnceLock<Option<Lib>> = OnceLock::new();
    L.get_or_init(|| {
        if std::env::var("DS_STREAMLINE").is_ok_and(|v| v.trim() == "0") {
            crate::info!("streamline: off (DS_STREAMLINE=0): no DLSS, DLSS Frame Generation or Reflex");
            return None;
        }
        match load() {
            Ok(l) => Some(l),
            Err(LoadError::Missing) => {
                crate::info!("streamline: not installed (sl.interposer.dll and sl.common.dll with at least one of DLSS, DLSS Frame Generation or Reflex in the upscalers folder)");
                None
            }
            Err(LoadError::Failed(e)) => {
                crate::warn!("streamline: unavailable: {e}");
                super::upscale::streamline_failed(&e);
                None
            }
        }
    })
    .as_ref()
}

/// Whether slInit loaded feature `f` (a feature may still be unsupported on the device:
/// `supported`).
pub fn loaded(f: u32) -> bool {
    lib().is_some_and(|l| l.reqs(f).is_some())
}

unsafe extern "C" fn sl_log(ty: u32, msg: *const c_char) {
    if msg.is_null() {
        return;
    }
    let m = unsafe { CStr::from_ptr(msg) }.to_string_lossy();
    let m = m.trim_end();
    // (Reflex's older low-latency library, NvLowLatencyVk.dll, is only its fall-back on drivers
    // without VK_NV_low_latency2: its absence is no fault.)
    let ty = if m.contains("NvLowLatencyVk.dll") { 0 } else { ty };
    match ty {
        // (An error is written out at once: the next thing may be the end of the process.)
        2 => {
            crate::warn!("Streamline error: {m}");
            crate::log::try_flush();
        }
        1 => crate::warn!("Streamline: {m}"),
        _ if verbose_log() => crate::info!("Streamline: {m}"),
        _ => crate::debug!("Streamline: {m}"),
    }
}

/// `DS_SL_LOG=verbose`: Streamline's own verbose log in the port's (diagnosis).
fn verbose_log() -> bool {
    static V: OnceLock<bool> = OnceLock::new();
    *V.get_or_init(|| std::env::var("DS_SL_LOG").is_ok_and(|v| v.trim().eq_ignore_ascii_case("verbose")))
}

/// The system folder (nvapi64.dll, NVIDIA's driver library, is there with every NVIDIA driver).
fn system_dir() -> Option<PathBuf> {
    let mut sys = [0u16; 260];
    let n = unsafe { windows_sys::Win32::System::SystemInformation::GetSystemDirectoryW(sys.as_mut_ptr(), sys.len() as u32) } as usize;
    (n > 0 && n < sys.len()).then(|| PathBuf::from(String::from_utf16_lossy(&sys[..n])))
}

/// The files Streamline loads (`streamline_files_line`).
const SL_FILES: [&str; 9] = ["nvngx_dlss.dll", "nvngx_dlssg.dll", "sl.interposer.dll", "sl.common.dll", "sl.dlss.dll", "sl.dlss_g.dll", "sl.reflex.dll", "sl.pcl.dll", "NvLowLatencyVk.dll"];

/// Streamline's files, NVIDIA's driver library and the port's version, as a line
/// (`streamline-ok.txt` and `streamline-crashed.txt` hold the line they were written for: a check
/// is made again once it differs, a driver update included).
fn streamline_files_line(dir: &Path) -> String {
    let mut v = vec![format!("Regaliant {}", env!("CARGO_PKG_VERSION"))];
    let files = SL_FILES.map(|f| dir.join(f));
    for p in files.iter().cloned().chain(system_dir().map(|s| s.join("nvapi64.dll"))) {
        let m = std::fs::metadata(&p).ok();
        let t = m.as_ref().and_then(|m| m.modified().ok()).and_then(|t| t.duration_since(std::time::UNIX_EPOCH).ok()).map_or(0, |d| d.as_secs());
        let name = p.file_name().map_or(String::new(), |n| n.to_string_lossy().into_owned());
        v.push(format!("{name} {} {t}", m.map_or(0, |m| m.len())));
    }
    v.join("; ")
}

/// Whether process `pid` still runs.
fn process_alive(pid: u32) -> bool {
    use windows_sys::Win32::Foundation::CloseHandle;
    use windows_sys::Win32::System::Threading::{GetExitCodeProcess, OpenProcess, PROCESS_QUERY_LIMITED_INFORMATION};
    unsafe {
        let h = OpenProcess(PROCESS_QUERY_LIMITED_INFORMATION, 0, pid);
        if h.is_null() {
            return false;
        }
        let mut code = 0u32;
        let alive = GetExitCodeProcess(h, &mut code) != 0 && code == 259;
        CloseHandle(h);
        alive
    }
}

/// Streamline loads that ended a process, or checks that did not end (`note_streamline_death`),
/// in a row for the same files: at `STREAMLINE_DEATHS` Streamline is left out. One alone is no
/// proof: a launcher check or preparation step is ended from outside when the launcher closes or
/// a check takes too long, and such an end in the load must not switch it off (the review of
/// 2026-11-24 night); it only makes the separate check run again.
const STREAMLINE_DEATHS: u32 = 2;

/// One more death in Streamline's load for the files of `line` (`streamline-deaths.txt`: the
/// count and the line), the separate check made to run again; at `STREAMLINE_DEATHS` the files'
/// line goes to `streamline-crashed.txt`.
fn note_streamline_death(cache: &Path, line: &str, what: &str) {
    let deaths = cache.join("streamline-deaths.txt");
    let n = std::fs::read_to_string(&deaths)
        .ok()
        .and_then(|t| t.split_once(' ').filter(|(_, l)| l.trim() == line.trim()).and_then(|(n, _)| n.parse::<u32>().ok()))
        .unwrap_or(0)
        + 1;
    let _ = std::fs::remove_file(cache.join("streamline-ok.txt"));
    if n >= STREAMLINE_DEATHS {
        crate::warn!("streamline: {what} ({n} in a row with these files): Streamline (DLSS, DLSS Frame Generation, Reflex) is left out");
        let _ = std::fs::write(cache.join("streamline-crashed.txt"), line.trim());
        let _ = std::fs::remove_file(&deaths);
    } else {
        crate::warn!("streamline: {what}: Streamline's separate check runs again before it is loaded");
        let _ = std::fs::write(&deaths, format!("{n} {}", line.trim()));
    }
}

/// Starts that died while Streamline loaded (2026-11-24 night: a player's every start, the shader
/// preparation's compute step too, ended with an access violation right after XeSS's query, where
/// Streamline loads, with their own nvngx_dlss.dll in the upscalers folder; the launcher said the
/// preparation "crashed (status 0xc0000005)"). Each process leaves `streamline-loading-<pid>.txt`
/// in the cache folder while it loads Streamline. A marker of a process that no longer runs (or
/// one older than five minutes: the load takes seconds) is claimed by renaming it (a death counts
/// once, however many processes start together) and counted (`note_streamline_death`). Returns
/// why Streamline is left out, if it is (`streamline-crashed.txt` written for these files).
fn streamline_crashed_before(dir: &Path) -> Option<String> {
    let cache = crate::paths().cache.clone();
    let crashed = cache.join("streamline-crashed.txt");
    if let Ok(rd) = std::fs::read_dir(&cache) {
        for e in rd.flatten() {
            let name = e.file_name().to_string_lossy().into_owned();
            let Some(pid) = name.strip_prefix("streamline-loading-").and_then(|s| s.strip_suffix(".txt")).and_then(|s| s.parse::<u32>().ok()) else { continue };
            let old = e.metadata().ok().and_then(|m| m.modified().ok()).and_then(|t| t.elapsed().ok()).is_some_and(|a| a.as_secs() > 300);
            if pid == std::process::id() || !(old || !process_alive(pid)) {
                continue;
            }
            let claimed = cache.join(format!("streamline-loading-{pid}.claimed-{}", std::process::id()));
            if std::fs::rename(e.path(), &claimed).is_err() {
                continue;
            }
            let line = std::fs::read_to_string(&claimed).unwrap_or_default();
            let _ = std::fs::remove_file(&claimed);
            note_streamline_death(&cache, &line, &format!("an earlier start (process {pid}) ended while it loaded Streamline"));
        }
    }
    let was = std::fs::read_to_string(&crashed).ok()?;
    if was.trim() == streamline_files_line(dir) {
        Some(format!(
            "Streamline's load ended starts or its separate check with these files, so it is left out (it is tried again when they, the graphics driver or Regaliant's version change, or after deleting {})",
            crashed.display()
        ))
    } else {
        let _ = std::fs::remove_file(&crashed);
        None
    }
}

/// The features to load from `dir`: those whose plugin (and NVIDIA's own model library, which
/// the player provides) is there. DLSS only when the upscalers prepare it (`DS_UPSCALER`), DLSS
/// Frame Generation unless `DS_FRAMEGEN=0`, Reflex and PC Latency unless `DS_REFLEX=0` (DLSS
/// Frame Generation needs Reflex). Each NVIDIA model library must carry NVIDIA's signature.
fn wanted_features(dir: &Path) -> Result<Vec<u32>, String> {
    let has = |f: &str| dir.join(f).is_file();
    let off = |v: &str| std::env::var(v).is_ok_and(|x| x.trim() == "0");
    let mut v = Vec::new();
    if super::upscale::prepares_dlss() && has("sl.dlss.dll") && has("nvngx_dlss.dll") {
        check_signature(&dir.join("nvngx_dlss.dll"), "NVIDIA Corporation")?;
        v.push(FEATURE_DLSS);
    }
    let reflex = !off("DS_REFLEX") && has("sl.reflex.dll") && has("sl.pcl.dll");
    if reflex {
        v.push(FEATURE_REFLEX);
        v.push(FEATURE_PCL);
    }
    if !off("DS_FRAMEGEN") && reflex && has("sl.dlss_g.dll") && has("nvngx_dlssg.dll") {
        check_signature(&dir.join("nvngx_dlssg.dll"), "NVIDIA Corporation")?;
        v.push(FEATURE_DLSS_G);
    }
    Ok(v)
}

fn load() -> Result<Lib, LoadError> {
    let dir = std::env::var_os("DS_SL_DIR").map(PathBuf::from).unwrap_or_else(lib_dir);
    if !dir.join("sl.interposer.dll").is_file() || !dir.join("sl.common.dll").is_file() || wanted_features(&dir)?.is_empty() {
        return Err(LoadError::Missing);
    }
    // (Without NVIDIA's driver Streamline has nothing to run on: its library nvapi64.dll is in
    // the system folder with every NVIDIA driver.)
    if system_dir().is_some_and(|s| !s.join("nvapi64.dll").is_file()) {
        return Err(LoadError::Missing);
    }
    if let Some(why) = streamline_crashed_before(&dir) {
        return Err(why.into());
    }
    let cache = crate::paths().cache.clone();
    let _ = std::fs::create_dir_all(&cache);
    let line = streamline_files_line(&dir);
    // A start loads Streamline itself only once a separate process has loaded these files
    // (`probe_streamline`; `streamline-ok.txt` holds their line): the player's runtime that
    // ended every start ended only that process, and the game starts without Streamline.
    let ok = cache.join("streamline-ok.txt");
    if std::fs::read_to_string(&ok).ok().as_deref().map(str::trim) != Some(line.as_str()) {
        let t0 = std::time::Instant::now();
        match run_probe(&dir) {
            ProbeEnd::Ended(code) => {
                crate::info!("streamline: its separate check loaded these files in {:.2} s (exit {code})", t0.elapsed().as_secs_f32());
                let _ = std::fs::write(&ok, &line);
            }
            // (A fault's NTSTATUS: the load itself crashed, not an end from outside.)
            ProbeEnd::Faulted(code) => {
                let crashed = cache.join("streamline-crashed.txt");
                let _ = std::fs::write(&crashed, &line);
                return Err(format!(
                    "Streamline's separate check of these files crashed (code {code:#x}), so it is left out (it is tried again when they, the graphics driver or Regaliant's version change, or after deleting {})",
                    crashed.display()
                )
                .into());
            }
            ProbeEnd::Other(why) => {
                note_streamline_death(&cache, &line, &format!("Streamline's separate check {why}"));
                return Err(format!("Streamline's separate check {why}: Streamline is left out for this start").into());
            }
        }
    }
    // (Removed whatever the load's outcome; left behind only if the process dies in it.)
    let marker = cache.join(format!("streamline-loading-{}.txt", std::process::id()));
    let _ = std::fs::write(&marker, &line);
    // (Diagnosis: `DS_SL_CRASH_TEST=2` ends the process here, as a crash in the load would; `1`
    // ends the separate check instead.)
    if std::env::var_os("DS_SL_CRASH_TEST").is_some_and(|v| v == "2") {
        crate::warn!("streamline: DS_SL_CRASH_TEST: ending the process while Streamline loads");
        std::process::abort();
    }
    let r = load_from(&dir);
    let _ = std::fs::remove_file(&marker);
    // (Loaded: the deaths in a row end here.)
    if r.is_ok() {
        let _ = std::fs::remove_file(cache.join("streamline-deaths.txt"));
    }
    r
}

/// Exit codes of `dsport --probe-streamline`: loaded, refused without a fault, out of time (its
/// own limit, `PROBE_SECS`).
const PROBE_LOADED: u32 = 0;
const PROBE_REFUSED: u32 = 3;
const PROBE_TIMED_OUT: u32 = 4;
const PROBE_SECS: u64 = 30;

/// How the separate check ended.
enum ProbeEnd {
    /// Loaded, or refused without a fault (`PROBE_LOADED`, `PROBE_REFUSED`).
    Ended(u32),
    /// A fault's code (an NTSTATUS error, 0xC0000000 and up: an access violation, a fail-fast).
    Faulted(u32),
    /// Out of time, ended from outside, not started or not waited for.
    Other(String),
}

/// Runs `dsport --probe-streamline <dir>` at normal priority (a launcher check's lower one is not
/// inherited) and waits up to `PROBE_SECS` and a little more (it ends itself at `PROBE_SECS`).
fn run_probe(dir: &Path) -> ProbeEnd {
    use std::os::windows::process::CommandExt;
    const CREATE_NO_WINDOW: u32 = 0x0800_0000;
    const NORMAL_PRIORITY_CLASS: u32 = 0x0000_0020;
    let exe = match std::env::current_exe() {
        Ok(e) => e,
        Err(e) => return ProbeEnd::Other(format!("could not start ({e})")),
    };
    let mut child = match std::process::Command::new(exe)
        .arg("--probe-streamline")
        .arg(dir)
        .stdin(std::process::Stdio::null())
        .stdout(std::process::Stdio::null())
        .stderr(std::process::Stdio::null())
        .creation_flags(CREATE_NO_WINDOW | NORMAL_PRIORITY_CLASS)
        .spawn()
    {
        Ok(c) => c,
        Err(e) => return ProbeEnd::Other(format!("could not start ({e})")),
    };
    let t0 = std::time::Instant::now();
    loop {
        match child.try_wait() {
            Ok(Some(status)) => {
                let code = status.code().map_or(u32::MAX, |c| c as u32);
                return match code {
                    PROBE_LOADED | PROBE_REFUSED => ProbeEnd::Ended(code),
                    PROBE_TIMED_OUT => ProbeEnd::Other(format!("did not end within {PROBE_SECS} s")),
                    c if c >= 0xC000_0000 && c != u32::MAX => ProbeEnd::Faulted(c),
                    c => ProbeEnd::Other(format!("was ended from outside (code {c:#x})")),
                };
            }
            Ok(None) if t0.elapsed() < std::time::Duration::from_secs(PROBE_SECS + 5) => std::thread::sleep(std::time::Duration::from_millis(20)),
            Ok(None) => {
                let _ = child.kill();
                return ProbeEnd::Other(format!("did not end within {PROBE_SECS} s"));
            }
            Err(e) => return ProbeEnd::Other(format!("could not be waited for ({e})")),
        }
    }
}

/// `dsport --probe-streamline <dir>` (`main`, before anything else of a start): loads Streamline
/// from `dir` as a start does (its library, slInit with every feature installed, their
/// requirements) and ends at once (no library's exit code runs): `PROBE_LOADED`, or
/// `PROBE_REFUSED` when it refused without a fault; at `PROBE_SECS` it ends itself
/// (`PROBE_TIMED_OUT`), so a load that hangs does not outlive the start that waits for it. A
/// fault in it ends only this process (`load`).
pub fn probe_streamline(dir: &Path) -> ! {
    use windows_sys::Win32::System::Threading::{GetCurrentProcess, TerminateProcess};
    std::thread::spawn(|| {
        std::thread::sleep(std::time::Duration::from_secs(PROBE_SECS));
        unsafe { TerminateProcess(GetCurrentProcess(), PROBE_TIMED_OUT) };
    });
    if std::env::var_os("DS_SL_CRASH_TEST").is_some_and(|v| v == "1") {
        std::process::abort();
    }
    let code = if load_from(dir).is_ok() { PROBE_LOADED } else { PROBE_REFUSED };
    unsafe { TerminateProcess(GetCurrentProcess(), code) };
    std::process::exit(code as i32)
}

/// The port's NGX project id (Streamline passes it with the engine type "custom" when no NVIDIA
/// application id is given).
const NGX_PROJECT_ID: &CStr = c"48421953-2aae-4cec-9b59-7e92288edb4a";

fn load_from(dir: &Path) -> Result<Lib, LoadError> {
    let dir = dir.to_path_buf();
    let wanted = wanted_features(&dir)?;
    if wanted.is_empty() {
        return Err(LoadError::Missing);
    }
    let path = dir.join("sl.interposer.dll");
    let h = load_lib(&path, "NVIDIA Corporation")?;
    let f = |name: &CStr| export(h, &path, name);
    // SAFETY: the signatures of sl_core_api.h / sl_helpers_vk.h.
    unsafe {
        let init: unsafe extern "C" fn(*const SlPreferences, u64) -> i32 = std::mem::transmute(f(c"slInit")?);
        let requirements: unsafe extern "C" fn(u32, *mut SlFeatureRequirements) -> i32 = std::mem::transmute(f(c"slGetFeatureRequirements")?);
        let mut lib = Lib {
            is_supported: std::mem::transmute(f(c"slIsFeatureSupported")?),
            set_vulkan_info: std::mem::transmute(f(c"slSetVulkanInfo")?),
            frame_token: std::mem::transmute(f(c"slGetNewFrameToken")?),
            set_constants: std::mem::transmute(f(c"slSetConstants")?),
            evaluate: std::mem::transmute(f(c"slEvaluateFeature")?),
            feature_function: std::mem::transmute(f(c"slGetFeatureFunction")?),
            version: std::mem::transmute(f(c"slGetFeatureVersion")?),
            set_tag_for_frame: std::mem::transmute(f(c"slSetTagForFrame")?),
            set_feature_loaded: std::mem::transmute(f(c"slSetFeatureLoaded")?),
            free_resources: std::mem::transmute(f(c"slFreeResources")?),
            gipa: std::mem::transmute(f(c"vkGetInstanceProcAddr")?),
            features: Vec::new(),
        };
        // slInit: the features installed, the port restores its own command buffer state (it
        // does after any foreign recording), manual hooking, frame-based resource tagging (one
        // frame index per presented frame for the constants, tags and markers), no over-the-air
        // downloads, logs into the port's log.
        let t0 = std::time::Instant::now();
        let dir_w = dir.as_os_str().encode_wide_null();
        let paths = [dir_w.as_ptr()];
        let prefs = SlPreferences {
            base: SlBase::new(SL_PREFERENCES, 1),
            show_console: false,
            log_level: if verbose_log() { 2 } else { 1 },
            paths_to_plugins: paths.as_ptr(),
            num_paths_to_plugins: 1,
            path_to_logs_and_data: std::ptr::null(),
            allocate_callback: std::ptr::null(),
            release_callback: std::ptr::null(),
            log_message_callback: Some(sl_log),
            flags: SL_DISABLE_CL_STATE_TRACKING | SL_USE_MANUAL_HOOKING | SL_FRAME_BASED_TAGGING,
            features_to_load: wanted.as_ptr(),
            num_features_to_load: wanted.len() as u32,
            application_id: 0,
            engine: 0,
            engine_version: c"1.0".as_ptr(),
            project_id: NGX_PROJECT_ID.as_ptr(),
            render_api: SL_VULKAN,
        };
        let r = init(&prefs, SL_SDK_VERSION);
        crate::exception::reclaim_unhandled_filter("slInit");
        if r != SL_OK {
            return Err(format!("slInit failed ({r}: {})", result_name(r)).into());
        }
        let v = |x: [u32; 3]| format!("{}.{}.{}", x[0], x[1], x[2]);
        let strings = |n: u32, p: *const *const c_char| names(n, p).into_iter().map(|c| c.to_string_lossy().into_owned()).collect::<Vec<_>>();
        for &feature in &wanted {
            let mut reqs: SlFeatureRequirements = std::mem::zeroed();
            reqs.base = SlBase::new(SL_FEATURE_REQUIREMENTS, 2);
            let r = requirements(feature, &mut reqs);
            if r != SL_OK {
                crate::info!("streamline: {} not available here ({r}: {})", feature_name(feature), result_name(r));
                continue;
            }
            // (FeatureRequirementFlags::eVulkanSupported.)
            if reqs.flags & REQ_VULKAN == 0 {
                crate::info!("streamline: {} does not support Vulkan here", feature_name(feature));
                continue;
            }
            let q = Reqs {
                flags: reqs.flags,
                instance_ext: names(reqs.vk_num_instance_extensions, reqs.vk_instance_extensions),
                device_ext: names(reqs.vk_num_device_extensions, reqs.vk_device_extensions),
                features12: strings(reqs.vk_num_features12, reqs.vk_features12),
                features13: strings(reqs.vk_num_features13, reqs.vk_features13),
                graphics_queues: reqs.vk_graphics_queues,
                compute_queues: reqs.vk_compute_queues,
                optical_flow_queues: reqs.vk_optical_flow_queues,
            };
            crate::info!(
                "streamline: {} loaded (OS {} needs {}, driver {} needs {}{}): instance extensions {:?}, device extensions {:?}, Vulkan 1.2 features {:?}, 1.3 features {:?}, extra queues: {} graphics, {} compute, {} optical flow",
                feature_name(feature),
                v(reqs.os_detected),
                v(reqs.os_required),
                v(reqs.driver_detected),
                v(reqs.driver_required),
                if reqs.flags & REQ_HWS != 0 { ", hardware-accelerated GPU scheduling" } else { "" },
                q.instance_ext,
                q.device_ext,
                q.features12,
                q.features13,
                q.graphics_queues,
                q.compute_queues,
                q.optical_flow_queues
            );
            lib.features.push((feature, q));
        }
        if lib.features.is_empty() {
            return Err("none of its features can run here".to_string().into());
        }
        crate::info!("streamline: loaded from {} in {:.2} s", dir.display(), t0.elapsed().as_secs_f32());
        Ok(lib)
    }
}

// ---------------------------------------------------------------------------
// The device.

/// What the device must enable for the features supported on it.
#[derive(Default)]
pub struct Needs {
    pub extensions: Vec<CString>,
    pub features: Vec<String>,
    /// Extra queues of the graphics family (Streamline's own, after the port's).
    pub graphics_queues: u32,
    pub compute_queues: u32,
}

/// The features supported on the device (bit per feature: `bit`).
static SUPPORTED: AtomicU32 = AtomicU32::new(0);
/// Why a feature is not supported, for the menu's tooltips and the log (feature, reason).
static UNSUPPORTED: Mutex<Vec<(u32, String)>> = Mutex::new(Vec::new());

fn bit(f: u32) -> u32 {
    match f {
        FEATURE_DLSS => 1,
        FEATURE_REFLEX => 2,
        FEATURE_PCL => 4,
        FEATURE_DLSS_G => 8,
        _ => 0,
    }
}

/// Whether feature `f` is supported on the device (its needs were enabled; known once the
/// device is made).
pub fn supported(f: u32) -> bool {
    SUPPORTED.load(Relaxed) & bit(f) != 0
}

/// Why feature `f` cannot run here (None: it can, or it was never installed).
pub fn unsupported_reason(f: u32) -> Option<String> {
    UNSUPPORTED.lock().iter().find(|(x, _)| *x == f).map(|(_, r)| r.clone())
}

fn unsupported(f: u32, why: String) {
    crate::info!("streamline: {} not used: {why}", feature_name(f));
    UNSUPPORTED.lock().push((f, why));
    if let Some(l) = lib() {
        // (Unhooked: an unsupported plugin takes no part in the swapchain or present.)
        unsafe { (l.set_feature_loaded)(f, false) };
    }
}

/// Instance extensions every loaded feature needs (before vkCreateInstance), among those the
/// loader has (`avail`); a feature whose extension is missing is left out.
pub fn instance_extensions(avail: &[vk::ExtensionProperties]) -> Vec<CString> {
    let Some(l) = lib() else { return Vec::new() };
    let has = |e: &CStr| avail.iter().any(|x| unsafe { CStr::from_ptr(x.extension_name.as_ptr()) } == e);
    let mut out: Vec<CString> = Vec::new();
    for (f, r) in &l.features {
        if let Some(e) = r.instance_ext.iter().find(|e| !has(e)) {
            unsupported(*f, format!("the Vulkan loader lacks the instance extension {}", e.to_string_lossy()));
            continue;
        }
        for e in &r.instance_ext {
            if !out.contains(e) {
                out.push(e.clone());
            }
        }
    }
    out
}

/// What the device needs for the features supported on `pdev` (before vkCreateDevice).
/// `queues_left`: queues of the graphics family left after the port's own.
pub fn device_needs(instance: &ash::Instance, pdev: vk::PhysicalDevice, queues_left: u32) -> Needs {
    let mut n = Needs::default();
    let Some(l) = lib() else { return n };
    let avail = unsafe { instance.enumerate_device_extension_properties(pdev) }.unwrap_or_default();
    let has = |e: &CStr| avail.iter().any(|x| unsafe { CStr::from_ptr(x.extension_name.as_ptr()) } == e);
    let known_bad: Vec<u32> = UNSUPPORTED.lock().iter().map(|(f, _)| *f).collect();
    let mut supported = 0u32;
    let mut left = queues_left;
    // (DLSS Frame Generation after Reflex: it needs it.)
    let mut order: Vec<&(u32, Reqs)> = l.features.iter().collect();
    order.sort_by_key(|(f, _)| *f == FEATURE_DLSS_G);
    for (f, r) in order {
        let f = *f;
        if known_bad.contains(&f) {
            continue;
        }
        if f == FEATURE_DLSS_G && supported & bit(FEATURE_REFLEX) == 0 {
            unsupported(f, "it needs Reflex, which cannot run here".into());
            continue;
        }
        let info = SlAdapterInfo { base: SlBase::new(SL_ADAPTER_INFO, 1), luid: std::ptr::null(), luid_size: 0, vk_physical_device: pdev };
        let res = unsafe { (l.is_supported)(f, &info) };
        if res != SL_OK {
            let why = match (f, res) {
                (FEATURE_DLSS_G, 4) => "Windows' Hardware-accelerated GPU Scheduling is off (Settings > System > Display > Graphics: turn it on and restart)".to_string(),
                (FEATURE_DLSS_G, 6 | 7) => "this GPU does not support it (DLSS Frame Generation needs a GeForce RTX 40 series or newer)".to_string(),
                (_, 2) => "the NVIDIA driver is too old for it".to_string(),
                _ => format!("not supported on this GPU, driver or Windows ({res}: {})", result_name(res)),
            };
            unsupported(f, why);
            continue;
        }
        if let Err(e) = super::upscale::features_supported(instance, pdev, r.features12.iter().chain(&r.features13)) {
            unsupported(f, e);
            continue;
        }
        // The device's extensions. (VK_EXT_buffer_device_address must not be enabled with the
        // 1.2 bufferDeviceAddress feature: the KHR extension in its place, as Streamline's own
        // device creation does. VK_NV_optical_flow is left out: without the native optical
        // flow queue DLSS Frame Generation runs its optical flow in its interop mode, as the
        // manual hooking guide says, and its current models need none.)
        let mut exts = Vec::new();
        let mut missing = None;
        for e in &r.device_ext {
            if e.as_c_str() == c"VK_NV_optical_flow" {
                continue;
            }
            let e = if e.as_c_str() == ash::ext::buffer_device_address::NAME { ash::khr::buffer_device_address::NAME.to_owned() } else { e.clone() };
            if !has(&e) {
                missing = Some(e);
                break;
            }
            exts.push(e);
        }
        if let Some(e) = missing {
            unsupported(f, format!("the device lacks {}", e.to_string_lossy()));
            continue;
        }
        let want_q = r.graphics_queues + r.compute_queues;
        if want_q > left {
            unsupported(f, format!("it needs {want_q} more queues of the graphics family, {left} left"));
            continue;
        }
        left -= want_q;
        for e in exts {
            if !n.extensions.contains(&e) {
                n.extensions.push(e);
            }
        }
        for x in r.features12.iter().chain(&r.features13) {
            if x != "opticalFlow" && !n.features.contains(x) {
                n.features.push(x.clone());
            }
        }
        n.graphics_queues += r.graphics_queues;
        n.compute_queues += r.compute_queues;
        supported |= bit(f);
    }
    SUPPORTED.store(supported, Relaxed);
    SL_GRAPHICS_QUEUES.store(n.graphics_queues, Relaxed);
    n
}

/// The graphics queues Streamline asked for (`device_needs`): its compute queues follow them.
static SL_GRAPHICS_QUEUES: AtomicU32 = AtomicU32::new(0);

/// slSetVulkanInfo is done: Streamline's proxies and features are usable.
static READY: AtomicBool = AtomicBool::new(false);

pub fn ready() -> bool {
    READY.load(SeqCst)
}

/// The functions Streamline intercepts in manual hooking (sl_hooks.h), through its own
/// vkGetInstanceProcAddr / vkGetDeviceProcAddr: the presenter calls these instead of the
/// driver's once Streamline is ready.
#[derive(Clone, Copy)]
pub struct Proxies {
    pub create_surface: vk::PFN_vkCreateWin32SurfaceKHR,
    pub destroy_surface: vk::PFN_vkDestroySurfaceKHR,
    pub create_swapchain: vk::PFN_vkCreateSwapchainKHR,
    pub destroy_swapchain: vk::PFN_vkDestroySwapchainKHR,
    pub get_images: vk::PFN_vkGetSwapchainImagesKHR,
    pub acquire: vk::PFN_vkAcquireNextImageKHR,
    pub present: vk::PFN_vkQueuePresentKHR,
}

static PROXIES: OnceLock<Proxies> = OnceLock::new();

pub fn proxies() -> Option<&'static Proxies> {
    if ready() { PROXIES.get() } else { None }
}

/// Called right after vkCreateDevice: the port's queues of family `qf` are `queues` (Streamline's
/// extra queues follow them in the same family: `device_needs`). Hands the device to Streamline
/// (slSetVulkanInfo, which starts NGX) on a thread of its own; the features are ready once it is
/// done (`ready`).
pub fn device_created(instance: &ash::Instance, pdev: vk::PhysicalDevice, device: &ash::Device, qf: u32, queues: u32, extra_graphics: u32) {
    let Some(l) = lib() else { return };
    if SUPPORTED.load(Relaxed) == 0 {
        return;
    }
    let info = SlVulkanInfo {
        base: SlBase::new(SL_VULKAN_INFO, 3),
        device: device.handle(),
        instance: instance.handle(),
        physical_device: pdev,
        // (Its graphics queues first, then its compute queues: until 2026-11-26 the compute ones
        // were placed after all of its queues, past those the device has, and DLSS-G's pacer
        // crashed in getDeviceQueue as frame generation turned on.)
        compute_queue_index: queues + SL_GRAPHICS_QUEUES.load(Relaxed).min(extra_graphics),
        compute_queue_family: qf,
        graphics_queue_index: queues,
        graphics_queue_family: qf,
        optical_flow_queue_index: 0,
        optical_flow_queue_family: 0,
        use_native_optical_flow: false,
        compute_queue_create_flags: 0,
        graphics_queue_create_flags: 0,
        optical_flow_queue_create_flags: 0,
    };
    crate::info!(
        "streamline: its queues in family {qf}: graphics from index {}, compute from index {} (the port's {queues} first, {extra_graphics} of Streamline's)",
        info.graphics_queue_index,
        info.compute_queue_index
    );
    let spawned = std::thread::Builder::new().name("sl-init".into()).spawn(move || {
        let t0 = std::time::Instant::now();
        let r = unsafe { (l.set_vulkan_info)(&info) };
        crate::exception::reclaim_unhandled_filter("slSetVulkanInfo");
        if r != SL_OK {
            crate::warn!("streamline: slSetVulkanInfo failed ({r}: {}): DLSS, DLSS Frame Generation and Reflex unavailable", result_name(r));
            for f in [FEATURE_DLSS, FEATURE_DLSS_G, FEATURE_REFLEX, FEATURE_PCL] {
                if supported(f) {
                    UNSUPPORTED.lock().push((f, format!("slSetVulkanInfo failed ({r})")));
                }
            }
            SUPPORTED.store(0, Relaxed);
            super::upscale::streamline_failed("slSetVulkanInfo failed");
            return;
        }
        // Streamline's entry points for the functions it intercepts.
        // SAFETY: Streamline's loader functions, valid for this instance and device.
        let proxies = unsafe {
            let gipa = l.gipa;
            let inst = |name: &CStr| gipa(info.instance, name.as_ptr());
            let gdpa: Option<vk::PFN_vkGetDeviceProcAddr> = std::mem::transmute(inst(c"vkGetDeviceProcAddr"));
            let dev = |name: &CStr| gdpa.and_then(|g| g(info.device, name.as_ptr()));
            (|| -> Option<Proxies> {
                Some(Proxies {
                    create_surface: std::mem::transmute(inst(c"vkCreateWin32SurfaceKHR")?),
                    destroy_surface: std::mem::transmute(inst(c"vkDestroySurfaceKHR")?),
                    create_swapchain: std::mem::transmute(dev(c"vkCreateSwapchainKHR")?),
                    destroy_swapchain: std::mem::transmute(dev(c"vkDestroySwapchainKHR")?),
                    get_images: std::mem::transmute(dev(c"vkGetSwapchainImagesKHR")?),
                    acquire: std::mem::transmute(dev(c"vkAcquireNextImageKHR")?),
                    present: std::mem::transmute(dev(c"vkQueuePresentKHR")?),
                })
            })()
        };
        let Some(proxies) = proxies else {
            crate::warn!("streamline: its Vulkan entry points are missing: DLSS, DLSS Frame Generation and Reflex unavailable");
            SUPPORTED.store(0, Relaxed);
            super::upscale::streamline_failed("no Vulkan entry points from Streamline");
            return;
        };
        let _ = PROXIES.set(proxies);
        let mut parts = Vec::new();
        for f in [FEATURE_DLSS, FEATURE_DLSS_G, FEATURE_REFLEX, FEATURE_PCL] {
            if !supported(f) {
                continue;
            }
            let mut ver: SlFeatureVersion = unsafe { std::mem::zeroed() };
            ver.base = SlBase::new(SL_FEATURE_VERSION, 1);
            if unsafe { (l.version)(f, &mut ver) } == SL_OK {
                let ngx = if ver.ngx[0] != 0 { format!(", NGX {}.{}.{}", ver.ngx[0], ver.ngx[1], ver.ngx[2]) } else { String::new() };
                parts.push(format!("{} (Streamline {}.{}.{}{ngx})", feature_name(f), ver.sl[0], ver.sl[1], ver.sl[2]));
            } else {
                parts.push(feature_name(f).to_string());
            }
        }
        // (Ready before the features are told: the first present through Streamline then finds
        // them ready.)
        READY.store(true, SeqCst);
        crate::info!("streamline: ready in {:.2} s: {}", t0.elapsed().as_secs_f32(), parts.join(", "));
        super::latency::streamline_ready();
        super::framegen::streamline_ready();
        super::upscale::streamline_ready();
    });
    if let Err(e) = spawned {
        crate::warn!("streamline: no thread for its start ({e})");
        SUPPORTED.store(0, Relaxed);
    }
}

// ---------------------------------------------------------------------------
// Per frame.

/// A frame token (sl::FrameToken*): Streamline's handle of the port's frame index.
#[derive(Clone, Copy)]
pub struct Token(*const c_void);

// SAFETY: a handle Streamline keeps for the frame index; any thread may pass it back.
unsafe impl Send for Token {}
unsafe impl Sync for Token {}

impl Token {
    /// The sl::FrameToken pointer (C++ takes it by reference).
    pub fn ptr(self) -> *const c_void {
        self.0
    }
}

/// slGetNewFrameToken is not thread safe (sl_core_api.h): one caller at a time.
static TOKEN_LOCK: Mutex<()> = Mutex::new(());

/// The token of frame `frame` (the port's index: the number of the flip that shows it).
pub fn token(frame: u32) -> Option<Token> {
    let l = lib()?;
    if !ready() {
        return None;
    }
    let _g = TOKEN_LOCK.lock();
    let mut t: *const c_void = std::ptr::null();
    let r = unsafe { (l.frame_token)(&mut t, &frame) };
    if r != SL_OK || t.is_null() {
        static N: AtomicU32 = AtomicU32::new(0);
        if N.fetch_add(1, Relaxed) < 4 {
            crate::warn!("streamline: slGetNewFrameToken({frame}) failed ({r}: {})", result_name(r));
        }
        return None;
    }
    Some(Token(t))
}

/// The frame whose common constants were set last (u32::MAX: none).
static CONSTANTS_FRAME: AtomicU32 = AtomicU32::new(u32::MAX);

/// The common constants of `frame`, set once a frame (DLSS-G: "do not set common constants
/// multiple times in a single frame"); true when they are set for it.
pub fn set_constants(frame: u32, tok: Token, c: &SlConstants) -> bool {
    let Some(l) = lib() else { return false };
    if CONSTANTS_FRAME.swap(frame, SeqCst) == frame {
        return true;
    }
    let vp = viewport();
    let r = unsafe { (l.set_constants)(c, tok.0, &vp) };
    if r != SL_OK {
        static N: AtomicU32 = AtomicU32::new(0);
        if N.fetch_add(1, Relaxed) < 4 {
            crate::warn!("streamline: slSetConstants failed ({r}: {})", result_name(r));
        }
        CONSTANTS_FRAME.store(u32::MAX, SeqCst);
        return false;
    }
    true
}

/// Tags `tags` for `frame` (slSetTagForFrame), copies of volatile ones recorded into `cb`.
pub fn set_tags(tok: Token, tags: &[SlResourceTag], cb: vk::CommandBuffer) -> i32 {
    let Some(l) = lib() else { return -1 };
    let vp = viewport();
    unsafe { (l.set_tag_for_frame)(tok.0, &vp, tags.as_ptr(), tags.len() as u32, cb) }
}

/// slEvaluateFeature of `feature` for the frame, inputs as given, recorded into `cb`.
pub fn evaluate(feature: u32, tok: Token, inputs: &[*const SlBase], cb: vk::CommandBuffer) -> i32 {
    let Some(l) = lib() else { return -1 };
    unsafe { (l.evaluate)(feature, tok.0, inputs.as_ptr(), inputs.len() as u32, cb) }
}

/// A feature's own function (slGetFeatureFunction; None: unavailable).
pub fn feature_fn(feature: u32, name: &CStr) -> Option<*const c_void> {
    let l = lib()?;
    let mut p: *const c_void = std::ptr::null();
    let r = unsafe { (l.feature_function)(feature, name.as_ptr(), &mut p) };
    if r != SL_OK || p.is_null() {
        crate::warn!("streamline: no {} from {} ({r}: {})", name.to_string_lossy(), feature_name(feature), result_name(r));
        return None;
    }
    Some(p)
}

/// slSetFeatureLoaded: a loaded feature takes part in the swapchain's creation and the presents
/// (DLSS-G: its own swapchain), an unloaded one is unhooked.
pub fn set_feature_loaded(feature: u32, on: bool) -> i32 {
    let Some(l) = lib() else { return -1 };
    unsafe { (l.set_feature_loaded)(feature, on) }
}

/// slFreeResources of a feature for the port's viewport.
pub fn free_resources(feature: u32) -> i32 {
    let Some(l) = lib() else { return -1 };
    let vp = viewport();
    unsafe { (l.free_resources)(feature, &vp) }
}

/// Status line.
pub fn status_line() -> String {
    let Some(l) = lib() else { return String::new() };
    let loaded: Vec<&str> = l.features.iter().map(|(f, _)| feature_name(*f)).collect();
    let on: Vec<&str> = [FEATURE_DLSS, FEATURE_DLSS_G, FEATURE_REFLEX, FEATURE_PCL].into_iter().filter(|f| supported(*f)).map(feature_name).collect();
    format!("    streamline: loaded {}; on this device {}; {}\n", loaded.join(", "), on.join(", "), if ready() { "ready" } else { "not ready" })
}

/// Streamline's structures against sl_*.h of SDK 2.14.16 (MSVC x64), checked as the port builds:
/// a layout that differs from the headers fails the build rather than misleading the library.
const _: () = {
    use std::mem::{offset_of, size_of};
    assert!(size_of::<SlBase>() == 32);
    assert!(size_of::<SlPreferences>() == 144);
    assert!(offset_of!(SlPreferences, flags) == 88);
    assert!(offset_of!(SlPreferences, render_api) == 136);
    assert!(size_of::<SlResource>() == 112);
    assert!(offset_of!(SlResource, native) == 40);
    assert!(offset_of!(SlResource, state) == 64);
    assert!(offset_of!(SlResource, gpu_virtual_address) == 88);
    assert!(size_of::<SlResourceTag>() == 64);
    assert!(offset_of!(SlResourceTag, extent) == 48);
    assert!(size_of::<SlViewport>() == 40);
    assert!(size_of::<SlFeatureRequirements>() == 184);
    assert!(offset_of!(SlFeatureRequirements, vk_device_extensions) == 120);
    assert!(offset_of!(SlFeatureRequirements, vk_optical_flow_queues) == 176);
    assert!(size_of::<SlAdapterInfo>() == 56);
    assert!(size_of::<SlVulkanInfo>() == 96);
    assert!(offset_of!(SlVulkanInfo, use_native_optical_flow) == 80);
    assert!(size_of::<SlConstants>() == 456);
    assert!(offset_of!(SlConstants, jitter_offset) == 352);
    assert!(offset_of!(SlConstants, depth_inverted) == 444);
    assert!(offset_of!(SlConstants, min_relative_linear_depth_object_separation) == 452);
    assert!(size_of::<SlDlssOptions>() == 88);
    assert!(offset_of!(SlDlssOptions, dlaa_preset) == 60);
    assert!(offset_of!(SlDlssOptions, use_auto_exposure) == 84);
    assert!(size_of::<SlFeatureVersion>() == 56);
    assert!(offset_of!(SlDlssgOptions, mode) == 32);
    assert!(offset_of!(SlDlssgOptions, ui_buffer_format) == 88);
    assert!(offset_of!(SlDlssgOptions, on_error_callback) == 96);
    assert!(offset_of!(SlDlssgOptions, b_reserved15) == 104);
    assert!(offset_of!(SlDlssgOptions, queue_parallelism_mode) == 108);
    assert!(offset_of!(SlDlssgOptions, enable_user_interface_recomposition) == 112);
    assert!(offset_of!(SlDlssgOptions, dynamic_target_frame_rate) == 116);
    assert!(size_of::<SlDlssgOptions>() == 120);
    assert!(offset_of!(SlDlssgState, status) == 40);
    assert!(offset_of!(SlDlssgState, num_frames_to_generate_max) == 52);
    assert!(offset_of!(SlDlssgState, b_is_vsync_support_available) == 57);
    assert!(offset_of!(SlDlssgState, inputs_processing_completion_fence) == 64);
    assert!(offset_of!(SlDlssgState, b_is_dynamic_mfg_supported) == 80);
    assert!(size_of::<SlDlssgState>() == 88);
    assert!(size_of::<SlReflexOptions>() == 48);
    assert!(offset_of!(SlReflexOptions, virtual_key) == 42);
    assert!(size_of::<SlReflexReport>() == 152);
    assert!(size_of::<SlReflexReport2>() == 48);
    assert!(offset_of!(SlReflexState, stats_window_message) == 36);
    assert!(offset_of!(SlReflexState, frame_report) == 40);
    assert!(offset_of!(SlReflexState, flash_indicator_driver_controlled) == 40 + 64 * 152);
    assert!(size_of::<SlPclState>() == 40);
    assert!(SL_SDK_VERSION == 0x0002_000e_0010_fedc);
};

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn result_names() {
        assert_eq!(result_name(0), "ok");
        assert_eq!(result_name(4), "Windows' hardware-accelerated GPU scheduling is off");
        assert_eq!(result_name(39), "out of video memory");
        assert_eq!(result_name(1000), "unknown");
    }
}
