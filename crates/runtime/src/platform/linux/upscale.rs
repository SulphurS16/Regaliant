//! Optional vendor upscalers are disabled on Linux; the original TAA draw remains active.
use ash::vk;
use std::ffi::CString;
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum Kind {
    Off = 0,
    Xess = 1,
    Dlss = 2,
    Fsr = 3,
}

pub struct Tex {
    pub image: vk::Image,
    pub view: vk::ImageView,
    pub range: vk::ImageSubresourceRange,
    pub format: vk::Format,
    /// The image's own size (the render size may be a part of it).
    pub size: (u32, u32),
    pub usage: vk::ImageUsageFlags,
    /// Its layout while the upscaler runs.
    pub layout: vk::ImageLayout,
}

/// The inputs of one replacement: the scene colour, depth and motion vectors
/// sampled (at the render size), the output written (at the output size:
/// the same in the native anti-aliasing modes).
pub struct Inputs {
    pub color: Tex,
    pub depth: Tex,
    pub velocity: Tex,
    pub output: Tex,
    /// The render size (the inputs').
    pub in_width: u32,
    pub in_height: u32,
    /// The output size.
    pub width: u32,
    pub height: u32,
    /// The colour is tone-mapped (sRGB-encoded) rather than the HDR scene.
    pub ldr: bool,
    /// The frame's jitter, in pixels of the render size (the title's, `crate::jitter`).
    pub jitter_x: f32,
    pub jitter_y: f32,
}


#[derive(Default)]
pub struct DeviceNeeds { pub extensions: Vec<CString>, pub mutable_descriptor: bool, pub features: Vec<String> }
pub fn choice() -> Kind { Kind::Off }
pub fn set_choice(_v: &str) {}
pub fn active() -> bool { false }
pub fn last_eval_ok() -> bool { false }
pub fn pin_taa(_why: &'static str) {}
pub fn note_sl_present() {}
pub fn offered(_name: &str) -> bool { false }
pub fn choices() -> Vec<(String, String)> { Vec::new() }
pub fn render_scale() -> f32 { 1.0 }
pub fn mip_bias() -> f32 { 0.0 }
pub fn instance_extensions(_avail: &[vk::ExtensionProperties]) -> Vec<CString> { Vec::new() }
pub fn device_needs(_instance: &ash::Instance, _pdev: vk::PhysicalDevice) -> DeviceNeeds { DeviceNeeds::default() }
pub fn apply_features(_f10: &mut vk::PhysicalDeviceFeatures, _f11: &mut vk::PhysicalDeviceVulkan11Features<'_>, _f12: &mut vk::PhysicalDeviceVulkan12Features<'_>, _f13: &mut vk::PhysicalDeviceVulkan13Features<'_>, _n: &DeviceNeeds) {}
pub fn device_created(_instance: &ash::Instance, _pdev: vk::PhysicalDevice, _device: &ash::Device, _qf: u32, _queues: u32) {}
pub fn present_hook() -> Option<vk::PFN_vkQueuePresentKHR> { None }
pub fn record(_cmd: vk::CommandBuffer, _inp: &Inputs) -> bool { false }
pub fn set_jitter_scale(_x: f32, _y: f32) {}

#[path = "../../gpu/upscale_view.rs"]
mod view;
pub use view::{main_view, SMALL_VIEWS};
