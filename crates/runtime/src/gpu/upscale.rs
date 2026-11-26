//! Temporal upscalers in place of the title's TAA: NVIDIA DLSS, Intel XeSS and AMD FSR. The
//! title's TAA (cs 0x100893aa00 on the graphics queue, `gpu::exec`) reads the scene colour
//! (HDR, at the render size), depth, a stencil, its motion vectors (R16G16), its history and a
//! mask, and writes the anti-aliased picture at the output size (it also upscales: 2560x1440 to
//! 3840x2160 in the title's Performance mode). With an upscaler chosen (PC tab, Graphics:
//! Anti-Aliasing), that dispatch is replaced by the upscaler's evaluation with the same colour,
//! depth and motion vectors in and the same image out; the title's history goes unused. The
//! title jitters its projection itself (Halton (2, 3), eight phases): each frame's offset is
//! taken from its jitter function (`crate::jitter`) and handed to the upscaler.
//!
//! The choice is live: every upscaler whose library is installed is prepared when the Vulkan
//! device is made (its extensions and features), its context made at its first use, and a
//! switch resets the history. The libraries are loaded from the `upscalers` folder beside
//! dsport.exe (`DS_UPSCALER_DIR` names another), each after a check of its Authenticode
//! signature and its publisher (WinVerifyTrust: NVIDIA, Intel or AMD; `DS_UPSCALER_UNSIGNED=1`
//! skips it for development copies). The port's GPL-3.0 section 7 permission covers linking
//! them.
//!
//! XeSS: Intel's libxess.dll (Intel Simplified Software License). Its Vulkan path needs the
//! mutable-descriptor-type extension and feature, enabled when the device is made.
//!
//! DLSS: NVIDIA Streamline 2.14 (`gpu::streamline`, which loads it with DLSS Frame Generation
//! and Reflex) and DLSS itself, nvngx_dlss.dll (NVIDIA's RTX SDK license: not shipped, the
//! player's own copy; without it DLSS is not offered): each frame the four images tagged on the
//! slEvaluateFeature call, with the frame's common constants (shared with DLSS Frame
//! Generation: `frame_constants`) under the frame's index, the flip that shows it.
//!
//! FSR: AMD's FSR API (amd_fidelityfx_loader_vk.dll, which loads amd_fidelityfx_upscaler_vk.dll
//! beside it: FSR 4 on the AMD GPUs it supports, else FSR 3.1; MIT).
//!
//! The design and most of this code come from the author's own earlier work.
//!
//! Debug: `DS_UPSCALER=off|xess|dlss|fsr` chooses at start and prepares only that library;
//! `DS_UPSCALE_MV=<x>,<y>` multiplies the motion vectors' scale, `DS_UPSCALE_JITTER=<x>,<y>`
//! the jitter passed to the upscaler, `DS_UP_JITTER=<0-3>` flips its signs.

use ash::vk;
use parking_lot::Mutex;
use std::ffi::{CStr, CString, c_char, c_void};
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicBool, AtomicU8, AtomicU32, AtomicU64, Ordering::Relaxed, Ordering::SeqCst};
use windows_sys::Win32::Foundation::HMODULE;

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum Kind {
    Off = 0,
    Xess = 1,
    Dlss = 2,
    Fsr = 3,
}

impl Kind {
    fn parse(v: &str) -> Kind {
        match v.trim() {
            "xess" => Kind::Xess,
            "dlss" => Kind::Dlss,
            "fsr" => Kind::Fsr,
            _ => Kind::Off,
        }
    }

    fn from_u8(v: u8) -> Kind {
        match v {
            1 => Kind::Xess,
            2 => Kind::Dlss,
            3 => Kind::Fsr,
            _ => Kind::Off,
        }
    }

    fn name(self) -> &'static str {
        match self {
            Kind::Off => "TAA",
            Kind::Xess => "XeSS",
            Kind::Dlss => "DLSS",
            Kind::Fsr => "FSR",
        }
    }
}

/// The upscaler chosen now (`Kind` as u8; u8::MAX until read from the
/// settings).
static CHOICE: AtomicU8 = AtomicU8::new(u8::MAX);
/// The choice changed: the next frame starts a new history.
static SWITCHED: AtomicBool = AtomicBool::new(false);

/// The upscaler chosen now: the PC tab's Anti-Aliasing (DS_UPSCALER at start).
pub fn choice() -> Kind {
    let v = CHOICE.load(SeqCst);
    if v != u8::MAX {
        return Kind::from_u8(v);
    }
    let k = Kind::parse(&std::env::var("DS_UPSCALER").ok().unwrap_or_default());
    CHOICE.store(k as u8, SeqCst);
    k
}

/// The game's TAA for the whole run, whatever the settings choose, and why: the shader
/// preparation's self-play (`prepplay::setup`, before the log and the settings; `set_choice`
/// logs it).
static PINNED_TAA: std::sync::OnceLock<&'static str> = std::sync::OnceLock::new();

pub fn pin_taa(why: &'static str) {
    let _ = PINNED_TAA.set(why);
    CHOICE.store(Kind::Off as u8, SeqCst);
}

/// A new choice from the settings (applied from the next frame). An upscaler that is not ready
/// yet takes over once it is (`now_ready`); until then, and for good if it never is, the game's
/// TAA runs (`active`). The switch starts a new history at the upscaler's first frame.
pub fn set_choice(v: &str) {
    let k = Kind::parse(v);
    if let Some(why) = PINNED_TAA.get() {
        static TOLD: AtomicBool = AtomicBool::new(false);
        if k != Kind::Off && !TOLD.swap(true, Relaxed) {
            crate::info!("upscaler: {} chosen in the settings, the game's TAA for this run ({why})", k.name());
        }
        return;
    }
    // (SeqCst with `note_sl_present`'s: one of the two sees the other's store, so the last line
    // logged for a choice of DLSS is right.)
    if CHOICE.swap(k as u8, SeqCst) != k as u8 {
        SWITCHED.store(true, Relaxed);
        let note = if k == Kind::Off || ready(k) {
            ""
        } else if may_become_ready(k) {
            " (the game's TAA until it is ready)"
        } else {
            " (not available here: the game's TAA stays)"
        };
        crate::info!("upscaler: {} chosen{note}", k.name());
    }
}

/// The device is made (`device_created`): XeSS and FSR are ready by then or not at all.
static DEVICE_MADE: AtomicBool = AtomicBool::new(false);

/// Whether `k`, not ready now, may still become ready in this session: before the device is made
/// any upscaler may; after it only DLSS, while it starts on its own thread.
fn may_become_ready(k: Kind) -> bool {
    if k == Kind::Off || failed_flag(k).load(Relaxed) {
        return false;
    }
    !DEVICE_MADE.load(Relaxed) || (k == Kind::Dlss && DLSS_DEVICE_OK.load(Relaxed))
}

/// `k` just became ready: when it is the one chosen, it replaces the game's TAA from the next
/// frame (the switch of `set_choice` is still pending, so its history starts new).
fn now_ready(k: Kind) {
    if choice() == k && ready(k) {
        crate::info!("upscaler: {} in use from now (the chosen Anti-Aliasing)", k.name());
    }
}

/// Whether this session prepares `k`'s library: DS_UPSCALER names the one
/// (development runs), else each that is installed.
fn prepares(k: Kind) -> bool {
    match std::env::var("DS_UPSCALER") {
        Ok(v) => Kind::parse(&v) == k,
        Err(_) => true,
    }
}

static XESS_READY: AtomicBool = AtomicBool::new(false);
static DLSS_READY: AtomicBool = AtomicBool::new(false);
static FSR_READY: AtomicBool = AtomicBool::new(false);
static XESS_FAILED: AtomicBool = AtomicBool::new(false);
static DLSS_FAILED: AtomicBool = AtomicBool::new(false);
static FSR_FAILED: AtomicBool = AtomicBool::new(false);

fn failed_flag(k: Kind) -> &'static AtomicBool {
    match k {
        Kind::Dlss => &DLSS_FAILED,
        Kind::Fsr => &FSR_FAILED,
        _ => &XESS_FAILED,
    }
}

/// Whether `k` is ready to run in this session (DLSS: once a frame's present went through
/// Streamline, `SL_PRESENTED`).
fn ready(k: Kind) -> bool {
    match k {
        Kind::Off => false,
        Kind::Xess => XESS_READY.load(Relaxed) && !XESS_FAILED.load(Relaxed),
        Kind::Dlss => DLSS_READY.load(SeqCst) && !DLSS_FAILED.load(Relaxed) && SL_PRESENTED.load(SeqCst),
        Kind::Fsr => FSR_READY.load(Relaxed) && !FSR_FAILED.load(Relaxed),
    }
}

/// A frame's present went through Streamline's vkQueuePresentKHR (`present_hook`): DLSS runs only
/// after that. Streamline's per-frame bookkeeping (presentCommon) needs every frame's present; in
/// runs that presented nothing through it (no window: the hidden present went to the driver
/// directly until 2026-11-25), DLSS evaluated anyway warned "presentCommon() was not observed" and
/// in three runs of four ended the process seconds later, an access violation inside the driver
/// at a submission (NGX DLSS 310.9.16, driver 610.88; a window's session ran clean).
static SL_PRESENTED: AtomicBool = AtomicBool::new(false);

/// A present went through Streamline's vkQueuePresentKHR (`vk::present`).
pub fn note_sl_present() {
    if !SL_PRESENTED.swap(true, SeqCst) {
        now_ready(Kind::Dlss);
    }
}

/// Whether an upscaler replaces the title's TAA now (chosen and ready).
pub fn active() -> bool {
    ready(choice())
}

/// The scene's TAA: the largest the title ran within `MAIN_VIEW_HOLD` flips (width, height, the
/// last flip it ran in). The title also runs its TAA on small views (the equipment menu's
/// character preview at 640x360, in turn with the scene's); those keep the title's own result.
/// Until 2026-11-24 the upscaler followed both: its output image and its feature were remade at
/// every change, a frame or two apart, and the images were never freed (issue #4).
static MAIN_VIEW: parking_lot::Mutex<(u32, u32, u64)> = parking_lot::Mutex::new((0, 0, 0));
const MAIN_VIEW_HOLD: u64 = 120;
/// The title's TAA runs left to it as a smaller view than the scene's.
pub static SMALL_VIEWS: AtomicU64 = AtomicU64::new(0);

/// Whether the title's TAA at `w`x`h` in flip `frame` is the scene's (`MAIN_VIEW`): the size it
/// last had, a larger one, or any once the scene's has not run for `MAIN_VIEW_HOLD` flips.
pub fn main_view(w: u32, h: u32, frame: u64) -> bool {
    let mut m = MAIN_VIEW.lock();
    let (mw, mh, at) = *m;
    if (w, h) == (mw, mh) || u64::from(w) * u64::from(h) > u64::from(mw) * u64::from(mh) || frame > at + MAIN_VIEW_HOLD {
        if (w, h) != (mw, mh) && mw != 0 {
            crate::info!("upscaler: the scene's view is {w}x{h} now (it was {mw}x{mh}, last at flip {at})");
        }
        *m = (w, h, frame);
        return true;
    }
    drop(m);
    if SMALL_VIEWS.fetch_add(1, Relaxed) == 0 {
        crate::info!("upscaler: the title's TAA at {w}x{h} keeps its own result (a smaller view beside the scene's {mw}x{mh}, such as a menu's character preview)");
    }
    false
}

/// The last evaluation recorded (at replay) succeeded: the merge that follows it copies its
/// output only then (`gpu::upmerge`; a failed one leaves the title's TAA output as it is).
static EVAL_OK: AtomicBool = AtomicBool::new(false);

/// Whether the upscaler's evaluation just recorded into the command buffer succeeded.
pub fn last_eval_ok() -> bool {
    EVAL_OK.load(Relaxed)
}

/// Whether `name` is ready or will be shortly (DLSS starts on a thread of its own after the
/// device: the menu, made meanwhile on a slower PC, offers it already).
pub fn offered(name: &str) -> bool {
    let k = Kind::parse(name);
    ready(k) || (k == Kind::Dlss && DLSS_DEVICE_OK.load(Relaxed) && !DLSS_FAILED.load(Relaxed))
}

/// The choices the settings offer: the TAA and the upscalers ready here.
pub fn choices() -> Vec<(String, String)> {
    let mut v = vec![("off".to_string(), "TAA (the game's)".to_string())];
    if ready(Kind::Dlss) {
        v.push(("dlss".into(), "DLSS (DLAA)".into()));
    }
    if ready(Kind::Xess) {
        v.push(("xess".into(), "XeSS Native AA".into()));
    }
    if ready(Kind::Fsr) {
        v.push(("fsr".into(), "FSR Native AA".into()));
    }
    v
}

/// The render scale of upscaling done by the port: none (the title's TAA, which the upscalers
/// replace, already scales its render size up to the output).
pub fn render_scale() -> f32 {
    1.0
}

/// The textures' level-of-detail bias while the game renders below the
/// output (the upscalers' guidance: log2 of the render scale).
pub fn mip_bias() -> f32 {
    let s = render_scale();
    if s < 1.0 { s.log2() } else { 0.0 }
}

/// `k` cannot run in this session: logged once, the game's TAA stays for it.
fn fail(k: Kind, msg: impl std::fmt::Display) {
    if !failed_flag(k).swap(true, Relaxed) {
        crate::warn!("upscaler: {} unavailable: {msg}", k.name());
    }
}

// ---------------------------------------------------------------------------
// Loading.

pub(super) trait WideNull {
    fn encode_wide_null(&self) -> Vec<u16>;
}

impl WideNull for std::ffi::OsStr {
    fn encode_wide_null(&self) -> Vec<u16> {
        use std::os::windows::ffi::OsStrExt;
        self.encode_wide().chain(std::iter::once(0)).collect()
    }
}

/// The upscalers' folder: `DS_UPSCALER_DIR`, else `upscalers` beside dsport.exe (a
/// development checkout's: local-data/upscalers; `dsconfig::paths`).
pub(super) fn lib_dir() -> PathBuf {
    if let Some(d) = std::env::var_os("DS_UPSCALER_DIR") {
        return PathBuf::from(d);
    }
    crate::paths().upscalers()
}

/// Checks that `path` carries a valid Authenticode signature whose signer's
/// name starts with `publisher` (no revocation check, nothing fetched from
/// the network). DS_UPSCALER_UNSIGNED=1 skips it (development builds).
pub(super) fn check_signature(path: &Path, publisher: &str) -> Result<(), String> {
    use windows_sys::Win32::Security::Cryptography::{CERT_NAME_SIMPLE_DISPLAY_TYPE, CertGetNameStringW};
    use windows_sys::Win32::Security::WinTrust::*;
    if std::env::var("DS_UPSCALER_UNSIGNED").is_ok_and(|v| v == "1") {
        return Ok(());
    }
    let wide = path.as_os_str().encode_wide_null();
    // SAFETY: plain structures for WinVerifyTrust, alive until its state is
    // closed; the signer's certificate is read before that.
    unsafe {
        let mut file: WINTRUST_FILE_INFO = std::mem::zeroed();
        file.cbStruct = std::mem::size_of::<WINTRUST_FILE_INFO>() as u32;
        file.pcwszFilePath = wide.as_ptr();
        let mut data: WINTRUST_DATA = std::mem::zeroed();
        data.cbStruct = std::mem::size_of::<WINTRUST_DATA>() as u32;
        data.dwUIChoice = WTD_UI_NONE;
        data.fdwRevocationChecks = WTD_REVOKE_NONE;
        data.dwUnionChoice = WTD_CHOICE_FILE;
        data.Anonymous.pFile = &mut file;
        data.dwStateAction = WTD_STATEACTION_VERIFY;
        data.dwProvFlags = WTD_CACHE_ONLY_URL_RETRIEVAL;
        let mut action = WINTRUST_ACTION_GENERIC_VERIFY_V2;
        let r = WinVerifyTrust(std::ptr::null_mut(), &mut action, &mut data as *mut WINTRUST_DATA as *mut c_void);
        // (The signer: the first certificate of the first signer's chain.)
        let mut signer = String::new();
        if r == 0 {
            let prov = WTHelperProvDataFromStateData(data.hWVTStateData);
            let sgnr = if prov.is_null() { std::ptr::null_mut() } else { WTHelperGetProvSignerFromChain(prov, 0, 0, 0) };
            if !sgnr.is_null() && (*sgnr).csCertChain > 0 && !(*sgnr).pasCertChain.is_null() {
                let cert = (*(*sgnr).pasCertChain).pCert;
                if !cert.is_null() {
                    let mut buf = [0u16; 256];
                    let n = CertGetNameStringW(cert, CERT_NAME_SIMPLE_DISPLAY_TYPE, 0, std::ptr::null(), buf.as_mut_ptr(), buf.len() as u32);
                    if n > 1 {
                        signer = String::from_utf16_lossy(&buf[..n as usize - 1]);
                    }
                }
            }
        }
        data.dwStateAction = WTD_STATEACTION_CLOSE;
        WinVerifyTrust(std::ptr::null_mut(), &mut action, &mut data as *mut WINTRUST_DATA as *mut c_void);
        if r != 0 {
            return Err(format!("{} has no valid signature (WinVerifyTrust {:#x})", path.display(), r as u32));
        }
        if !signer.starts_with(publisher) {
            return Err(format!("{} is signed by {signer:?}, not {publisher}", path.display()));
        }
        Ok(())
    }
}

/// Why a library was not loaded.
pub(super) enum LoadError {
    /// Not installed (quietly skipped).
    Missing,
    Failed(String),
}

impl From<String> for LoadError {
    fn from(s: String) -> Self {
        LoadError::Failed(s)
    }
}

/// Loads a library signed by `publisher` (its dependencies from its own
/// folder).
pub(super) fn load_lib(path: &Path, publisher: &str) -> Result<HMODULE, LoadError> {
    use windows_sys::Win32::System::LibraryLoader::{LOAD_WITH_ALTERED_SEARCH_PATH, LoadLibraryExW};
    if !path.is_file() {
        return Err(LoadError::Missing);
    }
    check_signature(path, publisher)?;
    let wide = path.as_os_str().encode_wide_null();
    let h = unsafe { LoadLibraryExW(wide.as_ptr(), std::ptr::null_mut(), LOAD_WITH_ALTERED_SEARCH_PATH) };
    if h.is_null() {
        return Err(LoadError::Failed(format!("loading {} failed", path.display())));
    }
    Ok(h)
}

/// A library's export.
pub(super) fn export(h: HMODULE, path: &Path, name: &CStr) -> Result<*const c_void, String> {
    use windows_sys::Win32::System::LibraryLoader::GetProcAddress;
    match unsafe { GetProcAddress(h, name.as_ptr() as *const u8) } {
        Some(p) => Ok(p as *const c_void),
        None => Err(format!("{} has no {}", path.display(), name.to_string_lossy())),
    }
}

/// C strings out of a library's list (count, array of pointers).
pub(super) unsafe fn names(count: u32, list: *const *const c_char) -> Vec<CString> {
    if list.is_null() {
        return Vec::new();
    }
    (0..count as usize).filter_map(|i| unsafe { list.add(i).read().as_ref() }.map(|p| unsafe { CStr::from_ptr(p) }.to_owned())).collect()
}

// ---------------------------------------------------------------------------
// What the instance and device need.

/// Instance extensions the prepared upscalers need (before
/// vkCreateInstance), among those the loader has (`avail`); an upscaler
/// whose extension is missing is dropped.
pub fn instance_extensions(avail: &[vk::ExtensionProperties]) -> Vec<CString> {
    let has = |e: &CStr| avail.iter().any(|x| unsafe { CStr::from_ptr(x.extension_name.as_ptr()) } == e);
    let mut out: Vec<CString> = Vec::new();
    for (k, list) in [(Kind::Xess, xess_instance_extensions())] {
        if let Some(e) = list.iter().find(|e| !has(e)) {
            fail(k, format!("the Vulkan loader lacks the instance extension {}", e.to_string_lossy()));
            continue;
        }
        for e in list {
            if !out.contains(&e) {
                out.push(e);
            }
        }
    }
    // Streamline's (DLSS, DLSS Frame Generation, Reflex: `gpu::streamline`).
    for e in super::streamline::instance_extensions(avail) {
        if !out.contains(&e) {
            out.push(e);
        }
    }
    out
}

/// What the device must enable for the prepared upscalers.
#[derive(Default)]
pub struct DeviceNeeds {
    pub extensions: Vec<CString>,
    /// VK_EXT_mutable_descriptor_type's feature.
    pub mutable_descriptor: bool,
    /// Vulkan 1.0-1.3 features by name (`apply_features`).
    pub features: Vec<String>,
    /// Queues of the graphics family Streamline asks for after the port's own (`gpu::streamline`).
    pub extra_graphics_queues: u32,
}

static XESS_DEVICE_OK: AtomicBool = AtomicBool::new(false);
static DLSS_DEVICE_OK: AtomicBool = AtomicBool::new(false);
static FSR_DEVICE_OK: AtomicBool = AtomicBool::new(false);

/// Device extensions, features and queues of the upscalers and Streamline's features (DLSS,
/// DLSS Frame Generation, Reflex: `gpu::streamline`) that can run on this device (before
/// vkCreateDevice). `queues_left`: queues of the graphics family past the port's own.
pub fn device_needs(instance: &ash::Instance, pdev: vk::PhysicalDevice, queues_left: u32) -> DeviceNeeds {
    let mut n = DeviceNeeds::default();
    let s = super::streamline::device_needs(instance, pdev, queues_left);
    let dlss = (super::streamline::supported(super::streamline::FEATURE_DLSS) && !DLSS_FAILED.load(Relaxed)).then(|| DeviceNeeds { extensions: Vec::new(), mutable_descriptor: false, features: Vec::new(), extra_graphics_queues: 0 });
    if super::streamline::loaded(super::streamline::FEATURE_DLSS) && dlss.is_none() {
        fail(Kind::Dlss, super::streamline::unsupported_reason(super::streamline::FEATURE_DLSS).unwrap_or_else(|| "not supported here".into()));
    }
    for e in s.extensions {
        if !n.extensions.contains(&e) {
            n.extensions.push(e);
        }
    }
    for f in s.features {
        if !n.features.contains(&f) {
            n.features.push(f);
        }
    }
    n.extra_graphics_queues = s.graphics_queues + s.compute_queues;
    for (k, needs, ok) in [
        (Kind::Xess, xess_device_needs(instance, pdev), &XESS_DEVICE_OK),
        (Kind::Dlss, dlss, &DLSS_DEVICE_OK),
        (Kind::Fsr, fsr_device_needs(instance, pdev), &FSR_DEVICE_OK),
    ] {
        let Some(d) = needs else { continue };
        if failed_flag(k).load(Relaxed) {
            continue;
        }
        for e in d.extensions {
            if !n.extensions.contains(&e) {
                n.extensions.push(e);
            }
        }
        for f in d.features {
            if !n.features.contains(&f) {
                n.features.push(f);
            }
        }
        n.mutable_descriptor |= d.mutable_descriptor;
        ok.store(true, Relaxed);
    }
    n
}

/// A feature the port knows, by its Vulkan name: its field in the 1.0 to
/// 1.3 feature structures.
fn feature_slot<'a>(
    f10: &'a mut vk::PhysicalDeviceFeatures,
    f11: &'a mut vk::PhysicalDeviceVulkan11Features<'_>,
    f12: &'a mut vk::PhysicalDeviceVulkan12Features<'_>,
    f13: &'a mut vk::PhysicalDeviceVulkan13Features<'_>,
    name: &str,
) -> Option<&'a mut vk::Bool32> {
    Some(match name {
        "shaderInt16" => &mut f10.shader_int16,
        "shaderInt64" => &mut f10.shader_int64,
        "storageBuffer16BitAccess" => &mut f11.storage_buffer16_bit_access,
        "uniformAndStorageBuffer16BitAccess" => &mut f11.uniform_and_storage_buffer16_bit_access,
        "storagePushConstant16" => &mut f11.storage_push_constant16,
        "shaderDrawParameters" => &mut f11.shader_draw_parameters,
        "samplerYcbcrConversion" => &mut f11.sampler_ycbcr_conversion,
        "variablePointersStorageBuffer" => &mut f11.variable_pointers_storage_buffer,
        "variablePointers" => &mut f11.variable_pointers,
        // Vulkan 1.2 (the names Streamline's plugins may ask for: sl_helpers_vk.h).
        "samplerMirrorClampToEdge" => &mut f12.sampler_mirror_clamp_to_edge,
        "drawIndirectCount" => &mut f12.draw_indirect_count,
        "storageBuffer8BitAccess" => &mut f12.storage_buffer8_bit_access,
        "uniformAndStorageBuffer8BitAccess" => &mut f12.uniform_and_storage_buffer8_bit_access,
        "storagePushConstant8" => &mut f12.storage_push_constant8,
        "shaderBufferInt64Atomics" => &mut f12.shader_buffer_int64_atomics,
        "shaderSharedInt64Atomics" => &mut f12.shader_shared_int64_atomics,
        "shaderFloat16" => &mut f12.shader_float16,
        "shaderInt8" => &mut f12.shader_int8,
        "descriptorIndexing" => &mut f12.descriptor_indexing,
        "shaderInputAttachmentArrayDynamicIndexing" => &mut f12.shader_input_attachment_array_dynamic_indexing,
        "shaderUniformTexelBufferArrayDynamicIndexing" => &mut f12.shader_uniform_texel_buffer_array_dynamic_indexing,
        "shaderStorageTexelBufferArrayDynamicIndexing" => &mut f12.shader_storage_texel_buffer_array_dynamic_indexing,
        "shaderUniformBufferArrayNonUniformIndexing" => &mut f12.shader_uniform_buffer_array_non_uniform_indexing,
        "shaderSampledImageArrayNonUniformIndexing" => &mut f12.shader_sampled_image_array_non_uniform_indexing,
        "shaderStorageBufferArrayNonUniformIndexing" => &mut f12.shader_storage_buffer_array_non_uniform_indexing,
        "shaderStorageImageArrayNonUniformIndexing" => &mut f12.shader_storage_image_array_non_uniform_indexing,
        "shaderInputAttachmentArrayNonUniformIndexing" => &mut f12.shader_input_attachment_array_non_uniform_indexing,
        "shaderUniformTexelBufferArrayNonUniformIndexing" => &mut f12.shader_uniform_texel_buffer_array_non_uniform_indexing,
        "shaderStorageTexelBufferArrayNonUniformIndexing" => &mut f12.shader_storage_texel_buffer_array_non_uniform_indexing,
        "descriptorBindingUniformBufferUpdateAfterBind" => &mut f12.descriptor_binding_uniform_buffer_update_after_bind,
        "descriptorBindingSampledImageUpdateAfterBind" => &mut f12.descriptor_binding_sampled_image_update_after_bind,
        "descriptorBindingStorageImageUpdateAfterBind" => &mut f12.descriptor_binding_storage_image_update_after_bind,
        "descriptorBindingStorageBufferUpdateAfterBind" => &mut f12.descriptor_binding_storage_buffer_update_after_bind,
        "descriptorBindingUniformTexelBufferUpdateAfterBind" => &mut f12.descriptor_binding_uniform_texel_buffer_update_after_bind,
        "descriptorBindingStorageTexelBufferUpdateAfterBind" => &mut f12.descriptor_binding_storage_texel_buffer_update_after_bind,
        "descriptorBindingUpdateUnusedWhilePending" => &mut f12.descriptor_binding_update_unused_while_pending,
        "descriptorBindingPartiallyBound" => &mut f12.descriptor_binding_partially_bound,
        "descriptorBindingVariableDescriptorCount" => &mut f12.descriptor_binding_variable_descriptor_count,
        "runtimeDescriptorArray" => &mut f12.runtime_descriptor_array,
        "samplerFilterMinmax" => &mut f12.sampler_filter_minmax,
        "scalarBlockLayout" => &mut f12.scalar_block_layout,
        "imagelessFramebuffer" => &mut f12.imageless_framebuffer,
        "uniformBufferStandardLayout" => &mut f12.uniform_buffer_standard_layout,
        "shaderSubgroupExtendedTypes" => &mut f12.shader_subgroup_extended_types,
        "separateDepthStencilLayouts" => &mut f12.separate_depth_stencil_layouts,
        "hostQueryReset" => &mut f12.host_query_reset,
        "timelineSemaphore" => &mut f12.timeline_semaphore,
        "bufferDeviceAddress" => &mut f12.buffer_device_address,
        "bufferDeviceAddressCaptureReplay" => &mut f12.buffer_device_address_capture_replay,
        "bufferDeviceAddressMultiDevice" => &mut f12.buffer_device_address_multi_device,
        "vulkanMemoryModel" => &mut f12.vulkan_memory_model,
        "vulkanMemoryModelDeviceScope" => &mut f12.vulkan_memory_model_device_scope,
        "vulkanMemoryModelAvailabilityVisibilityChains" => &mut f12.vulkan_memory_model_availability_visibility_chains,
        "shaderOutputViewportIndex" => &mut f12.shader_output_viewport_index,
        "shaderOutputLayer" => &mut f12.shader_output_layer,
        "subgroupBroadcastDynamicId" => &mut f12.subgroup_broadcast_dynamic_id,
        // Vulkan 1.3.
        "robustImageAccess" => &mut f13.robust_image_access,
        "inlineUniformBlock" => &mut f13.inline_uniform_block,
        "descriptorBindingInlineUniformBlockUpdateAfterBind" => &mut f13.descriptor_binding_inline_uniform_block_update_after_bind,
        "pipelineCreationCacheControl" => &mut f13.pipeline_creation_cache_control,
        "privateData" => &mut f13.private_data,
        "shaderDemoteToHelperInvocation" => &mut f13.shader_demote_to_helper_invocation,
        "shaderTerminateInvocation" => &mut f13.shader_terminate_invocation,
        "subgroupSizeControl" => &mut f13.subgroup_size_control,
        "computeFullSubgroups" => &mut f13.compute_full_subgroups,
        "synchronization2" => &mut f13.synchronization2,
        "textureCompressionASTC_HDR" => &mut f13.texture_compression_astc_hdr,
        "shaderZeroInitializeWorkgroupMemory" => &mut f13.shader_zero_initialize_workgroup_memory,
        "dynamicRendering" => &mut f13.dynamic_rendering,
        "shaderIntegerDotProduct" => &mut f13.shader_integer_dot_product,
        "maintenance4" => &mut f13.maintenance4,
        _ => return None,
    })
}

/// Turns on the features named in `n` (those the port knows).
pub fn apply_features(
    f10: &mut vk::PhysicalDeviceFeatures,
    f11: &mut vk::PhysicalDeviceVulkan11Features<'_>,
    f12: &mut vk::PhysicalDeviceVulkan12Features<'_>,
    f13: &mut vk::PhysicalDeviceVulkan13Features<'_>,
    n: &DeviceNeeds,
) {
    for name in &n.features {
        if let Some(v) = feature_slot(f10, f11, f12, f13, name) {
            *v = vk::TRUE;
        }
    }
}

/// Whether the device has each feature in `names` (None: unknown to the
/// port).
fn device_features(instance: &ash::Instance, pdev: vk::PhysicalDevice, names: &[&str]) -> Vec<Option<bool>> {
    let mut f11 = vk::PhysicalDeviceVulkan11Features::default();
    let mut f12 = vk::PhysicalDeviceVulkan12Features::default();
    let mut f13 = vk::PhysicalDeviceVulkan13Features::default();
    let mut f2 = vk::PhysicalDeviceFeatures2::default().push_next(&mut f11).push_next(&mut f12).push_next(&mut f13);
    unsafe { instance.get_physical_device_features2(pdev, &mut f2) };
    let mut f10 = f2.features;
    names.iter().map(|n| feature_slot(&mut f10, &mut f11, &mut f12, &mut f13, n).map(|v| *v != 0)).collect()
}

/// The features in `names`, each one the device has (Err: one it lacks or
/// the port does not know).
pub(super) fn features_supported<'a>(instance: &ash::Instance, pdev: vk::PhysicalDevice, names: impl IntoIterator<Item = &'a String>) -> Result<(), String> {
    // (opticalFlow is VK_NV_optical_flow's, left out with it: `gpu::streamline::device_needs`.)
    let names: Vec<&String> = names.into_iter().filter(|n| n.as_str() != "opticalFlow").collect();
    let refs: Vec<&str> = names.iter().map(|s| s.as_str()).collect();
    for (name, have) in names.iter().zip(device_features(instance, pdev, &refs)) {
        match have {
            None => return Err(format!("it asks for the Vulkan feature {name}, unknown to the port")),
            Some(false) => return Err(format!("the device lacks the Vulkan feature {name}")),
            Some(true) => {}
        }
    }
    Ok(())
}

/// Called right after vkCreateDevice: `queues` of family `qf` are the
/// port's.
pub fn device_created(instance: &ash::Instance, pdev: vk::PhysicalDevice, device: &ash::Device, qf: u32, queues: u32, extra_graphics: u32) {
    if XESS_DEVICE_OK.load(Relaxed) && xess().is_some() {
        XESS_READY.store(true, Relaxed);
        crate::info!("upscaler: XeSS ready");
        now_ready(Kind::Xess);
    }
    if FSR_DEVICE_OK.load(Relaxed) && ffx().is_some() {
        FSR_READY.store(true, Relaxed);
        crate::info!("upscaler: FSR ready");
        now_ready(Kind::Fsr);
    }
    // Streamline's features (DLSS among them) start on a thread of their own
    // (`streamline_ready`).
    super::streamline::device_created(instance, pdev, device, qf, queues, extra_graphics);
    DEVICE_MADE.store(true, Relaxed);
    let k = choice();
    if k != Kind::Off && !ready(k) && !may_become_ready(k) {
        crate::info!("upscaler: {} (the chosen Anti-Aliasing) is not available here: the game's TAA stays", k.name());
    }
}

// ---------------------------------------------------------------------------
// The per-frame inputs.

/// One image the upscaler reads or writes.
#[derive(Clone, Copy)]
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
    /// The frame's index: the number of the flip that will show it (`gpu::streamline`).
    pub frame: u32,
}

impl Inputs {
    /// Output size over render size (1: native anti-aliasing).
    fn ratio(&self) -> f32 {
        self.width as f32 / self.in_width.max(1) as f32
    }
}

/// Records the chosen upscaler into `cmd` after the game's TAA or the game's
/// scaling draw (layouts: as each `Tex` says - the caller's barriers).
/// Returns false when it cannot run (then the game's own draw runs).
pub fn record(cmd: vk::CommandBuffer, inp: &Inputs) -> bool {
    let reset = SWITCHED.swap(false, Relaxed);
    match choice() {
        Kind::Xess if ready(Kind::Xess) => xess_record(cmd, inp, reset),
        Kind::Dlss if ready(Kind::Dlss) => dlss_record(cmd, inp, reset),
        Kind::Fsr if ready(Kind::Fsr) => fsr_record(cmd, inp, reset),
        _ => false,
    }
}

fn env_pair(name: &str) -> (f32, f32) {
    std::env::var(name)
        .ok()
        .and_then(|v| {
            let (a, b) = v.split_once([',', ';'])?;
            Some((a.trim().parse().ok()?, b.trim().parse().ok()?))
        })
        .unwrap_or((1.0, 1.0))
}

/// The motion vectors' scale multipliers (DS_UPSCALE_MV).
fn mv_scale() -> (f32, f32) {
    static V: std::sync::OnceLock<(f32, f32)> = std::sync::OnceLock::new();
    *V.get_or_init(|| env_pair("DS_UPSCALE_MV"))
}

/// The jitter's scale for the upscaler (multiplied with the title's offset in pixels): live
/// (`set_jitter_scale`, the `upjitter` control command), else `DS_UPSCALE_JITTER` with the signs
/// of `DS_UP_JITTER`, else the default: half of the title's offset with y flipped (its clip
/// space's pixel is two units; y up against the images' y down).
fn jitter_scale() -> (f32, f32) {
    let live = JITTER_SCALE.load(Relaxed);
    if live != 0 {
        return (f32::from_bits((live >> 32) as u32), f32::from_bits(live as u32));
    }
    static V: std::sync::OnceLock<(f32, f32)> = std::sync::OnceLock::new();
    *V.get_or_init(|| {
        let set = std::env::var("DS_UPSCALE_JITTER").is_ok() || std::env::var("DS_UP_JITTER").is_ok();
        if !set {
            return (0.5, -0.5);
        }
        let (x, y) = env_pair("DS_UPSCALE_JITTER");
        match std::env::var("DS_UP_JITTER").ok().and_then(|v| v.trim().parse::<u32>().ok()).unwrap_or(0) {
            1 => (-x, -y),
            2 => (x, -y),
            3 => (-x, y),
            _ => (x, y),
        }
    })
}

static JITTER_SCALE: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);

/// Tests: the jitter's scale from now on (`upjitter <x> <y>`).
pub fn set_jitter_scale(x: f32, y: f32) {
    JITTER_SCALE.store(((x.to_bits() as u64) << 32 | y.to_bits() as u64).max(1), Relaxed);
}

/// The Halton(2, 3) point `i` (from 1), centred: [-0.5, 0.5).
fn halton(mut i: u32, base: u32) -> f32 {
    let (mut f, mut r) = (1.0f32, 0.0f32);
    while i > 0 {
        f /= base as f32;
        r += f * (i % base) as f32;
        i /= base;
    }
    r - 0.5
}


/// A development switch (`<name>=0|1`; None when unset).
fn env_flag(name: &str) -> Option<bool> {
    std::env::var(name).ok().map(|v| v.trim() != "0")
}

/// Logs the first few failures of a call made on the recorder.
fn warn_some(counter: &AtomicU32, what: &str, r: i32) {
    if counter.fetch_add(1, Relaxed) < 4 {
        crate::warn!("upscaler: {what} failed ({r})");
    }
}

// ---------------------------------------------------------------------------
// XeSS's C interface (xess.h, xess_vk.h; structures packed to 8, which is
// their natural layout).

/// xess_vk_image_view_info.
#[repr(C)]
#[derive(Clone, Copy, Default)]
struct XessView {
    image_view: vk::ImageView,
    image: vk::Image,
    range: vk::ImageSubresourceRange,
    format: vk::Format,
    width: u32,
    height: u32,
}

#[repr(C)]
#[derive(Clone, Copy, Default)]
struct Xess2d {
    x: u32,
    y: u32,
}

/// xess_vk_execute_params_t.
#[repr(C)]
#[derive(Clone, Copy, Default)]
struct XessExecute {
    color: XessView,
    velocity: XessView,
    depth: XessView,
    exposure: XessView,
    responsive: XessView,
    output: XessView,
    jitter_x: f32,
    jitter_y: f32,
    exposure_scale: f32,
    reset_history: u32,
    input_width: u32,
    input_height: u32,
    color_base: Xess2d,
    velocity_base: Xess2d,
    depth_base: Xess2d,
    responsive_base: Xess2d,
    reserved0: Xess2d,
    output_base: Xess2d,
}

/// xess_vk_init_params_t.
#[repr(C)]
#[derive(Clone, Copy, Default)]
struct XessInit {
    output: Xess2d,
    quality: i32,
    flags: u32,
    creation_node_mask: u32,
    visible_node_mask: u32,
    temp_buffer_heap: vk::DeviceMemory,
    buffer_heap_offset: u64,
    temp_texture_heap: vk::DeviceMemory,
    texture_heap_offset: u64,
    pipeline_cache: vk::PipelineCache,
}

const XESS_QUALITY_AA: i32 = 106;

type Handle = *mut c_void;
type LogFn = unsafe extern "C" fn(*const c_char, i32);

struct XessLib {
    instance_exts: unsafe extern "C" fn(*mut u32, *mut *const *const c_char, *mut u32) -> i32,
    device_exts: unsafe extern "C" fn(vk::Instance, vk::PhysicalDevice, *mut u32, *mut *const *const c_char) -> i32,
    create: unsafe extern "C" fn(vk::Instance, vk::PhysicalDevice, vk::Device, *mut Handle) -> i32,
    init: unsafe extern "C" fn(Handle, *const XessInit) -> i32,
    execute: unsafe extern "C" fn(Handle, vk::CommandBuffer, *const XessExecute) -> i32,
    velocity_scale: unsafe extern "C" fn(Handle, f32, f32) -> i32,
    logging: unsafe extern "C" fn(Handle, i32, LogFn) -> i32,
}

// SAFETY: plain function pointers into a library that stays loaded.
unsafe impl Send for XessLib {}
unsafe impl Sync for XessLib {}

/// libxess.dll and its functions (None: not prepared, not installed or
/// unusable).
fn xess() -> Option<&'static XessLib> {
    static L: std::sync::OnceLock<Option<XessLib>> = std::sync::OnceLock::new();
    L.get_or_init(|| {
        if !prepares(Kind::Xess) {
            return None;
        }
        match load_xess() {
            Ok(l) => Some(l),
            Err(LoadError::Missing) => {
                crate::info!("upscaler: XeSS not installed");
                None
            }
            Err(LoadError::Failed(e)) => {
                fail(Kind::Xess, e);
                None
            }
        }
    })
    .as_ref()
}

fn load_xess() -> Result<XessLib, LoadError> {
    let path = match std::env::var_os("DS_XESS_DLL") {
        Some(p) => PathBuf::from(p),
        None => lib_dir().join("libxess.dll"),
    };
    let h = load_lib(&path, "Intel Corporation")?;
    let f = |name: &CStr| export(h, &path, name);
    // SAFETY: the signatures of xess.h / xess_vk.h.
    unsafe {
        let lib = XessLib {
            instance_exts: std::mem::transmute(f(c"xessVKGetRequiredInstanceExtensions")?),
            device_exts: std::mem::transmute(f(c"xessVKGetRequiredDeviceExtensions")?),
            create: std::mem::transmute(f(c"xessVKCreateContext")?),
            init: std::mem::transmute(f(c"xessVKInit")?),
            execute: std::mem::transmute(f(c"xessVKExecute")?),
            velocity_scale: std::mem::transmute(f(c"xessSetVelocityScale")?),
            logging: std::mem::transmute(f(c"xessSetLoggingCallback")?),
        };
        crate::info!("upscaler: XeSS loaded from {}", path.display());
        Ok(lib)
    }
}

fn xess_instance_extensions() -> Vec<CString> {
    let Some(x) = xess() else { return Vec::new() };
    let (mut n, mut list, mut api) = (0u32, std::ptr::null(), 0u32);
    let r = unsafe { (x.instance_exts)(&mut n, &mut list, &mut api) };
    if r != 0 {
        fail(Kind::Xess, format!("its instance requirements: error {r}"));
        return Vec::new();
    }
    let v = unsafe { names(n, list) };
    crate::info!("upscaler: XeSS asks for Vulkan {}.{} and instance extensions {:?}", vk::api_version_major(api), vk::api_version_minor(api), v);
    v
}

fn xess_device_needs(instance: &ash::Instance, pdev: vk::PhysicalDevice) -> Option<DeviceNeeds> {
    let x = xess()?;
    if XESS_FAILED.load(Relaxed) {
        return None;
    }
    let (mut n, mut list) = (0u32, std::ptr::null());
    let r = unsafe { (x.device_exts)(instance.handle(), pdev, &mut n, &mut list) };
    if r != 0 {
        fail(Kind::Xess, format!("it cannot run on this device (error {r})"));
        return None;
    }
    let extensions = unsafe { names(n, list) };
    // (The feature its Vulkan path requires.)
    let mut mdt = vk::PhysicalDeviceMutableDescriptorTypeFeaturesEXT::default();
    let mut f2 = vk::PhysicalDeviceFeatures2::default().push_next(&mut mdt);
    unsafe { instance.get_physical_device_features2(pdev, &mut f2) };
    if mdt.mutable_descriptor_type == 0 {
        fail(Kind::Xess, "the device lacks mutableDescriptorType");
        return None;
    }
    let avail = unsafe { instance.enumerate_device_extension_properties(pdev) }.unwrap_or_default();
    if let Some(e) = extensions.iter().find(|e| !avail.iter().any(|a| unsafe { CStr::from_ptr(a.extension_name.as_ptr()) } == e.as_c_str())) {
        fail(Kind::Xess, format!("the device lacks {}", e.to_string_lossy()));
        return None;
    }
    crate::info!("upscaler: XeSS asks for device extensions {extensions:?} and mutableDescriptorType");
    Some(DeviceNeeds { extensions, mutable_descriptor: true, features: Vec::new(), extra_graphics_queues: 0 })
}

unsafe extern "C" fn xess_log(msg: *const c_char, level: i32) {
    if msg.is_null() {
        return;
    }
    let m = unsafe { CStr::from_ptr(msg) }.to_string_lossy();
    if level >= 3 {
        crate::warn!("XeSS: {m}");
    } else {
        crate::info!("XeSS: {m}");
    }
}

/// The context, made at the first use for its sizes.
struct XessState {
    ctx: Handle,
    key: (u32, u32, u32, u32, bool),
    frame: u32,
}

/// XeSS's preset nearest to an output/render ratio (its fixed scalings:
/// 1.0, 1.3, 1.5, 1.7, 2.0, 2.3, 3.0).
fn xess_quality(ratio: f32) -> i32 {
    [(1.0, XESS_QUALITY_AA), (1.3, 105), (1.5, 104), (1.7, 103), (2.0, 102), (2.3, 101), (3.0, 100)]
        .into_iter()
        .min_by(|a, b| (a.0 - ratio).abs().total_cmp(&(b.0 - ratio).abs()))
        .map_or(XESS_QUALITY_AA, |q| q.1)
}

const XESS_INIT_FLAG_LDR_INPUT_COLOR: u32 = 1 << 6;
/// The depth is reversed (near 1, far 0). (Bit 3 asks for a responsive-pixel mask.)
const XESS_INIT_FLAG_INVERTED_DEPTH: u32 = 1 << 1;

// SAFETY: the context handle is used under this mutex and on the recorder.
unsafe impl Send for XessState {}

static XESS_STATE: Mutex<Option<XessState>> = Mutex::new(None);

fn xess_view(t: &Tex, w: u32, h: u32) -> XessView {
    XessView { image_view: t.view, image: t.image, range: t.range, format: t.format, width: w, height: h }
}

fn xess_record(cmd: vk::CommandBuffer, inp: &Inputs, reset: bool) -> bool {
    let Some(x) = xess() else { return false };
    let key = (inp.in_width, inp.in_height, inp.width, inp.height, inp.ldr);
    let mut st = XESS_STATE.lock();
    if st.as_ref().is_some_and(|s| s.key != key) {
        // (Other sizes: a new context; the old one leaks - rare.)
        *st = None;
    }
    if st.is_none() {
        let c = super::vk::ctx();
        let mut h: Handle = std::ptr::null_mut();
        let r = unsafe { (x.create)(c.instance.handle(), c.pdev, c.device.handle(), &mut h) };
        if r != 0 || h.is_null() {
            fail(Kind::Xess, format!("xessVKCreateContext failed ({r})"));
            return false;
        }
        unsafe { (x.logging)(h, 2, xess_log) };
        let quality = xess_quality(inp.ratio());
        let flags = (if inp.ldr { XESS_INIT_FLAG_LDR_INPUT_COLOR } else { 0 }) | XESS_INIT_FLAG_INVERTED_DEPTH;
        let init = XessInit { output: Xess2d { x: inp.width, y: inp.height }, quality, flags, ..Default::default() };
        let r = unsafe { (x.init)(h, &init) };
        if r != 0 {
            fail(Kind::Xess, format!("xessVKInit failed ({r})"));
            return false;
        }
        // The game's motion vectors: screen fraction, current - previous;
        // XeSS's: pixels, current -> previous.
        let (mx, my) = mv_scale();
        unsafe { (x.velocity_scale)(h, -(inp.in_width as f32) * mx, -(inp.in_height as f32) * my) };
        if inp.ldr {
            crate::info!("upscaler: XeSS upscaling {}x{} to {}x{} (preset {quality}) after the game's TAA", inp.in_width, inp.in_height, inp.width, inp.height);
        } else {
            crate::info!("upscaler: XeSS native anti-aliasing at {}x{} after the game's TAA", inp.width, inp.height);
        }
        *st = Some(XessState { ctx: h, key, frame: 0 });
    }
    let s = st.as_mut().unwrap();
    let (jx, jy) = (inp.jitter_x, inp.jitter_y);
    let (sx, sy) = jitter_scale();
    let (w, h) = (inp.in_width, inp.in_height);
    // (XeSS's jitter: the image's shift, as rendered.)
    let p = XessExecute {
        color: xess_view(&inp.color, w, h),
        velocity: xess_view(&inp.velocity, w, h),
        depth: xess_view(&inp.depth, w, h),
        output: xess_view(&inp.output, inp.width, inp.height),
        jitter_x: jx * sx,
        jitter_y: jy * sy,
        exposure_scale: 1.0,
        reset_history: (s.frame == 0 || reset) as u32,
        input_width: w,
        input_height: h,
        ..Default::default()
    };
    s.frame += 1;
    let (h, exec) = (s.ctx as usize, x.execute);
    drop(st);
    unsafe {
        // (Into the command buffer the deferred list is played into, `cb`: a batch fused into an
        // earlier one's buffer at submission plays there, its own buffer never begun; see
        // `dlss_record`.)
        super::vk::defer::external(cmd, Box::new(move |cb| {
            static N: AtomicU32 = AtomicU32::new(0);
            let r = (exec)(h as Handle, cb, &p);
            EVAL_OK.store(r == 0, Relaxed);
            if r != 0 {
                warn_some(&N, "xessVKExecute", r);
            }
        }))
    };
    true
}

// ---------------------------------------------------------------------------
// DLSS through Streamline (`gpu::streamline`: the library, its structures and the frames).

use super::streamline::{self as sl, SlBase, SlConstants, SlDlssOptions, SlExtent, SlResource, SlResourceTag, SlViewport, IDENTITY, SL_FALSE, SL_OK, SL_TRUE};

/// DLSSMode::eDLAA.
const SL_DLSS_DLAA: u32 = 6;

/// DLSS's mode for an output/render ratio (DLAA 1, quality 1.5, balanced
/// 1.72, performance 2, ultra performance 3).
fn dlss_mode(ratio: f32) -> u32 {
    match ratio {
        r if r < 1.2 => SL_DLSS_DLAA,
        r if r < 1.6 => 3,
        r if r < 1.9 => 2,
        r if r < 2.6 => 1,
        _ => 4,
    }
}

/// Whether this session prepares DLSS's library (`DS_UPSCALER` names the one in development runs).
pub(super) fn prepares_dlss() -> bool {
    prepares(Kind::Dlss)
}

/// Streamline cannot run in this session (`gpu::streamline`): DLSS neither.
pub(super) fn streamline_failed(why: &str) {
    fail(Kind::Dlss, why);
}

/// Streamline is ready (`gpu::streamline::device_created`, its own thread): DLSS is, where the
/// device supports it.
pub(super) fn streamline_ready() {
    if sl::supported(sl::FEATURE_DLSS) {
        DLSS_READY.store(true, SeqCst);
        crate::info!("upscaler: DLSS ready: in use from the first present through Streamline");
        // (Ready before the first present through Streamline: `note_sl_present` then logs it.)
        if SL_PRESENTED.load(SeqCst) {
            now_ready(Kind::Dlss);
        }
    }
}

/// The common constants of a frame (`gpu::streamline::set_constants`, shared by DLSS and DLSS
/// Frame Generation): the jitter of the frame in pixels of the render size, the motion vectors'
/// scale, the render size's aspect, and whether the history starts anew. The camera's matrices
/// and vectors are not known to the port: neutral values (an identity camera, whose basis is
/// orthonormal, so DLSS-G's scene change detection stays valid and the reset flag decides).
pub fn frame_constants(jitter: (f32, f32), aspect: f32, reset: bool) -> SlConstants {
    // The game's motion vectors: screen fraction, current - previous; Streamline's: scaled
    // into [-1, 1] screen fractions, current -> previous.
    let (mx, my) = mv_scale();
    SlConstants {
        base: SlBase::new(sl::SL_CONSTANTS, 2),
        camera_view_to_clip: IDENTITY,
        clip_to_camera_view: IDENTITY,
        clip_to_lens_clip: IDENTITY,
        clip_to_prev_clip: IDENTITY,
        prev_clip_to_clip: IDENTITY,
        // (DLSS's jitter: the image's shift, as rendered.)
        jitter_offset: [jitter.0, jitter.1],
        mvec_scale: [-mx, -my],
        camera_pinhole_offset: [0.0, 0.0],
        camera_pos: [0.0, 0.0, 0.0],
        camera_up: [0.0, 1.0, 0.0],
        camera_right: [1.0, 0.0, 0.0],
        camera_fwd: [0.0, 0.0, 1.0],
        camera_near: 0.1,
        camera_far: 10000.0,
        camera_fov: 1.0,
        camera_aspect_ratio: aspect,
        motion_vectors_invalid_value: 0.0,
        // (The title's depth is reversed: near 1, far 0.)
        depth_inverted: SL_TRUE,
        camera_motion_included: SL_TRUE,
        motion_vectors_3d: SL_FALSE,
        reset: if reset { SL_TRUE } else { SL_FALSE },
        orthographic_projection: SL_FALSE,
        motion_vectors_dilated: SL_FALSE,
        motion_vectors_jittered: SL_FALSE,
        min_relative_linear_depth_object_separation: 40.0,
    }
}

/// The jitter Streamline is given for an upscaler's inputs (the title's offset times
/// `jitter_scale`).
pub fn sl_jitter(inp: &Inputs) -> (f32, f32) {
    let (sx, sy) = jitter_scale();
    (inp.jitter_x * sx, inp.jitter_y * sy)
}

/// slDLSSSetOptions (from the plugin, once the device is set).
type DlssSetOptions = unsafe extern "C" fn(*const SlViewport, *const SlDlssOptions) -> i32;

struct DlssState {
    set_options: DlssSetOptions,
    key: (u32, u32, u32, u32, bool),
    frame: u32,
}

static DLSS_STATE: Mutex<Option<DlssState>> = Mutex::new(None);

pub fn sl_resource(t: &Tex) -> SlResource {
    SlResource::image(t.image, t.view, t.layout, t.format, t.size, t.usage)
}

fn dlss_record(cmd: vk::CommandBuffer, inp: &Inputs, reset: bool) -> bool {
    if sl::lib().is_none() {
        return false;
    }
    let mut st = DLSS_STATE.lock();
    if st.is_none() {
        let Some(p) = sl::feature_fn(sl::FEATURE_DLSS, c"slDLSSSetOptions") else {
            fail(Kind::Dlss, "no slDLSSSetOptions");
            return false;
        };
        // SAFETY: sl_dlss.h's PFun_slDLSSSetOptions.
        let set_options: DlssSetOptions = unsafe { std::mem::transmute(p) };
        *st = Some(DlssState { set_options, key: (0, 0, 0, 0, false), frame: 0 });
    }
    let d = st.as_mut().unwrap();
    let key = (inp.in_width, inp.in_height, inp.width, inp.height, inp.ldr);
    let new_size = d.key != key;
    let mode = dlss_mode(inp.ratio());
    if new_size {
        if inp.ldr {
            crate::info!("upscaler: DLSS upscaling {}x{} to {}x{} (mode {mode}) after the game's TAA", inp.in_width, inp.in_height, inp.width, inp.height);
        } else {
            crate::info!("upscaler: DLSS native anti-aliasing (DLAA) at {}x{} after the game's TAA", inp.width, inp.height);
        }
        d.key = key;
    }
    let reset = reset || d.frame == 0 || new_size;
    d.frame += 1;
    let set_options = d.set_options;
    drop(st);
    let consts = frame_constants(sl_jitter(inp), inp.in_width as f32 / inp.in_height.max(1) as f32, reset);
    let opts = SlDlssOptions {
        base: SlBase::new(sl::SL_DLSS_OPTIONS, 3),
        mode,
        output_width: inp.width,
        output_height: inp.height,
        sharpness: 0.0,
        pre_exposure: 1.0,
        exposure_scale: 1.0,
        color_buffers_hdr: if inp.ldr || env_flag("DS_DLSS_HDR") == Some(false) { SL_FALSE } else { SL_TRUE },
        indicator_invert_x: SL_FALSE,
        indicator_invert_y: SL_FALSE,
        dlaa_preset: 0,
        quality_preset: 0,
        balanced_preset: 0,
        performance_preset: 0,
        ultra_performance_preset: 0,
        ultra_quality_preset: 0,
        use_auto_exposure: if inp.ldr || env_flag("DS_DLSS_AUTOEXP") == Some(false) { SL_FALSE } else { SL_TRUE },
        alpha_upscaling_enabled: SL_FALSE,
    };
    let res = [sl_resource(&inp.depth), sl_resource(&inp.velocity), sl_resource(&inp.color), sl_resource(&inp.output)];
    let extent = SlExtent { top: 0, left: 0, width: inp.in_width, height: inp.in_height };
    let out_extent = SlExtent { top: 0, left: 0, width: inp.width, height: inp.height };
    let frame = inp.frame;
    // Recorded into the command buffer the deferred list is played into (`cb`), not the one it was
    // recorded for (`cmd`): a batch fused into an earlier one's buffer at submission plays there,
    // and its own buffer is never begun. Until 2026-11-25 the upscalers recorded into `cmd`: with
    // DLSS (NGX 310.9.16) on, runs whose batches were fused hit an access violation inside the
    // driver (a write through a null object at 0x10) within seconds, from DLSS's own recording
    // (the frames of `exception::veh_body`'s first-chance line); unfused, the commands were in
    // the right buffer by chance.
    unsafe {
        super::vk::defer::external(cmd, Box::new(move |cb| {
            static N_TOKEN: AtomicU32 = AtomicU32::new(0);
            static N_CONST: AtomicU32 = AtomicU32::new(0);
            static N_OPT: AtomicU32 = AtomicU32::new(0);
            static N_EVAL: AtomicU32 = AtomicU32::new(0);
            let viewport = sl::viewport();
            EVAL_OK.store(false, Relaxed);
            // (The frame's token by the port's index, the flip that shows it: the same frame
            // DLSS Frame Generation and Reflex name, `gpu::streamline`.)
            let Some(token) = sl::token(frame) else {
                warn_some(&N_TOKEN, "slGetNewFrameToken", -1);
                return;
            };
            if !sl::set_constants(frame, token, &consts) {
                warn_some(&N_CONST, "slSetConstants", -1);
            }
            let r = set_options(&viewport, &opts);
            if r != SL_OK {
                warn_some(&N_OPT, "slDLSSSetOptions", r);
            }
            let tag = |i: usize, ty: u32| SlResourceTag::new(&res[i], ty, sl::SL_VALID_UNTIL_EVALUATE, if i == 3 { out_extent } else { extent });
            let tags = [tag(0, sl::SL_DEPTH), tag(1, sl::SL_MOTION_VECTORS), tag(2, sl::SL_SCALING_INPUT_COLOR), tag(3, sl::SL_SCALING_OUTPUT_COLOR)];
            let inputs: [*const SlBase; 5] = [&viewport.base, &tags[0].base, &tags[1].base, &tags[2].base, &tags[3].base];
            let r = sl::evaluate(sl::FEATURE_DLSS, token, &inputs, cb);
            EVAL_OK.store(r == SL_OK, Relaxed);
            if r != SL_OK {
                warn_some(&N_EVAL, "slEvaluateFeature (DLSS)", r);
            }
        }))
    };
    true
}

// ---------------------------------------------------------------------------
// AMD's FSR API (ffx_api.h, ffx_api_types.h, vk/ffx_api_vk.h, ffx_upscale.h
// of FSR SDK 2.4.0; natural C layout - the sizes are checked in the tests).

/// ffxApiHeader.
#[repr(C)]
#[derive(Clone, Copy)]
struct FfxHeader {
    ty: u64,
    next: *mut FfxHeader,
}

// SAFETY: `next` is null except while a creation chain is passed.
unsafe impl Send for FfxHeader {}

impl FfxHeader {
    const fn new(ty: u64) -> Self {
        Self { ty, next: std::ptr::null_mut() }
    }
}

#[repr(C)]
#[derive(Clone, Copy, Default)]
struct FfxDims {
    width: u32,
    height: u32,
}

/// FfxApiResourceDescription (the unions as their first members).
#[repr(C)]
#[derive(Clone, Copy, Default)]
struct FfxResDesc {
    ty: u32,
    format: u32,
    width: u32,
    height: u32,
    depth: u32,
    mip_count: u32,
    flags: u32,
    usage: u32,
}

/// FfxApiResource (Vulkan: the VkImage).
#[repr(C)]
#[derive(Clone, Copy, Default)]
struct FfxResource {
    resource: u64,
    desc: FfxResDesc,
    state: u32,
}

/// ffxCreateContextDescUpscale.
#[repr(C)]
struct FfxCreateUpscale {
    header: FfxHeader,
    flags: u32,
    max_render_size: FfxDims,
    max_upscale_size: FfxDims,
    message: Option<unsafe extern "C" fn(u32, *const u16)>,
}

/// ffxCreateBackendVKDesc (its VkDeviceContext inline).
#[repr(C)]
struct FfxBackendVk {
    header: FfxHeader,
    device: vk::Device,
    physical_device: vk::PhysicalDevice,
    get_device_proc_addr: vk::PFN_vkGetDeviceProcAddr,
}

/// ffxDispatchDescUpscale.
#[repr(C)]
#[derive(Clone, Copy)]
struct FfxDispatchUpscale {
    header: FfxHeader,
    command_list: vk::CommandBuffer,
    color: FfxResource,
    depth: FfxResource,
    motion_vectors: FfxResource,
    exposure: FfxResource,
    reactive: FfxResource,
    transparency_and_composition: FfxResource,
    output: FfxResource,
    jitter_offset: [f32; 2],
    motion_vector_scale: [f32; 2],
    render_size: FfxDims,
    upscale_size: FfxDims,
    enable_sharpening: bool,
    sharpness: f32,
    frame_time_delta: f32,
    pre_exposure: f32,
    reset: bool,
    camera_near: f32,
    camera_far: f32,
    camera_fov_angle_vertical: f32,
    view_space_to_meters_factor: f32,
    flags: u32,
}

/// ffxQueryGetProviderVersion.
#[repr(C)]
struct FfxProviderVersion {
    header: FfxHeader,
    version_id: u64,
    version_name: *const c_char,
}

const FFX_CREATE_UPSCALE: u64 = 0x0001_0000;
const FFX_DISPATCH_UPSCALE: u64 = 0x0001_0001;
const FFX_BACKEND_VK: u64 = 0x0100_0003;
const FFX_QUERY_PROVIDER_VERSION: u64 = 6;
const FFX_OK: u32 = 0;
// FfxApiCreateContextUpscaleFlags; FfxApiDispatchFsrUpscaleFlags.
const FFX_UPSCALE_HDR: u32 = 1 << 0;
/// The depth is reversed (near 1, far 0: the title's draws test GREATER_OR_EQUAL).
const FFX_UPSCALE_DEPTH_INVERTED: u32 = 1 << 3;
const FFX_UPSCALE_AUTO_EXPOSURE: u32 = 1 << 5;
const FFX_UPSCALE_NON_LINEAR_COLORSPACE: u32 = 1 << 8;
const FFX_UPSCALE_FLAG_NON_LINEAR_COLOR_SRGB: u32 = 1 << 1;
// FfxApiResourceType, -Usage, -State.
const FFX_TEXTURE2D: u32 = 2;
const FFX_USAGE_UAV: u32 = 1 << 1;
const FFX_USAGE_DEPTH: u32 = 1 << 2;
const FFX_USAGE_STENCIL: u32 = 1 << 5;
const FFX_STATE_UNORDERED_ACCESS: u32 = 1 << 1;
const FFX_STATE_COMPUTE_READ: u32 = 1 << 2;

type FfxContext = *mut c_void;

/// ffxConfigureDescGlobalDebug.
#[repr(C)]
struct FfxGlobalDebug {
    header: FfxHeader,
    effect_id: u64,
    message: Option<unsafe extern "C" fn(u32, *const u16)>,
    debug_level: u32,
}

const FFX_CONFIGURE_GLOBAL_DEBUG: u64 = 7;
const FFX_EFFECT_UPSCALE: u64 = 0x0001_0000;
const FFX_UPSCALE_DEBUG_CHECKING: u32 = 1 << 7;

struct FfxLib {
    configure: unsafe extern "C" fn(*mut FfxContext, *const FfxHeader) -> u32,
    create: unsafe extern "C" fn(*mut FfxContext, *mut FfxHeader, *const c_void) -> u32,
    query: unsafe extern "C" fn(*mut FfxContext, *mut FfxHeader) -> u32,
    dispatch: unsafe extern "C" fn(*mut FfxContext, *const FfxHeader) -> u32,
}

// SAFETY: plain function pointers into a library that stays loaded.
unsafe impl Send for FfxLib {}
unsafe impl Sync for FfxLib {}

/// FSR's loader library (None: not prepared, not installed or unusable).
fn ffx() -> Option<&'static FfxLib> {
    static L: std::sync::OnceLock<Option<FfxLib>> = std::sync::OnceLock::new();
    L.get_or_init(|| {
        if !prepares(Kind::Fsr) {
            return None;
        }
        match load_ffx() {
            Ok(l) => Some(l),
            Err(LoadError::Missing) => {
                crate::info!("upscaler: FSR not installed");
                None
            }
            Err(LoadError::Failed(e)) => {
                fail(Kind::Fsr, e);
                None
            }
        }
    })
    .as_ref()
}

fn load_ffx() -> Result<FfxLib, LoadError> {
    let dir = std::env::var_os("DS_FFX_DIR").map(PathBuf::from).unwrap_or_else(lib_dir);
    // (The loader loads the effect library beside it: both are checked.)
    let effect = dir.join("amd_fidelityfx_upscaler_vk.dll");
    if !effect.is_file() {
        return Err(LoadError::Missing);
    }
    check_signature(&effect, "Advanced Micro Devices")?;
    let path = dir.join("amd_fidelityfx_loader_vk.dll");
    let h = load_lib(&path, "Advanced Micro Devices")?;
    let f = |name: &CStr| export(h, &path, name);
    // SAFETY: the signatures of ffx_api.h.
    unsafe {
        let lib = FfxLib {
            configure: std::mem::transmute(f(c"ffxConfigure")?),
            create: std::mem::transmute(f(c"ffxCreateContext")?),
            query: std::mem::transmute(f(c"ffxQuery")?),
            dispatch: std::mem::transmute(f(c"ffxDispatch")?),
        };
        crate::info!("upscaler: FSR loaded from {}", dir.display());
        Ok(lib)
    }
}

/// FSR's Vulkan backend enables nothing itself and uses what the device
/// offers: these features where the device has them.
fn fsr_device_needs(instance: &ash::Instance, pdev: vk::PhysicalDevice) -> Option<DeviceNeeds> {
    ffx()?;
    if FSR_FAILED.load(Relaxed) {
        return None;
    }
    let want = [
        "shaderFloat16",
        "shaderInt16",
        "storageBuffer16BitAccess",
        "uniformAndStorageBuffer16BitAccess",
        "shaderStorageBufferArrayNonUniformIndexing",
        "subgroupSizeControl",
        "computeFullSubgroups",
    ];
    let have = device_features(instance, pdev, &want);
    let features: Vec<String> = want.iter().zip(have).filter(|(_, h)| *h == Some(true)).map(|(n, _)| n.to_string()).collect();
    crate::info!("upscaler: FSR gets the device features {features:?}");
    Some(DeviceNeeds { extensions: Vec::new(), mutable_descriptor: false, features, extra_graphics_queues: 0 })
}

unsafe extern "C" fn ffx_log(ty: u32, msg: *const u16) {
    if msg.is_null() {
        return;
    }
    let n = (0..).take_while(|&i| unsafe { *msg.add(i) } != 0).count();
    let m = String::from_utf16_lossy(unsafe { std::slice::from_raw_parts(msg, n) });
    match ty {
        0 => crate::warn!("FSR error: {}", m.trim_end()),
        1 => crate::warn!("FSR: {}", m.trim_end()),
        _ => crate::info!("FSR: {}", m.trim_end()),
    }
}

/// DS_FSR_DEBUG=1: FSR's parameter checks and all its messages.
fn fsr_debug() -> bool {
    static D: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *D.get_or_init(|| std::env::var("DS_FSR_DEBUG").is_ok_and(|v| v == "1"))
}

/// FfxApiSurfaceFormat of the formats the hook hands over (0: none).
fn ffx_format(f: vk::Format) -> u32 {
    match f {
        vk::Format::R16G16B16A16_SFLOAT => 4,
        vk::Format::R32G32_SFLOAT => 6,
        vk::Format::D24_UNORM_S8_UINT | vk::Format::X8_D24_UNORM_PACK32 => 8,
        vk::Format::R8G8B8A8_UNORM => 10,
        vk::Format::B8G8R8A8_UNORM => 14,
        vk::Format::B10G11R11_UFLOAT_PACK32 => 16,
        vk::Format::A2B10G10R10_UNORM_PACK32 => 17,
        vk::Format::R16G16_SFLOAT => 18,
        vk::Format::R16_UNORM | vk::Format::D16_UNORM | vk::Format::D16_UNORM_S8_UINT => 23,
        vk::Format::R32_SFLOAT | vk::Format::D32_SFLOAT | vk::Format::D32_SFLOAT_S8_UINT => 28,
        _ => 0,
    }
}

fn ffx_resource(t: &Tex, state: u32) -> FfxResource {
    use ash::vk::Handle;
    let depth = matches!(t.format, vk::Format::D16_UNORM | vk::Format::D16_UNORM_S8_UINT | vk::Format::D24_UNORM_S8_UINT | vk::Format::X8_D24_UNORM_PACK32 | vk::Format::D32_SFLOAT | vk::Format::D32_SFLOAT_S8_UINT);
    let stencil = matches!(t.format, vk::Format::D16_UNORM_S8_UINT | vk::Format::D24_UNORM_S8_UINT | vk::Format::D32_SFLOAT_S8_UINT);
    let mut usage = 0;
    if depth {
        usage |= FFX_USAGE_DEPTH;
    }
    if stencil {
        usage |= FFX_USAGE_STENCIL;
    }
    if t.usage.contains(vk::ImageUsageFlags::STORAGE) {
        usage |= FFX_USAGE_UAV;
    }
    FfxResource {
        resource: t.image.as_raw(),
        desc: FfxResDesc { ty: FFX_TEXTURE2D, format: ffx_format(t.format), width: t.size.0, height: t.size.1, depth: 1, mip_count: 1, flags: 0, usage },
        state,
    }
}

/// The context (made at the first use for its sizes) and the frame clock.
struct FsrState {
    ctx: usize,
    key: (u32, u32, u32, u32, bool),
    frame: u32,
    last: std::time::Instant,
}

static FSR_STATE: Mutex<Option<FsrState>> = Mutex::new(None);

fn fsr_record(cmd: vk::CommandBuffer, inp: &Inputs, reset: bool) -> bool {
    let Some(f) = ffx() else { return false };
    for t in [&inp.color, &inp.depth, &inp.velocity, &inp.output] {
        if ffx_format(t.format) == 0 {
            fail(Kind::Fsr, format!("no FSR format for {:?}", t.format));
            return false;
        }
    }
    let key = (inp.in_width, inp.in_height, inp.width, inp.height, inp.ldr);
    let mut st = FSR_STATE.lock();
    if st.as_ref().is_some_and(|s| s.key != key) {
        // (Other sizes: a new context; the old one leaks - rare.)
        *st = None;
    }
    if st.is_none() {
        let c = super::vk::ctx();
        let mut backend = FfxBackendVk {
            header: FfxHeader::new(FFX_BACKEND_VK),
            device: c.device.handle(),
            physical_device: c.pdev,
            get_device_proc_addr: c.instance.fp_v1_0().get_device_proc_addr,
        };
        // (HDR scene colour with auto exposure; the upscaling path's
        // finished picture is sRGB-encoded LDR.)
        let colour = if inp.ldr { FFX_UPSCALE_NON_LINEAR_COLORSPACE } else { FFX_UPSCALE_HDR | FFX_UPSCALE_AUTO_EXPOSURE } | FFX_UPSCALE_DEPTH_INVERTED;
        let mut desc = FfxCreateUpscale {
            header: FfxHeader { ty: FFX_CREATE_UPSCALE, next: &mut backend.header },
            flags: colour | if fsr_debug() { FFX_UPSCALE_DEBUG_CHECKING } else { 0 },
            max_render_size: FfxDims { width: inp.in_width, height: inp.in_height },
            max_upscale_size: FfxDims { width: inp.width, height: inp.height },
            message: Some(ffx_log),
        };
        let mut ctx: FfxContext = std::ptr::null_mut();
        let r = unsafe { (f.create)(&mut ctx, &mut desc.header, std::ptr::null()) };
        if r != FFX_OK || ctx.is_null() {
            fail(Kind::Fsr, format!("ffxCreateContext failed ({r})"));
            return false;
        }
        // Its messages to the port's log: errors and warnings (the
        // context's own through `message`), everything with
        // DS_FSR_DEBUG=1 - the loader's and the upscaler effect's.
        for effect_id in [0, FFX_EFFECT_UPSCALE] {
            let dbg = FfxGlobalDebug { header: FfxHeader::new(FFX_CONFIGURE_GLOBAL_DEBUG), effect_id, message: Some(ffx_log), debug_level: if fsr_debug() { 0x0fff_ffff } else { 2 } };
            let r = unsafe { (f.configure)(&mut ctx, &dbg.header) };
            if r != FFX_OK && fsr_debug() {
                crate::info!("upscaler: FSR's message setup ({effect_id:#x}): {r}");
            }
        }
        let mut ver = FfxProviderVersion { header: FfxHeader::new(FFX_QUERY_PROVIDER_VERSION), version_id: 0, version_name: std::ptr::null() };
        let name = if unsafe { (f.query)(&mut ctx, &mut ver.header) } == FFX_OK && !ver.version_name.is_null() {
            unsafe { CStr::from_ptr(ver.version_name) }.to_string_lossy().into_owned()
        } else {
            "?".into()
        };
        if inp.ldr {
            crate::info!("upscaler: FSR upscaling {}x{} to {}x{} after the game's TAA (provider {name})", inp.in_width, inp.in_height, inp.width, inp.height);
        } else {
            crate::info!("upscaler: FSR native anti-aliasing at {}x{} after the game's TAA (provider {name})", inp.width, inp.height);
        }
        *st = Some(FsrState { ctx: ctx as usize, key, frame: 0, last: std::time::Instant::now() });
    }
    let s = st.as_mut().unwrap();
    let (jx, jy) = (inp.jitter_x, inp.jitter_y);
    let (sx, sy) = jitter_scale();
    let (mx, my) = mv_scale();
    let now = std::time::Instant::now();
    let dt_ms = if s.frame == 0 { 16.7 } else { (now - s.last).as_secs_f32() * 1000.0 };
    s.last = now;
    // (The camera's planes and field of view: FSR 3's depth tests.)
    let cam: Result<(f32, f32, f32), String> = Err("the title's camera is not read yet".into());
    if s.frame == 0 {
        if let Err(e) = &cam {
            crate::warn!("upscaler: FSR without the game camera ({e}): stand-in planes 0.1 / 10000, 57 deg");
        }
    }
    let (near, far, fov) = cam.unwrap_or((0.1, 10000.0, 1.0));
    let d = FfxDispatchUpscale {
        header: FfxHeader::new(FFX_DISPATCH_UPSCALE),
        command_list: cmd,
        color: ffx_resource(&inp.color, FFX_STATE_COMPUTE_READ),
        depth: ffx_resource(&inp.depth, FFX_STATE_COMPUTE_READ),
        motion_vectors: ffx_resource(&inp.velocity, FFX_STATE_COMPUTE_READ),
        exposure: FfxResource::default(),
        reactive: FfxResource::default(),
        transparency_and_composition: FfxResource::default(),
        output: ffx_resource(&inp.output, FFX_STATE_UNORDERED_ACCESS),
        // (FSR's jitter: the image's shift, as rendered.)
        jitter_offset: [jx * sx, jy * sy],
        // The game's motion vectors: screen fraction, current - previous;
        // FSR's: pixels, current -> previous.
        motion_vector_scale: [-(inp.in_width as f32) * mx, -(inp.in_height as f32) * my],
        render_size: FfxDims { width: inp.in_width, height: inp.in_height },
        upscale_size: FfxDims { width: inp.width, height: inp.height },
        enable_sharpening: false,
        sharpness: 0.0,
        frame_time_delta: dt_ms.clamp(1.0, 100.0),
        pre_exposure: 1.0,
        reset: reset || s.frame == 0,
        camera_near: near,
        camera_far: far,
        camera_fov_angle_vertical: fov,
        view_space_to_meters_factor: 1.0,
        flags: if inp.ldr { FFX_UPSCALE_FLAG_NON_LINEAR_COLOR_SRGB } else { 0 },
    };
    if s.frame == 0 {
        crate::info!("upscaler: FSR camera near {near:.3}, far {far:.1}, vertical field of view {:.1} deg", fov.to_degrees());
    }
    s.frame += 1;
    let (ctx, dispatch) = (s.ctx, f.dispatch);
    drop(st);
    unsafe {
        // (Into the command buffer the deferred list is played into, `cb`: see `dlss_record`.)
        super::vk::defer::external(cmd, Box::new(move |cb| {
            static N: AtomicU32 = AtomicU32::new(0);
            let mut c = ctx as FfxContext;
            // (The whole description, its command buffer the one played into: a closure naming
            // only `d.header` would capture just those 16 bytes.)
            let mut d = d;
            d.command_list = cb;
            let desc: *const FfxHeader = (&d as *const FfxDispatchUpscale).cast();
            let r = dispatch(&mut c, desc);
            EVAL_OK.store(r == FFX_OK, Relaxed);
            if r != FFX_OK {
                warn_some(&N, "ffxDispatch (FSR)", r as i32);
            }
        }))
    };
    true
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn layouts_and_halton() {
        assert_eq!(std::mem::size_of::<XessView>(), 48);
        assert_eq!(std::mem::size_of::<XessExecute>(), 360);
        assert_eq!(std::mem::size_of::<XessInit>(), 64);
        assert_eq!(halton(1, 2), 0.0);
        assert_eq!(halton(2, 2), -0.25);
        assert!((halton(1, 3) - (1.0 / 3.0 - 0.5)).abs() < 1e-6);
        for i in 1..=8 {
            let (x, y) = (halton(i, 2), halton(i, 3));
            assert!((-0.5..0.5).contains(&x) && (-0.5..0.5).contains(&y));
        }
    }

    /// FSR's structures (natural C layout, checked against the headers).
    #[test]
    fn ffx_layouts() {
        use std::mem::{offset_of, size_of};
        assert_eq!(size_of::<FfxHeader>(), 16);
        assert_eq!(size_of::<FfxResDesc>(), 32);
        assert_eq!(size_of::<FfxResource>(), 48);
        assert_eq!(size_of::<FfxCreateUpscale>(), 48);
        assert_eq!(offset_of!(FfxCreateUpscale, message), 40);
        assert_eq!(size_of::<FfxBackendVk>(), 40);
        assert_eq!(offset_of!(FfxDispatchUpscale, color), 24);
        assert_eq!(offset_of!(FfxDispatchUpscale, output), 312);
        assert_eq!(offset_of!(FfxDispatchUpscale, jitter_offset), 360);
        assert_eq!(offset_of!(FfxDispatchUpscale, enable_sharpening), 392);
        assert_eq!(offset_of!(FfxDispatchUpscale, sharpness), 396);
        assert_eq!(offset_of!(FfxDispatchUpscale, reset), 408);
        assert_eq!(offset_of!(FfxDispatchUpscale, camera_near), 412);
        assert_eq!(offset_of!(FfxDispatchUpscale, flags), 428);
        assert_eq!(size_of::<FfxDispatchUpscale>(), 432);
        assert_eq!(size_of::<FfxProviderVersion>(), 32);
        assert_eq!(size_of::<FfxGlobalDebug>(), 40);
        assert_eq!(ffx_format(vk::Format::B10G11R11_UFLOAT_PACK32), 16);
        assert_eq!(ffx_format(vk::Format::R16G16_SFLOAT), 18);
        assert_eq!(ffx_format(vk::Format::D32_SFLOAT), 28);
    }

    #[test]
    fn modes_by_ratio() {
        assert_eq!(xess_quality(1.0), XESS_QUALITY_AA);
        assert_eq!(xess_quality(1.5), 104);
        assert_eq!(xess_quality(1.0 / 0.58), 103);
        assert_eq!(xess_quality(2.0), 102);
        assert_eq!(xess_quality(3.0), 100);
        assert_eq!(dlss_mode(1.0), SL_DLSS_DLAA);
        assert_eq!(dlss_mode(1.5), 3);
        assert_eq!(dlss_mode(1.0 / 0.58), 2);
        assert_eq!(dlss_mode(2.0), 1);
        assert_eq!(dlss_mode(3.0), 4);
    }

    #[test]
    fn choices_parse() {
        assert_eq!(Kind::parse("dlss"), Kind::Dlss);
        assert_eq!(Kind::parse(" xess "), Kind::Xess);
        assert_eq!(Kind::parse("off"), Kind::Off);
        assert_eq!(Kind::parse("fsr"), Kind::Fsr);
        assert_eq!(Kind::parse("taa"), Kind::Off);
        for k in [Kind::Off, Kind::Xess, Kind::Dlss, Kind::Fsr] {
            assert_eq!(Kind::from_u8(k as u8), k);
        }
    }
}
