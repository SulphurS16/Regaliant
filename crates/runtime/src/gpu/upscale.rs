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
//! DLSS: NVIDIA Streamline 2.14 (sl.interposer.dll with its plugins sl.common.dll and
//! sl.dlss.dll, MIT, shipped) and DLSS itself, nvngx_dlss.dll (NVIDIA's RTX SDK license: not
//! shipped, the player's own copy; without it Streamline is not started), in Streamline's
//! "manual hooking" mode: slInit before the device, the DLSS plugin's Vulkan extensions and
//! features added to the instance and device, slSetVulkanInfo once it exists (on a thread of
//! its own: DLSS is offered once it is done), the present through Streamline's
//! vkQueuePresentKHR, and each frame the four images tagged on the slEvaluateFeature call.
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
    if k == Kind::Dlss && HELPER.load(Relaxed) {
        return false;
    }
    match std::env::var("DS_UPSCALER") {
        Ok(v) => Kind::parse(&v) == k,
        Err(_) => true,
    }
}

/// A helper process (the launcher's checks, the shader preparation's steps: `main`): Streamline is
/// not loaded. They never present a frame of their own through it, end with their own exit codes
/// before a clean shutdown, and use the game's own TAA; with Streamline on trial each left a
/// marker that the next start counted as a death, two of them switching DLSS off (the review of
/// 2026-11-25 night), and the preparation's steps were where a player's DLSS file crashed on
/// 2026-11-24.
pub static HELPER: AtomicBool = AtomicBool::new(false);

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
        streamline_settled("DLSS ready and a frame presented through Streamline");
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
    // (A TAA at the render size in effect is the scene's at once: after the render size shrank,
    // an Upscaling mode chosen, it was taken for a menu's smaller view for MAIN_VIEW_HOLD flips,
    // ~3 s of the title's TAA alone.)
    if (w, h) == (mw, mh) || (w, h) == crate::dispmode::render_size() || u64::from(w) * u64::from(h) > u64::from(mw) * u64::from(mh) || frame > at + MAIN_VIEW_HOLD {
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
        v.push(("dlss".into(), "DLSS".into()));
    }
    if ready(Kind::Xess) {
        v.push(("xess".into(), "XeSS".into()));
    }
    if ready(Kind::Fsr) {
        v.push(("fsr".into(), "FSR".into()));
    }
    v
}

/// The textures' level-of-detail bias added to the title's own while an upscaler fills in from
/// below its output, in 1/16 steps (0 otherwise): the upscalers' guidance, log2 of the render
/// size over the output's (`note_sizes`, each vblank), so that surfaces keep the detail the
/// output shows. The title's own TAA keeps the title's biases at every Render Scale.
static MIP_BIAS16: std::sync::atomic::AtomicI32 = std::sync::atomic::AtomicI32::new(0);

/// The sizes in effect (`dispmode::tick`): the render size and the upscaler's output.
pub fn note_sizes(render: (u32, u32), out: (u32, u32)) {
    let b = if active() && render.1 > 0 && render.1 < out.1 { ((render.1 as f32 / out.1 as f32).log2() * 16.0).round() as i32 } else { 0 };
    if MIP_BIAS16.swap(b, Relaxed) != b {
        crate::info!("upscaler: texture bias {:+.2} for {}x{} to {}x{}", b as f32 / 16.0, render.0, render.1, out.0, out.1);
    }
}

/// The textures' added bias now, in 1/16 steps (`MIP_BIAS16`).
pub fn mip_bias16() -> i32 {
    MIP_BIAS16.load(Relaxed)
}

fn fail(k: Kind, msg: impl std::fmt::Display) {
    if !failed_flag(k).swap(true, Relaxed) {
        crate::warn!("upscaler: {} unavailable: {msg}", k.name());
    }
    if k == Kind::Dlss {
        streamline_settled("DLSS unavailable without a crash");
    }
}

// ---------------------------------------------------------------------------
// Loading.

trait WideNull {
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
fn lib_dir() -> PathBuf {
    if let Some(d) = std::env::var_os("DS_UPSCALER_DIR") {
        return PathBuf::from(d);
    }
    crate::paths().upscalers()
}

/// Checks that `path` carries a valid Authenticode signature whose signer's
/// name starts with `publisher` (no revocation check, nothing fetched from
/// the network). DS_UPSCALER_UNSIGNED=1 skips it (development builds).
fn check_signature(path: &Path, publisher: &str) -> Result<(), String> {
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
enum LoadError {
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
fn load_lib(path: &Path, publisher: &str) -> Result<HMODULE, LoadError> {
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
fn export(h: HMODULE, path: &Path, name: &CStr) -> Result<*const c_void, String> {
    use windows_sys::Win32::System::LibraryLoader::GetProcAddress;
    match unsafe { GetProcAddress(h, name.as_ptr() as *const u8) } {
        Some(p) => Ok(p as *const c_void),
        None => Err(format!("{} has no {}", path.display(), name.to_string_lossy())),
    }
}

/// C strings out of a library's list (count, array of pointers).
unsafe fn names(count: u32, list: *const *const c_char) -> Vec<CString> {
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
    for (k, list) in [(Kind::Xess, xess_instance_extensions()), (Kind::Dlss, sl().map(|s| s.reqs.instance_ext.clone()).unwrap_or_default())] {
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
}

static XESS_DEVICE_OK: AtomicBool = AtomicBool::new(false);
static DLSS_DEVICE_OK: AtomicBool = AtomicBool::new(false);
static FSR_DEVICE_OK: AtomicBool = AtomicBool::new(false);

/// Device extensions and features of the upscalers that can run on this
/// device (before vkCreateDevice).
pub fn device_needs(instance: &ash::Instance, pdev: vk::PhysicalDevice) -> DeviceNeeds {
    let mut n = DeviceNeeds::default();
    for (k, needs, ok) in [
        (Kind::Xess, xess_device_needs(instance, pdev), &XESS_DEVICE_OK),
        (Kind::Dlss, dlss_device_needs(instance, pdev), &DLSS_DEVICE_OK),
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
        "storageBuffer16BitAccess" => &mut f11.storage_buffer16_bit_access,
        "uniformAndStorageBuffer16BitAccess" => &mut f11.uniform_and_storage_buffer16_bit_access,
        "timelineSemaphore" => &mut f12.timeline_semaphore,
        "descriptorIndexing" => &mut f12.descriptor_indexing,
        "bufferDeviceAddress" => &mut f12.buffer_device_address,
        "shaderFloat16" => &mut f12.shader_float16,
        "shaderStorageBufferArrayNonUniformIndexing" => &mut f12.shader_storage_buffer_array_non_uniform_indexing,
        "subgroupSizeControl" => &mut f13.subgroup_size_control,
        "computeFullSubgroups" => &mut f13.compute_full_subgroups,
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
fn features_supported(instance: &ash::Instance, pdev: vk::PhysicalDevice, names: &[String]) -> Result<(), String> {
    let refs: Vec<&str> = names.iter().map(String::as_str).collect();
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
pub fn device_created(instance: &ash::Instance, pdev: vk::PhysicalDevice, device: &ash::Device, qf: u32, queues: u32) {
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
    if DLSS_DEVICE_OK.load(Relaxed) {
        dlss_device_created(instance, pdev, device, qf, queues);
    }
    DEVICE_MADE.store(true, Relaxed);
    let k = choice();
    if k != Kind::Off && !ready(k) && !may_become_ready(k) {
        crate::info!("upscaler: {} (the chosen Anti-Aliasing) is not available here: the game's TAA stays", k.name());
    }
}

/// Streamline's vkQueuePresentKHR once DLSS is ready (its per-frame
/// bookkeeping runs there); None: the usual present.
pub fn present_hook() -> Option<vk::PFN_vkQueuePresentKHR> {
    SL_PRESENT.get().copied()
}

static SL_PRESENT: std::sync::OnceLock<vk::PFN_vkQueuePresentKHR> = std::sync::OnceLock::new();

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
}

impl Inputs {
    /// Output size over render size (1: native anti-aliasing).
    fn ratio(&self) -> f32 {
        self.width as f32 / self.in_width.max(1) as f32
    }

    /// The render size is below the output's.
    fn upscales(&self) -> bool {
        self.in_width < self.width || self.in_height < self.height
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
    Some(DeviceNeeds { extensions, mutable_descriptor: true, features: Vec::new() })
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
        if inp.upscales() {
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
// Streamline's C interface (sl_struct.h, sl_core_types.h, sl_consts.h,
// sl_dlss.h, sl_helpers_vk.h of SDK 2.14.16; MSVC x64 layout, which
// repr(C) reproduces - the sizes are checked in the tests).

/// sl::StructType (a GUID).
#[repr(C)]
#[derive(Clone, Copy)]
struct SlType(u32, u16, u16, [u8; 8]);

/// sl::BaseStructure.
#[repr(C)]
#[derive(Clone, Copy)]
struct SlBase {
    next: *const SlBase,
    ty: SlType,
    version: usize,
}

impl SlBase {
    const fn new(ty: SlType, version: usize) -> Self {
        Self { next: std::ptr::null(), ty, version }
    }
}

// SAFETY: `next` is always null here (no chained structures): the per-frame
// structures move to the recorder thread as plain data.
unsafe impl Send for SlBase {}

const SL_PREFERENCES: SlType = SlType(0x1ca10965, 0xbf8e, 0x432b, [0x8d, 0xa1, 0x67, 0x16, 0xd8, 0x79, 0xfb, 0x14]);
const SL_RESOURCE: SlType = SlType(0x3a9d70cf, 0x2418, 0x4b72, [0x83, 0x91, 0x13, 0xf8, 0x72, 0x1c, 0x72, 0x61]);
const SL_RESOURCE_TAG: SlType = SlType(0x4c6a5aad, 0xb445, 0x496c, [0x87, 0xff, 0x1a, 0xf3, 0x84, 0x5b, 0xe6, 0x53]);
const SL_VIEWPORT: SlType = SlType(0x171b6435, 0x9b3c, 0x4fc8, [0x99, 0x94, 0xfb, 0xe5, 0x25, 0x69, 0xaa, 0xa4]);
const SL_FEATURE_REQUIREMENTS: SlType = SlType(0x66714097, 0xac6d, 0x4bc6, [0x89, 0x15, 0x1e, 0x0f, 0x55, 0xa6, 0xb6, 0x1f]);
const SL_FEATURE_VERSION: SlType = SlType(0x6d5b51f0, 0x076b, 0x486d, [0x99, 0x95, 0x5a, 0x56, 0x10, 0x43, 0xf5, 0xc1]);
const SL_ADAPTER_INFO: SlType = SlType(0x0677315f, 0xa746, 0x4492, [0x9f, 0x42, 0xcb, 0x61, 0x42, 0xc9, 0xc3, 0xd4]);
const SL_VULKAN_INFO: SlType = SlType(0x0eed6fd5, 0x82cd, 0x43a9, [0xbd, 0xb5, 0x47, 0xa5, 0xba, 0x2f, 0x45, 0xd6]);
const SL_CONSTANTS: SlType = SlType(0xdcd35ad7, 0x4e4a, 0x4bad, [0xa9, 0x0c, 0xe0, 0xc4, 0x9e, 0xb2, 0x3a, 0xfe]);
const SL_DLSS_OPTIONS: SlType = SlType(0x6ac826e4, 0x4c61, 0x4101, [0xa9, 0x2d, 0x63, 0x8d, 0x42, 0x10, 0x57, 0xb8]);

/// sl::kSDKVersion of 2.14.16.
const SL_SDK_VERSION: u64 = (2u64 << 48) | (14u64 << 32) | (16u64 << 16) | 0xfedc;
const SL_FEATURE_DLSS: u32 = 0;
const SL_OK: i32 = 0;
// PreferenceFlags.
const SL_DISABLE_CL_STATE_TRACKING: u64 = 1 << 0;
const SL_USE_MANUAL_HOOKING: u64 = 1 << 2;
// BufferType.
const SL_DEPTH: u32 = 0;
const SL_MOTION_VECTORS: u32 = 1;
const SL_SCALING_INPUT_COLOR: u32 = 3;
const SL_SCALING_OUTPUT_COLOR: u32 = 4;
/// ResourceLifecycle::eValidUntilEvaluate.
const SL_VALID_UNTIL_EVALUATE: u32 = 2;
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
/// RenderAPI::eVulkan.
const SL_VULKAN: u32 = 2;
// sl::Boolean.
const SL_FALSE: u8 = 0;
const SL_TRUE: u8 = 1;

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
struct SlResource {
    base: SlBase,
    ty: u8,
    native: u64,
    memory: u64,
    view: u64,
    state: u32,
    width: u32,
    height: u32,
    native_format: u32,
    mip_levels: u32,
    array_layers: u32,
    gpu_virtual_address: u64,
    flags: u32,
    usage: u32,
    internal_flags: u16,
    reserved: u16,
}

#[repr(C)]
#[derive(Clone, Copy)]
struct SlExtent {
    top: u32,
    left: u32,
    width: u32,
    height: u32,
}

/// sl::ResourceTag.
#[repr(C)]
#[derive(Clone, Copy)]
struct SlResourceTag {
    base: SlBase,
    resource: *const SlResource,
    ty: u32,
    lifecycle: u32,
    extent: SlExtent,
}

/// sl::ViewportHandle.
#[repr(C)]
#[derive(Clone, Copy)]
struct SlViewport {
    base: SlBase,
    value: u32,
}

/// sl::FeatureRequirements (version 2).
#[repr(C)]
struct SlFeatureRequirements {
    base: SlBase,
    flags: u32,
    max_cpu_threads: u32,
    max_viewports: u32,
    num_required_tags: u32,
    required_tags: *const u32,
    os_detected: [u32; 3],
    os_required: [u32; 3],
    driver_detected: [u32; 3],
    driver_required: [u32; 3],
    vk_compute_queues: u32,
    vk_graphics_queues: u32,
    vk_num_device_extensions: u32,
    vk_device_extensions: *const *const c_char,
    vk_num_instance_extensions: u32,
    vk_instance_extensions: *const *const c_char,
    vk_num_features12: u32,
    vk_features12: *const *const c_char,
    vk_num_features13: u32,
    vk_features13: *const *const c_char,
    vk_optical_flow_queues: u32,
}

/// sl::FeatureVersion.
#[repr(C)]
struct SlFeatureVersion {
    base: SlBase,
    sl: [u32; 3],
    ngx: [u32; 3],
}

/// sl::AdapterInfo.
#[repr(C)]
struct SlAdapterInfo {
    base: SlBase,
    luid: *const u8,
    luid_size: u32,
    vk_physical_device: vk::PhysicalDevice,
}

/// sl::VulkanInfo (version 3).
#[repr(C)]
struct SlVulkanInfo {
    base: SlBase,
    device: vk::Device,
    instance: vk::Instance,
    physical_device: vk::PhysicalDevice,
    compute_queue_index: u32,
    compute_queue_family: u32,
    graphics_queue_index: u32,
    graphics_queue_family: u32,
    optical_flow_queue_index: u32,
    optical_flow_queue_family: u32,
    use_native_optical_flow: bool,
    compute_queue_create_flags: u32,
    graphics_queue_create_flags: u32,
    optical_flow_queue_create_flags: u32,
}

/// A row-major 4x4 matrix (sl::float4x4).
type SlMatrix = [[f32; 4]; 4];

const IDENTITY: SlMatrix = [[1.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0], [0.0, 0.0, 1.0, 0.0], [0.0, 0.0, 0.0, 1.0]];

/// sl::Constants (version 2).
#[repr(C)]
#[derive(Clone, Copy)]
struct SlConstants {
    base: SlBase,
    camera_view_to_clip: SlMatrix,
    clip_to_camera_view: SlMatrix,
    clip_to_lens_clip: SlMatrix,
    clip_to_prev_clip: SlMatrix,
    prev_clip_to_clip: SlMatrix,
    jitter_offset: [f32; 2],
    mvec_scale: [f32; 2],
    camera_pinhole_offset: [f32; 2],
    camera_pos: [f32; 3],
    camera_up: [f32; 3],
    camera_right: [f32; 3],
    camera_fwd: [f32; 3],
    camera_near: f32,
    camera_far: f32,
    camera_fov: f32,
    camera_aspect_ratio: f32,
    motion_vectors_invalid_value: f32,
    depth_inverted: u8,
    camera_motion_included: u8,
    motion_vectors_3d: u8,
    reset: u8,
    orthographic_projection: u8,
    motion_vectors_dilated: u8,
    motion_vectors_jittered: u8,
    min_relative_linear_depth_object_separation: f32,
}

/// sl::DLSSOptions (version 3).
#[repr(C)]
#[derive(Clone, Copy)]
struct SlDlssOptions {
    base: SlBase,
    mode: u32,
    output_width: u32,
    output_height: u32,
    sharpness: f32,
    pre_exposure: f32,
    exposure_scale: f32,
    color_buffers_hdr: u8,
    indicator_invert_x: u8,
    indicator_invert_y: u8,
    dlaa_preset: u32,
    quality_preset: u32,
    balanced_preset: u32,
    performance_preset: u32,
    ultra_performance_preset: u32,
    ultra_quality_preset: u32,
    use_auto_exposure: u8,
    alpha_upscaling_enabled: u8,
}

/// The port's NGX project id (Streamline passes it with the engine type
/// "custom" when no NVIDIA application id is given).
const NGX_PROJECT_ID: &CStr = c"48421953-2aae-4cec-9b59-7e92288edb4a";

/// DLSS's Vulkan needs (slGetFeatureRequirements).
#[derive(Default)]
struct DlssReqs {
    instance_ext: Vec<CString>,
    device_ext: Vec<CString>,
    features12: Vec<String>,
    features13: Vec<String>,
    queues: u32,
}

struct SlLib {
    is_supported: unsafe extern "C" fn(u32, *const SlAdapterInfo) -> i32,
    set_vulkan_info: unsafe extern "C" fn(*const SlVulkanInfo) -> i32,
    frame_token: unsafe extern "C" fn(*mut *const c_void, *const u32) -> i32,
    set_constants: unsafe extern "C" fn(*const SlConstants, *const c_void, *const SlViewport) -> i32,
    evaluate: unsafe extern "C" fn(u32, *const c_void, *const *const SlBase, u32, vk::CommandBuffer) -> i32,
    feature_function: unsafe extern "C" fn(u32, *const c_char, *mut *const c_void) -> i32,
    version: unsafe extern "C" fn(u32, *mut SlFeatureVersion) -> i32,
    gipa: vk::PFN_vkGetInstanceProcAddr,
    reqs: DlssReqs,
}

// SAFETY: plain function pointers into a library that stays loaded.
unsafe impl Send for SlLib {}
unsafe impl Sync for SlLib {}

/// Streamline loaded (`sl`): read without loading it (`streamline_loaded`).
static SL_LOADED: std::sync::atomic::AtomicBool = std::sync::atomic::AtomicBool::new(false);

/// Whether Streamline is loaded now, without loading it: a run without a window presents its
/// hidden image only then (`present::headless_present`; NVIDIA's present hangs without it).
pub fn streamline_loaded() -> bool {
    SL_LOADED.load(std::sync::atomic::Ordering::Relaxed)
}

/// Streamline, initialised with DLSS (None: not prepared, not installed or
/// unusable).
fn sl() -> Option<&'static SlLib> {
    static L: std::sync::OnceLock<Option<SlLib>> = std::sync::OnceLock::new();
    L.get_or_init(|| {
        if !prepares(Kind::Dlss) {
            return None;
        }
        match load_sl() {
            Ok(l) => {
                SL_LOADED.store(true, std::sync::atomic::Ordering::Relaxed);
                Some(l)
            }
            Err(LoadError::Missing) => {
                crate::info!("upscaler: DLSS not installed (nvngx_dlss.dll - the player's own - and Streamline in the upscalers folder)");
                None
            }
            Err(LoadError::Failed(e)) => {
                fail(Kind::Dlss, e);
                None
            }
        }
    })
    .as_ref()
}

unsafe extern "C" fn sl_log(ty: u32, msg: *const c_char) {
    if msg.is_null() {
        return;
    }
    let m = unsafe { CStr::from_ptr(msg) }.to_string_lossy();
    let m = m.trim_end();
    // (Its Reflex low-latency library is not shipped - the port uses no
    // Reflex: that failure is expected.)
    let ty = if m.contains("NvLowLatencyVk.dll") { 0 } else { ty };
    match ty {
        2 => crate::warn!("Streamline error: {m}"),
        1 => crate::warn!("Streamline: {m}"),
        _ => crate::debug!("Streamline: {m}"),
    }
}

/// The system folder (nvapi64.dll, NVIDIA's driver library, is there with every NVIDIA driver).
fn system_dir() -> Option<PathBuf> {
    let mut sys = [0u16; 260];
    let n = unsafe { windows_sys::Win32::System::SystemInformation::GetSystemDirectoryW(sys.as_mut_ptr(), sys.len() as u32) } as usize;
    (n > 0 && n < sys.len()).then(|| PathBuf::from(String::from_utf16_lossy(&sys[..n])))
}

/// Streamline's files, NVIDIA's driver library and the port's version, as a line
/// (`streamline-ok.txt` and `streamline-crashed.txt` hold the line they were written for: a check
/// is made again once it differs, a driver update included).
fn streamline_files_line(dir: &Path) -> String {
    let mut v = vec![format!("Regaliant {}", env!("REGALIANT_VERSION"))];
    let files = ["nvngx_dlss.dll", "sl.interposer.dll", "sl.common.dll", "sl.dlss.dll"].map(|f| dir.join(f));
    for p in files.iter().cloned().chain(system_dir().map(|s| s.join("nvapi64.dll"))) {
        let m = std::fs::metadata(&p).ok();
        let t = m.as_ref().and_then(|m| m.modified().ok()).and_then(|t| t.duration_since(std::time::UNIX_EPOCH).ok()).map_or(0, |d| d.as_secs());
        let name = p.file_name().map_or(String::new(), |n| n.to_string_lossy().into_owned());
        v.push(format!("{name} {} {t}", m.map_or(0, |m| m.len())));
    }
    v.join("; ")
}

/// When process `pid` was created (FILETIME ticks; None: not found). A marker's process is the
/// one that left it only while this matches: Windows reuses the number of an ended process.
fn process_created(pid: u32) -> Option<u64> {
    use windows_sys::Win32::Foundation::{CloseHandle, FILETIME};
    use windows_sys::Win32::System::Threading::{GetProcessTimes, OpenProcess, PROCESS_QUERY_LIMITED_INFORMATION};
    unsafe {
        let h = OpenProcess(PROCESS_QUERY_LIMITED_INFORMATION, 0, pid);
        if h.is_null() {
            return None;
        }
        let z = FILETIME { dwLowDateTime: 0, dwHighDateTime: 0 };
        let (mut c, mut e, mut k, mut u) = (z, z, z, z);
        let ok = GetProcessTimes(h, &mut c, &mut e, &mut k, &mut u) != 0;
        CloseHandle(h);
        ok.then(|| ((c.dwHighDateTime as u64) << 32) | c.dwLowDateTime as u64)
    }
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
/// in a row for the same files: at `STREAMLINE_DEATHS` DLSS is left out. One alone is no proof: a
/// launcher check or preparation step is ended from outside when the launcher closes or a check
/// takes too long, and such an end in the load must not switch DLSS off (the review of 2026-11-24
/// night); it only makes the separate check run again.
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
        crate::warn!("upscaler: {what} ({n} in a row with these files): DLSS is left out");
        let _ = std::fs::write(cache.join("streamline-crashed.txt"), line.trim());
        let _ = std::fs::remove_file(&deaths);
    } else {
        crate::warn!("upscaler: {what}: Streamline's separate check runs again before DLSS is loaded");
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
/// why DLSS is left out, if it is (`streamline-crashed.txt` written for these files).
fn streamline_crashed_before(dir: &Path) -> Option<String> {
    let cache = crate::paths().cache.clone();
    let crashed = cache.join("streamline-crashed.txt");
    if let Ok(rd) = std::fs::read_dir(&cache) {
        for e in rd.flatten() {
            let name = e.file_name().to_string_lossy().into_owned();
            let Some(pid) = name.strip_prefix("streamline-loading-").and_then(|s| s.strip_suffix(".txt")).and_then(|s| s.parse::<u32>().ok()) else { continue };
            if pid == std::process::id() {
                continue;
            }
            // (The files' line, then the process's creation time: a live process with that
            // number and time is still on trial, however long. A marker without the time, from
            // 0.1.7 and before, counts as its process's after five minutes.)
            let text = std::fs::read_to_string(e.path()).unwrap_or_default();
            let (line, created) = text.split_once('\n').map_or((text.as_str(), 0), |(l, c)| (l, c.trim().parse::<u64>().unwrap_or(0)));
            let gone = if created != 0 {
                !process_alive(pid) || process_created(pid) != Some(created)
            } else {
                let old = e.metadata().ok().and_then(|m| m.modified().ok()).and_then(|t| t.elapsed().ok()).is_some_and(|a| a.as_secs() > 300);
                old || !process_alive(pid)
            };
            if !gone {
                continue;
            }
            let claimed = cache.join(format!("streamline-loading-{pid}.claimed-{}", std::process::id()));
            if std::fs::rename(e.path(), &claimed).is_err() {
                continue;
            }
            let line = line.to_string();
            let _ = std::fs::remove_file(&claimed);
            note_streamline_death(&cache, &line, &format!("an earlier start (process {pid}) ended while it loaded Streamline (DLSS)"));
        }
    }
    let was = std::fs::read_to_string(&crashed).ok()?;
    if was.trim() == streamline_files_line(dir) {
        Some(format!(
            "Streamline's load ended starts or its separate check with these files, so DLSS is left out (it is tried again when they, the graphics driver or Regaliant's version change, or after deleting {})",
            crashed.display()
        ))
    } else {
        let _ = std::fs::remove_file(&crashed);
        None
    }
}

fn load_sl() -> Result<SlLib, LoadError> {
    let dir = std::env::var_os("DS_SL_DIR").map(PathBuf::from).unwrap_or_else(lib_dir);
    // DLSS itself: the player's own nvngx_dlss.dll (from a game with DLSS or
    // NVIDIA's DLSS SDK; NVIDIA's license keeps it out of the package), signed
    // by NVIDIA. Without it Streamline is not started.
    let ngx = dir.join("nvngx_dlss.dll");
    if !ngx.is_file() {
        return Err(LoadError::Missing);
    }
    // (Without NVIDIA's driver Streamline has nothing to run on: its library nvapi64.dll is in
    // the system folder with every NVIDIA driver.)
    if system_dir().is_some_and(|s| !s.join("nvapi64.dll").is_file()) {
        return Err("no NVIDIA graphics driver here (nvapi64.dll is not in the system folder): DLSS needs an NVIDIA RTX card".to_string().into());
    }
    check_signature(&ngx, "NVIDIA Corporation")?;
    if let Some(why) = streamline_crashed_before(&dir) {
        return Err(why.into());
    }
    let cache = crate::paths().cache.clone();
    let _ = std::fs::create_dir_all(&cache);
    let line = streamline_files_line(&dir);
    // A start loads Streamline itself only once a separate process has loaded these files
    // (`probe_streamline`; `streamline-ok.txt` holds their line): the player's DLSS runtime that
    // ended every start ended only that process, and the game starts without DLSS.
    let ok = cache.join("streamline-ok.txt");
    if std::fs::read_to_string(&ok).ok().as_deref().map(str::trim) != Some(line.as_str()) {
        let t0 = std::time::Instant::now();
        match run_probe(&dir) {
            ProbeEnd::Ended(code) => {
                crate::info!("upscaler: Streamline's separate check loaded these files in {:.2} s (exit {code})", t0.elapsed().as_secs_f32());
                let _ = std::fs::write(&ok, &line);
            }
            // (A fault's NTSTATUS: the load itself crashed, not an end from outside.)
            ProbeEnd::Faulted(code) => {
                let crashed = cache.join("streamline-crashed.txt");
                let _ = std::fs::write(&crashed, &line);
                return Err(format!(
                    "Streamline's separate check of these files crashed (code {code:#x}), so DLSS is left out (it is tried again when they, the graphics driver or Regaliant's version change, or after deleting {})",
                    crashed.display()
                )
                .into());
            }
            ProbeEnd::Other(why) => {
                note_streamline_death(&cache, &line, &format!("Streamline's separate check {why}"));
                return Err(format!("Streamline's separate check {why}: DLSS is left out for this start").into());
            }
        }
    }
    // (Left behind only if the process dies while Streamline is on trial: from its load to
    // DLSS's first frame presented through it, `streamline_settled`. Until 0.1.7 it covered the
    // load alone, and a player's start that died in DLSS's support query right after it, with
    // their own nvngx_dlss.dll, died the same way at every start: 2026-11-25.)
    let marker = cache.join(format!("streamline-loading-{}.txt", std::process::id()));
    let _ = std::fs::write(&marker, format!("{line}\n{}", process_created(std::process::id()).unwrap_or(0)));
    // (Diagnosis: `DS_SL_CRASH_TEST=2` ends the process here, as a crash in the load would; `1`
    // ends the separate check instead.)
    if std::env::var_os("DS_SL_CRASH_TEST").is_some_and(|v| v == "2") {
        crate::warn!("upscaler: DS_SL_CRASH_TEST: ending the process while Streamline loads");
        std::process::abort();
    }
    let r = load_sl_from(&dir);
    if r.is_ok() {
        *SL_MARKER.lock() = Some(marker);
    } else {
        let _ = std::fs::remove_file(&marker);
    }
    r
}

/// This start's marker while Streamline is on trial (`load_sl`).
static SL_MARKER: parking_lot::Mutex<Option<PathBuf>> = parking_lot::Mutex::new(None);

/// Streamline's trial in this start ended without a death (`why`: DLSS presented a frame through
/// it, failed without a crash, or the start ended cleanly first): the marker goes and so do the
/// deaths in a row.
pub fn streamline_settled(why: &str) {
    let Some(m) = SL_MARKER.lock().take() else { return };
    let _ = std::fs::remove_file(&m);
    let _ = std::fs::remove_file(crate::paths().cache.join("streamline-deaths.txt"));
    crate::info!("upscaler: Streamline's start went through ({why})");
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
/// from `dir` as a start does (its library, slInit with DLSS, DLSS's requirements) and ends at
/// once (no library's exit code runs): `PROBE_LOADED`, or `PROBE_REFUSED` when it refused without
/// a fault; at `PROBE_SECS` it ends itself (`PROBE_TIMED_OUT`), so a load that hangs does not
/// outlive the start that waits for it. A fault in it ends only this process (`load_sl`).
pub fn probe_streamline(dir: &Path) -> ! {
    use windows_sys::Win32::System::Threading::{GetCurrentProcess, TerminateProcess};
    std::thread::spawn(|| {
        std::thread::sleep(std::time::Duration::from_secs(PROBE_SECS));
        unsafe { TerminateProcess(GetCurrentProcess(), PROBE_TIMED_OUT) };
    });
    if std::env::var_os("DS_SL_CRASH_TEST").is_some_and(|v| v == "1") {
        std::process::abort();
    }
    let code = match load_sl_from(dir) {
        Ok(lib) => {
            probe_support(&lib);
            PROBE_LOADED
        }
        Err(_) => PROBE_REFUSED,
    };
    unsafe { TerminateProcess(GetCurrentProcess(), code) };
    std::process::exit(code as i32)
}

/// The separate check's second step: DLSS's support query (slIsFeatureSupported, where NGX loads
/// the player's nvngx_dlss.dll) on each NVIDIA card of a Vulkan instance made with DLSS's
/// instance extensions, as a start makes it before its device. A start of 0.1.7 died there right
/// after a check that only loaded the files (2026-11-25). Its answer does not matter here, only
/// that the process survives it.
fn probe_support(lib: &SlLib) {
    let Ok(entry) = (unsafe { ash::Entry::load() }) else { return };
    let avail = unsafe { entry.enumerate_instance_extension_properties(None) }.unwrap_or_default();
    let has = |e: &CStr| avail.iter().any(|x| unsafe { CStr::from_ptr(x.extension_name.as_ptr()) } == e);
    let exts: Vec<*const std::ffi::c_char> = lib.reqs.instance_ext.iter().filter(|e| has(e)).map(|e| e.as_ptr()).collect();
    let app = vk::ApplicationInfo::default().api_version(vk::API_VERSION_1_3);
    let ici = vk::InstanceCreateInfo::default().application_info(&app).enabled_extension_names(&exts);
    let Ok(instance) = (unsafe { entry.create_instance(&ici, None) }) else { return };
    for pdev in unsafe { instance.enumerate_physical_devices() }.unwrap_or_default() {
        if unsafe { instance.get_physical_device_properties(pdev) }.vendor_id != 0x10DE {
            continue;
        }
        let info = SlAdapterInfo { base: SlBase::new(SL_ADAPTER_INFO, 1), luid: std::ptr::null(), luid_size: 0, vk_physical_device: pdev };
        // (Diagnosis: `DS_SL_CRASH_TEST=4` ends the check here.)
        if std::env::var_os("DS_SL_CRASH_TEST").is_some_and(|v| v == "4") {
            std::process::abort();
        }
        let _ = unsafe { (lib.is_supported)(SL_FEATURE_DLSS, &info) };
    }
}

fn load_sl_from(dir: &Path) -> Result<SlLib, LoadError> {
    let dir = dir.to_path_buf();
    let path = dir.join("sl.interposer.dll");
    let h = load_lib(&path, "NVIDIA Corporation")?;
    let f = |name: &CStr| export(h, &path, name);
    // SAFETY: the signatures of sl_core_api.h / sl_helpers_vk.h.
    unsafe {
        let init: unsafe extern "C" fn(*const SlPreferences, u64) -> i32 = std::mem::transmute(f(c"slInit")?);
        let requirements: unsafe extern "C" fn(u32, *mut SlFeatureRequirements) -> i32 = std::mem::transmute(f(c"slGetFeatureRequirements")?);
        let mut lib = SlLib {
            is_supported: std::mem::transmute(f(c"slIsFeatureSupported")?),
            set_vulkan_info: std::mem::transmute(f(c"slSetVulkanInfo")?),
            frame_token: std::mem::transmute(f(c"slGetNewFrameToken")?),
            set_constants: std::mem::transmute(f(c"slSetConstants")?),
            evaluate: std::mem::transmute(f(c"slEvaluateFeature")?),
            feature_function: std::mem::transmute(f(c"slGetFeatureFunction")?),
            version: std::mem::transmute(f(c"slGetFeatureVersion")?),
            gipa: std::mem::transmute(f(c"vkGetInstanceProcAddr")?),
            reqs: DlssReqs::default(),
        };
        // slInit: DLSS only, the port restores its own command buffer state
        // (it does after any foreign recording), manual hooking, no
        // over-the-air downloads, logs into the port's log.
        let t0 = std::time::Instant::now();
        let dir_w = dir.as_os_str().encode_wide_null();
        let paths = [dir_w.as_ptr()];
        let features = [SL_FEATURE_DLSS];
        let prefs = SlPreferences {
            base: SlBase::new(SL_PREFERENCES, 1),
            show_console: false,
            log_level: 1,
            paths_to_plugins: paths.as_ptr(),
            num_paths_to_plugins: 1,
            path_to_logs_and_data: std::ptr::null(),
            allocate_callback: std::ptr::null(),
            release_callback: std::ptr::null(),
            log_message_callback: Some(sl_log),
            flags: SL_DISABLE_CL_STATE_TRACKING | SL_USE_MANUAL_HOOKING,
            features_to_load: features.as_ptr(),
            num_features_to_load: 1,
            application_id: 0,
            engine: 0,
            engine_version: c"1.0".as_ptr(),
            project_id: NGX_PROJECT_ID.as_ptr(),
            render_api: SL_VULKAN,
        };
        let r = init(&prefs, SL_SDK_VERSION);
        if r != SL_OK {
            return Err(format!("slInit failed ({r})").into());
        }
        let mut reqs: SlFeatureRequirements = std::mem::zeroed();
        reqs.base = SlBase::new(SL_FEATURE_REQUIREMENTS, 2);
        let r = requirements(SL_FEATURE_DLSS, &mut reqs);
        if r != SL_OK {
            return Err(format!("no DLSS here ({r}): it needs an NVIDIA RTX GPU and driver").into());
        }
        // (FeatureRequirementFlags::eVulkanSupported.)
        if reqs.flags & 4 == 0 {
            return Err("DLSS does not support Vulkan here".to_string().into());
        }
        let v = |x: [u32; 3]| format!("{}.{}.{}", x[0], x[1], x[2]);
        crate::info!(
            "upscaler: Streamline loaded from {} in {:.2} s (OS {} needs {}, driver {} needs {})",
            dir.display(),
            t0.elapsed().as_secs_f32(),
            v(reqs.os_detected),
            v(reqs.os_required),
            v(reqs.driver_detected),
            v(reqs.driver_required)
        );
        let strings = |n: u32, p: *const *const c_char| names(n, p).into_iter().map(|c| c.to_string_lossy().into_owned()).collect::<Vec<_>>();
        lib.reqs = DlssReqs {
            instance_ext: names(reqs.vk_num_instance_extensions, reqs.vk_instance_extensions),
            device_ext: names(reqs.vk_num_device_extensions, reqs.vk_device_extensions),
            features12: strings(reqs.vk_num_features12, reqs.vk_features12),
            features13: strings(reqs.vk_num_features13, reqs.vk_features13),
            queues: reqs.vk_compute_queues + reqs.vk_graphics_queues + reqs.vk_optical_flow_queues,
        };
        crate::info!(
            "upscaler: DLSS asks for instance extensions {:?}, device extensions {:?}, Vulkan 1.2 features {:?}, 1.3 features {:?}, {} extra queues",
            lib.reqs.instance_ext,
            lib.reqs.device_ext,
            lib.reqs.features12,
            lib.reqs.features13,
            lib.reqs.queues
        );
        Ok(lib)
    }
}

fn dlss_device_needs(instance: &ash::Instance, pdev: vk::PhysicalDevice) -> Option<DeviceNeeds> {
    let s = sl()?;
    if DLSS_FAILED.load(Relaxed) {
        return None;
    }
    let info = SlAdapterInfo { base: SlBase::new(SL_ADAPTER_INFO, 1), luid: std::ptr::null(), luid_size: 0, vk_physical_device: pdev };
    // (Diagnosis: `DS_SL_CRASH_TEST=3` ends the process here, as the player's DLSS file did in
    // 0.1.7, 2026-11-25.)
    if std::env::var_os("DS_SL_CRASH_TEST").is_some_and(|v| v == "3") {
        crate::warn!("upscaler: DS_SL_CRASH_TEST: ending the process at DLSS's support query");
        crate::log::flush();
        std::process::abort();
    }
    // (Diagnosis: `DS_SL_CRASH_TEST=5` breaks here only for a debugger, as NVIDIA's code does:
    // with the host's flag hidden (`hosted`) the start goes on, with `DS_KEEP_DEBUGGER_FLAG=1` it
    // ends here and the host writes where into the log.)
    if std::env::var_os("DS_SL_CRASH_TEST").is_some_and(|v| v == "5") {
        use windows_sys::Win32::System::Diagnostics::Debug::{DebugBreak, IsDebuggerPresent};
        let debugged = unsafe { IsDebuggerPresent() } != 0;
        crate::warn!("upscaler: DS_SL_CRASH_TEST=5: a break at DLSS's support query only for a debugger ({})", if debugged { "one is seen: breaking" } else { "none is seen" });
        crate::log::flush();
        if debugged {
            unsafe { DebugBreak() };
        }
    }
    let r = unsafe { (s.is_supported)(SL_FEATURE_DLSS, &info) };
    if r != SL_OK {
        fail(Kind::Dlss, format!("not supported on this GPU or driver (Streamline result {r})"));
        return None;
    }
    if s.reqs.queues > 0 || !s.reqs.features13.is_empty() {
        fail(Kind::Dlss, format!("it asks for {} extra queues and 1.3 features {:?}, which the port does not provide", s.reqs.queues, s.reqs.features13));
        return None;
    }
    if let Err(e) = features_supported(instance, pdev, &s.reqs.features12) {
        fail(Kind::Dlss, e);
        return None;
    }
    // The device's extensions. (VK_EXT_buffer_device_address must not be
    // enabled with the 1.2 bufferDeviceAddress feature: the KHR extension
    // in its place, as Streamline's own device creation does.)
    let avail = unsafe { instance.enumerate_device_extension_properties(pdev) }.unwrap_or_default();
    let has = |n: &CStr| avail.iter().any(|e| unsafe { CStr::from_ptr(e.extension_name.as_ptr()) } == n);
    let mut extensions = Vec::new();
    for e in &s.reqs.device_ext {
        let e = if e.as_c_str() == ash::ext::buffer_device_address::NAME { ash::khr::buffer_device_address::NAME.to_owned() } else { e.clone() };
        if !has(&e) {
            fail(Kind::Dlss, format!("the device lacks {}", e.to_string_lossy()));
            return None;
        }
        if !extensions.contains(&e) {
            extensions.push(e);
        }
    }
    Some(DeviceNeeds { extensions, mutable_descriptor: false, features: s.reqs.features12.clone() })
}

/// Hands the device to Streamline (slSetVulkanInfo, which starts NGX) on a
/// thread of its own; DLSS is ready - offered and its present hook used -
/// once that is done.
fn dlss_device_created(instance: &ash::Instance, pdev: vk::PhysicalDevice, device: &ash::Device, qf: u32, queues: u32) {
    let Some(s) = sl() else { return };
    // (No extra queues: Streamline's would start after the port's.)
    let info = SlVulkanInfo {
        base: SlBase::new(SL_VULKAN_INFO, 3),
        device: device.handle(),
        instance: instance.handle(),
        physical_device: pdev,
        compute_queue_index: queues,
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
    let spawned = std::thread::Builder::new().name("dlss-init".into()).spawn(move || {
        let t0 = std::time::Instant::now();
        let r = unsafe { (s.set_vulkan_info)(&info) };
        if r != SL_OK {
            fail(Kind::Dlss, format!("slSetVulkanInfo failed ({r})"));
            return;
        }
        // Streamline's vkQueuePresentKHR, through its vkGetDeviceProcAddr.
        // SAFETY: Streamline's loader functions, valid for this device.
        let present = unsafe {
            let gdpa: Option<vk::PFN_vkGetDeviceProcAddr> = std::mem::transmute((s.gipa)(info.instance, c"vkGetDeviceProcAddr".as_ptr()));
            gdpa.and_then(|g| g(info.device, c"vkQueuePresentKHR".as_ptr())).map(|p| std::mem::transmute::<unsafe extern "system" fn(), vk::PFN_vkQueuePresentKHR>(p))
        };
        let Some(present) = present else {
            fail(Kind::Dlss, "no vkQueuePresentKHR from Streamline");
            return;
        };
        let mut ver: SlFeatureVersion = unsafe { std::mem::zeroed() };
        ver.base = SlBase::new(SL_FEATURE_VERSION, 1);
        let ngx = if unsafe { (s.version)(SL_FEATURE_DLSS, &mut ver) } == SL_OK { format!(", NGX DLSS {}.{}.{}", ver.ngx[0], ver.ngx[1], ver.ngx[2]) } else { String::new() };
        // (Ready before its present is published: the first present through Streamline then
        // finds DLSS ready, and `note_sl_present` logs it in use; the review of 2026-11-25.)
        DLSS_READY.store(true, SeqCst);
        crate::info!("upscaler: DLSS ready in {:.2} s (Streamline {}.{}.{}{ngx}): in use from the first present through Streamline", t0.elapsed().as_secs_f32(), ver.sl[0], ver.sl[1], ver.sl[2]);
        let _ = SL_PRESENT.set(present);
    });
    if let Err(e) = spawned {
        fail(Kind::Dlss, format!("no thread for its start ({e})"));
    }
}

/// slDLSSSetOptions (from the plugin, once the device is set).
type DlssSetOptions = unsafe extern "C" fn(*const SlViewport, *const SlDlssOptions) -> i32;

struct DlssState {
    set_options: DlssSetOptions,
    key: (u32, u32, u32, u32, bool),
    frame: u32,
}

static DLSS_STATE: Mutex<Option<DlssState>> = Mutex::new(None);

fn sl_resource(t: &Tex) -> SlResource {
    use ash::vk::Handle;
    SlResource {
        base: SlBase::new(SL_RESOURCE, 1),
        ty: 0,
        native: t.image.as_raw(),
        memory: 0,
        view: t.view.as_raw(),
        state: t.layout.as_raw() as u32,
        width: t.size.0,
        height: t.size.1,
        native_format: t.format.as_raw() as u32,
        mip_levels: 1,
        array_layers: 1,
        gpu_virtual_address: 0,
        flags: 0,
        usage: t.usage.as_raw(),
        internal_flags: 0,
        reserved: 0,
    }
}

fn dlss_record(cmd: vk::CommandBuffer, inp: &Inputs, reset: bool) -> bool {
    let Some(s) = sl() else { return false };
    let mut st = DLSS_STATE.lock();
    if st.is_none() {
        let mut p: *const c_void = std::ptr::null();
        let r = unsafe { (s.feature_function)(SL_FEATURE_DLSS, c"slDLSSSetOptions".as_ptr(), &mut p) };
        if r != SL_OK || p.is_null() {
            fail(Kind::Dlss, format!("no slDLSSSetOptions ({r})"));
            return false;
        }
        // SAFETY: sl_dlss.h's PFun_slDLSSSetOptions.
        let set_options: DlssSetOptions = unsafe { std::mem::transmute(p) };
        *st = Some(DlssState { set_options, key: (0, 0, 0, 0, false), frame: 0 });
    }
    let d = st.as_mut().unwrap();
    let key = (inp.in_width, inp.in_height, inp.width, inp.height, inp.ldr);
    let new_size = d.key != key;
    let mode = dlss_mode(inp.ratio());
    if new_size {
        if inp.upscales() {
            crate::info!("upscaler: DLSS upscaling {}x{} to {}x{} (mode {mode}) after the game's TAA", inp.in_width, inp.in_height, inp.width, inp.height);
        } else {
            crate::info!("upscaler: DLSS native anti-aliasing (DLAA) at {}x{} after the game's TAA", inp.width, inp.height);
        }
        d.key = key;
    }
    let (jx, jy) = (inp.jitter_x, inp.jitter_y);
    let (sx, sy) = jitter_scale();
    // The game's motion vectors: screen fraction, current - previous;
    // Streamline's: scaled into [-1, 1] screen fractions, current ->
    // previous.
    let (mx, my) = mv_scale();
    let reset = reset || d.frame == 0 || new_size;
    d.frame += 1;
    let set_options = d.set_options;
    drop(st);
    let consts = SlConstants {
        base: SlBase::new(SL_CONSTANTS, 2),
        // (DLSS's upscaling reads the jitter and the motion vectors; the
        // camera's matrices and vectors are for frame generation: neutral
        // values.)
        camera_view_to_clip: IDENTITY,
        clip_to_camera_view: IDENTITY,
        clip_to_lens_clip: IDENTITY,
        clip_to_prev_clip: IDENTITY,
        prev_clip_to_clip: IDENTITY,
        // (DLSS's jitter: the image's shift, as rendered.)
        jitter_offset: [jx * sx, jy * sy],
        mvec_scale: [-mx, -my],
        camera_pinhole_offset: [0.0, 0.0],
        camera_pos: [0.0, 0.0, 0.0],
        camera_up: [0.0, 1.0, 0.0],
        camera_right: [1.0, 0.0, 0.0],
        camera_fwd: [0.0, 0.0, 1.0],
        camera_near: 0.1,
        camera_far: 10000.0,
        camera_fov: 1.0,
        camera_aspect_ratio: inp.in_width as f32 / inp.in_height.max(1) as f32,
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
    };
    let opts = SlDlssOptions {
        base: SlBase::new(SL_DLSS_OPTIONS, 3),
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
    let (frame_token, set_constants, evaluate) = (s.frame_token, s.set_constants, s.evaluate);
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
            let viewport = SlViewport { base: SlBase::new(SL_VIEWPORT, 1), value: 0 };
            let mut token: *const c_void = std::ptr::null();
            EVAL_OK.store(false, Relaxed);
            let r = frame_token(&mut token, std::ptr::null());
            if r != SL_OK || token.is_null() {
                warn_some(&N_TOKEN, "slGetNewFrameToken", r);
                return;
            }
            let r = set_constants(&consts, token, &viewport);
            if r != SL_OK {
                warn_some(&N_CONST, "slSetConstants", r);
            }
            let r = set_options(&viewport, &opts);
            if r != SL_OK {
                warn_some(&N_OPT, "slDLSSSetOptions", r);
            }
            let tag = |i: usize, ty: u32| SlResourceTag { base: SlBase::new(SL_RESOURCE_TAG, 1), resource: &res[i], ty, lifecycle: SL_VALID_UNTIL_EVALUATE, extent: if i == 3 { out_extent } else { extent } };
            let tags = [tag(0, SL_DEPTH), tag(1, SL_MOTION_VECTORS), tag(2, SL_SCALING_INPUT_COLOR), tag(3, SL_SCALING_OUTPUT_COLOR)];
            let inputs: [*const SlBase; 5] = [&viewport.base, &tags[0].base, &tags[1].base, &tags[2].base, &tags[3].base];
            let r = evaluate(SL_FEATURE_DLSS, token, inputs.as_ptr(), inputs.len() as u32, cb);
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
    Some(DeviceNeeds { extensions: Vec::new(), mutable_descriptor: false, features })
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
        if inp.upscales() {
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

    /// Streamline's structures as its MSVC build lays them out (offsets
    /// from the SDK's headers, e.g. its own static_assert on VulkanInfo).
    #[test]
    fn streamline_layouts() {
        use std::mem::{offset_of, size_of};
        assert_eq!(size_of::<SlBase>(), 32);
        assert_eq!(size_of::<SlPreferences>(), 144);
        assert_eq!(offset_of!(SlPreferences, flags), 88);
        assert_eq!(offset_of!(SlPreferences, render_api), 136);
        assert_eq!(size_of::<SlResource>(), 112);
        assert_eq!(offset_of!(SlResource, native), 40);
        assert_eq!(offset_of!(SlResource, state), 64);
        assert_eq!(offset_of!(SlResource, gpu_virtual_address), 88);
        assert_eq!(size_of::<SlResourceTag>(), 64);
        assert_eq!(offset_of!(SlResourceTag, extent), 48);
        assert_eq!(size_of::<SlViewport>(), 40);
        assert_eq!(size_of::<SlFeatureRequirements>(), 184);
        assert_eq!(offset_of!(SlFeatureRequirements, vk_device_extensions), 120);
        assert_eq!(offset_of!(SlFeatureRequirements, vk_optical_flow_queues), 176);
        assert_eq!(size_of::<SlAdapterInfo>(), 56);
        assert_eq!(size_of::<SlVulkanInfo>(), 96);
        assert_eq!(offset_of!(SlVulkanInfo, use_native_optical_flow), 80);
        assert_eq!(size_of::<SlConstants>(), 456);
        assert_eq!(offset_of!(SlConstants, jitter_offset), 352);
        assert_eq!(offset_of!(SlConstants, depth_inverted), 444);
        assert_eq!(offset_of!(SlConstants, min_relative_linear_depth_object_separation), 452);
        assert_eq!(size_of::<SlDlssOptions>(), 88);
        assert_eq!(offset_of!(SlDlssOptions, dlaa_preset), 60);
        assert_eq!(offset_of!(SlDlssOptions, use_auto_exposure), 84);
        assert_eq!(size_of::<SlFeatureVersion>(), 56);
        assert_eq!(SL_SDK_VERSION, 0x0002_000e_0010_fedc);
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
