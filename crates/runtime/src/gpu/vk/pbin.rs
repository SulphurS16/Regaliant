//! VK_KHR_pipeline_binary (the driver's pipeline binaries in the port's own
//! pipeline store, `gpu::pcache`): the structures and commands, which ash
//! 0.38 (Vulkan 1.3.281) predates. Declarations from the Vulkan reference
//! pages (docs.vulkan.org, PROVENANCE.md); the structure type numbers are
//! checked against the driver when the device is made (`supported`: its
//! properties and features read back as vulkaninfo reports them).
//!
//! On the RTX 4090's driver (610.88) a query of the global key
//! (vkGetPipelineKeyKHR without a create info) made every later
//! vkCreatePipelineBinariesKHR fault in the driver (outside the validation
//! layer; `dsport --pb-test`): it is not made, the store keys binaries by the
//! device's pipeline cache UUID and driver version instead (`Fns::global`).

use ash::vk;
use std::ffi::{CStr, c_void};

pub const NAME: &CStr = c"VK_KHR_pipeline_binary";
pub const ST_FEATURES: i32 = 1000483000;
pub const ST_CREATE_INFO: i32 = 1000483001;
pub const ST_BINARY_INFO: i32 = 1000483002;
pub const ST_KEY: i32 = 1000483003;
pub const ST_PROPERTIES: i32 = 1000483004;
pub const ST_RELEASE_INFO: i32 = 1000483005;
pub const ST_DATA_INFO: i32 = 1000483006;
pub const ST_PIPELINE_CREATE_INFO: i32 = 1000483007;
pub const ST_CACHE_CONTROL: i32 = 1000483008;
pub const ST_HANDLES_INFO: i32 = 1000483009;
/// VK_MAX_PIPELINE_BINARY_KEY_SIZE_KHR.
pub const MAX_KEY: usize = 32;
/// VK_PIPELINE_CREATE_2_CAPTURE_DATA_BIT_KHR.
pub const CAPTURE_DATA: u64 = 0x8000_0000;

/// Keep the flags2 node on both capture and replay. Only CAPTURE_DATA may differ;
/// dropping the entire node changes the pNext chain recorded by the driver.
pub fn creation_flags(flags: vk::PipelineCreateFlags, capture: bool) -> vk::PipelineCreateFlags2CreateInfoKHR<'static> {
    vk::PipelineCreateFlags2CreateInfoKHR::default().flags(vk::PipelineCreateFlags2KHR::from_raw(flags.as_raw() as u64 | if capture { CAPTURE_DATA } else { 0 }))
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn capture_and_replay_flags_differ_only_by_capture_bit() {
        for flags in [vk::PipelineCreateFlags::empty(), vk::PipelineCreateFlags::DISPATCH_BASE] {
            let a = creation_flags(flags, true);
            let b = creation_flags(flags, false);
            assert_eq!(a.s_type, b.s_type);
            assert_eq!(a.flags.as_raw() ^ b.flags.as_raw(), CAPTURE_DATA);
            assert_eq!(b.flags.as_raw(), flags.as_raw() as u64);
            assert!(!b.flags.intersects(vk::PipelineCreateFlags2KHR::FAIL_ON_PIPELINE_COMPILE_REQUIRED));
        }
    }
}

/// VkPipelineBinaryKHR (a non-dispatchable handle).
pub type Binary = u64;

#[repr(C)]
pub struct Features {
    pub s_type: i32,
    pub p_next: *mut c_void,
    pub pipeline_binaries: u32,
}

#[repr(C)]
pub struct Properties {
    pub s_type: i32,
    pub p_next: *mut c_void,
    pub internal_cache: u32,
    pub internal_cache_control: u32,
    pub prefers_internal_cache: u32,
    pub precompiled_internal_cache: u32,
    pub compressed_data: u32,
}

#[repr(C)]
pub struct CacheControl {
    pub s_type: i32,
    pub p_next: *const c_void,
    pub disable_internal_cache: u32,
}

#[repr(C)]
#[derive(Clone, Copy)]
pub struct Key {
    pub s_type: i32,
    pub p_next: *mut c_void,
    pub key_size: u32,
    pub key: [u8; MAX_KEY],
}

impl Key {
    pub fn new() -> Key {
        Key { s_type: ST_KEY, p_next: std::ptr::null_mut(), key_size: 0, key: [0; MAX_KEY] }
    }

    pub fn bytes(&self) -> &[u8] {
        &self.key[..(self.key_size as usize).min(MAX_KEY)]
    }
}

#[repr(C)]
pub struct Data {
    pub data_size: usize,
    pub p_data: *mut c_void,
}

#[repr(C)]
pub struct KeysAndData {
    pub binary_count: u32,
    pub p_keys: *const Key,
    pub p_data: *const Data,
}

#[repr(C)]
pub struct PipelineCreateInfo {
    pub s_type: i32,
    pub p_next: *mut c_void,
}

#[repr(C)]
pub struct CreateInfo {
    pub s_type: i32,
    pub p_next: *const c_void,
    pub p_keys_and_data: *const KeysAndData,
    pub pipeline: vk::Pipeline,
    pub p_pipeline_create_info: *const PipelineCreateInfo,
}

/// VkPipelineBinaryInfoKHR: extends VkComputePipelineCreateInfo and
/// VkGraphicsPipelineCreateInfo (the pipeline made from these binaries).
#[repr(C)]
pub struct BinaryInfo {
    pub s_type: i32,
    pub p_next: *const c_void,
    pub binary_count: u32,
    pub p_binaries: *const Binary,
}

#[repr(C)]
pub struct ReleaseInfo {
    pub s_type: i32,
    pub p_next: *mut c_void,
    pub pipeline: vk::Pipeline,
}

#[repr(C)]
pub struct DataInfo {
    pub s_type: i32,
    pub p_next: *mut c_void,
    pub binary: Binary,
}

#[repr(C)]
pub struct HandlesInfo {
    pub s_type: i32,
    pub p_next: *const c_void,
    pub count: u32,
    pub p_binaries: *mut Binary,
}

type CreateFn = unsafe extern "system" fn(vk::Device, *const CreateInfo, *const c_void, *mut HandlesInfo) -> vk::Result;
type DestroyFn = unsafe extern "system" fn(vk::Device, Binary, *const c_void);
type DataFn = unsafe extern "system" fn(vk::Device, *const DataInfo, *mut Key, *mut usize, *mut c_void) -> vk::Result;
type ReleaseFn = unsafe extern "system" fn(vk::Device, *const ReleaseInfo, *const c_void) -> vk::Result;

/// The extension's commands on the device, and the store's compatibility
/// key (binaries stored under another one are not used): the device's
/// pipeline cache UUID, driver version, vendor/device ids and enabled device contract.
pub struct Fns {
    device: vk::Device,
    create: CreateFn,
    destroy: DestroyFn,
    data: DataFn,
    release: ReleaseFn,
    pub global: Vec<u8>,
    /// Binaries every pipeline shares (the driver's 2 MB dictionary), made
    /// once a session from their stored data, by key.
    shared: parking_lot::Mutex<Vec<(Vec<u8>, Binary)>>,
}

/// A pipeline's binary as stored: its own key and data, or the key of a
/// shared one (`Fns::share`).
pub enum Part {
    Own(Vec<u8>, Vec<u8>),
    Shared(Vec<u8>),
}

/// Whether the device has the extension and its feature, by the structures
/// above read back (properties as vulkaninfo gives them: an internal cache;
/// features: pipelineBinaries).
pub unsafe fn supported(instance: &ash::Instance, pdev: vk::PhysicalDevice) -> bool {
    let has_ext = unsafe { instance.enumerate_device_extension_properties(pdev) }.unwrap_or_default().iter().any(|e| unsafe { CStr::from_ptr(e.extension_name.as_ptr()) } == NAME);
    if !has_ext {
        return false;
    }
    let mut props = Properties { s_type: ST_PROPERTIES, p_next: std::ptr::null_mut(), internal_cache: 0xFFFF, internal_cache_control: 0xFFFF, prefers_internal_cache: 0xFFFF, precompiled_internal_cache: 0xFFFF, compressed_data: 0xFFFF };
    let mut p2 = vk::PhysicalDeviceProperties2::default();
    p2.p_next = &mut props as *mut Properties as *mut c_void;
    unsafe { instance.get_physical_device_properties2(pdev, &mut p2) };
    let mut feats = Features { s_type: ST_FEATURES, p_next: std::ptr::null_mut(), pipeline_binaries: 0xFFFF };
    let mut f2 = vk::PhysicalDeviceFeatures2::default();
    f2.p_next = &mut feats as *mut Features as *mut c_void;
    unsafe { instance.get_physical_device_features2(pdev, &mut f2) };
    let bools = [props.internal_cache, props.internal_cache_control, props.prefers_internal_cache, props.precompiled_internal_cache, props.compressed_data, feats.pipeline_binaries];
    crate::info!(
        "Vulkan pipeline binaries: feature {}, internal cache {}, its control {}, prefers it {}, precompiled {}, compressed {}",
        feats.pipeline_binaries,
        props.internal_cache,
        props.internal_cache_control,
        props.prefers_internal_cache,
        props.precompiled_internal_cache,
        props.compressed_data
    );
    // (A structure type the driver did not know leaves its fields as set.)
    bools.iter().all(|&b| b <= 1) && feats.pipeline_binaries == 1
}

impl Fns {
    /// The commands (None: not all found) and the global key.
    pub unsafe fn load(instance: &ash::Instance, pdev: vk::PhysicalDevice, device: vk::Device, compatibility: &[u8]) -> Option<Fns> {
        let get = |n: &CStr| unsafe { instance.get_device_proc_addr(device, n.as_ptr()) };
        let create = get(c"vkCreatePipelineBinariesKHR")?;
        let destroy = get(c"vkDestroyPipelineBinaryKHR")?;
        let data = get(c"vkGetPipelineBinaryDataKHR")?;
        let release = get(c"vkReleaseCapturedPipelineDataKHR")?;
        // SAFETY: the driver's entry points for these commands, with the
        // prototypes above.
        let (create, destroy, data, release): (CreateFn, DestroyFn, DataFn, ReleaseFn) = unsafe { (std::mem::transmute(create), std::mem::transmute(destroy), std::mem::transmute(data), std::mem::transmute(release)) };
        let props = unsafe { instance.get_physical_device_properties(pdev) };
        let mut global = props.pipeline_cache_uuid.to_vec();
        for v in [props.driver_version, props.vendor_id, props.device_id] {
            global.extend_from_slice(&v.to_le_bytes());
        }
        let contract = compatibility.iter().fold(0xcbf2_9ce4_8422_2325u64, |h, &b| (h ^ b as u64).wrapping_mul(0x100_0000_01b3));
        global.extend_from_slice(&contract.to_le_bytes());
        Some(Fns { device, create, destroy, data, release, global, shared: parking_lot::Mutex::new(Vec::new()) })
    }

    /// Whether the shared binary of `key` is made this session.
    pub fn has_shared(&self, key: &[u8]) -> bool {
        self.shared.lock().iter().any(|e| e.0 == key)
    }

    /// Makes the shared binary of `key` from its data (once a session).
    pub fn share(&self, key: &[u8], data: &[u8]) -> Result<(), vk::Result> {
        if self.has_shared(key) {
            return Ok(());
        }
        let h = self.make(&[(key, data)])?;
        let mut g = self.shared.lock();
        if g.iter().any(|e| e.0 == key) {
            self.destroy(&h);
        } else {
            g.push((key.to_vec(), h[0]));
        }
        Ok(())
    }

    /// The binaries of the pipeline that `create_info` (a VkComputePipelineCreateInfo
    /// or VkGraphicsPipelineCreateInfo) makes, from the driver's internal
    /// cache (right after that pipeline was made): each one's key and data
    /// (`dsport --pb-test`).
    pub fn probe(&self, create_info: *const c_void) -> Result<Vec<(Vec<u8>, Vec<u8>)>, vk::Result> {
        let pci = PipelineCreateInfo { s_type: ST_PIPELINE_CREATE_INFO, p_next: create_info as *mut c_void };
        let ci = CreateInfo { s_type: ST_CREATE_INFO, p_next: std::ptr::null(), p_keys_and_data: std::ptr::null(), pipeline: vk::Pipeline::null(), p_pipeline_create_info: &pci };
        self.binaries_of(&ci, None, &|_| false)
    }

    /// The binaries of `pipeline` (made with CAPTURE_DATA): each one's key
    /// and data (empty for keys `known` says are stored already: the 2 MB
    /// shared one is not copied again); its captured data released.
    pub fn capture(&self, pipeline: vk::Pipeline, known: &dyn Fn(&[u8]) -> bool) -> Result<Vec<(Vec<u8>, Vec<u8>)>, vk::Result> {
        let ci = CreateInfo { s_type: ST_CREATE_INFO, p_next: std::ptr::null(), p_keys_and_data: std::ptr::null(), pipeline, p_pipeline_create_info: std::ptr::null() };
        self.binaries_of(&ci, Some(pipeline), known)
    }

    /// The binaries `ci` names, each one's key and data (`release`: the
    /// pipeline whose captured data is released after).
    fn binaries_of(&self, ci: &CreateInfo, release: Option<vk::Pipeline>, known: &dyn Fn(&[u8]) -> bool) -> Result<Vec<(Vec<u8>, Vec<u8>)>, vk::Result> {
        let mut hi = HandlesInfo { s_type: ST_HANDLES_INFO, p_next: std::ptr::null(), count: 0, p_binaries: std::ptr::null_mut() };
        let r = unsafe { (self.create)(self.device, ci, std::ptr::null(), &mut hi) };
        let out = if r != vk::Result::SUCCESS || hi.count == 0 {
            Err(if r == vk::Result::SUCCESS { vk::Result::INCOMPLETE } else { r })
        } else {
            let mut handles = vec![0 as Binary; hi.count as usize];
            hi.p_binaries = handles.as_mut_ptr();
            let r = unsafe { (self.create)(self.device, ci, std::ptr::null(), &mut hi) };
            let mut got = Vec::with_capacity(handles.len());
            let mut err = (r != vk::Result::SUCCESS).then_some(r);
            for &h in &handles {
                if err.is_some() || h == 0 {
                    err.get_or_insert(vk::Result::ERROR_UNKNOWN);
                    continue;
                }
                let di = DataInfo { s_type: ST_DATA_INFO, p_next: std::ptr::null_mut(), binary: h };
                let mut k = Key::new();
                let mut size = 0usize;
                let r = unsafe { (self.data)(self.device, &di, &mut k, &mut size, std::ptr::null_mut()) };
                if r != vk::Result::SUCCESS {
                    err = Some(r);
                    continue;
                }
                if known(k.bytes()) {
                    got.push((k.bytes().to_vec(), Vec::new()));
                    continue;
                }
                let mut buf = vec![0u8; size];
                let r = unsafe { (self.data)(self.device, &di, &mut k, &mut size, buf.as_mut_ptr() as *mut c_void) };
                if r != vk::Result::SUCCESS {
                    err = Some(r);
                    continue;
                }
                buf.truncate(size);
                got.push((k.bytes().to_vec(), buf));
            }
            for h in handles.into_iter().filter(|&h| h != 0) {
                unsafe { (self.destroy)(self.device, h, std::ptr::null()) };
            }
            match err {
                Some(e) => Err(e),
                None => Ok(got),
            }
        };
        if let Some(pipeline) = release {
            let ri = ReleaseInfo { s_type: ST_RELEASE_INFO, p_next: std::ptr::null_mut(), pipeline };
            let _ = unsafe { (self.release)(self.device, &ri, std::ptr::null()) };
        }
        out
    }

    /// A pipeline's binaries from its stored parts, in their order: its
    /// own made now (the second list: to be destroyed with `destroy` once the
    /// pipeline is made), the shared ones as made (`share`).
    pub fn from_parts(&self, parts: &[Part]) -> Result<(Vec<Binary>, Vec<Binary>), vk::Result> {
        let own: Vec<(&[u8], &[u8])> = parts
            .iter()
            .filter_map(|p| match p {
                Part::Own(k, d) => Some((k.as_slice(), d.as_slice())),
                Part::Shared(_) => None,
            })
            .collect();
        let made = if own.is_empty() { Vec::new() } else { self.make(&own)? };
        let shared = self.shared.lock();
        let mut it = made.iter();
        let mut all = Vec::with_capacity(parts.len());
        for p in parts {
            match p {
                Part::Own(..) => all.push(*it.next().unwrap_or(&0)),
                Part::Shared(k) => match shared.iter().find(|e| &e.0 == k) {
                    Some(e) => all.push(e.1),
                    None => {
                        drop(shared);
                        self.destroy(&made);
                        return Err(vk::Result::ERROR_UNKNOWN);
                    }
                },
            }
        }
        Ok((all, made))
    }

    /// Binaries made from keys and data.
    fn make(&self, stored: &[(&[u8], &[u8])]) -> Result<Vec<Binary>, vk::Result> {
        let keys: Vec<Key> = stored
            .iter()
            .map(|(k, _)| {
                let mut key = Key::new();
                let n = k.len().min(MAX_KEY);
                key.key[..n].copy_from_slice(&k[..n]);
                key.key_size = n as u32;
                key
            })
            .collect();
        let data: Vec<Data> = stored.iter().map(|(_, d)| Data { data_size: d.len(), p_data: d.as_ptr() as *mut c_void }).collect();
        let kd = KeysAndData { binary_count: stored.len() as u32, p_keys: keys.as_ptr(), p_data: data.as_ptr() };
        let ci = CreateInfo { s_type: ST_CREATE_INFO, p_next: std::ptr::null(), p_keys_and_data: &kd, pipeline: vk::Pipeline::null(), p_pipeline_create_info: std::ptr::null() };
        let mut handles = vec![0 as Binary; stored.len()];
        let mut hi = HandlesInfo { s_type: ST_HANDLES_INFO, p_next: std::ptr::null(), count: handles.len() as u32, p_binaries: handles.as_mut_ptr() };
        let r = unsafe { (self.create)(self.device, &ci, std::ptr::null(), &mut hi) };
        if r != vk::Result::SUCCESS || handles.iter().any(|&h| h == 0) {
            for &h in handles.iter().filter(|&&h| h != 0) {
                unsafe { (self.destroy)(self.device, h, std::ptr::null()) };
            }
            return Err(if r == vk::Result::SUCCESS { vk::Result::ERROR_UNKNOWN } else { r });
        }
        Ok(handles)
    }

    pub fn destroy(&self, handles: &[Binary]) {
        for &h in handles.iter().filter(|&&h| h != 0) {
            unsafe { (self.destroy)(self.device, h, std::ptr::null()) };
        }
    }
}
