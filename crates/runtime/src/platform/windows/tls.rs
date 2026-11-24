use windows_sys::Win32::System::Threading::{TlsAlloc, TlsGetValue, TlsSetValue};

pub fn alloc_slot() -> Result<u32, String> {
    let s = unsafe { TlsAlloc() };
    if s == u32::MAX || s >= 64 {
        return Err(format!("TlsAlloc returned {s}; need a direct TEB slot (<64)"));
    }
    Ok(s)
}
pub fn set(slot: u32, tp: u64) {
    unsafe { TlsSetValue(slot, tp as *const _) };
}
pub fn get(slot: u32) -> u64 {
    unsafe { TlsGetValue(slot) as u64 }
}
