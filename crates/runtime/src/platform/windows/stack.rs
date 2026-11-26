pub unsafe fn allocate(size: usize) -> u64 {
    use windows_sys::Win32::System::Memory::{MEM_COMMIT, MEM_RESERVE, PAGE_READWRITE, VirtualAlloc};
    unsafe { VirtualAlloc(std::ptr::null(), size, MEM_COMMIT | MEM_RESERVE, PAGE_READWRITE) as u64 }
}
pub unsafe fn free(base: u64, _size: usize) {
    use windows_sys::Win32::System::Memory::{MEM_RELEASE, VirtualFree};
    unsafe { VirtualFree(base as *mut _, 0, MEM_RELEASE) };
}
