pub unsafe fn flush_instruction_cache(base: u64, size: usize) {
    unsafe {
        windows_sys::Win32::System::Diagnostics::Debug::FlushInstructionCache(
            windows_sys::Win32::System::Threading::GetCurrentProcess(), base as *const _, size,
        );
    }
}
