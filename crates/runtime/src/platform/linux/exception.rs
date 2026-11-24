// Loader diagnostics use raw addresses for host frames until an ELF/DWARF symbolizer is wired.
pub fn host_image() -> (u64, u64) {
    (0, 0)
}

pub fn describe_guest(addr: u64) -> String {
    crate::loader::describe_addr(addr)
}
pub fn frame_label(addr: u64) -> String {
    describe_guest(addr)
}
pub fn host_symbol(addr: u64) -> String {
    let mut info: libc::Dl_info = unsafe { std::mem::zeroed() };
    if unsafe { libc::dladdr(addr as _, &mut info) } != 0 && !info.dli_sname.is_null() {
        format!(
            "{}+{:#x}",
            unsafe { std::ffi::CStr::from_ptr(info.dli_sname) }.to_string_lossy(),
            addr.saturating_sub(info.dli_saddr as u64)
        )
    } else {
        format!("host {addr:#x}")
    }
}
pub fn host_line(_addr: u64) -> Option<String> {
    None
}
pub fn sample_thread(_tid: u32, _depth: usize) -> Option<String> {
    None
}
pub fn sample_frames(_tid: u32, _depth: usize) -> Option<(Vec<u64>, Vec<u64>)> {
    None
}
pub fn arm_watch_current_thread(_guest: bool) {}
pub fn arm_watch_here(_guest: bool) {}
pub fn set_watch(_slot: usize, _addr: u64, _rw: u32, _len: u32, _guest_only: bool) -> usize {
    crate::warn!("hardware watchpoints unavailable on Linux");
    0
}
pub fn race_watch(_addr: u64, _len: u64) {
    crate::warn!("race watchpoints unavailable on Linux");
}
pub fn race_unwatch() -> Vec<String> {
    Vec::new()
}
pub fn current_guest_returns(depth: usize, readable: &dyn Fn(u64) -> bool) -> Vec<u64> {
    let mut rbp: u64;
    unsafe {
        std::arch::asm!("mov {}, rbp",out(reg)rbp,options(nostack,nomem,preserves_flags));
    }
    let mut out = Vec::new();
    for _ in 0..depth.min(128) {
        let mut frame = [0u64; 2];
        let mut n = 0;
        if !readable(rbp)
            || unsafe {
                crate::platform::os::ReadProcessMemory(
                    crate::platform::os::INVALID_HANDLE_VALUE,
                    rbp as _,
                    frame.as_mut_ptr().cast(),
                    16,
                    &mut n,
                )
            } == 0
        {
            break;
        }
        if frame[1] >= crate::mem::GUEST_BASE && frame[1] < crate::mem::GUEST_END {
            out.push(frame[1]);
        }
        if frame[0] <= rbp || frame[0] - rbp > 16 << 20 {
            break;
        }
        rbp = frame[0];
    }
    out
}
pub fn current_guest_frames(depth: usize) -> Vec<String> {
    current_guest_returns(depth, &|a| a >= 0x10000)
        .into_iter()
        .map(describe_guest)
        .collect()
}

/// Read through the kernel so a diagnostic text hook cannot fault on an invalid string.
pub(crate) unsafe fn peek_string(a: u64, wide: bool) -> String {
    let mut bytes = [0u8; 240];
    let mut n = 0;
    unsafe { crate::platform::os::ReadProcessMemory(-1isize as _, a as _, bytes.as_mut_ptr().cast(), bytes.len(), &mut n); }
    if wide {
        String::from_utf16_lossy(&bytes[..n].chunks_exact(2).map(|b| u16::from_le_bytes([b[0],b[1]])).take_while(|&v| v != 0).collect::<Vec<_>>())
    } else {
        String::from_utf8_lossy(&bytes[..n].split(|&b| b == 0).next().unwrap_or_default()).into_owned()
    }
}
