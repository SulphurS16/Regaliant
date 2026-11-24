use std::collections::HashMap;

#[link(name = "ntdll")]
unsafe extern "system" {
    fn NtQuerySystemInformation(class: u32, buf: *mut core::ffi::c_void, len: u32, ret: *mut u32) -> i32;
}

/// Kernel time and user time (100 ns units; charged by the clock's ticks,
/// so only sums over a window of seconds mean something) and context
/// switches of this process's threads by host thread id, from the system's
/// process list (SystemProcessInformation: x64 layouts, the process record's
/// threads from +0x100, 0x50 bytes each). One call of a few ms: for the
/// profile window's ends only.
pub fn thread_times() -> HashMap<u32, (u64, u64, u64)> {
    let mut out = HashMap::new();
    let pid = std::process::id() as u64;
    let mut buf: Vec<u64> = vec![0; 1 << 17];
    for _ in 0..4 {
        let mut need = 0u32;
        let st = unsafe { NtQuerySystemInformation(5, buf.as_mut_ptr() as *mut _, (buf.len() * 8) as u32, &mut need) };
        if st == 0 {
            break;
        }
        // STATUS_INFO_LENGTH_MISMATCH: a larger buffer.
        if st as u32 != 0xC000_0004 {
            return out;
        }
        buf = vec![0; (need as usize / 8 + 1) * 2];
    }
    let base = buf.as_ptr() as *const u8;
    let len = buf.len() * 8;
    let rd32 = |o: usize| unsafe { (base.add(o) as *const u32).read_unaligned() };
    let rd64 = |o: usize| unsafe { (base.add(o) as *const u64).read_unaligned() };
    let mut at = 0usize;
    loop {
        if at + 0x100 > len {
            break;
        }
        let (next, threads) = (rd32(at) as usize, rd32(at + 4) as usize);
        if rd64(at + 0x50) == pid {
            for k in 0..threads {
                let t = at + 0x100 + k * 0x50;
                if t + 0x50 > len {
                    break;
                }
                out.insert(rd64(t + 0x30) as u32, (rd64(t), rd64(t + 8), rd32(t + 0x40) as u64));
            }
            break;
        }
        if next == 0 {
            break;
        }
        at += next;
    }
    out
}


/// Scheduled CPU time estimated from the same cycle counter as the renderer census.
pub fn own_ns() -> u64 {
    (crate::gpu::threadcpu::ms(crate::gpu::threadcpu::own_cycles()) * 1_000_000.0) as u64
}
