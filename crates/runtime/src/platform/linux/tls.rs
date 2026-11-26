//! Guest GS storage. FS always remains libc's thread pointer, also in HLE and signal handlers.
//! The offsets match the Windows patcher and fiber assembly: stack bounds at 8/16 and the
//! guest pointer at 0x1480. GS is installed separately on every host thread entering the guest.
use std::cell::RefCell;

const ARCH_SET_GS: libc::c_int = 0x1001;
const ARCH_GET_GS: libc::c_int = 0x1004;
const SLOT: usize = 0x1480 / 8;

struct Binding {
    area: Box<[u64; SLOT + 64]>,
    previous: u64,
}

impl Binding {
    fn new() -> Result<Self, String> {
        let mut previous = 0u64;
        if unsafe { libc::syscall(libc::SYS_arch_prctl, ARCH_GET_GS, &mut previous) } != 0 {
            return Err(format!("ARCH_GET_GS: {}", std::io::Error::last_os_error()));
        }
        let mut area = Box::new([0u64; SLOT + 64]);
        // The fiber assembly can save/restore these fields; Linux itself does not use them.
        unsafe {
            let mut attr: libc::pthread_attr_t = std::mem::zeroed();
            let rc = libc::pthread_getattr_np(libc::pthread_self(), &mut attr);
            if rc != 0 { return Err(format!("pthread_getattr_np: {rc}")); }
            let mut stack = std::ptr::null_mut();
            let mut size = 0usize;
            let rc = libc::pthread_attr_getstack(&attr, &mut stack, &mut size);
            libc::pthread_attr_destroy(&mut attr);
            if rc != 0 { return Err(format!("pthread_attr_getstack: {rc}")); }
            area[1] = stack as u64 + size as u64;
            area[2] = stack as u64;
            if libc::syscall(libc::SYS_arch_prctl, ARCH_SET_GS, area.as_ptr()) != 0 {
                return Err(format!("ARCH_SET_GS: {}", std::io::Error::last_os_error()));
            }
        }
        Ok(Self { area, previous })
    }
}

impl Drop for Binding {
    fn drop(&mut self) {
        // Restore before the storage is freed. No destructor or signal handler depends on GS.
        if unsafe { libc::syscall(libc::SYS_arch_prctl, ARCH_SET_GS, self.previous) } != 0 {
            unsafe { libc::_exit(127) };
        }
    }
}

thread_local! {
    static BINDING: RefCell<Option<Binding>> = const { RefCell::new(None) };
}

pub fn alloc_slot() -> Result<u32, String> { Ok(0) }

pub fn set(slot: u32, tp: u64) {
    assert!(slot < 64);
    BINDING.with(|b| {
        let mut b = b.borrow_mut();
        if b.is_none() { *b = Some(Binding::new().expect("install guest GS area")); }
        b.as_mut().unwrap().area[SLOT + slot as usize] = tp;
    });
}

pub fn get(slot: u32) -> u64 {
    assert!(slot < 64);
    BINDING.with(|b| b.borrow().as_ref().map_or(0, |b| b.area[SLOT + slot as usize]))
}

#[cfg(test)]
mod tests {
    #[test]
    fn gs_is_per_thread_and_fs_is_unchanged() {
        let threads: Vec<_> = (1..=8u64).map(|value| std::thread::spawn(move || {
            let mut before = 0u64;
            let mut after = 0u64;
            unsafe { assert_eq!(libc::syscall(libc::SYS_arch_prctl, 0x1003, &mut before), 0); }
            super::set(0, value);
            std::thread::yield_now();
            let guest: u64;
            unsafe {
                std::arch::asm!("mov {}, gs:[0x1480]", out(reg) guest, options(nostack, readonly, preserves_flags));
                assert_eq!(libc::syscall(libc::SYS_arch_prctl, 0x1003, &mut after), 0);
            }
            assert_eq!(guest, value);
            assert_eq!(super::get(0), value);
            assert_eq!(before, after);
            // Exercise Rust TLS and libc while the guest GS base is active.
            assert!(std::thread::current().id() == std::thread::current().id());
            assert!(std::fs::metadata("/proc/self").is_ok());
        })).collect();
        for t in threads { t.join().unwrap(); }
    }
}
