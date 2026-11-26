use super::*;

pub fn enter(t: &'static GuestThread) {
    CURRENT.with(|c| c.set(Some(t)));
    crate::tls::set_current_tp(t.tp.load(Ordering::SeqCst));
    crate::log::set_thread_name(&t.name());
    crate::platform::fpu::enter();
    t.host_tid.store(unsafe { libc::syscall(libc::SYS_gettid) } as u64, Ordering::SeqCst);
}

pub fn apply_guest_cpus(mask: u64) {
    // The mask uses host logical CPU numbers, matching the development control on Windows.
    let tids: Vec<_> = THREADS.lock().iter().filter(|t| t.exited.try_lock().is_some_and(|e| e.is_none()))
        .map(|t| t.host_tid.load(Ordering::Relaxed) as libc::pid_t).filter(|&t| t != 0).collect();
    let mut set: libc::cpu_set_t = unsafe { std::mem::zeroed() };
    // The launch affinity is a ceiling, including for live guest_cpus overrides.
    if unsafe { libc::sched_getaffinity(std::process::id() as _, std::mem::size_of_val(&set), &mut set) } != 0 {
        crate::warn!("read process CPU affinity: {}", std::io::Error::last_os_error()); return;
    }
    if mask != 0 {
        for cpu in 0..libc::CPU_SETSIZE as usize {
            if cpu >= 64 || mask & (1 << cpu) == 0 { unsafe { libc::CPU_CLR(cpu, &mut set) }; }
        }
    }
    for tid in tids {
        if unsafe { libc::sched_setaffinity(tid, std::mem::size_of_val(&set), &set) } != 0 {
            crate::warn!("thread {tid} affinity {mask:#x}: {}", std::io::Error::last_os_error());
        }
    }
}

pub fn apply_power_optout(_on: bool) {
    crate::warn!("Windows power-throttling control is unavailable on Linux");
}
pub fn sampler_priority() {
    crate::warn!("Linux sampler priority elevation is not implemented");
}
pub fn short_sleep(d: std::time::Duration) {
    if d.is_zero() { std::thread::yield_now(); } else { std::thread::sleep(d); }
}

pub struct Waker { signaled: Mutex<bool>, cv: Condvar }
impl Waker {
    pub fn new() -> Self { Self { signaled: Mutex::new(false), cv: Condvar::new() } }
    pub fn wake(&self) { *self.signaled.lock() = true; self.cv.notify_one(); }
}
pub fn sleep_or_wake(d: std::time::Duration, waker: &Waker) -> bool {
    let deadline = std::time::Instant::now() + d;
    let mut signaled = waker.signaled.lock();
    while !*signaled {
        if waker.cv.wait_until(&mut signaled, deadline).timed_out() && !*signaled { return false; }
    }
    *signaled = false;
    true
}

pub fn spawn_host(name: &str, stack_size: usize, f: Box<dyn FnOnce() + Send>) -> std::io::Result<()> {
    std::thread::Builder::new().name(name.into()).stack_size(stack_size).spawn(move || {
        let _signals = crate::platform::signals::SignalStack::new().expect("guest signal stack");
        f();
    }).map(|_| ())
}

#[cfg(test)]
mod tests {
    #[test]
    fn wake_before_sleep_is_retained_and_auto_reset() {
        let w = super::Waker::new();
        w.wake();
        assert!(super::sleep_or_wake(std::time::Duration::ZERO, &w));
        assert!(!super::sleep_or_wake(std::time::Duration::from_millis(1), &w));
    }

    #[test]
    fn native_guest_thread_tls_join_and_exit() {
        use crate::thread;
        use std::sync::atomic::Ordering;
        unsafe extern "sysv64" fn entry(value: u64, _unused: u64) -> u64 {
            let tp: u64;
            unsafe { std::arch::asm!("mov {}, gs:[0x1480]", out(reg) tp, options(nostack, readonly, preserves_flags)); }
            let me = thread::current().unwrap();
            if tp != me.tp.load(Ordering::Relaxed) || unsafe { *(tp as *const u64) } != tp { return 0; }
            // This host allocation must still use libc's FS while guest GS is installed.
            let host = vec![value; 64];
            if value & 1 != 0 { thread::exit_current(host[31] + 1); }
            host[31] + 1
        }
        crate::tls::init_slot().unwrap();
        let threads: Vec<_> = (40..48).map(|value| {
            let t = thread::new_thread("linux-guest-test", entry as *const () as u64, value, 1 << 20, 700);
            thread::spawn(t).unwrap();
            (t, value)
        }).collect();
        for (t, value) in threads { assert_eq!(thread::join(t), value + 1); }
    }

}
