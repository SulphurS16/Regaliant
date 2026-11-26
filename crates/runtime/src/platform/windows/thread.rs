use super::*;

/// Binds the calling host thread to a guest thread (TLS slot + current pointer).
pub fn enter(t: &'static GuestThread) {
    CURRENT.with(|c| c.set(Some(t)));
    crate::tls::set_current_tp(t.tp.load(Ordering::SeqCst));
    crate::log::set_thread_name(&t.name());
    t.host_tid.store(unsafe { windows_sys::Win32::System::Threading::GetCurrentThreadId() } as u64, Ordering::SeqCst);
    crate::exception::arm_watch_current_thread(true);
    let cpus = crate::gpu::tune::guest_cpus();
    if cpus != 0 {
        set_cpus(unsafe { windows_sys::Win32::System::Threading::GetCurrentThread() }, cpus);
    }
}

/// Restricts a host thread to the logical CPUs of `mask` (0: all of the
/// process's).
fn set_cpus(h: windows_sys::Win32::Foundation::HANDLE, mask: u64) {
    use windows_sys::Win32::System::Threading::{GetCurrentProcess, GetProcessAffinityMask, SetThreadAffinityMask};
    let mut m = mask as usize;
    if m == 0 {
        let (mut process, mut system) = (0usize, 0usize);
        if unsafe { GetProcessAffinityMask(GetCurrentProcess(), &mut process, &mut system) } == 0 {
            return;
        }
        m = process;
    }
    if unsafe { SetThreadAffinityMask(h, m) } == 0 {
        crate::warn!("thread affinity {m:#x} not set ({})", std::io::Error::last_os_error());
    }
}

/// `tune guest_cpus <mask>`: the title's threads, running and later ones,
/// on the logical CPUs of `mask` (0: all of the process's).
pub fn apply_guest_cpus(mask: u64) {
    use windows_sys::Win32::System::Threading::{OpenThread, THREAD_QUERY_INFORMATION, THREAD_SET_INFORMATION};
    let tids: Vec<u32> = THREADS.lock().iter().filter(|t| t.exited.try_lock().is_some_and(|e| e.is_none())).map(|t| t.host_tid.load(Ordering::Relaxed) as u32).filter(|&tid| tid != 0).collect();
    for tid in &tids {
        let h = unsafe { OpenThread(THREAD_SET_INFORMATION | THREAD_QUERY_INFORMATION, 0, *tid) };
        if h.is_null() {
            continue;
        }
        set_cpus(h, mask);
        unsafe { windows_sys::Win32::Foundation::CloseHandle(h) };
    }
    crate::info!("guest threads ({}) restricted to CPU mask {mask:#x} (0: all)", tids.len());
}

/// Whether `apply_power_optout` holds a 1 ms timer resolution.
static TIMER_PERIOD_HELD: std::sync::atomic::AtomicBool = std::sync::atomic::AtomicBool::new(false);

/// `tune power_optout` (FINDINGS.md 2026-10-22): on, the process opts out of
/// Windows' power throttling (execution speed, and the rule that drops the
/// timer-resolution guarantee for a window that is hidden, occluded or
/// minimized) and asks for a 1 ms timer resolution (`timeBeginPeriod`: a
/// process keeps the 15.6 ms tick for timed waits otherwise; our sleeps use
/// high-resolution waitable timers, but timed waits on condition variables
/// and events round to the tick); off, the system's defaults.
pub fn apply_power_optout(on: bool) {
    use windows_sys::Win32::Media::{timeBeginPeriod, timeEndPeriod};
    use windows_sys::Win32::System::Threading::{GetCurrentProcess, ProcessPowerThrottling, SetProcessInformation, PROCESS_POWER_THROTTLING_CURRENT_VERSION, PROCESS_POWER_THROTTLING_EXECUTION_SPEED, PROCESS_POWER_THROTTLING_IGNORE_TIMER_RESOLUTION, PROCESS_POWER_THROTTLING_STATE};
    let state = PROCESS_POWER_THROTTLING_STATE {
        Version: PROCESS_POWER_THROTTLING_CURRENT_VERSION,
        // Controlled with the state bits clear: throttling off. Not
        // controlled: the system decides.
        ControlMask: if on { PROCESS_POWER_THROTTLING_EXECUTION_SPEED | PROCESS_POWER_THROTTLING_IGNORE_TIMER_RESOLUTION } else { 0 },
        StateMask: 0,
    };
    let set = unsafe { SetProcessInformation(GetCurrentProcess(), ProcessPowerThrottling, &state as *const _ as *const core::ffi::c_void, std::mem::size_of::<PROCESS_POWER_THROTTLING_STATE>() as u32) } != 0;
    let held = TIMER_PERIOD_HELD.load(Ordering::Relaxed);
    let timer = if on && !held {
        let ok = unsafe { timeBeginPeriod(1) } == 0;
        TIMER_PERIOD_HELD.store(ok, Ordering::Relaxed);
        ok
    } else if !on && held {
        unsafe { timeEndPeriod(1) };
        TIMER_PERIOD_HELD.store(false, Ordering::Relaxed);
        true
    } else {
        true
    };
    crate::info!(
        "power throttling {} and timer resolution {} (`tune power_optout` {}){}",
        if on { "opted out" } else { "left to the system" },
        if on { "1 ms" } else { "the system's" },
        on as u8,
        if set && timer { "" } else { "; a call failed" }
    );
}

/// Raises the calling thread (a sampler) above every thread it samples: at a lower priority
/// than the front end's threads (above normal, `gpu::prio`) it ran only when one of them
/// blocked, so its samples found them idle far more often than they were (2026-11-15: the
/// graphics workers sampled 22% busy while their thread cycles were 98% in their runs).
pub fn sampler_priority() {
    use windows_sys::Win32::System::Threading::{GetCurrentThread, SetThreadPriority, THREAD_PRIORITY_TIME_CRITICAL};
    unsafe { SetThreadPriority(GetCurrentThread(), THREAD_PRIORITY_TIME_CRITICAL) };
}

/// Creates a host thread whose whole stack is committed up front. PS4 code does
/// not probe the stack, so lazily grown (guard-page) Windows stacks would fault
/// when a large frame is touched below the guard page.
/// Sleeps about `d` on a high-resolution waitable timer the calling thread
/// keeps (std's `thread::sleep` creates and closes one per call: in the
/// command processors' 100 us polling sleeps that was ~4.5% of the compute
/// queues' busy samples in `ZwClose` alone, run 20261014-161648). Falls back
/// to `thread::sleep` when the timer cannot be made.
pub fn short_sleep(d: std::time::Duration) {
    use windows_sys::Win32::Foundation::{CloseHandle, HANDLE};
    use windows_sys::Win32::System::Threading::{CreateWaitableTimerExW, SetWaitableTimer, WaitForSingleObject, CREATE_WAITABLE_TIMER_HIGH_RESOLUTION, INFINITE, TIMER_ALL_ACCESS};
    struct Timer(HANDLE);
    impl Drop for Timer {
        fn drop(&mut self) {
            if !self.0.is_null() {
                unsafe { CloseHandle(self.0) };
            }
        }
    }
    thread_local! {
        static TIMER: Timer = Timer(unsafe { CreateWaitableTimerExW(std::ptr::null(), std::ptr::null(), CREATE_WAITABLE_TIMER_HIGH_RESOLUTION, TIMER_ALL_ACCESS) });
    }
    // Zero: yield as std's sleep does (Sleep(0)).
    if d.is_zero() || !crate::gpu::tune::timer_sleep() {
        std::thread::sleep(d);
        return;
    }
    // Relative due time in 100 ns units (negative), at least one unit.
    let due = -((d.as_nanos() / 100).clamp(1, i64::MAX as u128) as i64);
    let slept = TIMER
        .try_with(|t| !t.0.is_null() && unsafe { SetWaitableTimer(t.0, &due, 0, None, std::ptr::null(), 0) } != 0 && unsafe { WaitForSingleObject(t.0, INFINITE) } == 0)
        .unwrap_or(false);
    if !slept {
        std::thread::sleep(d);
    }
}

/// An event a waiting thread can be woken with before its time (`sleep_or_wake`): auto-reset,
/// kept for the process's life.
pub struct Waker(isize);

impl Waker {
    pub fn new() -> Waker {
        use windows_sys::Win32::System::Threading::CreateEventW;
        Waker(unsafe { CreateEventW(std::ptr::null(), 0, 0, std::ptr::null()) } as isize)
    }

    /// Ends the waiting thread's sleep (or its next one).
    pub fn wake(&self) {
        if self.0 != 0 {
            unsafe { windows_sys::Win32::System::Threading::SetEvent(self.0 as _) };
        }
    }
}

/// Sleeps about `d` on the calling thread's high-resolution timer as `short_sleep` does, or
/// until `waker` is woken; true when woken. (Without the timer or the event: `short_sleep`.)
pub fn sleep_or_wake(d: std::time::Duration, waker: &Waker) -> bool {
    use windows_sys::Win32::Foundation::{CloseHandle, HANDLE};
    use windows_sys::Win32::System::Threading::{CancelWaitableTimer, CreateWaitableTimerExW, SetWaitableTimer, WaitForMultipleObjects, CREATE_WAITABLE_TIMER_HIGH_RESOLUTION, INFINITE, TIMER_ALL_ACCESS};
    struct Timer(HANDLE);
    impl Drop for Timer {
        fn drop(&mut self) {
            if !self.0.is_null() {
                unsafe { CloseHandle(self.0) };
            }
        }
    }
    thread_local! {
        static TIMER: Timer = Timer(unsafe { CreateWaitableTimerExW(std::ptr::null(), std::ptr::null(), CREATE_WAITABLE_TIMER_HIGH_RESOLUTION, TIMER_ALL_ACCESS) });
    }
    if waker.0 == 0 || d.is_zero() {
        short_sleep(d);
        return false;
    }
    let due = -((d.as_nanos() / 100).clamp(1, i64::MAX as u128) as i64);
    let r = TIMER.try_with(|t| {
        if t.0.is_null() || unsafe { SetWaitableTimer(t.0, &due, 0, None, std::ptr::null(), 0) } == 0 {
            return None;
        }
        let handles = [waker.0 as HANDLE, t.0];
        let r = unsafe { WaitForMultipleObjects(2, handles.as_ptr(), 0, INFINITE) };
        if r == 0 {
            unsafe { CancelWaitableTimer(t.0) };
        }
        Some(r == 0)
    });
    match r {
        Ok(Some(woken)) => woken,
        _ => {
            short_sleep(d);
            false
        }
    }
}

pub fn spawn_host(name: &str, stack_size: usize, f: Box<dyn FnOnce() + Send>) -> std::io::Result<()> {
    use windows_sys::Win32::System::Threading::{CreateThread, SetThreadDescription};
    unsafe extern "system" fn proc_(param: *mut std::ffi::c_void) -> u32 {
        let f = unsafe { Box::from_raw(param as *mut Box<dyn FnOnce() + Send>) };
        f();
        0
    }
    let stack = (stack_size + 0xFFFF) & !0xFFFF;
    let param = Box::into_raw(Box::new(f)) as *mut std::ffi::c_void;
    let mut tid = 0u32;
    let h = unsafe { CreateThread(std::ptr::null(), stack, Some(proc_), param, 0, &mut tid) };
    if h.is_null() {
        drop(unsafe { Box::from_raw(param as *mut Box<dyn FnOnce() + Send>) });
        return Err(std::io::Error::last_os_error());
    }
    let wname: Vec<u16> = name.encode_utf16().chain(std::iter::once(0)).collect();
    unsafe {
        SetThreadDescription(h, wname.as_ptr());
        windows_sys::Win32::Foundation::CloseHandle(h);
    }
    Ok(())
}

