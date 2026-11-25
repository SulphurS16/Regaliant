//! Synchronous guest crash reporting. No allocator, logger lock, Rust TLS or symbolizer is
//! called from the handler. The raw registers and pre-registered module offsets survive even
//! when the fault happened in a host allocator. Recoverable page faults rendezvous with a
//! service thread through a pipe and futex; shared renderer recovery runs outside the handler.
use super::vm;
#[path = "signal_report.rs"]
mod report;
pub use report::{CompileNote, compiler_crash_file};
use std::sync::OnceLock;
use std::sync::atomic::{AtomicI32, Ordering};

pub static DIRECT: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);
static REPORT_FD: AtomicI32 = AtomicI32::new(-1);
static MODULES: OnceLock<Vec<(u64, u64, String)>> = OnceLock::new();
pub fn modules(list: Vec<(u64, u64, String)>) {
    let _ = MODULES.set(list);
}

/// Must stay alive for every thread executing guest code (including guest-created threads).
pub struct SignalStack {
    previous: libc::stack_t,
    _memory: vm::Reservation,
}
impl SignalStack {
    pub fn new() -> std::io::Result<Self> {
        let size = 256 * 1024;
        let memory = vm::Reservation::new_at(0, size + 2 * vm::PAGE)?;
        memory.protect(vm::PAGE, size, vm::READ | vm::WRITE)?;
        let mut previous: libc::stack_t = unsafe { std::mem::zeroed() };
        let stack = libc::stack_t {
            ss_sp: (memory.base() + vm::PAGE) as *mut _,
            ss_flags: 0,
            ss_size: size,
        };
        if unsafe { libc::sigaltstack(&stack, &mut previous) } != 0 {
            return Err(std::io::Error::last_os_error());
        }
        Ok(Self {
            previous,
            _memory: memory,
        })
    }
}
impl Drop for SignalStack {
    fn drop(&mut self) {
        if unsafe { libc::sigaltstack(&self.previous, std::ptr::null_mut()) } != 0 {
            unsafe { libc::_exit(127) };
        }
    }
}

struct Line {
    bytes: [u8; 2048],
    used: usize,
}
impl Line {
    fn text(&mut self, text: &[u8]) {
        for &b in text {
            if self.used == self.bytes.len() {
                break;
            }
            self.bytes[self.used] = b;
            self.used += 1;
        }
    }
    fn hex(&mut self, value: u64) {
        self.text(b"0x");
        for shift in (0..16).rev() {
            self.text(&[b"0123456789abcdef"[((value >> (shift * 4)) & 15) as usize]]);
        }
    }
    unsafe fn write(&self, fd: i32) {
        let mut at = 0;
        while at < self.used {
            let n =
                unsafe { libc::write(fd, self.bytes.as_ptr().add(at) as *const _, self.used - at) };
            if n <= 0 {
                break;
            }
            at += n as usize;
        }
    }
}

unsafe extern "C" fn handler(signal: i32, info: *mut libc::siginfo_t, context: *mut libc::c_void) {
    // POSIX permits saving/restoring errno in a handler. Host FS remains installed, so this
    // libc TLS slot is valid even when the interrupted code is a guest instruction.
    let errno = unsafe { libc::__errno_location() };
    let saved_errno = unsafe { *errno };
    if signal == libc::SIGSEGV && !info.is_null() && !context.is_null() {
        let uc = unsafe { &*(context as *const libc::ucontext_t) };
        let error = uc.uc_mcontext.gregs[libc::REG_ERR as usize] as u64;
        if error & (2 | 16 | 32 | 64) == 2
            && crate::gpu::resident::linux_write_fault(unsafe { (*info).si_addr() } as u64)
        {
            DIRECT.fetch_add(1, Ordering::Relaxed);
            RECOVERED.fetch_add(1, Ordering::Relaxed);
            unsafe {
                *errno = saved_errno;
            }
            return;
        }
    }
    if signal == libc::SIGSEGV && !info.is_null() && !context.is_null() {
        let uc = unsafe { &*(context as *const libc::ucontext_t) };
        let error = uc.uc_mcontext.gregs[libc::REG_ERR as usize] as u64;
        if error & (16 | 32 | 64) == 0
            && unsafe {
                recover(
                    (*info).si_addr() as u64,
                    error & 2 != 0,
                    uc.uc_mcontext.gregs[libc::REG_RIP as usize] as u64,
                )
            }
        {
            unsafe {
                *errno = saved_errno;
            }
            return;
        }
    }
    let mut line = Line {
        bytes: [0; 2048],
        used: 0,
    };
    line.text(b"FATAL Linux signal ");
    line.hex(signal as u64);
    let uc = unsafe { &*(context as *const libc::ucontext_t) };
    let g = &uc.uc_mcontext.gregs;
    let rip = g[libc::REG_RIP as usize] as u64;
    line.text(b" rip=");
    line.hex(rip);
    if let Some(mods) = MODULES.get() {
        for (base, end, name) in mods {
            if rip >= *base && rip < *end {
                line.text(b" ");
                line.text(name.as_bytes());
                line.text(b"+");
                line.hex(rip - base);
                break;
            }
        }
    }
    line.text(b" address=");
    line.hex(unsafe { (*info).si_addr() } as u64);
    line.text(b" code=");
    line.hex(unsafe { (*info).si_code } as u64);
    for (name, reg) in [
        (b"\nrsp=".as_slice(), libc::REG_RSP),
        (b" rbp=", libc::REG_RBP),
        (b" rax=", libc::REG_RAX),
        (b" rbx=", libc::REG_RBX),
        (b" rcx=", libc::REG_RCX),
        (b" rdx=", libc::REG_RDX),
        (b"\nrdi=", libc::REG_RDI),
        (b" rsi=", libc::REG_RSI),
        (b" r8=", libc::REG_R8),
        (b" r9=", libc::REG_R9),
        (b"\nr10=", libc::REG_R10),
        (b" r11=", libc::REG_R11),
        (b" r12=", libc::REG_R12),
        (b" r13=", libc::REG_R13),
        (b" r14=", libc::REG_R14),
        (b" r15=", libc::REG_R15),
        (b"\neflags=", libc::REG_EFL),
        (b" page_error=", libc::REG_ERR),
    ] {
        line.text(name);
        line.hex(g[reg as usize] as u64);
    }
    line.text(b"\nstack words (rsp):");
    let mut words = [0u64; 24];
    let local = libc::iovec {
        iov_base: words.as_mut_ptr().cast(),
        iov_len: std::mem::size_of_val(&words),
    };
    let remote = libc::iovec {
        iov_base: g[libc::REG_RSP as usize] as *mut _,
        iov_len: local.iov_len,
    };
    let n = unsafe {
        libc::syscall(
            libc::SYS_process_vm_readv,
            libc::getpid(),
            &local,
            1,
            &remote,
            1,
            0,
        )
    };
    for word in words.iter().take(n.max(0) as usize / 8) {
        line.text(b" ");
        line.hex(*word);
    }
    line.text(b"\n");
    unsafe {
        line.write(libc::STDERR_FILENO);
        let fd = REPORT_FD.load(Ordering::Relaxed);
        if fd >= 0 {
            line.write(fd);
        }
        report::fatal(rip, g[libc::REG_RSI as usize] as u64, &words[..n.max(0) as usize / 8], fd);
        libc::_exit(128 + signal);
    }
}

/// Install once, before spawning guest threads. fd must remain open for the process lifetime.
pub fn install(fd: i32) -> std::io::Result<()> {
    REPORT_FD.store(fd, Ordering::Relaxed);
    ensure_installed()
}

pub fn ensure_installed() -> std::io::Result<()> {
    static INSTALLED: OnceLock<Result<(), i32>> = OnceLock::new();
    match INSTALLED
        .get_or_init(|| install_inner().map_err(|e| e.raw_os_error().unwrap_or(libc::EIO)))
    {
        Ok(()) => Ok(()),
        Err(e) => Err(std::io::Error::from_raw_os_error(*e)),
    }
}
fn install_inner() -> std::io::Result<()> {
    start_service()?;
    for signal in [
        libc::SIGSEGV,
        libc::SIGILL,
        libc::SIGBUS,
        libc::SIGFPE,
        libc::SIGTRAP,
    ] {
        let mut action: libc::sigaction = unsafe { std::mem::zeroed() };
        action.sa_sigaction = handler as *const () as usize;
        action.sa_flags = libc::SA_SIGINFO | libc::SA_ONSTACK;
        unsafe { libc::sigemptyset(&mut action.sa_mask) };
        if unsafe { libc::sigaction(signal, &action, std::ptr::null_mut()) } != 0 {
            return Err(std::io::Error::last_os_error());
        }
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    #[test]
    fn direct_resident_write_recovery() {
        if std::env::var_os("REGALIANT_DIRECT_TEST").is_none() {
            let out = std::process::Command::new(std::env::current_exe().unwrap())
                .args([
                    "--exact",
                    "platform::signals::tests::direct_resident_write_recovery",
                    "--nocapture",
                ])
                .env("REGALIANT_DIRECT_TEST", "1")
                .env("DS_SECTIONS", "1")
                .env("DS_RESIDENT", "0").env("DS_LINUX_WRITE_WATCH", "mprotect").env("DS_LINUX_VOLATILE_BACKOFF", "0")
                .env_remove("DS_LINUX_DIRECT_FAULT")
                .output()
                .unwrap();
            assert!(
                out.status.success(),
                "{} {}",
                String::from_utf8_lossy(&out.stdout),
                String::from_utf8_lossy(&out.stderr)
            );
            let out = std::process::Command::new(std::env::current_exe().unwrap())
                .args(["--exact", "platform::signals::tests::direct_resident_write_recovery", "--nocapture"])
                .env("REGALIANT_DIRECT_TEST", "readonly").env("DS_SECTIONS", "1").env("DS_RESIDENT", "0").env("DS_LINUX_WRITE_WATCH", "mprotect").env("DS_LINUX_VOLATILE_BACKOFF", "0")
                .env_remove("DS_LINUX_DIRECT_FAULT").output().unwrap();
            assert_eq!(out.status.code(), Some(139), "guest read-only protection must remain fatal: {}", String::from_utf8_lossy(&out.stderr));
            return;
        }
        use crate::gpu::resident as r;
        use crate::platform::os::*;
        crate::mem::reserve_guest_range().unwrap();
        crate::gpu::resident::init_linux_faults();
        let _stack = super::SignalStack::new().unwrap();
        super::ensure_installed().unwrap();
        let a =
            unsafe { VirtualAlloc(0x1700000000usize as _, 4096 * 8, MEM_COMMIT, PAGE_READWRITE) }
                as u64;
        assert_ne!(a, 0);
        let mut written = Vec::new();
        r::take_written(a, 4096 * 8, &mut written).unwrap();
        assert_eq!(written.len(), 8);
        // Concurrent faults can legitimately collide on LINUX_REMAP's reader CAS
        // and recover through the service. Require exact direct coverage in a serial
        // pass, then exact total recovery in the concurrent pass (including FPU/SIMD).
        for concurrent in [false, true] {
            let before = super::DIRECT.load(super::Ordering::Relaxed);
            let recovered = super::RECOVERED.load(super::Ordering::Relaxed);
            std::thread::scope(|scope| {
                for i in 0..8u64 {
                    let task = scope.spawn(move || unsafe {
                        let original = crate::platform::fpu::controls();
                        let guest = crate::platform::fpu::Controls { x87: 0x077f, mxcsr: 0xbfc0 };
                        crate::platform::fpu::set(guest);
                        let input = [0x12345678abcdef01u64, 0x7ff80000feedface];
                        let mut output = [0u64; 2];
                        // The fault interrupts a live SIMD value. Neither the direct handler nor
                        // the kernel's sigreturn may replace guest controls or registers.
                        std::arch::asm!("movdqu xmm0, [{src}]", "mov [{dst}], {value}", "movdqu [{out}], xmm0",
                            src = in(reg) input.as_ptr(), dst = in(reg) a + i * 4096,
                            value = in(reg) 100 + i, out = in(reg) output.as_mut_ptr(),
                            out("xmm0") _, options(nostack));
                        let restored = crate::platform::fpu::controls();
                        crate::platform::fpu::set(original);
                        assert_eq!(restored, guest);
                        assert_eq!(input, output);
                    });
                    if !concurrent { task.join().unwrap(); }
                }
            });
            let direct = super::DIRECT.load(super::Ordering::Relaxed) - before;
            if !concurrent { assert_eq!(direct, 8); }
            assert!(direct <= 8);
            assert_eq!(super::RECOVERED.load(super::Ordering::Relaxed) - recovered, 8);
            written.clear();
            r::take_written(a, 4096 * 8, &mut written).unwrap();
            assert_eq!(written.len(), 8);
        }
        // A kernel read into protected memory must be announced; it does not deliver SIGSEGV.
        r::host_write_begin(a, 4096);
        let fd = unsafe { libc::memfd_create(c"fault-test".as_ptr(), 0) };
        assert!(fd >= 0);
        let word = 777u64;
        assert_eq!(
            unsafe { libc::write(fd, (&word as *const u64).cast(), 8) },
            8
        );
        assert_eq!(unsafe { libc::pread(fd, a as _, 8, 0) }, 8);
        unsafe {
            libc::close(fd);
        }
        r::host_write_end(a, 4096);
        assert_eq!(unsafe { *(a as *const u64) }, 777);
        let guard = r::remap_guard();
        let mut old = 0;
        let readonly = std::env::var("REGALIANT_DIRECT_TEST").ok().as_deref() == Some("readonly");
        let protection = if readonly { PAGE_READONLY } else { PAGE_READWRITE };
        assert_ne!(
            unsafe { VirtualProtect(a as _, 4096 * 8, protection, &mut old) },
            0
        );
        r::remapped(a, 4096 * 8);
        drop(guard);
        if readonly {
            unsafe { std::ptr::write_volatile(a as *mut u64, 123); }
            panic!("write to a guest read-only page returned");
        }
        unsafe {
            VirtualFree(a as _, 0, MEM_RELEASE);
        }
    }

    #[test]
    fn synchronous_fault_report() {
        if let Ok(kind) = std::env::var("REGALIANT_SIGNAL_TEST") {
            let _stack = super::SignalStack::new().unwrap();
            super::install(-1).unwrap();
            super::super::tls::set(0, 123);
            unsafe {
                match kind.as_str() {
                    "ill" => std::arch::asm!("ud2", options(noreturn)),
                    "trap" => std::arch::asm!("int3"),
                    "fpe" => {
                        std::arch::asm!("xor edx, edx", "mov eax, 1", "xor ecx, ecx", "div ecx", out("rax") _, out("rdx") _, out("rcx") _)
                    }
                    "bus" => {
                        use std::os::fd::{AsRawFd, FromRawFd};
                        let fd = libc::memfd_create(
                            c"regaliant-sigbus-test".as_ptr(),
                            libc::MFD_CLOEXEC,
                        );
                        assert!(fd >= 0);
                        let fd = std::os::fd::OwnedFd::from_raw_fd(fd);
                        let memory = super::vm::Reservation::new_at(0, super::vm::PAGE).unwrap();
                        memory
                            .replace(
                                0,
                                super::vm::PAGE,
                                super::vm::READ,
                                Some((fd.as_raw_fd(), 0)),
                            )
                            .unwrap();
                        // Mapping beyond EOF raises SIGBUS even though the VMA is readable.
                        std::ptr::read_volatile(memory.base() as *const u64);
                    }
                    "segv" => {
                        let memory = super::vm::Reservation::new_at(0, super::vm::PAGE).unwrap();
                        std::ptr::write_volatile(memory.base() as *mut u64, 1);
                    }
                    _ => unreachable!(),
                }
            }
            panic!("fault did not terminate child");
        }
        for (kind, signal) in [
            ("ill", libc::SIGILL),
            ("trap", libc::SIGTRAP),
            ("fpe", libc::SIGFPE),
            ("segv", libc::SIGSEGV),
            ("bus", libc::SIGBUS),
        ] {
            let out = std::process::Command::new(std::env::current_exe().unwrap())
                .args([
                    "--exact",
                    "platform::signals::tests::synchronous_fault_report",
                    "--nocapture",
                ])
                .env("REGALIANT_SIGNAL_TEST", kind)
                .output()
                .unwrap();
            assert_eq!(
                out.status.code(),
                Some(128 + signal),
                "{kind}: {}",
                String::from_utf8_lossy(&out.stderr)
            );
            assert!(String::from_utf8_lossy(&out.stderr).contains("FATAL Linux signal"));
        }
    }
}

use std::sync::atomic::AtomicU64;
static REQUEST_FD: AtomicI32 = AtomicI32::new(-1);
static SERVICE_TID: AtomicI32 = AtomicI32::new(0);
pub static RECOVERED: AtomicU64 = AtomicU64::new(0);
struct Request {
    address: u64,
    write: bool,
    rip: u64,
    done: AtomicI32,
}

// Only raw syscalls, fixed stack storage and lock-free atomics run in signal context.
unsafe fn recover(address: u64, write: bool, rip: u64) -> bool {
    let fd = REQUEST_FD.load(Ordering::Acquire);
    if fd < 0
        || unsafe { libc::syscall(libc::SYS_gettid) } as i32 == SERVICE_TID.load(Ordering::Relaxed)
    {
        return false;
    }
    let request = Request {
        address,
        write,
        rip,
        done: AtomicI32::new(0),
    };
    let pointer = &request as *const Request as usize;
    if unsafe {
        libc::write(
            fd,
            (&pointer as *const usize).cast(),
            std::mem::size_of::<usize>(),
        )
    } != std::mem::size_of::<usize>() as isize
    {
        return false;
    }
    while request.done.load(Ordering::Acquire) == 0 {
        let timeout = libc::timespec {
            tv_sec: 30,
            tv_nsec: 0,
        };
        let result = unsafe {
            libc::syscall(
                libc::SYS_futex,
                request.done.as_ptr(),
                libc::FUTEX_WAIT | libc::FUTEX_PRIVATE_FLAG,
                0,
                &timeout,
            )
        };
        if result == -1
            && unsafe { *libc::__errno_location() } == libc::ETIMEDOUT
            && request.done.load(Ordering::Acquire) == 0
        {
            // Never abandon a live request: its address is on this signal stack. Terminate
            // while it remains valid if recovery deadlocks or the GPU stops responding.
            let mut line = Line {
                bytes: [0; 2048],
                used: 0,
            };
            line.text(b"FATAL Linux fault recovery timed out (30s) address=");
            line.hex(address);
            line.text(b" rip=");
            line.hex(rip);
            line.text(b" write=");
            line.hex(write as u64);
            line.text(b" tid=");
            line.hex(unsafe { libc::syscall(libc::SYS_gettid) } as u64);
            line.text(b"\n");
            unsafe {
                line.write(libc::STDERR_FILENO);
                let fd = REPORT_FD.load(Ordering::Relaxed);
                if fd >= 0 {
                    line.write(fd);
                }
                libc::_exit(126);
            }
        }
    }
    request.done.load(Ordering::Acquire) == 1
}
fn answer(pointer: usize) {
    let r = unsafe { &*(pointer as *const Request) };
    crate::gpu::resident::FAULT_RIP.with(|v| v.set(r.rip));
    let generation = crate::sections::split_generation();
    let split = crate::sections::wait_split(r.address);
    let resident = crate::gpu::resident::on_fault(r.address, r.write);
    let tracked = r.write && crate::gpu::track::on_write_fault(r.address);
    // Run after the renderer has disarmed its page, preserving both dirty records.
    let watched = r.write && super::os::write_watch_fault(r.address as usize);
    let remapped = !(split || resident || tracked || watched)
        && super::os::mapping_allows(r.address as usize, r.write);
    // Another split can start after wait_split returned false and make the subsequent
    // mapping query see a placeholder. A completed split's generation also covers the case
    // where both its beginning and end fell between the checks. Retry against the new mapping.
    let split_race = generation != crate::sections::split_generation();
    let handled = split || resident || tracked || watched || remapped || split_race;
    if handled {
        let n = RECOVERED.fetch_add(1, Ordering::Relaxed) + 1;
        if n <= 8 || n % 10000 == 0 {
            crate::info!(
                "Linux recovered fault #{n}: {:#x} write={} rip={:#x} split={split} resident={resident} tracked={tracked} write_watch={watched} remapped={remapped} split_race={split_race}",
                r.address,
                r.write,
                r.rip
            );
        }
    }
    if !handled {
        crate::error!(
            "Linux unrecovered page fault: address={:#x} rip={} write={} host={}",
            r.address,
            crate::loader::describe_addr(r.rip),
            r.write,
            crate::mem::describe_host(r.address, 1)
        );
        if let Some(memory) = crate::mem::MEM.try_lock() {
            if let Some(v) = memory.find(r.address) {
                crate::error!("guest VMA: {v:?}");
            }
        }
        if let Some(dir) = crate::run_dir() {
            if let Ok(maps) = std::fs::read("/proc/self/maps") {
                let _ = std::fs::write(dir.join("fault-maps.txt"), maps);
            }
        }
        crate::log::flush();
    }
    let done = r.done.as_ptr();
    r.done.store(if handled { 1 } else { 2 }, Ordering::Release);
    // The handler may return immediately; do not dereference the request after this store.
    unsafe {
        libc::syscall(
            libc::SYS_futex,
            done,
            libc::FUTEX_WAKE | libc::FUTEX_PRIVATE_FLAG,
            1,
        );
    }
}
fn start_service() -> std::io::Result<()> {
    let mut pipe = [-1; 2];
    if unsafe { libc::pipe2(pipe.as_mut_ptr(), libc::O_CLOEXEC) } != 0 {
        return Err(std::io::Error::last_os_error());
    }
    REQUEST_FD.store(pipe[1], Ordering::Release);
    std::thread::Builder::new()
        .name("linux-fault-service".into())
        .stack_size(2 << 20)
        .spawn(move || {
            SERVICE_TID.store(
                unsafe { libc::syscall(libc::SYS_gettid) } as i32,
                Ordering::Release,
            );
            let _stack = SignalStack::new().expect("fault service signal stack");
            loop {
                let mut pointer = 0usize;
                let n = unsafe {
                    libc::read(
                        pipe[0],
                        (&mut pointer as *mut usize).cast(),
                        std::mem::size_of::<usize>(),
                    )
                };
                if n < 0 && std::io::Error::last_os_error().raw_os_error() == Some(libc::EINTR) {
                    continue;
                }
                if n != std::mem::size_of::<usize>() as isize {
                    unsafe {
                        libc::_exit(126);
                    }
                }
                let address = unsafe { (*(pointer as *const Request)).address };
                if crate::gpu::resident::fault_may_wait(address) {
                    if std::thread::Builder::new()
                        .name("linux-fault-readback".into())
                        .spawn(move || {
                            let _stack = SignalStack::new().expect("readback signal stack");
                            answer(pointer);
                        })
                        .is_err()
                    {
                        unsafe {
                            libc::_exit(126);
                        }
                    }
                } else {
                    answer(pointer);
                }
            }
        })?;
    Ok(())
}
