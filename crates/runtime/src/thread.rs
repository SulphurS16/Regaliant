//! Guest threads: each guest thread is a host thread running guest code natively
//! on its own stack, with its TLS slot pointing at the guest thread pointer.

use parking_lot::{Condvar, Mutex};
use std::cell::Cell;
use std::sync::atomic::{AtomicBool, AtomicI32, AtomicU64, Ordering};

std::arch::global_asm!(
    // u64 ds_guest_thread_call(entry, a0, a1, *mut saved_rsp)  [sysv64]
    ".globl ds_guest_thread_call",
    "ds_guest_thread_call:",
    "push rbp",
    "mov rbp, rsp",
    "push rbx",
    "push r12",
    "push r13",
    "push r14",
    "push r15",
    "sub rsp, 8",
    "mov [rcx], rsp",
    "mov rax, rdi",
    "mov rdi, rsi",
    "mov rsi, rdx",
    "call rax",
    "ds_guest_thread_resume:",
    "add rsp, 8",
    "pop r15",
    "pop r14",
    "pop r13",
    "pop r12",
    "pop rbx",
    "pop rbp",
    "ret",
    // ! ds_guest_thread_exit(saved_rsp, value)  [sysv64]
    ".globl ds_guest_thread_exit",
    "ds_guest_thread_exit:",
    "mov rsp, rdi",
    "mov rax, rsi",
    "jmp ds_guest_thread_resume",
);

unsafe extern "sysv64" {
    fn ds_guest_thread_call(entry: u64, a0: u64, a1: u64, saved_rsp: *mut u64) -> u64;
    fn ds_guest_thread_exit(saved_rsp: u64, value: u64) -> !;
}

pub const THREAD_MAGIC: u64 = 0x4453_5448_5245_4144; // "DSTHREAD"
pub const PTHREAD_KEYS_MAX: usize = 256;

pub struct GuestThread {
    pub magic: u64,
    pub id: u64,
    pub name: Mutex<String>,
    pub tp: AtomicU64,
    pub entry: u64,
    pub arg: u64,
    pub stack_size: u64,
    pub detached: AtomicBool,
    pub exited: Mutex<Option<u64>>,
    pub exit_cv: Condvar,
    pub saved_rsp: AtomicU64,
    pub errno: AtomicI32,
    pub priority: AtomicI32,
    pub affinity: AtomicU64,
    pub specific: Mutex<Vec<u64>>,
    pub host_tid: AtomicU64,
    /// Set while the thread is blocked inside an HLE wait (stall diagnostics).
    pub wait: Mutex<Option<WaitInfo>>,
    /// Milliseconds since process start when the thread exited (0 = running).
    pub exit_ms: AtomicU64,
}

/// What a guest thread is blocked on inside an HLE wait.
pub struct WaitInfo {
    pub what: &'static str,
    pub obj: u64,
    pub since: std::time::Instant,
    /// Object-specific details (owner, count, ...); must not block.
    pub detail: Option<fn(u64) -> String>,
}

/// Clears the current thread's wait record when dropped.
pub struct WaitGuard(Option<&'static GuestThread>);

impl Drop for WaitGuard {
    fn drop(&mut self) {
        if let Some(t) = self.0 {
            *t.wait.lock() = None;
        }
    }
}

/// Records that the current guest thread is about to block on `obj`.
pub fn waiting(what: &'static str, obj: u64, detail: Option<fn(u64) -> String>) -> WaitGuard {
    let t = current();
    if let Some(t) = t {
        *t.wait.lock() = Some(WaitInfo { what, obj, since: std::time::Instant::now(), detail });
    }
    WaitGuard(t)
}

/// "blocked 1.2s in mutex 0x... <detail>" for a thread inside an HLE wait.
pub fn describe_wait(t: &GuestThread) -> Option<String> {
    let w = t.wait.try_lock()?;
    let w = w.as_ref()?;
    let mut s = format!("blocked {:.2}s in {} {:#x}", w.since.elapsed().as_secs_f64(), w.what, w.obj);
    if let Some(d) = w.detail {
        s.push(' ');
        s.push_str(&d(w.obj));
    }
    Some(s)
}

/// Guest thread id -> name, without blocking (diagnostics).
pub fn name_of(id: u64) -> String {
    match THREADS.try_lock() {
        Some(list) => list.iter().find(|t| t.id == id).map(|t| format!("{id} '{}'", t.name())).unwrap_or_else(|| id.to_string()),
        None => id.to_string(),
    }
}

impl GuestThread {
    pub fn handle(&'static self) -> u64 {
        self as *const _ as u64
    }
    pub fn name(&self) -> String {
        self.name.lock().clone()
    }
}

static NEXT_ID: AtomicU64 = AtomicU64::new(1);
pub static THREADS: Mutex<Vec<&'static GuestThread>> = Mutex::new(Vec::new());

thread_local! {
    static CURRENT: Cell<Option<&'static GuestThread>> = const { Cell::new(None) };
}

pub fn current() -> Option<&'static GuestThread> {
    CURRENT.with(|c| c.get())
}

/// Validates a guest-supplied thread handle.
pub fn from_handle(h: u64) -> Option<&'static GuestThread> {
    if h == 0 {
        return None;
    }
    let threads = THREADS.lock();
    threads.iter().copied().find(|t| t.handle() == h && t.magic == THREAD_MAGIC)
}

pub fn new_thread(name: &str, entry: u64, arg: u64, stack_size: u64, priority: i32) -> &'static GuestThread {
    let t: &'static GuestThread = Box::leak(Box::new(GuestThread {
        magic: THREAD_MAGIC,
        id: NEXT_ID.fetch_add(1, Ordering::Relaxed),
        name: Mutex::new(name.to_string()),
        tp: AtomicU64::new(0),
        entry,
        arg,
        stack_size,
        detached: AtomicBool::new(false),
        exited: Mutex::new(None),
        exit_cv: Condvar::new(),
        saved_rsp: AtomicU64::new(0),
        errno: AtomicI32::new(0),
        priority: AtomicI32::new(priority),
        affinity: AtomicU64::new(0x7F),
        specific: Mutex::new(vec![0; PTHREAD_KEYS_MAX]),
        host_tid: AtomicU64::new(0),
        wait: Mutex::new(None),
        exit_ms: AtomicU64::new(0),
    }));
    let tp = crate::tls::alloc_thread_area(t.handle());
    t.tp.store(tp, Ordering::SeqCst);
    THREADS.lock().push(t);
    t
}

/// Runs guest code `entry(a0, a1)` on the current (already entered) thread.
/// Returns when the function returns or the thread calls pthread_exit.
pub fn run_entry(t: &'static GuestThread, entry: u64, a0: u64, a1: u64) -> u64 {
    unsafe { ds_guest_thread_call(entry, a0, a1, t.saved_rsp.as_ptr()) }
}

/// Terminates the current guest thread with `value` (pthread_exit semantics).
pub fn exit_current(value: u64) -> ! {
    let t = current().expect("pthread_exit on a non-guest thread");
    run_key_destructors(t);
    crate::hle::libs::fiber::leave_fibers();
    let rsp = t.saved_rsp.load(Ordering::SeqCst);
    unsafe { ds_guest_thread_exit(rsp, value) }
}

fn finish(t: &'static GuestThread, value: u64) {
    t.exit_ms.store(crate::log::elapsed_ms() as u64, Ordering::SeqCst);
    *t.exited.lock() = Some(value);
    t.exit_cv.notify_all();
    crate::debug!("thread '{}' exited with {:#x}", t.name(), value);
}

pub fn run_key_destructors(t: &'static GuestThread) {
    // POSIX: up to PTHREAD_DESTRUCTOR_ITERATIONS rounds.
    for _ in 0..4 {
        let mut any = false;
        for key in 0..PTHREAD_KEYS_MAX {
            let val = {
                let mut s = t.specific.lock();
                std::mem::replace(&mut s[key], 0)
            };
            if val == 0 {
                continue;
            }
            if let Some(dtor) = crate::hle::kernel::pthread::key_destructor(key) {
                any = true;
                unsafe {
                    crate::hle::call_guest(dtor, &[val]);
                }
            }
        }
        if !any {
            break;
        }
    }
}

#[cfg(windows)]
#[path = "platform/windows/thread.rs"]
mod host;
#[cfg(target_os = "linux")]
#[path = "platform/linux/thread.rs"]
mod host;
pub use host::*;

/// Live Bink I/O threads: the title's movie player makes one per movie and it exits with it.
static MOVIE_THREADS: AtomicU64 = AtomicU64::new(0);

/// A Bink movie is playing (the PlayStation Studios logo, the attract movie, ...): presentation
/// then follows the display's refresh whatever the frame rate limit (`videoout`), and frame
/// generation stays off.
pub fn movie_playing() -> bool {
    MOVIE_THREADS.load(Ordering::Relaxed) != 0
}

/// Starts a host thread that runs a guest thread function.
pub fn spawn(t: &'static GuestThread) -> std::io::Result<()> {
    let host_stack = (t.stack_size as usize).max(256 << 10) + (1 << 20);
    spawn_host(
        &t.name(),
        host_stack,
        Box::new(move || {
            enter(t);
            crate::debug!("thread '{}' start entry={:#x} arg={:#x}", t.name(), t.entry, t.arg);
            // (A movie plays while its Bink I/O thread lives: `movie_playing`.)
            let movie = t.name() == "Bink IO";
            if movie {
                let n = MOVIE_THREADS.fetch_add(1, Ordering::SeqCst) + 1;
                crate::info!("movie playing (Bink I/O thread {} began; {n} live)", t.id);
            }
            let v = run_entry(t, t.entry, t.arg, 0);
            run_key_destructors(t);
            if movie {
                let n = MOVIE_THREADS.fetch_sub(1, Ordering::SeqCst) - 1;
                crate::info!("movie ended (Bink I/O thread {} exited; {n} live)", t.id);
            }
            finish(t, v);
        }),
    )
}

pub fn join(t: &'static GuestThread) -> u64 {
    let mut g = t.exited.lock();
    let _w = if g.is_none() { Some(waiting("join", t.handle(), Some(|h| format!("thread {}", unsafe { &*(h as *const GuestThread) }.name())))) } else { None };
    while g.is_none() {
        t.exit_cv.wait(&mut g);
    }
    g.unwrap()
}
