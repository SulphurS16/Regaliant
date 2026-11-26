//! Bounded, on-demand hang evidence. Never suspend a target while taking host locks.
//! The signal handler only uses atomics and process_vm_readv; formatting is on the watchdog.
use std::sync::atomic::{AtomicI32, AtomicU64, AtomicUsize, Ordering::*};
use std::io::Write;

const LIMIT: usize = 512;
struct Sample {
    tid: AtomicI32,
    state: AtomicUsize, // 0 unused, 1 requested, 2 handler, 3 ready
    registers: [AtomicU64; 3], // RIP, RSP, RBP
    stack: [AtomicU64; 64],
    returns: [AtomicU64; 32],
}
static SAMPLES: [Sample; LIMIT] = [const { Sample {
    tid: AtomicI32::new(0), state: AtomicUsize::new(0),
    registers: [const { AtomicU64::new(0) }; 3],
    stack: [const { AtomicU64::new(0) }; 64],
    returns: [const { AtomicU64::new(0) }; 32],
} }; LIMIT];
static REPORT: parking_lot::Mutex<()> = parking_lot::Mutex::new(());

unsafe fn read(at: u64, words: &mut [u64]) -> usize {
    let len = std::mem::size_of_val(words);
    let local = libc::iovec { iov_base: words.as_mut_ptr().cast(), iov_len: len };
    let remote = libc::iovec { iov_base: at as _, iov_len: len };
    let n = unsafe { libc::syscall(libc::SYS_process_vm_readv, libc::getpid(), &local, 1, &remote, 1, 0) };
    n.max(0) as usize / 8
}
unsafe extern "C" fn handler(_: i32, _: *mut libc::siginfo_t, context: *mut libc::c_void) {
    let errno = unsafe { libc::__errno_location() };
    let saved = unsafe { *errno };
    let tid = unsafe { libc::syscall(libc::SYS_gettid) } as i32;
    if let Some(s) = SAMPLES.iter().find(|s| s.tid.load(Acquire) == tid) {
        if !context.is_null() && s.state.compare_exchange(1, 2, AcqRel, Acquire).is_ok() {
            let g = &unsafe { &*(context as *const libc::ucontext_t) }.uc_mcontext.gregs;
            for (v, r) in s.registers.iter().zip([libc::REG_RIP, libc::REG_RSP, libc::REG_RBP]) {
                v.store(g[r as usize] as u64, Relaxed);
            }
            let mut words = [0; 64];
            let n = unsafe { read(g[libc::REG_RSP as usize] as u64, &mut words) };
            for (v, w) in s.stack.iter().zip(&words[..n]) { v.store(*w, Relaxed); }
            let mut bp = g[libc::REG_RBP as usize] as u64;
            for v in &s.returns {
                let mut pair = [0; 2];
                if unsafe { read(bp, &mut pair) } != 2 { break; }
                v.store(pair[1], Relaxed);
                if pair[0] <= bp || pair[0] - bp > 16 * 1024 * 1024 { break; }
                bp = pair[0];
            }
            s.state.store(3, Release);
        }
    }
    unsafe { *errno = saved };
}

fn arm() -> std::io::Result<i32> {
    let signal = libc::SIGRTMIN() + 6;
    let mut old: libc::sigaction = unsafe { std::mem::zeroed() };
    if unsafe { libc::sigaction(signal, std::ptr::null(), &mut old) } != 0 { return Err(std::io::Error::last_os_error()); }
    if old.sa_sigaction == handler as *const () as usize { return Ok(signal); }
    if old.sa_sigaction != libc::SIG_DFL { return Err(std::io::Error::other("sampling signal already owned; left unchanged")); }
    let mut action: libc::sigaction = unsafe { std::mem::zeroed() };
    action.sa_sigaction = handler as *const () as usize;
    action.sa_flags = libc::SA_SIGINFO | libc::SA_ONSTACK | libc::SA_RESTART;
    unsafe { libc::sigemptyset(&mut action.sa_mask) };
    if unsafe { libc::sigaction(signal, &action, std::ptr::null_mut()) } != 0 { return Err(std::io::Error::last_os_error()); }
    Ok(signal)
}

/// Snapshots include all threads (not only guest or CP threads). An unanswered prior
/// request prevents slot reuse, but kernel/HLE wait records can still be taken again.
/// Missing/blocked signals are explicit. Raw stack words are candidates, not an unwind.
/// Maps allow offline symbolization without dladdr/loader locks in a deadlocked process.
pub fn dump(dir: &std::path::Path, why: &str) -> std::io::Result<()> {
    let _report = REPORT.try_lock().ok_or_else(|| std::io::Error::other("another thread report is in progress"))?;
    let mut f = std::fs::File::create(dir.join("hang-threads.txt"))?;
    writeln!(f, "{why}\nNon-atomic snapshot; raw stack words are not a backtrace.")?;
    let maps = std::fs::read("/proc/self/maps")?;
    let mut m = std::fs::File::create(dir.join("hang-maps.txt"))?;
    m.write_all(&maps)?; m.sync_all()?;
    let mut tasks: Vec<_> = std::fs::read_dir("/proc/self/task")?.flatten().filter_map(|t| {
        Some((t.file_name().to_string_lossy().parse::<i32>().ok()?, t.path()))
    }).collect();
    tasks.sort_by_key(|t| t.0);
    let signal = if SAMPLES.iter().any(|s| matches!(s.state.load(Acquire), 1 | 2)) {
        Err(std::io::Error::other("previous stack sample still pending; new wait records only"))
    } else {
        // All previous handlers have published completion. Never recycle a slot that a
        // delayed signal could still write. Pending requests remain valid for their owner.
        for s in &SAMPLES {
            s.state.store(0, Release);
            s.tid.store(0, Release);
            for word in s.registers.iter().chain(&s.stack).chain(&s.returns) { word.store(0, Relaxed); }
        }
        arm()
    };
    writeln!(f, "sampler: {signal:?}; threads={} capacity={LIMIT}", tasks.len())?;
    let guests = crate::thread::THREADS.try_lock().map(|g| g.clone()).unwrap_or_default();
    for (i, (tid, path)) in tasks.iter().enumerate() {
        writeln!(f, "\ntid {tid}")?;
        for name in ["comm", "stat", "wchan", "syscall", "stack", "status"] {
            match std::fs::read_to_string(path.join(name)) {
                Ok(v) => writeln!(f, "{name}: {}", v.trim_end())?,
                Err(e) => writeln!(f, "{name}: unavailable ({e})")?,
            }
        }
        for t in guests.iter().filter(|t| t.host_tid.load(Relaxed) == *tid as u64) {
            // Do not call detail callbacks which may need an object's lock or lifetime.
            if let Some(w) = t.wait.try_lock() {
                if let Some(w) = w.as_ref() { writeln!(f, "HLE: {} object={:#x} elapsed={:.3}s", w.what, w.obj, w.since.elapsed().as_secs_f64())?; }
                else { writeln!(f, "HLE: no registered wait; guest id={}", t.id)?; }
            } else { writeln!(f, "HLE: wait registry busy")?; }
        }
        if i < LIMIT {
            if let Ok(signal) = signal {
                let s = &SAMPLES[i];
                s.tid.store(*tid, Release); s.state.store(1, Release);
                let r = unsafe { libc::syscall(libc::SYS_tgkill, libc::getpid(), *tid, signal) };
                if r != 0 { writeln!(f, "sample request failed: {}", std::io::Error::last_os_error())?; }
            }
        }
    }
    f.sync_all()?;
    // All requests sent together: one bounded deadline, not N per-thread timeouts.
    std::thread::sleep(std::time::Duration::from_millis(200));
    for s in SAMPLES.iter().take(if signal.is_ok() { tasks.len().min(LIMIT) } else { 0 }) {
        let tid = s.tid.load(Acquire);
        if s.state.load(Acquire) != 3 { writeln!(f, "tid {tid}: sample unavailable (exited, signal blocked or not scheduled)")?; continue; }
        let regs = s.registers.each_ref().map(|a| a.load(Relaxed));
        writeln!(f, "tid {tid}: rip={:#x} rsp={:#x} rbp={:#x}", regs[0], regs[1], regs[2])?;
        for (name, words) in [("frame-pointer returns", s.returns.as_slice()), ("raw stack", s.stack.as_slice())] {
            write!(f, "{name}:")?;
            for w in words { write!(f, " {:#x}", w.load(Relaxed))?; }
            writeln!(f)?;
        }
    }
    writeln!(f, "{}", crate::gpu::cp::hang_waits())?;
    f.sync_all()
}

#[cfg(test)]
mod tests {
    #[test]
    fn hang_report_samples_waiting_host_threads() {
        const NAME: &str = "platform::hang::tests::hang_report_samples_waiting_host_threads";
        if std::env::var_os("REGALIANT_HANG_TEST").is_none() {
            let out = std::process::Command::new(std::env::current_exe().unwrap()).args(["--exact", NAME, "--nocapture"])
                .env("REGALIANT_HANG_TEST", "1").output().unwrap();
            assert!(out.status.success(), "{}", String::from_utf8_lossy(&out.stderr)); return;
        }
        let (tx, rx) = std::sync::mpsc::channel();
        let (ready, started) = std::sync::mpsc::channel();
        let worker = std::thread::Builder::new().name("hang-test-sleeper".into()).spawn(move || {
            ready.send(unsafe { libc::syscall(libc::SYS_gettid) } as i32).unwrap();
            rx.recv().unwrap();
        }).unwrap();
        let tid = started.recv().unwrap();
        let dir = std::env::temp_dir().join(format!("regaliant-hang-{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        super::dump(&dir, "test hang").unwrap();
        let text = std::fs::read_to_string(dir.join("hang-threads.txt")).unwrap();
        assert!(text.contains("hang-test-sleep")); assert!(text.contains(&format!("tid {tid}: rip=0x")));
        assert!(text.contains("raw stack:")); assert!(text.contains("wchan:"));
        // Ctrl+F10 may precede a later watchdog: it must not consume the only report.
        super::dump(&dir, "second report").unwrap();
        let text = std::fs::read_to_string(dir.join("hang-threads.txt")).unwrap();
        assert!(text.contains("second report"));
        assert!(text.contains(&format!("tid {tid}: rip=0x")));
        tx.send(()).unwrap(); worker.join().unwrap();
        std::fs::remove_dir_all(dir).unwrap();
    }
}
