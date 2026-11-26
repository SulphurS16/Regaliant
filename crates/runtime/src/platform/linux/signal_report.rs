//! Fatal-only diagnostics. Read current mappings at the fault, including dlopen'd drivers.
//! No loader lock, allocation, logger, TLS or file formatting in the signal path.
use super::Line;
use std::sync::atomic::{AtomicI32, AtomicU8, Ordering};

static CRASH_FD: AtomicI32 = AtomicI32::new(-1);
struct Slot {
    tid: AtomicI32,
    text: [AtomicU8; 1024],
}
static COMPILES: [Slot; 64] = [const { Slot {
    tid: AtomicI32::new(0), text: [const { AtomicU8::new(0) }; 1024],
} }; 64];

/// Open before compiling, never from the handler. Same driver-scoped quarantine as Windows.
pub fn compiler_crash_file(path: &std::path::Path) {
    use std::os::fd::IntoRawFd;
    if let Ok(f) = std::fs::OpenOptions::new().create(true).append(true).open(path) {
        let fd = f.into_raw_fd();
        if CRASH_FD.compare_exchange(-1, fd, Ordering::AcqRel, Ordering::Acquire).is_err() {
            unsafe { libc::close(fd); }
        }
    }
}

/// Preformat the driver id and compile key outside the handler. Fixed atomic slots also let
/// a driver's own crashing worker report the caller(s), without attributing their pipeline.
pub struct CompileNote(Option<usize>);
impl CompileNote {
    pub fn new(text: &str) -> Self {
        let tid = unsafe { libc::syscall(libc::SYS_gettid) as i32 };
        for (i, s) in COMPILES.iter().enumerate() {
            if s.tid.compare_exchange(0, -1, Ordering::Acquire, Ordering::Relaxed).is_ok() {
                for (j, b) in s.text.iter().enumerate() {
                    b.store(if j < 1023 { text.as_bytes().get(j).copied().unwrap_or(0) } else { 0 }, Ordering::Relaxed);
                }
                s.tid.store(tid, Ordering::Release);
                return Self(Some(i));
            }
        }
        Self(None)
    }
}
impl Drop for CompileNote {
    fn drop(&mut self) {
        if let Some(i) = self.0 { COMPILES[i].tid.store(0, Ordering::Release); }
    }
}

fn mapping(line: &[u8]) -> Option<(u64, u64, u64)> {
    fn hex(v: &[u8]) -> Option<u64> {
        if v.is_empty() { return None; }
        v.iter().try_fold(0u64, |a, b| {
            let n = match b { b'0'..=b'9' => b - b'0', b'a'..=b'f' => b - b'a' + 10, _ => return None };
            a.checked_mul(16)?.checked_add(n as u64)
        })
    }
    let mut w = line.split(|b| *b == b' ').filter(|w| !w.is_empty());
    let mut range = w.next()?.split(|b| *b == b'-');
    let start = hex(range.next()?)?;
    let end = hex(range.next()?)?;
    w.next()?; // permissions
    Some((start, end, hex(w.next()?)?))
}

unsafe fn emit(line: &Line, fd: i32) {
    unsafe { line.write(libc::STDERR_FILENO); if fd >= 0 { line.write(fd); } }
}
unsafe fn read_memory(addr: u64, bytes: &mut [u8]) -> usize {
    let local = libc::iovec { iov_base: bytes.as_mut_ptr().cast(), iov_len: bytes.len() };
    let remote = libc::iovec { iov_base: addr as *mut _, iov_len: bytes.len() };
    unsafe { libc::syscall(libc::SYS_process_vm_readv, libc::getpid(), &local, 1, &remote, 1, 0).max(0) as usize }
}

pub(super) unsafe fn fatal(rip: u64, rsi: u64, words: &[u64], fd: i32) {
    let tid = unsafe { libc::syscall(libc::SYS_gettid) as i32 };
    let mut name = [0u8; 16];
    unsafe { libc::syscall(libc::SYS_prctl, libc::PR_GET_NAME, name.as_mut_ptr(), 0, 0, 0); }
    let mut l = Line { bytes: [0; 2048], used: 0 };
    l.text(b"fault tid="); l.hex(tid as u64); l.text(b" name=");
    l.text(&name[..name.iter().position(|b| *b == 0).unwrap_or(name.len())]);
    l.text(b"\nrip bytes:");
    let mut bytes = [0u8; 32];
    let n = unsafe { read_memory(rip, &mut bytes) };
    for b in &bytes[..n] {
        l.text(&[b' ', b"0123456789abcdef"[(b >> 4) as usize], b"0123456789abcdef"[(b & 15) as usize]]);
    }
    if n == 0 { l.text(b" unreadable"); }
    l.text(b"\n"); unsafe { emit(&l, fd); }
    // NVIDIA's compiler fatal callback takes a message pointer in RSI. Read it safely even
    // for other faults; never dereference a register directly or assume it is a C string.
    let mut message = [0u8; 512];
    let n = unsafe { read_memory(rsi, &mut message) };
    if n != 0 {
        l.used = 0; l.text(b"rsi pointee (printable; may be compiler diagnostic): ");
        for &b in &message[..n] {
            if b == 0 { break; }
            l.text(&[if b.is_ascii_graphic() || b == b' ' { b } else { b'.' }]);
        }
        l.text(b"\n"); unsafe { emit(&l, fd); }
    }
    // Stream /proc, rather than a fixed full snapshot that truncates late mappings or one
    // captured at startup before the driver loads its compiler. Bounded memory, raw syscalls.
    let maps = unsafe { libc::syscall(libc::SYS_openat, libc::AT_FDCWD, c"/proc/self/maps".as_ptr(), libc::O_RDONLY | libc::O_CLOEXEC, 0) as i32 };
    let mut compiler = false;
    let mut found = [false; 25];
    if maps >= 0 {
        let mut buf = [0u8; 4096];
        let mut row = [0u8; 1536];
        let mut used = 0;
        loop {
            let n = unsafe { libc::syscall(libc::SYS_read, maps, buf.as_mut_ptr(), buf.len()) };
            if n <= 0 { break; }
            for &b in &buf[..n as usize] {
                if b != b'\n' { if used < row.len() { row[used] = b; used += 1; } continue; }
                if let Some((base, end, offset)) = mapping(&row[..used]) {
                    for (i, addr) in std::iter::once(rip).chain(words.iter().copied()).enumerate() {
                        if addr < base || addr >= end { continue; }
                        found[i] = true;
                        if i == 0 { compiler = row[..used].windows(b"libnvidia-gpucomp.so".len()).any(|w| w == b"libnvidia-gpucomp.so"); }
                        l.used = 0;
                        if i == 0 { l.text(b"rip mapping "); } else { l.text(b"stack["); l.hex((i - 1) as u64); l.text(b"] mapping "); }
                        l.hex(addr); l.text(b" file_offset="); l.hex(offset + addr - base);
                        l.text(b" "); l.text(&row[..used]); l.text(b"\n"); unsafe { emit(&l, fd); }
                    }
                }
                used = 0;
            }
        }
        unsafe { libc::syscall(libc::SYS_close, maps); }
    }
    for (i, addr) in std::iter::once(rip).chain(words.iter().copied()).enumerate() {
        if !found[i] {
            l.used = 0; l.text(b"unmapped address index (rip=0): "); l.hex(i as u64);
            l.text(b" address="); l.hex(addr); l.text(b"\n"); unsafe { emit(&l, fd); }
        }
    }
    for s in &COMPILES {
        let owner = s.tid.load(Ordering::Acquire);
        if owner <= 0 { continue; }
        let mut note = Line { bytes: [0; 2048], used: 0 };
        for b in &s.text {
            let b = b.load(Ordering::Relaxed);
            if b == 0 { break; }
            note.text(&[b]);
        }
        if s.tid.load(Ordering::Acquire) != owner { continue; }
        l.used = 0; l.text(b"active pipeline tid="); l.hex(owner as u64);
        l.text(if owner == tid { b" (faulting thread): " } else { b" (other thread; not attributed): " });
        l.text(&note.bytes[..note.used]); l.text(b"\n"); unsafe { emit(&l, fd); }
        // Only a confirmed compiler mapping on the caller's own thread. Never blacklist all
        // in-flight compiles because an internal driver worker died without attribution.
        if compiler && owner == tid && CRASH_FD.load(Ordering::Relaxed) >= 0 {
            note.text(b"\n");
            unsafe { note.write(CRASH_FD.load(Ordering::Relaxed)); }
        }
    }
}

#[cfg(test)]
mod tests {
    #[test]
    fn map_offsets_and_malformed_rows() {
        assert_eq!(super::mapping(b"7fc044d98000-7fc049599000 r-xp 00398000 00:01 123 /lib/driver.so"), Some((0x7fc044d98000, 0x7fc049599000, 0x398000)));
        assert_eq!(0x398000 + 0x7fc044d9b07eu64 - 0x7fc044d98000, 0x39b07e);
        for s in [b"".as_slice(), b"bad", b"ffffffffff0ffffff-ff r-xp 0"] { assert!(super::mapping(s).is_none()); }
    }
    #[test]
    fn attributed_compiler_fault_persists_only_its_note() {
        const CHILD: &str = "REGALIANT_COMPILER_FAULT_TEST";
        if let Some(path) = std::env::var_os(CHILD) {
            use std::os::fd::AsRawFd;
            let root = std::path::PathBuf::from(path);
            super::super::ensure_installed().unwrap();
            let _stack = super::super::SignalStack::new().unwrap();
            super::compiler_crash_file(&root.join("crashed.txt"));
            let _note = super::CompileNote::new("c0000000000000123 test-driver compute pipeline");
            // A file-backed executable stub, named for the mapping classifier. Exercises the
            // same native signal path without loading Vulkan or deliberately crashing a driver.
            let path = root.join("libnvidia-gpucomp.so.test");
            std::fs::write(&path, [0x0f, 0x0b]).unwrap();
            let file = std::fs::File::open(path).unwrap();
            unsafe {
                let code = libc::mmap(std::ptr::null_mut(), 4096, libc::PROT_READ | libc::PROT_EXEC, libc::MAP_PRIVATE, file.as_raw_fd(), 0);
                assert_ne!(code, libc::MAP_FAILED);
                std::mem::transmute::<*mut libc::c_void, extern "C" fn()>(code)();
            }
        } else {
            let root = std::env::temp_dir().join(format!("regaliant-compiler-fault-{}", std::process::id()));
            std::fs::create_dir(&root).unwrap();
            let out = std::process::Command::new(std::env::current_exe().unwrap())
                .args(["--exact", "platform::signals::report::tests::attributed_compiler_fault_persists_only_its_note", "--nocapture"])
                .env(CHILD, &root).output().unwrap();
            let note = std::fs::read_to_string(root.join("crashed.txt")).unwrap();
            std::fs::remove_dir_all(root).unwrap();
            assert_eq!(out.status.code(), Some(132), "{}", String::from_utf8_lossy(&out.stderr));
            assert_eq!(note, "c0000000000000123 test-driver compute pipeline\n");
        }
    }
    #[test]
    fn fatal_ud2_reports_mapping_bytes_and_current_compile() {
        const CHILD: &str = "REGALIANT_SIGILL_TEST";
        if std::env::var_os(CHILD).is_none() {
            let out = std::process::Command::new(std::env::current_exe().unwrap())
                .args(["--exact", "platform::signals::report::tests::fatal_ud2_reports_mapping_bytes_and_current_compile", "--nocapture"])
                .env(CHILD, "1").output().unwrap();
            assert_eq!(out.status.code(), Some(132));
            let text = String::from_utf8_lossy(&out.stderr);
            for needle in ["rip bytes: 0f 0b", "rip mapping", "file_offset=", "stack[", "active pipeline tid=", "(faulting thread): c0000000000000123 test-driver test compile"] { assert!(text.contains(needle), "missing {needle}: {text}"); }
        } else {
            super::super::ensure_installed().unwrap();
            let _stack = super::super::SignalStack::new().unwrap();
            let _note = super::CompileNote::new("c0000000000000123 test-driver test compile");
            unsafe { std::arch::asm!("ud2", options(noreturn)); }
        }
    }
}
