pub fn generated(path: &str) -> Option<std::sync::Arc<Vec<u8>>> {
    crate::pcsettings::generated(path)
}
pub fn setting(name: &str) -> Option<u32> {
    crate::pcsettings::store::value(name)
}
pub fn host_write(base: u64, size: u64) -> crate::gpu::track::HostWrite {
    crate::gpu::track::host_write(base, size)
}
pub fn trace_writes(f: impl Fn() -> String, base: u64, size: u64) {
    crate::gpu::exec::trace_writes(f, base, size);
}
pub fn note_since(t: std::time::Instant, f: impl FnOnce() -> String) {
    crate::gpu::hitch::note_since(crate::gpu::hitch::Kind::File, t, f);
}
pub fn console_line(line: &str) {
    if line.contains("Loading world") {
        crate::cullread::note_level_load();
    }
    if let Some(rest) = line.split("dsport-shot:").nth(1) {
        let name: String = rest
            .trim()
            .chars()
            .filter(|c| c.is_ascii_alphanumeric() || *c == '_' || *c == '-')
            .collect();
        if !name.is_empty() {
            *crate::hle::libs::videoout::SHOT_REQUEST.lock() = Some(name);
        }
    }
    if let Some(rest) = line.split("dsport-frames:").nth(1) {
        let report = crate::gpu::pacing::report_and_reset();
        if rest.trim() != "reset" { crate::info!("frames at {}: {report}", rest.trim()); }
    }
    if line.contains("dsport-textreset") { crate::text_hooks::text_reset(); }
    if let Some(rest) = line.split("dsport-control:").nth(1) {
        let command = rest.trim();
        if !command.is_empty() && !crate::input::control_now(command) {
            crate::warn!("unsupported Linux automation control: {command}");
        }
    }
}

pub fn save_root(path: std::path::PathBuf) -> std::path::PathBuf {
    crate::platform::files::output_path(&path, &[])
        .unwrap_or_else(|e| {
            crate::error!("unsafe save output: {e}");
            crate::fatal_exit(0xD542, "save path rejected")
        })
}

pub fn prepare_kernel_io(base: u64, len: u64, into_guest: bool) {
    if into_guest {
        crate::gpu::resident::disarm_for_host(base, len);
        super::os::prepare_kernel_write(base, len);
    }
}
pub fn is_memory_fault(e: &std::io::Error) -> bool {
    e.raw_os_error() == Some(libc::EFAULT)
}

/// Keep a section split/decommit from replacing the kernel's destination mid-copy.
/// Release before kernel_io logs/retries: those paths may consult MEM. The operation
/// must not acquire MEM (the mapping writer holds it while waiting for GUEST_READERS).
pub fn with_kernel_io_pages<T>(base: u64, len: u64, into_guest: bool, op: impl FnOnce() -> T) -> T {
    let _pages = crate::mem::GUEST_READERS.read_recursive();
    prepare_kernel_io(base, len, into_guest);
    op()
}

struct ReadNote { base: u64, end: u64, text: String }
static READS: parking_lot::Mutex<std::collections::VecDeque<ReadNote>> = parking_lot::Mutex::new(std::collections::VecDeque::new());
fn read_history_enabled() -> bool {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ON.get_or_init(|| std::env::var("DS_LINUX_IO_HISTORY").ok().as_deref() == Some("1"))
}
pub fn note_kernel_read(base: u64, len: u64, offset: u64, result: &std::io::Result<usize>, what: &dyn Fn() -> String) {
    if !read_history_enabled() { return; }
    let text = format!("{} at {:.0} ms, flip {}: destination {base:#x}+{len:#x}, offset {offset:#x}, result {result:?}",
        what(), crate::log::elapsed_ms(), crate::gpu::FRAMES.load(std::sync::atomic::Ordering::Relaxed));
    let mut reads = READS.lock();
    if reads.len() >= 8192 { reads.pop_front(); }
    reads.push_back(ReadNote { base, end: base.saturating_add(len), text });
}
pub fn read_history(address: u64) -> Vec<String> {
    if !read_history_enabled() { return vec!["disabled (DS_LINUX_IO_HISTORY=1 enables)".into()]; }
    let Some(reads) = READS.try_lock() else { return vec!["file-read history busy".into()]; };
    reads.iter().rev().filter(|r| r.base <= address && address < r.end).take(8).map(|r| r.text.clone()).collect()
}

#[cfg(test)]
mod tests {
    #[test]
    fn kernel_copy_excludes_mapping_replacement() {
        super::with_kernel_io_pages(0, 0, false, || {
            std::thread::scope(|scope| {
                scope.spawn(|| assert!(crate::mem::GUEST_READERS.try_write().is_none())).join().unwrap();
            });
        });
    }
}
