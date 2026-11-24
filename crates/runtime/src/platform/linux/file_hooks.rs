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
