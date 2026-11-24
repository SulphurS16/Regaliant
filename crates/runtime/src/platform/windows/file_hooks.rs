pub fn generated(path: &str) -> Option<std::sync::Arc<Vec<u8>>> { crate::pcsettings::generated(path) }
pub fn setting(name: &str) -> Option<u32> { crate::pcsettings::store::value(name) }
pub fn host_write(base: u64, size: u64) -> crate::gpu::track::HostWrite { crate::gpu::track::host_write(base, size) }
pub fn trace_writes(f: impl Fn() -> String, base: u64, size: u64) { crate::gpu::exec::trace_writes(f, base, size); }
pub fn note_since(t: std::time::Instant, f: impl FnOnce() -> String) { crate::gpu::hitch::note_since(crate::gpu::hitch::Kind::File, t, f); }
pub fn console_line(line: &str) {
        // The title loads a world (its own line at every level load): `maskwait subload`'s window.
        if line.contains("Loading world") {
            crate::cullread::note_level_load();
        }
        // Developer automation (tools/tour.py) asks for a screenshot by
        // printing `dsport-shot: <name>` (taken at the next flip).
        if let Some(rest) = line.split("dsport-shot:").nth(1) {
            let name: String = rest.trim().chars().filter(|c| c.is_ascii_alphanumeric() || *c == '_' || *c == '-').collect();
            if !name.is_empty() {
                *crate::hle::libs::videoout::SHOT_REQUEST.lock() = Some(name);
            }
        }
        // `dsport-frames: reset` starts a frame-time window, `dsport-frames:
        // <name>` logs it (tools/tour.py --frames: each location's frame
        // rate at the developer camera).
        if let Some(rest) = line.split("dsport-frames:").nth(1) {
            let rest = rest.trim();
            let report = crate::gpu::pacing::report_and_reset();
            if rest != "reset" {
                crate::info!("frames at {rest}: {report}");
            }
        }
        // `dsport-textreset` forgets the title's strings logged so far
        // (`text:` lines come once per string: tools/devscript.sh scripts
        // checking several bosses' banners in one run).
        if line.contains("dsport-textreset") {
            crate::exception::text_reset();
        }
        // `dsport-control: <line>` runs a control command (`press cross
        // 200ms`, `audio capture 30`, ...) at the script's point.
        if let Some(rest) = line.split("dsport-control:").nth(1) {
            let rest = rest.trim();
            if !rest.is_empty() && !crate::input::control_now(rest) {
                crate::INJECTED.lock().push(rest.to_string());
            }
        }
}

pub fn save_root(path: std::path::PathBuf) -> std::path::PathBuf { path }

pub fn prepare_kernel_io(_base:u64,_len:u64,_into_guest:bool) {}
pub fn is_memory_fault(e: &std::io::Error) -> bool { matches!(e.raw_os_error(),Some(998|1784)) }
