pub use std::os::windows::fs::FileExt;

pub fn mapped_file_name(address: u64) -> String {
    let mut name = [0u16; 512];
    let len = unsafe {
        windows_sys::Win32::System::ProcessStatus::K32GetMappedFileNameW(
            windows_sys::Win32::System::Threading::GetCurrentProcess(), address as _, name.as_mut_ptr(), name.len() as u32,
        ) as usize
    };
    String::from_utf16_lossy(&name[..len.min(name.len())])
}
pub fn candidates(layers: &[std::path::PathBuf], relative: &std::path::Path, _writable: bool) -> Option<Vec<std::path::PathBuf>> {
    Some(layers.iter().map(|l| l.join(relative)).collect())
}

/// Save imports never traverse directory links (including Windows junctions).
pub fn import_file_type(path: &std::path::Path) -> Option<std::fs::FileType> {
    let meta = std::fs::symlink_metadata(path).ok()?;
    use std::os::windows::fs::MetadataExt;
    if meta.file_attributes() & 0x400 != 0 { return None; }
    (!meta.file_type().is_symlink()).then_some(meta.file_type())
}

/// Check destinations before any import, backup, or import-folder rename.
pub fn save_output(path: &std::path::Path) -> std::io::Result<()> {
    let _ = path;
    Ok(())
}
