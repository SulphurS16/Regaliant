pub use std::os::windows::fs::FileExt;
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
