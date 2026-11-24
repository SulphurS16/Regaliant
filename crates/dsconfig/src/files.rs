//! Host file reads for folder and archive game sources.
use std::path::{Path, PathBuf};
#[cfg(windows)]
pub fn game_path(root: &Path, rel: &str) -> std::io::Result<PathBuf> {
    Ok(rel.split(['/', '\\']).filter(|c| !c.is_empty()).fold(root.to_path_buf(), |p, c| p.join(c)))
}
#[cfg(target_os = "linux")]
pub fn game_path(root: &Path, rel: &str) -> std::io::Result<PathBuf> {
    use std::io::{Error, ErrorKind};
    let base = root.canonicalize()?;
    let mut path = base.clone();
    if rel.starts_with(['/', '\\']) { return Err(Error::new(ErrorKind::PermissionDenied, "absolute game path")); }
    for c in rel.split(['/', '\\']).filter(|c| !c.is_empty() && *c != ".") {
        if c == ".." { return Err(Error::new(ErrorKind::PermissionDenied, "game path traversal")); }
        let exact = path.join(c);
        path = if exact.exists() { exact } else {
            let mut found = None;
            for entry in std::fs::read_dir(&path)? {
                let e = entry?;
                if e.file_name().as_encoded_bytes().eq_ignore_ascii_case(c.as_bytes()) {
                    if found.is_some() { return Err(Error::other("ambiguous case-insensitive game path")); }
                    found = Some(e.path());
                }
            }
            found.ok_or_else(|| Error::new(ErrorKind::NotFound, rel.to_string()))?
        };
        path = path.canonicalize()?;
        if !path.starts_with(&base) { return Err(Error::new(ErrorKind::PermissionDenied, "symlink escapes game root")); }
    }
    Ok(path)
}

#[cfg(target_os = "linux")]
pub trait FileExt {
    fn seek_read(&self, buf: &mut [u8], offset: u64) -> std::io::Result<usize>;
}
#[cfg(target_os = "linux")]
impl FileExt for std::fs::File {
    fn seek_read(&self, buf: &mut [u8], offset: u64) -> std::io::Result<usize> {
        std::os::unix::fs::FileExt::read_at(self, buf, offset)
    }
}

/// Linux archive creation follows the runtime's read-only game-source rule.
#[cfg(target_os = "linux")]
pub fn archive_output(src: &Path, out: &Path) -> std::io::Result<()> {
    use std::io::{Error, ErrorKind};
    let root = src.canonicalize()?;
    let mut forbidden = vec![root.clone()];
    let mut dirs = vec![root];
    while let Some(dir) = dirs.pop() {
        for e in std::fs::read_dir(dir)? {
            let e = e?;
            let ty = e.file_type()?;
            if ty.is_symlink() {
                // The archive writer follows directory entries; reject links to avoid cycles
                // or silently including data that Source's folder reader refuses to follow.
                return Err(Error::new(ErrorKind::PermissionDenied, "archive source contains a symlink"));
            }
            if ty.is_dir() { dirs.push(e.path()); }
        }
    }
    if let Some(v) = std::env::var_os("DS_READONLY_ROOTS") {
        forbidden.extend(std::env::split_paths(&v).filter(|p| !p.as_os_str().is_empty())
            .map(|p| p.canonicalize().unwrap_or(p)));
    }
    let resolve = |path: &Path| -> std::io::Result<PathBuf> {
        if path.components().any(|c| c == std::path::Component::ParentDir) {
            return Err(Error::new(ErrorKind::PermissionDenied, "parent traversal in archive output"));
        }
        let absolute = std::path::absolute(path)?;
        let mut ancestor = absolute.as_path();
        let mut tail = Vec::new();
        while !ancestor.exists() {
            tail.push(ancestor.file_name().ok_or_else(|| Error::other("invalid archive output"))?);
            ancestor = ancestor.parent().ok_or_else(|| Error::other("invalid archive output"))?;
        }
        let mut resolved = ancestor.canonicalize()?;
        for name in tail.into_iter().rev() { resolved.push(name); }
        Ok(resolved)
    };
    for p in [out.to_path_buf(), out.with_extension("zar.partial")] {
        let real = resolve(&p)?;
        if forbidden.iter().any(|f| real.starts_with(f)) {
            return Err(Error::new(ErrorKind::PermissionDenied, "archive output is in a read-only tree"));
        }
    }
    Ok(())
}
