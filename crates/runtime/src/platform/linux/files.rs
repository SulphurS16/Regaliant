//! Read-only game paths on case-sensitive filesystems. Exact spellings win; a folded lookup
//! must be unambiguous. Canonicalization prevents symlinks escaping the mount.
use std::path::{Component, Path, PathBuf};
use std::io;

// Fixed before worker startup; every writable mount and diagnostic uses the same policy.
static GAME_ROOTS: std::sync::OnceLock<Vec<PathBuf>> = std::sync::OnceLock::new();
pub fn protect_game(root: &Path) -> io::Result<()> {
    let root = root.canonicalize()?;
    let mut roots = vec![root.clone()];
    let mut dirs = if root.is_dir() { vec![root] } else { Vec::new() };
    while let Some(dir) = dirs.pop() {
        for e in std::fs::read_dir(dir)? {
            let e = e?;
            let kind = e.file_type()?;
            if kind.is_symlink() {
                // Protect external targets too, without following cycles or walking them.
                if let Ok(target) = e.path().canonicalize() { roots.push(target); }
            } else if kind.is_dir() { dirs.push(e.path()); }
        }
    }
    GAME_ROOTS.set(roots).map_err(|_| io::Error::other("game output policy already initialized"))
}

pub struct ReadOnlyMount { root: PathBuf }
impl ReadOnlyMount {
    pub fn new(root: &Path) -> io::Result<Self> { Ok(Self { root: root.canonicalize()? }) }

    pub fn resolve(&self, relative: &Path) -> io::Result<PathBuf> {
        let mut path = self.root.clone();
        for component in relative.components() {
            let Component::Normal(name) = component else {
                if component == Component::CurDir { continue; }
                return Err(io::Error::new(io::ErrorKind::PermissionDenied, "absolute path or parent traversal on game mount"));
            };
            let exact = path.join(name);
            path = if exact.exists() { exact } else {
                let mut found = None;
                for entry in std::fs::read_dir(&path)? {
                    let entry = entry?;
                    if entry.file_name().as_encoded_bytes().eq_ignore_ascii_case(name.as_encoded_bytes()) {
                        if found.is_some() { return Err(io::Error::other("ambiguous case-insensitive game path")); }
                        found = Some(entry.path());
                    }
                }
                found.ok_or_else(|| io::Error::new(io::ErrorKind::NotFound, relative.display().to_string()))?
            };
            path = path.canonicalize()?;
            if !path.starts_with(&self.root) { return Err(io::Error::new(io::ErrorKind::PermissionDenied, "symlink escapes game mount")); }
        }
        Ok(path)
    }

    #[cfg(test)]
    pub fn open(&self, relative: &Path) -> io::Result<std::fs::File> {
        // No caller can request write/create/truncate flags through this interface.
        std::fs::File::open(self.resolve(relative)?)
    }
}

/// Resolve the existing ancestor before creating an output directory; catch symlink aliases
/// as well as lexical paths into a forbidden tree.
pub fn output_path(path: &Path, forbidden: &[&Path]) -> io::Result<PathBuf> {
    if !path.is_absolute() { return Err(io::Error::new(io::ErrorKind::InvalidInput, "output path must be absolute")); }
    let mut ancestor = path;
    let mut suffix = Vec::new();
    while !ancestor.exists() {
        suffix.push(ancestor.file_name().ok_or_else(|| io::Error::other("invalid output path"))?);
        ancestor = ancestor.parent().ok_or_else(|| io::Error::other("output has no existing ancestor"))?;
    }
    let mut resolved = ancestor.canonicalize()?;
    for part in suffix.into_iter().rev() { resolved.push(part); }
    let extras: Vec<PathBuf> = std::env::var_os("DS_READONLY_ROOTS")
        .map(|v| std::env::split_paths(&v).filter(|p| !p.as_os_str().is_empty()).collect())
        .unwrap_or_default();
    let game = GAME_ROOTS.get().map(Vec::as_slice).unwrap_or(&[]);
    let forbidden: Vec<&Path> = forbidden.iter().copied().chain(game.iter().map(PathBuf::as_path)).chain(extras.iter().map(PathBuf::as_path)).collect();
    // Refuse any parent components rather than normalize a possibly nonexistent path through
    // them: create_dir_all must not traverse into a forbidden tree along the way.
    if path.components().any(|c| c == Component::ParentDir) || forbidden.iter().any(|f| resolved.starts_with(f.canonicalize().unwrap_or_else(|_| std::path::absolute(f).unwrap_or_else(|_| f.to_path_buf())))) {
        return Err(io::Error::new(io::ErrorKind::PermissionDenied, "output directory is in a read-only game/storage tree"));
    }
    Ok(resolved)
}

/// Check existing output descendants once at startup, before cache/settings writers start.
/// Later guest operations also validate their individual destination through `candidates`.
pub fn validate_output_tree(path: &Path) -> io::Result<()> {
    let mut dirs = vec![output_path(path, &[])?];
    let mut seen = std::collections::HashSet::new();
    while let Some(dir) = dirs.pop() {
        if !dir.is_dir() || !seen.insert(dir.clone()) { continue; }
        for e in std::fs::read_dir(dir)? {
            let p = output_path(&e?.path(), &[])?;
            if p.is_dir() { dirs.push(p); }
        }
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn protected_game_targets_and_extra_roots() {
        if std::env::var_os("REGALIANT_OUTPUT_TEST").is_none() {
            let root = std::env::temp_dir().join(format!("regaliant-output-policy-{}", std::process::id()));
            let out = std::process::Command::new(std::env::current_exe().unwrap())
                .args(["--exact", "platform::files::tests::protected_game_targets_and_extra_roots", "--nocapture"])
                .env("REGALIANT_OUTPUT_TEST", &root).env("DS_READONLY_ROOTS", root.join("extra"))
                .output().unwrap();
            assert!(out.status.success(), "{}{}", String::from_utf8_lossy(&out.stdout), String::from_utf8_lossy(&out.stderr));
            return;
        }
        let root = PathBuf::from(std::env::var_os("REGALIANT_OUTPUT_TEST").unwrap());
        for p in ["game", "external", "extra", "output"] { std::fs::create_dir_all(root.join(p)).unwrap(); }
        std::os::unix::fs::symlink(root.join("external"), root.join("game/linked")).unwrap();
        std::os::unix::fs::symlink(root.join("game"), root.join("alias")).unwrap();
        protect_game(&root.join("game")).unwrap();
        for p in ["game/new", "alias/new", "external/new", "extra/new"] {
            assert!(output_path(&root.join(p), &[]).is_err(), "{p}");
        }
        assert!(output_path(&root.join("output/new"), &[]).is_ok());
        std::os::unix::fs::symlink(root.join("game"), root.join("output/link")).unwrap();
        assert!(validate_output_tree(&root.join("output")).is_err());
        std::fs::remove_dir_all(root).unwrap();
    }

    #[test]
    fn case_resolution_and_read_only_boundaries() {
        let root = std::env::temp_dir().join(format!("regaliant-paths-{}", std::process::id()));
        std::fs::create_dir_all(root.join("game/Textures")).unwrap();
        std::fs::write(root.join("game/Textures/Stone.DDS"), b"asset").unwrap();
        let mount = ReadOnlyMount::new(&root.join("game")).unwrap();
        assert!(mount.open(Path::new("textures/stone.dds")).is_ok());
        assert!(mount.resolve(Path::new("../outside")).is_err());
        assert!(mount.resolve(Path::new("/etc/passwd")).is_err());
        std::os::unix::fs::symlink("/etc", root.join("game/escape")).unwrap();
        assert!(mount.open(Path::new("escape/passwd")).is_err());
        std::os::unix::fs::symlink(root.join("game"), root.join("alias")).unwrap();
        assert!(output_path(&root.join("alias/new/output"), &[&root.join("game")]).is_err());
        std::fs::write(root.join("game/Textures/STONE.dds"), b"different").unwrap();
        assert!(mount.open(Path::new("textures/stone.dds")).is_err());
        assert!(mount.open(Path::new("Textures/Stone.DDS")).is_ok());
        std::fs::remove_dir_all(root).unwrap();
    }
}

pub trait FileExt {
    fn seek_read(&self, buf: &mut [u8], offset: u64) -> io::Result<usize>;
    fn seek_write(&self, buf: &[u8], offset: u64) -> io::Result<usize>;
}

pub fn mapped_file_name(address: u64) -> String {
    let maps = std::fs::read_to_string("/proc/self/maps").unwrap_or_default();
    for line in maps.lines() {
        let mut fields = line.split_whitespace();
        let Some((lo, hi)) = fields.next().and_then(|s| s.split_once('-')) else { continue };
        let (Ok(lo), Ok(hi)) = (u64::from_str_radix(lo, 16), u64::from_str_radix(hi, 16)) else { continue };
        if (lo..hi).contains(&address) { return fields.skip(4).collect::<Vec<_>>().join(" "); }
    }
    String::new()
}
impl FileExt for std::fs::File {
    fn seek_read(&self, buf: &mut [u8], offset: u64) -> io::Result<usize> { std::os::unix::fs::FileExt::read_at(self, buf, offset) }
    fn seek_write(&self, buf: &[u8], offset: u64) -> io::Result<usize> { std::os::unix::fs::FileExt::write_at(self, buf, offset) }
}

pub fn candidates(layers: &[PathBuf], relative: &Path, writable: bool) -> Option<Vec<PathBuf>> {
    let mut result = Vec::new();
    for layer in layers {
        let mount = ReadOnlyMount::new(layer).ok()?;
        match mount.resolve(relative) {
            Ok(path) => {
                if writable { output_path(&path, &[]).ok()?; }
                result.push(path);
            }
            Err(e) if e.kind() == io::ErrorKind::NotFound => {
                // A not-yet-created file: resolve and check every existing ancestor, including
                // symlinks, before permitting the caller's create/rename/remove operations.
                let mut parent = relative;
                let mut tail = Vec::new();
                let mut path = loop {
                    if parent.as_os_str().is_empty() { break mount.root.clone(); }
                    match mount.resolve(parent) {
                        Ok(p) => break p,
                        Err(e) if e.kind() == io::ErrorKind::NotFound => {
                            tail.push(parent.file_name()?);
                            parent = parent.parent()?;
                        }
                        Err(_) => return None,
                    }
                };
                for name in tail.into_iter().rev() { path.push(name); }
                if writable { output_path(&path, &[]).ok()?; }
                result.push(path);
            }
            Err(_) => return None,
        }
    }
    (!result.is_empty()).then_some(result)
}

/// Save imports never traverse directory links (including Windows junctions).
pub fn import_file_type(path: &std::path::Path) -> Option<std::fs::FileType> {
    let meta = std::fs::symlink_metadata(path).ok()?;
    (!meta.file_type().is_symlink()).then_some(meta.file_type())
}

/// Check destinations before any import, backup, or import-folder rename.
pub fn save_output(path: &std::path::Path) -> std::io::Result<()> {
    output_path(path, &[]).map(|_| ())
}

/// Before driver startup: play and isolated preparation share the same fallback driver
/// cache. A preparation-only cache discarded that work whenever pipeline binaries refused.
/// Explicit caller paths win, but obey the same game/symlink read-only policy.
pub fn configure_driver_cache(cache: &Path) -> Result<(), String> {
    for (key, name) in [("__GL_SHADER_DISK_CACHE_PATH", "nvidia"), ("MESA_SHADER_CACHE_DIR", "mesa"), ("CUDA_CACHE_PATH", "cuda")] {
        let path = std::env::var_os(key).map(PathBuf::from).unwrap_or_else(|| cache.join("driver-cache").join(name));
        let path = std::path::absolute(path).map_err(|e| e.to_string())?;
        validate_output_tree(&path).map_err(|e| format!("{key}: {e}"))?;
        std::fs::create_dir_all(&path).map_err(|e| format!("{key}: {e}"))?;
        // Startup only, before the graphics driver or its compiler threads are loaded.
        unsafe { std::env::set_var(key, &path); }
        println!("Driver cache {key}={}", path.display());
    }
    // Explicit environment choices win; preparation inherits these same settings.
    for (key, value) in [("__GL_SHADER_DISK_CACHE_SIZE", "8589934592"), ("__GL_SHADER_DISK_CACHE_SKIP_CLEANUP", "1"), ("MESA_SHADER_CACHE_MAX_SIZE", "8G")] {
        if std::env::var_os(key).is_none() { unsafe { std::env::set_var(key, value); } }
        println!("Driver cache {key}={:?}", std::env::var_os(key));
    }
    Ok(())
}
