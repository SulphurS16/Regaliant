//! Durable source-compile intent. A compiler abort, SIGKILL or power loss cannot erase the
//! identity of the call it interrupted. Successful returns alone remove their marker.
use std::{fs::{File, OpenOptions}, io::Write, path::{Path, PathBuf}};

pub struct Pending(PathBuf);
impl Pending {
    pub fn begin(root: &Path, identity: &str, kind: char, key: u64, fingerprint: u64) -> Result<Self, String> {
        // identity is hashed to keep arbitrary driver names out of paths. Full identity is
        // retained in the record. Include SPIR-V/layout fingerprint: fixed code may retry.
        let hash = identity.bytes().fold(0xcbf29ce484222325u64, |h,b| (h ^ b as u64).wrapping_mul(0x100000001b3));
        let dir = root.join("compile-pending");
        std::fs::create_dir_all(&dir).map_err(|e| format!("compile journal directory: {e}"))?;
        let path = dir.join(format!("{hash:016x}-{kind}{key:016x}-{fingerprint:016x}.pending"));
        let mut f = OpenOptions::new().create_new(true).write(true).open(&path)
            .map_err(|e| format!("{kind}{key:016x}: source compile not attempted: {} ({e}); an unfinished prior compile is quarantined", path.display()))?;
        writeln!(f, "{kind}{key:016x} {fingerprint:016x} {identity}").and_then(|_| f.sync_all())
            .and_then(|_| File::open(&dir)?.sync_all()).and_then(|_| File::open(root)?.sync_all())
            .map_err(|e| format!("compile journal could not be made durable: {e}"))?;
        Ok(Self(path))
    }
    pub fn finish(self, success: bool) -> Result<(), String> {
        if success {
            std::fs::remove_file(&self.0).and_then(|_| File::open(self.0.parent().unwrap())?.sync_all())
                .map_err(|e| format!("successful compile's journal could not be cleared: {e}"))?;
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    #[test]
    fn failed_or_interrupted_call_is_skipped_but_new_code_and_driver_can_retry() {
        let root = std::env::temp_dir().join(format!("regaliant-compile-journal-{}",std::process::id()));
        std::fs::create_dir(&root).unwrap();
        let first = super::Pending::begin(&root,"driver/source",'c',123,456).unwrap();
        drop(first); // No unwinding cleanup; a terminated compiler leaves exactly this state.
        assert!(super::Pending::begin(&root,"driver/source",'c',123,456).is_err());
        super::Pending::begin(&root,"driver/source",'c',123,457).unwrap().finish(true).unwrap();
        super::Pending::begin(&root,"driver/source",'c',123,457).unwrap().finish(false).unwrap();
        assert!(super::Pending::begin(&root,"driver/source",'c',123,457).is_err());
        super::Pending::begin(&root,"new-driver/source",'c',123,456).unwrap().finish(true).unwrap();
        std::fs::remove_dir_all(root).unwrap();
    }
}
