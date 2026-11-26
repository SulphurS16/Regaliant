//! Durable source-compile intent. A compiler abort, SIGKILL or power loss cannot erase the
//! identity of the call it interrupted. Successful returns alone remove their marker.
use std::{fs::{File, OpenOptions}, io::Write, path::{Path, PathBuf}};

/// Interrupted compiles of one call allowed before it is quarantined.
const MAX_INTERRUPTED: usize = 3;

#[derive(Debug)]
pub struct Pending(PathBuf);
impl Pending {
    pub fn begin(root: &Path, identity: &str, kind: char, key: u64, fingerprint: u64) -> Result<Self, String> {
        // identity is hashed to keep arbitrary driver names out of paths. Full identity is
        // retained in the record. Include SPIR-V/layout fingerprint: fixed code may retry.
        let hash = identity.bytes().fold(0xcbf29ce484222325u64, |h,b| (h ^ b as u64).wrapping_mul(0x100000001b3));
        let dir = root.join("compile-pending");
        std::fs::create_dir_all(&dir).map_err(|e| format!("compile journal directory: {e}"))?;
        let path = dir.join(format!("{hash:016x}-{kind}{key:016x}-{fingerprint:016x}.pending"));
        let me = std::process::id();
        let mut f = match OpenOptions::new().create_new(true).write(true).open(&path) {
            Ok(mut f) => { writeln!(f, "{kind}{key:016x} {fingerprint:016x} {identity}").map_err(|e| format!("compile journal: {e}"))?; f }
            Err(e) if e.kind() == std::io::ErrorKind::AlreadyExists => {
                // A marker outlives its call when the process ends inside the compile: a driver
                // crash, but also a killed test run or a quit mid-compile. Interrupted calls get
                // MAX_INTERRUPTED tries in all; a call the driver failed, or one under way in this
                // process, is not tried again.
                let text = std::fs::read_to_string(&path).unwrap_or_default();
                let tries = text.lines().filter(|l| l.starts_with("try ")).count().max(1);
                let why = if text.lines().any(|l| l == "failed") { Some("its compile failed before".to_string()) }
                    else if text.lines().any(|l| l == format!("try {me}")) { Some("it is being compiled now".to_string()) }
                    else if tries >= MAX_INTERRUPTED { Some(format!("{tries} compiles of it were interrupted")) }
                    else { None };
                if let Some(why) = why {
                    return Err(format!("{kind}{key:016x}: source compile not attempted: {} ({why}); an unfinished prior compile is quarantined", path.display()));
                }
                OpenOptions::new().append(true).open(&path).map_err(|e| format!("compile journal: {e}"))?
            }
            Err(e) => return Err(format!("{kind}{key:016x}: source compile not attempted: {} ({e})", path.display())),
        };
        writeln!(f, "try {me}").and_then(|_| f.sync_all())
            .and_then(|_| File::open(&dir)?.sync_all()).and_then(|_| File::open(root)?.sync_all())
            .map_err(|e| format!("compile journal could not be made durable: {e}"))?;
        Ok(Self(path))
    }
    pub fn finish(self, success: bool) -> Result<(), String> {
        if success {
            std::fs::remove_file(&self.0).and_then(|_| File::open(self.0.parent().unwrap())?.sync_all())
                .map_err(|e| format!("successful compile's journal could not be cleared: {e}"))?;
        } else {
            OpenOptions::new().append(true).open(&self.0).and_then(|mut f| { writeln!(f, "failed")?; f.sync_all() })
                .map_err(|e| format!("failed compile's journal could not be marked: {e}"))?;
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

    #[test]
    fn compiles_interrupted_in_other_processes_retry_twice_then_quarantine() {
        let root = std::env::temp_dir().join(format!("regaliant-compile-journal-retry-{}",std::process::id()));
        std::fs::create_dir(&root).unwrap();
        let first = super::Pending::begin(&root,"driver/source",'g',7,8).unwrap();
        let path = first.0.clone();
        drop(first);
        // As left by a killed run in another process (old markers have no try lines at all).
        std::fs::write(&path, "g0000000000000007 0000000000000008 driver/source\ntry 1\n").unwrap();
        drop(super::Pending::begin(&root,"driver/source",'g',7,8).unwrap()); // second try, this process
        for n in [2, 3] {
            let text = std::fs::read_to_string(&path).unwrap().replace(&format!("try {}", std::process::id()), &format!("try {n}"));
            std::fs::write(&path, text).unwrap();
            if n == 2 { drop(super::Pending::begin(&root,"driver/source",'g',7,8).unwrap()); } // third try
        }
        assert!(super::Pending::begin(&root,"driver/source",'g',7,8).unwrap_err().contains("3 compiles of it were interrupted"));
        std::fs::write(&path, "g0000000000000007 0000000000000008 driver/source\n").unwrap();
        super::Pending::begin(&root,"driver/source",'g',7,8).unwrap().finish(true).unwrap();
        assert!(!path.exists());
        std::fs::remove_dir_all(root).unwrap();
    }
}
