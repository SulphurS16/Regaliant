//! File fingerprints, using a platform-selected SHA-256 implementation.
#[cfg(windows)]
#[path = "sha/windows.rs"]
mod host;
#[cfg(target_os = "linux")]
#[path = "sha/linux.rs"]
mod host;
pub use host::{file_sha256, Sha256};

/// A digest as lowercase hexadecimal.
pub fn hex(digest: &[u8; 32]) -> String {
    digest.iter().map(|b| format!("{b:02x}")).collect()
}

#[cfg(test)]
mod tests {
    #[test]
    fn known_value() {
        let p = std::env::temp_dir().join(format!("dsconfig-sha-{}", std::process::id()));
        std::fs::write(&p, b"abc").unwrap();
        let h = super::file_sha256(&p).unwrap();
        let _ = std::fs::remove_file(&p);
        assert_eq!(h, "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad");
    }
}
