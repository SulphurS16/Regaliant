use sha2::{Digest, Sha256 as Hash};
use std::io::Read;
use std::path::Path;

pub fn file_sha256(path: &Path) -> std::io::Result<String> {
    let mut file = std::fs::File::open(path)?;
    let mut hash = Hash::new();
    let mut buf = vec![0u8; 1 << 20];
    loop {
        let n = file.read(&mut buf)?;
        if n == 0 { break; }
        hash.update(&buf[..n]);
    }
    Ok(format!("{:x}", hash.finalize()))
}

/// Incremental archive hashing, matching the Windows provider's interface.
pub struct Sha256(Hash);
impl Sha256 {
    pub fn new() -> std::io::Result<Self> { Ok(Self(Hash::new())) }
    pub fn update(&mut self, bytes: &[u8]) -> std::io::Result<()> { self.0.update(bytes); Ok(()) }
    pub fn finish(self) -> std::io::Result<[u8; 32]> { Ok(self.0.finalize().into()) }
}
