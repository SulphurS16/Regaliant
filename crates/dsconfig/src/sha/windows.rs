//! SHA-256 through Windows' own provider (BCrypt): the game files' fingerprints.

use std::io::Read;
use std::path::Path;
use windows_sys::Win32::Security::Cryptography::{BCRYPT_HASH_HANDLE, BCRYPT_SHA256_ALG_HANDLE, BCryptCreateHash, BCryptDestroyHash, BCryptFinishHash, BCryptHashData};

/// A SHA-256 computed piece by piece (`zar`'s archives, hashed as they are written).
pub struct Sha256 {
    h: BCRYPT_HASH_HANDLE,
}

// SAFETY: a BCrypt hash object is used by one thread at a time (`&mut self`).
unsafe impl Send for Sha256 {}

impl Sha256 {
    pub fn new() -> std::io::Result<Sha256> {
        let mut h: BCRYPT_HASH_HANDLE = std::ptr::null_mut();
        let st = unsafe { BCryptCreateHash(BCRYPT_SHA256_ALG_HANDLE, &mut h, std::ptr::null_mut(), 0, std::ptr::null(), 0, 0) };
        if st != 0 {
            return Err(std::io::Error::other(format!("BCryptCreateHash {st:#x}")));
        }
        Ok(Sha256 { h })
    }

    pub fn update(&mut self, data: &[u8]) -> std::io::Result<()> {
        for c in data.chunks(1 << 30) {
            let st = unsafe { BCryptHashData(self.h, c.as_ptr(), c.len() as u32, 0) };
            if st != 0 {
                return Err(std::io::Error::other(format!("BCryptHashData {st:#x}")));
            }
        }
        Ok(())
    }

    pub fn finish(self) -> std::io::Result<[u8; 32]> {
        let mut out = [0u8; 32];
        let st = unsafe { BCryptFinishHash(self.h, out.as_mut_ptr(), 32, 0) };
        if st != 0 {
            return Err(std::io::Error::other(format!("BCryptFinishHash {st:#x}")));
        }
        Ok(out)
    }
}

impl Drop for Sha256 {
    fn drop(&mut self) {
        unsafe { BCryptDestroyHash(self.h) };
    }
}

/// The file's SHA-256 as 64 lowercase hex digits.
pub fn file_sha256(path: &Path) -> std::io::Result<String> {
    let mut f = std::fs::File::open(path)?;
    let mut h: BCRYPT_HASH_HANDLE = std::ptr::null_mut();
    let st = unsafe { BCryptCreateHash(BCRYPT_SHA256_ALG_HANDLE, &mut h, std::ptr::null_mut(), 0, std::ptr::null(), 0, 0) };
    if st != 0 {
        return Err(std::io::Error::other(format!("BCryptCreateHash {st:#x}")));
    }
    let mut buf = vec![0u8; 1 << 20];
    let r = (|| {
        loop {
            let n = f.read(&mut buf)?;
            if n == 0 {
                break;
            }
            let st = unsafe { BCryptHashData(h, buf.as_ptr(), n as u32, 0) };
            if st != 0 {
                return Err(std::io::Error::other(format!("BCryptHashData {st:#x}")));
            }
        }
        let mut out = [0u8; 32];
        let st = unsafe { BCryptFinishHash(h, out.as_mut_ptr(), 32, 0) };
        if st != 0 {
            return Err(std::io::Error::other(format!("BCryptFinishHash {st:#x}")));
        }
        Ok(out.iter().map(|b| format!("{b:02x}")).collect())
    })();
    unsafe { BCryptDestroyHash(h) };
    r
}
