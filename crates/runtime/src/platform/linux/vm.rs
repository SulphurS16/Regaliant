//! Owned virtual address reservations. Fixed mappings never replace an existing mapping;
//! commits/protection changes operate only inside the reservation returned by mmap.
use std::io;

pub const PAGE: usize = 4096;
pub const NONE: i32 = libc::PROT_NONE;
pub const READ: i32 = libc::PROT_READ;
pub const WRITE: i32 = libc::PROT_WRITE;
pub const EXEC: i32 = libc::PROT_EXEC;

#[derive(Debug)]
pub struct Reservation {
    base: usize,
    size: usize,
}

impl Reservation {
    pub fn new_at(address: usize, size: usize) -> io::Result<Self> {
        if size == 0 || size % PAGE != 0 || address % PAGE != 0 || address.checked_add(size).is_none() {
            return Err(io::Error::new(io::ErrorKind::InvalidInput, "unaligned or overflowing reservation"));
        }
        let fixed = if address == 0 { 0 } else { libc::MAP_FIXED_NOREPLACE };
        let p = unsafe { libc::mmap(address as *mut _, size, NONE, libc::MAP_PRIVATE | libc::MAP_ANONYMOUS | libc::MAP_NORESERVE | fixed, -1, 0) };
        if p == libc::MAP_FAILED { return Err(io::Error::last_os_error()); }
        if address != 0 && p as usize != address {
            // Kernels predating MAP_FIXED_NOREPLACE can ignore the flag. Never accept a hint
            // being relocated, and never fall back to destructive MAP_FIXED.
            unsafe { libc::munmap(p, size) };
            return Err(io::Error::other("kernel did not honor MAP_FIXED_NOREPLACE"));
        }
        Ok(Self { base: p as usize, size })
    }

    pub fn base(&self) -> usize { self.base }
    pub fn size(&self) -> usize { self.size }

    fn range(&self, offset: usize, size: usize) -> io::Result<*mut libc::c_void> {
        if size == 0 || offset % PAGE != 0 || size % PAGE != 0 || offset.checked_add(size).is_none_or(|e| e > self.size) {
            return Err(io::Error::new(io::ErrorKind::InvalidInput, "range outside reservation or not page aligned"));
        }
        Ok((self.base + offset) as *mut _)
    }

    /// Commits anonymous zero-fill-on-demand pages, or changes their access. A guard page is
    /// simply NONE: Linux has no Windows-style one-shot PAGE_GUARD flag.
    pub fn protect(&self, offset: usize, size: usize, prot: i32) -> io::Result<()> {
        let p = self.range(offset, size)?;
        if prot & !(READ | WRITE | EXEC) != 0 {
            return Err(io::Error::new(io::ErrorKind::InvalidInput, "invalid protection"));
        }
        if unsafe { libc::mprotect(p, size, prot) } != 0 { return Err(io::Error::last_os_error()); }
        Ok(())
    }

    /// Drops private pages while retaining ownership of their addresses. A later commit sees
    /// zeros; no other thread may access this range concurrently with decommit.
    pub fn decommit(&self, offset: usize, size: usize) -> io::Result<()> {
        let p = self.range(offset, size)?;
        self.protect(offset, size, NONE)?;
        if unsafe { libc::madvise(p, size, libc::MADV_DONTNEED) } != 0 { return Err(io::Error::last_os_error()); }
        Ok(())
    }
}

impl Drop for Reservation {
    fn drop(&mut self) { unsafe { libc::munmap(self.base as *mut _, self.size) }; }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn fixed_collision_never_replaces_memory() {
        let a = Reservation::new_at(0, PAGE).unwrap();
        a.protect(0, PAGE, READ | WRITE).unwrap();
        unsafe { *(a.base() as *mut u64) = 0xfeedbeef; }
        assert!(Reservation::new_at(a.base(), PAGE).is_err());
        assert_eq!(unsafe { *(a.base() as *const u64) }, 0xfeedbeef);
    }

    #[test]
    fn decommit_recommit_is_zero_and_bounds_are_checked() {
        let a = Reservation::new_at(0, 3 * PAGE).unwrap();
        a.protect(PAGE, PAGE, READ | WRITE).unwrap();
        unsafe { *((a.base() + PAGE) as *mut u64) = 42; }
        a.decommit(PAGE, PAGE).unwrap();
        a.protect(PAGE, PAGE, READ | WRITE).unwrap();
        assert_eq!(unsafe { *((a.base() + PAGE) as *const u64) }, 0);
        assert!(a.protect(3 * PAGE, PAGE, READ).is_err());
        assert!(a.protect(1, PAGE, READ).is_err());
        assert!(a.protect(0, usize::MAX, READ).is_err());
    }
}

impl Reservation {
    /// Replaces only pages whose addresses this object already owns. MAP_FIXED_NOREPLACE is
    /// used to acquire the reservation; MAP_FIXED here replaces its PROT_NONE pages with a
    /// shared memfd view (or fresh anonymous memory). It cannot clobber an unrelated mapping.
    pub fn replace(&self, offset: usize, size: usize, prot: i32, backing: Option<(i32, u64)>) -> io::Result<()> {
        let p = self.range(offset, size)?;
        let (flags, fd, file_offset) = match backing {
            Some((fd, off)) if off % PAGE as u64 == 0 && off <= i64::MAX as u64 => (libc::MAP_SHARED, fd, off as i64),
            Some(_) => return Err(io::Error::new(io::ErrorKind::InvalidInput, "invalid backing offset")),
            None => (libc::MAP_PRIVATE | libc::MAP_ANONYMOUS | libc::MAP_NORESERVE, -1, 0),
        };
        if unsafe { libc::mmap(p, size, prot, flags | libc::MAP_FIXED, fd, file_offset) } == libc::MAP_FAILED { return Err(io::Error::last_os_error()); }
        Ok(())
    }
}
