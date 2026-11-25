//! The host heap: mimalloc, or the Windows process heap with
//! `DS_ALLOC=system` (decided once, at the process's first allocation).
//!
//! The command processors allocate and free many small blocks per draw
//! (prepared stages, binding lists): with the Windows heap ~19% of the
//! graphics command processor's busy samples and ~23% of the parallel
//! workers' were in RtlAllocateHeap/RtlFreeHeap (profile run
//! 20261011-211613).
//!
//! mimalloc places its segments at hint addresses between 2 and 6 TiB on
//! 64-bit Windows, inside the guest address range (`mem::GUEST_BASE` ..
//! `GUEST_END`), which must be reserved in one piece. `main` reserves it
//! before it allocates anything (nothing is allocated before `main`); should
//! an allocation ever come first, it reserves the range itself.

use std::alloc::{GlobalAlloc, Layout, System};
use std::sync::atomic::{AtomicU8, Ordering};

pub struct HostAlloc;

const UNDECIDED: u8 = 0;
const MIMALLOC: u8 = 1;
const SYSTEM: u8 = 2;
const DECIDING: u8 = 3;

static MODE: AtomicU8 = AtomicU8::new(UNDECIDED);

#[inline]
fn mode() -> u8 {
    let m = MODE.load(Ordering::Acquire);
    if m == MIMALLOC || m == SYSTEM {
        return m;
    }
    decide()
}

#[cold]
fn decide() -> u8 {
    if MODE.compare_exchange(UNDECIDED, DECIDING, Ordering::AcqRel, Ordering::Acquire).is_ok() {
        crate::mem::reserve_guest_range_early();
        // (Without the guest range the start ends at status 10 (`main`): Windows' heap then, so
        // mimalloc's segments, hinted into 2-6 TiB, do not appear in the report as the holder.
        // Should main's later attempt succeed, the session runs on it: `name` says so in the
        // log's first line.)
        let for_range = !crate::mem::guest_range_reserved();
        if for_range {
            SYSTEM_FOR_RANGE.store(true, Ordering::Relaxed);
        }
        let m = if env_is_system() || for_range { SYSTEM } else { MIMALLOC };
        MODE.store(m, Ordering::Release);
        return m;
    }
    loop {
        let m = MODE.load(Ordering::Acquire);
        if m == MIMALLOC || m == SYSTEM {
            return m;
        }
        std::hint::spin_loop();
    }
}

/// `DS_ALLOC=system` (read without allocating).
fn env_is_system() -> bool {
    crate::mem::env_is("DS_ALLOC", "system")
}

/// Windows' heap was chosen because the guest range was not reserved at the first allocation.
static SYSTEM_FOR_RANGE: std::sync::atomic::AtomicBool = std::sync::atomic::AtomicBool::new(false);

/// Which heap is in use (status line).
pub fn name() -> &'static str {
    match MODE.load(Ordering::Acquire) {
        SYSTEM if SYSTEM_FOR_RANGE.load(Ordering::Relaxed) => "Windows heap (the guest range was held at the first allocation, reserved later: slower)",
        SYSTEM => "Windows heap (DS_ALLOC=system)",
        _ => "mimalloc",
    }
}

unsafe impl GlobalAlloc for HostAlloc {
    #[inline]
    unsafe fn alloc(&self, l: Layout) -> *mut u8 {
        unsafe {
            if mode() == SYSTEM {
                System.alloc(l)
            } else {
                mimalloc::MiMalloc.alloc(l)
            }
        }
    }

    #[inline]
    unsafe fn alloc_zeroed(&self, l: Layout) -> *mut u8 {
        unsafe {
            if mode() == SYSTEM {
                System.alloc_zeroed(l)
            } else {
                mimalloc::MiMalloc.alloc_zeroed(l)
            }
        }
    }

    #[inline]
    unsafe fn dealloc(&self, p: *mut u8, l: Layout) {
        unsafe {
            if mode() == SYSTEM {
                System.dealloc(p, l)
            } else {
                mimalloc::MiMalloc.dealloc(p, l)
            }
        }
    }

    #[inline]
    unsafe fn realloc(&self, p: *mut u8, l: Layout, n: usize) -> *mut u8 {
        unsafe {
            if mode() == SYSTEM {
                System.realloc(p, l, n)
            } else {
                mimalloc::MiMalloc.realloc(p, l, n)
            }
        }
    }
}
