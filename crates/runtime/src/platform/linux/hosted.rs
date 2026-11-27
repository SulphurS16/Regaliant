//! Page arming on Linux, in-process (on by default; `DS_LINUX_ARMING=0` turns it off).
//!
//! Windows arms guest pages through a debugger-host process: a fault handled inside the process
//! overwrites the 128 bytes below the faulting thread's stack pointer, where the title's leaf
//! functions keep values (`hosted.rs`). Linux has no such hazard: the kernel skips the red zone
//! when it builds a signal frame, and dsport's handlers run on alternate stacks. So the same
//! interface is served here by the process itself:
//!
//! - `arm` sets the pages' bits, then makes them read-only (`arm_noaccess`: no-access). A
//!   write to one raises SIGSEGV; the handler (`linux_fault`, async-signal-safe: atomics and
//!   one mprotect) makes the page read-write, clears its bit and posts it in a ring, and the
//!   access runs again. Front end 2 drains the ring (`drain_faults`) and treats those pages
//!   as written.
//! - `disarm` makes armed pages read-write first and clears their bits after, so a read-only
//!   page never lacks its bit (its write would reach the other handlers). `forget` (after a
//!   mapping change, which set the protection itself) only clears bits.
//! - Remaps exclude the handler through `resident`'s signal-reader gate (`signal_admit`); a
//!   fault refused there goes to the fault service thread (`service_fault`), which waits.
//! - A page armed and not faulted since needs no take for front end 2's checks
//!   (`fe2::word_ok`): what makes arming worth it. Async uffd write protection stays the
//!   tracker underneath: an armed page's write, once allowed, also marks it written there.
//!
//! Held pages (`hold`, front end 2's C4) are not served: they need the host to suspend the
//! faulting thread, and C4 stays off on Linux.
use parking_lot::{ReentrantMutex, ReentrantMutexGuard};
use std::sync::OnceLock;
use std::sync::atomic::{AtomicPtr, AtomicU64, Ordering};

pub const HOLD_RUNS: usize = 1024;
const PAGE: u64 = 4096;
/// Pages per leaf of the bitmap (128 MiB of guest space), as on Windows (`leaf_end`).
const LEAF_PAGES: u64 = 1 << 15;
const LEAF_WORDS: usize = (LEAF_PAGES / 64) as usize;
const L1: usize = ((crate::mem::GUEST_END - crate::mem::GUEST_BASE) / PAGE / LEAF_PAGES) as usize;
/// Fault ring entries.
const RING: usize = 1 << 16;

#[derive(Clone, Copy, Debug)]
pub struct HeldRequest {
    pub page: u64,
    pub write: bool,
    pub suspended: bool,
    pub rip: u64,
    pub tid: u32,
    pub stamp: i64,
}

struct Leaf {
    words: [AtomicU64; LEAF_WORDS],
}

static L1_TABLE: [AtomicPtr<Leaf>; L1] = [const { AtomicPtr::new(std::ptr::null_mut()) }; L1];
/// Bits cleared so far per leaf (`leaf_events`), and in all.
static LEAF_EVENTS: [AtomicU64; L1] = [const { AtomicU64::new(0) }; L1];
/// The fault ring: entry = page | 1 (written) | 2 (posted), 0 while free; the faulting rip.
static RING_PAGES: [AtomicU64; RING] = [const { AtomicU64::new(0) }; RING];
static RING_RIPS: [AtomicU64; RING] = [const { AtomicU64::new(0) }; RING];
static HEAD: AtomicU64 = AtomicU64::new(0);
static TAIL: AtomicU64 = AtomicU64::new(0);
/// [faults answered in the handler, by the service, refused (remap: service), arms, armed pages,
/// disarms, mprotect failures, ring overflows].
static STATS: [AtomicU64; 8] = [const { AtomicU64::new(0) }; 8];
const F_SIGNAL: usize = 0;
const F_SERVICE: usize = 1;
const F_REFUSED: usize = 2;
const S_ARMS: usize = 3;
const S_ARMED_PAGES: usize = 4;
const S_DISARMS: usize = 5;
const S_FAILED: usize = 6;
const S_OVERFLOW: usize = 7;

static ARM: ReentrantMutex<()> = ReentrantMutex::new(());

fn wanted() -> bool {
    static ON: OnceLock<bool> = OnceLock::new();
    *ON.get_or_init(|| std::env::var("DS_LINUX_ARMING").ok().as_deref() != Some("0"))
}

/// Whether pages are armed (front end 2 then arms, and its checks pass armed pages).
pub fn on() -> bool {
    wanted() && crate::platform::write_watch::enabled()
}

pub fn init() {
    if wanted() {
        crate::info!("Linux page arming: {} (in-process write faults)", if on() { "on" } else { "off: needs the async write watch" });
    }
}

pub fn wants_host() -> bool {
    false
}

#[inline]
fn in_range(a: u64) -> bool {
    (crate::mem::GUEST_BASE..crate::mem::GUEST_END).contains(&a)
}

#[inline]
fn leaf_of(a: u64) -> usize {
    ((a - crate::mem::GUEST_BASE) / PAGE / LEAF_PAGES) as usize
}

/// The word and mask of page `a`'s bit; with `create`, its leaf made (never in signal context).
#[inline]
fn bit(a: u64, create: bool) -> Option<(&'static AtomicU64, u64)> {
    if !in_range(a) {
        return None;
    }
    let slot = &L1_TABLE[leaf_of(a)];
    let mut leaf = slot.load(Ordering::Acquire);
    if leaf.is_null() {
        if !create {
            return None;
        }
        let fresh = Box::into_raw(Box::new(Leaf { words: [const { AtomicU64::new(0) }; LEAF_WORDS] }));
        leaf = match slot.compare_exchange(std::ptr::null_mut(), fresh, Ordering::AcqRel, Ordering::Acquire) {
            Ok(_) => fresh,
            Err(other) => {
                drop(unsafe { Box::from_raw(fresh) });
                other
            }
        };
    }
    let page = (a - crate::mem::GUEST_BASE) / PAGE % LEAF_PAGES;
    let leaf: &'static Leaf = unsafe { &*leaf };
    Some((&leaf.words[(page / 64) as usize], 1u64 << (page % 64)))
}

#[inline]
fn is_armed(a: u64) -> bool {
    bit(a, false).is_some_and(|(w, m)| w.load(Ordering::Acquire) & m != 0)
}

fn mprotect(a: u64, n: u64, prot: i32) -> bool {
    unsafe { libc::syscall(libc::SYS_mprotect, a, n, prot) == 0 }
}

/// Clears the bits of [lo, hi) (counting a leaf event per leaf touched); the pages' protection
/// is the caller's.
fn clear_bits(lo: u64, hi: u64) {
    let mut last = usize::MAX;
    let mut a = lo;
    while a < hi {
        let Some((w, m)) = bit(a, false) else {
            a = leaf_end(a).min(hi);
            continue;
        };
        if w.fetch_and(!m, Ordering::AcqRel) & m != 0 {
            STATS[S_ARMED_PAGES].fetch_sub(1, Ordering::Relaxed);
            let leaf = leaf_of(a);
            if leaf != last {
                LEAF_EVENTS[leaf].fetch_add(1, Ordering::AcqRel);
                last = leaf;
            }
        }
        a += PAGE;
    }
}

pub fn page_held(_a: u64) -> bool {
    false
}
pub fn held(_a: u64) -> bool {
    false
}
pub fn held_pages() -> u64 {
    0
}
pub fn splitting(_lo: u64, _hi: u64) {}
pub fn set_held_answered() {}
pub fn set_held_served(_on: bool) {}
pub fn set_word_mode() {}

/// Holds off every other arm and disarm (re-entrant).
pub fn arm_lock() -> ReentrantMutexGuard<'static, ()> {
    ARM.lock()
}
pub fn try_arm_lock_for(d: std::time::Duration) -> Option<ReentrantMutexGuard<'static, ()>> {
    ARM.try_lock_for(d)
}

/// Arms guest [base, base+len) (whole pages) against writes: bits set, then read-only.
pub fn arm(base: u64, len: u64) -> Result<(), String> {
    arm_as(base, len, libc::PROT_READ)
}

/// Arms guest [base, base+len) against any access: bits set, then no-access.
pub fn arm_noaccess(base: u64, len: u64) -> Result<(), String> {
    arm_as(base, len, libc::PROT_NONE)
}

fn arm_as(base: u64, len: u64, prot: i32) -> Result<(), String> {
    if !on() {
        return Err("Linux page arming is off (DS_LINUX_ARMING=0)".into());
    }
    let _lock = arm_lock();
    let (lo, hi) = (base & !(PAGE - 1), base.saturating_add(len).next_multiple_of(PAGE));
    if lo < crate::mem::GUEST_BASE || hi > crate::mem::GUEST_END || hi <= lo {
        return Err(format!("{lo:#x}..{hi:#x}: outside the guest range"));
    }
    let mut newly = Vec::new();
    for a in (lo..hi).step_by(PAGE as usize) {
        let (w, m) = bit(a, true).expect("inside the guest range");
        if w.fetch_or(m, Ordering::AcqRel) & m == 0 {
            newly.push(a);
        }
    }
    STATS[S_ARMED_PAGES].fetch_add(newly.len() as u64, Ordering::Relaxed);
    if !mprotect(lo, hi - lo, prot) {
        let err = std::io::Error::last_os_error();
        STATS[S_FAILED].fetch_add(1, Ordering::Relaxed);
        // (This call's pages go back: mprotect stops at the first hole, having changed the
        // mappings before it, so they are made read-write again, then their bits cleared (a set
        // bit on a writable page would pass its writes unseen). Pages armed before keep theirs.)
        for a in newly {
            if mprotect(a, PAGE, libc::PROT_READ | libc::PROT_WRITE) || std::io::Error::last_os_error().raw_os_error() == Some(libc::ENOMEM) {
                clear_bits(a, a + PAGE);
            }
        }
        return Err(format!("mprotect {lo:#x}+{:#x}: {err}", hi - lo));
    }
    STATS[S_ARMS].fetch_add(1, Ordering::Relaxed);
    Ok(())
}

pub fn hold(_base: u64, _len: u64) -> Result<(), String> {
    Err("held pages need the Windows host (C4 stays off on Linux)".into())
}
pub fn hold_by_host(runs: &[(u64, u64)]) -> Vec<bool> {
    vec![false; runs.len()]
}

/// The end of the 128 MiB of guest space holding `a` (`leaf_events`' unit).
#[inline]
pub fn leaf_end(a: u64) -> u64 {
    let leaf = PAGE * LEAF_PAGES;
    crate::mem::GUEST_BASE + ((a.max(crate::mem::GUEST_BASE) - crate::mem::GUEST_BASE) / leaf + 1) * leaf
}

/// Armed bits cleared so far in the 128 MiB holding `a`: unchanged, no armed page there lost
/// its bit.
#[inline]
pub fn leaf_events(a: u64) -> u64 {
    if !on() || !in_range(a) {
        return 0;
    }
    LEAF_EVENTS[leaf_of(a)].load(Ordering::Acquire)
}

/// Disarms guest [base, base+len): armed pages read-write first, then their bits cleared, run
/// by run (pages not armed keep their protection).
pub fn disarm(base: u64, len: u64) {
    if !on() {
        return;
    }
    let _lock = arm_lock();
    let lo = (base & !(PAGE - 1)).max(crate::mem::GUEST_BASE);
    let hi = base.saturating_add(len).next_multiple_of(PAGE).min(crate::mem::GUEST_END);
    let mut a = lo;
    while a < hi {
        let Some((w, _)) = bit(a, false) else {
            a = leaf_end(a);
            continue;
        };
        let i = (a - crate::mem::GUEST_BASE) / PAGE % 64;
        let rest = w.load(Ordering::Acquire) >> i;
        if rest == 0 {
            a += (64 - i) * PAGE;
            continue;
        }
        a += rest.trailing_zeros() as u64 * PAGE;
        if a >= hi {
            break;
        }
        let mut e = a + PAGE;
        while e < hi && is_armed(e) {
            e += PAGE;
        }
        if mprotect(a, e - a, libc::PROT_READ | libc::PROT_WRITE) {
            clear_bits(a, e);
            STATS[S_DISARMS].fetch_add(1, Ordering::Relaxed);
        } else if STATS[S_FAILED].fetch_add(1, Ordering::Relaxed) < 8 {
            crate::warn!("Linux arming: disarming {a:#x}+{:#x} failed ({}): its pages stay armed", e - a, std::io::Error::last_os_error());
        }
        a = e;
    }
}

/// The armed bits of the 64 pages from `a` (a multiple of 64 pages from the guest range's
/// start): bit i for page a + 4096 * i.
#[inline]
pub fn armed_word(a: u64) -> u64 {
    if !on() {
        return 0;
    }
    bit(a, false).map_or(0, |(w, _)| w.load(Ordering::Acquire))
}

/// Whether every page of guest [base, base+len) is armed.
pub fn all_armed(base: u64, len: u64) -> bool {
    if !on() {
        return false;
    }
    let (lo, hi) = (base & !(PAGE - 1), (base + len.max(1)).next_multiple_of(PAGE));
    (lo..hi).step_by(PAGE as usize).all(is_armed)
}

/// Disarms guest [base, base+len) if a page of it is armed (kernel copies and imports need
/// writable pages).
pub fn disarm_if_armed(base: u64, len: u64) {
    if !on() || len == 0 {
        return;
    }
    let (lo, hi) = (base & !(PAGE - 1), base.saturating_add(len).next_multiple_of(PAGE).min(crate::mem::GUEST_END));
    let mut a = lo.max(crate::mem::GUEST_BASE);
    while a < hi {
        let end = leaf_end(a).min(hi);
        if bit(a, false).is_some() && (a..end).step_by(PAGE as usize).any(is_armed) {
            disarm(a, end - a);
        }
        a = end;
    }
}

pub fn armed(a: u64) -> bool {
    on() && is_armed(a)
}

/// Guest [base, base+len) changed its mapping or protection: no longer armed (bits cleared,
/// protection left as the change set it).
pub fn forget(base: u64, len: u64) {
    if !on() {
        return;
    }
    let lo = (base & !(PAGE - 1)).max(crate::mem::GUEST_BASE);
    let hi = base.saturating_add(len).next_multiple_of(PAGE).min(crate::mem::GUEST_END);
    if lo < hi {
        clear_bits(lo, hi);
    }
}

/// Posts page `page`'s fault (async-signal-safe).
fn post(page: u64, write: bool, rip: u64) {
    let h = HEAD.fetch_add(1, Ordering::AcqRel);
    let i = (h % RING as u64) as usize;
    RING_RIPS[i].store(rip, Ordering::Relaxed);
    if RING_PAGES[i].swap(page | write as u64 | 2, Ordering::AcqRel) != 0 {
        // (An entry the reader had not taken yet: overwritten, so faults were lost.)
        STATS[S_OVERFLOW].fetch_add(1, Ordering::Relaxed);
    }
}

/// Ends page `addr`'s arming for an access fault: read-write, then its bit cleared and the
/// fault posted. Async-signal-safe (atomics and one mprotect). False: not an armed page.
fn answer(addr: u64, write: bool, rip: u64) -> bool {
    let page = addr & !(PAGE - 1);
    let Some((w, m)) = bit(page, false) else { return false };
    if w.load(Ordering::Acquire) & m == 0 {
        return false;
    }
    if !mprotect(page, PAGE, libc::PROT_READ | libc::PROT_WRITE) {
        return false;
    }
    // (Two threads faulting on one page both make it writable; the one clearing the bit posts.)
    if w.fetch_and(!m, Ordering::AcqRel) & m != 0 {
        STATS[S_ARMED_PAGES].fetch_sub(1, Ordering::Relaxed);
        LEAF_EVENTS[leaf_of(page)].fetch_add(1, Ordering::AcqRel);
        post(page, write, rip);
    }
    true
}

/// The SIGSEGV handler's arming path (async-signal-safe): a fault on an armed page, answered
/// here unless a remap holds the signal readers out (then the fault service, `service_fault`).
pub fn linux_fault(addr: u64, write: bool, rip: u64) -> bool {
    if !on() || !in_range(addr) || !is_armed(addr & !(PAGE - 1)) {
        return false;
    }
    if !crate::gpu::resident::signal_admit() {
        STATS[F_REFUSED].fetch_add(1, Ordering::Relaxed);
        return false;
    }
    let ok = answer(addr, write, rip);
    crate::gpu::resident::signal_leave();
    if ok {
        STATS[F_SIGNAL].fetch_add(1, Ordering::Relaxed);
    }
    ok
}

/// The fault service thread's arming path (outside signal context; may wait for a remap).
pub fn service_fault(addr: u64, write: bool, rip: u64) -> bool {
    if !on() || !in_range(addr) {
        return false;
    }
    let _mapping = crate::gpu::resident::arm_guard();
    let ok = answer(addr, write, rip);
    if ok {
        STATS[F_SERVICE].fetch_add(1, Ordering::Relaxed);
    }
    ok
}

pub fn take_held(_out: &mut Vec<HeldRequest>) {}
pub fn wait_held(_ms: u32) -> bool {
    false
}
pub fn release(_base: u64, _len: u64) -> Result<u64, String> {
    Err("held pages need the Windows host".into())
}
pub fn released_bits(_base: u64, _len: u64) {}
pub fn resume_waiter(_r: &HeldRequest) -> Result<u64, String> {
    Err("held pages need the Windows host".into())
}

/// The faults answered since the last call: (page, a write, the faulting instruction); false
/// when faults were lost (treat all armed pages as touched).
pub fn drain_faults(out: &mut Vec<(u64, bool, u64)>) -> bool {
    if !on() {
        return true;
    }
    let head = HEAD.load(Ordering::Acquire);
    let mut tail = TAIL.load(Ordering::Acquire);
    let mut lost = head.saturating_sub(tail) > RING as u64;
    if lost {
        tail = head - RING as u64;
    }
    while tail < head {
        let i = (tail % RING as u64) as usize;
        let e = RING_PAGES[i].load(Ordering::Acquire);
        if e == 0 {
            // (Claimed by a fault not yet posted: taken next time.)
            break;
        }
        let rip = RING_RIPS[i].load(Ordering::Relaxed);
        RING_PAGES[i].store(0, Ordering::Release);
        out.push((e & !(PAGE - 1), e & 1 != 0, rip));
        tail += 1;
    }
    TAIL.store(tail, Ordering::Release);
    if STATS[S_OVERFLOW].swap(0, Ordering::AcqRel) != 0 {
        lost = true;
    }
    !lost
}

pub fn drain(out: &mut Vec<u64>) -> bool {
    let mut v = Vec::new();
    let ok = drain_faults(&mut v);
    out.extend(v.into_iter().map(|(p, _, _)| p));
    ok
}

/// Write faults on armed pages answered so far.
pub fn faults() -> u64 {
    STATS[F_SIGNAL].load(Ordering::Relaxed) + STATS[F_SERVICE].load(Ordering::Relaxed)
}

pub fn status() -> String {
    if !on() {
        return "Linux page arming off (DS_LINUX_ARMING=0)".into();
    }
    let s = |i: usize| STATS[i].load(Ordering::Relaxed);
    format!(
        "Linux page arming: {} armed pages now; {} arms, {} disarms, {} failures; faults answered {} in the handler, {} by the service ({} refused during remaps); {} ring overflows",
        s(S_ARMED_PAGES), s(S_ARMS), s(S_DISARMS), s(S_FAILED), s(F_SIGNAL), s(F_SERVICE), s(F_REFUSED), s(S_OVERFLOW)
    )
}
