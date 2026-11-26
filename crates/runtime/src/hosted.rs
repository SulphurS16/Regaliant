//! Hosted mode: dsport under a host process that takes the write faults on
//! guest pages dsport armed (by default under front end 2, the default front
//! end since 2026-10-30, and with `--host`: `wants_host`; RENDERER.md
//! "Resident front end", stage H1).
//!
//! A fault handled inside the process overwrites the 128 bytes below the
//! faulting thread's stack pointer with the state Windows saves for the
//! handler, and the title's leaf functions (System V) keep values there; a
//! debugger is told of a fault before Windows builds that frame, so faults
//! it answers leave them intact (FINDINGS.md 2026-10-26, `faultbench`). Each
//! one stops the whole process for 15-50 us, so this is for writes that are
//! rare.
//!
//! The host (`dsport <args>` where `wants_host`) starts dsport again with the
//! same arguments as its debuggee and shares a mapping with it (named in
//! `DS_HOSTED`, which keeps the debuggee from hosting again): a bitmap of
//! armed guest pages (two levels: an index of 128 MiB leaves over the guest
//! range, leaves allocated by dsport) and a ring of the pages the host made
//! accessible again. A fault on an armed page (read-only or no-access) is
//! answered in the host: the page made read-write, its bit cleared, the page
//! with the access and the faulting instruction put in the ring, the access
//! retried. A fault on a guest page that allows the access by the time the
//! host sees it raced one the host answered for another thread (or a
//! disarm): the access is retried, as is one inside a range a view split
//! has unmapped for the moment (section-backed memory: `splitting`).
//! Every other exception goes to dsport's own handlers (first chance not
//! handled; the first PASSED_LOG are recorded for the status line). dsport arms a page by setting its bit before
//! protecting it, so an armed page's fault never reaches its own handlers,
//! and disarms by unprotecting before clearing the bit, one arm or disarm
//! at a time (`arm_lock`); after a mapping change the bits are only cleared
//! (`forget`).
//!
//! Held pages (`hold`; FRONTEND2.md "GPU-held pages"): armed no-access with
//! a held bit besides, their newest words elsewhere. A fault on one is not
//! answered: the host suspends the faulting thread and posts a request
//! (page, access, instruction, thread) in a second ring, signalling an event;
//! dsport puts the page's words back, makes it accessible (`release`) and
//! resumes the thread (`resume_waiter`), which runs the access again. Word
//! answers leave held pages alone; a disarm or mapping change clears their
//! held bits too (counted: their words were not put back).

use std::sync::atomic::{AtomicU32, AtomicU64, Ordering};
use windows_sys::Win32::Foundation::*;
use windows_sys::Win32::System::Diagnostics::Debug::*;
use windows_sys::Win32::System::Memory::*;
use windows_sys::Win32::System::Threading::*;

const PAGE: u64 = 4096;
/// Pages per leaf of the bitmap (128 MiB of guest space).
const LEAF_PAGES: u64 = 1 << 15;
/// Leaves over the guest range.
const L1: usize = ((crate::mem::GUEST_END - crate::mem::GUEST_BASE) / PAGE / LEAF_PAGES) as usize;
/// Leaves that can be allocated (512 GiB of armed guest space).
const MAX_LEAVES: usize = 4096;
/// Ring entries.
const RING: usize = 1 << 16;
/// Exceptions passed on that are recorded (`Shared::passed_log`).
const PASSED_LOG: usize = 16;
/// Held-page requests in flight at most (each one's thread is suspended
/// until dsport takes it: no more than the threads that exist).
const HELD_RING: usize = 4096;
/// Runs one hold request carries (`hold_by_host`).
pub const HOLD_RUNS: usize = 1024;
const MAGIC: u64 = 0x6473_706f_7274_4832; // "dsportH2"

/// The shared mapping.
#[repr(C)]
struct Shared {
    magic: u64,
    /// Write faults the host answered, its time answering them (ns), the
    /// exceptions it passed on, ring entries lost to a full ring.
    faults: AtomicU64,
    host_ns: AtomicU64,
    passed: AtomicU64,
    overflow: AtomicU64,
    /// Faults retried because the page allowed the access by then.
    retried: AtomicU64,
    /// Debug events by code (1-9: exception, thread and process creation,
    /// thread and process exit, DLL load and unload, debug string, RIP), and
    /// the time from each event's report until the host continued it (ns).
    events: [AtomicU64; 10],
    event_ns: [AtomicU64; 10],
    /// Ring entries written (by the host).
    head: AtomicU64,
    /// Ring entries dsport took.
    tail: AtomicU64,
    leaves_used: AtomicU32,
    /// WORD_MODE: a fault makes its 64-page word's armed pages read-write.
    flags: AtomicU32,
    /// Leaf index + 1 per 128 MiB of guest space (0: none).
    l1: [AtomicU32; L1],
    /// Per 128 MiB of guest space: armed bits cleared there so far (faults
    /// the host answered, disarms, mapping changes): a reader's view of
    /// the leaf is current while the count stays the same.
    leaf_events: [AtomicU64; L1],
    /// The same over the whole guest range.
    all_events: AtomicU64,
    /// The pages of the arm in progress, [lo, hi) (hi 0: none): their bits
    /// may be set before their protection (the host's word answers leave
    /// them alone).
    arming: [AtomicU64; 2],
    /// Pages the host made read-write with a faulted page's word.
    word_pages: AtomicU64,
    /// The first PASSED_LOG exceptions passed on: (code | first chance <<
    /// 32 | access kind << 40, address accessed, faulting instruction,
    /// thread id).
    passed_log: [[AtomicU64; 4]; PASSED_LOG],
    /// The guest range a view split has unmapped for the moment, [lo, hi)
    /// (hi 0: none; section-backed memory, `sections::split_begin`), and
    /// the faults inside it the host retried.
    split: [AtomicU64; 2],
    split_retried: AtomicU64,
    /// Faults on held pages: requests posted, posted for threads that could
    /// not be suspended (they fault again until the page is released),
    /// faults not posted for a full ring.
    held_faults: AtomicU64,
    held_spun: AtomicU64,
    held_full: AtomicU64,
    /// Faults on held pages answered as on armed pages (`HELD_ANSWERED`).
    held_answered: AtomicU64,
    /// Requests written (by the host) and taken (by dsport).
    held_head: AtomicU64,
    held_tail: AtomicU64,
    /// (page | 1 for a write | 2 when its thread was not suspended, the
    /// faulting instruction, thread id, QueryPerformanceCounter at the fault).
    held_ring: [[AtomicU64; 4]; HELD_RING],
    /// Holds the host makes in its event loop, in order with its answers
    /// (`hold_by_host`): requests written (dsport) and served (the host),
    /// their runs [lo, hi), and per run 1 held, 2 refused (a page no longer
    /// armed or already held: a fault was answered since dsport looked), 3
    /// failed (protection).
    hold_req: AtomicU64,
    hold_done: AtomicU64,
    hold_n: AtomicU64,
    hold_runs: [[AtomicU64; 3]; HOLD_RUNS],
    /// dsport's run folder (its log's), UTF-16, and its length (written once by dsport:
    /// `note_run_dir`): where the host notes how the game ended when the game could not.
    run_dir: [std::sync::atomic::AtomicU16; 520],
    run_dir_len: AtomicU32,
    /// (page | 1 for a write, the faulting instruction).
    ring: [[AtomicU64; 2]; RING],
    leaves: [[AtomicU64; (LEAF_PAGES / 64) as usize]; MAX_LEAVES],
    /// Held bits, by leaf as `leaves` (a held page's armed bit is set too).
    held: [[AtomicU64; (LEAF_PAGES / 64) as usize]; MAX_LEAVES],
}

/// dsport's view of the mapping (hosted), or null.
static SHARED: std::sync::atomic::AtomicPtr<Shared> = std::sync::atomic::AtomicPtr::new(std::ptr::null_mut());

fn wide(s: &str) -> Vec<u16> {
    s.encode_utf16().chain(std::iter::once(0)).collect()
}

fn shared() -> Option<&'static Shared> {
    let p = SHARED.load(Ordering::Acquire);
    // SAFETY: a mapping of `Shared`'s size, kept for the process's life.
    (!p.is_null()).then(|| unsafe { &*p })
}

/// The bit of guest page `a` (leaf allocated when `create`).
fn bit(s: &Shared, a: u64, create: bool) -> Option<(&AtomicU64, u64)> {
    if !(crate::mem::GUEST_BASE..crate::mem::GUEST_END).contains(&a) {
        return None;
    }
    let page = (a - crate::mem::GUEST_BASE) / PAGE;
    let slot = &s.l1[(page / LEAF_PAGES) as usize];
    let mut leaf = slot.load(Ordering::Acquire);
    if leaf == 0 {
        if !create {
            return None;
        }
        let n = s.leaves_used.fetch_add(1, Ordering::AcqRel);
        if n as usize >= MAX_LEAVES {
            return None;
        }
        leaf = match slot.compare_exchange(0, n + 1, Ordering::AcqRel, Ordering::Acquire) {
            Ok(_) => n + 1,
            // Another thread's leaf (this one stays unused).
            Err(other) => other,
        };
    }
    let i = page % LEAF_PAGES;
    Some((&s.leaves[(leaf - 1) as usize][(i / 64) as usize], 1u64 << (i % 64)))
}

/// The held bit of guest page `a` (its leaf is its armed bit's).
fn held_bit(s: &Shared, a: u64) -> Option<(&AtomicU64, u64)> {
    if !(crate::mem::GUEST_BASE..crate::mem::GUEST_END).contains(&a) {
        return None;
    }
    let page = (a - crate::mem::GUEST_BASE) / PAGE;
    let leaf = s.l1[(page / LEAF_PAGES) as usize].load(Ordering::Acquire);
    if leaf == 0 {
        return None;
    }
    let i = page % LEAF_PAGES;
    Some((&s.held[(leaf - 1) as usize][(i / 64) as usize], 1u64 << (i % 64)))
}

fn is_held(s: &Shared, a: u64) -> bool {
    held_bit(s, a).is_some_and(|(w, m)| w.load(Ordering::Acquire) & m != 0)
}

/// Whether guest page `a` is held now (no-access while the GPU's newer words are elsewhere: a
/// read releases it). Lock-free: for sweeps that pass such pages over (`uistate`).
pub fn page_held(a: u64) -> bool {
    shared().is_some_and(|s| is_held(s, a))
}

/// Whether dsport runs under its host.
pub fn on() -> bool {
    !SHARED.load(Ordering::Acquire).is_null()
}

/// Maps the host's shared state when `DS_HOSTED` names it (at start-up).
pub fn init() {
    let Ok(name) = std::env::var("DS_HOSTED") else { return };
    let size = std::mem::size_of::<Shared>();
    unsafe {
        let h = OpenFileMappingW(FILE_MAP_ALL_ACCESS, 0, wide(&name).as_ptr());
        if h.is_null() {
            crate::error!("hosted: cannot open the host's mapping {name}: error {}", GetLastError());
            return;
        }
        let v = MapViewOfFile(h, FILE_MAP_ALL_ACCESS, 0, 0, size);
        if v.Value.is_null() {
            crate::error!("hosted: cannot map {name}: error {}", GetLastError());
            return;
        }
        let s = v.Value as *mut Shared;
        if (*s).magic != MAGIC {
            crate::error!("hosted: {name} is not the host's mapping");
            return;
        }
        SHARED.store(s, Ordering::Release);
        // (0x0010_0000: SYNCHRONIZE.)
        let e = OpenEventW(0x0010_0000, 0, wide(&format!("{name}-held")).as_ptr());
        if e.is_null() {
            crate::warn!("hosted: cannot open the host's held-page event: error {} (requests are polled)", GetLastError());
        } else {
            HELD_EVENT.store(e as u64, Ordering::Release);
        }
    }
    crate::info!("hosted: under the host (write faults on armed guest pages answered there; {name})");
    // Every 5 s with debug events or faults since the last look: their
    // counts (each one stops the whole process while the host handles it).
    let _ = std::thread::Builder::new().name("host-events".into()).spawn(|| {
        let Some(s) = shared() else { return };
        let mut last = [0u64; 11];
        loop {
            std::thread::sleep(std::time::Duration::from_secs(5));
            let mut now = [0u64; 11];
            for (k, n) in now.iter_mut().enumerate().take(10) {
                *n = s.events[k].load(Ordering::Relaxed);
            }
            now[10] = s.faults.load(Ordering::Relaxed);
            if now == last {
                continue;
            }
            const NAMES: [&str; 10] = ["", "exceptions", "thread starts", "process start", "thread exits", "process exit", "DLL loads", "DLL unloads", "debug strings", "RIPs"];
            let parts: Vec<String> = (1..10).filter(|&k| now[k] != last[k]).map(|k| format!("{} {}", NAMES[k], now[k] - last[k])).collect();
            crate::info!("hosted: in 5 s: {}; armed-page faults {}", parts.join(", "), now[10] - last[10]);
            last = now;
        }
    });
}

/// The run folder (where the log is) for the host's note of how the game ended (`run_host`).
pub fn note_run_dir(dir: &std::path::Path) {
    let Some(s) = shared() else { return };
    let w: Vec<u16> = dir.as_os_str().to_string_lossy().encode_utf16().collect();
    if w.len() > s.run_dir.len() {
        return;
    }
    for (a, c) in s.run_dir.iter().zip(&w) {
        a.store(*c, Ordering::Relaxed);
    }
    s.run_dir_len.store(w.len() as u32, Ordering::Release);
}

/// The host: how the game ended when no handler of the game's own reported it (an exception it
/// did not handle, second chance, the debugger's to see: one inside a library such as NVIDIA's,
/// whose own crash handler or a fast fail ends the process before the game's can run), appended
/// to the game's log and written to `host-exit.txt` beside it. (2026-11-26: the first Frame
/// Generation test ended with nothing in the log.)
fn note_end(s: &Shared, text: &str) {
    let n = (s.run_dir_len.load(Ordering::Acquire) as usize).min(s.run_dir.len());
    if n == 0 {
        eprintln!("{text}");
        return;
    }
    let w: Vec<u16> = s.run_dir[..n].iter().map(|a| a.load(Ordering::Relaxed)).collect();
    let dir = std::path::PathBuf::from(String::from_utf16_lossy(&w));
    let _ = std::fs::write(dir.join("host-exit.txt"), format!("{text}\n"));
    use std::io::Write;
    if let Ok(mut f) = std::fs::OpenOptions::new().append(true).open(dir.join("dsport.log")) {
        let _ = writeln!(f, "[host] E: {text}");
    }
}

/// The name of the debuggee's module that holds `addr` and the offset into it, from the DLL load
/// events seen (the nearest base below: modules do not overlap).
fn module_at(mods: &[(u64, String)], addr: u64) -> String {
    match mods.iter().filter(|(b, _)| *b <= addr).max_by_key(|(b, _)| *b) {
        Some((b, n)) => format!("{n}+{:#x}", addr - b),
        None => format!("{addr:#x}"),
    }
}

/// A file handle's path (a debug event's image file), its file name.
fn handle_file_name(f: HANDLE) -> String {
    let mut buf = [0u16; 520];
    let n = unsafe { windows_sys::Win32::Storage::FileSystem::GetFinalPathNameByHandleW(f, buf.as_mut_ptr(), buf.len() as u32, 0) } as usize;
    let p = String::from_utf16_lossy(&buf[..n.min(buf.len())]);
    // (Numbered copies under NVIDIA's folders keep their folder's name: NGX's models and
    // Streamline's plugin overrides.)
    let parts: Vec<&str> = p.rsplit('\\').take(4).collect();
    match parts.as_slice() {
        [file, ..] if p.to_ascii_lowercase().contains("\\nvidia\\ngx") => format!("{} ({file})", parts.iter().rev().cloned().collect::<Vec<_>>().join("\\")),
        [file, ..] => file.to_string(),
        [] => "?".into(),
    }
}

/// A view split has unmapped guest [lo, hi) until the next call (hi 0:
/// done): the host retries faults inside it instead of passing them on.
pub fn splitting(lo: u64, hi: u64) {
    if let Some(s) = shared() {
        s.split[0].store(lo, Ordering::SeqCst);
        s.split[1].store(hi, Ordering::SeqCst);
    }
}

/// `Shared::flags`: a fault on an armed page makes every armed page of its
/// 64-page word read-write and clears their bits (the title writes runs of
/// pages: several pages of a word faulted within the millisecond before
/// dsport could disarm the rest, FINDINGS.md 2026-10-28). Not for pages
/// armed against reads (`arm_noaccess`): set by front end 2 only.
const WORD_MODE: u32 = 1;

/// `Shared::flags`: a fault on a held page is answered as on an armed page
/// (made read-write, its bits cleared, put in the ring): no thread waits
/// (the hold census, `gpu::fe2::hold_census`, where the page's words are
/// written back as ever).
const HELD_ANSWERED: u32 = 2;

/// Faults on held pages are answered from now on (`HELD_ANSWERED`).
pub fn set_held_answered() {
    if let Some(s) = shared() {
        s.flags.fetch_or(HELD_ANSWERED, Ordering::AcqRel);
    }
}

/// Whether held pages may be made: their faults are answered
/// (`HELD_ANSWERED`) or a release service runs (`set_held_served`).
static HELD_SERVED: std::sync::atomic::AtomicBool = std::sync::atomic::AtomicBool::new(false);

/// A release service takes held-page requests from now on (or no longer).
pub fn set_held_served(on: bool) {
    HELD_SERVED.store(on, Ordering::Release);
}

/// Faults answer for their whole 64-page word from now on (`WORD_MODE`).
pub fn set_word_mode() {
    if let Some(s) = shared() {
        s.flags.fetch_or(WORD_MODE, Ordering::AcqRel);
    }
}

/// Arming and disarming one at a time: a disarm between an arm's bits and
/// its protection would leave read-only pages without bits (a write there
/// would reach dsport's own handlers). Callers that check something before
/// arming (`gpu::fe2`) hold it across the check too.
static ARM_LOCK: parking_lot::ReentrantMutex<()> = parking_lot::ReentrantMutex::new(());

/// Holds off every other arm and disarm (re-entrant: `arm` and `disarm`
/// under it are fine).
pub fn arm_lock() -> parking_lot::ReentrantMutexGuard<'static, ()> {
    ARM_LOCK.lock()
}

/// `arm_lock` within `d`, else None.
pub fn try_arm_lock_for(d: std::time::Duration) -> Option<parking_lot::ReentrantMutexGuard<'static, ()>> {
    ARM_LOCK.try_lock_for(d)
}

/// Arms guest [base, base+len) (whole pages) against writes: its bits set,
/// then read-only.
pub fn arm(base: u64, len: u64) -> Result<(), String> {
    arm_as(base, len, PAGE_READONLY, false)
}

/// Arms guest [base, base+len) against any access: its bits set, then
/// no-access.
pub fn arm_noaccess(base: u64, len: u64) -> Result<(), String> {
    arm_as(base, len, PAGE_NOACCESS, false)
}

fn arm_as(base: u64, len: u64, prot: u32, hold: bool) -> Result<(), String> {
    let s = shared().ok_or("not hosted")?;
    let _lock = arm_lock();
    let (lo, hi) = (base & !(PAGE - 1), (base + len).next_multiple_of(PAGE));
    // (Arming never touches a held page: its newest words are elsewhere,
    // and a read-only one would let its stale words be read. The runs
    // between held pages are armed.)
    if !hold && HELD_PAGES.load(Ordering::Relaxed) > 0 && (lo..hi).step_by(PAGE as usize).any(|a| is_held(s, a)) {
        let mut a = lo;
        while a < hi {
            if is_held(s, a) {
                a += PAGE;
                continue;
            }
            let mut e = a + PAGE;
            while e < hi && !is_held(s, e) {
                e += PAGE;
            }
            arm_as(a, e - a, prot, false)?;
            a = e;
        }
        return Ok(());
    }
    // (Told to the host first: its word answers skip these pages.)
    s.arming[0].store(lo, Ordering::Release);
    s.arming[1].store(hi, Ordering::Release);
    let r = (|| {
        for a in (lo..hi).step_by(PAGE as usize) {
            // (Bits set so far go again: armed bits on writable pages would
            // pass their writes unseen.)
            let Some((w, m)) = bit(s, a, true) else {
                disarm_bits(s, lo, a, Clear::Undo);
                return Err(format!("{a:#x}: outside the guest range or out of leaves"));
            };
            w.fetch_or(m, Ordering::AcqRel);
            if hold && let Some((hw, hm)) = held_bit(s, a)
                && hw.fetch_or(hm, Ordering::AcqRel) & hm == 0
            {
                HELD_PAGES.fetch_add(1, Ordering::Relaxed);
            }
        }
        // (One call per allocation of the mixed range: `mem::host_pieces`;
        // a failed one puts the pieces before it back.)
        let mut old = 0u32;
        for (a, e) in crate::mem::host_pieces(lo, hi) {
            if unsafe { VirtualProtect(a as _, (e - a) as usize, prot, &mut old) } == 0 {
                let err = unsafe { GetLastError() };
                for (b, f) in crate::mem::host_pieces(lo, a) {
                    unsafe { VirtualProtect(b as _, (f - b) as usize, PAGE_READWRITE, &mut old) };
                }
                disarm_bits(s, lo, hi, Clear::Undo);
                return Err(format!("VirtualProtect {a:#x}+{:#x}: error {err}", e - a));
            }
        }
        Ok(())
    })();
    s.arming[1].store(0, Ordering::Release);
    r
}

/// Why armed (and held) bits are cleared: an arm or hold undone, held pages
/// released (their words put back), or pages disarmed or changed by a
/// mapping change (held pages among them dropped: counted).
#[derive(Clone, Copy, PartialEq)]
enum Clear {
    Undo,
    Released,
    Dropped,
}

/// Held pages now, and those dropped without a release (`Clear::Dropped`).
static HELD_PAGES: AtomicU64 = AtomicU64::new(0);
static HELD_DROPPED: AtomicU64 = AtomicU64::new(0);

fn disarm_bits(s: &Shared, lo: u64, hi: u64, why: Clear) {
    let mut last_leaf = usize::MAX;
    for a in (lo..hi).step_by(PAGE as usize) {
        if let Some((w, m)) = bit(s, a, false) {
            // (Held bits first: the host never sees a held page unarmed.)
            if let Some((hw, hm)) = held_bit(s, a)
                && hw.fetch_and(!hm, Ordering::AcqRel) & hm != 0
            {
                HELD_PAGES.fetch_sub(1, Ordering::Relaxed);
                if why == Clear::Dropped {
                    HELD_DROPPED.fetch_add(1, Ordering::Relaxed);
                }
            }
            w.fetch_and(!m, Ordering::AcqRel);
            let leaf = leaf_of(a);
            if leaf != last_leaf {
                s.leaf_events[leaf].fetch_add(1, Ordering::AcqRel);
                s.all_events.fetch_add(1, Ordering::AcqRel);
                last_leaf = leaf;
            }
        }
    }
}

/// The 128 MiB of guest space holding `a` (an index of `Shared::l1`).
fn leaf_of(a: u64) -> usize {
    ((a - crate::mem::GUEST_BASE) / PAGE / LEAF_PAGES) as usize
}

/// The end of the 128 MiB of guest space holding `a` (`leaf_events`' unit).
#[inline]
pub fn leaf_end(a: u64) -> u64 {
    let leaf = PAGE * LEAF_PAGES;
    crate::mem::GUEST_BASE + ((a.max(crate::mem::GUEST_BASE) - crate::mem::GUEST_BASE) / leaf + 1) * leaf
}

/// Armed bits cleared so far in the 128 MiB of guest space holding `a`
/// (0 when not hosted): unchanged, no armed page there lost its bit.
#[inline]
pub fn leaf_events(a: u64) -> u64 {
    let Some(s) = shared() else { return 0 };
    if !(crate::mem::GUEST_BASE..crate::mem::GUEST_END).contains(&a) {
        return 0;
    }
    s.leaf_events[leaf_of(a)].load(Ordering::Acquire)
}

/// Disarm runs whose pages could not be made read-write (they keep their
/// bits: the host still answers their faults).
static DISARM_FAILED: AtomicU64 = AtomicU64::new(0);

/// Disarms guest [base, base+len): its armed pages read-write again, then
/// their bits cleared, run by run within the host's regions. One
/// VirtualProtect cannot span separately allocated or uncommitted regions,
/// and clearing the bits after such a failed call left read-only pages
/// without bits, whose writes reach dsport's own handlers (a title crash
/// on a page of a writable mapping: FINDINGS.md 2026-10-27 night); pages
/// not armed keep their protection.
#[track_caller]
pub fn disarm(base: u64, len: u64) {
    let Some(s) = shared() else { return };
    let _lock = arm_lock();
    let lo = (base & !(PAGE - 1)).max(crate::mem::GUEST_BASE);
    let hi = base.saturating_add(len).next_multiple_of(PAGE).min(crate::mem::GUEST_END);
    if let Some(p) = crate::gpu::fe2::traced(lo, hi) {
        let at = std::panic::Location::caller();
        crate::gpu::fe2::trace_event(p, &format!("a host disarm over it: {lo:#x}..{hi:#x} from {}:{}", at.file().rsplit(['/', '\\']).next().unwrap_or(""), at.line()));
    }
    let armed = |a: u64| bit(s, a, false).is_some_and(|(w, m)| w.load(Ordering::Acquire) & m != 0);
    let mut a = lo;
    while a < hi {
        // The next armed page from a (unarmed words and leaves skipped).
        let Some((w, _)) = bit(s, a, false) else {
            a = crate::mem::GUEST_BASE + ((a - crate::mem::GUEST_BASE) / (PAGE * LEAF_PAGES) + 1) * PAGE * LEAF_PAGES;
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
        // (A held page stays held: it is released, its words put back, by
        // whoever holds it, never disarmed.)
        if HELD_PAGES.load(Ordering::Relaxed) > 0 && is_held(s, a) {
            a += PAGE;
            continue;
        }
        // Its run: armed pages up to the end of the host region holding it.
        let mut info: MEMORY_BASIC_INFORMATION = unsafe { std::mem::zeroed() };
        let region_end = if unsafe { VirtualQuery(a as _, &mut info, std::mem::size_of::<MEMORY_BASIC_INFORMATION>()) } != 0 { info.BaseAddress as u64 + info.RegionSize as u64 } else { a + PAGE };
        let mut e = a + PAGE;
        let any_held = HELD_PAGES.load(Ordering::Relaxed) > 0;
        while e < hi && e < region_end && armed(e) && !(any_held && is_held(s, e)) {
            e += PAGE;
        }
        let mut old = 0u32;
        if unsafe { VirtualProtect(a as _, (e - a) as usize, PAGE_READWRITE, &mut old) } != 0 {
            disarm_bits(s, a, e, Clear::Dropped);
        } else {
            let err = unsafe { GetLastError() };
            if DISARM_FAILED.fetch_add(1, Ordering::Relaxed) < 8 {
                crate::warn!("hosted: disarming {a:#x}+{:#x} failed (error {err}): its pages stay armed", e - a);
            }
        }
        a = e;
    }
}

/// The armed bits of the 64 pages from `a` (a multiple of 64 pages from the
/// guest range's start): bit i for page a + 4096 * i (0 when not hosted or
/// none of them was ever armed).
#[inline]
pub fn armed_word(a: u64) -> u64 {
    let Some(s) = shared() else { return 0 };
    bit(s, a, false).map_or(0, |(w, _)| w.load(Ordering::Acquire))
}

/// Whether every page of guest [base, base+len) is armed (no fault on any
/// of them since it was armed, no disarm, no mapping change).
pub fn all_armed(base: u64, len: u64) -> bool {
    let Some(s) = shared() else { return false };
    let (lo, hi) = (base & !(PAGE - 1), (base + len.max(1)).next_multiple_of(PAGE));
    (lo..hi).step_by(PAGE as usize).all(|a| bit(s, a, false).is_some_and(|(w, m)| w.load(Ordering::Acquire) & m != 0))
}

/// Disarms guest [base, base+len) if a page of it is armed (imports need
/// writable pages: the driver pins them for writing).
pub fn disarm_if_armed(base: u64, len: u64) {
    let Some(s) = shared() else { return };
    let (lo, hi) = (base & !(PAGE - 1), base.saturating_add(len).next_multiple_of(PAGE).min(crate::mem::GUEST_END));
    // Leaf by leaf: a range without a leaf has nothing armed.
    let mut a = lo;
    while a < hi {
        let leaf_end = ((a - crate::mem::GUEST_BASE) / (PAGE * LEAF_PAGES) + 1) * PAGE * LEAF_PAGES + crate::mem::GUEST_BASE;
        let end = leaf_end.min(hi);
        if bit(s, a, false).is_some() && (a..end).step_by(PAGE as usize).any(|p| bit(s, p, false).is_some_and(|(w, m)| w.load(Ordering::Acquire) & m != 0)) {
            disarm(a, end - a);
        }
        a = end;
    }
}

/// Whether guest page `a` is armed now (its bit set).
pub fn armed(a: u64) -> bool {
    shared().and_then(|s| bit(s, a, false)).is_some_and(|(w, m)| w.load(Ordering::Acquire) & m != 0)
}

/// Guest [base, base+len) changed its mapping or protection: its pages are
/// no longer armed (bits cleared, protection left as the change set it).
pub fn forget(base: u64, len: u64) {
    let Some(s) = shared() else { return };
    let (lo, hi) = (base & !(PAGE - 1), base.saturating_add(len).next_multiple_of(PAGE));
    disarm_bits(s, lo, hi.min(crate::mem::GUEST_END), Clear::Dropped);
    crate::hostcensus::forget(base, len);
}

/// Holds guest [base, base+len) (whole pages; the caller has checked under
/// `arm_lock` that they are readable and writable and not armed, and keeps
/// their newest words elsewhere): their armed and held bits set, then
/// no-access. A CPU access to one waits, its thread suspended by the host,
/// until dsport has put the page's words back and released it (`take_held`,
/// `release`, `resume_waiter`).
pub fn hold(base: u64, len: u64) -> Result<(), String> {
    // (A held page's thread waits for a release: never without a service,
    // unless the host answers them.)
    let s = shared().ok_or("not hosted")?;
    if s.flags.load(Ordering::Acquire) & HELD_ANSWERED == 0 && !HELD_SERVED.load(Ordering::Acquire) {
        return Err("no release service for held pages".into());
    }
    arm_as(base, len, PAGE_NOACCESS, true)
}

/// Holds `runs` (whole pages, armed by dsport's checks) through the host:
/// it serves the request in its event loop, between debug events, so no
/// fault it answers can interleave with a hold: a fault raised before finds
/// its page armed and is answered first (the run is then refused: a page
/// no longer armed), one raised after finds it held. Per run: true held.
/// At most HOLD_RUNS runs; waits up to 2 s (then none is held).
pub fn hold_by_host(runs: &[(u64, u64)]) -> Vec<bool> {
    let mut out = vec![false; runs.len()];
    let Some(s) = shared() else { return out };
    if s.flags.load(Ordering::Acquire) & HELD_ANSWERED == 0 && !HELD_SERVED.load(Ordering::Acquire) {
        return out;
    }
    static ONE: parking_lot::Mutex<()> = parking_lot::Mutex::new(());
    let _one = ONE.lock();
    let n = runs.len().min(HOLD_RUNS);
    for (i, &(lo, hi)) in runs[..n].iter().enumerate() {
        s.hold_runs[i][0].store(lo & !(PAGE - 1), Ordering::Relaxed);
        s.hold_runs[i][1].store(hi.next_multiple_of(PAGE), Ordering::Relaxed);
        s.hold_runs[i][2].store(0, Ordering::Relaxed);
    }
    s.hold_n.store(n as u64, Ordering::Relaxed);
    let seq = s.hold_req.load(Ordering::Acquire) + 1;
    s.hold_req.store(seq, Ordering::Release);
    let t = std::time::Instant::now();
    while s.hold_done.load(Ordering::Acquire) < seq {
        if t.elapsed() > std::time::Duration::from_secs(2) {
            // (The host may still serve it: the request taken back first.)
            if s.hold_req.compare_exchange(seq, seq - 1, Ordering::AcqRel, Ordering::Acquire).is_ok() {
                HOLD_LATE.fetch_add(1, Ordering::Relaxed);
                return out;
            }
        }
        std::thread::sleep(std::time::Duration::from_micros(200));
    }
    for (i, o) in out[..n].iter_mut().enumerate() {
        let r = s.hold_runs[i][2].load(Ordering::Acquire);
        *o = r == 1;
        if r == 1 {
            let pages = (s.hold_runs[i][1].load(Ordering::Relaxed) - s.hold_runs[i][0].load(Ordering::Relaxed)) / PAGE;
            HELD_PAGES.fetch_add(pages, Ordering::Relaxed);
        }
    }
    out
}

/// Hold requests the host did not serve within 2 s.
static HOLD_LATE: AtomicU64 = AtomicU64::new(0);

/// Whether guest page `a` is held now.
pub fn held(a: u64) -> bool {
    shared().is_some_and(|s| is_held(s, a))
}

/// Pages held now (0 when not hosted).
pub fn held_pages() -> u64 {
    HELD_PAGES.load(Ordering::Relaxed)
}

/// A request the host posted: an access to a held page, its thread
/// suspended (unless `suspended` is false: it faults again until the page
/// is released) until `resume_waiter`.
#[derive(Clone, Copy, Debug)]
pub struct HeldRequest {
    pub page: u64,
    pub write: bool,
    pub suspended: bool,
    /// The faulting instruction, the thread, QueryPerformanceCounter at the
    /// fault.
    pub rip: u64,
    pub tid: u32,
    pub stamp: i64,
}

/// The host's event for posted requests (a HANDLE; 0: none).
static HELD_EVENT: AtomicU64 = AtomicU64::new(0);
/// One taker at a time.
static HELD_TAKE: parking_lot::Mutex<()> = parking_lot::Mutex::new(());

/// The requests posted since the last call.
pub fn take_held(out: &mut Vec<HeldRequest>) {
    let Some(s) = shared() else { return };
    let _one = HELD_TAKE.lock();
    let head = s.held_head.load(Ordering::Acquire);
    let mut tail = s.held_tail.load(Ordering::Acquire);
    while tail < head {
        let e = &s.held_ring[(tail % HELD_RING as u64) as usize];
        let w = e[0].load(Ordering::Acquire);
        out.push(HeldRequest {
            page: w & !(PAGE - 1),
            write: w & 1 != 0,
            suspended: w & 2 == 0,
            rip: e[1].load(Ordering::Acquire),
            tid: e[2].load(Ordering::Acquire) as u32,
            stamp: e[3].load(Ordering::Acquire) as i64,
        });
        tail += 1;
    }
    s.held_tail.store(tail, Ordering::Release);
}

/// Waits up to `ms` for the host to post a request (true: signalled).
pub fn wait_held(ms: u32) -> bool {
    let e = HELD_EVENT.load(Ordering::Acquire);
    if e == 0 {
        std::thread::sleep(std::time::Duration::from_millis(ms as u64));
        return false;
    }
    unsafe { WaitForSingleObject(e as HANDLE, ms) == WAIT_OBJECT_0 }
}

/// The held pages of guest [base, base+len) have their newest words in
/// guest memory (put there by the caller): made read-write, then their held
/// and armed bits cleared, run by run within the host's regions. The pages
/// released; a run whose protection could not be changed stays held.
pub fn release(base: u64, len: u64) -> Result<u64, String> {
    let s = shared().ok_or("not hosted")?;
    let _lock = arm_lock();
    let lo = (base & !(PAGE - 1)).max(crate::mem::GUEST_BASE);
    let hi = base.saturating_add(len).next_multiple_of(PAGE).min(crate::mem::GUEST_END);
    let mut a = lo;
    let mut n = 0u64;
    let mut failed = None;
    while a < hi {
        if !is_held(s, a) {
            a += PAGE;
            continue;
        }
        let mut e = a + PAGE;
        while e < hi && is_held(s, e) {
            e += PAGE;
        }
        let mut old = 0u32;
        for (p, q) in crate::mem::host_pieces(a, e) {
            if unsafe { VirtualProtect(p as _, (q - p) as usize, PAGE_READWRITE, &mut old) } != 0 {
                disarm_bits(s, p, q, Clear::Released);
                n += (q - p) / PAGE;
            } else if failed.is_none() {
                failed = Some(format!("VirtualProtect {p:#x}+{:#x}: error {}", q - p, unsafe { GetLastError() }));
            }
        }
        a = e;
    }
    match failed {
        Some(e) => Err(format!("{e} (those pages stay held; {n} released)")),
        None => Ok(n),
    }
}

/// Held pages of guest [base, base+len) the caller made read-write itself
/// (their newest words put back with every other thread suspended): their
/// held and armed bits cleared. Allocates nothing and takes no lock (the
/// caller holds `arm_lock`).
pub fn released_bits(base: u64, len: u64) {
    let Some(s) = shared() else { return };
    let (lo, hi) = (base & !(PAGE - 1), base.saturating_add(len).next_multiple_of(PAGE).min(crate::mem::GUEST_END));
    disarm_bits(s, lo, hi, Clear::Released);
}

/// Waits from a fault on a held page to its thread's resume: count, total
/// and longest (ns); resumes that failed.
static HELD_WAITS: [AtomicU64; 4] = [const { AtomicU64::new(0) }; 4];

fn qpc() -> i64 {
    let mut t = 0i64;
    unsafe { windows_sys::Win32::System::Performance::QueryPerformanceCounter(&mut t) };
    t
}

fn qpc_ns(ticks: i64) -> u64 {
    static FREQ: std::sync::OnceLock<i64> = std::sync::OnceLock::new();
    let f = *FREQ.get_or_init(|| {
        let mut f = 0i64;
        unsafe { windows_sys::Win32::System::Performance::QueryPerformanceFrequency(&mut f) };
        f.max(1)
    });
    (ticks.max(0) as u128 * 1_000_000_000 / f as u128) as u64
}

/// A request's page is released (or no longer held): its thread resumed
/// (the host suspended it); the wait from the fault (ns), or an error.
pub fn resume_waiter(r: &HeldRequest) -> Result<u64, String> {
    let wait = qpc_ns(qpc() - r.stamp);
    if r.suspended {
        let h = unsafe { OpenThread(THREAD_SUSPEND_RESUME, 0, r.tid) };
        let n = if h.is_null() {
            u32::MAX
        } else {
            let n = unsafe { ResumeThread(h) };
            unsafe { CloseHandle(h) };
            n
        };
        if n == u32::MAX || n == 0 {
            HELD_WAITS[3].fetch_add(1, Ordering::Relaxed);
            return Err(format!("thread {}: {}", r.tid, if n == 0 { "was not suspended".to_string() } else { format!("cannot resume it (error {})", unsafe { GetLastError() }) }));
        }
    }
    HELD_WAITS[0].fetch_add(1, Ordering::Relaxed);
    HELD_WAITS[1].fetch_add(wait, Ordering::Relaxed);
    HELD_WAITS[2].fetch_max(wait, Ordering::Relaxed);
    Ok(wait)
}

/// The faults the host answered since the last call: (page, a write, the
/// faulting instruction); false when the ring overflowed meanwhile (faults
/// were lost: treat all armed pages as touched).
pub fn drain_faults(out: &mut Vec<(u64, bool, u64)>) -> bool {
    let Some(s) = shared() else { return true };
    let head = s.head.load(Ordering::Acquire);
    let mut tail = s.tail.load(Ordering::Acquire);
    let lost = head.saturating_sub(tail) > RING as u64;
    if lost {
        tail = head - RING as u64;
    }
    while tail < head {
        let e = &s.ring[(tail % RING as u64) as usize];
        let (a, rip) = (e[0].load(Ordering::Acquire), e[1].load(Ordering::Acquire));
        out.push((a & !(PAGE - 1), a & 1 != 0, rip));
        tail += 1;
    }
    s.tail.store(tail, Ordering::Release);
    !lost && s.overflow.load(Ordering::Acquire) == 0
}

/// The pages the host made accessible since the last call (`drain_faults`).
pub fn drain(out: &mut Vec<u64>) -> bool {
    let mut v = Vec::new();
    let ok = drain_faults(&mut v);
    out.extend(v.into_iter().map(|(p, _, _)| p));
    ok
}

/// Write faults on armed pages the host answered so far (0 when not hosted).
pub fn faults() -> u64 {
    shared().map_or(0, |s| s.faults.load(Ordering::Relaxed))
}

/// Status line ("" when not hosted).
pub fn status() -> String {
    let Some(s) = shared() else { return String::new() };
    let f = s.faults.load(Ordering::Relaxed);
    const NAMES: [&str; 10] = ["", "exceptions", "thread starts", "process start", "thread exits", "process exit", "DLL loads", "DLL unloads", "debug strings", "RIPs"];
    let events: Vec<String> = (1..10)
        .filter_map(|k| {
            let n = s.events[k].load(Ordering::Relaxed);
            (n > 0).then(|| format!("{} {n} ({:.0} ms in the host)", NAMES[k], s.event_ns[k].load(Ordering::Relaxed) as f64 / 1e6))
        })
        .collect();
    // (What the first ones were: each was handled inside the process.)
    let passed = s.passed.load(Ordering::Relaxed);
    let which: Vec<String> = (0..(passed as usize).min(PASSED_LOG))
        .filter_map(|k| {
            let l = &s.passed_log[k];
            let tid = l[3].load(Ordering::Acquire);
            if tid == 0 {
                return None;
            }
            let w = l[0].load(Ordering::Relaxed);
            let (code, first, kind) = (w as u32, (w >> 32) & 0xff != 0, (w >> 40) & 0xff);
            let rip = l[2].load(Ordering::Relaxed);
            let at = if (crate::mem::GUEST_BASE..crate::mem::GUEST_END).contains(&rip) { crate::exception::describe_guest(rip) } else { crate::exception::host_symbol(rip) };
            let chance = if first { "" } else { ", second chance" };
            Some(if code == 0xC000_0005 {
                let access = match kind {
                    0 => "read",
                    1 => "write",
                    8 => "execute",
                    _ => "access",
                };
                format!("{code:#x} ({access} of {:#x}{chance}) at {at}, thread {tid}", l[1].load(Ordering::Relaxed))
            } else {
                format!("{code:#x}{chance} at {at}, thread {tid}")
            })
        })
        .collect();
    let which = if which.is_empty() { String::new() } else { format!(" ({})", which.join("; ")) };
    let hf = s.held_faults.load(Ordering::Relaxed);
    let ha = s.held_answered.load(Ordering::Relaxed);
    let held = if hf + ha + HELD_PAGES.load(Ordering::Relaxed) + HELD_DROPPED.load(Ordering::Relaxed) == 0 {
        String::new()
    } else {
        let w: Vec<u64> = HELD_WAITS.iter().map(|c| c.load(Ordering::Relaxed)).collect();
        format!(
            "; held pages: {} now, {ha} faults on them answered at once (`HELD_ANSWERED`), {hf} requests ({} for threads not suspended, {} faults not posted: ring full), waits {:.0} us on average and {:.0} us at most over {}, resumes that failed {}, held pages dropped without a release (disarms, mapping changes) {}",
            HELD_PAGES.load(Ordering::Relaxed).saturating_sub(ha),
            s.held_spun.load(Ordering::Relaxed),
            s.held_full.load(Ordering::Relaxed),
            w[1] as f64 / 1000.0 / w[0].max(1) as f64,
            w[2] as f64 / 1000.0,
            w[0],
            w[3],
            HELD_DROPPED.load(Ordering::Relaxed)
        )
    };
    format!(
        "    hosted: {} faults on armed pages answered by the host ({:.1} us each there; {} more pages made read-write with their words), {} retried (raced an answered one), {} retried inside a view split, {} exceptions passed on{which}, {} leaves of the armed-page map, ring overflows {}, disarms that failed {}{held}; debug events: {}\n{}",
        f,
        s.host_ns.load(Ordering::Relaxed) as f64 / 1000.0 / f.max(1) as f64,
        s.word_pages.load(Ordering::Relaxed),
        s.retried.load(Ordering::Relaxed),
        s.split_retried.load(Ordering::Relaxed),
        passed,
        s.leaves_used.load(Ordering::Relaxed),
        s.overflow.load(Ordering::Relaxed),
        DISARM_FAILED.load(Ordering::Relaxed),
        events.join(", "),
        crate::hostcensus::status()
    )
}

/// The host (`wants_host`): runs dsport with the other arguments as its
/// debuggee and answers write faults on armed pages; exits with its exit
/// code.
pub fn run_host() -> ! {
    let size = std::mem::size_of::<Shared>();
    let name = format!("Local\\dsport-host-{}", std::process::id());
    let s: &Shared = unsafe {
        let h = CreateFileMappingW(INVALID_HANDLE_VALUE, std::ptr::null(), PAGE_READWRITE, (size as u64 >> 32) as u32, size as u32, wide(&name).as_ptr());
        if h.is_null() {
            eprintln!("dsport --host: CreateFileMapping: error {}", GetLastError());
            std::process::exit(2);
        }
        let v = MapViewOfFile(h, FILE_MAP_ALL_ACCESS, 0, 0, size);
        if v.Value.is_null() {
            eprintln!("dsport --host: MapViewOfFile: error {}", GetLastError());
            std::process::exit(2);
        }
        let s = v.Value as *mut Shared;
        (*s).magic = MAGIC;
        &*s
    };
    // Signalled for each held-page request (dsport's service waits on it;
    // made before the debuggee, which opens it at start-up).
    let held_event = unsafe { CreateEventW(std::ptr::null(), 0, 0, wide(&format!("{name}-held")).as_ptr()) };
    let exe = std::env::current_exe().expect("own path");
    let args: Vec<String> = std::env::args().skip(1).filter(|a| a != "--host").collect();
    let cmdline = std::iter::once(format!("\"{}\"", exe.display())).chain(args.iter().map(|a| quote_arg(a))).collect::<Vec<_>>().join(" ");
    // SAFETY: before any other thread of the host exists.
    unsafe {
        std::env::set_var("DS_HOSTED", &name);
        // A process started by a debugger otherwise gets the checking heap.
        std::env::set_var("_NO_DEBUG_HEAP", "1");
    }
    let mut cmd = wide(&cmdline);
    let mut si: STARTUPINFOW = unsafe { std::mem::zeroed() };
    si.cb = std::mem::size_of::<STARTUPINFOW>() as u32;
    let mut pi: PROCESS_INFORMATION = unsafe { std::mem::zeroed() };
    let ok = unsafe { CreateProcessW(std::ptr::null(), cmd.as_mut_ptr(), std::ptr::null(), std::ptr::null(), 1, DEBUG_ONLY_THIS_PROCESS, std::ptr::null(), std::ptr::null(), &si, &mut pi) };
    if ok == 0 {
        eprintln!("dsport --host: CreateProcess: error {}", unsafe { GetLastError() });
        std::process::exit(2);
    }
    // Threads that fault on a held page without being suspended: (thread,
    // page) posted once until the page is no longer held.
    let mut spinning: Vec<(u32, u64)> = Vec::new();
    let mut first_bp = true;
    let mut code = 0u32;
    // The debuggee's modules (base, name) and the last exception no handler took.
    let mut mods: Vec<(u64, String)> = Vec::new();
    let mut fatal: Option<String> = None;
    loop {
        // (Hold requests between events: `hold_by_host`. Waits of 1 ms so
        // that a request waits no longer than that for a quiet process.)
        serve_holds(s, pi.hProcess);
        let mut ev: DEBUG_EVENT = unsafe { std::mem::zeroed() };
        if unsafe { WaitForDebugEvent(&mut ev, 1) } == 0 {
            if unsafe { GetLastError() } == windows_sys::Win32::Foundation::ERROR_SEM_TIMEOUT {
                continue;
            }
            break;
        }
        let t_event = std::time::Instant::now();
        let kind = (ev.dwDebugEventCode as usize).min(9);
        let mut status = DBG_CONTINUE;
        match ev.dwDebugEventCode {
            CREATE_PROCESS_DEBUG_EVENT => unsafe {
                let f = ev.u.CreateProcessInfo.hFile;
                if !f.is_null() {
                    mods.push((ev.u.CreateProcessInfo.lpBaseOfImage as u64, handle_file_name(f)));
                    CloseHandle(f);
                }
            },
            LOAD_DLL_DEBUG_EVENT => unsafe {
                let f = ev.u.LoadDll.hFile;
                let base = ev.u.LoadDll.lpBaseOfDll as u64;
                mods.retain(|(b, _)| *b != base);
                if !f.is_null() {
                    mods.push((base, handle_file_name(f)));
                    CloseHandle(f);
                }
            },
            UNLOAD_DLL_DEBUG_EVENT => unsafe {
                let base = ev.u.UnloadDll.lpBaseOfDll as u64;
                mods.retain(|(b, _)| *b != base);
            },
            EXCEPTION_DEBUG_EVENT => unsafe {
                let t = std::time::Instant::now();
                let e = ev.u.Exception;
                let rec = e.ExceptionRecord;
                let exc = rec.ExceptionCode as u32;
                let (write, addr) = (rec.ExceptionInformation[0] == 1, rec.ExceptionInformation[1] as u64);
                let held = exc == 0xC000_0005 && e.dwFirstChance != 0 && is_held(s, addr);
                let armed = exc == 0xC000_0005 && e.dwFirstChance != 0 && bit(s, addr, false).is_some_and(|(w, m)| w.load(Ordering::Acquire) & m != 0);
                // (`HELD_ANSWERED`: held pages answered as armed ones.)
                let answered = held && s.flags.load(Ordering::Acquire) & HELD_ANSWERED != 0;
                if held && !answered {
                    // Not answered: the thread waits, suspended, for dsport
                    // to put the page's words back and resume it (it runs
                    // the access again then).
                    let page = addr & !(PAGE - 1);
                    let tid = ev.dwThreadId;
                    spinning.retain(|&(_, p)| is_held(s, p));
                    let full = s.held_head.load(Ordering::Acquire).saturating_sub(s.held_tail.load(Ordering::Acquire)) >= HELD_RING as u64;
                    let mut suspended = false;
                    if !full {
                        let h = OpenThread(THREAD_SUSPEND_RESUME, 0, tid);
                        if !h.is_null() {
                            suspended = SuspendThread(h) != u32::MAX;
                            CloseHandle(h);
                        }
                    }
                    if full {
                        s.held_full.fetch_add(1, Ordering::Relaxed);
                    } else if suspended || !spinning.contains(&(tid, page)) {
                        if !suspended {
                            spinning.push((tid, page));
                            s.held_spun.fetch_add(1, Ordering::Relaxed);
                        }
                        let head = s.held_head.load(Ordering::Acquire);
                        let slot = &s.held_ring[(head % HELD_RING as u64) as usize];
                        slot[0].store(page | write as u64 | (!suspended as u64) << 1, Ordering::Release);
                        slot[1].store(rec.ExceptionAddress as u64, Ordering::Release);
                        slot[2].store(tid as u64, Ordering::Release);
                        slot[3].store(qpc() as u64, Ordering::Release);
                        s.held_head.store(head + 1, Ordering::Release);
                        s.held_faults.fetch_add(1, Ordering::Relaxed);
                        if !held_event.is_null() {
                            SetEvent(held_event);
                        }
                    }
                } else if armed {
                    let page = addr & !(PAGE - 1);
                    let mut old = 0u32;
                    VirtualProtectEx(pi.hProcess, page as _, PAGE as usize, PAGE_READWRITE, &mut old);
                    if let Some((w, m)) = bit(s, page, false) {
                        w.fetch_and(!m, Ordering::AcqRel);
                    }
                    if let Some((hw, hm)) = held_bit(s, page)
                        && hw.fetch_and(!hm, Ordering::AcqRel) & hm != 0
                    {
                        s.held_answered.fetch_add(1, Ordering::Relaxed);
                    }
                    if s.flags.load(Ordering::Acquire) & WORD_MODE != 0 {
                        let n = answer_word(s, pi.hProcess, page);
                        s.word_pages.fetch_add(n, Ordering::Relaxed);
                    }
                    s.leaf_events[leaf_of(page)].fetch_add(1, Ordering::AcqRel);
                    s.all_events.fetch_add(1, Ordering::AcqRel);
                    let head = s.head.load(Ordering::Acquire);
                    if head.saturating_sub(s.tail.load(Ordering::Acquire)) >= RING as u64 {
                        s.overflow.fetch_add(1, Ordering::Relaxed);
                    }
                    let slot = &s.ring[(head % RING as u64) as usize];
                    slot[0].store(page | write as u64, Ordering::Release);
                    slot[1].store(rec.ExceptionAddress as u64, Ordering::Release);
                    s.head.store(head + 1, Ordering::Release);
                    s.faults.fetch_add(1, Ordering::Relaxed);
                    s.host_ns.fetch_add(t.elapsed().as_nanos() as u64, Ordering::Relaxed);
                } else if exc == 0xC000_0005 && e.dwFirstChance != 0 && (s.split[0].load(Ordering::SeqCst)..s.split[1].load(Ordering::SeqCst)).contains(&addr) {
                    // Inside a view split (section-backed memory): tried
                    // again until the split maps the page back (dsport's own
                    // wait would run on the faulting thread's stack, over
                    // the guest's red zone).
                    s.split_retried.fetch_add(1, Ordering::Relaxed);
                } else if exc == 0xC000_0005 && e.dwFirstChance != 0 && rec.ExceptionInformation[0] <= 1 && allows(pi.hProcess, addr, write) {
                    s.retried.fetch_add(1, Ordering::Relaxed);
                } else if exc == 0x8000_0003 && first_bp {
                    first_bp = false;
                } else {
                    status = DBG_EXCEPTION_NOT_HANDLED;
                    // (Second chance: no handler in the game took it; the process ends now.)
                    if e.dwFirstChance == 0 {
                        let at = module_at(&mods, rec.ExceptionAddress as u64);
                        let detail = match exc {
                            0xC000_0005 => format!(" ({} of {addr:#x})", match rec.ExceptionInformation[0] { 0 => "read", 1 => "write", 8 => "execute", _ => "access" }),
                            0xC000_0409 => format!(" (fast fail, code {})", rec.ExceptionInformation[0]),
                            _ => String::new(),
                        };
                        fatal = Some(format!("the game ended on exception {exc:#010x}{detail} at {at}, thread {}, which no handler of its own took", ev.dwThreadId));
                    }
                    let k = s.passed.fetch_add(1, Ordering::Relaxed) as usize;
                    if k < PASSED_LOG {
                        let l = &s.passed_log[k];
                        l[0].store(exc as u64 | (e.dwFirstChance as u64) << 32 | (rec.ExceptionInformation[0] as u64 & 0xff) << 40, Ordering::Relaxed);
                        l[1].store(addr, Ordering::Relaxed);
                        l[2].store(rec.ExceptionAddress as u64, Ordering::Relaxed);
                        l[3].store(ev.dwThreadId as u64, Ordering::Release);
                    }
                }
            },
            EXIT_PROCESS_DEBUG_EVENT => unsafe {
                code = ev.u.ExitProcess.dwExitCode;
                // (A status code as the exit code without an exception seen: a library's crash
                // handler ended the process, or Windows did.)
                match &fatal {
                    Some(f) => note_end(s, &format!("{f}; exit code {code:#x}")),
                    None if code >= 0xC000_0000 => note_end(s, &format!("the game ended with exit code {code:#010x} and no report of its own (ended by a library's crash handler, a fast fail or Windows)")),
                    None => {}
                }
                ContinueDebugEvent(ev.dwProcessId, ev.dwThreadId, DBG_CONTINUE);
                break;
            },
            _ => {}
        }
        s.events[kind].fetch_add(1, Ordering::Relaxed);
        s.event_ns[kind].fetch_add(t_event.elapsed().as_nanos() as u64, Ordering::Relaxed);
        unsafe { ContinueDebugEvent(ev.dwProcessId, ev.dwThreadId, status) };
    }
    // The game's process goes on ending after its exit event: its memory is freed and its
    // handles closed (the one-session lock among them) only then. The host ends after it, so
    // that whoever waits for the host finds the session gone: the launcher's next shader
    // preparation step, started at the host's exit, met the lock still held and ended
    // ("another game session runs", 2026-11-20). At most a minute.
    unsafe { WaitForSingleObject(pi.hProcess, 60_000) };
    std::process::exit(code as i32);
}

/// `WORD_MODE` (the debuggee stopped): the armed pages of `page`'s 64-page
/// word other than those of an arm in progress made read-write, run by run
/// (page by page where a run fails: it may cross separately allocated
/// regions), and their bits cleared where that worked. The pages it did.
/// The host serves dsport's hold request, if one is pending (`hold_by_host`):
/// a run whose pages are all armed and none held gets its held bits, then
/// no-access; else it is refused.
fn serve_holds(s: &Shared, process: HANDLE) {
    let req = s.hold_req.load(Ordering::Acquire);
    if s.hold_done.load(Ordering::Acquire) >= req {
        return;
    }
    let n = (s.hold_n.load(Ordering::Acquire) as usize).min(HOLD_RUNS);
    for i in 0..n {
        let (lo, hi) = (s.hold_runs[i][0].load(Ordering::Acquire), s.hold_runs[i][1].load(Ordering::Acquire));
        let ok = lo < hi && (lo..hi).step_by(PAGE as usize).all(|a| bit(s, a, false).is_some_and(|(w, m)| w.load(Ordering::Acquire) & m != 0) && !is_held(s, a));
        let r = if !ok {
            2
        } else {
            for a in (lo..hi).step_by(PAGE as usize) {
                if let Some((hw, hm)) = held_bit(s, a) {
                    hw.fetch_or(hm, Ordering::AcqRel);
                }
            }
            let mut old = 0u32;
            if unsafe { VirtualProtectEx(process, lo as _, (hi - lo) as usize, PAGE_NOACCESS, &mut old) } != 0 {
                1
            } else {
                // (Page by page across allocations; any that fails: the run's
                // held bits taken back, its pages read-only again.)
                let all = (lo..hi).step_by(PAGE as usize).all(|a| unsafe { VirtualProtectEx(process, a as _, PAGE as usize, PAGE_NOACCESS, &mut old) } != 0);
                if all {
                    1
                } else {
                    for a in (lo..hi).step_by(PAGE as usize) {
                        unsafe { VirtualProtectEx(process, a as _, PAGE as usize, PAGE_READONLY, &mut old) };
                        if let Some((hw, hm)) = held_bit(s, a) {
                            hw.fetch_and(!hm, Ordering::AcqRel);
                        }
                    }
                    3
                }
            }
        };
        s.hold_runs[i][2].store(r, Ordering::Release);
    }
    s.hold_done.store(req, Ordering::Release);
}

unsafe fn answer_word(s: &Shared, process: HANDLE, page: u64) -> u64 {
    let wa = page & !(64 * PAGE - 1);
    let Some((w, _)) = bit(s, wa, false) else { return 0 };
    let mut bits = w.load(Ordering::Acquire);
    // (Held pages wait for their words.)
    if let Some((hw, _)) = held_bit(s, wa) {
        bits &= !hw.load(Ordering::Acquire);
    }
    let (alo, ahi) = (s.arming[0].load(Ordering::Acquire), s.arming[1].load(Ordering::Acquire));
    if ahi > wa && alo < wa + 64 * PAGE {
        for k in 0..64 {
            let a = wa + k * PAGE;
            if (alo..ahi).contains(&a) {
                bits &= !(1u64 << k);
            }
        }
    }
    let mut done = 0u64;
    let mut r = bits;
    while r != 0 {
        let b = r.trailing_zeros() as u64;
        let run = (!(r >> b)).trailing_zeros().min(64 - b as u32) as u64;
        let m = if run >= 64 { u64::MAX } else { ((1u64 << run) - 1) << b };
        r &= !m;
        let mut old = 0u32;
        if VirtualProtectEx(process, (wa + b * PAGE) as _, (run * PAGE) as usize, PAGE_READWRITE, &mut old) != 0 {
            done |= m;
        } else {
            for k in b..b + run {
                if VirtualProtectEx(process, (wa + k * PAGE) as _, PAGE as usize, PAGE_READWRITE, &mut old) != 0 {
                    done |= 1u64 << k;
                }
            }
        }
    }
    if done != 0 {
        w.fetch_and(!done, Ordering::AcqRel);
    }
    done.count_ones() as u64
}

/// Whether the debuggee's guest page at `addr` is committed and allows the
/// access (a read, or a write when `write`).
fn allows(process: HANDLE, addr: u64, write: bool) -> bool {
    if !(crate::mem::GUEST_BASE..crate::mem::GUEST_END).contains(&addr) {
        return false;
    }
    let mut m: MEMORY_BASIC_INFORMATION = unsafe { std::mem::zeroed() };
    if unsafe { VirtualQueryEx(process, addr as _, &mut m, std::mem::size_of::<MEMORY_BASIC_INFORMATION>()) } == 0 || m.State != MEM_COMMIT {
        return false;
    }
    let p = m.Protect & 0xff;
    if write {
        matches!(p, PAGE_READWRITE | PAGE_EXECUTE_READWRITE)
    } else {
        matches!(p, PAGE_READONLY | PAGE_READWRITE | PAGE_EXECUTE_READ | PAGE_EXECUTE_READWRITE)
    }
}

/// Writes `addr` with the 16 qwords below the stack pointer set to a pattern
/// and the AVX upper state live; the mask of slots changed after the write
/// (bit k: [rsp - 8 (k + 1)]), as `faultbench`'s probe.
#[inline(never)]
unsafe fn probe(addr: *mut u32) -> u64 {
    let mask: u64;
    unsafe {
        core::arch::asm!(
            "vpcmpeqd ymm1, ymm1, ymm1",
            "mov rax, 0x5a5a5a5a5a5a5a5a",
            "mov qword ptr [rsp-8], rax",
            "mov qword ptr [rsp-16], rax",
            "mov qword ptr [rsp-24], rax",
            "mov qword ptr [rsp-32], rax",
            "mov qword ptr [rsp-40], rax",
            "mov qword ptr [rsp-48], rax",
            "mov qword ptr [rsp-56], rax",
            "mov qword ptr [rsp-64], rax",
            "mov qword ptr [rsp-72], rax",
            "mov qword ptr [rsp-80], rax",
            "mov qword ptr [rsp-88], rax",
            "mov qword ptr [rsp-96], rax",
            "mov qword ptr [rsp-104], rax",
            "mov qword ptr [rsp-112], rax",
            "mov qword ptr [rsp-120], rax",
            "mov qword ptr [rsp-128], rax",
            "mov dword ptr [{a}], 1",
            "xor {m}, {m}",
            "cmp qword ptr [rsp-8], rax", "setne cl", "movzx ecx, cl", "or {m}, rcx",
            "cmp qword ptr [rsp-16], rax", "setne cl", "movzx ecx, cl", "shl ecx, 1", "or {m}, rcx",
            "cmp qword ptr [rsp-24], rax", "setne cl", "movzx ecx, cl", "shl ecx, 2", "or {m}, rcx",
            "cmp qword ptr [rsp-32], rax", "setne cl", "movzx ecx, cl", "shl ecx, 3", "or {m}, rcx",
            "cmp qword ptr [rsp-40], rax", "setne cl", "movzx ecx, cl", "shl ecx, 4", "or {m}, rcx",
            "cmp qword ptr [rsp-48], rax", "setne cl", "movzx ecx, cl", "shl ecx, 5", "or {m}, rcx",
            "cmp qword ptr [rsp-56], rax", "setne cl", "movzx ecx, cl", "shl ecx, 6", "or {m}, rcx",
            "cmp qword ptr [rsp-64], rax", "setne cl", "movzx ecx, cl", "shl ecx, 7", "or {m}, rcx",
            "cmp qword ptr [rsp-72], rax", "setne cl", "movzx ecx, cl", "shl ecx, 8", "or {m}, rcx",
            "cmp qword ptr [rsp-80], rax", "setne cl", "movzx ecx, cl", "shl ecx, 9", "or {m}, rcx",
            "cmp qword ptr [rsp-88], rax", "setne cl", "movzx ecx, cl", "shl ecx, 10", "or {m}, rcx",
            "cmp qword ptr [rsp-96], rax", "setne cl", "movzx ecx, cl", "shl ecx, 11", "or {m}, rcx",
            "cmp qword ptr [rsp-104], rax", "setne cl", "movzx ecx, cl", "shl ecx, 12", "or {m}, rcx",
            "cmp qword ptr [rsp-112], rax", "setne cl", "movzx ecx, cl", "shl ecx, 13", "or {m}, rcx",
            "cmp qword ptr [rsp-120], rax", "setne cl", "movzx ecx, cl", "shl ecx, 14", "or {m}, rcx",
            "cmp qword ptr [rsp-128], rax", "setne cl", "movzx ecx, cl", "shl ecx, 15", "or {m}, rcx",
            a = in(reg) addr,
            m = out(reg) mask,
            out("rax") _,
            out("rcx") _,
            out("ymm1") _,
            options(nostack),
        );
    }
    mask
}

/// `DS_HOST_TEST=1` (hosted, at start-up): arms pages of a 1 MiB guest
/// region one at a time and writes each from code that keeps values below
/// its stack pointer; the values, the ring and the pages' state are checked.
pub fn self_test_maybe() {
    if std::env::var("DS_HOST_TEST").ok().as_deref() != Some("1") {
        return;
    }
    match self_test() {
        Ok(s) => crate::info!("host test passed: {s}"),
        Err(e) => crate::error!("host test failed: {e}"),
    }
}

fn self_test() -> Result<String, String> {
    if !on() {
        return Err("not hosted (run with --host)".into());
    }
    const N: u64 = 4000;
    let base = crate::mem::MEM
        .lock()
        .alloc_internal(1 << 20, crate::mem::PROT_CPU_READ | crate::mem::PROT_CPU_WRITE, "host-test")
        .map_err(|e| format!("no guest memory for the test: {e:?}"))?;
    let mut drained = Vec::new();
    drain(&mut drained);
    drained.clear();
    let t = std::time::Instant::now();
    let mut changed = 0u64;
    for i in 0..N {
        let a = base + (i % 256) * PAGE;
        arm(a, PAGE)?;
        if unsafe { probe(a as *mut u32) } != 0 {
            changed += 1;
        }
    }
    let us = t.elapsed().as_secs_f64() * 1e6 / N as f64;
    let ok = drain(&mut drained);
    let wrong: Vec<&u64> = drained.iter().enumerate().filter(|(i, a)| **a != base + (*i as u64 % 256) * PAGE).map(|(_, a)| a).take(3).collect();
    if changed != 0 {
        return Err(format!("the values below the stack pointer changed in {changed} of {N} faults"));
    }
    if drained.len() as u64 != N || !wrong.is_empty() || !ok {
        return Err(format!("{} pages in the ring for {N} faults (wrong: {wrong:x?}, overflow: {})", drained.len(), !ok));
    }
    let tracking = if crate::sections::on() { format!("; {}", tracking_test()?) } else { String::new() };
    let held = held_test()?;
    Ok(format!("{N} write faults on armed guest pages answered by the host, {us:.1} us per arm and fault, the values below the stack pointer intact, every page in the ring in order{tracking}; {held}"))
}

/// Held pages through the host (`hold`, `take_held`, `release`,
/// `resume_waiter`), the test in the service's place: reads of held pages
/// by another thread, a write from code that keeps values below its stack
/// pointer, and two threads reading one page wait, suspended by the host,
/// until the test has released the page and put new values there; they see
/// the new values only, the values below the stack pointer intact, both
/// waiters resumed by one release; a word answer for an armed page leaves a
/// held page of its word held; nothing is passed on to dsport. On a failure
/// the pages are released and waiting threads resumed (threads that never
/// ran on are left).
fn held_test() -> Result<String, String> {
    use std::sync::Arc;
    use std::time::{Duration, Instant};
    const PAGES: u64 = 8;
    const OLD: u32 = 0x0bad_0000;
    const NEW: u32 = 0x600d_0000;
    let s = shared().ok_or("not hosted")?;
    let a = crate::mem::MEM
        .lock()
        .alloc_internal(1 << 20, crate::mem::PROT_CPU_READ | crate::mem::PROT_CPU_WRITE, "host-held-test")
        .map_err(|e| format!("no guest memory for the held test: {e:?}"))?;
    let page = move |k: u64| a + k * PAGE;
    let fill = move |k: u64, v: u32| {
        for w in 0..PAGE / 4 {
            unsafe { std::ptr::write_volatile((page(k) + 4 * w) as *mut u32, v + k as u32) };
        }
    };
    let word = move |k: u64, w: u64| unsafe { std::ptr::read_volatile((page(k) + 4 * w) as *const u32) };
    // (The protection as the system has it: `mem::os_readable` reads armed
    // pages as readable.)
    let noaccess = |p: u64| {
        let mut m: MEMORY_BASIC_INFORMATION = unsafe { std::mem::zeroed() };
        let n = unsafe { VirtualQuery(p as _, &mut m, std::mem::size_of::<MEMORY_BASIC_INFORMATION>()) };
        n != 0 && m.Protect & PAGE_NOACCESS != 0
    };
    for k in 0..PAGES {
        fill(k, OLD);
    }
    let passed = s.passed.load(Ordering::Acquire);
    let mut stray = Vec::new();
    take_held(&mut stray);
    set_held_served(true);
    let held_now = hold(a, PAGES * PAGE);
    set_held_served(false);
    held_now?;
    let waits = parking_lot::Mutex::new(Vec::<u64>::new());
    // Requests until `want` (5 s at most).
    let collect = |want: usize| -> Result<Vec<HeldRequest>, String> {
        let t = Instant::now();
        let mut got = Vec::new();
        while got.len() < want {
            if t.elapsed() > Duration::from_secs(5) {
                return Err(format!("{} of {want} requests within 5 s ({got:x?})", got.len()));
            }
            wait_held(5);
            take_held(&mut got);
        }
        Ok(got)
    };
    // The service's part: the page released with new values, the thread resumed.
    let serve = |reqs: &[HeldRequest]| -> Result<(), String> {
        for r in reqs {
            if !(a..page(PAGES)).contains(&r.page) {
                return Err(format!("a request for {:#x}, outside the test's pages", r.page));
            }
            if held(r.page) {
                release(r.page, PAGE)?;
                fill((r.page - a) / PAGE, NEW);
            }
            waits.lock().push(resume_waiter(r)?);
        }
        Ok(())
    };
    let run = || -> Result<String, String> {
        if let Some(k) = (0..PAGES).find(|&k| !held(page(k)) || !noaccess(page(k))) {
            return Err(format!("page {k} not held and no-access after `hold`"));
        }
        // Reads of pages 0-3 by one thread, page by page.
        let seen: Arc<[AtomicU32; 5]> = Arc::new(Default::default());
        let r2 = seen.clone();
        let reader = std::thread::Builder::new()
            .name("held-test-reader".into())
            .spawn(move || {
                r2[0].store(unsafe { GetCurrentThreadId() }, Ordering::Release);
                for k in 0..4u64 {
                    r2[1 + k as usize].store(word(k, 7), Ordering::Release);
                }
            })
            .map_err(|e| e.to_string())?;
        let mut tids = Vec::new();
        for k in 0..4u64 {
            let r = collect(1)?;
            if r.len() != 1 || r[0].page != page(k) || r[0].write || !r[0].suspended {
                return Err(format!("read of page {k}: requests {r:x?}"));
            }
            tids.push(r[0].tid);
            serve(&r)?;
        }
        reader.join().map_err(|_| "the reader panicked")?;
        let tid = seen[0].load(Ordering::Acquire);
        if let Some(k) = (0..4u64).find(|&k| seen[1 + k as usize].load(Ordering::Acquire) != NEW + k as u32) {
            return Err(format!("the reader saw {:#x} in page {k}, not the released {:#x}", seen[1 + k as usize].load(Ordering::Acquire), NEW + k as u32));
        }
        if tids.iter().any(|&t| t != tid) {
            return Err(format!("requests named threads {tids:?}, the reader is {tid}"));
        }
        // A write to page 4 from code that keeps values below its stack pointer.
        let mask = Arc::new(AtomicU64::new(u64::MAX));
        let m2 = mask.clone();
        let writer = std::thread::Builder::new()
            .name("held-test-writer".into())
            .spawn(move || m2.store(unsafe { probe(page(4) as *mut u32) }, Ordering::Release))
            .map_err(|e| e.to_string())?;
        let r = collect(1)?;
        if r.len() != 1 || r[0].page != page(4) || !r[0].write || !r[0].suspended {
            return Err(format!("write of page 4: requests {r:x?}"));
        }
        serve(&r)?;
        writer.join().map_err(|_| "the writer panicked")?;
        if mask.load(Ordering::Acquire) != 0 {
            return Err(format!("the values below the stack pointer changed across the wait (mask {:#x})", mask.load(Ordering::Acquire)));
        }
        if word(4, 0) != 1 || word(4, 1) != NEW + 4 {
            return Err(format!("page 4 holds {:#x} {:#x} after the write, not 1 {:#x}", word(4, 0), word(4, 1), NEW + 4));
        }
        // Two threads reading page 5: both wait, one release resumes both.
        let two: Arc<[AtomicU32; 2]> = Arc::new(Default::default());
        let mut both = Vec::new();
        for i in 0..2 {
            let t2 = two.clone();
            both.push(std::thread::Builder::new().name("held-test-pair".into()).spawn(move || t2[i].store(word(5, 3), Ordering::Release)).map_err(|e| e.to_string())?);
        }
        let r = collect(2)?;
        if r.len() != 2 || r.iter().any(|q| q.page != page(5) || q.write || !q.suspended) || r[0].tid == r[1].tid {
            return Err(format!("two readers of page 5: requests {r:x?}"));
        }
        serve(&r)?;
        for t in both {
            t.join().map_err(|_| "a reader of page 5 panicked")?;
        }
        if two.iter().any(|v| v.load(Ordering::Acquire) != NEW + 5) {
            return Err(format!("the readers of page 5 saw {:#x} and {:#x}, not {:#x}", two[0].load(Ordering::Acquire), two[1].load(Ordering::Acquire), NEW + 5));
        }
        // A word answer (page 7 armed against writes, written) leaves held
        // page 6 of its word held.
        let same_word = (page(6) - crate::mem::GUEST_BASE) / (64 * PAGE) == (page(7) - crate::mem::GUEST_BASE) / (64 * PAGE);
        release(page(7), PAGE)?;
        fill(7, NEW);
        arm(page(7), PAGE)?;
        let had = s.flags.load(Ordering::Acquire) & WORD_MODE != 0;
        if !had {
            s.flags.fetch_or(WORD_MODE, Ordering::AcqRel);
        }
        unsafe { std::ptr::write_volatile(page(7) as *mut u32, 9) };
        if !had {
            s.flags.fetch_and(!WORD_MODE, Ordering::AcqRel);
        }
        if same_word && (!held(page(6)) || !noaccess(page(6))) {
            return Err("a word answer for an armed page made held page 6 of its word accessible".into());
        }
        let mut faults = Vec::new();
        drain_faults(&mut faults);
        if !faults.iter().any(|f| f.0 == page(7) && f.1) {
            return Err(format!("the write fault on armed page 7 is not in the ring ({faults:x?})"));
        }
        release(page(6), PAGE)?;
        let mut more = Vec::new();
        take_held(&mut more);
        if !more.is_empty() {
            return Err(format!("requests nobody caused: {more:x?}"));
        }
        let p = s.passed.load(Ordering::Acquire) - passed;
        if p != 0 {
            return Err(format!("{p} exceptions passed on to dsport during the held test"));
        }
        let w = waits.lock();
        Ok(format!(
            "held pages: reads by another thread, a write from code that keeps values below its stack pointer and two readers of one page waited, suspended by the host, until released with new values ({} waits from the fault to the resume, {:.0} us on average, {:.0} us at most, the test polling as the service), saw only the new values, the values below the stack pointer intact, one release resumed both readers{}; nothing passed on to dsport",
            w.len(),
            w.iter().sum::<u64>() as f64 / 1000.0 / w.len().max(1) as f64,
            w.iter().copied().max().unwrap_or(0) as f64 / 1000.0,
            if same_word { ", a word answer left a held page of its word held" } else { " (the word answer not tested: pages 6 and 7 lie in different words)" }
        ))
    };
    let r = run();
    if r.is_err() {
        // Every page released, every waiting thread resumed.
        let _ = release(a, PAGES * PAGE);
        let mut left = Vec::new();
        take_held(&mut left);
        for q in &left {
            let _ = resume_waiter(q);
        }
    }
    r
}

/// Section-backed memory's write tracking through the host
/// (`gpu::resident::take_written`): a take reports new pages and arms them
/// through the host; writes (one from code that keeps values below its
/// stack pointer) are answered by the host, the values intact, nothing
/// passed on to dsport, and each written page is reported by the next take
/// only.
fn tracking_test() -> Result<String, String> {
    const PAGES: u64 = 16;
    let s = shared().ok_or("not hosted")?;
    let a = crate::mem::MEM
        .lock()
        .alloc_internal(1 << 20, crate::mem::PROT_CPU_READ | crate::mem::PROT_CPU_WRITE, "host-track-test")
        .map_err(|e| format!("no guest memory for the tracking test: {e:?}"))?;
    let passed = s.passed.load(Ordering::Acquire);
    let mut out = Vec::new();
    let mut take = |what: &str, want: &[u64]| -> Result<(), String> {
        out.clear();
        crate::gpu::resident::take_written(a, PAGES * PAGE, &mut out).ok_or("a take failed")?;
        let want: Vec<u64> = want.iter().map(|k| a + k * PAGE).collect();
        if out != want {
            return Err(format!("{what}: a take reported {:?} (pages), not {:?}", out.iter().map(|p| (p - a) / PAGE).collect::<Vec<_>>(), want.iter().map(|p| (p - a) / PAGE).collect::<Vec<_>>()));
        }
        Ok(())
    };
    take("new pages", &(0..PAGES).collect::<Vec<_>>())?;
    take("nothing written", &[])?;
    if !all_armed(a, PAGES * PAGE) {
        return Err("taken pages not armed through the host".into());
    }
    if unsafe { probe((a + 3 * PAGE) as *mut u32) } != 0 {
        return Err("a write fault on a taken page changed the values below the stack pointer".into());
    }
    take("page 3 written", &[3])?;
    take("nothing written since", &[])?;
    unsafe {
        std::ptr::write_volatile((a + 5 * PAGE) as *mut u32, 7);
        std::ptr::write_volatile((a + 9 * PAGE + 100) as *mut u32, 7);
    }
    take("pages 5 and 9 written", &[5, 9])?;
    take("nothing written since", &[])?;
    disarm(a, PAGES * PAGE);
    let more = s.passed.load(Ordering::Acquire) - passed;
    if more != 0 {
        return Err(format!("{more} exceptions passed on to dsport during the tracking test"));
    }
    Ok("write tracking on sections through the host: writes to taken pages answered there (the values below the stack pointer intact, nothing passed on) and reported by the next take only".into())
}

/// Arguments of a mode that runs no game: never hosted by default. (The launcher's shader
/// preparation, `--precompile-all`, is hosted as a game session is since 2026-11-19: on another
/// player's PC the driver refused 259 of 262 pipeline binaries the unhosted preparation had made
/// in the first game session, and took those game sessions made; the device and its settings
/// were the same line for line.)
const NO_GAME: [&str; 4] = ["--pipeline-test", "--dispatch-bench", "--compile-bench", "--record-bench"];

/// Whether this process hosts dsport instead of running it: with `--host`,
/// and by default when front end 2 is wanted (it arms quiet guest pages, and
/// a write fault answered inside the process would overwrite the 128 bytes
/// below the faulting thread's stack pointer), except with `--no-host`, in
/// the debuggee (`DS_HOSTED`), under another debugger (`dsdbg`) or in a mode
/// that runs no game.
pub fn wants_host() -> bool {
    let args: Vec<String> = std::env::args().skip(1).collect();
    if args.iter().any(|a| a == "--host") {
        return true;
    }
    if std::env::var_os("DS_HOSTED").is_some() || args.iter().any(|a| a == "--no-host" || NO_GAME.contains(&a.as_str())) {
        return false;
    }
    // SAFETY: no arguments.
    if unsafe { IsDebuggerPresent() } != 0 {
        return false;
    }
    crate::gpu::fe2::wanted()
}

/// `a` as one argument of a Windows command line, the way the C runtime's
/// parser reads it back: quoted when it holds blanks or quotes (or is empty),
/// the backslashes before a quote, its own or the closing one, doubled.
fn quote_arg(a: &str) -> String {
    if !a.is_empty() && !a.contains([' ', '\t', '\n', '\x0b', '"']) {
        return a.to_string();
    }
    let mut s = String::with_capacity(a.len() + 2);
    s.push('"');
    let mut slashes = 0usize;
    for c in a.chars() {
        if c == '\\' {
            slashes += 1;
            continue;
        }
        let n = if c == '"' { 2 * slashes + 1 } else { slashes };
        s.extend(std::iter::repeat('\\').take(n));
        s.push(c);
        slashes = 0;
    }
    s.extend(std::iter::repeat('\\').take(2 * slashes));
    s.push('"');
    s
}

#[cfg(test)]
mod tests {
    use super::quote_arg;

    #[test]
    fn arguments_quoted_as_the_runtime_reads_them() {
        assert_eq!(quote_arg("--seconds"), "--seconds");
        assert_eq!(quote_arg(r"local-data\control.txt"), r"local-data\control.txt");
        assert_eq!(quote_arg(""), r#""""#);
        assert_eq!(quote_arg(r"C:\Games\My Game\PPSA01341-app0"), r#""C:\Games\My Game\PPSA01341-app0""#);
        // A trailing backslash would escape the closing quote.
        assert_eq!(quote_arg(r"D:\My Games\"), r#""D:\My Games\\""#);
        assert_eq!(quote_arg(r#"say "hi""#), r#""say \"hi\"""#);
        assert_eq!(quote_arg(r#"a\"b c"#), r#""a\\\"b c""#);
    }
}
