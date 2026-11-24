//! libSceFiber: user-mode fibers on title-provided stacks.
//!
//! A fiber runs on the host thread that runs or switches to it, sharing that
//! thread's guest TLS (as on the console). Context switches save the System V
//! callee-saved registers, MXCSR and the x87 control word on the outgoing
//! stack and swap stack pointers.
//!
//! A fiber may be resumed on another thread as soon as it is marked
//! suspended, so the outgoing fiber's stack pointer is stored before its
//! busy word (in the SceFiber object) is cleared, and a resumer waits for
//! that word (`acquire`).
//!
//! The title sizes fiber stacks for its own code (16 KiB for most job
//! fibers), so HLE functions and the exception handler do not run on them:
//! called on a fiber's own stack they move to a host stack owned by that
//! fiber (`on_host_stack`, 1 MiB, allocated on first use), and guest
//! callbacks made from there return to the fiber's stack below the caller's
//! frames (`guest_callback`). Both stacks belong to the fiber, so a fiber
//! that switches away inside a callback can resume on any thread. The
//! thread's stack bounds in the TEB (used by `__chkstk` probes) follow every
//! stack change: fiber switches save and restore them with the registers.
//!
//! The lowest 256 bytes of every context hold a pattern: a switch away from
//! a fiber whose pattern changed reports how close its stack came to (or
//! past) the end of its context.
//!
//! Signatures confirmed at the title's call sites: `_sceFiberInitializeImpl(
//! fiber, name, entry, argOnInitialize, addrContext, sizeContext, optParam,
//! sdkVersion)`, `sceFiberSwitch(fiber, argOnRunTo, &argOnRun)`,
//! `sceFiberReturnToThread(argOnReturn, &argOnRun)`; `SceFiber` is 0x100 bytes.

use crate::hle::{Registry, cstr};
use crate::{reg, sysv_on_guest_stack as sysv};
use parking_lot::Mutex;
use std::cell::Cell;
use std::collections::HashMap;

const FIBER_MAGIC: u64 = 0x5245_4249_4653_4400; // "\0DSFIBER"
const ERR_NULL: i32 = 0x8059_0001u32 as i32;
const ERR_ALIGN: i32 = 0x8059_0002u32 as i32;
const ERR_RANGE: i32 = 0x8059_0003u32 as i32;
const ERR_INVALID: i32 = 0x8059_0004u32 as i32;
const ERR_PERMISSION: i32 = 0x8059_0005u32 as i32;
const ERR_STATE: i32 = 0x8059_0006u32 as i32;

std::arch::global_asm!(
    // void ds_fiber_switch(u64* save_rsp [rdi], u64 new_rsp [rsi],
    //                      u64* busy [rdx] (0: none))  [sysv64]
    // The busy word is cleared after the outgoing stack pointer is stored:
    // from then on another thread may resume the outgoing context (this
    // thread no longer touches that stack).
    ".globl ds_fiber_switch",
    "ds_fiber_switch:",
    "push rbp",
    "push rbx",
    "push r12",
    "push r13",
    "push r14",
    "push r15",
    "push qword ptr gs:[8]", // TEB StackBase
    "push qword ptr gs:[0x10]", // TEB StackLimit
    "sub rsp, 8",
    "stmxcsr [rsp]",
    "fnstcw [rsp+4]",
    "mov [rdi], rsp",
    "test rdx, rdx",
    "jz 2f",
    "mov qword ptr [rdx], 0",
    "2:",
    "mov rsp, rsi",
    "ldmxcsr [rsp]",
    "fldcw [rsp+4]",
    "add rsp, 8",
    "pop qword ptr gs:[0x10]",
    "pop qword ptr gs:[8]",
    "pop r15",
    "pop r14",
    "pop r13",
    "pop r12",
    "pop rbx",
    "pop rbp",
    "ret",
    // First activation of a fiber: r12 = entry, r13 = argOnInitialize, r14 = argOnRun.
    ".globl ds_fiber_start",
    "ds_fiber_start:",
    "mov rdi, r13",
    "mov rsi, r14",
    "call r12",
    "call {returned}",
    "ud2",
    returned = sym fiber_entry_returned,
);

#[cfg(windows)]
std::arch::global_asm!(
    // void ds_call_on_stack(u64 top [rcx], void (*f)(void*) [rdx], void* arg [r8],
    //                       u64 limit [r9])  [win64]
    // Calls f(arg) with rsp just below `top` and the TEB stack bounds set to
    // [limit, top), then restores both. The frame is described by unwind
    // data (rbp frame), so host unwinding continues past the stack switch.
    ".globl ds_call_on_stack",
    ".seh_proc ds_call_on_stack",
    "ds_call_on_stack:",
    "push rbp",
    ".seh_pushreg rbp",
    "mov rbp, rsp",
    ".seh_setframe rbp, 0",
    ".seh_endprologue",
    "push qword ptr gs:[8]",
    "push qword ptr gs:[0x10]",
    "mov qword ptr gs:[8], rcx",
    "mov qword ptr gs:[0x10], r9",
    "mov rsp, rcx",
    "and rsp, -16",
    "sub rsp, 32",
    "mov rcx, r8",
    "call rdx",
    "mov rcx, [rbp-8]",
    "mov qword ptr gs:[8], rcx",
    "mov rcx, [rbp-16]",
    "mov qword ptr gs:[0x10], rcx",
    "mov rsp, rbp",
    "pop rbp",
    "ret",
    ".seh_endproc",
);

#[cfg(target_os = "linux")]
std::arch::global_asm!(
    ".globl ds_call_on_stack",
    ".type ds_call_on_stack,@function",
    "ds_call_on_stack:",
    ".cfi_startproc",
    "push rbp",
    ".cfi_def_cfa_offset 16",
    ".cfi_offset rbp, -16",
    "mov rbp, rsp",
    ".cfi_def_cfa_register rbp",
    "push qword ptr gs:[8]",
    "push qword ptr gs:[0x10]",
    // SysV: rdi=top, rsi=callback, rdx=arg, rcx=limit. No Win64 shadow space.
    "mov qword ptr gs:[8], rdi",
    "mov qword ptr gs:[0x10], rcx",
    "mov rsp, rdi",
    "and rsp, -16",
    "mov rdi, rdx",
    "call rsi",
    "mov rcx, [rbp-8]",
    "mov qword ptr gs:[8], rcx",
    "mov rcx, [rbp-16]",
    "mov qword ptr gs:[0x10], rcx",
    "mov rsp, rbp",
    "pop rbp",
    ".cfi_def_cfa rsp, 8",
    "ret",
    ".cfi_endproc",
    ".size ds_call_on_stack, .-ds_call_on_stack",
);

unsafe extern "sysv64" {
    fn ds_fiber_switch(save_rsp: *mut u64, new_rsp: u64, busy: *mut u64);
    fn ds_fiber_start();
}

unsafe extern "C" {
    fn ds_call_on_stack(top: u64, f: unsafe extern "C" fn(*mut u8), arg: *mut u8, limit: u64);
}

#[inline(always)]
fn current_rsp() -> u64 {
    let rsp: u64;
    unsafe { std::arch::asm!("mov {}, rsp", out(reg) rsp, options(nomem, nostack, preserves_flags)) };
    rsp
}

/// Runs `f` on the stack ending at `top` (TEB bounds [limit, top)).
unsafe fn call_on_stack<R, F: FnOnce() -> R>(top: u64, limit: u64, f: F) -> R {
    struct Call<F, R> {
        f: Option<F>,
        r: Option<R>,
    }
    unsafe extern "C" fn tramp<R, F: FnOnce() -> R>(p: *mut u8) {
        let c = unsafe { &mut *(p as *mut Call<F, R>) };
        c.r = Some((c.f.take().unwrap())());
    }
    let mut c = Call { f: Some(f), r: None };
    unsafe { ds_call_on_stack(top, tramp::<R, F>, &mut c as *mut Call<F, R> as *mut u8, limit) };
    c.r.unwrap()
}

extern "sysv64" fn fiber_entry_returned() -> ! {
    let f = CURRENT.with(|c| c.get());
    crate::error!("fiber {f:#x} returned from its entry function (not allowed)");
    crate::fatal_exit(0xD520, "fiber entry returned");
}

/// Saved stack pointer of a suspended fiber, kept in the title's SceFiber
/// object (stable address even while other threads create fibers).
const OBJ_RSP: u64 = 0x08;
/// Busy word in the SceFiber object: 1 from the moment a thread takes the
/// fiber to run it until its context is saved again by a switch away.
const OBJ_BUSY: u64 = 0x10;

/// The fiber's host stack (lowest address; 0 until first used).
const OBJ_ALT_BASE: u64 = 0x18;
/// Where the next move to the host stack starts (below the frames of an HLE
/// call whose guest callback is running).
const OBJ_ALT_TOP: u64 = 0x20;
/// The fiber-stack pointer of the HLE call running on the host stack (0:
/// none), below which its guest callbacks run.
const OBJ_GUEST_RSP: u64 = 0x28;
const ALT_SIZE: u64 = 1 << 20;
/// Room left between a frame and the stack its callee moves to.
const CALL_GAP: u64 = 0x400;

fn field(fiber: u64, off: u64) -> *mut u64 {
    (fiber + off) as *mut u64
}

/// When the fiber running on this thread is on its own stack: that fiber.
#[inline]
pub fn on_fiber_stack() -> Option<u64> {
    let fiber = CURRENT.with(|c| c.get());
    if fiber == 0 {
        return None;
    }
    in_fiber_stack(current_rsp()).then_some(fiber)
}

/// Runs `f` (an HLE function or the exception handler, entered on the
/// fiber's own stack) on the fiber's host stack.
pub unsafe fn on_host_stack<R, F: FnOnce() -> R>(fiber: u64, f: F) -> R {
    unsafe {
        let mut base = *field(fiber, OBJ_ALT_BASE);
        if base == 0 {
            base = crate::platform::stack::allocate(ALT_SIZE as usize);
            if base == 0 {
                crate::fatal_exit(0xD521, "fiber host stack allocation failed");
            }
            *field(fiber, OBJ_ALT_BASE) = base;
            *field(fiber, OBJ_ALT_TOP) = base + ALT_SIZE;
        }
        let prev = *field(fiber, OBJ_GUEST_RSP);
        *field(fiber, OBJ_GUEST_RSP) = current_rsp();
        let r = call_on_stack(*field(fiber, OBJ_ALT_TOP), base, f);
        *field(fiber, OBJ_GUEST_RSP) = prev;
        r
    }
}

/// The guest stack words above the current HLE call (`n` qwords from the
/// fiber-stack pointer recorded when it moved to the host stack), when the
/// call came from a fiber (diagnostics: saved registers of guest frames).
pub fn guest_stack_words(n: usize) -> Option<Vec<u64>> {
    let fiber = CURRENT.with(|c| c.get());
    if fiber == 0 {
        return None;
    }
    let g = unsafe { *field(fiber, OBJ_GUEST_RSP) };
    if g == 0 || !in_fiber_stack(g) {
        return None;
    }
    let hi = STACK_HI.with(|c| c.get());
    let n = n.min(((hi - g) / 8) as usize);
    Some((0..n).map(|i| unsafe { *((g + 8 * i as u64) as *const u64) }).collect())
}

/// Runs a guest function called by HLE code. On a fiber's host stack it
/// runs on the fiber's own stack instead, below the frames of the HLE call
/// (the title's code expects its own stacks), and HLE calls it makes move to
/// the host stack below the frames already there.
pub unsafe fn guest_callback<R, F: FnOnce() -> R>(f: F) -> R {
    unsafe {
        let fiber = CURRENT.with(|c| c.get());
        if fiber != 0 {
            let base = *field(fiber, OBJ_ALT_BASE);
            let guest = *field(fiber, OBJ_GUEST_RSP);
            let rsp = current_rsp();
            if base != 0 && guest != 0 && (base..base + ALT_SIZE).contains(&rsp) {
                let top = *field(fiber, OBJ_ALT_TOP);
                *field(fiber, OBJ_ALT_TOP) = (rsp - CALL_GAP) & !15;
                *field(fiber, OBJ_GUEST_RSP) = 0;
                let r = call_on_stack((guest - CALL_GAP) & !15, STACK_LO.with(|c| c.get()), f);
                *field(fiber, OBJ_GUEST_RSP) = guest;
                *field(fiber, OBJ_ALT_TOP) = top;
                return r;
            }
        }
        f()
    }
}

fn busy(fiber: u64) -> &'static std::sync::atomic::AtomicU64 {
    unsafe { &*((fiber + OBJ_BUSY) as *const std::sync::atomic::AtomicU64) }
}

/// Claims a fiber the bookkeeping just marked running: waits until its last
/// switch away has stored its stack pointer, then returns that pointer.
fn claim(fiber: u64) -> u64 {
    use std::sync::atomic::Ordering;
    let b = busy(fiber);
    let mut spins = 0u64;
    while b.compare_exchange_weak(0, 1, Ordering::Acquire, Ordering::Relaxed).is_err() {
        std::hint::spin_loop();
        spins += 1;
        if spins % (1 << 24) == 0 {
            crate::warn!("fiber {fiber:#x}: still waiting for its context ({spins} spins)");
        }
    }
    unsafe { *((fiber + OBJ_RSP) as *const u64) }
}

/// Pattern in the lowest bytes of each context (contexts of at least
/// GUARD_MIN_CONTEXT bytes).
const GUARD_WORD: u64 = 0x4453_4755_4152_4421; // "!DRAUGSD"
const GUARD_WORDS: u64 = 32;
const GUARD_MIN_CONTEXT: u64 = 0x2000;

/// Damage to a context's guard pattern: the highest changed word's offset
/// from the context start and the number of changed words, if any.
fn guard_damage(f: &Fiber) -> Option<(u64, u64)> {
    if f.ctx_size < GUARD_MIN_CONTEXT {
        return None;
    }
    let words = (0..GUARD_WORDS).filter(|i| unsafe { *((f.ctx + i * 8) as *const u64) } != GUARD_WORD);
    let (mut hi, mut n) = (0, 0);
    for i in words {
        hi = i * 8 + 8;
        n += 1;
    }
    (n > 0).then_some((hi, n))
}

/// Checks the guard of a fiber being switched away from (bookkeeping lock
/// held); returns the report to log once per fiber.
fn guard_report(fiber: u64, f: &mut Fiber) -> Option<String> {
    if f.guard_reported {
        return None;
    }
    let (hi, n) = guard_damage(f)?;
    f.guard_reported = true;
    Some(format!(
        "fiber {fiber:#x} '{}' (context {:#x}+{:#x}): its stack reached the lowest {hi} bytes of the context ({n} of {GUARD_WORDS} guard words changed{})",
        f.name, f.ctx, f.ctx_size, if n == GUARD_WORDS { "; it may have run past the context" } else { "" }
    ))
}

#[derive(Clone)]
struct Fiber {
    name: String,
    entry: u64,
    arg_init: u64,
    ctx: u64,
    ctx_size: u64,
    started: bool,
    running: bool,
    /// Where the value passed by the next Run/Switch to this fiber goes.
    arg_on_run_ptr: u64,
    guard_reported: bool,
}

static FIBERS: Mutex<Option<HashMap<u64, Fiber>>> = Mutex::new(None);

thread_local! {
    /// Fiber currently running on this host thread (0 = the thread itself).
    static CURRENT: Cell<u64> = const { Cell::new(0) };
    /// Thread stack pointer saved by sceFiberRun while its fibers run.
    static RUN_RSP: Cell<u64> = const { Cell::new(0) };
    /// sceFiberRun's argOnReturn pointer.
    static RUN_RESULT_PTR: Cell<u64> = const { Cell::new(0) };
    /// Bounds of the running fiber's own stack (0: the thread's stack).
    static STACK_LO: Cell<u64> = const { Cell::new(0) };
    static STACK_HI: Cell<u64> = const { Cell::new(0) };
}

fn set_stack(lo: u64, hi: u64) {
    STACK_LO.with(|c| c.set(lo));
    STACK_HI.with(|c| c.set(hi));
}

thread_local! {
    /// The thread's own TEB stack bounds (StackBase, StackLimit), recorded by
    /// sceFiberRun.
    static THREAD_BOUNDS: Cell<(u64, u64)> = const { Cell::new((0, 0)) };
}

fn teb_bounds() -> (u64, u64) {
    let (base, limit): (u64, u64);
    unsafe {
        std::arch::asm!("mov {}, qword ptr gs:[8]", "mov {}, qword ptr gs:[0x10]", out(reg) base, out(reg) limit, options(nostack, preserves_flags));
    }
    (base, limit)
}

/// A guest thread leaves whatever fiber it runs (pthread_exit on a fiber):
/// its own TEB stack bounds come back before it returns to its own stack.
pub fn leave_fibers() {
    if CURRENT.with(|c| c.get()) != 0 {
        let (base, limit) = THREAD_BOUNDS.with(|c| c.get());
        if base != 0 {
            unsafe {
                std::arch::asm!("mov qword ptr gs:[8], {}", "mov qword ptr gs:[0x10], {}", in(reg) base, in(reg) limit, options(nostack, preserves_flags));
            }
        }
        CURRENT.with(|c| c.set(0));
        set_stack(0, 0);
    }
}

/// Lowest address of the stack of the fiber running on this thread, 0 when
/// the thread runs on its own stack.
#[inline]
pub fn stack_lo() -> u64 {
    STACK_LO.with(|c| c.get())
}

/// True when `rsp` is on the stack of the fiber running on this thread.
#[inline]
pub fn in_fiber_stack(rsp: u64) -> bool {
    STACK_LO.with(|c| c.get()) <= rsp && rsp < STACK_HI.with(|c| c.get())
}

/// The fiber whose context contains `addr` (for fault reports; does not wait
/// for the bookkeeping lock).
pub fn describe_stack(addr: u64) -> Option<String> {
    let g = FIBERS.try_lock()?;
    let (fiber, f) = g.as_ref()?.iter().find(|(_, f)| (f.ctx..f.ctx + f.ctx_size).contains(&addr))?;
    Some(format!("fiber {fiber:#x} '{}' context {:#x}+{:#x} (at +{:#x})", f.name, f.ctx, f.ctx_size, addr - f.ctx))
}

fn with<R>(f: impl FnOnce(&mut HashMap<u64, Fiber>) -> R) -> R {
    f(FIBERS.lock().get_or_insert_with(HashMap::new))
}

fn valid(fiber: u64) -> bool {
    fiber != 0 && unsafe { *(fiber as *const u64) } == FIBER_MAGIC && with(|m| m.contains_key(&fiber))
}

/// Prepares the initial stack of a fiber: ds_fiber_switch "returns" into
/// ds_fiber_start with the entry and its first argument in callee-saved regs
/// and the TEB stack bounds set to the context. Returns the initial saved
/// stack pointer.
unsafe fn prepare_stack(f: &Fiber) -> u64 {
    let top = (f.ctx + f.ctx_size) & !15;
    let x = top - 8; // return slot; rsp == top (16-aligned) after `ret`
    unsafe {
        let w = |a: u64, v: u64| std::ptr::write(a as *mut u64, v);
        w(x, ds_fiber_start as *const () as u64);
        w(x - 8, 0); // rbp
        w(x - 16, 0); // rbx
        w(x - 24, f.entry); // r12
        w(x - 32, f.arg_init); // r13
        w(x - 40, 0); // r14 = argOnRun (filled on first activation)
        w(x - 48, 0); // r15
        w(x - 56, f.ctx + f.ctx_size); // TEB StackBase
        w(x - 64, f.ctx); // TEB StackLimit
        w(x - 72, 0x1F80 | (0x037F << 32)); // MXCSR | x87 control word
    }
    x - 72
}

/// Delivers `arg` to a fiber about to run: as its entry argument the first time,
/// otherwise through the pointer of the call that suspended it.
unsafe fn deliver(fiber: u64, f: &mut Fiber, arg: u64) {
    unsafe {
        if !f.started {
            let rsp = *((fiber + OBJ_RSP) as *const u64);
            std::ptr::write((rsp + 32) as *mut u64, arg); // r14 slot (x - 40 == rsp + 32)
            f.started = true;
        } else if f.arg_on_run_ptr != 0 {
            std::ptr::write(f.arg_on_run_ptr as *mut u64, arg);
        }
    }
}

sysv! {
    fn sceFiberOptParamInitialize(p: *mut u8) -> i32 {
        if p.is_null() { return ERR_NULL; }
        std::ptr::write_bytes(p, 0, 0x80);
        0
    }
    fn _sceFiberInitializeImpl(fiber: u64, name: *const u8, entry: u64, arg_init: u64, ctx: u64, ctx_size: u64, opt: u64, sdk: u32) -> i32 {
        if fiber == 0 || entry == 0 { return ERR_NULL; }
        if fiber % 8 != 0 || ctx % 16 != 0 { return ERR_ALIGN; }
        if ctx == 0 || ctx_size < 512 { return ERR_RANGE; }
        let f = Fiber {
            name: cstr(name).chars().take(31).collect(),
            entry, arg_init, ctx, ctx_size, started: false, running: false, arg_on_run_ptr: 0, guard_reported: false,
        };
        let rsp = prepare_stack(&f);
        if ctx_size >= GUARD_MIN_CONTEXT {
            for i in 0..GUARD_WORDS {
                *((ctx + i * 8) as *mut u64) = GUARD_WORD;
            }
        }
        std::ptr::write_bytes(fiber as *mut u8, 0, 0x100);
        *(fiber as *mut u64) = FIBER_MAGIC;
        *((fiber + OBJ_RSP) as *mut u64) = rsp;
        crate::info!("fiber {fiber:#x} '{}' entry {} ctx {ctx:#x}+{ctx_size:#x}", f.name, crate::loader::describe_addr(entry));
        with(|m| m.insert(fiber, f));
        0
    }
    fn sceFiberFinalize(fiber: u64) -> i32 {
        if fiber == 0 { return ERR_NULL; }
        if !valid(fiber) { return ERR_INVALID; }
        if with(|m| m.get(&fiber).is_some_and(|f| f.running)) { return ERR_STATE; }
        with(|m| m.remove(&fiber));
        let alt = *field(fiber, OBJ_ALT_BASE);
        if alt != 0 {
            crate::platform::stack::free(alt, ALT_SIZE as usize);
            *field(fiber, OBJ_ALT_BASE) = 0;
        }
        *(fiber as *mut u64) = 0;
        0
    }
    fn sceFiberRun(fiber: u64, arg: u64, ret_out: *mut u64) -> i32 {
        if fiber == 0 { return ERR_NULL; }
        if !valid(fiber) { return ERR_INVALID; }
        if CURRENT.with(|c| c.get()) != 0 { return ERR_PERMISSION; }
        let ok = with(|m| {
            let f = m.get_mut(&fiber).unwrap();
            if f.running { return None; }
            f.running = true;
            unsafe { deliver(fiber, f, arg) };
            Some((f.ctx, f.ctx + f.ctx_size))
        });
        let Some((lo, hi)) = ok else { return ERR_STATE };
        THREAD_BOUNDS.with(|c| c.set(teb_bounds()));
        CURRENT.with(|c| c.set(fiber));
        set_stack(lo, hi);
        RUN_RESULT_PTR.with(|c| c.set(ret_out as u64));
        let rsp = claim(fiber);
        ds_fiber_switch(RUN_RSP.with(|c| c.as_ptr()), rsp, std::ptr::null_mut());
        // Back on the thread: a fiber called sceFiberReturnToThread.
        0
    }
    fn sceFiberSwitch(fiber: u64, arg: u64, arg_on_run: *mut u64) -> i32 {
        if fiber == 0 { return ERR_NULL; }
        if !valid(fiber) { return ERR_INVALID; }
        let cur = CURRENT.with(|c| c.get());
        if cur == 0 { return ERR_PERMISSION; }
        let ok = with(|m| {
            if m.get(&fiber).is_none_or(|f| f.running) { return None; }
            let f = m.get_mut(&fiber).unwrap();
            f.running = true;
            unsafe { deliver(fiber, f, arg) };
            let (lo, hi) = (f.ctx, f.ctx + f.ctx_size);
            let c = m.get_mut(&cur).unwrap();
            c.running = false;
            c.arg_on_run_ptr = arg_on_run as u64;
            Some((lo, hi, guard_report(cur, c)))
        });
        let Some((lo, hi, report)) = ok else { return ERR_STATE };
        if let Some(r) = report { crate::error!("{r}"); }
        CURRENT.with(|c| c.set(fiber));
        set_stack(lo, hi);
        let rsp = claim(fiber);
        ds_fiber_switch((cur + OBJ_RSP) as *mut u64, rsp, (cur + OBJ_BUSY) as *mut u64);
        0
    }
    fn sceFiberReturnToThread(ret: u64, arg_on_run: *mut u64) -> i32 {
        let cur = CURRENT.with(|c| c.get());
        if cur == 0 { return ERR_PERMISSION; }
        let p = RUN_RESULT_PTR.with(|c| c.get());
        if p != 0 { *(p as *mut u64) = ret; }
        let report = with(|m| {
            let c = m.get_mut(&cur).unwrap();
            c.running = false;
            c.arg_on_run_ptr = arg_on_run as u64;
            guard_report(cur, c)
        });
        if let Some(r) = report { crate::error!("{r}"); }
        CURRENT.with(|c| c.set(0));
        set_stack(0, 0);
        ds_fiber_switch((cur + OBJ_RSP) as *mut u64, RUN_RSP.with(|c| c.get()), (cur + OBJ_BUSY) as *mut u64);
        0
    }
    fn sceFiberGetSelf(out: *mut u64) -> i32 {
        if out.is_null() { return ERR_NULL; }
        let cur = CURRENT.with(|c| c.get());
        if cur == 0 { return ERR_PERMISSION; }
        *out = cur;
        0
    }
    fn sceFiberGetInfo(fiber: u64, info: *mut u64) -> i32 {
        if fiber == 0 || info.is_null() { return ERR_NULL; }
        let Some(f) = with(|m| m.get(&fiber).cloned()) else { return ERR_INVALID };
        // SceFiberInfo: size, entry, argOnInitialize, addrContext, sizeContext,
        // name[32], sizeContextMargin.
        *info.add(1) = f.entry;
        *info.add(2) = f.arg_init;
        *info.add(3) = f.ctx;
        *info.add(4) = f.ctx_size;
        crate::hle::write_cstr(info.add(5) as *mut u8, 32, &f.name);
        // Margin (untouched stack) is not tracked: report the whole context.
        *info.add(9) = f.ctx_size;
        0
    }
    fn sceFiberRename(fiber: u64, name: *const u8) -> i32 {
        if fiber == 0 || name.is_null() { return ERR_NULL; }
        let n: String = cstr(name).chars().take(31).collect();
        if with(|m| m.get_mut(&fiber).map(|f| f.name = n)).is_none() { return ERR_INVALID; }
        0
    }
}

pub fn register(r: &mut Registry) {
    reg!(r, "libSceFiber", [
        sceFiberOptParamInitialize, _sceFiberInitializeImpl, sceFiberFinalize, sceFiberRun, sceFiberSwitch,
        sceFiberReturnToThread, sceFiberGetSelf, sceFiberGetInfo, sceFiberRename,
    ]);
}

#[cfg(all(test, target_os = "linux"))]
mod linux_tests {
    use super::*;

    unsafe extern "sysv64" fn entry(initial: u64, arg: u64) {
        assert_eq!(initial, 17);
        assert_eq!(arg, 23);
        let fiber = on_fiber_stack().expect("on guest fiber");
        let answer = unsafe { on_host_stack(fiber, || {
            assert!(on_fiber_stack().is_none());
            let v = unsafe { guest_callback(|| {
                assert!(on_fiber_stack().is_some());
                42
            }) };
            v + 1
        }) };
        let mut resumed = 0;
        assert_eq!(unsafe { sceFiberReturnToThread(answer, &mut resumed) }, 0);
        assert_eq!(resumed, 71);
        unsafe { sceFiberReturnToThread(99, std::ptr::null_mut()) };
        unreachable!("test fiber must remain suspended");
    }

    #[test]
    fn fiber_host_callback_and_resume_keep_stack_bounds() {
        crate::platform::tls::set(0, 0);
        let bounds = teb_bounds();
        let context = crate::platform::vm::Reservation::new_at(0, 64 << 10).unwrap();
        context.protect(0, 64 << 10, crate::platform::vm::READ | crate::platform::vm::WRITE).unwrap();
        let mut object = [0u64; 32];
        let fiber = object.as_mut_ptr() as u64;
        unsafe {
            assert_eq!(_sceFiberInitializeImpl(fiber, c"linux-test".as_ptr().cast(), entry as *const () as u64, 17, context.base() as u64, 64 << 10, 0, 0), 0);
            let mut result = 0;
            assert_eq!(sceFiberRun(fiber, 23, &mut result), 0);
            assert_eq!(result, 43);
            assert_eq!(teb_bounds(), bounds);
            assert_eq!(sceFiberRun(fiber, 71, &mut result), 0);
            assert_eq!(result, 99);
            assert_eq!(teb_bounds(), bounds);
            assert_eq!(sceFiberFinalize(fiber), 0);
        }
    }
}
