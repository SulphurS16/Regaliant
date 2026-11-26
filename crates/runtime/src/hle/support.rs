use super::libs;
use std::sync::atomic::Ordering;

/// Defines HLE functions with the System V ABI and optional entry tracing.
/// Called on a fiber's own (title-sized) stack, the body runs on the
/// fiber's host stack (see `libs::fiber`).
#[macro_export]
macro_rules! sysv {
    ( $( $(#[$m:meta])* fn $name:ident ( $( $arg:ident : $ty:ty ),* $(,)? ) $( -> $ret:ty )? $body:block )* ) => {
        $(
            $(#[$m])*
            #[allow(non_snake_case, unused_variables, unused_unsafe, clippy::missing_safety_doc)]
            pub unsafe extern "sysv64" fn $name( $( $arg : $ty ),* ) $( -> $ret )? {
                if $crate::log::TRACE_HLE.load(std::sync::atomic::Ordering::Relaxed) {
                    $crate::hle::trace_call(stringify!($name), &[ $( $crate::hle::ArgFmt::fmt_arg(&$arg) ),* ]);
                }
                let __body = move || $( -> $ret )? { unsafe { $body } };
                match $crate::hle::libs::fiber::on_fiber_stack() {
                    Some(fiber) => unsafe { $crate::hle::libs::fiber::on_host_stack(fiber, __body) },
                    None => __body(),
                }
            }
        )*
    };
}

/// `sysv!` for functions that must run on the caller's stack (fiber
/// switches save and replace it).
#[macro_export]
macro_rules! sysv_on_guest_stack {
    ( $( $(#[$m:meta])* fn $name:ident ( $( $arg:ident : $ty:ty ),* $(,)? ) $( -> $ret:ty )? $body:block )* ) => {
        $(
            $(#[$m])*
            #[allow(non_snake_case, unused_variables, unused_unsafe, clippy::missing_safety_doc)]
            pub unsafe extern "sysv64" fn $name( $( $arg : $ty ),* ) $( -> $ret )? {
                if $crate::log::TRACE_HLE.load(std::sync::atomic::Ordering::Relaxed) {
                    $crate::hle::trace_call(stringify!($name), &[ $( $crate::hle::ArgFmt::fmt_arg(&$arg) ),* ]);
                }
                unsafe { $body }
            }
        )*
    };
}

/// Registers functions by identifier under a library name.
#[macro_export]
macro_rules! reg {
    ($r:expr, $lib:expr, [ $( $f:ident ),* $(,)? ]) => {
        $( $r.func($lib, stringify!($f), $f as *const () as usize as u64); )*
    };
}

/// Room below which an entry on a fiber stack is reported.
const STACK_WARN: u64 = 4 << 10;

/// Reports entries (the exception handler: the kernel's exception frame is
/// on the faulting stack) with less than STACK_WARN bytes left on a fiber
/// stack, once per name and at every new minimum.
#[inline(always)]
pub fn stack_check(name: &'static str) {
    let lo = libs::fiber::stack_lo();
    if lo != 0 {
        let rsp: u64;
        unsafe { std::arch::asm!("mov {}, rsp", out(reg) rsp, options(nomem, nostack, preserves_flags)) };
        let room = rsp.wrapping_sub(lo);
        if room < STACK_WARN {
            stack_low(name, room, lo);
        }
    }
}

#[cold]
#[inline(never)]
fn stack_low(name: &'static str, room: u64, lo: u64) {
    static MIN: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(u64::MAX);
    static SEEN: parking_lot::Mutex<Option<std::collections::HashSet<&'static str>>> = parking_lot::Mutex::new(None);
    let new_min = MIN.fetch_min(room, Ordering::Relaxed) > room;
    let first = SEEN.lock().get_or_insert_with(Default::default).insert(name);
    if new_min || first {
        crate::warn!("{name} entered on a fiber stack with {room:#x} bytes left (stack bottom {lo:#x}){}", if new_min { " (new minimum)" } else { "" });
    }
}

pub trait ArgFmt {
    fn fmt_arg(&self) -> String;
}
macro_rules! argfmt_int { ($($t:ty),*) => { $( impl ArgFmt for $t { fn fmt_arg(&self) -> String { format!("{:#x}", self) } } )* } }
argfmt_int!(u8, u16, u32, u64, usize, i8, i16, i32, i64, isize);
impl ArgFmt for bool {
    fn fmt_arg(&self) -> String {
        self.to_string()
    }
}
impl ArgFmt for f32 {
    fn fmt_arg(&self) -> String {
        self.to_string()
    }
}
impl ArgFmt for f64 {
    fn fmt_arg(&self) -> String {
        self.to_string()
    }
}
impl<T> ArgFmt for *const T {
    fn fmt_arg(&self) -> String {
        format!("{:p}", *self)
    }
}
impl<T> ArgFmt for *mut T {
    fn fmt_arg(&self) -> String {
        format!("{:p}", *self)
    }
}

pub fn trace_call(name: &str, args: &[String]) {
    crate::log::write(crate::log::Level::Debug, format_args!("hle {}({})", name, args.join(", ")));
}

// ---------------------------------------------------------------------------
// Calling guest code from host code

/// Calls a guest function with up to 6 integer arguments (System V ABI).
pub unsafe fn call_guest(addr: u64, args: &[u64]) -> u64 {
    unsafe {
        type F6 = unsafe extern "sysv64" fn(u64, u64, u64, u64, u64, u64) -> u64;
        let f: F6 = std::mem::transmute(addr as usize);
        let g = |i: usize| args.get(i).copied().unwrap_or(0);
        libs::fiber::guest_callback(|| f(g(0), g(1), g(2), g(3), g(4), g(5)))
    }
}

// ---------------------------------------------------------------------------
// Guest memory helpers

pub unsafe fn cstr<'a>(p: *const u8) -> &'a str {
    if p.is_null() {
        return "";
    }
    unsafe {
        let s = std::ffi::CStr::from_ptr(p as *const std::ffi::c_char);
        s.to_str().unwrap_or("<non-utf8>")
    }
}

pub unsafe fn write_cstr(dst: *mut u8, cap: usize, s: &str) {
    if dst.is_null() || cap == 0 {
        return;
    }
    let n = s.len().min(cap - 1);
    unsafe {
        std::ptr::copy_nonoverlapping(s.as_ptr(), dst, n);
        *dst.add(n) = 0;
    }
}

// ---------------------------------------------------------------------------
// Explicit fixed-result functions (documented offline/no-device behavior).

static STUB_SEEN: parking_lot::Mutex<Option<std::collections::HashSet<&'static str>>> = parking_lot::Mutex::new(None);

pub fn stub_log(name: &'static str, ret: i64) {
    let mut g = STUB_SEEN.lock();
    let set = g.get_or_insert_with(Default::default);
    if set.insert(name) {
        crate::info!("{name} -> {ret:#x} (fixed HLE result)");
    }
}

/// Defines functions that ignore their arguments and return a fixed value.
#[macro_export]
macro_rules! fixed {
    ( $( $name:ident => $ret:expr ),* $(,)? ) => {
        $(
            #[allow(non_snake_case)]
            pub extern "sysv64" fn $name() -> i64 {
                let r = ($ret) as i64;
                $crate::hle::stub_log(stringify!($name), r);
                r
            }
        )*
    };
}
