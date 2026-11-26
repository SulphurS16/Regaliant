use std::sync::OnceLock;

/// Text hooks (`DS_TEXT_HOOK=<eboot offset>:<address expr>[:<length expr>]`,
/// several separated by `;`): inline hooks (`patch::Patcher::hook`) that log
/// each distinct UTF-8 string an expression points to, the first time it is
/// seen (the `textreset` control command forgets them). Pointed at the
/// lookup of a title's text database, they show the strings its UI fetches:
/// a view of the screens without rendering. Expressions as for
/// `DS_BREAK_PEEK`, over the registers at the hook.
static TEXT_HOOKS: OnceLock<Vec<(Expr, Option<Expr>)>> = OnceLock::new();
static TEXT_SEEN: parking_lot::RwLock<Option<std::collections::HashSet<u64>>> = parking_lot::RwLock::new(None);

/// A peek expression compiled for hooks that run millions of times: base
/// register (index in encoding order) or constant, then `+`/`-`/`*` steps.
struct Expr {
    base: Result<usize, u64>,
    ops: Vec<ExprOp>,
}

enum ExprOp {
    Add(u64),
    Sub(u64),
    Deref,
}

const REG_NAMES: [&str; 16] = ["rax", "rcx", "rdx", "rbx", "rsp", "rbp", "rsi", "rdi", "r8", "r9", "r10", "r11", "r12", "r13", "r14", "r15"];

fn compile_expr(e: &str) -> Option<Expr> {
    let base_end = e.find(['+', '-', '*']).unwrap_or(e.len());
    let b = &e[..base_end];
    let base = match REG_NAMES.iter().position(|n| *n == b) {
        Some(i) => Ok(i),
        None => Err(u64::from_str_radix(b.trim_start_matches("0x"), 16).ok()?),
    };
    let mut ops = Vec::new();
    let mut rest = &e[base_end..];
    while !rest.is_empty() {
        if let Some(r) = rest.strip_prefix('*') {
            ops.push(ExprOp::Deref);
            rest = r;
        } else {
            let add = rest.starts_with('+');
            let r = &rest[1..];
            let end = r.find(['+', '-', '*']).unwrap_or(r.len());
            let v = u64::from_str_radix(&r[..end], 16).ok()?;
            ops.push(if add { ExprOp::Add(v) } else { ExprOp::Sub(v) });
            rest = &r[end..];
        }
    }
    Some(Expr { base, ops })
}

thread_local! {
    /// The last memory region found readable (VirtualQuery is a system call).
    static READABLE: std::cell::Cell<(u64, u64)> = const { std::cell::Cell::new((0, 0)) };
}

fn readable_cached(a: u64, n: u64) -> bool {
    use crate::platform::os::Win32::System::Memory::{MEMORY_BASIC_INFORMATION, MEM_COMMIT, PAGE_GUARD, PAGE_NOACCESS, VirtualQuery};
    let (s, e) = READABLE.with(|r| r.get());
    if a >= s && a + n <= e {
        return true;
    }
    let mut mbi: MEMORY_BASIC_INFORMATION = unsafe { std::mem::zeroed() };
    if a < 0x1000 || unsafe { VirtualQuery(a as *const _, &mut mbi, std::mem::size_of::<MEMORY_BASIC_INFORMATION>()) } == 0 {
        return false;
    }
    let (rs, re) = (mbi.BaseAddress as u64, mbi.BaseAddress as u64 + mbi.RegionSize as u64);
    let ok = mbi.State == MEM_COMMIT && mbi.Protect & (PAGE_NOACCESS | PAGE_GUARD) == 0 && mbi.Protect != 0 && a + n <= re;
    if ok {
        READABLE.with(|r| r.set((rs, re)));
    }
    ok
}

fn eval_expr(e: &Expr, regs: &[u64]) -> Option<u64> {
    let mut v = match e.base {
        Ok(i) => regs[i],
        Err(c) => c,
    };
    for op in &e.ops {
        match op {
            ExprOp::Add(x) => v = v.wrapping_add(*x),
            ExprOp::Sub(x) => v = v.wrapping_sub(*x),
            ExprOp::Deref => {
                if !readable_cached(v & !7, 8) {
                    return None;
                }
                v = unsafe { *((v & !7) as *const u64) };
            }
        }
    }
    Some(v)
}

/// Installs the `DS_TEXT_HOOK` hooks into a module being patched.
pub fn install_text_hooks(p: &mut crate::patch::Patcher, ranges: &[(u64, u64)]) {
    let Ok(list) = std::env::var("DS_TEXT_HOOK") else { return };
    let mut sites = Vec::new();
    let hooks = TEXT_HOOKS.get_or_init(|| {
        let mut v = Vec::new();
        for spec in list.split(';').map(str::trim).filter(|s| !s.is_empty()) {
            let mut it = spec.split(':');
            let off = it.next().and_then(|o| u64::from_str_radix(o.trim_start_matches("0x"), 16).ok());
            let ptr = it.next().and_then(compile_expr);
            let len = it.next().map(compile_expr);
            match (off, ptr, len) {
                (Some(off), Some(ptr), None) => {
                    sites.push(off);
                    v.push((ptr, None));
                }
                (Some(off), Some(ptr), Some(Some(len))) => {
                    sites.push(off);
                    v.push((ptr, Some(len)));
                }
                _ => crate::warn!("DS_TEXT_HOOK '{spec}': want <offset>:<address expr>[:<length expr>]"),
            }
        }
        v
    });
    for (i, off) in sites.into_iter().enumerate().take(hooks.len()) {
        match p.hook(off, ranges, text_hook_entry as *const () as usize as u64, i as u32) {
            Ok(t) => crate::info!("text hook at eboot+{off:#x} (trampoline {t:#x})"),
            Err(e) => crate::warn!("text hook at eboot+{off:#x}: {e}"),
        }
    }
}

/// Forgets the strings logged so far (`textreset`).
pub fn text_reset() {
    if let Some(s) = TEXT_SEEN.write().as_mut() {
        s.clear();
    }
}

extern "sysv64" fn text_hook_entry(regs: *const u64, index: u32) {
    let Some((ptr, len)) = TEXT_HOOKS.get().and_then(|h| h.get(index as usize)) else { return };
    let r = unsafe { std::slice::from_raw_parts(regs, 16) };
    let Some(a) = eval_expr(ptr, r) else { return };
    if TEXT_SEEN.read().as_ref().is_some_and(|s| s.contains(&a)) {
        return;
    }
    if !TEXT_SEEN.write().get_or_insert_with(Default::default).insert(a) {
        return;
    }
    let mut s = unsafe { crate::exception::peek_string(a, false) };
    if let Some(n) = len.as_ref().and_then(|l| eval_expr(l, r)) {
        s = s.chars().take(n as usize).collect();
    }
    crate::log::write(crate::log::Level::Info, format_args!("text: {s}"));
}

