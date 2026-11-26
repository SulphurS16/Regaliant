//! Vectored exception handling: fatal faults in guest code or in our own image
//! are reported with guest module offsets, registers and a frame-pointer
//! backtrace, then the process terminates. Exceptions in other DLLs are left to
//! their own SEH handlers.

use std::sync::OnceLock;
use std::sync::atomic::{AtomicBool, Ordering};
use windows_sys::Win32::System::Diagnostics::Debug::{AddVectoredExceptionHandler, CONTEXT, EXCEPTION_POINTERS, IMAGE_RUNTIME_FUNCTION_ENTRY};

const EXCEPTION_CONTINUE_SEARCH: i32 = 0;

static HOST_IMAGE: OnceLock<(u64, u64)> = OnceLock::new();
static IN_HANDLER: AtomicBool = AtomicBool::new(false);
/// Host thread id of the thread producing the fatal report.
static REPORTER: std::sync::atomic::AtomicU32 = std::sync::atomic::AtomicU32::new(0);

/// Samples a running thread: suspends it, reads its context and unwinds it
/// (see `unwind`), then resumes it and symbolizes the frames. Nothing is
/// allocated while the thread is suspended (it may hold the heap lock).
/// Used for hang diagnosis (time limit, control command `stacks`).
pub fn sample_thread(host_tid: u32, depth: usize) -> Option<String> {
    let (host, guest) = sample_frames(host_tid, depth)?;
    let mut s = String::new();
    if !host.is_empty() {
        let names: Vec<String> = host.iter().map(|&a| host_symbol(a)).collect();
        s.push_str(&format!("host: {}\n        ", names.join(" <- ")));
    }
    let g: Vec<String> = guest.iter().map(|&a| describe(a)).collect();
    s.push_str(&format!("guest: {}", if g.is_empty() { "(no guest frames)".to_string() } else { g.join(" <- ") }));
    Some(s)
}

/// The host return addresses (innermost first, at most 24) and the first
/// `depth` guest frames of a running thread (`sample_thread`).
pub fn sample_frames(host_tid: u32, depth: usize) -> Option<(Vec<u64>, Vec<u64>)> {
    use windows_sys::Win32::System::Diagnostics::Debug::GetThreadContext;
    use windows_sys::Win32::System::Threading::{OpenThread, ResumeThread, SuspendThread, THREAD_GET_CONTEXT, THREAD_QUERY_INFORMATION, THREAD_SUSPEND_RESUME};
    const CONTEXT_FULL_AMD64: u32 = 0x0010_000B;
    if host_tid == 0 || host_tid == unsafe { windows_sys::Win32::System::Threading::GetCurrentThreadId() } {
        return None;
    }
    let mut host = [0u64; 24];
    let mut guest = [0u64; 32];
    let (mut nh, mut ng) = (0usize, 0usize);
    let mut ok = false;
    if UNWIND_MISSED.load(std::sync::atomic::Ordering::Relaxed) {
        refresh_unwind_tables();
    }
    unsafe {
        let h = OpenThread(THREAD_SUSPEND_RESUME | THREAD_GET_CONTEXT | THREAD_QUERY_INFORMATION, 0, host_tid);
        if h.is_null() {
            return None;
        }
        if SuspendThread(h) != u32::MAX {
            let mut c: AlignedContext = std::mem::zeroed();
            c.0.ContextFlags = CONTEXT_FULL_AMD64;
            if GetThreadContext(h, &mut c.0) != 0 {
                (nh, ng) = unwind_with(&mut c, &mut host, &mut guest[..depth.min(32)], snapshot_function);
                ok = true;
            }
            ResumeThread(h);
        }
        windows_sys::Win32::Foundation::CloseHandle(h);
    }
    if !ok {
        return None;
    }
    Some((host[..nh].to_vec(), guest[..ng].to_vec()))
}

/// A host code address as its function: "name" in our image, "module!name"
/// in other modules ("module" alone where no symbol is near); no offsets.
/// Cached per address (profiling).
pub fn frame_label(addr: u64) -> String {
    static CACHE: parking_lot::Mutex<Option<std::collections::HashMap<u64, String>>> = parking_lot::Mutex::new(None);
    if let Some(s) = CACHE.lock().as_ref().and_then(|m| m.get(&addr)) {
        return s.clone();
    }
    let sym = host_symbol(addr);
    let (b, e) = host_image();
    let label = if addr >= b && addr < e {
        sym.split("+0x").next().unwrap_or(&sym).to_string()
    } else {
        let module = host_module(addr);
        // Symbols of stripped modules are their nearest export, often far away.
        let near = sym.rsplit_once("+0x").and_then(|(_, d)| u64::from_str_radix(d, 16).ok()).is_some_and(|d| d < 0x4000);
        match (near, sym.starts_with("0x")) {
            (true, false) => format!("{module}!{}", sym.split("+0x").next().unwrap_or(&sym)),
            _ => module,
        }
    };
    CACHE.lock().get_or_insert_with(Default::default).insert(addr, label.clone());
    label
}

/// The module containing `addr` and the offset into it (another module's
/// address in a fatal report: a driver's own frames).
fn module_offset(addr: u64) -> Option<(String, u64)> {
    use windows_sys::Win32::System::LibraryLoader::{GET_MODULE_HANDLE_EX_FLAG_FROM_ADDRESS, GET_MODULE_HANDLE_EX_FLAG_UNCHANGED_REFCOUNT, GetModuleHandleExW};
    unsafe {
        let mut m = std::ptr::null_mut();
        if GetModuleHandleExW(GET_MODULE_HANDLE_EX_FLAG_FROM_ADDRESS | GET_MODULE_HANDLE_EX_FLAG_UNCHANGED_REFCOUNT, addr as *const u16, &mut m) == 0 {
            return None;
        }
        Some((host_module(addr), addr - m as u64))
    }
}

/// Another module's address by the export it lies in: "module!Export" within 0x40 bytes of an
/// export's start (the system call stubs are 0x20 bytes, so a thread in a system call is named by
/// it), else "module+offset" (the offset's 4 KiB page). Cached per address (the hitch flight recorder's samples).
pub fn export_label(addr: u64) -> String {
    use windows_sys::Win32::System::LibraryLoader::{GET_MODULE_HANDLE_EX_FLAG_FROM_ADDRESS, GET_MODULE_HANDLE_EX_FLAG_UNCHANGED_REFCOUNT, GetModuleHandleExW};
    type Exports = std::collections::HashMap<u64, std::sync::Arc<Vec<(u32, String)>>>;
    static CACHE: parking_lot::Mutex<Option<(std::collections::HashMap<u64, String>, Exports)>> = parking_lot::Mutex::new(None);
    if let Some(s) = CACHE.lock().as_ref().and_then(|c| c.0.get(&addr)) {
        return s.clone();
    }
    let mut m = std::ptr::null_mut();
    if unsafe { GetModuleHandleExW(GET_MODULE_HANDLE_EX_FLAG_FROM_ADDRESS | GET_MODULE_HANDLE_EX_FLAG_UNCHANGED_REFCOUNT, addr as *const u16, &mut m) } == 0 {
        return format!("{addr:#x}");
    }
    let base = m as u64;
    let module = host_module(addr);
    let exports = {
        let mut g = CACHE.lock();
        let c = g.get_or_insert_with(Default::default);
        c.1.entry(base).or_insert_with(|| std::sync::Arc::new(unsafe { export_table(base) })).clone()
    };
    let rva = (addr - base) as u32;
    let i = exports.partition_point(|e| e.0 <= rva);
    let label = match i.checked_sub(1).map(|i| &exports[i]) {
        Some((at, name)) if rva - at < 0x40 => format!("{module}!{name}"),
        // (Inside a function of no export: by its 4 KiB page, so samples group.)
        _ => format!("{module}+{:#x}", rva & !0xFFF),
    };
    CACHE.lock().get_or_insert_with(Default::default).0.insert(addr, label.clone());
    label
}

/// A loaded module's named exports by RVA (code and data alike; forwarders left out).
unsafe fn export_table(base: u64) -> Vec<(u32, String)> {
    unsafe {
        let nt = base + *((base + 0x3C) as *const u32) as u64;
        // (IMAGE_OPTIONAL_HEADER64's data directories start at +0x70; the export one first.)
        let dir_rva = *((nt + 0x18 + 0x70) as *const u32) as u64;
        let dir_size = *((nt + 0x18 + 0x74) as *const u32) as u64;
        if dir_rva == 0 {
            return Vec::new();
        }
        let d = base + dir_rva;
        let names = *((d + 0x18) as *const u32) as usize;
        let funcs = base + *((d + 0x1C) as *const u32) as u64;
        let name_rvas = base + *((d + 0x20) as *const u32) as u64;
        let ords = base + *((d + 0x24) as *const u32) as u64;
        let mut v = Vec::with_capacity(names);
        for k in 0..names {
            let ord = *((ords + 2 * k as u64) as *const u16) as u64;
            let f = *((funcs + 4 * ord) as *const u32);
            if (f as u64) >= dir_rva && (f as u64) < dir_rva + dir_size {
                continue;
            }
            let np = (base + *((name_rvas + 4 * k as u64) as *const u32) as u64) as *const std::ffi::c_char;
            v.push((f, std::ffi::CStr::from_ptr(np).to_string_lossy().into_owned()));
        }
        v.sort_unstable();
        v
    }
}

/// Whether module `name` (holding `addr`) is NVIDIA's: the driver's (nv*), Streamline's (sl.*),
/// or one under an NVIDIA folder (NGX's models and Streamline's plugin overrides in ProgramData).
fn nvidia_module(name: &str, addr: u64) -> bool {
    let n = name.to_ascii_lowercase();
    n.starts_with("nv") || n.starts_with("sl.") || module_path(addr).is_some_and(|p| p.to_ascii_lowercase().contains("\\nvidia"))
}

/// The full path of the module containing `addr`.
fn module_path(addr: u64) -> Option<String> {
    use windows_sys::Win32::System::LibraryLoader::{GET_MODULE_HANDLE_EX_FLAG_FROM_ADDRESS, GET_MODULE_HANDLE_EX_FLAG_UNCHANGED_REFCOUNT, GetModuleFileNameW, GetModuleHandleExW};
    unsafe {
        let mut m = std::ptr::null_mut();
        if GetModuleHandleExW(GET_MODULE_HANDLE_EX_FLAG_FROM_ADDRESS | GET_MODULE_HANDLE_EX_FLAG_UNCHANGED_REFCOUNT, addr as *const u16, &mut m) == 0 {
            return None;
        }
        let mut buf = [0u16; 520];
        let n = GetModuleFileNameW(m, buf.as_mut_ptr(), buf.len() as u32) as usize;
        Some(String::from_utf16_lossy(&buf[..n.min(buf.len())]))
    }
}

/// The file name of the module containing `addr` ("?" if none).
fn host_module(addr: u64) -> String {
    use windows_sys::Win32::System::LibraryLoader::{GET_MODULE_HANDLE_EX_FLAG_FROM_ADDRESS, GET_MODULE_HANDLE_EX_FLAG_UNCHANGED_REFCOUNT, GetModuleFileNameW, GetModuleHandleExW};
    unsafe {
        let mut m = std::ptr::null_mut();
        if GetModuleHandleExW(GET_MODULE_HANDLE_EX_FLAG_FROM_ADDRESS | GET_MODULE_HANDLE_EX_FLAG_UNCHANGED_REFCOUNT, addr as *const u16, &mut m) == 0 {
            return "?".into();
        }
        let mut buf = [0u16; 260];
        let n = GetModuleFileNameW(m, buf.as_mut_ptr(), buf.len() as u32) as usize;
        let path = String::from_utf16_lossy(&buf[..n.min(buf.len())]);
        path.rsplit('\\').next().unwrap_or(&path).trim_end_matches(".dll").trim_end_matches(".DLL").to_string()
    }
}

#[repr(C, align(16))]
struct AlignedContext(CONTEXT);

/// A loaded module's unwind table: image [base, end) and its
/// RUNTIME_FUNCTION array (exception directory).
#[derive(Clone, Copy)]
struct UnwindTable {
    base: u64,
    end: u64,
    table: u64,
    count: u32,
}

/// The loaded modules' unwind tables, taken while no thread is suspended
/// (`refresh_unwind_tables`). Sampling a suspended thread cannot use
/// RtlLookupFunctionEntry: the loader holds its function-table lock while it
/// maps a module, so a thread sampled inside LoadLibrary left the sampler
/// blocked and the thread suspended (Vulkan initialization stuck "loading
/// the Vulkan loader" / "creating the instance" in profiled runs
/// 20261012-043754 and -051931).
static UNWIND_TABLES: parking_lot::RwLock<Vec<UnwindTable>> = parking_lot::RwLock::new(Vec::new());
/// A lookup found no module (one loaded since): refresh before the next
/// suspension.
static UNWIND_MISSED: std::sync::atomic::AtomicBool = std::sync::atomic::AtomicBool::new(true);

fn refresh_unwind_tables() {
    use windows_sys::Win32::System::ProcessStatus::EnumProcessModules;
    use windows_sys::Win32::System::Threading::GetCurrentProcess;
    let mut mods = vec![std::ptr::null_mut(); 2048];
    let mut needed = 0u32;
    let mut v = Vec::new();
    unsafe {
        if EnumProcessModules(GetCurrentProcess(), mods.as_mut_ptr(), (mods.len() * 8) as u32, &mut needed) == 0 {
            return;
        }
        for &m in &mods[..(needed as usize / 8).min(mods.len())] {
            let base = m as u64;
            if !readable(base) {
                continue;
            }
            // PE32+ headers: the NT headers at e_lfanew; SizeOfImage at +56
            // of the optional header, the exception directory (data
            // directory 3) at +112 + 3 * 8.
            let nt = base + *((base + 0x3c) as *const u32) as u64;
            if !readable(nt) || *(nt as *const u32) != 0x4550 || *((nt + 24) as *const u16) != 0x20b {
                continue;
            }
            let opt = nt + 24;
            let dir = opt + 112 + 3 * 8;
            let (va, size) = (*(dir as *const u32) as u64, *((dir + 4) as *const u32));
            if va != 0 && size >= 12 {
                v.push(UnwindTable { base, end: base + *((opt + 56) as *const u32) as u64, table: base + va, count: size / 12 });
            }
        }
    }
    v.sort_by_key(|t| t.base);
    *UNWIND_TABLES.write() = v;
    UNWIND_MISSED.store(false, std::sync::atomic::Ordering::Relaxed);
}

/// RtlLookupFunctionEntry from the snapshot, without the loader's locks:
/// the entry for `pc` (null: no module or no entry, a leaf function).
unsafe fn snapshot_function(pc: u64, base: &mut u64) -> *const IMAGE_RUNTIME_FUNCTION_ENTRY {
    let Some(g) = UNWIND_TABLES.try_read() else { return std::ptr::null() };
    let i = g.partition_point(|t| t.base <= pc);
    let Some(t) = i.checked_sub(1).map(|i| g[i]).filter(|t| pc < t.end) else {
        UNWIND_MISSED.store(true, std::sync::atomic::Ordering::Relaxed);
        return std::ptr::null();
    };
    if !readable(t.table) {
        return std::ptr::null();
    }
    let rva = (pc - t.base) as u32;
    let entries = unsafe { std::slice::from_raw_parts(t.table as *const [u32; 3], t.count as usize) };
    let k = entries.partition_point(|e| e[1] <= rva);
    match entries.get(k) {
        Some(e) if e[0] <= rva => {
            *base = t.base;
            // An odd unwind-data address points at another entry.
            if e[2] & 1 != 0 { (t.base + (e[2] & !1) as u64) as *const IMAGE_RUNTIME_FUNCTION_ENTRY } else { e as *const [u32; 3] as *const IMAGE_RUNTIME_FUNCTION_ENTRY }
        }
        _ => std::ptr::null(),
    }
}

unsafe fn system_function(pc: u64, base: &mut u64) -> *const IMAGE_RUNTIME_FUNCTION_ENTRY {
    unsafe { windows_sys::Win32::System::Diagnostics::Debug::RtlLookupFunctionEntry(pc, base, std::ptr::null_mut()) }
}

/// Guest frames of the calling thread (for diagnostics in HLE functions):
/// the first is the guest call site of the HLE function.
pub fn current_guest_frames(depth: usize) -> Vec<String> {
    use windows_sys::Win32::System::Diagnostics::Debug::RtlCaptureContext;
    let mut c: AlignedContext = unsafe { std::mem::zeroed() };
    unsafe { RtlCaptureContext(&mut c.0) };
    let (mut host, mut guest) = ([0u64; 64], [0u64; 32]);
    let (_, ng) = unsafe { unwind(&mut c, &mut host, &mut guest[..depth.min(32)]) };
    guest[..ng].iter().map(|&a| describe(a)).collect()
}

/// The calling thread's guest return addresses (at most `depth`), the guest
/// frame-pointer chain checked with `readable` instead of an operating-system
/// query per frame (`mem::note_unmap`: the memory map, which its caller
/// holds). Host frames are unwound from their unwind tables (no system calls
/// for frames that have them).
pub fn current_guest_returns(depth: usize, readable: &dyn Fn(u64) -> bool) -> Vec<u64> {
    use windows_sys::Win32::System::Diagnostics::Debug::{RtlCaptureContext, RtlLookupFunctionEntry, RtlVirtualUnwind};
    let mut c: AlignedContext = unsafe { std::mem::zeroed() };
    unsafe { RtlCaptureContext(&mut c.0) };
    // This thread's own stack (host frames may live there, not in guest
    // memory).
    let (mut lo, mut hi) = (0usize, 0usize);
    unsafe { windows_sys::Win32::System::Threading::GetCurrentThreadStackLimits(&mut lo, &mut hi) };
    let on_stack = |a: u64| (lo as u64) <= a && a + 8 <= hi as u64;
    let mut out = Vec::new();
    for _ in 0..64 {
        let pc = c.0.Rip;
        if pc == 0 || in_guest(pc) {
            break;
        }
        let mut base = 0u64;
        let fe = unsafe { RtlLookupFunctionEntry(pc, &mut base, std::ptr::null_mut()) };
        if fe.is_null() {
            // Code without unwind data (a stub): its return address at [rsp].
            if c.0.Rsp & 7 != 0 || !(on_stack(c.0.Rsp) || readable(c.0.Rsp)) {
                return out;
            }
            c.0.Rip = unsafe { *(c.0.Rsp as *const u64) };
            c.0.Rsp += 8;
        } else {
            let mut handler_data = std::ptr::null_mut();
            let mut establisher = 0u64;
            unsafe { RtlVirtualUnwind(0, base, pc, fe, &mut c.0, &mut handler_data, &mut establisher, std::ptr::null_mut()) };
        }
    }
    if !in_guest(c.0.Rip) || depth == 0 {
        return out;
    }
    out.push(c.0.Rip);
    let mut bp = c.0.Rbp;
    while out.len() < depth && bp & 7 == 0 && readable(bp) && readable(bp + 8) {
        let ret = unsafe { *((bp + 8) as *const u64) };
        if ret < 0x10000 {
            break;
        }
        out.push(ret);
        let next = unsafe { *(bp as *const u64) };
        if next <= bp {
            break;
        }
        bp = next;
    }
    out
}

/// `describe` for callers outside this module (module + offset).
pub fn describe_guest(addr: u64) -> String {
    describe(addr)
}

pub fn readable(a: u64) -> bool {
    use windows_sys::Win32::System::Memory::{MEMORY_BASIC_INFORMATION, MEM_COMMIT, PAGE_GUARD, PAGE_NOACCESS, VirtualQuery};
    if a < 0x10000 {
        return false;
    }
    unsafe {
        let mut mbi: MEMORY_BASIC_INFORMATION = std::mem::zeroed();
        VirtualQuery(a as *const _, &mut mbi, std::mem::size_of::<MEMORY_BASIC_INFORMATION>()) != 0
            && mbi.State == MEM_COMMIT
            && mbi.Protect & (PAGE_NOACCESS | PAGE_GUARD) == 0
            && mbi.Protect != 0
            && a + 16 <= mbi.BaseAddress as u64 + mbi.RegionSize as u64
    }
}

/// Unwinds a context of this process. Host frames are unwound with the
/// registered unwind tables (RtlVirtualUnwind) until rip lands in guest code;
/// that first guest frame is exact (the guest call site of the HLE function),
/// later ones follow the guest frame-pointer chain. Fills the arrays without
/// allocating; returns (host frames, guest frames).
unsafe fn unwind(c: &mut AlignedContext, host: &mut [u64], guest: &mut [u64]) -> (usize, usize) {
    unsafe { unwind_with(c, host, guest, system_function) }
}

/// `unwind` with the function-entry lookup given (`snapshot_function` for
/// suspended threads).
unsafe fn unwind_with(c: &mut AlignedContext, host: &mut [u64], guest: &mut [u64], lookup: unsafe fn(u64, &mut u64) -> *const IMAGE_RUNTIME_FUNCTION_ENTRY) -> (usize, usize) {
    use windows_sys::Win32::System::Diagnostics::Debug::RtlVirtualUnwind;
    let (mut nh, mut ng) = (0usize, 0usize);
    for _ in 0..64 {
        let pc = c.0.Rip;
        if pc == 0 || in_guest(pc) {
            break;
        }
        if nh < host.len() {
            host[nh] = pc;
            nh += 1;
        }
        let mut base = 0u64;
        let fe = unsafe { lookup(pc, &mut base) };
        if fe.is_null() {
            // Leaf function or code without unwind data: return address at [rsp].
            if !readable(c.0.Rsp) {
                return (nh, ng);
            }
            c.0.Rip = unsafe { *(c.0.Rsp as *const u64) };
            c.0.Rsp += 8;
        } else {
            let mut handler_data = std::ptr::null_mut();
            let mut establisher = 0u64;
            unsafe { RtlVirtualUnwind(0, base, pc, fe, &mut c.0, &mut handler_data, &mut establisher, std::ptr::null_mut()) };
        }
    }
    if !in_guest(c.0.Rip) || guest.is_empty() {
        return (nh, ng);
    }
    guest[0] = c.0.Rip;
    ng = 1;
    let mut bp = c.0.Rbp;
    while ng < guest.len() && bp & 7 == 0 && readable(bp) {
        let ret = unsafe { *((bp + 8) as *const u64) };
        if ret < 0x10000 {
            break;
        }
        guest[ng] = ret;
        ng += 1;
        let next = unsafe { *(bp as *const u64) };
        if next <= bp {
            break;
        }
        bp = next;
    }
    (nh, ng)
}

/// Whether unwinding `ctx` reaches our image or guest code within `depth` frames.
unsafe fn called_from_our_code(ctx: &CONTEXT, depth: usize) -> bool {
    use windows_sys::Win32::System::Diagnostics::Debug::{RtlLookupFunctionEntry, RtlVirtualUnwind};
    let (b, e) = host_image();
    let mut c: AlignedContext = unsafe { std::mem::zeroed() };
    c.0 = *ctx;
    for _ in 0..depth {
        let pc = c.0.Rip;
        let mut base = 0u64;
        let fe = unsafe { RtlLookupFunctionEntry(pc, &mut base, std::ptr::null_mut()) };
        if fe.is_null() {
            if !readable(c.0.Rsp) {
                return false;
            }
            c.0.Rip = unsafe { *(c.0.Rsp as *const u64) };
            c.0.Rsp += 8;
        } else {
            let mut handler_data = std::ptr::null_mut();
            let mut establisher = 0u64;
            unsafe { RtlVirtualUnwind(0, base, pc, fe, &mut c.0, &mut handler_data, &mut establisher, std::ptr::null_mut()) };
        }
        let pc = c.0.Rip;
        if (pc >= b && pc < e) || in_guest(pc) {
            return true;
        }
    }
    false
}

/// "frames: a <- b <- ..." for a fault context (no symbol lookups).
unsafe fn frames_line(ctx: &CONTEXT) -> String {
    let mut c: AlignedContext = unsafe { std::mem::zeroed() };
    c.0 = *ctx;
    let mut host = [0u64; 16];
    let mut guest = [0u64; 16];
    let (nh, ng) = unsafe { unwind(&mut c, &mut host, &mut guest) };
    let mut v: Vec<String> = host[..nh].iter().map(|&a| format!("{a:#x}")).collect();
    v.extend(guest[..ng].iter().map(|&a| describe(a)));
    format!("frames: {}", v.join(" <- "))
}

/// dbghelp's state (it is not thread safe): None before initialization,
/// then whether it succeeded.
static SYM: parking_lot::Mutex<Option<bool>> = parking_lot::Mutex::new(None);

/// Initializes dbghelp once (under the SYM lock): the symbol search path is
/// our executable's directory only, so no symbol server is ever contacted.
fn sym_ready(g: &mut Option<bool>) -> bool {
    use windows_sys::Win32::System::Diagnostics::Debug::{SYMOPT_DEFERRED_LOADS, SYMOPT_LOAD_LINES, SYMOPT_UNDNAME, SymInitializeW, SymSetOptions};
    use windows_sys::Win32::System::Threading::GetCurrentProcess;
    *g.get_or_insert_with(|| unsafe {
        let dir: Vec<u16> = std::env::current_exe()
            .ok()
            .and_then(|p| p.parent().map(|d| d.to_path_buf()))
            .map(|d| d.as_os_str().to_string_lossy().into_owned())
            .unwrap_or_default()
            .encode_utf16()
            .chain(std::iter::once(0))
            .collect();
        SymSetOptions(SYMOPT_UNDNAME | SYMOPT_DEFERRED_LOADS | SYMOPT_LOAD_LINES);
        SymInitializeW(GetCurrentProcess(), dir.as_ptr(), 1) != 0
    })
}

/// Source file name and line of a host code address in our executable
/// (profiles: which line of a function its samples fall on), cached.
pub fn host_line(addr: u64) -> Option<String> {
    use windows_sys::Win32::System::Diagnostics::Debug::{IMAGEHLP_LINEW64, SymGetLineFromAddrW64};
    use windows_sys::Win32::System::Threading::GetCurrentProcess;
    static LINES: parking_lot::Mutex<Option<std::collections::HashMap<u64, Option<String>>>> = parking_lot::Mutex::new(None);
    if let Some(l) = LINES.lock().as_ref().and_then(|m| m.get(&addr)) {
        return l.clone();
    }
    let line = {
        let mut g = SYM.lock();
        if !sym_ready(&mut g) {
            return None;
        }
        unsafe {
            let mut l: IMAGEHLP_LINEW64 = std::mem::zeroed();
            l.SizeOfStruct = std::mem::size_of::<IMAGEHLP_LINEW64>() as u32;
            let mut disp = 0u32;
            if SymGetLineFromAddrW64(GetCurrentProcess(), addr, &mut disp, &mut l) != 0 && !l.FileName.is_null() {
                let mut n = 0;
                while *l.FileName.add(n) != 0 && n < 1024 {
                    n += 1;
                }
                let path = String::from_utf16_lossy(std::slice::from_raw_parts(l.FileName, n));
                let file = path.rsplit(['\\', '/']).next().unwrap_or(&path).to_string();
                Some(format!("{file}:{}", l.LineNumber))
            } else {
                None
            }
        }
    };
    LINES.lock().get_or_insert_with(Default::default).insert(addr, line.clone());
    line
}

/// Name of a host code address: dbghelp symbol (our PDB, or a DLL's
/// exports) + offset. The symbol search path is our executable's directory
/// only, so no symbol server is ever contacted.
pub fn host_symbol(addr: u64) -> String {
    use windows_sys::Win32::System::Diagnostics::Debug::{SYMBOL_INFOW, SymFromAddrW};
    use windows_sys::Win32::System::Threading::GetCurrentProcess;
    let mut g = SYM.lock();
    let ready = sym_ready(&mut g);
    let (b, e) = host_image();
    let fallback = if addr >= b && addr < e {
        format!("dsport+{:#x}", addr - b)
    } else {
        match module_offset(addr) {
            Some((m, off)) => format!("{m}+{off:#x}"),
            None => format!("{addr:#x}"),
        }
    };
    if !ready || !(addr >= b && addr < e) {
        // (Other modules: their exports would name the wrong functions.)
        return fallback;
    }
    unsafe {
        let mut buf = [0u64; (std::mem::size_of::<SYMBOL_INFOW>() + 2 * 256) / 8 + 1];
        let si = buf.as_mut_ptr() as *mut SYMBOL_INFOW;
        (*si).SizeOfStruct = std::mem::size_of::<SYMBOL_INFOW>() as u32;
        (*si).MaxNameLen = 255;
        let mut disp = 0u64;
        if SymFromAddrW(GetCurrentProcess(), addr, &mut disp, si) != 0 {
            let n = ((*si).NameLen as usize).min(255);
            let name = String::from_utf16_lossy(std::slice::from_raw_parts((*si).Name.as_ptr(), n));
            format!("{name}+{disp:#x}")
        } else {
            fallback
        }
    }
}

/// [start, end) of our own executable image.
pub fn host_image() -> (u64, u64) {
    *HOST_IMAGE.get_or_init(|| unsafe {
        let base = windows_sys::Win32::System::LibraryLoader::GetModuleHandleW(std::ptr::null()) as u64;
        let nt = base + *((base + 0x3C) as *const u32) as u64;
        let size = *((nt + 0x50) as *const u32) as u64;
        (base, base + size)
    })
}

pub fn install() {
    host_image();
    let h = unsafe { AddVectoredExceptionHandler(1, Some(veh)) };
    unsafe { windows_sys::Win32::System::Diagnostics::Debug::SetUnhandledExceptionFilter(Some(unhandled)) };
    crate::info!("vectored exception handler installed: {:?}", h);
}

/// Our unhandled-exception filter again, where a library loaded since replaced it (Streamline's
/// and NGX's install their own crash handlers: a fault deep inside them then ended the game with
/// nothing in its log, the first Frame Generation test of 2026-11-26). Logged when it was taken.
pub fn reclaim_unhandled_filter(after: &str) {
    let prev = unsafe { windows_sys::Win32::System::Diagnostics::Debug::SetUnhandledExceptionFilter(Some(unhandled)) };
    if let Some(f) = prev
        && f as usize != unhandled as usize
    {
        let who = module_offset(f as usize as u64).map_or_else(|| format!("{:#x}", f as usize), |(m, o)| format!("{m}+{o:#x}"));
        crate::info!("exception: the unhandled-exception filter taken back from {who} (set during {after})");
    }
}

/// Set while `unhandled` reports a fault the vectored handler left to other
/// code (one deep inside a driver) that no handler took.
static UNHANDLED: AtomicBool = AtomicBool::new(false);

/// The sites (rip) of access violations inside NVIDIA's modules logged at first chance
/// (`veh_body`): each once, eight at most, so a site the driver's own handlers take again and
/// again (a probe of memory, say) does not use up the lines a later fatal one needs.
static DRIVER_FAULT_SITES: [std::sync::atomic::AtomicU64; 8] = [const { std::sync::atomic::AtomicU64::new(0) }; 8];

/// Whether `rip` is a driver fault site not logged yet (then it counts as logged).
fn new_driver_fault_site(rip: u64) -> bool {
    for s in &DRIVER_FAULT_SITES {
        match s.compare_exchange(0, rip, Ordering::SeqCst, Ordering::SeqCst) {
            Ok(_) => return true,
            Err(v) if v == rip => return false,
            Err(_) => {}
        }
    }
    false
}

/// A fault's frames for its log line, unwound with the registered tables only while the stack
/// pointer stays inside the faulting stack's committed region (one VirtualQuery: a corrupt chain
/// must not fault inside the handler, whose own fault would end a process the driver's handlers
/// may have saved), up to the first guest frame.
unsafe fn stack_bounded_frames(ctx: &CONTEXT, out: &mut [u64]) -> usize {
    use windows_sys::Win32::System::Diagnostics::Debug::{RtlLookupFunctionEntry, RtlVirtualUnwind};
    use windows_sys::Win32::System::Memory::{MEMORY_BASIC_INFORMATION, VirtualQuery};
    let mut mbi: MEMORY_BASIC_INFORMATION = unsafe { std::mem::zeroed() };
    if unsafe { VirtualQuery(ctx.Rsp as *const std::ffi::c_void, &mut mbi, std::mem::size_of::<MEMORY_BASIC_INFORMATION>()) } == 0 {
        return 0;
    }
    let (lo, hi) = (ctx.Rsp, mbi.BaseAddress as u64 + mbi.RegionSize as u64);
    let mut c: AlignedContext = unsafe { std::mem::zeroed() };
    c.0 = *ctx;
    let mut n = 0;
    while n < out.len() {
        let pc = c.0.Rip;
        if pc == 0 || in_guest(pc) {
            break;
        }
        out[n] = pc;
        n += 1;
        if c.0.Rsp < lo || c.0.Rsp + 8 > hi {
            break;
        }
        let mut base = 0u64;
        let fe = unsafe { RtlLookupFunctionEntry(pc, &mut base, std::ptr::null_mut()) };
        if fe.is_null() {
            c.0.Rip = unsafe { *(c.0.Rsp as *const u64) };
            c.0.Rsp += 8;
        } else {
            let mut handler_data = std::ptr::null_mut();
            let mut establisher = 0u64;
            unsafe { RtlVirtualUnwind(0, base, pc, fe, &mut c.0, &mut handler_data, &mut establisher, std::ptr::null_mut()) };
        }
        if c.0.Rsp < lo || c.0.Rsp >= hi {
            break;
        }
    }
    n
}

/// The last chance for a fault no handler took (the first sessions' access
/// violations inside nvoglv64.dll ended the process with no report of
/// ours): the fatal report, with the host frames down to our code. (Under the port's host, a
/// debugger, Windows does not call it: `veh_body` logs a driver's access violation at first
/// chance instead.)
unsafe extern "system" fn unhandled(info: *const EXCEPTION_POINTERS) -> i32 {
    UNHANDLED.store(true, Ordering::SeqCst);
    unsafe { veh_body(info as *mut EXCEPTION_POINTERS) };
    EXCEPTION_CONTINUE_SEARCH
}

fn code_name(code: u32) -> &'static str {
    match code {
        0xC0000005 => "ACCESS_VIOLATION",
        0xC000001D => "ILLEGAL_INSTRUCTION",
        0xC0000096 => "PRIVILEGED_INSTRUCTION",
        0xC0000094 => "INTEGER_DIVIDE_BY_ZERO",
        0xC0000095 => "INTEGER_OVERFLOW",
        0x80000003 => "BREAKPOINT",
        0xC00000FD => "STACK_OVERFLOW",
        0xC000008C => "ARRAY_BOUNDS_EXCEEDED",
        0xC000008E => "FLOAT_DIVIDE_BY_ZERO",
        0xC0000090 => "FLOAT_INVALID_OPERATION",
        0xC0000409 => "STACK_BUFFER_OVERRUN",
        0x80000001 => "GUARD_PAGE",
        _ => "?",
    }
}

fn in_guest(addr: u64) -> bool {
    match crate::loader::MODULES.try_read() {
        Some(m) => m.iter().any(|x| x.contains(addr)),
        None => false,
    }
}

fn describe(addr: u64) -> String {
    if crate::loader::MODULES.try_read().is_some() && crate::loader::STUBS.try_lock().is_some() {
        crate::loader::describe_addr(addr)
    } else {
        format!("{addr:#x}")
    }
}

/// Debug breakpoints (`DS_BREAK=<eboot offsets>`): int3 patched over the
/// first byte of a guest instruction; each hit is logged. A replaced
/// `push rbp` is emulated; any other instruction is executed by restoring its
/// byte and single-stepping, then the int3 is re-armed (hits on other threads
/// during that step are missed). (address, hit count, original byte)
pub static BREAKPOINTS: parking_lot::Mutex<Vec<(u64, u64, u8)>> = parking_lot::Mutex::new(Vec::new());

/// `DS_BREAK_STATS=1`: hits are counted per (breakpoint, caller chain) instead
/// of being logged one by one (logging slows the title enough to change its
/// behavior); `dump_break_stats` prints and resets the table.
static BREAK_STATS: parking_lot::Mutex<Option<std::collections::HashMap<(u64, [u64; 4]), u64>>> = parking_lot::Mutex::new(None);

pub fn break_stats_enabled() -> bool {
    static ON: OnceLock<bool> = OnceLock::new();
    *ON.get_or_init(|| std::env::var_os("DS_BREAK_STATS").is_some())
}

/// Logs the breakpoint hit histogram collected since the last call.
pub fn dump_break_stats() {
    // try_lock: also called from the fatal-fault path.
    let Some(map) = BREAK_STATS.try_lock_for(std::time::Duration::from_millis(100)).and_then(|mut g| g.take()) else { return };
    let mut v: Vec<_> = map.into_iter().collect();
    v.sort_by(|a, b| b.1.cmp(&a.1));
    let total: u64 = v.iter().map(|e| e.1).sum();
    let mut s = format!("breakpoint stats: {total} hits");
    for ((bp, chain), n) in v.iter().take(40) {
        s.push_str(&format!("
  {n:>7} {}", describe(*bp)));
        for &a in chain.iter().filter(|&&a| a != 0) {
            s.push_str(&format!(" <{}", describe(a)));
        }
    }
    crate::log::write(crate::log::Level::Info, format_args!("{s}"));
}

thread_local! {
    /// Breakpoint to re-arm after this thread's single step (0 = none).
    static REARM: std::cell::Cell<u64> = const { std::cell::Cell::new(0) };
}

/// Writes one code byte. The page is left writable: restoring its protection
/// would race with other threads patching the same page.
unsafe fn patch_code_byte(addr: u64, byte: u8) {
    use windows_sys::Win32::System::Memory::{PAGE_EXECUTE_READWRITE, VirtualProtect};
    static LOCK: parking_lot::Mutex<()> = parking_lot::Mutex::new(());
    let _g = LOCK.lock();
    unsafe {
        let mut old = 0u32;
        if VirtualProtect(addr as *const _, 1, PAGE_EXECUTE_READWRITE, &mut old) == 0 {
            return;
        }
        std::ptr::write_volatile(addr as *mut u8, byte);
        windows_sys::Win32::System::Diagnostics::Debug::FlushInstructionCache(windows_sys::Win32::System::Threading::GetCurrentProcess(), addr as *const _, 1);
    }
}
/// Remaining breakpoint hits to log unconditionally (control `breaklog N`).
pub static BREAK_VERBOSE: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);

/// Installs a logging breakpoint at `addr`, which must be the first byte of
/// a guest instruction.
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
    use windows_sys::Win32::System::Memory::{MEMORY_BASIC_INFORMATION, MEM_COMMIT, PAGE_GUARD, PAGE_NOACCESS, VirtualQuery};
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
        match p.hook(off, ranges, text_hook_entry as usize as u64, i as u32) {
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
    let mut s = unsafe { peek_string(a, false) };
    if let Some(n) = len.as_ref().and_then(|l| eval_expr(l, r)) {
        s = s.chars().take(n as usize).collect();
    }
    crate::log::write(crate::log::Level::Info, format_args!("text: {s}"));
}

pub fn install_breakpoint(addr: u64) -> Result<(), String> {
    let orig = unsafe { *(addr as *const u8) };
    if orig == 0xCC {
        return Err(format!("{addr:#x} already holds int3"));
    }
    BREAKPOINTS.lock().push((addr, 0, orig));
    unsafe { patch_code_byte(addr, 0xCC) };
    if unsafe { *(addr as *const u8) } != 0xCC {
        return Err("could not patch code".into());
    }
    Ok(())
}

/// Up to 120 characters of the NUL-terminated string at `a` (UTF-16 or
/// bytes), stopping at unreadable memory.
unsafe fn peek_string(a: u64, wide: bool) -> String {
    use windows_sys::Win32::System::Memory::{MEMORY_BASIC_INFORMATION, MEM_COMMIT, PAGE_GUARD, PAGE_NOACCESS, VirtualQuery};
    let mut mbi: MEMORY_BASIC_INFORMATION = unsafe { std::mem::zeroed() };
    if a < 0x1000 || unsafe { VirtualQuery(a as *const _, &mut mbi, std::mem::size_of::<MEMORY_BASIC_INFORMATION>()) } == 0 {
        return String::new();
    }
    if mbi.State != MEM_COMMIT || mbi.Protect & (PAGE_NOACCESS | PAGE_GUARD) != 0 || mbi.Protect == 0 {
        return String::new();
    }
    let end = mbi.BaseAddress as u64 + mbi.RegionSize as u64;
    let unit = if wide { 2 } else { 1 };
    let n = (((end - a) / unit) as usize).min(120);
    if wide {
        let v: Vec<u16> = (0..n).map(|i| unsafe { *((a + 2 * i as u64) as *const u16) }).take_while(|&c| c != 0).collect();
        String::from_utf16_lossy(&v)
    } else {
        let v: Vec<u8> = (0..n).map(|i| unsafe { *((a + i as u64) as *const u8) }).take_while(|&c| c != 0).collect();
        String::from_utf8_lossy(&v).into_owned()
    }
}

/// Evaluates a peek expression `base(+hex|*)*` against register values;
/// `*` loads a qword (when readable).
unsafe fn eval_peek(expr: &str, regs: &[(&str, u64)]) -> Option<u64> {
    let readable = |a: u64| -> bool {
        use windows_sys::Win32::System::Memory::{MEMORY_BASIC_INFORMATION, MEM_COMMIT, PAGE_GUARD, PAGE_NOACCESS, VirtualQuery};
        let mut mbi: MEMORY_BASIC_INFORMATION = unsafe { std::mem::zeroed() };
        if unsafe { VirtualQuery(a as *const _, &mut mbi, std::mem::size_of::<MEMORY_BASIC_INFORMATION>()) } == 0 {
            return false;
        }
        mbi.State == MEM_COMMIT && mbi.Protect & (PAGE_NOACCESS | PAGE_GUARD) == 0 && mbi.Protect != 0 && a + 8 <= mbi.BaseAddress as u64 + mbi.RegionSize as u64
    };
    let base_end = expr.find(['+', '-', '*']).unwrap_or(expr.len());
    let base = &expr[..base_end];
    let mut v = regs.iter().find(|(n, _)| *n == base).map(|(_, v)| *v).or_else(|| u64::from_str_radix(base.trim_start_matches("0x"), 16).ok())?;
    let mut rest = &expr[base_end..];
    while !rest.is_empty() {
        if let Some(r) = rest.strip_prefix('*') {
            if !readable(v & !7) {
                return None;
            }
            v = unsafe { *((v & !7) as *const u64) };
            rest = r;
        } else if let Some(r) = rest.strip_prefix('+') {
            let end = r.find(['+', '-', '*']).unwrap_or(r.len());
            v = v.wrapping_add(u64::from_str_radix(&r[..end], 16).ok()?);
            rest = &r[end..];
        } else if let Some(r) = rest.strip_prefix('-') {
            let end = r.find(['+', '-', '*']).unwrap_or(r.len());
            v = v.wrapping_sub(u64::from_str_radix(&r[..end], 16).ok()?);
            rest = &r[end..];
        } else {
            return None;
        }
    }
    Some(v)
}

unsafe fn handle_breakpoint(ctx: &mut CONTEXT) -> bool {
    unsafe {
        let rip = ctx.Rip;
        let hit = {
            let mut bps = BREAKPOINTS.lock();
            bps.iter_mut().find(|b| b.0 == rip || b.0 + 1 == rip).map(|b| {
                b.1 += 1;
                (b.0, b.1, b.2)
            })
        };
        let Some((bp, n, orig)) = hit else { return false };
        if *(bp as *const u8) != 0xCC && rip == bp {
            // Restored by another thread's step: just execute the instruction.
            return true;
        }
        if break_stats_enabled() {
            // Return address (meaningful at function entries) and rbp chain.
            let mut key = [0u64; 4];
            key[0] = *(ctx.Rsp as *const u64);
            let depth: usize = std::env::var("DS_BREAK_DEPTH").ok().and_then(|v| v.parse().ok()).unwrap_or(0).min(3);
            let mut fp = ctx.Rbp;
            for slot in key.iter_mut().skip(1).take(depth) {
                use windows_sys::Win32::System::Memory::{MEMORY_BASIC_INFORMATION, MEM_COMMIT, VirtualQuery};
                let mut mbi: MEMORY_BASIC_INFORMATION = std::mem::zeroed();
                if fp < 0x1000 || fp & 7 != 0 || VirtualQuery(fp as *const _, &mut mbi, std::mem::size_of::<MEMORY_BASIC_INFORMATION>()) == 0 || mbi.State != MEM_COMMIT {
                    break;
                }
                *slot = *((fp + 8) as *const u64);
                fp = *(fp as *const u64);
            }
            *BREAK_STATS.lock().get_or_insert_with(Default::default).entry((bp, key)).or_insert(0) += 1;
        }
        let verbose = BREAK_VERBOSE.fetch_update(Ordering::Relaxed, Ordering::Relaxed, |v| v.checked_sub(1)).is_ok();
        if !break_stats_enabled() && (n <= 64 || n.is_power_of_two() || verbose) {
            let ret = *(ctx.Rsp as *const u64);
            // DS_BREAK_PEEK: comma-separated expressions logged with each hit.
            let mut extra = String::new();
            // DS_BREAK_DEPTH: caller frames (rbp chain) appended to each hit.
            let depth: usize = std::env::var("DS_BREAK_DEPTH").ok().and_then(|v| v.parse().ok()).unwrap_or(0);
            if depth > 0 {
                let mut bp_ = ctx.Rbp;
                extra.push_str(" chain");
                for _ in 0..depth {
                    if bp_ < 0x1000 || bp_ & 7 != 0 {
                        break;
                    }
                    use windows_sys::Win32::System::Memory::{MEMORY_BASIC_INFORMATION, MEM_COMMIT, VirtualQuery};
                    let mut mbi: MEMORY_BASIC_INFORMATION = std::mem::zeroed();
                    if VirtualQuery(bp_ as *const _, &mut mbi, std::mem::size_of::<MEMORY_BASIC_INFORMATION>()) == 0 || mbi.State != MEM_COMMIT {
                        break;
                    }
                    let r = *((bp_ + 8) as *const u64);
                    extra.push_str(&format!(" <{}", describe(r)));
                    bp_ = *(bp_ as *const u64);
                }
            }
            // DS_BREAK_XMM=<n>: the low float of xmm0..xmm(n-1) (float
            // arguments of the System V ABI).
            if let Some(n) = std::env::var("DS_BREAK_XMM").ok().and_then(|v| v.parse::<usize>().ok()) {
                let x = &ctx.Anonymous.Anonymous;
                let regs = [x.Xmm0, x.Xmm1, x.Xmm2, x.Xmm3, x.Xmm4, x.Xmm5, x.Xmm6, x.Xmm7];
                for (k, r) in regs.iter().enumerate().take(n.min(8)) {
                    extra.push_str(&format!(" xmm{k}={}", f32::from_bits(r.Low as u32)));
                }
            }
            if let Ok(spec) = std::env::var("DS_BREAK_PEEK") {
                let regs = [
                    ("rdi", ctx.Rdi),
                    ("rsi", ctx.Rsi),
                    ("rdx", ctx.Rdx),
                    ("rcx", ctx.Rcx),
                    ("r8", ctx.R8),
                    ("r9", ctx.R9),
                    ("rsp", ctx.Rsp),
                    ("rax", ctx.Rax),
                    ("rbx", ctx.Rbx),
                    ("rbp", ctx.Rbp),
                    ("r12", ctx.R12),
                    ("r13", ctx.R13),
                    ("r14", ctx.R14),
                    ("r15", ctx.R15),
                ];
                for e in spec.split(',').map(str::trim).filter(|e| !e.is_empty()) {
                    // `s:<expr>` / `w:<expr>`: the UTF-8 / UTF-16 string at the address.
                    if let Some((kind, x)) = e.strip_prefix("s:").map(|x| (1, x)).or_else(|| e.strip_prefix("w:").map(|x| (2, x))) {
                        match eval_peek(x, &regs) {
                            Some(a) => extra.push_str(&format!(" {e}={:?}", peek_string(a, kind == 2))),
                            None => extra.push_str(&format!(" {e}=?")),
                        }
                        continue;
                    }
                    match eval_peek(e, &regs) {
                        Some(v) => extra.push_str(&format!(" {e}={v:#x}")),
                        None => extra.push_str(&format!(" {e}=?")),
                    }
                }
            }
            crate::log::write(
                crate::log::Level::Info,
                format_args!(
                    "break {} #{n}: rdi={:#x} rsi={:#x} rdx={:#x} rcx={:#x} r8={:#x} r9={:#x} ret {}{}",
                    describe(bp),
                    ctx.Rdi,
                    ctx.Rsi,
                    ctx.Rdx,
                    ctx.Rcx,
                    ctx.R8,
                    ctx.R9,
                    describe(ret),
                    extra
                ),
            );
        }
        if orig == 0x55 {
            // Emulate the replaced `push rbp`.
            ctx.Rsp -= 8;
            *(ctx.Rsp as *mut u64) = ctx.Rbp;
            ctx.Rip = bp + 1;
        } else {
            // Execute the original instruction with the trap flag set; the
            // single-step exception re-arms the breakpoint.
            patch_code_byte(bp, orig);
            ctx.Rip = bp;
            ctx.EFlags |= 0x100;
            REARM.with(|r| r.set(bp));
        }
        true
    }
}

/// Hardware watchpoints in the four debug-address registers (DR0-DR3), on
/// every guest thread and runtime worker thread that registered itself
/// (`arm_watch_current_thread`). Slot 0 can come from `DS_WATCH=<hex address>`
/// (8 bytes; writes, or reads and writes with `DS_WATCH_RW=1`) and any slot
/// from the `watch` control command at run time (`set_watch`). Each hit is
/// logged with the value at the address, the accessing thread and its frames
/// (first `DS_WATCH_MAX`, default 64, hits per slot).
static SLOT_ADDR: [std::sync::atomic::AtomicU64; 4] = [const { std::sync::atomic::AtomicU64::new(0) }; 4];
/// RW bits (1 writes, 3 reads and writes) | LEN bits << 4 (0 = 1 byte,
/// 1 = 2, 3 = 4, 2 = 8) | guest threads only << 8.
static SLOT_MODE: [std::sync::atomic::AtomicU32; 4] = [const { std::sync::atomic::AtomicU32::new(0) }; 4];
static SLOT_HITS: [std::sync::atomic::AtomicU32; 4] = [const { std::sync::atomic::AtomicU32::new(0) }; 4];
/// Registered threads: (host thread id, runs guest code).
static WATCH_TIDS: parking_lot::Mutex<Vec<(u32, bool)>> = parking_lot::Mutex::new(Vec::new());
const CONTEXT_DEBUG_REGISTERS_AMD64: u32 = 0x0010_0010;

fn watch_env_init() {
    static INIT: std::sync::Once = std::sync::Once::new();
    INIT.call_once(|| {
        let a = std::env::var("DS_WATCH").ok().and_then(|v| u64::from_str_radix(v.trim().trim_start_matches("0x"), 16).ok()).unwrap_or(0) & !7;
        if a != 0 {
            let rw = if std::env::var("DS_WATCH_RW").ok().as_deref() == Some("1") { 3 } else { 1 };
            SLOT_MODE[0].store(rw | (2 << 4), Ordering::SeqCst);
            SLOT_ADDR[0].store(a, Ordering::SeqCst);
        }
    });
}

/// Sets the debug registers of a context from the slots.
fn apply_slots(c: &mut CONTEXT, guest: bool) {
    let mut dr7 = c.Dr7 & !0xFFFF_00FF;
    for i in 0..4 {
        let a = SLOT_ADDR[i].load(Ordering::SeqCst);
        let m = SLOT_MODE[i].load(Ordering::SeqCst);
        let on = a != 0 && (guest || m & 0x100 == 0);
        let a = if on { a } else { 0 };
        match i {
            0 => c.Dr0 = a,
            1 => c.Dr1 = a,
            2 => c.Dr2 = a,
            _ => c.Dr3 = a,
        }
        if on {
            dr7 |= (1 << (2 * i)) | (((m & 3) as u64) << (16 + 4 * i)) | ((((m >> 4) & 3) as u64) << (18 + 4 * i));
        }
    }
    c.Dr7 = dr7;
}

/// Registers the calling thread for watchpoints and arms the active slots
/// on it (`guest`: the thread runs guest code; guest-only slots skip others).
pub fn arm_watch_current_thread(guest: bool) {
    use windows_sys::Win32::System::Diagnostics::Debug::{GetThreadContext, SetThreadContext};
    use windows_sys::Win32::System::Threading::{GetCurrentThread, GetCurrentThreadId};
    watch_env_init();
    WATCH_TIDS.lock().push((unsafe { GetCurrentThreadId() }, guest));
    if SLOT_ADDR.iter().all(|a| a.load(Ordering::SeqCst) == 0) {
        return;
    }
    unsafe {
        let mut c: AlignedContext = std::mem::zeroed();
        c.0.ContextFlags = CONTEXT_DEBUG_REGISTERS_AMD64;
        if GetThreadContext(GetCurrentThread(), &mut c.0) == 0 {
            return;
        }
        apply_slots(&mut c.0, guest);
        if SetThreadContext(GetCurrentThread(), &c.0) == 0 {
            crate::warn!("watch: SetThreadContext failed on this thread");
        }
    }
}

/// Re-applies the active watch slots to the calling thread (`set_watch` updates every other
/// registered thread; a hook arming a slot on the thread it runs on calls this after it).
pub fn arm_watch_here(guest: bool) {
    use windows_sys::Win32::System::Diagnostics::Debug::{GetThreadContext, SetThreadContext};
    use windows_sys::Win32::System::Threading::GetCurrentThread;
    unsafe {
        let mut c: AlignedContext = std::mem::zeroed();
        c.0.ContextFlags = CONTEXT_DEBUG_REGISTERS_AMD64;
        if GetThreadContext(GetCurrentThread(), &mut c.0) == 0 {
            return;
        }
        apply_slots(&mut c.0, guest);
        if SetThreadContext(GetCurrentThread(), &c.0) == 0 {
            crate::warn!("watch: SetThreadContext failed on this thread");
        }
    }
}

/// Sets watch slot `slot` (address 0: off) and re-arms every registered
/// thread (suspended while its debug registers change). Returns how many
/// threads were updated.
pub fn set_watch(slot: usize, addr: u64, rw: u32, len: u32, guest_only: bool) -> usize {
    use windows_sys::Win32::System::Diagnostics::Debug::{GetThreadContext, SetThreadContext};
    use windows_sys::Win32::System::Threading::{GetCurrentThreadId, OpenThread, ResumeThread, SuspendThread, THREAD_GET_CONTEXT, THREAD_SET_CONTEXT, THREAD_SUSPEND_RESUME};
    watch_env_init();
    let lenbits = match len {
        1 => 0,
        2 => 1,
        4 => 3,
        _ => 2,
    };
    SLOT_MODE[slot].store(rw | (lenbits << 4) | if guest_only { 0x100 } else { 0 }, Ordering::SeqCst);
    SLOT_ADDR[slot].store(addr & !(len.max(1) as u64 - 1), Ordering::SeqCst);
    SLOT_HITS[slot].store(0, Ordering::SeqCst);
    let me = unsafe { GetCurrentThreadId() };
    let mut tids = WATCH_TIDS.lock();
    let mut done = 0;
    tids.retain(|&(tid, guest)| {
        if tid == me {
            return true;
        }
        unsafe {
            let h = OpenThread(THREAD_SUSPEND_RESUME | THREAD_GET_CONTEXT | THREAD_SET_CONTEXT, 0, tid);
            if h.is_null() {
                return false; // exited
            }
            if SuspendThread(h) != u32::MAX {
                let mut c: AlignedContext = std::mem::zeroed();
                c.0.ContextFlags = CONTEXT_DEBUG_REGISTERS_AMD64;
                if GetThreadContext(h, &mut c.0) != 0 {
                    apply_slots(&mut c.0, guest);
                    if SetThreadContext(h, &c.0) != 0 {
                        done += 1;
                    }
                }
                ResumeThread(h);
            }
            windows_sys::Win32::Foundation::CloseHandle(h);
        }
        true
    });
    done
}

unsafe fn handle_watch_hit(ctx: &mut CONTEXT) -> bool {
    let hit = (ctx.Dr6 & 0xF) as usize;
    if hit == 0 {
        return false;
    }
    let mut ours = false;
    for i in 0..4 {
        let a = SLOT_ADDR[i].load(Ordering::Relaxed);
        if hit & (1 << i) == 0 || a == 0 {
            continue;
        }
        ours = true;
        static MAX: OnceLock<u32> = OnceLock::new();
        let n = SLOT_HITS[i].fetch_add(1, Ordering::Relaxed);
        if n < *MAX.get_or_init(|| std::env::var("DS_WATCH_MAX").ok().and_then(|v| v.parse().ok()).unwrap_or(64)) {
            // Read through the kernel: a plain read of a read-watched address
            // would hit the watchpoint again inside this handler.
            let mut v = 0u64;
            unsafe {
                windows_sys::Win32::System::Diagnostics::Debug::ReadProcessMemory(
                    windows_sys::Win32::System::Threading::GetCurrentProcess(),
                    (a & !7) as *const _,
                    &mut v as *mut u64 as *mut _,
                    8,
                    std::ptr::null_mut(),
                );
            }
            let kind = if SLOT_MODE[i].load(Ordering::Relaxed) & 3 == 3 { "access" } else { "write" };
            let frames = unsafe { frames_line(ctx) };
            crate::log::write(
                crate::log::Level::Info,
                format_args!(
                    "watch {i} {kind} #{n}: [{a:#x}] = {v:#018x} by [{}] at {} (rax {:#x} rbx {:#x} rcx {:#x} rdx {:#x} rsi {:#x} rdi {:#x} rbp {:#x} r12 {:#x} r13 {:#x} r14 {:#x} r15 {:#x} rsp {:#x}, [rsp] {}); {frames}",
                    crate::log::thread_name(),
                    describe(ctx.Rip),
                    ctx.Rax,
                    ctx.Rbx,
                    ctx.Rcx,
                    ctx.Rdx,
                    ctx.Rsi,
                    ctx.Rdi,
                    ctx.Rbp,
                    ctx.R12,
                    ctx.R13,
                    ctx.R14,
                    ctx.R15,
                    ctx.Rsp,
                    // The return address of a leaf function (frames from the
                    // frame-pointer chain skip it).
                    if readable(ctx.Rsp) { describe(unsafe { *(ctx.Rsp as *const u64) }) } else { "?".to_string() }
                ),
            );
        }
    }
    if ours {
        ctx.Dr6 = 0;
    }
    ours
}

/// Single-step exception: watchpoint hits and/or the step after a
/// non-`push rbp` breakpoint (which re-arms it).
unsafe fn handle_single_step(ctx: &mut CONTEXT) -> bool {
    let watched = unsafe { handle_watch_hit(ctx) };
    let bp = REARM.with(|r| r.replace(0));
    if bp == 0 {
        return watched;
    }
    unsafe { patch_code_byte(bp, 0xCC) };
    ctx.EFlags &= !0x100;
    true
}

/// Pages watched for CPU writes (diagnostics, `DS_RACE_WATCH`): (page, old
/// protection); the first write to each is recorded with its guest stack.
static RACE_PAGES: parking_lot::Mutex<Vec<(u64, u32)>> = parking_lot::Mutex::new(Vec::new());
static RACE_HITS: parking_lot::Mutex<Vec<String>> = parking_lot::Mutex::new(Vec::new());
static RACE_ACTIVE: AtomicBool = AtomicBool::new(false);

/// Write-protects [addr, addr+len) until the first CPU write to each page.
pub fn race_watch(addr: u64, len: u64) {
    use windows_sys::Win32::System::Memory::{PAGE_READONLY, VirtualProtect};
    let mut g = RACE_PAGES.lock();
    RACE_ACTIVE.store(true, Ordering::SeqCst);
    let mut p = addr & !0xFFF;
    while p < addr + len {
        let mut old = 0u32;
        if unsafe { VirtualProtect(p as *mut _, 4096, PAGE_READONLY, &mut old) } != 0 {
            g.push((p, old));
        }
        p += 4096;
    }
}

/// Restores the watched pages; returns the writes recorded so far.
pub fn race_unwatch() -> Vec<String> {
    use windows_sys::Win32::System::Memory::VirtualProtect;
    let mut g = RACE_PAGES.lock();
    for (p, old) in g.drain(..) {
        let mut tmp = 0u32;
        unsafe { VirtualProtect(p as *mut _, 4096, old, &mut tmp) };
    }
    std::mem::take(&mut *RACE_HITS.lock())
}

fn race_fault(addr: u64, ctx: &CONTEXT) -> bool {
    use windows_sys::Win32::System::Memory::VirtualProtect;
    if !RACE_ACTIVE.load(Ordering::Relaxed) {
        return false;
    }
    let page = addr & !0xFFF;
    let mut g = RACE_PAGES.lock();
    let Some(i) = g.iter().position(|x| x.0 == page) else { return false };
    let (p, old) = g.remove(i);
    drop(g);
    let mut tmp = 0u32;
    unsafe { VirtualProtect(p as *mut _, 4096, old, &mut tmp) };
    let mut c: AlignedContext = unsafe { std::mem::zeroed() };
    c.0 = *ctx;
    let (mut host, mut guest) = ([0u64; 4], [0u64; 12]);
    let (_, ng) = unsafe { unwind(&mut c, &mut host, &mut guest) };
    let frames: Vec<String> = guest[..ng].iter().map(|&a| describe(a)).collect();
    crate::warn!("race watch: write to {addr:#x} by '{}': {}", crate::log::thread_name(), frames.join(" <- "));
    true
}

unsafe extern "system" fn veh(info: *mut EXCEPTION_POINTERS) -> i32 {
    // On a fiber's own stack (sized by the title), the kernel's exception
    // frame is already there; the handler itself runs on the fiber's host stack.
    crate::hle::stack_check("exception handler");
    match crate::hle::libs::fiber::on_fiber_stack() {
        Some(fiber) => unsafe { crate::hle::libs::fiber::on_host_stack(fiber, || veh_body(info)) },
        None => unsafe { veh_body(info) },
    }
}

unsafe fn veh_body(info: *mut EXCEPTION_POINTERS) -> i32 {
    unsafe {
        let rec = &*(*info).ExceptionRecord;
        let code = rec.ExceptionCode as u32;
        if code == 0xC0000005 && rec.NumberParameters >= 2 && rec.ExceptionInformation[0] == 1 && race_fault(rec.ExceptionInformation[1] as u64, &*(*info).ContextRecord) {
            return -1; // EXCEPTION_CONTINUE_EXECUTION
        }
        // An access inside a view being split (`sections`): retried after it.
        if code == 0xC0000005 && rec.NumberParameters >= 2 && crate::sections::wait_split(rec.ExceptionInformation[1] as u64) {
            return -1; // EXCEPTION_CONTINUE_EXECUTION
        }
        // A CPU write to a page the residency mirror armed, or any access to
        // a GPU-owned page of mirror mode (`gpu::resident`).
        if code == 0xC0000005 && rec.NumberParameters >= 2 && {
            crate::gpu::resident::FAULT_RIP.with(|r| r.set((*(*info).ContextRecord).Rip));
            crate::gpu::resident::on_fault(rec.ExceptionInformation[1] as u64, rec.ExceptionInformation[0] == 1)
        } {
            return -1; // EXCEPTION_CONTINUE_EXECUTION
        }
        // A CPU write to a write-tracked guest page (texture cache).
        if code == 0xC0000005 && rec.NumberParameters >= 2 && rec.ExceptionInformation[0] == 1 && crate::gpu::track::on_write_fault(rec.ExceptionInformation[1] as u64) {
            return -1; // EXCEPTION_CONTINUE_EXECUTION
        }
        if code == 0x80000003 && handle_breakpoint(&mut *(*info).ContextRecord) {
            return -1; // EXCEPTION_CONTINUE_EXECUTION
        }
        if code == 0x80000004 && handle_single_step(&mut *(*info).ContextRecord) {
            return -1;
        }
        let ctx = &*(*info).ContextRecord;
        let interesting = matches!(
            code,
            0xC0000005
                | 0xC000001D
                | 0xC0000096
                | 0xC0000094
                | 0xC0000095
                | 0x80000003
                | 0xC00000FD
                | 0xC000008C
                | 0xC000008E
                | 0xC0000090
                | 0xC0000409
        );
        if !interesting {
            return EXCEPTION_CONTINUE_SEARCH;
        }
        let rip = ctx.Rip;
        let host = host_image();
        let in_host = rip >= host.0 && rip < host.1;
        let stub = crate::loader::STUBS.try_lock().and_then(|s| s.describe(rip)).is_some();
        if !in_guest(rip) && !in_host && !stub && !UNHANDLED.load(Ordering::SeqCst) {
            // A fault in another module counts as ours when the faulting
            // function was called (almost) directly by our image or by guest
            // code, e.g. the C runtime's memcpy given a bad guest pointer.
            // Exceptions deeper inside system code are left to its handlers
            // (and reported by `unhandled` when none takes them).
            // (An integer division by zero called from our code is ours too: reported with our
            // frames; the NVIDIA driver crashed so on 2026-11-15, with the scouts of that day:
            // FINDINGS.md. Deeper in another module it is left to that module's own handlers, as
            // an access violation is: a 2026-11-15 review.)
            if !matches!(code, 0xC0000005 | 0xC0000094) || !called_from_our_code(ctx, 3) {
                // (A fault inside the driver while it compiles one of our pipelines: which one,
                // logged at first chance; its handlers, or none, decide the rest. A breakpoint
                // too since 2026-11-24 night: NVIDIA's 621.40 compiler stops on one (status
                // 0x80000003, a player's compute step twice, issue #5's silent ends): unhandled,
                // the process ended with nothing in its log and nothing on the crash list, so
                // the launcher's step met it again in every attempt.)
                if matches!(code, 0xC0000005 | 0xC0000094 | 0xC000001D | 0x80000003) {
                    if let Some(what) = crate::gpu::pcache::crumb_report() {
                        let module = module_offset(rip).map_or_else(|| format!("{rip:#x}"), |(m, o)| format!("{m}+{o:#x}"));
                        // (Under the port's host the game has a debugger attached, so code that
                        // breaks only for a debugger breaks here, and the host passes every such
                        // exception on: the process ended with 0x80000003 where a game without a
                        // debugger goes on (the first review of 2026-11-25). Stepped over, as
                        // without a debugger: a one-byte int3 (Windows reports a breakpoint at
                        // the int3's own byte), not one in a run of 0xCC padding (a stray jump),
                        // the first in the compile only, the compile's binaries then not kept;
                        // no suspect line for it, the compile's in-flight record covers a death
                        // after it (the second review).)
                        if code == 0x80000003 && crate::hosted::on() {
                            let at = rip as *const u8;
                            let padding = rip & 0xFFF != 0xFFF && *at.add(1) == 0xCC;
                            if *at == 0xCC && !padding && crate::gpu::pcache::note_stepped_break() {
                                (*(*info).ContextRecord).Rip += 1;
                                crate::warn!("driver debug break at {module} while compiling {what}: stepped over, as a game without a debugger goes on (the port's host is one)");
                                crate::log::try_flush();
                                return -1; // EXCEPTION_CONTINUE_EXECUTION
                            }
                        }
                        crate::warn!("driver fault {code:#010x} at {module} while compiling {what}");
                        crate::log::try_flush();
                        // (A suspect: the driver's own handlers may still take it. The fatal
                        // report confirms it, `pcache::note_crash_suspect`.)
                        crate::gpu::pcache::note_crash_suspect();
                    }
                }
                // (An access violation deep inside NVIDIA's driver, left to its handlers: under
                // the port's host, a debugger, no unhandled-exception filter runs, so when none
                // takes it the game ends with no report of ours (DLSS's sessions of 2026-11-25:
                // only Streamline's minidump). Logged at first chance with our frames, each site
                // once, eight at most.)
                // (Streamline's plugins and NGX's libraries too since 2026-11-26: named sl.* in the
                // upscalers folder, or numbered copies under ProgramData\NVIDIA\NGX that the
                // NVIDIA app keeps up to date; a fault in them ended the first Frame Generation
                // test with nothing in the log.)
                if code == 0xC0000005
                    && let Some((m, o)) = module_offset(rip).filter(|(m, _)| nvidia_module(m, rip))
                    && new_driver_fault_site(rip)
                {
                    let (access, target) = if rec.NumberParameters >= 2 {
                        let a = match rec.ExceptionInformation[0] {
                            0 => "read",
                            1 => "write",
                            8 => "execute",
                            _ => "access",
                        };
                        (a, rec.ExceptionInformation[1] as u64)
                    } else {
                        ("access", 0)
                    };
                    let mut host = [0u64; 24];
                    let nh = stack_bounded_frames(ctx, &mut host);
                    let frames: Vec<String> = host[..nh].iter().map(|&a| module_offset(a).map_or_else(|| format!("{a:#x}"), |(m, o)| format!("{m}+{o:#x}"))).collect();
                    crate::warn!("access violation inside {m}+{o:#x} ({access} of {target:#x}), first chance, left to its handlers; frames: {}", frames.join(" <- "));
                    crate::log::try_flush();
                }
                return EXCEPTION_CONTINUE_SEARCH;
            }
        }
        let me = windows_sys::Win32::System::Threading::GetCurrentThreadId();
        if IN_HANDLER.swap(true, Ordering::SeqCst) {
            if REPORTER.load(Ordering::SeqCst) == me {
                // A fault inside the report itself: note it and stop immediately.
                {
                    use std::io::Write;
                    let _ = writeln!(std::io::stderr(), "nested fault {code:#010x} at {rip:#x} while writing a fatal report");
                }
                crate::log::try_flush();
                windows_sys::Win32::System::Threading::TerminateProcess(
                    windows_sys::Win32::System::Threading::GetCurrentProcess(),
                    0xD50F,
                );
            }
            // Another thread is already reporting a fatal fault: note this one
            // and park until that report terminates the process.
            crate::log::write(
                crate::log::Level::Error,
                format_args!("concurrent fatal fault {:#010x} at {} [{}]; {}", code, describe(rip), crate::log::thread_name(), frames_line(ctx)),
            );
            crate::log::try_flush();
            loop {
                std::thread::sleep(std::time::Duration::from_secs(1));
            }
        }
        REPORTER.store(me, Ordering::SeqCst);
        // (A fatal fault in the driver while this thread has it compile a pipeline (another
        // module's code, called closely from ours or deeper), or the injected compiler crash: the
        // pipeline goes to the crash list, so no later session makes it again (the preparation's
        // injected crash of 2026-11-21 night met it in every attempt and the preparation gave up).
        // A fault in our own code inside a compile (reading or decoding its binaries) is our bug,
        // not the driver's: until the review of 2026-11-21 night it listed the pipeline too.)
        let compile_crash = (!in_host && !in_guest(rip) && !stub) || crate::gpu::pcache::sim_crash_now();
        if break_stats_enabled() {
            dump_break_stats();
        }
        let mut r = String::new();
        use std::fmt::Write;
        let _ = writeln!(r, "FATAL {} ({:#010x}) at {} [{}]", code_name(code), code, describe(rip), crate::log::thread_name());
        // The first line goes out at once: if composing the rest faults, it survives.
        crate::log::write(crate::log::Level::Error, format_args!("fatal fault (report follows): {}", r.trim_end()));
        crate::log::try_flush();
        // (The crash list's file work after that line: a fault inside it (a stack overflow's)
        // had ended the process before any FATAL line; the review of 2026-11-21 night.)
        if compile_crash {
            crate::gpu::pcache::note_crash();
        }
        if in_guest(rip) && *(rip as *const u8) == 0xCD && *((rip + 1) as *const u8) == 0x45 {
            // PS5 libc's fatal-error trap: `mov dword ptr fs:[28h], code; int 45h`
            // (the FS store is rewritten to the guest TCB).
            let tp = crate::tls::current_tp();
            let ec = if tp != 0 { *((tp + 0x28) as *const u32) } else { 0 };
            let _ = writeln!(r, "  guest error trap `int 45h`: TCB error code {ec:#010x}");
        }
        if code == 0xC0000005 && rec.NumberParameters >= 2 {
            let kind = match rec.ExceptionInformation[0] {
                0 => "read",
                1 => "write",
                8 => "execute",
                _ => "?",
            };
            let addr = rec.ExceptionInformation[1] as u64;
            let _ = writeln!(r, "  {kind} of {:#x} ({})", addr, describe(addr));
            // The page as the host sees it and as the guest mapped it.
            {
                use windows_sys::Win32::System::Memory::{MEMORY_BASIC_INFORMATION, VirtualQuery};
                let mut mbi: MEMORY_BASIC_INFORMATION = std::mem::zeroed();
                if VirtualQuery(addr as *const _, &mut mbi, std::mem::size_of::<MEMORY_BASIC_INFORMATION>()) != 0 {
                    let _ = writeln!(
                        r,
                        "  host page: region {:#x}+{:#x} state {:#x} protect {:#x}",
                        mbi.BaseAddress as u64, mbi.RegionSize as u64, mbi.State, mbi.Protect
                    );
                }
                match crate::mem::MEM.try_lock() {
                    Some(m) => match m.find(addr) {
                        Some(v) => {
                            let _ = writeln!(r, "  guest mapping: {:#x}+{:#x} {:?} prot {:#x} phys {:#x} '{}'", v.start, v.size, v.kind, v.prot, v.phys, v.name);
                        }
                        None => {
                            let _ = writeln!(r, "  guest mapping: none");
                        }
                    },
                    None => {
                        let _ = writeln!(r, "  guest mapping: (map lock held)");
                    }
                }
                for l in crate::mem::unmaps_covering(addr) {
                    let _ = writeln!(r, "  unmapped earlier: {l}");
                }
            }
        }
        let _ = writeln!(
            r,
            "  rax={:#018x} rbx={:#018x} rcx={:#018x} rdx={:#018x}\n  rsi={:#018x} rdi={:#018x} rbp={:#018x} rsp={:#018x}\n  r8 ={:#018x} r9 ={:#018x} r10={:#018x} r11={:#018x}\n  r12={:#018x} r13={:#018x} r14={:#018x} r15={:#018x} efl={:#x}",
            ctx.Rax, ctx.Rbx, ctx.Rcx, ctx.Rdx, ctx.Rsi, ctx.Rdi, ctx.Rbp, ctx.Rsp, ctx.R8, ctx.R9, ctx.R10, ctx.R11, ctx.R12,
            ctx.R13, ctx.R14, ctx.R15, ctx.EFlags
        );
        // Memory at registers that point to readable memory (checked with
        // VirtualQuery: the handler must not take runtime locks).
        {
            use windows_sys::Win32::System::Memory::{MEMORY_BASIC_INFORMATION, MEM_COMMIT, PAGE_NOACCESS, PAGE_GUARD, VirtualQuery};
            let readable = |a: u64| -> bool {
                let mut mbi: MEMORY_BASIC_INFORMATION = std::mem::zeroed();
                if VirtualQuery(a as *const _, &mut mbi, std::mem::size_of::<MEMORY_BASIC_INFORMATION>()) == 0 {
                    return false;
                }
                let end = mbi.BaseAddress as u64 + mbi.RegionSize as u64;
                mbi.State == MEM_COMMIT && mbi.Protect & (PAGE_NOACCESS | PAGE_GUARD) == 0 && mbi.Protect != 0 && a + 64 <= end
            };
            let regs = [
                ("rax", ctx.Rax),
                ("rbx", ctx.Rbx),
                ("rcx", ctx.Rcx),
                ("rdx", ctx.Rdx),
                ("rsi", ctx.Rsi),
                ("rdi", ctx.Rdi),
                ("r8", ctx.R8),
                ("r9", ctx.R9),
                ("r10", ctx.R10),
                ("r11", ctx.R11),
                ("r12", ctx.R12),
                ("r13", ctx.R13),
                ("r14", ctx.R14),
                ("r15", ctx.R15),
            ];
            for (name, v) in regs {
                let a = v & !7;
                if (crate::mem::GUEST_BASE..crate::mem::GUEST_END).contains(&a) && readable(a) {
                    let q: Vec<String> = (0..8).map(|i| format!("{:016x}", *((a + i * 8) as *const u64))).collect();
                    let _ = writeln!(r, "  [{name}] {a:#x}: {}", q.join(" "));
                }
            }
            // Unmaps of the memory registers point into: a pointer read from
            // freed and reused memory faults elsewhere (the Nexus crash read
            // an owner pointer from memory holding float data).
            for (name, v) in regs {
                if (crate::mem::GUEST_BASE..crate::mem::GUEST_END).contains(&v) {
                    for l in crate::mem::unmaps_covering(v) {
                        let _ = writeln!(r, "  [{name}] unmapped earlier: {l}");
                    }
                    // Mirror mode: GPU-written dwords stored to its page.
                    if let Some(h) = crate::gpu::resident::writeback_history(v) {
                        let _ = writeln!(r, "  [{name}] mirror write-backs into its page: {h}");
                    }
                }
            }
            // The stack (the frame and its caller's links) and, on a fiber
            // stack, the fiber's context.
            if let Some(d) = crate::hle::libs::fiber::describe_stack(ctx.Rsp) {
                let _ = writeln!(r, "  stack: {d}");
            }
            for k in 0..4u64 {
                let a = (ctx.Rsp & !7) + k * 64;
                if !readable(a) {
                    break;
                }
                let q: Vec<String> = (0..8).map(|i| format!("{:016x}", *((a + i * 8) as *const u64))).collect();
                let _ = writeln!(r, "  [rsp+{:#x}] {a:#x}: {}", k * 64, q.join(" "));
            }
            // DS_CRASH_PEEK: comma-separated `base(+hex|*)*[:count]` expressions,
            // e.g. `rbx+b4c8*:8` (load the pointer at rbx+0xb4c8, dump 8 qwords).
            if let Ok(spec) = std::env::var("DS_CRASH_PEEK") {
                for item in spec.split(',').map(str::trim).filter(|s| !s.is_empty()) {
                    let (expr, count) = match item.split_once(':') {
                        Some((e, c)) => (e, u64::from_str_radix(c, 16).unwrap_or(8).min(64)),
                        None => (item, 8),
                    };
                    let mut chars = expr;
                    let base_end = chars.find(['+', '*']).unwrap_or(chars.len());
                    let base = &chars[..base_end];
                    let mut v = regs.iter().find(|(n, _)| *n == base).map(|(_, v)| *v).or_else(|| u64::from_str_radix(base.trim_start_matches("0x"), 16).ok());
                    chars = &chars[base_end..];
                    while let (Some(cur), false) = (v, chars.is_empty()) {
                        if let Some(rest) = chars.strip_prefix('*') {
                            v = if readable(cur & !7) { Some(*((cur & !7) as *const u64)) } else { None };
                            chars = rest;
                        } else if let Some(rest) = chars.strip_prefix('+') {
                            let end = rest.find(['+', '*']).unwrap_or(rest.len());
                            v = u64::from_str_radix(&rest[..end], 16).ok().map(|o| cur.wrapping_add(o));
                            chars = &rest[end..];
                        } else {
                            v = None;
                        }
                    }
                    match v {
                        Some(a) if readable(a & !7) => {
                            let a = a & !7;
                            let q: Vec<String> = (0..count).map(|i| format!("{:016x}", *((a + i * 8) as *const u64))).collect();
                            let _ = writeln!(r, "  peek {item} = {a:#x}: {}", q.join(" "));
                        }
                        _ => {
                            let _ = writeln!(r, "  peek {item}: unreadable");
                        }
                    }
                }
            }
        }
        if code != 0xC00000FD {
            // Exact host frames (unwind tables) down to the first guest frame,
            // then the guest frame-pointer chain (checked page by page, so it
            // also works on guest-allocated fiber stacks).
            let mut c: AlignedContext = std::mem::zeroed();
            c.0 = *ctx;
            let mut host = [0u64; 24];
            let mut guest = [0u64; 32];
            let (nh, ng) = unwind(&mut c, &mut host, &mut guest);
            if nh > 0 {
                let _ = writeln!(r, "  host frames:");
                for &a in &host[..nh] {
                    let _ = writeln!(r, "    {}", host_symbol(a));
                }
            }
            let _ = writeln!(r, "  guest frames:");
            for &a in &guest[..ng] {
                let _ = writeln!(r, "    {}", describe(a));
            }
        }
        crate::fatal_report(&r);
        crate::fatal_exit(0xD501, "fatal exception");
    }
}

#[cfg(test)]
mod tests {
    use windows_sys::Win32::System::Diagnostics::Debug::{AddVectoredExceptionHandler, RemoveVectoredExceptionHandler, EXCEPTION_POINTERS};
    use windows_sys::Win32::System::Memory::{VirtualAlloc, VirtualFree, VirtualProtect, MEM_COMMIT, MEM_RELEASE, MEM_RESERVE, PAGE_READONLY, PAGE_READWRITE};

    static PAGE: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);
    /// Held by each test that faults on `PAGE`: run side by side, one test's
    /// page replaced the other's and its fault went unhandled (a crashed
    /// test binary, FINDINGS.md 2026-11-01).
    static PAGE_TESTS: std::sync::Mutex<()> = std::sync::Mutex::new(());

    unsafe extern "system" fn unprotect(info: *mut EXCEPTION_POINTERS) -> i32 {
        let rec = unsafe { &*(*info).ExceptionRecord };
        let page = PAGE.load(std::sync::atomic::Ordering::Acquire);
        if rec.ExceptionCode == 0xC000_0005u32 as i32 && rec.ExceptionInformation[1] as u64 & !0xFFF == page {
            let mut old = 0;
            unsafe { VirtualProtect(page as _, 4096, PAGE_READWRITE, &mut old) };
            return -1; // EXCEPTION_CONTINUE_EXECUTION
        }
        0
    }

    /// A handled fault keeps the AVX registers' full width (the title's
    /// vector code faults on write-tracked pages mid-computation).
    #[test]
    fn handled_fault_keeps_ymm() {
        let _serial = PAGE_TESTS.lock().unwrap_or_else(|e| e.into_inner());
        let page = unsafe { VirtualAlloc(std::ptr::null(), 4096, MEM_COMMIT | MEM_RESERVE, PAGE_READONLY) } as u64;
        assert_ne!(page, 0);
        PAGE.store(page, std::sync::atomic::Ordering::Release);
        let h = unsafe { AddVectoredExceptionHandler(1, Some(unprotect)) };
        let src: [u64; 8] = [0x1111_2222_3333_4444, 0x5555_6666_7777_8888, 0x9999_aaaa_bbbb_cccc, 0xdddd_eeee_ffff_0001, 0x0102_0304_0506_0708, 0x1112_1314_1516_1718, 0x2122_2324_2526_2728, 0x3132_3334_3536_3738];
        let mut out = [0u64; 8];
        unsafe {
            std::arch::asm!(
                "vmovdqu ymm1, [{s}]",
                "vmovdqu ymm14, [{s} + 32]",
                "mov dword ptr [{p}], 1",
                "vmovdqu [{o}], ymm1",
                "vmovdqu [{o} + 32], ymm14",
                "vzeroupper",
                s = in(reg) src.as_ptr(),
                p = in(reg) page,
                o = in(reg) out.as_mut_ptr(),
                out("ymm1") _, out("ymm14") _,
            );
        }
        unsafe {
            RemoveVectoredExceptionHandler(h);
            VirtualFree(page as _, 0, MEM_RELEASE);
        }
        assert_eq!(out, src);
    }

    /// Which bytes below the stack pointer a handled fault's dispatch
    /// overwrites (System V code, the title, keeps data in the 128-byte red
    /// zone there without moving the stack pointer). Prints the offsets
    /// whose marker changed.
    #[test]
    fn handled_fault_below_the_stack_pointer() {
        let _serial = PAGE_TESTS.lock().unwrap_or_else(|e| e.into_inner());
        let page = unsafe { VirtualAlloc(std::ptr::null(), 4096, MEM_COMMIT | MEM_RESERVE, PAGE_READONLY) } as u64;
        assert_ne!(page, 0);
        PAGE.store(page, std::sync::atomic::Ordering::Release);
        let h = unsafe { AddVectoredExceptionHandler(1, Some(unprotect)) };
        // Markers at rsp-8 .. rsp-4096 (stack memory already touched: a
        // large local first), copied out after the fault.
        let mut out = [0u64; 512];
        let mut probe = [0u8; 16384];
        std::hint::black_box(&mut probe);
        unsafe {
            std::arch::asm!(
                "mov rcx, 1",
                "2:",
                "mov rax, rsp",
                "shl rcx, 3",
                "sub rax, rcx",
                "shr rcx, 3",
                "mov rdx, rcx",
                "mov r8, 0x5a5a000000000000",
                "or rdx, r8",
                "mov [rax], rdx",
                "inc rcx",
                "cmp rcx, 513",
                "jne 2b",
                "mov dword ptr [{p}], 1",
                "mov rcx, 1",
                "3:",
                "mov rax, rsp",
                "shl rcx, 3",
                "sub rax, rcx",
                "shr rcx, 3",
                "mov rdx, [rax]",
                "mov [{o} + rcx * 8 - 8], rdx",
                "inc rcx",
                "cmp rcx, 513",
                "jne 3b",
                p = in(reg) page,
                o = in(reg) out.as_mut_ptr(),
                out("rax") _, out("rcx") _, out("rdx") _, out("r8") _,
            );
        }
        unsafe {
            RemoveVectoredExceptionHandler(h);
            VirtualFree(page as _, 0, MEM_RELEASE);
        }
        let changed: Vec<usize> = (1..=512).filter(|&k| out[k - 1] != (0x5a5a_0000_0000_0000 | k as u64)).map(|k| k * 8).collect();
        eprintln!("bytes below rsp overwritten by the fault (offsets): {} of 4096; first {:?}, last {:?}", changed.len() * 8, changed.first(), changed.last());
    }
}
