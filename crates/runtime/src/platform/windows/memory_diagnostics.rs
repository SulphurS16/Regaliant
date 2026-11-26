//! Windows startup address-space diagnostics.
/// `DS_SIM_MODULE=<dll path>` (a test, with `DS_SIM_OCCUPIED=1`): that module loaded before the
/// reservation, as a tool another program puts into every process would be, for the failed
/// start's list of such modules (`foreign_modules`). Read without allocating (the heap may not be
/// set up yet).
pub fn sim_module() {
    use windows_sys::Win32::System::Environment::GetEnvironmentVariableW;
    use windows_sys::Win32::System::LibraryLoader::LoadLibraryW;
    const NAME: [u16; 14] = {
        let b = b"DS_SIM_MODULE\0";
        let mut a = [0u16; 14];
        let mut i = 0;
        while i < a.len() {
            a[i] = b[i] as u16;
            i += 1;
        }
        a
    };
    let mut buf = [0u16; 520];
    let len = unsafe { GetEnvironmentVariableW(NAME.as_ptr(), buf.as_mut_ptr(), buf.len() as u32) } as usize;
    if len > 0 && len < buf.len() {
        unsafe { LoadLibraryW(buf.as_ptr()) };
    }
}

/// The modules loaded in this process that are neither Windows' own (under its folder) nor this
/// program's (its folder): on a failed start, the likely owners of private memory in the guest
/// range, such as a tool another program puts into every process (issue #15). At most 16.
pub fn foreign_modules() -> Vec<String> {
    use windows_sys::Win32::Foundation::HMODULE;
    use windows_sys::Win32::System::LibraryLoader::GetModuleHandleW;
    use windows_sys::Win32::System::ProcessStatus::{K32EnumProcessModules, K32GetModuleFileNameExW};
    use windows_sys::Win32::System::SystemInformation::GetSystemWindowsDirectoryW;
    use windows_sys::Win32::System::Threading::GetCurrentProcess;
    let proc = unsafe { GetCurrentProcess() };
    // (The program itself, by its handle: its folder may be a drive's root, which the folder test
    // below leaves out, or unknown.)
    let exe = unsafe { GetModuleHandleW(std::ptr::null()) };
    let mut mods: [HMODULE; 512] = [std::ptr::null_mut(); 512];
    let mut needed = 0u32;
    if unsafe { K32EnumProcessModules(proc, mods.as_mut_ptr(), std::mem::size_of_val(&mods) as u32, &mut needed) } == 0 {
        return Vec::new();
    }
    let n = (needed as usize / std::mem::size_of::<HMODULE>()).min(mods.len());
    let mut wdir = [0u16; 260];
    let wl = unsafe { GetSystemWindowsDirectoryW(wdir.as_mut_ptr(), wdir.len() as u32) } as usize;
    let windows = if wl > 0 && wl < wdir.len() { format!("{}\\", String::from_utf16_lossy(&wdir[..wl]).trim_end_matches('\\').to_ascii_lowercase()) } else { String::new() };
    // (The program's folder, unless it is a drive's root: then every module on that drive would
    // be left out; the review of 2026-11-25 evening.)
    let own = std::env::current_exe().ok().and_then(|p| p.parent().map(|d| format!("{}\\", d.to_string_lossy().trim_end_matches('\\').to_ascii_lowercase()))).filter(|d| d.len() > 3);
    let mut out = Vec::new();
    for &m in &mods[..n] {
        if m == exe {
            continue;
        }
        let mut name = [0u16; 512];
        let len = unsafe { K32GetModuleFileNameExW(proc, m, name.as_mut_ptr(), name.len() as u32) } as usize;
        if len == 0 || len >= name.len() {
            continue;
        }
        let path = String::from_utf16_lossy(&name[..len]);
        let low = path.to_ascii_lowercase();
        if (!windows.is_empty() && low.starts_with(&windows)) || own.as_deref().is_some_and(|d| low.starts_with(d)) {
            continue;
        }
        out.push(crate::mem::users_folder_hidden(&crate::log::redact(&path)));
        if out.len() == 16 {
            break;
        }
    }
    out
}

