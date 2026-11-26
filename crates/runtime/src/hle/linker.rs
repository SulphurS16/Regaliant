//! Shared native import registry and unresolved-import diagnostics.
use parking_lot::RwLock;
use std::collections::HashMap;
use std::sync::OnceLock;
use std::sync::atomic::{AtomicBool, Ordering};

#[derive(Clone, Debug)]
pub struct Export {
    pub addr: u64,
    pub name: String,
    pub is_data: bool,
}

#[derive(Default)]
pub struct Registry {
    pub by_lib: HashMap<String, HashMap<String, Export>>,
}

impl Registry {
    pub fn func(&mut self, lib: &str, name: &str, addr: u64) {
        let nid = sce_elf::nid::nid_for_name(name);
        self.by_lib
            .entry(lib.to_string())
            .or_default()
            .insert(nid, Export { addr, name: name.to_string(), is_data: false });
    }
    pub fn func_nid(&mut self, lib: &str, nid: &str, label: &str, addr: u64) {
        self.by_lib
            .entry(lib.to_string())
            .or_default()
            .insert(nid.to_string(), Export { addr, name: label.to_string(), is_data: false });
    }
    pub fn data(&mut self, lib: &str, name: &str, addr: u64) {
        let nid = sce_elf::nid::nid_for_name(name);
        self.by_lib
            .entry(lib.to_string())
            .or_default()
            .insert(nid, Export { addr, name: name.to_string(), is_data: true });
    }

    /// Libraries exported by the same system module share one HLE table; PS5
    /// "*.native"/"*_native" library names alias their plain names.
    fn canonical(lib: &str) -> &str {
        match lib {
            "libScePosix" | "libkernel_unity" | "libkernel_cpumode_platform" | "libkernel_pre250mmap"
            | "libkernel_ps2emu" | "libkernel_exception" | "libSceCoredump" | "libkernel_sysc_se" => "libkernel",
            "libSceSaveData_native" => "libSceSaveData",
            "libSceSaveDataDialog.native" => "libSceSaveDataDialog",
            "libSceMsgDialog.native" => "libSceMsgDialog",
            other => other,
        }
    }

    pub fn lookup(&self, lib: &str, nid: &str) -> Option<&Export> {
        if let Some(e) = self.by_lib.get(lib).and_then(|m| m.get(nid)) {
            return Some(e);
        }
        let c = Self::canonical(lib);
        if c != lib {
            if let Some(e) = self.by_lib.get(c).and_then(|m| m.get(nid)) {
                return Some(e);
            }
        }
        // Functions present in both the kernel and POSIX tables.
        if c == "libkernel" {
            return self.by_lib.get("libScePosix").and_then(|m| m.get(nid));
        }
        None
    }
}

static REGISTRY: OnceLock<Registry> = OnceLock::new();

pub fn registry() -> &'static Registry {
    REGISTRY.get_or_init(|| {
        let mut r = Registry::default();
        super::register_platform(&mut r);
        r
    })
}

/// NID -> name table for diagnostics (data/ps4-names.txt + data/ps5-names.txt).
pub fn nid_names() -> &'static HashMap<String, String> {
    static NAMES: OnceLock<HashMap<String, String>> = OnceLock::new();
    NAMES.get_or_init(|| {
        let mut m = HashMap::new();
        let lists = [include_str!("../../../../data/ps4-names.txt"), include_str!("../../../../data/ps5-names.txt")];
        for line in lists.iter().flat_map(|l| l.lines()) {
            let n = line.trim();
            if n.is_empty() || n.starts_with('#') || n.contains(' ') || n.ends_with('?') {
                continue;
            }
            m.insert(sce_elf::nid::nid_for_name(n), n.to_string());
        }
        m
    })
}

pub fn describe_nid(nid: &str) -> String {
    match nid_names().get(nid) {
        Some(n) => format!("{n} ({nid})"),
        None => nid.to_string(),
    }
}

// ---------------------------------------------------------------------------
// Unresolved imports

#[derive(Clone, Debug)]
pub struct Unresolved {
    pub nid: String,
    pub lib: String,
    pub importer: String,
    pub is_data: bool,
}

pub static UNRESOLVED: RwLock<Vec<Unresolved>> = RwLock::new(Vec::new());
pub static LENIENT_IMPORTS: AtomicBool = AtomicBool::new(false);

std::arch::global_asm!(
    ".globl ds_unresolved_entry",
    "ds_unresolved_entry:",
    "push rbp",
    "mov rbp, rsp",
    "sub rsp, 64",
    "and rsp, -16",
    "mov [rsp], rdi",
    "mov [rsp+8], rsi",
    "mov [rsp+16], rdx",
    "mov [rsp+24], rcx",
    "mov [rsp+32], r8",
    "mov [rsp+40], r9",
    "mov rdi, rax",
    "mov rsi, rsp",
    "mov rdx, [rbp+8]",
    "call {handler}",
    "mov rsp, rbp",
    "pop rbp",
    "ret",
    handler = sym unresolved_handler,
);

unsafe extern "C" {
    fn ds_unresolved_entry();
}

extern "sysv64" fn unresolved_handler(index: u64, args: *const u64, ret: u64) -> u64 {
    let info = UNRESOLVED.read().get(index as usize).cloned();
    let a = unsafe { std::slice::from_raw_parts(args, 6) };
    let (desc, lib, importer) = match &info {
        Some(u) => (describe_nid(&u.nid), u.lib.clone(), u.importer.clone()),
        None => (format!("#{index}"), "?".into(), "?".into()),
    };
    let caller = crate::loader::describe_addr(ret);
    if LENIENT_IMPORTS.load(Ordering::Relaxed) {
        crate::warn!(
            "UNIMPLEMENTED {desc} [{lib}] imported by {importer}, called from {caller}, args {:#x} {:#x} {:#x} {:#x} -> returning 0 (lenient)",
            a[0],
            a[1],
            a[2],
            a[3]
        );
        return 0;
    }
    crate::error!(
        "UNIMPLEMENTED {desc} [{lib}] imported by {importer}, called from {caller}, args {:#x} {:#x} {:#x} {:#x} {:#x} {:#x}",
        a[0],
        a[1],
        a[2],
        a[3],
        a[4],
        a[5]
    );
    crate::fatal_exit(0xD530, "unimplemented import called");
}

/// Builds a 24-byte trap stub: `mov rax, index; jmp [rip+0]; dq ds_unresolved_entry`.
pub fn unresolved_stub_code(index: u64) -> [u8; 24] {
    let mut c = [0u8; 24];
    c[0] = 0x48;
    c[1] = 0xB8;
    c[2..10].copy_from_slice(&index.to_le_bytes());
    c[10] = 0xFF;
    c[11] = 0x25;
    c[12..16].copy_from_slice(&0u32.to_le_bytes());
    c[16..24].copy_from_slice(&(ds_unresolved_entry as usize as u64).to_le_bytes());
    c
}

