//! The processor the game's own code needs. The title was built for the PS5's Zen 2: its code
//! uses AVX, AVX2, FMA, F16C, BMI1, BMI2, LZCNT, MOVBE and POPCNT throughout (a scan of
//! 01.007.000's executable and libraries: `dsdis <module> --isa`), so a processor without one of
//! them stops at the first use. Intel processors from Haswell (2013) and AMD's from Zen (2017) on
//! have them. The executable also holds 12 SSE4A `insertq` instructions (AMD only): without SSE4A
//! they are rewritten at load into equivalent SSE2 and SSE4.1 code (`sse4a_rewrite`).

use iced_x86::{Decoder, DecoderOptions, Instruction, Mnemonic, OpKind, Register};

/// The extensions the game's code uses that a processor may lack, by name.
fn missing() -> Vec<&'static str> {
    let need = [
        ("AVX", std::arch::is_x86_feature_detected!("avx")),
        ("AVX2", std::arch::is_x86_feature_detected!("avx2")),
        ("FMA", std::arch::is_x86_feature_detected!("fma")),
        ("F16C", std::arch::is_x86_feature_detected!("f16c")),
        ("BMI1", std::arch::is_x86_feature_detected!("bmi1")),
        ("BMI2", std::arch::is_x86_feature_detected!("bmi2")),
        ("LZCNT", std::arch::is_x86_feature_detected!("lzcnt")),
        ("MOVBE", std::arch::is_x86_feature_detected!("movbe")),
        ("POPCNT", std::arch::is_x86_feature_detected!("popcnt")),
        ("SSE4.1", std::arch::is_x86_feature_detected!("sse4.1")),
        ("CMPXCHG16B", std::arch::is_x86_feature_detected!("cmpxchg16b")),
    ];
    // (Test hook: `DS_CPU_PRETEND_MISSING=<name>` takes that one as missing.)
    let pretend = std::env::var("DS_CPU_PRETEND_MISSING").unwrap_or_default();
    need.iter().filter(|(n, ok)| !ok || pretend.eq_ignore_ascii_case(n)).map(|(n, _)| *n).collect()
}

/// Checks the processor; without what the game needs the player is told and the game ends.
pub fn check() {
    let m = missing();
    if m.is_empty() {
        crate::info!("CPU: the extensions the game's code uses are all there (AVX, AVX2, FMA, F16C, BMI1, BMI2, LZCNT, MOVBE, POPCNT){}", if sse4a() { "" } else { "; no SSE4A: its insertq instructions are rewritten" });
        return;
    }
    crate::error!("CPU: lacks {} that the game's code uses", m.join(", "));
    crate::player_error(&format!(
        "This processor lacks instructions the game's code uses: {}.\n\nThe game cannot run on it. Intel processors from the Haswell generation (2013) on and AMD processors from Zen (2017) on have them.",
        m.join(", ")
    ));
    crate::log::flush();
    std::process::exit(crate::EXIT_CPU as i32);
}

/// Whether the processor has SSE4A (`DS_NO_SSE4A=1` takes it as missing: the rewrite tested on
/// an AMD processor).
pub fn sse4a() -> bool {
    std::env::var("DS_NO_SSE4A").ok().as_deref() != Some("1") && std::arch::is_x86_feature_detected!("sse4a")
}

fn xmm_index(r: Register) -> Option<u32> {
    let i = r as u32;
    let x0 = Register::XMM0 as u32;
    (i >= x0 && i < x0 + 16).then(|| i - x0)
}

/// The SSE4A instructions inside the functions `ranges` (module-relative), decoded from each
/// function's start: (offset, destination, source, length, index) of every `insertq` with
/// immediates; other SSE4A forms are reported.
fn sse4a_sites(image: &[u8], base: u64, ranges: &[(u64, u64)]) -> (Vec<(u64, u32, u32, u32, u32)>, Vec<(u64, String)>) {
    let (mut sites, mut other) = (Vec::new(), Vec::new());
    let mut ins = Instruction::default();
    for &(s, e) in ranges {
        let Some(code) = image.get(s as usize..e as usize) else { continue };
        let mut d = Decoder::with_ip(64, code, base + s, DecoderOptions::NONE);
        while d.can_decode() {
            d.decode_out(&mut ins);
            match ins.mnemonic() {
                Mnemonic::Insertq if ins.op_count() == 4 && ins.op2_kind() == OpKind::Immediate8 && ins.op3_kind() == OpKind::Immediate8_2nd => {
                    match (xmm_index(ins.op0_register()), xmm_index(ins.op1_register())) {
                        (Some(dst), Some(src)) => sites.push((ins.ip() - base, dst, src, ins.immediate8() as u32 & 0x3F, ins.immediate8_2nd() as u32 & 0x3F)),
                        _ => other.push((ins.ip() - base, format!("{ins}"))),
                    }
                }
                Mnemonic::Insertq | Mnemonic::Extrq | Mnemonic::Movntss | Mnemonic::Movntsd => other.push((ins.ip() - base, format!("{ins}"))),
                _ => {}
            }
        }
    }
    (sites, other)
}

/// The code that does what `insertq xmm<dst>, xmm<src>, len, idx` does: the low `len` bits of the
/// source's low quadword into the destination's low quadword at bit `idx`, the rest of it kept
/// (the upper quadword, undefined after `insertq`, kept too). Registers, flags and the red zone
/// below the stack pointer are left as they were.
fn insertq_code(dst: u32, src: u32, len: u32, idx: u32) -> Result<Vec<u8>, String> {
    use iced_x86::code_asm::*;
    let xmm = |i: u32| -> AsmRegisterXmm {
        [xmm0, xmm1, xmm2, xmm3, xmm4, xmm5, xmm6, xmm7, xmm8, xmm9, xmm10, xmm11, xmm12, xmm13, xmm14, xmm15][i as usize]
    };
    let mask: u64 = if len == 0 { u64::MAX } else { (1u64 << len) - 1 };
    let f = || -> Result<Vec<u8>, IcedError> {
        let mut a = CodeAssembler::new(64)?;
        a.lea(rsp, ptr(rsp - 128))?;
        a.push(rax)?;
        a.push(rcx)?;
        a.push(rdx)?;
        a.pushfq()?;
        a.movq(rax, xmm(src))?;
        a.mov(rcx, mask)?;
        a.and(rax, rcx)?;
        if idx != 0 {
            a.shl(rax, idx)?;
            a.shl(rcx, idx)?;
        }
        a.not(rcx)?;
        a.movq(rdx, xmm(dst))?;
        a.and(rdx, rcx)?;
        a.or(rdx, rax)?;
        a.pinsrq(xmm(dst), rdx, 0)?;
        a.popfq()?;
        a.pop(rdx)?;
        a.pop(rcx)?;
        a.pop(rax)?;
        a.lea(rsp, ptr(rsp + 128))?;
        // (Position-independent: `Patcher::replace` appends the jump back.)
        a.assemble(0)
    };
    f().map_err(|e| format!("assemble: {e}"))
}

/// Without SSE4A, the module's `insertq` instructions become jumps to equivalent code.
pub fn sse4a_rewrite(p: &mut crate::patch::Patcher, ranges: &[(u64, u64)], module: &str) {
    if sse4a() {
        return;
    }
    let (sites, other) = sse4a_sites(p.image_view(), p.base, ranges);
    if sites.is_empty() && other.is_empty() {
        return;
    }
    let mut done = 0;
    for &(rel, dst, src, len, idx) in &sites {
        match insertq_code(dst, src, len, idx).and_then(|code| p.replace(rel, ranges, &code)) {
            Ok(_) => done += 1,
            Err(e) => crate::error!("CPU: {module}+{rel:#x}: insertq xmm{dst},xmm{src},{len},{idx} not rewritten ({e}): it faults on this processor"),
        }
    }
    for (rel, s) in &other {
        crate::error!("CPU: {module}+{rel:#x}: {s} (SSE4A) has no rewrite: it faults on this processor");
    }
    crate::info!("CPU: no SSE4A: {done} of {} insertq instructions of {module} rewritten into SSE2 and SSE4.1 code", sites.len());
}

#[cfg(test)]
mod tests {
    /// The rewrite's arithmetic against `insertq`'s definition (low quadword).
    #[test]
    fn insertq_semantics() {
        fn insertq(d: u64, s: u64, len: u32, idx: u32) -> u64 {
            let mask = if len == 0 { u64::MAX } else { (1u64 << len) - 1 };
            (d & !(mask << idx)) | ((s & mask) << idx)
        }
        assert_eq!(insertq(0x0000_0000_4433_2211, 0x0000_0000_4433_2211, 8, 8), 0x0000_0000_4433_1111);
        assert_eq!(insertq(0xAAAA_BBBB_CCCC_DDDD, 0x1111_2222_3333_4444, 16, 16), 0xAAAA_BBBB_4444_DDDD);
        assert!(super::insertq_code(9, 4, 16, 16).is_ok());
        assert!(super::insertq_code(3, 3, 8, 8).is_ok());
    }

    /// The rewrite's machine code run natively: the destination's low quadword as `insertq`
    /// leaves it, its upper quadword, the other registers and the flags kept.
    #[test]
    fn insertq_code_runs() {
        use iced_x86::code_asm::*;
        #[cfg(windows)]
        use windows_sys::Win32::System::Memory::{MEM_COMMIT, MEM_RELEASE, MEM_RESERVE, PAGE_EXECUTE_READWRITE, VirtualAlloc, VirtualFree};
        fn reg(i: u32) -> AsmRegisterXmm {
            [xmm0, xmm1, xmm2, xmm3, xmm4, xmm5, xmm6, xmm7, xmm8, xmm9, xmm10, xmm11, xmm12, xmm13, xmm14, xmm15][i as usize]
        }
        for &(dst, src, len, idx) in &[(3u32, 3u32, 8u32, 8u32), (0, 0, 8, 8), (9, 4, 16, 16), (3, 4, 16, 16)] {
            // A harness (Win64: rcx -> [dst lo, dst hi, src lo, src hi, saved, saved, rax, rflags]):
            // the registers loaded, the rewrite run, the results and rax, rcx, rdx and the flags
            // stored back, xmm6-15 kept for the caller.
            let mut a = CodeAssembler::new(64).unwrap();
            a.sub(rsp, 40).unwrap();
            a.movdqu(xmmword_ptr(rsp), reg(dst)).unwrap();
            a.movdqu(xmmword_ptr(rsp + 16), reg(src)).unwrap();
            a.movdqu(reg(dst), xmmword_ptr(rcx)).unwrap();
            if src != dst {
                a.movdqu(reg(src), xmmword_ptr(rcx + 16)).unwrap();
            }
            a.mov(rax, 0x1234_5678_9abc_def0u64).unwrap();
            a.mov(rdx, 0x0fed_cba9_8765_4321u64).unwrap();
            a.stc().unwrap();
            let code = super::insertq_code(dst, src, len, idx).unwrap();
            a.db(&code).unwrap();
            a.setc(byte_ptr(rcx + 56)).unwrap();
            a.mov(qword_ptr(rcx + 48), rax).unwrap();
            a.mov(qword_ptr(rcx + 40), rdx).unwrap();
            a.movdqu(xmmword_ptr(rcx), reg(dst)).unwrap();
            a.movdqu(reg(dst), xmmword_ptr(rsp)).unwrap();
            a.movdqu(reg(src), xmmword_ptr(rsp + 16)).unwrap();
            a.add(rsp, 40).unwrap();
            a.ret().unwrap();
            #[cfg(windows)]
            let mem = unsafe { VirtualAlloc(std::ptr::null(), 4096, MEM_COMMIT | MEM_RESERVE, PAGE_EXECUTE_READWRITE) } as *mut u8;
            #[cfg(target_os = "linux")]
            let allocation = crate::platform::vm::Reservation::new_at(0, 4096).unwrap();
            #[cfg(target_os = "linux")]
            let mem = {
                allocation.protect(0, 4096, crate::platform::vm::READ | crate::platform::vm::WRITE | crate::platform::vm::EXEC).unwrap();
                allocation.base() as *mut u8
            };
            assert!(!mem.is_null());
            let bytes = a.assemble(mem as u64).unwrap();
            unsafe { std::ptr::copy_nonoverlapping(bytes.as_ptr(), mem, bytes.len()) };
            let f: extern "win64" fn(*mut u64) = unsafe { std::mem::transmute(mem) };
            let (d_lo, d_hi, s_lo, s_hi) = (0x8877_6655_4433_2211u64, 0xffee_ddcc_bbaa_9988u64, 0x0102_0304_0506_0708u64, 0x1112_1314_1516_1718u64);
            let mut v = [d_lo, d_hi, s_lo, s_hi, 0, 0, 0, 0];
            f(v.as_mut_ptr());
            #[cfg(windows)]
            unsafe { VirtualFree(mem as _, 0, MEM_RELEASE) };
            let src_lo = if src == dst { d_lo } else { s_lo };
            let mask = if len == 0 { u64::MAX } else { (1u64 << len) - 1 };
            let want = (d_lo & !(mask << idx)) | ((src_lo & mask) << idx);
            assert_eq!(v[0], want, "insertq xmm{dst},xmm{src},{len},{idx}: low quadword");
            assert_eq!(v[1], d_hi, "upper quadword kept");
            assert_eq!(v[6], 0x1234_5678_9abc_def0, "rax kept");
            assert_eq!(v[5], 0x0fed_cba9_8765_4321, "rdx kept");
            assert_eq!(v[7] & 1, 1, "carry flag kept");
        }
    }
}
