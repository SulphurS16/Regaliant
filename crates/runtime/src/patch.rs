//! Load-time code patching of guest modules.
//!
//! * `mov r64, fs:[0]` (the TCB self pointer) is rewritten in place to
//!   `mov r64, gs:[TEB.TlsSlots[slot]]`, where the slot holds the guest thread
//!   pointer. Same length, no trampolines.
//! * `mov r64, fs:[disp]` with disp != 0 jumps to a trampoline that loads the
//!   thread pointer and then the field.
//! * `syscall` becomes `ud2`, so a raw FreeBSD syscall can never reach the Windows
//!   kernel; the exception handler reports the site if one is ever executed.
//! * Any other FS-relative instruction (PS5 libc stores error codes with
//!   `mov dword ptr fs:[28h], imm32`) jumps to a generic trampoline that
//!   executes it against the guest thread pointer (`fs_trampoline`); forms
//!   that cannot be rewritten are listed at load time.
//! * Diagnostic hooks (`DS_TEXT_HOOK`): a `jmp` to a trampoline that saves the
//!   registers, calls a host function with them, replays the displaced
//!   instructions and jumps back (`hook`).
//! * Title fixes (`titlefix`): a `jmp` to a trampoline that runs inserted
//!   instructions, then the displaced ones, and jumps back (`insert`).

use iced_x86::{Decoder, DecoderOptions, FlowControl, Instruction, Mnemonic, OpKind, Register};

#[derive(Default, Debug)]
pub struct PatchStats {
    pub functions: usize,
    pub fs_zero: usize,
    pub fs_raw_extra: usize,
    pub fs_disp: usize,
    /// Other FS-relative instructions rewritten through generic trampolines.
    pub fs_tramp: usize,
    pub fs_unknown: Vec<(u64, String, (u64, u64))>,
    pub syscalls: Vec<u64>,
}

/// TEB offset of TlsSlots[slot] on x64.
pub fn teb_tls_slot_offset(slot: u32) -> u32 {
    0x1480 + slot * 8
}

fn gpr64_index(r: Register) -> Option<u8> {
    use Register::*;
    Some(match r {
        RAX => 0,
        RCX => 1,
        RDX => 2,
        RBX => 3,
        RSP => 4,
        RBP => 5,
        RSI => 6,
        RDI => 7,
        R8 => 8,
        R9 => 9,
        R10 => 10,
        R11 => 11,
        R12 => 12,
        R13 => 13,
        R14 => 14,
        R15 => 15,
        _ => return Option::None,
    })
}

/// Encodes `mov reg64, gs:[disp32]`.
fn enc_mov_gs(reg: u8, disp: u32) -> Vec<u8> {
    let rex = 0x48 | if reg >= 8 { 0x04 } else { 0 };
    let mut v = vec![0x65, rex, 0x8B, ((reg & 7) << 3) | 0x04, 0x25];
    v.extend_from_slice(&disp.to_le_bytes());
    v
}

/// Encodes `mov reg64, [reg64 + disp32]`.
fn enc_mov_load(reg: u8, disp: i32) -> Vec<u8> {
    let rex = 0x48 | if reg >= 8 { 0x05 } else { 0 };
    let mut v = vec![rex, 0x8B, 0x80 | ((reg & 7) << 3) | (reg & 7)];
    if reg & 7 == 4 {
        v.push(0x24);
    }
    v.extend_from_slice(&disp.to_le_bytes());
    v
}

fn enc_jmp(from: u64, to: u64) -> Option<[u8; 5]> {
    let rel = to as i64 - (from as i64 + 5);
    let rel32 = i32::try_from(rel).ok()?;
    let mut b = [0xE9, 0, 0, 0, 0];
    b[1..].copy_from_slice(&rel32.to_le_bytes());
    Some(b)
}

pub struct Patcher<'a> {
    /// Host-writable view of the module image, indexed by module-relative vaddr.
    pub image: &'a mut [u8],
    pub base: u64,
    pub tls_slot_disp: u32,
    /// Trampoline space (absolute address + writable slice) near the module.
    pub tramp_base: u64,
    pub tramp: &'a mut [u8],
    pub tramp_used: usize,
}

impl<'a> Patcher<'a> {
    fn emit_trampoline(&mut self, code: &[u8]) -> Option<u64> {
        if self.tramp_used + code.len() > self.tramp.len() {
            return None;
        }
        let at = self.tramp_base + self.tramp_used as u64;
        self.tramp[self.tramp_used..self.tramp_used + code.len()].copy_from_slice(code);
        self.tramp_used = (self.tramp_used + code.len() + 15) & !15;
        Some(at)
    }

    /// Generic rewrite of an FS-relative instruction: a 5-byte `jmp` to a
    /// trampoline that loads the guest thread pointer into a scratch register
    /// the instruction does not use, executes the instruction with that
    /// register as the base (FS prefix dropped), restores the scratch register
    /// and jumps back. Flags are untouched (lea/push/pop/mov/jmp only) and the
    /// System V red zone below rsp is skipped before pushing.
    fn fs_trampoline(&mut self, ins: &Instruction) -> Result<(), String> {
        use iced_x86::{BlockEncoder, BlockEncoderOptions, Code, FlowControl, InstructionBlock, InstructionInfoFactory, MemoryOperand};
        if ins.len() < 5 {
            return Err(format!("{} bytes: too short for a jump", ins.len()));
        }
        if ins.flow_control() != FlowControl::Next {
            return Err("control-flow instruction".into());
        }
        if ins.is_ip_rel_memory_operand() {
            return Err("rip-relative".into());
        }
        let mut info = InstructionInfoFactory::new();
        let used: Vec<Register> = info.info(ins).used_registers().iter().map(|u| u.register().full_register()).collect();
        if used.contains(&Register::RSP) || ins.memory_base() == Register::RSP || ins.memory_index() == Register::RSP {
            return Err("uses rsp".into());
        }
        let candidates = [
            Register::RAX, Register::RCX, Register::RDX, Register::RBX, Register::RSI, Register::RDI, Register::R8, Register::R9,
            Register::R10, Register::R11, Register::R12, Register::R13, Register::R14, Register::R15,
        ];
        let scratch = *candidates
            .iter()
            .find(|r| !used.contains(r) && ins.memory_base().full_register() != **r && ins.memory_index().full_register() != **r)
            .ok_or("no free scratch register")?;
        let e = |r: Result<Instruction, iced_x86::IcedError>| r.map_err(|e| e.to_string());
        let mut list = vec![
            e(Instruction::with2(Code::Lea_r64_m, Register::RSP, MemoryOperand::with_base_displ(Register::RSP, -0x80)))?,
            e(Instruction::with1(Code::Push_r64, scratch))?,
            e(Instruction::with2(
                Code::Mov_r64_rm64,
                scratch,
                MemoryOperand::new(Register::None, Register::None, 1, self.tls_slot_disp as i64, 4, false, Register::GS),
            ))?,
        ];
        let mut rewritten = *ins;
        rewritten.set_segment_prefix(Register::None);
        if ins.memory_base() != Register::None {
            // tp + base first; the instruction keeps index*scale + disp.
            list.push(e(Instruction::with2(Code::Lea_r64_m, scratch, MemoryOperand::new(scratch, ins.memory_base(), 1, 0, 0, false, Register::None)))?);
        }
        if matches!(
            ins.code(),
            Code::Mov_AL_moffs8
                | Code::Mov_AX_moffs16
                | Code::Mov_EAX_moffs32
                | Code::Mov_RAX_moffs64
                | Code::Mov_moffs8_AL
                | Code::Mov_moffs16_AX
                | Code::Mov_moffs32_EAX
                | Code::Mov_moffs64_RAX
        ) {
            // moffs forms (A0-A3) cannot take a base register.
            return Err("absolute moffs form".into());
        }
        let disp = rewritten.memory_displacement64() as i64;
        if i32::try_from(disp).is_err() {
            return Err(format!("displacement {disp:#x} does not fit 32 bits"));
        }
        rewritten.set_memory_base(scratch);
        // Absolute [disp32] decodes with an 8-byte displacement size in 64-bit
        // mode; relative to a base register it is a sign-extended disp32.
        rewritten.set_memory_displ_size(if disp == 0 { 0 } else { 4 });
        // Linux cannot tolerate an unpatched FS access (FS belongs to libc). iced-x86's
        // displacement size describes the address size here: 8 for a sign-extended disp32
        // with a 64-bit base. Keep the existing Windows patch policy unchanged in this port.
        #[cfg(target_os = "linux")]
        rewritten.set_memory_displ_size(if disp == 0 { 0 } else { 8 });
        list.push(rewritten);
        list.push(e(Instruction::with1(Code::Pop_r64, scratch))?);
        list.push(e(Instruction::with2(Code::Lea_r64_m, Register::RSP, MemoryOperand::with_base_displ(Register::RSP, 0x80)))?);
        list.push(e(Instruction::with_branch(Code::Jmp_rel32_64, ins.next_ip()))?);
        let at = self.tramp_base + self.tramp_used as u64;
        let enc = BlockEncoder::encode(64, InstructionBlock::new(&list, at), BlockEncoderOptions::NONE).map_err(|e| e.to_string())?;
        let t = self.emit_trampoline(&enc.code_buffer).ok_or("trampoline space exhausted")?;
        debug_assert_eq!(t, at);
        let j = enc_jmp(ins.ip(), t).ok_or("trampoline out of rel32 range")?;
        let off = (ins.ip() - self.base) as usize;
        self.image[off..off + 5].copy_from_slice(&j);
        for b in &mut self.image[off + 5..off + ins.len()] {
            *b = 0xCC; // never executed: control returns to next_ip
        }
        Ok(())
    }

    fn handle_fs(&mut self, ins: &Instruction, range: (u64, u64), stats: &mut PatchStats, fmt: &mut iced_x86::FastFormatter) {
        let rel = ins.ip() - self.base;
        let off = rel as usize;
        let len = ins.len();
        let bytes = self.image[off..off + len].to_vec();
        // `mov r64, fs:[disp32]`, possibly after padding prefixes (PS5 TLS
        // relaxation emits `66 66 66 64 48 8b 04 25 ...`): the last 9 bytes.
        let tail = len.checked_sub(9).map(|p| &bytes[p..]).unwrap_or(&[]);
        let simple_abs = ins.mnemonic() == Mnemonic::Mov
            && ins.op_count() == 2
            && ins.op0_kind() == OpKind::Register
            && ins.op1_kind() == OpKind::Memory
            && ins.memory_base() == Register::None
            && ins.memory_index() == Register::None
            && gpr64_index(ins.op0_register()).is_some()
            && len >= 9
            && bytes[..len - 9].iter().all(|&b| b == 0x66)
            && tail[0] == 0x64
            && (tail[1] & 0xF8) == 0x48
            && tail[2] == 0x8B
            && (tail[3] & 0xC7) == 0x04
            && tail[4] == 0x25;
        if simple_abs {
            let reg = gpr64_index(ins.op0_register()).unwrap();
            let disp = ins.memory_displacement64();
            if disp == 0 {
                let new = enc_mov_gs(reg, self.tls_slot_disp);
                self.image[off + len - 9..off + len].copy_from_slice(&new);
                stats.fs_zero += 1;
                return;
            }
            // fs:[disp] -> trampoline: mov reg, gs:[slot]; mov reg, [reg+disp]; jmp back
            let mut code = enc_mov_gs(reg, self.tls_slot_disp);
            code.extend(enc_mov_load(reg, disp as i32));
            let jmp_pos = self.tramp_base + self.tramp_used as u64 + code.len() as u64;
            if let Some(j) = enc_jmp(jmp_pos, ins.next_ip()) {
                code.extend_from_slice(&j);
                if let Some(t) = self.emit_trampoline(&code) {
                    if let Some(j2) = enc_jmp(ins.ip(), t) {
                        self.image[off..off + 5].copy_from_slice(&j2);
                        for b in &mut self.image[off + 5..off + 9] {
                            *b = 0xCC; // never executed: control returns to next_ip
                        }
                        stats.fs_disp += 1;
                        return;
                    }
                }
            }
        }
        let mut text = String::new();
        fmt.format(ins, &mut text);
        match self.fs_trampoline(ins) {
            Ok(()) => stats.fs_tramp += 1,
            Err(why) => {
                // Not patched: usually bytes that are not instructions (decoder
                // desynchronized inside a function range). Recorded for fault diagnostics.
                stats.fs_unknown.push((rel, format!("{text} ({why})"), range));
            }
        }
    }

    /// Inline hook at module-relative `rel`: `handler(regs, arg)` runs before
    /// the instructions there, with `regs` the 16 general registers in
    /// encoding order (rax, rcx, rdx, rbx, rsp, rbp, rsi, rdi, r8..r15) as they
    /// were at the hook (rsp as seen by the trampoline). Flags and the SSE
    /// state (fxsave) are preserved. The displaced instructions must not be
    /// RIP-relative or branches, and no branch in the function may target their
    /// interior.
    pub fn hook(&mut self, rel: u64, ranges: &[(u64, u64)], handler: u64, arg: u32) -> Result<u64, String> {
        use iced_x86::code_asm::*;
        let len = self.displaced_len(rel, ranges)?;
        let at = self.base + rel;
        let tramp_at = self.tramp_base + self.tramp_used as u64;
        let back = at + len as u64;
        let asm = || -> Result<Vec<u8>, IcedError> {
            let mut a = CodeAssembler::new(64)?;
            a.lea(rsp, ptr(rsp - 0x80))?; // System V red zone
            for r in [r15, r14, r13, r12, r11, r10, r9, r8, rdi, rsi, rbp, rsp, rbx, rdx, rcx, rax] {
                a.push(r)?;
            }
            a.pushfq()?;
            a.mov(rbx, rsp)?;
            a.and(rsp, -16)?;
            a.sub(rsp, 512)?;
            a.fxsave64(ptr(rsp))?;
            a.lea(rdi, ptr(rbx + 8))?;
            a.mov(esi, arg)?;
            a.mov(rax, handler)?;
            a.call(rax)?;
            a.fxrstor64(ptr(rsp))?;
            a.mov(rsp, rbx)?;
            a.popfq()?;
            for r in [rax, rcx, rdx, rbx] {
                a.pop(r)?;
            }
            a.lea(rsp, ptr(rsp + 8))?; // saved rsp
            for r in [rbp, rsi, rdi, r8, r9, r10, r11, r12, r13, r14, r15] {
                a.pop(r)?;
            }
            a.lea(rsp, ptr(rsp + 0x80))?;
            let start = (at - self.base) as usize;
            a.db(&self.image[start..start + len])?;
            a.jmp(back)?;
            a.assemble(tramp_at)
        };
        let bytes = asm().map_err(|e| format!("assemble: {e}"))?;
        self.divert(rel, len, &bytes, tramp_at)
    }

    /// Inline insertion at module-relative `rel`: `code` (machine code that
    /// does not depend on its address) runs before the instructions there,
    /// which follow it in a trampoline. The constraints of `hook` apply;
    /// `code` must leave the registers and the stack as it found them, and
    /// may change flags only where the displaced instructions set them
    /// before any use.
    pub fn insert(&mut self, rel: u64, ranges: &[(u64, u64)], code: &[u8]) -> Result<u64, String> {
        use iced_x86::code_asm::*;
        let len = self.displaced_len(rel, ranges)?;
        let at = self.base + rel;
        let tramp_at = self.tramp_base + self.tramp_used as u64;
        let start = rel as usize;
        let asm = || -> Result<Vec<u8>, IcedError> {
            let mut a = CodeAssembler::new(64)?;
            a.db(code)?;
            a.db(&self.image[start..start + len])?;
            a.jmp(at + len as u64)?;
            a.assemble(tramp_at)
        };
        let bytes = asm().map_err(|e| format!("assemble: {e}"))?;
        self.divert(rel, len, &bytes, tramp_at)
    }

    /// The module image as patched so far (module-relative).
    pub fn image_view(&self) -> &[u8] {
        self.image
    }

    /// Replaces the instruction at module-relative `rel` with `code`
    /// (position-independent): a jump to a trampoline that runs `code`, then
    /// the other instructions the jump displaces, and jumps back. The
    /// constraints of `insert` apply.
    pub fn replace(&mut self, rel: u64, ranges: &[(u64, u64)], code: &[u8]) -> Result<u64, String> {
        use iced_x86::code_asm::*;
        let len = self.displaced_len(rel, ranges)?;
        let first = {
            let mut d = Decoder::with_ip(64, &self.image[rel as usize..], self.base + rel, DecoderOptions::NONE);
            d.decode().len()
        };
        let at = self.base + rel;
        let tramp_at = self.tramp_base + self.tramp_used as u64;
        let start = rel as usize;
        let asm = || -> Result<Vec<u8>, IcedError> {
            let mut a = CodeAssembler::new(64)?;
            a.db(code)?;
            a.db(&self.image[start + first..start + len])?;
            a.jmp(at + len as u64)?;
            a.assemble(tramp_at)
        };
        let bytes = asm().map_err(|e| format!("assemble: {e}"))?;
        self.divert(rel, len, &bytes, tramp_at)
    }

    /// Emits trampoline `bytes` (assembled for `tramp_at`, the next free
    /// trampoline address) and replaces the `len` bytes at `rel` with a jump
    /// to it, padded with nops.
    fn divert(&mut self, rel: u64, len: usize, bytes: &[u8], tramp_at: u64) -> Result<u64, String> {
        let t = self.emit_trampoline(bytes).ok_or("trampoline space exhausted")?;
        debug_assert_eq!(t, tramp_at);
        let j = enc_jmp(self.base + rel, t).ok_or("trampoline out of rel32 range")?;
        let off = rel as usize;
        self.image[off..off + 5].copy_from_slice(&j);
        for b in &mut self.image[off + 5..off + len] {
            *b = 0x90;
        }
        Ok(t)
    }

    /// The length of the instructions at module-relative `rel` that a 5-byte
    /// jump displaces. They must not be RIP-relative or branches, and no
    /// branch in the function may target their interior.
    fn displaced_len(&self, rel: u64, ranges: &[(u64, u64)]) -> Result<usize, String> {
        let &(fstart, fend) = ranges.iter().find(|r| r.0 <= rel && rel < r.1).ok_or("not inside a known function")?;
        let code = &self.image[fstart as usize..fend as usize];
        let mut targets = Vec::new();
        let mut displaced: Vec<Instruction> = Vec::new();
        let mut dec = Decoder::with_ip(64, code, self.base + fstart, DecoderOptions::NONE);
        let mut ins = Instruction::default();
        let at = self.base + rel;
        while dec.can_decode() {
            dec.decode_out(&mut ins);
            if matches!(ins.flow_control(), FlowControl::ConditionalBranch | FlowControl::UnconditionalBranch) && ins.op0_kind() == OpKind::NearBranch64 {
                targets.push(ins.near_branch64());
            }
            if ins.ip() >= at && displaced.iter().map(|i| i.len()).sum::<usize>() < 5 {
                displaced.push(ins);
            }
        }
        let len: usize = displaced.iter().map(|i| i.len()).sum();
        if displaced.first().map(|i| i.ip()) != Some(at) || len < 5 {
            return Err("not an instruction boundary (or function too short)".into());
        }
        if displaced.iter().any(|i| i.is_ip_rel_memory_operand() || i.flow_control() != FlowControl::Next) {
            return Err("displaced instructions are RIP-relative or change control flow".into());
        }
        if targets.iter().any(|&t| t > at && t < at + len as u64) {
            return Err("a branch targets the patched bytes".into());
        }
        Ok(len)
    }

    /// Decodes each function range and applies patches. `ranges` are module-relative.
    pub fn patch_functions(&mut self, ranges: &[(u64, u64)], stats: &mut PatchStats) {
        let mut fmt = iced_x86::FastFormatter::new();
        let mut ins = Instruction::default();
        for &(start, end) in ranges {
            if end as usize > self.image.len() || start >= end {
                continue;
            }
            stats.functions += 1;
            let code = self.image[start as usize..end as usize].to_vec();
            let mut dec = Decoder::with_ip(64, &code, self.base + start, DecoderOptions::NONE);
            let mut fs_sites = Vec::new();
            let mut sys_sites = Vec::new();
            while dec.can_decode() {
                dec.decode_out(&mut ins);
                if ins.is_invalid() {
                    continue;
                }
                if ins.segment_prefix() == Register::FS || (ins.op_count() > 0 && has_fs_mem(&ins)) {
                    fs_sites.push(ins);
                } else if ins.mnemonic() == Mnemonic::Syscall {
                    sys_sites.push(ins.ip() - self.base);
                }
            }
            for s in fs_sites {
                self.handle_fs(&s, (start, end), stats, &mut fmt);
            }
            for rel in sys_sites {
                let off = rel as usize;
                self.image[off] = 0x0F;
                self.image[off + 1] = 0x0B;
                stats.syscalls.push(rel);
            }
        }
    }

    /// Byte-pattern pass for `mov r64, fs:[0]` outside FDE-described code.
    pub fn patch_raw_fs_zero(&mut self, code_end: u64, stats: &mut PatchStats) {
        let n = (code_end as usize).min(self.image.len());
        let mut i = 0usize;
        while i + 9 <= n {
            let b = &self.image[i..i + 9];
            if b[0] == 0x64
                && (b[1] == 0x48 || b[1] == 0x4C)
                && b[2] == 0x8B
                && (b[3] & 0xC7) == 0x04
                && b[4] == 0x25
                && b[5..9] == [0, 0, 0, 0]
            {
                let reg = ((b[3] >> 3) & 7) | if b[1] == 0x4C { 8 } else { 0 };
                let new = enc_mov_gs(reg, self.tls_slot_disp);
                self.image[i..i + 9].copy_from_slice(&new);
                stats.fs_raw_extra += 1;
                i += 9;
            } else {
                i += 1;
            }
        }
    }
}

fn has_fs_mem(ins: &Instruction) -> bool {
    (0..ins.op_count()).any(|i| ins.op_kind(i) == OpKind::Memory) && ins.memory_segment() == Register::FS
}

/// Debug aid: disassembles a function range around `site` (module-relative).
pub fn dump_around(base: u64, range: (u64, u64), site: u64) {
    let code = unsafe { std::slice::from_raw_parts((base + range.0) as *const u8, (range.1 - range.0) as usize) };
    let mut dec = Decoder::with_ip(64, code, base + range.0, DecoderOptions::NONE);
    let mut fmt = iced_x86::FastFormatter::new();
    let mut ins = Instruction::default();
    let mut out = String::new();
    while dec.can_decode() {
        dec.decode_out(&mut ins);
        let rel = ins.ip() - base;
        if rel + 96 >= site && rel <= site + 32 {
            let mut t = String::new();
            fmt.format(&ins, &mut t);
            let off = (ins.ip() - base - range.0) as usize;
            let bytes: String = code[off..off + ins.len()].iter().map(|b| format!("{b:02x}")).collect();
            out.push_str(&format!("    {:#x}: {:<24} {}\n", rel, bytes, t));
        }
    }
    crate::info!("function {:#x}..{:#x} around {:#x}:\n{}", range.0, range.1, site, out);
}

#[cfg(all(test, target_os = "linux"))]
mod linux_tests {
    use super::*;
    use crate::platform::{tls, vm};

    #[test]
    fn patched_tls_load_and_error_store_run_without_changing_host_fs() {
        // mov rax,fs:[0]; mov dword ptr fs:[28h],0A0020307h; ret
        let original = [0x64, 0x48, 0x8b, 0x04, 0x25, 0, 0, 0, 0,
            0x64, 0xc7, 0x04, 0x25, 0x28, 0, 0, 0, 7, 3, 2, 0xa0, 0xc3];
        let area = vm::Reservation::new_at(0, 8192).unwrap();
        area.protect(0, 8192, vm::READ | vm::WRITE).unwrap();
        let bytes = unsafe { std::slice::from_raw_parts_mut(area.base() as *mut u8, 8192) };
        bytes[..original.len()].copy_from_slice(&original);
        let (image, tramp) = bytes.split_at_mut(4096);
        let mut p = Patcher { image, base: area.base() as u64, tls_slot_disp: teb_tls_slot_offset(0), tramp_base: area.base() as u64 + 4096, tramp, tramp_used: 0 };
        let mut stats = PatchStats::default();
        p.patch_functions(&[(0, original.len() as u64)], &mut stats);
        assert!(stats.fs_unknown.is_empty(), "{:?}", stats.fs_unknown);
        assert_eq!(stats.fs_zero, 1);
        assert_eq!(stats.fs_tramp, 1);
        area.protect(0, 8192, vm::READ | vm::EXEC).unwrap();
        let mut tcb = [0u64; 16];
        let tp = tcb.as_mut_ptr() as u64;
        tcb[0] = tp;
        let previous = tls::get(0);
        tls::set(0, tp);
        let f: unsafe extern "sysv64" fn() -> u64 = unsafe { std::mem::transmute(area.base()) };
        let result = unsafe { f() };
        tls::set(0, previous);
        assert_eq!(result, tp);
        assert_eq!(tcb[5] as u32, 0xa0020307);
    }

    #[test]
    fn guest_syscalls_become_traps() {
        let mut image = [0x0f, 0x05, 0xc3];
        let mut tramp = [0; 32];
        let mut p = Patcher { image: &mut image, base: 0x10000, tls_slot_disp: teb_tls_slot_offset(0), tramp_base: 0x20000, tramp: &mut tramp, tramp_used: 0 };
        let mut stats = PatchStats::default();
        p.patch_functions(&[(0, 3)], &mut stats);
        assert_eq!(stats.syscalls, [0]);
        assert_eq!(image, [0x0f, 0x0b, 0xc3]);
    }
}
