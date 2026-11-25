//! Additional page-fault evidence, collected by the recovery service, never the signal handler.
//! Reserved host memory alone does not establish a guest allocation or permission to commit it.

fn read_words(address: u64, words: &mut [u64]) -> bool {
    let len = std::mem::size_of_val(words);
    if address.checked_add(len as u64).is_none() { return false; }
    let local = libc::iovec { iov_base: words.as_mut_ptr().cast(), iov_len: len };
    let remote = libc::iovec { iov_base: address as _, iov_len: len };
    unsafe { libc::process_vm_readv(libc::getpid(), &local, 1, &remote, 1, 0) == len as isize }
}

pub(super) fn unrecovered(address: u64, rip: u64, pointers: [u64; 4]) {
    let locations = [("fault", address), ("rbx", pointers[0]), ("rsi", pointers[1]),
        ("rdi", pointers[2]), ("rsp", pointers[3])];
    if let Some(memory) = crate::mem::MEM.try_lock() {
        for (name, at) in locations {
            crate::error!("fault context {name} {at:#x}: guest VMA {:?}", memory.find(at));
        }
        let before = memory.vmas.range(..=address).next_back().map(|(_, v)| v);
        let after = memory.vmas.range(address..).next().map(|(_, v)| v);
        crate::error!("fault neighboring VMAs: before={before:?}; after={after:?}");
    } else {
        crate::error!("fault guest VMAs unavailable: memory map lock busy (not evidence of an unmapped guest range)");
    }
    // Windows already includes these notes in its fatal report. The Linux signal handler
    // cannot lock or symbolize; the service can, before releasing the faulting thread.
    for (name, at) in locations {
        if !(crate::mem::GUEST_BASE..crate::mem::GUEST_END).contains(&at) { continue; }
        crate::error!("fault context {name} {at:#x}: file-read history {:?}", crate::platform::file_hooks::read_history(at));
        let notes = crate::mem::unmaps_covering(at);
        crate::error!("fault context {name} {at:#x}: recent unmap/protection notes {notes:?}");
        if let Some(history) = crate::gpu::resident::writeback_history(at) {
            crate::error!("fault context {name} {at:#x}: mirror write-backs {history}");
        }
    }
    // PPSA01341 01.007.000's four-byte stream read. This is observation only: a zero
    // available length may be an unchecked-stream sentinel, not necessarily corruption.
    // In the tour fault cursor=0x55fffe and available=0; the unsigned remaining count
    // wrapped, bypassing the refill branch. Do not recover by inventing zero-filled data.
    if rip == crate::mem::MODULE_BASE + 0x813a72 {
        let mut fields = [0u64; 7];
        if let Some(at) = pointers[0].checked_add(0x108) {
            if read_words(at, &mut fields) {
                let [cursor, available, refill, buffer, field128, consumed, field138] = fields;
                crate::error!("fault stream object {:#x}: cursor={cursor:#x} available={available:#x} remaining_unsigned={:#x} cursor_past_available={} refill_arg={refill:#x} buffer={buffer:#x} computed_source={:?} field128={field128:#x} consumed={consumed:#x} field138={field138:#x}",
                    pointers[0], available.wrapping_sub(cursor), cursor > available, buffer.checked_add(cursor));
                crate::error!("fault stream buffer: {}", crate::mem::describe_host(buffer, 1));
                let mut vtable = [0];
                if read_words(pointers[0], &mut vtable) {
                    let mut methods = [0u64; 4];
                    if read_words(vtable[0], &mut methods) {
                        crate::error!("fault stream vtable {:#x}: methods {methods:#x?}", vtable[0]);
                    }
                }
            } else {
                crate::error!("fault stream object unreadable at {:#x}", pointers[0]);
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn diagnostic_reads_do_not_fault_on_reserved_pages() {
        let memory = crate::platform::vm::Reservation::new_at(0, 8192).unwrap();
        memory.protect(0, 4096, crate::platform::vm::READ | crate::platform::vm::WRITE).unwrap();
        let mut words = [0; 2];
        assert!(read_words(memory.base() as u64, &mut words));
        assert_eq!(words, [0, 0]);
        assert!(!read_words(memory.base() as u64 + 4096, &mut words));
        assert!(!read_words(memory.base() as u64 + 4096 - 8, &mut words));
        assert!(!read_words(u64::MAX, &mut words));
    }
}
