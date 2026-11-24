// x86-64 instruction and data caches are coherent. Called after writing code, before publishing
// its entry point to other threads (which must still use normal release/acquire synchronization).
pub unsafe fn flush_instruction_cache(_base: u64, _size: usize) {
    std::sync::atomic::fence(std::sync::atomic::Ordering::SeqCst);
}
