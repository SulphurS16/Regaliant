//! Fiber host stacks have a non-accessible page at each end.
use super::vm;
pub unsafe fn allocate(size: usize) -> u64 {
    let Ok(memory) = vm::Reservation::new_at(0, size + 2 * vm::PAGE) else { return 0 };
    if memory.protect(vm::PAGE, size, vm::READ | vm::WRITE).is_err() { return 0; }
    let base = memory.base() + vm::PAGE;
    std::mem::forget(memory);
    base as u64
}
pub unsafe fn free(base: u64, size: usize) {
    unsafe { libc::munmap((base as usize - vm::PAGE) as *mut _, size + 2 * vm::PAGE) };
}
