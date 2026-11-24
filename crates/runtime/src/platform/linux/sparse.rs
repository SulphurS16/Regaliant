//! Linux sparse-buffer submissions: combine consecutive mappings with identical backing.
//! The caller retains its per-page list for accounting/rollback. No binding wait changes.
//! Off by default (DS_LINUX_COALESCE_SPARSE=1 turns it on): the first version sorted the
//! binds by offset, which could reorder an unbind and a later rebind of the same page, and
//! the machine hard-froze in runs with it on. Binds are now merged only in the order given.
use ash::vk;
use std::sync::{OnceLock, atomic::{AtomicU64, Ordering::Relaxed}};
static INPUT: AtomicU64 = AtomicU64::new(0);
static OUTPUT: AtomicU64 = AtomicU64::new(0);
fn coalesce(input: &[vk::SparseMemoryBind]) -> Vec<vk::SparseMemoryBind> {
    let mut out: Vec<vk::SparseMemoryBind> = Vec::with_capacity(input.len());
    for &b in input {
        if let Some(a) = out.last_mut() {
            if a.memory != vk::DeviceMemory::null() && a.memory == b.memory && a.flags == b.flags
                && a.resource_offset.checked_add(a.size) == Some(b.resource_offset)
                && a.memory_offset.checked_add(a.size) == Some(b.memory_offset) {
                if let Some(size) = a.size.checked_add(b.size) { a.size = size; continue; }
            }
        }
        out.push(b);
    }
    out
}
pub fn binds(input: &[vk::SparseMemoryBind]) -> std::borrow::Cow<'_, [vk::SparseMemoryBind]> {
    static ON: OnceLock<bool> = OnceLock::new();
    if !*ON.get_or_init(|| std::env::var("DS_LINUX_COALESCE_SPARSE").ok().as_deref() == Some("1")) {
        return std::borrow::Cow::Borrowed(input);
    }
    let out = coalesce(input);
    INPUT.fetch_add(input.len() as u64, Relaxed);
    OUTPUT.fetch_add(out.len() as u64, Relaxed);
    std::borrow::Cow::Owned(out)
}
pub fn status() -> String { format!("Linux sparse buffer binds: {} page entries -> {} contiguous ranges", INPUT.load(Relaxed), OUTPUT.load(Relaxed)) }
#[cfg(test)]
mod tests {
    use super::*;
    use ash::vk::Handle;
    fn expand(v: &[vk::SparseMemoryBind]) -> Vec<(u64,u64,u64,u32)> {
        let mut r = Vec::new();
        for b in v { for off in (0..b.size).step_by(65536) { r.push((b.resource_offset+off,b.memory.as_raw(),b.memory_offset+off,b.flags.as_raw())); } }
        r.sort_unstable(); r
    }
    #[test]
    fn coalescing_preserves_every_page_and_backing_boundary() {
        let mut v: Vec<_> = (0..192).map(|p| vk::SparseMemoryBind::default().resource_offset(p*65536).size(65536)
            .memory(vk::DeviceMemory::from_raw(1+p/96)).memory_offset((p%96)*65536)).collect();
        let out = coalesce(&v);
        assert_eq!(out.len(),2);
        assert_eq!(expand(&out),expand(&v));
        v[10].memory_offset += 65536;
        v[20].flags = vk::SparseMemoryBindFlags::METADATA;
        v[30].memory = vk::DeviceMemory::null();
        v.remove(40);
        let out = coalesce(&v);
        assert_eq!(expand(&out),expand(&v));
        assert!(out.len()>2);
        assert!(coalesce(&[]).is_empty());
    }
    #[test]
    fn coalescing_keeps_an_unbind_before_a_later_rebind_of_the_same_page() {
        let page = |off: u64, mem: u64, moff: u64| vk::SparseMemoryBind::default().resource_offset(off).size(65536)
            .memory(vk::DeviceMemory::from_raw(mem)).memory_offset(moff);
        // Bind page 1, unbind page 0, then rebind page 0: the result must end with page 0 bound.
        let v = [page(65536, 1, 65536), page(0, 0, 0), page(0, 1, 0)];
        let out = coalesce(&v);
        let last_for_page0 = out.iter().rev().find(|b| b.resource_offset <= 0 && 0 < b.resource_offset + b.size).unwrap();
        assert_eq!(last_for_page0.memory.as_raw(), 1);
        assert_eq!(out.iter().map(|b| b.size).sum::<u64>(), 3 * 65536);
    }
}
