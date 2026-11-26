#[cfg(test)]
mod tests {
    use crate::mem::*;
    #[test]
    fn physical_aliases_remap_release_and_vma_splitting() {
        if std::env::var_os("REGALIANT_SHARED_MEM_TEST").is_none() {
            let out = std::process::Command::new(std::env::current_exe().unwrap())
                .args([
                    "--exact",
                    "platform::mem_tests::tests::physical_aliases_remap_release_and_vma_splitting",
                    "--nocapture",
                ])
                .env("REGALIANT_SHARED_MEM_TEST", "1")
                .env("DS_SECTIONS", "1")
                .output()
                .unwrap();
            assert!(
                out.status.success(),
                "{}\n{}",
                String::from_utf8_lossy(&out.stdout),
                String::from_utf8_lossy(&out.stderr)
            );
            return;
        }
        reserve_guest_range().unwrap();
        let mut m = MEM.lock();
        let physical = m.dmem_allocate(0, DMEM_SIZE, 3 * PAGE, PAGE, 3).unwrap();
        let a = MMAP_BASE;
        let b = MMAP_BASE + 0x10_0000;
        m.map(a, 3 * PAGE, VmaKind::Direct, 3, physical, 0, "test-a")
            .unwrap();
        m.map(b, 3 * PAGE, VmaKind::Direct, 3, physical, 0, "test-b")
            .unwrap();
        unsafe {
            *((a + PAGE) as *mut u64) = 0xcafe;
        }
        assert_eq!(unsafe { *((b + PAGE) as *const u64) }, 0xcafe);
        m.protect(a + PAGE, a + 2 * PAGE, PROT_CPU_READ).unwrap();
        assert_eq!(m.find(a + PAGE).unwrap().prot, PROT_CPU_READ);
        assert_eq!(m.find(a + PAGE).unwrap().phys, physical + PAGE);
        m.unmap_range(a, a + 3 * PAGE);
        m.map(a, 3 * PAGE, VmaKind::Direct, 3, physical, 0, "remapped")
            .unwrap();
        assert_eq!(unsafe { *((a + PAGE) as *const u64) }, 0xcafe);
        m.dmem_release(physical, 3 * PAGE);
        assert!(m.find(a).is_none() && m.find(b).is_none());
        assert_eq!(
            m.dmem_allocate(0, DMEM_SIZE, 3 * PAGE, PAGE, 3),
            Some(physical)
        );
        m.map(a, 3 * PAGE, VmaKind::Direct, 3, physical, 0, "reused")
            .unwrap();
        assert_eq!(unsafe { *((a + PAGE) as *const u64) }, 0);
        m.dmem_release(physical, 3 * PAGE);
        m.map(a, 3 * PAGE, VmaKind::Flexible, 3, 0, 0, "flex")
            .unwrap();
        assert_eq!(m.flexible_used, 3 * PAGE);
        m.unmap_range(a + PAGE, a + 2 * PAGE);
        assert_eq!(m.flexible_used, 2 * PAGE);
        assert!(
            m.find(a).is_some() && m.find(a + PAGE).is_none() && m.find(a + 2 * PAGE).is_some()
        );
        m.unmap_range(a, a + 3 * PAGE);
        assert_eq!(m.flexible_used, 0);
    }
    #[test]
    fn readers_survive_section_split_races() {
        if std::env::var_os("REGALIANT_SPLIT_TEST").is_none() {
            let out = std::process::Command::new(std::env::current_exe().unwrap())
                .args([
                    "--exact",
                    "platform::mem_tests::tests::readers_survive_section_split_races",
                    "--nocapture",
                ])
                .env("REGALIANT_SPLIT_TEST", "1")
                .env("DS_SECTIONS", "1")
                .output()
                .unwrap();
            assert!(
                out.status.success(),
                "{}\n{}",
                String::from_utf8_lossy(&out.stdout),
                String::from_utf8_lossy(&out.stderr)
            );
            return;
        }
        use crate::platform::os::{PAGE_READWRITE, mapping_allows};
        use crate::sections::{self as sec, Sec};
        use std::sync::atomic::{AtomicBool, Ordering};
        reserve_guest_range().unwrap();
        crate::platform::signals::ensure_installed().unwrap();
        let at = MMAP_BASE + 0x1000000;
        sec::map_view(at, PAGE, Sec::Dmem, 0, PAGE_READWRITE).unwrap();
        unsafe {
            std::ptr::write_volatile(at as *mut u64, 0x12345678);
        }
        let stop = AtomicBool::new(false);
        let ready = std::sync::Barrier::new(2);
        std::thread::scope(|scope| {
            let reader = scope.spawn(|| {
                let _stack = crate::platform::signals::SignalStack::new().unwrap();
                ready.wait();
                while !stop.load(Ordering::Acquire) {
                    assert_eq!(
                        unsafe { std::ptr::read_volatile(at as *const u64) },
                        0x12345678
                    );
                }
            });
            ready.wait();
            for iteration in 0..500 {
                sec::split_begin(at, at + PAGE, sec::SPLIT_PROTECT);
                sec::unmap_view(at).unwrap();
                if iteration == 0 {
                    std::thread::sleep(std::time::Duration::from_millis(2));
                } else {
                    std::thread::yield_now();
                }
                sec::map_view(at, PAGE, Sec::Dmem, 0, PAGE_READWRITE).unwrap();
                sec::split_end();
            }
            stop.store(true, Ordering::Release);
            reader.join().unwrap();
        });
        assert!(
            crate::platform::signals::RECOVERED.load(Ordering::Relaxed) > 0,
            "reader must exercise the signal service"
        );
        assert!(mapping_allows(at as usize, false));
        sec::unmap_view(at).unwrap();
        assert!(!mapping_allows(at as usize, false));
    }
}
