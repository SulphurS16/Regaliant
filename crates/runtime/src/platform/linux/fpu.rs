//! New guest threads match the Windows CreateThread entry environment: masked exceptions,
//! round-to-nearest, x87 53-bit precision. Shared fresh guest fibers independently use 0x037f.
//! Round 5 used 0x037f for threads too; matching fibers was not evidence for thread defaults.
//! Linux pthreads inherit their creator's controls (which may be running guest code).
//! sigreturn restores the interrupted FP state; a recovery handler must not reset it.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Controls {
    pub x87: u16,
    pub mxcsr: u32,
}

pub const GUEST: Controls = Controls { x87: 0x027f, mxcsr: 0x1f80 };

pub fn controls() -> Controls {
    let mut c = GUEST;
    unsafe { std::arch::asm!("fnstcw [{cw}]", "stmxcsr [{mx}]", cw = in(reg) &mut c.x87, mx = in(reg) &mut c.mxcsr, options(nostack, preserves_flags)); }
    c
}

/// The supplied controls must mask exceptions and contain only supported MXCSR bits.
pub unsafe fn set(c: Controls) {
    unsafe { std::arch::asm!("fldcw [{cw}]", "ldmxcsr [{mx}]", cw = in(reg) &c.x87, mx = in(reg) &c.mxcsr, options(nostack, preserves_flags)); }
}

pub fn enter() {
    let before = controls();
    crate::info!("Linux guest FP entry: x87 {:#x} MXCSR {:#x} -> x87 {:#x} MXCSR {:#x}", before.x87, before.mxcsr, GUEST.x87, GUEST.mxcsr);
    // Logging formats a floating-point timestamp and can set MXCSR exception flags.
    unsafe {
        std::arch::asm!("fninit", options(nostack));
        set(GUEST);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn guest_controls_do_not_inherit_host_rounding_or_flush_modes() {
        std::thread::spawn(|| {
            let original = controls();
            let changed = Controls { x87: 0x077f, mxcsr: 0xbfc0 };
            unsafe { set(changed); }
            let child = std::thread::spawn(move || {
                let inherited = controls();
                enter();
                (inherited, controls())
            }).join().unwrap();
            unsafe { set(original); }
            assert_eq!(child.0, changed);
            assert_eq!(child.1, GUEST);
        }).join().unwrap();
    }
}
