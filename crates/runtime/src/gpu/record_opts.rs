//! Opt-in recording experiments. Neither changes guest memory reads or queue ordering.
use std::time::{Duration, Instant};

#[cfg(target_os = "linux")]
pub fn tsc() -> bool {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ON.get_or_init(|| std::env::var("DS_RECORD_TSC").ok().as_deref() == Some("1"))
}

pub fn static_bind() -> bool {
    #[cfg(target_os = "linux")]
    {
        static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
        *ON.get_or_init(|| std::env::var("DS_RECORD_STATIC_BIND").ok().as_deref() == Some("1"))
    }
    #[cfg(windows)]
    { false }
}

/// Elapsed statistics only. Never use this for guest clocks, submission deadlines or waits.
/// Diagnostics retain Instant. No lazy calibration here: memory tests can precede time::init.
#[derive(Clone, Copy)]
pub enum Timer {
    Host(Instant),
    #[cfg(target_os = "linux")]
    Tsc { start: u64, ns_per_tick: f64 },
}
impl Timer {
    #[inline]
    pub fn start() -> Self {
        #[cfg(target_os = "linux")]
        if tsc() && !super::timeline::on() && !super::exec::profiling_on() {
            static INVARIANT: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
            // __cpuid required unsafe on older supported toolchains.
            #[allow(unused_unsafe)]
            let invariant = *INVARIANT.get_or_init(|| unsafe {
                core::arch::x86_64::__cpuid(0x80000000).eax >= 0x80000007 &&
                core::arch::x86_64::__cpuid(0x80000007).edx & (1 << 8) != 0
            });
            if invariant {
                if let Some(hz) = crate::hle::kernel::time::initialized_tsc_hz().filter(|&h| h > 0) {
                    return Self::Tsc { start: unsafe { core::arch::x86_64::_rdtsc() }, ns_per_tick: 1e9 / hz as f64 };
                }
            }
        }
        Self::Host(Instant::now())
    }
    #[inline]
    pub fn elapsed(&self) -> Duration {
        match *self {
            Self::Host(t) => t.elapsed(),
            #[cfg(target_os = "linux")]
            Self::Tsc { start, ns_per_tick } => ticks_elapsed(start, unsafe { core::arch::x86_64::_rdtsc() }, ns_per_tick),
        }
    }
}
#[cfg(any(target_os = "linux", test))]
fn ticks_elapsed(start: u64, end: u64, ns_per_tick: f64) -> Duration {
    Duration::from_nanos((end.saturating_sub(start) as f64 * ns_per_tick) as u64)
}

/// Facts derived only from an immutable translation/plan, never from a guest descriptor.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct BindingFacts {
    pub res_info: usize, pub addr_info: usize,
    pub marks: bool, pub address: bool, pub storage_image: bool, pub uniform: bool,
}
impl BindingFacts {
    pub fn new(plan: &rdna::analysis::Plan, bindings: &[rdna::translate::Binding]) -> Self {
        use rdna::translate::{self, BindingKind as B};
        let mut f = Self { res_info: translate::res_info_offset(plan) as usize,
            addr_info: translate::addr_info_offset(plan) as usize, marks: false, address: false, storage_image: false, uniform: false };
        for b in bindings {
            match b.kind {
                B::Marks => f.marks = true,
                B::Address(_) => f.address = true,
                B::StorageImage(_) | B::StorageImageMips(_) => f.storage_image = true,
                B::SgprUniform => f.uniform = true,
                _ => {}
            }
        }
        f
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn timer_conversion_and_backwards_counter() {
        assert_eq!(ticks_elapsed(100, 4100, 0.25), Duration::from_nanos(1000));
        assert_eq!(ticks_elapsed(100, 99, 0.25), Duration::ZERO);
    }
    #[test]
    fn binding_facts_follow_plan_and_exact_binding_kinds() {
        use rdna::{analysis::{Plan, ScalarLoad, Ptr}, translate::{Binding, BindingKind as B}};
        let plan = Plan { loads: vec![ScalarLoad { ptr: Ptr::User(0), offset: 0, dwords: 8 }, ScalarLoad { ptr: Ptr::User(2), offset: 32, dwords: 4 }], ..Plan::default() };
        let f = BindingFacts::new(&plan, &[]);
        assert_eq!(f.res_info, 44);
        assert_eq!(f.addr_info, 46);
        assert!(!f.marks && !f.address && !f.storage_image && !f.uniform);
        for (i, kind) in [B::Marks, B::Address(2), B::StorageImageMips(3), B::SgprUniform].into_iter().enumerate() {
            let f = BindingFacts::new(&plan, &[Binding { binding: 7, kind }]);
            assert_eq!([f.marks, f.address, f.storage_image, f.uniform], std::array::from_fn(|j| i == j));
        }
        assert!(!BindingFacts::new(&plan, &[Binding { binding: 0, kind: B::CubeFaces(0) }]).storage_image);
    }
}
