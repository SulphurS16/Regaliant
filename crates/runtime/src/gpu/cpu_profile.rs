//! Bounded, opt-in gpu-gfx attribution. TLS accumulators never drop events at a trace
//! capacity. TSC measures elapsed self time (including preemption), not scheduled CPU.
#[derive(Clone, Copy)]
#[repr(usize)]
pub enum Part {
    PacketLoop, PacketDispatch, Registers, StateChange, Draw, Dispatch, ShaderPrepare,
    BindStage, Descriptors, Barrier, ImageLookup, BufferLookup, Fe2Placement,
    ResourceSync, Parallel, Wait, Fe2Ensure, Fe2Versions, Fe2Memo,
}
#[cfg(target_os = "linux")]
mod linux {
    use super::Part;
    use std::{cell::RefCell, marker::PhantomData, rc::Rc, sync::atomic::{AtomicU8, AtomicU64, Ordering::*}};
    const N: usize = 19;
    const NAMES: [&str; N] = ["packet loop", "packet dispatch", "register writes", "state-change detection",
        "draw emission", "dispatch emission", "shader preparation", "resource binding", "descriptor building",
        "barrier computation", "image lookup", "buffer lookup", "FE2 placement", "resource synchronization",
        "parallel planning", "waits", "FE2 ensure/upload checks", "FE2 fine-page version checks", "FE2 placement memo"];
    // 0 idle, 1 accumulating, 2 drain. Only gpu-gfx owns its TLS. Publication happens
    // on its next scope boundary, clipping open scopes; the control thread never reads live TLS.
    static STATE: AtomicU8 = AtomicU8::new(0);
    static GENERATION: AtomicU64 = AtomicU64::new(0);
    static STOP: AtomicU64 = AtomicU64::new(0);
    static RESULT: parking_lot::Mutex<Option<Tally>> = parking_lot::Mutex::new(None);
    #[derive(Clone, Copy, Default)]
    struct Count { calls: u64, inclusive: u64, own: u64 }
    #[derive(Clone, Copy, Default)]
    struct Entry { part: usize, begin: u64, children: u64 }
    struct Tally {
        generation: u64, counts: [Count; N], stack: [Entry; 128], depth: usize,
        overflow: u64, first: u64, last: u64, cpu_start: u64, cpu_end: u64, nesting_errors: u64,
    }
    impl Default for Tally {
        fn default() -> Self { Self { generation:0, counts:[Count::default();N], stack:[Entry::default();128], depth:0, overflow:0, first:0, last:0, cpu_start:0, cpu_end:0, nesting_errors:0 } }
    }
    impl Tally {
        fn enter(&mut self, part: Part, t: u64) -> bool {
            if self.depth == self.stack.len() { self.overflow += 1; return false; }
            if self.first == 0 { self.first = t; }
            self.stack[self.depth] = Entry { part:part as usize, begin:t, children:0 };
            self.depth += 1;
            true
        }
        fn leave(&mut self, t: u64) {
            self.depth -= 1;
            let e = self.stack[self.depth];
            let elapsed = t.saturating_sub(e.begin);
            let c = &mut self.counts[e.part];
            c.calls += 1; c.inclusive += elapsed; c.own += elapsed.saturating_sub(e.children);
            if self.depth > 0 { self.stack[self.depth - 1].children += elapsed; }
            self.last = t;
        }
    }
    thread_local! {
        static GFX: bool = crate::log::thread_name() == "gpu-gfx";
        static LOCAL: RefCell<Tally> = RefCell::new(Tally::default());
    }
    #[inline]
    fn ticks() -> u64 { unsafe { core::arch::x86_64::_mm_lfence(); let t=core::arch::x86_64::_rdtsc(); core::arch::x86_64::_mm_lfence(); t } }
    fn cpu_ns() -> u64 {
        let mut t: libc::timespec = unsafe { std::mem::zeroed() };
        unsafe { libc::clock_gettime(libc::CLOCK_THREAD_CPUTIME_ID, &mut t); }
        t.tv_sec as u64 * 1_000_000_000 + t.tv_nsec as u64
    }
    pub struct Scope { entered: Option<usize>, generation: u64, _local: PhantomData<Rc<()>> }
    #[inline]
    pub fn scope(part: Part) -> Scope {
        let state = STATE.load(Acquire);
        let mut generation = 0;
        let entered = if state == 1 && GFX.with(|gfx| *gfx) { LOCAL.with(|local| {
            let mut t = local.borrow_mut();
            generation = GENERATION.load(Relaxed);
            if t.generation != generation { *t = Tally { generation, cpu_start:cpu_ns(), ..Tally::default() }; }
            t.enter(part, ticks()).then_some(t.depth - 1)
        }) } else { None };
        if state == 2 { publish(); }
        Scope { entered, generation, _local:PhantomData }
    }
    impl Drop for Scope {
        #[inline]
        fn drop(&mut self) {
            let Some(depth) = self.entered else { return };
            let end = ticks();
            let stop = STOP.load(Acquire);
            LOCAL.with(|local| {
                let mut t = local.borrow_mut();
                if t.generation != self.generation { return; } // guard from an already published window
                if t.depth != depth + 1 { t.nesting_errors += 1; }
                t.leave(if stop == 0 { end } else { end.min(stop) });
            });
            if STATE.load(Relaxed) == 2 { publish(); }
        }
    }
    fn publish() {
        if !GFX.with(|gfx| *gfx) { return; }
        LOCAL.with(|local| {
            let mut t = local.borrow_mut();
            if t.cpu_end == 0 { t.cpu_end = cpu_ns(); }
            if STATE.load(Acquire) == 2 {
                // A command buffer may span many frames. Do not wait for its outer
                // guard to leave: close every active scope at STOP on this owning
                // thread, then make later drops inert through the generation check.
                let stop = STOP.load(Acquire);
                while t.depth > 0 { t.leave(stop); }
                *RESULT.lock() = Some(std::mem::take(&mut *t));
                STATE.store(0, Release);
            }
        });
    }
    pub fn start() -> Result<(), String> {
        if STATE.load(Acquire) != 0 { return Err("CPU self profile already active".into()); }
        *RESULT.lock() = None;
        STOP.store(0, Relaxed);
        GENERATION.fetch_add(1, Relaxed);
        STATE.store(1, Release);
        Ok(())
    }
    pub fn stop() { STOP.store(ticks(), Release); STATE.store(2, Release); }
    pub fn pending() -> bool { STATE.load(Acquire) != 0 }
    pub fn write(label: &str, flips: u64) -> Result<(), String> {
        let t = RESULT.lock().take().ok_or("no gpu-gfx CPU self profile")?;
        let hz = crate::hle::kernel::time::tsc_hz() as f64;
        let mut rows: Vec<_> = t.counts.iter().enumerate().filter(|(_,c)| c.calls > 0).map(|(i,c)| serde_json::json!({
            "name":NAMES[i], "calls":c.calls, "self_ticks":c.own, "inclusive_ticks":c.inclusive,
            "self_ms_per_flip":c.own as f64 * 1000. / hz / flips.max(1) as f64,
            "inclusive_ms_per_flip":c.inclusive as f64 * 1000. / hz / flips.max(1) as f64,
        })).collect();
        rows.sort_by(|a,b| b["self_ticks"].as_u64().cmp(&a["self_ticks"].as_u64()));
        let top: Vec<_> = rows.iter().filter(|r| r["name"] != "waits").take(10).cloned().collect();
        let data = serde_json::json!({"phase":label,"flips":flips,"tsc_hz":hz,"overflow":t.overflow,
            "valid":t.overflow == 0 && t.nesting_errors == 0 && t.first != 0 && flips >= 16,
            "nesting_errors":t.nesting_errors, "thread_cpu_ms_per_flip":t.cpu_end.saturating_sub(t.cpu_start) as f64 / 1e6 / flips.max(1) as f64,"rows":rows,"top10":top,
            "elapsed_ms":t.last.saturating_sub(t.first) as f64 * 1000. / hz,
            "note":"Nested self elapsed TSC time, excludes instrumented children/waits; includes preemption and instrumentation overhead. Not scheduled CPU. Separate from clean FPS. No event-cap loss."});
        let dir = crate::run_dir().ok_or("no run directory")?;
        std::fs::write(dir.join(format!("{label}-cpu-self.json")),data.to_string()).map_err(|e| e.to_string())
    }
    #[cfg(test)]
    mod tests {
        use super::*;
        #[test]
        #[ignore = "CPU timing diagnostic; run release with --nocapture"]
        fn disabled_scope_cost() {
            assert_eq!(STATE.load(Acquire),0);
            let start=std::time::Instant::now();
            for _ in 0..1_000_000 { drop(std::hint::black_box(scope(Part::PacketLoop))); }
            eprintln!("disabled CPU scope: {:.2} ns/call",start.elapsed().as_nanos() as f64 / 1_000_000.);
        }
        #[test]
        fn tls_profile_clips_open_scopes_without_waiting_for_the_command_buffer() {
            std::thread::Builder::new().name("gpu-gfx".into()).spawn(|| {
                start().unwrap();
                let outer=scope(Part::PacketLoop);
                let inner=scope(Part::Wait);
                stop();
                drop(inner); assert!(!pending());
                let t=RESULT.lock().take().unwrap();
                assert_eq!(t.counts[Part::PacketLoop as usize].calls,1);
                assert_eq!(t.counts[Part::Wait as usize].calls,1);
                assert_eq!(t.last,STOP.load(Acquire));
                assert_eq!(t.nesting_errors,0);
                start().unwrap();
                let next=scope(Part::Registers);
                drop(outer); // the old command buffer may finish in a new profile
                stop(); drop(next);
                let t=RESULT.lock().take().unwrap();
                assert_eq!(t.counts[Part::Registers as usize].calls,1);
                assert_eq!(t.counts[Part::PacketLoop as usize].calls,0);
                assert_eq!(t.nesting_errors,0);
            }).unwrap().join().unwrap();
        }
        #[test]
        fn nested_self_time_excludes_children_and_waits() {
            let mut t = Tally::default();
            assert!(t.enter(Part::PacketLoop,10));
            assert!(t.enter(Part::BindStage,20));
            assert!(t.enter(Part::Wait,25)); t.leave(45); t.leave(70); t.leave(110);
            assert_eq!(t.counts[Part::PacketLoop as usize].own,50);
            assert_eq!(t.counts[Part::BindStage as usize].own,30);
            assert_eq!(t.counts[Part::Wait as usize].own,20);
            assert_eq!(t.counts.iter().map(|c| c.own).sum::<u64>(),100);
        }
        #[test]
        fn all_calls_survive_the_old_event_limit() {
            let mut t = Tally::default();
            for i in 0..300_000 { assert!(t.enter(Part::Registers,i*2)); t.leave(i*2+1); }
            assert_eq!(t.counts[Part::Registers as usize].calls,300_000);
            assert_eq!(t.counts[Part::Registers as usize].own,300_000);
            assert_eq!(t.overflow,0);
        }
        #[test]
        fn stop_clips_open_scopes_and_overflow_is_visible() {
            let mut t = Tally::default();
            for _ in 0..128 { assert!(t.enter(Part::PacketLoop,10)); }
            assert!(!t.enter(Part::Draw,12));
            for _ in 0..128 { t.leave(20); }
            assert_eq!(t.overflow,1);
            assert_eq!(t.counts[0].own,10);
        }
    }
}
#[cfg(target_os = "linux")]
pub use linux::*;
#[cfg(not(target_os = "linux"))]
pub struct Scope;
#[cfg(not(target_os = "linux"))]
#[inline]
pub fn scope(_: Part) -> Scope { Scope }
