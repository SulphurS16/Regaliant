//! Opt-in display-rate FG. Guest flip ownership ends after a bounded private copy,
//! never after WSI acquisition/presentation. FSR interpolates (not extrapolates), so
//! playback trails the real stream by one measured real interval plus 4 ms.
use super::*;
use std::{collections::VecDeque, io::Write};

static START: OnceLock<Instant> = OnceLock::new();
fn ns(at: Instant) -> u64 { at.saturating_duration_since(*START.get_or_init(|| at)).as_nanos() as u64 }
static RUNNING: AtomicBool = AtomicBool::new(false);
static STOP: AtomicBool = AtomicBool::new(false);
static THREAD: Mutex<Option<std::thread::JoinHandle<()>>> = Mutex::new(None);
static WAKE: parking_lot::Condvar = parking_lot::Condvar::new();
// published, real presents, generated presents, repeats, pool/queue drops, copy ns, age ns
static STATS: [AtomicU64; 7] = [const { AtomicU64::new(0) }; 7];
pub fn counters() -> [u64; 7] { STATS.each_ref().map(|x| x.load(Relaxed)) }
pub fn stopped() -> bool { STOP.load(Relaxed) }

struct Pair { real: Image, generated: Image }
impl Drop for Pair {
    fn drop(&mut self) { unsafe { self.real.destroy(); self.generated.destroy(); } }
}
#[derive(Clone)]
struct Frame { pair: Arc<Pair>, at: u64, flip: u64, generated: bool, shot: bool }
// plain: packets in a row without an interpolation (history resets, no TAA): no playback lag then.
struct Queue { frames: VecDeque<Frame>, interval: u64, previous: Option<u64>, paused: bool, plain: u32 }
static QUEUE: Mutex<Queue> = Mutex::new(Queue { frames: VecDeque::new(), interval: 33_333_333, previous: None, paused: false, plain: 0 });
struct Copies { pool: v::CommandPool, cmd: v::CommandBuffer, fence: v::Fence, pairs: Vec<Arc<Pair>> }
static COPIES: Mutex<Option<Copies>> = Mutex::new(None);

/// Prefer the newest eligible real image. Only use an interpolation when no real
/// image became eligible. Never show an older midpoint after its newer endpoint.
fn select(times: impl Iterator<Item = (usize, u64, bool)>, due: u64, last: Option<u64>) -> Option<usize> {
    let mut real = None; let mut generated = None;
    for (i, at, is_generated) in times {
        if at > due || last.is_some_and(|l| at <= l) { continue; }
        let best = if is_generated { &mut generated } else { &mut real };
        if best.is_none_or(|(_, t)| at >= t) { *best = Some((i, at)); }
    }
    real.or(generated).map(|(i, _)| i)
}
const PLAIN_RUN: u32 = 3;
fn delay(interval: u64) -> u64 { interval.clamp(8_000_000, 96_000_000) + 4_000_000 }

pub fn pause() {
    let mut q = QUEUE.lock(); q.frames.clear(); q.previous = None; q.paused = true;
    WAKE.notify_all();
}
pub fn disable() { STOP.store(true, Relaxed); WAKE.notify_all(); }
pub fn stop() {
    disable();
    let mut t = THREAD.lock();
    if let Some(h) = t.take() {
        let until = Instant::now() + Duration::from_secs(2);
        while !h.is_finished() && Instant::now() < until { std::thread::sleep(Duration::from_millis(10)); }
        if h.is_finished() { let _ = h.join(); }
        else { crate::warn!("adaptive FG presenter still in WSI at shutdown; process-lifetime resources remain pinned"); }
    }
}

/// Both inputs are complete and in GENERAL. Return only after the private copy's
/// fence. Queue overflow drops a packet, it never waits for a display slot.
pub fn enqueue(real: v::Image, size: (u32, u32), generated: Option<v::Image>, at: Instant, shot: bool) -> Result<(), String> {
    if STOP.load(Relaxed) { return Ok(()); }
    let at = ns(at);
    let flip = crate::gpu::FRAMES.load(Relaxed);
    let before = Instant::now();
    let mut copy = COPIES.lock();
    if copy.is_none() {
        unsafe {
            let d = &ctx().device;
            let pool = d.create_command_pool(&v::CommandPoolCreateInfo::default().queue_family_index(ctx().queue_family)
                .flags(v::CommandPoolCreateFlags::RESET_COMMAND_BUFFER), None).map_err(|e| e.to_string())?;
            let cmd = d.allocate_command_buffers(&v::CommandBufferAllocateInfo::default().command_pool(pool).command_buffer_count(1)).map_err(|e| e.to_string())?[0];
            let fence = d.create_fence(&v::FenceCreateInfo::default(), None).map_err(|e| e.to_string())?;
            *copy = Some(Copies { pool, cmd, fence, pairs: Vec::new() });
        }
    }
    let c = copy.as_mut().unwrap();
    // Every successful copy waited its fence, and the presenter waits its read fence.
    // Error paths leave the pair pinned in the pool and disable FG (no unsafe reuse).
    c.pairs.retain(|p| Arc::strong_count(p) != 1 || (p.real.desc.width, p.real.desc.height) == size);
    let pair = if let Some(p) = c.pairs.iter().find(|p| Arc::strong_count(p) == 1).cloned() { p } else {
        if c.pairs.len() >= 4 { STATS[4].fetch_add(1, Relaxed); return Ok(()); }
        let real = Image::new(size.0, size.1, COLOR)?;
        let generated = match Image::new(size.0, size.1, COLOR) {
            Ok(image) => image,
            Err(e) => { unsafe { real.destroy(); } return Err(e); }
        };
        let p = Arc::new(Pair { real, generated });
        c.pairs.push(p.clone()); p
    };
    unsafe {
        let d = &ctx().device;
        d.reset_command_pool(c.pool, v::CommandPoolResetFlags::empty()).map_err(|e| e.to_string())?;
        d.begin_command_buffer(c.cmd, &v::CommandBufferBeginInfo::default()).map_err(|e| e.to_string())?;
        blit(c.cmd, real, size, &pair.real);
        if let Some(g) = generated { blit(c.cmd, g, size, &pair.generated); }
        d.end_command_buffer(c.cmd).map_err(|e| e.to_string())?;
        d.reset_fences(&[c.fence]).map_err(|e| e.to_string())?;
        ctx().submit(c.cmd, &[], &[], c.fence)?;
        if let Err(e) = d.wait_for_fences(&[c.fence], true, 5_000_000_000) {
            // Could still be referenced by the GPU; never reclaim this allocation.
            std::mem::forget(pair); STOP.store(true, Relaxed);
            return Err(format!("adaptive private copy: {e}"));
        }
    }
    drop(copy);
    STATS[5].fetch_add(before.elapsed().as_nanos() as u64, Relaxed);
    let mut q = QUEUE.lock();
    if let Some(prev) = q.previous {
        let interval = at.saturating_sub(prev);
        if (5_000_000..=100_000_000).contains(&interval) { q.interval = (q.interval * 7 + interval) / 8; }
        if generated.is_some() && interval <= 100_000_000 {
            q.frames.push_back(Frame { pair: pair.clone(), at: prev + interval / 2, flip, generated: true, shot });
        }
    }
    q.plain = if generated.is_some() { 0 } else { q.plain.saturating_add(1) };
    q.previous = Some(at); q.paused = false;
    q.frames.push_back(Frame { pair, at, flip, generated: false, shot });
    while q.frames.len() > 8 { q.frames.pop_front(); STATS[4].fetch_add(1, Relaxed); }
    STATS[0].fetch_add(1, Relaxed);
    if !RUNNING.swap(true, Relaxed) {
        crate::info!("FG adaptive presenter: private four-packet pool, display-rate slots, newest eligible real then midpoint; playback lag one real interval + 4ms (8..96ms interval). Legacy pacing remains default.");
        match std::thread::Builder::new().name("fg-present".into()).spawn(worker) {
            Ok(t) => *THREAD.lock() = Some(t),
            Err(e) => { STOP.store(true, Relaxed); return Err(e.to_string()); }
        }
    }
    WAKE.notify_all();
    Ok(())
}
fn worker() {
    crate::log::set_thread_name("fg-present");
    let mut file = crate::run_dir().and_then(|p| std::fs::File::create(p.join("fg-adaptive.csv")).ok());
    if let Some(f) = file.as_mut() { let _ = writeln!(f, "slot_ns,flip,source_ns,generated,repeat,age_ms,playback_ms,present_ok,present_ms"); }
    let mut next = Instant::now(); let mut last: Option<Frame> = None; let mut last_time = None;
    while !STOP.load(Relaxed) {
        let tick = Duration::from_secs_f64(1.0 / crate::displayclock::nominal_refresh().clamp(30.0, 240.0));
        let mut q = QUEUE.lock();
        if q.paused { last = None; last_time = None; }
        let now = Instant::now();
        if now < next { WAKE.wait_for(&mut q, next - now); continue; }
        if q.paused || q.frames.is_empty() && last.is_none() {
            WAKE.wait_for(&mut q, tick); next = Instant::now(); continue;
        }
        // Without interpolations to place between real frames, holding real frames back only
        // adds latency and uneven repeats: show each as soon as it is ready.
        let slot = ns(now); let lag = if q.plain >= PLAIN_RUN { 0 } else { delay(q.interval) };
        let index = select(q.frames.iter().enumerate().map(|(i,f)| (i,f.at,f.generated)), slot.saturating_sub(lag), last_time);
        let repeat = index.is_none();
        if let Some(i) = index {
            let f = q.frames[i].clone(); last_time = Some(f.at);
            q.frames.retain(|x| x.at > f.at); last = Some(f);
        }
        drop(q);
        if let Some(f) = last.as_ref() {
            let image = if f.generated { &f.pair.generated } else { &f.pair.real };
            let _guard = PresentScope::new(f.generated);
            crate::gpu::vk::framewatch::SHOWN_FLIP.store(f.flip, Relaxed);
            if !repeat && f.shot { vk::present::shot_next(); }
            // FIFO retains the last image when no new source is ready.
            let ok = !repeat && vk::present::present(vk::present::Source::Image { image: image.image(), format: COLOR,
                layout: v::ImageLayout::GENERAL, width: image.desc.width, height: image.desc.height, hdr: false });
            // Linux present() waits its blit fence before returning. This CPU Arc is
            // therefore sufficient to keep the source alive, including resize/minimize.
            if ok || repeat {
                STATS[if repeat { 3 } else if f.generated { 2 } else { 1 }].fetch_add(1, Relaxed);
                if f.generated && !repeat { GENERATED.fetch_add(1, Relaxed); }
                STATS[6].fetch_add(slot.saturating_sub(f.at), Relaxed);
            }
            if let Some(out) = file.as_mut() {
                let _ = writeln!(out, "{slot},{},{},{},{},{:.3},{:.3},{ok},{:.3}", f.flip, f.at, f.generated, repeat,
                    slot.saturating_sub(f.at) as f64/1e6, lag as f64/1e6, now.elapsed().as_secs_f64()*1e3);
                let _ = out.flush();
            }
        }
        next += tick;
        if next < Instant::now() { next = Instant::now(); }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn presenter_recursion_and_generated_tag_are_thread_local() {
        assert!(!bypass_present_hook());
        let outer = PresentScope::new(true);
        assert!(presenting()); assert!(bypass_present_hook());
        std::thread::spawn(|| { assert!(!presenting()); assert!(!bypass_present_hook()); }).join().unwrap();
        { let _inner = PresentScope::new(false); assert!(!presenting()); assert!(bypass_present_hook()); }
        assert!(presenting()); drop(outer); assert!(!bypass_present_hook());
    }
    #[test]
    fn display_slots_are_monotonic_and_prefer_real() {
        let candidates = [(0, 10, false), (1, 15, true), (2, 20, false)];
        assert_eq!(select(candidates.into_iter(), 15, None), Some(0));
        assert_eq!(select(candidates.into_iter(), 15, Some(10)), Some(1));
        assert_eq!(select(candidates.into_iter(), 20, Some(10)), Some(2));
        assert_eq!(select(candidates.into_iter(), 40, Some(20)), None);
    }
    #[test]
    fn sixty_hz_consumes_thirty_three_real_stream_without_backpressure() {
        let frames: Vec<_> = (0..200).flat_map(|i| [(i*2,i as u64*30_000_000,false), (i*2+1,i as u64*30_000_000+15_000_000,true)]).collect();
        let mut last = None; let (mut real, mut generated) = (0,0);
        for slot in 0..360 {
            if let Some(i) = select(frames.iter().copied(), slot*16_666_667, last) {
                let (_, at, g) = frames[i]; last = Some(at);
                if g { generated+=1; } else { real+=1; }
            }
        }
        assert!(real >= 198); assert!(generated >= 150);
        assert_eq!(delay(30_000_000),34_000_000);
        // A midpoint is unavailable until its future endpoint finishes generation.
        let mut last = None; let mut shown = 0;
        for slot in 0..360 {
            let wall = slot * 16_666_667u64;
            let arrived = frames.iter().copied().filter(|(_,at,g)|
                at + if *g { 18_000_000 } else { 3_000_000 } <= wall);
            if let Some(i) = select(arrived, wall.saturating_sub(delay(30_000_000)), last) {
                assert!(last.is_none_or(|l| frames[i].1 > l));
                last = Some(frames[i].1); shown += 1;
            }
        }
        assert!(shown >= 350);
    }
}
