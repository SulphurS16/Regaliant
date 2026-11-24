//! Swapchain presentation: blits a rendered image (or uploaded pixels) to the window.
//!
//! The frame keeps its aspect (centred, black bars where the window's aspect
//! differs by 1% or more). A frame more than twice the window's size is first
//! halved by 2:1 linear blits (2x2 box filters) until the last blit reduces by
//! at most 2:1: a single linear blit samples 2x2 texels per pixel, so beyond
//! 2:1 it skips texels (a 3840x2160 frame in a 1280x720 window was point
//! sampled: aliased HUD text and edges).

use super::{ctx, window};
use ash::vk;
use parking_lot::Mutex;

const FRAMES: usize = 2;

/// An intermediate image of the halving chain.
struct Half {
    image: vk::Image,
    memory: vk::DeviceMemory,
    format: vk::Format,
    width: u32,
    height: u32,
}

struct Presenter {
    surface: vk::SurfaceKHR,
    swapchain: vk::SwapchainKHR,
    images: Vec<vk::Image>,
    format: vk::Format,
    extent: vk::Extent2D,
    pool: vk::CommandPool,
    cmds: Vec<vk::CommandBuffer>,
    fences: Vec<vk::Fence>,
    acquire: Vec<vk::Semaphore>,
    done: Vec<vk::Semaphore>,
    frame: usize,
    staging: Option<(vk::Buffer, vk::DeviceMemory, *mut u8, u64)>,
    upload: Option<(vk::Image, vk::DeviceMemory, u32, u32)>,
    /// The halving chain's images, largest first.
    halves: Vec<Half>,
    /// The last (frame size, window size) logged.
    logged: (u32, u32, u32, u32),
    /// The PC settings' sharpening pass (made on first use; `sharpen_failed` after an error).
    sharpen: Option<super::sharpen::Sharpen>,
    sharpen_failed: bool,
    /// The swapchain holds exclusive fullscreen (VK_EXT_full_screen_exclusive, application
    /// controlled), on this monitor.
    exclusive: Option<isize>,
    /// The swapchain is HDR10 (ST 2084 PQ, BT.2020): the title's display buffers hold HDR.
    hdr: bool,
    /// The frames shown are HDR (the title's display buffers' format; the next swapchain's).
    want_hdr: bool,
    /// The frame watch (`framewatch`: one-frame flashes), made on first use.
    watch: Option<super::framewatch::Watch>,
    watch_failed: bool,
    /// The system text box drawn over the frame (`crate::textbox`), made on first use.
    text_box: Option<TextBox>,
    /// The last present's id on this swapchain (`DS_PRESENT_STATS=1`: `waiter`).
    present_id: u64,
}

/// On-screen frame times (`DS_PRESENT_STATS=1`): each present's id goes to a thread that waits
/// until it is shown and counts how many of the display's refreshes each frame stayed. The
/// swapchain is not destroyed while it waits (`PW_LOCK`, and `PW_GEN` drops older waits).
static PW_LOCK: Mutex<()> = Mutex::new(());
static PW_GEN: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);
static PW_TX: Mutex<Option<std::sync::mpsc::Sender<(u64, u64, u64, std::time::Instant)>>> = Mutex::new(None);
/// Frames by refreshes shown (1 to 7, 8 and more), all intervals and those over 25 ms.
static PW_STATS: Mutex<([u64; 9], u64, u64)> = Mutex::new(([0; 9], 0, 0));
/// The time from a present's call to its present wait's return (when the display showed it),
/// in quarter milliseconds up to 64 ms (the last bucket: more).
static PW_LATENCY: Mutex<[u64; 257]> = Mutex::new([0; 257]);

fn present_waiter(rx: std::sync::mpsc::Receiver<(u64, u64, u64, std::time::Instant)>) {
    crate::log::set_thread_name("present-wait");
    let c = ctx();
    let pw = ash::khr::present_wait::Device::new(&c.instance, &c.device);
    let mut prev: Option<std::time::Instant> = None;
    let mut missed = 0u32;
    while let Ok(first) = rx.recv() {
        // (Behind, after a wait that timed out: only the newest present is waited for.)
        let (mut generation, mut sc, mut id, mut at) = first;
        while let Ok(m) = rx.try_recv() {
            (generation, sc, id, at) = m;
            prev = None;
        }
        let start = std::time::Instant::now();
        loop {
            let g = PW_LOCK.lock();
            if PW_GEN.load(std::sync::atomic::Ordering::Relaxed) != generation {
                prev = None;
                break;
            }
            use ash::vk::Handle;
            let r = unsafe { pw.wait_for_present(vk::SwapchainKHR::from_raw(sc), id, 20_000_000) };
            drop(g);
            match r {
                Ok(()) => {
                    let now = std::time::Instant::now();
                    #[cfg(target_os = "linux")]
                    crate::platform::present_timing::completed(id);
                    let q = (now.saturating_duration_since(at).as_secs_f64() * 4000.0) as usize;
                    PW_LATENCY.lock()[q.min(256)] += 1;
                    if let Some(p) = prev {
                        let gap = now.duration_since(p).as_secs_f64();
                        let refresh = crate::displayclock::snapshot().map_or(1.0 / 60.0, |s| s.period.as_secs_f64());
                        #[cfg(target_os = "linux")]
                        let refresh = { let _ = refresh; 1.0 / crate::displayclock::nominal_refresh() };
                        let n = ((gap / refresh).round() as usize).clamp(1, 8);
                        let mut s = PW_STATS.lock();
                        s.0[n] += 1;
                        s.1 += 1;
                        s.2 += (gap > 0.025) as u64;
                    }
                    prev = Some(now);
                    break;
                }
                Err(vk::Result::TIMEOUT) if start.elapsed() < std::time::Duration::from_secs(1) => continue,
                r => {
                    // (A window covered by others: its presents may never be shown.)
                    missed += 1;
                    if missed <= 3 {
                        crate::warn!("present waits: present {id} not shown within a second ({r:?}){}", if missed == 3 { "; not logged again" } else { "" });
                    }
                    prev = None;
                    break;
                }
            }
        }
    }
}

/// The on-screen counts afresh (`profile reset`).
pub fn reset_stats() {
    *PW_STATS.lock() = ([0; 9], 0, 0);
    *PW_LATENCY.lock() = [0; 257];
}

/// The present-to-display times' part of the status line (p50 and p90 in ms; "" when none).
fn latency_part() -> String {
    let h = *PW_LATENCY.lock();
    let n: u64 = h.iter().sum();
    if n == 0 {
        return String::new();
    }
    let at = |q: f64| -> f64 {
        let want = (q * n as f64).ceil() as u64;
        let mut acc = 0;
        for (i, &c) in h.iter().enumerate() {
            acc += c;
            if acc >= want {
                return i as f64 / 4.0;
            }
        }
        64.0
    };
    format!("; present to display p50 {:.2} ms, p90 {:.2} ms ({n} presents)", at(0.5), at(0.9))
}

/// Status: the frames shown and for how many refreshes each stayed (`DS_PRESENT_STATS=1`).
pub fn stats_line() -> String {
    let s = PW_STATS.lock();
    if s.1 == 0 {
        return String::new();
    }
    let pct = |n: u64| 100.0 * n as f64 / s.1 as f64;
    let refresh = crate::displayclock::snapshot().map_or(1000.0 / 60.0, |x| x.period.as_secs_f64() * 1e3);
    #[cfg(target_os = "linux")]
    let refresh = { let _ = refresh; 1000.0 / crate::displayclock::nominal_refresh() };
    format!(
        "    on screen (present waits): {} frame intervals; shown 1 refresh {:.1}%, 2 {:.1}%, 3 {:.1}%, 4 {:.1}%, 5+ {:.1}% (a refresh {:.2} ms); over 25 ms {}{}\n",
        s.1,
        pct(s.0[1]),
        pct(s.0[2]),
        pct(s.0[3]),
        pct(s.0[4]),
        pct(s.0[5..].iter().sum()),
        refresh,
        s.2,
        latency_part()
    )
}

/// The system text box's image (BGRA8 UNORM), its staging buffers (one per presenter slot) and
/// the content it holds (the dialog's generation, the frame height and HDR it was drawn for).
struct TextBox {
    image: vk::Image,
    memory: vk::DeviceMemory,
    w: u32,
    h: u32,
    staging: Vec<(vk::Buffer, vk::DeviceMemory, *mut u8)>,
    key: Option<(u64, u32, bool)>,
}

impl TextBox {
    unsafe fn destroy(self) {
        let c = ctx();
        unsafe {
            for (b, m, _) in self.staging {
                c.device.destroy_buffer(b, None);
                c.device.unmap_memory(m);
                c.device.free_memory(m, None);
            }
            c.device.destroy_image(self.image, None);
            c.device.free_memory(self.memory, None);
        }
    }
}

/// Draws the system text box (the character's name: `hle::libs::ime_dialog`) over the frame in
/// `dst` (TRANSFER_DST_OPTIMAL), centred across the picture at 38% of its height.
unsafe fn draw_text_box(p: &mut Presenter, cmd: vk::CommandBuffer, slot: usize, dst: vk::Image, picture: (u32, u32, u32, u32)) {
    // A size whose image or staging buffers could not be made, and when: tried again after 2 s,
    // not every frame.
    static FAILED: Mutex<Option<((u32, u32), std::time::Instant)>> = Mutex::new(None);
    let Some((generation, content)) = crate::hle::libs::ime_dialog::overlay() else { return };
    let c = ctx();
    let (rx, ry, rw, rh) = picture;
    let (w, h) = crate::textbox::size(rh, content.keys.as_ref().map_or(0, |k| k.rows.len()));
    if w > rw || h > rh || FAILED.lock().is_some_and(|(s, at)| s == (w, h) && at.elapsed() < std::time::Duration::from_secs(2)) {
        return;
    }
    let failed = || {
        if FAILED.lock().replace(((w, h), std::time::Instant::now())).is_none_or(|(s, _)| s != (w, h)) {
            crate::warn!("text box: no image of {w}x{h} for it (tried again every 2 s)");
        }
    };
    unsafe {
        if p.text_box.as_ref().is_none_or(|t| (t.w, t.h) != (w, h)) {
            // (The other slot may still read the old one.)
            let others: Vec<vk::Fence> = p.fences.iter().enumerate().filter(|&(i, _)| i != slot).map(|(_, &f)| f).collect();
            c.device.wait_for_fences(&others, true, u64::MAX).ok();
            if let Some(old) = p.text_box.take() {
                old.destroy();
            }
            let ici = vk::ImageCreateInfo::default()
                .image_type(vk::ImageType::TYPE_2D)
                .format(vk::Format::B8G8R8A8_UNORM)
                .extent(vk::Extent3D { width: w, height: h, depth: 1 })
                .mip_levels(1)
                .array_layers(1)
                .samples(vk::SampleCountFlags::TYPE_1)
                .tiling(vk::ImageTiling::OPTIMAL)
                .usage(vk::ImageUsageFlags::TRANSFER_DST | vk::ImageUsageFlags::TRANSFER_SRC);
            let Ok(image) = c.device.create_image(&ici, None) else {
                failed();
                return;
            };
            let Ok(memory) = c.alloc(c.device.get_image_memory_requirements(image), vk::MemoryPropertyFlags::DEVICE_LOCAL) else {
                c.device.destroy_image(image, None);
                failed();
                return;
            };
            if c.device.bind_image_memory(image, memory, 0).is_err() {
                c.device.destroy_image(image, None);
                c.device.free_memory(memory, None);
                failed();
                return;
            }
            let mut staging = Vec::new();
            for _ in 0..FRAMES {
                match c.host_buffer((w * h * 4) as u64, vk::BufferUsageFlags::TRANSFER_SRC) {
                    Ok(b) => staging.push(b),
                    Err(e) => {
                        crate::warn!("text box: no staging buffer ({e})");
                        TextBox { image, memory, w, h, staging, key: None }.destroy();
                        failed();
                        return;
                    }
                }
            }
            p.text_box = Some(TextBox { image, memory, w, h, staging, key: None });
        }
        let hdr = p.hdr;
        let t = p.text_box.as_mut().unwrap();
        let key = (generation, rh, hdr);
        if t.key != Some(key) {
            let (mut px, _, _) = crate::textbox::render(&content, rh);
            if hdr {
                crate::textbox::to_pq(&mut px, 203.0);
            }
            std::ptr::copy_nonoverlapping(px.as_ptr(), t.staging[slot].2, px.len());
            let range = vk::ImageSubresourceRange { aspect_mask: vk::ImageAspectFlags::COLOR, base_mip_level: 0, level_count: 1, base_array_layer: 0, layer_count: 1 };
            let to_dst = vk::ImageMemoryBarrier::default()
                .src_access_mask(vk::AccessFlags::TRANSFER_READ)
                .dst_access_mask(vk::AccessFlags::TRANSFER_WRITE)
                .old_layout(vk::ImageLayout::UNDEFINED)
                .new_layout(vk::ImageLayout::TRANSFER_DST_OPTIMAL)
                .src_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
                .dst_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
                .image(t.image)
                .subresource_range(range);
            c.device.cmd_pipeline_barrier(cmd, vk::PipelineStageFlags::TRANSFER, vk::PipelineStageFlags::TRANSFER, vk::DependencyFlags::empty(), &[], &[], &[to_dst]);
            let copy = vk::BufferImageCopy::default()
                .image_subresource(vk::ImageSubresourceLayers { aspect_mask: vk::ImageAspectFlags::COLOR, mip_level: 0, base_array_layer: 0, layer_count: 1 })
                .image_extent(vk::Extent3D { width: w, height: h, depth: 1 });
            c.device.cmd_copy_buffer_to_image(cmd, t.staging[slot].0, t.image, vk::ImageLayout::TRANSFER_DST_OPTIMAL, &[copy]);
            let to_src = vk::ImageMemoryBarrier::default()
                .src_access_mask(vk::AccessFlags::TRANSFER_WRITE)
                .dst_access_mask(vk::AccessFlags::TRANSFER_READ)
                .old_layout(vk::ImageLayout::TRANSFER_DST_OPTIMAL)
                .new_layout(vk::ImageLayout::TRANSFER_SRC_OPTIMAL)
                .src_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
                .dst_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
                .image(t.image)
                .subresource_range(range);
            c.device.cmd_pipeline_barrier(cmd, vk::PipelineStageFlags::TRANSFER, vk::PipelineStageFlags::TRANSFER, vk::DependencyFlags::empty(), &[], &[], &[to_src]);
            t.key = Some(key);
        }
        // Over the frame just blitted to `dst`.
        barrier(c, cmd, dst, vk::ImageLayout::TRANSFER_DST_OPTIMAL, vk::ImageLayout::TRANSFER_DST_OPTIMAL);
        let x = rx + (rw - w) / 2;
        let y = (ry + rh * 38 / 100).saturating_sub(h / 2).max(ry);
        blit(c, cmd, t.image, vk::ImageLayout::TRANSFER_SRC_OPTIMAL, (0, 0, w, h), dst, (x, y, w, h));
    }
}

unsafe impl Send for Presenter {}

static PRESENTER: Mutex<Option<Presenter>> = Mutex::new(None);

/// The next presented frame is read back (`shot_next`).
static SHOT_NEXT: std::sync::atomic::AtomicBool = std::sync::atomic::AtomicBool::new(false);

/// Control channel `wshot`: the next presented frame to `window-<n>.png`.
pub fn shot_next() {
    SHOT_NEXT.store(true, std::sync::atomic::Ordering::SeqCst);
}

/// `DS_WINDOW_SHOT=<n,...>`: presented frame numbers whose swapchain image is
/// read back to `window-<n>.png` (what the window shows, after scaling).
fn window_shot(frame: usize) -> bool {
    static LIST: std::sync::OnceLock<Vec<usize>> = std::sync::OnceLock::new();
    SHOT_NEXT.swap(false, std::sync::atomic::Ordering::SeqCst)
        || LIST.get_or_init(|| std::env::var("DS_WINDOW_SHOT").map(|v| v.split(',').filter_map(|x| x.trim().parse().ok()).collect()).unwrap_or_default()).contains(&frame)
}

pub enum Source {
    /// An image in TRANSFER_SRC_OPTIMAL-compatible state; `layout` is its current layout; `hdr`:
    /// its values are HDR10 (PQ-encoded BT.2020, the title's HDR output).
    Image { image: vk::Image, format: vk::Format, layout: vk::ImageLayout, width: u32, height: u32, hdr: bool },
    /// Linear BGRA8 pixels.
    Pixels { data: Vec<u8>, width: u32, height: u32 },
}

unsafe fn create_swapchain(p: &mut Presenter) -> Result<(), String> {
    let c = ctx();
    unsafe {
        let caps = c.surface_i.get_physical_device_surface_capabilities(c.pdev, p.surface).map_err(|e| format!("surface caps: {e}"))?;
        let formats = c.surface_i.get_physical_device_surface_formats(c.pdev, p.surface).map_err(|e| format!("surface formats: {e}"))?;
        {
            // (Once: the formats and color spaces the display offers; HDR10's appear while
            // Windows' HDR is on for it.)
            static ONCE: std::sync::Once = std::sync::Once::new();
            ONCE.call_once(|| {
                let list: Vec<String> = formats.iter().map(|f| format!("{:?}/{:?}", f.format, f.color_space)).collect();
                crate::info!("present: surface formats {}", list.join(", "));
            });
        }
        // Display buffers hold display-encoded values (the title applies its
        // own transfer function): UNORM swapchain formats show them as is. SDR goes to 8 bits
        // per channel as PC games present (until 2026-11-13 10-bit first, the title's buffers
        // being 10:10:10:2: the user saw a grey picture and NVIDIA overlay in exclusive
        // fullscreen); `DS_PRESENT_10BIT=1` keeps 10-bit SDR (diagnosis). HDR10 is 10-bit.
        let srgb_space = |f: &&vk::SurfaceFormatKHR| f.color_space == vk::ColorSpaceKHR::SRGB_NONLINEAR;
        // The title's HDR output (PQ-encoded BT.2020 in 10:10:10:2 display buffers) goes to an
        // HDR10 swapchain as it is, where the display offers one.
        let hdr10 = p.want_hdr && c.colorspaces;
        let hdr_fmt = formats
            .iter()
            .find(|f| hdr10 && f.color_space == vk::ColorSpaceKHR::HDR10_ST2084_EXT && matches!(f.format, vk::Format::A2B10G10R10_UNORM_PACK32 | vk::Format::A2R10G10B10_UNORM_PACK32))
            .copied();
        if p.want_hdr && hdr_fmt.is_none() {
            static ONCE: std::sync::Once = std::sync::Once::new();
            ONCE.call_once(|| crate::warn!("present: the frames are HDR but the display offers no HDR10 format: shown as SDR values"));
        }
        let ten = std::env::var("DS_PRESENT_10BIT").is_ok_and(|v| v == "1");
        let fmt = hdr_fmt
            .or_else(|| formats.iter().filter(srgb_space).find(|f| ten && f.format == vk::Format::A2B10G10R10_UNORM_PACK32).copied())
            .or_else(|| formats.iter().filter(srgb_space).find(|f| f.format == vk::Format::B8G8R8A8_UNORM).copied())
            .or_else(|| formats.iter().filter(srgb_space).find(|f| f.format == vk::Format::R8G8B8A8_UNORM).copied())
            .or_else(|| formats.iter().filter(srgb_space).find(|f| f.format == vk::Format::A2B10G10R10_UNORM_PACK32).copied())
            .or_else(|| formats.first().copied())
            .ok_or("no surface format")?;
        let mut extent = caps.current_extent;
        if extent.width == u32::MAX {
            extent = vk::Extent2D { width: window::CLIENT_W.load(std::sync::atomic::Ordering::Relaxed), height: window::CLIENT_H.load(std::sync::atomic::Ordering::Relaxed) };
        }
        if extent.width == 0 || extent.height == 0 {
            return Err("window minimized".into());
        }
        // (The surface's size against the window's client, measured on the window's thread: they
        // differ when Windows scales the size for this thread, as on 2026-11-23 before the process
        // was made DPI aware; the swapchain must take the surface's. Logged once per size, five
        // times at most: a resize in progress can differ for a moment too.)
        {
            static WARNED: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);
            static TIMES: std::sync::atomic::AtomicU32 = std::sync::atomic::AtomicU32::new(0);
            let (cw, ch) = (window::CLIENT_W.load(std::sync::atomic::Ordering::Relaxed), window::CLIENT_H.load(std::sync::atomic::Ordering::Relaxed));
            let key = (extent.width as u64) << 32 | extent.height as u64;
            if (cw, ch) != (extent.width, extent.height) && WARNED.swap(key, std::sync::atomic::Ordering::Relaxed) != key && TIMES.fetch_add(1, std::sync::atomic::Ordering::Relaxed) < 5 {
                crate::warn!("swapchain: the surface is {}x{}, the window's client {cw}x{ch} (Windows display scaling applied to this thread?)", extent.width, extent.height);
            }
        }
        let count = (caps.min_image_count + 1).min(if caps.max_image_count == 0 { u32::MAX } else { caps.max_image_count });
        let old = p.swapchain;
        // Exclusive fullscreen: held by the swapchain when the window asks for it (the PC tab's
        // Fullscreen), and never taken by the driver on its own for a window or borderless
        // fullscreen (a driver may change a display's mode for a window that covers it).
        let exclusive = if c.fse.is_some() { window::exclusive_wanted() } else { None };
        let mut fse_info = vk::SurfaceFullScreenExclusiveInfoEXT::default().full_screen_exclusive(if exclusive.is_some() {
            vk::FullScreenExclusiveEXT::APPLICATION_CONTROLLED
        } else {
            vk::FullScreenExclusiveEXT::DISALLOWED
        });
        let mut fse_win = vk::SurfaceFullScreenExclusiveWin32InfoEXT::default().hmonitor(exclusive.unwrap_or(0) as vk::HMONITOR);
        #[cfg(windows)]
        let present_mode = vk::PresentModeKHR::FIFO;
        #[cfg(target_os = "linux")]
        let present_mode = {
            let modes = c.surface_i.get_physical_device_surface_present_modes(c.pdev, p.surface).map_err(|e| format!("surface present modes: {e}"))?;
            let requested = std::env::var("DS_LINUX_PRESENT_MODE").ok();
            let mode = linux_present_mode(&modes, requested.as_deref());
            crate::info!("Linux swapchain modes {modes:?}; requested {:?}, selected {mode:?}; title pacing uses its own timer (no compositor vblank clock)", requested.as_deref().unwrap_or("fifo"));
            mode
        };
        let mut sci = vk::SwapchainCreateInfoKHR::default()
            .surface(p.surface)
            .min_image_count(count)
            .image_format(fmt.format)
            .image_color_space(fmt.color_space)
            .image_extent(extent)
            .image_array_layers(1)
            .image_usage(vk::ImageUsageFlags::TRANSFER_DST | vk::ImageUsageFlags::COLOR_ATTACHMENT | (caps.supported_usage_flags & vk::ImageUsageFlags::TRANSFER_SRC))
            .image_sharing_mode(vk::SharingMode::EXCLUSIVE)
            .pre_transform(caps.current_transform)
            .composite_alpha(vk::CompositeAlphaFlagsKHR::OPAQUE)
            .present_mode(present_mode)
            .clipped(true)
            .old_swapchain(old);
        if c.fse.is_some() {
            sci = sci.push_next(&mut fse_info);
            if exclusive.is_some() {
                sci = sci.push_next(&mut fse_win);
            }
        }
        let sc = c.swapchain_d.create_swapchain(&sci, None).map_err(|e| format!("create swapchain: {e}"))?;
        if old != vk::SwapchainKHR::null() {
            // The old swapchain's images are only the presenter's: its frames done and the queue
            // idle under the queue's lock (a device-wide idle wait touches the queue while the
            // GPU threads submit to it).
            c.device.wait_for_fences(&p.fences, true, u64::MAX).ok();
            {
                let q = c.queue.lock();
                c.device.queue_wait_idle(*q).ok();
            }
            // (No present wait on it meanwhile, and none after: `present_waiter`.)
            let _pw = PW_LOCK.lock();
            PW_GEN.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
            c.swapchain_d.destroy_swapchain(old, None);
            for s in p.done.drain(..) {
                c.device.destroy_semaphore(s, None);
            }
        }
        p.swapchain = sc;
        p.present_id = 0;
        p.hdr = fmt.color_space == vk::ColorSpaceKHR::HDR10_ST2084_EXT;
        if p.hdr {
            if let Some(m) = c.hdr_metadata.as_ref() {
                // HDR10's usual mastering description (BT.2020 primaries, D65 white); the title's
                // own Display Adjustment sets its brightness.
                let xy = |x: f32, y: f32| vk::XYColorEXT { x, y };
                let md = vk::HdrMetadataEXT::default()
                    .display_primary_red(xy(0.708, 0.292))
                    .display_primary_green(xy(0.170, 0.797))
                    .display_primary_blue(xy(0.131, 0.046))
                    .white_point(xy(0.3127, 0.3290))
                    .max_luminance(1000.0)
                    .min_luminance(0.001)
                    .max_content_light_level(1000.0)
                    .max_frame_average_light_level(400.0);
                m.set_hdr_metadata(&[sc], &[md]);
            }
        }
        p.images = c.swapchain_d.get_swapchain_images(sc).map_err(|e| format!("swapchain images: {e}"))?;
        p.format = fmt.format;
        p.extent = extent;
        for _ in 0..p.images.len() {
            p.done.push(c.device.create_semaphore(&vk::SemaphoreCreateInfo::default(), None).map_err(|e| e.to_string())?);
        }
        p.exclusive = None;
        if let (Some(f), Some(mon)) = (c.fse.as_ref(), exclusive) {
            match f.acquire_full_screen_exclusive_mode(sc) {
                Ok(()) => p.exclusive = Some(mon),
                Err(e) => crate::warn!("present: exclusive fullscreen not acquired ({e}): shown as borderless"),
            }
        }
        crate::info!(
            "swapchain {}x{} {:?} {:?} ({} images{})",
            extent.width,
            extent.height,
            fmt.format,
            fmt.color_space,
            p.images.len(),
            if p.exclusive.is_some() { ", exclusive fullscreen" } else { "" }
        );
    }
    Ok(())
}

/// Runs without a window (`--headless`: the drivers' measurement runs) present too: one
/// 64x64 image a flip to a swapchain on a window that is never shown. A process that
/// presents nothing is left at about a third of the GPU's clock rate by the driver's power
/// management while the CPU side limits its frame, and every batch completes late: 48
/// flips/s where a run with a window makes 62 (FINDINGS.md 2026-11-17 night). `tune
/// headless_present 0` (`DS_HEADLESS_PRESENT=0`): none, as until then.
///
/// It runs on a thread of its own (`gpu-hidden`, woken by each flip; the vblank thread never
/// waits for it) and a queue of its own (`Ctx::present_queue`, a second queue of the graphics
/// family; none on a device with one: then nothing is presented). Its image is asked for with
/// a fence and presented only once free (a later flip if not yet), at most one in `HIDDEN_GAP`.
/// (2026-11-20: presented from the vblank thread on the main queue, a present twice blocked in
/// the driver while holding the main queue's lock, its own submission queued behind the main
/// queue's work, which waited for submissions the submission thread could not make without that
/// lock: the GPU idled, the game hung, and the ended process kept the driver's threads and its
/// video memory until a restart; once at ~90 flips a second, once at 59.94 beside 19 compile
/// threads.)
struct Hidden {
    swapchain: vk::SwapchainKHR,
    cmds: Vec<vk::CommandBuffer>,
    fences: Vec<vk::Fence>,
    /// Per slot: the fence an image asked for signals once it is free.
    acquired: Vec<vk::Fence>,
    done: Vec<vk::Semaphore>,
    images: Vec<vk::Image>,
    frame: usize,
    /// An image asked for and its slot, not free yet at the flip that asked.
    pending: Option<(u32, usize)>,
    last: Option<std::time::Instant>,
    presented: u64,
    skipped: u64,
    held: u64,
    waited: u64,
}

/// The least time between two of `Hidden`'s presents (some 83 a second at most: every flip at
/// 59.94 Hz, every other one of the fast-forward's 120).
const HIDDEN_GAP: std::time::Duration = std::time::Duration::from_millis(12);
// (Vulkan handles; used under the lock.)
unsafe impl Send for Hidden {}

static HIDDEN: parking_lot::Mutex<Option<Hidden>> = parking_lot::Mutex::new(None);
static HIDDEN_FAILED: std::sync::atomic::AtomicBool = std::sync::atomic::AtomicBool::new(false);

unsafe fn hidden_make() -> Result<Hidden, String> {
    let c = ctx();
    let (hwnd, hinstance) = window::create_hidden()?;
    unsafe {
        let surface = window::surface(c, hwnd, hinstance)?;
        if !c.surface_i.get_physical_device_surface_support(c.pdev, c.queue_family, surface).unwrap_or(false) {
            return Err("queue family cannot present".into());
        }
        let caps = c.surface_i.get_physical_device_surface_capabilities(c.pdev, surface).map_err(|e| format!("surface caps: {e}"))?;
        let formats = c.surface_i.get_physical_device_surface_formats(c.pdev, surface).map_err(|e| format!("surface formats: {e}"))?;
        let fmt = formats.first().copied().ok_or("no surface format")?;
        let modes = c.surface_i.get_physical_device_surface_present_modes(c.pdev, surface).unwrap_or_default();
        // (Never held back by a display's refresh: the flips have their own clock.)
        let mode = [vk::PresentModeKHR::IMMEDIATE, vk::PresentModeKHR::MAILBOX].into_iter().find(|m| modes.contains(m)).unwrap_or(vk::PresentModeKHR::FIFO);
        let extent = if caps.current_extent.width == u32::MAX || caps.current_extent.width == 0 { vk::Extent2D { width: 64, height: 64 } } else { caps.current_extent };
        let count = (caps.min_image_count + 1).min(if caps.max_image_count == 0 { u32::MAX } else { caps.max_image_count });
        let sci = vk::SwapchainCreateInfoKHR::default()
            .surface(surface)
            .min_image_count(count)
            .image_format(fmt.format)
            .image_color_space(fmt.color_space)
            .image_extent(extent)
            .image_array_layers(1)
            .image_usage(vk::ImageUsageFlags::COLOR_ATTACHMENT)
            .image_sharing_mode(vk::SharingMode::EXCLUSIVE)
            .pre_transform(caps.current_transform)
            .composite_alpha(vk::CompositeAlphaFlagsKHR::OPAQUE)
            .present_mode(mode)
            .clipped(true);
        let swapchain = c.swapchain_d.create_swapchain(&sci, None).map_err(|e| format!("create swapchain: {e}"))?;
        let images = c.swapchain_d.get_swapchain_images(swapchain).map_err(|e| format!("swapchain images: {e}"))?;
        let pool = c
            .device
            .create_command_pool(&vk::CommandPoolCreateInfo::default().queue_family_index(c.queue_family).flags(vk::CommandPoolCreateFlags::RESET_COMMAND_BUFFER), None)
            .map_err(|e| e.to_string())?;
        let cmds = c.device.allocate_command_buffers(&vk::CommandBufferAllocateInfo::default().command_pool(pool).command_buffer_count(FRAMES as u32)).map_err(|e| e.to_string())?;
        let sem = || c.device.create_semaphore(&vk::SemaphoreCreateInfo::default(), None).map_err(|e| e.to_string());
        let mut h = Hidden { swapchain, cmds, fences: Vec::new(), acquired: Vec::new(), done: Vec::new(), images, frame: 0, pending: None, last: None, presented: 0, skipped: 0, held: 0, waited: 0 };
        for _ in 0..FRAMES {
            h.fences.push(c.device.create_fence(&vk::FenceCreateInfo::default().flags(vk::FenceCreateFlags::SIGNALED), None).map_err(|e| e.to_string())?);
            h.acquired.push(c.device.create_fence(&vk::FenceCreateInfo::default(), None).map_err(|e| e.to_string())?);
        }
        // (A present's semaphore per image, as the window's presenter has them: an image
        // comes back only after its present let go of the semaphore.)
        for _ in 0..h.images.len() {
            h.done.push(sem()?);
        }
        crate::info!("present: no window: one {}x{} image a flip presented to a swapchain that is never shown ({mode:?}, {} images; `tune headless_present`)", extent.width, extent.height, h.images.len());
        Ok(h)
    }
}

/// Flips seen by the vblank thread (`headless_present`), for the hidden present's thread.
static HIDDEN_KICK: (parking_lot::Mutex<u64>, parking_lot::Condvar) = (parking_lot::Mutex::new(0), parking_lot::Condvar::new());
static HIDDEN_THREAD: std::sync::Once = std::sync::Once::new();

/// A flip of a run without a window: the hidden present's thread woken (started at the first),
/// nothing waited for here.
fn headless_present() {
    if !crate::gpu::tune::headless_present() || HIDDEN_FAILED.load(std::sync::atomic::Ordering::Relaxed) {
        return;
    }
    HIDDEN_THREAD.call_once(|| {
        if ctx().present_queue.is_none() {
            crate::info!("present: no window and no second queue of the graphics family: nothing presented (the driver may keep the GPU's clock low)");
            HIDDEN_FAILED.store(true, std::sync::atomic::Ordering::Relaxed);
            return;
        }
        let _ = crate::thread::spawn_host("gpu-hidden", 4 << 20, Box::new(|| {
            crate::log::set_thread_name("gpu-hidden");
            let mut seen = 0u64;
            loop {
                {
                    let mut k = HIDDEN_KICK.0.lock();
                    while *k == seen {
                        HIDDEN_KICK.1.wait(&mut k);
                    }
                    seen = *k;
                }
                hidden_present_once();
                if HIDDEN_FAILED.load(std::sync::atomic::Ordering::Relaxed) {
                    return;
                }
            }
        }));
    });
    *HIDDEN_KICK.0.lock() += 1;
    HIDDEN_KICK.1.notify_one();
}

/// The hidden present's work for a flip (its own thread): skipped while the frame slot's last
/// submission or a free image is not there yet, or the last present was less than `HIDDEN_GAP`
/// ago (nothing here waits, on the CPU or on the GPU).
fn hidden_present_once() {
    let c = ctx();
    let Some(pq) = c.present_queue.as_ref() else { return };
    let mut g = HIDDEN.lock();
    if g.is_none() {
        match unsafe { hidden_make() } {
            Ok(h) => *g = Some(h),
            Err(e) => {
                crate::warn!("present: no window and no hidden swapchain ({e})");
                HIDDEN_FAILED.store(true, std::sync::atomic::Ordering::Relaxed);
                return;
            }
        }
    }
    let h = g.as_mut().unwrap();
    if h.last.is_some_and(|t| t.elapsed() < HIDDEN_GAP) {
        h.held += 1;
        return;
    }
    unsafe {
        // (An image asked for at an earlier flip: presented once it is free.)
        let (idx, f) = match h.pending {
            Some(p) => p,
            None => {
                let f = h.frame % FRAMES;
                if c.device.get_fence_status(h.fences[f]) != Ok(true) {
                    h.skipped += 1;
                    return;
                }
                match c.swapchain_d.acquire_next_image(h.swapchain, 0, vk::Semaphore::null(), h.acquired[f]) {
                    Ok((i, _)) => {
                        h.pending = Some((i, f));
                        (i, f)
                    }
                    Err(_) => {
                        h.skipped += 1;
                        return;
                    }
                }
            }
        };
        if c.device.get_fence_status(h.acquired[f]) != Ok(true) {
            h.waited += 1;
            return;
        }
        h.pending = None;
        h.frame += 1;
        let cmd = h.cmds[f];
        c.device.reset_fences(&[h.fences[f], h.acquired[f]]).ok();
        c.device.reset_command_buffer(cmd, vk::CommandBufferResetFlags::empty()).ok();
        c.device.begin_command_buffer(cmd, &vk::CommandBufferBeginInfo::default().flags(vk::CommandBufferUsageFlags::ONE_TIME_SUBMIT)).ok();
        let range = vk::ImageSubresourceRange { aspect_mask: vk::ImageAspectFlags::COLOR, base_mip_level: 0, level_count: 1, base_array_layer: 0, layer_count: 1 };
        let to_present = vk::ImageMemoryBarrier::default()
            .image(h.images[idx as usize])
            .old_layout(vk::ImageLayout::UNDEFINED)
            .new_layout(vk::ImageLayout::PRESENT_SRC_KHR)
            .src_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
            .dst_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
            .subresource_range(range);
        c.device.cmd_pipeline_barrier(cmd, vk::PipelineStageFlags::COLOR_ATTACHMENT_OUTPUT, vk::PipelineStageFlags::BOTTOM_OF_PIPE, vk::DependencyFlags::empty(), &[], &[], &[to_present]);
        c.device.end_command_buffer(cmd).ok();
        // (No wait: the image's fence said it is free.)
        let signal = [h.done[idx as usize]];
        let cmds = [cmd];
        let si = vk::SubmitInfo::default().command_buffers(&cmds).signal_semaphores(&signal);
        let scs = [h.swapchain];
        let idxs = [idx];
        let pi = vk::PresentInfoKHR::default().wait_semaphores(&signal).swapchains(&scs).image_indices(&idxs);
        let q = pq.lock();
        if c.device.queue_submit(*q, &[si], h.fences[f]).is_ok() {
            let _ = c.swapchain_d.queue_present(*q, &pi);
            h.presented += 1;
            h.last = Some(std::time::Instant::now());
        }
    }
}

/// Status part: the hidden swapchain's presents (`headless_present`).
pub fn hidden_line() -> String {
    // (Without waiting: a present stuck in the driver holds the lock, and every exit reads this.)
    let Some(g) = HIDDEN.try_lock() else { return "    hidden swapchain: busy (a present in the driver)\n".to_string() };
    match g.as_ref() {
        Some(h) => format!("hidden swapchain: {} flips presented, {} skipped (no free image or the slot's submission still running), {} within {} ms of the last, {} waited for their image to be free", h.presented, h.skipped, h.held, HIDDEN_GAP.as_millis(), h.waited),
        None => String::new(),
    }
}

pub fn init(title: &str) -> Result<(), String> {
    super::init()?;
    let w = window::create(title, 1280, 720)?;
    let c = ctx();
    unsafe {
        let surface = window::surface(c, w.hwnd, w.hinstance)?;
        let supported = c.surface_i.get_physical_device_surface_support(c.pdev, c.queue_family, surface).unwrap_or(false);
        if !supported {
            return Err("queue family cannot present".into());
        }
        let pool = c
            .device
            .create_command_pool(&vk::CommandPoolCreateInfo::default().queue_family_index(c.queue_family).flags(vk::CommandPoolCreateFlags::RESET_COMMAND_BUFFER), None)
            .map_err(|e| e.to_string())?;
        let cmds = c
            .device
            .allocate_command_buffers(&vk::CommandBufferAllocateInfo::default().command_pool(pool).command_buffer_count(FRAMES as u32))
            .map_err(|e| e.to_string())?;
        let mut p = Presenter {
            surface,
            swapchain: vk::SwapchainKHR::null(),
            images: Vec::new(),
            format: vk::Format::UNDEFINED,
            extent: vk::Extent2D::default(),
            pool,
            cmds,
            fences: (0..FRAMES).map(|_| c.device.create_fence(&vk::FenceCreateInfo::default().flags(vk::FenceCreateFlags::SIGNALED), None).unwrap()).collect(),
            acquire: (0..FRAMES).map(|_| c.device.create_semaphore(&vk::SemaphoreCreateInfo::default(), None).unwrap()).collect(),
            done: Vec::new(),
            frame: 0,
            staging: None,
            upload: None,
            halves: Vec::new(),
            logged: (0, 0, 0, 0),
            sharpen: None,
            sharpen_failed: false,
            exclusive: None,
            hdr: false,
            want_hdr: false,
            watch: None,
            watch_failed: false,
            text_box: None,
            present_id: 0,
        };
        super::framewatch::init_from_env();
        create_swapchain(&mut p)?;
        *PRESENTER.lock() = Some(p);
    }
    ACTIVE.store(true, std::sync::atomic::Ordering::Relaxed);
    Ok(())
}

fn barrier(c: &super::Ctx, cmd: vk::CommandBuffer, img: vk::Image, old: vk::ImageLayout, new: vk::ImageLayout) {
    let b = vk::ImageMemoryBarrier::default()
        .src_access_mask(vk::AccessFlags::MEMORY_WRITE)
        .dst_access_mask(vk::AccessFlags::MEMORY_READ | vk::AccessFlags::MEMORY_WRITE)
        .old_layout(old)
        .new_layout(new)
        .src_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
        .dst_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
        .image(img)
        .subresource_range(vk::ImageSubresourceRange { aspect_mask: vk::ImageAspectFlags::COLOR, base_mip_level: 0, level_count: 1, base_array_layer: 0, layer_count: 1 });
    unsafe {
        c.device.cmd_pipeline_barrier(cmd, vk::PipelineStageFlags::ALL_COMMANDS, vk::PipelineStageFlags::ALL_COMMANDS, vk::DependencyFlags::empty(), &[], &[], &[b]);
    }
}

/// Presents one frame. Returns false if presentation is not possible (minimized).
pub fn present(src: Source) -> bool {
    let c = ctx();
    let mut g = PRESENTER.lock();
    let Some(p) = g.as_mut() else {
        drop(g);
        headless_present();
        // Without a window: the frame watch still sees each frame (`framewatch::headless`).
        if let Source::Image { image, layout, width, height, .. } = src {
            super::framewatch::init_from_env();
            super::framewatch::headless(image, layout, width, height);
        }
        return false;
    };
    unsafe {
        // HDR frames from SDR ones or back (the title's display buffers' format): a new swapchain.
        if let Source::Image { hdr, .. } = &src {
            if *hdr != p.want_hdr {
                p.want_hdr = *hdr;
                crate::info!("present: the title's frames are {}", if *hdr { "HDR10" } else { "SDR" });
                window::RESIZED.store(true, std::sync::atomic::Ordering::SeqCst);
            }
        }
        // The window's exclusive fullscreen changed (Alt+Tab, the mode): a new swapchain.
        if window::exclusive_wanted() != p.exclusive && c.fse.is_some() {
            window::RESIZED.store(true, std::sync::atomic::Ordering::SeqCst);
        }
        if window::RESIZED.swap(false, std::sync::atomic::Ordering::SeqCst) {
            if let Err(e) = create_swapchain(p) {
                crate::debug!("swapchain recreate: {e}");
                return false;
            }
        }
        #[cfg(target_os = "linux")]
        let frame_start = std::time::Instant::now();
        let f = p.frame % FRAMES;
        p.frame += 1;
        c.device.wait_for_fences(&[p.fences[f]], true, u64::MAX).ok();
        if let Some(w) = p.watch.as_mut() {
            w.collect(f);
        }
        #[cfg(target_os = "linux")]
        let acquire_start = std::time::Instant::now();
        let (idx, _) = match c.swapchain_d.acquire_next_image(p.swapchain, u64::MAX, p.acquire[f], vk::Fence::null()) {
            Ok(r) => r,
            Err(vk::Result::ERROR_OUT_OF_DATE_KHR | vk::Result::ERROR_FULL_SCREEN_EXCLUSIVE_MODE_LOST_EXT) => {
                window::RESIZED.store(true, std::sync::atomic::Ordering::SeqCst);
                return false;
            }
            Err(e) => {
                crate::error!("acquire: {e}");
                return false;
            }
        };
        #[cfg(target_os = "linux")]
        let record_start = std::time::Instant::now();
        c.device.reset_fences(&[p.fences[f]]).ok();
        let cmd = p.cmds[f];
        c.device.reset_command_buffer(cmd, vk::CommandBufferResetFlags::empty()).ok();
        c.device.begin_command_buffer(cmd, &vk::CommandBufferBeginInfo::default().flags(vk::CommandBufferUsageFlags::ONE_TIME_SUBMIT)).ok();
        let dst = p.images[idx as usize];
        barrier(c, cmd, dst, vk::ImageLayout::UNDEFINED, vk::ImageLayout::TRANSFER_DST_OPTIMAL);
        let (src_img, src_format, src_layout, sw, sh) = match src {
            Source::Image { image, format, layout, width, height, .. } => {
                // GENERAL images are blitted in place (after a barrier that
                // makes earlier GPU writes visible).
                let to = if layout == vk::ImageLayout::GENERAL { layout } else { vk::ImageLayout::TRANSFER_SRC_OPTIMAL };
                barrier(c, cmd, image, layout, to);
                (image, format, layout, width, height)
            }
            Source::Pixels { data, width, height } => match upload_pixels(p, cmd, &data, width, height) {
                Ok(img) => (img, vk::Format::B8G8R8A8_SRGB, vk::ImageLayout::TRANSFER_SRC_OPTIMAL, width, height),
                Err(e) => {
                    crate::error!("present upload: {e}");
                    c.device.end_command_buffer(cmd).ok();
                    return false;
                }
            },
        };
        let blit_layout = if src_layout == vk::ImageLayout::GENERAL { src_layout } else { vk::ImageLayout::TRANSFER_SRC_OPTIMAL };
        let fitted = crate::gpu::tune::present_fit();
        let (rx, ry, rw, rh) = if fitted { fit(sw, sh, p.extent.width, p.extent.height) } else { (0, 0, p.extent.width, p.extent.height) };
        if (rw, rh) != (p.extent.width, p.extent.height) {
            let black = vk::ClearColorValue { float32: [0.0, 0.0, 0.0, 1.0] };
            let range = vk::ImageSubresourceRange { aspect_mask: vk::ImageAspectFlags::COLOR, base_mip_level: 0, level_count: 1, base_array_layer: 0, layer_count: 1 };
            c.device.cmd_clear_color_image(cmd, dst, vk::ImageLayout::TRANSFER_DST_OPTIMAL, &black, &[range]);
            barrier(c, cmd, dst, vk::ImageLayout::TRANSFER_DST_OPTIMAL, vk::ImageLayout::TRANSFER_DST_OPTIMAL);
        }
        // Halvings while the frame is more than twice the destination's size.
        let (mut from, mut from_layout, mut fw, mut fh) = (src_img, blit_layout, sw, sh);
        let mut level = 0;
        while fitted && (fw > 2 * rw || fh > 2 * rh) {
            let (hw, hh) = (fw.div_ceil(2), fh.div_ceil(2));
            let half = match half_image(p, f, level, src_format, hw, hh) {
                Ok(h) => h,
                Err(e) => {
                    static ONCE: std::sync::Once = std::sync::Once::new();
                    ONCE.call_once(|| crate::warn!("present: no {hw}x{hh} halving image ({e}): one blit"));
                    break;
                }
            };
            barrier(c, cmd, half, vk::ImageLayout::UNDEFINED, vk::ImageLayout::TRANSFER_DST_OPTIMAL);
            blit(c, cmd, from, from_layout, (0, 0, fw, fh), half, (0, 0, hw, hh));
            barrier(c, cmd, half, vk::ImageLayout::TRANSFER_DST_OPTIMAL, vk::ImageLayout::TRANSFER_SRC_OPTIMAL);
            (from, from_layout, fw, fh) = (half, vk::ImageLayout::TRANSFER_SRC_OPTIMAL, hw, hh);
            level += 1;
        }
        // The PC settings' sharpening: the frame blitted at its shown size, sharpened, blitted
        // to the window.
        let amount = super::sharpen::amount();
        let mut sharpened = false;
        if amount > 0.0 && !p.sharpen_failed {
            if p.sharpen.is_none() {
                match super::sharpen::Sharpen::new() {
                    Ok(s) => p.sharpen = Some(s),
                    Err(e) => {
                        crate::warn!("present: no sharpening ({e})");
                        p.sharpen_failed = true;
                    }
                }
            }
            let fences: Vec<vk::Fence> = p.fences.iter().enumerate().filter(|&(i, _)| i != f).map(|(_, &x)| x).collect();
            if let Some(s) = p.sharpen.as_mut() {
                match s.ensure(rw, rh, || {
                    c.device.wait_for_fences(&fences, true, u64::MAX).ok();
                }) {
                    Ok((a, b, fresh)) => {
                        s.begin(cmd, fresh);
                        blit_to(c, cmd, from, from_layout, (0, 0, fw, fh), a, vk::ImageLayout::GENERAL, (0, 0, rw, rh));
                        s.record(cmd, amount);
                        blit(c, cmd, b, vk::ImageLayout::GENERAL, (0, 0, rw, rh), dst, (rx, ry, rw, rh));
                        sharpened = true;
                    }
                    Err(e) => {
                        crate::warn!("present: no sharpening images ({e})");
                        p.sharpen_failed = true;
                    }
                }
            }
        }
        if !sharpened {
            blit(c, cmd, from, from_layout, (0, 0, fw, fh), dst, (rx, ry, rw, rh));
        }
        // The frame watch's copy of the frame as shown (before the window's scaling).
        if super::framewatch::ON.load(std::sync::atomic::Ordering::Relaxed) && !p.watch_failed {
            if p.watch.is_none() {
                match super::framewatch::Watch::new(FRAMES) {
                    Ok(w) => p.watch = Some(w),
                    Err(e) => {
                        crate::warn!("frame watch: unavailable ({e})");
                        p.watch_failed = true;
                    }
                }
            }
            if let Some(w) = p.watch.as_mut() {
                w.record(cmd, f, from, from_layout, fw, fh);
            }
        }
        // The system text box (the character's name) over the frame.
        draw_text_box(p, cmd, f, dst, (rx, ry, rw, rh));
        if p.logged != (sw, sh, p.extent.width, p.extent.height) {
            p.logged = (sw, sh, p.extent.width, p.extent.height);
            crate::info!("present: {sw}x{sh} frame into a {}x{} window at {rw}x{rh}+{rx}+{ry}: {level} halving(s), then {fw}x{fh} -> {rw}x{rh}", p.extent.width, p.extent.height);
        }
        if src_layout != vk::ImageLayout::TRANSFER_SRC_OPTIMAL && src_layout != vk::ImageLayout::GENERAL && src_img != vk::Image::null() {
            barrier(c, cmd, src_img, vk::ImageLayout::TRANSFER_SRC_OPTIMAL, src_layout);
        }
        // Window shot: copy the swapchain image to a host buffer.
        let shot = if window_shot(p.frame) {
            let (w, h) = (p.extent.width, p.extent.height);
            match c.host_buffer(w as u64 * h as u64 * 4, vk::BufferUsageFlags::TRANSFER_DST) {
                Ok((buf, mem, ptr)) => {
                    barrier(c, cmd, dst, vk::ImageLayout::TRANSFER_DST_OPTIMAL, vk::ImageLayout::TRANSFER_SRC_OPTIMAL);
                    let region = vk::BufferImageCopy::default()
                        .image_subresource(vk::ImageSubresourceLayers { aspect_mask: vk::ImageAspectFlags::COLOR, mip_level: 0, base_array_layer: 0, layer_count: 1 })
                        .image_extent(vk::Extent3D { width: w, height: h, depth: 1 });
                    c.device.cmd_copy_image_to_buffer(cmd, dst, vk::ImageLayout::TRANSFER_SRC_OPTIMAL, buf, &[region]);
                    barrier(c, cmd, dst, vk::ImageLayout::TRANSFER_SRC_OPTIMAL, vk::ImageLayout::PRESENT_SRC_KHR);
                    Some((buf, mem, ptr, w, h))
                }
                Err(e) => {
                    crate::warn!("window shot: {e}");
                    barrier(c, cmd, dst, vk::ImageLayout::TRANSFER_DST_OPTIMAL, vk::ImageLayout::PRESENT_SRC_KHR);
                    None
                }
            }
        } else {
            barrier(c, cmd, dst, vk::ImageLayout::TRANSFER_DST_OPTIMAL, vk::ImageLayout::PRESENT_SRC_KHR);
            None
        };
        c.device.end_command_buffer(cmd).ok();
        let done = p.done[idx as usize];
        // The acquire orders the first layout transition as well as the blit.
        #[cfg(target_os = "linux")]
        let acquire_stage = vk::PipelineStageFlags::ALL_COMMANDS;
        #[cfg(windows)]
        let acquire_stage = vk::PipelineStageFlags::TRANSFER;
        if let Err(e) = c.submit(cmd, &[(p.acquire[f], acquire_stage)], &[done], p.fences[f]) {
            crate::error!("{e}");
            return false;
        }
        #[cfg(target_os = "linux")]
        let ready_start = std::time::Instant::now();
        // Earlier queue submissions can wait on cpu_done, signalled from the completion
        // thread after their CPU effects. WSI binary waits cannot depend on a timeline
        // signal that has not happened/submitted yet (VUID-03268). Wait outside the queue
        // lock so the completion/submission threads can resolve those dependencies.
        #[cfg(target_os = "linux")]
        if let Err(e) = c.device.wait_for_fences(&[p.fences[f]], true, u64::MAX) {
            crate::error!("present completion fence: {e}");
            return false;
        }
        #[cfg(target_os = "linux")]
        let ready_end = std::time::Instant::now();
        if let Some((buf, mem, ptr, w, h)) = shot {
            #[cfg(windows)]
            c.device.wait_for_fences(&[p.fences[f]], true, 5_000_000_000).ok();
            let raw = std::slice::from_raw_parts(ptr, (w * h * 4) as usize).to_vec();
            if let Some(rgba) = crate::gpu::images::to_rgba8(p.format, &raw) {
                if let Some(dir) = crate::run_dir() {
                    let path = dir.join(format!("window-{}.png", p.frame));
                    if std::fs::write(&path, crate::gpu::png::encode(w, h, &rgba)).is_ok() {
                        crate::info!("window shot of presented frame {}: {}", p.frame, path.display());
                    }
                }
            }
            c.device.destroy_buffer(buf, None);
            c.device.unmap_memory(mem);
            c.device.free_memory(mem, None);
        }
        let scs = [p.swapchain];
        let idxs = [idx];
        let waits = [done];
        // (`DS_PRESENT_STATS=1`: the present's id for `present_waiter`.)
        let pw = super::PRESENT_WAIT.load(std::sync::atomic::Ordering::Relaxed);
        if pw {
            p.present_id += 1;
        }
        let ids = [p.present_id];
        let mut pid = vk::PresentIdKHR::default().present_ids(&ids);
        let mut pi = vk::PresentInfoKHR::default().wait_semaphores(&waits).swapchains(&scs).image_indices(&idxs);
        if pw {
            pi = pi.push_next(&mut pid);
        }
        let q = c.queue.lock();
        // (Through Streamline's present once DLSS runs: its per-frame bookkeeping.)
        #[cfg(target_os = "linux")]
        let present_start = std::time::Instant::now();
        let presented = match crate::gpu::upscale::present_hook() {
            Some(f) => match f(*q, &pi) {
                vk::Result::SUCCESS => Ok(false),
                vk::Result::SUBOPTIMAL_KHR => Ok(true),
                e => Err(e),
            },
            None => c.swapchain_d.queue_present(*q, &pi),
        };
        drop(q);
        #[cfg(target_os = "linux")]
        {
            if presented.is_ok() {
                crate::platform::present_timing::submit(if pw { p.present_id } else { p.frame as u64 }, [frame_start, acquire_start, record_start, ready_start, ready_end, present_start, std::time::Instant::now()]);
            }
            use std::sync::atomic::{AtomicU64, Ordering::Relaxed};
            static CALLS: AtomicU64 = AtomicU64::new(0);
            static NS: AtomicU64 = AtomicU64::new(0);
            let elapsed = present_start.elapsed().as_nanos() as u64;
            let n = CALLS.fetch_add(1, Relaxed) + 1;
            let total = NS.fetch_add(elapsed, Relaxed) + elapsed;
            if n <= 8 || n % 120 == 0 || presented.is_err() {
                crate::info!("Linux present #{n}: result={presented:?}, call {:.3} ms, average {:.3} ms, image {}; slot {:.3} ms acquire {:.3} ms record/submit {:.3} ms ready {:.3} ms capture/queue {:.3} ms; {}", elapsed as f64 / 1e6, total as f64 / n as f64 / 1e6, idxs[0],
                    (acquire_start - frame_start).as_secs_f64() * 1000.0,
                    (record_start - acquire_start).as_secs_f64() * 1000.0,
                    (ready_start - record_start).as_secs_f64() * 1000.0,
                    (ready_end - ready_start).as_secs_f64() * 1000.0,
                    (present_start - ready_end).as_secs_f64() * 1000.0, stats_line());
            }
        }
        if pw && presented.is_ok() {
            use ash::vk::Handle;
            let mut tx = PW_TX.lock();
            if tx.is_none() {
                let (s, r) = std::sync::mpsc::channel();
                std::thread::Builder::new().name("present-wait".into()).spawn(move || present_waiter(r)).ok();
                *tx = Some(s);
            }
            if let Some(s) = tx.as_ref() {
                s.send((PW_GEN.load(std::sync::atomic::Ordering::Relaxed), p.swapchain.as_raw(), p.present_id, std::time::Instant::now())).ok();
            }
        }
        match presented {
            Ok(false) => {}
            Ok(true) | Err(vk::Result::ERROR_OUT_OF_DATE_KHR | vk::Result::ERROR_FULL_SCREEN_EXCLUSIVE_MODE_LOST_EXT) => window::RESIZED.store(true, std::sync::atomic::Ordering::SeqCst),
            Err(e) => crate::error!("present: {e}"),
        }
    }
    true
}

/// Where a `sw`x`sh` frame goes in a `dw`x`dh` window (x, y, width, height):
/// as large as fits with its aspect kept, centred; the whole window when the
/// aspects differ by under 1%.
/// A window shows the frames (`init` succeeded): the render scale follows its area.
static ACTIVE: std::sync::atomic::AtomicBool = std::sync::atomic::AtomicBool::new(false);

/// A window shows the frames.
pub fn active() -> bool {
    ACTIVE.load(std::sync::atomic::Ordering::Relaxed)
}

/// The window's area the title's frame fills (its output's shape; pixels; None without a window).
pub fn area() -> Option<(u32, u32)> {
    use std::sync::atomic::Ordering::Relaxed;
    if !ACTIVE.load(Relaxed) {
        return None;
    }
    let (w, h) = (window::CLIENT_W.load(Relaxed), window::CLIENT_H.load(Relaxed));
    if w < 64 || h < 64 {
        return None;
    }
    let (ow, oh) = crate::dispmode::output_size();
    let (_, _, fw, fh) = fit(ow, oh, w, h);
    Some((fw, fh))
}

pub fn fit(sw: u32, sh: u32, dw: u32, dh: u32) -> (u32, u32, u32, u32) {
    let (a, b) = (sw as u64 * dh as u64, dw as u64 * sh as u64);
    if sw == 0 || sh == 0 || a.abs_diff(b) * 100 < a.max(b) {
        return (0, 0, dw, dh);
    }
    if a > b {
        // Wider than the window: bars above and below.
        let h = ((dw as u64 * sh as u64 + sw as u64 / 2) / sw as u64).clamp(1, dh as u64) as u32;
        (0, (dh - h) / 2, dw, h)
    } else {
        let w = ((dh as u64 * sw as u64 + sh as u64 / 2) / sh as u64).clamp(1, dw as u64) as u32;
        ((dw - w) / 2, 0, w, dh)
    }
}

/// A linear blit of rectangle `s` (x, y, width, height) of `src` to rectangle
/// `d` of `dst` (in TRANSFER_DST_OPTIMAL).
fn blit(c: &super::Ctx, cmd: vk::CommandBuffer, src: vk::Image, src_layout: vk::ImageLayout, s: (u32, u32, u32, u32), dst: vk::Image, d: (u32, u32, u32, u32)) {
    blit_to(c, cmd, src, src_layout, s, dst, vk::ImageLayout::TRANSFER_DST_OPTIMAL, d);
}

#[allow(clippy::too_many_arguments)]
fn blit_to(c: &super::Ctx, cmd: vk::CommandBuffer, src: vk::Image, src_layout: vk::ImageLayout, s: (u32, u32, u32, u32), dst: vk::Image, dst_layout: vk::ImageLayout, d: (u32, u32, u32, u32)) {
    let layers = vk::ImageSubresourceLayers { aspect_mask: vk::ImageAspectFlags::COLOR, mip_level: 0, base_array_layer: 0, layer_count: 1 };
    let corners = |(x, y, w, h): (u32, u32, u32, u32)| [vk::Offset3D { x: x as i32, y: y as i32, z: 0 }, vk::Offset3D { x: (x + w) as i32, y: (y + h) as i32, z: 1 }];
    let region = vk::ImageBlit { src_subresource: layers, src_offsets: corners(s), dst_subresource: layers, dst_offsets: corners(d) };
    unsafe { c.device.cmd_blit_image(cmd, src, src_layout, dst, dst_layout, &[region], vk::Filter::LINEAR) };
}

/// The halving chain's image `level` at `w`x`h`: the frame's format when it
/// can be blitted to and from, else 10:10:10:2. Made or remade on demand
/// (after the presenter's other frames in flight, the only other users, are
/// done; `current` is the frame being recorded).
unsafe fn half_image(p: &mut Presenter, current: usize, level: usize, format: vk::Format, w: u32, h: u32) -> Result<vk::Image, String> {
    let c = ctx();
    let blittable = |f: vk::Format| {
        let props = unsafe { c.instance.get_physical_device_format_properties(c.pdev, f) };
        props.optimal_tiling_features.contains(vk::FormatFeatureFlags::BLIT_SRC | vk::FormatFeatureFlags::BLIT_DST | vk::FormatFeatureFlags::SAMPLED_IMAGE_FILTER_LINEAR)
    };
    let format = if blittable(format) { format } else { vk::Format::A2B10G10R10_UNORM_PACK32 };
    if let Some(h_) = p.halves.get(level) {
        if (h_.format, h_.width, h_.height) == (format, w, h) {
            return Ok(h_.image);
        }
    }
    unsafe {
        let others: Vec<vk::Fence> = p.fences.iter().enumerate().filter(|&(i, _)| i != current).map(|(_, &f)| f).collect();
        c.device.wait_for_fences(&others, true, u64::MAX).ok();
        for old in p.halves.drain(level..) {
            c.device.destroy_image(old.image, None);
            c.device.free_memory(old.memory, None);
        }
        let ici = vk::ImageCreateInfo::default()
            .image_type(vk::ImageType::TYPE_2D)
            .format(format)
            .extent(vk::Extent3D { width: w, height: h, depth: 1 })
            .mip_levels(1)
            .array_layers(1)
            .samples(vk::SampleCountFlags::TYPE_1)
            .tiling(vk::ImageTiling::OPTIMAL)
            .usage(vk::ImageUsageFlags::TRANSFER_DST | vk::ImageUsageFlags::TRANSFER_SRC);
        let image = c.device.create_image(&ici, None).map_err(|e| e.to_string())?;
        let memory = match c.alloc(c.device.get_image_memory_requirements(image), vk::MemoryPropertyFlags::DEVICE_LOCAL) {
            Ok(m) => m,
            Err(e) => {
                c.device.destroy_image(image, None);
                return Err(e);
            }
        };
        c.device.bind_image_memory(image, memory, 0).map_err(|e| e.to_string())?;
        p.halves.push(Half { image, memory, format, width: w, height: h });
        Ok(image)
    }
}

unsafe fn upload_pixels(p: &mut Presenter, cmd: vk::CommandBuffer, data: &[u8], w: u32, h: u32) -> Result<vk::Image, String> {
    let c = ctx();
    let size = (w * h * 4) as u64;
    unsafe {
        if p.staging.as_ref().is_none_or(|s| s.3 < size) {
            let (b, m, ptr) = c.host_buffer(size, vk::BufferUsageFlags::TRANSFER_SRC)?;
            p.staging = Some((b, m, ptr, size));
        }
        if p.upload.as_ref().is_none_or(|u| u.2 != w || u.3 != h) {
            let ici = vk::ImageCreateInfo::default()
                .image_type(vk::ImageType::TYPE_2D)
                .format(vk::Format::B8G8R8A8_SRGB)
                .extent(vk::Extent3D { width: w, height: h, depth: 1 })
                .mip_levels(1)
                .array_layers(1)
                .samples(vk::SampleCountFlags::TYPE_1)
                .tiling(vk::ImageTiling::OPTIMAL)
                .usage(vk::ImageUsageFlags::TRANSFER_DST | vk::ImageUsageFlags::TRANSFER_SRC);
            let img = c.device.create_image(&ici, None).map_err(|e| e.to_string())?;
            let mem = c.alloc(c.device.get_image_memory_requirements(img), vk::MemoryPropertyFlags::DEVICE_LOCAL)?;
            c.device.bind_image_memory(img, mem, 0).map_err(|e| e.to_string())?;
            p.upload = Some((img, mem, w, h));
        }
        let (sb, _, sp, _) = p.staging.unwrap();
        std::ptr::copy_nonoverlapping(data.as_ptr(), sp, size as usize);
        let img = p.upload.unwrap().0;
        barrier(c, cmd, img, vk::ImageLayout::UNDEFINED, vk::ImageLayout::TRANSFER_DST_OPTIMAL);
        let region = vk::BufferImageCopy {
            buffer_offset: 0,
            buffer_row_length: 0,
            buffer_image_height: 0,
            image_subresource: vk::ImageSubresourceLayers { aspect_mask: vk::ImageAspectFlags::COLOR, mip_level: 0, base_array_layer: 0, layer_count: 1 },
            image_offset: vk::Offset3D::default(),
            image_extent: vk::Extent3D { width: w, height: h, depth: 1 },
        };
        c.device.cmd_copy_buffer_to_image(cmd, sb, img, vk::ImageLayout::TRANSFER_DST_OPTIMAL, &[region]);
        barrier(c, cmd, img, vk::ImageLayout::TRANSFER_DST_OPTIMAL, vk::ImageLayout::TRANSFER_SRC_OPTIMAL);
        Ok(img)
    }
}

#[cfg(test)]
mod tests {
    use super::fit;

    #[test]
    fn fit_keeps_the_frame_shape() {
        assert_eq!(fit(3840, 2160, 1280, 720), (0, 0, 1280, 720));
        assert_eq!(fit(3840, 2160, 1280, 1024), (0, 152, 1280, 720));
        assert_eq!(fit(3840, 2160, 1600, 600), (266, 0, 1067, 600));
        // Under 1% off: the whole window, no 1-pixel bars.
        assert_eq!(fit(3840, 2160, 1366, 768), (0, 0, 1366, 768));
        assert_eq!(fit(0, 0, 800, 600), (0, 0, 800, 600));
    }
}

// FIFO gave the most consistent warm movement intervals on the Linux relay.
// MAILBOX remains available explicitly; both are synchronized to the compositor.
// FIFO is mandatory and is also the fallback for unsupported requests.
#[cfg(target_os = "linux")]
fn linux_present_mode(modes: &[vk::PresentModeKHR], requested: Option<&str>) -> vk::PresentModeKHR {
    let want = match requested {
        Some("fifo") => vk::PresentModeKHR::FIFO,
        Some("mailbox") => vk::PresentModeKHR::MAILBOX,
        Some("immediate") => vk::PresentModeKHR::IMMEDIATE,
        _ => vk::PresentModeKHR::FIFO,
    };
    if modes.contains(&want) { want } else { vk::PresentModeKHR::FIFO }
}

#[cfg(all(test, target_os = "linux"))]
mod linux_tests {
    use super::*;
    #[test]
    fn present_mode_is_supported_and_fifo_falls_back() {
        use vk::PresentModeKHR as M;
        assert_eq!(linux_present_mode(&[M::FIFO], None), M::FIFO);
        assert_eq!(linux_present_mode(&[M::FIFO, M::MAILBOX], None), M::FIFO);
        assert_eq!(linux_present_mode(&[M::FIFO, M::MAILBOX], Some("fifo")), M::FIFO);
        assert_eq!(linux_present_mode(&[M::FIFO, M::MAILBOX], Some("mailbox")), M::MAILBOX);
        assert_eq!(linux_present_mode(&[M::FIFO, M::MAILBOX], Some("immediate")), M::FIFO);
        assert_eq!(linux_present_mode(&[M::FIFO, M::IMMEDIATE], Some("immediate")), M::IMMEDIATE);
    }
}
