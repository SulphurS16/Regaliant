//! The guest page table of GPU-resolved chains (RENDERER.md "GPU-resolved
//! item passes"; `rdna::translate::Spec::gpu_loads`, `gpu_res`): per 4 KiB
//! guest page of [PT_BASE, PT_END) a u64, the device address where the
//! page's contents are for the GPU, 0 where the table maps nothing (shaders
//! read 0 there). The entries are the import windows of guest memory
//! (`exec::import_span`): mapped when a window is imported, cleared when it
//! is dropped after a remap.
//!
//! Updates are queued (`map`, `unmap`) and copied into the table by a command
//! buffer submitted on the main queue right before the next batch
//! (`submit_pending`, from `exec::submit_batch`, which runs in sequence
//! order): a batch queued after an update sees it, as does every later one.
//! Batches dropped from the table's windows still run against the old
//! windows' memory, which stays until no batch can use it (`exec`'s
//! graveyard).

use super::vk::{ctx, Ctx};
use ash::vk;
use parking_lot::Mutex;
use rdna::translate::{PT_BASE, PT_END, PT_FEEDBACK, PT_PAGE_SHIFT};
use std::sync::atomic::{AtomicU64, Ordering};

const PAGE: u64 = 1 << PT_PAGE_SHIFT;
const PAGES: u64 = (PT_END - PT_BASE) >> PT_PAGE_SHIFT;
/// The table's size (8 bytes per page), then the feedback words' address.
pub const BYTES: u64 = PAGES * 8 + 256;

struct Table {
    buf: vk::Buffer,
    va: u64,
    /// The feedback words (host visible: `PT_FEEDBACK`).
    feedback: *mut u32,
}

unsafe impl Send for Table {}
unsafe impl Sync for Table {}

/// The table, once made (None: device addresses unavailable or it failed).
static TABLE: std::sync::OnceLock<Option<Table>> = std::sync::OnceLock::new();

/// Updates not copied into the table yet: (first page, pages, device
/// address of the first page or 0 to clear), in order.
static PENDING: Mutex<Vec<(u64, u64, u64)>> = Mutex::new(Vec::new());

/// Update submissions still running: (command buffer, fence, staging
/// buffer, its memory), freed once their fence signalled.
struct Flight {
    cmd: vk::CommandBuffer,
    fence: vk::Fence,
    staging: vk::Buffer,
    mem: vk::DeviceMemory,
}

struct Updater {
    pool: vk::CommandPool,
    flights: Vec<Flight>,
}

static UPDATER: Mutex<Option<Updater>> = Mutex::new(None);

/// Pages mapped and cleared, update submissions and the bytes they copied.
static STATS: [AtomicU64; 4] = [const { AtomicU64::new(0) }; 4];

fn table() -> Option<&'static Table> {
    TABLE
        .get_or_init(|| {
            let c = ctx();
            if !c.device_addresses {
                crate::warn!("gpu: guest page table: no device addresses");
                return None;
            }
            let usage = vk::BufferUsageFlags::STORAGE_BUFFER | vk::BufferUsageFlags::TRANSFER_DST;
            let (buf, _mem, va) = match c.device_buffer_addressed(BYTES, usage) {
                Ok(x) => x,
                Err(e) => {
                    crate::warn!("gpu: guest page table ({} MiB): {e}", BYTES >> 20);
                    return None;
                }
            };
            let (_fbuf, _fmem, fptr) = match c.host_buffer(64, vk::BufferUsageFlags::STORAGE_BUFFER) {
                Ok(x) => x,
                Err(e) => {
                    crate::warn!("gpu: guest page table feedback: {e}");
                    return None;
                }
            };
            unsafe { std::ptr::write_bytes(fptr, 0, 64) };
            let fva = c.buffer_address(_fbuf);
            // Cleared once (no page mapped).
            unsafe {
                let pool = c.device.create_command_pool(&vk::CommandPoolCreateInfo::default().queue_family_index(c.queue_family), None).ok()?;
                let cmd = c.device.allocate_command_buffers(&vk::CommandBufferAllocateInfo::default().command_pool(pool).command_buffer_count(1)).ok()?[0];
                let fence = c.device.create_fence(&vk::FenceCreateInfo::default(), None).ok()?;
                c.device.begin_command_buffer(cmd, &vk::CommandBufferBeginInfo::default().flags(vk::CommandBufferUsageFlags::ONE_TIME_SUBMIT)).ok();
                c.device.cmd_fill_buffer(cmd, buf, 0, vk::WHOLE_SIZE, 0);
                let mb = vk::MemoryBarrier::default().src_access_mask(vk::AccessFlags::TRANSFER_WRITE).dst_access_mask(vk::AccessFlags::TRANSFER_WRITE);
                c.device.cmd_pipeline_barrier(cmd, vk::PipelineStageFlags::TRANSFER, vk::PipelineStageFlags::TRANSFER, vk::DependencyFlags::empty(), &[mb], &[], &[]);
                c.device.cmd_update_buffer(cmd, buf, PT_FEEDBACK, &fva.to_le_bytes());
                c.device.end_command_buffer(cmd).ok();
                if c.submit(cmd, &[], &[], fence).is_ok() {
                    c.device.wait_for_fences(&[fence], true, u64::MAX).ok();
                }
                c.device.destroy_fence(fence, None);
                c.device.destroy_command_pool(pool, None);
            }
            crate::info!("gpu: guest page table made ({} MiB for {:#x}..{:#x}, {} KiB pages)", BYTES >> 20, PT_BASE, PT_END, PAGE >> 10);
            Some(Table { buf, va, feedback: fptr as *mut u32 })
        })
        .as_ref()
        .inspect(|_| {
            // The imports made before the table: mapped once.
            if !SYNCED.swap(true, Ordering::AcqRel) {
                super::exec::map_imports_into_table();
            }
        })
}

/// The imports made before the table are mapped (`table`).
static SYNCED: std::sync::atomic::AtomicBool = std::sync::atomic::AtomicBool::new(false);

/// The table's device address (0: none), made on first use.
pub fn address() -> u64 {
    table().map_or(0, |t| t.va)
}

/// Whether the table exists (no side effect).
pub fn made() -> bool {
    TABLE.get().is_some_and(|t| t.is_some())
}

/// The table's buffer, made on first use (front end 2's self-test writes an
/// entry of its own and clears it again).
pub(super) fn buffer() -> Option<vk::Buffer> {
    table().map(|t| t.buf)
}

/// The pages of guest [guest, guest+len) clipped to the table: (first page,
/// pages, the clipped range's start).
fn pages(guest: u64, len: u64) -> Option<(u64, u64, u64)> {
    let lo = guest.max(PT_BASE) & !(PAGE - 1);
    let hi = guest.saturating_add(len).min(PT_END).next_multiple_of(PAGE);
    (hi > lo).then(|| ((lo - PT_BASE) >> PT_PAGE_SHIFT, (hi - lo) >> PT_PAGE_SHIFT, lo))
}

/// Guest [guest, guest+len) is at device address `dev` (its first byte)
/// from the next batch on; the range's pages are whole pages of it.
pub fn map(guest: u64, len: u64, dev: u64) {
    if !made() || dev == 0 {
        return;
    }
    let Some((first, n, lo)) = pages(guest, len) else { return };
    // The device address of page `lo` (import windows start on a page).
    let d = dev.wrapping_add(lo.wrapping_sub(guest));
    PENDING.lock().push((first, n, d));
    STATS[0].fetch_add(n, Ordering::Relaxed);
}

/// Guest [guest, guest+len) is no longer mapped from the next batch on.
pub fn unmap(guest: u64, len: u64) {
    if !made() {
        return;
    }
    let Some((first, n, _)) = pages(guest, len) else { return };
    PENDING.lock().push((first, n, 0));
    STATS[1].fetch_add(n, Ordering::Relaxed);
}

/// Clears guest [guest, guest+len) and maps `maps` (guest, bytes, device
/// address) over it as one update: a batch submitted between a clear and
/// its maps would read nothing there.
pub fn unmap_then_map(guest: u64, len: u64, maps: &[(u64, u64, u64)]) {
    if !made() {
        return;
    }
    let mut p = PENDING.lock();
    if let Some((first, n, _)) = pages(guest, len) {
        p.push((first, n, 0));
        STATS[1].fetch_add(n, Ordering::Relaxed);
    }
    for &(g, l, dev) in maps {
        if dev == 0 {
            continue;
        }
        if let Some((first, n, lo)) = pages(g, l) {
            p.push((first, n, dev.wrapping_add(lo.wrapping_sub(g))));
            STATS[0].fetch_add(n, Ordering::Relaxed);
        }
    }
}

/// Copies the queued updates into the table on the main queue, before the
/// batch about to be submitted (called in sequence order, `submit_batch`),
/// after batch `cwait` of the compute family's queue (ordered async
/// compute: its shaders read the old entries; 0: none).
pub(super) fn submit_pending(c: &Ctx, cwait: u64) -> Result<(), String> {
    let _trace = super::timeline::section("submit pagetable");
    if !made() {
        return Ok(());
    }
    let runs = std::mem::take(&mut *PENDING.lock());
    if runs.is_empty() {
        return Ok(());
    }
    let t = table().ok_or("no page table")?;
    let mut u = UPDATER.lock();
    let up = match u.as_mut() {
        Some(x) => x,
        None => {
            let pool = unsafe { c.device.create_command_pool(&vk::CommandPoolCreateInfo::default().queue_family_index(c.queue_family).flags(vk::CommandPoolCreateFlags::RESET_COMMAND_BUFFER), None) }.map_err(|e| format!("page table pool: {e}"))?;
            u.insert(Updater { pool, flights: Vec::new() })
        }
    };
    // Earlier updates that completed: freed.
    up.flights.retain(|f| {
        let done = unsafe { c.device.get_fence_status(f.fence) }.unwrap_or(false);
        if done {
            unsafe {
                c.device.destroy_fence(f.fence, None);
                c.device.free_command_buffers(up.pool, &[f.cmd]);
                c.device.destroy_buffer(f.staging, None);
                c.device.free_memory(f.mem, None);
            }
        }
        !done
    });
    let total: u64 = runs.iter().map(|r| r.1).sum();
    let bytes = (total * 8).max(8);
    let (staging, mem, ptr) = c.host_buffer(bytes, vk::BufferUsageFlags::TRANSFER_SRC)?;
    let mut regions = Vec::with_capacity(runs.len());
    let mut at = 0u64;
    unsafe {
        let words = ptr as *mut u64;
        for &(first, n, dev) in &runs {
            for k in 0..n {
                words.add((at + k) as usize).write(if dev == 0 { 0 } else { dev + k * PAGE });
            }
            regions.push(vk::BufferCopy { src_offset: at * 8, dst_offset: first * 8, size: n * 8 });
            at += n;
        }
    }
    let cmd = unsafe { c.device.allocate_command_buffers(&vk::CommandBufferAllocateInfo::default().command_pool(up.pool).command_buffer_count(1)) }.map_err(|e| format!("page table command buffer: {e}"))?[0];
    let fence = unsafe { c.device.create_fence(&vk::FenceCreateInfo::default(), None) }.map_err(|e| format!("page table fence: {e}"))?;
    unsafe {
        c.device.begin_command_buffer(cmd, &vk::CommandBufferBeginInfo::default().flags(vk::CommandBufferUsageFlags::ONE_TIME_SUBMIT)).ok();
        // Later runs of the same pages win: copies in order, each behind
        // the last when they overlap (rare: a window dropped and imported
        // again before a batch).
        let mut done: Vec<(u64, u64)> = Vec::new();
        for r in &regions {
            let (a, b) = (r.dst_offset, r.dst_offset + r.size);
            if done.iter().any(|&(x, y)| x < b && a < y) {
                let mb = vk::MemoryBarrier::default().src_access_mask(vk::AccessFlags::TRANSFER_WRITE).dst_access_mask(vk::AccessFlags::TRANSFER_WRITE);
                c.device.cmd_pipeline_barrier(cmd, vk::PipelineStageFlags::TRANSFER, vk::PipelineStageFlags::TRANSFER, vk::DependencyFlags::empty(), &[mb], &[], &[]);
                done.clear();
            }
            c.device.cmd_copy_buffer(cmd, staging, t.buf, std::slice::from_ref(r));
            done.push((a, b));
        }
        let mb = vk::MemoryBarrier::default().src_access_mask(vk::AccessFlags::TRANSFER_WRITE).dst_access_mask(vk::AccessFlags::MEMORY_READ);
        c.device.cmd_pipeline_barrier(cmd, vk::PipelineStageFlags::TRANSFER, vk::PipelineStageFlags::ALL_COMMANDS, vk::DependencyFlags::empty(), &[mb], &[], &[]);
        c.device.end_command_buffer(cmd).ok();
    }
    match &c.compute {
        Some(q) if cwait > c.batch_value_c() => c.submit_timeline(false, &[cmd], &[(q.batches, cwait, vk::PipelineStageFlags2::ALL_COMMANDS)], &[], fence)?,
        _ => c.submit(cmd, &[], &[], fence)?,
    }
    up.flights.push(Flight { cmd, fence, staging, mem });
    STATS[2].fetch_add(1, Ordering::Relaxed);
    STATS[3].fetch_add(bytes, Ordering::Relaxed);
    Ok(())
}

/// Reads through unmapped pages the shaders counted, and the last one's
/// guest address.
pub fn misses() -> (u32, u64) {
    let Some(t) = TABLE.get().and_then(|t| t.as_ref()) else { return (0, 0) };
    // SAFETY: host-visible words the shaders update atomically.
    unsafe {
        let n = std::ptr::read_volatile(t.feedback);
        let lo = std::ptr::read_volatile(t.feedback.add(2)) as u64;
        let hi = std::ptr::read_volatile(t.feedback.add(3)) as u64;
        (n, lo | hi << 32)
    }
}

/// Diagnosis: the table's entry for guest page `guest` as the GPU sees it
/// (a synchronous copy into host memory).
pub fn read_entry(guest: u64) -> Option<u64> {
    let t = TABLE.get()?.as_ref()?;
    if !(PT_BASE..PT_END).contains(&guest) {
        return None;
    }
    let c = ctx();
    let page = (guest - PT_BASE) >> PT_PAGE_SHIFT;
    unsafe {
        let (hb, hm, hp) = c.host_buffer(8, vk::BufferUsageFlags::TRANSFER_DST).ok()?;
        let pool = c.device.create_command_pool(&vk::CommandPoolCreateInfo::default().queue_family_index(c.queue_family), None).ok()?;
        let cmd = c.device.allocate_command_buffers(&vk::CommandBufferAllocateInfo::default().command_pool(pool).command_buffer_count(1)).ok()?[0];
        let fence = c.device.create_fence(&vk::FenceCreateInfo::default(), None).ok()?;
        c.device.begin_command_buffer(cmd, &vk::CommandBufferBeginInfo::default().flags(vk::CommandBufferUsageFlags::ONE_TIME_SUBMIT)).ok();
        let mb = vk::MemoryBarrier::default().src_access_mask(vk::AccessFlags::MEMORY_WRITE).dst_access_mask(vk::AccessFlags::TRANSFER_READ);
        c.device.cmd_pipeline_barrier(cmd, vk::PipelineStageFlags::ALL_COMMANDS, vk::PipelineStageFlags::TRANSFER, vk::DependencyFlags::empty(), &[mb], &[], &[]);
        c.device.cmd_copy_buffer(cmd, t.buf, hb, &[vk::BufferCopy { src_offset: page * 8, dst_offset: 0, size: 8 }]);
        c.device.end_command_buffer(cmd).ok();
        let ok = c.submit(cmd, &[], &[], fence).is_ok() && c.device.wait_for_fences(&[fence], true, 2_000_000_000).is_ok();
        let v = std::ptr::read_volatile(hp as *const u64);
        c.device.destroy_fence(fence, None);
        c.device.destroy_command_pool(pool, None);
        c.device.destroy_buffer(hb, None);
        c.device.free_memory(hm, None);
        ok.then_some(v)
    }
}

/// Status line ("" before the table is made).
pub fn status() -> String {
    if !made() {
        return String::new();
    }
    let (n, last) = misses();
    let why = if n > 0 {
        let e = read_entry(last).map_or("unread".to_string(), |v| format!("{v:#x}"));
        let w = super::exec::window_of(last).map_or("none".to_string(), |(a, b, va)| format!("{a:#x}..{b:#x} at {va:#x}"));
        format!(" (its entry now {e}, import window {w})")
    } else {
        String::new()
    };
    format!(
        "    guest page table: {} pages mapped, {} cleared, {} update submissions ({} KiB), windows imported again right after a drop {} ({} ms), left to their next use for being cold {}; imports dropped for being cold while the host heap ran short {} ({} MiB, {} passes); shader reads through unmapped pages {n} (last {last:#x}{why})\n",
        STATS[0].load(Ordering::Relaxed),
        STATS[1].load(Ordering::Relaxed),
        STATS[2].load(Ordering::Relaxed),
        STATS[3].load(Ordering::Relaxed) >> 10,
        super::exec::REIMPORTS.load(Ordering::Relaxed),
        super::exec::REIMPORT_NS.load(Ordering::Relaxed) / 1_000_000,
        super::exec::REIMPORTS_SKIPPED.load(Ordering::Relaxed),
        super::exec::COLD_EVICTED[0].load(Ordering::Relaxed),
        super::exec::COLD_EVICTED[1].load(Ordering::Relaxed) >> 20,
        super::exec::COLD_EVICTED[2].load(Ordering::Relaxed)
    )
}
