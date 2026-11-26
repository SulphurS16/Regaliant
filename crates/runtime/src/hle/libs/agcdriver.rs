//! libSceAgcDriver: submission, GPU events and resource registration.
//!
//! Submissions go to the command processor (`crate::gpu`): DCBs to the
//! graphics queue, ACBs to their compute queue. End-of-pipe interrupts
//! (RELEASE_MEM with an interrupt) trigger every event added with
//! `sceAgcDriverAddEqEvent` (filter EVFILT_GRAPHICS, ident = event id,
//! data = interrupt context id).
//!
//! Resource registration (owners/resources named by the title for GPU capture
//! tools) is recorded and used to name guest memory in GPU diagnostics.

use crate::hle::{Registry, cstr};
use crate::{reg, sysv};
use parking_lot::Mutex;
use std::collections::BTreeMap;
use std::sync::atomic::{AtomicU32, Ordering};

const MAX_NAME: u32 = 127;

#[derive(Clone, Debug)]
pub struct Resource {
    pub owner: u32,
    pub addr: u64,
    pub size: u64,
    pub name: String,
    pub kind: u32,
}

static NEXT_HANDLE: AtomicU32 = AtomicU32::new(1);
static OWNERS: Mutex<BTreeMap<u32, String>> = Mutex::new(BTreeMap::new());
static RESOURCES: Mutex<BTreeMap<u32, Resource>> = Mutex::new(BTreeMap::new());

static EQ_EVENTS: Mutex<Vec<(u64, i32, u64)>> = Mutex::new(Vec::new());
const ERR_EINVAL: i32 = 0x8002_0016u32 as i32;
const ERR_EBADF: i32 = 0x8002_0009u32 as i32;

/// Called by the command processor for RELEASE_MEM interrupts.
pub fn raise_eop(ctx: u32) {
    let list = EQ_EVENTS.lock().clone();
    for (eq, id, _) in list {
        if let Some(q) = unsafe { crate::hle::kernel::equeue::from_handle(eq) } {
            q.trigger(id as u64, crate::hle::kernel::equeue::EVFILT_GRAPHICS, Some(ctx as u64), None);
        }
    }
}

/// Name of the registered resource containing `addr`, if any (diagnostics).
pub fn resource_at(addr: u64) -> Option<Resource> {
    RESOURCES.lock().values().find(|r| addr >= r.addr && addr < r.addr + r.size.max(1)).cloned()
}

sysv! {
    fn sceAgcDriverGetResourceRegistrationMaxNameLength() -> u32 { MAX_NAME }
    fn sceAgcDriverQueryResourceRegistrationUserMemoryRequirements(size: *mut u64) -> i32 {
        // Bookkeeping lives on the host side; no title memory is needed.
        if !size.is_null() { *size = 0; }
        0
    }
    fn sceAgcDriverInitResourceRegistration(mem: u64, size: u64) -> i32 { 0 }
    fn sceAgcDriverGetDefaultOwner() -> u32 { 0 }
    fn sceAgcDriverRegisterOwner(out: *mut u32, name: *const u8) -> i32 {
        let h = NEXT_HANDLE.fetch_add(1, Ordering::Relaxed);
        OWNERS.lock().insert(h, cstr(name).to_string());
        crate::debug!("agc: owner {h} '{}'", cstr(name));
        if !out.is_null() { *out = h; }
        0
    }
    fn sceAgcDriverRegisterResource(out: *mut u32, owner: u32, addr: u64, size: u64, name: *const u8, kind: u32, user: u64) -> i32 {
        let h = NEXT_HANDLE.fetch_add(1, Ordering::Relaxed);
        let r = Resource { owner, addr, size, name: cstr(name).to_string(), kind };
        crate::debug!("agc: resource {h} '{}' type {kind:#x} {addr:#x}+{size:#x} owner {owner}", r.name);
        RESOURCES.lock().insert(h, r);
        if !out.is_null() { *out = h; }
        0
    }
    fn sceAgcDriverUnregisterResource(h: u32) -> i32 {
        RESOURCES.lock().remove(&h);
        0
    }
    fn sceAgcDriverUnregisterOwnerAndResources(owner: u32) -> i32 {
        OWNERS.lock().remove(&owner);
        RESOURCES.lock().retain(|_, r| r.owner != owner);
        0
    }
    fn sceAgcDriverGetResourceName(h: u32, buf: *mut u8, size: u32) -> i32 {
        match RESOURCES.lock().get(&h) {
            Some(r) => { crate::hle::write_cstr(buf, size as usize, &r.name); 0 }
            None => 0x8A6C_0035u32 as i32,
        }
    }
    fn sceAgcDriverGetResourceType(h: u32, kind: *mut u32) -> i32 {
        match RESOURCES.lock().get(&h) {
            Some(r) => { if !kind.is_null() { *kind = r.kind; } 0 }
            None => 0x8A6C_0035u32 as i32,
        }
    }
    /// No GPU capture tool is attached.
    fn sceAgcDriverIsCaptureInProgress() -> i32 { 0 }

    // ---- submission. Packet = {u32* addr, u32 size in dwords, u8 flags}.
    fn sceAgcDriverSubmitDcb(packet: *const u64) -> i32 {
        if packet.is_null() { return ERR_EINVAL; }
        let addr = *packet;
        let size = *(packet as *const u32).add(2);
        crate::platform::graphics::submit(crate::platform::graphics::GFX, vec![(addr, size)]);
        0
    }
    fn sceAgcDriverSubmitMultiDcbs(addrs: *const u64, sizes: *const u32, count: u32) -> i32 {
        if count == 0 { return 0; }
        if addrs.is_null() || sizes.is_null() { return ERR_EINVAL; }
        let v: Vec<(u64, u32)> = (0..count as usize).map(|i| (*addrs.add(i), *sizes.add(i))).filter(|(a, s)| *a != 0 && *s != 0).collect();
        crate::platform::graphics::submit(crate::platform::graphics::GFX, v);
        0
    }
    fn sceAgcDriverSubmitAcb(queue: u32, packet: *const u64) -> i32 {
        if packet.is_null() { return ERR_EINVAL; }
        let addr = *packet;
        let size = *(packet as *const u32).add(2);
        crate::platform::graphics::submit(queue, vec![(addr, size)]);
        0
    }
    fn sceAgcDriverSubmitMultiAcbs(queue: u32, addrs: *const u64, sizes: *const u32, count: u32) -> i32 {
        if count == 0 { return 0; }
        if addrs.is_null() || sizes.is_null() { return ERR_EINVAL; }
        let v: Vec<(u64, u32)> = (0..count as usize).map(|i| (*addrs.add(i), *sizes.add(i))).filter(|(a, s)| *a != 0 && *s != 0).collect();
        crate::platform::graphics::submit(queue, v);
        0
    }

    // ---- GPU events
    fn sceAgcDriverAddEqEvent(eq: u64, id: i32, udata: u64) -> i32 {
        let Some(q) = crate::hle::kernel::equeue::from_handle(eq) else { return ERR_EBADF };
        q.add(id as u64, crate::hle::kernel::equeue::EVFILT_GRAPHICS, crate::hle::kernel::equeue::EV_CLEAR, udata);
        let mut g = EQ_EVENTS.lock();
        g.retain(|(e, i, _)| !(*e == eq && *i == id));
        g.push((eq, id, udata));
        crate::info!("sceAgcDriverAddEqEvent(equeue '{}', id {id:#x})", q.name);
        0
    }
    fn sceAgcDriverDeleteEqEvent(eq: u64, id: i32) -> i32 {
        let Some(q) = crate::hle::kernel::equeue::from_handle(eq) else { return ERR_EBADF };
        q.remove(id as u64, crate::hle::kernel::equeue::EVFILT_GRAPHICS);
        EQ_EVENTS.lock().retain(|(e, i, _)| !(*e == eq && *i == id));
        0
    }
    /// Kevent {u64 ident, i16 filter, u16 flags, u32 fflags, u64 data, u64 udata}.
    fn sceAgcDriverGetEqEventType(ev: *const u8) -> i32 {
        if ev.is_null() { return 0; }
        let filter = *(ev.add(8) as *const i16);
        if filter == crate::hle::kernel::equeue::EVFILT_GRAPHICS { *(ev as *const u64) as i32 } else { *(ev.add(16) as *const u64) as i32 }
    }
    fn sceAgcDriverGetEqContextId(ev: *const u8) -> u32 {
        if ev.is_null() { return 0; }
        let filter = *(ev.add(8) as *const i16);
        if filter == crate::hle::kernel::equeue::EVFILT_GRAPHICS { *(ev.add(16) as *const u64) as u32 } else { *(ev as *const u64) as u32 }
    }

    // ---- workload streams (GPU profiling of the title's workloads): accepted.
    fn sceAgcDriverRegisterWorkloadStream(id: u32, stream: u64) -> i32 { 0 }
    fn sceAgcDriverUnregisterWorkloadStream(id: u32) -> i32 { 0 }
}

pub fn register(r: &mut Registry) {
    reg!(r, "libSceAgcDriver", [
        sceAgcDriverGetResourceRegistrationMaxNameLength, sceAgcDriverQueryResourceRegistrationUserMemoryRequirements,
        sceAgcDriverInitResourceRegistration, sceAgcDriverGetDefaultOwner, sceAgcDriverRegisterOwner,
        sceAgcDriverRegisterResource, sceAgcDriverUnregisterResource, sceAgcDriverUnregisterOwnerAndResources,
        sceAgcDriverGetResourceName, sceAgcDriverGetResourceType, sceAgcDriverIsCaptureInProgress,
        sceAgcDriverSubmitDcb, sceAgcDriverSubmitMultiDcbs, sceAgcDriverSubmitAcb, sceAgcDriverSubmitMultiAcbs,
        sceAgcDriverAddEqEvent, sceAgcDriverDeleteEqEvent, sceAgcDriverGetEqEventType, sceAgcDriverGetEqContextId,
        sceAgcDriverRegisterWorkloadStream, sceAgcDriverUnregisterWorkloadStream,
    ]);
}
