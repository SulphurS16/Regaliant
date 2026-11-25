//! PS5 system services on the startup path: Json2 (C++ API), HTTP/2, NP Web
//! API 2, game-update check, game intents, system gestures, Share, NP state and
//! the Universal Data System (telemetry).
//!
//! This runtime behaves as an offline console: contexts are created so the
//! title's initialization proceeds, but no request ever reaches a network.
//! Each such answer is logged once as a fixed result.

use crate::hle::{Registry, stub_log};
use crate::{reg, sysv};
use std::sync::atomic::{AtomicI32, Ordering};

static NEXT: AtomicI32 = AtomicI32::new(1);
const WEBAPI2_ERROR_INVALID_ARGUMENT: i32 = 0x8055_3402u32 as i32;
/// NpWebApi2 "not signed in" (error codes per public sources, docs/PROVENANCE.md).
const WEBAPI2_ERROR_NOT_SIGNED_IN: i32 = 0x8055_3407u32 as i32;

const UDS_ERROR_INVALID_ARGUMENT: i32 = 0x8055_3102u32 as i32;
const GAME_UPDATE_ERROR_INVALID_ARG: i32 = 0x8041_2803u32 as i32;
const GAME_UPDATE_ERROR_INVALID_SIZE: i32 = 0x8041_2804u32 as i32;
const HTTP2_ERROR_BEFORE_SEND: i32 = 0x817B_1065u32 as i32;
const HTTP2_ERROR_TIMEOUT: i32 = 0x817B_1068u32 as i32;
const HTTP2_ERROR_NULL_POINTER: i32 = 0x817B_1225u32 as i32;
const NP_ERROR_SIGNED_OUT: i32 = 0x8055_0006u32 as i32;
const SIGNIN_INITIALIZED: i32 = 1;
const SIGNIN_RUNNING: i32 = 2;
const SIGNIN_FINISHED: i32 = 3;
/// (status, polled while running)
static SIGNIN: parking_lot::Mutex<(i32, bool)> = parking_lot::Mutex::new((0, false));

/// Opaque UDS handles: 64-byte host blocks.
fn uds_alloc() -> u64 {
    let p = Box::into_raw(Box::new([0u64; 8])) as u64;
    UDS_LIVE.lock().insert(p);
    p
}
fn uds_free(p: u64) {
    if p != 0 && UDS_LIVE.lock().remove(&p) {
        drop(unsafe { Box::from_raw(p as *mut [u64; 8]) });
    }
    if p != 0 {
        let mut g = UDS_DATA.lock();
        g.0.remove(&p);
        g.1.remove(&p);
    }
}
/// What the port keeps of the events (`achievements`): event handle -> (name, its property
/// object), property object -> its top-level values.
type UdsData = (std::collections::BTreeMap<u64, (String, u64)>, std::collections::BTreeMap<u64, Vec<(String, crate::achievements::Value)>>);
static UDS_DATA: parking_lot::Mutex<UdsData> = parking_lot::Mutex::new((std::collections::BTreeMap::new(), std::collections::BTreeMap::new()));
fn uds_set(obj: u64, key: *const u8, v: crate::achievements::Value) -> i32 {
    if obj != 0 && !key.is_null() {
        let k = cstr_or(key);
        let mut g = UDS_DATA.lock();
        let props = g.1.entry(obj).or_default();
        props.retain(|(x, _)| *x != k);
        props.push((k, v));
    }
    uds_check(obj)
}
fn uds_check(p: u64) -> i32 {
    if p == 0 { UDS_ERROR_INVALID_ARGUMENT } else { 0 }
}
/// Handles allocated here (only these are freed; titles may pass their own).
static UDS_LIVE: parking_lot::Mutex<std::collections::BTreeSet<u64>> = parking_lot::Mutex::new(std::collections::BTreeSet::new());

fn cstr_or(p: *const u8) -> String {
    if p.is_null() { String::new() } else { unsafe { crate::hle::cstr(p) }.to_string() }
}
/// One entry of the add-on entitlement list (`sceNpEntitlementAccessGetAddcontEntitlementInfoList`):
/// 28 bytes, the entitlement label a NUL-terminated string at its start. (From the title's own
/// caller, 2026-11-24: it asks for the count with no list, allocates count x 28 bytes, asks for the
/// list and compares each entry's label with its own, `DEMONSSOULSGIFT1` and `...2`, the
/// pre-order gifts; it reads nothing else of an entry.)
const ADDCONT_ENTRY: usize = 28;

/// The add-on labels the title asks for (its caller's own two, 2026-11-24: the pre-order and the
/// Digital Deluxe content), named in the scan's log line.
const TITLE_ADDONS: [&str; 2] = ["DEMONSSOULSGIFT1", "DEMONSSOULSGIFT2"];
/// The add-on scan reads at most this many files' first `ADDON_READ_BYTES` (`labels_in`).
const ADDON_READS: usize = 32;
const ADDON_READ_BYTES: u64 = 1 << 20;

/// Where the player's add-on packages are looked for: a folder named "Add-ons" in the settings
/// folder (%APPDATA%\Regaliant, under AppData\Roaming), in the data folder (%LOCALAPPDATA%\
/// Regaliant, under AppData\Local) and beside the program. (A player put the packages under Local
/// as well as Roaming on 2026-11-24, unsure which the notes meant; a checkout's or a portable
/// install's one folder holds the first two.)
fn addon_dirs() -> Vec<std::path::PathBuf> {
    let p = crate::paths();
    let mut v: Vec<std::path::PathBuf> = Vec::new();
    for d in [p.user.join("Add-ons"), p.cache.join("Add-ons"), p.exe_dir.join("Add-ons")] {
        if !v.contains(&d) {
            v.push(d);
        }
    }
    v
}

/// One folder's add-on scan (`scan_addons`).
#[derive(Default)]
struct AddonScan {
    /// Each label found, with where (a name, or inside a file).
    labels: Vec<(String, String)>,
    /// Files and folders seen.
    entries: usize,
    /// The first files that gave no label.
    others: Vec<String>,
}

/// The player's own add-on packages: the entitlement labels of the .pkg files, or of the folders
/// unpacked from them, under the folders `addon_dirs` names, at most three levels down: from the
/// content ID in a file's or folder's name (`addon_label`), else inside a file's first bytes
/// (`labels_in`: a package's header, an unpacked folder's param.json), whatever the names. Scanned
/// once, with one log line saying what was found where, or what each folder held. Without them
/// the title hears of no add-ons, as on a console that has none installed.
fn addon_labels() -> &'static Vec<String> {
    static L: std::sync::OnceLock<Vec<String>> = std::sync::OnceLock::new();
    L.get_or_init(|| {
        let mut budget = ADDON_READS;
        let mut found: Vec<String> = Vec::new();
        let mut places = Vec::new();
        for dir in addon_dirs() {
            if !dir.is_dir() {
                places.push(format!("{} (no such folder)", dir.display()));
                continue;
            }
            let mut s = AddonScan::default();
            scan_addons(&dir, &dir, 3, &mut s, &mut budget);
            places.push(if s.labels.is_empty() {
                let more = if s.others.len() < s.entries { ", ..." } else { "" };
                let some = if s.others.is_empty() { String::new() } else { format!(": {}{more}", s.others.join(", ")) };
                format!("{} ({} files and folders, no add-on among them{some})", dir.display(), s.entries)
            } else {
                format!("{} ({})", dir.display(), s.labels.iter().map(|(l, w)| format!("{l} from {w}")).collect::<Vec<_>>().join(", "))
            });
            found.extend(s.labels.into_iter().map(|(l, _)| l));
        }
        found.sort();
        found.dedup();
        let asked = TITLE_ADDONS.join(" and ");
        let unknown = !found.is_empty() && !found.iter().any(|l| TITLE_ADDONS.contains(&l.as_str()));
        crate::info!(
            "add-ons: {} (the game asks for {asked}{}); looked in {}",
            if found.is_empty() { "none found".to_string() } else { format!("owned: {}", found.join(", ")) },
            if unknown { ", none of which was found" } else { "" },
            places.join("; ")
        );
        found
    })
}

fn scan_addons(top: &std::path::Path, dir: &std::path::Path, depth: u32, s: &mut AddonScan, budget: &mut usize) {
    if !crate::platform::files::import_file_type(dir).is_some_and(|t| t.is_dir()) { return; }
    let Ok(rd) = std::fs::read_dir(dir) else { return };
    let mut entries: Vec<std::fs::DirEntry> = rd.flatten().collect();
    entries.sort_by_key(|e| e.file_name());
    for e in entries {
        s.entries += 1;
        let path = e.path();
        let rel = path.strip_prefix(top).unwrap_or(&path).display().to_string();
        // (Not into junctions or links: `file_type` does not follow them.)
        let kind = crate::platform::files::import_file_type(&path);
        if !kind.is_some_and(|t| t.is_file() || t.is_dir()) { continue; }
        if let Some(l) = addon_label(&e.file_name().to_string_lossy()) {
            s.labels.push((l, format!("the name {rel}")));
        } else if kind.is_some_and(|t| t.is_file()) {
            let inside = if *budget > 0 {
                *budget -= 1;
                labels_in(&path)
            } else {
                Vec::new()
            };
            if inside.is_empty() && s.others.len() < 8 {
                s.others.push(rel.clone());
            }
            s.labels.extend(inside.into_iter().map(|l| (l, format!("inside {rel}"))));
        }
        if depth > 0 && kind.is_some_and(|t| t.is_dir()) {
            scan_addons(top, &path, depth - 1, s, budget);
        }
    }
}

/// The add-on labels in a file's first `ADDON_READ_BYTES`.
fn labels_in(path: &std::path::Path) -> Vec<String> {
    use std::io::Read;
    let mut b = Vec::new();
    let Ok(f) = std::fs::File::open(path) else { return Vec::new() };
    if f.take(ADDON_READ_BYTES).read_to_end(&mut b).is_err() {
        return Vec::new();
    }
    labels_in_bytes(&b)
}

/// Each content ID of the game's in `b` (`addon_label`), as a package's header or an unpacked
/// folder's param.json holds it.
fn labels_in_bytes(b: &[u8]) -> Vec<String> {
    const PAT: &[u8] = b"PPSA0134";
    let mut out = Vec::new();
    let mut i = 0;
    while let Some(k) = b[i..].windows(PAT.len()).position(|w| w == PAT) {
        let at = i + k;
        // ("PPSA0134" + digit + "_00-" + the 16-character label.)
        let end = (at + PAT.len() + 1 + 4 + 16).min(b.len());
        if let Some(l) = std::str::from_utf8(&b[at..end]).ok().and_then(addon_label)
            && !out.contains(&l)
        {
            out.push(l);
        }
        i = at + 1;
    }
    out
}

/// The entitlement label in a package's content ID as a file or folder name has it: the 16
/// capital letters and digits after `PPSA0134<digit>_00-` (`EP9000-PPSA01341_00-DEMONSSOULSGIFT1.pkg`
/// gives `DEMONSSOULSGIFT1`); or a name that is the label alone (`DEMONSSOULSGIFT1.pkg`, a
/// package renamed).
fn addon_label(name: &str) -> Option<String> {
    let label_at = |s: &str| -> Option<String> {
        let label: String = s.chars().take(16).collect();
        (label.len() == 16 && label.chars().all(|c| c.is_ascii_uppercase() || c.is_ascii_digit())).then_some(label)
    };
    if let Some(i) = name.find("PPSA0134") {
        let rest = &name[i + 8..];
        return label_at(rest.strip_prefix(|c: char| c.is_ascii_digit())?.strip_prefix("_00-")?);
    }
    let stem = name.split('.').next()?;
    (stem.len() == 16).then(|| label_at(stem)).flatten()
}

fn next_id() -> i32 {
    NEXT.fetch_add(1, Ordering::Relaxed)
}

sysv! {
    // sce::Json::MemAllocator::MemAllocator() — base-object constructor; the
    // title's derived allocator installs its own vtable afterwards.
    fn _ZN3sce4Json12MemAllocatorC2Ev(this: u64) -> u64 { this }
    fn _ZN3sce4Json12MemAllocatorD2Ev(this: u64) {}
    // sce::Json::Initializer
    fn _ZN3sce4Json11InitializerC1Ev(this: u64) -> u64 { this }
    fn _ZN3sce4Json11InitializerD1Ev(this: u64) {}
    fn _ZN3sce4Json11Initializer10initializeEPKNS0_13InitParameterE(this: u64, param: u64) -> i32 {
        stub_log("sce::Json::Initializer::initialize (no JSON work is requested offline)", 0);
        0
    }

    fn sceHttp2Init(net_id: i32, ssl_id: i32, pool_size: u64, max_requests: i32) -> i32 {
        let id = next_id();
        crate::info!("sceHttp2Init(pool {pool_size:#x}, max {max_requests}) -> context {id} (offline: no request will reach a network)");
        id
    }
    fn sceHttp2Term(ctx: i32) -> i32 { 0 }
    fn sceNpWebApi2Initialize(http_ctx: i32, pool_size: u64) -> i32 {
        let id = next_id();
        crate::info!("sceNpWebApi2Initialize(http {http_ctx}, pool {pool_size:#x}) -> context {id} (offline)");
        id
    }
    fn sceNpWebApi2Terminate(ctx: i32) -> i32 { 0 }
    /// (library context, user id) -> user context. Contexts exist offline;
    /// requests made through them fail.
    fn sceNpWebApi2CreateUserContext(lib_ctx: i32, user: i32) -> i32 {
        let id = next_id();
        crate::info!("sceNpWebApi2CreateUserContext(lib {lib_ctx}, user {user}) -> {id} (offline: requests will fail)");
        id
    }
    fn sceNpWebApi2DeleteUserContext(user_ctx: i32) -> i32 { 0 }
    /// (user context, api group, path, method, content parameter, *request id)
    fn sceNpWebApi2CreateRequest(user_ctx: i32, group: *const u8, path: *const u8, method: *const u8, content: u64, out: *mut i64) -> i32 {
        if out.is_null() { return WEBAPI2_ERROR_INVALID_ARGUMENT; }
        *out = next_id() as i64;
        crate::debug!("sceNpWebApi2CreateRequest {} {} {}", cstr_or(method), cstr_or(group), cstr_or(path));
        0
    }
    fn sceNpWebApi2AddHttpRequestHeader(req: i64, name: *const u8, value: *const u8) -> i32 { 0 }
    /// Offline console: the user is not signed in to PSN.
    fn sceNpWebApi2SendRequest(req: i64, data: u64, size: u64, info: u64) -> i32 {
        stub_log("sceNpWebApi2SendRequest (offline: not signed in)", WEBAPI2_ERROR_NOT_SIGNED_IN as i64);
        WEBAPI2_ERROR_NOT_SIGNED_IN
    }
    fn sceNpWebApi2ReadData(req: i64, data: u64, size: u64) -> i32 { WEBAPI2_ERROR_NOT_SIGNED_IN }
    fn sceNpWebApi2DeleteRequest(req: i64) -> i32 { 0 }
    // Push events: handles, filters and callbacks exist; no event is delivered.
    fn sceNpWebApi2PushEventCreateHandle(lib_ctx: i32) -> i32 { next_id() }
    fn sceNpWebApi2PushEventDeleteHandle(lib_ctx: i32, handle: i32) -> i32 { 0 }
    fn sceNpWebApi2PushEventCreateFilter(lib_ctx: i32, handle: i32, service: u64, label: u32, params: u64, count: u64) -> i32 { next_id() }
    fn sceNpWebApi2PushEventRegisterCallback(user_ctx: i32, filter: i32, cb: u64, arg: u64) -> i32 {
        stub_log("sceNpWebApi2PushEventRegisterCallback (offline: no push events)", 1);
        next_id()
    }
    fn sceNpWebApi2PushEventDeletePushContext(user_ctx: i32, push_ctx: u64) -> i32 { 0 }

    fn sceGameUpdateInitialize(param: u64) -> i32 {
        stub_log("sceGameUpdateInitialize (offline: update checks report no update)", 0);
        0
    }
    fn sceNpGameIntentInitialize(param: u64) -> i32 {
        stub_log("sceNpGameIntentInitialize (no game intents are delivered)", 0);
        0
    }

    // ---- System gestures (DualSense touch pad): no touch input is mapped yet,
    // so recognizers never report events.
    fn sceSystemGestureInitializePrimitiveTouchRecognizer(param: u64) -> i32 { 0 }
    fn sceSystemGestureFinalizePrimitiveTouchRecognizer() -> i32 { 0 }
    fn sceSystemGestureOpen(input_type: i32, param: u64) -> i32 {
        stub_log("sceSystemGestureOpen (touch pad input not mapped: no gesture events)", 1);
        1
    }
    fn sceSystemGestureClose(h: i32) -> i32 { 0 }
    fn sceSystemGestureCreateTouchRecognizer(h: i32, rec: *mut u8, kind: u32, rect: u64, param: u64) -> i32 {
        if !rec.is_null() { *(rec as *mut u32) = kind; }
        0
    }
    fn sceSystemGestureUpdatePrimitiveTouchRecognizer(data: u64) -> i32 { 0 }
    fn sceSystemGestureUpdateTouchRecognizer(h: i32, rec: u64) -> i32 { 0 }
    fn sceSystemGestureGetTouchEventsCount(h: i32, rec: u64) -> i32 { 0 }
    fn sceSystemGestureGetTouchEventByIndex(h: i32, rec: u64, index: u32, ev: u64) -> i32 { 0x8080_0001u32 as i32 }

    // ---- Share (screenshots/video sharing): accepted, nothing is captured.
    fn sceShareInitialize() -> i32 { stub_log("sceShareInitialize (sharing features inactive)", 0); 0 }
    fn sceShareSetContentParam(param: u64) -> i32 { 0 }

    // ---- NP (PSN): offline console.
    fn sceNpRegisterStateCallbackA(cb: u64, arg: u64) -> i32 { stub_log("sceNpRegisterStateCallbackA (offline: state stays signed out)", 1); 1 }
    fn sceNpUnregisterStateCallbackA(id: i32) -> i32 { 0 }
    fn sceNpSessionSignalingInitialize(param: u64) -> i32 { stub_log("sceNpSessionSignalingInitialize (offline)", 0); 0 }
    /// libSceUserServicePlatformPrivacyWs1 export D-CzAxQL0XI (unnamed): reported OK.
    fn ds_user_service_platform_privacy(a: u64, b: u64, c: u64) -> i32 {
        stub_log("libSceUserServicePlatformPrivacyWs1 D-CzAxQL0XI (unnamed; returns 0)", 0);
        0
    }

    // ---- NP entitlement access (add-on content): the player's own add-on packages under
    // "Add-ons" are reported as owned (`addon_labels`), none without them. Error code and the
    // initialization's structure size per public sources; the list's entries per the title's
    // own use (`ADDCONT_ENTRY`).
    fn sceNpEntitlementAccessInitialize(init: *const u8, boot: *mut u8) -> i32 {
        if init.is_null() || boot.is_null() { return 0x817D_0002u32 as i32; }
        std::ptr::write_bytes(boot, 0, 32);
        stub_log("sceNpEntitlementAccessInitialize (add-ons: the player's own packages under Add-ons)", 0);
        0
    }
    fn sceNpEntitlementAccessGetAddcontEntitlementInfoList(label: u32, list: *mut u8, num: u32, hit: *mut u32) -> i32 {
        if hit.is_null() || (list.is_null() && num != 0) { return 0x817D_0002u32 as i32; }
        let owned = addon_labels();
        // (A count with no list; else the entries that fit.)
        let n = if list.is_null() { owned.len() } else { owned.len().min(num as usize) };
        if !list.is_null() {
            for (i, l) in owned.iter().take(n).enumerate() {
                let e = list.add(i * ADDCONT_ENTRY);
                std::ptr::write_bytes(e, 0, ADDCONT_ENTRY);
                std::ptr::copy_nonoverlapping(l.as_ptr(), e, l.len().min(16));
            }
        }
        *hit = n as u32;
        static LOGGED: std::sync::atomic::AtomicU32 = std::sync::atomic::AtomicU32::new(0);
        if LOGGED.fetch_add(1, Ordering::Relaxed) < 4 {
            let what = if owned.is_empty() { "none (the add-ons line says where they were looked for)".to_string() } else { owned.join(", ") };
            crate::info!("sceNpEntitlementAccessGetAddcontEntitlementInfoList ({}, {num} asked): {n} add-on(s) reported as owned: {what}", if list.is_null() { "the count" } else { "the list" });
        }
        0
    }

    // ---- Universal Data System (telemetry): contexts exist, events are dropped.
    fn sceNpUniversalDataSystemInitialize(param: u64) -> i32 { stub_log("sceNpUniversalDataSystemInitialize (offline: telemetry events are discarded)", 0); 0 }
    fn sceNpUniversalDataSystemCreateContext(out: *mut i32, user: i32, a: u32, b: u64) -> i32 {
        if !out.is_null() { *out = next_id(); }
        0
    }
    fn sceNpUniversalDataSystemCreateHandle(out: *mut i32) -> i32 {
        if !out.is_null() { *out = next_id(); }
        0
    }
    fn sceNpUniversalDataSystemRegisterContext(ctx: i32, h: i32, flags: u32) -> i32 { 0 }
    // ---- UDS events: property objects/arrays and events are opaque handles
    // (host allocations); posting discards the event.
    fn sceNpUniversalDataSystemCreateEvent(name: *const u8, props: u64, out: *mut u64, props_out: *mut u64) -> i32 {
        if name.is_null() || out.is_null() { return UDS_ERROR_INVALID_ARGUMENT; }
        *out = uds_alloc();
        let p = if props != 0 { props } else { uds_alloc() };
        if !props_out.is_null() { *props_out = p; }
        UDS_DATA.lock().0.insert(*out, (cstr_or(name), p));
        0
    }
    fn sceNpUniversalDataSystemDestroyEvent(ev: u64) -> i32 { uds_free(ev); 0 }
    fn sceNpUniversalDataSystemCreateEventPropertyObject(out: *mut u64) -> i32 {
        if out.is_null() { return UDS_ERROR_INVALID_ARGUMENT; }
        *out = uds_alloc();
        0
    }
    fn sceNpUniversalDataSystemDestroyEventPropertyObject(obj: u64) -> i32 { uds_free(obj); 0 }
    fn sceNpUniversalDataSystemCreateEventPropertyArray(out: *mut u64) -> i32 {
        if out.is_null() { return UDS_ERROR_INVALID_ARGUMENT; }
        *out = uds_alloc();
        0
    }
    fn sceNpUniversalDataSystemDestroyEventPropertyArray(arr: u64) -> i32 { uds_free(arr); 0 }
    fn sceNpUniversalDataSystemEventPropertyObjectSetString(obj: u64, key: *const u8, v: *const u8) -> i32 { uds_set(obj, key, crate::achievements::Value::Str(cstr_or(v))) }
    fn sceNpUniversalDataSystemEventPropertyObjectSetInt32(obj: u64, key: *const u8, v: i32) -> i32 { uds_set(obj, key, crate::achievements::Value::Int(v as i64)) }
    fn sceNpUniversalDataSystemEventPropertyObjectSetUInt32(obj: u64, key: *const u8, v: u32) -> i32 { uds_set(obj, key, crate::achievements::Value::Int(v as i64)) }
    fn sceNpUniversalDataSystemEventPropertyObjectSetUInt64(obj: u64, key: *const u8, v: u64) -> i32 { uds_set(obj, key, crate::achievements::Value::Int(v as i64)) }
    /// (object, key, array or NULL, *array used)
    fn sceNpUniversalDataSystemEventPropertyObjectSetArray(obj: u64, key: *const u8, v: u64, out: *mut u64) -> i32 {
        if !out.is_null() { *out = if v != 0 { v } else { uds_alloc() }; }
        uds_check(obj)
    }
    fn sceNpUniversalDataSystemEventPropertyArraySetString(arr: u64, v: *const u8) -> i32 { uds_check(arr) }
    fn sceNpUniversalDataSystemEventPropertyArraySetFloat32(arr: u64, v: f32) -> i32 { uds_check(arr) }
    /// (array, object or NULL, *object used)
    fn sceNpUniversalDataSystemEventPropertyArraySetObject(arr: u64, v: u64, out: *mut u64) -> i32 {
        if !out.is_null() { *out = if v != 0 { v } else { uds_alloc() }; }
        uds_check(arr)
    }
    fn sceNpUniversalDataSystemPostEvent(ctx: i32, h: i32, ev: u64, options: u64) -> i32 {
        // The trophies' events reach the port's achievements; nothing goes to a network.
        let e = {
            let g = UDS_DATA.lock();
            g.0.get(&ev).map(|(name, p)| (name.clone(), g.1.get(p).cloned().unwrap_or_default()))
        };
        if let Some((name, props)) = e {
            crate::achievements::event(&name, &props);
        }
        stub_log("sceNpUniversalDataSystemPostEvent (offline: events reach the port's achievements, nothing else)", 0);
        0
    }

    // ---- Game updates: no update is available (offline console).
    fn sceGameUpdateCreateRequest() -> i32 { next_id() }
    fn sceGameUpdateDeleteRequest(req: i32) -> i32 { 0 }
    fn sceGameUpdateAbortRequest(req: i32) -> i32 { 0 }
    /// (request, param {size_t size, ...}, result {size_t size, bool found,
    /// bool addcont_found, ..., char content_version[11], ...})
    fn sceGameUpdateCheck(req: i32, param: *const u64, result: *mut u64) -> i32 {
        if param.is_null() || result.is_null() { return GAME_UPDATE_ERROR_INVALID_ARG; }
        let size = *result;
        if size < 16 || size > 0x400 { return GAME_UPDATE_ERROR_INVALID_SIZE; }
        std::ptr::write_bytes(result as *mut u8, 0, size as usize);
        *result = size;
        stub_log("sceGameUpdateCheck (offline: no update found)", 0);
        0
    }

    // ---- HTTP/2: templates and requests exist; nothing reaches a network,
    // so sends complete with a timeout.
    fn sceHttp2CreateTemplate(ctx: i32, user_agent: *const u8, version: i32, auto_proxy: i32) -> i32 { next_id() }
    fn sceHttp2DeleteTemplate(id: i32) -> i32 { 0 }
    fn sceHttp2CreateRequestWithURL(tmpl: i32, method: *const u8, url: *const u8, length: u64) -> i32 {
        if method.is_null() || url.is_null() { return HTTP2_ERROR_NULL_POINTER; }
        crate::debug!("sceHttp2CreateRequestWithURL {} {}", cstr_or(method), cstr_or(url));
        next_id()
    }
    fn sceHttp2DeleteRequest(id: i32) -> i32 { 0 }
    fn sceHttp2AddRequestHeader(id: i32, name: *const u8, value: *const u8, mode: u32) -> i32 { 0 }
    fn sceHttp2SendRequestAsync(id: i32, data: u64, size: u64, kq: u64, opt: u64) -> i32 {
        stub_log("sceHttp2SendRequestAsync (offline: completes with a timeout)", 0);
        0
    }
    /// Async result {i32 event_type, i32 req_id, i32 result, pad, void* reserved}.
    fn sceHttp2WaitAsync(id: i32, result: *mut i32, timeout: *mut u32, opt: u64) -> i32 {
        if result.is_null() { return HTTP2_ERROR_NULL_POINTER; }
        std::ptr::write_bytes(result as *mut u8, 0, 24);
        *result.add(1) = id;
        *result.add(2) = HTTP2_ERROR_TIMEOUT;
        0
    }
    fn sceHttp2ReadDataAsync(id: i32, data: u64, size: u64, kq: u64, opt: u64) -> i32 { HTTP2_ERROR_BEFORE_SEND }
    fn sceHttp2GetStatusCode(id: i32, status: *mut i32) -> i32 { HTTP2_ERROR_BEFORE_SEND }
    fn sceHttp2GetResponseContentLength(id: i32, result: *mut i32, length: *mut u64) -> i32 { HTTP2_ERROR_BEFORE_SEND }
    fn sceHttp2GetAllResponseHeaders(id: i32, header: *mut u64, size: *mut u64) -> i32 { HTTP2_ERROR_BEFORE_SEND }

    // ---- NP account queries: signed out.
    fn sceNpGetAccountIdA(user: i32, id: *mut u64) -> i32 {
        if !id.is_null() { *id = 0; }
        NP_ERROR_SIGNED_OUT
    }
    fn sceNpGetAccountCountryA(user: i32, code: *mut u8) -> i32 {
        if !code.is_null() { std::ptr::write_bytes(code, 0, 4); }
        NP_ERROR_SIGNED_OUT
    }
    fn sceNpCreateRequest() -> i32 { next_id() }
    fn sceNpCheckPremium(req: i32, param: u64, result: *mut u8) -> i32 {
        if !result.is_null() { std::ptr::write_bytes(result, 0, 16); }
        NP_ERROR_SIGNED_OUT
    }

    // ---- Sign-in dialog: shown, and the user does not sign in.
    fn sceSigninDialogInitialize() -> i32 { *SIGNIN.lock() = (SIGNIN_INITIALIZED, false); 0 }
    fn sceSigninDialogOpen(param: u64) -> i32 {
        crate::info!("sign-in dialog opened (offline console: the user does not sign in)");
        *SIGNIN.lock() = (SIGNIN_RUNNING, false);
        0
    }
    fn sceSigninDialogUpdateStatus() -> i32 {
        let mut g = SIGNIN.lock();
        if g.0 == SIGNIN_RUNNING {
            if g.1 { g.0 = SIGNIN_FINISHED; }
            g.1 = true;
        }
        g.0
    }
    fn sceSigninDialogGetStatus() -> i32 { SIGNIN.lock().0 }
    /// {i32 result (1 = user canceled), i32 reserved[3]}
    fn sceSigninDialogGetResult(result: *mut i32) -> i32 {
        if result.is_null() { return 0x8135_0003u32 as i32; }
        std::ptr::write_bytes(result as *mut u8, 0, 16);
        *result = 1;
        0
    }
    fn sceSigninDialogClose() -> i32 { let mut g = SIGNIN.lock(); if g.0 == SIGNIN_RUNNING { g.0 = SIGNIN_FINISHED; } 0 }
    fn sceSigninDialogTerminate() -> i32 { *SIGNIN.lock() = (0, false); 0 }

    /// Whether a system dialog is on screen (the save-data dialog).
    fn sceCommonDialogIsUsed() -> u32 { super::savedata::dialog_running() as u32 }

    /// 24-byte memory statistics (the title zeroes it first, eboot+0x650132):
    /// nothing is allocated because events are discarded.
    fn sceNpUniversalDataSystemGetMemoryStat(stat: *mut u64) -> i32 {
        if stat.is_null() { return 0x8155_1002u32 as i32; }
        std::ptr::write_bytes(stat as *mut u8, 0, 24);
        0
    }
}

pub fn register(r: &mut Registry) {
    reg!(r, "libSceJson2", [
        _ZN3sce4Json12MemAllocatorC2Ev, _ZN3sce4Json12MemAllocatorD2Ev, _ZN3sce4Json11InitializerC1Ev,
        _ZN3sce4Json11InitializerD1Ev, _ZN3sce4Json11Initializer10initializeEPKNS0_13InitParameterE,
    ]);
    reg!(r, "libSceHttp2", [sceHttp2Init, sceHttp2Term]);
    reg!(r, "libSceNpWebApi2", [
        sceNpWebApi2Initialize, sceNpWebApi2Terminate, sceNpWebApi2CreateUserContext, sceNpWebApi2DeleteUserContext,
        sceNpWebApi2CreateRequest, sceNpWebApi2AddHttpRequestHeader, sceNpWebApi2SendRequest, sceNpWebApi2ReadData,
        sceNpWebApi2DeleteRequest, sceNpWebApi2PushEventCreateHandle, sceNpWebApi2PushEventDeleteHandle,
        sceNpWebApi2PushEventCreateFilter, sceNpWebApi2PushEventRegisterCallback, sceNpWebApi2PushEventDeletePushContext,
    ]);
    reg!(r, "libSceGameUpdate", [sceGameUpdateInitialize]);
    reg!(r, "libSceNpGameIntent", [sceNpGameIntentInitialize]);
    reg!(r, "libSceSystemGesture", [
        sceSystemGestureInitializePrimitiveTouchRecognizer, sceSystemGestureFinalizePrimitiveTouchRecognizer,
        sceSystemGestureOpen, sceSystemGestureClose, sceSystemGestureCreateTouchRecognizer,
        sceSystemGestureUpdatePrimitiveTouchRecognizer, sceSystemGestureUpdateTouchRecognizer,
        sceSystemGestureGetTouchEventsCount, sceSystemGestureGetTouchEventByIndex,
    ]);
    reg!(r, "libSceShare", [sceShareInitialize, sceShareSetContentParam]);
    reg!(r, "libSceNpManager", [sceNpRegisterStateCallbackA, sceNpUnregisterStateCallbackA]);
    reg!(r, "libSceNpSessionSignaling", [sceNpSessionSignalingInitialize]);
    r.func_nid("libSceUserServicePlatformPrivacyWs1", "D-CzAxQL0XI", "D-CzAxQL0XI (unnamed)", ds_user_service_platform_privacy as *const () as u64);
    reg!(r, "libSceNpEntitlementAccess", [sceNpEntitlementAccessInitialize, sceNpEntitlementAccessGetAddcontEntitlementInfoList]);
    reg!(r, "libSceNpUniversalDataSystem", [
        sceNpUniversalDataSystemInitialize, sceNpUniversalDataSystemCreateContext, sceNpUniversalDataSystemCreateHandle,
        sceNpUniversalDataSystemRegisterContext, sceNpUniversalDataSystemGetMemoryStat,
        sceNpUniversalDataSystemCreateEvent, sceNpUniversalDataSystemDestroyEvent, sceNpUniversalDataSystemCreateEventPropertyObject,
        sceNpUniversalDataSystemDestroyEventPropertyObject, sceNpUniversalDataSystemCreateEventPropertyArray,
        sceNpUniversalDataSystemDestroyEventPropertyArray, sceNpUniversalDataSystemEventPropertyObjectSetString,
        sceNpUniversalDataSystemEventPropertyObjectSetInt32, sceNpUniversalDataSystemEventPropertyObjectSetUInt32,
        sceNpUniversalDataSystemEventPropertyObjectSetUInt64, sceNpUniversalDataSystemEventPropertyObjectSetArray,
        sceNpUniversalDataSystemEventPropertyArraySetString, sceNpUniversalDataSystemEventPropertyArraySetFloat32,
        sceNpUniversalDataSystemEventPropertyArraySetObject, sceNpUniversalDataSystemPostEvent,
    ]);
    reg!(r, "libSceGameUpdate", [sceGameUpdateCreateRequest, sceGameUpdateDeleteRequest, sceGameUpdateAbortRequest, sceGameUpdateCheck]);
    reg!(r, "libSceHttp2", [
        sceHttp2CreateTemplate, sceHttp2DeleteTemplate, sceHttp2CreateRequestWithURL, sceHttp2DeleteRequest, sceHttp2AddRequestHeader,
        sceHttp2SendRequestAsync, sceHttp2WaitAsync, sceHttp2ReadDataAsync, sceHttp2GetStatusCode, sceHttp2GetResponseContentLength,
        sceHttp2GetAllResponseHeaders,
    ]);
    reg!(r, "libSceNpManager", [sceNpGetAccountIdA, sceNpGetAccountCountryA, sceNpCreateRequest, sceNpCheckPremium]);
    reg!(r, "libSceSigninDialog", [
        sceSigninDialogInitialize, sceSigninDialogOpen, sceSigninDialogUpdateStatus, sceSigninDialogGetStatus, sceSigninDialogGetResult,
        sceSigninDialogClose, sceSigninDialogTerminate,
    ]);
    reg!(r, "libSceCommonDialog", [sceCommonDialogIsUsed]);
}

#[cfg(test)]
mod tests {
    use super::{addon_label, labels_in_bytes};

    #[test]
    #[cfg(target_os = "linux")]
    fn addon_scan_skips_symlinks() {
        let base = std::env::temp_dir().join(format!("dsport-addons-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&base);
        let dir = base.join("Add-ons");
        let outside = base.join("outside");
        std::fs::create_dir_all(&dir).unwrap();
        std::fs::create_dir_all(&outside).unwrap();
        std::fs::write(outside.join("DEMONSSOULSGIFT2.pkg"), b"fixture").unwrap();
        std::fs::write(dir.join("DEMONSSOULSGIFT1.pkg"), b"fixture").unwrap();
        std::os::unix::fs::symlink(&outside, dir.join("linked-folder")).unwrap();
        std::os::unix::fs::symlink(outside.join("DEMONSSOULSGIFT2.pkg"), dir.join("DEMONSSOULSGIFT2.pkg")).unwrap();
        let mut got = super::AddonScan::default();
        let mut budget = super::ADDON_READS;
        super::scan_addons(&dir, &dir, 3, &mut got, &mut budget);
        assert_eq!(got.labels.iter().map(|(label, _)| label.as_str()).collect::<Vec<_>>(), ["DEMONSSOULSGIFT1"]);
        got.labels.clear();
        super::scan_addons(&dir, &dir.join("linked-folder"), 3, &mut got, &mut budget);
        assert!(got.labels.is_empty());
        std::fs::remove_dir_all(base).unwrap();
    }

    /// Add-on labels from package names as players have them (2026-11-24: the pre-order GIFT1
    /// and GIFT2 packages, as a .pkg file or the folders unpacked from it).
    #[test]
    fn addon_labels_from_package_names() {
        assert_eq!(addon_label("EP9000-PPSA01341_00-DEMONSSOULSGIFT1.pkg").as_deref(), Some("DEMONSSOULSGIFT1"));
        assert_eq!(addon_label("EP9000-PPSA01341_00-DEMONSSOULSGIFT2").as_deref(), Some("DEMONSSOULSGIFT2"));
        assert_eq!(addon_label("UP9000-PPSA01342_00-DEMONSSOULSGIFT1").as_deref(), Some("DEMONSSOULSGIFT1"));
        assert_eq!(addon_label("common"), None);
        assert_eq!(addon_label("playgo-chunk.dat"), None);
        assert_eq!(addon_label("EP9000-PPSA01341_00-SHORT"), None);
        assert_eq!(addon_label("EP9000-PPSA01341_00-demonssoulsgift1"), None);
        assert_eq!(addon_label("EP9000-PPSA01341-DEMONSSOULSGIFT1"), None);
        // A package renamed to its label alone.
        assert_eq!(addon_label("DEMONSSOULSGIFT2.pkg").as_deref(), Some("DEMONSSOULSGIFT2"));
        assert_eq!(addon_label("DEMONSSOULSGIFT1").as_deref(), Some("DEMONSSOULSGIFT1"));
        assert_eq!(addon_label("PS5 SAVES BACKUP.pkg"), None);
        assert_eq!(addon_label("DEMONSSOULSGIFT12.pkg"), None);
    }

    /// Add-on labels inside files of any name (2026-11-24: a package renamed, an unpacked
    /// folder's param.json).
    #[test]
    fn addon_labels_inside_files() {
        let json = br#"{"contentId":"EP9000-PPSA01341_00-DEMONSSOULSGIFT2","titleId":"PPSA01341"}"#;
        assert_eq!(labels_in_bytes(json), vec!["DEMONSSOULSGIFT2".to_string()]);
        let mut header = vec![0u8; 64];
        header.extend_from_slice(b"UP9000-PPSA01342_00-DEMONSSOULSGIFT1\0\0\0EP9000-PPSA01342_00-DEMONSSOULSGIFT1");
        assert_eq!(labels_in_bytes(&header), vec!["DEMONSSOULSGIFT1".to_string()]);
        assert!(labels_in_bytes(b"PPSA01341 PPSA0134").is_empty());
        assert!(labels_in_bytes(b"EP9000-PPSA01341_00-DEMONSSOULSGIF").is_empty());
        assert!(labels_in_bytes(b"").is_empty());
    }
}
