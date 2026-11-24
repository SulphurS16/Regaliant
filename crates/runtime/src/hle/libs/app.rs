//! Application services: AppContent (+ param.sfo), UserService, SystemService, PlayGo.

use crate::hle::{Registry, cstr, write_cstr};
use crate::{fixed, reg, sysv};
use std::collections::HashMap;
use std::sync::OnceLock;
use std::sync::atomic::{AtomicBool, Ordering};

#[derive(Debug, Clone)]
pub enum SfoValue {
    Int(u32),
    Str(String),
}

/// Parses `/app0/sce_sys/param.sfo` once.
pub fn param_sfo() -> &'static HashMap<String, SfoValue> {
    static SFO: OnceLock<HashMap<String, SfoValue>> = OnceLock::new();
    SFO.get_or_init(|| {
        let mut m = HashMap::new();
        let Ok(b) = crate::hle::kernel::fs::read_file("/app0/sce_sys/param.sfo") else {
            crate::warn!("param.sfo not found");
            return m;
        };
        let rd32 = |o: usize| u32::from_le_bytes(b[o..o + 4].try_into().unwrap());
        let rd16 = |o: usize| u16::from_le_bytes(b[o..o + 2].try_into().unwrap());
        if b.len() < 20 || rd32(0) != 0x4653_5000 {
            crate::warn!("param.sfo has a bad header");
            return m;
        }
        let (keys, data, n) = (rd32(8) as usize, rd32(12) as usize, rd32(16) as usize);
        for i in 0..n {
            let e = 20 + i * 16;
            if e + 16 > b.len() {
                break;
            }
            let ko = keys + rd16(e) as usize;
            let fmt = rd16(e + 2);
            let len = rd32(e + 4) as usize;
            let off = data + rd32(e + 12) as usize;
            let kend = b[ko..].iter().position(|&c| c == 0).map(|p| ko + p).unwrap_or(b.len());
            let key = String::from_utf8_lossy(&b[ko..kend]).into_owned();
            let v = if fmt == 0x0404 {
                SfoValue::Int(rd32(off))
            } else {
                let s = &b[off..off + len];
                let end = s.iter().position(|&c| c == 0).unwrap_or(s.len());
                SfoValue::Str(String::from_utf8_lossy(&s[..end]).into_owned())
            };
            m.insert(key, v);
        }
        crate::info!("param.sfo: TITLE_ID={:?} APP_VER={:?} CATEGORY={:?}", m.get("TITLE_ID"), m.get("APP_VER"), m.get("CATEGORY"));
        m
    })
}

fn sfo_int(key: &str) -> Option<u32> {
    match param_sfo().get(key) {
        Some(SfoValue::Int(v)) => Some(*v),
        _ => None,
    }
}

/// Parses `/app0/sce_sys/param.json` (PS5) once.
pub fn param_json() -> &'static serde_json::Value {
    static J: OnceLock<serde_json::Value> = OnceLock::new();
    J.get_or_init(|| {
        let Ok(b) = crate::hle::kernel::fs::read_file("/app0/sce_sys/param.json") else {
            crate::warn!("param.json not found");
            return serde_json::Value::Null;
        };
        match serde_json::from_slice(&b).map_err(|e| e.to_string()) {
            Ok(v) => v,
            Err(e) => {
                crate::warn!("param.json unreadable: {e}");
                serde_json::Value::Null
            }
        }
    })
}

fn json_int(key: &str) -> Option<i64> {
    param_json().get(key).and_then(|v| v.as_i64())
}

pub const USER_ID: i32 = 1;
static USER_EVENT_SENT: AtomicBool = AtomicBool::new(false);
static PLAYGO_CHUNKS: std::sync::OnceLock<Vec<u16>> = std::sync::OnceLock::new();

const PLAYGO_HANDLE: i32 = 1;
const LOCUS_LOCAL_FAST: i8 = 3;
const PG_INVALID_ARGUMENT: i32 = 0x80B2_0004u32 as i32;
const PG_BAD_HANDLE: i32 = 0x80B2_0009u32 as i32;
const PG_BAD_POINTER: i32 = 0x80B2_000Au32 as i32;
const PG_BAD_SIZE: i32 = 0x80B2_000Bu32 as i32;
const PG_BAD_CHUNK_ID: i32 = 0x80B2_000Cu32 as i32;
const PG_BAD_SPEED: i32 = 0x80B2_000Du32 as i32;
const PG_BAD_LOCUS: i32 = 0x80B2_0010u32 as i32;

/// PS5 packages define chunks in `playgo-chunkdefs.xml` (packaging file at
/// the application root). Every defined chunk is installed in the supplied
/// tree, so all report as local. Returns the defined chunk ids, sorted.
fn playgo_chunks() -> &'static [u16] {
    PLAYGO_CHUNKS.get_or_init(|| {
        let mut ids: Vec<u16> = crate::hle::kernel::fs::read_file("/app0/playgo-chunkdefs.xml")
            .ok()
            .and_then(|b| String::from_utf8(b).ok())
            .map(|x| {
                x.split("<chunk id=\"").skip(1).filter_map(|t| t.split('"').next()?.parse::<u16>().ok()).collect()
            })
            .unwrap_or_default();
        ids.sort_unstable();
        ids.dedup();
        if ids.is_empty() {
            ids.push(0);
        }
        crate::info!("PlayGo: {} chunk definition(s) (ids {}..={}), all reported installed (local)", ids.len(), ids[0], ids[ids.len() - 1]);
        ids
    })
}

fn chunk_defined(id: u16) -> bool {
    playgo_chunks().binary_search(&id).is_ok()
}

static PLAYGO_SPEED: std::sync::atomic::AtomicI32 = std::sync::atomic::AtomicI32::new(1); // TRICKLE

sysv! {
    // ---- AppContent
    fn sceAppContentInitialize(init: *const u8, boot: *mut u8) -> i32 {
        if !boot.is_null() { std::ptr::write_bytes(boot, 0, 40); }
        let j = param_json();
        crate::info!("param.json: {} {} {}", j.get("titleId").and_then(|v| v.as_str()).unwrap_or("?"),
            j.get("contentId").and_then(|v| v.as_str()).unwrap_or("?"), j.get("contentVersion").and_then(|v| v.as_str()).unwrap_or("?"));
        0
    }
    fn sceAppContentAppParamGetInt(id: i32, value: *mut i32) -> i32 {
        if value.is_null() { return 0x80D9_0002u32 as i32; }
        *value = match id {
            0 => 3, // SKU_FLAG_FULL
            1..=4 => json_int(&format!("userDefinedParam{id}")).unwrap_or(0) as i32,
            _ => return 0x80D9_0002u32 as i32,
        };
        crate::debug!("sceAppContentAppParamGetInt({id}) = {}", *value);
        0
    }
    fn sceAppContentGetAddcontInfoList(label: u32, list: *mut u8, num: u32, hit: *mut u32) -> i32 {
        if !hit.is_null() { *hit = 0; }
        crate::info!("sceAppContentGetAddcontInfoList: no additional content installed");
        0
    }

    // ---- UserService
    fn sceUserServiceInitialize(params: *const u8) -> i32 { 0 }
    fn sceUserServiceTerminate() -> i32 { 0 }
    fn sceUserServiceGetInitialUser(user: *mut i32) -> i32 {
        if user.is_null() { return 0x8096_0003u32 as i32; }
        *user = USER_ID;
        0
    }
    fn sceUserServiceGetLoginUserIdList(list: *mut i32) -> i32 {
        if list.is_null() { return 0x8096_0003u32 as i32; }
        *list = USER_ID;
        for i in 1..4 { *list.add(i) = -1; }
        0
    }
    fn sceUserServiceGetUserName(user: i32, name: *mut u8, size: u64) -> i32 {
        write_cstr(name, size as usize, "Hunter");
        0
    }
    /// Game presets {u64 size, u32 fields...}: every field 0 = not set, so
    /// the title keeps its own defaults (eboot+0x7c91e9 reads them that way).
    fn sceUserServiceGetGamePresets(user: i32, presets: *mut u64) -> i32 {
        if presets.is_null() { return 0x8096_0003u32 as i32; }
        let size = (*presets).clamp(8, 0x100);
        std::ptr::write_bytes((presets as *mut u8).add(8), 0, size as usize - 8);
        0
    }
    fn sceUserServiceGetEvent(ev: *mut i32) -> i32 {
        if !USER_EVENT_SENT.swap(true, Ordering::SeqCst) {
            *ev = 0; // LOGIN
            *ev.add(1) = USER_ID;
            return 0;
        }
        0x8096_0007u32 as i32 // NO_EVENT
    }

    // ---- SystemService
    fn sceSystemServiceGetStatus(status: *mut u8) -> i32 {
        // SceSystemServiceStatus: int32 eventNum + 5 bools (+ reserved, left untouched).
        if !status.is_null() { std::ptr::write_bytes(status, 0, 9); }
        0
    }
    fn sceSystemServiceReceiveEvent(ev: *mut u8) -> i32 { 0x80A1_0003u32 as i32 }
    fn sceSystemServiceParamGetInt(id: i32, value: *mut i32) -> i32 {
        if value.is_null() { return 0x80A1_0002u32 as i32; }
        *value = match id {
            1 => 1,    // LANG: English (United States)
            2 => 0,    // DATE_FORMAT: YYYYMMDD
            3 => 1,    // TIME_FORMAT: 24 hour
            4 => 0,    // TIME_ZONE (minutes)
            5 => 0,    // SUMMERTIME
            7 => 0,    // GAME_PARENTAL_LEVEL: off
            1000 => 1, // ENTER_BUTTON_ASSIGN: cross
            _ => {
                crate::warn!("sceSystemServiceParamGetInt: unknown id {id}");
                return 0x80A1_0002u32 as i32;
            }
        };
        0
    }
    fn sceSystemServiceParamGetString(id: i32, buf: *mut u8, size: u64) -> i32 {
        write_cstr(buf, size as usize, "PS4");
        0
    }
    /// SceSystemServiceDisplaySafeAreaInfo {float ratio; reserved[128]}: the
    /// whole display is safe (the console default).
    fn sceSystemServiceGetDisplaySafeAreaInfo(info: *mut f32) -> i32 {
        if info.is_null() { return 0x80A1_0002u32 as i32; }
        std::ptr::write_bytes(info as *mut u8, 0, 4 + 128);
        *info = safe_area_ratio();
        0
    }
    /// System setting to skip notice screens: off, so the title shows its
    /// notices as on a console with default settings.
    fn sceSystemServiceGetNoticeScreenSkipFlag(flag: *mut u8) -> i32 {
        if flag.is_null() { return 0x80A1_0002u32 as i32; }
        *flag = 0;
        0
    }
    fn sceSystemServiceHideSplashScreen() -> i32 {
        crate::info!("sceSystemServiceHideSplashScreen");
        0
    }

    // ---- PlayGo: argument checks and error codes follow the library's
    // documented behavior (codes per public sources, docs/PROVENANCE.md). Titles
    // probe chunk ids and rely on BAD_CHUNK_ID for undefined ones: this
    // title collects accepted ids (0..999 probed) into a 128-bit set.
    fn scePlayGoOpen(handle: *mut i32, param: *const u8) -> i32 {
        if handle.is_null() { return PG_BAD_POINTER; }
        if !param.is_null() { return PG_INVALID_ARGUMENT; }
        playgo_chunks();
        *handle = PLAYGO_HANDLE;
        0
    }
    fn scePlayGoGetLocus(handle: i32, ids: *const u16, n: u32, loci: *mut i8) -> i32 {
        if handle != PLAYGO_HANDLE { return PG_BAD_HANDLE; }
        if ids.is_null() || loci.is_null() { return PG_BAD_POINTER; }
        if n == 0 { return PG_BAD_SIZE; }
        for i in 0..n as usize {
            if !chunk_defined(*ids.add(i)) { return PG_BAD_CHUNK_ID; }
            *loci.add(i) = LOCUS_LOCAL_FAST;
        }
        0
    }
    fn scePlayGoGetInstallSpeed(handle: i32, speed: *mut i32) -> i32 {
        if handle != PLAYGO_HANDLE { return PG_BAD_HANDLE; }
        if speed.is_null() { return PG_BAD_POINTER; }
        *speed = PLAYGO_SPEED.load(Ordering::Relaxed);
        0
    }
    fn scePlayGoSetInstallSpeed(handle: i32, speed: i32) -> i32 {
        if handle != PLAYGO_HANDLE { return PG_BAD_HANDLE; }
        if !(0..=2).contains(&speed) { return PG_BAD_SPEED; }
        PLAYGO_SPEED.store(speed, Ordering::Relaxed);
        0
    }
    fn scePlayGoGetEta(handle: i32, ids: *const u16, n: u32, eta: *mut i64) -> i32 {
        if handle != PLAYGO_HANDLE { return PG_BAD_HANDLE; }
        if ids.is_null() || eta.is_null() { return PG_BAD_POINTER; }
        if n == 0 { return PG_BAD_SIZE; }
        for i in 0..n as usize {
            if !chunk_defined(*ids.add(i)) { return PG_BAD_CHUNK_ID; }
        }
        *eta = 0; // installed
        0
    }
    fn scePlayGoGetToDoList(handle: i32, todo: *mut u8, n: u32, entries: *mut u32) -> i32 {
        if handle != PLAYGO_HANDLE { return PG_BAD_HANDLE; }
        if entries.is_null() || (n != 0 && todo.is_null()) { return PG_BAD_POINTER; }
        *entries = 0; // nothing left to install
        0
    }
    fn scePlayGoPrefetch(handle: i32, ids: *const u16, n: u32, min_locus: i32) -> i32 {
        if handle != PLAYGO_HANDLE { return PG_BAD_HANDLE; }
        if ids.is_null() { return PG_BAD_POINTER; }
        if n == 0 { return PG_BAD_SIZE; }
        if !matches!(min_locus, 0 | 2 | 3) { return PG_BAD_LOCUS; }
        for i in 0..n as usize {
            if !chunk_defined(*ids.add(i)) { return PG_BAD_CHUNK_ID; }
        }
        0
    }
    /// With no list and no entries requested, reports the number of chunks.
    fn scePlayGoGetChunkId(handle: i32, out: *mut u16, n: u32, entries: *mut u32) -> i32 {
        if handle != PLAYGO_HANDLE { return PG_BAD_HANDLE; }
        if entries.is_null() || (n != 0 && out.is_null()) { return PG_BAD_POINTER; }
        let chunks = playgo_chunks();
        if out.is_null() {
            *entries = chunks.len() as u32;
            return 0;
        }
        let k = chunks.len().min(n as usize);
        for (i, id) in chunks[..k].iter().enumerate() { *out.add(i) = *id; }
        *entries = k as u32;
        0
    }
}

fixed! {
    scePlayGoInitialize => 0,
    sceSystemServiceLaunchWebBrowser => 0x80A1_0002u32,
}

pub fn register(r: &mut Registry) {
    reg!(r, "libSceAppContent", [sceAppContentInitialize, sceAppContentAppParamGetInt, sceAppContentGetAddcontInfoList]);
    reg!(r, "libSceUserService", [
        sceUserServiceInitialize, sceUserServiceTerminate, sceUserServiceGetInitialUser,
        sceUserServiceGetLoginUserIdList, sceUserServiceGetUserName, sceUserServiceGetEvent, sceUserServiceGetGamePresets,
    ]);
    reg!(r, "libSceSystemService", [
        sceSystemServiceGetStatus, sceSystemServiceReceiveEvent, sceSystemServiceParamGetInt,
        sceSystemServiceParamGetString, sceSystemServiceHideSplashScreen, sceSystemServiceGetDisplaySafeAreaInfo, sceSystemServiceLaunchWebBrowser,
        sceSystemServiceGetNoticeScreenSkipFlag,
    ]);
    reg!(r, "libScePlayGo", [
        scePlayGoInitialize, scePlayGoOpen, scePlayGoGetLocus, scePlayGoGetChunkId, scePlayGoSetInstallSpeed,
        scePlayGoGetInstallSpeed, scePlayGoGetEta, scePlayGoGetToDoList, scePlayGoPrefetch,
    ]);
    let _ = cstr; // shared helper import
}

/// The display safe area the title is told (1.0: the whole screen; the PC tab's HUD Size, or
/// `DS_SAFE_AREA=<ratio>` for tests).
fn safe_area_ratio() -> f32 {
    if let Some(v) = std::env::var("DS_SAFE_AREA").ok().and_then(|v| v.trim().parse::<f32>().ok()) {
        return v.clamp(0.5, 1.0);
    }
    crate::platform::loader_hooks::safe_area().unwrap_or(1.0)
}
