//! PlayStation controllers through Windows' HID, without Steam Input, DS4Windows or another helper
//! program: DUALSHOCK 4, DualSense and DualSense Edge, over USB or Bluetooth, connected at any
//! time (a watcher looks for new ones every 1.5 s; a thread per controller waits for its reports).
//!
//! Their input reports are put in XInput's layout (`Gamepad`: Cross A, Circle B, Square X,
//! Triangle Y, L1 / R1 the shoulders, L2 / R2 the triggers' travel, Options Menu, the touch pad's
//! click View, the sticks in XInput's range), so the pad layer (`input`) maps them as it maps an
//! Xbox pad and merges every controller into the game's one pad. Create / Share, PS and mute
//! reach nothing, as on the PS5. Layouts:
//!   DUALSHOCK 4: report 0x01 over USB, and over Bluetooth until a program writes to it; then
//!     0x11 with the same data 2 bytes later. Data: left X, left Y, right X, right Y (0-255,
//!     128 the centre, Y down), the d-pad (low 4 bits: 0 up, clockwise to 7, 8 none) with Square
//!     0x10, Cross 0x20, Circle 0x40, Triangle 0x80; then L1 0x01, R1 0x02, L2 0x04, R2 0x08,
//!     Share 0x10, Options 0x20, L3 0x40, R3 0x80; then PS 0x01, the touch pad's click 0x02; then
//!     the L2 and R2 travel (0-255).
//!   DualSense / Edge: report 0x01 over USB (64 bytes) and 0x31 over Bluetooth (one byte later):
//!     the sticks as above, the L2 and R2 travel, a counter, then the three button bytes of the
//!     DUALSHOCK 4 (Create for Share; mute 0x04 beside PS and the touch pad). Over Bluetooth
//!     before a program writes to it: report 0x01 in the DUALSHOCK 4's layout.
//!
//! The game's vibration drives their motors and its light bar colour their light bar: output
//! reports written by a thread per controller with the latest state, one every 8 ms at most (the
//! DualSense: report 0x02 over USB, 0x31 with a CRC-32 over Bluetooth, its compatible vibration;
//! the DUALSHOCK 4: 0x05, or 0x11 with a CRC-32). The report layouts are interface facts from
//! public sources (docs/PROVENANCE.md). The shader preparation's self-play never opens them
//! (`input::SYSTEM_OFF` keeps `input` from asking).

use parking_lot::{Condvar, Mutex};
use std::collections::HashSet;
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering::Relaxed};
use std::time::Duration;
use windows_sys::Win32::Devices::DeviceAndDriverInstallation::{
    DIGCF_DEVICEINTERFACE, DIGCF_PRESENT, SP_DEVICE_INTERFACE_DATA, SP_DEVICE_INTERFACE_DETAIL_DATA_W, SetupDiDestroyDeviceInfoList, SetupDiEnumDeviceInterfaces,
    SetupDiGetClassDevsW, SetupDiGetDeviceInterfaceDetailW,
};
use windows_sys::Win32::Devices::HumanInterfaceDevice::{
    HIDD_ATTRIBUTES, HIDP_CAPS, HIDP_STATUS_SUCCESS, HidD_FreePreparsedData, HidD_GetAttributes, HidD_GetFeature, HidD_GetHidGuid, HidD_GetPreparsedData, HidD_GetProductString, HidP_GetCaps,
};
use windows_sys::Win32::Foundation::{CloseHandle, ERROR_IO_PENDING, GENERIC_READ, GENERIC_WRITE, GetLastError, HANDLE, INVALID_HANDLE_VALUE, WAIT_OBJECT_0};
use windows_sys::Win32::Storage::FileSystem::{CreateFileW, FILE_FLAG_OVERLAPPED, FILE_SHARE_READ, FILE_SHARE_WRITE, OPEN_EXISTING, ReadFile, WriteFile};
use windows_sys::Win32::System::IO::{GetOverlappedResult, OVERLAPPED};
use windows_sys::Win32::System::Threading::{CreateEventW, INFINITE, ResetEvent, WaitForSingleObject};

// XInput's button bits.
pub const DPAD_UP: u16 = 0x0001;
pub const DPAD_DOWN: u16 = 0x0002;
pub const DPAD_LEFT: u16 = 0x0004;
pub const DPAD_RIGHT: u16 = 0x0008;
pub const START: u16 = 0x0010;
pub const BACK: u16 = 0x0020;
pub const LEFT_THUMB: u16 = 0x0040;
pub const RIGHT_THUMB: u16 = 0x0080;
pub const LEFT_SHOULDER: u16 = 0x0100;
pub const RIGHT_SHOULDER: u16 = 0x0200;
pub const BUTTON_A: u16 = 0x1000;
pub const BUTTON_B: u16 = 0x2000;
pub const BUTTON_X: u16 = 0x4000;
pub const BUTTON_Y: u16 = 0x8000;

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum Model {
    DualShock4,
    DualSense,
    DualSenseEdge,
}

const SONY: u16 = 0x054C;
const KNOWN: &[(u16, Model, &str)] = &[
    (0x05C4, Model::DualShock4, "DUALSHOCK 4"),
    (0x09CC, Model::DualShock4, "DUALSHOCK 4"),
    (0x0BA0, Model::DualShock4, "DUALSHOCK 4 (USB wireless adaptor)"),
    (0x0CE6, Model::DualSense, "DualSense"),
    (0x0DF2, Model::DualSenseEdge, "DualSense Edge"),
];
/// The Bluetooth HID service in a device path: the controller is connected over Bluetooth.
const BT_SERVICE: &str = "00001124-0000-1000-8000-00805f9b34fb";

/// A controller's state in XInput's layout.
#[derive(Clone, Copy, Default, PartialEq, Eq, Debug)]
pub struct Gamepad {
    pub buttons: u16,
    pub lt: u8,
    pub rt: u8,
    pub lx: i16,
    pub ly: i16,
    pub rx: i16,
    pub ry: i16,
}

/// A stick's X (0-255, 128 the centre) in XInput's range, so that the pad layer's scale (`v *
/// 257`, its `axis`) gives every one of the controller's 256 positions back.
fn axis_x(v: u8) -> i16 {
    (v as i32 * 257 - 32768) as i16
}

/// A stick's Y (0-255, down) in XInput's range (up).
fn axis_y(v: u8) -> i16 {
    (32767 - v as i32 * 257) as i16
}

/// One input report (report id first) in XInput's layout; None when it is not a state report.
pub fn parse_report(model: Model, bluetooth: bool, r: &[u8]) -> Option<Gamepad> {
    if r.len() < 10 {
        return None;
    }
    // `full`: the DualSense's layout (the triggers' travel before the buttons).
    let (d, full): (&[u8], bool) = match model {
        Model::DualShock4 => match r[0] {
            0x01 => (&r[1..], false),
            0x11..=0x19 if r.len() >= 12 => (&r[3..], false),
            _ => return None,
        },
        Model::DualSense | Model::DualSenseEdge => match r[0] {
            0x01 if !bluetooth && r.len() >= 11 => (&r[1..], true),
            0x01 if bluetooth => (&r[1..], false),
            0x31 if r.len() >= 12 => (&r[2..], true),
            _ => return None,
        },
    };
    let (b0, b1, b2, l2, r2) = if full { (d[7], d[8], d[9], d[4], d[5]) } else { (d[4], d[5], d[6], d[7], d[8]) };
    const HAT: [u16; 8] = [
        DPAD_UP,
        DPAD_UP | DPAD_RIGHT,
        DPAD_RIGHT,
        DPAD_DOWN | DPAD_RIGHT,
        DPAD_DOWN,
        DPAD_DOWN | DPAD_LEFT,
        DPAD_LEFT,
        DPAD_UP | DPAD_LEFT,
    ];
    let mut w = 0u16;
    if b0 & 0x0F < 8 {
        w |= HAT[(b0 & 0x0F) as usize];
    }
    for (bit, x) in [(0x10, BUTTON_X), (0x20, BUTTON_A), (0x40, BUTTON_B), (0x80, BUTTON_Y)] {
        if b0 & bit != 0 {
            w |= x;
        }
    }
    // (L2 0x04 and R2 0x08 come as their travel; Share / Create 0x10 reaches nothing.)
    for (bit, x) in [(0x01, LEFT_SHOULDER), (0x02, RIGHT_SHOULDER), (0x20, START), (0x40, LEFT_THUMB), (0x80, RIGHT_THUMB)] {
        if b1 & bit != 0 {
            w |= x;
        }
    }
    if b2 & 0x02 != 0 {
        w |= BACK; // the touch pad's click
    }
    Some(Gamepad { buttons: w, lt: l2, rt: r2, lx: axis_x(d[0]), ly: axis_y(d[1]), rx: axis_x(d[2]), ry: axis_y(d[3]) })
}

// ---------------------------------------------------------------------------
// Output reports

/// What the controllers are asked to show: the motors (left the strong one) and the light bar.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub struct Out {
    pub left: u8,
    pub right: u8,
    pub rgb: [u8; 3],
}

/// The light bar's colour until the game gives one: the PS5's first player's blue.
const DEFAULT_RGB: [u8; 3] = [0, 0, 128];

/// The CRC-32 (IEEE, as zlib's) of the parts in turn: the Bluetooth reports' check.
pub fn crc32(parts: &[&[u8]]) -> u32 {
    let mut c = 0xFFFF_FFFFu32;
    for p in parts {
        for &b in *p {
            c ^= b as u32;
            for _ in 0..8 {
                c = if c & 1 != 0 { (c >> 1) ^ 0xEDB8_8320 } else { c >> 1 };
            }
        }
    }
    !c
}

/// A Bluetooth report's check in its last 4 bytes of the first 78: the CRC-32 of the output
/// header 0xA2 and the 74 bytes before it.
fn bt_check(r: &mut [u8]) {
    let crc = crc32(&[&[0xA2], &r[..74]]);
    r[74..78].copy_from_slice(&crc.to_le_bytes());
}

/// The DualSense's output report: its common part (47 bytes) after the report id (USB 0x02) or
/// after the id, the sequence number and the tag 0x10 (Bluetooth 0x31, a CRC at bytes 74-77).
/// `setup`: the first report, which only ends the light bar's start-up animation so that its
/// colour can be set. `v2`: the vibration of firmware 2.21 and later. `len`: Windows' report
/// length (the device's longest output report; the rest stays zero).
pub fn dualsense_output(bluetooth: bool, seq: u8, o: &Out, v2: bool, setup: bool, len: usize) -> Vec<u8> {
    let mut c = [0u8; 47];
    if setup {
        c[38] = 0x02; // the light bar's setup
        c[41] = 0x02; // light out
    } else {
        c[0] = 0x02 | if v2 { 0 } else { 0x01 }; // the motors in place of the haptics; v1 vibration
        c[1] = 0x04 | 0x10; // the light bar, the player lights
        c[2] = o.right;
        c[3] = o.left;
        c[38] = if v2 { 0x04 } else { 0 }; // v2 vibration
        c[43] = 0x04; // the first player: the middle light
        c[44..47].copy_from_slice(&o.rgb);
    }
    let mut r = vec![0u8; len.max(if bluetooth { 78 } else { 48 })];
    if bluetooth {
        r[0] = 0x31;
        r[1] = seq << 4;
        r[2] = 0x10;
        r[3..50].copy_from_slice(&c);
        bt_check(&mut r);
    } else {
        r[0] = 0x02;
        r[1..48].copy_from_slice(&c);
    }
    r
}

/// The DUALSHOCK 4's output report: 0x05 over USB, 0x11 over Bluetooth (HID and CRC flags and a
/// 4 ms report interval, then audio control; a CRC at bytes 74-77).
pub fn dualshock4_output(bluetooth: bool, o: &Out, len: usize) -> Vec<u8> {
    let mut c = [0u8; 10];
    c[0] = 0x01 | 0x02; // the motors, the light bar
    c[3] = o.right;
    c[4] = o.left;
    c[5..8].copy_from_slice(&o.rgb);
    let mut r = vec![0u8; len.max(if bluetooth { 78 } else { 32 })];
    if bluetooth {
        r[0] = 0x11;
        r[1] = 0xC0 | 0x04;
        r[3..13].copy_from_slice(&c);
        bt_check(&mut r);
    } else {
        r[0] = 0x05;
        r[1..11].copy_from_slice(&c);
    }
    r
}

// ---------------------------------------------------------------------------
// The controllers

struct H(HANDLE);
unsafe impl Send for H {}
unsafe impl Sync for H {}

struct Device {
    path: Vec<u16>,
    model: Model,
    name: &'static str,
    bluetooth: bool,
    read: H,
    /// A synchronous handle for the output reports (None: the controller refused writing).
    write: Option<H>,
    in_len: usize,
    out_len: usize,
    /// The DualSense's vibration of firmware 2.21 and later.
    v2: bool,
    alive: AtomicBool,
    /// The state, its report count and whether a state report came.
    state: Mutex<(Gamepad, u32, bool)>,
    out: Mutex<(Out, bool, bool)>, // what to show, changed, stop
    out_cv: Condvar,
    threads: Mutex<Vec<std::thread::JoinHandle<()>>>,
}

fn link(d: &Device) -> &'static str {
    if d.bluetooth { "Bluetooth" } else { "USB" }
}

static DEVS: Mutex<Vec<Arc<Device>>> = Mutex::new(Vec::new());
/// HID devices that are not PlayStation controllers (not opened again).
static OTHER: Mutex<Option<HashSet<Vec<u16>>>> = Mutex::new(None);
static WATCHING: AtomicBool = AtomicBool::new(false);
/// What every controller is asked to show (`set_motors`, `set_lightbar`).
static WANT: Mutex<Out> = Mutex::new(Out { left: 0, right: 0, rgb: DEFAULT_RGB });
/// Controllers ever connected (the log's numbering).
static COUNT: AtomicU64 = AtomicU64::new(0);
/// The game's end (`motors_off`): the motors stay off, whatever a writer had pending.
static STOPPED: AtomicBool = AtomicBool::new(false);
/// One output report at a time on any controller, so that `motors_off`'s is the last.
static WRITING: Mutex<()> = Mutex::new(());

fn read_reports(d: Arc<Device>) {
    let mut buf = vec![0u8; d.in_len.max(64)];
    unsafe {
        let ev = CreateEventW(std::ptr::null(), 1, 0, std::ptr::null());
        // (Without an event nothing is read: the device ends as an unplugged one, its writer
        // told to stop below.)
        if ev.is_null() {
            crate::warn!("controller: {} ({}): no event for its reads (error {})", d.name, link(&d), GetLastError());
        }
        let mut have = false;
        while !ev.is_null() {
            let mut ov: OVERLAPPED = std::mem::zeroed();
            ov.hEvent = ev;
            ResetEvent(ev);
            let mut got = 0u32;
            if ReadFile(d.read.0, buf.as_mut_ptr(), buf.len() as u32, std::ptr::null_mut(), &mut ov) == 0 && GetLastError() != ERROR_IO_PENDING {
                break;
            }
            if WaitForSingleObject(ev, INFINITE) != WAIT_OBJECT_0 || GetOverlappedResult(d.read.0, &ov, &mut got, 0) == 0 {
                break; // unplugged or switched off
            }
            let Some(p) = parse_report(d.model, d.bluetooth, &buf[..got as usize]) else { continue };
            if !have {
                crate::info!("controller: {} ({}): input report {:#04x}", d.name, link(&d), buf[0]);
                have = true;
            }
            let mut s = d.state.lock();
            if !s.2 || s.0 != p {
                s.1 = s.1.wrapping_add(1);
            }
            s.0 = p;
            s.2 = true;
        }
        if !ev.is_null() {
            CloseHandle(ev);
        }
    }
    d.alive.store(false, Relaxed);
    let mut o = d.out.lock();
    o.2 = true;
    d.out_cv.notify_all();
}

/// Writes the output reports: the light bar's setup first (the DualSense), then the latest state
/// whenever it changes, 8 ms apart at least.
fn write_reports(d: Arc<Device>) {
    let Some(w) = d.write.as_ref() else { return };
    let mut seq = 0u8;
    let mut first = true;
    loop {
        let o = {
            let mut g = d.out.lock();
            while !g.1 && !g.2 && !first {
                d.out_cv.wait(&mut g);
            }
            if g.2 {
                return;
            }
            g.1 = false;
            g.0
        };
        // (The DualSense's first report sets its light bar up.)
        let kinds: &[bool] = match d.model {
            Model::DualShock4 => &[false],
            _ if first => &[true, false],
            _ => &[false],
        };
        first = false;
        for &setup in kinds {
            {
                // (Under the write lock, the motors off once the game has ended: `motors_off`'s
                // report stays the last that sets them.)
                let _w = WRITING.lock();
                let mut o = o;
                if STOPPED.load(Relaxed) {
                    o.left = 0;
                    o.right = 0;
                }
                let r = match d.model {
                    Model::DualShock4 => dualshock4_output(d.bluetooth, &o, d.out_len),
                    _ => {
                        let r = dualsense_output(d.bluetooth, seq, &o, d.v2, setup, d.out_len);
                        seq = (seq + 1) & 15;
                        r
                    }
                };
                let mut done = 0u32;
                if unsafe { WriteFile(w.0, r.as_ptr(), r.len() as u32, &mut done, std::ptr::null_mut()) } == 0 {
                    crate::warn!("controller: {} ({}): an output report was refused (error {})", d.name, link(&d), unsafe { GetLastError() });
                    return;
                }
            }
            std::thread::sleep(Duration::from_millis(8));
        }
    }
}

/// Opens a HID device that is a PlayStation controller this reads: Ok(None) when it cannot be
/// opened now (asked again later), Err(()) when it is another device (not asked again).
fn open_device(path: &[u16]) -> Result<Option<Device>, ()> {
    unsafe {
        // No access rights: enough for the vendor and product ids, and opens any HID device.
        let probe = CreateFileW(path.as_ptr(), 0, FILE_SHARE_READ | FILE_SHARE_WRITE, std::ptr::null(), OPEN_EXISTING, 0, std::ptr::null_mut());
        if probe == INVALID_HANDLE_VALUE {
            return Ok(None);
        }
        let mut a: HIDD_ATTRIBUTES = std::mem::zeroed();
        a.Size = std::mem::size_of::<HIDD_ATTRIBUTES>() as u32;
        let got = HidD_GetAttributes(probe, &mut a);
        CloseHandle(probe);
        if !got {
            return Err(());
        }
        let Some(&(_, model, name)) = KNOWN.iter().find(|k| a.VendorID == SONY && a.ProductID == k.0) else {
            note_other(path, a.VendorID, a.ProductID);
            return Err(());
        };
        let read = CreateFileW(path.as_ptr(), GENERIC_READ, FILE_SHARE_READ | FILE_SHARE_WRITE, std::ptr::null(), OPEN_EXISTING, FILE_FLAG_OVERLAPPED, std::ptr::null_mut());
        if read == INVALID_HANDLE_VALUE {
            // Another program holds it alone (DS4Windows with HidHide, for one): asked again
            // later; said once per controller.
            static SAID: Mutex<Vec<Vec<u16>>> = Mutex::new(Vec::new());
            let mut said = SAID.lock();
            if !said.iter().any(|p| p == path) {
                said.push(path.to_vec());
                crate::warn!("controller: a {name} is connected but another program holds it alone (error {}): its input comes through that program if at all", GetLastError());
            }
            return Ok(None);
        }
        let mut pp = 0isize;
        let mut caps: HIDP_CAPS = std::mem::zeroed();
        let capsok = HidD_GetPreparsedData(read, &mut pp) && HidP_GetCaps(pp, &mut caps) == HIDP_STATUS_SUCCESS;
        if pp != 0 {
            HidD_FreePreparsedData(pp);
        }
        if !capsok || caps.UsagePage != 0x01 || (caps.Usage != 0x04 && caps.Usage != 0x05) {
            CloseHandle(read); // not its gamepad collection
            return Err(());
        }
        let write = CreateFileW(path.as_ptr(), GENERIC_WRITE, FILE_SHARE_READ | FILE_SHARE_WRITE, std::ptr::null(), OPEN_EXISTING, 0, std::ptr::null_mut());
        let write = if write == INVALID_HANDLE_VALUE { None } else { Some(H(write)) };
        let text = String::from_utf16_lossy(path).to_ascii_lowercase();
        let bluetooth = text.contains(BT_SERVICE) || caps.InputReportByteLength > 64;
        // The DualSense's firmware (feature report 0x20, its update version at byte 44): the
        // vibration of 2.21 and later needs its own flag; the Edge has it always, and an
        // unread version is taken as current.
        let mut v2 = true;
        if model == Model::DualSense {
            let mut f = vec![0u8; (caps.FeatureReportByteLength as usize).max(64)];
            f[0] = 0x20;
            if HidD_GetFeature(read, f.as_mut_ptr().cast(), f.len() as u32) {
                let version = u16::from_le_bytes([f[44], f[45]]);
                v2 = version >= 0x0215;
                crate::info!("controller: DualSense firmware update version {version:#06x} (vibration {})", if v2 { "v2" } else { "v1" });
            }
        }
        Ok(Some(Device {
            path: path.to_vec(),
            model,
            name,
            bluetooth,
            read: H(read),
            write,
            in_len: caps.InputReportByteLength as usize,
            out_len: caps.OutputReportByteLength as usize,
            v2,
            alive: AtomicBool::new(true),
            state: Mutex::new((Gamepad::default(), 0, false)),
            out: Mutex::new((*WANT.lock(), true, false)),
            out_cv: Condvar::new(),
            threads: Mutex::new(Vec::new()),
        }))
    }
}

/// Game controllers found that are not read here (`note_other`), for the summary.
static OTHER_PADS: Mutex<Vec<String>> = Mutex::new(Vec::new());

/// A HID device that is no PlayStation controller this port knows: when it is a game controller
/// (a joystick or gamepad collection), said once with its ids and name and how it is (not) read,
/// so a player's log tells an unsupported controller from none at all.
fn note_other(path: &[u16], vid: u16, pid: u16) {
    unsafe {
        let h = CreateFileW(path.as_ptr(), 0, FILE_SHARE_READ | FILE_SHARE_WRITE, std::ptr::null(), OPEN_EXISTING, 0, std::ptr::null_mut());
        if h == INVALID_HANDLE_VALUE {
            return;
        }
        let mut pp = 0isize;
        let mut caps: HIDP_CAPS = std::mem::zeroed();
        let ok = HidD_GetPreparsedData(h, &mut pp) && HidP_GetCaps(pp, &mut caps) == HIDP_STATUS_SUCCESS;
        if pp != 0 {
            HidD_FreePreparsedData(pp);
        }
        if !ok || caps.UsagePage != 0x01 || (caps.Usage != 0x04 && caps.Usage != 0x05) {
            CloseHandle(h);
            return;
        }
        let mut buf = [0u16; 128];
        let named = HidD_GetProductString(h, buf.as_mut_ptr().cast(), (buf.len() * 2) as u32);
        CloseHandle(h);
        let product = if named { String::from_utf16_lossy(&buf[..buf.iter().position(|&c| c == 0).unwrap_or(buf.len())]) } else { String::new() };
        let what = if product.trim().is_empty() { format!("{vid:04x}:{pid:04x}") } else { format!("{vid:04x}:{pid:04x} '{}'", product.trim()) };
        let how = match vid {
            0x045E => "an Xbox controller, read through XInput",
            SONY => "a Sony controller this port does not know, not read",
            _ => "not read: Xbox and PlayStation controllers are; Steam Input can make others an Xbox controller for the games Steam starts",
        };
        crate::info!("controller: game controller {what} found: {how}");
        OTHER_PADS.lock().push(what);
    }
}

/// The running programs among `names` (executable names, compared without case).
fn running(names: &[&str]) -> Vec<String> {
    use windows_sys::Win32::System::Diagnostics::ToolHelp::{CreateToolhelp32Snapshot, Process32FirstW, Process32NextW, PROCESSENTRY32W, TH32CS_SNAPPROCESS};
    let mut out: Vec<String> = Vec::new();
    unsafe {
        let snap = CreateToolhelp32Snapshot(TH32CS_SNAPPROCESS, 0);
        if snap == INVALID_HANDLE_VALUE {
            return out;
        }
        let mut e: PROCESSENTRY32W = std::mem::zeroed();
        e.dwSize = std::mem::size_of::<PROCESSENTRY32W>() as u32;
        let mut ok = Process32FirstW(snap, &mut e) != 0;
        while ok {
            let n = e.szExeFile.iter().position(|&c| c == 0).unwrap_or(e.szExeFile.len());
            let exe = String::from_utf16_lossy(&e.szExeFile[..n]);
            if names.iter().any(|w| w.eq_ignore_ascii_case(&exe)) && !out.iter().any(|o| o.eq_ignore_ascii_case(&exe)) {
                out.push(exe);
            }
            ok = Process32NextW(snap, &mut e) != 0;
        }
        CloseHandle(snap);
    }
    out
}

/// Once, a minute into the session: when no controller was seen at all (no XInput controller
/// connected, no PlayStation controller found), what Windows offered instead and the running
/// programs that commonly hide or translate controllers (a player's report of a controller that
/// "stopped working", 2026-11-24, whose log said nothing about it).
fn summarize_none_seen() {
    if crate::input::SYSTEM_OFF.load(Relaxed) || crate::input::XINPUT_SEEN.load(Relaxed) || COUNT.load(Relaxed) > 0 {
        return;
    }
    let hid = hid_paths().len();
    let others = OTHER_PADS.lock().clone();
    let tools = running(&["DS4Windows.exe", "DSX.exe", "DualSenseX.exe", "x360ce.exe", "Steam.exe"]);
    crate::info!(
        "controller: none seen in the first minute: no XInput controller in its 4 slots, no PlayStation controller among {hid} HID devices; other game controllers: {}; programs that hide or translate controllers running: {}",
        if others.is_empty() { "none".to_string() } else { others.join(", ") },
        if tools.is_empty() { "none".to_string() } else { tools.join(", ") }
    );
}

/// The HID devices present now (their paths, NUL-terminated).
fn hid_paths() -> Vec<Vec<u16>> {
    let mut out = Vec::new();
    unsafe {
        let mut hid: windows_sys::core::GUID = std::mem::zeroed();
        HidD_GetHidGuid(&mut hid);
        let set = SetupDiGetClassDevsW(&hid, std::ptr::null(), std::ptr::null_mut(), DIGCF_PRESENT | DIGCF_DEVICEINTERFACE);
        if set == -1 {
            return out;
        }
        for i in 0.. {
            let mut ifd: SP_DEVICE_INTERFACE_DATA = std::mem::zeroed();
            ifd.cbSize = std::mem::size_of::<SP_DEVICE_INTERFACE_DATA>() as u32;
            if SetupDiEnumDeviceInterfaces(set, std::ptr::null(), &hid, i, &mut ifd) == 0 {
                break;
            }
            let mut need = 0u32;
            SetupDiGetDeviceInterfaceDetailW(set, &ifd, std::ptr::null_mut(), 0, &mut need, std::ptr::null_mut());
            if (need as usize) < std::mem::size_of::<SP_DEVICE_INTERFACE_DETAIL_DATA_W>() {
                continue;
            }
            // (u32 words: the detail's alignment.)
            let mut buf = vec![0u32; (need as usize).div_ceil(4)];
            let det = buf.as_mut_ptr() as *mut SP_DEVICE_INTERFACE_DETAIL_DATA_W;
            (*det).cbSize = std::mem::size_of::<SP_DEVICE_INTERFACE_DETAIL_DATA_W>() as u32;
            if SetupDiGetDeviceInterfaceDetailW(set, &ifd, det, need, std::ptr::null_mut(), std::ptr::null_mut()) != 0 {
                let p = std::ptr::addr_of!((*det).DevicePath) as *const u16;
                let max = (need as usize - 4) / 2;
                let n = (0..max).take_while(|&k| *p.add(k) != 0).count();
                let mut path = std::slice::from_raw_parts(p, n).to_vec();
                path.push(0);
                out.push(path);
            }
        }
        SetupDiDestroyDeviceInfoList(set);
    }
    out
}

/// One look for controllers connected and gone.
fn scan() {
    let paths = hid_paths();
    let gone: Vec<Arc<Device>> = {
        let mut devs = DEVS.lock();
        let (live, dead): (Vec<_>, Vec<_>) = devs.drain(..).partition(|d| d.alive.load(Relaxed));
        *devs = live;
        dead
    };
    for d in gone {
        for t in d.threads.lock().drain(..) {
            let _ = t.join();
        }
        unsafe {
            CloseHandle(d.read.0);
            if let Some(w) = &d.write {
                CloseHandle(w.0);
            }
        }
        crate::info!("controller: {} ({}) disconnected", d.name, link(&d));
    }
    for p in paths {
        if OTHER.lock().as_ref().is_some_and(|o| o.contains(&p)) || DEVS.lock().iter().any(|d| d.path == p) {
            continue;
        }
        match open_device(&p) {
            Err(()) => {
                OTHER.lock().get_or_insert_with(HashSet::new).insert(p);
            }
            Ok(None) => {}
            Ok(Some(d)) => {
                let d = Arc::new(d);
                let n = COUNT.fetch_add(1, Relaxed) + 1;
                crate::info!(
                    "controller: PlayStation controller {n} connected: {} over {} (input reports of {} bytes, output {}{})",
                    d.name,
                    link(&d),
                    d.in_len,
                    d.out_len,
                    if d.write.is_some() { "" } else { "; it refused writing: no vibration or light bar" }
                );
                let (r, w) = (d.clone(), d.clone());
                let mut threads = d.threads.lock();
                if let Ok(t) = std::thread::Builder::new().name("pad-reader".into()).spawn(move || read_reports(r)) {
                    threads.push(t);
                }
                if d.write.is_some()
                    && let Ok(t) = std::thread::Builder::new().name("pad-writer".into()).spawn(move || write_reports(w))
                {
                    threads.push(t);
                }
                drop(threads);
                DEVS.lock().push(d);
            }
        }
    }
}

fn start_watcher() {
    if WATCHING.swap(true, Relaxed) {
        return;
    }
    crate::info!("controller: looking for PlayStation controllers (Windows' HID, USB and Bluetooth) every 1.5 s");
    let _ = std::thread::Builder::new().name("pad-watcher".into()).spawn(|| {
        let started = std::time::Instant::now();
        let mut summarized = false;
        loop {
            scan();
            if !summarized && started.elapsed() >= Duration::from_secs(60) {
                summarized = true;
                summarize_none_seen();
            }
            std::thread::sleep(Duration::from_millis(1500));
        }
    });
}

/// A PlayStation controller now: its state, its report count, and whether it counts (its state
/// changed since it was found: a device that only ever reports one state does not drive the game).
pub struct State {
    pub pad: Gamepad,
    pub packet: u32,
    pub active: bool,
}

/// The PlayStation controllers connected now that have reported (the watcher starts with the
/// first call).
pub fn states() -> Vec<State> {
    start_watcher();
    DEVS.lock()
        .iter()
        .filter(|d| d.alive.load(Relaxed))
        .filter_map(|d| {
            let s = d.state.lock();
            s.2.then_some(State { pad: s.0, packet: s.1, active: s.1 >= 2 })
        })
        .collect()
}

fn publish(f: impl Fn(&mut Out)) {
    let want = {
        let mut w = WANT.lock();
        f(&mut w);
        if STOPPED.load(Relaxed) {
            w.left = 0;
            w.right = 0;
        }
        *w
    };
    for d in DEVS.lock().iter() {
        let mut o = d.out.lock();
        if o.0 != want {
            o.0 = want;
            o.1 = true;
            d.out_cv.notify_all();
        }
    }
}

/// The motors' strength (0-255; left the strong one) on every PlayStation controller.
pub fn set_motors(left: u8, right: u8) {
    publish(|o| {
        o.left = left;
        o.right = right;
    });
}

/// The light bar's colour on every PlayStation controller (None: the default blue).
pub fn set_lightbar(rgb: Option<[u8; 3]>) {
    publish(|o| o.rgb = rgb.unwrap_or(DEFAULT_RGB));
}

/// The motors stopped at once, written from this thread (the game's end: a controller keeps its
/// last vibration otherwise).
pub fn motors_off() {
    STOPPED.store(true, Relaxed);
    let want = {
        let mut w = WANT.lock();
        w.left = 0;
        w.right = 0;
        *w
    };
    for d in DEVS.lock().iter() {
        let Some(w) = &d.write else { continue };
        let r = match d.model {
            Model::DualShock4 => dualshock4_output(d.bluetooth, &want, d.out_len),
            _ => dualsense_output(d.bluetooth, 0, &want, d.v2, false, d.out_len),
        };
        let mut done = 0u32;
        let _w = WRITING.lock();
        unsafe { WriteFile(w.0, r.as_ptr(), r.len() as u32, &mut done, std::ptr::null_mut()) };
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A DualSense USB report: sticks (lx, ly, rx, ry), L2, R2, then the three button bytes.
    fn ds_usb(sticks: [u8; 4], l2: u8, r2: u8, b: [u8; 3]) -> Vec<u8> {
        let mut r = vec![0u8; 64];
        r[0] = 0x01;
        r[1..5].copy_from_slice(&sticks);
        r[5] = l2;
        r[6] = r2;
        r[7] = 0x5A; // the counter
        r[8..11].copy_from_slice(&b);
        r
    }

    #[test]
    fn dualsense_usb_and_bluetooth() {
        // Cross with the d-pad up, L1, Options, the touch pad's click; Create and PS pressed too.
        let r = ds_usb([0, 255, 128, 128], 200, 10, [0x20, 0x01 | 0x10 | 0x20, 0x01 | 0x02]);
        let p = parse_report(Model::DualSense, false, &r).unwrap();
        assert_eq!(p.buttons, BUTTON_A | DPAD_UP | LEFT_SHOULDER | START | BACK);
        assert_eq!((p.lt, p.rt), (200, 10));
        assert_eq!((p.lx, p.ly), (-32768, -32768));
        assert_eq!((p.rx, p.ry), (128, -129));
        // Over Bluetooth (0x31): the same one byte later.
        let mut bt = vec![0u8; 78];
        bt[0] = 0x31;
        bt[1] = 0x10;
        bt[2..65].copy_from_slice(&r[1..64]);
        assert_eq!(parse_report(Model::DualSense, true, &bt), Some(p));
        assert_eq!(parse_report(Model::DualSenseEdge, true, &bt), Some(p));
        // Before a program writes to it: 0x01 in the DUALSHOCK 4's layout; the d-pad left down.
        let simple = [0x01, 128, 128, 255, 0, 0x80 | 5, 0x02 | 0x80, 0x00, 7, 255];
        let s = parse_report(Model::DualSense, true, &simple).unwrap();
        assert_eq!(s.buttons, BUTTON_Y | DPAD_DOWN | DPAD_LEFT | RIGHT_SHOULDER | RIGHT_THUMB);
        assert_eq!((s.lt, s.rt, s.rx, s.ry), (7, 255, 32767, 32767));
        // Not a state report.
        assert_eq!(parse_report(Model::DualSense, false, &[0x05; 64]), None);
        assert_eq!(parse_report(Model::DualSense, false, &r[..9]), None);
    }

    #[test]
    fn dualshock4_usb_and_bluetooth() {
        // The d-pad none (8), Square and Circle, R1, Share (nothing), the touch pad's click.
        let mut r = vec![0u8; 64];
        r[0] = 0x01;
        r[1..5].copy_from_slice(&[128, 128, 128, 128]);
        r[5] = 0x08 | 0x10 | 0x40;
        r[6] = 0x02 | 0x10;
        r[7] = 0x02;
        r[8] = 1;
        r[9] = 2;
        let p = parse_report(Model::DualShock4, false, &r).unwrap();
        assert_eq!(p.buttons, BUTTON_X | BUTTON_B | RIGHT_SHOULDER | BACK);
        assert_eq!((p.lt, p.rt), (1, 2));
        let mut bt = vec![0u8; 78];
        bt[0] = 0x11;
        bt[3..66].copy_from_slice(&r[1..64]);
        assert_eq!(parse_report(Model::DualShock4, true, &bt), Some(p));
    }

    #[test]
    fn crc32_known_value() {
        assert_eq!(crc32(&[b"123456789"]), 0xCBF4_3926);
        assert_eq!(crc32(&[b"1234", b"56789"]), 0xCBF4_3926);
    }

    #[test]
    fn output_reports() {
        let o = Out { left: 200, right: 30, rgb: [1, 2, 3] };
        let u = dualsense_output(false, 0, &o, true, false, 48);
        assert_eq!(u.len(), 48);
        assert_eq!(&u[..5], &[0x02, 0x02, 0x14, 30, 200]);
        assert_eq!((u[39], u[44]), (0x04, 0x04));
        assert_eq!(&u[45..48], &[1, 2, 3]);
        let v1 = dualsense_output(false, 0, &o, false, false, 48);
        assert_eq!((v1[1], v1[39]), (0x03, 0));
        let setup = dualsense_output(false, 0, &o, true, true, 48);
        assert_eq!((setup[1], setup[3], setup[39], setup[42]), (0, 0, 0x02, 0x02));
        let b = dualsense_output(true, 3, &o, true, false, 547);
        assert_eq!(b.len(), 547);
        assert_eq!(&b[..7], &[0x31, 0x30, 0x10, 0x02, 0x14, 30, 200]);
        assert_eq!(&b[46..50], &[0x04, 1, 2, 3]);
        assert_eq!(u32::from_le_bytes([b[74], b[75], b[76], b[77]]), crc32(&[&[0xA2], &b[..74]]));
        let d = dualshock4_output(false, &o, 32);
        assert_eq!(&d[..9], &[0x05, 0x03, 0, 0, 30, 200, 1, 2, 3]);
        let db = dualshock4_output(true, &o, 78);
        assert_eq!(&db[..11], &[0x11, 0xC4, 0, 0x03, 0, 0, 30, 200, 1, 2, 3]);
        assert_eq!(u32::from_le_bytes([db[74], db[75], db[76], db[77]]), crc32(&[&[0xA2], &db[..74]]));
    }

    #[test]
    fn sticks_round_trip() {
        // The pad layer's scale gives every position back (`input::axis`).
        for v in 0..=255u8 {
            let x = (axis_x(v) as f32 + 32768.0) / 65535.0;
            assert_eq!((x * 255.0).round() as u8, v);
            let y = 1.0 - (axis_y(v) as f32 + 32768.0) / 65535.0;
            assert_eq!((y * 255.0).round() as u8, v);
        }
    }
}
