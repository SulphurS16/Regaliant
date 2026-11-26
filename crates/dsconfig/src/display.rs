//! Monitors and display modes as Windows lists them, and the display choices kept in
//! display.txt (`paths::Paths::display_file`): shared by the runtime (the PC tab's Monitor,
//! Resolution and Refresh Rate, exclusive fullscreen) and the launcher (its Display page, before
//! a start). display.txt: `monitor=<GDI device name>` (empty: the primary monitor),
//! `resolution=<w>x<h>` (empty: the desktop's) and `refresh=<Hz>` (0: the desktop's rate, or the
//! highest one at another resolution).

use std::path::Path;

/// A monitor as Windows lists it now.
#[derive(Clone, Debug, Default)]
pub struct Monitor {
    /// The GDI device name (`\\.\DISPLAY1`).
    pub device: String,
    /// The monitor's own name (its EDID's), else the device name.
    pub name: String,
    /// Its area on the desktop: left, top, right, bottom.
    pub rect: [i32; 4],
    pub primary: bool,
    /// Windows' HDR (advanced color) supported and turned on for it.
    pub hdr_supported: bool,
    pub hdr_on: bool,
    /// The brightness Windows gives SDR content on it in HDR mode, in nits (80 by default).
    pub sdr_white_nits: f32,
}

/// A display mode (32 bits per pixel, not interlaced).
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct Mode {
    pub w: u32,
    pub h: u32,
    pub hz: u32,
}

#[cfg(windows)]
#[path = "display/windows.rs"]
mod host;
#[cfg(target_os = "linux")]
#[path = "display/linux.rs"]
mod host;
pub use host::*;

/// The display choices (display.txt).
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct Choice {
    /// The monitor's GDI device name (empty: the primary).
    pub monitor: String,
    /// The fullscreen resolution (None: the desktop's).
    pub res: Option<(u32, u32)>,
    /// The fullscreen refresh rate in Hz (0: automatic).
    pub hz: u32,
}

/// Reads display.txt (missing or unreadable: the defaults).
pub fn read_choice(path: &Path) -> Choice {
    let mut c = Choice::default();
    for (k, v) in crate::paths::read_kv(path) {
        match k.as_str() {
            "monitor" => c.monitor = v,
            "resolution" => c.res = v.split_once('x').and_then(|(w, h)| Some((w.trim().parse().ok()?, h.trim().parse().ok()?))),
            "refresh" => c.hz = v.parse().unwrap_or(0),
            _ => {}
        }
    }
    c
}

/// Writes display.txt.
pub fn write_choice(path: &Path, c: &Choice) -> std::io::Result<()> {
    let res = c.res.map(|(w, h)| format!("{w}x{h}")).unwrap_or_default();
    std::fs::write(path, format!("# Display choices (the PC tab's Monitor, Resolution and Refresh Rate); written by the port\nmonitor={}\nresolution={res}\nrefresh={}\n", c.monitor, c.hz))
}
