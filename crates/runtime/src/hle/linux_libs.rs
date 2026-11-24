//! Shared library HLE connected to Linux host services, including AGC and video output.

#[path = "libs/acm.rs"]
pub mod acm;
#[path = "libs/acm_conv.rs"]
pub mod acm_conv;
#[path = "libs/acm_fft.rs"]
pub mod acm_fft;
#[path = "libs/agc.rs"]
pub mod agc;
#[path = "libs/agc_cmd.rs"]
pub mod agc_cmd;
#[path = "libs/agc_defaults.rs"]
pub mod agc_defaults;
#[path = "libs/ajm.rs"]
pub mod ajm;
#[path = "libs/ampr.rs"]
pub mod ampr;
#[path = "libs/app.rs"]
pub mod app;
#[path = "libs/audio2.rs"]
pub mod audio2;
#[path = "libs/devices.rs"]
pub mod devices;
#[path = "libs/fiber.rs"]
pub mod fiber;
#[path = "libs/libc_internal.rs"]
pub mod libc_internal;
#[path = "libs/online.rs"]
pub mod online;
#[path = "libs/ps5svc.rs"]
pub mod ps5svc;
#[path = "libs/rtc.rs"]
pub mod rtc;
#[path = "libs/savedata.rs"]
pub mod savedata;
#[path = "libs/sysmodule.rs"]
pub mod sysmodule;

pub fn register(r: &mut super::Registry) {
    sysmodule::register(r);
    libc_internal::register(r);
    ampr::register(r);
    agc::register(r);
    agc_cmd::register(r);
    agcdriver::register(r);
    videoout::register(r);
    ime_dialog::register(r);
    fiber::register(r);
    audio2::register(r);
    ps5svc::register(r);
    app::register(r);
    online::register(r);
    devices::register(r);
    ajm::register(r);
    acm::register(r);
    rtc::register(r);
    savedata::register(r);
}

#[path = "libs/agcdriver.rs"]
pub mod agcdriver;

#[path = "libs/videoout.rs"]
pub mod videoout;

#[path = "libs/ime_dialog.rs"]
pub mod ime_dialog;
