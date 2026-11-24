//! SDL gamepad samples in the shared binding layer's legacy controller format.
#![allow(non_snake_case, non_camel_case_types)]
use parking_lot::Mutex;
use sdl3::gamepad::{Axis, Button};
use std::collections::{HashMap, HashSet};
#[derive(Clone, Copy, Default)]
pub struct XINPUT_GAMEPAD {
    pub wButtons: u16,
    pub bLeftTrigger: u8,
    pub bRightTrigger: u8,
    pub sThumbLX: i16,
    pub sThumbLY: i16,
    pub sThumbRX: i16,
    pub sThumbRY: i16,
}
#[derive(Clone, Copy, Default)]
pub struct XINPUT_STATE {
    pub dwPacketNumber: u32,
    pub Gamepad: XINPUT_GAMEPAD,
}
pub struct XINPUT_VIBRATION {
    pub wLeftMotorSpeed: u16,
    pub wRightMotorSpeed: u16,
}
static STATE: Mutex<Option<XINPUT_STATE>> = Mutex::new(None);
static PROMPT_STYLE: std::sync::atomic::AtomicU8 = std::sync::atomic::AtomicU8::new(2);
pub fn prompt_style() -> u8 { PROMPT_STYLE.load(std::sync::atomic::Ordering::Relaxed) }
pub fn set_type(kind: sdl3::gamepad::GamepadType) {
    use sdl3::gamepad::GamepadType::*;
    PROMPT_STYLE.store(if matches!(kind, PS3 | PS4 | PS5) { 1 } else { 2 }, std::sync::atomic::Ordering::Relaxed);
}
pub fn publish(connected: bool, buttons: &HashSet<Button>, axes: &HashMap<Axis, i16>) {
    if !connected {
        *STATE.lock() = None;
        return;
    }
    let mut p = XINPUT_GAMEPAD::default();
    for (b, bit) in [
        (Button::South, 0x1000),
        (Button::East, 0x2000),
        (Button::West, 0x4000),
        (Button::North, 0x8000),
        (Button::LeftShoulder, 0x100),
        (Button::RightShoulder, 0x200),
        (Button::LeftStick, 0x40),
        (Button::RightStick, 0x80),
        (Button::Back, 0x20),
        (Button::Touchpad, 0x20), // DualSense touchpad click uses the shared View binding.
        (Button::Start, 0x10),
        (Button::DPadUp, 1),
        (Button::DPadDown, 2),
        (Button::DPadLeft, 4),
        (Button::DPadRight, 8),
    ] {
        if buttons.contains(&b) {
            p.wButtons |= bit;
        }
    }
    let a = |a| axes.get(&a).copied().unwrap_or(0);
    p.sThumbLX = a(Axis::LeftX);
    p.sThumbLY = a(Axis::LeftY).saturating_neg();
    p.sThumbRX = a(Axis::RightX);
    p.sThumbRY = a(Axis::RightY).saturating_neg();
    p.bLeftTrigger = (a(Axis::TriggerLeft).max(0) as u32 * 255 / 32767) as u8;
    p.bRightTrigger = (a(Axis::TriggerRight).max(0) as u32 * 255 / 32767) as u8;
    let mut state = STATE.lock();
    let n = state.map_or(0, |s| s.dwPacketNumber).wrapping_add(1);
    *state = Some(XINPUT_STATE {
        dwPacketNumber: n,
        Gamepad: p,
    });
}
pub unsafe fn XInputGetState(index: u32, out: *mut XINPUT_STATE) -> u32 {
    if index != 0 || out.is_null() {
        return 1167;
    }
    match *STATE.lock() {
        Some(s) => {
            unsafe {
                *out = s;
            }
            0
        }
        None => 1167,
    }
}
pub unsafe fn XInputSetState(_index: u32, v: *mut XINPUT_VIBRATION) -> u32 {
    if v.is_null() {
        return 87;
    }
    unsafe {
        crate::platform::frontend::haptics(
            (*v).wLeftMotorSpeed as f32 / 65535.0,
            (*v).wRightMotorSpeed as f32 / 65535.0,
        );
    }
    0
}

// SDL publishes a current sample from its event loop; polling an absent slot is a local
// mutex read, not the potentially slow Windows driver call. Do not debounce hotplug/input.
pub fn poll_disconnected() -> bool { true }
pub fn initial_sample_active() -> bool { true }
