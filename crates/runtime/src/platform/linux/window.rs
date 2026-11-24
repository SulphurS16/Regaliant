//! Main-thread SDL Wayland/X11 window, surface creation and input for the shared renderer.
use ash::vk;
use sdl3::event::{Event, WindowEvent};
use std::collections::{HashMap, HashSet};
use std::sync::atomic::{AtomicBool, AtomicI32, AtomicI64, AtomicU32, Ordering};

#[derive(Default, Debug)]
pub struct Input {
    pub keys: HashSet<sdl3::keyboard::Scancode>,
    pub mouse_buttons: HashSet<sdl3::mouse::MouseButton>,
    pub pointer: (f32, f32),
    pub relative: (f32, f32),
    pub wheel: (f32, f32),
    pub pad_buttons: HashSet<sdl3::gamepad::Button>,
    pub pad_axes: HashMap<sdl3::gamepad::Axis, i16>,
}

struct State {
    hidden: bool,
    focused: bool,
    capture_wanted: bool,
    text_active: bool,
    swallowed: HashSet<sdl3::keyboard::Scancode>,
    pub window: sdl3::video::Window,
    events: sdl3::EventPump,
    pads: HashMap<sdl3::joystick::JoystickId, sdl3::gamepad::Gamepad>,
    gamepads: sdl3::GamepadSubsystem,
    active_pad: Option<sdl3::joystick::JoystickId>,
    _video: sdl3::VideoSubsystem,
    _sdl: sdl3::Sdl,
    pub input: Input,
}

impl State {
    pub fn new(hidden: bool, title: &str, width: u32, height: u32) -> Result<Self, String> {
        let sdl = sdl3::init().map_err(|e| e.to_string())?;
        let video = sdl.video().map_err(|e| e.to_string())?;
        // SDL's default selects the available compositor backend; SDL_VIDEO_DRIVER can force
        // wayland or x11. No protocol code, mouse grab or focus request lives in this backend.
        let mut builder = video.window(title, width, height);
        builder.vulkan().resizable().high_pixel_density();
        if hidden {
            builder.hidden();
        }
        let window = builder.build().map_err(|e| e.to_string())?;
        // Finish fallible SDL initialization before transferring Vulkan ownership to Self.
        let input = (|| {
            let events = sdl.event_pump().map_err(|e| e.to_string())?;
            let gamepads = sdl.gamepad().map_err(|e| e.to_string())?;
            let pads = gamepads
                .gamepads()
                .map_err(|e| e.to_string())?
                .into_iter()
                .filter_map(|id| gamepads.open(id).ok().map(|pad| (id, pad)))
                .collect();
            Ok::<_, String>((events, gamepads, pads))
        })();
        let (events, gamepads, pads) = input?;
        Ok(Self {
            hidden,
            focused: false,
            capture_wanted: false,
            text_active: false,
            swallowed: HashSet::new(),
            window,
            events,
            pads,
            gamepads,
            active_pad: None,
            _video: video,
            _sdl: sdl,
            input: Input::default(),
        })
    }

    /// Pumps on the main thread. Returns false on a window-close request.
    pub fn poll(&mut self) -> bool {
        if self.hidden || crate::input::SYSTEM_OFF.load(Ordering::Relaxed) {
            // Relay windows neither consume physical controls nor send controller rumble.
            return !self
                .events
                .poll_iter()
                .any(|e| matches!(e, Event::Quit { .. } | Event::Window { win_event: WindowEvent::CloseRequested, .. }));
        }
        self.input.relative = (0.0, 0.0);
        self.input.wheel = (0.0, 0.0);
        for event in self.events.poll_iter() {
            match event {
                Event::Quit { .. }
                | Event::Window {
                    win_event: WindowEvent::CloseRequested,
                    ..
                } => return false,
                Event::Window {
                    win_event: WindowEvent::FocusLost,
                    ..
                } => {
                    self.focused = false;
                    self.capture_wanted = false;
                    self.swallowed.clear();
                    self.input.keys.clear();
                    self.input.mouse_buttons.clear();
                }
                Event::Window {
                    win_event: WindowEvent::FocusGained,
                    ..
                } => self.focused = true,
                Event::Window { win_event: WindowEvent::DisplayChanged(_), .. } => {
                    note_display(&self.window);
                    RESIZED.store(true, Ordering::Release);
                }
                Event::Window { win_event: WindowEvent::PixelSizeChanged(_, _), .. } => {
                    RESIZED.store(true, Ordering::Release);
                }
                Event::TextInput { text, .. } => {
                    for c in text.encode_utf16() {
                        crate::hle::libs::ime_dialog::char_typed(c, false, 0);
                    }
                }
                Event::KeyDown {
                    scancode: Some(k),
                    repeat,
                    keymod,
                    ..
                } => {
                    use sdl3::keyboard::{Mod, Scancode as K};
                    if self.swallowed.contains(&k) {
                        continue;
                    }
                    let ctrl = keymod.intersects(Mod::LCTRLMOD | Mod::RCTRLMOD);
                    let alt = keymod.intersects(Mod::LALTMOD | Mod::RALTMOD);
                    let vk = virtual_key(k).unwrap_or(0);
                    if crate::hle::libs::ime_dialog::key_down(vk, repeat, ctrl, k as u16) {
                        if !crate::hle::libs::ime_dialog::active() {
                            self.swallowed.insert(k);
                        }
                        continue;
                    }
                    if !repeat && (k == K::F11 || (k == K::Return && alt)) {
                        set_fullscreen(None);
                        let _ = crate::pcsettings::store::set_by_name(
                            "Setting_PC_WindowMode",
                            &REQUESTED_MODE.load(Ordering::Relaxed).to_string(),
                        );
                        self.swallowed.insert(k);
                        continue;
                    }
                    if !repeat && k == K::F1 {
                        self.capture_wanted = !self.capture_wanted;
                    }
                    if !repeat && k == K::F10 {
                        crate::hle::libs::videoout::burst_hotkey();
                        continue;
                    }
                    let general = match vk {
                        0xa0 | 0xa1 => 0x10,
                        0xa2 | 0xa3 => 0x11,
                        0xa4 | 0xa5 => 0x12,
                        v => v,
                    };
                    if !repeat
                        && crate::pcsettings::bindings::key_down(
                            general,
                            (general != vk).then_some(vk),
                        )
                    {
                        self.swallowed.insert(k);
                        continue;
                    }
                    self.input.keys.insert(k);
                    crate::input::note_device(crate::input::DEVICE_KBM);
                }
                Event::KeyUp {
                    scancode: Some(k), ..
                } => {
                    self.swallowed.remove(&k);
                    self.input.keys.remove(&k);
                    if let Some(vk) = virtual_key(k) {
                        let general = match vk {
                            0xa0 | 0xa1 => 0x10,
                            0xa2 | 0xa3 => 0x11,
                            0xa4 | 0xa5 => 0x12,
                            v => v,
                        };
                        crate::pcsettings::bindings::key_up(general, (general != vk).then_some(vk));
                    }
                }
                Event::MouseMotion {
                    x, y, xrel, yrel, ..
                } => {
                    self.input.pointer = (x, y);
                    self.input.relative.0 += xrel;
                    self.input.relative.1 += yrel;
                    if !CAPTURED.load(Ordering::Relaxed) {
                        use crate::pcsettings::pointer::{CLIENT_X, CLIENT_Y, MOVES};
                        let (px, py) = pointer_pixels_at((x, y), self.window.size(), self.window.size_in_pixels());
                        CLIENT_X.store(px, Ordering::Relaxed);
                        CLIENT_Y.store(py, Ordering::Relaxed);
                        MOVES.fetch_add(1, Ordering::Relaxed);
                        crate::input::note_device(crate::input::DEVICE_KBM);
                    }
                }
                Event::MouseButtonDown { mouse_btn, .. } => {
                    use sdl3::mouse::MouseButton as B;
                    let index = match mouse_btn {
                        B::Left => Some(0),
                        B::Right => Some(1),
                        B::Middle => Some(2),
                        B::X1 => Some(3),
                        B::X2 => Some(4),
                        _ => None,
                    };
                    if crate::pcsettings::bindings::mouse_down(index, 0) {
                        continue;
                    }
                    if crate::pcsettings::pointer::ACTIVE.load(Ordering::Relaxed) {
                        use crate::pcsettings::pointer::{Event as P, event};
                        if mouse_btn == B::Left {
                            event(P::Left);
                        }
                        if mouse_btn == B::Right {
                            event(P::Right);
                        }
                    } else if !CAPTURED.load(Ordering::Relaxed) {
                        self.capture_wanted = true;
                    } else {
                        self.input.mouse_buttons.insert(mouse_btn);
                    }
                }
                Event::MouseButtonUp { mouse_btn, .. } => {
                    self.input.mouse_buttons.remove(&mouse_btn);
                }
                Event::MouseWheel { x, y, .. } => {
                    if crate::pcsettings::bindings::mouse_down(None, (y * 120.0) as i32) {
                        continue;
                    }
                    if crate::pcsettings::pointer::ACTIVE.load(Ordering::Relaxed) {
                        crate::pcsettings::pointer::event(
                            crate::pcsettings::pointer::Event::Wheel(y as i32),
                        );
                    } else {
                        self.input.wheel.0 += x;
                        self.input.wheel.1 += y;
                    }
                }
                Event::GamepadAdded { which, .. } => {
                    if let Ok(pad) = self.gamepads.open(which) {
                        self.pads.insert(which, pad);
                    }
                }
                Event::GamepadRemoved { which, .. } => {
                    self.pads.remove(&which);
                    self.input.pad_axes.clear();
                    self.input.pad_buttons.clear();
                }
                Event::GamepadButtonDown { button, .. } => {
                    self.input.pad_buttons.insert(button);
                }
                Event::GamepadButtonUp { button, .. } => {
                    self.input.pad_buttons.remove(&button);
                }
                Event::GamepadAxisMotion { axis, value, .. } => {
                    self.input.pad_axes.insert(axis, value);
                }
                _ => {}
            }
        }
        let active = crate::hle::libs::ime_dialog::active();
        if active != self.text_active {
            if active {
                self._video.text_input().start(&self.window);
            } else {
                self._video.text_input().stop(&self.window);
            }
            self.text_active = active;
        }
        let capture = self.focused
            && self.capture_wanted
            && !active
            && !crate::pcsettings::pointer::ACTIVE.load(Ordering::Relaxed);
        if CAPTURED.swap(capture, Ordering::Relaxed) != capture {
            self._sdl
                .mouse()
                .set_relative_mouse_mode(&self.window, capture);
        }
        if !capture {
            self.input.relative = (0.0, 0.0);
            self.input.mouse_buttons.clear();
        }
        // SDL's standardized positional mapping works for Xbox and DualSense. Sample
        // one pad, never merge one controller's buttons with another's axes.
        let selected = self.pads.keys().min().copied();
        self.input.pad_buttons.clear();
        self.input.pad_axes.clear();
        if let Some(pad) = selected.and_then(|id| self.pads.get(&id)) {
            use sdl3::gamepad::{Axis::*, Button::*};
            crate::platform::controller::set_type(pad.r#type());
            if self.active_pad != selected { crate::info!("SDL active gamepad: {:?}, type {:?}, prompt style {}; mapping {:?}", pad.name(), pad.r#type(), crate::platform::controller::prompt_style(), pad.mapping()); }
            for b in [South, East, West, North, LeftShoulder, RightShoulder, LeftStick, RightStick, Back, Touchpad, Start, DPadUp, DPadDown, DPadLeft, DPadRight] {
                if pad.button(b) { self.input.pad_buttons.insert(b); }
            }
            for a in [LeftX, LeftY, RightX, RightY, TriggerLeft, TriggerRight] { self.input.pad_axes.insert(a, pad.axis(a)); }
        }
        self.active_pad = selected;
        publish(&self.input, selected.is_some());
        if let Some([r, g, b]) = crate::platform::frontend::take_lightbar() {
            if let Some(pad) = selected.and_then(|id| self.pads.get_mut(&id)) { let _ = pad.set_led(r, g, b); }
        }
        let rumble = crate::platform::frontend::rumble();
        if rumble != 0 {
            if let Some(pad) = selected.and_then(|id| self.pads.get_mut(&id)) {
                let _ = pad.set_rumble(rumble as u16, (rumble >> 16) as u16, 100);
            }
        }
        true
    }
}

use std::cell::RefCell;
use std::sync::OnceLock;
thread_local! { static STATE: RefCell<Option<State>> = const { RefCell::new(None) }; }
pub struct Window {
    pub hwnd: isize,
    pub hinstance: isize,
}
static WINDOW: OnceLock<Window> = OnceLock::new();
pub static CLOSED: AtomicBool = AtomicBool::new(false);
pub static RESIZED: AtomicBool = AtomicBool::new(false);
pub static CLIENT_W: AtomicU32 = AtomicU32::new(1280);
pub static CLIENT_H: AtomicU32 = AtomicU32::new(720);
pub static KEYS: [AtomicBool; 256] = [const { AtomicBool::new(false) }; 256];
pub static MOUSE_DX: AtomicI64 = AtomicI64::new(0);
pub static MOUSE_DY: AtomicI64 = AtomicI64::new(0);
pub static MOUSE_WHEEL: AtomicI32 = AtomicI32::new(0);
pub static MOUSE_BUTTONS: AtomicU32 = AtomicU32::new(0);
pub static CAPTURED: AtomicBool = AtomicBool::new(false);
pub static CAPTURE_TEST: AtomicBool = AtomicBool::new(false);
pub static FULLSCREEN: AtomicBool = AtomicBool::new(false);
pub fn create(title: &str, width: u32, height: u32) -> Result<&'static Window, String> {
    if let Some(w) = WINDOW.get() {
        return Ok(w);
    }
    STATE.with(|cell| {
        let hidden = std::env::var("DS_LINUX_HIDDEN").ok().as_deref() == Some("1");
        let (width, height) = std::env::var("DS_LINUX_WINDOW_SIZE")
            .ok()
            .and_then(|s| {
                let (w, h) = s.split_once('x')?;
                Some((
                    w.parse::<u32>().ok()?.clamp(320, 7680),
                    h.parse::<u32>().ok()?.clamp(180, 4320),
                ))
            })
            .unwrap_or((width, height));
        let state = State::new(hidden, title, width, height)?;
        note_display(&state.window);
        let (w, h) = state.window.size_in_pixels();
        crate::info!("SDL drawable: logical {:?}, pixels {w}x{h} (high pixel density enabled)", state.window.size());
        CLIENT_W.store(w, Ordering::Relaxed);
        CLIENT_H.store(h, Ordering::Relaxed);
        let hwnd = state.window.raw() as isize;
        let _ = WINDOW.set(Window { hwnd, hinstance: 0 });
        *cell.borrow_mut() = Some(state);
        crate::info!(
            "SDL renderer window created (hidden={hidden}); no focus or pointer-grab request"
        );
        Ok(WINDOW.get().unwrap())
    })
}
pub fn create_hidden() -> Result<(isize, isize), String> {
    if std::env::var("DS_LINUX_HIDDEN").ok().as_deref() != Some("1") {
        return Err("hidden helper requires the main-thread hidden SDL window".into());
    }
    let w = create("Regaliant", 1280, 720)?;
    Ok((w.hwnd, w.hinstance))
}
pub fn instance_extensions() -> Result<Vec<std::ffi::CString>, String> {
    create("Regaliant", 1280, 720)?;
    STATE.with(|s| {
        s.borrow()
            .as_ref()
            .ok_or("SDL must be initialized on the main thread".to_string())?
            .window
            .vulkan_instance_extensions()
            .map(|v| {
                v.iter()
                    .map(|s| std::ffi::CString::new(s.as_str()).unwrap())
                    .collect()
            })
            .map_err(|e| e.to_string())
    })
}
pub unsafe fn surface(
    c: &crate::gpu::vk::Ctx,
    _hwnd: isize,
    _hinstance: isize,
) -> Result<vk::SurfaceKHR, String> {
    STATE.with(|s| {
        unsafe {
            s.borrow()
                .as_ref()
                .ok_or("SDL surface must be created on the main thread".to_string())?
                .window
                .vulkan_create_surface(c.instance.handle())
        }
        .map_err(|e| e.to_string())
    })
}
fn note_display(window: &sdl3::video::Window) {
    use sdl3::sys::video::*;
    let mode = unsafe { SDL_GetCurrentDisplayMode(SDL_GetDisplayForWindow(window.raw())) };
    if let Some(mode) = unsafe { mode.as_ref() } {
        if mode.refresh_rate > 1.0 {
            crate::displayclock::MODE_MILLIHZ.store((mode.refresh_rate as f64 * 1000.0).round() as u32, Ordering::Relaxed);
            crate::info!("SDL display: {}x{} {:.3} Hz (nominal, not measured vblank)", mode.w, mode.h, mode.refresh_rate);
        }
    }
}

/// Relay-only synthetic window close, delivered through the same SDL event path as the WM.
pub fn request_close() -> Result<(), String> {
    STATE.with(|cell| {
        let state = cell.borrow();
        let s = state.as_ref().ok_or("no SDL window")?;
        s._sdl.event().map_err(|e| e.to_string())?.push_event(Event::Window { timestamp: 0, window_id: s.window.id(), win_event: WindowEvent::CloseRequested }).map_err(|e| e.to_string())
    })
}

pub fn poll() -> bool {
    STATE.with(|s| {
        let mut s = s.borrow_mut();
        let Some(s) = s.as_mut() else { return false };
        let requested = REQUESTED_MODE.load(Ordering::Relaxed);
        if !s.hidden && requested != APPLIED_MODE.load(Ordering::Relaxed) {
            match s.window.set_fullscreen(requested != 0) {
                Ok(()) => {
                    APPLIED_MODE.store(requested, Ordering::Relaxed);
                    FULLSCREEN.store(requested != 0, Ordering::Relaxed);
                    RESIZED.store(true, Ordering::Release);
                    crate::info!("SDL window mode {requested} applied");
                }
                Err(e) => {
                    crate::warn!("SDL fullscreen: {e}");
                    REQUESTED_MODE.store(APPLIED_MODE.load(Ordering::Relaxed), Ordering::Relaxed);
                }
            }
        }
        let open = s.poll();
        // Query after pumping: SDL applies fractional scale/pixel-size changes here, including
        // display-scale events not exposed by this version of the Rust event wrapper.
        let (w, h) = s.window.size_in_pixels();
        let width_changed = CLIENT_W.swap(w, Ordering::Relaxed) != w;
        let height_changed = CLIENT_H.swap(h, Ordering::Relaxed) != h;
        if width_changed || height_changed {
            RESIZED.store(true, Ordering::Release);
        }
        if width_changed || height_changed {
            crate::info!("SDL drawable: logical {:?}, pixels {w}x{h}", s.window.size());
        }
        if !open {
            CLOSED.store(true, Ordering::Release);
        }
        open
    })
}
pub fn get() -> Option<&'static Window> {
    WINDOW.get()
}
pub fn key_down(k: u8) -> bool {
    KEYS[k as usize].load(Ordering::Relaxed)
}
pub fn pointer_mode_changed() {}
pub fn pointer_pixels(n: usize) -> Vec<u8> {
    vec![0; n * n * 4]
}
static REQUESTED_MODE: AtomicU32 = AtomicU32::new(0);
static APPLIED_MODE: AtomicU32 = AtomicU32::new(0);
pub fn set_fullscreen(on: Option<bool>) {
    set_window_mode(
        if on.unwrap_or(REQUESTED_MODE.load(Ordering::Relaxed) == 0) {
            1
        } else {
            0
        },
    );
}
pub fn set_window_mode(mode: u32) {
    REQUESTED_MODE.store(mode.min(1), Ordering::Relaxed);
}
pub fn simulate_activation(_active: bool) {}
pub fn simulate_display_change() {
    RESIZED.store(true, Ordering::Release);
}
pub fn reapply_mode() {}
pub fn exclusive_wanted() -> Option<isize> {
    None
}
pub fn window_mode() -> u32 {
    APPLIED_MODE.load(Ordering::Relaxed)
}
pub fn current_monitor() -> Option<(isize, String)> {
    None
}
pub fn set_client_size(w: u32, h: u32) {
    STATE.with(|s| {
        if let Some(s) = s.borrow_mut().as_mut() {
            let _ = s.window.set_size(w, h);
        }
    });
}
fn virtual_key(k: sdl3::keyboard::Scancode) -> Option<u8> {
    use sdl3::keyboard::Scancode as K;
    let n = k as u32;
    if (K::A as u32..=K::Z as u32).contains(&n) {
        return Some((0x41 + n - K::A as u32) as u8);
    }
    if (K::F1 as u32..=K::F12 as u32).contains(&n) {
        return Some((0x70 + n - K::F1 as u32) as u8);
    }
    if (K::_1 as u32..=K::_9 as u32).contains(&n) {
        return Some((b'1' as u32 + n - K::_1 as u32) as u8);
    }
    Some(match k {
        K::_0 => b'0',
        K::Escape => 0x1b,
        K::Return | K::KpEnter => 0x0d,
        K::Space => 0x20,
        K::Tab => 9,
        K::Backspace => 8,
        K::LShift => 0xa0,
        K::RShift => 0xa1,
        K::LCtrl => 0xa2,
        K::RCtrl => 0xa3,
        K::LAlt => 0xa4,
        K::RAlt => 0xa5,
        K::Left => 0x25,
        K::Up => 0x26,
        K::Right => 0x27,
        K::Down => 0x28,
        K::Home => 0x24,
        K::End => 0x23,
        K::PageUp => 0x21,
        K::PageDown => 0x22,
        K::Insert => 0x2d,
        K::Delete => 0x2e,
        K::Minus => 0xbd,
        K::Equals => 0xbb,
        K::LeftBracket => 0xdb,
        K::RightBracket => 0xdd,
        K::Backslash => 0xdc,
        K::Semicolon => 0xba,
        K::Apostrophe => 0xde,
        K::Grave => 0xc0,
        K::Comma => 0xbc,
        K::Period => 0xbe,
        K::Slash => 0xbf,
        _ => return None,
    })
}

fn publish(input: &Input, connected: bool) {
    for k in &KEYS {
        k.store(false, Ordering::Relaxed);
    }
    for &k in &input.keys {
        if let Some(v) = virtual_key(k) {
            KEYS[v as usize].store(true, Ordering::Relaxed);
        }
    }
    for (modifier, l, r) in [(0x10, 0xa0, 0xa1), (0x11, 0xa2, 0xa3), (0x12, 0xa4, 0xa5)] {
        KEYS[modifier].store(
            KEYS[l].load(Ordering::Relaxed) || KEYS[r].load(Ordering::Relaxed),
            Ordering::Relaxed,
        );
    }
    MOUSE_DX.fetch_add(input.relative.0 as i64, Ordering::Relaxed);
    MOUSE_DY.fetch_add(input.relative.1 as i64, Ordering::Relaxed);
    MOUSE_WHEEL.fetch_add((input.wheel.1 * 120.0) as i32, Ordering::Relaxed);
    let mut buttons = 0;
    for (b, bit) in [
        (sdl3::mouse::MouseButton::Left, 1),
        (sdl3::mouse::MouseButton::Right, 2),
        (sdl3::mouse::MouseButton::Middle, 4),
        (sdl3::mouse::MouseButton::X1, 8),
        (sdl3::mouse::MouseButton::X2, 16),
    ] {
        if input.mouse_buttons.contains(&b) {
            buttons |= bit;
        }
    }
    MOUSE_BUTTONS.store(buttons, Ordering::Relaxed);
    crate::platform::controller::publish(connected, &input.pad_buttons, &input.pad_axes);
}

pub fn time_domain() -> ash::vk::TimeDomainEXT {
    ash::vk::TimeDomainEXT::CLOCK_MONOTONIC
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn sdl_pad_and_keyboard_mapping() {
        // The shared IME test changes process-wide pad swallowing. Exercise the complete
        // input path in its own process, without an SDL window or physical devices.
        if std::env::var_os("REGALIANT_INPUT_TEST").is_none() {
            let out = std::process::Command::new(std::env::current_exe().unwrap())
                .args([
                    "--exact",
                    "gpu::vk::window::tests::sdl_pad_and_keyboard_mapping",
                    "--nocapture",
                ])
                .env("REGALIANT_INPUT_TEST", "1")
                .output()
                .unwrap();
            assert!(
                out.status.success(),
                "{}\n{}",
                String::from_utf8_lossy(&out.stdout),
                String::from_utf8_lossy(&out.stderr)
            );
            return;
        }
        use sdl3::gamepad::{Axis, Button, GamepadType};
        crate::platform::controller::set_type(GamepadType::PS5);
        assert_eq!(crate::platform::controller::prompt_style(), 1);
        crate::platform::controller::set_type(GamepadType::XboxOne);
        assert_eq!(crate::platform::controller::prompt_style(), 2);
        use sdl3::keyboard::Scancode as Key;
        let mut input = Input::default();
        publish(&input, false);
        assert_eq!(crate::input::pad_state(), crate::input::PadState::default());
        input.pad_buttons.insert(Button::South);
        input.pad_buttons.insert(Button::Touchpad);
        input.pad_axes.insert(Axis::LeftX, -32768);
        input.pad_axes.insert(Axis::TriggerRight, 32767);
        publish(&input, true);
        let pad = crate::input::pad_state();
        assert_eq!(pad.lx, 0);
        assert_eq!(pad.r2, 255);
        assert_eq!(pad.buttons, crate::input::BTN_CROSS | crate::input::BTN_R2 | crate::input::BTN_TOUCHPAD);
        input.keys.extend([Key::W, Key::LAlt, Key::LShift]);
        input.mouse_buttons.insert(sdl3::mouse::MouseButton::Left);
        publish(&input, true);
        assert!(key_down(b'W') && key_down(0x10) && key_down(0x12));
        assert_eq!(MOUSE_BUTTONS.load(Ordering::Relaxed), 1);
        publish(&Input::default(), false);
    }
}

/// SDL pointer events are logical; the UI letterbox calculation uses drawable pixels.
fn pointer_pixels_at((x, y): (f32, f32), (w, h): (u32, u32), (pw, ph): (u32, u32)) -> (i32, i32) {
    ((x * pw as f32 / w.max(1) as f32).round() as i32,
     (y * ph as f32 / h.max(1) as f32).round() as i32)
}

#[cfg(test)]
mod hidpi_tests {
    #[test]
    fn fractional_pointer_matches_drawable() {
        assert_eq!(super::pointer_pixels_at((1200.0, 675.0), (2400, 1350), (3840, 2160)), (1920, 1080));
        assert_eq!(super::pointer_pixels_at((100.0, 50.0), (1100, 760), (1760, 1216)), (160, 80));
        assert_eq!(super::pointer_pixels_at((100.0, 50.0), (1100, 760), (1100, 760)), (100, 50));
    }
}
