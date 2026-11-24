//! SDL owns Wayland/X11 protocol and input handling. This surface is ready for the Vulkan
//! renderer port; the bootstrap does not create a swapchain or render game frames yet.
use ash::vk;
use sdl3::event::{Event, WindowEvent};
use std::collections::{HashMap, HashSet};
use std::sync::atomic::{AtomicU32, Ordering};

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

static RUMBLE: AtomicU32 = AtomicU32::new(0);
static LIGHTBAR: std::sync::atomic::AtomicU32 = std::sync::atomic::AtomicU32::new(0);
pub fn take_lightbar() -> Option<[u8; 3]> {
    let v = LIGHTBAR.swap(0, Ordering::Relaxed);
    (v != 0).then_some([(v >> 16) as u8, (v >> 8) as u8, v as u8])
}
pub fn set_lightbar(rgb: Option<[u8; 3]>) {
    let value = rgb.map_or(0, |[r, g, b]| 0x1000000 | ((r as u32) << 16) | ((g as u32) << 8) | b as u32);
    LIGHTBAR.store(value, Ordering::Relaxed);
}
pub fn haptics(left: f32, right: f32) {
    let l = (left.clamp(0.0, 1.0) * 65535.0) as u32;
    let r = (right.clamp(0.0, 1.0) * 65535.0) as u32;
    RUMBLE.store(l | (r << 16), Ordering::Relaxed);
}

pub struct Frontend {
    surface: vk::SurfaceKHR,
    surface_api: ash::khr::surface::Instance,
    instance: ash::Instance,
    _entry: ash::Entry,
    pub window: sdl3::video::Window,
    events: sdl3::EventPump,
    pads: HashMap<sdl3::joystick::JoystickId, sdl3::gamepad::Gamepad>,
    gamepads: sdl3::GamepadSubsystem,
    _video: sdl3::VideoSubsystem,
    _sdl: sdl3::Sdl,
    pub input: Input,
}

impl Frontend {
    pub fn new(hidden: bool) -> Result<Self, String> {
        let sdl = sdl3::init().map_err(|e| e.to_string())?;
        let video = sdl.video().map_err(|e| e.to_string())?;
        // SDL's default selects the available compositor backend; SDL_VIDEO_DRIVER can force
        // wayland or x11. No protocol code, mouse grab or focus request lives in this backend.
        let mut builder = video.window("Regaliant — Linux bootstrap", 1280, 720);
        builder.vulkan().resizable().high_pixel_density();
        if hidden { builder.hidden(); }
        let window = builder.build().map_err(|e| e.to_string())?;
        let extensions: Vec<_> = window.vulkan_instance_extensions().map_err(|e| e.to_string())?.iter()
            .map(|s| std::ffi::CString::new(s.as_str()).unwrap()).collect();
        let names: Vec<_> = extensions.iter().map(|s| s.as_ptr()).collect();
        let entry = unsafe { ash::Entry::load() }.map_err(|e| e.to_string())?;
        let app = vk::ApplicationInfo::default().application_name(c"Regaliant").api_version(vk::API_VERSION_1_3);
        let info = vk::InstanceCreateInfo::default().application_info(&app).enabled_extension_names(&names);
        let instance = unsafe { entry.create_instance(&info, None) }.map_err(|e| format!("Vulkan instance: {e:?}"))?;
        let surface = match unsafe { window.vulkan_create_surface(instance.handle()) } {
            Ok(s) => s,
            Err(e) => { unsafe { instance.destroy_instance(None) }; return Err(e.to_string()); }
        };
        let surface_api = ash::khr::surface::Instance::new(&entry, &instance);
        // Finish fallible SDL initialization before transferring Vulkan ownership to Self.
        let input = (|| {
            let events = sdl.event_pump().map_err(|e| e.to_string())?;
            let gamepads = sdl.gamepad().map_err(|e| e.to_string())?;
            let pads = gamepads.gamepads().map_err(|e| e.to_string())?.into_iter()
                .filter_map(|id| gamepads.open(id).ok().map(|pad| (id, pad))).collect();
            Ok::<_, String>((events, gamepads, pads))
        })();
        let (events, gamepads, pads) = match input {
            Ok(v) => v,
            Err(e) => { unsafe { surface_api.destroy_surface(surface, None); instance.destroy_instance(None); } return Err(e); }
        };
        Ok(Self { surface, surface_api, instance, _entry: entry, window, events, pads, gamepads, _video: video, _sdl: sdl, input: Input::default() })
    }

    pub fn present_devices(&self) -> Result<Vec<String>, String> {
        let mut out = Vec::new();
        for device in unsafe { self.instance.enumerate_physical_devices() }.map_err(|e| format!("Vulkan devices: {e:?}"))? {
            let queues = unsafe { self.instance.get_physical_device_queue_family_properties(device) };
            for (index, q) in queues.iter().enumerate() {
                if q.queue_flags.contains(vk::QueueFlags::GRAPHICS) && unsafe { self.surface_api.get_physical_device_surface_support(device, index as u32, self.surface) }.unwrap_or(false) {
                    let props = unsafe { self.instance.get_physical_device_properties(device) };
                    out.push(unsafe { std::ffi::CStr::from_ptr(props.device_name.as_ptr()) }.to_string_lossy().into_owned());
                    break;
                }
            }
        }
        if out.is_empty() { return Err("no Vulkan graphics queue supports this surface".into()); }
        Ok(out)
    }

    /// Pumps on the main thread. Returns false on a window-close request.
    pub fn poll(&mut self) -> bool {
        self.input.relative = (0.0, 0.0);
        self.input.wheel = (0.0, 0.0);
        for event in self.events.poll_iter() {
            match event {
                Event::Quit { .. } | Event::Window { win_event: WindowEvent::CloseRequested, .. } => return false,
                Event::Window { win_event: WindowEvent::FocusLost, .. } => { self.input.keys.clear(); self.input.mouse_buttons.clear(); }
                Event::KeyDown { scancode: Some(k), .. } => { self.input.keys.insert(k); }
                Event::KeyUp { scancode: Some(k), .. } => { self.input.keys.remove(&k); }
                Event::MouseMotion { x, y, xrel, yrel, .. } => { self.input.pointer = (x, y); self.input.relative.0 += xrel; self.input.relative.1 += yrel; }
                Event::MouseButtonDown { mouse_btn, .. } => { self.input.mouse_buttons.insert(mouse_btn); }
                Event::MouseButtonUp { mouse_btn, .. } => { self.input.mouse_buttons.remove(&mouse_btn); }
                Event::MouseWheel { x, y, .. } => { self.input.wheel.0 += x; self.input.wheel.1 += y; }
                Event::GamepadAdded { which, .. } => { if let Ok(pad) = self.gamepads.open(which) { self.pads.insert(which, pad); } }
                Event::GamepadRemoved { which, .. } => { self.pads.remove(&which); self.input.pad_axes.clear(); self.input.pad_buttons.clear(); }
                Event::GamepadButtonDown { button, .. } => { self.input.pad_buttons.insert(button); }
                Event::GamepadButtonUp { button, .. } => { self.input.pad_buttons.remove(&button); }
                Event::GamepadAxisMotion { axis, value, .. } => { self.input.pad_axes.insert(axis, value); }
                _ => {}
            }
        }
        crate::platform::controller::publish(!self.pads.is_empty(), &self.input.pad_buttons, &self.input.pad_axes);
        if let Some([r, g, b]) = take_lightbar() { for pad in self.pads.values_mut() { let _ = pad.set_led(r, g, b); } }
        let rumble = RUMBLE.load(Ordering::Relaxed);
        if rumble != 0 { for pad in self.pads.values_mut() { let _ = pad.set_rumble(rumble as u16, (rumble >> 16) as u16, 100); } }
        true
    }
}
impl Drop for Frontend {
    fn drop(&mut self) {
        unsafe { self.surface_api.destroy_surface(self.surface, None); self.instance.destroy_instance(None); }
    }
}

pub fn rumble() -> u32 { RUMBLE.load(Ordering::Relaxed) }
