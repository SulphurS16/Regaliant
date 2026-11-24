//! SDL3 playback (PipeWire/PulseAudio/ALSA according to the installed SDL driver). The shared
//! mixer retains its bounded grain queues; this backend limits queued output to 20 ms.
use sdl3::sys::audio::*;
use std::sync::atomic::{AtomicBool, Ordering};
static READY: AtomicBool = AtomicBool::new(false);

/// Called once on the main thread. SDL audio stays initialized until process exit because
/// the existing mixer worker has process lifetime.
pub fn initialize() -> Result<(), String> {
    if !READY.load(Ordering::Acquire) {
        if !unsafe { sdl3::sys::init::SDL_InitSubSystem(sdl3::sys::init::SDL_INIT_AUDIO) } {
            return Err(sdl3::get_error().to_string());
        }
        READY.store(true, Ordering::Release);
    }
    super::SPATIAL.store(0, Ordering::Relaxed);
    Ok(())
}

pub fn priority() {} // SDL manages its device thread; the producer does not request realtime policy.

pub unsafe fn devices() -> Result<Vec<(String, String)>, String> {
    if !READY.load(Ordering::Acquire) { return Err("SDL audio is not initialized".into()); }
    let mut count = 0;
    let ids = unsafe { SDL_GetAudioPlaybackDevices(&mut count) };
    if ids.is_null() { return Err(sdl3::get_error().to_string()); }
    let mut out = Vec::new();
    for i in 0..count.max(0) as usize {
        let id = unsafe { *ids.add(i) };
        let name = unsafe { SDL_GetAudioDeviceName(id) };
        if !name.is_null() { out.push((id.0.to_string(), unsafe { std::ffi::CStr::from_ptr(name) }.to_string_lossy().into_owned())); }
    }
    unsafe { sdl3::sys::stdinc::SDL_free(ids.cast()) };
    Ok(out)
}

struct Output(*mut SDL_AudioStream);
impl Drop for Output { fn drop(&mut self) { unsafe { SDL_DestroyAudioStream(self.0) }; } }
impl Output {
    fn new(channels: usize) -> Result<Self, String> {
        if !READY.load(Ordering::Acquire) { return Err("SDL audio initialization failed or was skipped".into()); }
        let spec = SDL_AudioSpec { format: SDL_AUDIO_F32, channels: channels as i32, freq: super::RATE as i32 };
        let id = super::output_device().and_then(|s| s.parse::<u32>().ok()).map(SDL_AudioDeviceID).unwrap_or(SDL_AUDIO_DEVICE_DEFAULT_PLAYBACK);
        let ptr = unsafe { SDL_OpenAudioDeviceStream(id, &spec, None, std::ptr::null_mut()) };
        if ptr.is_null() { return Err(sdl3::get_error().to_string()); }
        let output = Self(ptr);
        if !unsafe { SDL_ResumeAudioStreamDevice(ptr) } { return Err(sdl3::get_error().to_string()); }
        super::DEVICE_CHANNELS.store(channels as u32, Ordering::Relaxed);
        Ok(output)
    }
    fn queued(&self) -> Result<usize, String> {
        let bytes = unsafe { SDL_GetAudioStreamQueued(self.0) };
        if bytes < 0 { Err(sdl3::get_error().to_string()) } else { Ok(bytes as usize) }
    }
    fn push(&self, samples: &[f32]) -> Result<(), String> {
        if !unsafe { SDL_PutAudioStreamData(self.0, samples.as_ptr().cast(), std::mem::size_of_val(samples) as i32) } { return Err(sdl3::get_error().to_string()); }
        Ok(())
    }
}

pub fn run() -> Result<(), String> {
    loop {
        let channels = (super::OUT_CHANNELS.load(Ordering::Relaxed) as usize).clamp(2, 8);
        let generation = super::DEVICE_GEN.load(Ordering::Relaxed);
        let output = Output::new(channels)?;
        crate::info!("audio: SDL3 output, 48000 Hz, {channels} channels");
        let frames = 480;
        let mut buffer = vec![0f32; frames * channels];
        while super::OUT_CHANNELS.load(Ordering::Relaxed) as usize == channels && super::DEVICE_GEN.load(Ordering::Relaxed) == generation {
            if output.queued()? < buffer.len() * 4 * 2 {
                super::mix_into(&mut buffer, frames, channels);
                output.push(&buffer)?;
            } else { std::thread::sleep(std::time::Duration::from_millis(1)); }
        }
    }
}

/// Silence only. Verifies stream creation, queueing and device-clock consumption, not audibility.
pub fn smoke_test() -> Result<(), String> {
    initialize()?;
    let output = Output::new(2)?;
    let driver = unsafe { SDL_GetCurrentAudioDriver() };
    let driver = if driver.is_null() { "unknown".into() } else { unsafe { std::ffi::CStr::from_ptr(driver) }.to_string_lossy() };
    let device = unsafe { SDL_GetAudioStreamDevice(output.0) };
    let seconds = std::env::var("DS_AUDIO_CHECK_SECONDS").ok().and_then(|s| s.parse::<u64>().ok()).unwrap_or(2).clamp(1, 60);
    println!("SDL audio check: driver={driver} device={} duration={seconds}s rate=48000 channels=2 format=F32 silence=true", device.0);
    let buffer = vec![0f32; 480 * 2];
    let started = std::time::Instant::now();
    let mut submitted = 0usize;
    let mut last_consumed = 0;
    let mut reported = 0;
    while started.elapsed().as_secs() < seconds {
        let mut queued = output.queued()?;
        while queued < buffer.len() * 8 {
            output.push(&buffer)?;
            submitted += buffer.len() * 4;
            queued += buffer.len() * 4;
        }
        let elapsed = started.elapsed().as_secs();
        if elapsed > reported {
            reported = elapsed;
            let consumed = submitted.saturating_sub(output.queued()?);
            let paused = unsafe { SDL_AudioDevicePaused(device) };
            println!("SDL audio stream: elapsed={elapsed}s paused={paused} submitted_bytes={submitted} consumed_bytes={consumed} queued_bytes={}", output.queued()?);
            if paused || (elapsed > 2 && consumed == last_consumed) { return Err("audio stream stopped consuming silence".into()); }
            last_consumed = consumed;
        }
        std::thread::sleep(std::time::Duration::from_millis(10));
    }
    let consumed = submitted.saturating_sub(output.queued()?);
    println!("SDL audio check complete: driver={driver} elapsed={:.3}s consumed_bytes={consumed}; queue consumption is not proof of audibility", started.elapsed().as_secs_f64());
    if consumed == 0 { Err("audio queue did not drain".into()) } else { Ok(()) }
}
