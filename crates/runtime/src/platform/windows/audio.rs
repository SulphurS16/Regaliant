use windows::Win32::Foundation::{CloseHandle, WAIT_OBJECT_0};
use windows::Win32::Media::Audio::*;
use windows::Win32::Media::KernelStreaming::{
    SPEAKER_BACK_LEFT, SPEAKER_BACK_RIGHT, SPEAKER_FRONT_CENTER, SPEAKER_FRONT_LEFT, SPEAKER_FRONT_RIGHT, SPEAKER_LOW_FREQUENCY, SPEAKER_SIDE_LEFT, SPEAKER_SIDE_RIGHT, WAVE_FORMAT_EXTENSIBLE,
};
use windows::Win32::Media::Multimedia::KSDATAFORMAT_SUBTYPE_IEEE_FLOAT;
use windows::Win32::System::Com::{CLSCTX_ALL, COINIT_MULTITHREADED, CoCreateInstance, CoInitializeEx};
use windows::Win32::System::Threading::{CreateEventW, WaitForSingleObject};

/// Runs the output, opening the client again whenever the wanted channel count changes
/// (`OUT_CHANNELS`; 12: Windows' spatial sound, a 7.1.4 static stream); returns on a lost
/// device.
pub fn run() -> Result<(), String> {
    unsafe {
        CoInitializeEx(None, COINIT_MULTITHREADED).ok().map_err(|e| format!("CoInitializeEx: {e}"))?;
    }
    loop {
        let och = super::OUT_CHANNELS.load(std::sync::atomic::Ordering::Relaxed) as usize;
        if och == 12 && super::SPATIAL.load(std::sync::atomic::Ordering::Relaxed) != 0 && super::spatial_allowed() {
            match unsafe { spatial_session() } {
                Ok(true) => continue,
                Ok(false) => return Ok(()),
                Err(e) => {
                    // Not available on this device (spatial sound off in Windows): the renderer
                    // picks another layout at its next grain; a stereo client meanwhile.
                    if super::SPATIAL.swap(0, std::sync::atomic::Ordering::Relaxed) != 0 {
                        crate::info!("audio: Windows spatial sound not available ({e}): the 3D sound is rendered here");
                    }
                }
            }
        }
        match unsafe { session(och.max(2)) } {
            Ok(true) => continue,
            Ok(false) => return Ok(()),
            Err(e) => return Err(e),
        }
    }
}

/// The device the output opens (the PC tab's, else the system's default).
unsafe fn open_device(en: &IMMDeviceEnumerator) -> Result<IMMDevice, String> {
    unsafe {
        let chosen = super::output_device().and_then(|id| {
            let w: Vec<u16> = id.encode_utf16().chain(std::iter::once(0)).collect();
            en.GetDevice(windows::core::PCWSTR(w.as_ptr())).ok()
        });
        match chosen {
            Some(d) => Ok(d),
            None => en.GetDefaultAudioEndpoint(eRender, eConsole).map_err(|e| format!("default endpoint: {e}")),
        }
    }
}

/// A spatial stream kept for the process's life: stopped when another output is wanted and
/// started again when spatial sound is wanted on the same device. (Releasing one ended the
/// process with heap corruption in Windows' heap, with or without our writes to its buffers:
/// FINDINGS.md 2026-11-06 night.)
struct Kept {
    generation: u32,
    _client: ISpatialAudioClient,
    _formats: IAudioFormatEnumerator,
    stream: ISpatialAudioObjectRenderStream,
    objects: Vec<ISpatialAudioObject>,
    event: windows::Win32::Foundation::HANDLE,
}

thread_local! {
    static KEPT: std::cell::RefCell<Vec<Kept>> = const { std::cell::RefCell::new(Vec::new()) };
}

/// Windows' spatial sound: a 7.1.4 static-object stream fed from the mixer's 12 channels
/// (FL FR FC LFE BL BR SL SR TFL TFR TBL TBR). Ok(true) when the wanted output changed.
unsafe fn spatial_session() -> Result<bool, String> {
    use std::sync::atomic::Ordering::Relaxed;
    let generation = super::DEVICE_GEN.load(Relaxed);
    let found = KEPT.with(|k| {
        let mut k = k.borrow_mut();
        k.iter().position(|x| x.generation == generation).map(|i| k.remove(i))
    });
    let kept = match found {
        Some(k) => {
            unsafe { k.stream.Start() }.map_err(|e| format!("restart: {e}"))?;
            crate::info!("audio: Windows spatial sound: the 7.1.4 stream started again");
            k
        }
        None => unsafe { spatial_open(generation)? },
    };
    let r = unsafe { spatial_run(&kept) };
    let _ = unsafe { kept.stream.Stop() };
    crate::info!("audio: Windows spatial sound stream stopped (kept)");
    KEPT.with(|k| k.borrow_mut().push(kept));
    r
}

/// A new 7.1.4 static-object stream on the output device, started.
unsafe fn spatial_open(generation: u32) -> Result<Kept, String> {
    use std::sync::atomic::Ordering::Relaxed;
    use windows::Win32::System::Com::StructuredStorage::PROPVARIANT;
    use windows::Win32::System::Variant::VT_BLOB;
    unsafe {
        let en: IMMDeviceEnumerator = CoCreateInstance(&MMDeviceEnumerator, None, CLSCTX_ALL).map_err(|e| format!("device enumerator: {e}"))?;
        let dev = open_device(&en)?;
        super::DEVICE_FORM.store(form_factor(&dev).unwrap_or(u32::MAX), Relaxed);
        let client: ISpatialAudioClient = dev.Activate(CLSCTX_ALL, None).map_err(|e| format!("spatial client: {e}"))?;
        let formats = client.GetSupportedAudioObjectFormatEnumerator().map_err(|e| format!("formats: {e}"))?;
        let fmt = formats.GetFormat(0).map_err(|e| format!("format: {e}"))?;
        if fmt.is_null() || (*fmt).nSamplesPerSec != super::RATE || (*fmt).nChannels != 1 || (*fmt).wBitsPerSample != 32 {
            return Err("its object format is not 48 kHz mono float".into());
        }
        let types = [
            AudioObjectType_FrontLeft,
            AudioObjectType_FrontRight,
            AudioObjectType_FrontCenter,
            AudioObjectType_LowFrequency,
            AudioObjectType_BackLeft,
            AudioObjectType_BackRight,
            AudioObjectType_SideLeft,
            AudioObjectType_SideRight,
            AudioObjectType_TopFrontLeft,
            AudioObjectType_TopFrontRight,
            AudioObjectType_TopBackLeft,
            AudioObjectType_TopBackRight,
        ];
        let mask = AudioObjectType(types.iter().fold(0, |m, t| m | t.0));
        let event = CreateEventW(None, false, false, None).map_err(|e| format!("event: {e}"))?;
        let params = SpatialAudioObjectRenderStreamActivationParams {
            ObjectFormat: fmt,
            StaticObjectTypeMask: mask,
            MinDynamicObjectCount: 0,
            MaxDynamicObjectCount: 0,
            Category: AudioCategory_GameEffects,
            EventHandle: event,
            NotifyObject: std::mem::ManuallyDrop::new(None),
        };
        let mut pv = PROPVARIANT::default();
        (*pv.Anonymous.Anonymous).vt = VT_BLOB;
        (*pv.Anonymous.Anonymous).Anonymous.blob.cbSize = std::mem::size_of::<SpatialAudioObjectRenderStreamActivationParams>() as u32;
        (*pv.Anonymous.Anonymous).Anonymous.blob.pBlobData = &params as *const _ as *mut u8;
        // (A stream that failed to start is kept as well: nothing of it is released.)
        let stream: ISpatialAudioObjectRenderStream = client.ActivateSpatialAudioStream(&pv).map_err(|e| format!("stream: {e}"))?;
        let mut objects = Vec::new();
        for t in types {
            match stream.ActivateSpatialAudioObject(t) {
                Ok(o) => objects.push(o),
                Err(e) => {
                    std::mem::forget(stream);
                    return Err(format!("object {t:?}: {e}"));
                }
            }
        }
        if let Err(e) = stream.Start() {
            std::mem::forget((stream, objects));
            return Err(format!("start: {e}"));
        }
        super::SPATIAL.store(1, Relaxed);
        crate::info!("audio: Windows spatial sound: a 7.1.4 static-object stream, 48 kHz");
        Ok(Kept { generation, _client: client, _formats: formats, stream, objects, event })
    }
}

/// Feeds a started spatial stream until the wanted output or device changes (Ok(true)).
unsafe fn spatial_run(k: &Kept) -> Result<bool, String> {
    use std::sync::atomic::Ordering::Relaxed;
    unsafe {
        let mut scratch: Vec<f32> = Vec::new();
        loop {
            let w = WaitForSingleObject(k.event, 200);
            if w != WAIT_OBJECT_0 && w.0 != 0x102 {
                return Err("spatial event wait failed".into());
            }
            if super::OUT_CHANNELS.load(Relaxed) != 12 || super::DEVICE_GEN.load(Relaxed) != k.generation {
                return Ok(true);
            }
            // Updates only when the stream asked for one (its event).
            if w != WAIT_OBJECT_0 {
                continue;
            }
            let (mut avail, mut frames) = (0u32, 0u32);
            if let Err(e) = k.stream.BeginUpdatingAudioObjects(&mut avail, &mut frames) {
                crate::warn!("audio: spatial stream lost ({e})");
                return Ok(true);
            }
            let n = frames as usize;
            scratch.resize(12 * n, 0.0);
            super::mix_into(&mut scratch, n, 12);
            for (ch, o) in k.objects.iter().enumerate() {
                let (mut p, mut len) = (std::ptr::null_mut::<u8>(), 0u32);
                if o.GetBuffer(&mut p, &mut len).is_ok() && !p.is_null() {
                    let dst = std::slice::from_raw_parts_mut(p as *mut f32, (len as usize / 4).min(n));
                    for (f, d) in dst.iter_mut().enumerate() {
                        *d = scratch[12 * f + ch];
                    }
                }
            }
            let _ = k.stream.EndUpdatingAudioObjects();
        }
    }
}

/// A device's form factor (PKEY_AudioEndpoint_FormFactor).
unsafe fn form_factor(d: &IMMDevice) -> Option<u32> {
    use windows::Win32::Foundation::PROPERTYKEY;
    use windows::Win32::System::Com::STGM_READ;
    use windows::Win32::System::Com::StructuredStorage::PropVariantClear;
    use windows::Win32::System::Variant::VT_UI4;
    const FORM_FACTOR: PROPERTYKEY = PROPERTYKEY { fmtid: windows::core::GUID::from_u128(0x1da5d803_d492_4edd_8c23_e0c0ffee7f0e), pid: 0 };
    unsafe {
        let store = d.OpenPropertyStore(STGM_READ).ok()?;
        let mut pv = store.GetValue(&FORM_FACTOR).ok()?;
        let v = (pv.Anonymous.Anonymous.vt == VT_UI4).then(|| pv.Anonymous.Anonymous.Anonymous.ulVal);
        let _ = PropVariantClear(&mut pv);
        v
    }
}

/// The active render endpoints: (id, friendly name).
pub unsafe fn devices() -> Result<Vec<(String, String)>, String> {
    use windows::Win32::Foundation::PROPERTYKEY;
    use windows::Win32::System::Com::STGM_READ;
    use windows::Win32::System::Com::StructuredStorage::PropVariantClear;
    use windows::Win32::System::Variant::VT_LPWSTR;
    // PKEY_Device_FriendlyName.
    const FRIENDLY_NAME: PROPERTYKEY = PROPERTYKEY { fmtid: windows::core::GUID::from_u128(0xa45c254e_df1c_4efd_8020_67d146a850e0), pid: 14 };
    unsafe {
        let _ = CoInitializeEx(None, COINIT_MULTITHREADED);
        let en: IMMDeviceEnumerator = CoCreateInstance(&MMDeviceEnumerator, None, CLSCTX_ALL).map_err(|e| format!("device enumerator: {e}"))?;
        let col = en.EnumAudioEndpoints(eRender, DEVICE_STATE_ACTIVE).map_err(|e| format!("endpoints: {e}"))?;
        let n = col.GetCount().map_err(|e| format!("count: {e}"))?;
        let mut out = Vec::new();
        for i in 0..n {
            let Ok(d) = col.Item(i) else { continue };
            let Ok(id) = d.GetId() else { continue };
            let ids = id.to_string().unwrap_or_default();
            windows::Win32::System::Com::CoTaskMemFree(Some(id.0 as *const _));
            let mut name = String::new();
            if let Ok(store) = d.OpenPropertyStore(STGM_READ) {
                if let Ok(mut pv) = store.GetValue(&FRIENDLY_NAME) {
                    if pv.Anonymous.Anonymous.vt == VT_LPWSTR {
                        name = pv.Anonymous.Anonymous.Anonymous.pwszVal.to_string().unwrap_or_default();
                    }
                    let _ = PropVariantClear(&mut pv);
                }
            }
            if !ids.is_empty() {
                out.push((ids, if name.is_empty() { format!("Device {}", i + 1) } else { name }));
            }
        }
        Ok(out)
    }
}

/// One client of `och` channels: Ok(true) when the wanted channels or device changed (open
/// again).
unsafe fn session(och: usize) -> Result<bool, String> {
    unsafe {
        crate::debug!("audio: session({och}): enumerator");
        let en: IMMDeviceEnumerator = CoCreateInstance(&MMDeviceEnumerator, None, CLSCTX_ALL).map_err(|e| format!("device enumerator: {e}"))?;
        let generation = super::DEVICE_GEN.load(std::sync::atomic::Ordering::Relaxed);
        let dev = open_device(&en)?;
        crate::info!("audio: session({och}): device open");
        let client: IAudioClient = dev.Activate(CLSCTX_ALL, None).map_err(|e| format!("activate: {e}"))?;
        crate::info!("audio: session({och}): client activated");
        super::DEVICE_FORM.store(form_factor(&dev).unwrap_or(u32::MAX), std::sync::atomic::Ordering::Relaxed);
        if let Ok(mix) = client.GetMixFormat() {
            if !mix.is_null() {
                super::DEVICE_CHANNELS.store((*mix).nChannels as u32, std::sync::atomic::Ordering::Relaxed);
                windows::Win32::System::Com::CoTaskMemFree(Some(mix as *const _));
            }
        }
        // The wanted channels (a surround bed on a stereo device: Windows downmixes it, or with
        // its spatial sound on (Windows Sonic, Dolby Atmos) renders it for the headphones);
        // 7.1.4 (diagnosis only) as stereo.
        let wanted = och;
        let dev_ch = super::DEVICE_CHANNELS.load(std::sync::atomic::Ordering::Relaxed) as usize;
        let och = if och > 8 { 2 } else { och };
        crate::info!("audio: opening a WASAPI client of {och} channels ({wanted} wanted, the device mixes {dev_ch})");
        let mut fmt: WAVEFORMATEXTENSIBLE = std::mem::zeroed();
        fmt.Format.wFormatTag = WAVE_FORMAT_EXTENSIBLE as u16;
        fmt.Format.nChannels = och as u16;
        fmt.Format.nSamplesPerSec = super::RATE;
        fmt.Format.wBitsPerSample = 32;
        fmt.Format.nBlockAlign = 4 * och as u16;
        fmt.Format.nAvgBytesPerSec = super::RATE * 4 * och as u32;
        fmt.Format.cbSize = 22;
        fmt.Samples.wValidBitsPerSample = 32;
        fmt.dwChannelMask = match och {
            6 => SPEAKER_FRONT_LEFT | SPEAKER_FRONT_RIGHT | SPEAKER_FRONT_CENTER | SPEAKER_LOW_FREQUENCY | SPEAKER_BACK_LEFT | SPEAKER_BACK_RIGHT,
            8 => SPEAKER_FRONT_LEFT | SPEAKER_FRONT_RIGHT | SPEAKER_FRONT_CENTER | SPEAKER_LOW_FREQUENCY | SPEAKER_BACK_LEFT | SPEAKER_BACK_RIGHT | SPEAKER_SIDE_LEFT | SPEAKER_SIDE_RIGHT,
            _ => SPEAKER_FRONT_LEFT | SPEAKER_FRONT_RIGHT,
        };
        fmt.SubFormat = KSDATAFORMAT_SUBTYPE_IEEE_FLOAT;
        let flags = AUDCLNT_STREAMFLAGS_EVENTCALLBACK | AUDCLNT_STREAMFLAGS_AUTOCONVERTPCM | AUDCLNT_STREAMFLAGS_SRC_DEFAULT_QUALITY;
        // 30 ms device buffer (100 ns units).
        client
            .Initialize(AUDCLNT_SHAREMODE_SHARED, flags, 300_000, 0, &fmt as *const _ as *const WAVEFORMATEX, None)
            .map_err(|e| format!("initialize: {e}"))?;
        let event = CreateEventW(None, false, false, None).map_err(|e| format!("event: {e}"))?;
        client.SetEventHandle(event).map_err(|e| format!("event handle: {e}"))?;
        let render: IAudioRenderClient = client.GetService().map_err(|e| format!("render client: {e}"))?;
        let total = client.GetBufferSize().map_err(|e| format!("buffer size: {e}"))?;
        client.Start().map_err(|e| format!("start: {e}"))?;
        crate::info!(
            "audio: WASAPI shared-mode output, 48 kHz float, {och} channels (the device mixes {}, form factor {}), device buffer {total} frames",
            super::DEVICE_CHANNELS.load(std::sync::atomic::Ordering::Relaxed),
            match super::DEVICE_FORM.load(std::sync::atomic::Ordering::Relaxed) {
                1 => "speakers",
                3 => "headphones",
                5 => "headset",
                9 => "a display (HDMI)",
                8 => "S/PDIF",
                u32::MAX => "unknown",
                _ => "other",
            }
        );
        let mut scratch = vec![0f32; och * total as usize];
        loop {
            let w = WaitForSingleObject(event, 200);
            if w != WAIT_OBJECT_0 && w.0 != 0x102 {
                break;
            }
            if super::OUT_CHANNELS.load(std::sync::atomic::Ordering::Relaxed) as usize != wanted || super::DEVICE_GEN.load(std::sync::atomic::Ordering::Relaxed) != generation {
                let _ = client.Stop();
                let _ = CloseHandle(event);
                return Ok(true);
            }
            let padding = match client.GetCurrentPadding() {
                Ok(p) => p,
                Err(e) => {
                    crate::warn!("audio: device lost ({e}); switching to the null device");
                    let _ = CloseHandle(event);
                    return Err(e.to_string());
                }
            };
            let avail = (total - padding) as usize;
            if avail == 0 {
                continue;
            }
            super::mix_into(&mut scratch, avail, och);
            if let Ok(p) = render.GetBuffer(avail as u32) {
                std::ptr::copy_nonoverlapping(scratch.as_ptr(), p as *mut f32, och * avail);
                let _ = render.ReleaseBuffer(avail as u32, 0);
            }
        }
        let _ = CloseHandle(event);
        Err("event wait failed".into())
    }
}

pub fn priority() {
    unsafe {
        use windows_sys::Win32::System::Threading::{GetCurrentThread, SetThreadPriority, THREAD_PRIORITY_HIGHEST};
        SetThreadPriority(GetCurrentThread(), THREAD_PRIORITY_HIGHEST);
    }
}
