/// time_t <-> SYSTEMTIME through FILETIME (100 ns since 1601).
fn to_systemtime(t: i64) -> Option<windows_sys::Win32::Foundation::SYSTEMTIME> {
    use windows_sys::Win32::Foundation::{FILETIME, SYSTEMTIME};
    let ft100 = (t.checked_add(11_644_473_600)?).checked_mul(10_000_000)?;
    let ft = FILETIME { dwLowDateTime: ft100 as u32, dwHighDateTime: (ft100 >> 32) as u32 };
    let mut st: SYSTEMTIME = unsafe { std::mem::zeroed() };
    (unsafe { windows_sys::Win32::System::Time::FileTimeToSystemTime(&ft, &mut st) } != 0).then_some(st)
}

fn from_systemtime(st: &windows_sys::Win32::Foundation::SYSTEMTIME) -> Option<i64> {
    use windows_sys::Win32::Foundation::FILETIME;
    let mut ft = FILETIME { dwLowDateTime: 0, dwHighDateTime: 0 };
    if unsafe { windows_sys::Win32::System::Time::SystemTimeToFileTime(st, &mut ft) } == 0 {
        return None;
    }
    let v = ((ft.dwHighDateTime as i64) << 32) | ft.dwLowDateTime as i64;
    Some(v / 10_000_000 - 11_644_473_600)
}

/// Host time zone: (standard offset west of UTC in seconds, DST seconds at UTC time `t`).
pub fn zone_at_utc(t: i64) -> Option<(i32, i32)> {
    use windows_sys::Win32::System::Time::{GetTimeZoneInformation, SystemTimeToTzSpecificLocalTime, TIME_ZONE_INFORMATION};
    let mut tzi: TIME_ZONE_INFORMATION = unsafe { std::mem::zeroed() };
    unsafe { GetTimeZoneInformation(&mut tzi) };
    let utc = to_systemtime(t)?;
    let mut local = unsafe { std::mem::zeroed() };
    if unsafe { SystemTimeToTzSpecificLocalTime(std::ptr::null(), &utc, &mut local) } == 0 {
        return None;
    }
    let offset = from_systemtime(&local)? - t; // local - utc
    let west = tzi.Bias * 60; // UTC = local + bias (standard time)
    Some((west, (offset + west as i64) as i32))
}

