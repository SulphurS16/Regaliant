/// Standard offset west of UTC and daylight-saving offset at the given UTC instant.
pub fn zone_at_utc(t: i64) -> Option<(i32, i32)> {
    let mut local: libc::tm = unsafe { std::mem::zeroed() };
    if unsafe { libc::localtime_r(&t, &mut local) }.is_null() { return None; }
    let offset = local.tm_gmtoff;
    let dst = if local.tm_isdst > 0 {
        // Ask libc for the standard-time interpretation; do not assume DST is one hour.
        local.tm_isdst = 0;
        let standard = unsafe { libc::mktime(&mut local) };
        standard.checked_sub(t)?
    } else { 0 };
    Some(((dst - offset).try_into().ok()?, dst.try_into().ok()?))
}
/// A save's timestamp in the host's local time zone.
pub fn local_time(unix: i64) -> String {
    if unix <= 0 { return String::new(); }
    let mut local: libc::tm = unsafe { std::mem::zeroed() };
    if unsafe { libc::localtime_r(&unix, &mut local) }.is_null() { return String::new(); }
    format!("{:04}-{:02}-{:02} {:02}:{:02}", local.tm_year + 1900, local.tm_mon + 1, local.tm_mday, local.tm_hour, local.tm_min)
}
