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
