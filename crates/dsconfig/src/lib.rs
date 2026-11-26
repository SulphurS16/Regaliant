//! What the port's runtime (`dsport`) and its launcher share: where the port keeps its files
//! (`paths`), the game folder's checks (`game`), the import of saves made elsewhere (`saves`).

pub mod display;
pub mod game;
pub mod paths;
pub mod saves;
pub mod self_file;
pub mod sha;
pub mod zar;

// (The release version is each program's own: the launcher's package version, the runtime's
// `REGALIANT_VERSION`. This crate's version stays 0.1.7, see its Cargo.toml.)

/// The time now in UTC as `YYYYMMDD-HHMMSS` (run folders and backups are named by it).
pub fn utc_stamp() -> String {
    let secs = std::time::SystemTime::now().duration_since(std::time::UNIX_EPOCH).map(|d| d.as_secs()).unwrap_or(0) as i64;
    stamp_of(secs)
}

/// `secs` since 1970 in UTC as `YYYYMMDD-HHMMSS`.
pub fn stamp_of(secs: i64) -> String {
    let days = secs.div_euclid(86400);
    let rem = secs.rem_euclid(86400);
    // Civil-from-days (Howard Hinnant's algorithm).
    let z = days + 719468;
    let era = z.div_euclid(146097);
    let doe = z - era * 146097;
    let yoe = (doe - doe / 1460 + doe / 36524 - doe / 146096) / 365;
    let y = yoe + era * 400;
    let doy = doe - (365 * yoe + yoe / 4 - yoe / 100);
    let mp = (5 * doy + 2) / 153;
    let d = doy - (153 * mp + 2) / 5 + 1;
    let m = if mp < 10 { mp + 3 } else { mp - 9 };
    let y = if m <= 2 { y + 1 } else { y };
    format!("{:04}{:02}{:02}-{:02}{:02}{:02}", y, m, d, rem / 3600, (rem / 60) % 60, rem % 60)
}

/// A `YYYYMMDD-HHMMSS` stamp shown as `YYYY-MM-DD HH:MM UTC` (anything else as it is).
pub fn stamp_text(s: &str) -> String {
    let b = s.as_bytes();
    if b.len() >= 15 && b[8] == b'-' && b[..8].iter().chain(&b[9..15]).all(u8::is_ascii_digit) {
        format!("{}-{}-{} {}:{} UTC", &s[..4], &s[4..6], &s[6..8], &s[9..11], &s[11..13])
    } else {
        s.to_string()
    }
}

#[cfg(test)]
mod tests {
    #[test]
    fn stamps() {
        assert_eq!(super::stamp_of(0), "19700101-000000");
        assert_eq!(super::stamp_of(1_793_000_000), "20261026-073320");
        assert_eq!(super::stamp_text("20261110-225234-before-restore"), "2026-11-10 22:52 UTC");
        assert_eq!(super::stamp_text("other"), "other");
    }
}
