/// A player's diagnostic switches: the `DS_NAME=value` lines of `diagnostics.txt` in the settings
/// folder (`Paths::diagnostics_file`; `#` comments and other lines ignored), set in this
/// process's environment before anything reads them; the hosted game and the shader
/// preparation's sessions inherit them. A way to try a switch on a player's PC without a new
/// build (issue #4, 2026-11-24: RTX 50-series cards, which no PC here has). Returns what it set.
fn apply_diagnostics() -> Vec<String> {
    let Some(path) = dsconfig::paths::Paths::current().diagnostics_file() else { return Vec::new() };
    let bytes = match std::fs::read(&path) {
        Ok(b) => b,
        Err(e) if e.kind() == std::io::ErrorKind::NotFound => return Vec::new(),
        Err(e) => return vec![format!("not read ({e})")],
    };
    // (Notepad writes UTF-8, PowerShell 5.1's `>` UTF-16 with a byte-order mark, Out-File
    // UTF-8 with one: all are taken.)
    let text = if bytes.starts_with(&[0xFF, 0xFE]) {
        String::from_utf16_lossy(&bytes[2..].chunks_exact(2).map(|c| u16::from_le_bytes([c[0], c[1]])).collect::<Vec<u16>>())
    } else {
        String::from_utf8_lossy(&bytes).into_owned()
    };
    // (The hosted game process: the names its host set from this file, `DS_DIAGNOSTICS_SET`, are
    // the file's, not the environment's own.)
    let inherited: Vec<String> = std::env::var("DS_DIAGNOSTICS_SET").map(|v| v.split(';').filter(|s| !s.is_empty()).map(String::from).collect()).unwrap_or_default();
    let mut notes = Vec::new();
    let mut set = Vec::new();
    for line in text.lines().map(|l| l.trim_matches(|c: char| c.is_whitespace() || c == '\u{feff}')).filter(|l| !l.is_empty() && !l.starts_with('#')) {
        let Some((k, v)) = line.split_once('=') else {
            notes.push(format!("'{line}' left (no =)"));
            continue;
        };
        let (k, v) = (k.trim(), v.trim());
        let name_ok = k.len() > 3 && k.starts_with("DS_") && k.bytes().all(|b| b.is_ascii_uppercase() || b.is_ascii_digit() || b == b'_');
        if !name_ok || v.len() > 1024 || v.contains('\0') {
            notes.push(format!("'{k}' left (not a DS_ name)"));
            continue;
        }
        // (The host marks its game process with DS_HOSTED: from the file it would run the game
        // unhosted. DS_DIAGNOSTICS_SET is this function's own.)
        if k == "DS_HOSTED" || k == "DS_DIAGNOSTICS_SET" {
            notes.push(format!("'{k}' left (the port's own)"));
            continue;
        }
        // (Test switches stay out of the players' file: `DS_SIM_*` make the game fail on purpose,
        // and `DS_SIM_MODULE` loads the module it names; the review of 2026-11-25 evening.)
        if k.starts_with("DS_SIM_") {
            notes.push(format!("'{k}' left (a test switch)"));
            continue;
        }
        // (What the environment already holds wins: the launcher's settings, a developer's.)
        if std::env::var_os(k).is_some() {
            if inherited.iter().any(|n| n == k) {
                set.push((k.to_string(), std::env::var(k).unwrap_or_default()));
            } else {
                notes.push(format!("{k} left (already set to {:?})", std::env::var(k).unwrap_or_default()));
            }
            continue;
        }
        // SAFETY: at the start of `main`, before any thread of ours: nothing reads the environment
        // concurrently.
        unsafe { std::env::set_var(k, v) };
        set.push((k.to_string(), v.to_string()));
    }
    let mut out: Vec<String> = set.iter().map(|(k, v)| format!("{k}={v}")).collect();
    out.extend(notes);
    let names: Vec<String> = set.into_iter().map(|(k, _)| k).collect();
    // SAFETY: as above. (The hosted game inherits the list with the values.)
    unsafe { std::env::set_var("DS_DIAGNOSTICS_SET", names.join(";")) };
    DIAGNOSTICS_SET.set(names).ok();
    out
}

/// The names `apply_diagnostics` set: safe mode's and the shader preparation's own settings
/// replace them (`set_unless_given`).
static DIAGNOSTICS_SET: std::sync::OnceLock<Vec<String>> = std::sync::OnceLock::new();

/// Sets `k` to `v` unless the environment gave it a value of its own: a value from the
/// player's diagnostics.txt does not count (safe mode's and the self-play's settings win over
/// it). SAFETY: only before any thread of ours starts.
pub(crate) unsafe fn set_unless_given(k: &str, v: &str) {
    let from_file = DIAGNOSTICS_SET.get().is_some_and(|s| s.iter().any(|n| n == k));
    if std::env::var_os(k).is_none() || from_file {
        unsafe { std::env::set_var(k, v) };
    }
}


#[cfg(all(test, target_os = "linux"))]
mod diagnostics_tests {
    #[test]
    fn settings_file_and_environment_precedence() {
        if std::env::var_os("REGALIANT_DIAGNOSTICS_TEST").is_some() {
            let notes = super::apply_diagnostics();
            assert!(!notes.is_empty());
            assert_eq!(std::env::var("DS_DIAG_TEST_A").unwrap(), "file");
            assert_eq!(std::env::var("DS_DIAG_TEST_B").unwrap(), "environment");
            assert!(std::env::var_os("DS_HOSTED").is_none());
            assert!(std::env::var_os("NOT_A_SWITCH").is_none());
            assert!(std::env::var_os("DS_SIM_MODULE").is_none());
            unsafe { super::set_unless_given("DS_DIAG_TEST_A", "preparation"); }
            assert_eq!(std::env::var("DS_DIAG_TEST_A").unwrap(), "preparation");
            return;
        }
        let root = std::env::temp_dir().join(format!("regaliant-diagnostics-{}", std::process::id()));
        std::fs::create_dir_all(&root).unwrap();
        let text = "# comment\nDS_DIAG_TEST_A=file\nDS_DIAG_TEST_B=file\nDS_HOSTED=1\nDS_SIM_MODULE=should-not-load\nNOT_A_SWITCH=1\n";
        for utf16 in [false, true] {
            let bytes = if utf16 { [vec![0xff, 0xfe], text.encode_utf16().flat_map(u16::to_le_bytes).collect()].concat() } else { text.as_bytes().to_vec() };
            std::fs::write(root.join("diagnostics.txt"), bytes).unwrap();
            let out = std::process::Command::new(std::env::current_exe().unwrap())
                .args(["--exact", "diagnostics_tests::settings_file_and_environment_precedence", "--nocapture"])
                .env("REGALIANT_DIAGNOSTICS_TEST", &root).env("DSPORT_HOME", &root)
                .env("DS_DIAG_TEST_B", "environment").env_remove("DS_DIAG_TEST_A")
                .env_remove("DS_HOSTED").env_remove("DS_DIAGNOSTICS_SET").env_remove("NOT_A_SWITCH").env_remove("DS_SIM_MODULE")
                .output().unwrap();
            assert!(out.status.success(), "{}{}", String::from_utf8_lossy(&out.stdout), String::from_utf8_lossy(&out.stderr));
        }
        std::fs::remove_dir_all(root).unwrap();
    }
}
