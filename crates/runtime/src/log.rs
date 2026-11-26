//! Minimal process-wide logger: buffered file sink plus stderr for warnings/errors.

use parking_lot::Mutex;
use std::cell::RefCell;
use std::fmt;
use std::fs::File;
use std::io::{BufWriter, Write};
use std::path::Path;
use std::sync::OnceLock;
use std::sync::atomic::{AtomicBool, AtomicU8, Ordering};
use std::time::Instant;

#[derive(Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Debug)]
#[repr(u8)]
pub enum Level {
    Error = 0,
    Warn = 1,
    Info = 2,
    Debug = 3,
    Trace = 4,
}

pub static LEVEL: AtomicU8 = AtomicU8::new(Level::Info as u8);
/// When set, every HLE call is logged with its arguments and result.
pub static TRACE_HLE: AtomicBool = AtomicBool::new(false);

static SINK: Mutex<Option<BufWriter<File>>> = Mutex::new(None);
static START: OnceLock<Instant> = OnceLock::new();
/// The log file (`init`): the port's host adds a line there when the game ends at an exception
/// nothing handled (`hosted`).
static PATH: OnceLock<std::path::PathBuf> = OnceLock::new();

pub fn path() -> Option<&'static Path> {
    PATH.get().map(|p| p.as_path())
}

/// A tap of the lines written (`prepplay`'s self-play reads the log as the development drivers
/// do): while on, each line is kept, the last `TAP_KEEP`, with a running count.
static TAP_ON: AtomicBool = AtomicBool::new(false);
static TAP: Mutex<(u64, std::collections::VecDeque<String>)> = Mutex::new((0, std::collections::VecDeque::new()));
const TAP_KEEP: usize = 16_384;

/// Starts keeping the lines written from now on (`tap_read`).
pub fn tap_on() {
    TAP_ON.store(true, Ordering::Release);
}

/// The lines written since line `cursor` (a count of lines since the tap began; lines older
/// than the last `TAP_KEEP` are gone) and the cursor after them.
pub fn tap_read(cursor: u64) -> (Vec<String>, u64) {
    let g = TAP.lock();
    let (total, lines) = (&g.0, &g.1);
    let first = total - lines.len() as u64;
    let from = cursor.max(first);
    let out = lines.iter().skip((from - first) as usize).cloned().collect();
    (out, *total)
}

thread_local! {
    static THREAD_NAME: RefCell<String> = RefCell::new(String::new());
}

pub fn init(path: &Path) -> std::io::Result<()> {
    START.get_or_init(Instant::now);
    let _ = PATH.set(path.to_path_buf());
    let f = File::create(path)?;
    *SINK.lock() = Some(BufWriter::with_capacity(1 << 20, f));
    // Buffered lines reach the file within a quarter second (warnings are not flushed where they
    // are logged: one arrivals boot logs ~9,000 of them, many on the command processors).
    let _ = std::thread::Builder::new().name("log-flush".into()).spawn(|| loop {
        std::thread::sleep(std::time::Duration::from_millis(250));
        try_flush();
    });
    Ok(())
}

/// Warnings copied to stderr: the first ones only (each copy is a write on the thread that logs).
const STDERR_WARNINGS: u32 = 100;
static WARNINGS: std::sync::atomic::AtomicU32 = std::sync::atomic::AtomicU32::new(0);

pub fn set_thread_name(name: &str) {
    THREAD_NAME.with(|n| *n.borrow_mut() = name.to_string());
}

pub fn thread_name() -> String {
    THREAD_NAME.with(|n| {
        let n = n.borrow();
        if n.is_empty() {
            std::thread::current().name().unwrap_or("?").to_string()
        } else {
            n.clone()
        }
    })
}

#[inline]
pub fn enabled(level: Level) -> bool {
    level as u8 <= LEVEL.load(Ordering::Relaxed)
}

pub fn elapsed_ms() -> f64 {
    START.get_or_init(Instant::now).elapsed().as_secs_f64() * 1000.0
}

/// Personal data kept out of logs and crash reports (players attach them to public issues): the
/// user profile's path, the user and computer names (whole words). (pattern in lowercase,
/// replacement, whole words only.)
fn redactions() -> &'static [(String, &'static str, bool)] {
    static R: OnceLock<Vec<(String, &'static str, bool)>> = OnceLock::new();
    R.get_or_init(|| {
        let var = |n: &str| std::env::var(n).ok().map(|v| v.trim().to_ascii_lowercase()).filter(|v| v.len() >= 2);
        let mut v = Vec::new();
        if let Some(p) = var("USERPROFILE").filter(|p| p.len() >= 4) {
            let p = p.replace('/', "\\");
            // (Debug formatting prints paths with their backslashes doubled.)
            v.push((p.replace('\\', "\\\\"), "%USERPROFILE%", false));
            v.push((p.replace('\\', "/"), "%USERPROFILE%", false));
            v.push((p, "%USERPROFILE%", false));
        }
        if let Some(u) = var("USERNAME") {
            v.push((u, "<user>", true));
        }
        if let Some(c) = var("COMPUTERNAME") {
            v.push((c, "<computer>", true));
        }
        v
    })
}

/// `s` with personal data replaced (ASCII case-insensitive).
pub fn redact(s: &str) -> std::borrow::Cow<'_, str> {
    let r = redactions();
    if r.is_empty() {
        return std::borrow::Cow::Borrowed(s);
    }
    let mut hay = s.to_ascii_lowercase();
    if !r.iter().any(|(p, _, _)| hay.contains(p.as_str())) {
        return std::borrow::Cow::Borrowed(s);
    }
    let mut out = s.to_string();
    for (pat, rep, words) in r {
        if !hay.contains(pat.as_str()) {
            continue;
        }
        let word = |c: Option<char>| c.is_some_and(|c| c.is_ascii_alphanumeric() || c == '_');
        let mut res = String::with_capacity(out.len());
        let mut i = 0usize;
        while let Some(off) = hay[i..].find(pat.as_str()) {
            let at = i + off;
            let end = at + pat.len();
            let bounded = !*words || (!word(out[..at].chars().next_back()) && !word(out[end..].chars().next()));
            res.push_str(&out[i..at]);
            res.push_str(if bounded { rep } else { &out[at..end] });
            i = end;
        }
        res.push_str(&out[i..]);
        out = res;
        hay = out.to_ascii_lowercase();
    }
    std::borrow::Cow::Owned(out)
}

pub fn write(level: Level, args: fmt::Arguments) {
    let tag = match level {
        Level::Error => "E",
        Level::Warn => "W",
        Level::Info => "I",
        Level::Debug => "D",
        Level::Trace => "T",
    };
    let line = format!("[{:>10.3}][{}] {}: {}\n", elapsed_ms(), thread_name(), tag, args);
    let line = redact(&line);
    if TAP_ON.load(Ordering::Relaxed) {
        let mut t = TAP.lock();
        t.0 += 1;
        t.1.push_back(line.trim_end().to_string());
        if t.1.len() > TAP_KEEP {
            t.1.pop_front();
        }
    }
    // Errors go to stderr and the file at once; warnings to stderr only the first ones, and to
    // the file through its buffer (flushed every 250 ms, and by the crash handlers).
    let echo = level == Level::Error || (level == Level::Warn && WARNINGS.fetch_add(1, Ordering::Relaxed) < STDERR_WARNINGS);
    if echo {
        let _ = std::io::stderr().write_all(line.as_bytes());
    }
    let mut sink = SINK.lock();
    if let Some(w) = sink.as_mut() {
        let _ = w.write_all(line.as_bytes());
        if level == Level::Error {
            let _ = w.flush();
        }
    } else if !echo {
        let _ = std::io::stderr().write_all(line.as_bytes());
    }
}

pub fn flush() {
    if let Some(w) = SINK.lock().as_mut() {
        let _ = w.flush();
    }
}

/// Flush without blocking if another thread holds the sink (used from crash handlers).
pub fn try_flush() {
    if let Some(mut g) = SINK.try_lock() {
        if let Some(w) = g.as_mut() {
            let _ = w.flush();
        }
    }
}

#[macro_export]
macro_rules! error { ($($t:tt)*) => { $crate::log::write($crate::log::Level::Error, format_args!($($t)*)) } }
#[macro_export]
macro_rules! warn { ($($t:tt)*) => { if $crate::log::enabled($crate::log::Level::Warn) { $crate::log::write($crate::log::Level::Warn, format_args!($($t)*)) } } }
#[macro_export]
macro_rules! info { ($($t:tt)*) => { if $crate::log::enabled($crate::log::Level::Info) { $crate::log::write($crate::log::Level::Info, format_args!($($t)*)) } } }
#[macro_export]
macro_rules! debug { ($($t:tt)*) => { if $crate::log::enabled($crate::log::Level::Debug) { $crate::log::write($crate::log::Level::Debug, format_args!($($t)*)) } } }
#[macro_export]
macro_rules! trace { ($($t:tt)*) => { if $crate::log::enabled($crate::log::Level::Trace) { $crate::log::write($crate::log::Level::Trace, format_args!($($t)*)) } } }

#[cfg(test)]
mod tests {
    #[test]
    fn redaction() {
        let user = std::env::var("USERNAME").unwrap_or_default();
        let profile = std::env::var("USERPROFILE").unwrap_or_default();
        if user.len() < 2 || profile.len() < 4 {
            return;
        }
        let s = format!("save {}\\AppData\\Roaming\\X and {} (by {user}) {user}x", profile.to_ascii_uppercase(), profile.replace('\\', "/"));
        assert_eq!(super::redact(&s), format!("save %USERPROFILE%\\AppData\\Roaming\\X and %USERPROFILE% (by <user>) {user}x"));
        let dbg = format!("{:?}", vec![std::path::PathBuf::from(format!("{profile}\\AppData"))]);
        assert_eq!(super::redact(&dbg), "[\"%USERPROFILE%\\\\AppData\"]");
        assert!(matches!(super::redact("nothing personal"), std::borrow::Cow::Borrowed(_)));
    }
}
