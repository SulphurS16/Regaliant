//! The Shaders page: the game's shaders prepared ahead, in two steps: the game plays by itself in
//! the background to learn them (`dsport --prepare-play newgame`, then `tour`: a new game, then
//! every level with its walks, fights and boss; no window, no sound, the player's controllers,
//! keyboard and mouse left alone; its saves and settings its own), and beside the play, on the
//! CPU it leaves idle, every compute program of the game's shader files is compiled into the
//! player's own pipeline store (`--precompile-all` on its own when the new game cannot be
//! played). Its progress, a stop that lets the current step end, and the option to prepare them
//! when Play is pressed the first time. Prepared shaders hold for one graphics card and driver (prepared.txt records them,
//! played.txt the self-play's stages) and for the game's translation of them: a new driver, or a
//! version of the port that translates the shaders differently (`dsport
//! --translation-fingerprint`, checked once at start on a thread of its own), asks for another
//! preparation.

use crate::sys;
use crate::ui::{self, Builder};
use crate::App;
use dsconfig::paths::read_kv;
use std::path::PathBuf;
use std::process::Child;
use std::time::Instant;
use windows_sys::Win32::Foundation::HWND;

const ID_PREPARE: i32 = 201;
const ID_STOP: i32 = 202;
const ID_AUTO: i32 = 203;
/// The timer that follows a preparation.
pub const TIMER: usize = 3;
/// The timer that waits for the translation check (`start_check`).
pub const CHECK_TIMER: usize = 4;

/// The self-play's exit status when the stop file ended it (`prepplay::EXIT_STOPPED`).
const STAGE_STOPPED: i32 = 12;
/// The game's exit status when another session holds the player's data folder
/// (`main::EXIT_RUNNING`): the step did not run.
const GAME_RUNNING: i32 = 9;

/// A step ended itself for nothing having moved in it (`prepplay::EXIT_HUNG`): begun once more.
const STAGE_HUNG: i32 = 14;
/// The shaders folder's drive is full (`prepplay::EXIT_DISK_FULL`): the preparation ends there.
const STAGE_DISK_FULL: i32 = 15;
/// The shaders folder refuses writes (`prepplay::EXIT_UNWRITABLE`): the preparation ends there.
const STAGE_UNWRITABLE: i32 = 16;
/// A levels part's session ended at a level's start for video memory (`prepplay::EXIT_CONTINUE`):
/// the part goes on in a new session from that level.
const STAGE_CONTINUE: i32 = 17;
/// Such new sessions in a preparation at most (each takes a minute or so of loading).
const MAX_CONTINUES: u8 = 40;

/// Whether a step's exit status is worth one more try: the graphics device lost
/// (`main::EXIT_DEVICE_LOST`, 5), memory exhausted (`EXIT_MEMORY`, 7), hung (14) or a crash (a
/// status the game does not give itself). Not: played (0), stopped (12), could not go on (13: it
/// would stop there again), the drive full (15), the game files, the CPU, a start too early, no
/// graphics card (3, 8, 10, 11), the game's modules not loaded (`EXIT_START`, 0xD502).
fn worth_retry(code: i32) -> bool {
    !matches!(code, 0 | 3 | 8 | 9 | 10 | 11 | STAGE_STOPPED | 13 | STAGE_DISK_FULL | STAGE_UNWRITABLE | 20 | 21 | 22 | 0xD502)
}

/// An exit status as the player is told it: small ones in decimal, Windows' exception codes in
/// hexadecimal (as the launcher shows a game's).
fn code_text(c: i32) -> String {
    if (0..=255).contains(&c) { c.to_string() } else { format!("{:#x}", c as u32) }
}

/// What a step's exit status means, in the player's words, with the status after it.
fn reason(code: i32) -> String {
    let what = match code {
        1 | 2 => "it ended with an error",
        3 => "the game's files could not be read",
        5 => "the graphics card stopped responding",
        7 => "Windows ran out of memory",
        8 => "this PC's processor lacks instructions the game needs",
        10 => "the game could not set itself up at its start, most often because another program holds memory it needs (Open logs shows why)",
        11 => "no graphics card the game can use was found",
        13 => "it could not go on",
        STAGE_HUNG => "it stopped responding",
        STAGE_DISK_FULL => "the drive is full",
        STAGE_UNWRITABLE => "the shaders folder cannot be written to",
        0xD502 => "the game could not start",
        _ => "it crashed",
    };
    format!("{what} (status {})", code_text(code))
}

/// Statuses no other step would end otherwise on this PC: the game's files, the processor, the
/// memory range, no graphics card, the game's start.
fn fatal(code: i32) -> bool {
    matches!(code, 3 | 8 | 10 | 11 | 0xD502)
}

/// A record written whole or not at all: beside it first, flushed to the drive, then put in its
/// place (a power loss while it was written would leave an empty file, read as nothing prepared:
/// the whole preparation again).
fn write_record(p: &std::path::Path, text: &str) {
    use std::io::Write;
    let tmp = p.with_extension("txt.new");
    let whole = std::fs::File::create(&tmp).and_then(|mut f| {
        f.write_all(text.as_bytes())?;
        f.sync_all()
    });
    if whole.and_then(|_| std::fs::rename(&tmp, p)).is_err() {
        let _ = std::fs::remove_file(&tmp);
        let _ = std::fs::write(p, text);
    }
}

/// The launcher's own log (`<logs>\launcher.log`): the preparation's steps, their ends and the
/// launcher's decisions, for a player's report (the review of 2026-11-20: none was kept). Kept
/// under ~1 MB (the older half dropped). No folder names.
pub fn log(app: &App, line: &str) {
    use std::io::Write;
    let p = app.paths.logs().join("launcher.log");
    let _ = std::fs::create_dir_all(app.paths.logs());
    if std::fs::metadata(&p).is_ok_and(|m| m.len() > 1 << 20) {
        if let Ok(b) = std::fs::read(&p) {
            let _ = std::fs::write(&p, &b[b.len() / 2..]);
        }
    }
    if let Ok(mut f) = std::fs::OpenOptions::new().create(true).append(true).open(&p) {
        let _ = writeln!(f, "{} {line}", utc_stamp());
    }
}

/// This build's translation fingerprint (`dsport --translation-fingerprint`): None while the
/// check runs, then its answer.
static CHECK: std::sync::Mutex<Option<Result<String, String>>> = std::sync::Mutex::new(None);
/// The game's version, graphics card and driver as its check reported them: the card and driver
/// the game itself uses (the launcher's own choice could differ on a PC with two graphics cards,
/// the review of 2026-11-20).
static GAME_ID: std::sync::Mutex<Option<(String, String, String)>> = std::sync::Mutex::new(None);

/// The game's files are of another version than the launcher (a new version unpacked while the
/// old game still ran keeps the old program): what the player is told.
fn version_mismatch() -> Option<String> {
    let g = GAME_ID.lock().unwrap_or_else(|e| e.into_inner());
    let (v, _, _) = g.as_ref()?;
    (v != dsconfig::VERSION).then(|| format!("The game's program is version {v}, the launcher {}. Close the game, then unpack the whole new version again. Nothing is prepared until then.", dsconfig::VERSION))
}
static CHECK_STARTED: std::sync::atomic::AtomicBool = std::sync::atomic::AtomicBool::new(false);
/// When the check began (`check_pending`'s deadline).
static CHECK_AT: std::sync::OnceLock<Instant> = std::sync::OnceLock::new();

/// Whether Play and Prepare now still wait for the check: begun, no answer, and not past the
/// longest a check takes with its reader's margin (an answer never coming, its thread gone, had
/// kept them waiting for the launcher's whole session; the review of 2026-11-21 night).
fn check_pending() -> bool {
    CHECK_STARTED.load(std::sync::atomic::Ordering::Relaxed) && checked_translation().is_none() && CHECK_AT.get().is_none_or(|t| t.elapsed().as_secs() < 200)
}

/// Asks the game for its translation fingerprint on a thread of its own (a few seconds).
pub fn start_check(app: &App) {
    if CHECK_STARTED.swap(true, std::sync::atomic::Ordering::Relaxed) {
        return;
    }
    let _ = CHECK_AT.set(Instant::now());
    let exe = dsconfig::paths::game_exe_in(&app.paths.exe_dir);
    let dir = app.paths.exe_dir.clone();
    // (`DS_SIM_CHECK_FAIL=1`, a test: the check gives no answer, as one whose driver call did not
    // return or that failed.)
    if std::env::var("DS_SIM_CHECK_FAIL").ok().as_deref() == Some("1") {
        *CHECK.lock().unwrap_or_else(|e| e.into_inner()) = Some(Err("no answer (DS_SIM_CHECK_FAIL, a test)".into()));
        return;
    }
    std::thread::spawn(move || {
        const CREATE_NO_WINDOW: u32 = 0x0800_0000;
        const BELOW_NORMAL_PRIORITY_CLASS: u32 = 0x0000_4000;
        let mut cmd = std::process::Command::new(&exe);
        cmd.arg("--translation-fingerprint").current_dir(&dir).stdin(std::process::Stdio::null()).stdout(std::process::Stdio::piped()).stderr(std::process::Stdio::null());
        // (In the launcher's job, and waited for 75 s at most: a check whose driver call does not
        // return ends itself after 60 s, and is ended here otherwise; until 2026-11-20 the
        // launcher would have waited for it for ever. Its output read only once it has ended.)
        let r = sys::spawn_in_job(&mut cmd, CREATE_NO_WINDOW | BELOW_NORMAL_PRIORITY_CLASS).map_err(|e| e.to_string()).and_then(|mut child| {
            // (Its output read on a thread of its own and waited for at most 10 s after the exit: a
            // process the check left behind holding the pipe had kept the read, and so the answer
            // and everything waiting for it, for ever; the review of 2026-11-21 night.)
            let (tx, rx) = std::sync::mpsc::channel::<String>();
            if let Some(mut so) = child.stdout.take() {
                std::thread::spawn(move || {
                    use std::io::Read;
                    let mut out = String::new();
                    let _ = so.read_to_string(&mut out);
                    let _ = tx.send(out);
                });
            }
            let t = Instant::now();
            let status = loop {
                match child.try_wait() {
                    Ok(Some(s)) => break Ok(s),
                    Ok(None) if t.elapsed().as_secs() < 165 => std::thread::sleep(std::time::Duration::from_millis(50)),
                    Ok(None) => {
                        let _ = child.kill();
                        break Err("no answer in 165 s".to_string());
                    }
                    Err(e) => break Err(e.to_string()),
                }
            }?;
            let out = rx.recv_timeout(std::time::Duration::from_secs(10)).unwrap_or_default();
            let get = |k: &str| out.lines().find_map(|l| l.trim().strip_prefix(k).map(|v| v.trim().to_string())).filter(|v| !v.is_empty());
            if let Some(v) = get("version = ") {
                *GAME_ID.lock().unwrap_or_else(|e| e.into_inner()) = Some((v, get("device = ").unwrap_or_default(), get("driver = ").unwrap_or_default()));
            }
            get("translation = ").ok_or_else(|| format!("no answer (exit code {:?})", status.code()))
        });
        *CHECK.lock().unwrap_or_else(|e| e.into_inner()) = Some(r);
    });
}

/// The prepared shaders' probe (`dsport --pb-check probe`: the store check's pipelines made again
/// from the store, nothing changed): its exit status once it has one (0 the driver takes them, 20
/// it refuses them now, else unknown). Run at the launcher's start once a store is prepared
/// (2026-11-20 night: this PC's driver refused ~3,400 stored pipelines, binaries and cache data
/// alike, after a Windows session event, with the same driver version and cache id, and the other
/// player's, an RTX 4070 SUPER, refused hundreds every session; the launcher had gone on saying
/// "prepared" while the game compiled as if nothing was).
static PROBE: std::sync::Mutex<Option<i32>> = std::sync::Mutex::new(None);
static PROBE_STARTED: std::sync::atomic::AtomicBool = std::sync::atomic::AtomicBool::new(false);
/// The probe's answer handled (or no probe needed).
static PROBE_HANDLED: std::sync::atomic::AtomicBool = std::sync::atomic::AtomicBool::new(false);
/// The translation check's line logged (`check_tick`).
static CHECK_LOGGED: std::sync::atomic::AtomicBool = std::sync::atomic::AtomicBool::new(false);

/// The longest the probe may take (4 small pipelines: a few seconds; ended after it).
const PROBE_SECS: u64 = 30;

/// Starts the probe on a thread of its own (after the translation check: one game process at a
/// time), ended after `PROBE_SECS`.
fn start_probe(app: &App) {
    if PROBE_STARTED.swap(true, std::sync::atomic::Ordering::Relaxed) {
        return;
    }
    let exe = dsconfig::paths::game_exe_in(&app.paths.exe_dir);
    let dir = app.paths.exe_dir.clone();
    std::thread::spawn(move || {
        const CREATE_NO_WINDOW: u32 = 0x0800_0000;
        const BELOW_NORMAL_PRIORITY_CLASS: u32 = 0x0000_4000;
        let mut cmd = std::process::Command::new(&exe);
        cmd.arg("--pb-check").arg("probe").current_dir(&dir).stdin(std::process::Stdio::null()).stdout(std::process::Stdio::null()).stderr(std::process::Stdio::null());
        let code = sys::spawn_in_job(&mut cmd, CREATE_NO_WINDOW | BELOW_NORMAL_PRIORITY_CLASS).ok().and_then(|mut c| {
            let t = Instant::now();
            loop {
                match c.try_wait() {
                    Ok(Some(s)) => break s.code(),
                    Ok(None) if t.elapsed().as_secs() < PROBE_SECS => std::thread::sleep(std::time::Duration::from_millis(50)),
                    _ => {
                        let _ = c.kill();
                        break None;
                    }
                }
            }
        });
        *PROBE.lock().unwrap_or_else(|e| e.into_inner()) = Some(code.unwrap_or(-1));
    });
}

/// Whether the probe runs and has not answered yet: Play and Prepare now wait for its answer with
/// the window live (`defer_play`, `start`). Until 2026-11-21 night they waited in `wait_probe` on
/// the window's own thread, up to a minute on a slow PC, the window "not responding" meanwhile.
fn probe_in_flight() -> bool {
    use std::sync::atomic::Ordering::Relaxed;
    PROBE_STARTED.load(Relaxed) && !PROBE_HANDLED.load(Relaxed) && PROBE.lock().unwrap_or_else(|e| e.into_inner()).is_none()
}

/// Whether Play or Prepare now waits for the probe: it runs, or it is due and starts now (the
/// check answered, nothing else running). The check's timer delivers its answer (`check_tick`).
fn probe_pending(app: &App) -> bool {
    use std::sync::atomic::Ordering::Relaxed;
    if PROBE_HANDLED.load(Relaxed) {
        return false;
    }
    if !PROBE_STARTED.load(Relaxed) {
        if checked_translation().is_none() || running(app) || app.running.is_some() || !probe_due(app) {
            return false;
        }
        start_probe(app);
    }
    probe_in_flight()
}

/// Waits for the probe to end (it is ended `PROBE_SECS` after its start, so it has an answer
/// then; the margin covers a slow start of its process, the review of 2026-11-21).
fn wait_probe() {
    let t = Instant::now();
    while PROBE_STARTED.load(std::sync::atomic::Ordering::Relaxed) && PROBE.lock().unwrap_or_else(|e| e.into_inner()).is_none() && t.elapsed().as_secs() < 2 * PROBE_SECS {
        std::thread::sleep(std::time::Duration::from_millis(50));
    }
}

/// A probe in flight waited for and its answer handled (the game started now, the launcher's own
/// choice or the player's, does not run beside it). Nothing when none was started.
pub fn settle_probe(app: &mut App) {
    use std::sync::atomic::Ordering::Relaxed;
    // (A probe not started yet runs now when it is due: Play's own wait may have come before the
    // translation check answered; the review of 2026-11-21.)
    if !PROBE_HANDLED.load(Relaxed) {
        ensure_probe(app);
    }
}

/// Whether the probe is due: the store holds prepared shaders the driver should take.
fn probe_due(app: &App) -> bool {
    version_mismatch().is_none() && compute_state(app).0 && matches!(check_state(app).as_deref(), Some("binaries" | "cache")) && outcome(app).is_none()
}

/// Before a preparation begins and before Play starts the game: the probe run when it is due and
/// its answer handled, so that a store the driver refuses is set aside first and the probe never
/// runs beside a step or the game (the review of 2026-11-21: started by the check's timer alone,
/// it could run beside the first step of a preparation begun during the check, or beside the
/// game Play had started meanwhile, and a refusal found then was never acted on).
fn ensure_probe(app: &mut App) {
    use std::sync::atomic::Ordering::Relaxed;
    if let Some(r) = checked_translation() {
        log_check_line(app, &r);
    }
    if PROBE_HANDLED.load(Relaxed) {
        return;
    }
    if !PROBE_STARTED.load(Relaxed) {
        // (The check not answered yet, a slow drive's: the store cannot be judged, and the probe
        // would run beside the check. Left to the check's timer, which starts it only when
        // nothing runs; the review of 2026-11-21.)
        if checked_translation().is_none() {
            return;
        }
        if !probe_due(app) {
            PROBE_HANDLED.store(true, Relaxed);
            return;
        }
        start_probe(app);
    }
    wait_probe();
    let probe = *PROBE.lock().unwrap_or_else(|e| e.into_inner());
    handle_probe(app, probe.unwrap_or(-1));
}

/// The store's refusals (`store-refused.txt`): the times (Unix seconds, UTC stamps) the probe
/// found the driver refusing the prepared shaders, the last three.
fn refused_file(app: &App) -> PathBuf {
    app.paths.shaders().join("store-refused.txt")
}

fn refusals(app: &App) -> Vec<(u64, String)> {
    read_kv(&refused_file(app))
        .into_iter()
        .filter(|(k, _)| k == "refused")
        .filter_map(|(_, v)| {
            let mut w = v.split_whitespace();
            Some((w.next()?.parse().ok()?, w.next()?.to_string()))
        })
        .collect()
}

/// Whether the driver refused the prepared shaders after the marker's date (a `YYYYMMDD-HHMMSS`
/// stamp): the store is stale until a preparation makes it again.
fn refused_after(app: &App, date: &str) -> bool {
    !date.is_empty() && refusals(app).iter().any(|(_, d)| d.as_str() > date)
}

/// The probe's answer: a refusal is recorded, the store check's record dropped (the check runs
/// again before the next preparation and stores its pipelines again), and the page says the
/// shaders must be prepared again; a second refusal within three days ends preparing on this PC
/// for now (the outcome record: Play starts the game).
fn handle_probe(app: &mut App, code: i32) {
    use std::sync::atomic::Ordering::Relaxed;
    if PROBE_HANDLED.swap(true, Relaxed) {
        return;
    }
    let what = match code {
        0 => "the driver takes them",
        20 => "the driver refuses them now",
        22 => "the store check's pipelines are not in the store",
        9 => "a game session runs",
        _ => "no answer",
    };
    log(app, &format!("the prepared shaders' probe: status {} ({what})", code_text(code)));
    // (Nothing to probe with: the check runs again at the next preparation and stores them, so
    // that later probes can tell; the store itself is taken as it is.)
    if code == 22 {
        let _ = std::fs::remove_file(check_file(app));
        log(app, "the store check runs again at the next preparation");
        return;
    }
    if code != 20 {
        return;
    }
    let now = std::time::SystemTime::now().duration_since(std::time::UNIX_EPOCH).map_or(0, |d| d.as_secs());
    let mut r = refusals(app);
    let again = r.iter().any(|(t, _)| now.saturating_sub(*t) < 3 * 24 * 3600);
    r.push((now, utc_stamp()));
    let keep = &r[r.len().saturating_sub(3)..];
    let lines: String = keep.iter().map(|(t, d)| format!("refused = {t} {d}\n")).collect();
    write_record(
        &refused_file(app),
        &format!("# When the graphics driver refused the prepared shaders (the launcher's probe at its start: the store check's pipelines made again from the store); written by the launcher\n{lines}"),
    );
    let _ = std::fs::remove_file(check_file(app));
    if again {
        record_outcome(app, "its graphics driver discarded the prepared shaders twice in three days, so preparing them again would not last");
    } else {
        log(app, "the prepared shaders are stale: the next preparation compiles them again, without the game playing again");
    }
    refresh(app);
    app.update_play();
}

/// The launcher's start line with the game's check, once: from the check's timer, or first from
/// `ensure_probe` when Play or Prepare now came before the timer saw the answer (the probe's line
/// had come before it, 2026-11-21).
fn log_check_line(app: &App, r: &Result<String, String>) {
    static DONE: std::sync::atomic::AtomicBool = std::sync::atomic::AtomicBool::new(false);
    if DONE.swap(true, std::sync::atomic::Ordering::Relaxed) {
        return;
    }
    let id = GAME_ID.lock().unwrap_or_else(|e| e.into_inner()).clone();
    log(app, &format!("launcher {} started; the game's check: translation {}; game {}", dsconfig::VERSION, r.as_deref().unwrap_or_else(|e| e), id.map_or("unknown".to_string(), |(v, d, r)| if d.is_empty() { format!("{v}, no graphics card reported") } else { format!("{v} on {d}, driver {r}") })));
}

/// The check's answer once it has one.
fn checked_translation() -> Option<Result<String, String>> {
    CHECK.lock().unwrap_or_else(|e| e.into_inner()).clone()
}

/// The check timer: the page shows the check's outcome once it has one.
pub fn check_tick(app: &mut App) {
    use std::sync::atomic::Ordering::Relaxed;
    // (The probe's answer, once the check's line is logged; the timer stops when both are done.)
    if CHECK_LOGGED.load(Relaxed) {
        let probe = *PROBE.lock().unwrap_or_else(|e| e.into_inner());
        if let Some(code) = probe {
            handle_probe(app, code);
        }
        if PROBE_HANDLED.load(Relaxed) {
            unsafe { windows_sys::Win32::UI::WindowsAndMessaging::KillTimer(app.hwnd, CHECK_TIMER) };
            run_pending(app, "the prepared shaders' probe answered");
        }
        return;
    }
    if let Some(r) = checked_translation() {
        CHECK_LOGGED.store(true, Relaxed);
        log_check_line(app, &r);
        if let Some(t) = version_mismatch() {
            log(app, &t);
            app.set_status(&t);
        }
        refresh(app);
        app.update_play();
        // (The store's probe when it holds prepared shaders the driver should take; else none. Not
        // beside a preparation or the game: those began after `ensure_probe`, which ran it first,
        // or (Play without preparing) the game would turn it away.)
        if !PROBE_HANDLED.load(Relaxed) && !running(app) && app.running.is_none() && probe_due(app) {
            start_probe(app);
        } else if !PROBE_STARTED.load(Relaxed) {
            PROBE_HANDLED.store(true, Relaxed);
        }
        // (A preparation asked for or Play pressed during the check: begun now, the store judged
        // with the check's translation; with the probe just started, once it answers.)
        if !probe_in_flight() {
            run_pending(app, "the game's check answered");
        }
    }
}

/// A preparation asked for, or Play pressed, while the game's check or the probe had not answered:
/// begun now that they have (`why` says which answered).
fn run_pending(app: &mut App, why: &str) {
    // (Play pressed too: the preparation begins with the game to start after it, Play's own
    // question not asked; the review of 2026-11-21 night: Play then asked whether to stop it.)
    let play = std::mem::take(&mut app.shaders.pending_play);
    if let Some(play_after) = app.shaders.pending_start.take() {
        log(app, &format!("{why}: the preparation asked for meanwhile begins{}", if play { ", the game after it (Play pressed too)" } else { "" }));
        start(app, play_after || play);
        // (A start that went no further (a version mismatch, the player declining for lack of
        // space, the game's program missing): the page's own text again, not "Checking the game
        // first" with Prepare now off; the review of 2026-11-21 night.)
        if !running(app) && app.shaders.pending_start.is_none() {
            refresh(app);
            app.update_play();
        }
        return;
    }
    if play {
        log(app, &format!("{why}: Play pressed meanwhile goes on"));
        crate::play(app);
    }
}

/// The preparation's steps, in order (the compute step only when the new game failed).
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum Step {
    /// The compute programs compiled on their own (`--precompile-all`).
    Compute,
    /// The game plays a new game by itself (`--prepare-play newgame`).
    NewGame,
    /// The game visits the i-th of `LEVEL_PARTS` parts of its levels by itself (`--prepare-play
    /// tour --prepare-chunk i/n`, the save `NewGame` made): each part its own session, so that
    /// the memory a session's level changes take stays bounded.
    Levels(u8),
    /// The check before the first preparation on a card and driver (`--pb-check write`, then
    /// `read` in the next process): that what one process stores the next one takes without
    /// compiling (the other PC's driver refused 259 of the 262 pipelines its preparation had
    /// stored, 2026-11-19: an hour for nothing). A refusal turns the store to cache data (the game
    /// writes that), checked again; when that does not serve either, preparing cannot help on
    /// this PC and is not done (`store-check.txt`).
    CheckWrite,
    CheckRead,
}

/// The levels' parts (sessions).
const LEVEL_PARTS: u8 = 3;

/// A step's attempts in a preparation when it fails for the PC's sake (`worth_retry`): two more
/// tries (one until 2026-11-21: the other PC's last part hung, ran again, met NVIDIA's compiler
/// crash, and the preparation ended unfinished after 2.8 hours).
const MAX_ATTEMPTS: u8 = 3;
/// Attempts begun again in a preparation for crashes that added a pipeline to the compiler-crash
/// list (the game makes no such pipeline again, so each crash is another pipeline: bounded by the
/// pipelines a driver cannot compile; this PC's driver crashed on 7 in a day of boots).
const MAX_CRASH_RETRIES: u8 = 25;

/// The lines of the store's compiler-crash list (`crashed.txt`, the game's: the pipelines whose
/// compile crashed this card's driver), 0 when there is none.
fn crashed_lines(app: &App) -> usize {
    // (With the suspects a session left, `crash-suspect.txt`: a step that died on a driver fault
    // it had not confirmed yet; the next session confirms them.)
    let dir = app.paths.shaders().join("pipelines");
    ["crashed.txt", "crash-suspect.txt"].iter().map(|f| std::fs::read_to_string(dir.join(f)).map_or(0, |t| t.lines().filter(|l| !l.trim().is_empty()).count())).sum()
}

/// Whether the compute list is made for this translation, card and driver, as the steps' order
/// decides it: never when the store could not be judged at the preparation's start
/// (`untrusted`), as its marker may describe pipelines the store no longer serves.
fn compute_done(app: &App) -> bool {
    !app.shaders.untrusted && compute_state(app).0
}

/// The steps' times on the development PC (2026-11-20 night: the whole preparation through the
/// launcher on a fresh store, 52.6 minutes: the new game 401 s, the compute step 1,111 s on 23
/// threads, the levels' parts 373, 684 and 585 s): the yardsticks of the whole preparation's
/// estimate (`estimate`). The preparation has no deadline since 2026-11-20 (the user: slower PCs
/// get every level and every compute program too, warned of the time): it plays every level and
/// makes the whole compute list. Since 2026-11-20 evening the compute step comes between the new
/// game and the levels: the list's rest (what the new game's play left) on its own, so that the
/// levels' first dispatches find their pipelines in the store.
const REF_NEWGAME: f64 = 401.0;
const REF_PART: f64 = 547.0;
const REF_COMPUTE: f64 = 1111.0;
/// The aim for the whole preparation: past it (and a tenth) the page says this PC takes longer.
const AIM_SECS: f64 = 3600.0;
/// A step whose progress has not changed for this long is ended (a backstop: the steps end
/// themselves sooner when nothing moves in them, `prepplay::watch_hangs` and the compute step's
/// watch; 45 minutes until 2026-11-20).
const STALL_SECS: u64 = 15 * 60;

/// The no-progress limit (`STALL_SECS`; `DS_SIM_STALL_SECS=<s>`, a test: that many seconds).
fn stall_secs() -> u64 {
    std::env::var("DS_SIM_STALL_SECS").ok().and_then(|v| v.parse().ok()).unwrap_or(STALL_SECS)
}

/// Ends the step's process (`DS_SIM_UNKILLABLE=1`, a test: does nothing, as for a process the
/// graphics driver holds).
fn kill_step(c: &mut Child) {
    if std::env::var("DS_SIM_UNKILLABLE").ok().as_deref() != Some("1") {
        let _ = c.kill();
    }
}

impl Step {
    /// The self-play stage (None: the compute step).
    fn stage(self) -> Option<&'static str> {
        match self {
            Step::Compute | Step::CheckWrite | Step::CheckRead => None,
            Step::NewGame => Some("newgame"),
            Step::Levels(_) => Some("tour"),
        }
    }

    /// The session's time limit (a stage that hangs ends; generous: a slower PC's parts take
    /// hours, and the progress watchdog, `STALL_SECS`, ends a hung one long before).
    fn seconds(self) -> u64 {
        match self {
            Step::Compute | Step::CheckWrite | Step::CheckRead => 0,
            Step::NewGame => 4 * 3600,
            Step::Levels(_) => 8 * 3600,
        }
    }

    /// The step's time on the development PC (`REF_*`).
    fn yardstick(self) -> f64 {
        match self {
            Step::Compute => REF_COMPUTE,
            Step::CheckWrite | Step::CheckRead => 10.0,
            Step::NewGame => REF_NEWGAME,
            Step::Levels(_) => REF_PART,
        }
    }

    /// Its number among the self-play's steps (1 the new game).
    fn number(self) -> u8 {
        match self {
            Step::Levels(i) => i + 1,
            _ => 1,
        }
    }
}

#[derive(Default)]
pub struct Page {
    status: usize,
    bar: usize,
    prepare: usize,
    stop: usize,
    auto: usize,
    child: Option<Child>,
    /// The step the child runs.
    step: Option<Step>,
    /// When the step began.
    started: Option<Instant>,
    /// The game starts when the preparation is done (it began from Play).
    pub play_after: bool,
    /// A stop was asked for (the child finishes its current shaders).
    stopping: Option<Instant>,
    /// The game starts once the stop is done.
    then_play: bool,
    /// When this preparation began and the levels parts' times in it (the estimate, `estimate`).
    prep_started: Option<Instant>,
    part_secs: Vec<f64>,
    /// The progress file's contents last seen and when they changed (`STALL_SECS`).
    progress_seen: Option<(String, Instant)>,
    /// The slower-PC note was shown in the launcher's status line (once a preparation).
    slow_noted: bool,
    /// The compute step's time (the estimate's speed before a part of the levels is played) and
    /// the step after it (the levels' next part; None: the compute step is the last, after a new
    /// game that could not be played).
    compute_secs: Option<f64>,
    after_compute: Option<Step>,
    /// Each step's failed attempts in this preparation (hung, crashed, the device lost, memory:
    /// `worth_retry`), up to `MAX_ATTEMPTS`; a crash that added a pipeline to the compiler-crash
    /// list (`crashed_lines`) is not counted, as the next attempt leaves that pipeline out.
    attempts: Vec<(Step, u8)>,
    /// The compiler-crash list's length when the running step began, and the attempts begun again
    /// for such crashes in this preparation (`MAX_CRASH_RETRIES`).
    crashed_at_begin: usize,
    /// The new game runs again without its background compute list (`DS_PREPARE_BACKGROUND=0`):
    /// set when an attempt of it ended in the driver's shader compiler. The list's compiles run
    /// in the game's own process, so a crash among them cost the whole new game (2026-11-21
    /// night, an injected crash in one pipeline of 4,000: four attempts lost after 771, 35, 197
    /// and 133 s, every crash in a list worker); the compute step after it makes the list in a
    /// process of its own, begun again where it was.
    newgame_no_list: bool,
    crash_retries: u8,
    /// A preparation asked for while the game's check had not answered (its `play_after`): begun
    /// by the check's timer once the answer comes (`check_tick`), so that the store is judged with
    /// the check's translation (the other PC, 2026-11-21: Prepare pressed 27 s into its 50 s check
    /// left an old marker trusted, the compute step was skipped and the levels compiled it).
    pending_start: Option<bool>,
    /// Play pressed while the game's check had not answered: Play runs again once it answers
    /// (`defer_play`, `check_tick`), so that it judges the store with the check's translation.
    pending_play: bool,
    /// The step begins again after a crash or a hang: a levels part goes on from the level its
    /// last attempt reached (`<progress>.resume`, the game's), which `begin` keeps.
    resuming: bool,
    /// Levels parts gone on in a new session for video memory in this preparation
    /// (`STAGE_CONTINUE`, at most `MAX_CONTINUES`).
    continues: u8,
    /// The step the preparation goes on with after its check, and the check's round (2: again,
    /// with the store turned to cache data).
    after_check: Option<Step>,
    check_round: u8,
    /// A step begun again at this time: it met its own previous step's session still ending
    /// (the one-session lock), `retries` times so far.
    retry: Option<(Step, Instant)>,
    retries: u32,
    /// When the last step's child ended.
    ended: Option<Instant>,
    /// The step's child was ended for its progress unchanged for `STALL_SECS`.
    stalled: bool,
    /// The store's stale pipelines could not be set aside (`retire_stale_store`): the steps take
    /// no stored pipeline as made (`DS_STORE_UNTRUSTED`).
    untrusted: bool,
    /// The player declined preparing for lack of space in this launcher session: Play starts the
    /// game.
    pub space_declined: bool,
    /// The timer's last tick (a gap of over 10 s: the PC slept, the watchdog's time starts again).
    last_tick: Option<Instant>,
    /// The step before failed (hung, crashed, the graphics card lost, memory), after its one more
    /// try: one more failure ends the preparation (`tick`).
    last_failed: bool,
    /// When the launcher ended the step's process (a stop past 90 s, no progress): one that has
    /// not ended 30 s later is left behind (`tick`).
    killed_at: Option<Instant>,
    /// The preparation was stopped for a low battery (`tick`): the page says so.
    battery_stop: bool,
    /// Low battery readings in a row while a step runs (`tick`: three, 1.5 s, stop it).
    battery_low: u8,
    /// Play started the game without preparing, on battery (`note_battery`): the page and the
    /// status line after the game say so while the PC is still on battery and nothing is
    /// prepared (the review of 2026-11-21: the status line's note was replaced at once).
    battery_note: bool,
}

/// The battery's charge at which a preparation does not begin or stops (an hour of the PC's full
/// load, hidden, would empty a laptop's battery; 2026-11-21).
const BATTERY_LOW: u8 = 20;
/// What the page says when the battery stopped or held back a preparation.
const BATTERY_NOTE: &str = "The PC runs on battery, so the shaders are prepared later. Plug it in, then press Prepare now or Play.";

/// On battery when Play would prepare first: the game starts now (the preparation's hour of full
/// load would come before it), and the page says how to prepare later.
pub fn note_battery(app: &mut App, pct: u8) {
    log(app, &format!("preparation not begun on Play: the PC runs on battery ({pct}%)"));
    app.shaders.battery_note = true;
    app.set_status(BATTERY_NOTE);
    refresh(app);
}

/// The battery note while it holds: Play left the shaders unprepared on battery, the PC is still
/// on battery and nothing is prepared.
pub fn battery_note(app: &App) -> Option<&'static str> {
    (app.shaders.battery_note && sys::battery().is_some() && !state(app).0).then_some(BATTERY_NOTE)
}

fn h(v: usize) -> HWND {
    v as HWND
}

fn progress_file(app: &App) -> PathBuf {
    app.paths.shaders().join("prepare-progress.txt")
}

fn marker(app: &App) -> PathBuf {
    app.paths.shaders().join("prepared.txt")
}

/// The self-play's stages and how they ended (`record`).
fn played_file(app: &App) -> PathBuf {
    app.paths.shaders().join("played.txt")
}

/// The translation, card and driver the store's compiled pipelines were made for (written at
/// each preparation's start, `retire_stale_store`).
fn store_identity_file(app: &App) -> PathBuf {
    app.paths.shaders().join("store-identity.txt")
}

/// At a preparation's start: the store's compiled pipelines set aside when they were made for
/// another translation, card or driver (the identity recorded at the last preparation's start,
/// else the prepared marker's, else the self-play record's). They are of no use then (each would
/// be compiled again at its first use), and the preparation's steps take a pipeline the store
/// holds as made (`compbench::work_list`: after a driver update or a new version nothing would
/// have been made again, the release review of 2026-11-20). The folder is renamed (at once)
/// and deleted on a thread of its own; its binaries mode file (the store check's) is kept.
fn retire_stale_store(app: &App) -> bool {
    let (translation, device, driver) = identity();
    // (The game's check did not answer, or not yet: the store's pipelines cannot be told stale,
    // so none is taken as made (each goes through the store's own check, which makes a stale one
    // again; the release review of 2026-11-20 night: after a driver update a slow check had left
    // the stale ones trusted).)
    if translation.is_empty() || device.is_empty() {
        log(app, "the game's check gave no translation or graphics card: no stored pipeline taken as made");
        return false;
    }
    let mut trusted = true;
    let shaders = app.paths.shaders();
    let get = |kv: &[(String, String)], k: &str| kv.iter().find(|(x, _)| x == k).map(|(_, v)| v.clone()).unwrap_or_default();
    let recorded = [store_identity_file(app), marker(app), played_file(app)].into_iter().map(|f| read_kv(&f)).find(|kv| !kv.is_empty());
    // (Refused by the driver since the marker was written: stale whatever the identity says.)
    let refused = refused_after(app, &get(&read_kv(&marker(app)), "date"));
    // (Why the store's pipelines were set aside, while none are prepared again: kept across the
    // identity's rewrites, for the page's text.)
    let mut set_aside = if read_kv(&marker(app)).is_empty() { set_aside_reason(app) } else { None };
    if let Some(kv) = recorded {
        let dir = shaders.join("pipelines");
        // (Pipelines kept as the driver's binaries while the game keeps cache data, binaries
        // turned off (`DS_PIPELINE_BINARIES=0`): none of them serves, whatever the records say.)
        let form_stale = !binaries_form() && dir_has_binaries(&dir);
        let same_identity = get(&kv, "translation") == translation && get(&kv, "device") == device && get(&kv, "driver") == driver && !refused;
        // (The same translation, card and driver, the binaries alone stale: only their folder is
        // set aside, the store's cache data kept (the review of 2026-11-21 night: the whole folder
        // went, a mixed store's usable cache data with it), and the marker dropped, so that the
        // compute step makes what the cache data lacks.)
        if same_identity && form_stale {
            let old = shaders.join(format!("pipelines-old-{}", utc_stamp()));
            if std::fs::create_dir_all(&old).is_ok() && std::fs::rename(dir.join("binaries"), old.join("binaries")).is_ok() {
                log(app, "the stored pipelines' binaries, which this version no longer keeps, set aside; their cache data kept");
                let _ = std::fs::remove_file(marker(app));
                set_aside = Some("form".to_string());
            } else {
                log(app, "the stored pipelines' binaries could not be set aside: none taken as made");
                trusted = false;
            }
        }
        let same = same_identity;
        if !same && dir.is_dir() {
            let mode = std::fs::read(dir.join("binaries-mode.txt")).ok();
            // (The compiler-crash list holds for the driver's cache id its lines name, whatever
            // the store's translation: kept, so a store set aside meets those crashes no more;
            // the review of 2026-11-21 night.)
            let crashed = std::fs::read(dir.join("crashed.txt")).ok();
            let old = shaders.join(format!("pipelines-old-{}", utc_stamp()));
            if std::fs::rename(&dir, &old).is_ok() {
                log(
                    app,
                    if refused {
                        "the stored pipelines are refused by the graphics driver now: set aside"
                    } else if form_stale {
                        "the stored pipelines are the driver's binaries, which this version no longer keeps: set aside"
                    } else {
                        "the stored pipelines were made for another translation, card or driver: set aside"
                    },
                );
                if let Some(m) = mode {
                    let _ = std::fs::create_dir_all(&dir);
                    let _ = std::fs::write(dir.join("binaries-mode.txt"), m);
                }
                if let Some(c) = crashed {
                    let _ = std::fs::create_dir_all(&dir);
                    let _ = std::fs::write(dir.join("crashed.txt"), c);
                }
                // (The marker described the pipelines set aside: none are prepared until the
                // compile is done again, and a preparation stopped before then goes on from
                // there at the next start (the review of 2026-11-21: the marker had stayed, a
                // refusal after its date set every new start's store aside again, and the store
                // check's record, kept, left the new store without the check's pipelines, so
                // the probe could no longer tell). The check runs again and stores them.)
                let _ = std::fs::remove_file(marker(app));
                let _ = std::fs::remove_file(check_file(app));
                set_aside = Some(
                    if refused {
                        "refused"
                    } else if form_stale {
                        "form"
                    } else if get(&kv, "translation") != translation {
                        "translation"
                    } else if get(&kv, "device") != device {
                        "card"
                    } else {
                        "driver"
                    }
                    .to_string(),
                );
            } else {
                // (A file in it held open, e.g. by a game or a virus scanner: its pipelines stay,
                // and the steps take none of them as made; the identity stays the old one, so the
                // next preparation tries again.)
                log(app, "the stored pipelines were made for another translation, card or driver, but could not be set aside: none taken as made");
                trusted = false;
            }
        }
    }
    if !trusted {
        return false;
    }
    // (Folders set aside before, this one included: deleted in the background.)
    let olds: Vec<PathBuf> = std::fs::read_dir(&shaders).map(|r| r.flatten().map(|e| e.path()).filter(|p| p.is_dir() && p.file_name().is_some_and(|n| n.to_string_lossy().starts_with("pipelines-old-"))).collect()).unwrap_or_default();
    if !olds.is_empty() {
        let _ = std::thread::Builder::new().name("old-pipelines".into()).spawn(move || {
            for o in olds {
                let _ = std::fs::remove_dir_all(o);
            }
        });
    }
    let why = set_aside.map_or(String::new(), |w| format!("set_aside = {w}\n"));
    write_record(
        &store_identity_file(app),
        &format!("# The translation, graphics card and driver the stored pipelines are made for; written by the launcher at a preparation's start\ntranslation = {translation}\ndevice = {device}\ndriver = {driver}\n{why}"),
    );
    true
}

/// The resume record (`<progress>.resume`, the game's: `chunk=i/n`, `level=k`) moved one level on:
/// a part that hung at a level goes on from the next one.
fn resume_next_level(app: &App) {
    let f = progress_file(app).with_extension("resume");
    let Ok(t) = std::fs::read_to_string(&f) else { return };
    let out: String = t
        .lines()
        .map(|l| match l.trim().strip_prefix("level=").and_then(|v| v.parse::<u64>().ok()) {
            Some(k) => format!("level={}\n", k + 1),
            None => format!("{l}\n"),
        })
        .collect();
    let _ = std::fs::write(&f, out);
    log(app, "the part that hung goes on from the level after the one it hung in");
}

/// Whether the game keeps the driver's pipeline binaries (unless `DS_PIPELINE_BINARIES=0`, as the
/// game's `vk::pipeline_binaries` decides; cache data then).
fn binaries_form() -> bool {
    std::env::var("DS_PIPELINE_BINARIES").ok().as_deref() != Some("0")
}

/// Whether the store's pipelines folder holds pipeline binaries (its `binaries` folder not empty).
fn dir_has_binaries(dir: &std::path::Path) -> bool {
    std::fs::read_dir(dir.join("binaries")).is_ok_and(|mut r| r.next().is_some())
}

/// Why the store's pipelines were last set aside (`retire_stale_store`: refused, form,
/// translation, card, driver), from the identity record.
fn set_aside_reason(app: &App) -> Option<String> {
    read_kv(&store_identity_file(app)).into_iter().find(|(k, _)| k == "set_aside").map(|(_, v)| v).filter(|v| !v.is_empty())
}

/// Whether the compute shaders are prepared for this graphics card, driver and translation; the
/// status line.
fn compute_state(app: &App) -> (bool, String) {
    // (They are prepared from the game's files: nothing to do before a folder is chosen.)
    if !app.game.ok {
        return (false, "Choose the game's folder on the Game page first.".into());
    }
    let kv = read_kv(&marker(app));
    if kv.is_empty() {
        // (Set aside at a preparation's start and not compiled again yet: why.)
        return (
            false,
            match set_aside_reason(app).as_deref() {
                Some("refused") => "The graphics driver no longer takes the prepared shaders. Prepare them again.",
                Some("form") => "This version keeps prepared shaders in another form. Prepare them again.",
                Some("translation") => "The game was updated since the shaders were prepared. Prepare them again.",
                Some("card") => "The graphics card changed since the shaders were prepared. Prepare them again.",
                Some("driver") => "The graphics driver changed since the shaders were prepared. Prepare them again.",
                _ => "The shaders are not prepared on this PC yet.",
            }
            .into(),
        );
    }
    let get = |k: &str| kv.iter().find(|(x, _)| x == k).map(|(_, v)| v.clone()).unwrap_or_default();
    let date = get("date");
    let date = if date.len() >= 8 { format!("{}-{}-{}", &date[..4], &date[4..6], &date[6..8]) } else { date };
    // (A marker from before the translation fingerprint, or from a version of the port that
    // translates the shaders differently: its pipelines no longer match.)
    let updated = (false, "The game was updated since the shaders were prepared. Prepare them again.".to_string());
    let translation = get("translation");
    if translation.is_empty() {
        return updated;
    }
    if let Some(Ok(current)) = checked_translation() {
        if current != translation {
            return updated;
        }
    }
    let (_, device, driver) = identity();
    if device.is_empty() {
        return (false, sys::gpu().err().unwrap_or_else(|| "The graphics card could not be read.".into()));
    }
    // (The driver refused them since they were made: `handle_probe`.)
    if refused_after(app, &get("date")) {
        return (false, "The graphics driver no longer takes the prepared shaders. Prepare them again.".into());
    }
    // (Prepared as the driver's binaries while the game keeps cache data, binaries turned off:
    // none serves, so Play prepares again; the review of 2026-11-21 night: only a preparation's
    // start looked.)
    if !binaries_form() && dir_has_binaries(&app.paths.shaders().join("pipelines")) {
        return (false, "This version keeps prepared shaders in another form. Prepare them again.".into());
    }
    if device == get("device") && driver == get("driver") {
        (true, format!("The shaders are prepared (on {date}, for this graphics card and driver)."))
    } else if device == get("device") {
        (false, "The graphics driver changed since the shaders were prepared. Prepare them again.".into())
    } else {
        (false, "The graphics card changed since the shaders were prepared. Prepare them again.".into())
    }
}

/// How the self-play went (played.txt): the new game's end, the levels' parts played, the last
/// part's end (0: played to its end or to the preparation's time, 3: stopped, other: it could
/// not go on); for this game version's translation, this card and driver (else nothing).
#[derive(Default, Clone, Copy, PartialEq, Eq, Debug)]
struct Played {
    newgame: Option<i32>,
    parts: u8,
    tour: Option<i32>,
    /// The preparation's time ran out before the last part ended (the latest levels left).
    cut: bool,
}

/// The translation, card and driver the self-play's record holds for.
fn identity() -> (String, String, String) {
    let translation = match checked_translation() {
        Some(Ok(t)) => t,
        _ => String::new(),
    };
    // (The card and driver the game uses, as its check reported them; else the launcher's own
    // reading.)
    if let Some((_, d, r)) = GAME_ID.lock().unwrap_or_else(|e| e.into_inner()).clone().filter(|(_, d, r)| !d.is_empty() && !r.is_empty()) {
        return (translation, d, r);
    }
    match sys::gpu() {
        Ok(g) => (translation, g.name, g.driver.to_string()),
        Err(_) => (translation, String::new(), String::new()),
    }
}

/// The preparation's rules in a translation fingerprint (its last part, `r9`): what the self-play
/// learned holds while they stay, across drivers and versions that translate differently.
fn rules_of(t: &str) -> &str {
    t.rsplit('-').next().filter(|r| r.starts_with('r')).unwrap_or(t)
}

/// The self-play played to its end for this card and the preparation's rules.
fn played_complete(p: &Played) -> bool {
    p.newgame == Some(0) && p.tour.is_some_and(|c| c != STAGE_STOPPED)
}

/// The self-play's attempt ended for this card and the preparation's rules: played to its end, or
/// a new game that could not go on (twice: the compute step then made what it could). A stale
/// store then needs the compile alone (the review of 2026-11-21: after such a new game, a refusal
/// or a new driver replayed it).
fn played_ended(p: &Played) -> bool {
    played_complete(p) || p.newgame.is_some_and(|c| c != 0 && c != STAGE_STOPPED)
}

/// The record of a preparation that could not finish (`record_outcome`): Play starts the game
/// and does not prepare again until something changes (the card, driver or version) or Prepare
/// now is pressed (the review of 2026-11-20: a compute step that failed restarted at every Play).
fn outcome_file(app: &App) -> PathBuf {
    app.paths.shaders().join("prepare-outcome.txt")
}

fn record_outcome(app: &App, why: &str) {
    let (translation, device, driver) = identity();
    log(app, &format!("preparation given up: {why}"));
    write_record(
        &outcome_file(app),
        &format!("# A shader preparation that could not finish on this card, driver and version; written by the launcher\noutcome = gave up\nwhy = {why}\ntranslation = {translation}\ndevice = {device}\ndriver = {driver}\ndate = {}\n", utc_stamp()),
    );
}

/// Why the preparation could not finish here, for this card, driver and version (None: no such
/// record, or one for another).
fn outcome(app: &App) -> Option<String> {
    let kv = read_kv(&outcome_file(app));
    let get = |k: &str| kv.iter().find(|(x, _)| x == k).map(|(_, v)| v.clone()).unwrap_or_default();
    let (translation, device, driver) = identity();
    (!kv.is_empty() && get("translation") == translation && get("device") == device && get("driver") == driver).then(|| get("why"))
}

fn played(app: &App) -> Played {
    let kv = read_kv(&played_file(app));
    let get = |k: &str| kv.iter().find(|(x, _)| x == k).map(|(_, v)| v.clone()).unwrap_or_default();
    // (For this card and the preparation's rules: a new driver or a version that translates
    // differently keeps what the self-play learned, its pipelines made again by the compute step.)
    let (translation, device, _) = identity();
    if kv.is_empty() || get("device") != device || (!translation.is_empty() && rules_of(&get("translation")) != rules_of(&translation)) {
        return Played::default();
    }
    let code = |k: &str| get(k).parse::<i32>().ok();
    Played { newgame: code("newgame"), parts: get("parts").parse().unwrap_or(0), tour: code("tour"), cut: get("cut") == "1" }
}

/// The self-play's record changed by `f` (with the translation, card and driver now).
fn record(app: &App, f: impl FnOnce(&mut Played)) {
    let mut p = played(app);
    f(&mut p);
    let (translation, device, driver) = identity();
    // (The check gave no answer: the record keeps what it held, never an empty translation, card
    // or driver, which the next start read as another version and played everything again; the
    // review of 2026-11-21 night.)
    let old = read_kv(&played_file(app));
    let keep = |now: String, k: &str| if now.is_empty() { old.iter().find(|(x, _)| x == k).map(|(_, v)| v.clone()).unwrap_or_default() } else { now };
    let (translation, device, driver) = (keep(translation, "translation"), keep(device, "device"), keep(driver, "driver"));
    let line = |v: Option<i32>| v.map_or(String::new(), |c| c.to_string());
    let t = format!(
        "# The game's self-play that learned the shaders (dsport --prepare-play); written by the launcher\n# (0: played to its end, {STAGE_STOPPED}: stopped, other: it could not go on; parts: the levels' parts played; cut: the time ran out first)\nnewgame = {}\nparts = {}\ntour = {}\ncut = {}\ntranslation = {translation}\ndriver = {driver}\ndevice = {device}\n",
        line(p.newgame),
        p.parts,
        line(p.tour),
        p.cut as u8,
    );
    write_record(&played_file(app), &t);
}

/// The prepared marker as the launcher writes it when the self-play's last part ended without
/// one: the date, this card and driver and this version's translation (`compute_state`'s keys).
fn write_marker(app: &App) {
    let (translation, device, driver) = identity();
    if translation.is_empty() || device.is_empty() {
        return;
    }
    let t = format!(
        "# The game's shaders prepared ahead; written by the launcher (the self-play's last part ended without marking them)\ndate = {}\nprograms = unknown\ndevice = {device}\ndriver = {driver}\ntranslation = {translation}\n",
        utc_stamp()
    );
    write_record(&marker(app), &t);
}

/// Whether `dir` holds a character's save within `depth` levels (a `SAVEDATA...PlayerProfile...`
/// folder with its `USR-DATA` file: the saves folder itself is made at every session's start,
/// and the options' save at the first boot, before any character).
fn has_save(dir: &std::path::Path, depth: u32) -> bool {
    let Ok(rd) = std::fs::read_dir(dir) else { return false };
    rd.flatten().any(|e| {
        let p = e.path();
        if !p.is_dir() {
            return false;
        }
        (e.file_name().to_string_lossy().contains("PlayerProfile") && p.join("USR-DATA").is_file()) || (depth > 0 && has_save(&p, depth - 1))
    })
}

/// Now as `YYYYMMDD-HHMMSS` (UTC).
fn utc_stamp() -> String {
    let secs = std::time::SystemTime::now().duration_since(std::time::UNIX_EPOCH).map_or(0, |d| d.as_secs()) as i64;
    let (days, rem) = (secs.div_euclid(86400), secs.rem_euclid(86400));
    // Civil from days (Howard Hinnant).
    let z = days + 719468;
    let era = z.div_euclid(146097);
    let doe = z - era * 146097;
    let yoe = (doe - doe / 1460 + doe / 36524 - doe / 146096) / 365;
    let doy = doe - (365 * yoe + yoe / 4 - yoe / 100);
    let mp = (5 * doy + 2) / 153;
    let d = doy - (153 * mp + 2) / 5 + 1;
    let m = if mp < 10 { mp + 3 } else { mp - 9 };
    let y = yoe + era * 400 + if m <= 2 { 1 } else { 0 };
    format!("{y:04}{m:02}{d:02}-{:02}{:02}{:02}", rem / 3600, (rem / 60) % 60, rem % 60)
}

/// The store check's record (`Step::CheckRead`): its result for a card and driver.
fn check_file(app: &App) -> PathBuf {
    app.paths.shaders().join("store-check.txt")
}

/// The check's result for this card and driver: "binaries", "cache" (the store keeps cache data),
/// "nohelp" (neither reaches from one session to the next) or "unknown"; None: not checked.
fn check_state(app: &App) -> Option<String> {
    let kv = read_kv(&check_file(app));
    let get = |k: &str| kv.iter().find(|(x, _)| x == k).map(|(_, v)| v.clone()).unwrap_or_default();
    let (_, device, driver) = identity();
    (!device.is_empty() && get("device") == device && get("driver") == driver && !get("result").is_empty()).then(|| get("result"))
}

fn record_check(app: &App, result: &str) {
    let (_, device, driver) = identity();
    let t = format!("# The check before preparing the shaders (dsport --pb-check): whether what one game session stores the next one takes; written by the launcher\nresult = {result}\ndevice = {device}\ndriver = {driver}\ndate = {}\n", utc_stamp());
    write_record(&check_file(app), &t);
}

/// The note when preparing cannot help on this PC.
const NO_HELP: &str = "This PC's graphics driver does not keep compiled shaders from one session to the next, so preparing them cannot help here. The game compiles each shader the first time it needs it.";

/// Whether the shaders are prepared for this graphics card and driver (the compute shaders, and
/// the self-play played or tried to its end); the status line.
pub fn state(app: &App) -> (bool, String) {
    // (The game's program of another version: nothing prepared with it, Play starts the game.)
    if let Some(t) = version_mismatch() {
        return (true, t);
    }
    // (Preparing cannot help on this card and driver: nothing to prepare, Play starts the game.)
    if check_state(app).as_deref() == Some("nohelp") {
        return (true, NO_HELP.into());
    }
    // (A preparation that could not finish here: Play starts the game, Prepare now tries again.)
    if let Some(why) = outcome(app) {
        return (true, format!("The shaders could not be fully prepared on this PC: {why}. The game compiles the rest while you play. Prepare now tries again."));
    }
    let (ok, text) = compute_state(app);
    if !ok {
        // (A new driver or version: the self-play's learning holds, only the compile is redone.)
        if app.game.ok && played_ended(&played(app)) && text.ends_with("Prepare them again.") {
            return (false, text.replace("Prepare them again.", "Prepare now compiles them again, without the game playing again."));
        }
        return (ok, text);
    }
    let ended = |c: Option<i32>| c.is_some_and(|c| c != STAGE_STOPPED);
    let p = played(app);
    match (p.newgame, p.tour) {
        (Some(0), Some(0)) if !p.cut => (true, text),
        (Some(0), Some(0)) => (true, format!("{text} The latest levels are left to their first visits.")),
        // (A stage that could not go on: what it learned is kept, and play learns the rest.)
        (Some(c), _) if c != 0 && ended(Some(c)) => (true, format!("{text} Some graphics shaders could not be learned ahead.")),
        (Some(0), t) if ended(t) => (true, format!("{text} Some graphics shaders could not be learned ahead.")),
        _ => (false, "The shaders are partly prepared. Prepare now finishes them.".into()),
    }
}

pub fn prepared(app: &mut App) -> bool {
    wait_check();
    // (The probe's answer, before Play decides: a store the driver refuses is prepared again.)
    ensure_probe(app);
    state(app).0
}

/// Waits for the translation check to end (a few seconds at most): Play waits for it, also to
/// start the game, as the check's run of the game holds the game's one-session lock meanwhile
/// (a start then is told the game already runs: seen 2026-11-14 with Play pressed 2 s after the
/// launcher opened and the preparation on Play turned off).
pub fn wait_check() {
    let t = Instant::now();
    while CHECK_STARTED.load(std::sync::atomic::Ordering::Relaxed) && checked_translation().is_none() && t.elapsed().as_secs() < 20 {
        std::thread::sleep(std::time::Duration::from_millis(50));
    }
}

pub fn build(app: &mut App) {
    let b: Builder = app.builder();
    unsafe {
        let r = (16, 62, 748, 62);
        app.put(2, b.label("The game compiles each shader the first time it needs it, which makes play stutter. Preparing them here lets the game play by itself in the background: a new game, then every level with its enemies and bosses. Nothing of it shows or sounds, but the graphics card is busy. It takes about an hour once on a fast PC, and longer on others. This page then says how long.", r, -1), r);
        let r = (16, 134, 748, 40);
        app.shaders.status = app.put(2, b.label("", r, -1), r) as usize;
        let r = (16, 178, 748, 20);
        app.shaders.bar = app.put(2, b.progress(r, -1), r) as usize;
        let r = (16, 210, 140, 30);
        app.shaders.prepare = app.put(2, b.button("Prepare now", r, ID_PREPARE), r) as usize;
        let r = (164, 210, 100, 30);
        app.shaders.stop = app.put(2, b.button("Stop", r, ID_STOP), r) as usize;
        let r = (16, 256, 748, 40);
        app.shaders.auto = app.put(2, b.check("Prepare the shaders when Play is pressed and they are not prepared yet.", r, ID_AUTO), r) as usize;
        ui::set_check(h(app.shaders.auto), app.launch.prepare_shaders);
        let where_ = format!(
            "Prepared shaders are kept in {}. They take up to 5 GB. Preparing uses up to {} of this PC's {} processor threads.",
            ui::short(&app.paths.shaders(), 70),
            sys::compute_threads(),
            sys::threads()
        );
        let r = (16, 306, 748, 60);
        app.put(2, b.label(&where_, r, -1), r);
    }
    refresh(app);
}

/// The page's state when nothing runs.
pub fn refresh(app: &mut App) {
    if running(app) {
        return;
    }
    // (A preparation waiting for the game's check: its own text and buttons, which a page change
    // had replaced, so it could not be stopped; the review of 2026-11-21 night.)
    if app.shaders.pending_start.is_some() {
        unsafe {
            ui::set_text(h(app.shaders.status), "Checking the game first. The shader preparation begins once that is done.");
            ui::show(h(app.shaders.bar), false);
            ui::enable(h(app.shaders.prepare), false);
            ui::enable(h(app.shaders.stop), true);
        }
        return;
    }
    let (_, text) = state(app);
    let text = battery_note(app).map_or(text, str::to_string);
    unsafe {
        ui::set_text(h(app.shaders.status), &text);
        ui::show(h(app.shaders.bar), false);
        ui::enable(h(app.shaders.prepare), app.game.ok);
        ui::enable(h(app.shaders.stop), false);
    }
}

pub fn running(app: &App) -> bool {
    app.shaders.child.is_some() || app.shaders.retry.is_some()
}

/// Play pressed while the game's check has not answered: Play goes on once it answers
/// (`check_tick`), the window live meanwhile. True when deferred. Until 2026-11-21 night Play
/// waited 20 s and then judged the store without the check's translation, so a store an update
/// had made stale could start the game unprepared (the other PC's check took 50 s).
pub fn defer_play(app: &mut App) -> bool {
    let check = check_pending();
    if !check && !probe_pending(app) {
        return false;
    }
    if !app.shaders.pending_play {
        log(app, if check { "Play pressed during the game's check: it goes on once the check answers" } else { "Play pressed during the prepared shaders' probe: it goes on once the probe answers" });
    }
    app.shaders.pending_play = true;
    app.set_status("Checking the game first. Play goes on once that is done.");
    true
}

/// Whether a stop was asked for (the preparation ends on its own).
pub fn stopping(app: &App) -> bool {
    app.shaders.stopping.is_some()
}

/// Starts the preparation (`play_after`: the game starts when it is done) at its first step not
/// done: the compute shaders unless they are prepared, then the self-play's stages not played
/// (all three again when everything is done: Prepare now).
pub fn start(app: &mut App, play_after: bool) {
    // (Not without the game's files: it ended with "an error (code 3)", 2026-11-19.)
    if running(app) || app.running.is_some() || !app.game.ok {
        return;
    }
    let exe = dsconfig::paths::game_exe_in(&app.paths.exe_dir);
    if !exe.is_file() {
        unsafe { ui::message(app.hwnd, &format!("{} was not found next to the launcher.", dsconfig::paths::GAME_EXE), crate::TITLE, windows_sys::Win32::UI::WindowsAndMessaging::MB_ICONERROR) };
        return;
    }
    // (The game's check first: without its translation the store cannot be judged, and none of
    // it is taken as made. Begun by the check's timer once it answers; the window stays live.)
    let check = check_pending();
    if check || probe_pending(app) {
        if app.shaders.pending_start.is_none() {
            log(app, if check { "preparation asked for during the game's check: it begins once the check answers" } else { "preparation asked for during the prepared shaders' probe: it begins once the probe answers" });
        }
        app.shaders.pending_start = Some(play_after || app.shaders.pending_start.unwrap_or(false));
        unsafe {
            ui::set_text(h(app.shaders.status), "Checking the game first. The shader preparation begins once that is done.");
            ui::enable(h(app.shaders.prepare), false);
            ui::enable(h(app.shaders.stop), true);
        }
        app.update_play();
        return;
    }
    app.shaders.pending_start = None;
    app.shaders.retry = None;
    app.shaders.retries = 0;
    app.shaders.newgame_no_list = false;
    app.shaders.ended = None;
    let dir = app.paths.shaders();
    if let Some(free) = sys::free_space(&dir) {
        if free < 5 << 30 {
            let t = format!("Only {} is free on the drive of {}. Prepared shaders take up to 5 GB. Prepare them anyway?", sys::gb(free), dir.display());
            if unsafe { ui::message(app.hwnd, &t, crate::TITLE, windows_sys::Win32::UI::WindowsAndMessaging::MB_ICONWARNING | windows_sys::Win32::UI::WindowsAndMessaging::MB_YESNO) } != windows_sys::Win32::UI::WindowsAndMessaging::IDYES {
                // (Not asked again in this session: Play starts the game, the review of
                // 2026-11-20: No had done nothing, at every Play.)
                log(app, &format!("preparation not begun: {} free, the player declined", sys::gb(free)));
                app.shaders.space_declined = true;
                if play_after {
                    crate::start_game(app);
                }
                return;
            }
        }
    }
    // (A low battery: not begun; Play starts the game.)
    if let Some(pct) = sys::battery().filter(|&p| p <= BATTERY_LOW) {
        log(app, &format!("preparation not begun: the battery is at {pct}%"));
        finish(app, Some(BATTERY_NOTE.into()));
        if play_after {
            crate::start_game(app);
        }
        return;
    }
    let _ = std::fs::create_dir_all(&dir);
    wait_check();
    if version_mismatch().is_some() {
        if play_after {
            crate::start_game(app);
        }
        return;
    }
    // (The probe first: a store the driver refuses is set aside below, not taken as made.)
    ensure_probe(app);
    let _ = std::fs::remove_file(outcome_file(app));
    app.shaders.untrusted = !retire_stale_store(app);
    // (Prepare now on a preparation that was done: the shaders listed as having crashed the
    // driver's compiler get a fresh chance too. One that crashed for a passing reason (memory ran
    // out) was otherwise left out for good, its draws or dispatches going without it every
    // session; the review of 2026-11-21 night. One that crashes again is listed again at once.)
    if !play_after && state(app).0 {
        let had = crashed_lines(app);
        if had > 0 {
            let dir = app.paths.shaders().join("pipelines");
            for f in ["crashed.txt", "crash-suspect.txt"] {
                let _ = std::fs::remove_file(dir.join(f));
            }
            log(app, &format!("Prepare now on prepared shaders: the {had} listed as having crashed the driver's compiler get a fresh chance"));
        }
    }
    // (The next part of the levels when the new game was played for this version, card and
    // driver and the levels are not done; else, and when everything is done, the new game.)
    let p = played(app);
    let save = app.paths.cache.join("prepare").join("user").join("saves");
    // (The self-play played to its end, the compiled pipelines stale: a new driver or version.
    // The compute step alone makes again everything learned; the game does not play again.)
    let recompile = played_ended(&p) && !compute_done(app);
    let first = match p {
        _ if recompile => Step::Compute,
        Played { newgame: Some(0), parts, tour, .. } if parts < LEVEL_PARTS && tour != Some(0) && has_save(&save, 4) => Step::Levels(parts + 1),
        _ => Step::NewGame,
    };
    // (The compute step before the levels when its list is not made: stopped there before.)
    let (first, after_compute) = match first {
        Step::Levels(_) if !compute_done(app) => (Step::Compute, Some(first)),
        _ => (first, None),
    };
    // (The store check first, once a card and driver.)
    app.shaders.after_check = None;
    app.shaders.check_round = 1;
    // (Prepare now pressed on a card and driver recorded as one preparing cannot help: the check
    // runs again, as the driver or its settings may have changed since; Play leaves it be. The
    // review of 2026-11-21 night: the record had stood for good.)
    if !play_after && check_state(app).as_deref() == Some("nohelp") {
        log(app, "Prepare now on a card and driver recorded as one preparing cannot help: the store check runs again");
        let _ = std::fs::remove_file(check_file(app));
    }
    let first = match check_state(app).as_deref() {
        Some("nohelp") => {
            finish(app, Some(NO_HELP.into()));
            if play_after {
                crate::start_game(app);
            }
            return;
        }
        None => {
            app.shaders.after_check = Some(first);
            Step::CheckWrite
        }
        _ => first,
    };
    app.shaders.prep_started = Some(Instant::now());
    app.shaders.attempts.clear();
    app.shaders.crash_retries = 0;
    app.shaders.continues = 0;
    app.shaders.part_secs.clear();
    app.shaders.slow_noted = false;
    app.shaders.compute_secs = None;
    app.shaders.after_compute = after_compute;
    app.shaders.play_after = play_after;
    app.shaders.stopping = None;
    app.shaders.then_play = false;
    app.shaders.stalled = false;
    app.shaders.last_failed = false;
    app.shaders.battery_stop = false;
    app.shaders.battery_low = 0;
    app.shaders.battery_note = false;
    log(
        app,
        &format!(
            "preparation begun ({}): first step {first:?}{}{}; {} free on the shaders' drive; {} of {} processor threads for compiling; {}",
            if play_after { "from Play" } else { "Prepare now" },
            if recompile { ", the compile again only" } else { "" },
            if app.shaders.untrusted { ", stored pipelines not trusted" } else { "" },
            sys::free_space(&app.paths.shaders()).map_or("unknown".to_string(), sys::gb),
            sys::compute_threads(),
            sys::threads(),
            sys::power_text()
        ),
    );
    if begin(app, first) {
        sys::keep_awake(true);
        unsafe {
            ui::set_progress(h(app.shaders.bar), 0);
            ui::show(h(app.shaders.bar), true);
            ui::enable(h(app.shaders.prepare), false);
            ui::enable(h(app.shaders.stop), true);
            windows_sys::Win32::UI::WindowsAndMessaging::SetTimer(app.hwnd, TIMER, 500, None);
        }
        app.update_play();
        app.set_status(if play_after { "Preparing the shaders first. The game starts when they are done." } else { "" });
    } else {
        // (Its first step could not begin: the drive full, or the game's program not started.
        // The page says why, and Play starts the game: until 2026-11-20 night Play did nothing.)
        finish(app, None);
        if play_after {
            crate::start_game(app);
        }
    }
}

/// Starts `step`'s child (false when it could not be started: the status says why).
fn begin(app: &mut App, step: Step) -> bool {
    // (The shaders folder's drive nearly full: nothing more could be kept, the step not begun;
    // the page then says why, `outcome`.)
    if let Some(free) = sys::free_space(&app.paths.shaders()).filter(|&f| f < 1 << 30) {
        record_outcome(app, &format!("the drive of the shaders folder is full ({} free)", sys::gb(free)));
        app.shaders.child = None;
        app.shaders.step = None;
        return false;
    }
    let exe = dsconfig::paths::game_exe_in(&app.paths.exe_dir);
    let progress = progress_file(app);
    let _ = std::fs::remove_file(&progress);
    let _ = std::fs::remove_file(progress.with_extension("stop"));
    let _ = std::fs::remove_file(progress.with_extension("health"));
    // (A part begun again keeps its resume record; a step begun anew starts from its first level.)
    if !std::mem::take(&mut app.shaders.resuming) {
        let _ = std::fs::remove_file(progress.with_extension("resume"));
    }
    const CREATE_NO_WINDOW: u32 = 0x0800_0000;
    const BELOW_NORMAL_PRIORITY_CLASS: u32 = 0x0000_4000;
    let mut cmd = std::process::Command::new(&exe);
    match (step, step.stage()) {
        (Step::CheckWrite, _) => cmd.arg("--pb-check").arg("write"),
        (Step::CheckRead, _) => cmd.arg("--pb-check").arg("read"),
        (_, None) => cmd.arg("--precompile-all").arg(sys::compute_threads().to_string()),
        (_, Some(s)) => cmd.arg("--prepare-play").arg(s).arg("--seconds").arg(step.seconds().to_string()).arg("--prepare-threads").arg(sys::prepare_threads().to_string()),
    };
    if let Step::Levels(i) = step {
        cmd.arg("--prepare-chunk").arg(format!("{i}/{LEVEL_PARTS}"));
    }
    cmd.arg("--progress").arg(&progress).current_dir(&app.paths.exe_dir);
    if app.shaders.untrusted {
        cmd.env("DS_STORE_UNTRUSTED", "1");
    }
    if step == Step::NewGame && app.shaders.newgame_no_list {
        cmd.env("DS_PREPARE_BACKGROUND", "0");
    }
    match sys::spawn_in_job(&mut cmd, CREATE_NO_WINDOW | BELOW_NORMAL_PRIORITY_CLASS) {
        Ok(c) => {
            log(app, &format!("step {step:?} begun"));
            app.shaders.crashed_at_begin = crashed_lines(app);
            app.shaders.child = Some(c);
            app.shaders.step = Some(step);
            app.shaders.started = Some(Instant::now());
            app.shaders.progress_seen = None;
            let text = match step {
                Step::CheckWrite | Step::CheckRead => "Preparing the shaders: first a check that this PC keeps them from one session to the next...".to_string(),
                Step::Compute => "Preparing the compute shaders: reading the game's shader files...".to_string(),
                Step::NewGame => format!("Preparing the shaders, step 1 of {}: the game starts a new game in the background...", LEVEL_PARTS + 1),
                Step::Levels(_) => format!("Preparing the shaders, step {} of {}: the game loads its save in the background...", step.number(), LEVEL_PARTS + 1),
            };
            unsafe { ui::set_text(h(app.shaders.status), &text) };
            true
        }
        Err(e) => {
            log(app, &format!("step {step:?} could not begin: {e}"));
            app.shaders.child = None;
            app.shaders.step = None;
            record_outcome(app, &format!("the game's program could not be started for it ({e})"));
            false
        }
    }
}

/// Stops the preparation, then starts the game.
pub fn stop_then_play(app: &mut App) {
    let between = app.shaders.child.is_none() && app.shaders.retry.is_some();
    stop(app);
    if between {
        crate::start_game(app);
    } else {
        app.shaders.then_play = true;
    }
}

/// The launcher closes while a step runs (`close`, after `stop`): the step's processes are ended
/// now, as the job would end them when the launcher exits, and their records of compiles in
/// flight are removed (`inflight/inflight-<pid>.bin` in the store). Those ends are the player's,
/// not a graphics driver's crash, and the game would otherwise take the compiles for suspects and
/// leave them out of later preparations (the review of 2026-11-24 night; the game's
/// `pcache::Inflight`).
pub fn end_for_close(app: &mut App) {
    let pids = crate::sys::end_job(5000);
    let dir = app.paths.shaders().join("pipelines").join("inflight");
    for pid in &pids {
        let _ = std::fs::remove_file(dir.join(format!("inflight-{pid}.bin")));
    }
    if !pids.is_empty() {
        log(app, &format!("closing: the preparation's processes {pids:?} ended, their records of compiles in flight removed"));
    }
}

/// Asks the preparation to stop (its current shaders finish first).
pub fn stop(app: &mut App) {
    // (Asked for during the game's check and not begun yet: dropped.)
    if app.shaders.pending_start.take().is_some() {
        log(app, "the preparation asked for during the game's check or the shaders' probe was stopped before it began");
        refresh(app);
        app.update_play();
        return;
    }
    // (Between a step and its new start: nothing runs, it ends at once.)
    if app.shaders.child.is_none() && app.shaders.retry.take().is_some() {
        app.shaders.play_after = false;
        finish(app, Some("Preparation stopped. The next one goes on from the step it stopped in.".into()));
        return;
    }
    if app.shaders.child.is_none() || app.shaders.stopping.is_some() {
        return;
    }
    app.shaders.play_after = false;
    app.shaders.stopping = Some(Instant::now());
    let _ = std::fs::write(progress_file(app).with_extension("stop"), "stop\n");
    unsafe {
        ui::set_text(h(app.shaders.status), "Stopping after the shaders being compiled now...");
        ui::enable(h(app.shaders.stop), false);
    }
}

/// The preparation ended (`note`: what the page says when it is not the usual state).
fn finish(app: &mut App, note: Option<String>) {
    app.shaders.step = None;
    unsafe { windows_sys::Win32::UI::WindowsAndMessaging::KillTimer(app.hwnd, TIMER) };
    sys::keep_awake(false);
    // (The launcher's status line may say the game starts when they are done: not any more,
    // unless it does, the review of 2026-11-20.)
    app.set_status("");
    refresh(app);
    app.update_play();
    let shown = note.clone().unwrap_or_else(|| state(app).1);
    log(app, &format!("preparation ended: {shown}"));
    if let Some(n) = note {
        unsafe { ui::set_text(h(app.shaders.status), &n) };
    }
}

/// The timer: progress, and the end of each step.
pub fn tick(app: &mut App) {
    // (A step begun again: see `retry`.)
    if app.shaders.child.is_none() {
        if let Some((step, at)) = app.shaders.retry {
            if Instant::now() >= at {
                app.shaders.retry = None;
                if !begin(app, step) {
                    finish(app, None);
                    if std::mem::take(&mut app.shaders.play_after) {
                        crate::start_game(app);
                    }
                }
            }
        }
        return;
    }
    let Some(child) = app.shaders.child.as_mut() else { return };
    let exited = match child.try_wait() {
        Ok(Some(s)) => Some(s.code().unwrap_or(-1)),
        Ok(None) => None,
        Err(_) => Some(-1),
    };
    // (A stop that takes too long: the child is ended.)
    if exited.is_none() && app.shaders.stopping.is_some_and(|t| t.elapsed().as_secs() > 90) && app.shaders.killed_at.is_none() {
        kill_step(child);
        app.shaders.killed_at = Some(Instant::now());
    }
    let step = app.shaders.step.unwrap_or(Step::Compute);
    // (The battery low while the PC runs on it: the step stops after what it compiles now, and the
    // next preparation goes on from there.)
    // (Three low readings in a row, 1.5 s: a reading that flickers, a dock renegotiating, does not
    // end the preparation. Not when the step is stopping already for a game started meanwhile
    // (its stop file): that pause stays a pause, and Play's game is not started beside it; the
    // review of 2026-11-21.)
    let low = sys::battery().filter(|&p| p <= BATTERY_LOW);
    app.shaders.battery_low = if low.is_some() { app.shaders.battery_low.saturating_add(1) } else { 0 };
    let stop_file = progress_file(app).with_extension("stop").exists();
    if exited.is_none() && app.shaders.stopping.is_none() && !app.shaders.battery_stop && app.shaders.battery_low >= 3 && !stop_file {
        if let Some(pct) = low {
            app.shaders.battery_stop = true;
            log(app, &format!("step {step:?}: the battery is at {pct}%: stopped"));
            // (A preparation Play began: the game starts once the step has stopped.)
            let play = app.shaders.play_after;
            stop(app);
            app.shaders.then_play |= play;
            return;
        }
    }
    // (Ended, but its process does not go: a thread of it stuck in the graphics driver keeps it,
    // and its video memory, until Windows restarts (seen 2026-11-20). Left behind after 30 s:
    // the preparation ends, the player told to restart, and Play's game started (the release
    // review of 2026-11-20 night: the launcher had waited for it for ever).)
    if exited.is_none() && app.shaders.killed_at.is_some_and(|t| t.elapsed().as_secs() > 30) {
        log(app, &format!("step {step:?}: its process did not end 30 s after it was ended (held by the graphics driver?): left behind"));
        app.shaders.child = None;
        app.shaders.killed_at = None;
        app.shaders.stopping = None;
        app.shaders.stalled = false;
        record_outcome(app, "one of its steps is stuck in the graphics driver, so restart the PC before you play");
        finish(app, None);
        if std::mem::take(&mut app.shaders.then_play) | std::mem::take(&mut app.shaders.play_after) {
            crate::start_game(app);
        }
        return;
    }
    // (A step whose progress has not changed for `STALL_SECS`: hung, ended. Its numbers, not the
    // file's time: the compute step writes the file every second, the review of 2026-11-20. A gap
    // of over 10 s between ticks, the PC asleep, starts that time again.)
    let slept = app.shaders.last_tick.is_some_and(|t| t.elapsed().as_secs() > 10);
    app.shaders.last_tick = Some(Instant::now());
    if exited.is_none() {
        let content = std::fs::read_to_string(progress_file(app)).unwrap_or_default();
        match &app.shaders.progress_seen {
            Some((seen, at)) if *seen == content && !slept => {
                if at.elapsed().as_secs() > stall_secs() && !app.shaders.stalled {
                    app.shaders.stalled = true;
                    log(app, &format!("step {step:?}: no progress for {} s: ended", stall_secs()));
                    if let Some(c) = app.shaders.child.as_mut() {
                        kill_step(c);
                        app.shaders.killed_at = Some(Instant::now());
                    }
                }
            }
            _ => app.shaders.progress_seen = Some((content, Instant::now())),
        }
    }
    if let Some(code) = exited {
        app.shaders.child = None;
        app.shaders.killed_at = None;
        let secs = app.shaders.started.map_or(0.0, |s| s.elapsed().as_secs_f64());
        let _ = std::fs::remove_file(progress_file(app).with_extension("stop"));
        let asked = app.shaders.stopping.take().is_some();
        // (Stopped without the launcher asking: a game started meanwhile asked the step to stop,
        // `dsport`'s `ask_preparation_to_stop`; the compute step's stop status is 4.)
        let paused = !asked && (code == STAGE_STOPPED || (step == Step::Compute && code == 4));
        let stopped = asked || paused;
        let code = if stopped && code != 0 { STAGE_STOPPED } else { code };
        log(app, &format!("step {step:?} ended: status {} after {secs:.0} s{}", code_text(code), if app.shaders.stalled { " (ended: no progress)" } else if paused { " (stopped for a game started meanwhile)" } else if code == STAGE_HUNG { " (it found itself hung: its log has the threads' stacks)" } else { "" }));
        // (The step's health line, `prepplay::health_line`: what its tests hold to their limits,
        // and what a player's logs show of how the step went.)
        let health = std::fs::read_to_string(progress_file(app).with_extension("health")).ok().and_then(|t| t.lines().next().map(|l| l.trim().to_string())).filter(|l| !l.is_empty());
        if let Some(l) = &health {
            log(app, &format!("step {step:?} health: {}", l.trim_start_matches("prepare-health: ")));
        }
        // (Played to its end, then hung or stalled in its teardown: its health line, written
        // before that, says status 0. Done, not a failure to play again; the review of 2026-11-21
        // night: first only for a hang, which the hang's own line then replaced, and the compute
        // step's teardown has no hang watch but its exit's.)
        let code = if code != 0 && code != GAME_RUNNING && health.as_deref().is_some_and(|l| l.contains("; status=0;")) {
            log(app, &format!("step {step:?} had ended its work (its health line says status 0) before it hung in its teardown: taken as done"));
            0
        } else {
            code
        };
        // (What happened, in the player's words: the outcome and the notes.)
        let why = if std::mem::take(&mut app.shaders.stalled) { format!("it made no progress for {} minutes", stall_secs() / 60) } else { reason(code) };
        // (Another game session held the player's folder: the step did not run.)
        if code == GAME_RUNNING {
            // (Its own previous step's session still ending: the step again 3 s later, 10 times
            // at most. The host waits for the game's process to end since 2026-11-20, so this
            // is a fallback: before, the levels' first part met the new game's lock.)
            if app.shaders.ended.is_some_and(|t| t.elapsed().as_secs() < 90) && app.shaders.retries < 10 {
                app.shaders.retries += 1;
                app.shaders.retry = Some((step, Instant::now() + std::time::Duration::from_secs(3)));
                // (A session that met the lock played nothing: a part goes on from where its
                // record says. The retry's begin had found `resuming` used up and deleted the
                // record, the part replayed from its first level; the review of 2026-11-21 night.)
                app.shaders.resuming = matches!(step, Step::Levels(_));
                return;
            }
            app.shaders.play_after = false;
            // (A process of an earlier step still in the launcher's job, past its host's end and
            // these retries: one the graphics driver holds (2026-11-20: such processes stayed with
            // GBs of memory until a restart). The player is told to restart Windows, not to close
            // a game that is not theirs; the review of 2026-11-21 night.)
            let left = sys::job_processes();
            if !left.is_empty() {
                log(app, &format!("a process of an earlier step is still there after its end (ids {left:?}): the graphics driver holds it"));
                record_outcome(app, "a process of the preparation is stuck in the graphics driver, and only restarting Windows ends it");
                finish(app, Some("A process of the shader preparation is stuck in the graphics driver. Restart Windows, then press Prepare now.".into()));
                return;
            }
            finish(app, Some("The game is running, so the shaders cannot be prepared now. Close the game and press Prepare now.".into()));
            return;
        }
        app.shaders.ended = Some(Instant::now());
        app.shaders.retries = 0;
        // (A self-play step that crashed or lost the graphics device: begun once more, its
        // levels played again (what it made is kept: the second try is quicker). A preparation
        // that marked such a part done left its levels to the player, 2026-11-20.)
        // (The shaders folder's drive full: nothing more can be kept; the player told, the game
        // started if Play asked.)
        if code == STAGE_DISK_FULL || code == STAGE_UNWRITABLE {
            let free = sys::free_space(&app.paths.shaders()).map(sys::gb).unwrap_or_default();
            record_outcome(app, &if code == STAGE_DISK_FULL { format!("the drive of the shaders folder is full ({free} free)") } else { "the shaders folder cannot be written to (a security program or the folder's permissions may block it)".to_string() });
            finish(app, None);
            if std::mem::take(&mut app.shaders.play_after) {
                crate::start_game(app);
            }
            return;
        }
        // (No other step would end otherwise on this PC: the game's files, the processor, the
        // memory range, no graphics card, the game's start. The preparation ends at once and the
        // page says why; Play starts the game, which tells its own message.)
        if !stopped && fatal(code) {
            record_outcome(app, &why);
            finish(app, None);
            if std::mem::take(&mut app.shaders.play_after) {
                crate::start_game(app);
            }
            return;
        }
        // (Failed for the PC's sake: hung, crashed, the graphics card lost, memory. A step is
        // begun once more; when the step before failed too, it keeps failing on this PC and the
        // preparation ends there, the player told and let play: until 2026-11-20 a PC on which
        // every step hung waited for each of them, twice.)
        // (A new game that could not go on runs once more too: the levels need its save, and its
        // menus' navigation can slip, 2026-11-21: once in about fifteen new games.)
        // (The slip is not the PC's failure: it is not counted against the step before, the review
        // of 2026-11-21: a store check that crashed, then one slip, had ended the preparation.)
        // (A levels part's session that ended at a level's start for video memory, `EXIT_CONTINUE`
        // (17): the part goes on in a new session from that level, no attempt counted, at most
        // MAX_CONTINUES times a preparation.)
        if !stopped && code == STAGE_CONTINUE && matches!(step, Step::Levels(_)) {
            if app.shaders.continues < MAX_CONTINUES {
                app.shaders.continues += 1;
                log(app, &format!("step {step:?} goes on in a new session (video memory was under pressure; {} so far)", app.shaders.continues));
                app.shaders.resuming = true;
                if begin(app, step) {
                    return;
                }
                // (Its new session could not start (its reason recorded by `begin`): not played
                // to its end; the review of 2026-11-21 night: it had been recorded as played.)
                finish(app, None);
                if std::mem::take(&mut app.shaders.play_after) {
                    crate::start_game(app);
                }
                return;
            } else {
                log(app, &format!("step {step:?} would go on in a new session again ({MAX_CONTINUES} so far): taken as played to here"));
            }
        }
        let code = if code == STAGE_CONTINUE { 0 } else { code };
        let failed = !stopped && worth_retry(code);
        let slip = !stopped && step == Step::NewGame && code == 13;
        let retryable = matches!(step, Step::NewGame | Step::Levels(_) | Step::Compute);
        // (A failure that added a pipeline to the compiler-crash list: the driver's shader
        // compiler crashed on it, and the next attempt leaves it out of its background compiles.
        // Begun again without counting against the step, a few times a preparation; the other PC,
        // 2026-11-21: its last part met that crash after a hang had used its one more try.)
        let compiler_crash = failed && crashed_lines(app) > app.shaders.crashed_at_begin;
        if compiler_crash && retryable && app.shaders.crash_retries < MAX_CRASH_RETRIES {
            app.shaders.crash_retries += 1;
            log(app, &format!("step {step:?} begun again: {why}, in the graphics driver's shader compiler (that pipeline is left out from now on)"));
            app.set_status("A step of the shader preparation ended in the graphics driver's shader compiler. It runs again without that shader.");
            // (A part goes on from the level it reached, that shader left out; a new game runs
            // again without its background list, `newgame_no_list`.)
            app.shaders.resuming = matches!(step, Step::Levels(_));
            if step == Step::NewGame && !app.shaders.newgame_no_list {
                app.shaders.newgame_no_list = true;
                log(app, "the new game runs again without its background compiles: the compute step makes them");
            }
            if begin(app, step) {
                return;
            }
        }
        if failed && !compiler_crash && app.shaders.last_failed {
            record_outcome(app, &format!("its steps kept failing, the last because {why}"));
            finish(app, None);
            if std::mem::take(&mut app.shaders.play_after) {
                crate::start_game(app);
            }
            return;
        }
        // (Up to `MAX_ATTEMPTS` attempts a step; one more try only until 2026-11-21.)
        let tries = app.shaders.attempts.iter().find(|(s, _)| *s == step).map_or(0, |e| e.1);
        if (failed || slip) && retryable && tries + 1 < MAX_ATTEMPTS {
            match app.shaders.attempts.iter_mut().find(|(s, _)| *s == step) {
                Some(e) => e.1 += 1,
                None => app.shaders.attempts.push((step, 1)),
            }
            log(app, &format!("step {step:?} begun again (attempt {} of {MAX_ATTEMPTS}): {why}", tries + 2));
            // (In the launcher's status line: the page's own is the new step's at once.)
            app.set_status(&format!("A step of the shader preparation ended unexpectedly: {why}. It runs again."));
            // (A part goes on from the level it reached; after a hang from the next one, as the
            // level that hung would hang again.)
            if matches!(step, Step::Levels(_)) {
                app.shaders.resuming = true;
                if code == STAGE_HUNG {
                    resume_next_level(app);
                }
            }
            if begin(app, step) {
                return;
            }
        }
        // (Only the self-play's and the compute step's failures chain: a store check that failed
        // had ended the next step at its first failure, at every preparation on a PC whose check
        // fails; the review of 2026-11-21 night.)
        app.shaders.last_failed = failed && !compiler_crash && retryable;
        // (The step's time, for the estimate: a whole new game or part only.)
        if code == 0 {
            let secs = app.shaders.started.map_or(0.0, |s| s.elapsed().as_secs_f64());
            match step {
                Step::Levels(_) => app.shaders.part_secs.push(secs),
                Step::Compute => app.shaders.compute_secs = Some(secs),
                Step::NewGame | Step::CheckWrite | Step::CheckRead => {}
            }
        }
        // (Levels left: the self-play's marker says so, `levels_cut`; the clock alone would also
        // count a last part whose compute tail ended at the deadline after every level.)
        let cut = read_kv(&marker(app)).iter().any(|(k, v)| k == "levels_cut" && v == "1");
        match step {
            // (A new game played again makes a new save: the levels are played again too. One
            // stopped leaves the record as it was: a store prepared before stays prepared.)
            Step::NewGame if code != STAGE_STOPPED => record(app, |p| *p = Played { newgame: Some(code), ..Played::default() }),
            Step::NewGame => {}
            // (A part that could not go on is passed: the next part plays its own levels.)
            Step::Levels(i) if code != STAGE_STOPPED => record(app, |p| {
                p.parts = p.parts.max(i);
                if i >= LEVEL_PARTS {
                    p.tour = Some(code);
                    p.cut = cut;
                }
            }),
            Step::Levels(_) => record(app, |p| p.tour = Some(STAGE_STOPPED)),
            Step::Compute | Step::CheckWrite | Step::CheckRead => {}
        }
        // The last part ended without marking the store prepared (it could not go on, or the
        // session ended unasked): the launcher marks it, so that Play does not prepare again
        // (the release review of 2026-11-19).
        if step == Step::Levels(LEVEL_PARTS) && code != STAGE_STOPPED && !compute_state(app).0 {
            write_marker(app);
        }
        if stopped {
            let note = if std::mem::take(&mut app.shaders.battery_stop) {
                BATTERY_NOTE
            } else if paused {
                "Preparation paused: the game was started. The next one goes on from the step it stopped in."
            } else {
                "Preparation stopped. The next one goes on from the step it stopped in."
            };
            finish(app, Some(note.into()));
            if std::mem::take(&mut app.shaders.then_play) {
                crate::start_game(app);
            }
            return;
        }
        // The store check: its read decides (0: the store's way works; 20: refused, and in the
        // second round, with cache data, preparing cannot help; else unknown, the preparation
        // goes on).
        if matches!(step, Step::CheckWrite | Step::CheckRead) {
            let go = app.shaders.after_check.unwrap_or(Step::NewGame);
            let next = match (step, code) {
                (Step::CheckWrite, 0) => Some(Step::CheckRead),
                (Step::CheckRead, 0) => {
                    let cache = std::fs::read_to_string(app.paths.shaders().join("pipelines").join("binaries-mode.txt")).is_ok_and(|t| t.lines().any(|l| l.replace(' ', "") == "binaries=off"));
                    // (The form the store keeps: cache data unless binaries are asked for, or the
                    // mode file turned them off.)
                    record_check(app, if cache || !binaries_form() { "cache" } else { "binaries" });
                    Some(go)
                }
                // (A second round turns binaries to cache data and checks that: only when the
                // store kept binaries; with cache data it would repeat the same check.)
                (Step::CheckRead, 20) if app.shaders.check_round < 2 && binaries_form() => {
                    app.shaders.check_round = 2;
                    Some(Step::CheckWrite)
                }
                (Step::CheckRead, 20) => {
                    record_check(app, "nohelp");
                    finish(app, Some(NO_HELP.into()));
                    if std::mem::take(&mut app.shaders.play_after) {
                        crate::start_game(app);
                    }
                    return;
                }
                // (No answer (the device lost, a crash, its watch): not recorded, so the check
                // runs again at the next preparation and the probe stays on; a recorded "unknown"
                // had turned the probe off for good, the review of 2026-11-21 night.)
                _ => {
                    log(app, &format!("the store check gave no answer (status {}): it runs again at the next preparation", code_text(code)));
                    Some(go)
                }
            };
            if let Some(n) = next {
                if begin(app, n) {
                    return;
                }
            }
            finish(app, None);
            // (Play starts the game: the release review of 2026-11-20 night, the step after the
            // check not begun had left Play doing nothing.)
            if std::mem::take(&mut app.shaders.play_after) {
                crate::start_game(app);
            }
            return;
        }
        let next = match step {
            Step::Compute if code != 0 || !compute_state(app).0 => {
                // (Recorded for this card, driver and version: Play starts the game instead of
                // preparing again, Prepare now tries again; the review of 2026-11-20.)
                let why = if code != 0 { format!("its compute step failed because {why}") } else { "its compute step's record could not be written".to_string() };
                record_outcome(app, &why);
                finish(app, None);
                if std::mem::take(&mut app.shaders.play_after) {
                    crate::start_game(app);
                }
                return;
            }
            // (The compute list made: the levels next, their first dispatches finding their
            // pipelines in the store.)
            Step::Compute => app.shaders.after_compute.take(),
            Step::NewGame if code == 0 && !compute_done(app) => {
                app.shaders.after_compute = Some(Step::Levels(1));
                Some(Step::Compute)
            }
            Step::NewGame if code == 0 => Some(Step::Levels(1)),
            // (The new game could not be played: the compute shaders on their own, as before.)
            Step::NewGame if !compute_done(app) => Some(Step::Compute),
            // (The next part even past the preparation's time: it marks the shaders prepared at
            // once then.)
            Step::Levels(i) if i < LEVEL_PARTS => Some(Step::Levels(i + 1)),
            _ => None,
        };
        if let Some(n) = next {
            if begin(app, n) {
                return;
            }
        }
        // (A stage that could not go on: what it learned is kept, and the game can be played.)
        let note = (code != 0 && step != Step::Compute).then(|| format!("The game's self-play could not finish: {why}. Some graphics shaders are learned only while you play. Open logs shows its log."));
        finish(app, note);
        if std::mem::take(&mut app.shaders.play_after) {
            crate::start_game(app);
        }
        return;
    }
    if app.shaders.stopping.is_some() {
        return;
    }
    // Progress: "<done> <of>" and "<code done> <code>" (the self-play: its steps twice, then the
    // background compute shaders made and listed).
    let Ok(t) = std::fs::read_to_string(progress_file(app)) else { return };
    let nums: Vec<u64> = t.split_whitespace().filter_map(|w| w.parse().ok()).collect();
    if nums.len() >= 4 {
        let (done, total, wdone, wtotal) = (nums[0], nums[1], nums[2], nums[3]);
        let frac = if wtotal > 0 { wdone as f64 / wtotal as f64 } else { 0.0 };
        let el = app.shaders.started.map_or(0.0, |s| s.elapsed().as_secs_f64());
        let left = if frac > 0.02 { el * (1.0 - frac) / frac } else { 0.0 };
        let eta = if frac > 0.02 && step != Step::NewGame { format!(" About {} minutes left in this step.", (left / 60.0).ceil().max(1.0) as u64) } else { String::new() };
        let compute = match nums.get(4..6) {
            Some(&[c, all]) if all > 0 => format!(" Compute shaders: {c} of {all}."),
            _ => String::new(),
        };
        let steps = LEVEL_PARTS + 1;
        let cl = match nums.get(4..6) {
            Some(&[c, all]) => Some((c, all)),
            _ => None,
        };
        let (budget, slow) = match estimate(app, step, frac, cl) {
            Some((left, all)) if all > AIM_SECS * 1.1 => (
                format!(" Preparing takes longer on this PC: about {} minutes in all, {} more. Stop keeps what is done, and Prepare now goes on from there.", (all / 60.0).ceil() as u64, (left / 60.0).ceil().max(1.0) as u64),
                Some((left / 60.0).ceil().max(1.0) as u64),
            ),
            Some((left, _)) => (format!(" About {} minutes left in all.", (left / 60.0).ceil().max(1.0) as u64), None),
            None => (String::new(), None),
        };
        let text = match step {
            Step::CheckWrite | Step::CheckRead => "Preparing the shaders: first a check that this PC keeps them from one session to the next...".to_string(),
            Step::Compute if app.shaders.after_compute.is_some() => format!("Preparing the shaders: the game's compute shaders, {done} of {total} ({:.0}%).{}", frac * 100.0, if budget.is_empty() { eta.as_str() } else { budget.as_str() }),
            Step::Compute => format!("Preparing the compute shaders: {done} of {total} ({:.0}%).{eta}", frac * 100.0),
            Step::NewGame => format!("Preparing the shaders, step 1 of {steps}: the game plays a new game in the background, {done} of {total}.{budget}{compute}"),
            Step::Levels(_) => format!("Preparing the shaders, step {} of {steps}: the game visits its levels in the background, {done} of {total}.{budget}{compute}", step.number()),
        };
        // (The launcher's status line too, once: the player waiting for Play sees it there.)
        if let Some(m) = slow {
            if !app.shaders.slow_noted {
                app.shaders.slow_noted = true;
                app.set_status(&format!("Preparing the shaders takes longer on this PC: about {m} more minutes. The Shaders page shows the progress."));
            }
        }
        unsafe {
            ui::set_text(h(app.shaders.status), &text);
            ui::set_progress(h(app.shaders.bar), (frac * 1000.0) as u32);
        }
    }
}

/// The whole preparation's time left and its projected whole time (s), from the steps' times so
/// far against the development PC's (`REF_*`): the speed factor the new game's time, else the
/// levels parts', else the current step's own progress; the compute list's rest after the
/// levels at the rate it has been made so far (None before anything is measured).
fn estimate(app: &App, step: Step, frac: f64, compute: Option<(u64, u64)>) -> Option<(f64, f64)> {
    let el = app.shaders.prep_started?.elapsed().as_secs_f64();
    let step_el = app.shaders.started?.elapsed().as_secs_f64();
    let parts = &app.shaders.part_secs;
    // (A step's own progress counts once a tenth of it is done: the compute step's and a part's.)
    let own = frac > 0.1 && matches!(step, Step::Compute | Step::Levels(_));
    // The PC's speed against the development PC's: the levels parts' times once there are some,
    // else the compute step's (its CPU), else the current step's own progress. (None during the
    // new game: its waits run in real time on any PC.)
    let f = if !parts.is_empty() {
        parts.iter().sum::<f64>() / parts.len() as f64 / REF_PART
    } else if let Some(c) = app.shaders.compute_secs {
        c / REF_COMPUTE
    } else if own {
        step_el / frac / step.yardstick()
    } else {
        return None;
    };
    let f = f.clamp(0.6, 30.0);
    let cur = if own { step_el / frac } else { step.yardstick() * f };
    // The levels' parts still to come.
    let after = match step {
        Step::Compute if app.shaders.after_compute.is_some() => LEVEL_PARTS as f64,
        Step::Levels(i) => (LEVEL_PARTS - i) as f64,
        _ => 0.0,
    };
    let mut left = (cur - step_el).max(0.0) + after * REF_PART * f;
    // (The levels' own lists: a rest made after their last part.)
    if let Some((made, all)) = compute.filter(|_| matches!(step, Step::Levels(_))) {
        if all > 0 && made < all && made > 0 {
            let rate = made as f64 / el.max(1.0);
            let rest = (all - made) as f64 - rate * left;
            if rest > 0.0 {
                // (Faster once the game is done: the whole CPU.)
                left += rest / (rate * 1.5);
            }
        }
    }
    Some((left, el + left))
}

pub fn command(app: &mut App, id: i32, code: u32) {
    match (id, code) {
        (ID_PREPARE, 0) => start(app, false),
        (ID_STOP, 0) => stop(app),
        (ID_AUTO, 0) => {
            app.launch.prepare_shaders = unsafe { ui::checked(h(app.shaders.auto)) };
            app.save_launch();
        }
        _ => {}
    }
}
