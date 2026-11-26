//! dsport: runs the PS5 Demon's Souls executable (PPSA01341 01.007.000) natively
//! on Windows x86-64 with high-level emulation of the PS5 system libraries.

// No console window: players start it from the launcher or Explorer (redirected output still
// reaches development tools).
#![windows_subsystem = "windows"]

#[macro_use]
mod log;
mod ehframe;
mod achievements;
mod audio;
mod cpucheck;
mod exception;
mod gpu;
mod guestprof;
mod heap;
mod heapsites;
mod hosted;
mod hostcensus;
mod placement;
mod hostload;
mod hle;
mod input;
mod loader;
mod pspad;
mod mem;
mod sections;
mod spatial;
mod patch;
mod pcsettings;
mod jobidle;
mod camhook;
mod gamestep;
mod game;
mod dispmode;
mod display;
mod displayclock;
mod jitter;
mod engine;
mod thread;
mod titlefix;
mod vgdiag;
mod cullread;
mod cpudraw;
mod titlemem;
mod textbox;
mod uistate;
mod vrr;
mod tls;
mod prepplay;

use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::time::{Duration, Instant};

pub struct Config {
    /// Where the port keeps its files (`dsconfig::paths`: the checkout's local-data, a portable
    /// folder, or the player's AppData folders).
    pub paths: dsconfig::paths::Paths,
    /// Why the game folder cannot be used (told to the player at start), if so.
    pub game_problem: Option<String>,
    /// No game folder chosen yet (`game_problem` says so): a player's start opens the launcher.
    pub game_unset: bool,
    /// Application root (`PPSA01341-app0`): read-only assets mounted at /app0.
    pub app: PathBuf,
    /// Directory holding the decrypted executable and `sce_module/`.
    pub code: PathBuf,
    /// The game's files: the application folder, or its `.zar` archive (`game`).
    pub source: dsconfig::zar::Source,
    /// The decrypted executable's folder in `source` ("" or "decrypted").
    pub code_dir: String,
    pub run_dir: PathBuf,
    pub seconds: Option<u64>,
    /// Development control channel (see `poll_control`).
    pub control: Option<PathBuf>,
    /// `--cache <dir>`: where compiled pipelines (`pipelines/`) and the
    /// learned shader specs (`shader-specs.txt`, `.code`) are kept, instead
    /// of local-data (play-test packages extracted side by side share one).
    pub cache: Option<PathBuf>,
    /// `--pipeline-test <capture.txt>`: create pipelines for a captured frame and exit.
    pub pipeline_test: Option<String>,
    /// `--record-bench <capture> [<cs address>]` (`gpu::exec::record_bench`).
    pub record_bench: Option<(String, Option<u64>)>,
    /// `--compile-bench [min instructions] [max programs]` (`gpu::compbench`).
    pub compile_bench: Option<Vec<String>>,
    /// `--precompile-all [threads]` (`gpu::compbench::precompile_all`).
    pub precompile_all: Option<usize>,
    /// `--pb-check write|read` (`gpu::pcache::pb_check`: the launcher's check that the store's
    /// pipelines reach from one process to the next on this PC).
    pub pb_check: Option<String>,
    /// `--translation-fingerprint`: prints `translation = <fingerprint>` (the launcher's check
    /// that the prepared shaders match this build: `gpu::compbench::translation_fingerprint`).
    pub translation_fingerprint: bool,
    /// `--progress <file>`: the precompile's progress for the launcher.
    pub progress: Option<PathBuf>,
    /// `--safe-mode` (the launcher's, for one start): the PC settings' defaults for this session
    /// only, in a window, without an upscaler, with the GPU front end's cautious fallbacks.
    pub safe_mode: bool,
    /// `--dispatch-bench` (`gpu::dispbench`).
    pub dispatch_bench: bool,
    /// `--prepare-play <newgame|tour>`: the shader preparation's self-play (`prepplay`).
    pub prepare_play: Option<prepplay::Stage>,
    /// `--prepare-threads <n>`: the self-play's background compute preparation's threads (0: half
    /// the logical CPUs).
    pub prepare_threads: usize,
    /// `--prepare-chunk <i>/<n>`: the self-play's levels stage plays the i-th of n parts of the
    /// levels (each its own session: the memory a session's level changes take is bounded).
    pub prepare_chunk: Option<(usize, usize)>,
    /// `--prepare-deadline <unix seconds>`: when the whole preparation's time is up (the levels
    /// stage ends at the next level after it).
    pub prepare_deadline: Option<u64>,
    pub trace: bool,
    pub lenient: bool,
    pub debug: bool,
    pub headless: bool,
}

static RUN_DIR: std::sync::OnceLock<PathBuf> = std::sync::OnceLock::new();
static PATHS: std::sync::OnceLock<dsconfig::paths::Paths> = std::sync::OnceLock::new();

/// Where the port keeps its files (set at start).
pub fn paths() -> &'static dsconfig::paths::Paths {
    PATHS.get_or_init(dsconfig::paths::Paths::current)
}

/// This run's output directory (logs, captures).
pub fn run_dir() -> Option<&'static Path> {
    RUN_DIR.get().map(|p| p.as_path())
}

/// The host heap (`heap`: mimalloc or the Windows heap) reporting implausibly
/// large requests (almost always a guest value misread as a size) with the
/// requesting host frames, before the request fails and aborts the process.
struct GuardedAlloc;

const HUGE_ALLOC: usize = 4 << 30;

#[cold]
fn report_huge_alloc(size: usize) {
    use std::io::Write;
    let mut frames = [std::ptr::null_mut::<std::ffi::c_void>(); 24];
    let n = unsafe { windows_sys::Win32::System::Diagnostics::Debug::RtlCaptureStackBackTrace(1, 24, frames.as_mut_ptr(), std::ptr::null_mut()) } as usize;
    let mut err = std::io::stderr().lock();
    let _ = writeln!(err, "huge host allocation of {size} bytes ({size:#x}) requested; frames:");
    for f in &frames[..n] {
        let _ = writeln!(err, "    {}", exception::host_symbol(*f as u64));
    }
}

unsafe impl std::alloc::GlobalAlloc for GuardedAlloc {
    unsafe fn alloc(&self, l: std::alloc::Layout) -> *mut u8 {
        if l.size() > HUGE_ALLOC {
            report_huge_alloc(l.size());
        }
        let p = unsafe { heap::HostAlloc.alloc(l) };
        if heapsites::tracked(l.size()) {
            heapsites::note_alloc(p, l.size());
        }
        p
    }
    unsafe fn alloc_zeroed(&self, l: std::alloc::Layout) -> *mut u8 {
        if l.size() > HUGE_ALLOC {
            report_huge_alloc(l.size());
        }
        let p = unsafe { heap::HostAlloc.alloc_zeroed(l) };
        if heapsites::tracked(l.size()) {
            heapsites::note_alloc(p, l.size());
        }
        p
    }
    unsafe fn dealloc(&self, p: *mut u8, l: std::alloc::Layout) {
        if heapsites::tracked(l.size()) {
            heapsites::note_free(p);
        }
        unsafe { heap::HostAlloc.dealloc(p, l) }
    }
    unsafe fn realloc(&self, p: *mut u8, l: std::alloc::Layout, new_size: usize) -> *mut u8 {
        if new_size > HUGE_ALLOC {
            report_huge_alloc(new_size);
        }
        if heapsites::tracked(l.size()) {
            heapsites::note_free(p);
        }
        let q = unsafe { heap::HostAlloc.realloc(p, l, new_size) };
        if heapsites::tracked(new_size) {
            heapsites::note_alloc(q, new_size);
        }
        q
    }
}

#[global_allocator]
static ALLOCATOR: GuardedAlloc = GuardedAlloc;

/// A failure before the log exists, written to `early-failure.txt` in a dated folder of its own
/// in the logs folder, as a session's (so the launcher's Open logs selects it), personal paths
/// left out as in the log; the sentence naming that file for the player's message, or nothing
/// when it could not be written.
fn early_failure_note(what: &str) -> String {
    let dir = paths().logs().join(utc_stamp());
    let file = dir.join("early-failure.txt");
    let text = format!("{}\n", log::redact(what));
    if std::fs::create_dir_all(&dir).is_ok() && std::fs::write(&file, text).is_ok() {
        format!("\n\nThe details are in {} (the launcher's Open logs shows its folder).", log::redact(&file.display().to_string()))
    } else {
        String::new()
    }
}

pub(crate) fn utc_stamp() -> String {
    let secs = std::time::SystemTime::now().duration_since(std::time::UNIX_EPOCH).unwrap().as_secs() as i64;
    let days = secs.div_euclid(86400);
    let rem = secs.rem_euclid(86400);
    // Civil-from-days (Howard Hinnant).
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

fn parse_args() -> Config {
    let mut paths = dsconfig::paths::Paths::current();
    let mut seconds = None;
    let mut trace = false;
    let mut lenient = false;
    let mut debug = false;
    let mut headless = false;
    let mut control = None;
    let mut cache = None;
    let mut app = None;
    let mut pipeline_test = None;
    let mut compile_bench: Option<Vec<String>> = None;
    let mut precompile_all: Option<usize> = None;
    let mut pb_check: Option<String> = None;
    let mut translation_fingerprint = false;
    let mut progress = None;
    let mut safe_mode = false;
    let mut dispatch_bench = false;
    let mut record_bench = None;
    let mut prepare_play = None;
    let mut prepare_threads = 0usize;
    let mut prepare_chunk = None;
    let mut prepare_deadline = None;
    let mut args = std::env::args().skip(1).peekable();
    while let Some(a) = args.next() {
        match a.as_str() {
            "--seconds" => seconds = args.next().and_then(|s| s.parse().ok()),
            "--trace" => trace = true,
            "--lenient" => lenient = true,
            "--debug" => debug = true,
            "--headless" => headless = true,
            // (Read by `hosted::wants_host`: front end 2 unhosted, its arming off.)
            "--no-host" => {}
            "--root" => {
                // A development checkout's root (its local-data holds everything).
                let root = PathBuf::from(args.next().expect("--root needs a path"));
                let data = root.join("local-data");
                paths = dsconfig::paths::Paths { layout: dsconfig::paths::Layout::Dev, exe_dir: paths.exe_dir.clone(), user: data.clone(), cache: data, root: Some(root) };
            }
            "--control" => control = Some(PathBuf::from(args.next().expect("--control needs a path"))),
            "--cache" => cache = Some(PathBuf::from(args.next().expect("--cache needs a directory"))),
            "--app" => app = Some(PathBuf::from(args.next().expect("--app needs a path"))),
            "--pipeline-test" => pipeline_test = Some(args.next().expect("--pipeline-test needs a capture file")),
            "--dispatch-bench" => dispatch_bench = true,
            "--progress" => progress = Some(PathBuf::from(args.next().expect("--progress needs a file"))),
            "--safe-mode" => safe_mode = true,
            "--prepare-play" => {
                let s = args.next().unwrap_or_default();
                prepare_play = Some(prepplay::Stage::parse(&s).unwrap_or_else(|| {
                    eprintln!("--prepare-play needs a stage (newgame or tour), not '{s}'");
                    std::process::exit(2);
                }));
            }
            "--prepare-threads" => prepare_threads = args.next().and_then(|v| v.parse().ok()).unwrap_or(0),
            "--prepare-chunk" => {
                prepare_chunk = args.next().and_then(|v| {
                    let (i, n) = v.split_once('/')?;
                    let (i, n) = (i.trim().parse::<usize>().ok()?, n.trim().parse::<usize>().ok()?);
                    (n >= 1 && (1..=n).contains(&i)).then_some((i, n))
                })
            }
            "--prepare-deadline" => prepare_deadline = args.next().and_then(|v| v.parse().ok()),
            "--pb-test" => std::process::exit(gpu::pcache::pb_test()),
            "--pb-check" => pb_check = Some(args.next().unwrap_or_default()),
            "--compile-bench" => {
                let min = args.next().expect("--compile-bench needs <min instructions> <max programs>");
                let max = args.next().expect("--compile-bench needs <min instructions> <max programs>");
                compile_bench = Some(vec![min, max]);
            }
            "--precompile-all" => {
                // (An optional thread count.)
                let n = args.peek().and_then(|a| a.parse::<usize>().ok());
                if n.is_some() {
                    args.next();
                }
                precompile_all = Some(n.unwrap_or(8));
            }
            "--translation-fingerprint" => translation_fingerprint = true,
            "--record-bench" => {
                let file = args.next().expect("--record-bench needs a capture file");
                let cs = args.next().and_then(|a| u64::from_str_radix(a.trim_start_matches("0x"), 16).ok());
                record_bench = Some((file, cs));
            }
            other => {
                eprintln!("unknown argument {other}");
                std::process::exit(2);
            }
        }
    }
    if safe_mode {
        // The GPU front end's cautious fallbacks (no arming of GPU-written pages, no held
        // pages), no upscaler library loaded and no NVIDIA Streamline (no DLSS, Frame Generation
        // or Reflex): set before any thread starts or reads them.
        for (k, v) in [("DS_FE2_ARM_GPU", "0"), ("DS_C4", "0"), ("DS_UPSCALER", "off"), ("DS_STREAMLINE", "0")] {
            // SAFETY: still the only thread (parse_args runs before any is started).
            unsafe { set_unless_given(k, v) };
        }
    }
    // The game: `--app`, else the folder chosen in the launcher (or the checkout's own copy),
    // checked as the launcher checks it (`dsconfig::game`).
    // (`DS_GAME=<folder or .zar>`: another copy for a development run, the archive A/Bs' arm.)
    let chosen = app.or_else(|| std::env::var_os("DS_GAME").filter(|v| !v.is_empty()).map(PathBuf::from)).or_else(|| paths.chosen_game());
    let (app, code, source, code_dir, game_problem) = match chosen.as_deref().map(dsconfig::game::check) {
        Some(Ok(g)) => (g.app, g.code, g.source, g.code_dir, None),
        Some(Err(e)) => {
            let dir = chosen.clone().unwrap_or_default();
            let what = if dsconfig::zar::is_archive_path(&dir) { "The game archive" } else { "The game folder" };
            (dir.clone(), dir.join("decrypted"), dsconfig::zar::Source::Dir(dir.clone()), "decrypted".to_string(), Some(format!("{e}\n\n{what}: {}\nStart {} to choose another.", dir.display(), dsconfig::paths::LAUNCHER_EXE)))
        }
        None => (PathBuf::new(), PathBuf::new(), dsconfig::zar::Source::Dir(PathBuf::new()), String::new(), Some(format!("The game folder is not set. Start {} and choose it.", dsconfig::paths::LAUNCHER_EXE))),
    };
    let game_unset = chosen.is_none();
    // (The launcher's quick checks log apart from the game's sessions: `Paths::check_logs`.)
    let run_dir = if pb_check.is_some() || translation_fingerprint { paths.check_logs() } else { paths.logs() }.join(utc_stamp());
    Config {
        code,
        app,
        source,
        code_dir,
        run_dir,
        paths,
        game_problem,
        game_unset,
        seconds,
        trace,
        lenient,
        debug,
        // The launcher's translation check and shader preparation (and the compile bench) run
        // no game: no window (each showed an empty one until 2026-11-14) and no message boxes.
        headless: headless || translation_fingerprint || precompile_all.is_some() || compile_bench.is_some() || prepare_play.is_some() || pb_check.is_some(),
        control,
        cache,
        pipeline_test,
        record_bench,
        compile_bench,
        precompile_all,
        pb_check,
        translation_fingerprint,
        progress,
        safe_mode,
        dispatch_bench,
        prepare_play,
        prepare_threads,
        prepare_chunk,
        prepare_deadline,
    }
}

/// Keeps the newest `keep` run directories without a KEEP marker
/// (`DS_KEEP_RUNS=<n>` sets the number); those with one are never removed
/// and take no place among them (until 2026-11-05 they did, and with ~170
/// marked the unmarked ones lasted ~30 boots). A run marks itself when its
/// GPU device is lost or it crashes (`keep_run`).
fn prune_runs(runs: &Path, keep: usize) {
    let keep = std::env::var("DS_KEEP_RUNS").ok().and_then(|v| v.parse().ok()).unwrap_or(keep);
    let Ok(rd) = std::fs::read_dir(runs) else { return };
    // (Run folders are named by their time; the quick checks' `checks` folder is not one.)
    let dated = |p: &Path| p.file_name().and_then(|n| n.to_str()).is_some_and(|n| n.starts_with(|c: char| c.is_ascii_digit()));
    let mut dirs: Vec<PathBuf> = rd.flatten().map(|e| e.path()).filter(|p| p.is_dir() && dated(p) && !p.join("KEEP").exists()).collect();
    dirs.sort();
    if dirs.len() <= keep {
        return;
    }
    let n = dirs.len() - keep;
    for d in dirs.into_iter().take(n) {
        let _ = std::fs::remove_dir_all(&d);
    }
}

/// Keeps this run's directory from pruning (`prune_runs`): a line naming
/// why, appended to its KEEP marker.
pub fn keep_run(reason: &str) {
    if let Some(dir) = RUN_DIR.get() {
        use std::io::Write;
        if let Ok(mut f) = std::fs::OpenOptions::new().create(true).append(true).open(dir.join("KEEP")) {
            let _ = writeln!(f, "{reason}");
        }
    }
}

/// Development control channel (`--control <file>`): executes complete lines
/// appended to the file since the last poll.
///   press <buttons> [hold]   e.g. `press down`, `press cross 10`, `press ls_up+r1 30`; the hold
///                            is in flips (default 6), or wall-clock with a suffix: `4s`, `300ms`
///   key <input> [ms]         hold a key or mouse button of the keyboard and mouse layout
///                            (controls.txt names: `key W 1000`, `key Mouse1`; default 200 ms)
///   mouse <dx> <dy>          mouse motion in counts; `wheel <n>` wheel notches (up positive)
///   capture on|off           mouse input as if the window had captured the cursor (headless)
///   stick l|r <x> <y> [ms]   hold a stick at a tilt (-128..127 from the centre; default 1000 ms)
///   find32 <hex u32>|<lo>-<hi>  guest addresses (4-byte aligned) holding a 32-bit value (first 64)
///   fscan start|changed|same|floats [+|-|=|*]  search the title's flexible memory for a value by
///                            how it changes (pages, then f32 values; see `fscan`)
///   fmat                     yaw-only rotation matrices in the title's flexible memory (see `fmat`)
///   cam                      the gameplay camera's controllers (raw mouse look), with
///                            DS_CAMDEBUG=1 also its rigs
///   pcset <setting> <value>  a PC setting (`Setting_PC_...`) as if changed in the SETTINGS screen
///   convars                  the PC settings' title console variables and their values
///   convar <name> <value>    one of them set (`title`: the title's own again; tests)
///   shot <name>              screenshot of the next presented flip -> <name>.png (the run directory)
///   burst <name> <n> [x y w h] [half]  shots of the next n flips, each one (cropped, at half size) -> <name>_NNN.png
///   dump                     dump all GPU images at the next flip
///   trace                    log every draw/dispatch of an upcoming flip (FRAME lines)
///   loglevel trace|debug|info  change logging at run time (trace = every HLE call)
///   textreset                log text-hook strings again (`DS_TEXT_HOOK`)
///   mem <addr> <len>         write guest memory (hex) to mem-<addr>.bin in the run directory
///   emucheck <cs,..>[:<n>]   interpreter check of the next dispatches of those shaders (`off`)
///   find <hex u64>|<lo>-<hi> guest addresses (8-byte aligned) holding that value (or a value
///                            in [lo, hi), listed) (first 64)
///   watch <slot> <addr>|off [w|rw] [len] [all|guest]  hardware watchpoint (DR0-3) at run time
///   findstr <text>           guest addresses holding that ASCII text (first 64)
///   poke <addr> <u64> [<mask>]  write guest memory (experiments; logged)
///   note <text>              log the text (a marker for scripted runs)
///   quit                     orderly shutdown
fn poll_control(path: &Path, offset: &mut u64) {
    use std::io::{Read, Seek, SeekFrom};
    // Lines the title's automation asked for first (`INJECTED`).
    let mut buf: String = std::mem::take(&mut *INJECTED.lock()).into_iter().map(|l| l + "\n").collect();
    if let Ok(mut f) = std::fs::File::open(path) {
        let len = f.metadata().map(|m| m.len()).unwrap_or(0);
        if len < *offset {
            *offset = 0; // truncated: start over
        }
        let mut file = String::new();
        if len > *offset && f.seek(SeekFrom::Start(*offset)).is_ok() && f.read_to_string(&mut file).is_ok() {
            if let Some(end) = file.rfind('\n') {
                *offset += end as u64 + 1;
                buf.push_str(&file[..=end]);
            }
        }
    }
    if buf.is_empty() {
        return;
    }
    for line in buf.lines().map(str::trim).filter(|l| !l.is_empty() && !l.starts_with('#')) {
        let mut parts = line.split_whitespace();
        let cmd = parts.next().unwrap_or("");
        let flip = gpu::FRAMES.load(std::sync::atomic::Ordering::Relaxed);
        let result = match cmd {
            "press" => {
                let names = parts.next().unwrap_or("");
                match parts.next() {
                    Some(h) if h.ends_with("ms") => match h.trim_end_matches("ms").parse::<u64>() {
                        Ok(ms) => input::press_for(names, Duration::from_millis(ms)),
                        Err(_) => Err(format!("bad hold '{h}'")),
                    },
                    Some(h) if h.ends_with('s') => match h.trim_end_matches('s').parse::<f64>() {
                        Ok(s) if s > 0.0 && s < 3600.0 => input::press_for(names, Duration::from_secs_f64(s)),
                        _ => Err(format!("bad hold '{h}'")),
                    },
                    h => input::press_now(names, h.and_then(|h| h.parse().ok()).unwrap_or(6)),
                }
            }
            "gamestep" => {
                // `gamestep`: the title's frame steps since the last report (game time against
                // real time); `gamestep log <n>`: the last n, one a line.
                match (parts.next(), parts.next().and_then(|v| v.parse::<usize>().ok())) {
                    (Some("log"), n) => info!("gamestep log:\n{}", gamestep::log(n.unwrap_or(120))),
                    (Some("pair"), n) => gamestep::pair_log(n.unwrap_or(60) as u32),
                    (Some("dump"), n) => info!("{}", gamestep::dump(n.unwrap_or(2000))),
                    _ => info!("{}", gamestep::report()),
                }
                Ok(())
            }
            "walktest" => {
                // `walktest <x> <y> <ms> [name]`: the left stick held at a tilt, the player's
                // speed over the hold's steady part logged (`gamestep::walktest`).
                let x = parts.next().and_then(|v| v.parse::<i32>().ok());
                let y = parts.next().and_then(|v| v.parse::<i32>().ok());
                let ms = parts.next().and_then(|v| v.parse::<u64>().ok());
                match (x, y, ms) {
                    (Some(x), Some(y), Some(ms)) if (1000..=60_000).contains(&ms) => {
                        gamestep::walktest(x, y, ms, parts.next().unwrap_or("-").to_string());
                        Ok(())
                    }
                    _ => Err("usage: walktest <x> <y> <ms 1000..60000> [name]".into()),
                }
            }
            "itemcensus" => {
                // `itemcensus` (`DS_ITEM_CENSUS=1`): a frame's prepared draws against the frame
                // before by content, the totals since the last report.
                info!("{}", gpu::itemcensus::report());
                Ok(())
            }
            "turntest" => {
                // `turntest <x> <ms> [name]`: the right stick held sideways, the camera's turn
                // over the hold's steady part logged (`gamestep::turntest`).
                let x = parts.next().and_then(|v| v.parse::<i32>().ok());
                let ms = parts.next().and_then(|v| v.parse::<u64>().ok());
                match (x, ms) {
                    (Some(x), Some(ms)) if (2000..=60_000).contains(&ms) => {
                        gamestep::turntest(x, ms, parts.next().unwrap_or("-").to_string());
                        Ok(())
                    }
                    _ => Err("usage: turntest <x> <ms 2000..60000> [name]".into()),
                }
            }
            "pcset" => {
                // `pcset <Setting_PC_...> <value>`: a PC setting as if changed in the menu.
                let name = parts.next().unwrap_or("");
                let value = parts.next().unwrap_or("");
                pcsettings::store::set_by_name(name, value)
            }
            "rsize" => {
                // Diagnosis: `rsize <w>x<h>` (0x0: the title's own) in a run without a window.
                let v = parts.next().unwrap_or("");
                match v.split_once(['x', 'X']).and_then(|(w, h)| Some((w.parse::<u32>().ok()?, h.parse::<u32>().ok()?))) {
                    Some((w, h)) => dispmode::set_headless_size(w, h),
                    None => Err(format!("rsize: '{v}' is not <w>x<h>")),
                }
            }
            "uielems" => {
                // Diagnosis: `uielems <tag>` writes every UIElement component's first bytes.
                let tag = parts.next().unwrap_or("x");
                match RUN_DIR.get() {
                    Some(dir) => uistate::dump_elements(dir, tag).map(|m| info!("{m}")),
                    None => Err("no run directory".into()),
                }
            }
            "bindinput" => {
                // Tests: `bindinput <input>` (as in controls.txt) for a key binding waiting for its key.
                pcsettings::bindings::test_input(parts.next().unwrap_or(""))
            }
            "pointer" => {
                // Tests: `pointer <x> <y>` moves the menu pointer to canvas coordinates (1920x1080);
                // `pointer screen <x> <y>` to a point of the picture as shown, through the window;
                // `pointer click|right|wheel <n>`.
                use pcsettings::pointer::{Event, event, test_move, test_screen};
                match (parts.next(), parts.next()) {
                    (Some("screen"), Some(x)) => match (x.parse::<f32>(), parts.next().and_then(|y| y.parse::<f32>().ok())) {
                        (Ok(x), Some(y)) => test_screen(x, y),
                        _ => Err("usage: pointer screen <x> <y>".into()),
                    },
                    (Some("click"), _) => {
                        event(Event::Left);
                        Ok(())
                    }
                    (Some("right"), _) => {
                        event(Event::Right);
                        Ok(())
                    }
                    (Some("wheel"), Some(n)) => n.parse::<i32>().map(|n| event(Event::Wheel(n))).map_err(|e| e.to_string()),
                    (Some(x), Some(y)) => match (x.parse::<f32>(), y.parse::<f32>()) {
                        (Ok(x), Ok(y)) => {
                            test_move(x, y);
                            Ok(())
                        }
                        _ => Err("usage: pointer <x> <y> | click | right | wheel <n>".into()),
                    },
                    _ => Err("usage: pointer <x> <y> | click | right | wheel <n>".into()),
                }
            }
            "uinodes" => {
                // Diagnosis: `uinodes <tag>` writes every runtime UI node's first bytes.
                let tag = parts.next().unwrap_or("x");
                match RUN_DIR.get() {
                    Some(dir) => uistate::dump_nodes(dir, tag).map(|m| info!("{m}")),
                    None => Err("no run directory".into()),
                }
            }
            "uistack" => {
                // Diagnosis: the title's UI manager: the active menu stack and HUD scenes.
                info!("{}", uistate::report());
                Ok(())
            }
            "convars" => {
                crate::pcsettings::convar::report();
                Ok(())
            }
            "convar" => {
                // `convar <name> <value|title>`: one of the PC settings' title console variables.
                let name = parts.next().unwrap_or("");
                let value = parts.next().unwrap_or("");
                crate::pcsettings::convar::set_for_test(name, value)
            }
            "shot" => {
                let name: String = parts.next().unwrap_or("x").chars().filter(|c| c.is_ascii_alphanumeric() || *c == '_' || *c == '-').collect();
                *hle::libs::videoout::SHOT_REQUEST.lock() = Some(name);
                Ok(())
            }
            "framewatch" => {
                // `framewatch 1|0`: the presenter's one-frame flash watch (`gpu::vk::framewatch`).
                let on = parts.next() != Some("0");
                gpu::vk::framewatch::ON.store(on, std::sync::atomic::Ordering::Relaxed);
                info!("control: frame watch {}", if on { "on" } else { "off" });
                Ok(())
            }
            "burst" => {
                let name: String = parts.next().unwrap_or("burst").chars().filter(|c| c.is_ascii_alphanumeric() || *c == '_' || *c == '-').collect();
                let count = parts.next().and_then(|v| v.parse::<u32>().ok()).unwrap_or(30).clamp(1, 600);
                let rest: Vec<&str> = parts.collect();
                let half = rest.contains(&"half");
                let nums: Vec<u32> = rest.iter().filter_map(|v| v.parse().ok()).collect();
                let crop = (nums.len() == 4).then(|| (nums[0], nums[1], nums[2], nums[3]));
                *hle::libs::videoout::BURST.lock() = Some(hle::libs::videoout::Burst { name, left: count, index: 0, crop, half });
                Ok(())
            }
            "dump" => {
                gpu::DUMP_REQUEST.store(true, std::sync::atomic::Ordering::Relaxed);
                Ok(())
            }
            "breaklog" => {
                let n = parts.next().and_then(|v| v.parse().ok()).unwrap_or(1000);
                exception::BREAK_VERBOSE.store(n, std::sync::atomic::Ordering::Relaxed);
                Ok(())
            }
            "loglevel" => {
                let (lvl, hle) = match parts.next().unwrap_or("") {
                    "trace" => (log::Level::Debug, true),
                    "debug" => (log::Level::Debug, false),
                    "info" => (log::Level::Info, false),
                    other => {
                        warn!("control: unknown log level '{other}'");
                        (log::Level::Info, false)
                    }
                };
                log::LEVEL.store(lvl as u8, std::sync::atomic::Ordering::Relaxed);
                log::TRACE_HLE.store(hle, std::sync::atomic::Ordering::Relaxed);
                Ok(())
            }
            "ime" => {
                // Text for a pending system IME dialog (e.g. the character name).
                let text = line.trim_start().strip_prefix("ime").unwrap_or("").trim().to_string();
                *hle::libs::ime_dialog::CONTROL_TEXT.lock() = Some(text);
                Ok(())
            }
            "eqbias" => {
                // Diagnosis of EQUAL depth tests: `eqbias <units>|always|off`.
                let v = match parts.next().unwrap_or("off") {
                    "off" => None,
                    "always" => Some(f32::INFINITY),
                    x => x.parse().ok(),
                };
                gpu::exec::set_equal_bias(v);
                Ok(())
            }
            "find" => {
                // Debug: `find <hex u64>` or `find <hex lo>-<hex hi>`: readable
                // guest memory holding the value (or a value in [lo, hi)) at an
                // 8-byte aligned address (first 64 hits; ranges list the values).
                let hex = |s: &str| u64::from_str_radix(s.trim().trim_start_matches("0x"), 16).ok();
                let arg = parts.next().unwrap_or("");
                let range = match arg.split_once('-') {
                    Some((lo, hi)) => hex(lo).zip(hex(hi)),
                    None => hex(arg).map(|v| (v, v + 1)),
                };
                match range {
                    Some((vlo, vhi)) => {
                        let spans: Vec<(u64, u64, String)> = mem::MEM.lock().vmas.values().filter(|m| m.kind != mem::VmaKind::Reserved && m.prot & 1 != 0).map(|m| (m.start, m.end(), m.name.clone())).collect();
                        let mut hits = Vec::new();
                        let t0 = Instant::now();
                        for (lo, hi, name) in &spans {
                            let mut a = (*lo + 7) & !7;
                            while a + 8 <= *hi && hits.len() < 64 {
                                let v = unsafe { *(a as *const u64) };
                                if (vlo..vhi).contains(&v) {
                                    hits.push(if vhi - vlo > 1 { format!("{a:#x} ({name}) = {v:#x}") } else { format!("{a:#x} ({name})") });
                                }
                                a += 8;
                            }
                        }
                        info!("find {arg}: {} hit(s) in {} mappings ({} ms): {}", hits.len(), spans.len(), t0.elapsed().as_millis(), hits.join(", "));
                        Ok(())
                    }
                    None => Err("usage: find <hex u64> | <hex lo>-<hex hi>".into()),
                }
            }
            "fmat" => fmat(),
            "note" => {
                // A marker in the log for scripted runs.
                info!("control: note {}", parts.by_ref().collect::<Vec<_>>().join(" "));
                Ok(())
            }
            "camdump" => {
                // Debug (`DS_CAMDEBUG=1`): the camera rigs' controllers' fields.
                info!("camdump:
{}", camhook::dump());
                Ok(())
            }
            "cam" => {
                // Debug: the gameplay camera's controllers (`camhook`) and,
                // with DS_CAMDEBUG=1, its rigs.
                info!("cam: {}", camhook::describe());
                Ok(())
            }
            "wshot" => {
                // The next presented frame as the window shows it
                // (`window-<n>.png` in the run directory).
                gpu::vk::present::shot_next();
                Ok(())
            }
            "window" => {
                // `window fullscreen|windowed|toggle`: what Alt+Enter does;
                // `window size <w>x<h>`: the windowed window's client size.
                let size = |v: &str| v.split_once('x').and_then(|(w, h)| Some((w.parse::<u32>().ok()?, h.parse::<u32>().ok()?))).filter(|&(w, h)| (160..=16384).contains(&w) && (90..=16384).contains(&h));
                match (parts.next(), parts.next()) {
                    (Some("fullscreen"), _) => Ok(gpu::vk::window::set_fullscreen(Some(true))),
                    (Some("windowed"), _) => Ok(gpu::vk::window::set_fullscreen(Some(false))),
                    (Some("toggle"), _) => Ok(gpu::vk::window::set_fullscreen(None)),
                    (Some("exclusive"), _) => Ok(gpu::vk::window::set_window_mode(2)),
                    // `window background` / `window foreground`: what Alt+Tab away and back does
                    // (the window's own activation messages).
                    (Some("background"), _) => Ok(gpu::vk::window::simulate_activation(false)),
                    (Some("foreground"), _) => Ok(gpu::vk::window::simulate_activation(true)),
                    // `window displaychange`: what a monitor's mode change sends the window.
                    (Some("displaychange"), _) => Ok(gpu::vk::window::simulate_display_change()),
                    (Some("size"), Some(v)) if size(v).is_some() => {
                        let (w, h) = size(v).unwrap();
                        gpu::vk::window::set_client_size(w, h);
                        Ok(())
                    }
                    _ => Err("usage: window fullscreen|exclusive|windowed|toggle|background|foreground | window size <w>x<h>".into()),
                }
            }
            "fscan" => {
                // Debug (finding a value the title keeps, e.g. the camera's
                // angles): `fscan start` hashes the pages of the title's
                // flexible memory; `fscan changed` keeps the pages that changed
                // since (intersected with earlier ones), `fscan same` drops the
                // pages that changed and copies the rest once few remain;
                // `fscan floats [+|-|=|*]` lists the copied pages' f32 values
                // that rose, fell, stayed or changed since the copy (keeping
                // those as the candidates, first 48 listed) and copies again.
                fscan(parts.next().unwrap_or(""), parts.next().unwrap_or("*"))
            }
            "find32" => {
                // Debug: `find32 <hex u32>` or `find32 <hex lo>-<hex hi>`: readable
                // guest memory holding the 32-bit value (or one in [lo, hi)) at a
                // 4-byte aligned address (first 64 hits, with the value as f32).
                let hex = |s: &str| u32::from_str_radix(s.trim().trim_start_matches("0x"), 16).ok();
                let arg = parts.next().unwrap_or("");
                let range = match arg.split_once('-') {
                    Some((lo, hi)) => hex(lo).zip(hex(hi)),
                    None => hex(arg).map(|v| (v, v.saturating_add(1))),
                };
                match range {
                    Some((vlo, vhi)) => {
                        let spans: Vec<(u64, u64, String)> = mem::MEM.lock().vmas.values().filter(|m| m.kind != mem::VmaKind::Reserved && m.prot & 1 != 0).map(|m| (m.start, m.end(), m.name.clone())).collect();
                        let mut hits = Vec::new();
                        let t0 = Instant::now();
                        for (lo, hi, name) in &spans {
                            let mut a = (*lo + 3) & !3;
                            while a + 4 <= *hi && hits.len() < 64 {
                                let v = unsafe { *(a as *const u32) };
                                if (vlo..vhi).contains(&v) {
                                    hits.push(format!("{a:#x} ({name}) = {v:#x} ({})", f32::from_bits(v)));
                                }
                                a += 4;
                            }
                        }
                        info!("find32 {arg}: {} hit(s) in {} mappings ({} ms): {}", hits.len(), spans.len(), t0.elapsed().as_millis(), hits.join(", "));
                        Ok(())
                    }
                    None => Err("usage: find32 <hex u32> | <hex lo>-<hex hi>".into()),
                }
            }
            "stick" => {
                // Diagnosis: `stick l|r <x> <y> [ms]`: holds a stick at a tilt
                // (-128..127 from the centre, y down; default 1000 ms).
                let side = parts.next();
                let x = parts.next().and_then(|v| v.parse::<i32>().ok());
                let y = parts.next().and_then(|v| v.parse::<i32>().ok());
                let ms = parts.next().and_then(|v| v.parse::<u64>().ok()).unwrap_or(1000);
                match (side, x, y) {
                    (Some(sd @ ("l" | "r")), Some(x), Some(y)) => {
                        input::hold_stick(sd == "r", x, y, Duration::from_millis(ms));
                        Ok(())
                    }
                    _ => Err("usage: stick l|r <x> <y> [ms]".into()),
                }
            }
            "poke" => {
                // Experiment: `poke <hex address> <hex u64> [<hex mask>]`: writes
                // (old & !mask) | (value & mask) to the 8-byte aligned guest
                // address (mask: all bits by default); logs old and new values.
                let hex = |s: Option<&str>| s.and_then(|v| u64::from_str_radix(v.trim_start_matches("0x"), 16).ok());
                match (hex(parts.next()), hex(parts.next())) {
                    (Some(a), Some(v)) if a % 8 == 0 && gpu::cp::guest_readable(a, 8) => {
                        let mask = hex(parts.next()).unwrap_or(u64::MAX);
                        let p = a as *mut u64;
                        let old = unsafe { p.read_volatile() };
                        let new = (old & !mask) | (v & mask);
                        unsafe { p.write_volatile(new) };
                        info!("control: poke {a:#x}: {old:#018x} -> {new:#018x}");
                        Ok(())
                    }
                    _ => Err("usage: poke <hex 8-byte aligned address> <hex u64> [<hex mask>]".into()),
                }
            }
            "findstr" => {
                // Debug: `findstr <text>`: readable guest memory holding the
                // ASCII text (first 64 hits; the rest of the line is the text).
                let text = line.trim_start().strip_prefix("findstr").unwrap_or("").trim();
                if text.is_empty() {
                    Err("usage: findstr <text>".into())
                } else {
                    let needle = text.as_bytes();
                    let spans: Vec<(u64, u64, String)> = mem::MEM.lock().vmas.values().filter(|m| m.kind != mem::VmaKind::Reserved && m.prot & 1 != 0).map(|m| (m.start, m.end(), m.name.clone())).collect();
                    let mut hits = Vec::new();
                    let t0 = Instant::now();
                    for (lo, hi, name) in &spans {
                        if hits.len() >= 64 || hi - lo < needle.len() as u64 {
                            continue;
                        }
                        let hay = unsafe { std::slice::from_raw_parts(*lo as *const u8, (hi - lo) as usize) };
                        let mut i = 0;
                        while let Some(k) = hay[i..].windows(needle.len()).position(|w| w == needle) {
                            hits.push(format!("{:#x} ({name})", lo + (i + k) as u64));
                            i += k + 1;
                            if hits.len() >= 64 {
                                break;
                            }
                        }
                    }
                    info!("findstr {text}: {} hit(s) in {} mappings ({} ms): {}", hits.len(), spans.len(), t0.elapsed().as_millis(), hits.join(", "));
                    Ok(())
                }
            }
            "key" => {
                // Diagnosis: `key <input> [ms]` holds a key or mouse button of
                // the keyboard and mouse layout (default 200 ms).
                let name = parts.next().unwrap_or("");
                let ms = parts.next().and_then(|v| v.parse::<u64>().ok()).unwrap_or(200);
                input::inject_hold(name, Duration::from_millis(ms))
            }
            "mousemove" => {
                // Diagnosis: `mousemove <dx/s> <dy/s> <ms>` steady mouse motion (counts a second)
                // fed at every vblank, as a polled mouse delivers it.
                let dx = parts.next().and_then(|v| v.parse::<f64>().ok());
                let dy = parts.next().and_then(|v| v.parse::<f64>().ok());
                let ms = parts.next().and_then(|v| v.parse::<u64>().ok());
                match (dx, dy, ms) {
                    (Some(dx), Some(dy), Some(ms)) => {
                        input::inject_motion_for(dx, dy, std::time::Duration::from_millis(ms));
                        Ok(())
                    }
                    _ => Err("usage: mousemove <dx/s> <dy/s> <ms>".into()),
                }
            }
            "mouse" => {
                // Diagnosis: `mouse <dx> <dy>` mouse motion in counts.
                let dx = parts.next().and_then(|v| v.parse::<i64>().ok());
                let dy = parts.next().and_then(|v| v.parse::<i64>().ok());
                match (dx, dy) {
                    (Some(dx), Some(dy)) => {
                        input::inject_motion(dx, dy, 0);
                        Ok(())
                    }
                    _ => Err("usage: mouse <dx> <dy>".into()),
                }
            }
            "wheel" => {
                // Diagnosis: `wheel <notches>` (up positive).
                match parts.next().and_then(|v| v.parse::<i32>().ok()) {
                    Some(n) => {
                        input::inject_motion(0, 0, n);
                        Ok(())
                    }
                    None => Err("usage: wheel <notches>".into()),
                }
            }
            "capture" if matches!(parts.clone().next(), Some("on" | "off")) => {
                // Diagnosis: `capture on|off`: mouse input counts as if the
                // window had captured the cursor (headless runs). Any other
                // `capture` is a frame capture (below).
                let on = parts.next() == Some("on");
                gpu::vk::window::CAPTURE_TEST.store(on, std::sync::atomic::Ordering::Relaxed);
                Ok(())
            }
            "watch" => {
                // Debug: `watch <slot 0-3> <hex address> [w|rw] [1|2|4|8] [all|guest]`
                // or `watch <slot> off`: hardware watchpoint on every
                // registered thread (writes: all threads by default; rw:
                // guest threads by default). Hits are logged (`watch N ...`).
                // (The address as `debug_addr` reads it: a pointer chain from the executable too.)
                let slot = parts.next().and_then(|v| v.parse::<usize>().ok()).filter(|&s| s < 4);
                let arg = parts.next().unwrap_or("");
                let addr = if arg == "off" { Some(0) } else { debug_addr(arg).ok() };
                let (mut rw, mut len, mut guest) = (1u32, 8u32, None::<bool>);
                for o in parts.by_ref() {
                    match o {
                        "w" => rw = 1,
                        "rw" => rw = 3,
                        "1" | "2" | "4" | "8" => len = o.parse().unwrap(),
                        "all" => guest = Some(false),
                        "guest" => guest = Some(true),
                        _ => {}
                    }
                }
                match (slot, addr) {
                    (Some(s), Some(a)) => {
                        let guest_only = guest.unwrap_or(rw == 3);
                        let n = exception::set_watch(s, a, rw, len, guest_only);
                        info!("control: watch {s} {a:#x} {} len {len}{}: {n} thread(s) armed", if rw == 3 { "rw" } else { "w" }, if guest_only { " (guest threads)" } else { "" });
                        Ok(())
                    }
                    _ => Err("usage: watch <slot 0-3> <hex address>|off [w|rw] [1|2|4|8] [all|guest]".into()),
                }
            }
            "watchwrites" => {
                // Diagnosis: `watchwrites <hex lo>:<hex hi>` / `watchwrites off`: the GPU-side
                // writers of that guest range logged from now on (`DS_WATCH_WRITES`'s notes).
                let arg = parts.next().unwrap_or("");
                let p = |x: &str| u64::from_str_radix(x.trim().trim_start_matches("0x"), 16).ok();
                let r = if arg == "off" { Some(None) } else { arg.split_once(':').and_then(|(a, b)| Some(Some((p(a)?, p(b)?)))) };
                match r {
                    Some(r) => {
                        gpu::exec::set_watch_range(r);
                        info!("control: watchwrites {}", r.map_or("off".to_string(), |(a, b)| format!("{a:#x}..{b:#x}")));
                        Ok(())
                    }
                    None => Err("usage: watchwrites <hex lo>:<hex hi>|off".into()),
                }
            }
            "emucheck" => {
                // Diagnosis: `emucheck <hex shader>[,...][:<groups>]` / `emucheck off`.
                let rest = line.trim_start().strip_prefix("emucheck").unwrap_or("").trim();
                gpu::emucheck::arm(rest)
            }
            "probe" => {
                // Diagnosis: `probe <shader>:<pc>:<v<k>|s<k>>[u|i][:<frames>]` / `probe off`.
                let rest = line.trim_start().strip_prefix("probe").unwrap_or("").trim();
                gpu::shaders::set_probe(rest)
            }
            "csdump" => {
                // Diagnosis: `csdump <hex compute shader address>`: its next dispatch writes
                // its buffer resources' guest memory to the run directory.
                match parts.next().and_then(|a| u64::from_str_radix(a.trim_start_matches("0x"), 16).ok()) {
                    Some(a) => {
                        gpu::exec::request_cs_dump(a);
                        Ok(())
                    }
                    None => Err("usage: csdump <hex address>".into()),
                }
            }
            "skipcs" => {
                // Diagnosis: `skipcs <hex compute shader address>` toggles
                // skipping its dispatches.
                match parts.next().and_then(|a| u64::from_str_radix(a.trim_start_matches("0x"), 16).ok()) {
                    Some(a) => {
                        let on = gpu::exec::toggle_skip_cs(a);
                        info!("control: dispatches of compute shader {a:#x} {}", if on { "skipped" } else { "run again" });
                        Ok(())
                    }
                    None => Err("usage: skipcs <hex address>".into()),
                }
            }
            "skipvs" => {
                // Diagnosis: `skipvs <hex vertex shader address>` toggles
                // skipping the draws that use it.
                match parts.next().and_then(|a| u64::from_str_radix(a.trim_start_matches("0x"), 16).ok()) {
                    Some(a) => {
                        let on = gpu::exec::toggle_skip_vs(a);
                        info!("control: draws with vertex shader {a:#x} {}", if on { "skipped" } else { "drawn again" });
                        Ok(())
                    }
                    None => Err("usage: skipvs <hex address>".into()),
                }
            }
            "textreset" => {
                exception::text_reset();
                Ok(())
            }
            "engine" => {
                // `engine reset` restarts the engine census and front end 4's
                // counters; `engine report` logs them (`engine`).
                match parts.next() {
                    Some("reset") => {
                        engine::reset();
                        Ok(())
                    }
                    Some("report") => {
                        info!("engine report at flip {flip}:\n{}", engine::report());
                        Ok(())
                    }
                    _ => Err("usage: engine reset|report".into()),
                }
            }
            "profile" => {
                // `profile reset` restarts the DS_PROFILE_GPU sampling and
                // dispatch timings; `profile report` logs them.
                match parts.next() {
                    Some("reset") => {
                        gpu::reset_profile();
                        Ok(())
                    }
                    Some("report") => {
                        info!("profile at flip {flip}:\n{}", gpu::profile_report());
                        Ok(())
                    }
                    _ => Err("usage: profile reset|report".into()),
                }
            }
            "replay" => {
                // `replay <iterations> [<switch> <a> <b>] [timeline=<k>]`: one
                // frame of the title replayed through the GPU front end with
                // the title held, optionally alternating a `tune` switch, and
                // the timeline of the last k iterations (`gpu::replay`).
                let n = parts.next().and_then(|v| v.parse::<u32>().ok()).filter(|n| (1..=200).contains(n));
                let mut rest: Vec<&str> = parts.by_ref().collect();
                let mut timeline = 0u32;
                if let Some(k) = rest.last().and_then(|t| t.strip_prefix("timeline=")).and_then(|k| k.parse().ok()) {
                    timeline = k;
                    rest.pop();
                }
                let ab = match rest.as_slice() {
                    [] => Ok(None),
                    [name, a, b] => match (a.parse::<u64>(), b.parse::<u64>()) {
                        (Ok(a), Ok(b)) => Ok(Some((name.to_string(), a, b))),
                        _ => Err(()),
                    },
                    _ => Err(()),
                };
                match (n, ab) {
                    (Some(n), Ok(ab)) => gpu::replay::start(n, ab, timeline),
                    _ => Err("usage: replay <iterations 1-200> [<switch> <a> <b>] [timeline=<k>]".into()),
                }
            }
            "timeline" => {
                // `timeline <flips>`: spans of command processing and GPU
                // batches over the next flips (`gpu::timeline`).
                match parts.next().and_then(|n| n.parse::<u32>().ok()) {
                    Some(n) if n > 0 => gpu::timeline::start(n),
                    _ => Err("usage: timeline <flips>".into()),
                }
            }
            "labels" => {
                // `labels <n>`: the last n events of the queues' label log
                // (waits, CP writes, flips) to labels-<flip>.txt in the run
                // directory (`gpu::cp::label_dump`, `tools/labelflow.py`).
                match (parts.next().and_then(|n| n.parse::<usize>().ok()), run_dir()) {
                    (Some(n), Some(dir)) if n > 0 => std::fs::write(dir.join(format!("labels-{flip}.txt")), gpu::cp::label_dump(n)).map_err(|e| e.to_string()),
                    _ => Err("usage: labels <events>".into()),
                }
            }
            "stacks" => {
                info!("guest thread stacks:\n{}", thread_summary_ex(true));
                Ok(())
            }
            "status" => {
                // The status report without stacks (GPU, image cache with its
                // upload breakdown, audio, file traffic).
                info!("status at flip {flip}:\n{}{}", thread_summary(), hle::libs::agc_cmd::emit_status());
                Ok(())
            }
            "waits" => {
                // `DS_WAIT_CENSUS=1`: the title's waits since the last report.
                info!("waits at flip {flip}:\n{}", hle::waitcensus::report());
                Ok(())
            }
            "iostat" => {
                info!("file traffic at flip {flip}:\n{}", hle::iostat::status());
                Ok(())
            }
            "wbs" => {
                // `wbs on|off` (`DS_WB_SAME=2`): write-backs leave out the
                // pages they would store unchanged.
                match gpu::fe2::wbs_control(parts.next().unwrap_or("")) {
                    Ok(m) => {
                        info!("control: wbs: {m} at flip {flip}");
                        Ok(())
                    }
                    Err(e) => Err(e),
                }
            }
            "ptaudit" => {
                // `ptaudit` (diagnosis): the guest page table read back,
                // each entry checked against the live import windows.
                info!("control: ptaudit at flip {flip}: {}", gpu::exec::pt_audit());
                Ok(())
            }
            "decalwait" => {
                // `decalwait off|on|sub` (A/B, `cullread::DECAL_WAIT`): the title's walks of the
                // static sections' decal masks wait for the GPU's writes (`sub`: first for the
                // graphics CP to take up the title's earlier submissions too).
                let v = match parts.next().unwrap_or("") {
                    "off" | "0" => Some(0),
                    "on" | "1" => Some(1),
                    "sub" | "2" => Some(2),
                    _ => None,
                };
                match v {
                    Some(v) => {
                        cullread::DECAL_WAIT.store(v, std::sync::atomic::Ordering::Relaxed);
                        info!("control: decalwait {v} at flip {flip}");
                        Ok(())
                    }
                    None => Err("decalwait: off, on or sub".into()),
                }
            }
            "maskwait" => {
                // `maskwait off|on` (A/B, `cullread`): the title's walks of
                // its GPU frustum culling masks wait for the GPU's writes.
                match parts.next().unwrap_or("") {
                    v @ ("on" | "1" | "off" | "0" | "sub" | "subhard" | "suball" | "subload") => {
                        let on = v != "off" && v != "0";
                        cullread::MASK_WAIT_ON.store(on, std::sync::atomic::Ordering::Relaxed);
                        cullread::MASK_WAIT_SUB.store(v.starts_with("sub"), std::sync::atomic::Ordering::Relaxed);
                        cullread::MASK_KNOWN.store(v != "subhard", std::sync::atomic::Ordering::Relaxed);
                        cullread::MASK_ALL.store(v == "suball", std::sync::atomic::Ordering::Relaxed);
                        cullread::MASK_LOAD_ONLY.store(v == "subload", std::sync::atomic::Ordering::Relaxed);
                        info!("control: maskwait {v} at flip {flip}");
                        Ok(())
                    }
                    other => Err(format!("maskwait: on, off, sub, subhard, suball or subload, not {other:?}")),
                }
            }
            "c4" => {
                // `c4 off|on` (`DS_C4=1`): front end 4's held pages (off
                // releases them all).
                match gpu::fe2::c4_control(parts.next().unwrap_or("")) {
                    Ok(m) => {
                        info!("control: c4: {m} at flip {flip}");
                        Ok(())
                    }
                    Err(e) => Err(e),
                }
            }
            "armbehind" => {
                // `armbehind off|on` (front end 2's arming behind the
                // graphics CP and queue 0x50): off disarms what it armed.
                match gpu::fe2::behind_control(parts.next().unwrap_or("")) {
                    Ok(m) => {
                        info!("control: armbehind: {m} at flip {flip}");
                        Ok(())
                    }
                    Err(e) => Err(e),
                }
            }
            "c4arm" => {
                // `c4arm off|on` (front end 4's own arming): off releases
                // the held pages its rounds armed and holds none of them.
                match gpu::fe2::c4_arm_control(parts.next().unwrap_or("")) {
                    Ok(m) => {
                        info!("control: c4arm: {m} at flip {flip}");
                        Ok(())
                    }
                    Err(e) => Err(e),
                }
            }
            "c5" => {
                // `c5 off|on` (front end 4's C5): stores into held pages
                // marked again, or not.
                match gpu::fe2::c5_control(parts.next().unwrap_or("")) {
                    Ok(m) => {
                        info!("control: c5: {m} at flip {flip}");
                        Ok(())
                    }
                    Err(e) => Err(e),
                }
            }
            "wbb" => {
                // `wbb lazy|eager` (`DS_WB_BUCKETS=1`, diagnosis): the
                // write-back leaves out the last census's untouched buckets.
                match gpu::fe2::wbb_control(parts.next().unwrap_or("")) {
                    Ok(m) => {
                        info!("control: wbb: {m} at flip {flip}");
                        Ok(())
                    }
                    Err(e) => Err(e),
                }
            }
            "audio" => {
                // `audio capture <seconds>`: the mix as played, to
                // `<run dir>/audio-<n>.wav`.
                match (parts.next(), parts.next().and_then(|s| s.parse::<f64>().ok())) {
                    (Some("capture"), Some(s)) if s > 0.0 && s <= 600.0 => audio::start_capture(s),
                    // `audio split <seconds>`: each AudioOut2 port group's mix to its own file.
                    (Some("split"), Some(s)) if s > 0.0 && s <= 120.0 => hle::libs::audio2::start_split(s),
                    // `audio ports <seconds>`: each AudioOut port's buffers as the title wrote them.
                    (Some("ports"), Some(s)) if s > 0.0 && s <= 300.0 => hle::libs::devices::start_raw_capture(s),
                    _ => Err("usage: audio capture <seconds 0-600> | audio split <seconds 0-120> | audio ports <seconds 0-300>".into()),
                }
            }
            "frametimes" => {
                // Intervals between presented flips since the last
                // `frametimes` (`gpu::pacing`).
                info!("frame times at flip {flip}: {}", gpu::pacing::report_and_reset());
                Ok(())
            }
            "capture" => {
                // `capture [<from>-<to>] [dispatches] [images=<base>[,..]]
                // [buffers=<shader>[,..]] [pixel=<x>,<y>]`:
                // the next frame, optionally with per-draw reports for that
                // range of its events, dumps of the images its dispatches
                // write and of the buffers those compute shaders bind.
                for arg in parts.by_ref() {
                    if arg == "dispatches" {
                        gpu::capture::set_dispatch_dumps(true);
                    } else if let Some(p) = arg.strip_prefix("pixel=") {
                        // ... reporting the cb0 value of that pixel per draw.
                        let xy: Vec<u32> = p.split(',').filter_map(|v| v.parse().ok()).collect();
                        gpu::capture::set_watch_pixel((xy.len() == 2).then(|| (xy[0], xy[1])));
                    } else if let Some(list) = arg.strip_prefix("ps=") {
                        // ... reports of the targets of draws with these pixel shaders.
                        gpu::capture::set_draw_ps(list.split(',').filter_map(|a| u64::from_str_radix(a.trim_start_matches("0x"), 16).ok()).collect());
                    } else if let Some(list) = arg.strip_prefix("vs=") {
                        // ... and of draws with these vertex programs (with
                        // DS_VS_DUMP, their vertices' registers or positions).
                        gpu::capture::set_draw_vs(list.split(',').filter_map(|a| u64::from_str_radix(a.trim_start_matches("0x"), 16).ok()).collect());
                    } else if let Some(list) = arg.strip_prefix("images=") {
                        // ... only these images (PNG and raw texels) after
                        // each dispatch that writes them.
                        gpu::capture::set_dispatch_dumps(true);
                        gpu::capture::set_image_dumps(list.split(',').filter_map(|a| u64::from_str_radix(a.trim().trim_start_matches("0x"), 16).ok()).collect());
                    } else if let Some(cs) = arg.strip_prefix("buffers=") {
                        // ... and the buffers that compute shader binds.
                        gpu::capture::set_dispatch_dumps(true);
                        gpu::capture::set_buffer_dumps(cs.split(',').filter_map(|a| u64::from_str_radix(a.trim().trim_start_matches("0x"), 16).ok()).collect());
                    } else if let Some((a, b)) = arg.split_once('-') {
                        match (a.parse(), b.parse()) {
                            (Ok(a), Ok(b)) => gpu::capture::set_draw_range((a, b)),
                            _ => warn!("control: bad draw range '{a}-{b}'"),
                        }
                    }
                }
                gpu::capture::arm_next();
                Ok(())
            }
            "peek" => {
                // Debug: `peek <address> [<hex length up to 0x100>]`: logs the bytes at the
                // address (`debug_addr`: hex, `eboot+<hex>`, `,<hex>` pointer steps).
                let spec = parts.next().unwrap_or("");
                let n = parts.next().and_then(|v| u64::from_str_radix(v.trim_start_matches("0x"), 16).ok()).unwrap_or(8).clamp(1, 0x100);
                match debug_addr(spec) {
                    Ok(a) if gpu::cp::guest_readable(a, n) || mem::os_readable(a, n) => {
                        let bytes = unsafe { std::slice::from_raw_parts(a as *const u8, n as usize) };
                        let hex: Vec<String> = bytes.iter().map(|b| format!("{b:02x}")).collect();
                        info!("control: peek {spec} = {a:#x}: {}", hex.join(" "));
                        Ok(())
                    }
                    Ok(a) => Err(format!("{a:#x}+{n:#x} is not readable memory")),
                    Err(e) => Err(format!("usage: peek <hex address | eboot+<hex>>[,<hex step>...] [<hex length>]: {e}")),
                }
            }
            "mem" => {
                // Debug: `mem <hex address> <hex length>` writes guest memory
                // to mem-<address>.bin in the run directory.
                let hex = |s: Option<&str>| s.and_then(|v| u64::from_str_radix(v.trim_start_matches("0x"), 16).ok());
                match (hex(parts.next()), hex(parts.next())) {
                    (Some(a), Some(n)) if n > 0 && n <= 256 << 20 => {
                        // Guest memory GPU work may read, or any readable process
                        // memory (the title's heap objects live outside those ranges).
                        if !gpu::cp::guest_readable(a, n) && !mem::os_readable(a, n) {
                            Err(format!("{a:#x}+{n:#x} is not readable memory"))
                        } else if let Some(dir) = RUN_DIR.get() {
                            let bytes = unsafe { std::slice::from_raw_parts(a as *const u8, n as usize) };
                            std::fs::write(dir.join(format!("mem-{a:x}.bin")), bytes).map_err(|e| e.to_string())
                        } else {
                            Err("no run directory".into())
                        }
                    }
                    _ => Err("usage: mem <hex address> <hex length up to 0x10000000>".into()),
                }
            }
            "fillimg" => {
                // Experiment: `fillimg <hex address> <value>` fills the cached
                // images at that guest address with one value (all channels,
                // layers and mips); `fillimg cs=<hex shader> <value>` those
                // the compute shader last bound as storage images.
                let arg = parts.next().unwrap_or("");
                let v = parts.next().and_then(|v| v.parse::<f32>().ok());
                let hex = |a: &str| u64::from_str_radix(a.trim_start_matches("0x"), 16).ok();
                match (arg.strip_prefix("cs="), v) {
                    (Some(cs), Some(v)) => match hex(cs) {
                        Some(cs) => {
                            let (bases, n) = gpu::images::with(|im| {
                                let b = im.written_by(cs);
                                let n: usize = b.iter().map(|&a| im.fill_at(a, v)).sum();
                                (b, n)
                            });
                            info!("control: fillimg cs={cs:#x} {v}: {n} image(s) filled at {bases:x?}");
                            Ok(())
                        }
                        None => Err("usage: fillimg cs=<hex shader> <value>".into()),
                    },
                    (None, Some(v)) => match hex(arg) {
                        Some(a) => {
                            let n = gpu::images::with(|im| im.fill_at(a, v));
                            info!("control: fillimg {a:#x} {v}: {n} image(s) filled");
                            Ok(())
                        }
                        None => Err("usage: fillimg <hex address>|cs=<hex shader> <value>".into()),
                    },
                    _ => Err("usage: fillimg <hex address>|cs=<hex shader> <value>".into()),
                }
            }
            "upscaler" => {
                // `upscaler off|dlss|fsr|xess`: the Anti-Aliasing's upscaler, live (tests).
                let v = parts.next().unwrap_or("off");
                gpu::upscale::set_choice(v);
                Ok(())
            }
            "upjitter" => {
                // `upjitter <x> <y>`: the upscalers' jitter scale from now on (tests).
                let x = parts.next().and_then(|v| v.parse::<f32>().ok()).unwrap_or(0.5);
                let y = parts.next().and_then(|v| v.parse::<f32>().ok()).unwrap_or(-0.5);
                gpu::upscale::set_jitter_scale(x, y);
                Ok(())
            }
            "wsinfo" | "wstrim" => {
                // Diagnosis: `wsinfo` logs the process's working set, private bytes and page
                // faults; `wstrim` empties the working set first (what play touches again in a
                // while is its hot set: the RAM it needs to run without paging).
                use windows_sys::Win32::System::ProcessStatus::{EmptyWorkingSet, GetProcessMemoryInfo, PROCESS_MEMORY_COUNTERS, PROCESS_MEMORY_COUNTERS_EX};
                use windows_sys::Win32::System::Threading::GetCurrentProcess;
                let info = || {
                    let mut c: PROCESS_MEMORY_COUNTERS_EX = unsafe { std::mem::zeroed() };
                    c.cb = std::mem::size_of::<PROCESS_MEMORY_COUNTERS_EX>() as u32;
                    unsafe { GetProcessMemoryInfo(GetCurrentProcess(), &mut c as *mut PROCESS_MEMORY_COUNTERS_EX as *mut PROCESS_MEMORY_COUNTERS, c.cb) };
                    format!("working set {} MiB (peak {}), private {} MiB, page faults {}", c.WorkingSetSize >> 20, c.PeakWorkingSetSize >> 20, c.PrivateUsage >> 20, c.PageFaultCount)
                };
                if cmd == "wstrim" {
                    info!("wstrim: before: {}", info());
                    let t = Instant::now();
                    unsafe { EmptyWorkingSet(GetCurrentProcess()) };
                    info!("wstrim: emptied in {:.0} ms: {}", t.elapsed().as_secs_f64() * 1000.0, info());
                } else {
                    info!("wsinfo: {}", info());
                }
                Ok(())
            }
            "pcstate" => {
                // Tests: every PC setting's stored value beside what is in effect, and the
                // title's console variables (settings tests: an effect, and the same after a
                // restart).
                pcsettings::state_report();
                Ok(())
            }
            "devicelost" => {
                // Tests: the GPU device taken as lost (the game ends as on a real loss).
                warn!("devicelost: the GPU device taken as lost (test)");
                gpu::exec::lose_device();
                Ok(())
            }
            "taadump" => {
                // `taadump <name>`: the TAA's input colour at its next dispatch (diagnosis).
                let name: String = parts.next().unwrap_or("taa").chars().filter(|c| c.is_ascii_alphanumeric() || *c == '_' || *c == '-' || *c == ':').collect();
                *gpu::exec::TAA_DUMP.lock() = Some(name);
                Ok(())
            }
            "imgpng" => {
                // Diagnosis: `imgpng <hex address> <name>`: the GPU image at that guest address as
                // <run dir>/<name>.png.
                let a = parts.next().and_then(|a| u64::from_str_radix(a.trim_start_matches("0x"), 16).ok());
                let name: String = parts.next().unwrap_or("img").chars().filter(|c| c.is_ascii_alphanumeric() || *c == '_' || *c == '-').collect();
                match (a, RUN_DIR.get()) {
                    (Some(a), Some(dir)) => {
                        let r = gpu::images::with(|im| im.dump_one(a, dir, &name));
                        info!("control: imgpng {a:#x} {name}: {}", r.as_deref().unwrap_or_else(|e| e));
                        r.map(|_| ())
                    }
                    _ => Err("usage: imgpng <hex address> <name>".into()),
                }
            }
            "imgprobe" => {
                // Diagnosis: `imgprobe <flips> <u> <v>`: at each of the next flips, a 4x4 block's
                // means at (u, v) of every GPU-written image used in the last two frames.
                // (`taa` last: at the TAA's dispatch, the frame's work before it completed.)
                let n = parts.next().and_then(|v| v.parse::<u32>().ok()).unwrap_or(60).clamp(1, 600);
                let u = parts.next().and_then(|v| v.parse::<f32>().ok()).unwrap_or(0.5);
                let v = parts.next().and_then(|v| v.parse::<f32>().ok()).unwrap_or(0.5);
                if parts.next() == Some("taa") {
                    *gpu::exec::TAA_IMG_PROBE.lock() = Some((n, u, v));
                } else {
                    *hle::libs::videoout::PROBE.lock() = Some((n, u, v));
                }
                Ok(())
            }
            "skiprw" => {
                // Diagnosis: `skiprw <hex image base>|off`: draws whose pixel stage writes the image
                // at that base left out from now on (the G-buffer's in-place decals).
                let arg = parts.next().unwrap_or("");
                let v = if arg == "off" { Some(0) } else { u64::from_str_radix(arg.trim_start_matches("0x"), 16).ok() };
                let lo = parts.next().and_then(|x| x.parse::<u64>().ok()).unwrap_or(0);
                let hi = parts.next().and_then(|x| x.parse::<u64>().ok()).unwrap_or(u64::MAX);
                match v {
                    Some(v) => {
                        *gpu::exec::draw::SKIP_RW_RANGE.lock() = (lo, hi);
                        let was = gpu::exec::draw::SKIP_RW.swap(v, std::sync::atomic::Ordering::Relaxed);
                        info!("control: skiprw {v:#x} (was {was:#x}; {} draws left out so far)", gpu::exec::draw::SKIP_RW_N.load(std::sync::atomic::Ordering::Relaxed));
                        Ok(())
                    }
                    None => Err("usage: skiprw <hex image base>|off".into()),
                }
            }
            "atprobe" => {
                // Diagnosis: `atprobe <n> <hex cs> <hex image base> <u> <v>`: before each of the
                // next n dispatches of that program (its earlier work completed), a 4x4 block at
                // (u, v) of the GPU-written image at that base.
                let n = parts.next().and_then(|v| v.parse::<u32>().ok()).unwrap_or(60).clamp(1, 20000);
                let h = |x: &str| u64::from_str_radix(x.trim().trim_start_matches("0x"), 16).ok();
                let cs: Vec<u64> = parts.next().unwrap_or("").split(',').filter_map(h).collect();
                let base = parts.next().and_then(|x| if x == "0" { Some(0) } else { h(x) });
                let u = parts.next().and_then(|v| v.parse::<f32>().ok()).unwrap_or(0.5);
                let v = parts.next().and_then(|v| v.parse::<f32>().ok()).unwrap_or(0.5);
                match base {
                    Some(base) if !cs.is_empty() => {
                        *gpu::exec::AT_PROBE.lock() = Some((n, cs, base, u, v));
                        gpu::exec::AT_PROBE_ON.store(true, std::sync::atomic::Ordering::Relaxed);
                        gpu::exec::start_mem_poll();
                        Ok(())
                    }
                    _ => Err("usage: atprobe <n> <hex cs>[,<hex cs>...] <hex image base> <u> <v>".into()),
                }
            }
            "atlasprobe" => {
                // Diagnosis: `atlasprobe <flips> <hex base>`: at each of the next flips' TAA
                // dispatch, the frame's work before it completed, each layer of the GPU-written
                // image at that base sampled every 16th row: its covered texels and a hash.
                let n = parts.next().and_then(|v| v.parse::<u32>().ok()).unwrap_or(60).clamp(1, 1200);
                match parts.next().and_then(|a| u64::from_str_radix(a.trim_start_matches("0x"), 16).ok()) {
                    Some(b) => {
                        *gpu::exec::TAA_ATLAS_PROBE.lock() = Some((n, b));
                        Ok(())
                    }
                    None => Err("usage: atlasprobe <flips> <hex base>".into()),
                }
            }
            "metamap" => {
                // Diagnosis: `metamap` logs every fast-clear metadata address and its surface.
                let lines = gpu::images::with(|im| im.meta_report());
                info!("control: metamap: {} metadata address(es)\n    {}", lines.len(), lines.join("\n    "));
                Ok(())
            }
            "metadrop" => {
                // Diagnosis: `metadrop <hex address>[,...]` forgets the surfaces of those metadata
                // addresses (their writes then clear nothing).
                let list: Vec<u64> = parts.next().unwrap_or("").split(',').filter_map(|a| u64::from_str_radix(a.trim().trim_start_matches("0x"), 16).ok()).collect();
                if list.is_empty() {
                    Err("usage: metadrop <hex address>[,...]".into())
                } else {
                    for m in list {
                        let r = gpu::images::with(|im| im.meta_drop(m));
                        info!("control: metadrop {}", r.unwrap_or_else(|| format!("{m:#x}: no surface")));
                    }
                    Ok(())
                }
            }
            "imgstat" => {
                // Diagnosis: `imgstat <hex address>` logs value statistics of
                // the cached images at that guest address.
                match parts.next().and_then(|a| u64::from_str_radix(a.trim_start_matches("0x"), 16).ok()) {
                    Some(a) => {
                        let lines = gpu::images::with(|im| im.stats_at(a));
                        info!("control: imgstat {a:#x}: {} image(s)\n    {}", lines.len(), lines.join("\n    "));
                        Ok(())
                    }
                    None => Err("usage: imgstat <hex address>".into()),
                }
            }
            "tune" => {
                // Measurements: `tune <switch> <value>` (`gpu::tune`)
                // changes a GPU switch for the rest of the run.
                match (parts.next(), parts.next().and_then(|v| v.parse::<u64>().ok())) {
                    (Some(name), Some(v)) if gpu::tune::set(name, v) => {
                        info!("control: tune {name} = {v} at flip {}", gpu::FRAMES.load(std::sync::atomic::Ordering::Relaxed));
                        if name == "guest_cpus" {
                            thread::apply_guest_cpus(gpu::tune::guest_cpus());
                        }
                        if name == "power_optout" {
                            thread::apply_power_optout(v != 0);
                        }
                        Ok(())
                    }
                    _ => Err("usage: tune <switch> <value> (see gpu::tune)".into()),
                }
            }
            "tiny" => {
                // Diagnosis: `tiny <hex ps>` toggles limiting that pixel
                // shader's draws to one pixel (vertex cost vs pixel cost).
                match parts.next().and_then(|a| u64::from_str_radix(a.trim_start_matches("0x"), 16).ok()) {
                    Some(a) => {
                        let mut g = gpu::exec::draw::TINY_SCISSOR.lock();
                        let on = if let Some(i) = g.iter().position(|&x| x == a) { g.remove(i); false } else { g.push(a); true };
                        gpu::exec::draw::TINY_ANY.store(!g.is_empty(), std::sync::atomic::Ordering::Relaxed);
                        info!("control: draws with ps {a:#x} {}", if on { "limited to one pixel" } else { "full again" });
                        Ok(())
                    }
                    None => Err("usage: tiny <hex ps address>".into()),
                }
            }
            "texcheck" => {
                // Diagnosis: `texcheck` lists the textures whose guest bytes
                // changed since their last upload with no write marked.
                for l in gpu::images::Images::check_sources() {
                    info!("{l}");
                }
                Ok(())
            }
            "reupload" => {
                // Diagnosis: `reupload [<hex address>]` uploads the textures
                // sourced from guest memory (at that address) again at their
                // next use.
                let base = parts.next().and_then(|a| u64::from_str_radix(a.trim_start_matches("0x"), 16).ok());
                let n = gpu::images::with(|im| im.force_reupload(base));
                info!("control: {n} texture(s) marked for upload");
                Ok(())
            }
            "clearimg" => {
                // Diagnosis: `clearimg <hex address>` zero-fills the cached
                // images at that guest address.
                match parts.next().and_then(|a| u64::from_str_radix(a.trim_start_matches("0x"), 16).ok()) {
                    Some(a) => {
                        let n = gpu::images::with(|im| im.clear_at(a));
                        info!("control: cleared {n} image(s) at {a:#x}");
                        Ok(())
                    }
                    None => Err("usage: clearimg <hex address>".into()),
                }
            }
            "vmas" => {
                // Debug: guest address space layout.
                let m = crate::mem::MEM.lock();
                let mut s = format!("{} guest mappings:", m.vmas.len());
                for v in m.vmas.values() {
                    s.push_str(&format!("\n  {:#014x}-{:#014x} {:>10} KiB {:?} prot {:#x} phys {:#x} '{}'", v.start, v.end(), v.size >> 10, v.kind, v.prot, v.phys, v.name));
                }
                drop(m);
                info!("{s}");
                Ok(())
            }
            "quit" => {
                info!("control: quit at flip {flip}");
                shutdown(0);
            }
            _ => Err(format!("unknown command '{cmd}'")),
        };
        match result {
            Ok(()) => info!("control: '{line}' at flip {flip}"),
            Err(e) => warn!("control: '{line}': {e}"),
        }
    }
}

/// Control lines asked for by the title's developer automation (a guest
/// print of `dsport-control: <line>`, `hle::kernel::fs`), run before the
/// control file's next lines (tools/devscript.sh scripts pressing buttons at
/// their own points, e.g. at a boss's fog door).
pub static INJECTED: parking_lot::Mutex<Vec<String>> = parking_lot::Mutex::new(Vec::new());

/// `fscan` state: page hashes, candidate pages, their copies, candidate
/// float addresses.
#[derive(Default)]
struct FScan {
    pages: Vec<(u64, u64)>,
    cand: Option<Vec<u64>>,
    copy: std::collections::BTreeMap<u64, Vec<u8>>,
    floats: Option<Vec<u64>>,
}

static FSCAN: parking_lot::Mutex<Option<FScan>> = parking_lot::Mutex::new(None);
/// Candidate pages copied for `fscan floats` (64 MiB) at most.
const FSCAN_COPY_PAGES: usize = 16384;

/// An address for the debug commands (`peek`, `watch`): hex, or `eboot+<hex>` (the main
/// executable's load address plus an offset), then `,<hex>` steps that each read the 8 bytes at
/// the address so far and add the offset: `eboot+330ccb0,720,e8` is a field of an object a field
/// of the object a global points to points to.
fn debug_addr(spec: &str) -> Result<u64, String> {
    let hex = |s: &str| u64::from_str_radix(s.trim().trim_start_matches("0x"), 16).map_err(|_| format!("{s}: not hex"));
    let mut steps = spec.split(',');
    let first = steps.next().unwrap_or("");
    let mut a = match first.strip_prefix("eboot+") {
        Some(o) => pcsettings::convar::base().checked_add(hex(o)?).filter(|_| pcsettings::convar::base() != 0).ok_or("the executable's address is not known yet")?,
        None => hex(first)?,
    };
    for s in steps {
        if !(gpu::cp::guest_readable(a, 8) || mem::os_readable(a, 8)) {
            return Err(format!("{a:#x} is not readable"));
        }
        a = unsafe { (a as *const u64).read_unaligned() }.wrapping_add(hex(s)?);
    }
    Ok(a)
}

fn page_hash(a: u64) -> u64 {
    let p = a as *const u64;
    let mut h = 0xcbf2_9ce4_8422_2325u64;
    for i in 0..512 {
        h = (h ^ unsafe { p.add(i).read_volatile() }).wrapping_mul(0x100_0000_01b3);
    }
    h
}

/// Debug (finding the gameplay camera's frame): 4x4 f32 matrices in the
/// title's flexible memory whose upper 3x3 is a rotation about the vertical
/// axis only (rows [c 0 s] [0 1 0] [-s 0 c], 16 bytes apart, c*c + s*s = 1,
/// not the identity); logs each with c, s and its fourth row (translation),
/// at most 64.
fn fmat() -> Result<(), String> {
    let t0 = Instant::now();
    let spans: Vec<(u64, u64)> = mem::MEM.lock().vmas.values().filter(|m| m.kind == mem::VmaKind::Flexible && m.prot & 3 == 3).map(|m| (m.start, m.end())).collect();
    let rd = |a: u64| f32::from_bits(unsafe { (a as *const u32).read_volatile() });
    let near = |a: f32, b: f32| (a - b).abs() < 1e-4;
    let mut found: Vec<String> = Vec::new();
    let mut n = 0usize;
    for (lo, hi) in spans {
        let mut a = lo.next_multiple_of(16);
        while a + 64 <= hi {
            let (c, s) = (rd(a), rd(a + 8));
            if c.is_finite() && s.is_finite() && c.abs() <= 1.0 && s.abs() > 1e-3 && near(c * c + s * s, 1.0) && near(rd(a + 0x28), c) && near(rd(a + 0x20), -s) && near(rd(a + 0x14), 1.0) && near(rd(a + 4), 0.0) && near(rd(a + 0x10), 0.0) && near(rd(a + 0x18), 0.0) && near(rd(a + 0x24), 0.0) {
                n += 1;
                if found.len() < 64 {
                    found.push(format!("{a:#x} c {c:.5} s {s:.5} t ({:.3}, {:.3}, {:.3})", rd(a + 0x30), rd(a + 0x34), rd(a + 0x38)));
                }
            }
            a += 16;
        }
    }
    info!("fmat: {n} yaw matrices ({} ms): {}", t0.elapsed().as_millis(), found.join("; "));
    Ok(())
}

fn fscan(cmd: &str, arg: &str) -> Result<(), String> {
    let t0 = Instant::now();
    let mut g = FSCAN.lock();
    if cmd == "start" {
        let spans: Vec<(u64, u64)> = mem::MEM.lock().vmas.values().filter(|m| m.kind == mem::VmaKind::Flexible && m.prot & 3 == 3).map(|m| (m.start, m.end())).collect();
        let mut st = FScan::default();
        for (lo, hi) in spans {
            let mut a = lo;
            while a + 4096 <= hi {
                st.pages.push((a, page_hash(a)));
                a += 4096;
            }
        }
        info!("fscan start: {} pages ({} ms)", st.pages.len(), t0.elapsed().as_millis());
        *g = Some(st);
        return Ok(());
    }
    let st = g.as_mut().ok_or("fscan start first")?;
    match cmd {
        "changed" | "same" => {
            let mut changed = std::collections::HashSet::new();
            for (a, h) in st.pages.iter_mut() {
                let n = page_hash(*a);
                if n != *h {
                    changed.insert(*a);
                    *h = n;
                }
            }
            let prev = st.cand.take().unwrap_or_else(|| st.pages.iter().map(|p| p.0).collect());
            let next: Vec<u64> = prev.into_iter().filter(|a| changed.contains(a) == (cmd == "changed")).collect();
            if next.len() <= FSCAN_COPY_PAGES {
                st.copy = next.iter().map(|&a| (a, unsafe { std::slice::from_raw_parts(a as *const u8, 4096) }.to_vec())).collect();
            }
            let listed = if next.len() <= 16 { format!(": {}", next.iter().map(|a| format!("{a:#x}")).collect::<Vec<_>>().join(" ")) } else { String::new() };
            info!("fscan {cmd}: {} pages changed, {} candidate pages{} ({} ms){listed}", changed.len(), next.len(), if next.len() <= FSCAN_COPY_PAGES { " (copied)" } else { "" }, t0.elapsed().as_millis());
            st.cand = Some(next);
            Ok(())
        }
        "floats" => {
            if st.copy.is_empty() {
                return Err("no copied pages (fscan same first)".into());
            }
            let mut found: Vec<(u64, f32, f32)> = Vec::new();
            for (&a, old) in st.copy.iter_mut() {
                let now = unsafe { std::slice::from_raw_parts(a as *const u8, 4096) };
                for i in (0..4096).step_by(4) {
                    let (o, n) = (f32::from_le_bytes(old[i..i + 4].try_into().unwrap()), f32::from_le_bytes(now[i..i + 4].try_into().unwrap()));
                    let addr = a + i as u64;
                    if let Some(f) = &st.floats {
                        if f.binary_search(&addr).is_err() {
                            continue;
                        }
                    }
                    // Not plausible values: non-finite, huge, or tiny (integers
                    // and pointers read as floats).
                    let odd = |v: f32| !v.is_finite() || v.abs() > 1e5 || (v != 0.0 && v.abs() < 1e-6);
                    if odd(o) || odd(n) {
                        continue;
                    }
                    let keep = match arg {
                        "+" => n > o,
                        "-" => n < o,
                        "=" => n == o,
                        _ => n != o,
                    };
                    if keep {
                        found.push((addr, o, n));
                    }
                }
                old.copy_from_slice(now);
            }
            let list: Vec<String> = found.iter().take(48).map(|(a, o, n)| format!("{a:#x} {o} -> {n}")).collect();
            info!("fscan floats {arg}: {} candidates ({} ms): {}", found.len(), t0.elapsed().as_millis(), list.join(", "));
            st.floats = Some(found.iter().map(|f| f.0).collect());
            Ok(())
        }
        "reset" => {
            st.floats = None;
            info!("fscan reset: float candidates cleared");
            Ok(())
        }
        _ => Err("usage: fscan start|changed|same|floats [+|-|=|*]|reset".into()),
    }
}

pub fn fatal_report(text: &str) {
    crate::log::write(crate::log::Level::Error, format_args!("{text}"));
    if let Some(dir) = RUN_DIR.get() {
        let _ = std::fs::write(dir.join("crash.txt"), crate::log::redact(text).as_bytes());
    }
    keep_run("crashed (crash.txt)");
}

fn thread_summary() -> String {
    thread_summary_ex(false)
}

/// Every thread's stacks (ours, then the title's and the queues'), taken on a thread of their
/// own and given up after 20 s (a thread stuck inside the driver may not answer a stack walk).
fn stacks_bounded() -> String {
    let (tx, rx) = std::sync::mpsc::channel();
    let _ = std::thread::Builder::new().name("stacks".into()).spawn(move || {
        let _ = tx.send(format!("our threads:\n{}the title's threads and the queues:\n{}", host_thread_stacks(), thread_summary_ex(true)));
    });
    rx.recv_timeout(Duration::from_secs(20)).unwrap_or_else(|_| "the threads' stacks were not taken within 20 s (a thread did not answer)".to_string())
}

/// Guest threads and their states; with `with_stacks`, each running thread is
/// sampled (rip, frame chain, guest return addresses) for hang diagnosis.
/// The stacks of this process's own threads by name (the GPU front end's, front end 2's, the
/// host's event loop's), for the stall watch (`DS_STALL_WATCH`): the queue threads' workers,
/// the completion and release threads are in no other dump.
fn host_thread_stacks() -> String {
    use windows_sys::Win32::System::Diagnostics::ToolHelp::{CreateToolhelp32Snapshot, Thread32First, Thread32Next, TH32CS_SNAPTHREAD, THREADENTRY32};
    let pid = unsafe { windows_sys::Win32::System::Threading::GetCurrentProcessId() };
    let snap = unsafe { CreateToolhelp32Snapshot(TH32CS_SNAPTHREAD, 0) };
    if snap == windows_sys::Win32::Foundation::INVALID_HANDLE_VALUE {
        return String::new();
    }
    let mut tids: Vec<u32> = Vec::new();
    let mut te: THREADENTRY32 = unsafe { std::mem::zeroed() };
    te.dwSize = std::mem::size_of::<THREADENTRY32>() as u32;
    let mut ok = unsafe { Thread32First(snap, &mut te) } != 0;
    while ok {
        if te.th32OwnerProcessID == pid {
            tids.push(te.th32ThreadID);
        }
        ok = unsafe { Thread32Next(snap, &mut te) } != 0;
    }
    unsafe { windows_sys::Win32::Foundation::CloseHandle(snap) };
    let mut named: Vec<(String, u32)> = tids
        .into_iter()
        .map(|t| (guestprof::thread_description(t), t))
        .filter(|(n, _)| (n.starts_with("gpu") || n.starts_with("fe2") || n.starts_with("host") || n.starts_with("forward") || n.starts_with("vblank") || n.starts_with("prepare")) && !n.starts_with("gpu-precompile"))
        .collect();
    named.sort();
    let mut s = String::new();
    for (name, tid) in named {
        if let Some(st) = exception::sample_thread(tid, 0) {
            s.push_str(&format!("    '{name}' ({tid}): {st}\n"));
        }
    }
    s
}

/// Ctrl+F10 in the window: what every thread waits on, then the stacks (the `stacks` report:
/// ours and the title's, the GPU queues, the audio), written to the log, for a player's report
/// of a hang the game still draws through (issue #14: a loading screen that never ends, its fog
/// moving at 60 fps). On a thread of its own, so the window's messages go on meanwhile. The waits
/// come first because they need no thread to answer; the stacks need each thread's context, and
/// a thread stuck inside the driver may never give it (`stacks_bounded` waits 20 s). One report
/// at a time; none after one whose stacks were not taken (its sampling thread may still wait).
pub(crate) fn stacks_hotkey() {
    use std::sync::atomic::{AtomicBool, Ordering};
    static BUSY: AtomicBool = AtomicBool::new(false);
    if BUSY.swap(true, Ordering::SeqCst) {
        info!("Ctrl+F10: the previous report is not finished (or its stacks were not taken): none now");
        return;
    }
    let spawned = std::thread::Builder::new().name("stacks-hotkey".into()).spawn(|| {
        let flip = gpu::FRAMES.load(Ordering::Relaxed);
        info!("Ctrl+F10 at flip {flip}: every thread and what it waits on:\n{}", thread_summary_ex(false));
        let stacks = stacks_bounded();
        let taken = !stacks.starts_with("the threads' stacks were not taken");
        info!("Ctrl+F10 at flip {flip}: the stacks:\n{stacks}");
        if taken {
            BUSY.store(false, Ordering::SeqCst);
        }
    });
    if spawned.is_err() {
        BUSY.store(false, Ordering::SeqCst);
    }
}

fn thread_summary_ex(with_stacks: bool) -> String {
    let mut s = String::new();
    let list: Vec<&'static thread::GuestThread> = match thread::THREADS.try_lock() {
        Some(threads) => threads.iter().copied().collect(),
        None => return s,
    };
    for t in list {
        let state = match t.exited.try_lock() {
            Some(g) => match *g {
                Some(v) => format!("exited({v:#x}) at {:.3} s", t.exit_ms.load(std::sync::atomic::Ordering::Relaxed) as f64 / 1000.0),
                None => "running".into(),
            },
            None => "?".into(),
        };
        let tid = t.host_tid.load(std::sync::atomic::Ordering::Relaxed);
        s.push_str(&format!("    thread {} '{}' host-tid {} {}\n", t.id, t.name(), tid, state));
        if state == "running" {
            if let Some(w) = thread::describe_wait(t) {
                s.push_str(&format!("        {w}\n"));
            }
        }
        if with_stacks && state == "running" {
            if let Some(st) = exception::sample_thread(tid as u32, 16) {
                s.push_str(&format!("        {st}\n"));
            }
        }
    }
    // (Not before the Vulkan device exists: its sections read it. A preparation stopped 10 s in,
    // before the device, panicked here: "Vulkan not initialized", the other PC, 2026-11-21.)
    if gpu::vk::try_ctx().is_some() {
        s.push_str(&gpu::status_ex(with_stacks));
    } else {
        s.push_str("    gpu: no Vulkan device yet\n");
    }
    s.push_str(&displayclock::status_line());
    if gpu::vk::try_ctx().is_some() {
        s.push_str(&gpu::vk::present::stats_line());
    }
    s.push_str(&audio::status());
    s.push_str(&hle::libs::audio2::status());
    s.push_str(&hle::libs::ajm::status());
    s.push_str(&hle::iostat::status());
    s
}

/// Exit codes the launcher knows (`crates/launcher`): the game files cannot be used (told at
/// start), the processor lacks what the game's code needs, the GPU device was lost.
pub const EXIT_GAME_FILES: u32 = 3;
pub const EXIT_CPU: u32 = 8;
pub const EXIT_DEVICE_LOST: u32 = 5;
/// The game's modules could not be loaded, linked or patched (told by `fatal_exit`).
pub const EXIT_START: u32 = 0xD502;
/// Another session runs with the same data folder (told at start).
pub const EXIT_RUNNING: u32 = 9;
/// The process could not be set up before its log (told at start).
pub const EXIT_EARLY: u32 = 10;
/// Windows ran out of memory for the game (told by `fatal_exit`).
pub const EXIT_MEMORY: u32 = 7;
/// No usable Vulkan device at start (told by `fatal_exit`): a lost GPU that
/// needs Windows restarted, a missing or old driver.
pub const EXIT_NO_GPU: u32 = 11;

/// Whether this run has no window (`--headless`): no message boxes then.
pub static HEADLESS: std::sync::atomic::AtomicBool = std::sync::atomic::AtomicBool::new(false);

/// Tells the player about a failure in a message box (the log alone reaches no one), unless
/// this is a development run (`DS_MESSAGES=1` shows them there too) or a headless one.
pub fn player_error(text: &str) {
    let dev = paths().layout == dsconfig::paths::Layout::Dev;
    let show = std::env::var("DS_MESSAGES").map(|v| v == "1").unwrap_or(!dev);
    if !show || HEADLESS.load(std::sync::atomic::Ordering::Relaxed) {
        return;
    }
    use windows_sys::Win32::UI::WindowsAndMessaging::{MB_ICONERROR, MB_OK, MB_SETFOREGROUND, MB_TOPMOST, MessageBoxW};
    let w = |s: &str| s.encode_utf16().chain(std::iter::once(0)).collect::<Vec<u16>>();
    let (t, m) = (w("Regaliant"), w(text));
    unsafe { MessageBoxW(std::ptr::null_mut(), m.as_ptr(), t.as_ptr(), MB_OK | MB_ICONERROR | MB_SETFOREGROUND | MB_TOPMOST) };
}

/// What the player is told when the game has to end (`fatal_exit`), with its log folder.
fn fatal_text(code: u32, reason: &str) -> String {
    let log = RUN_DIR.get().map(|d| format!("\n\nThe log is in:\n{}", d.display())).unwrap_or_default();
    match code {
        EXIT_DEVICE_LOST => format!("The graphics card stopped responding, so the game has to close.\n\nUpdating the graphics driver may help. If it happens again, please report it and attach the log.{log}"),
        EXIT_MEMORY => format!("Windows ran out of memory for the game, so it has to close.\n\nClose other programs and start the game again. If it happens again, let Windows manage the page file size (Settings > System > About > Advanced system settings > Performance > Advanced > Virtual memory).{log}"),
        EXIT_START => format!("The game could not start ({reason}).\n\nPlease report it and attach the log.{log}"),
        EXIT_NO_GPU => format!("The game could not start the graphics card ({reason}).\n\nIf Windows reported a graphics driver problem, restart Windows. Otherwise update the graphics driver: the game needs a Vulkan 1.3 graphics card.{log}"),
        _ => format!("The game met an error it cannot recover from and has to close.\n\nPlease report it and attach the log.{log}"),
    }
}

/// This program's build as symbol stores name it: its PDB's GUID and age, read from the debug
/// directory of its own image. A player's log names the exact build, and the release's symbols
/// folder (`scripts/package.ps1`: `dist/symbols/<package>-<build id>`) is named by it ("none":
/// no CodeView record, a build without debug information).
fn build_id() -> String {
    unsafe {
        let base = windows_sys::Win32::System::LibraryLoader::GetModuleHandleW(std::ptr::null()) as *const u8;
        if base.is_null() {
            return "none".into();
        }
        let rd32 = |p: *const u8| std::ptr::read_unaligned(p as *const u32);
        let nt = base.add(rd32(base.add(0x3c)) as usize);
        let opt = nt.add(24);
        if std::ptr::read_unaligned(opt as *const u16) != 0x20b {
            return "none".into();
        }
        let (dir, size) = (rd32(opt.add(112 + 6 * 8)) as usize, rd32(opt.add(112 + 6 * 8 + 4)) as usize);
        for i in 0..size / 28 {
            let e = base.add(dir + 28 * i);
            if rd32(e.add(12)) != 2 {
                continue;
            }
            let cv = base.add(rd32(e.add(20)) as usize);
            if std::slice::from_raw_parts(cv, 4) != b"RSDS" {
                continue;
            }
            let g = std::slice::from_raw_parts(cv.add(4), 16);
            let d4: String = g[8..16].iter().map(|b| format!("{b:02X}")).collect();
            return format!(
                "{:08X}{:04X}{:04X}{d4}{:X}",
                u32::from_le_bytes([g[0], g[1], g[2], g[3]]),
                u16::from_le_bytes([g[4], g[5]]),
                u16::from_le_bytes([g[6], g[7]]),
                rd32(cv.add(20))
            );
        }
        "none".into()
    }
}

/// The command line, the memory and Windows' build (the run's log, for a player's report).
fn system_line() -> String {
    let args: Vec<String> = std::env::args().skip(1).collect();
    let mem = {
        use windows_sys::Win32::System::SystemInformation::{GlobalMemoryStatusEx, MEMORYSTATUSEX};
        let mut m: MEMORYSTATUSEX = unsafe { std::mem::zeroed() };
        m.dwLength = std::mem::size_of::<MEMORYSTATUSEX>() as u32;
        let g = |b: u64| b as f64 / (1u64 << 30) as f64;
        if unsafe { GlobalMemoryStatusEx(&mut m) } != 0 {
            format!("memory {:.1} GiB ({:.1} free), commit limit {:.1} GiB ({:.1} free)", g(m.ullTotalPhys), g(m.ullAvailPhys), g(m.ullTotalPageFile), g(m.ullAvailPageFile))
        } else {
            "memory unknown".to_string()
        }
    };
    // (RtlGetVersion: the real build, not the manifest's.)
    let build = unsafe {
        #[repr(C)]
        struct Ver {
            size: u32,
            major: u32,
            minor: u32,
            build: u32,
            platform: u32,
            csd: [u16; 128],
        }
        let ntdll = windows_sys::Win32::System::LibraryLoader::GetModuleHandleA(c"ntdll.dll".as_ptr() as *const u8);
        match windows_sys::Win32::System::LibraryLoader::GetProcAddress(ntdll, c"RtlGetVersion".as_ptr() as *const u8) {
            Some(f) => {
                let f: unsafe extern "system" fn(*mut Ver) -> i32 = std::mem::transmute(f);
                let mut v: Ver = std::mem::zeroed();
                v.size = std::mem::size_of::<Ver>() as u32;
                if f(&mut v) == 0 { format!("Windows {}.{} build {}", v.major, v.minor, v.build) } else { "Windows build unknown".to_string() }
            }
            None => "Windows build unknown".to_string(),
        }
    };
    // (Mains or battery, and the battery saver: a laptop on battery runs the preparation slower.)
    let power = unsafe {
        use windows_sys::Win32::System::Power::{GetSystemPowerStatus, SYSTEM_POWER_STATUS};
        let mut s: SYSTEM_POWER_STATUS = std::mem::zeroed();
        if GetSystemPowerStatus(&mut s) != 0 {
            let source = match s.ACLineStatus {
                0 => "on battery",
                1 => "on mains power",
                _ => "power source unknown",
            };
            let battery = if s.BatteryFlag & 128 != 0 { String::new() } else if s.BatteryLifePercent <= 100 { format!(", battery {}%", s.BatteryLifePercent) } else { ", battery level unknown".to_string() };
            format!("{source}{battery}{}", if s.SystemStatusFlag == 1 { ", battery saver on" } else { "" })
        } else {
            "power unknown".to_string()
        }
    };
    format!("{build}; {mem}; {} logical processors; {power}; args {:?}", std::thread::available_parallelism().map_or(0, |n| n.get()), args)
}

/// One game session at a time for a data folder (two would share saves, settings and the
/// shader store): a second start is told and ends. Modes that run no game are not counted.
fn single_instance(cfg: &Config) {
    // (The launcher's compute step and store check take no lock (a game started meanwhile asks
    // them to stop, `ask_preparation_to_stop`), but do not begin beside a game session: a game
    // started on its own during the store check (which writes no progress file) found nothing to
    // stop, and the compute step after it ran on all CPUs beside the player's game; the release
    // review of 2026-11-20 night. They end with EXIT_RUNNING, which the launcher handles.)
    if (cfg.precompile_all.is_some() && cfg.progress.is_some()) || cfg.pb_check.is_some() {
        use windows_sys::Win32::System::Threading::OpenMutexW;
        const SYNCHRONIZE: u32 = 0x0010_0000;
        let h = unsafe { OpenMutexW(SYNCHRONIZE, 0, instance_name(cfg).as_ptr()) };
        if !h.is_null() {
            unsafe { windows_sys::Win32::Foundation::CloseHandle(h) };
            info!("a game session runs with this data folder: this step does not begin");
            log::flush();
            std::process::exit(EXIT_RUNNING as i32);
        }
    }
    // (The launcher's translation check runs no game either and ends before any part of a
    // session's start-up: holding the lock, it turned away a Play pressed while it ran,
    // 2026-11-14.)
    if cfg.precompile_all.is_some() || cfg.compile_bench.is_some() || cfg.translation_fingerprint || cfg.pb_check.is_some() {
        return;
    }
    use windows_sys::Win32::Foundation::{ERROR_ALREADY_EXISTS, GetLastError};
    use windows_sys::Win32::System::Threading::CreateMutexW;
    let name = instance_name(cfg);
    // A game started while the shader preparation runs (hidden: from the launcher, or started
    // on its own beside it): the preparation is asked to stop, as the launcher's Stop asks (its
    // stop file; it goes on from there the next time), and its hold on the data folder waited
    // for, up to 30 s. (The review of 2026-11-20: such a start was told the game was already
    // running, with nothing to be seen.)
    let preparing = cfg.prepare_play.is_none() && ask_preparation_to_stop(cfg);
    // (Kept for the process's life: never closed.)
    let mut h = unsafe { CreateMutexW(std::ptr::null(), 0, name.as_ptr()) };
    if !h.is_null() && unsafe { GetLastError() } == ERROR_ALREADY_EXISTS && preparing {
        let t = Instant::now();
        while t.elapsed() < Duration::from_secs(30) {
            unsafe { windows_sys::Win32::Foundation::CloseHandle(h) };
            std::thread::sleep(Duration::from_millis(250));
            h = unsafe { CreateMutexW(std::ptr::null(), 0, name.as_ptr()) };
            if !h.is_null() && unsafe { GetLastError() } != ERROR_ALREADY_EXISTS {
                info!("the shader preparation stopped for this session after {:.1} s (it goes on the next time)", t.elapsed().as_secs_f64());
                return;
            }
        }
    }
    if !h.is_null() && unsafe { GetLastError() } == ERROR_ALREADY_EXISTS {
        error!("another game session runs with the same data folder: this start ends");
        player_error("The game is already running.");
        log::flush();
        std::process::exit(EXIT_RUNNING as i32);
    }
}

/// The one-session lock's name (`single_instance`): named by the data folder, so a checkout and a
/// player's install do not exclude each other. The shader preparation's self-play takes the
/// player's: it shares their shader store.
fn instance_name(cfg: &Config) -> Vec<u16> {
    let user = prepplay::player_user().unwrap_or(&cfg.paths.user);
    let key: String = user.display().to_string().chars().map(|c| if c.is_ascii_alphanumeric() { c.to_ascii_lowercase() } else { '_' }).collect();
    format!("Local\\Regaliant_{key}").encode_utf16().chain(std::iter::once(0)).collect()
}

/// A shader preparation step seems to run (its progress file written in the last 2 minutes): its
/// stop file written (true), as the launcher's Stop writes it.
fn ask_preparation_to_stop(cfg: &Config) -> bool {
    let progress = cfg.paths.shaders().join("prepare-progress.txt");
    let recent = std::fs::metadata(&progress).and_then(|m| m.modified()).ok().and_then(|t| t.elapsed().ok()).is_some_and(|d| d < Duration::from_secs(120));
    if recent && std::fs::write(progress.with_extension("stop"), "stop (a game session started)\n").is_ok() {
        info!("a shader preparation step runs: asked to stop for this session");
        return true;
    }
    false
}

/// Only the first thread that ends the game tells the player (others wait for the end).
static ENDING: std::sync::atomic::AtomicBool = std::sync::atomic::AtomicBool::new(false);

/// No usable Vulkan device at start (`gpu::exec::enabled`): the player is told and the game
/// ends with EXIT_NO_GPU. Not through `fatal_exit`: its thread summary reads the GPU front
/// end's status, which needs the Vulkan context that is missing (that exit panicked inside
/// itself and the process hung, 2026-11-11, with the GPU lost until a restart).
pub fn no_gpu_exit(reason: &str) -> ! {
    if ENDING.swap(true, std::sync::atomic::Ordering::SeqCst) {
        loop {
            std::thread::sleep(Duration::from_secs(1));
        }
    }
    display::restore_on_exit();
    session_ended();
    crate::log::write(crate::log::Level::Error, format_args!("terminating: {reason}"));
    crate::log::try_flush();
    player_error(&fatal_text(EXIT_NO_GPU, reason));
    unsafe {
        windows_sys::Win32::System::Threading::TerminateProcess(windows_sys::Win32::System::Threading::GetCurrentProcess(), EXIT_NO_GPU);
    }
    loop {
        std::thread::sleep(Duration::from_secs(1));
    }
}

pub fn fatal_exit(code: u32, reason: &str) -> ! {
    if ENDING.swap(true, std::sync::atomic::Ordering::SeqCst) {
        loop {
            std::thread::sleep(Duration::from_secs(1));
        }
    }
    // Exclusive fullscreen's display mode back first (the desktop must not stay changed).
    display::restore_on_exit();
    session_ended();
    // (This session's crash suspects: a fatal compile crash was recorded by its report already,
    // and no other end here is one; left behind, the next session would have listed them.)
    gpu::pcache::clear_crash_suspects();
    // After a fault, every thread's frames (what the others were doing:
    // the title's teardown when a job read freed memory).
    let summary = if reason == "fatal exception" { thread_summary_ex(true) } else { thread_summary() };
    crate::log::write(crate::log::Level::Error, format_args!("terminating: {reason}\n{summary}"));
    crate::log::try_flush();
    // The player is told (the game's window out of the way of the message).
    if let Some(w) = gpu::vk::window::get() {
        unsafe { windows_sys::Win32::UI::WindowsAndMessaging::ShowWindow(w.hwnd as _, windows_sys::Win32::UI::WindowsAndMessaging::SW_MINIMIZE) };
    }
    player_error(&fatal_text(code, reason));
    unsafe {
        windows_sys::Win32::System::Threading::TerminateProcess(windows_sys::Win32::System::Threading::GetCurrentProcess(), code);
    }
    loop {
        std::thread::sleep(Duration::from_secs(1));
    }
}

/// A profile-instrumented build (`-Cprofile-generate` with `--cfg
/// dsport_pgo_gen`) writes its counters here: TerminateProcess skips the
/// profile runtime's exit hook.
fn pgo_write() {
    #[cfg(dsport_pgo_gen)]
    unsafe {
        unsafe extern "C" {
            fn __llvm_profile_write_file() -> i32;
        }
        let r = __llvm_profile_write_file();
        info!("profile counters written (status {r})");
        crate::log::flush();
    }
}

/// The session ended in an orderly way (`shutdown`, `fatal_exit`): its marker goes.
fn session_ended() {
    let m = paths().running_marker();
    if std::fs::read_to_string(&m).is_ok_and(|t| t.starts_with(&format!("pid {}\n", std::process::id()))) {
        let _ = std::fs::remove_file(m);
    }
}

/// A session found hung (`prepplay`'s and the compute step's watches): `why` and every thread's
/// stacks into the log, then the process ended at once with `code`, not through `shutdown`
/// (its GPU teardown would wait for the hung work).
pub fn hung(why: &str, code: i32) -> ! {
    error!("{why}; {}", stacks_bounded());
    // (A self-play stage's health line too, its counters as they stood.)
    prepplay::health_if_playing(code);
    session_ended();
    // (Its compiles in flight stay on record: a compile that does not return is a reason to end.)
    gpu::pcache::clear_crash_suspects_keep_inflight();
    crate::log::flush();
    unsafe {
        windows_sys::Win32::System::Threading::TerminateProcess(windows_sys::Win32::System::Threading::GetCurrentProcess(), code as u32);
    }
    loop {
        std::thread::sleep(Duration::from_secs(1));
    }
}

pub fn shutdown(code: i32) -> ! {
    // (A PlayStation controller keeps its last vibration after the game ends otherwise.)
    pspad::motors_off();
    display::restore_on_exit();
    session_ended();
    hle::libs::videoout::wait_shots();
    gpu::shutdown();
    // (A clean end: faults the driver took itself were not crashes, `pcache::note_crash_suspect`.)
    gpu::pcache::clear_crash_suspects();
    // (The exact key specs still queued: written, the session's last ones.)
    gpu::precompile::exact_flush();
    info!("shutdown with status {code}\n{}", thread_summary());
    crate::log::flush();
    pgo_write();
    unsafe {
        windows_sys::Win32::System::Threading::TerminateProcess(windows_sys::Win32::System::Threading::GetCurrentProcess(), code as u32);
    }
    loop {
        std::thread::sleep(Duration::from_secs(1));
    }
}

fn read_proc_param(m: &loader::LoadedModule) -> (u64, String, i32) {
    let mut stack = 0u64;
    let mut name = "GAME_MainThread".to_string();
    let mut prio = hle::kernel::pthread::PRIO_DEFAULT;
    if let Some(pp) = m.proc_param {
        unsafe {
            let size = *(pp as *const u64);
            let magic = *((pp + 8) as *const u32);
            let sdk = *((pp + 0x10) as *const u64);
            hle::kernel::misc::SDK_VERSION.store(sdk, std::sync::atomic::Ordering::Relaxed);
            info!("proc param at {pp:#x}: size {size:#x} magic {magic:#x} sdk {sdk:#x}");
            let field = |off: u64| if off + 8 <= size { *((pp + off) as *const u64) } else { 0 };
            let name_pp = field(0x20);
            if name_pp != 0 && *(name_pp as *const u64) != 0 {
                name = hle::cstr(*(name_pp as *const u64) as *const u8).to_string();
            }
            let prio_p = field(0x28);
            if prio_p != 0 {
                prio = *(prio_p as *const i32);
            }
            let stack_p = field(0x30);
            if stack_p != 0 {
                stack = *(stack_p as *const u32) as u64;
            }
        }
    }
    (stack, name, prio)
}

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

fn main() {
    // `--probe-streamline <dir>`: a start's separate check that Streamline loads (`gpu::streamline::
    // probe_streamline`), before anything else: no log, no window, no error popups.
    {
        let mut args = std::env::args().skip(1);
        if args.next().as_deref() == Some("--probe-streamline") {
            unsafe {
                use windows_sys::Win32::System::Diagnostics::Debug::{SetErrorMode, SEM_FAILCRITICALERRORS, SEM_NOGPFAULTERRORBOX, SEM_NOOPENFILEERRORBOX};
                SetErrorMode(SEM_FAILCRITICALERRORS | SEM_NOGPFAULTERRORBOX | SEM_NOOPENFILEERRORBOX);
            }
            gpu::streamline::probe_streamline(std::path::Path::new(&args.next().unwrap_or_default()));
        }
    }
    // Per-monitor DPI awareness for the whole process, before any window, monitor or surface is
    // asked about (a player's report, 2026-11-23: at 175% Windows scaling the picture covered the
    // top-left 2194x1234 of a 3840x2160 fullscreen. Only the window's thread was aware; the
    // swapchain, made on the vblank thread, took the surface's size as Windows scales it for an
    // unaware thread).
    let aware = unsafe { windows_sys::Win32::UI::HiDpi::SetProcessDpiAwarenessContext(windows_sys::Win32::UI::HiDpi::DPI_AWARENESS_CONTEXT_PER_MONITOR_AWARE_V2) } != 0;
    gpu::vk::window::PROCESS_DPI_AWARE.store(aware, std::sync::atomic::Ordering::Relaxed);
    // A run without a window (the launcher's preparation and checks, the drivers' runs) known
    // from the command line before anything can fail: no message box (`player_error`) and none
    // of Windows' own error popups, which a hidden run would leave waiting for a click (the
    // review of 2026-11-20: the memory range's error came before the arguments were read).
    // The preparation's runs keep the PC from sleeping while they last.
    {
        let args: Vec<String> = std::env::args().skip(1).collect();
        if args.iter().any(|a| matches!(a.as_str(), "--headless" | "--prepare-play" | "--precompile-all" | "--pb-check" | "--translation-fingerprint" | "--compile-bench")) {
            HEADLESS.store(true, std::sync::atomic::Ordering::Relaxed);
            unsafe {
                use windows_sys::Win32::System::Diagnostics::Debug::{SetErrorMode, SEM_FAILCRITICALERRORS, SEM_NOGPFAULTERRORBOX, SEM_NOOPENFILEERRORBOX};
                SetErrorMode(SEM_FAILCRITICALERRORS | SEM_NOGPFAULTERRORBOX | SEM_NOOPENFILEERRORBOX);
            }
        }
        if args.iter().any(|a| matches!(a.as_str(), "--prepare-play" | "--precompile-all")) {
            unsafe {
                use windows_sys::Win32::System::Power::{SetThreadExecutionState, ES_CONTINUOUS, ES_SYSTEM_REQUIRED};
                SetThreadExecutionState(ES_CONTINUOUS | ES_SYSTEM_REQUIRED);
            }
        }
    }
    // The player's diagnostic switches, before anything reads one (the host's choice included;
    // the hosted game inherits them).
    let diagnostics = apply_diagnostics();
    // Under front end 2 (the default) and with `--host` this process only
    // hosts dsport (its debugger; `hosted::wants_host`).
    if hosted::wants_host() {
        hosted::run_host();
    }
    // The guest address range must be reserved before the process grows
    // (the first heap allocation reserved it already, see `heap`).
    // (Before the log exists: the player is told, and the launcher shows the code.)
    if let Err(e) = mem::reserve_guest_range() {
        eprintln!("{e}");
        // (Before the log exists, the reason also goes to a file in the logs folder, which the
        // launcher's Open logs shows: a hidden step's player saw only "status 10", issue #15.)
        let note = early_failure_note(&e);
        player_error(&format!("The game could not reserve the memory range its program needs ({e}).

This happens when other software takes that range. Restart the PC and try again.{note}"));
        std::process::exit(EXIT_EARLY as i32);
    }
    let slot = match tls::init_slot() {
        Ok(s) => s,
        Err(e) => {
            eprintln!("{e}");
            let note = early_failure_note(&format!("setting up the threads' slot failed: {e}"));
            player_error(&format!("The game could not set up its threads ({e}).{note}"));
            std::process::exit(EXIT_EARLY as i32);
        }
    };
    let mut cfg = parse_args();
    // The shader preparation's self-play: its own folders, the player's shader store, no window,
    // sound or system input (before any thread starts).
    if let Some(stage) = cfg.prepare_play {
        prepplay::setup(stage, &mut cfg);
    }
    PATHS.set(cfg.paths.clone()).ok();
    HEADLESS.store(cfg.headless, std::sync::atomic::Ordering::Relaxed);
    // (The run's own logs folder: the self-play's log is among the player's.)
    prune_runs(cfg.run_dir.parent().unwrap_or(&cfg.paths.logs()), cfg.paths.logs_kept());
    if let Err(e) = std::fs::create_dir_all(&cfg.run_dir).and_then(|_| log::init(&cfg.run_dir.join("dsport.log"))) {
        player_error(&format!("The game could not write its log in {}.\n\n{e}", cfg.run_dir.display()));
        std::process::exit(1);
    }
    RUN_DIR.set(cfg.run_dir.clone()).ok();
    log::set_thread_name("host");
    if !diagnostics.is_empty() {
        crate::warn!("diagnostics.txt in the settings folder: {}", diagnostics.join(", "));
    }
    // Panics (host threads abort the process) are logged with the thread
    // that panicked: the run log is what post-mortems read.
    // A panic ends the game as a crash does (the player told, the display mode back).
    std::panic::set_hook(Box::new(move |info| {
        crate::error!("panic: {info}\n{}", std::backtrace::Backtrace::force_capture());
        crate::fatal_exit(0xD540, "panic");
    }));
    if cfg.debug || cfg.trace {
        log::LEVEL.store(log::Level::Debug as u8, std::sync::atomic::Ordering::Relaxed);
    }
    log::TRACE_HLE.store(cfg.trace, std::sync::atomic::Ordering::Relaxed);
    hle::LENIENT_IMPORTS.store(cfg.lenient, std::sync::atomic::Ordering::Relaxed);
    info!("Regaliant {} (build {}) | run dir {} | TLS slot {} | host heap: {} | guest write watch: {}", env!("CARGO_PKG_VERSION"), build_id(), cfg.run_dir.display(), slot, heap::name(), mem::write_watch_state());
    info!("start: {}", system_line());
    info!("files: {:?} layout, settings and saves in {}, logs and caches in {}", cfg.paths.layout, cfg.paths.user.display(), cfg.paths.cache.display());
    if let Some(p) = &cfg.game_problem {
        error!("game files: {}", p.replace('\n', " "));
        // No game folder chosen yet: a new player who opened the game rather than its launcher.
        // The launcher opens, where choosing the folder is the first step, instead of an error
        // to click away (2026-11-19).
        if cfg.game_unset && !cfg.headless && cfg.paths.layout != dsconfig::paths::Layout::Dev {
            let launcher = cfg.paths.exe_dir.join(dsconfig::paths::LAUNCHER_EXE);
            if launcher.is_file() && std::process::Command::new(&launcher).current_dir(&cfg.paths.exe_dir).spawn().is_ok() {
                info!("game files: no game folder chosen; the launcher opens");
                log::flush();
                std::process::exit(EXIT_GAME_FILES as i32);
            }
        }
        player_error(p);
        log::flush();
        std::process::exit(EXIT_GAME_FILES as i32);
    }
    cpucheck::check();
    single_instance(&cfg);
    // A session in progress (the launcher's: one left behind did not end normally), written once
    // this start holds the data folder (written before the check, a second start replaced the
    // running game's marker and then deleted it: the release review of 2026-11-20 night). Runs
    // of no game write none (the launcher's check left its marker behind, then read as a session
    // that did not end normally, or replaced a running game's).
    if cfg.precompile_all.is_none() && cfg.compile_bench.is_none() && !cfg.translation_fingerprint && cfg.prepare_play.is_none() && cfg.pb_check.is_none() {
        let _ = std::fs::write(cfg.paths.running_marker(), format!("pid {}\nlog {}\n", std::process::id(), cfg.run_dir.display()));
    }
    // (Its driver reads the log from here on; the title reads the tour's automation at its boot.)
    // (`DS_ZAR_THREADS=<n>`: threads that decompress one large read of a `.zar` game.)
    if let Some(n) = std::env::var("DS_ZAR_THREADS").ok().and_then(|v| v.parse().ok()) {
        dsconfig::zar::set_threads(n);
    }
    game::set(cfg.source.clone());
    prepplay::start();
    exception::install();
    heapsites::start();
    hosted::init();
    hosted::note_run_dir(&cfg.run_dir);
    hosted::self_test_maybe();
    if cfg.dispatch_bench {
        let code = gpu::dispbench::run();
        log::flush();
        std::process::exit(code);
    }
    if let Some(c) = &cfg.pipeline_test {
        let code = gpu::exec::pipeline_test(c);
        log::flush();
        std::process::exit(code);
    }
    if let Some((c, cs)) = &cfg.record_bench {
        let code = gpu::exec::record_bench(c, *cs);
        log::flush();
        std::process::exit(code);
    }
    // The launcher's translation check: the translator's answer for a sample of the game's
    // compute programs (it depends on the device and the translator only), before any part of a
    // session: no window, settings, saves or stores. It takes no one-session lock and may run
    // beside a game session (it ran their start-up until 2026-11-14: a pending Save Folder move
    // or PS5 import applied under a running game, the stores' unfinished files deleted).
    if cfg.translation_fingerprint {
        // (A check takes seconds, 2.3 s on the development PC, ~50 s on the other PC's first: one
        // not done in 150 s is a driver call that does not return (60 s until 2026-11-21 night,
        // which a first check on a slow drive could pass). The launcher waits a little longer.)
        let _ = std::thread::Builder::new().name("check-watch".into()).spawn(|| {
            std::thread::sleep(Duration::from_secs(150));
            hung("translation check: not done in 150 s", prepplay::EXIT_HUNG);
        });
        // (This program's version first: the launcher compares it with its own, a new version's
        // files unpacked over a game still running would leave the old game beside the new
        // launcher.)
        println!("version = {}", std::env::var("DS_SIM_VERSION").unwrap_or_else(|_| dsconfig::VERSION.to_string()));
        // (`DS_SIM_CHECK_DELAY_S=<s>`, a test of a slow PC's check: the answer that many seconds
        // later, the process and its session lock held meanwhile, as the other PC's 50 s check
        // held them. Under the 60 s watch above.)
        if let Some(s) = std::env::var("DS_SIM_CHECK_DELAY_S").ok().and_then(|v| v.trim().parse::<u64>().ok()) {
            std::thread::sleep(Duration::from_secs(s.min(140)));
        }
        // (Translations made here are for the game's hosted sessions: their specs as those make
        // them, `fe2::specs_as_hosted`.)
        gpu::fe2::specs_as_hosted();
        // (Through its cache beside the store: the bundles read only after an install, an
        // update or another card or driver.)
        let fp_dir = cfg.cache.clone().unwrap_or_else(|| cfg.paths.shaders());
        let code = match gpu::compbench::translation_fingerprint_cached(game::source(), Some(&fp_dir), &build_id()) {
            Ok(fp) => {
                // (The card and driver this program uses, as the prepared marker names them: the
                // launcher's records follow the game's own device, the review of 2026-11-20.)
                if let Some(c) = gpu::vk::try_ctx() {
                    println!("device = {}", c.device_name);
                    println!("driver = {}", unsafe { c.instance.get_physical_device_properties(c.pdev) }.driver_version);
                }
                println!("translation = {fp}");
                info!("translation fingerprint: {fp}");
                0
            }
            Err(e) => {
                error!("translation fingerprint: {e}");
                2
            }
        };
        log::flush();
        std::process::exit(code);
    }
    gpu::particles::start(game::source().clone());
    gpu::capture::init_from_env();
    gpu::start_profiler();
    guestprof::start();
    hostload::start();
    if std::env::var_os("DS_TEST_FAULT").is_some() {
        info!("DS_TEST_FAULT: writing to an invalid address from host code");
        unsafe { std::ptr::write_volatile(0x32c3be8 as *mut u64, 1) };
    }
    thread::apply_power_optout(gpu::tune::power_optout());
    hle::kernel::time::init();
    // (Its samples are timed by the process clock.)
    guestprof::start_trace();
    // Compiled pipelines and learned shader specs: local-data, or the
    // `--cache` directory (Play.bat: one next to the package folder, so a
    // new package starts with what earlier ones learned on this machine).
    // Before the window: its Vulkan device reads the store's binaries mode
    // (`pcache::binaries_off_for`; a windowed session had made its device
    // before the store's folder was known and kept the binaries on where the
    // store check had turned them off, the release review of 2026-11-20).
    let cache_root = cfg.cache.clone().unwrap_or_else(|| cfg.paths.shaders());
    if cfg.cache.is_some() {
        hle::kernel::fs::ensure_dir(&cache_root);
        info!("pipeline and shader-spec cache: {}", cache_root.display());
    }
    // (The launcher's probe only reads the store: nothing written from its opening on.)
    if cfg.pb_check.as_deref() == Some("probe") {
        gpu::pcache::set_read_only();
    }
    // DS_PIPELINE_DIR=<directory name under the cache>: another pipeline
    // store (experiments that start without compiled pipelines).
    let pipe_dir = std::env::var("DS_PIPELINE_DIR").ok().filter(|d| !d.is_empty() && d.chars().all(|c| c.is_ascii_alphanumeric() || c == '-' || c == '_'));
    // A translator switch makes other SPIR-V under the same pipeline keys:
    // no pipeline store then unless one is named for the experiment.
    match (pipe_dir, gpu::trcache::translator_switch()) {
        (Some(d), _) => gpu::pcache::set_dir(cache_root.join(d)),
        (None, Some(v)) => info!("pipeline store: off while {v} is set (DS_PIPELINE_DIR=<name> keeps one)"),
        (None, None) => gpu::pcache::set_dir(cache_root.join("pipelines")),
    }
    if !cfg.headless {
        if let Err(e) = gpu::vk::present::init("Regaliant - Demon's Souls") {
            warn!("no presentation window: {e}");
        }
    }

    // Guest file system. The supplied game tree is mounted read-only; all
    // writable state lives in the port's own folders (`dsconfig::paths`).
    let rt = cfg.paths.runtime();
    for d in ["temp0", "download0", "data", "devlog"] {
        hle::kernel::fs::ensure_dir(&rt.join(d));
    }
    match game::source() {
        dsconfig::zar::Source::Zar { .. } => info!("application root: the archive {} (read-only, read in place), code in its folder '{}'", game::source().display(), cfg.code_dir),
        _ => info!("application root {} (read-only), code {}", cfg.app.display(), cfg.code.display()),
    }
    hle::kernel::fs::mount_game("/app0", game::source());
    hle::kernel::fs::mount_game("/hostapp", game::source());
    hle::kernel::fs::mount("/temp0", vec![rt.join("temp0")], true);
    hle::kernel::fs::mount("/temp", vec![rt.join("temp0")], true);
    hle::kernel::fs::mount("/download0", vec![rt.join("download0")], true);
    hle::kernel::fs::mount("/data", vec![rt.join("data")], true);
    // The title writes its own log files to /devlog/app when it can (useful diagnostics).
    hle::kernel::fs::ensure_dir(&rt.join("devlog").join("logs"));
    hle::kernel::fs::mount("/devlog/app", vec![rt.join("devlog")], true);
    // Keyboard and mouse layout (written with the defaults when missing).
    input::init(&cfg.paths.controls_file());
    // `DS_ACHIEVEMENTS_FILE`: another file (test runs: theirs, apart from a player's).
    let achievements_file = std::env::var_os("DS_ACHIEVEMENTS_FILE").map(PathBuf::from).unwrap_or_else(|| cfg.paths.achievements_file());
    achievements::init(game::source(), achievements_file, "en-GB");
    // Monitors and the PC tab's display choices (before its menu lists are made).
    // Safe mode: the display choices and PC settings at their defaults for this session (a
    // scratch file for the display's, the player's files left as they are).
    if cfg.safe_mode {
        info!("safe mode (the launcher's, for this start): default settings, a window, no upscaler, no Frame Generation or Reflex, HDR off, arming and held pages off");
        let scratch = cfg.paths.cache.join("safe-mode-display.txt");
        let _ = std::fs::remove_file(&scratch);
        display::init(scratch);
    } else {
        display::init(cfg.paths.display_file());
    }
    if !cfg.headless {
        display::apply_test_mode();
    }
    pcsettings::init(cfg.paths.settings_file(), cfg.safe_mode);
    // The saves: the PC tab's Save Folder (the port's own by default), a PS5 import asked for
    // (a game session's: the shader preparation leaves them as they are).
    // (Not the store check either: it runs without the one-session lock, and a pending Save
    // Folder move or PS5 import applied there would run beside a game started on its own; the
    // release review of 2026-11-20 night.)
    if cfg.precompile_all.is_none() && cfg.compile_bench.is_none() && cfg.pb_check.is_none() {
        hle::libs::savedata::init(pcsettings::saves::start(&cfg.paths, &hle::libs::savedata::title_id()));
    }
    // Translations kept across runs (`gpu::trcache`). (Not by the launcher's probe, which translates
    // nothing: the store's pruning would delete another process's translations in progress; the
    // review of 2026-11-21.)
    if cfg.pb_check.as_deref() != Some("probe") {
        gpu::trcache::set_dir(cache_root.join("translations"));
    }
    // DS_SPECS=<file name under the cache, or a path>: where the shader
    // programs' learned specs and code are kept (`gpu::precompile`).
    let specs = std::env::var("DS_SPECS").ok().filter(|v| !v.is_empty()).map(std::path::PathBuf::from);
    gpu::precompile::set_store(match specs {
        Some(p) if p.components().count() > 1 => p,
        Some(p) => cache_root.join(p),
        None => cache_root.join("shader-specs.txt"),
    });
    // (Translations made here are for the game's hosted sessions: their specs as those make
    // them, `fe2::specs_as_hosted`.)
    if cfg.compile_bench.is_some() || cfg.precompile_all.is_some() {
        gpu::fe2::specs_as_hosted();
    }
    if let Some(a) = &cfg.compile_bench {
        let code = gpu::compbench::run(a, game::source());
        gpu::pcache::clear_crash_suspects();
        log::flush();
        std::process::exit(code);
    }
    if let Some(n) = cfg.precompile_all {
        // (Every compile of the compute step is ahead of use: the crash list's background rule.)
        gpu::pcache::background_process();
        let code = gpu::compbench::precompile_all(game::source(), n, cfg.progress.as_deref(), Some(&cache_root.join("prepared.txt")));
        prepplay::write_health(cfg.progress.as_deref(), &prepplay::health_line("compute", code));
        gpu::pcache::clear_crash_suspects();
        log::flush();
        // (Its work done: a teardown that stalls (a driver thread) ends the process with its own
        // status 60 s later, not the launcher's no-progress limit 15 minutes later with none; the
        // review of 2026-11-21 night.)
        let _ = std::thread::Builder::new().name("exit-watch".into()).spawn(move || {
            std::thread::sleep(Duration::from_secs(60));
            unsafe { windows_sys::Win32::System::Threading::TerminateProcess(windows_sys::Win32::System::Threading::GetCurrentProcess(), code as u32) };
        });
        std::process::exit(code);
    }
    if let Some(phase) = &cfg.pb_check {
        // (A check takes seconds: one not done in 2 minutes is a driver call that does not return;
        // 5 minutes until 2026-11-20, a long wait on a PC whose next steps would hang too.)
        let _ = std::thread::Builder::new().name("pb-check-watch".into()).spawn(|| {
            std::thread::sleep(Duration::from_secs(120));
            hung("pb-check: not done in 120 s", prepplay::EXIT_HUNG);
        });
        let code = gpu::pcache::pb_check(phase);
        log::flush();
        std::process::exit(code);
    }
    // A player's install executes the GPU work unless DS_GPU says otherwise (a development run
    // sets DS_GPU=1 itself; the launcher's preparation and fingerprint runs above are as before).
    if cfg.paths.layout != dsconfig::paths::Layout::Dev {
        gpu::exec::default_on();
    }

    // Modules: the decrypted executable and the title's own libraries.
    let t0 = Instant::now();
    // (From the folder as before, or read out of the game's archive.)
    let load_module = |rel: &str, is_main: bool| -> Result<Arc<loader::LoadedModule>, String> {
        match game::source() {
            dsconfig::zar::Source::Dir(_) => loader::load(&rel.split('/').fold(cfg.code.clone(), |p, c| p.join(c)), is_main),
            src => {
                let full = game::join(&cfg.code_dir, rel);
                let bytes = src.read(&full).map_err(|e| format!("{full} in {}: {e}", src.display()))?;
                loader::load_bytes(bytes, &src.path().join(full.replace('/', "\\")), is_main)
            }
        }
    };
    let eboot = load_module("eboot.bin", true).unwrap_or_else(|e| {
        error!("loading eboot failed: {e}");
        fatal_exit(EXIT_START, "load failure");
    });
    let mut libs: Vec<Arc<loader::LoadedModule>> = Vec::new();
    for name in ["libc.prx", "libSceNpCppWebApi.prx"] {
        if !game::source().is_file(&game::join(&cfg.code_dir, &format!("sce_module/{name}"))) {
            error!("{name} not found in {}", game::source().display());
            fatal_exit(EXIT_START, "missing title module");
        }
        match load_module(&format!("sce_module/{name}"), false) {
            Ok(m) => libs.push(m),
            Err(e) => {
                error!("loading {name} failed: {e}");
                fatal_exit(EXIT_START, "load failure");
            }
        }
    }
    let _ = hle::registry();
    let modules: Vec<Arc<loader::LoadedModule>> = loader::MODULES.read().clone();
    for m in &modules {
        match loader::relocate(m) {
            Ok(s) => info!("{}: linked (lle {}, hle {}, unresolved {}, weak-null {})", m.name, s.lle, s.hle, s.unresolved, s.weak_null),
            Err(e) => {
                error!("relocating {} failed: {e}", m.name);
                fatal_exit(EXIT_START, "link failure");
            }
        }
    }
    for m in &modules {
        match loader::finalize(m) {
            Ok(s) => loader::report_patch(m, &s),
            Err(e) => {
                error!("finalizing {} failed: {e}", m.name);
                fatal_exit(EXIT_START, "patch failure");
            }
        }
    }
    let unresolved = hle::UNRESOLVED.read().len();
    {
        // Planning aid: every unresolved function import, by library.
        let mut lines: Vec<String> = hle::UNRESOLVED.read().iter().map(|u| format!("{}\t{}\t{}", u.lib, u.nid, hle::describe_nid(&u.nid))).collect();
        lines.sort();
        let _ = std::fs::write(cfg.run_dir.join("unresolved.txt"), lines.join("\n") + "\n");
    }
    info!("load+link+patch took {:.1} ms; {} distinct unresolved function imports", t0.elapsed().as_secs_f64() * 1000.0, unresolved);

    // Guest main thread.
    let (stack, name, prio) = read_proc_param(&eboot);
    let stack = stack.max(8 << 20);
    let main_t = thread::new_thread(&name, eboot.entry, 0, stack, prio);
    let eboot2 = eboot.clone();
    thread::spawn_host(
        &name,
        stack as usize + (1 << 20),
        Box::new(move || {
            thread::enter(main_t);
            for m in &libs {
                if let Some(init) = m.init {
                    info!("starting module {} at {}", m.name, loader::describe_addr(init));
                    let r = thread::run_entry(main_t, init, 0, 0);
                    info!("module {} start returned {:#x}", m.name, r);
                }
            }
            let params = mem::arena_alloc(40, 8);
            unsafe {
                *(params as *mut u32) = 1;
                *((params + 8) as *mut u64) = mem::arena_cstr("eboot.bin");
            }
            if let Ok(list) = std::env::var("DS_BREAK") {
                for off in list.split(',').filter_map(|v| u64::from_str_radix(v.trim().trim_start_matches("0x"), 16).ok()) {
                    match exception::install_breakpoint(eboot2.base + off) {
                        Ok(()) => info!("breakpoint at {}", loader::describe_addr(eboot2.base + off)),
                        Err(e) => warn!("breakpoint {off:#x}: {e}"),
                    }
                }
            }
            info!("entering {} at {}", eboot2.name, loader::describe_addr(eboot2.entry));
            let r = thread::run_entry(main_t, eboot2.entry, params, hle::kernel::misc::rtld_exit_fn());
            info!("eboot entry returned {r:#x}");
            shutdown(r as i32);
        }),
    )
    .expect("spawn main thread");

    let start = Instant::now();
    let mut control = cfg.control.clone().map(|p| (p, 0u64));
    let mut ticks = 0u64;
    // Boot watch (`DS_BOOT_WATCH=<s>`, default 15, 0 off): a boot that has
    // flipped no frame by then dumps the guest thread stacks once (some
    // boots stalled ~40 s early in the title's start-up, FINDINGS.md
    // 2026-10-18 night; a normal boot flips within ~5 s).
    let mut boot_watch = std::env::var("DS_BOOT_WATCH").ok().and_then(|v| v.parse::<u64>().ok()).unwrap_or(15);
    // Stall watch (`DS_STALL_WATCH=<ms>`, diagnosis; 0, the default: off): once frames flip, a
    // gap of that long without a flip logs every thread's stack and wait record while it lasts
    // (at most `DS_STALL_WATCH_MAX` times a run, default 4; once a gap).
    let stall_watch = std::env::var("DS_STALL_WATCH").ok().and_then(|v| v.parse::<u64>().ok()).unwrap_or(0);
    let mut stall_left = std::env::var("DS_STALL_WATCH_MAX").ok().and_then(|v| v.parse::<u32>().ok()).unwrap_or(4);
    let mut stall_at = (0u64, Instant::now(), false);
    // (Time in which this loop did not run, the PC asleep or the process held: not counted
    // against the time limit; an overnight sleep had ended a preparation's step at its limit.)
    let (mut slept, mut last_tick) = (Duration::ZERO, Instant::now());
    loop {
        std::thread::sleep(Duration::from_millis(250));
        ticks += 1;
        let gap = last_tick.elapsed();
        last_tick = Instant::now();
        if gap > Duration::from_secs(10) {
            slept += gap;
            info!("the process did not run for {:.0} s (the PC asleep or the process held): time limits extended", gap.as_secs_f64());
        }
        if stall_watch > 0 && stall_left > 0 {
            let f = gpu::FRAMES.load(std::sync::atomic::Ordering::Relaxed);
            if f != stall_at.0 {
                stall_at = (f, Instant::now(), false);
            } else if f > 30 && gpu::vk::try_ctx().is_some() && !stall_at.2 && stall_at.1.elapsed() >= Duration::from_millis(stall_watch) {
                stall_at.2 = true;
                stall_left -= 1;
                warn!("stall watch: no flip for {} ms after flip {f} (DS_STALL_WATCH); our threads:\n{}the title's threads and the queues:\n{}", stall_at.1.elapsed().as_millis(), host_thread_stacks(), thread_summary_ex(true));
            }
        }
        if boot_watch > 0 && start.elapsed() >= Duration::from_secs(boot_watch) {
            if gpu::FRAMES.load(std::sync::atomic::Ordering::Relaxed) == 0 && gpu::exec::enabled() {
                warn!("boot watch: no frame flipped {boot_watch} s after start (DS_BOOT_WATCH); guest thread stacks:\n{}", thread_summary_ex(true));
            }
            boot_watch = 0;
        }
        if ticks % 8 == 0 && exception::break_stats_enabled() {
            exception::dump_break_stats();
        }
        if ticks % 4 == 0 {
            titlemem::tick(start.elapsed().as_secs());
        }
        crate::log::try_flush();
        if let Some((path, offset)) = control.as_mut() {
            poll_control(path, offset);
        }
        if gpu::vk::window::CLOSED.load(std::sync::atomic::Ordering::SeqCst) {
            info!("window closed by user");
            shutdown(0);
        }
        if let Some(s) = cfg.seconds {
            if start.elapsed().saturating_sub(slept) >= Duration::from_secs(s) {
                info!("time limit of {s}s reached; {}", stacks_bounded());
                // (A self-play stage that ran out of time did not play to its end.)
                if cfg.prepare_play.is_some() {
                    error!("prepare-play: the stage did not end within its time limit of {s} s");
                    shutdown(prepplay::EXIT_FAILED);
                }
                shutdown(0);
            }
        }
    }
}
