//! The shader preparation's self-play (`--prepare-play <stage>`, run by the launcher after the
//! compute programs are prepared): the game plays itself before the player's first session, so
//! the graphics pipelines a playthrough meets, and the compute programs' real specs, are made
//! and kept in the player's shader store ahead of it. (A store prepared for compute alone left a
//! first session ~380 graphics pipelines to make while it played and ~180 compute programs whose
//! guessed variants did not match, 2026-11-19.)
//!
//! Nothing of it shows or sounds: no window (`--headless`: a swapchain that is never shown) and
//! no message boxes, no audio device (`audio::SILENT`); the system's controllers, keyboard and
//! mouse are neither read nor rumbled (`input::SYSTEM_OFF`). It plays through its own presses,
//! reading its log as the development drivers do (`log::tap_read`). Its settings (the player's
//! PC settings and display choices, copied, its saves always its own), saves, trophies and the
//! title's own folders are its own (`<cache>/prepare`); the shader store is the player's, and the
//! player's one-session lock is taken (the two never run together).
//!
//! Stages:
//! - `newgame`: from no saves, the first-run screens, a new game (character creation with its
//!   menus opened), the opening cinematic played, the tutorial's first steps, the in-game menu
//!   with its character preview.
//! - `tour`: the save `newgame` made, continued; the title's own automation (its developers'
//!   screenshot positions, `misc/automation/demonssoulsscreenshots.txt`) visits the levels'
//!   positions and looks four ways at each (`DS_PREPARE_TOUR` changes the choice).
//!
//! Progress in `--progress <file>` (`<steps done> <steps>` on two lines, as the precompile writes
//! it); the stage stops when the launcher's stop file appears (`<file>` with the extension
//! `.stop`, as the precompile reads it). Exit status 0 when the stage was played to its
//! end, `EXIT_STOPPED` when the stop file ended it, `EXIT_FAILED` when it could not go on (its log
//! says where).

use std::collections::VecDeque;
use std::path::{Path, PathBuf};
use std::sync::OnceLock;
use std::sync::atomic::Ordering::Relaxed;
use std::time::{Duration, Instant};

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum Stage {
    NewGame,
    Tour,
}

impl Stage {
    pub fn parse(s: &str) -> Option<Stage> {
        match s {
            "newgame" => Some(Stage::NewGame),
            "tour" => Some(Stage::Tour),
            _ => None,
        }
    }

    pub fn name(self) -> &'static str {
        match self {
            Stage::NewGame => "newgame",
            Stage::Tour => "tour",
        }
    }
}

/// A stage stopped by the launcher's stop file (apart from `main`'s exit codes: 3 is the game
/// files' error, 9 another session running).
pub const EXIT_STOPPED: i32 = 12;
/// A stage that could not go on (a screen it waited for did not come).
pub const EXIT_FAILED: i32 = 13;

/// The character's name in the self-play's own save.
const NAME: &str = "Prepare";

/// The text-database lookup of PPSA01341 01.007.000 (`exception::text_hook`): the stages read the
/// screens by their texts, as the development drivers do.
const TEXT_HOOK: &str = "86d2ee:r13+8*:r13+10*";

/// PC settings never taken from the player's: where saves are kept (the Save Folder may be the
/// player's Documents or Saved Games) and a PS5 import asked for.
const PRIVATE_SETTINGS: [&str; 2] = ["Setting_PC_SaveFolder", "Setting_PC_ImportSaves"];

struct Setup {
    stage: Stage,
    /// The player's settings folder: its one-session lock is taken (`main::single_instance`).
    player_user: PathBuf,
    progress: Option<PathBuf>,
    /// The self-play's own title folders (the tour's automation file goes to its devlog).
    runtime: PathBuf,
    /// The player's shader store (its `prepared.txt` is written when the levels end).
    store: PathBuf,
    /// The background compute preparation's threads (`--prepare-threads`; 0: half the CPUs).
    threads: usize,
    /// The levels' part this session plays (`--prepare-chunk`: i of n; None: all).
    chunk: Option<(usize, usize)>,
    /// When the whole preparation's time is up (`--prepare-deadline`).
    deadline: Option<std::time::SystemTime>,
    /// The player's own folders (the settings copied from them, `prepare_files`).
    player_paths: dsconfig::paths::Paths,
    /// The self-play's own folders (`prepare_files` clears and fills them: never through the
    /// global `paths()`, which would be the player's had anything read it before `setup`).
    own_paths: dsconfig::paths::Paths,
}

/// The longest the last levels session waits at its end for the background compute preparation
/// (never past the deadline).
const COMPUTE_TAIL: Duration = Duration::from_secs(300);

/// A wait's time as the stages count it: it runs only while the game flips. A game frozen (a
/// shader compiled in place for seconds on a slow PC, a load from a slow disk) extends every
/// wait and sleep by as long, at most `FROZEN_MAX` a wait (the other PC's first sessions froze
/// 3-21 s at a time in the menus, 2026-11-19: fixed waits would have given up there).
struct GameClock {
    start: Instant,
    last: Instant,
    flip: u64,
    frozen: Duration,
    /// Spans in which this process did not run (the PC asleep or the process held: a poll more
    /// than `SLEPT_GAP` after the last), left out whole (the review of 2026-11-20: a sleep had
    /// counted as a wait's time and failed a part).
    slept: Duration,
}

/// A gap between two of a wait's polls (every 0.1-1 s) this long: the process did not run.
const SLEPT_GAP: Duration = Duration::from_secs(10);

/// The longest a frozen game extends one wait.
const FROZEN_MAX: Duration = Duration::from_secs(900);

impl GameClock {
    fn new() -> GameClock {
        let now = Instant::now();
        GameClock { start: now, last: now, flip: crate::hle::libs::videoout::QUEUED_FLIP.load(Relaxed), frozen: Duration::ZERO, slept: Duration::ZERO }
    }

    /// The wait's time so far, without the spans in which the game flipped nothing nor those in
    /// which the process did not run.
    fn elapsed(&mut self) -> Duration {
        let now = Instant::now();
        let f = crate::hle::libs::videoout::QUEUED_FLIP.load(Relaxed);
        let gap = now - self.last;
        if gap > SLEPT_GAP {
            self.slept += gap;
            self.flip = f;
        } else if f == self.flip {
            self.frozen += gap;
        } else {
            self.flip = f;
        }
        self.last = now;
        (now - self.start).saturating_sub(self.slept).saturating_sub(self.frozen.min(FROZEN_MAX))
    }
}

/// Spans of the stages' waits the game spent frozen, in all (the stage's summary line).
static FROZEN_TOTAL_MS: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);

/// The session ended itself: nothing had moved for `HANG_SECS` (`watch_hangs`). The launcher
/// begins the step once more, then goes on without it.
pub const EXIT_HUNG: i32 = 14;
/// The shader store's drive is full (`pcache::store_full`): the step ends, the launcher tells the
/// player and stops preparing.
pub const EXIT_DISK_FULL: i32 = 15;
/// The shader store's folder refuses writes (`pcache::store_unwritable`): the step ends, the
/// launcher tells the player and stops preparing.
pub const EXIT_UNWRITABLE: i32 = 16;
/// A levels part ended at a level's start because video memory was under pressure in the level
/// before (`images::PRESSURE_SPELLS`): the launcher begins the part again in a new session from
/// that level (`<progress>.resume`), its memory fresh, no attempt counted. Since 2026-11-21 night:
/// a part's session kept what each level left (the other PC's 12 GB card ran 2 GB past its budget
/// by its part's fifth level, its allocations refused and the part hung), and how many levels a
/// card holds depends on it and on the other programs beside the game.
pub const EXIT_CONTINUE: i32 = 17;
/// The progress file and the driver's last text for it (`Play::write_progress`).
static PROGRESS: std::sync::Mutex<Option<(PathBuf, String)>> = std::sync::Mutex::new(None);

/// The session's movement as the hang watch counts it: flips and first uses ended.
fn movement() -> (u64, u64, u64) {
    (crate::gpu::FRAMES.load(Relaxed), crate::gpu::exec::FIRST_USES[0].load(Relaxed), crate::gpu::exec::draw::GFX_FIRST_USES[0].load(Relaxed))
}

/// The progress file written whole: the driver's text, then a fourth line, the session's
/// movement (its flips and first uses ended), rewritten by the hang watch whenever it moved. The
/// launcher ends a step whose file stays the same for 15 minutes, and the driver's text alone
/// stays the same through a long step on a slow PC (the release review of 2026-11-20 night).
fn write_progress_file() {
    let g = PROGRESS.lock().unwrap_or_else(|e| e.into_inner());
    if let Some((p, t)) = g.as_ref() {
        let (f, c, gfx) = movement();
        let tmp = p.with_extension("tmp");
        if std::fs::write(&tmp, format!("{t}{}\n", f + c + gfx)).is_ok() {
            let _ = std::fs::rename(&tmp, p);
        }
    }
}

/// How long nothing may move (no flip, no pipeline made, no item of the compute list made)
/// before a self-play session counts as hung; before its first flip (the boot) `HANG_BOOT_SECS`.
const HANG_SECS: u64 = 180;
const HANG_BOOT_SECS: u64 = 240;

/// The self-play's hang watch, a thread of its own: a session in which nothing has moved for
/// `HANG_SECS` is hung (a driver call that does not return, a guest thread that waits for
/// ever): its threads' stacks go into the log and it ends with `EXIT_HUNG` at once
/// (`crate::hung`). (2026-11-20: hangs inside the driver had each left the preparation waiting
/// for the launcher's 45-minute watchdog.)
fn watch_hangs() {
    let _ = std::thread::Builder::new().name("prepare-watch".into()).spawn(|| {
        crate::log::set_thread_name("prepare-watch");
        // (The game's own progress: its flips and the first uses its command processors waited
        // for, which end one by one while a slow PC compiles in place; not the background list's
        // items, which move on beside a frozen game.)
        let moved = movement;
        let (mut last, mut since) = (moved(), Instant::now());
        let mut tick = Instant::now();
        loop {
            std::thread::sleep(Duration::from_secs(5));
            // (The PC asleep or the process held: not a hang.)
            if tick.elapsed() > SLEPT_GAP {
                since = Instant::now();
            }
            tick = Instant::now();
            if crate::gpu::pcache::store_full() {
                crate::error!("prepare-play: the shader store's drive is full ({} MiB free; {} pipeline files not written): the session ends", crate::gpu::pcache::store_free_bytes().unwrap_or(0) >> 20, crate::gpu::pcache::WRITE_FAILS.load(Relaxed));
                end(EXIT_DISK_FULL);
            }
            if crate::gpu::pcache::store_unwritable() {
                crate::error!("prepare-play: the shader store's folder refuses writes ({} pipeline files not written, none written): the session ends", crate::gpu::pcache::WRITE_FAILS.load(Relaxed));
                end(EXIT_UNWRITABLE);
            }
            let now = moved();
            if now != last {
                (last, since) = (now, Instant::now());
                write_progress_file();
                continue;
            }
            let limit = if now.0 == 0 { HANG_BOOT_SECS } else { HANG_SECS };
            if since.elapsed() >= Duration::from_secs(limit) {
                crate::hung(&format!("prepare-play: nothing moved for {limit} s (no flip after flip {}, no pipeline made, no compute item made): the session is hung", now.0), EXIT_HUNG);
            }
        }
    });
}

/// The levels' tour opens with `wait 120.0` (the title ignores level loads until it is in the
/// game, and its automation starts at boot); set once the self-play sees the game in play, the
/// next automation wait with more than a minute left ends at once (`automation_wait_hook`; on
/// this PC each part's tour began ~148 s after the game was in play, 2026-11-20).
static SKIP_FIRST_WAIT: std::sync::atomic::AtomicBool = std::sync::atomic::AtomicBool::new(false);
static WAIT_SKIPPED_S: std::sync::atomic::AtomicU32 = std::sync::atomic::AtomicU32::new(0);

/// The title's automation wait step (eboot+0xaf7bf0, title fix 'the self-play's opening wait
/// ended once in play': `vmovss xmm0,[rdi]; vsubss xmm0,xmm0,[rsi+8]`, the seconds left less the
/// frame's time, waiting while above 0): with `SKIP_FIRST_WAIT` set and more than 60 s left, the
/// seconds left become 0 (once). Inert in every other session (the flag is the self-play's).
pub extern "sysv64" fn automation_wait_hook(regs: *const u64, _arg: u32) {
    if !SKIP_FIRST_WAIT.load(Relaxed) {
        return;
    }
    let rdi = unsafe { *regs.add(7) };
    if rdi < 0x1000 || !crate::mem::readable(rdi, 4) {
        return;
    }
    let left = f32::from_bits(unsafe { std::ptr::read_volatile(rdi as *const u32) });
    if left > 60.0 && SKIP_FIRST_WAIT.swap(false, Relaxed) {
        unsafe { std::ptr::write_volatile(rdi as *mut u32, 0f32.to_bits()) };
        WAIT_SKIPPED_S.store(left as u32, Relaxed);
    }
}

/// Whether the preparation's deadline has passed.
fn past_deadline() -> bool {
    SETUP.get().and_then(|s| s.deadline).is_some_and(|d| std::time::SystemTime::now() >= d)
}

/// The time left before the deadline (None: no deadline).
fn time_left() -> Option<Duration> {
    let d = SETUP.get()?.deadline?;
    Some(d.duration_since(std::time::SystemTime::now()).unwrap_or(Duration::ZERO))
}

static SETUP: OnceLock<Setup> = OnceLock::new();

/// The background compute preparation was started (`compute_tail` waits for it only then).
static BACKGROUND_ON: std::sync::atomic::AtomicBool = std::sync::atomic::AtomicBool::new(false);

/// Levels were left: this part's deadline ended it before its last level, or it began after
/// the deadline (`prepared.txt`'s `levels_cut = 1`, the launcher's note).
static LEVELS_CUT: std::sync::atomic::AtomicBool = std::sync::atomic::AtomicBool::new(false);

/// The store marked prepared (`compbench::write_marker`, with `levels_cut` when levels were
/// left), once the device exists (none: nothing written, the launcher marks it).
fn mark_prepared(store: &Path, seconds: f64) {
    if crate::gpu::vk::try_ctx().is_none() {
        crate::warn!("prepare-play: no graphics device: the store is not marked here");
        return;
    }
    crate::gpu::compbench::background_stop();
    let b = &crate::gpu::compbench::BACKGROUND;
    let marker = store.join("prepared.txt");
    crate::gpu::compbench::write_marker(crate::game::source(), &marker, b[1].load(Relaxed) as usize, b[0].load(Relaxed) as usize, b[2].load(Relaxed) as usize, seconds);
    if LEVELS_CUT.load(Relaxed) {
        use std::io::Write;
        if let Ok(mut f) = std::fs::OpenOptions::new().append(true).open(&marker) {
            let _ = writeln!(f, "levels_cut = 1");
        }
    }
}

/// The player's settings folder, whose one-session lock a self-play run takes.
pub fn player_user() -> Option<&'static Path> {
    SETUP.get().map(|s| s.player_user.as_path())
}

/// Sets a self-play run up (`main`, before any thread starts): its own folders under the
/// player's cache (`prepare`), the player's shader store, no window, no sound, no system input,
/// the text hook its stages read the screens by. Its log stays in the player's logs folder
/// (`<time>-prepare-<stage>`).
#[cfg(windows)]
pub fn setup(stage: Stage, cfg: &mut crate::Config) {
    let player = cfg.paths.clone();
    let store = cfg.cache.clone().unwrap_or_else(|| player.shaders());
    let dir = player.cache.join("prepare");
    let user = dir.join("user");
    let paths = dsconfig::paths::Paths { layout: player.layout, exe_dir: player.exe_dir.clone(), user, cache: dir, root: player.root.clone() };
    if let Some(name) = cfg.run_dir.file_name().map(|n| n.to_string_lossy().into_owned()) {
        cfg.run_dir = cfg.run_dir.with_file_name(format!("{name}-prepare-{}", stage.name()));
    }
    // SAFETY: still the only thread (`main` calls this right after `parse_args`).
    unsafe {
        crate::set_unless_given("DS_TEXT_HOOK", TEXT_HOOK);
        std::env::set_var("DS_IME_TEXT", NAME);
    }
    crate::audio::SILENT.store(true, Relaxed);
    crate::input::SYSTEM_OFF.store(true, Relaxed);
    // The self-play's own pacing, whatever the player's Frame Rate Limit (their PC settings are
    // copied: at 30 FPS the levels' frame-counted waits took twice their time, Uncapped would
    // shorten them under what the script asks for). The new game: the title's 59.94 frames a
    // second at most, a frame shown once it is ready (off the vblank grid: frames of 17-33 ms
    // had waited for the second vblank). The levels: a fast-forward (`fast_forward`).
    // SAFETY: still the only thread.
    unsafe {
        // (No hidden present either, `present::Hidden`: presents to a window never shown blocked
        // in the driver three times on 2026-11-20 and took every other driver call with them;
        // the GPU's clock may run lower, the self-play a little slower.)
        // (`set_unless_given`: a player's diagnostics.txt does not override them.)
        for (k, v) in [("DS_FRAME_CAP", "0"), ("DS_UNCAPPED", "0"), ("DS_HEADLESS_PRESENT", "0")] {
            crate::set_unless_given(k, v);
        }
        if stage == Stage::NewGame {
            crate::set_unless_given("DS_FLIP_GRID", "0");
        }
        // (The tour moves on and seldom comes back: the mirror gives a place's pages back after
        // 1,200 flips without a placement (10 s at its 120 a second) rather than 3,600, so a 12 GB
        // card's budget holds; the other PC's ran 2 GB over it, 2026-11-21.)
        if stage == Stage::Tour {
            crate::set_unless_given("DS_MIRROR_COLD", "1200");
        }
    }
    // The levels at the self-play's game speed (`level_speed`; the new game's screens at their own).
    if stage == Stage::Tour {
        crate::gamestep::SPEED_X100.store((level_speed() * 100.0).round() as u32, Relaxed);
        if fast_forward() {
            crate::gamestep::FIXED_FEED.store(true, Relaxed);
            crate::input::FLIP_TIMED.store(true, Relaxed);
        }
    }
    let runtime = paths.runtime();
    let own_paths = paths.clone();
    cfg.paths = paths;
    cfg.cache = Some(store.clone());
    cfg.headless = true;
    let threads = if cfg.prepare_threads > 0 { cfg.prepare_threads } else { std::thread::available_parallelism().map_or(4, |n| n.get() / 2).max(2) };
    let deadline = cfg.prepare_deadline.map(|s| std::time::UNIX_EPOCH + Duration::from_secs(s));
    let _ = SETUP.set(Setup { stage, player_user: player.user.clone(), player_paths: player, own_paths, progress: cfg.progress.clone(), runtime, store, threads, chunk: cfg.prepare_chunk, deadline });
}

/// Linux's CLI parent has already selected and validated an isolated user/cache root.
#[cfg(target_os = "linux")]
pub fn setup_linux(stage: Stage, paths: &dsconfig::paths::Paths, store: &Path, progress: PathBuf, threads: usize, chunk: Option<(usize, usize)>) -> Result<(), String> {
    unsafe {
        crate::set_unless_given("DS_TEXT_HOOK", TEXT_HOOK);
        std::env::set_var("DS_IME_TEXT", NAME);
        for (k, v) in [("DS_FRAME_CAP", "0"), ("DS_UNCAPPED", "0"), ("DS_HEADLESS_PRESENT", "0")] { std::env::set_var(k, v); }
        if stage == Stage::Tour { crate::set_unless_given("DS_MIRROR_COLD", "1200"); }
    }
    crate::audio::SILENT.store(true, Relaxed);
    crate::input::SYSTEM_OFF.store(true, Relaxed);
    if stage == Stage::Tour {
        crate::gamestep::SPEED_X100.store((level_speed() * 100.0).round() as u32, Relaxed);
        if fast_forward() { crate::gamestep::FIXED_FEED.store(true, Relaxed); crate::input::FLIP_TIMED.store(true, Relaxed); }
    }
    let setup = Setup { stage, player_user: paths.user.clone(), player_paths: paths.clone(), own_paths: paths.clone(),
        progress: Some(progress), runtime: paths.runtime(), store: store.to_path_buf(), threads, chunk, deadline: None };
    // Before mounts/settings/save initialization. start() only launches the shared driver later.
    prepare_files(&setup);
    SETUP.set(setup).map_err(|_| "preparation already configured".to_string())
}

/// The self-play's own files made ready (`start`: after the one-session lock, before the settings
/// and saves are read): the player's PC settings (some choose shader variants: the upscaler, the
/// Visual Mode), display choices and game folder copied, never their saves, key bindings or
/// trophies; for a new game, no saves, trophies or title folders of an earlier one.
fn prepare_files(s: &Setup) {
    let paths = &s.own_paths;
    let player = &s.player_paths;
    let user = &paths.user;
    let _ = std::fs::create_dir_all(user);
    if player.user != paths.user {
        match std::fs::read_to_string(player.settings_file()) {
            Ok(t) => {
                let kept: String = t.lines().filter(|l| !PRIVATE_SETTINGS.iter().any(|k| l.trim_start().starts_with(k))).map(|l| format!("{l}\n")).collect();
                let _ = std::fs::write(user.join("pcsettings.txt"), kept);
            }
            Err(_) => {
                let _ = std::fs::remove_file(user.join("pcsettings.txt"));
            }
        }
        for f in ["display.txt", "game.txt"] {
            let src = player.user.join(f);
            if src.is_file() {
                let _ = std::fs::copy(&src, user.join(f));
            } else {
                let _ = std::fs::remove_file(user.join(f));
            }
        }
    }
    if s.stage == Stage::NewGame {
        for d in [paths.saves(), paths.save_backups(), paths.runtime(), newgame_saves(paths), newgame_saves(paths).with_extension("part")] {
            let _ = std::fs::remove_dir_all(d);
        }
        for f in [paths.save_folder_record(), paths.achievements_file()] {
            let _ = std::fs::remove_file(f);
        }
    }
    if s.stage == Stage::Tour {
        newgame_save_for_part(paths);
    }
}

/// The new game's save as its step left it (`newgame_save_for_part`).
fn newgame_saves(paths: &dsconfig::paths::Paths) -> std::path::PathBuf {
    paths.user.join("saves-newgame")
}

/// Each part of the levels starts from the new game's save: the first part to start keeps a copy
/// of it (written whole, then named), and every part after it, or one played again, starts from
/// that copy. A save damaged in a part (a write cut short by a power loss) then stops no other
/// part. (Until 2026-11-20 each part went on from the save the part before it left.)
fn newgame_save_for_part(paths: &dsconfig::paths::Paths) {
    let (saves, kept) = (paths.saves(), newgame_saves(paths));
    if !kept.is_dir() {
        let part = kept.with_extension("part");
        let _ = std::fs::remove_dir_all(&part);
        match copy_synced(&saves, &part).and_then(|n| std::fs::rename(&part, &kept).map(|_| n)) {
            Ok(n) => crate::info!("prepare-play: the new game's save kept for the levels' parts ({n} files)"),
            Err(e) => crate::warn!("prepare-play: the new game's save not kept ({e}): this part plays the save as it is"),
        }
        return;
    }
    let _ = std::fs::remove_dir_all(&saves);
    match copy_synced(&kept, &saves) {
        Ok(n) => crate::info!("prepare-play: this part starts from the new game's save ({n} files)"),
        Err(e) => crate::warn!("prepare-play: the new game's save could not be put in place ({e})"),
    }
}

/// Copies folder `from` into `to` (made), each file flushed to the drive: the files copied.
fn copy_synced(from: &Path, to: &Path) -> std::io::Result<u64> {
    std::fs::create_dir_all(to)?;
    let mut n = 0;
    for e in std::fs::read_dir(from)? {
        let e = e?;
        let target = to.join(e.file_name());
        if e.file_type()?.is_dir() {
            n += copy_synced(&e.path(), &target)?;
        } else {
            std::fs::copy(e.path(), &target)?;
            std::fs::OpenOptions::new().write(true).open(&target)?.sync_all()?;
            n += 1;
        }
    }
    Ok(n)
}

/// Starts the stage (`main`, once the log is open; before the title starts: the tour's
/// automation is read at its boot): its driver thread plays it and ends the session.
pub fn start() {
    let Some(s) = SETUP.get() else { return };
    #[cfg(windows)]
    prepare_files(s);
    crate::log::tap_on();
    crate::info!(
        "prepare-play: stage {} (the shader preparation's self-play: no window, no sound, the system's controllers, keyboard and mouse left alone; its settings, saves and title folders in {})",
        s.stage.name(),
        crate::paths().cache.display()
    );
    // A part of the levels begun after the preparation's time is up: the store is marked
    // prepared as it is (once the device exists), and the session ends.
    if s.stage == Stage::Tour && past_deadline() {
        let store = s.store.clone();
        let _ = std::thread::Builder::new().name("prepare-play".into()).spawn(move || {
            crate::log::set_thread_name("prepare-play");
            let t = Instant::now();
            while crate::gpu::vk::try_ctx().is_none() && t.elapsed() < Duration::from_secs(120) {
                std::thread::sleep(Duration::from_millis(100));
            }
            crate::info!("prepare-play: the preparation's time was up before this part of the levels: the store is marked prepared");
            LEVELS_CUT.store(true, Relaxed);
            mark_prepared(&store, 0.0);
            end(0);
        });
        return;
    }
    let devlog = s.runtime.join("devlog");
    let _ = std::fs::create_dir_all(&devlog);
    let args = devlog.join("commandlineargs.txt");
    let total = match s.stage {
        Stage::NewGame => {
            let _ = std::fs::remove_file(&args);
            NEWGAME_STEPS
        }
        Stage::Tour => match tour_script(crate::game::source()) {
            Ok((text, views)) => {
                // (No '-' in the name: `automate_loadfile=#/prepare-tour.txt` failed to execute,
                // its usage printed, 2026-11-19.)
                let written = std::fs::write(devlog.join("preparetour.txt"), text).and_then(|_| std::fs::write(&args, "+automate_loadfile=#/preparetour.txt +automate_go\n"));
                if let Err(e) = written {
                    crate::error!("prepare-play: writing the tour into {}: {e}", devlog.display());
                    end(EXIT_FAILED);
                }
                crate::info!("prepare-play: the levels stage has {views} steps");
                views + TOUR_STEPS
            }
            Err(e) => {
                crate::error!("prepare-play: the tour: {e}");
                end(EXIT_FAILED);
            }
        },
    };
    // The compute programs' guessed variants (the launcher's former first step, `--precompile-all`)
    // made beside the play on threads of the lowest priority: the CPU the play leaves idle
    // (`DS_PREPARE_BACKGROUND=0`: none, a test).
    if std::env::var("DS_PREPARE_BACKGROUND").ok().as_deref() != Some("0") {
        // (The new game's waits run in real time: its list at full speed meanwhile, the
        // launcher's compute step after it then has that much less to make. `DS_PREPARE_NG_FULL=0`:
        // at the lowest priority as in the levels, a test.)
        if s.stage == Stage::NewGame && std::env::var("DS_PREPARE_NG_FULL").ok().as_deref() != Some("0") {
            crate::gpu::compbench::full_from_start();
        }
        crate::gpu::compbench::background(crate::game::source().clone(), s.threads);
        BACKGROUND_ON.store(true, Relaxed);
    }
    let (stage, progress, store) = (s.stage, s.progress.clone(), s.store.clone());
    watch_hangs();
    let spawned = std::thread::Builder::new().name("prepare-play".into()).spawn(move || {
        crate::log::set_thread_name("prepare-play");
        let mut p = Play::new(progress, total);
        let r = match stage {
            Stage::NewGame => p.newgame(),
            Stage::Tour => p.tour(),
        };
        let code = match r {
            Ok(()) => {
                // (Past the deadline the queued pipelines get less time.)
                p.drain(if past_deadline() { 30 } else { 180 });
                // The last part of the levels, or one the deadline ended (no part follows): the
                // background compute preparation's rest, for at most `COMPUTE_TAIL` and never past
                // the deadline; then the store is marked prepared (with how much of the list was
                // made).
                let last = SETUP.get().and_then(|s| s.chunk).is_none_or(|(i, n)| i == n) || past_deadline();
                if stage == Stage::Tour && last {
                    // (A stop during the tail leaves the levels played: status 0 all the same.)
                    if p.compute_tail().is_err() {
                        crate::info!("prepare-play: stopped during the compute tail (the levels were played)");
                    }
                    mark_prepared(&store, p.t0.elapsed().as_secs_f64());
                }
                p.done();
                crate::info!("prepare-play: stage {} played to its end in {:.0} s", stage.name(), p.t0.elapsed().as_secs_f64());
                0
            }
            Err(End::Stopped) => {
                crate::info!("prepare-play: stopped (the launcher's stop file) after {:.0} s", p.t0.elapsed().as_secs_f64());
                EXIT_STOPPED
            }
            Err(End::Continue) => {
                // (The pipelines queued for the background made first, briefly: the next session
                // learns again what is left of the level it goes on from.)
                p.drain(60);
                crate::info!("prepare-play: this session ends at a level's start, video memory under pressure in the level before; the part goes on in a new session from it ({:.0} s)", p.t0.elapsed().as_secs_f64());
                EXIT_CONTINUE
            }
            Err(End::Failed(why)) => {
                crate::error!("prepare-play: stage {} could not go on: {why} (after {:.0} s)", stage.name(), p.t0.elapsed().as_secs_f64());
                p.drain(60);
                let last = SETUP.get().and_then(|s| s.chunk).is_none_or(|(i, n)| i == n) || past_deadline();
                if stage == Stage::Tour && last {
                    mark_prepared(&store, p.t0.elapsed().as_secs_f64());
                }
                EXIT_FAILED
            }
        };
        end(code);
    });
    if let Err(e) = spawned {
        crate::error!("prepare-play: no driver thread ({e})");
        end(EXIT_FAILED);
    }
}

/// A finished wait's frozen spans into the total (the stage's summary line).
fn note_frozen(c: &GameClock) {
    FROZEN_TOTAL_MS.fetch_add(c.frozen.as_millis() as u64, Relaxed);
}

/// The stage reached its own end (`end`): its health line stands, whatever its teardown meets
/// after it (`health_if_playing`).
static ENDED: std::sync::atomic::AtomicBool = std::sync::atomic::AtomicBool::new(false);

fn end(code: i32) -> ! {
    crate::info!("prepare-play: the game was frozen for {:.0} s of the stage's waits in all (their time ran on)", FROZEN_TOTAL_MS.load(Relaxed) as f64 / 1000.0);
    health_if_playing(code);
    ENDED.store(true, Relaxed);
    crate::shutdown(code)
}

/// A preparation step's health in one line (`prepare-health:`, `key=value` pairs), the signals its
/// tests hold to their limits (tools/prephealth.py): first uses compiled in place (in the levels
/// they should be few once the compute step has made the list), the game's frozen time, video
/// memory's peak against its budget, failed allocations (`DS_SIM_VRAM_MIB`'s among them),
/// pipelines left out for crashing the driver; since 2026-11-22 what video memory running out
/// cost the picture (mirror placements bound as zeros, images not made) and the most images in
/// system memory at once (`vk::SYSTEM_PEAK`). Until 2026-11-21 night the tests judged a step by
/// its status alone, and a preparation that froze the game for an hour on first uses and ran past
/// its video memory still passed (the other PC).
pub fn health_line(stage: &str, code: i32) -> String {
    use crate::gpu::exec::{FIRST_USES, draw};
    let secs = crate::log::elapsed_ms() / 1000.0;
    format!(
        "prepare-health: stage={stage}; status={code}; secs={secs:.0}; compute_first_uses={}; compute_first_use_s={:.0}; gfx_first_uses={}; gfx_first_use_s={:.0}; frozen_s={:.0}; vram_peak_pct={}; vram_budget_mib={}; alloc_fails={}; sim_alloc_fails={}; alloc_refused={}; vram_pressure={}; zero_placements={}; images_not_made={}; system_images_peak_mib={}; crash_skips={}; background_made={}; background_items={}",
        FIRST_USES[0].load(Relaxed),
        FIRST_USES[1].load(Relaxed) as f64 / 1e9,
        draw::GFX_FIRST_USES[0].load(Relaxed),
        draw::GFX_FIRST_USES[1].load(Relaxed) as f64 / 1e9,
        FROZEN_TOTAL_MS.load(Relaxed) as f64 / 1000.0,
        crate::gpu::images::VRAM_PEAK[0].load(Relaxed) / 10,
        crate::gpu::images::VRAM_PEAK[1].load(Relaxed),
        crate::gpu::vk::GATE_STATS[1].load(Relaxed),
        crate::gpu::vk::SIM_VRAM_FAILS.load(Relaxed),
        crate::gpu::vk::GATE_STATS[0].load(Relaxed),
        crate::gpu::images::PRESSURE_SPELLS.load(Relaxed),
        crate::gpu::images::ZERO_PLACEMENTS.load(Relaxed),
        crate::gpu::images::IMAGE_FAILS.load(Relaxed),
        crate::gpu::vk::SYSTEM_PEAK.load(Relaxed) >> 20,
        crate::gpu::pcache::CRASH_SKIPS.load(Relaxed),
        crate::gpu::compbench::BACKGROUND[1].load(Relaxed),
        crate::gpu::compbench::BACKGROUND[0].load(Relaxed),
    )
}

/// The levels this part's level list dropped for a resume (`resume_from`): level indices in the
/// resume record count from the part's first level.
static RESUME_BASE: std::sync::atomic::AtomicU32 = std::sync::atomic::AtomicU32::new(0);

/// The resume record beside the progress file (`<progress>.resume`): the part (i/n) and the index
/// of the level its session started last, from the part's first. The launcher keeps it for an
/// attempt begun again after a crash or a hang (moving it one level on after a hang) and drops it
/// when a step begins anew.
fn resume_file() -> Option<PathBuf> {
    SETUP.get().and_then(|s| s.progress.as_ref()).map(|p| p.with_extension("resume"))
}

/// The level a part begun again goes on from (0: its first).
fn resume_from(chunk: (usize, usize)) -> usize {
    let Some(t) = resume_file().and_then(|f| std::fs::read_to_string(f).ok()) else { return 0 };
    let get = |k: &str| t.lines().find_map(|l| l.trim().strip_prefix(k).map(str::to_string));
    if get("chunk=").as_deref() != Some(&format!("{}/{}", chunk.0, chunk.1)) {
        return 0;
    }
    get("level=").and_then(|v| v.parse().ok()).unwrap_or(0)
}

/// A level of this part started (its index from the part's first): the resume record.
fn note_resume(level: usize) {
    let (Some(f), Some((i, n))) = (resume_file(), SETUP.get().and_then(|s| s.chunk)) else { return };
    let text = format!("chunk={i}/{n}\nlevel={level}\n");
    #[cfg(windows)]
    let _ = std::fs::write(f, text);
    #[cfg(target_os = "linux")]
    let _ = crate::platform::prepare::atomic_write(&f, text.as_bytes());
}

/// A self-play stage's health line when it ends otherwise than through `end` (found hung). Not
/// after `end`: a stage played to its end whose teardown then hung keeps its line (status 0), so
/// the launcher takes it as done; the review of 2026-11-21 night: the hang's line had replaced it,
/// and a new game played to its end was played again, twice.
pub fn health_if_playing(code: i32) {
    if ENDED.load(Relaxed) {
        return;
    }
    if let Some(s) = SETUP.get() {
        let name = match (s.stage, s.chunk) {
            (Stage::Tour, Some((i, n))) => format!("tour {i}/{n}"),
            (st, _) => st.name().to_string(),
        };
        write_health(s.progress.as_deref(), &health_line(&name, code));
    }
}

/// The health line into the log and beside the step's progress file (`<progress>.health`, which
/// the launcher copies into its own log at the step's end).
pub fn write_health(progress: Option<&std::path::Path>, line: &str) {
    crate::info!("{line}");
    if let Some(p) = progress {
        let _ = std::fs::write(p.with_extension("health"), format!("{line}\n"));
    }
}

enum End {
    Stopped,
    /// A levels part's session ends at a level's start for video memory (`EXIT_CONTINUE`).
    Continue,
    Failed(String),
}

/// The log's lines as the stages read them (`log::tap_read`), the last `KEEP`.
struct Watch {
    lines: VecDeque<String>,
    /// The index of `lines[0]` among all lines read.
    first: usize,
    cursor: u64,
}

const KEEP: usize = 200_000;

impl Watch {
    fn poll(&mut self) {
        let (new, c) = crate::log::tap_read(self.cursor);
        self.cursor = c;
        self.lines.extend(new);
        while self.lines.len() > KEEP {
            self.lines.pop_front();
            self.first += 1;
        }
    }

    /// The index of the next line.
    fn mark(&mut self) -> usize {
        self.poll();
        self.first + self.lines.len()
    }

    fn since(&self, at: usize) -> impl Iterator<Item = &String> {
        self.lines.iter().skip(at.saturating_sub(self.first))
    }

    fn found(&mut self, pat: &str, at: usize) -> bool {
        self.poll();
        self.since(at).any(|l| matches(l, pat))
    }

    fn count(&mut self, pat: &str, at: usize) -> usize {
        self.poll();
        self.since(at).filter(|l| matches(l, pat)).count()
    }

    /// Which of `pats` the last line from `at` on that matches one of them matches.
    fn last_of(&mut self, pats: &[&str], at: usize) -> Option<usize> {
        self.poll();
        let mut last = None;
        for l in self.since(at) {
            if let Some(i) = pats.iter().position(|p| matches(l, p)) {
                last = Some(i);
            }
        }
        last
    }
}

/// Character creation's entries below the name, by the help text the one under the cursor shows
/// (the title logs each text it sets): class, gift, saved creations, foundation, appearance,
/// Finalise.
const CREATION_ENTRIES: [&str; 6] = [
    "text: Select a character class.",
    "text: Choose a starting gift.",
    "text: Save the current character creation or load a character creation.",
    "text: Define the foundation of the character.",
    "text: Change the overall appearance of the character.",
    "text: Finalise your character.",
];

/// `pat`'s parts (split at `.*`) in `line`, in order.
fn matches(line: &str, pat: &str) -> bool {
    let mut at = 0;
    for part in pat.split(".*") {
        match line[at..].find(part) {
            Some(i) => at += i + part.len(),
            None => return false,
        }
    }
    true
}

struct Play {
    w: Watch,
    progress: Option<PathBuf>,
    stop: Option<PathBuf>,
    done_steps: u32,
    total: u32,
    t0: Instant,
    /// The compute list's rest is being made (`compute_tail`): the progress file's second line
    /// (the launcher's bar) is the list's items made and listed.
    tail: bool,
    /// Linux relay uses defaults and captures every transition.
    relay: bool,
    /// No readback requests inside timed intervals.
    measuring: bool,
}

/// The new game's steps (progress): the title, character creation, its menus, the character
/// made, the cinematic, in the game, the walk, the menu.
const NEWGAME_STEPS: u32 = 8;
/// The tour's steps besides its views: the title, in the game, the tour started.
const TOUR_STEPS: u32 = 3;

impl Play {
    fn new(progress: Option<PathBuf>, total: u32) -> Play {
        // (The launcher's stop file, as the precompile reads it.)
        let stop = progress.as_ref().map(|p| p.with_extension("stop"));
        let p = Play { w: Watch { lines: VecDeque::new(), first: 0, cursor: 0 }, progress, stop, done_steps: 0, total: total.max(1), t0: Instant::now(), tail: false, relay: false, measuring: false };
        p.write_progress();
        p
    }

    fn write_progress(&self) {
        if let Some(p) = &self.progress {
            // (A third line: the background compute preparation's items made and listed.)
            let b = &crate::gpu::compbench::BACKGROUND;
            let (made, all) = (b[1].load(Relaxed), b[0].load(Relaxed));
            let bar = if self.tail && all > 0 { (made, all) } else { (self.done_steps as u64, self.total as u64) };
            let t = format!("{} {}\n{} {}\n{made} {all}\n", self.done_steps, self.total, bar.0, bar.1);
            *PROGRESS.lock().unwrap_or_else(|e| e.into_inner()) = Some((p.clone(), t));
            write_progress_file();
        }
    }

    /// A step done (`what`: what comes next).
    fn step(&mut self, what: &str) {
        if self.relay { self.shot(what); }
        self.done_steps = (self.done_steps + 1).min(self.total);
        self.write_progress();
        crate::info!("prepare-play: {} of {} ({what}) at {:.0} s", self.done_steps, self.total, self.t0.elapsed().as_secs_f64());
    }

    fn done(&mut self) {
        self.done_steps = self.total;
        self.write_progress();
    }

    fn check_stop(&self) -> Result<(), End> {
        if self.stop.as_ref().is_some_and(|f| f.exists()) {
            return Err(End::Stopped);
        }
        Ok(())
    }

    /// Sleeps `secs` of the game's time (`GameClock`: a frozen game's spans not counted).
    fn sleep(&mut self, secs: f64) -> Result<(), End> {
        let want = Duration::from_secs_f64(secs);
        let mut c = GameClock::new();
        loop {
            self.check_stop()?;
            let e = c.elapsed();
            if e >= want {
                note_frozen(&c);
                return Ok(());
            }
            std::thread::sleep((want - e).min(Duration::from_millis(100)));
        }
    }

    /// Waits up to `secs` for a line matching `pat` from line `at` on.
    fn wait_for(&mut self, pat: &str, at: usize, secs: f64) -> Result<bool, End> {
        let want = Duration::from_secs_f64(secs);
        let mut c = GameClock::new();
        loop {
            if self.w.found(pat, at) {
                note_frozen(&c);
                return Ok(true);
            }
            self.check_stop()?;
            if c.elapsed() >= want {
                note_frozen(&c);
                return Ok(false);
            }
            std::thread::sleep(Duration::from_millis(250));
        }
    }

    fn shot(&self, what: &str) {
        let tag: String = what.chars().map(|c| if c.is_ascii_alphanumeric() { c } else { '-' }).collect();
        // Measurement/timeline boundaries are markers only: no GPU readback in either interval.
        if !self.measuring && !what.starts_with("gameplay-measure-") && !what.starts_with("gameplay-timeline-") && !what.starts_with("movement-measure-") {
            *crate::hle::libs::videoout::SHOT_REQUEST.lock() = Some(format!("step-{:04}-{}", self.t0.elapsed().as_secs(), tag));
        }
        crate::info!("relay step: {what}; {}", crate::uistate::report());
    }

    fn press(&self, names: &str, ms: u64) {
        if self.relay { self.shot(&format!("press-{names}-{ms}ms")); }
        if let Err(e) = crate::input::press_for(names, Duration::from_millis(ms)) {
            crate::warn!("prepare-play: press {names}: {e}");
        }
    }

    /// A press held for `flips` of the game's frames, whatever their time (`input::press_flips`).
    fn press_flips(&self, names: &str, flips: u64) {
        if self.relay { self.shot(&format!("press-{names}-{flips}flips")); }
        if let Err(e) = crate::input::press_flips(names, flips) {
            crate::warn!("prepare-play: press {names}: {e}");
        }
    }

    /// Places character creation's cursor on entry `want` (`CREATION_ENTRIES`; `from`: the entry
    /// it is on, None above them), a press at a time, by the help text the entry under the cursor
    /// shows: an entry too far is undone with a press up, a press not taken is made again
    /// (2026-11-21: after a 462 ms frame the menu took one press down as two, the gift and the
    /// saved creations 274 ms apart, every press after it landed an entry off and the character
    /// was not saved). Directions are held for two of the game's frames, whatever their time:
    /// a long frame no longer holds them long enough for the menu's own repeat.
    fn to_entry(&mut self, want: usize, from: Option<usize>) -> Result<bool, End> {
        let mut at = from;
        for _ in 0..8 {
            let dir = match at {
                Some(i) if i == want => return Ok(true),
                Some(i) if i > want => "up",
                _ => "down",
            };
            let m = self.w.mark();
            self.press_flips(dir, 2);
            // (The entry's help text; a second move's too, if the menu took the press twice.)
            if self.wait_any(&CREATION_ENTRIES, m, 5.0)? {
                self.sleep(1.0)?;
                at = self.w.last_of(&CREATION_ENTRIES, m);
            }
        }
        Ok(at == Some(want))
    }

    /// Cross until a line matching `pat` comes (at most `tries` presses, 6 s apart).
    fn cross_until(&mut self, pat: &str, at: usize, tries: u32) -> Result<bool, End> {
        for _ in 0..tries {
            if self.w.found(pat, at) {
                return Ok(true);
            }
            self.press("cross", 300);
            if self.wait_for(pat, at, 6.0)? {
                return Ok(true);
            }
        }
        Ok(false)
    }

    /// In the game: HUD scenes up and no menu open (`uistate`), for two seconds running, or
    /// `hud_text` logged since `at`.
    fn wait_in_game(&mut self, at: usize, hud_text: &str, secs: f64) -> Result<bool, End> {
        let want = Duration::from_secs_f64(secs);
        let mut c = GameClock::new();
        let mut since: Option<Instant> = None;
        loop {
            if self.w.found(hud_text, at) {
                return Ok(true);
            }
            let up = crate::uistate::hud_count().is_some_and(|n| n > 0) && crate::uistate::menu_depth() == Some(0);
            match (up, since) {
                (true, None) => since = Some(Instant::now()),
                (true, Some(t)) if t.elapsed() >= Duration::from_secs(2) => return Ok(true),
                (false, _) => since = None,
                _ => {}
            }
            self.check_stop()?;
            if c.elapsed() >= want {
                note_frozen(&c);
                return Ok(false);
            }
            std::thread::sleep(Duration::from_millis(250));
        }
    }

    /// The pipelines queued for the background graphics workers made (at most `secs`): they are
    /// kept as they are made, and this session ends after them.
    /// (`secs`: at least that long a wait; past it the wait goes on while a pipeline is made at
    /// least every 5 minutes: a slower PC's queue made whole.)
    fn drain(&mut self, secs: u64) {
        let s = &crate::gpu::exec::draw::ASYNC_GFX_STATS;
        // (The workers' own state: their statistics leave out jobs dropped as made meanwhile, and
        // the arithmetic of them never reached 0, the whole limit waited at every end.)
        let idle = crate::gpu::exec::draw::async_gfx_idle;
        let mut t = Instant::now();
        let (mut made, mut moved, mut shown) = (s[3].load(Relaxed), Instant::now(), Instant::now());
        let mut tick = Instant::now();
        while !idle() && self.check_stop().is_ok() {
            // (A gap of over 10 s: the PC slept or the process was held; the waits start again,
            // as `watch_hangs`'s do. The review of 2026-11-21 night: a sleep had ended the drain
            // with pipelines not made, and the store was marked prepared.)
            let now = Instant::now();
            if now.duration_since(tick) > Duration::from_secs(10) {
                (t, moved) = (now, now);
            }
            tick = now;
            let m = s[3].load(Relaxed);
            if m != made {
                (made, moved) = (m, Instant::now());
            }
            if t.elapsed() >= Duration::from_secs(secs) && moved.elapsed() >= Duration::from_secs(300) {
                break;
            }
            if shown.elapsed() >= Duration::from_secs(5) {
                self.write_progress();
                shown = Instant::now();
            }
            std::thread::sleep(Duration::from_millis(250));
        }
        crate::info!("prepare-play: {} graphics pipelines queued, {} made, {} failed; the workers {} ({:.0} s waited)", s[1].load(Relaxed), s[3].load(Relaxed), s[7].load(Relaxed), if idle() { "idle" } else { "still busy" }, t.elapsed().as_secs_f64());
    }

    /// Waits up to `secs` for a line matching any of `pats` from line `at` on.
    fn wait_any(&mut self, pats: &[&str], at: usize, secs: f64) -> Result<bool, End> {
        let want = Duration::from_secs_f64(secs);
        let mut c = GameClock::new();
        loop {
            if pats.iter().any(|p| self.w.found(p, at)) {
                note_frozen(&c);
                return Ok(true);
            }
            self.check_stop()?;
            if c.elapsed() >= want {
                note_frozen(&c);
                return Ok(false);
            }
            std::thread::sleep(Duration::from_millis(250));
        }
    }

    /// The title's boot to its title screen ("PRESS ANY BUTTON"), through its attract movie and
    /// first-run screens (Cross held on each, as the development drivers do), then to the main
    /// menu.
    fn title(&mut self) -> Result<(), End> {
        if !self.wait_any(&["attract_movie.bk2", "text: "], 0, 300.0)? {
            return Err(End::Failed("the title did not start (no attract movie, no text)".into()));
        }
        self.sleep(2.0)?;
        let m = self.w.mark();
        let mut ok = false;
        for _ in 0..10 {
            if self.w.found("PRESS ANY BUTTON", m) || self.w.found("main_menu.bk2", m) {
                ok = true;
                break;
            }
            self.press("cross", 4000);
            if self.wait_any(&["PRESS ANY BUTTON", "main_menu.bk2"], m, 8.0)? {
                ok = true;
                break;
            }
        }
        if !ok {
            return Err(End::Failed("no title screen".into()));
        }
        self.sleep(2.0)?;
        if !self.cross_until("main_menu.bk2", m, 6)? {
            return Err(End::Failed("no main menu".into()));
        }
        self.sleep(2.0)
    }

    fn newgame(&mut self) -> Result<(), End> {
        self.title()?;
        self.step("the main menu: a new game");
        // NEW GAME is the first entry without saves; the offline notice; character creation.
        let m = self.w.mark();
        if !self.cross_until("Enter your player's name", m, 8)? {
            return Err(End::Failed("no character creation".into()));
        }
        self.step("character creation");
        // (Its save is looked for from here: 2026-11-21 the cursor reached Finalise an entry early
        // once, the character was saved 10 s before the Finalise press below, and the wait after
        // that press found no save.)
        let created_from = self.w.mark();
        self.sleep(3.0)?;
        let name = if self.relay { "Regaliant" } else { NAME };
        let name_event = format!("ime dialog: entered '{name}'");
        if (self.relay || cfg!(target_os = "linux")) && !self.w.found(&name_event, 0) {
            let m = self.w.mark();
            self.press("cross", 300);
            if !self.wait_for(&name_event, m, 12.0)? {
                return Err(End::Failed("character name was not accepted by IME".into()));
            }
            self.sleep(2.0)?;
        }
        // The entries below the name: class, gift, saved creations, foundation, appearance,
        // Finalise (`CREATION_ENTRIES`), the cursor placed on each by its help text (`to_entry`).
        // Each but the saved creations is opened (Cross), changed (Right) and left (Circle): their
        // previews. (Cross on an empty saved-creation slot opens nothing, and Circle there would
        // open the quit prompt: that row is only passed.) An entry the cursor could not be placed
        // on is passed too.
        let mut at = None;
        for (i, e) in ["class", "gift", "saveload", "foundation", "appearance"].into_iter().enumerate() {
            if !self.to_entry(i, at)? {
                crate::warn!("prepare-play: the cursor could not be placed on character creation's {e}");
                at = self.w.last_of(&CREATION_ENTRIES, created_from);
                continue;
            }
            at = Some(i);
            self.sleep(1.0)?;
            if self.relay || e == "saveload" {
                continue;
            }
            for b in ["cross", "right", "circle"] {
                self.press(b, 300);
                self.sleep(3.0)?;
            }
        }
        if !self.to_entry(5, at)? {
            crate::warn!("prepare-play: the cursor may not be on Finalise");
        }
        if self.relay { crate::gpu::nan_probe::early_phase("before character finalise / level loading"); }
        self.step("the character made");
        let m = self.w.mark();
        self.press("cross", 300);
        if !self.wait_for("committed [\"SAVEDATA0PlayerProfile0\"]", m, 60.0)? {
            if !self.w.found("committed [\"SAVEDATA0PlayerProfile0\"]", created_from) {
                return Err(End::Failed("the character was not saved".into()));
            }
            crate::warn!("prepare-play: the character was saved before the Finalise press (the cursor reached it early): going on");
        }
        // The opening cinematic, skipped (a prerecorded movie: its few pipelines are the attract
        // movie's; ~2.5 minutes).
        if self.wait_for("create thread 'Bink IO'", m, 60.0)? {
            self.step("the opening cinematic skipped");
            self.sleep(3.0)?;
            self.press("cross", 4000);
        }
        if !self.wait_in_game(m, "Crescent Moon Grass", 420.0)? {
            return Err(End::Failed("not in the game after the cinematic".into()));
        }
        if self.relay { crate::gpu::nan_probe::early_phase("first gameplay detected"); }
        self.step("in the game: the tutorial's first steps");
        self.sleep(8.0)?;
        for (what, secs) in [("ls_up", 6.0), ("rs_right", 3.0), ("ls_up", 6.0), ("rs_left", 5.0), ("ls_up", 6.0), ("rs_right", 6.0), ("ls_up", 4.0)] {
            self.press(what, (secs * 1000.0) as u64);
            self.sleep(secs + 0.7)?;
        }
        if self.relay {
            for b in ["r1", "r1", "r2"] { self.press(b, 300); self.sleep(2.0)?; }
        }
        if self.relay && std::env::var("DS_RELAY_TARGETS").ok().as_deref() == Some("1") {
            // A frame at a known gameplay step, rather than a boot-dependent flip number.
            // This intentionally stalls for readbacks. Exclude it from the performance window.
            self.shot("gameplay-target-capture-begin");
            let before = crate::gpu::capture::completed_flip();
            crate::gpu::capture::arm_next();
            let start = Instant::now();
            while crate::gpu::capture::completed_flip() == before && start.elapsed() < Duration::from_secs(300) {
                self.check_stop()?;
                std::thread::sleep(Duration::from_millis(250));
            }
            if crate::gpu::capture::completed_flip() == before {
                return Err(End::Failed("gameplay target capture did not finish in 300 seconds".into()));
            }
            self.shot("gameplay-target-capture-end");
        }
        self.measure_gameplay()?;
        self.step("the in-game menu");
        self.menu()?;
        self.sleep(4.0)?;
        Ok(())
    }

    fn measure_gameplay(&mut self) -> Result<(), End> {
        if self.relay {
            let seconds = std::env::var("DS_RELAY_GAMEPLAY_SECONDS").ok().and_then(|s| s.parse::<u64>().ok()).unwrap_or(0).min(180);
            if seconds > 0 {
                self.sleep(5.0)?;
                #[cfg(target_os = "linux")]
                { self.relay_settle()?; self.relay_capture_barrier()?; }
                self.shot("measurement-scene");
                #[cfg(target_os = "linux")]
                self.relay_capture_barrier()?;
                let cpu_profile = cfg!(target_os = "linux") && std::env::var("DS_RELAY_CPU_PROFILE").ok().as_deref() == Some("1");
                if cpu_profile { crate::gpu::reset_profile(); }
                let _ = crate::gpu::pacing::report_and_reset();
                self.measuring = true;
                self.shot("gameplay-measure-begin");
                #[cfg(target_os = "linux")]
                crate::platform::present_timing::begin();
                // Wall time: a slow renderer must not extend the measurement.
                #[cfg(target_os = "linux")]
                if std::env::var("DS_RELAY_IDLE_COMPARE").ok().as_deref() == Some("1") {
                    let original = [crate::gpu::tune::job_idle(), crate::gpu::tune::job_sleep(), crate::gpu::tune::job_watch()];
                    let mut samples = Vec::new();
                    let result = (|| -> Result<(), End> {
                        for (label, values) in [("legacy", [50, 200, 2000]), ("bounded-sleep", original)] {
                            for (key, value) in ["job_idle", "job_sleep", "job_watch"].into_iter().zip(values) { crate::gpu::tune::set(key, value); }
                            let before = crate::jobidle::relay_snapshot();
                            let start = Instant::now();
                            while start.elapsed() < Duration::from_secs_f64(seconds as f64 / 2.0) {
                                self.check_stop()?;
                                std::thread::sleep(Duration::from_millis(100));
                            }
                            let sample = crate::jobidle::relay_delta(&before, &crate::jobidle::relay_snapshot(), start.elapsed().as_secs_f64(), label);
                            crate::info!("relay idle comparison: {sample}");
                            samples.push(sample);
                        }
                        Ok(())
                    })();
                    for (key, value) in ["job_idle", "job_sleep", "job_watch"].into_iter().zip(original) { crate::gpu::tune::set(key, value); }
                    if let Some(dir) = crate::run_dir() {
                        let _ = std::fs::write(dir.join("job-idle-comparison.json"), serde_json::to_vec_pretty(&samples).unwrap());
                    }
                    result?;
                } else {
                    let start = Instant::now();
                    while start.elapsed() < Duration::from_secs(seconds) {
                        self.check_stop()?;
                        std::thread::sleep(Duration::from_millis(250));
                    }
                }
                #[cfg(windows)]
                {
                    let start = Instant::now();
                    while start.elapsed() < Duration::from_secs(seconds) {
                        self.check_stop()?;
                        std::thread::sleep(Duration::from_millis(250));
                    }
                }
                #[cfg(target_os = "linux")]
                crate::platform::present_timing::end("gameplay");
                self.shot("gameplay-measure-end");
                self.measuring = false;
                if cpu_profile {
                    if let Some(dir) = crate::run_dir() {
                        let _ = std::fs::write(dir.join("cpu-gameplay-profile.txt"), crate::gpu::status_ex(false));
                    }
                }
                crate::info!("relay gameplay frame times: {}", crate::gpu::pacing::report_and_reset());
                self.shot("gameplay-timeline-begin");
                match crate::gpu::timeline::start(16) {
                    Ok(()) => {
                        let start = Instant::now();
                        while crate::gpu::timeline::on() && start.elapsed() < Duration::from_secs(30) {
                            self.check_stop()?;
                            std::thread::sleep(Duration::from_millis(250));
                        }
                    }
                    Err(e) => crate::warn!("relay timeline: {e}"),
                }
                self.shot("gameplay-timeline-end");
            }
        }
        Ok(())
    }

    /// The in-game menu opened (Options, again until its equipment screen's character preview is
    /// rendered: its 640x360 depth target) and closed (Circle).
    fn menu(&mut self) -> Result<(), End> {
        let all = self.w.mark();
        for _ in 0..3 {
            let m = self.w.mark();
            self.press("options", 300);
            if self.wait_for("new render target .*640x360 D32", m, 8.0)? {
                break;
            }
            self.sleep(10.0)?;
        }
        if !self.w.found("new render target .*640x360 D32", all) && !self.w.found("new render target .*640x360 D32", 0) {
            crate::warn!("prepare-play: the menu's character preview was not seen");
        }
        self.sleep(3.0)?;
        // Its other pages (R1), then closed.
        for _ in 0..4 {
            self.press("r1", 300);
            self.sleep(2.5)?;
        }
        self.press("circle", 300);
        self.sleep(3.0)
    }

    fn continue_save(&mut self) -> Result<(), End> {
        self.title()?;
        self.step("the main menu: the save continued");
        let m = self.w.mark();
        if !self.cross_until("Continue Offline", m, 6)? {
            return Err(End::Failed("no offline notice".into()));
        }
        self.sleep(2.0)?;
        let m = self.w.mark();
        self.press("cross", 300);
        if !self.wait_in_game(m, "Crescent Moon Grass", 180.0)? {
            return Err(End::Failed("not in the game after the load".into()));
        }
        Ok(())
    }

    #[cfg(target_os = "linux")]
    fn relay_continue(&mut self) -> Result<(), End> {
        self.continue_save()?;
        self.relay_settle()?;
        self.step("saved game resumed in gameplay");
        self.relay_capture_barrier()?;
        self.measure_gameplay()?;
        // Same route on cold/warm runs. Both passes stay capture-free, even when
        // pressing a button; boundary PNGs are requested and drained outside them.
        for pass in 1..=2 {
            self.relay_settle()?;
            self.shot(&format!("movement-pass-{pass}-before"));
            self.relay_capture_barrier()?;
            self.measuring = true;
            self.shot(&format!("movement-measure-{pass}-begin"));
            crate::platform::present_timing::begin();
            for (button, flips) in [("ls_up", 60), ("ls_down", 60), ("rs_right", 90), ("rs_left", 90), ("r1", 12), ("r2", 12)] {
                self.press_flips(button, flips);
                // Wait for a fixed number of game flips, not wall-time distance.
                self.relay_flips(flips as u64 + 30)?;
            }
            crate::platform::present_timing::end(&format!("movement-{pass}"));
            self.shot(&format!("movement-measure-{pass}-end"));
            self.measuring = false;
            self.shot(&format!("movement-pass-{pass}-after"));
            self.relay_capture_barrier()?;
        }
        if std::env::var("DS_RELAY_UI_PROBE").ok().as_deref() == Some("1") {
            // Separate diagnostic session only: target readbacks and clear tracing invalidate
            // timings. Two captures bracket a camera move; no user input is consumed.
            struct Trace;
            impl Drop for Trace { fn drop(&mut self) { crate::gpu::images::CLEAR_PROBE.store(false, Relaxed); } }
            let _trace = Trace;
            crate::gpu::images::CLEAR_PROBE.store(true, Relaxed);
            for frame in 0..2 {
                self.step(&format!("ui-clear-probe-{frame}-begin"));
                crate::info!("UI clear probe metadata:\n{}", crate::gpu::images::with(|im| im.meta_report()).join("\n"));
                let before = crate::gpu::capture::completed_flip();
                crate::gpu::capture::arm_next();
                let start = Instant::now();
                while crate::gpu::capture::completed_flip() == before && start.elapsed() < Duration::from_secs(90) {
                    self.check_stop()?;
                    std::thread::sleep(Duration::from_millis(100));
                }
                if crate::gpu::capture::completed_flip() == before { return Err(End::Failed("UI target capture timed out".into())); }
                self.shot(&format!("ui-clear-probe-{frame}-end"));
                crate::info!("UI clear probe completed frame {frame}");
                self.press("rs_right", 200);
                self.sleep(0.3)?;
            }
        }
        if std::env::var("DS_RELAY_SHORT").ok().as_deref() != Some("1") { self.menu()?; }
        self.sleep(1.0)
    }

    #[cfg(target_os = "linux")]
    fn relay_flips(&self, count: u64) -> Result<(), End> {
        let start = crate::gpu::FRAMES.load(Relaxed);
        let deadline = Instant::now() + Duration::from_secs(90);
        while crate::gpu::FRAMES.load(Relaxed).saturating_sub(start) < count {
            self.check_stop()?;
            if Instant::now() >= deadline { return Err(End::Failed("relay frame wait exceeded 90s".into())); }
            std::thread::sleep(Duration::from_millis(10));
        }
        Ok(())
    }

    #[cfg(target_os = "linux")]
    fn relay_capture_barrier(&self) -> Result<(), End> {
        // Two flips let an outstanding request finish its readback and enqueue its writer.
        self.relay_flips(2)?;
        crate::hle::libs::videoout::wait_shots();
        if !crate::hle::libs::videoout::shots_idle() {
            return Err(End::Failed("relay capture request did not drain".into()));
        }
        Ok(())
    }

    #[cfg(target_os = "linux")]
    fn relay_settle(&mut self) -> Result<(), End> {
        // The 1.07 loading screen has HUD=304/768 and can briefly have no menu.
        // Real Outpost gameplay has HUD=880. The shared loose HUD test is for
        // automation progress, not permission to start a performance measurement.
        let deadline = Instant::now() + Duration::from_secs(240);
        let mut stable = None;
        loop {
            self.check_stop()?;
            let up = crate::uistate::hud_count().is_some_and(|n| n >= 800)
                && crate::uistate::menu_depth() == Some(0);
            if !up { stable = None; }
            else {
                let (since, frame) = *stable.get_or_insert_with(|| (Instant::now(), crate::gpu::FRAMES.load(Relaxed)));
                if since.elapsed() >= Duration::from_secs(5) && crate::gpu::FRAMES.load(Relaxed).saturating_sub(frame) >= 120 {
                    crate::info!("relay gameplay ready: stable HUD/menu and 120+ flips; {}", crate::uistate::report());
                    return Ok(());
                }
            }
            if Instant::now() >= deadline { return Err(End::Failed(format!("no stable gameplay after 240s: {}", crate::uistate::report()))); }
            std::thread::sleep(Duration::from_millis(100));
        }
    }

    fn tour(&mut self) -> Result<(), End> {
        self.continue_save()?;
        #[cfg(target_os = "linux")]
        self.relay_settle()?;
        self.step("in the game: the tour's start");
        // The automation's own waits, then its views (`dsport-tour:` before each). One that
        // started before the save was loaded had its level loads ignored.
        if self.w.found("dsport-tour: start", 0) {
            return Err(End::Failed("the tour started before the game was loaded".into()));
        }
        let m = self.w.mark();
        // (The tour's opening wait ended now that the game is in play: `automation_wait_hook`.)
        SKIP_FIRST_WAIT.store(true, Relaxed);
        let started = self.wait_for("dsport-tour: start", m, 300.0);
        SKIP_FIRST_WAIT.store(false, Relaxed);
        if !started? {
            return Err(End::Failed("the tour did not start".into()));
        }
        let s = WAIT_SKIPPED_S.load(Relaxed);
        crate::info!("prepare-play: the tour began at {:.0} s ({})", self.t0.elapsed().as_secs_f64(), if s > 0 { format!("its opening wait ended with {s} s left") } else { "its opening wait had ended by itself".to_string() });
        // The fast-forward's frame rate from here on, the game in play (not from the boot: the
        // opening wait counts frames, and at 120 a second it would end before a slow PC had
        // loaded the save, the release review of 2026-11-20): the vblank thread's uncapped
        // timer, flips at most `FAST_FPS` a second.
        if fast_forward() {
            crate::gpu::tune::set("uncapped", 1);
            crate::gpu::tune::set("frame_cap", FAST_FPS);
            crate::info!("prepare-play: the levels as a fast-forward: up to {FAST_FPS} frames a second");
        }
        self.step("the levels");
        let (mut seen, mut fogs, mut levels) = (0usize, 0usize, 0usize);
        // (Video memory's pressure count when the last level started: a split, `End::Continue`.)
        let mut spells_at_level = crate::gpu::images::PRESSURE_SPELLS.load(Relaxed);
        let mut last = GameClock::new();
        // Presses of Cross still to come at a fog door, at these flips (a press while the player
        // still moves may not take: -214433's 2-2 door kept its prompt). Flips, not time: the
        // fast-forward's frames are each a frame at 60 a second.
        let mut fog_presses: Vec<u64> = Vec::new();
        loop {
            if self.w.found("dsport-tour: done", m) {
                self.coverage("the end");
                return Ok(());
            }
            let n = self.w.count("dsport-tour: step", m);
            if n > seen {
                for _ in seen..n {
                    self.done_steps = (self.done_steps + 1).min(self.total);
                }
                self.write_progress();
                seen = n;
                note_frozen(&last);
                last = GameClock::new();
            }
            // A boss's fog door: its prompt pressed (the script prints `dsport-textreset` before
            // each door, so each prompt is logged again).
            let f = self.w.count("Enter fog", m);
            if f > fogs {
                fogs = f;
                let now = crate::gpu::FRAMES.load(Relaxed);
                fog_presses = vec![now, now + 90, now + 180];
                crate::info!("prepare-play: a fog door's prompt ({fogs} so far): Cross now and twice after");
            }
            if fog_presses.first().is_some_and(|&t| crate::gpu::FRAMES.load(Relaxed) >= t) {
                fog_presses.remove(0);
                self.press("cross", 200);
            }
            let l = self.w.count("dsport-tour: level", m);
            if l > levels {
                levels = l;
                self.coverage(&format!("level {l} starts"));
                note_resume(RESUME_BASE.load(Relaxed) as usize + l - 1);
                // (Video memory under pressure since the level before began, and this session has
                // played a level: it ends here and the part goes on in a new one from this level,
                // `EXIT_CONTINUE`. `DS_PREPARE_SPLIT=0`: as before, one session a part.)
                let spells = crate::gpu::images::PRESSURE_SPELLS.load(Relaxed);
                if l >= 2 && spells > spells_at_level && std::env::var("DS_PREPARE_SPLIT").ok().as_deref() != Some("0") && SETUP.get().is_some_and(|s| s.chunk.is_some()) {
                    crate::info!("prepare-play: video memory was under pressure {} times in the level before: this session ends here", spells - spells_at_level);
                    return Err(End::Continue);
                }
                spells_at_level = spells;
                // The preparation's time is up: the session ends at this level's start (the
                // levels before it were played; the rest are left to the player's first visits).
                if past_deadline() {
                    crate::info!("prepare-play: the preparation's time is up at this part's level {l}: the levels from it on are left");
                    LEVELS_CUT.store(true, Relaxed);
                    return Ok(());
                }
            }
            // (A level load and its resources take ~10-60 s; a step that never ends is a stall.)
            if last.elapsed() > Duration::from_secs(600) {
                note_frozen(&last);
                return Err(End::Failed(format!("the levels stopped after {seen} steps")));
            }
            self.sleep(0.25)?;
        }
    }

    /// The background compute preparation's rest, for at most `COMPUTE_TAIL` (the stop file ends
    /// it too: Err).
    fn compute_tail(&mut self) -> Result<(), End> {
        let b = &crate::gpu::compbench::BACKGROUND;
        if !BACKGROUND_ON.load(Relaxed) {
            return Ok(());
        }
        let t = Instant::now();
        self.tail = true;
        let left = || b[0].load(Relaxed).saturating_sub(b[1].load(Relaxed));
        crate::info!("prepare-play: the background compute preparation has {} of {} items left", left(), b[0].load(Relaxed));
        // (The game only waits now: its frames at 10 a second, the list's workers at normal
        // priority and more of them as the CPUs and the free memory allow, `tail_mode`.)
        if left() > 0 {
            crate::gpu::tune::set("frame_cap", 10);
            crate::gpu::compbench::tail_mode();
        }
        // (No deadline: until every item is made, while one is made at least every 10 minutes:
        // a slower PC gets them all. With a deadline: at most `COMPUTE_TAIL`, never past it.)
        let tail = time_left().map(|l| l.min(COMPUTE_TAIL));
        let (mut made, mut moved) = (b[1].load(Relaxed) + b[2].load(Relaxed), Instant::now());
        let mut tick = Instant::now();
        while b[0].load(Relaxed) == 0 || left() > 0 {
            if tail.is_some_and(|tl| t.elapsed() >= tl) {
                break;
            }
            // (A gap of over 10 s: the PC slept or the process was held; the no-progress time
            // starts again, as in `drain`; the review of 2026-11-21 night.)
            let now = Instant::now();
            if now.duration_since(tick) > Duration::from_secs(10) {
                moved = now;
            }
            tick = now;
            let m = b[1].load(Relaxed) + b[2].load(Relaxed);
            if m != made {
                (made, moved) = (m, Instant::now());
            }
            if moved.elapsed() >= Duration::from_secs(600) {
                crate::warn!("prepare-play: the background compute preparation made nothing for 10 minutes: its rest is left");
                break;
            }
            self.sleep(1.0)?;
            self.write_progress();
        }
        crate::info!("prepare-play: the background compute preparation made {} of {} items ({} failed; {:.0} s waited at the end)", b[1].load(Relaxed), b[0].load(Relaxed), b[2].load(Relaxed), t.elapsed().as_secs_f64());
        Ok(())
    }

    /// The pipelines met so far (the coverage log: what each level added).
    fn coverage(&self, at: &str) {
        use crate::gpu::exec::{FIRST_USES, draw};
        crate::info!(
            "prepare-play: at {at} ({:.0} s): {} graphics pipelines queued for the background, {} graphics and {} compute pipelines missing at their first use ({:.0} s of compute creating or waiting); the background list: {} of {} items made ({} held by the store already); {}",
            self.t0.elapsed().as_secs_f64(),
            draw::ASYNC_GFX_STATS[1].load(Relaxed),
            draw::GFX_FIRST_USES[0].load(Relaxed),
            FIRST_USES[0].load(Relaxed),
            FIRST_USES[1].load(Relaxed) as f64 / 1e9,
            crate::gpu::compbench::BACKGROUND[1].load(Relaxed),
            crate::gpu::compbench::BACKGROUND[0].load(Relaxed),
            crate::gpu::compbench::STORED_SKIPS.load(Relaxed),
            crate::gpu::exec::import_backlog()
        );
    }
}

/// The levels stage's choices: `DS_PREPARE_TOUR=<camera positions a level>[:<headings>[:<dwell
/// s>]]` (0 positions: all; default 3, 3 headings, 0.5 s), `DS_PREPARE_LEVELS=<w-l>,...` (only
/// those levels), `DS_PREPARE_SHOTS=1` (screenshots at its walks and bosses: tests),
/// `DS_PREPARE_SPEED=<x>` (the game's speed, default 2: `gamestep::SPEED_X100`; the walks' and
/// fights' waits and holds are its share of their real-time lengths).
struct LevelOpts {
    per_level: usize,
    dirs: u32,
    dwell: f64,
    only: Vec<(u32, u32)>,
    shots: bool,
    speed: f64,
    /// The part of the levels this session plays (`--prepare-chunk`: i of n, in their order).
    chunk: Option<(usize, usize)>,
}

/// Whether the levels play as a fast-forward (`DS_PREPARE_FAST=0`: not, a test): each frame the
/// game time of a frame at 60 a second (`gamestep::FIXED_FEED`), the automation's waits and the
/// self-play's presses and sticks counted in frames (`input::FLIP_TIMED`), and once the tour
/// has begun the frames as fast as the PC makes them up to `FAST_FPS` a second. What a
/// frame does stays; a PC above 60 frames a second plays the levels in less time, one below
/// it in the same frames as before (2026-11-20: the levels had run at ~30 frames a second,
/// their frame-counted waits twice as long as written).
fn fast_forward() -> bool {
    std::env::var("DS_PREPARE_FAST").ok().as_deref() != Some("0")
}

/// The fast-forward's frame limit (set once the tour has begun, `Play::tour`): game time up to 4
/// times real time at the levels' speed of 2, so that streaming and the pipelines compiled in
/// the background keep up.
const FAST_FPS: u64 = 120;

/// The levels stage's game speed (`DS_PREPARE_SPEED`, 0.25 to 4; default 2).
fn level_speed() -> f64 {
    std::env::var("DS_PREPARE_SPEED").ok().and_then(|v| v.trim().parse::<f64>().ok()).unwrap_or(2.0).clamp(0.25, 4.0)
}

impl LevelOpts {
    fn from_env() -> LevelOpts {
        let mut o = LevelOpts { per_level: 3, dirs: 3, dwell: 0.5, only: Vec::new(), shots: std::env::var("DS_PREPARE_SHOTS").ok().as_deref() == Some("1"), speed: level_speed(), chunk: None };
        if let Ok(v) = std::env::var("DS_PREPARE_TOUR") {
            let mut it = v.split(':');
            if let Some(p) = it.next().and_then(|p| p.trim().parse().ok()) {
                o.per_level = p;
            }
            if let Some(d) = it.next().and_then(|d| d.trim().parse::<u32>().ok()) {
                o.dirs = d.clamp(1, 8);
            }
            if let Some(w) = it.next().and_then(|w| w.trim().parse::<f64>().ok()) {
                o.dwell = w.clamp(0.1, 5.0);
            }
        }
        if let Ok(v) = std::env::var("DS_PREPARE_LEVELS") {
            o.only = v.split([',', ';']).filter_map(|l| level_of(l.trim())).collect();
        }
        o
    }
}

/// "w-l" as (world, level).
fn level_of(s: &str) -> Option<(u32, u32)> {
    let (a, b) = s.split_once('-')?;
    Some((a.trim().parse().ok()?, b.trim().parse().ok()?))
}

/// A point of interest of the title's developers (`misc/dspointsofinterest.txt`: `+poi
/// group="..." name="..." script="..." command="+cp11_loadLevel <w> <l>"
/// commandIfNoSpawnLocation="SetPlayerPos ..."`) that places the player.
struct Poi {
    group: String,
    name: String,
    level: (u32, u32),
    /// The placement (`SetPlayerPos posX=.. ... camEulerZ=..`).
    pos: String,
    /// The level's load with its words after the level when it has any (`areaLoaderGroup=..`:
    /// the part of the level the point is in), else None.
    load: Option<String>,
}

fn pois(text: &str) -> Vec<Poi> {
    let field = |l: &str, k: &str| -> Option<String> {
        let key = format!("{k}=\"");
        let i = l.find(&key)? + key.len();
        let j = l[i..].find('"')?;
        Some(l[i..i + j].to_string())
    };
    let mut out = Vec::new();
    for line in text.lines() {
        let l = line.trim();
        if !l.starts_with("+poi") {
            continue;
        }
        let (Some(group), Some(name), Some(command)) = (field(l, "group"), field(l, "name"), field(l, "command")) else { continue };
        let pos = field(l, "commandIfNoSpawnLocation").unwrap_or_default().trim().trim_start_matches('+').trim().to_string();
        let mut w = command.split_whitespace().skip_while(|t| !t.ends_with("cp11_loadLevel")).skip(1);
        let (Some(a), Some(b)) = (w.next().and_then(|v| v.parse().ok()), w.next().and_then(|v| v.parse().ok())) else { continue };
        // (Only `name=value` words after the level are taken with it: an area group.)
        let extra: Vec<&str> = w.filter(|t| t.contains('=') && t.chars().all(|c| c.is_ascii_alphanumeric() || c == '=' || c == '_')).collect();
        let load = (!extra.is_empty()).then(|| format!("+cp11_loadLevel {a} {b} {}", extra.join(" ")));
        // (A name or group with quotes or markers would break the script's lines.)
        if pos.starts_with("SetPlayerPos ") && !pos.contains('"') {
            out.push(Poi { group: group.replace(['"', '\\'], ""), name: name.replace(['"', '\\'], ""), level: (a, b), pos, load });
        }
    }
    out
}

/// The levels stage's automation script (the title's developer automation, as `tools/tour.py`
/// and `tools/areas.py` write the development tours) and its number of steps, from the title's
/// own files: the screenshot positions of `misc/automation/demonssoulsscreenshots.txt` and the
/// points of interest of `misc/dspointsofinterest.txt`.
fn tour_script(game: &dsconfig::zar::Source) -> Result<(String, u32), String> {
    let text = |rel: &str| game.read(rel).and_then(|b| String::from_utf8(b).map_err(|e| std::io::Error::new(std::io::ErrorKind::InvalidData, e)));
    let src = "misc/automation/demonssoulsscreenshots.txt";
    let shots = text(src).map_err(|e| format!("{src}: {e}"))?;
    let poi_file = "misc/dspointsofinterest.txt";
    let poi = text(poi_file).unwrap_or_else(|e| {
        crate::warn!("prepare-play: {poi_file}: {e} (the camera views only)");
        String::new()
    });
    let mut o = LevelOpts::from_env();
    o.chunk = SETUP.get().and_then(|s| s.chunk);
    levels_lines(&shots, &poi, &o).ok_or_else(|| format!("no positions in {src}"))
}

/// The levels stage's script (None: no positions). Each level is loaded once and, in this
/// order (the gameplay camera before the free one):
/// - a walk and a fight where it starts (the camera turned, a few steps, the nearest enemy
///   locked on and attacked; the player can be neither hurt nor killed, `cp11_characters_*`);
/// - its boss: the player at its fog door's point, walking at it (the driver presses Cross at
///   the "Enter fog" prompt, `Play::tour`), then fighting it;
/// - every fourth of its area points that place the player (world 1's), each a walk and a fight;
/// - the camera at its screenshot positions, `dirs` headings each;
/// - the gameplay camera back (`ToggleDebugCamera`).
/// Each step prints `dsport-tour: step <n> (<what>)`, each level `dsport-tour: level <w>-<l>`.
fn levels_lines(text: &str, poi_text: &str, o: &LevelOpts) -> Option<(String, u32)> {
    let (per_level, dirs) = (o.per_level, o.dirs);
    // (level, camera words) in the file's order.
    let mut locs: Vec<((u32, u32), Vec<(String, f64)>)> = Vec::new();
    for line in text.lines() {
        let l = line.trim();
        let Some(rest) = l.strip_prefix("*callmacro doArea") else { continue };
        let rest = rest.trim_start_matches("Stats").trim_start_matches("Internal");
        // Two quoted strings (the area, the image), the world and level, then name=value words.
        let mut q = rest.split('"');
        let (_, _area, _, _png) = (q.next(), q.next(), q.next(), q.next());
        let Some(tail) = q.next() else { continue };
        let mut words = tail.split_whitespace();
        let (Some(w), Some(lv)) = (words.next().and_then(|v| v.parse().ok()), words.next().and_then(|v| v.parse().ok())) else { continue };
        let cam: Vec<(String, f64)> = words.filter_map(|kv| kv.split_once('=')).filter_map(|(k, v)| Some((k.to_string(), v.parse::<f64>().ok()?))).collect();
        if cam.len() == 6 {
            locs.push(((w, lv), cam));
        }
    }
    let points = pois(poi_text);
    if locs.is_empty() && points.is_empty() {
        return None;
    }
    // The levels in the screenshots' order, then those with only a boss.
    let mut levels: Vec<(u32, u32)> = Vec::new();
    for lv in locs.iter().map(|(l, _)| *l).chain(points.iter().filter(|p| p.group == "Boss Fog Doors").map(|p| p.level)) {
        if !levels.contains(&lv) {
            levels.push(lv);
        }
    }
    if !o.only.is_empty() {
        levels.retain(|l| o.only.contains(l));
    }
    // By depth: the tutorial and the Nexus, then every world's first level, every world's second,
    // and so on (after 1-1 a player may go to any world: what a deadline leaves is the deepest).
    levels.sort_by_key(|&(w, l)| if w == 8 || w == 0 { (0, w == 0, l) } else { (l, false, w) });
    // The chunk's share, in that order (a deadline that ends the preparation early leaves the
    // deepest levels).
    if let Some((i, n)) = o.chunk {
        let len = levels.len();
        let (from, to) = ((i - 1) * len / n, i * len / n);
        levels = levels[from..to].to_vec();
        // (This part begun again after a crash or a hang: from the level its last attempt
        // reached, `resume_from`, not its first; a crash had cost the part's whole replay, about
        // an hour on the other PC; the review of 2026-11-21 night.)
        let from_level = resume_from((i, n));
        // (Past its last level: the part's attempts reached every one of its levels, the last
        // one hung or crashed; nothing is left to play. The clamp below had played that level
        // again, which hung again; the review of 2026-11-21 night.)
        if from_level > 0 && from_level >= levels.len() {
            crate::info!("prepare-play: this part's attempts reached all of its {} levels: nothing left to play", levels.len());
            end(0);
        }
        let skip = from_level.min(levels.len().saturating_sub(1));
        if skip > 0 {
            crate::info!("prepare-play: this part goes on from its level {} of {} (its last attempt reached it)", skip + 1, levels.len());
            levels.drain(..skip);
            RESUME_BASE.store(skip as u32, Relaxed);
        }
    }
    // (The automation counts frames from the title's boot, and the title ignores level loads
    // until it is in the game: its first wait covers the self-play's way through the title, the
    // main menu and the save's load, ~50-80 s.)
    let mut out: Vec<String> = [
        "// The shader preparation's levels (written by the game's self-play from misc/automation/demonssoulsscreenshots.txt and misc/dspointsofinterest.txt)",
        "wait 120.0",
        "waitForLevelLoading",
        "waitForResourceLoading",
        "print \"dsport-tour: start\"",
        "+cp11_characters_playerNoDead 1",
        "+cp11_characters_playerNoDamage 1",
        "+cp11_characters_playerNoHit 1",
    ]
    .map(String::from)
    .to_vec();
    let mut steps = 0u32;
    let ctl = |c: &str| format!("print \"dsport-control: {c}\"");
    // Waits and holds of the walks and fights at the game's speed (the game time they cover
    // stays; the waits count frames, `gamestep`).
    let sp = o.speed.clamp(0.25, 4.0);
    let w = |s: f64| format!("wait {:.2}", s / sp);
    let ms = |m: f64| ((m / sp).round() as u64).max(60);
    // The nearest enemy locked on and attacked `n` times, a step in after each.
    let fight = |out: &mut Vec<String>, n: u32| {
        out.push(ctl("press r3 200ms"));
        out.push(w(0.5));
        for _ in 0..n {
            out.extend([ctl("press r1 200ms"), w(0.8), ctl(&format!("stick l 0 -127 {}", ms(300.0))), w(0.45)]);
        }
        out.push(ctl("press r3 200ms"));
        out.push(w(0.3));
    };
    // The camera turned, a few steps forward, a fight, the camera turned back.
    let walk = |out: &mut Vec<String>, attacks: u32| {
        out.extend([ctl(&format!("stick r 127 0 {}", ms(1500.0))), w(2.0), ctl(&format!("stick l 0 -127 {}", ms(2000.0))), w(2.5)]);
        if attacks > 0 {
            fight(out, attacks);
        }
        out.extend([ctl(&format!("stick r -127 0 {}", ms(1500.0))), w(2.0)]);
    };
    let shot = |out: &mut Vec<String>, name: String| {
        if o.shots {
            out.push(format!("print \"dsport-shot: {name}\""));
            out.push("wait 0.3".into());
        }
    };
    for lv in &levels {
        let tag = format!("{}-{}", lv.0, lv.1);
        let ftag = format!("{}_{}", lv.0, lv.1);
        out.push(format!("print \"dsport-tour: level {tag}\""));
        out.push(format!("+cp11_loadLevel {} {}", lv.0, lv.1));
        out.extend(["wait 5.0", "waitForLevelLoading", "waitForResourceLoading", "wait 2.0"].map(String::from));
        // Where the level starts (no fight in the Nexus: its people are not enemies).
        steps += 1;
        out.push(format!("print \"dsport-tour: step {steps} (walk {tag} start)\""));
        shot(&mut out, format!("lv{ftag}_start"));
        walk(&mut out, if *lv == (0, 1) { 0 } else { 4 });
        // The boss through its fog door (its point is at the fog: the prompt comes at once, and
        // the driver presses it), then three rounds of a lock-on and attacks with the camera
        // turned between them (the door's way in may face a wall: -211521).
        for b in points.iter().filter(|p| p.level == *lv && p.group == "Boss Fog Doors") {
            steps += 1;
            out.push("print \"dsport-textreset\"".into());
            out.push(format!("print \"dsport-tour: step {steps} (boss {tag} {})\"", b.name));
            // (A door in a part of the level loaded on its own: 2-1's, 2-2's and 2-3's.)
            if let Some(load) = &b.load {
                out.push(load.clone());
                out.extend(["wait 5.0", "waitForLevelLoading", "waitForResourceLoading"].map(String::from));
            }
            out.push(format!("+{}", b.pos));
            out.extend(["wait 3.0", "waitForResourceLoading", "wait 1.0"].map(String::from));
            shot(&mut out, format!("lv{ftag}_door"));
            out.push(ctl(&format!("stick l 0 -127 {}", ms(1000.0))));
            out.push(w(5.0));
            shot(&mut out, format!("lv{ftag}_boss"));
            for _ in 0..3 {
                fight(&mut out, 4);
                out.extend([ctl(&format!("stick r 127 0 {}", ms(800.0))), w(1.2)]);
            }
            shot(&mut out, format!("lv{ftag}_fight"));
        }
        // The level's cutscenes (the points beside their triggers): a few steps in, then the
        // cutscene's first seconds (its scene, people and effects show early; the rest of a long
        // one runs on into the next step).
        for c in points.iter().filter(|p| p.level == *lv && p.group == "Cutscenes") {
            steps += 1;
            out.push(format!("print \"dsport-tour: step {steps} (cutscene {tag} {})\"", c.name));
            if let Some(load) = &c.load {
                out.push(load.clone());
                out.extend(["wait 5.0", "waitForLevelLoading", "waitForResourceLoading"].map(String::from));
            }
            out.push(format!("+{}", c.pos));
            out.extend(["wait 2.0", "waitForResourceLoading", "wait 0.5"].map(String::from));
            out.extend([ctl(&format!("stick l 0 -127 {}", ms(1500.0))), w(7.0)]);
            shot(&mut out, format!("lv{ftag}_cutscene"));
        }
        // World 1's area points (every fourth one).
        let areas: Vec<&Poi> = points.iter().filter(|p| p.level == *lv && !["NPC", "Cutscenes", "Boss Fog Doors"].contains(&p.group.as_str()) && !p.group.starts_with("Tutorial")).collect();
        for (k, p) in areas.iter().enumerate().filter(|(k, _)| k % 4 == 0) {
            steps += 1;
            out.push(format!("print \"dsport-tour: step {steps} (walk {tag} {})\"", p.name));
            out.push(format!("+{}", p.pos));
            out.extend(["wait 2.0", "waitForResourceLoading", "wait 0.5"].map(String::from));
            if k == 0 {
                shot(&mut out, format!("lv{ftag}_area"));
            }
            walk(&mut out, 4);
        }
        // The camera at the screenshot positions (`per_level` spread over the level's list, 0
        // all), `dirs` headings each.
        let here: Vec<&Vec<(String, f64)>> = locs.iter().filter(|(l, _)| l == lv).map(|(_, c)| c).collect();
        let picked: Vec<&Vec<(String, f64)>> = if per_level == 0 || here.len() <= per_level {
            here
        } else {
            (0..per_level).map(|i| here[i * here.len() / per_level]).collect()
        };
        for (i, cam) in picked.iter().enumerate() {
            for d in 0..dirs {
                let words: Vec<String> = cam
                    .iter()
                    .map(|(k, v)| {
                        let v = if k == "eulerY" { (v + 360.0 * d as f64 / dirs as f64).rem_euclid(360.0) } else { *v };
                        format!("{k}={v:.2}")
                    })
                    .collect();
                steps += 1;
                out.push(format!("print \"dsport-tour: step {steps} (view {tag})\""));
                out.push(format!("+SetCameraPos {}", words.join(" ")));
                if d == 0 {
                    // A new position: its areas streamed in (the camera moves the area loader).
                    out.extend(["wait 1.0", "waitForResourceLoading", format!("wait {:.1}", o.dwell).as_str()].map(String::from));
                    if i == 0 {
                        shot(&mut out, format!("lv{ftag}_view"));
                    }
                } else {
                    out.push(format!("wait {:.1}", o.dwell));
                }
            }
        }
        // The gameplay camera back (`SetCameraPos` turns the debug camera on, and it stays on
        // through level loads: the next level's walk looked into the void, -211521).
        if !picked.is_empty() {
            out.extend(["+ToggleDebugCamera", "wait 0.5"].map(String::from));
        }
        // (No `cp11_allSuicideExceptPlayer`, every enemy's death at once: after it and a teleport
        // the title's physics wrote through a freed object at eboot+0x15d5605, the known
        // kill-all fault of FINDINGS.md 2026-11-16, at the next level's boss door, -230554. The
        // fights' kills give the deaths' effects.)
        shot(&mut out, format!("lv{ftag}_end"));
    }
    out.push("print \"dsport-tour: done\"".into());
    out.push(String::new());
    Some((out.join("\n"), steps))
}

#[cfg(test)]
mod tests {
    #[test]
    fn patterns() {
        assert!(super::matches("I: new render target 0x10 640x360 D32 (x)", "new render target .*640x360 D32"));
        assert!(!super::matches("I: new render target 0x10 640x360 R8", "new render target .*640x360 D32"));
        assert!(super::matches("committed [\"SAVEDATA0PlayerProfile0\"] ok", "committed [\"SAVEDATA0PlayerProfile0\"]"));
    }

    /// The levels from texts of the title's shapes (made-up positions and names): each level
    /// loaded once, a walk where it starts, its boss, every fourth area point, its camera
    /// positions with their headings, the macro's own lines and unplaced points skipped.
    #[test]
    fn levels() {
        let text = "*macro doAreaInternal areaName screenshotName world level camX camY camZ camRotX camRotY camRotZ\n\
            \x20   *callmacro doAreaInternal areaName screenshotName world level camX camY camZ camRotX camRotY camRotZ\n\
            *callmacro doArea \"Area A\" \"#/a/A_0.png\" 1 1 posX=1.00 posY=2.00 posZ=3.00 eulerX=-5.00 eulerY=300.00 eulerZ=0.00\n\
            *callmacro doAreaStats \"Area A\" \"#/a/A_1.png\" 1 1 posX=4.00 posY=5.00 posZ=6.00 eulerX=0.00 eulerY=10.00 eulerZ=0.00\n\
            *callmacro doArea \"Area B\" \"#/b/B_0.png\" 2 1 posX=7.00 posY=8.00 posZ=9.00 eulerX=0.00 eulerY=90.00 eulerZ=0.00\n";
        let poi = "+poi group=\"Boss Fog Doors\" name=\"Boss A\" script=\"$/a.script\" command=\"+cp11_loadLevel 1 1\" commandIfNoSpawnLocation=\"SetPlayerPos posX=1.00 posY=0.00 posZ=0.00 eulerX=0.00 eulerY=0.00 eulerZ=0.00 camEulerX=0.00 camEulerY=0.00 camEulerZ=0.00\"\n\
            +poi group=\"Test 1-1\" name=\"P0\" script=\"$/p.script\" command=\"+cp11_loadLevel 1 1\" commandIfNoSpawnLocation=\"SetPlayerPos posX=2.00 posY=0.00 posZ=0.00\"\n\
            +poi group=\"Test 1-1\" name=\"P1\" script=\"$/p.script\" command=\"+cp11_loadLevel 1 1\" commandIfNoSpawnLocation=\"SetPlayerPos posX=3.00 posY=0.00 posZ=0.00\"\n\
            +poi group=\"Test 1-1\" name=\"P2\" script=\"$/p.script\" command=\"+cp11_loadLevel 1 1\" commandIfNoSpawnLocation=\"+SetPlayerPos posX=4.00 posY=0.00 posZ=0.00\"\n\
            +poi group=\"Test 2-1\" name=\"Q0\" script=\"$/q.script\" command=\"+cp11_loadLevel 2 1\"\n\
            +poi group=\"NPC\" name=\"N0\" script=\"$/n.script\" command=\"+cp11_loadLevel 2 1\" commandIfNoSpawnLocation=\"SetPlayerPos posX=5.00 posY=0.00 posZ=0.00\"\n\
            +poi group=\"Cutscenes\" name=\"Scene B\" script=\"$/s.script\" command=\"+cp11_loadLevel 2 1\" commandIfNoSpawnLocation=\"SetPlayerPos posX=8.00 posY=0.00 posZ=0.00\"\n\
            +poi group=\"Boss Fog Doors\" name=\"Boss C\" script=\"$/c.script\" command=\"+cp11_loadLevel 3 1 areaLoaderGroup=under_side\" commandIfNoSpawnLocation=\"SetPlayerPos posX=6.00 posY=0.00 posZ=0.00\"\n";
        let o = |per_level, dirs, only: Vec<(u32, u32)>| super::LevelOpts { per_level, dirs, dwell: 0.5, only, shots: false, speed: 2.0, chunk: None };
        let (s, steps) = super::levels_lines(text, poi, &o(0, 4, Vec::new())).unwrap();
        // 1-1: start, boss, area P0, 2 x 4 views; 2-1: start, cutscene, 4 views; 3-1: start, boss.
        assert_eq!(steps, 19);
        // (Each level's load, and the 3-1 door's part of its level.)
        assert_eq!(s.matches("+cp11_loadLevel").count(), 4);
        assert!(s.contains("+cp11_loadLevel 3 1 areaLoaderGroup=under_side\n"));
        assert!(s.contains("dsport-tour: step 13 (cutscene 2-1 Scene B)") && s.contains("+SetPlayerPos posX=8.00"));
        assert!(s.contains("+SetPlayerPos posX=1.00") && s.contains("+SetPlayerPos posX=2.00") && !s.contains("SetPlayerPos posX=4.00"));
        assert!(!s.contains("posX=3.00") && !s.contains("posX=5.00") && s.contains("+SetPlayerPos posX=6.00"));
        // 300 + 90 wraps to 30.
        assert!(s.contains("posX=1.00 posY=2.00 posZ=3.00 eulerX=-5.00 eulerY=30.00 eulerZ=0.00"));
        assert!(s.contains("dsport-tour: step 19 (boss 3-1 Boss C)") && s.ends_with("print \"dsport-tour: done\"\n"));
        assert_eq!(s.matches("dsport-textreset").count(), 2);
        // The gameplay camera back after each level's views (3-1 has none: no toggle).
        assert_eq!(s.matches("+ToggleDebugCamera").count(), 2);
        // 2-1 alone: its start, its cutscene, one position's two headings.
        let (_, one) = super::levels_lines(text, poi, &o(1, 2, vec![(2, 1)])).unwrap();
        assert_eq!(one, 4);
        // The second of three parts: the second level (2-1) alone, in the levels' order.
        let mut part = o(1, 2, Vec::new());
        part.chunk = Some((2, 3));
        let (s2, n2) = super::levels_lines(text, poi, &part).unwrap();
        assert_eq!(n2, 4);
        assert!(s2.contains("+cp11_loadLevel 2 1") && !s2.contains("+cp11_loadLevel 1 1") && !s2.contains("+cp11_loadLevel 3 1"));
        assert!(super::levels_lines("nothing here", "", &o(0, 4, Vec::new())).is_none());
    }
}

/// The shared text-driven new-game/continue workflow, in the caller's isolated relay folder.
/// No preparation folder deletion, player-save copying, or shader-tour changes are performed.
#[cfg(target_os = "linux")]
pub static RELAY_RESULT: std::sync::atomic::AtomicI32 = std::sync::atomic::AtomicI32::new(-1);
#[cfg(target_os = "linux")]
pub fn start_relay(resume: bool) -> Result<(), String> {
    crate::log::tap_on();
    std::thread::Builder::new().name("linux-selfplay".into()).spawn(move || {
        crate::log::set_thread_name("linux-selfplay");
        let mut p = Play::new(None, NEWGAME_STEPS);
        p.relay = true;
        let result = if resume { p.relay_continue() } else { p.newgame() };
        let success = result.is_ok();
        let message = match result {
            Ok(()) => if resume && cfg!(target_os = "linux") && std::env::var("DS_RELAY_SHORT").ok().as_deref() == Some("1") { "saved game resumed, movement and attacks (short relay)" } else if resume { "saved game resumed, movement, attacks and menu" } else { "new game reached gameplay, movement, attacks and menu" }.to_string(),
            Err(End::Failed(why)) => format!("FAILED: {why}"),
            Err(_) => "STOPPED before completing self-play".to_string(),
        };
        p.shot("selfplay-result");
        crate::info!("Linux self-play result: {message}");
        if let Some(dir) = crate::run_dir() {
            let _ = std::fs::write(dir.join("selfplay.txt"), message);
            let _ = crate::uistate::dump_nodes(dir, "selfplay-final");
        }
        // Let the requested final capture reach a guest flip before the main thread drains.
        std::thread::sleep(Duration::from_secs(5));
        RELAY_RESULT.store(if success { 0 } else { EXIT_FAILED }, std::sync::atomic::Ordering::Release);
    }).map(|_| ()).map_err(|e| e.to_string())
}
