//! Resumable Linux driver for the shared launcher preparation: new game, compute,
//! then all three level-tour parts. Only shader data is shared with normal play.
use std::path::{Path, PathBuf};
use std::process::{Command, Stdio};
use std::time::{Duration, Instant};
use std::os::{fd::AsRawFd, unix::process::CommandExt};
use std::sync::atomic::{AtomicBool, Ordering::Relaxed};
use serde::{Deserialize, Serialize};
static CANCELLED: AtomicBool = AtomicBool::new(false);
extern "C" fn cancel(_: i32) { CANCELLED.store(true, Relaxed); }
const PAUSED: i32 = crate::prepplay::EXIT_STOPPED;
const MARKER: &str = "Regaliant isolated Linux preparation v1\n";

pub fn atomic_write(path: &Path, bytes: &[u8]) -> Result<(), String> {
    use std::io::Write;
    let tmp = path.with_extension(format!("{}.tmp", std::process::id()));
    let mut f = std::fs::File::create(&tmp).map_err(|e| e.to_string())?;
    f.write_all(bytes).and_then(|_| f.sync_all()).map_err(|e| e.to_string())?;
    std::fs::rename(tmp, path).map_err(|e| e.to_string())?;
    std::fs::File::open(path.parent().unwrap()).and_then(|f| f.sync_all()).map_err(|e| e.to_string())
}

fn bounded(cmd: &mut Command, seconds: u64, stop: &Path) -> Result<i32, String> {
    let parent = unsafe { libc::getpid() };
    unsafe { cmd.pre_exec(move || {
        if libc::setpgid(0, 0) != 0 || libc::prctl(libc::PR_SET_PDEATHSIG, libc::SIGKILL) != 0 {
            return Err(std::io::Error::last_os_error());
        }
        if libc::getppid() != parent { libc::_exit(125); }
        Ok(())
    }); }
    let mut child = cmd.spawn().map_err(|e| e.to_string())?;
    let start = Instant::now();
    let mut stopping = None;
    let result = loop {
        match child.try_wait() {
            Ok(Some(s)) => break if stopping.is_some() { Ok(PAUSED) } else { s.code().ok_or_else(|| format!("preparation child terminated: {s}")) },
            Err(e) => break Err(e.to_string()),
            _ => {}
        }
        if (CANCELLED.load(Relaxed) || start.elapsed() >= Duration::from_secs(seconds)) && stopping.is_none() {
            let _ = atomic_write(stop, b"stop\n");
            stopping = Some(Instant::now());
        }
        if stopping.is_some_and(|t| t.elapsed() >= Duration::from_secs(5)) { break Ok(PAUSED); }
        std::thread::sleep(Duration::from_millis(100));
    };
    unsafe { libc::kill(-(child.id() as i32), libc::SIGKILL); }
    let _ = child.wait();
    result
}

struct PreparationLock(std::fs::File);
impl Drop for PreparationLock {
    fn drop(&mut self) {
        // A concurrently forked child can briefly hold this open-file description until
        // CLOEXEC runs. Release the lock explicitly instead of waiting for its last close.
        unsafe { libc::flock(self.0.as_raw_fd(), libc::LOCK_UN); }
    }
}
fn lock_root(root: &Path) -> Result<PreparationLock, String> {
    let lock = std::fs::OpenOptions::new().create(true).truncate(false).read(true).write(true).open(root.join("lock")).map_err(|e| e.to_string())?;
    if unsafe { libc::flock(lock.as_raw_fd(), libc::LOCK_EX | libc::LOCK_NB) } != 0 { return Err("shader preparation is already running for this directory".into()); }
    Ok(PreparationLock(lock))
}

#[derive(Serialize, Deserialize, Debug)]
struct State {
    app: PathBuf,
    shaders: PathBuf,
    eboot: String,
    translation: String,
    selection: String,
    completed: Vec<String>,
    driver: Option<serde_json::Value>,
}
impl State {
    fn save(&self, root: &Path) -> Result<(), String> {
        atomic_write(&root.join("state.json"), &serde_json::to_vec_pretty(self).map_err(|e| e.to_string())?)
    }
    fn invalidate_pipelines(&mut self) { self.completed.retain(|s| s == "newgame"); }
}

fn copy_tree(from: &Path, to: &Path) -> Result<(), String> {
    std::fs::create_dir_all(to).map_err(|e| e.to_string())?;
    for e in std::fs::read_dir(from).map_err(|e| e.to_string())? {
        let e = e.map_err(|e| e.to_string())?;
        let ty = e.file_type().map_err(|e| e.to_string())?;
        let dest = to.join(e.file_name());
        if ty.is_dir() { copy_tree(&e.path(), &dest)?; }
        else if ty.is_file() {
            std::fs::copy(e.path(), &dest).map_err(|e| e.to_string())?;
            std::fs::File::open(&dest).and_then(|f| f.sync_all()).map_err(|e| e.to_string())?;
        } else { return Err(format!("preparation seed contains a symlink/special file: {}", e.path().display())); }
    }
    Ok(())
}

/// Internal stage arguments cannot turn the player's ordinary save folder into a preparation.
pub fn validate_child(root: &Path, paths: &dsconfig::paths::Paths) -> Result<(), String> {
    let root = root.canonicalize().map_err(|e| e.to_string())?;
    if std::fs::read_to_string(root.join("owned.txt")).ok().as_deref() != Some(MARKER)
        || paths.user.canonicalize().ok() != Some(root.join("user"))
        || paths.cache.canonicalize().ok() != Some(root.join("user")) {
        return Err("--prepare-play requires the preparation's owned, isolated user directory".into());
    }
    Ok(())
}

pub fn run(app: &Path, threads: u32, seconds: u64) -> Result<i32, String> {
    let game = dsconfig::game::check(app).map_err(|e| e.to_string())?;
    super::files::protect_game(&game.app).map_err(|e| e.to_string())?;
    let paths = dsconfig::paths::Paths::current();
    let forbidden = [&*game.app];
    let cache = super::files::output_path(&paths.cache, &forbidden).map_err(|e| e.to_string())?;
    let shaders = std::env::var_os("DS_LINUX_SHADER_CACHE").map(PathBuf::from).unwrap_or_else(|| paths.shaders());
    let shaders = super::files::output_path(&shaders, &forbidden).map_err(|e| e.to_string())?;
    let root = super::files::output_path(&cache.join("prepare/linux-full"), &forbidden).map_err(|e| e.to_string())?;
    for output in [&paths.user, &cache, &shaders] { super::files::validate_output_tree(output).map_err(|e| e.to_string())?; }
    std::fs::create_dir_all(&root).map_err(|e| e.to_string())?;
    let _lock = lock_root(&root)?;
    if !root.join("owned.txt").exists() { atomic_write(&root.join("owned.txt"), MARKER.as_bytes())?; }
    if std::fs::read_to_string(root.join("owned.txt")).ok().as_deref() != Some(MARKER) { return Err("unrecognized preparation directory".into()); }
    let user = root.join("user");
    for name in ["user", "tmp", "driver", "xdg-cache", "xdg-data", "xdg-config", "xdg-state"] {
        std::fs::create_dir_all(root.join(name)).map_err(|e| e.to_string())?;
    }
    let translation = format!("{}:portable-modes-mrc1", rdna::SOURCE_HASH);
    let selection = ["DS_PREPARE_LEVELS", "DS_PREPARE_TOUR"].map(|k| std::env::var(k).unwrap_or_default()).join(";");
    let limited = selection != ";";
    let mut state = if root.join("state.json").exists() {
        serde_json::from_slice::<State>(&std::fs::read(root.join("state.json")).map_err(|e| e.to_string())?).map_err(|e| format!("preparation state: {e}"))?
    } else {
        // Copy settings only, never a player's saves or save-folder redirection.
        if let Ok(t) = std::fs::read_to_string(paths.settings_file()) {
            let t = t.lines().filter(|l| !l.trim_start().starts_with("Setting_PC_SaveFolder") && !l.trim_start().starts_with("Setting_PC_ImportSaves")).collect::<Vec<_>>().join("\n");
            atomic_write(&user.join("pcsettings.txt"), t.as_bytes())?;
        }
        if let Ok(b) = std::fs::read(paths.display_file()) { atomic_write(&user.join("display.txt"), &b)?; }
        let mut completed = Vec::new();
        // Development relay only: a copy of its immutable tutorial seed, never player data.
        if let Some(seed) = std::env::var_os("DS_PREPARE_SEED") {
            copy_tree(&PathBuf::from(seed).join("saves"), &user.join("saves"))?;
            completed.push("newgame".into());
        }
        State { app: game.app.clone(), shaders: shaders.clone(), eboot: dsconfig::game::executable_hash(&game).map_err(|e| e.to_string())?, translation: translation.clone(), selection: selection.clone(), completed, driver: None }
    };
    if state.app != game.app || state.shaders != shaders || state.selection != selection { return Err(format!("preparation location/selection changed; retain {} and use a separate DSPORT_HOME", root.display())); }
    if state.eboot != dsconfig::game::executable_hash(&game).map_err(|e| e.to_string())? { return Err("game executable changed; use a fresh preparation directory".into()); }
    if state.translation != translation {
        state.invalidate_pipelines(); state.translation = translation.clone();
        for i in 1..=3 { let _ = std::fs::remove_file(root.join(format!("tour-{i}.resume"))); }
    }
    state.save(&root)?;
    atomic_write(&root.join("summary.txt"), b"Preparation incomplete; stage logs and state.json record progress. Repeat the same command after interruption or failure.\n")?;
    unsafe {
        let mut sa: libc::sigaction = std::mem::zeroed();
        sa.sa_sigaction = cancel as *const () as usize;
        libc::sigemptyset(&mut sa.sa_mask);
        for sig in [libc::SIGINT, libc::SIGTERM] { if libc::sigaction(sig, &sa, std::ptr::null_mut()) != 0 { return Err(std::io::Error::last_os_error().to_string()); } }
    }
    println!("Preparing {} shaders, hidden and silent; resume with the same command. State/logs: {}", if limited { "selected levels (not full coverage)" } else { "all levels in the shared Windows tour" }, root.display());
    let start = Instant::now();
    let parts = if limited { 1 } else { 3 };
    let mut stages = vec!["check-write".to_string(), "check-read".into(), "newgame".into()];
    // The bounded relay learns real variants in its selected level; the full preparation
    // also runs the Windows compute-list stage before touring all levels.
    if !limited { stages.push("compile".into()); }
    if state.driver.is_some() { stages.insert(0, "check-probe".into()); }
    stages.extend((1..=parts).map(|i| format!("tour-{i}")));
    for stage in stages {
        if !stage.starts_with("check-") && state.completed.contains(&stage) { continue; }
        loop {
            let left = seconds.saturating_sub(start.elapsed().as_secs());
            if left == 0 || CANCELLED.load(Relaxed) {
                atomic_write(&root.join("summary.txt"), b"Paused; completed stages and the interrupted level are retained. Repeat the same command.\n")?;
                return Ok(PAUSED);
            }
            let progress = root.join(format!("{stage}.progress"));
            let stop = progress.with_extension("stop");
            let _ = std::fs::remove_file(&stop);
            let _ = std::fs::remove_file(progress.with_extension("health"));
            let attempt = format!("{}-{}-{}", dsconfig::utc_stamp(), std::process::id(), start.elapsed().as_millis());
            let log_path = root.join(format!("{stage}-{attempt}.log"));
            let log = std::fs::File::create(&log_path).map_err(|e| e.to_string())?;
            let mut cmd = Command::new(std::env::current_exe().map_err(|e| e.to_string())?);
            for (k, _) in std::env::vars_os() {
                if k.to_string_lossy().starts_with("DS_") && !["DS_PREPARE_LEVELS", "DS_PREPARE_TOUR", "DS_PREPARE_SHOTS", "DS_READONLY_ROOTS", "DS_COMPILE_WORKERS", "DS_COMPILE_AHEAD", "DS_PRECOMPILE_WORKERS", "DS_ASYNC_GFX_WORKERS", "DS_RELAY_FPS_LIMIT", "DS_RELAY_STOP_FILE"].contains(&k.to_string_lossy().as_ref()) { cmd.env_remove(k); }
            }
            cmd.args([std::ffi::OsStr::new("--app"), game.app.as_os_str()]).arg("--offscreen")
                .env("DSPORT_HOME", &user).env("DS_LINUX_SHADER_CACHE", &shaders)
                .env("DS_LINUX_HIDDEN", "1").env("SDL_AUDIO_DRIVER", "dummy")
                .env("DS_RELAY_GAMEPLAY_SECONDS", "0").env("DS_VK_VALIDATION", "0")
                .env("VK_LOADER_LAYERS_DISABLE", "*validation*").env_remove("VK_INSTANCE_LAYERS").env_remove("VK_LOADER_LAYERS_ENABLE")
                .env("TMPDIR", root.join("tmp")).env("TMP", root.join("tmp")).env("TEMP", root.join("tmp"))
                .env("XDG_CACHE_HOME", root.join("xdg-cache")).env("XDG_DATA_HOME", root.join("xdg-data"))
                .env("XDG_CONFIG_HOME", root.join("xdg-config")).env("XDG_STATE_HOME", root.join("xdg-state"))
                .env("MESA_SHADER_CACHE_DIR", root.join("driver")).env("__GL_SHADER_DISK_CACHE_PATH", root.join("driver"))
                .env("CUDA_CACHE_PATH", root.join("driver"))
                .stdin(Stdio::null()).stderr(log.try_clone().map_err(|e| e.to_string())?).stdout(log);
            if let Some(phase) = stage.strip_prefix("check-") { cmd.args(["--pb-check", phase]); }
            else if stage == "compile" { cmd.args(["--precompile-all", &threads.to_string()]); }
            else {
                cmd.args(["--prepare-play", if stage == "newgame" { "newgame" } else { "tour" }])
                    .arg("--prepare-root").arg(&root).arg("--prepare-threads").arg(threads.to_string())
                    .arg("--progress").arg(&progress).args(["--run-seconds", "3600"]);
                if let Some(i) = stage.strip_prefix("tour-") { cmd.args(["--prepare-chunk", &format!("{i}/{parts}")]); }
                if limited { cmd.env("DS_PREPARE_BACKGROUND", "0"); }
            }
            println!("Preparation {stage}: {} ({}s remaining)", log_path.display(), left);
            let code = bounded(&mut cmd, left.min(3600), &stop)?;
            state.save(&root)?;
            if code == PAUSED {
                atomic_write(&root.join("summary.txt"), b"Paused at deadline or interrupt; repeat the same command to resume.\n")?;
                return Ok(PAUSED);
            }
            if code == crate::prepplay::EXIT_CONTINUE { continue; }
            if stage == "check-probe" && [20, 21, 22].contains(&code) {
                state.invalidate_pipelines();
                for i in 1..=3 { let _ = std::fs::remove_file(root.join(format!("tour-{i}.resume"))); }
                state.save(&root)?;
                println!("Previous pipelines were not reusable; preparation will replay after write/read checks.");
                break;
            }
            if code != 0 { return Err(format!("{stage} exited {code}; resume state retained; inspect {}", log_path.display())); }
            if stage == "check-read" {
                let text = std::fs::read_to_string(&log_path).map_err(|e| e.to_string())?;
                let identity = text.lines().find(|l| l.contains("pipeline store identity:")).map(|l| l.split("pipeline store identity:").nth(1).unwrap().trim().to_string());
                let identity = serde_json::to_value(identity).map_err(|e| e.to_string())?;
                if state.driver.as_ref().is_some_and(|old| old != &identity) {
                    state.invalidate_pipelines();
                    for i in 1..=3 { let _ = std::fs::remove_file(root.join(format!("tour-{i}.resume"))); }
                }
                state.driver = Some(identity);
            }
            if stage == "newgame" || stage.starts_with("tour-") {
                let text = std::fs::read_to_string(&log_path).map_err(|e| e.to_string())?;
                let expected = format!("prepare-play: stage {} played to its end", if stage == "newgame" { "newgame" } else { "tour" });
                if !text.contains(&expected) { return Err(format!("{stage} exited without completing; checkpoint retained, inspect {}", log_path.display())); }
            }
            if !stage.starts_with("check-") { state.completed.push(stage.clone()); }
            state.save(&root)?;
            break;
        }
    }
    let message = if limited { "Selected-level preparation completed; this is NOT whole-game coverage.\n" } else { "New game, compute list and all three shared level-tour parts completed. Unseen gameplay variants can still compile later.\n" };
    atomic_write(&root.join("summary.txt"), message.as_bytes())?;
    println!("{message}");
    Ok(0)
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn child_failure_timeout_and_success() {
        let stop = std::env::temp_dir().join(format!("regaliant-prepare-test-{}.stop", std::process::id()));
        assert_eq!(bounded(Command::new("/bin/sh").args(["-c", "exit 0"]), 2, &stop).unwrap(), 0);
        assert_eq!(bounded(Command::new("/bin/sh").args(["-c", "exit 7"]), 2, &stop).unwrap(), 7);
        assert_eq!(bounded(Command::new("/bin/sh").args(["-c", "sleep 30"]), 0, &stop).unwrap(), PAUSED);
        assert!(stop.is_file());
        let _ = std::fs::remove_file(stop);
    }
    #[test]
    fn ownership_checkpoint_and_lock_survive_restart() {
        let root = std::env::temp_dir().join(format!("regaliant-owned-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&root);
        std::fs::create_dir_all(root.join("user")).unwrap();
        let paths = dsconfig::paths::Paths { layout:dsconfig::paths::Layout::Portable, exe_dir:root.clone(), user:root.join("user"), cache:root.join("user"), root:None };
        assert!(validate_child(&root, &paths).is_err());
        atomic_write(&root.join("owned.txt"), MARKER.as_bytes()).unwrap();
        assert!(validate_child(&root, &paths).is_ok());
        let mut player = paths.clone(); player.user = root.clone();
        assert!(validate_child(&root, &player).is_err());
        let lock = lock_root(&root).unwrap();
        assert!(lock_root(&root).is_err());
        drop(lock);
        let _lock = lock_root(&root).unwrap();
        let checkpoint = root.join("tour-1.resume");
        atomic_write(&checkpoint, b"chunk=1/3\nlevel=2\n").unwrap();
        atomic_write(&checkpoint, b"chunk=1/3\nlevel=3\n").unwrap();
        assert_eq!(std::fs::read_to_string(checkpoint).unwrap(), "chunk=1/3\nlevel=3\n");
        std::os::unix::fs::symlink(root.join("owned.txt"), root.join("user/link")).unwrap();
        assert!(copy_tree(&root.join("user"), &root.join("copy")).is_err());
        std::fs::remove_dir_all(root).unwrap();
    }
    #[test]
    fn translator_or_driver_change_keeps_only_the_isolated_newgame() {
        let mut s = State { app:"a".into(), shaders:"s".into(), eboot:"e".into(), translation:"t".into(), selection:";".into(), completed:vec!["newgame".into(),"compile".into(),"tour-1".into()], driver:None };
        s.invalidate_pipelines();
        assert_eq!(s.completed, ["newgame"]);
    }
}
