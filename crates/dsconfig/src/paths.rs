//! Where the port keeps its files.
//!
//! - Development checkout (the executable in the build output `<root>\target\...` of a folder
//!   holding `local-data\`, the workspace's `Cargo.toml` and `crates\`): everything in
//!   `<root>\local-data` (not in Git), logs in `local-data\runs`, the game in `<root>\Game\...`
//!   unless another folder is chosen. A package elsewhere in the checkout (its `dist\`) is an
//!   install: 2026-11-15, one run from there took the development layout, whose GPU work is off
//!   unless the run sets DS_GPU=1, and showed a black screen.
//! - Portable (`portable.txt` beside the executable, or `DSPORT_HOME=<folder>`): everything in
//!   `<exe folder>\user` (or that folder).
//! - Installed (anything else): settings, key bindings, saves and their backups in
//!   `%APPDATA%\Regaliant`; logs, prepared shaders and the game's own runtime folders in
//!   `%LOCALAPPDATA%\Regaliant`.

use std::path::{Path, PathBuf};

/// The folder name under `%APPDATA%` and `%LOCALAPPDATA%`.
pub const APP_FOLDER: &str = "Regaliant";
pub const PORTABLE_MARKER: &str = "portable.txt";
/// The runtime's and the launcher's file names as shipped (the package names the development
/// build's runtime, `GAME_EXE_DEV`, `GAME_EXE`).
pub const GAME_EXE: &str = "regaliant.exe";
pub const GAME_EXE_DEV: &str = "dsport.exe";
pub const LAUNCHER_EXE: &str = "regaliant-launcher.exe";

/// The runtime in the folder `dir` (the launcher's): `GAME_EXE`, or in a development build's
/// output folder `GAME_EXE_DEV`.
pub fn game_exe_in(dir: &Path) -> PathBuf {
    let shipped = dir.join(GAME_EXE);
    if !shipped.exists() && dir.join(GAME_EXE_DEV).exists() {
        return dir.join(GAME_EXE_DEV);
    }
    shipped
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Layout {
    Dev,
    Portable,
    Installed,
}

#[derive(Clone, Debug)]
pub struct Paths {
    pub layout: Layout,
    /// The executable's folder.
    pub exe_dir: PathBuf,
    /// Settings, key bindings, saves and their backups.
    pub user: PathBuf,
    /// Logs, prepared shaders, the game's runtime folders.
    pub cache: PathBuf,
    /// The development checkout (`Layout::Dev` only).
    pub root: Option<PathBuf>,
}

/// The checkout whose build holds `dir`: a folder holding `local-data\`, `Cargo.toml` and
/// `crates\` together (a stray folder of one name above a player's folder is not one), `dir`
/// under its `target\` (cargo's output) or `local-data\` (the run scripts' copy of the build,
/// `local-data\bin`): a package copied into the checkout's `dist\` is an install. (On
/// 2026-11-15 a rule of `target\` alone ran the scripts' copy as an install, in the user's
/// folders.)
fn dev_root(dir: &Path) -> Option<PathBuf> {
    let mut p = Some(dir);
    while let Some(d) = p {
        if d.join("local-data").is_dir() && d.join("Cargo.toml").is_file() && d.join("crates").is_dir() {
            return (dir.starts_with(d.join("target")) || dir.starts_with(d.join("local-data"))).then(|| d.to_path_buf());
        }
        p = d.parent();
    }
    None
}

impl Paths {
    pub fn for_exe(exe: &Path) -> Paths {
        let exe_dir = exe.parent().map(Path::to_path_buf).unwrap_or_default();
        let portable = |dir: PathBuf| Paths { layout: Layout::Portable, exe_dir: exe_dir.clone(), user: dir.clone(), cache: dir, root: None };
        if let Some(home) = std::env::var_os("DSPORT_HOME").filter(|h| !h.is_empty()) {
            return portable(PathBuf::from(home));
        }
        if exe_dir.join(PORTABLE_MARKER).is_file() {
            return portable(exe_dir.join("user"));
        }
        if let Some(root) = dev_root(&exe_dir) {
            let data = root.join("local-data");
            return Paths { layout: Layout::Dev, exe_dir, user: data.clone(), cache: data, root: Some(root) };
        }
        Paths::installed(&exe_dir)
    }

    /// The installed layout's folders for a program in `exe_dir`.
    pub fn installed(exe_dir: &Path) -> Paths {
        let env_dir = |name: &str| std::env::var_os(name).map(PathBuf::from).unwrap_or_else(|| exe_dir.to_path_buf());
        Paths {
            layout: Layout::Installed,
            exe_dir: exe_dir.to_path_buf(),
            user: env_dir("APPDATA").join(APP_FOLDER),
            cache: env_dir("LOCALAPPDATA").join(APP_FOLDER),
            root: None,
        }
    }

    pub fn current() -> Paths {
        Paths::for_exe(&std::env::current_exe().unwrap_or_default())
    }

    /// A player's diagnostic switches (`DS_NAME=value` lines the game sets in its environment at
    /// its start; not read in a checkout, whose runs set their own).
    pub fn diagnostics_file(&self) -> Option<PathBuf> {
        (self.layout != Layout::Dev).then(|| self.user.join("diagnostics.txt"))
    }

    /// The PC settings (the game's PC tab), `audio-device.txt` beside it.
    pub fn settings_file(&self) -> PathBuf {
        self.user.join("pcsettings.txt")
    }

    /// Window mode's monitor, resolution and refresh rate.
    pub fn display_file(&self) -> PathBuf {
        self.user.join("display.txt")
    }

    /// The keyboard and mouse layout.
    pub fn controls_file(&self) -> PathBuf {
        self.user.join("controls.txt")
    }

    pub fn achievements_file(&self) -> PathBuf {
        self.user.join("achievements.txt")
    }

    /// The game folder the player chose (one line).
    pub fn game_file(&self) -> PathBuf {
        self.user.join("game.txt")
    }

    /// The launcher's options for the next start (`key = value` lines).
    pub fn launch_file(&self) -> PathBuf {
        self.user.join("launch.txt")
    }

    /// The saves when the PC tab's Save Folder is the port's own.
    pub fn saves(&self) -> PathBuf {
        self.user.join("saves")
    }

    /// Copies of earlier saves, newest last by name.
    pub fn save_backups(&self) -> PathBuf {
        self.user.join("save-backups")
    }

    /// The save folder in use (one line; the PC tab's Save Folder).
    pub fn save_folder_record(&self) -> PathBuf {
        self.user.join("save-folder.txt")
    }

    /// One folder per run (log, crash report, screenshots).
    pub fn logs(&self) -> PathBuf {
        match self.layout {
            Layout::Dev => self.cache.join("runs"),
            _ => self.cache.join("logs"),
        }
    }

    /// The run folders of the launcher's quick checks (the pipeline store's probe at its start,
    /// the translation check): a `checks` folder in the logs, so that the newest folder there is
    /// the newest session's (a player who reopened the launcher to fetch the logs sent the
    /// probe's half-second log instead, 2026-11-24). In a checkout: the runs, as before.
    pub fn check_logs(&self) -> PathBuf {
        match self.layout {
            Layout::Dev => self.logs(),
            _ => self.logs().join("checks"),
        }
    }

    /// How many run folders are kept (a development run's KEEP mark keeps one apart).
    pub fn logs_kept(&self) -> usize {
        match self.layout {
            Layout::Dev => 100,
            // (One shader preparation alone makes 8-10 sessions' folders, retries more: with 10
            // a player's report lost its first steps, the review of 2026-11-20.)
            _ => 40,
        }
    }

    /// The game's own writable folders (temp0, data, download0, devlog).
    pub fn runtime(&self) -> PathBuf {
        self.cache.join("runtime")
    }

    /// Compiled pipelines, translated shaders and learned shader specs.
    pub fn shaders(&self) -> PathBuf {
        match self.layout {
            Layout::Dev => self.cache.clone(),
            _ => self.cache.join("shaders"),
        }
    }

    /// The upscalers' libraries (DLSS needs the player's own `nvngx_dlss.dll` there).
    pub fn upscalers(&self) -> PathBuf {
        match &self.root {
            Some(r) => r.join("local-data").join("upscalers"),
            None => self.exe_dir.join("upscalers"),
        }
    }

    /// Marks a running game: one left behind means the last session did not end normally.
    pub fn running_marker(&self) -> PathBuf {
        self.cache.join("running.txt")
    }

    /// The game folder: the chosen one (`game.txt`), else the checkout's own copy.
    pub fn chosen_game(&self) -> Option<PathBuf> {
        if let Ok(t) = std::fs::read_to_string(self.game_file()) {
            let t = t.trim().trim_matches('"');
            if !t.is_empty() {
                return Some(PathBuf::from(t));
            }
        }
        self.root.as_ref().map(|r| {
            let game = r.join("Game");
            let eu = game.join("Demon-Souls-Remake-PPSA01341-1.07").join(crate::game::APP_DIR);
            let us = game.join("Demon-Souls-Remake-PPSA01342-1.07").join(crate::game::APP_DIR_US);
            if eu.is_dir() { eu } else if us.is_dir() { us } else { eu }
        })
    }

    /// Keeps `dir` as the game folder (`game.txt`).
    pub fn set_game(&self, dir: &Path) -> std::io::Result<()> {
        std::fs::create_dir_all(&self.user)?;
        std::fs::write(self.game_file(), format!("{}\n", dir.display()))
    }
}

/// Reads `key = value` lines (`#` comments, blank lines skipped).
pub fn read_kv(path: &Path) -> Vec<(String, String)> {
    let Ok(t) = std::fs::read_to_string(path) else { return Vec::new() };
    t.lines()
        .filter(|l| !l.trim_start().starts_with('#'))
        .filter_map(|l| l.split_once('='))
        .map(|(k, v)| (k.trim().to_string(), v.trim().to_string()))
        .filter(|(k, _)| !k.is_empty())
        .collect()
}

/// The launcher's options for the next start.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct Launch {
    /// Safe mode for the next start only: a window, the default PC settings, no upscaler.
    pub safe_mode: bool,
    /// Prepare the game's shaders before the first start (the launcher's job).
    pub prepare_shaders: bool,
}

impl Launch {
    pub fn load(p: &Paths) -> Launch {
        let kv = read_kv(&p.launch_file());
        let get = |k: &str| kv.iter().find(|(x, _)| x == k).map(|(_, v)| v.as_str());
        Launch { safe_mode: get("safe_mode") == Some("1"), prepare_shaders: get("prepare_shaders") != Some("0") }
    }

    pub fn save(&self, p: &Paths) -> std::io::Result<()> {
        std::fs::create_dir_all(&p.user)?;
        let t = format!(
            "# The launcher's options for the next start; written by the launcher\nsafe_mode = {}\nprepare_shaders = {}\n",
            self.safe_mode as u8, self.prepare_shaders as u8
        );
        std::fs::write(p.launch_file(), t)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn layouts() {
        let tmp = std::env::temp_dir().join(format!("dsconfig-paths-{}", std::process::id()));
        let dev = tmp.join("checkout");
        std::fs::create_dir_all(dev.join("local-data")).unwrap();
        std::fs::create_dir_all(dev.join("crates")).unwrap();
        std::fs::create_dir_all(dev.join("target").join("release")).unwrap();
        std::fs::write(dev.join("Cargo.toml"), "").unwrap();
        // (DSPORT_HOME would override every layout.)
        if std::env::var_os("DSPORT_HOME").is_none() {
            let p = Paths::for_exe(&dev.join("target").join("release").join(GAME_EXE));
            assert_eq!(p.layout, Layout::Dev);
            assert_eq!(p.user, dev.join("local-data"));
            assert_eq!(p.logs(), dev.join("local-data").join("runs"));
            assert_eq!(p.shaders(), dev.join("local-data"));
            assert_eq!(p.upscalers(), dev.join("local-data").join("upscalers"));

            // (The run scripts' copy of the build in local-data\bin: the development layout.)
            let bin = dev.join("local-data").join("bin");
            std::fs::create_dir_all(&bin).unwrap();
            let p = Paths::for_exe(&bin.join(GAME_EXE_DEV));
            assert_eq!(p.layout, Layout::Dev);
            // (A package in the checkout's dist\: an install, not the development layout.)
            let pkg = dev.join("dist").join("Regaliant-0.1.0");
            std::fs::create_dir_all(&pkg).unwrap();
            let p = Paths::for_exe(&pkg.join(GAME_EXE));
            assert_eq!(p.layout, Layout::Installed);

            let rel = tmp.join("Games").join("Regaliant");
            std::fs::create_dir_all(&rel).unwrap();
            // (A stray local-data folder above a player's folder.)
            std::fs::create_dir_all(tmp.join("local-data")).unwrap();
            let p = Paths::for_exe(&rel.join(GAME_EXE));
            assert_eq!(p.layout, Layout::Installed);
            assert!(p.user.ends_with(APP_FOLDER) && p.cache.ends_with(APP_FOLDER));
            assert_eq!(p.upscalers(), rel.join("upscalers"));

            std::fs::write(rel.join(PORTABLE_MARKER), "").unwrap();
            let p = Paths::for_exe(&rel.join(GAME_EXE));
            assert_eq!(p.layout, Layout::Portable);
            assert_eq!(p.user, rel.join("user"));
            assert_eq!(p.logs(), rel.join("user").join("logs"));
            assert_eq!(p.shaders(), rel.join("user").join("shaders"));
        }
        let _ = std::fs::remove_dir_all(&tmp);
    }

    #[test]
    fn launch_options() {
        let tmp = std::env::temp_dir().join(format!("dsconfig-launch-{}", std::process::id()));
        let p = Paths { layout: Layout::Portable, exe_dir: tmp.clone(), user: tmp.clone(), cache: tmp.clone(), root: None };
        assert_eq!(Launch::load(&p), Launch { safe_mode: false, prepare_shaders: true });
        let l = Launch { safe_mode: true, prepare_shaders: false };
        l.save(&p).unwrap();
        assert_eq!(Launch::load(&p), l);
        let _ = std::fs::remove_dir_all(&tmp);
    }
}
