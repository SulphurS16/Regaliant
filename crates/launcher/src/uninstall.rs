//! Uninstall: removes what Regaliant added to this PC, asking first, and the saves only when the
//! player says so (issue #19 and the user's request of 2026-11-25).
//!
//! - Removed: the program files the package holds, by name only (`PACKAGE_FILES`: a player may
//!   have put the package in a folder with other files, even the game's), the prepared shaders,
//!   logs and the game's runtime folders (the cache folder), and the settings (the user folder).
//! - Kept always: the game's folder and its .zar archive (Regaliant only reads them), add-on
//!   packages ("Add-ons" folders: the player's own), the player's own `nvngx_dlss.dll`.
//! - Saves (each Save Folder's saves, the save backups, the PS5 Saves import folder, the trophy
//!   progress): kept unless the player chooses to delete them.
//!
//! The launcher's own program file cannot be deleted while it runs: the player is shown its folder
//! to delete. Until 0.1.9 a hidden command deleted it after the launcher closed (`cmd /c ping ... &
//! del ...`), the pattern antivirus programs flag as self-deleting malware: Kaspersky reported
//! 0.1.8's launcher as "selfdel" (issue #11's comments, 2026-11-26).

use crate::{App, TITLE};
use dsconfig::paths::{Layout, Paths};
use std::path::{Path, PathBuf};
use windows_sys::Win32::UI::WindowsAndMessaging::*;

/// The files a package holds besides the launcher itself (`scripts\package.ps1`'s allowlist), and
/// the portable layout's marker.
pub(crate) const PACKAGE_FILES: &[&str] = &[
    "regaliant.exe",
    "README.txt",
    "portable.txt",
    "upscalers\\libxess.dll",
    "upscalers\\libxess.license.txt",
    "upscalers\\libxess.third-party-programs.txt",
    "upscalers\\sl.interposer.dll",
    "upscalers\\sl.common.dll",
    "upscalers\\sl.dlss.dll",
    "upscalers\\streamline.license.txt",
    "upscalers\\streamline.3rd-party-licenses.md",
    "upscalers\\amd_fidelityfx_loader_vk.dll",
    "upscalers\\amd_fidelityfx_upscaler_vk.dll",
    "upscalers\\fidelityfx.license.md",
];

/// Names in the settings folder kept with the saves (unless the player deletes them), as name
/// prefixes: the PS5 import folder is renamed "PS5 Saves (imported <time>)" after an import and
/// holds the player's own decrypted PS5 saves (the review of 2026-11-25 night).
const SAVE_ENTRIES: &[&str] = &["saves", "save-backups", "PS5 Saves", "achievements.txt", "save-folder.txt"];
/// Of those, the ones that hold saves (the question is asked only when one of them has any).
const SAVE_HOLDERS: &[&str] = &["saves", "save-backups", "PS5 Saves"];
/// Add-on packages: the player's own, always kept.
const ADDONS: &str = "Add-ons";

fn known_folder(id: &windows::core::GUID) -> Option<PathBuf> {
    use windows::Win32::UI::Shell::{KF_FLAG_DEFAULT, SHGetKnownFolderPath};
    unsafe {
        let p = SHGetKnownFolderPath(id, KF_FLAG_DEFAULT, None).ok()?;
        let s = p.to_string().ok();
        windows::Win32::System::Com::CoTaskMemFree(Some(p.0 as *const _));
        s.map(PathBuf::from)
    }
}

/// The saves folders of the Save Folder's other two choices (Documents, Saved Games) that this
/// install used: the one its save-folder.txt names. Another install may keep its saves in the
/// other one, and a folder this install never used is never touched.
fn other_save_folders(paths: &Paths) -> Vec<PathBuf> {
    use windows::Win32::UI::Shell::{FOLDERID_Documents, FOLDERID_SavedGames};
    let used = std::fs::read_to_string(paths.save_folder_record()).unwrap_or_default();
    let used = used.trim().trim_end_matches(['\\', '/']).to_lowercase();
    let mut v = Vec::new();
    if let Some(d) = known_folder(&FOLDERID_Documents) {
        v.push(d.join("My Games").join("Demon's Souls").join("saves"));
    }
    if let Some(d) = known_folder(&FOLDERID_SavedGames) {
        v.push(d.join("Demon's Souls").join("saves"));
    }
    v.retain(|f| !used.is_empty() && f.to_string_lossy().trim_end_matches(['\\', '/']).to_lowercase() == used);
    v
}

/// Bytes under `p` (files only).
fn size_of(p: &Path) -> u64 {
    let Ok(rd) = std::fs::read_dir(p) else { return std::fs::metadata(p).map_or(0, |m| m.len()) };
    rd.flatten()
        .map(|e| match e.file_type() {
            Ok(t) if t.is_dir() => size_of(&e.path()),
            Ok(_) => e.metadata().map_or(0, |m| m.len()),
            Err(_) => 0,
        })
        .sum()
}

fn human(b: u64) -> String {
    if b >= 1 << 30 { format!("{:.1} GB", b as f64 / (1u64 << 30) as f64) } else { format!("{} MB", (b >> 20).max(1)) }
}

/// Removes `p` (a file or a folder with all it holds); failures are collected.
fn remove(p: &Path, failed: &mut Vec<String>) {
    let r = if p.is_dir() { std::fs::remove_dir_all(p) } else { std::fs::remove_file(p) };
    if let Err(e) = r {
        if p.exists() {
            failed.push(format!("{}: {e}", p.display()));
        }
    }
}

/// Whether entry `name` starts with one of `keep` (case-insensitive).
fn kept_name(name: &str, keep: &[&str]) -> bool {
    keep.iter().any(|k| name.len() >= k.len() && name[..k.len()].eq_ignore_ascii_case(k))
}

/// Whether `p` must stay whatever folder it is in: a game archive (.zar), or the game folder the
/// launcher uses or a folder holding it (a player may have put the game inside Regaliant's own
/// folders; the review of 2026-11-25 night).
fn protected(p: &Path, game: Option<&Path>) -> bool {
    if p.extension().is_some_and(|e| e.eq_ignore_ascii_case("zar")) {
        return true;
    }
    // (Game files by what they hold, whatever the records say: game.txt may be gone already.)
    if p.is_dir() && holds_game(p, 4) {
        return true;
    }
    // (Compared lowercase with one separator: paths may mix / and \.)
    let norm = |q: &Path| q.to_string_lossy().to_lowercase().replace('/', "\\").trim_end_matches('\\').to_string();
    game.is_some_and(|g| {
        let (g, p) = (norm(g), norm(p));
        g == p || g.starts_with(&format!("{p}\\"))
    })
}

/// Whether folder `dir` holds a game's files within `depth` levels: an `eboot.bin`, an `sce_sys`
/// folder or a .zar archive.
fn holds_game(dir: &Path, depth: u32) -> bool {
    let Ok(rd) = std::fs::read_dir(dir) else { return false };
    for e in rd.flatten() {
        let name = e.file_name().to_string_lossy().to_lowercase();
        let is_dir = e.file_type().is_ok_and(|t| t.is_dir());
        if (!is_dir && (name == "eboot.bin" || name.ends_with(".zar"))) || (is_dir && name == "sce_sys") {
            return true;
        }
        if is_dir && depth > 1 && holds_game(&e.path(), depth - 1) {
            return true;
        }
    }
    false
}

/// Removes the entries of `dir` except those `keep` names (prefixes) and the protected ones, then
/// `dir` itself when that left it empty.
fn clear_folder(dir: &Path, keep: &[&str], game: Option<&Path>, failed: &mut Vec<String>) {
    let Ok(rd) = std::fs::read_dir(dir) else { return };
    for e in rd.flatten() {
        let name = e.file_name().to_string_lossy().into_owned();
        if kept_name(&name, keep) || protected(&e.path(), game) {
            continue;
        }
        remove(&e.path(), failed);
    }
    let _ = std::fs::remove_dir(dir);
}

/// Whether any of `places` (folders) holds anything.
fn saves_found(places: &[PathBuf]) -> bool {
    places.iter().any(|p| std::fs::read_dir(p).is_ok_and(|mut rd| rd.next().is_some()))
}

/// Whether `p` is one of this layout's own folders: in the installed layout a folder named after
/// the app, in the portable one the `user` folder beside the programs. Nothing else is cleared.
fn ours(paths: &Paths, p: &Path) -> bool {
    match paths.layout {
        Layout::Installed => p.file_name().is_some_and(|n| n.eq_ignore_ascii_case(dsconfig::paths::APP_FOLDER)),
        Layout::Portable => p.parent() == Some(paths.exe_dir.as_path()) && p.file_name().is_some_and(|n| n.eq_ignore_ascii_case("user")),
        Layout::Dev => false,
    }
}

/// The folders that hold saves: the settings folder's (the PS5 import folder by its prefix) and
/// `others` (`other_save_folders`).
fn save_places(paths: &Paths, others: &[PathBuf]) -> Vec<PathBuf> {
    let mut v: Vec<PathBuf> = Vec::new();
    if let Ok(rd) = std::fs::read_dir(&paths.user) {
        for e in rd.flatten() {
            let name = e.file_name().to_string_lossy().into_owned();
            if kept_name(&name, SAVE_HOLDERS) && e.file_type().is_ok_and(|t| t.is_dir()) {
                v.push(e.path());
            }
        }
    }
    v.extend(others.iter().cloned());
    v
}

/// The Uninstall button.
pub fn run(app: &mut App) {
    let paths = app.paths.clone();
    if paths.layout == Layout::Dev {
        unsafe { crate::ui::message(app.hwnd, "This copy runs from a development checkout. Uninstall is for installed copies.", TITLE, MB_ICONINFORMATION | MB_OK) };
        return;
    }
    if app.running.is_some() {
        app.set_status("Close the game first.");
        return;
    }
    if crate::shaders::running(app) || crate::game::compressing(app) {
        app.set_status("Stop the shader preparation or the compression first.");
        return;
    }
    if crate::update::downloading(app) {
        app.set_status("An update is being downloaded. Uninstall when it is done.");
        return;
    }
    let data = human(size_of(&paths.cache) + if paths.user != paths.cache { size_of(&paths.user) } else { 0 });
    let text = format!("Uninstall Regaliant from this PC?\n\nThis removes its program files, its prepared shaders, logs and settings ({data}).\n\nYour game files and add-ons are not touched.");
    if unsafe { crate::ui::message(app.hwnd, &text, TITLE, MB_ICONWARNING | MB_OKCANCEL | MB_DEFBUTTON2) } != IDOK {
        return;
    }
    let others = other_save_folders(&paths);
    let mut delete_saves = false;
    if saves_found(&save_places(&paths, &others)) {
        let text = "Delete your saves too?\n\nYes deletes them for good.\nNo keeps them, so a later install continues where you left off.";
        match unsafe { crate::ui::message(app.hwnd, text, TITLE, MB_ICONQUESTION | MB_YESNOCANCEL | MB_DEFBUTTON2) } {
            IDYES => delete_saves = true,
            IDNO => {}
            _ => return,
        }
    }
    let (failed, kept) = perform(&paths, &others, delete_saves);
    let mut text = if failed.is_empty() { "Regaliant was removed.".to_string() } else { format!("Some files could not be removed:\n{}", failed.join("\n")) };
    text.push_str(&format!("\n\nTo finish, delete the folder it ran from once this window has closed:\n{}\nThe launcher cannot delete its own file while it runs.", paths.exe_dir.display()));
    if !kept.is_empty() {
        text.push_str("\n\n");
        text.push_str(&kept.join("\n"));
    }
    unsafe { crate::ui::message(app.hwnd, &text, TITLE, if failed.is_empty() { MB_ICONINFORMATION | MB_OK } else { MB_ICONWARNING | MB_OK }) };
    show_folder(&paths.exe_dir);
    unsafe { PostMessageW(app.hwnd, WM_CLOSE, 0, 0) };
}

/// The removal itself (`run` asked first; `others`: `other_save_folders`): what failed and what
/// was kept, as lines for the player.
fn perform(paths: &Paths, others: &[PathBuf], delete_saves: bool) -> (Vec<String>, Vec<String>) {
    let (user, cache) = (&paths.user, &paths.cache);
    let mut failed = Vec::new();
    // (The game folder the launcher uses, read before its record goes: never removed.)
    let game = paths.chosen_game();
    let game = game.as_deref();
    // Prepared shaders, logs, the runtime folders (add-ons kept).
    if ours(paths, cache) && cache != user {
        clear_folder(cache, &[ADDONS], game, &mut failed);
    }
    // Settings, and the saves when asked (add-ons kept).
    if ours(paths, user) {
        let mut keep = vec![ADDONS];
        if !delete_saves {
            keep.extend_from_slice(SAVE_ENTRIES);
        }
        clear_folder(user, &keep, game, &mut failed);
    }
    if delete_saves {
        for f in others {
            if f.is_dir() && !protected(f, game) {
                remove(f, &mut failed);
                // (The backups the game keeps beside them, then their "Demon's Souls" folder
                // when nothing else is in it.)
                if let Some(parent) = f.parent() {
                    let backups = parent.join("save-backups");
                    if backups.is_dir() && !protected(&backups, game) {
                        remove(&backups, &mut failed);
                    }
                    let _ = std::fs::remove_dir(parent);
                }
            }
        }
    }
    // The program files, by name.
    for f in PACKAGE_FILES {
        let p = paths.exe_dir.join(f);
        if p.is_file() {
            remove(&p, &mut failed);
        }
    }
    // The update's own folders (earlier versions' backups, unfinished updates) and its journal.
    for p in crate::update::own_files(&paths.exe_dir) {
        remove(&p, &mut failed);
    }
    let up = paths.exe_dir.join("upscalers");
    let _ = std::fs::remove_dir(&up);
    let mut kept = Vec::new();
    if !delete_saves {
        let places: Vec<PathBuf> = save_places(paths, others).into_iter().filter(|p| saves_found(std::slice::from_ref(p))).collect();
        if let Some(p) = places.iter().find(|p| others.contains(p)).or(places.first()) {
            kept.push(format!("Your saves are kept in {}.", p.display()));
        }
    }
    if up.join("nvngx_dlss.dll").is_file() {
        kept.push(format!("Your own DLSS file is kept in {}.", up.display()));
    }
    (failed, kept)
}

/// The package's folder in File Explorer, selected in its parent, for the player to delete once
/// the launcher has closed.
fn show_folder(exe_dir: &Path) {
    use std::os::windows::process::CommandExt;
    let _ = std::process::Command::new("explorer.exe").raw_arg(format!("/select,\"{}\"", exe_dir.display())).spawn();
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A portable install in a temporary folder: everything Regaliant adds goes, the saves only
    /// when asked, while add-ons, the player's DLSS file, a game folder, an archive and another
    /// file beside the programs stay.
    #[test]
    fn removes_only_its_own() {
        let root = std::env::temp_dir().join(format!("regaliant-uninstall-test-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&root);
        let put = |p: &str| {
            let f = root.join(p);
            std::fs::create_dir_all(f.parent().unwrap()).unwrap();
            std::fs::write(f, b"x").unwrap();
        };
        let programs = ["regaliant.exe", "regaliant-launcher.exe", "README.txt", "portable.txt", "upscalers/libxess.dll", "upscalers/nvngx_dlss.dll", "upscalers/sl.dlss.dll"];
        // (Inside the settings folder too: the renamed PS5 import folder, an archive and the game
        // folder the launcher uses, which a player may have put there.)
        let data = [
            "user/pcsettings.txt",
            "user/shaders/a.bin",
            "user/logs/x/dsport.log",
            "user/saves/u/SAVEDATA0PlayerProfile0/USR-DATA",
            "user/save-backups/b",
            "user/Add-ons/pkg",
            "user/PS5 Saves (imported 20261125-120000)/SAVEDATA0PlayerProfile0/USR-DATA",
            "user/mygame.zar",
            "user/games/PPSA01341-app0/eboot.bin",
        ];
        let others = ["PPSA01341-app0/eboot.bin", "game.zar", "notes.txt"];
        for f in programs.iter().chain(&data).chain(&others) {
            put(f);
        }
        std::fs::write(root.join("user/game.txt"), root.join("user/games/PPSA01341-app0").to_string_lossy().as_bytes()).unwrap();
        let paths = Paths::for_exe(&root.join("regaliant-launcher.exe"));
        assert_eq!(paths.layout, Layout::Portable);
        // (A Save Folder choice stands in for Documents' inside the test's own folder: the PC's own
        // folders are never touched by a test. Its backups beside it go with it.)
        put("docs/My Games/Demon's Souls/saves/u/SAVEDATA0PlayerProfile0/USR-DATA");
        put("docs/My Games/Demon's Souls/save-backups/b");
        let others = vec![root.join("docs/My Games/Demon's Souls/saves")];
        assert!(saves_found(&save_places(&paths, &others)));
        let (failed, kept) = perform(&paths, &others, false);
        assert!(failed.is_empty(), "{failed:?}");
        for gone in ["regaliant.exe", "README.txt", "portable.txt", "upscalers/libxess.dll", "upscalers/sl.dlss.dll", "user/pcsettings.txt", "user/shaders", "user/logs", "user/game.txt"] {
            assert!(!root.join(gone).exists(), "{gone} left");
        }
        let always = ["upscalers/nvngx_dlss.dll", "user/Add-ons/pkg", "user/mygame.zar", "user/games/PPSA01341-app0/eboot.bin", "PPSA01341-app0/eboot.bin", "game.zar", "notes.txt", "regaliant-launcher.exe"];
        for stays in always.iter().chain(&["user/saves/u/SAVEDATA0PlayerProfile0/USR-DATA", "user/save-backups/b", "user/PS5 Saves (imported 20261125-120000)/SAVEDATA0PlayerProfile0/USR-DATA"]) {
            assert!(root.join(stays).exists(), "{stays} removed");
        }
        assert_eq!(kept.len(), 2, "{kept:?}");
        assert!(kept[0].contains("docs"), "the note names the Save Folder in use: {kept:?}");
        // Then the saves too; add-ons, archives and the game's files still stay.
        assert!(root.join("docs/My Games/Demon's Souls/saves/u/SAVEDATA0PlayerProfile0/USR-DATA").exists());
        let (failed, _) = perform(&paths, &others, true);
        assert!(failed.is_empty(), "{failed:?}");
        for gone in ["user/saves", "user/save-backups", "user/PS5 Saves (imported 20261125-120000)", "docs/My Games/Demon's Souls"] {
            assert!(!root.join(gone).exists(), "{gone} left");
        }
        for stays in always {
            assert!(root.join(stays).exists(), "{stays} removed");
        }
        // A settings folder with only its record of the save folder asks nothing.
        std::fs::create_dir_all(root.join("user")).unwrap();
        std::fs::write(root.join("user/save-folder.txt"), b"x").unwrap();
        assert!(!saves_found(&save_places(&paths, &[])));
        let _ = std::fs::remove_dir_all(&root);
    }
}
