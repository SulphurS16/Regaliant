//! The save folder (Game & Controls: Save Folder) and the import of saves made elsewhere, both
//! applied when the port starts: the title keeps its saves open while it runs.
//!
//! A changed folder that holds no saves of the game gets a copy of those in the folder used
//! before (save-folder.txt in the settings' folder names it), so the progress moves along; the
//! old folder is left as it was. Then the saves a player copied into the save folder in another
//! layout (another program's `_SaveData`, a title's folder without the user folder above it:
//! `dsconfig::saves::misplaced`) are imported and their folder renamed "<name> (imported
//! <time>)". Import PS5 Saves takes the saves in the folder "PS5 Saves" (`import_dir`): a
//! decrypted PS5 save or another program's save folder (`dsconfig::saves`), keeps the saves it
//! replaces in save-backups, and renames the folder when done.

use std::path::{Path, PathBuf};

/// The Save Folder choice's options.
pub const FOLDERS: [&str; 3] = ["Game Folder", "Documents", "Saved Games"];
/// The folder the import reads (`import_dir`).
pub const IMPORT_DIR: &str = "PS5 Saves";

/// Where the import looks: the checkout's root in development, else the settings' folder.
pub fn import_dir(paths: &dsconfig::paths::Paths) -> PathBuf {
    paths.root.as_ref().unwrap_or(&paths.user).join(IMPORT_DIR)
}

fn known_folder(id: &windows::core::GUID) -> Option<PathBuf> {
    use windows::Win32::UI::Shell::{KF_FLAG_DEFAULT, SHGetKnownFolderPath};
    unsafe {
        let p = SHGetKnownFolderPath(id, KF_FLAG_DEFAULT, None).ok()?;
        let s = p.to_string().ok();
        windows::Win32::System::Com::CoTaskMemFree(Some(p.0 as *const _));
        s.map(PathBuf::from)
    }
}

/// The saves' folder of a Save Folder option (0, or a known folder Windows does not give: the
/// port's own, `port`).
pub fn folder(option: u32, port: &Path) -> PathBuf {
    use windows::Win32::UI::Shell::{FOLDERID_Documents, FOLDERID_SavedGames};
    let port = port.to_path_buf();
    match option {
        1 => known_folder(&FOLDERID_Documents).map(|d| d.join("My Games").join("Demon's Souls").join("saves")).unwrap_or(port),
        2 => known_folder(&FOLDERID_SavedGames).map(|d| d.join("Demon's Souls").join("saves")).unwrap_or(port),
        _ => port,
    }
}

fn has_saves(saves: &Path) -> bool {
    !dsconfig::saves::find(saves).is_empty()
}

fn copy_dir(from: &Path, to: &Path) -> std::io::Result<u64> {
    std::fs::create_dir_all(to)?;
    let mut n = 0;
    for e in std::fs::read_dir(from)? {
        let e = e?;
        let target = to.join(e.file_name());
        if e.file_type()?.is_dir() {
            n += copy_dir(&e.path(), &target)?;
        } else {
            std::fs::copy(e.path(), &target)?;
            n += 1;
        }
    }
    Ok(n)
}

/// The saves' folder for this start (after `super::init`): the Save Folder setting's, with
/// the saves of the folder used before copied in when it has none; then a pending import.
pub fn start(paths: &dsconfig::paths::Paths, title: &str) -> PathBuf {
    let option = super::store::value("Setting_PC_SaveFolder").unwrap_or(0);
    let dir = folder(option, &paths.saves());
    let record = paths.save_folder_record();
    let before = std::fs::read_to_string(&record).ok().map(|s| PathBuf::from(s.trim())).filter(|p| !p.as_os_str().is_empty());
    if let Some(prev) = before.filter(|p| *p != dir) {
        if !has_saves(&dir) && has_saves(&prev) {
            match copy_dir(&prev, &dir) {
                Ok(n) => crate::info!("saves: the save folder is now {}: {n} files of the saves copied from {}", dir.display(), prev.display()),
                Err(e) => crate::warn!("saves: copying the saves from {} to {} failed: {e}", prev.display(), dir.display()),
            }
        } else {
            crate::info!("saves: the save folder is now {} (was {})", dir.display(), prev.display());
        }
    }
    if let Err(e) = std::fs::create_dir_all(&dir) {
        crate::warn!("saves: cannot make the save folder {}: {e}", dir.display());
    }
    let _ = std::fs::write(&record, format!("{}\n", dir.display()));
    // Saves a player copied into the save folder in another layout (2026-11-26 evening: saves
    // from another program put there by hand were not seen: the port reads
    // `<saves>\<user>\<title>\<save>`).
    for m in dsconfig::saves::misplaced(&dir, title) {
        if import(&m, &dir, title, "saves copied into the save folder") > 0 {
            let stamp = crate::utc_stamp();
            let to = m.with_file_name(format!("{} (imported {stamp})", m.file_name().unwrap_or_default().to_string_lossy()));
            match std::fs::rename(&m, &to) {
                Ok(()) => crate::info!("saves: {} renamed to {}", m.display(), to.display()),
                Err(e) => crate::warn!("saves: renaming {} failed: {e}", m.display()),
            }
        }
    }
    if super::store::value("Setting_PC_ImportSaves") == Some(1) {
        let from = import_dir(paths);
        if import(&from, &dir, title, "Import PS5 Saves") > 0 {
            let stamp = crate::utc_stamp();
            let to = from.with_file_name(format!("{} (imported {stamp})", from.file_name().unwrap_or_default().to_string_lossy()));
            match std::fs::rename(&from, &to) {
                Ok(()) => crate::info!("saves: {} renamed to {}", from.display(), to.display()),
                Err(e) => crate::warn!("saves: renaming {} failed: {e}", from.display()),
            }
        }
        super::store::set_value("Setting_PC_ImportSaves", 0);
    }
    dir
}

/// Imports the saves found in `from` into `saves` (`dsconfig::saves::import`: the current ones
/// kept in save-backups first) and logs what it did (`why`: what asked for it). Returns how many.
pub fn import(from: &Path, saves: &Path, title: &str, why: &str) -> usize {
    let r = dsconfig::saves::import(from, saves, title, &crate::utc_stamp());
    for k in &r.kept {
        crate::info!("saves: import ({why}): the current save kept in {}", k.display());
    }
    for (name, src, n) in &r.imported {
        crate::info!("saves: import ({why}): {name} imported from {} ({n} files)", src.display());
    }
    for note in &r.notes {
        crate::warn!("saves: import ({why}): {note}");
    }
    r.imported.len()
}

#[cfg(test)]
mod tests {
    #[test]
    fn the_import_writes_where_the_game_reads() {
        // (The save-data module's paths are `<saves>\<user id>\<title>`: the import's user folder
        // is that id's.)
        assert_eq!(crate::hle::libs::app::USER_ID.to_string(), dsconfig::saves::USER);
    }
}
