//! Saves made elsewhere brought into the port's save folder: decrypted PS5 saves and the save
//! folders other programs keep. The game uses it at its start (Import PS5 Saves, and save folders a
//! player put straight into the save folder in another layout), the launcher from any folder the
//! player picks (Saves: Import saves...).
//!
//! The port keeps a save as `<saves>\<user>\<title>\<save>\`: the game's own files (this title
//! writes `USR-DATA`) and the port's metadata in `sce_sys\` (`param.bin`, `blocks`, `icon0.png`).
//! Elsewhere a save is a folder named `SAVEDATA...` holding a `USR-DATA` file, or
//! `sdimg_SAVEDATA...` (the name decrypting tools keep for the PS5's save images), at most `DEPTH`
//! folders down: a PS5 export (`PPSA0134x\...`) or another program's save folder
//! (`_SaveData\<title>\<save>`, a guest user's `_SaveData\[users]\<id>\<title>\<save>`, or that
//! program's own folder above it). Left out: the PS5's own backups beside its saves (`sdimg_sce_bu_...`) and files still being written (`*.tmp`).

use std::path::{Path, PathBuf};

/// The login user's folder in the save folder (the runtime's user id 1).
pub const USER: &str = "1";
/// How many folders down a save is looked for.
pub const DEPTH: u32 = 6;
/// The save parameters' size (title, subtitle, detail, user value, time: the port's
/// `sce_sys\param.bin`, and such a program's).
pub const PARAM_SIZE: u64 = 1328;

/// The game's name of a save folder: `SAVEDATA...` as it is, `sdimg_SAVEDATA...` without its
/// prefix; the PS5's backups (`sdimg_sce_bu_SAVEDATA...`) and anything else: None.
pub fn save_name(folder: &str) -> Option<&str> {
    let name = folder.strip_prefix("sdimg_").unwrap_or(folder);
    name.starts_with("SAVEDATA").then_some(name)
}

fn walk(dir: &Path, depth: u32, out: &mut Vec<PathBuf>) {
    let Ok(rd) = std::fs::read_dir(dir) else { return };
    for e in rd.flatten() {
        let p = e.path();
        if !p.is_dir() {
            continue;
        }
        let name = e.file_name().to_string_lossy().to_string();
        if save_name(&name).is_some() && p.join("USR-DATA").is_file() {
            out.push(p);
        } else if depth > 0 && !name.starts_with('.') && !name.starts_with("sdimg_sce_bu_") && !name.starts_with("sce_") {
            walk(&p, depth - 1, out);
        }
    }
}

/// The save folders under `dir`, in name order.
pub fn find(dir: &Path) -> Vec<PathBuf> {
    let mut v = Vec::new();
    walk(dir, DEPTH, &mut v);
    v.sort();
    v
}

/// One import's outcome.
#[derive(Default, Debug)]
pub struct Report {
    /// (the game's save name, the folder it came from, files copied).
    pub imported: Vec<(String, PathBuf, u64)>,
    /// The current saves kept before they were replaced.
    pub kept: Vec<PathBuf>,
    /// What was left out or failed, as lines for the log.
    pub notes: Vec<String>,
}

fn copy_tree(from: &Path, to: &Path) -> std::io::Result<u64> {
    std::fs::create_dir_all(to)?;
    let mut n = 0;
    for e in std::fs::read_dir(from)? {
        let e = e?;
        let name = e.file_name();
        if name.to_string_lossy().ends_with(".tmp") {
            continue;
        }
        let target = to.join(&name);
        if e.file_type()?.is_dir() {
            n += copy_tree(&e.path(), &target)?;
        } else {
            std::fs::copy(e.path(), &target)?;
            n += 1;
        }
    }
    Ok(n)
}

/// The metadata of a save that is new here: the port's own files of `src\sce_sys` (a parameter
/// file of the shared size, the icon, the block count, from `blocks` or another program's
/// `blocks.bin`); the PS5's `param.sfo` and the like are kept beside them.
fn copy_meta(src: &Path, dst: &Path) -> std::io::Result<u64> {
    let (s, d) = (src.join("sce_sys"), dst.join("sce_sys"));
    if !s.is_dir() {
        return Ok(0);
    }
    std::fs::create_dir_all(&d)?;
    let mut n = 0;
    for e in std::fs::read_dir(&s)?.flatten() {
        let name = e.file_name().to_string_lossy().to_string();
        let p = e.path();
        if name.ends_with(".tmp") || !p.is_file() {
            continue;
        }
        match name.as_str() {
            "param.bin" if std::fs::metadata(&p)?.len() != PARAM_SIZE => continue,
            "blocks" | "blocks.bin" => {
                let b = std::fs::read(&p)?;
                let Some(v) = b.get(..8).map(|x| u64::from_le_bytes(x.try_into().unwrap())) else { continue };
                if !(1..=1 << 24).contains(&v) {
                    continue;
                }
                std::fs::write(d.join("blocks"), v.to_le_bytes())?;
            }
            _ => {
                std::fs::copy(&p, d.join(&name))?;
            }
        }
        n += 1;
    }
    Ok(n)
}

/// Copies the saves found in `from` into `saves` (`<saves>\<USER>\<title>\<save>`). A save that is
/// there already is kept first in `save-backups` beside the save folder (`<stamp>-before-import`)
/// and keeps its own metadata; its game files are replaced. `from` is left as it is.
pub fn import(from: &Path, saves: &Path, title: &str, stamp: &str) -> Report {
    let mut r = Report::default();
    let found = find(from);
    if found.is_empty() {
        r.notes.push(format!("no save folder (SAVEDATA... or sdimg_SAVEDATA... with USR-DATA, at most {DEPTH} folders down) in {}", from.display()));
        return r;
    }
    let base = saves.join(USER).join(title);
    let mut names = std::collections::HashSet::new();
    for src in &found {
        let folder = src.file_name().unwrap_or_default().to_string_lossy().to_string();
        let Some(name) = save_name(&folder).map(str::to_string) else { continue };
        // (A name found twice, as SAVEDATA... and sdimg_SAVEDATA... or in two folders: the first
        // in name order.)
        if !names.insert(name.clone()) {
            r.notes.push(format!("{} left out: {name} was imported already", src.display()));
            continue;
        }
        let dst = base.join(&name);
        let had = dst.is_dir();
        if had {
            // (A folder of its own: two imports within the same second, two misplaced folders
            // holding the same save at one start, wrote the second backup over the first, the
            // player's own save then nowhere: the review of 2026-11-27.)
            let dir = saves.parent().unwrap_or(saves).join("save-backups").join(USER).join(title).join(&name);
            let mut keep = dir.join(format!("{stamp}-before-import"));
            let mut k = 2;
            while keep.exists() {
                keep = dir.join(format!("{stamp}-before-import-{k}"));
                k += 1;
            }
            if let Err(e) = copy_tree(&dst, &keep) {
                r.notes.push(format!("keeping {} failed ({e}): {name} not imported", dst.display()));
                continue;
            }
            r.kept.push(keep);
        }
        let mut n = 0u64;
        let mut failed = None;
        let rd = match std::fs::read_dir(src) {
            Ok(rd) => rd,
            Err(e) => {
                r.notes.push(format!("{}: {e}", src.display()));
                continue;
            }
        };
        if let Err(e) = std::fs::create_dir_all(&dst) {
            r.notes.push(format!("{}: {e}", dst.display()));
            continue;
        }
        for e in rd.flatten() {
            let p = e.path();
            let fname = e.file_name();
            if fname.to_string_lossy().ends_with(".tmp") {
                continue;
            }
            let res = if p.is_dir() {
                if fname == "sce_sys" {
                    if had { Ok(0) } else { copy_meta(src, &dst) }
                } else {
                    copy_tree(&p, &dst.join(&fname))
                }
            } else {
                std::fs::copy(&p, dst.join(&fname)).map(|_| 1)
            };
            match res {
                Ok(k) => n += k,
                Err(e) => failed = Some(format!("{} -> {}: {e}", p.display(), dst.display())),
            }
        }
        match failed {
            None => r.imported.push((name, src.clone(), n)),
            Some(f) => r.notes.push(f),
        }
    }
    r
}

/// Folders straight in the save folder, or in its user folder, that hold saves in another layout
/// (another program's `_SaveData`, a title's folder without the user folder above it, save
/// folders themselves): what a player copied there by hand. The port's own (`<USER>\<title>`), other
/// users' folders and folders an import renamed are left alone.
/// The file a misplaced folder gets when it was imported but could not be renamed
/// (`misplaced` passes it over).
pub const IMPORTED_MARK: &str = "imported-by-regaliant.txt";

pub fn misplaced(saves: &Path, title: &str) -> Vec<PathBuf> {
    let mut out = Vec::new();
    let is_user = |n: &str| !n.is_empty() && n.bytes().all(|b| b.is_ascii_digit());
    let skip = |n: &str| n.starts_with('.') || n.contains(" (imported ");
    // (Marked when it could not be renamed after its import: not again at every start.)
    let marked = |p: &Path| p.join(IMPORTED_MARK).is_file();
    let holds = |p: &Path| {
        let name = p.file_name().unwrap_or_default().to_string_lossy().to_string();
        (save_name(&name).is_some() && p.join("USR-DATA").is_file()) || !find(p).is_empty()
    };
    let Ok(rd) = std::fs::read_dir(saves) else { return out };
    for e in rd.flatten() {
        let p = e.path();
        let n = e.file_name().to_string_lossy().to_string();
        if !p.is_dir() || skip(&n) {
            continue;
        }
        if n == USER {
            for u in std::fs::read_dir(&p).into_iter().flatten().flatten() {
                let un = u.file_name().to_string_lossy().to_string();
                if u.path().is_dir() && un != title && !skip(&un) && !marked(&u.path()) && holds(&u.path()) {
                    out.push(u.path());
                }
            }
        } else if !is_user(&n) && !marked(&p) && holds(&p) {
            out.push(p);
        }
    }
    out.sort();
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    fn tmp(name: &str) -> PathBuf {
        let base = std::env::temp_dir().join(format!("dsconfig-saves-{name}-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&base);
        std::fs::create_dir_all(&base).unwrap();
        base
    }

    fn save(dir: &Path, data: &str) {
        std::fs::create_dir_all(dir.join("sce_sys")).unwrap();
        std::fs::write(dir.join("USR-DATA"), data).unwrap();
    }

    #[test]
    fn imports_a_ps5_export_and_keeps_the_current() {
        let base = tmp("ps5");
        let from = base.join("PS5 Saves");
        let player = from.join("export").join("PPSA01342").join("SAVEDATA0PlayerProfile0");
        save(&player, "ps5");
        std::fs::write(player.join("sce_sys").join("param.sfo"), "meta").unwrap();
        let saves = base.join("saves");
        let cur = saves.join(USER).join("PPSA01341").join("SAVEDATA0PlayerProfile0");
        save(&cur, "pc");
        std::fs::write(cur.join("sce_sys").join("param.bin"), "ours").unwrap();
        let r = import(&from, &saves, "PPSA01341", "20261126-200000");
        assert_eq!(r.imported.len(), 1, "{:?}", r.notes);
        assert_eq!(std::fs::read_to_string(cur.join("USR-DATA")).unwrap(), "ps5");
        // (Our metadata kept, the import's not copied over it.)
        assert!(cur.join("sce_sys").join("param.bin").is_file());
        assert!(!cur.join("sce_sys").join("param.sfo").exists());
        assert_eq!(r.kept.len(), 1);
        assert_eq!(std::fs::read_to_string(r.kept[0].join("USR-DATA")).unwrap(), "pc");
        // (The source as it was.)
        assert!(player.join("USR-DATA").is_file());
        let _ = std::fs::remove_dir_all(&base);
    }

    #[test]
    fn takes_sdimg_folders_without_the_ps5_backups() {
        let base = tmp("sdimg");
        let from = base.join("PS5 Saves");
        for (dir, data) in [("sdimg_SAVEDATA0PlayerProfile0", "player0"), ("sdimg_SAVEDATA0OptionsProfile0", "options0"), ("sdimg_sce_bu_SAVEDATA0PlayerProfile0", "backup")] {
            save(&from.join("PPSA01341").join(dir), data);
        }
        let saves = base.join("saves");
        let r = import(&from, &saves, "PPSA01341", "s");
        assert_eq!(r.imported.len(), 2);
        let got = saves.join(USER).join("PPSA01341");
        assert_eq!(std::fs::read_to_string(got.join("SAVEDATA0PlayerProfile0").join("USR-DATA")).unwrap(), "player0");
        assert_eq!(std::fs::read_to_string(got.join("SAVEDATA0OptionsProfile0").join("USR-DATA")).unwrap(), "options0");
        assert!(!got.join("sdimg_sce_bu_SAVEDATA0PlayerProfile0").exists());
        assert_eq!(save_name("sdimg_sce_bu_SAVEDATA0PlayerProfile0"), None);
        let _ = std::fs::remove_dir_all(&base);
    }

    #[test]
    fn imports_another_programs_folder() {
        // Its whole folder, a guest user's saves six folders down, its own metadata and a file it
        // was still writing.
        let base = tmp("emu");
        let emu = base.join("other program 0.9");
        let player = emu.join("_SaveData").join("[users]").join("2").join("PPSA01342").join("SAVEDATA0PlayerProfile0");
        save(&player, "emu player");
        std::fs::write(player.join("sce_sys").join("param.bin"), vec![0u8; PARAM_SIZE as usize]).unwrap();
        std::fs::write(player.join("sce_sys").join("blocks.bin"), 96u64.to_le_bytes()).unwrap();
        std::fs::write(player.join("sce_sys").join("param.bin.tmp"), "partial").unwrap();
        std::fs::write(player.join("USR-DATA.tmp"), "partial").unwrap();
        let options = emu.join("_SaveData").join("PPSA01341").join("SAVEDATA0OptionsProfile0");
        save(&options, "emu options");
        std::fs::write(options.join("sce_sys").join("param.bin"), "a size that is not the struct's").unwrap();
        let memory = emu.join("_SaveData").join("PPSA01341").join("sce_sdmemory").join("1");
        std::fs::create_dir_all(&memory).unwrap();
        std::fs::write(memory.join("memory.dat"), "memory").unwrap();
        let saves = base.join("saves");
        let r = import(&emu, &saves, "PPSA01341", "s");
        assert_eq!(r.imported.len(), 2, "{:?}", r.notes);
        let got = saves.join(USER).join("PPSA01341");
        let p = got.join("SAVEDATA0PlayerProfile0");
        assert_eq!(std::fs::read_to_string(p.join("USR-DATA")).unwrap(), "emu player");
        assert_eq!(std::fs::metadata(p.join("sce_sys").join("param.bin")).unwrap().len(), PARAM_SIZE);
        assert_eq!(std::fs::read(p.join("sce_sys").join("blocks")).unwrap(), 96u64.to_le_bytes());
        assert!(!p.join("sce_sys").join("blocks.bin").exists());
        assert!(!p.join("sce_sys").join("param.bin.tmp").exists());
        assert!(!p.join("USR-DATA.tmp").exists());
        let o = got.join("SAVEDATA0OptionsProfile0");
        assert_eq!(std::fs::read_to_string(o.join("USR-DATA")).unwrap(), "emu options");
        // (A parameter file of another size is not taken: the port writes its own.)
        assert!(!o.join("sce_sys").join("param.bin").exists());
        assert!(!got.join("sce_sdmemory").exists());
        let _ = std::fs::remove_dir_all(&base);
    }

    #[test]
    fn finds_saves_copied_into_the_save_folder_by_hand() {
        let base = tmp("misplaced");
        let saves = base.join("saves");
        // Ours, left alone; another user's, left alone.
        save(&saves.join(USER).join("PPSA01341").join("SAVEDATA0PlayerProfile0"), "ours");
        save(&saves.join("2").join("PPSA01341").join("SAVEDATA0PlayerProfile0"), "other user");
        // Another program's folder, a title folder without the user folder, another region's
        // title under our user, and an import done before.
        save(&saves.join("_SaveData").join("PPSA01341").join("SAVEDATA0PlayerProfile0"), "emu");
        save(&saves.join("PPSA01342").join("SAVEDATA0OptionsProfile0"), "title only");
        save(&saves.join(USER).join("PPSA01342").join("SAVEDATA0PlayerProfile0"), "other region");
        save(&saves.join("_SaveData (imported 20261126-100000)").join("PPSA01341").join("SAVEDATA0PlayerProfile0"), "done");
        let m = misplaced(&saves, "PPSA01341");
        let names: Vec<String> = m.iter().map(|p| p.strip_prefix(&saves).unwrap().display().to_string()).collect();
        assert_eq!(names, vec![format!("1{}PPSA01342", std::path::MAIN_SEPARATOR), "PPSA01342".to_string(), "_SaveData".to_string()]);
        let _ = std::fs::remove_dir_all(&base);
    }

    #[test]
    fn two_imports_in_one_second_keep_both_backups() {
        // (The review of 2026-11-27: the second backup wrote over the first, the player's own save
        // then nowhere.)
        let base = tmp("twice");
        let saves = base.join("saves");
        let cur = saves.join(USER).join("PPSA01341").join("SAVEDATA0PlayerProfile0");
        save(&cur, "the player's own");
        let a = base.join("a");
        save(&a.join("PPSA01341").join("SAVEDATA0PlayerProfile0"), "first import");
        let b = base.join("b");
        save(&b.join("PPSA01341").join("SAVEDATA0PlayerProfile0"), "second import");
        let r1 = import(&a, &saves, "PPSA01341", "20261127-080000");
        let r2 = import(&b, &saves, "PPSA01341", "20261127-080000");
        assert_eq!((r1.kept.len(), r2.kept.len()), (1, 1));
        assert_ne!(r1.kept[0], r2.kept[0]);
        assert_eq!(std::fs::read_to_string(r1.kept[0].join("USR-DATA")).unwrap(), "the player's own");
        assert_eq!(std::fs::read_to_string(r2.kept[0].join("USR-DATA")).unwrap(), "first import");
        assert_eq!(std::fs::read_to_string(cur.join("USR-DATA")).unwrap(), "second import");
        let _ = std::fs::remove_dir_all(&base);
    }

    #[test]
    fn a_marked_folder_is_not_found_again() {
        let base = tmp("marked");
        let saves = base.join("saves");
        save(&saves.join("_SaveData").join("PPSA01341").join("SAVEDATA0PlayerProfile0"), "emu");
        assert_eq!(misplaced(&saves, "PPSA01341").len(), 1);
        std::fs::write(saves.join("_SaveData").join(IMPORTED_MARK), "imported").unwrap();
        assert!(misplaced(&saves, "PPSA01341").is_empty());
        let _ = std::fs::remove_dir_all(&base);
    }

    #[test]
    fn nothing_to_import_changes_nothing() {
        let base = tmp("empty");
        let from = base.join("PS5 Saves");
        std::fs::create_dir_all(from.join("SAVEDATA0PlayerProfile0")).unwrap();
        let r = import(&from, &base.join("saves"), "PPSA01341", "s");
        assert!(r.imported.is_empty() && r.notes.len() == 1);
        assert!(!base.join("saves").exists());
        let _ = std::fs::remove_dir_all(&base);
    }
}
