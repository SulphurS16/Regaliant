//! The Saves page: where the saves are, the backups the game makes before a session first
//! changes a save (and before an import or a restore), a restore of one, the import of saves
//! made elsewhere (a decrypted PS5 save, another program's save folder: `dsconfig::saves`) from a
//! folder the player picks, and the folder the game's own Import PS5 Saves reads. Backups live beside the save folder: `save-backups\<user>\<title>\
//! <save>\<UTC time>[-before-...]`, as the runtime writes them.

use crate::ui::{self, Builder};
use crate::App;
use std::path::{Path, PathBuf};
use windows_sys::Win32::Foundation::HWND;
use windows_sys::Win32::UI::WindowsAndMessaging::{IDOK, MB_ICONERROR, MB_ICONINFORMATION, MB_ICONQUESTION, MB_OK, MB_OKCANCEL};

const ID_OPEN_SAVES: i32 = 301;
const ID_LIST: i32 = 302;
const ID_RESTORE: i32 = 303;
const ID_OPEN_BACKUPS: i32 = 304;
const ID_OPEN_IMPORT: i32 = 305;
const ID_IMPORT: i32 = 306;

#[derive(Default)]
pub struct Page {
    folder: usize,
    list: usize,
    restore: usize,
    import: usize,
    /// The listed backups (save directory under the save folder, backup folder).
    rows: Vec<(PathBuf, PathBuf)>,
}

fn h(v: usize) -> HWND {
    v as HWND
}

/// The save folder in use (the one the game last recorded, else the port's own).
pub fn save_folder(app: &App) -> PathBuf {
    std::fs::read_to_string(app.paths.save_folder_record()).ok().map(|s| PathBuf::from(s.trim())).filter(|p| !p.as_os_str().is_empty()).unwrap_or_else(|| app.paths.saves())
}

fn backups_root(saves: &Path) -> PathBuf {
    saves.parent().unwrap_or(saves).join("save-backups")
}

/// The folder a PS5 save is imported from.
fn import_dir(app: &App) -> PathBuf {
    app.paths.root.as_ref().unwrap_or(&app.paths.user).join("PS5 Saves")
}

fn label(save: &str) -> String {
    match save {
        "SAVEDATA0PlayerProfile0" => "Characters".into(),
        "SAVEDATA0OptionsProfile0" => "Options".into(),
        s => s.to_string(),
    }
}

fn subdirs(p: &Path) -> Vec<PathBuf> {
    let mut v: Vec<PathBuf> = std::fs::read_dir(p).map(|rd| rd.flatten().map(|e| e.path()).filter(|p| p.is_dir()).collect()).unwrap_or_default();
    v.sort();
    v
}

pub fn build(app: &mut App) {
    let b: Builder = app.builder();
    unsafe {
        let r = (16, 67, 100, 20);
        app.put(3, b.label("Save folder:", r, -1), r);
        let r = (116, 64, 544, 24);
        app.saves.folder = app.put(3, b.field("", r, -1), r) as usize;
        let r = (668, 63, 96, 26);
        app.put(3, b.button("Open", r, ID_OPEN_SAVES), r);
        let r = (16, 102, 748, 20);
        app.put(3, b.label("Backups, made before a session first changes a save. The newest are kept.", r, -1), r);
        let r = (16, 124, 748, 230);
        app.saves.list = app.put(3, ui::list_view(&b, r, ID_LIST, &[("Save", 200), ("Made", 200), ("Why", 300)]), r) as usize;
        let r = (16, 362, 140, 28);
        app.saves.restore = app.put(3, b.button("Restore...", r, ID_RESTORE), r) as usize;
        let r = (164, 362, 180, 28);
        app.put(3, b.button("Open backups folder", r, ID_OPEN_BACKUPS), r);
        let r = (16, 404, 400, 44);
        app.put(3, b.label("Saves made elsewhere, such as a decrypted PS5 save: choose the folder that holds them.", r, -1), r);
        let r = (424, 410, 150, 28);
        app.saves.import = app.put(3, b.button("Import saves...", r, ID_IMPORT), r) as usize;
        let r = (582, 410, 182, 28);
        app.put(3, b.button("Open PS5 Saves folder", r, ID_OPEN_IMPORT), r);
    }
    refresh(app);
}

/// Lists the save folder's backups, newest first.
pub fn refresh(app: &mut App) {
    if app.saves.list == 0 {
        return;
    }
    let saves = save_folder(app);
    unsafe {
        ui::set_text(h(app.saves.folder), &saves.display().to_string());
        ui::lv_clear(h(app.saves.list));
    }
    let mut rows: Vec<(String, PathBuf, PathBuf)> = Vec::new();
    let root = backups_root(&saves);
    for user in subdirs(&root) {
        for title in subdirs(&user) {
            for save in subdirs(&title) {
                for b in subdirs(&save) {
                    let (Some(u), Some(t), Some(s), Some(name)) = (user.file_name(), title.file_name(), save.file_name(), b.file_name()) else { continue };
                    let dir = saves.join(u).join(t).join(s);
                    rows.push((name.to_string_lossy().into_owned(), dir, b.clone()));
                }
            }
        }
    }
    rows.sort_by(|a, b| b.0.cmp(&a.0));
    app.saves.rows.clear();
    for (name, dir, b) in rows {
        let why = if name.ends_with("-before-import") {
            "Before saves were imported"
        } else if name.ends_with("-before-restore") {
            "Before a backup was restored"
        } else {
            "Before a session changed it"
        };
        let save = dir.file_name().map(|s| label(&s.to_string_lossy())).unwrap_or_default();
        unsafe { ui::lv_add(h(app.saves.list), &[save, dsconfig::stamp_text(&name), why.to_string()]) };
        app.saves.rows.push((dir, b));
    }
    update(app);
}

/// Restore is offered while a backup is selected and the game is not running.
pub fn update(app: &App) {
    if app.saves.restore == 0 {
        return;
    }
    let sel = unsafe { ui::lv_selected(h(app.saves.list)) }.is_some();
    unsafe { ui::enable(h(app.saves.restore), sel && app.running.is_none()) };
    if app.saves.import != 0 {
        unsafe { ui::enable(h(app.saves.import), app.running.is_none()) };
    }
}

/// Imports the saves in a folder the player picks into the save folder in use (the saves they
/// replace kept as backups first; the picked folder left as it is). Not while the game runs: it
/// keeps its saves open.
fn import(app: &mut App) {
    if app.running.is_some() {
        return;
    }
    let Some(from) = ui::pick_folder(app.hwnd, "Choose the folder that holds the saves") else { return };
    let title = app.game.id.clone().unwrap_or_else(|| dsconfig::game::TITLE_ID.to_string());
    let r = dsconfig::saves::import(&from, &save_folder(app), &title, &dsconfig::utc_stamp());
    let names: Vec<String> = r.imported.iter().map(|(n, _, _)| label(n)).collect();
    crate::shaders::log(app, &format!("saves imported from a folder the player chose: {} ({} kept as backups, {} left out)", if names.is_empty() { "none".to_string() } else { names.join(", ") }, r.kept.len(), r.notes.len()));
    let text = if names.is_empty() {
        format!("No saves of the game were found in {}.\n\nChoose the folder that holds them, for example a decrypted PS5 save or another program's save folder.", ui::short(&from, 80))
    } else {
        let mut t = format!("Imported: {}.", names.join(", "));
        if !r.kept.is_empty() {
            t.push_str("\n\nThe saves they replaced are kept as backups.");
        }
        if !r.notes.is_empty() {
            t.push_str("\n\nSome saves could not be imported.");
        }
        t
    };
    let icon = if names.is_empty() { MB_ICONERROR } else { MB_ICONINFORMATION };
    unsafe { ui::message(app.hwnd, &text, crate::TITLE, icon | MB_OK) };
    if !names.is_empty() {
        app.set_status("Saves imported.");
    }
    refresh(app);
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

/// Restores the selected backup over its save (the save as it is now kept as a backup first).
fn restore(app: &mut App) {
    if app.running.is_some() {
        return;
    }
    let Some(i) = (unsafe { ui::lv_selected(h(app.saves.list)) }) else { return };
    let Some((dir, backup)) = app.saves.rows.get(i).cloned() else { return };
    let name = dir.file_name().map(|s| label(&s.to_string_lossy())).unwrap_or_default();
    let when = backup.file_name().map(|s| dsconfig::stamp_text(&s.to_string_lossy())).unwrap_or_default();
    let text = format!("Restore the {name} save from {when}?\n\nThe save as it is now is kept as a backup first.");
    if unsafe { ui::message(app.hwnd, &text, crate::TITLE, MB_ICONQUESTION | MB_OKCANCEL) } != IDOK {
        return;
    }
    let result = (|| -> std::io::Result<()> {
        if dir.is_dir() {
            let keep = backup.parent().unwrap_or(&backup).join(format!("{}-before-restore", dsconfig::utc_stamp()));
            copy_dir(&dir, &keep)?;
            // (The save replaced whole: no file of the later state stays beside the restored ones.)
            std::fs::remove_dir_all(&dir)?;
        }
        copy_dir(&backup, &dir)?;
        Ok(())
    })();
    match result {
        Ok(()) => {
            unsafe { ui::message(app.hwnd, &format!("The {name} save from {when} is restored."), crate::TITLE, MB_ICONINFORMATION | MB_OK) };
            app.set_status("Save restored.");
        }
        Err(e) => {
            unsafe { ui::message(app.hwnd, &format!("The save could not be restored: {e}\n\nThe backups are unchanged."), crate::TITLE, MB_ICONERROR | MB_OK) };
        }
    }
    refresh(app);
}

pub fn command(app: &mut App, id: i32, code: u32) {
    match (id, code) {
        (ID_OPEN_SAVES, 0) => ui::open_path(&save_folder(app)),
        (ID_OPEN_BACKUPS, 0) => ui::open_path(&backups_root(&save_folder(app))),
        (ID_OPEN_IMPORT, 0) => ui::open_path(&import_dir(app)),
        (ID_IMPORT, 0) => import(app),
        (ID_RESTORE, 0) => restore(app),
        _ => {}
    }
}

/// The list's selection changed.
pub fn notify(app: &mut App) {
    update(app);
}
