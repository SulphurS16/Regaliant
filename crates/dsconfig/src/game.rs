//! The game folder's checks: Demon's Souls (PS5, PPSA01341 or PPSA01342) at version 01.007.000,
//! with its executable and its two libraries decrypted, and its data folders. The runtime checks
//! the same at start and the launcher when a folder is chosen; both show `Problem`'s text. The
//! game can also be a `.zar` archive of that folder (`zar`, since 2026-11-23), read in place.

use crate::zar::Source;
use std::path::{Path, PathBuf};

/// The release the port was made and tested with (Europe).
pub const TITLE_ID: &str = "PPSA01341";
/// The US release (issue #17). The port takes the saves' folder and the add-on content ids from
/// the game's own files, so a US copy keeps its own `PPSA01342` saves and its bonus packages.
pub const TITLE_ID_US: &str = "PPSA01342";
/// The two releases the port names in its checks and its notes.
pub const TITLE_IDS: [&str; 2] = [TITLE_ID, TITLE_ID_US];

/// Demon's Souls' PS5 title ids (PPSA0134x: the regions' releases). Only PPSA01341 was tested;
/// the others are taken when their version matches (the executable's fingerprint says whether
/// its code is the tested one).
pub fn is_demons_souls(id: &str) -> bool {
    id.len() == 9 && id.starts_with("PPSA0134") && id.as_bytes()[8].is_ascii_digit()
}
/// The name of a release the port documents, for messages; other PPSA0134x releases are still
/// taken by `is_demons_souls`, but the port makes no claim about them.
pub fn release_name(id: &str) -> Option<&'static str> {
    match id {
        TITLE_ID => Some("European (PPSA01341)"),
        TITLE_ID_US => Some("US (PPSA01342)"),
        _ => None,
    }
}
/// The one version the port supports.
pub const VERSION: &str = "01.007.000";
/// The application folders' usual names, one for each documented release.
pub const APP_DIR: &str = "PPSA01341-app0";
pub const APP_DIR_US: &str = "PPSA01342-app0";
pub const APP_DIRS: [&str; 2] = [APP_DIR, APP_DIR_US];
/// The decrypted executable this port was made and tested with (SHA-256).
pub const TESTED_EBOOT: &str = "4a6ca8a2904d02a9ff11aebcb10379c89974cd9bff1a37e09c21a7e782447674";
/// The title's own libraries the port loads (decrypted, in `sce_module`).
pub const MODULES: [&str; 2] = ["libc.prx", "libSceNpCppWebApi.prx"];
/// Data folders every complete copy holds.
const DATA_DIRS: [&str; 8] = ["cp11demonssouls", "levels", "sound", "text", "textures", "ui", "movies", "characters"];

#[derive(Clone, Debug)]
pub struct Game {
    /// The title id (`TITLE_ID`, or another region's release).
    pub id: String,
    /// The application folder (data and `sce_sys`), mounted read-only; an archive's own file.
    pub app: PathBuf,
    /// The folder holding the decrypted `eboot.bin` and `sce_module`; an archive's own file.
    pub code: PathBuf,
    /// The game's files: the application folder, or the archive.
    pub source: Source,
    /// The decrypted executable's folder in `source` ("" or "decrypted").
    pub code_dir: String,
    pub version: String,
}

impl std::fmt::Debug for Source {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "Source({})", self.display())
    }
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub enum Problem {
    NotFound,
    /// An archive without the game's folder in its root or one folder down.
    NotFoundInArchive,
    /// An archive that could not be opened (its reason).
    Archive(String),
    OtherGame(String),
    Version(String),
    Param(String),
    NoExecutable,
    Encrypted,
    NoModule(String),
    EncryptedModule(String),
    /// A SELF container the port cannot rebuild an ELF from: (its file, why), such as one cut
    /// short by a copy or with compressed segments (`self_file`).
    Unusable(String, String),
    MissingData(String),
}

impl std::fmt::Display for Problem {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Problem::NotFound => write!(f, "No Demon's Souls game files were found in this folder. Choose the folder that holds eboot.bin and sce_sys."),
            Problem::NotFoundInArchive => write!(f, "No Demon's Souls game files were found in this .zar archive. It must hold the folder with eboot.bin and sce_sys."),
            Problem::Archive(e) => write!(f, "This .zar archive could not be read ({e})."),
            Problem::OtherGame(id) => write!(f, "This folder holds another game ({id}). This port runs the PS5 game Demon's Souls ({}).", TITLE_IDS.join(" or ")),
            Problem::Version(v) => write!(f, "This is version {v} of the game. This port needs version {VERSION}."),
            Problem::Param(e) => write!(f, "The game's sce_sys\\param.json could not be read ({e})."),
            Problem::NoExecutable => write!(f, "The game's eboot.bin is missing."),
            Problem::Encrypted => write!(f, "The game's eboot.bin is encrypted. The port needs the decrypted eboot.bin (a plain ELF, or a SELF with decrypted segments), in the game folder or in its decrypted folder."),
            Problem::NoModule(m) => write!(f, "{m} is missing. The port needs the game's decrypted libraries in sce_module."),
            Problem::EncryptedModule(m) => write!(f, "{m} is encrypted. The port needs the game's decrypted libraries in sce_module (plain ELF files, or SELFs with decrypted segments)."),
            Problem::Unusable(file, why) => write!(f, "The game's {file} could not be read ({why}). Copy it from your dump again, or convert it to an ELF file."),
            Problem::MissingData(d) => write!(f, "The game's {d} folder is missing. The game files are incomplete."),
        }
    }
}

fn has_param(d: &Path) -> bool {
    d.join("sce_sys").join("param.json").is_file()
}

/// The application folder at or under `dir`: `dir` itself, one of the releases' `...-app0`
/// folders, or one folder further down (a dump's folder holding the application folder).
pub fn locate(dir: &Path) -> Option<PathBuf> {
    if has_param(dir) {
        return Some(dir.to_path_buf());
    }
    for app in APP_DIRS {
        if has_param(&dir.join(app)) {
            return Some(dir.join(app));
        }
    }
    let mut subs: Vec<PathBuf> = std::fs::read_dir(dir).ok()?.flatten().map(|e| e.path()).filter(|p| p.is_dir()).collect();
    subs.sort();
    subs.into_iter().find_map(|s| {
        std::iter::once(s.clone()).chain(APP_DIRS.iter().map(|app| s.join(app))).find(|d| has_param(d))
    })
}

/// What one of the game's executables is (`exe_state`).
#[derive(Debug, PartialEq, Eq)]
enum Exe {
    /// Not there, unreadable or shorter than a signature.
    Missing,
    /// A plain ELF, or a SELF container with its segments decrypted.
    Usable,
    Encrypted,
    /// A SELF container that cannot be rebuilt into an ELF, and why.
    Unusable(String),
}

/// What `rel` in the game's files is: a decrypted executable or not, and why not.
fn exe_state(src: &Source, rel: &str) -> Exe {
    let Ok(f) = src.open_file(rel) else { return Exe::Missing };
    let mut b = [0u8; 4];
    if !matches!(f.read_at(0, &mut b), Ok(4)) {
        return Exe::Missing;
    }
    if b == *b"\x7fELF" {
        return Exe::Usable;
    }
    if !crate::self_file::is_self(&b) {
        return Exe::Encrypted;
    }
    // (A SELF container whose segments are decrypted holds a usable ELF: `self_file`.)
    match src.read(rel) {
        Err(e) => Exe::Unusable(e.to_string()),
        Ok(bytes) => match crate::self_file::decrypted_elf(&bytes) {
            Ok(_) => Exe::Usable,
            Err(e) if e == crate::self_file::ENCRYPTED => Exe::Encrypted,
            Err(e) => Exe::Unusable(e),
        },
    }
}

/// A module's ELF: the file itself, or the ELF its decrypted SELF container holds (issue #8).
pub fn executable_bytes(src: &Source, rel: &str) -> Result<Vec<u8>, String> {
    let bytes = src.read(rel).map_err(|e| format!("{rel}: {e}"))?;
    if crate::self_file::is_self(&bytes) {
        return crate::self_file::decrypted_elf(&bytes).map_err(|e| format!("{rel}: {e}"));
    }
    Ok(bytes)
}

/// The title's id and content version from `sce_sys/param.json`.
pub fn param(app: &Path) -> Result<(String, String), Problem> {
    param_from(&Source::Dir(app.to_path_buf()))
}

/// `param` from the game's files.
pub fn param_from(src: &Source) -> Result<(String, String), Problem> {
    let t = src.read("sce_sys/param.json").map_err(|e| Problem::Param(e.to_string()))?;
    let t = String::from_utf8(t).map_err(|e| Problem::Param(e.to_string()))?;
    let v: serde_json::Value = serde_json::from_str(&t).map_err(|e| Problem::Param(e.to_string()))?;
    let get = |k: &str| v.get(k).and_then(|x| x.as_str()).map(|s| s.trim().to_string());
    let id = get("titleId").ok_or_else(|| Problem::Param("no titleId".into()))?;
    let ver = get("contentVersion").ok_or_else(|| Problem::Param("no contentVersion".into()))?;
    Ok((id, ver))
}

/// Checks the game folder at or under `dir`, or the archive `dir` (`.zar`).
pub fn check(dir: &Path) -> Result<Game, Problem> {
    let source = if crate::zar::is_archive_path(dir) {
        let s = Source::open(dir).map_err(|e| Problem::Archive(e.to_string()))?;
        if !s.is_file("sce_sys/param.json") {
            return Err(Problem::NotFoundInArchive);
        }
        s
    } else {
        Source::Dir(locate(dir).ok_or(Problem::NotFound)?)
    };
    let (id, version) = param_from(&source)?;
    if !is_demons_souls(&id) {
        return Err(Problem::OtherGame(id));
    }
    if version != VERSION {
        return Err(Problem::Version(version));
    }
    // The decrypted executable: in `decrypted\` (as dumps keep it beside the encrypted one) or
    // the application folder's own.
    let decrypted = exe_state(&source, "decrypted/eboot.bin");
    let code_dir = if decrypted == Exe::Usable {
        "decrypted"
    } else {
        match (decrypted, exe_state(&source, "eboot.bin")) {
            (_, Exe::Usable) => "",
            // (Neither usable: the decrypted copy's reason first, as the one the player meant.)
            (Exe::Unusable(why), _) => return Err(Problem::Unusable("decrypted\\eboot.bin".into(), why)),
            (_, Exe::Unusable(why)) => return Err(Problem::Unusable("eboot.bin".into(), why)),
            (Exe::Encrypted, _) | (_, Exe::Encrypted) => return Err(Problem::Encrypted),
            _ => return Err(Problem::NoExecutable),
        }
    };
    let under = |rel: &str| if code_dir.is_empty() { rel.to_string() } else { format!("{code_dir}/{rel}") };
    for m in MODULES {
        let rel = under(&format!("sce_module/{m}"));
        match exe_state(&source, &rel) {
            Exe::Usable => {}
            Exe::Encrypted => return Err(Problem::EncryptedModule(m.into())),
            Exe::Unusable(why) => return Err(Problem::Unusable(rel.replace('/', "\\"), why)),
            Exe::Missing => return Err(Problem::NoModule(m.into())),
        }
    }
    if let Some(d) = DATA_DIRS.iter().find(|d| !source.is_dir(d)) {
        return Err(Problem::MissingData(d.to_string()));
    }
    let (app, code) = match &source {
        Source::Dir(a) => (a.clone(), if code_dir.is_empty() { a.clone() } else { a.join(code_dir) }),
        Source::Zar { archive, .. } => (archive.path().to_path_buf(), archive.path().to_path_buf()),
    };
    Ok(Game { id, app, code, source, code_dir: code_dir.to_string(), version })
}

/// What the launcher knows about a copy's executable. A documented release whose fingerprint is
/// not on file is *unverified*, not a known bad copy: the port's patches are placed by the tested
/// European program's addresses, and are skipped where the game's bytes differ, so no claim about
/// another release's code can be made from the title id alone.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Fingerprint {
    /// The executable the port was built and tested with.
    Tested,
    /// A fingerprint is on file for this release, but this copy does not match it.
    Different,
    /// No fingerprint is on file for this release (its code has not been verified).
    Unverified,
}

/// The SHA-256 on file for a documented release, if one has been supplied. Only the European
/// copy's is known; a US copy's cannot be inferred from the European one or from the title id.
pub fn known_fingerprint(id: &str) -> Option<&'static str> {
    match id {
        TITLE_ID => Some(TESTED_EBOOT),
        _ => None,
    }
}

/// The decrypted executable's SHA-256, whether it is a folder's file or an archive's own file.
fn executable_sha256(g: &Game) -> std::io::Result<String> {
    let rel = if g.code_dir.is_empty() { "eboot.bin".to_string() } else { format!("{}/eboot.bin", g.code_dir) };
    // (A decrypted SELF container: the ELF it holds, rebuilt byte for byte, is what is compared.)
    let f = g.source.open_file(&rel)?;
    let mut head = [0u8; 4];
    if f.read_at(0, &mut head)? == 4 && crate::self_file::is_self(&head) {
        let elf = executable_bytes(&g.source, &rel).map_err(std::io::Error::other)?;
        let mut sha = crate::sha::Sha256::new()?;
        sha.update(&elf)?;
        return Ok(crate::sha::hex(&sha.finish()?));
    }
    if let Source::Dir(_) = &g.source {
        return crate::sha::file_sha256(&g.code.join("eboot.bin"));
    }
    let mut sha = crate::sha::Sha256::new()?;
    let mut buf = vec![0u8; 8 << 20];
    let mut off = 0u64;
    loop {
        let n = f.read_at(off, &mut buf)?;
        if n == 0 {
            break;
        }
        sha.update(&buf[..n])?;
        off += n as u64;
    }
    Ok(crate::sha::hex(&sha.finish()?))
}

/// How this copy's executable compares with the fingerprint on file for its release.
pub fn fingerprint(g: &Game) -> std::io::Result<Fingerprint> {
    let Some(want) = known_fingerprint(&g.id) else { return Ok(Fingerprint::Unverified) };
    Ok(if executable_sha256(g)? == want { Fingerprint::Tested } else { Fingerprint::Different })
}

/// Whether the executable is the one the port was tested with (its SHA-256).
pub fn tested_executable(g: &Game) -> std::io::Result<bool> {
    Ok(fingerprint(g)? == Fingerprint::Tested)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn fake(root: &Path, version: &str, decrypted_dir: bool) -> PathBuf {
        fake_with_id(root, version, TITLE_ID, decrypted_dir)
    }

    fn fake_with_id(root: &Path, version: &str, id: &str, decrypted_dir: bool) -> PathBuf {
        let app = root.join(format!("{id}-app0"));
        std::fs::create_dir_all(app.join("sce_sys")).unwrap();
        std::fs::write(app.join("sce_sys").join("param.json"), format!("{{\"titleId\": \"{id}\", \"contentVersion\": \"{version}\"}}")).unwrap();
        for d in DATA_DIRS {
            std::fs::create_dir_all(app.join(d)).unwrap();
        }
        let code = if decrypted_dir { app.join("decrypted") } else { app.clone() };
        std::fs::create_dir_all(code.join("sce_module")).unwrap();
        std::fs::write(code.join("eboot.bin"), b"\x7fELF....").unwrap();
        if decrypted_dir {
            // (The retail container beside it: a SELF whose segments are encrypted.)
            std::fs::write(app.join("eboot.bin"), crate::self_file::tests::encrypted_sample()).unwrap();
        }
        for m in MODULES {
            std::fs::write(code.join("sce_module").join(m), b"\x7fELF....").unwrap();
        }
        app
    }

    #[test]
    fn checks() {
        let tmp = std::env::temp_dir().join(format!("dsconfig-game-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&tmp);
        let dump = tmp.join("Demon-Souls-dump");
        let app = fake(&dump, VERSION, true);
        // (The dump's folder, the application folder, and the folder above the dump.)
        for d in [&dump, &app, &tmp] {
            let g = check(d).unwrap();
            assert_eq!(g.app, app);
            assert_eq!(g.code, app.join("decrypted"));
        }
        // (A library in the decrypted folder cut short: named with that folder.)
        let s = crate::self_file::tests::sample().1;
        let libc = app.join("decrypted").join("sce_module").join("libc.prx");
        std::fs::write(&libc, &s[..s.len() - 0x40]).unwrap();
        assert!(matches!(check(&dump).unwrap_err(), Problem::Unusable(f, why) if f == "decrypted\\sce_module\\libc.prx" && why.contains("past its end")));
        std::fs::remove_file(&libc).unwrap();
        assert_eq!(check(&dump).unwrap_err(), Problem::NoModule("libc.prx".into()));
        // (The decrypted copy cut short beside an encrypted one: the decrypted copy's reason.)
        std::fs::write(app.join("decrypted").join("eboot.bin"), &s[..s.len() - 0x40]).unwrap();
        let e = check(&dump).unwrap_err();
        assert!(matches!(&e, Problem::Unusable(f, why) if f == "decrypted\\eboot.bin" && why.contains("past its end")), "{e:?}");
        assert!(!e.to_string().contains(';'), "{e}");
        std::fs::remove_dir_all(app.join("decrypted")).unwrap();
        assert_eq!(check(&dump).unwrap_err(), Problem::Encrypted);
        // (A container cut short by a copy says why; an empty file is a missing one.)
        std::fs::write(app.join("eboot.bin"), &s[..s.len() - 0x40]).unwrap();
        assert!(matches!(check(&dump).unwrap_err(), Problem::Unusable(f, why) if f == "eboot.bin" && why.contains("past its end")));
        std::fs::write(app.join("eboot.bin"), b"").unwrap();
        assert_eq!(check(&dump).unwrap_err(), Problem::NoExecutable);
        // (The decrypted container itself is used as it is.)
        std::fs::write(app.join("eboot.bin"), &s).unwrap();
        std::fs::create_dir_all(app.join("sce_module")).unwrap();
        for m in MODULES {
            std::fs::write(app.join("sce_module").join(m), &s).unwrap();
        }
        assert_eq!(check(&dump).unwrap().code, app);
        assert!(!Problem::Unusable("eboot.bin".into(), "x".into()).to_string().contains(';'));
        std::fs::remove_dir_all(&tmp).unwrap();

        let app = fake(&tmp, "01.006.000", false);
        assert_eq!(check(&tmp).unwrap_err(), Problem::Version("01.006.000".into()));
        std::fs::remove_dir_all(&tmp).unwrap();
        let app2 = fake(&tmp, VERSION, false);
        assert_eq!(app, app2);
        assert_eq!(check(&tmp).unwrap().code, app2);
        std::fs::remove_dir_all(app2.join("ui")).unwrap();
        assert_eq!(check(&tmp).unwrap_err(), Problem::MissingData("ui".into()));
        std::fs::remove_dir_all(&tmp).unwrap();
        assert_eq!(check(&tmp).unwrap_err(), Problem::NotFound);
        assert!(!Problem::Encrypted.to_string().contains(';'));
    }

    #[test]
    fn us_release() {
        assert!(is_demons_souls(TITLE_ID_US));
        assert_eq!(release_name(TITLE_ID), Some("European (PPSA01341)"));
        assert_eq!(release_name(TITLE_ID_US), Some("US (PPSA01342)"));
        assert_eq!(release_name("PPSA01343"), None);
        assert_eq!(known_fingerprint(TITLE_ID), Some(TESTED_EBOOT));
        assert_eq!(known_fingerprint(TITLE_ID_US), None);

        let tmp = std::env::temp_dir().join(format!("dsconfig-game-us-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&tmp);
        let dump = tmp.join("Demon-Souls-US-dump");
        let app = fake_with_id(&dump, VERSION, TITLE_ID_US, true);
        assert_eq!(app.file_name().unwrap().to_string_lossy(), APP_DIR_US);
        // A US dump is found by its own application folder, from the dump, the app folder and
        // the folder above it; its id and version are what the game's param.json says.
        for d in [&dump, &app, &tmp] {
            let g = check(d).unwrap();
            assert_eq!(g.id, TITLE_ID_US);
            assert_eq!(g.version, VERSION);
            assert_eq!(g.code, app.join("decrypted"));
            assert_eq!(fingerprint(&g).unwrap(), Fingerprint::Unverified);
        }
        assert_eq!(locate(&dump).as_deref(), Some(app.as_path()));
        // A European copy with the US version's number is still a European copy.
        std::fs::write(app.join("sce_sys").join("param.json"), format!("{{\"titleId\": \"{TITLE_ID}\", \"contentVersion\": \"{VERSION}\"}}")).unwrap();
        assert_eq!(check(&dump).unwrap().id, TITLE_ID);
        // Another game still names both releases it can run.
        let e = Problem::OtherGame("PPSA00000".into()).to_string();
        assert!(e.contains(TITLE_ID) && e.contains(TITLE_ID_US), "{e}");
        std::fs::remove_dir_all(&tmp).unwrap();
    }

    #[test]
    fn archives() {
        let tmp = std::env::temp_dir().join(format!("dsconfig-game-zar-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&tmp);
        let dump = tmp.join("dump");
        let app = fake(&dump, VERSION, true);
        let zar = tmp.join("game.zar");
        crate::zar::write_folder(&dump, &zar, 3, 2, &|_, _| true).unwrap();
        let g = check(&zar).unwrap();
        assert_eq!(g.code_dir, "decrypted");
        assert!(g.source.is_archive() && g.app == zar);
        assert!(!tested_executable(&g).unwrap());
        // The folder's own check: as before, the source its folder.
        let f = check(&dump).unwrap();
        assert_eq!((f.app.clone(), f.code.clone(), f.code_dir.as_str()), (app.clone(), app.join("decrypted"), "decrypted"));
        // Without the game: not found in it; not an archive: refused.
        std::fs::remove_dir_all(app.join("sce_sys")).unwrap();
        let other = tmp.join("other.zar");
        crate::zar::write_folder(&dump, &other, 3, 2, &|_, _| true).unwrap();
        assert_eq!(check(&other).unwrap_err(), Problem::NotFoundInArchive);
        std::fs::write(tmp.join("junk.zar"), b"not an archive at all, just text, long enough for a footer........................................................................................................").unwrap();
        assert!(matches!(check(&tmp.join("junk.zar")), Err(Problem::Archive(_))));
        assert!(!Problem::NotFoundInArchive.to_string().contains(';'));
        std::fs::remove_dir_all(&tmp).unwrap();
    }
}
