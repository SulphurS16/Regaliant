//! Checks a `.zar` archive whose executable and libraries are decrypted SELF containers (issue
//! #8: the player's archive was refused like their folder). Copies a game folder's `param.json`,
//! `eboot.bin` and its two libraries (the dump's top-level files, SELF containers) into a scratch
//! folder with one small file in each data folder the check wants, packs it into a `.zar` beside
//! it, then runs the game check and the tested-version hash on the archive. Reads the game
//! folder only; everything it makes is in a new `zarself-<process id>` folder under the given
//! one, removed at the end (nothing else there is touched; a folder that is, holds or lies in
//! the game folder is refused).
//!     cargo run --release -p dsconfig --example zarself -- <PPSA01341-app0 folder> <scratch folder>
use dsconfig::game;
use std::path::PathBuf;

fn main() {
    let mut args = std::env::args().skip(1);
    let app = PathBuf::from(args.next().expect("usage: zarself <PPSA01341-app0 folder> <scratch folder>"));
    let parent = PathBuf::from(args.next().expect("usage: zarself <PPSA01341-app0 folder> <scratch folder>"));
    #[cfg(target_os = "linux")]
    dsconfig::files::archive_output(&app, &parent).expect("scratch must be outside read-only trees");
    std::fs::create_dir_all(&parent).unwrap();
    let (a, p) = (std::fs::canonicalize(&app).unwrap(), std::fs::canonicalize(&parent).unwrap());
    if a.starts_with(&p) || p.starts_with(&a) {
        eprintln!("refused: the scratch folder {} is, holds or lies in the game folder {}", p.display(), a.display());
        std::process::exit(2);
    }
    let scratch = p.join(format!("zarself-{}", std::process::id()));
    if scratch.exists() {
        eprintln!("refused: {} exists already", scratch.display());
        std::process::exit(2);
    }
    let dir = scratch.join("PPSA01341-app0");
    std::fs::create_dir_all(dir.join("sce_sys")).unwrap();
    std::fs::create_dir_all(dir.join("sce_module")).unwrap();
    let source = dsconfig::zar::Source::Dir(app.clone());
    std::fs::write(dir.join("sce_sys").join("param.json"), source.read("sce_sys/param.json").unwrap()).unwrap();
    let mut files = vec!["eboot.bin".to_string()];
    files.extend(game::MODULES.iter().map(|m| format!("sce_module/{m}")));
    for f in &files {
        let head = source.read(f).unwrap();
        println!("{f}: {} bytes, a SELF container: {}", head.len(), dsconfig::self_file::is_self(&head));
        std::fs::write(dir.join(f), head).unwrap();
    }
    for d in ["cp11demonssouls", "levels", "sound", "text", "textures", "ui", "movies", "characters"] {
        std::fs::create_dir_all(dir.join(d)).unwrap();
        std::fs::write(dir.join(d).join("stub.txt"), b"stub").unwrap();
    }
    let zar = scratch.join("selfs.zar");
    let w = dsconfig::zar::write_folder(&dir, &zar, dsconfig::zar::LEVEL, 4, &|_, _| true).unwrap();
    println!("archive: {} ({} bytes): {w:?}", zar.display(), std::fs::metadata(&zar).unwrap().len());
    let r = game::check(&zar);
    match &r {
        Ok(g) => {
            println!("game check on the archive: passes (code folder '{}')", g.code_dir);
            println!("tested executable: {:?}", game::tested_executable(g));
            let elf = game::executable_bytes(&g.source, "eboot.bin").unwrap();
            println!("eboot.bin from the archive rebuilt: {} bytes, an ELF: {}", elf.len(), elf.starts_with(b"\x7fELF"));
        }
        Err(e) => println!("game check on the archive: {e}"),
    }
    std::fs::remove_dir_all(&scratch).unwrap();
    println!("scratch removed");
    std::process::exit(if r.is_ok() { 0 } else { 1 });
}
