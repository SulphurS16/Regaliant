//! Checks `self_file` on a game folder's own files, in memory (nothing written): for the
//! executable and each library, the dump's SELF container (this game folder's: its segments
//! decrypted in place) rebuilds the decrypted ELF byte for byte, and so does the container with
//! its data entries replaced by the decrypted ELF's segments; scrambled into random bytes (as
//! encryption leaves them), it is refused.
//!     cargo run --release -p dsconfig --example selfcheck -- <PPSA01341-app0 or PPSA01342-app0 folder>
use dsconfig::self_file;

fn sha(b: &[u8]) -> String {
    let mut s = dsconfig::sha::Sha256::new().unwrap();
    s.update(b).unwrap();
    dsconfig::sha::hex(&s.finish().unwrap())
}

fn main() {
    let app = std::path::PathBuf::from(std::env::args().nth(1).expect("usage: selfcheck <PPSA01341-app0 or PPSA01342-app0 folder>"));
    let mut failed = false;
    for rel in ["eboot.bin", "sce_module/libc.prx", "sce_module/libSceNpCppWebApi.prx"] {
        let retail = std::fs::read(app.join(rel)).expect("the retail file");
        let plain = std::fs::read(app.join("decrypted").join(rel)).expect("the decrypted file");
        print!("{rel}: the dump's container {} ({} bytes); ", if self_file::is_self(&retail) { "a SELF" } else { "not a SELF" }, retail.len());
        match self_file::decrypted_elf(&retail) {
            Ok(elf) => {
                let same = elf == plain;
                print!("rebuilt {}; ", if same { "identical to the decrypted ELF" } else { "DIFFERENT" });
                failed |= !same;
            }
            Err(e) => {
                print!("REFUSED ({e}); ");
                failed = true;
            }
        }
        // Its code segment scrambled as encryption leaves it: refused.
        let mut enc = retail.clone();
        {
            let n = u16::from_le_bytes([enc[0x18], enc[0x19]]) as usize;
            let e = 0x20 + 0x20;
            let _ = n;
            let off = u64::from_le_bytes(enc[e + 8..e + 16].try_into().unwrap()) as usize;
            let size = u64::from_le_bytes(enc[e + 16..e + 24].try_into().unwrap()) as usize;
            let mut x = 0x9e37_79b9_7f4a_7c15u64;
            for b in enc[off..off + size.min(4 << 20)].iter_mut() {
                x ^= x << 13;
                x ^= x >> 7;
                x ^= x << 17;
                *b = (x >> 24) as u8;
            }
        }
        match self_file::decrypted_elf(&enc) {
            Err(e) => print!("scrambled: refused ({e}); "),
            Ok(_) => {
                print!("scrambled: TAKEN; ");
                failed = true;
            }
        }
        // The container with its data entries decrypted in place.
        let mut dec = retail.clone();
        let n = u16::from_le_bytes([dec[0x18], dec[0x19]]) as usize;
        let eh = 0x20 + n * 0x20;
        let phoff = u64::from_le_bytes(dec[eh + 0x20..eh + 0x28].try_into().unwrap()) as usize;
        for i in 0..n {
            let e = 0x20 + i * 0x20;
            let props = u64::from_le_bytes(dec[e..e + 8].try_into().unwrap());
            if props >> 11 & 1 == 0 {
                continue;
            }
            let seg = (props >> 20 & 0xfff) as usize;
            let off = u64::from_le_bytes(dec[e + 8..e + 16].try_into().unwrap()) as usize;
            let size = u64::from_le_bytes(dec[e + 16..e + 24].try_into().unwrap()) as usize;
            let p = eh + phoff + seg * 0x38;
            let p_off = u64::from_le_bytes(dec[p + 8..p + 16].try_into().unwrap()) as usize;
            dec[off..off + size].copy_from_slice(&plain[p_off..p_off + size]);
        }
        match self_file::decrypted_elf(&dec) {
            Ok(elf) => {
                let same = elf == plain;
                println!("decrypted container rebuilt: {} bytes, {} (sha256 {})", elf.len(), if same { "identical to the decrypted ELF" } else { "DIFFERENT" }, &sha(&elf)[..16]);
                failed |= !same;
            }
            Err(e) => {
                println!("decrypted container REFUSED: {e}");
                failed = true;
            }
        }
    }
    // (A second folder, e.g. one holding the containers alone: the launcher's check of it.)
    if let Some(other) = std::env::args().nth(2) {
        match dsconfig::game::check(std::path::Path::new(&other)) {
            Ok(g) => {
                let tested = dsconfig::game::tested_executable(&g);
                println!("{other}: the game check passes (code folder '{}'), the tested executable: {tested:?}", g.code_dir);
                failed |= !matches!(tested, Ok(true));
            }
            Err(e) => {
                println!("{other}: the game check FAILS: {e}");
                failed = true;
            }
        }
    }
    if failed {
        std::process::exit(1);
    }
}
