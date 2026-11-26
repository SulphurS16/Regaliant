//! A developer tool for the `.zar` support (`dsconfig::zar`), not shipped:
//!
//! - `zartool write <folder> <out.zar> [level] [threads]`: an archive of the folder, with its time
//!   and sizes;
//! - `zartool verify <archive> <folder>`: the archive's integrity hash, and every file of the
//!   folder byte for byte against the archive's (and nothing more in the archive);
//! - `zartool bench <archive> [threads] [rounds]`: every file decompressed, the rate per thread
//!   count;
//! - `zartool levels <folder> <scratch dir> <level,...>`: archives of the folder at each level
//!   (written to the scratch folder, then removed), their sizes and times, and the rate at which
//!   each reads back.

use dsconfig::zar::{Archive, Source, write_folder};
use std::path::{Path, PathBuf};
use std::time::Instant;

fn gib(b: u64) -> f64 {
    b as f64 / (1u64 << 30) as f64
}

fn main() {
    let a: Vec<String> = std::env::args().skip(1).collect();
    let code = match a.first().map(String::as_str) {
        Some("write") if a.len() >= 3 => write(Path::new(&a[1]), Path::new(&a[2]), a.get(3).and_then(|v| v.parse().ok()).unwrap_or(dsconfig::zar::LEVEL), a.get(4).and_then(|v| v.parse().ok()).unwrap_or(threads())),
        Some("verify") if a.len() >= 3 => verify(Path::new(&a[1]), Path::new(&a[2])),
        Some("bench") if a.len() >= 2 => bench(Path::new(&a[1]), a.get(2).and_then(|v| v.parse().ok()), a.get(3).and_then(|v| v.parse().ok()).unwrap_or(1)),
        Some("levels") if a.len() >= 4 => levels(Path::new(&a[1]), Path::new(&a[2]), &a[3]),
        Some("readall") if a.len() >= 2 => readall(&a[1..]),
        _ => {
            eprintln!("usage: zartool write <folder> <out.zar> [level] [threads] | verify <archive> <folder> | bench <archive> [threads] [rounds] | levels <folder> <scratch> <l1,l2,...> | readall <file or folder>...");
            2
        }
    };
    std::process::exit(code);
}

fn threads() -> usize {
    std::thread::available_parallelism().map_or(4, |n| n.get()).saturating_sub(1).max(1)
}

fn write(src: &Path, out: &Path, level: i32, threads: usize) -> i32 {
    let t0 = Instant::now();
    let last = std::sync::Mutex::new(Instant::now());
    let r = write_folder(src, out, level, threads, &|d, t| {
        let mut l = last.lock().unwrap();
        if l.elapsed().as_secs() >= 10 {
            *l = Instant::now();
            eprintln!("  {:.1} of {:.1} GiB read, {:.0} s", gib(d), gib(t), t0.elapsed().as_secs_f64());
        }
        true
    });
    match r {
        Ok(w) => {
            let s = t0.elapsed().as_secs_f64();
            println!("wrote {} files: {:.2} GiB read, archive {:.2} GiB ({:.1}%), level {level}, {threads} threads, {s:.1} s ({:.0} MiB/s read)", w.files, gib(w.bytes_in), gib(w.bytes_out), w.bytes_out as f64 * 100.0 / w.bytes_in.max(1) as f64, w.bytes_in as f64 / s / (1 << 20) as f64);
            0
        }
        Err(e) => {
            eprintln!("write failed: {e}");
            1
        }
    }
}

fn verify(archive: &Path, folder: &Path) -> i32 {
    let t0 = Instant::now();
    match dsconfig::zar::check_hash(archive) {
        Ok((stored, computed)) if stored == computed => println!("integrity hash matches ({:.0} s)", t0.elapsed().as_secs_f64()),
        Ok(_) => {
            eprintln!("integrity hash DIFFERS");
            return 1;
        }
        Err(e) => {
            eprintln!("hash: {e}");
            return 1;
        }
    }
    let zar = match Source::open(archive) {
        Ok(s) => s,
        Err(e) => {
            eprintln!("open: {e}");
            return 1;
        }
    };
    let dir = Source::Dir(folder.to_path_buf());
    let theirs = dir.files_under("");
    let ours = zar.files_under("");
    let (mut bad, mut bytes) = (0u64, 0u64);
    let mut buf_a = vec![0u8; 8 << 20];
    let mut buf_b = vec![0u8; 8 << 20];
    for f in &theirs {
        let (Ok(a), Ok(b)) = (dir.open_file(f), zar.open_file(f)) else {
            eprintln!("missing or unreadable: {f}");
            bad += 1;
            continue;
        };
        let (la, lb) = (a.len().unwrap_or(0), b.len().unwrap_or(u64::MAX));
        if la != lb {
            eprintln!("size differs: {f} ({la} and {lb})");
            bad += 1;
            continue;
        }
        let mut off = 0u64;
        while off < la {
            let n = a.read_at(off, &mut buf_a).unwrap_or(0);
            let m = b.read_at(off, &mut buf_b[..n]).unwrap_or(0);
            if n == 0 || m != n || buf_a[..n] != buf_b[..n] {
                eprintln!("content differs: {f} near {off:#x}");
                bad += 1;
                break;
            }
            off += n as u64;
        }
        bytes += la;
    }
    let extra = ours.len() as i64 - theirs.len() as i64;
    println!("{} files, {:.2} GiB compared in {:.0} s: {bad} differ; the archive has {} files ({extra:+} against the folder)", theirs.len(), gib(bytes), t0.elapsed().as_secs_f64(), ours.len());
    if bad > 0 || extra != 0 {
        1
    } else {
        0
    }
}

/// Every file of the archive read (decompressed) on `threads` threads, 8 MiB reads.
fn read_all(a: &Archive, files: &[u32], threads: usize) -> (u64, f64) {
    let next = std::sync::atomic::AtomicUsize::new(0);
    let total = std::sync::atomic::AtomicU64::new(0);
    let t0 = Instant::now();
    std::thread::scope(|s| {
        for _ in 0..threads {
            s.spawn(|| {
                let mut buf = vec![0u8; 8 << 20];
                loop {
                    let i = next.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
                    let Some(&n) = files.get(i) else { break };
                    let mut off = 0u64;
                    loop {
                        let got = a.read_at(n, off, &mut buf).expect("read");
                        if got == 0 {
                            break;
                        }
                        off += got as u64;
                    }
                    total.fetch_add(off, std::sync::atomic::Ordering::Relaxed);
                }
            });
        }
    });
    (total.into_inner(), t0.elapsed().as_secs_f64())
}

fn file_nodes(a: &Archive) -> Vec<u32> {
    let root = a.lookup("").unwrap();
    a.files_under(root).iter().filter_map(|p| a.lookup(p)).collect()
}

fn bench(archive: &Path, threads: Option<usize>, rounds: usize) -> i32 {
    let a = match Archive::open(archive) {
        Ok(a) => a,
        Err(e) => {
            eprintln!("open: {e}");
            return 1;
        }
    };
    let files = file_nodes(&a);
    let list: Vec<usize> = match threads {
        Some(t) => vec![t],
        None => vec![1, 2, 4, 8],
    };
    for t in list {
        for r in 0..rounds {
            let (bytes, s) = read_all(&a, &files, t);
            println!("{t} threads, round {r}: {:.2} GiB in {s:.2} s: {:.0} MiB/s", gib(bytes), bytes as f64 / s / (1 << 20) as f64);
        }
    }
    0
}

/// Every byte of the files and folders given, read in order with 8 MiB reads (the file cache
/// then holds them instead of what it held: the cold A/Bs' eviction), and the rate.
fn readall(paths: &[String]) -> i32 {
    use std::io::Read;
    let t0 = Instant::now();
    let mut files: Vec<PathBuf> = Vec::new();
    for p in paths {
        let p = PathBuf::from(p);
        if p.is_dir() {
            let mut stack = vec![p];
            while let Some(d) = stack.pop() {
                let Ok(rd) = std::fs::read_dir(&d) else { continue };
                for e in rd.flatten() {
                    if e.file_type().is_ok_and(|t| t.is_dir()) {
                        stack.push(e.path());
                    } else {
                        files.push(e.path());
                    }
                }
            }
        } else {
            files.push(p);
        }
    }
    let mut buf = vec![0u8; 8 << 20];
    let mut bytes = 0u64;
    for f in &files {
        let Ok(mut h) = std::fs::File::open(f) else { continue };
        while let Ok(n) = h.read(&mut buf) {
            if n == 0 {
                break;
            }
            bytes += n as u64;
        }
    }
    let s = t0.elapsed().as_secs_f64();
    println!("read {} files, {:.2} GiB in {s:.1} s: {:.0} MiB/s", files.len(), gib(bytes), bytes as f64 / s / (1 << 20) as f64);
    0
}

fn levels(folder: &Path, scratch: &Path, list: &str) -> i32 {
    #[cfg(target_os = "linux")]
    if let Err(e) = dsconfig::files::archive_output(folder, scratch) { eprintln!("scratch directory: {e}"); return 1; }
    let _ = std::fs::create_dir_all(scratch);
    for l in list.split(',').filter_map(|v| v.trim().parse::<i32>().ok()) {
        let out: PathBuf = scratch.join(format!("level{l}.zar"));
        let t0 = Instant::now();
        let w = match write_folder(folder, &out, l, threads(), &|_, _| true) {
            Ok(w) => w,
            Err(e) => {
                eprintln!("level {l}: {e}");
                return 1;
            }
        };
        let ws = t0.elapsed().as_secs_f64();
        let a = Archive::open(&out).expect("open");
        let files = file_nodes(&a);
        // (Twice: the first read takes the archive from the drive, the second from the cache.)
        let (_, _) = read_all(&a, &files, 1);
        let (b1, s1) = read_all(&a, &files, 1);
        println!(
            "level {l}: {:.3} GiB -> {:.3} GiB ({:.1}%), written in {ws:.1} s ({:.0} MiB/s), read back on 1 thread at {:.0} MiB/s",
            gib(w.bytes_in),
            gib(w.bytes_out),
            w.bytes_out as f64 * 100.0 / w.bytes_in.max(1) as f64,
            w.bytes_in as f64 / ws / (1 << 20) as f64,
            b1 as f64 / s1 / (1 << 20) as f64
        );
        drop(a);
        let _ = std::fs::remove_file(&out);
    }
    0
}
