//! `.zar` archives of the game: the game folder as one compressed file, read in place, and
//! written from a folder (the launcher's Compress).
//!
//! The format (version 1; every number big-endian; the public .zar format's layout, as
//! docs/PROVENANCE.md records it; this code is the port's own): the files' bytes one after
//! another, cut into 64 KiB blocks, each block a zstd frame, or kept as it is when it would not
//! shrink (then 64 KiB long); the last block padded with zeros; padding to 8 bytes. Then the
//! offset records, one per 16 blocks: the offset of its first block in the data section (u64)
//! and each block's stored size minus one (u16). Then the names, each once: a length byte (two
//! bytes, the first with its top bit set, for 128 and longer: the low 7 bits, then the rest),
//! then the bytes. Then the tree: 16-byte entries in breadth-first order, the root first: a
//! name offset (0x7FFFFFFF for the root) with the top bit set for a file; for a file its offset
//! in the data and its size, 48 bits each (the size's top 16 bits in the last word's upper half,
//! the offset's in its lower half); for a directory its first child's index and its child
//! count, the children sorted by name with ASCII letters folded to lower case. Then two empty
//! meta sections. Last a 144-byte footer: the six sections' offsets and sizes (data, offset
//! records, names, tree, meta directory, meta data), the SHA-256 of the whole file with this
//! hash zeroed, the file's size, the version 0x61BF3A01 and the magic 0x169F52D6.

use std::collections::HashMap;
use std::fs::File;
use std::io::{self, Read, Write};
#[cfg(windows)]
use std::os::windows::fs::FileExt;
#[cfg(target_os = "linux")]
use crate::files::FileExt;
use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::sync::Mutex;

pub const BLOCK: usize = 64 * 1024;
const PER_RECORD: usize = 16;
const RECORD_BYTES: usize = 8 + 2 * PER_RECORD;
const ENTRY_BYTES: usize = 16;
const FOOTER_BYTES: usize = 16 * 6 + 32 + 8 + 4 + 4;
const MAGIC: u32 = 0x169F_52D6;
const VERSION1: u32 = 0x61BF_3A01;
const ROOT_NAME: u32 = 0x7FFF_FFFF;
/// Decompressed blocks kept for reads that take part of a block (16 MiB).
const CACHE_BLOCKS: usize = 256;
/// File handles for reads from several threads at once (each read takes one: Windows serializes
/// the reads of one synchronous handle).
const HANDLES: usize = 16;
/// The most whole blocks read with one read of their stored bytes (16 MiB decompressed).
const MAX_RUN: u64 = 256;
/// Threads that decompress one read's blocks (1: the reading thread alone; `set_threads`).
static THREADS: std::sync::atomic::AtomicUsize = std::sync::atomic::AtomicUsize::new(1);
/// Blocks a read needs before its decompression is shared out (2 MiB).
const PAR_MIN: u64 = 32;

/// How many threads decompress one large read's blocks (1: the reading thread alone).
pub fn set_threads(n: usize) {
    THREADS.store(n.clamp(1, 16), std::sync::atomic::Ordering::Relaxed);
}

fn be16(b: &[u8]) -> u16 {
    u16::from_be_bytes([b[0], b[1]])
}
fn be32(b: &[u8]) -> u32 {
    u32::from_be_bytes(b[..4].try_into().unwrap())
}
fn be64(b: &[u8]) -> u64 {
    u64::from_be_bytes(b[..8].try_into().unwrap())
}
fn bad(why: &str) -> io::Error {
    io::Error::new(io::ErrorKind::InvalidData, format!("not a usable .zar archive: {why}"))
}

/// A tree entry: a file (its offset in the data and its size) or a directory (its first child's
/// index and its child count).
#[derive(Clone, Copy, Debug)]
struct Entry {
    name: u32,
    file: bool,
    a: u64,
    b: u64,
}

/// What a directory holds: a name, whether it is a directory, a file's size, its write time (in
/// an archive, the archive's own).
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct DirEntry {
    pub name: String,
    pub dir: bool,
    pub size: u64,
    pub modified: Option<std::time::SystemTime>,
}

pub struct Archive {
    path: PathBuf,
    size: u64,
    /// The archive file's write time (its entries' own: the format keeps none).
    modified: Option<std::time::SystemTime>,
    handles: Mutex<Vec<File>>,
    data_off: u64,
    data_size: u64,
    records: Vec<(u64, [u16; PER_RECORD])>,
    names: Vec<u8>,
    tree: Vec<Entry>,
    /// Every node by its path from the root (lower case, `/` between names; the root is "").
    by_path: HashMap<String, u32>,
    cache: Mutex<Cache>,
}

#[derive(Default)]
struct Cache {
    blocks: HashMap<u64, (Arc<Vec<u8>>, u64)>,
    clock: u64,
}

thread_local! {
    static DCTX: std::cell::RefCell<Option<zstd::bulk::Decompressor<'static>>> = const { std::cell::RefCell::new(None) };
    /// A run's stored bytes (`load_run`), kept for the thread's next read.
    static STORED: std::cell::RefCell<Vec<u8>> = const { std::cell::RefCell::new(Vec::new()) };
}

/// A block's stored bytes into `out` (64 KiB): copied when kept as it was, else decompressed.
fn decode(stored: &[u8], out: &mut [u8]) -> io::Result<()> {
    if stored.len() == BLOCK {
        out.copy_from_slice(stored);
        return Ok(());
    }
    let n = DCTX.with(|c| -> io::Result<usize> {
        let mut c = c.borrow_mut();
        if c.is_none() {
            *c = Some(zstd::bulk::Decompressor::new()?);
        }
        c.as_mut().unwrap().decompress_to_buffer(stored, out)
    })?;
    if n != BLOCK {
        return Err(bad("a block decompressed to the wrong size"));
    }
    Ok(())
}

/// The name at `off` in the names section ("" for the root's or a bad offset).
fn name_at(names: &[u8], off: u32) -> &str {
    let o = off as usize;
    if off == ROOT_NAME || o >= names.len() {
        return "";
    }
    let (len, at) = if names[o] & 0x80 != 0 {
        if o + 1 >= names.len() {
            return "";
        }
        ((names[o] & 0x7F) as usize | (names[o + 1] as usize) << 7, o + 2)
    } else {
        (names[o] as usize, o + 1)
    };
    names.get(at..at + len).and_then(|n| std::str::from_utf8(n).ok()).unwrap_or("")
}

/// A relative path's names, `/` or `\` between them, empty ones and `.` skipped.
fn parts(path: &str) -> impl Iterator<Item = &str> {
    path.split(['/', '\\']).filter(|p| !p.is_empty() && *p != ".")
}

/// A path's key in `by_path`: its names in lower case (ASCII letters), `/` between them.
fn key(path: &str) -> String {
    parts(path).map(|p| p.to_ascii_lowercase()).collect::<Vec<_>>().join("/")
}

impl Archive {
    /// Opens an archive: its footer checked, its offset records, names and tree read.
    pub fn open(path: &Path) -> io::Result<Archive> {
        let f = File::open(path)?;
        let md = f.metadata()?;
        let size = md.len();
        if size <= FOOTER_BYTES as u64 {
            return Err(bad("too short"));
        }
        let mut ft = [0u8; FOOTER_BYTES];
        SeekReadExact::seek_read_exact(&f, &mut ft, size - FOOTER_BYTES as u64)?;
        let sec: Vec<(u64, u64)> = (0..6).map(|i| (be64(&ft[i * 16..]), be64(&ft[i * 16 + 8..]))).collect();
        let (total, version, magic) = (be64(&ft[128..]), be32(&ft[136..]), be32(&ft[140..]));
        if magic != MAGIC || version != VERSION1 {
            return Err(bad("no .zar footer (another format or version)"));
        }
        if total != size {
            return Err(bad("its size is not the size its footer gives (cut short?)"));
        }
        if sec.iter().any(|&(o, s)| o.checked_add(s).is_none_or(|e| e > size)) {
            return Err(bad("a section reaches past its end"));
        }
        let read = |(o, s): (u64, u64), limit: u64| -> io::Result<Vec<u8>> {
            if s > limit {
                return Err(bad("a section is too large"));
            }
            let mut v = vec![0u8; s as usize];
            SeekReadExact::seek_read_exact(&f, &mut v, o)?;
            Ok(v)
        };
        let rec = read(sec[1], u32::MAX as u64)?;
        let names = read(sec[2], 0x7FFF_FFFF)?;
        let tree_raw = read(sec[3], u32::MAX as u64)?;
        if rec.is_empty() || rec.len() % RECORD_BYTES != 0 || tree_raw.is_empty() || tree_raw.len() % ENTRY_BYTES != 0 {
            return Err(bad("its offset records or tree are malformed"));
        }
        let records: Vec<(u64, [u16; PER_RECORD])> = rec
            .chunks_exact(RECORD_BYTES)
            .map(|r| {
                let mut s = [0u16; PER_RECORD];
                for (i, v) in s.iter_mut().enumerate() {
                    *v = be16(&r[8 + 2 * i..]);
                }
                (be64(r), s)
            })
            .collect();
        let tree: Vec<Entry> = tree_raw
            .chunks_exact(ENTRY_BYTES)
            .map(|e| {
                let (w0, w1, w2, w3) = (be32(e), be32(&e[4..]), be32(&e[8..]), be32(&e[12..]));
                if w0 & 0x8000_0000 != 0 {
                    let off = w1 as u64 | ((w3 & 0xFFFF) as u64) << 32;
                    let size = w2 as u64 | ((w3 >> 16) as u64) << 32;
                    Entry { name: w0 & 0x7FFF_FFFF, file: true, a: off, b: size }
                } else {
                    Entry { name: w0 & 0x7FFF_FFFF, file: false, a: w1 as u64, b: w2 as u64 }
                }
            })
            .collect();
        if tree[0].file || !name_at(&names, tree[0].name).is_empty() {
            return Err(bad("its first entry is not the root"));
        }
        // (Every record within the data and every file within the blocks the records describe: a
        // damaged archive is refused here, not met as a huge read later.)
        let capacity = records.len() as u64 * PER_RECORD as u64 * BLOCK as u64;
        if records.iter().any(|(b, _)| *b > sec[0].1) {
            return Err(bad("an offset record past its data"));
        }
        if tree.iter().any(|e| e.file && e.a.checked_add(e.b).is_none_or(|end| end > capacity)) {
            return Err(bad("a file past its data"));
        }
        // Every node's path, the tree walked from the root (a directory's children past the
        // end, or a cycle back to the root, refuse the archive).
        let mut by_path = HashMap::with_capacity(tree.len());
        by_path.insert(String::new(), 0u32);
        let mut queue = std::collections::VecDeque::from([(0u32, String::new())]);
        let mut seen = 1usize;
        while let Some((n, p)) = queue.pop_front() {
            let e = tree[n as usize];
            if e.file {
                continue;
            }
            let (first, count) = (e.a, e.b);
            if count > 0 && (first == 0 || first.checked_add(count).is_none_or(|end| end > tree.len() as u64)) {
                return Err(bad("a directory's entries are out of range"));
            }
            for c in first..first + count {
                seen += 1;
                if seen > tree.len() + 1 {
                    return Err(bad("its tree loops"));
                }
                let name = name_at(&names, tree[c as usize].name);
                if name.is_empty() {
                    return Err(bad("an entry has no name"));
                }
                let cp = if p.is_empty() { name.to_ascii_lowercase() } else { format!("{p}/{}", name.to_ascii_lowercase()) };
                if cp.len() > 4096 {
                    return Err(bad("a path too long"));
                }
                by_path.insert(cp.clone(), c as u32);
                queue.push_back((c as u32, cp));
            }
        }
        Ok(Archive {
            path: path.to_path_buf(),
            size,
            modified: md.modified().ok(),
            handles: Mutex::new(vec![f]),
            data_off: sec[0].0,
            data_size: sec[0].1,
            records,
            names,
            tree,
            by_path,
            cache: Mutex::new(Cache::default()),
        })
    }

    pub fn path(&self) -> &Path {
        &self.path
    }

    /// The archive file's size.
    pub fn size(&self) -> u64 {
        self.size
    }

    /// The archive file's write time, which its entries report as theirs.
    pub fn modified(&self) -> Option<std::time::SystemTime> {
        self.modified
    }

    /// The node at a path from the root (`/` or `\` between names; letters' case ignored).
    pub fn lookup(&self, path: &str) -> Option<u32> {
        self.by_path.get(&key(path)).copied()
    }

    pub fn is_file(&self, node: u32) -> bool {
        self.tree.get(node as usize).is_some_and(|e| e.file)
    }

    pub fn is_dir(&self, node: u32) -> bool {
        self.tree.get(node as usize).is_some_and(|e| !e.file)
    }

    /// A file's size (0 for a directory).
    pub fn file_size(&self, node: u32) -> u64 {
        self.tree.get(node as usize).filter(|e| e.file).map_or(0, |e| e.b)
    }

    /// A directory's entries, in the archive's order.
    pub fn list(&self, node: u32) -> Vec<DirEntry> {
        let Some(e) = self.tree.get(node as usize).filter(|e| !e.file) else { return Vec::new() };
        (e.a..e.a + e.b)
            .filter_map(|c| self.tree.get(c as usize))
            .map(|c| DirEntry { name: name_at(&self.names, c.name).to_string(), dir: !c.file, size: if c.file { c.b } else { 0 }, modified: self.modified })
            .collect()
    }

    /// Every file under directory `node`, by its path from there (`/` between names).
    pub fn files_under(&self, node: u32) -> Vec<String> {
        let mut out = Vec::new();
        let mut stack = vec![(node, String::new())];
        while let Some((n, p)) = stack.pop() {
            for (i, d) in self.list(n).into_iter().enumerate() {
                let child = self.tree[n as usize].a as u32 + i as u32;
                let cp = if p.is_empty() { d.name.clone() } else { format!("{p}/{}", d.name) };
                if d.dir {
                    stack.push((child, cp));
                } else {
                    out.push(cp);
                }
            }
        }
        out.sort();
        out
    }

    fn with_handle<T>(&self, f: impl FnOnce(&File) -> io::Result<T>) -> io::Result<T> {
        let h = self.handles.lock().unwrap().pop();
        let h = match h {
            Some(h) => h,
            None => File::open(&self.path)?,
        };
        let r = f(&h);
        let mut pool = self.handles.lock().unwrap();
        if pool.len() < HANDLES {
            pool.push(h);
        }
        r
    }

    /// Where block `i` is stored: its offset in the data section and its stored size.
    fn block_at(&self, i: u64) -> io::Result<(u64, usize)> {
        let (r, k) = ((i / PER_RECORD as u64) as usize, (i % PER_RECORD as u64) as usize);
        let (base, sizes) = self.records.get(r).ok_or_else(|| bad("a block past its records"))?;
        let off = base.checked_add(sizes[..k].iter().map(|&s| s as u64 + 1).sum::<u64>()).ok_or_else(|| bad("a block past its data"))?;
        let stored = sizes[k] as usize + 1;
        if off.checked_add(stored as u64).is_none_or(|end| end > self.data_size) {
            return Err(bad("a block past its data"));
        }
        Ok((off, stored))
    }

    /// Block `i` of the data, decompressed, into `out` (64 KiB).
    fn load_block(&self, i: u64, out: &mut [u8]) -> io::Result<()> {
        let (off, stored) = self.block_at(i)?;
        let at = self.data_off + off;
        if stored == BLOCK {
            return self.with_handle(|h| SeekReadExact::seek_read_exact(h, out, at));
        }
        let mut src = vec![0u8; stored];
        self.with_handle(|h| SeekReadExact::seek_read_exact(h, &mut src, at))?;
        decode(&src, out)
    }

    /// Blocks `first..first + count` decompressed into `out` (`count` blocks long): their stored
    /// bytes, which lie together, taken with one read, then decompressed straight into `out` (on
    /// several threads for a large read: `set_threads`).
    fn load_run(&self, first: u64, count: u64, out: &mut [u8]) -> io::Result<()> {
        if count == 1 {
            return self.load_block(first, out);
        }
        let spans: Vec<(u64, usize)> = (first..first + count).map(|i| self.block_at(i)).collect::<io::Result<_>>()?;
        let start = spans[0].0;
        if !spans.windows(2).all(|w| w[0].0 + w[0].1 as u64 == w[1].0) {
            // (Not one after another, as another writer may store them: block by block.)
            for (k, chunk) in out.chunks_mut(BLOCK).enumerate() {
                self.load_block(first + k as u64, chunk)?;
            }
            return Ok(());
        }
        let len = (spans[spans.len() - 1].0 + spans[spans.len() - 1].1 as u64 - start) as usize;
        STORED.with(|b| -> io::Result<()> {
            let mut src = b.borrow_mut();
            src.resize(len, 0);
            self.with_handle(|h| SeekReadExact::seek_read_exact(h, &mut src[..len], self.data_off + start))?;
            let src = &src[..len];
            let threads = THREADS.load(std::sync::atomic::Ordering::Relaxed).min(count.div_ceil(PAR_MIN / 2) as usize);
            if threads <= 1 || count < PAR_MIN {
                for (&(o, s), chunk) in spans.iter().zip(out.chunks_mut(BLOCK)) {
                    decode(&src[(o - start) as usize..(o - start) as usize + s], chunk)?;
                }
                return Ok(());
            }
            // (Each thread a share of the blocks, in order; the reading thread takes the first.)
            let per = (spans.len()).div_ceil(threads);
            std::thread::scope(|sc| -> io::Result<()> {
                let mut parts = spans.chunks(per).zip(out.chunks_mut(per * BLOCK));
                let first_part = parts.next();
                let handles: Vec<_> = parts
                    .map(|(sp, o)| {
                        sc.spawn(move || -> io::Result<()> {
                            for (&(off, s), chunk) in sp.iter().zip(o.chunks_mut(BLOCK)) {
                                decode(&src[(off - start) as usize..(off - start) as usize + s], chunk)?;
                            }
                            Ok(())
                        })
                    })
                    .collect();
                let mut r = Ok(());
                if let Some((sp, o)) = first_part {
                    for (&(off, s), chunk) in sp.iter().zip(o.chunks_mut(BLOCK)) {
                        if let Err(e) = decode(&src[(off - start) as usize..(off - start) as usize + s], chunk) {
                            r = Err(e);
                            break;
                        }
                    }
                }
                for h in handles {
                    match h.join() {
                        Ok(Ok(())) => {}
                        Ok(Err(e)) => r = r.and(Err(e)),
                        Err(_) => r = r.and(Err(io::Error::other("a decompressing thread failed"))),
                    }
                }
                r
            })
        })
    }

    /// Block `i` through the cache.
    fn cached_block(&self, i: u64) -> io::Result<Arc<Vec<u8>>> {
        {
            let mut c = self.cache.lock().unwrap();
            c.clock += 1;
            let t = c.clock;
            if let Some(b) = c.blocks.get_mut(&i) {
                b.1 = t;
                return Ok(b.0.clone());
            }
        }
        let mut v = vec![0u8; BLOCK];
        self.load_block(i, &mut v)?;
        let v = Arc::new(v);
        let mut c = self.cache.lock().unwrap();
        if c.blocks.len() >= CACHE_BLOCKS {
            if let Some(old) = c.blocks.iter().min_by_key(|(_, (_, t))| *t).map(|(k, _)| *k) {
                c.blocks.remove(&old);
            }
        }
        c.clock += 1;
        let t = c.clock;
        c.blocks.insert(i, (v.clone(), t));
        Ok(v)
    }

    /// Reads file `node` from `off` into `buf`; the bytes read (fewer at its end).
    pub fn read_at(&self, node: u32, off: u64, buf: &mut [u8]) -> io::Result<usize> {
        let e = *self.tree.get(node as usize).filter(|e| e.file).ok_or_else(|| io::Error::new(io::ErrorKind::NotFound, "not a file in the archive"))?;
        if off >= e.b {
            return Ok(0);
        }
        let n = buf.len().min((e.b - off) as usize);
        let mut pos = e.a + off;
        let mut done = 0;
        while done < n {
            let (blk, within) = (pos / BLOCK as u64, (pos % BLOCK as u64) as usize);
            let step = (BLOCK - within).min(n - done);
            if within == 0 && step == BLOCK {
                // (Whole blocks: straight into the caller's buffer, a run of them at once.)
                let run = (((n - done) / BLOCK) as u64).min(MAX_RUN);
                let bytes = run as usize * BLOCK;
                self.load_run(blk, run, &mut buf[done..done + bytes])?;
                done += bytes;
                pos += bytes as u64;
                continue;
            }
            let b = self.cached_block(blk)?;
            buf[done..done + step].copy_from_slice(&b[within..within + step]);
            done += step;
            pos += step as u64;
        }
        Ok(n)
    }

    /// File `node` whole.
    pub fn read_all(&self, node: u32) -> io::Result<Vec<u8>> {
        // (Whole files are the small ones: the game's settings, its modules, the shader bundles.)
        if self.file_size(node) > 1 << 30 {
            return Err(io::Error::new(io::ErrorKind::InvalidData, "a file too large to read whole"));
        }
        let mut v = vec![0u8; self.file_size(node) as usize];
        let n = self.read_at(node, 0, &mut v)?;
        v.truncate(n);
        Ok(v)
    }
}

trait SeekReadExact {
    fn seek_read_exact(&self, buf: &mut [u8], off: u64) -> io::Result<()>;
}

impl SeekReadExact for File {
    fn seek_read_exact(&self, mut buf: &mut [u8], mut off: u64) -> io::Result<()> {
        while !buf.is_empty() {
            match self.seek_read(buf, off) {
                Ok(0) => return Err(io::Error::new(io::ErrorKind::UnexpectedEof, "the archive ends early")),
                Ok(n) => {
                    buf = &mut buf[n..];
                    off += n as u64;
                }
                Err(e) if e.kind() == io::ErrorKind::Interrupted => {}
                Err(e) => return Err(e),
            }
        }
        Ok(())
    }
}

// ---------------------------------------------------------------------------
// The game's files from a folder or an archive

/// Where the game's files come from: a folder, or an archive (from `base`, the archive's folder
/// that holds the game, "" for its root).
#[derive(Clone)]
pub enum Source {
    Dir(PathBuf),
    Zar { archive: Arc<Archive>, base: String },
}

/// One of the game's files, open for reads at any offset.
pub enum SourceFile {
    Host(File),
    Zar { archive: Arc<Archive>, node: u32 },
}

impl SourceFile {
    pub fn read_at(&self, off: u64, buf: &mut [u8]) -> io::Result<usize> {
        match self {
            SourceFile::Host(f) => f.seek_read(buf, off),
            SourceFile::Zar { archive, node } => archive.read_at(*node, off, buf),
        }
    }

    pub fn len(&self) -> io::Result<u64> {
        match self {
            SourceFile::Host(f) => Ok(f.metadata()?.len()),
            SourceFile::Zar { archive, node } => Ok(archive.file_size(*node)),
        }
    }

    pub fn is_empty(&self) -> io::Result<bool> {
        Ok(self.len()? == 0)
    }
}

/// Whether `p` names an archive (by its extension; a folder so named is a folder).
pub fn is_archive_path(p: &Path) -> bool {
    p.extension().is_some_and(|e| e.eq_ignore_ascii_case("zar")) && !p.is_dir()
}

impl Source {
    /// The game at `path`: an archive (`.zar`, the game in its root or in a folder of it that
    /// holds `sce_sys/param.json`) or a folder.
    pub fn open(path: &Path) -> io::Result<Source> {
        if !is_archive_path(path) {
            return Ok(Source::Dir(path.to_path_buf()));
        }
        let a = Arc::new(Archive::open(path)?);
        let base = if a.lookup("sce_sys/param.json").is_some_and(|n| a.is_file(n)) {
            String::new()
        } else {
            a.list(0)
                .into_iter()
                .filter(|d| d.dir)
                .map(|d| d.name)
                .find(|n| a.lookup(&format!("{n}/sce_sys/param.json")).is_some_and(|x| a.is_file(x)))
                .unwrap_or_default()
        };
        Ok(Source::Zar { archive: a, base })
    }

    /// For logs: the folder, or the archive and its folder.
    pub fn display(&self) -> String {
        match self {
            Source::Dir(p) => p.display().to_string(),
            Source::Zar { archive, base } if base.is_empty() => archive.path().display().to_string(),
            Source::Zar { archive, base } => format!("{} ({base})", archive.path().display()),
        }
    }

    pub fn is_archive(&self) -> bool {
        matches!(self, Source::Zar { .. })
    }

    /// The folder or archive file it reads.
    pub fn path(&self) -> &Path {
        match self {
            Source::Dir(p) => p,
            Source::Zar { archive, .. } => archive.path(),
        }
    }

    fn node(&self, rel: &str) -> Option<(&Arc<Archive>, u32)> {
        match self {
            Source::Zar { archive, base } => {
                let p = if base.is_empty() { rel.to_string() } else { format!("{base}/{rel}") };
                archive.lookup(&p).map(|n| (archive, n))
            }
            Source::Dir(_) => None,
        }
    }

    fn host(&self, rel: &str) -> io::Result<PathBuf> {
        match self {
            Source::Dir(d) => crate::files::game_path(d, rel),
            Source::Zar { .. } => Err(bad("archive has no host path")),
        }
    }

    pub fn is_file(&self, rel: &str) -> bool {
        match self {
            Source::Dir(_) => self.host(rel).is_ok_and(|p| p.is_file()),
            Source::Zar { .. } => self.node(rel).is_some_and(|(a, n)| a.is_file(n)),
        }
    }

    pub fn is_dir(&self, rel: &str) -> bool {
        match self {
            Source::Dir(_) => self.host(rel).is_ok_and(|p| p.is_dir()),
            Source::Zar { .. } => self.node(rel).is_some_and(|(a, n)| a.is_dir(n)),
        }
    }

    /// A file's write time (in an archive, the archive's own).
    pub fn modified(&self, rel: &str) -> Option<std::time::SystemTime> {
        match self {
            Source::Dir(_) => std::fs::metadata(self.host(rel).ok()?).ok()?.modified().ok(),
            Source::Zar { archive, .. } => self.node(rel).and(archive.modified()),
        }
    }

    /// A file's size.
    pub fn len(&self, rel: &str) -> io::Result<u64> {
        match self {
            Source::Dir(_) => Ok(std::fs::metadata(self.host(rel)?)?.len()),
            Source::Zar { .. } => match self.node(rel) {
                Some((a, n)) if a.is_file(n) => Ok(a.file_size(n)),
                _ => Err(io::Error::new(io::ErrorKind::NotFound, format!("{rel}: not in the archive"))),
            },
        }
    }

    /// A file whole.
    pub fn read(&self, rel: &str) -> io::Result<Vec<u8>> {
        match self {
            Source::Dir(_) => std::fs::read(self.host(rel)?),
            Source::Zar { .. } => match self.node(rel) {
                Some((a, n)) if a.is_file(n) => a.read_all(n),
                _ => Err(io::Error::new(io::ErrorKind::NotFound, format!("{rel}: not in the archive"))),
            },
        }
    }

    /// A file open for reads at any offset.
    pub fn open_file(&self, rel: &str) -> io::Result<SourceFile> {
        match self {
            Source::Dir(_) => Ok(SourceFile::Host(File::open(self.host(rel)?)?)),
            Source::Zar { .. } => match self.node(rel) {
                Some((a, n)) if a.is_file(n) => Ok(SourceFile::Zar { archive: a.clone(), node: n }),
                _ => Err(io::Error::new(io::ErrorKind::NotFound, format!("{rel}: not in the archive"))),
            },
        }
    }

    /// A directory's entries.
    pub fn list(&self, rel: &str) -> io::Result<Vec<DirEntry>> {
        match self {
            Source::Dir(_) => {
                let mut out = Vec::new();
                for e in std::fs::read_dir(self.host(rel)?)?.flatten() {
                    let md = e.metadata()?;
                    out.push(DirEntry { name: e.file_name().to_string_lossy().into_owned(), dir: md.is_dir(), size: if md.is_dir() { 0 } else { md.len() }, modified: md.modified().ok() });
                }
                // Match NTFS/archive enumeration on Linux too: shader discovery order affects
                // which duplicate program supplies its resource variants and fingerprint.
                #[cfg(target_os = "linux")]
                out.sort_by(|x, y| x.name.bytes().map(|c| c.to_ascii_uppercase()).cmp(y.name.bytes().map(|c| c.to_ascii_uppercase())));
                Ok(out)
            }
            Source::Zar { .. } => match self.node(rel) {
                Some((a, n)) if a.is_dir(n) => {
                    // (In the order NTFS lists a folder, letters compared in upper case: what
                    // depends on the order, the shader lists, meets the files as from the folder.)
                    let mut l = a.list(n);
                    l.sort_by(|x, y| x.name.bytes().map(|c| c.to_ascii_uppercase()).cmp(y.name.bytes().map(|c| c.to_ascii_uppercase())));
                    Ok(l)
                }
                _ => Err(io::Error::new(io::ErrorKind::NotFound, format!("{rel}: not in the archive"))),
            },
        }
    }

    /// Every file under a directory, by its path from it (`/` between names), sorted.
    pub fn files_under(&self, rel: &str) -> Vec<String> {
        match self {
            Source::Dir(_) => {
                let Ok(root) = self.host(rel) else { return Vec::new() };
                let mut out = Vec::new();
                let mut stack = vec![(root, String::new())];
                while let Some((d, p)) = stack.pop() {
                    let Ok(rd) = std::fs::read_dir(&d) else { continue };
                    for e in rd.flatten() {
                        let name = e.file_name().to_string_lossy().into_owned();
                        let cp = if p.is_empty() { name } else { format!("{p}/{name}") };
                        if e.file_type().is_ok_and(|t| t.is_dir()) {
                            stack.push((e.path(), cp));
                        } else {
                            out.push(cp);
                        }
                    }
                }
                out.sort();
                out
            }
            Source::Zar { .. } => match self.node(rel) {
                Some((a, n)) if a.is_dir(n) => a.files_under(n),
                _ => Vec::new(),
            },
        }
    }
}

// ---------------------------------------------------------------------------
// Writing an archive from a folder

/// The zstd level `write_folder` is given by the launcher.
pub const LEVEL: i32 = 6;

/// A folder's files and their bytes (all of it, its folders' too).
pub fn folder_size(dir: &Path) -> io::Result<(u64, u64)> {
    let (mut files, mut bytes) = (0u64, 0u64);
    let mut stack = vec![dir.to_path_buf()];
    while let Some(d) = stack.pop() {
        for e in std::fs::read_dir(&d)? {
            let e = e?;
            let md = e.metadata()?;
            if md.is_dir() {
                stack.push(e.path());
            } else {
                files += 1;
                bytes += md.len();
            }
        }
    }
    Ok((files, bytes))
}

/// What a write made: files, their bytes, the archive's size.
#[derive(Clone, Copy, Debug, Default)]
pub struct Written {
    pub files: u64,
    pub bytes_in: u64,
    pub bytes_out: u64,
}

/// A name's order among its directory's: ASCII letters folded to lower case, byte by byte, a
/// shorter name before a longer one that starts with it (the format's own order).
fn name_order(a: &str, b: &str) -> std::cmp::Ordering {
    a.bytes().map(|c| c.to_ascii_lowercase()).cmp(b.bytes().map(|c| c.to_ascii_lowercase()))
}

struct Node {
    name: String,
    file: Option<(u64, u64)>, // offset in the data and size
    children: Vec<usize>,
}

/// Writes `out` from folder `src` (its tree as it is, files sorted by path): `level` is zstd's
/// level (the format's own writer uses 6), `threads` the compressing threads. `progress(done,
/// total)` is told the input bytes read and returns false to stop (the partial file is removed,
/// and an `Interrupted` error returned). The archive is written beside `out` and renamed to it
/// once complete.
pub fn write_folder(src: &Path, out: &Path, level: i32, threads: usize, progress: &dyn Fn(u64, u64) -> bool) -> io::Result<Written> {
    #[cfg(target_os = "linux")]
    crate::files::archive_output(src, out)?;
    // The tree, in a stable order: names sorted (the format's order) within each folder.
    let mut nodes = vec![Node { name: String::new(), file: None, children: Vec::new() }];
    let mut files: Vec<(usize, PathBuf, u64)> = Vec::new();
    let mut stack = vec![(0usize, src.to_path_buf())];
    while let Some((at, dir)) = stack.pop() {
        let mut ents: Vec<(String, PathBuf, bool, u64)> = Vec::new();
        for e in std::fs::read_dir(&dir)? {
            let e = e?;
            let md = e.metadata()?;
            let name = e.file_name().into_string().map_err(|_| io::Error::new(io::ErrorKind::InvalidData, format!("{}: a name that is not Unicode", e.path().display())))?;
            ents.push((name, e.path(), md.is_dir(), md.len()));
        }
        ents.sort_by(|a, b| name_order(&a.0, &b.0));
        for w in ents.windows(2) {
            if w[0].0.eq_ignore_ascii_case(&w[1].0) {
                return Err(io::Error::new(io::ErrorKind::InvalidData, format!("{}: two names that differ only in case", w[1].1.display())));
            }
        }
        for (name, path, is_dir, len) in ents {
            let i = nodes.len();
            nodes.push(Node { name, file: if is_dir { None } else { Some((0, len)) }, children: Vec::new() });
            nodes[at].children.push(i);
            if is_dir {
                stack.push((i, path));
            } else {
                files.push((i, path, len));
            }
        }
    }
    // (Files in the order their folders were read, so a folder's files lie together.)
    files.sort_by(|a, b| a.0.cmp(&b.0));
    let total: u64 = files.iter().map(|f| f.2).sum();

    let partial = out.with_extension("zar.partial");
    let r = (|| -> io::Result<Written> {
        let threads = threads.clamp(1, 64);
        let file = File::create(&partial)?;
        // The data: the blocks numbered in reading order, compressed on `threads` threads (each
        // with its own compressor) while this thread reads on, and written in that order by a
        // writer thread that hashes them. (One after another, the whole game took 300 s on the
        // tested PC, 2026-11-23.)
        type Packed = (u64, io::Result<Vec<u8>>, Vec<u8>);
        type Data = (io::BufWriter<File>, crate::sha::Sha256, Vec<(u64, [u16; PER_RECORD])>, u64);
        let (mut offset, mut done) = (0u64, 0u64);
        let (mut w, mut sha, records, mut written) = std::thread::scope(|s| -> io::Result<Data> {
            let (job_tx, job_rx) = std::sync::mpsc::sync_channel::<(u64, Vec<u8>)>(threads * 4);
            let job_rx = Arc::new(Mutex::new(job_rx));
            let (done_tx, done_rx) = std::sync::mpsc::sync_channel::<Packed>(threads * 4);
            for _ in 0..threads {
                let (rx, tx) = (job_rx.clone(), done_tx.clone());
                s.spawn(move || {
                    let mut c = zstd::bulk::Compressor::new(level);
                    loop {
                        let job = rx.lock().unwrap().recv();
                        let Ok((seq, raw)) = job else { break };
                        let packed = match &mut c {
                            Ok(c) => c.compress(&raw),
                            Err(e) => Err(io::Error::new(e.kind(), e.to_string())),
                        };
                        if tx.send((seq, packed, raw)).is_err() {
                            break;
                        }
                    }
                });
            }
            // (Once the compressing threads are gone, the reader's sends fail and the writer's
            // receiving ends.)
            drop(job_rx);
            drop(done_tx);
            let writer = s.spawn(move || -> io::Result<Data> {
                let mut w = io::BufWriter::with_capacity(8 << 20, file);
                let mut sha = crate::sha::Sha256::new()?;
                let mut records: Vec<(u64, [u16; PER_RECORD])> = Vec::new();
                let mut written = 0u64;
                let mut next = 0u64;
                // (Blocks compressed ahead of the next one in order wait here.)
                let mut ahead: HashMap<u64, (io::Result<Vec<u8>>, Vec<u8>)> = HashMap::new();
                for (seq, packed, raw) in done_rx {
                    ahead.insert(seq, (packed, raw));
                    while let Some((packed, raw)) = ahead.remove(&next) {
                        let p = packed?;
                        // (Kept as it is when compressing did not make it smaller.)
                        let stored: &[u8] = if p.len() >= BLOCK { &raw } else { &p };
                        if next % PER_RECORD as u64 == 0 {
                            records.push((written, [0; PER_RECORD]));
                        }
                        records.last_mut().unwrap().1[(next % PER_RECORD as u64) as usize] = (stored.len() - 1) as u16;
                        w.write_all(stored)?;
                        sha.update(stored)?;
                        written += stored.len() as u64;
                        next += 1;
                    }
                }
                if !ahead.is_empty() {
                    return Err(io::Error::other("a block was not compressed"));
                }
                Ok((w, sha, records, written))
            });
            // The reader: this thread.
            let read = (|| -> io::Result<()> {
                let stopped = || io::Error::other("the compressing threads stopped");
                let mut seq = 0u64;
                let mut cur: Vec<u8> = Vec::with_capacity(BLOCK);
                let mut buf = vec![0u8; 8 << 20];
                for (node, path, _) in &files {
                    let mut f = File::open(path)?;
                    let start = offset;
                    loop {
                        let n = f.read(&mut buf)?;
                        if n == 0 {
                            break;
                        }
                        let mut rest = &buf[..n];
                        while !rest.is_empty() {
                            let take = (BLOCK - cur.len()).min(rest.len());
                            cur.extend_from_slice(&rest[..take]);
                            rest = &rest[take..];
                            if cur.len() == BLOCK {
                                job_tx.send((seq, std::mem::replace(&mut cur, Vec::with_capacity(BLOCK)))).map_err(|_| stopped())?;
                                seq += 1;
                            }
                        }
                        offset += n as u64;
                        done += n as u64;
                        if !progress(done, total) {
                            return Err(io::Error::new(io::ErrorKind::Interrupted, "stopped"));
                        }
                    }
                    // (The size read, not the size listed: a file that changed meanwhile.)
                    nodes[*node].file = Some((start, offset - start));
                }
                // (The last block padded with zeros; an archive of empty files gets one block of
                // zeros, so that it has offset records.)
                if !cur.is_empty() || seq == 0 {
                    cur.resize(BLOCK, 0);
                    job_tx.send((seq, cur)).map_err(|_| stopped())?;
                }
                Ok(())
            })();
            drop(job_tx);
            // (The writer's own error first: the reader's then only says that it stopped.)
            let wrote = writer.join().map_err(|_| io::Error::other("the writing thread failed"))??;
            read?;
            Ok(wrote)
        })?;
        let emit = |w: &mut io::BufWriter<File>, sha: &mut crate::sha::Sha256, b: &[u8], written: &mut u64| -> io::Result<()> {
            w.write_all(b)?;
            sha.update(b)?;
            *written += b.len() as u64;
            Ok(())
        };
        let data = (0u64, written);
        while written % 8 != 0 {
            emit(&mut w, &mut sha, &[0], &mut written)?;
        }
        // The offset records.
        let rec_at = written;
        for (base, sizes) in &records {
            let mut r = [0u8; RECORD_BYTES];
            r[..8].copy_from_slice(&base.to_be_bytes());
            for (i, s) in sizes.iter().enumerate() {
                r[8 + 2 * i..10 + 2 * i].copy_from_slice(&s.to_be_bytes());
            }
            emit(&mut w, &mut sha, &r, &mut written)?;
        }
        let rec = (rec_at, written - rec_at);
        // The names, each once.
        let names_at = written;
        let mut name_off: HashMap<&str, u32> = HashMap::new();
        let mut names_len = 0u32;
        for n in nodes.iter().skip(1) {
            if name_off.contains_key(n.name.as_str()) {
                continue;
            }
            let b = n.name.as_bytes();
            if b.len() > 0x7FFF {
                return Err(io::Error::new(io::ErrorKind::InvalidData, format!("{}: a name too long for the format", n.name)));
            }
            name_off.insert(&n.name, names_len);
            let head: Vec<u8> = if b.len() >= 0x80 { vec![(b.len() & 0x7F) as u8 | 0x80, (b.len() >> 7) as u8] } else { vec![b.len() as u8] };
            emit(&mut w, &mut sha, &head, &mut written)?;
            emit(&mut w, &mut sha, b, &mut written)?;
            names_len += (head.len() + b.len()) as u32;
        }
        let names = (names_at, written - names_at);
        // The tree in breadth-first order: each folder's children together, from `first`.
        let mut order = vec![0usize];
        let mut first = vec![0u32; nodes.len()];
        let mut q = 0;
        while q < order.len() {
            let n = order[q];
            q += 1;
            if nodes[n].file.is_none() {
                first[n] = order.len() as u32;
                order.extend(nodes[n].children.iter().copied());
            }
        }
        let tree_at = written;
        for &n in &order {
            let node = &nodes[n];
            let mut e = [0u8; ENTRY_BYTES];
            let name = if n == 0 { ROOT_NAME } else { name_off[node.name.as_str()] };
            match node.file {
                Some((off, size)) => {
                    if off >> 48 != 0 || size >> 48 != 0 {
                        return Err(io::Error::new(io::ErrorKind::InvalidData, "a file past the format's 256 TiB"));
                    }
                    e[..4].copy_from_slice(&(name | 0x8000_0000).to_be_bytes());
                    e[4..8].copy_from_slice(&(off as u32).to_be_bytes());
                    e[8..12].copy_from_slice(&(size as u32).to_be_bytes());
                    e[12..16].copy_from_slice(((((size >> 32) as u32) << 16) | ((off >> 32) as u32 & 0xFFFF)).to_be_bytes().as_slice());
                }
                None => {
                    e[..4].copy_from_slice(&name.to_be_bytes());
                    e[4..8].copy_from_slice(&first[n].to_be_bytes());
                    e[8..12].copy_from_slice(&(node.children.len() as u32).to_be_bytes());
                }
            }
            emit(&mut w, &mut sha, &e, &mut written)?;
        }
        let tree = (tree_at, written - tree_at);
        let meta = (written, 0u64);
        // The footer, hashed with its hash zeroed.
        let footer = |hash: &[u8; 32], total_size: u64| -> [u8; FOOTER_BYTES] {
            let mut f = [0u8; FOOTER_BYTES];
            for (i, (o, s)) in [data, rec, names, tree, meta, meta].iter().enumerate() {
                f[i * 16..i * 16 + 8].copy_from_slice(&o.to_be_bytes());
                f[i * 16 + 8..i * 16 + 16].copy_from_slice(&s.to_be_bytes());
            }
            f[96..128].copy_from_slice(hash);
            f[128..136].copy_from_slice(&total_size.to_be_bytes());
            f[136..140].copy_from_slice(&VERSION1.to_be_bytes());
            f[140..144].copy_from_slice(&MAGIC.to_be_bytes());
            f
        };
        let total_size = written + FOOTER_BYTES as u64;
        sha.update(&footer(&[0; 32], total_size))?;
        let hash = sha.finish()?;
        w.write_all(&footer(&hash, total_size))?;
        let f = w.into_inner().map_err(|e| e.into_error())?;
        f.sync_all()?;
        Ok(Written { files: files.len() as u64, bytes_in: done, bytes_out: total_size })
    })();
    match r {
        Ok(stats) => match std::fs::rename(&partial, out) {
            Ok(()) => Ok(stats),
            Err(e) => {
                let _ = std::fs::remove_file(&partial);
                Err(e)
            }
        },
        Err(e) => {
            let _ = std::fs::remove_file(&partial);
            Err(e)
        }
    }
}

/// The archive's stored integrity hash and the one its bytes give (equal for a sound archive;
/// reads the whole file).
pub fn check_hash(path: &Path) -> io::Result<([u8; 32], [u8; 32])> {
    let mut f = File::open(path)?;
    let size = f.metadata()?.len();
    if size <= FOOTER_BYTES as u64 {
        return Err(bad("too short"));
    }
    let mut sha = crate::sha::Sha256::new()?;
    let mut left = size - FOOTER_BYTES as u64;
    let mut buf = vec![0u8; 8 << 20];
    while left > 0 {
        let want = left.min(buf.len() as u64) as usize;
        let n = f.read(&mut buf[..want])?;
        if n == 0 {
            return Err(bad("it ends early"));
        }
        sha.update(&buf[..n])?;
        left -= n as u64;
    }
    let mut ft = [0u8; FOOTER_BYTES];
    f.read_exact(&mut ft)?;
    let mut stored = [0u8; 32];
    stored.copy_from_slice(&ft[96..128]);
    ft[96..128].fill(0);
    sha.update(&ft)?;
    Ok((stored, sha.finish()?))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn tmp(name: &str) -> PathBuf {
        let d = std::env::temp_dir().join(format!("dsconfig-zar-{name}-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&d);
        std::fs::create_dir_all(&d).unwrap();
        d
    }

    /// A game-like folder: nested folders, files of sizes about the block's, compressible and not.
    fn make_tree(root: &Path) -> Vec<(String, Vec<u8>)> {
        let mut files = Vec::new();
        let mut seed = 0x1234_5678u32;
        let mut noise = |n: usize| -> Vec<u8> {
            (0..n)
                .map(|_| {
                    seed ^= seed << 13;
                    seed ^= seed >> 17;
                    seed ^= seed << 5;
                    seed as u8
                })
                .collect()
        };
        let specs: Vec<(&str, Vec<u8>)> = vec![
            ("sce_sys/param.json", br#"{"titleId": "PPSA01341"}"#.to_vec()),
            ("eboot.bin", noise(70_000)),
            ("Empty.txt", Vec::new()),
            ("misc/automation/Shots.txt", b"wait 1.0\n".repeat(20_000)),
            ("misc/b.bin", noise(BLOCK)),
            ("misc/a.bin", noise(BLOCK - 1)),
            ("misc/C.bin", noise(BLOCK + 1)),
            ("sound/streams/music.at9", noise(3 * BLOCK + 123)),
            ("sound/zeros.bin", vec![0u8; 5 * BLOCK + 7]),
        ];
        for (p, d) in specs {
            let full = parts(p).fold(root.to_path_buf(), |a, c| a.join(c));
            std::fs::create_dir_all(full.parent().unwrap()).unwrap();
            std::fs::write(&full, &d).unwrap();
            files.push((p.to_string(), d));
        }
        std::fs::create_dir_all(root.join("emptydir")).unwrap();
        files
    }

    #[test]
    fn round_trip() {
        let d = tmp("rt");
        let src = d.join("game");
        let files = make_tree(&src);
        let out = d.join("game.zar");
        let w = write_folder(&src, &out, 6, 4, &|_, _| true).unwrap();
        assert_eq!(w.files, files.len() as u64);
        assert!(w.bytes_out < w.bytes_in, "{w:?}");
        let (stored, computed) = check_hash(&out).unwrap();
        assert_eq!(stored, computed);
        let a = Archive::open(&out).unwrap();
        for (p, data) in &files {
            let n = a.lookup(p).unwrap_or_else(|| panic!("{p} missing"));
            assert!(a.is_file(n));
            assert_eq!(a.file_size(n), data.len() as u64);
            assert_eq!(&a.read_all(n).unwrap(), data, "{p}");
            // Case folded, other separators.
            assert_eq!(a.lookup(&p.to_ascii_uppercase().replace('/', "\\")), Some(n));
            // Reads at offsets across blocks.
            for (off, len) in [(0usize, 1usize), (1, 70_000), (BLOCK - 3, 10), (BLOCK, BLOCK), (data.len().saturating_sub(5), 100)] {
                let mut buf = vec![0u8; len];
                let got = a.read_at(n, off as u64, &mut buf).unwrap();
                let want = &data[off.min(data.len())..(off + len).min(data.len())];
                assert_eq!(&buf[..got], want, "{p} at {off}+{len}");
            }
        }
        // Directories, their order and their entries.
        let misc = a.lookup("MISC").unwrap();
        assert!(a.is_dir(misc));
        let names: Vec<String> = a.list(misc).into_iter().map(|e| e.name).collect();
        assert_eq!(names, ["a.bin", "automation", "b.bin", "C.bin"]);
        assert!(a.lookup("emptydir").is_some_and(|n| a.is_dir(n) && a.list(n).is_empty()));
        assert_eq!(a.lookup("misc/nothing"), None);
        assert_eq!(a.files_under(a.lookup("sound").unwrap()), ["streams/music.at9", "zeros.bin"]);
        let _ = std::fs::remove_dir_all(&d);
    }

    #[test]
    fn sources_agree() {
        let d = tmp("src");
        let src = d.join("dump").join("PPSA01341-app0");
        let files = make_tree(&src);
        let out = d.join("dump.zar");
        write_folder(&d.join("dump"), &out, 3, 2, &|_, _| true).unwrap();
        let dir = Source::open(&src).unwrap();
        let zar = Source::open(&out).unwrap();
        assert!(zar.is_archive() && !dir.is_archive());
        // (The game in a folder of the archive: found by its param.json.)
        assert!(matches!(&zar, Source::Zar { base, .. } if base == "PPSA01341-app0"));
        assert_eq!(dir.list("misc").unwrap().iter().map(|e| &e.name).collect::<Vec<_>>(),
                   zar.list("misc").unwrap().iter().map(|e| &e.name).collect::<Vec<_>>());
        for s in [&dir, &zar] {
            for (p, data) in &files {
                assert!(s.is_file(p), "{p}");
                assert_eq!(s.len(p).unwrap(), data.len() as u64);
                assert_eq!(&s.read(p).unwrap(), data);
                let f = s.open_file(p).unwrap();
                let mut b = vec![0u8; 9];
                let n = f.read_at(2, &mut b).unwrap();
                assert_eq!(&b[..n], &data[2.min(data.len())..11.min(data.len())]);
            }
            assert!(s.is_dir("misc/automation") && !s.is_file("misc"));
            let mut l: Vec<String> = s.list("misc").unwrap().into_iter().map(|e| e.name.to_ascii_lowercase()).collect();
            l.sort();
            assert_eq!(l, ["a.bin", "automation", "b.bin", "c.bin"]);
            assert_eq!(s.files_under("misc").len(), 4);
        }
        let _ = std::fs::remove_dir_all(&d);
    }

    #[test]
    fn refuses_damage() {
        let d = tmp("bad");
        let src = d.join("g");
        make_tree(&src);
        let out = d.join("g.zar");
        write_folder(&src, &out, 1, 1, &|_, _| true).unwrap();
        let mut b = std::fs::read(&out).unwrap();
        // Cut short: the footer's size no longer matches.
        std::fs::write(d.join("cut.zar"), &b[..b.len() - 1]).unwrap();
        assert!(Archive::open(&d.join("cut.zar")).is_err());
        // Another magic.
        let n = b.len();
        b[n - 1] ^= 0xFF;
        std::fs::write(d.join("magic.zar"), &b).unwrap();
        assert!(Archive::open(&d.join("magic.zar")).is_err());
        // Stopped by its progress: no archive, no partial file.
        let r = write_folder(&src, &d.join("stopped.zar"), 1, 1, &|_, _| false);
        assert!(r.is_err() && !d.join("stopped.zar").exists() && !d.join("stopped.zar.partial").exists());
        let _ = std::fs::remove_dir_all(&d);
    }

    #[test]
    fn refuses_bad_entries() {
        // A file entry whose size reaches past the data: refused at open, not read.
        let d = tmp("entries");
        let src = d.join("g");
        make_tree(&src);
        let out = d.join("g.zar");
        write_folder(&src, &out, 1, 1, &|_, _| true).unwrap();
        let mut b = std::fs::read(&out).unwrap();
        let f = b.len() - FOOTER_BYTES;
        let tree_at = be64(&b[f + 48..]) as usize;
        let mut i = tree_at + ENTRY_BYTES;
        while b[i] & 0x80 == 0 {
            i += ENTRY_BYTES;
        }
        // (The size's top 16 bits: the last word's upper half.)
        b[i + 12] = 0xFF;
        std::fs::write(d.join("big.zar"), &b).unwrap();
        let e = Archive::open(&d.join("big.zar")).err().expect("a file past the data must be refused");
        assert!(e.to_string().contains("past its data"), "{e}");
        // A folder named like an archive is a folder.
        std::fs::create_dir_all(d.join("dir.zar")).unwrap();
        assert!(!is_archive_path(&d.join("dir.zar")) && is_archive_path(&out));
        let _ = std::fs::remove_dir_all(&d);
    }

    #[test]
    fn listing_order() {
        // An archive's folders list as NTFS lists them (letters compared in upper case), so '_'
        // comes after the letters.
        let d = tmp("order");
        let src = d.join("g");
        std::fs::create_dir_all(&src).unwrap();
        for n in ["_cmn.txt", "b.txt", "A.txt", "zz.txt"] {
            std::fs::write(src.join(n), b"x").unwrap();
        }
        let out = d.join("g.zar");
        write_folder(&src, &out, 1, 1, &|_, _| true).unwrap();
        let z = Source::open(&out).unwrap();
        let names: Vec<String> = z.list("").unwrap().into_iter().map(|e| e.name).collect();
        assert_eq!(names, ["A.txt", "b.txt", "zz.txt", "_cmn.txt"]);
        let _ = std::fs::remove_dir_all(&d);
    }

    #[test]
    fn big_reads() {
        // Runs of whole blocks (one read of their stored bytes), shared out on threads or not,
        // and reads that start and end inside blocks around them.
        let d = tmp("big");
        let src = d.join("g");
        std::fs::create_dir_all(&src).unwrap();
        let mut seed = 0x9E37_79B9u32;
        let data: Vec<u8> = (0..300 * BLOCK + 777)
            .map(|i| {
                seed ^= seed << 13;
                seed ^= seed >> 17;
                seed ^= seed << 5;
                // (Half the blocks compressible, half not.)
                if (i / BLOCK) % 2 == 0 { (i % 7) as u8 } else { seed as u8 }
            })
            .collect();
        std::fs::write(src.join("big.bin"), &data).unwrap();
        let out = d.join("g.zar");
        write_folder(&src, &out, 3, 4, &|_, _| true).unwrap();
        let a = Archive::open(&out).unwrap();
        let n = a.lookup("big.bin").unwrap();
        for threads in [1, 4] {
            set_threads(threads);
            for (off, len) in [(0usize, data.len()), (BLOCK, 100 * BLOCK), (5, 290 * BLOCK), (BLOCK * 3 - 1, 40 * BLOCK + 2)] {
                let mut buf = vec![0u8; len];
                let got = a.read_at(n, off as u64, &mut buf).unwrap();
                assert_eq!(got, len.min(data.len() - off));
                assert!(buf[..got] == data[off..off + got], "{threads} threads at {off}+{len}");
            }
        }
        set_threads(1);
        let _ = std::fs::remove_dir_all(&d);
    }

    #[test]
    fn format_order() {
        assert_eq!(name_order("abc", "ABD"), std::cmp::Ordering::Less);
        assert_eq!(name_order("ab", "abc"), std::cmp::Ordering::Less);
        assert_eq!(name_order("B.bin", "a.bin"), std::cmp::Ordering::Greater);
        assert_eq!(name_order("_x", "a"), std::cmp::Ordering::Less);
    }
}
