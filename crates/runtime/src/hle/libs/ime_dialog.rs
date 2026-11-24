//! System IME dialog (libSceImeDialog): text entry such as the character name. The console's
//! on-screen keyboard is replaced by a text box the port draws in the game's own window
//! (`crate::textbox`, put on each frame by the presenter): the window gives its keys to the box
//! while it is open (`char_typed`, `key_down`), Enter confirms, Escape cancels, Backspace erases,
//! Ctrl+V pastes. With a controller the box shows a keyboard of its own (`pad_input`; from the
//! first button or direction, or from the start when the controller was in use): the d-pad or
//! the left stick moves over the keys, A types one, X erases, Y types a space, LB is Shift, RB
//! the accented letters and symbols, Menu confirms, B cancels. The game sees no input
//! meanwhile, as on the console while its keyboard shows. (Until 2026-11-13 a separate Windows
//! text box took the foreground: the game left exclusive fullscreen for the desktop.) Automated
//! runs supply the text with `DS_IME_TEXT` or the `ime <text>` control command. The entered text
//! is written to the title's UTF-16 buffer when it next polls the dialog status.

use crate::hle::Registry;
use crate::{reg, sysv};
use parking_lot::Mutex;
use std::sync::atomic::{AtomicU32, Ordering::Relaxed};
use std::time::Instant;

const STATUS_NONE: i32 = 0;
const STATUS_RUNNING: i32 = 1;
const STATUS_FINISHED: i32 = 2;
const END_OK: i32 = 0;
const END_CANCELED: i32 = 1;
const END_ABORTED: i32 = 2;

struct Dialog {
    /// Guest char16 buffer receiving the text (SceImeDialogParam.inputTextBuffer).
    buf: u64,
    max_len: u32,
    status: i32,
    end: i32,
    text: Option<Vec<u16>>,
    delivered: bool,
    /// The text being typed, the title and placeholder the box shows, when it opened.
    edit: Vec<u16>,
    title: String,
    placeholder: String,
    opened: Instant,
    /// The controller's keyboard shows (until a key is typed on the keyboard).
    pad_mode: bool,
    /// Its selected key (row, column), Shift (0 off, 1 the next letter, 2 locked), the page
    /// (0 letters, 1 accents and symbols), a held direction or erase and when it repeats.
    sel: (usize, usize),
    shift: u8,
    page: u8,
    repeat: Option<(u32, Instant)>,
    /// The keys (scan codes) pressed since the box opened: a key held from before (the one that
    /// opened it) does not type its repeats.
    fresh: Vec<u16>,
}

static DIALOG: Mutex<Option<Dialog>> = Mutex::new(None);
/// Text supplied through the control channel (`ime <text>`).
pub static CONTROL_TEXT: Mutex<Option<String>> = Mutex::new(None);
/// Edits since start (the box is drawn again when they or the caret change).
static EDITS: AtomicU32 = AtomicU32::new(0);

unsafe fn read_u16z(p: u64, max: usize) -> Vec<u16> {
    let mut v = Vec::new();
    if p == 0 {
        return v;
    }
    for i in 0..max {
        let c = unsafe { *((p + 2 * i as u64) as *const u16) };
        if c == 0 {
            break;
        }
        v.push(c);
    }
    v
}

fn finish(d: &mut Dialog, text: Option<Vec<u16>>, end: i32) {
    if let Some(mut t) = text {
        t.truncate(d.max_len as usize);
        d.text = Some(t);
    }
    d.end = end;
    d.status = STATUS_FINISHED;
    EDITS.fetch_add(1, Relaxed);
}

/// The box is open (the window's keys and the pad are its).
pub fn active() -> bool {
    DIALOG.lock().as_ref().is_some_and(|d| d.status == STATUS_RUNNING)
}

/// A character typed in the game's window (WM_CHAR; `repeat`: its key held down, `scan`: its
/// key's scan code): true when the box took it.
pub fn char_typed(c: u16, repeat: bool, scan: u16) -> bool {
    let mut g = DIALOG.lock();
    let Some(d) = g.as_mut().filter(|d| d.status == STATUS_RUNNING) else { return false };
    if repeat && !d.fresh.contains(&scan) {
        return true;
    }
    d.pad_mode = false;
    if c >= 0x20 && c != 0x7F && (d.edit.len() as u32) < d.max_len.max(1) {
        d.edit.push(c);
    }
    EDITS.fetch_add(1, Relaxed);
    true
}

/// A key pressed in the game's window (WM_KEYDOWN; `repeat`: held down, `scan`: its scan code):
/// true when the box took it (every key while it is open: the game sees none).
pub fn key_down(vk: u8, repeat: bool, ctrl: bool, scan: u16) -> bool {
    let mut g = DIALOG.lock();
    let Some(d) = g.as_mut().filter(|d| d.status == STATUS_RUNNING) else { return false };
    if !d.fresh.contains(&scan) {
        if repeat {
            // Held since before the box opened (the key that opened it): not the box's.
            return true;
        }
        d.fresh.push(scan);
    }
    if d.pad_mode {
        d.pad_mode = false;
        EDITS.fetch_add(1, Relaxed);
    }
    match vk {
        // Backspace (Ctrl: the last word).
        0x08 => {
            if ctrl {
                while d.edit.last() == Some(&(b' ' as u16)) {
                    d.edit.pop();
                }
                while d.edit.last().is_some_and(|&c| c != b' ' as u16) {
                    d.edit.pop();
                }
            } else {
                d.edit.pop();
            }
            EDITS.fetch_add(1, Relaxed);
        }
        0x0D if !repeat => {
            let t = d.edit.clone();
            crate::info!("ime dialog: confirmed in the game's window");
            finish(d, Some(t), END_OK);
        }
        0x1B if !repeat => {
            crate::info!("ime dialog: canceled in the game's window");
            finish(d, None, END_CANCELED);
        }
        b'V' if ctrl => {
            if let Some(t) = clipboard_text() {
                for c in t.encode_utf16().take_while(|&c| c != b'\r' as u16 && c != b'\n' as u16) {
                    if c >= 0x20 && c != 0x7F && (d.edit.len() as u32) < d.max_len.max(1) {
                        d.edit.push(c);
                    }
                }
                EDITS.fetch_add(1, Relaxed);
            }
        }
        _ => {}
    }
    true
}

// ---------------------------------------------------------------------------
// The controller's keyboard

const KEY_ROWS: usize = 5;
const KEY_COLS: usize = 10;
/// The character rows of the two pages, ten keys each (Shift: the letters' capitals).
const PAGES: [[&str; 4]; 2] = [
    ["1234567890", "qwertyuiop", "asdfghjkl'", "zxcvbnm,.-"],
    ["!?&():;\"#@", "àáâäãåæçñß", "èéêëìíîïýÿ", "òóôöõøùúûü"],
];

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
enum Key {
    Char(char),
    Shift,
    Page,
    Space,
    Erase,
    Done,
}

/// The bottom row, two columns a key.
const BOTTOM: [Key; 5] = [Key::Shift, Key::Page, Key::Space, Key::Erase, Key::Done];

/// The pad's bits (`input::PAD_BUTTONS`) the keyboard uses.
const A: u32 = 1 << 0;
const B: u32 = 1 << 1;
const X: u32 = 1 << 2;
const Y: u32 = 1 << 3;
const LB: u32 = 1 << 4;
const RB: u32 = 1 << 5;
const LT: u32 = 1 << 6;
const MENU: u32 = 1 << 11;
/// The d-pad (up, down, left, right: bits 12-15); the left stick past half its travel too.
const DIRS: u32 = 0xF << 12;
/// A held direction or erase repeats after 400 ms, then every 90 ms.
const REPEAT_FIRST: std::time::Duration = std::time::Duration::from_millis(400);
const REPEAT_NEXT: std::time::Duration = std::time::Duration::from_millis(90);

fn upper(c: char) -> char {
    let mut u = c.to_uppercase();
    match (u.next(), u.next()) {
        (Some(x), None) => x,
        // (ß: "SS".)
        _ => c,
    }
}

fn key_at(d: &Dialog, row: usize, col: usize) -> Key {
    if row < KEY_ROWS - 1 {
        let c = PAGES[d.page as usize & 1][row].chars().nth(col).unwrap_or(' ');
        Key::Char(if d.shift != 0 { upper(c) } else { c })
    } else {
        BOTTOM[(col / 2).min(BOTTOM.len() - 1)]
    }
}

/// One step over the keys (0 up, 1 down, 2 left, 3 right), wrapping around; the bottom row
/// moves a key (two columns) at a time and keeps the column it was entered at.
fn step(d: &mut Dialog, dir: u32) {
    let (r, c) = d.sel;
    let keys = BOTTOM.len();
    d.sel = match dir {
        0 => ((r + KEY_ROWS - 1) % KEY_ROWS, c),
        1 => ((r + 1) % KEY_ROWS, c),
        2 if r == KEY_ROWS - 1 => (r, (c / 2 + keys - 1) % keys * 2),
        3 if r == KEY_ROWS - 1 => (r, (c / 2 + 1) % keys * 2),
        2 => (r, (c + KEY_COLS - 1) % KEY_COLS),
        _ => (r, (c + 1) % KEY_COLS),
    };
}

fn type_char(d: &mut Dialog, c: char) {
    let mut units = [0u16; 2];
    let units = c.encode_utf16(&mut units);
    if d.edit.len() + units.len() <= d.max_len.max(1) as usize {
        d.edit.extend_from_slice(units);
    }
}

fn erase(d: &mut Dialog) {
    d.edit.pop();
    // (An empty name starts with a capital again.)
    if d.edit.is_empty() && d.shift == 0 {
        d.shift = 1;
    }
}

/// A key typed on the controller's keyboard: true for Done.
fn press(d: &mut Dialog, key: Key) -> bool {
    match key {
        Key::Char(c) => {
            type_char(d, c);
            if d.shift == 1 && c.is_alphabetic() {
                d.shift = 0;
            }
        }
        Key::Shift => d.shift = (d.shift + 1) % 3,
        Key::Page => d.page ^= 1,
        Key::Space => type_char(d, ' '),
        Key::Erase => erase(d),
        Key::Done => return true,
    }
    false
}

/// The pad's buttons held (bit i: `input::PAD_BUTTONS[i]`) and its left stick (XInput's range,
/// y up), each poll: true while the box is open (the pad is its alone). A confirm or cancel
/// keeps its button from the game until it is released.
pub fn pad_input(phys: u32, lx: i16, ly: i16) -> bool {
    static LAST: AtomicU32 = AtomicU32::new(0);
    let mut held = phys;
    const HALF: i16 = 16384;
    if ly > HALF {
        held |= 1 << 12;
    }
    if ly < -HALF {
        held |= 1 << 13;
    }
    if lx < -HALF {
        held |= 1 << 14;
    }
    if lx > HALF {
        held |= 1 << 15;
    }
    let was = LAST.swap(held, Relaxed);
    let mut g = DIALOG.lock();
    let Some(d) = g.as_mut().filter(|d| d.status == STATUS_RUNNING) else { return false };
    let pressed = held & !was;
    if pressed == 0 && d.repeat.is_none() {
        return true;
    }
    let before = (d.edit.len(), d.sel, d.shift, d.page, d.pad_mode);
    let done = |d: &mut Dialog, bit: u32, end: i32| {
        if end == END_OK {
            let t = d.edit.clone();
            crate::info!("ime dialog: confirmed with the controller");
            finish(d, Some(t), END_OK);
        } else {
            crate::info!("ime dialog: canceled with the controller");
            finish(d, None, END_CANCELED);
        }
        crate::input::swallow_pad(bit);
    };
    if pressed & MENU != 0 {
        done(d, MENU, END_OK);
        return true;
    }
    if pressed & B != 0 {
        done(d, B, END_CANCELED);
        return true;
    }
    if !d.pad_mode {
        // The first press shows the keyboard and does nothing else.
        if pressed != 0 {
            d.pad_mode = true;
            d.repeat = None;
            EDITS.fetch_add(1, Relaxed);
        }
        return true;
    }
    let now = Instant::now();
    if pressed & A != 0 {
        let key = key_at(d, d.sel.0, d.sel.1);
        if press(d, key) {
            done(d, A, END_OK);
            return true;
        }
    }
    if pressed & Y != 0 {
        type_char(d, ' ');
    }
    if pressed & (LB | LT) != 0 {
        d.shift = (d.shift + 1) % 3;
    }
    if pressed & RB != 0 {
        d.page ^= 1;
    }
    // A direction or erase (the one just pressed first): at once, then repeated while held.
    let pick = |bits: u32| {
        if bits & DIRS != 0 {
            (bits >> 12 & 0xF).trailing_zeros()
        } else if bits & X != 0 {
            4
        } else {
            u32::MAX
        }
    };
    let new = pressed & (DIRS | X);
    let what = if new != 0 { pick(new) } else { pick(held & (DIRS | X)) };
    if what == u32::MAX {
        d.repeat = None;
    } else {
        let act = |d: &mut Dialog| if what == 4 { erase(d) } else { step(d, what) };
        match d.repeat {
            Some((w, next)) if w == what && new == 0 => {
                if now >= next {
                    act(d);
                    d.repeat = Some((what, now + REPEAT_NEXT));
                }
            }
            _ => {
                act(d);
                d.repeat = Some((what, now + REPEAT_FIRST));
            }
        }
    }
    if (d.edit.len(), d.sel, d.shift, d.page, d.pad_mode) != before {
        EDITS.fetch_add(1, Relaxed);
    }
    true
}

/// The keyboard as the box draws it.
fn keys_view(d: &Dialog) -> crate::textbox::Keys {
    let mut rows: Vec<Vec<(String, u8, bool)>> = (0..KEY_ROWS - 1)
        .map(|r| {
            (0..KEY_COLS)
                .map(|c| match key_at(d, r, c) {
                    Key::Char(ch) => (ch.to_string(), 1, false),
                    _ => (String::new(), 1, false),
                })
                .collect()
        })
        .collect();
    rows.push(
        BOTTOM
            .iter()
            .map(|k| match k {
                Key::Shift => ((if d.shift == 2 { "Caps" } else { "Shift" }).to_string(), 2, d.shift != 0),
                Key::Page => ((if d.page == 0 { "àé#" } else { "abc" }).to_string(), 2, false),
                Key::Space => ("Space".to_string(), 2, false),
                Key::Erase => ("Delete".to_string(), 2, false),
                Key::Done => ("Done".to_string(), 2, false),
                Key::Char(_) => (String::new(), 2, false),
            })
            .collect(),
    );
    let (r, c) = d.sel;
    crate::textbox::Keys { rows, selected: (r, if r == KEY_ROWS - 1 { c / 2 } else { c }) }
}

/// The controller keyboard's buttons, named as the prompts show them.
fn pad_hint() -> String {
    if crate::pcsettings::prompts::shown_set() == 1 {
        "Cross Type    Square Delete    Triangle Space    L1 Shift\nOptions Done    Circle Cancel".into()
    } else {
        "(A) Type    (X) Delete    (Y) Space    (LB) Shift\n(Menu) Done    (B) Cancel".into()
    }
}

/// What the box shows now and a generation that changes with it (None: no box).
pub fn overlay() -> Option<(u64, crate::textbox::Content)> {
    let g = DIALOG.lock();
    let d = g.as_ref().filter(|d| d.status == STATUS_RUNNING)?;
    // The caret blinks every 530 ms (Windows' default).
    let phase = (d.opened.elapsed().as_millis() / 530) as u64;
    let content = crate::textbox::Content {
        title: if d.title.is_empty() { "Enter Text".into() } else { d.title.clone() },
        text: String::from_utf16_lossy(&d.edit),
        placeholder: d.placeholder.clone(),
        max_len: d.max_len,
        caret: phase % 2 == 0,
        hint: if d.pad_mode { pad_hint() } else { "Enter  Confirm      Esc  Cancel".into() },
        keys: d.pad_mode.then(|| keys_view(d)),
    };
    // (The prompts' set is in the generation: the hint names its buttons.)
    let set = if d.pad_mode { crate::pcsettings::prompts::shown_set() as u64 } else { 0 };
    let generation = (EDITS.load(Relaxed) as u64) << 35 | set << 33 | (phase & 1) << 32;
    Some((generation, content))
}

#[cfg(windows)]
fn clipboard_text() -> Option<String> {
    use windows_sys::Win32::System::DataExchange::{CloseClipboard, GetClipboardData, OpenClipboard};
    use windows_sys::Win32::System::Memory::{GlobalLock, GlobalUnlock};
    const CF_UNICODETEXT: u32 = 13;
    unsafe {
        if OpenClipboard(std::ptr::null_mut()) == 0 {
            return None;
        }
        let h = GetClipboardData(CF_UNICODETEXT);
        let mut out = None;
        if !h.is_null() {
            let p = GlobalLock(h as _) as *const u16;
            if !p.is_null() {
                let mut n = 0;
                while n < 4096 && *p.add(n) != 0 {
                    n += 1;
                }
                out = Some(String::from_utf16_lossy(std::slice::from_raw_parts(p, n)));
                GlobalUnlock(h as _);
            }
        }
        CloseClipboard();
        out
    }
}

sysv! {
    fn sceImeDialogInit(param: *const u8, extended: *const u8) -> i32 {
        if param.is_null() {
            return crate::hle::kernel::sce(22); // EINVAL
        }
        let max_len = *(param.add(0x24) as *const u32);
        let buf = *(param.add(0x28) as *const u64);
        let placeholder = *(param.add(0x40) as *const u64);
        let title = *(param.add(0x48) as *const u64);
        let s = |v: Vec<u16>| String::from_utf16_lossy(&v);
        let title_s = s(read_u16z(title, 256));
        let placeholder_s = s(read_u16z(placeholder, 256));
        let initial = read_u16z(buf, max_len.min(1024) as usize);
        crate::info!("ime dialog: title '{}', placeholder '{}', initial '{}', max {} chars", title_s, placeholder_s, s(initial.clone()), max_len);
        // The controller's keyboard from the start when the controller was in use.
        let pad_mode = crate::input::last_device() == crate::input::DEVICE_PAD;
        let shift = initial.is_empty() as u8;
        let mut d = Dialog {
            buf,
            max_len,
            status: STATUS_RUNNING,
            end: END_CANCELED,
            text: None,
            delivered: false,
            edit: initial,
            title: title_s,
            placeholder: placeholder_s,
            opened: Instant::now(),
            pad_mode,
            sel: (1, 0),
            shift,
            page: 0,
            repeat: None,
            fresh: Vec::new(),
        };
        if let Ok(t) = std::env::var("DS_IME_TEXT") {
            crate::info!("ime dialog: text supplied by DS_IME_TEXT");
            finish(&mut d, Some(t.encode_utf16().collect()), END_OK);
        } else {
            crate::info!("ime dialog: a text box in the game's window (Enter confirms, Esc cancels; the controller's keyboard {})", if pad_mode { "shown" } else { "on its first press" });
            EDITS.fetch_add(1, Relaxed);
        }
        *DIALOG.lock() = Some(d);
        0
    }
    fn sceImeDialogGetStatus() -> i32 {
        let mut g = DIALOG.lock();
        let Some(d) = g.as_mut() else { return STATUS_NONE };
        if d.status == STATUS_RUNNING {
            if let Some(t) = CONTROL_TEXT.lock().take() {
                crate::info!("ime dialog: text supplied by control command");
                finish(d, Some(t.encode_utf16().collect()), END_OK);
            }
        }
        if d.status == STATUS_FINISHED && !d.delivered {
            d.delivered = true;
            if let (Some(t), true) = (&d.text, d.end == END_OK && d.buf != 0) {
                for (i, c) in t.iter().chain(std::iter::once(&0)).enumerate() {
                    *((d.buf + 2 * i as u64) as *mut u16) = *c;
                }
                crate::info!("ime dialog: entered '{}'", String::from_utf16_lossy(t));
            }
        }
        d.status
    }
    fn sceImeDialogGetResult(result: *mut i32) -> i32 {
        let g = DIALOG.lock();
        let end = g.as_ref().map(|d| d.end).unwrap_or(END_CANCELED);
        if !result.is_null() {
            std::ptr::write_bytes(result as *mut u8, 0, 16);
            *result = end;
        }
        0
    }
    fn sceImeDialogTerm() -> i32 {
        DIALOG.lock().take();
        EDITS.fetch_add(1, Relaxed);
        0
    }
    fn sceImeDialogAbort() -> i32 {
        if let Some(d) = DIALOG.lock().as_mut() {
            finish(d, None, END_ABORTED);
        }
        0
    }
}

pub fn register(r: &mut Registry) {
    reg!(r, "libSceImeDialog", [sceImeDialogInit, sceImeDialogGetStatus, sceImeDialogGetResult, sceImeDialogTerm, sceImeDialogAbort]);
}

#[cfg(test)]
mod tests {
    use super::*;

    fn open(pad_mode: bool) {
        *DIALOG.lock() = Some(Dialog {
            buf: 0,
            max_len: 6,
            status: STATUS_RUNNING,
            end: END_CANCELED,
            text: None,
            delivered: false,
            edit: Vec::new(),
            title: String::new(),
            placeholder: String::new(),
            opened: Instant::now(),
            pad_mode,
            sel: (1, 0),
            shift: 1,
            page: 0,
            repeat: None,
            fresh: Vec::new(),
        });
    }

    /// A press and its release (the release after a confirm or cancel finds the box closed).
    fn tap(bit: u32) {
        assert!(pad_input(bit, 0, 0));
        pad_input(0, 0, 0);
    }

    fn text() -> String {
        DIALOG.lock().as_ref().map(|d| String::from_utf16_lossy(d.text.as_ref().unwrap_or(&d.edit))).unwrap()
    }

    const UP: u32 = 1 << 12;
    const DOWN: u32 = 1 << 13;
    const RIGHT: u32 = 1 << 15;

    // (One test: the dialog and the pad's last buttons are process-wide.)
    #[test]
    fn controller_keyboard() {
        // Not shown yet: the first press only shows it.
        open(false);
        tap(DOWN);
        assert!(DIALOG.lock().as_ref().unwrap().pad_mode);
        assert_eq!(DIALOG.lock().as_ref().unwrap().sel, (1, 0));
        // "S" (row 2, column 1; Shift on for the first letter), "o", "l" (Shift off again).
        tap(DOWN);
        tap(RIGHT);
        tap(A);
        for _ in 0..7 {
            tap(RIGHT);
        }
        tap(UP);
        tap(A);
        tap(DOWN);
        tap(A);
        assert_eq!(text(), "Sol");
        // X erases, Y types a space; the stick moves too (left past half its travel: "k").
        tap(X);
        tap(Y);
        assert!(pad_input(0, -30000, 0));
        assert!(pad_input(0, 0, 0));
        tap(A);
        assert_eq!(text(), "So k");
        // Six at most: "So kkk" then nothing more.
        for _ in 0..4 {
            tap(A);
        }
        assert_eq!(text(), "So kkk");
        // The bottom row moves a key at a time: from column 7 down to Delete (7 / 2), right to
        // Done, which confirms.
        for _ in 0..2 {
            tap(DOWN);
        }
        assert_eq!(DIALOG.lock().as_ref().unwrap().sel, (4, 7));
        tap(A);
        assert_eq!(text(), "So kk");
        tap(RIGHT);
        assert_eq!(DIALOG.lock().as_ref().unwrap().sel, (4, 8));
        tap(A);
        {
            let g = DIALOG.lock();
            let d = g.as_ref().unwrap();
            assert_eq!((d.status, d.end), (STATUS_FINISHED, END_OK));
            assert_eq!(String::from_utf16_lossy(d.text.as_ref().unwrap()), "So kk");
        }
        // Erasing everything brings the capital back; B cancels; Menu confirms.
        open(true);
        DIALOG.lock().as_mut().unwrap().shift = 0;
        tap(A);
        tap(X);
        assert_eq!(DIALOG.lock().as_ref().unwrap().shift, 1);
        tap(B);
        assert_eq!(DIALOG.lock().as_ref().unwrap().end, END_CANCELED);
        open(true);
        tap(A);
        tap(MENU);
        let end = DIALOG.lock().as_ref().unwrap().end;
        assert_eq!((end, text()), (END_OK, "Q".to_string()));
        *DIALOG.lock() = None;
    }
}

#[cfg(target_os = "linux")]
fn clipboard_text() -> Option<String> {
    unsafe {
        let p = sdl3::sys::clipboard::SDL_GetClipboardText();
        if p.is_null() { return None; }
        let text = std::ffi::CStr::from_ptr(p).to_string_lossy().into_owned();
        sdl3::sys::stdinc::SDL_free(p.cast());
        Some(text)
    }
}
