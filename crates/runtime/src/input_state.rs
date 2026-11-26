pub const BTN_L3: u32 = 0x2;
pub const BTN_R3: u32 = 0x4;
pub const BTN_OPTIONS: u32 = 0x8;
pub const BTN_UP: u32 = 0x10;
pub const BTN_RIGHT: u32 = 0x20;
pub const BTN_DOWN: u32 = 0x40;
pub const BTN_LEFT: u32 = 0x80;
pub const BTN_L2: u32 = 0x100;
pub const BTN_R2: u32 = 0x200;
pub const BTN_L1: u32 = 0x400;
pub const BTN_R1: u32 = 0x800;
pub const BTN_TRIANGLE: u32 = 0x1000;
pub const BTN_CIRCLE: u32 = 0x2000;
pub const BTN_CROSS: u32 = 0x4000;
pub const BTN_SQUARE: u32 = 0x8000;
pub const BTN_TOUCHPAD: u32 = 0x100000;

/// Pad state in DS4 terms (sticks 0..255, centre 0x80; y grows downwards).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct PadState {
    pub buttons: u32,
    pub lx: u8,
    pub ly: u8,
    pub rx: u8,
    pub ry: u8,
    pub l2: u8,
    pub r2: u8,
}

impl Default for PadState {
    fn default() -> Self {
        PadState { buttons: 0, lx: 0x80, ly: 0x80, rx: 0x80, ry: 0x80, l2: 0, r2: 0 }
    }
}

