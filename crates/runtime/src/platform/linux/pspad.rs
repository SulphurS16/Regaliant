//! SDL already merges PlayStation controllers into the common controller samples.
pub struct Gamepad { pub buttons: u16, pub lt: u8, pub rt: u8, pub lx: i16, pub ly: i16, pub rx: i16, pub ry: i16 }
pub struct State { pub active: bool, pub pad: Gamepad, pub packet: u32 }
pub fn states() -> Vec<State> { Vec::new() }
pub fn set_motors(_left: u8, _right: u8) {} // Sent by the SDL XInput adapter once.
pub fn set_lightbar(rgb: Option<[u8; 3]>) { crate::platform::frontend::set_lightbar(rgb); }
