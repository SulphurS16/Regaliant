//! High-level emulation registry and helpers (PS5 system libraries).
//!
//! HLE functions are Rust `extern "sysv64"` functions called directly by guest
//! code. They are registered under (library name, NID). Imports that resolve to
//! nothing get a trap stub that reports the call; by default that is fatal so
//! missing behavior is never hidden.

pub mod iostat;
pub mod waitcensus;
pub mod kernel;
#[cfg_attr(target_os = "linux", path = "linux_libs.rs")]
pub mod libs;

mod support;
pub use support::*;

mod linker;
pub use linker::*;

fn register_platform(r: &mut Registry) {
    kernel::register(r);
    libs::register(r);
}
