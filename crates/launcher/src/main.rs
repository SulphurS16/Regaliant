//! regaliant-launcher: sets up and starts Regaliant, the Demon's Souls PC port, for players.
//!
//! - Game: the game folder, checked when chosen (title, version 01.007.000, the decrypted
//!   executable and libraries, the data folders); what this PC brings; safe mode for one start.
//! - Shaders: the game's shaders prepared ahead, with progress (the first Play prepares them
//!   unless that is turned off).
//! - Saves: the save folder, the backups the game makes and their restore, the PS5 import folder.
//! - Play / Reset settings / Open logs / Join the Discord (the community's invite in the
//!   browser). When the game stops with an error, its last error lines and its log folder are
//!   shown.
//!
//! The settings are the files the game reads (`dsconfig::paths`); the game's own settings are
//! in its menus (SETTINGS > PC).

#![cfg_attr(windows, windows_subsystem = "windows")]

#[cfg(windows)]
include!("windows_main.rs");
#[cfg(target_os = "linux")]
include!("linux_main.rs");
#[cfg(not(any(windows, target_os = "linux")))]
compile_error!("Regaliant supports Windows and Linux x86-64");
