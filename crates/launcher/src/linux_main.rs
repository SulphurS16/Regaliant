// The Windows launcher is not ported yet. Start the runtime directly on Linux.
fn main() {
    eprintln!("The graphical launcher is not available on Linux yet.\nRun dsport --app <game-folder-or-zar>; see docs/linux.md for build and play instructions.");
    std::process::exit(2);
}
