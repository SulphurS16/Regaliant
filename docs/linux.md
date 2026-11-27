# Linux (experimental)

Regaliant runs the decrypted PS5 executable natively on Linux x86-64, with Vulkan graphics
and SDL3 windowing, input and audio. Gameplay, save resumption, fractional-scale fullscreen
and the full shader-preparation tour have been exercised on Linux. Performance and hardware
coverage remain limited; steady 60 fps is not guaranteed. Optional
[FSR3 frame generation](linux-frame-generation.md) is experimental and off by default.

## Build and play

Install Rust/Cargo, a C/C++ compiler, CMake, pkg-config, SDL3 development files and a Vulkan
loader and driver. Fontconfig and a system TrueType/OpenType font provide text entry and
keyboard labels. See [install.md](install.md) for the required decrypted game files.

From the checkout:

```sh
CARGO_HOME="$PWD/.cargo-home" cargo build --release -p dsport
target/release/dsport --app /path/to/PPSA01341-app0
```

The second command starts a resizable window with real audio. Alternatively, export
`DS_GAME_DIR` and run `target/release/dsport`. `--app` also accepts a `.zar` archive, read
without extraction into the game tree. Plain ELF files and SELF containers with decrypted
segments work in folders or archives; encrypted or compressed SELF segments are unsupported.
NVIDIA is preferred when several GPUs are available. SDL selects Wayland or X11;
`SDL_VIDEO_DRIVER=wayland` or `SDL_VIDEO_DRIVER=x11` selects one explicitly.

### Before the first start

- **Open files.** The game opens many files at once; with the common soft limit of 1024 it
  dies opening `.cani` files. Raise it in the shell that starts dsport: `ulimit -n 524288`
  (check with `ulimit -n`; some distributions already default to 524288). If the hard limit
  is lower, raise `nofile` in `/etc/security/limits.conf` or `DefaultLimitNOFILE` in systemd's
  `system.conf`.
- **SDL3 link.** The build links against `libSDL3.so`. Distributions that ship only
  `libSDL3.so.0` in the runtime package put the unversioned link in the development package
  (Ubuntu/Debian: `libsdl3-dev`; Arch's `sdl3` includes it). Install that rather than making
  the link by hand.
- **Memory overcommit.** One RADV system needed `sudo sysctl vm.overcommit_memory=1` (the
  NVIDIA test machine runs with the default, 0). If start-up fails with allocation or mapping
  errors, try it; what fails without it is not known yet.

On first start, select a language, adjust brightness, choose Continue Offline, and create a
character. Shaders compile as new scenes and effects appear, so initial visits can stutter.
Preparation can compile those shaders before playing; its duration depends on hardware and
cooling. No game files, keys or shader binaries are supplied by this repository.

## Controls, display and sound

Keyboard, mouse and SDL gamepads (including Xbox, DualSense and DualShock 4) use the game's
PC settings tab and `controls.txt`. PlayStation controllers select PlayStation button labels;
the touchpad click maps to the View/touchpad binding. Rumble and the light bar use SDL;
adaptive triggers and motion sensors are not mapped. Controller output still needs broader
physical-device testing. Basic character-name entry is supported.

F1 captures or releases the mouse. Controller input hides the pointer over the window;
mouse movement or a mouse button shows it again when capture is off. In fullscreen it also
hides after three seconds without mouse activity. Hiding does not grab the mouse.
Alt+Enter or F11 switches between windowed and borderless fullscreen. Closing the window
flushes completed saves and shader records before exiting. Quit through the game menu when
possible: closing during a save cannot guarantee that an unfinished save completes.

The swapchain uses physical pixels on scaled displays, and mouse coordinates follow the
UI's pixel size. On a 3840×2160 screen, select **PC → Render Scale → 100%** for native 4K.
**Visual Mode** instead uses the title's chosen render size. A 4K swapchain alone does not
mean the game renders at 4K. Shapes wider than 16:9 cap the internal render height at
**1512 lines**, including Cinematic and higher render-scale choices, to fit the title's
render-target pool. Output remains at the display size. The separate Resolution/Refresh Rate
controls do not change Linux desktop modes; choose the monitor mode in your compositor.

FIFO presentation is the default. `DS_LINUX_PRESENT_MODE=mailbox` selects MAILBOX where
supported; `immediate` is a tearing diagnostic override. Guest timing remains 59.94 Hz.
The present-wait observer runs separately from the game thread. Compositor VRR control and
exclusive display-mode switching are not implemented. Motion Blur Fix offers **Game's Blur**,
**Reduced Blur** and **No Blur**; these labels retain the previous saved values.

Audio uses SDL's normal output device, including PipeWire-backed devices. In-game sound has
been play-tested. Movie audio, especially during startup compilation, still needs broader
verification. `SDL_AUDIO_DRIVER=dummy` selects silent output for unattended diagnostics.

## Files and caches

Settings, controls, achievements and saves use `$XDG_DATA_HOME/Regaliant`
(default `~/.local/share/Regaliant`). Saves are in its `saves` directory.
Logs, shader stores and preparation state use `$XDG_CACHE_HOME/Regaliant`
(default `~/.cache/Regaliant`). `DSPORT_HOME=/absolute/path` puts both in that directory;
`DS_LINUX_SHADER_CACHE=/absolute/path` selects a separate renderer cache.

The game source is read-only. Output paths into it, including symlink aliases and targets,
are rejected. `DS_READONLY_ROOTS=/read-only/tree:/another/tree` protects additional trees.
Never put saves, caches or logs inside the game folder. The command-line archive tool
`cargo run -p dsconfig --example zartool --` works on Linux; archive creation rejects source
symlinks and protected output paths.

The runtime gives NVIDIA and Mesa private driver-cache directories under its cache root,
unless their environment overrides already exist. NVIDIA defaults to an 8 GiB limit and
`__GL_SHADER_DISK_CACHE_SKIP_CLEANUP=1`; Mesa uses `MESA_SHADER_CACHE_DIR` and
`MESA_SHADER_CACHE_MAX_SIZE`. Startup logs show the chosen paths and limits. These driver
caches supplement the persistent Vulkan pipeline-binary store. A driver, enabled-feature or
shader-translation change can require rebuilding incompatible pipelines.

`diagnostics.txt` in the settings folder accepts `DS_NAME=value` lines and `#` comments.
UTF-8 and UTF-16LE with a byte-order mark work. Environment values take precedence;
preparation also enforces its required switches. Startup simulation switches (`DS_SIM_*`)
are ignored in this file; deliberate developer tests can still set them in the environment.
Remove diagnostic overrides before comparing performance.

## Add-ons and PS5 save import

Put your own pre-order or Digital Deluxe packages in `Add-ons` under the settings folder.
Keep names such as `EP9000-PPSA01341_00-DEMONSSOULSGIFT1.pkg` or `...GIFT2.pkg`; unpacked
folders with those names also work. Restart to detect them. Upstream also searches `Add-ons`
under the cache and executable directories; avoid those locations when you need isolated
preparation settings.

To import a decrypted PS5 save, **copy** its folders into `PS5 Saves` under the settings
folder. Each `SAVEDATA...` or `sdimg_SAVEDATA...` folder must contain `USR-DATA`; an enclosing
`PPSA01341` folder is fine. Enable **Import PS5 Saves** in SETTINGS > PC, quit and restart.
Existing saves are backed up in `save-backups`; the import folder is renamed after success.
PS5 backup folders (`sdimg_sce_bu_...`) and symbolic links are skipped. The importer does not
decrypt saves. Keep original exports elsewhere and never import from inside the game root.
Windows Documents/Saved Games relocation choices do not change Linux's save location.

## Prepare shaders

```sh
python3 scripts/linux-prepare-shaders.py --app /path/to/PPSA01341-app0
```

This runs the full multi-level tour hidden and silent with isolated saves; it never replaces
player saves. The default limits are eight allowed logical CPUs (preferring performance cores),
nice 10, two compiler workers and a 30 fps cap. `DS_RELAY_CPUS` and `DS_COMPILE_WORKERS`
override the CPU limits. The thermal guard pauses after ten continuous seconds at or above
103 °C, resumes below 97 °C, and aborts after 60 continuous seconds at or above 105 °C.
Missing temperature sensors stop the command. Limits are overridable through the relay's
`DS_RELAY_TEMP_*` variables; use `--help` for command options.

Allow roughly an hour on a fast unrestricted PC; CPU limits and cooling can extend this to
several hours. Attempts stop after four hours by default; `--seconds 600` requests ten minutes.
Ctrl+C stops children and preserves progress. Repeat the command to resume. Exit 12 means
paused; exit 75 means the guard stopped it. Do not prepare while playing.

Preparation uses the same shader store and translation settings as normal play, including
`DS_PLAIN_MODES` and `DS_MAX_RECONV` from the environment or the player's `diagnostics.txt`.
Use the same environment for preparation and play. Graphics/driver/translation changes may
restart shader stages while retaining the isolated save. New effects and variants can still
compile during play. Frame generation is disabled in the tour; its SDK pipelines compile
separately on first activation. The direct runtime `--prepare-shaders N` option does not
provide the wrapper's CPU affinity or thermal guard.

Progress and checkpoints are in `prepare/linux-full` under the cache directory. Stage health
files, not exit 0 alone, confirm completion of `newgame`, `compile` and the three tour parts.
The full tour has completed with uffd, producing about 3.1 GB of shader-store data on the tested
configuration; that is not a universal size or visual-coverage guarantee. Monitor logs are
fsynced under `local-data/linux-port/shader-preparation/`.

Async userfaultfd write protection (`uffd`) is the Linux default. It requires kernel support
for async WP and `PAGEMAP_SCAN` (Linux 6.7+), with suitable permissions. Check for
`Linux write watch: uffd async WP` in the child log. Unsupported systems prominently warn
and fall back to mprotect. That backend had tour corruption; its race fixes have not passed
the same full-tour validation. `--write-watch mprotect` or `DS_LINUX_WRITE_WATCH=mprotect`
retains it for diagnosis. `--io-history` adds bounded file-read evidence without discarding
completed stages. See [maintainer notes](linux-notes.md) for memory and synchronization limits.

## Troubleshooting and developer verification

Keep the affected `linux-runs/...` directory under the cache root, particularly `dsport.log`,
`crash.log` and `fault-maps.txt` if present. Ctrl+F10 requests a Linux thread snapshot in a new
`threads-N` subdirectory: registered HLE waits, kernel wait states, sampled instruction
pointers, frame-pointer chains, raw stack words and mappings. F10 remains the capture burst.
Preparation's stall watchdog writes `hang-threads.txt` and `hang-maps.txt` before exiting.
Unavailable kernel stacks or blocked sampling signals are reported explicitly; raw stack words
are not a complete unwind. Review paths and raw memory in reports before posting them publicly.

```sh
CARGO_HOME="$PWD/.cargo-home" CARGO_BUILD_JOBS=2 cargo check --workspace
CARGO_HOME="$PWD/.cargo-home" CARGO_BUILD_JOBS=2 cargo build --release -p dsport
CARGO_HOME="$PWD/.cargo-home" CARGO_BUILD_JOBS=2 RUST_TEST_THREADS=2 cargo test --workspace
python3 -m unittest discover -s scripts -p 'test_linux_*.py'
```

A Windows cross-check needs MinGW and a Rust toolchain with `x86_64-pc-windows-gnu`.
`scripts/linux-windows-check.sh` uses isolated rustup in `.cargo-home/bin`, repository
`CARGO_HOME`, and `RUSTUP_HOME=$PWD/local-data/linux-port/toolchains/rustup`. Install that
rustup with `--no-modify-path --default-toolchain none`, then install stable and the target
with those environment variables. The check has a 600-second timeout; it does not test
Windows execution or MSVC linking.

For GPU relays, export `DS_GAME_DIR=/path/to/PPSA01341-app0`; there is no default game path.
Also set `DS_RELAY_SAVE_SEED=/path/to/save-seed` (a directory containing `saves/`) and optionally
`DS_RELAY_SHADER_SEED=/path/to/shaders`. Seeds are copied, never modified.
`scripts/linux-gpu-test.sh` and `scripts/linux-present-test.sh` both default to a short,
timeout-bounded, silent offscreen/visible suite. `--full` enables the long suite; `--dry-run`
prints its plan. Results and frame dumps stay under gitignored `local-data/linux-port/`.
Use `DS_READONLY_ROOTS` for any additional protected seed/source trees.

Specialized relays have `--help` and `--dry-run`: `linux-pipeline-reuse.py` verifies two-run
binary reuse, `linux-startup-test.py` repeats startup, `linux-steady.py` warms a fixed route
then measures it, and `linux-framegen-test.py` checks generated frames. They use the shared
`~/.cache/cpu-heavy.lock`. General relays retain temperature pauses; steady measurements
record temperature without SIGSTOP and abort on sustained 105 °C. Cold pipelines, captures,
validation, or thermal pauses invalidate performance comparisons. KHR presentation completion
is not a physical input-to-photon latency measurement. Watch moving scenes as well as PNGs.

## Unsupported or pending

- Windows graphical launcher and archive-creation UI; use the Linux CLI.
- DLSS, XeSS, Streamline and FSR **upscaling**, HDR, exclusive display modes and compositor VRR
  control. Native FSR3 **frame generation** is a separate experimental feature.
- WDDM cross-process VRAM/process accounting. Image eviction still uses Vulkan memory budgets.
- Windows Sonic, endpoint form-factor detection, Windows priority/power controls, cycle counters,
  hardware watchpoints and DLL-injection diagnostics. Linux has its own thread/mapping reports.
- Windows save relocation, custom themed cursor, full IME composition, bidirectional text and
  font fallback. Controller and movie-audio hardware coverage remains limited.
- Whole-run parallel recording (`DS_PAR_WHOLE_RUN=1`) is visually incorrect; keep it off.
  C4 held pages remain disabled on Linux, so eager FE2 write-back still costs copies and waits.

For fixed-flip brightness comparisons across binaries and tracking backends, use the
[exposure checker and revision matrix](linux-exposure-check.md).

Implementation constraints and regression rationale are in [linux-notes.md](linux-notes.md).

For a bounded six-case comparison of frame generation, depth clears and texture detiling,
see [play-session diagnostics](linux-play-diagnostics.md).

The [three-run Nexus check](linux-nexus-check.md) verifies prepared pipeline reuse, captures
early streaming, and compares the opt-in `DS_FG_PACING=adaptive` presenter. Legacy FG pacing
remains the default pending visual and latency validation.
