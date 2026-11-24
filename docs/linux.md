# Linux (experimental)

Regaliant runs the decrypted PS5 executable natively on Linux x86-64, with Vulkan graphics
and SDL3 windowing, input and audio. Tutorial gameplay, save resumption and window-close
checks have passed automated tests. A full physical play session and the full shader tour
still need testing. Performance varies by scene; steady 60 fps is not yet established.

## Build and play

Install Rust/Cargo, a C/C++ compiler, CMake, pkg-config, SDL3 development files and a Vulkan
loader and driver. Fontconfig and a system TrueType/OpenType font provide text entry and
keyboard labels. See [install.md](install.md) for the required decrypted game files.

From the checkout:

```sh
CARGO_HOME="$PWD/.cargo-home" cargo build --release -p dsport
target/release/dsport --app /path/to/PPSA01341-app0
```

The second command starts a normal resizable window with real audio. `--app` also accepts a
`.zar` archive of the game folder; it is read directly without extracting into the game tree.
Linux archive creation rejects source symlinks and outputs inside the source or extra read-only
roots. Plain ELF files and SELF containers with decrypted segments both work, in folders or
archives. Encrypted or compressed SELF segments are not supported.
Alternatively, export `DS_GAME_DIR` and run `target/release/dsport`.
NVIDIA is preferred when several GPUs are available. SDL selects Wayland or X11;
`SDL_VIDEO_DRIVER=wayland` or `SDL_VIDEO_DRIVER=x11` selects one explicitly.

On first start, select a language, adjust brightness, choose Continue Offline, and create a
character. Shaders compile as new scenes and effects appear, so initial visits can stutter.
Use the preparation command below to compile shaders before playing.

Keyboard, mouse and SDL gamepads (including Xbox, DualSense and DualShock 4) use the game's
PC settings tab and `controls.txt`. PlayStation controllers select PlayStation button labels;
the touchpad click maps to the shared View/touchpad binding. Rumble and the light bar use SDL;
adaptive triggers and motion sensors are not mapped. Actual controller output needs physical
testing. Click the window to capture the mouse; F1 releases or recaptures it. Alt+Enter or F11
switches between windowed and borderless fullscreen. Closing the window flushes completed
saves and shader records before exiting. Quit through the game menu when possible; closing
during a game save cannot guarantee an unfinished save completes.

## Files and diagnostics

Normal play stores settings, controls, achievements and saves under
`$XDG_DATA_HOME/Regaliant` (default `~/.local/share/Regaliant`). Logs, shader stores and preparation
state use `$XDG_CACHE_HOME/Regaliant` (default `~/.cache/Regaliant`). `DSPORT_HOME=/absolute/path`
puts both in that directory instead. `DS_LINUX_SHADER_CACHE=/absolute/path` selects a separate
persistent renderer cache.

The game source is read-only. Output paths into it, including symlink aliases and targets,
are rejected. `DS_READONLY_ROOTS=/read-only/tree:/another/tree` protects additional trees.
Never put saves, caches or logs inside the game folder. Do not add game files or captured
shader binaries to Git.

`diagnostics.txt` in the settings folder accepts `DS_NAME=value` lines and `#` comments, as on
Windows. Existing environment values take precedence; preparation's required switches take
precedence over file defaults. UTF-8 and UTF-16LE text with a byte-order mark are accepted.
Remove diagnostic overrides before comparing performance.

## Add-ons and PS5 save import

Put your own pre-order or Digital Deluxe packages in `Add-ons` under the settings folder:
`$XDG_DATA_HOME/Regaliant/Add-ons` (default `~/.local/share/Regaliant/Add-ons`). Keep the package
names, such as `EP9000-PPSA01341_00-DEMONSSOULSGIFT1.pkg` or `...GIFT2.pkg`; unpacked folders
with those names also work. Restart the game to detect them. Shader preparation uses its own
settings folder without your add-ons.

To import a decrypted PS5 save, **copy** its folders into `PS5 Saves` under the settings folder.
Each `SAVEDATA...` or `sdimg_SAVEDATA...` folder must contain `USR-DATA`; an enclosing
`PPSA01341` folder is fine. Enable **Import PS5 Saves** in SETTINGS > PC, then quit and restart.
Existing saves are backed up in `save-backups` before replacement. The import folder is renamed
after a successful import. PS5 backup folders (`sdimg_sce_bu_...`) and symbolic links are skipped.
The importer does not decrypt saves. Keep original exports elsewhere; never use the game folder
as the import location. `DSPORT_HOME` also relocates Add-ons and PS5 Saves.

Linux keeps saves in the settings folder's `saves` directory. The Windows Documents/Saved Games
relocation choices do not change that location on Linux.

## Prepare shaders

```sh
target/release/dsport --app /path/to/PPSA01341-app0 --prepare-shaders 4
```

This runs hidden and silent with an isolated save: new game, compute preparation and the
shared Windows multi-level self-play tour. It does not use or overwrite player saves. Four is
the compiler thread count (1–64). The default time limit is four hours; add
`--prepare-seconds 600` for a bounded attempt. Ctrl+C stops the current child and keeps progress.
Run the same command to resume. Exit 12 means paused, not fully prepared.

Progress, logs and checkpoints are in `prepare/linux-full` under the cache directory. A
checkpoint is not proof of visual correctness. The full Linux tour remains unverified.
Changing the game, cache location or selected levels requires a separate preparation directory.
Driver or translation-policy changes invalidate completed shader stages but retain the isolated
save. Compiled caches persist between processes. Upgrades may rebuild incompatible pipelines;
0.1.2 changes Blackwell's translation modes and reconvergence declaration.

`--precompile-all N` compiles already known shader specs without playing. `--pb-check write`,
then `--pb-check read`, verifies pipeline reuse across processes; both take `--app`. Use `timeout`
for unattended commands. Source pipeline creation is not itself proof of a driver compile:
inspect `pipeline-reuse.json` for actual binary hits and refusals.

## Pacing and memory tracking

FIFO presentation is the default. `DS_LINUX_PRESENT_MODE=mailbox` selects MAILBOX where
supported; `immediate` is a tearing diagnostic override. Guest timing remains 59.94 Hz.
Presentation timing uses KHR present-wait completion, not a physical scanout timestamp.
Compositor VRR control and display-mode switching are not implemented.

Exact `mprotect` dirty tracking is the default. `DS_LINUX_WRITE_WATCH=uffd` enables async
userfaultfd write protection with `PAGEMAP_SCAN` on supporting kernels (Linux 6.7+). It removes
userspace write-fault signals but has not improved whole-game throughput in measurements.
Unsupported permissions or features log a fallback. Both paths retain announced host writes
and exact upload tracking. C4 held pages remain disabled; FE2 eager write-back has extra copy
and wait costs.

Linux idle job workers now sleep after 10 µs without work, for 500 µs between polls, with no
spinning watcher. The 13 guest workers are retained because jobs have worker-specific masks.
`DS_JOB_IDLE=50 DS_JOB_SLEEP=200 DS_JOB_WATCH=2000` restores the old policy for comparison.
The frame-rate impact of the new defaults still needs the guarded GPU relay.

## Developer verification and GPU relay

```sh
CARGO_BUILD_JOBS=2 CARGO_HOME="$PWD/.cargo-home" cargo check --workspace
CARGO_BUILD_JOBS=2 RUST_TEST_THREADS=2 CARGO_HOME="$PWD/.cargo-home" cargo test --workspace
python3 -m unittest discover -s scripts -p 'test_linux*.py'
export DS_GAME_DIR=/path/to/PPSA01341-app0
export DS_RELAY_SAVE_SEED=/path/to/prior-relay/user
export DS_RELAY_SHADER_SEED=/path/to/prepared-shaders # optional
scripts/linux-gpu-test.sh # includes one visible run; choose either wrapper
```

Both scripts default to the same short suite: one clean offscreen gameplay measurement and
one visible 800x450 FIFO run, each bounded to 180 seconds (200-second hard timeout). Game audio
is silent and input is scripted. There is no default game path. Set
`DS_RELAY_SAVE_SEED=/path/to/prior-relay/user` to an immutable relay save directory containing
`saves/`; optionally set `DS_RELAY_SHADER_SEED=/path/to/shaders` to reuse prepared shaders.
Each session uses a copy. Without a save seed the short suite stops before building or running.

Both scripts accept `--app PATH`, a positional game path, `--dry-run` and `--full`. Only `--full`
creates a fresh save when needed, runs bounded shader preparation and the long tracking,
cache, diagnostic, FIFO/MAILBOX and audio checks. It can take over an hour; use it deliberately.
`DS_SCATTER_FIXTURE=/path/to/scatter-fixture` optionally enables the full suite's isolated fog
regression. Fixtures come from your own captures and are never shipped.

Every workload, including builds and preparation children, has these limits:

- Eight allowed logical CPUs, preferring performance cores using sysfs core type/capacity/
  frequency and sibling topology. Override with `DS_RELAY_CPUS=0-3,8-11`. Runs use `nice 10`.
- `DS_COMPILE_WORKERS=2` by default. Shader pools are capped and a shared driver-compilation
  permit prevents foreground/background pools from each using the whole budget. Cargo uses
  the same job limit. Drivers may still create their own internal threads within the CPU set.
- CPU package temperature is sampled every two seconds. Above `DS_RELAY_TEMP_PAUSE=95` °C,
  the owned process groups stop; below `DS_RELAY_TEMP_RESUME=85` °C, they resume. A pause longer
  than `DS_RELAY_TEMP_ABORT_S=60` seconds aborts the run with a graceful-stop request and bounded
  TERM/KILL fallback. The whole pause is bounded even if temperature sits between thresholds.
- Before each workload, cooldown waits for the resume threshold, up to
  `DS_RELAY_COOLDOWN_S=120` seconds. Missing or unreadable sensors stop the relay. It prefers
  the `Package id 0` sensor, falling back to the hottest coretemp/k10temp sensor.
- Non-measurement offscreen workloads have a 30 fps ceiling, including shader preparation.
  Performance measurements have no extra relay cap.

`monitor.jsonl` records temperatures, system load and pause/resume events with an fsync after
**every line**. `thermal.json` records the outcome. Any pause invalidates the whole session's
performance measurements; those numbers are excluded from comparisons. GNU timeout remains
an independent wall bound. Thermal shutdown flushes completed saves where possible, but a
hard-kill fallback cannot guarantee completion of an in-progress save. No desktop power or
fan settings are changed.

The short offscreen measurement compares the old worker policy and the new bounded-sleep
policy for 15 seconds each in the same stationary scene. Read `job-idle-summary.json` for BPE
worker CPU seconds per gameplay second, failed takes, jobs and fps. The overall 30-second fps
mixes both policies; use the per-policy values when comparing. Warm-up/order can affect this
short A/B, and a thermal pause invalidates both samples. Movement remains capture-free; boundary
PNGs and KHR present-wait percentiles verify the visible path.

All output stays under `local-data/linux-port/{gpu-test,present-test}`. `latest.txt` identifies
the run; `summary.txt` records outcomes. Extra protected trees use `DS_READONLY_ROOTS`.

Inspect `comparison.json`, `measurement.txt`, `pipeline-reuse-summary.json`,
`presentation-summary.json`, thread CPU/wait reports, validation/crash/unresolved-import logs,
and frame PNGs. `frame-stalls.jsonl` records long-frame operation/wait reasons; categories
overlap and must not be added as independent costs. Self-play success alone does not prove
correct rendering. Keep handoff notes in gitignored `local-data/linux-port/STATUS.md`.

For a Windows compile check, install a Rust toolchain with the `x86_64-pc-windows-gnu` target
and MinGW. `scripts/linux-windows-check.sh` uses the isolated rustup under `.cargo-home/bin`,
with `RUSTUP_HOME=$PWD/local-data/linux-port/toolchains/rustup` and repository `CARGO_HOME`.
It checks the workspace under a 600-second timeout; it does not test Windows execution or
MSVC linking. Install isolated rustup with `--no-modify-path --default-toolchain none`, then
install stable and the Windows GNU target using those environment variables.

## Unsupported or pending

- Graphical launcher, including its archive creation UI. The shared archive reader and
  `cargo run -p dsconfig --example zartool --` command-line archive tool work on Linux.
- DLSS, XeSS, Streamline, FSR, HDR, exclusive display modes and compositor VRR control.
- WDDM cross-process VRAM/process snapshots. Shared image-cache eviction still uses Vulkan
  memory budgets; Linux cannot apply the Windows other-process accounting adjustment.
- Windows Sonic, endpoint form-factor detection, thread cycle counts, physical-core topology,
  Windows priority/power controls, hardware watchpoints and cross-thread stack sampling.
- Windows save relocation, custom themed cursor, full IME composition/bidirectional shaping
  and font fallback. Basic character-name entry is implemented.
- Full-tour preparation, physical input/audio playtesting and long-session stability.

The platform layer leaves host FS/TLS intact and patches supported guest FS accesses to GS.
Signal handlers use alternate stacks and avoid allocator/logger/mutex/Vulkan calls on direct
recovery. Shutdown persists completed saves and shader records, then ends process-lifetime
workers with `_exit`; it is not a reusable in-process renderer teardown.
