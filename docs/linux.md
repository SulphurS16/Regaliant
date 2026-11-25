# Linux (experimental)

Regaliant runs the decrypted PS5 executable natively on Linux x86-64, with Vulkan graphics
and SDL3 windowing, input and audio. Tutorial gameplay, save resumption and window-close
checks have passed automated tests. Players have confirmed keyboard/controller gameplay and
in-game audio and sharp fractional-scale fullscreen output. Movie audio and the full shader
tour still need verification after the latest fixes. Performance varies by scene; steady 60 fps is not yet established.

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
testing. F1 captures or releases the mouse. Controller input hides the pointer over the window;
mouse movement or a mouse button shows it again when capture is off. In fullscreen, the pointer
also hides after three seconds without mouse activity. These visibility changes do not grab the
mouse. Alt+Enter or F11
switches between windowed and borderless fullscreen. Closing the window flushes completed
saves and shader records before exiting. Quit through the game menu when possible; closing
during a game save cannot guarantee an unfinished save completes.

On a scaled Wayland display, the window now requests a high-density drawable: the swapchain
uses physical pixels, while SDL mouse coordinates are converted to match the UI. Both windowed
and borderless modes follow compositor scaling. On a 3840×2160 screen, use **PC → Render Scale →
100%** for native 4K rendering; **Visual Mode** keeps the title's usual 2560×1440 render size.
The separate Resolution/Refresh Rate controls do not change Linux desktop modes; choose the
monitor mode in your compositor. A 4K swapchain alone does not mean the game renders at 4K.
Logs report logical and drawable sizes after resize/scale changes.

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
python3 scripts/linux-prepare-shaders.py --app /path/to/PPSA01341-app0
```

This runs the full multi-level tour hidden and silent, using an isolated save. It never uses or
replaces your player saves. It uses eight allowed logical CPUs (preferring performance cores),
nice 10, two shader compiler workers, and a 30 fps cap. The thermal guard pauses after ten
continuous seconds at or above 103 °C, resumes below 97 °C, and aborts after 60 continuous
seconds at or above 105 °C. Missing temperature
sensors stop the command. `DS_RELAY_CPUS` and `DS_COMPILE_WORKERS` adjust CPU limits; the relay
thermal variables below also apply. Monitor records are fsynced under
`local-data/linux-port/shader-preparation/`.

Allow roughly an hour on a fast, unrestricted PC; with these CPU limits and cooling pauses,
several hours may be needed. The full Linux tour has not yet been timed. Each attempt stops
after four hours by default; `--seconds 600` makes a ten-minute attempt. Ctrl+C stops the
children and preserves progress. Repeat the same command to resume. Exit 12 means paused;
exit 75 means the guard stopped it. Do not run preparation while playing.

Preparation uses the same XDG shader store and translation overrides as normal play, including
`DS_PLAIN_MODES` and `DS_MAX_RECONV` from the environment or your settings folder's
`diagnostics.txt`. Use the same environment for both commands. Changing translation settings or
PC graphics settings restarts shader stages while retaining the isolated save. New effects and
variants can still compile during play. The direct `--prepare-shaders N` runtime option remains
available, but does not itself provide CPU affinity or a thermal guard.

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
MAILBOX can replace queued frames without tearing, but cannot turn 27 ms of rendering into
60 new frames per second. FIFO does not inherently force every renderer below 60 fps to 30:
queue depth and compositor scheduling matter too. FIFO_RELAXED can tear when a frame is late,
so it is not a tear-free alternative and is not enabled here. See the
[Vulkan present-mode definitions](https://docs.vulkan.org/refpages/latest/refpages/source/VkPresentModeKHR.html).
The present-wait observer runs on a separate thread; it is not a per-frame game-thread wait.
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

General relays, builds and preparation children have these limits. The native-4K steady
measurement below uses the as-played thermal exception:

- Eight allowed logical CPUs, preferring performance cores using sysfs core type/capacity/
  frequency and sibling topology. Override with `DS_RELAY_CPUS=0-3,8-11`. Runs use `nice 10`.
- `DS_COMPILE_WORKERS=2` by default. Shader pools are capped and a shared driver-compilation
  permit prevents foreground/background pools from each using the whole budget. Cargo uses
  the same job limit. Drivers may still create their own internal threads within the CPU set.
- CPU package temperature is sampled every second. At or above `DS_RELAY_TEMP_PAUSE=103` °C
  for `DS_RELAY_TEMP_HOLD_S=10` continuous seconds, the owned process groups stop; below
  `DS_RELAY_TEMP_RESUME=97` °C, they resume. At or above `DS_RELAY_TEMP_ABORT=105` °C for
  `DS_RELAY_TEMP_ABORT_S=60` continuous seconds, the run aborts with a graceful-stop request
  and bounded TERM/KILL fallback. All thresholds are overridable. The monitor also enforces the
  command's wall timeout while its process group is stopped.
- Before each workload, cooldown waits for the resume threshold, up to
  `DS_RELAY_COOLDOWN_S=120` seconds. Missing or unreadable sensors stop the relay. It prefers
  the `Package id 0` sensor, falling back to the hottest coretemp/k10temp sensor.
- Non-measurement offscreen workloads have a 30 fps ceiling, including shader preparation.
  Performance measurements have no extra relay cap.

`monitor.jsonl` records temperatures, system load and pause/resume events with an fsync after
**every line**. `thermal.json` records the outcome. These general relays invalidate performance measurements after any pause; those numbers are
excluded from comparisons. The native 4K relay below uses exact phase boundaries instead. GNU timeout remains
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
- Full-tour preparation, broader controller/audio hardware coverage and long-session stability.

The platform layer leaves host FS/TLS intact and patches supported guest FS accesses to GS.
Signal handlers use alternate stacks and avoid allocator/logger/mutex/Vulkan calls on direct
recovery. Shutdown persists completed saves and shader records, then ends process-lifetime
workers with `_exit`; it is not a reusable in-process renderer teardown.

## Resolution and combat-UI diagnostics

Linux NVIDIA devices now retain the subgroup translation modes; maximal reconvergence is off
by default with them. This fixes the reported Blackwell rainbow-speck scene. Windows keeps its
upstream device policy. `DS_PLAIN_MODES` and `DS_MAX_RECONV` remain explicit overrides, including
in shader preparation.

Render Scale **100%** on a 4K display renders the scene at 3840×2160. Choose **Visual Mode** for
the title's usual 2560×1440 scene while keeping the sharp 4K swapchain and UI. The runtime logs
the actual render-size changes; a message saying “the mode's was 2560x1440” describes the size
being replaced, not the active size. Comparing 4K fullscreen with 1440p offscreen is not a
regression measurement. First-use shader compilation can also dominate short FPS averages.

For a controlled relay, export `DS_GAME_DIR`, `DS_RELAY_SAVE_SEED` (an isolated snapshot containing
`saves/`) and `DS_RELAY_SHADER_SEED`. Build first, then run one case at a time:

```sh
python3 scripts/linux-playtest.py --case 1440
python3 scripts/linux-playtest.py --case 4k
```

Each case opens a visible fullscreen window, uses null audio and scripted input, and runs for
at most 180 seconds (timeout 210). Both use the same copied save, a 4K-capable drawable, and an
explicit guest render size. They retain the thermal guard, eight-CPU placement and two compile
workers. `--repeat 2` reuses the first run's caches. `--pipeline-cache --repeat 2` compares the
ordinary Vulkan pipeline cache when the driver refuses stored KHR pipeline binaries; this is
an experiment, not the play default. Output is under `local-data/linux-port/playtest/`.
Read per-session `metrics.json`, `thread-summary.txt`, `waits-*/wait-summary.txt`, presentation
percentiles and the runtime's `pipeline-reuse.json`. No GPU timestamps or validation are enabled
in these timing cases. A thermal pause invalidates a measurement.

`--case diagnostic` instead runs offscreen, capped at 30 fps, for at most 300 seconds (timeout
330). It traces clear-metadata writes/bindings and dumps two gameplay frames' target images and
command descriptions around a camera move after the scripted movement. These readbacks
invalidate timings. The save must put a visible enemy nearby to reproduce floating health
bars; the script does not assert that combat happened. Inspect `ui-clear-probe-*` PNGs,
`capture-*.txt`, `targets-*/stats.txt` and the `trace clears` lines. The attachment-cache
invalidation fix has a CPU clear-routing regression test; whether it resolves the reported
combat trail still needs this GPU check. It does not clear targets unconditionally each frame.

`python3 scripts/linux-session-report.py /path/to/dsport.log` summarizes an existing log without
modifying it. Hitch spans overlap across threads and cannot establish a per-thread CPU profile.

## Pipeline reuse checks

The pipeline store has two formats: ordinary Vulkan pipeline-cache blobs (`pipelines/*.bin`)
when binaries are disabled, and captured `VK_KHR_pipeline_binary` data (`binaries/*.pb` plus
shared binary data) when supported. Shared data is scoped by the same device compatibility
namespace as its parent pipeline file; a matching binary key alone cannot reuse older data. Captured binaries are intended to replay without NVIDIA's
internal cache. Ordinary cache blobs are not a guarantee that shader compilation is avoided.

The binary header now records schema `DSPBIN04`, a device compatibility key (pipeline-cache
UUID, driver/vendor/device IDs, enabled extensions and features), and the code/create-info
fingerprint. The latter covers translated code, reconvergence, descriptor layouts, push
constants, pipeline flags and requested subgroup sizes. Graphics state keys include formats,
blend state and topology; window dimensions are dynamic viewport/scissor state. Previous
schema files rebuild once. An existing `binaries-mode.txt` decision to use ordinary caches is
retained for normal play; the isolated relay omits that old decision when testing fresh binaries.
Changing a translation or device feature policy also rebuilds
incompatible entries. These are counted separately from driver refusals. This compatibility
identifier is application-managed: upstream disabled the actual `vkGetPipelineKeyKHR` global-key
query after it triggered a driver crash. That workaround remains, so a driver session change
not reflected in these fields is still a possible cause of cross-process refusal.

Capture and replay now keep the same flags2 structure in the create-info chain. Only the
capture bit and the binary-input structure differ, as allowed by the
[Vulkan binary replay contract](https://docs.vulkan.org/refpages/latest/refpages/source/VkPipelineBinaryInfoKHR.html).
Whether that change eliminates this NVIDIA driver's refusals still needs the hardware relay.

Normal play and shader preparation also share the fallback driver cache under
`$XDG_CACHE_HOME/Regaliant/driver-cache/` (default `~/.cache/Regaliant/driver-cache/`). Previously,
preparation isolated this cache, so normal play could not benefit from it when a binary was
refused. Explicit `__GL_SHADER_DISK_CACHE_PATH`, `MESA_SHADER_CACHE_DIR` and `CUDA_CACHE_PATH`
values are preserved and checked against read-only roots. NVIDIA's requested cache budget
defaults to 8 GiB; `__GL_SHADER_DISK_CACHE_SIZE` overrides it. Linux defaults
`__GL_SHADER_DISK_CACHE_SKIP_CLEANUP=1` to retain preparation work; an explicit value is preserved.
Mesa defaults to `MESA_SHADER_CACHE_MAX_SIZE=8G`, also overridable. The full tour's cache footprint
has not been measured; with cleanup disabled, monitor disk usage rather than treating the budget
as a hard disk-space bound. Startup logs the selected paths and settings before Vulkan starts. NVIDIA documents the cache path/size controls in its
[driver README](https://download.nvidia.com/XFree86/Linux-x86_64/570.124.04/README/openglenvvariables.html).
The actual Vulkan driver's use of that cache is checked by the relay, not assumed from the
requested budget. Rerun the normal preparation command after this upgrade; it retains its
isolated save and repeats shader stages with the new contract and shared driver cache.

For the reviewer, export `DS_GAME_DIR`, `DS_RELAY_SAVE_SEED` (containing `saves/`) and
`DS_RELAY_SHADER_SEED`, then run:

```sh
python3 scripts/linux-pipeline-reuse.py
```

This copies the seed translations/specs but omits old binaries, runs twice from the same save,
and keeps the new binaries and driver cache between runs. Read-only seeds stay untouched; only
new private copies receive owner write permission. A write/rename preflight rejects an unusable
store before starting the game. Both sessions are hidden, silent,
limited to 30 fps, two compiler workers and the relay CPU/thermal guard. Each lasts at most
300 seconds with a 330-second timeout; `--seconds` adjusts that bound. Fresh captures are
replayed immediately as a diagnostic. This is a reuse test, not an FPS benchmark.

Read `local-data/linux-port/pipeline-reuse/run-*/comparison.json` and `summary.txt`. Passing
requires at least 100 matching pipelines covering 95% of the second run, at least 99% reuse,
no store-write failures, no fresh-replay failures and no
more than one second of aggregate source-creation calls for those matching pipelines. Newly
encountered or changed pipelines are reported separately. Per-pipeline outcomes, creation
latencies, stale reasons and cache environment are in each runtime's `pipeline-reuse.json`.
An incomplete route, truncated ledger or thermal interruption cannot pass. Optional
`--cold-driver` gives the second run an empty driver cache while retaining the port's binaries,
to test independence from the internal cache; it never deletes an existing cache.


## Diagnosing a camera-dependent black surface

The reported corridor artifact is not yet attributed to a particular draw. Linux and Windows
share the same unmapped-texture handling: zero-fill on first upload, retain prior contents on
later partial uploads. A partial-residency warning alone does not identify the black surface.
Those warnings now include image addresses and dimensions for comparison with a frame capture.

For a diagnostic play session, add `DS_SCENE_PROBE=1` to the normal play command. Press F9 once
at the bad camera angle and again at a nearby angle that renders correctly. Each press requests
a window PNG, `capture-<flip>.txt`, intermediate `targets-<flip>/` images/statistics, and sampled
textures in `scene-<flip>/textures/` under that session's log directory. Texture PNG support is
format-dependent; the text records the bound descriptors and depth, clipping and culling state.
Readback can stall the game, so use this only to capture the problem, not to measure performance.
F9 keeps its ordinary input behavior without `DS_SCENE_PROBE=1`. Do not publish captures containing
game assets with a source-code patch.

## Native 4K performance measurement

Build the release runtime, export `DS_GAME_DIR`, and run:

```sh
python3 scripts/linux-steady.py
```

This uses `local-data/linux-port/save-seed` and automatically selects the newest pipeline-reuse
run with verified binary hits. `--save-seed` / `DS_RELAY_SAVE_SEED` and `--shader-seed` /
`DS_RELAY_SHADER_SEED` override those sources. It copies them into a new private directory;
neither player saves nor the seed caches are modified. When selecting a reuse run automatically,
its driver cache is copied too. A thermal pause in that old run does not invalidate its binary
replay evidence, but does invalidate its FPS measurements.

The test opens a visible borderless fullscreen window on SDL's selected display, uses native
3840×2160 scene rendering and output, TAA, no upscaler, FIFO presentation and an uncapped game.
A 60 Hz display still limits FIFO to its refresh rate. It checks the actual swapchain and guest
render-size logs, so a different display resolution cannot silently pass as a 4K measurement.
Offscreen rendering is not equivalent: it omits the compositor and swapchain waits.

The saved-load path retries a missed Continue Offline confirmation at most six times, eight
seconds apart, using two-frame presses. Retries require the same observed menu and boot HUD;
a change permanently stops retries, so unknown dialogs receive no input. It waits at most five
minutes for gameplay HUD state with no menu and advancing frames, then requires the existing
five-second/120-frame settling check. A load timeout logs UI state and requests a screenshot.
This avoids treating a lingering dialog or the loading HUD as gameplay.

After resuming and settling, it records 30 seconds stationary, then 60 seconds of scripted
walking: alternate forward/backward every five seconds, with a short neutral gap. There is no
physical input, mouse grab or audible audio. Collision and combat may change the pose; compare
the before/after PNGs as well as the numbers. Screenshots finish outside the timed intervals. The boundary barrier tracks GPU readback through
PNG completion, including error paths, and waits up to 90 seconds; it does not assume that two
flips or a ten-second encoder wait are enough.
Validation, capture probes and GPU timestamps are off throughout the clean session. After each
timed phase and its boundary PNG, a separate 32-flip CPU-only trace records waits with normal
batching. A separate 32-flip TSC profile attributes nested recording self time, as described
below. The old four-flip event detail is opt-in only. All profiles are excluded from the
FPS intervals. The relay
uses eight CPUs, nice 10, two compiler workers, the as-played thermal policy below and a 630-second wall timeout per process (600-second runtime
limit); successful routes finish early and every process is reaped. The player's settings are untouched.

Clean measurement processes run **first**, followed by a separate diagnostic process. It measures
five seconds stationary and five seconds walking, with 32-flip GPU segment traces, CPU recording profiles and write-back counts by engine allocation. Its write-back census
keeps the normal split-line store pattern; it does not skip any writes. Profiling and readback
alter timing: these are **diagnostic timings**, not baseline FPS. Segment duration includes its
barriers and waits and does not measure shader-core occupancy. An optional five-second `perf`
sample is attempted when the tool and permissions allow it; `perf-console.log` reports failures.
Use `--no-gpu-timing` to omit this diagnostic process.

`--tracking both` compares mprotect and uffd from the same original save and cache snapshots.
`tracking-comparison.json` reports FPS changes, faults and CPU work only when both runs have
valid, fully warm measurements. Missing CPU wait traces fail the measurement check.
The default suite interleaves four clean sessions, then runs the GPU diagnostic. Allow
roughly 15–25 minutes plus cooling; every launch may wait up to ten minutes to reach 70 °C.
A failed launch, load, measurement or analysis does not cancel the other sessions. Even after a
guard abort, the next process must independently pass sensor checks before it can start.
Every launch must pass the bounded cooldown, including in observe mode. Explicit user cancellation stops the suite. `sessions.json` and the top of
`summary.txt` list every planned session as pending, running, completed, failed or cancelled,
with its attempts; results retain exit codes, self-play failures and analysis exceptions. Exact
monotonic phase clocks determine whether a pause overlaps a measurement. Pauses are never
subtracted from FPS: an overlapping phase is invalid, while a pause during loading alone does
not invalidate a later phase. There are no inserted warm retries: the next planned visit
uses that backend's warmed cache, preserving the A,B,A,B sequence. Paused FPS and trace timings are
omitted from the text summary, with raw evidence retained in JSON. Trace clocks are matched
against the guard too. Older outputs without clocks remain conservatively invalid.

For a route already measured, use its updated shader directory as `--shader-seed`; it includes
pipelines compiled during walking. Always use a new writable copy, leaving the original seed
unchanged. For example:

```sh
python3 scripts/linux-steady.py --shader-seed /path/to/walking-shader-seed --tracking both
```

The normal tracking default remains mprotect. Earlier same-scene measurements did not establish
an uffd speed advantage. Changing that default requires comparable gameplay and correct frames.

Read `local-data/linux-port/steady/run-*/summary.txt` first. Per-run `steady-summary.json` has:

- Frame-time and FPS percentiles, KHR presentation-completion percentiles and present-stage costs.
- Faults/second, upload and write-back bytes/flip, uffd scan time, job-worker activity, and pipeline
  creation counts/time within each interval. Source pipeline creation disqualifies a fully warm
  result. The interleaved suite reports that pair as invalid; the next planned visit reuses
  the private warmed cache. Runtime or guard failures do not prevent independent sessions.
- CPU work and runnable delay per thread and per flip. Threads present at both boundaries are
  counted; residual non-running time includes sleep and unclassified waits. It is not CPU work.
- Sampled device-wide GPU clocks, utilization and power, and per-flip wait coverage for recording,
  submission, fences, labels, batch slots and videoout. Clean-adjacent CPU traces use normal batching;
  GPU segment traces separately identify write-back, draw, compute and barrier time.
- `stationary/walking-cpu-profile.txt`, `*-gpu-status.txt` and `*-writeback.txt` in the diagnostic
  session's log folder. The write-back census reports allocation owners and CPU-write/renderer-read
  activity. An “untouched” bucket does **not** prove the guest CPU never reads it; it cannot justify
  skipping write-back without a coherent read-fault mechanism.

Concurrent CPU and GPU lanes overlap: do not add their milliseconds into one frame budget.
KHR completion is a CPU observation rather than a physical scanout timestamp. Thermal pauses overlapping a timed phase,
missing evidence, wrong resolution, failed runs or tracking fallback invalidate the comparison.
The earlier approximately 56 fps tutorial result rendered the scene at 2560×1440; native 4K has
2.25 times as many scene pixels, even though both modes use 4K output targets. That old number
is not a native-4K performance baseline. No 60 fps claim is made until this relay returns.


Linux now returns from an explicit address wake even when the label word is unchanged.
A submitted batch can publish a future label write before the write executes; the
renderer rechecks that promise and installs the same GPU dependencies. This is enabled
by default after successful frame and wait-time comparisons. `DS_LINUX_ADDRESS_WAKE=0`
restores the old behavior for diagnosis. Windows keeps its native wait implementation.

For a controlled native-4K comparison, use a warm shader store and an existing save:

```sh
python3 scripts/linux-steady.py --shader-seed /path/to/warm-shader-store --tracking both --no-gpu-timing
```

Keep `DS_GAME_DIR` exported as above; optionally pass `--save-seed`. The default comparison
is **mprotect, uffd, mprotect, uffd**, with the wake fix in every run. Each session cools
before launch to **70 °C or less**, with a bounded 600-second wait. Set
`DS_RELAY_SESSION_TEMP` to a lower temperature if desired and `DS_RELAY_COOLDOWN_S` to
change the wait bound (at most 600 seconds). If cooling fails, that session does not
launch and is reported as failed. Cooling and temperatures are fsynced to `monitor.jsonl`.
Startup can heat the CPU again: compare the actual phase temperatures too.

Each backend reuses its private warmed cache for its second visit. The script preserves
A,B,A,B order: it reports any interval with new pipeline compilation as invalid instead
of inserting a warm retry. `tracking-comparison.json` and `summary.txt` give each pair's
FPS difference, temperatures and kernel/user CPU costs. A failed first pair is never
silently replaced by results from the second pair. No original save or cache seed is
modified. Normal play still defaults to mprotect; uffd has not established an advantage.

The script takes `~/.cache/cpu-heavy.lock` for the entire suite, including cooling, or
uses the path in `DS_CPU_HEAVY_LOCK`. It refuses an existing holder. Do not also wrap it
in `flock`. Normal player launches do not take this relay lock.

Profiling needs no `perf` permissions. After each clean phase, a separate 32-flip sample
accumulates nested TSC scopes on gpu-gfx. Fixed-size thread-local counters retain every
call, rather than dropping events at the old detailed trace's buffer limit. They cover
the packet loop, packet dispatch, registers/state checks, draw/dispatch emission,
shader/resource preparation, descriptors, barriers, image/buffer lookups and FE2
placement/version checks. There are no per-scope allocations or shared recording locks.
Read `stationary-cpu-self.json` and `walking-cpu-self.json` in the session's runtime log
folder, or their top ten entries in `summary.txt`.

Self time excludes nested scopes and instrumented waits. TSC measures elapsed time,
including preemption and profiling overhead; the report separately gives boundary-sampled
thread CPU time. These adjacent profiles do not supply clean FPS. Stack overflow or
non-nested guards invalidate a profile rather than silently dropping samples. Ordinary
wait traces remain, including `label-waits.json` with per-producer wait distributions.
The older, event-heavy detail trace is optional via `DS_CPU_EVENT_DETAIL=1`; its event
cap makes it unsuitable as the source of complete recording attribution.

`--perf-gpu-gfx` remains an optional separate sampling run when host permissions allow it.
It preflights permissions before launching the suite, never changes kernel settings,
and is unnecessary for the default TSC profile. That optional run also samples direct
write-fault recovery; its estimate excludes kernel signal entry/return. Use the clean
mprotect/uffd comparison and per-thread user/kernel CPU deltas for the broader cost.

Steady measurements run **as played**: no temperature-triggered SIGSTOP. The guard samples
package temperature, per-CPU frequency and available kernel throttle counters every second,
fsyncs each `monitor.jsonl` record, and aborts after 105 °C continuously for 60 seconds.
Clock and throttle fields unavailable from sysfs remain unknown, not zero. Each phase's
`cpu_environment` in `steady-summary.json` summarizes the matching samples. Package
throttle counters replicated across CPUs must not be added together. These are hardware
samples alongside frame times, not estimates of lost performance. Existing timeout and
sensor-failure checks remain active. `--thermal-pause` restores preparation-style cooling;
long shader preparation retains that pause policy by default. Temperature thresholds remain
overridable with the documented environment variables.

Legacy detailed events use per-thread buffers and format their names after capture, avoiding a
shared trace lock and string allocation per packet/lap. Phase boundaries also wait for
the trace writer to finish, so serialization cannot spill into the next timed phase. The detailed four-flip trace still
has instrumentation overhead: do not substitute
its frame time for clean FPS. `record_packet` previously included uninstrumented draw and
compute work, not just decoding. Separate categories now cover repeat draws, GPU append,
indirect arguments, draw/dispatch residual work, memory writes/copies and releases. Compare
these spans between variants together with ordinary wait traces and clean phase CPU times.

## Startup crashes and recording experiments

Linux fatal reports include the faulting thread, bytes at the instruction pointer, and current
host mappings with file offsets for the instruction and stack words. A compiler fault also
lists active pipeline keys. `DS_PIPELINE_CRUMBS=1` saves the input SPIR-V before driver calls;
use it for diagnosis, not performance measurements. Only a fault inside NVIDIA's compiler on
the thread compiling a named pipeline adds that pipeline to the existing driver-specific
`pipelines/crashed.txt` quarantine. Other in-flight pipelines are not blamed. This avoids
recompiling a known crashing pipeline at every launch, but its draw/dispatch may be omitted;
it is a containment measure, not a rendering fix. Linux also writes a durable intent before
source compilation, under `pipelines/compile-pending/`, and clears it only after success.
An unfinished compile is skipped on the next launch, even if no signal report was written.
A first unknown compiler fault can still terminate the process. Keep reports and markers
for diagnosis. Corrected translator code gets a new quarantine identity.

The invalid cube texel fetch implicated in NVIDIA's startup compiler abort has been replaced
with an exact fetch through a cube-face array view. See the [design and offline regression
checks](linux-cube-fetch.md). Existing translations regenerate once; pipeline binaries whose
SPIR-V and creation contract are unchanged remain reusable.

To reproduce startup failures without changing player data:

```sh
export DS_GAME_DIR=/path/to/PPSA01341-app0
python3 scripts/linux-startup-test.py --shader-seed /path/to/suspect/shaders \
  --save-seed /path/to/save-seed
```

This runs ten hidden, silent, capped 30-second starts with timeouts and thermal supervision.
It leaves the game at its initial menu without scripted gameplay. Persistent private stores
alternate with fresh copies of the suspect store; the source seed stays untouched. Results
are under `local-data/linux-port/startup/`. A failure does not cancel the other starts unless
the thermal guard or launch fails. `--dry-run` prints the plan without starting the game.
Use `--seconds 60` for a colder store, and repeat `--expect-pipeline c<hex-key>` for previously
crashing keys to require successful creation, not just a quarantined skip.

The preparation-reuse and FE2-leaf experiments were removed after they showed no gameplay
benefit. The next experiment, `DS_PAR_WHOLE_RUN=1`, remains off by default; see the
[parallel recording design and A/B command](linux-parallel-recording.md).
