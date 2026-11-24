#!/usr/bin/env bash
# Gentle relay: hidden measurement then one small visible run; silent, scripted input.
set -euo pipefail
repo_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
cd -- "$repo_dir"
export CARGO_HOME="$PWD/.cargo-home"
game_dir="${DS_GAME_DIR:-}"
mode=offscreen
run_seconds=180
hard_seconds=200
out_root="$repo_dir/local-data/linux-port/gpu-test"
if [[ ${1:-} == --present ]]; then
    mode=present
    out_root="$repo_dir/local-data/linux-port/present-test"
    shift
fi
dry_run=0
full=0
while [[ $# -gt 0 ]]; do
    case "$1" in
        --full) full=1; run_seconds=600; hard_seconds=660; shift ;;
        --dry-run) dry_run=1; shift ;;
        --app|--game-dir) [[ $# -ge 2 ]] || { printf '%s needs a path\n' "$1" >&2; exit 2; }; game_dir="$2"; shift 2 ;;
        --*) printf 'Unknown option: %s\n' "$1" >&2; exit 2 ;;
        *) [[ -z "$game_dir" ]] || { printf 'Give the game path once.\n' >&2; exit 2; }; game_dir="$1"; shift ;;
    esac
done
if [[ -z "$game_dir" ]]; then
    printf 'Set DS_GAME_DIR to your game folder or .zar archive, or pass --app /path/to/PPSA01341-app0.\n' >&2
    exit 2
fi
# Extra read-only trees are supplied by the caller; the game source is always protected.
export DS_GAME_DIR="$game_dir"
export DS_READONLY_ROOTS="${DS_READONLY_ROOTS:+$DS_READONLY_ROOTS:}$game_dir"
mode_args=()
if [[ $mode == offscreen ]]; then mode_args=(--offscreen); fi
if [[ $dry_run == 1 ]]; then
    printf 'Build (guarded, two Cargo jobs by default): CARGO_HOME=%q python3 scripts/linux_guard.py <run>/build-guard timeout -k 5s 600s cargo build --release -p dsport\n' "$CARGO_HOME"
    printf 'Run: timeout -k 5s %ss %q --app %q %s --relay-continue --run-seconds %s\n' "$hard_seconds" "$repo_dir/target/release/dsport" "$game_dir" "${mode_args[*]}" "$run_seconds"
    printf 'CPU: DS_RELAY_CPUS or 8 allowed logical CPUs selected by sysfs performance-core topology; nice 10. Compile workers: %s.\n' "${DS_COMPILE_WORKERS:-2}"
    printf 'Thermal guard: poll2s, pause at %s C sustained %ss, resume below %s C, abort at %s C sustained %ss; bounded cooldown. Paused timings INVALID.\n' "${DS_RELAY_TEMP_PAUSE:-103}" "${DS_RELAY_TEMP_HOLD_S:-10}" "${DS_RELAY_TEMP_RESUME:-97}" "${DS_RELAY_TEMP_ABORT:-105}" "${DS_RELAY_TEMP_ABORT_S:-60}"
    if [[ $full == 0 ]]; then
        printf 'SHORT suite: one clean offscreen measurement + one visible FIFO run, each 180s / timeout200s. Requires DS_RELAY_SAVE_SEED containing saves/; DS_RELAY_SHADER_SEED optional. No preparation, scatter, tracking stress or repeated profiles.\n'
    else
        printf 'FULL suite explicitly requested: guarded tracking/scatter/preparation and repeated profiles; cooldown before each. Non-performance offscreen workloads capped at30fps.\n'
    fi
    printf 'All run output: %s/run-<UTC>-<pid>/; latest summary: %s/summary.txt\n' "$out_root" "$out_root"
    exit 0
fi
if [[ $# != 0 ]]; then printf 'Usage: %s [--dry-run]\n' "$0" >&2; exit 2; fi
# Reject an output tree redirected outside this checkout before making any directories.
python3 - "$repo_dir" "$out_root" "$game_dir" <<'PY'
from pathlib import Path
import os, sys
repo, output, game = (Path(p).resolve() for p in sys.argv[1:])
if not game.exists(): raise SystemExit(f'Game source not found: {game}')
forbidden = [game] + [Path(p).resolve() for p in os.environ['DS_READONLY_ROOTS'].split(':') if p]
if game.is_dir():
    for directory, dirs, files in os.walk(game, followlinks=False):
        for name in dirs + files:
            p = Path(directory)/name
            if p.is_symlink(): forbidden.append(p.resolve())
# Check existing descendants too: a reused cache must not hide output symlinks.
outputs = [output]
if output.exists():
    for directory, dirs, files in os.walk(output, followlinks=False):
        outputs.extend((Path(directory)/n).resolve() for n in dirs + files if (Path(directory)/n).is_symlink())
for p in outputs:
    if not p.is_relative_to(repo) or any(p.is_relative_to(f) for f in forbidden):
        raise SystemExit(f'Unsafe relay output: {p}')
PY
if [[ $full == 0 && -z ${DS_RELAY_SAVE_SEED:-} ]]; then
    printf 'Short relay needs DS_RELAY_SAVE_SEED pointing to a prior relay user directory containing saves/. Use --full explicitly to create a new isolated save.\n' >&2
    exit 2
fi
run_dir="$out_root/run-$(date -u +%Y%m%dT%H%M%SZ)-$$"
mkdir -p -- "$run_dir"
# Do not inherit developer switches that redirect captures or change the tested renderer.
for name in ${!DS_@} ${!DSPORT_@}; do
    case "$name" in DS_GAME_DIR|DS_READONLY_ROOTS|DS_RELAY_*|DS_COMPILE_WORKERS|DS_SCATTER_FIXTURE) ;; *) unset "$name" ;; esac
done
export DS_RELAY_FULL="$full"
export DS_COMPILE_WORKERS="${DS_COMPILE_WORKERS:-2}"
export DS_RELAY_FPS_LIMIT=30
export DSPORT_HOME="$run_dir/user"
export XDG_CACHE_HOME="$run_dir/xdg-cache" XDG_DATA_HOME="$run_dir/xdg-data"
export XDG_CONFIG_HOME="$run_dir/xdg-config" XDG_STATE_HOME="$run_dir/xdg-state"
export TMPDIR="$run_dir/tmp" TMP="$run_dir/tmp" TEMP="$run_dir/tmp"
export MESA_SHADER_CACHE_DIR="$XDG_CACHE_HOME/mesa" __GL_SHADER_DISK_CACHE_PATH="$XDG_CACHE_HOME/nvidia"
export CUDA_CACHE_PATH="$XDG_CACHE_HOME/cuda"
export SDL_AUDIO_DRIVER=dummy DS_LINUX_HIDDEN=1 DS_GPU=1 DS_SECTIONS=1 DS_C4=0
export DS_LINUX_SHADER_CACHE="$out_root/shaders" DS_PRESENT_STATS=1
if [[ $mode == present ]]; then export DS_LINUX_HIDDEN=0 DS_LINUX_WINDOW_SIZE=800x450; fi
export DS_UPSCALER=off DS_VK_VALIDATION=1 DS_HEADLESS_PRESENT=0 RUST_BACKTRACE=full
export DS_SCREENSHOT=1,2,3,5,10,30,60,120,240,480,960,1920,3600
export DS_LINUX_FAULT_TIMING=0
# linux-relay.py pins each child profile; command.json records its actual switches.
# No GPU binaries are copied into source control; captures and cache stay in this run directory.
mkdir -p -- "$DSPORT_HOME" "$XDG_CACHE_HOME" "$XDG_DATA_HOME" "$XDG_CONFIG_HOME" "$XDG_STATE_HOME" "$TMPDIR"
mkdir -p -- "$MESA_SHADER_CACHE_DIR" "$__GL_SHADER_DISK_CACHE_PATH" "$CUDA_CACHE_PATH"
ulimit -c 0 # raw crash.log + stderr are collected; do not create system-wide game-data cores
{
    date -u
    uname -a
    if command -v lscpu >/dev/null; then lscpu; fi
    rustc --version
    cargo --version
    git status --short
    git diff --stat
    printf 'Wayland=%s DISPLAY=%s SDL_VIDEO_DRIVER=%s\n' "${WAYLAND_DISPLAY:-}" "${DISPLAY:-}" "${SDL_VIDEO_DRIVER:-auto}"
    printf 'command: timeout -k 5s %ss %q --app %q %s --relay-continue --run-seconds %s\n' "$hard_seconds" "$repo_dir/target/release/dsport" "$game_dir" "${mode_args[*]}" "$run_seconds"
    env | LC_ALL=C sort | sed -n '/^DS_/p; /^DSPORT_HOME=/p; /^XDG_/p; /^VK_/p; /^SDL_/p; /^CARGO_HOME=/p'
} > "$run_dir/environment.txt" 2>&1
printf '%s\n' "$run_dir" > "$out_root/latest.txt"
printf 'GPU relay in progress: %s\n' "$run_dir" > "$out_root/summary.txt"
if python3 scripts/linux_guard.py "$run_dir/build-guard" timeout -k 5s 600s cargo build --release -p dsport > "$run_dir/build.log" 2>&1; then
    :
else
    code=$?
    printf 'Build failed (exit %s). Inspect %s/build.log. GPU was not started.\n' "$code" "$run_dir" | tee "$out_root/summary.txt"
    exit "$code"
fi
# Optional standalone fixture from a developer's own capture (never shipped game data).
if [[ $full == 1 && $mode == offscreen && -n ${DS_SCATTER_FIXTURE:-} ]]; then
    cp -rL -- "$DS_SCATTER_FIXTURE" "$run_dir/scatter-fixture"
fi
if command -v vulkaninfo >/dev/null; then
    timeout -k 2s 20s vulkaninfo --summary > "$run_dir/vulkaninfo.txt" 2>&1 || true
fi
if command -v nvidia-smi >/dev/null; then
    timeout -k 2s 10s nvidia-smi --query-gpu=name,driver_version,pci.bus_id,memory.total --format=csv > "$run_dir/nvidia-smi.txt" 2>&1 || true
fi
# A separate session keeps all runtime children in an owned process group. Always reap and
# kill residual members, including on interruption. The runtime's own bound allows PNG writers
# to finish; GNU timeout is the independent hard limit if any thread/driver call hangs.
exec python3 "$repo_dir/scripts/linux-relay.py" "$repo_dir" "$game_dir" "$run_dir" "$out_root" "$mode" "$run_seconds" "$hard_seconds"
