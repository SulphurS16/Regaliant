#!/usr/bin/env bash
# Development bootstrap: keep runtime output in this checkout, outside the supplied game tree.
set -euo pipefail
repo_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
cd -- "$repo_dir"
export DSPORT_HOME="$repo_dir/local-data/linux-port/user"
export XDG_CACHE_HOME="$repo_dir/local-data/linux-port/xdg-cache"
export XDG_DATA_HOME="$repo_dir/local-data/linux-port/xdg-data"
export XDG_CONFIG_HOME="$repo_dir/local-data/linux-port/xdg-config"
export TMPDIR="$repo_dir/local-data/linux-port/tmp"
export MESA_SHADER_CACHE_DIR="$XDG_CACHE_HOME/mesa_shader_cache"
export __GL_SHADER_DISK_CACHE_PATH="$XDG_CACHE_HOME/nvidia"
mkdir -p -- "$DSPORT_HOME" "$XDG_CACHE_HOME" "$XDG_DATA_HOME" "$XDG_CONFIG_HOME" "$TMPDIR"
exec "$repo_dir/target/release/dsport" "$@"
