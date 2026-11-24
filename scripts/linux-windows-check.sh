#!/usr/bin/env bash
# Check the Windows workspace with the isolated toolchain installed for the Linux port.
set -euo pipefail
repo_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
cd -- "$repo_dir"
export CARGO_HOME="$PWD/.cargo-home"
export RUSTUP_HOME="$PWD/local-data/linux-port/toolchains/rustup"
if [[ ! -x "$CARGO_HOME/bin/rustup" ]]; then
    printf 'Missing isolated rustup; see docs/linux.md (Windows cross-check).\n' >&2
    exit 2
fi
exec timeout -k 5s 600s "$CARGO_HOME/bin/rustup" run stable cargo check --workspace --target x86_64-pc-windows-gnu
