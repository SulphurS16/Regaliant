#!/usr/bin/env bash
# Gentle relay in present-test/: one hidden measurement and one silent 800x450 FIFO run. --full opts into the extended suite.
set -euo pipefail
exec "$(dirname -- "${BASH_SOURCE[0]}")/linux-gpu-test.sh" --present "$@"
