# Runs cargo with the registry/git caches isolated under <project>\.cargo-home and
# build output under <project>\target (the shared rustup toolchain in the user
# profile is used read-only).
# Note: do not use ErrorActionPreference=Stop here; cargo writes progress to stderr.
$root = Split-Path -Parent $PSScriptRoot
# (Waits while another checkout of this repository measures, when quiet.ps1 is there.)
$quiet = Join-Path $PSScriptRoot 'quiet.ps1'
if (Test-Path $quiet) { & $quiet -Wait }
$env:CARGO_HOME = Join-Path $root '.cargo-home'
$env:CARGO_TARGET_DIR = Join-Path $root 'target'
& "$env:USERPROFILE\.cargo\bin\cargo.exe" @args 2>&1 | ForEach-Object { "$_" }
exit $LASTEXITCODE
