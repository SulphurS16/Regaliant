# Profile-guided optimization of dsport on top of the release-lto profile.
#   .\scripts\pgo.ps1 gen     instrumented build -> target\release-pgo-gen\dsport.exe; its runs
#                             (DSPORT_PROFILE=release-pgo-gen) write counters to local-data\pgo\raw
#                             when they shut down (`quit`)
#   .\scripts\pgo.ps1 merge   local-data\pgo\raw\*.profraw -> local-data\pgo\merged.profdata
#   .\scripts\pgo.ps1 use     optimized build -> target\release-pgo\dsport.exe
# llvm-profdata comes from the project-local copy of rustc's own llvm-tools component (it must match
# rustc's LLVM; the shared toolchain is not changed): local-data\tools\llvm-tools-<version>-...
# Builds go through --target so the flags reach dsport's crates but not build scripts.
# A profile names dsport's functions by their symbols, whose hashes cargo derives from the
# package's version (the hash of the build's fingerprint folder, dsport-<hash>): `merge` writes the
# instrumented build's into merged.profdata.unit, and `use` (like package.ps1 -Pgo) refuses a build
# whose hash differs, as it would go without PGO for all of dsport's own code. So train after the
# version change of a release, not before (0.1.2-0.1.4 and the first 0.1.6 package were built from
# an earlier version's profile that way, ~7% slower on the 1-1 walk: FINDINGS.md 2026-11-25).
# Note: do not use ErrorActionPreference=Stop here; cargo writes progress to stderr.
param([Parameter(Mandatory = $true)][ValidateSet('gen', 'merge', 'use')][string]$Step)
$root = Split-Path -Parent $PSScriptRoot
$pgo = Join-Path $root 'local-data\pgo'
$raw = Join-Path $pgo 'raw'
$merged = Join-Path $pgo 'merged.profdata'
$target = 'x86_64-pc-windows-msvc'
$cargo = Join-Path $PSScriptRoot 'cargo.ps1'
# Arguments separated by 0x1f: the paths contain a space.
function Set-Flags([string[]]$flags) { $env:CARGO_ENCODED_RUSTFLAGS = $flags -join [char]0x1f }
# The dsport program cargo built last with a profile: its fingerprint folder's name (dsport-<hash>,
# the folder holding bin-dsport), or $null.
function Get-Unit([string]$prof) {
    $fp = Join-Path $root "target\$target\$prof\.fingerprint"
    $u = Get-ChildItem $fp -Directory -Filter 'dsport-*' -ErrorAction SilentlyContinue |
        Where-Object { Test-Path (Join-Path $_.FullName 'bin-dsport') } |
        Sort-Object { (Get-Item (Join-Path $_.FullName 'bin-dsport')).LastWriteTime } -Descending |
        Select-Object -First 1
    if ($u) { $u.Name } else { $null }
}
function Copy-Build([string]$prof) {
    $from = Join-Path $root "target\$target\$prof"
    $to = Join-Path $root "target\$prof"
    New-Item -ItemType Directory -Force $to | Out-Null
    Copy-Item (Join-Path $from 'dsport.exe') $to -Force
    if (Test-Path (Join-Path $from 'dsport.pdb')) { Copy-Item (Join-Path $from 'dsport.pdb') $to -Force }
    "copied to target\$prof\dsport.exe"
}
switch ($Step) {
    'gen' {
        New-Item -ItemType Directory -Force $raw | Out-Null
        Set-Flags @("-Cprofile-generate=$raw", '--cfg', 'dsport_pgo_gen')
        & $cargo build --profile release-pgo-gen --target $target -p dsport
        if ($LASTEXITCODE -ne 0) { "build failed"; exit 1 }
        Copy-Build 'release-pgo-gen'
    }
    'merge' {
        $tool = Get-ChildItem (Join-Path $root 'local-data\tools') -Recurse -Filter 'llvm-profdata.exe' -ErrorAction SilentlyContinue | Select-Object -First 1
        if (-not $tool) { "llvm-profdata.exe not found under local-data\tools"; exit 2 }
        $files = @(Get-ChildItem $raw -Filter '*.profraw' -ErrorAction SilentlyContinue)
        if (-not $files.Count) { "no .profraw files in $raw"; exit 2 }
        $unit = Get-Unit 'release-pgo-gen'
        if (-not $unit) { "no instrumented build to name (pgo.ps1 gen first)"; exit 2 }
        & $tool.FullName merge -o $merged @($files | ForEach-Object { $_.FullName })
        if ($LASTEXITCODE -ne 0) { "merge failed"; exit 1 }
        Set-Content -Path "$merged.unit" -Value $unit -Encoding ascii
        "merged $($files.Count) file(s) into $merged ($((Get-Item $merged).Length) bytes), the functions of $unit"
    }
    'use' {
        if (-not (Test-Path $merged)) { "no $merged (run merge first)"; exit 2 }
        # (The instrumented build differs only in `pgo_write`; LLVM 22 names the option this way.)
        Set-Flags @("-Cprofile-use=$merged", '-Cllvm-args=-no-pgo-warn-mismatch')
        & $cargo build --profile release-pgo --target $target -p dsport
        if ($LASTEXITCODE -ne 0) { "build failed"; exit 1 }
        $want = Get-Content "$merged.unit" -ErrorAction SilentlyContinue | Select-Object -First 1
        $got = Get-Unit 'release-pgo'
        if (-not $want -or $want -ne $got) {
            "the profile names the functions of $(if ($want) { $want } else { 'an unknown build (no merged.profdata.unit)' }), this build is ${got}: it went without PGO for dsport's own code; train again (gen, tools/pgo_train.sh, merge)"
            exit 1
        }
        Copy-Build 'release-pgo'
    }
}
