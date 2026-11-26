# Builds the player package:
#   dist\Regaliant-<version>\     regaliant.exe, regaliant-launcher.exe, README.txt, upscalers\
#   dist\Regaliant-<version>.zip  (the same files, no top folder)
# (regaliant.exe is the runtime the build names dsport.exe.)
# README.txt is the player README (with the port's copyright and license notice), then the
# third-party notices of the two programs (THIRD_PARTY_NOTICES.txt, made by scripts\notices.py
# from what was just built), then the text of the port's license, the GNU GPL version 3.
# The programs are built with the dist profile (LTO, line tables in a PDB), a static C runtime,
# and the paths of the machine that builds them remapped; the package is refused if one of those
# paths, the user profile or the user name is still found in them. The PDBs stay here with the
# programs as built (dist\symbols\<package>-<build id>, the id the game's log names): never in
# the package.
# Only the files listed here go in (an allowlist): nothing of the game, no keys, saves, caches
# or logs, and not NVIDIA's nvngx_dlss.dll (players add their own copy).
#   .\scripts\package.ps1 [-NoBuild] [-Pgo <profile.profdata>] [-NoUpscalers] [-Upscalers <dir>]
# -Pgo: dsport.exe built with that profile (profile-guided optimization from scripts\pgo.ps1;
# the launcher without it).
# The "upscalers" folder: XeSS, NVIDIA Streamline and FSR as their makers ship them, with their
# license texts, from local-data\upscalers (scripts\upscalers.ps1); each DLL must carry a valid
# signature. -NoUpscalers leaves it out.
# Note: do not use ErrorActionPreference=Stop here; cargo writes progress to stderr.
param([switch]$NoBuild, [string]$Pgo, [switch]$NoUpscalers, [string]$Upscalers)
$root = Split-Path -Parent $PSScriptRoot
$target = 'x86_64-pc-windows-msvc'
$ver = (Select-String -Path (Join-Path $root 'Cargo.toml') -Pattern '^version = "(.+)"').Matches[0].Groups[1].Value
# The project-local Cargo cache when there is one (scripts\cargo.ps1), else the user's.
$cargoHome = Join-Path $root '.cargo-home'
if (Test-Path $cargoHome) { $env:CARGO_HOME = $cargoHome } else { $cargoHome = if ($env:CARGO_HOME) { $env:CARGO_HOME } else { Join-Path $env:USERPROFILE '.cargo' } }
$cargo = Join-Path $env:USERPROFILE '.cargo\bin\cargo.exe'
if (-not (Test-Path $cargo)) { $cargo = 'cargo' }
if (-not $NoBuild) {
  # (Encoded flags: the paths may hold spaces.)
  $flags = @('-C', 'target-feature=+crt-static', "--remap-path-prefix=$root=ds", "--remap-path-prefix=$cargoHome=cargo", "--remap-path-prefix=$($env:USERPROFILE)=home")
  Push-Location $root
  if ($Pgo) {
    if (-not (Test-Path $Pgo)) { Pop-Location; Write-Host "no profile at $Pgo"; exit 1 }
    $env:CARGO_ENCODED_RUSTFLAGS = ($flags + @("-Cprofile-use=$((Resolve-Path $Pgo).Path)")) -join [char]0x1f
    & $cargo build --profile dist --locked --target $target -p dsport 2>&1 | ForEach-Object { "$_" }
    $code = $LASTEXITCODE
    if ($code -eq 0) {
      # The profile must name this build's functions (scripts\pgo.ps1: their symbols' hashes come
      # from the version), else dsport's own code goes without PGO.
      $want = Get-Content "$((Resolve-Path $Pgo).Path).unit" -ErrorAction SilentlyContinue | Select-Object -First 1
      $got = Get-ChildItem (Join-Path $root "target\$target\dist\.fingerprint") -Directory -Filter 'dsport-*' -ErrorAction SilentlyContinue |
        Where-Object { Test-Path (Join-Path $_.FullName 'bin-dsport') } |
        Sort-Object { (Get-Item (Join-Path $_.FullName 'bin-dsport')).LastWriteTime } -Descending |
        Select-Object -First 1 | ForEach-Object { $_.Name }
      if (-not $want -or $want -ne $got) {
        Pop-Location
        Remove-Item Env:CARGO_ENCODED_RUSTFLAGS
        Write-Host "the profile names the functions of $(if ($want) { $want } else { 'an unknown build (no .unit file beside it)' }), this build is ${got}: it would go without PGO for dsport's own code (train again after the version change: scripts\pgo.ps1)"
        exit 1
      }
      Write-Host "PGO: the profile names this build's functions ($got)"
      $env:CARGO_ENCODED_RUSTFLAGS = $flags -join [char]0x1f
      & $cargo build --profile dist --locked --target $target -p launcher 2>&1 | ForEach-Object { "$_" }
      $code = $LASTEXITCODE
    }
  } else {
    $env:CARGO_ENCODED_RUSTFLAGS = $flags -join [char]0x1f
    & $cargo build --profile dist --locked --target $target -p dsport -p launcher 2>&1 | ForEach-Object { "$_" }
    $code = $LASTEXITCODE
  }
  Pop-Location
  Remove-Item Env:CARGO_ENCODED_RUSTFLAGS
  if ($code -ne 0) { Write-Host "build failed"; exit 1 }
  # The third-party notices of what was just built (THIRD_PARTY_NOTICES.txt).
  & python -I (Join-Path $PSScriptRoot 'notices.py')
  if ($LASTEXITCODE -ne 0) { Write-Host "notices failed"; exit 1 }
}
$bin = Join-Path $root "target\$target\dist"
$name = "Regaliant-$ver"
$dist = Join-Path $root 'dist'
$final = Join-Path $dist $name
# (Built in a fresh staging folder, then swapped in: a launcher still running from the previous
# package locks its file, and the zip must not take that one.)
$out = Join-Path $dist ".stage-$name"
if (Test-Path $out) { Remove-Item -Recurse -Force $out -ErrorAction Stop }
New-Item -ItemType Directory -Force $out | Out-Null
# Shipped name = built name.
$programs = [ordered]@{ 'regaliant.exe' = 'dsport.exe'; 'regaliant-launcher.exe' = 'regaliant-launcher.exe' }
foreach ($f in $programs.Keys) {
  $src = Join-Path $bin $programs[$f]
  if (-not (Test-Path $src)) { Write-Host "missing $src"; exit 1 }
  Copy-Item $src (Join-Path $out $f) -ErrorAction Stop
  if ((Get-FileHash $src).Hash -ne (Get-FileHash (Join-Path $out $f)).Hash) { Write-Host "$f was not copied whole"; exit 1 }
}
# The upscalers' libraries and their licenses (an allowlist of names). DLSS itself,
# nvngx_dlss.dll, and DLSS Frame Generation's nvngx_dlssg.dll are not among them: they are under
# NVIDIA's own license, so players add their own. Streamline's Frame Generation and Reflex plugins
# are (MIT; NvLowLatencyVk.dll under the Reflex SDK license beside it).
$upNames = @(
  'libxess.dll', 'libxess.license.txt', 'libxess.third-party-programs.txt',
  'sl.interposer.dll', 'sl.common.dll', 'sl.dlss.dll', 'streamline.license.txt', 'streamline.3rd-party-licenses.md',
  'sl.dlss_g.dll', 'sl.reflex.dll', 'sl.pcl.dll', 'NvLowLatencyVk.dll', 'reflex.license.txt',
  'amd_fidelityfx_loader_vk.dll', 'amd_fidelityfx_upscaler_vk.dll', 'fidelityfx.license.md'
)
if (-not $NoUpscalers) {
  $upSrc = if ($Upscalers) { $Upscalers } else { Join-Path $root 'local-data\upscalers' }
  $upOut = Join-Path $out 'upscalers'
  New-Item -ItemType Directory -Force $upOut | Out-Null
  foreach ($f in $upNames) {
    $src = Join-Path $upSrc $f
    if (-not (Test-Path $src)) { Write-Host "missing $src (run scripts\upscalers.ps1)"; exit 1 }
    Copy-Item $src $upOut -ErrorAction Stop
    if ($f.EndsWith('.dll') -and (Get-AuthenticodeSignature (Join-Path $upOut $f)).Status -ne 'Valid') { Write-Host "$f has no valid signature; not packaged"; exit 1 }
  }
}
# Paths and names of this machine must not be in the executables. The computer's name counts as
# a whole word only (a short one is found inside other text: PCV in the translator's "VopcVop3"),
# and outside machine code (a call's displacement once read "pcV", 2026-11-14): names come into a
# program as text, which its code sections never hold.
$needles = @($root, $root.Replace('\', '/'), $cargoHome, $env:USERPROFILE) | Where-Object { $_ }
if ($env:USERNAME -and $env:USERNAME.Length -ge 4) { $needles += "\$($env:USERNAME)\" }
$words = @($env:COMPUTERNAME) | Where-Object { $_ }
foreach ($f in $programs.Keys) {
  $bytes = [IO.File]::ReadAllBytes((Join-Path $out $f))
  $text = [Text.Encoding]::GetEncoding(28591).GetString($bytes).ToLowerInvariant()
  $wide = [Text.Encoding]::Unicode.GetString($bytes).ToLowerInvariant()
  foreach ($n in $needles) {
    $k = $n.ToLowerInvariant()
    if ($text.Contains($k) -or $wide.Contains($k)) { Write-Host "$f contains a path or name of this machine ($n); not packaged"; exit 1 }
  }
  # (The executable sections blanked for the name: the PE section table's executable flag. And a
  # match whose four bytes, read as an address, point into those sections is an address in a
  # table (the exception directory's function entries, unwind data), not text: 2026-11-22 a
  # function's start 0x566350 read "PcV" in .pdata and the unwind data of .rdata.)
  $data = [byte[]]$bytes.Clone()
  $pe = [BitConverter]::ToInt32($data, 0x3c)
  $nsec = [BitConverter]::ToUInt16($data, $pe + 6)
  $opt = [BitConverter]::ToUInt16($data, $pe + 20)
  $code = @()
  for ($i = 0; $i -lt $nsec; $i++) {
    $s = $pe + 24 + $opt + 40 * $i
    $size = [BitConverter]::ToInt32($data, $s + 16); $at = [BitConverter]::ToInt32($data, $s + 20)
    if (([BitConverter]::ToUInt32($data, $s + 36) -band 0x20000000) -and $at -lt $data.Length) {
      [Array]::Clear($data, $at, [Math]::Min($size, $data.Length - $at))
      $va = [BitConverter]::ToUInt32($data, $s + 12); $vs = [BitConverter]::ToUInt32($data, $s + 8)
      $code += , @($va, ($va + $vs))
    }
  }
  $isCodeAddress = { param([int]$o) if ($o + 4 -gt $data.Length) { return $false }; $v = [BitConverter]::ToUInt32($data, $o); foreach ($r in $code) { if ($v -ge $r[0] -and $v -lt $r[1]) { return $true } }; return $false }
  $text = [Text.Encoding]::GetEncoding(28591).GetString($data).ToLowerInvariant()
  $wide = [Text.Encoding]::Unicode.GetString($data).ToLowerInvariant()
  foreach ($n in $words) {
    $re = '(?<![a-z0-9])' + [regex]::Escape($n.ToLowerInvariant()) + '(?![a-z0-9])'
    $hits = @([regex]::Matches($text, $re) | Where-Object { -not (& $isCodeAddress $_.Index) })
    $hits += @([regex]::Matches($wide, $re) | Where-Object { -not (& $isCodeAddress (2 * $_.Index)) })
    if ($hits.Count) { Write-Host "$f contains the name of this machine ($n) at byte $(if ($hits[0].Index) { $hits[0].Index } else { 0 }); not packaged"; exit 1 }
    $skipped = @([regex]::Matches($text, $re)).Count
    if ($skipped) { Write-Host "${f}: $skipped matches of the machine's name are addresses into its code (tables), not text" }
  }
}
# README.txt: the player README, the notices, the license (CRLF, UTF-8).
$read = { param($p) ([IO.File]::ReadAllText($p)).Replace("`r`n", "`n").TrimEnd() }
$rule = '=' * 78
$license = Join-Path $root 'release\LICENSE'
if (-not (Test-Path $license)) { $license = Join-Path $root 'LICENSE' }
$readme = (& $read (Join-Path $root 'docs\PLAYER_README.txt')) + "`n`n`n$rule`nThird-party notices`n$rule`n`n" + (& $read (Join-Path $root 'THIRD_PARTY_NOTICES.txt')) + "`n`n`n$rule`nLicense: GNU General Public License, version 3`n$rule`n`n" + (& $read $license) + "`n"
[IO.File]::WriteAllText((Join-Path $out 'README.txt'), $readme.Replace("`n", "`r`n"), (New-Object Text.UTF8Encoding $true))
# Allowlist: nothing else may be in the package.
$allowed = @($programs.Keys) + @('README.txt') + @($upNames | ForEach-Object { "upscalers\$_" })
$extra = @(Get-ChildItem -Recurse -File $out | Where-Object { $allowed -notcontains $_.FullName.Substring($out.Length + 1) })
if ($extra.Count -gt 0) { Write-Host "unexpected files in the package: $($extra.FullName -join ', ')"; exit 1 }
# The symbols: the programs as built and their PDBs, beside the package (tools\symbolize.ps1 reads
# a player's log's stacks with them), named by the build id the game's log gives (its PDB's GUID
# and age, from the program's debug directory).
function Get-BuildId([string]$path) {
  $b = [IO.File]::ReadAllBytes($path)
  $pe = [BitConverter]::ToInt32($b, 0x3c)
  $nsec = [BitConverter]::ToUInt16($b, $pe + 6)
  $optSize = [BitConverter]::ToUInt16($b, $pe + 20)
  $opt = $pe + 24
  if ([BitConverter]::ToUInt16($b, $opt) -ne 0x20b) { return 'none' }
  $dirRva = [BitConverter]::ToUInt32($b, $opt + 112 + 48); $dirSize = [BitConverter]::ToUInt32($b, $opt + 112 + 52)
  $off = -1
  for ($i = 0; $i -lt $nsec; $i++) {
    $s = $opt + $optSize + 40 * $i
    $va = [BitConverter]::ToUInt32($b, $s + 12); $vs = [Math]::Max([BitConverter]::ToUInt32($b, $s + 8), [BitConverter]::ToUInt32($b, $s + 16))
    if ($dirRva -ge $va -and $dirRva -lt $va + $vs) { $off = $dirRva - $va + [BitConverter]::ToUInt32($b, $s + 20) }
  }
  if ($off -lt 0) { return 'none' }
  for ($e = 0; $e -lt [Math]::Floor($dirSize / 28); $e++) {
    $d = $off + 28 * $e
    if ([BitConverter]::ToUInt32($b, $d + 12) -ne 2) { continue }
    $cv = [BitConverter]::ToUInt32($b, $d + 24)
    if ([Text.Encoding]::ASCII.GetString($b, $cv, 4) -ne 'RSDS') { continue }
    $d4 = -join ($b[($cv + 12)..($cv + 19)] | ForEach-Object { $_.ToString('X2') })
    return ('{0:X8}{1:X4}{2:X4}{3}{4:X}' -f [BitConverter]::ToUInt32($b, $cv + 4), [BitConverter]::ToUInt16($b, $cv + 8), [BitConverter]::ToUInt16($b, $cv + 10), $d4, [BitConverter]::ToUInt32($b, $cv + 20))
  }
  return 'none'
}
$buildId = Get-BuildId (Join-Path $bin 'dsport.exe')
$symDir = Join-Path $dist "symbols\$name-$buildId"
New-Item -ItemType Directory -Force $symDir | Out-Null
foreach ($f in 'dsport.exe', 'dsport.pdb', 'regaliant-launcher.exe', 'regaliant_launcher.pdb') {
  $src = Join-Path $bin $f
  if (Test-Path $src) { Copy-Item $src $symDir -Force } else { Write-Host "no $f to keep in $symDir (a player's stacks then cannot be read)" }
}
$zip = Join-Path $dist "$name.zip"
if (Test-Path $zip) { Remove-Item -Force $zip }
Compress-Archive -Path (Join-Path $out '*') -DestinationPath $zip
$size = (Get-ChildItem -Recurse -File $out | Measure-Object Length -Sum).Sum
$count = (Get-ChildItem -Recurse -File $out).Count
# The staged files become dist\Regaliant-<version> unless the earlier folder is in use.
if (Test-Path $final) {
  if (Test-Path "$final.old") { Remove-Item -Recurse -Force "$final.old" -ErrorAction SilentlyContinue }
  try { Rename-Item $final "$name.old" -ErrorAction Stop } catch { }
}
if (-not (Test-Path $final)) {
  Move-Item $out $final -ErrorAction Stop
  if (Test-Path "$final.old") { Remove-Item -Recurse -Force "$final.old" -ErrorAction SilentlyContinue }
  $out = $final
} else {
  Write-Host "$final is in use (a program from it is running): the new package's files are in $out; the zip is complete"
}
$hash = (Get-FileHash $zip -Algorithm SHA256).Hash.ToLower()
Write-Host ("package {0}: {1} files, {2:N1} MB; zip {3:N1} MB, SHA-256 {4}; build {5}, symbols in {6}" -f $out, $count, ($size / 1MB), ((Get-Item $zip).Length / 1MB), $hash, $buildId, $symDir)
