# Fetches the upscalers' libraries for the player package into local-data\upscalers
# (git-ignored): Intel XeSS (libxess.dll), NVIDIA Streamline with DLSS
# (sl.interposer.dll, sl.common.dll, sl.dlss.dll; DLSS Frame Generation and
# Reflex: sl.dlss_g.dll, sl.reflex.dll, sl.pcl.dll, NvLowLatencyVk.dll;
# nvngx_dlss.dll and nvngx_dlssg.dll for development runs only - the package
# leaves them out, players add their own)
# and AMD FSR
# (amd_fidelityfx_loader_vk.dll, amd_fidelityfx_upscaler_vk.dll), each with
# its license texts, from the vendors' release archives on GitHub - pinned
# versions and SHA-256 hashes. The archives are kept in
# local-data\downloads\<name>\ and reused when present; every DLL must carry a
# valid Authenticode signature. The package script ships the folder as
# `upscalers` beside dsport.exe (gpu::upscale loads them from there).
#   .\scripts\upscalers.ps1 [-Offline]
param([switch]$Offline)
$root = Split-Path -Parent $PSScriptRoot
Add-Type -AssemblyName System.IO.Compression.FileSystem
[Net.ServicePointManager]::SecurityProtocol = [Net.SecurityProtocolType]::Tls12
$sdks = @(
  @{
    name = 'xess-3.0.2'
    url = 'https://github.com/intel/xess/releases/download/v3.0.2/XeSS_SDK_3.0.2.zip'
    sha = '88b8a373f30e33f3558a77a93e634f11b8132fc3047ea1a8edeead32b8471990'
    files = [ordered]@{
      'bin/libxess.dll' = 'libxess.dll'
      'LICENSE.txt' = 'libxess.license.txt'
      'third-party-programs.txt' = 'libxess.third-party-programs.txt'
    }
  },
  @{
    name = 'streamline-2.14.16'
    url = 'https://github.com/NVIDIA-RTX/Streamline/releases/download/v2.14.16/streamline-sdk-v2.14.16.zip'
    sha = '32041e51db0cab709a271b72a352281b690c2abbd669a827809401a5dd23956b'
    files = [ordered]@{
      'bin/x64/sl.interposer.dll' = 'sl.interposer.dll'
      'bin/x64/sl.common.dll' = 'sl.common.dll'
      'bin/x64/sl.dlss.dll' = 'sl.dlss.dll'
      'bin/x64/sl.dlss_g.dll' = 'sl.dlss_g.dll'
      'bin/x64/sl.reflex.dll' = 'sl.reflex.dll'
      'bin/x64/sl.pcl.dll' = 'sl.pcl.dll'
      'bin/x64/NvLowLatencyVk.dll' = 'NvLowLatencyVk.dll'
      'bin/x64/reflex.license.txt' = 'reflex.license.txt'
      'bin/x64/nvngx_dlss.dll' = 'nvngx_dlss.dll'
      'bin/x64/nvngx_dlssg.dll' = 'nvngx_dlssg.dll'
      'bin/x64/nvngx_dlss.license.txt' = 'nvngx_dlss.license.txt'
      'license.txt' = 'streamline.license.txt'
      '3rd-party-licenses.md' = 'streamline.3rd-party-licenses.md'
    }
  },
  @{
    name = 'fidelityfx-2.4.0'
    url = 'https://github.com/GPUOpen-LibrariesAndSDKs/FidelityFX-SDK/archive/refs/tags/v2.4.0.zip'
    file = 'FidelityFX-SDK-2.4.0.zip'
    sha = '27120e3bbe0d0b30213825ad40f19077563f04f04572e252109a392d5fb522cb'
    files = [ordered]@{
      'FidelityFX-SDK-2.4.0/Kits/FidelityFX/signedbin/amd_fidelityfx_loader_vk.dll' = 'amd_fidelityfx_loader_vk.dll'
      'FidelityFX-SDK-2.4.0/Kits/FidelityFX/signedbin/amd_fidelityfx_upscaler_vk.dll' = 'amd_fidelityfx_upscaler_vk.dll'
      'FidelityFX-SDK-2.4.0/Kits/FidelityFX/docs/license.md' = 'fidelityfx.license.md'
    }
  }
)
$out = Join-Path $root 'local-data\upscalers'
New-Item -ItemType Directory -Force $out | Out-Null
foreach ($s in $sdks) {
  $dir = Join-Path $root "local-data\downloads\$($s.name)"
  $zip = Join-Path $dir $(if ($s.file) { $s.file } else { [IO.Path]::GetFileName($s.url) })
  if (-not (Test-Path $zip)) {
    if ($Offline) { Write-Host "missing $zip (offline)"; exit 1 }
    New-Item -ItemType Directory -Force $dir | Out-Null
    Write-Host "downloading $($s.url)"
    Invoke-WebRequest -UseBasicParsing -Uri $s.url -OutFile $zip
    if (-not (Test-Path $zip)) { Write-Host "download failed: $($s.url)"; exit 1 }
  }
  $h = (Get-FileHash -Algorithm SHA256 $zip).Hash.ToLowerInvariant()
  if ($h -ne $s.sha) { Write-Host "${zip}: SHA-256 $h, expected $($s.sha)"; exit 1 }
  $za = [IO.Compression.ZipFile]::OpenRead($zip)
  try {
    foreach ($k in $s.files.Keys) {
      $e = $za.GetEntry($k)
      if (-not $e) { Write-Host "$zip has no $k"; exit 1 }
      $dst = Join-Path $out $s.files[$k]
      try {
        [IO.Compression.ZipFileExtensions]::ExtractToFile($e, $dst, $true)
      } catch {
        Write-Host "cannot write ${dst} (in use by a running dsport.exe?): $($_.Exception.Message)"
        exit 1
      }
      if ($dst.EndsWith('.dll')) {
        $sig = Get-AuthenticodeSignature $dst
        if ($sig.Status -ne 'Valid') { Write-Host "${dst}: signature $($sig.Status)"; exit 1 }
      }
    }
  } finally {
    $za.Dispose()
  }
}
Get-ChildItem -File $out | ForEach-Object { '{0,-36} {1,12:N0}' -f $_.Name, $_.Length }
