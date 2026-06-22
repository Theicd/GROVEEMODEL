# Downloads OpenSERP into search\ if missing (used by Setup.exe post-install).
param(
  [Parameter(Mandatory = $true)]
  [string]$SearchDir
)

$ErrorActionPreference = "Stop"
$OpenSerpVersion = "0.8.3"
$ScriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
$OpenSerpExe = Join-Path $SearchDir "openserp.exe"
$ConfigPath = Join-Path $SearchDir "config.yaml"

New-Item -ItemType Directory -Force -Path $SearchDir | Out-Null
if (-not (Test-Path $ConfigPath)) {
  $cfgSrc = Join-Path $ScriptDir "config.yaml"
  if (Test-Path $cfgSrc) {
    Copy-Item -Force $cfgSrc $ConfigPath
  }
}

if (Test-Path $OpenSerpExe) { exit 0 }

$arch = if ([Environment]::Is64BitOperatingSystem) { "amd64" } else { "386" }
$tgzName = "openserp-windows-$arch-$OpenSerpVersion.tgz"
$tgzUrl = "https://github.com/karust/openserp/releases/download/v$OpenSerpVersion/$tgzName"
$tgzPath = Join-Path $env:TEMP $tgzName

Invoke-WebRequest -Uri $tgzUrl -OutFile $tgzPath -UseBasicParsing
$extractDir = Join-Path $env:TEMP "openserp-extract-$OpenSerpVersion"
if (Test-Path $extractDir) { Remove-Item -Recurse -Force $extractDir }
New-Item -ItemType Directory -Force -Path $extractDir | Out-Null
tar -xzf $tgzPath -C $extractDir
$found = Get-ChildItem -Path $extractDir -Filter "openserp.exe" -Recurse | Select-Object -First 1
if (-not $found) { throw "openserp.exe not found in archive" }
Copy-Item -Force $found.FullName $OpenSerpExe
