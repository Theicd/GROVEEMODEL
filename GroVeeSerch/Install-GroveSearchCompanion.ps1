# Grove Search Companion - one-click install for Windows
# Installs OpenSERP to %LOCALAPPDATA%\GROVEE\SearchCompanion and creates a desktop shortcut.

$ErrorActionPreference = "Stop"
$OpenSerpVersion = "0.8.3"
$InstallRoot = Join-Path $env:LOCALAPPDATA "GROVEE\SearchCompanion"
$ScriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
$OpenSerpExe = Join-Path $InstallRoot "openserp.exe"
$ConfigPath = Join-Path $InstallRoot "config.yaml"

Write-Host ""
Write-Host "  Grove Search Companion - Install" -ForegroundColor Cyan
Write-Host "  ================================" -ForegroundColor Cyan
Write-Host ""

New-Item -ItemType Directory -Force -Path $InstallRoot | Out-Null

Copy-Item -Force (Join-Path $ScriptDir "config.yaml") $ConfigPath
Copy-Item -Force (Join-Path $ScriptDir "Start-GroveSearchCompanion.ps1") (Join-Path $InstallRoot "Start-GroveSearchCompanion.ps1")
Copy-Item -Force (Join-Path $ScriptDir "Start-GroveSearch.bat") (Join-Path $InstallRoot "Start-GroveSearch.bat")

if (-not (Test-Path $OpenSerpExe)) {
  $arch = if ([Environment]::Is64BitOperatingSystem) { "amd64" } else { "386" }
  $tgzName = "openserp-windows-$arch-$OpenSerpVersion.tgz"
  $tgzUrl = "https://github.com/karust/openserp/releases/download/v$OpenSerpVersion/$tgzName"
  $tgzPath = Join-Path $env:TEMP $tgzName

  Write-Host "  Downloading OpenSERP $OpenSerpVersion ($arch)..." -ForegroundColor Yellow
  Invoke-WebRequest -Uri $tgzUrl -OutFile $tgzPath -UseBasicParsing

  $extractDir = Join-Path $env:TEMP "openserp-extract-$OpenSerpVersion"
  if (Test-Path $extractDir) { Remove-Item -Recurse -Force $extractDir }
  New-Item -ItemType Directory -Force -Path $extractDir | Out-Null

  tar -xzf $tgzPath -C $extractDir

  $found = Get-ChildItem -Path $extractDir -Filter "openserp.exe" -Recurse | Select-Object -First 1
  if (-not $found) {
    throw "openserp.exe not found in archive"
  }
  Copy-Item -Force $found.FullName $OpenSerpExe
  Write-Host "  OpenSERP installed." -ForegroundColor Green
} else {
  Write-Host "  OpenSERP already present - skipping download." -ForegroundColor Green
}

$desktop = [Environment]::GetFolderPath("Desktop")
$shortcutPath = Join-Path $desktop "Grove Search.lnk"
$wsh = New-Object -ComObject WScript.Shell
$sc = $wsh.CreateShortcut($shortcutPath)
$sc.TargetPath = Join-Path $InstallRoot "Start-GroveSearch.bat"
$sc.WorkingDirectory = $InstallRoot
$sc.Description = "Grove Search Companion for GROVEEMODEL"
$sc.Save()

Write-Host ""
Write-Host "  Done!" -ForegroundColor Green
Write-Host "  1. Double-click Grove Search on your Desktop" -ForegroundColor White
Write-Host "  2. Open GROVEEMODEL Plugins panel - status should turn green" -ForegroundColor White
Write-Host "  Install folder: $InstallRoot" -ForegroundColor DarkGray
Write-Host ""
