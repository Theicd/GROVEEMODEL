# Starts Grove Search Companion (OpenSERP) on http://127.0.0.1:7000

$ErrorActionPreference = "Stop"
$InstallRoot = Join-Path $env:LOCALAPPDATA "GROVEE\SearchCompanion"
$OpenSerpExe = Join-Path $InstallRoot "openserp.exe"
$ConfigPath = Join-Path $InstallRoot "config.yaml"
$LogPath = Join-Path $InstallRoot "companion.log"

if (-not (Test-Path $OpenSerpExe)) {
  Write-Host "OpenSERP not installed. Run Install-GroveSearchCompanion.ps1 first." -ForegroundColor Red
  Read-Host "Press Enter to exit"
  exit 1
}

try {
  $probe = Invoke-WebRequest -Uri "http://127.0.0.1:7000/health" -UseBasicParsing -TimeoutSec 2
  if ($probe.StatusCode -eq 200) {
    Write-Host "Grove Search already running on http://127.0.0.1:7000" -ForegroundColor Green
    Read-Host "Press Enter to close this window"
    exit 0
  }
} catch {
  # not running
}

Write-Host ""
Write-Host "  Grove Search Companion" -ForegroundColor Cyan
Write-Host "  http://127.0.0.1:7000" -ForegroundColor Cyan
Write-Host "  Log: $LogPath" -ForegroundColor DarkGray
Write-Host "  Close this window to stop the search engine." -ForegroundColor Yellow
Write-Host ""

Set-Location $InstallRoot
& $OpenSerpExe serve -a 127.0.0.1 -p 7000 --config $ConfigPath 2>&1 | Tee-Object -FilePath $LogPath
