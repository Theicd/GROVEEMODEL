# Starts Grove Search Companion (OpenSERP) on http://127.0.0.1:7000

$ErrorActionPreference = "Stop"
$ScriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
$InstallRoot = Join-Path $env:LOCALAPPDATA "GROVEE\SearchCompanion"
$OpenSerpExe = Join-Path $InstallRoot "openserp.exe"
$ConfigPath = Join-Path $InstallRoot "config.yaml"
$LogPath = Join-Path $InstallRoot "companion.log"
$HealthUrl = "http://127.0.0.1:7000/health"

function Test-CompanionHealth {
  try {
    $probe = Invoke-WebRequest -Uri $HealthUrl -UseBasicParsing -TimeoutSec 2
    return ($probe.StatusCode -eq 200)
  } catch {
    return $false
  }
}

function Sync-CompanionFiles {
  New-Item -ItemType Directory -Force -Path $InstallRoot | Out-Null
  foreach ($name in @("config.yaml", "Start-GroveSearchCompanion.ps1", "Start-GroveSearch.bat")) {
    $src = Join-Path $ScriptDir $name
    if (Test-Path $src) {
      Copy-Item -Force $src (Join-Path $InstallRoot $name)
    }
  }
}

Sync-CompanionFiles

if (-not (Test-Path $OpenSerpExe)) {
  Write-Host "OpenSERP not found. Running installer..." -ForegroundColor Yellow
  $installer = Join-Path $ScriptDir "Install-GroveSearchCompanion.ps1"
  if (-not (Test-Path $installer)) {
    Write-Host "Missing Install-GroveSearchCompanion.ps1 next to this script." -ForegroundColor Red
    Read-Host "Press Enter to exit"
    exit 1
  }
  & $installer
  if (-not (Test-Path $OpenSerpExe)) {
    Write-Host "Install did not produce openserp.exe." -ForegroundColor Red
    Read-Host "Press Enter to exit"
    exit 1
  }
}

if (Test-CompanionHealth) {
  Write-Host "Grove Search already running at $HealthUrl" -ForegroundColor Green
  Read-Host "Press Enter to close"
  exit 0
}

Write-Host ""
Write-Host "  Grove Search Companion" -ForegroundColor Cyan
Write-Host "  $HealthUrl" -ForegroundColor Cyan
Write-Host "  Log: $LogPath" -ForegroundColor DarkGray
Write-Host ""

$serveArgs = "serve -a 127.0.0.1 -p 7000 --config `"$ConfigPath`""
$cmd = "Set-Location '$InstallRoot'; & '$OpenSerpExe' $serveArgs 2>&1 | Tee-Object -FilePath '$LogPath'"
Start-Process powershell.exe -ArgumentList @("-NoProfile", "-ExecutionPolicy", "Bypass", "-Command", $cmd) -WindowStyle Minimized

$ready = $false
for ($i = 0; $i -lt 20; $i++) {
  Start-Sleep -Milliseconds 500
  if (Test-CompanionHealth) {
    $ready = $true
    break
  }
}

if ($ready) {
  Write-Host "  Grove Search is running." -ForegroundColor Green
  Write-Host "  Open GROVEEMODEL Plugins - status should turn green." -ForegroundColor White
} else {
  Write-Host "  Process started but health check timed out." -ForegroundColor Yellow
  Write-Host "  Check log: $LogPath" -ForegroundColor DarkGray
}

Read-Host "Press Enter to close"
