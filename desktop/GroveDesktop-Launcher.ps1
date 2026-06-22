# GROVEE Desktop — starts local UI + search companion, opens browser.
param(
  [string]$InstallRoot = $PSScriptRoot
)

$ErrorActionPreference = "Stop"

$UiPort = 5180
$SearchPort = 7000
$AppDir = Join-Path $InstallRoot "app"
$SearchDir = Join-Path $InstallRoot "search"
$BinDir = Join-Path $InstallRoot "bin"
$Miniserve = Join-Path $BinDir "miniserve.exe"
$OpenSerp = Join-Path $SearchDir "openserp.exe"
$Config = Join-Path $SearchDir "config.yaml"
$LogDir = Join-Path $InstallRoot "logs"
$UiUrl = "http://127.0.0.1:${UiPort}/"
$HealthUrl = "http://127.0.0.1:${SearchPort}/health"

function Test-LocalHttp([string]$Url) {
  try {
    $r = Invoke-WebRequest -Uri $Url -UseBasicParsing -TimeoutSec 3
    return $r.StatusCode -ge 200 -and $r.StatusCode -lt 500
  } catch {
    return $false
  }
}

function Wait-LocalHttp([string]$Url, [int]$Seconds = 45) {
  $deadline = (Get-Date).AddSeconds($Seconds)
  while ((Get-Date) -lt $deadline) {
    if (Test-LocalHttp $Url) { return $true }
    Start-Sleep -Milliseconds 400
  }
  return $false
}

if (-not (Test-Path $AppDir)) {
  Write-Host "GROVEE app files missing: $AppDir" -ForegroundColor Red
  Write-Host "Run the installer again." -ForegroundColor Yellow
  Read-Host "Press Enter to exit"
  exit 1
}

if (-not (Test-Path $OpenSerp)) {
  Write-Host "OpenSERP not found. Run the GROVEE Desktop installer." -ForegroundColor Red
  Read-Host "Press Enter to exit"
  exit 1
}

New-Item -ItemType Directory -Force -Path $LogDir | Out-Null

Write-Host ""
Write-Host "  GROVEE Desktop" -ForegroundColor Cyan
Write-Host "  ==============" -ForegroundColor Cyan
Write-Host ""

# --- Search companion (OpenSERP) ---
if (-not (Test-LocalHttp $HealthUrl)) {
  Write-Host "  Starting search engine (port $SearchPort)..." -ForegroundColor Yellow
  $searchLog = Join-Path $LogDir "search.log"
  $searchArgs = "serve -a 127.0.0.1 -p $SearchPort --config `"$Config`""
  Start-Process -FilePath $OpenSerp -ArgumentList $searchArgs -WorkingDirectory $SearchDir `
    -WindowStyle Minimized -RedirectStandardOutput $searchLog -RedirectStandardError $searchLog
  if (-not (Wait-LocalHttp $HealthUrl)) {
    Write-Host "  Search engine did not start. See: $searchLog" -ForegroundColor Red
    Read-Host "Press Enter to exit"
    exit 1
  }
  Write-Host "  Search engine ready." -ForegroundColor Green
} else {
  Write-Host "  Search engine already running." -ForegroundColor Green
}

# --- Local UI static server ---
if (-not (Test-LocalHttp $UiUrl)) {
  if (-not (Test-Path $Miniserve)) {
    Write-Host "  miniserve.exe missing in $BinDir" -ForegroundColor Red
    Read-Host "Press Enter to exit"
    exit 1
  }
  Write-Host "  Starting GROVEE interface (port $UiPort)..." -ForegroundColor Yellow
  $uiLog = Join-Path $LogDir "ui.log"
  Start-Process -FilePath $Miniserve -ArgumentList "-p", $UiPort, "-i", "127.0.0.1", "-q", $AppDir `
    -WindowStyle Hidden -RedirectStandardOutput $uiLog -RedirectStandardError $uiLog
  if (-not (Wait-LocalHttp $UiUrl)) {
    Write-Host "  UI server did not start. See: $uiLog" -ForegroundColor Red
    Read-Host "Press Enter to exit"
    exit 1
  }
  Write-Host "  Interface ready." -ForegroundColor Green
} else {
  Write-Host "  Interface already running." -ForegroundColor Green
}

Write-Host ""
Write-Host "  Opening $UiUrl" -ForegroundColor Cyan
Write-Host "  Keep this window open while using GROVEE (close to stop)." -ForegroundColor DarkGray
Write-Host ""

Start-Process $UiUrl

# Hold launcher so child processes stay associated; user closes to exit.
try {
  while ($true) {
    Start-Sleep -Seconds 2
    if (-not (Test-LocalHttp $UiUrl) -and -not (Test-LocalHttp $HealthUrl)) {
      Write-Host "  Services stopped." -ForegroundColor Yellow
      break
    }
  }
} catch {
  # Ctrl+C
}

Write-Host "  Goodbye." -ForegroundColor DarkGray
