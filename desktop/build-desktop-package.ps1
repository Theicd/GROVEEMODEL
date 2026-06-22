# Build GROVEE Desktop installer package for Windows download button.
# Output: public/plugins/GroveDesktop-Setup-1.0.0.exe (if Inno Setup installed)
#         public/plugins/grove-desktop-win.zip (always)

$ErrorActionPreference = "Stop"

$DesktopDir = Split-Path -Parent $MyInvocation.MyCommand.Path
$RepoRoot = Split-Path -Parent $DesktopDir
$OutDir = Join-Path $RepoRoot "public\plugins"
$Staging = Join-Path $RepoRoot "desktop-staging"
$Version = "1.0.0"
$MiniserveVersion = "0.24.0"
$MiniserveUrl = "https://github.com/svenstaro/miniserve/releases/download/v$MiniserveVersion/miniserve-$MiniserveVersion-x86_64-pc-windows-msvc.exe"
$MiniserveCache = Join-Path $DesktopDir "bin\miniserve.exe"
$DocsDir = Join-Path $RepoRoot "docs"
$ZipPath = Join-Path $OutDir "grove-desktop-win.zip"
$SetupExe = Join-Path $OutDir "GroveDesktop-Setup-$Version.exe"

function Ensure-Miniserve {
  if (Test-Path $MiniserveCache) {
    Write-Host "[desktop] miniserve.exe cached." -ForegroundColor DarkGray
    return
  }
  Write-Host "[desktop] Downloading miniserve $MiniserveVersion..." -ForegroundColor Yellow
  New-Item -ItemType Directory -Force -Path (Split-Path $MiniserveCache) | Out-Null
  Invoke-WebRequest -Uri $MiniserveUrl -OutFile $MiniserveCache -UseBasicParsing
  Write-Host "[desktop] miniserve.exe ready." -ForegroundColor Green
}

function Ensure-AppBundle {
  if (-not (Test-Path (Join-Path $DocsDir "index.html"))) {
    Write-Host "[desktop] Building docs bundle (npm run build:pages-docs)..." -ForegroundColor Yellow
    Push-Location $RepoRoot
    try {
      npm run build:pages-docs
    } finally {
      Pop-Location
    }
  }
  if (-not (Test-Path (Join-Path $DocsDir "index.html"))) {
    throw "docs/index.html missing - run npm run build:pages-docs"
  }
}

function Build-Staging {
  if (Test-Path $Staging) { Remove-Item -Recurse -Force $Staging }
  New-Item -ItemType Directory -Force -Path $Staging | Out-Null

  $files = @(
    "config.yaml",
    "GroveDesktop-Launcher.ps1",
    "Install-GroveDesktop.ps1",
    "Install-OpenSerp.ps1",
    "Start-GroveDesktop.bat",
    "Run-Install.bat",
    "README-he.txt"
  )
  foreach ($f in $files) {
    Copy-Item -Force (Join-Path $DesktopDir $f) (Join-Path $Staging $f)
  }

  Write-Host "[desktop] Copying app UI ($DocsDir)..." -ForegroundColor Yellow
  Copy-Item -Recurse -Force $DocsDir (Join-Path $Staging "app")

  New-Item -ItemType Directory -Force -Path (Join-Path $Staging "bin") | Out-Null
  Copy-Item -Force $MiniserveCache (Join-Path $Staging "bin\miniserve.exe")

  # Post-install helper: download OpenSERP on first run if Setup skipped network step
  Copy-Item -Force (Join-Path $DesktopDir "Install-GroveDesktop.ps1") (Join-Path $Staging "Install-GroveDesktop.ps1")

  "$Version" | Set-Content -Path (Join-Path $Staging "version.txt") -Encoding ASCII
  Write-Host "[desktop] Staging ready: $Staging" -ForegroundColor Green
}

function Build-Zip {
  New-Item -ItemType Directory -Force -Path $OutDir | Out-Null
  if (Test-Path $ZipPath) { Remove-Item -Force $ZipPath }
  Compress-Archive -Path (Join-Path $Staging "*") -DestinationPath $ZipPath -CompressionLevel Optimal
  Write-Host "[desktop] ZIP: $ZipPath" -ForegroundColor Green
}

function Build-InnoSetup {
  $isccCandidates = @(
    "${env:ProgramFiles(x86)}\Inno Setup 6\ISCC.exe",
    "$env:ProgramFiles\Inno Setup 6\ISCC.exe",
    "${env:LocalAppData}\Programs\Inno Setup 6\ISCC.exe"
  )
  $iscc = $isccCandidates | Where-Object { Test-Path $_ } | Select-Object -First 1
  if (-not $iscc) {
    Write-Host "[desktop] Inno Setup not found - skipping .exe (ZIP still built)." -ForegroundColor Yellow
    Write-Host "          Install Inno Setup 6 to produce GroveDesktop-Setup.exe" -ForegroundColor DarkGray
    return $false
  }

  $iss = Join-Path $DesktopDir "grove-desktop.iss"
  Push-Location $DesktopDir
  try {
    & $iscc "/DSTAGING=$Staging" $iss
  } finally {
    Pop-Location
  }

  if (Test-Path $SetupExe) {
    Write-Host "[desktop] Setup EXE: $SetupExe" -ForegroundColor Green
    return $true
  }
  Write-Host "[desktop] ISCC ran but $SetupExe not found." -ForegroundColor Yellow
  return $false
}

Write-Host ""
Write-Host "  GROVEE Desktop package build" -ForegroundColor Cyan
Write-Host ""

Ensure-AppBundle
Ensure-Miniserve
Build-Staging
Build-Zip
$hasExe = Build-InnoSetup

Write-Host ""
if ($hasExe) {
  Write-Host "  Done. Publish: $SetupExe" -ForegroundColor Green
} else {
  Write-Host "  Done. Publish ZIP: $ZipPath" -ForegroundColor Green
  Write-Host '  (Install Inno Setup 6 to also build Setup.exe)' -ForegroundColor DarkGray
}
Write-Host ""
