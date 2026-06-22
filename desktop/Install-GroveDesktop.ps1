# GROVEE Desktop — interactive installer (folder picker + OpenSERP download)

param(

  [string]$InstallRoot = "",

  [switch]$Silent,

  [switch]$SkipShortcut

)



$ErrorActionPreference = "Stop"



$DesktopVersion = "1.0.0"

$OpenSerpVersion = "0.8.3"

$ScriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path



function Write-Step([string]$Text) {

  Write-Host ""

  Write-Host "  $Text" -ForegroundColor Cyan

  Write-Host ""

}



function Pick-InstallFolder {

  $default = Join-Path $env:LOCALAPPDATA "GROVEE\Desktop"

  if ($InstallRoot) { return $InstallRoot.TrimEnd('\') }



  try {

    Add-Type -AssemblyName System.Windows.Forms | Out-Null

    [System.Windows.Forms.Application]::EnableVisualStyles()

    $dialog = New-Object System.Windows.Forms.FolderBrowserDialog

    $dialog.Description = "בחר תיקייה להתקנת GROVEE Desktop"

    $dialog.SelectedPath = $default

    $dialog.ShowNewFolderButton = $true

  if ($Silent) { return $default }

    $result = $dialog.ShowDialog()

    if ($result -eq [System.Windows.Forms.DialogResult]::OK -and $dialog.SelectedPath) {

      return $dialog.SelectedPath.TrimEnd('\')

    }

    Write-Host "  Installation cancelled." -ForegroundColor Yellow

    exit 0

  } catch {

    Write-Host "  Default install folder: $default" -ForegroundColor DarkGray

    if ($Silent) { return $default }

    $answer = Read-Host "  Press Enter to use default, or type another path"

    if ($answer.Trim()) { return $answer.Trim().TrimEnd('\') }

    return $default

  }

}



function Install-OpenSerp([string]$SearchDir) {
  $helper = Join-Path $ScriptDir "Install-OpenSerp.ps1"
  if (Test-Path $helper) {
    Write-Host "  Downloading OpenSERP..." -ForegroundColor Yellow
    & $helper -SearchDir $SearchDir
    Write-Host "  OpenSERP installed." -ForegroundColor Green
    return
  }
  $OpenSerpExe = Join-Path $SearchDir "openserp.exe"

  if (Test-Path $OpenSerpExe) {

    Write-Host "  OpenSERP already installed." -ForegroundColor Green

    return

  }



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

  if (-not $found) { throw "openserp.exe not found in archive" }

  Copy-Item -Force $found.FullName $OpenSerpExe

  Write-Host "  OpenSERP installed." -ForegroundColor Green

}



function New-DesktopShortcut([string]$Root) {

  $desktop = [Environment]::GetFolderPath("Desktop")

  $shortcutPath = Join-Path $desktop "GROVEE.lnk"

  $target = Join-Path $Root "Start-GroveDesktop.bat"

  $wsh = New-Object -ComObject WScript.Shell

  $sc = $wsh.CreateShortcut($shortcutPath)

  $sc.TargetPath = $target

  $sc.WorkingDirectory = $Root

  $sc.Description = "GROVEE Desktop — local AI interface"

  $sc.Save()

  Write-Host "  Desktop shortcut: GROVEE" -ForegroundColor Green

}



function Write-Uninstaller([string]$Root) {

  $uninstall = Join-Path $Root "Uninstall-GroveDesktop.ps1"

  @"

# GROVEE Desktop uninstaller

`$Root = "$Root"

Write-Host "Removing GROVEE from `$Root ..." -ForegroundColor Yellow

`$desktop = [Environment]::GetFolderPath("Desktop")

`$lnk = Join-Path `$desktop "GROVEE.lnk"

if (Test-Path `$lnk) { Remove-Item -Force `$lnk }

if (Test-Path `$Root) { Remove-Item -Recurse -Force `$Root }

Write-Host "Done." -ForegroundColor Green

Read-Host "Press Enter"

"@ | Set-Content -Path $uninstall -Encoding UTF8

}



Clear-Host

Write-Host ""

Write-Host "  GROVEE Desktop — Setup" -ForegroundColor Cyan

Write-Host "  ======================" -ForegroundColor Cyan

Write-Host "  Version $DesktopVersion" -ForegroundColor DarkGray



Write-Step "Step 1/4 — Welcome"

Write-Host @"

  This installs GROVEE on your PC:

  • Local interface in your browser (no Node.js)

  • Local search engine (Google / Bing / DuckDuckGo via OpenSERP)



  AI models download on first use from the internet (~1–4 GB).

  Search needs internet. Chat can work offline after models are cached.

"@



$targetRoot = Pick-InstallFolder

Write-Host "  Install to: $targetRoot" -ForegroundColor White



Write-Step "Step 2/4 — Copy files"

$appSrc = Join-Path $ScriptDir "app"

if (-not (Test-Path $appSrc)) {

  Write-Host "  ERROR: app\ folder missing next to installer." -ForegroundColor Red

  Write-Host "  Use GroveDesktop-Setup.exe from the official download." -ForegroundColor Yellow

  Read-Host "Press Enter to exit"

  exit 1

}



$searchDir = Join-Path $targetRoot "search"

$binDir = Join-Path $targetRoot "bin"

$logsDir = Join-Path $targetRoot "logs"



foreach ($d in @($targetRoot, $searchDir, $binDir, $logsDir)) {

  New-Item -ItemType Directory -Force -Path $d | Out-Null

}



Write-Host "  Copying interface files..." -ForegroundColor Yellow

$appDest = Join-Path $targetRoot "app"

if (Test-Path $appDest) { Remove-Item -Recurse -Force $appDest }

Copy-Item -Recurse -Force $appSrc $appDest



Copy-Item -Force (Join-Path $ScriptDir "config.yaml") (Join-Path $searchDir "config.yaml")

Copy-Item -Force (Join-Path $ScriptDir "GroveDesktop-Launcher.ps1") (Join-Path $targetRoot "GroveDesktop-Launcher.ps1")
Copy-Item -Force (Join-Path $ScriptDir "Install-OpenSerp.ps1") (Join-Path $targetRoot "Install-OpenSerp.ps1")
Copy-Item -Force (Join-Path $ScriptDir "Start-GroveDesktop.bat") (Join-Path $targetRoot "Start-GroveDesktop.bat")



$miniserveSrc = Join-Path $ScriptDir "bin\miniserve.exe"

if (Test-Path $miniserveSrc) {

  Copy-Item -Force $miniserveSrc (Join-Path $binDir "miniserve.exe")

} else {

  Write-Host "  WARNING: bin\miniserve.exe missing — UI server may fail." -ForegroundColor Red

}



"$DesktopVersion" | Set-Content -Path (Join-Path $targetRoot "version.txt") -Encoding ASCII

Write-Host "  Files copied." -ForegroundColor Green



Write-Step "Step 3/4 — Search engine"

Install-OpenSerp $searchDir



Write-Step "Step 4/4 — Shortcuts"

Write-Uninstaller $targetRoot

if (-not $SkipShortcut) {

  New-DesktopShortcut $targetRoot

}



Write-Host ""

Write-Host "  Installation complete!" -ForegroundColor Green

Write-Host ""

Write-Host "  Double-click the GROVEE icon on your Desktop to start." -ForegroundColor White

Write-Host "  Folder: $targetRoot" -ForegroundColor DarkGray

Write-Host ""



if (-not $Silent) {

  $launch = Read-Host "  Launch GROVEE now? (Y/n)"

  if ($launch -ne "n" -and $launch -ne "N") {

    Start-Process (Join-Path $targetRoot "Start-GroveDesktop.bat")

  }

}

