# Build grove-search-companion-win.zip into public/plugins/ for GROVEEMODEL download button.

$ErrorActionPreference = "Stop"
$Root = Split-Path -Parent $MyInvocation.MyCommand.Path
$OutDir = Join-Path (Split-Path -Parent $Root) "public\plugins"
$ZipPath = Join-Path $OutDir "grove-search-companion-win.zip"

New-Item -ItemType Directory -Force -Path $OutDir | Out-Null
if (Test-Path $ZipPath) { Remove-Item -Force $ZipPath }

$files = @(
  "config.yaml",
  "Install-GroveSearchCompanion.ps1",
  "Start-GroveSearchCompanion.ps1",
  "Run-Install.bat",
  "Start-GroveSearch.bat",
  "README-he.txt"
)

$staging = Join-Path $env:TEMP "grove-companion-staging"
if (Test-Path $staging) { Remove-Item -Recurse -Force $staging }
New-Item -ItemType Directory -Force -Path $staging | Out-Null

foreach ($f in $files) {
  Copy-Item (Join-Path $Root $f) (Join-Path $staging $f)
}

Compress-Archive -Path (Join-Path $staging "*") -DestinationPath $ZipPath -Force
Remove-Item -Recurse -Force $staging

Write-Host "Built: $ZipPath"
