# Push ONLY GROVEE UI + camera vision. Never touches ar_game.
param([string]$Token = $env:GITHUB_TOKEN)
$ErrorActionPreference = "Stop"
Set-Location "c:\BRAIN\GROVEEMODEL-main"
if (-not $Token) { throw "Token required" }

$log = "push-grovee-result.txt"
function Log($m) { Add-Content $log $m; Write-Host $m }
Set-Content $log "=== GROVEE clean push ==="

# Drop bad local commit that bundled ar_game; keep your files on disk
Log "Reset to GitHub main (keeps local files, drops ar_game commit)..."
& git reset e99eac40befd5269eeb10d461b562154f57ef90d 2>&1 | ForEach-Object { Log $_ }

Log "Staging GROVEE files only..."
& git add .gitignore package.json package-lock.json vite.config.ts `
  tsconfig.json tsconfig.app.json tsconfig.node.json vitest.config.ts `
  eslint.config.js index.html favicon.svg icons.svg README.md `
  start-local.bat .npmrc .nvmrc .nojekyll qa-browser.mjs `
  .github scripts/check-node.mjs tests assets public/models app/src 2>&1 | Out-Null

# Never stage ar_game or legacy image-gen
& git reset HEAD ar_game 2>&1 | Out-Null
& git reset HEAD docs 2>&1 | Out-Null

Log "Staged files:"
& git diff --cached --name-only 2>&1 | ForEach-Object { Log "  + $_" }

$names = & git diff --cached --name-only 2>&1
if ($names) {
  & git -c user.name=Theicd -c user.email=dror201031@gmail.com `
    commit -m "GROVEE: vision-lab camera UI, hand overlay, YOLO models" 2>&1 | ForEach-Object { Log $_ }
}

Log "Pushing..."
$url = "https://x-access-token:${Token}@github.com/Theicd/GROVEEMODEL.git"
& git push $url main 2>&1 | ForEach-Object { Log $_ }
Log "HEAD: $(& git rev-parse HEAD)"
Log "DONE - see push-grovee-result.txt"
