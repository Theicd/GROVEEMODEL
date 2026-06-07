param([string]$Token = $env:GITHUB_TOKEN)
$ErrorActionPreference = "Stop"
Set-Location "c:\BRAIN\GROVEEMODEL-main"

if (-not $Token) { throw "Token required" }

$log = Join-Path $PWD "push-grovee-ui-result.txt"
Set-Content -Path $log -Value "" -Encoding utf8
function Log([string]$msg) { Add-Content -Path $log -Value $msg -Encoding utf8; Write-Host $msg }

Log "=== GROVEE UI push ==="
Log (Get-Date -Format "yyyy-MM-dd HH:mm:ss")

# Drop ar_game from git tracking (keep local folder)
Log "Removing ar_game from git index..."
& git rm -r --cached -f ar_game 2>&1 | Out-Null
Log "ar_game untracked done"

$paths = @(
  ".github", ".gitignore", ".nojekyll", ".npmrc", ".nvmrc", "README.md",
  "eslint.config.js", "favicon.svg", "icons.svg", "index.html",
  "package.json", "package-lock.json", "qa-browser.mjs",
  "scripts/check-node.mjs", "start-local.bat", "tests",
  "tsconfig.app.json", "tsconfig.json", "tsconfig.node.json",
  "vite.config.ts", "vitest.config.ts", "assets", "public/models",
  "app/src"
)

foreach ($p in $paths) {
  if (Test-Path $p) { & git add $p 2>&1 | Out-Null }
}

# Unstage legacy image-gen if present
foreach ($legacy in @(
  "app/src/cloudImage.ts", "app/src/cloudImage.test.ts",
  "app/src/localImageGen.ts", "app/src/localImageGen.test.ts",
  "app/src/offlinePack.ts", "app/src/offlinePack.test.ts",
  "app/src/storageAudit.ts", "app/src/storageAudit.test.ts"
)) {
  & git reset HEAD -- $legacy 2>&1 | Out-Null
}

Log ""
Log "=== staged summary ==="
& git diff --cached --stat 2>&1 | ForEach-Object { Log $_ }

$staged = (& git diff --cached --name-only 2>&1) -join "`n"
if ($staged.Trim()) {
  $msg = "GROVEE camera UI: vision-lab, overlays, models; remove ar_game from repo"
  & git -c user.name=Theicd -c user.email=dror201031@gmail.com commit -m $msg 2>&1 | ForEach-Object { Log $_ }
  Log ("commit: " + (& git rev-parse HEAD))
} else {
  Log "No staged changes"
}

Log ""
Log "=== push ==="
$pushUrl = "https://x-access-token:${Token}@github.com/Theicd/GROVEEMODEL.git"
& git push $pushUrl main 2>&1 | ForEach-Object { Log $_ }
Log ("push exit: " + $LASTEXITCODE)
Log ("local HEAD: " + (& git rev-parse HEAD))
& git ls-remote $pushUrl refs/heads/main 2>&1 | ForEach-Object { Log ("remote: " + $_) }
Log "DONE"
