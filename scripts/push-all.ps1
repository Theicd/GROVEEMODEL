# One-shot push helper — pass -Token or set GITHUB_TOKEN (never commit a token in this file).
param([string]$Token = $env:GITHUB_TOKEN)
$ErrorActionPreference = "Stop"
Set-Location "c:\BRAIN\GROVEEMODEL-main"

if (-not $Token) { throw "Token required (-Token or GITHUB_TOKEN)" }
$token = $Token

$log = Join-Path $PWD "push-all-result.txt"
"" | Set-Content $log -Encoding utf8

function Log($msg) { Add-Content $log $msg -Encoding utf8; Write-Host $msg }

Log "=== git status (before) ==="
git status --short 2>&1 | ForEach-Object { Log $_ }

# Stage project files; skip temp/exit markers and huge optional trees
git add -A 2>&1 | ForEach-Object { Log $_ }
git reset HEAD -- "_npm_build.exit" "_npm_install.exit" "_npm_test.exit" 2>$null
git reset HEAD -- "push-all-result.txt" "scripts/push-all.ps1" 2>$null

Log ""
Log "=== staged ==="
git diff --cached --stat 2>&1 | ForEach-Object { Log $_ }

$status = git status --porcelain 2>&1
if ($status -match "^(A|M|D|R|C)") {
  git -c user.name="Theicd" -c user.email="dror201031@gmail.com" commit -m @"
Sync remaining GROVEE workspace files for vision lab deployment.

Adds public vision models and any missing app sources from local workspace.
"@ 2>&1 | ForEach-Object { Log $_ }
  Log "commit: $(git rev-parse HEAD)"
} else {
  Log "No changes to commit."
}

Log ""
Log "=== push ==="
$pushUrl = "https://x-access-token:${token}@github.com/Theicd/GROVEEMODEL.git"
git push $pushUrl main 2>&1 | ForEach-Object { Log $_ }
Log "push exit: $LASTEXITCODE"
Log "remote main: $(git ls-remote $pushUrl refs/heads/main)"

Log "DONE"
