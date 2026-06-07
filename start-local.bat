@echo off
setlocal EnableExtensions

cd /d "%~dp0"
title GROVEE local dev

echo.
echo  GROVEE - local dev server
echo  ==========================
echo.

where node >nul 2>&1
if errorlevel 1 (
  echo [ERROR] Node.js is not installed.
  echo         Download LTS: https://nodejs.org/
  pause
  exit /b 1
)

where npm >nul 2>&1
if errorlevel 1 (
  echo [ERROR] npm not found in PATH.
  pause
  exit /b 1
)

echo Node:
node -v
echo npm:
call npm -v
echo.

call node scripts\check-node.mjs
if errorlevel 1 (
  echo.
  echo Tip: install Node 18.18+ or 22 LTS from https://nodejs.org/
  echo      Or with nvm-windows:  nvm install 22   then   nvm use 22
  pause
  exit /b 1
)

rem Suppress engine warnings on Node 21 (eslint peer hints only)
set "npm_config_engine_strict=false"

if exist "node_modules\vite\package.json" (
  findstr /C:"\"version\": \"8." node_modules\vite\package.json >nul 2>&1
  if not errorlevel 1 (
    echo Old Vite 8 detected — reinstalling dependencies for Node 21...
    rmdir /s /q node_modules 2>nul
  )
)

if not exist "node_modules\nul" (
  echo Installing dependencies ^(first time only^)...
  echo.
  call npm install --no-fund
  if errorlevel 1 (
    echo.
    echo [ERROR] npm install failed.
    pause
    exit /b 1
  )
  echo.
) else (
  echo Dependencies OK.
  echo.
)

set "PORT=5173"
set "URL=http://127.0.0.1:%PORT%/"

echo Starting Vite at %URL%
echo Press Ctrl+C to stop.
echo.
echo First run downloads Gemma from Hugging Face ^(needs internet^).
echo.

call npm run dev -- --host 127.0.0.1 --port %PORT% --open

if errorlevel 1 (
  echo.
  echo [ERROR] Dev server failed to start.
  echo.
  echo If you see a Node version error:
  echo   - Need Node 18.18+ (your build uses Vite 5)
  echo   - Recommended: Node 22 LTS from https://nodejs.org/
  echo   - Or nvm: nvm install 22 ^& nvm use 22
  echo   - Then delete node_modules and run this script again.
  echo.
  pause
  exit /b 1
)

pause
