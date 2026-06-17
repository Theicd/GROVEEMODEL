@echo off
setlocal EnableExtensions

cd /d "%~dp0"
title GROVEE local dev (port 5180)

echo %CD% | findstr /I /C:"GROVEEMODEL-main" >nul 2>&1
if not errorlevel 1 (
  echo.
  echo [ERROR] Wrong folder — GROVEEMODEL-main is deprecated.
  echo         Run: C:\Users\Avatar001\CascadeProjects\GROVEEMODEL\start-local.bat
  echo.
  pause
  exit /b 1
)

echo.
echo  GROVEE - local dev server
echo  ==========================
echo  URL: http://127.0.0.1:5180/
echo  Look for: HAL-5180  +  Load model button on intro screen
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
set "GROVEE_DEV_PORT=5180"

echo Freeing port %GROVEE_DEV_PORT% if a previous dev server is still running...
call node scripts\kill-grovee-main-dev.mjs
call node scripts\kill-grovee-dev-port.mjs
if errorlevel 1 (
  echo.
  echo [WARN] Could not fully free port %GROVEE_DEV_PORT%. Close other GROVEEMODEL windows and try again.
  echo.
)
echo.

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

set "URL=http://127.0.0.1:5180/"

echo Starting at %URL%
echo Press Ctrl+C to stop.
echo.
echo First run downloads Gemma from Hugging Face ^(needs internet^).
echo.

rem Port/host/strictPort are set in package.json — do NOT override here
call npm run dev -- --open

if errorlevel 1 (
  echo.
  echo [ERROR] Dev server failed to start.
  echo.
  echo If port 5180 is busy, close the other GROVEEMODEL window and try again.
  echo If you see a Node version error:
  echo   - Need Node 18.18+ ^(your build uses Vite 5^)
  echo   - Recommended: Node 22 LTS from https://nodejs.org/
  echo   - Or nvm: nvm install 22 ^& nvm use 22
  echo   - Then delete node_modules and run this script again.
  echo.
  pause
  exit /b 1
)

pause
