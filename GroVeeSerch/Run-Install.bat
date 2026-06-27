@echo off
chcp 65001 >nul
title Grove Search Companion — Install
powershell -NoProfile -ExecutionPolicy Bypass -File "%~dp0Install-GroveSearchCompanion.ps1"
pause
