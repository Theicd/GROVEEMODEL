@echo off
chcp 65001 >nul
title Grove Search — Running on http://127.0.0.1:7000
powershell -NoProfile -ExecutionPolicy Bypass -File "%~dp0Start-GroveSearchCompanion.ps1"
