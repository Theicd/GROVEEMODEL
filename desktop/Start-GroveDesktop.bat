@echo off
chcp 65001 >nul
title GROVEE Desktop
cd /d "%~dp0"
powershell -NoProfile -ExecutionPolicy Bypass -File "%~dp0GroveDesktop-Launcher.ps1" -InstallRoot "%~dp0"
