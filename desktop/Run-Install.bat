@echo off
chcp 65001 >nul
title GROVEE Desktop — Install
cd /d "%~dp0"
powershell -NoProfile -ExecutionPolicy Bypass -File "%~dp0Install-GroveDesktop.ps1"
