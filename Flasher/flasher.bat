@echo off
:: Check if running as admin
net session >nul 2>&1
if %errorLevel% == 0 (
    echo Running as Administrator
    start "" python flasher.py
) else (
    echo Requesting Administrator privileges...
    powershell -Command "Start-Process cmd -ArgumentList '/c cd /d %~dp0 && %~nx0' -Verb RunAs"
    exit
)