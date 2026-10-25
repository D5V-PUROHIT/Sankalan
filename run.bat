@echo off
title Sankalan AI Platform
echo ===================================================
echo             Starting Sankalan Platform             
echo ===================================================
echo.
cd /d "%~dp0backend"
echo Launching server at http://127.0.0.1:8000 ...
py -3.11 app.py
if %ERRORLEVEL% NEQ 0 (
    echo.
    echo Fallback: Trying with standard python command...
    python app.py
)
pause
