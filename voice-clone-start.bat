@echo off
title IndicF5 Neo Voice Clone Server
cd /d "%~dp0"

echo Checking virtual environment...
if exist ".venv\Scripts\activate.bat" (
    echo Activating virtual environment...
    call .venv\Scripts\activate.bat
) else (
    echo [WARNING] .venv virtual environment not found in this folder.
    echo Running with system default Python...
)

echo Starting IndicF5 Neo server...
python launch.py

echo Server stopped.
pause
