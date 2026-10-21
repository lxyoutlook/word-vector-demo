@echo off
chcp 65001 >nul
setlocal
cd /d "%~dp0"
set PYTHONUTF8=1
if exist ".venv\Scripts\python.exe" (
    ".venv\Scripts\python.exe" -X utf8 bootstrap.py
    goto done
)
where py >nul 2>nul
if not errorlevel 1 (
    py -3.12 --version >nul 2>nul
    if not errorlevel 1 (
        py -3.12 -X utf8 bootstrap.py
    ) else (
        py -3 -X utf8 bootstrap.py
    )
) else (
    python -X utf8 bootstrap.py
)
:done
if errorlevel 1 pause
