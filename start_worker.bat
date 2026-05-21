@echo off
setlocal
cd /d "%~dp0"
echo Starting RagMemory worker...
uv run python scripts\run_worker.py
echo.
echo Worker stopped. Press any key to close this window.
pause >nul
