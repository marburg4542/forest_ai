@echo off
REM Stop the background forest_ai server started by forest_ai.bat.

cd /d "%~dp0"

where py >nul 2>&1 && (
    py launch.py --stop %*
) || (
    python launch.py --stop %*
)

REM brief pause so a double-clicked window can be read before it closes.
REM `timeout` refuses to run when stdin is redirected; ping always works.
ping -n 3 127.0.0.1 >nul 2>&1
