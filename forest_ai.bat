@echo off
REM Double-click this to use forest_ai: it starts the server if it is not
REM already up, then opens the interface. Running it twice is harmless.
REM
REM The console window you may see for a moment is this script; the server
REM itself runs windowless in the background. Stop it with:  stop.bat

cd /d "%~dp0"

where py >nul 2>&1 && (
    py launch.py %*
) || (
    python launch.py %*
)

REM only pause when something went wrong, so a successful launch just closes
if errorlevel 1 (
    echo.
    echo forest_ai could not start. See the message above.
    pause
)
