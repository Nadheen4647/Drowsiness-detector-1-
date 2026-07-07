@echo off
title AI-Drive Eyeguard Starter
echo ===================================================
echo   AI-DRIVE EYEGUARD LOCAL WEB SERVER STARTER
echo ===================================================
echo.

:: Check if a server is already running on port 8000
netstat -ano | findstr :8000 >nul
if %errorlevel% equ 0 (
    echo [SYSTEM] Web server is already running on port 8000.
) else (
    echo [SYSTEM] Launching Python HTTP server on port 8000...
    start /min "AI Driver Server" python -m http.server 8000
    :: Give the server a moment to bind to the port
    timeout /t 1 /nobreak >nul
)

echo [SYSTEM] Opening dashboard in your default browser...
start http://localhost:8000/project.html
echo.
echo [SYSTEM] Initialization complete. You can close this window.
timeout /t 3 >nul
