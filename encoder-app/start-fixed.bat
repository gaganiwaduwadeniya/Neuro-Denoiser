@echo off
echo ===============================================
echo UltraHD Encoder - Fixed Startup Script
echo ===============================================
echo.

REM Set BROWSER environment variable to none to prevent auto-opening browser
set BROWSER=none

REM Check if Node.js is installed
where node >nul 2>nul
if %ERRORLEVEL% NEQ 0 (
    echo Error: Node.js is not installed or not in PATH.
    echo Please install Node.js and try again.
    goto :EXIT
)

REM Install NPM dependencies if needed
if not exist "server\node_modules" (
    echo Installing server dependencies...
    cd server
    call npm install
    cd ..
)

if not exist "client\node_modules" (
    echo Installing client dependencies...
    cd client
    call npm install
    cd ..
)

REM Start the server
echo.
echo Starting server on port 5000...
cd server
start cmd /k "title Encoder Server && npm start"
cd ..

REM Wait for server to initialize (longer wait)
echo Waiting for server to initialize...
timeout /t 8 /nobreak > nul

REM Start the client
echo.
echo Starting client on port 3000...
cd client
start cmd /k "title Encoder Client && npm start"
cd ..

echo.
echo ===============================================
echo Encoder application is starting up!
echo.
echo Server running at: http://localhost:5000
echo Client running at: http://localhost:3000
echo.
echo If the application doesn't work:
echo 1. Try closing both windows and run this script again
echo 2. Check if ports 3000 and 5000 are available
echo 3. Make sure you have all dependencies installed
echo ===============================================
echo.
echo Press any key to exit this window (the app will continue running)...
pause > nul

:EXIT 