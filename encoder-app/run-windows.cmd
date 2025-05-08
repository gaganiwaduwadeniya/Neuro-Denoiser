@echo off
echo Starting UltraHD Encoder Application...

echo Starting server...
start cmd /k "cd server && npm start"

echo Starting client...
start cmd /k "cd client && npm start"

echo.
echo Application started in separate windows.
echo Close the windows to stop the application.
pause 