@echo off
REM Accessibility Vision Assistant - Quick Launcher for Windows
REM Double-click this file to run the application

echo ========================================
echo   Accessibility Vision Assistant
echo ========================================
echo.

REM Change to script directory
cd /d "%~dp0"

REM Check if virtual environment exists
if not exist "venv\Scripts\activate.bat" (
    echo ERROR: Virtual environment not found!
    echo Please run setup first:
    echo   python setup.py
    echo.
    pause
    exit /b 1
)

REM Activate virtual environment
echo Activating virtual environment...
call venv\Scripts\activate.bat

REM Check if activation worked
if errorlevel 1 (
    echo ERROR: Failed to activate virtual environment
    pause
    exit /b 1
)

REM Run the application
echo.
echo Starting Accessibility Vision Assistant...
echo.
python app.py

REM Keep window open if there's an error
if errorlevel 1 (
    echo.
    echo ERROR: Application exited with an error
    echo Check the messages above for details
    pause
)

deactivate
