@echo off
echo Starting TechGyant Insights API...
echo.

REM Navigate to the API directory
cd /d "c:\Users\ADVANCED TECH\Documents\TechGyantInsights\api"

REM Check if Python is installed
python --version >nul 2>&1
if errorlevel 1 (
    echo ERROR: Python is not installed or not in PATH
    pause
    exit /b 1
)

echo Python found, installing dependencies...
pip install -r requirements.txt

echo.
echo Starting API server...
echo Open your browser to: http://localhost:8000/docs
echo Press Ctrl+C to stop the server
echo.

REM Try to run the simple version first
python main_simple.py

pause
