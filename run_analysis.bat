@echo off
echo ============================================================
echo PYTHON REPORT GENERATOR - WINDOWS LAUNCHER
echo ============================================================
echo.

REM Check if Python is installed
python --version >nul 2>&1
if errorlevel 1 (
    echo ERROR: Python is not installed or not in PATH
    echo Please install Python 3.7+ from https://python.org
    pause
    exit /b 1
)

echo Python found! Checking dependencies...
echo.

REM Check if requirements are installed
python -c "import pandas, numpy, matplotlib, seaborn, reportlab, PIL" >nul 2>&1
if errorlevel 1 (
    echo Installing required packages...
    pip install -r requirements.txt
    if errorlevel 1 (
        echo ERROR: Failed to install requirements
        pause
        exit /b 1
    )
    echo.
)

echo All dependencies are ready!
echo.
echo Choose an option:
echo 1. Run sample data analysis (recommended for first time)
echo 2. Run with your own CSV file
echo 3. Exit
echo.

set /p choice="Enter your choice (1-3): "

if "%choice%"=="1" (
    echo.
    echo Running sample analysis...
    python quick_start.py
) else if "%choice%"=="2" (
    echo.
    echo Launching main application...
    python main.py
) else if "%choice%"=="3" (
    echo Goodbye!
    exit /b 0
) else (
    echo Invalid choice. Please run the script again.
)

echo.
pause
