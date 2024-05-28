@echo off
REM Check if Python is installed
python --version >nul 2>&1
if %errorlevel% neq 0 (
    echo Python is not installed. Please install Python to run this script.
    pause
    exit /b
)

REM Check if tkinter can be imported
python -c "import tkinter" >nul 2>&1
if %errorlevel% neq 0 (
    echo Installing tkinter...
    REM Install tkinter using pip
    python -m pip install tk
)

REM Run the Python script
python "C:\Users\Gili Wolf\.vscode\Computational-bioligy\Exrecise1.py"
pause
