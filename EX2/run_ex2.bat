@echo off
REM Check if Python is installed
python --version >nul 2>&1
if %errorlevel% neq 0 (
    echo Python is not installed. Please install Python to run this script.
    pause
    exit /b
)

REM Define a function to check and install a Python package
:check_and_install
python -c "import %1" >nul 2>&1
if %errorlevel% neq 0 (
    echo Installing %1...
    python -m pip install %1
)
goto :eof

REM Check and install required libraries
call :check_and_install numpy
call :check_and_install pandas
call :check_and_install matplotlib

REM Run the Python script
python "Exercise2.py"
pause
