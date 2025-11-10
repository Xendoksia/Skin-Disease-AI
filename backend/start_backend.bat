@echo off
echo ================================
echo Skin Disease Analysis Backend
echo ================================
echo.

REM Check if virtual environment exists
if not exist "venv" (
    echo Creating virtual environment...
    python -m venv venv
    echo.
)

echo Activating virtual environment...
call venv\Scripts\activate
echo.

echo Installing/Updating requirements...
pip install -r requirements.txt
echo.

echo Starting Flask server...
echo Backend will be available at http://localhost:5000
echo Press Ctrl+C to stop the server
echo.
python app.py

pause
