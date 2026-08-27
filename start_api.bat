@echo off
REM Activate the virtual environment
call .venv\Scripts\activate.bat

REM Run the Python script with the specified arguments
python.exe -I api.py