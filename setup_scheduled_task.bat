@echo off
REM ============================================================
REM  setup_scheduled_task.bat
REM  Registers pipeline.py to run automatically every day at 8 AM
REM  using Windows Task Scheduler. Run this ONCE as Administrator.
REM ============================================================

set TASK_NAME=AIBusinessAnalyticsPipeline
set SCRIPT_DIR=%~dp0
set PYTHON_PATH=%SCRIPT_DIR%venv\Scripts\python.exe
set PIPELINE_SCRIPT=%SCRIPT_DIR%pipeline.py

echo Registering scheduled task: %TASK_NAME%
echo Will run: %PYTHON_PATH% %PIPELINE_SCRIPT%
echo Schedule: Daily at 08:00

schtasks /create ^
  /tn "%TASK_NAME%" ^
  /tr "\"%PYTHON_PATH%\" \"%PIPELINE_SCRIPT%\" --config \"%SCRIPT_DIR%pipeline_config.json\"" ^
  /sc daily ^
  /st 08:00 ^
  /f

if %ERRORLEVEL% EQU 0 (
    echo.
    echo SUCCESS: Task "%TASK_NAME%" registered.
    echo It will now run automatically every day at 8:00 AM, even if
    echo VS Code and Streamlit are closed.
    echo.
    echo To test it immediately without waiting: schtasks /run /tn "%TASK_NAME%"
    echo To remove it later:                     schtasks /delete /tn "%TASK_NAME%" /f
) else (
    echo.
    echo FAILED to register task. Make sure you ran this file as Administrator
    echo ^(right-click -^> "Run as administrator"^).
)

pause
