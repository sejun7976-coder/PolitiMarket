@echo off
setlocal EnableExtensions
cd /d "%~dp0"

echo Starting Crawler GUI Server...

set "WEB_DIR=%~dp0"
for %%I in ("%WEB_DIR%..") do set "PROJECT_DIR=%%~fI"
set "VENV_DIR=%PROJECT_DIR%\.venv"
set "VENV_PYTHON=%VENV_DIR%\Scripts\python.exe"
set "REQUIREMENTS_FILE=%PROJECT_DIR%\requirements.txt"

set "PYTHON_CMD="
set "PYTHON_ARGS="

if exist "%VENV_PYTHON%" (
    set "PYTHON_CMD=%VENV_PYTHON%"
    goto install_requirements
)

where py >nul 2>nul
if not errorlevel 1 (
    py -3.12 -c "import sys; raise SystemExit(0 if sys.version_info >= (3, 10) else 1)" >nul 2>nul
    if not errorlevel 1 (
        set "PYTHON_CMD=py"
        set "PYTHON_ARGS=-3.12"
        goto create_venv
    )

    py -3 -c "import sys; raise SystemExit(0 if sys.version_info >= (3, 10) else 1)" >nul 2>nul
    if not errorlevel 1 (
        set "PYTHON_CMD=py"
        set "PYTHON_ARGS=-3"
        goto create_venv
    )
)

where python >nul 2>nul
if not errorlevel 1 (
    python -c "import sys; raise SystemExit(0 if sys.version_info >= (3, 10) else 1)" >nul 2>nul
    if not errorlevel 1 (
        set "PYTHON_CMD=python"
        set "PYTHON_ARGS="
        goto create_venv
    )
)

echo Python 3.10 or newer was not found.
echo Install Python from https://www.python.org/downloads/windows/ and enable "Add python.exe to PATH".
pause
exit /b 1

:create_venv
echo Creating virtual environment in "%VENV_DIR%"...
"%PYTHON_CMD%" %PYTHON_ARGS% -m venv "%VENV_DIR%"
if errorlevel 1 (
    echo Failed to create the virtual environment.
    pause
    exit /b 1
)
set "PYTHON_CMD=%VENV_PYTHON%"
set "PYTHON_ARGS="

:install_requirements
if exist "%REQUIREMENTS_FILE%" (
    echo Installing/updating Python packages from "%REQUIREMENTS_FILE%"...
    "%PYTHON_CMD%" -m pip install --upgrade pip
    if errorlevel 1 (
        echo Failed to upgrade pip.
        pause
        exit /b 1
    )
    "%PYTHON_CMD%" -m pip install -r "%REQUIREMENTS_FILE%"
    if errorlevel 1 (
        echo Failed to install required Python packages.
        pause
        exit /b 1
    )
)

if "%FILTER_ALLOW_KEYWORD_SENTIMENT_FALLBACK%"=="" (
    set "FILTER_ALLOW_KEYWORD_SENTIMENT_FALLBACK=1"
)

:run
start "" http://127.0.0.1:8080/
"%PYTHON_CMD%" %PYTHON_ARGS% gui_server.py
pause
