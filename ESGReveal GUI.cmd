@echo off
setlocal
pushd "%~dp0" >nul

set "ROOT=%CD%"
set "SETUP_CMD=%ROOT%\ESGReveal Setup.cmd"
set "PYTHONW_EXE=%ROOT%\.venv\Scripts\pythonw.exe"

if not exist "%PYTHONW_EXE%" (
    call "%SETUP_CMD%"
    if errorlevel 1 (
        popd >nul
        exit /b 1
    )
)

if not exist "%PYTHONW_EXE%" (
    echo [ESGReveal] GUI ?????pythonw.exe????? ????????
    popd >nul
    pause
    exit /b 1
)

start "" "%PYTHONW_EXE%" -c "from esgreveal.gui import launch_gui; raise SystemExit(launch_gui())"
popd >nul
exit /b 0
