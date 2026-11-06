@echo off
setlocal
REM Windows runner that launches the Streamlit audit interface
cd /d "%~dp0"
echo ==========================================================
echo  Launching Streamlit Human-in-the-Loop Audit UI
echo ==========================================================
for /f "delims=" %%i in ('wsl wslpath -u "%~dp0"') do set "WSL_DIR=%%i"
wsl bash -c "cd '%WSL_DIR%' && ./run_audit.sh"
pause
