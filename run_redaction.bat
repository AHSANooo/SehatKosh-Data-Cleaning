@echo off
setlocal
REM Windows runner that delegates seamlessly to WSL2 redaction_env
cd /d "%~dp0"
echo ==========================================================
echo  Starting SehatKosh De-identification Pipeline via WSL2
echo ==========================================================
for /f "delims=" %%i in ('wsl wslpath -u "%~dp0"') do set "WSL_DIR=%%i"
wsl bash -c "cd '%WSL_DIR%' && ./run_redaction.sh %*"
pause
