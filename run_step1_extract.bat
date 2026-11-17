@echo off
setlocal
cd /d "%~dp0"
echo ==========================================================
echo  Step 1: Automated Page Orientation Fix & Line Extraction
echo ==========================================================
for /f "delims=" %%i in ('wsl wslpath -u "%~dp0"') do set "WSL_DIR=%%i"
wsl bash -c "cd '%WSL_DIR%' && ./run_step1_extract.sh %*"
pause
