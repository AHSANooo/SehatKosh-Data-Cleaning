@echo off
setlocal
cd /d "%~dp0"
echo ==========================================================
echo  Step 4: Fine-Tuning TrOCR on Verified Dataset
echo ==========================================================
for /f "delims=" %%i in ('wsl wslpath -u "%~dp0"') do set "WSL_DIR=%%i"
wsl bash -c "cd '%WSL_DIR%' && ./run_step4_train.sh %*"
pause
