@echo off
setlocal
cd /d "%~dp0"
echo ==========================================================
echo  Step 2: Batched PyTorch TrOCR Draft Pre-Labeling
echo ==========================================================
for /f "delims=" %%i in ('wsl wslpath -u "%~dp0"') do set "WSL_DIR=%%i"
wsl bash -c "cd '%WSL_DIR%' && ./run_step2_prelabel.sh %*"
pause
