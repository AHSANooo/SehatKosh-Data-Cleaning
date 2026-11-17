@echo off
setlocal
cd /d "%~dp0"
echo ==========================================================
echo  Step 3: Launching Streamlit TrOCR Review UI
echo ==========================================================
for /f "delims=" %%i in ('wsl wslpath -u "%~dp0"') do set "WSL_DIR=%%i"
wsl bash -c "cd '%WSL_DIR%' && ./run_step3_review.sh %*"
pause
