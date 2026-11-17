@echo off
setlocal
cd /d "%~dp0"
echo ==========================================================
echo  Step 3: Launching Streamlit TrOCR Review UI
echo ==========================================================
wsl bash ./run_step3_review.sh %*
pause
