@echo off
setlocal
cd /d "%~dp0"
echo ==========================================================
echo  Step 4: Fine-Tuning TrOCR on Verified Dataset
echo ==========================================================
wsl bash ./run_step4_train.sh %*
pause
