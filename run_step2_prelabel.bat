@echo off
setlocal
cd /d "%~dp0"
echo ==========================================================
echo  Step 2: Batched PyTorch TrOCR Draft Pre-Labeling
echo ==========================================================
wsl bash ./run_step2_prelabel.sh %*
pause
