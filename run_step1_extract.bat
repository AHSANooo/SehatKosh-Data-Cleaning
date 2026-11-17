@echo off
setlocal
cd /d "%~dp0"
echo ==========================================================
echo  Step 1: Automated Page Orientation Fix and Line Extraction
echo ==========================================================
wsl bash ./run_step1_extract.sh %*
pause
