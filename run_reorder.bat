@echo off
REM 1-Click runner to reorder redacted documents in ascending order
cd /d "%~dp0"
echo ==========================================================
echo  Reordering Redacted Documents (1 to N)
echo ==========================================================
python reorder_documents.py %*
pause

