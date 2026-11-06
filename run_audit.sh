#!/bin/bash
# Script to launch Streamlit Audit App inside WSL/Linux
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

if [ -f "./redaction_env/bin/activate" ]; then
    source ./redaction_env/bin/activate
elif [ -f "$HOME/redaction_env/bin/activate" ]; then
    source "$HOME/redaction_env/bin/activate"
else
    echo "Virtual environment redaction_env not found!"
    exit 1
fi

echo "=========================================================="
echo " Launching Streamlit Rapid Audit UI..."
echo " Open in your browser: http://localhost:8501"
echo "=========================================================="

streamlit run streamlit_audit.py --server.port 8501 --server.headless true

