#!/bin/bash
# Script to run batch de-identification pipeline inside WSL/Linux
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
echo " Starting SehatKosh Medical Document De-identification..."
echo " Running on hardware: GTX 1050 / Intel i5 (Auto device)"
echo "=========================================================="

python3 batch_redact.py "$@"

