#!/bin/bash
# Step 4: Fine-Tuning TrOCR on Verified Dataset
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

if [ -f "./redaction_env/bin/activate" ]; then
    source ./redaction_env/bin/activate
elif [ -f "$HOME/redaction_env/bin/activate" ]; then
    source "$HOME/redaction_env/bin/activate"
fi

python3 train_trocr.py "$@"
