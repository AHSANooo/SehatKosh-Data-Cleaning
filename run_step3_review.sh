#!/bin/bash
# Step 3: Streamlit TrOCR Review UI
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

if [ -f "./redaction_env/bin/activate" ]; then
    source ./redaction_env/bin/activate
elif [ -f "$HOME/redaction_env/bin/activate" ]; then
    source "$HOME/redaction_env/bin/activate"
fi

echo "Opening Streamlit Review App at http://localhost:8501 ..."
streamlit run streamlit_review.py --server.port 8501 --server.headless true
