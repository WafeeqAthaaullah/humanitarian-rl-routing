#!/usr/bin/env bash
# Streamlit command center launcher
set -e
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$REPO_ROOT"
PYTHON_CMD="$(command -v python || command -v python3)"
echo "Starting Humanitarian RL Command Center at http://localhost:8501 ..."
"$PYTHON_CMD" -m streamlit run streamlit_app.py
