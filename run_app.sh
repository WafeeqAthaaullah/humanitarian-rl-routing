#!/usr/bin/env bash
# ==============================================================================
# Dubai Humanitarian RL Routing - Full Stack Web Application Launcher
# Starts FastAPI backend (port 8000) and Vite React frontend (port 5173)
# ==============================================================================

set -e

# Change to repository root directory
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$REPO_ROOT"

echo "==================================================================="
echo " Starting Dubai Humanitarian RL Routing Command Center"
echo "==================================================================="

# Check for python
PYTHON_CMD="python"
if ! command -v python &> /dev/null; then
    if command -v python3 &> /dev/null; then
        PYTHON_CMD="python3"
    else
        echo "Error: Python is not installed or not in PATH."
        exit 1
    fi
fi

# Ensure frontend production build exists or dev server runs
if [ ! -d "frontend/dist" ]; then
    echo "Building frontend static assets..."
    (cd frontend && npm install && npm run build)
fi

echo "Starting FastAPI Backend Server on http://127.0.0.1:8000 ..."
echo "API Docs available at: http://127.0.0.1:8000/docs"
echo "Unified Web App available at: http://127.0.0.1:8000"

# Trap CTRL+C to cleanly terminate child processes
trap 'kill $(jobs -p) 2>/dev/null || true; exit 0' SIGINT SIGTERM EXIT

# Start FastAPI backend in background
$PYTHON_CMD -m uvicorn backend.main:app --host 127.0.0.1 --port 8000 --reload &
BACKEND_PID=$!

# Optional: Start Vite React development server if frontend/node_modules exists
if [ -d "frontend/node_modules" ]; then
    echo "Starting React + Vite Dev Server on http://localhost:5173 ..."
    (cd frontend && npm run dev) &
    FRONTEND_PID=$!
fi

echo ""
echo "-------------------------------------------------------------------"
echo " Operational Command Center is LIVE:"
echo " - React Dev Server: http://localhost:5173"
echo " - Unified Production App: http://127.0.0.1:8000"
echo " - FastAPI Swagger UI: http://127.0.0.1:8000/docs"
echo " Press CTRL+C to stop both servers."
echo "-------------------------------------------------------------------"
echo ""

wait
