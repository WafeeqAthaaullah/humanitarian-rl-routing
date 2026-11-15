@echo off
REM ==============================================================================
REM Dubai Humanitarian RL Routing - Windows Command Center Launcher
REM Starts FastAPI backend (port 8000) and Vite React frontend (port 5173)
REM ==============================================================================

cd /d "%~dp0"

echo ===================================================================
echo  Starting Dubai Humanitarian RL Routing Command Center
echo ===================================================================

REM Check if frontend is built
if not exist "frontend\dist" (
    echo [Setup] Building frontend production bundle...
    cd frontend
    call npm install
    call npm run build
    cd ..
)

echo.
echo [1/2] Starting FastAPI Backend on http://127.0.0.1:8000 ...
start "Dubai RL Backend" cmd /k "python -m uvicorn backend.main:app --host 127.0.0.1 --port 8000 --reload"

echo.
echo [2/2] Starting React + Vite Dev Server on http://localhost:5173 ...
start "Dubai RL Frontend" cmd /k "cd frontend && npm run dev"

echo.
echo -------------------------------------------------------------------
echo  Command Center Launched!
echo  - Interactive Web App (Vite): http://localhost:5173
echo  - Unified App (FastAPI):     http://127.0.0.1:8000
echo  - Interactive API Docs:      http://127.0.0.1:8000/docs
echo -------------------------------------------------------------------
echo.
pause
