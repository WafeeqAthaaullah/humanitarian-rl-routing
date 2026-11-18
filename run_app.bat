@echo off
REM Streamlit command center launcher
cd /d "%~dp0"
echo Starting Humanitarian RL Command Center at http://localhost:8501 ...
python -m streamlit run streamlit_app.py
