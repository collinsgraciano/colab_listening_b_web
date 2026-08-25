@echo off
cd /d "%~dp0"
echo Starting colab_listening_b Web UI on http://localhost:8765
echo Press Ctrl+C to stop.
python -m uvicorn app.main:app --host 127.0.0.1 --port 8765 --reload
pause
