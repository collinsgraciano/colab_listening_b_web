@echo off
cd /d "%~dp0"
echo Starting colab_listening_b Web UI on http://localhost:59510
python -m uvicorn app.main:app --host 0.0.0.0 --port 59510 --reload
pause
