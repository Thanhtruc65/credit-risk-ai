@echo off
cd /d "%~dp0"
:loop
echo [%date% %time%] Starting FastAPI Project (Port 8000)...
python -m uvicorn app:app --host 0.0.0.0 --port 8000
echo [%date% %time%] Restarting in 3 seconds...
timeout /t 3
goto loop
