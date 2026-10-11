@echo off
REM Launch the FastAPI service detached from any terminal/tool session.
cd /d "C:\Users\HP\Desktop\Embedded"
python -m uvicorn src.api.main:app --host 127.0.0.1 --port 8000 >> "%TEMP%\opencode\api.txt" 2>&1