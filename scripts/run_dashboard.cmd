@echo off
REM Launch the Streamlit dashboard detached from any terminal/tool session.
cd /d "C:\Users\HP\Desktop\Embedded"
python -m streamlit run app.py --server.headless=true --server.port=8501 --server.address=127.0.0.1 >> "%TEMP%\opencode\st.txt" 2>&1