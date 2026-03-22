@echo off
echo ========================================
echo  Multimodal RAG Course Platform
echo  Quick Start Script
echo ========================================
echo.

REM Check if virtual environment exists
if not exist "venv" (
    echo Creating virtual environment...
    python -m venv venv
)

echo Activating virtual environment...
call venv\Scripts\activate

echo.
echo Installing Python dependencies...
pip install -r requirements.txt

echo.
echo ========================================
echo  Setup Complete!
echo ========================================
echo.
echo Next steps:
echo 1. Ensure Ollama is running: ollama serve
echo 2. Start backend: python -m src.server
echo 3. In another terminal, start frontend:
echo    cd app
echo    npm install
echo    npm run dev
echo 4. Open http://localhost:3000
echo.
pause
