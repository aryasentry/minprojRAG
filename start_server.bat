@echo off
REM Start backend server with venv

echo ========================================
echo Starting Backend Server
echo ========================================
echo.

REM Add ffmpeg to PATH
set PATH=C:\Users\kalid\Downloads\miniprojRAG\ffmpeg\ffmpeg-8.0.1-essentials_build\bin;%PATH%

REM Start server
.venv\Scripts\python.exe -m src.server
