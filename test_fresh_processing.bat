@echo off
REM Test fresh processing with your existing video

echo ========================================
echo Testing Fresh Video Processing
echo ========================================
echo.
echo This will:
echo 1. Clear all existing chunks/outputs
echo 2. Process input\video.mp4 from scratch  
echo 3. Create audio_chunks.json and video_chunks.json
echo 4. Embed all chunks into FAISS
echo.
echo Press Ctrl+C to cancel, or
pause
echo.

python process_fresh.py input\video.mp4

echo.
echo ========================================
echo Processing Complete!
echo ========================================
echo.
echo Next steps:
echo 1. Start backend:   python -m src.server
echo 2. Start frontend:  cd app ^&^& npm run dev
echo 3. Open browser:    http://localhost:3000
echo.
pause
