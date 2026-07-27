@echo off
call "C:\Program Files (x86)\Intel\oneAPI\setvars.bat" >nul 2>&1
cd /d D:\xu_github\whisper_xpu
build-gpu\whisper-xpu-app\Release\whisper_xpu_app.exe --cpu --model models\ggml-large-v3-turbo-q8_0.bin
