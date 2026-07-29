@echo off
set "VS2022INSTALLDIR=C:\Program Files (x86)\Microsoft Visual Studio\2022\BuildTools"
call "C:\Program Files (x86)\Intel\oneAPI\2025.3\oneapi-vars.bat" >nul 2>&1
set "ZES_ENABLE_SYSMAN=1"
set "DNNL_INTEL_TOOLSET=Intel C++ Compiler 2025"
cd /d "D:\xu_github\whisper_xpu"
set "MODEL=models\ggml-tiny.bin"
set "DEV=--cpu"
if /i "%2"=="gpu" set "DEV=--device 0"
if not "%1"=="" set "MODEL=%1"
echo === test_robust: rapid_switch (%DEV%) ===
build-gpu\tests\streaming_pipeline\Release\test_robust.exe --case rapid_switch --model "%MODEL%" %DEV%
echo exit=%errorlevel%
echo === test_robust: record_before_load (%DEV%) ===
build-gpu\tests\streaming_pipeline\Release\test_robust.exe --case record_before_load --model "%MODEL%" %DEV%
echo exit=%errorlevel%
echo === test_robust: reload_while_recording (%DEV%) ===
build-gpu\tests\streaming_pipeline\Release\test_robust.exe --case reload_while_recording --model "%MODEL%" %DEV%
echo exit=%errorlevel%
