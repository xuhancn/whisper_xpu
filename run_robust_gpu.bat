@echo off
set "VS2022INSTALLDIR=C:\Program Files (x86)\Microsoft Visual Studio\2022\BuildTools"
call "C:\Program Files (x86)\Intel\oneAPI\2025.3\oneapi-vars.bat" >nul 2>&1
set "ZES_ENABLE_SYSMAN=1"
set "DNNL_INTEL_TOOLSET=Intel C++ Compiler 2025"
cd /d "D:\xu_github\whisper_xpu"
echo === test_robust: rapid_switch GPU tiny dev0 ===
build-gpu\tests\streaming_pipeline\Release\test_robust.exe --case rapid_switch --model "models\ggml-tiny.bin" --device 0
echo exit=%errorlevel%
