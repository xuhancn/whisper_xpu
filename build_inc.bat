@echo off
set "VS2022INSTALLDIR=C:\Program Files (x86)\Microsoft Visual Studio\2022\BuildTools"
call "C:\Program Files (x86)\Intel\oneAPI\2025.3\oneapi-vars.bat" >nul 2>&1
set "DNNL_INTEL_TOOLSET=Intel C++ Compiler 2025"
cd /d D:\xu_github\whisper_xpu
echo === incremental build: test_pipeline + whisper_xpu_app ===
cmake --build build-gpu --config Release --target test_pipeline whisper_xpu_app --parallel
echo EXIT_BUILD=%ERRORLEVEL%
