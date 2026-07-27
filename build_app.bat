@echo off
set "VS2022INSTALLDIR=C:\Program Files (x86)\Microsoft Visual Studio\2022\BuildTools"
call "C:\Program Files (x86)\Intel\oneAPI\setvars.bat" >nul 2>&1
set "DNNL_INTEL_TOOLSET=Intel C++ Compiler 2025"
set "PATH=C:\Users\Xuhan\.conda\envs\pytorch_xpu_win\Library\bin;%PATH%"
cd /d D:\xu_github\whisper_xpu
cmake --build build-gpu --config Release --target whisper_xpu_app --parallel
