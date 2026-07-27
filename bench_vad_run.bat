@echo off
set "VS2022INSTALLDIR=C:\Program Files (x86)\Microsoft Visual Studio\2022\BuildTools"
call "C:\Program Files (x86)\Intel\oneAPI\setvars.bat" >nul 2>&1
set "DNNL_INTEL_TOOLSET=Intel C++ Compiler 2025"
set "PATH=C:\Users\Xuhan\.conda\envs\pytorch_xpu_win\Library\bin;%PATH%"
cd /d D:\xu_github\whisper_xpu
echo === BUILD bench_vad ===
cmake --build build-gpu --config Release --target bench_vad --parallel
if %ERRORLEVEL% neq 0 (echo BUILD_FAILED & exit /b %ERRORLEVEL%)
echo === COPY exe + fresh DLL to app dir (DLLs live there) ===
copy /Y "build-gpu\tests\bench_vad\Release\bench_vad.exe" "build-gpu\whisper-xpu-app\Release\bench_vad.exe" >nul
copy /Y "build-gpu\whisper-xpu-core\Release\whisper_xpu_sycl_core.dll" "build-gpu\whisper-xpu-app\Release\whisper_xpu_sycl_core.dll" >nul
echo === RUN bench_vad (CPU first, then GPU) ===
cd "build-gpu\whisper-xpu-app\Release"
bench_vad.exe --model "%1" 2> "D:\xu_github\whisper_xpu\bench_vad.log"
echo EXIT=%ERRORLEVEL%
