@echo off
set "VS2022INSTALLDIR=C:\Program Files (x86)\Microsoft Visual Studio\2022\BuildTools"
call "C:\Program Files (x86)\Intel\oneAPI\setvars.bat"
set "PATH=C:\Users\Xuhan\.conda\envs\pytorch_xpu_win\Library\bin;C:\Users\Xuhan\.conda\envs\pytorch_xpu_win\Scripts;%PATH%"
cd /d D:\xu_github\whisper_xpu

echo ========== CMake Configure ==========
cmake -S . -B build-sycl -DCMAKE_BUILD_TYPE=Release
if %ERRORLEVEL% neq 0 (
    echo CMake configure FAILED
    pause
    exit /b %ERRORLEVEL%
)

echo ========== CMake Build (parallel) ==========
set "CMAKE_BUILD_PARALLEL_LEVEL=%NUMBER_OF_PROCESSORS%"
cmake --build build-sycl --config Release --parallel
if %ERRORLEVEL% neq 0 (
    echo Build FAILED
    pause
    exit /b %ERRORLEVEL%
)

echo ========== Build SUCCESS ==========
echo oneDNN linkage:
dumpbin /dependents build-sycl\bin\Release\ggml-sycl.dll 2>nul | findstr /i dnnl
echo.
echo App EXE:
dir build-sycl\whisper-xpu-app\Release\whisper_xpu_app.exe
pause
