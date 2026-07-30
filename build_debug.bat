@echo off
set "VS2022INSTALLDIR=C:\Program Files (x86)\Microsoft Visual Studio\2022\BuildTools"
call "C:\Program Files (x86)\Intel\oneAPI\2025.3\oneapi-vars.bat" >nul 2>&1
set "DNNL_INTEL_TOOLSET=Intel C++ Compiler 2025"
cd /d D:\xu_github\whisper_xpu
echo === [CONFIGURE] RelWithDebInfo ===
if not exist build-dbg cmake -B build-dbg -G "Visual Studio 17 2022" -A x64 -DCMAKE_CONFIGURATION_TYPES=RelWithDebInfo
echo EXIT_CONFIGURE=%ERRORLEVEL%
if errorlevel 1 ( echo ### CONFIGURE FAILED & exit /b 1 )
echo === [BUILD] whisper_xpu_app RelWithDebInfo ===
cmake --build build-dbg --config RelWithDebInfo --target whisper_xpu_app --parallel
echo EXIT_BUILD=%ERRORLEVEL%
