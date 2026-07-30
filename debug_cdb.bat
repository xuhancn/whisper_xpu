@echo off
set "VS2022INSTALLDIR=C:\Program Files (x86)\Microsoft Visual Studio\2022\BuildTools"
call "C:\Program Files (x86)\Intel\oneAPI\2025.3\oneapi-vars.bat" >nul 2>&1
set "ZES_ENABLE_SYSMAN=1"
set "DNNL_INTEL_TOOLSET=Intel C++ Compiler 2025"
set "CDB=C:\Program Files (x86)\Windows Kits\10\Debuggers\x64\cdb.exe"
set "LOG=D:\xu_github\whisper_xpu\crash.log"
cd /d "D:\xu_github\whisper_xpu\build-dbg\whisper-xpu-app\RelWithDebInfo"
echo === running under cdb; capturing first AV to %LOG% ===
"%CDB%" -g -G -c "sxe -c \"!analyze -v; kb; q\" av; sxe -c \"!analyze -v; kb; q\" c0000005; g" whisper_xpu_app.exe --model "D:\xu_github\whisper_xpu\models\ggml-large-v3-turbo-q8_0.bin" --device 0 > "%LOG%" 2>&1
echo EXIT_CDB=%ERRORLEVEL%
echo === crash log tail ===
type "%LOG%" | findstr /v "ModLoad: NatVis ReturnHr directxdatabase whisper_model_load: whisper_init_state whisper_backend_init_gpu" | more +0
