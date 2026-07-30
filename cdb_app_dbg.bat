@echo off
REM Launch the DEBUG GUI app (RelWithDebInfo + PDBs) under cdb.  -G skips the
REM initial breakpoint so the app runs at ~full speed; the -cf script loads
REM symbols then `g`.  If the app crashes during human repro, cdb breaks and
REM the sxe filters auto-dump kv 60 + !analyze -v to crash.log, then quit.
REM The MSVC-built app + transcription_scheduler have PDBs (line numbers);
REM ggml-sycl.dll is icpx-built (no PDB — frames show as offsets).
REM
REM USAGE: run this bat, then in the app: rapidly switch model/device in
REM Settings, Record/Stop, close/reopen.  Watch the cdb console; on crash
REM read D:\xu_github\whisper_xpu\cdb_app_dbg.log.
set "VS2022INSTALLDIR=C:\Program Files (x86)\Microsoft Visual Studio\2022\BuildTools"
call "C:\Program Files (x86)\Intel\oneAPI\2025.3\oneapi-vars.bat" >nul 2>&1
set "ZES_ENABLE_SYSMAN=1"
set "DNNL_INTEL_TOOLSET=Intel C++ Compiler 2025"
set "CDB=C:\Program Files (x86)\Windows Kits\10\Debuggers\x64\cdb.exe"
set "LOG=D:\xu_github\whisper_xpu\cdb_app_dbg.log"
set "EXEDIR=D:\xu_github\whisper_xpu\build-dbg\whisper-xpu-app\RelWithDebInfo"
set "SCRIPT=D:\xu_github\whisper_xpu\cdb_app_dbg_cmd.txt"
pushd "%EXEDIR%"
echo === launching DEBUG app under cdb (PDBs; crash auto-dumps to %LOG%) ===
echo Repro: Settings rapid model/device switch, Record/Stop, close.
"%CDB%" -G -cf "%SCRIPT%" whisper_xpu_app.exe --model "D:\xu_github\whisper_xpu\models\ggml-tiny.bin" --device 0 > "%LOG%" 2>&1
echo EXIT_CDB=%ERRORLEVEL%
popd
