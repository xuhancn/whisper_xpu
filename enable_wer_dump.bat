@echo off
REM Configure Windows Error Reporting to auto-capture a FULL minidump when
REM whisper_xpu_app.exe crashes. NO debugger is attached, so this does NOT
REM trigger the Intel Level Zero driver debug-path bug. Dumps land in
REM D:\xu_github\whisper_xpu\wer_dumps . Analyze offline with:
REM   cdb -z D:\xu_github\whisper_xpu\wer_dumps\whisper_xpu_app.exe.<pid>.dmp
REM RUN THIS BAT AS ADMINISTRATOR (right-click, Run as administrator).

set "DUMPDIR=D:\xu_github\whisper_xpu\wer_dumps"
if not exist "%DUMPDIR%" mkdir "%DUMPDIR%"

set "KEY=HKLM\SOFTWARE\Microsoft\Windows\Windows Error Reporting\LocalDumps\whisper_xpu_app.exe"
reg add "%KEY%" /v DumpFolder /t REG_SZ /d "%DUMPDIR%" /f
reg add "%KEY%" /v DumpType /t REG_DWORD /d 2 /f
reg add "%KEY%" /v DumpCount /t REG_DWORD /d 10 /f

echo.
echo === WER LocalDumps configured for whisper_xpu_app.exe ===
echo Dumps will be written to: %DUMPDIR%
echo (10 dumps max, full type = 2)
echo.
reg query "%KEY%"
