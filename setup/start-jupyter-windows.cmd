@echo off
setlocal
powershell.exe -NoLogo -NoProfile -ExecutionPolicy Bypass -File "%~dp0lib\common.ps1" -StartJupyter %*
exit /b %ERRORLEVEL%
