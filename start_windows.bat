@echo off
setlocal
REM Double-click this file (Windows) to start Cell Analyzer.
cd /d "%~dp0"
set "ENVNAME=cell-analyzer"
call :findpy
if not defined PY (
    REM fall back to the environment name used before the rename
    set "ENVNAME=cellpipeline"
    call :findpy
)
if not defined PY (
    echo.
    echo   The 'cell-analyzer' environment was not found - run install_windows.bat first.
    pause
    exit /b 1
)
call :activate
echo Starting Cell Analyzer with %PY%
echo (Keep this window open while you work - closing it closes the program.)
"%PY%" -m cell_analyzer
if errorlevel 1 pause
exit /b 0

:findpy
REM sets PY (python.exe) and ENVDIR of the 'cell-analyzer' environment (or %ENVNAME%), searching every usual conda installation
set "PY="
set "ENVDIR="
for %%P in ("%USERPROFILE%\miniforge3" "%USERPROFILE%\mambaforge" "%USERPROFILE%\miniconda3" "%USERPROFILE%\anaconda3" "%LOCALAPPDATA%\miniforge3" "%LOCALAPPDATA%\miniconda3" "%LOCALAPPDATA%\anaconda3" "C:\ProgramData\miniforge3" "C:\ProgramData\miniconda3" "C:\ProgramData\anaconda3") do (
    if not defined PY if exist "%%~P\envs\%ENVNAME%\python.exe" (
        set "PY=%%~P\envs\%ENVNAME%\python.exe"
        set "ENVDIR=%%~P\envs\%ENVNAME%"
    )
)
if not defined PY if exist "%USERPROFILE%\.conda\envs\%ENVNAME%\python.exe" (
    set "PY=%USERPROFILE%\.conda\envs\%ENVNAME%\python.exe"
    set "ENVDIR=%USERPROFILE%\.conda\envs\%ENVNAME%"
)
exit /b 0

:activate
REM Same effect as "conda activate": the environment's own DLL folders go FIRST on PATH, so Windows
REM does not load same-named DLLs installed by other programs (crash code 0xC06D007F / 3228369023).
set "PATH=%ENVDIR%;%ENVDIR%\Library\mingw-w64\bin;%ENVDIR%\Library\usr\bin;%ENVDIR%\Library\bin;%ENVDIR%\Scripts;%ENVDIR%\bin;%PATH%"
set "CONDA_PREFIX=%ENVDIR%"
set "CONDA_DEFAULT_ENV=%ENVNAME%"
set "CONDA_DLL_SEARCH_MODIFICATION_ENABLE=1"
set "PYTHONNOUSERSITE=1"
exit /b 0
