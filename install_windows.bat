@echo off
setlocal
REM Double-click this file (Windows) to install everything Cell Analyzer needs.
REM   install_windows.bat --fresh   removes the old environment and builds it again.
cd /d "%~dp0"
set "ENVNAME=cell-analyzer"
set "CONDA_BAT="
for %%P in ("%USERPROFILE%\miniforge3" "%USERPROFILE%\mambaforge" "%USERPROFILE%\miniconda3" "%USERPROFILE%\anaconda3" "%LOCALAPPDATA%\miniforge3" "%LOCALAPPDATA%\miniconda3" "%LOCALAPPDATA%\anaconda3" "C:\ProgramData\miniforge3" "C:\ProgramData\miniconda3" "C:\ProgramData\anaconda3") do (
    if not defined CONDA_BAT if exist "%%~P\condabin\conda.bat" set "CONDA_BAT=%%~P\condabin\conda.bat"
)
if not defined CONDA_BAT (
    echo.
    echo   Conda was not found. Install Miniforge first ^(USER_GUIDE.html, section 3^), then run this again.
    pause
    exit /b 1
)
echo Using %CONDA_BAT%
for %%R in ("%CONDA_BAT%\..\..") do set "CONDA_ROOT=%%~fR"

REM Miniforge 26.9.4 bug: environments cannot be created ("Could not find share/conda-launchers/cli-64.exe").
REM Repair the base installation once by reinstalling the package 'conda-launchers'.
if exist "%CONDA_ROOT%\python.exe" (
    "%CONDA_ROOT%\python.exe" tools\check_conda_launchers.py "%CONDA_ROOT%"
    if errorlevel 1 (
        echo   repairing conda: reinstalling conda-launchers in the base installation...
        call "%CONDA_BAT%" install -n base -y --force-reinstall conda-launchers
    )
)
echo Installing / updating the 'cell-analyzer' environment. This takes 5-15 minutes the first time...

call :findpy
if defined PY if /i "%~1"=="--fresh" (
    echo   --fresh: removing the old environment first
    call "%CONDA_BAT%" env remove -y -n %ENVNAME%
    set "PY="
)
set "EXISTED=%PY%"
if defined PY (
    call "%CONDA_BAT%" env update -n %ENVNAME% -f environment.yml --prune
) else (
    call "%CONDA_BAT%" env create -y -f environment.yml || call :repair_and_create
)

echo   checking the environment...
call :findpy
if not defined PY goto :rebuild
call :activate
"%PY%" tools\check_env.py && goto :ok

:rebuild
if not defined EXISTED goto :failed
echo   The existing environment is damaged - rebuilding it from scratch ^(5-15 min^)...
call "%CONDA_BAT%" env remove -y -n %ENVNAME%
call "%CONDA_BAT%" env create -y -f environment.yml || call :repair_and_create
call :findpy
if not defined PY goto :failed
call :activate
"%PY%" tools\check_env.py && goto :ok

:failed
echo.
echo   Something went wrong - see the messages above ^(USER_GUIDE.html, Troubleshooting^).
pause
exit /b 1

:ok
echo.
echo   Done. You can now start the program with start_windows.bat
pause
exit /b 0

:repair_and_create
REM environment creation failed: repair conda-launchers (see above) unconditionally and try once more
echo   creating the environment failed - repairing conda and trying again...
call "%CONDA_BAT%" install -n base -y --force-reinstall conda-launchers
call "%CONDA_BAT%" env remove -y -n %ENVNAME% >nul 2>&1
call "%CONDA_BAT%" env create -y -f environment.yml
exit /b %errorlevel%

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
