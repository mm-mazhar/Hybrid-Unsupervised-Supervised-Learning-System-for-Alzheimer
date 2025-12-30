@echo off
setlocal enabledelayedexpansion

:: Check if requirements.txt exists
if not exist requirements.txt (
    echo Error: requirements.txt not found.
    exit /b 1
)

:: Initialize dependencies variable
set deps=

:: Read requirements.txt line by line, ignoring comments
for /f "usebackq tokens=*" %%i in ("requirements.txt") do (
    set line=%%i
    :: Ignore lines that start with "#" (comments) or are empty
    if not "!line!"=="" if not "!line:~0,1!"=="#" (
        set deps=!deps! !line!
    )
)

:: Trim leading space (if any)
set deps=%deps:~1%

:: Check if there are dependencies to add
if "%deps%"=="" (
    echo Error: No valid dependencies found in requirements.txt.
    exit /b 1
)

:: Run uv add command
echo Running: uv add %deps%
uv add %deps%

:: Check for errors
if %errorlevel% neq 0 (
    echo Error: Failed to add dependencies to pyproject.toml.
    exit /b 1
)

echo Dependencies successfully added to pyproject.toml.
exit /b 0
