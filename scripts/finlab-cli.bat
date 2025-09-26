@echo off
REM FinLab CLI Windows Batch Script
REM This script provides Windows compatibility for the FinLab CLI

REM Get the directory where this script is located
set SCRIPT_DIR=%~dp0

REM Set the Python path to include the src directory
set PYTHONPATH=%SCRIPT_DIR%..\src;%PYTHONPATH%

REM Run the CLI with Python
python "%SCRIPT_DIR%finlab-cli" %*