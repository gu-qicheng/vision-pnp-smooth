@echo off
setlocal
cd /d "%~dp0"
if not exist ".venv\Scripts\python.exe" (
    goto setup_environment
)
"%~dp0.venv\Scripts\python.exe" -c "import numpy, cv2" >nul 2>&1
if errorlevel 1 goto setup_environment
goto run_demo

:setup_environment
echo 正在创建或修复项目虚拟环境并安装依赖...
powershell -NoProfile -ExecutionPolicy Bypass -File "%~dp0setup_pnp_env.ps1"
if errorlevel 1 (
    echo 环境创建失败。
    exit /b 1
)

:run_demo
"%~dp0.venv\Scripts\python.exe" "%~dp0pnp_square_demo.py" --calibration "%~dp0camera_calibration.npz" --camera-index 0
endlocal
