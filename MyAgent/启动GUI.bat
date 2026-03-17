@echo off
cd /d "%~dp0"

echo ========================================
echo   MyAgent - 启动 Web 图形界面
echo ========================================
echo.

REM 释放 7860 端口（可选，失败不报错）
netstat -ano | findstr :7860 >nul 2>&1
if %errorlevel% equ 0 (
    for /f "tokens=5" %%a in ('netstat -ano ^| findstr :7860') do (
        taskkill /F /PID %%a >nul 2>&1
    )
    timeout /t 1 /nobreak >nul 2>&1
)

REM 直接用 python 运行（请确保已安装 Python 并加入 PATH，或从 Anaconda Prompt 运行本 bat）
echo [启动] 正在启动 Web 服务...
echo [访问] 启动成功后请打开浏览器访问: http://127.0.0.1:7860
echo.

python webui.py
if errorlevel 1 (
    echo.
    echo [错误] 启动失败。若未安装 Python 或未加入 PATH，请：
    echo   1. 打开 Anaconda Prompt 或 命令提示符
    echo   2. 执行: conda activate myagent
    echo   3. 执行: cd /d "%~dp0"
    echo   4. 执行: python webui.py
    echo.
)

echo.
pause
