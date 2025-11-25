@echo off
REM 集成测试快速启动脚本
REM 使用 .venv 虚拟环境运行集成测试

echo ======================================================================
echo VRChat 社交辅助工具 - 集成测试
echo ======================================================================
echo.

REM 检查虚拟环境是否存在
if not exist ".venv\Scripts\python.exe" (
    echo [错误] 虚拟环境不存在！
    echo 请先创建虚拟环境: python -m venv .venv
    pause
    exit /b 1
)

echo [信息] 使用虚拟环境: .venv
echo.

REM 运行集成测试
.venv\Scripts\python.exe tests\integrated_test.py %*

pause
