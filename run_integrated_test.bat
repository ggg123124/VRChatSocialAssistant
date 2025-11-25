@echo off
REM VRChat 社交辅助工具 - 集成测试启动脚本

echo ======================================================================
echo VRChat 社交辅助工具 - 集成测试
echo ======================================================================
echo.

REM 检查 Python 是否可用
python --version >nul 2>&1
if errorlevel 1 (
    echo 错误: 未找到 Python，请先安装 Python 3.8+
    pause
    exit /b 1
)

echo 启动集成测试程序...
echo.

REM 运行测试程序
python tests\integrated_test.py %*

if errorlevel 1 (
    echo.
    echo 程序异常退出
    pause
)
