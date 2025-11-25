@echo off
chcp 65001 >nul
echo ====================================
echo VRChat 社交辅助工具 - 完整流程测试
echo ====================================
echo.
echo 正在启动集成测试程序...
echo.

cd /d "%~dp0"
python tests\integrated_test.py --full

echo.
echo 测试程序已退出
pause
