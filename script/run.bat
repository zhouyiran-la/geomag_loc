@echo off
chcp 65001 >nul
REM 滑窗生生成地磁序列数据集脚本


REM ==============================================
REM 运行示例： run.bat 300 1 0
REM 参数说明：
REM   %1 -> --win
REM   %2 -> --stride
REM   %3 -> --mode
REM ==============================================

REM 检查参数数量
if "%~3"=="" (
    echo 用法: run.bat [win] [stride] [mode]
    echo 示例: run.bat 300 1 0
    pause
    exit /b
)

set WIN=%1
set STRIDE=%2
set MODE=%3

echo 滑窗长度: %WIN%
echo 步长: %STRIDE%
echo 模式: %MODE%
echo -------------------------------------------
python ".\preprocess\get_seq_by_slide_window.py" ".\data\origin\4.19数据\50\norm" ".\data\preprocessed\4.19数据\50" --win %WIN% --stride %STRIDE% --mode %MODE%
python ".\preprocess\get_seq_by_slide_window.py" ".\data\origin\4.19数据\100\norm" ".\data\preprocessed\4.19数据\100" --win %WIN% --stride %STRIDE% --mode %MODE%
python ".\preprocess\get_seq_by_slide_window.py" ".\data\origin\4.25数据\50\norm" ".\data\preprocessed\4.25数据\50" --win %WIN% --stride %STRIDE% --mode %MODE%
python ".\preprocess\get_seq_by_slide_window.py" ".\data\origin\4.25数据\100\norm" ".\data\preprocessed\4.25数据\100" --win %WIN% --stride %STRIDE% --mode %MODE%
python ".\preprocess\get_seq_by_slide_window.py" ".\data\origin\4.26数据\50\norm" ".\data\preprocessed\4.26数据\50" --win %WIN% --stride %STRIDE% --mode %MODE%
python ".\preprocess\get_seq_by_slide_window.py" ".\data\origin\4.26数据\100\norm" ".\data\preprocessed\4.26数据\100" --win %WIN% --stride %STRIDE% --mode %MODE%
python ".\preprocess\get_seq_by_slide_window.py" ".\data\origin\4.26数据\xy用\norm" ".\data\preprocessed\4.26数据\xy用" --win %WIN% --stride %STRIDE% --mode %MODE%
python ".\preprocess\get_seq_by_slide_window.py" ".\data\origin\4.28数据\50\norm" ".\data\preprocessed\4.28数据\50" --win %WIN% --stride %STRIDE% --mode %MODE%
python ".\preprocess\get_seq_by_slide_window.py" ".\data\origin\4.28数据\100\norm" ".\data\preprocessed\4.28数据\100" --win %WIN% --stride %STRIDE% --mode %MODE%
echo -------------------------------------------
echo ✅ 所有任务执行完成


pause
