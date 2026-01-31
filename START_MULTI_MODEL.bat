@echo off
REM AGI 自主系统 V6.1 - 快速开始脚本
REM 多基座模型版本

echo ========================================================================
echo AGI AUTONOMOUS CORE V6.1 - Multi-Base Model Edition
echo ========================================================================
echo.

REM 检查 Python
python --version >nul 2>&1
if errorlevel 1 (
    echo [Error] Python not found. Please install Python 3.8+
    pause
    exit /b 1
)

echo [1] Checking dependencies...
pip show openai >nul 2>&1
if errorlevel 1 (
    echo [Install] Installing required packages...
    pip install openai python-dotenv
)

echo.
echo [2] Checking API KEY configuration...
if not exist .env (
    echo [Config] Creating .env file from template...
    copy .env.multi_model .env >nul
    echo.
    echo ========================================================================
    echo IMPORTANT: Please edit .env file and add your API KEYs
    echo ========================================================================
    echo.
    echo Available models:
    echo   - DEEPSEEK_API_KEY    (recommended for code generation)
    echo   - ZHIPU_API_KEY       (good for Chinese)
    echo   - KIMI_API_KEY        (long context)
    echo   - QWEN_API_KEY        (balanced)
    echo   - GEMINI_API_KEY      (multimodal)
    echo.
    notepad .env
    echo.
    echo Press any key when you have configured your API KEYs...
    pause >nul
)

echo.
echo [3] Choose mode:
echo ========================================================================
echo   1. Run single model (interactive)
echo   2. Run all available models (comparison)
echo   3. Quick model comparison test
echo   4. Run V6.0 (original DeepSeek version)
echo ========================================================================

set /p choice="Enter choice (1-4): "

if "%choice%"=="1" (
    echo.
    echo Available models:
    echo   1. deepseek
    echo   2. zhipu
    echo   3. kimi
    echo   4. qwen
    echo   5. gemini
    echo.
    set /p model="Enter model name: "
    echo.
    echo Starting AGI V6.1 with %model%...
    python AGI_AUTONOMOUS_CORE_V6_1_MULTI_BASE.py --model %model%
)

if "%choice%"=="2" (
    echo.
    echo Starting all available models in parallel...
    python AGI_AUTONOMOUS_CORE_V6_1_MULTI_BASE.py --model all
)

if "%choice%"=="3" (
    echo.
    echo Starting quick comparison test...
    python compare_models.py
)

if "%choice%"=="4" (
    echo.
    echo Starting V6.0 (DeepSeek only)...
    python AGI_AUTONOMOUS_CORE_V6_0.py
)

echo.
echo Done! Check data/autonomous_outputs_v6_1/ for results.
pause
