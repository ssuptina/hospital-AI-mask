@echo off
setlocal enabledelayedexpansion

color 0A
cls
echo =====================================================
echo    CV-Based Mask Detection System - ENHANCED
echo    Complete Setup ^& Training Automation
echo =====================================================
echo.

echo [STEP 1/6] Checking Python installation...
python --version >nul 2>&1
if errorlevel 1 (
    echo ERROR: Python not found. Please install Python 3.8+
    pause
    exit /b 1
)
echo OK Python found

echo.
echo [STEP 2/6] Activating virtual environment...
if exist venv\Scripts\activate.bat (
    call venv\Scripts\activate.bat
    echo OK Virtual environment activated
) else (
    echo Creating virtual environment...
    python -m venv venv
    call venv\Scripts\activate.bat
    echo OK Virtual environment created and activated
)

echo.
echo [STEP 3/6] Installing dependencies...
pip install -q --upgrade pip
pip install -q -r requirements_enhanced.txt
if errorlevel 1 (
    echo ERROR: Failed to install dependencies
    pause
    exit /b 1
)
echo OK All dependencies installed

echo.
echo [STEP 4/6] Downloading face detection models...
python advanced_face_detector\download_models.py
echo OK Face detection models ready

echo.
echo [STEP 5/6] Verifying dataset structure...
if not exist dataset\mask_correct (
    mkdir dataset\mask_correct
    echo Created dataset\mask_correct - add images here
)
if not exist dataset\mask_incorrect (
    mkdir dataset\mask_incorrect
    echo Created dataset\mask_incorrect - add images here
)
if not exist dataset\no_mask (
    mkdir dataset\no_mask
    echo Created dataset\no_mask - add images here
)
echo OK Dataset folders ready (add 500-1000 images per folder)

echo.
echo [STEP 6/6] Checking for trained model...
if exist src\best_model.h5 (
    echo OK Pre-trained model found
) else (
    echo WARN No trained model found.
    echo      Run: cd src ^&^& python train_mask_classifier.py
)

echo.
echo =====================================================
echo  SETUP COMPLETE
echo =====================================================
echo.
echo NEXT STEPS:
echo   1. Add images to dataset\mask_correct\, mask_incorrect\, no_mask\
echo   2. Train: cd src ^&^& python train_mask_classifier.py
echo   3. Run app: cd web_app ^&^& ..\venv\Scripts\python app_enhanced.py
echo   4. Open: http://localhost:5000
echo.
pause
