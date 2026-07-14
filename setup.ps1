# ResearchIQ - Local Environment Setup Script
# Run this script in PowerShell to configure your project dependencies.

Write-Host "=== ResearchIQ Project Setup ===" -ForegroundColor Cyan

# 1. Check Python installation
$pythonCheck = Get-Command python -ErrorAction SilentlyContinue
if (-not $pythonCheck) {
    Write-Error "Python was not found on your system PATH. Please download and install Python 3.10+ from python.org"
    Exit
}

Write-Host "Python detected: $(python --version)" -ForegroundColor Green

# 2. Setup execution policy for scripts in the current PowerShell process
Write-Host "Adjusting Execution Policy..." -ForegroundColor Yellow
Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope Process -Force

# 3. Create virtual environment
if (-not (Test-Path "venv")) {
    Write-Host "Creating Virtual Environment (venv)..." -ForegroundColor Yellow
    python -m venv venv
    Write-Host "Virtual environment created." -ForegroundColor Green
} else {
    Write-Host "Virtual environment already exists." -ForegroundColor Green
}

# 4. Activate environment and install requirements
Write-Host "Activating venv and installing requirements.txt..." -ForegroundColor Yellow
& "venv/Scripts/Activate.ps1"

python -m pip install --upgrade pip
pip install -r backend/requirements.txt

Write-Host "=== SETUP COMPLETED SUCCESSFULLY ===" -ForegroundColor Green
Write-Host "Instructions to run the project:" -ForegroundColor Cyan
Write-Host "1. Open backend/.env and paste your GEMINI_API_KEY." -ForegroundColor White
Write-Host "2. Run this command to start the server: python backend/main.py" -ForegroundColor White
Write-Host "3. Open your browser and navigate to http://127.0.0.1:8000/docs to view the interactive API documentation." -ForegroundColor White
