# CardioMap — quick start (Windows PowerShell)
$root = $PSScriptRoot
Set-Location $root

if (-not (Test-Path "models/cath.joblib")) {
    Write-Host "Training models (first run)..."
    python -m ml.train
    python -m ml.explain
}

Write-Host "Building frontend..."
Set-Location web
npm run build
Set-Location $root

Write-Host "Starting API + built UI at http://127.0.0.1:8000"
python -m uvicorn api.main:app --host 127.0.0.1 --port 8000
