# PowerShell script to run the API and Frontend services
# This is SEPARATE from the training pipeline

Write-Host "========================================" -ForegroundColor Cyan
Write-Host "Legal Text Decoder - ML Service Starter" -ForegroundColor Cyan
Write-Host "========================================" -ForegroundColor Cyan
Write-Host ""

# Check if models exist
$OUTPUT_DIR = if ($env:OUTPUT_DIR) { $env:OUTPUT_DIR } else { "/app/output" }
$MODELS_DIR = "$OUTPUT_DIR/models"

if (-not (Test-Path $MODELS_DIR)) {
    Write-Host "ERROR: Models directory not found at $MODELS_DIR" -ForegroundColor Red
    Write-Host "Please run the training pipeline first!" -ForegroundColor Yellow
    exit 1
}

if (-not (Test-Path "$MODELS_DIR/baseline_model.pkl")) {
    Write-Host "WARNING: Baseline model not found!" -ForegroundColor Yellow
}

Write-Host "Starting services..." -ForegroundColor Green
Write-Host ""

# Determine Python interpreter
$PY = if (Get-Command python3 -ErrorAction SilentlyContinue) { "python3" } 
      elseif (Get-Command python -ErrorAction SilentlyContinue) { "python" }
      else { 
          Write-Host "ERROR: No Python interpreter found!" -ForegroundColor Red
          exit 1
      }

# Change to script directory
$ScriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
Set-Location $ScriptDir

# Start API in background
Write-Host "Starting API server on port 8000..." -ForegroundColor Yellow
$ApiJob = Start-Job -ScriptBlock {
    param($PythonExe, $ScriptPath)
    Set-Location $ScriptPath
    & $PythonExe api/app.py
} -ArgumentList $PY, $ScriptDir

Write-Host "API Job ID: $($ApiJob.Id)" -ForegroundColor Gray

# Wait for API to be ready
Write-Host "Waiting for API to start..." -ForegroundColor Yellow
$Ready = $false
for ($i = 1; $i -le 30; $i++) {
    try {
        $Response = Invoke-WebRequest -Uri "http://localhost:8000/" -TimeoutSec 2 -ErrorAction SilentlyContinue
        if ($Response.StatusCode -eq 200) {
            Write-Host "✓ API is ready!" -ForegroundColor Green
            $Ready = $true
            break
        }
    } catch {
        # Continue waiting
    }
    Start-Sleep -Seconds 1
}

if (-not $Ready) {
    Write-Host "ERROR: API failed to start in 30 seconds" -ForegroundColor Red
    Stop-Job -Job $ApiJob
    Remove-Job -Job $ApiJob
    exit 1
}

# Display info
Write-Host ""
Write-Host "========================================" -ForegroundColor Cyan
Write-Host "Access the application at:" -ForegroundColor Cyan
Write-Host "  Frontend: http://localhost:8501" -ForegroundColor White
Write-Host "  API Docs: http://localhost:8000/docs" -ForegroundColor White
Write-Host "========================================" -ForegroundColor Cyan
Write-Host ""
Write-Host "Press Ctrl+C to stop both services" -ForegroundColor Yellow
Write-Host ""

# Start Frontend (this will block)
try {
    & streamlit run frontend/app.py --server.port 8501 --server.address 0.0.0.0
}
finally {
    # Cleanup on exit
    Write-Host ""
    Write-Host "Stopping services..." -ForegroundColor Yellow
    Stop-Job -Job $ApiJob -ErrorAction SilentlyContinue
    Remove-Job -Job $ApiJob -ErrorAction SilentlyContinue
}
