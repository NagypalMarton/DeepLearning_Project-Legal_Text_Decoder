# Universal Docker launcher for Legal Text Decoder (PowerShell)
# Works on Windows PowerShell and PowerShell Core

param(
    [switch]$NoBuild = $false,
    [switch]$CpuOnly = $false
)

$ErrorActionPreference = "Stop"

# Colors
function Write-ColorOutput {
    param([string]$Color, [string]$Message)
    $colors = @{
        "Red" = "Red"
        "Green" = "Green"
        "Yellow" = "Yellow"
        "Blue" = "Cyan"
    }
    Write-Host $Message -ForegroundColor $colors[$Color]
}

# Image name
$IMAGE_NAME = "deeplearning_project-legal_text_decoder:1.0"
$CURRENT_DIR = $PWD.Path

Write-ColorOutput "Blue" "╔════════════════════════════════════════╗"
Write-ColorOutput "Blue" "║   Legal Text Decoder - Docker Launch  ║"
Write-ColorOutput "Blue" "╚════════════════════════════════════════╝"
Write-Host ""
Write-ColorOutput "Yellow" "Platform: Windows"
Write-ColorOutput "Yellow" "Working Directory: $CURRENT_DIR"
Write-Host ""

# Check if Docker is installed
try {
    docker version | Out-Null
} catch {
    Write-ColorOutput "Red" "❌ Docker not found! Please install Docker Desktop."
    exit 1
}

# Check for data directory
if (-not (Test-Path "$CURRENT_DIR\data")) {
    Write-ColorOutput "Yellow" "⚠️  Warning: 'data' directory not found!"
    Write-ColorOutput "Yellow" "   Creating empty 'data' directory..."
    New-Item -ItemType Directory -Path "$CURRENT_DIR\data" -Force | Out-Null
}

# Check for GPU support
$GPU_FLAG = ""
if (-not $CpuOnly) {
    try {
        nvidia-smi | Out-Null
        Write-ColorOutput "Green" "✅ GPU detected!"
        $GPU_FLAG = "--gpus all"
        Write-ColorOutput "Blue" "   Running with GPU acceleration..."
    } catch {
        Write-ColorOutput "Yellow" "⚠️  No GPU detected or NVIDIA driver not installed."
        Write-ColorOutput "Yellow" "   Running in CPU-only mode (this will be slower)..."
        Write-ColorOutput "Yellow" "   Use -CpuOnly flag to suppress this check."
    }
} else {
    Write-ColorOutput "Yellow" "⚠️  CPU-only mode forced."
}

# Build image if not exists or not skipping build
if (-not $NoBuild) {
    try {
        docker image inspect $IMAGE_NAME | Out-Null
        Write-ColorOutput "Green" "✅ Image found: $IMAGE_NAME"
    } catch {
        Write-ColorOutput "Yellow" "📦 Image not found. Building..."
        docker build -t $IMAGE_NAME .
        if ($LASTEXITCODE -ne 0) {
            Write-ColorOutput "Red" "❌ Build failed!"
            exit $LASTEXITCODE
        }
        Write-ColorOutput "Green" "✅ Build complete!"
    }
} else {
    Write-ColorOutput "Blue" "⏭️  Skipping build check (NoBuild flag set)"
}

Write-Host ""
Write-ColorOutput "Blue" "🚀 Starting training pipeline..."
Write-Host ""

# Run container with proper volume mounting for Windows
$dataMount = "$CURRENT_DIR\data:/app/data"
$outputMount = "$CURRENT_DIR\output:/app/output"

$dockerArgs = @(
    "run",
    "--rm"
)

if ($GPU_FLAG) {
    $dockerArgs += $GPU_FLAG
}

$dockerArgs += @(
    "-v", $dataMount,
    "-v", $outputMount,
    $IMAGE_NAME
)

& docker $dockerArgs

$EXIT_CODE = $LASTEXITCODE

Write-Host ""
if ($EXIT_CODE -eq 0) {
    Write-ColorOutput "Green" "✅ Training completed successfully!"
    Write-ColorOutput "Green" "   Results are in: $CURRENT_DIR\output\"
} else {
    Write-ColorOutput "Red" "❌ Training failed with exit code: $EXIT_CODE"
    exit $EXIT_CODE
}
