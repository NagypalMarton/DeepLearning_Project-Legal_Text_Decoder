#!/usr/bin/env bash
# Universal Docker launcher for Legal Text Decoder
# Works on Linux, macOS, and Windows (Git Bash/WSL)

set -euo pipefail

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Image name
IMAGE_NAME="deeplearning_project-legal_text_decoder:1.0"

# Detect platform
detect_platform() {
    case "$(uname -s)" in
        Linux*)     echo "linux";;
        Darwin*)    echo "macos";;
        CYGWIN*)    echo "windows";;
        MINGW*)     echo "windows";;
        MSYS*)      echo "windows";;
        *)          echo "unknown";;
    esac
}

# Check if GPU is available
check_gpu() {
    if command -v nvidia-smi &> /dev/null; then
        if nvidia-smi &> /dev/null; then
            return 0
        fi
    fi
    return 1
}

# Get absolute path for current directory
get_abs_path() {
    if [[ "$PLATFORM" == "windows" ]]; then
        pwd -W 2>/dev/null || pwd
    else
        pwd
    fi
}

PLATFORM=$(detect_platform)
CURRENT_DIR=$(get_abs_path)

echo -e "${BLUE}╔════════════════════════════════════════╗${NC}"
echo -e "${BLUE}║   Legal Text Decoder - Docker Launch  ║${NC}"
echo -e "${BLUE}╚════════════════════════════════════════╝${NC}"
echo ""
echo -e "${YELLOW}Platform:${NC} $PLATFORM"
echo -e "${YELLOW}Working Directory:${NC} $CURRENT_DIR"
echo ""

# Check if Docker is installed
if ! command -v docker &> /dev/null; then
    echo -e "${RED}❌ Docker not found! Please install Docker first.${NC}"
    exit 1
fi

# Check for data directory
if [ ! -d "$CURRENT_DIR/data" ]; then
    echo -e "${YELLOW}⚠️  Warning: 'data' directory not found!${NC}"
    echo -e "${YELLOW}   Creating empty 'data' directory...${NC}"
    mkdir -p "$CURRENT_DIR/data"
fi

# Check for GPU support
GPU_FLAG=""
if check_gpu; then
    echo -e "${GREEN}✅ GPU detected!${NC}"
    GPU_FLAG="--gpus all"
    echo -e "${BLUE}   Running with GPU acceleration...${NC}"
else
    echo -e "${YELLOW}⚠️  No GPU detected or nvidia-docker not installed.${NC}"
    echo -e "${YELLOW}   Running in CPU-only mode (this will be slower)...${NC}"
fi

# Build image if not exists
if ! docker image inspect "$IMAGE_NAME" &> /dev/null; then
    echo -e "${YELLOW}📦 Image not found. Building...${NC}"
    docker build -t "$IMAGE_NAME" .
    echo -e "${GREEN}✅ Build complete!${NC}"
else
    echo -e "${GREEN}✅ Image found: $IMAGE_NAME${NC}"
fi

echo ""
echo -e "${BLUE}🚀 Starting training pipeline...${NC}"
echo ""

# Run container with platform-specific volume mounting
docker run --rm $GPU_FLAG \
    -v "$CURRENT_DIR/data:/app/data" \
    -v "$CURRENT_DIR/output:/app/output" \
    "$IMAGE_NAME"

EXIT_CODE=$?

echo ""
if [ $EXIT_CODE -eq 0 ]; then
    echo -e "${GREEN}✅ Training completed successfully!${NC}"
    echo -e "${GREEN}   Results are in: $CURRENT_DIR/output/${NC}"
else
    echo -e "${RED}❌ Training failed with exit code: $EXIT_CODE${NC}"
    exit $EXIT_CODE
fi
