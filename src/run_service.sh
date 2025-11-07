#!/bin/bash
# Script to run the API and Frontend services together
# This is SEPARATE from the training pipeline

set -euo pipefail

echo "========================================"
echo "Legal Text Decoder - ML Service Starter"
echo "========================================"
echo ""

# Check if models exist
OUTPUT_DIR="${OUTPUT_DIR:-/app/output}"
MODELS_DIR="$OUTPUT_DIR/models"

if [ ! -d "$MODELS_DIR" ]; then
    echo "ERROR: Models directory not found at $MODELS_DIR"
    echo "Please run the training pipeline first!"
    exit 1
fi

if [ ! -f "$MODELS_DIR/baseline_model.pkl" ]; then
    echo "WARNING: Baseline model not found!"
fi

echo "Starting services..."
echo ""

# Determine Python interpreter
if command -v python3 >/dev/null 2>&1; then
    PY=python3
elif command -v python >/dev/null 2>&1; then
    PY=python
else
    echo "ERROR: No Python interpreter found!"
    exit 1
fi

# Start API in background
echo "Starting API server on port 8000..."
cd "$(dirname "$0")"
$PY api/app.py > /tmp/api.log 2>&1 &
API_PID=$!
echo "API PID: $API_PID"

# Wait for API to be ready
echo "Waiting for API to start..."
for i in {1..30}; do
    if curl -s http://localhost:8000/ > /dev/null 2>&1; then
        echo "✓ API is ready!"
        break
    fi
    sleep 1
    if [ $i -eq 30 ]; then
        echo "ERROR: API failed to start in 30 seconds"
        kill $API_PID 2>/dev/null || true
        exit 1
    fi
done

# Start Frontend
echo ""
echo "Starting Frontend on port 8501..."
echo ""
echo "========================================"
echo "Access the application at:"
echo "  Frontend: http://localhost:8501"
echo "  API Docs: http://localhost:8000/docs"
echo "========================================"
echo ""
echo "Press Ctrl+C to stop both services"
echo ""

# Trap to cleanup on exit
cleanup() {
    echo ""
    echo "Stopping services..."
    kill $API_PID 2>/dev/null || true
    exit 0
}
trap cleanup INT TERM

# Start Frontend (this will block)
streamlit run frontend/app.py --server.port 8501 --server.address 0.0.0.0

# If streamlit exits, cleanup API
cleanup
