#!/usr/bin/env bash
set -euo pipefail

# directory containing this script
dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Output directory from environment variable or default
OUTPUT_DIR="${OUTPUT_DIR:-/app/output}"

# Log file path
LOG_FILE="$OUTPUT_DIR/training_log.txt"

# Ensure output directory exists
mkdir -p "$OUTPUT_DIR"

# Function to log with timestamp
log() {
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] $*" | tee -a "$LOG_FILE"
}

# Redirect all output to log file
exec > >(tee -a "$LOG_FILE") 2>&1

log "Starting training process..."
log "Output directory: $OUTPUT_DIR"
log "Log file: $LOG_FILE"

# choose python interpreter
if command -v python3 >/dev/null 2>&1; then
    py=python3
elif command -v python >/dev/null 2>&1; then
    py=python
else
    log "ERROR: No Python interpreter (python3 or python) found in PATH."
    exit 1
fi

log "Using Python interpreter: $py"

# Define scripts in execution order
scripts=(
    "01_data_acquisition_and_analysis.py"
    "02_data_cleansing_and_preparation.py"
    "03_baseline_model.py"
    "04_incremental_model_development.py"
    "05_defining_evaluation_criteria.py"
    "06_advanced_evaluation.py"
)


# Optional: Add API/frontend services after training
START_API_SERVICE="${START_API_SERVICE:-0}"
START_FRONTEND_SERVICE="${START_FRONTEND_SERVICE:-0}"

if [[ "$START_API_SERVICE" == "1" || "$START_API_SERVICE" == "true" ]]; then
    scripts+=("07_start_api_service.py")
    log "API service will be started after training (START_API_SERVICE=$START_API_SERVICE)"
fi
if [[ "$START_FRONTEND_SERVICE" == "1" || "$START_FRONTEND_SERVICE" == "true" ]]; then
    scripts+=("08_start_frontend_service.py")
    log "Frontend service will be started after training (START_FRONTEND_SERVICE=$START_FRONTEND_SERVICE)"
fi

log "Found ${#scripts[@]} Python script(s) to execute"

for script in "${scripts[@]}"; do
    file="$dir/$script"
    if [ ! -f "$file" ]; then
        log "WARNING: $script not found, skipping"
        continue
    fi
    
    # Skip API and Frontend scripts - they will be started after training
    if [[ "$script" == "07_start_api_service.py" ]] || [[ "$script" == "08_start_frontend_service.py" ]]; then
        log "Skipping $script (will start in background after training)"
        continue
    fi
    
    log "Running: $script"
    if ! "$py" "$file"; then
        log "ERROR: $script failed"
        exit 1
    fi
    log "Completed: $script"
done

log "All training scripts executed successfully"

# Start API and Frontend services in background (non-blocking)
if [[ "$START_API_SERVICE" == "1" || "$START_API_SERVICE" == "true" ]]; then
    log "Starting API service in background..."
    nohup "$py" "$dir/07_start_api_service.py" >> "$OUTPUT_DIR/api.log" 2>&1 &
    log "API service started (PID: $!)"
fi

if [[ "$START_FRONTEND_SERVICE" == "1" || "$START_FRONTEND_SERVICE" == "true" ]]; then
    log "Starting Frontend service in background..."
    nohup "$py" "$dir/08_start_frontend_service.py" >> "$OUTPUT_DIR/frontend.log" 2>&1 &
    log "Frontend service started (PID: $!)"
fi

log "Training process completed - services running in background"
log "Keeping container alive..."

# Keep container alive indefinitely
sleep infinity
