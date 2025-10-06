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

shopt -s nullglob
py_files=("$dir"/*.py)

if [ "${#py_files[@]}" -eq 0 ]; then
    log "No Python files found in $dir"
    exit 0
fi

log "Found ${#py_files[@]} Python file(s) to execute"

for file in "${py_files[@]}"; do
    log "Running: $(basename "$file")"
    if ! "$py" "$file"; then
        log "ERROR: $(basename "$file") failed"
        exit 1
    fi
    log "Completed: $(basename "$file")"
done

log "All Python scripts executed successfully"
log "Training process completed"
