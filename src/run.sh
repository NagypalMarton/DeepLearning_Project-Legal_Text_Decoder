#!/usr/bin/env bash
set -e  # Exit on error

echo "Starting ML Pipeline..."

# Get script directory
cd "$(dirname "$0")"

# Python interpreter
PYTHON=${PYTHON:-python3}

# Pipeline scripts in order
SCRIPTS=(
    "01_data_acquisition_and_analysis.py"
    "02_data_cleansing_and_preparation.py"
    "03_baseline_model.py"
    "04_incremental_model_development.py"
    "05_defining_evaluation_criteria.py"
    "06_advanced_evaluation_robustness.py"
    "07_advanced_evaluation_explainability.py"
)

# Run each script
for script in "${SCRIPTS[@]}"; do
    echo "Running: $script"
    $PYTHON "$script"
done

echo "Pipeline completed successfully!"
