#!/usr/bin/env bash
set -euo pipefail

# directory containing this script
dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# choose python interpreter
if command -v python3 >/dev/null 2>&1; then
    py=python3
elif command -v python >/dev/null 2>&1; then
    py=python
else
    echo "No Python interpreter (python3 or python) found in PATH." >&2
    exit 1
fi

shopt -s nullglob
py_files=("$dir"/*.py)

if [ "${#py_files[@]}" -eq 0 ]; then
    echo "No Python files found in $dir"
    exit 0
fi

for file in "${py_files[@]}"; do
    echo "Running: $(basename "$file")"
    if ! "$py" "$file"; then
        echo "Error: $(basename "$file") failed" >&2
        exit 1
    fi
done
