#!/bin/bash
# run_app.sh — Helper script to run OFN GA Explorer

# 1. Path to your new Python 3.12
PYTHON_EXE="/opt/homebrew/bin/python3.12"

# 2. Ensure we don't try to write bytecode
export PYTHONDONTWRITEBYTECODE=1

# 3. Add local dependencies and project root to PYTHONPATH
export PYTHONPATH=$(pwd):$(pwd)/deps_312:$PYTHONPATH

# 4. Ensure Matplotlib has a writable config directory
mkdir -p mpg_config
export MPLCONFIGDIR=$(pwd)/mpg_config

# 5. Run the application
$PYTHON_EXE run.py
