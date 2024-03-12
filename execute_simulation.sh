#!/bin/bash

# Get the project directory
projectDir=$(dirname "$(realpath "$0")")

# Create virtual environment
python3 -m venv "$projectDir/venv"

# Activate virtual environment
source "$projectDir/venv/bin/activate"

# Install dependencies from requirements.txt
pip install -r "$projectDir/requirements.txt"

# Run Python script
python3 "$projectDir/simulator.py"

# Deactivate virtual environment
deactivate

# Hang on completion
echo "Script execution completed. Press Enter to exit."
read -r
