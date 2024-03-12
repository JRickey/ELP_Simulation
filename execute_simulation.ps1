# Get the project directory
$projectDir = Split-Path -Parent -Path $MyInvocation.MyCommand.Definition

# Create virtual environment
python -m venv "$projectDir\venv"

# Activate virtual environment
$activateScript = "$projectDir\venv\Scripts\Activate.ps1"
. $activateScript

# Install dependencies from requirements.txt
pip install -r "$projectDir\requirements.txt"

# Run Python script
$pythonScript = "$projectDir\simulator.py"
python $pythonScript

# Deactivate virtual environment
deactivate

# Hang on completion
Write-Host "Script execution completed. Press Enter to exit."
$null = $Host.UI.RawUI.ReadKey("NoEcho,IncludeKeyDown")
