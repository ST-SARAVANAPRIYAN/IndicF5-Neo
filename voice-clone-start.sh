#!/bin/bash
# Set terminal window title
echo -ne "\033]0;IndicF5 Neo Voice Clone Server\007"

# Navigate to script directory
cd "$(dirname "$0")"

# Activate venv if it exists
if [ -d ".venv" ]; then
    echo "Activating virtual environment..."
    source .venv/bin/activate
else
    echo "WARNING: .venv folder not found. Running with default python..."
fi

echo "Starting IndicF5 Neo server..."
python launch.py
