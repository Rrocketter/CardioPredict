#!/bin/bash

# CardioPredict Research Platform Startup Script

echo "🫀 CardioPredict Research Platform"
echo "=================================="
echo "Starting simplified research version..."
echo ""

# Check if virtual environment exists
if [ ! -d "venv" ]; then
    echo "Creating virtual environment..."
    python3 -m venv venv
fi

# Activate virtual environment
echo "Activating virtual environment..."
source venv/bin/activate

# Install requirements
echo "Installing requirements..."
pip install -r requirements_research.txt

echo ""
echo "✓ Setup complete!"
echo "✓ Simplified for scientific publication"
echo "✓ No authentication required"
echo "✓ Open access research platform"
echo ""
echo "Starting Flask development server..."
echo "Visit: http://localhost:5000"
echo ""

# Run the simplified app
python app_simple.py
