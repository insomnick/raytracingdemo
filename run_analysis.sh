#!/bin/bash

# Exit on any error
set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

echo "$SCRIPT_DIR"

echo "Raytracing Performance Analysis"
echo "============================================"

# Build project with cmake
echo "Building project..."
if ! cmake . -B ./build; then
    echo "ERROR: CMake configuration failed!"
    exit 1
fi

if ! cmake --build ./build -j; then
    echo "ERROR: Build failed!"
    exit 1
fi

echo "Build successful!"
echo "============================================"

# Run Tests
echo "Running raytracing demo..."
if ! ./build/raytracingdemo; then
    echo "ERROR: Raytracing demo execution failed!"
    exit 1
fi

echo "Raytracing demo completed successfully!"
echo "============================================"

# Check if virtual environment exists
if [ ! -d "venv" ]; then
    echo "Setting up virtual environment..."
    if ! python3 -m venv venv; then
        echo "ERROR: Failed to create virtual environment!"
        exit 1
    fi
    echo "Virtual environment created"
fi

# Activate virtual environment
echo "Activating virtual environment..."
if ! source venv/bin/activate; then
    echo "ERROR: Failed to activate virtual environment!"
    exit 1
fi

# Check if dependencies are installed
if ! python -c "import pandas, matplotlib, seaborn, numpy, scipy" 2>/dev/null; then
    echo "Installing dependencies..."
    if ! pip install -r scripts/requirements.txt; then
        echo "ERROR: Failed to install Python dependencies!"
        exit 1
    fi
    echo "Dependencies installed"
fi
echo "============================================"

# Validate test data
echo "Running data validation..."
if ! python scripts/validate_data.py testruns; then
    echo "ERROR: Data validation failed!"
    exit 1
fi

# Run the analysis
echo "Running performance analysis..."
if ! python scripts/bvh_analysis.py "$@"; then
    echo "ERROR: Performance analysis failed!"
    exit 1
fi

echo "DONE"
echo "============================================"
echo "Analysis completed successfully! Check out other scripst for detailed analysis and visualizations."
