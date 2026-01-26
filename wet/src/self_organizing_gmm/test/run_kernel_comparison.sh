#!/bin/bash
# Script to run kernel comparison for MeanShift in SOGMM pipeline

# Get the directory where this script is located
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"

# Navigate to the script directory
cd "$SCRIPT_DIR"

echo "======================================"
echo "MeanShift Kernel Comparison"
echo "======================================"
echo ""
echo "This script will compare three kernels:"
echo "  1. Flat (uniform) kernel"
echo "  2. Gaussian kernel"
echo "  3. Cauchy kernel"
echo ""
echo "Results will be saved to: ./kernel_comparison_results/"
echo ""

# Check if Python environment is activated
if [ -z "$VIRTUAL_ENV" ]; then
    echo "WARNING: No Python virtual environment detected."
    echo "Please activate your environment first with:"
    echo "  source /path/to/gira3d-reconstruction/workon"
    echo ""
    read -p "Do you want to continue anyway? (y/n) " -n 1 -r
    echo
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        exit 1
    fi
fi

# Run the comparison
echo "Starting kernel comparison..."
python3 kernel_comparison.py

echo ""
echo "======================================"
echo "Comparison completed!"
echo "Check ./kernel_comparison_results/ for detailed results."
echo "======================================"
