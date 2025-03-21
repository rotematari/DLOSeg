#!/bin/bash
set -e

echo "Installing sam2..."
cd src/Grounded-SAM-2
pip install -e ./src/Grounded-SAM-2

echo "Installing main package..."
pip install -e .