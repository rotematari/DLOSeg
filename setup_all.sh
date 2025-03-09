#!/bin/bash
set -e

echo "Installing sam2_rt..."
pip install -e ./src/sam2_realtime

echo "Installing main package..."
pip install -e .