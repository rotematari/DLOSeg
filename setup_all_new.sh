#!/bin/bash
set -e

echo "Installing dependencies..."
# CUDA 12.1
# pip install torch==2.5.1 torchvision==0.20.1 torchaudio==2.5.1 --index-url https://download.pytorch.org/whl/cu121
pip install -r requirements.txt

echo "Installing Grounded sam2 ..."
echo "Installing Segment Anything 2 ..."
cd src/Grounded_SAM_2
pip install -e .

echo "Installing Grounding DINO  ..."
pip install --no-build-isolation -e grounding_dino

cd ../..
echo "Installing main package..."
pip install -e .


echo "install zed sdk python "
python3 /usr/local/zed/get_python_api.py
