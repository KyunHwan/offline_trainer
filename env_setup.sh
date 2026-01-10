#!/bin/bash

echo "Installing pytorch and related libraries for using pytorch models"
uv pip install torch==2.9.0 torchvision==0.24.0 --index-url https://download.pytorch.org/whl/cu130
uv pip install flow_matching
uv pip install schedulefree
uv pip install einops

echo "Installing lerobot and related libraries for dataset"
uv pip install --no-deps lerobot
uv pip install torchcodec

echo "Setting up Depth Anything v3 as submodule"
uv pip install -e ./policy_constructor/model_constructor/blocks/experiments/backbones/vision/externals/depth_anything_3
