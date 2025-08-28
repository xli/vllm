#!/bin/bash

CUDA_VERSION="${CUDA_VERSION:-12.8.1}"

echo "Setting up CUDA environment..."

CUDA_MAJOR_MINOR=$(echo "$CUDA_VERSION" | cut -d. -f1,2)
export CUDA_HOME="$HOME/local/cuda-$CUDA_MAJOR_MINOR"
export PATH="$CUDA_HOME/bin:$PATH"
export CUDA_NVCC_EXECUTABLE="$CUDA_HOME/bin/nvcc"
export LD_LIBRARY_PATH="$CUDA_HOME/lib64:$LD_LIBRARY_PATH"
export TORCH_CUDA_ARCH_LIST="7.0 7.5 8.0 8.6 8.9 9.0+PTX"
