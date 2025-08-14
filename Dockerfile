# Production RunPod Serverless AI Worker
FROM nvidia/cuda:12.1.0-cudnn8-runtime-ubuntu22.04

# System dependencies for video processing and AI models
RUN apt-get update && apt-get install -y --no-install-recommends \
    python3.10 python3.10-venv python3-pip \
    git build-essential cmake ninja-build \
    ffmpeg libgl1-mesa-glx libglib2.0-0 \
    wget curl unzip \
    && rm -rf /var/lib/apt/lists/*

# Set Python 3.10 as default for python and python3
RUN update-alternatives --install /usr/bin/python python /usr/bin/python3.10 1 \
 && update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.10 1

WORKDIR /app

# Upgrade pip and install build tools
RUN python -m pip install --upgrade pip setuptools wheel

# Copy requirements first (Docker layer caching)
COPY requirements.txt /app/requirements.txt

# Install PyTorch with CUDA 12.1 support
RUN pip install --no-cache-dir torch==2.3.1 torchvision==0.18.1 --index-url https://download.pytorch.org/whl/cu121

# Install core dependencies
RUN pip install --no-cache-dir -r /app/requirements.txt

# Install AI models from Git (separate layer for caching)
RUN pip install --no-cache-dir \
    git+https://github.com/pq-yang/MatAnyone.git#egg=matanyone \
    git+https://github.com/facebookresearch/segment-anything-2.git#egg=segment-anything-2

# Install MatAnyone additional dependencies
RUN pip install --no-cache-dir \
    omegaconf==2.3.0 hydra-core==1.3.2 easydict==0.1.10 \
    imageio==2.25.0 av>=10.0.0 scipy>=1.10.0

# Copy worker code
COPY handler.py /app/handler.py
COPY test_input.json /app/test_input.json

# Set environment variables for model caching and GPU
ENV PYTHONUNBUFFERED=1
ENV CUDA_VISIBLE_DEVICES=0
ENV HF_HOME=/tmp/huggingface
ENV TRANSFORMERS_CACHE=/tmp/huggingface/transformers
ENV TORCH_HOME=/tmp/torch

# Create cache directories
RUN mkdir -p /tmp/huggingface /tmp/torch /tmp/workspace

# RunPod Serverless entrypoint
CMD ["python", "-u", "/app/handler.py"]
