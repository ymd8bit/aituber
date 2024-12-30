FROM nvidia/cuda:12.4.1-cudnn-devel-ubuntu22.04

# Set environment variables.
ENV DEBIAN_FRONTEND=noninteractive
ENV TZ=Asia/Tokyo
ENV CUDA_HOME="/usr/local/cuda"
ENV CUDA_ARCHITECTURES=90;89;86;80;75;70;61;52;37

# Install required apt packages and clear cache afterwards.
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
    git \
    software-properties-common \
    build-essential \
    cmake \
    curl \
    wget \
    ffmpeg \
    sudo \
    vim-tiny \
    pkg-config \
    protobuf-compiler

RUN apt-get install -y --no-install-recommends \
    python-is-python3 \
    python3-dev \
    python3-pip \
    python3-venv

RUN pip install --upgrade pip
RUN pip install --index-url https://download.pytorch.org/whl/cu124 \
    torch==2.5.1 torchvision==0.20.1 torchaudio==2.5.1
# install onnxruntime for Cuda12
RUN pip install onnxruntime-gpu==v1.20.1
