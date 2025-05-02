FROM pytorch/pytorch:2.1.0-cuda12.1-cudnn8-runtime
ENV DEBIAN_FRONTEND=noninteractive
WORKDIR /app

# System dependencies
RUN apt-get update && apt-get install -y \
    git \
    curl \
    ffmpeg \
    libgl1 \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    && rm -rf /var/lib/apt/lists/*

# Copy initial requirements file
COPY initial_requirements.txt .

# Start the installation process that worked for you
RUN pip install --upgrade pip

# First, install numpy clean
RUN pip uninstall numpy -y || true
RUN pip install numpy --no-cache-dir

# Install PyTorch stack
RUN pip uninstall torch torchvision torchaudio -y || true
RUN pip install torch==2.5.1 torchvision==0.20.1 torchaudio==2.5.1 \
    --index-url https://download.pytorch.org/whl/cu121

# Install transformers and other dependencies
RUN pip install transformers==4.38.2
RUN pip install accelerate sentencepiece attrdict einops

# Install xformers compatible with PyTorch
RUN pip install xformers --index-url https://download.pytorch.org/whl/cu121

# Install timm
RUN pip install "timm>=0.9.16"

# Install Gradio and related packages
RUN pip install gradio==3.48.0 gradio-client==0.6.1 mdtex2html==1.3.0 pypinyin==0.50.0 \
    tiktoken==0.5.2 tqdm==4.64.0 colorama==0.4.5 Pygments==2.12.0 markdown==3.4.1 SentencePiece==0.1.96

# Finally, install DeepSeek-VL2
RUN pip install git+https://github.com/deepseek-ai/DeepSeek-VL2@main
Run pip install einops


