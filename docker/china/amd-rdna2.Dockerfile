# Base image for AMD ROCm with vLLM support - RDNA 2.0 Architecture
# Optimized for RX 7600 XT (gfx1102) - 16GB VRAM
# Requires host with ROCm HIP SDK 7.1.1 for Windows (Ryzen 9 MAX AI 395)
# PyTorch: 2.9.0+rocmsdk20251116
# NOTE: Despite being marketed as RDNA 2.0, RX 7600 XT uses gfx1102 (Navi 33)
FROM rocm/pytorch:latest

# Set ROCm architecture for RX 7600 XT (gfx1102, NOT gfx1031!)
ENV PYTORCH_ROCM_ARCH="gfx1102"
ENV HSA_OVERRIDE_GFX_VERSION="11.0.2"
ENV TORCH_ROCM_AOTRITON_ENABLE_EXPERIMENTAL=1
ENV ROCM_HOME=/opt/rocm

# Install system dependencies and Chinese font support
RUN apt-get update && \
    apt-get install -y \
        fonts-noto-core \
        fonts-noto-cjk \
        fontconfig \
        libgl1 \
        libglib2.0-0 \
        git \
        curl \
        wget && \
    fc-cache -fv && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

# Upgrade pip
RUN python3 -m pip install -U pip -i https://mirrors.aliyun.com/pypi/simple

# Install PyTorch 2.9.0 with ROCm support matching host environment
# Host: PyTorch 2.9.0+rocmsdk20251116, ROCm HIP SDK 7.1.1
RUN python3 -m pip install --pre torch torchvision \
    -i https://pypi.tuna.tsinghua.edu.cn/simple/ \
    --extra-index-url https://download.pytorch.org/whl/nightly/rocm6.2

# Install vLLM with ROCm support for gfx1102
WORKDIR /workspace
RUN git clone --recursive https://github.com/ROCm/aiter.git && \
    cd aiter && \
    git submodule sync && git submodule update --init --recursive && \
    python3 setup.py develop && \
    cd ..

RUN git clone https://github.com/vllm-project/vllm.git && \
    cd vllm && \
    cp -r /opt/rocm/share/amd_smi /workspace/vllm/ && \
    python3 -m pip install amd_smi/ && \
    python3 -m pip install --upgrade numba \
        scipy \
        "huggingface-hub[cli,hf_transfer]" \
        setuptools_scm && \
    python3 -m pip install -r requirements/rocm.txt && \
    python3 setup.py develop

# Install MinerU with core dependencies
RUN python3 -m pip install -U "mineru[core]>=2.7.0" \
    numpy==1.26.4 \
    opencv-python==4.11.0.86 \
    -i https://mirrors.aliyun.com/pypi/simple && \
    python3 -m pip cache purge

# Copy AMD ROCm optimization patches (Triton kernels for Conv3D)
# Note: RDNA 2.0 may have different performance characteristics than RDNA 3.0
# The same patches should work, but may need tuning
COPY amd-rocm-patches/ /workspace/amd-rocm-patches/

# Apply AMD-specific patches for vLLM Qwen2-VL model
RUN if [ -f /workspace/amd-rocm-patches/apply_patches.sh ]; then \
        bash /workspace/amd-rocm-patches/apply_patches.sh; \
    fi

# Download models from ModelScope (China mirror)
RUN /bin/bash -c "mineru-models-download -s modelscope -m all"

# Set working directory
WORKDIR /workspace

# Set environment variables for runtime
ENV MINERU_MODEL_SOURCE=local
ENV HIP_VISIBLE_DEVICES=0

# Note: RX 7600 XT (gfx1102) has 16GB VRAM vs RX 7900 XTX's 24GB
# Recommend using --gpu-memory-utilization 0.75 or lower

# Entrypoint
ENTRYPOINT ["/bin/bash", "-c", "exec \"$@\"", "--"]

# Default command
CMD ["mineru-gradio", "--server-name", "0.0.0.0", "--server-port", "7860", "--enable-vllm-engine", "true", "--gpu-memory-utilization", "0.75"]
