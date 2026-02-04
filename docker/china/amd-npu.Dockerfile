# Base image for AMD NPU + Integrated GPU (Experimental)
# For Ryzen 9 HX370 with Radeon 8060S (Integrated) + NPU M890
# Requires host with ROCm HIP SDK 6.4.2 for Windows (Ryzen 9 HX370)
# ⚠️ WARNING: NPU support for vLLM/MinerU is EXPERIMENTAL and may not work
FROM rocm/pytorch:rocm6.2.4_ubuntu22.04_py3.10_pytorch_release_2.5.0

# Set ROCm architecture for integrated Radeon 8060S (RDNA 3.0-based)
ENV PYTORCH_ROCM_ARCH="gfx1100"
ENV HSA_OVERRIDE_GFX_VERSION="11.0.0"
ENV TORCH_ROCM_AOTRITON_ENABLE_EXPERIMENTAL=1
ENV ROCM_HOME=/opt/rocm

# NPU-specific environment variables (experimental)
ENV NPU_ENABLED="experimental"
ENV GPU_MODEL="Radeon8060S_Integrated"

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

# Install PyTorch with ROCm support
# Note: Ryzen 9 HX370 uses ROCm HIP SDK 6.4.2 (older than MAX AI 395's 7.1.1)
RUN python3 -m pip install --pre torch torchvision \
    -i https://pypi.tuna.tsinghua.edu.cn/simple/ \
    --extra-index-url https://download.pytorch.org/whl/nightly/rocm6.2

# NOTE: vLLM may not fully support NPU acceleration
# This build focuses on using the integrated GPU (8060S)
# NPU support would require additional frameworks like ONNX Runtime or vendor-specific SDKs

WORKDIR /workspace

# Install MinerU with core dependencies (simpler setup for NPU testing)
RUN python3 -m pip install -U "mineru[core]>=2.7.0" \
    numpy==1.26.4 \
    opencv-python==4.11.0.86 \
    -i https://mirrors.aliyun.com/pypi/simple && \
    python3 -m pip cache purge

# Download models from ModelScope (China mirror)
RUN /bin/bash -c "mineru-models-download -s modelscope -m all"

# Set working directory
WORKDIR /workspace

# Set environment variables for runtime
ENV MINERU_MODEL_SOURCE=local
ENV HIP_VISIBLE_DEVICES=0

# Note: Integrated GPU has shared system memory
# Performance will be significantly lower than dedicated GPUs
# Recommend disabling vLLM engine for better compatibility

# Entrypoint
ENTRYPOINT ["/bin/bash", "-c", "exec \"$@\"", "--"]

# Default command - vLLM disabled for NPU compatibility
CMD ["mineru-gradio", "--server-name", "0.0.0.0", "--server-port", "7860", "--enable-vllm-engine", "false", "--gpu-memory-utilization", "0.6"]
