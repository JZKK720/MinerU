# AMD ROCm Setup for MinerU - RX 7900 XTX Configuration

# AMD ROCm Setup for MinerU - Multi-GPU Configuration Guide

## ⚠️ CRITICAL: Windows Users - READ THIS FIRST!

**If you're on Windows with ROCm HIP SDK:**
- ✅ You MUST use WSL2 with Ubuntu 22.04
- ✅ You MUST install ROCm INSIDE WSL2 (not just Windows)
- ✅ Docker containers run Linux (via WSL2), not Windows

**👉 [Follow WSL2 Setup Guide First](./WSL2_SETUP.md)**

After completing WSL2 setup, return here for MinerU-specific configuration.

---

## Supported Hardware Configurations

### Configuration 1: RX 7900 XTX (Primary - Recommended)
- **GPU**: AMD Radeon RX 7900 XTX 24GB (External, RDNA 3.0 / gfx1100)
- **CPU**: AMD Ryzen 9 MAX AI 395
- **RAM**: 128GB
- **ROCm HIP SDK**: 7.1.1 for Windows
- **PyTorch**: 2.9.0+rocmsdk20251116
- **ROCm Version**: (7, 1)
- **Performance**: ~1.3s/iteration with optimizations

### Configuration 2: RX 7600 XT  
- **GPU**: AMD Radeon RX 7600 XT 16GB (Navi 33 / gfx1102)
- **Note**: Despite being marketed as RDNA 2.0, uses gfx1102 architecture
- **CPU**: AMD Ryzen 9 MAX AI 395 (same system as 7900 XTX)
- **ROCm HIP SDK**: 7.1.1 for Windows
- **PyTorch**: 2.9.0+rocmsdk20251116
- **Expected Performance**: Lower than 7900 XTX due to less VRAM
- **Use Case**: Budget-friendly alternative for smaller documents

### Configuration 3: Ryzen 9 HX370 with NPU M890 (⚠️ EXPERIMENTAL)
- **CPU**: AMD Ryzen 9 HX370
- **Integrated GPU**: AMD Radeon 8060S (RDNA 3.0-based / gfx1100)
- **NPU**: AMD XDNA M890 NPU
- **ROCm HIP SDK**: 6.4.2 for Windows (older than MAX AI 395)
- **Note**: vLLM/MinerU NPU support is **NOT GUARANTEED**. This config uses the integrated GPU only.
- **Performance**: Significantly lower than dedicated GPUs, vLLM engine disabled

## Hardware Configuration

## Critical Configuration Changes

### 1. Docker Compose Configuration (`compose.yaml`)

The compose file now supports **THREE GPU PROFILES**:

#### Profile 1: RX 7900 XTX (RDNA 3.0) - **RECOMMENDED**
```bash
docker compose --profile rdna3-7900xtx up -d
```

**Environment Variables:**
```yaml
environment:
  MINERU_MODEL_SOURCE: local
  HSA_OVERRIDE_GFX_VERSION: "11.0.0"      # RDNA 3.0
  PYTORCH_ROCM_ARCH: "gfx1100"            # RX 7900 XTX architecture
  TORCH_ROCM_AOTRITON_ENABLE_EXPERIMENTAL: "1"
  HIP_VISIBLE_DEVICES: "0"                # External GPU
  ROCM_HOME: "/opt/rocm"
  GPU_MODEL: "RX7900XTX"
```

**GPU Memory**: 24GB, utilization set to 85% (~20.4GB usable)

#### Profile 2: RX 7600 XT (gfx1102)
```bash
docker compose --profile rdna2-7600xt up -d
```

**Environment Variables:**
```yaml
environment:
  MINERU_MODEL_SOURCE: local
  HSA_OVERRIDE_GFX_VERSION: "11.0.2"      # gfx1102
  PYTORCH_ROCM_ARCH: "gfx1102"            # RX 7600 XT (Navi 33)
  TORCH_ROCM_AOTRITON_ENABLE_EXPERIMENTAL: "1"
  HIP_VISIBLE_DEVICES: "0"
  ROCM_HOME: "/opt/rocm"
  GPU_MODEL: "RX7600XT"
```

**GPU Memory**: 16GB, utilization set to 75% (~12GB usable, conservative)

#### Profile 3: Ryzen NPU M890 + 8060S (⚠️ EXPERIMENTAL)
```bash
docker compose --profile npu-m890 up -d
```

**Environment Variables:**
```yaml
environment:
  MINERU_MODEL_SOURCE: local
  HSA_OVERRIDE_GFX_VERSION: "11.0.0"      # For integrated 8060S
  PYTORCH_ROCM_ARCH: "gfx1100"
  TORCH_ROCM_AOTRITON_ENABLE_EXPERIMENTAL: "1"
  HIP_VISIBLE_DEVICES: "0"                # Integrated GPU
  ROCM_HOME: "/opt/rocm"
  GPU_MODEL: "Radeon8060S"
  NPU_ENABLED: "experimental"
```

**Memory**: Shared system memory, utilization 60%, **vLLM engine DISABLED**

#### AMD ROCm Environment Variables
```yaml
environment:
  MINERU_MODEL_SOURCE: local
  HSA_OVERRIDE_GFX_VERSION: "11.0.0"      # Required for RDNA 3.0 (gfx1100)
  PYTORCH_ROCM_ARCH: "gfx1100"            # RX 7900 XTX architecture
  TORCH_ROCM_AOTRITON_ENABLE_EXPERIMENTAL: "1"  # Enable Triton experimental features
  HIP_VISIBLE_DEVICES: "0"                # Use external GPU (RX 7900 XTX)
  ROCM_HOME: "/opt/rocm"
```

#### Device Passthrough (Replaces NVIDIA GPU Config)
```yaml
devices:
  - /dev/kfd:/dev/kfd    # Kernel Fusion Driver
  - /dev/dri:/dev/dri    # Direct Rendering Infrastructure
group_add:
  - video               # Video group access
  - render              # Render group access
```

**REMOVED**: NVIDIA-specific configuration
```yaml
# ❌ This was removed:
deploy:
  resources:
    reservations:
      devices:
        - driver: nvidia
```

### 2. Port Mappings (Anti-Conflict Configuration)
- **mineru-openai-server**: Host `31001` → Container `30000`
- **mineru-api**: Host `31002` → Container `8000`
- **mineru-gradio**: Host `31003` → Container `7860`

### 3. AMD ROCm Dockerfiles Created

**Three Dockerfiles for different GPU architectures:**

#### A. RDNA 3.0 (RX 7900 XTX) - `docker/china/amd-rocm.Dockerfile`
- Base: `rocm/pytorch:rocm6.2.4_ubuntu22.04_py3.10_pytorch_release_2.5.0`
- Architecture: gfx1100
- VRAM: 24GB
- Host ROCm: HIP SDK 7.1.1 (Ryzen 9 MAX AI 395)
- PyTorch: 2.9.0+rocmsdk20251116
- vLLM: Enabled with full optimization
- Model source: ModelScope (China)

#### B. Navi 33 (RX 7600 XT) - `docker/china/amd-rdna2.Dockerfile`
- Base: Same as RDNA 3.0
- Architecture: gfx1102 (NOT gfx1031!)
- VRAM: 16GB
- Host ROCm: HIP SDK 7.1.1 (Ryzen 9 MAX AI 395)
- PyTorch: 2.9.0+rocmsdk20251116
- vLLM: Enabled (may need tuning)
- Memory utilization: 75% (conservative)

#### C. NPU M890 + 8060S - `docker/china/amd-npu.Dockerfile`
- Base: Same as above
- Architecture: gfx1100 (integrated GPU)
- Host ROCm: HIP SDK 6.4.2 (Ryzen 9 HX370)
- Memory: Shared system RAM
- vLLM: **DISABLED** (not NPU-compatible)
- Status: ⚠️ EXPERIMENTAL

**Location**: `docker/china/amd-rocm.Dockerfile`

**Key Features**:
- Base image: `rocm/pytorch:rocm6.2.4_ubuntu22.04_py3.10_pytorch_release_2.5.0`
- Configured for RDNA 3.0 (gfx1100) architecture
- Includes vLLM with ROCm support
- Chinese font support (Noto CJK)
- Model source: ModelScope (China mirrors)

## ⚠️ CRITICAL: Required Manual Steps

### Step 1: Apply Triton Kernel Patches for AMD

The RX 7900 XTX (RDNA 3.0) has **no native MIOpen kernel** for Conv3D operations used in Qwen2-VL. You **MUST** apply custom Triton kernel patches.

**Reference**: [docs/zh/usage/acceleration_cards/AMD.md](../../docs/zh/usage/acceleration_cards/AMD.md)

#### Create patch directory structure:
```bash
mkdir -p docker/china/amd-rocm-patches
```

#### Required patch files:

**1. `docker/china/amd-rocm-patches/qwen2_vl_vision_kernels.py`**

Create the Triton kernel implementation (choose Solution 1 or 2 from AMD.md):
- **Solution 1**: Triton Conv3D kernel (~1.5-1.8s/it)
- **Solution 2**: GEMM-based kernel (~1.3s/it) - **Recommended for 7900 XTX**

**2. `docker/china/amd-rocm-patches/apply_patches.sh`**
```bash
#!/bin/bash
set -e

echo "Applying AMD ROCm patches for vLLM..."

# Find vLLM installation path
VLLM_PATH=$(python3 -c "import vllm; import os; print(os.path.dirname(vllm.__file__))")

if [ -z "$VLLM_PATH" ]; then
    echo "ERROR: vLLM not found!"
    exit 1
fi

echo "Found vLLM at: $VLLM_PATH"

# Copy Triton kernel
cp /workspace/amd-rocm-patches/qwen2_vl_vision_kernels.py \
   $VLLM_PATH/model_executor/models/

# Patch qwen2_vl.py to use custom kernels
sed -i '33a from .qwen2_vl_vision_kernels import triton_conv3d_patchify' \
   $VLLM_PATH/model_executor/models/qwen2_vl.py

# Apply forward() method patch
python3 /workspace/amd-rocm-patches/patch_qwen2_vl.py

echo "Patches applied successfully!"
```

**3. `docker/china/amd-rocm-patches/patch_qwen2_vl.py`**

Create this Python script to patch the `Qwen2VisionPatchEmbed` class (implementation from AMD.md Solution 2).

### Step 2: Fix mineru_vl_utils for LoRA Error

After running for the first time, you may encounter a LoRA tokenizer error. Apply this fix:

```bash
# Find installation path
pip show mineru_vl_utils

# Edit: <PATH>/mineru_vl_utils/vlm_client/vllm_async_engine_client.py
# Line 58, replace:
#   self.tokenizer = vllm_async_llm.tokenizer.get_lora_tokenizer()
# With:
try:
    self.tokenizer = vllm_async_llm.tokenizer.get_lora_tokenizer()
except AttributeError:
    self.tokenizer = vllm_async_llm.tokenizer
```

### Step 3: DocLayout-YOLO Optimization (Optional but Recommended)

For layout detection speed improvements (1.6it/s → 15it/s), apply patches to `doclayout-yolo`.

**Reference**: [DocLayout-YOLO Issue #120](https://github.com/opendatalab/DocLayout-YOLO/issues/120#issuecomment-3368144275)

```bash
pip show doclayout-yolo
# Follow the linked instructions to patch g2l_crm.py
```

## Building and Running

### Build AMD ROCm Images

#### RX 7900 XTX (RDNA 3.0) - Recommended
```bash
cd docker/china
docker build -f amd-rocm.Dockerfile -t mineru:amd-rdna3 .
```

#### RX 7600 XT (RDNA 2.0)
```bash
cd docker/china
docker build -f amd-rdna2.Dockerfile -t mineru:amd-rdna2 .
```

#### Ryzen NPU M890 + 8060S (Experimental)
```bash
cd docker/china
docker build -f amd-npu.Dockerfile -t mineru:amd-npu .
```

### Run Services

#### Option 1: RX 7900 XTX (Best Performance)
```bash
# Gradio UI (recommended for testing)
docker compose --profile rdna3-7900xtx up -d

# Check logs
docker logs -f mineru-gradio-7900xtx
```

#### Option 2: RX 7600 XT (Budget Option)
```bash
docker compose --profile rdna2-7600xt up -d
docker logs -f mineru-gradio-7600xt
```

#### Option 3: Ryzen NPU (Experimental, Low Performance)
```bash
docker compose --profile npu-m890 up -d
docker logs -f mineru-gradio-npu
```

### Update compose.yaml to use AMD image
Edit `docker/compose.yaml` and change:
```yaml
image: mineru:amd-rocm  # Change from mineru:latest
```

### Run Services
```bash
# Gradio UI (recommended for testing)
docker compose --profile gradio up -d

# API Server
docker compose --profile api up -d

# OpenAI-compatible server
docker compose --profile openai-server up -d
```

### Access URLs (All Profiles Use Same Ports)
- **Gradio UI**: http://localhost:31003
- **API Server**: http://localhost:31002  
- **OpenAI Server**: http://localhost:31001

## Performance Comparison

### RX 7900 XTX (RDNA 3.0) - 24GB VRAM
- **Layout Detection**: 15 it/s (with patches)
- **VLM Processing**: ~1.3s/iteration (GEMM optimization)
- **200-page PDF**: ~100 seconds (~1.99 it/s)
- **Status**: ✅ Fully tested and optimized

### RX 7600 XT (gfx1102 / Navi 33) - 16GB VRAM
- **Expected Performance**: 60-75% of RX 7900 XTX
- **VRAM Limitation**: May struggle with large documents
- **Architecture Note**: Despite RDNA 2.0 marketing, uses gfx1102
- **Status**: ⚠️ Requires testing, same patches should work

### Ryzen NPU M890 + Radeon 8060S (Integrated)
- **Performance**: Significantly lower (CPU-like speeds)
- **vLLM**: Disabled (not compatible)
- **Use Case**: Testing only, not recommended for production
- **Status**: ⚠️ EXPERIMENTAL - NPU not utilized by MinerU

### Access URLs
- **Gradio UI**: http://localhost:31003
- **API Server**: http://localhost:31002
- **OpenAI Server**: http://localhost:31001

## Troubleshooting

### Check ROCm GPU Access
```bash
# On host - should show your GPU(s)
rocm-smi

# In container
docker exec -it mineru-gradio-7900xtx rocm-smi  # For 7900 XTX
docker exec -it mineru-gradio-7600xt rocm-smi   # For 7600 XT
docker exec -it mineru-gradio-npu rocm-smi      # For NPU/8060S
```

### Verify GPU Architecture
```bash
rocminfo | grep gfx
# RX 7900 XTX should show: gfx1100
# RX 7600 XT should show: gfx1102
# Radeon 8060S should show: gfx1100
```

### GPU Selection Issues
If you have multiple GPUs, ensure the correct one is selected:

```bash
# List all GPUs
rocm-smi --showproductname

# Set specific GPU for RX 7900 XTX (usually device 0 for external)
# Already configured in compose.yaml as HIP_VISIBLE_DEVICES: "0"

# If 7900 XTX is not device 0, check with:
rocm-smi --showid
# Then update HIP_VISIBLE_DEVICES in compose.yaml accordingly
```

### Check ROCm GPU Access
```bash
# On host
rocm-smi

# In container
docker exec -it mineru-gradio rocm-smi
```

### Verify GPU Architecture
```bash
rocminfo | grep gfx
# Should show: gfx1100
```

### Check vLLM Performance
Expected speeds for RX 7900 XTX:
- **Without patches**: Extremely slow (12s per iteration) ❌
- **With Solution 1 patches**: ~1.5-1.8s per iteration ✅
- **With Solution 2 patches**: ~1.3s per iteration ✅✅

### Common Issues

1. **"No GPU detected"**
   - Check `rocm-smi` on host
   - Verify `/dev/kfd` and `/dev/dri` exist
   - Check user is in `video` and `render` groups
   - Ensure correct HIP_VISIBLE_DEVICES value

2. **"VRAM out of memory"**
   - **RX 7900 XTX (24GB)**: Lower `--gpu-memory-utilization` from 0.85 to 0.75
   - **RX 7600 XT (16GB)**: Lower from 0.75 to 0.6 or 0.5
   - **Radeon 8060S (Integrated)**: Lower from 0.6 to 0.4
   - Reduce batch sizes or process smaller documents

3. **Slow inference (>5s/it)**
   - Triton patches not applied correctly
   - Check vLLM installation path
   - Rebuild container after applying patches
   - Verify correct GPU architecture (gfx1100 vs gfx1031)

4. **LoRA tokenizer error**
   - Apply mineru_vl_utils patch from Step 2

5. **NPU not detected / not utilized**
   - **Expected behavior**: MinerU doesn't support NPU directly
   - NPU config uses integrated GPU (8060S) only
   - True NPU support requires framework modifications
   - Consider using RX 7900 XTX or 7600 XT instead

6. **Wrong GPU being used (multi-GPU systems)**
   - Check `rocm-smi --showid`
   - Update `HIP_VISIBLE_DEVICES` in compose.yaml
   - External RX 7900 XTX is typically device 0
   - Integrated 8060S may be device 1

### Common Issues

1. **"No GPU detected"**
   - Check `rocm-smi` on host
   - Verify `/dev/kfd` and `/dev/dri` exist
   - Check user is in `video` and `render` groups

2. **"VRAM out of memory"**
   - Add `--gpu-memory-utilization 0.5` to command
   - The 7900 XTX has 24GB, should be sufficient

3. **Slow inference (>5s/it)**
   - Triton patches not applied correctly
   - Check vLLM installation path
   - Rebuild container after applying patches

4. **LoRA tokenizer error**
   - Apply mineru_vl_utils patch from Step 2

## Performance Notes

### GPU Comparison Table

| GPU Model | Architecture | VRAM | vLLM Support | Expected Speed | Status |
|-----------|-------------|------|--------------|----------------|--------|
| **RX 7900 XTX** | RDNA 3.0 (gfx1100) | 24GB | ✅ Full | ~1.3s/it | ✅ Tested |
| **RX 7600 XT** | Navi 33 (gfx1102) | 16GB | ⚠️ Likely | ~1.8-2.5s/it | ⚠️ Untested |
| **Radeon 8060S** | RDNA 3.0 (gfx1100) | Shared | ❌ Disabled | >5s/it | ⚠️ Experimental |

### RX 7900 XTX Optimization (from AMD.md testing)

**Layout Detection (DocLayout-YOLO)**:
- Unoptimized: 1.6 it/s
- With Triton patches: 15 it/s (9.4x speedup)

**VLM Processing**:
- Unoptimized: ~12s/it (unusable)
- Solution 1 (Triton Conv3D): 1.5-1.8s/it
- Solution 2 (GEMM-based): 1.3s/it

**200-page PDF**:
- Processing speed: ~1.99 it/s
- Total time: ~100 seconds

### Memory Usage
- RX 7900 XTX: 24GB VRAM
- Recommended allocation: 80-90% (19-22GB)
- Use `--gpu-memory-utilization 0.8` for optimal performance

## Next Steps

### Quick Start Checklist

**For RX 7900 XTX (Recommended):**
1. ✅ Docker compose configured for AMD ROCm
2. ✅ Dockerfile created for RDNA 3.0
3. ⚠️  **TODO**: Create and apply Triton kernel patches
4. ⚠️  **TODO**: Build image: `docker build -f docker/china/amd-rocm.Dockerfile -t mineru:amd-rdna3 docker/china/`
5. ⚠️  **TODO**: Run: `docker compose --profile rdna3-7900xtx up -d`
6. ⚠️  **TODO**: Test with sample PDF

**For RX 7600 XT:**
1. ✅ Docker compose configured
2. ✅ Dockerfile created for RDNA 2.0
3. ⚠️  **TODO**: Apply same Triton patches (may need tuning)
4. ⚠️  **TODO**: Build image: `docker build -f docker/china/amd-rdna2.Dockerfile -t mineru:amd-rdna2 docker/china/`
5. ⚠️  **TODO**: Run: `docker compose --profile rdna2-7600xt up -d`
6. ⚠️  **TODO**: Test and adjust memory utilization

**For Ryzen NPU M890 (Experimental):**
1. ✅ Docker compose configured
2. ✅ Dockerfile created (GPU-only, no true NPU support)
3. ⚠️  **NOTE**: This uses integrated GPU only, performance will be poor
4. ⚠️  **TODO**: Build image: `docker build -f docker/china/amd-npu.Dockerfile -t mineru:amd-npu docker/china/`
5. ⚠️  **TODO**: Run: `docker compose --profile npu-m890 up -d`
6. ❌  **NOT RECOMMENDED** for production use

## Next Steps

1. ✅ Docker compose configured for AMD ROCm
2. ✅ Dockerfile created for RX 7900 XTX
3. ⚠️  **TODO**: Create and apply Triton kernel patches
4. ⚠️  **TODO**: Build AMD ROCm image
5. ⚠️  **TODO**: Test with sample PDF

## References

- [AMD ROCm Setup Guide](../../docs/zh/usage/acceleration_cards/AMD.md)
- [DocLayout-YOLO ROCm Patches](https://github.com/opendatalab/DocLayout-YOLO/issues/120#issuecomment-3368144275)
- [vLLM ROCm Documentation](https://docs.vllm.com.cn/en/latest/getting_started/installation/gpu.html#amd-rocm)
- [ROCm GPU Isolation](https://rocm.docs.amd.com/en/docs-6.2.4/conceptual/gpu-isolation.html)
