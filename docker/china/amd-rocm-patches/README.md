# AMD ROCm Triton Kernel Patches for MinerU

This directory contains performance patches for running MinerU with vLLM on AMD ROCm GPUs.

## 🎯 Purpose

AMD RDNA GPUs lack optimized MIOpen kernels for certain Conv3D operations in Qwen2-VL, causing severe performance degradation (~12s/iteration CPU fallback). These Triton kernel patches replace the problematic operations with GPU-optimized implementations.

## 📊 Performance Impact

| Configuration | Before Patches | After Patches | Speedup |
|---------------|---------------|---------------|---------|
| **RX 7900 XTX (gfx1100)** | 12s/it | **1.3s/it** | **9.2x** |
| **RX 7600 XT (gfx1102)** | 12s/it | ~1.8-2.5s/it* | ~5-7x* |

*Estimated based on GPU compute capabilities

## 📁 Files

- **`qwen2_vl_vision_kernels.py`** - Triton-optimized Conv3D kernels
  - Method 1: Direct Conv3D (1.5-1.8s/it)
  - Method 2: GEMM-based (1.3s/it) ⭐ **RECOMMENDED**

- **`patch_qwen2_vl.py`** - Python script to patch vLLM's Qwen2-VL model
  - Automatically locates vLLM installation
  - Creates backups before patching
  - Applies Method 2 (GEMM-based approach)

- **`apply_patches.sh`** - Bash script to apply all patches
  - Patches vLLM Qwen2-VL model
  - Fixes mineru_vl_utils LoRA tokenizer issue
  - Provides detailed status and instructions

- **`README.md`** - This file

## 🚀 Quick Start

### Option 1: Automatic (Recommended)

```bash
cd /path/to/MinerU/docker/china/amd-rocm-patches
chmod +x apply_patches.sh
./apply_patches.sh
```

### Option 2: Manual

```bash
# 1. Apply vLLM patches
python3 patch_qwen2_vl.py

# 2. Fix mineru_vl_utils (if needed)
# Edit line 58 in vllm_async_engine_client.py
# See "Manual Fix" section below
```

## 🔧 Manual Fix for mineru_vl_utils

If automatic patching fails, manually edit the file:

```bash
# Find the file
pip show mineru_vl_utils
# Look for "Location: /path/to/site-packages"

# Edit this file:
# /path/to/site-packages/mineru_vl_utils/vlm_client/vllm_async_engine_client.py
```

**Find line 58:**
```python
self.tokenizer = vllm_async_llm.tokenizer.get_lora_tokenizer()
```

**Replace with:**
```python
try:
    self.tokenizer = vllm_async_llm.tokenizer.get_lora_tokenizer()
except AttributeError:
    # If get_lora_tokenizer method doesn't exist, use original tokenizer
    self.tokenizer = vllm_async_llm.tokenizer
```

## ⚙️ Configuration

### Required Environment Variables

```bash
export TORCH_ROCM_AOTRITON_ENABLE_EXPERIMENTAL=1
export MINERU_MODEL_SOURCE=local
```

Add to your `~/.bashrc` or `~/.zshrc` for persistence.

### GPU-Specific Tuning (Advanced)

The GEMM kernel is tuned for **RX 7900 XTX (gfx1100)**. For other GPUs, you may need to adjust:

**RX 7600 XT (gfx1102) suggested config:**

Edit `qwen2_vl_vision_kernels.py` line 188:

```python
best_config = {
    'BLOCK_M': 64,   # Reduced from 128
    'BLOCK_N': 64,   # Reduced from 128
    'BLOCK_K': 32,   # Same
}
num_stages = 3      # Reduced from 4
num_warps = 4       # Reduced from 8
```

**To find optimal config for your GPU:**
1. Use `rocprof` to profile execution
2. Experiment with BLOCK sizes: [64, 128, 256]
3. Adjust num_warps: [4, 8, 16]
4. Test num_stages: [2, 3, 4, 5]

## 🧪 Testing

```bash
# Test with Gradio interface
mineru-gradio --server-name 0.0.0.0 --server-port 7860 --enable-vllm-engine true

# Monitor GPU usage
watch -n 1 rocm-smi

# Check performance
# Look for "Processed prompts" throughput in terminal output
```

**Expected output (RX 7900 XTX):**
```
Processed prompts: 100%|█████| 14/14 [00:18<00:00, 1.30s/it]
```

## 🔍 Troubleshooting

### Patch fails with "vLLM not found"

```bash
# Verify vLLM installation
python3 -c "import vllm; print(vllm.__version__)"

# If not installed:
pip install vllm
```

### Still slow after patching

1. **Verify patches applied:**
   ```bash
   python3 -c "import vllm.model_executor.models.qwen2_vl_vision_kernels; print('Patches OK')"
   ```

2. **Check environment variables:**
   ```bash
   echo $TORCH_ROCM_AOTRITON_ENABLE_EXPERIMENTAL  # Should show "1"
   ```

3. **Verify GPU is being used:**
   ```bash
   rocm-smi  # GPU memory should increase when running
   ```

### AttributeError: 'Tokenizer' object has no attribute 'get_lora_tokenizer'

This means the mineru_vl_utils fix wasn't applied. Manually apply the fix (see "Manual Fix" section above).

### Performance still poor on RX 7600 XT

Try the suggested tuning config for gfx1102 (see "GPU-Specific Tuning" section).

## 📚 References

- [MinerU AMD ROCm Guide](https://github.com/opendatalab/MinerU/blob/master/docs/zh/usage/acceleration_cards/AMD.md)
- [vLLM ROCm Installation](https://docs.vllm.com.cn/en/latest/getting_started/installation/gpu.html#amd-rocm)
- [Triton Language Documentation](https://triton-lang.org/)

## 🐛 Known Issues

1. **AMD autotune not effective**: Manual tuning required for optimal performance on different GPU models
2. **Flash Attention incompatible**: Do not use flash_attn with Triton backend (causes errors)
3. **ROCm version sensitivity**: Best results with ROCm 6.2+ or ROCm 7.0+

## 📝 Technical Details

### Why Conv3D is Slow on AMD

The Qwen2-VL model uses Conv3D with bfloat16 for patch embedding. AMD's MIOpen library lacks optimized kernels for this specific operation, causing fallback to CPU implementation (~12s/iteration).

### Method 1: Direct Conv3D (1.5-1.8s/it)

Implements Conv3D directly in Triton with manual loop unrolling and memory coalescing optimizations.

### Method 2: GEMM-based (1.3s/it) ⭐

Converts 5D Conv3D operation into 2D matrix multiplication (GEMM):
- Reshape input: `[N, C, D, H, W]` → `[N, C*D*H*W]`
- Reshape weights: `[C_out, C, KD, KH, KW]` → `[C_out, C*KD*KH*KW]`
- Perform: `output = input @ weights.T`

This leverages AMD's highly optimized hipBLAS (ROCm GEMM) operations, achieving ~1.3s/iteration on RX 7900 XTX.

### Remaining Bottlenecks

After optimization, the main bottlenecks are:
- **hipBLAS operations**: ~25% of execution time (already optimal)
- **vLLM Triton backend**: ~25% of execution time (requires upstream vLLM optimization)

Total optimized time split: 50% ROCm kernels, 50% vLLM framework overhead.

## 📄 License

These patches are provided under the same license as MinerU (Apache 2.0).

## 🤝 Contributing

If you find optimal configurations for other AMD GPUs, please contribute them back:

1. Test with your GPU
2. Document the configuration and performance
3. Submit a pull request or open an issue

---

**Questions?** Open an issue at: https://github.com/opendatalab/MinerU/issues
