# ✅ AMD ROCm Setup Complete - Final Status

## 📦 Created Files

### Docker Configuration
- ✅ [docker/docker-compose.yaml](./docker-compose.yaml) - Two GPU profiles (RX 7900 XTX, RX 7600 XT)
- ✅ [docker/china/amd-rocm.Dockerfile](./china/amd-rocm.Dockerfile) - RX 7900 XTX (gfx1100)
- ✅ [docker/china/amd-rdna2.Dockerfile](./china/amd-rdna2.Dockerfile) - RX 7600 XT (gfx1102)

### Triton Kernel Patches (9.2x Performance Boost)
- ✅ [docker/china/amd-rocm-patches/qwen2_vl_vision_kernels.py](./china/amd-rocm-patches/qwen2_vl_vision_kernels.py)
- ✅ [docker/china/amd-rocm-patches/patch_qwen2_vl.py](./china/amd-rocm-patches/patch_qwen2_vl.py)
- ✅ [docker/china/amd-rocm-patches/apply_patches.sh](./china/amd-rocm-patches/apply_patches.sh)
- ✅ [docker/china/amd-rocm-patches/README.md](./china/amd-rocm-patches/README.md)

### Documentation
- ✅ [docker/WSL2_SETUP.md](./WSL2_SETUP.md) - **START HERE**
- ✅ [docker/AMD_ROCM_SETUP.md](./AMD_ROCM_SETUP.md) - GPU configuration details
- ✅ [docker/MULTI_GPU_QUICK_REF.md](./MULTI_GPU_QUICK_REF.md) - Quick reference
- ✅ [docker/SETUP_CHANGES.md](./SETUP_CHANGES.md) - Port changes log

## 🚀 Quick Start (From WSL2)

### 1. Install ROCm in WSL2 Ubuntu (15-20 min)
```bash
# Follow WSL2_SETUP.md steps 1-3
wget https://repo.radeon.com/rocm/rocm.gpg.key -O - | gpg --dearmor | sudo tee /etc/apt/keyrings/rocm.gpg
echo "deb [arch=amd64 signed-by=/etc/apt/keyrings/rocm.gpg] https://repo.radeon.com/rocm/apt/6.2.4 jammy main" | sudo tee /etc/apt/sources.list.d/rocm.list
sudo apt update && sudo apt install -y rocm-hip-sdk rocm-libs
```

### 2. Clone Repository
```bash
cd ~
git clone https://github.com/JZKK720/MinerU.git
cd MinerU/docker/china
```

### 3. Build Docker Image (30-45 min)
```bash
# For RX 7900 XTX (recommended)
docker build -f amd-rocm.Dockerfile -t mineru:amd-rdna3 .

# OR for RX 7600 XT
docker build -f amd-rdna2.Dockerfile -t mineru:amd-rdna2 .
```

### 4. Run MinerU
```bash
cd ~/MinerU/docker

# RX 7900 XTX
docker compose --profile rdna3-7900xtx up -d

# OR RX 7600 XT
docker compose --profile rdna2-7600xt up -d
```

### 5. Access Services
- 🌐 Gradio UI: http://localhost:31003
- 🔌 API Server: http://localhost:31002
- 🤖 OpenAI Server: http://localhost:31001

## 📊 Expected Performance

| GPU | Architecture | Before Patches | After Patches | Speedup |
|-----|--------------|---------------|---------------|---------|
| **RX 7900 XTX** | gfx1100 (RDNA 3.0) | 12s/it | **1.3s/it** | **9.2x** |
| **RX 7600 XT** | gfx1102 (Navi 33) | 12s/it | ~1.8-2.5s/it | ~5-7x |

## 🔧 Configuration Summary

### RX 7900 XTX (gfx1100)
```yaml
Environment:
  HSA_OVERRIDE_GFX_VERSION: "11.0.0"
  PYTORCH_ROCM_ARCH: "gfx1100"
  GPU_MEMORY_UTILIZATION: 0.85  # 20.4GB / 24GB
  
Triton Tuning:
  BLOCK_M: 128, BLOCK_N: 128, BLOCK_K: 32
  num_stages: 4, num_warps: 8
```

### RX 7600 XT (gfx1102)
```yaml
Environment:
  HSA_OVERRIDE_GFX_VERSION: "11.0.2"
  PYTORCH_ROCM_ARCH: "gfx1102"
  GPU_MEMORY_UTILIZATION: 0.75  # 12GB / 16GB
  
Triton Tuning (suggested):
  BLOCK_M: 64, BLOCK_N: 64, BLOCK_K: 32
  num_stages: 3, num_warps: 4
```

## 🧪 Verification Steps

### 1. Verify ROCm Installation (In WSL2)
```bash
rocm-smi
# Should show your GPU(s)

rocminfo | grep "gfx"
# Should show gfx1100 or gfx1102
```

### 2. Verify Docker GPU Access
```bash
docker run --rm --device=/dev/kfd --device=/dev/dri rocm/pytorch:rocm6.2.4_ubuntu22.04_py3.10_pytorch_release_2.5.0 rocm-smi
# Should show your GPU(s) from inside container
```

### 3. Test MinerU Performance
```bash
# Upload a PDF in Gradio UI and watch terminal output
# Look for:
# "Processed prompts: 100%|█████| X/X [00:XX<00:00, 1.3s/it]"
```

## 🐛 Troubleshooting

### Docker build fails with "COPY amd-rocm-patches/: no such file or directory"
**Cause**: Building from wrong directory  
**Fix**: Must build from `docker/china/` directory where Dockerfile is located

```bash
cd ~/MinerU/docker/china
docker build -f amd-rocm.Dockerfile -t mineru:amd-rdna3 .
```

### GPU not detected in WSL2
```bash
# Check Windows-side GPU drivers
nvidia-smi  # Should show AMD GPU

# Verify WSL2 GPU passthrough
ls -la /dev/dri  # Should show card0, renderD128, etc.

# Reinstall ROCm in WSL2 if needed
sudo apt remove --purge rocm-hip-sdk
sudo apt autoremove
# Then reinstall (see WSL2_SETUP.md)
```

### Performance still slow after patches
```bash
# 1. Verify patches applied
docker exec -it mineru-gradio-7900xtx bash
python3 -c "import vllm.model_executor.models.qwen2_vl_vision_kernels; print('Patches OK')"

# 2. Check environment variables
env | grep TORCH_ROCM
# Should show: TORCH_ROCM_AOTRITON_ENABLE_EXPERIMENTAL=1

# 3. Monitor GPU during inference
watch -n 1 rocm-smi
# GPU memory should be utilized
```

### Port conflicts (31001, 31002, 31003 already in use)
```bash
# Find what's using the ports
netstat -tuln | grep -E "31001|31002|31003"

# Stop conflicting services or change ports in docker-compose.yaml
```

## 📚 Key Documentation Files

1. **[WSL2_SETUP.md](./WSL2_SETUP.md)** - Start here for Windows+WSL2 setup
2. **[AMD_ROCM_SETUP.md](./AMD_ROCM_SETUP.md)** - Detailed GPU configuration
3. **[amd-rocm-patches/README.md](./china/amd-rocm-patches/README.md)** - Triton kernel details
4. **[MULTI_GPU_QUICK_REF.md](./MULTI_GPU_QUICK_REF.md)** - Quick lookup reference

## 🎯 Next Steps

1. ✅ **You are here** - All configuration files created
2. ⏭️ Follow [WSL2_SETUP.md](./WSL2_SETUP.md) to install ROCm in WSL2
3. ⏭️ Clone repo in WSL2 and build Docker images
4. ⏭️ Run docker compose and test performance

## 🤝 Support

**Questions?** 
- Open issue: https://github.com/opendatalab/MinerU/issues
- Reference: [AMD ROCm Guide](https://github.com/opendatalab/MinerU/blob/master/docs/zh/usage/acceleration_cards/AMD.md)

---

**Configuration Date**: February 4, 2026  
**GPU Support**: RX 7900 XTX (gfx1100), RX 7600 XT (gfx1102)  
**ROCm Version**: 6.2.4 (Docker), 6.2+ or 7.0+ (Host WSL2)  
**Performance**: 9.2x speedup with Triton kernel optimizations
