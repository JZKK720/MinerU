# MinerU Multi-GPU Quick Reference

## Your Hardware Options

### Option 1: RX 7900 XTX (24GB) - ⭐ BEST CHOICE
```bash
docker compose --profile rdna3-7900xtx up -d
```
- **Architecture**: RDNA 3.0 (gfx1100)
- **VRAM**: 24GB
- **Expected Speed**: ~1.3s/iteration
- **Status**: Fully tested and optimized

### Option 2: RX 7600 XT (16GB) - Budget Alternative
```bash
docker compose --profile rdna2-7600xt up -d
```
- **Architecture**: Navi 33 (gfx1102) - NOT gfx1031!
- **VRAM**: 16GB
- **Expected Speed**: ~1.8-2.5s/iteration (estimate)
- **Status**: Requires testing

### Option 3: Ryzen NPU M890 + 8060S - ⚠️ NOT RECOMMENDED
```bash
docker compose --profile npu-m890 up -d
```
- **Architecture**: Integrated GPU (gfx1100)
- **Memory**: Shared system RAM
- **Expected Speed**: Very slow (>5s/iteration)
- **Status**: Experimental, vLLM disabled

## Build Commands

```bash
cd docker/china

# Build for RX 7900 XTX
docker build -f amd-rocm.Dockerfile -t mineru:amd-rdna3 .

# Build for RX 7600 XT
docker build -f amd-rdna2.Dockerfile -t mineru:amd-rdna2 .

# Build for NPU (not recommended)
docker build -f amd-npu.Dockerfile -t mineru:amd-npu .
```

## Access URLs (All Configurations)
- Gradio UI: http://localhost:31003
- API Server: http://localhost:31002
- OpenAI Server: http://localhost:31001

## GPU Architecture Reference

| GPU | Architecture | gfx Version | HSA Override | ROCm HIP SDK |
|-----|-------------|-------------|--------------|---------------|
| RX 7900 XTX | RDNA 3.0 | gfx1100 | 11.0.0 | 7.1.1 (MAX AI 395) |
| RX 7600 XT | Navi 33 | gfx1102 | 11.0.2 | 7.1.1 (MAX AI 395) |
| Radeon 8060S | RDNA 3.0 (integrated) | gfx1100 | 11.0.0 | 6.4.2 (HX370) |

| GPU | Total VRAM | Recommended % | Usable Memory |
|-----|-----------|---------------|---------------|
| RX 7900 XTX | 24GB | 85% | ~20.4GB |
| RX 7600 XT | 16GB | 75% | ~12GB |
| Radeon 8060S | Shared | 60% | Variable |

## Important Notes

⚠️ **CRITICAL**: Before running ANY configuration, you MUST:
1. Create Triton kernel patches (see AMD_ROCM_SETUP.md)
2. Apply patches to vLLM
3. Fix mineru_vl_utils LoRA error
4. Build the Docker image for your GPU

📝 **Reference Documentation**:
- Full setup: [AMD_ROCM_SETUP.md](./AMD_ROCM_SETUP.md)
- Port changes: [SETUP_CHANGES.md](./SETUP_CHANGES.md)
- AMD optimization guide: [docs/zh/usage/acceleration_cards/AMD.md](../docs/zh/usage/acceleration_cards/AMD.md)

## Recommendation

**Use RX 7900 XTX** if available - it has:
- ✅ Most VRAM (24GB)
- ✅ Latest architecture (RDNA 3.0)
- ✅ Tested and documented optimizations
- ✅ Best performance (~1.3s/iteration)

The RX 7600 XT is a budget alternative but untested.
The NPU/8060S option is **not recommended** for actual use.
