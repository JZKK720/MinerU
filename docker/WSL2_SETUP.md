# WSL2 + ROCm Setup for MinerU on Windows

## ⚠️ CRITICAL: Architecture Requirements

**Your Windows Setup:**
- Windows 11/10 with WSL2
- WSL2 Distribution: Ubuntu 22.04
- Windows ROCm HIP SDK: 7.1.1 (MAX AI 395)

**What You Need:**
- ✅ WSL2 with Ubuntu 22.04 (you have this)
- ⚠️ **ROCm must be installed INSIDE WSL2**, not just Windows
- ⚠️ GPU passthrough from Windows → WSL2 → Docker

## Step-by-Step Setup

### Step 1: Verify WSL2 GPU Support

Open PowerShell and check WSL2 version:
```powershell
wsl --version
# Should show WSL version 2.x.x
```

Check if GPU is accessible in WSL2:
```powershell
wsl
# Inside WSL2:
ls -la /dev/dri
# Should show renderD128, card0, etc.
```

### Step 2: Install ROCm Inside WSL2

Open your WSL2 Ubuntu terminal:

```bash
# Update system
sudo apt update && sudo apt upgrade -y

# Add ROCm repository (for Ubuntu 22.04)
wget https://repo.radeon.com/rocm/rocm.gpg.key -O - | gpg --dearmor | sudo tee /etc/apt/keyrings/rocm.gpg > /dev/null

echo "deb [arch=amd64 signed-by=/etc/apt/keyrings/rocm.gpg] https://repo.radeon.com/rocm/apt/6.2.4 jammy main" | sudo tee /etc/apt/sources.list.d/rocm.list

sudo apt update

# Install ROCm for your GPU architecture
# For RX 7900 XTX (gfx1100) and RX 7600 XT (gfx1102):
sudo apt install -y rocm-hip-sdk rocm-libs

# Add user to video and render groups
sudo usermod -a -G video,render $USER

# Log out and back in, or run:
newgrp video
newgrp render
```

### Step 3: Verify ROCm Installation in WSL2

```bash
# Check ROCm version
/opt/rocm/bin/rocminfo

# Should show your GPUs:
# - RX 7900 XTX: gfx1100
# - RX 7600 XT: gfx1102

# Check with rocm-smi
/opt/rocm/bin/rocm-smi

# Set environment variables permanently
echo 'export ROCM_HOME=/opt/rocm' >> ~/.bashrc
echo 'export PATH=$ROCM_HOME/bin:$PATH' >> ~/.bashrc
source ~/.bashrc
```

### Step 4: Install Docker in WSL2

```bash
# Remove old Docker versions
sudo apt remove docker docker-engine docker.io containerd runc

# Install Docker
curl -fsSL https://get.docker.com -o get-docker.sh
sudo sh get-docker.sh

# Add user to docker group
sudo usermod -aG docker $USER
newgrp docker

# Verify Docker installation
docker --version

# Test Docker
docker run hello-world
```

### Step 5: Configure Docker for ROCm GPU Access

Create/edit Docker daemon config:
```bash
sudo mkdir -p /etc/docker
sudo nano /etc/docker/daemon.json
```

Add this configuration:
```json
{
  "runtimes": {
    "rocm": {
      "path": "/usr/bin/rocm-runtime",
      "runtimeArgs": []
    }
  },
  "default-runtime": "rocm"
}
```

Restart Docker:
```bash
sudo systemctl restart docker
```

### Step 6: Verify Docker GPU Access

Test GPU access in Docker:
```bash
docker run -it --rm \
  --device=/dev/kfd \
  --device=/dev/dri \
  --group-add video \
  --group-add render \
  rocm/pytorch:rocm6.2.4_ubuntu22.04_py3.10_pytorch_release_2.5.0 \
  rocm-smi
```

You should see your GPUs listed.

### Step 7: Clone MinerU Repository

```bash
cd ~
git clone https://github.com/opendatalab/MinerU.git
cd MinerU/docker/china
```

### Step 8: Build Docker Images

#### For RX 7900 XTX (gfx1100):
```bash
docker build -f amd-rocm.Dockerfile -t mineru:amd-rdna3 .
```

#### For RX 7600 XT (gfx1102):
```bash
docker build -f amd-rdna2.Dockerfile -t mineru:amd-rdna2 .
```

### Step 9: Run MinerU Services

Navigate to docker folder:
```bash
cd ~/MinerU/docker
```

#### For RX 7900 XTX:
```bash
docker compose --profile rdna3-7900xtx up -d
```

#### For RX 7600 XT:
```bash
docker compose --profile rdna2-7600xt up -d
```

### Step 10: Access from Windows

Services are accessible from Windows browser:
- **Gradio UI**: http://localhost:31003
- **API Server**: http://localhost:31002
- **OpenAI Server**: http://localhost:31001

## GPU Architecture Verification

Check which GPU Docker is using:
```bash
# Inside WSL2
docker exec -it mineru-gradio-7900xtx rocm-smi

# Should show your selected GPU
```

## Troubleshooting

### Issue 1: GPU Not Visible in WSL2

**Check Windows GPU support for WSL2:**
```powershell
# In PowerShell (as Administrator)
wsl --update
wsl --shutdown
wsl
```

**Verify in WSL2:**
```bash
ls -la /dev/dri
# Should show: card0, card1, renderD128, renderD129, etc.
```

### Issue 2: Multiple GPUs - Wrong GPU Selected

**List all GPUs:**
```bash
rocm-smi --showproductname
```

**Set specific GPU in compose.yaml:**
```yaml
environment:
  HIP_VISIBLE_DEVICES: "0"  # RX 7900 XTX (usually 0)
  # or
  HIP_VISIBLE_DEVICES: "1"  # RX 7600 XT or 8060S
```

### Issue 3: "No GPU detected" in Docker

**Verify device passthrough:**
```bash
docker run -it --rm \
  --device=/dev/kfd \
  --device=/dev/dri \
  --group-add video \
  --group-add render \
  ubuntu:22.04 \
  ls -la /dev/dri
```

Should show devices. If not, check user groups:
```bash
groups
# Should include: video, render, docker
```

### Issue 4: ROCm Version Mismatch

**Check WSL2 ROCm version:**
```bash
/opt/rocm/bin/rocminfo | grep "ROCm Version"
```

**Match with Docker base image:**
- Current Dockerfiles use ROCm 6.2.4
- If your WSL2 has different version, you may need to adjust

### Issue 5: Performance Issues

**Windows → WSL2 → Docker has overhead:**
- Expect 5-15% performance penalty vs native Linux
- Still much better than CPU-only
- RX 7900 XTX should still achieve ~1.5-1.8s/iteration

## Important Notes

### 🔴 Architecture Differences

| Component | Architecture | Notes |
|-----------|-------------|-------|
| **Windows Host** | x64 | Runs ROCm HIP SDK |
| **WSL2 Ubuntu** | x64 Linux | Runs ROCm natively |
| **Docker Container** | x64 Linux | Uses WSL2's ROCm |

### 🟡 GPU Passthrough Chain

```
Windows (ROCm HIP SDK)
    ↓
WSL2 Ubuntu (ROCm installation)
    ↓
Docker Container (/dev/kfd, /dev/dri)
    ↓
MinerU (PyTorch + vLLM)
```

### 🟢 ROCm Versions

- **Windows**: ROCm HIP SDK 7.1.1 (MAX AI 395)
- **WSL2**: Should install ROCm 6.2.4 (matches Docker base image)
- **Docker**: Uses ROCm 6.2.4 from base image

Version mismatch is usually fine, but stay close to avoid driver issues.

## Performance Expectations

### RX 7900 XTX via WSL2
- **Without patches**: Very slow
- **With Triton patches**: ~1.5-1.8s/iteration
- **Native Linux**: ~1.3s/iteration
- **WSL2 Overhead**: ~10-15%

### RX 7600 XT via WSL2
- Expected: ~2.0-2.5s/iteration (estimate)
- Not yet tested

## Next Steps

1. ✅ Verify WSL2 GPU access
2. ✅ Install ROCm in WSL2
3. ✅ Install Docker in WSL2
4. ⚠️ **TODO**: Build Docker images
5. ⚠️ **TODO**: Apply Triton kernel patches
6. ⚠️ **TODO**: Run and test

## References

- [WSL2 GPU Support](https://learn.microsoft.com/en-us/windows/wsl/tutorials/gpu-compute)
- [ROCm Installation Guide](https://rocm.docs.amd.com/en/latest/deploy/linux/install.html)
- [Docker GPU Access](https://docs.docker.com/config/containers/resource_constraints/#gpu)
- [MinerU AMD Setup](./AMD_ROCM_SETUP.md)

## Critical Reminder

⚠️ **You CANNOT use Windows ROCm HIP SDK directly from Docker**
✅ **You MUST install ROCm inside WSL2 Ubuntu**

The Windows ROCm HIP SDK is for native Windows applications, not for WSL2/Docker containers.
