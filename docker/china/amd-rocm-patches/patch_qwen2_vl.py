#!/usr/bin/env python3
"""
Patch script for Qwen2-VL model to use Triton-optimized Conv3D kernels.

This script modifies vLLM's Qwen2-VL implementation to use AMD ROCm-optimized
Triton kernels instead of the default Conv3D operations that cause severe
performance degradation on AMD GPUs.

Usage:
    python patch_qwen2_vl.py

The script will:
1. Locate your vLLM installation
2. Backup the original qwen2_vl.py file
3. Apply patches for Method 2 (GEMM-based, recommended)
4. Copy the Triton kernel file to vLLM's models directory

Requirements:
    - vLLM installed in the current Python environment
    - Write permissions to the vLLM installation directory
"""

import os
import shutil
import sys
from pathlib import Path


def find_vllm_models_dir():
    """Locate the vLLM models directory."""
    try:
        import vllm
        vllm_path = Path(vllm.__file__).parent
        models_dir = vllm_path / "model_executor" / "models"
        
        if not models_dir.exists():
            print(f"❌ ERROR: vLLM models directory not found at {models_dir}")
            return None
            
        print(f"✅ Found vLLM installation at: {vllm_path}")
        print(f"✅ Models directory: {models_dir}")
        return models_dir
    except ImportError:
        print("❌ ERROR: vLLM is not installed in this environment")
        print("   Install with: pip install vllm")
        return None


def backup_file(file_path):
    """Create a backup of the original file."""
    backup_path = f"{file_path}.backup"
    if not Path(backup_path).exists():
        shutil.copy2(file_path, backup_path)
        print(f"✅ Backup created: {backup_path}")
    else:
        print(f"ℹ️  Backup already exists: {backup_path}")
    return backup_path


def patch_qwen2_vl_imports(file_path):
    """Add import for Triton kernel at line 33."""
    with open(file_path, 'r', encoding='utf-8') as f:
        lines = f.readlines()
    
    # Check if already patched
    if any('qwen2_vl_vision_kernels' in line for line in lines):
        print("ℹ️  Import already patched")
        return False
    
    # Find the import section (around line 33)
    insert_index = None
    for i, line in enumerate(lines):
        if 'import torch.nn.functional as F' in line:
            insert_index = i + 1
            break
    
    if insert_index is None:
        print("⚠️  WARNING: Could not find import location, adding at line 33")
        insert_index = 33
    
    # Insert the import
    lines.insert(insert_index, "from .qwen2_vl_vision_kernels import triton_conv3d_patchify\n")
    
    with open(file_path, 'w', encoding='utf-8') as f:
        f.writelines(lines)
    
    print(f"✅ Added Triton kernel import at line {insert_index + 1}")
    return True


def patch_qwen2_vl_class(file_path):
    """Patch the Qwen2VisionPatchEmbed class to use Method 2 (GEMM-based)."""
    with open(file_path, 'r', encoding='utf-8') as f:
        content = f.read()
    
    # Check if already patched
    if 'triton_conv3d_patchify' in content and 'x_reshaped_5d' in content:
        print("ℹ️  Qwen2VisionPatchEmbed class already patched")
        return False
    
    # Find and replace the forward method
    original_forward = """    def forward(self, x: torch.Tensor) -> torch.Tensor:
        L, C = x.shape
        x = x.view(L, -1, self.temporal_patch_size, self.patch_size,
                   self.patch_size)
        x = self.proj(x).view(L, -1)
        return x"""
    
    patched_forward = """    def forward(self, x: torch.Tensor) -> torch.Tensor:
        L, C = x.shape
        x_reshaped_5d = x.view(L, -1, self.temporal_patch_size, self.patch_size,
                               self.patch_size)

        return triton_conv3d_patchify(x_reshaped_5d, self.proj.weight)"""
    
    if original_forward in content:
        content = content.replace(original_forward, patched_forward)
        
        with open(file_path, 'w', encoding='utf-8') as f:
            f.write(content)
        
        print("✅ Patched Qwen2VisionPatchEmbed.forward() to use GEMM-based Triton kernel (Method 2)")
        return True
    else:
        print("⚠️  WARNING: Could not find exact forward() method to patch")
        print("   Manual patching may be required")
        return False


def copy_triton_kernels(models_dir, script_dir):
    """Copy the Triton kernel file to vLLM models directory."""
    source = script_dir / "qwen2_vl_vision_kernels.py"
    destination = models_dir / "qwen2_vl_vision_kernels.py"
    
    if not source.exists():
        print(f"❌ ERROR: Triton kernel file not found at {source}")
        return False
    
    shutil.copy2(source, destination)
    print(f"✅ Copied Triton kernels to: {destination}")
    return True


def main():
    print("=" * 70)
    print("Qwen2-VL AMD ROCm Triton Kernel Patcher")
    print("Method: GEMM-based Conv3D (Method 2 - Recommended)")
    print("=" * 70)
    print()
    
    # Find vLLM installation
    models_dir = find_vllm_models_dir()
    if not models_dir:
        sys.exit(1)
    
    qwen2_vl_file = models_dir / "qwen2_vl.py"
    if not qwen2_vl_file.exists():
        print(f"❌ ERROR: qwen2_vl.py not found at {qwen2_vl_file}")
        sys.exit(1)
    
    print()
    print("-" * 70)
    print("Step 1: Backing up original file")
    print("-" * 70)
    backup_file(qwen2_vl_file)
    
    print()
    print("-" * 70)
    print("Step 2: Patching imports")
    print("-" * 70)
    patch_qwen2_vl_imports(qwen2_vl_file)
    
    print()
    print("-" * 70)
    print("Step 3: Patching Qwen2VisionPatchEmbed class")
    print("-" * 70)
    patch_qwen2_vl_class(qwen2_vl_file)
    
    print()
    print("-" * 70)
    print("Step 4: Copying Triton kernel file")
    print("-" * 70)
    script_dir = Path(__file__).parent
    copy_triton_kernels(models_dir, script_dir)
    
    print()
    print("=" * 70)
    print("✅ PATCHING COMPLETE!")
    print("=" * 70)
    print()
    print("Next Steps:")
    print("1. Set environment variable: export TORCH_ROCM_AOTRITON_ENABLE_EXPERIMENTAL=1")
    print("2. Test with: mineru-gradio --enable-vllm-engine true")
    print("3. Expected performance: ~1.3s/it on RX 7900 XTX")
    print()
    print("To restore original file:")
    print(f"  cp {qwen2_vl_file}.backup {qwen2_vl_file}")
    print()


if __name__ == "__main__":
    main()
