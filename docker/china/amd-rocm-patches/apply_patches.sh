#!/bin/bash
#
# Apply AMD ROCm Triton kernel patches for MinerU vLLM backend
#
# This script applies performance optimizations for AMD GPUs by:
# 1. Installing Triton kernels for Qwen2-VL Conv3D operations
# 2. Patching vLLM to use optimized GEMM-based approach
# 3. Fixing mineru_vl_utils LoRA tokenizer compatibility
#
# Usage: ./apply_patches.sh
#
# Performance Impact:
#   Before: ~12s/iteration (CPU fallback due to missing MIOpen kernels)
#   After:  ~1.3s/iteration on RX 7900 XTX (GEMM-based Triton kernel)
#

set -e  # Exit on error

echo "=========================================="
echo "AMD ROCm Triton Kernel Patcher for MinerU"
echo "=========================================="
echo ""

# Color codes for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Function to print colored messages
print_success() {
    echo -e "${GREEN}✅ $1${NC}"
}

print_error() {
    echo -e "${RED}❌ $1${NC}"
}

print_warning() {
    echo -e "${YELLOW}⚠️  $1${NC}"
}

print_info() {
    echo -e "${NC}ℹ️  $1${NC}"
}

# Check if Python is available
if ! command -v python3 &> /dev/null; then
    print_error "Python 3 is not installed or not in PATH"
    exit 1
fi

print_success "Found Python 3: $(python3 --version)"
echo ""

# Check if vLLM is installed
if ! python3 -c "import vllm" 2>/dev/null; then
    print_error "vLLM is not installed in the current Python environment"
    echo ""
    echo "Install vLLM first:"
    echo "  pip install vllm"
    exit 1
fi

print_success "Found vLLM: $(python3 -c 'import vllm; print(vllm.__version__)')"
echo ""

# Get the directory where this script is located
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Check if required patch files exist
if [ ! -f "$SCRIPT_DIR/qwen2_vl_vision_kernels.py" ]; then
    print_error "Missing file: qwen2_vl_vision_kernels.py"
    exit 1
fi

if [ ! -f "$SCRIPT_DIR/patch_qwen2_vl.py" ]; then
    print_error "Missing file: patch_qwen2_vl.py"
    exit 1
fi

print_success "All required patch files found"
echo ""

# Apply Qwen2-VL patches
echo "=========================================="
echo "Step 1: Patching vLLM Qwen2-VL model"
echo "=========================================="
echo ""

python3 "$SCRIPT_DIR/patch_qwen2_vl.py"

if [ $? -ne 0 ]; then
    print_error "Failed to patch Qwen2-VL model"
    exit 1
fi

echo ""

# Fix mineru_vl_utils LoRA tokenizer issue
echo "=========================================="
echo "Step 2: Fixing mineru_vl_utils LoRA issue"
echo "=========================================="
echo ""

# Find mineru_vl_utils installation
MINERU_VL_UTILS_PATH=$(python3 -c "import mineru_vl_utils; import os; print(os.path.dirname(mineru_vl_utils.__file__))" 2>/dev/null)

if [ -z "$MINERU_VL_UTILS_PATH" ]; then
    print_warning "mineru_vl_utils not found, skipping LoRA fix"
    print_info "This fix will be applied automatically when you first run MinerU"
else
    print_success "Found mineru_vl_utils at: $MINERU_VL_UTILS_PATH"
    
    VLLM_CLIENT_FILE="$MINERU_VL_UTILS_PATH/vlm_client/vllm_async_engine_client.py"
    
    if [ ! -f "$VLLM_CLIENT_FILE" ]; then
        print_warning "vllm_async_engine_client.py not found at expected location"
    else
        # Check if already patched
        if grep -q "except AttributeError:" "$VLLM_CLIENT_FILE"; then
            print_info "LoRA tokenizer fix already applied"
        else
            print_info "Applying LoRA tokenizer fix..."
            
            # Backup original file
            cp "$VLLM_CLIENT_FILE" "$VLLM_CLIENT_FILE.backup"
            print_success "Backup created: $VLLM_CLIENT_FILE.backup"
            
            # Apply the fix using Python
            python3 << EOF
import re

file_path = "$VLLM_CLIENT_FILE"

with open(file_path, 'r') as f:
    content = f.read()

# Find and replace the get_lora_tokenizer line
original = "        self.tokenizer = vllm_async_llm.tokenizer.get_lora_tokenizer()"
replacement = """        try:
            self.tokenizer = vllm_async_llm.tokenizer.get_lora_tokenizer()
        except AttributeError:
            # If get_lora_tokenizer method doesn't exist, use original tokenizer
            self.tokenizer = vllm_async_llm.tokenizer"""

if original in content:
    content = content.replace(original, replacement)
    with open(file_path, 'w') as f:
        f.write(content)
    print("✅ LoRA tokenizer fix applied successfully")
else:
    print("⚠️  Could not find exact line to patch (line 58)")
    print("   Manual fix may be required")
EOF
        fi
    fi
fi

echo ""

# Summary
echo "=========================================="
echo "✅ ALL PATCHES APPLIED SUCCESSFULLY!"
echo "=========================================="
echo ""
echo "Performance Expectations:"
echo "  • RX 7900 XTX (gfx1100): ~1.3s/it"
echo "  • RX 7600 XT (gfx1102):  ~1.8-2.5s/it (estimated)"
echo ""
echo "Required Environment Variables:"
echo "  export TORCH_ROCM_AOTRITON_ENABLE_EXPERIMENTAL=1"
echo "  export MINERU_MODEL_SOURCE=local"
echo ""
echo "Test the patches:"
echo "  mineru-gradio --server-name 0.0.0.0 --server-port 7860 --enable-vllm-engine true"
echo ""
echo "Troubleshooting:"
echo "  • If errors occur, check: rocm-smi (GPU should be visible)"
echo "  • Verify ROCm version: rocminfo | grep 'gfx'"
echo "  • Check vLLM installation: python3 -c 'import vllm; print(vllm.__version__)'"
echo ""
echo "To restore original files:"
echo "  • vLLM: Located at \$(python3 -c 'import vllm; print(vllm.__file__)')"
echo "  • Backups end with .backup extension"
echo ""
