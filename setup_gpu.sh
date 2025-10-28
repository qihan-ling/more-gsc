#!/bin/bash
# GPU Setup Script for GSC
# This script helps diagnose and fix CuPy installation issues

echo "======================================================================"
echo "GSC GPU Setup Diagnostic"
echo "======================================================================"

# Check CUDA version
echo ""
echo "[1/5] Checking CUDA version..."
if command -v nvcc &> /dev/null; then
    echo "  nvcc found:"
    nvcc --version | grep "release"
else
    echo "  nvcc not found (this is OK, checking nvidia-smi instead)"
fi

if command -v nvidia-smi &> /dev/null; then
    CUDA_VERSION=$(nvidia-smi | grep "CUDA Version" | sed 's/.*CUDA Version: \([0-9.]*\).*/\1/')
    echo "  CUDA Version from nvidia-smi: $CUDA_VERSION"

    # Determine major version
    CUDA_MAJOR=$(echo $CUDA_VERSION | cut -d. -f1)
    echo "  CUDA Major Version: $CUDA_MAJOR"
else
    echo "  ERROR: nvidia-smi not found!"
    exit 1
fi

# Check current CuPy installation
echo ""
echo "[2/5] Checking CuPy installation..."
CUPY_INSTALLED=$(python -c "import cupy; print(cupy.__version__)" 2>/dev/null)
if [ $? -eq 0 ]; then
    echo "  CuPy version: $CUPY_INSTALLED"
    CUPY_CUDA=$(python -c "import cupy; print(cupy.cuda.runtime.runtimeGetVersion())" 2>/dev/null)
    if [ $? -eq 0 ]; then
        CUPY_CUDA_MAJOR=$((CUPY_CUDA / 1000))
        echo "  CuPy compiled for CUDA: $CUPY_CUDA_MAJOR.x"
    else
        echo "  WARNING: Could not determine CuPy CUDA version"
    fi
else
    echo "  CuPy not installed"
    CUPY_INSTALLED=""
fi

# Check for version mismatch
echo ""
echo "[3/5] Checking for CUDA version mismatch..."
if [ -n "$CUPY_INSTALLED" ] && [ -n "$CUPY_CUDA_MAJOR" ] && [ -n "$CUDA_MAJOR" ]; then
    if [ "$CUPY_CUDA_MAJOR" != "$CUDA_MAJOR" ]; then
        echo "  ❌ MISMATCH DETECTED!"
        echo "     System CUDA: $CUDA_MAJOR.x"
        echo "     CuPy CUDA:   $CUPY_CUDA_MAJOR.x"
        echo ""
        echo "  This is causing the error:"
        echo "  'libnvrtc.so.$CUPY_CUDA_MAJOR.x: cannot open shared object file'"
        NEEDS_REINSTALL=1
    else
        echo "  ✓ Versions match (both CUDA $CUDA_MAJOR.x)"
        NEEDS_REINSTALL=0
    fi
else
    echo "  Unable to verify (CuPy may not be installed)"
    NEEDS_REINSTALL=1
fi

# Recommend fix
echo ""
echo "[4/5] Recommended action..."
if [ "$NEEDS_REINSTALL" -eq 1 ]; then
    echo "  You need to reinstall CuPy with the correct CUDA version."
    echo ""
    echo "  Run these commands:"
    echo "  ─────────────────────────────────────────────────────────────"
    if [ -n "$CUPY_INSTALLED" ]; then
        echo "  # Uninstall incorrect version"
        echo "  pip uninstall -y cupy-cuda${CUPY_CUDA_MAJOR}x cupy"
        echo ""
    fi
    echo "  # Install correct version for CUDA $CUDA_MAJOR.x"
    echo "  pip install cupy-cuda${CUDA_MAJOR}x"
    echo "  ─────────────────────────────────────────────────────────────"
else
    echo "  ✓ No action needed - CuPy is correctly installed!"
fi

# Test import
echo ""
echo "[5/5] Testing CuPy import..."
python -c "import cupy as cp; print(f'  ✓ CuPy imports successfully'); print(f'  Device: {cp.cuda.Device()}'); print(f'  Memory: {cp.cuda.Device().mem_info[1] / 1e9:.2f} GB')" 2>&1

if [ $? -eq 0 ]; then
    echo ""
    echo "======================================================================"
    echo "SUCCESS! GPU setup is working correctly."
    echo "You can now run: python cho_grammar1_gpu.py"
    echo "======================================================================"
else
    echo ""
    echo "======================================================================"
    echo "CuPy import failed. Please follow the recommended action above."
    echo "======================================================================"
    exit 1
fi
