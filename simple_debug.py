"""
Simple diagnostic - add this code right after net.initialize()
"""

# After calling net.initialize(train_opts=train_opts):
print("\n" + "="*70)
print("DIAGNOSTIC CHECKS")
print("="*70)

# Check 1: mask0
print("\n1. Checking mask0 (gradient mask):")
mask0 = net.train_opts['mask0']
if hasattr(net, 'use_sparse') and net.use_sparse:
    print(f"   mask0 is sparse")
    print(f"   Non-zero entries: {mask0.nnz:,}")
    print(f"   Total entries: {mask0.shape[0] * mask0.shape[1]:,}")
    print(f"   Percentage non-zero: {100.0 * mask0.nnz / (mask0.shape[0] * mask0.shape[1]):.4f}%")
    if mask0.nnz == 0:
        print("   ❌ PROBLEM FOUND: mask0 is empty! No gradients will be computed.")
        print("   This is why training isn't learning!")
else:
    import numpy as np
    nnz = np.count_nonzero(mask0)
    print(f"   mask0 shape: {mask0.shape}")
    print(f"   Non-zero entries: {nnz:,}")
    print(f"   Total entries: {mask0.size:,}")
    print(f"   Percentage non-zero: {100.0 * nnz / mask0.size:.4f}%")
    if nnz == 0:
        print("   ❌ PROBLEM FOUND: mask0 is all zeros! No gradients will be computed.")
        print("   This is why training isn't learning!")

# Check 2: Training options
print("\n2. Training options:")
print(f"   Learning rate: {net.train_opts['lrate']}")
print(f"   Optimizer: {net.train_opts['optimizer']}")
print(f"   update_w: {net.train_opts['update_w']}")
print(f"   update_gram_only: {net.train_opts['update_gram_only']}")
print(f"   Coefficients: {net.train_opts['coef']}")

# Check 3: Look for potential fixes
print("\n3. Potential fixes:")
if hasattr(net, 'use_sparse') and net.use_sparse:
    if mask0.nnz == 0:
        print("   Option A: Set update_gram_only=True (only update existing non-zero weights)")
        print("   Option B: Check if get_mask0() is working correctly for your grammar")
        print("   Option C: Disable masking by setting all mask entries to 1")

print("="*70)
