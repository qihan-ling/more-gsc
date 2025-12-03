#!/usr/bin/env python3
"""
Test script to verify if there's a bug in sparse matrix submatrix extraction
using np.ix_() vs alternative methods.
"""

import sys
sys.path.insert(0, '/home/user/more-gsc')

import numpy as np
from scipy import sparse

print("="*70)
print("Testing sparse matrix submatrix extraction methods")
print("="*70)

# Create a test sparse matrix (simulating mask0)
print("\n1. Creating test mask0 matrix...")
n = 100
density = 0.1
np.random.seed(42)
mask0_dense = (np.random.rand(n, n) < density).astype(float)
mask0_csr = sparse.csr_matrix(mask0_dense)

print(f"   mask0 shape: {mask0_csr.shape}")
print(f"   mask0 nnz: {mask0_csr.nnz}")

# Simulate extracting a submatrix for a tree
key_idx = np.array([5, 12, 25, 37, 42, 58, 73, 89])  # Simulated binding indices
print(f"\n2. Extracting submatrix for key_idx (size={len(key_idx)})...")

# Method 1: Using np.ix_ (current code)
print("\n   Method 1: mask0[np.ix_(key_idx, key_idx)]")
try:
    submat1 = mask0_csr[np.ix_(key_idx, key_idx)]
    print(f"   Result shape: {submat1.shape}")
    print(f"   Result nnz: {submat1.nnz}")
    print(f"   Result sum: {submat1.sum()}")

    # Check if it's the same as dense version
    submat1_dense = mask0_dense[np.ix_(key_idx, key_idx)]
    print(f"   Dense equivalent nnz: {np.count_nonzero(submat1_dense)}")
    print(f"   Dense equivalent sum: {submat1_dense.sum()}")

    # Check if they match
    if sparse.issparse(submat1):
        match = np.allclose(submat1.toarray(), submat1_dense)
    else:
        match = np.allclose(submat1, submat1_dense)
    print(f"   Match with dense: {match}")

except Exception as e:
    print(f"   ERROR: {e}")

# Method 2: Alternative (rows first, then cols)
print("\n   Method 2: mask0[key_idx, :][:, key_idx]")
try:
    submat2 = mask0_csr[key_idx, :][:, key_idx]
    print(f"   Result shape: {submat2.shape}")
    print(f"   Result nnz: {submat2.nnz}")
    print(f"   Result sum: {submat2.sum()}")

    # Check if it matches dense
    if sparse.issparse(submat2):
        match2 = np.allclose(submat2.toarray(), submat1_dense)
    else:
        match2 = np.allclose(submat2, submat1_dense)
    print(f"   Match with dense: {match2}")

except Exception as e:
    print(f"   ERROR: {e}")

# Method 3: Convert to array for key positions
print("\n   Method 3: Using todense() then indexing")
try:
    submat3 = np.array(mask0_csr[key_idx, :][:, key_idx].todense())
    print(f"   Result shape: {submat3.shape}")
    print(f"   Result nnz: {np.count_nonzero(submat3)}")
    print(f"   Result sum: {submat3.sum()}")

    match3 = np.allclose(submat3, submat1_dense)
    print(f"   Match with dense: {match3}")

except Exception as e:
    print(f"   ERROR: {e}")

print("\n" + "="*70)
print("Conclusion:")
print("="*70)
if match and match2 and match3:
    print("✓ All methods produce identical results - no bug in indexing")
else:
    print("✗ METHODS DISAGREE - there IS a bug!")
    print("\n  This means sparse matrix indexing with np.ix_() is broken!")
    print("  This would cause incorrect gradient computation in sparse mode.")

print("\n")
