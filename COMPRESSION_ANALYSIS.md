# Compression Analysis: USE_COMPRESSED=True

## Summary
**You are absolutely correct to question this.** The compression scheme (dim_f=150, dim_r=60) **does NOT reduce the WC matrix size**, which means it provides **NO speed benefit** for your sparse matrix training. It only introduces approximation error.

## What Compression Actually Does

### WITHOUT Compression (dim_f=None, dim_r=None)
```python
encodings = {'similarity': sim}
# dim_f = num_fillers (full dimensionality)
# dim_r = num_roles (full dimensionality)
```

### WITH Compression (dim_f=150, dim_r=60)
```python
encodings = {
    'similarity': sim,
    'dim_f': 150,  # Compress filler encodings
    'dim_r': 60,   # Compress role encodings
}
```

## Key Insight: WC Size is NOT Affected

Looking at the code:

```python
self.binding_names = [f + bsep + r for r in self.role_names for f in self.filler_names]
self.num_bindings = len(self.binding_names)  # = num_fillers × num_roles

# WC is ALWAYS:
self.WC = sparse.dok_matrix((self.num_bindings, self.num_bindings))
# Size: 844,800 × 844,800 (same whether compressed or not!)
```

**The WC matrix size depends on `num_bindings`, which equals `num_fillers × num_roles`.**

Compression (dim_f, dim_r) only affects the **encoding matrices F and R**, not the number of distinct filler/role types.

## What Gets Compressed

### Filler Encoding Matrix F
- WITHOUT compression: `F` is (num_fillers × num_fillers)
- WITH compression: `F` is (num_fillers × 150) via random projection

### Role Encoding Matrix R
- WITHOUT compression: `R` is (num_roles × num_roles)
- WITH compression: `R` is (num_roles × 60) via random projection

### The Problem
These encoding matrices F and R are used in the **change-of-basis transformations** (C2N, N2C), but:
1. They're NOT the training bottleneck (WC operations are)
2. The WC matrix size stays the same
3. Compression introduces **approximation error** with **no speed benefit**

## Why Compression Was Added (Probably)

Looking at the code history, compression was likely added for **dense** implementations where:
- Memory for dense WC = (num_bindings)² × 8 bytes
- With compression, you could reduce num_bindings
- This made sense for GPU/JAX implementations

But you're using **sparse** WC, where:
- Memory is only ~2GB (with 25M non-zeros)
- The bottleneck is sparse matrix-vector products, not memory
- Compression doesn't reduce WC size anyway!

## Impact on Training

### Approximation Error
Random projection (Johnson-Lindenstrauss) preserves distances approximately:
```
||F_compressed(x) - F_compressed(y)||² ≈ ||F_full(x) - F_full(y)||²
```

But introduces:
- Reconstruction error
- Changed geometry of the representation space
- Potential impact on gradient flow

### Your Observation
> "I have run it in the toy grammar 1 case and unsurprisingly the performance worsened."

**This confirms compression hurts accuracy with no speed benefit for sparse training.**

## Recommendation

### **REMOVE COMPRESSION**

Change ALL training scripts from:
```python
USE_COMPRESSED = True  # ❌ WRONG for sparse

encodings = {
    'similarity': sim,
    'dim_f': 150,  # Remove
    'dim_r': 60,   # Remove
}
```

To:
```python
USE_COMPRESSED = False  # ✓ CORRECT for sparse

encodings = {
    'similarity': sim,
    # No dim_f or dim_r - use full dimensions
}
```

## Expected Results

### NO impact on speed
- WC size unchanged
- Sparse matrix operations unchanged
- Training time: same

### IMPROVED accuracy
- No approximation error from random projection
- Better gradient quality
- Should match toy grammar performance

## When Compression WOULD Help

Compression only makes sense if:
1. Using **dense** WC matrix (you're not)
2. AND need to reduce memory (you have 500GB)
3. AND willing to sacrifice accuracy

For sparse training on your grammar, compression is **pure downside**.

## Action Items

1. **Test without compression** on toy grammar to verify improvement
2. **Update all training scripts** to remove compression
3. **Re-run training** - should see better convergence
4. Speed will be the same (controlled by dt, num_trials, q_max)

---

## Technical Details: The Compression Scheme

The code uses random projection:

```python
def encode_symbols(num_symbols, coord='N', dp=0., dim=None, seed=None):
    if dim is None:
        dim = num_symbols  # Full dimensionality

    # Random projection matrix: num_symbols × dim
    F = np.random.randn(num_symbols, dim)

    # Normalize
    F = F / np.linalg.norm(F, axis=1, keepdims=True)

    return F
```

With compression (dim < num_symbols), this is a **lossy** dimensionality reduction.

Without compression (dim = num_symbols), F is square and approximately orthogonal.

For TPR training with sparse WC, you want the **lossless** version.
