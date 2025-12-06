# Critical Differences: gsc.py vs only_gscnet_speedup_sap.py

## Executive Summary

**Mystery**: The original `gsc.py` achieves **perfect parsing** (S3=1.0, S4=1.0 at all commitment levels), while the SAP version fails catastrophically (S3=0.2-0.5, S4=0.0 at high commitment).

This document identifies the critical code differences that may explain this performance gap.

---

## 1. Matrix Materialization: S = C @ C.T

### gsc.py (line 2306)
```python
self.S = self.C.dot(self.C.T)  # inverse of similarity matrix
```
- **Materializes the S matrix** and stores it in memory
- S shape: (num_bindings, num_bindings) = (405, 405) for G1
- Memory: ~1.3 MB for G1 (manageable)

### SAP (line 1909)
```python
# CRITICAL: DON'T CREATE S!
# The old code did: self.S = self.C.dot(self.C.T)
# With 1.5M bindings, S would be 1.5M × 1.5M = 18 TB!
#
# Instead, we compute C @ (C.T @ v) on-the-fly in update_stateC()
```
- **Never materializes S** - uses lazy evaluation
- Computes `C @ (C.T @ v)` on-the-fly instead of `S @ v`

**Impact**:
- Mathematically equivalent: `S @ v = (C @ C.T) @ v = C @ (C.T @ v)`
- BUT: Different numerical precision characteristics
- Lazy evaluation may accumulate rounding errors differently

---

## 2. Gradient Computation in update_stateC()

### gsc.py (line 3100)
```python
def update_stateC(self):
    gradC = self.scale_constants * self.S.dot(self.HGradC())

    self.t += self.dt
    self.actC = self.actC + self.dt * gradC
    self.add_noiseC()
```
**Key**: Uses pre-computed S matrix

### SAP (lines 2744-2757)
```python
def update_stateC(self):
    hgrad = self.HGradC()

    if self.use_jax:
        gradC = _lazy_s_multiply(
            self.C, self.C_T, hgrad, self.scale_constants
        )
    else:
        # NumPy version
        temp = self.C_T.dot(hgrad)
        gradC = self.C.dot(temp)
        gradC = self.scale_constants * gradC

    self.t += self.dt
    self.actC = self.actC + self.dt * gradC
    self.add_noiseC()
```
**Key**: Computes S @ hgrad as C @ (C.T @ hgrad) without materializing S

**Mathematical Analysis**:

For vector `v` (the HGradC):
- gsc.py: `scale_constants * (S @ v)`
- SAP: `scale_constants * (C @ (C.T @ v))`

If `scale_constants` is a scalar, these are identical.
If `scale_constants` is a vector (element-wise multiplication), the order matters:
- gsc.py: Element-wise multiply `scale_constants` with result of `S @ v`
- SAP: Same, element-wise multiply `scale_constants` with result of `C @ (C.T @ v)`

**Should still be mathematically equivalent**, BUT:
1. **Associativity of matrix multiplication** ensures same result
2. **Numerical precision** may differ due to operation order
3. **Floating point rounding errors** accumulate differently

---

## 3. The Gc Matrix (Bowl Term)

### gsc.py (line 2302)
```python
self.Gc = self.C.T.dot(self.C)
```
- Computes and stores Gc for bowl term calculations
- Used in `Hb()` and `HbGrad()` (neural coordinates)

### SAP (line 1901)
```python
print("DEBUG _add_change_of_basis_matrices")
# self.Gc = self.C.T.dot(self.C)
```
- **COMMENTED OUT** - Gc is never computed!
- This would break `Hb()` and `HbGrad()` if called
- BUT: `HGradC()` (conceptual coordinates) doesn't use Gc

**Impact**:
- Both use `HGradC()` which computes bowl term as:
  ```python
  hgrad_b = self.opts['bowl_strength'] * (self.opts['bowl_center'] - actC)
  ```
- This is the **simplified bowl term** in conceptual coordinates
- The missing Gc should NOT affect parsing dynamics if bowl_strength is used correctly

---

## 4. Change-of-Basis Matrix Computation

### gsc.py (lines 2293-2298)
```python
N = np.kron(self.R, self.F)
if N.shape[0] == N.shape[1]:
    C = np.linalg.inv(N)
else:
    C = np.linalg.pinv(N)
```
- Computes full N matrix
- Then inverts it (either inv or pinv)

### SAP (lines 1885-1889)
```python
# OPTIMIZATION: pinv(kron(R, F)) = kron(pinv(R), pinv(F))
R_pinv = np.linalg.pinv(self.R, rcond=1e-10)
F_pinv = np.linalg.pinv(self.F, rcond=1e-10)
C = np.kron(R_pinv, F_pinv)
```
- Uses Kronecker product property to avoid large matrix inversion
- Computes pinv of small R and F separately
- **Much faster** for large grammars
- Uses explicit `rcond=1e-10` tolerance

**Impact**:
- Different numerical tolerance in pinv
- gsc.py uses default `rcond` (depends on numpy version)
- SAP uses explicit `rcond=1e-10`
- **This could produce slightly different C matrices!**

**CRITICAL**: Even tiny differences in C would propagate through:
1. C is used in every gradient computation
2. Small C errors → errors in gradC → errors accumulate over time steps
3. Over 1000 training epochs, this could lead to different learned WC

---

## 5. Numerical Precision Chain

The complete chain of operations that could accumulate errors:

### Training (1000 epochs)
For each epoch:
```
For each word position:
    1. Run dynamics: actC evolves via gradC
    2. gradC depends on C @ (C.T @ HGradC)  [SAP]
       or S @ HGradC  [gsc.py]
    3. Compute cost gradients
    4. Update WC weights
```

### Small differences in C:
- gsc.py: C from `pinv(N)` with default rcond
- SAP: C from `kron(pinv(R, rcond=1e-10), pinv(F, rcond=1e-10))`

Even if `∆C = 1e-10`, after:
- 1000 epochs
- ~100 dynamics steps per epoch
- ~10,000 gradient applications
- `∆WC` could be significant!

### Small differences in gradient computation:
- gsc.py: `S @ v` as single matrix multiply
- SAP: `C @ (C.T @ v)` as two matrix multiplies

Floating point errors:
- Single multiply: one rounding operation
- Double multiply: two rounding operations
- Different sparsity patterns may amplify errors differently

---

## 6. Hypothesis: The Root Cause

### Theory A: Numerical Precision of C Matrix

The different `rcond` values in pinv could produce C matrices that are slightly different:

```python
# gsc.py (default rcond)
C_gsc = np.linalg.pinv(N)  # rcond = max(M, N) * eps

# SAP (explicit rcond)
C_sap = np.kron(
    np.linalg.pinv(R, rcond=1e-10),
    np.linalg.pinv(F, rcond=1e-10)
)
```

If default rcond is more permissive (larger), it may include more singular vectors, leading to better gradient flow.

### Theory B: S vs Lazy Evaluation

Even though mathematically equivalent, the operations differ:

**gsc.py**:
```python
gradC = scale_constants * (S @ hgrad)
# S is pre-computed and dense
# Single dense matrix-vector multiply
# Numerical properties of S are "frozen" at initialization
```

**SAP**:
```python
temp = C.T @ hgrad  # First multiply
gradC = C @ temp     # Second multiply
gradC = scale_constants * gradC
# C is re-used each time
# Two matrix-vector multiplies per step
# Accumulates rounding errors differently
```

### Theory C: Sparse vs Dense WC (Secondary)

We know sparse performs worse than dense SAP, but both dense SAP and sparse SAP are worse than gsc.py. So sparsity isn't the primary issue.

---

## 7. Experimental Verification

### Test 1: Check C Matrix Difference
```python
import gsc
import only_gscnet_speedup_sap as sap

# Create both networks with identical parameters
hg = gsc.HarmonicGrammar(pcfg=PCFG_G1, root='S', max_sent_len=5)
sim = hg.get_simlist(dp=0.0)

net_gsc = gsc.GscNet(hg=hg, encodings={'similarity': sim},
                     opts={'use_sparse_wc': False}, seed=1024)

net_sap = sap.GscNet(hg=hg, encodings={'similarity': sim},
                     opts={'use_sparse_wc': False, 'use_jax': False},
                     seed=1024)

# Compare C matrices
diff_C = net_gsc.C - net_sap.C
print(f"C max diff: {np.abs(diff_C).max()}")
print(f"C mean diff: {np.abs(diff_C).mean()}")
print(f"C relative diff: {np.abs(diff_C).max() / np.abs(net_gsc.C).max()}")
```

### Test 2: Compare S @ v vs C @ (C.T @ v)
```python
test_vec = np.random.randn(net_gsc.num_bindings)

# gsc.py method
result_gsc = net_gsc.S.dot(test_vec)

# SAP method
temp = net_sap.C_T.dot(test_vec)
result_sap = net_sap.C.dot(temp)

diff = result_gsc - result_sap
print(f"Gradient method diff: {np.abs(diff).max()}")
```

### Test 3: Training with Matched C
```python
# Force SAP to use same C as gsc.py
net_sap.C = net_gsc.C.copy()
net_sap.C_T = net_gsc.C.T.copy()

# Train both identically
for i in range(100):
    net_gsc.train2(train_opts={'num_epochs': 10})
    net_sap.train2(train_opts={'num_epochs': 10})

# Compare final WC
diff_WC = net_gsc.WC - net_sap.WC
print(f"WC diff after training: {np.abs(diff_WC).max()}")
```

---

## 8. Recommended Fix

### Option 1: Match gsc.py's Numerical Behavior Exactly

In SAP's `_add_change_of_basis_matrices()`:
```python
# Replace Kronecker decomposition with direct method
N = np.kron(self.R, self.F)
if N.shape[0] == N.shape[1]:
    C = np.linalg.inv(N)
else:
    C = np.linalg.pinv(N)  # Use default rcond like gsc.py
```

### Option 2: Materialize S for Small Grammars

In SAP's `__init__()`:
```python
if self.num_bindings < 1000:  # Small grammar
    self.S = self.C.dot(self.C.T)  # Materialize S
    self._use_materialized_S = True
else:  # Large grammar
    self._use_materialized_S = False

def update_stateC(self):
    if self._use_materialized_S:
        gradC = self.scale_constants * self.S.dot(self.HGradC())
    else:
        # Lazy evaluation for large grammars
        hgrad = self.HGradC()
        temp = self.C_T.dot(hgrad)
        gradC = self.scale_constants * self.C.dot(temp)
```

### Option 3: Match rcond Value

In SAP's `_add_change_of_basis_matrices()`:
```python
# Use numpy's default rcond calculation
R_pinv = np.linalg.pinv(self.R)  # Remove explicit rcond
F_pinv = np.linalg.pinv(self.F)  # Remove explicit rcond
C = np.kron(R_pinv, F_pinv)
```

---

## 9. Next Steps

1. **Verify C matrix difference** between gsc.py and SAP
2. **Test if materializing S improves SAP performance** for G1
3. **Check if rcond value affects results**
4. **Compare gradient numerical precision** over many time steps
5. **Profile where numerical errors accumulate** during training

---

## 10. Conclusion

The performance difference between gsc.py (perfect parsing) and SAP (failing) likely stems from **subtle numerical differences** in:

1. **C matrix computation** (different rcond tolerance)
2. **Gradient evaluation** (S @ v vs C @ (C.T @ v))
3. **Accumulated rounding errors** over 1000 training epochs

These small differences don't break the code, but **steer gradient descent to different local minima**, resulting in dramatically different learned weights.

**Recommendation**: For G1 grammar (405 bindings), materialize S just like gsc.py does. The memory cost is negligible (~1 MB) and eliminates one source of numerical divergence.
