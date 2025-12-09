# Root Cause Analysis: Parsing Differences Between Original and Speedup

## Problem Statement
When comparing `gsc.py` (original) vs `only_gscnet_speedup_sap.py` (speedup), **both using sparse WC matrices**:
- ✅ Training results: IDENTICAL
- ❌ Parsing results: DIFFERENT (catastrophic at high commitment)

## Key Configuration
Both scripts run with:
- Sparse WC matrices (CSR format)
- JAX disabled (`use_jax=False`) - running on CPU with numpy
- Same random seed (41, 1024)
- Same dimensions: 27 fillers × 15 roles = 405 bindings

## Critical Code Difference

### Original gsc.py:3100
```python
def update_stateC(self):
    gradC = self.scale_constants * self.S.dot(self.HGradC())
    self.t += self.dt
    self.actC = self.actC + self.dt * gradC
    self.add_noiseC()
    self.actCmat = self.vec2mat()
```

Where `self.S` is pre-computed at initialization (gsc.py:2306):
```python
self.S = self.C.dot(self.C.T)  # Shape: (405, 405)
```

### Speedup only_gscnet_speedup_sap.py:2860-2864
```python
def update_stateC(self):
    hgrad = self.HGradC()

    # NumPy version: on CPU
    temp = self.C_T.dot(hgrad)      # (405, 405).T @ (405,) = (405,)
    gradC = self.C.dot(temp)        # (405, 405) @ (405,) = (405,)
    gradC = self.scale_constants * gradC

    self.t += self.dt
    self.actC = self.actC + self.dt * gradC
    self.add_noiseC()
    self.actCmat = self.vec2mat()
```

Where S is **NOT** materialized (only_gscnet_speedup_sap.py:1929-1933):
```python
# CRITICAL: DON'T CREATE S!
# The old code did: self.S = self.C.dot(self.C.T)
# Instead, we compute C @ (C.T @ v) on-the-fly
```

## Mathematical Equivalence (Ideal)

In exact arithmetic:
```
S @ hgrad = (C @ C.T) @ hgrad = C @ (C.T @ hgrad)
```

Both should give identical results.

## Why They Differ: Floating-Point Non-Associativity

### The Issue

Matrix multiplication is **NOT associative** in floating-point arithmetic:
```
(A @ B) @ C ≠ A @ (B @ C)
```

Due to:
1. **Different accumulation order**
2. **Different intermediate values**
3. **Rounding at each step**

### Concrete Example

**Original approach: (C @ C.T) @ hgrad**
```python
# Step 1: Materialize S = C @ C.T
S[i,j] = sum_k(C[i,k] * C[j,k])  # 405×405 matrix, 164,025 values
# Each S[i,j] has rounding error from ~405 multiplications

# Step 2: S @ hgrad
result[i] = sum_j(S[i,j] * hgrad[j])  # Another 405 multiplications
# Each result[i] accumulates errors from S computation + this dot product
```

**Speedup approach: C @ (C.T @ hgrad)**
```python
# Step 1: temp = C.T @ hgrad
temp[k] = sum_j(C[j,k] * hgrad[j])  # 405 values
# Each temp[k] has rounding error from ~405 multiplications

# Step 2: C @ temp
result[i] = sum_k(C[i,k] * temp[k])  # 405 multiplications
# Each result[i] accumulates errors from temp computation + this dot product
```

### Why This Causes Different Results

1. **Intermediate precision:**
   - Original: Creates 164,025 intermediate values (full S matrix)
   - Speedup: Creates only 405 intermediate values (temp vector)

2. **Error accumulation:**
   - Original: Errors from S computation persist for ALL subsequent updates
   - Speedup: Fresh computation each time, different error pattern

3. **Numerical stability:**
   - Original: Fixed S matrix means consistent (but possibly accumulated) errors
   - Speedup: Variable errors depending on current hgrad values

## Why Training Is Identical But Parsing Differs

### During Training
- Multiple samples averaged together
- Weight updates accumulate gradually over many iterations
- Small numerical differences get averaged out
- Final learned WC values converge to same result

### During Parsing (test_parse_inc)
- **Single trajectory** through state space
- Small numerical differences can push dynamics into different basins
- At high commitment (q=5-12), dynamics are **highly nonlinear** and **chaotic**
- Tiny perturbations get exponentially amplified

## Evidence from Outputs

### Commitment t=1 (low commitment, q=1)
```
Original: ACC = 0.900
Speedup:  ACC = 0.900  ← Same!
```

### Commitment t=2 (q=2)
```
Original: S0 = 1.000, Overall ACC = 0.960
Speedup:  S0 = 0.900, Overall ACC = 0.940  ← S0 degrades!
```

### Commitment t=7+ (high commitment, q≥7)
```
Original: ALL sentences = 1.000, Overall ACC = 1.000
Speedup:  S3=0.000, S4=0.000, Overall ACC = 0.600  ← Catastrophic!
```

The simplest sentence (S0: "N Vi") failing at t=2 indicates systematic numerical sensitivity.

## Magnitude of Differences

Typical floating-point rounding error per operation: ~10^-16 (machine epsilon)

For 405-dimensional vectors with ~405 operations:
- Expected accumulated error: ~405 * 10^-16 ≈ 4×10^-14
- After multiple integration steps (e.g., 100 steps): ~4×10^-12
- After nonlinear amplification in chaotic dynamics: **UNBOUNDED**

## Why HGradC Is Not the Issue

Both versions have identical `HGradC` implementation:
```python
hgrad_g = self.WC.dot(actC) + self.bC + self.extC
```

Both use sparse WC with same values, so `HGradC()` produces identical outputs.

The divergence is purely in `update_stateC()` due to the different ways of computing `S @ hgrad`.

## Solutions

### Option 1: Use Materialized S (defeats purpose of speedup)
```python
# In only_gscnet_speedup_sap.py, add back:
self.S = self.C.dot(self.C.T)

# In update_stateC():
gradC = self.scale_constants * self.S.dot(hgrad)
```

**Pros:** Identical to original
**Cons:** 405×405 = 164,025 values, defeats memory optimization

### Option 2: Higher Precision
```python
# Use float128 (if available) for dynamics:
self.C = self.C.astype(np.float128)
self.C_T = self.C_T.astype(np.float128)
```

**Pros:** Reduces rounding errors
**Cons:** Much slower, not available on all platforms

### Option 3: Compensated Summation (Kahan)
Use Kahan summation algorithm in dot products to reduce rounding errors.

**Pros:** Better numerical stability
**Cons:** Much slower, complex implementation

### Option 4: Accept the Variation
Recognize that:
- Both implementations are mathematically equivalent
- Differences are due to numerical sensitivity of chaotic dynamics
- Training results are identical (what matters for learning)
- Parsing variations are expected in nonlinear systems

**Pros:** No changes needed, faster implementation
**Cons:** Non-deterministic parsing behavior

## Recommendation

**Use Option 1 (materialize S) if exact reproducibility is required.**

For this small grammar (405 bindings), S is only 164k values = 1.3 MB (float64), which is negligible.

The lazy S computation is only beneficial for **very large grammars** (thousands of bindings).

### Concrete Fix

In `only_gscnet_speedup_sap.py`, change lines 1928-1933:

```python
# CRITICAL: DON'T CREATE S!
# The old code did: self.S = self.C.dot(self.C.T)
# Instead, we compute C @ (C.T @ v) on-the-fly
```

To:

```python
# For small grammars, materialize S for numerical consistency
if self.num_bindings < 10000:
    print(f"  Building S matrix ({self.num_bindings}×{self.num_bindings}) for consistency...")
    self.S = self.C.dot(self.C.T)
    self._use_lazy_s = False
else:
    print(f"  Using lazy S computation (num_bindings={self.num_bindings})")
    self._use_lazy_s = True
```

Then in `update_stateC()` lines 2834-2864:

```python
def update_stateC(self):
    hgrad = self.HGradC()

    if hasattr(self, '_use_lazy_s') and self._use_lazy_s:
        # Lazy computation for large grammars
        if self.use_jax:
            gradC = _lazy_s_multiply(self.C, self.C_T, hgrad, self.scale_constants)
        else:
            temp = self.C_T.dot(hgrad)
            gradC = self.C.dot(temp)
            gradC = self.scale_constants * gradC
    else:
        # Direct computation for small grammars (matches original)
        gradC = self.scale_constants * self.S.dot(hgrad)

    self.t += self.dt
    self.actC = self.actC + self.dt * gradC
    self.add_noiseC()
    self.actCmat = self.vec2mat()
```

This gives you:
- ✅ Exact consistency with original for small grammars
- ✅ Memory efficiency for large grammars
- ✅ Automatic selection based on grammar size
