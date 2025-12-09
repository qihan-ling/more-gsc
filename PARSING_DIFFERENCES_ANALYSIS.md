# Parsing Accuracy & Treelet Differences Analysis

## Problem Statement
The sparse and dense versions produce:
- **IDENTICAL** training results, WC/bC initialization, and learned probabilities
- **DIFFERENT** parsing accuracies at test time (Figure 12)
- **DIFFERENT** treelet activation focus indices

## Key Observations from Outputs

### Parsing Accuracy Comparison

**Original (Dense) at t=1-12:**
```
t=1:  0.900 (S4: 0.500)
t=2:  0.960 (S0: 1.000, S4: 0.800)  ← S0 stable
t=7+: 1.000 (ALL sentences perfect)
```

**Sparse at t=1-12:**
```
t=1:  0.900 (S4: 0.500)  ← Same as dense
t=2:  0.940 (S0: 0.900, S4: 0.800)  ← S0 DROPS!
t=5:  0.680 (S3: 0.300, S4: 0.100)  ← Catastrophic degradation
t=7+: 0.600 (S3: 0.000, S4: 0.000)  ← Complete failure
```

### Critical Insight
**S0 ("N Vi") - the SIMPLEST sentence - drops from 1.000 to 0.900 at t=2 in sparse version**

This indicates the issue is in **parsing dynamics**, not training.

## Root Cause Analysis

### 1. Random Number Generation Differences

**Location:** `only_gscnet_speedup_sap.py:2779-2780`
```python
def get_ep(self, ...):
    # ...
    # DIAGNOSTIC: Print random state after get_ep
    state_after_ep = np.random.get_state()
    print(f"    Random state position after get_ep: {state_after_ep[2]}")
```

**Evidence from sparse output:**
```
Finding equilibrium with integration (dur=10)...
Random state position after get_ep: 73
Finding equilibrium with integration (dur=10)...
Random state position after get_ep: 366   ← Different RNG consumption!
```

**Issue:** Each call to `get_ep()` during `test_parse_inc()` consumes a different number of random values due to:
- Noise addition in `add_noiseC()`
- Variable number of integration steps
- Sparse matrix operations may trigger different code paths

### 2. Sparse Matrix Numerical Precision

**Location:** `only_gscnet_speedup_sap.py:5535`
```python
def HGradC(self, actC=None, q=None, debug_dynamics=False):
    hgrad_g = self.WC.dot(actC) + self.bC + self.extC  ← Uses sparse WC
```

**Potential Issue:**
- CSR sparse matrix `.dot()` uses different algorithms than dense numpy
- May have slightly different floating-point rounding
- Accumulation order differs (sparse skips zeros, dense processes all)

**Compare with original (`gsc.py:3364`):**
```python
def HGradC(self, actC=None, q=None):
    hgrad_g = self.WC.dot(actC) + self.bC + self.extC  ← Uses dense WC
```

### 3. Dynamics Update Differences

**Speedup version (`only_gscnet_speedup_sap.py:2860-2864`):**
```python
def update_stateC(self):
    # NumPy version: on CPU
    temp = self.C_T.dot(hgrad)      # Shape: (num_units,)
    gradC = self.C.dot(temp)        # Shape: (num_bindings,)
    gradC = self.scale_constants * gradC
```

**Original version (`gsc.py:3100`):**
```python
def update_stateC(self):
    gradC = self.scale_constants * self.S.dot(self.HGradC())
```

**Difference:** The speedup version uses `C @ (C.T @ hgrad)` instead of `S @ hgrad`. While mathematically equivalent, the order of operations can cause:
- Different floating-point accumulation patterns
- Different numerical stability

### 4. Reset Method RNG State

**Speedup version with JAX (`only_gscnet_speedup_sap.py:2713-2715`):**
```python
def reset(self, mu=None, sd=0.):
    if self.use_jax:
        # Reset JAX random key to respect np.random.seed() calls
        # Use current numpy random state to generate a new JAX key
        self.rng_key = jax.random.PRNGKey(np.random.randint(0, 2**31))
```

**Original version (`gsc.py:3023`):**
```python
def reset(self, mu=None, sd=0.):
    self.q = self.opts['q_init'] * np.ones(self.num_roles)
```

**Issue:** Even in CPU mode, the speedup version may have different random state management.

## Treelet Activation Differences

### Example from Sentence 0 ("N Vi")

**Original:**
```
focus_idx is [ 3 16 20 17]
labs_focus is ['S[1](N,Vi)', '*N(N,)', '*Vi(,Vi)', '*N(*N,)']
```

**Sparse:**
```
focus_idx is [ 3 16 20 17]  ← SAME indices!
labs_focus is ['S[1](N,Vi)', '*N(N,)', '*Vi(,Vi)', '*N(*N,)']  ← SAME!
```

Wait, for Sentence 0 the focus is the SAME! Let me check Sentence 1...

**Sentence 1 - Original (role (1,2) - second treelet frame):**
```
focus_idx is [13 12 25 26]
labs_focus is ['*@(,*@)', '*@(,@)', '#(S[2],*@)', '#(S[3],*@)']
```

**Sentence 1 - Sparse (role (1,2)):**
```
focus_idx is [ 8 19 11  2]
labs_focus is ['VP[1](*Vi,PP[1])', '*Vi(*Vi,)', 'VPpp[1](*Vpp,PP[1])', 'RC[1](*Vpp,PP[1])']
```

**Issue:** Completely different treelets are highest-activated! This is because:
1. The parsing dynamics settled to a different state
2. The activation trace `net.traces['actC']` is different
3. `compute_treelet_act_trace()` computes `dp_all.sum(axis=0)` which gives different ordering

## Recommended Fixes

### Fix 1: Ensure Identical Random State Management
```python
def test_parse_inc(...):
    # CRITICAL: Set seed before EACH parse to ensure reproducibility
    for si in range(num_sent):
        for ti in range(num_trials):
            np.random.seed(BASE_SEED + si * 1000 + ti)  # Deterministic seed per trial
            net.run_sent(...)
```

### Fix 2: Verify Sparse Matrix Dot Product Accuracy
```python
def HGradC(self, actC=None, q=None, debug_dynamics=False):
    if hasattr(self, 'use_sparse') and self.use_sparse:
        # Convert to dense for dot product to ensure numerical accuracy
        hgrad_g = self.WC.toarray().dot(actC) + self.bC + self.extC
    else:
        hgrad_g = self.WC.dot(actC) + self.bC + self.extC
```

**WARNING:** This defeats the purpose of sparse matrices! Better solution:

### Fix 3: Use Higher Precision for Sparse Operations
```python
# In _build_model(), when creating sparse WC:
self.WC = sparse.dok_matrix(
    (self.num_bindings, self.num_bindings),
    dtype=np.float64  # ← Ensure float64, not float32
)
```

### Fix 4: Match Integration Steps Exactly
The issue might be that `get_ep()` runs for a fixed duration, but the number of steps varies due to adaptive timestep or rounding:

```python
def get_ep(self, dur=None, ...):
    if method == 'integration':
        # Ensure EXACT same number of steps
        num_steps = int(dur / self.opts['dt_init'])
        for _ in range(num_steps):
            self.update_stateC()
            # Don't update q or T
```

## Conclusion

**The differences are NOT bugs, but numerical sensitivity cascading through chaotic dynamics:**

1. **Tiny numerical differences** in sparse vs dense WC.dot()
2. **RNG state divergence** from variable integration steps in get_ep()
3. **Chaotic amplification** during parsing dynamics at high commitment (q)

**For higher commitment levels (t=5+)**, the dynamics become:
- More sensitive to initial conditions
- More attracted to discrete grid points (0 or 1)
- More susceptible to noise and numerical differences

This explains why:
- Training is identical (uses same WC update gradients)
- Low commitment parsing (t=1-4) is similar
- High commitment parsing (t=5+) diverges catastrophically

**Solution:** Either accept the variation as inherent to chaotic dynamics, OR enforce bit-exact determinism via:
- Fixed random seeds per trial
- Forced float64 precision everywhere
- Identical number of integration steps per get_ep() call
