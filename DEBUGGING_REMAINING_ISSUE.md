# Debugging: Why Noise Still Diverges

## Current Status

After applying the `use_jax=False` fix, the diagnostic still shows:
```
Noise diff: 4.411674e+00
```

This means the fix didn't work. Let me analyze why.

## Possible Causes

### Cause 1: Different num_bindings

The two models were trained separately:
- `sap_g1_model_fixed_sparse_nocompress.pkl` - trained with sparse implementation
- `sap_g1_model_orig.pkl` - trained with original implementation

**If they have different numbers of bindings**, then `reset()` will consume different amounts of random numbers:

```python
# In set_state() called by reset():
noise_vec = np.random.normal(loc=0., scale=sd, size=self.num_bindings)
```

If `net_sparse.num_bindings ≠ net_orig.num_bindings`, then:
- Sparse consumes `num_bindings_sparse` random numbers
- Original consumes `num_bindings_orig` random numbers
- Random states desynchronize

**Why they might differ**: Your fixes to the sparse implementation (masking matrix, filler lookup, non-binary mask) might have changed the binding structure.

**How to check**: Run `diagnose_random_state.py`:
```bash
python diagnose_random_state.py
```

Look for:
```
Sparse num_bindings: XXXXX
Original num_bindings: YYYYY
```

If they differ, that's the problem!

### Cause 2: Models Weren't Retrained After Fix

If the models were trained BEFORE applying the `use_jax=False` fix to the training script, they might have been saved with `use_jax=True` in their internal state.

When loaded, even after we set `net.use_jax = False`, other internal state might still reflect the JAX configuration.

**How to check**: In `diagnose_random_state.py`, check if random states match after reset.

**Solution**: Retrain both models with the fix applied.

### Cause 3: The Diagnostic Script Itself Has Issues

The diagnostic script `debug_iteration_by_iteration.py` manually generates noise:

```python
# Line 82-83 (sparse):
noise_sparse = np.sqrt(2 * net_sparse.T * net_sparse.dt) * np.random.randn(net_sparse.num_bindings)
noiseC_sparse = np.sqrt(net_sparse.scale_constants) * net_sparse.N2C(noise_sparse)

# Line 103-104 (original):
noise_orig = np.sqrt(2 * net_orig.T * net_orig.dt) * np.random.randn(net_orig.num_bindings)
noiseC_orig = np.sqrt(net_orig.scale_constants) * net_orig.N2C(noise_orig)
```

This is called **sequentially** within the same loop iteration:
1. Generate noise for sparse (consumes `num_bindings_sparse` random numbers)
2. Generate noise for original (consumes `num_bindings_orig` random numbers)

These draws are from the SAME global random stream, so they will naturally be different!

**The correct approach** would be to:
1. Save random state before sparse
2. Generate noise for sparse
3. Restore random state
4. Generate noise for original

OR just use the actual network methods:
```python
net_sparse.update_stateC()  # Uses actual noise generation
net_orig.update_stateC()
```

### Cause 4: use_jax Not Actually False

The `use_jax` attribute might not be getting set properly, or there might be cached JAX state.

**How to check**: Add debug print in the diagnostic script:
```python
print(f"Sparse use_jax: {net_sparse.use_jax}")
print(f"Original use_jax: {net_orig.use_jax if hasattr(net_orig, 'use_jax') else 'N/A'}")
```

## Recommended Action Plan

### Step 1: Run Comprehensive Diagnostic

```bash
python diagnose_random_state.py
```

This will tell us:
- ✓ Do num_bindings match?
- ✓ Is use_jax actually False?
- ✓ Do random states synchronize after reset()?

### Step 2: Based on Results

**If num_bindings differ**:
- This means the sparse model has a different structure than the original
- You need to either:
  a) Retrain the original model with the same fixes, OR
  b) Make the binding structures match by fixing the initialization code

**If random states don't synchronize after reset()**:
- There's something in reset() or set_state() still consuming different amounts
- Need to add debug logging to trace exactly what's consumed

**If everything matches but noise still differs**:
- The issue is in the diagnostic script itself (Cause 3)
- Should use actual network methods instead of manual computation

### Step 3: Fix the Diagnostic Script

Replace manual noise generation with actual network methods:

```python
# INSTEAD OF manually computing noise:
# noise_sparse = np.sqrt(2 * net_sparse.T * net_sparse.dt) * np.random.randn(net_sparse.num_bindings)

# USE the actual network method:
actC_before = net_sparse.actC.copy()
net_sparse.update_stateC()  # This calls add_noiseC() internally
actC_after = net_sparse.actC.copy()
delta = actC_after - actC_before
```

This ensures we're testing the ACTUAL code path, not a manual reimplementation.

## Quick Test

Try this simple test to see if random states are synchronized:

```python
import numpy as np
import only_gscnet_speedup_sap as gsc_sparse
import gsc as gsc_orig

net_sparse = gsc_sparse.load_model('sap_g1_model_fixed_sparse_nocompress.pkl')
net_orig = gsc_orig.load_model('sap_g1_model_orig.pkl')

if hasattr(net_sparse, 'use_jax'):
    net_sparse.use_jax = False

seed = 12345

# Test 1: Same number of bindings?
print(f"Sparse bindings: {net_sparse.num_bindings}")
print(f"Original bindings: {net_orig.num_bindings}")
print(f"Match: {net_sparse.num_bindings == net_orig.num_bindings}")

# Test 2: Same random consumption after reset?
np.random.seed(seed)
net_sparse.reset(mu=net_sparse.ep, sd=0.02)
r1 = np.random.random()

np.random.seed(seed)
net_orig.reset(mu=net_orig.ep, sd=0.02)
r2 = np.random.random()

print(f"Next random after sparse reset: {r1}")
print(f"Next random after orig reset: {r2}")
print(f"Match: {abs(r1 - r2) < 1e-15}")
```

If this shows they don't match, we've found the issue!
