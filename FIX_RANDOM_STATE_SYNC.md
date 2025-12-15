# Fix for Parsing Accuracy Issue: Random State Synchronization

## Problem Summary

**Training**: Perfect match between sparse and original implementations ✓
**Parsing**: Divergence at commitment >= 5 for longer sentences ✗

## Root Cause Identified

The sparse implementation has **dual random number generators**:
1. **NumPy random** (CPU, global state) - used when `use_jax=False`
2. **JAX random** (GPU, functional) - used when `use_jax=True`

The original implementation only has NumPy random.

### The Critical Issue in reset()

**File**: `only_gscnet_speedup_sap.py`
**Lines**: 2715 and 2994-3007

When `reset(mu=..., sd=0.02)` is called:

**If `use_jax=True`** (sparse with JAX enabled):
```python
# Line 2715: Initialize JAX random key
self.rng_key = jax.random.PRNGKey(np.random.randint(0, 2**31))  # Consumes 1 NumPy random

# Line 2998-3000: Generate noise using JAX random
self.rng_key, subkey = jax.random.split(self.rng_key)
noise_vec = jax.random.normal(subkey, ...) * sd  # Uses JAX random, NO NumPy consumption

# Total NumPy consumption: 1 number
```

**If `use_jax=False`** (sparse without JAX, or original):
```python
# Line 2715: Skipped (use_jax is False)

# Line 3003-3004: Generate noise using NumPy random
noise_vec = np.random.normal(loc=0., scale=sd, size=self.num_bindings)  # Consumes num_bindings numbers

# Total NumPy consumption: num_bindings numbers (could be 100,000+!)
```

### Why Training Matched

Training configuration in `sap_grammar_training_test2.py` line 53:
```python
'use_jax': False,  # Sparse only supported on CPU currently
```

Both implementations used NumPy random → same consumption → synchronized!

### Why Parsing Diverged

When models are loaded, `use_jax` might default to `True` if JAX is available (line 1226):
```python
self.use_jax = JAX_AVAILABLE and self.opts.get('use_jax', True)
```

If sparse has `use_jax=True` and original has no JAX:
- Different NumPy random consumption per reset()
- Desynchronization at every trial
- Massive divergence in parsing results

## The Fix Applied

**Quick Fix**: Force `use_jax=False` after loading models to match training configuration.

### Files Modified:

1. **`sap_grammar_training_test2.py`** (line 241-247):
   ```python
   net = gsc.load_model('sap_g1_model_fixed_sparse_nocompress.pkl')

   # CRITICAL FIX: Force use_jax=False
   if hasattr(net, 'use_jax'):
       net.use_jax = False
       print("✓ Forced model to use_jax=False for random state synchronization")
   ```

2. **`compare_parsing_divergence.py`** (line 21-25)
3. **`debug_iteration_by_iteration.py`** (line 17-20)
4. **`debug_gradient_divergence.py`** (line 17-20)

## Testing the Fix

### Step 1: Verify Random State Synchronization

Run the iteration diagnostic:
```bash
python debug_iteration_by_iteration.py
```

**Expected output** (after fix):
```
Step 1:
  Before:      ~1e-14 (match)
  hgrad diff:  ~1e-12 (match)
  gradC diff:  ~1e-12 (match)
  After grad:  ~1e-14 (match)
  Noise diff:  ~1e-14 (MATCH!)  ← Should be near zero now
  After noise: ~1e-14 (match)
```

### Step 2: Verify Parsing Accuracy

Run the full parsing test:
```bash
python sap_grammar_training_test2.py
```

**Expected results** at commitment t=5:

| Sentence | Sparse (Before) | Sparse (After Fix) | Original |
|----------|-----------------|-------------------|----------|
| S0: N Vi | 100% | 100% | 100% |
| S1: N Vi P N | 100% | 100% | 100% |
| S2: N BE Vpp | 100% | 100% | 100% |
| S3: N BE Vpp P N | **30%** | **100%** ✓ | 100% |
| S4: N Vpp P N Vi | **10%** | **80%** ✓ | 80% |

**Overall accuracy**: Should match original at ~0.96 (was 0.68)

## Long-Term Solutions

### Option 1: Remove JAX Random Entirely

Make sparse implementation always use NumPy random:
```python
# Remove JAX random code paths from:
# - reset() line 2715
# - set_state() lines 2996-3000
# - add_noiseC() lines 2791-2795
```

**Pros**: Simple, guaranteed synchronization
**Cons**: Lose GPU acceleration for noise generation (minimal impact)

### Option 2: Save use_jax in Model File

Ensure `use_jax` setting is explicitly saved in opts and restored on load:
```python
# In save_model():
net.opts['use_jax'] = net.use_jax

# In load_model():
if 'use_jax' in net.opts:
    net.use_jax = net.opts['use_jax']
```

**Pros**: Preserves training configuration automatically
**Cons**: Requires modifying save/load logic

### Option 3: Separate Random Streams

Don't try to synchronize random states between implementations:
- Use different seeds for testing
- Compare statistical properties (accuracy over many trials) instead of exact trial-by-trial matching

**Pros**: More flexible, allows different implementations
**Cons**: Makes debugging harder, can't do exact comparison

## Recommended Approach

**For now**: Use the quick fix (force `use_jax=False`) ✓ **Applied**

**Long-term**: Option 2 (save use_jax in model file) for cleaner solution.

## Verification Commands

```bash
# 1. Check use_jax value in loaded model
python check_use_jax.py

# 2. Run iteration-by-iteration diagnostic
python debug_iteration_by_iteration.py

# 3. Run gradient comparison
python debug_gradient_divergence.py

# 4. Run full parsing test
python sap_grammar_training_test2.py

# 5. Compare sparse vs original parsing
python compare_parsing_divergence.py
```

## Success Criteria

✓ Noise diff in iteration diagnostic: < 1e-10
✓ S3 parsing accuracy at t=5: ~100% (was 30%)
✓ S4 parsing accuracy at t=5: ~80% (was 10%)
✓ Overall parsing accuracy: ~0.96 (was 0.68)

## Files Reference

- **Root cause**: `only_gscnet_speedup_sap.py:2715` (reset JAX key)
- **Root cause**: `only_gscnet_speedup_sap.py:2996-3000` (JAX noise in set_state)
- **Training config**: `sap_grammar_training_test2.py:53` (use_jax=False)
- **Default config**: `only_gscnet_speedup_sap.py:1226` (use_jax defaults to True)
- **Parsing tests**: `sap_grammar_training_test2.py:318-335`

## Next Steps

1. Run `debug_iteration_by_iteration.py` to verify noise synchronization
2. Run `sap_grammar_training_test2.py` to verify parsing accuracy
3. If tests pass, commit the fix
4. Consider implementing long-term solution (Option 2)
