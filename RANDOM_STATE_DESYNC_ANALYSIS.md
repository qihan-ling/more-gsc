# Random State Desynchronization: Complete Analysis

## The Mystery

**Observation**: Training matches perfectly, but parsing diverges immediately at Word 1.

**Key Diagnostic Finding** (`debug_iteration_by_iteration.py`):
```
Step 1:
  Before:      1.29e-14 (match)
  hgrad diff:  6.03e-13 (match)
  gradC diff:  3.63e-12 (match)
  After grad:  2.28e-14 (match)
  Noise diff:  4.41 (HUGE!)       ← Divergence in noise
  After noise: 4.41 (divergence!)
```

Gradients match perfectly, but noise differs → Random state desynchronization.

## Root Cause: Dual Random Number Generators

### The Sparse Implementation Has Two Random Paths

**NumPy Random** (CPU, global state):
- Used when `use_jax=False`
- Controlled by `np.random.seed()`
- Global mutable state

**JAX Random** (GPU, functional):
- Used when `use_jax=True`
- Controlled by `self.rng_key`
- Immutable, requires explicit key passing

### The Original Implementation

**Only NumPy Random**:
- No JAX support at all
- Always uses global `np.random` state

## The Problem: Different Random Consumption

### During `reset(mu=..., sd=0.02)`:

**Sparse with `use_jax=True`**:
```python
# Line 2715 in reset():
self.rng_key = jax.random.PRNGKey(np.random.randint(0, 2**31))  # ← Consumes 1 NumPy random

# Line 2725 calls set_state(mu=mu, sd=sd):
# Line 2998-3000 in set_state():
self.rng_key, subkey = jax.random.split(self.rng_key)
noise_vec = jax.random.normal(subkey, ...) * sd  # ← Uses JAX random, NO NumPy consumption

# Total NumPy random numbers consumed: 1
```

**Sparse with `use_jax=False`**:
```python
# Line 2715 in reset():
# (not executed, use_jax=False)

# Line 2725 calls set_state(mu=mu, sd=sd):
# Line 3003-3004 in set_state():
noise_vec = np.random.normal(loc=0., scale=sd, size=self.num_bindings)  # ← Consumes num_bindings NumPy randoms

# Total NumPy random numbers consumed: num_bindings (could be 100,000+!)
```

**Original** (no JAX):
```python
# reset() doesn't have line 2715

# set_state() always uses NumPy:
noise_vec = np.random.normal(loc=0., scale=sd, size=self.num_bindings)

# Total NumPy random numbers consumed: num_bindings
```

## Why Training Matches

**Training configuration** (`sap_grammar_training_test2.py` line 53):
```python
'use_jax': False,  # Sparse only supported on CPU currently
```

During training:
- Sparse: `use_jax=False` → consumes `num_bindings` NumPy randoms per reset()
- Original: no JAX → consumes `num_bindings` NumPy randoms per reset()
- **SYNCHRONIZED!**

## Why Parsing Diverges

**Hypothesis**: When models are loaded for parsing, `use_jax` might become `True`.

Default behavior (line 1226):
```python
self.use_jax = JAX_AVAILABLE and self.opts.get('use_jax', True)
```

If JAX is available in the environment and the loaded model doesn't have `use_jax` explicitly set to False in opts:
- Sparse: `use_jax=True` → consumes only 1 NumPy random per reset()
- Original: no JAX → consumes `num_bindings` NumPy randoms per reset()
- **DESYNCHRONIZED!**

This desynchronization happens at EVERY reset() call, which includes:
1. Initial reset before parsing
2. Reset at the start of each trial

The offsets compound, causing massive divergence.

## The Fix

### Option 1: Force use_jax=False During Parsing (Quick Fix)

Ensure diagnostic and test scripts explicitly set `use_jax=False`:

```python
net_sparse = gsc_sparse.load_model('model.pkl')
net_sparse.use_jax = False  # Force NumPy random to match original
```

### Option 2: Fix reset() to Not Consume Random Numbers (Better)

The line at 2715 is problematic because it ties JAX random key initialization to NumPy's global state:

```python
# CURRENT (PROBLEMATIC):
self.rng_key = jax.random.PRNGKey(np.random.randint(0, 2**31))
```

**Why this line exists**: To make JAX respect `np.random.seed()` calls.

**Why it breaks**: Consumes different amounts of NumPy random depending on use_jax setting.

**Better approach**: Initialize rng_key once in `__init__`, then just use JAX's functional random:

```python
# In __init__:
if self.use_jax:
    seed = np.random.randint(0, 2**31)
    self.rng_key = jax.random.PRNGKey(seed)

# In reset() - REMOVE line 2715:
# Don't re-initialize rng_key, just keep using the functional JAX random
# JAX random is deterministic based on the key, no need to sync with NumPy
```

**Problem with this approach**: Breaks `np.random.seed()` reproducibility for JAX path.

### Option 3: Make Original Also Consume 1 Random Number (Hacky)

Add to original `gsc.py` reset():
```python
# Dummy consumption to match sparse implementation's line 2715
_ = np.random.randint(0, 2**31)
```

**Problem**: Hacky, pollutes original code.

### Option 4: Conditional Consumption Based on Context (Cleanest)

Make the random consumption in reset() match regardless of use_jax:

```python
# In reset(), REPLACE lines 2710-2717 with:
if self.use_jax:
    self.q = self.opts['q_init'] * jnp.ones(self.num_roles, dtype=jnp.float32)
else:
    self.q = self.opts['q_init'] * np.ones(self.num_roles)

# NEW: Always consume the same amount of NumPy random for sync
# If using JAX random for noise, consume placeholder to stay in sync with NumPy-only mode
if self.use_jax and mu is not None:
    # Will use JAX random in set_state, but need to consume num_bindings from NumPy to stay synced
    _ = np.random.normal(size=self.num_bindings)  # Consume but don't use
    # Initialize JAX key
    self.rng_key = jax.random.PRNGKey(np.random.randint(0, 2**31))
```

**Problem**: Wasteful, consumes random numbers unnecessarily.

## Recommended Solution

**For immediate fix**: Use Option 1 - explicitly set `use_jax=False` after loading models in test scripts.

**For long-term fix**: Redesign the random number system to either:
1. Always use NumPy random (remove JAX random entirely)
2. Always use JAX random (convert original to also use JAX)
3. Make the two implementations completely independent (don't try to synchronize random states)

## Verification

After applying fix, run `debug_iteration_by_iteration.py` and verify:
```
Step 1:
  Noise diff: ~0 (should be < 1e-10)
```

Then run full parsing tests and verify accuracy matches.
