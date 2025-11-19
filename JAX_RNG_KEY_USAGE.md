# JAX Random Key Usage in only_gscnet_speedup.py

## Overview

After JAX modifications for speedup, the network uses JAX's explicit random key (`self.rng_key`) instead of NumPy's implicit global RNG. This document explains the correct usage pattern and how `net.reset()` aligns with it.

## JAX RNG Pattern

JAX uses functional random number generation with explicit keys. The key pattern is:

```python
self.rng_key, subkey = jax.random.split(self.rng_key)
# Use subkey for random operation
value = jax.random.normal(subkey, shape=...)
```

This pattern:
1. **Splits** the key into two: new main key and a subkey
2. **Uses** the subkey for the random operation
3. **Updates** self.rng_key for future operations
4. **Ensures** each random call uses an independent stream

## Places Where rng_key is Split

### 1. `_dynamics_step_jax()` (line 649)
```python
rng_key, subkey = jax.random.split(rng_key)
noise = jnp.sqrt(2 * T * dt) * jax.random.normal(subkey, ...)
return actC_new, q_new, rng_key  # Return updated key
```
Used in JIT-compiled dynamics loop.

### 2. `add_noiseC()` (line 2224)
```python
self.rng_key, subkey = jax.random.split(self.rng_key)
noise = jnp.sqrt(2 * self.T * self.dt) * jax.random.normal(subkey, ...)
```
Called during `update_stateC()` when `log_trace=True`.

### 3. `set_random_state()` (line 2256)
```python
self.rng_key, subkey = jax.random.split(self.rng_key)
self.actC = jax.random.uniform(subkey, ...)
```
Called by `reset()` when `mu=None`.

### 4. `set_state()` (line 2431)
```python
self.rng_key, subkey = jax.random.split(self.rng_key)
noise_vec = jax.random.normal(subkey, ...) * sd
self.actC = mu + noise_vec
```
Called by `reset(mu, sd)` to add initialization noise.

## Places Where rng_key is Assigned

### 1. Initialization in `__init__()` (line 1049)
```python
self.rng_key = jax.random.PRNGKey(seed if seed is not None else 0)
```
Creates initial key from seed parameter.

### 2. After JIT-compiled loop in `runC()` (line 2120)
```python
init_carry = (self.actC, self.q, self.rng_key)
final_carry = jax.lax.fori_loop(0, num_steps, body_fun, init_carry)
self.actC, self.q, self.rng_key = final_carry
```
Updates key after fast JAX dynamics.

### 3. **In `reset()` (line 2162) - THE FIX**
```python
self.rng_key = jax.random.PRNGKey(np.random.randint(0, 2**31))
```
Creates fresh key from NumPy's RNG to respect `np.random.seed()` calls.

## Why the Fix is Correct

### Problem
- `cho_grammar1_fulljax.py` runs parsing tests that advance `self.rng_key`
- Then plotting calls `np.random.seed(1024 + sent_idx)` followed by `net.reset(mu, sd)`
- Without the fix, `reset()` didn't reset `rng_key`, so contaminated state was used
- The `sd=0.01` noise added in `set_state()` used the wrong random stream

### Solution
```python
# In reset() when self.use_jax:
self.rng_key = jax.random.PRNGKey(np.random.randint(0, 2**31))
```

This works because:
1. **Before reset()**: User calls `np.random.seed(X)` to set numpy's RNG
2. **In reset()**: Creates fresh JAX key using `np.random.randint()` which is deterministic given the seed
3. **After reset()**: `set_state(mu, sd)` splits this fresh key and generates noise
4. **Result**: Same seed → same JAX key → same noise → same dynamics → same results

### Alignment with JAX Patterns

The fix aligns with JAX best practices:
- ✓ Creates a fresh, independent key (not reusing contaminated state)
- ✓ Respects user's `np.random.seed()` calls for reproducibility
- ✓ Allows subsequent `split()` operations to work correctly
- ✓ No key reuse (each reset gets a new key)

## Testing

Run `test_rng_key_reset.py` to verify:
1. Same seed → same initial state (even after contamination)
2. Full pipeline reproducibility (parsing + plotting)
3. Correct top-4 treelet rankings

## Expected Behavior

**Debug script** (fresh load):
```
Top 4: [8, 19, 2, 11] = VP[1](*Vi,PP[1]), *Vi(*Vi,), RC[1](*Vpp,PP[1]), VPpp[1](*Vpp,PP[1])
```

**Plotting script** (after fix):
```
Same top 4: [8, 19, 2, 11]
```

Before the fix, plotting gave `[13, 12, 8, 25]` due to contaminated `rng_key`.
