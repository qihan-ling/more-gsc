# Fix for Multi-GPU Hang Issue

## The Problem

When `pmap` is created at module import time with 3 GPUs, JAX tries to:
1. Compile the entire function graph for 3 devices
2. Initialize NCCL (cross-GPU communication)
3. Allocate memory on all 3 GPUs
4. This can hang for hours or fail silently

## The Solution: Lazy Initialization

**DON'T create pmap at import time. Create it when first called.**

---

## Correct Implementation

### Replace the Multi-GPU Code with This:

**Around line 2398-2401 in gsc.py:**

```python
# Create batched version using vmap (single GPU - always works)
_run_trials_batched_jax_single = vmap(_run_single_trial_jax, in_axes=(0, None, None, None))

# Multi-GPU version will be created lazily on first call
_run_trials_batched_jax_multi = None
_multi_gpu_initialized = False

def _run_trials_batched_jax(rng_keys, net_params, prefix, update_q_discrete):
    """
    Wrapper that chooses single or multi-GPU execution.
    Multi-GPU version is initialized lazily on first call.
    """
    global _run_trials_batched_jax_multi, _multi_gpu_initialized

    num_devices = jax.local_device_count()

    if num_devices == 1:
        # Single GPU: use vmap
        return _run_trials_batched_jax_single(rng_keys, net_params, prefix, update_q_discrete)

    else:
        # Multi-GPU: use pmap (initialized lazily)
        if not _multi_gpu_initialized:
            print(f"Initializing multi-GPU mode for {num_devices} devices...")
            print("This may take 30-60 seconds on first run...")

            try:
                # Test basic pmap first
                print("Testing basic pmap...")
                test_fn = pmap(lambda x: x + 1)
                test_arr = jnp.ones((num_devices, 10))
                _ = test_fn(test_arr)
                print("Basic pmap test: SUCCESS")

                # Create the actual multi-GPU function
                print("Creating multi-GPU trial function...")
                _vmap_fn = vmap(_run_single_trial_jax, in_axes=(0, None, None, None))
                _run_trials_batched_jax_multi = pmap(_vmap_fn, in_axes=(0, None, None, None))
                _multi_gpu_initialized = True
                print(f"Multi-GPU initialization: COMPLETE ({num_devices} devices)")

            except Exception as e:
                print(f"Multi-GPU initialization FAILED: {e}")
                print("Falling back to single GPU mode")
                return _run_trials_batched_jax_single(rng_keys, net_params, prefix, update_q_discrete)

        # Use multi-GPU version
        return _run_trials_batched_jax_multi(rng_keys, net_params, prefix, update_q_discrete)
```

**Key changes:**
1. ✅ Single-GPU version always available
2. ✅ Multi-GPU version created **only when first called**
3. ✅ Includes error handling and fallback
4. ✅ Provides progress feedback

---

## Additional Required Change

**In estimate_prob_inc_jax() around line 5828-5833:**

You need to ensure random keys are properly shaped for multi-GPU:

```python
# Generate random keys for each trial
if rng_seed is None:
    rng_seed = np.random.randint(0, 1000000)
rng = jax.random.PRNGKey(rng_seed)

num_devices = jax.local_device_count()

if num_devices > 1:
    # Multi-GPU: ensure divisible by device count
    trials_per_device = (num_trials + num_devices - 1) // num_devices  # Round up
    total_trials = trials_per_device * num_devices

    if total_trials != num_trials:
        print(f"Adjusting num_trials: {num_trials} → {total_trials} (divisible by {num_devices} GPUs)")
        num_trials = total_trials

    rng_keys = jax.random.split(rng, num_trials)
    # Reshape to (num_devices, trials_per_device, 2)
    rng_keys = rng_keys.reshape(num_devices, trials_per_device, 2)
else:
    # Single GPU: no reshaping needed
    rng_keys = jax.random.split(rng, num_trials)

print(f"Running {num_trials} trials on {num_devices} device(s)...")

# Run all trials in parallel
actC_batch, grid_point_batch = _run_trials_batched_jax(
    rng_keys, net_params, prefix, update_q_discrete
)

# Reshape results if multi-GPU
if num_devices > 1:
    # Flatten from (num_devices, trials_per_device, ...) to (total_trials, ...)
    actC_batch = jnp.reshape(actC_batch, (-1,) + actC_batch.shape[2:])
    grid_point_batch = jnp.reshape(grid_point_batch, (-1,) + grid_point_batch.shape[2:])
```

---

## What Was Causing the Hang

### Original problematic code:
```python
# This runs at MODULE IMPORT TIME (when gsc.py is loaded)
_pmap_fn = pmap(_vmap_fn, in_axes=(0, None, None, None))  # ← HANGS HERE!
```

**Why it hangs:**
1. JAX tries to compile the entire function graph immediately
2. With 3 GPUs and complex nested vmap/pmap, compilation is huge
3. NCCL initialization for 3 GPUs can take forever or fail
4. No timeout, no error handling → appears frozen

### Fixed version:
```python
# Only creates pmap when FIRST CALLED (lazy initialization)
if not _multi_gpu_initialized:
    print("Initializing multi-GPU mode...")  # ← You see progress
    _run_trials_batched_jax_multi = pmap(_vmap_fn, ...)  # ← Happens later
```

---

## Testing After Fix

### Step 1: Test with single GPU first

```bash
CUDA_VISIBLE_DEVICES=0 python your_script.py
```

**Should complete in seconds and start training**

### Step 2: Test multi-GPU with small workload

```python
# Small test
train_opts = {
    'num_trials': 12,  # Small, divisible by 3 GPUs
    'num_epochs': 2,
}
```

**Expected output:**
```
Multi-GPU enabled: 3 devices
Initializing multi-GPU mode for 3 devices...
This may take 30-60 seconds on first run...
Testing basic pmap...
Basic pmap test: SUCCESS
Creating multi-GPU trial function...
Multi-GPU initialization: COMPLETE (3 devices)
Running 12 trials on 3 devices...
[Training starts...]
```

### Step 3: Full training run

Once small test works, run full training.

---

## Alternative: Use Environment Variable to Control

Add this at the start of your script:

```python
import os

# Set this to control GPU usage
USE_MULTI_GPU = False  # Set to True when you've verified it works

if not USE_MULTI_GPU:
    os.environ['CUDA_VISIBLE_DEVICES'] = '0'  # Force single GPU

import gsc
# ... rest of script
```

---

## Common Issues & Fixes

### Issue 1: NCCL Timeout

**Symptom**: Hangs during "Initializing multi-GPU mode"

**Fix**:
```bash
export NCCL_DEBUG=INFO  # See what NCCL is doing
export NCCL_TIMEOUT=300  # 5 minute timeout
```

### Issue 2: GPU Communication Failure

**Symptom**: Error about "NCCL" or "collective"

**Fix**: Check GPUs can communicate:
```bash
# Test GPU peer-to-peer
nvidia-smi topo -m

# Should show "PIX" or "NV" links between GPUs
# If "PHB", communication will be slow
```

### Issue 3: Out of Memory

**Symptom**: CUDA OOM error

**Fix**: Reduce num_trials per call:
```python
train_opts = {
    'num_trials': 300,  # Instead of 500
}
```

---

## Summary

**The hang is caused by eagerly creating pmap at import time.**

**Fix: Use lazy initialization** (create pmap only when first called)

**Test progression:**
1. Single GPU (CUDA_VISIBLE_DEVICES=0) - should work immediately
2. Multi-GPU with small workload (12 trials, 2 epochs) - verify no hang
3. Full training - should now work

**If still hangs**: Check NCCL_DEBUG=INFO output for communication issues
