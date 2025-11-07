# Multi-GPU Training Guide for GSC Parser

## Current Status: Single GPU

The current implementation uses `vmap` which parallelizes across trials on **one GPU**.

## Option 1: Automatic Multi-GPU (JAX Default) - ZERO CODE CHANGES ⭐

### Setup

JAX can automatically distribute computation across GPUs with environment variables:

```bash
# Check available GPUs
python -c "import jax; print(jax.devices())"

# Example output:
# [CudaDevice(id=0), CudaDevice(id=1), CudaDevice(id=2)]
```

### How It Works

JAX's JIT compiler may automatically distribute large arrays across GPUs when:
- Arrays exceed single GPU memory
- Operations are parallelizable

**To enable (try this first)**:

```bash
export XLA_FLAGS="--xla_gpu_autotune_level=2"
python cho_grammar1.py
```

### Expected Speedup

- **Minimal to None** for current code
- Why: `vmap` runs on single GPU by default

---

## Option 2: Explicit Data Parallelism with `pmap` - RECOMMENDED ⭐⭐⭐

### What Changes

Replace `vmap` (single GPU, parallel over trials) with `pmap` (multi-GPU, parallel over devices AND trials).

### Code Modifications

**File: gsc.py**

#### Change 1: Import pmap (line ~20)

```python
from jax import vmap, jit, pmap  # Add pmap
```

#### Change 2: Create multi-GPU batched function (line ~2400)

**OLD:**
```python
_run_trials_batched_jax = vmap(_run_single_trial_jax, in_axes=(0, None, None, None))
```

**NEW:**
```python
# Check available devices
_num_devices = jax.local_device_count()

def _run_trials_batched_jax_multi_gpu(rng_keys, net_params, prefix, update_q_discrete):
    """
    Distribute trials across multiple GPUs using pmap.

    rng_keys shape: (num_devices, trials_per_device, ...)
    """
    # Inner vmap: parallelize trials on each GPU
    _vmap_fn = vmap(_run_single_trial_jax, in_axes=(0, None, None, None))

    # Outer pmap: distribute across GPUs
    _pmap_fn = pmap(_vmap_fn, in_axes=(0, None, None, None))

    return _pmap_fn(rng_keys, net_params, prefix, update_q_discrete)

# Use appropriate version based on device count
if _num_devices > 1:
    print(f"Using {_num_devices} GPUs for training")
    _run_trials_batched_jax = _run_trials_batched_jax_multi_gpu
else:
    print("Using single GPU")
    _run_trials_batched_jax = vmap(_run_single_trial_jax, in_axes=(0, None, None, None))
```

#### Change 3: Reshape random keys for devices (line ~5828)

**OLD:**
```python
rng_keys = jax.random.split(rng, num_trials)  # Shape: (num_trials,)
```

**NEW:**
```python
num_devices = jax.local_device_count()

if num_devices > 1:
    # Ensure num_trials is divisible by num_devices
    trials_per_device = num_trials // num_devices
    num_trials_adjusted = trials_per_device * num_devices

    if num_trials_adjusted != num_trials:
        print(f"Adjusting num_trials: {num_trials} → {num_trials_adjusted} (divisible by {num_devices} GPUs)")

    rng_keys = jax.random.split(rng, num_trials_adjusted)
    # Reshape for pmap: (num_devices, trials_per_device)
    rng_keys = rng_keys.reshape(num_devices, trials_per_device, -1)
else:
    rng_keys = jax.random.split(rng, num_trials)
```

#### Change 4: Reshape results back (line ~5836)

**OLD:**
```python
actC_batch = np.array(actC_batch)  # Shape: (num_trials, num_bindings)
grid_point_batch = np.array(grid_point_batch)
```

**NEW:**
```python
if num_devices > 1:
    # Flatten from (num_devices, trials_per_device, ...) to (num_trials, ...)
    actC_batch = np.array(actC_batch).reshape(-1, actC_batch.shape[-1])
    grid_point_batch = np.array(grid_point_batch).reshape(-1, grid_point_batch.shape[-1])
else:
    actC_batch = np.array(actC_batch)
    grid_point_batch = np.array(grid_point_batch)
```

---

## Option 3: Model Parallelism - For Very Large Models

### When Needed

If WC matrix (21 GB for 1,756 rules) doesn't fit in **single GPU memory**.

### What Changes

Split the weight matrix WC across GPUs:
- GPU 0: WC rows 0-35,000
- GPU 1: WC rows 35,001-71,600

**This requires extensive code changes** - only pursue if single GPU has insufficient memory.

---

## Expected Speedup

### For 2 GPUs (Data Parallelism)

| Component | Speedup | Notes |
|-----------|---------|-------|
| GPU dynamics | **~1.9x** | Near-linear scaling |
| Post-processing | **1.0x** | Runs on CPU |
| Overall per-epoch | **~1.7x** | Weighted average |

**Training time reduction:**
- 21 days → **12 days** (with 2 GPUs)
- 21 days → **8 days** (with 3 GPUs)

### For 3 GPUs

| Component | Speedup |
|-----------|---------|
| GPU dynamics | **~2.7x** |
| Overall per-epoch | **~2.4x** |

---

## Memory Requirements

### Current (1,756 rules, max_len=20)

- WC matrix: **21 GB**
- Total GPU memory needed: **~25 GB** per GPU

### Multi-GPU Impact

**Data Parallelism (Option 2)**:
- Each GPU holds **full WC matrix copy** (21 GB)
- Memory per GPU: **25 GB** (same as single GPU)
- ✅ Works with 24 GB GPUs if single GPU works

**Model Parallelism (Option 3)**:
- WC matrix **split** across GPUs
- Memory per GPU: **~8-13 GB** (with 2-3 GPUs)
- ✅ Enables training with smaller GPUs

---

## Recommended Approach

### Step 1: Check Your GPUs

```bash
# Check available GPUs
python -c "import jax; print('Devices:', jax.devices()); print('Count:', jax.local_device_count())"

# Check memory
nvidia-smi
```

### Step 2: Choose Strategy

**If you have 2-3 GPUs with 24+ GB each:**
→ Use **Option 2 (Data Parallelism with pmap)**
- Easiest to implement
- ~2-2.5x speedup
- Each GPU needs 25 GB

**If your GPUs have < 24 GB:**
→ Use **Option 3 (Model Parallelism)**
- More complex to implement
- Enables training with smaller GPUs

**If GPUs have different sizes:**
→ Stick with single GPU (largest one)
- JAX requires symmetric device setup for pmap

### Step 3: Validate

```python
# Test with small grammar first
grammar_rules = 200
max_sent_len = 10
num_trials = 500

# Should see ~2x speedup with 2 GPUs
```

---

## Implementation Priority

### Minimal Changes (Try First)

1. Set environment variable: `export XLA_FLAGS="--xla_gpu_autotune_level=2"`
2. Run existing code
3. Check if JAX uses multiple GPUs: `nvidia-smi` during training

**Expected speedup: 0-20%** (unlikely to help much)

### Medium Changes (Recommended)

Implement Option 2 (pmap):
- 4 code modifications in gsc.py
- ~30 minutes to implement
- **Expected speedup: 1.7-2.5x with 2-3 GPUs**

### Major Changes (If Needed)

Implement Option 3 (model parallelism):
- Extensive refactoring required
- 2-4 days to implement
- Only needed if memory is insufficient

---

## Code Template for Option 2

See the code changes above. Here's the complete modification:

```python
# At top of file (after line 20)
from jax import vmap, jit, pmap

# Around line 2395-2405
_num_devices = jax.local_device_count()

def _run_trials_batched_jax_multi_gpu(rng_keys, net_params, prefix, update_q_discrete):
    _vmap_fn = vmap(_run_single_trial_jax, in_axes=(0, None, None, None))
    _pmap_fn = pmap(_vmap_fn, in_axes=(0, None, None, None))
    return _pmap_fn(rng_keys, net_params, prefix, update_q_discrete)

if _num_devices > 1:
    print(f"Multi-GPU enabled: {_num_devices} devices")
    _run_trials_batched_jax = _run_trials_batched_jax_multi_gpu
else:
    _run_trials_batched_jax = vmap(_run_single_trial_jax, in_axes=(0, None, None, None))

# In estimate_prob_inc_jax (around line 5828)
num_devices = jax.local_device_count()
if num_devices > 1:
    trials_per_device = num_trials // num_devices
    num_trials_adjusted = trials_per_device * num_devices
    rng_keys = jax.random.split(rng, num_trials_adjusted)
    rng_keys = rng_keys.reshape(num_devices, trials_per_device, -1)
else:
    rng_keys = jax.random.split(rng, num_trials)

# After GPU execution (around line 5836)
if num_devices > 1:
    actC_batch = np.array(actC_batch).reshape(-1, actC_batch.shape[-1])
    grid_point_batch = np.array(grid_point_batch).reshape(-1, grid_point_batch.shape[-1])
```

---

## Testing

```python
# Small test to verify multi-GPU works
import jax
print(f"Available devices: {jax.devices()}")
print(f"Device count: {jax.local_device_count()}")

# Run 10 epochs with small corpus
# Compare time with single vs multi-GPU
```

---

## Summary

**For 2-3 GPUs with sufficient memory (24+ GB each):**

✅ **Implement Option 2 (pmap)**
- 4 small code changes
- **Expected speedup: 1.7-2.5x**
- **Training time: 21 days → 8-12 days**

**For smaller GPUs or insufficient memory:**

⚠️ **Implement Option 3 (model parallelism)**
- Major refactoring required
- Enables training on smaller GPUs
- Development time: 2-4 days
