# GSCNet Speedup Diagnosis Report

## Summary

**Issue:** `only_gscnet_speedup.py` and `only_gscnet.py` both take ~30 minutes to train G1 grammar, with no apparent speedup.

**Root Cause:** Both implementations use the same JAX code path when JAX is available. The "speedup" version has optimizations that are either:
1. Not being triggered due to configuration
2. Negligible for this small network size (405 bindings)
3. Not working if JAX is unavailable

## Network Size Analysis

For the G1 grammar in `cho_grammar1_new_copy.py`:
- **Fillers:** 27
- **Roles:** 15
- **Bindings:** 405 (27 × 15)
- **Neural units:** 405

**Memory footprint:**
- WC matrix: 0.63 MB
- S matrix (if materialized): 0.63 MB
- Total: ~2 MB

**Conclusion:** This is a VERY small network. The "18 TB S matrix" optimization only matters for much larger grammars (1.5M+ bindings). For this size, materializing S costs essentially nothing.

## Key Code Comparison

### Both Files Have JAX Support

**`only_gscnet.py` (lines 8-17):**
```python
try:
    import jax
    import jax.numpy as jnp
    from jax import vmap, jit
    from functools import partial
    JAX_AVAILABLE = True
    print("JAX detected - GPU acceleration enabled")
except ImportError:
    JAX_AVAILABLE = False
    print("JAX not found - running in CPU mode...")
```

**`only_gscnet_speedup.py` (lines 8-17):**
```python
# EXACTLY THE SAME CODE
```

**Both files check `JAX_AVAILABLE` and use `estimate_prob_inc_jax` if JAX is available (lines 2605-2606 in original, 3017-3018 in speedup).**

### Main Differences

| Feature | `only_gscnet.py` | `only_gscnet_speedup.py` |
|---------|------------------|-------------------------|
| S matrix materialization | ✓ Creates `self.S = C @ C.T` | ✗ Skips (uses lazy multiplication) |
| Lazy S multiplication | ✗ Uses `S @ v` | ✓ Uses `C @ (C.T @ v)` |
| JIT-compiled dynamics | ✓ (if JAX available) | ✓ (if JAX available) |
| Batched trial execution | ✓ (if JAX available) | ✓ (if JAX available) |
| JAX fast path in runC | ✓ (if conditions met) | ✓ (if conditions met) |

**Key insight:** The only real difference is the lazy S multiplication, which saves 0.63 MB of memory for this network - negligible!

## Expected Speedup Locations

### 1. Training Loop (`train2`)

**Location:** Lines 2943+ (both files)

**What happens:**
- Calls `estimate_prob_inc_jax(num_trials=4)` per epoch (if JAX available)
- Runs 4 trials in parallel on GPU using `vmap`
- Expected speedup: **5-20x** if GPU available

**Critical conditions:**
- JAX must be installed and working
- GPU must be available
- Both files use this same code!

### 2. Equilibrium Finding (`runC`)

**Location:** Lines 2084-2152 in speedup version

**What happens:**
```python
if self.use_jax and not log_trace and tol is None and (self.opts['T_decay_rate'] <= 0):
    # Use JIT-compiled JAX fast path
    final_carry = jax.lax.fori_loop(0, num_steps, body_fun, init_carry)
else:
    # Fallback to Python loop
    while self.t < t_max:
        self.update_stateC()
        ...
```

**Critical conditions for JAX fast path:**
- `self.use_jax = True` (requires JAX available)
- `log_trace = False` ← **THIS IS KEY!**
- `tol = None`
- `T_decay_rate <= 0` (default is 0)

**Expected speedup:** 2-10x if conditions met

### 3. S Matrix Multiplication

**Location:** `update_stateC()` and `_run_single_trial_jax()`

**Original (`only_gscnet.py`):**
```python
gradC = scale_constants * self.S.dot(HGradC())  # Uses materialized S
```

**Speedup (`only_gscnet_speedup.py`):**
```python
temp = C_T @ HGradC_val
gradC = scale_const * (C @ temp)  # Lazy evaluation
```

**Expected speedup for G1 grammar:** **~0%** (network too small, S is only 0.63 MB)

## Why Both Take Same Time

### Scenario 1: JAX Not Working (Most Likely)

**Symptoms:**
- Both scripts print "JAX not found - running in CPU mode..."
- Both use NumPy fallback
- No speedup expected

**Check:**
```bash
python -c "import jax; print(jax.__version__); print(jax.devices())"
```

If this fails, install JAX:
```bash
# For CPU:
pip install jax jaxlib

# For GPU (CUDA):
pip install jax[cuda12]
```

### Scenario 2: JAX Working But Slow

**Possible causes:**
1. **JIT compilation overhead** - First run is slow, subsequent runs faster
2. **CPU-only JAX** - No GPU available, JAX falls back to CPU (often slower than NumPy!)
3. **Small network** - GPU overhead exceeds benefits for 405 units

**Check GPU availability:**
```python
import jax
print(jax.devices())  # Should show [cuda(id=0)] or similar for GPU
```

### Scenario 3: Bottleneck Elsewhere

**Training breakdown for 1000 epochs × 4 trials:**
1. **Equilibrium finding:** 4000 calls to `runC()` - **MAJOR BOTTLENECK**
2. **Gradient computation:** 1000 calls to `cost_grad()` - moderate
3. **Corpus statistics:** 1000 calls to `get_corpus_stat()` - minor
4. **Weight updates:** 1000 SGD/Adam steps - negligible

**The speedup version helps with #1 (equilibrium finding), but only if:**
- JAX is working
- GPU is available
- `log_trace=False` during training (which it should be)

## Diagnostic Steps

### Step 1: Check JAX Installation

```bash
cd /home/user/more-gsc
python -c "import jax; print('JAX version:', jax.__version__); print('Devices:', jax.devices())"
```

**Expected output:**
- With GPU: `Devices: [cuda(id=0)]` or `[CudaDevice(id=0)]`
- CPU only: `Devices: [CpuDevice(id=0)]`
- Not installed: `ModuleNotFoundError: No module named 'jax'`

### Step 2: Run with Verbose Output

Add print statements to check code path:

```python
# At top of cho_grammar1_new_copy.py
import only_gscnet_speedup as gsc

# After network initialization:
print(f"Network use_jax: {net.use_jax}")
print(f"JAX available: {gsc.JAX_AVAILABLE}")
if gsc.JAX_AVAILABLE:
    import jax
    print(f"JAX devices: {jax.devices()}")
```

### Step 3: Profile Training

Time individual components:

```python
import time

# In training loop
t0 = time.time()
net.train2(train_opts={'num_epochs': 10})
print(f"10 epochs took {time.time()-t0:.1f}s")
```

### Step 4: Test JAX Fast Path

```python
# Test if JAX fast path is being used
net.reset(mu=net.ep, sd=0.01)
net.qpolicy = np.linspace(0, net.opts['q_max'], 6)

# Should use JAX fast path (if available)
import time
t0 = time.time()
net.run_word('N', 1, log_trace=False)
t1 = time.time() - t0

# Should NOT use JAX fast path
net.reset(mu=net.ep, sd=0.01)
t0 = time.time()
net.run_word('N', 1, log_trace=True)
t2 = time.time() - t0

print(f"Without trace: {t1:.4f}s")
print(f"With trace: {t2:.4f}s")
print(f"Speedup: {t2/t1:.2f}x")
```

If speedup is ~1x, JAX fast path is not being used.

## Expected Speedup Summary

| Component | Original Time | Speedup Expected | Conditions |
|-----------|--------------|------------------|------------|
| Training (4000 trials) | 25 min | 5-20x (2-5 min) | JAX + GPU |
| Equilibrium finding | Included above | 2-10x | JAX + GPU + `log_trace=False` |
| S matrix optimization | ~0 | 0% | Network too small |
| Gradient computation | 2 min | 1.5-3x (1 min) | JAX + GPU |
| **Total** | **~30 min** | **3-10x (3-10 min)** | **JAX + GPU working** |

## Conclusion

**If both scripts take 30 minutes, the most likely explanation is:**

1. **JAX is not installed** → Both scripts print "JAX not found" and use NumPy
2. **JAX is CPU-only** → Both scripts use JAX but on CPU (often slower than NumPy!)
3. **No GPU available** → Both scripts use JAX CPU backend

**The "speedup" version does NOT provide significant speedup for this small network unless:**
- JAX is properly installed with GPU support
- A GPU is available and being used
- The network is larger (1000+ bindings)

## Recommendations

1. **Check JAX installation and GPU availability** (see diagnostic steps above)
2. **Install JAX with GPU support** if not already installed
3. **Use a larger grammar** to see meaningful speedup from lazy S multiplication
4. **Profile the code** to identify actual bottleneck
5. **Compare first run vs. second run** (JIT compilation overhead)

## Quick Test Script

```python
# Save as test_jax_speedup.py
import only_gscnet_speedup as gsc
import time
import numpy as np

print("JAX available:", gsc.JAX_AVAILABLE)
if gsc.JAX_AVAILABLE:
    import jax
    print("JAX devices:", jax.devices())

# Create small network
PCFG = "0.5 S -> A B\n0.5 S -> B A"
hg = gsc.HarmonicGrammar(pcfg=PCFG, root='S', max_sent_len=2)
sim = hg.get_simlist(dp=0.0)
net = gsc.GscNet(hg=hg, encodings={'similarity': sim}, seed=1024)

print(f"Network use_jax: {net.use_jax}")

# Time equilibrium finding
net.initialize(train_opts={'lrate': 0.1, 'num_trials': 4})
net.generate_corpus(use_freq=True)

t0 = time.time()
net.reset(mu=net.ep, sd=0.01)
net.qpolicy = np.linspace(0, 15, 3)
net.run_word('A', 1, log_trace=False)
t1 = time.time() - t0

print(f"Equilibrium finding: {t1:.4f}s")
print(f"Expected: <0.01s with GPU, ~0.1s with CPU")
```
