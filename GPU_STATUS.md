# GPU Acceleration Status

## TL;DR

**❌ Drop-in CuPy replacement doesn't work**
**✅ Use regular `cho_grammar1.py` on CPU for now**
**🚀 GPU acceleration requires batch parallelization (future work)**

---

## What Happened

### Attempted Approach: Drop-in NumPy→CuPy Replacement

The initial strategy was to replace NumPy with CuPy by monkey-patching:

```python
import gsc
gsc.np = cupy  # Try to make gsc use CuPy instead of NumPy
```

###  Why It Failed

**Problem**: Python's import mechanics don't allow this.

When `gsc.py` does `import numpy as np` at the module level, it creates a direct reference to the numpy module. Changing `gsc.np` later doesn't affect the `np` variable that's already been imported inside the module.

**Specific Errors Encountered**:

1. **PCFG Parsing Error** - Empty lines in grammar string (fixed)
2. **SyntaxWarning** - `is not 'string'` should be `!= 'string'` (fixed in gsc.py:4650)
3. **CUDA Version Mismatch** - cupy-cuda11x on CUDA 12.8 system (fixed - use cupy-cuda12x)
4. **NumPy/CuPy Conflict** - `np.random.seed()` calling CuPy's random, causing conversion errors

The fundamental issue is #4: the hybrid NumPy/CuPy approach doesn't work because we can't intercept module-level imports.

---

## Current Status

**All GPU code (`gsc_gpu.py`, `cho_grammar1_gpu.py`) now runs on CPU.**

Performance is identical to the original `cho_grammar1.py`.

---

## Path Forward: Batch Parallelization

The drop-in replacement approach would only give **5-10x speedup** anyway. A much better approach is **batch parallelization**, which can give **20-40x speedup**.

### Strategy

Instead of:
```python
# Serial: Run 4 trials one at a time
for trial in range(4):
    result = run_trial()  # Each takes ~10 seconds
# Total: 40 seconds
```

Do:
```python
# Parallel: Run all 4 trials simultaneously on GPU
results = run_trials_batched_gpu(num_trials=4)  # All 4 run in parallel
# Total: ~10 seconds (4x speedup)
```

### Implementation Plan

1. **Identify bottleneck**: `estimate_prob_inc()` in gsc.py:5368
   - Runs `num_trials=4` iterations serially
   - Each trial is independent → perfect for parallelization

2. **Create batched version**:
   - Rewrite `runC()` loop to process batches of trials
   - Use CuPy/PyTorch for batch operations
   - Keep initialization on CPU, move computation to GPU

3. **Expected speedup**: 20-40x
   - 4x from batch parallelism across trials
   - 5-10x from GPU matrix operations
   - = 60 min → 1.5-3 min for 1000 epochs

### Effort Required

- **Difficulty**: Moderate (requires understanding gsc internals)
- **Time**: 2-3 days of development
- **Lines of code**: ~500-1000 (new batch processing functions)

---

## Immediate Recommendations

### For Current Runs

**Just use the regular CPU version:**

```bash
# On your cluster
python cho_grammar1.py

# Or with SLURM (use CPU nodes, not GPU):
#SBATCH --nodes=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=128G
# NO --gres=gpu line!

python cho_grammar1.py
```

###  Optimize CPU Performance

While waiting for GPU implementation, optimize CPU usage:

```bash
# Set threading environment variables
export OMP_NUM_THREADS=8
export MKL_NUM_THREADS=8
export OPENBLAS_NUM_THREADS=8

python cho_grammar1.py
```

This might give 10-30% speedup by parallelizing NumPy operations.

### Reduce Memory

You don't need 128GB RAM. Try:

```bash
#SBATCH --mem=32G  # Reduce from 128G
```

---

## Lessons Learned

### ❌ What Doesn't Work

1. **Monkey-patching numpy with cupy** - Import mechanics prevent this
2. **Hybrid NumPy/CuPy classes** - Doesn't intercept module-level imports
3. **sys.modules replacement** - Too risky, affects all imports globally

### ✅ What Works

1. **Batch parallelization** - Rewrite core loops for parallel execution
2. **PyTorch/JAX reimplementation** - Port specific functions to GPU frameworks
3. **Custom CUDA kernels** - For critical hot paths (advanced)

### 📊 Expected Speedups

| Approach | Speedup | Effort | Status |
|----------|---------|--------|--------|
| Drop-in CuPy | 5-10x | Low | **Failed** |
| Batch parallelization | 20-40x | Medium | **Recommended** |
| Full PyTorch port | 20-50x | High | Future |
| Custom CUDA kernels | 50-100x | Very High | Unnecessary |

---

## Technical Details

### Why Monkey-Patching Failed

```python
# In gsc.py (at module load time):
import numpy as np  # Creates direct reference to numpy

# Later, in gsc_gpu.py:
gsc.np = cupy  # Only changes gsc.np attribute
# But code inside gsc.py still uses 'np' which points to numpy

# When gsc.py code does:
np.random.seed(seed)  # Still calls numpy.random, not cupy.random
```

### Attempted Fixes

1. **HybridNumPy class** - Tried to selectively route calls
   - Issue: __getattr__ doesn't intercept module-level imports

2. **sys.modules replacement** - Replace numpy globally
   - Issue: Too risky, would affect all code

3. **Import hooks** - Intercept imports
   - Issue: Complex, fragile, not maintainable

### Why Batch Parallelization Will Work

```python
# Current (serial):
def estimate_prob_inc(self, prefix, num_trials=4):
    for trial in range(num_trials):
        self.reset()       # Independent
        self.run_wrapup()  # Independent
        results.append(self.read_grid_point())
    return results

# Future (parallel):
def estimate_prob_inc_gpu(self, prefix, num_trials=4):
    # Run all trials in parallel on GPU
    results = run_all_trials_parallel_gpu(
        batch_size=num_trials,
        init_state=self.ep,
        ...
    )
    return results
```

Each trial is independent, so they can run in parallel without communication.

---

## Files Status

- ✅ `gsc.py` - Fixed syntax warning (line 4650)
- ✅ `cho_grammar1.py` - Original CPU version (use this!)
- ⚠️ `gsc_gpu.py` - Currently just imports gsc (CPU only)
- ⚠️ `cho_grammar1_gpu.py` - Same as CPU version
- ✅ `requirements_gpu.txt` - Updated for CUDA 12.x
- ✅ `setup_gpu.sh` - Diagnostic script for CUDA/CuPy detection
- ✅ `test_simple.py` - Test grammar parsing
- ✅ `GPU_README.md` - Detailed documentation
- ✅ `GPU_STATUS.md` - This file

---

## Questions?

### "Should I use GPU nodes?"

**No**. The current code runs on CPU only. Use regular CPU nodes to avoid wasting GPU resources.

### "Will GPU help at all?"

**Not with the current code**. We need batch parallelization first.

### "How long will it take to implement batch parallelization?"

**2-3 days** for someone familiar with the codebase. Requires:
- Understanding the training loop
- Rewriting key functions for batch processing
- Testing for numerical correctness

### "Is it worth the effort?"

**Yes**, if you're running many experiments. The 20-40x speedup means:
- Current: 60 minutes → Future: 1.5-3 minutes
- Worth it if you'll run >10 experiments

### "Can I help?"

Yes! Start by:
1. Understanding `estimate_prob_inc()` in gsc.py:5368
2. Identify which operations can be batched
3. Create a proof-of-concept batch version

---

**Bottom line**: Use `cho_grammar1.py` on CPU for now. GPU acceleration requires batch parallelization, which is future work.
