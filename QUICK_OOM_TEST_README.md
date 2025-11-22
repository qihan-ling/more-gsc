# Quick OOM Testing Guide

## Problem
The full `sap_grammar_training.py` script can take **5+ hours** before hitting an OOM error, making debugging extremely time-consuming.

## Solution
Use the quick test scripts to detect OOM issues in **2-5 minutes** instead of hours.

---

## 1. Quick OOM Stress Test (RECOMMENDED)

**File:** `quick_oom_test.py`

**What it does:**
- Tests all initialization steps (the most OOM-prone)
- Runs only 3 training epochs (vs 500 in full script)
- Monitors memory at each step
- Detects memory leaks early
- Completes in 2-5 minutes

**How to run:**
```bash
python quick_oom_test.py
```

**What to look for:**

✅ **GOOD** - Memory stays under 8GB:
```
TEST COMPLETE
✅ SUCCESS: Memory usage is reasonable!
   Full training should complete without OOM issues
```

⚠️ **WARNING** - High memory after initialize():
```
⚠️ WARNING: Memory usage is very high after initialize()
   This may cause OOM on systems with <32GB RAM
```

⚠️ **WARNING** - Memory leak detected:
```
Training Memory Growth Analysis:
  Average per epoch: +250.5 MB
  ⚠️ WARNING: Significant memory growth detected!
     Estimated epochs until OOM (~16GB): 20
```

---

## 2. Memory Profiler Utilities

**File:** `memory_profiler.py`

Provides utilities for detailed memory monitoring. Can be used in any script.

### Usage Examples:

#### A) Monitor checkpoints:
```python
from memory_profiler import MemoryMonitor

monitor = MemoryMonitor()

# ... your code ...
monitor.checkpoint("After HarmonicGrammar init")

# ... more code ...
monitor.checkpoint("After GscNet init")

# ... end of script ...
monitor.summary()  # Print full report
```

#### B) Profile specific functions:
```python
from memory_profiler import profile_memory

@profile_memory
def my_expensive_function():
    # ... code ...
    pass
```

#### C) Analyze sparse matrix memory:
```python
from memory_profiler import monitor_sparse_matrices

# After creating your network:
monitor_sparse_matrices(net, "GscNet")
```

This shows:
- Which matrices are sparse vs dense
- How much memory each uses
- How much memory is saved by sparsity

---

## 3. Interpreting Results

### Memory Thresholds:
- **< 8 GB**: Safe for most systems ✅
- **8-12 GB**: Acceptable, may need 16GB+ RAM ⚠️
- **12-16 GB**: High risk on 16GB systems ⚠️
- **> 16 GB**: Will OOM on most systems ❌

### Memory Growth:
- **< 10 MB/epoch**: Minimal, safe ✅
- **10-100 MB/epoch**: Moderate, monitor ⚠️
- **> 100 MB/epoch**: Memory leak likely ❌

### Common Issues:

1. **OOM at initialize()**
   - Check if sparse matrices are being densified
   - Look for `np.zeros_like()` on sparse matrices
   - Verify `use_sparse` flag is set correctly

2. **OOM during training**
   - Memory leak in training loop
   - Check if gradients are accumulating
   - Verify optimizer states are sparse

3. **Gradual memory growth**
   - Trace storage growing unbounded
   - Large objects not being garbage collected
   - Add `gc.collect()` after epochs

---

## 4. Quick Testing Workflow

### Before full training run:
```bash
# 1. Run quick test (2-5 min)
python quick_oom_test.py

# 2. If it passes, run longer test (10-15 min)
python quick_oom_test.py  # modify to run 10 epochs

# 3. If still good, proceed with full training
python sap_grammar_training.py
```

### To add monitoring to your existing script:
```python
# Add at the top of sap_grammar_training.py
from memory_profiler import MemoryMonitor

monitor = MemoryMonitor()

# Add checkpoints at key steps:
net = gsc.GscNet(...)
monitor.checkpoint("After GscNet init")

net.generate_corpus(...)
monitor.checkpoint("After corpus generation")

net.initialize(...)
monitor.checkpoint("After initialize")  # ← Where OOM occurred

# In training loop:
for epoch_block in range(n_epochs // 5):
    net.train2(...)
    if epoch_block % 10 == 0:
        monitor.checkpoint(f"After epoch {epoch_block * 5}")
        monitor.check_oom_risk(threshold_gb=14.0)

# At end:
monitor.summary()
```

---

## 5. System Requirements Estimation

Based on quick test results, estimate RAM needed:

```
Required RAM ≈ (Memory after initialize) × 1.5 + (Avg growth/epoch × num_epochs)
```

Example:
- Memory after initialize: 6 GB
- Growth per epoch: 20 MB
- Training epochs: 500

```
Required RAM ≈ 6 × 1.5 + (0.02 × 500)
            ≈ 9 + 10
            ≈ 19 GB
```

So you'd need a system with **24-32 GB** RAM to be safe.

---

## 6. Troubleshooting

### "ModuleNotFoundError: No module named 'psutil'"
```bash
pip install psutil
```

### Test script itself runs out of memory
- Reduce corpus size further (change `nsamples=100` to `nsamples=10`)
- Reduce training epochs (change `range(3)` to `range(1)`)

### False positives (test passes but full training fails)
- Increase test epochs to 10-20 for better leak detection
- Monitor memory growth trend more carefully

---

## Summary

| Script | Runtime | Purpose | Use When |
|--------|---------|---------|----------|
| `quick_oom_test.py` | 2-5 min | Fast OOM detection | After any code changes |
| `memory_profiler.py` | - | Detailed monitoring | Debugging specific issues |
| `sap_grammar_training.py` | 5+ hours | Full training | After quick test passes |

**Pro tip:** Always run `quick_oom_test.py` after making any changes to the code. It will save you hours of waiting!
