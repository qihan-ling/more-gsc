# Critical Finding: Divergence at Word 1

## Your Discovery

Running `compare_parsing_divergence.py` revealed:

```
✓ Initial states match (diff: 1.29e-14)
❌ After Word 1 "N": HUGE divergence (diff: 17.4)

Top diverging bindings:
  BE:0/(1,2):   sparse=0.28, orig=0.52 (Δ=0.24)
  _/(1,5):      sparse=0.49, orig=0.28 (Δ=0.21)
  Vi:0/(1,3):   sparse=0.14, orig=0.33 (Δ=0.19)
  *Vpp:0/(4,2): sparse=0.09, orig=0.28 (Δ=0.19)
  Vi:0/(1,2):   sparse=0.34, orig=0.16 (Δ=0.18)
```

## What This Tells Us

Since divergence happens **immediately** during the first word:

1. ❌ NOT an accumulation issue (would take multiple words)
2. ❌ NOT a high-activation issue (happens at start with normal values)
3. ✅ **Fundamental difference in gradient computation or dynamics**

## The Culprit Chain

```
run_word(word, wpos=1, log_trace=False)
  ↓
  set_input(bname)  ← Sets extC
  ↓
  runC(duration, log_trace=False)
    ↓
    update_stateC() [called many times in loop]
      ↓
      HGradC()
        ↓
        hgrad_g = WC.dot(actC) + bC + extC  ← LIKELY CULPRIT
        ↓
      temp = C_T.dot(hgrad)
      gradC = C.dot(temp)
      actC = actC + dt * scale_constants * gradC
```

## Most Likely Root Cause

**The `WC.dot(actC)` operation at line 5535 in `HGradC()`**

### Why This Operation?

1. **It's called thousands of times** during `runC()` loop
2. **Sparse .dot() may behave differently** than dense
3. **Your 3 training fixes corrected WC construction**, so WC should be identical
4. **But the .dot() operation itself may have precision issues**

## Investigation Steps

### Step 1: Verify WC Matrices Match

Run `debug_gradient_divergence.py` to check:
```python
# Compare WC matrices themselves
WC_sparse_dense = net_sparse.WC.toarray()
WC_diff = np.abs(WC_sparse_dense - net_orig.WC).sum()
```

**If WC_diff > 1e-10**: Your training fixes didn't fully work - WC matrices differ
**If WC_diff ≈ 0**: WC matrices match, but `.dot()` operation produces different results

### Step 2: If WC Matrices Match But .dot() Differs

This means the sparse matrix multiplication has a bug. Possible causes:

**A. Dtype mismatch during multiplication:**
```python
# Check at only_gscnet_speedup_sap.py:5535
hgrad_g = self.WC.dot(actC) + self.bC + self.extC

# WC might be float64 but actC might be float32 (or vice versa)
# Result could have precision loss
```

**Fix:**
```python
# Ensure consistent dtype
hgrad_g = self.WC.dot(actC.astype(self.WC.dtype)) + self.bC + self.extC
```

**B. Sparse CSR dot product precision issue:**
```python
# At construction (around line 2290), WC is converted to CSR:
self.WC = self.WC.tocsr()

# CSR .dot() might have different numerical behavior than dense
```

**Fix:** Try using dense WC for inference:
```python
def run_sent(self, ...):
    # Temporarily convert to dense for parsing
    if hasattr(self, 'use_sparse') and self.use_sparse:
        WC_backup = self.WC
        self.WC = self.WC.toarray()
        self.use_sparse = False

    # ... run parsing ...

    # Restore sparse
    if WC_backup is not None:
        self.WC = WC_backup
        self.use_sparse = True
```

**C. CSR format data corruption or indexing issue:**

Check if CSR matrix was constructed correctly:
```python
# After converting to CSR
print(f"WC CSR: data dtype={WC.data.dtype}, indices dtype={WC.indices.dtype}")
print(f"WC CSR nnz: {WC.nnz}, should match dok nnz")

# Verify by comparing spot checks
test_row = 0
dense_row = WC_dense[test_row, :]
sparse_row = WC_sparse.getrow(test_row).toarray().flatten()
assert np.allclose(dense_row, sparse_row)
```

### Step 3: Quick Test

**Hypothesis**: Sparse WC.dot() is the issue

**Test**: Temporarily force dense WC during parsing:

```python
# In test_parse_inc() or run_sent(), add at the start:
if hasattr(net, 'use_sparse') and net.use_sparse:
    print("Converting WC to dense for parsing test...")
    net.WC = net.WC.toarray()
    net.use_sparse = False
```

Then run parsing test. **If accuracy improves to match original, you've confirmed the issue!**

## Expected Findings

### Scenario A: WC Matrices Differ
→ Training fixes incomplete, need to revisit weight/bias initialization

### Scenario B: WC Identical, .dot() Differs
→ Sparse matrix multiplication bug, needs dtype fix or dense conversion for inference

### Scenario C: WC and .dot() Match, extC Differs
→ Input setting bug in `set_input()` for sparse version

## Quick Fix to Try First

Add this to `run_sent()` at line 5585, right after `self.reset()`:

```python
self.reset(mu=self.ep, sd=0.02)

# TEMPORARY FIX: Use dense WC for parsing inference
if hasattr(self, 'use_sparse') and self.use_sparse:
    if sparse.issparse(self.WC):
        self._WC_sparse_backup = self.WC
        self.WC = self.WC.toarray()
        print("[TEMP FIX] Using dense WC for parsing")
```

And at the end of `run_sent()`:

```python
# Restore sparse WC
if hasattr(self, '_WC_sparse_backup'):
    self.WC = self._WC_sparse_backup
    del self._WC_sparse_backup
```

**If this fixes parsing accuracy → confirmed sparse .dot() is the issue!**

## Summary

1. ✅ Divergence pinpointed to first word
2. ✅ Likely cause: `WC.dot(actC)` in `HGradC()`
3. 🔍 Next: Run `debug_gradient_divergence.py` to see if WC differs or .dot() differs
4. 🔧 Quick test: Force dense WC during parsing to verify hypothesis
