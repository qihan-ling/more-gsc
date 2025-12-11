# Parsing Accuracy Issue: Commitment >= 5

## Status
- ✅ **Training results**: Perfect match after your 3 fixes
- ✅ **Activation graphs**: Working (after reordering sections)
- ❌ **Parsing accuracy**: Fails at commitment >= 5 on longer sentences

## The Problem Pattern

At commitment t=5:

| Sentence | Length | Sparse Accuracy | Original Accuracy |
|----------|--------|-----------------|-------------------|
| S0: N Vi | 2 words | 100% ✓ | 100% ✓ |
| S1: N Vi P N | 3 words | 100% ✓ | 100% ✓ |
| S2: N BE Vpp | 3 words | 100% ✓ | 100% ✓ |
| S3: N BE Vpp P N | 4 words | **30%** ✗ | 100% ✓ |
| S4: N Vpp P N Vi | 5 words | **10%** ✗ | 80% ✓ |

**Pattern**: Sparse fails specifically on **longer, more complex sentences** at **higher commitment levels**.

## Why This Happens

Since your 3 fixes made training match perfectly, the bug must be in the **inference/parsing code path** only.

### Key Observations:

1. **Training uses**: `log_trace=True`, many epochs, gradient accumulation
2. **Parsing uses**: `log_trace=False`, single forward pass per trial, different code path

3. **At high commitment (t=5)**:
   - q values are higher: q = 5/5 = 1.0 per word
   - Activations reach higher levels
   - More complex dynamics over longer sequences

4. **Sparse-specific issue**:
   - Sparse `WC.dot(actC)` may accumulate numerical errors differently than dense
   - Over 4-5 words at high q, small errors compound
   - Results in wrong final parse

## Potential Root Causes

### 1. **Sparse Matrix Precision at High Activations**

At `only_gscnet_speedup_sap.py:5535`:
```python
hgrad_g = self.WC.dot(actC) + self.bC + self.extC
```

When actC values are high (from q=1.0), sparse matrix multiplication might:
- Lose precision in dot product accumulation
- Handle extreme values differently than dense

### 2. **JAX Fallback Path Mismatch**

At `only_gscnet_speedup_sap.py:4454-4462`:
```python
if sparse.issparse(self.WC):
    # Falls back to CPU version even with JAX enabled
    return self.estimate_prob_inc(...)
```

- Original: Uses JAX for all operations
- Sparse: Falls back to CPU `estimate_prob_inc()`
- These paths might compute differently at high q

### 3. **Gradient Accumulation Over Long Sequences**

At `only_gscnet_speedup_sap.py:2860-2864`:
```python
temp = self.C_T.dot(hgrad)
gradC = self.C.dot(temp)
gradC = self.scale_constants * gradC
self.actC = self.actC + self.dt * gradC
```

Over 4-5 words with high q:
- Many timesteps accumulate
- Sparse operations might drift from dense
- Errors compound

## Diagnostic Strategy

### Step 1: Run Divergence Analysis
```bash
python compare_parsing_divergence.py
```

This will:
1. Run both implementations on S3 at t=5 with **same random seed**
2. Compare states after each word
3. Identify exact divergence point

### Step 2: Check Specific Hypotheses

**Hypothesis A: Sparse WC.dot() precision**
- Compare `WC.dot(actC)` output between sparse and dense
- Check if differences grow with higher actC values

**Hypothesis B: Different code paths**
- Verify sparse uses CPU fallback in `estimate_prob_inc()`
- Check if `estimate_prob_inc()` has bugs that `estimate_prob_inc_jax()` doesn't

**Hypothesis C: Gradient accumulation**
- Track `gradC` values over time
- See if sparse drifts from original during long sequences

### Step 3: Test the Fix

Once divergence point is found:
1. Apply fix to sparse implementation
2. Rerun parsing tests at all commitment levels
3. Verify S3 and S4 now match original at t=5+

## Expected Investigation Flow

```
1. Generate both model files:
   - Run sap_grammar_training_test2.py → sap_g1_model_fixed_sparse_nocompress.pkl
   - Run sap_grammar_training_test3.py → sap_g1_model_orig.pkl

2. Run diagnostic:
   python compare_parsing_divergence.py

   Look for output like:
   "⚠ DIVERGENCE DETECTED at word N!"
   "Top 5 diverging bindings: ..."

3. Based on where divergence occurs:
   - Word 1: Initial state/reset issue
   - Word 2-3: Early dynamics issue
   - Word 4-5: Accumulation issue

4. Examine the specific operation that caused divergence:
   - WC.dot(actC)? → Check sparse matrix precision
   - update_stateC()? → Check gradient computation
   - run_word()? → Check word processing logic

5. Apply targeted fix and verify
```

## Files for Investigation

- `compare_parsing_divergence.py`: Find exact divergence point
- `only_gscnet_speedup_sap.py:5524`: HGradC() - gradient computation
- `only_gscnet_speedup_sap.py:2834`: update_stateC() - state update with sparse ops
- `only_gscnet_speedup_sap.py:3268`: run_word() - word processing
- `only_gscnet_speedup_sap.py:5556`: run_sent() - sentence processing
- `only_gscnet_speedup_sap.py:4454`: JAX fallback check

## Success Criteria

After fix is applied, parsing accuracy should match:

```
Commitment t=5:
  S0: sparse=100%, original=100% ✓
  S1: sparse=100%, original=100% ✓
  S2: sparse=100%, original=100% ✓
  S3: sparse=100%, original=100% ✓  ← Currently 30%
  S4: sparse=80%,  original=80%  ✓  ← Currently 10%

Overall: sparse=0.96, original=0.96 ✓  ← Currently 0.68
```

And this should hold for all commitment levels t=1 through t=12.
