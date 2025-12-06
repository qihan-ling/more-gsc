# Investigation Report: gsc.py vs only_gscnet_speedup_sap.py Code Differences

## Executive Summary

**Critical Finding**: The SAP version **FIXED** a major bug in `gsc.py`, but this should make it *better*, not worse. The performance difference must come from elsewhere.

---

## Investigation 1: WC Matrix Initialization

### Bug Found in gsc.py (FIXED in SAP)

**Location**: `gsc.py` line 2851 in `build_model()` competition rules

**The Bug:**
```python
elif r1 == 's' and r2 != 's':
    for role in roles.role_names:
        bname1 = rule['f1'] + bsep + role       # Defines bname1
        mother_roles = roles.get_mothers(role)
        focus_mother_roles = mother_roles[r2]
        for mr in focus_mother_roles:
            if mr in roles.role_names:
                bname2 = rule['f2'] + bsep + mr  # Defines bname2
                self.set_weight(b1name, b2name, rule['H'],  # ❌ USES WRONG VARIABLES!
                                cumulative=cumulative, c2n=False)
```

**Problem**: Uses undefined variables `b1name`, `b2name` instead of `bname1`, `bname2`.

**What Actually Happens**: Python uses `b1name` and `b2name` from the **previous loop iteration** (binary/copy rules), causing competition rules to be applied to WRONG bindings!

**Impact on G1 Grammar**:
- Competition rules for sentences are incorrectly connected
- WC matrix has wrong weights for complex sentence structures
- May explain why complex sentences (S3, S4) fail

**SAP Version** (line 2068): ✅ **FIXED** - correctly uses `bname1`, `bname2`

### Other WC Initialization Differences

1. **Sparse Matrix Support** (SAP only):
   ```python
   # SAP version:
   if self.use_sparse:
       self.WC = sparse.lil_matrix((self.num_bindings, self.num_bindings))
   else:
       self.WC = np.zeros((self.num_bindings, self.num_bindings))
   ```
   - gsc.py: Always uses dense numpy arrays
   - SAP: Supports both dense and sparse (for large grammars)

2. **Optimized Binary/Copy Rule Processing** (SAP only):
   - Pre-computes filler indices to avoid repeated string concatenation
   - Direct matrix updates: `self.WC[idx1, idx2] += H`
   - Progress reporting for large rule sets

3. **Optimized Role Lookups** (SAP only):
   - Uses pre-computed `role_is_bracketed` array instead of function calls
   - Uses `role_mothers_idx` for direct indexing
   - Uses `role_tuples` instead of string parsing

**Conclusion**: SAP version has **better optimizations** and **fixes a critical bug**. It should perform *better* than gsc.py, not worse!

---

## Investigation 2: run_sent/runC Implementation

### Checked Functions:
- `run_sent()` - Nearly identical
- `runC()` - Nearly identical
- `update_stateC()` - Nearly identical
- `add_noiseC()` - Nearly identical

### Only Minor Difference Found:

**get_ep() call in runC (line 4066 vs 2660):**
- gsc.py: `self.runC(dur)` - doesn't specify log_trace
- SAP: `self.runC(dur, log_trace=False)` - explicitly sets it

**Impact**: Negligible (log_trace likely defaults to False anyway)

### SAP Enhancements:
1. **Better JAX/NumPy handling** in noise generation
2. **Configurable ep_dur** for large grammars:
   ```python
   dur = self.opts.get('ep_dur', 10)  # SAP allows override
   ```
3. **Diagnostic random state tracking** (line 2672-2673)

**Conclusion**: No significant differences that would cause S3/S4 failures.

---

## Investigation 3: get_ep() Equilibrium Point Calculation

### Comparison:

| Aspect | gsc.py | only_gscnet_speedup_sap.py |
|--------|---------|----------------------------|
| **Default duration** | `dur=10` (hardcoded) | `dur=None` → `opts.get('ep_dur', 10)` (configurable) |
| **Integration method** | Identical | Identical |
| **Newton method** | Identical | Identical |
| **runC call** | `self.runC(dur)` | `self.runC(dur, log_trace=False)` |
| **Random state tracking** | None | Diagnostic print (line 2672-2673) |

**Conclusion**: Functionally equivalent. SAP version just adds configurability for large grammars.

---

## The Paradox

### What We Know:
1. **gsc.py has a critical WC initialization bug** (wrong competition weights)
2. **SAP version fixed this bug**
3. **SAP version has better optimizations**
4. **Yet user reports gsc.py performs better on S3/S4**

### Possible Explanations:

#### Theory 1: The Bug is "Helpful" for G1
The incorrect competition weights in gsc.py might *accidentally* create better dynamics for G1 grammar's specific structure. This would be a case of "two wrongs making a right."

**Test**: Check if the buggy variable values from binary/copy rules happen to be beneficial for S3/S4 structures.

#### Theory 2: Different Training Results
If the models were trained separately:
- gsc.py's buggy WC → different gradient flow → different learned weights
- Even after 1000 epochs, the learned WC could be fundamentally different
- SAP's "correct" initialization might lead to a different (worse) local minimum

**Test**: Train both versions with identical seeds and compare final WC matrices.

#### Theory 3: Sparse vs Dense Matrix Bug
The user tested:
- Dense SAP → bad results (same as gsc.py supposedly)
- Sparse SAP → worse results

But we haven't actually verified gsc.py's results. Maybe:
- gsc.py also has bad S3/S4 performance
- Or there's a subtle numerical difference in sparse matrix operations

**Test**: Actually run gsc.py training and parsing to confirm it produces good results.

#### Theory 4: Different Default Parameters
Hidden parameter differences we haven't found yet:
- Different default `m`, `q_rate`, `dt`, `T_init`
- Different `ep_dur` affecting equilibrium quality
- Different optimization settings

**Test**: Print all opts values from both implementations during initialization.

---

## Recommended Next Steps

### 1. Verify gsc.py Actually Works Better
```bash
python -c "
import gsc
import numpy as np

# Train with gsc.py
net = gsc.GscNet(...)  # G1 setup
# ... train ...
# ... test parsing at t=12 ...
print('S3 accuracy:', ...)
print('S4 accuracy:', ...)
"
```

### 2. Compare Trained WC Matrices
```python
# Load models trained with each version
net_gsc = gsc.load_model('gsc_trained.pkl')
net_sap = gsc_sap.load_model('sap_trained.pkl')

# Compare WC matrices
diff = net_gsc.WC - net_sap.WC
print(f"WC difference: max={np.abs(diff).max()}, mean={np.abs(diff).mean()}")

# Check specific competition weights that the bug affects
# (where r1='s' and r2 != 's')
```

### 3. Test the Bug's Impact Directly
Patch gsc.py to fix the typo:
```python
# Line 2851, change:
self.set_weight(b1name, b2name, rule['H'],  # OLD (buggy)
# to:
self.set_weight(bname1, bname2, rule['H'],  # FIXED
```

Then retrain and see if performance changes.

### 4. Isolate Sparse Matrix Issue
If sparse is worse than dense, there may be a numerical bug in sparse operations:
```python
# Check if sparse WC.dot() produces same results as dense
WC_dense = net_dense.WC
WC_sparse = net_sparse.WC.tocsr()  # Convert for fast multiplication
actC_test = np.random.rand(num_bindings)

result_dense = WC_dense.dot(actC_test)
result_sparse = WC_sparse.dot(actC_test)

diff = result_dense - result_sparse
print(f"Sparse vs dense difference: {np.abs(diff).max()}")
```

---

## Conclusion

The code investigation reveals that **only_gscnet_speedup_sap.py should perform BETTER** than gsc.py because it:
1. Fixes a critical variable name bug in competition rules
2. Has better optimizations
3. Supports sparse matrices for large grammars

The reported performance difference is **paradoxical** and requires experimental verification. Most likely:
- gsc.py's bug accidentally helps G1 (unlikely)
- Different training trajectories led to different local minima
- There's a subtle sparse matrix numerical issue
- Or the original claim needs verification (gsc.py might also have bad S3/S4 performance)
