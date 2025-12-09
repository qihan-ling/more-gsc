# Complete Bug Analysis: Parsing & Treelet Activation Issues

## Problem Statement
Both `sap_grammar_training_test2.py` (speedup) and `sap_grammar_training_test3.py` (original) produce incorrect:
1. Parsing accuracy results
2. Treelet activation trajectories

But `cho_grammar1.py` produces correct parsing accuracy.

## Root Causes Found

### Bug #1: `initialize_traces()` Doesn't Clear Old Data ✅ FIXED

**Location:** `gsc.py:3504` and `only_gscnet_speedup_sap.py:5477`

**Problem:**
```python
# OLD BUGGY CODE:
if hasattr(self, 'traces'):
    for key in trace_list:
        self.traces[key] = list(self.traces[key])  # ← Keeps old data!
```

When `finalize_traces()` converts traces to numpy arrays, the next `initialize_traces()` call did:
- `list(numpy_array_2D)` → list of 1D arrays (PRESERVES old data!)

**Impact:**
- Sentence 0: traces = S0 data only ✓
- Sentence 1: traces = S0 + S1 data (MIXED!)
- Sentence 2: traces = S0 + S1 + S2 data (MORE MIXED!)
- Result: All sentences show same/wrong treelet activations

**Fix Applied:**
```python
# FIXED CODE:
if hasattr(self, 'traces'):
    for key in trace_list:
        self.traces[key] = []  # ← Properly clears!
```

**Status:** ✅ Fixed in both files (commit 011c3a8)

---

### Bug #2: Random Seed Resetting Per Commitment Level

**Location:** `sap_grammar_training_test3.py:326`

**Problem:**
```python
for t in commitment_levels:
    np.random.seed(1024 + t)  # ← Resets seed for EACH commitment level!
    parse_results = gsc.test_parse_inc(...)
```

**Why this is wrong:**
- Each commitment level gets a DIFFERENT random seed
- The dynamics depend on random noise in `add_noiseC()`
- Different seeds → different noise → different parsing trajectories
- This makes results non-comparable across commitment levels

**What cho_grammar1.py does correctly:**
```python
# NO seed reset per commitment level
for t in commitment_levels:
    # Uses natural random state progression
    parse_results = gsc.test_parse_inc(...)
```

**Fix:**
Remove the `np.random.seed(1024 + t)` line from the commitment level loop.

**Status:** ⚠️ NOT YET FIXED

---

### Bug #3: Model Loading vs Training

**cho_grammar1.py (WORKS):**
1. Trains model inline (lines 111-114):
   ```python
   for epoch_block in range(n_epochs // 10):
       net.train2(train_opts={'num_epochs': 10},
                  savefilename='g1_model.pkl')
   ```
2. Reloads model (line 140):
   ```python
   net = gsc.load_model('g1_model.pkl')
   ```
3. Runs parsing tests

**test3.py (BROKEN):**
1. All training code is COMMENTED OUT
2. Tries to load non-existent model (line 239):
   ```python
   net = gsc.load_model('sap_g1_model_orig.pkl')  # ← Doesn't exist!
   ```
3. Would crash unless model was created separately

**Fix:**
Either:
- Option A: Run training first and save model
- Option B: Remove model loading and train inline like cho_grammar1.py

**Status:** ⚠️ NOT YET FIXED

---

## Differences Summary

| Feature | cho_grammar1.py (✓ Works) | test2/test3.py (✗ Broken) |
|---------|---------------------------|---------------------------|
| Training | ✓ Inline training | ✗ Commented out |
| Model Loading | ✓ Loads own trained model | ✗ Loads non-existent model |
| Random Seed per t | ✓ No reset | ✗ Resets per commitment |
| Treelet Plotting | ✓ None (avoids bug) | ✗ Has plotting + bug |
| initialize_traces bug | ✓ Not triggered | ✗ Triggered |

---

## Recommended Fixes

### Fix for test2.py and test3.py:

1. **Remove random seed reset:**
   ```python
   # DELETE THIS LINE:
   # np.random.seed(1024 + t)
   ```

2. **Enable training or ensure model exists:**
   ```python
   # Option A: Uncomment training code
   # Option B: Train separately and ensure .pkl file exists
   # Option C: Use cho_grammar1.py's approach
   ```

3. **initialize_traces fix is already applied** ✅

### Alternative: Use cho_grammar1.py as Template

The cleanest approach is to use cho_grammar1.py's structure:
```python
# 1. Train inline
for epoch_block in range(n_epochs // 10):
    net.train2(train_opts={'num_epochs': 10}, savefilename='model.pkl')

# 2. Reload for testing
net = gsc.load_model('model.pkl')

# 3. Test parsing WITHOUT seed resets
for t in commitment_levels:
    # NO np.random.seed() here!
    parse_results = gsc.test_parse_inc(net, dq=dq, ...)
```

---

## Testing the Fixes

After applying all fixes, verify:

1. **Parsing Accuracy:** Should match cho_grammar1.py output
   - All commitment levels should show improving accuracy
   - No catastrophic drops at high commitment

2. **Treelet Activations:** Should be sentence-specific
   - S0 ("N Vi"): Should show *N, *Vi, S[1] treelets
   - S3 ("N BE Vpp P N"): Should show *BE, VP[2], VP[3] treelets
   - Each sentence should have DIFFERENT top-4 treelets

---

## Files Modified

1. ✅ `gsc.py:3504` - Fixed initialize_traces
2. ✅ `only_gscnet_speedup_sap.py:5477` - Fixed initialize_traces
3. ⚠️ `sap_grammar_training_test2.py:326` - Remove seed reset (TODO)
4. ⚠️ `sap_grammar_training_test3.py:326` - Remove seed reset (TODO)
5. ⚠️ Both test scripts - Fix model loading/training (TODO)
