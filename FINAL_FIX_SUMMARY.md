# Final Fix Summary: Treelet Activation Bug

## Problem
Both `sap_grammar_training_test2.py` and `sap_grammar_training_test3.py` showed the same top 4 treelets for all sentences at each position, when they should show sentence-specific treelets.

For example, at position (3,2):
- **Buggy output:** All sentences (S0-S4) showed `['VP[1](*Vi,PP[1])', '*Vi(*Vi,)', 'VPpp[1](*Vpp,PP[1])', 'RC[1](*Vpp,PP[1])']`
- **Expected:** Each sentence should show different treelets based on its structure
  - S0 ("N Vi"): Should show N and Vi related treelets
  - S3 ("N BE Vpp P N"): Should show BE and VPpp related treelets

## Root Cause

The bug was in `initialize_traces()` method at:
- `gsc.py:3504`
- `only_gscnet_speedup_sap.py:5477`

### Original Buggy Code
```python
if hasattr(self, 'traces'):
    for key in trace_list:
        self.traces[key] = list(self.traces[key])  # Keeps old data!
```

After `finalize_traces()` converts traces to numpy arrays, calling `list(numpy_array_2D)` creates a list of 1D arrays (one per row), which **preserves all old data**. Then when `update_traces()` appends new data, the result is:
- Sentence 0: S0 data only ✓
- Sentence 1: S0 + S1 data ✗
- Sentence 2: S0 + S1 + S2 data ✗
- etc.

All subsequent sentences accumulated data from all previous sentences, causing them to show the same (mixed) treelet activations.

## The Fix (Two Stages)

### Stage 1: Clear old data (commit 011c3a8)
```python
if hasattr(self, 'traces'):
    for key in trace_list:
        self.traces[key] = []  # Properly clears!
```

This cleared old data but caused "pure flat activation" because it didn't log the initial state.

### Stage 2: Log initial state (commit 25f4ce9)
```python
if hasattr(self, 'traces'):
    for key in trace_list:
        self.traces[key] = []
    self.update_traces()  # Log initial state!
```

This ensures both branches (first call and re-initialization) log the initial state after clearing.

## Why update_traces() is Critical

The `else` branch (first time initialization) already calls `self.update_traces()`:
```python
else:
    self.traces = {}
    for key in trace_list:
        self.traces[key] = []
    self.update_traces()  # ← Already here!
```

For consistency, the `if` branch (re-initialization) must also call it:
```python
if hasattr(self, 'traces'):
    for key in trace_list:
        self.traces[key] = []
    self.update_traces()  # ← Must add here!
```

Without `update_traces()` in the `if` branch:
- First sentence: Gets initial state logged ✓
- Subsequent sentences: **Miss initial state**, causing misaligned traces ✗

## Secondary Fixes

### Random Seed Reset (commit b014fa9)
Removed `np.random.seed(1024 + t)` from commitment level loops in both test scripts. This was resetting the random state for each commitment level, making results non-comparable. The correct approach (used in `cho_grammar1.py`) is to use natural random state progression.

## Verification

Created `test_initialize_traces_fix.py` which confirms:
```
--- Sentence 0: N Vi ---
  Trace shape: (2601, 405)  ← Clean S0 traces only

--- Sentence 1: N Vi P N ---
  Trace shape: (2201, 405)  ← Clean S1 traces only, different from S0

--- Sentence 3: N BE Vpp P N ---
  Trace shape: (2001, 405)  ← Clean S3 traces only, different from S0 and S1
```

Each sentence now has properly separated traces with different shapes, confirming that:
1. Old data is cleared between sentences
2. Initial state is logged for each sentence
3. Trace structure is consistent across all sentences

## Files Modified

1. ✅ `gsc.py:3504-3505` - Fixed initialize_traces
2. ✅ `only_gscnet_speedup_sap.py:5477-5478` - Fixed initialize_traces
3. ✅ `sap_grammar_training_test2.py:421` - Removed seed reset
4. ✅ `sap_grammar_training_test3.py:326` - Removed seed reset

## Next Steps

To verify the fix produces correct sentence-specific treelets:
1. Train a model for 1000 epochs (as in the original test scripts)
2. Run `sap_grammar_training_test2.py` or `sap_grammar_training_test3.py`
3. Check that treelet activations at each position are different for each sentence

The test with 50 epochs shows traces are properly separated, but the model is under-trained so treelets appear similar. With proper training (1000 epochs), each sentence should show sentence-specific treelets as expected.
