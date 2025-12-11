# Final Fix: Treelet Accumulation Bug

## The Problem

When processing multiple sentences in a loop (like in `sap_grammar_training_test2.py`), all sentences showed the SAME top 4 treelets because traces were accumulating data from ALL previous sentences.

## Root Cause

The `initialize_traces()` method at:
- `gsc.py:3504`
- `only_gscnet_speedup_sap.py:5477`

Had this buggy code:
```python
if hasattr(self, 'traces'):
    for key in trace_list:
        self.traces[key] = list(self.traces[key])  # ← BUG: Preserves old data!
```

When `self.traces[key]` is a numpy array (from `finalize_traces()`), calling `list()` on it creates a list of all rows, **preserving all old data**.

## Why debug_N_Vi_P_N_activation.py Worked

It processes only ONE sentence in isolation, so there's no accumulated data from previous sentences to mix in.

## Why sap_grammar_training_test2.py Failed

It loops over ALL sentences:
```python
for si, (sent, targ) in enumerate(zip(net.corpus['sentence'], net.corpus['target'])):
    filename = plot_sentence_treelets(net, sent, si, targ)
```

With the bug:
- **S0**: Clean data ✓
- **S1**: S0 + S1 mixed data ✗
- **S2**: S0 + S1 + S2 mixed data ✗
- **S3**: S0 + S1 + S2 + S3 mixed data ✗

All sentences compute treelet activations over this accumulated mixed data, causing them to show the same top treelets!

## The Hidden Complexity: Double Initialization

**Important:** `runC()` internally calls `initialize_traces()` when `log_trace=True`:

```python
def runC(self, duration, ... log_trace=False, ...):
    if log_trace:
        self.initialize_traces(trace_list)  # ← Internal call!

    for self.step in range(num_steps):
        if log_trace:
            self.update_traces()

    if log_trace:
        self.finalize_traces()
```

So when user code does:
```python
net.initialize_traces(trace_list='all')  # External call
net.run_word(word, 1, log_trace=True)    # Internal initialize_traces() call!
```

The external call is immediately overwritten by the internal call. This is why we only need to fix the if-branch to clear data, without adding `update_traces()`.

## The Fix

Simply clear old data instead of preserving it:
```python
if hasattr(self, 'traces'):
    for key in trace_list:
        self.traces[key] = []  # ← Clear old data!
else:
    self.traces = {}
    for key in trace_list:
        self.traces[key] = []
    self.update_traces()  # ← Only in else branch
```

**Why NOT add `update_traces()` to if-branch:**
- The user's external `initialize_traces()` is overwritten by runC's internal call
- runC handles all trace logging internally
- Adding `update_traces()` would create an extra entry that gets immediately cleared

## Result

With this fix:
- **S0**: Clean S0 data only ✓
- **S1**: Clean S1 data only ✓
- **S2**: Clean S2 data only ✓
- **S3**: Clean S3 data only ✓

Each sentence gets its own clean traces, producing sentence-specific treelet activations!

## Files Modified

1. `gsc.py:3504` - Changed `list(self.traces[key])` to `[]`
2. `only_gscnet_speedup_sap.py:5477` - Changed `list(self.traces[key])` to `[]`

Commit: b014fa9 (reverted from incorrect fix in 25f4ce9)
