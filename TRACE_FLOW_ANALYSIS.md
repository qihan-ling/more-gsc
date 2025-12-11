# Trace Flow Analysis: Why Treelets Show Same Results

## The Problem

Both `debug_N_Vi_P_N_activation.py` and `sap_grammar_training_test2.py` use this pattern:
```python
net.reset(mu=net.ep, sd=0.01)
net.initialize_traces(trace_list='all')           # ← Call #1 (external)
for wi, word in enumerate(words):
    net.run_word(word, wi + 1, log_trace=True)   # ← Each calls initialize_traces internally!
net.run_wrapup(log_trace=True)
```

## The Hidden Behavior

When `run_word(log_trace=True)` is called, it internally calls `runC(log_trace=True)`.

Inside `runC()` at line 2651-2652:
```python
if log_trace:
    self.initialize_traces(trace_list)  # ← Call #2 (internal, EVERY time!)
```

So `initialize_traces()` is called:
1. Once by the user (before all words)
2. Once by EACH `run_word()` call
3. Once by `run_wrapup()`

## Current Buggy Behavior

### Original Code:
```python
if hasattr(self, 'traces'):
    for key in trace_list:
        self.traces[key] = list(self.traces[key])  # ← Preserves old data!
```

### Flow for Sentence S0:
```
User: initialize_traces()
  → else branch (first time)
  → creates empty dict
  → update_traces() → [initial_state]

Word 1: run_word(..., log_trace=True)
  → runC() calls initialize_traces()
  → if branch: list([initial_state]) → preserves old data
  → runC loop: update_traces() N times
  → finalize_traces() → numpy array shape (N+1, 405)
  → Traces contain: [user_initial_state, word1_states...]

Word 2: run_word(..., log_trace=True)
  → runC() calls initialize_traces()
  → if branch: list(numpy_array) → creates list of ALL rows!
  → runC loop: update_traces() M times
  → finalize_traces() → numpy array shape (N+M+1, 405)
  → Traces contain: [S0_word1_all_states..., word2_states...]

Wrapup: run_wrapup(log_trace=True)
  → runC() calls initialize_traces()
  → if branch: list(numpy_array) → preserves ALL accumulated data
  → runC loop: update_traces() K times
  → finalize_traces() → numpy array shape (N+M+K+1, 405)
```

**Result:** Single sentence accumulates traces across all words (incorrect but maybe OK)

### Flow for Sentence S1 (AFTER S0):
```
User: initialize_traces()
  → if branch (traces exist from S0)
  → list(S0_numpy_array) → creates list of ALL S0 rows!
  → NO update_traces() call in if branch!
  → Now traces = [all_S0_rows...]

Word 1: run_word(..., log_trace=True)
  → runC() calls initialize_traces()
  → if branch: list([all_S0_rows...]) → copy of list
  → Still contains all S0 data!
  → runC loop: update_traces() N times
  → finalize_traces()
  → Traces contain: [all_S0_data..., S1_word1_states...]

Word 2, Wrapup: Continue accumulating...
```

**Result:** S1 traces contain S0 + S1 data (mixed!)
**Result:** S2 traces contain S0 + S1 + S2 data (more mixed!)

This is why all sentences show the same top treelets - they're all computing over accumulated data from all previous sentences!

## Why debug_N_Vi_P_N_activation.py Works

Hypothesis: It's run in a FRESH Python session where no previous sentences have been processed.

If it's the very first sentence after loading the model:
- `initialize_traces()` goes to `else` branch (no existing traces)
- Creates fresh empty traces
- No accumulation from previous sentences

## The Real Fix Needed

The issue is that `initialize_traces()` is called INSIDE `runC()`, creating a dual-initialization problem:

1. **User's call** is meant to initialize for the WHOLE sentence
2. **runC's call** is meant to initialize for a SINGLE word/segment

These two purposes conflict!

### Option 1: Don't call initialize_traces in runC
Remove line 2652 from runC(). User must call initialize_traces() before calling run_word().

Pros: Clean separation
Cons: Breaks existing code that relies on runC() auto-initializing

### Option 2: Add a parameter to skip internal initialization
```python
def run_word(..., log_trace=False, skip_init_traces=False):
    ...
    if self.opts['use_runC']:
        self.runC(..., log_trace=log_trace, skip_init_traces=skip_init_traces)
```

### Option 3: Clear traces properly in if branch
```python
if hasattr(self, 'traces'):
    for key in trace_list:
        self.traces[key] = []  # Clear instead of preserving
```

But this has the side effect of clearing the user's initialization!

## Why My Fix Caused "Flat" Plots

With my fix `self.traces[key] = []`:
1. User calls `initialize_traces()` → clears and logs initial state
2. `run_word()` calls `runC()` which calls `initialize_traces()` → **clears the user's data!**
3. If I also add `update_traces()` here, it logs the state after set_input
4. RunC loop should still work...

Wait, this should still produce varying traces. Unless there's another issue.

## Next Steps

1. Check if `debug_N_Vi_P_N_activation.py` is being run in a fresh session
2. Check what `finalize_traces()` does and if it's causing issues
3. Verify that runC's loop is actually running and updating traces
