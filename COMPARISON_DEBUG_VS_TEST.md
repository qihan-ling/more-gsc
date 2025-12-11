# Why debug_N_Vi_P_N_activation.py Works but sap_grammar_training_test2.py Fails

## Key Insight: The Double-Initialization Problem

Both scripts use the same pattern, but **runC() internally calls initialize_traces()** which interferes with the user's call.

## The Hidden Call Inside runC()

At `only_gscnet_speedup_sap.py:2651-2652`:
```python
def runC(self, duration, ... log_trace=False, ...):
    ...
    if log_trace:
        self.initialize_traces(trace_list)  # ← INTERNAL CALL!

    # Loop and update traces
    for self.step in range(num_steps):
        ...
        if log_trace:
            self.update_traces()

    if log_trace:
        self.finalize_traces()  # ← Converts list to numpy array
```

So every call to `run_word(log_trace=True)` triggers:
1. `initialize_traces()` inside `runC()`
2. Multiple `update_traces()` calls in the loop
3. `finalize_traces()` to convert to numpy array

## Comparison

### debug_N_Vi_P_N_activation.py (WORKS)

```python
# Line 14: Load model
net = gsc.load_model('ds_jax_sap_test_on_g1_model.pkl')

# Lines 132-141: Process ONE sentence
net.reset(mu=net.ep, sd=0.01)
net.initialize_traces(trace_list='all')  # External call

for wi, word in enumerate(words):
    net.run_word(word, wi + 1, log_trace=True)  # Internal initialize_traces() call
net.run_wrapup(log_trace=True)

# Line 166-168: Use traces
actC_trace = net.traces['actC']
dp_all = gsc.compute_treelet_act_trace(net, actC_trace, rules, rname)
```

**Why it works:**
- Processes ONLY ONE sentence (S3)
- Fresh model load means `self.traces` doesn't exist initially
- First `initialize_traces()` goes to `else` branch → creates empty dict
- Subsequent calls (from runC) go to `if` branch → `list(self.traces[key])`
- Since it's the FIRST sentence after loading, there's no accumulated data from previous sentences!

### sap_grammar_training_test2.py (FAILS)

```python
# Line 239: Load model
net = gsc.load_model('sap_g1_model_fixed_sparse_nocompress.pkl')

# Lines 301-327: Test parsing accuracy using test_parse_inc()
# This calls run_sent() with log_trace=False, so no traces are created
# But it does process all sentences, potentially leaving state

# Lines 478-482: Loop over ALL sentences
for si, (sent, targ) in enumerate(zip(net.corpus['sentence'], net.corpus['target'])):
    filename = plot_sentence_treelets(net, sent, si, targ)
```

Inside `plot_sentence_treelets()` for each sentence:
```python
def plot_sentence_treelets(net, sent, sent_idx, target):
    net.reset(mu=net.ep, sd=0.01)
    net.initialize_traces(trace_list='all')  # External call

    for wi, word in enumerate(words):
        net.run_word(word, wi + 1, log_trace=True)
    net.run_wrapup(log_trace=True)

    gsc.plot_treelet_act_trace(net, rname='(3,2)', ...)  # Uses net.traces
```

**Why it fails:**

#### For Sentence S0:
```
External initialize_traces():
  → hasattr(self, 'traces') = False (first time)
  → else branch: creates empty dict, update_traces()
  → traces = [initial_state]

run_word(word1, log_trace=True):
  → runC internally calls initialize_traces()
  → hasattr(self, 'traces') = True
  → if branch: self.traces[key] = list([initial_state])  ← Keeps old data!
  → runC loop: update_traces() N times
  → finalize_traces() → numpy array
  → traces = [initial, word1_state1, word1_state2, ..., word1_stateN]

run_word(word2, log_trace=True):
  → runC internally calls initialize_traces()
  → self.traces[key] is numpy array
  → if branch: self.traces[key] = list(numpy_array)  ← List of ALL rows!
  → runC loop: update_traces() M times
  → traces = [initial, word1_all, word2_state1, ..., word2_stateM]

Final S0 traces: Mixed data from all words in S0
```

#### For Sentence S1 (AFTER S0):
```
External initialize_traces():
  → hasattr(self, 'traces') = True (from S0)
  → if branch: self.traces[key] = list(S0_numpy_array)  ← Keeps ALL S0 data!
  → NO update_traces() call in if branch
  → traces = [all_S0_rows...]

run_word(word1, log_trace=True):
  → runC internally calls initialize_traces()
  → self.traces[key] is a list with all S0 data
  → if branch: self.traces[key] = list([all_S0_data...])  ← Still has S0 data!
  → runC loop: update_traces() N times
  → traces = [all_S0_data..., S1_word1_states...]

Final S1 traces: MIXED S0 + S1 data!
```

#### For Sentence S2 (AFTER S0, S1):
```
Similar accumulation: S0 + S1 + S2 mixed data
```

## The Root Cause

The `initialize_traces()` if-branch preserves old data:
```python
if hasattr(self, 'traces'):
    for key in trace_list:
        self.traces[key] = list(self.traces[key])  # ← BUG: Preserves data!
```

When processing multiple sentences in a loop:
- S0 gets clean data ✓
- S1 gets S0+S1 mixed data ✗
- S2 gets S0+S1+S2 mixed data ✗
- S3 gets S0+S1+S2+S3 mixed data ✗

When computing treelet activations over this mixed data, all sentences show similar results because they're all computing over accumulated data from ALL previous sentences!

## The Correct Fix

Change the if-branch to clear old data instead of preserving it:
```python
if hasattr(self, 'traces'):
    for key in trace_list:
        self.traces[key] = []  # ← Clear old data!
    # Do NOT add update_traces() here - runC will handle it
```

**Why NOT add update_traces():**
The user's external `initialize_traces()` call is immediately overwritten by runC's internal call anyway, so adding `update_traces()` to the if-branch would be redundant and might cause confusion.

The real fix is just to ensure old data is cleared, not accumulated.

## Test to Verify

Run `debug_N_Vi_P_N_activation.py` on S3 AFTER running it on S0, S1, S2 in the same Python session. With the bug, S3 will show mixed results. With the fix, S3 will show correct results.
