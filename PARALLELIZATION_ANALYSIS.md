# Analysis: Functions with Trial Loops for JAX Parallelization

This document identifies all functions in gsc.py that use trial loops and analyzes their potential for JAX parallelization to speed up training and inference.

---

## Summary Table

| Function | Type | Trial Loop | Parallelizable? | Priority | Complexity | Est. Speedup |
|----------|------|------------|-----------------|----------|------------|--------------|
| `estimate_prob_inc()` | Inference | ✅ Yes (5678) | ✅ **DONE** | - | - | 2.5×-107× |
| `estimate_prob_inc_jax()` | Inference | ✅ Parallel | ✅ **DONE** | - | - | Currently used |
| `test()` | Inference | ✅ Yes (6764) | ✅ High | **HIGH** | Medium | 5-50× |
| `parse2()` | Inference | ✅ Yes (7366) | ✅ High | Medium | Low | 2-10× |
| `test_parse_inc()` | Inference | ✅ Yes (7456) | ✅ High | **HIGH** | Medium | 10-100× |
| `train_inc_parser()` | Training | ✅ Yes (7667) | ⚠️ Partial | Low | **High** | Limited |
| `train_parser()` | Training | ⚠️ Inner loop | ⚠️ Partial | Medium | **Very High** | 5-20× |

---

## Detailed Analysis

### 1. `estimate_prob_inc()` - ✅ ALREADY PARALLELIZED

**Location**: Lines 5669-5712
**Trial Loop**: Line 5678: `for trial_id in range(num_trials)`
**Status**: ✅ **Replaced by `estimate_prob_inc_jax()`**

**What it does**:
- Runs `num_trials` independent stochastic parsing trials
- Each trial: reset → run_prefix → run_wrapup → extract tree
- Aggregates tree distributions across trials

**JAX Implementation**:
- Already implemented as `estimate_prob_inc_jax()` (lines 5717-5795)
- Uses `vmap` for parallel trial execution
- Speedup: 2.5× (10 trials) to 107× (500 trials)

---

### 2. `test()` - ⚠️ NEEDS JAX VERSION

**Location**: Lines 6744-6782
**Trial Loop**: Line 6764: `for ti in range(num_trials)`
**Nested Loop**: Outer loop over sentences (line 6754)

**What it does**:
- Tests model accuracy on corpus sentences
- For each sentence:
  - Runs `num_trials` parsing attempts
  - Computes accuracy = fraction of correct parses
- Reports accuracy per sentence

**Parallelization Potential**: **HIGH**
- Inner trial loop is fully independent (reset for each trial)
- Outer sentence loop is also independent
- **Double parallelization possible**: batch over (sentences × trials)

**JAX Implementation Strategy**:
```python
def test_jax(self, num_trials=10):
    """JAX-accelerated version of test()."""
    # Extract parameters
    net_params = _extract_net_params_for_jax(self)

    # For each sentence in corpus
    results = []
    for si, sent in enumerate(self.corpus['sentence']):
        targ = self.corpus['target'][si]

        # Run num_trials in parallel
        rng_keys = jax.random.split(jax.random.PRNGKey(si), num_trials)
        # Batch process: all trials for this sentence
        actC_batch, grid_point_batch = _run_trials_batched_jax(
            rng_keys, net_params, sent, update_q_discrete=False
        )

        # Check accuracy
        accuracy = compute_accuracy_batch(actC_batch, targ)
        results.append(accuracy)

        print(f'Sentence {si} ACC = {accuracy:.3f}')

    return np.array(results)
```

**Estimated Speedup**: 5-50× (depending on num_trials and num_sentences)

**Complexity**: Medium
- Needs to handle full sentence input (not just prefix)
- Similar to existing `estimate_prob_inc_jax()`
- Main difference: checking exact match vs distribution

**Priority**: **HIGH** - Testing is frequently used during model evaluation

---

### 3. `parse2()` - ⚠️ NEEDS JAX VERSION

**Location**: Lines 7344-7394
**Trial Loop**: Line 7366: `for ti in range(num_trials)`

**What it does**:
- Parses a single generated sentence multiple times
- For each trial:
  - Reset network
  - Process sentence word-by-word with incremental input
  - Runs wrapup
  - Checks correctness
- Primarily for debugging/visualization

**Parallelization Potential**: **HIGH**
- Trial loop is fully independent
- Each trial starts with reset

**JAX Implementation Strategy**:
```python
def parse2_jax(net, dq, num_trials=1, ...):
    """JAX-accelerated version of parse2()."""
    net_params = _extract_net_params_for_jax(net)

    # Generate sentence
    sent, targ, p = net.generate_sentence(...)

    # Run all trials in parallel
    rng_keys = jax.random.split(jax.random.PRNGKey(0), num_trials)
    actC_batch, grid_point_batch = _run_trials_batched_jax(
        rng_keys, net_params, sent, update_q_discrete=False
    )

    # Check correctness for each trial
    for ti in range(num_trials):
        if np.allclose(actC_batch[ti], targ):
            print('Correct')
        else:
            print('False')

    return net
```

**Estimated Speedup**: 2-10× (usually small num_trials)

**Complexity**: Low
- Very similar to existing JAX infrastructure
- Just needs sentence input handling

**Priority**: Medium - Mainly used for debugging, not production

---

### 4. `test_parse_inc()` - ⚠️ NEEDS JAX VERSION

**Location**: Lines 7413-7480
**Trial Loop**: Line 7456: `for ti in range(num_trials)`
**Nested Loop**: Outer loop over sentences (line 7436)

**What it does**:
- Tests incremental parser accuracy on corpus
- For each sentence:
  - Runs `num_trials` parsing attempts
  - Uses `run_sent()` for incremental word-by-word processing
  - Computes accuracy
- Returns detailed results dict with correct/incorrect parses

**Parallelization Potential**: **VERY HIGH**
- Double nested loop: sentences × trials
- Both loops are independent
- This is likely the **most time-consuming** inference function

**JAX Implementation Strategy**:
```python
def test_parse_inc_jax(net, dq, num_sent=None, num_trials=10, ...):
    """JAX-accelerated version of test_parse_inc()."""
    net_params = _extract_net_params_for_jax(net)
    net.qpolicy = dq.cumsum()

    res = {}
    for si in range(num_sent):
        sent0 = net.corpus['sentence'][si]
        targ = net.corpus['target'][si]

        # Run all trials in parallel for this sentence
        rng_keys = jax.random.split(jax.random.PRNGKey(si), num_trials)
        actC_batch, grid_point_batch = _run_trials_batched_jax(
            rng_keys, net_params, sent0, update_q_discrete=False
        )

        # Compute accuracy
        correct = np.array([np.allclose(actC_batch[ti], targ)
                           for ti in range(num_trials)])
        acc = correct.mean()

        res[si] = {
            'sentence': sent0,
            'parse_corr': targ,
            'acc': acc,
            'parse_incorr': actC_batch[~correct]
        }

        print(f'Sentence {si} ACC = {acc:.3f}')

    return res
```

**Estimated Speedup**: 10-100× (high num_trials, many sentences)

**Complexity**: Medium
- Similar structure to `test_jax()`
- Needs to handle `run_sent()` logic (incremental processing)
- Already have prefix handling infrastructure

**Priority**: **VERY HIGH** - This is likely the **bottleneck** during evaluation/testing

---

### 5. `train_inc_parser()` - ⚠️ DIFFICULT TO PARALLELIZE

**Location**: Lines 7658-7737
**Trial Loop**: Line 7667: `for ti in range(num_trials)`

**What it does**:
- Training loop for incremental parser
- For each trial:
  - Generate random sentence
  - Parse it
  - **Update qpolicy based on errors** (lines 7688-7714)
- Updates are **sequential** and **stateful**

**Parallelization Potential**: **LOW**
- Trials are **NOT independent** - each modifies `net.qpolicy`
- Updates depend on previous state
- This is **online learning** with sequential updates

**Partial Parallelization Strategy**:
Could parallelize the **forward pass** (parsing) but not the **updates**:
```python
def train_inc_parser_jax(net, num_trials, lrate=0.1):
    """Partially parallelized training."""

    # Could batch-generate sentences
    sentences = [net.generate_sentence() for _ in range(num_trials)]

    # Parallel forward passes (but still sequential updates)
    for ti in range(num_trials):
        sent, targ, p = sentences[ti]

        # This part could be JAX
        actC, grid_point = _run_single_trial_jax(...)

        # But this part must remain sequential (updates qpolicy)
        if not np.allclose(actC, targ):
            # Update qpolicy based on error
            net.qpolicy = update_qpolicy(net.qpolicy, error_pos, lrate)
```

**Estimated Speedup**: Limited (maybe 2-3× if we parallelize forward passes)

**Complexity**: **VERY HIGH**
- Training logic is complex and stateful
- Would require major restructuring
- Low benefit for high cost

**Priority**: Low - Sequential nature limits parallelization benefits

---

### 6. `train_parser()` (estimate_prob_inc inside training) - ⚠️ ALREADY USES JAX

**Location**: Lines 5837+ (inside training loop)

**What it does**:
- Main training loop for the parser
- Calls `estimate_prob_inc()` inside training iterations
- This is where **most training time** is spent

**Current Status**:
- If `estimate_prob_inc_jax()` is called instead of `estimate_prob_inc()`, training is **already accelerated**
- The outer training loop updates parameters sequentially (can't parallelize)
- Inner probability estimation is parallelizable (already done)

**Action Needed**:
Check if training code calls `estimate_prob_inc()` or `estimate_prob_inc_jax()`:
```python
# Search for this pattern in training code
stat = self.estimate_prob_inc(...)  # Should be changed to:
stat = self.estimate_prob_inc_jax(...)  # For speedup
```

**Estimated Speedup**: 10-100× (if switching to JAX version)

**Complexity**: Low - Just change function call

**Priority**: **CRITICAL** if training uses CPU version

---

## Recommendations

### Immediate Actions (High Priority)

1. **Create `test_jax()`** (Lines 6744-6782)
   - **Impact**: Speed up model testing significantly
   - **Effort**: Medium (1-2 hours)
   - **Benefit**: Essential for fast model evaluation

2. **Create `test_parse_inc_jax()`** (Lines 7413-7480)
   - **Impact**: Speed up incremental parser testing
   - **Effort**: Medium (2-3 hours)
   - **Benefit**: This is likely the **biggest bottleneck** in evaluation
   - **Note**: Very similar to existing JAX infrastructure

3. **Verify training uses JAX version**
   - Check if `train_parser()` calls `estimate_prob_inc_jax()`
   - If not, update to use JAX version
   - **Impact**: Massive training speedup (10-100×)
   - **Effort**: Minimal (5 minutes)

### Medium Priority

4. **Create `parse2_jax()`** (Lines 7344-7394)
   - **Impact**: Mainly for debugging/visualization
   - **Effort**: Low (1 hour)
   - **Benefit**: Nice to have, not critical

### Low Priority (Not Recommended)

5. **Don't parallelize `train_inc_parser()`**
   - Sequential updates make parallelization impractical
   - Benefit is minimal, cost is high
   - Leave as CPU-only

---

## Implementation Priority Order

**Phase 1: High Impact, Low Effort** ⭐⭐⭐
1. Verify training uses `estimate_prob_inc_jax()` (5 min)
2. Create `test_jax()` (1-2 hours)

**Phase 2: High Impact, Medium Effort** ⭐⭐
3. Create `test_parse_inc_jax()` (2-3 hours)

**Phase 3: Low Impact, Low Effort** ⭐
4. Create `parse2_jax()` (1 hour)

**Not Recommended**:
- `train_inc_parser()` parallelization (high cost, low benefit)

---

## Summary Statistics

- **Total functions analyzed**: 7
- **Already parallelized**: 1 (`estimate_prob_inc_jax`)
- **High priority for parallelization**: 2 (`test`, `test_parse_inc`)
- **Medium priority**: 1 (`parse2`)
- **Low priority / not recommended**: 2 (`train_inc_parser`, partial training)

**Potential total speedup**: If all high-priority functions are parallelized, expect **10-100× faster** inference and testing workloads.
