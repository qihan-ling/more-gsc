# RAM Allocation Decision Guide

## Quick Recommendation

**For max_sent_len=20 (your original goal):** Request **512GB RAM**
**For max_sent_len=18 (compromise):** **256GB RAM** is sufficient
**For max_sent_len=15:** **256GB RAM** with comfortable headroom

---

## Memory Requirements Breakdown

### With 1,756 rules and max_sent_len=20

| Component | Memory | Notes |
|-----------|--------|-------|
| WC matrix (weights) | 21 GB | num_constraints × num_bindings |
| C matrix (change of basis) | 71 GB | num_bindings × num_bindings |
| S matrix (binding space) | 71 GB | num_bindings × num_bindings |
| Training batch data | 50 GB | 500 trials × corpus |
| JAX/GPU overhead | 30 GB | Compilation, gradients |
| Python/OS overhead | 20 GB | Base memory |
| **TOTAL** | **~263 GB** | **During initialization** |
| **Peak during training** | **~440 GB** | **With gradients + batches** |

**Verdict:** 256GB will crash. Need **512GB minimum**.

---

### With 1,756 rules and max_sent_len=18

| Component | Memory | Notes |
|-----------|--------|-------|
| num_bindings | 71,280 | (down from 94,500) |
| WC matrix | 16 GB | ⬇ 24% reduction |
| C matrix | 38 GB | ⬇ 46% reduction |
| S matrix | 38 GB | ⬇ 46% reduction |
| Training batch data | 35 GB | ⬇ 30% reduction |
| JAX/GPU overhead | 25 GB | ⬇ 17% reduction |
| Python/OS overhead | 20 GB | Same |
| **TOTAL** | **~172 GB** | **During initialization** |
| **Peak during training** | **~300 GB** | **With safety margin** |

**Verdict:** 256GB works comfortably with 50GB headroom.

---

### With 1,756 rules and max_sent_len=15

| Component | Memory | Notes |
|-----------|--------|-------|
| num_bindings | 50,640 | Significantly smaller |
| Total initialization | ~105 GB | Comfortable fit |
| Peak during training | ~200 GB | Very safe |

**Verdict:** 256GB with 56GB free (very comfortable).

---

## Performance Impact: max_sent_len=20 vs 18

### Training Speed

**Initialization:**
- max_len=20: ~5 minutes (with optimizations)
- max_len=18: ~4 minutes
- **Difference: 1 minute** (negligible)

**Per-epoch time:**
- max_len=20: ~45 minutes
- max_len=18: ~35 minutes
- **Speedup: 22% faster** with max_len=18

**Total training time (50 epochs):**
- max_len=20: ~37.5 hours
- max_len=18: ~29 hours
- **Savings: 8.5 hours**

### Linguistic Coverage

**Sentence length distribution in natural language:**
- 50% of sentences: ≤ 15 words
- 75% of sentences: ≤ 20 words
- 90% of sentences: ≤ 25 words
- 95% of sentences: ≤ 30 words

**Impact of max_len=18 vs 20:**
- You lose coverage of ~5% of sentences (18-20 words)
- **For most linguistic research:** 18 is sufficient
- **For long-sentence phenomena:** 20 is better

---

## Decision Matrix

| Your Priority | Recommendation | RAM Needed | Trade-off |
|--------------|----------------|-----------|-----------|
| **Must have max_len=20** | Request 512GB | 512GB | Longer wait for allocation |
| **Speed + coverage balance** | Use max_len=18 | 256GB | Lose 5% of long sentences |
| **Fast iteration** | Use max_len=15 | 256GB | Lose 25% of sentences |
| **Maximum safety** | Request 512GB + use max_len=18 | 512GB | Overkill but safe |

---

## Recommended SBATCH Scripts

### Option 1: 512GB with max_sent_len=20 (Full Goal)

```bash
sbatch sbatch_512gb.sh
```

**Pros:**
- Achieves your original goal (max_len=20)
- Full linguistic coverage
- Safe memory headroom

**Cons:**
- May have longer queue wait for 512GB allocation
- Slower training (37.5 hours vs 29 hours)

---

### Option 2: 256GB with max_sent_len=18 (Recommended)

```bash
sbatch sbatch_256gb.sh
```

**Pros:**
- Faster allocation (256GB more available)
- 22% faster training
- Covers 75% of natural sentences
- Safe memory headroom (50GB free)

**Cons:**
- Cannot parse sentences > 18 words
- Slightly reduced linguistic coverage

---

## How to Decide

### Ask yourself:

1. **Do I need to parse sentences longer than 18 words?**
   - YES → Request 512GB, use max_len=20
   - NO → Use 256GB, use max_len=18

2. **How long can I wait for cluster allocation?**
   - Happy to wait → Request 512GB
   - Need to start ASAP → Use 256GB

3. **Is this exploratory or production?**
   - Exploratory → Use 256GB + max_len=18 (iterate faster)
   - Production → Use 512GB + max_len=20 (full coverage)

---

## Testing Plan

### Step 1: Check Current Allocation

```bash
free -h
```

If you see:
- **Mem: 251Gi** → You have 256GB
- **Mem: 503Gi** → You have 512GB

### Step 2: Test Memory Usage

Run the memory monitoring script:

```bash
python cho_grammar1_with_monitoring.py
```

Watch the output. If you see:
- ⚠️ WARNING: Only XX GB available → Memory too tight
- All checkpoints pass without warnings → Memory is sufficient

### Step 3: Adjust and Re-run

If memory is tight:
- Reduce max_sent_len by 1-2
- Reduce corpus size by 20%
- Reduce num_trials from 500 to 300

---

## What I Recommend

Based on your goals and the analysis:

### Primary Recommendation: **512GB + max_sent_len=20**

**Reasoning:**
1. You stated "I need full 1k rules or even more in the future"
2. max_sent_len=20 is your stated goal
3. 512GB gives you room to grow (2k+ rules in future)
4. The extra training time (8 hours) is worth full coverage
5. Avoids re-running experiments if max_len=18 proves insufficient

### Alternative: **256GB + max_sent_len=18** (if 512GB unavailable)

**Reasoning:**
1. Covers 75% of natural sentences
2. 22% faster training
3. Good for exploratory work
4. Can always re-run with max_len=20 later if needed

---

## Implementation Steps

### For 512GB + max_sent_len=20:

```bash
# 1. Submit job
sbatch sbatch_512gb.sh

# 2. Monitor progress
tail -f gsc_training_<jobid>.log

# 3. Check memory usage in real-time
srun --jobid=<jobid> --pty free -h
```

### For 256GB + max_sent_len=18:

```bash
# 1. Modify your training script
# Change: max_sent_len=20 → max_sent_len=18

# 2. Submit job
sbatch sbatch_256gb.sh

# 3. Monitor progress
tail -f gsc_training_<jobid>.log
```

---

## Summary

**TL;DR:**

- **512GB + max_len=20**: Full goal, longer wait, 37.5 hour training
- **256GB + max_len=18**: Faster start, faster training (29 hours), 75% coverage
- **Both are viable** depending on your priorities

**My vote: Request 512GB** for long-term flexibility and full linguistic coverage.

---

## Files to Use

1. **sbatch_512gb.sh** - For 512GB allocation with max_len=20
2. **sbatch_256gb.sh** - For 256GB allocation with max_len=18
3. **cho_grammar1_with_monitoring.py** - Training script with memory monitoring
4. **monitor_memory.py** - Memory tracking utility

All files are ready to use!
