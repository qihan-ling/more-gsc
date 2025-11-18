# Training Strategy for MAXLEN=24 (Cannot Reduce)

## Reality Check

With MAXLEN=24 and 1,072 rules, sentence generation is **inherently slow**. There's no way around this if you must use MAXLEN=24. Here's what to expect:

### Generation Speed Estimates

Based on the grammar complexity:
- **Per sentence:** 0.5 - 3 seconds (varies with random tree depth)
- **Fast estimate:** 2 sentences/second
- **Slow estimate:** 0.3 sentences/second

### Time Estimates for Different nsamples

| nsamples | Fast (2 sent/s) | Slow (0.3 sent/s) | Unique Sentences (est) |
|----------|----------------|-------------------|------------------------|
| 1,000 | **8 minutes** | **55 minutes** | 20-50 |
| 2,500 | **21 minutes** | **2.3 hours** | 50-100 |
| 5,000 | **42 minutes** | **4.6 hours** | 80-150 |
| 10,000 | **1.4 hours** | **9.3 hours** | 120-250 |
| 50,000 | **7 hours** | **46 hours** | 300-500 |
| 100,000 | **14 hours** | **92 hours** | 400-700 |

## ✅ RECOMMENDED STRATEGY

### Option 1: Overnight Run (RECOMMENDED)

```python
# Start before bed, let it run overnight
nsamples = 5,000  # ~1-5 hours generation
n_epochs = 400
# Total time: 2-6 hours
```

**Pros:**
- Reasonable corpus diversity (80-150 unique sentences)
- Good probability estimates (~33 samples per sentence)
- Enough data for meaningful training

**Cons:**
- Still requires several hours
- May not capture all rare sentence types

### Option 2: Quick Test Run

```python
# Fast test to verify everything works
nsamples = 1,000  # ~10-60 minutes generation
n_epochs = 300
# Total time: 30-90 minutes
```

**Pros:**
- Faster feedback
- Good for debugging/testing

**Cons:**
- Small corpus (20-50 unique sentences)
- Poor probability estimates (~20 samples per sentence)
- May underfit

### Option 3: Weekend Long Run

```python
# Maximum quality, requires patience
nsamples = 10,000  # ~2-10 hours generation
n_epochs = 500
# Total time: 3-12 hours
```

**Pros:**
- Large corpus (120-250 unique sentences)
- Best probability estimates (~40-80 samples per sentence)
- Most complete grammar coverage

**Cons:**
- Very long runtime
- Need to babysit or run on server

## 📋 UPDATED HYPERPARAMETER RECOMMENDATIONS

### For nsamples = 1,000 (Quick Test)

```python
lrate = 0.1              # Higher LR for small corpus
num_trials = 100         # 2x expected unique sentences (~50)
n_epochs = 300           # Fewer epochs for small corpus
```

### For nsamples = 2,500 (Overnight)

```python
lrate = 0.1              # Higher LR
num_trials = 150         # 2x expected unique (~75)
n_epochs = 400
```

### For nsamples = 5,000 (RECOMMENDED)

```python
lrate = 0.05-0.1         # Start at 0.1
num_trials = 200         # 2x expected unique (~100)
n_epochs = 500
```

### For nsamples = 10,000 (Weekend)

```python
lrate = 0.05             # More conservative
num_trials = 300         # 2x expected unique (~150)
n_epochs = 500
```

## 🚀 HOW TO RUN

### Step 1: Run the Optimized Script

```bash
python sap_grammar_training_maxlen24.py
```

The script will:
1. Ask you to choose corpus size (1000/2500/5000)
2. Check for cached corpus (so you don't regenerate if it crashes)
3. Show detailed progress with ETA
4. Auto-tune hyperparameters based on corpus size
5. Save checkpoints every 5 epochs

### Step 2: Monitor Progress

You'll see output like:
```
[  100/5000] ( 2.0%) | Rate:  1.85 sent/s | Last:  0.6s | Unique:  42 | ETA: 44.1min
[  200/5000] ( 4.0%) | Rate:  1.92 sent/s | Last:  0.5s | Unique:  67 | ETA: 41.6min
...
```

**What to watch:**
- **Rate**: Should be 0.3-3.0 sent/s (varies randomly)
- **Unique**: Should grow slower as you get more samples
- **ETA**: Tells you when it will finish

### Step 3: Be Patient During Generation

**This is NORMAL:**
- Sometimes a sentence takes 5+ seconds (deep tree)
- Rate fluctuates wildly (0.2 to 5.0 sent/s)
- Long periods with no output (generating deep trees)

**If it seems stuck:**
- Wait at least 60 seconds before panicking
- Check that "Rate" is not 0.00
- If Rate > 0, it's working (just slow)

## ⚡ OPTIMIZATIONS ALREADY INCLUDED

The script includes several optimizations:

1. **Corpus Caching**
   - Saves generated corpus to disk
   - If training crashes, you can reload without regenerating
   - Filename: `corpus_cache_maxlen24_n{nsamples}.pkl`

2. **Adaptive Hyperparameters**
   - Automatically adjusts `num_trials` based on corpus size
   - Auto-reduces learning rate if convergence is slow
   - Smart epoch count based on corpus diversity

3. **Progress Reporting**
   - Updates every 10 sentences
   - Shows generation rate and ETA
   - Reports unique sentence count

4. **Efficient JAX Training**
   - Uses JAX-optimized dynamics loop
   - Parallel trial execution on GPU
   - Fast parameter updates

## 🎯 REALISTIC EXPECTATIONS

### What You CAN Achieve with MAXLEN=24

✅ Train a working parser for the grammar
✅ Learn probability distributions over sentences
✅ Achieve reasonable KL divergence and accuracy
✅ Capture major sentence patterns

### What You CANNOT Achieve

❌ Sample the full space (would need millions of sentences)
❌ Get perfect probability estimates for rare sentences
❌ Complete in < 1 hour with good corpus size
❌ Match the diversity of MAXLEN=10 with same nsamples

## 📊 COMPARISON: Your Original Settings vs. Reality

| Your Proposal | Reality Check | Recommendation |
|---------------|---------------|----------------|
| nsamples = 100,000 | ❌ Would take **14-92 hours**! | ✅ Use 5,000 (1-5 hours) |
| lrate = 0.02 | ⚠️ Too conservative for small corpus | ✅ Use 0.1 |
| num_trials = 500 | ⚠️ Too many for small corpus | ✅ Use 150-200 |
| n_epochs = 100 | ❌ Too few for complex grammar | ✅ Use 400-500 |

## 🔧 TROUBLESHOOTING

### "It's been running for 2 hours with no output"

- Check if the process is still running
- Look at the last reported rate
- If rate was > 0, it's working (just very slow)
- Consider reducing nsamples

### "Generation rate is < 0.5 sent/s"

- This is expected with MAXLEN=24
- The grammar generates deep trees sometimes
- Nothing is wrong, just be patient

### "I need faster results"

Your options:
1. Reduce nsamples (but get smaller corpus)
2. Use multiple random seeds in parallel
3. Run overnight
4. Consider if MAXLEN=24 is truly required

### "Can I speed up generation somehow?"

No, not without modifying the grammar or reducing MAXLEN. The bottleneck is:
- 1,072 rules → many choices at each node
- MAXLEN=24 → very deep recursion
- Binary tree structure → exponential search space

## 💡 ALTERNATIVE APPROACH: Staged Training

If you really need MAXLEN=24 but want faster iteration:

1. **Stage 1:** Train with MAXLEN=10, nsamples=20,000 (fast, ~2 min)
2. **Stage 2:** Generate small corpus with MAXLEN=24, nsamples=1,000
3. **Stage 3:** Fine-tune the MAXLEN=10 model on MAXLEN=24 data

This gives you:
- Fast initial training
- Opportunity to debug with fast feedback
- Final model with MAXLEN=24 capability

## 🎬 RECOMMENDED WORKFLOW

```bash
# Day 1: Quick test
python sap_grammar_training_maxlen24.py
# Choose option 1 (1,000 samples, ~30-60 min total)
# Verify everything works

# Day 2: Overnight run
python sap_grammar_training_maxlen24.py
# Choose option 3 (5,000 samples)
# Start before bed, check in morning
# Should complete in 2-6 hours

# Day 3: Analysis
# Examine results, tune hyperparameters if needed
```

## ✅ SUMMARY

**Bottom line:** With MAXLEN=24, you must accept slow generation. The optimized script makes it bearable by:

1. Using realistic nsamples (1,000-5,000)
2. Providing progress feedback
3. Caching corpus to prevent data loss
4. Auto-tuning hyperparameters
5. Efficient JAX training

**Recommended settings:**
- **nsamples:** 5,000 (overnight run)
- **lrate:** 0.1
- **num_trials:** 200
- **n_epochs:** 500

**Total expected time:** 2-6 hours (mostly corpus generation)

This is the best you can do without reducing MAXLEN. Good luck! 🍀
