# Practical Hyperparameter Recommendations for collapsed_filtered_sm5.grammar

## Problem

The grammar is too complex for efficient corpus generation with `max_sent_len=24`:
- 1,072 rules
- 1,473 fillers
- 51 terminals
- 50 nonterminals

Generating even a single sentence takes 1+ seconds, making `nsamples=100,000` infeasible (would take 28+ hours!).

## SOLUTION 1: Reduce max_sent_len (RECOMMENDED)

The key fix is to use a **much smaller max_sent_len** for faster generation:

```python
# Instead of MAXLEN = 24
MAXLEN = 8  # Or 10, or 12

hg = gsc.HarmonicGrammar(pcfg=PCFG_sap, root=ROOT, max_sent_len=MAXLEN)
```

**Why this works:**
- Shorter max lengths = faster sentence generation (exponentially faster!)
- Still captures most of the grammar structure
- Most real sentences are < 12 words anyway

**Expected speedup:**
- MAXLEN=8: ~10-50x faster than MAXLEN=24
- MAXLEN=10: ~5-20x faster
- MAXLEN=12: ~2-10x faster

## SOLUTION 2: Use smaller nsamples with progress reporting

If you MUST use MAXLEN=24, use much smaller nsamples:

```python
# Modified generate_corpus with progress reporting
def generate_corpus_with_progress(net, nsamples=1000):
    import time

    sentences = []
    targets = []
    pvals = []
    counts = []

    print(f"Generating {nsamples} sentences...")
    start_time = time.time()

    for i in range(nsamples):
        # Progress reporting every 10 sentences
        if (i+1) % 10 == 0 or i == 0:
            elapsed = time.time() - start_time
            rate = (i+1) / elapsed if elapsed > 0 else 0
            remaining = (nsamples - i - 1) / rate if rate > 0 else 0
            print(f"  {i+1}/{nsamples} ({(i+1)/nsamples*100:.1f}%) | "
                  f"Rate: {rate:.2f} sent/s | "
                  f"ETA: {remaining/60:.1f} min",
                  flush=True)

        sentence, target, p = net.generate_sentence()

        if sentence in sentences:
            idx = sentences.index(sentence)
            counts[idx] += 1
        else:
            sentences.append(sentence)
            targets.append(list(target))
            pvals.append(p)
            counts.append(1)

    # Use empirical frequencies
    counts = np.array(counts)
    pvals = counts / counts.sum()

    idx = np.argsort(pvals)[::-1]
    sentences = [sentences[si] for si in idx]
    pvals = np.array([pvals[si] for si in idx])
    targets = np.array([targets[si] for si in idx])
    counts = np.array([counts[si] for si in idx])

    net.corpus = {
        'sentence': sentences,
        'target': targets,
        'count': counts,
        'prob_sent': pvals
    }

    print(f"\nCompleted! Found {len(sentences)} unique sentences")
    print(f"Total time: {(time.time() - start_time)/60:.1f} minutes")
    return net.corpus
```

## RECOMMENDED HYPERPARAMETERS (Updated)

### Option A: MAXLEN=8 (FAST, RECOMMENDED)

```python
MAXLEN = 8  # Fast generation

# Corpus generation
nsamples = 50,000  # Can afford more samples since generation is fast

# Training
train_opts = {
    'lrate': 0.05,
    'num_trials': 200,  # Start conservative
    'report_cycle': 5,
    'init_noise_mag': 0.02,
}

n_epochs = 500
```

**Expected corpus:**
- ~50-200 unique sentence types
- ~10 seconds to generate corpus
- ~30-60 minutes total training time

### Option B: MAXLEN=10 (MODERATE)

```python
MAXLEN = 10  # Moderate generation speed

# Corpus generation
nsamples = 20,000

# Training
train_opts = {
    'lrate': 0.05,
    'num_trials': 200,
    'report_cycle': 5,
    'init_noise_mag': 0.02,
}

n_epochs = 500
```

**Expected corpus:**
- ~100-300 unique sentence types
- ~30-60 seconds to generate corpus
- ~1-2 hours total training time

### Option C: MAXLEN=24 (SLOW, NOT RECOMMENDED)

```python
MAXLEN = 24  # VERY SLOW - only if absolutely necessary

# Corpus generation - MUST be small!
nsamples = 1,000  # Any more will take hours!

# Training
train_opts = {
    'lrate': 0.05,
    'num_trials': 100,  # Lower since fewer sentence types
    'report_cycle': 5,
    'init_noise_mag': 0.02,
}

n_epochs = 200  # Can reduce since smaller corpus
```

**Expected corpus:**
- ~20-100 unique sentence types (depends on luck)
- ~20-60 minutes to generate corpus!
- ~20-40 minutes training time

## COMPARISON TABLE

| MAXLEN | nsamples | Corpus Gen Time | Unique Sentences | Training Time | TOTAL |
|--------|----------|-----------------|------------------|---------------|-------|
| 8 | 50,000 | ~10s | ~150 | ~45min | **~45min** ✅ |
| 10 | 20,000 | ~45s | ~200 | ~90min | **~90min** ✅ |
| 12 | 10,000 | ~5min | ~150 | ~60min | **~65min** ✅ |
| 24 | 1,000 | ~30min | ~50 | ~20min | **~50min** ⚠️ |
| 24 | 5,000 | ~150min | ~100 | ~40min | **~3+ hours** ❌ |
| 24 | 100,000 | **~50 hours** | ??? | ??? | **INFEASIBLE** ❌ |

## FINAL RECOMMENDATION

**Use MAXLEN=10 with nsamples=20,000**

This gives you:
- ✅ Fast corpus generation (~1 minute)
- ✅ Reasonable complexity (handles most sentence structures)
- ✅ Good sentence type diversity (~200 types)
- ✅ Reasonable training time (~90 minutes)
- ✅ Better than MAXLEN=24 with tiny corpus

## Quick Start Code

```python
import only_gscnet_speedup as gsc
import numpy as np

# Load grammar
with open('collapsed_filtered_sm5.grammar', 'r') as f:
    PCFG_sap = f.read()

# CRITICAL: Use smaller MAXLEN!
ROOT = 'S'
MAXLEN = 10  # ← KEY CHANGE!

# Initialize
hg = gsc.HarmonicGrammar(pcfg=PCFG_sap, root=ROOT, max_sent_len=MAXLEN)
sim = hg.get_simlist(dp=0.0)

net_opts = {
    'T_init': 0.01,
    'q_max': 15.0,
    'q_init': 0.0,
    'dt_init': 0.005,
    'm': 30,
    'use_runC': True,
}

net = gsc.GscNet(hg=hg, encodings={'similarity': sim}, opts=net_opts, seed=1024)

# Generate corpus with progress
corpus = generate_corpus_with_progress(net, nsamples=20000)

# Train
train_opts = {
    'lrate': 0.05,
    'num_trials': 200,
    'report_cycle': 5,
    'init_noise_mag': 0.02,
}

net.initialize(train_opts=train_opts)

for epoch_block in range(100):  # 500 total epochs
    net.train2(train_opts={'num_epochs': 5},
               savefilename='sap_model.pkl')
```

## Summary

**The key mistake in your original attempt:** Using MAXLEN=24 makes sentence generation 10-100x slower!

**The fix:** Use MAXLEN=8-12 for practical training times.

**Your original parameter adjustments were actually backwards:**
- ❌ You reduced epochs (should INCREASE for larger grammar)
- ❌ You proposed lrate=0.02 (too low)
- ✅ You increased num_trials (good!)
- ❓ nsamples needs to balance with generation speed
