# Analysis: Original gsc.py Model Performance

## Key Discovery

The file `sap_g1_original_gsc_model_round1.output` shows **PERFECT parsing** results that are dramatically better than current round14 models:

### Parsing Accuracy at Commitment t=12

| Model | S0 | S1 | S2 | S3 | S4 |
|-------|----|----|----|----|----|
| **Original gsc (round1)** | **1.000** | **1.000** | **1.000** | **1.000** ✅ | **1.000** ✅ |
| Round14 Dense | 1.000 | 0.900 | 0.900 | 0.500 ❌ | 0.000 ❌ |
| Round14 Sparse | 1.000 | 1.000 | 1.000 | 0.200 ❌ | 0.000 ❌ |

### Performance Across All Commitment Levels

**Original gsc model:**
- S3 maintains **1.0 accuracy** at ALL commitment levels (t=1-12)
- S4 reaches **1.0 accuracy from t=7 onwards** (0.5-0.9 at t=1-6)

**Current round14 models:**
- S3 drops to 0.2-0.5 at high commitment
- S4 **completely fails** (0.0) from t=5 onwards

---

## What Produced the Original Results?

### Script Used
- File: `sap_grammar_training_test3.py`
- Implementation: **`import gsc as gsc`** (original gsc.py, NOT SAP version)
- Model: Loads pre-trained `sap_g1_model_orig.pkl`

### Configuration from Output File

```
Global random seed set to 41 for testing
use_sparse: False
WC type: numpy.ndarray (dense)
WC shape: (405, 405)
dim_f used: 27 (FULL dimension, not compressed)
dim_r used: 15 (FULL dimension, not compressed)
num_fillers: 27
num_roles: 15
num_bindings: 405
```

---

## Critical Questions

### 1. When Was sap_g1_model_orig.pkl Created?

The model file is loaded but not in git (too large). The test script was added today (Dec 6, 2025) in commit dc1cef3.

**Need to find**: The training script that created this model.

### 2. What's Different from Round14?

Both use:
- ✅ Dense matrices (not sparse)
- ✅ Full dimensions (not compressed)
- ✅ Same G1 grammar
- ✅ Same hyperparameters (T=0.01, q_max=15, dt=0.005, m=30)

**Possible differences:**
- Different `gsc.py` version (commit/branch)?
- Different training procedure?
- Different random seed during training?
- Different number of epochs?
- Hidden parameter differences?

### 3. gsc.py vs only_gscnet_speedup_sap.py

The original uses **`gsc.py`** while round14 uses **`only_gscnet_speedup_sap.py`**.

From earlier investigation:
- SAP version FIXED a bug in gsc.py (competition rules typo)
- But that bug isn't triggered for G1 grammar
- Both should be functionally equivalent for G1

**Yet the results are drastically different!**

---

## Hypothesis: Training vs Testing Code Difference

### Possibility 1: Different Training Dynamics
Even if the dynamics equations are identical, subtle numerical differences could lead to different training trajectories and final weights.

**Check:**
- Are there JAX vs NumPy differences during training?
- Different batch sizes or trial counts?
- Different optimization (Adam) parameters?

### Possibility 2: Different Testing Code
Maybe `gsc.test_parse_inc()` differs from `only_gscnet_speedup_sap.test_parse_inc()` in subtle ways.

**But**: We compared test_parse_inc and found them identical!

### Possibility 3: Model Wasn't Actually Trained with gsc.py
Maybe `sap_g1_model_orig.pkl` was actually trained with a different implementation (only_gscnet_speedup.py with JAX?) and just tested with gsc.py.

**Need to verify**: What script actually created the .pkl file?

---

## Recommended Investigation Steps

### Step 1: Find the Training Script
```bash
# Search for any script that saves to sap_g1_model_orig.pkl
git log --all -S "savefilename.*orig" --source --all

# Check if there's a training script we missed
ls -la *.py | grep -i train
```

### Step 2: Compare gsc.py Versions
```bash
# Check if gsc.py has changed significantly
git log --oneline -- gsc.py | head -20

# Compare current gsc.py with version from successful training
git diff <good_commit> HEAD -- gsc.py
```

### Step 3: Reproduce the Original Training
```python
# Use original gsc.py to train a new model
import gsc
import numpy as np

np.random.seed(41)  # Same as original

hg = gsc.HarmonicGrammar(pcfg=PCFG_G1, root='S', max_sent_len=5)
sim = hg.get_simlist(dp=0.0)

net = gsc.GscNet(hg=hg, encodings={'similarity': sim},
                 opts={
                     'T_init': 0.01,
                     'q_max': 15.0,
                     'q_init': 0.0,
                     'dt_init': 0.005,
                     'm': 30,
                     'use_runC': True,
                     'ep_method': 'integration',
                 },
                 seed=1024)

net.generate_corpus(use_freq=True)
net.initialize(train_opts={'lrate': 0.1, 'num_trials': 4})

# Train for 1000 epochs like original
for i in range(100):
    net.train2(train_opts={'num_epochs': 10},
               savefilename='test_orig_reproduction.pkl')

# Test parsing
dq = np.ones(5) * (12.0 / 5.0)
results = gsc.test_parse_inc(net, dq=dq, num_trials=10)
print(f"S3 accuracy: {results[3]['acc']}")
print(f"S4 accuracy: {results[4]['acc']}")
```

### Step 4: Compare WC Matrices
```python
# Load both models
net_orig = gsc.load_model('sap_g1_model_orig.pkl')
net_round14 = gsc.load_model('...round14 model...')

# Compare learned weights
diff = net_orig.WC - net_round14.WC
print(f"WC max diff: {np.abs(diff).max()}")
print(f"WC mean diff: {np.abs(diff).mean()}")

# Check bias differences
diff_b = net_orig.bC - net_round14.bC
print(f"bC max diff: {np.abs(diff_b).max()}")
```

---

## Next Steps

**Immediate action needed:**
1. Find or recreate the training script for `sap_g1_model_orig.pkl`
2. Identify ANY code or parameter differences between that training and round14
3. Test if current `gsc.py` can reproduce the perfect results

**Question for user:**
- Do you have the training script that created `sap_g1_model_orig.pkl`?
- Or do you know which commit/branch of `gsc.py` was used?
- Was any special configuration or procedure used?
