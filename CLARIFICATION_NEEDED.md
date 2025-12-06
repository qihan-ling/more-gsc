# Clarification Needed: gsc.py vs only_gscnet_speedup_sap.py Performance

## Your Statement

> "the dense version is of the same performance with the original non-jax gsc.py training results. the gsc.py file does not use jax but still leads to better performance than round14."

## Need Clarification

To help diagnose the issue, please clarify:

### Question 1: Which comparison are you making?

**Option A**: Training comparison
- Trained model using `gsc.py` (with `use_jax: False`) → got GOOD parsing results
- Trained model using `only_gscnet_speedup_sap.py` (dense, `use_jax: False`) → got BAD parsing results (round14)
- **Same training code, different implementations → different outcomes**

**Option B**: Testing comparison
- Trained a model once (with either implementation)
- Tested parsing using `gsc.py` → got GOOD results
- Tested same model using `only_gscnet_speedup_sap.py` → got BAD results
- **Same model, different test code → different outcomes**

**Option C**: Historical comparison
- Old results from `gsc.py` (before SAP fork) showed GOOD performance
- Current results from `only_gscnet_speedup_sap.py` show BAD performance (round14)
- **Question**: What changed between then and now?

### Question 2: File versions

Which files produced which results?

| Implementation | JAX Status | Training Result | Parsing Result (t=12) |
|---------------|------------|-----------------|---------------------|
| `gsc.py` with `use_jax: False` | ❌ Disabled | ? | S3: ? S4: ? |
| `gsc.py` with `use_jax: True` (default) | ✅ Enabled | ? | S3: ? S4: ? |
| `only_gscnet_speedup.py` (cho_grammar1) | ✅ Enabled (default) | Good (reference figure) | S3: 0.6 S4: 0.1-0.3 |
| `only_gscnet_speedup_sap.py` dense | ❌ Disabled | Bad | S3: 0.5 S4: 0.0 |
| `only_gscnet_speedup_sap.py` sparse | ❌ Disabled | Bad | S3: 0.0-0.2 S4: 0.0 |

## Hypothesis to Test

If `gsc.py` (no JAX) produces better results than `only_gscnet_speedup_sap.py` (no JAX), there must be a **code difference** unrelated to JAX.

### Possible Differences

1. **Different default parameters**
   - `gsc.py` might have different `m`, `q_rate`, `dt`, etc.

2. **Different WC initialization**
   - Sparse matrix setup bug in SAP version

3. **Different run_sent/runC implementation**
   - Bug introduced during SAP fork

4. **Different corpus generation**
   - Sentence ordering or probability calculation

5. **Different equilibrium point calculation**
   - `get_ep()` implementation differs

## Recommended Test

Create a minimal test to isolate the difference:

```python
# test_gsc_vs_sap.py
import gsc
import only_gscnet_speedup_sap as gsc_sap
import numpy as np

PCFG_G1 = '''
0.35 S -> N Vi
0.60 S -> N VP
0.05 S -> NP Vi
1.0 NP -> N RC
1.0 RC -> Vpp PP
1.0 VPpp -> Vpp PP
1.0 PP -> P N
0.5 VP -> Vi PP
0.3 VP -> BE Vpp
0.2 VP -> BE VPpp
'''

# IDENTICAL setup for both
net_opts = {
    'use_jax': False,  # FORCE CPU for both
    'T_init': 0.01,
    'q_max': 15.0,
    'q_init': 0.0,
    'dt_init': 0.005,
    'm': 30,
    'use_runC': True,
    'ep_method': 'integration',
}

# Test 1: Train with gsc.py
print("="*70)
print("Training with gsc.py (no JAX)")
print("="*70)
hg1 = gsc.HarmonicGrammar(pcfg=PCFG_G1, root='S', max_sent_len=5)
sim1 = hg1.get_simlist(dp=0.0)
net1 = gsc.GscNet(hg=hg1, encodings={'similarity': sim1},
                  opts=net_opts, seed=1024)
net1.generate_corpus(use_freq=True)
net1.initialize(train_opts={'lrate': 0.1, 'num_trials': 4})

# Train briefly
for i in range(10):
    net1.train2(train_opts={'num_epochs': 10})

# Test parsing at t=12
dq = np.ones(5) * (12.0 / 5.0)
np.random.seed(2024)
results1 = gsc.test_parse_inc(net1, dq=dq, num_trials=10, disp=False)

print("\nResults from gsc.py:")
for si in results1:
    print(f"  S{si}: {results1[si]['acc']:.3f}")

# Test 2: Train with only_gscnet_speedup_sap.py
print("\n" + "="*70)
print("Training with only_gscnet_speedup_sap.py (no JAX)")
print("="*70)
hg2 = gsc_sap.HarmonicGrammar(pcfg=PCFG_G1, root='S', max_sent_len=5)
sim2 = hg2.get_simlist(dp=0.0)
net2 = gsc_sap.GscNet(hg=hg2, encodings={'similarity': sim2},
                      opts=net_opts, seed=1024)
net2.generate_corpus(use_freq=True)
net2.initialize(train_opts={'lrate': 0.1, 'num_trials': 4})

# Train identically
for i in range(10):
    net2.train2(train_opts={'num_epochs': 10})

# Test parsing at t=12
np.random.seed(2024)
results2 = gsc_sap.test_parse_inc(net2, dq=dq, num_trials=10, disp=False)

print("\nResults from only_gscnet_speedup_sap.py:")
for si in results2:
    print(f"  S{si}: {results2[si]['acc']:.3f}")

# Compare
print("\n" + "="*70)
print("COMPARISON:")
print("="*70)
for si in results1:
    print(f"S{si}: gsc.py={results1[si]['acc']:.3f}, SAP={results2[si]['acc']:.3f}, diff={results1[si]['acc']-results2[si]['acc']:.3f}")
```

## What This Test Will Tell Us

- If both produce BAD results → problem is in the training/model, not the parsing code
- If gsc.py produces GOOD results, SAP produces BAD → there's a parsing code bug in SAP version
- If both produce GOOD results → round14 used different parameters or had a training bug

Please run this test or clarify which comparison you were making!
