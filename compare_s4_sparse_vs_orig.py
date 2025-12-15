"""
Compare S4 parsing between sparse and original implementations
"""

import numpy as np
import only_gscnet_speedup_sap as gsc_sparse
import gsc as gsc_orig

print("="*70)
print("Comparing S4 Parsing: Sparse vs Original")
print("="*70)

# Load models
net_sparse = gsc_sparse.load_model('sap_g1_model_fixed_sparse_nocompress.pkl')
net_orig = gsc_orig.load_model('sap_g1_model_orig.pkl')

# Force use_jax=False for sparse
if hasattr(net_sparse, 'use_jax'):
    net_sparse.use_jax = False
    net_sparse.opts['use_jax'] = False

# Test configuration
t = 5
max_sent_len = 5
dq = np.ones(max_sent_len) * (float(t) / max_sent_len)

print("\nTesting with 100 trials, seed=41")
print("-"*70)

# Test sparse
np.random.seed(41)
results_sparse = gsc_sparse.test_parse_inc(
    net_sparse,
    dq=dq,
    num_sent=5,
    num_trials=100,
    estr=2,
    estr_null=2,
    disp=False
)

# Test original
np.random.seed(41)
results_orig = gsc_orig.test_parse_inc(
    net_orig,
    dq=dq,
    num_sent=5,
    num_trials=100,
    estr=2,
    estr_null=2,
    disp=False
)

# Compare all sentences
for si in range(5):
    sent = net_sparse.corpus['sentence'][si]
    sent_str = ' '.join([w.split('/')[0] for w in sent])

    acc_sparse = results_sparse[si]['acc'] if si in results_sparse else 0.0
    acc_orig = results_orig[si]['acc'] if si in results_orig else 0.0

    diff = acc_sparse - acc_orig
    match = "✓" if abs(diff) < 0.15 else "✗"

    print(f"S{si} {sent_str:20s}: Sparse={acc_sparse*100:5.1f}%  Orig={acc_orig*100:5.1f}%  Diff={diff*100:+5.1f}%  {match}")

print("="*70)
