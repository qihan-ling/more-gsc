"""Diagnose why sparse implementation fails at commitment >= 5"""

import only_gscnet_speedup_sap as gsc_sparse
import gsc as gsc_orig
import numpy as np

print("="*70)
print("Comparing Sparse vs Original at Different Commitment Levels")
print("="*70)

# Load both models
print("\nLoading models...")
net_sparse = gsc_sparse.load_model('sap_g1_model_fixed_sparse_nocompress.pkl')
net_orig = gsc_orig.load_model('sap_g1_model_orig.pkl')

# Test sentence S1: "N Vi P N"
sent_idx = 1
sent = net_sparse.corpus['sentence'][sent_idx]
sent_words = [bname.split('/')[0] for bname in sent]
print(f"\nTest sentence: {' '.join(sent_words)}")

# Test at commitment levels 3, 4, and 5
for commitment in [3, 4, 5]:
    print(f"\n{'='*70}")
    print(f"Testing at commitment level t={commitment}")
    print(f"{'='*70}")

    max_sent_len = net_sparse.hg.opts['max_sent_len']
    dq = np.ones(max_sent_len) * (float(commitment) / max_sent_len)

    # SPARSE: Run one trial
    print("\n--- SPARSE ---")
    net_sparse.qpolicy = dq.cumsum()
    net_sparse.qpolicy = np.insert(net_sparse.qpolicy, 0, 0.)
    net_sparse.reset(mu=net_sparse.ep, sd=0.02)

    print(f"Initial actC sum: {net_sparse.actC.sum():.6f}")
    print(f"Initial q values: {net_sparse.q[:3]}")

    # Run first word with diagnostics
    net_sparse.run_word(sent_words[0], 1, debug_dynamics=True)
    print(f"After word 1 - actC sum: {net_sparse.actC.sum():.6f}")
    print(f"After word 1 - actC max: {net_sparse.actC.max():.6f}")
    print(f"After word 1 - actC min: {net_sparse.actC.min():.6f}")

    # Check for divergence
    if np.any(np.isnan(net_sparse.actC)) or np.any(np.isinf(net_sparse.actC)):
        print("ERROR: Activations contain NaN or Inf!")
    if net_sparse.actC.max() > 10.0:
        print(f"WARNING: Very high activations detected: {net_sparse.actC.max():.2f}")

    # ORIGINAL: Run one trial
    print("\n--- ORIGINAL ---")
    net_orig.qpolicy = dq.cumsum()
    net_orig.qpolicy = np.insert(net_orig.qpolicy, 0, 0.)
    net_orig.reset(mu=net_orig.ep, sd=0.02)

    print(f"Initial actC sum: {net_orig.actC.sum():.6f}")
    print(f"Initial q values: {net_orig.q[:3]}")

    # Run first word
    net_orig.run_word(sent_words[0], 1)
    print(f"After word 1 - actC sum: {net_orig.actC.sum():.6f}")
    print(f"After word 1 - actC max: {net_orig.actC.max():.6f}")
    print(f"After word 1 - actC min: {net_orig.actC.min():.6f}")

    # Check for divergence
    if np.any(np.isnan(net_orig.actC)) or np.any(np.isinf(net_orig.actC)):
        print("ERROR: Activations contain NaN or Inf!")
    if net_orig.actC.max() > 10.0:
        print(f"WARNING: Very high activations detected: {net_orig.actC.max():.2f}")

    # Compare final states
    print(f"\n--- COMPARISON ---")
    actC_diff = np.abs(net_sparse.actC - net_orig.actC).sum()
    print(f"Total absolute difference in actC: {actC_diff:.6f}")
    if actC_diff > 0.1:
        print(f"WARNING: Large difference detected!")
        print(f"  Sparse actC sum: {net_sparse.actC.sum():.6f}")
        print(f"  Original actC sum: {net_orig.actC.sum():.6f}")

print("\n" + "="*70)
print("Diagnosis complete!")
print("="*70)
