"""
Compare sparse vs original during parsing at high commitment to find divergence point.

Since training matches perfectly, the bug must be in the parsing/inference path.
Specifically affects longer sentences (S3, S4) at commitment >= 5.
"""

import only_gscnet_speedup_sap as gsc_sparse
import gsc as gsc_orig
import numpy as np

print("="*70)
print("Parsing Divergence Analysis: High Commitment (t=5)")
print("="*70)

# Load both models
print("\nLoading models...")
net_sparse = gsc_sparse.load_model('sap_g1_model_fixed_sparse_nocompress.pkl')
net_orig = gsc_orig.load_model('sap_g1_model_orig.pkl')

# CRITICAL FIX: Force use_jax=False to match training configuration
# Training used use_jax=False, so parsing must too for random state sync
if hasattr(net_sparse, 'use_jax'):
    net_sparse.use_jax = False
    print("✓ Forced sparse model to use_jax=False for random state synchronization")

# Test on S3: "N BE Vpp P N" (fails at t=5: 30% vs 100%)
sent_idx = 3
sent = net_sparse.corpus['sentence'][sent_idx]
sent_words = [bname.split('/')[0] for bname in sent]
print(f"\nTest sentence S{sent_idx}: {' '.join(sent_words)}")
print(f"Sparse achieves 30%, Original achieves 100% at t=5")

# Set commitment level t=5
commitment = 5
max_sent_len = net_sparse.hg.opts['max_sent_len']
dq = np.ones(max_sent_len) * (float(commitment) / max_sent_len)

print(f"\nCommitment t={commitment}, dq={dq[0]:.3f} per word")

# Set qpolicy for both networks
net_sparse.qpolicy = dq.cumsum()
net_sparse.qpolicy = np.insert(net_sparse.qpolicy, 0, 0.)
net_orig.qpolicy = dq.cumsum()
net_orig.qpolicy = np.insert(net_orig.qpolicy, 0, 0.)

print(f"qpolicy: {net_sparse.qpolicy}")

# CRITICAL: Use SAME random seed for both to ensure identical noise
print("\n" + "="*70)
print("Running ONE trial with synchronized random state")
print("="*70)

# Set same seed
seed = 12345
np.random.seed(seed)

# SPARSE: Reset with noise
print("\n--- SPARSE: Reset ---")
initial_state_sparse = net_sparse.ep.copy()
net_sparse.reset(mu=net_sparse.ep, sd=0.02)
noise_sparse = net_sparse.actC - initial_state_sparse
print(f"Reset noise sum: {noise_sparse.sum():.10f}")
print(f"Reset noise std: {noise_sparse.std():.10f}")
print(f"Initial actC sum: {net_sparse.actC.sum():.10f}")

# Reset seed and do ORIGINAL: Reset with noise
np.random.seed(seed)
print("\n--- ORIGINAL: Reset ---")
initial_state_orig = net_orig.ep.copy()
net_orig.reset(mu=net_orig.ep, sd=0.02)
noise_orig = net_orig.actC - initial_state_orig
print(f"Reset noise sum: {noise_orig.sum():.10f}")
print(f"Reset noise std: {noise_orig.std():.10f}")
print(f"Initial actC sum: {net_orig.actC.sum():.10f}")

# Compare initial states
print("\n--- Initial State Comparison ---")
diff_noise = np.abs(noise_sparse - noise_orig).sum()
diff_actC = np.abs(net_sparse.actC - net_orig.actC).sum()
print(f"Noise difference: {diff_noise:.10e}")
print(f"actC difference: {diff_actC:.10e}")

if diff_actC > 1e-10:
    print("WARNING: Initial states differ! Random noise not synchronized!")
else:
    print("✓ Initial states match")

# Now process words one by one and compare
for wi, word in enumerate(sent_words):
    print(f"\n{'='*70}")
    print(f"Word {wi+1}: {word}")
    print(f"{'='*70}")

    # SPARSE
    print("\n--- SPARSE ---")
    actC_before_sparse = net_sparse.actC.copy()
    net_sparse.run_word(word, wi + 1, log_trace=False)
    actC_after_sparse = net_sparse.actC.copy()
    delta_sparse = actC_after_sparse - actC_before_sparse
    print(f"  actC before: sum={actC_before_sparse.sum():.6f}, max={actC_before_sparse.max():.6f}")
    print(f"  actC after:  sum={actC_after_sparse.sum():.6f}, max={actC_after_sparse.max():.6f}")
    print(f"  delta:       sum={delta_sparse.sum():.6f}, max={np.abs(delta_sparse).max():.6f}")

    # ORIGINAL
    print("\n--- ORIGINAL ---")
    actC_before_orig = net_orig.actC.copy()
    net_orig.run_word(word, wi + 1, log_trace=False)
    actC_after_orig = net_orig.actC.copy()
    delta_orig = actC_after_orig - actC_before_orig
    print(f"  actC before: sum={actC_before_orig.sum():.6f}, max={actC_before_orig.max():.6f}")
    print(f"  actC after:  sum={actC_after_orig.sum():.6f}, max={actC_after_orig.max():.6f}")
    print(f"  delta:       sum={delta_orig.sum():.6f}, max={np.abs(delta_orig).max():.6f}")

    # COMPARE
    print("\n--- COMPARISON ---")
    diff_before = np.abs(actC_before_sparse - actC_before_orig).sum()
    diff_after = np.abs(actC_after_sparse - actC_after_orig).sum()
    diff_delta = np.abs(delta_sparse - delta_orig).sum()

    print(f"  Difference before word: {diff_before:.6e}")
    print(f"  Difference after word:  {diff_after:.6e}")
    print(f"  Difference in delta:    {diff_delta:.6e}")

    if diff_after > 1e-6:
        print(f"  ⚠ DIVERGENCE DETECTED at word {wi+1}!")
        # Find which bindings diverged most
        abs_diff = np.abs(actC_after_sparse - actC_after_orig)
        top_diff_idx = np.argsort(abs_diff)[-5:][::-1]
        print(f"  Top 5 diverging bindings:")
        for idx in top_diff_idx:
            print(f"    {net_sparse.binding_names[idx]:30s}: sparse={actC_after_sparse[idx]:.6f}, orig={actC_after_orig[idx]:.6f}, diff={abs_diff[idx]:.6f}")
        break
    else:
        print(f"  ✓ States still match after word {wi+1}")

print("\n" + "="*70)
print("Analysis complete!")
print("="*70)
