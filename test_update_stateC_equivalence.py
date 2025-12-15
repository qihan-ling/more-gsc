"""
Test if update_stateC produces identical results for sparse vs original
"""

import numpy as np
import only_gscnet_speedup_sap as gsc_sparse
import gsc as gsc_orig

print("="*70)
print("Testing update_stateC equivalence")
print("="*70)

# Load models
net_sparse = gsc_sparse.load_model('sap_g1_model_fixed_sparse_nocompress.pkl')
net_orig = gsc_orig.load_model('sap_g1_model_orig.pkl')

# Force use_jax=False for sparse
if hasattr(net_sparse, 'use_jax'):
    net_sparse.use_jax = False
    net_sparse.opts['use_jax'] = False

# Set both to same state
seed = 12345
np.random.seed(seed)
net_sparse.reset(mu=net_sparse.ep, sd=0.02)

np.random.seed(seed)
net_orig.reset(mu=net_orig.ep, sd=0.02)

# Set same input (first word of S4)
sent = ['N', 'Vpp', 'P', 'N', 'Vi']
word = sent[0]
bname = word + net_sparse.hg.opts['bsep'] + '(1,1)'
net_sparse.set_input(bname)
net_orig.set_input(bname)

print("\n1. Initial state comparison:")
actC_diff_before = np.sum(np.abs(net_sparse.actC - net_orig.actC))
print(f"   actC difference: {actC_diff_before:.2e}")

# Run 100 dynamics steps and check if they stay synchronized
print("\n2. Running 100 dynamics steps...")

for step in range(100):
    # Save states
    actC_sparse_before = net_sparse.actC.copy()
    actC_orig_before = net_orig.actC.copy()

    # Sync random states before each update
    rng_state = np.random.get_state()

    np.random.set_state(rng_state)
    net_sparse.update_stateC()

    np.random.set_state(rng_state)
    net_orig.update_stateC()

    # Check difference
    actC_diff = np.sum(np.abs(net_sparse.actC - net_orig.actC))

    if step < 5 or step % 20 == 0:
        print(f"   Step {step+1:3d}: actC diff = {actC_diff:.2e}")

    if actC_diff > 1e-3:
        print(f"\n   ❌ DIVERGENCE at step {step+1}!")
        print(f"      Difference: {actC_diff:.2e}")

        # Check gradients
        hgrad_sparse = net_sparse.HGradC()
        hgrad_orig = net_orig.HGradC()
        hgrad_diff = np.sum(np.abs(hgrad_sparse - hgrad_orig))
        print(f"      HGradC diff: {hgrad_diff:.2e}")

        # Check gradC computation
        temp_sparse = net_sparse.C_T.dot(hgrad_sparse)
        gradC_sparse = net_sparse.C.dot(temp_sparse)
        gradC_sparse = net_sparse.scale_constants * gradC_sparse

        gradC_orig = net_orig.scale_constants * net_orig.S.dot(hgrad_orig)

        gradC_diff = np.sum(np.abs(gradC_sparse - gradC_orig))
        print(f"      gradC diff: {gradC_diff:.2e}")

        break

print("\n" + "="*70)
print("CONCLUSION")
print("="*70)

if actC_diff < 1e-10:
    print("\n✓ update_stateC is NUMERICALLY IDENTICAL")
    print("  The parsing difference must be due to something else")
elif actC_diff < 1e-6:
    print("\n⚠ update_stateC has SMALL numerical differences")
    print("  These could accumulate over many iterations in parsing")
    print("  This might explain the 82% vs 100% difference")
else:
    print("\n❌ update_stateC has SIGNIFICANT numerical differences")
    print("  This is likely the cause of parsing accuracy differences")

print("="*70)
