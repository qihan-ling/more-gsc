"""
Check if equilibrium points (ep) are the same between models
"""

import numpy as np
import only_gscnet_speedup_sap as gsc_sparse
import gsc as gsc_orig

print("="*70)
print("Checking equilibrium points (ep)")
print("="*70)

# Load models
net_sparse = gsc_sparse.load_model('sap_g1_model_fixed_sparse_nocompress.pkl')
net_orig = gsc_orig.load_model('sap_g1_model_orig.pkl')

if hasattr(net_sparse, 'use_jax'):
    net_sparse.use_jax = False
    net_sparse.opts['use_jax'] = False

print("\n1. Equilibrium point (ep) comparison:")
ep_diff = np.sum(np.abs(net_sparse.ep - net_orig.ep))
ep_max_diff = np.max(np.abs(net_sparse.ep - net_orig.ep))

print(f"   Shape: {net_sparse.ep.shape}")
print(f"   Sum of absolute differences: {ep_diff:.2e}")
print(f"   Max absolute difference: {ep_max_diff:.2e}")

if ep_max_diff < 1e-10:
    print("   ✓ Equilibrium points are IDENTICAL")
elif ep_max_diff < 1e-6:
    print("   ⚠ Equilibrium points are VERY CLOSE but not identical")
else:
    print("   ❌ Equilibrium points DIFFER significantly")

# Check a few specific values
print(f"\n2. Sample ep values:")
print(f"   Sparse ep[0:5]:   {net_sparse.ep[0:5]}")
print(f"   Original ep[0:5]: {net_orig.ep[0:5]}")

# Now check what happens after reset with the same seed
print("\n3. States after reset(mu=ep, sd=0.02):")

seed = 42
np.random.seed(seed)
net_sparse.reset(mu=net_sparse.ep, sd=0.02)
actC_sparse = net_sparse.actC.copy()

np.random.seed(seed)
net_orig.reset(mu=net_orig.ep, sd=0.02)
actC_orig = net_orig.actC.copy()

actC_diff = np.sum(np.abs(actC_sparse - actC_orig))
actC_max_diff = np.max(np.abs(actC_sparse - actC_orig))

print(f"   Sum of absolute differences: {actC_diff:.2e}")
print(f"   Max absolute difference: {actC_max_diff:.2e}")

if actC_max_diff < 1e-10:
    print("   ✓ States after reset are IDENTICAL")
elif actC_max_diff < 1e-6:
    print("   ⚠ States after reset are VERY CLOSE but not identical")
else:
    print("   ❌ States after reset DIFFER significantly")

print("\n" + "="*70)
print("CONCLUSION")
print("="*70)

if ep_max_diff > 1e-6:
    print("\n❌ FOUND THE PROBLEM!")
    print("   Equilibrium points (ep) differ between models!")
    print("\n   Each trial starts with reset(mu=ep, sd=0.02), so:")
    print("   - Different ep → different starting states")
    print("   - Different starting states → different parsing trajectories")
    print("   - Different trajectories → different parsing results")
    print("\n   Why might ep differ?")
    print("   - Different bowl_center computation")
    print("   - Different equilibrium calculation method")
    print("   - Numerical precision in get_ep()")
elif actC_max_diff > 1e-6:
    print("\n❌ FOUND THE PROBLEM!")
    print("   Even with same ep, states after reset() differ!")
    print("   This could be due to:")
    print("   - Different random number generation in set_state()")
    print("   - Different noise application")
else:
    print("\n✓ Both ep and states after reset are identical")
    print("  The issue must be in the dynamics during parsing...")

print("="*70)
