"""
Check if sparse and original models have identical weights
"""

import numpy as np
import only_gscnet_speedup_sap as gsc_sparse
import gsc as gsc_orig

print("="*70)
print("Comparing Model Weights: Sparse vs Original")
print("="*70)

# Load models
net_sparse = gsc_sparse.load_model('sap_g1_model_fixed_sparse_nocompress.pkl')
net_orig = gsc_orig.load_model('sap_g1_model_orig.pkl')

print("\n1. Checking WC (constraint weights)...")

# Convert sparse to dense for comparison
if hasattr(net_sparse.WC, 'toarray'):
    WC_sparse = net_sparse.WC.toarray()
else:
    WC_sparse = net_sparse.WC

WC_orig = net_orig.WC

if WC_sparse.shape != WC_orig.shape:
    print(f"  ❌ SHAPE MISMATCH!")
    print(f"     Sparse: {WC_sparse.shape}")
    print(f"     Original: {WC_orig.shape}")
else:
    diff = np.abs(WC_sparse - WC_orig)
    max_diff = np.max(diff)
    mean_diff = np.mean(diff)

    print(f"  Shape: {WC_sparse.shape}")
    print(f"  Max difference: {max_diff:.2e}")
    print(f"  Mean difference: {mean_diff:.2e}")

    if max_diff < 1e-10:
        print(f"  ✓ WC weights are IDENTICAL")
    elif max_diff < 1e-6:
        print(f"  ⚠ WC weights are VERY CLOSE but not identical")
    else:
        print(f"  ❌ WC weights DIFFER significantly")

print("\n2. Checking bC (binding biases)...")
diff = np.abs(net_sparse.bC - net_orig.bC)
max_diff = np.max(diff)
mean_diff = np.mean(diff)

print(f"  Shape: {net_sparse.bC.shape}")
print(f"  Max difference: {max_diff:.2e}")
print(f"  Mean difference: {mean_diff:.2e}")

if max_diff < 1e-10:
    print(f"  ✓ bC biases are IDENTICAL")
elif max_diff < 1e-6:
    print(f"  ⚠ bC biases are VERY CLOSE but not identical")
else:
    print(f"  ❌ bC biases DIFFER significantly")

print("\n3. Checking other parameters...")
print(f"  T_init - Sparse: {net_sparse.opts.get('T_init')}, Original: {net_orig.opts.get('T_init')}")
print(f"  dt_init - Sparse: {net_sparse.opts.get('dt_init')}, Original: {net_orig.opts.get('dt_init')}")
print(f"  q_init - Sparse: {net_sparse.opts.get('q_init')}, Original: {net_orig.opts.get('q_init')}")
print(f"  m - Sparse: {net_sparse.opts.get('m')}, Original: {net_orig.opts.get('m')}")

print("\n" + "="*70)
print("CONCLUSION")
print("="*70)

if max_diff > 1e-6:
    print("\n❌ Models have DIFFERENT weights!")
    print("   This explains why parsing results differ.")
    print("   'Training outputs match' might mean training curves look similar,")
    print("   but the models converged to different local optima.")
    print("\n   SOLUTION: Ensure both models are trained with:")
    print("   - Same random seed")
    print("   - Same training data order")
    print("   - Same initialization")
else:
    print("\n✓ Models have (nearly) identical weights")
    print("   The parsing difference must be due to numerical computation")
    print("   differences during parsing dynamics, not model weights.")

print("="*70)
