"""
Quick check: Do the models have the same num_bindings?
"""

import only_gscnet_speedup_sap as gsc_sparse
import gsc as gsc_orig

net_sparse = gsc_sparse.load_model('sap_g1_model_fixed_sparse_nocompress.pkl')
net_orig = gsc_orig.load_model('sap_g1_model_orig.pkl')

print("="*70)
print("Checking num_bindings")
print("="*70)
print(f"Sparse:   {net_sparse.num_bindings}")
print(f"Original: {net_orig.num_bindings}")
print(f"Match: {net_sparse.num_bindings == net_orig.num_bindings}")
print("="*70)

if net_sparse.num_bindings != net_orig.num_bindings:
    print("\n❌ FOUND THE PROBLEM!")
    print("Different num_bindings means different random consumption in:")
    print("  - reset() -> set_state() -> np.random.normal(size=num_bindings)")
    print("  - add_noiseC() -> np.random.randn(num_bindings)")
    print("\nThis causes random state desynchronization during parsing,")
    print("leading to different noise sequences and different results.")
