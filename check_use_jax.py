"""
Quick check: What is use_jax value in the loaded models?
"""

import only_gscnet_speedup_sap as gsc_sparse
import gsc as gsc_orig

print("="*70)
print("Checking use_jax setting in loaded models")
print("="*70)

# Load models
net_sparse = gsc_sparse.load_model('sap_g1_model_fixed_sparse_nocompress.pkl')

print(f"\nSparse model:")
print(f"  use_jax: {net_sparse.use_jax if hasattr(net_sparse, 'use_jax') else 'NOT SET'}")

print(f"\nOriginal model:")
print(f"  use_jax: {net_orig.load_model('sap_g1_model_orig.pkl').use_jax if hasattr(net_orig.load_model('sap_g1_model_orig.pkl'), 'use_jax') else 'NOT SET'}")

# Also check if JAX is available in the sparse implementation
if hasattr(gsc_sparse, 'JAX_AVAILABLE'):
    print(f"\nJAX_AVAILABLE in sparse implementation: {gsc_sparse.JAX_AVAILABLE}")

print("\n" + "="*70)
