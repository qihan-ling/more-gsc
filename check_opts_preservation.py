"""Check if opts['use_jax'] is preserved in pickled model"""

import only_gscnet_speedup_sap as gsc_sparse

print("="*70)
print("Checking if opts['use_jax'] is preserved in pickled model")
print("="*70)

# Load model
net = gsc_sparse.load_model('sap_g1_model_fixed_sparse_nocompress.pkl')

print("\nModel attributes:")
print(f"  net.use_jax: {net.use_jax if hasattr(net, 'use_jax') else 'NOT SET'}")
print(f"  net.opts.get('use_jax'): {net.opts.get('use_jax', 'NOT IN OPTS')}")

print("\nSetting net.use_jax = False...")
net.use_jax = False

print(f"  net.use_jax after setting: {net.use_jax}")
print(f"  net.opts.get('use_jax') after setting: {net.opts.get('use_jax', 'NOT IN OPTS')}")

print("\nDoes opts contain use_jax key?")
print(f"  'use_jax' in net.opts: {'use_jax' in net.opts}")

if 'use_jax' not in net.opts:
    print("\n❌ PROBLEM FOUND!")
    print("  opts['use_jax'] is not in the pickled model!")
    print("  This might cause issues if any code checks opts instead of the attribute")
    print("\n  SOLUTION: Also set opts['use_jax'] = False")
    print("  net.opts['use_jax'] = False")

print("="*70)
