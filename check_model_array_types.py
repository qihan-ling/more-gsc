"""
Check if models were saved with different array types (JAX vs numpy)
and if that's causing the dynamics divergence
"""

import numpy as np

print("="*70)
print("CHECKING MODEL STATE AFTER LOADING")
print("="*70)

import only_gscnet_speedup as gsc1
net1 = gsc1.load_model('g1_ds_speedup_model_copy_trainjax.pkl')

import only_gscnet_speedup_sap as gsc2
net2 = gsc2.load_model('ds_jax_sap_test_on_g1_model.pkl')

print("\n" + "="*70)
print("ARRAY TYPES IN LOADED MODELS")
print("="*70)

print(f"\nSpeedup model:")
print(f"  actC type: {type(net1.actC)}")
print(f"  WC type: {type(net1.WC)}")
print(f"  bC type: {type(net1.bC)}")
print(f"  scale_constants type: {type(net1.scale_constants)}")
print(f"  ep type: {type(net1.ep)}")

print(f"\nSAP model:")
print(f"  actC type: {type(net2.actC)}")
print(f"  WC type: {type(net2.WC)}")
print(f"  bC type: {type(net2.bC)}")
print(f"  scale_constants type: {type(net2.scale_constants)}")
print(f"  ep type: {type(net2.ep)}")

# Check if use_jax flag is set
print(f"\nJAX usage:")
print(f"  Speedup use_jax: {net1.use_jax if hasattr(net1, 'use_jax') else 'Not set'}")
print(f"  SAP use_jax: {net2.use_jax if hasattr(net2, 'use_jax') else 'Not set'}")

print("\n" + "="*70)
print("CRITICAL PARAMETERS")
print("="*70)

print(f"\nscale_constants (first 10):")
print(f"  Speedup: {net1.scale_constants[:10]}")
print(f"  SAP:     {net2.scale_constants[:10]}")
if np.allclose(net1.scale_constants, net2.scale_constants):
    print(f"  ✓ MATCH")
else:
    diff = np.abs(net1.scale_constants - net2.scale_constants)
    print(f"  ❌ DIFFER! Max diff: {np.max(diff):.10f}")

print(f"\nscaling_factor:")
print(f"  Speedup: {net1.opts.get('scaling_factor', 'Not set')}")
print(f"  SAP:     {net2.opts.get('scaling_factor', 'Not set')}")

print(f"\nbowl_strength:")
print(f"  Speedup: {net1.opts['bowl_strength']}")
print(f"  SAP:     {net2.opts['bowl_strength']}")

print(f"\nbowl_center (sum):")
print(f"  Speedup: {np.sum(net1.opts['bowl_center'])}")
print(f"  SAP:     {np.sum(net2.opts['bowl_center'])}")

print(f"\nT (temperature):")
print(f"  Speedup: {net1.T}")
print(f"  SAP:     {net2.T}")

print(f"\ndt (timestep):")
print(f"  Speedup: {net1.dt}")
print(f"  SAP:     {net2.dt}")

print("\n" + "="*70)
print("TEST: CONVERT JAX ARRAYS TO NUMPY")
print("="*70)

# Try to ensure both models use numpy arrays
try:
    import jax.numpy as jnp

    # Convert speedup model to numpy if it's JAX
    if isinstance(net1.actC, jnp.ndarray):
        print("\n✓ Speedup model has JAX arrays - converting to numpy")
        net1.actC = np.array(net1.actC)
        net1.WC = np.array(net1.WC) if isinstance(net1.WC, jnp.ndarray) else net1.WC
        net1.bC = np.array(net1.bC)
        net1.scale_constants = np.array(net1.scale_constants)
        net1.ep = np.array(net1.ep)
        net1.use_jax = False
    else:
        print("\n  Speedup model already uses numpy arrays")

    # Convert SAP model to numpy if it's JAX
    if isinstance(net2.actC, jnp.ndarray):
        print("✓ SAP model has JAX arrays - converting to numpy")
        net2.actC = np.array(net2.actC)
        net2.WC = np.array(net2.WC) if isinstance(net2.WC, jnp.ndarray) else net2.WC
        net2.bC = np.array(net2.bC)
        net2.scale_constants = np.array(net2.scale_constants)
        net2.ep = np.array(net2.ep)
        net2.use_jax = False
    else:
        print("  SAP model already uses numpy arrays")

except ImportError:
    print("\nJAX not available, models must be numpy")

print("\n" + "="*70)
print("RE-TEST DYNAMICS WITH NUMPY-ONLY")
print("="*70)

# Set up identical sentence
test_words = ['N', 'Vi', 'P', 'N']

for net in [net1, net2]:
    for wi, word in enumerate(test_words):
        net.run_word(word, wi + 1, log_trace=False)

# Reset with same seed
np.random.seed(1024)
net1.reset(mu=net1.ep, sd=0.01)

np.random.seed(1024)
net2.reset(mu=net2.ep, sd=0.01)

print(f"\nAfter reset (first 10 actC values):")
print(f"  Speedup: {net1.actC[:10]}")
print(f"  SAP:     {net2.actC[:10]}")

if np.allclose(net1.actC, net2.actC):
    print(f"  ✓ Initial states MATCH")
else:
    diff = np.abs(net1.actC - net2.actC)
    print(f"  ❌ Initial states DIFFER! Max diff: {np.max(diff):.10f}")

# Compute first HGradC
hgrad1 = net1.HGradC(net1.actC, net1.q)
hgrad2 = net2.HGradC(net2.actC, net2.q)

print(f"\nHGradC (first 10 values):")
print(f"  Speedup: {hgrad1[:10]}")
print(f"  SAP:     {hgrad2[:10]}")

if np.allclose(hgrad1, hgrad2, atol=1e-6):
    print(f"  ✓ HGradC MATCH")
else:
    diff = np.abs(hgrad1 - hgrad2)
    print(f"  ❌ HGradC DIFFER! Max diff: {np.max(diff):.10f}")
    max_idx = np.argmax(diff)
    print(f"\n  Largest difference at index {max_idx}:")
    print(f"    Binding: {net1.binding_names[max_idx]}")
    print(f"    Speedup: {hgrad1[max_idx]:.10f}")
    print(f"    SAP:     {hgrad2[max_idx]:.10f}")

print("\n" + "="*70)
print("DIAGNOSIS")
print("="*70)

print("""
If HGradC differs even after converting to numpy:
  - The issue is in the HGradC implementation itself
  - Or in the parameters used (scale_constants, bowl_strength, etc.)

If HGradC matches after converting to numpy:
  - The issue was JAX vs numpy numerical differences
  - Models should be saved/loaded consistently
""")

print("="*70)
