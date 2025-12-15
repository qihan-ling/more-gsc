"""Quick diagnostic to identify why noise still diverges"""

import numpy as np
import only_gscnet_speedup_sap as gsc_sparse
import gsc as gsc_orig

print("="*70)
print("QUICK DIAGNOSTIC: Why is noise still diverging?")
print("="*70)

# Load models
net_sparse = gsc_sparse.load_model('sap_g1_model_fixed_sparse_nocompress.pkl')
net_orig = gsc_orig.load_model('sap_g1_model_orig.pkl')

# Apply fix
if hasattr(net_sparse, 'use_jax'):
    net_sparse.use_jax = False

# Test 1: Number of bindings
print("\n[Test 1] Checking num_bindings...")
print(f"  Sparse:   {net_sparse.num_bindings}")
print(f"  Original: {net_orig.num_bindings}")

if net_sparse.num_bindings != net_orig.num_bindings:
    print("  ❌ FOUND THE PROBLEM!")
    print("  → num_bindings differ!")
    print("  → reset() will consume different amounts of random numbers")
    print("  → This causes random state desynchronization")
    print("\n  SOLUTION:")
    print("  The models have different internal structures.")
    print("  You need to retrain them with matching configurations.")
else:
    print("  ✓ num_bindings match")

# Test 2: Random consumption in reset()
print("\n[Test 2] Checking random consumption in reset()...")
seed = 12345

np.random.seed(seed)
net_sparse.reset(mu=net_sparse.ep, sd=0.02)
r1 = np.random.random()  # Draw next random number

np.random.seed(seed)
net_orig.reset(mu=net_orig.ep, sd=0.02)
r2 = np.random.random()  # Draw next random number

print(f"  Next random after sparse reset:   {r1:.15f}")
print(f"  Next random after original reset: {r2:.15f}")
print(f"  Difference: {abs(r1 - r2):.2e}")

if abs(r1 - r2) > 1e-15:
    print("  ❌ FOUND THE PROBLEM!")
    print("  → reset() consumes different amounts of random numbers")
    print("  → Random states are not synchronized after reset()")
    if net_sparse.num_bindings == net_orig.num_bindings:
        print("\n  This is strange because num_bindings match...")
        print("  → Check if there's other random consumption in reset()")
        print("  → Or if use_jax is actually False")
else:
    print("  ✓ Random states synchronized after reset()")

# Test 3: use_jax value
print("\n[Test 3] Checking use_jax settings...")
print(f"  Sparse use_jax: {net_sparse.use_jax if hasattr(net_sparse, 'use_jax') else 'N/A'}")
print(f"  Original use_jax: {net_orig.use_jax if hasattr(net_orig, 'use_jax') else 'N/A'}")

if hasattr(net_sparse, 'use_jax') and net_sparse.use_jax:
    print("  ❌ FOUND THE PROBLEM!")
    print("  → use_jax is True, but should be False!")
    print("  → The fix didn't work")
else:
    print("  ✓ use_jax is False (or not present)")

# Test 4: Actual noise generation
print("\n[Test 4] Testing actual noise generation...")

# Reset both with same seed
seed = 12345
np.random.seed(seed)
net_sparse.reset(mu=net_sparse.ep, sd=0.02)
state1 = np.random.get_state()

np.random.seed(seed)
net_orig.reset(mu=net_orig.ep, sd=0.02)
state2 = np.random.get_state()

# Now generate noise from the same starting point
np.random.set_state(state1)
noise1 = np.random.randn(5)  # Just 5 test numbers

np.random.set_state(state2)
noise2 = np.random.randn(5)

print(f"  5 test randoms from sparse state:   {noise1}")
print(f"  5 test randoms from original state: {noise2}")
print(f"  Difference: {np.max(np.abs(noise1 - noise2)):.2e}")

if np.max(np.abs(noise1 - noise2)) > 1e-15:
    print("  ❌ Random states differ after reset()")
else:
    print("  ✓ Random states are identical")

# Summary
print("\n" + "="*70)
print("SUMMARY")
print("="*70)

problems_found = []

if net_sparse.num_bindings != net_orig.num_bindings:
    problems_found.append("num_bindings differ")

if abs(r1 - r2) > 1e-15:
    problems_found.append("random states not synchronized after reset()")

if hasattr(net_sparse, 'use_jax') and net_sparse.use_jax:
    problems_found.append("use_jax is True (should be False)")

if problems_found:
    print("\n❌ PROBLEMS FOUND:")
    for i, problem in enumerate(problems_found, 1):
        print(f"  {i}. {problem}")

    if "num_bindings differ" in problems_found:
        print("\n→ PRIMARY ISSUE: Different model structures")
        print("  The sparse and original models have different numbers of bindings.")
        print("  This means they were trained with different configurations.")
        print("\n  SOLUTION: You need to either:")
        print("    a) Ensure both use the same grammar/configuration, OR")
        print("    b) Accept that exact random synchronization isn't possible")
        print("       and compare statistical properties instead")
else:
    print("\n✓ NO PROBLEMS FOUND!")
    print("\n  Random states should be synchronized.")
    print("  If noise still diverges in the diagnostic script, the issue is")
    print("  likely in HOW the diagnostic script generates noise, not in the models.")
    print("\n  → Check if the diagnostic script is using the actual network")
    print("    methods (update_stateC, add_noiseC) or manual computation.")

print("="*70)
