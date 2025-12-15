"""
Comprehensive diagnostic to understand random state synchronization issue
"""

import numpy as np
import only_gscnet_speedup_sap as gsc_sparse
import gsc as gsc_orig

print("="*70)
print("Random State Synchronization Diagnostic")
print("="*70)

# Load models
print("\n1. Loading models...")
net_sparse = gsc_sparse.load_model('sap_g1_model_fixed_sparse_nocompress.pkl')
net_orig = gsc_orig.load_model('sap_g1_model_orig.pkl')

# Force use_jax=False
if hasattr(net_sparse, 'use_jax'):
    net_sparse.use_jax = False
    print("   ✓ Forced sparse model to use_jax=False")

# Check basic properties
print("\n2. Checking model properties...")
print(f"   Sparse num_bindings: {net_sparse.num_bindings}")
print(f"   Original num_bindings: {net_orig.num_bindings}")
print(f"   Sparse use_jax: {net_sparse.use_jax if hasattr(net_sparse, 'use_jax') else 'N/A'}")
print(f"   Original use_jax: {net_orig.use_jax if hasattr(net_orig, 'use_jax') else 'N/A'}")

if net_sparse.num_bindings != net_orig.num_bindings:
    print("\n   ⚠ WARNING: num_bindings differ!")
    print("   This will cause random state desynchronization!")

# Test reset() random consumption
print("\n3. Testing reset() random consumption...")

# Track how many random numbers are consumed by reset()
seed = 12345

# Test sparse reset
np.random.seed(seed)
rng_before_sparse = np.random.get_state()[1][0]  # Get first element of state
net_sparse.reset(mu=net_sparse.ep, sd=0.02)
rng_after_sparse = np.random.get_state()[1][0]

np.random.seed(seed)
rng_before_orig = np.random.get_state()[1][0]
net_orig.reset(mu=net_orig.ep, sd=0.02)
rng_after_orig = np.random.get_state()[1][0]

print(f"   Sparse reset: state changed from {rng_before_sparse} to {rng_after_sparse}")
print(f"   Original reset: state changed from {rng_before_orig} to {rng_after_orig}")

# Now check if random states are synchronized after both resets
print("\n4. Testing synchronized reset...")
seed = 12345
np.random.seed(seed)
net_sparse.reset(mu=net_sparse.ep, sd=0.02)
state_after_sparse_reset = np.random.get_state()

np.random.seed(seed)
net_orig.reset(mu=net_orig.ep, sd=0.02)
state_after_orig_reset = np.random.get_state()

# Compare states
states_match = np.array_equal(state_after_sparse_reset[1], state_after_orig_reset[1])
print(f"   Random states match after reset: {states_match}")

if not states_match:
    print("   ⚠ Random states DO NOT match after reset!")
    print("   This means reset() is consuming different amounts of random numbers")

    # Draw a test random number from each state to show they're different
    np.random.set_state(state_after_sparse_reset)
    test_rand_sparse = np.random.randn()

    np.random.set_state(state_after_orig_reset)
    test_rand_orig = np.random.randn()

    print(f"   Next random number (sparse state): {test_rand_sparse:.6f}")
    print(f"   Next random number (orig state): {test_rand_orig:.6f}")
    print(f"   Difference: {abs(test_rand_sparse - test_rand_orig):.6f}")

# Test noise generation
print("\n5. Testing noise generation in first iteration...")

# Reset both with same seed
seed = 12345
np.random.seed(seed)
net_sparse.reset(mu=net_sparse.ep, sd=0.02)

np.random.seed(seed)
net_orig.reset(mu=net_orig.ep, sd=0.02)

# Set same input
sent = ['N', 'BE', 'Vpp', 'P', 'N']  # S3
word = sent[0]
wpos = 1
bname = word + net_sparse.hg.opts['bsep'] + '(1,%d)' % wpos
net_sparse.set_input(bname)
net_orig.set_input(bname)

# Save states before first iteration
actC_before_sparse = net_sparse.actC.copy()
actC_before_orig = net_orig.actC.copy()

# Manually compute one iteration for sparse (matching the diagnostic script)
hgrad_sparse = net_sparse.HGradC()
temp_sparse = net_sparse.C_T.dot(hgrad_sparse)
gradC_sparse = net_sparse.C.dot(temp_sparse)
gradC_sparse = net_sparse.scale_constants * gradC_sparse
actC_after_grad_sparse = actC_before_sparse + net_sparse.dt * gradC_sparse

# Generate noise the way the diagnostic script does
noise_sparse = np.sqrt(2 * net_sparse.T * net_sparse.dt) * np.random.randn(net_sparse.num_bindings)
noiseC_sparse = np.sqrt(net_sparse.scale_constants) * net_sparse.N2C(noise_sparse)

# Same for original
hgrad_orig = net_orig.HGradC()
gradC_orig = net_orig.S.dot(hgrad_orig)
gradC_orig = net_orig.scale_constants * gradC_orig
actC_after_grad_orig = actC_before_orig + net_orig.dt * gradC_orig

noise_orig = np.sqrt(2 * net_orig.T * net_orig.dt) * np.random.randn(net_orig.num_bindings)
noiseC_orig = np.sqrt(net_orig.scale_constants) * net_orig.N2C(noise_orig)

print(f"   Noise diff: {np.sum(np.abs(noiseC_sparse - noiseC_orig)):.6f}")

if np.sum(np.abs(noiseC_sparse - noiseC_orig)) > 1e-10:
    print("   ⚠ Noise differs!")
    print("\n6. Investigating cause...")

    # Check if random states were synchronized before noise generation
    print("   Possible causes:")
    print("   a) Random states not synchronized after reset")
    print("   b) Different num_bindings causing different consumption")
    print("   c) set_input() consuming random numbers")
    print("   d) Something else in the setup consuming random numbers")

print("\n" + "="*70)
print("SUMMARY")
print("="*70)

if net_sparse.num_bindings != net_orig.num_bindings:
    print("❌ PROBLEM: num_bindings differ between implementations")
    print(f"   Sparse: {net_sparse.num_bindings}")
    print(f"   Original: {net_orig.num_bindings}")
    print("   → This causes different random consumption in set_state()")
elif not states_match:
    print("❌ PROBLEM: Random states not synchronized after reset()")
    print("   → reset() is consuming different amounts of random numbers")
    print("   → Need to investigate reset() and set_state() implementations")
else:
    print("✓ Random states are synchronized after reset")
    print("✓ num_bindings match")
    print("   → Noise should match in first iteration")

print("="*70)
