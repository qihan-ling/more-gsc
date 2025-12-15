"""
Check if sparse and original run different numbers of dynamics iterations
"""

import numpy as np
import only_gscnet_speedup_sap as gsc_sparse
import gsc as gsc_orig

print("="*70)
print("Checking dynamics iteration counts during parsing")
print("="*70)

# Load models
net_sparse = gsc_sparse.load_model('sap_g1_model_fixed_sparse_nocompress.pkl')
net_orig = gsc_orig.load_model('sap_g1_model_orig.pkl')

if hasattr(net_sparse, 'use_jax'):
    net_sparse.use_jax = False
    net_sparse.opts['use_jax'] = False

# Test on S4 (where we see the biggest difference)
sent = ['N', 'Vpp', 'P', 'N', 'Vi']
print(f"\nTesting sentence: {' '.join(sent)}")
print("-"*70)

# Set same seed and run one parsing trial
seed = 42
duration = 10.0  # Typical duration for parsing

# Sparse
np.random.seed(seed)
net_sparse.reset(mu=net_sparse.ep, sd=0.02)

# Set input word by word and count iterations
total_iters_sparse = 0
for i, word in enumerate(sent):
    bname = word + net_sparse.hg.opts['bsep'] + f'(1,{i+1})'
    net_sparse.set_input(bname)

    # Track iterations
    t_before = net_sparse.t
    net_sparse.runC(duration, update_T=True, update_q=True, log_trace=False)
    t_after = net_sparse.t

    iters = int((t_after - t_before) / net_sparse.dt)
    total_iters_sparse += iters
    print(f"  Sparse word {i+1} ({word:4s}): {iters:4d} iterations (t: {t_before:.3f} -> {t_after:.3f})")

# Original
np.random.seed(seed)
net_orig.reset(mu=net_orig.ep, sd=0.02)

total_iters_orig = 0
for i, word in enumerate(sent):
    bname = word + net_orig.hg.opts['bsep'] + f'(1,{i+1})'
    net_orig.set_input(bname)

    t_before = net_orig.t
    net_orig.runC(duration, update_T=True, update_q=True, log_trace=False)
    t_after = net_orig.t

    iters = int((t_after - t_before) / net_orig.dt)
    total_iters_orig += iters
    print(f"  Original word {i+1} ({word:4s}): {iters:4d} iterations (t: {t_before:.3f} -> {t_after:.3f})")

print("-"*70)
print(f"Total iterations - Sparse: {total_iters_sparse}, Original: {total_iters_orig}")
print(f"Difference: {abs(total_iters_sparse - total_iters_orig)} iterations")

if abs(total_iters_sparse - total_iters_orig) > 0:
    print("\n❌ FOUND THE PROBLEM!")
    print("Models run for DIFFERENT numbers of dynamics iterations!")
    print("\nThis causes different random number consumption:")
    print(f"  - Each iteration consumes {net_sparse.num_bindings} random numbers")
    print(f"  - Difference: {abs(total_iters_sparse - total_iters_orig) * net_sparse.num_bindings} random numbers")
    print("\nThis desynchronizes the random streams across trials,")
    print("causing different noise patterns and different parsing results.")
    print("\nPossible causes:")
    print("  1. Different convergence behavior (check_convergence)")
    print("  2. Different divergence detection (check_divergence)")
    print("  3. Numerical differences causing early termination")
else:
    print("\n✓ Same number of iterations")
    print("  The issue must be elsewhere...")

print("="*70)
