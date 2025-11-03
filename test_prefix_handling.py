#!/usr/bin/env python3
"""
Test script to verify JAX prefix handling matches CPU version.
"""

import numpy as np
import time
import gsc

# Load the trained model
print("Loading model...")
net = gsc.load_obj('g1_model.pkl')
print(f"Model loaded: {net.num_bindings} bindings, {net.num_roles} roles, {net.num_fillers} fillers")

# Test with a simple prefix
prefix = ['N:0']  # Simple one-word prefix
print(f"\nTesting with prefix: {prefix}")

# Run CPU version
print("\n1. Running CPU version...")
net.reset()
net.run_prefix(prefix)
net.run_wrapup()
cpu_grid = net.read_grid_point()  # Returns list of "filler/role" strings
print(f"   CPU grid point: {cpu_grid}")

# Run JAX version (single trial for comparison)
if gsc.JAX_AVAILABLE:
    print("\n2. Running JAX version...")
    import jax

    # Extract parameters
    net_params = gsc._extract_net_params_for_jax(net)

    # Run single trial
    rng_key = jax.random.PRNGKey(42)
    actC_jax, grid_point_jax = gsc._run_single_trial_jax(rng_key, net_params, prefix, False)

    # Convert JAX grid point (filler indices per role) to binding names
    grid_point_np = np.array(grid_point_jax)
    jax_grid = [f"{net.filler_names[int(fi)]}/{net.role_names[ri]}"
                for ri, fi in enumerate(grid_point_np)]
    print(f"   JAX grid point: {jax_grid}")

    # Compare
    print("\n3. Comparison:")
    if cpu_grid == jax_grid:
        print("   ✓ Grid points MATCH!")
    else:
        print("   ✗ Grid points differ (expected due to different RNG)")
        matches = sum(1 for c, j in zip(cpu_grid, jax_grid) if c == j)
        print(f"   Matches: {matches}/{len(cpu_grid)} roles")
else:
    print("\nJAX not available - skipping JAX test")

# Now test with estimate_prob_inc_jax to verify it works with prefix
if gsc.JAX_AVAILABLE:
    print("\n4. Testing estimate_prob_inc_jax with prefix...")

    # Run CPU version with multiple trials
    print("   CPU version (10 trials):")
    t0 = time.time()
    cpu_stat, cpu_actC_list = net.estimate_prob_inc(prefix, num_trials=10)
    cpu_time = time.time() - t0
    print(f"   Time: {cpu_time:.3f}s")
    print(f"   Unique trees: {len(cpu_stat['trees'])}")

    # Show first 3 trees
    for i, (tree_key, prob) in enumerate(list(cpu_stat['trees'].items())[:3]):
        tree_bindings = [net.binding_names[idx] for idx in tree_key]
        print(f"     {i+1}. p={prob:.3f}: {len(tree_bindings)} bindings")

    # Run JAX version
    print("\n   JAX version (10 trials):")
    t0 = time.time()
    jax_stat, jax_actC_list = net.estimate_prob_inc_jax(prefix, num_trials=10, rng_seed=42)
    jax_time = time.time() - t0
    print(f"   Time: {jax_time:.3f}s")
    print(f"   Unique trees: {len(jax_stat['trees'])}")

    # Show first 3 trees
    for i, (tree_key, prob) in enumerate(list(jax_stat['trees'].items())[:3]):
        tree_bindings = [net.binding_names[idx] for idx in tree_key]
        print(f"     {i+1}. p={prob:.3f}: {len(tree_bindings)} bindings")

    print(f"\n   Speedup: {cpu_time/jax_time:.2f}×")

print("\n✓ Prefix handling test complete!")
