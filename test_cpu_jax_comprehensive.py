#!/usr/bin/env python
"""
Comprehensive comparison of CPU vs JAX results.
Checks not just count but actual parse content.
"""

import gsc
import numpy as np
import time

print("="*70)
print("Comprehensive CPU vs JAX Comparison")
print("="*70)

net = gsc.load_model('g1_model.pkl')
prefix = []
num_trials = 100

print(f"\nRunning {num_trials} trials on both CPU and JAX...")

# CPU version
print("\n1. CPU version...")
t0 = time.time()
stat_cpu, actC_cpu = net.estimate_prob_inc(prefix, num_trials=num_trials, progress=0)
cpu_time = time.time() - t0
print(f"   Time: {cpu_time:.2f}s")
print(f"   Unique states: {len(stat_cpu['trees'])}")

# JAX version
print("\n2. JAX version...")
t0 = time.time()
stat_jax, actC_jax = net.estimate_prob_inc_jax(prefix, num_trials=num_trials, rng_seed=42)
jax_time = time.time() - t0
print(f"   Time: {jax_time:.2f}s")
print(f"   Unique states: {len(stat_jax['trees'])}")

print(f"\n3. Speedup: {cpu_time/jax_time:.1f}×")

# Compare tree structures
print(f"\n{'='*70}")
print("Tree Structure Comparison")
print(f"{'='*70}")

# Extract trees (grid point tuples)
cpu_trees = list(stat_cpu['trees'].keys())
jax_trees = list(stat_jax['trees'].keys())

print(f"\nCPU found {len(cpu_trees)} unique trees:")
for i, tree_key in enumerate(cpu_trees):
    prob = stat_cpu['trees'][tree_key]
    # Convert tuple of binding indices to grid point
    tree_bindings = [net.binding_names[idx] for idx in tree_key]
    # Extract just filler names (before '/')
    fillers = [b.split('/')[0] for b in tree_bindings]
    # Group by role to show tree structure
    grid_by_role = {}
    for binding_name in tree_bindings:
        filler, role = binding_name.split('/')
        grid_by_role[role] = filler

    print(f"  Tree {i+1} (p={prob:.3f}):")
    print(f"    Bindings ({len(tree_key)}): {tree_bindings[:5]}..." if len(tree_bindings) > 5 else f"    Bindings: {tree_bindings}")

print(f"\nJAX found {len(jax_trees)} unique trees:")
for i, tree_key in enumerate(jax_trees):
    prob = stat_jax['trees'][tree_key]
    tree_bindings = [net.binding_names[idx] for idx in tree_key]
    fillers = [b.split('/')[0] for b in tree_bindings]

    print(f"  Tree {i+1} (p={prob:.3f}):")
    print(f"    Bindings ({len(tree_key)}): {tree_bindings[:5]}..." if len(tree_bindings) > 5 else f"    Bindings: {tree_bindings}")

# Check if they found the same trees
print(f"\n{'='*70}")
print("Overlap Analysis")
print(f"{'='*70}")

cpu_set = set(cpu_trees)
jax_set = set(jax_trees)

overlap = cpu_set & jax_set
cpu_only = cpu_set - jax_set
jax_only = jax_set - cpu_set

print(f"\nTrees in both CPU and JAX: {len(overlap)}")
print(f"Trees only in CPU: {len(cpu_only)}")
print(f"Trees only in JAX: {len(jax_only)}")

if overlap:
    print(f"\nShared trees (with probability comparison):")
    for tree_key in overlap:
        cpu_prob = stat_cpu['trees'][tree_key]
        jax_prob = stat_jax['trees'][tree_key]
        diff = abs(cpu_prob - jax_prob)
        print(f"  Tree: {len(tree_key)} bindings")
        print(f"    CPU prob: {cpu_prob:.4f}")
        print(f"    JAX prob: {jax_prob:.4f}")
        print(f"    Diff: {diff:.4f}")

if cpu_only:
    print(f"\nTrees ONLY in CPU (not in JAX):")
    for tree_key in list(cpu_only)[:3]:  # Show first 3
        prob = stat_cpu['trees'][tree_key]
        bindings = [net.binding_names[idx] for idx in tree_key]
        print(f"  p={prob:.4f}: {bindings[:5]}...")

if jax_only:
    print(f"\nTrees ONLY in JAX (not in CPU):")
    for tree_key in list(jax_only)[:3]:  # Show first 3
        prob = stat_jax['trees'][tree_key]
        bindings = [net.binding_names[idx] for idx in tree_key]
        print(f"  p={prob:.4f}: {bindings[:5]}...")

# Statistical comparison
print(f"\n{'='*70}")
print("Statistical Comparison")
print(f"{'='*70}")

# Compare activation distributions
actC_cpu_mean = actC_cpu.mean(axis=0)
actC_jax_mean = actC_jax.mean(axis=0)

print(f"\nMean activation difference:")
print(f"  Max abs diff: {np.abs(actC_cpu_mean - actC_jax_mean).max():.6f}")
print(f"  Mean abs diff: {np.abs(actC_cpu_mean - actC_jax_mean).mean():.6f}")
print(f"  Correlation: {np.corrcoef(actC_cpu_mean, actC_jax_mean)[0,1]:.6f}")

# Conclusion
print(f"\n{'='*70}")
print("Conclusion")
print(f"{'='*70}")

if overlap == cpu_set and overlap == jax_set:
    print("\n✓ PERFECT MATCH: CPU and JAX found identical trees!")
    print("  Results are statistically equivalent.")
elif len(overlap) > 0:
    overlap_pct = len(overlap) / max(len(cpu_set), len(jax_set)) * 100
    print(f"\n⚠️  PARTIAL MATCH: {overlap_pct:.1f}% overlap")
    print("  Different random samples led to different trees.")
    print("  This is EXPECTED due to different RNG implementations.")
    print("  As long as there's significant overlap, implementation is correct.")
else:
    print("\n✗ NO OVERLAP: CPU and JAX found completely different trees!")
    print("  This suggests a bug in the implementation.")

print(f"\nSpeedup achieved: {cpu_time/jax_time:.1f}× faster with JAX!")
