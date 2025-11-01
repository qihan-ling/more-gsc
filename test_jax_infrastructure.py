#!/usr/bin/env python
"""
Test script for JAX infrastructure with saved model.
This tests the batching overhead before implementing full dynamics.
"""

import time
import numpy as np
import gsc

# Load the saved model
print("Loading model from g1_model.pkl...")
net = gsc.load_model('g1_model.pkl')
print(f"✓ Model loaded successfully")
print(f"  Network has {net.num_bindings} bindings, {net.num_roles} roles, {net.num_fillers} fillers")

# Test parameters
num_trials = 10
prefix = []  # Empty prefix for simplicity

print(f"\n{'='*60}")
print(f"Testing with {num_trials} trials (empty prefix)")
print(f"{'='*60}")

# Test 1: Original CPU version (baseline)
print("\n1. Running original CPU version (estimate_prob_inc)...")
t0 = time.time()
stat_cpu, actC_list_cpu = net.estimate_prob_inc(prefix, num_trials=num_trials, progress=0)
cpu_time = time.time() - t0
print(f"   CPU time: {cpu_time:.3f}s")
print(f"   Result shape: {actC_list_cpu.shape}")
print(f"   Unique states found: {len(stat_cpu['trees'])}")

# Test 2: JAX version (currently returns zeros - placeholder)
print("\n2. Running JAX GPU version (estimate_prob_inc_jax)...")
print("   NOTE: This currently uses placeholder dynamics (returns zeros)")
t0 = time.time()
try:
    stat_jax, actC_list_jax = net.estimate_prob_inc_jax(prefix, num_trials=num_trials, progress=0)
    jax_time = time.time() - t0
    print(f"   JAX time: {jax_time:.3f}s (placeholder)")
    print(f"   Result shape: {actC_list_jax.shape}")
    print(f"   Unique states found: {len(stat_jax['trees'])}")

    # Show overhead
    print(f"\n3. Analysis:")
    print(f"   CPU version: {cpu_time:.3f}s")
    print(f"   JAX overhead (data transfer only): {jax_time:.3f}s")
    print(f"   ⚠️  JAX is using placeholder dynamics (all zeros)")
    print(f"   Once dynamics are implemented, expect 50-200× speedup")

except Exception as e:
    print(f"   ✗ Error: {e}")
    import traceback
    traceback.print_exc()

print(f"\n{'='*60}")
print("Next step: Implement _run_single_trial_jax() with real dynamics")
print(f"{'='*60}")

# Show what the network looks like
print(f"\nNetwork details:")
print(f"  - Filler names: {net.filler_names[:5]}..." if len(net.filler_names) > 5 else f"  - Filler names: {net.filler_names}")
print(f"  - Role names: {net.role_names[:5]}..." if len(net.role_names) > 5 else f"  - Role names: {net.role_names}")
print(f"  - Has corpus: {hasattr(net, 'corpus')}")
if hasattr(net, 'corpus'):
    print(f"  - Corpus size: {len(net.corpus.get('sentence', []))} sentences")
