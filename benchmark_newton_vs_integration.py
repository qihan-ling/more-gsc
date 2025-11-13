#!/usr/bin/env python3
"""
Benchmark Newton vs Integration methods for equilibrium calculation
"""
import matplotlib
matplotlib.use('Agg')
import only_gscnet as gsc
import numpy as np
import time

PCFG_G1 = '''
0.35 S -> N Vi
0.60 S -> N VP
0.05 S -> NP Vi
1.0 NP -> N RC
1.0 RC -> Vpp PP
1.0 VPpp -> Vpp PP
1.0 PP -> P N
0.5 VP -> Vi PP
0.3 VP -> BE Vpp
0.2 VP -> BE VPpp
'''

print("="*70)
print("Benchmarking Newton vs Integration for Equilibrium Calculation")
print("="*70)

# Create HarmonicGrammar
hg = gsc.HarmonicGrammar(pcfg=PCFG_G1, root='S', max_sent_len=5)
sim = hg.get_simlist(dp=0.0)

net_opts = {
    'T_init': 0.01,
    'q_max': 15.0,
    'q_init': 0.0,
    'dt_init': 0.005,
    'm': 30,
    'use_runC': True,
}

# Create network
net = gsc.GscNet(hg=hg, encodings={'similarity': sim}, opts=net_opts, seed=1024)

print(f"\nNetwork size:")
print(f"  Fillers: {net.num_fillers}")
print(f"  Roles: {net.num_roles}")
print(f"  Bindings: {net.num_bindings}")
print(f"  Weight matrix: {net.num_bindings} × {net.num_bindings} = {net.num_bindings**2:,} elements")

# Benchmark Newton method
print("\n" + "="*70)
print("Testing Newton Method")
print("="*70)

times_newton = []
for trial in range(5):
    # Reset to random state
    net.set_random_state(minact=0.2, maxact=0.8)

    t0 = time.time()
    net.get_ep(method='newton')
    elapsed = time.time() - t0
    times_newton.append(elapsed)
    print(f"  Trial {trial+1}: {elapsed:.4f}s")

avg_newton = np.mean(times_newton)
std_newton = np.std(times_newton)
print(f"\nNewton average: {avg_newton:.4f}s ± {std_newton:.4f}s")

# Benchmark Integration method
print("\n" + "="*70)
print("Testing Integration Method (dur=10)")
print("="*70)

times_integration = []
for trial in range(5):
    # Reset to random state
    net.set_random_state(minact=0.2, maxact=0.8)

    t0 = time.time()
    net.get_ep(method='integration', dur=10)
    elapsed = time.time() - t0
    times_integration.append(elapsed)
    print(f"  Trial {trial+1}: {elapsed:.4f}s")

avg_integration = np.mean(times_integration)
std_integration = np.std(times_integration)
print(f"\nIntegration average: {avg_integration:.4f}s ± {std_integration:.4f}s")

# Compare
print("\n" + "="*70)
print("Comparison")
print("="*70)
speedup = avg_newton / avg_integration
print(f"Newton:      {avg_newton:.4f}s")
print(f"Integration: {avg_integration:.4f}s")
if speedup > 1:
    print(f"Integration is {speedup:.2f}x FASTER than Newton")
else:
    print(f"Newton is {1/speedup:.2f}x FASTER than Integration")

# Estimate for training
print("\n" + "="*70)
print("Estimated Training Time Impact")
print("="*70)
print("""
During training, get_ep() is called once per epoch to update equilibrium.
For 200 epochs:
""")
print(f"  Newton:      {avg_newton * 200:.1f}s = {avg_newton * 200 / 60:.1f} minutes")
print(f"  Integration: {avg_integration * 200:.1f}s = {avg_integration * 200 / 60:.1f} minutes")
print(f"  Difference:  {abs(avg_newton - avg_integration) * 200:.1f}s = {abs(avg_newton - avg_integration) * 200 / 60:.1f} minutes")

# Complexity analysis
print("\n" + "="*70)
print("Complexity Analysis")
print("="*70)
print(f"""
Current network: {net.num_bindings} bindings

Newton method:
  - Requires Hessian computation: O(n²) to compute, O(n³) to solve
  - Converges in ~5-20 iterations typically
  - Complexity: O(k × n³) where k = iteration count
  - For n={net.num_bindings}: ~{net.num_bindings**3:,} operations per iteration

Integration method:
  - Runs dynamics for fixed duration: dur/dt steps
  - Each step: O(n²) for matrix-vector multiply
  - Complexity: O((dur/dt) × n²)
  - For n={net.num_bindings}, dur=10, dt=0.005: ~{int(10/0.005)} steps × {net.num_bindings**2:,} = ~{int(10/0.005) * net.num_bindings**2:,} operations

Scaling to 1000-rule grammar:
  - Estimated bindings for 1k rules: ~1000-5000 (depends on grammar structure)
  - Newton: O(n³) → 10-100x more compute for 3-5x larger network
  - Integration: O(n²) → 3-25x more compute for 3-5x larger network

RECOMMENDATION: Integration scales better for large grammars.
""")

print("\n" + "="*70)
print("Possible Solutions")
print("="*70)
print("""
1. Use Newton for small grammars (<500 rules), Integration for large
2. Use Newton initially (first 50 epochs), then switch to Integration
3. Use Integration but with longer duration (dur=20 or 30) for accuracy
4. Implement approximate Newton (quasi-Newton) methods like BFGS
5. Use Integration but verify equilibrium quality periodically

The accuracy vs speed tradeoff is real. For your use case (1k grammar),
Integration might be necessary for practical training times.
""")
