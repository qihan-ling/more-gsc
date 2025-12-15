"""
Test S4 parsing with different random seeds to check if it's variance or systematic
"""

import numpy as np
import only_gscnet_speedup_sap as gsc

print("="*70)
print("Testing S4 parsing with different seeds and more trials")
print("="*70)

# Load model
net = gsc.load_model('sap_g1_model_fixed_sparse_nocompress.pkl')
if hasattr(net, 'use_jax'):
    net.use_jax = False
    net.opts['use_jax'] = False

# Test configuration
t = 5
max_sent_len = net.hg.opts['max_sent_len']
dq = np.ones(max_sent_len) * (float(t) / max_sent_len)

print("\nS4: N Vpp P N Vi (expected: 80%)")
print("-"*70)

# Test with multiple seeds
seeds = [41, 42, 100, 200, 1024]
results = []

for seed in seeds:
    np.random.seed(seed)
    parse_results = gsc.test_parse_inc(
        net,
        dq=dq,
        num_sent=5,  # Just test first 5 sentences
        num_trials=100,  # More trials for better statistics
        estr=2,
        estr_null=2,
        disp=False
    )

    s4_acc = parse_results[4]['acc'] if 4 in parse_results else 0.0
    results.append(s4_acc)
    print(f"Seed {seed:4d}: S4 accuracy = {s4_acc*100:5.1f}%")

avg_acc = np.mean(results)
std_acc = np.std(results)

print("-"*70)
print(f"Average: {avg_acc*100:.1f}% ± {std_acc*100:.1f}%")
print(f"Expected: 80%")
print("="*70)

if avg_acc < 0.6:
    print("\n❌ SYSTEMATIC PROBLEM: Average is significantly below expected")
    print("   Possible causes:")
    print("   1. Model has different weights than original (training diverged)")
    print("   2. Parsing parameters (estr, dq) are incorrect")
    print("   3. Model loading changes some state")
elif avg_acc > 0.7:
    print("\n✓ Results are close to expected")
    print("  Small variations are normal due to random noise in parsing")
else:
    print("\n⚠ Borderline: Could be statistical or could be systematic")
    print("  Run with even more trials to be sure")
