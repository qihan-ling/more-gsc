"""
Profile training to identify bottlenecks in cho_grammar1.py
"""
import gsc
import numpy as np
import time
from collections import defaultdict

# ============================================================================
# Grammar 1 (G1) from Section 4.1
# ============================================================================

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

ROOT = 'S'
MAXLEN = 5

# ============================================================================
# Initialize network
# ============================================================================

print("Initializing network...")
t_init_start = time.time()

hg = gsc.HarmonicGrammar(pcfg=PCFG_G1, root=ROOT, max_sent_len=MAXLEN)
sim = hg.get_simlist(dp=0.0)

net_opts = {
    'T_init': 0.01,
    'q_max': 15.0,
    'q_init': 0.0,
    'dt_init': 0.005,
    'm': 30,
    'use_runC': True,
}

net = gsc.GscNet(hg=hg, encodings={'similarity': sim},
                 opts=net_opts, seed=1024)

t_corpus_start = time.time()
net.generate_corpus(use_freq=True)
t_corpus_end = time.time()

print(f"Corpus generation: {t_corpus_end - t_corpus_start:.2f}s")
print(f"Number of unique sentences: {len(net.corpus['sentence'])}")

train_opts = {
    'lrate': 0.1,
    'num_trials': 4,  # Start with 4 trials
    'ema_stat_weight': 0.0,
    'trace_varnames': ['kl_trees', 'kl_treelets', 'prob_sent', 'acc'],
    'report_cycle': 10,
    'init_noise_mag': 0.02,
    'average_weight': False,
    'average_filler_bias': False,
}

net.initialize(train_opts=train_opts)

t_init_end = time.time()
print(f"Total initialization: {t_init_end - t_init_start:.2f}s\n")

# ============================================================================
# Profiled training loop - 100 epochs for quick test
# ============================================================================

print("="*70)
print("PROFILING TRAINING (100 epochs)...")
print("="*70)

# Timing accumulators
timings = defaultdict(float)
call_counts = defaultdict(int)

# Monkey-patch key functions to add timing
original_estimate_prob_inc = net.estimate_prob_inc
original_estimate_prob_inc_jax = net.estimate_prob_inc_jax
original_get_corpus_stat = net.get_corpus_stat
original_cost = net.cost
original_cost_grad = net.cost_grad

def timed_estimate_prob_inc(*args, **kwargs):
    t0 = time.time()
    result = original_estimate_prob_inc(*args, **kwargs)
    timings['estimate_prob_inc'] += time.time() - t0
    call_counts['estimate_prob_inc'] += 1
    return result

def timed_estimate_prob_inc_jax(*args, **kwargs):
    t0 = time.time()
    result = original_estimate_prob_inc_jax(*args, **kwargs)
    timings['estimate_prob_inc_jax'] += time.time() - t0
    call_counts['estimate_prob_inc_jax'] += 1
    return result

def timed_get_corpus_stat(*args, **kwargs):
    t0 = time.time()
    result = original_get_corpus_stat(*args, **kwargs)
    timings['get_corpus_stat'] += time.time() - t0
    call_counts['get_corpus_stat'] += 1
    return result

def timed_cost(*args, **kwargs):
    t0 = time.time()
    result = original_cost(*args, **kwargs)
    timings['cost'] += time.time() - t0
    call_counts['cost'] += 1
    return result

def timed_cost_grad(*args, **kwargs):
    t0 = time.time()
    result = original_cost_grad(*args, **kwargs)
    timings['cost_grad'] += time.time() - t0
    call_counts['cost_grad'] += 1
    return result

# Apply monkey patches
net.estimate_prob_inc = timed_estimate_prob_inc
net.estimate_prob_inc_jax = timed_estimate_prob_inc_jax
net.get_corpus_stat = timed_get_corpus_stat
net.cost = timed_cost
net.cost_grad = timed_cost_grad

# Run training
t_train_start = time.time()

n_epochs = 100  # Quick test

for epoch_block in range(n_epochs // 10):
    t_block_start = time.time()

    net.train2(
        train_opts={'num_epochs': 10},
        savefilename=None  # Disable saving to isolate compute time
    )

    t_block_end = time.time()
    print(f"Block {epoch_block+1}/10: {t_block_end - t_block_start:.2f}s")

t_train_end = time.time()
total_train_time = t_train_end - t_train_start

# ============================================================================
# Report results
# ============================================================================

print("\n" + "="*70)
print("PROFILING RESULTS")
print("="*70)

print(f"\nTotal training time: {total_train_time:.2f}s")
print(f"Time per epoch: {total_train_time / n_epochs:.3f}s")

# Calculate time spent in other code
accounted_time = sum(timings.values())
other_time = total_train_time - accounted_time

print(f"\nTime breakdown:")
print(f"{'Function':<30} {'Total (s)':<12} {'Per call (ms)':<15} {'% of total':<12} {'Calls':<8}")
print("-" * 85)

# Sort by total time
sorted_funcs = sorted(timings.items(), key=lambda x: x[1], reverse=True)

for func_name, total_time in sorted_funcs:
    calls = call_counts[func_name]
    per_call = (total_time / calls * 1000) if calls > 0 else 0
    pct = (total_time / total_train_time * 100)
    print(f"{func_name:<30} {total_time:<12.2f} {per_call:<15.3f} {pct:<12.1f}% {calls:<8}")

print("-" * 85)
print(f"{'Other (loop overhead, etc.)':<30} {other_time:<12.2f} {'-':<15} {other_time/total_train_time*100:<12.1f}%")
print("-" * 85)
print(f"{'TOTAL':<30} {total_train_time:<12.2f}")

# ============================================================================
# Test with different num_trials settings
# ============================================================================

print("\n" + "="*70)
print("TESTING DIFFERENT NUM_TRIALS (10 epochs each)")
print("="*70)

# Reset network
net = gsc.GscNet(hg=hg, encodings={'similarity': sim},
                 opts=net_opts, seed=1024)
net.generate_corpus(use_freq=True)

for num_trials in [4, 50, 500]:
    train_opts['num_trials'] = num_trials
    net.initialize(train_opts=train_opts)

    print(f"\nTesting num_trials={num_trials}...")

    t0 = time.time()
    net.train2(train_opts={'num_epochs': 10}, savefilename=None)
    elapsed = time.time() - t0

    print(f"  Time for 10 epochs: {elapsed:.2f}s ({elapsed/10:.3f}s per epoch)")

print("\n" + "="*70)
print("Profile complete!")
print("="*70)
