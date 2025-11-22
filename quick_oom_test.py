"""Quick OOM Memory Stress Test for GSC Network

This script performs a rapid memory stress test to detect OOM issues
without waiting hours. It runs minimal training iterations while monitoring
memory usage at each step.

Expected runtime: 2-5 minutes (vs 5+ hours for full training)
"""

import matplotlib.pyplot as plt
import gsc
import numpy as np
import psutil
import os
import gc

def get_memory_usage_mb():
    """Get current process memory usage in MB"""
    process = psutil.Process(os.getpid())
    return process.memory_info().rss / 1024 / 1024

def print_memory_step(step_name, mem_start):
    """Print memory usage for a step"""
    mem_current = get_memory_usage_mb()
    mem_delta = mem_current - mem_start
    print(f"  [{step_name}]")
    print(f"    Current: {mem_current:.1f} MB | Delta: {mem_delta:+.1f} MB")
    return mem_current

print("="*70)
print("QUICK OOM STRESS TEST")
print("="*70)
print(f"Initial memory: {get_memory_usage_mb():.1f} MB\n")

# Load grammar
with open('collapsed_filtered_sm5.grammar', 'r') as f:
    PCFG_sap = f.read()

ROOT = 'S'
MAXLEN = 24

print("STEP 1: Initialize HarmonicGrammar")
mem_start = get_memory_usage_mb()
hg = gsc.HarmonicGrammar(pcfg=PCFG_sap, root=ROOT, max_sent_len=MAXLEN)
mem_after_hg = print_memory_step("After HarmonicGrammar", mem_start)

print(f"\nNumber of fillers: {len(hg.filler_names)}")

# Set all filler similarities to 0
sim = hg.get_simlist(dp=0.0)

# Network options
net_opts = {
    'use_jax': False,
    'T_init': 0.01,
    'q_max': 15.0,
    'q_init': 0.0,
    'dt_init': 0.005,
    'm': 30,
    'use_runC': True,
    'ep_method': 'integration',
}

encodings = {
    'similarity': sim,
    'dim_f': 150,
    'dim_r': 60
}

print("\nSTEP 2: Initialize GscNet")
mem_start = get_memory_usage_mb()
net = gsc.GscNet(hg=hg, encodings=encodings, opts=net_opts, seed=1024)
mem_after_net = print_memory_step("After GscNet.__init__", mem_start)

print("\nSTEP 3: Generate corpus (REDUCED: 100 samples instead of 5000)")
mem_start = get_memory_usage_mb()
# Use much smaller corpus for quick test
net.generate_corpus(use_freq=True, nsamples=100)
mem_after_corpus = print_memory_step("After generate_corpus", mem_start)

print("\nSTEP 4: Initialize training (THIS IS WHERE OOM OCCURRED)")
train_opts = {
    'lrate': 0.1,
    'num_trials': 20,  # Reduced from 200
    'ema_stat_weight': 0.0,
    'trace_varnames': ['kl_trees', 'kl_treelets', 'prob_sent', 'acc'],
    'report_cycle': 1,  # Report every iteration for detailed monitoring
    'init_noise_mag': 0.02,
    'average_weight': False,
    'average_filler_bias': False,
}

mem_start = get_memory_usage_mb()
print(f"  Memory before initialize(): {mem_start:.1f} MB")

# This is where the OOM error occurred
net.initialize(train_opts=train_opts)

mem_after_init = print_memory_step("After initialize()", mem_start)

print("\nSTEP 5: Run SHORT training test (3 epochs to check for memory leaks)")
mem_before_train = get_memory_usage_mb()
mem_history = [mem_before_train]

print("  Running 3 test epochs...")
for epoch in range(3):
    mem_epoch_start = get_memory_usage_mb()
    print(f"\n  Epoch {epoch+1}/3:")
    print(f"    Memory at start: {mem_epoch_start:.1f} MB")

    net.train2(
        train_opts={'num_epochs': 1},
        savefilename=None  # Don't save during test
    )

    mem_epoch_end = get_memory_usage_mb()
    mem_delta = mem_epoch_end - mem_epoch_start
    mem_history.append(mem_epoch_end)

    print(f"    Memory at end: {mem_epoch_end:.1f} MB (delta: {mem_delta:+.1f} MB)")

    # Force garbage collection
    gc.collect()

mem_after_train = get_memory_usage_mb()

print("\n" + "="*70)
print("MEMORY STRESS TEST SUMMARY")
print("="*70)

print("\nMemory Usage by Step:")
print(f"  1. After HarmonicGrammar:  {mem_after_hg:.1f} MB")
print(f"  2. After GscNet init:      {mem_after_net:.1f} MB")
print(f"  3. After corpus gen:       {mem_after_corpus:.1f} MB")
print(f"  4. After initialize():     {mem_after_init:.1f} MB  ← OOM occurred here")
print(f"  5. After training (3 ep):  {mem_after_train:.1f} MB")

print(f"\nTotal memory used: {mem_after_train:.1f} MB ({mem_after_train/1024:.2f} GB)")

# Check for memory leaks during training
if len(mem_history) > 1:
    mem_growth_per_epoch = np.diff(mem_history)
    avg_growth = np.mean(mem_growth_per_epoch)

    print(f"\nTraining Memory Growth Analysis:")
    print(f"  Average per epoch: {avg_growth:+.1f} MB")

    if avg_growth > 100:
        print("  ⚠️  WARNING: Significant memory growth detected!")
        print("     This may indicate a memory leak in training loop")
        estimated_epochs_to_oom = (16000 - mem_after_train) / avg_growth
        print(f"     Estimated epochs until OOM (~16GB): {estimated_epochs_to_oom:.0f}")
    elif avg_growth > 10:
        print("  ⚠️  CAUTION: Moderate memory growth detected")
        print("     Monitor long training runs carefully")
    else:
        print("  ✅ Memory growth is minimal - training should be stable")

print("\n" + "="*70)
print("TEST COMPLETE")
print("="*70)

if mem_after_init > 15000:  # >15GB
    print("\n⚠️  WARNING: Memory usage is very high after initialize()")
    print("   This may cause OOM on systems with <32GB RAM")
elif mem_after_train < 8000:  # <8GB
    print("\n✅ SUCCESS: Memory usage is reasonable!")
    print("   Full training should complete without OOM issues")
else:
    print("\n✓ Memory usage is acceptable")
    print("  Monitor long training runs on systems with <16GB RAM")
