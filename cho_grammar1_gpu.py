"""
GPU-accelerated version of cho_grammar1.py using CuPy

This script runs the same Grammar 1 training but with GPU acceleration.
To run: python cho_grammar1_gpu.py

Note: Requires CuPy installation. Install with:
    pip install cupy-cuda11x  (for CUDA 11.x)
    pip install cupy-cuda12x  (for CUDA 12.x)
"""

import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend for cluster
import matplotlib.pyplot as plt
import gsc_gpu as gsc  # GPU-accelerated version
import numpy as np
import time

print("="*70)
print("GPU-ACCELERATED GRAMMAR 1 TRAINING")
print("="*70)

# Show GPU info
if gsc.GPU_AVAILABLE:
    gsc.print_gpu_memory()
else:
    print("Warning: Running on CPU (CuPy not available)")

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
# Initialize network with paper's specifications
# ============================================================================

print("\nInitializing Harmonic Grammar...")
start_time = time.time()

hg = gsc.HarmonicGrammar(pcfg=PCFG_G1, root=ROOT, max_sent_len=MAXLEN)

# Display fillers (should have 27 fillers × 15 roles = 405 units)
print(f"Filler names: {hg.filler_names}")
print(f"Number of fillers: {len(hg.filler_names)}")

# Set all filler similarities to 0 (linear independence)
sim = hg.get_simlist(dp=0.0)

# Network options matching paper's parameters
net_opts = {
    'T_init': 0.01,      # computational temperature
    'q_max': 15.0,       # maximum commitment
    'q_init': 0.0,       # initial commitment
    'dt_init': 0.005,    # time step
    'm': 30,             # resource constraint (Hq1 strength)
    'use_runC': True,    # use C implementation for speed
}

# Initialize network
print("\nInitializing GscNet...")
net = gsc.GscNet(hg=hg, encodings={'similarity': sim},
                 opts=net_opts, seed=1024)
net.generate_corpus(use_freq=True)

print(f"Initialization time: {time.time() - start_time:.2f}s")

if gsc.GPU_AVAILABLE:
    gsc.print_gpu_memory()

# Display target probabilities
print("\n" + "="*70)
print("Target sentence probabilities:")
for si, sent in enumerate(net.corpus['sentence']):
    sent_str = ' '.join([bname.split('/')[0] for bname in sent])
    prob = net.corpus['prob_sent'][si]
    print(f"Sentence {si}: p = {prob:.4f} ({sent_str})")

# ============================================================================
# Training setup (matching Section 4 parameters)
# ============================================================================

train_opts = {
    'lrate': 0.1,                  # learning rate
    'num_trials': 4,               # production trials per iteration
    'ema_stat_weight': 0.0,        # no EMA smoothing initially
    'trace_varnames': ['kl_trees', 'kl_treelets', 'prob_sent', 'acc'],
    'report_cycle': 10,            # report every 10 iterations
    'init_noise_mag': 0.02,
    'average_weight': False,
    'average_filler_bias': False,
}

net.initialize(train_opts=train_opts)

# ============================================================================
# Training loop for Figure 11
# ============================================================================

print("\n" + "="*70)
print("Training Grammar 1 (GPU-accelerated)...")
print("="*70)

n_epochs = 1000  # Train for sufficient epochs to reach convergence

training_start = time.time()
epoch_times = []

for epoch_block in range(n_epochs // 10):
    block_start = time.time()

    net.train2(
        train_opts={'num_epochs': 10},
        savefilename='g1_model_gpu.pkl'
    )

    block_time = time.time() - block_start
    epoch_times.append(block_time)

    # Print progress with timing
    epochs_done = (epoch_block + 1) * 10
    elapsed = time.time() - training_start
    avg_time_per_10 = np.mean(epoch_times)
    remaining = (n_epochs - epochs_done) / 10 * avg_time_per_10

    print(f"Epochs {epochs_done}/{n_epochs} | "
          f"Block: {block_time:.1f}s | "
          f"Elapsed: {elapsed/60:.1f}m | "
          f"ETA: {remaining/60:.1f}m")

    # Show GPU memory every 100 epochs
    if gsc.GPU_AVAILABLE and (epoch_block + 1) % 10 == 0:
        gsc.print_gpu_memory()

total_training_time = time.time() - training_start

print("\n" + "="*70)
print("Training complete!")
print(f"Total training time: {total_training_time/60:.2f} minutes")
print(f"Average time per 10 epochs: {np.mean(epoch_times):.2f}s")
print("="*70)

# Calculate final statistics (last 100 updates)
final_kl = np.mean(net.traces_train['kl_trees'][-100:])
final_kl_sd = np.std(net.traces_train['kl_trees'][-100:])
final_acc = np.mean(net.traces_train['acc'][-100:])
final_acc_sd = np.std(net.traces_train['acc'][-100:])

print(f"Final KL divergence: {final_kl:.3f} (SD = {final_kl_sd:.3f})")
print(f"Final production accuracy: {final_acc:.3f} (SD = {final_acc_sd:.3f})")

# Display final learned probabilities
print("\nFinal learned probabilities Q(S):")
final_probs = np.mean(net.traces_train['prob_sent'][-100:], axis=0)
for si, prob in enumerate(final_probs):
    print(f"Sentence {si}: Q = {prob:.3f}")

# ============================================================================
# Plot Figure 11 (Training dynamics)
# ============================================================================

net = gsc.load_model('g1_model_gpu.pkl', use_gpu=True)

print("\n" + "="*70)
print("Generating training plots (Figure 11)...")
print("="*70)
gsc.plot_train_result(net, legend=True, linewidth=1.5)

# ============================================================================
# Parsing tests for Figure 12
# ============================================================================

print("\n" + "="*70)
print("Testing parsing accuracy (Figure 12)...")
print("="*70)

# Test parsing at different commitment levels (t ∈ {1, 2, ..., 12})
commitment_levels = list(range(1, 13))
parsing_accuracy = []

parsing_start = time.time()

for t in commitment_levels:
    # Create commitment policy: use fixed commitment per word
    max_sent_len = net.hg.opts['max_sent_len']
    dq = np.ones(max_sent_len) * (float(t) / max_sent_len)

    try:
        parse_results = gsc.test_parse_inc(
            net,
            dq=dq,
            num_trials=10,
            estr=2,
            estr_null=2,
            disp=False
        )

        # Calculate overall accuracy
        n_correct = sum([parse_results[si]['acc'] for si in parse_results])
        n_total = len(parse_results)
        acc = n_correct / n_total if n_total > 0 else 0.0

    except Exception as e:
        print(f"  Warning: Parsing test failed at t={t}: {e}")
        acc = 0.0

    parsing_accuracy.append(acc)
    print(f"Commitment t={t:2d}: Parsing accuracy = {acc:.3f}")

parsing_time = time.time() - parsing_start
print(f"Parsing tests time: {parsing_time:.2f}s")

# Plot Figure 12
plt.figure(figsize=(8, 6))
plt.plot(commitment_levels, parsing_accuracy, 'o-', linewidth=2, markersize=8)
plt.xlabel('Commitment Level (t)', fontsize=12)
plt.ylabel('Parsing Accuracy', fontsize=12)
plt.title('Grammar 1 (G1) Parsing Accuracy (GPU)', fontsize=14)
plt.grid(True, alpha=0.3)
plt.ylim([0, 1.05])
plt.tight_layout()
plt.savefig('figure12_g1_parsing_gpu.png', dpi=300, bbox_inches='tight')
print("Saved: figure12_g1_parsing_gpu.png")

# ============================================================================
# Final summary
# ============================================================================

print("\n" + "="*70)
print("REPLICATION COMPLETE (GPU)")
print("="*70)
print(f"Total runtime: {(time.time() - start_time)/60:.2f} minutes")
print(f"Training time: {total_training_time/60:.2f} minutes")
print(f"Parsing time: {parsing_time/60:.2f} minutes")
print("\nFigures saved:")
print("  - Plots from plot_train_result() (saved by gsc)")
print("  - figure12_g1_parsing_gpu.png")
if gsc.GPU_AVAILABLE:
    print("\n" + "="*70)
    gsc.print_gpu_memory()
print("="*70)
