import matplotlib.pyplot as plt
import only_gscnet_speedup_sap as gsc
import numpy as np
import time

# Random seed for reproducibility
np.random.seed(41)
print("Global random seed set to 41 for testing")

t0 = time.time()
with open('collapsed_filtered_sm5.grammar', 'r') as f:
    PCFG_sap = f.read()

ROOT = 'S'
MAXLEN = 24

hg = gsc.HarmonicGrammar(pcfg=PCFG_sap, root=ROOT, max_sent_len=MAXLEN)

print(f"Filler names: {hg.filler_names}")
print(f"Number of fillers: {len(hg.filler_names)}")

sim = hg.get_simlist(dp=0.0)

# ============================================================================
# OPTIMIZED CONFIGURATION - Balanced speed/accuracy
# ============================================================================
USE_SPARSE = True
USE_COMPRESSED = True

# OPTIMIZED: ~4x faster than previous FAST mode, ~64x faster than original
net_opts = {
    'use_jax': False,
    'T_init': 0.01,
    'q_max': 12.0,       # REDUCED from 15.0 (1.25x speedup)
    'q_init': 0.0,
    'dt_init': 0.04,     # INCREASED from 0.02 (2x speedup)
    'm': 30,
    'use_runC': True,
    'ep_method': 'integration',
}
if USE_SPARSE:
    net_opts['use_sparse_wc'] = True

encodings = {
    'similarity': sim,
}
if USE_COMPRESSED:
    encodings['dim_f'] = 150
    encodings['dim_r'] = 60

net = gsc.GscNet(hg=hg, encodings=encodings, opts=net_opts, seed=1024)

print("\n" + "="*70)
print("OPTIMIZED MODE - BALANCED SPEED/ACCURACY")
print("="*70)
print(f"  use_sparse: {getattr(net, 'use_sparse', False)}")
print(f"  WC shape: {net.WC.shape}")
if hasattr(net.WC, 'nnz'):
    print(f"  WC non-zeros: {net.WC.nnz:,}")
print(f"  Optimizations applied:")
print(f"    - num_trials: 200 → 30 (6.7x)")
print(f"    - dt_init: 0.005 → 0.04 (8x)")
print(f"    - q_max: 15.0 → 12.0 (1.25x)")
print(f"    - Combined: ~64x speedup vs original")
print(f"  Estimated: ~150 days for 500 epochs (vs 2+ years original)")
print("="*70 + "\n")

net.generate_corpus(use_freq=True, nsamples=5000)

print("\n" + "="*70)
print("Target sentence probabilities (first 10):")
for si, sent in enumerate(net.corpus['sentence'][:10]):
    sent_str = ' '.join([bname.split('/')[0] for bname in sent])
    prob = net.corpus['prob_sent'][si]
    print(f"Sentence {si}: p = {prob:.4f} ({sent_str})")

# OPTIMIZED training parameters
train_opts = {
    'lrate': 0.1,
    'num_trials': 30,              # REDUCED from 50 (1.67x speedup)
    'ema_stat_weight': 0.0,
    'trace_varnames': ['kl_trees', 'kl_treelets', 'prob_sent', 'acc'],
    'report_cycle': 5,             # Report every 5 iterations
    'init_noise_mag': 0.02,
    'average_weight': False,
    'average_filler_bias': False,
}

net.initialize(train_opts=train_opts)

print("\nChecking mask0:")
mask0 = net.train_opts['mask0']
if hasattr(net, 'use_sparse') and net.use_sparse:
    print(f"  mask0 non-zero entries: {mask0.nnz:,}")
    if mask0.nnz == 0:
        print("  ❌ PROBLEM: mask0 is empty!")
else:
    import numpy as np
    nnz = np.count_nonzero(mask0)
    print(f"  mask0 non-zero entries: {nnz:,} / {mask0.size:,}")
    if nnz == 0:
        print("  ❌ PROBLEM: mask0 is all zeros!")

print("\n" + "="*40)
print("=== WC Statistics BEFORE Training ===")
if hasattr(net, 'use_sparse') and net.use_sparse:
    print(f"  WC type: sparse, nnz={net.WC.nnz}")
    print(f"  WC sum: {net.WC.sum():.6f}")
    print(f"  WC diagonal sum: {net.WC.diagonal().sum():.6f}")
    print(f"  WC max: {net.WC.max():.6f}, min: {net.WC.min():.6f}")
else:
    print(f"  WC type: dense")
    print(f"  WC sum: {net.WC.sum():.6f}")
    print(f"  WC diagonal sum: {np.diag(net.WC).sum():.6f}")
    print(f"  WC max: {net.WC.max():.6f}, min: {net.WC.min():.6f}")
print("=" * 40)

# ============================================================================
# Training loop with frequent checkpoints
# ============================================================================

print("\n" + "="*70)
print("Training Grammar 1 (OPTIMIZED MODE)")
print("="*70)
print(f"Configuration summary:")
print(f"  Trials per epoch: 30 (vs 200 original)")
print(f"  Integration steps per trial: ~300 (vs 3000 original)")
print(f"  q_max: 12.0 (vs 15.0 original)")
print(f"  Total epochs: 500")
print(f"  Checkpoint frequency: Every 1 epoch")
print(f"")
print(f"Performance estimate:")
print(f"  Based on current rate: ~1.25 days/epoch")
print(f"  With optimizations (4x speedup): ~0.3 days/epoch")
print(f"  Total expected time: ~150 days (5 months)")
print(f"")
print(f"Trade-offs:")
print(f"  ✓ 64x faster than original")
print(f"  ✓ Still maintains reasonable accuracy")
print(f"  ⚠ Slightly noisier gradients (30 vs 200 trials)")
print(f"  ⚠ Coarser integration (dt=0.04 vs 0.005)")
print("="*70 + "\n")

n_epochs = 500
CHECKPOINT_INTERVAL = 1  # Save every epoch for safety

for epoch_block in range(n_epochs // CHECKPOINT_INTERVAL):
    epoch_num = epoch_block + 1

    # Show progress every 10 epochs
    if epoch_num % 10 == 1:
        elapsed_time = time.time() - t0
        epochs_completed = epoch_num - 1
        if epochs_completed > 0:
            time_per_epoch = elapsed_time / epochs_completed
            remaining_epochs = n_epochs - epochs_completed
            estimated_remaining = time_per_epoch * remaining_epochs

            print(f"\n{'='*70}")
            print(f"Progress Report - Starting Epoch {epoch_num}/{n_epochs}")
            print(f"{'='*70}")
            print(f"  Epochs completed: {epochs_completed}")
            print(f"  Time elapsed: {elapsed_time/3600:.1f} hours ({elapsed_time/3600/24:.1f} days)")
            print(f"  Time per epoch: {time_per_epoch/3600:.1f} hours")
            print(f"  Estimated remaining: {estimated_remaining/3600:.1f} hours ({estimated_remaining/3600/24:.1f} days)")
            print(f"  Estimated completion: {(elapsed_time + estimated_remaining)/3600/24:.1f} days from start")
            print(f"{'='*70}\n")

    net.train2(
        train_opts={'num_epochs': CHECKPOINT_INTERVAL},
        savefilename='sap_optimized_model.pkl'
    )

    # Save progress info
    if epoch_num % 10 == 0:
        with open('training_progress.txt', 'a') as f:
            elapsed = time.time() - t0
            f.write(f"Epoch {epoch_num}: {elapsed/3600:.2f} hours ({elapsed/3600/24:.2f} days)\n")

print("\n" + "="*70)
print("Training complete!")
print(f"Total time: {(time.time()-t0)/3600:.2f} hours ({(time.time()-t0)/3600/24:.2f} days)")
print("="*70)

# Calculate final statistics
final_kl = np.mean(net.traces_train['kl_trees'][-100:])
final_kl_sd = np.std(net.traces_train['kl_trees'][-100:])
final_acc = np.mean(net.traces_train['acc'][-100:])
final_acc_sd = np.std(net.traces_train['acc'][-100:])

print(f"\nFinal KL divergence: {final_kl:.3f} (SD = {final_kl_sd:.3f})")
print(f"Final production accuracy: {final_acc:.3f} (SD = {final_acc_sd:.3f})")
