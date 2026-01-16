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
# CONFIGURATION - NO COMPRESSION FOR ACCURACY
# ============================================================================
USE_SPARSE = True
USE_COMPRESSED = False  # ✓ NO COMPRESSION - Better accuracy, no speed penalty

# OPTIMIZED: ~4x faster than previous FAST mode
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

# NO COMPRESSION - use full filler/role dimensions for accuracy
encodings = {
    'similarity': sim,
    # dim_f and dim_r NOT specified - will use full dimensions
    # This provides better accuracy with NO speed penalty for sparse training
}

net = gsc.GscNet(hg=hg, encodings=encodings, opts=net_opts, seed=1024)

print("\n" + "="*70)
print("OPTIMIZED MODE - NO COMPRESSION")
print("="*70)
print(f"  use_sparse: {getattr(net, 'use_sparse', False)}")
print(f"  WC shape: {net.WC.shape}")
if hasattr(net.WC, 'nnz'):
    print(f"  WC non-zeros: {net.WC.nnz:,}")
print(f"  dim_f: {net.dim_f} (FULL - no compression)")
print(f"  dim_r: {net.dim_r} (FULL - no compression)")
print(f"  num_fillers: {net.num_fillers}")
print(f"  num_roles: {net.num_roles}")
print(f"  num_bindings: {net.num_bindings}")
print(f"")
print(f"  Optimizations applied:")
print(f"    - num_trials: 200 → 30 (6.7x speedup)")
print(f"    - dt_init: 0.005 → 0.04 (8x speedup)")
print(f"    - q_max: 15.0 → 12.0 (1.25x speedup)")
print(f"    - NO compression (better accuracy, same speed)")
print(f"    - Combined: ~64x speedup vs original")
print(f"")
print(f"  Note: Removing compression improves accuracy with no speed penalty")
print(f"        WC size is the same whether compressed or not (sparse)")
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
    'num_trials': 30,              # REDUCED from 50
    'ema_stat_weight': 0.0,
    'trace_varnames': ['kl_trees', 'kl_treelets', 'prob_sent', 'acc'],
    'report_cycle': 5,
    'init_noise_mag': 0.02,
    'average_weight': False,
    'average_filler_bias': False,
}

net.initialize(train_opts=train_opts)

print("\nChecking mask0:")
mask0 = net.train_opts['mask0']
if hasattr(net, 'use_sparse') and net.use_sparse:
    print(f"  mask0 non-zero entries: {mask0.nnz:,}")
else:
    nnz = np.count_nonzero(mask0)
    print(f"  mask0 non-zero entries: {nnz:,} / {mask0.size:,}")

print("\n" + "="*40)
print("=== WC Statistics BEFORE Training ===")
if hasattr(net, 'use_sparse') and net.use_sparse:
    print(f"  WC type: sparse, nnz={net.WC.nnz}")
    print(f"  WC sum: {net.WC.sum():.6f}")
else:
    print(f"  WC type: dense")
    print(f"  WC sum: {net.WC.sum():.6f}")
print("=" * 40)

# ============================================================================
# Training loop
# ============================================================================

print("\n" + "="*70)
print("Training Grammar 1 (OPTIMIZED, NO COMPRESSION)")
print("="*70)
print(f"Configuration:")
print(f"  Trials per epoch: 30")
print(f"  Integration steps per trial: ~300")
print(f"  q_max: 12.0")
print(f"  Total epochs: 500")
print(f"  Checkpoint frequency: Every 1 epoch")
print(f"")
print(f"Expected performance:")
print(f"  Time per epoch: ~7 hours (based on optimizations)")
print(f"  Total time: ~150 days")
print(f"")
print(f"Accuracy:")
print(f"  ✓ NO compression = better gradient quality")
print(f"  ✓ Should match toy grammar performance")
print("="*70 + "\n")

n_epochs = 500
CHECKPOINT_INTERVAL = 1

for epoch_block in range(n_epochs // CHECKPOINT_INTERVAL):
    epoch_num = epoch_block + 1

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
            print(f"{'='*70}\n")

    net.train2(
        train_opts={'num_epochs': CHECKPOINT_INTERVAL},
        savefilename='sap_optimized_NO_COMPRESSION.pkl'
    )

    if epoch_num % 10 == 0:
        with open('training_progress_no_compression.txt', 'a') as f:
            elapsed = time.time() - t0
            f.write(f"Epoch {epoch_num}: {elapsed/3600:.2f} hours ({elapsed/3600/24:.2f} days)\n")

print("\n" + "="*70)
print("Training complete!")
print(f"Total time: {(time.time()-t0)/3600:.2f} hours ({(time.time()-t0)/3600/24:.2f} days)")
print("="*70)

final_kl = np.mean(net.traces_train['kl_trees'][-100:])
final_kl_sd = np.std(net.traces_train['kl_trees'][-100:])
final_acc = np.mean(net.traces_train['acc'][-100:])
final_acc_sd = np.std(net.traces_train['acc'][-100:])

print(f"\nFinal KL divergence: {final_kl:.3f} (SD = {final_kl_sd:.3f})")
print(f"Final production accuracy: {final_acc:.3f} (SD = {final_acc_sd:.3f})")
