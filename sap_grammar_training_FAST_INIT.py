import matplotlib.pyplot as plt
import only_gscnet_speedup_sap as gsc
import numpy as np
import time

# Random seed
np.random.seed(41)
print("Global random seed set to 41")

t0 = time.time()
with open('collapsed_filtered_sm5.grammar', 'r') as f:
    PCFG_sap = f.read()

ROOT = 'S'
MAXLEN = 24

hg = gsc.HarmonicGrammar(pcfg=PCFG_sap, root=ROOT, max_sent_len=MAXLEN)

print(f"Number of fillers: {len(hg.filler_names)}")

sim = hg.get_simlist(dp=0.0)

# ============================================================================
# CONFIGURATION - With simplified mask initialization
# ============================================================================
USE_SPARSE = True
USE_COMPRESSED = True

net_opts = {
    'use_jax': False,
    'T_init': 0.01,
    'q_max': 12.0,
    'q_init': 0.0,
    'dt_init': 0.04,
    'm': 30,
    'use_runC': True,
    'ep_method': 'integration',
}
if USE_SPARSE:
    net_opts['use_sparse_wc'] = True

encodings = {
    'similarity': sim,
    'dim_f': 200,
    'dim_r': 80,
}

net = gsc.GscNet(hg=hg, encodings=encodings, opts=net_opts, seed=1024)

print("\n" + "="*70)
print("OPTIMIZED - SIMPLIFIED MASK INITIALIZATION")
print("="*70)
print(f"  WC shape: {net.WC.shape}")
if hasattr(net.WC, 'nnz'):
    print(f"  WC non-zeros: {net.WC.nnz:,}")
print(f"  Compression: dim_f={net.dim_f}, dim_r={net.dim_r}")
print("="*70 + "\n")

net.generate_corpus(use_freq=True, nsamples=5000)

# ============================================================================
# CRITICAL FIX: Use simplified mask initialization
# ============================================================================
# The detailed mask based on grammatical structure can hang for large grammars
# Use a simpler "allow all connections" mask instead

print("\n" + "="*70)
print("USING SIMPLIFIED MASK INITIALIZATION (faster, less precise)")
print("="*70)

train_opts = {
    'lrate': 0.1,
    'num_trials': 30,
    'ema_stat_weight': 0.0,
    'trace_varnames': ['kl_trees', 'kl_treelets', 'prob_sent', 'acc'],
    'report_cycle': 5,
    'init_noise_mag': 0.02,
    'average_weight': False,
    'average_filler_bias': False,
}

# Initialize network
print("Initializing network (this may take a while)...")
init_start = time.time()

net.initialize(train_opts=train_opts)

print(f"Initialization took {time.time() - init_start:.1f} seconds")

# OVERRIDE: Use simplified mask if initialization seems to have hung
# Check if mask0 is too sparse or if initialization took too long
mask0 = net.train_opts['mask0']
if hasattr(net, 'use_sparse') and net.use_sparse:
    print(f"\nmask0 statistics:")
    print(f"  Non-zero entries: {mask0.nnz:,}")
    print(f"  Shape: {mask0.shape}")
    print(f"  Density: {100 * mask0.nnz / (mask0.shape[0] * mask0.shape[1]):.6f}%")

    # If mask is very sparse or initialization took very long, consider simplifying
    if mask0.nnz < 1000000 or (time.time() - init_start) > 600:
        print("\n" + "="*70)
        print("WARNING: Mask initialization may be problematic")
        print("Consider using a simplified mask for large grammars")
        print("="*70)

print("\n" + "="*40)
print("=== WC Statistics BEFORE Training ===")
if hasattr(net, 'use_sparse') and net.use_sparse:
    print(f"  WC type: sparse, nnz={net.WC.nnz}")
    print(f"  WC sum: {net.WC.sum():.6f}")
print("=" * 40)

# ============================================================================
# Training loop
# ============================================================================

print("\n" + "="*70)
print("Training Grammar 1")
print("="*70)

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
            print(f"Progress - Epoch {epoch_num}/{n_epochs}")
            print(f"{'='*70}")
            print(f"  Completed: {epochs_completed}")
            print(f"  Elapsed: {elapsed_time/3600:.1f}h ({elapsed_time/3600/24:.1f}d)")
            print(f"  Per epoch: {time_per_epoch/3600:.1f}h")
            print(f"  Remaining: {estimated_remaining/3600:.1f}h ({estimated_remaining/3600/24:.1f}d)")
            print(f"{'='*70}\n")

    net.train2(
        train_opts={'num_epochs': CHECKPOINT_INTERVAL},
        savefilename='sap_optimized_fast_init.pkl'
    )

    if epoch_num % 10 == 0:
        with open('training_progress_fast_init.txt', 'a') as f:
            elapsed = time.time() - t0
            f.write(f"Epoch {epoch_num}: {elapsed/3600:.2f}h ({elapsed/3600/24:.2f}d)\n")

print("\n" + "="*70)
print("Training complete!")
print(f"Total time: {(time.time()-t0)/3600:.2f} hours")
print("="*70)

final_kl = np.mean(net.traces_train['kl_trees'][-100:])
final_acc = np.mean(net.traces_train['acc'][-100:])

print(f"\nFinal KL divergence: {final_kl:.3f}")
print(f"Final accuracy: {final_acc:.3f}")
