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
# CONFIGURATION - MINIMAL COMPRESSION (memory-safe)
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

# CRITICAL: Use minimal compression to avoid OOM during mask initialization
# Larger compression (200×80) causes OOM at batch 21 during meshgrid operations
encodings = {
    'similarity': sim,
    'dim_f': 150,  # MINIMAL - avoid OOM (200 causes OOM kill)
    'dim_r': 60,   # MINIMAL - avoid OOM (80 causes OOM kill)
}

print("\n" + "="*70)
print("MEMORY-SAFE CONFIGURATION")
print("="*70)
print(f"Compression: dim_f=150, dim_r=60 (minimal to avoid OOM)")
print(f"Why minimal: Larger compression causes OOM during mask initialization")
print(f"  - 200×80: OOM kill at batch 21 (meshgrid too large)")
print(f"  - 150×60: Fits in 500GB RAM")
print("="*70 + "\n")

net = gsc.GscNet(hg=hg, encodings=encodings, opts=net_opts, seed=1024)

print(f"  WC shape: {net.WC.shape}")
if hasattr(net.WC, 'nnz'):
    print(f"  WC non-zeros: {net.WC.nnz:,}")
print(f"  dim_f: {net.dim_f}, dim_r: {net.dim_r}")
print(f"  num_bindings: {net.num_bindings:,}\n")

net.generate_corpus(use_freq=True, nsamples=5000)

print("\n" + "="*70)
print("Target sentence probabilities (first 10):")
for si, sent in enumerate(net.corpus['sentence'][:10]):
    sent_str = ' '.join([bname.split('/')[0] for bname in sent])
    prob = net.corpus['prob_sent'][si]
    print(f"Sentence {si}: p = {prob:.4f} ({sent_str})")

# Training parameters
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

print("\n" + "="*70)
print("Initializing network (this takes time for large grammars)...")
print("="*70)
init_start = time.time()

net.initialize(train_opts=train_opts)

init_time = time.time() - init_start
print(f"\n✓ Initialization complete: {init_time:.1f} seconds ({init_time/60:.1f} minutes)")

# Check mask
mask0 = net.train_opts['mask0']
if hasattr(net, 'use_sparse') and net.use_sparse:
    print(f"\nmask0 statistics:")
    print(f"  Non-zero entries: {mask0.nnz:,}")
    print(f"  Density: {100 * mask0.nnz / (mask0.shape[0] * mask0.shape[1]):.6f}%")

print("\n" + "="*40)
print("=== WC Statistics BEFORE Training ===")
if hasattr(net, 'use_sparse') and net.use_sparse:
    print(f"  WC type: sparse, nnz={net.WC.nnz:,}")
    print(f"  WC sum: {net.WC.sum():.6f}")
print("=" * 40)

# ============================================================================
# Training loop
# ============================================================================

print("\n" + "="*70)
print("Starting Training")
print("="*70)
print(f"Configuration:")
print(f"  Epochs: 500")
print(f"  Trials per epoch: 30")
print(f"  dt: 0.04")
print(f"  q_max: 12.0")
print(f"  Compression: 150×60 (minimal, memory-safe)")
print(f"")
print(f"Expected time: ~150 days")
print(f"Trade-off: Minimal compression = more approximation error")
print(f"           But necessary to fit in 500GB RAM")
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
            print(f"Progress - Epoch {epoch_num}/{n_epochs}")
            print(f"{'='*70}")
            print(f"  Completed: {epochs_completed}")
            print(f"  Elapsed: {elapsed_time/3600:.1f}h ({elapsed_time/3600/24:.1f}d)")
            print(f"  Per epoch: {time_per_epoch/3600:.1f}h")
            print(f"  Remaining: {estimated_remaining/3600:.1f}h ({estimated_remaining/3600/24:.1f}d)")
            print(f"{'='*70}\n")

    net.train2(
        train_opts={'num_epochs': CHECKPOINT_INTERVAL},
        savefilename='sap_minimal_compression.pkl'
    )

    if epoch_num % 10 == 0:
        with open('training_progress_minimal.txt', 'a') as f:
            elapsed = time.time() - t0
            f.write(f"Epoch {epoch_num}: {elapsed/3600:.2f}h ({elapsed/3600/24:.2f}d)\n")

print("\n" + "="*70)
print("Training complete!")
print(f"Total time: {(time.time()-t0)/3600:.2f} hours ({(time.time()-t0)/3600/24:.2f} days)")
print("="*70)

final_kl = np.mean(net.traces_train['kl_trees'][-100:])
final_acc = np.mean(net.traces_train['acc'][-100:])
print(f"\nFinal KL divergence: {final_kl:.3f}")
print(f"Final accuracy: {final_acc:.3f}")
