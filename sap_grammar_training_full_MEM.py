from save_load_model_efficiently import save_model_efficient, load_model_efficient
import matplotlib.pyplot as plt
import only_gscnet_speedup_sap as gsc
import numpy as np
import time
import sys

# Import memory-efficient save
sys.path.insert(0, '.')

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
# CONFIGURATION - MEMORY-SAFE
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
    'dtype': 'float32',  # Use float32 for 50% memory savings vs float64
}
if USE_SPARSE:
    net_opts['use_sparse_wc'] = True

encodings = {
    'similarity': sim,
    'dim_f': 150,
    'dim_r': 60,
}

print("\n" + "="*70)
print("MEMORY-SAFE TRAINING")
print("="*70)
print(f"Features:")
print(f"  - float32 precision (50% memory savings vs float64)")
print(f"  - Efficient checkpointing (avoids OOM during save)")
print(f"  - Saves every 2 epochs")
print(f"  - Can resume from checkpoint if interrupted")
print("="*70 + "\n")

net = gsc.GscNet(hg=hg, encodings=encodings, opts=net_opts, seed=1024)

print(f"  WC shape: {net.WC.shape}")
if hasattr(net.WC, 'nnz'):
    print(f"  WC non-zeros: {net.WC.nnz:,}")
print(f"  Compression: dim_f={net.dim_f}, dim_r={net.dim_r}\n")

net.generate_corpus(use_freq=True, nsamples=5000)

print("\n" + "="*70)
print("Target sentence probabilities (first 10):")
for si, sent in enumerate(net.corpus['sentence'][:10]):
    sent_str = ' '.join([bname.split('/')[0] for bname in sent])
    prob = net.corpus['prob_sent'][si]
    print(f"Sentence {si}: p = {prob:.4f} ({sent_str})")

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
print("Initializing network...")
print("="*70)
init_start = time.time()

net.initialize(train_opts=train_opts)

init_time = time.time() - init_start
print(f"\n✓ Initialization complete: {init_time:.1f}s ({init_time/60:.1f}min)")

mask0 = net.train_opts['mask0']
if hasattr(net, 'use_sparse') and net.use_sparse:
    print(f"\nmask0: {mask0.nnz:,} non-zero entries")

print("\n" + "="*40)
print("=== WC Statistics BEFORE Training ===")
if hasattr(net, 'use_sparse') and net.use_sparse:
    print(f"  WC type: sparse, nnz={net.WC.nnz:,}")
    print(f"  WC sum: {net.WC.sum():.6f}")
print("=" * 40)

# ============================================================================
# Training loop with MEMORY-EFFICIENT checkpointing
# ============================================================================

print("\n" + "="*70)
print("Starting Training")
print("="*70)
print(f"Configuration:")
print(f"  Total epochs: 500")
print(f"  Checkpoint interval: 2 epochs (memory-safe)")
print(f"  Trials per epoch: 30")
print(f"  dt: 0.04, q_max: 12.0")
print(f"")
print(f"Expected time: ~150 days")
print("="*70 + "\n")

n_epochs = 500
CHECKPOINT_INTERVAL = 2  # Save every 2 epochs

for epoch_block in range(n_epochs // CHECKPOINT_INTERVAL):
    block_start_epoch = epoch_block * CHECKPOINT_INTERVAL + 1

    if block_start_epoch % 10 == 1 or block_start_epoch == 1:
        elapsed_time = time.time() - t0
        epochs_completed = net.epoch_num if hasattr(net, 'epoch_num') else 0
        if epochs_completed > 0:
            time_per_epoch = elapsed_time / epochs_completed
            remaining_epochs = n_epochs - epochs_completed
            estimated_remaining = time_per_epoch * remaining_epochs

            print(f"\n{'='*70}")
            print(f"Progress - Starting Epoch {block_start_epoch}/{n_epochs}")
            print(f"{'='*70}")
            print(f"  Completed: {epochs_completed}")
            print(
                f"  Elapsed: {elapsed_time/3600:.1f}h ({elapsed_time/3600/24:.1f}d)")
            if epochs_completed > 0:
                print(f"  Per epoch: {time_per_epoch/3600:.1f}h")
                print(
                    f"  Remaining: {estimated_remaining/3600:.1f}h ({estimated_remaining/3600/24:.1f}d)")
            print(f"{'='*70}\n")

    # Train for CHECKPOINT_INTERVAL epochs WITHOUT saving in train2()
    # (save manually after to use efficient method)
    net.train2(
        train_opts={'num_epochs': CHECKPOINT_INTERVAL},
        savefilename=None  # Don't save in train2() - we'll do it manually
    )

    # MEMORY-EFFICIENT checkpoint save
    current_epoch = net.epoch_num
    checkpoint_name = f'sap_checkpoint_epoch_{current_epoch:04d}.pkl'

    try:
        save_model_efficient(net, checkpoint_name)

        # Also maintain a "latest" checkpoint
        # save_model_efficient(net, 'sap_checkpoint_latest.pkl')

        # Log progress
        with open('training_progress_memory_safe.txt', 'a') as f:
            elapsed = time.time() - t0
            f.write(
                f"Epoch {current_epoch}: {elapsed/3600:.2f}h ({elapsed/3600/24:.2f}d)\n")

    except Exception as e:
        print(f"\n⚠️  WARNING: Checkpoint save failed: {e}")
        print(f"  Continuing training without checkpoint...")
        # Continue training even if save fails

print("\n" + "="*70)
print("Training complete!")
print(
    f"Total time: {(time.time()-t0)/3600:.2f}h ({(time.time()-t0)/3600/24:.2f}d)")
print("="*70)

final_kl = np.mean(net.traces_train['kl_trees'][-100:])
final_acc = np.mean(net.traces_train['acc'][-100:])
print(f"\nFinal KL divergence: {final_kl:.3f}")
print(f"Final accuracy: {final_acc:.3f}")

# Save final model
try:
    save_model_efficient(net, 'sap_final_model.pkl')
    print(f"\n✓ Final model saved to sap_final_model.pkl")
except Exception as e:
    print(f"\n⚠️  WARNING: Final save failed: {e}")
