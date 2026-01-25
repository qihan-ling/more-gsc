"""
Resume training from checkpoint.

Usage:
    python3 sap_resume_training.py [checkpoint_file]

Example:
    python3 sap_resume_training.py sap_checkpoint_epoch_0010.pkl
    python3 sap_resume_training.py sap_checkpoint_latest.pkl
"""

import matplotlib.pyplot as plt
import only_gscnet_speedup_sap as gsc
import numpy as np
import time
import sys

# Import memory-efficient save
sys.path.insert(0, '.')
from save_efficient import save_model_efficient, load_model_efficient

# ============================================================================
# Load checkpoint
# ============================================================================

checkpoint_file = sys.argv[1] if len(sys.argv) > 1 else 'sap_checkpoint_latest.pkl'

print("="*70)
print("RESUMING TRAINING FROM CHECKPOINT")
print("="*70)
print(f"Checkpoint file: {checkpoint_file}")
print("="*70 + "\n")

# Load checkpoint metadata to get config
import pickle
with open(checkpoint_file, 'rb') as f:
    state = pickle.load(f)

print("Checkpoint info:")
print(f"  Epoch: {state['epoch_num']}")
print(f"  WC nnz: {state['WC'].nnz:,}")
print(f"  Compression: dim_f={state['encodings_config']['dim_f']}, dim_r={state['encodings_config']['dim_r']}")

# ============================================================================
# Reconstruct network (MUST match original configuration)
# ============================================================================

print("\nReconstructing network structure...")

# Use SAME random seed as original
np.random.seed(41)

# Load grammar
with open('collapsed_filtered_sm5.grammar', 'r') as f:
    PCFG_sap = f.read()

ROOT = state['hg_root']
MAXLEN = state['hg_maxlen']

hg = gsc.HarmonicGrammar(pcfg=PCFG_sap, root=ROOT, max_sent_len=MAXLEN)
sim = hg.get_simlist(dp=state['encodings_config']['similarity_dp'])

# Reconstruct with SAME configuration
net_opts = state['opts']

encodings = {
    'similarity': sim,
    'dim_f': state['encodings_config']['dim_f'],
    'dim_r': state['encodings_config']['dim_r'],
}

# Recreate network
net = gsc.GscNet(hg=hg, encodings=encodings, opts=net_opts, seed=state['seed'])

print(f"  ✓ Network structure recreated")

# Regenerate corpus (needed for training)
net.generate_corpus(use_freq=True, nsamples=5000)
print(f"  ✓ Corpus regenerated")

# Initialize training options
train_opts = state['train_opts']
net.initialize(train_opts=train_opts)
print(f"  ✓ Training initialized")

# ============================================================================
# Load checkpoint state
# ============================================================================

print(f"\nLoading checkpoint...")
load_model_efficient(checkpoint_file, net)

print(f"\n✓ Checkpoint loaded successfully")
print(f"  Resuming from epoch {net.epoch_num}")

# ============================================================================
# Continue training
# ============================================================================

print("\n" + "="*70)
print("CONTINUING TRAINING")
print("="*70)

t0 = time.time()

n_epochs_total = 500
epochs_remaining = n_epochs_total - net.epoch_num
CHECKPOINT_INTERVAL = 2

print(f"Total target epochs: {n_epochs_total}")
print(f"Already completed: {net.epoch_num}")
print(f"Remaining: {epochs_remaining}")
print(f"Checkpoint interval: {CHECKPOINT_INTERVAL} epochs")
print("="*70 + "\n")

if epochs_remaining <= 0:
    print("Training already complete!")
    sys.exit(0)

for epoch_block in range(epochs_remaining // CHECKPOINT_INTERVAL):
    block_start_epoch = net.epoch_num + 1

    if (block_start_epoch - net.epoch_num) % 10 == 1:
        elapsed_time = time.time() - t0
        epochs_completed_this_session = epoch_block * CHECKPOINT_INTERVAL
        if epochs_completed_this_session > 0:
            time_per_epoch = elapsed_time / epochs_completed_this_session
            remaining_epochs = epochs_remaining - epochs_completed_this_session
            estimated_remaining = time_per_epoch * remaining_epochs

            print(f"\n{'='*70}")
            print(f"Progress - Epoch {block_start_epoch}/{n_epochs_total}")
            print(f"{'='*70}")
            print(f"  This session: {epochs_completed_this_session} epochs")
            print(f"  Elapsed this session: {elapsed_time/3600:.1f}h")
            print(f"  Per epoch: {time_per_epoch/3600:.1f}h")
            print(f"  Remaining: {remaining_epochs} epochs ({estimated_remaining/3600:.1f}h, {estimated_remaining/3600/24:.1f}d)")
            print(f"{'='*70}\n")

    # Train for CHECKPOINT_INTERVAL epochs
    net.train2(
        train_opts={'num_epochs': CHECKPOINT_INTERVAL},
        savefilename=None  # Save manually
    )

    # Save checkpoint
    current_epoch = net.epoch_num
    checkpoint_name = f'sap_checkpoint_epoch_{current_epoch:04d}.pkl'

    try:
        save_model_efficient(net, checkpoint_name)
        save_model_efficient(net, 'sap_checkpoint_latest.pkl')

        with open('training_progress_resumed.txt', 'a') as f:
            elapsed = time.time() - t0
            f.write(f"Epoch {current_epoch}: {elapsed/3600:.2f}h this session\n")

    except Exception as e:
        print(f"\n⚠️  WARNING: Checkpoint save failed: {e}")

print("\n" + "="*70)
print("Training complete!")
print(f"Session time: {(time.time()-t0)/3600:.2f}h ({(time.time()-t0)/3600/24:.2f}d)")
print("="*70)

final_kl = np.mean(net.traces_train['kl_trees'][-100:])
final_acc = np.mean(net.traces_train['acc'][-100:])
print(f"\nFinal KL divergence: {final_kl:.3f}")
print(f"Final accuracy: {final_acc:.3f}")

# Save final model
try:
    save_model_efficient(net, 'sap_final_model.pkl')
    print(f"\n✓ Final model saved")
except Exception as e:
    print(f"\n⚠️  Final save failed: {e}")
