"""
Memory-efficient save/load for large sparse models.

Add these functions to your training script to avoid OOM during checkpoint saves.
"""

import pickle
import numpy as np
from scipy import sparse


def save_model_efficient(net, filename):
    """
    Save only essential state to avoid OOM during pickling.

    Saves:
    - Training state (WC, bC, traces, epoch_num)
    - NOT: Large constant matrices (encodings, masks can be reconstructed)
    """
    print(f"Saving checkpoint to {filename}...")
    import time
    t0 = time.time()

    # Extract only what's needed to resume training
    state = {
        'WC': net.WC,  # Sparse matrix (already memory-efficient)
        'bC': net.bC,
        'epoch_num': net.epoch_num,
        'traces_train': net.traces_train,
        'train_opts': net.train_opts,
        'opts': net.opts,
        'seed': net.seed if hasattr(net, 'seed') else None,

        # Training state
        'hg_pcfg': net.hg.pcfg_str,  # PCFG string (small)
        'hg_root': net.hg.opts['root'],
        'hg_maxlen': net.hg.opts['max_sent_len'],

        # Encodings config (not the matrices themselves)
        'encodings_config': {
            'dim_f': net.dim_f,
            'dim_r': net.dim_r,
            'similarity': net.encodings.get('similarity', None) if isinstance(net.encodings, dict) else None,
        },

        # Optimizer state if using Adam
        'optim': net.optim if hasattr(net, 'optim') else None,
    }

    # Save with protocol 4 (more efficient for large arrays)
    with open(filename, 'wb') as f:
        pickle.dump(state, f, protocol=4)

    print(f"  Checkpoint saved in {time.time() - t0:.1f}s")

    # Report sizes
    import os
    size_mb = os.path.getsize(filename) / 1024 / 1024
    print(f"  File size: {size_mb:.1f} MB")


def load_model_efficient(filename, net=None):
    """
    Load checkpoint and resume training.

    If net is provided, update it in-place.
    If net is None, you need to reconstruct it first.
    """
    print(f"Loading checkpoint from {filename}...")

    with open(filename, 'rb') as f:
        state = pickle.load(f)

    if net is None:
        print("ERROR: Must provide net object to load into")
        print(
            "Reconstruct the network first, then call load_model_efficient(filename, net)")
        return state  # Return state dict so user can inspect it

    # Verify dimensions match
    saved_config = state.get('encodings_config', {})
    if saved_config:
        if saved_config.get('dim_f') != net.dim_f:
            print(f"  WARNING: dim_f mismatch! Saved: {saved_config.get('dim_f')}, Current: {net.dim_f}")
        if saved_config.get('dim_r') != net.dim_r:
            print(f"  WARNING: dim_r mismatch! Saved: {saved_config.get('dim_r')}, Current: {net.dim_r}")

    # Restore training state
    net.WC = state['WC']
    net.bC = state['bC']
    net.epoch_num = state.get('epoch_num', 0)
    
    # Restore traces (handle missing key gracefully)
    if 'traces_train' in state:
        net.traces_train = state['traces_train']
    
    # Restore train_opts if present
    if 'train_opts' in state and state['train_opts'] is not None:
        # Merge with existing train_opts to preserve any new options
        if hasattr(net, 'train_opts') and net.train_opts is not None:
            net.train_opts.update(state['train_opts'])
        else:
            net.train_opts = state['train_opts']

    # Restore optimizer state
    if 'optim' in state and state['optim'] is not None:
        net.optim = state['optim']

    # Print status
    print(f"  ✓ Checkpoint loaded")
    print(f"    Resuming from epoch {net.epoch_num}")
    
    # Handle both sparse and dense WC
    if hasattr(net.WC, 'nnz'):
        print(f"    WC nnz: {net.WC.nnz:,}")
    else:
        print(f"    WC shape: {net.WC.shape}")

    return net


# ============================================================================
# Example Usage in Training Script
# ============================================================================

# AT THE TOP OF YOUR SCRIPT, add these imports:
# import sys
# sys.path.insert(0, '.')
# from save_efficient import save_model_efficient, load_model_efficient

# REPLACE save_model(net, filename) calls with:
# save_model_efficient(net, filename)

# TO RESUME FROM CHECKPOINT:
"""
# 1. Recreate network structure (MUST match original)
net = gsc.GscNet(hg=hg, encodings=encodings, opts=net_opts, seed=1024)
net.generate_corpus(use_freq=True, nsamples=5000)
net.initialize(train_opts=train_opts)

# 2. Load checkpoint
load_model_efficient('sap_minimal_compression.pkl', net)

# 3. Continue training
net.train2(
    train_opts={'num_epochs': remaining_epochs},
    savefilename='sap_minimal_compression.pkl'
)
"""
