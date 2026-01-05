#!/usr/bin/env python3
"""
ULTRA-FAST training mode for initial testing
This uses very aggressive optimizations to complete quickly
"""
import matplotlib.pyplot as plt
import only_gscnet_speedup_sap as gsc
import numpy as np
import time

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

# ULTRA-FAST configuration
USE_SPARSE = True
USE_COMPRESSED = True

net_opts = {
    'use_jax': False,
    'T_init': 0.01,
    'q_max': 15.0,
    'q_init': 0.0,
    'dt_init': 0.05,     # VERY LARGE: 0.005 -> 0.05 (10x faster)
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
print("ULTRA-FAST MODE VERIFICATION:")
print("="*70)
print(f"  use_sparse: {getattr(net, 'use_sparse', False)}")
print(f"  WC shape: {net.WC.shape}")
if hasattr(net.WC, 'nnz'):
    print(f"  WC non-zeros: {net.WC.nnz:,}")
print(f"  dt_init: {net.opts['dt_init']} (10x larger than original)")
print("="*70 + "\n")

net.generate_corpus(use_freq=True, nsamples=5000)

# ULTRA-FAST training setup
train_opts = {
    'lrate': 0.1,
    'num_trials': 20,              # VERY LOW: 200 -> 20 (10x faster)
    'ema_stat_weight': 0.0,
    'trace_varnames': ['kl_trees', 'kl_treelets', 'prob_sent', 'acc'],
    'report_cycle': 2,             # Report every 2 iterations
    'init_noise_mag': 0.02,
    'average_weight': False,
    'average_filler_bias': False,
}

net.initialize(train_opts=train_opts)

print("\n" + "="*70)
print("ULTRA-FAST MODE TRAINING")
print("="*70)
print(f"Optimizations (vs original):")
print(f"  num_trials: 200 -> 20 (10x speedup)")
print(f"  dt_init: 0.005 -> 0.05 (10x speedup)")
print(f"  Combined: ~100x faster")
print(f"")
print(f"Estimated time:")
print(f"  Per epoch: ~5-15 minutes (vs 17-50 hours)")
print(f"  For 100 epochs: ~8-25 hours (vs 70-200 days)")
print(f"")
print(f"NOTE: This is for TESTING ONLY to verify code is working")
print(f"      Results may be less accurate due to aggressive optimization")
print("="*70 + "\n")

# Train for just 100 epochs to test
n_epochs = 100

for epoch_block in range(n_epochs // 10):
    print(f"\n{'='*70}")
    print(f"Training epochs {epoch_block*10 + 1} to {(epoch_block+1)*10}")
    print(f"{'='*70}")

    block_start = time.time()
    net.train2(
        train_opts={'num_epochs': 10},
        savefilename='sap_test_ULTRAFAST.pkl'
    )
    block_time = time.time() - block_start

    print(f"\nBlock completed in {block_time/60:.1f} minutes")
    print(f"Checkpoint saved: epoch {(epoch_block+1)*10}")

    # Estimate remaining time
    epochs_done = (epoch_block + 1) * 10
    epochs_remaining = n_epochs - epochs_done
    time_per_epoch = block_time / 10
    estimated_remaining = time_per_epoch * epochs_remaining

    print(f"Estimated time remaining: {estimated_remaining/60:.1f} minutes")

print("\n" + "="*70)
print("ULTRA-FAST TEST COMPLETE!")
print(f"Total time: {(time.time()-t0)/60:.1f} minutes ({(time.time()-t0)/3600:.2f} hours)")
print("="*70)

# Show final stats
final_kl = np.mean(net.traces_train['kl_trees'][-20:])
final_acc = np.mean(net.traces_train['acc'][-20:])

print(f"\nFinal KL divergence: {final_kl:.3f}")
print(f"Final accuracy: {final_acc:.3f}")
print(f"\nIf this completed successfully, the code is working!")
print(f"You can now try the FAST or original settings for better accuracy.")
