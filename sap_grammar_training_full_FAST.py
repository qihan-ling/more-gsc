import matplotlib.pyplot as plt
import only_gscnet_speedup_sap as gsc
import numpy as np
import time

# TEST: Using seed=41 to check if different seeds produce different results
np.random.seed(41)
print("Global random seed set to 41 for testing")

t0 = time.time()  # Start timing
with open('collapsed_filtered_sm5.grammar', 'r') as f:
    PCFG_sap = f.read()

ROOT = 'S'
MAXLEN = 24

# ============================================================================
# Initialize network with paper's specifications
# ============================================================================

hg = gsc.HarmonicGrammar(pcfg=PCFG_sap, root=ROOT, max_sent_len=MAXLEN)

# Display fillers (should have 27 fillers × 15 roles = 405 units)
print(f"Filler names: {hg.filler_names}")
print(f"Number of fillers: {len(hg.filler_names)}")

# Set all filler similarities to 0 (linear independence)
sim = hg.get_simlist(dp=0.0)

# ============================================================================
# CONFIGURATION: Toggle these to test different modes
# ============================================================================
USE_SPARSE = True      # True = sparse WC matrix, False = dense
USE_COMPRESSED = True  # True = compressed encodings, False = full dimension
# ============================================================================

# Network options - OPTIMIZED FOR SPEED
net_opts = {
    'use_jax': False,  # Sparse only supported on CPU currently
    'T_init': 0.01,      # computational temperature
    'q_max': 15.0,       # maximum commitment
    'q_init': 0.0,       # initial commitment (FIXED: was 'q_0')
    'dt_init': 0.02,     # INCREASED from 0.005 for 4x speedup
    'm': 30,             # resource constraint (Hq1 strength)
    'use_runC': True,    # use C implementation for speed
    'ep_method': 'integration',
}
if USE_SPARSE:
    net_opts['use_sparse_wc'] = True

encodings = {
    'similarity': sim,
}
if USE_COMPRESSED:
    encodings['dim_f'] = 150  # Compressed filler encoding
    encodings['dim_r'] = 60   # Compressed role encoding

# Initialize network
net = gsc.GscNet(hg=hg, encodings=encodings,
                 opts=net_opts, seed=1024)

# ============================================================================
# DIAGNOSTIC: Verify what mode we're running in
# ============================================================================
print("\n" + "="*70)
print("MODE VERIFICATION:")
print("="*70)
print(f"  use_sparse: {getattr(net, 'use_sparse', False)}")
print(f"  WC type: {type(net.WC).__module__}.{type(net.WC).__name__}")
print(f"  WC shape: {net.WC.shape}")
if hasattr(net.WC, 'nnz'):
    print(f"  WC non-zeros: {net.WC.nnz:,} ({100*net.WC.nnz/net.WC.shape[0]/net.WC.shape[1]:.4f}% fill)")
print(f"  dim_f used: {net.dim_f if hasattr(net, 'dim_f') else 'N/A (full)'}")
print(f"  dim_r used: {net.dim_r if hasattr(net, 'dim_r') else 'N/A (full)'}")
print(f"  num_fillers: {net.num_fillers}")
print(f"  num_roles: {net.num_roles}")
print(f"  num_bindings: {net.num_bindings}")
print("="*70 + "\n")

net.generate_corpus(use_freq=True, nsamples=5000)

# Display target probabilities
print("\n" + "="*70)
print("Target sentence probabilities (first 10):")
for si, sent in enumerate(net.corpus['sentence'][:10]):
    sent_str = ' '.join([bname.split('/')[0] for bname in sent])
    prob = net.corpus['prob_sent'][si]
    print(f"Sentence {si}: p = {prob:.4f} ({sent_str})")

# ============================================================================
# Training setup - OPTIMIZED FOR SPEED
# ============================================================================

train_opts = {
    'lrate': 0.1,
    'num_trials': 50,              # REDUCED from 200 for 4x speedup
    'ema_stat_weight': 0.0,
    'trace_varnames': ['kl_trees', 'kl_treelets', 'prob_sent', 'acc'],
    'report_cycle': 5,             # More frequent reporting (from 10)
    'init_noise_mag': 0.02,
    'average_weight': False,
    'average_filler_bias': False,
}

net.initialize(train_opts=train_opts)

# Diagnostic check
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

# ============================================================================
# DEBUG: Print WC statistics before training
# ============================================================================
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
# Training loop - OPTIMIZED
# ============================================================================

print("\n" + "="*70)
print("Training Grammar 1 (OPTIMIZED - FAST MODE)")
print("="*70)
print(f"Optimizations applied:")
print(f"  - Reduced num_trials: 200 → 50 (4x speedup)")
print(f"  - Increased dt: 0.005 → 0.02 (4x speedup)")
print(f"  - Combined speedup: ~16x faster")
print(f"  - Estimated time per epoch: ~1-3 hours (vs 17-50 hours)")
print("="*70 + "\n")

# Train for sufficient epochs to reach convergence
n_epochs = 500

# Save checkpoints less frequently
CHECKPOINT_INTERVAL = 50  # Save every 50 epochs instead of every 5
for epoch_block in range(n_epochs // CHECKPOINT_INTERVAL):
    print(f"\n{'='*70}")
    print(f"Training epochs {epoch_block*CHECKPOINT_INTERVAL + 1} to {(epoch_block+1)*CHECKPOINT_INTERVAL}")
    print(f"{'='*70}")
    net.train2(
        train_opts={'num_epochs': CHECKPOINT_INTERVAL},
        savefilename='sap_1k_model_sparse_FAST.pkl'
    )
    print(f"\nCheckpoint saved: epoch {(epoch_block+1)*CHECKPOINT_INTERVAL}")

print("\n" + "="*70)
print("Training complete!")
print(f"Total time: {(time.time()-t0)/3600:.2f} hours")
print("="*70)

# Calculate final statistics (last 100 updates)
final_kl = np.mean(net.traces_train['kl_trees'][-100:])
final_kl_sd = np.std(net.traces_train['kl_trees'][-100:])
final_acc = np.mean(net.traces_train['acc'][-100:])
final_acc_sd = np.std(net.traces_train['acc'][-100:])

print(f"\nFinal KL divergence: {final_kl:.3f} (SD = {final_kl_sd:.3f})")
print(f"Final production accuracy: {final_acc:.3f} (SD = {final_acc_sd:.3f})")
