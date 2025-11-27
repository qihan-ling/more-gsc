"""
Diagnostic script to debug why Grammar 1 training isn't learning
"""
import numpy as np
import sys

# Import the GSC module
sys.path.insert(0, '/home/user/more-gsc')
import only_gscnet_speedup_sap as gsc

# ============================================================================
# Setup Grammar 1 (same as cho_grammar1.py)
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

hg = gsc.HarmonicGrammar(pcfg=PCFG_G1, root=ROOT, max_sent_len=MAXLEN)
sim = hg.get_simlist(dp=0.0)

net_opts = {
    'T_init': 0.01,
    'q_max': 15.0,
    'q_init': 0.0,
    'dt_init': 0.005,
    'm': 30,
    'use_runC': True,
}

net = gsc.GscNet(hg=hg, encodings={'similarity': sim},
                 opts=net_opts, seed=1024)
net.generate_corpus(use_freq=True)

train_opts = {
    'lrate': 0.1,
    'num_trials': 4,
    'ema_stat_weight': 0.0,
    'trace_varnames': ['kl_trees', 'kl_treelets', 'prob_sent', 'acc'],
    'report_cycle': 1,  # Report every epoch for debugging
    'init_noise_mag': 0.02,
    'average_weight': False,
    'average_filler_bias': False,
}

net.initialize(train_opts=train_opts)

print("="*70)
print("DIAGNOSTIC INFORMATION")
print("="*70)

# Check 1: Training options
print("\n1. Training Options:")
print(f"   Learning rate: {net.train_opts['lrate']}")
print(f"   Optimizer: {net.train_opts['optimizer']}")
print(f"   Update weights: {net.train_opts['update_w']}")
print(f"   bias1_only: {net.train_opts['bias1_only']}")
print(f"   bias2_only: {net.train_opts.get('bias2_only', False)}")
print(f"   Coefficients: {net.train_opts['coef']}")
print(f"   update_gram_only: {net.train_opts['update_gram_only']}")

# Check 2: mask0
print("\n2. Checking mask0:")
mask0 = net.train_opts['mask0']
if hasattr(net, 'use_sparse') and net.use_sparse:
    print(f"   mask0 is sparse: shape={mask0.shape}, nnz={mask0.nnz}")
    print(f"   Non-zero entries: {mask0.nnz}")
    print(f"   Total possible entries: {mask0.shape[0] * mask0.shape[1]}")
    if mask0.nnz == 0:
        print("   ⚠️  WARNING: mask0 has NO non-zero entries! This will prevent learning.")
else:
    print(f"   mask0 shape: {mask0.shape}")
    print(f"   Non-zero entries: {np.count_nonzero(mask0)}")
    print(f"   Total entries: {mask0.size}")
    print(f"   Min/Max: {mask0.min():.3f} / {mask0.max():.3f}")
    if np.count_nonzero(mask0) == 0:
        print("   ⚠️  WARNING: mask0 is all zeros! This will prevent learning.")

# Check 3: Initial parameters
print("\n3. Initial Parameters:")
if hasattr(net, 'use_sparse') and net.use_sparse:
    print(f"   WC is sparse: shape={net.WC.shape}, nnz={net.WC.nnz}")
else:
    print(f"   WC shape: {net.WC.shape}")
    print(f"   WC non-zero: {np.count_nonzero(net.WC)}")
print(f"   bC shape: {net.bC.shape}")
print(f"   bC min/max: {net.bC.min():.3f} / {net.bC.max():.3f}")

# Store initial parameters for comparison
if hasattr(net, 'use_sparse') and net.use_sparse:
    WC_initial = net.WC.copy()
else:
    WC_initial = net.WC.copy()
bC_initial = net.bC.copy()

# Check 4: Run one epoch and examine gradients
print("\n4. Running one training epoch with detailed diagnostics:")
print("-" * 70)

# Temporarily modify the train2 method to capture gradients
# We'll manually run one iteration of the training loop

net.epoch_num = 0
prefix_list = [[]]
prefix_weights = np.array([1.0])

# Initialize gradients
if net.use_jax:
    import jax.numpy as jnp
    dWC = jnp.zeros(net.WC.shape, dtype=jnp.float32)
    dbC = jnp.zeros(net.bC.shape, dtype=jnp.float32)
else:
    if hasattr(net, 'use_sparse') and net.use_sparse:
        from scipy import sparse
        dWC = sparse.dok_matrix(net.WC.shape, dtype=np.float64)
    else:
        dWC = np.zeros(net.WC.shape)
    dbC = np.zeros(net.bC.shape)

print("   Computing gradients...")

# Run through one training iteration
for pi, prefix in enumerate(prefix_list):
    if prefix_weights[pi] > 0:
        if len(prefix) > 0:
            scale_dWC = net.train_opts['scale_dWC_parser']
            prefix_bnames = [ftype + net.hg.opts['bsep'] + '(1,{})'.format(wi + 1)
                             for wi, ftype in enumerate(prefix)]
        else:
            scale_dWC = 1.0
            prefix_bnames = []

        stat_P = net.get_corpus_stat(net.subset_corpus(prefix_bnames))
        stat_Q = net.estimate_prob_inc(prefix=prefix, num_trials=net.train_opts['num_trials'])

        if isinstance(stat_Q, tuple):
            stat_Q, actC_set = stat_Q

        net.clear_input()
        if len(prefix_bnames) > 0:
            prefix_bnames = prefix_bnames[-1]
            net.set_input(prefix_bnames)

        extC_token = net.extC.astype(bool).astype(int)
        kl_curr, xent_curr, err, err_log = net.cost(stat_P, stat_Q)

        print(f"\n   Iteration {pi}:")
        print(f"     KL divergence: {kl_curr}")
        print(f"     Error keys: {list(err.keys())}")
        for key in err.keys():
            if isinstance(err[key], dict):
                print(f"       err['{key}']: {len(err[key])} entries, values range: {min(err[key].values()) if err[key] else 0:.3e} to {max(err[key].values()) if err[key] else 0:.3e}")
            else:
                print(f"       err['{key}']: {err[key]}")

        # Compute gradients
        dWC_curr, destr_curr, dq_curr, dbC_curr = net.cost_grad(err, extC_token)

        print(f"\n     Gradient statistics:")
        if hasattr(net, 'use_sparse') and net.use_sparse:
            print(f"       dWC_curr: nnz={dWC_curr.nnz}, max={abs(dWC_curr).max() if dWC_curr.nnz > 0 else 0:.3e}")
        else:
            print(f"       dWC_curr: shape={dWC_curr.shape}, min/max={dWC_curr.min():.3e} / {dWC_curr.max():.3e}")
            print(f"       dWC_curr non-zero: {np.count_nonzero(dWC_curr)}")
        print(f"       dbC_curr: shape={dbC_curr.shape}, min/max={dbC_curr.min():.3e} / {dbC_curr.max():.3e}")
        print(f"       dbC_curr non-zero: {np.count_nonzero(dbC_curr)}")

        dWC += dWC_curr * scale_dWC * prefix_weights[pi]
        dbC += dbC_curr * prefix_weights[pi]

# Check final accumulated gradients
print(f"\n   Final accumulated gradients:")
if hasattr(net, 'use_sparse') and net.use_sparse:
    print(f"     dWC: nnz={dWC.nnz}, max={abs(dWC).max() if dWC.nnz > 0 else 0:.3e}")
    if dWC.nnz == 0:
        print("     ⚠️  WARNING: dWC has no non-zero entries!")
else:
    print(f"     dWC non-zero: {np.count_nonzero(dWC)}, min/max={dWC.min():.3e} / {dWC.max():.3e}")
    if np.count_nonzero(dWC) == 0:
        print("     ⚠️  WARNING: dWC is all zeros!")
print(f"     dbC non-zero: {np.count_nonzero(dbC)}, min/max={dbC.min():.3e} / {dbC.max():.3e}")
if np.count_nonzero(dbC) == 0:
    print("     ⚠️  WARNING: dbC is all zeros!")

print("\n" + "="*70)
print("RUNNING 2 TRAINING EPOCHS")
print("="*70)

# Now run actual training
net.train2(train_opts={'num_epochs': 2}, savefilename=None)

# Check if parameters changed
print("\n5. Parameter Changes After 2 Epochs:")
if hasattr(net, 'use_sparse') and net.use_sparse:
    param_diff = (net.WC - WC_initial)
    print(f"   WC changed: {param_diff.nnz} entries")
    if param_diff.nnz > 0:
        print(f"   WC max change: {abs(param_diff).max():.3e}")
    else:
        print("   ⚠️  WARNING: WC did not change at all!")
else:
    diff_WC = net.WC - WC_initial
    print(f"   WC max abs change: {np.max(np.abs(diff_WC)):.3e}")
    print(f"   WC entries that changed: {np.count_nonzero(diff_WC)}")
    if np.max(np.abs(diff_WC)) == 0:
        print("   ⚠️  WARNING: WC did not change at all!")

diff_bC = net.bC - bC_initial
print(f"   bC max abs change: {np.max(np.abs(diff_bC)):.3e}")
print(f"   bC entries that changed: {np.count_nonzero(diff_bC)}")
if np.max(np.abs(diff_bC)) == 0:
    print("   ⚠️  WARNING: bC did not change at all!")

print("\n" + "="*70)
print("DIAGNOSIS COMPLETE")
print("="*70)
