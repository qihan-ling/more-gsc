#!/usr/bin/env python3
"""
Minimal test: Run just ONE trial to measure actual speed
This will help diagnose if the issue is:
1. Setup/initialization taking forever
2. Each trial taking forever
3. Something else
"""
import only_gscnet_speedup_sap as gsc
import numpy as np
import time

print("="*70)
print("SINGLE TRIAL SPEED TEST")
print("="*70)

np.random.seed(41)

print("\nLoading grammar...")
t0 = time.time()
with open('collapsed_filtered_sm5.grammar', 'r') as f:
    PCFG_sap = f.read()
print(f"  Loaded in {time.time()-t0:.2f}s")

ROOT = 'S'
MAXLEN = 24

print("\nInitializing HarmonicGrammar...")
t0 = time.time()
hg = gsc.HarmonicGrammar(pcfg=PCFG_sap, root=ROOT, max_sent_len=MAXLEN)
print(f"  Initialized in {time.time()-t0:.2f}s")
print(f"  Fillers: {len(hg.filler_names)}")

sim = hg.get_simlist(dp=0.0)

net_opts = {
    'use_jax': False,
    'T_init': 0.01,
    'q_max': 15.0,
    'q_init': 0.0,
    'dt_init': 0.02,  # FAST setting
    'm': 30,
    'use_runC': True,
    'ep_method': 'integration',
    'use_sparse_wc': True,
}

encodings = {
    'similarity': sim,
    'dim_f': 150,
    'dim_r': 60,
}

print("\nInitializing GscNet...")
t0 = time.time()
net = gsc.GscNet(hg=hg, encodings=encodings, opts=net_opts, seed=1024)
print(f"  Initialized in {time.time()-t0:.2f}s")
print(f"  WC shape: {net.WC.shape}")
if hasattr(net.WC, 'nnz'):
    print(f"  WC non-zeros: {net.WC.nnz:,}")

print("\nGenerating corpus...")
t0 = time.time()
net.generate_corpus(use_freq=True, nsamples=5000)
print(f"  Generated in {time.time()-t0:.2f}s")

train_opts = {
    'lrate': 0.1,
    'num_trials': 1,  # JUST ONE TRIAL
    'ema_stat_weight': 0.0,
    'trace_varnames': ['kl_trees', 'kl_treelets', 'prob_sent', 'acc'],
    'report_cycle': 1,
    'init_noise_mag': 0.02,
    'average_weight': False,
    'average_filler_bias': False,
}

print("\nInitializing training...")
t0 = time.time()
net.initialize(train_opts=train_opts)
print(f"  Initialized in {time.time()-t0:.2f}s")

print("\n" + "="*70)
print("RUNNING SINGLE TRIAL TEST")
print("="*70)
print(f"Configuration:")
print(f"  dt_init: {net.opts['dt_init']}")
print(f"  q_max: {net.opts['q_max']}")
print(f"  q_rate: {net.opts['q_rate']}")
print(f"  Expected integration steps: {net.opts['q_max'] / net.opts['q_rate'] / net.opts['dt_init']:.0f}")
print("="*70)

print("\nStarting SINGLE trial training (1 epoch, 1 trial)...")
print("This will help us measure:")
print("  - Time per trial")
print("  - Whether code completes at all")
print("")

t_start = time.time()
try:
    net.train2(
        train_opts={'num_epochs': 1},
        savefilename=None  # Don't save
    )
    t_elapsed = time.time() - t_start

    print("\n" + "="*70)
    print("SUCCESS!")
    print("="*70)
    print(f"Single trial completed in: {t_elapsed:.1f} seconds ({t_elapsed/60:.2f} minutes)")
    print(f"")
    print(f"Extrapolations:")
    print(f"  50 trials (FAST mode): {t_elapsed * 50 / 60:.1f} minutes")
    print(f"  200 trials (original): {t_elapsed * 200 / 60:.1f} minutes")
    print(f"  ")
    print(f"  FAST mode (50 trials × 500 epochs):")
    print(f"    Total time: {t_elapsed * 50 * 500 / 3600:.1f} hours ({t_elapsed * 50 * 500 / 3600 / 24:.1f} days)")
    print(f"  ")
    print(f"  Original (200 trials × 500 epochs):")
    print(f"    Total time: {t_elapsed * 200 * 500 / 3600:.1f} hours ({t_elapsed * 200 * 500 / 3600 / 24:.1f} days)")

except Exception as e:
    print("\n" + "="*70)
    print("ERROR!")
    print("="*70)
    print(f"Exception: {e}")
    import traceback
    traceback.print_exc()

print("\n" + "="*70)
print("Test complete")
print("="*70)
