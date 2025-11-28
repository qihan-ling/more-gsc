import time
import only_gscnet_speedup_sap as gsc
import numpy as np

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
MAXLEN = 24

print("="*70)
print("ACTUAL TRAINING RUN PROFILING")
print("="*70)

# Initialize network
print("\n1. Initializing network...")
t0 = time.time()
hg = gsc.HarmonicGrammar(pcfg=PCFG_G1, root=ROOT, max_sent_len=MAXLEN)
sim = hg.get_simlist(dp=0.0)

net_opts = {
    'use_jax': False,
    'T_init': 0.01,
    'q_max': 15.0,
    'q_init': 0.0,
    'dt_init': 0.005,
    'm': 30,
    'use_runC': True,
    'ep_method': 'integration',
}

encodings = {'similarity': sim}

net = gsc.GscNet(hg=hg, encodings=encodings, opts=net_opts, seed=1024)
print(f"   Time: {time.time() - t0:.2f}s")
print(f"   - Number of bindings: {net.num_bindings}")
print(f"   - Using sparse: {hasattr(net, 'use_sparse') and net.use_sparse}")

print("\n2. Generating corpus...")
t0 = time.time()
net.generate_corpus(use_freq=True, nsamples=5000)
print(f"   Time: {time.time() - t0:.2f}s")
print(f"   - Unique sentences: {len(net.corpus['sentence'])}")

train_opts = {
    'lrate': 0.1,
    'num_trials': 4,
    'ema_stat_weight': 0.0,
    'trace_varnames': ['kl_trees', 'kl_treelets', 'prob_sent', 'acc'],
    'report_cycle': 10,
    'init_noise_mag': 0.02,
    'average_weight': False,
    'average_filler_bias': False,
}

print("\n3. Initializing training...")
t0 = time.time()
net.initialize(train_opts=train_opts)
print(f"   Time: {time.time() - t0:.2f}s")

print("\n" + "="*70)
print("RUNNING ACTUAL TRAINING (10 EPOCHS)")
print("="*70)

# Run 10 epochs and time each one
epoch_times = []
for i in range(10):
    print(f"\nEpoch {i+1}/10:")
    t_epoch = time.time()

    net.train2(
        train_opts={'num_epochs': 1},
        savefilename=None  # Don't save to avoid I/O overhead
    )

    epoch_dur = time.time() - t_epoch
    epoch_times.append(epoch_dur)
    print(f"  Epoch time: {epoch_dur:.2f}s")

print("\n" + "="*70)
print("TRAINING PERFORMANCE ANALYSIS")
print("="*70)

avg_epoch = np.mean(epoch_times)
std_epoch = np.std(epoch_times)
min_epoch = np.min(epoch_times)
max_epoch = np.max(epoch_times)

print(f"\nPer-Epoch Statistics:")
print(f"  Average:  {avg_epoch:.2f}s")
print(f"  Std Dev:  {std_epoch:.2f}s")
print(f"  Min:      {min_epoch:.2f}s")
print(f"  Max:      {max_epoch:.2f}s")

print(f"\nProjections:")
print(f"  Time for 140 epochs:  {avg_epoch * 140 / 60:.1f} minutes ({avg_epoch * 140 / 3600:.2f} hours)")
print(f"  Time for 1000 epochs: {avg_epoch * 1000 / 60:.1f} minutes ({avg_epoch * 1000 / 3600:.2f} hours)")

print("\n" + "="*70)
print("INTERPRETATION")
print("="*70)

if avg_epoch < 2:
    print("✓ FAST: < 2 seconds per epoch")
    print("  Expected for small grammar with dense matrices")
    print(f"  140 epochs should complete in ~{avg_epoch * 140 / 60:.0f} minutes")
elif avg_epoch < 10:
    print("⚠️  MODERATE: 2-10 seconds per epoch")
    print(f"  140 epochs will take ~{avg_epoch * 140 / 60:.0f} minutes")
    print("  This is slower than expected for a 10-rule grammar")
elif avg_epoch < 60:
    print("⚠️  SLOW: 10-60 seconds per epoch")
    print(f"  140 epochs will take ~{avg_epoch * 140 / 60:.0f} minutes ({avg_epoch * 140 / 3600:.1f} hours)")
    print("  There may be a performance issue")
else:
    print("❌ VERY SLOW: > 1 minute per epoch")
    print(f"  140 epochs will take {avg_epoch * 140 / 3600:.1f} hours")
    print("  Significant performance problem detected")
    print("\n  Possible causes:")
    print("  - Sparse matrices being used incorrectly")
    print("  - Inefficient dynamics loop (run_wrapup)")
    print("  - Large corpus size relative to grammar")

print("="*70)
