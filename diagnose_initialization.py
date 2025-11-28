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
print("QUICK DIAGNOSTIC - NETWORK SIZE")
print("="*70)

# Initialize network
hg = gsc.HarmonicGrammar(pcfg=PCFG_G1, root=ROOT, max_sent_len=MAXLEN)
print(f"Number of fillers: {len(hg.filler_names)}")
print(f"Number of roles: {len(hg.role_names)}")

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

print("\nInitializing network (before training initialization)...")
t0 = time.time()
net = gsc.GscNet(hg=hg, encodings=encodings, opts=net_opts, seed=1024)
print(f"Time: {time.time() - t0:.2f}s")

print(f"\nNetwork dimensions:")
print(f"  num_bindings: {net.num_bindings}")
print(f"  WC shape: {net.WC.shape}")
print(f"  WC size (elements): {net.WC.shape[0] * net.WC.shape[1]:,}")
print(f"  WC memory (dense): {net.WC.nbytes / 1e6:.1f} MB")
print(f"  Using sparse: {hasattr(net, 'use_sparse') and net.use_sparse}")

print("\nGenerating small corpus for testing...")
net.generate_corpus(use_freq=True, nsamples=100)  # Small corpus for speed

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

print("\n" + "="*70)
print("PROFILING INITIALIZATION")
print("="*70)

# Profile just the get_mask0 call
print("\nTesting get_mask0() directly...")
print("WARNING: This may take several minutes if there's a problem\n")

t0 = time.time()
print("Calling net.initialize()...")
net.initialize(train_opts=train_opts)
t_init = time.time() - t0

print(f"\nInitialization time: {t_init:.2f}s")

if t_init > 10:
    print(f"\n❌ PROBLEM DETECTED!")
    print(f"   Initialization took {t_init:.0f} seconds (should be < 1s)")
    print(f"   The get_mask0() function is the bottleneck")
    print(f"\n   Likely cause:")
    print(f"   - Dense mask0 construction using np.ix_() is inefficient")
    print(f"   - With {net.num_bindings} bindings, this creates huge temporary arrays")
elif t_init > 1:
    print(f"\n⚠️  Initialization is slow ({t_init:.1f}s)")
    print(f"   Should be < 1s for a small grammar")
else:
    print(f"\n✓ Initialization is fast")

print("\n" + "="*70)
print("RECOMMENDATION")
print("="*70)

if t_init > 10:
    print("The dense get_mask0() implementation is too slow.")
    print("We need to optimize it using vectorized operations instead of np.ix_()")
    print("\nEstimated training time with current performance:")
    print(f"  Just initialization: {t_init:.0f}s per run")
    print(f"  This alone makes training impractical")
