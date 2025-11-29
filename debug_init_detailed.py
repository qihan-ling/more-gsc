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
print("DETAILED INITIALIZATION PROFILING")
print("="*70)

# Initialize network
print("\n1. Creating network...")
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
print(f"   - num_bindings: {net.num_bindings}")
print(f"   - WC shape: {net.WC.shape}")

print("\n2. Generating corpus...")
t0 = time.time()
net.generate_corpus(use_freq=True, nsamples=100)  # Small corpus
print(f"   Time: {time.time() - t0:.2f}s")

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

print("\n3. Profiling initialize() step-by-step...")
print("   Calling net.initialize()...")

# Monkey-patch to add timing
original_get_mask0 = net.get_mask0

def timed_get_mask0():
    t = time.time()
    result = original_get_mask0()
    print(f"      [TIMING] get_mask0() took {time.time() - t:.2f}s")
    return result

net.get_mask0 = timed_get_mask0

# Also time update_train_opts
original_update_train_opts = net.update_train_opts

def timed_update_train_opts(opts):
    t = time.time()
    result = original_update_train_opts(opts)
    print(f"      [TIMING] update_train_opts() took {time.time() - t:.2f}s")
    return result

net.update_train_opts = timed_update_train_opts

# Now call initialize
t_init = time.time()
net.initialize(train_opts=train_opts)
total_init = time.time() - t_init

print(f"\n   Total initialize() time: {total_init:.2f}s")

if total_init > 10:
    print(f"\n   ❌ SLOW INITIALIZATION ({total_init:.0f}s)")
    print(f"   Expected: < 2s")
    print(f"   Something is still wrong!")
else:
    print(f"\n   ✓ Initialization is fast")

print("\n4. Testing one training trial...")
t0 = time.time()
net.reset(mu=net.ep, sd=net.train_opts['init_noise_mag'])
t_reset = time.time() - t0

t0 = time.time()
net.run_wrapup(update_q_discrete=False)
t_wrapup = time.time() - t0

print(f"   reset(): {t_reset:.3f}s")
print(f"   run_wrapup(): {t_wrapup:.3f}s")
print(f"   Total trial: {t_reset + t_wrapup:.3f}s")

if t_wrapup > 10:
    print(f"\n   ❌ run_wrapup() is SLOW ({t_wrapup:.0f}s)")
    print(f"   This explains slow training!")
elif t_wrapup > 1:
    print(f"\n   ⚠️  run_wrapup() is moderate ({t_wrapup:.1f}s)")
else:
    print(f"\n   ✓ Trial execution is fast")

print("\n" + "="*70)
print("SUMMARY")
print("="*70)
print(f"  Network creation: ~1-2s")
print(f"  Initialization: {total_init:.1f}s")
print(f"  Trial execution: {t_wrapup:.1f}s/trial")
print(f"\nProjected epoch time: {(t_wrapup * 4):.1f}s (4 trials per epoch)")
print("="*70)
