import time
import only_gscnet_speedup_sap as gsc
import numpy as np
import cProfile
import pstats
from io import StringIO

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
print("DETAILED TRAINING PERFORMANCE PROFILING")
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
    # Let auto-detection choose
}

encodings = {'similarity': sim}

net = gsc.GscNet(hg=hg, encodings=encodings, opts=net_opts, seed=1024)
print(f"   Time: {time.time() - t0:.2f}s")
print(f"   - Number of bindings: {net.num_bindings}")
print(f"   - WC shape: {net.WC.shape}")
print(f"   - Using sparse: {hasattr(net, 'use_sparse') and net.use_sparse}")
if hasattr(net, 'use_sparse') and net.use_sparse:
    import scipy.sparse as sparse
    print(f"   - WC non-zeros: {net.WC.nnz:,}")
else:
    print(f"   - WC is dense array")

print("\n2. Generating corpus...")
t0 = time.time()
net.generate_corpus(use_freq=True, nsamples=5000)
print(f"   Time: {time.time() - t0:.2f}s")
print(f"   - Corpus size: {len(net.corpus['sentence'])}")

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
print("PROFILING ONE COMPLETE EPOCH (DETAILED)")
print("="*70)

# Profile individual components
timings = {}

# 4a. Profile estimate_prob_inc with individual trial breakdown
print("\n4a. Profiling estimate_prob_inc (4 trials)...")
trial_times = []

# Manually run trials to profile each one
print("   Running trials individually:")
for trial_id in range(4):
    t_trial = time.time()

    # Reset
    t_reset = time.time()
    net.reset(mu=net.ep, sd=net.train_opts['init_noise_mag'])
    t_reset_dur = time.time() - t_reset

    # Run wrapup
    t_wrapup = time.time()
    net.run_wrapup(update_q_discrete=False)
    t_wrapup_dur = time.time() - t_wrapup

    # Read grid point
    t_read = time.time()
    gp = net.read_grid_point(disp=False)
    t_read_dur = time.time() - t_read

    # Find bindings
    t_find = time.time()
    idx = net.find_bindings_fast(gp)
    net.set_discrete_state(gp)
    t_find_dur = time.time() - t_find

    trial_dur = time.time() - t_trial
    trial_times.append(trial_dur)

    print(f"     Trial {trial_id+1}: {trial_dur:.3f}s (reset: {t_reset_dur:.3f}s, wrapup: {t_wrapup_dur:.3f}s, read: {t_read_dur:.3f}s, find: {t_find_dur:.3f}s)")

timings['estimate_prob_inc_total'] = sum(trial_times)
timings['estimate_prob_inc_avg_per_trial'] = np.mean(trial_times)

# Now run the full estimate_prob_inc to also get corpus building time
print("\n   Running full estimate_prob_inc...")
t0 = time.time()
stat_Q = net.estimate_prob_inc(prefix=[], num_trials=4)
timings['estimate_prob_inc_full'] = time.time() - t0

# 4b. Profile get_corpus_stat
print("\n4b. Profiling get_corpus_stat...")
t0 = time.time()
stat_P = net.get_corpus_stat(net.corpus)
timings['get_corpus_stat'] = time.time() - t0
print(f"   Time: {timings['get_corpus_stat']:.3f}s")
print(f"   - Corpus sentences: {len(net.corpus['sentence'])}")
print(f"   - Num roles: {net.num_roles}")
print(f"   - Iterations: {len(net.corpus['sentence']) * net.num_roles}")

# 4c. Profile cost
print("\n4c. Profiling cost computation...")
net.clear_input()
extC_token = net.extC.astype(bool).astype(int)
t0 = time.time()
kl_curr, xent_curr, err, err_log = net.cost(stat_P, stat_Q)
timings['cost'] = time.time() - t0
print(f"   Time: {timings['cost']:.3f}s")

# 4d. Profile cost_grad
print("\n4d. Profiling cost_grad...")
t0 = time.time()
dWC_curr, destr_curr, dq_curr, dbC_curr = net.cost_grad(err, extC_token)
timings['cost_grad'] = time.time() - t0
print(f"   Time: {timings['cost_grad']:.3f}s")

print("\n" + "="*70)
print("SUMMARY - TIME PER EPOCH")
print("="*70)

total = sum(timings.values())
for key, val in sorted(timings.items(), key=lambda x: x[1], reverse=True):
    pct = 100 * val / total if total > 0 else 0
    print(f"  {key:35s}: {val:8.3f}s ({pct:5.1f}%)")
print(f"  {'TOTAL':35s}: {total:8.3f}s")

print("\n" + "="*70)
print("EXTRAPOLATIONS")
print("="*70)
print(f"  Time per epoch:           {total:.1f}s")
print(f"  Time for 140 epochs:      {total * 140 / 60:.1f} minutes ({total * 140 / 3600:.1f} hours)")
print(f"  Time for 1000 epochs:     {total * 1000 / 60:.1f} minutes ({total * 1000 / 3600:.1f} hours)")

# Deep profiling with cProfile
print("\n" + "="*70)
print("DETAILED FUNCTION-LEVEL PROFILING (cProfile)")
print("="*70)
print("Running one complete training iteration with cProfile...\n")

pr = cProfile.Profile()
pr.enable()

# Run one complete iteration
net.reset(mu=net.ep, sd=net.train_opts['init_noise_mag'])
net.run_wrapup(update_q_discrete=False)

pr.disable()

# Print top 30 time-consuming functions
s = StringIO()
ps = pstats.Stats(pr, stream=s).sort_stats('cumulative')
ps.print_stats(30)

print("Top 30 functions by cumulative time:")
print(s.getvalue())

print("\n" + "="*70)
print("BOTTLENECK ANALYSIS")
print("="*70)

# Identify bottleneck
if timings['estimate_prob_inc_full'] > 0.5 * total:
    print("⚠️  BOTTLENECK: estimate_prob_inc (trial execution)")
    print(f"    Average per trial: {timings['estimate_prob_inc_avg_per_trial']:.3f}s")
    print(f"    This suggests the neural dynamics (run_wrapup) is slow")
    print(f"    Check: runC, update_stateC, HGradC functions")
elif timings['get_corpus_stat'] > 0.3 * total:
    print("⚠️  BOTTLENECK: get_corpus_stat")
    print(f"    Processing {len(net.corpus['sentence'])} sentences")
    print(f"    This suggests corpus statistics computation is slow")
elif timings['cost'] > 0.3 * total:
    print("⚠️  BOTTLENECK: cost computation")
elif timings['cost_grad'] > 0.3 * total:
    print("⚠️  BOTTLENECK: cost_grad computation")
else:
    print("✓  No single obvious bottleneck - time distributed across components")

print("="*70)
