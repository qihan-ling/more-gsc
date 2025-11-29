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
MAXLEN = 5

print("="*70)
print("TESTING CORRECTNESS: Comparing gradient computation")
print("="*70)

# Test both versions
results = {}

for module_name in ['only_gscnet_speedup', 'only_gscnet_speedup_sap']:
    print(f"\n{'='*70}")
    print(f"Testing: {module_name}")
    print(f"{'='*70}")

    # Import the module
    if module_name == 'only_gscnet_speedup':
        import only_gscnet_speedup as gsc
    else:
        import only_gscnet_speedup_sap as gsc

    # Setup
    hg = gsc.HarmonicGrammar(pcfg=PCFG_G1, root=ROOT, max_sent_len=MAXLEN)
    sim = hg.get_simlist(dp=0.0)

    net_opts = {
        'use_jax': False,  # Use CPU for both to compare directly
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
    net.generate_corpus(use_freq=True, nsamples=5000)

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

    net.initialize(train_opts=train_opts)

    # Run one training step and capture gradients
    print("\nRunning one training iteration...")

    # Get initial statistics
    stat_P = net.get_corpus_stat(net.corpus)
    stat_Q = net.estimate_prob_inc(prefix=[], num_trials=4)

    # Compute cost and error
    net.clear_input()
    extC_token = net.extC.astype(bool).astype(int)
    kl_curr, xent_curr, err, err_log = net.cost(stat_P, stat_Q)

    # Compute gradients
    dWC_curr, destr_curr, dq_curr, dbC_curr = net.cost_grad(err, extC_token)

    # Store results
    results[module_name] = {
        'kl': kl_curr,
        'dWC_sum': np.sum(np.abs(dWC_curr)) if not hasattr(dWC_curr, 'nnz') else dWC_curr.sum(),
        'dWC_max': np.max(np.abs(dWC_curr)) if not hasattr(dWC_curr, 'nnz') else abs(dWC_curr).max(),
        'dbC_sum': np.sum(np.abs(dbC_curr)),
        'dbC_max': np.max(np.abs(dbC_curr)),
        'WC_type': type(net.WC).__name__,
        'dWC_type': type(dWC_curr).__name__,
    }

    print(f"\nResults for {module_name}:")
    print(f"  KL divergence: {kl_curr['trees']:.6f}")
    print(f"  WC type: {type(net.WC)}")
    print(f"  dWC type: {type(dWC_curr)}")
    print(f"  dWC sum(abs): {results[module_name]['dWC_sum']:.6f}")
    print(f"  dWC max(abs): {results[module_name]['dWC_max']:.6f}")
    print(f"  dbC sum(abs): {results[module_name]['dbC_sum']:.6f}")
    print(f"  dbC max(abs): {results[module_name]['dbC_max']:.6f}")

print("\n" + "="*70)
print("COMPARISON")
print("="*70)

if len(results) == 2:
    r1 = results['only_gscnet_speedup']
    r2 = results['only_gscnet_speedup_sap']

    kl_diff = abs(r1['kl']['trees'] - r2['kl']['trees'])
    dWC_diff = abs(r1['dWC_sum'] - r2['dWC_sum'])
    dbC_diff = abs(r1['dbC_sum'] - r2['dbC_sum'])

    print(f"\nKL difference: {kl_diff:.10f}")
    print(f"dWC sum difference: {dWC_diff:.10f}")
    print(f"dbC sum difference: {dbC_diff:.10f}")

    if kl_diff < 1e-6 and dWC_diff < 1e-6 and dbC_diff < 1e-6:
        print("\n✓ GRADIENTS MATCH - No correctness bug!")
        print("  The learning difference must be due to something else.")
    else:
        print("\n❌ GRADIENTS DO NOT MATCH - Correctness bug found!")
        print("  This explains the worse parsing accuracy.")
        print(f"\n  Relative differences:")
        print(f"    dWC: {dWC_diff / max(r1['dWC_sum'], 1e-10) * 100:.2f}%")
        print(f"    dbC: {dbC_diff / max(r1['dbC_sum'], 1e-10) * 100:.2f}%")

print("="*70)
