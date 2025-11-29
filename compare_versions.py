import time
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
MAXLEN = 5  # Small for fast comparison

print("="*70)
print("COMPARING only_gscnet_speedup.py vs only_gscnet_speedup_sap.py")
print("="*70)

# Test both versions
for module_name in ['only_gscnet_speedup', 'only_gscnet_speedup_sap']:
    print(f"\n{'='*70}")
    print(f"Testing: {module_name}")
    print(f"{'='*70}")

    # Import the module
    if module_name == 'only_gscnet_speedup':
        try:
            import only_gscnet_speedup as gsc
        except ImportError:
            print(f"  ❌ {module_name}.py not found - skipping")
            continue
    else:
        import only_gscnet_speedup_sap as gsc

    # Setup
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

    print(f"\n1. Network setup...")
    t0 = time.time()
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
    print(f"   Setup time: {time.time() - t0:.1f}s")
    print(f"   num_bindings: {net.num_bindings}")

    # Run 10 epochs
    print(f"\n2. Training 10 epochs...")
    t0 = time.time()

    for i in range(10):
        net.train2(train_opts={'num_epochs': 1}, savefilename=None)

    train_time = time.time() - t0
    per_epoch = train_time / 10

    print(f"   Total time: {train_time:.1f}s")
    print(f"   Per epoch: {per_epoch:.2f}s")
    print(f"   Projected 1000 epochs: {per_epoch * 1000 / 60:.1f} minutes")

    # Check final KL
    final_kl = np.mean(net.traces_train['kl_trees'][-10:])
    final_acc = np.mean(net.traces_train['acc'][-10:])

    print(f"\n3. Learning quality:")
    print(f"   Final KL: {final_kl:.4f}")
    print(f"   Final accuracy: {final_acc:.4f}")

print("\n" + "="*70)
print("INTERPRETATION")
print("="*70)
print("\nIf SAP version is slower AND has worse KL/accuracy:")
print("  → There's a correctness bug, not just performance")
print("\nIf SAP version is slower but same KL/accuracy:")
print("  → Pure performance issue")
print("="*70)
