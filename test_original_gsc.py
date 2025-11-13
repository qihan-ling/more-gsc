#!/usr/bin/env python3
"""
Test parsing with ORIGINAL gsc.py (before split) to verify it works
"""
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend
import matplotlib.pyplot as plt
import gsc  # ORIGINAL, before split
import numpy as np

print("="*70)
print("Testing ORIGINAL gsc.py (before split)")
print("="*70)

# Same grammar as cho_grammar1_new_copy.py
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

print("\nInitializing network with original gsc.py...")
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

net = gsc.GscNet(hg=hg, encodings={'similarity': sim}, opts=net_opts, seed=1024)
net.generate_corpus(use_freq=True)

print("\n" + "="*70)
print("Corpus generated:")
print("="*70)
for si, sent in enumerate(net.corpus['sentence']):
    word_seq = ' '.join([bname.split('/')[0] for bname in sent])
    prob = net.corpus['prob_sent'][si]
    print(f"S{si}: {word_seq:20s} (p={prob:.4f})")

# Check if we need to train or if model exists
import os
if os.path.exists('g1_original_test.pkl'):
    print("\nLoading existing model...")
    net = gsc.load_model('g1_original_test.pkl')
else:
    print("\n" + "="*70)
    print("Training with original gsc.py (200 epochs)...")
    print("="*70)

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

    for epoch_block in range(20):  # 200 epochs total
        net.train2(
            train_opts={'num_epochs': 10},
            savefilename='g1_original_test.pkl'
        )

    print("\nTraining complete!")
    final_kl = np.mean(net.traces_train['kl_trees'][-100:])
    final_acc = np.mean(net.traces_train['acc'][-100:])
    print(f"Final KL: {final_kl:.3f}, Final Acc: {final_acc:.3f}")

# Test parsing S4 at different commitment levels
print("\n" + "="*70)
print("Testing S4 parsing with ORIGINAL gsc.py:")
print("="*70)

commitment_levels = [1, 3, 5, 7, 10, 12]
max_sent_len = net.hg.opts['max_sent_len']

for t in commitment_levels:
    dq = np.ones(max_sent_len) * (float(t) / max_sent_len)

    # Set seed for reproducibility
    np.random.seed(2048 + t)

    try:
        parse_results = gsc.test_parse_inc(
            net,
            dq=dq,
            num_trials=10,
            estr=2,
            estr_null=2,
            disp=False
        )

        # Find S4
        s4_acc = None
        for si in range(len(net.corpus['sentence'])):
            word_seq = ' '.join([bname.split('/')[0] for bname in net.corpus['sentence'][si]])
            if word_seq == 'N Vpp P N Vi':
                if si in parse_results:
                    s4_acc = parse_results[si]['acc']
                break

        if s4_acc is not None:
            marker = "✓" if s4_acc >= 0.8 else "✗"
            print(f"t={t:2d}: S4 accuracy = {s4_acc:.2f} {marker}")
        else:
            print(f"t={t:2d}: S4 not found")
    except Exception as e:
        print(f"t={t:2d}: ERROR - {e}")

print("\n" + "="*70)
print("Interpretation:")
print("="*70)
print("""
If S4 accuracy is HIGH (≥0.8) with original gsc.py:
  → The bug was introduced during the split into only_gscnet.py
  → Need to compare gsc.py vs only_gscnet.py line-by-line

If S4 accuracy is LOW with original gsc.py too:
  → The issue predates the split
  → May be in the original paper replication attempt
""")
