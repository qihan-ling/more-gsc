#!/usr/bin/env python3
"""
Test if using Newton method instead of integration produces stronger equilibrium points
"""
import matplotlib
matplotlib.use('Agg')
import only_gscnet as gsc
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

print("="*70)
print("Testing Newton vs Integration for equilibrium calculation")
print("="*70)

# Create HarmonicGrammar
hg = gsc.HarmonicGrammar(pcfg=PCFG_G1, root='S', max_sent_len=5)
sim = hg.get_simlist(dp=0.0)

net_opts = {
    'T_init': 0.01,
    'q_max': 15.0,
    'q_init': 0.0,
    'dt_init': 0.005,
    'm': 30,
    'use_runC': True,
}

# Create network
net = gsc.GscNet(hg=hg, encodings={'similarity': sim}, opts=net_opts, seed=1024)
net.generate_corpus(use_freq=True)

print("\nCorpus sentences:")
for si, sent in enumerate(net.corpus['sentence']):
    word_seq = ' '.join([bname.split('/')[0] for bname in sent])
    prob = net.corpus['prob_sent'][si]
    print(f"  S{si}: {word_seq:25s} (p={prob:.4f})")

# Initialize training
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

print("\n" + "="*70)
print("Equilibrium method being used:")
print("="*70)
print(f"  net.train_opts['ep_method'] = {net.train_opts['ep_method']}")

if net.train_opts['ep_method'] == 'newton':
    print("  ✓ Using Newton method (more accurate)")
elif net.train_opts['ep_method'] == 'integration':
    print("  ✗ Using integration method (less accurate)")
else:
    print(f"  ? Unknown method: {net.train_opts['ep_method']}")

print("\n" + "="*70)
print("Current equilibrium point values for S4-relevant bindings:")
print("="*70)

# Find S4 sentence index
s4_idx = None
for si, sent in enumerate(net.corpus['sentence']):
    word_seq = ' '.join([bname.split('/')[0] for bname in sent])
    if word_seq == 'N Vpp P N Vi':
        s4_idx = si
        break

if s4_idx is not None:
    print(f"\nS4 (index {s4_idx}): N Vpp P N Vi")

    # Check key bindings for S4
    key_bindings = ['Vi:1/(1,5)', 'Vpp:0/(1,3)', 'N:0/(1,1)', 'P:0/(3,1)', 'N:1/(3,3)']

    for binding_name in key_bindings:
        if binding_name in net.binding_names:
            idx = net.binding_names.index(binding_name)
            ep_val = net.ep[idx]
            print(f"  {binding_name:20s}: ep = {ep_val:7.4f}")
        else:
            print(f"  {binding_name:20s}: NOT FOUND")
else:
    print("\nS4 not found in corpus!")

print("\n" + "="*70)
print("RECOMMENDATION:")
print("="*70)
print("""
To test this fix:
1. Delete old trained models (g1.pkl, *.pkl)
2. Retrain with Newton method: python cho_grammar1_new_copy.py
3. Check if S4 parsing accuracy improves
4. Compare equilibrium values before/after

The Newton method should produce stronger, more accurate equilibrium points
for rare structures like S4, leading to better parsing accuracy.
""")
