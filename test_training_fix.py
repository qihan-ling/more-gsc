#!/usr/bin/env python
import gsc
import numpy as np

# Grammar 1 from the paper
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

# Initialize network
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

train_opts = {
    'lrate': 0.1,
    'num_trials': 4,
    'ema_stat_weight': 0.0,
    'trace_varnames': ['kl_trees', 'prob_sent'],
    'report_cycle': 1,
    'init_noise_mag': 0.02,
    'average_weight': False,
    'average_filler_bias': False,
}

net.initialize(train_opts=train_opts)

print("=" * 70)
print("Training Grammar 1 - Testing Fix")
print("=" * 70)
print(f"use_second_order_bias: {net.opts['use_second_order_bias']}")
print(f"bias1_only: {net.train_opts['bias1_only']}")
print(f"optimizer: {net.train_opts['optimizer']}")
print("=" * 70)

# Train for just 30 epochs to see if learning happens
net.train2(train_opts={'num_epochs': 30})

print("\n" + "=" * 70)
print("Training Complete!")
print("=" * 70)

# Check if probabilities changed from zero
final_probs = net.traces_train['prob_sent'][-1]
print(f"Final sentence probabilities: {final_probs}")
print(f"Sum of probabilities: {np.sum(final_probs):.4f}")
print(f"Final KL divergence: {net.traces_train['kl_trees'][-1]:.4f}")

# Compare to initial
initial_probs = net.traces_train['prob_sent'][0]
print(f"\nInitial probabilities: {initial_probs}")
print(f"Probabilities changed: {not np.allclose(initial_probs, final_probs)}")
