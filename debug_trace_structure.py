"""Debug script to examine trace structure with the fix"""

import only_gscnet_speedup_sap as gsc
import numpy as np

# Simple setup
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

hg = gsc.HarmonicGrammar(pcfg=PCFG_G1, root='S', max_sent_len=5)
sim = hg.get_simlist(dp=0.0)
net = gsc.GscNet(hg=hg, encodings={'similarity': sim},
                 opts={'T_init': 0.01, 'q_max': 15.0, 'q_init': 0.0,
                       'dt_init': 0.005, 'm': 30, 'use_runC': True},
                 seed=1024)
net.generate_corpus(use_freq=True)

# Quick training
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
net.train2(train_opts={'num_epochs': 10}, savefilename='debug_model.pkl')

print("\n" + "="*70)
print("Testing trace structure - what happens with run_word(log_trace=True)?")
print("="*70)

# Test sentence S1: "N Vi P N"
sent = net.corpus['sentence'][1]
words = [bname.split('/')[0] for bname in sent]
print(f"\nSentence: {' '.join(words)}")

# Reset
net.reset(mu=net.ep, sd=0.01)
print(f"\nAfter reset - actC sum: {net.actC.sum():.6f}")

# External initialize_traces (like in user's test script)
print("\n--- Calling net.initialize_traces(trace_list='all') externally ---")
net.initialize_traces(trace_list='all')
print(f"After external initialize_traces:")
print(f"  Type of traces['actC']: {type(net.traces['actC'])}")
print(f"  Length: {len(net.traces['actC'])}")
if len(net.traces['actC']) > 0:
    print(f"  First entry shape: {np.array(net.traces['actC'][0]).shape}")
    print(f"  First entry sum: {np.array(net.traces['actC'][0]).sum():.6f}")

# First word
print(f"\n--- Calling net.run_word('{words[0]}', 1, log_trace=True) ---")
net.run_word(words[0], 1, log_trace=True)
print(f"After first run_word:")
print(f"  Type of traces['actC']: {type(net.traces['actC'])}")
if isinstance(net.traces['actC'], np.ndarray):
    print(f"  Shape: {net.traces['actC'].shape}")
    print(f"  First 5 timesteps - sum of activations:")
    for i in range(min(5, len(net.traces['actC']))):
        print(f"    t={i}: {net.traces['actC'][i].sum():.6f}")
    print(f"  Last 5 timesteps:")
    for i in range(max(0, len(net.traces['actC'])-5), len(net.traces['actC'])):
        print(f"    t={i}: {net.traces['actC'][i].sum():.6f}")
else:
    print(f"  Length: {len(net.traces['actC'])}")

# Second word
print(f"\n--- Calling net.run_word('{words[1]}', 2, log_trace=True) ---")
print(f"Before second run_word - type: {type(net.traces['actC'])}")
net.run_word(words[1], 2, log_trace=True)
print(f"After second run_word:")
print(f"  Type of traces['actC']: {type(net.traces['actC'])}")
if isinstance(net.traces['actC'], np.ndarray):
    print(f"  Shape: {net.traces['actC'].shape}")
    print(f"  First 5 timesteps - sum of activations:")
    for i in range(min(5, len(net.traces['actC']))):
        print(f"    t={i}: {net.traces['actC'][i].sum():.6f}")
    print(f"  Last 5 timesteps:")
    for i in range(max(0, len(net.traces['actC'])-5), len(net.traces['actC'])):
        print(f"    t={i}: {net.traces['actC'][i].sum():.6f}")

print("\n" + "="*70)
print("DIAGNOSIS:")
print("  - If traces are accumulating correctly, each word should have")
print("    hundreds of timesteps with VARYING activation sums")
print("  - If 'flat', all timesteps would have the SAME activation sum")
print("  - The second word's trace should be INDEPENDENT of first word")
print("="*70)
