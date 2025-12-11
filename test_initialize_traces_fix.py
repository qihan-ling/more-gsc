"""Quick test to verify initialize_traces() fix for treelet activations"""

import only_gscnet_speedup_sap as gsc
import numpy as np

# ============================================================================
# Grammar 1 (G1) from Section 4.1
# ============================================================================

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

# ============================================================================
# Initialize network
# ============================================================================

hg = gsc.HarmonicGrammar(pcfg=PCFG_G1, root=ROOT, max_sent_len=MAXLEN)

print(f"Number of fillers: {len(hg.filler_names)}")

# Set all filler similarities to 0 (linear independence)
sim = hg.get_simlist(dp=0.0)

# Network options
net_opts = {
    'T_init': 0.01,
    'q_max': 15.0,
    'q_init': 0.0,
    'dt_init': 0.005,
    'm': 30,
    'use_runC': True,
}

# Initialize network
net = gsc.GscNet(hg=hg, encodings={'similarity': sim},
                 opts=net_opts, seed=1024)
net.generate_corpus(use_freq=True)

print("\n" + "="*70)
print("Corpus sentences:")
for si, sent in enumerate(net.corpus['sentence']):
    sent_str = ' '.join([bname.split('/')[0] for bname in sent])
    print(f"S{si}: {sent_str}")

# ============================================================================
# Quick training (just 50 epochs to get reasonable weights)
# ============================================================================

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
print("Quick training (50 epochs)...")
print("="*70)

net.train2(train_opts={'num_epochs': 50}, savefilename='test_model.pkl')

print("\nTraining complete!")

# ============================================================================
# Test treelet activations for multiple sentences
# ============================================================================

print("\n" + "="*70)
print("Testing treelet activations (should be sentence-specific)")
print("="*70)

# Get rules for treelet computation
rules0 = net.hg.g.get_rules()
rules = []
for rule in rules0:
    if rule not in rules:
        rules.append(rule)

# Test role (3,2) for first 3 sentences
rname = '(3,2)'

for si in [0, 1, 3]:  # Test S0, S1, S3
    sent = net.corpus['sentence'][si]
    words = [bname.split('/')[0] for bname in sent]
    sent_str = ' '.join(words)

    print(f"\n--- Sentence {si}: {sent_str} ---")

    # Reset and run sentence
    net.reset(mu=net.ep, sd=0.01)
    net.initialize_traces(trace_list='all')  # This should properly clear and log initial state

    for wi, word in enumerate(words):
        net.run_word(word, wi + 1, log_trace=True)
    net.run_wrapup(log_trace=True)

    # Compute treelet activations
    actC_trace = net.traces['actC']
    print(f"  Trace shape: {np.array(actC_trace).shape}")

    dp_all = gsc.compute_treelet_act_trace(net, actC_trace, rules, rname)

    # Find top 4 by total activation
    temp = np.argsort(dp_all.sum(axis=0))
    focus_idx = temp[::-1][:4]

    print(f"  Top 4 treelets at role {rname}:")
    for rank, idx in enumerate(focus_idx):
        rule = rules[idx]
        label = gsc.rule2str(rule, suppress_pos=True)
        total_activation = dp_all[:, idx].sum()
        print(f"    {rank+1}. {label:30s} (activation: {total_activation:.2f})")

print("\n" + "="*70)
print("Test complete!")
print("\nExpected results:")
print("  - S0 ('N Vi') should show N and Vi related treelets")
print("  - S1 ('N Vi P N') should show PP and VP related treelets")
print("  - S3 ('N BE Vpp P N') should show BE and VPpp related treelets")
print("  - Each sentence should have DIFFERENT top 4 treelets!")
print("="*70)
