#!/usr/bin/env python3
"""
Example script to train GSC with the mini Berkeley Parser grammar.

This shows the correct way to load and use the collapsed grammar files.
"""

import gsc

# ============================================================================
# Load Grammar from File (DO NOT copy-paste!)
# ============================================================================

print("Loading grammar from file...")
with open('collapsed_grammar_sm5_mini.txt', 'r') as f:
    PCFG_MINI = f.read()

# Verify grammar loaded correctly
num_lines = len(PCFG_MINI.strip().split('\n'))
print(f"  Loaded {num_lines} grammar rules")

# ============================================================================
# Initialize HarmonicGrammar
# ============================================================================

ROOT = 'S'      # Root category
MAXLEN = 15     # Maximum sentence length

print(f"\nInitializing HarmonicGrammar with ROOT={ROOT}, MAXLEN={MAXLEN}...")
hg = gsc.HarmonicGrammar(pcfg=PCFG_MINI, root=ROOT, max_sent_len=MAXLEN)

print(f"  Number of fillers: {len(hg.filler_names)}")
print(f"  Filler names (first 10): {hg.filler_names[:10]}")

# ============================================================================
# Set up similarity encoding
# ============================================================================

print("\nSetting up similarity encoding...")
sim = hg.get_simlist(dp=0.0)  # Linear independence (all similarities = 0)

# ============================================================================
# Network options
# ============================================================================

net_opts = {
    'T_init': 0.01,      # computational temperature
    'q_max': 15.0,       # maximum commitment
    'q_init': 0.0,       # initial commitment
    'dt_init': 0.005,    # time step
    'm': 30,             # resource constraint (Hq1 strength)
    'use_runC': True,    # use C implementation for speed
}

# ============================================================================
# Initialize GscNet
# ============================================================================

print("\nInitializing GscNet...")
net = gsc.GscNet(hg=hg, encodings={'similarity': sim},
                 opts=net_opts, seed=1024)

print("  Generating corpus...")
net.generate_corpus(use_freq=True)

print(f"\n  Corpus size: {len(net.corpus['sentence'])} sentences")

# Display some target probabilities
print("\nTarget sentence probabilities (first 5):")
for si in range(min(5, len(net.corpus['sentence']))):
    sent = net.corpus['sentence'][si]
    sent_str = ' '.join([bname.split('/')[0] for bname in sent])
    prob = net.corpus['prob_sent'][si]
    print(f"  Sentence {si}: p = {prob:.6f} ({sent_str[:60]}...)")

# ============================================================================
# Training setup
# ============================================================================

train_opts = {
    'lrate': 0.1,                  # learning rate
    'num_trials': 4,               # production trials per iteration
    'ema_stat_weight': 0.0,        # no EMA smoothing initially
    'trace_varnames': ['kl_trees', 'kl_treelets', 'prob_sent', 'acc'],
    'report_cycle': 10,            # report every 10 iterations
    'init_noise_mag': 0.02,
    'average_weight': False,
    'average_filler_bias': False,
}

print("\nInitializing training...")
net.initialize(train_opts=train_opts)

print("\n" + "="*70)
print("Ready to train!")
print("="*70)
print("\nTo start training, run:")
print("  net.train2(train_opts={'num_epochs': 100}, savefilename='mini_model.pkl')")
print("\nNote: Training 3,840 rules will take significantly longer than toy Grammar 1 (10 rules).")
print("Consider starting with fewer epochs to test timing.")
print("="*70)
