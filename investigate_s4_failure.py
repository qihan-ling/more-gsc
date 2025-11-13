#!/usr/bin/env python3
"""
Investigate why S4 parsing fails - check for differences between implementations
"""
import matplotlib
matplotlib.use('Agg')
import numpy as np

print("="*70)
print("Investigating S4 Parsing Failure")
print("="*70)

# Test with original gsc.py
print("\n1. Testing ORIGINAL gsc.py:")
print("-"*70)

import gsc as gsc_orig

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

hg_orig = gsc_orig.HarmonicGrammar(pcfg=PCFG_G1, root='S', max_sent_len=5)
sim_orig = hg_orig.get_simlist(dp=0.0)

net_opts = {
    'T_init': 0.01,
    'q_max': 15.0,
    'q_init': 0.0,
    'dt_init': 0.005,
    'm': 30,
    'use_runC': True,
}

net_orig = gsc_orig.GscNet(hg=hg_orig, encodings={'similarity': sim_orig},
                           opts=net_opts, seed=1024)
net_orig.generate_corpus(use_freq=True)

print(f"Network size: {len(net_orig.binding_names)} bindings")
print(f"EP method: {net_orig.opts.get('ep_method', 'integration')}")

# Test with split only_gscnet.py
print("\n2. Testing SPLIT only_gscnet.py:")
print("-"*70)

import only_gscnet as gsc_new

hg_new = gsc_new.HarmonicGrammar(pcfg=PCFG_G1, root='S', max_sent_len=5)
sim_new = hg_new.get_simlist(dp=0.0)

net_new = gsc_new.GscNet(hg=hg_new, encodings={'similarity': sim_new},
                         opts=net_opts, seed=1024)
net_new.generate_corpus(use_freq=True)

print(f"Network size: {len(net_new.binding_names)} bindings")
print(f"EP method: {net_new.opts.get('ep_method', 'integration')}")

# Compare key attributes
print("\n" + "="*70)
print("Comparing Implementations")
print("="*70)

checks = [
    ("Number of fillers", len(net_orig.filler_names), len(net_new.filler_names)),
    ("Number of roles", len(net_orig.role_names), len(net_new.role_names)),
    ("Number of bindings", len(net_orig.binding_names), len(net_new.binding_names)),
    ("Corpus size", len(net_orig.corpus['sentence']), len(net_new.corpus['sentence'])),
]

all_match = True
for name, orig_val, new_val in checks:
    match = "✓" if orig_val == new_val else "✗"
    print(f"  {name:25s}: {orig_val:5} vs {new_val:5}  {match}")
    if orig_val != new_val:
        all_match = False

# Find S4 in both
s4_orig = None
s4_new = None

for si, sent in enumerate(net_orig.corpus['sentence']):
    word_seq = ' '.join([bname.split('/')[0] for bname in sent])
    if word_seq == 'N Vpp P N Vi':
        s4_orig = si
        break

for si, sent in enumerate(net_new.corpus['sentence']):
    word_seq = ' '.join([bname.split('/')[0] for bname in sent])
    if word_seq == 'N Vpp P N Vi':
        s4_new = si
        break

print(f"\n  S4 index in original: {s4_orig}")
print(f"  S4 index in new:      {s4_new}")

if s4_orig is not None and s4_new is not None:
    s4_prob_orig = net_orig.corpus['prob_sent'][s4_orig]
    s4_prob_new = net_new.corpus['prob_sent'][s4_new]
    print(f"  S4 probability orig:  {s4_prob_orig:.4f}")
    print(f"  S4 probability new:   {s4_prob_new:.4f}")

# Check if corpus sentences match
print("\nCorpus comparison:")
corpus_match = True
for si in range(min(len(net_orig.corpus['sentence']), len(net_new.corpus['sentence']))):
    sent_orig = ' '.join([b.split('/')[0] for b in net_orig.corpus['sentence'][si]])
    sent_new = ' '.join([b.split('/')[0] for b in net_new.corpus['sentence'][si]])
    prob_orig = net_orig.corpus['prob_sent'][si]
    prob_new = net_new.corpus['prob_sent'][si]

    if sent_orig != sent_new or abs(prob_orig - prob_new) > 1e-6:
        print(f"  S{si}: MISMATCH")
        print(f"    Orig: {sent_orig:30s} p={prob_orig:.4f}")
        print(f"    New:  {sent_new:30s} p={prob_new:.4f}")
        corpus_match = False

if corpus_match:
    print("  ✓ All corpus sentences match")

print("\n" + "="*70)
print("NEXT STEPS")
print("="*70)

if not all_match:
    print("""
Basic network properties differ between implementations!
This explains why S4 parsing would be different.
Need to investigate why these differ.
""")
elif not corpus_match:
    print("""
Corpus generation differs between implementations!
This could affect training and parsing.
Need to check corpus generation code.
""")
else:
    print("""
Basic setup looks identical. S4 parsing differences might be due to:

1. **Subtle numerical differences** in fast lookup implementations
2. **Training dynamics** (weight initialization, gradient computation)
3. **Integration convergence** (may need longer dur for better EPs)
4. **Random seed effects** during training/parsing

Suggestions:
- Compare equilibrium points after initialization (before training)
- Train both versions and compare learned weights
- Test parsing with the SAME trained model on both implementations
- Try increasing integration duration: dur=20 or dur=30
""")
