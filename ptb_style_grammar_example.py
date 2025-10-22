#!/usr/bin/env python3
"""
Example: Penn Treebank-style Grammar for GSC

This shows how to create a multi-level grammar where:
- S, NP, VP are phrasal (non-terminal) nodes
- N, V are intermediate non-terminals (not used directly, replaced by POS tags)
- DT, NN, VBD, etc. are POS tags (pre-terminals / terminals for GSC)

Hierarchy:
    S -> NP VP
    NP -> DT NN
    VP -> VBD
    (POS tags are the "terminal" level for GSC input)
"""

import gsc
import matplotlib.pyplot as plt
import numpy as np


# ============================================================================
# Option 1: Direct POS-level Grammar (Recommended for Penn Treebank)
# ============================================================================
# In this approach, we skip "N" and "V" abstractions entirely
# and use POS tags as the terminal symbols

PCFG_PTB_STYLE = '''
# Sentence structures
0.40 S -> NP VP
0.30 S -> NP VP PP
0.20 S -> NP VP SBAR
0.10 S -> NP VP ADVP

# Noun phrases
0.40 NP -> DT NN
0.25 NP -> DT JJ NN
0.15 NP -> NNP
0.10 NP -> PRP
0.10 NP -> DT NN PP

# Verb phrases
0.35 VP -> VBD
0.25 VP -> VBD NP
0.20 VP -> VBZ NP
0.10 VP -> VBD PP
0.10 VP -> VBP SBAR

# Prepositional phrases
1.0 PP -> IN NP

# Subordinate clauses
1.0 SBAR -> IN S

# Adverb phrases
1.0 ADVP -> RB
'''


# ============================================================================
# Option 2: Abstracted Grammar (if you want N/V as intermediate categories)
# ============================================================================
# This adds an extra layer where N and V are abstract categories
# that expand to specific POS tags

PCFG_WITH_ABSTRACTIONS = '''
# Sentence structures
0.40 S -> NP VP
0.30 S -> NP VP PP
0.20 S -> NP VP SBAR
0.10 S -> NP VP ADVP

# Noun phrases expand to abstract N category
0.50 NP -> DT N
0.25 NP -> DT JJ N
0.15 NP -> N
0.10 NP -> PRP

# Abstract N category expands to specific noun POS tags
0.70 N -> NN
0.20 N -> NNP
0.10 N -> NNS

# Verb phrases expand to abstract V category
0.35 VP -> V
0.25 VP -> V NP
0.20 VP -> V NP PP
0.10 VP -> V PP
0.10 VP -> V SBAR

# Abstract V category expands to specific verb POS tags
0.40 V -> VBD
0.30 V -> VBZ
0.20 V -> VBP
0.10 V -> VB

# Prepositional phrases
1.0 PP -> IN NP

# Subordinate clauses
1.0 SBAR -> IN S

# Adverb phrases
1.0 ADVP -> RB
'''


# ============================================================================
# Example 1: Using Direct POS-level Grammar
# ============================================================================
print("="*70)
print("Example 1: Direct POS-level Grammar (Penn Treebank style)")
print("="*70)

ROOT = 'S'
MAXLEN = 10

# Initialize with PTB-style grammar
hg1 = gsc.HarmonicGrammar(pcfg=PCFG_PTB_STYLE, root=ROOT, max_sent_len=MAXLEN)

print(f"\nFillers: {hg1.filler_names}")
print(f"Number of fillers: {len(hg1.filler_names)}")
print(f"Number of roles: {len(hg1.role_names)}")
print(f"Total binding units: {len(hg1.filler_names) * len(hg1.role_names)}")

# Display some non-terminal symbols
nonterminals = [f for f in hg1.filler_names if f in ['S', 'NP', 'VP', 'PP']]
print(f"\nPhrasal symbols: {nonterminals}")

# Display POS tags (terminals in GSC sense)
pos_tags = [f for f in hg1.filler_names if f in ['DT', 'NN', 'VBD', 'VBZ', 'IN', 'JJ', 'NNP', 'PRP']]
print(f"POS tag symbols: {pos_tags}")

# Generate sample sentences
print("\nGenerating sample sentences...")
sample_sentences = []
for _ in range(5):
    sent = hg1.pcfg.generate_sentence(max_len=MAXLEN)
    if sent:
        sample_sentences.append(sent)
        print(f"  {' '.join(sent)}")


# ============================================================================
# Example 2: Using Abstracted Grammar (N/V as intermediate)
# ============================================================================
print("\n" + "="*70)
print("Example 2: Abstracted Grammar (N/V as intermediate categories)")
print("="*70)

hg2 = gsc.HarmonicGrammar(pcfg=PCFG_WITH_ABSTRACTIONS, root=ROOT, max_sent_len=MAXLEN)

print(f"\nFillers: {hg2.filler_names}")
print(f"Number of fillers: {len(hg2.filler_names)}")

# Display abstract categories
abstract_cats = [f for f in hg2.filler_names if f in ['N', 'V']]
print(f"\nAbstract categories: {abstract_cats}")

# Display POS tags that these expand to
pos_from_n = [f for f in hg2.filler_names if f in ['NN', 'NNP', 'NNS']]
pos_from_v = [f for f in hg2.filler_names if f in ['VBD', 'VBZ', 'VBP', 'VB']]
print(f"N expands to: {pos_from_n}")
print(f"V expands to: {pos_from_v}")

# Generate sample sentences
print("\nGenerating sample sentences...")
for _ in range(5):
    sent = hg2.pcfg.generate_sentence(max_len=MAXLEN)
    if sent:
        print(f"  {' '.join(sent)}")


# ============================================================================
# Example 3: Training with PTB-style grammar
# ============================================================================
print("\n" + "="*70)
print("Example 3: Training GSC Network with PTB-style Grammar")
print("="*70)

# Use the simpler direct POS grammar for training
hg = gsc.HarmonicGrammar(pcfg=PCFG_PTB_STYLE, root=ROOT, max_sent_len=MAXLEN)

# Set similarity (linear independence)
sim = hg.get_simlist(dp=0.0)

# Network options
net_opts = {
    'T_init': 0.01,
    'q_max': 15.0,
    'q_0': 0.0,
    'dt': 0.005,
    'm': 30,
    'lam_x': 0.5,
    'lam_q': 0.04,
    'use_runC': True,
}

# Initialize network
net = gsc.GscNet(hg=hg, encodings={'similarity': sim},
                 opts=net_opts, seed=1024)

# Generate training corpus
print("\nGenerating training corpus...")
net.generate_corpus(nsamples=1000, use_freq=True)
print(f"Generated {len(net.corpus['sentence'])} sentences")

# Display first few sentences with probabilities
print("\nSample corpus sentences:")
for si in range(min(5, len(net.corpus['sentence']))):
    sent = net.corpus['sentence'][si]
    prob = net.corpus['prob_sent'][si]
    # Extract just the filler symbols
    sent_str = ' '.join([bname.split('/')[0] for bname in sent])
    print(f"  p={prob:.4f}: {sent_str}")

# Training setup
train_opts = {
    'lrate': 0.1,
    'num_trials': 4,
    'trace_varnames': ['kl_trees', 'kl_treelets', 'prob_sent', 'acc'],
    'report_cycle': 10,
}

net.initialize(train_opts=train_opts)

# Train for a few epochs to demonstrate
print("\nTraining for 50 epochs...")
net.train2(train_opts={'num_epochs': 50}, savefilename='ptb_style_model.pkl')

# Show final statistics
if hasattr(net, 'trace_train') and net.trace_train:
    final_kl = np.mean(net.trace_train['kl_trees'][-10:])
    final_acc = np.mean(net.trace_train['acc'][-10:])
    print(f"\nFinal KL divergence: {final_kl:.4f}")
    print(f"Final accuracy: {final_acc:.4f}")


# ============================================================================
# Key Insight: GSC Input Format
# ============================================================================
print("\n" + "="*70)
print("Key Insight: What are 'terminals' in GSC?")
print("="*70)

print("""
In GSC, the "terminal" symbols are whatever appears in the generated sentences.

Original cho_grammar1.py:
  Grammar: S -> N Vi
  Terminals: N, Vi, P
  Input format: N/(1,1) Vi/(1,2)

Penn Treebank style (Option 1 - Direct POS):
  Grammar: S -> NP VP, NP -> DT NN, VP -> VBD
  Terminals: DT, NN, VBD (POS tags)
  Input format: DT/(1,1) NN/(1,2) VBD/(1,3)

Penn Treebank style (Option 2 - With N/V abstraction):
  Grammar: S -> NP VP, NP -> DT N, N -> NN
  Terminals: Still POS tags (NN, VBD, etc.)
  Input format: DT/(1,1) NN/(1,2) VBD/(1,3)

The abstraction (N, V) affects the PARSE TREE structure but not the input!

Example parse tree with abstraction:
        S
       / \\
      NP  VP
     / \\   |
    DT  N   V
        |   |
       NN  VBD

GSC input is the TERMINAL level: DT/(1,1) NN/(1,2) VBD/(1,3)
""")

print("\n" + "="*70)
print("Recommendation: Use Direct POS Grammar (Option 1)")
print("="*70)
print("""
For Penn Treebank training, I recommend Option 1 (direct POS-level grammar):

Advantages:
  ✓ Simpler grammar (fewer non-terminals)
  ✓ Matches PTB annotation directly
  ✓ Faster training (fewer rules to learn)
  ✓ POS tags already provide good abstraction

Use Option 2 (N/V abstraction) only if:
  × You need to model abstract categories explicitly
  × You're studying category learning
  × You want to match specific theoretical requirements

The ptb_to_gsc.py script already produces Option 1 format!
""")
