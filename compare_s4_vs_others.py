#!/usr/bin/env python3
"""
Compare S4's target structure with other sentences to identify structural differences
"""
import only_gscnet as gsc
import numpy as np

# Load trained model
print("Loading model...")
net = gsc.load_model('g1_ds_speedup_model_copy.pkl')

print(f"\n{'='*70}")
print("Analyzing all sentence structures")
print(f"{'='*70}")

# Analyze each sentence
for si, sent in enumerate(net.corpus['sentence']):
    word_seq = ' '.join([bname.split('/')[0] for bname in sent])
    targ = net.corpus['target'][si]

    # Count active bindings in target
    active_bindings = [net.binding_names[i] for i in range(len(net.binding_names)) if targ[i] > 0.5]

    # Remove empty fillers to get actual sentence
    f_empty_type = net.hg.g.get_types(net.hg.opts['f_empty'])
    sent0 = [bname for bname in sent if bname.split('/')[0] not in f_empty_type]

    print(f"\nS{si}: {word_seq}")
    print(f"  Sentence length: {len(sent0)} words")
    print(f"  Active bindings in target: {len(active_bindings)}")
    print(f"  Target probability: {net.corpus['prob_sent'][si]:.4f}")

    # Show binding names
    print(f"  Sample target bindings:")
    for ab in active_bindings[:8]:
        print(f"    {ab}")

    # Check if target contains any unusual patterns
    # Look for specific role positions
    role_counts = {}
    for ab in active_bindings:
        role = ab.split('/')[1]
        role_counts[role] = role_counts.get(role, 0) + 1

    print(f"  Role distribution in target:")
    for role, count in sorted(role_counts.items()):
        print(f"    {role}: {count} bindings")

# Now check the learned biases and weights for S4-specific structures
print(f"\n{'='*70}")
print("Checking learned parameters for S4-specific structures")
print(f"{'='*70}")

# S4 uses: S -> NP Vi, NP -> N RC, RC -> Vpp PP
# Key filler: Vpp at various positions

# Find Vpp-related bindings
vpp_bindings = [i for i, bname in enumerate(net.binding_names) if 'Vpp' in bname]
print(f"\nVpp-related bindings ({len(vpp_bindings)} total):")
for vpp_idx in vpp_bindings[:10]:
    bname = net.binding_names[vpp_idx]
    bias = net.bC[vpp_idx]
    print(f"  {bname}: bias = {bias:.4f}")

# Find NP-related bindings
np_bindings = [i for i, bname in enumerate(net.binding_names) if '/NP' in bname or bname.startswith('NP/')]
print(f"\nNP-related bindings ({len(np_bindings)} total):")
for np_idx in np_bindings[:10]:
    bname = net.binding_names[np_idx]
    bias = net.bC[np_idx]
    print(f"  {bname}: bias = {bias:.4f}")

# Find RC-related bindings
rc_bindings = [i for i, bname in enumerate(net.binding_names) if '/RC' in bname or bname.startswith('RC/')]
print(f"\nRC-related bindings ({len(rc_bindings)} total):")
for rc_idx in rc_bindings[:10]:
    bname = net.binding_names[rc_idx]
    bias = net.bC[rc_idx]
    print(f"  {bname}: bias = {bias:.4f}")

# Compare bias magnitudes for S4's target vs S1's target
print(f"\n{'='*70}")
print("Comparing target bias magnitudes: S4 vs S1")
print(f"{'='*70}")

s4_idx = [si for si, sent in enumerate(net.corpus['sentence'])
          if ' '.join([bname.split('/')[0] for bname in sent]) == 'N Vpp P N Vi'][0]
s1_idx = [si for si, sent in enumerate(net.corpus['sentence'])
          if ' '.join([bname.split('/')[0] for bname in sent]) == 'N Vi P N'][0]

s4_targ = net.corpus['target'][s4_idx]
s1_targ = net.corpus['target'][s1_idx]

s4_active = [i for i in range(len(net.binding_names)) if s4_targ[i] > 0.5]
s1_active = [i for i in range(len(net.binding_names)) if s1_targ[i] > 0.5]

s4_biases = net.bC[s4_active]
s1_biases = net.bC[s1_active]

print(f"\nS4 target biases:")
print(f"  Mean: {s4_biases.mean():.4f}")
print(f"  Std: {s4_biases.std():.4f}")
print(f"  Min: {s4_biases.min():.4f}")
print(f"  Max: {s4_biases.max():.4f}")

print(f"\nS1 target biases:")
print(f"  Mean: {s1_biases.mean():.4f}")
print(f"  Std: {s1_biases.std():.4f}")
print(f"  Min: {s1_biases.min():.4f}")
print(f"  Max: {s1_biases.max():.4f}")

print(f"\nInterpretation:")
if s4_biases.mean() < s1_biases.mean():
    print(f"  S4's target bindings have LOWER average bias than S1")
    print(f"  → S4's attractor basin may be weaker!")
else:
    print(f"  S4's target bindings have SIMILAR or HIGHER bias than S1")
    print(f"  → Bias strength is not the issue")

# Check if there are competing attractors near S4's target
print(f"\n{'='*70}")
print("Checking for competing structures at S4's sentence length")
print(f"{'='*70}")

# S4 is 5 words: N Vpp P N Vi
# Could the network be confusing it with other 5-word structures?

print("\nAll 5-word sentences in corpus:")
for si, sent in enumerate(net.corpus['sentence']):
    f_empty_type = net.hg.g.get_types(net.hg.opts['f_empty'])
    sent0 = [bname for bname in sent if bname.split('/')[0] not in f_empty_type]
    if len(sent0) == 5:
        word_seq = ' '.join([bname.split('/')[0] for bname in sent])
        prob = net.corpus['prob_sent'][si]
        print(f"  S{si}: {word_seq} (prob={prob:.4f})")

print("\nNote: If S4 is the only 5-word sentence, its unique structure")
print("      may be harder for the network to learn with limited training.")
