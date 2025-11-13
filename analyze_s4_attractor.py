#!/usr/bin/env python3
"""
Analyze why S4 falls into wrong attractor basin
"""
import only_gscnet as gsc
import numpy as np

print("Loading model...")
net = gsc.load_model('g1_ds_speedup_model_copy.pkl')

print("\n" + "="*70)
print("S4 Attractor Basin Analysis")
print("="*70)

# Find S4
s4_idx = [si for si, sent in enumerate(net.corpus['sentence'])
          if ' '.join([bname.split('/')[0] for bname in sent]) == 'N Vpp P N Vi'][0]

sent = net.corpus['sentence'][s4_idx]
targ = net.corpus['target'][s4_idx]

print("\nS4 Target structure (correct):")
target_bindings = [net.binding_names[i] for i in range(len(net.binding_names)) if targ[i] > 0.5]
print(f"  {len(target_bindings)} active bindings")

# Key structural bindings for S4
key_s4_bindings = [b for b in target_bindings if any(x in b for x in ['NP[', 'RC[', 'Vpp:', 'Vi:1'])]
print(f"\nKey S4 structural bindings:")
for kb in key_s4_bindings:
    idx = net.binding_names.index(kb)
    bias = net.bC[idx]
    print(f"  {kb:30s} bias={bias:7.4f}")

print("\n" + "="*70)
print("Common wrong parse that S4 converges to:")
print("="*70)
print("\nPattern: N Vi P N @ (treating it like S1 with empty 5th word)")

# The wrong parse from debug output
wrong_pattern_bindings = [
    'N:0/(1,1)', 'Vi:0/(1,2)', 'P:0/(1,3)', 'N:1/(1,4)', '@:1/(1,5)',
    '*N:0/(2,1)', '*Vi:0/(2,2)', 'PP[1]:1/(2,3)', '*@:1/(2,4)',
    '*N:0/(3,1)', 'VP[1]:1/(3,2)', '*@:1/(3,3)', '*N:0/(4,1)',
    'S[2]:0/(4,1)', '#:0/(5,1)'
]

print(f"\nWrong parse structural bindings (VP-based, not NP-RC):")
wrong_structural = [b for b in wrong_pattern_bindings if any(x in b for x in ['VP[', 'S[2]', 'Vi:0'])]
for wb in wrong_structural[:10]:
    if wb in net.binding_names:
        idx = net.binding_names.index(wb)
        bias = net.bC[idx]
        print(f"  {wb:30s} bias={bias:7.4f}")

print("\n" + "="*70)
print("Bias comparison: S4 correct structure vs wrong VP structure")
print("="*70)

# Compare key competing bindings
comparisons = [
    ('NP[1]:0/(4,1)', 'S[2]:0/(4,1)', 'Top-level: NP vs S structure'),
    ('RC[1]:1/(3,2)', 'VP[1]:1/(3,2)', 'Level 3: RC vs VP'),
    ('Vpp:0/(1,2)', 'Vi:0/(1,2)', 'Position 2: Vpp vs Vi'),
    ('Vi:1/(1,5)', '@:1/(1,5)', 'Position 5: Vi vs empty'),
]

print("\nDirect competition (correct vs wrong):")
for correct, wrong, desc in comparisons:
    if correct in net.binding_names and wrong in net.binding_names:
        idx_c = net.binding_names.index(correct)
        idx_w = net.binding_names.index(wrong)
        bias_c = net.bC[idx_c]
        bias_w = net.bC[idx_w]
        diff = bias_c - bias_w
        winner = "✓ CORRECT STRONGER" if diff > 0 else "✗ WRONG STRONGER"
        print(f"\n{desc}:")
        print(f"  {correct:20s} bias={bias_c:7.4f}")
        print(f"  {wrong:20s} bias={bias_w:7.4f}")
        print(f"  Difference: {diff:7.4f}  {winner}")

print("\n" + "="*70)
print("Grammar probability analysis:")
print("="*70)

print("\nS4's structure uses:")
print("  S -> NP Vi    (prob = 0.05)  ← RARE!")
print("  NP -> N RC    (prob = 1.0)")
print("  RC -> Vpp PP  (prob = 1.0)")
print("  PP -> P N     (prob = 1.0)")
print("  Total path probability: 0.05")

print("\nWrong parse structure uses:")
print("  S -> N VP     (prob = 0.60)  ← COMMON!")
print("  VP -> Vi PP   (prob = 0.5)")
print("  PP -> P N     (prob = 1.0)")
print("  Total path probability: 0.30")

print("\nThe wrong structure is 6x more probable in the grammar!")
print("The network's learned weights favor the high-probability VP path over rare NP path.")

print("\n" + "="*70)
print("Energy landscape hypothesis:")
print("="*70)
print("""
The S4 target (S->NP Vi) has a WEAK attractor basin because:
1. It appears in only 5% of training examples
2. The competing VP structure (S->N VP) appears in 60% of examples
3. The network learns stronger weights for frequent structures

When the network sees "N Vpp P N Vi", it tries to fit it into the
more familiar VP pattern, misinterpreting:
- Vpp → Vi (more common verb type)
- Vi at end → @ (empty, to fit VP pattern)

Solution approaches:
A) Increase training data for S4 (but violates grammar frequencies)
B) Reduce noise (sd) in parsing to stay closer to equilibrium
C) Adjust commitment schedule specifically for longer/rare sentences
D) Check if there's a bug in how rare structures are trained
""")
