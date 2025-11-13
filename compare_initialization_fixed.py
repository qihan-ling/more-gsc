#!/usr/bin/env python3
"""
Compare grammar setup between gsc.py and only_gscnet.py - CRITICAL BUG FOUND!
The WC matrices have different shapes: (195,195) vs (405,405)
This means different numbers of fillers: 13 vs 27
"""
import gsc  # Original
import only_gscnet as gsc_new  # Split version
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
print("CRITICAL: Comparing Filler Generation")
print("="*70)

# Initialize both with same grammar
hg_orig = gsc.HarmonicGrammar(pcfg=PCFG_G1, root='S', max_sent_len=5)
hg_new = gsc_new.HarmonicGrammar(pcfg=PCFG_G1, root='S', max_sent_len=5)

print(f"\nNumber of fillers:")
print(f"  Original: {len(hg_orig.filler_names)}")
print(f"  New:      {len(hg_new.filler_names)}")

if len(hg_orig.filler_names) != len(hg_new.filler_names):
    print(f"\n⚠️  CRITICAL BUG: {abs(len(hg_orig.filler_names) - len(hg_new.filler_names))} filler difference!")

    print(f"\nOriginal fillers ({len(hg_orig.filler_names)}):")
    for i, f in enumerate(hg_orig.filler_names):
        print(f"  {i:2d}: {f}")

    print(f"\nNew fillers ({len(hg_new.filler_names)}):")
    for i, f in enumerate(hg_new.filler_names):
        print(f"  {i:2d}: {f}")

    # Find differences
    orig_set = set(hg_orig.filler_names)
    new_set = set(hg_new.filler_names)
    extra_in_new = new_set - orig_set
    missing_in_new = orig_set - new_set

    if extra_in_new:
        print(f"\n  Extra fillers in NEW ({len(extra_in_new)}):")
        for f in sorted(extra_in_new):
            print(f"    {f}")
    if missing_in_new:
        print(f"\n  Missing fillers in NEW ({len(missing_in_new)}):")
        for f in sorted(missing_in_new):
            print(f"    {f}")

print(f"\nNumber of roles:")
print(f"  Original: {len(hg_orig.role_names)}")
print(f"  New:      {len(hg_new.role_names)}")

print(f"\nNumber of bindings:")
print(f"  Original: {len(hg_orig.binding_names)} = {len(hg_orig.filler_names)} fillers × {len(hg_orig.role_names)} roles")
print(f"  New:      {len(hg_new.binding_names)} = {len(hg_new.filler_names)} fillers × {len(hg_new.role_names)} roles")

print("\n" + "="*70)
print("DIAGNOSIS:")
print("="*70)

if len(hg_orig.filler_names) != len(hg_new.filler_names):
    print("""
The split code generates MORE fillers than the original!

This causes:
- Different network size (405 bindings vs 195 bindings)
- Different weight matrices
- Different training dynamics
- WEAKER equilibrium points for S4's rare structure

Root cause: Something in the Grammar or filler generation changed during split.

Next step: Find where filler generation differs between:
  - gsc.py Grammar class
  - only_datastructure_speedup.py Grammar class
""")
else:
    print("\n✓ Filler counts match - bug is elsewhere")
