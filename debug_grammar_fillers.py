#!/usr/bin/env python3
"""
Debug where extra fillers come from: Grammar vs HarmonicGrammar
"""
import gsc
import only_datastructure_speedup as gsc_new

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
print("Comparing Grammar-level fillers")
print("="*70)

# Create Grammar objects (not HarmonicGrammar)
pcfg_orig = gsc.PCFG(PCFG_G1, root='S')
pcfg_new = gsc_new.PCFG(PCFG_G1, root='S')

print(f"\nAt Grammar level (PCFG class):")
print(f"  Original fillers: {len(pcfg_orig.filler_names)}")
print(f"  New fillers:      {len(pcfg_new.filler_names)}")

if len(pcfg_orig.filler_names) == len(pcfg_new.filler_names):
    print("  ✓ Grammar-level filler counts match!")
else:
    print(f"  ✗ Difference: {abs(len(pcfg_orig.filler_names) - len(pcfg_new.filler_names))} fillers")

print(f"\nOriginal Grammar fillers ({len(pcfg_orig.filler_names)}):")
for i, f in enumerate(pcfg_orig.filler_names):
    print(f"  {i:2d}: {f}")

print(f"\nNew Grammar fillers ({len(pcfg_new.filler_names)}):")
for i, f in enumerate(pcfg_new.filler_names):
    print(f"  {i:2d}: {f}")

print("\n" + "="*70)
print("Comparing HarmonicGrammar-level fillers")
print("="*70)

# Now create HarmonicGrammar objects
hg_orig = gsc.HarmonicGrammar(pcfg=PCFG_G1, root='S', max_sent_len=5)
hg_new = gsc_new.HarmonicGrammar(pcfg=PCFG_G1, root='S', max_sent_len=5)

print(f"\nAt HarmonicGrammar level:")
print(f"  Original fillers: {len(hg_orig.filler_names)}")
print(f"  New fillers:      {len(hg_new.filler_names)}")

if len(hg_orig.filler_names) == len(hg_new.filler_names):
    print("  ✓ HarmonicGrammar-level filler counts match!")
else:
    print(f"  ✗ Difference: {abs(len(hg_orig.filler_names) - len(hg_new.filler_names))} fillers")

print(f"\nOriginal HarmonicGrammar fillers ({len(hg_orig.filler_names)}):")
for i, f in enumerate(hg_orig.filler_names):
    print(f"  {i:2d}: {f}")

print(f"\nNew HarmonicGrammar fillers ({len(hg_new.filler_names)}):")
for i, f in enumerate(hg_new.filler_names):
    print(f"  {i:2d}: {f}")

# Find differences
if len(hg_orig.filler_names) != len(hg_new.filler_names):
    orig_set = set(hg_orig.filler_names)
    new_set = set(hg_new.filler_names)
    extra_in_new = new_set - orig_set
    missing_in_new = orig_set - new_set

    if extra_in_new:
        print(f"\n  Extra fillers in NEW HarmonicGrammar ({len(extra_in_new)}):")
        for f in sorted(extra_in_new):
            print(f"    {f}")
    if missing_in_new:
        print(f"\n  Missing fillers in NEW HarmonicGrammar ({len(missing_in_new)}):")
        for f in sorted(missing_in_new):
            print(f"    {f}")

print("\n" + "="*70)
print("DIAGNOSIS:")
print("="*70)
print("""
If Grammar-level fillers match but HarmonicGrammar-level differ:
  → Bug is in HarmonicGrammar.__init__() or _optimize_using_fast_lookups()

If Grammar-level fillers already differ:
  → Bug is in PCFG class tokenization (_tokenize_cnf or _tokenize_fillers)
""")
