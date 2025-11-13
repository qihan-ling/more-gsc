#!/usr/bin/env python3
"""
Compare rules after CNF tokenization to find where extra fillers come from
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
print("Comparing PCFG Grammar Rules")
print("="*70)

# Create grammar objects
pcfg_orig = gsc.PCFG(PCFG_G1, root='S')
pcfg_new = gsc_new.PCFG(PCFG_G1, root='S')

print(f"\nNumber of rules:")
print(f"  Original: {len(pcfg_orig.rules)}")
print(f"  New:      {len(pcfg_new.rules)}")

print(f"\nNumber of fillers:")
print(f"  Original: {len(pcfg_orig.filler_names)}")
print(f"  New:      {len(pcfg_new.filler_names)}")

# Show rules with non-terminal mothers in NEW
print(f"\n" + "="*70)
print("Rules with non-terminal mothers in NEW (these create filler names!):")
print("="*70)

non_terminal_mothers = set()
for rule in pcfg_new.rules:
    mother = rule['m']
    # Check if mother is a non-terminal (NP, VP, RC, etc.)
    if any(nt in mother for nt in ['NP[', 'VP[', 'RC[', 'VPpp[', 'PP[', 'S[']):
        non_terminal_mothers.add(mother)
        print(f"  {mother} -> {rule['d1']}, {rule['d2']}")

print(f"\nTotal non-terminal mothers in NEW: {len(non_terminal_mothers)}")

# Check original
non_terminal_mothers_orig = set()
for rule in pcfg_orig.rules:
    mother = rule['m']
    if any(nt in mother for nt in ['NP[', 'VP[', 'RC[', 'VPpp[', 'PP[', 'S[']):
        non_terminal_mothers_orig.add(mother)

print(f"Total non-terminal mothers in ORIGINAL: {len(non_terminal_mothers_orig)}")

if len(non_terminal_mothers_orig) > 0:
    print(f"\nNon-terminal mothers in ORIGINAL:")
    for m in sorted(non_terminal_mothers_orig):
        print(f"  {m}")

print(f"\n" + "="*70)
print("DIAGNOSIS:")
print("="*70)

if len(non_terminal_mothers) > len(non_terminal_mothers_orig):
    print(f"""
The NEW version has {len(non_terminal_mothers)} rules with non-terminal mothers!
These become fillers through _add_names(), which extracts all symbols from rules.

In the Tensor Product Representation:
- Fillers should be: terminal symbols (N, Vi, Vpp, P, BE, etc.)
- Roles should be: positions/structures (NP, VP, RC, etc.)

The bug: _tokenize_cnf_optimized() or _tokenize_fillers_optimized() creates
rules where non-terminals appear as mothers, and _add_names() treats them as fillers!

Fix: Modify _add_names() to ONLY extract terminal symbols as fillers, or
     fix the tokenization to not create these rules.
""")
else:
    print("\nBoth versions have same number of non-terminal mothers - bug is elsewhere")
