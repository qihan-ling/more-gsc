#!/usr/bin/env python3
"""
Visual comparison of grammar structures: terminals vs. non-terminals

This demonstrates the difference between treating N/V as terminals
vs. treating them as non-terminals that expand to POS tags.
"""

import sys
import os
sys.path.insert(0, os.path.dirname(__file__))


def parse_pcfg(pcfg_str):
    """Simple PCFG parser"""
    rules = {}
    for line in pcfg_str.strip().split('\n'):
        line = line.split('#')[0].strip()  # Remove comments
        if not line or '->' not in line:
            continue
        parts = line.split('->')
        if len(parts) != 2:
            continue
        lhs_part = parts[0].strip().split()
        if len(lhs_part) == 2:
            prob, lhs = lhs_part
        else:
            prob = 1.0
            lhs = lhs_part[0]

        rhs = parts[1].strip().split()

        if lhs not in rules:
            rules[lhs] = []
        rules[lhs].append((float(prob), rhs))

    return rules


def get_terminals(rules):
    """Find terminal symbols (appear on RHS but never on LHS)"""
    all_symbols = set()
    nonterminals = set(rules.keys())

    for lhs, productions in rules.items():
        for prob, rhs in productions:
            all_symbols.update(rhs)

    terminals = all_symbols - nonterminals
    return terminals


def print_tree(symbol, rules, indent=0, max_depth=5):
    """Print a parse tree showing expansions"""
    if indent > max_depth:
        return

    prefix = "  " * indent

    if symbol not in rules:
        # Terminal symbol
        print(f"{prefix}{symbol} (TERMINAL)")
        return

    # Non-terminal - show possible expansions
    print(f"{prefix}{symbol} (NON-TERMINAL) can expand to:")
    for prob, rhs in rules[symbol][:2]:  # Show first 2 expansions
        rhs_str = " ".join(rhs)
        print(f"{prefix}  [{prob:.2f}] {symbol} -> {rhs_str}")
        for child in rhs:
            print_tree(child, rules, indent + 2, max_depth)
        if len(rules[symbol]) > 1:
            print()  # Blank line between alternatives


print("="*80)
print("Grammar Structure Comparison")
print("="*80)

# ============================================================================
# 1. Original cho_grammar1 style (N, Vi as terminals)
# ============================================================================

PCFG_ORIGINAL = '''
0.35 S -> N Vi
0.30 S -> N Vi PP
1.0 PP -> P N
'''

print("\n" + "="*80)
print("1. ORIGINAL cho_grammar1 Style (N, Vi as TERMINALS)")
print("="*80)

rules1 = parse_pcfg(PCFG_ORIGINAL)
terminals1 = get_terminals(rules1)

print(f"\nGrammar rules:")
for lhs, productions in rules1.items():
    for prob, rhs in productions:
        print(f"  {prob:.2f} {lhs} -> {' '.join(rhs)}")

print(f"\nNon-terminals: {sorted(rules1.keys())}")
print(f"Terminals: {sorted(terminals1)}")

print(f"\nParse tree structure:")
print_tree('S', rules1)

print(f"\nGSC Input Format: N/(1,1) Vi/(1,2)")
print(f"                  └─────┴─── These are the TERMINALS")


# ============================================================================
# 2. Direct POS style (DT, NN, VBD as terminals)
# ============================================================================

PCFG_DIRECT_POS = '''
0.40 S -> NP VP
0.40 NP -> DT NN
0.35 VP -> VBD
1.0 PP -> IN NP
'''

print("\n\n" + "="*80)
print("2. DIRECT POS Style (DT, NN, VBD as TERMINALS)")
print("="*80)

rules2 = parse_pcfg(PCFG_DIRECT_POS)
terminals2 = get_terminals(rules2)

print(f"\nGrammar rules:")
for lhs, productions in rules2.items():
    for prob, rhs in productions:
        print(f"  {prob:.2f} {lhs} -> {' '.join(rhs)}")

print(f"\nNon-terminals: {sorted(rules2.keys())}")
print(f"Terminals: {sorted(terminals2)}")

print(f"\nParse tree structure:")
print_tree('S', rules2)

print(f"\nGSC Input Format: DT/(1,1) NN/(1,2) VBD/(1,3)")
print(f"                  └──────┴────────┴─── These are the TERMINALS")


# ============================================================================
# 3. Abstracted style (N, V as non-terminals expanding to POS)
# ============================================================================

PCFG_ABSTRACT = '''
0.40 S -> NP VP
0.50 NP -> DT N
0.70 N -> NN
0.20 N -> NNP
0.35 VP -> V
0.40 V -> VBD
0.30 V -> VBZ
'''

print("\n\n" + "="*80)
print("3. ABSTRACTED Style (N, V as NON-TERMINALS)")
print("="*80)

rules3 = parse_pcfg(PCFG_ABSTRACT)
terminals3 = get_terminals(rules3)

print(f"\nGrammar rules:")
for lhs, productions in rules3.items():
    for prob, rhs in productions:
        print(f"  {prob:.2f} {lhs} -> {' '.join(rhs)}")

print(f"\nNon-terminals: {sorted(rules3.keys())}")
print(f"Terminals: {sorted(terminals3)}")

print(f"\nParse tree structure:")
print_tree('S', rules3, max_depth=8)

print(f"\nGSC Input Format: DT/(1,1) NN/(1,2) VBD/(1,3)")
print(f"                  └──────┴────────┴─── Still POS tags!")
print(f"\nNote: N and V are in the parse tree but NOT in the input!")


# ============================================================================
# Summary
# ============================================================================

print("\n\n" + "="*80)
print("SUMMARY: What changes when N/V become non-terminals?")
print("="*80)

print("""
┌─────────────────────┬───────────────────┬─────────────────────┬──────────────────┐
│ Approach            │ N/V Status        │ Terminals           │ GSC Input        │
├─────────────────────┼───────────────────┼─────────────────────┼──────────────────┤
│ cho_grammar1        │ TERMINALS         │ N, Vi, P            │ N/(1,1) Vi/(1,2) │
│ Direct POS          │ Not in grammar    │ DT, NN, VBD         │ DT/(1,1) NN/...  │
│ Abstracted          │ NON-TERMINALS     │ Still DT, NN, VBD   │ DT/(1,1) NN/...  │
└─────────────────────┴───────────────────┴─────────────────────┴──────────────────┘

KEY INSIGHT:
  Making N/V non-terminals adds DEPTH to the parse tree,
  but does NOT change the GSC input format!

  The GSC input always uses the TERMINAL symbols (leaf nodes).

  Original:     S -> N Vi         (N is terminal, appears in input)
  Abstracted:   S -> NP VP -> DT N V -> DT NN VBD
                (NN is terminal, appears in input; N is just in parse tree)
""")

print("\n" + "="*80)
print("How to convert cho_grammar1.py to PTB style:")
print("="*80)

print("""
Step 1: Decide if you need N/V abstraction
  - NO: Use Direct POS style (simpler, faster) ← RECOMMENDED
  - YES: Use Abstracted style (for research purposes)

Step 2: Replace terminal symbols in PCFG
  Before:  0.35 S -> N Vi
  After:   0.40 S -> NP VP
           0.40 NP -> DT NN
           0.35 VP -> VBD

Step 3: GSC input format changes automatically
  Before:  net.set_input(['N/(1,1)', 'Vi/(1,2)'])
  After:   net.set_input(['DT/(1,1)', 'NN/(1,2)', 'VBD/(1,3)'])

Step 4: The ptb_to_gsc.py tool does this for you!
  Just run: python ptb_to_gsc.py your_ptb_file.txt --binarize
""")

print("\n" + "="*80)
print("Complete! See grammar_hierarchy_explained.md for more details.")
print("="*80)
