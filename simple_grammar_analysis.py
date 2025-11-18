#!/usr/bin/env python3
"""Simple grammar analysis without heavy dependencies"""
import re
from collections import defaultdict

# Parse the grammar file
grammar_file = 'collapsed_filtered_sm5.grammar'
rules = defaultdict(list)
terminals = set()

with open(grammar_file, 'r') as f:
    for line in f:
        line = line.strip()
        if not line:
            continue
        
        # Parse: probability LHS -> RHS
        parts = line.split()
        if len(parts) >= 4:
            prob = float(parts[0])
            lhs = parts[1]
            # RHS starts from index 3
            rhs = parts[3:]
            
            rules[lhs].append((prob, rhs))
            
            # Detect terminals (all caps, typically POS tags)
            for symbol in rhs:
                # If symbol doesn't start with @ and is all caps, likely terminal
                if not symbol.startswith('@') and symbol.isupper() and len(symbol) <= 4:
                    terminals.add(symbol)

print("=" * 80)
print("GRAMMAR STRUCTURE ANALYSIS")
print("=" * 80)

print(f"\nTotal nonterminals: {len(rules)}")
print(f"Detected terminals: {len(terminals)}")
print(f"\nNonterminals with most rules:")
for nt in sorted(rules.keys(), key=lambda x: len(rules[x]), reverse=True)[:15]:
    print(f"  {nt}: {len(rules[nt])} rules")

# Count rules from S (root)
if 'S' in rules:
    print(f"\n'S' (root) has {len(rules['S'])} production rules")
    print("Top 10 S rules:")
    for i, (prob, rhs) in enumerate(sorted(rules['S'], key=lambda x: x[0], reverse=True)[:10]):
        print(f"  {prob:.4f} S -> {' '.join(rhs)}")

print("\n" + "=" * 80)
print("SAMPLE TERMINALS DETECTED")
print("=" * 80)
sample_terminals = sorted(list(terminals))[:20]
print(f"\n{' '.join(sample_terminals)}")

