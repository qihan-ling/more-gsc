#!/usr/bin/env python3
"""
Diagnostic script to trace GSC's grammar transformation.

This patches gsc.py to print the state at different stages.
"""

import sys
sys.path.insert(0, '/home/user/more-gsc')

# Patch PCFG class to add diagnostics
original_file = '/home/user/more-gsc/gsc.py'
with open(original_file, 'r') as f:
    gsc_code = f.read()

# Check if use_hnf is set correctly
print("Checking default use_hnf setting...")
if "self.opts['use_hnf'] = False" in gsc_code:
    print("  ✓ use_hnf = False (default)")
else:
    print("  ✗ use_hnf might not be False!")

print("\nChecking _tokenize_cnf logic...")
if "if not self.opts['use_hnf']:" in gsc_code:
    print("  ✓ _tokenize_cnf checks use_hnf")
else:
    print("  ✗ _tokenize_cnf might not check use_hnf!")

print("\nTo diagnose your issue, add these print statements to gsc.py:")
print("\n1. After line 177 in _cnf2hnf():")
print("   print(f'After HNF: {len(self.rules)} rules')")
print("   unary_count = sum(1 for r in self.rules if r.get(\\'d2\\') is None)")
print("   print(f'  Unary rules: {unary_count}')")

print("\n2. After line 229 in _tokenize_cnf():")
print("   print(f'After tokenize: {len(self.rules)} rules')")
print("   unary_count = sum(1 for r in self.rules if r.get(\\'d2\\') is None)")
print("   print(f'  Unary rules: {unary_count}')")

print("\n3. At the start of _tokenize_cnf():")
print("   print(f'use_hnf = {self.opts[\\'use_hnf\\']}')")

print("\nOr run this simpler test:")
print("-" * 70)
print("""
import gsc

# Simple test grammar
test_grammar = '''
0.5 S -> NP VP
0.5 S -> VP PP
1.0 NP -> DT NN
1.0 VP -> VB NP
1.0 PP -> IN NP
'''

try:
    print("Testing with simple grammar...")
    hg = gsc.HarmonicGrammar(pcfg=test_grammar, root='S', max_sent_len=5)
    print(f"✓ Success! Grammar has {len(hg.g0.rules)} rules")

    # Check for unary rules
    unary_rules = [r for r in hg.g0.rules if r.get('d2') is None]
    print(f"  Unary rules: {len(unary_rules)}")

    if len(unary_rules) > 0:
        print("  Sample unary rule:", unary_rules[0])

except Exception as e:
    print(f"✗ Error: {e}")
    import traceback
    traceback.print_exc()
""")
