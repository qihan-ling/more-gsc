#!/usr/bin/env python3
"""
Test that the collapsed grammar can be loaded by GSC.
"""

import gsc

# Test with a small subset of the collapsed grammar
TEST_GRAMMAR = '''
4.9600640933 S -> NP VP
3.5632482695 S -> VP
12.1018326731 PP -> IN NP
9.7807213630 ADVP -> RB
'''

ROOT = 'S'
MAXLEN = 10

try:
    print("Testing collapsed grammar with GSC...")
    hg = gsc.HarmonicGrammar(pcfg=TEST_GRAMMAR, root=ROOT, max_sent_len=MAXLEN)
    print("✓ Grammar loaded successfully!")
    print(f"  Number of fillers: {len(hg.filler_names)}")
    print(f"  Filler names (first 10): {hg.filler_names[:10]}")
except Exception as e:
    print(f"✗ Error loading grammar: {e}")
    import traceback
    traceback.print_exc()
