#!/usr/bin/env python3
"""
Simplest possible test - just try to parse the grammar with regular gsc
"""

import gsc

PCFG_G1 = '''0.35 S -> N Vi
0.60 S -> N VP
0.05 S -> NP Vi
1.0 NP -> N RC
1.0 RC -> Vpp PP
1.0 VPpp -> Vpp PP
1.0 PP -> P N
0.5 VP -> Vi PP
0.3 VP -> BE Vpp
0.2 VP -> BE VPpp'''

print("Testing grammar parsing...")
try:
    hg = gsc.HarmonicGrammar(pcfg=PCFG_G1, root='S', max_sent_len=5)
    print(f"SUCCESS! Fillers: {len(hg.filler_names)}")
except Exception as e:
    print(f"FAILED: {e}")
    import traceback
    traceback.print_exc()
