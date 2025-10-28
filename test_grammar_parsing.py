"""
Test if the grammar string parses correctly
"""

# First test with original gsc (CPU)
print("="*70)
print("TEST 1: Original gsc (CPU)")
print("="*70)

import gsc

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

try:
    hg = gsc.HarmonicGrammar(pcfg=PCFG_G1, root='S', max_sent_len=5)
    print("✓ Grammar parsed successfully (CPU)")
    print(f"  Number of fillers: {len(hg.filler_names)}")
except Exception as e:
    print(f"✗ FAILED: {e}")
    import traceback
    traceback.print_exc()

# Now test with gsc_gpu
print("\n" + "="*70)
print("TEST 2: gsc_gpu (GPU)")
print("="*70)

# Need to reload to test gsc_gpu
import importlib
import sys

# Remove gsc from modules so we can load gsc_gpu fresh
if 'gsc' in sys.modules:
    del sys.modules['gsc']

import gsc_gpu as gsc_gpu_module

try:
    hg2 = gsc_gpu_module.HarmonicGrammar(pcfg=PCFG_G1, root='S', max_sent_len=5)
    print("✓ Grammar parsed successfully (GPU)")
    print(f"  Number of fillers: {len(hg2.filler_names)}")
except Exception as e:
    print(f"✗ FAILED: {e}")
    import traceback
    traceback.print_exc()

print("\n" + "="*70)
print("DONE")
print("="*70)
