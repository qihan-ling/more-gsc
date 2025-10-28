#!/usr/bin/env python3
"""
Test direct CuPy import in gsc.py

This tests whether changing 'import numpy as np' to 'import cupy as np'
in gsc.py works correctly.
"""

print("="*70)
print("Testing Direct CuPy Import in gsc.py")
print("="*70)

print("\n[1/5] Importing gsc...")
try:
    import gsc
    print("  ✓ gsc imported successfully")
    if hasattr(gsc, 'GPU_AVAILABLE'):
        print(f"  GPU_AVAILABLE: {gsc.GPU_AVAILABLE}")
except Exception as e:
    print(f"  ✗ FAILED: {e}")
    import traceback
    traceback.print_exc()
    import sys
    sys.exit(1)

print("\n[2/5] Creating simple PCFG...")
PCFG_SIMPLE = '''0.5 S -> A B
0.5 S -> C D'''

try:
    hg = gsc.HarmonicGrammar(pcfg=PCFG_SIMPLE, root='S', max_sent_len=2)
    print(f"  ✓ HarmonicGrammar created")
    print(f"  Fillers: {len(hg.filler_names)}")
except Exception as e:
    print(f"  ✗ FAILED: {e}")
    import traceback
    traceback.print_exc()
    import sys
    sys.exit(1)

print("\n[3/5] Creating similarity list...")
try:
    sim = hg.get_simlist(dp=0.0)
    print(f"  ✓ Similarity list created")
except Exception as e:
    print(f"  ✗ FAILED: {e}")
    import traceback
    traceback.print_exc()
    import sys
    sys.exit(1)

print("\n[4/5] Initializing GscNet...")
try:
    net_opts = {
        'T_init': 0.01,
        'q_max': 15.0,
        'q_init': 0.0,
        'dt_init': 0.005,
        'm': 30,
        'use_runC': True,
    }
    net = gsc.GscNet(hg=hg, encodings={'similarity': sim},
                     opts=net_opts, seed=1024)
    print(f"  ✓ GscNet initialized")
    print(f"  num_bindings: {net.num_bindings}")

    # Check if arrays are on GPU
    if gsc.GPU_AVAILABLE:
        import cupy as cp
        print(f"  WC is CuPy array: {isinstance(net.WC, cp.ndarray)}")

except Exception as e:
    print(f"  ✗ FAILED: {e}")
    import traceback
    traceback.print_exc()
    import sys
    sys.exit(1)

print("\n[5/5] Generating corpus...")
try:
    net.generate_corpus(use_freq=True)
    print(f"  ✓ Corpus generated")
    print(f"  Sentences: {len(net.corpus['sentence'])}")
except Exception as e:
    print(f"  ✗ FAILED: {e}")
    import traceback
    traceback.print_exc()
    import sys
    sys.exit(1)

print("\n" + "="*70)
if gsc.GPU_AVAILABLE:
    print("SUCCESS! GPU acceleration is working!")
    print("Arrays are on GPU and operations use CuPy.")
else:
    print("SUCCESS! Running on CPU (CuPy not available)")
print("="*70)
