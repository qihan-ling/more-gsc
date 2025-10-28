"""
Diagnostic script to test GPU setup without running full training

This script tests each component step-by-step to identify issues.
"""

import sys

print("="*70)
print("GPU DIAGNOSTIC TEST")
print("="*70)

# Test 1: Import CuPy
print("\n[1/6] Testing CuPy import...")
try:
    import cupy as cp
    print(f"  ✓ CuPy imported successfully")
    print(f"  Device: {cp.cuda.Device()}")
    print(f"  Memory: {cp.cuda.Device().mem_info[1] / 1e9:.2f} GB")
except Exception as e:
    print(f"  ✗ FAILED: {e}")
    sys.exit(1)

# Test 2: Basic CuPy operations
print("\n[2/6] Testing basic CuPy operations...")
try:
    a = cp.array([1, 2, 3])
    b = cp.array([4, 5, 6])
    c = cp.dot(a, b)
    result = cp.asnumpy(c)
    assert result == 32, f"Expected 32, got {result}"
    print(f"  ✓ Basic operations work (dot product = {result})")
except Exception as e:
    print(f"  ✗ FAILED: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# Test 3: Import gsc_gpu
print("\n[3/6] Testing gsc_gpu import...")
try:
    import gsc_gpu as gsc
    print(f"  ✓ gsc_gpu imported")
    print(f"  GPU_AVAILABLE: {gsc.GPU_AVAILABLE}")
except Exception as e:
    print(f"  ✗ FAILED: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# Test 4: Create simple PCFG
print("\n[4/6] Testing HarmonicGrammar initialization...")
try:
    PCFG_SIMPLE = '''
    0.5 S -> A B
    0.5 S -> C D
    '''
    hg = gsc.HarmonicGrammar(pcfg=PCFG_SIMPLE, root='S', max_sent_len=2)
    print(f"  ✓ HarmonicGrammar created")
    print(f"  Fillers: {len(hg.filler_names)}")
except Exception as e:
    print(f"  ✗ FAILED: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# Test 5: Get similarity list
print("\n[5/6] Testing similarity list...")
try:
    sim = hg.get_simlist(dp=0.0)
    print(f"  ✓ Similarity list created")
    print(f"  Type: {type(sim)}")
except Exception as e:
    print(f"  ✗ FAILED: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# Test 6: Initialize GscNet
print("\n[6/6] Testing GscNet initialization...")
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
    print(f"  WC shape: {net.WC.shape}")
    print(f"  WC type: {type(net.WC)}")
except Exception as e:
    print(f"  ✗ FAILED: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# Test 7: Test corpus generation
print("\n[7/7] Testing corpus generation...")
try:
    net.generate_corpus(use_freq=True)
    print(f"  ✓ Corpus generated")
    print(f"  Number of sentences: {len(net.corpus['sentence'])}")
except Exception as e:
    print(f"  ✗ FAILED: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

print("\n" + "="*70)
print("ALL TESTS PASSED!")
print("="*70)
print("\nGPU setup is working correctly.")
print("You can now run: python cho_grammar1_gpu.py")
