#!/usr/bin/env python
"""
Quick diagnostic to check why speedup isn't working.
Run this to identify the exact issue.
"""

print("="*70)
print("GSCNET SPEEDUP DIAGNOSTIC")
print("="*70)

# Check 1: JAX availability
print("\n1. Checking JAX availability...")
try:
    import jax
    import jax.numpy as jnp
    print(f"   ✓ JAX is installed: {jax.__version__}")
    print(f"   ✓ JAX devices: {jax.devices()}")

    # Check if GPU is available
    if any('cuda' in str(d).lower() or 'gpu' in str(d).lower() for d in jax.devices()):
        print("   ✓ GPU detected!")
        has_gpu = True
    else:
        print("   ✗ No GPU detected - using CPU")
        print("     (JAX on CPU can be slower than NumPy!)")
        has_gpu = False
    JAX_OK = True
except ImportError as e:
    print(f"   ✗ JAX not installed: {e}")
    print("     Install with: pip install jax jaxlib")
    JAX_OK = False
    has_gpu = False

# Check 2: Import GSCNet modules
print("\n2. Importing GSCNet modules...")
try:
    import only_gscnet as gsc_orig
    print("   ✓ only_gscnet.py imported")
    print(f"     JAX_AVAILABLE in original: {gsc_orig.JAX_AVAILABLE}")
except Exception as e:
    print(f"   ✗ Failed to import only_gscnet.py: {e}")
    gsc_orig = None

try:
    import only_gscnet_speedup as gsc_speedup
    print("   ✓ only_gscnet_speedup.py imported")
    print(f"     JAX_AVAILABLE in speedup: {gsc_speedup.JAX_AVAILABLE}")
except Exception as e:
    print(f"   ✗ Failed to import only_gscnet_speedup.py: {e}")
    gsc_speedup = None

# Check 3: Compare implementations
print("\n3. Comparing implementations...")
if gsc_orig and gsc_speedup:
    if gsc_orig.JAX_AVAILABLE == gsc_speedup.JAX_AVAILABLE:
        if gsc_orig.JAX_AVAILABLE:
            print("   ℹ Both files will use JAX (same baseline)")
        else:
            print("   ℹ Both files will use NumPy (same baseline)")

    # Check for S matrix
    import inspect

    # Check original
    orig_src = inspect.getsource(gsc_orig.GscNet._add_change_of_basis_matrices)
    if 'self.S = self.C.dot(self.C.T)' in orig_src or 'self.S = C.dot(C.T)' in orig_src:
        print("   • Original: Creates S matrix (C @ C.T)")
    else:
        print("   • Original: Uses lazy S multiplication")

    # Check speedup
    speedup_src = inspect.getsource(gsc_speedup.GscNet._add_change_of_basis_matrices)
    if 'self.S = self.C.dot(self.C.T)' in speedup_src or 'self.S = C.dot(C.T)' in speedup_src:
        print("   • Speedup: Creates S matrix (C @ C.T)")
    else:
        print("   • Speedup: Uses lazy S multiplication")
else:
    print("   ✗ Cannot compare (import failed)")

# Check 4: Network size for G1 grammar
print("\n4. Analyzing G1 grammar network size...")
if gsc_speedup:
    try:
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
        hg = gsc_speedup.HarmonicGrammar(pcfg=PCFG_G1, root='S', max_sent_len=5)
        print(f"   • Fillers: {len(hg.filler_names)}")
        print(f"   • Roles: {len(hg.role_names)}")
        print(f"   • Bindings: {len(hg.filler_names) * len(hg.role_names)}")

        num_bindings = len(hg.filler_names) * len(hg.role_names)
        s_size_mb = num_bindings * num_bindings * 4 / 1024**2
        print(f"   • S matrix size: {s_size_mb:.2f} MB")

        if s_size_mb < 10:
            print(f"   ℹ Network is SMALL - S matrix optimization negligible")
        elif s_size_mb < 100:
            print(f"   ℹ Network is MEDIUM - S matrix optimization minor")
        else:
            print(f"   ✓ Network is LARGE - S matrix optimization significant")
    except Exception as e:
        print(f"   ✗ Failed to analyze grammar: {e}")

# Check 5: Test actual speedup
print("\n5. Testing actual speedup (if possible)...")
if gsc_speedup and JAX_OK:
    try:
        import time
        import numpy as np

        # Create tiny test network
        PCFG_test = "0.5 S -> A B\n0.5 S -> B A"
        hg = gsc_speedup.HarmonicGrammar(pcfg=PCFG_test, root='S', max_sent_len=2)
        sim = hg.get_simlist(dp=0.0)
        net = gsc_speedup.GscNet(hg=hg, encodings={'similarity': sim}, seed=1024)

        print(f"   • Network use_jax flag: {net.use_jax}")
        print(f"   • Network use_runC flag: {net.opts['use_runC']}")
        print(f"   • Network T_decay_rate: {net.opts.get('T_decay_rate', 'not set')}")

        # Initialize for training
        train_opts = {
            'lrate': 0.1,
            'num_trials': 4,
            'ema_stat_weight': 0.0,
            'init_noise_mag': 0.02,
        }
        net.initialize(train_opts=train_opts)
        net.generate_corpus(use_freq=True)

        # Test 1: run_word with log_trace=False (should use JAX fast path)
        print("\n   Testing run_word with log_trace=False:")
        net.reset(mu=net.ep, sd=0.01)
        net.qpolicy = np.linspace(0, net.opts['q_max'], 3)

        t0 = time.time()
        net.run_word('A', 1, log_trace=False)
        t_no_trace = time.time() - t0
        print(f"     Time: {t_no_trace:.4f}s")

        # Test 2: run_word with log_trace=True (should NOT use JAX fast path)
        print("\n   Testing run_word with log_trace=True:")
        net.reset(mu=net.ep, sd=0.01)

        t0 = time.time()
        net.run_word('A', 1, log_trace=True)
        t_with_trace = time.time() - t0
        print(f"     Time: {t_with_trace:.4f}s")

        if t_no_trace > 0:
            speedup_ratio = t_with_trace / t_no_trace
            print(f"     Ratio (with_trace/without_trace): {speedup_ratio:.2f}x")

            if speedup_ratio < 1.2:
                print("     ✗ No speedup detected - JAX fast path likely not being used")
            elif speedup_ratio < 2.0:
                print("     ~ Minor speedup - JAX may be on CPU")
            else:
                print("     ✓ Significant speedup - JAX fast path working!")

        # Test 3: estimate_prob_inc_jax
        print("\n   Testing estimate_prob_inc_jax (training function):")
        t0 = time.time()
        stat = net.estimate_prob_inc_jax(prefix=[], num_trials=4)
        t_jax = time.time() - t0
        print(f"     Time for 4 trials: {t_jax:.4f}s ({t_jax/4:.4f}s per trial)")

        if t_jax/4 < 0.01:
            print("     ✓ Very fast - GPU likely being used")
        elif t_jax/4 < 0.1:
            print("     ~ Moderate speed - possible CPU JAX")
        else:
            print("     ✗ Slow - likely CPU NumPy fallback")

    except Exception as e:
        print(f"   ✗ Failed to test speedup: {e}")
        import traceback
        traceback.print_exc()
else:
    print("   ⊘ Skipped (JAX not available or import failed)")

# Final diagnosis
print("\n" + "="*70)
print("DIAGNOSIS SUMMARY")
print("="*70)

if not JAX_OK:
    print("\n🔴 PRIMARY ISSUE: JAX is not installed or not working")
    print("   → Both scripts will use NumPy (no speedup expected)")
    print("   → Install JAX with: pip install jax jaxlib")
    if gsc_orig and gsc_speedup:
        print("   → Both only_gscnet.py and only_gscnet_speedup.py have same JAX code")
        print("   → They will perform identically without JAX")
elif not has_gpu:
    print("\n🟡 WARNING: JAX is using CPU, not GPU")
    print("   → Speedup may be minimal or negative")
    print("   → JAX on CPU can be slower than NumPy due to overhead")
    print("   → Install JAX with GPU support: pip install jax[cuda12]")
else:
    print("\n🟢 JAX is installed with GPU support")
    print("   → Speedup should be significant (5-20x for training)")
    print("   → If both scripts take same time, possible issues:")
    print("     • JIT compilation overhead (first run is slow)")
    print("     • Network too small to benefit from GPU")
    print("     • Check that log_trace=False during training")

if gsc_speedup:
    print("\nKEY INSIGHT:")
    print("   • Both only_gscnet.py and only_gscnet_speedup.py use JAX if available")
    print("   • Main difference: lazy S multiplication (only helps for large networks)")
    print("   • For G1 grammar (405 bindings), S matrix is only 0.63 MB")
    print("   • Expected speedup from lazy S: ~0% for this network size")
    print("   • Expected speedup from JAX+GPU: 5-20x (if working)")

print("\nNEXT STEPS:")
if not JAX_OK:
    print("   1. Install JAX: pip install jax jaxlib")
    print("   2. Re-run this diagnostic")
elif not has_gpu:
    print("   1. Install JAX with GPU: pip install jax[cuda12]")
    print("   2. Or use a larger grammar to see CPU speedup")
else:
    print("   1. Run training and check initial output for 'JAX detected'")
    print("   2. Compare first vs. second run (JIT compilation)")
    print("   3. Profile to identify actual bottleneck")
    print("   4. Try larger grammar to see more pronounced speedup")

print("="*70)
