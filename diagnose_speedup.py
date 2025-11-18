"""
Diagnostic script to identify why JAX speedup isn't working
"""
import only_gscnet_speedup as gsc
import numpy as np
import time

# Check JAX availability
print("="*70)
print("1. JAX AVAILABILITY CHECK")
print("="*70)
try:
    import jax
    import jax.numpy as jnp
    print(f"✓ JAX is available: {jax.__version__}")
    print(f"✓ JAX devices: {jax.devices()}")
    JAX_AVAILABLE = True
except ImportError as e:
    print(f"✗ JAX not available: {e}")
    JAX_AVAILABLE = False

print("\n" + "="*70)
print("2. NETWORK INITIALIZATION CHECK")
print("="*70)

# Simple grammar for testing
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

ROOT = 'S'
MAXLEN = 5

hg = gsc.HarmonicGrammar(pcfg=PCFG_G1, root=ROOT, max_sent_len=MAXLEN)
sim = hg.get_simlist(dp=0.0)

net_opts = {
    'T_init': 0.01,
    'q_max': 15.0,
    'q_init': 0.0,
    'dt_init': 0.005,
    'm': 30,
    'use_runC': True,  # CHECK THIS
}

net = gsc.GscNet(hg=hg, encodings={'similarity': sim},
                 opts=net_opts, seed=1024)

print(f"Network use_jax flag: {net.use_jax}")
print(f"Network use_runC flag: {net.opts['use_runC']}")
print(f"Network T_decay_rate: {net.opts.get('T_decay_rate', 'not set')}")

# Check if parameters are JAX arrays
if JAX_AVAILABLE and net.use_jax:
    print(f"WC type: {type(net.WC)}")
    print(f"bC type: {type(net.bC)}")
    print(f"estr type: {type(net.estr)}")
else:
    print("JAX arrays not initialized")

print("\n" + "="*70)
print("3. DYNAMICS EXECUTION TEST")
print("="*70)

# Initialize for training
train_opts = {
    'lrate': 0.1,
    'num_trials': 4,
    'ema_stat_weight': 0.0,
    'trace_varnames': ['kl_trees', 'kl_treelets', 'prob_sent', 'acc'],
    'report_cycle': 10,
    'init_noise_mag': 0.02,
    'average_weight': False,
    'average_filler_bias': False,
}
net.initialize(train_opts=train_opts)
net.generate_corpus(use_freq=True)

# Test run_word with log_trace=False (should use JAX fast path)
print("\nTest 1: run_word with log_trace=False")
net.reset(mu=net.ep, sd=0.01)
net.qpolicy = np.linspace(0, net.opts['q_max'], net.hg.opts['max_sent_len'] + 1)

t0 = time.time()
net.run_word('N', 1, log_trace=False)
t1 = time.time() - t0

# Check if JAX fast path was used
print(f"  Time: {t1:.4f}s")
print(f"  use_jax: {net.use_jax}")
print(f"  use_runC: {net.opts['use_runC']}")
print(f"  T_decay_rate: {net.opts.get('T_decay_rate', 0)}")

# Test run_word with log_trace=True (should NOT use JAX fast path)
print("\nTest 2: run_word with log_trace=True")
net.reset(mu=net.ep, sd=0.01)

t0 = time.time()
net.run_word('N', 1, log_trace=True)
t2 = time.time() - t0

print(f"  Time: {t2:.4f}s")
print(f"  Speed difference: {t2/t1:.2f}x slower with log_trace=True")

print("\n" + "="*70)
print("4. TRAINING EXECUTION TEST")
print("="*70)

# Test estimate_prob_inc_jax
print("\nTesting estimate_prob_inc_jax (used during training):")
t0 = time.time()
stat_Q = net.estimate_prob_inc_jax(prefix=[], num_trials=4)
t3 = time.time() - t0
print(f"  Time for 4 trials: {t3:.4f}s")
print(f"  Time per trial: {t3/4:.4f}s")

print("\n" + "="*70)
print("5. DIAGNOSIS SUMMARY")
print("="*70)

issues = []

if not JAX_AVAILABLE:
    issues.append("✗ JAX is not installed or not working")
elif not net.use_jax:
    issues.append("✗ Network has use_jax=False (JAX disabled)")
else:
    print("✓ JAX is available and enabled")

if net.opts.get('T_decay_rate', 0) > 0:
    issues.append(f"✗ T_decay_rate={net.opts['T_decay_rate']} > 0 (prevents JAX fast path)")
else:
    print("✓ T_decay_rate <= 0 (allows JAX fast path)")

if not net.opts.get('use_runC', False):
    issues.append("✗ use_runC=False (runC not being called)")
else:
    print("✓ use_runC=True (runC will be called)")

print("\nPotential issues:")
if issues:
    for issue in issues:
        print(f"  {issue}")
else:
    print("  No obvious configuration issues found")

print("\n" + "="*70)
print("EXPECTED SPEEDUP LOCATIONS:")
print("="*70)
print("""
1. TRAINING LOOP (train2):
   - Uses estimate_prob_inc_jax() which runs trials in parallel on GPU
   - Should see massive speedup here if JAX is working

2. EQUILIBRIUM FINDING (runC with log_trace=False):
   - Uses JIT-compiled dynamics_step_jax via jax.lax.fori_loop
   - Lazy S matrix multiplication (C @ C.T @ v instead of materializing S)
   - Should be much faster than Python loop

3. GRADIENT COMPUTATION:
   - JAX arrays allow for potential GPU acceleration of matrix operations
   - Adam optimizer can use JIT-compiled updates

WHY SPEEDUP MIGHT NOT WORK:

1. log_trace=True disables JAX fast path in runC:
   - The condition at only_gscnet_speedup.py:2102 requires log_trace=False
   - When generating plots (cho_grammar1_new_copy.py:253,256), log_trace=True
   - But during training, estimate_prob_inc_jax uses its own fast path

2. Training uses estimate_prob_inc_jax which should be fast:
   - Bypasses run_word/run_wrapup entirely
   - Uses _run_single_trial_jax with vmap for parallelization
   - This should give huge speedup during training

3. If training is still slow, possible issues:
   - JAX not properly installed (check above)
   - GPU not available (check jax.devices() above)
   - JIT compilation overhead on first run
   - Data transfer between CPU/GPU
   - Network too small to benefit from parallelization
""")
