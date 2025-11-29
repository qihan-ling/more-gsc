import only_gscnet_speedup_sap as gsc
import numpy as np
import time

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
MAXLEN = 24

print("="*70)
print("DIAGNOSING run_wrapup() SLOWNESS")
print("="*70)

hg = gsc.HarmonicGrammar(pcfg=PCFG_G1, root=ROOT, max_sent_len=MAXLEN)
sim = hg.get_simlist(dp=0.0)

net_opts = {
    'use_jax': False,
    'T_init': 0.01,
    'q_max': 15.0,
    'q_init': 0.0,
    'dt_init': 0.005,
    'm': 30,
    'use_runC': True,
    'ep_method': 'integration',
}

encodings = {'similarity': sim}
net = gsc.GscNet(hg=hg, encodings=encodings, opts=net_opts, seed=1024)

print(f"\nNetwork dimensions:")
print(f"  num_bindings: {net.num_bindings}")
print(f"  WC shape: {net.WC.shape}")
print(f"  WC elements: {net.WC.shape[0] * net.WC.shape[1]:,}")

print(f"\nDynamics parameters:")
print(f"  q_max: {net.opts['q_max']}")
print(f"  q_rate: {net.opts['q_rate']}")
print(f"  dt: {net.dt}")

# Calculate expected iterations
dur = net.opts['q_max']
duration = dur / net.opts['q_rate']
num_steps = int(duration / net.dt)

print(f"\nExpected iterations in run_wrapup():")
print(f"  duration = q_max / q_rate = {dur} / {net.opts['q_rate']} = {duration}")
print(f"  num_steps = duration / dt = {duration} / {net.dt} = {num_steps:,}")

print(f"\nTiming single update_stateC() call...")
net.generate_corpus(use_freq=True, nsamples=100)
net.initialize(train_opts={'lrate': 0.1, 'num_trials': 4})
net.reset(mu=net.ep, sd=0.02)

# Time one update_stateC call
t0 = time.time()
net.update_stateC()
t_single = time.time() - t0

print(f"  Single update_stateC(): {t_single*1000:.2f}ms")
print(f"\nProjected run_wrapup() time:")
print(f"  {num_steps:,} × {t_single*1000:.2f}ms = {num_steps * t_single:.1f}s")

if num_steps * t_single > 100:
    print(f"\n  ❌ PROBLEM: {num_steps:,} iterations is too many!")
    print(f"  With {t_single*1000:.1f}ms per iteration, this takes {num_steps * t_single:.0f}s")
    print(f"\n  Possible causes:")
    print(f"  1. WC.dot(actC) is slow for large dense matrix")
    print(f"  2. dt is too small ({net.dt}) → too many steps")
    print(f"  3. Matrix operations not optimized")
