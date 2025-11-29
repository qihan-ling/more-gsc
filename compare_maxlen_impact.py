import only_gscnet_speedup_sap as gsc
import numpy as np

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

print("="*70)
print("COMPARING DIFFERENT MAX_SENT_LEN VALUES")
print("="*70)

for MAXLEN in [5, 10, 15, 20, 24]:
    print(f"\n{'='*70}")
    print(f"max_sent_len = {MAXLEN}")
    print(f"{'='*70}")

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

    print(f"  num_fillers: {len(net.filler_names)}")
    print(f"  num_roles: {net.num_roles}")
    print(f"  num_bindings: {net.num_bindings}")
    print(f"  WC elements: {net.num_bindings**2:,}")
    print(f"  WC memory: {net.num_bindings**2 * 8 / 1e6:.1f} MB")

    # Estimate WC.dot() time based on size
    if net.num_bindings < 1000:
        est_dot = "< 1ms"
    elif net.num_bindings < 3000:
        est_dot = "1-10ms"
    elif net.num_bindings < 6000:
        est_dot = "10-50ms"
    else:
        est_dot = "50-200ms"

    print(f"  Est. WC.dot() time: {est_dot}")

    # Calculate expected run_wrapup time
    num_steps = int(15.0 / 0.005)  # q_max=15, q_rate=1.0, dt=0.005

    if net.num_bindings < 1000:
        trial_time = num_steps * 0.001
    elif net.num_bindings < 3000:
        trial_time = num_steps * 0.005
    elif net.num_bindings < 6000:
        trial_time = num_steps * 0.03
    else:
        trial_time = num_steps * 0.15

    print(f"  Est. trial time: {trial_time:.1f}s")
    print(f"  Est. epoch (4 trials): {trial_time*4/60:.1f} min")

print("\n" + "="*70)
print("RECOMMENDATION")
print("="*70)
print("\nFor the 10-rule toy grammar, max_sent_len=24 is excessive!")
print("The grammar can only generate short sentences.")
print("\nSuggested max_sent_len values:")
print("  - max_sent_len=5:  Fast (~1s per epoch)")
print("  - max_sent_len=10: Moderate (~30s per epoch)")
print("  - max_sent_len=15: Slow (~3 min per epoch)")
print("  - max_sent_len=24: Very slow (~40 min per epoch)")
print("\nFor testing, use max_sent_len=5 or 7!")
print("="*70)
