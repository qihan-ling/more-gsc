#!/usr/bin/env python3
"""
Debug script to investigate S4 (N Vpp P N Vi) parsing failures.
"""
import only_gscnet as gsc
import numpy as np

# Load trained model
print("Loading model...")
net = gsc.load_model('g1_ds_speedup_model_copy.pkl')

# Find S4 in corpus
s4_idx = None
for si, sent in enumerate(net.corpus['sentence']):
    word_seq = ' '.join([bname.split('/')[0] for bname in sent])
    if word_seq == 'N Vpp P N Vi':
        s4_idx = si
        break

if s4_idx is None:
    print("ERROR: S4 not found!")
    exit(1)

print(f"\n{'='*70}")
print(f"S4 Analysis: Index {s4_idx}")
print(f"{'='*70}")

sent = net.corpus['sentence'][s4_idx]
targ = net.corpus['target'][s4_idx]

# Display sentence info
word_seq = ' '.join([bname.split('/')[0] for bname in sent])
print(f"\nSentence: {word_seq}")
print(f"Full bindings: {sent}")

# Display target parse
print(f"\nTarget parse (correct structure):")
targ_bindings = [net.binding_names[i] for i in range(len(net.binding_names)) if targ[i] > 0.5]
print(f"  Active bindings ({len(targ_bindings)}): {targ_bindings[:10]}...")  # Show first 10

# Expected structure: S -> NP Vi, NP -> N RC, RC -> Vpp PP, PP -> P N
print(f"\nExpected structure:")
print(f"  S -> NP Vi (prob 0.05 - rarest structure!)")
print(f"  NP -> N RC")
print(f"  RC -> Vpp PP")
print(f"  PP -> P N")

# Test at different commitment levels
print(f"\n{'='*70}")
print(f"Testing S4 at different commitment levels")
print(f"{'='*70}")

commitment_levels = [1, 3, 5, 7, 10, 12]
max_sent_len = net.hg.opts['max_sent_len']

# Remove empty fillers
f_empty_type = net.hg.g.get_types(net.hg.opts['f_empty'])
sent0 = [bname for bname in sent if bname.split('/')[0] not in f_empty_type]

print(f"\nSentence without empty fillers: {sent0}")
print(f"Sentence length: {len(sent0)} words")

for t in commitment_levels:
    print(f"\n{'-'*70}")
    print(f"Commitment level t={t}")
    print(f"{'-'*70}")

    # Create commitment policy
    dq = np.ones(max_sent_len) * (float(t) / max_sent_len)
    qpolicy = dq.cumsum()
    qpolicy = np.insert(qpolicy, 0, 0.)

    print(f"dq = {dq}")
    print(f"qpolicy = {qpolicy}")
    print(f"Total commitment for S4 ({len(sent0)} words): {qpolicy[len(sent0)]:.2f}")

    # Set seed and run 3 trials to see consistency
    np.random.seed(2048 + t)

    success_count = 0
    failed_parses = []

    for trial in range(3):
        # Run parsing
        net.qpolicy = qpolicy
        net.reset(mu=net.ep, sd=0.02)

        # Track initial state
        initial_actC = net.actC.copy()

        # Run sentence
        for ii, bname in enumerate(sent0):
            net.extC *= 0.5  # decay_factor from test_parse_inc
            net.set_input(bname, use_type=True, cumulative=True)

            # Calculate runtime for this word
            word_rt = (qpolicy[ii + 1] - qpolicy[ii]) / net.opts['q_rate']

            if net.opts['use_runC']:
                net.runC(word_rt)
            else:
                net.run(word_rt)

        # Wrapup
        net.run_wrapup()

        # Get final state
        final_actC = net.actC.copy()

        # Discretize
        net.set_discrete_state(net.read_grid_point())

        # Check if correct
        is_correct = np.allclose(net.actC, targ)

        if is_correct:
            success_count += 1
            print(f"  Trial {trial}: ✓ CORRECT")
        else:
            print(f"  Trial {trial}: ✗ WRONG")
            failed_parses.append(net.actC.copy())

            # Show what it converged to
            wrong_bindings = [net.binding_names[i] for i in range(len(net.binding_names))
                            if net.actC[i] > 0.5]
            print(f"    Converged to ({len(wrong_bindings)} bindings):")

            # Show first few wrong bindings
            for wb in wrong_bindings[:8]:
                print(f"      {wb}")

            # Identify mismatches
            target_set = set([net.binding_names[i] for i in range(len(net.binding_names)) if targ[i] > 0.5])
            wrong_set = set(wrong_bindings)

            missing = target_set - wrong_set
            extra = wrong_set - target_set

            if missing:
                print(f"    Missing bindings ({len(missing)}): {list(missing)[:5]}...")
            if extra:
                print(f"    Extra bindings ({len(extra)}): {list(extra)[:5]}...")

    print(f"\n  Accuracy for t={t}: {success_count}/3 = {success_count/3:.2f}")

# Compare S4 with S1 (which works well)
print(f"\n{'='*70}")
print(f"Comparison: S4 vs S1")
print(f"{'='*70}")

s1_idx = None
for si, sent in enumerate(net.corpus['sentence']):
    word_seq = ' '.join([bname.split('/')[0] for bname in sent])
    if word_seq == 'N Vi P N':
        s1_idx = si
        break

s1_sent = net.corpus['sentence'][s1_idx]
s1_targ = net.corpus['target'][s1_idx]
s1_sent0 = [bname for bname in s1_sent if bname.split('/')[0] not in f_empty_type]

print(f"\nS1: {' '.join([bname.split('/')[0] for bname in s1_sent0])}")
print(f"  Length: {len(s1_sent0)} words")
print(f"  Structure: S -> N VP, VP -> Vi PP, PP -> P N")
print(f"  Grammar prob: N (terminal), VP[1]=0.5, PP=1.0")

print(f"\nS4: {' '.join([bname.split('/')[0] for bname in sent0])}")
print(f"  Length: {len(sent0)} words")
print(f"  Structure: S -> NP Vi, NP -> N RC, RC -> Vpp PP, PP -> P N")
print(f"  Grammar prob: NP[2]=0.05 (!), RC=1.0, VPpp=1.0, PP=1.0")

print(f"\nKey difference: S4 uses S->NP Vi (5% prob) vs S1 uses S->N VP (60% prob)")
print(f"S4's rare structure may have weaker attractor basin!")

# Check energy landscape around S4's target
print(f"\n{'='*70}")
print(f"Energy Analysis at t=5 (where S4 starts failing)")
print(f"{'='*70}")

t = 5
dq = np.ones(max_sent_len) * (float(t) / max_sent_len)
qpolicy = dq.cumsum()
qpolicy = np.insert(qpolicy, 0, 0.)
net.qpolicy = qpolicy

np.random.seed(2048 + t)
net.reset(mu=net.ep, sd=0.02)

print(f"\nInitial state statistics:")
print(f"  actC mean: {net.actC.mean():.6f}")
print(f"  actC std: {net.actC.std():.6f}")
print(f"  actC max: {net.actC.max():.6f}")
print(f"  actC min: {net.actC.min():.6f}")

# Distance from initial state to target
initial_to_target = np.linalg.norm(net.actC - targ)
print(f"  Initial distance to S4 target: {initial_to_target:.4f}")

# Also check distance to S1 target (for comparison)
initial_to_s1_target = np.linalg.norm(net.actC - s1_targ)
print(f"  Initial distance to S1 target: {initial_to_s1_target:.4f}")

print(f"\nConclusion: If S4 target is much farther, noise+input may not reach it.")
