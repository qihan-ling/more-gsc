"""Debug script to trace treelet construction for role (3,2)"""

import only_gscnet as gsc

import numpy as np


# Load your trained model

print("Loading model...")

#net = gsc.load_model('g1_ds_speedup_model_copy.pkl')
net = gsc.load_model('g1_ds_speedup_model_copy_trainjax.pkl')

print("\n" + "="*70)

print("Debugging treelet construction for role (3,2)")

print("="*70)


# Check what get_daughters returns for role (3,2)

daughters = net.hg.roles.get_daughters('(3,2)')

print(f"\nDaughters of (3,2):")

for key, val in daughters.items():

    print(f"  {key}: {val}")


# Get treelet frame

treelet_frame = gsc.get_treelet_frame(net, '(3,2)')

print(f"\nTreelet frame: {treelet_frame}")


# Get all rules

rules0 = net.hg.g.get_rules()

rules = []

for rule in rules0:

    if rule not in rules:

        rules.append(rule)


print(f"\nTotal unique rules: {len(rules)}")

print("\nFirst 20 rules:")

for i, rule in enumerate(rules[:20]):

    print(f"{i}: {gsc.rule2str(rule, suppress_pos=True)}")


# Get treelets for role (3,2)

treelets = gsc.get_treelets(net, rules, '(3,2)')

print(f"\nTotal treelets at (3,2): {len(treelets)}")

print("\nFirst 10 treelets:")

for i, treelet in enumerate(treelets[:10]):

    print(f"{i}: {treelet}")


# Check binding names that match these treelets

print("\n" + "="*70)

print("Checking if treelet bindings exist in binding_names:")

print("="*70)

for i, treelet in enumerate(treelets[:10]):

    print(f"\nTreelet {i}: {treelet}")

    try:

        idx = net.find_bindings_fast(treelet)

        print(f"  Binding indices: {idx}")

        print(f"  Actual binding names: {[net.binding_names[j] for j in idx]}")

    except Exception as e:

        print(f"  ERROR: {e}")

# Find S1 (N Vi P N)

s1_idx = None

for si, sent in enumerate(net.corpus['sentence']):

    word_seq = ' '.join([bname.split('/')[0] for bname in sent])

    if word_seq == 'N Vi P N':

        s1_idx = si

        break


if s1_idx is None:

    print("ERROR: S1 (N Vi P N) not found!")

    exit(1)


print(f"Found S1 at index {s1_idx}")

sent = net.corpus['sentence'][s1_idx]

targ = net.corpus['target'][s1_idx]


# Run the sentence to generate traces

words = [bname.split('/')[0] for bname in sent]

# Set random seed to match plotting function for reproducibility
np.random.seed(1024 + s1_idx)

net.reset(mu=net.ep, sd=0.01)

net.initialize_traces(trace_list='all')


for wi, word in enumerate(words):

    net.run_word(word, wi + 1, log_trace=True)

net.run_wrapup(log_trace=True)


# Now compute treelet activations for role (3,2)

print("\n" + "="*70)

print("Computing treelet activations for role (3,2)")

print("="*70)


rules0 = net.hg.g.get_rules()

rules = []

for rule in rules0:

    if rule not in rules:

        rules.append(rule)


rname = '(3,2)'

actC_trace = net.traces['actC']

dp_all = gsc.compute_treelet_act_trace(net, actC_trace, rules, rname)


# Find top 4 by total activation

temp = np.argsort(dp_all.sum(axis=0))

focus_idx = temp[::-1][:4]


print(f"\nTop 4 treelets by activation sum:")

for rank, idx in enumerate(focus_idx):

    rule = rules[idx]

    label = gsc.rule2str(rule, suppress_pos=True)

    total_activation = dp_all[:, idx].sum()

    print(f"\n{rank+1}. Rule index {idx}: {label}")

    print(f"   Total activation: {total_activation:.6f}")

    print(f"   Rule details: m={rule['m']}, d1={rule['d1']}, d2={rule['d2']}")

    # Get the treelet bindings

    treelets = gsc.get_treelets(net, rules, rname)

    treelet = treelets[idx]

    print(f"   Treelet: {treelet}")

    # Check binding indices

    try:

        binding_idx = net.find_bindings_fast(treelet)

        print(f"   Binding indices: {binding_idx}")

        print(
            f"   Actual bindings: {[net.binding_names[j] for j in binding_idx]}")

        # Check activations at final timestep

        final_act = actC_trace[-1, binding_idx]

        print(f"   Final activations: {final_act}")

    except Exception as e:

        print(f"   ERROR getting bindings: {e}")


print("\n" + "="*70)

print(
    "Expected from paper: VP[1](*Vi,PP[1]), *Vi(*Vi,), RC[1](*Vpp,PP[1]), VPpp[1](*Vpp,PP[1])")

print("="*70)
