import matplotlib.pyplot as plt
import gsc
import numpy as np

# ============================================================================
# Grammar 1 (G1) from Section 4.1
# ============================================================================

PCFG_G1 = '''
0.35 S -> N Vi
0.30 S -> N Vi PP      # 0.60 * 0.5
0.18 S -> N BE Vpp    # 0.60 * 0.3
0.12 S -> N BE VPpp   # 0.60 * 0.2
0.05 S -> NP Vi

1.0 NP -> N RC
1.0 RC -> Vpp PP
1.0 VPpp -> Vpp PP
1.0 PP -> P N
'''

ROOT = 'S'
MAXLEN = 5

# ============================================================================
# Initialize network with paper's specifications
# ============================================================================

hg = gsc.HarmonicGrammar(pcfg=PCFG_G1, root=ROOT, max_sent_len=MAXLEN)

# Display fillers (should have 27 fillers × 15 roles = 405 units)
print(f"Filler names: {hg.filler_names}")
print(f"Number of fillers: {len(hg.filler_names)}")

# Set all filler similarities to 0 (linear independence)
sim = hg.get_simlist(dp=0.0)

# Network options matching paper's parameters
net_opts = {
    'T_init': 0.01,      # computational temperature
    'q_max': 15.0,       # maximum commitment
    'q_init': 0.0,       # initial commitment (FIXED: was 'q_0')
    'dt_init': 0.005,    # time step (FIXED: was 'dt')
    'm': 30,             # resource constraint (Hq1 strength)
    'use_runC': True,    # use C implementation for speed
}

# Initialize network
net = gsc.GscNet(hg=hg, encodings={'similarity': sim},
                 opts=net_opts, seed=1024)
net.generate_corpus(use_freq=True)

# Display target probabilities
print("\n" + "="*70)
print("Target sentence probabilities:")
for si, sent in enumerate(net.corpus['sentence']):
    sent_str = ' '.join([bname.split('/')[0] for bname in sent])
    prob = net.corpus['prob_sent'][si]
    print(f"Sentence {si}: p = {prob:.4f} ({sent_str})")

# ============================================================================
# Training setup (matching Section 4 parameters)
# ============================================================================

train_opts = {
    'lrate': 0.1,                  # learning rate
    'num_trials': 4,               # production trials per iteration
    'ema_stat_weight': 0.0,        # no EMA smoothing initially
    'trace_varnames': ['kl_trees', 'kl_treelets', 'prob_sent', 'acc'],
    'report_cycle': 10,            # report every 10 iterations
    'init_noise_mag': 0.02,
    'average_weight': False,
    'average_filler_bias': False,
}

net.initialize(train_opts=train_opts)

# ============================================================================
# Training loop for Figure 11
# ============================================================================

print("\n" + "="*70)
print("Training Grammar 1...")
print("="*70)

n_epochs = 1000  # Train for sufficient epochs to reach convergence

for epoch_block in range(n_epochs // 10):
    net.train2(
        train_opts={'num_epochs': 10},
        savefilename='g1_model.pkl'
    )

print("\n" + "="*70)
print("Training complete!")

# Calculate final statistics (last 100 updates)
# FIXED: Changed 'trace_train' to 'traces_train'
final_kl = np.mean(net.traces_train['kl_trees'][-100:])
final_kl_sd = np.std(net.traces_train['kl_trees'][-100:])
final_acc = np.mean(net.traces_train['acc'][-100:])
final_acc_sd = np.std(net.traces_train['acc'][-100:])

print(f"Final KL divergence: {final_kl:.3f} (SD = {final_kl_sd:.3f})")
print(f"Final production accuracy: {final_acc:.3f} (SD = {final_acc_sd:.3f})")

# Display final learned probabilities
print("\nFinal learned probabilities Q(S):")
final_probs = np.mean(net.traces_train['prob_sent'][-100:], axis=0)
for si, prob in enumerate(final_probs):
    print(f"Sentence {si}: Q = {prob:.3f}")

# ============================================================================
# Plot Figure 11 (Training dynamics)
# ============================================================================

net = gsc.load_model('g1_model.pkl')

# FIXED: plot_train_result() doesn't return a figure object and calls plt.show()
# internally, so we just call it directly
print("\n" + "="*70)
print("Generating training plots (Figure 11)...")
print("Note: plot_train_result() will display 3 separate plots")
print("="*70)
gsc.plot_train_result(net, legend=True, linewidth=1.5)

# ============================================================================
# Parsing tests for Figure 12
# ============================================================================

print("\n" + "="*70)
print("Testing parsing accuracy (Figure 12)...")
print("="*70)

# Helper function to extract word sequence from binding names
def get_word_sequence(sent):
    """Extract word types from binding names (e.g., 'N/(1,1)' -> 'N')"""
    return ' '.join([bname.split('/')[0] for bname in sent])

# Test parsing at different commitment levels (t ∈ {1, 2, ..., 12})
commitment_levels = list(range(1, 13))
num_sentences = len(net.corpus['sentence'])

# Track accuracy for each sentence separately
# parsing_accuracy_per_sent[si] will contain accuracies across all commitment levels for sentence si
parsing_accuracy_per_sent = {si: [] for si in range(num_sentences)}

for t in commitment_levels:
    # Create commitment policy: use fixed commitment per word
    # For simplicity, use uniform commitment across all word positions
    max_sent_len = net.hg.opts['max_sent_len']
    dq = np.ones(max_sent_len) * (float(t) / max_sent_len)

    # Test parsing using gsc.test_parse_inc
    try:
        parse_results = gsc.test_parse_inc(
            net,
            dq=dq,
            num_trials=10,
            estr=2,
            estr_null=2,
            disp=False
        )

        # Track accuracy for each sentence separately
        for si in range(num_sentences):
            if si in parse_results:
                acc_si = parse_results[si]['acc']
            else:
                acc_si = 0.0
            parsing_accuracy_per_sent[si].append(acc_si)

        # Calculate overall accuracy for display
        n_correct = sum([parse_results[si]['acc'] for si in parse_results])
        n_total = len(parse_results)
        acc_overall = n_correct / n_total if n_total > 0 else 0.0

    except Exception as e:
        print(f"  Warning: Parsing test failed at t={t}: {e}")
        # Append 0.0 for all sentences if parsing failed
        for si in range(num_sentences):
            parsing_accuracy_per_sent[si].append(0.0)
        acc_overall = 0.0

    print(f"Commitment t={t:2d}: Overall accuracy = {acc_overall:.3f}")
    # Also print per-sentence accuracies
    for si in range(num_sentences):
        word_seq = get_word_sequence(net.corpus['sentence'][si])
        print(f"  S{si} ({word_seq}): {parsing_accuracy_per_sent[si][-1]:.3f}")

# Plot Figure 12 - One line per sentence type
plt.figure(figsize=(10, 6))

# Define colors and markers for each sentence
colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd']  # 5 distinct colors
markers = ['o', 's', '^', 'D', 'v']  # Different markers for each sentence

# Plot each sentence type
for si in range(num_sentences):
    word_seq = get_word_sequence(net.corpus['sentence'][si])
    plt.plot(commitment_levels, parsing_accuracy_per_sent[si],
             color=colors[si], marker=markers[si],
             linewidth=2, markersize=6,
             label=f'S{si}: {word_seq}')

plt.xlabel('Commitment Level (t)', fontsize=12)
plt.ylabel('Parsing Accuracy', fontsize=12)
plt.title('Grammar 1 (G1) Parsing Accuracy by Sentence Type', fontsize=14)
plt.legend(loc='best', fontsize=10, framealpha=0.9)
plt.grid(True, alpha=0.3)
plt.ylim([0, 1.05])
plt.tight_layout()
plt.savefig('figure12_g1_parsing.png', dpi=300, bbox_inches='tight')
plt.show()

print("\n" + "="*70)
print("Generating treelet activation trajectories for each sentence type...")
print("="*70)

# ============================================================================
# Plot treelet activation trajectories at roles (2,1) and (3,2)
# for each sentence type
# ============================================================================

# Helper function to run network on a specific sentence and generate plots
def plot_sentence_treelets(net, sent, sent_idx, target):
    """Run network on specific sentence and plot treelet activations"""

    # Get word sequence for display
    word_seq = get_word_sequence(sent)

    # Extract word types only (without binding info)
    words = [bname.split('/')[0] for bname in sent]

    print(f"\nGenerating plots for Sentence {sent_idx}: {word_seq}")

    # Reset network and run on this specific sentence with trace logging
    net.reset(mu=net.ep, sd=0.01)
    net.initialize_traces(trace_list='all')

    # Run word by word
    for wi, word in enumerate(words):
        net.run_word(word, wi + 1, log_trace=True)

    # Run wrapup
    net.run_wrapup(log_trace=True)

    # Create figure with 2 subplots
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8))

    # Top panel: Role (2,1)
    plt.sca(ax1)
    gsc.plot_treelet_act_trace(
        net,
        rname='(2,1)',
        num_treelets=4,
        tmin=0,
        tmax=net.t,  # Use actual time reached
        downsampling=max(1, int(len(net.traces['t']) / 200)),  # Adaptive downsampling
        suppress_pos=True,
        add_prob=False,
        legend_pos='upper right'
    )
    ax1.set_title(f'S{sent_idx}: {word_seq} - Treelet Activations at Role (2,1)', fontsize=11)
    ax1.grid(True, alpha=0.3)

    # Bottom panel: Role (3,2)
    plt.sca(ax2)
    gsc.plot_treelet_act_trace(
        net,
        rname='(3,2)',
        num_treelets=4,
        tmin=0,
        tmax=net.t,
        downsampling=max(1, int(len(net.traces['t']) / 200)),
        suppress_pos=True,
        add_prob=False,
        legend_pos='upper right'
    )
    ax2.set_title(f'S{sent_idx}: {word_seq} - Treelet Activations at Role (3,2)', fontsize=11)
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    filename = f'treelet_activations_S{sent_idx}_{word_seq.replace(" ", "_")}.png'
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    plt.show()

    return filename

# Generate plots for each sentence in the corpus
filenames = []
for si, (sent, targ) in enumerate(zip(net.corpus['sentence'], net.corpus['target'])):
    filename = plot_sentence_treelets(net, sent, si, targ)
    filenames.append(filename)

print("\n" + "="*70)
print("Replication complete!")
print("Figures saved as:")
print("  - Plots from plot_train_result() (displayed interactively)")
print("  - figure12_g1_parsing.png")
print("\nTreelet activation trajectories for each sentence:")
for si, filename in enumerate(filenames):
    word_seq = get_word_sequence(net.corpus['sentence'][si])
    marker = " <-- S1 = 'N Vi P N'" if word_seq == 'N Vi P N' else ""
    print(f"  - {filename}{marker}")
print("="*70)
