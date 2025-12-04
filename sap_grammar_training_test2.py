import matplotlib.pyplot as plt
import only_gscnet_speedup_sap as gsc
import numpy as np
import time

# TEST: Using seed=41 to check if different seeds produce different results
np.random.seed(41)
print("Global random seed set to 41 for testing")

t0 = time.time()  # Start timing
# with open('collapsed_filtered_sm5.grammar', 'r') as f:
#    PCFG_sap = f.read()

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

# ============================================================================
# Initialize network with paper's specifications
# ============================================================================

hg = gsc.HarmonicGrammar(pcfg=PCFG_G1, root=ROOT, max_sent_len=MAXLEN)

# Display fillers (should have 27 fillers × 15 roles = 405 units)
print(f"Filler names: {hg.filler_names}")
print(f"Number of fillers: {len(hg.filler_names)}")

# Set all filler similarities to 0 (linear independence)
sim = hg.get_simlist(dp=0.0)

# ============================================================================
# CONFIGURATION: Toggle these to test different modes
# ============================================================================
USE_SPARSE = True      # True = sparse WC matrix, False = dense
USE_COMPRESSED = False  # True = compressed encodings, False = full dimension
# ============================================================================

# Network options matching paper's parameters
net_opts = {
    'use_jax': False,  # Sparse only supported on CPU currently
    'T_init': 0.01,      # computational temperature
    'q_max': 15.0,       # maximum commitment
    'q_init': 0.0,       # initial commitment (FIXED: was 'q_0')
    'dt_init': 0.005,    # time step (FIXED: was 'dt')
    'm': 30,             # resource constraint (Hq1 strength)
    'use_runC': True,    # use C implementation for speed
    'ep_method': 'integration',
}
if USE_SPARSE:
    net_opts['use_sparse_wc'] = True

encodings = {
    'similarity': sim,
}
if USE_COMPRESSED:
    encodings['dim_f'] = 150  # Compressed filler encoding
    encodings['dim_r'] = 60   # Compressed role encoding

# Initialize network
net = gsc.GscNet(hg=hg, encodings=encodings,
                 opts=net_opts, seed=1024)

# ============================================================================
# DIAGNOSTIC: Verify what mode we're running in
# ============================================================================
print("\n" + "="*70)
print("MODE VERIFICATION:")
print("="*70)
print(f"  use_sparse: {getattr(net, 'use_sparse', False)}")
print(f"  WC type: {type(net.WC).__module__}.{type(net.WC).__name__}")
print(f"  WC shape: {net.WC.shape}")
if hasattr(net.WC, 'nnz'):
    print(f"  WC non-zeros: {net.WC.nnz:,} ({100*net.WC.nnz/net.WC.shape[0]/net.WC.shape[1]:.4f}% fill)")
print(f"  dim_f used: {net.dim_f if hasattr(net, 'dim_f') else 'N/A (full)'}")
print(f"  dim_r used: {net.dim_r if hasattr(net, 'dim_r') else 'N/A (full)'}")
print(f"  num_fillers: {net.num_fillers}")
print(f"  num_roles: {net.num_roles}")
print(f"  num_bindings: {net.num_bindings}")
print("="*70 + "\n")

net.generate_corpus(use_freq=True, nsamples=5000)

# Display target probabilities
print("\n" + "="*70)
print("Target sentence probabilities (first 10):")
for si, sent in enumerate(net.corpus['sentence'][:10]):
    sent_str = ' '.join([bname.split('/')[0] for bname in sent])
    prob = net.corpus['prob_sent'][si]
    print(f"Sentence {si}: p = {prob:.4f} ({sent_str})")

# ============================================================================
# Training setup (matching Section 4 parameters)
# ============================================================================

# NOTE: Random seed is controlled by GscNet constructor (seed=1024 at line 70)

train_opts = {
    # double the learning rate because trial size increases from 4 to 500
    'lrate': 0.1,
    'num_trials': 4,               # production trials per iteration
    'ema_stat_weight': 0.0,        # no EMA smoothing initially
    'trace_varnames': ['kl_trees', 'kl_treelets', 'prob_sent', 'acc'],
    'report_cycle': 10,            # report every 10 iterations
    'init_noise_mag': 0.02,
    'average_weight': False,
    'average_filler_bias': False,
}

net.initialize(train_opts=train_opts)

# Diagnostic check
print("\nChecking mask0:")
mask0 = net.train_opts['mask0']
if hasattr(net, 'use_sparse') and net.use_sparse:
    print(f"  mask0 non-zero entries: {mask0.nnz:,}")
    if mask0.nnz == 0:
        print("  ❌ PROBLEM: mask0 is empty!")
else:
    import numpy as np
    nnz = np.count_nonzero(mask0)
    print(f"  mask0 non-zero entries: {nnz:,} / {mask0.size:,}")
    if nnz == 0:
        print("  ❌ PROBLEM: mask0 is all zeros!")
# ============================================================================
# DEBUG: Print WC statistics before training
# ============================================================================
print("\n" + "="*40)
print("=== WC Statistics BEFORE Training ===")
if hasattr(net, 'use_sparse') and net.use_sparse:
    print(f"  WC type: sparse, nnz={net.WC.nnz}")
    print(f"  WC sum: {net.WC.sum():.6f}")
    print(f"  WC diagonal sum: {net.WC.diagonal().sum():.6f}")
    print(f"  WC max: {net.WC.max():.6f}, min: {net.WC.min():.6f}")
    # DEBUG: Print first 10 non-zero entries to compare with dense
    wc_coo = net.WC.tocoo()
    print(f"  First 10 non-zero entries (row, col, val):")
    for i in range(min(10, len(wc_coo.data))):
        print(f"    ({wc_coo.row[i]}, {wc_coo.col[i]}): {wc_coo.data[i]:.6f}")
else:
    print(f"  WC type: dense")
    print(f"  WC sum: {net.WC.sum():.6f}")
    print(f"  WC diagonal sum: {np.diag(net.WC).sum():.6f}")
    print(f"  WC max: {net.WC.max():.6f}, min: {net.WC.min():.6f}")
    # DEBUG: Print first 10 non-zero entries to compare with sparse
    nonzero = np.nonzero(net.WC)
    print(f"  First 10 non-zero entries (row, col, val):")
    for i in range(min(10, len(nonzero[0]))):
        r, c = nonzero[0][i], nonzero[1][i]
        print(f"    ({r}, {c}): {net.WC[r, c]:.6f}")
print("=" * 40)

# DEBUG: Test dynamics computation
print("\n=== Dynamics Test ===")
test_actC = np.random.RandomState(42).rand(net.num_bindings)
if hasattr(net, 'use_sparse') and net.use_sparse:
    wc_dot_result = net.WC.dot(test_actC)
else:
    wc_dot_result = net.WC.dot(test_actC)
print(f"  WC.dot(test_actC) sum: {wc_dot_result.sum():.10f}")
print(f"  WC.dot(test_actC) first 5: {wc_dot_result[:5]}")
print("=" * 40)

# ============================================================================
# Training loop for Figure 11
# ============================================================================

print("\n" + "="*70)
print("Training Grammar 1...")
print("="*70)

# Train for sufficient epochs to reach convergence
n_epochs = 1000

#exec(open('debug_training_update.py').read())
#exec(open('debug_weight_update.py').read())
#exec(open('debug_across_10epoch_update.py').read())
for epoch_block in range(n_epochs // 5):
    net.train2(
        train_opts={'num_epochs': 5},
        savefilename='sap_g1_model_sparse_nocompress.pkl'
    )

print("\n" + "="*70)
print("Training complete!")

# DEBUG: Print WC statistics after training
print("\n" + "="*40)
print("=== WC Statistics AFTER Training ===")
if hasattr(net, 'use_sparse') and net.use_sparse:
    print(f"  WC type: sparse, nnz={net.WC.nnz}")
    print(f"  WC sum: {net.WC.sum():.6f}")
    print(f"  WC diagonal sum: {net.WC.diagonal().sum():.6f}")
    print(f"  WC max: {net.WC.max():.6f}, min: {net.WC.min():.6f}")
else:
    print(f"  WC type: dense")
    print(f"  WC sum: {net.WC.sum():.6f}")
    print(f"  WC diagonal sum: {np.diag(net.WC).sum():.6f}")
    print(f"  WC max: {net.WC.max():.6f}, min: {net.WC.min():.6f}")
print("=" * 40)

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

net = gsc.load_model('sap_g1_model_sparse_nocompress.pkl')


# ============================================================================
# DEBUG: Analyze WC structure after training
# ============================================================================
print("\n" + "="*70)
print("DEBUG: Analyzing model structure for sparse vs dense comparison")
print("="*70)

# Check overall WC structure
gsc.debug_wc_structure(
    net,
    # Check PP treelet weights (critical for "N Vi P N")
    treelet_rules=[
        ('PP[1]:1', 'P:0', 'N:1'),    # PP -> P N
        ('VP[1]:1', 'Vi:0', 'PP[1]:1'), # VP -> Vi PP  
        ('S[2]:0', 'N:0', 'VP[1]:1'),   # S -> N VP
    ],
    # Check connectivity of key bindings
    check_bindings=[
        'P:0/(1,3)',      # P at position 3
        'N:1/(1,4)',      # N at position 4
        'PP[1]:1/(2,3)',  # PP at level 2, position 3
        'VP[1]:1/(3,2)',  # VP at level 3, position 2
    ]
)

# Debug parsing for S1 "N Vi P N" specifically
print("\n" + "="*70)
print("DEBUG: Detailed parsing of 'N Vi P N' (S1)")
print("="*70)
# Get target bindings for S1
s1_target = net.corpus['target'][1]  # S1 is index 1
s1_target_bnames = [net.binding_names[i] for i in np.where(s1_target == 1)[0]]
print(f"Target bindings for S1: {s1_target_bnames}")

gsc.debug_parse_comparison(
    net,
    sent=['N', 'Vi', 'P', 'N'],
    target_bnames=s1_target_bnames
)

# FIXED: plot_train_result() doesn't return a figure object and calls plt.show()
# internally, so we just call it directly
print("\n" + "="*70)
print("Generating training plots (Figure 11)...")
print("Note: plot_train_result() will display 3 separate plots")
print("="*70)
gsc.plot_train_result(
    net, savefilename_prefix='sap_g1_model_sparse_nocompres', legend=True, linewidth=1.5)

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
        np.random.seed(1024 + t)
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
colors = ['#1f77b4', '#ff7f0e', '#2ca02c',
          '#d62728', '#9467bd']  # 5 distinct colors
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
plt.savefig('sap_g1_model_sparse_nocompres_parsing.png',
            dpi=300, bbox_inches='tight')
# plt.show()

print("\n" + "="*70)
print("Generating treelet activation trajectories for each sentence type...")
print("="*70)

# ============================================================================
# Plot treelet activation trajectories at roles (2,1) and (3,2)
# for each sentence type
# ============================================================================

# Helper function to run network on a specific sentence and generate plots

print("\n=== CORPUS SENTENCE ORDER ===")
for si, sent in enumerate(net.corpus['sentence']):
    word_seq = get_word_sequence(sent)
    print(f"S{si}: {word_seq}")
print("="*70)


def plot_sentence_treelets(net, sent, sent_idx, target):
    """Run network on specific sentence and plot treelet activations"""

    # Get word sequence for display
    word_seq = get_word_sequence(sent)

    # Extract word types only (without binding info)
    words = [bname.split('/')[0] for bname in sent]

    print(f"\nGenerating plots for Sentence {sent_idx}: {word_seq}")

    # Reset network and run on this specific sentence with trace logging
    np.random.seed(1024 + sent_idx)
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
        # Adaptive downsampling
        downsampling=max(1, int(len(net.traces['t']) / 200)),
        suppress_pos=True,
        add_prob=False,
        legend_pos='upper right'
    )
    ax1.set_title(
        f'S{sent_idx}: {word_seq} - Treelet Activations at Role (2,1)', fontsize=11)
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
    ax2.set_title(
        f'S{sent_idx}: {word_seq} - Treelet Activations at Role (3,2)', fontsize=11)
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    filename = f'sap_g1_model_sparse_nocompres_S{sent_idx}_{word_seq.replace(" ", "_")}.png'
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    # plt.show()

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
print(f"Finished running at {time.time()-t0:.2f}s")

