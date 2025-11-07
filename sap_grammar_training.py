import matplotlib.pyplot as plt
import gsc
import numpy as np
with open('collapsed_filtered_sm5.grammar', 'r') as f:
    PCFG_sap = f.read()

ROOT = 'S'
MAXLEN = 24

# ============================================================================
# Initialize network with paper's specifications
# ============================================================================

hg = gsc.HarmonicGrammar(pcfg=PCFG_sap, root=ROOT, max_sent_len=MAXLEN)

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
    # double the learning rate because trial size increases from 4 to 500
    'lrate': 0.2,
    'num_trials': 500,               # production trials per iteration
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

# Train for sufficient epochs to reach convergence [REDUCE n_epochs from 1000 to 100 because trial size increases from 4 to 500]
n_epochs = 100

for epoch_block in range(n_epochs // 5):
    net.train2(
        train_opts={'num_epochs': 5},
        savefilename='sap_model.pkl'
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

net = gsc.load_model('sap_model.pkl')

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

# Test parsing at different commitment levels (t ∈ {1, 2, ..., 12})
commitment_levels = list(range(1, 13))
parsing_accuracy = []

for t in commitment_levels:
    # Create commitment policy: use fixed commitment per word
    # For simplicity, use uniform commitment across all word positions
    max_sent_len = net.hg.opts['max_sent_len']
    dq = np.ones(max_sent_len) * (float(t) / max_sent_len)

    # Test parsing using gsc.test_parse_inc
    # FIXED: Implemented actual parsing test instead of placeholder
    try:
        parse_results = gsc.test_parse_inc(
            net,
            dq=dq,
            num_trials=10,
            estr=2,
            estr_null=2,
            disp=False
        )

        # Calculate overall accuracy
        n_correct = sum([parse_results[si]['acc'] for si in parse_results])
        n_total = len(parse_results)
        acc = n_correct / n_total if n_total > 0 else 0.0

    except Exception as e:
        print(f"  Warning: Parsing test failed at t={t}: {e}")
        acc = 0.0

    parsing_accuracy.append(acc)
    print(f"Commitment t={t:2d}: Parsing accuracy = {acc:.3f}")

# Plot Figure 12
plt.figure(figsize=(8, 6))
plt.plot(commitment_levels, parsing_accuracy, 'o-', linewidth=2, markersize=8)
plt.xlabel('Commitment Level (t)', fontsize=12)
plt.ylabel('Parsing Accuracy', fontsize=12)
plt.title('SAP Parsing Accuracy', fontsize=14)
plt.grid(True, alpha=0.3)
plt.ylim([0, 1.05])
plt.tight_layout()
plt.savefig('figure12_sap_parsing.png', dpi=300, bbox_inches='tight')
# plt.show()

print("\n" + "="*70)
print("Replication complete!")
print("Figures saved as:")
print("  - Plots from plot_train_result() (displayed interactively)")
print("  - figure12_sap_parsing.png")
print("="*70)
