"""
Train a GSC model on the ClassicGP_MVPR grammar with probabilities
extracted from the Berkeley parser (sm5).

Grammar includes relative clause (SBAR -> WHNP VP) and reduced relative
(NP -> NP VP) structures.

Usage:
    python SAP_analysis/train_classicgp_mvpr.py
"""

import matplotlib.pyplot as plt
import only_gscnet_speedup as gsc
import numpy as np
import os
import time

t0 = time.time()

# ============================================================================
# Load grammar
# ============================================================================

PROBS_PATH = os.path.join('SAP_stimuli', 'ClassicGP_MVPR_probs.txt')
SAVE_PREFIX = 'SAP_analysis/classicgp_mvpr'
SAVE_MODEL = f'{SAVE_PREFIX}_model.pkl'

with open(PROBS_PATH) as f:
    pcfg_str = f.read()

print("Loaded PCFG:")
print(pcfg_str)

ROOT = 'S'
MAXLEN = 10

# ============================================================================
# Initialize network
# ============================================================================

hg = gsc.HarmonicGrammar(pcfg=pcfg_str, root=ROOT, max_sent_len=MAXLEN)

print(f"Filler names: {hg.filler_names}")
print(f"Number of fillers: {len(hg.filler_names)}")

sim = hg.get_simlist(dp=0.0)

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

REQUIRED_SENTENCES = {
    'DT NN WP VBD VBN DT NN VBD JJ NN',
    'DT NN VBN DT NN VBD JJ NN',
}
NSAMPLES = 50000
MAX_EXTRA_SAMPLES = 10_000_000

net.generate_corpus(nsamples=NSAMPLES, use_freq=True)

get_ws = lambda s: ' '.join(b.split('/')[0] for b in s)
missing = REQUIRED_SENTENCES - {get_ws(s) for s in net.corpus['sentence']}
if missing:
    print(f"Sampling for {len(missing)} required sentences...")
    for extra in range(MAX_EXTRA_SAMPLES):
        sent, target, p = net.generate_sentence()
        ws = get_ws(sent)
        if ws in missing:
            net.corpus['sentence'].append(sent)
            net.corpus['target'] = np.vstack([net.corpus['target'], [target]])
            net.corpus['count'] = np.append(net.corpus['count'], 1)
            missing.discard(ws)
            print(f"  Found '{ws}' after {extra+1} extra samples")
            if not missing:
                break
    if missing:
        print(f"  WARNING: Still missing after {MAX_EXTRA_SAMPLES} extra samples: {missing}")
    net.corpus['prob_sent'] = net.corpus['count'] / net.corpus['count'].sum()
    idx = np.argsort(net.corpus['prob_sent'])[::-1]
    net.corpus['sentence'] = [net.corpus['sentence'][i] for i in idx]
    net.corpus['target'] = net.corpus['target'][idx]
    net.corpus['count'] = net.corpus['count'][idx]
    net.corpus['prob_sent'] = net.corpus['prob_sent'][idx]

print("\n" + "=" * 70)
print("Target sentence probabilities:")
for si, sent in enumerate(net.corpus['sentence']):
    sent_str = ' '.join([bname.split('/')[0] for bname in sent])
    prob = net.corpus['prob_sent'][si]
    print(f"Sentence {si}: p = {prob:.6f} ({sent_str})")

# ============================================================================
# Training
# ============================================================================

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

print("\n" + "=" * 70)
print("Training ClassicGP_MVPR model...")
print("=" * 70)

n_epochs = 1000

for epoch_block in range(n_epochs // 10):
    net.train2(
        train_opts={'num_epochs': 10},
        savefilename=SAVE_MODEL,
    )

print("\n" + "=" * 70)
print("Training complete!")

final_kl = np.mean(net.traces_train['kl_trees'][-100:])
final_kl_sd = np.std(net.traces_train['kl_trees'][-100:])
final_acc = np.mean(net.traces_train['acc'][-100:])
final_acc_sd = np.std(net.traces_train['acc'][-100:])

print(f"Final KL divergence: {final_kl:.3f} (SD = {final_kl_sd:.3f})")
print(f"Final production accuracy: {final_acc:.3f} (SD = {final_acc_sd:.3f})")

print("\nFinal learned probabilities Q(S):")
final_probs = np.mean(net.traces_train['prob_sent'][-100:], axis=0)
for si, prob in enumerate(final_probs):
    sent = net.corpus['sentence'][si]
    sent_str = ' '.join([bname.split('/')[0] for bname in sent])
    print(f"Sentence {si}: Q = {prob:.4f} ({sent_str})")

# ============================================================================
# Plot training dynamics
# ============================================================================

net = gsc.load_model(SAVE_MODEL)

print("\n" + "=" * 70)
print("Generating training plots...")
print("=" * 70)
gsc.plot_train_result(
    net, savefilename_prefix=f'{SAVE_PREFIX}_train',
    legend=True, linewidth=1.5)

# ============================================================================
# Parsing tests (random 10 sentences)
# ============================================================================

print("\n" + "=" * 70)
print("Testing parsing accuracy...")
print("=" * 70)


def get_word_sequence(sent):
    return ' '.join([bname.split('/')[0] for bname in sent])


num_sentences = len(net.corpus['sentence'])
N_PARSE_SAMPLE = min(10, num_sentences)
np.random.seed(1024)
parse_sample_indices = sorted(
    np.random.choice(num_sentences, size=N_PARSE_SAMPLE, replace=False))
print(f"Randomly selected {N_PARSE_SAMPLE} sentences for parsing test:")
for si in parse_sample_indices:
    print(f"  S{si}: {get_word_sequence(net.corpus['sentence'][si])}")

commitment_levels = list(range(1, 13))
parsing_accuracy_per_sent = {si: [] for si in parse_sample_indices}

for t in commitment_levels:
    max_sent_len = net.hg.opts['max_sent_len']
    dq = np.ones(max_sent_len) * (float(t) / max_sent_len)

    try:
        np.random.seed(1024 + t)
        parse_results = gsc.test_parse_inc(
            net, dq=dq, num_trials=10, estr=2, estr_null=2, disp=False)

        for si in parse_sample_indices:
            acc_si = parse_results[si]['acc'] if si in parse_results else 0.0
            parsing_accuracy_per_sent[si].append(acc_si)

        accs = [parse_results[si]['acc']
                for si in parse_sample_indices if si in parse_results]
        acc_overall = np.mean(accs) if accs else 0.0

    except Exception as e:
        print(f"  Warning: Parsing test failed at t={t}: {e}")
        for si in parse_sample_indices:
            parsing_accuracy_per_sent[si].append(0.0)
        acc_overall = 0.0

    print(f"Commitment t={t:2d}: Overall accuracy = {acc_overall:.3f}")

plt.figure(figsize=(10, 6))
cmap = plt.cm.tab20
for idx, si in enumerate(parse_sample_indices):
    word_seq = get_word_sequence(net.corpus['sentence'][si])
    plt.plot(commitment_levels, parsing_accuracy_per_sent[si],
             color=cmap(idx / max(N_PARSE_SAMPLE, 1)),
             linewidth=2, label=f'S{si}: {word_seq}')

plt.xlabel('Commitment Level (t)', fontsize=12)
plt.ylabel('Parsing Accuracy', fontsize=12)
plt.title('ClassicGP_MVPR Parsing Accuracy by Sentence Type', fontsize=14)
plt.legend(loc='best', fontsize=8, framealpha=0.9, ncol=2)
plt.grid(True, alpha=0.3)
plt.ylim([0, 1.05])
plt.tight_layout()
plt.savefig(f'{SAVE_PREFIX}_parsing.png', dpi=300, bbox_inches='tight')

# ============================================================================
# Treelet activation trajectories
# ============================================================================

print("\n" + "=" * 70)
print("Generating treelet activation trajectories...")
print("=" * 70)

print("\n=== CORPUS SENTENCE ORDER ===")
for si, sent in enumerate(net.corpus['sentence']):
    print(f"S{si}: {get_word_sequence(sent)}")
print("=" * 70)


TREELET_SENTENCES = [
    'DT NN WP VBD VBN DT NN VBD JJ NN',
    'DT NN VBN DT NN VBD JJ NN',
]
TREELET_ROLES = ['(1,3)', '(8,2)'] # (1,3) checks if WHNP is present for treelet1, (8,2) checks if SBAR is present for treelet1
NUM_TREELETS = 10

corpus_word_seqs = {
    si: get_word_sequence(sent)
    for si, sent in enumerate(net.corpus['sentence'])
}
treelet_indices = []
for target_seq in TREELET_SENTENCES:
    found = [si for si, ws in corpus_word_seqs.items() if ws == target_seq]
    if found:
        treelet_indices.append(found[0])
        print(f"  Found '{target_seq}' as S{found[0]}")
    else:
        print(f"  WARNING: '{target_seq}' not found in corpus!")


def plot_sentence_treelets(net, sent, sent_idx):
    """Run network on a sentence and plot treelet activations at key roles."""
    word_seq = get_word_sequence(sent)
    words = [bname.split('/')[0] for bname in sent]
    print(f"\nGenerating plots for Sentence {sent_idx}: {word_seq}")

    np.random.seed(1024 + sent_idx)
    net.reset(mu=net.ep, sd=0.01)
    net.initialize_traces(trace_list='all')

    for wi, word in enumerate(words):
        net.run_word(word, wi + 1, log_trace=True)
    net.run_wrapup(log_trace=True)

    n_panels = len(TREELET_ROLES)
    fig, axes = plt.subplots(n_panels, 1, figsize=(12, 5 * n_panels))
    if n_panels == 1:
        axes = [axes]

    for ax, rname in zip(axes, TREELET_ROLES):
        plt.sca(ax)
        try:
            gsc.plot_treelet_act_trace(
                net, rname=rname, num_treelets=NUM_TREELETS,
                tmin=0, tmax=net.t,
                downsampling=max(1, int(len(net.traces['t']) / 200)),
                suppress_pos=True, add_prob=False,
                legend_pos='upper right')
        except Exception as e:
            ax.text(0.5, 0.5, f'Role {rname}: {e}',
                    transform=ax.transAxes, ha='center')
        ax.set_title(f'S{sent_idx}: {word_seq} — Role {rname}', fontsize=11)
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    fname = (f'{SAVE_PREFIX}_S{sent_idx}_'
             f'{word_seq.replace(" ", "_")}.png')
    plt.savefig(fname, dpi=300, bbox_inches='tight')
    return fname


filenames = []
for si in treelet_indices:
    sent = net.corpus['sentence'][si]
    fname = plot_sentence_treelets(net, sent, si)
    filenames.append(fname)

print("\n" + "=" * 70)
print("Done!")
print(f"Model:   {SAVE_MODEL}")
print(f"Parsing: {SAVE_PREFIX}_parsing.png")
print("Treelet plots:")
for fname in filenames:
    print(f"  {fname}")
print("=" * 70)
print(f"Total time: {time.time() - t0:.1f}s")
