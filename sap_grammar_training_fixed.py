import matplotlib.pyplot as plt
import only_gscnet_speedup as gsc
import numpy as np
import time

def generate_corpus_with_progress(net, nsamples=20000):
    """Generate corpus with progress reporting"""
    sentences = []
    targets = []
    pvals = []
    counts = []

    print(f"\nGenerating {nsamples} sentences...")
    start_time = time.time()

    for i in range(nsamples):
        # Progress reporting every 100 sentences
        if (i+1) % 100 == 0 or i == 0:
            elapsed = time.time() - start_time
            rate = (i+1) / elapsed if elapsed > 0 else 0
            remaining = (nsamples - i - 1) / rate if rate > 0 else 0
            print(f"  {i+1}/{nsamples} ({(i+1)/nsamples*100:.1f}%) | "
                  f"Rate: {rate:.2f} sent/s | "
                  f"Unique so far: {len(sentences)} | "
                  f"ETA: {remaining/60:.1f} min",
                  flush=True, end='\r')

        sentence, target, p = net.generate_sentence()

        if sentence in sentences:
            idx = sentences.index(sentence)
            counts[idx] += 1
        else:
            sentences.append(sentence)
            targets.append(list(target))
            pvals.append(p)
            counts.append(1)

    print()  # New line after progress

    # Use empirical frequencies
    counts = np.array(counts)
    pvals = counts / counts.sum()

    idx = np.argsort(pvals)[::-1]
    sentences = [sentences[si] for si in idx]
    pvals = np.array([pvals[si] for si in idx])
    targets = np.array([targets[si] for si in idx])
    counts = np.array([counts[si] for si in idx])

    net.corpus = {
        'sentence': sentences,
        'target': targets,
        'count': counts,
        'prob_sent': pvals
    }

    total_time = time.time() - start_time
    print(f"\nCorpus generation complete!")
    print(f"  Total time: {total_time/60:.1f} minutes ({total_time:.1f} seconds)")
    print(f"  Unique sentences: {len(sentences)}")
    print(f"  Generation rate: {nsamples/total_time:.2f} sentences/second")
    print(f"  Samples per unique sentence: {nsamples/len(sentences):.1f}")

    return net.corpus


# Load grammar
with open('collapsed_filtered_sm5.grammar', 'r') as f:
    PCFG_sap = f.read()

ROOT = 'S'
# CRITICAL FIX: Reduce from 24 to 10 for much faster generation!
MAXLEN = 10

print("=" * 80)
print("SAP GRAMMAR TRAINING (OPTIMIZED)")
print("=" * 80)
print(f"Grammar: collapsed_filtered_sm5.grammar (1072 rules)")
print(f"Root: {ROOT}")
print(f"Max sentence length: {MAXLEN} (REDUCED from 24 for speed)")
print()

# Initialize
print("Initializing grammar...")
hg = gsc.HarmonicGrammar(pcfg=PCFG_sap, root=ROOT, max_sent_len=MAXLEN)
print(f"  Number of fillers: {len(hg.filler_names)}")
print(f"  Number of roles: {len(hg.role_names)}")

# Set all filler similarities to 0
sim = hg.get_simlist(dp=0.0)

# Network options
net_opts = {
    'T_init': 0.01,
    'q_max': 15.0,
    'q_init': 0.0,
    'dt_init': 0.005,
    'm': 30,
    'use_runC': True,
}

# Initialize network
print("Initializing network...")
net = gsc.GscNet(hg=hg, encodings={'similarity': sim}, opts=net_opts, seed=1024)

# Generate corpus with progress reporting
corpus = generate_corpus_with_progress(net, nsamples=20000)

# Display target probabilities (top 10)
print("\n" + "="*80)
print("Top 10 target sentence probabilities:")
print("="*80)
for si, sent in enumerate(net.corpus['sentence'][:10]):
    sent_str = ' '.join([bname.split('/')[0] for bname in sent])
    prob = net.corpus['prob_sent'][si]
    count = net.corpus['count'][si]
    print(f"  {si}: p={prob:.4f} (n={count:4d}) | {sent_str}")

# Determine num_trials based on corpus size
n_sentences = len(net.corpus['sentence'])
num_trials = min(max(n_sentences * 2, 100), 500)

print("\n" + "="*80)
print("TRAINING CONFIGURATION")
print("="*80)
print(f"  Unique sentences in corpus: {n_sentences}")
print(f"  Recommended num_trials: {num_trials} (2x corpus size, capped)")
print()

# Training setup
train_opts = {
    'lrate': 0.05,  # Start conservative
    'num_trials': num_trials,
    'ema_stat_weight': 0.0,
    'trace_varnames': ['kl_trees', 'kl_treelets', 'prob_sent', 'acc'],
    'report_cycle': 5,
    'init_noise_mag': 0.02,
    'average_weight': False,
    'average_filler_bias': False,
}

net.initialize(train_opts=train_opts)

print("Training parameters:")
print(f"  Learning rate: {train_opts['lrate']}")
print(f"  Trials per epoch: {train_opts['num_trials']}")
print(f"  Report cycle: {train_opts['report_cycle']}")
print()

# Training loop
print("=" * 80)
print("TRAINING")
print("=" * 80)

n_epochs = 500
num_epochs_per_block = 5
training_start = time.time()

for epoch_block in range(n_epochs // num_epochs_per_block):
    block_start = time.time()

    net.train2(
        train_opts={'num_epochs': num_epochs_per_block},
        savefilename='sap_model_fixed.pkl'
    )

    block_time = time.time() - block_start
    current_epoch = (epoch_block + 1) * num_epochs_per_block

    # Detailed progress every 100 epochs
    if current_epoch % 100 == 0:
        recent_kl = np.mean(net.traces_train['kl_trees'][-10:])
        recent_acc = np.mean(net.traces_train['acc'][-10:])
        elapsed = time.time() - training_start
        eta = (elapsed / current_epoch) * (n_epochs - current_epoch)

        print(f"\n>>> Epoch {current_epoch}/{n_epochs} ({current_epoch/n_epochs*100:.1f}%)")
        print(f"    Recent KL: {recent_kl:.4f}")
        print(f"    Recent Acc: {recent_acc:.4f}")
        print(f"    Block time: {block_time:.1f}s")
        print(f"    Elapsed: {elapsed/60:.1f}min | ETA: {eta/60:.1f}min")

        # Adaptive hyperparameters
        if current_epoch == 100 and recent_kl > 1.5:
            print("    >>> Slow convergence, increasing lrate to 0.1")
            train_opts['lrate'] = 0.1
            net.train_opts['lrate'] = 0.1

        if current_epoch == 200 and recent_kl > 1.0:
            print("    >>> Still slow, doubling num_trials")
            old_trials = train_opts['num_trials']
            train_opts['num_trials'] = min(old_trials * 2, 500)
            net.train_opts['num_trials'] = train_opts['num_trials']
            print(f"    >>> num_trials: {old_trials} -> {train_opts['num_trials']}")

total_training_time = time.time() - training_start

print("\n" + "="*80)
print("TRAINING COMPLETE!")
print("="*80)

# Calculate final statistics (last 100 updates)
final_kl = np.mean(net.traces_train['kl_trees'][-100:])
final_kl_sd = np.std(net.traces_train['kl_trees'][-100:])
final_acc = np.mean(net.traces_train['acc'][-100:])
final_acc_sd = np.std(net.traces_train['acc'][-100:])

print(f"\nFinal Statistics (last 100 epochs):")
print(f"  KL divergence: {final_kl:.3f} (SD = {final_kl_sd:.3f})")
print(f"  Production accuracy: {final_acc:.3f} (SD = {final_acc_sd:.3f})")
print(f"\nTiming:")
print(f"  Total training time: {total_training_time/60:.1f} minutes")
print(f"  Average time per epoch: {total_training_time/n_epochs:.2f} seconds")

# Display final learned probabilities (top 10)
print("\n" + "="*80)
print("Top 10 final learned probabilities Q(S):")
print("="*80)
final_probs = np.mean(net.traces_train['prob_sent'][-100:], axis=0)
for si in range(min(10, len(final_probs))):
    sent_str = ' '.join([bname.split('/')[0] for bname in net.corpus['sentence'][si]])
    print(f"  {si}: Q={final_probs[si]:.3f} | {sent_str}")

print("\n" + "="*80)
print("Model saved to: sap_model_fixed.pkl")
print("="*80)
