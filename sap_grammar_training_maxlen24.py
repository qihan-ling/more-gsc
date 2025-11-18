"""
Training script for collapsed_filtered_sm5.grammar with MAXLEN=24

WARNING: Sentence generation is VERY SLOW with MAXLEN=24!
Expected corpus generation time: 30 minutes - 2 hours depending on nsamples.

Strategy:
1. Use small nsamples (1,000-5,000)
2. Aggressive progress reporting
3. Save intermediate results
4. Optimize training parameters for small corpus
"""
import matplotlib.pyplot as plt
import only_gscnet_speedup as gsc
import numpy as np
import time
import pickle
import os


def generate_corpus_with_progress_and_cache(net, nsamples=1000, cache_file='corpus_cache.pkl'):
    """
    Generate corpus with progress reporting and caching.

    If cache exists, load it. Otherwise generate and save.
    This prevents having to regenerate if training crashes.
    """
    # Try to load from cache first
    if os.path.exists(cache_file):
        print(f"\n{'='*80}")
        print(f"FOUND CACHED CORPUS: {cache_file}")
        print(f"{'='*80}")
        response = input("Load cached corpus? (y/n): ")
        if response.lower() == 'y':
            with open(cache_file, 'rb') as f:
                net.corpus = pickle.load(f)
            print(f"Loaded {len(net.corpus['sentence'])} unique sentences from cache")
            return net.corpus
        else:
            print("Regenerating corpus...")

    sentences = []
    targets = []
    pvals = []
    counts = []

    print(f"\n{'='*80}")
    print(f"CORPUS GENERATION (This will be SLOW with MAXLEN=24!)")
    print(f"{'='*80}")
    print(f"Target: {nsamples} sentences")
    print(f"Expected time: {nsamples * 0.5 / 60:.1f} - {nsamples * 2 / 60:.1f} minutes")
    print(f"(Assuming 0.5-2 seconds per sentence)\n")

    start_time = time.time()
    last_report_time = start_time

    for i in range(nsamples):
        sentence_start = time.time()

        # Generate sentence
        sentence, target, p = net.generate_sentence()

        sentence_time = time.time() - sentence_start

        # Update corpus
        if sentence in sentences:
            idx = sentences.index(sentence)
            counts[idx] += 1
        else:
            sentences.append(sentence)
            targets.append(list(target))
            pvals.append(p)
            counts.append(1)

        # Progress reporting every 10 sentences OR every 30 seconds
        current_time = time.time()
        should_report = ((i+1) % 10 == 0) or (current_time - last_report_time > 30)

        if should_report or i == 0:
            elapsed = current_time - start_time
            rate = (i+1) / elapsed if elapsed > 0 else 0
            remaining = (nsamples - i - 1) / rate if rate > 0 else 0

            print(f"  [{i+1:4d}/{nsamples}] ({(i+1)/nsamples*100:5.1f}%) | "
                  f"Rate: {rate:5.2f} sent/s | "
                  f"Last: {sentence_time:4.1f}s | "
                  f"Unique: {len(sentences):3d} | "
                  f"ETA: {remaining/60:5.1f}min",
                  flush=True)

            last_report_time = current_time

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

    print(f"\n{'='*80}")
    print(f"CORPUS GENERATION COMPLETE!")
    print(f"{'='*80}")
    print(f"  Total time: {total_time/60:.1f} minutes ({total_time:.0f} seconds)")
    print(f"  Unique sentences: {len(sentences)}")
    print(f"  Generation rate: {nsamples/total_time:.2f} sentences/second")
    print(f"  Average per sentence: {total_time/nsamples:.2f} seconds")
    print(f"  Samples per unique: {nsamples/len(sentences):.1f}")

    # Save to cache
    with open(cache_file, 'wb') as f:
        pickle.dump(net.corpus, f)
    print(f"  Corpus cached to: {cache_file}")

    return net.corpus


# ============================================================================
# CONFIGURATION
# ============================================================================

print("="*80)
print("SAP GRAMMAR TRAINING (MAXLEN=24 - SLOW BUT COMPLETE)")
print("="*80)
print()

# Load grammar
with open('collapsed_filtered_sm5.grammar', 'r') as f:
    PCFG_sap = f.read()

ROOT = 'S'
MAXLEN = 24  # REQUIRED - cannot reduce

print(f"Grammar: collapsed_filtered_sm5.grammar (1072 rules)")
print(f"Root: {ROOT}")
print(f"Max sentence length: {MAXLEN}")
print()

# ============================================================================
# HYPERPARAMETER RECOMMENDATIONS FOR SMALL CORPUS
# ============================================================================

print("="*80)
print("RECOMMENDED PARAMETERS FOR MAXLEN=24")
print("="*80)
print()
print("Since MAXLEN=24 makes generation slow, we use:")
print("  - Small nsamples (1,000 - 5,000)")
print("  - Moderate num_trials (50-200)")
print("  - Moderate epochs (300-500)")
print("  - Higher learning rate (0.1) since smaller corpus")
print()

# Ask user for nsamples
print("Choose corpus size:")
print("  1. FAST (1,000 samples)   - ~30 min generation, ~20-50 unique sentences")
print("  2. MEDIUM (2,500 samples) - ~75 min generation, ~50-100 unique sentences")
print("  3. LARGE (5,000 samples)  - ~150 min generation, ~80-150 unique sentences")
print()

choice = input("Enter choice (1/2/3) or custom number: ").strip()

if choice == '1':
    nsamples = 1000
elif choice == '2':
    nsamples = 2500
elif choice == '3':
    nsamples = 5000
else:
    try:
        nsamples = int(choice)
        print(f"Using custom nsamples={nsamples}")
    except:
        print("Invalid choice, using 2500")
        nsamples = 2500

print(f"\nUsing nsamples = {nsamples}")
print()

# ============================================================================
# INITIALIZE NETWORK
# ============================================================================

print("Initializing grammar...")
t0 = time.time()
hg = gsc.HarmonicGrammar(pcfg=PCFG_sap, root=ROOT, max_sent_len=MAXLEN)
print(f"  Grammar initialized in {time.time()-t0:.2f}s")
print(f"  Number of fillers: {len(hg.filler_names)}")
print(f"  Number of roles: {len(hg.role_names)}")

sim = hg.get_simlist(dp=0.0)

net_opts = {
    'T_init': 0.01,
    'q_max': 15.0,
    'q_init': 0.0,
    'dt_init': 0.005,
    'm': 30,
    'use_runC': True,
}

print("\nInitializing network...")
net = gsc.GscNet(hg=hg, encodings={'similarity': sim}, opts=net_opts, seed=1024)
print("  Network ready")

# ============================================================================
# GENERATE CORPUS (with caching)
# ============================================================================

corpus = generate_corpus_with_progress_and_cache(
    net,
    nsamples=nsamples,
    cache_file=f'corpus_cache_maxlen{MAXLEN}_n{nsamples}.pkl'
)

# Display top sentences
n_unique = len(corpus['sentence'])
print(f"\n{'='*80}")
print(f"CORPUS STATISTICS")
print(f"{'='*80}")
print(f"Total unique sentences: {n_unique}")
print()
print("Top 10 most frequent sentences:")
for si in range(min(10, n_unique)):
    sent_str = ' '.join([bname.split('/')[0] for bname in corpus['sentence'][si]])
    prob = corpus['prob_sent'][si]
    count = corpus['count'][si]
    print(f"  {si+1:2d}. p={prob:.4f} (n={count:4d}) | {sent_str}")

# ============================================================================
# TRAINING SETUP
# ============================================================================

# Adaptive num_trials based on corpus size
if n_unique < 50:
    num_trials = 100
elif n_unique < 100:
    num_trials = 150
else:
    num_trials = 200

print(f"\n{'='*80}")
print("TRAINING CONFIGURATION")
print(f"{'='*80}")
print(f"  Unique sentences: {n_unique}")
print(f"  num_trials: {num_trials} (adaptive)")
print()

train_opts = {
    'lrate': 0.1,  # Higher LR for smaller corpus
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
print(f"  Report every: {train_opts['report_cycle']} epochs")

# ============================================================================
# TRAINING
# ============================================================================

# Adaptive n_epochs based on corpus size
if n_unique < 50:
    n_epochs = 300
elif n_unique < 100:
    n_epochs = 400
else:
    n_epochs = 500

num_epochs_per_block = 5

print()
print(f"{'='*80}")
print(f"TRAINING ({n_epochs} epochs)")
print(f"{'='*80}")
print()

training_start = time.time()

for epoch_block in range(n_epochs // num_epochs_per_block):
    block_start = time.time()

    net.train2(
        train_opts={'num_epochs': num_epochs_per_block},
        savefilename=f'sap_model_maxlen{MAXLEN}.pkl'
    )

    block_time = time.time() - block_start
    current_epoch = (epoch_block + 1) * num_epochs_per_block

    # Progress every 50 epochs
    if current_epoch % 50 == 0 or current_epoch == n_epochs:
        recent_kl = np.mean(net.traces_train['kl_trees'][-10:])
        recent_acc = np.mean(net.traces_train['acc'][-10:])
        elapsed = time.time() - training_start
        eta = (elapsed / current_epoch) * (n_epochs - current_epoch)

        print(f"\n>>> Epoch {current_epoch}/{n_epochs} ({current_epoch/n_epochs*100:.1f}%)")
        print(f"    KL: {recent_kl:.4f} | Acc: {recent_acc:.4f}")
        print(f"    Time: {block_time:.1f}s/block | Elapsed: {elapsed/60:.1f}min | ETA: {eta/60:.1f}min")

        # Adaptive adjustments
        if current_epoch == 100 and recent_kl > 2.0:
            print("    >>> Very slow convergence, reducing lrate to 0.05")
            train_opts['lrate'] = 0.05
            net.train_opts['lrate'] = 0.05

total_training_time = time.time() - training_start

# ============================================================================
# RESULTS
# ============================================================================

print(f"\n{'='*80}")
print("TRAINING COMPLETE!")
print(f"{'='*80}")

final_kl = np.mean(net.traces_train['kl_trees'][-100:])
final_kl_sd = np.std(net.traces_train['kl_trees'][-100:])
final_acc = np.mean(net.traces_train['acc'][-100:])
final_acc_sd = np.std(net.traces_train['acc'][-100:])

print(f"\nFinal Statistics (last 100 epochs):")
print(f"  KL divergence: {final_kl:.3f} ± {final_kl_sd:.3f}")
print(f"  Production accuracy: {final_acc:.3f} ± {final_acc_sd:.3f}")

print(f"\nTiming:")
print(f"  Training time: {total_training_time/60:.1f} minutes")
print(f"  Avg per epoch: {total_training_time/n_epochs:.2f} seconds")

print(f"\nTop 10 learned probabilities:")
final_probs = np.mean(net.traces_train['prob_sent'][-100:], axis=0)
for si in range(min(10, len(final_probs))):
    sent_str = ' '.join([bname.split('/')[0] for bname in net.corpus['sentence'][si]])
    target_p = net.corpus['prob_sent'][si]
    learned_p = final_probs[si]
    print(f"  {si+1:2d}. Target: {target_p:.4f} | Learned: {learned_p:.4f} | {sent_str}")

print(f"\n{'='*80}")
print(f"Model saved to: sap_model_maxlen{MAXLEN}.pkl")
print(f"Corpus cached to: corpus_cache_maxlen{MAXLEN}_n{nsamples}.pkl")
print(f"{'='*80}")
