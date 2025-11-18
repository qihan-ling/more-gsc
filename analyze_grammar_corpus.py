#!/usr/bin/env python3
"""
Analyze the collapsed_filtered_sm5 grammar to determine:
1. Number of unique sentence types generated
2. Corpus statistics to inform hyperparameter choices
"""
import sys
sys.path.insert(0, '/home/user/more-gsc')
import only_gscnet_speedup as gsc
import numpy as np

# Load the grammar
with open('collapsed_filtered_sm5.grammar', 'r') as f:
    PCFG_sap = f.read()

ROOT = 'S'
MAXLEN = 24

print("=" * 80)
print("GRAMMAR ANALYSIS FOR HYPERPARAMETER TUNING")
print("=" * 80)
print(f"Grammar: collapsed_filtered_sm5.grammar")
print(f"Root: {ROOT}")
print(f"Max sentence length: {MAXLEN}")
print()

# Initialize the grammar
hg = gsc.HarmonicGrammar(pcfg=PCFG_sap, root=ROOT, max_sent_len=MAXLEN)
print(f"Number of fillers: {len(hg.filler_names)}")
print(f"Number of roles: {len(hg.role_names)}")
print()

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
net = gsc.GscNet(hg=hg, encodings={'similarity': sim}, opts=net_opts, seed=1024)

# Test different nsamples values
test_values = [5000, 10000, 20000, 50000, 100000]

print("=" * 80)
print("CORPUS GENERATION ANALYSIS")
print("=" * 80)

results = []
for nsamples in test_values:
    print(f"\nGenerating corpus with nsamples={nsamples}...")
    net.generate_corpus(nsamples=nsamples, use_freq=True)

    n_unique = len(net.corpus['sentence'])
    samples_per_sentence = nsamples / n_unique if n_unique > 0 else 0

    # Get distribution statistics
    probs = net.corpus['prob_sent']
    top_5_probs = probs[:5] if len(probs) >= 5 else probs
    min_prob = probs[-1] if len(probs) > 0 else 0

    results.append({
        'nsamples': nsamples,
        'n_unique': n_unique,
        'samples_per': samples_per_sentence,
        'top_5': top_5_probs,
        'min_prob': min_prob
    })

    print(f"  Unique sentences: {n_unique}")
    print(f"  Samples per sentence: {samples_per_sentence:.1f}")
    print(f"  Top 5 probabilities: {[f'{p:.4f}' for p in top_5_probs]}")
    print(f"  Minimum probability: {min_prob:.6f}")

print()
print("=" * 80)
print("SUMMARY & RECOMMENDATIONS")
print("=" * 80)

# Use the largest sample size for final recommendations
final_result = results[-1]
n_unique = final_result['n_unique']
samples_per = final_result['samples_per']

print(f"\nWith nsamples={final_result['nsamples']}:")
print(f"  - Found {n_unique} unique sentence types")
print(f"  - Average {samples_per:.1f} samples per sentence")

# Recommendations
print("\n" + "=" * 80)
print("RECOMMENDED HYPERPARAMETERS")
print("=" * 80)

# nsamples recommendation
if samples_per < 500:
    recommended_nsamples = n_unique * 1000
    print(f"\nnsamples: {recommended_nsamples:,}")
    print(f"  Reason: Current {samples_per:.0f} samples/sentence is low")
    print(f"  Target: ~1000 samples per unique sentence for robust probability estimates")
elif samples_per < 1000:
    recommended_nsamples = final_result['nsamples']
    print(f"\nnsamples: {recommended_nsamples:,}")
    print(f"  Reason: Current {samples_per:.0f} samples/sentence is acceptable")
else:
    recommended_nsamples = final_result['nsamples']
    print(f"\nnsamples: {recommended_nsamples:,}")
    print(f"  Reason: Current {samples_per:.0f} samples/sentence is good")

# num_trials recommendation
recommended_trials_min = n_unique * 2
recommended_trials_max = n_unique * 5
print(f"\nnum_trials: {recommended_trials_min}-{recommended_trials_max}")
print(f"  Reason: Need 2-5x coverage of {n_unique} sentence types per epoch")
print(f"  Start with: {min(recommended_trials_min, 200)}")
print(f"  Scale up to: {min(recommended_trials_max, 500)} if convergence is slow")

# lrate recommendation
print(f"\nlrate: 0.05-0.1")
print(f"  Reason: Larger grammar with {n_unique} sentence types")
print(f"  Start with: 0.05 (conservative)")
print(f"  If stable after 100 epochs: increase to 0.1")

# n_epochs recommendation
print(f"\nn_epochs: 500-1000")
print(f"  Reason: {n_unique} sentence types need substantial training")
print(f"  Monitor: KL divergence should plateau")
print(f"  Early stopping: If no improvement for 100 epochs")

print(f"\nnum_epochs (per checkpoint): 5-10")
print(f"  Reason: Checkpoint frequency, doesn't affect convergence")

print("\n" + "=" * 80)
print("CONVERGENCE ANALYSIS")
print("=" * 80)
print(f"\nWith num_trials=500 and {n_unique} sentence types:")
print(f"  Expected coverage per epoch: {(500/n_unique)*100:.1f}%")
print(f"  Expected hits per sentence type: {500/n_unique:.1f}")
print(f"  Assessment: ", end="")
if 500/n_unique >= 5:
    print("EXCELLENT - High coverage")
elif 500/n_unique >= 2:
    print("GOOD - Adequate coverage")
elif 500/n_unique >= 1:
    print("ACCEPTABLE - Moderate coverage")
else:
    print(f"LOW - Consider increasing to {n_unique * 2} trials")

print("\n" + "=" * 80)
