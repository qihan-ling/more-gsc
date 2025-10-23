#!/usr/bin/env python3
"""
Demonstration: GSC with Static Embeddings and Temporal Decay

This script demonstrates the two new features:
1. Static embeddings - Pre-computed word/POS embeddings
2. Temporal decay - Gradual forgetting of earlier input (working memory)

Author: Claude
Date: 2025-10-23
"""

import numpy as np
import matplotlib.pyplot as plt
from gsc_enhanced import create_enhanced_network, GscNetEnhanced
from gsc_extensions import (EmbeddingManager, TemporalDecayManager,
                           create_pos_embeddings_from_words)
import gsc


# ============================================================================
# Example 1: Basic Usage with Synthetic Embeddings
# ============================================================================

print("="*80)
print("Example 1: Basic Enhanced GSC with Synthetic Embeddings + Decay")
print("="*80)

# Simple PTB-style PCFG
PCFG_SIMPLE = '''
0.50 S -> NP VP
0.30 S -> NP VP PP
0.20 S -> S CC S

0.40 NP -> DT NN
0.30 NP -> DT JJ NN
0.20 NP -> NNP
0.10 NP -> PRP

0.40 VP -> VBD
0.30 VP -> VBD NP
0.20 VP -> VBZ NP
0.10 VP -> VBP SBAR

1.0 PP -> IN NP
1.0 SBAR -> IN S
'''

# Create synthetic embeddings (in practice, load from Llama/Word2Vec/etc)
print("\n[1] Creating synthetic POS embeddings...")
pos_tags = ['S', 'NP', 'VP', 'PP', 'SBAR', 'DT', 'NN', 'JJ', 'NNP', 'PRP',
            'VBD', 'VBZ', 'VBP', 'IN', 'CC']

synthetic_embeddings = {}
np.random.seed(42)
for pos in pos_tags:
    emb = np.random.randn(300)
    emb /= np.linalg.norm(emb)  # Normalize
    synthetic_embeddings[pos] = emb

print(f"   Created {len(synthetic_embeddings)} embeddings of dimension 300")

# Create enhanced network
print("\n[2] Initializing enhanced GSC network...")
print("   Features:")
print("   - Static embeddings: YES (300d -> 100d projection)")
print("   - Temporal decay: YES (λ = 0.15, exponential)")

net = create_enhanced_network(
    pcfg_string=PCFG_SIMPLE,
    root='S',
    max_sent_len=8,
    embedding_dict=synthetic_embeddings,
    projection_dim=100,
    decay_rate=0.15,
    seed=1024
)

print(f"\n   Network created:")
print(f"   - Fillers: {net.num_fillers}")
print(f"   - Roles: {net.num_roles}")
print(f"   - Bindings: {net.num_bindings}")

if net.embedding_manager:
    F = net.embedding_manager.get_filler_embeddings()
    print(f"   - Embedding matrix shape: {F.shape}")
    print(f"   - Embeddings projected from 300d to {F.shape[0]}d")

# Test parsing
print("\n[3] Testing incremental parsing...")
test_sentence = ['DT/(1,1)', 'JJ/(1,2)', 'NN/(1,3)', 'VBD/(1,4)']
print(f"   Sentence: {' '.join([b.split('/')[0] for b in test_sentence])}")

# Parse with decay
print("\n   Parsing WITH decay:")
result_with_decay = net.parse_sentence_incremental(
    test_sentence,
    apply_decay=True,
    reset_first=True
)

decay_strengths = net.get_decay_strength_by_position()
print(f"   Decay strengths:")
for i, strength in enumerate(decay_strengths):
    print(f"      Position {i+1}: {strength*100:.1f}% retained")

# Parse without decay (for comparison)
net.decay_manager.decay_rate = 0.0  # Temporarily disable
print("\n   Parsing WITHOUT decay:")
result_no_decay = net.parse_sentence_incremental(
    test_sentence,
    apply_decay=False,
    reset_first=True
)
net.decay_manager.decay_rate = 0.15  # Re-enable

print(f"\n   External input strength comparison:")
print(f"      With decay: {np.linalg.norm(result_with_decay['extC']):.4f}")
print(f"      Without decay: {np.linalg.norm(result_no_decay['extC']):.4f}")


# ============================================================================
# Example 2: Decay Rate Comparison
# ============================================================================

print("\n" + "="*80)
print("Example 2: Comparing Different Decay Rates")
print("="*80)

decay_rates = [0.0, 0.05, 0.10, 0.15, 0.20, 0.30]
sentence_length = 7

print(f"\nSimulating decay over {sentence_length} word positions")
print(f"Testing decay rates: {decay_rates}\n")

# Create plot data
position_distances = np.arange(0, sentence_length)
decay_curves = {}

for rate in decay_rates:
    decay_mgr = TemporalDecayManager(decay_rate=rate, decay_type='exponential')
    decay_values = [decay_mgr._compute_decay_factor(d) for d in position_distances]
    decay_curves[rate] = decay_values

    print(f"Decay rate λ = {rate:.2f}:")
    for d, val in zip(position_distances[:5], decay_values[:5]):
        print(f"   Position -{d}: {val*100:5.1f}% retained")
    print()

# Plot decay curves
plt.figure(figsize=(10, 6))
for rate, values in decay_curves.items():
    plt.plot(position_distances, values, marker='o', label=f'λ = {rate:.2f}')

plt.xlabel('Position Distance (words back)', fontsize=12)
plt.ylabel('Activation Retained (%)', fontsize=12)
plt.title('Temporal Decay: Effect of Decay Rate', fontsize=14)
plt.legend()
plt.grid(True, alpha=0.3)
plt.ylim([0, 1.05])
plt.tight_layout()
plt.savefig('decay_rate_comparison.png', dpi=150)
print("   Plot saved as 'decay_rate_comparison.png'")


# ============================================================================
# Example 3: Embedding Similarity Analysis
# ============================================================================

print("\n" + "="*80)
print("Example 3: Analyzing Embedding Similarities")
print("="*80)

print("\nComputing pairwise similarities for POS tags...")

if net.embedding_manager:
    similarity_matrix = net.embedding_manager.compute_similarity_matrix()

    # Show some interesting similarities
    interesting_pairs = [
        ('NN', 'NNP'),   # Singular vs proper noun
        ('VBD', 'VBZ'),  # Past vs present verb
        ('DT', 'NN'),    # Determiner vs noun
        ('NN', 'VBD'),   # Noun vs verb
    ]

    filler_names = net.filler_names
    print(f"\nSample POS tag similarities:")
    for pos1, pos2 in interesting_pairs:
        if pos1 in filler_names and pos2 in filler_names:
            idx1 = filler_names.index(pos1)
            idx2 = filler_names.index(pos2)
            sim = similarity_matrix[idx1, idx2]
            print(f"   {pos1:5s} <-> {pos2:5s}: {sim:6.3f}")

    # Visualize similarity matrix (for available fillers)
    available_pos = [p for p in ['DT', 'NN', 'VBD', 'JJ', 'IN'] if p in filler_names]
    if len(available_pos) >= 3:
        indices = [filler_names.index(p) for p in available_pos]
        sub_sim = similarity_matrix[np.ix_(indices, indices)]

        plt.figure(figsize=(8, 7))
        plt.imshow(sub_sim, cmap='RdYlBu_r', vmin=-1, vmax=1)
        plt.colorbar(label='Similarity (dot product)')
        plt.xticks(range(len(available_pos)), available_pos)
        plt.yticks(range(len(available_pos)), available_pos)
        plt.title('POS Tag Embedding Similarities', fontsize=14)

        # Add values to cells
        for i in range(len(available_pos)):
            for j in range(len(available_pos)):
                text = plt.text(j, i, f'{sub_sim[i, j]:.2f}',
                              ha="center", va="center", color="black",
                              fontsize=9)

        plt.tight_layout()
        plt.savefig('embedding_similarity_matrix.png', dpi=150)
        print("\n   Similarity matrix saved as 'embedding_similarity_matrix.png'")


# ============================================================================
# Example 4: Garden Path Effect with Decay
# ============================================================================

print("\n" + "="*80)
print("Example 4: Garden Path Sentences and Temporal Decay")
print("="*80)

print("\nHypothesis: Temporal decay may help or hurt garden path recovery")
print("Test: 'The horse raced past the barn fell'")
print("      Structure: [The horse [that was raced past the barn]] fell")

# Simplified representation
garden_path = ['DT/(1,1)', 'NN/(1,2)', 'VBD/(1,3)', 'IN/(1,4)',
               'DT/(1,5)', 'NN/(1,6)', 'VBD/(1,7)']

print(f"\nParsing garden path sentence with different decay rates...")

garden_path_results = {}
for rate in [0.0, 0.1, 0.2]:
    net.decay_manager.decay_rate = rate
    result = net.parse_sentence_incremental(
        garden_path,
        apply_decay=True,
        reset_first=True,
        noise_sd=0.02
    )

    garden_path_results[rate] = result
    final_activation = np.linalg.norm(result['actC'])

    print(f"\n   Decay rate λ = {rate:.1f}:")
    print(f"      Final activation magnitude: {final_activation:.4f}")

    # Get decay strengths for each word
    strengths = []
    for hist in result['input_history']:
        distance = result['final_position'] - hist['position']
        factor = net.decay_manager._compute_decay_factor(distance)
        strengths.append(factor)

    print(f"      Word retention at end: {np.mean(strengths)*100:.1f}%")

print("\nNote: In real experiments, would measure:")
print("   - Parsing accuracy")
print("   - Reanalysis success rate")
print("   - Processing time per word")


# ============================================================================
# Summary
# ============================================================================

print("\n" + "="*80)
print("Summary: Key Features Demonstrated")
print("="*80)

print("""
1. STATIC EMBEDDINGS:
   ✓ Pre-computed embeddings loaded into network
   ✓ Dimensionality reduction (300d -> 100d)
   ✓ Similarity-based filler representations
   ✓ Ready for real embeddings (Word2Vec, GloVe, Llama, etc.)

2. TEMPORAL DECAY:
   ✓ Exponential decay of earlier inputs
   ✓ Configurable decay rate (λ parameter)
   ✓ Position-based decay tracking
   ✓ Incremental parsing with decay

3. APPLICATIONS:
   ✓ Nonsense word testing (via embeddings)
   ✓ Garden path modeling (via decay)
   ✓ Working memory limitations (via decay)
   ✓ Semantic-syntactic interactions

NEXT STEPS:
   - Load real embeddings from Llama/Word2Vec/GloVe
   - Test on Penn Treebank sentences
   - Tune decay rate to human reading time data
   - Generate and test nonsense words
   - Compare with/without decay on parsing accuracy
""")

print("="*80)
print("Demo complete! Plots saved:")
print("  - decay_rate_comparison.png")
print("  - embedding_similarity_matrix.png")
print("="*80)
