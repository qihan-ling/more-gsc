"""
Example of how to refactor get_corpus_stat() to be GPU-native

This shows the architectural changes needed for true GPU acceleration.
"""

import numpy as np
try:
    import cupy as cp
    GPU_AVAILABLE = True
except ImportError:
    cp = np
    GPU_AVAILABLE = False

# ============================================================================
# Current approach (hybrid CPU-GPU with dictionaries)
# ============================================================================

def get_corpus_stat_current(corpus, num_bindings):
    """Current approach: Uses dictionaries (requires CPU transfers)"""
    stat = {}
    stat['trees'] = {}  # Dictionary - MUST be on CPU

    # Transfer to CPU if needed
    corpus_target = corpus['target']
    if hasattr(corpus_target, 'get'):
        corpus_target = corpus_target.get()

    for si, state in enumerate(corpus_target):
        p = corpus['prob_sent'][si]
        indices = np.where(state == 1)[0]
        gp_key = tuple(indices)  # Tuple key
        stat['trees'][gp_key] = p  # Dictionary assignment

    return stat

# ============================================================================
# GPU-native approach (array-based, no dictionaries)
# ============================================================================

def get_corpus_stat_gpu(corpus, num_bindings):
    """GPU approach: Uses arrays indexed by sentence ID"""

    # All operations can stay on GPU!
    num_sentences = len(corpus['target'])

    # Replace dict with array (indexed by sentence ID)
    stat_trees = np.zeros(num_sentences)  # CuPy array if GPU available
    stat_bindings = np.zeros(num_bindings)

    # No CPU transfer needed - all operations on GPU
    for si in range(num_sentences):
        state = corpus['target'][si]
        p = corpus['prob_sent'][si]

        # Direct array assignment (works on GPU!)
        stat_trees[si] = p

        # Accumulate binding probabilities
        indices = np.where(state == 1)[0]
        for idx in indices:
            stat_bindings[idx] += p

    # Return arrays instead of dicts
    return {
        'trees_array': stat_trees,  # Array indexed by sentence ID
        'bindings_array': stat_bindings,  # Array indexed by binding ID
        'sentence_ids': np.arange(num_sentences)  # For lookup
    }

# ============================================================================
# Even better: Fully vectorized (no Python loops)
# ============================================================================

def get_corpus_stat_vectorized(corpus, num_bindings):
    """Fully vectorized: Uses GPU matrix operations"""

    # corpus['target'] shape: (num_sentences, num_bindings)
    # Each row is a binary vector with 1s at active bindings

    # Trees: Just store probability per sentence (already vectorized!)
    stat_trees = corpus['prob_sent']  # Shape: (num_sentences,)

    # Bindings: Matrix-vector multiply (fully on GPU!)
    # corpus['target'].T @ corpus['prob_sent'] gives binding probabilities
    stat_bindings = corpus['target'].T @ corpus['prob_sent']  # GPU matrix multiply!

    # No loops, no transfers - pure GPU computation!
    return {
        'trees_array': stat_trees,
        'bindings_array': stat_bindings
    }

# ============================================================================
# Usage comparison
# ============================================================================

if __name__ == '__main__':
    # Create dummy corpus
    num_sentences = 100
    num_bindings = 405

    corpus = {
        'target': np.random.rand(num_sentences, num_bindings) > 0.95,
        'prob_sent': np.random.rand(num_sentences)
    }
    corpus['prob_sent'] /= corpus['prob_sent'].sum()

    # If GPU available, move to GPU
    if GPU_AVAILABLE:
        corpus['target'] = cp.asarray(corpus['target'])
        corpus['prob_sent'] = cp.asarray(corpus['prob_sent'])

    print("="*70)
    print("COMPARISON")
    print("="*70)

    # Current approach
    import time
    start = time.time()
    stat1 = get_corpus_stat_current(corpus, num_bindings)
    time1 = time.time() - start
    print(f"\nCurrent (dict-based): {time1*1000:.2f}ms")
    print(f"  Returns: {type(stat1['trees'])} with {len(stat1['trees'])} entries")

    # GPU approach
    start = time.time()
    stat2 = get_corpus_stat_gpu(corpus, num_bindings)
    time2 = time.time() - start
    print(f"\nGPU (array-based): {time2*1000:.2f}ms")
    print(f"  Returns: {type(stat2['trees_array'])} shape {stat2['trees_array'].shape}")
    print(f"  Speedup: {time1/time2:.2f}x")

    # Vectorized approach
    start = time.time()
    stat3 = get_corpus_stat_vectorized(corpus, num_bindings)
    time3 = time.time() - start
    print(f"\nVectorized (matrix ops): {time3*1000:.2f}ms")
    print(f"  Returns: {type(stat3['trees_array'])} shape {stat3['trees_array'].shape}")
    print(f"  Speedup: {time1/time3:.2f}x")

    print("\n" + "="*70)
    print("KEY INSIGHT")
    print("="*70)
    print("Instead of using gp_key tuples like (3, 5, 7) to index a dict,")
    print("use the sentence index directly to index an array!")
    print("\nOld: stat['trees'][(3,5,7)] = 0.42")
    print("New: stat_trees[sentence_idx] = 0.42")
