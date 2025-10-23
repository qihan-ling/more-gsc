"""
Enhanced GSC Network with Embeddings and Temporal Decay

This module provides GscNetEnhanced, which extends the base GscNet with:
1. Static embedding support
2. Temporal decay of external input

Author: Claude
Date: 2025-10-23
"""

import numpy as np
import sys
import gsc
from gsc_extensions import EmbeddingManager, TemporalDecayManager


class GscNetEnhanced(gsc.GscNet):
    """Enhanced GSC Network with embeddings and temporal decay

    Extensions:
    - Static embeddings for fillers (from pre-trained models)
    - Temporal decay of external input (working memory limitations)
    """

    def __init__(self, hg=None, encodings=None, opts=None, qpolicy=None,
                 seed=None, embedding_manager=None, decay_manager=None):
        """Initialize enhanced GSC network

        Args:
            hg: HarmonicGrammar instance
            encodings: Encoding specifications (can include embeddings)
            opts: Network options (can include decay parameters)
            qpolicy: Commitment policy
            seed: Random seed
            embedding_manager: EmbeddingManager instance (optional)
            decay_manager: TemporalDecayManager instance (optional)
        """
        # Store managers before calling parent init
        self.embedding_manager = embedding_manager
        self.decay_manager = decay_manager

        # If embedding manager provided, update encodings
        if embedding_manager is not None:
            if encodings is None:
                encodings = {}

            # Get filler embeddings
            F = embedding_manager.get_filler_embeddings()
            similarity_matrix = embedding_manager.compute_similarity_matrix()

            encodings['F'] = F
            encodings['dim_f'] = F.shape[0]

            # Convert similarity matrix to list format
            if hg is not None:
                filler_names = hg.filler_names
                similarity_list = []
                n_fillers = len(filler_names)
                for i in range(n_fillers):
                    for j in range(i + 1, n_fillers):
                        similarity_list.append([
                            [filler_names[i], filler_names[j]],
                            similarity_matrix[i, j]
                        ])
                encodings['similarity'] = similarity_list

        # Add decay parameters to options
        if decay_manager is not None:
            if opts is None:
                opts = {}
            opts['decay_rate'] = decay_manager.decay_rate
            opts['decay_type'] = decay_manager.decay_type

        # Initialize parent class
        super().__init__(hg=hg, encodings=encodings, opts=opts,
                        qpolicy=qpolicy, seed=seed)

        # Track current position for decay
        self.current_position = 0

        # Store copy of external input at each position
        self.input_history = []

    def set_input_with_decay(self, binding_names, cumulative=True,
                            use_type=True, ignore_copy_symbols=True,
                            apply_decay=True):
        """Set external input with optional temporal decay

        This is an enhanced version of set_input that applies decay
        to previous inputs based on the decay manager settings.

        Args:
            binding_names: List of binding names to activate
            cumulative: If True, add to existing input; if False, replace
            use_type: Allow type matching (e.g., A matches A:0)
            ignore_copy_symbols: Ignore copy symbols
            apply_decay: If True, apply temporal decay to previous inputs
        """
        # Apply decay to existing input if enabled
        if apply_decay and self.decay_manager is not None:
            self._apply_temporal_decay()

        # Set new input using parent method
        self.set_input(binding_names, cumulative=cumulative,
                      use_type=use_type,
                      ignore_copy_symbols=ignore_copy_symbols)

        # Record this input
        if self.decay_manager is not None:
            self.decay_manager.record_activation(
                self.extC.copy(),
                position=self.current_position,
                time=self.t
            )

        # Store in history
        self.input_history.append({
            'position': self.current_position,
            'time': self.t,
            'input': self.extC.copy(),
            'binding_names': binding_names
        })

    def _apply_temporal_decay(self):
        """Apply temporal decay to current external input"""
        if self.decay_manager is None:
            return

        # Compute decay weights
        weights = self.decay_manager.compute_decay_weights(
            current_position=self.current_position,
            current_time=self.t,
            num_bindings=self.num_bindings
        )

        if weights is not None:
            # Apply decay to external input
            self.extC *= weights
            self.ext = self.C2N(self.extC)

    def advance_position(self):
        """Advance to next word position

        Call this after processing each word to update position counter
        """
        self.current_position += 1

        # Update scale constants for new position
        if hasattr(self, 'update_scale_constants'):
            self.update_scale_constants(pos=self.current_position)

    def reset_position(self):
        """Reset position counter to beginning"""
        self.current_position = 0
        self.input_history = []

        if self.decay_manager is not None:
            self.decay_manager.reset()

    def reset(self, mu=None, sd=0.):
        """Reset network (override parent to also reset position)"""
        super().reset(mu=mu, sd=sd)
        self.reset_position()

    def parse_sentence_incremental(self, sentence, commitment_schedule=None,
                                   reset_first=True, noise_sd=0.02,
                                   apply_decay=True):
        """Parse sentence incrementally with optional decay

        Args:
            sentence: List of binding names (e.g., ['DT/(1,1)', 'NN/(1,2)'])
            commitment_schedule: List of commitment values per word
                               (None = use default from qpolicy)
            reset_first: If True, reset network before parsing
            noise_sd: Standard deviation of initial noise
            apply_decay: If True, apply temporal decay between words

        Returns:
            Dict with parsing results
        """
        if reset_first:
            self.reset(self.ep, noise=noise_sd)

        n_words = len(sentence)

        if commitment_schedule is None:
            # Use default commitment schedule
            commitment_schedule = [self.qpolicy[i] for i in range(1, n_words + 1)]

        # Parse word by word
        for i, word_bindings in enumerate(sentence):
            # Update position
            self.current_position = i + 1

            # Set input for this word
            self.set_input_with_decay(
                word_bindings,
                cumulative=True,
                apply_decay=apply_decay and i > 0  # Don't decay first word
            )

            # Run network dynamics
            if i < len(commitment_schedule):
                q_level = commitment_schedule[i]
                self.runC(duration=q_level)

        # Extract final parse
        parse_result = {
            'actC': self.actC.copy(),
            'extC': self.extC.copy(),
            'final_position': self.current_position,
            'input_history': self.input_history.copy()
        }

        return parse_result

    def get_decay_strength_by_position(self):
        """Get decay strength for each input position

        Returns:
            Array of decay strengths (1.0 = no decay, 0.0 = fully decayed)
        """
        if self.decay_manager is None:
            return np.ones(len(self.input_history))

        strengths = []
        for hist in self.input_history:
            distance = self.current_position - hist['position']
            decay_factor = self.decay_manager._compute_decay_factor(distance)
            strengths.append(decay_factor)

        return np.array(strengths)


def create_enhanced_network(pcfg_string, root='S', max_sent_len=10,
                            embedding_dict=None, projection_dim=100,
                            decay_rate=0.1, seed=1024):
    """Convenience function to create enhanced GSC network

    Args:
        pcfg_string: PCFG in GSC format
        root: Root symbol
        max_sent_len: Maximum sentence length
        embedding_dict: Dict of symbol -> embedding vector
        projection_dim: Target dimension for embeddings (None = no projection)
        decay_rate: Temporal decay rate (0 = no decay)
        seed: Random seed

    Returns:
        GscNetEnhanced instance
    """
    # Create harmonic grammar
    hg = gsc.HarmonicGrammar(pcfg=pcfg_string, root=root,
                             max_sent_len=max_sent_len)

    # Create embedding manager if embeddings provided
    embedding_manager = None
    if embedding_dict is not None:
        embedding_manager = EmbeddingManager(
            embedding_dict=embedding_dict,
            embedding_dim=len(next(iter(embedding_dict.values()))),
            filler_names=hg.filler_names,
            projection_dim=projection_dim
        )

        # Create projection if needed
        if projection_dim is not None:
            embedding_manager.create_projection_matrix(method='random')

    # Create decay manager
    decay_manager = None
    if decay_rate > 0:
        decay_manager = TemporalDecayManager(
            decay_rate=decay_rate,
            decay_type='exponential',
            position_based=True
        )

    # Network options
    net_opts = {
        'T_init': 0.01,
        'q_max': 15.0,
        'q_init': 0.0,
        'dt_init': 0.005,
        'm': 30,
        'lam_x': 0.5,
        'lam_q': 0.04,
        'use_runC': True,
    }

    # Create enhanced network
    net = GscNetEnhanced(
        hg=hg,
        opts=net_opts,
        seed=seed,
        embedding_manager=embedding_manager,
        decay_manager=decay_manager
    )

    return net


if __name__ == '__main__':
    print("="*70)
    print("Enhanced GSC Network Demo")
    print("="*70)

    # Simple PCFG for testing
    PCFG_SIMPLE = '''
    0.40 S -> NP VP
    0.40 NP -> DT NN
    0.35 VP -> VBD
    '''

    # Create synthetic embeddings for POS tags
    pos_tags = ['S', 'NP', 'VP', 'DT', 'NN', 'VBD']
    embeddings = {}
    for pos in pos_tags:
        emb = np.random.randn(300)
        emb /= np.linalg.norm(emb)
        embeddings[pos] = emb

    print(f"\nCreated embeddings for {len(embeddings)} POS tags")
    print(f"Embedding dimension: 300")

    # Create enhanced network with embeddings and decay
    print("\nInitializing enhanced GSC network...")
    print("  - Static embeddings: YES (300 -> 100 dim projection)")
    print("  - Temporal decay: YES (rate = 0.15)")

    net = create_enhanced_network(
        pcfg_string=PCFG_SIMPLE,
        root='S',
        max_sent_len=5,
        embedding_dict=embeddings,
        projection_dim=100,
        decay_rate=0.15,
        seed=42
    )

    print(f"\nNetwork created:")
    print(f"  Fillers: {net.num_fillers}")
    print(f"  Roles: {net.num_roles}")
    print(f"  Bindings: {net.num_bindings}")

    if net.embedding_manager:
        F = net.embedding_manager.get_filler_embeddings()
        print(f"  Embedding matrix: {F.shape}")

    if net.decay_manager:
        print(f"  Decay rate: {net.decay_manager.decay_rate}")

    # Test incremental parsing with decay
    print("\n" + "="*70)
    print("Testing incremental parsing with decay")
    print("="*70)

    test_sentence = ['DT/(1,1)', 'NN/(1,2)', 'VBD/(1,3)']
    print(f"\nSentence: {' '.join([b.split('/')[0] for b in test_sentence])}")

    result = net.parse_sentence_incremental(
        test_sentence,
        apply_decay=True
    )

    print(f"\nParsing complete!")
    print(f"  Final position: {result['final_position']}")
    print(f"  Input history length: {len(result['input_history'])}")

    # Show decay strengths
    decay_strengths = net.get_decay_strength_by_position()
    print(f"\nDecay strengths by position:")
    for i, strength in enumerate(decay_strengths):
        print(f"  Position {i+1}: {strength:.3f} ({strength*100:.1f}% retained)")

    print("\n" + "="*70)
    print("Demo complete!")
    print("="*70)
