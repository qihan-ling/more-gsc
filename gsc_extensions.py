"""
GSC Extensions: Static Embeddings and Temporal Decay

This module provides extensions to the base GSC network for:
1. Static word embeddings (pre-computed from embedding models)
2. Temporal decay of activations (noisy context)

Author: Claude
Date: 2025-10-23
"""

import numpy as np
import time


class EmbeddingManager:
    """Manages static embeddings for GSC fillers

    Supports:
    - Loading pre-computed embeddings
    - Mapping words to filler symbols via embeddings
    - Dimensionality reduction (PCA, random projection)
    """

    def __init__(self, embedding_dict=None, embedding_dim=None,
                 filler_names=None, projection_dim=None):
        """Initialize embedding manager

        Args:
            embedding_dict: Dict mapping symbol names to embedding vectors
                           e.g., {'cat': np.array([...]), 'dog': np.array([...])}
            embedding_dim: Original dimension of embeddings
            filler_names: List of filler symbols to create embeddings for
            projection_dim: Target dimension after projection (None = no projection)
        """
        self.embedding_dict = embedding_dict or {}
        self.embedding_dim = embedding_dim
        self.filler_names = filler_names or []
        self.projection_dim = projection_dim

        self.projection_matrix = None
        self.filler_embeddings = None

        if embedding_dict and filler_names:
            self._build_filler_embeddings()

    def _build_filler_embeddings(self):
        """Build embedding matrix for all fillers"""
        n_fillers = len(self.filler_names)

        if self.projection_dim is not None:
            dim = self.projection_dim
        elif self.embedding_dim is not None:
            dim = self.embedding_dim
        else:
            # Infer from first available embedding
            if self.embedding_dict:
                dim = len(next(iter(self.embedding_dict.values())))
            else:
                dim = 300  # default

        self.filler_embeddings = np.zeros((dim, n_fillers))

        for i, filler in enumerate(self.filler_names):
            if filler in self.embedding_dict:
                emb = self.embedding_dict[filler]
                if self.projection_matrix is not None:
                    emb = self.project_embedding(emb)
                self.filler_embeddings[:, i] = emb
            else:
                # Random embedding for unknown fillers
                self.filler_embeddings[:, i] = np.random.randn(dim)
                self.filler_embeddings[:, i] /= np.linalg.norm(
                    self.filler_embeddings[:, i])

    def add_embedding(self, symbol, embedding):
        """Add embedding for a symbol

        Args:
            symbol: Symbol name (e.g., 'cat', 'DT')
            embedding: Embedding vector (numpy array)
        """
        self.embedding_dict[symbol] = embedding

        if symbol in self.filler_names:
            self._build_filler_embeddings()

    def create_projection_matrix(self, method='random', n_components=None):
        """Create projection matrix for dimensionality reduction

        Args:
            method: 'random' or 'pca'
            n_components: Target dimensionality
        """
        if n_components is None:
            n_components = self.projection_dim or 300

        self.projection_dim = n_components

        if method == 'random':
            # Random projection (Johnson-Lindenstrauss)
            self.projection_matrix = np.random.randn(
                n_components, self.embedding_dim) / np.sqrt(n_components)

        elif method == 'pca':
            # PCA requires data
            if not self.embedding_dict:
                raise ValueError("Need embeddings to compute PCA")

            # Stack all embeddings
            embeddings = np.array(list(self.embedding_dict.values()))

            # Center data
            mean = embeddings.mean(axis=0)
            centered = embeddings - mean

            # SVD
            U, S, Vt = np.linalg.svd(centered, full_matrices=False)

            # Take top n_components
            self.projection_matrix = Vt[:n_components, :]
            self.pca_mean = mean

        # Rebuild filler embeddings with projection
        if self.filler_names:
            self._build_filler_embeddings()

    def project_embedding(self, embedding):
        """Project embedding to lower dimension

        Args:
            embedding: Original embedding vector

        Returns:
            Projected embedding
        """
        if self.projection_matrix is None:
            return embedding

        if hasattr(self, 'pca_mean'):
            embedding = embedding - self.pca_mean

        return self.projection_matrix @ embedding

    def get_filler_embeddings(self):
        """Get embedding matrix for all fillers

        Returns:
            Matrix of shape (dim, n_fillers) where each column is a filler embedding
        """
        if self.filler_embeddings is None:
            self._build_filler_embeddings()

        return self.filler_embeddings

    def compute_similarity_matrix(self):
        """Compute pairwise similarity (dot product) matrix for fillers

        Returns:
            Similarity matrix of shape (n_fillers, n_fillers)
        """
        if self.filler_embeddings is None:
            self._build_filler_embeddings()

        # Normalize embeddings
        norms = np.linalg.norm(self.filler_embeddings, axis=0, keepdims=True)
        norms[norms == 0] = 1  # Avoid division by zero
        normalized = self.filler_embeddings / norms

        # Compute dot products
        similarity = normalized.T @ normalized

        return similarity


class TemporalDecayManager:
    """Manages temporal decay of activations

    Models working memory limitations by decaying older information.
    """

    def __init__(self, decay_rate=0.0, decay_type='exponential',
                 position_based=True):
        """Initialize decay manager

        Args:
            decay_rate: Decay rate parameter (λ)
                       0.0 = no decay, 0.1 = slow, 0.3 = fast
            decay_type: 'exponential', 'linear', or 'step'
            position_based: If True, decay based on position distance
                          If False, decay based on time steps
        """
        self.decay_rate = decay_rate
        self.decay_type = decay_type
        self.position_based = position_based

        # Track activation history
        self.activation_history = []
        self.time_history = []
        self.position_history = []

    def reset(self):
        """Reset decay history"""
        self.activation_history = []
        self.time_history = []
        self.position_history = []

    def compute_decay_weights(self, current_position=None, current_time=None,
                             num_bindings=None):
        """Compute decay weights for all previous activations

        Args:
            current_position: Current word position (for position-based decay)
            current_time: Current time step (for time-based decay)
            num_bindings: Number of binding units

        Returns:
            Decay weights array of shape (num_bindings,)
        """
        if num_bindings is None:
            if self.activation_history:
                num_bindings = len(self.activation_history[0])
            else:
                return None

        weights = np.ones(num_bindings)

        if self.decay_rate == 0.0:
            return weights

        if not self.activation_history:
            return weights

        if self.position_based and current_position is not None:
            # Decay based on position distance
            for i, pos in enumerate(self.position_history):
                distance = current_position - pos
                if distance > 0:
                    decay_factor = self._compute_decay_factor(distance)
                    # Apply to activations from that position
                    # (This is simplified - in practice, need to track which
                    # bindings were activated at each position)
                    weights *= decay_factor

        elif not self.position_based and current_time is not None:
            # Decay based on time elapsed
            for i, t in enumerate(self.time_history):
                time_elapsed = current_time - t
                if time_elapsed > 0:
                    decay_factor = self._compute_decay_factor(time_elapsed)
                    weights *= decay_factor

        return weights

    def _compute_decay_factor(self, distance):
        """Compute decay factor based on distance/time

        Args:
            distance: Distance in positions or time steps

        Returns:
            Decay factor (0 to 1)
        """
        if self.decay_type == 'exponential':
            return np.exp(-self.decay_rate * distance)

        elif self.decay_type == 'linear':
            factor = 1.0 - self.decay_rate * distance
            return max(0.0, factor)

        elif self.decay_type == 'step':
            # Step decay at certain threshold
            threshold = 1.0 / self.decay_rate if self.decay_rate > 0 else np.inf
            return 1.0 if distance < threshold else 0.5

        return 1.0

    def apply_decay(self, activations, current_position=None, current_time=None):
        """Apply decay to activations

        Args:
            activations: Current activation vector
            current_position: Current position
            current_time: Current time

        Returns:
            Decayed activations
        """
        weights = self.compute_decay_weights(
            current_position, current_time, len(activations))

        if weights is not None:
            return activations * weights

        return activations

    def record_activation(self, activations, position=None, time=None):
        """Record activation state for decay tracking

        Args:
            activations: Activation vector to record
            position: Position index
            time: Time step
        """
        self.activation_history.append(activations.copy())
        if position is not None:
            self.position_history.append(position)
        if time is not None:
            self.time_history.append(time)


def create_embedding_encoding(filler_names, embedding_manager):
    """Create encoding dict for GSC from embedding manager

    Args:
        filler_names: List of filler symbols
        embedding_manager: EmbeddingManager instance

    Returns:
        Dict suitable for passing to GscNet encodings parameter
    """
    # Get filler embedding matrix
    F = embedding_manager.get_filler_embeddings()

    # Compute similarity matrix
    similarity_matrix = embedding_manager.compute_similarity_matrix()

    # Convert similarity matrix to list format for GSC
    similarity_list = []
    n_fillers = len(filler_names)
    for i in range(n_fillers):
        for j in range(i + 1, n_fillers):
            if i != j:
                similarity_list.append([
                    [filler_names[i], filler_names[j]],
                    similarity_matrix[i, j]
                ])

    encoding = {
        'F': F,
        'similarity': similarity_list,
        'dim_f': F.shape[0]
    }

    return encoding


def add_decay_to_gsc_opts(opts, decay_rate=0.1, decay_type='exponential'):
    """Add decay parameters to GSC network options

    Args:
        opts: GSC network options dict
        decay_rate: Decay rate parameter
        decay_type: Type of decay function

    Returns:
        Updated opts dict
    """
    if opts is None:
        opts = {}

    opts['decay_rate'] = decay_rate
    opts['decay_type'] = decay_type

    return opts


# Example usage functions

def load_embeddings_from_file(filepath, max_words=None):
    """Load pre-computed embeddings from file

    Expected format: space-separated values
    word emb1 emb2 emb3 ...

    Args:
        filepath: Path to embeddings file
        max_words: Maximum number of words to load

    Returns:
        Dict mapping words to embedding vectors
    """
    embeddings = {}

    with open(filepath, 'r', encoding='utf-8') as f:
        for i, line in enumerate(f):
            if max_words and i >= max_words:
                break

            parts = line.strip().split()
            if len(parts) < 2:
                continue

            word = parts[0]
            vector = np.array([float(x) for x in parts[1:]])
            embeddings[word] = vector

    return embeddings


def create_pos_embeddings_from_words(word_embeddings, pos_to_words):
    """Create POS tag embeddings by averaging word embeddings

    Args:
        word_embeddings: Dict of word -> embedding
        pos_to_words: Dict of POS tag -> list of example words
                     e.g., {'NN': ['cat', 'dog', 'bird'],
                            'VBD': ['ran', 'sat', 'jumped']}

    Returns:
        Dict of POS tag -> embedding
    """
    pos_embeddings = {}

    for pos, words in pos_to_words.items():
        vectors = []
        for word in words:
            if word in word_embeddings:
                vectors.append(word_embeddings[word])

        if vectors:
            # Average embeddings
            pos_embeddings[pos] = np.mean(vectors, axis=0)
        else:
            # Random if no words found
            dim = len(next(iter(word_embeddings.values())))
            pos_embeddings[pos] = np.random.randn(dim)
            pos_embeddings[pos] /= np.linalg.norm(pos_embeddings[pos])

    return pos_embeddings


if __name__ == '__main__':
    # Example: Create synthetic embeddings for demonstration
    print("="*70)
    print("GSC Extensions: Embedding and Decay Demo")
    print("="*70)

    # Create synthetic POS embeddings (in practice, these come from real models)
    pos_tags = ['DT', 'NN', 'VBD', 'JJ', 'IN']
    embedding_dim = 300

    synthetic_embeddings = {}
    for pos in pos_tags:
        # Random embedding
        emb = np.random.randn(embedding_dim)
        emb /= np.linalg.norm(emb)
        synthetic_embeddings[pos] = emb

    print(f"\nCreated {len(synthetic_embeddings)} synthetic embeddings")
    print(f"Embedding dimension: {embedding_dim}")

    # Create embedding manager
    emb_manager = EmbeddingManager(
        embedding_dict=synthetic_embeddings,
        embedding_dim=embedding_dim,
        filler_names=pos_tags,
        projection_dim=100  # Project to 100 dims
    )

    print(f"\nCreating projection from {embedding_dim} to 100 dimensions...")
    emb_manager.create_projection_matrix(method='random')

    # Get filler embeddings
    F = emb_manager.get_filler_embeddings()
    print(f"Filler embedding matrix shape: {F.shape}")

    # Compute similarity
    sim = emb_manager.compute_similarity_matrix()
    print(f"Similarity matrix shape: {sim.shape}")
    print(f"Sample similarities:")
    for i in range(min(3, len(pos_tags))):
        for j in range(i+1, min(4, len(pos_tags))):
            print(f"  {pos_tags[i]} - {pos_tags[j]}: {sim[i,j]:.3f}")

    # Test decay manager
    print("\n" + "="*70)
    print("Temporal Decay Demo")
    print("="*70)

    decay_manager = TemporalDecayManager(
        decay_rate=0.15,
        decay_type='exponential',
        position_based=True
    )

    print(f"Decay rate: {decay_manager.decay_rate}")
    print(f"Decay type: {decay_manager.decay_type}")

    # Simulate decay over positions
    print("\nDecay factors by position distance:")
    for distance in [0, 1, 2, 3, 5, 10]:
        factor = decay_manager._compute_decay_factor(distance)
        percent = factor * 100
        print(f"  Distance {distance}: {factor:.3f} ({percent:.1f}% retained)")

    print("\n" + "="*70)
    print("Demo complete!")
    print("="*70)
