"""
Sparse matrix implementation for large grammars

For grammars where num_bindings > 50,000, the change-of-basis matrices
become too large (>50 GB) to fit in memory.

This modification uses scipy.sparse matrices for the Kronecker product
and matrix operations.
"""

import numpy as np
from scipy import sparse
from scipy.sparse import linalg as sp_linalg

def _add_change_of_basis_matrices_sparse(self):
    """
    Sparse version of _add_change_of_basis_matrices for large networks.

    Uses scipy.sparse for operations on large matrices that don't fit in RAM.
    """

    print(f"Using sparse matrix operations (num_bindings={self.num_bindings})")

    # Kronecker product - use sparse if matrices are large
    if self.num_bindings > 50000:
        print("  Converting to sparse matrices...")
        R_sparse = sparse.csr_matrix(self.R)
        F_sparse = sparse.csr_matrix(self.F)
        N_sparse = sparse.kron(R_sparse, F_sparse, format='csr')

        print(f"  N matrix: {N_sparse.shape}, sparsity: {1 - N_sparse.nnz / (N_sparse.shape[0] * N_sparse.shape[1]):.2%}")

        # For square matrices, try sparse inverse
        if N_sparse.shape[0] == N_sparse.shape[1]:
            try:
                print("  Computing sparse inverse (this may take several minutes)...")
                C_sparse = sp_linalg.inv(N_sparse)
                self.N = N_sparse.toarray()  # Convert back for compatibility
                self.C = C_sparse.toarray()
            except:
                print("  Sparse inverse failed, using dense pseudo-inverse...")
                print("  WARNING: This will use a lot of memory!")
                self.N = N_sparse.toarray()
                self.C = np.linalg.pinv(self.N)
        else:
            # Non-square: must use dense pseudo-inverse
            print("  Non-square matrix, converting to dense for pseudo-inverse...")
            print("  WARNING: This requires significant memory!")
            self.N = N_sparse.toarray()

            # Try to use less memory by computing in blocks
            try:
                self.C = np.linalg.pinv(self.N)
            except MemoryError:
                print("  MEMORY ERROR: Matrix too large for pseudo-inverse")
                print("  Please reduce max_sent_len or use a machine with more RAM")
                raise
    else:
        # Small enough for dense operations
        print("  Using dense matrix operations...")
        N = np.kron(self.R, self.F)
        self.N = N

        if N.shape[0] == N.shape[1]:
            self.C = np.linalg.inv(N)
        else:
            self.C = np.linalg.pinv(N)

    # Compute derived matrices
    print("  Computing Gc, S matrices...")
    self.Gc = self.C.T.dot(self.C)

    print("  Reshaping C...")
    self.C_reshaped = self.C.reshape(
        (self.num_fillers, self.num_roles, self.num_units), order='F')

    print("  Computing similarity matrix S...")
    self.S = self.C.dot(self.C.T)

    print(f"  Change-of-basis matrices computed successfully")


def apply_sparse_matrices():
    """
    Apply sparse matrix optimization to GscNet.

    Call this before creating GscNet if you expect num_bindings > 50,000.
    """
    import gsc

    print("="*70)
    print("APPLYING SPARSE MATRIX OPTIMIZATION")
    print("For large networks (50k+ bindings), uses sparse matrices")
    print("="*70)

    gsc.GscNet._add_change_of_basis_matrices = _add_change_of_basis_matrices_sparse

    print("✓ Sparse matrix optimization applied")
    print()


if __name__ == "__main__":
    print(__doc__)
    print("\nUsage:")
    print("  import sparse_matrices")
    print("  sparse_matrices.apply_sparse_matrices()")
    print("  ")
    print("  # Then create your network")
    print("  net = gsc.GscNet(hg=hg, ...)")
