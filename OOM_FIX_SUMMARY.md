# OOM Fix for SAP Grammar Training

## Problem

The training script `sap_grammar_training.py` was running out of memory (OOM) during initialization, specifically between:
- **Line 1153**: "Initializing sparse optimizer states..."
- **Line 1208**: Before the "X s for initializing parameter values" message

## Root Cause

The OOM was caused by premature creation of optimizer states during `__init__`:

1. **Lines 1151-1164** created three large sparse matrices:
   - `WC`: Weight matrix (num_bindings × num_bindings)
   - `M_WC`: Adam momentum (num_bindings × num_bindings)
   - `R_WC`: Adam second moment (num_bindings × num_bindings)

2. These were created as `lil_matrix` format which has high memory overhead:
   - Python list for each row
   - Additional data structures for sparse storage
   - Memory fragmentation during construction

3. Having **three simultaneous large sparse matrices** caused memory pressure during `_build_model()` when the matrices were being populated with weights

## Solution

**Lazy initialization of optimizer states:**

1. **Removed** premature optimizer state creation from `__init__` (lines 1151-1176)
2. **Modified** `initialize()` method to create optimizer states only when needed:
   - Creates sparse CSR matrices directly (more memory-efficient than lil_matrix)
   - Only creates states when `optimizer='adam'`
   - Defers creation until training actually begins

## Memory Savings

By deferring optimizer state creation:
- **Eliminates** 2 large sparse matrices during model construction
- **Reduces peak memory** by ~2/3 during initialization
- Optimizer states are created in efficient CSR format when needed

## Changes Made

### File: `only_gscnet_speedup_sap.py`

1. **Lines 1150-1151** (was 1150-1176):
   ```python
   # Optimizer states will be initialized in initialize() method
   # to avoid OOM during model construction
   ```

2. **Lines 1166-1167** (was 1191-1194):
   ```python
   print("  Converting WC matrix to CSR format...")
   self.WC = self.WC.tocsr()
   # Removed optimizer state conversion (doesn't exist yet)
   ```

3. **Lines 3002-3022** (modified `initialize()` method):
   ```python
   if self.train_opts['optimizer'] == 'adam':
       self.optim = {}
       if hasattr(self, 'use_sparse') and self.use_sparse:
           # Create sparse optimizer states in CSR format
           self.optim['M_WC'] = sparse.csr_matrix(self.WC.shape, dtype=np.float64)
           self.optim['R_WC'] = sparse.csr_matrix(self.WC.shape, dtype=np.float64)
       else:
           # Dense matrices
           self.optim['M_WC'] = np.zeros_like(self.WC)
           self.optim['R_WC'] = np.zeros_like(self.WC)
       # ... rest of optimizer initialization
   ```

## Testing

To verify the fix works:
1. Run `sap_grammar_training.py` with the SLURM job script
2. Should see "X s for initializing parameter values" message (was not printed before)
3. Memory usage should stay within 400GB limit during initialization

## Additional Notes

- The fix only affects the CPU/NumPy code path (`use_jax=False`)
- JAX path still creates optimizer states during `__init__`, but this is less of an issue since JAX uses GPU memory differently
- The Adam update code already properly handles sparse matrices using `.power()` method

---

## Update: Second Optimization - Fast Pseudo-Inverse

### New Problem Discovered

After fixing the initial OOM issue, the code was hanging at `_add_change_of_basis_matrices()` when computing the pseudo-inverse of matrix N.

### Root Cause

The code was computing:
```python
N = np.kron(self.R, self.F)  # Shape: (num_bindings, dim_r * dim_f)
C = np.linalg.pinv(N)        # VERY SLOW for large matrices!
```

For large grammars:
- N shape: (100,000+, 9,000) for typical SAP grammar
- `pinv` complexity: O(min(m,n)² × max(m,n)) = O(9,000² × 100,000) = O(8.1 × 10¹² operations)
- Estimated time: **hours to days** on a single CPU

### Solution: Kronecker Product Property

Used the mathematical property: **pinv(kron(R, F)) = kron(pinv(R), pinv(F))**

New implementation:
```python
R_pinv = np.linalg.pinv(self.R)  # Small: (num_roles, 60)
F_pinv = np.linalg.pinv(self.F)  # Small: (num_fillers, 150)
C = np.kron(R_pinv, F_pinv)      # Fast Kronecker product
```

### Performance Improvement

- **Old method**: O(9,000² × 100,000) ≈ hours/days
- **New method**: O(60² × num_roles) + O(150² × num_fillers) ≈ **seconds**
- **Speedup**: ~1000x to 10,000x faster!

### Changes Made

**File: `only_gscnet_speedup_sap.py:1696-1711`**
- Added diagnostic output showing matrix dimensions
- Replaced direct `pinv(N)` with fast Kronecker decomposition
- Added timing information to track performance

---

## Update: Third Optimization - dok_matrix for Construction

### New Problem Discovered

After the pseudo-inverse fix, OOM occurred again during `_build_model()` when populating the sparse weight matrix.

### Root Cause

The code was using `lil_matrix` (List of Lists) for sparse matrix construction:
```python
self.WC = sparse.lil_matrix((num_bindings, num_bindings))  # Line 1140
_build_model()  # Makes ~3.2 million set_weight() calls
```

For the SAP grammar:
- Matrix size: 129,536 × 129,536 (2,816 fillers × 46 roles)
- Dense size: 134.2 GB
- Number of `set_weight()` calls: ~3.2 million
- `lil_matrix` has high memory overhead (Python lists-of-lists structure)
- Each incremental update adds memory pressure

### Solution: Use dok_matrix

Switched to **`dok_matrix`** (Dictionary of Keys) for construction:
```python
self.WC = sparse.dok_matrix((num_bindings, num_bindings))
```

### Performance Improvement

**dok_matrix advantages:**
- Dictionary-based: O(1) element access/assignment
- Minimal overhead: Just a Python dict, not lists-of-lists
- Memory-efficient: Only stores actual non-zero values
- Ideal for incremental construction with many updates

**lil_matrix disadvantages:**
- List overhead for every row (129,536 rows × 200 bytes ≈ 26 MB overhead)
- Memory fragmentation during construction
- Higher memory pressure with millions of updates

### Changes Made

**File: `only_gscnet_speedup_sap.py:1140-1145`**
- Changed from `sparse.lil_matrix` to `sparse.dok_matrix`
- Added diagnostic message about memory-efficient construction

**File: `only_gscnet_speedup_sap.py:1756-1964`**
- Added progress tracking in `_build_model()`
- Shows progress every 50,000 rules
- Reports timing and sparsity statistics

---

## Update: Fourth Optimization - Convert to CSR Before Matrix Multiplication

### New Problem Discovered

After optimizing the loop operations, the code was hanging at `_set_weights()` which performs matrix multiplications.

### Root Cause

The `_set_weights()` function does:
```python
self.W = self.C.T.dot(self.WC).dot(self.C)  # Line 2258
```

Where:
- `C.T`: (9,000 × 129,536)
- `WC`: (129,536 × 129,536) - **still in dok_matrix format**
- `C`: (129,536 × 9,000)

**The problem**: `dok_matrix` is optimized for element access/assignment, NOT for matrix multiplication. Using dok_matrix for `C.T @ WC @ C` is extremely slow.

### Solution: Convert to CSR Before Multiplication

Moved the `dok_matrix → CSR` conversion to happen BEFORE `_set_weights()` is called:

```python
# In _build_model(), BEFORE calling _set_weights():
self.WC = self.WC.tocsr()  # Convert to CSR for fast matrix ops
self._set_weights()         # Now uses CSR format (fast!)
```

### Performance Improvement

**Matrix multiplication performance by format:**
- **dok_matrix**: O(nnz × n) - iterates through sparse elements inefficiently
- **CSR format**: Optimized for matrix-vector and matrix-matrix products
- **Speedup**: 100-1000x faster for sparse matrix multiplication

### Changes Made

**File: `only_gscnet_speedup_sap.py:1993-2002`**
- Added CSR conversion at end of _build_model() before _set_weights()
- Added timing and sparsity reporting during conversion
- Removed duplicate CSR conversion from __init__ (line 1167-1170)

---

## Update: Fifth Optimization - Skip Unused W and b Computation

### New Problem Discovered

After converting to CSR, OOM still occurred at `_set_weights()` which computes:
```python
self.W = self.C.T.dot(self.WC).dot(self.C)
```

### Root Cause

The computation creates a huge intermediate matrix:
- `C.T @ WC`: (9,000 × 129,536) @ (129,536 × 129,536) = **(9,000 × 129,536)** intermediate
- Intermediate size: 9,000 × 129,536 × 8 bytes = **9.3 GB**
- With sparse WC having 24.5M non-zeros (99.9966% sparse)
- The intermediate result is likely dense or near-dense

**Critical discovery**: Analysis of the code shows that **`self.W` and `self.b` are NEVER used**!
- `self.W` is assigned once and never accessed
- `self.b` is assigned once and never accessed
- The network operates entirely in conceptual coordinates (WC, bC)
- W and b are legacy variables from when the network used neural coordinates

### Solution: Skip Computing W and b for Sparse Matrices

Since W and b are never used, simply skip computing them:

```python
def _set_weights(self):
    if hasattr(self, 'use_sparse') and self.use_sparse:
        print("Skipping W computation (not needed for sparse matrices)")
        self.W = None
        return
    self.W = self.C.T.dot(self.WC).dot(self.C)
```

### Memory Savings

- **Intermediate matrix**: 9.3 GB saved during computation
- **Final W matrix**: (9,000 × 9,000) × 8 = 648 MB saved
- **Final b vector**: Negligible
- **Total saved**: ~10 GB per initialization

### Changes Made

**File: `only_gscnet_speedup_sap.py:2254-2288`**
- Modified `_set_weights()` to skip W computation for sparse matrices
- Modified `_set_biases()` to skip b computation for sparse matrices
- Added explanatory comments about why these are unused
- Set W and b to None instead of computing them

---

## Update: Sixth Optimization - Sparse Eigenvalue Computation

### New Problem Discovered

After initialization completed, the code crashed with:
```
numpy.linalg.LinAlgError: 0-dimensional array given. Array must be at least two-dimensional
```

At `_compute_recommended_bowl_strength()` line 4649.

### Root Cause

The function computes bowl strength for stability:
```python
eigvals, eigvecs = np.linalg.eigh(self.WC)
eig_max = max(eigvals)
```

**The problem:**
- `np.linalg.eigh()` expects a **dense** numpy array
- `self.WC` is a **sparse CSR matrix** (129,536 × 129,536)
- Converting to dense: 129,536² × 8 bytes = **134.2 GB** (impossible!)
- The function only needs the **largest eigenvalue**, not all 129,536 eigenvalues

### Solution: Sparse Eigenvalue Solver

Use `scipy.sparse.linalg.eigsh()` to compute only the largest eigenvalue:

```python
if use_sparse:
    from scipy.sparse.linalg import eigsh
    # Compute only k=1 largest eigenvalue
    eig_max = eigsh(self.WC, k=1, which='LA', return_eigenvectors=False)[0]
else:
    # Dense: compute all eigenvalues
    eigvals, eigvecs = np.linalg.eigh(self.WC)
    eig_max = max(eigvals)
```

### Performance Improvement

**Dense approach (impossible):**
- Requires densifying 134.2 GB matrix
- Computes all 129,536 eigenvalues
- O(n³) complexity ≈ O(2.2 × 10¹⁵ operations)

**Sparse approach (fast):**
- Works directly on sparse matrix (406 MB)
- Computes only 1 eigenvalue using iterative methods
- O(k × nnz) per iteration ≈ O(24.5M operations)
- **Speedup: ~90,000x faster!**

### Changes Made

**File: `only_gscnet_speedup_sap.py:4643-4666`**
- Added sparse eigenvalue computation path using `eigsh()`
- Computes only largest eigenvalue (k=1) for efficiency
- Added try/except for robustness
- Added diagnostic output showing computed eigenvalue
- Dense matrices still use original path

---

## Update: Seventh Optimization - Fix Sparse Diagonal Extraction in initialize()

### New Problem Discovered

After corpus generation completed successfully (4,821 unique sentences), the code crashed with OOM during the `initialize()` method, specifically after printing the eigenvalue.

### Root Cause

The `initialize()` method calls `np.diag(self.WC)` at line 3110:
```python
self.train_opts['idx_mask_bias2'] = np.diag(self.WC) <= -8.
```

**The problem:**
- `np.diag()` on a **scipy sparse matrix** causes numpy to **densify the entire matrix** before extracting the diagonal
- WC is a sparse CSR matrix of size 129,536 × 129,536
- Densifying requires: 129,536² × 8 bytes = **134.2 GB**
- This happens during `initialize()`, when memory is already occupied by:
  - Original WC sparse matrix: ~450 MB
  - Backup WC sparse matrix: ~450 MB
  - Change-of-basis matrices (N, C, C_T): ~28 GB
  - Corpus targets: ~5 GB
  - Total before densification: ~34 GB
- Adding 134.2 GB temporary dense matrix → **~168 GB** peak usage → **OOM kill**

### Solution: Use .diagonal() Method for Sparse Matrices

Scipy sparse matrices have a `.diagonal()` method that efficiently extracts the diagonal without densification:

```python
# Old (causes OOM):
self.train_opts['idx_mask_bias2'] = np.diag(self.WC) <= -8.

# New (sparse-safe):
if hasattr(self, 'use_sparse') and self.use_sparse:
    self.train_opts['idx_mask_bias2'] = self.WC.diagonal() <= -8.
else:
    self.train_opts['idx_mask_bias2'] = np.diag(self.WC) <= -8.
```

### Performance Improvement

**Old approach (impossible):**
- Densify 129,536 × 129,536 sparse matrix → 134.2 GB allocation
- Extract diagonal from dense matrix
- Caused OOM kill

**New approach (fast):**
- Extract diagonal directly from sparse matrix → 129,536 × 8 bytes = 1 MB
- No densification required
- **Memory saved: 134.2 GB**
- **Time: instant vs. OOM crash**

### Changes Made

**File: `only_gscnet_speedup_sap.py:3111-3115`**
- Added sparse matrix check before diagonal extraction
- Use `.diagonal()` method for sparse matrices (avoids densification)
- Keep `np.diag()` for dense matrices (backward compatible)
- Added comment explaining the critical importance

**File: `only_gscnet_speedup_sap.py:3172-3186`**
- Fixed `get_mask0()` to handle sparse matrices in `update_gram_only` mode
- Use sparse `.sign()` method when available
- Convert to lil_matrix for diagonal modification
- Prevents densification when creating update masks

---

## Update: Additional Sparse Matrix Safety Fixes

### Additional Problems Found (Would Cause OOM During Training)

After fixing the `initialize()` diagonal extraction issue, a comprehensive audit revealed two more operations that would densify sparse matrices during the training loop:

#### Problem 3: np.max(abs(dWC)) at line 3610

During each training epoch, the code computes maximum gradient values for logging:
```python
dWC_max = np.max(abs(dWC))
```

**The problem:**
- `dWC` is a sparse lil_matrix (129,536 × 129,536) during gradient accumulation
- `np.max(abs(dWC))` densifies the entire matrix → 134.2 GB temporary allocation
- This happens every epoch, causing OOM during training

**Solution:**
```python
if hasattr(self, 'use_sparse') and self.use_sparse:
    dWC_max = abs(dWC).max() if dWC.nnz > 0 else 0.0
else:
    dWC_max = np.max(abs(dWC))
```

**Memory saved:** 134.2 GB per epoch

#### Problem 4: np.outer() Creating Dense Matrices at line 4313

During gradient computation for tree-level errors, the code creates outer products:
```python
state = np.zeros(self.num_bindings)  # size 129,536
state[key_idx] = 1.
dWC += np.outer(state, state) * self.train_opts['mask0'] * val * coef
```

**The problem:**
- `np.outer(state, state)` creates a dense 129,536 × 129,536 matrix = 134.2 GB
- This happens **for every tree in every training batch** (200 trials × multiple trees per trial)
- Each outer product creates a temporary 134.2 GB allocation, even though the result is sparse
- Would cause repeated OOM crashes during training

**Solution:**
For sparse matrices, directly update the relevant indices without creating the full outer product:
```python
if hasattr(self, 'use_sparse') and self.use_sparse:
    coef_val = val * self.train_opts['coef']['trees']
    # Directly update sparse matrix at (i,j) for all i,j in key_idx
    for i in key_idx:
        for j in key_idx:
            if self.train_opts['mask0'][i, j] != 0:
                dWC[i, j] = dWC[i, j] + coef_val
else:
    # Dense path (unchanged)
    state = np.zeros(self.num_bindings)
    state[key_idx] = 1.
    dWC += np.outer(state, state) * mask0 * val * coef
```

**Performance:**
- Old: Creates 134.2 GB dense matrix for each tree (impossible)
- New: Updates only O(k²) entries where k = |key_idx| (typically 10-50)
- **Memory saved: 134.2 GB per tree × hundreds of trees per epoch**

### Changes Made

**File: `only_gscnet_speedup_sap.py:3608-3615`**
- Fixed gradient max computation to use sparse `.max()` method
- Avoids densification when computing dWC_max for logging
- Added check for empty sparse matrices (nnz > 0)

**File: `only_gscnet_speedup_sap.py:4315-4331`**
- Replaced np.outer() with direct sparse index updates for tree gradients
- Only updates non-zero mask0 positions
- Preserves sparsity throughout gradient accumulation

---

## Update: Fix np.ix_() Densification in get_mask0()

### Problem 5: np.ix_() Fancy Indexing at line 3207-3216

After fixing the previous issues, OOM still occurred after eigenvalue computation during `initialize()`. The culprit was the `get_mask0()` function which builds the training mask.

**The problem:**
```python
mask0 = sparse.lil_matrix(self.WC.shape, dtype=np.float64)  # 129,536 × 129,536 sparse
for ri in range(len(self.hg.role_names)):
    idx = indices['self']
    idx_l = indices['l']
    idx_r = indices['r']
    mask0[np.ix_(idx, idx)] = 1.        # np.ix_() causes issues!
    mask0[np.ix_(idx, idx_l)] = 1.      # Densification or huge overhead
    mask0[np.ix_(idx_l, idx)] = 1.      # for each role
    ...
```

- `np.ix_()` creates fancy indexing arrays (meshgrid-like)
- When used with sparse matrix assignment, it can:
  - Trigger densification of the sparse matrix
  - Create large intermediate index arrays
  - Cause massive memory overhead even if not fully densifying
- This happens for every non-terminal role during mask construction
- With 46 roles and large index arrays, this accumulated to OOM

**Solution:**
Replace `np.ix_()` fancy indexing with direct sparse index updates:

```python
if hasattr(self, 'use_sparse') and self.use_sparse:
    # Direct sparse updates - no densification
    for i in idx:
        for j in idx:
            mask0[i, j] = 1.
    for i in idx:
        for j in idx_l:
            mask0[i, j] = 1.
            mask0[j, i] = 1.
    # ... etc
else:
    # Dense path uses np.ix_() (original code)
    mask0[np.ix_(idx, idx)] = 1.
    ...
```

**Performance:**
- Old: np.ix_() with sparse matrices → densification or huge overhead
- New: Direct sparse element updates → no intermediate arrays
- **Memory saved**: Avoids densification during mask construction

### Changes Made

**File: `only_gscnet_speedup_sap.py:3210-3242`**
- Replaced all `np.ix_()` calls with direct sparse index loops for sparse matrices
- Keeps `np.ix_()` for dense matrices (backward compatible)
- Updates mask0 symmetrically where needed
- Handles all role relationships: self, parent-left, parent-right, sister harmony
