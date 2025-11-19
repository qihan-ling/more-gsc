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
