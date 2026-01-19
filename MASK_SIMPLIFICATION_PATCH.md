# Mask Simplification Patch

If `get_mask0()` hangs indefinitely, you can simplify it to use a less detailed mask.

## Option A: Skip Complex Parent-Daughter Relationships

Edit `only_gscnet_speedup_sap.py` around line 3613:

```python
# ORIGINAL (detailed but slow):
if not self.hg.roles.role_is_terminal[ri]:
    indices = self.get_role_and_daughter_indices_fast(ri)
    if indices != None:
        # ... lots of meshgrid operations

# SIMPLIFIED (faster but less grammatically constrained):
if not self.hg.roles.role_is_terminal[ri]:
    # Skip parent-daughter connections for roles with >5000 bindings
    idx = self.role_to_binding_indices[ri]
    if len(idx) > 5000:
        print(f"      Skipping parent-daughter for role {ri} (too many bindings: {len(idx)})")
        continue
    indices = self.get_role_and_daughter_indices_fast(ri)
    # ... rest of code
```

## Option B: Use Dense Self-Attention Only

Replace the entire batch loop (lines 3598-3651) with:

```python
# Simplified: Only diagonal self-attention, no parent-daughter
for ri in range(total_roles):
    idx = self.role_to_binding_indices[ri]
    if len(idx) > 0:
        idx_array = np.array(idx)
        rows_self, cols_self = np.meshgrid(idx_array, idx_array, indexing='ij')
        row_list.append(rows_self.ravel())
        col_list.append(cols_self.ravel())
```

This creates a mask that only connects bindings within the same role, not across roles.

## Option C: Universal Mask (fastest but least constrained)

```python
def get_mask0_simple(self):
    """Simplified mask: allow all connections (identity-like)."""
    if hasattr(self, 'use_sparse') and self.use_sparse:
        # Allow updates everywhere - let gradient learning figure out structure
        mask0 = sparse.csr_matrix(np.ones(self.WC.shape, dtype=np.float64))
    else:
        mask0 = np.ones(self.WC.shape)
    return mask0
```

Then modify line 3488:
```python
self.train_opts['mask0'] = self.get_mask0_simple()  # Use simplified version
```

## Trade-offs

| Mask Type | Speed | Grammatical Constraints | Accuracy |
|-----------|-------|-------------------------|----------|
| Detailed (original) | Slow | Strong | Best |
| Simplified (skip large roles) | Medium | Medium | Good |
| Self-attention only | Fast | Weak | OK |
| Universal | Fastest | None | May work |

For large grammars like SAP, the **simplified** or **self-attention** approach is often necessary.
