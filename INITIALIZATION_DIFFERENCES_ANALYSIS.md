# Critical Initialization Differences Between gsc.py and SAP

## Executive Summary

Based on output file analysis, **gsc.py and SAP produce different WC matrices BEFORE training even begins**. This completely invalidates the previous numerical precision hypothesis - the models start with different weights, so they will obviously train to different solutions.

---

## 1. Observed Differences Before Training

### A. mask0 Non-Zero Entries

**gsc.py (original):**
```
mask0 non-zero entries: 40,095 / 164,025
```

**SAP (dense):**
```
mask0 non-zero entries: 36,450 / 164,025
```

**Difference**: 3,645 fewer entries in SAP's mask0 (8.9% fewer positions available for training)

### B. WC Matrix Statistics

**gsc.py (original):**
```
WC sum: -1308.000000
WC diagonal sum: -2828.000000
First 10 entries:
  (0, 0): -10.000000
  (1, 1): -10.000000
  (2, 2): -10.000000
  (3, 3): -10.000000
  (4, 4): -10.000000
```

**SAP (dense):**
```
WC sum: -818.000000
WC diagonal sum: -2338.000000
First 10 entries:
  (0, 0): -10.000000
  (1, 1): -4.000000   ← DIFFERENT!
  (2, 2): -4.000000   ← DIFFERENT!
  (3, 3): -10.000000
  (4, 4): -4.000000   ← DIFFERENT!
```

**Differences:**
- Total WC sum differs by -490 (-1308 vs -818)
- Diagonal sum differs by -490 (-2828 vs -2338)
- Specific diagonal positions differ by 6 points (e.g., -10 vs -4)

### C. Dynamics Test

**gsc.py:**
```
WC.dot(test_actC) sum: -636.8202978977
WC.dot(test_actC) first 5: [-3.74540119 -9.50714306 -5.42979096 -3.56107319  1.36228328]
```

**SAP:**
```
WC.dot(test_actC) sum: -401.6997954602
WC.dot(test_actC) first 5: [-3.74540119 -3.80285723 -1.03782731 -3.56107319  2.29839512]
```

**Differences:**
- Position 1: -9.507 vs -3.803 (difference of -5.704)
- Position 2: -5.430 vs -1.038 (difference of -4.392)
- Position 4: 1.362 vs 2.298 (difference of -0.936)

---

## 2. Root Cause Analysis

### Diagonal Value Formula

The diagonal of WC is set by `bias2weight()`:
```python
self.WC = self.WC + np.diag(2 * self.bC)
```

So diagonal value = 2 × bC value

**For positions (1, 1), (2, 2), (4, 4):**
- gsc.py diagonal: -10 → bC = -5
- SAP diagonal: -4 → bC = -2
- **Difference: bC differs by -3 at these positions**

### Affected Bindings

Based on filler ordering (column-major layout):
- Position 0: Filler '#:0' (root symbol)
- Position 1: Filler '*@:1' (empty/copy filler)
- Position 2: Filler '*BE:0' (copy filler)
- Position 3: Filler '*N:0' (copy filler)
- Position 4: Filler '*Vi:0' (copy filler)

**Pattern**: The difference affects **copy fillers** (marked with '*')

### Likely Culprit: _adjust_default_param_vals()

This function runs after `_build_model()` and modifies bC values for special cases. Key areas to investigate:

1. **Empty filler handling** - Different treatment of '*@' and other copy fillers
2. **Root binding adjustments** - Different H_root_illegitimate applications
3. **Terminal vs non-terminal illegitimate penalties** - Different logic for copy fillers

---

## 3. Code Differences to Investigate

### A. get_mask0() - mask0 Generation

**gsc.py (lines 5949-5967):**
```python
for role in self.role_names:
    idx = self.find_roles(role)
    mask0[np.ix_(idx, idx)] = 1.
    if not self.hg.roles.is_terminal(role):
        daughters = self.hg.roles.get_daughters(role)
        idx_l = self.find_roles(daughters['l'])
        idx_r = self.find_roles(daughters['r'])
        mask0[np.ix_(idx, idx_l)] = 1.
        mask0[np.ix_(idx_l, idx)] = 1.
        mask0[np.ix_(idx, idx_r)] = 1.
        mask0[np.ix_(idx_r, idx)] = 1.
        if self.train_opts['update_sister_harmony']:
            mask0[np.ix_(idx_l, idx_r)] = 1.
            mask0[np.ix_(idx_r, idx_l)] = 1.
```

**SAP (lines 3553-3586, dense path):**
```python
for ri in range(len(self.hg.role_names)):
    if not self.hg.roles.role_is_terminal[ri]:
        indices = self.get_role_and_daughter_indices_fast(ri)
        if indices != None:
            idx = np.array(indices['self'])
            idx_l = np.array(indices['l'])
            idx_r = np.array(indices['r'])
            # ... meshgrid logic ...
```

**CRITICAL DIFFERENCE**:
- gsc.py includes ALL roles (terminal and non-terminal) in self-role mask `mask0[np.ix_(idx, idx)] = 1.`
- SAP only processes non-terminal roles, so might skip terminal role self-connections

**This explains the 3,645 entry difference!**

### B. find_roles() vs get_role_and_daughter_indices_fast()

**gsc.py:**
```python
daughters = self.hg.roles.get_daughters(role)
idx_l = self.find_roles(daughters['l'])  # daughters['l'] is a LIST
idx_r = self.find_roles(daughters['r'])  # daughters['r'] is a LIST
```

**SAP:**
```python
indices = self.get_role_and_daughter_indices_fast(ri)
idx_l = np.array(indices['l'])  # Pre-computed, single daughter?
idx_r = np.array(indices['r'])  # Pre-computed, single daughter?
```

**Potential Issue**:
- `get_daughters()` may return MULTIPLE left/right daughters
- `get_role_and_daughter_indices_fast()` might only return ONE daughter
- This could miss connections, reducing mask0 entries

### C. _adjust_default_param_vals() - Copy Filler Handling

Need to check if gsc.py and SAP handle copy fillers differently in this function. The -3 difference in bC for positions 1, 2, 4 (all copy fillers) suggests different bias adjustments.

Possible scenarios:
1. Different H_copy_illegitimate application
2. Different empty filler bias settings
3. Different root symbol adjustments for copy fillers

---

## 4. Impact on Training

### Why This Matters

**Different initial WC → Different gradient landscapes → Different local minima**

Even with identical training code, starting from different WC matrices will produce completely different trained models.

**Specific impacts:**
1. **Fewer trainable positions** (36k vs 40k) means SAP can't learn as rich of a model
2. **Different copy filler biases** (-5 vs -2) changes how the network handles phrase copying
3. **Missing terminal role connections** may prevent learning certain grammatical patterns

### S3/S4 Parsing Failure Explanation

S3 and S4 require complex prepositional phrase (PP) structures:
- S3: `[S [N] [VP [BE] [VPpp [Vpp] [PP [P] [N]]]]]`
- S4: `[S [NP [N] [RC [Vpp] [PP [P] [N]]]] [Vi]]`

If SAP's mask0 is missing connections for:
- PP-related bindings
- Copy operations for nested structures
- Specific role-daughter relationships

Then the network **literally cannot learn the weights** needed for S3/S4 parsing, because those positions are excluded from training (mask0 = 0).

---

## 5. Recommended Fixes

### Fix 1: Correct get_mask0() to Match gsc.py Logic

**In SAP's get_mask0(), add terminal role self-connections:**

```python
# BEFORE (SAP - incorrect):
for ri in range(len(self.hg.role_names)):
    if not self.hg.roles.role_is_terminal[ri]:
        indices = self.get_role_and_daughter_indices_fast(ri)
        # ... process non-terminals only ...

# AFTER (corrected):
for ri in range(len(self.hg.role_names)):
    # Add self-role connections for ALL roles (including terminals)
    idx = self.role_to_binding_indices[ri]
    rows_self, cols_self = np.meshgrid(idx, idx, indexing='ij')
    row_list.append(rows_self.ravel())
    col_list.append(cols_self.ravel())

    # Then process daughter connections for non-terminals
    if not self.hg.roles.role_is_terminal[ri]:
        indices = self.get_role_and_daughter_indices_fast(ri)
        # ... process daughters ...
```

### Fix 2: Verify get_role_and_daughter_indices_fast() Returns ALL Daughters

Check if this function returns all daughters or just the first one:

```python
def get_role_and_daughter_indices_fast(self, role_idx):
    # Ensure this returns ALL left and right daughters, not just one
    # daughters['l'] and daughters['r'] may be LISTS
```

### Fix 3: Match _adjust_default_param_vals() Logic

Compare the full `_adjust_default_param_vals()` implementations line-by-line to find where copy filler biases diverge.

Specifically check:
1. How empty fillers ('*@') are handled
2. How copy fillers ('*BE', '*N', '*Vi', '*Vpp') get their biases set
3. Whether H_copy_illegitimate is applied correctly

### Fix 4: Add Diagnostic Logging

Add to both gsc.py and SAP at end of `__init__()`:

```python
print("\n=== WC INITIALIZATION DIAGNOSTICS ===")
print(f"WC sum: {self.WC.sum():.6f}")
print(f"WC diagonal sum: {np.diag(self.WC).sum():.6f}")
print(f"bC sum: {self.bC.sum():.6f}")
print("First 10 diagonal entries:")
for i in range(10):
    print(f"  ({i},{i}): WC={self.WC[i,i]:.2f}, bC (before bias2weight)={self.bC[i]:.2f if hasattr(self, 'bC_before_bias2weight') else 'N/A'}")
    print(f"    Binding: {self.binding_names[i]}")
print("="*40)
```

---

## 6. Verification Steps

### Step 1: Count mask0 Entries

```python
# In both gsc.py and SAP, after get_mask0():
mask0 = self.get_mask0()
print(f"mask0 non-zero entries: {np.count_nonzero(mask0):,} / {mask0.size:,}")
```

Expected: Both should print `40,095 / 164,025`

### Step 2: Compare bC Before bias2weight

```python
# In _build_model(), right before bias2weight():
self.bC_before_bias2weight = self.bC.copy()  # Save for diagnostics

# After bias2weight(), check specific positions:
for i in [0, 1, 2, 3, 4]:
    print(f"Position {i} ({self.binding_names[i]}): bC_before={self.bC_before_bias2weight[i]}, WC_diag={np.diag(self.WC)[i]}")
```

Expected: All positions should match between gsc.py and SAP

### Step 3: Compare Full WC Matrices

```python
# After both networks are created:
diff = net_gsc.WC - net_sap.WC
print(f"WC matrices differ at {np.count_nonzero(diff)} positions")
if np.count_nonzero(diff) > 0:
    print(f"Max difference: {np.abs(diff).max()}")
```

Expected: `WC matrices differ at 0 positions`

---

## 7. Next Steps

1. **Fix get_mask0()** to include terminal role self-connections
2. **Verify get_role_and_daughter_indices_fast()** returns all daughters
3. **Add diagnostics** to print WC initialization statistics
4. **Re-run training** with fixed SAP version
5. **Compare results** - S3/S4 parsing should improve dramatically

---

## Conclusion

The root cause is **NOT numerical precision** but rather **incorrect mask0 generation** and possibly **incorrect bias initialization** for copy fillers.

SAP's get_mask0() excludes 3,645 trainable positions that gsc.py includes, and sets different bias values for copy fillers. This prevents the network from learning the necessary weights for complex nested structures like S3 and S4.

**Fix priority**: Correct get_mask0() first (high impact, easy fix), then investigate copy filler bias differences.
