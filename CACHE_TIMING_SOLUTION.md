### INITIALIZATION FLOW ANALYSIS

## Problem
The fast lookup cache (`filler_name_to_idx`, `filler_is_terminal`, etc.) is created at line 1664,
AFTER all rule-adding functions complete. This means all rule-adding functions fall back to slow
O(n) operations.

## Current Flow

```
Line 1642: self.g0 = PCFG(...)                    ← Initial filler list
Line 1643: self.g = copy.deepcopy(self.g0)
Line 1645: self._add_names()                      ← Copies filler list to HG

Line 1648: self._add_additional_rules()           ← FILLER LIST CHANGES HERE!
           └─ Line 1960: self.g._add_names()      ← Rebuilds filler_names from rules
                                                     (adds copy symbols)

Line 1650: self._add_binary_rules()               ← Cache doesn't exist (slow)
Line 1653: self._add_copy_rules()                 ← Cache doesn't exist (slow)
Line 1656: self._add_unary_rules()                ← Cache doesn't exist (slow)
Line 1659: self._add_expansion_rules()            ← Cache doesn't exist (slow)

Line 1664: self.g._create_fastER_lookups_pcfg()   ← Cache created HERE (too late!)
```

## Key Insight

The filler list STABILIZES after `_add_additional_rules()` completes (line 1648).
- Before line 1648: Filler list may change (copy symbols added)
- After line 1648: Filler list is STABLE
- Lines 1650-1659: Need the cache, but it doesn't exist yet!

## Solution

Move the cache creation to RIGHT AFTER line 1648:

```
Line 1648: self._add_additional_rules()           ← Filler list stabilizes
Line 1649: self.g._create_fastER_lookups_pcfg()   ← BUILD CACHE HERE!
Line 1650: self._add_binary_rules()               ← Now has cache (fast!)
Line 1653: self._add_copy_rules()                 ← Now has cache (fast!)
Line 1656: self._add_unary_rules()                ← Now has cache (fast!)
Line 1659: self._add_expansion_rules()            ← Now has cache (fast!)
```

## Why This Works

1. **Filler list is stable**: `_add_additional_rules()` calls `_add_names()` which rebuilds
   the filler list. After this, no other function modifies the filler list.

2. **Cache won't be invalidated**: Since the filler list doesn't change after line 1648,
   the cache built at line 1649 will remain valid for all subsequent operations.

3. **All rule functions benefit**: `_add_binary_rules()`, `_add_copy_rules()`,
   `_add_unary_rules()`, and `_add_expansion_rules()` will all have access to the cache.

4. **_add_additional_rules() already optimized**: We manually added caching inside
   `_add_additional_rules()`, so it doesn't need the global cache.

## Expected Performance Impact

- **_add_binary_rules()**: Can use cached `is_terminal()`, `is_bracketed()` lookups
- **_add_copy_rules()**: Can use cached lookups
- **_add_unary_rules()**: Can use cached lookups (already optimized, but cache helps)
- **_add_expansion_rules()**: Can use cached lookups

Total expected speedup: Significant, as all O(n) operations become O(1).
