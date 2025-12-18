## Analysis: g0 vs g and Cache Build Order

### Why Both g0 and g?

**g0 (original grammar):**
- The unchanged base grammar from the PCFG file
- Used for sentence generation (line 2528: `self.g0.generate_sentence()`)
- Stays constant throughout the lifetime of the HarmonicGrammar

**g (augmented grammar):**
- A copy of g0 that gets MODIFIED
- Has copy rules, binary rules, etc. added to it
- Used for training and binding creation
- Filler list changes in `_add_additional_rules()`

### Current Order Problem

**Current implementation:**
```python
Line 1642: g0 = PCFG(...)                      # g0 created (no cache)
Line 1647: g0._create_fastER_lookups_pcfg()    # Build cache for g0
Line 1649: g = copy.deepcopy(g0)               # g gets COPY of g0's cache ❌ WASTED!
...
Line 1660: g._create_fastER_lookups_pcfg()    # REBUILD g's cache (overwrites copy)
```

**The problem:**
1. g0's cache is built at line 1647
2. deepcopy copies g0's cache to g at line 1649
3. g's filler list changes in `_add_additional_rules()` (line 1965)
4. g's cache is REBUILT at line 1660, **discarding the copied cache**

**Result:** The deepcopy of g0's cache is WASTED - it gets immediately thrown away.

### Optimal Order

**Better approach:**
```python
Line 1642: g0 = PCFG(...)                      # g0 created (no cache)
Line 1649: g = copy.deepcopy(g0)               # g copied (no cache yet)
...
Line 1660: g._create_fastER_lookups_pcfg()    # Build g's cache FIRST
Line 16XX: g0._create_fastER_lookups_pcfg()   # Build g0's cache AFTER
```

**Why this is better:**
1. No wasted deepcopy of cache
2. g0's cache built only once (not copied)
3. g's cache built once after filler list stabilizes
4. More logical: build caches when actually needed

### Performance Impact

The current order doesn't hurt correctness, but:
- **Wastes memory** during deepcopy (copies cache that gets discarded)
- **Wastes time** copying the cache dictionaries/arrays
- For a grammar with 1072 rules and 27 fillers:
  - rules_by_mother dict: ~1072 entries
  - filler_name_to_idx dict: 27 entries
  - Several numpy arrays: 27 entries each
  - Total wasted copy: ~1KB of data

Not huge, but unnecessary.

### Recommendation

Move `g0._create_fastER_lookups_pcfg()` to AFTER `g._create_fastER_lookups_pcfg()`:

```python
Line 1654: _add_additional_rules()            # Filler list stabilizes here
Line 1659: g._create_fastER_lookups_pcfg()   # Build g's cache
Line 1660: g0._create_fastER_lookups_pcfg()  # Build g0's cache (for generation)
```

This is cleaner and more efficient.
