## Why We CANNOT Copy g's Cache to g0

### The Question
Can we just copy g's cache to g0 instead of building separate caches?

```python
self.g0.filler_name_to_idx = self.g.filler_name_to_idx.copy()
self.g0.rules_by_mother = self.g.rules_by_mother.copy()
# etc.
```

### The Answer: NO! ❌

**g0 and g are DIFFERENT after initialization!**

### Proof

After `_add_additional_rules()` completes:

**g0 (original):**
- `filler_names`: `['S', 'N', 'VP', 'V']` (4 fillers)
- `rules`: Original 1072 rules only

**g (augmented):**
- `filler_names`: `['S', 'N', 'VP', 'V', '*S', '*N', '*VP', '*V', ...]` (many more!)
- `rules`: Original 1072 rules + copy rules + binary rules + etc.

### Why Copying Would Break Things

#### Problem 1: Filler Indices Would Be Invalid

If we copy `g.filler_name_to_idx` to `g0`:

```python
g.filler_name_to_idx = {
    'S': 0, 'N': 1, 'VP': 2, 'V': 3,
    '*S': 4, '*N': 5, '*VP': 6, '*V': 7,  # ← g0 doesn't have these!
    ...
}
```

Then `g0.filler_name_to_idx['*S']` would return index 4, but:
- `g0.filler_names[4]` doesn't exist (g0 only has 4 fillers)
- `g0.filler_is_terminal[4]` would be out of bounds
- **Runtime errors!**

#### Problem 2: Rule Indices Would Reference Non-Existent Rules

If we copy `g.rules_by_mother` to `g0`:

```python
g.rules_by_mother = {
    'S': [rule1, rule2, ...],
    '*S': [copy_rule1, copy_rule2, ...],  # ← g0 doesn't have these!
    ...
}
```

Then when `g0.generate_sentence()` tries to expand `'*S'`:
- `g0.rules_by_mother['*S']` would return copy rules
- But those rules don't exist in `g0.rules`
- **Invalid grammar state!**

### The Real Difference

**What happens in `_add_additional_rules()` (line 1648):**

```python
Line 1880: for rule in self.g.rules:        # ← Operates on g, NOT g0!
Line 1963: self.g.rules = rules_new + rules_copy  # ← Modifies g, NOT g0!
Line 1965: self.g._add_names()             # ← Rebuilds g.filler_names, NOT g0!
```

**Result:**
- `g0` stays **original** (1072 rules, 27 fillers)
- `g` becomes **augmented** (many more rules, many more fillers)

### The Correct Solution

Build **separate caches** for g0 and g because they have:
1. **Different filler lists** (g has copy symbols, g0 doesn't)
2. **Different rule sets** (g has augmented rules, g0 has original only)
3. **Different purposes** (g0 for generation, g for training)

```python
Line 1654: self.g._create_fastER_lookups_pcfg()   # ✓ g's cache for augmented grammar
Line 1658: self.g0._create_fastER_lookups_pcfg()  # ✓ g0's cache for original grammar
```

Each cache is built from its own `filler_names` and `rules`, ensuring consistency.

### Test Results

Running `test_g0_vs_g_content.py` shows:
- **g0.filler_names**: 4 original fillers
- **g.filler_names**: 12 fillers (includes copy symbols)
- **g0.rules**: 2 original rules
- **g.rules**: 4 rules (includes copy rules)

**Conclusion:** They are fundamentally different objects that need separate caches.
