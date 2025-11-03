# Critical Bug in JAX Prefix Handling: Missing Type Expansion

## Issue Description

The JAX prefix handling implementation is missing the **type expansion** feature from the CPU version's `set_input()` method. This causes completely different behavior when processing prefixes.

## Root Cause

### CPU Version (`set_input()` - lines 3475-3492)

When `use_type=True` (the default), CPU expands filler types to all matching fillers:

```python
if use_type:
    binding_names_new = []
    for bname in binding_names:
        f, r = bname.split(self.hg.opts['bsep'])
        # Find all fillers matching this type
        fi_list = g.find_fillers_type(f, ignore_bracket=True,
                                       ignore_copy=True,
                                       ignore_pos_f=g.opts['use_pos_f'])
        fillers_target = self.hg.g.get_fillers(fi_list)
        if ignore_copy_symbols:
            fillers_target = [f for f in fillers_target
                            if self.hg.g.opts['copy'] not in f]
        b_list = [f + bsep + r for f in fillers_target]
        binding_names_new += b_list
    binding_names = binding_names_new
```

**Example:**
- Input: `'N:0/(1,1)'`
- Expands to: `['N:0/(1,1)', '*N:0/(1,1)', '#N:0/(1,1)', ...]`
- Sets external input on **multiple** bindings

### JAX Version (`_compute_external_input_jax()` - lines 2115-2141)

Current implementation only sets input to the **exact** binding:

```python
def _compute_external_input_jax(binding_name, net_params):
    binding_names = net_params['binding_names']
    estr = net_params['estr']

    try:
        binding_idx = binding_names.index(binding_name)  # Exact match only!
    except ValueError:
        return jnp.zeros(num_bindings)

    extC = jnp.zeros(num_bindings)
    extC = extC.at[binding_idx].set(estr)
    return extC
```

**Example:**
- Input: `'N:0/(1,1)'`
- Sets input on: `'N:0/(1,1)'` **only**
- Misses: `'*N:0/(1,1)'`, `'#N:0/(1,1)'`, etc.

## Observed Behavior

### Test Results

```
Testing with prefix: ['N:0']

CPU grid point: ['PP[1]:1/(1,1)', 'VP[2]:1/(1,2)', ...]  # Complex parse
JAX grid point: ['N:0/(1,1)', 'Vi:0/(1,2)', ...]        # Simple parse

Matches: 0/15 roles  ← Complete mismatch!
```

The 0% overlap indicates that CPU and JAX are solving fundamentally different problems due to different external input.

## Impact

**High Severity** - The prefix handling is not functioning correctly, causing:
1. JAX produces different parse trees than CPU
2. Prefix constraints are too weak in JAX (only one binding vs multiple)
3. Results cannot be compared between CPU and JAX with prefixes

## Solution

We need to implement type expansion in JAX. This requires:

1. **Add grammar information to `net_params`** in `_extract_net_params_for_jax()`:
   ```python
   # Add to net_params dict
   'grammar': net.hg.g,  # Or extract relevant filler type mappings
   'filler_type_map': {filler: get_matching_fillers(filler) for filler in ...}
   ```

2. **Update `_compute_external_input_jax()`** to expand types:
   ```python
   def _compute_external_input_jax(binding_name, net_params):
       f, r = binding_name.split(net_params['bsep'])

       # Expand to all matching filler types
       matching_fillers = net_params['filler_type_map'].get(f, [f])

       # Build list of expanded binding names
       expanded_bindings = [mf + net_params['bsep'] + r for mf in matching_fillers]

       # Set external input for all expanded bindings
       extC = jnp.zeros(net_params['num_bindings'])
       for bname in expanded_bindings:
           try:
               idx = net_params['binding_names'].index(bname)
               extC = extC.at[idx].set(net_params['estr'])
           except ValueError:
               pass  # Binding not found, skip

       return extC
   ```

3. **Precompute filler type mappings** since JAX can't call Python methods during JIT:
   ```python
   def _build_filler_type_map(net):
       """Precompute mapping from filler names to all matching type fillers."""
       filler_type_map = {}
       g = net.hg.g

       for filler in net.filler_names:
           # Extract base type (remove positional/copy markers)
           base_type = extract_base_type(filler)  # e.g., 'N:0' from '*N:0'

           # Find all fillers matching this type
           fi_list = g.find_fillers_type(base_type,
                                         ignore_bracket=True,
                                         ignore_copy=True,
                                         ignore_pos_f=g.opts['use_pos_f'])
           matching = [net.filler_names[i] for i in fi_list
                      if g.opts['copy'] not in net.filler_names[i]]

           filler_type_map[filler] = matching

       return filler_type_map
   ```

## Workaround (Temporary)

Until the fix is implemented:
1. **Only test with `prefix=[]`** (no prefix) - this works correctly
2. **Document the limitation** that JAX prefix handling differs from CPU
3. **Use CPU version** for prefix-based parsing

## Status

- **Identified**: 2025-11-03
- **Priority**: High
- **Complexity**: Medium (requires grammar structure access)
- **Estimated Fix Time**: 2-3 hours

## Related Files

- `gsc.py:2115-2141` - `_compute_external_input_jax()` (needs fix)
- `gsc.py:2277-2302` - `_extract_net_params_for_jax()` (needs grammar info)
- `gsc.py:3457-3499` - `set_input()` (reference implementation)
- `test_prefix_handling.py` - Test showing the bug
