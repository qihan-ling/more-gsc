# Why only_gscnet_speedup.py Avoids Premature Digitization

## TL;DR

The working version (`only_gscnet_speedup.py` with JAX enabled) **likely doesn't actually avoid** premature digitization - rather, **JAX's numerical behavior creates more stable dynamics** that allow the network to recover from early mistakes during the wrapup phase, even at high commitment levels.

---

## Key Differences Between Working and Failing Versions

### Configuration Comparison

| Aspect | Working (cho_grammar1) | Failing (sap_grammar_test) |
|--------|------------------------|----------------------------|
| **File** | `only_gscnet_speedup.py` | `only_gscnet_speedup_sap.py` |
| **JAX** | Enabled (default) | Disabled (`use_jax: False`) |
| **Sparse** | No | Optional (tested both) |
| **Dynamics** | JAX JIT-compiled | NumPy loops |
| **Integration** | `jax.lax.fori_loop` | Python `while` loop |

### Test Setup (Identical)

Both use the **same** commitment schedule:
```python
dq = np.ones(max_sent_len) * (float(t) / max_sent_len)
qpolicy = dq.cumsum()
```

Both use the **same** grammar, same parameters (q_max=15, m=30, dt=0.005, T=0.01).

---

## Analysis: JAX vs NumPy Dynamics

### 1. Numerical Precision Differences

**JAX (float32 default):**
```python
@jit
def _dynamics_step_jax(...):
    # All operations in float32
    hgrad_q0 = -2 * jnp.repeat(q, num_fillers) * \
        actC * (1 - actC) * (1 - 2*actC)
    # ...
    actC_new = actC + dt * gradC
    return actC_new, q_new, rng_key
```

**NumPy (float64 default):**
```python
def update_stateC(self):
    hgrad = self.HGradC()  # float64
    temp = self.C_T.dot(hgrad)
    gradC = self.C.dot(temp)
    self.actC = self.actC + self.dt * gradC
```

**Impact**: Float32's lower precision may introduce **subtle numerical damping** that prevents runaway commitment effects.

### 2. Loop Structure Differences

**JAX Fast Path:**
```python
if self.use_jax and not log_trace:
    num_steps = int(duration / self.dt)

    def body_fun(i, carry):
        actC, q, rng_key = carry
        actC_new, q_new, rng_key_new = _dynamics_step_jax(...)
        return (actC_new, q_new, rng_key_new)

    # JIT-compiled fixed-length loop
    init_carry = (self.actC, self.q, self.rng_key)
    final_carry = jax.lax.fori_loop(0, num_steps, body_fun, init_carry)
```

**NumPy Path:**
```python
while self.t < t_max:
    self.update_stateC()  # Contains branching logic

    if update_T and (self.opts['T_decay_rate'] > 0):
        self.update_T()
    if update_q:
        self.update_q()
    if log_trace:
        self.update_traces()

    if self.check_divergence():  # Early exit possible
        break

    if tol is not None:
        self.check_convergence(tol=tol)
        if self.converged:
            break
```

**Impact**:
- JAX loop is **fully unrolled** and optimized by XLA compiler
- NumPy loop has **branching overhead** and potential early exits
- NumPy `check_divergence()` might terminate integration prematurely in high-q regimes

---

## Hypothesis: Why JAX Works Better

### Theory 1: Noise Handling

**JAX noise generation:**
```python
rng_key, subkey = jax.random.split(rng_key)
noise = jnp.sqrt(2 * T * dt) * jax.random.normal(subkey, shape=(...))
```
- Cryptographically secure RNG splitting
- Consistent noise across devices
- **No mutable state** (functional RNG)

**NumPy noise:**
```python
def add_noiseC(self):
    if self.use_jax:
        # [JAX path]
    else:
        noise = np.sqrt(2 * self.T * self.dt) * \
            np.random.randn(self.num_bindings)
        self.actC += noise
```
- Uses global random state
- **Potentially different noise patterns**
- Mutable RNG state

**Impact**: Different noise realizations could help JAX version **escape local minima** during high-commitment phases that NumPy version gets stuck in.

### Theory 2: Compiler Optimizations

XLA (JAX's compiler) may:
1. **Reorder operations** for numerical stability
2. **Fuse operations** reducing intermediate roundoff errors
3. **Optimize matrix multiplications** via better BLAS kernels
4. **Vectorize** more aggressively than NumPy

These could create **more stable attractor basins** that allow correction during wrapup.

### Theory 3: Wrapup Behavior

At high commitment (t=12), wrapup duration:
```python
duration_wrapup = (q_max - q) / q_rate
# After word 5: q ≈ 12.0, q_max = 12.0 (from qpolicy)
# → duration ≈ 0
```

**But this should be the same for both implementations!**

Unless... Let me check if there's a difference in how qpolicy is handled...

Actually, both implementations should have the same wrapup duration. So this isn't the answer.

### Theory 4: The REAL Difference - Check Divergence

Let me examine this more carefully:

```python
# NumPy version has:
if self.check_divergence():
    break
```

If `check_divergence()` triggers at high commitment due to strong Hq0 gradients, integration could **terminate early**, preventing the network from reaching a proper parse.

JAX fast path **skips divergence checking**:
```python
if self.use_jax and not log_trace and tol is None:
    # No divergence checking!
    final_carry = jax.lax.fori_loop(0, num_steps, body_fun, init_carry)
```

**This could be the key difference!**

---

## Testing the Hypothesis

### Prediction 1: Divergence Detection

If divergence checking is the culprit:
- NumPy version should show "divergence" warnings or early terminations at high commitment
- Disabling `check_divergence()` in NumPy version should improve S3/S4 performance

### Prediction 2: Noise Consistency

If noise differences matter:
- Setting the same random seed and using identical RNG should make results converge
- But the test scripts already use `np.random.seed(1024 + t)` before each parse...

### Prediction 3: Float Precision

If float32 vs float64 matters:
- Forcing NumPy to use float32 should improve results
- Forcing JAX to use float64 should worsen results

---

## Code Archaeology: What Changed?

Looking at git history:

**only_gscnet_speedup.py** (original):
- Focus on JAX optimization
- Commits: "fix JAX .at[] indexing", "fix rng_key in reset()"
- Stable since Nov 19 (when good figure was created)

**only_gscnet_speedup_sap.py** (fork):
- Created Nov 19: "duplicate speedup gscnet for sap"
- Immediate changes: "sparse matrix and reduce dimensions"
- Heavy debugging: "debugging sparse training's bad parsing results"
- **Diverged significantly** from original

### Key Insight

The SAP version was forked **specifically** to add sparse matrix support and handle OOM issues with large grammars. These modifications may have introduced subtle bugs or numerical instabilities.

---

## Most Likely Explanation

Based on the evidence, I believe the difference is **NOT** that JAX avoids premature digitization, but rather:

1. **Both versions experience premature digitization** at high commitment
2. **JAX version has more stable numerics** (float32, compiler opts, no divergence checking)
3. This stability allows the network to **continue integrating** even when gradients are large
4. The extra integration time provides opportunities for **noise-driven escape** from wrong attractors
5. **NumPy version hits divergence detection** and terminates early, freezing the incorrect parse

### The Smoking Gun: Divergence Detection

Check `only_gscnet_speedup_sap.py` for:
```python
def check_divergence(self):
    # If this returns True at high q, it aborts integration!
    # JAX fast path skips this entirely
```

This would explain:
- Why S3/S4 fail **completely** (0.0 accuracy) rather than partially
- Why simple sentences work (lower gradients, no divergence)
- Why the failure is sudden at specific commitment thresholds (where divergence triggers)

---

## Recommended Test

Run this experiment in `sap_grammar_training_test.py`:

```python
# TEST 1: Disable divergence checking
net_opts = {
    'use_jax': False,  # Keep as CPU
    'check_divergence_enabled': False,  # ADD THIS FLAG
    # ... rest of opts
}
```

If S3/S4 performance improves, we've found the culprit.

Alternatively:

```python
# TEST 2: Force JAX usage with SAP code
net_opts = {
    'use_jax': True,  # ENABLE JAX
    'use_sparse_wc': False,  # Disable sparse (JAX doesn't support it yet)
    # ... rest of opts
}
```

If this fixes the problem, it confirms JAX's numerical properties are protective.

---

## Conclusion

The `only_gscnet_speedup.py` version likely **does experience premature digitization**, but:

1. JAX's numerical behavior (float32, optimizations, no divergence checks) allows **continued integration**
2. This provides time for **stochastic escape** from incorrect attractors
3. **NumPy version aborts early** via divergence detection, cementing wrong parses

The "premature digitization" problem exists in both - JAX just has better **error recovery mechanisms**.
