# JAX Implementation Status - Detailed Comparison

## Overview
This documents what functionality from the original `estimate_prob_inc` trial execution has been implemented in the JAX version, and what's still missing.

---

## Full Original Trial Flow

### Original Code Path (CPU):
```python
def estimate_prob_inc(self, prefix, num_trials, ...):
    for trial_id in range(num_trials):
        # 1. RESET
        self.reset(mu=self.ep, sd=self.train_opts['init_noise_mag'])

        # 2. PREFIX (if len(prefix) > 0)
        if len(prefix) > 0:
            self.run_prefix(prefix, update_q_discrete)

        # 3. WRAPUP
        self.run_wrapup(update_q_discrete)

        # 4. READ RESULT
        gp = self.read_grid_point()
        self.set_discrete_state(gp)
```

---

## 1. RESET Functionality

### Original `reset()` (lines 3019-3036):
```python
def reset(self, mu=None, sd=0.):
    self.dt = self.opts['dt_init']                           # ✅ Implemented
    self.q = self.opts['q_init'] * np.ones(self.num_roles)   # ✅ Implemented
    self.T = self.opts['T_init']                             # ✅ Implemented
    self.t = 0.                                              # ✅ Implemented
    self.update_scale_constants(pos=0)                       # ❌ MISSING - sets all to 1.0

    if mu is None:
        self.set_random_state()                              # ✅ Implemented (as ep + noise)
    else:
        self.set_state(mu=mu, sd=sd)                         # ✅ Implemented

    self.clear_input()                                       # ✅ Implemented (implicitly)

    if hasattr(self, 'traces'):
        del self.traces                                      # ⚪ N/A (not needed for trials)
```

**Missing:**
- `update_scale_constants(pos=0)` - Sets `self.scale_constants = np.ones(num_bindings)`
  - **Impact:** Scale constants control time scales for different roles
  - **For basic wrapup:** Not critical (all 1.0)
  - **For incremental parsing:** CRITICAL

---

## 2. PREFIX Handling

### Original `run_prefix()` (lines 5133-5137):
```python
def run_prefix(self, prefix, update_q_discrete):
    for wi, fname in enumerate(prefix):
        self.run_word(fname, wi + 1, update_q_discrete)      # ❌ COMPLETELY MISSING
        self.store.append({'actC': self.actC, 'q': self.q})
```

### Original `run_word()` (lines 5139-5161):
```python
def run_word(self, fname, wpos, symmetric=True, update_q_discrete=False):
    # 1. Create binding name
    bname = fname + self.hg.opts['bsep'] + '(1,%d)' % wpos   # ❌ MISSING

    # 2. Get commitment schedule
    qinc = self.qpolicy[wpos] - self.qpolicy[wpos - 1]      # ❌ MISSING
    self.opts['q_max'] = self.qpolicy[wpos]                 # ❌ MISSING

    # 3. Set external input for this word
    self.set_input(bname)                                    # ❌ MISSING

    # 4. Update scale constants (controls which roles are active)
    if self.train_opts['update_scale_constants']:
        self.update_scale_constants(pos=wpos, symmetric)     # ❌ MISSING

    # 5. Run dynamics for this word
    if self.opts['use_runC']:
        self.runC(duration, log_trace, update_q)             # ❌ MISSING (see below)
```

**Missing:**
- **Entire prefix handling pipeline**
- `set_input()` - Sets `extC` (external input) for specific bindings
- `update_scale_constants(pos=wpos)` - Focuses network on current word position
- `qpolicy` scheduling - Controls commitment strength during incremental parsing

---

## 3. WRAPUP Handling

### Original `run_wrapup()` (lines 5163-5186):
```python
def run_wrapup(self, update_q_discrete, log_trace, clear_input):
    # 1. Calculate duration
    dur = np.max(self.opts['q_max'] - self.q)               # ✅ Implemented

    # 2. Clear input
    if clear_input:
        self.clear_input()                                   # ✅ Implemented (implicitly)

    # 3. Update scale constants
    if self.train_opts['update_scale_constants']:
        self.update_scale_constants(pos=0)                   # ❌ MISSING

    # 4. Handle discrete q updates
    if update_q_discrete:
        update_q = False
        self.q = self.opts['q_max'] * np.ones(num_roles)    # ⚠️ Partially (always updates q)
    else:
        update_q = True                                      # ✅ Implemented

    # 5. Run dynamics
    if self.opts['use_runC']:
        self.runC(dur / q_rate, log_trace, update_q)         # ⚠️ Simplified version
```

**Missing:**
- `update_scale_constants(pos=0)` calls
- Proper handling of `update_q_discrete` flag
- Scale constants application option

---

## 4. Core Dynamics - `runC()` and `update_stateC()`

### Original `runC()` (lines 2974-3013):
```python
def runC(self, duration, update_T=True, update_q=True, ...):
    t_max = self.t + duration                                # ✅ Implemented

    while self.t < t_max:                                    # ✅ Implemented (as jax.lax.scan)
        self.update_stateC()                                 # ⚠️ Simplified (see below)

        if update_T and (self.opts['T_decay_rate'] > 0):
            self.update_T()                                  # ❌ MISSING (T annealing)

        if update_q:
            self.update_q()                                  # ✅ Implemented (simplified)

        # Convergence/divergence checks                      # ❌ MISSING
```

### Original `update_stateC()` (lines 3156-3177):
```python
def update_stateC(self):
    # CRITICAL: Note the S matrix multiplication!
    gradC = self.scale_constants * self.S.dot(self.HGradC())  # ⚠️ S matrix MISSING!

    self.t += self.dt                                        # ✅ Implemented
    self.actC = self.actC + self.dt * gradC                 # ✅ Implemented
    self.add_noiseC()                                        # ✅ Implemented
    self.actCmat = self.vec2mat()                            # ⚠️ Done inline
```

**Current JAX Implementation:**
```python
# My simplified version:
gradC = hgrad_g + hgrad_b + hgrad_q0 + hgrad_q1             # ⚠️ Direct sum
actC = actC + dt * gradC                                     # ✅ Correct

# What's MISSING:
gradC = scale_constants * S.dot(HGradC())                    # ❌ No S matrix!
                                                             # ❌ No scale_constants!
```

**Missing:**
- **S matrix**: `self.S = self.C.dot(self.C.T)` - Inverse similarity matrix
  - This transforms gradients properly when using distributed representations
  - **Impact:** Could produce wrong dynamics if representations have similarity structure
- **scale_constants multiplication**: Controls time scales per binding
  - **Impact:** Critical for incremental parsing, less critical for full wrapup

---

## 5. Gradient Computation - `HGradC()`

### Original `HGradC()` (lines 3414-3432):
```python
def HGradC(self, actC=None, q=None):
    # 1. Grammar weights and biases
    hgrad_g = self.WC.dot(actC) + self.bC + self.extC        # ✅ Implemented

    # 2. Bowl constraints
    hgrad_b = self.opts['bowl_strength'] * (
        self.opts['bowl_center'] - actC)                     # ❌ Set to 0.0 (MISSING)

    # 3. Commitment energy (q term)
    hgrad_q0 = -2 * self.extend_rvec(rvec=q) *
        actC * (1 - actC) * (1 - 2 * actC)                   # ✅ Implemented

    # 4. Role-filling constraint
    ssq = np.sum(actCmat ** 2, axis=0)
    hgrad_q1 = -4 * self.opts['m'] * actC *
        self.extend_rvec(rvec=ssq - 1)                       # ✅ Implemented

    return (hgrad_g + hgrad_b + hgrad_q0 + hgrad_q1)
```

**Status:**
- ✅ `hgrad_g`: Grammar component
- ❌ `hgrad_b`: Bowl constraints (set to 0)
- ✅ `hgrad_q0`: Commitment energy
- ✅ `hgrad_q1`: Role-filling constraint

---

## 6. Grid Point Extraction

### Original `read_grid_point()` (lines 3832-3845):
```python
def read_grid_point(self, actC=None):
    actCmat = self.vec2mat(actC=self.actC)                   # ✅ Implemented
    winner_idx = np.argmax(actCmat, axis=0)                  # ✅ Implemented
    winners = [self.filler_names[ii] for ii in winner_idx]  # ⚠️ Returns indices only
    winners = ["%s/%s" % bb for bb in
               zip(winners, self.role_names)]                # ⚠️ String conversion in CPU
    return winners
```

**Status:**
- ✅ Argmax computation correct
- ⚪ String conversion done in CPU code (acceptable)

---

## Summary Table

| Component | CPU Function | JAX Status | Critical? | Notes |
|-----------|--------------|------------|-----------|-------|
| **RESET** | | | | |
| - Initialize state | `set_state()` | ✅ Implemented | Yes | Working |
| - Scale constants | `update_scale_constants(0)` | ❌ Missing | Medium | Sets all to 1.0 |
| - Clear input | `clear_input()` | ✅ Implicit | Yes | Working |
| **PREFIX** | | | | |
| - Word loop | `run_prefix()` | ❌ Missing | High | Entire feature missing |
| - Binding creation | `run_word()` | ❌ Missing | High | Not implemented |
| - Set input | `set_input()` | ❌ Missing | High | No external input |
| - Scale constants | `update_scale_constants(pos)` | ❌ Missing | High | Critical for parsing |
| - qpolicy | qpolicy schedule | ❌ Missing | High | Commitment scheduling |
| **WRAPUP** | | | | |
| - Duration calc | `dur = max(q_max-q)` | ✅ Implemented | Yes | Working |
| - Clear input | `clear_input()` | ✅ Implicit | Yes | Working |
| - Scale constants | `update_scale_constants(0)` | ❌ Missing | Low | Less critical here |
| **DYNAMICS** | | | | |
| - Loop structure | `while t < t_max` | ✅ `lax.scan` | Yes | Working |
| - Gradient: WC, bC | `hgrad_g` | ✅ Implemented | Yes | Working |
| - Gradient: bowl | `hgrad_b` | ❌ Set to 0 | Medium | May matter |
| - Gradient: commitment | `hgrad_q0` | ✅ Implemented | Yes | Working |
| - Gradient: role-fill | `hgrad_q1` | ✅ Implemented | Yes | Working |
| - **S matrix** | `S.dot(grad)` | ❌ Missing | **HIGH** | Critical! |
| - **scale_constants** | `scale * grad` | ❌ Missing | **HIGH** | Critical! |
| - Noise | `add_noiseC()` | ✅ Implemented | Yes | Working |
| - q update | `update_q()` | ✅ Simplified | Yes | Working |
| - T update | `update_T()` | ❌ Missing | Low | Annealing |
| **OUTPUT** | | | | |
| - Grid point | `read_grid_point()` | ✅ Implemented | Yes | Working |

---

## Impact Analysis

### For `estimate_prob_inc([], num_trials)` (empty prefix, wrapup only):

**CRITICAL Missing:**
1. **S matrix transformation** - May produce incorrect results if similarity ≠ identity
2. **scale_constants** - Less critical for pos=0 (all would be 1.0 anyway)

**MEDIUM Missing:**
3. **Bowl constraints** - May affect convergence
4. **Temperature annealing** - May affect exploration

### For `estimate_prob_inc(prefix, num_trials)` (incremental parsing):

**COMPLETELY MISSING:**
- Entire prefix handling pipeline
- Would fail or produce garbage results

---

## Recommended Fix Priority

### Phase 1: Fix Critical Issues (Empty Prefix)
1. Add S matrix transformation to gradient
2. Add scale_constants multiplication
3. Add bowl constraints (hgrad_b)
4. Extract and pass these from network:
   - `net.S`
   - `net.opts['bowl_strength']`
   - `net.opts['bowl_center']`
   - `net.opts['m']`

### Phase 2: Enable Prefix Handling
1. Implement `set_input_jax()` for external input
2. Implement `update_scale_constants_jax()` for role masking
3. Add prefix loop before wrapup
4. Add qpolicy scheduling

### Phase 3: Optional Refinements
1. Temperature annealing
2. Convergence checks
3. Discrete q updates

---

## Current Test Recommendations

**DO NOT test with non-empty prefix yet** - it will fail or give garbage.

**For empty prefix `estimate_prob_inc([], num_trials)`:**
- May work but could have subtle bugs from missing S matrix
- Results might differ from CPU version
- Need to add S matrix before trusting results

