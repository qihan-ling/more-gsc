# Analysis: Why Higher Commitment Degrades S3/S4 Parsing

## Executive Summary

Higher commitment levels cause **complete parsing failure** (0.0 accuracy) for complex sentences S3 and S4, while simple sentences maintain high accuracy. This analysis explains the mechanism behind this counterintuitive behavior.

---

## 1. How Commitment Works in GSCNet

### The Dynamics Equation

The network state evolves according to:
```
dactC/dt = gradC = S × HGradC
```

Where `HGradC` (Harmonic gradient in conceptual space) has four components:

```python
HGradC = hgrad_g + hgrad_b + hgrad_q0 + hgrad_q1
```

#### Component Breakdown:

1. **Grammar term** (`hgrad_g`):
   ```python
   hgrad_g = WC.dot(actC) + bC + extC
   ```
   - Encodes learned grammar rules and current input
   - Drives the system toward grammatical structures

2. **Bowl term** (`hgrad_b`):
   ```python
   hgrad_b = bowl_strength * (bowl_center - actC)
   ```
   - Weak centering force (usually bowl_strength = 0)

3. **Commitment term Hq0** (`hgrad_q0`):
   ```python
   hgrad_q0 = -2 * q * actC * (1 - actC) * (1 - 2*actC)
   ```
   - **KEY PLAYER**: Forces binary (0 or 1) activation states
   - Strength proportional to `q` (commitment level)
   - Direction determined by `(1 - 2*actC)`:
     - If `actC > 0.5` → pushes toward 1
     - If `actC < 0.5` → pushes toward 0
   - Acts like a **digital quantizer**

4. **Uniqueness term Hq1** (`hgrad_q1`):
   ```python
   hgrad_q1 = -4 * m * actC * (ssq - 1)
   where ssq = sum(actCmat**2, axis=fillers)
   ```
   - Enforces one-filler-per-role constraint
   - Strength controlled by `m` (default = 30)

---

## 2. Commitment Schedule in Parsing

### How `dq` Controls Integration Time

In `test_parse_inc()`, the commitment schedule is set as:
```python
# From test scripts:
dq = np.ones(max_sent_len) * (float(t) / max_sent_len)
qpolicy = dq.cumsum()  # [dq, 2*dq, 3*dq, 4*dq, 5*dq]
```

For commitment level `t`:
- **t=1**: qpolicy = [0.2, 0.4, 0.6, 0.8, 1.0]
- **t=5**: qpolicy = [1.0, 2.0, 3.0, 4.0, 5.0]
- **t=12**: qpolicy = [2.4, 4.8, 7.2, 9.6, 12.0]

### Integration Duration

For each word at position `wpos`:
```python
q_increment = qpolicy[wpos] - qpolicy[wpos-1]
duration = q_increment / q_rate
# q_rate defaults to 1.0
```

During wrapup:
```python
duration_wrapup = (q_max - current_q) / q_rate
```

**Key insight**: Higher `t` means:
1. Longer integration time per word
2. Higher final `q` value during wrapup
3. **Stronger Hq0 force** pushing toward binary states

---

## 3. The Problem: Premature Digitization

### Attractor Basin Dynamics

The commitment term `Hq0` creates a **double-well potential** for each binding unit:
- Two stable states: 0 (off) and 1 (on)
- Unstable equilibrium at 0.5

**At low commitment** (q ≈ 1-3):
- Hq0 is weak
- System can explore intermediate states
- Grammar term `hgrad_g` has time to guide toward correct parse
- Can "correct course" if initially heading wrong direction

**At high commitment** (q ≈ 8-12):
- Hq0 is very strong
- Intermediate states collapse rapidly to 0 or 1
- Grammar term influence is overwhelmed
- **Early decisions become locked in**

### Why S3 and S4 Fail Specifically

#### S3: "N BE Vpp P N" (Passive with PP modifier)
- Structure: `[S [N] [VP [BE] [VPpp [Vpp] [PP [P] [N]]]]]`
- Requires building nested PP within VPpp
- **Critical dependency**: P and N must activate PP binding before BE-Vpp competition resolves

#### S4: "N Vpp P N Vi" (Complex with RC)
- Structure: `[S [NP [N] [RC [Vpp] [PP [P] [N]]]] [Vi]]`
- Requires RC and PP construction before final Vi attachment
- **Critical dependency**: Multiple nested structures must stabilize simultaneously

### The Failure Mechanism

```
Time progression during high-commitment parsing:

t=0: Reset to equilibrium, q=0
     actC ≈ ep (equilibrium point, ~0.5 for all bindings)

Word 1 "N": extC activated for N/(1,1)
     q increases rapidly to ~2.4 (for t=12 case)
     Hq0 = -2 * 2.4 * actC * (1-actC) * (1-2*actC)  [STRONG!]
     → N/(1,1) quickly locks to 1
     → Competing bindings quickly lock to 0
     ✓ This works fine for simple N

Word 2 "BE": extC for BE/(1,2)
     q now at ~4.8
     Hq0 even stronger
     → BE/(1,2) locks to 1
     BUT: Haven't seen P, N yet - can't form VPpp structure
     → Network forced to make premature structural choice

Word 3 "Vpp": extC for Vpp/(1,3)
     q now at ~7.2
     Hq0 = -2 * 7.2 * actC * (1-actC) * (1-2*actC)  [VERY STRONG!]
     → Vpp/(1,3) locks to 1
     BUT: Still haven't integrated P, N into PP
     → VPpp structure can't form correctly

Word 4 "P": extC for P/(1,4)
     q now at ~9.6
     Needs to activate PP[1]:1/(2,3) binding (level 2, pos 3)
     BUT: Earlier bindings already locked due to high q
     → PP can't propagate correctly through rigid structure
     → Grammar weights WC can't reshape frozen activations

Word 5 "N": extC for N:1/(1,5)
     q now at ~12.0 (maximum!)
     → Complete rigidity
     → PP[1]:1/(2,3) never activates strongly enough

Wrapup: duration = (12.0 - 12.0) / 1.0 = 0
     → Minimal additional integration
     → Frozen in incorrect parse
     → Result: 0.0 accuracy
```

---

## 4. Why Low Commitment Works

**At commitment t=1-4**:
```
Word 1 "N": q increases to 0.2-0.8
     Hq0 weak, actC can stay at intermediate values

Word 2 "BE": q = 0.4-1.6
     Still relatively weak digitization

Word 3 "Vpp": q = 0.6-2.4
     Moderate digitization, but grammar still has influence

Word 4 "P": q = 0.8-3.2
     NOW PP[1]:1/(2,3) can begin to activate
     Grammar term can still reshape earlier activations

Word 5 "N": q = 1.0-4.0
     PP structure now evident from P, N inputs

Wrapup: duration = (1.0-4.0 - q_current) / 1.0
     Significant integration time
     q gradually increases, allowing gentle digitization
     Grammar term can guide toward correct attractor
     → Result: High accuracy
```

---

## 5. Comparison with Reference Results

### Reference (JAX implementation, figure12_g1_ds_speedup_model_copy_trainjax_parsing.png):
- S3: Never drops below 0.2, maintains 0.6-1.0
- S4: Gradual decline from 0.9 to 0.1-0.3

### Current (Numpy/Sparse implementation):
- S3: Drops to 0.0 at t=7-11 (sparse), 0.5-0.7 (dense)
- S4: **Complete failure 0.0** from t=5 onwards (both versions)

**Hypothesis**: The JAX implementation may have had:
1. Different integration scheme (more stable numerics)
2. Different WC connectivity (better propagation paths for nested structures)
3. Different commitment dynamics (perhaps slower q increase rate)
4. More robust wrapup phase allowing error correction

---

## 6. Root Cause Analysis

### Primary Issue: Temporal Credit Assignment

The commitment schedule creates a **temporal mismatch**:
- **Early words** (N, BE, Vpp) processed with partial information
- **High q forces early digitization** based on incomplete context
- **Later words** (P, N) arrive too late to reshape frozen structure

### Secondary Issues:

1. **WC Connectivity**:
   - Sparse matrix may have insufficient connectivity for PP propagation
   - `WC[PP[1]:1, P:0]` and `WC[PP[1]:1, N:1]` weights may be too weak
   - See recent commits debugging "bad parsing results"

2. **Wrapup Duration**:
   - At high commitment, `duration_wrapup ≈ 0`
   - No time for error correction
   - Grammar term can't rescue incorrect early decisions

3. **Attractor Landscape**:
   - High q creates deep, narrow wells
   - Correct parse may require escaping local minimum
   - Insufficient noise/temperature to enable transitions

---

## 7. Diagnostic Predictions

If this analysis is correct, we should observe:

1. **Activation trajectories**:
   - At t=12: PP[1]:1 bindings never exceed ~0.3 activation
   - At t=3: PP[1]:1 bindings reach >0.7 during wrapup

2. **WC propagation**:
   - `WC.dot(actC)` contribution for PP bindings is weak
   - Especially in sparse version vs dense

3. **Timing**:
   - Errors lock in during words 3-4, not during wrapup
   - Wrapup has insufficient duration to fix (duration ≈ 0)

4. **Sentence complexity correlation**:
   - S0 (N Vi): No nesting → works at all commitments
   - S1 (N Vi P N): One PP level → fails at t=11-12 only
   - S2 (N BE Vpp): No PP → works at all commitments
   - S3 (N BE Vpp P N): PP nested in VPpp → fails t=5+
   - S4 (N Vpp P N Vi): PP in RC → fails t=5+

---

## 8. Recommended Fixes

### Option 1: Adaptive Commitment Schedule
```python
# Instead of uniform dq across words:
dq = adjust_by_sentence_length(base_commitment, remaining_words)
# Give more "settling time" after seeing full sentence
```

### Option 2: Commitment Annealing
```python
# Start with low q during word processing
# Increase q only during wrapup
qpolicy_word = [0.5, 1.0, 1.5, 2.0, 2.5]  # slow increase
qpolicy_wrapup = target_commitment  # final digitization
```

### Option 3: Two-Phase Parsing
```python
# Phase 1: Low-q exploration (q_max = 3)
run_sent(sent, qpolicy_explore)
# Phase 2: High-q refinement (q_max = target)
# Start from Phase 1 endpoint
continue_integration(q_target)
```

### Option 4: Fix WC Connectivity
```python
# Ensure PP propagation paths are strong
# May require retraining or targeted weight adjustment
debug_wc_structure(net, focus_on=['PP', 'VPpp', 'RC'])
```

---

## 9. Next Steps

1. **Trace activation dynamics** for S3 at t=3 vs t=12
2. **Compare WC connectivity** between working JAX version and current version
3. **Test commitment annealing** schedule
4. **Verify PP binding activation** during word 4-5 processing

Would you like me to implement any of these diagnostic tests?
