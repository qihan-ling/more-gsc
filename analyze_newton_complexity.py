#!/usr/bin/env python3
"""
Analyze computational complexity of Newton vs Integration methods
"""
import numpy as np

print("="*70)
print("Computational Complexity Analysis: Newton vs Integration")
print("="*70)

# Current grammar (Grammar 1)
current_fillers = 27
current_roles = 15
current_bindings = current_fillers * current_roles

print(f"\nCurrent Grammar (Grammar 1):")
print(f"  Fillers: {current_fillers}")
print(f"  Roles: {current_roles}")
print(f"  Bindings: {current_bindings}")

# Estimated 1k-rule grammar
# A 1k-rule grammar might have ~50-100 unique non-terminals and terminals
# With position-specific fillers, this could be ~100-200 fillers
# Roles depend on max_sent_len (typically 10-20 for longer sentences)
estimated_fillers_1k = 150  # Conservative estimate
estimated_roles_1k = 20  # For max_sent_len=10
estimated_bindings_1k = estimated_fillers_1k * estimated_roles_1k

print(f"\nEstimated 1k-rule Grammar:")
print(f"  Fillers: ~{estimated_fillers_1k}")
print(f"  Roles: ~{estimated_roles_1k}")
print(f"  Bindings: ~{estimated_bindings_1k}")

print("\n" + "="*70)
print("Newton Method Complexity")
print("="*70)
print("""
Per iteration:
1. Compute gradient HGrad(): O(n²) - matrix-vector multiply
2. Compute Hessian HHess(): O(n²) - construct n×n matrix
3. Solve linear system: O(n³) - np.linalg.lstsq
4. Update solution: O(n)

Total per iteration: O(n³) dominated by linear solve
Typical iterations: 5-20
Total: O(k × n³) where k = iteration count
""")

n = current_bindings
ops_newton_current = 10 * (n**3)  # Assume 10 iterations average
print(f"Current grammar (n={n}):")
print(f"  Operations: ~{ops_newton_current:,.0f}")
print(f"  Time estimate: ~0.5s (measured)")

n = estimated_bindings_1k
ops_newton_1k = 10 * (n**3)
scaling_factor = ops_newton_1k / ops_newton_current
print(f"\n1k-rule grammar (n={n}):")
print(f"  Operations: ~{ops_newton_1k:,.0f}")
print(f"  Scaling: {scaling_factor:.1f}x more operations")
print(f"  Time estimate: ~{0.5 * scaling_factor:.1f}s per get_ep() call")

print("\n" + "="*70)
print("Integration Method Complexity")
print("="*70)
print("""
For fixed duration:
1. Number of steps: dur / dt
2. Per step: O(n²) - matrix-vector multiply (WC.dot(actC))
3. Total: O((dur/dt) × n²)

Typical values: dur=10, dt=0.005
Steps: 10/0.005 = 2000 steps
""")

dur = 10
dt = 0.005
steps = int(dur / dt)

n = current_bindings
ops_integration_current = steps * (n**2)
print(f"Current grammar (n={n}):")
print(f"  Steps: {steps}")
print(f"  Operations: ~{ops_integration_current:,.0f}")
print(f"  Time estimate: ~0.5s (measured)")

n = estimated_bindings_1k
ops_integration_1k = steps * (n**2)
scaling_factor = ops_integration_1k / ops_integration_current
print(f"\n1k-rule grammar (n={n}):")
print(f"  Steps: {steps}")
print(f"  Operations: ~{ops_integration_1k:,.0f}")
print(f"  Scaling: {scaling_factor:.1f}x more operations")
print(f"  Time estimate: ~{0.5 * scaling_factor:.1f}s per get_ep() call")

print("\n" + "="*70)
print("Training Impact (200 epochs)")
print("="*70)

print("\nCurrent Grammar:")
print(f"  Newton: 200 × 0.5s = 100s = 1.7 minutes")
print(f"  Integration: 200 × 0.5s = 100s = 1.7 minutes")
print(f"  (Currently about the same)")

newton_time_1k = 0.5 * (estimated_bindings_1k / current_bindings)**3 * 200
integration_time_1k = 0.5 * (estimated_bindings_1k / current_bindings)**2 * 200

print(f"\n1k-rule Grammar:")
print(f"  Newton: {newton_time_1k:.0f}s = {newton_time_1k/60:.1f} minutes")
print(f"  Integration: {integration_time_1k:.0f}s = {integration_time_1k/60:.1f} minutes")
print(f"  Difference: {newton_time_1k - integration_time_1k:.0f}s = {(newton_time_1k - integration_time_1k)/60:.1f} minutes")

if newton_time_1k > 3600:
    print(f"  ⚠️ Newton would take >{newton_time_1k/3600:.1f} hours for training!")

print("\n" + "="*70)
print("CONCLUSION")
print("="*70)

ratio = (estimated_bindings_1k / current_bindings)
print(f"""
Network size scaling: {ratio:.1f}x larger for 1k grammar

Newton:  O(n³) → {ratio**3:.1f}x slower
Integration: O(n²) → {ratio**2:.1f}x slower

For current small grammar (~400 bindings): methods are comparable
For 1k-rule grammar (~3000 bindings): Newton becomes impractical

YOUR CONCERN IS VALID: Integration is necessary for large grammars.
""")

print("\n" + "="*70)
print("POSSIBLE SOLUTIONS")
print("="*70)
print("""
1. **Hybrid approach**:
   - Use Newton for first 20-50 epochs (to establish strong EPs)
   - Switch to Integration for remaining epochs (for speed)

2. **Grammar-size adaptive**:
   - Newton for <500 bindings
   - Integration for ≥500 bindings

3. **Longer integration duration**:
   - Use dur=20 or dur=30 instead of dur=10
   - More expensive but more accurate
   - Still O(n²) so scales better than Newton

4. **Verify integration convergence**:
   - Check that integration actually reaches equilibrium
   - May need to increase dur for complex grammars

5. **Accept the tradeoff**:
   - Integration might give slightly weaker EPs for rare structures
   - But makes 1k-grammar training feasible
   - May need more training epochs or higher learning rates to compensate

RECOMMENDATION for your use case:
- Keep Integration as default for scalability
- Test with longer duration (dur=20 or 30) for better accuracy
- OR: Use Newton for small grammars, Integration for large ones
""")
