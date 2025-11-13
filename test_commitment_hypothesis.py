#!/usr/bin/env python3
"""
Test if equalizing commitment across sentence lengths fixes S4
"""
import numpy as np

print("="*70)
print("Commitment Schedule Bug Analysis")
print("="*70)

# Current implementation
max_sent_len = 5
sentences = [
    ("S0", 2, "N Vi"),
    ("S1", 4, "N Vi P N"),
    ("S2", 3, "N BE Vpp"),
    ("S3", 5, "N BE Vpp P N"),  # Actually "N BE Vpp P N" is 5 words
    ("S4", 5, "N Vpp P N Vi"),
]

print("\n" + "="*70)
print("CURRENT IMPLEMENTATION (may be buggy):")
print("="*70)
print("\nAll sentences use: dq = ones(max_sent_len) * (t / max_sent_len)")

for t in [1, 5, 10]:
    print(f"\n--- Commitment level t={t} ---")
    dq = np.ones(max_sent_len) * (float(t) / max_sent_len)
    qpolicy = np.insert(dq.cumsum(), 0, 0.)

    print(f"dq = {dq}")
    print(f"qpolicy = {qpolicy}")
    print("\nTotal commitment per sentence:")

    for name, length, words in sentences:
        total = qpolicy[length]
        per_word = total / length if length > 0 else 0
        marker = " ← GETS MORE!" if length == 5 else ""
        print(f"  {name} ({length} words, {words:20s}): total={total:.2f}, per_word={per_word:.3f}{marker}")

print("\n" + "="*70)
print("PROPOSED FIX (equal total commitment):")
print("="*70)
print("\nEach sentence gets: dq = ones(actual_len) * (t / actual_len)")

for t in [1, 5, 10]:
    print(f"\n--- Commitment level t={t} ---")
    print("Total commitment per sentence:")

    for name, length, words in sentences:
        dq_fixed = np.ones(length) * (float(t) / length)
        qpolicy_fixed = np.insert(dq_fixed.cumsum(), 0, 0.)
        total = qpolicy_fixed[length]
        per_word = total / length if length > 0 else 0
        print(f"  {name} ({length} words, {words:20s}): dq={dq_fixed}, total={total:.2f}")

print("\n" + "="*70)
print("ANALYSIS:")
print("="*70)

print("""
Current implementation:
  - S4 (5 words) gets 25% MORE total commitment than S1 (4 words)
  - At t=5: S1 gets 4.0, S4 gets 5.0
  - At t=10: S1 gets 8.0, S4 gets 10.0

Why this could cause S4 to fail:
  1. S4 has the rarest structure (S->NP Vi, 5% probability)
  2. With excessive commitment, the network over-commits too early
  3. Before the rare NP structure can emerge, it's forced into the
     more common VP structure (60% probability)
  4. Over-commitment locks it into the wrong attractor!

Proposed fix:
  - All sentences get SAME total commitment t
  - Each word gets t/actual_len commitment increment
  - S4 gets LESS per-word commitment (t/5 vs current t/5)
  - This gives the rare structure more "breathing room" to emerge

Key question:
  What did your ORIGINAL working replication (before speedup) use?
  - If it used per-sentence dq calculation → this is the bug!
  - If it used fixed max_sent_len dq → something else is wrong
""")

print("\n" + "="*70)
print("ACTION ITEM:")
print("="*70)
print("""
Please check your original replication script (before speedup changes).
Look for how it calculated dq for the parsing test.

If it had something like:
  dq = np.ones(len(sent)) * (t / len(sent))

Then we found the bug! The fix is in cho_grammar1_new_copy.py line 158.
""")
