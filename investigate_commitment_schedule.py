#!/usr/bin/env python3
"""
Investigate commitment schedule for different sentence lengths
"""
import only_gscnet as gsc
import numpy as np

print("Loading model...")
net = gsc.load_model('g1_ds_speedup_model_copy.pkl')

print("\n" + "="*70)
print("Commitment Schedule Analysis")
print("="*70)

max_sent_len = net.hg.opts['max_sent_len']
print(f"\nmax_sent_len = {max_sent_len}")

# Check each sentence's actual length
f_empty_type = net.hg.g.get_types(net.hg.opts['f_empty'])

print("\n" + "="*70)
print("Sentence lengths (without empty fillers):")
print("="*70)

sentence_info = []
for si, sent in enumerate(net.corpus['sentence']):
    word_seq = ' '.join([bname.split('/')[0] for bname in sent])
    sent0 = [bname for bname in sent if bname.split('/')[0] not in f_empty_type]
    actual_len = len(sent0)
    sentence_info.append((si, word_seq, actual_len))
    print(f"S{si}: {word_seq:20s} - {actual_len} words")

print("\n" + "="*70)
print("Commitment schedule for t=1:")
print("="*70)

t = 1
dq = np.ones(max_sent_len) * (float(t) / max_sent_len)
qpolicy = dq.cumsum()
qpolicy = np.insert(qpolicy, 0, 0.)

print(f"\nFor t={t}:")
print(f"  dq shape: {dq.shape}, values: {dq}")
print(f"  qpolicy shape: {qpolicy.shape}, values: {qpolicy}")

print("\nTotal commitment per sentence (qpolicy[actual_len] - qpolicy[0]):")
for si, word_seq, actual_len in sentence_info:
    total_commitment = qpolicy[actual_len] - qpolicy[0]
    per_word = total_commitment / actual_len if actual_len > 0 else 0
    print(f"  S{si} ({actual_len} words): total={total_commitment:.2f}, per_word={per_word:.2f}")

print("\n" + "="*70)
print("Commitment schedule for t=5:")
print("="*70)

t = 5
dq = np.ones(max_sent_len) * (float(t) / max_sent_len)
qpolicy = dq.cumsum()
qpolicy = np.insert(qpolicy, 0, 0.)

print(f"\nFor t={t}:")
print(f"  dq shape: {dq.shape}, values: {dq}")
print(f"  qpolicy shape: {qpolicy.shape}, values: {qpolicy}")

print("\nTotal commitment per sentence:")
for si, word_seq, actual_len in sentence_info:
    total_commitment = qpolicy[actual_len] - qpolicy[0]
    per_word = total_commitment / actual_len if actual_len > 0 else 0
    marker = " ← S4 fails here!" if si == 4 else ""
    print(f"  S{si} ({actual_len} words): total={total_commitment:.2f}, per_word={per_word:.2f}{marker}")

print("\n" + "="*70)
print("Problem Analysis:")
print("="*70)

print("""
The current implementation creates:
  dq = ones(max_sent_len) * (t / max_sent_len)

This means:
1. dq always has length 5 (max_sent_len)
2. qpolicy = cumsum(dq) with 0 inserted → length 6
3. Shorter sentences use fewer elements of qpolicy

Result: Different sentence lengths get DIFFERENT total commitments!
- S0 (2 words): gets 2/5 = 40% of intended commitment
- S1 (4 words): gets 4/5 = 80% of intended commitment
- S4 (5 words): gets 5/5 = 100% of intended commitment ✓

This is CORRECT if the intent is:
  "Commitment level t means: allocate t total commitment uniformly
   across max_sent_len=5 word positions, and each sentence uses
   only the positions it needs."

But if the intent is:
  "Every sentence should reach total commitment = t regardless of length"

Then this is WRONG, and we need:
  dq = ones(actual_sent_len) * (t / actual_sent_len)

Let me check what the original paper/code does...
""")

print("\n" + "="*70)
print("Checking if this is intentional or a bug:")
print("="*70)

print("""
Key question: In the original paper, does "commitment level t" mean:
A) Total commitment t allocated uniformly across max_sent_len positions
   → Different length sentences get different total commitment
   → Current implementation ✓

B) Total commitment t allocated uniformly across ACTUAL sentence length
   → All sentences get the same total commitment = t
   → Would need per-sentence dq calculation

If (B) is correct, then S4 has been getting MORE commitment than it should,
which could cause it to over-commit and fail!

Actually, wait... if S4 gets MORE commitment, it should HELP, not hurt.
Unless... over-commitment at high t causes the network to force wrong structure?
""")

print("\n" + "="*70)
print("Testing hypothesis: Does S4 fail because it gets too much commitment?")
print("="*70)

print("\nAt t=5:")
print("  S1 (4 words): total commitment = 4.0")
print("  S4 (5 words): total commitment = 5.0")
print("  Difference: S4 gets 25% MORE commitment than S1")

print("\nAt t=12:")
print("  S1 (4 words): total commitment = 9.6")
print("  S4 (5 words): total commitment = 12.0")
print("  Difference: S4 gets 25% MORE commitment than S1")

print("""
If S4's rare structure (S->NP Vi, 5% prob) is being over-committed,
the network might be forced into a wrong attractor because it commits
too strongly too early, before the rare structure can emerge.

The fix would be to ensure all sentences get the SAME total commitment:
  dq = ones(actual_len) * (t / actual_len)

This way at commitment level t=5:
  S1 (4 words): [1.25, 1.25, 1.25, 1.25] → total = 5.0
  S4 (5 words): [1.0, 1.0, 1.0, 1.0, 1.0] → total = 5.0
""")
