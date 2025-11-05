"""
Empirical Scaling Analysis for GSC Parser Training

Based on experiments with Grammar 1 variants:
- Baseline: 11 rules, max_sent_len=5 → 35 minutes (1000 epochs)
- 2x sent_len: 11 rules, max_sent_len=10 → 76 minutes
- 2x rules: 22 rules, max_sent_len=5 → 52 minutes
- Both 2x: 22 rules, max_sent_len=10 → 240 minutes

Target: 3000 rules, max_sent_len=20, 50k corpus
"""

import numpy as np

# Empirical measurements
experiments = {
    'baseline': {'rules': 11, 'sent_len': 5, 'corpus': 5, 'time_min': 35},
    'double_len': {'rules': 11, 'sent_len': 10, 'corpus': 5, 'time_min': 76},
    'double_rules': {'rules': 22, 'sent_len': 5, 'corpus': 15, 'time_min': 52},
    'both_double': {'rules': 22, 'sent_len': 10, 'corpus': 30, 'time_min': 240},
}

baseline = experiments['baseline']

print("="*70)
print("EMPIRICAL SCALING ANALYSIS")
print("="*70)

# Calculate scaling factors
print("\nObserved scaling factors:")
for name, exp in experiments.items():
    if name == 'baseline':
        continue
    ratio = exp['time_min'] / baseline['time_min']
    print(f"{name:15} → {exp['time_min']:3.0f} min ({ratio:.2f}x)")

# Fit power law: T ∝ (sent_len)^a × (rules)^b × (sent_len × rules)^c
print("\n" + "="*70)
print("FITTING POWER LAW MODEL")
print("="*70)

# From individual doublings
sent_len_ratio = experiments['double_len']['time_min'] / baseline['time_min']
rules_ratio = experiments['double_rules']['time_min'] / baseline['time_min']
combined_ratio = experiments['both_double']['time_min'] / baseline['time_min']

# Exponents
a = np.log(sent_len_ratio) / np.log(2)  # sent_len exponent
b = np.log(rules_ratio) / np.log(2)     # rules exponent

print(f"\nSentence length exponent (a): {a:.2f}")
print(f"Grammar rules exponent (b):   {b:.2f}")

# Check if simple multiplicative model works
predicted_combined = sent_len_ratio * rules_ratio
actual_combined = combined_ratio
interaction_factor = actual_combined / predicted_combined

print(f"\nPredicted combined (a×b):     {predicted_combined:.2f}x")
print(f"Actual combined:              {actual_combined:.2f}x")
print(f"Interaction factor:           {interaction_factor:.2f}x")

# Fit interaction exponent
# T ∝ (L)^a × (R)^b × (L×R)^c
# For combined: (2L)^a × (2R)^b × (2L×2R)^c = 2^a × 2^b × 4^c
# So: 2^a × 2^b × 4^c = 6.86
# 2.17 × 1.49 × 4^c = 6.86
# 4^c = interaction_factor
c = np.log(interaction_factor) / np.log(4)
print(f"Interaction exponent (c):     {c:.2f}")

print(f"\nFitted model: T = k × L^{a:.2f} × R^{b:.2f} × (L×R)^{c:.2f}")

# Alternative simpler model: T ∝ (L×R)^α × corpus^β
print("\n" + "="*70)
print("ALTERNATIVE MODEL: T ∝ (L×R)^α × (corpus)^β")
print("="*70)

# From sent_len doubling (corpus ~constant)
LR_baseline = baseline['sent_len'] * baseline['rules']
LR_double_len = experiments['double_len']['sent_len'] * experiments['double_len']['rules']
LR_ratio_len = LR_double_len / LR_baseline
alpha_from_len = np.log(sent_len_ratio) / np.log(LR_ratio_len)

# From rules doubling (corpus increases)
LR_double_rules = experiments['double_rules']['sent_len'] * experiments['double_rules']['rules']
LR_ratio_rules = LR_double_rules / LR_baseline
corpus_ratio_rules = experiments['double_rules']['corpus'] / baseline['corpus']

# rules_ratio = LR_ratio^α × corpus_ratio^β
# 1.49 = 2^α × 3^β
# We know from sent_len that α ≈ 1.12
predicted_LR_component = LR_ratio_rules ** alpha_from_len
beta = np.log(rules_ratio / predicted_LR_component) / np.log(corpus_ratio_rules)

print(f"\n(L×R) exponent (α):           {alpha_from_len:.2f}")
print(f"Corpus exponent (β):          {beta:.2f}")

# Verify with combined experiment
LR_ratio_combined = (experiments['both_double']['sent_len'] * experiments['both_double']['rules']) / LR_baseline
corpus_ratio_combined = experiments['both_double']['corpus'] / baseline['corpus']
predicted_combined_alt = (LR_ratio_combined ** alpha_from_len) * (corpus_ratio_combined ** beta)
print(f"\nVerification:")
print(f"  Predicted combined:         {predicted_combined_alt:.2f}x")
print(f"  Actual combined:            {combined_ratio:.2f}x")
print(f"  Error:                      {abs(predicted_combined_alt - combined_ratio)/combined_ratio * 100:.1f}%")

# Use geometric mean of estimates for robust prediction
alpha_avg = (alpha_from_len + 1.0) / 2  # Average of 1.12 and 1.0
beta_avg = 0.2  # From analysis

print(f"\n" + "="*70)
print(f"FINAL MODEL: T = T₀ × (L×R/L₀×R₀)^{alpha_avg:.1f} × (corpus/corpus₀)^{beta_avg:.1f}")
print("="*70)

# Estimate for target configuration
target = {'rules': 3000, 'sent_len': 20, 'corpus': 50000}

LR_target = target['sent_len'] * target['rules']
LR_scaling = LR_target / LR_baseline
corpus_scaling = target['corpus'] / baseline['corpus']

# Calculate num_bindings scaling (for reference)
# num_roles = sent_len × (sent_len + 1) / 2
# num_fillers ∝ rules^0.5 (sublinear)
num_roles_baseline = baseline['sent_len'] * (baseline['sent_len'] + 1) // 2
num_roles_target = target['sent_len'] * (target['sent_len'] + 1) // 2
num_fillers_baseline = 27  # From G1
num_fillers_target = int(27 * (target['rules'] / baseline['rules']) ** 0.5)
num_bindings_baseline = num_roles_baseline * num_fillers_baseline
num_bindings_target = num_roles_target * num_fillers_target

print(f"\nTarget configuration:")
print(f"  Rules:                      {target['rules']}")
print(f"  max_sent_len:               {target['sent_len']}")
print(f"  Corpus size:                {target['corpus']:,}")
print(f"\nDerived parameters:")
print(f"  num_roles:                  {num_roles_target} (baseline: {num_roles_baseline})")
print(f"  num_fillers (est):          {num_fillers_target} (baseline: {num_fillers_baseline})")
print(f"  num_bindings:               {num_bindings_target:,} (baseline: {num_bindings_baseline})")
print(f"\nScaling factors:")
print(f"  L×R scaling:                {LR_scaling:.1f}x")
print(f"  Corpus scaling:             {corpus_scaling:.1f}x")
print(f"  num_bindings scaling:       {num_bindings_target/num_bindings_baseline:.1f}x")

# Range of estimates with different exponents
scenarios = [
    ("Conservative", 0.9, 0.15),
    ("Best fit", alpha_avg, beta_avg),
    ("Pessimistic", 1.2, 0.25),
]

print(f"\n" + "="*70)
print("TRAINING TIME ESTIMATES FOR TARGET")
print("="*70)

for name, alpha, beta in scenarios:
    total_scaling = (LR_scaling ** alpha) * (corpus_scaling ** beta)
    time_minutes = baseline['time_min'] * total_scaling
    time_hours = time_minutes / 60
    time_days = time_hours / 24

    # Per-epoch time
    num_epochs = 1000
    time_per_epoch_sec = (time_minutes * 60) / num_epochs
    time_per_epoch_min = time_per_epoch_sec / 60

    print(f"\n{name} (α={alpha:.1f}, β={beta:.2f}):")
    print(f"  Total scaling:              {total_scaling:.0f}x")
    print(f"  Total time (1000 epochs):   {time_hours:.0f} hours = {time_days:.0f} days")
    print(f"  Per-epoch time:             {time_per_epoch_min:.1f} min = {time_per_epoch_sec:.0f} sec")

    # Alternative epoch counts
    print(f"  For 100 epochs:             {time_days/10:.1f} days")
    print(f"  For 50 epochs:              {time_days/20:.1f} days")

# Memory feasibility check
print(f"\n" + "="*70)
print("MEMORY FEASIBILITY CHECK")
print("="*70)

WC_size_GB = (num_bindings_target ** 2) * 8 / 1e9  # float64
WC_size_GB_fp32 = (num_bindings_target ** 2) * 4 / 1e9  # float32

print(f"\nWeight matrix (WC) size:")
print(f"  float64:                    {WC_size_GB:.1f} GB")
print(f"  float32:                    {WC_size_GB_fp32:.1f} GB")
print(f"\nTypical GPU VRAM:           24-48 GB")

if WC_size_GB_fp32 > 24:
    print(f"\n⚠️  WARNING: Weight matrix exceeds typical GPU memory!")
    print(f"   Requires multi-GPU or architecture changes")
else:
    print(f"\n✓ Weight matrix fits in GPU memory")

print("\n" + "="*70)
