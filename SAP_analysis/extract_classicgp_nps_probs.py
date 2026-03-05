"""
Extract collapsed probability values from berkeley_parser_sm5.grammar
for the mini grammar rules in ClassicGP_NPS.txt.

The Berkeley grammar uses latent-annotation splits (e.g., S_0, S_1, ..., S_12).
To recover base-grammar probabilities, we marginalize over all substates assuming
uniform mixing weights, then renormalize within each LHS so the mini grammar
rules sum to 1.

    P_collapsed(Y Z | X) = (1/n_X) * sum_i sum_{j,k} P(Y_j Z_k | X_i)

Since we renormalize at the end, the 1/n_X factor cancels and we can simply
sum all matching subscripted-rule probabilities per base rule.

Usage:
    python SAP_analysis/extract_classicgp_nps_probs.py
"""

import re
import os
from collections import defaultdict

MINI_GRAMMAR_PATH = os.path.join('SAP_stimuli', 'ClassicGP_NPS.txt')
BIG_GRAMMAR_PATH = os.path.join(
    'trained_berkeley_parser_sm5', 'berkeley_parser_sm5.grammar')
OUTPUT_PATH = os.path.join('SAP_stimuli', 'ClassicGP_NPS_probs.txt')

SUBSCRIPT_RE = re.compile(r'_\d+$')


def strip_subscript(symbol):
    return SUBSCRIPT_RE.sub('', symbol)


def load_mini_grammar(path):
    rules = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line and '->' in line:
                rules.append(line)
    return rules


def collapse_probabilities(mini_rules, big_grammar_path):
    mini_rule_set = set(mini_rules)
    raw_sums = defaultdict(float)
    match_count = defaultdict(int)

    with open(big_grammar_path) as f:
        for line in f:
            line = line.strip()
            if not line or '->' not in line:
                continue

            parts = line.split()
            if len(parts) != 5:
                continue

            lhs_sub, arrow, rhs1_sub, rhs2_sub, prob_str = parts
            base_lhs = strip_subscript(lhs_sub)
            base_rhs1 = strip_subscript(rhs1_sub)
            base_rhs2 = strip_subscript(rhs2_sub)
            base_rule = f"{base_lhs} -> {base_rhs1} {base_rhs2}"

            if base_rule in mini_rule_set:
                raw_sums[base_rule] += float(prob_str)
                match_count[base_rule] += 1

    lhs_totals = defaultdict(float)
    for rule, total in raw_sums.items():
        lhs = rule.split(' -> ')[0]
        lhs_totals[lhs] += total

    collapsed = {}
    for rule in mini_rules:
        lhs = rule.split(' -> ')[0]
        if raw_sums[rule] > 0:
            collapsed[rule] = raw_sums[rule] / lhs_totals[lhs]
        else:
            collapsed[rule] = 0.0

    return collapsed, raw_sums, match_count


def main():
    print(f"Loading mini grammar from {MINI_GRAMMAR_PATH}")
    mini_rules = load_mini_grammar(MINI_GRAMMAR_PATH)
    print(f"  Found {len(mini_rules)} rules:")
    for r in mini_rules:
        print(f"    {r}")

    print(f"\nScanning {BIG_GRAMMAR_PATH} ...")
    collapsed, raw_sums, match_count = collapse_probabilities(
        mini_rules, BIG_GRAMMAR_PATH)

    print("\nCollapsed & renormalized probabilities:")
    print("-" * 50)
    lines_out = []
    for rule in mini_rules:
        prob = collapsed[rule]
        n_matches = match_count[rule]
        raw = raw_sums[rule]
        print(f"  {prob:.6f} {rule}  "
              f"(raw_sum={raw:.6e}, {n_matches} subscripted rules)")
        lines_out.append(f"{prob:.6f} {rule}")

    # Verify normalization per LHS
    print("\nNormalization check:")
    lhs_groups = defaultdict(float)
    for rule in mini_rules:
        lhs = rule.split(' -> ')[0]
        lhs_groups[lhs] += collapsed[rule]
    for lhs, total in sorted(lhs_groups.items()):
        print(f"  {lhs}: sum = {total:.6f}")

    with open(OUTPUT_PATH, 'w') as f:
        f.write('\n'.join(lines_out) + '\n')
    print(f"\nSaved to {OUTPUT_PATH}")


if __name__ == '__main__':
    main()
