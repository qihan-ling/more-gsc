"""
Generic script to extract collapsed probability values from
berkeley_parser_sm5.grammar for any mini grammar rules file.

Handles both binary rules (X -> Y Z) and unary rules (X -> Y).

The Berkeley grammar uses latent-annotation splits (e.g., S_0, S_1, ..., S_12).
To recover base-grammar probabilities, we marginalize over all substates assuming
uniform mixing weights, then renormalize within each LHS so the mini grammar
rules sum to 1.

With --merge-at (enabled by default), intermediate binarization symbols like
@S and @SBAR are merged with their base counterparts (S, SBAR). This lets
rules like 'S -> SBAR S' match '@S -> SBAR S' in the Berkeley grammar and
'SBAR -> SBAR ,' match 'SBAR -> @SBAR ,'.

Usage:
    python SAP_analysis/extract_probs.py <mini_grammar.txt>
    python SAP_analysis/extract_probs.py --no-merge-at <mini_grammar.txt>

Output is written to the same directory as the input, with '_probs' appended
to the filename (e.g., Agreement.txt -> Agreement_probs.txt).
"""

import re
import os
import sys
from collections import defaultdict

BIG_GRAMMAR_PATH = os.path.join(
    'trained_berkeley_parser_sm5', 'berkeley_parser_sm5.grammar')

SUBSCRIPT_RE = re.compile(r'_\d+$')
AT_PREFIX_RE = re.compile(r'^@')


def strip_subscript(symbol):
    return SUBSCRIPT_RE.sub('', symbol)


def strip_at_prefix(symbol):
    """Merge @X intermediate binarization symbols with base symbol X."""
    return AT_PREFIX_RE.sub('', symbol)


def load_mini_grammar(path):
    rules = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line or '->' not in line:
                continue
            lhs, rhs = line.split('->')
            lhs = lhs.strip()
            rhs = rhs.strip()
            if rhs:
                rules.append(f"{lhs} -> {rhs}")
            else:
                print(f"  WARNING: skipping empty RHS rule: '{line}'")
    return rules


def rule_arity(rule):
    """Return number of RHS symbols (1 for unary, 2 for binary)."""
    rhs = rule.split(' -> ')[1]
    return len(rhs.split())


def collapse_probabilities(mini_rules, big_grammar_path, merge_at=True):
    mini_rule_set = set(mini_rules)
    raw_sums = defaultdict(float)
    match_count = defaultdict(int)

    with open(big_grammar_path) as f:
        for line in f:
            line = line.strip()
            if not line or '->' not in line:
                continue

            parts = line.split()

            if len(parts) == 4:
                lhs_sub, arrow, rhs1_sub, prob_str = parts
                base_lhs = strip_subscript(lhs_sub)
                base_rhs1 = strip_subscript(rhs1_sub)
                if merge_at:
                    base_lhs = strip_at_prefix(base_lhs)
                    base_rhs1 = strip_at_prefix(base_rhs1)
                base_rule = f"{base_lhs} -> {base_rhs1}"
            elif len(parts) == 5:
                lhs_sub, arrow, rhs1_sub, rhs2_sub, prob_str = parts
                base_lhs = strip_subscript(lhs_sub)
                base_rhs1 = strip_subscript(rhs1_sub)
                base_rhs2 = strip_subscript(rhs2_sub)
                if merge_at:
                    base_lhs = strip_at_prefix(base_lhs)
                    base_rhs1 = strip_at_prefix(base_rhs1)
                    base_rhs2 = strip_at_prefix(base_rhs2)
                base_rule = f"{base_lhs} -> {base_rhs1} {base_rhs2}"
            else:
                continue

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
    args = [a for a in sys.argv[1:] if not a.startswith('--')]
    flags = [a for a in sys.argv[1:] if a.startswith('--')]
    merge_at = '--no-merge-at' not in flags

    if len(args) < 1:
        print("Usage: python SAP_analysis/extract_probs.py "
              "[--no-merge-at] <mini_grammar.txt>")
        sys.exit(1)

    mini_path = args[0]
    base, ext = os.path.splitext(mini_path)
    output_path = f"{base}_probs{ext}"

    print(f"Loading mini grammar from {mini_path}")
    print(f"  merge @-prefixed symbols: {merge_at}")
    mini_rules = load_mini_grammar(mini_path)
    print(f"  Found {len(mini_rules)} rules:")
    for r in mini_rules:
        arity = rule_arity(r)
        print(f"    {r}  ({'unary' if arity == 1 else 'binary'})")

    print(f"\nScanning {BIG_GRAMMAR_PATH} ...")
    collapsed, raw_sums, match_count = collapse_probabilities(
        mini_rules, BIG_GRAMMAR_PATH, merge_at=merge_at)

    print("\nCollapsed & renormalized probabilities:")
    print("-" * 60)
    lines_out = []
    has_zero = False
    for rule in mini_rules:
        prob = collapsed[rule]
        n_matches = match_count[rule]
        raw = raw_sums[rule]
        tag = "  *** NO MATCH ***" if n_matches == 0 else ""
        print(f"  {prob:.6f} {rule}  "
              f"(raw_sum={raw:.6e}, {n_matches} subscripted rules){tag}")
        lines_out.append(f"{prob:.6f} {rule}")
        if n_matches == 0:
            has_zero = True

    print("\nNormalization check:")
    lhs_groups = defaultdict(float)
    for rule in mini_rules:
        lhs = rule.split(' -> ')[0]
        lhs_groups[lhs] += collapsed[rule]
    for lhs, total in sorted(lhs_groups.items()):
        status = " (OK)" if abs(total - 1.0) < 1e-6 else " *** NOT 1.0 ***"
        print(f"  {lhs}: sum = {total:.6f}{status}")

    if has_zero:
        print("\n*** WARNING: Some rules had 0 matches in the Berkeley grammar.")
        print("    These rules will have probability 0.0 in the output.")
        print("    You may need to assign probabilities manually or")
        print("    adjust the mini grammar.\n")

    with open(output_path, 'w') as f:
        f.write('\n'.join(lines_out) + '\n')
    print(f"Saved to {output_path}")


if __name__ == '__main__':
    main()
