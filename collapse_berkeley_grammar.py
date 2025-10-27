#!/usr/bin/env python3
"""
Collapse Berkeley Parser subcategories to create a GSC-compatible grammar.

This script implements Proposal 1: Full Collapse with Probability Summation.
It removes all latent subcategorization (e.g., JJ_1, JJ_2, ..., JJ_30 -> JJ)
and combines probabilities by summing them.
"""

import re
from collections import defaultdict
from pathlib import Path


def strip_subscript(category):
    """
    Remove subscript from a category.

    Examples:
        S_1 -> S
        VP_5 -> VP
        NP_10 -> NP
    """
    return re.sub(r'_\d+$', '', category)


def parse_grammar_line(line):
    """
    Parse a Berkeley Parser grammar line.

    Format: "S_1 -> SBAR_3 VP_5 3.8038446028221417E-6"
    Returns: (lhs, rhs_list, probability)
    """
    # Remove line number prefix if present (e.g., "1→")
    line = re.sub(r'^\s*\d+→', '', line).strip()

    if not line or '->' not in line:
        return None, None, None

    # Split into rule and probability
    parts = line.rsplit(maxsplit=1)
    if len(parts) != 2:
        return None, None, None

    rule_str, prob_str = parts

    try:
        probability = float(prob_str)
    except ValueError:
        return None, None, None

    # Parse the rule: "S_1 -> SBAR_3 VP_5"
    if '->' not in rule_str:
        return None, None, None

    lhs, rhs = rule_str.split('->', 1)
    lhs = lhs.strip()
    rhs_list = rhs.strip().split()

    return lhs, rhs_list, probability


def collapse_grammar(grammar_file, output_file=None, min_probability=1e-10):
    """
    Collapse Berkeley Parser grammar by removing subcategories.

    Args:
        grammar_file: Path to Berkeley Parser .grammar file
        output_file: Path to output GSC grammar (if None, returns string)
        min_probability: Minimum probability threshold for including rules

    Returns:
        Grammar string in GSC format
    """
    print(f"Reading grammar from: {grammar_file}")

    # Dictionary to accumulate probabilities: base_rule -> total_probability
    rule_probs = defaultdict(float)

    # Track statistics
    total_rules = 0
    skipped_rules = 0

    with open(grammar_file, 'r') as f:
        for line_num, line in enumerate(f, 1):
            lhs, rhs_list, probability = parse_grammar_line(line)

            if lhs is None:
                skipped_rules += 1
                continue

            total_rules += 1

            # Strip subscripts from all categories
            base_lhs = strip_subscript(lhs)
            base_rhs = [strip_subscript(cat) for cat in rhs_list]

            # Create base rule string
            base_rule = f"{base_lhs} -> {' '.join(base_rhs)}"

            # Accumulate probability
            rule_probs[base_rule] += probability

            # Progress indicator
            if line_num % 100000 == 0:
                print(f"  Processed {line_num:,} lines...")

    print(f"\nProcessing complete:")
    print(f"  Total Berkeley rules: {total_rules:,}")
    print(f"  Skipped (parse errors): {skipped_rules:,}")
    print(f"  Collapsed base rules (before normalization): {len(rule_probs):,}")
    print(f"  Reduction ratio: {len(rule_probs) / total_rules:.1%}")

    # Normalize probabilities by LHS (so rules with same LHS sum to 1.0)
    print(f"\nNormalizing probabilities by LHS...")

    # Group rules by LHS
    lhs_groups = defaultdict(dict)  # lhs -> {rhs: prob}
    for rule, prob in rule_probs.items():
        lhs, rhs = rule.split(' -> ', 1)
        lhs_groups[lhs][rhs] = prob

    # Normalize each LHS group
    normalized_rules = {}
    lhs_totals = {}
    for lhs, rhs_dict in lhs_groups.items():
        total = sum(rhs_dict.values())
        lhs_totals[lhs] = total
        for rhs, prob in rhs_dict.items():
            normalized_prob = prob / total if total > 0 else 0.0
            rule = f"{lhs} -> {rhs}"
            normalized_rules[rule] = normalized_prob

    print(f"  Normalized {len(lhs_groups):,} LHS categories")
    print(f"  Total normalized rules: {len(normalized_rules):,}")

    # Show normalization stats for a few categories
    print(f"\n  Sample LHS normalization:")
    for lhs in sorted(lhs_groups.keys())[:5]:
        print(f"    {lhs}: {len(lhs_groups[lhs])} rules, original sum = {lhs_totals[lhs]:.4f}")

    # Filter by minimum probability and format for GSC
    print(f"\nFiltering rules with probability >= {min_probability}...")

    gsc_rules = []
    filtered_count = 0

    for rule, prob in sorted(normalized_rules.items(), key=lambda x: -x[1]):
        if prob >= min_probability:
            gsc_rules.append(f"{prob:.10f} {rule}")
        else:
            filtered_count += 1

    print(f"  Rules after filtering: {len(gsc_rules):,}")
    print(f"  Rules filtered out: {filtered_count:,}")

    # Verify normalization (sample check)
    print(f"\n  Verification: Checking normalization for sample LHS categories...")
    sample_lhs = list(lhs_groups.keys())[:3]
    for lhs in sample_lhs:
        lhs_sum = sum(prob for rule, prob in normalized_rules.items()
                      if rule.startswith(f"{lhs} ->"))
        print(f"    {lhs}: sum of probabilities = {lhs_sum:.10f}")

    # Join into final grammar string
    grammar_str = '\n'.join(gsc_rules)

    # Save to file if specified
    if output_file:
        with open(output_file, 'w') as f:
            f.write(grammar_str)
        print(f"\nGrammar saved to: {output_file}")

    return grammar_str


def main():
    import argparse

    parser = argparse.ArgumentParser(
        description='Collapse Berkeley Parser grammar to GSC format'
    )
    parser.add_argument(
        'grammar_file',
        help='Path to Berkeley Parser .grammar file'
    )
    parser.add_argument(
        '-o', '--output',
        help='Output file for GSC grammar (default: collapsed_grammar.txt)',
        default='collapsed_grammar.txt'
    )
    parser.add_argument(
        '-m', '--min-prob',
        type=float,
        default=1e-10,
        help='Minimum probability threshold (default: 1e-10)'
    )
    parser.add_argument(
        '--preview',
        type=int,
        default=20,
        help='Number of top rules to preview (default: 20)'
    )

    args = parser.parse_args()

    # Collapse grammar
    grammar_str = collapse_grammar(
        args.grammar_file,
        output_file=args.output,
        min_probability=args.min_prob
    )

    # Preview top rules
    print(f"\n{'='*70}")
    print(f"Preview of top {args.preview} rules:")
    print(f"{'='*70}")

    lines = grammar_str.split('\n')
    for i, line in enumerate(lines[:args.preview], 1):
        print(f"{i:3d}. {line}")

    if len(lines) > args.preview:
        print(f"... ({len(lines) - args.preview:,} more rules)")

    print(f"\n{'='*70}")
    print("Done!")
    print(f"{'='*70}")


if __name__ == '__main__':
    main()
