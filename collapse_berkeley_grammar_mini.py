#!/usr/bin/env python3
"""
Collapse Berkeley Parser grammar to create a minimal GSC-compatible grammar.

This script extends collapse_berkeley_grammar.py with additional filtering:
1. Collapses subcategories and normalizes probabilities by LHS
2. Removes identity unary rules (e.g., "IN -> IN" with prob 1.0)
3. Keeps only high-probability rules using cumulative probability threshold
4. Re-normalizes probabilities after filtering
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


def collapse_grammar_mini(grammar_file, output_file=None,
                         min_probability=1e-10,
                         cumulative_threshold=0.95,
                         remove_unary_rules=True):
    """
    Collapse Berkeley Parser grammar and create minimal version.

    Args:
        grammar_file: Path to Berkeley Parser .grammar file
        output_file: Path to output GSC grammar (if None, returns string)
        min_probability: Minimum probability threshold (applied before cumulative)
        cumulative_threshold: Keep top rules until this cumulative probability (e.g., 0.95)
        remove_unary_rules: Remove all unary rules (GSC requires CNF-like format)

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

    print(f"\nStep 1: Collapse subcategories")
    print(f"  Total Berkeley rules: {total_rules:,}")
    print(f"  Skipped (parse errors): {skipped_rules:,}")
    print(f"  Collapsed base rules: {len(rule_probs):,}")
    print(f"  Reduction ratio: {len(rule_probs) / total_rules:.1%}")

    # Normalize probabilities by LHS (so rules with same LHS sum to 1.0)
    print(f"\nStep 2: Initial normalization by LHS")

    # Group rules by LHS
    lhs_groups = defaultdict(dict)  # lhs -> {rhs: prob}
    for rule, prob in rule_probs.items():
        lhs, rhs = rule.split(' -> ', 1)
        lhs_groups[lhs][rhs] = prob

    # Normalize each LHS group
    normalized_rules = {}
    for lhs, rhs_dict in lhs_groups.items():
        total = sum(rhs_dict.values())
        for rhs, prob in rhs_dict.items():
            normalized_prob = prob / total if total > 0 else 0.0
            rule = f"{lhs} -> {rhs}"
            normalized_rules[rule] = normalized_prob

    print(f"  Normalized {len(lhs_groups):,} LHS categories")
    print(f"  Total normalized rules: {len(normalized_rules):,}")

    # Step 3: Remove unary rules (GSC requires CNF-like format)
    if remove_unary_rules:
        print(f"\nStep 3: Remove unary rules (GSC requires binary/terminal rules only)")

        filtered_rules = {}
        unary_removed = 0

        for rule, prob in normalized_rules.items():
            lhs, rhs = rule.split(' -> ', 1)
            rhs_parts = rhs.split()

            # Check if it's a unary rule (exactly one symbol on RHS)
            if len(rhs_parts) == 1:
                unary_removed += 1
            else:
                filtered_rules[rule] = prob

        print(f"  Unary rules removed: {unary_removed:,}")
        print(f"  Remaining rules: {len(filtered_rules):,}")
        normalized_rules = filtered_rules

    # Step 4: Apply cumulative probability threshold per LHS
    print(f"\nStep 4: Apply cumulative probability threshold ({cumulative_threshold:.1%})")

    # Re-group by LHS after identity removal
    lhs_groups_filtered = defaultdict(list)  # lhs -> [(rhs, prob), ...]
    for rule, prob in normalized_rules.items():
        lhs, rhs = rule.split(' -> ', 1)
        lhs_groups_filtered[lhs].append((rhs, prob))

    # For each LHS, keep only top rules until cumulative threshold
    selected_rules = {}
    total_removed_by_threshold = 0

    for lhs, rhs_prob_list in lhs_groups_filtered.items():
        # Sort by probability (descending)
        rhs_prob_list.sort(key=lambda x: -x[1])

        # Calculate cumulative probability
        cumulative = 0.0
        kept_rules = []

        for rhs, prob in rhs_prob_list:
            if cumulative < cumulative_threshold:
                kept_rules.append((rhs, prob))
                cumulative += prob
            else:
                total_removed_by_threshold += 1

        # Store kept rules
        for rhs, prob in kept_rules:
            rule = f"{lhs} -> {rhs}"
            selected_rules[rule] = prob

    print(f"  Rules removed by cumulative threshold: {total_removed_by_threshold:,}")
    print(f"  Remaining rules: {len(selected_rules):,}")

    # Step 5: Re-normalize after threshold filtering
    print(f"\nStep 5: Re-normalize probabilities after filtering")

    # Re-group by LHS
    lhs_groups_final = defaultdict(dict)
    for rule, prob in selected_rules.items():
        lhs, rhs = rule.split(' -> ', 1)
        lhs_groups_final[lhs][rhs] = prob

    # Re-normalize
    final_rules = {}
    for lhs, rhs_dict in lhs_groups_final.items():
        total = sum(rhs_dict.values())
        for rhs, prob in rhs_dict.items():
            renormalized_prob = prob / total if total > 0 else 0.0
            rule = f"{lhs} -> {rhs}"
            final_rules[rule] = renormalized_prob

    print(f"  Re-normalized {len(lhs_groups_final):,} LHS categories")
    print(f"  Final rule count: {len(final_rules):,}")

    # Verify normalization (sample check)
    print(f"\n  Verification: Sample LHS normalization check")
    sample_lhs = list(lhs_groups_final.keys())[:5]
    for lhs in sample_lhs:
        lhs_sum = sum(prob for rule, prob in final_rules.items()
                      if rule.startswith(f"{lhs} ->"))
        print(f"    {lhs}: {len(lhs_groups_final[lhs])} rules, sum = {lhs_sum:.10f}")

    # Apply minimum probability filter and format for GSC
    print(f"\nStep 6: Apply minimum probability filter ({min_probability})")

    gsc_rules = []
    filtered_by_min = 0

    for rule, prob in sorted(final_rules.items(), key=lambda x: -x[1]):
        if prob >= min_probability:
            gsc_rules.append(f"{prob:.10f} {rule}")
        else:
            filtered_by_min += 1

    print(f"  Rules filtered by min probability: {filtered_by_min:,}")
    print(f"  Final output rules: {len(gsc_rules):,}")

    # Final statistics
    print(f"\n{'='*70}")
    print(f"SUMMARY:")
    print(f"  Original Berkeley rules: {total_rules:,}")
    print(f"  Final mini grammar rules: {len(gsc_rules):,}")
    print(f"  Total reduction: {(1 - len(gsc_rules) / total_rules):.2%}")
    print(f"{'='*70}")

    # Join into final grammar string
    grammar_str = '\n'.join(gsc_rules)

    # Save to file if specified
    if output_file:
        with open(output_file, 'w') as f:
            f.write(grammar_str)
        print(f"\nMini grammar saved to: {output_file}")

    return grammar_str


def main():
    import argparse

    parser = argparse.ArgumentParser(
        description='Collapse Berkeley Parser grammar to minimal GSC format'
    )
    parser.add_argument(
        'grammar_file',
        help='Path to Berkeley Parser .grammar file'
    )
    parser.add_argument(
        '-o', '--output',
        help='Output file for GSC grammar (default: collapsed_grammar_mini.txt)',
        default='collapsed_grammar_mini.txt'
    )
    parser.add_argument(
        '-m', '--min-prob',
        type=float,
        default=1e-10,
        help='Minimum probability threshold (default: 1e-10)'
    )
    parser.add_argument(
        '-c', '--cumulative',
        type=float,
        default=0.95,
        help='Cumulative probability threshold (default: 0.95 = 95%%)'
    )
    parser.add_argument(
        '--keep-unary',
        action='store_true',
        help='Keep unary rules (not recommended for GSC)'
    )
    parser.add_argument(
        '--preview',
        type=int,
        default=20,
        help='Number of top rules to preview (default: 20)'
    )

    args = parser.parse_args()

    # Collapse grammar
    grammar_str = collapse_grammar_mini(
        args.grammar_file,
        output_file=args.output,
        min_probability=args.min_prob,
        cumulative_threshold=args.cumulative,
        remove_unary_rules=not args.keep_unary
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
