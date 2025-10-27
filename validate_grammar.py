#!/usr/bin/env python3
"""
Validate a grammar file for GSC compatibility.
Checks for unary rules and normalization issues.
"""

from collections import defaultdict


def validate_grammar(grammar_text):
    """
    Validate grammar for GSC compatibility.

    Returns:
        (is_valid, issues, stats)
    """
    issues = []
    stats = {
        'total_rules': 0,
        'unary_rules': 0,
        'lhs_categories': set(),
        'normalization_errors': []
    }

    # Group rules by LHS
    lhs_groups = defaultdict(list)

    for line_num, line in enumerate(grammar_text.strip().split('\n'), 1):
        line = line.strip()
        if not line:
            continue

        # Parse line
        if '->' not in line:
            issues.append(f"Line {line_num}: Missing '->' : {line}")
            continue

        parts = line.split(' -> ')
        if len(parts) != 2:
            issues.append(f"Line {line_num}: Invalid format: {line}")
            continue

        left_part = parts[0].strip().split()
        if len(left_part) < 2:
            issues.append(f"Line {line_num}: Missing probability or LHS: {line}")
            continue

        try:
            prob = float(left_part[0])
        except ValueError:
            issues.append(f"Line {line_num}: Invalid probability '{left_part[0]}': {line}")
            continue

        lhs = left_part[1]
        rhs = parts[1].strip()
        rhs_parts = rhs.split()

        stats['total_rules'] += 1
        stats['lhs_categories'].add(lhs)

        # Check for unary rules
        if len(rhs_parts) == 1:
            stats['unary_rules'] += 1
            issues.append(f"Line {line_num}: UNARY RULE (GSC incompatible!): {lhs} -> {rhs}")

        # Check for None placeholders
        if 'None' in rhs_parts:
            issues.append(f"Line {line_num}: Contains 'None' placeholder: {line}")

        # Store for normalization check
        lhs_groups[lhs].append((prob, line_num, line))

    # Check normalization
    for lhs, rules in lhs_groups.items():
        total = sum(p for p, _, _ in rules)
        if abs(total - 1.0) > 1e-6:
            stats['normalization_errors'].append((lhs, total, len(rules)))
            issues.append(f"Normalization error: {lhs} rules sum to {total:.10f} (expected 1.0)")

    is_valid = len(issues) == 0

    return is_valid, issues, stats


def main():
    import argparse

    parser = argparse.ArgumentParser(description='Validate grammar for GSC')
    parser.add_argument('grammar_file', help='Grammar file to validate')

    args = parser.parse_args()

    # Load grammar
    with open(args.grammar_file, 'r') as f:
        grammar_text = f.read()

    # Validate
    is_valid, issues, stats = validate_grammar(grammar_text)

    # Print results
    print("="*70)
    print("GRAMMAR VALIDATION RESULTS")
    print("="*70)
    print(f"File: {args.grammar_file}")
    print(f"Total rules: {stats['total_rules']}")
    print(f"LHS categories: {len(stats['lhs_categories'])}")
    print(f"Unary rules: {stats['unary_rules']}")
    print(f"Normalization errors: {len(stats['normalization_errors'])}")
    print()

    if is_valid:
        print("✓ VALID - Grammar is GSC-compatible!")
    else:
        print("✗ INVALID - Found issues:")
        print()
        for i, issue in enumerate(issues[:20], 1):
            print(f"  {i}. {issue}")
        if len(issues) > 20:
            print(f"  ... and {len(issues) - 20} more issues")

    print("="*70)

    # Show normalization errors in detail
    if stats['normalization_errors']:
        print("\nNormalization errors (LHS categories that don't sum to 1.0):")
        for lhs, total, count in stats['normalization_errors'][:10]:
            print(f"  {lhs}: {count} rules, sum = {total:.10f}")

    return 0 if is_valid else 1


if __name__ == '__main__':
    exit(main())
