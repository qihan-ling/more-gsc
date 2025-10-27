#!/usr/bin/env python3
"""
Create a small test grammar from the NP grammar for quick testing.
"""

def load_and_filter_grammar(input_file, min_prob=0.001, max_rules=100):
    """
    Load grammar and filter to keep only high-probability rules.

    Args:
        input_file: Path to input grammar file
        min_prob: Minimum probability threshold
        max_rules: Maximum number of rules to keep

    Returns:
        Filtered grammar string
    """
    from collections import defaultdict

    # Load rules
    rules = []
    with open(input_file, 'r') as f:
        for line in f:
            line = line.strip()
            if not line or '->' not in line:
                continue

            parts = line.split(' -> ')
            if len(parts) != 2:
                continue

            left_part = parts[0].strip().split()
            if len(left_part) < 2:
                continue

            prob = float(left_part[0])
            lhs = left_part[1]
            rhs = parts[1].strip()

            if prob >= min_prob:
                rules.append((prob, lhs, rhs))

    # Sort by probability and take top rules
    rules.sort(key=lambda x: -x[0])
    rules = rules[:max_rules]

    # Group by LHS and re-normalize
    lhs_groups = defaultdict(list)
    for prob, lhs, rhs in rules:
        lhs_groups[lhs].append((prob, rhs))

    # Normalize each LHS group
    normalized_rules = []
    for lhs, rhs_list in lhs_groups.items():
        total = sum(p for p, _ in rhs_list)
        for prob, rhs in rhs_list:
            norm_prob = prob / total if total > 0 else 0.0
            normalized_rules.append((norm_prob, f"{lhs} -> {rhs}"))

    # Sort by probability for output
    normalized_rules.sort(key=lambda x: -x[0])

    # Format as grammar string
    grammar_lines = [f"{prob:.10f} {rule}" for prob, rule in normalized_rules]

    return '\n'.join(grammar_lines), len(lhs_groups), len(normalized_rules)


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description='Create small test grammar')
    parser.add_argument('input_file', help='Input grammar file')
    parser.add_argument('-o', '--output', default='test_grammar.txt',
                       help='Output file (default: test_grammar.txt)')
    parser.add_argument('-m', '--min-prob', type=float, default=0.01,
                       help='Minimum probability threshold (default: 0.01)')
    parser.add_argument('-n', '--max-rules', type=int, default=50,
                       help='Maximum number of rules (default: 50)')

    args = parser.parse_args()

    grammar_str, num_lhs, num_rules = load_and_filter_grammar(
        args.input_file,
        min_prob=args.min_prob,
        max_rules=args.max_rules
    )

    with open(args.output, 'w') as f:
        f.write(grammar_str)

    print(f"Created test grammar:")
    print(f"  LHS categories: {num_lhs}")
    print(f"  Total rules: {num_rules}")
    print(f"  Saved to: {args.output}")
    print(f"\nPreview:")
    for line in grammar_str.split('\n')[:10]:
        print(f"  {line}")
