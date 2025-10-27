#!/usr/bin/env python3
"""
Extract NP-rooted grammar from a collapsed grammar file.

This script extracts all NP rules and recursively includes rules for all
categories that appear on the right-hand side (daughter nodes), creating
a complete closure of the NP-rooted grammar.
"""

from collections import defaultdict


def parse_rule(line):
    """
    Parse a grammar line.

    Format: "0.5248699624 NP -> DT NN"
    Returns: (probability, lhs, rhs_list)
    """
    parts = line.strip().split()
    if len(parts) < 4 or parts[1] != '->':
        return None, None, None

    prob = float(parts[0])
    lhs = parts[1].replace('->', '').strip()  # In case of spacing issues
    if lhs == '->':
        lhs = parts[1]
        rhs_list = parts[3:]
    else:
        lhs = parts[1]
        rhs_list = parts[3:]

    # Better parsing
    prob_str = parts[0]
    arrow_idx = -1
    for i, p in enumerate(parts):
        if p == '->' or '->' in p:
            arrow_idx = i
            break

    if arrow_idx == -1:
        return None, None, None

    prob = float(parts[0])
    lhs = parts[arrow_idx - 1] if arrow_idx > 0 else parts[1]
    rhs_list = parts[arrow_idx + 1:]

    return prob, lhs, rhs_list


def load_grammar(grammar_file):
    """
    Load grammar into a dictionary: lhs -> [(prob, rhs_list), ...]
    """
    grammar = defaultdict(list)

    with open(grammar_file, 'r') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue

            # Parse: "0.5248699624 NP -> DT NN"
            parts = line.split(' -> ')
            if len(parts) != 2:
                continue

            left_part = parts[0].strip().split()
            if len(left_part) < 2:
                continue

            prob = float(left_part[0])
            lhs = left_part[1]
            rhs_list = parts[1].strip().split()

            grammar[lhs].append((prob, rhs_list))

    return grammar


def extract_closure(grammar, root='NP'):
    """
    Extract closure of rules starting from root category.

    Returns:
        Dictionary: lhs -> [(prob, rhs_list), ...]
    """
    selected_rules = defaultdict(list)
    to_process = {root}
    processed = set()

    print(f"Extracting closure starting from root: {root}")

    iteration = 0
    while to_process:
        iteration += 1
        current = to_process.pop()

        if current in processed:
            continue

        processed.add(current)

        # Get all rules for this LHS
        if current in grammar:
            rules = grammar[current]
            selected_rules[current] = rules

            print(f"  Iteration {iteration}: Processing '{current}' ({len(rules)} rules)")

            # Add all RHS categories to process queue
            for prob, rhs_list in rules:
                for rhs_cat in rhs_list:
                    if rhs_cat not in processed:
                        to_process.add(rhs_cat)

    return selected_rules


def format_grammar(grammar_dict):
    """
    Format grammar dictionary into GSC string format.

    Sort by probability (descending) for readability.
    """
    lines = []

    for lhs in sorted(grammar_dict.keys()):
        rules = grammar_dict[lhs]
        # Sort by probability descending
        for prob, rhs_list in sorted(rules, key=lambda x: -x[0]):
            rhs_str = ' '.join(rhs_list)
            lines.append(f"{prob:.10f} {lhs} -> {rhs_str}")

    return '\n'.join(lines)


def main():
    import argparse

    parser = argparse.ArgumentParser(
        description='Extract NP-rooted grammar closure from collapsed grammar'
    )
    parser.add_argument(
        'grammar_file',
        help='Path to collapsed grammar file'
    )
    parser.add_argument(
        '-o', '--output',
        help='Output file for NP grammar (default: np_grammar.txt)',
        default='np_grammar.txt'
    )
    parser.add_argument(
        '-r', '--root',
        default='NP',
        help='Root category to extract (default: NP)'
    )

    args = parser.parse_args()

    print(f"Loading grammar from: {args.grammar_file}")
    grammar = load_grammar(args.grammar_file)
    print(f"  Loaded {len(grammar)} LHS categories")
    print(f"  Total rules: {sum(len(rules) for rules in grammar.values())}")

    # Extract NP closure
    np_grammar = extract_closure(grammar, root=args.root)

    print(f"\n{'='*70}")
    print(f"Closure extraction complete:")
    print(f"  Root category: {args.root}")
    print(f"  LHS categories in closure: {len(np_grammar)}")
    print(f"  Total rules: {sum(len(rules) for rules in np_grammar.values())}")
    print(f"{'='*70}")

    # Format and save
    grammar_str = format_grammar(np_grammar)

    with open(args.output, 'w') as f:
        f.write(grammar_str)

    print(f"\nNP grammar saved to: {args.output}")

    # Show categories in closure
    print(f"\nCategories in closure (sorted):")
    for i, cat in enumerate(sorted(np_grammar.keys()), 1):
        rule_count = len(np_grammar[cat])
        print(f"  {i:2d}. {cat:15s} ({rule_count:3d} rules)")

    # Preview top rules
    print(f"\n{'='*70}")
    print(f"Preview of top 20 rules:")
    print(f"{'='*70}")

    lines = grammar_str.split('\n')
    # Sort all lines by probability
    rule_lines = []
    for line in lines:
        prob = float(line.split()[0])
        rule_lines.append((prob, line))
    rule_lines.sort(key=lambda x: -x[0])

    for i, (prob, line) in enumerate(rule_lines[:20], 1):
        print(f"{i:3d}. {line}")

    if len(rule_lines) > 20:
        print(f"... ({len(rule_lines) - 20} more rules)")

    print(f"\n{'='*70}")
    print("Done!")
    print(f"{'='*70}")


if __name__ == '__main__':
    main()
