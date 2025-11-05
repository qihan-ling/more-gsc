#!/usr/bin/env python3
"""
Filter and collapse grammar to target vocabulary.

Keeps only rules relevant to specified vocabulary, and collapses
out-of-vocabulary daughter rules into a single "OTHER" rule per LHS.
"""

from collections import defaultdict
import re


def normalize_symbol(symbol):
    """
    Normalize a symbol by removing subscripts and @ prefixes.

    Examples:
        @PP -> PP
        NP_1 -> NP
        VP[3] -> VP
    """
    # Remove @ prefix
    symbol = symbol.lstrip('@')
    # Remove subscripts like _1, _2
    symbol = re.sub(r'_\d+$', '', symbol)
    # Remove brackets like [1], [2]
    symbol = re.sub(r'\[\d+\]$', '', symbol)
    return symbol


def is_in_vocab(symbol, vocabulary):
    """
    Check if a symbol (or its normalized form) is in vocabulary.
    """
    if symbol in vocabulary:
        return True
    # Check normalized form
    normalized = normalize_symbol(symbol)
    return normalized in vocabulary


def parse_rule(line):
    """Parse a grammar rule line."""
    if not line.strip() or '->' not in line:
        return None, None, None, None

    parts = line.split(' -> ')
    if len(parts) != 2:
        return None, None, None, None

    left_part = parts[0].strip().split()
    if len(left_part) < 2:
        return None, None, None, None

    prob = float(left_part[0])
    lhs = left_part[1]
    rhs = parts[1].strip().split()

    if len(rhs) != 2:
        return None, None, None, None

    return prob, lhs, rhs[0], rhs[1]


def filter_and_collapse_grammar(input_file, vocabulary, output_file=None):
    """
    Filter grammar to vocabulary and collapse out-of-vocab rules.

    Args:
        input_file: Path to input grammar file
        vocabulary: List of vocabulary symbols
        output_file: Path to output file (optional)

    Returns:
        Filtered grammar string
    """
    print(f"Loading grammar from: {input_file}")
    print(f"Target vocabulary: {len(vocabulary)} symbols")

    # Group rules by LHS
    lhs_groups = defaultdict(list)

    with open(input_file, 'r') as f:
        for line_num, line in enumerate(f, 1):
            prob, lhs, rhs1, rhs2 = parse_rule(line)

            if prob is None:
                continue

            lhs_groups[lhs].append({
                'prob': prob,
                'rhs1': rhs1,
                'rhs2': rhs2,
                'line': line.strip()
            })

    print(f"\nOriginal grammar: {len(lhs_groups)} LHS categories")

    # Filter and collapse
    filtered_rules = []
    stats = {
        'lhs_kept': 0,
        'lhs_removed': 0,
        'rules_kept': 0,
        'rules_collapsed': 0,
        'rules_removed': 0
    }

    for lhs, rules in lhs_groups.items():
        # Check if LHS is in vocabulary
        if not is_in_vocab(lhs, vocabulary):
            stats['lhs_removed'] += 1
            stats['rules_removed'] += len(rules)
            continue

        stats['lhs_kept'] += 1

        # Separate rules into in-vocab and out-of-vocab
        in_vocab_rules = []
        out_of_vocab_rules = []

        for rule in rules:
            rhs1_in_vocab = is_in_vocab(rule['rhs1'], vocabulary)
            rhs2_in_vocab = is_in_vocab(rule['rhs2'], vocabulary)

            if rhs1_in_vocab and rhs2_in_vocab:
                in_vocab_rules.append(rule)
            else:
                out_of_vocab_rules.append(rule)

        # Add in-vocab rules
        for rule in in_vocab_rules:
            filtered_rules.append(f"{rule['prob']:.10f} {lhs} -> {rule['rhs1']} {rule['rhs2']}")
            stats['rules_kept'] += 1

        # Collapse out-of-vocab rules if any exist
        if out_of_vocab_rules:
            total_in_vocab_prob = sum(r['prob'] for r in in_vocab_rules)
            other_prob = 1.0 - total_in_vocab_prob

            if other_prob > 1e-10:  # Only add if probability is significant
                filtered_rules.append(f"{other_prob:.10f} {lhs} -> OTHER OTHER")
                stats['rules_collapsed'] += len(out_of_vocab_rules)
            else:
                stats['rules_removed'] += len(out_of_vocab_rules)

    # Sort rules by probability (descending) for readability
    rule_tuples = []
    for rule in filtered_rules:
        prob = float(rule.split()[0])
        rule_tuples.append((prob, rule))
    rule_tuples.sort(key=lambda x: -x[0])

    filtered_rules = [r[1] for r in rule_tuples]

    # Print statistics
    print(f"\n{'='*70}")
    print("FILTERING RESULTS:")
    print(f"{'='*70}")
    print(f"LHS categories:")
    print(f"  Kept:    {stats['lhs_kept']}")
    print(f"  Removed: {stats['lhs_removed']}")
    print(f"\nRules:")
    print(f"  Kept (in-vocab):        {stats['rules_kept']}")
    print(f"  Collapsed (out-of-vocab): {stats['rules_collapsed']}")
    print(f"  Removed (LHS not in vocab): {stats['rules_removed']}")
    print(f"  Total output rules:     {len(filtered_rules)}")
    print(f"{'='*70}")

    # Format as grammar string
    grammar_str = '\n'.join(filtered_rules)

    # Save if output file specified
    if output_file:
        with open(output_file, 'w') as f:
            f.write(grammar_str)
        print(f"\nFiltered grammar saved to: {output_file}")

    return grammar_str


def main():
    import argparse

    parser = argparse.ArgumentParser(
        description='Filter grammar to target vocabulary'
    )
    parser.add_argument(
        'input_file',
        help='Input grammar file'
    )
    parser.add_argument(
        '-o', '--output',
        default='filtered_grammar.txt',
        help='Output file (default: filtered_grammar.txt)'
    )
    parser.add_argument(
        '--vocab-file',
        help='File containing vocabulary (one symbol per line)'
    )

    args = parser.parse_args()

    # Define vocabulary
    if args.vocab_file:
        with open(args.vocab_file, 'r') as f:
            vocabulary = [line.strip() for line in f if line.strip()]
    else:
        # Default vocabulary from user's list
        vocabulary = [
            'S', 'PP', ',', 'NP', 'VP', 'IN', 'NNP', 'VBD', 'DT', 'NN',
            'SBAR', 'WHNP', 'WP', 'VBZ', 'ADJP', 'CC', 'JJ', 'RB', 'NNS',
            '.', 'VBN', 'VBG', 'ADVP', 'PRP', 'TO', 'WHADVP', 'WRB',
            'PRP$', 'VB', 'EX', 'MD', 'QP', 'CD', 'POS', 'FW'
        ]

    print(f"Using vocabulary: {vocabulary}")

    # Filter grammar
    grammar_str = filter_and_collapse_grammar(
        args.input_file,
        vocabulary,
        output_file=args.output
    )

    # Preview
    print(f"\n{'='*70}")
    print("Preview of filtered grammar (first 20 rules):")
    print(f"{'='*70}")
    lines = grammar_str.split('\n')
    for i, line in enumerate(lines[:20], 1):
        print(f"{i:3d}. {line}")

    if len(lines) > 20:
        print(f"... ({len(lines) - 20} more rules)")

    print(f"\n{'='*70}")
    print("Done!")
    print(f"{'='*70}")


if __name__ == '__main__':
    main()
