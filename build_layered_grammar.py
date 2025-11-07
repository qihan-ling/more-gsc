#!/usr/bin/env python3
"""
Build grammar bottom-up from terminal vocabulary.

Starting from terminal symbols, iteratively builds grammar layers
by finding rules where both children are in the current layer,
until reaching the root 'S'.
"""

import re
from collections import defaultdict


def strip_subscript(symbol):
    """
    Remove subscript from Berkeley Parser symbol.

    Examples:
        S_1 -> S
        SBAR_3 -> SBAR
        VP_5 -> VP
    """
    return re.sub(r'_\d+$', '', symbol)


def get_sm5_grammar(filepath):
    """
    Reads a Berkeley Parser grammar file.

    Args:
        filepath (str): The path to the grammar file.

    Returns:
        list: A list of tuples, each tuple is (probability, mother node, daughter1, daughter2).
    """
    rules = []
    with open(filepath, 'r') as f:
        for line in f:
            line = line.strip()
            if not line or '->' not in line:
                continue

            parts = line.split(' -> ')
            if len(parts) != 2:
                continue

            lhs = parts[0].strip()
            rhs_parts = parts[1].split()

            # Skip if not enough parts (need at least: daughter1 daughter2 probability)
            if len(rhs_parts) < 3:
                continue

            prob = rhs_parts[-1]  # Last element is probability
            daughters = rhs_parts[:-1]  # Everything except probability

            # We only want binary rules for now
            if len(daughters) != 2:
                continue

            daughter1, daughter2 = daughters
            rules.append((float(prob), lhs, daughter1, daughter2))

    return rules


def build_grammar_layers(rules, bottom_nodes, root='S'):
    """
    Build grammar layers bottom-up from terminal vocabulary.

    Args:
        rules: List of (prob, mother, daughter1, daughter2) tuples
        bottom_nodes: Set of terminal/bottom symbols
        root: Root symbol (default: 'S')

    Returns:
        filtered_rules: List of rules that connect bottom nodes to root
        layer_info: Dictionary with layer statistics
    """
    print(f"Building grammar layers from {len(bottom_nodes)} bottom nodes...")
    print(f"Total rules in original grammar: {len(rules)}")

    # Track which symbols are reachable at each layer
    current_layer = set(bottom_nodes)
    all_reachable = set(bottom_nodes)

    # Rules to keep
    kept_rules = []

    # Track layers for statistics
    layer_info = {'layers': []}
    layer_num = 0

    while True:
        layer_num += 1
        print(f"\n{'='*70}")
        print(f"Layer {layer_num}:")
        print(f"  Current symbols: {len(current_layer)}")

        # Find rules where BOTH daughters are in current reachable set
        new_mothers = set()
        layer_rules = []

        for prob, mother, d1, d2 in rules:
            # Strip subscripts to check membership
            d1_base = strip_subscript(d1)
            d2_base = strip_subscript(d2)
            mother_base = strip_subscript(mother)

            # Check if both daughters are reachable
            if d1_base in all_reachable and d2_base in all_reachable:
                # Check if mother is new (not yet in all_reachable)
                if mother_base not in all_reachable:
                    new_mothers.add(mother_base)
                    layer_rules.append((prob, mother, d1, d2))
                    kept_rules.append((prob, mother, d1, d2))
                elif mother_base in current_layer:
                    # Mother already exists but might be from this layer
                    layer_rules.append((prob, mother, d1, d2))
                    kept_rules.append((prob, mother, d1, d2))

        print(f"  New mother nodes: {len(new_mothers)}")
        print(f"  Rules added: {len(layer_rules)}")

        if new_mothers:
            sample = sorted(new_mothers)[:10]
            print(f"  Sample mothers: {sample}")

        # Track layer info
        layer_info['layers'].append({
            'layer': layer_num,
            'symbols': len(new_mothers),
            'rules': len(layer_rules),
            'new_mothers': sorted(new_mothers)
        })

        # Check if we've reached the root
        root_base = strip_subscript(root)
        if root_base in new_mothers:
            print(f"\n  ✓ Reached root '{root}'!")
            break

        # Check if no progress
        if not new_mothers:
            print(f"\n  ✗ No new mothers found. Cannot reach root '{root}'.")
            break

        # Update for next iteration
        all_reachable.update(new_mothers)
        current_layer = new_mothers

        # Safety check to avoid infinite loops
        if layer_num > 20:
            print(f"\n  ✗ Exceeded maximum layers (20). Stopping.")
            break

    print(f"\n{'='*70}")
    print(f"Layer building complete!")
    print(f"  Total layers: {layer_num}")
    print(f"  Total reachable symbols: {len(all_reachable)}")
    print(f"  Total kept rules: {len(kept_rules)}")
    print(f"{'='*70}")

    return kept_rules, layer_info


def collapse_and_normalize(rules, min_prob=1e-10, cumulative_threshold=0.95):
    """
    Collapse subscripts and normalize probabilities.

    Args:
        rules: List of (prob, mother, daughter1, daughter2) tuples
        min_prob: Minimum probability threshold
        cumulative_threshold: Keep top rules until this cumulative probability

    Returns:
        Normalized grammar string in GSC format
    """
    print(f"\nCollapsing subscripts and normalizing...")

    # Group rules by base form
    rule_groups = defaultdict(list)

    for prob, mother, d1, d2 in rules:
        mother_base = strip_subscript(mother)
        d1_base = strip_subscript(d1)
        d2_base = strip_subscript(d2)

        base_rule = f"{mother_base} -> {d1_base} {d2_base}"
        rule_groups[base_rule].append(prob)

    print(f"  Unique base rules: {len(rule_groups)}")

    # Sum probabilities for each base rule
    rule_probs = {}
    for base_rule, probs in rule_groups.items():
        rule_probs[base_rule] = sum(probs)

    # Group by LHS for normalization
    lhs_groups = defaultdict(dict)
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

    print(f"  Normalized rules: {len(normalized_rules)}")
    print(f"  LHS categories: {len(lhs_groups)}")

    # Apply cumulative probability threshold
    print(f"\nApplying cumulative threshold ({cumulative_threshold:.1%})...")

    lhs_groups_filtered = defaultdict(list)
    for rule, prob in normalized_rules.items():
        lhs, rhs = rule.split(' -> ', 1)
        lhs_groups_filtered[lhs].append((rhs, prob))

    # For each LHS, keep only top rules until cumulative threshold
    selected_rules = {}
    total_removed = 0

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
                total_removed += 1

        # Store kept rules
        for rhs, prob in kept_rules:
            rule = f"{lhs} -> {rhs}"
            selected_rules[rule] = prob

    print(f"  Rules removed by threshold: {total_removed}")
    print(f"  Remaining rules: {len(selected_rules)}")

    # Re-normalize after threshold filtering
    print(f"\nRe-normalizing after filtering...")

    lhs_groups_final = defaultdict(dict)
    for rule, prob in selected_rules.items():
        lhs, rhs = rule.split(' -> ', 1)
        lhs_groups_final[lhs][rhs] = prob

    final_rules = {}
    for lhs, rhs_dict in lhs_groups_final.items():
        total = sum(rhs_dict.values())
        for rhs, prob in rhs_dict.items():
            renormalized_prob = prob / total if total > 0 else 0.0
            rule = f"{lhs} -> {rhs}"
            final_rules[rule] = renormalized_prob

    print(f"  Final rule count: {len(final_rules)}")

    # Apply minimum probability filter
    filtered_count = 0
    gsc_rules = []
    for rule, prob in sorted(final_rules.items(), key=lambda x: -x[1]):
        if prob >= min_prob:
            gsc_rules.append(f"{prob:.10f} {rule}")
        else:
            filtered_count += 1

    if filtered_count > 0:
        print(f"  Rules filtered by min_prob: {filtered_count}")

    return '\n'.join(gsc_rules)


def main():
    import argparse

    parser = argparse.ArgumentParser(
        description='Build grammar bottom-up from terminal vocabulary'
    )
    parser.add_argument(
        'grammar_file',
        help='Path to Berkeley Parser SM5 grammar file'
    )
    parser.add_argument(
        '-o', '--output',
        default='layered_grammar.txt',
        help='Output file (default: layered_grammar.txt)'
    )
    parser.add_argument(
        '-c', '--cumulative',
        type=float,
        default=0.95,
        help='Cumulative probability threshold (default: 0.95)'
    )
    parser.add_argument(
        '-m', '--min-prob',
        type=float,
        default=1e-10,
        help='Minimum probability threshold (default: 1e-10)'
    )

    args = parser.parse_args()

    # Define bottom nodes (terminal symbols)
    bottom_nodes = {
        'WP', 'NNP', 'SYM', 'PRP', 'MD', 'VBG', 'POS', 'VB', 'RP', 'NNPS',
        'IN', 'CD', 'UH', 'LS', '.', 'WRB', 'TO', 'VBZ', 'VBD', 'VBN',
        'JJ', 'PRP$', 'JJS', 'RB', ',', 'NN', 'FW', 'NNS', 'PDT', 'WDT',
        'VBP', 'RBR', 'CC', 'EX', 'DT', 'JJR'
    }

    print(f"Loading grammar from: {args.grammar_file}")
    rules = get_sm5_grammar(args.grammar_file)
    print(f"Loaded {len(rules)} binary rules")

    print(f"\nBottom nodes: {len(bottom_nodes)} terminal symbols")
    print(f"Sample: {sorted(bottom_nodes)[:10]}")

    # Build grammar layers
    kept_rules, layer_info = build_grammar_layers(rules, bottom_nodes, root='S')

    # Collapse and normalize
    grammar_str = collapse_and_normalize(
        kept_rules,
        min_prob=args.min_prob,
        cumulative_threshold=args.cumulative
    )

    # Save to file
    with open(args.output, 'w') as f:
        f.write(grammar_str)

    print(f"\n{'='*70}")
    print(f"Grammar saved to: {args.output}")
    print(f"{'='*70}")

    # Preview
    lines = grammar_str.split('\n')
    print(f"\nPreview (first 20 rules):")
    for i, line in enumerate(lines[:20], 1):
        print(f"  {i:3d}. {line}")

    if len(lines) > 20:
        print(f"  ... ({len(lines) - 20} more rules)")

    # Show layer statistics
    print(f"\n{'='*70}")
    print("Layer Statistics:")
    print(f"{'='*70}")
    for layer in layer_info['layers']:
        print(f"  Layer {layer['layer']}: {layer['symbols']} symbols, {layer['rules']} rules")

    print(f"\n{'='*70}")
    print("Done!")
    print(f"{'='*70}")


if __name__ == '__main__':
    main()
