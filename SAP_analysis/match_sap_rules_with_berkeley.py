# Code written by Claude Opus 4.5
import re
from collections import Counter, defaultdict
from nltk import Tree

# --- Helper Functions ---


def get_sm5_grammar(filepath):
    """
    Load Berkeley split-merge grammar from file.

    Args:
        filepath (str): The path to the grammar file.

    Returns:
        list: A list of tuples (probability, lhs, rhs_list)
              where rhs_list is a list of RHS symbols.
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
            rhs_and_prob = parts[1].split()

            # Last token is the probability
            prob = float(rhs_and_prob[-1])
            # Everything else is RHS symbols
            rhs = rhs_and_prob[:-1]

            rules.append((prob, lhs, rhs))
    return rules


def strip_split_merge_suffix(symbol):
    """
    Remove split-merge suffix from a symbol.
    E.g., 'S_1' -> 'S', 'NP_12' -> 'NP', 'VP' -> 'VP'
    Also handles @-binarization markers: '@S_1' -> '@S'
    """
    match = re.match(r'^(.+?)_(\d+)$', symbol)
    if match:
        return match.group(1)
    return symbol


def berkeley_rule_to_base_pattern(lhs, rhs_list):
    """
    Convert a Berkeley rule to its base pattern string.

    E.g., ('S_1', ['NP_2', 'VP_5']) -> 'S -> NP VP'
    """
    base_lhs = strip_split_merge_suffix(lhs)
    base_rhs = [strip_split_merge_suffix(sym) for sym in rhs_list]
    return f"{base_lhs} -> {' '.join(base_rhs)}"


def extract_rules_from_tree(tree):
    """
    Recursively extract CFG rules from an NLTK Tree (binarized).
    Returns two lists: syntactic_rules and lexical_rules.
    """
    syntactic_rules = []
    lexical_rules = []

    if isinstance(tree, Tree):
        lhs = tree.label()
        children = list(tree)

        # Check if this is a pre-terminal (parent of a word)
        if len(children) == 1 and isinstance(children[0], str):
            # Lexical rule: POS -> word
            lexical_rules.append(f"{lhs} -> {children[0]}")
        else:
            # Syntactic rule: collect child labels
            child_labels = []
            for child in children:
                if isinstance(child, Tree):
                    child_labels.append(child.label())
                else:
                    child_labels.append(child)

            rhs = " ".join(child_labels)
            syntactic_rules.append(f"{lhs} -> {rhs}")

            # Recurse into children
            for child in children:
                if isinstance(child, Tree):
                    sub_syn, sub_lex = extract_rules_from_tree(child)
                    syntactic_rules.extend(sub_syn)
                    lexical_rules.extend(sub_lex)

    return syntactic_rules, lexical_rules


def load_sap_parses(filepath):
    """
    Load binarized parse trees from Berkeley Parser output.
    Returns list of tree strings.
    """
    trees = []
    with open(filepath, 'r') as f:
        for line in f:
            line = line.strip()
            if line:
                trees.append(line)
    return trees


# --- Main Script ---

def main():
    # File paths - adjust these as needed
    berkeley_grammar_file = '../trained_berkeley_parser_sm5/berkeley_parser_sm5.grammar'
    sap_parses_file = 'sap_parses.txt'

    # 1. Load and extract rules from binarized SAP parses
    print("Loading binarized SAP parses...")
    sap_trees = load_sap_parses(sap_parses_file)
    print(f"  Loaded {len(sap_trees)} parse trees")

    print("\nExtracting rules from SAP parses...")
    sap_syntactic_rules = Counter()
    sap_lexical_rules = Counter()
    parse_errors = 0

    for i, tree_str in enumerate(sap_trees):
        try:
            tree = Tree.fromstring(tree_str)
            syn_rules, lex_rules = extract_rules_from_tree(tree)
            sap_syntactic_rules.update(syn_rules)
            sap_lexical_rules.update(lex_rules)
        except Exception as e:
            parse_errors += 1
            if parse_errors <= 5:
                print(f"  Error parsing tree {i}: {e}")

    if parse_errors > 5:
        print(f"  ... and {parse_errors - 5} more errors")

    print(f"  Extracted {len(sap_syntactic_rules)} unique syntactic rules")
    print(f"  Extracted {len(sap_lexical_rules)} unique lexical rules")

    # Check for non-binary rules (should be none now)
    non_binary = [(r, c) for r, c in sap_syntactic_rules.items()
                  if len(r.split(' -> ')[1].split()) > 2]
    if non_binary:
        print(f"\n  WARNING: {len(non_binary)} non-binary rules found:")
        for rule, count in non_binary[:10]:
            print(f"    {rule}")
    else:
        print("  All syntactic rules are binary or unary (good!)")

    # Save extracted rules to files
    with open('sap_binarized_syntactic_rules.txt', 'w') as f:
        for rule, count in sap_syntactic_rules.most_common():
            f.write(f"{count}\t{rule}\n")

    with open('sap_binarized_lexical_rules.txt', 'w') as f:
        for rule, count in sap_lexical_rules.most_common():
            f.write(f"{count}\t{rule}\n")

    print("\n  Saved rules to 'sap_binarized_syntactic_rules.txt' and 'sap_binarized_lexical_rules.txt'")

    # 2. Load Berkeley grammar
    print("\n" + "=" * 70)
    print("Loading Berkeley grammar...")
    berkeley_rules = get_sm5_grammar(berkeley_grammar_file)
    print(f"  Loaded {len(berkeley_rules)} Berkeley rules")

    # 3. Match Berkeley rules against SAP rules
    print("\nMatching Berkeley rules to SAP base patterns...")

    sap_rule_set = set(sap_syntactic_rules.keys())

    # Dictionary: base_pattern -> total summed probability
    matched_rule_probs = defaultdict(float)
    # Dictionary: base_pattern -> count of Berkeley rules that matched
    matched_rule_counts = defaultdict(int)
    # Track all Berkeley rules by base LHS for probability mass calculation
    all_rules_by_base_lhs = defaultdict(float)

    matched_count = 0
    for prob, lhs, rhs in berkeley_rules:
        base_lhs = strip_split_merge_suffix(lhs)
        base_pattern = berkeley_rule_to_base_pattern(lhs, rhs)

        # Track total probability mass per base LHS (for all rules)
        all_rules_by_base_lhs[base_lhs] += prob

        # Check if this base pattern matches a SAP rule
        if base_pattern in sap_rule_set:
            matched_rule_probs[base_pattern] += prob
            matched_rule_counts[base_pattern] += 1
            matched_count += 1

    print(
        f"  Matched {matched_count} Berkeley rules to {len(matched_rule_probs)} unique SAP patterns")

    # Check for SAP rules not found in Berkeley grammar
    unmatched_sap = sap_rule_set - set(matched_rule_probs.keys())
    if unmatched_sap:
        print(
            f"\n  WARNING: {len(unmatched_sap)} SAP rules have NO match in Berkeley grammar:")
        for rule in sorted(unmatched_sap)[:15]:
            print(f"    {rule}")
        if len(unmatched_sap) > 15:
            print(f"    ... and {len(unmatched_sap) - 15} more")

    # 4. Compute probability mass per LHS from matched rules only
    matched_prob_by_lhs = defaultdict(float)
    for pattern, prob in matched_rule_probs.items():
        base_lhs = pattern.split(' -> ')[0]
        matched_prob_by_lhs[base_lhs] += prob

    # 5. Report LHS categories with < 0.9 total probability
    print("\n" + "=" * 70)
    print("PROBABILITY COVERAGE ANALYSIS")
    print("=" * 70)

    low_coverage_lhs = []
    for base_lhs in sorted(matched_prob_by_lhs.keys()):
        matched_prob = matched_prob_by_lhs[base_lhs]
        total_prob = all_rules_by_base_lhs[base_lhs]
        coverage = matched_prob / total_prob if total_prob > 0 else 0

        if coverage < 0.9:
            low_coverage_lhs.append(
                (base_lhs, matched_prob, total_prob, coverage))

    print(
        f"\nNumber of LHS categories with summed probability < 0.9: {len(low_coverage_lhs)}")
    print(f"Total LHS categories in matched rules: {len(matched_prob_by_lhs)}")

    if low_coverage_lhs:
        print("\nDetails of low-coverage categories (sorted by coverage):")
        print(f"{'LHS':<15} {'Matched Prob':>15} {'Total Prob':>15} {'Coverage':>10}")
        print("-" * 55)
        for lhs, matched, total, coverage in sorted(low_coverage_lhs, key=lambda x: x[3]):
            print(f"{lhs:<15} {matched:>15.6f} {total:>15.6f} {coverage:>10.2%}")

    # 6. Show high-coverage categories
    print("\n" + "-" * 70)
    print("High-coverage categories (>= 0.9):")
    high_coverage = []
    for base_lhs in sorted(matched_prob_by_lhs.keys()):
        matched_prob = matched_prob_by_lhs[base_lhs]
        total_prob = all_rules_by_base_lhs[base_lhs]
        coverage = matched_prob / total_prob if total_prob > 0 else 0
        if coverage >= 0.9:
            high_coverage.append(
                (base_lhs, matched_prob, total_prob, coverage))

    print(f"Count: {len(high_coverage)}")
    for lhs, matched, total, coverage in sorted(high_coverage, key=lambda x: -x[3])[:10]:
        print(f"  {lhs:<15} {coverage:>10.2%}")
    if len(high_coverage) > 10:
        print(f"  ... and {len(high_coverage) - 10} more")

    # 7. Summary of matched rules
    print("\n" + "=" * 70)
    print("TOP 20 MATCHED RULES (sorted by probability)")
    print("=" * 70)
    for pattern, prob in sorted(matched_rule_probs.items(), key=lambda x: -x[1])[:20]:
        count = matched_rule_counts[pattern]
        print(f"  {prob:.6f}  ({count:3d} variants)  {pattern}")
    if len(matched_rule_probs) > 20:
        print(f"  ... and {len(matched_rule_probs) - 20} more rules")

    # 8. Summary statistics
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print(f"SAP parses loaded:                {len(sap_trees)}")
    print(f"SAP syntactic rules (binarized):  {len(sap_syntactic_rules)}")
    print(f"SAP lexical rules:                {len(sap_lexical_rules)}")
    print(f"Berkeley rules matched:           {matched_count}")
    print(f"Unique base patterns matched:     {len(matched_rule_probs)}")
    print(f"SAP rules with no Berkeley match: {len(unmatched_sap)}")
    print(f"LHS categories with coverage < 0.9: {len(low_coverage_lhs)}")
    print(f"LHS categories with coverage >= 0.9: {len(high_coverage)}")

    # 9. Expand grammar: add ALL Berkeley rules that share LHS with SAP rules
    print("\n" + "=" * 70)
    print("EXPANDED GRAMMAR ANALYSIS")
    print("=" * 70)

    # Get all LHS categories used in SAP rules
    sap_lhs_categories = set()
    for rule in sap_rule_set:
        lhs = rule.split(' -> ')[0]
        sap_lhs_categories.add(lhs)

    print(f"\nSAP uses {len(sap_lhs_categories)} unique LHS categories:")
    print(f"  {sorted(sap_lhs_categories)}")

    # Collect ALL Berkeley rules (with split-merge variants) that have matching base LHS
    expanded_rules = []  # (prob, lhs, rhs, base_pattern)
    expanded_base_patterns = set()

    for prob, lhs, rhs in berkeley_rules:
        base_lhs = strip_split_merge_suffix(lhs)

        if base_lhs in sap_lhs_categories:
            base_pattern = berkeley_rule_to_base_pattern(lhs, rhs)
            expanded_rules.append((prob, lhs, rhs, base_pattern))
            expanded_base_patterns.add(base_pattern)

    print(f"\nExpanded grammar (all rules with matching LHS):")
    print(f"  Total Berkeley rules (with split-merge): {len(expanded_rules)}")
    print(f"  Unique base patterns: {len(expanded_base_patterns)}")

    # Count rules per LHS category
    rules_per_lhs = defaultdict(int)
    base_patterns_per_lhs = defaultdict(set)
    for prob, lhs, rhs, base_pattern in expanded_rules:
        base_lhs = strip_split_merge_suffix(lhs)
        rules_per_lhs[base_lhs] += 1
        base_patterns_per_lhs[base_lhs].add(base_pattern)

    print(f"\nRules per LHS category:")
    print(f"{'LHS':<15} {'Split-Merge Rules':>20} {'Unique Base Patterns':>25}")
    print("-" * 60)
    for lhs in sorted(rules_per_lhs.keys()):
        print(
            f"{lhs:<15} {rules_per_lhs[lhs]:>20} {len(base_patterns_per_lhs[lhs]):>25}")

    # Compare: SAP-only vs expanded
    print(f"\n" + "-" * 70)
    print("COMPARISON: SAP-matched vs Expanded")
    print("-" * 70)
    print(f"{'Metric':<40} {'SAP-matched':>15} {'Expanded':>15}")
    print("-" * 70)
    print(f"{'Berkeley rules (with split-merge)':<40} {matched_count:>15} {len(expanded_rules):>15}")
    print(f"{'Unique base patterns':<40} {len(matched_rule_probs):>15} {len(expanded_base_patterns):>15}")
    print(f"{'Expansion factor (rules)':<40} {'-':>15} {len(expanded_rules)/matched_count:.2f}x")
    print(f"{'Expansion factor (patterns)':<40} {'-':>15} {len(expanded_base_patterns)/len(matched_rule_probs):.2f}x")

    # Show new base patterns added (not in SAP)
    new_patterns = expanded_base_patterns - set(matched_rule_probs.keys())
    print(f"\nNew base patterns added (not in SAP): {len(new_patterns)}")
    if new_patterns:
        # Group by LHS and show counts
        new_by_lhs = defaultdict(list)
        for p in new_patterns:
            lhs = p.split(' -> ')[0]
            new_by_lhs[lhs].append(p)

        print(f"\nNew patterns by LHS:")
        for lhs in sorted(new_by_lhs.keys()):
            patterns = new_by_lhs[lhs]
            print(f"  {lhs}: {len(patterns)} new patterns")
            # Show first 3 examples
            for p in sorted(patterns)[:3]:
                print(f"      {p}")
            if len(patterns) > 3:
                print(f"      ... and {len(patterns) - 3} more")

    # 10. Collapse split-merge variants and analyze thresholds
    print("\n" + "=" * 70)
    print("COLLAPSED GRAMMAR (split-merge variants merged)")
    print("=" * 70)

    # Collapse: sum probabilities for each base pattern
    # base_pattern -> summed probability
    collapsed_grammar = defaultdict(float)
    for prob, lhs, rhs, base_pattern in expanded_rules:
        collapsed_grammar[base_pattern] += prob

    print(f"\nTotal collapsed rules (base patterns): {len(collapsed_grammar)}")

    # Analyze probability distribution
    probs = sorted(collapsed_grammar.values(), reverse=True)
    print(f"\nProbability distribution of collapsed rules:")
    print(f"  Max: {probs[0]:.6f}")
    print(f"  Min: {probs[-1]:.10f}")
    print(f"  Median: {probs[len(probs)//2]:.6f}")
    print(f"  Mean: {sum(probs)/len(probs):.6f}")

    # Threshold analysis
    thresholds = [0.0001, 0.001, 0.005, 0.01, 0.05, 0.1]
    print(f"\n{'Threshold':<15} {'Rules Remaining':>20} {'% of Rules':>15} {'Prob Mass Kept':>20}")
    print("-" * 70)

    total_prob_mass = sum(probs)
    for thresh in thresholds:
        remaining = [(p, r)
                     for r, p in collapsed_grammar.items() if p >= thresh]
        remaining_prob = sum(p for p, r in remaining)
        pct_rules = len(remaining) / len(collapsed_grammar) * 100
        pct_prob = remaining_prob / total_prob_mass * 100
        print(
            f"{thresh:<15} {len(remaining):>20} {pct_rules:>14.1f}% {pct_prob:>19.2f}%")

    # Show rules that would be kept at a reasonable threshold (0.001)
    reasonable_threshold = 0.001
    print(f"\n" + "-" * 70)
    print(f"Rules with probability >= {reasonable_threshold}:")
    print("-" * 70)

    filtered_rules = [(p, r) for r, p in collapsed_grammar.items()
                      if p >= reasonable_threshold]
    filtered_rules.sort(reverse=True)

    # Group by LHS
    filtered_by_lhs = defaultdict(list)
    for prob, rule in filtered_rules:
        lhs = rule.split(' -> ')[0]
        filtered_by_lhs[lhs].append((prob, rule))

    print(f"\nFiltered rules by LHS category:")
    print(f"{'LHS':<15} {'Rules':>10} {'Prob Mass':>15}")
    print("-" * 40)
    for lhs in sorted(filtered_by_lhs.keys()):
        rules = filtered_by_lhs[lhs]
        prob_mass = sum(p for p, r in rules)
        print(f"{lhs:<15} {len(rules):>10} {prob_mass:>15.4f}")

    # Check coverage after filtering
    print(f"\n" + "-" * 70)
    print(
        f"Coverage analysis after filtering (threshold = {reasonable_threshold}):")
    print("-" * 70)

    filtered_prob_by_lhs = defaultdict(float)
    for prob, rule in filtered_rules:
        lhs = rule.split(' -> ')[0]
        filtered_prob_by_lhs[lhs] += prob

    print(f"\n{'LHS':<15} {'Filtered Prob':>15} {'Total Prob':>15} {'Coverage':>10}")
    print("-" * 55)
    low_coverage_after = 0
    for lhs in sorted(filtered_prob_by_lhs.keys()):
        filtered_prob = filtered_prob_by_lhs[lhs]
        total_prob = all_rules_by_base_lhs[lhs]
        coverage = filtered_prob / total_prob if total_prob > 0 else 0
        status = "" if coverage >= 0.9 else " <-- LOW"
        print(
            f"{lhs:<15} {filtered_prob:>15.4f} {total_prob:>15.4f} {coverage:>9.2%}{status}")
        if coverage < 0.9:
            low_coverage_after += 1

    print(f"\nLHS categories with coverage < 90%: {low_coverage_after}")

    # 11. Save collapsed grammar to file
    output_file = 'collapsed_grammar.txt'
    with open(output_file, 'w') as f:
        f.write(f"# Collapsed Berkeley grammar (split-merge merged)\n")
        f.write(f"# Total rules: {len(collapsed_grammar)}\n")
        f.write(f"# Format: probability<tab>rule\n\n")
        for rule, prob in sorted(collapsed_grammar.items(), key=lambda x: -x[1]):
            f.write(f"{prob}\t{rule}\n")
    print(f"\nSaved full collapsed grammar to '{output_file}'")

    # Save filtered grammar
    filtered_output_file = f'collapsed_grammar_thresh_{reasonable_threshold}.txt'
    with open(filtered_output_file, 'w') as f:
        f.write(f"# Collapsed Berkeley grammar (split-merge merged)\n")
        f.write(f"# Threshold: {reasonable_threshold}\n")
        f.write(f"# Total rules: {len(filtered_rules)}\n")
        f.write(f"# Format: probability<tab>rule\n\n")
        for prob, rule in filtered_rules:
            f.write(f"{prob}\t{rule}\n")
    print(f"Saved filtered collapsed grammar to '{filtered_output_file}'")


if __name__ == '__main__':
    main()
