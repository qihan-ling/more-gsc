#!/usr/bin/env python3
"""
Penn Treebank to GSC Parser Preprocessor

This script:
1. Parses Penn Treebank trees (bracketed format)
2. Extracts PCFG rules with probabilities (MLE)
3. Converts sentences to GSC input format (FILLER/ROLE bindings)
"""

import re
from collections import Counter, defaultdict
from typing import List, Tuple, Dict, Optional
import sys


class TreeNode:
    """Simple tree node for Penn Treebank trees"""
    def __init__(self, label: str, children: Optional[List['TreeNode']] = None, word: Optional[str] = None):
        self.label = label
        self.children = children or []
        self.word = word  # For terminal nodes

    def is_terminal(self) -> bool:
        return self.word is not None

    def __repr__(self):
        if self.is_terminal():
            return f"({self.label} {self.word})"
        return f"({self.label} {' '.join(str(c) for c in self.children)})"


def parse_bracket_tree(s: str) -> TreeNode:
    """Parse Penn Treebank bracketed format into TreeNode structure

    Example: (S (NP (DT the) (NN cat)) (VP (VBD sat)))

    Creates proper tree structure where:
    - Pre-terminals (POS tags) have one terminal child (the word)
    - Terminals (words) have no children
    """
    s = s.strip()
    if not s.startswith('('):
        raise ValueError(f"Invalid tree format: {s}")

    # Remove outer parentheses
    s = s[1:-1].strip()

    # Find the label (first token)
    match = re.match(r'^(\S+)\s*(.*)', s)
    if not match:
        raise ValueError(f"Cannot parse label from: {s}")

    label = match.group(1)
    rest = match.group(2).strip()

    if not rest:
        # Leaf node without word (shouldn't happen in valid PTB)
        return TreeNode(label)

    # Check if this is a pre-terminal (POS tag with single word)
    if not rest.startswith('('):
        # This is a pre-terminal (POS tag)
        # Create a terminal child node for the word
        word_node = TreeNode(label='WORD', word=rest)
        return TreeNode(label, children=[word_node])

    # Parse children
    children = []
    depth = 0
    start = 0

    for i, char in enumerate(rest):
        if char == '(':
            if depth == 0:
                start = i
            depth += 1
        elif char == ')':
            depth -= 1
            if depth == 0:
                child_str = rest[start:i+1]
                children.append(parse_bracket_tree(child_str))

    return TreeNode(label, children=children)


def extract_rules(node: TreeNode, rules: List[Tuple[str, Tuple[str, ...]]],
                  binarize: bool = False) -> None:
    """Extract production rules from a tree

    Args:
        node: Current tree node
        rules: List to accumulate rules (LHS, (RHS1, RHS2, ...))
        binarize: Whether to binarize rules (convert n-ary to binary)
    """
    if node.is_terminal():
        return

    # Get the production rule
    if len(node.children) == 0:
        return

    # For pre-terminals (POS -> word), we might want to keep or abstract
    if len(node.children) == 1 and node.children[0].is_terminal():
        # This is a lexical rule: POS -> word
        # For PCFG, we typically abstract this to just the POS tag
        # and handle words separately
        word = node.children[0].word
        rules.append((node.label, (word,)))
        return

    # Extract children labels
    child_labels = tuple(child.label for child in node.children)

    if binarize and len(child_labels) > 2:
        # Binarize: A -> B C D becomes A -> B A', A' -> C D
        # Create intermediate symbols
        rules.append((node.label, (child_labels[0], f"{node.label}'")))

        for i in range(1, len(child_labels) - 2):
            rules.append((f"{node.label}'", (child_labels[i], f"{node.label}'")))

        rules.append((f"{node.label}'", (child_labels[-2], child_labels[-1])))
    else:
        rules.append((node.label, child_labels))

    # Recursively process children
    for child in node.children:
        extract_rules(child, rules, binarize)


def compute_pcfg_probabilities(rules: List[Tuple[str, Tuple[str, ...]]],
                               smooth: float = 0.0) -> Dict[Tuple[str, Tuple[str, ...]], float]:
    """Compute MLE probabilities for PCFG rules

    Args:
        rules: List of production rules
        smooth: Additive smoothing constant (default: 0)

    Returns:
        Dictionary mapping rule to probability
    """
    # Count rules
    rule_counts = Counter(rules)

    # Count LHS occurrences
    lhs_counts = Counter(rule[0] for rule in rules)

    # Compute probabilities
    probs = {}
    for rule, count in rule_counts.items():
        lhs = rule[0]
        probs[rule] = (count + smooth) / (lhs_counts[lhs] + smooth * len([r for r in rule_counts if r[0] == lhs]))

    return probs


def format_pcfg_for_gsc(rule_probs: Dict[Tuple[str, Tuple[str, ...]], float],
                        min_prob: float = 0.001) -> str:
    """Format PCFG rules in GSC format

    Args:
        rule_probs: Dictionary of rules with probabilities
        min_prob: Minimum probability threshold to include

    Returns:
        String in GSC PCFG format
    """
    # Group by LHS
    lhs_groups = defaultdict(list)
    for (lhs, rhs), prob in rule_probs.items():
        if prob >= min_prob:
            lhs_groups[lhs].append((rhs, prob))

    # Sort for consistency
    output_lines = []
    for lhs in sorted(lhs_groups.keys()):
        rules = sorted(lhs_groups[lhs], key=lambda x: -x[1])  # Sort by probability descending
        for rhs, prob in rules:
            rhs_str = ' '.join(rhs)
            output_lines.append(f"{prob:.6f} {lhs} -> {rhs_str}")
        output_lines.append("")  # Blank line between LHS groups

    return '\n'.join(output_lines)


def tree_to_role_positions(node: TreeNode,
                          level: int = 1,
                          position: int = 1,
                          positions: Optional[List[Tuple[str, int, int]]] = None) -> List[Tuple[str, int, int]]:
    """Convert tree to GSC role positions (bottom-up, left-to-right)

    Args:
        node: Current tree node
        level: Current level (1 = terminals)
        position: Current position at this level
        positions: Accumulator for (label, level, position) tuples

    Returns:
        List of (label, level, position) for all nodes
    """
    if positions is None:
        positions = []

    if node.is_terminal():
        # Terminal: use POS tag
        positions.append((node.label, level, position))
        return positions

    if len(node.children) == 1 and node.children[0].is_terminal():
        # Pre-terminal (POS tag)
        positions.append((node.label, level, position))
        return positions

    # Process children first (bottom-up)
    child_positions = []
    for i, child in enumerate(node.children):
        tree_to_role_positions(child, level, position + i, positions)
        child_positions.append((level, position + i))

    # Add current node at higher level
    # Position at higher level is determined by leftmost child
    parent_position = position
    positions.append((node.label, level + 1, parent_position))

    return positions


def tree_to_gsc_input(node: TreeNode, max_sent_len: int = 10) -> List[str]:
    """Convert Penn Treebank tree to GSC input format

    The GSC format is: FILLER/(level,position)
    - Level 1 = terminal/POS tags
    - Higher levels = phrasal nodes

    Args:
        node: Parsed tree
        max_sent_len: Maximum sentence length

    Returns:
        List of binding strings like ['DT/(1,1)', 'NN/(1,2)', 'VBD/(1,3)']
    """
    # Get terminal sequence (POS tags in order)
    terminals = []

    def collect_terminals(n):
        if n.is_terminal():
            return
        if len(n.children) == 1 and n.children[0].is_terminal():
            # Pre-terminal (POS tag) - this is what we want
            terminals.append(n.label)
        else:
            # Recurse in left-to-right order
            for child in n.children:
                collect_terminals(child)

    collect_terminals(node)

    # Create GSC input format with level 1 positions
    gsc_input = []
    for i, pos_tag in enumerate(terminals, 1):
        gsc_input.append(f"{pos_tag}/({1},{i})")

    return gsc_input


def process_ptb_file(filepath: str,
                    binarize: bool = False,
                    max_trees: Optional[int] = None) -> Tuple[List[Tuple[str, Tuple[str, ...]]], List[List[str]]]:
    """Process Penn Treebank file

    Args:
        filepath: Path to .mrg or bracketed tree file
        binarize: Whether to binarize rules
        max_trees: Maximum number of trees to process

    Returns:
        Tuple of (rules, gsc_sentences)
    """
    rules = []
    gsc_sentences = []

    with open(filepath, 'r', encoding='utf-8') as f:
        content = f.read()

    # PTB files contain one tree per line or multiple lines
    # Trees start with ( and end with )
    # We need to extract complete trees

    tree_strings = []
    current_tree = []
    depth = 0

    for line in content.split('\n'):
        line = line.strip()
        if not line:
            continue

        for char in line:
            if char == '(':
                depth += 1
            elif char == ')':
                depth -= 1

        current_tree.append(line)

        if depth == 0 and current_tree:
            tree_strings.append(' '.join(current_tree))
            current_tree = []

            if max_trees and len(tree_strings) >= max_trees:
                break

    # Parse each tree
    for i, tree_str in enumerate(tree_strings):
        try:
            tree = parse_bracket_tree(tree_str)

            # Extract rules
            tree_rules = []
            extract_rules(tree, tree_rules, binarize)
            rules.extend(tree_rules)

            # Convert to GSC format
            gsc_sent = tree_to_gsc_input(tree)
            gsc_sentences.append(gsc_sent)

        except Exception as e:
            print(f"Error processing tree {i}: {e}", file=sys.stderr)
            continue

    return rules, gsc_sentences


def main():
    """Main function with example usage"""
    import argparse

    parser = argparse.ArgumentParser(description='Convert Penn Treebank to GSC format')
    parser.add_argument('input', help='Penn Treebank file (.mrg or bracketed format)')
    parser.add_argument('--output-pcfg', default='ptb_grammar.txt',
                       help='Output PCFG file (default: ptb_grammar.txt)')
    parser.add_argument('--output-sentences', default='ptb_sentences.txt',
                       help='Output GSC sentences file (default: ptb_sentences.txt)')
    parser.add_argument('--binarize', action='store_true',
                       help='Binarize grammar rules')
    parser.add_argument('--min-prob', type=float, default=0.001,
                       help='Minimum probability threshold (default: 0.001)')
    parser.add_argument('--max-trees', type=int, default=None,
                       help='Maximum number of trees to process')
    parser.add_argument('--smooth', type=float, default=0.0,
                       help='Additive smoothing constant (default: 0.0)')

    args = parser.parse_args()

    print(f"Processing {args.input}...")
    rules, gsc_sentences = process_ptb_file(args.input,
                                            binarize=args.binarize,
                                            max_trees=args.max_trees)

    print(f"Extracted {len(rules)} rules from {len(gsc_sentences)} sentences")

    # Compute probabilities
    print("Computing PCFG probabilities...")
    rule_probs = compute_pcfg_probabilities(rules, smooth=args.smooth)

    # Format PCFG
    pcfg_str = format_pcfg_for_gsc(rule_probs, min_prob=args.min_prob)

    # Write PCFG
    with open(args.output_pcfg, 'w') as f:
        f.write(pcfg_str)
    print(f"Wrote PCFG to {args.output_pcfg}")

    # Write sentences
    with open(args.output_sentences, 'w') as f:
        for sent in gsc_sentences:
            f.write(' '.join(sent) + '\n')
    print(f"Wrote {len(gsc_sentences)} sentences to {args.output_sentences}")

    # Print statistics
    print("\n=== Statistics ===")
    print(f"Unique rules: {len(rule_probs)}")
    print(f"Unique non-terminals: {len(set(r[0] for r in rule_probs.keys()))}")
    print(f"Average sentence length: {sum(len(s) for s in gsc_sentences) / len(gsc_sentences):.1f}")


if __name__ == '__main__':
    main()
