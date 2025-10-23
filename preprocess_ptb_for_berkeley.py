#!/usr/bin/env python3
"""
Preprocess Penn Treebank for Berkeley Parser Training

This script prepares Penn Treebank data for training the Berkeley parser:
- Extracts trees from PTB .mrg files
- Handles standard train/dev/test splits (sections 02-21/22/23)
- Cleans and formats trees
- Outputs in format expected by Berkeley parser

Standard PTB WSJ splits:
  - Training: sections 02-21 (39,832 sentences)
  - Development: section 22 (1,700 sentences)
  - Test: section 23 (2,416 sentences)

Usage:
  python preprocess_ptb_for_berkeley.py \
      --ptb-root /path/to/ptb3/parsed/mrg/wsj \
      --output-dir ./berkeley_data

Author: Claude
Date: 2025-10-23
"""

import os
import sys
import argparse
import re
from pathlib import Path


class PTBPreprocessor:
    """Preprocesses Penn Treebank data for Berkeley parser"""

    def __init__(self, ptb_root, output_dir):
        """Initialize preprocessor

        Args:
            ptb_root: Root directory of PTB parsed/mrg/wsj
            output_dir: Output directory for processed files
        """
        self.ptb_root = Path(ptb_root)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Standard PTB splits
        self.train_sections = list(range(2, 22))  # 02-21
        self.dev_sections = [22]
        self.test_sections = [23]

    def process_all_splits(self):
        """Process train, dev, and test splits"""
        print("="*70)
        print("Penn Treebank Preprocessing for Berkeley Parser")
        print("="*70)

        splits = [
            ('train', self.train_sections),
            ('dev', self.dev_sections),
            ('test', self.test_sections)
        ]

        for split_name, sections in splits:
            output_file = self.output_dir / f'{split_name}.trees'
            self.process_split(sections, output_file, split_name)

        print("\n" + "="*70)
        print("Preprocessing complete!")
        print("="*70)
        print(f"Output directory: {self.output_dir}")
        print("Files created:")
        for split_name, _ in splits:
            filepath = self.output_dir / f'{split_name}.trees'
            if filepath.exists():
                num_lines = sum(1 for _ in open(filepath))
                print(f"  - {filepath.name}: {num_lines:,} trees")

    def process_split(self, sections, output_file, split_name):
        """Process one data split

        Args:
            sections: List of section numbers (e.g., [2, 3, 4, ..., 21])
            output_file: Output file path
            split_name: Name of split ('train', 'dev', 'test')
        """
        print(f"\n[{split_name.upper()}] Processing sections {sections}")

        all_trees = []
        total_files = 0

        for section in sections:
            section_dir = self.ptb_root / f"{section:02d}"

            if not section_dir.exists():
                print(f"  Warning: Section directory not found: {section_dir}")
                continue

            # Process all .mrg files in this section
            mrg_files = sorted(section_dir.glob("*.mrg"))
            total_files += len(mrg_files)

            for mrg_file in mrg_files:
                trees = self.extract_trees_from_file(mrg_file)
                all_trees.extend(trees)

        print(f"  Processed {total_files} files")
        print(f"  Extracted {len(all_trees):,} trees")

        # Write to output file
        with open(output_file, 'w', encoding='utf-8') as f:
            for tree in all_trees:
                f.write(tree + '\n')

        print(f"  Wrote to: {output_file}")

    def extract_trees_from_file(self, filepath):
        """Extract trees from a single .mrg file

        Args:
            filepath: Path to .mrg file

        Returns:
            List of tree strings
        """
        with open(filepath, 'r', encoding='utf-8') as f:
            content = f.read()

        # Extract trees (each starts with ( and ends with ))
        trees = []
        current_tree = []
        depth = 0
        in_tree = False

        for line in content.split('\n'):
            line = line.strip()

            if not line:
                continue

            # Track depth to find complete trees
            for char in line:
                if char == '(':
                    depth += 1
                    in_tree = True
                elif char == ')':
                    depth -= 1

            current_tree.append(line)

            # Complete tree found
            if in_tree and depth == 0:
                tree_str = ' '.join(current_tree)
                tree_str = self.clean_tree(tree_str)

                if tree_str:  # Only add non-empty trees
                    trees.append(tree_str)

                current_tree = []
                in_tree = False

        return trees

    def clean_tree(self, tree_str):
        """Clean and normalize tree string

        Args:
            tree_str: Raw tree string

        Returns:
            Cleaned tree string
        """
        # Remove extra whitespace
        tree_str = ' '.join(tree_str.split())

        # Remove empty nodes: ( )
        tree_str = re.sub(r'\(\s+\)', '', tree_str)

        # Berkeley parser doesn't like certain annotations
        # Remove function tags (e.g., NP-SBJ -> NP)
        # Uncomment if you want to remove function tags:
        # tree_str = re.sub(r'([A-Z]+)-[A-Z]+', r'\1', tree_str)

        # Remove traces and null elements
        # PTB has things like (NP-NONE- *T*) which parsers often skip
        tree_str = re.sub(r'\([A-Z]+-NONE-[^\)]*\)', '', tree_str)

        # Remove trace co-indexing (e.g., NP-1 -> NP)
        # Uncomment if you want to remove indices:
        # tree_str = re.sub(r'([A-Z]+)-\d+', r'\1', tree_str)

        # Clean up any double spaces created
        tree_str = ' '.join(tree_str.split())

        return tree_str.strip()

    def get_statistics(self):
        """Print statistics about the processed data"""
        print("\n" + "="*70)
        print("Dataset Statistics")
        print("="*70)

        for split_name in ['train', 'dev', 'test']:
            filepath = self.output_dir / f'{split_name}.trees'

            if not filepath.exists():
                continue

            trees = []
            with open(filepath, 'r') as f:
                trees = [line.strip() for line in f if line.strip()]

            # Count words (rough approximation)
            total_words = 0
            max_len = 0
            min_len = float('inf')

            for tree in trees:
                # Count terminals (words) - rough heuristic
                words = re.findall(r'\([A-Z$]+\s+([^\)]+)\)', tree)
                num_words = len(words)
                total_words += num_words
                max_len = max(max_len, num_words)
                min_len = min(min_len, num_words)

            avg_len = total_words / len(trees) if trees else 0

            print(f"\n{split_name.upper()}:")
            print(f"  Trees: {len(trees):,}")
            print(f"  Words: {total_words:,}")
            print(f"  Avg length: {avg_len:.1f}")
            print(f"  Min length: {min_len}")
            print(f"  Max length: {max_len}")


def main():
    parser = argparse.ArgumentParser(
        description='Preprocess Penn Treebank for Berkeley Parser',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Process PTB with standard splits
  python preprocess_ptb_for_berkeley.py \\
      --ptb-root /path/to/treebank_3/parsed/mrg/wsj \\
      --output-dir ./berkeley_data

  # Process and show statistics
  python preprocess_ptb_for_berkeley.py \\
      --ptb-root /path/to/ptb3/parsed/mrg/wsj \\
      --output-dir ./berkeley_data \\
      --statistics

  # Custom sections (e.g., just sections 2-5 for testing)
  python preprocess_ptb_for_berkeley.py \\
      --ptb-root /path/to/ptb3/parsed/mrg/wsj \\
      --output-dir ./berkeley_data_small \\
      --train-sections 2 3 4 5

Note: PTB3 structure should be:
  ptb3/
    parsed/
      mrg/
        wsj/
          00/  01/  02/  ...  24/
            *.mrg files
        """
    )

    parser.add_argument(
        '--ptb-root',
        type=str,
        required=True,
        help='Root directory of PTB parsed/mrg/wsj'
    )

    parser.add_argument(
        '--output-dir',
        type=str,
        default='./berkeley_data',
        help='Output directory for processed files (default: ./berkeley_data)'
    )

    parser.add_argument(
        '--train-sections',
        type=int,
        nargs='+',
        default=None,
        help='Custom training sections (default: 2-21)'
    )

    parser.add_argument(
        '--dev-sections',
        type=int,
        nargs='+',
        default=None,
        help='Custom dev sections (default: 22)'
    )

    parser.add_argument(
        '--test-sections',
        type=int,
        nargs='+',
        default=None,
        help='Custom test sections (default: 23)'
    )

    parser.add_argument(
        '--statistics',
        action='store_true',
        help='Print statistics after preprocessing'
    )

    args = parser.parse_args()

    # Verify PTB root exists
    if not os.path.exists(args.ptb_root):
        print(f"Error: PTB root directory not found: {args.ptb_root}")
        print("\nExpected structure:")
        print("  ptb3/parsed/mrg/wsj/")
        print("    00/  01/  02/  ...  24/")
        print("      *.mrg files")
        sys.exit(1)

    # Create preprocessor
    preprocessor = PTBPreprocessor(args.ptb_root, args.output_dir)

    # Override sections if specified
    if args.train_sections:
        preprocessor.train_sections = args.train_sections
    if args.dev_sections:
        preprocessor.dev_sections = args.dev_sections
    if args.test_sections:
        preprocessor.test_sections = args.test_sections

    # Process data
    preprocessor.process_all_splits()

    # Show statistics if requested
    if args.statistics:
        preprocessor.get_statistics()

    print("\nNext steps:")
    print("  1. Verify output files in:", args.output_dir)
    print("  2. Run: python train_berkeley_parser.py")


if __name__ == '__main__':
    main()
