#!/usr/bin/env python3
"""
Train Berkeley Parser with Split-Merge 5

This script trains a Berkeley (Petrov) parser on preprocessed Penn Treebank data
using the split-merge algorithm with 5 iterations for optimal accuracy.

The Berkeley parser uses latent variable grammars learned via EM:
- Starts with X-bar grammar
- Iteratively splits categories (e.g., NP -> NP-0, NP-1, ...)
- Merges back unhelpful splits
- SM-5 typically gives best F1 score (~90% on PTB)

References:
  Petrov et al. (2006): Learning Accurate, Compact, and Interpretable Tree Annotation
  https://github.com/slavpetrov/berkeleyparser

Requirements:
  - Java 8 or higher
  - Berkeley Parser jar file
  - Preprocessed PTB data (from preprocess_ptb_for_berkeley.py)

Usage:
  python train_berkeley_parser.py \
      --train-file berkeley_data/train.trees \
      --dev-file berkeley_data/dev.trees \
      --output-grammar berkeley_sm5.gr \
      --num-splits 5

Author: Claude
Date: 2025-10-23
"""

import os
import sys
import argparse
import subprocess
import time
from pathlib import Path


class BerkeleyParserTrainer:
    """Wrapper for training Berkeley parser"""

    def __init__(self, jar_path=None):
        """Initialize trainer

        Args:
            jar_path: Path to BerkeleyParser.jar (auto-detect if None)
        """
        self.jar_path = jar_path or self.find_berkeley_jar()

        if not self.jar_path or not os.path.exists(self.jar_path):
            raise FileNotFoundError(
                "Berkeley Parser jar not found. Please specify with --jar-path or "
                "download from: https://github.com/slavpetrov/berkeleyparser/releases"
            )

        print(f"Using Berkeley Parser: {self.jar_path}")

    def find_berkeley_jar(self):
        """Try to find Berkeley parser jar in common locations

        Returns:
            Path to jar file or None
        """
        common_names = [
            'BerkeleyParser.jar',
            'berkeleyParser.jar',
            'berkeley-parser.jar'
        ]

        common_paths = [
            '.',
            './berkeley',
            './lib',
            '../berkeley',
            os.path.expanduser('~/berkeley'),
        ]

        for path in common_paths:
            for name in common_names:
                jar_path = os.path.join(path, name)
                if os.path.exists(jar_path):
                    return jar_path

        return None

    def train(self, train_file, dev_file=None, output_grammar='grammar.gr',
              num_splits=5, num_iterations=50, use_smoothing=True,
              rare_word_threshold=20, max_sentence_length=40,
              memory='8g', num_threads=None):
        """Train Berkeley parser with split-merge

        Args:
            train_file: Training trees file
            dev_file: Development trees file (optional, for early stopping)
            output_grammar: Output grammar file
            num_splits: Number of split-merge iterations (5 recommended)
            num_iterations: EM iterations per split-merge cycle
            use_smoothing: Use smoothing (recommended)
            rare_word_threshold: Threshold for rare words
            max_sentence_length: Maximum sentence length to train on
            memory: JVM memory allocation (e.g., '8g')
            num_threads: Number of threads (None = auto)

        Returns:
            Path to output grammar file
        """
        print("="*70)
        print("Berkeley Parser Training with Split-Merge")
        print("="*70)

        # Verify input files
        if not os.path.exists(train_file):
            raise FileNotFoundError(f"Training file not found: {train_file}")

        print(f"\nConfiguration:")
        print(f"  Train file: {train_file}")
        print(f"  Dev file: {dev_file or 'None'}")
        print(f"  Output grammar: {output_grammar}")
        print(f"  Split-merge cycles: {num_splits}")
        print(f"  EM iterations per cycle: {num_iterations}")
        print(f"  Smoothing: {use_smoothing}")
        print(f"  Max sentence length: {max_sentence_length}")
        print(f"  Memory: {memory}")

        # Build command
        cmd = [
            'java',
            f'-Xmx{memory}',
            '-jar', self.jar_path,
            '-path', train_file,
            '-out', output_grammar,
            '-treebank', 'SINGLEFILE',  # Our preprocessed format
        ]

        # Add optional arguments
        if dev_file and os.path.exists(dev_file):
            cmd.extend(['-testpath', dev_file])

        cmd.extend([
            '-SMcycles', str(num_splits),
            '-EMIterations', str(num_iterations),
            '-rare', str(rare_word_threshold),
            '-maxLength', str(max_sentence_length),
        ])

        if use_smoothing:
            cmd.append('-smooth')

        if num_threads:
            cmd.extend(['-nThreads', str(num_threads)])

        # Display command
        print(f"\nCommand:")
        print(' '.join(cmd))

        # Run training
        print(f"\n{'='*70}")
        print("Training started...")
        print(f"{'='*70}\n")

        start_time = time.time()

        try:
            process = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                universal_newlines=True,
                bufsize=1
            )

            # Stream output in real-time
            for line in process.stdout:
                print(line, end='')

            process.wait()

            if process.returncode != 0:
                raise RuntimeError(f"Training failed with return code {process.returncode}")

        except KeyboardInterrupt:
            print("\n\nTraining interrupted by user")
            process.terminate()
            sys.exit(1)

        elapsed_time = time.time() - start_time

        print(f"\n{'='*70}")
        print("Training completed!")
        print(f"{'='*70}")
        print(f"Time elapsed: {elapsed_time/60:.1f} minutes")
        print(f"Grammar saved to: {output_grammar}")

        # Verify output
        if os.path.exists(output_grammar):
            size_mb = os.path.getsize(output_grammar) / (1024 * 1024)
            print(f"Grammar file size: {size_mb:.1f} MB")
        else:
            print("Warning: Grammar file not found!")

        return output_grammar

    def quick_test(self, grammar_file, test_sentence=None):
        """Quick test of trained grammar

        Args:
            grammar_file: Trained grammar file
            test_sentence: Test sentence (default: use simple example)
        """
        if test_sentence is None:
            test_sentence = "The cat sat on the mat ."

        print(f"\n{'='*70}")
        print("Quick Test")
        print(f"{'='*70}")
        print(f"Sentence: {test_sentence}")

        # Create temporary file with test sentence
        import tempfile
        with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False) as f:
            f.write(test_sentence + '\n')
            temp_file = f.name

        try:
            # Parse command
            cmd = [
                'java',
                '-Xmx2g',
                '-jar', self.jar_path,
                '-gr', grammar_file,
                '-inputFile', temp_file,
                '-maxLength', '100',
            ]

            # Run parser
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=30
            )

            if result.returncode == 0:
                print("\nParse tree:")
                print(result.stdout)
            else:
                print("Parsing failed")
                print(result.stderr)

        finally:
            os.unlink(temp_file)


def main():
    parser = argparse.ArgumentParser(
        description='Train Berkeley Parser with Split-Merge',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Train with SM-5 (recommended for best accuracy)
  python train_berkeley_parser.py \\
      --train-file berkeley_data/train.trees \\
      --dev-file berkeley_data/dev.trees \\
      --output-grammar berkeley_sm5.gr \\
      --num-splits 5

  # Quick training with SM-2 (faster, for testing)
  python train_berkeley_parser.py \\
      --train-file berkeley_data/train.trees \\
      --output-grammar berkeley_sm2.gr \\
      --num-splits 2 \\
      --memory 4g

  # Full training with custom settings
  python train_berkeley_parser.py \\
      --train-file berkeley_data/train.trees \\
      --dev-file berkeley_data/dev.trees \\
      --output-grammar berkeley_sm6.gr \\
      --num-splits 6 \\
      --num-iterations 100 \\
      --memory 16g

Berkeley Parser Download:
  https://github.com/slavpetrov/berkeleyparser/releases

Typical training times (on modern CPU):
  - SM-1: ~30 minutes
  - SM-2: ~1 hour
  - SM-5: ~4-6 hours  (recommended)
  - SM-6: ~8-10 hours

Memory requirements:
  - Minimum: 4GB
  - Recommended: 8GB
  - For large corpora: 16GB
        """
    )

    parser.add_argument(
        '--train-file',
        type=str,
        required=True,
        help='Training trees file (from preprocess_ptb_for_berkeley.py)'
    )

    parser.add_argument(
        '--dev-file',
        type=str,
        default=None,
        help='Development trees file (optional, for monitoring)'
    )

    parser.add_argument(
        '--output-grammar',
        type=str,
        default='berkeley_sm5.gr',
        help='Output grammar file (default: berkeley_sm5.gr)'
    )

    parser.add_argument(
        '--num-splits',
        type=int,
        default=5,
        help='Number of split-merge cycles (default: 5, recommended)'
    )

    parser.add_argument(
        '--num-iterations',
        type=int,
        default=50,
        help='EM iterations per split-merge cycle (default: 50)'
    )

    parser.add_argument(
        '--memory',
        type=str,
        default='8g',
        help='JVM memory allocation (default: 8g)'
    )

    parser.add_argument(
        '--jar-path',
        type=str,
        default=None,
        help='Path to BerkeleyParser.jar (auto-detect if not specified)'
    )

    parser.add_argument(
        '--rare-word-threshold',
        type=int,
        default=20,
        help='Rare word threshold (default: 20)'
    )

    parser.add_argument(
        '--max-sentence-length',
        type=int,
        default=40,
        help='Maximum sentence length to train on (default: 40)'
    )

    parser.add_argument(
        '--num-threads',
        type=int,
        default=None,
        help='Number of threads (default: auto)'
    )

    parser.add_argument(
        '--no-smoothing',
        action='store_true',
        help='Disable smoothing (not recommended)'
    )

    parser.add_argument(
        '--test',
        action='store_true',
        help='Run quick test after training'
    )

    args = parser.parse_args()

    # Create trainer
    try:
        trainer = BerkeleyParserTrainer(jar_path=args.jar_path)
    except FileNotFoundError as e:
        print(f"Error: {e}")
        print("\nPlease download Berkeley Parser from:")
        print("https://github.com/slavpetrov/berkeleyparser/releases")
        print("\nOr specify jar path with --jar-path")
        sys.exit(1)

    # Train
    try:
        grammar_file = trainer.train(
            train_file=args.train_file,
            dev_file=args.dev_file,
            output_grammar=args.output_grammar,
            num_splits=args.num_splits,
            num_iterations=args.num_iterations,
            use_smoothing=not args.no_smoothing,
            rare_word_threshold=args.rare_word_threshold,
            max_sentence_length=args.max_sentence_length,
            memory=args.memory,
            num_threads=args.num_threads
        )

        # Quick test if requested
        if args.test:
            trainer.quick_test(grammar_file)

        print("\nNext steps:")
        print(f"  1. Grammar saved to: {grammar_file}")
        print("  2. Extract rules: python extract_berkeley_grammar.py")

    except Exception as e:
        print(f"\nError during training: {e}")
        sys.exit(1)


if __name__ == '__main__':
    main()
