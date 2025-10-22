#!/usr/bin/env python3
"""
Train GSC parser using Penn Treebank extracted PCFG

This script demonstrates how to:
1. Load PCFG extracted from Penn Treebank
2. Initialize GSC network
3. Train on PTB sentences
"""

import gsc
import sys


def load_pcfg(filepath):
    """Load PCFG from file in GSC format

    Format:
        probability LHS -> RHS1 RHS2 ...
        0.5 S -> NP VP
        0.3 NP -> DT NN
    """
    with open(filepath, 'r') as f:
        return f.read()


def load_gsc_sentences(filepath, max_sentences=None):
    """Load GSC formatted sentences

    Format (one sentence per line):
        N/(1,1) Vi/(1,2) P/(2,1) N/(2,2)
    """
    sentences = []
    with open(filepath, 'r') as f:
        for i, line in enumerate(f):
            line = line.strip()
            if line:
                # Split into binding names
                bindings = line.split()
                sentences.append(bindings)

                if max_sentences and i + 1 >= max_sentences:
                    break

    return sentences


def main():
    import argparse

    parser = argparse.ArgumentParser(description='Train GSC parser from Penn Treebank')
    parser.add_argument('--pcfg', default='ptb_grammar.txt',
                       help='PCFG file (default: ptb_grammar.txt)')
    parser.add_argument('--sentences', default='ptb_sentences.txt',
                       help='GSC sentences file (default: ptb_sentences.txt)')
    parser.add_argument('--max-sent-len', type=int, default=10,
                       help='Maximum sentence length (default: 10)')
    parser.add_argument('--root', default='S',
                       help='Root symbol (default: S)')
    parser.add_argument('--num-epochs', type=int, default=100,
                       help='Number of training epochs (default: 100)')
    parser.add_argument('--learning-rate', type=float, default=0.1,
                       help='Learning rate (default: 0.1)')
    parser.add_argument('--num-samples', type=int, default=5000,
                       help='Number of corpus samples to generate (default: 5000)')
    parser.add_argument('--output', default='gsc_ptb_model.pkl',
                       help='Output model file (default: gsc_ptb_model.pkl)')
    parser.add_argument('--seed', type=int, default=1024,
                       help='Random seed (default: 1024)')

    args = parser.parse_args()

    # Load PCFG
    print(f"Loading PCFG from {args.pcfg}...")
    try:
        pcfg_str = load_pcfg(args.pcfg)
        print(f"PCFG loaded ({len(pcfg_str.splitlines())} lines)")
    except FileNotFoundError:
        print(f"ERROR: PCFG file not found: {args.pcfg}")
        print("Please run ptb_to_gsc.py first to extract PCFG from Penn Treebank")
        sys.exit(1)

    # Initialize Harmonic Grammar
    print(f"\nInitializing Harmonic Grammar (max_sent_len={args.max_sent_len}, root={args.root})...")
    try:
        hg = gsc.HarmonicGrammar(
            pcfg=pcfg_str,
            root=args.root,
            max_sent_len=args.max_sent_len
        )
        print(f"Grammar initialized:")
        print(f"  Fillers: {len(hg.fillers)}")
        print(f"  Roles: {len(hg.roles)}")
        print(f"  Binding units: {len(hg.fillers) * len(hg.roles)}")
    except Exception as e:
        print(f"ERROR initializing grammar: {e}")
        sys.exit(1)

    # Initialize network
    print("\nInitializing GSC network...")
    net_opts = {
        'T_init': 0.01,
        'q_max': 15.0,
        'q_init': 0.0,
        'dt_init': 0.005,
        'm': 30,
        'lam_x': 0.5,
        'lam_q': 0.04,
        'use_runC': True
    }

    # Similarity encoding (linear independence)
    dp = 0.0
    sim = [[1.0 if i == j else dp for j in range(len(hg.fillers))]
           for i in range(len(hg.fillers))]

    net = gsc.GscNet(
        hg=hg,
        encodings={'similarity': sim},
        opts=net_opts,
        seed=args.seed
    )

    print(f"Network initialized with {len(net.binding_names)} bindings")

    # Generate corpus from PCFG
    print(f"\nGenerating corpus ({args.num_samples} samples)...")
    net.generate_corpus(nsamples=args.num_samples, use_freq=True)
    print(f"Corpus generated: {len(net.corpus)} sentences")

    # Configure training
    train_opts = {
        'lrate': args.learning_rate,
        'num_trials': 4,
        'num_epochs': args.num_epochs,
        'trace_varnames': ['kl_trees', 'kl_treelets', 'prob_sent', 'acc']
    }

    net.initialize(train_opts=train_opts)

    # Train
    print(f"\nTraining for {args.num_epochs} epochs...")
    print("Metrics: KL(trees), KL(treelets), prob_sent, accuracy")
    print("-" * 60)

    net.train2(
        train_opts={'num_epochs': args.num_epochs},
        savefilename=args.output
    )

    print(f"\nTraining complete! Model saved to {args.output}")

    # Print final statistics
    if hasattr(net, 'traces') and net.traces:
        print("\n=== Final Statistics ===")
        for var in ['kl_trees', 'kl_treelets', 'prob_sent', 'acc']:
            if var in net.traces:
                values = net.traces[var]
                if values:
                    print(f"{var}: {values[-1]:.4f}")

    # Optional: Test on a few sentences from PTB
    print("\n=== Testing on PTB sentences ===")
    try:
        test_sentences = load_gsc_sentences(args.sentences, max_sentences=5)
        print(f"Loaded {len(test_sentences)} test sentences")

        for i, sent in enumerate(test_sentences, 1):
            print(f"\nSentence {i}: {' '.join(sent)}")
            # Here you would run the network on this sentence
            # This requires proper implementation of parsing with the GSC network
            # which involves setting input and running network dynamics

    except FileNotFoundError:
        print(f"Test sentences file not found: {args.sentences}")


if __name__ == '__main__':
    main()
