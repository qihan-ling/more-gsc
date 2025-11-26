import pandas as pd
import re

# --- Data Loading ---
agreement_file = '../SAP_stimuli/sap_items_Agreement.csv'
attachment_ambiguity_file = '../SAP_stimuli/sap_items_AttachmentAmbiguity.csv'
classic_gp_file = '../SAP_stimuli/sap_items_ClassicGP.csv'
filler_file = '../SAP_stimuli/sap_items_filler.csv'
relative_clause_file = '../SAP_stimuli/sap_items_RelativeClause.csv'

sap_filenames = [agreement_file, attachment_ambiguity_file,
                 classic_gp_file, filler_file, relative_clause_file]


def tokenize_for_berkeley(sentence):
    """
    Tokenize a sentence for Berkeley Parser (PTB-style tokenization).

    - Separates punctuation from words
    - Handles contractions
    - Handles special punctuation marks
    """
    # Add space before punctuation
    s = re.sub(r'([.,!?;:"\'\)\]\}])', r' \1', sentence)
    # Add space after opening brackets/quotes
    s = re.sub(r'([\(\[\{])', r'\1 ', s)

    # Handle contractions - split them
    # e.g., "don't" -> "do n't", "I'm" -> "I 'm"
    s = re.sub(r"n't\b", " n't", s)
    s = re.sub(r"'(s|m|re|ve|ll|d)\b", r" '\1", s, flags=re.IGNORECASE)

    # Handle double quotes -> PTB style `` and ''
    # Opening quotes (after space or start)
    s = re.sub(r'(^|[\s\(])"', r'\1 `` ', s)
    # Closing quotes
    s = re.sub(r'"', " '' ", s)

    # Normalize whitespace
    s = ' '.join(s.split())

    return s


def main():
    sap_sentences = []
    source_files = []

    print("Loading SAP sentences from CSV files...")
    for sap_filename in sap_filenames:
        try:
            df = pd.read_csv(sap_filename)
            sentences = df['Sentence'].tolist()
            sap_sentences.extend(sentences)
            source_files.extend([sap_filename] * len(sentences))
            print(f"  {sap_filename}: {len(sentences)} sentences")
        except FileNotFoundError:
            print(f"  WARNING: {sap_filename} not found, skipping")
        except KeyError:
            print(
                f"  WARNING: {sap_filename} has no 'Sentence' column, skipping")

    print(f"\nTotal sentences loaded: {len(sap_sentences)}")

    # Remove duplicates while preserving order
    seen = set()
    unique_sentences = []
    for s in sap_sentences:
        if s not in seen and pd.notna(s):
            seen.add(s)
            unique_sentences.append(s)

    print(f"Unique sentences: {len(unique_sentences)}")

    # Tokenize sentences
    print("\nTokenizing sentences for Berkeley Parser...")
    tokenized_sentences = []
    for sent in unique_sentences:
        tokenized = tokenize_for_berkeley(sent)
        tokenized_sentences.append(tokenized)

    # Save to file
    output_file = 'sap_sentences.txt'
    with open(output_file, 'w') as f:
        for sent in tokenized_sentences:
            f.write(sent + '\n')

    print(f"\nSaved {len(tokenized_sentences)} sentences to '{output_file}'")

    # Show a few examples
    print("\n" + "=" * 60)
    print("SAMPLE TOKENIZED SENTENCES (first 5):")
    print("=" * 60)
    for i, (orig, tok) in enumerate(zip(unique_sentences[:5], tokenized_sentences[:5])):
        print(f"\nOriginal:  {orig}")
        print(f"Tokenized: {tok}")

    # Print the command to run Berkeley Parser
    print("\n" + "=" * 60)
    print("NEXT STEP: Run Berkeley Parser")
    print("=" * 60)
    print("""
Run the following command (adjust paths as needed):

    java -jar BerkeleyParser-1.7.jar -gr berkeley_sm5.gr -binarize < sap_sentences.txt > sap_parses.txt

Or with explicit file arguments:

    java -jar BerkeleyParser-1.7.jar -gr berkeley_sm5.gr -binarize -inputFile sap_sentences.txt -outputFile sap_parses.txt

The -binarize flag is important to get trees that match your grammar's binary rules.
""")


if __name__ == '__main__':
    main()
