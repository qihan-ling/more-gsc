import pandas as pd
import re
from collections import defaultdict

# read into all SAP stimuli to get two lists
# list 1 is berkeley_neural_parser_result column from all 5 files
# list 2 is Sentence column from all 5 files
agreement_file = 'SAP_stimuli/sap_items_Agreement.csv'
attachment_ambiguity_file = 'SAP_stimuli/sap_items_AttachmentAmbiguity.csv'
classic_gp_file = 'SAP_stimuli/sap_items_ClassicGP.csv'
filler_file = 'SAP_stimuli/sap_items_filler.csv'
relative_clause_file = 'SAP_stimuli/sap_items_RelativeClause.csv'

sap_filenames = [agreement_file, attachment_ambiguity_file,
                 classic_gp_file, filler_file, relative_clause_file]
sap_sentences = []
sap_parsed_sents = []
for sap_filename in sap_filenames:
    df = pd.read_csv(sap_filename)
    sap_sentences += df['Sentence'].tolist()
    sap_parsed_sents += df['berkeley_neural_parser_result'].tolist()
# check if all SAP words are in sm5.lexicon
# for every sentence in Sentence-list: split by space, and search if it is berkeley_parser_sm5.word, if not, save it
# a list called 'out_of_vocab_words'.
sap_words = set()
sap_words.add(',')
for sent in sap_sentences:
    words = sent.split()
    if words[-1] != '.' and words[-1][-1] == '.':
        words[-1] = words[-1][:-1]
    for i in range(len(words)):
        word = words[i]
        if word[-1] == ',':
            words[i] = word[:-1]
    words = set(words)
    sap_words = sap_words | words


def get_sm5_words(filepath):
    """
    Reads a text file where each line is a word and returns a list of words.

    Args:
        filepath (str): The path to the text file.

    Returns:
        list: A list of strings, where each string is a word from the file.
    """
    words = set()
    with open(filepath, 'r') as file:
        for line in file:
            # .strip() removes leading/trailing whitespace, including newline characters
            word = line.strip()
            if word:  # Ensures empty lines are not added as empty strings
                words.add(word)
    return words


sm5_vocab = get_sm5_words(
    'trained_berkeley_parser_sm5/berkeley_parser_sm5.words')


def check_word_in_sm5(word, sm5_vocab):
    return word in sm5_vocab


out_of_vocab_set = set()
for word in sap_words:
    if not check_word_in_sm5(word, sm5_vocab):
        out_of_vocab_set.add(word)

if len(out_of_vocab_set) > 0:
    print(f"we have at least one word out of sm5 vocabulary!\n")
    print(
        f"the out_of_vocab_set is of size {len(out_of_vocab_set)} and its content is {out_of_vocab_set}")
"""
the out_of_vocab_set is of size 132 and its content is {'burp', 'divas', 're-wrapped', 'bulwarks', 'diva', 'floods', 'Binge', 
'worldwide', 'advisor', 'Rosalina', 'livres', "Winstanley's", 'Trosselheim', 'lick', 'nieces', 'nodded', "didn't", 'auditions', 
'groaned', 'heckled', 'Very', 'ballet', 'accomplices', 'bodybuilder', 'gossiped', 'pivot', 'Tour', 'esteemed', 'quad', 'niece', 
'orientation', 'knives', 'aunts', 'Wally', 'accomplice', 'bin', 'disliked', 'Aspen', 'flour', 'charmed', 'fireman', 'distrusted', 
'billionaires', 'clove', 'guitarist', 'underway', 'neuroscience', 'tutor', 'newfound', 'superstar', 'persuasive', 'cakes', 
'footprint', 'sundials', 'wellknown', 'excusing', 'potluck', 'skunk', 'talented', 'lamb', 'nephews', 'princes', 'oven', 
'bus-size', 'canister', 'sympathized', 'florist', 'surgeons', 'Tamara', 'dagger', 'Franny', 'noxious', 'ten', 'backdraft', 
'separates', 'outmaneuvered', 'photographed', 'Gaga', 'Voltaire', 'YouTube', 'zookeepers', 'mechanic', 'boatman', 'Noreen', 
'bodybuilders', 'coop', 'Hybrid', 'ingenuity', 'cameraman', 'baker', 'villain', 'jockey', 'Ursula', 'PGA', 'chatted', 'righteous', 
'scent', 'rancher', 'janitors', 'millionaire', 'apprentice', 'uncles', 'henchman', 'Steam', 'pirate', 'henchmen', 'florists', 
'sidekick', 'grimy', 'sailor', 'Mona', 'contestant', 'sixty', 'tranquil', 'snowboard', 'ER', 'skimmed', 'bakes', 'sidekicks', 
'bodyguards', 'advisors', 'nuns', 'Proper', 'weathe', 'Vineyard', 'reviewer', 'rehearsing', 'CEOs', 'bodyguard', 'whistle', 
'lemonade', 'Owls'}
"""

# check if all tags in berkeley_neural_parser_result columns are tags (X_0) in sm5.grammar to compile a list of tags


def extract_pos_tags(parse_str):
    """
    Extracts all POS and phrase-level tags from a Penn Treebank-style parse string,
    ignoring actual words.

    Handles tags like:
      - Standard POS tags: NN, NNS, VBZ, PRP$, WP$, etc.
      - Phrase tags: NP, VP, SBAR, etc.
      - Variants with suffixes: NNP-SBJ, VBZ=H, etc.
      - Punctuation tags: ',', '.', ':', etc.
    """

    # Match tags that come right after '(' and consist of:
    # uppercase/lowercase letters, digits, punctuation used in PTB tags ($ - = + , . : ;)
    tag_pattern = r'\(([A-Za-z0-9$.,:;\'`+=-]+)\b(?!\s*\))'

    return re.findall(tag_pattern, parse_str)


sap_tags = set()
for parse in sap_parsed_sents:
    tags = extract_pos_tags(parse)
    sap_tags = sap_tags | set(tags)

print(f"the number of sap tags is {len(sap_tags)}/n")
print(f"the sap tags are {sap_tags}")
"""
the number of sap tags is 45/n
the sap tags are {'JJS', 'NNPS', 'PDT', 'WP', 'SQ', 'NNP', 'CD', 'VBN', 'TO', 'PRP', 'PP', 'WHADVP', 'VBD', 'EX', 'FW', 'NNS', 'S', 
'POS', 'ADVP', 'WDT', 'RP', 'CONJP', 'WRB', 'JJR', 'NP', 'NN', 'PRT', 'ADJP', 'RBR', 'WHNP', 'SBAR', 'VBG', 'DT', 'JJ', 'SINV', 
'MD', 'RB', 'CC', 'VBP', 'IN', 'UCP', 'VBZ', 'VP', 'VB', 'QP'}
ALL sap tags are present in sm5.grammar
"""
# get the tags of the sap_vocab
sap_vocab_tag_dict = {}

# get the word-tag-dict of sm5


def get_sm5_word_tag_dict(filepath):
    """
    Reads a text file where each line is a word and returns a list of words.

    Args:
        filepath (str): The path to the text file.

    Returns:
        list: A list of strings, where each string is a word from the file.
    """
    word_tag_dict = {}
    with open(filepath, 'r') as file:
        for line in file:
            # .strip() removes leading/trailing whitespace, including newline characters
            tag_word = line.strip().split()[:2]
            tag = tag_word[0]
            word = tag_word[1]
            if word in word_tag_dict:
                word_tag_dict[word].append(tag)
            else:
                word_tag_dict[word] = [tag]
    return word_tag_dict


sm5_word_tag_dict = get_sm5_word_tag_dict(
    'trained_berkeley_parser_sm5/berkeley_parser_sm5.lexicon')

for word in sap_words:
    if word in sm5_word_tag_dict:
        sap_vocab_tag_dict[word] = sm5_word_tag_dict[word]

most_sap_vocab_tags = []
for value_list in sap_vocab_tag_dict.values():
    most_sap_vocab_tags.extend(value_list)
most_sap_vocab_tags = set(most_sap_vocab_tags)

print(f"size of most_sap_vocab_tags is {len(most_sap_vocab_tags)}")
print(f"most_sap_vocab_tags content is {most_sap_vocab_tags}")
"""
size of most_sap_vocab_tags is 36
most_sap_vocab_tags content is {'WP', 'NNP', 'SYM', 'PRP', 'MD', 'VBG', 'POS', 'VB', 'RP', 'NNPS', 'IN', 'CD', 'UH', 
'LS', '.', 'WRB', 'TO', 'VBZ', 'VBD', 'VBN', 'JJ', 'PRP$', 'JJS', 'RB', ',', 'NN', 'FW', 'NNS', 'PDT', 'WDT', 'VBP', 
'RBR', 'CC', 'EX', 'DT', 'JJR'}
"""


diff_sap_parse_tag = sap_tags ^ most_sap_vocab_tags
print(f"diff_sap_parse_tag is {diff_sap_parse_tag}")

union_sap_parse_tag = sap_tags | most_sap_vocab_tags
print(f"union_sap_parse_tag is {union_sap_parse_tag}")

"""
diff_sap_parse_tag is {'SBAR', ',', 'NP', 'SQ', 'PRT', 'WHADVP', 'SYM', 'PP', 'UCP', 'ADVP', 'LS', 'S', 'PRP$', 'CONJP', 'WHNP', 
'ADJP', 'SINV', 'VP', 'QP', 'UH', '.'}
union_sap_parse_tag is {'PRP', 'PDT', 'JJR', 'JJ', 'FW', 'VBP', 'NP', ',', 'WDT', 'WHADVP', 'SYM', 'PP', 'WRB', 'VBN', 'NNPS', 
'IN', 'LS', 'NNP', 'PRP$', 'CD', 'MD', 'ADJP', 'JJS', 'TO', 'VB', 'QP', 'UH', 'SQ', 'SBAR', 'NNS', 'VBG', 'POS', 'PRT', 'CC', 
'DT', 'UCP', 'ADVP', 'RBR', 'S', 'CONJP', 'WHNP', 'RB', 'SINV', 'EX', 'VBD', 'NN', 'VP', 'RP', 'WP', '.', 'VBZ'}
"""
# use the list of tags to filter sm5.grammar
# step 1: get sm5 grammar


def get_sm5_grammar(filepath):
    """

    Args:
        filepath (str): The path to the text file.

    Returns:
        list: A list of tuple, each tuple is (probability, mother node, daughter nodes ).
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
            # skip the unary rule
            if len(parts[1].split()) < 3:
                continue
            prob = parts[1].split()[2]
            rhs = parts[1][:-(len(prob)+1)].split()
            rules.append((float(prob), lhs, rhs[0], rhs[1]))
    return rules


sm5_rules = get_sm5_grammar(
    'trained_berkeley_parser_sm5/berkeley_parser_sm5.grammar')
print(f'sm5_rules[0] is {sm5_rules[:10]}')
# step 2: use sap words tag to find all possible lhs for them
# lhs_for_bottom_layer = []
# print_bool = True
# for rule in sm5_rules:
#     rhs = rule[2].split()
#     daughter_1 = rhs[0].split('_')[0]
#     daughter_2 = rhs[1].split('_')[0]
#     if print_bool:
#         print(f"daughter1 is {daughter_1}")
#         print(f"daughter2 is {daughter_2}")
#         print_bool = False
#     keep = 0
#     for word_tag in list(most_sap_vocab_tags):
#         if word_tag == daughter_1:
#             keep += 1
#         if word_tag == daughter_2:
#             keep += 1
#     if keep == 2:
#         lhs_for_bottom_layer.append(rule[1])
#     # else:
#     #     print(f"current rule is {rule}")
# print(f'lhs_for_bottom_layer length is {len(lhs_for_bottom_layer)}')
# """
# lhs_for_bottom_layer length is 209075 --> too big
# """
# print(f'lhs_for_bottom_layer is {lhs_for_bottom_layer}')


def strip_subscript(symbol):
    """
    Remove subscript from Berkeley Parser symbol.

    Examples:
        S_1 -> S
        SBAR_3 -> SBAR
        VP_5 -> VP
    """
    return re.sub(r'_\d+$', '', symbol)


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
    kept_rules = set()

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
        prev_kept_rule_num = len(kept_rules)
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
                    kept_rules.add((prob, mother, d1, d2))
                else:
                    # Mother already exists but might be from this layer
                    kept_rules.add((prob, mother, d1, d2))
        increased_num_kept_rules = len(kept_rules) - prev_kept_rule_num
        prev_kept_rule_num = len(kept_rules)
        print(f"  New mother nodes: {len(new_mothers)}")
        print(f"  Rules added: {increased_num_kept_rules}")

        if new_mothers:
            sample = sorted(new_mothers)[:10]
            print(f"  Sample mothers: {sample}")

        # Track layer info
        layer_info['layers'].append({
            'layer': layer_num,
            'symbols': len(new_mothers),
            'rules added for this layer': increased_num_kept_rules,
            'new_mothers': sorted(new_mothers)
        })

        # Check if we've reached the root
        # root_base = strip_subscript(root)
        # if root_base in new_mothers:
        #     print(f"\n  ✓ Reached root '{root}'!")
        #     break

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


kept_rules, layer_info = build_grammar_layers(
    sm5_rules, most_sap_vocab_tags, root='S')


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


def normalize_without_collapse(rules, min_prob=1e-10, cumulative_threshold=0.95):
    """
    Normalize probabilities WITHOUT collapsing subscripts.

    Keeps Berkeley Parser's latent subcategorization (S_1, S_2, etc.)
    but still normalizes and applies threshold filtering.

    Args:
        rules: List of (prob, mother, daughter1, daughter2) tuples
        min_prob: Minimum probability threshold
        cumulative_threshold: Keep top rules until this cumulative probability

    Returns:
        Normalized grammar string in GSC format (with subscripts preserved)
    """
    print(f"\nNormalizing without collapsing subscripts...")
    print(f"  Keeping Berkeley Parser latent subcategorization")

    # Group rules by exact mother (including subscript)
    # This preserves S_1, S_2, etc. as separate categories
    lhs_groups = defaultdict(list)
    for prob, mother, d1, d2 in rules:
        lhs_groups[mother].append((prob, d1, d2))

    print(f"  LHS categories (with subscripts): {len(lhs_groups)}")

    # Normalize each LHS group
    normalized_rules = {}
    for lhs, rule_list in lhs_groups.items():
        total = sum(p for p, _, _ in rule_list)
        for prob, d1, d2 in rule_list:
            normalized_prob = prob / total if total > 0 else 0.0
            rule = f"{lhs} -> {d1} {d2}"
            normalized_rules[rule] = normalized_prob

    print(f"  Normalized rules: {len(normalized_rules)}")

    # Apply cumulative probability threshold per LHS
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
    print(f"  Final LHS categories: {len(lhs_groups_final)}")

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


grammar_str = normalize_without_collapse(
    kept_rules,
    min_prob=1e-10,
    cumulative_threshold=0.90
)  # 11k rules

with open('filtered_sm5.grammar', 'w') as f:
    f.write(grammar_str)


collapsed_grammar_str = collapse_and_normalize(kept_rules,
                                               min_prob=1e-10,
                                               cumulative_threshold=0.95)
with open('collapsed_filtered_sm5.grammar', 'w') as f:
    f.write(collapsed_grammar_str)
# (optional) extract all rules used in berkeley_neural_parser_result
# (optional) check if every rule used is in berkeley_parser_sm5.grammar
# (optional) save out-of-grammar rule into a list

max_length = float('-inf')
for sent in sap_sentences:
    sent_len = len(sent.split())
    if max_length < sent_len:
        max_length = sent_len
print(f"max_sent_length of sap is {max_length}")  # 24
