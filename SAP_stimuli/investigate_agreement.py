"""
Investigate how the Berkeley parser sm5 grammar captures verb-noun
number agreement through its latent annotations.

Key questions:
  1. Which VP subscripts prefer singular verbs (VBZ) vs plural verbs (VBP)?
  2. Which NP subscripts prefer singular nouns (NN) vs plural nouns (NNS)?
  3. Do S -> NP_j VP_k rules pair singular NPs with singular VPs?

Usage:
    python SAP_stimuli/investigate_agreement.py
"""

import re
import os
from collections import defaultdict

GRAMMAR_PATH = os.path.join(
    'trained_berkeley_parser_sm5', 'berkeley_parser_sm5.grammar')

SUBSCRIPT_RE = re.compile(r'^(.+)_(\d+)$')


def parse_symbol(sym):
    m = SUBSCRIPT_RE.match(sym)
    if m:
        return m.group(1), int(m.group(2))
    return sym, None


def main():
    print("Scanning Berkeley grammar (streaming)...")

    verb_tags = {'VBZ', 'VBP', 'VBD', 'VBN', 'VBG', 'VB'}
    noun_tags = {'NN', 'NNS', 'NNP', 'NNPS'}

    # Accumulators
    vp_verb_mass = defaultdict(lambda: defaultdict(float))
    vp_total_mass = defaultdict(float)
    np_noun_mass = defaultdict(lambda: defaultdict(float))
    np_total_mass = defaultdict(float)
    s_np_vp_rules = []  # (np_sub, vp_sub, prob) for S -> NP VP
    vp_collapsed = defaultdict(lambda: defaultdict(float))  # rhs_pattern -> verb_tag -> mass

    with open(GRAMMAR_PATH) as f:
        for line in f:
            line = line.strip()
            if not line or '->' not in line:
                continue
            parts = line.split()
            if len(parts) not in (4, 5):
                continue

            lhs = parts[0]
            prob = float(parts[-1])
            rhs = parts[2:-1]

            lhs_base, lhs_sub = parse_symbol(lhs)
            rhs_parsed = [parse_symbol(r) for r in rhs]

            # VP analysis
            if lhs_base == 'VP' and lhs_sub is not None:
                vp_total_mass[lhs_sub] += prob
                head_base = rhs_parsed[0][0]
                if head_base in verb_tags:
                    vp_verb_mass[lhs_sub][head_base] += prob
                    # Collapsed VP rules by verb type (VBZ vs VBP only)
                    if head_base in ('VBZ', 'VBP'):
                        rhs_tail = ' '.join(b for b, _ in rhs_parsed[1:]) if len(rhs_parsed) > 1 else '(intrans)'
                        vp_collapsed[rhs_tail][head_base] += prob

            # NP analysis
            if lhs_base == 'NP' and lhs_sub is not None:
                np_total_mass[lhs_sub] += prob
                for r_base, _ in rhs_parsed:
                    if r_base in noun_tags:
                        np_noun_mass[lhs_sub][r_base] += prob

            # S -> NP VP pairing
            if lhs_base == 'S' and len(rhs) == 2:
                r0_base, r0_sub = rhs_parsed[0]
                r1_base, r1_sub = rhs_parsed[1]
                if r0_base == 'NP' and r1_base == 'VP' and r0_sub is not None and r1_sub is not None:
                    s_np_vp_rules.append((r0_sub, r1_sub, prob))

    print("  Done.\n")

    # ==================================================================
    # 1. VP subscript profiles
    # ==================================================================
    print("=" * 70)
    print("1. VP SUBSCRIPT PROFILES: VBZ (singular) vs VBP (plural)")
    print("=" * 70)
    print()

    vp_subs = sorted(vp_verb_mass.keys())
    print(f"{'VP_i':>6}  {'VBZ':>8}  {'VBP':>8}  {'VBD':>8}  "
          f"{'VBN':>8}  {'VB':>8}  preference")
    print("-" * 70)

    vp_singular = set()
    vp_plural = set()
    for sub in vp_subs:
        total = vp_total_mass[sub]
        fracs = {v: vp_verb_mass[sub][v] / total if total > 0 else 0
                 for v in verb_tags}
        vbz = fracs['VBZ']
        vbp = fracs['VBP']
        if vbz > 0.1 or vbp > 0.1:
            if vbz > vbp * 2:
                pref = "SINGULAR"
                vp_singular.add(sub)
            elif vbp > vbz * 2:
                pref = "PLURAL"
                vp_plural.add(sub)
            else:
                pref = "mixed"
        elif fracs['VBD'] > 0.3:
            pref = "past"
        else:
            pref = ""
        print(f"VP_{sub:>2}  {vbz:8.4f}  {vbp:8.4f}  "
              f"{fracs['VBD']:8.4f}  {fracs['VBN']:8.4f}  "
              f"{fracs['VB']:8.4f}  {pref}")

    print(f"\nSingular-preferring VP subs (VBZ >> VBP): {sorted(vp_singular)}")
    print(f"Plural-preferring VP subs   (VBP >> VBZ): {sorted(vp_plural)}")

    # ==================================================================
    # 2. NP subscript profiles
    # ==================================================================
    print()
    print("=" * 70)
    print("2. NP SUBSCRIPT PROFILES: NN (singular) vs NNS (plural)")
    print("=" * 70)
    print()

    np_subs = sorted(np_noun_mass.keys())
    print(f"{'NP_i':>6}  {'NN':>8}  {'NNS':>8}  {'NNP':>8}  preference")
    print("-" * 50)

    np_singular = set()
    np_plural = set()
    for sub in np_subs:
        total = np_total_mass[sub]
        fracs = {n: np_noun_mass[sub][n] / total if total > 0 else 0
                 for n in noun_tags}
        nn = fracs['NN']
        nns = fracs['NNS']
        if nn > 0.05 or nns > 0.05:
            if nn > nns * 2:
                pref = "SINGULAR"
                np_singular.add(sub)
            elif nns > nn * 2:
                pref = "PLURAL"
                np_plural.add(sub)
            else:
                pref = "mixed"
        else:
            pref = ""
        print(f"NP_{sub:>2}  {nn:8.4f}  {nns:8.4f}  "
              f"{fracs['NNP']:8.4f}  {pref}")

    print(f"\nSingular-preferring NP subs (NN >> NNS): {sorted(np_singular)}")
    print(f"Plural-preferring NP subs   (NNS >> NN): {sorted(np_plural)}")

    # ==================================================================
    # 3. S -> NP VP agreement pairing
    # ==================================================================
    print()
    print("=" * 70)
    print("3. S -> NP_j VP_k AGREEMENT PAIRING")
    print("=" * 70)
    print()

    pair_mass = defaultdict(float)
    for np_sub, vp_sub, prob in s_np_vp_rules:
        np_num = ('SG' if np_sub in np_singular
                  else 'PL' if np_sub in np_plural else '??')
        vp_num = ('SG' if vp_sub in vp_singular
                  else 'PL' if vp_sub in vp_plural else '??')
        pair_mass[(np_num, vp_num)] += prob

    total_pair = sum(pair_mass.values())
    hdr = 'NP_num \\ VP_num'
    print(f"{hdr:>18}  {'SG':>10}  {'PL':>10}  {'??':>10}  {'row sum':>10}")
    print("-" * 65)
    for np_num in ['SG', 'PL', '??']:
        row = {vp: pair_mass[(np_num, vp)] for vp in ['SG', 'PL', '??']}
        row_total = sum(row.values())
        print(f"{np_num:>18}  {row['SG']:10.6f}  {row['PL']:10.6f}  "
              f"{row['??']:10.6f}  {row_total:10.6f}")
    print(f"{'col sum':>18}  "
          f"{sum(pair_mass[(n,'SG')] for n in ['SG','PL','??']):10.6f}  "
          f"{sum(pair_mass[(n,'PL')] for n in ['SG','PL','??']):10.6f}  "
          f"{sum(pair_mass[(n,'??')] for n in ['SG','PL','??']):10.6f}  "
          f"{total_pair:10.6f}")

    if total_pair > 0:
        agree = pair_mass[('SG', 'SG')] + pair_mass[('PL', 'PL')]
        disagree = pair_mass[('SG', 'PL')] + pair_mass[('PL', 'SG')]
        unclass = total_pair - agree - disagree
        print(f"\nAgreement (SG+SG, PL+PL):    {agree:.6f} "
              f"({100*agree/total_pair:.1f}%)")
        print(f"Disagreement (SG+PL, PL+SG): {disagree:.6f} "
              f"({100*disagree/total_pair:.1f}%)")
        print(f"Unclassified:                {unclass:.6f} "
              f"({100*unclass/total_pair:.1f}%)")

    # ==================================================================
    # 4. Collapsed VP rules by verb type
    # ==================================================================
    print()
    print("=" * 70)
    print("4. COLLAPSED VP RULES: VBZ vs VBP by RHS structure")
    print("=" * 70)
    print()

    print(f"{'VP -> V + ...':>25}  {'VBZ mass':>10}  {'VBP mass':>10}  {'VBZ/VBP':>8}")
    print("-" * 60)
    for rhs_pat in sorted(vp_collapsed.keys()):
        vbz = vp_collapsed[rhs_pat].get('VBZ', 0)
        vbp = vp_collapsed[rhs_pat].get('VBP', 0)
        ratio = vbz / vbp if vbp > 0 else float('inf')
        label = f"V {rhs_pat}" if rhs_pat != '(intrans)' else "V (intrans)"
        print(f"{label:>25}  {vbz:10.6f}  {vbp:10.6f}  {ratio:8.2f}")

    total_vbz = sum(v.get('VBZ', 0) for v in vp_collapsed.values())
    total_vbp = sum(v.get('VBP', 0) for v in vp_collapsed.values())
    print(f"{'TOTAL':>25}  {total_vbz:10.6f}  {total_vbp:10.6f}  "
          f"{total_vbz/total_vbp if total_vbp > 0 else float('inf'):8.2f}")

    # ==================================================================
    # 5. Suggestion
    # ==================================================================
    print()
    print("=" * 70)
    print("5. SUGGESTED MINI GRAMMAR FOR AGREEMENT")
    print("=" * 70)
    print()
    print("To model agreement success/failure, add VBP (plural verb) rules:")
    print()
    print("  NP -> DT NN        (singular)")
    print("  NP -> DT NNS       (plural)")
    print("  NP -> JJ NN        (singular with adjective)")
    print("  VP -> VBZ NP       (singular transitive)")
    print("  VP -> VBZ           (singular intransitive)")
    print("  VP -> VBP NP       (plural transitive)")
    print("  VP -> VBP           (plural intransitive)")
    print("  S -> NP VP")
    print("  SBAR -> IN S")
    print("  SBAR -> SBAR ,")
    print("  S -> SBAR S")


if __name__ == '__main__':
    main()
