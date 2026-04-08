"""
Compute a number-split mini grammar for agreement from the Berkeley grammar.

Uses soft number assignment: each NP subscript gets a singular weight based
on how much of its probability mass goes to NN vs NNS rules, and each VP
subscript gets a singular weight based on VBZ vs VBP rules. These weights
are then used to split S -> NP VP into four agreement variants.

Output grammar symbols:
    NP_sg, NP_pl  — number-marked subject/object NPs
    VP_sg, VP_pl  — number-marked VPs (VBZ vs VBP)
    S, SBAR       — unchanged

Usage:
    python SAP_stimuli/compute_agreement_grammar.py
"""

import re
import os
from collections import defaultdict

GRAMMAR_PATH = os.path.join(
    'trained_berkeley_parser_sm5', 'berkeley_parser_sm5.grammar')
OUTPUT_RULES = os.path.join('SAP_stimuli', 'Agreement.txt')
OUTPUT_PROBS = os.path.join('SAP_stimuli', 'Agreement_probs.txt')

SUBSCRIPT_RE = re.compile(r'^(.+)_(\d+)$')
AT_RE = re.compile(r'^@')


def parse_sym(sym):
    m = SUBSCRIPT_RE.match(sym)
    if m:
        return m.group(1), int(m.group(2))
    return sym, None


def base(sym):
    """Strip subscript then strip @ prefix."""
    b, _ = parse_sym(sym)
    return AT_RE.sub('', b)


def main():
    print("Pass 1: scanning Berkeley grammar...\n")

    # --- accumulators ---
    # NP number: for each NP subscript, mass going to NN vs NNS
    np_nn_mass = defaultdict(float)    # NP_j -> total NN mass
    np_nns_mass = defaultdict(float)   # NP_j -> total NNS mass

    # VP number: for each VP subscript, mass going to VBZ vs VBP
    vp_vbz_mass = defaultdict(float)
    vp_vbp_mass = defaultdict(float)

    # S -> NP VP rules: (S_sub, NP_sub, VP_sub, prob)
    s_np_vp = []

    # VP -> VBZ/VBP NP rules: (VP_sub, NP_sub, prob)
    vp_vbz_np = []
    vp_vbp_np = []
    # VP -> VBZ/VBP (intransitive): (VP_sub, prob)
    vp_vbz_intrans = []
    vp_vbp_intrans = []

    # NP -> DT NN, JJ NN, DT NNS rules: (NP_sub, prob)
    np_dt_nn = []
    np_jj_nn = []
    np_dt_nns = []

    # SBAR rules (for @-merge passthrough)
    sbar_in_s_mass = 0.0
    sbar_sbar_comma_mass = 0.0
    s_sbar_s_mass = 0.0

    with open(GRAMMAR_PATH) as f:
        for line in f:
            line = line.strip()
            if not line or '->' not in line:
                continue
            parts = line.split()
            if len(parts) not in (4, 5):
                continue

            lhs_raw = parts[0]
            prob = float(parts[-1])
            rhs_raw = parts[2:-1]

            lhs_base = base(lhs_raw)
            _, lhs_sub = parse_sym(lhs_raw)
            rhs_bases = [base(r) for r in rhs_raw]
            rhs_subs = [parse_sym(r)[1] for r in rhs_raw]

            # NP -> ... rules containing NN or NNS
            if lhs_base == 'NP' and lhs_sub is not None:
                for rb in rhs_bases:
                    if rb == 'NN':
                        np_nn_mass[lhs_sub] += prob
                    elif rb == 'NNS':
                        np_nns_mass[lhs_sub] += prob

                # Specific rule patterns
                if len(rhs_bases) == 2:
                    if rhs_bases == ['DT', 'NN']:
                        np_dt_nn.append((lhs_sub, prob))
                    elif rhs_bases == ['JJ', 'NN']:
                        np_jj_nn.append((lhs_sub, prob))
                    elif rhs_bases == ['DT', 'NNS']:
                        np_dt_nns.append((lhs_sub, prob))

            # VP -> VBZ/VBP ... rules
            if lhs_base == 'VP' and lhs_sub is not None and len(rhs_bases) >= 1:
                if rhs_bases[0] == 'VBZ':
                    vp_vbz_mass[lhs_sub] += prob
                    if len(rhs_bases) == 2 and rhs_bases[1] == 'NP':
                        vp_vbz_np.append((lhs_sub, rhs_subs[1], prob))
                    elif len(rhs_bases) == 1:
                        vp_vbz_intrans.append((lhs_sub, prob))
                elif rhs_bases[0] == 'VBP':
                    vp_vbp_mass[lhs_sub] += prob
                    if len(rhs_bases) == 2 and rhs_bases[1] == 'NP':
                        vp_vbp_np.append((lhs_sub, rhs_subs[1], prob))
                    elif len(rhs_bases) == 1:
                        vp_vbp_intrans.append((lhs_sub, prob))

            # S -> NP VP rules
            if lhs_base == 'S' and len(rhs_bases) == 2:
                if rhs_bases == ['NP', 'VP']:
                    s_np_vp.append((lhs_sub, rhs_subs[0], rhs_subs[1], prob))
                elif rhs_bases == ['SBAR', 'S']:
                    s_sbar_s_mass += prob

            # SBAR rules
            if lhs_base == 'SBAR' and len(rhs_bases) == 2:
                if rhs_bases == ['IN', 'S']:
                    sbar_in_s_mass += prob
                elif rhs_bases == ['SBAR', ',']:
                    sbar_sbar_comma_mass += prob

    print("  Done scanning.\n")

    # --- Compute soft number weights ---
    all_np_subs = set(np_nn_mass.keys()) | set(np_nns_mass.keys())
    np_sg_weight = {}
    for j in all_np_subs:
        nn = np_nn_mass[j]
        nns = np_nns_mass[j]
        total = nn + nns
        np_sg_weight[j] = nn / total if total > 0 else 0.5

    all_vp_subs = set(vp_vbz_mass.keys()) | set(vp_vbp_mass.keys())
    vp_sg_weight = {}
    for k in all_vp_subs:
        vbz = vp_vbz_mass[k]
        vbp = vp_vbp_mass[k]
        total = vbz + vbp
        vp_sg_weight[k] = vbz / total if total > 0 else 0.5

    # --- Compute S rule probabilities ---
    s_sg_sg = 0.0
    s_sg_pl = 0.0
    s_pl_sg = 0.0
    s_pl_pl = 0.0
    s_np_vp_total = 0.0

    for s_sub, np_sub, vp_sub, prob in s_np_vp:
        sg_np = np_sg_weight.get(np_sub, 0.5)
        pl_np = 1.0 - sg_np
        sg_vp = vp_sg_weight.get(vp_sub, 0.5)
        pl_vp = 1.0 - sg_vp

        s_sg_sg += prob * sg_np * sg_vp
        s_sg_pl += prob * sg_np * pl_vp
        s_pl_sg += prob * pl_np * sg_vp
        s_pl_pl += prob * pl_np * pl_vp
        s_np_vp_total += prob

    print("=" * 60)
    print("S -> NP VP number pairing (raw mass):")
    print(f"  S -> NP_sg VP_sg: {s_sg_sg:.6f}")
    print(f"  S -> NP_sg VP_pl: {s_sg_pl:.6f}")
    print(f"  S -> NP_pl VP_sg: {s_pl_sg:.6f}")
    print(f"  S -> NP_pl VP_pl: {s_pl_pl:.6f}")
    agree = s_sg_sg + s_pl_pl
    disagree = s_sg_pl + s_pl_sg
    print(f"  Agreement:    {agree:.6f} ({100*agree/(agree+disagree):.1f}%)")
    print(f"  Disagreement: {disagree:.6f} ({100*disagree/(agree+disagree):.1f}%)")

    # --- Compute VP_sg rules ---
    vp_sg_vbz_np_sg = sum(prob * vp_sg_weight.get(vk, 0.5) * np_sg_weight.get(nk, 0.5)
                          for vk, nk, prob in vp_vbz_np)
    vp_sg_vbz_np_pl = sum(prob * vp_sg_weight.get(vk, 0.5) * (1 - np_sg_weight.get(nk, 0.5))
                          for vk, nk, prob in vp_vbz_np)
    vp_sg_vbz_intrans = sum(prob * vp_sg_weight.get(vk, 0.5)
                            for vk, prob in vp_vbz_intrans)

    # --- Compute VP_pl rules ---
    vp_pl_vbp_np_sg = sum(prob * (1 - vp_sg_weight.get(vk, 0.5)) * np_sg_weight.get(nk, 0.5)
                          for vk, nk, prob in vp_vbp_np)
    vp_pl_vbp_np_pl = sum(prob * (1 - vp_sg_weight.get(vk, 0.5)) * (1 - np_sg_weight.get(nk, 0.5))
                          for vk, nk, prob in vp_vbp_np)
    vp_pl_vbp_intrans = sum(prob * (1 - vp_sg_weight.get(vk, 0.5))
                            for vk, prob in vp_vbp_intrans)

    # --- Compute NP_sg rules ---
    np_sg_dt_nn = sum(prob * np_sg_weight.get(nk, 0.5) for nk, prob in np_dt_nn)
    np_sg_jj_nn = sum(prob * np_sg_weight.get(nk, 0.5) for nk, prob in np_jj_nn)

    # --- Compute NP_pl rules ---
    np_pl_dt_nns = sum(prob * (1 - np_sg_weight.get(nk, 0.5)) for nk, prob in np_dt_nns)

    # --- Normalize per LHS ---
    def normalize(rules_dict):
        total = sum(rules_dict.values())
        if total > 0:
            return {k: v / total for k, v in rules_dict.items()}
        return rules_dict

    s_rules = normalize({
        'S -> NP_sg VP_sg': s_sg_sg,
        'S -> NP_sg VP_pl': s_sg_pl,
        'S -> NP_pl VP_sg': s_pl_sg,
        'S -> NP_pl VP_pl': s_pl_pl,
        'S -> SBAR S': s_sbar_s_mass,
    })

    vp_sg_rules = normalize({
        'VP_sg -> VBZ NP_sg': vp_sg_vbz_np_sg,
        'VP_sg -> VBZ NP_pl': vp_sg_vbz_np_pl,
        'VP_sg -> VBZ': vp_sg_vbz_intrans,
    })

    vp_pl_rules = normalize({
        'VP_pl -> VBP NP_sg': vp_pl_vbp_np_sg,
        'VP_pl -> VBP NP_pl': vp_pl_vbp_np_pl,
        'VP_pl -> VBP': vp_pl_vbp_intrans,
    })

    np_sg_rules = normalize({
        'NP_sg -> DT NN': np_sg_dt_nn,
        'NP_sg -> JJ NN': np_sg_jj_nn,
    })

    np_pl_rules = normalize({
        'NP_pl -> DT NNS': np_pl_dt_nns,
    })

    sbar_total = sbar_in_s_mass + sbar_sbar_comma_mass
    sbar_rules = {
        'SBAR -> IN S': sbar_in_s_mass / sbar_total if sbar_total > 0 else 1.0,
        'SBAR -> SBAR ,': sbar_sbar_comma_mass / sbar_total if sbar_total > 0 else 0.0,
    }

    # --- Combine and output ---
    all_rules = {}
    all_rules.update(np_sg_rules)
    all_rules.update(np_pl_rules)
    all_rules.update(vp_sg_rules)
    all_rules.update(vp_pl_rules)
    all_rules.update(s_rules)
    all_rules.update(sbar_rules)

    # Print summary
    print()
    print("=" * 60)
    print("FINAL GRAMMAR WITH PROBABILITIES:")
    print("=" * 60)
    rule_lines = []
    plain_rules = []
    for rule, prob in all_rules.items():
        print(f"  {prob:.6f}  {rule}")
        rule_lines.append(f"{prob:.6f} {rule}")
        plain_rules.append(rule)

    # Normalization check
    print()
    print("Normalization check:")
    lhs_groups = defaultdict(float)
    for rule, prob in all_rules.items():
        lhs = rule.split(' -> ')[0]
        lhs_groups[lhs] += prob
    for lhs, total in sorted(lhs_groups.items()):
        ok = " (OK)" if abs(total - 1.0) < 1e-4 else f" *** {total:.6f} ***"
        print(f"  {lhs}: sum = {total:.6f}{ok}")

    # Write files
    with open(OUTPUT_RULES, 'w') as f:
        f.write('\n'.join(plain_rules) + '\n')
    print(f"\nSaved rules to {OUTPUT_RULES}")

    with open(OUTPUT_PROBS, 'w') as f:
        f.write('\n'.join(rule_lines) + '\n')
    print(f"Saved probs to {OUTPUT_PROBS}")


if __name__ == '__main__':
    main()
