"""Probe the trained Agreement model.

Target pair (one):
    A: 'DT NN VBZ DT NN'   (d=3)
    B: 'DT NNS VBZ DT NN'  (d=3, ungrammatical agreement)
spillover = 2.

Tests Hypothesis 1 (Designs A and B) and reports joint divergence J.

Run on the cluster after the model has been trained:
    python -m mini_gsc_analysis.probe_agreement
"""

from __future__ import annotations

import argparse
import os
import sys

THIS_DIR = os.path.abspath(os.path.dirname(__file__))
REPO_ROOT = os.path.abspath(os.path.join(THIS_DIR, os.pardir))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from mini_gsc_analysis.gsc_inference_utils import (
    TargetPair, pos_tokens, render_from_json, run_per_set_analysis,
)


SET_NAME = "agreement"
DEFAULT_MODEL_PATH = os.path.join("../SAP_analysis", "agreement_model.pkl")


def make_target_pairs() -> list[TargetPair]:
    return [
        TargetPair(
            name="agreement_sg_vs_pl_subject",
            sentence_a=pos_tokens("DT NN VBZ DT NN"),
            sentence_b=pos_tokens("DT NNS VBZ DT NN"),
            d_a=3, d_b=3, spillover=2,
        ),
    ]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-path", default=DEFAULT_MODEL_PATH,
                        help="path to the trained .pkl checkpoint")
    parser.add_argument("--n-control", type=int, default=100,
                        help="number of control pairs to sample")
    parser.add_argument("--control-source", default="corpus",
                        choices=["corpus", "random_pos"],
                        help="control sentence source")
    parser.add_argument("--control-seed", type=int, default=0)
    parser.add_argument("--run-seed", type=int, default=1024)
    parser.add_argument("--no-plots", action="store_true")
    parser.add_argument(
        "--render-only", metavar="JSON_PATH", default=None,
        help=("Skip simulation and re-render figures from an existing "
              f"results JSON (typically results/{SET_NAME}_results.json)."))
    parser.add_argument(
        "--fig-dir", default=None,
        help="Override directory for rendered figures.")
    args = parser.parse_args()

    if args.render_only:
        render_from_json(args.render_only, fig_dir=args.fig_dir,
                         set_name=SET_NAME)
        print(f"[{SET_NAME}] re-rendered from {args.render_only}")
        return

    run_per_set_analysis(
        set_name=SET_NAME,
        model_path=args.model_path,
        target_pairs=make_target_pairs(),
        n_control=args.n_control,
        control_source=args.control_source,
        control_seed=args.control_seed,
        run_seed=args.run_seed,
        fig_dir=args.fig_dir,
        make_plots=not args.no_plots,
    )


if __name__ == "__main__":
    main()
