"""Probe the trained Relative Clause model.

Target pair (one cross-type comparison: subject-RC vs object-RC):
    A (subj-RC): 'DT NN NN WP VBD DT NNS VBD IN DT NN'  (d=6, len=11)
    B (obj-RC):  'DT NN NN WP DT NNS VBD VBD IN DT NN'  (d=5, len=11)
spillover = 2.
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


SET_NAME = "relative_clause"
DEFAULT_MODEL_PATH = os.path.join(
    "SAP_analysis", "relative_clause_model.pkl")


def make_target_pairs() -> list[TargetPair]:
    return [
        TargetPair(
            name="subj_rc_vs_obj_rc",
            sentence_a=pos_tokens("DT NN NN WP VBD DT NNS VBD IN DT NN"),
            sentence_b=pos_tokens("DT NN NN WP DT NNS VBD VBD IN DT NN"),
            d_a=6, d_b=5, spillover=2,
        ),
    ]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-path", default=DEFAULT_MODEL_PATH)
    parser.add_argument("--n-control", type=int, default=100)
    parser.add_argument("--control-source", default="corpus",
                        choices=["corpus", "random_pos"])
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
