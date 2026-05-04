"""Probe the trained Classic-GP NP/Z model.

Target pair (one):
    A: 'IN DT NN VBD , DT NN VBD JJ NN'  (d=8, len=11; comma is a token)
    B: 'IN DT NN VBD DT NN VBD JJ NN'    (d=7, len=10)
spillover = 2.

NOTE: the comma token must be in the trained model's filler list. If your
trained NPZ checkpoint does not list ',', either retrain with the comma in
the lexicon or override the target pair via ``--alt-comma-token NULL``.
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


SET_NAME = "classicgp_npz"
DEFAULT_MODEL_PATH = os.path.join("SAP_analysis", "classicgp_npz_model.pkl")


def make_target_pairs(comma_token: str = ",") -> list[TargetPair]:
    return [
        TargetPair(
            name="npz_garden_path",
            sentence_a=pos_tokens(
                f"IN DT NN VBD {comma_token} DT NN VBD JJ NN"),
            sentence_b=pos_tokens("IN DT NN VBD DT NN VBD JJ NN"),
            d_a=8, d_b=7, spillover=2,
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
    parser.add_argument("--alt-comma-token", default=",",
                        help="filler used in place of the comma token")
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
        target_pairs=make_target_pairs(args.alt_comma_token),
        n_control=args.n_control,
        control_source=args.control_source,
        control_seed=args.control_seed,
        run_seed=args.run_seed,
        fig_dir=args.fig_dir,
        make_plots=not args.no_plots,
    )


if __name__ == "__main__":
    main()
