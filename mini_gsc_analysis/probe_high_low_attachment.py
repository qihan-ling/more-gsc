"""Probe the trained High/Low Attachment model.

Three target sentences (all length 11, all d=10), tested as three pairwise
comparisons (X-Y, X-Z, Y-Z):
    X: 'NNP VBD IN DT NN IN DT NNS WP VBZ JJ'
    Y: 'NNP VBD IN DT NNS IN DT NN WP VBZ JJ'
    Z: 'NNP VBD IN DT NN IN DT NN WP VBZ JJ'
spillover = 1 (next 1 position only).
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


SET_NAME = "high_low_attachment"
DEFAULT_MODEL_PATH = os.path.join(
    "SAP_analysis", "high_low_attachment_model.pkl")


def make_target_pairs() -> list[TargetPair]:
    X = pos_tokens("NNP VBD IN DT NN IN DT NNS WP VBZ JJ")
    Y = pos_tokens("NNP VBD IN DT NNS IN DT NN WP VBZ JJ")
    Z = pos_tokens("NNP VBD IN DT NN IN DT NN WP VBZ JJ")
    sp = 1
    return [
        TargetPair(name="attach_X_vs_Y",
                   sentence_a=X, sentence_b=Y, d_a=10, d_b=10, spillover=sp),
        TargetPair(name="attach_X_vs_Z",
                   sentence_a=X, sentence_b=Z, d_a=10, d_b=10, spillover=sp),
        TargetPair(name="attach_Y_vs_Z",
                   sentence_a=Y, sentence_b=Z, d_a=10, d_b=10, spillover=sp),
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
