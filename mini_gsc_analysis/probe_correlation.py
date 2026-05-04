"""Optional complementary correlation analysis.

Tests Hypothesis 2: across the control pool, does the joint divergence
``J(A, B)`` correlate with the per-sentence-difficulty gap
``|D_total(A) - D_total(B)|``? And does the target pair sit on the regression
line implied by the controls, or is it an outlier?

Inputs: each set's ``<set>_results.json`` (uses the raw per-pair table).
Outputs:
    * ``mini_gsc_analysis/results/correlation_summary.json`` -- per
      ``(set, J_metric, D_metric)`` Pearson and Spearman correlations across
      controls, and target-vs-fitted-line residuals.
    * ``mini_gsc_analysis/figs/correlation/scatter_<set>_<J>_<D>.png`` --
      scatter plots with target pairs highlighted.

Run after the per-set probes finish:
    python -m mini_gsc_analysis.probe_correlation
"""

from __future__ import annotations

import argparse
import glob
import os
import sys

import numpy as np

THIS_DIR = os.path.abspath(os.path.dirname(__file__))
REPO_ROOT = os.path.abspath(os.path.join(THIS_DIR, os.pardir))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from mini_gsc_analysis.gsc_inference_utils import (  # noqa: E402
    D_METRICS, J_METRICS, read_json, write_json,
)


def _pearson(x: np.ndarray, y: np.ndarray) -> tuple[float, int]:
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    mask = ~(np.isnan(x) | np.isnan(y))
    x = x[mask]; y = y[mask]
    n = int(x.size)
    if n < 3:
        return float("nan"), n
    x = x - x.mean()
    y = y - y.mean()
    denom = float(np.sqrt(np.sum(x * x)) * np.sqrt(np.sum(y * y)))
    if denom <= 0:
        return float("nan"), n
    return float(np.sum(x * y) / denom), n


def _rank(a: np.ndarray) -> np.ndarray:
    a = np.asarray(a, dtype=float)
    order = np.argsort(a, kind="mergesort")
    ranks = np.empty_like(order, dtype=float)
    ranks[order] = np.arange(1, len(a) + 1)
    # Average ranks for ties.
    s = np.sort(a)
    i = 0
    while i < len(a):
        j = i
        while j + 1 < len(a) and s[j + 1] == s[i]:
            j += 1
        if j > i:
            avg = 0.5 * (i + j) + 1
            ranks[np.argsort(a, kind="mergesort")[i:j + 1]] = avg
        i = j + 1
    return ranks


def _spearman(x: np.ndarray, y: np.ndarray) -> tuple[float, int]:
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    mask = ~(np.isnan(x) | np.isnan(y))
    x = x[mask]; y = y[mask]
    n = int(x.size)
    if n < 3:
        return float("nan"), n
    return _pearson(_rank(x), _rank(y))[0], n


def _per_set_correlation(results: dict, fig_dir: str) -> dict:
    set_name = results["set_name"]
    targets = results.get("raw_target_records", [])
    controls = results.get("raw_control_records", [])
    out = {"set_name": set_name, "by_metric_pair": {}}

    target_d_gaps = {}
    target_j_vals = {}
    ctrl_d_gaps = {}
    ctrl_j_vals = {}
    for r in targets + controls:
        for d_name in D_METRICS:
            gap = abs(r["D_total"][d_name + "_a"] - r["D_total"][d_name + "_b"])
            tgt = r.get("is_target", False)
            d = target_d_gaps if tgt else ctrl_d_gaps
            d.setdefault(d_name, []).append(gap)
        for j_name in J_METRICS:
            v = r["J"][j_name]
            tgt = r.get("is_target", False)
            d = target_j_vals if tgt else ctrl_j_vals
            d.setdefault(j_name, []).append(v)

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    os.makedirs(fig_dir, exist_ok=True)

    for j_name in J_METRICS:
        for d_name in D_METRICS:
            ctrl_x = np.asarray(ctrl_j_vals.get(j_name, []), dtype=float)
            ctrl_y = np.asarray(ctrl_d_gaps.get(d_name, []), dtype=float)
            tgt_x = np.asarray(target_j_vals.get(j_name, []), dtype=float)
            tgt_y = np.asarray(target_d_gaps.get(d_name, []), dtype=float)
            r_pearson, n_p = _pearson(ctrl_x, ctrl_y)
            r_spearman, n_s = _spearman(ctrl_x, ctrl_y)
            entry = {
                "pearson_r": r_pearson, "pearson_n": n_p,
                "spearman_r": r_spearman, "spearman_n": n_s,
                "control_x": ctrl_x.tolist(),
                "control_y": ctrl_y.tolist(),
                "target_x": tgt_x.tolist(),
                "target_y": tgt_y.tolist(),
            }
            # Linear regression for residuals (least-squares, controls only).
            if ctrl_x.size >= 3 and np.std(ctrl_x) > 0:
                slope, intercept = np.polyfit(ctrl_x, ctrl_y, 1)
                pred = slope * tgt_x + intercept if tgt_x.size else np.array([])
                resid = (tgt_y - pred) if tgt_y.size else np.array([])
                resid_sd = float(np.std(ctrl_y - (slope * ctrl_x + intercept),
                                        ddof=1)) if ctrl_x.size > 2 else 0.0
                entry["fit_slope"] = float(slope)
                entry["fit_intercept"] = float(intercept)
                entry["target_residuals"] = resid.tolist()
                entry["target_residual_z"] = (
                    (resid / resid_sd).tolist() if resid_sd > 0 else
                    [float("nan")] * resid.size)
            out["by_metric_pair"][f"{j_name}__{d_name}"] = entry

            # Plot
            fig, ax = plt.subplots(figsize=(6, 5))
            ax.scatter(ctrl_x, ctrl_y, color="#888", alpha=0.6,
                       label=f"control (n={ctrl_x.size})")
            if tgt_x.size:
                ax.scatter(tgt_x, tgt_y, color="red", s=60, marker="X",
                           label=f"target (n={tgt_x.size})")
            if "fit_slope" in entry:
                xs = np.linspace(np.nanmin(ctrl_x), np.nanmax(ctrl_x), 100)
                ax.plot(xs, entry["fit_slope"] * xs + entry["fit_intercept"],
                        "--", color="#444",
                        label=(f"OLS r={r_pearson:.2f} "
                               f"rho={r_spearman:.2f}"))
            ax.set_xlabel(j_name)
            ax.set_ylabel(f"|{d_name}(A) - {d_name}(B)|")
            ax.set_title(f"{set_name}: {j_name} vs |{d_name} gap|")
            ax.legend(fontsize=8)
            fig.tight_layout()
            fig.savefig(os.path.join(
                fig_dir, f"scatter_{set_name}_{j_name}_{d_name}.png"),
                dpi=150)
            plt.close(fig)
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--results-dir", default=os.path.join(THIS_DIR, "results"))
    parser.add_argument(
        "--fig-dir", default=os.path.join(THIS_DIR, "figs", "correlation"))
    parser.add_argument(
        "--include", default=None, nargs="*",
        help="restrict to specific sets")
    args = parser.parse_args()

    pattern = os.path.join(args.results_dir, "*_results.json")
    files = sorted(glob.glob(pattern))
    if args.include:
        keep = set(args.include)
        files = [f for f in files
                 if os.path.basename(f).split("_results.json")[0] in keep]
    if not files:
        print(f"no result files found at {pattern}")
        return

    os.makedirs(args.fig_dir, exist_ok=True)

    summary = {"per_set": {}}
    for f in files:
        try:
            data = read_json(f)
            entry = _per_set_correlation(data, args.fig_dir)
            summary["per_set"][entry["set_name"]] = entry
            print(f"processed {f}")
        except Exception as exc:
            print(f"could not process {f}: {exc}")

    out_path = os.path.join(args.results_dir, "correlation_summary.json")
    write_json(out_path, summary)
    print(f"wrote {out_path}")
    print(f"figures in {args.fig_dir}")


if __name__ == "__main__":
    main()
