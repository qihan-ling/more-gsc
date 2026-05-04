"""Cross-set ranking of target-vs-control gaps.

Reads each ``mini_gsc_analysis/results/<set>_results.json`` produced by the
per-set probe scripts and produces:

* ``mini_gsc_analysis/results/cross_set_summary.json`` -- table of
  ``(set, metric, design, z_target, percentile, p_one_sided, cohen_d)``.
* ``mini_gsc_analysis/figs/cross_set/*.png`` -- bar charts and a
  ``set x metric`` heatmap of z-scores.

Run on the cluster after every per-set probe has finished:
    python -m mini_gsc_analysis.probe_cross_set
"""

from __future__ import annotations

import argparse
import glob
import os
import sys
from typing import Optional

import numpy as np

THIS_DIR = os.path.abspath(os.path.dirname(__file__))
REPO_ROOT = os.path.abspath(os.path.join(THIS_DIR, os.pardir))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from mini_gsc_analysis.gsc_inference_utils import (  # noqa: E402
    D_METRICS, J_METRICS, read_json, write_json,
)


DESIGN_LABELS = {
    "design_a": "A_whole_sequence",
    "design_b": "B_critical_region",
}


def _collect_rows(results: dict) -> list[dict]:
    set_name = results["set_name"]
    rows = []
    # Design A: D metrics
    for name in D_METRICS:
        block = results["design_a"]["D"].get(name, {})
        for stats in block.get("per_target", []):
            rows.append({
                "set": set_name,
                "design": DESIGN_LABELS["design_a"],
                "metric_kind": "D",
                "metric": name,
                "pair": stats.get("pair_name", "?"),
                "target": stats.get("target", float("nan")),
                "z": stats.get("z", float("nan")),
                "percentile": stats.get("percentile", float("nan")),
                "p_one_sided": stats.get("p_one_sided", float("nan")),
                "control_mean": stats.get("control_mean", float("nan")),
                "control_std": stats.get("control_std", float("nan")),
                "n_control": stats.get("n_control", 0),
                "cohen_d": block.get("cohen_d_target_vs_control", float("nan")),
            })
    # Design A: J metrics
    for name in J_METRICS:
        block = results["design_a"]["J"].get(name, {})
        for stats in block.get("per_target", []):
            rows.append({
                "set": set_name,
                "design": DESIGN_LABELS["design_a"],
                "metric_kind": "J",
                "metric": name,
                "pair": stats.get("pair_name", "?"),
                "target": stats.get("target", float("nan")),
                "z": stats.get("z", float("nan")),
                "percentile": stats.get("percentile", float("nan")),
                "p_one_sided": stats.get("p_one_sided", float("nan")),
                "control_mean": stats.get("control_mean", float("nan")),
                "control_std": stats.get("control_std", float("nan")),
                "n_control": stats.get("n_control", 0),
                "cohen_d": block.get("cohen_d_target_vs_control", float("nan")),
            })
    # Design B: D metrics only (J does not have a per-word form here)
    for name in D_METRICS:
        block = results["design_b"]["D"].get(name, {})
        for stats in block.get("per_target", []):
            rows.append({
                "set": set_name,
                "design": DESIGN_LABELS["design_b"],
                "metric_kind": "D",
                "metric": name,
                "pair": stats.get("pair_name", "?"),
                "target": stats.get("target", float("nan")),
                "z": stats.get("z", float("nan")),
                "percentile": stats.get("percentile", float("nan")),
                "p_one_sided": stats.get("p_one_sided", float("nan")),
                "control_mean": stats.get("control_mean", float("nan")),
                "control_std": stats.get("control_std", float("nan")),
                "n_control": stats.get("n_control", 0),
                "cohen_d": float("nan"),
                "spillover": stats.get("spillover"),
                "window_a": stats.get("window_a"),
                "window_b": stats.get("window_b"),
            })
    return rows


def _set_metric_z_table(rows: list[dict], design: str, kind: str
                        ) -> tuple[list[str], list[str], np.ndarray]:
    """Average ``z_target`` across pairs within each ``(set, metric)``.

    Returns ``(set_order, metric_order, z_matrix)``.
    """
    sets = sorted({r["set"] for r in rows
                   if r["design"] == design and r["metric_kind"] == kind})
    metrics_ord = (list(D_METRICS) if kind == "D" else list(J_METRICS))
    Z = np.full((len(sets), len(metrics_ord)), np.nan)
    for i, s in enumerate(sets):
        for j, m in enumerate(metrics_ord):
            zs = [r["z"] for r in rows
                  if r["design"] == design and r["metric_kind"] == kind
                  and r["set"] == s and r["metric"] == m
                  and r["z"] == r["z"]]  # filter NaN
            if zs:
                Z[i, j] = float(np.mean(zs))
    return sets, metrics_ord, Z


def _render_heatmap(sets, metrics, Z, title, out_path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    if not sets or not metrics or Z.size == 0:
        return
    fig, ax = plt.subplots(figsize=(max(6, 1.0 * len(metrics) + 4),
                                    max(3, 0.6 * len(sets) + 1)))
    vmax = float(np.nanmax(np.abs(Z))) if np.any(~np.isnan(Z)) else 1.0
    if vmax <= 0:
        vmax = 1.0
    im = ax.imshow(Z, cmap="coolwarm", vmin=-vmax, vmax=vmax, aspect="auto")
    ax.set_xticks(range(len(metrics)))
    ax.set_xticklabels(metrics, rotation=30, ha="right")
    ax.set_yticks(range(len(sets)))
    ax.set_yticklabels(sets)
    for i in range(Z.shape[0]):
        for j in range(Z.shape[1]):
            v = Z[i, j]
            if v == v:  # not NaN
                ax.text(j, i, f"{v:.2f}", ha="center", va="center",
                        color=("white" if abs(v) > vmax / 2 else "black"),
                        fontsize=8)
    fig.colorbar(im, ax=ax, label="z (target vs control)")
    ax.set_title(title)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def _render_bars_per_metric(rows: list[dict], design: str, kind: str,
                            fig_dir: str) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    metrics_ord = (list(D_METRICS) if kind == "D" else list(J_METRICS))
    for m in metrics_ord:
        sub = [r for r in rows
               if r["design"] == design and r["metric_kind"] == kind
               and r["metric"] == m]
        if not sub:
            continue
        fig, ax = plt.subplots(figsize=(max(6, 0.5 * len(sub) + 4), 4))
        labels = [f"{r['set']}/{r['pair']}" for r in sub]
        zs = [r["z"] for r in sub]
        x = np.arange(len(sub))
        ax.bar(x, zs, color="#4477aa")
        ax.axhline(0, color="black", linewidth=0.5)
        ax.set_xticks(x)
        ax.set_xticklabels(labels, rotation=45, ha="right", fontsize=8)
        ax.set_ylabel("z(target gap vs control)")
        ax.set_title(f"{design} | {kind} metric: {m}")
        fig.tight_layout()
        fig.savefig(os.path.join(fig_dir, f"bar_{design}_{kind}_{m}.png"),
                    dpi=150)
        plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--results-dir", default=os.path.join(THIS_DIR, "results"),
        help="directory containing <set>_results.json files")
    parser.add_argument(
        "--fig-dir", default=os.path.join(THIS_DIR, "figs", "cross_set"))
    parser.add_argument(
        "--include", default=None, nargs="*",
        help="restrict to specific sets (default: all *_results.json)")
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

    all_rows: list[dict] = []
    for f in files:
        try:
            data = read_json(f)
            rows = _collect_rows(data)
            all_rows.extend(rows)
            print(f"loaded {f}: {len(rows)} rows")
        except Exception as exc:
            print(f"could not load {f}: {exc}")

    summary = {"rows": all_rows}

    # Heatmaps + bar charts
    for design in (DESIGN_LABELS["design_a"], DESIGN_LABELS["design_b"]):
        for kind in ("D", "J"):
            sets, metrics_ord, Z = _set_metric_z_table(all_rows, design, kind)
            if Z.size:
                _render_heatmap(
                    sets, metrics_ord, Z,
                    title=f"{design} | {kind}: mean z(target gap)",
                    out_path=os.path.join(
                        args.fig_dir, f"heatmap_{design}_{kind}.png"))
                summary[f"heatmap_{design}_{kind}_sets"] = sets
                summary[f"heatmap_{design}_{kind}_metrics"] = metrics_ord
                summary[f"heatmap_{design}_{kind}_z"] = Z.tolist()
            _render_bars_per_metric(all_rows, design, kind, args.fig_dir)

    out_path = os.path.join(args.results_dir, "cross_set_summary.json")
    write_json(out_path, summary)
    print(f"wrote {out_path}")
    print(f"figures in {args.fig_dir}")


if __name__ == "__main__":
    main()
