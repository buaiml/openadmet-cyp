"""Plot solo aux-head macro RMSE gain vs. molecule count (reports/head_search_analysis.md).

Gains come from the head-search JSON (solo configs only, `baseline - solo`, so
positive means the head helped) and molecule counts from the `n_molecules`
column of `scripts/aux_similarity.py`'s table -- molecules in the union that
carry at least one of the head's labels, i.e. exactly what the fixed
head-search adds to training when it selects that head. Error bars are the
solo config's seed SD; the grey band is the baseline's.

Usage:
    python scripts/plot_volume_vs_gain.py \
        [--results-path results/head_search.json] [--similarity-path results/aux_similarity.csv]
"""

import argparse
import csv
import json
import os
import sys
from pathlib import Path

# Must precede the pyplot import: the default cache lands in $HOME, which is
# over quota on the SCC.
os.environ.setdefault("MPLCONFIGDIR", os.environ.get("TMPDIR", "/tmp") + "/mpl-cache")
Path(os.environ["MPLCONFIGDIR"]).mkdir(parents=True, exist_ok=True)

import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

# Isoforms that also have a scored column: their aux head trains the same
# protein the metric is computed on.
_SELF_AUX = {"CYP1A2", "CYP2C9", "CYP2D6", "CYP3A4"}
_QHTS = {"CYP1A2", "CYP2C9", "CYP2C19", "CYP2D6", "CYP3A4"}

_COLORS = {"qHTS": "#1f77b4", "family": "#ff7f0e"}


def main() -> None:
    """Render the volume-vs-gain scatter to reports/data_volume_vs_gain.png."""
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--results-path", type=Path, default=Path("results/head_search.json"))
    parser.add_argument("--similarity-path", type=Path, default=Path("results/aux_similarity.csv"))
    parser.add_argument("--out", type=Path, default=Path("reports/data_volume_vs_gain.png"))
    args = parser.parse_args()

    entries = json.loads(args.results_path.read_text())
    baseline = next((e for e in entries if not e["candidates"]), None)
    if baseline is None:
        raise SystemExit(f"{args.results_path} has no baseline entry (one with an empty 'candidates')")
    solo = {e["candidates"][0]: e for e in entries if len(e["candidates"]) == 1}

    with args.similarity_path.open(newline="") as handle:
        counts = {r["name"]: int(r["n_molecules"]) for r in csv.DictReader(handle) if not r["name"].startswith("__")}

    missing = sorted(set(solo) - set(counts))
    if missing:
        print(f"WARNING: no molecule count for {missing}; dropped from the plot", file=sys.stderr)

    base = baseline["macro_rmse_mean"]
    base_sd = baseline["macro_rmse_sd"]
    points = [
        (gene, counts[gene], base - entry["macro_rmse_mean"], entry["macro_rmse_sd"])
        for gene, entry in solo.items()
        if gene in counts
    ]

    fig, ax = plt.subplots(figsize=(8, 5.5))
    ax.axhspan(-base_sd, base_sd, color="#999999", alpha=0.18, zorder=0, label="baseline ± 1 SD")
    ax.axhline(0, color="#999999", linewidth=0.8, zorder=1)

    groups = [
        ("family", lambda g: g not in _QHTS, "o", 55),
        ("qHTS", lambda g: g in _QHTS and g not in _SELF_AUX, "o", 55),
        ("qHTS, self-aux", lambda g: g in _SELF_AUX, "*", 170),
    ]
    for label, keep, marker, size in groups:
        rows = [p for p in points if keep(p[0])]
        color = _COLORS["family" if label == "family" else "qHTS"]
        ax.errorbar(
            [r[1] for r in rows], [r[2] for r in rows], yerr=[r[3] for r in rows],
            fmt="none", ecolor=color, alpha=0.5, linewidth=1.0, zorder=2,
        )
        ax.scatter(
            [r[1] for r in rows], [r[2] for r in rows], label=label, color=color, marker=marker, s=size,
            zorder=4, edgecolor="black" if marker == "*" else "white", linewidth=0.6,
        )

    for gene, n, gain, _sd in points:
        ax.annotate(gene, (n, gain), textcoords="offset points", xytext=(5, 4), fontsize=7.5, color="#333333")

    ax.set_xscale("log")
    ax.set_xlabel("Molecules carrying the aux head's labels (log scale)")
    ax.set_ylabel(f"Macro RMSE improvement over baseline ({base:.4f})")
    ax.set_title("Solo aux-head gain vs. molecule count")
    ax.legend(title="Tier", fontsize=8)
    ax.grid(True, which="both", alpha=0.25, zorder=0)

    fig.tight_layout()
    args.out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.out, dpi=150)
    print(f"Wrote {args.out}", file=sys.stderr)


if __name__ == "__main__":
    main()
