"""Plot aux-head macro RMSE gain against chemical proximity to the evaluation set.

Reads results/aux_similarity.csv (scripts/aux_similarity.py) and draws one
panel per proximity measure, so a gain driven by proximity is visually
separable from a gain that is not. The dashed line in the Tanimoto panel is
`__train_scored__` -- how close the baseline model's own training molecules
already sat to the eval set. An aux head to the *left* of that line brought
molecules that are further from the eval set than what the baseline already
had, which is the opposite of the "we just found similar molecules" story.

Stdlib + numpy + matplotlib only; see aux_similarity.py on why polars is
avoided on the SCC compute nodes.

Usage:
    python scripts/plot_similarity_vs_gain.py [--in results/aux_similarity.csv]
"""

import argparse
import csv
import math
import os
import sys
from pathlib import Path

# Must precede the pyplot import: the default cache lands in $HOME, which is
# over quota on the SCC, and matplotlib only warns before falling back slowly.
os.environ.setdefault("MPLCONFIGDIR", os.environ.get("TMPDIR", "/tmp") + "/mpl-cache")
Path(os.environ["MPLCONFIGDIR"]).mkdir(parents=True, exist_ok=True)

import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.colors  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402

_PANELS = [
    ("nn_tanimoto_mean", "Mean nearest-neighbour ECFP4 Tanimoto to eval set"),
    ("eval_in_aux_frac", "Fraction of eval molecules drawn from this head's library"),
    ("scaffold_overlap_frac", "Fraction of eval scaffolds present in this head's train rows"),
]


def _to_float(value: str) -> float:
    """Parse a CSV cell to float, mapping blanks and 'nan' to NaN."""
    if value is None or value == "":
        return float("nan")
    try:
        return float(value)
    except ValueError:
        return float("nan")


def main() -> None:
    """Render the proximity-vs-gain panels to reports/similarity_vs_gain.png."""
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--in", dest="in_path", type=Path, default=Path("results/aux_similarity.csv"))
    parser.add_argument("--out", type=Path, default=Path("reports/similarity_vs_gain.png"))
    args = parser.parse_args()

    with args.in_path.open(newline="") as handle:
        rows = list(csv.DictReader(handle))

    references = {r["name"]: r for r in rows if r["name"].startswith("__")}
    heads = [r for r in rows if not r["name"].startswith("__") and not math.isnan(_to_float(r["gain"]))]

    if not heads:
        raise SystemExit(f"{args.in_path} has no aux heads with a gain -- was head-search run with --strategy single or greedy?")

    names = [r["name"] for r in heads]
    gains = [_to_float(r["gain"]) for r in heads]
    sizes = [_to_float(r["n_molecules"]) for r in heads]

    fig, axes = plt.subplots(1, len(_PANELS), figsize=(5.0 * len(_PANELS), 5.0), sharey=True)

    points = None
    for ax, (column, label) in zip(axes, _PANELS):
        x = [_to_float(r[column]) for r in heads]
        points = ax.scatter(
            x,
            gains,
            c=sizes,
            cmap="viridis",
            norm=matplotlib.colors.LogNorm(),
            s=70,
            edgecolor="white",
            linewidth=0.6,
            zorder=3,
        )
        for name, xi, yi in zip(names, x, gains):
            ax.annotate(name, (xi, yi), textcoords="offset points", xytext=(5, 4), fontsize=7.5, color="#333333")

        if column == "nn_tanimoto_mean" and "__train_scored__" in references:
            baseline_x = _to_float(references["__train_scored__"][column])
            ax.axvline(baseline_x, color="#d62728", linestyle="--", linewidth=1.1, zorder=2)
            ax.annotate(
                "baseline train molecules",
                (baseline_x, ax.get_ylim()[1]),
                rotation=90,
                fontsize=7.5,
                color="#d62728",
                ha="right",
                va="top",
                textcoords="offset points",
                xytext=(-3, -4),
            )

        ax.axhline(0, color="#999999", linewidth=0.8, zorder=1)
        ax.margins(x=0.15)  # room for the point labels on the right edge
        ax.set_xlabel(label, fontsize=9)
        ax.grid(True, alpha=0.25, zorder=0)

    axes[0].set_ylabel("Macro RMSE improvement over baseline")
    fig.colorbar(points, ax=axes, label="Aux head molecules", fraction=0.025, pad=0.01)
    fig.suptitle("Does aux-head gain track chemical proximity to the evaluation set?", fontsize=12)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.out, dpi=150, bbox_inches="tight")
    print(f"Wrote {args.out}", file=sys.stderr)


if __name__ == "__main__":
    main()
