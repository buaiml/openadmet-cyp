"""Plot aux-head macro RMSE gain against chemical proximity to the evaluation set.

Reads results/aux_similarity.csv (scripts/aux_similarity.py) and draws one
panel per leakage channel, so a gain driven by proximity is visually separable
from a gain that is not. The dashed line in the Tanimoto panel is
`__train_scored__` -- how close the baseline model's own training molecules
already sat to the eval set. An aux head to the *left* of that line brought
molecules that are further from the eval set than what the baseline already
had, which is the opposite of the "we just found similar molecules" story.

Usage:
    python scripts/plot_similarity_vs_gain.py [--in results/aux_similarity.csv]
"""

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import polars as pl  # noqa: E402

_PANELS = [
    ("nn_tanimoto_mean", "Mean nearest-neighbour ECFP4 Tanimoto to eval set"),
    ("eval_in_aux_frac", "Fraction of eval molecules supervised by this head"),
    ("scaffold_overlap_frac", "Fraction of eval scaffolds present in this head's train rows"),
]


def main() -> None:
    """Render the proximity-vs-gain panels to reports/similarity_vs_gain.png."""
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--in", dest="in_path", type=Path, default=Path("results/aux_similarity.csv"))
    parser.add_argument("--out", type=Path, default=Path("reports/similarity_vs_gain.png"))
    args = parser.parse_args()

    df = pl.read_csv(args.in_path)
    references = {r["name"]: r for r in df.filter(pl.col("name").str.starts_with("__")).to_dicts()}
    heads = df.filter(~pl.col("name").str.starts_with("__") & pl.col("gain").is_not_null() & pl.col("gain").is_not_nan())

    if not len(heads):
        raise SystemExit(f"{args.in_path} has no aux heads with a gain -- was head-search run with --strategy single or greedy?")

    fig, axes = plt.subplots(1, len(_PANELS), figsize=(5.0 * len(_PANELS), 5.0), sharey=True)

    for ax, (column, label) in zip(axes, _PANELS):
        x = heads[column].to_numpy()
        y = heads["gain"].to_numpy()
        sizes = heads["n_molecules"].to_numpy()

        points = ax.scatter(
            x,
            y,
            c=sizes,
            cmap="viridis",
            norm=matplotlib.colors.LogNorm(),
            s=70,
            edgecolor="white",
            linewidth=0.6,
            zorder=3,
        )
        for name, xi, yi in zip(heads["name"].to_list(), x, y):
            ax.annotate(name, (xi, yi), textcoords="offset points", xytext=(5, 4), fontsize=7.5, color="#333333")

        if column == "nn_tanimoto_mean" and "__train_scored__" in references:
            baseline_x = references["__train_scored__"][column]
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
        ax.set_xlabel(label, fontsize=9)
        ax.grid(True, alpha=0.25, zorder=0)

    axes[0].set_ylabel("Macro RMSE improvement over baseline")
    fig.colorbar(points, ax=axes, label="Aux head molecules", fraction=0.025, pad=0.01)
    fig.suptitle("Does aux-head gain track chemical proximity to the evaluation set?", fontsize=12)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.out, dpi=150, bbox_inches="tight")
    print(f"Wrote {args.out}")


if __name__ == "__main__":
    main()
