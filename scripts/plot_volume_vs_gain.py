"""Plot solo aux-head macro RMSE gain vs. example count (reports/head_search_analysis.md).

Data hardcoded from results/data_counts.csv (row totals) and
results/head_search_results.csv (solo-head macro RMSE, baseline 0.9430) --
the BU SCC head-search run of 2026-09-14. Re-derive from those CSVs directly
if either is regenerated.
"""

from pathlib import Path

import matplotlib.pyplot as plt

_BASELINE = 0.9430

# gene, total rows, solo macro RMSE, tier, self-aux (isoform also has a scored column)
_SELF_AUX = {"CYP1A2", "CYP2C9", "CYP2D6", "CYP3A4"}

_DATA = [
    ("CYP3A4", 40432, 0.7569, "qHTS"),
    ("CYP2C9", 32527, 0.7378, "qHTS"),
    ("CYP1A2", 30958, 0.7313, "qHTS"),
    ("CYP2C19", 29780, 0.7450, "qHTS"),
    ("CYP2D6", 28847, 0.7416, "qHTS"),
    ("CYP19A1", 2058, 0.7889, "family"),
    ("TBXAS1", 1628, 0.9437, "family"),
    ("CYP11B2", 1659, 0.7909, "family"),
    ("CYP1B1", 1071, 0.7960, "family"),
    ("CYP1A1", 971, 0.7930, "family"),
    ("CYP2C8", 888, 0.9447, "family"),
    ("CYP4A11", 872, 0.9442, "family"),
    ("CYP4F2", 872, 0.9480, "family"),
    ("CYP11B1", 1393, 0.7797, "family"),
    ("CYP17A1", 856, 0.7988, "family"),
    ("CYP2B6", 682, 0.9424, "family"),
    ("CYP2A6", 575, 0.9429, "family"),
    ("CYP2E1", 235, 0.9422, "family"),
    ("CYP3A5", 192, 0.9416, "family"),
    ("CYP24A1", 148, 0.9438, "family"),
]

_COLORS = {"qHTS": "#1f77b4", "family": "#ff7f0e"}


def main() -> None:
    """Render the volume-vs-gain scatter to reports/data_volume_vs_gain.png."""
    fig, ax = plt.subplots(figsize=(8, 5.5))

    for tier in ("family", "qHTS"):
        rows = [d for d in _DATA if d[3] == tier and d[0] not in _SELF_AUX]
        x = [r[1] for r in rows]
        y = [_BASELINE - r[2] for r in rows]
        ax.scatter(x, y, label=tier, color=_COLORS[tier], s=55, zorder=3, edgecolor="white", linewidth=0.6)

    self_rows = [d for d in _DATA if d[0] in _SELF_AUX]
    ax.scatter(
        [r[1] for r in self_rows],
        [_BASELINE - r[2] for r in self_rows],
        label="qHTS, self-aux",
        color=_COLORS["qHTS"],
        marker="*",
        s=170,
        zorder=4,
        edgecolor="black",
        linewidth=0.6,
    )

    for gene, n, rmse, _tier in _DATA:
        ax.annotate(
            gene,
            (n, _BASELINE - rmse),
            textcoords="offset points",
            xytext=(5, 4),
            fontsize=7.5,
            color="#333333",
        )

    ax.axhline(0, color="#999999", linewidth=0.8, zorder=1)
    ax.set_xscale("log")
    ax.set_xlabel("Aux head example count (log scale)")
    ax.set_ylabel("Macro RMSE improvement over baseline (0.9430)")
    ax.set_title("Solo aux-head gain vs. example count")
    ax.legend(title="Tier")
    ax.grid(True, which="both", alpha=0.25, zorder=0)

    fig.tight_layout()
    out = Path("reports/data_volume_vs_gain.png")
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=150)
    print(f"Wrote {out}")


if __name__ == "__main__":
    main()
