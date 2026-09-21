"""Plot what the head-search scores actually measured: how long each run trained.

Left: the validation curve of three no-auxiliary runs that differ only in which
table they train on. On the full union the loss goes NaN in epoch 1 and chemprop
scores the test set from the epoch-0 checkpoint -- that checkpoint is the
reported 0.9430 "baseline".

Right: for every solo configuration of the 2026-09-14 search, the epoch at which
replaying chemprop's seeded sampler first hits a batch with no labelled row,
against the macro RMSE that configuration reported. Configurations that never
hit one are drawn at 30 ("never"). Rank correlation is Spearman, since the
relationship is monotone but not linear.

Data is inlined: the curves come from this repo's local reproduction (see
reports/head_search_baseline_bug.md) and the scores from
results/head_search_results.csv. Re-derive the death epochs with
scripts/diagnose_null_batches.py if the union is rebuilt.
"""

from pathlib import Path

import matplotlib.pyplot as plt
from scipy.stats import spearmanr

_QHTS = "#1f77b4"
_FAMILY = "#ff7f0e"
_INK = "#333333"

# (label, colour, [(epoch, val/mse)]) -- None marks where the run went NaN.
_CURVES = [
    ("challenge table (4,905 rows, 100% labelled)", _QHTS,
     [(0, 0.9141), (1, 0.8729), (2, 0.8230), (10, 0.6903), (20, 0.6402), (24, 0.6024), (29, 0.6549)]),
    ("+ qHTS molecules (20,870 rows, 23.5% labelled)", _FAMILY,
     [(0, 0.8453), (1, 0.9784), (2, 0.8202), (10, 0.6301), (20, 0.5588), (21, 0.5399), (28, 0.5676)]),
]
_DEAD = ("full union (38,376 rows, 10.5% labelled)", [(0, 0.9814)])

# gene, simulated first-empty-batch epoch (30 = never), reported macro RMSE, tier
_CONFIGS = [
    ("CYP1A2", 30, 0.7313, "qHTS"), ("CYP2C9", 30, 0.7378, "qHTS"), ("CYP2D6", 30, 0.7416, "qHTS"),
    ("CYP2C19", 30, 0.7450, "qHTS"), ("CYP3A4", 30, 0.7569, "qHTS"),
    ("CYP11B1", 14, 0.7797, "family"), ("CYP19A1", 24, 0.7889, "family"), ("CYP11B2", 30, 0.7909, "family"),
    ("CYP1A1", 5, 0.7930, "family"), ("CYP1B1", 5, 0.7960, "family"), ("CYP17A1", 1, 0.7988, "family"),
    ("CYP2B6", 5, 0.9424, "family"), ("CYP2A6", 1, 0.9429, "family"), ("TBXAS1", 6, 0.9437, "family"),
    ("CYP24A1", 1, 0.9438, "family"), ("CYP4A11", 6, 0.9442, "family"), ("CYP2C8", 5, 0.9447, "family"),
    ("CYP4F2", 6, 0.9480, "family"),
]
_BASELINE = ("baseline", 1, 0.9430)


def main() -> None:
    """Render reports/nan_death_vs_gain.png."""
    fig, (ax_curve, ax_scatter) = plt.subplots(1, 2, figsize=(13.5, 5.4))

    for label, colour, points in _CURVES:
        ax_curve.plot([e for e, _ in points], [v for _, v in points],
                      color=colour, linewidth=2, marker="o", markersize=4, label=label)

    label, points = _DEAD
    ax_curve.plot([e for e, _ in points], [v for _, v in points], color="#111111",
                  linewidth=2, marker="o", markersize=8, linestyle="none", label=label)
    ax_curve.annotate(
        "loss NaN at epoch 1, step 651\ntest scored from this checkpoint -> 0.9430",
        xy=(0, 0.9814), xytext=(3.0, 0.855), fontsize=8.5, color=_INK,
        arrowprops={"arrowstyle": "->", "color": _INK, "linewidth": 0.9},
    )
    ax_curve.set_xlabel("Epoch")
    ax_curve.set_ylabel("Validation MSE (scaled targets)")
    ax_curve.set_title("Same model, same split, no auxiliary heads")
    ax_curve.legend(fontsize=8, loc="lower left")
    ax_curve.grid(True, alpha=0.25)

    for tier, colour in (("qHTS", _QHTS), ("family", _FAMILY)):
        rows = [c for c in _CONFIGS if c[3] == tier]
        ax_scatter.scatter([r[1] for r in rows], [r[2] for r in rows], s=55, color=colour,
                           edgecolor="white", linewidth=0.6, zorder=3, label=f"{tier} auxiliary head")
    ax_scatter.scatter([_BASELINE[1]], [_BASELINE[2]], s=170, marker="*", color="#111111",
                       edgecolor="white", linewidth=0.6, zorder=4, label="no auxiliary head")

    # Configurations pile up on the two horizontal bands, so cycle the label
    # corner rather than stacking every name on the same side of its marker.
    for position, (gene, epoch, rmse, _tier) in enumerate(_CONFIGS):
        offset = [(6, 5), (6, -13), (-48, 5), (-48, -13), (6, 17), (-48, 17)][position % 6]
        ax_scatter.annotate(gene, (epoch, rmse), textcoords="offset points", xytext=offset,
                            fontsize=7.5, color=_INK)
    ax_scatter.annotate("baseline\n(no aux head)", (_BASELINE[1], _BASELINE[2]),
                        textcoords="offset points", xytext=(10, -26), fontsize=8, color=_INK,
                        arrowprops={"arrowstyle": "->", "color": _INK, "linewidth": 0.8})

    rho, p = spearmanr([c[1] for c in _CONFIGS], [c[2] for c in _CONFIGS])
    ax_scatter.text(0.03, 0.06, f"Spearman rho = {rho:+.2f}, p = {p:.4f}, n = {len(_CONFIGS)}",
                    transform=ax_scatter.transAxes, fontsize=8.5, color=_INK)
    # Headroom for the labels on the dead band and on the x = 30 column.
    ax_scatter.set_ylim(0.72, 0.985)
    ax_scatter.set_xlim(-1.5, 34)
    ax_scatter.set_xlabel("Epoch of first batch with no labelled row (30 = never)")
    ax_scatter.set_ylabel("Reported macro RMSE")
    ax_scatter.set_title("Reported score vs. how long the run survived")
    ax_scatter.legend(fontsize=8, loc="upper right")
    ax_scatter.grid(True, alpha=0.25, zorder=0)

    fig.tight_layout()
    out = Path("reports/nan_death_vs_gain.png")
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=150)
    print(f"Wrote {out}")


if __name__ == "__main__":
    main()
