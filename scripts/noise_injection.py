"""Rerun solo head-search configurations with noise injected into the auxiliary labels.

The question: does label noise in an auxiliary head reduce the gain it gives
the four scored isoforms? scripts/label_noise.py asks this across heads, which
confounds noise with everything else that differs between heads and has only
~20 points. This asks it causally: take one head that helps, corrupt its
labels by a known amount, and measure the gain again. If noise hurts, gain
falls monotonically with sigma.

Each configuration is a normal head-search run -- models.head_search's
evaluate_config, the same pinned split, seeds, epochs and chemprop flags --
on a copy of the union whose auxiliary columns for one candidate have been
corrupted. Nothing else in the table changes, so the scored labels, the split
and every other row are identical to the clean run.

Corruption:
  sigma    Gaussian noise added to every labelled training value of the
           candidate's auxiliary columns. With --noise-units log (default),
           sigma is in log units on pIC50 columns, and other readouts (the
           qHTS percent-inhibition columns) get sigma * sd(column) /
           sd(candidate pIC50 labels), i.e. the same signal-to-noise loss on
           their own scale. With --noise-units sd, sigma is a multiple of each
           column's own SD. Measured replicate SDs (label_noise.py) are the
           natural scale for sigma: ~0.3-0.7 log units.
  shuffled Each column's values permuted among its labelled training rows:
           the molecules stay, the labels become uninformative. If a head
           still helps when shuffled, its gain comes from the extra molecules
           reaching the encoder, not from what was measured on them.

Only train-split rows are corrupted by default. Validation labels pick the
checkpoint, and holding them clean keeps model selection identical across
noise levels; --corrupt-val corrupts them too.

The noise draw is fixed per (candidate, level, column) by --noise-seed, so
the seeds vary only the model initialization, as in head-search.

Usage (SCC GPU node, see scripts/noise_injection.qsub):
    python scripts/noise_injection.py \
        --data-path data/union_train.csv \
        --candidates CYP1A2 CYP2C8 \
        --sigmas 0 0.3 0.6 1.0 \
        --results-path results/noise_injection.json
"""

import argparse
import json
import shutil
import sys
import tempfile
from pathlib import Path

import numpy as np
import polars as pl

from models.head_search import _SCORED_DEFAULT, discover_candidates, evaluate_config


def corrupt(
    df: pl.DataFrame,
    columns: list[str],
    sigma: float | None,
    units: str,
    rows: np.ndarray,
    seed: list[int],
) -> pl.DataFrame:
    """Return df with `columns` corrupted on `rows`: Gaussian noise of `sigma`, or a permutation if sigma is None."""
    labelled = {c: rows & df[c].is_not_null().to_numpy() for c in columns}

    ref_sd = 1.0
    if sigma is not None and units == "log":
        pic50 = np.concatenate([df[c].to_numpy()[labelled[c]] for c in columns if "_pIC50_" in c] or [np.array([])])
        needs_ref = any("_pIC50_" not in c for c in columns)
        if needs_ref:
            if pic50.size < 2:
                raise SystemExit(
                    f"{columns} has non-pIC50 columns but no pIC50 labels to scale them against; use --noise-units sd"
                )
            ref_sd = float(np.std(pic50, ddof=1))

    updates = []
    for index, column in enumerate(columns):
        rng = np.random.default_rng(seed + [index])
        values = df[column].to_numpy().astype(float, copy=True)
        mask = labelled[column]
        if sigma is None:
            values[mask] = rng.permutation(values[mask])
        else:
            col_sd = float(np.std(values[mask], ddof=1)) if mask.sum() > 1 else 0.0
            if units == "sd":
                scale = sigma * col_sd
            elif "_pIC50_" in column:
                scale = sigma
            else:
                scale = sigma * col_sd / ref_sd
            values[mask] += rng.normal(0.0, scale, size=int(mask.sum()))
        updates.append(pl.Series(column, values).fill_nan(None))
    return df.with_columns(updates)


def _level_name(sigma: float | None) -> str:
    return "shuffled" if sigma is None else f"sigma={sigma:g}"


def _print_table(results: list[dict]) -> None:
    """Print each candidate's gain vs. baseline at every noise level, with the share of clean gain kept."""
    base = next((r for r in results if r["candidate"] is None), None)
    print(f"\n{'config':<28} {'macroRMSE':>10} {'sd':>7} {'gain':>8} {'of clean':>9}", file=sys.stderr)
    clean = {r["candidate"]: r for r in results if r["candidate"] is not None and r["sigma"] == 0}
    for r in results:
        line = f"{r['name']:<28} {r['macro_rmse_mean']:>10.4f} {r['macro_rmse_sd']:>7.4f}"
        if base is not None and r is not base:
            gain = base["macro_rmse_mean"] - r["macro_rmse_mean"]
            line += f" {gain:>+8.4f}"
            ref = clean.get(r["candidate"])
            if ref is not None and ref is not r:
                clean_gain = base["macro_rmse_mean"] - ref["macro_rmse_mean"]
                if clean_gain > 0:
                    line += f" {gain / clean_gain:>8.0%}"
        print(line, file=sys.stderr)
    print(
        "\nA gain difference under ~2 combined seed SDs between two levels is no difference (RULES.md).",
        file=sys.stderr,
    )


def main() -> None:
    """Entry point: head-search solo runs with injected auxiliary-label noise."""
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--data-path", type=Path, default=Path("data/union_train.csv"))
    parser.add_argument("--scored-columns", nargs="+", default=_SCORED_DEFAULT)
    parser.add_argument("--granularity", default="protein", choices=["protein", "source", "readout"])
    parser.add_argument(
        "--candidates",
        nargs="+",
        default=["CYP1A2", "CYP2C8"],
        help="Candidates to corrupt, named as head-search names them (default: the two best solo heads)",
    )
    parser.add_argument("--sigmas", nargs="+", type=float, default=[0.0, 0.3, 0.6, 1.0])
    parser.add_argument("--noise-units", default="log", choices=["log", "sd"])
    parser.add_argument("--no-shuffled", action="store_true", help="Skip the label-permutation control")
    parser.add_argument("--corrupt-val", action="store_true", help="Also corrupt validation-split labels")
    parser.add_argument("--skip-baseline", action="store_true", help="Do not rerun the no-aux baseline")
    parser.add_argument("--noise-seed", type=int, default=0)
    parser.add_argument("--seeds", type=int, default=3, help="Model seeds per configuration (default: 3)")
    parser.add_argument("--epochs", type=int, default=30, help="Epochs per run (default: 30)")
    parser.add_argument("--results-path", type=Path, default=Path("results/noise_injection.json"))
    parser.add_argument("--dry-run", action="store_true", help="Print the plan and run count, then exit")
    parser.add_argument(
        "--extra",
        nargs=argparse.REMAINDER,
        default=[],
        help="Everything after this flag is passed straight to `chemprop train`, as in head-search",
    )
    args = parser.parse_args()

    if shutil.which("chemprop") is None and not args.dry_run:
        raise SystemExit("chemprop CLI not found on PATH. Install it with: pip install chemprop")

    df = pl.read_csv(args.data_path, infer_schema_length=None)
    if "split" not in df.columns:
        raise SystemExit(f"{args.data_path} has no 'split' column; run add-split-column first")
    candidates = discover_candidates(df, args.scored_columns, args.granularity)
    unknown = [c for c in args.candidates if c not in candidates]
    if unknown:
        raise SystemExit(f"Unknown candidates {unknown}; available: {sorted(candidates)}")

    splits = {"train", "val"} if args.corrupt_val else {"train"}
    rows = df["split"].is_in(list(splits)).to_numpy()
    levels: list[float | None] = list(args.sigmas) + ([] if args.no_shuffled else [None])

    plan = [] if args.skip_baseline else [("baseline", None, None)]
    plan += [(f"{c} {_level_name(s)}", c, s) for c in args.candidates for s in levels]
    print(f"{len(plan)} configurations x {args.seeds} seeds = {len(plan) * args.seeds} runs", file=sys.stderr)
    for name, candidate, _ in plan:
        if candidate is not None:
            print(f"  {name}: {candidates[candidate]}", file=sys.stderr)
        else:
            print(f"  {name}", file=sys.stderr)
    if args.dry_run:
        return

    seeds = list(range(args.seeds))
    results: list[dict] = []
    args.results_path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="noise_injection_") as tmp:
        for name, candidate, sigma in plan:
            aux = candidates[candidate] if candidate is not None else []
            data_path = args.data_path
            if candidate is not None and sigma != 0:
                level_index = levels.index(sigma)
                seed = [args.noise_seed, args.candidates.index(candidate), level_index]
                noisy = corrupt(df, aux, sigma, args.noise_units, rows, seed)
                data_path = Path(tmp) / "noisy.csv"
                noisy.write_csv(data_path)

            print(f"  [{name}] {len(aux)} auxiliary columns", file=sys.stderr)
            record = evaluate_config(name, data_path, args.scored_columns, aux, seeds, args.epochs, args.extra)
            record.update(
                {
                    "candidate": candidate,
                    "sigma": "shuffled" if candidate is not None and sigma is None else sigma,
                    "noise_units": args.noise_units,
                    "corrupted_splits": sorted(splits),
                    "noise_seed": args.noise_seed,
                }
            )
            results.append(record)
            mean, sd = record["macro_rmse_mean"], record["macro_rmse_sd"]
            print(f"  [{name}] macro RMSE {mean:.4f} +/- {sd:.4f}", file=sys.stderr)
            # Written after every configuration so a job killed at the wall-clock limit keeps what it finished.
            args.results_path.write_text(json.dumps(results, indent=2))

    _print_table(results)
    print(f"\nWrote {args.results_path}", file=sys.stderr)


if __name__ == "__main__":
    main()
