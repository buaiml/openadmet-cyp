"""CLI for evaluating CYP challenge models on a train/validation split."""

import argparse
import csv
import re
import sys
from pathlib import Path

import numpy as np
import polars as pl
from scipy import stats
from rich import box
from rich.console import Console
from rich.markup import escape
from rich.table import Table

from data_tools.inputs import INPUT_REGISTRY, featurize, is_known_input
from data_tools.load import load_data
from models import REGISTRY, CYPModel
from models.utils import drop_nan_rows

_CYP_TARGETS = [
    "CYP1A2_pIC50_direct_inhibition",
    "CYP2C9_pIC50_direct_inhibition",
    "CYP2D6_pIC50_direct_inhibition",
    "CYP3A4_pIC50_direct_inhibition",
]

_ISOFORMS = {t.split("_")[0].removeprefix("CYP"): t for t in _CYP_TARGETS}

console = Console()

_PREDICTION_COLUMNS = ["smiles", "isoform", "model", "input", "split", "seed", "y_true", "y_pred"]


def _write_predictions(path: Path, rows: list[dict[str, object]]) -> None:
    """Write a new CSV, rejecting ambiguous join keys before creating the file."""
    seen = set()
    for row in rows:
        key = tuple(row[column] for column in _PREDICTION_COLUMNS[:6])
        if key in seen:
            raise ValueError(
                f"Cannot export predictions: duplicate SMILES {row['smiles']!r} within "
                f"isoform={row['isoform']}, model={row['model']}, input={row['input']}, "
                f"split={row['split']}, seed={row['seed']}; downstream SMILES joins would be ambiguous. "
                "No rows were deduplicated or written."
            )
        seen.add(key)
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("x", newline="", encoding="utf-8") as stream:
            writer = csv.DictWriter(stream, fieldnames=_PREDICTION_COLUMNS)
            writer.writeheader()
            writer.writerows(rows)
    except OSError as exc:
        raise OSError(f"Cannot write predictions CSV to {path}: {exc}") from exc


_METRICS = ["RMSE", "MAE", "R2", "ST-RAE", "Spearman", "Kendall"]
# Direction each metric improves in: True = lower is better.
_HEADERS = {"RMSE": "RMSE", "MAE": "MAE", "R2": "R2", "ST-RAE": "ST-RAE", "Spearman": "ρ", "Kendall": "τ"}
_LOWER_IS_BETTER = {"RMSE": True, "MAE": True, "R2": False, "ST-RAE": True, "Spearman": False, "Kendall": False}


def _compute_metrics(actual: pl.Series, predicted: pl.Series) -> dict[str, float]:
    """Compute RMSE, MAE, R2, ST-RAE, Spearman's rho and Kendall's tau from actual and predicted Series.

    ST-RAE is the relative absolute error of one target: total absolute error over
    the total absolute error of always predicting the validation mean (<1 beats the mean).
    Rank correlations are NaN when either side is constant.
    """
    a = np.array([float(v) for v in actual])
    p = np.array([float(v) for v in predicted])
    n = len(a)
    ss_res = float(((a - p) ** 2).sum())
    ss_tot = float(((a - a.mean()) ** 2).sum())
    abs_err = float(np.abs(a - p).sum())
    abs_dev = float(np.abs(a - a.mean()).sum())
    constant = a.std() == 0 or p.std() == 0
    return {
        "RMSE": (ss_res / n) ** 0.5,
        "MAE": abs_err / n,
        "R2": 1.0 - ss_res / ss_tot if ss_tot > 0.0 else 0.0,
        "ST-RAE": abs_err / abs_dev if abs_dev > 0.0 else float("nan"),
        "Spearman": float("nan") if constant else float(stats.spearmanr(a, p).statistic),
        "Kendall": float("nan") if constant else float(stats.kendalltau(a, p).statistic),
    }


def _evaluate_model(
    model: CYPModel,
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    *,
    return_predictions: bool = False,
) -> dict[str, float] | tuple[dict[str, float], np.ndarray]:
    """Fit and predict once; optionally return the predictions used for metrics."""
    model.fit(X_train, y_train)
    predictions = model.predict(X_val)
    preds = pl.Series("prediction", predictions.tolist())
    metrics = _compute_metrics(pl.Series("actual", y_val.tolist()), preds)
    return (metrics, predictions) if return_predictions else metrics


def _resolve_models(names: list[str]) -> list[type[CYPModel]]:
    """Return model classes for the given names, or all registered models if names == ['all']."""
    if names == ["all"]:
        return list(REGISTRY.values())
    unknown = [n for n in names if n not in REGISTRY]
    if unknown:
        raise ValueError(f"Unknown models: {unknown}. Available: {list(REGISTRY.keys())}")
    return [REGISTRY[n] for n in names]


def _resolve_inputs(groups: list[list[str]]) -> list[list[str]]:
    """Return the input sets to evaluate: each registered input alone for ['all'], else each group hstacked."""
    sets: list[list[str]] = []
    for names in groups:
        if names == ["all"]:
            sets.extend([n] for n in INPUT_REGISTRY)
            continue
        unknown = [n for n in names if not is_known_input(n)]
        if unknown:
            raise ValueError(f"Unknown input(s): {unknown}. Available: {list(INPUT_REGISTRY.keys())}")
        sets.append(names)
    return sets


def _short(label: str) -> str:
    """Shorten `chemprop[/long/path/best.pt]` to `chemprop[best.pt]` for display."""
    return re.sub(r"chemprop\[([^\]]*)\]", lambda m: f"chemprop[{Path(m.group(1)).name}]", label)


def _aggregate(results: list[tuple[str, str, str, dict[str, float]]]) -> list[tuple[str, str, dict[str, float]]]:
    """Macro-average each metric over isoforms per (model, input); NaNs are skipped per metric."""
    grouped: dict[tuple[str, str], list[dict[str, float]]] = {}
    for _, name, inp, m in results:
        grouped.setdefault((name, inp), []).append(m)
    return [
        (name, inp, {k: float(np.nanmean([m[k] for m in ms])) if not all(np.isnan(m[k]) for m in ms) else float("nan")
                     for k in _METRICS})
        for (name, inp), ms in grouped.items()
    ]


def _format_cell(metric: str, value: float, best: float) -> str:
    if np.isnan(value):
        return "[dim]n/a[/]"
    text = f"{value:.3f}"
    if value == best:
        return f"[bold green]{text}[/]"
    if metric in ("R2", "Spearman", "Kendall") and value < 0:
        return f"[red]{text}[/]"
    return text


def _print_results(results: list[tuple[str, str, str, dict[str, float]]], sort_by: str) -> None:
    """Print one metrics table with a block per isoform and a final macro-average block.

    The best value on each metric within a block is bold green.
    """
    by_isoform: dict[str, list[tuple[str, str, dict[str, float]]]] = {}
    for iso, name, inp, m in results:
        by_isoform.setdefault(iso, []).append((name, inp, m))
    blocks = list(by_isoform.items())
    if len(by_isoform) > 1:
        macro = _aggregate(results)
        reverse = not _LOWER_IS_BETTER[sort_by]
        macro.sort(key=lambda r: (-r[2][sort_by] if reverse else r[2][sort_by]) if not np.isnan(r[2][sort_by]) else float("inf"))
        blocks.append(("MEAN", macro))

    table = Table(box=box.SIMPLE_HEAD, header_style="bold cyan", title_style="bold", padding=(0, 1), pad_edge=False)
    table.add_column("Isoform", style="bold", no_wrap=True)
    table.add_column("Model", overflow="fold")
    table.add_column("Input", overflow="fold")
    for metric in _METRICS:
        arrow = " ▼" if _LOWER_IS_BETTER[metric] else " ▲"
        header = _HEADERS[metric] + (arrow if metric == sort_by else "")
        table.add_column(header, justify="right", no_wrap=True, min_width=len(header))

    for i, (iso, rows) in enumerate(blocks):
        best = {}
        for metric, lower in _LOWER_IS_BETTER.items():
            values = [r[2][metric] for r in rows if not np.isnan(r[2][metric])]
            best[metric] = (min if lower else max)(values) if values else float("nan")
        for j, (name, inp, m) in enumerate(rows):
            label = iso.replace("_pIC50_direct_inhibition", "") if j == 0 else ""
            table.add_row(
                f"[bold magenta]{label}[/]" if iso == "MEAN" else label,
                name,
                escape(_short(inp)),
                *(_format_cell(metric, m[metric], best[metric]) for metric in _METRICS),
                end_section=j == len(rows) - 1 and i < len(blocks) - 1,
            )
    console.print(table)


def _print_correlations(target: str, preds: dict[tuple[str, str], dict[int, float]]) -> None:
    """Print an all-vs-all Pearson correlation matrix of validation predictions for one target.

    Each config's predictions are keyed by validation row index, so configs whose
    NaN filtering dropped different rows are compared on their shared molecules.
    """
    labels = list(preds)
    table = Table(
        box=box.SIMPLE_HEAD,
        header_style="bold cyan",
        title=f"Prediction correlation (Pearson) · {target.replace('_pIC50_direct_inhibition', '')}",
        title_style="bold",
    )
    table.add_column("", style="bold", overflow="fold")
    for i in range(len(labels)):
        table.add_column(str(i), justify="right")
    for i, a in enumerate(labels):
        cells = []
        for j, b in enumerate(labels):
            shared = sorted(preds[a].keys() & preds[b].keys())
            x = np.array([preds[a][k] for k in shared])
            y = np.array([preds[b][k] for k in shared])
            if len(shared) < 2 or x.std() == 0 or y.std() == 0:
                cells.append("[dim]n/a[/]")
                continue
            r = float(np.corrcoef(x, y)[0, 1])
            color = "dim" if i == j else "green" if r >= 0.9 else "yellow" if r >= 0.6 else "red"
            cells.append(f"[{color}]{r:.2f}[/]")
        table.add_row(escape(f"{i}  {a[0]} [{_short(a[1])}]"), *cells)
    console.print()
    console.print(table)


def main() -> None:
    """Entry point for the evaluate-models CLI."""
    parser = argparse.ArgumentParser(description="Evaluate CYP models on a train/val split.")
    parser.add_argument("--data-dir", type=Path, default=Path("data"), help="Directory containing data CSVs")
    parser.add_argument("--val-split", type=float, default=0.2, help="Validation fraction")
    parser.add_argument(
        "--split",
        default="scaffold",
        choices=["random", "scaffold", "butina"],
        help="Train/val split strategy (default: scaffold)",
    )
    parser.add_argument("--seed", type=int, default=42, help="Random seed for splitting")
    parser.add_argument(
        "--butina-cutoff",
        type=float,
        default=0.4,
        help="Tanimoto distance cutoff for Butina clustering (default: 0.4).",
    )
    parser.add_argument("--models", nargs="+", default=["all"], help="Model names to evaluate, or 'all'")
    parser.add_argument(
        "--targets",
        nargs="+",
        default=None,
        choices=_CYP_TARGETS,
        help="CYP target column(s) to evaluate (default: all four isoforms)",
    )
    parser.add_argument(
        "--isoforms",
        nargs="+",
        default=None,
        choices=list(_ISOFORMS),
        help="Shorthand for --targets: isoform(s) to evaluate, direct inhibition pIC50 only "
        f"(e.g. --isoforms 3A4 2D6). Choices: {list(_ISOFORMS)}",
    )
    parser.add_argument(
        "--input",
        nargs="+",
        action="append",
        help="Input featurization(s) to use (hstacked if multiple), or 'all' to evaluate each registered "
        "input on its own. Repeat the flag to evaluate several input sets separately, "
        f"e.g. --input xtb_solv_v1 --input chemeleon. Default: rdkit. Available: {list(INPUT_REGISTRY.keys())}",
    )
    parser.add_argument(
        "--correlation",
        action="store_true",
        help="After evaluating, print an all-vs-all correlation matrix of validation predictions "
        "across every (model, input) combination, per target",
    )
    parser.add_argument("--cache-dir", default="data/features", help="Directory for cached feature matrices")
    parser.add_argument("--sort-by", default="MAE", choices=_METRICS, help="Metric to sort results by")
    parser.add_argument(
        "--predictions-out", type=Path,
        help="Write validation predictions to a new CSV (use results/); creates parents, refuses existing files",
    )
    parser.add_argument(
        "--predict-blind",
        action="store_true",
        help="Also predict the blinded test set (data/test.csv) with each fitted model and write one CSV per "
        "(model, input) to results/<model>_<input>_blind_predictions.csv. Models are fit on the train split only.",
    )
    args = parser.parse_args()
    args.input = args.input or [["rdkit"]]

    if args.targets is not None and args.isoforms is not None:
        parser.error("--targets and --isoforms are mutually exclusive")
    if args.isoforms is not None:
        args.targets = [_ISOFORMS[i] for i in args.isoforms]
    elif args.targets is None:
        args.targets = _CYP_TARGETS

    if not 0.0 < args.val_split < 1.0:
        print(f"--val-split must be in (0, 1), got {args.val_split}", file=sys.stderr)
        sys.exit(1)

    try:
        model_classes = _resolve_models(args.models)
        input_sets = _resolve_inputs(args.input)
    except ValueError as exc:
        print(str(exc), file=sys.stderr)
        sys.exit(1)

    cache_dir = Path(args.cache_dir)

    train_df, val_df = load_data(
        split_type=args.split,
        val_fraction=args.val_split,
        seed=args.seed,
        butina_cutoff=args.butina_cutoff,
        data_dir=args.data_dir,
    )
    print(f"load_data(split_type={args.split!r}, seed={args.seed}): {len(val_df)} val / {len(train_df)} train molecules", file=sys.stderr)

    results: list[tuple[str, str, str, dict[str, float]]] = []
    prediction_rows: list[dict[str, object]] = []
    # target -> (model, input) -> {val row index: prediction}
    corr_preds: dict[str, dict[tuple[str, str], dict[int, float]]] = {}
    blind_df = pl.read_csv(args.data_dir / "test.csv") if args.predict_blind else None
    # (model, input) -> target -> prediction per blind row (NaN where features are NaN)
    blind_preds: dict[tuple[str, str], dict[str, np.ndarray]] = {}
    for input_names in input_sets:
        input_label = " + ".join(input_names)
        # Features don't depend on the target isoform, so compute once and reuse across isoforms.
        train_features = featurize(train_df, input_names, cache_dir)
        val_features = featurize(val_df, input_names, cache_dir)
        if blind_df is not None:
            blind_features = featurize(blind_df, input_names, cache_dir)
            blind_valid = ~np.isnan(blind_features).any(axis=1)
            if not blind_valid.all():
                print(
                    f"Dropped {int((~blind_valid).sum())} of {len(blind_df)} blind molecules with NaN features "
                    f"({input_label}); their predictions are left empty",
                    file=sys.stderr,
                )
        for target in args.targets:
            label = f"{target}, {input_label}"
            X_train, y_train = drop_nan_rows(train_features, train_df[target].to_numpy(), label=f"train/{label}")
            X_val, y_val, val_mask = drop_nan_rows(
                val_features, val_df[target].to_numpy(), label=f"val/{label}", return_mask=True,
            )
            val_index = np.flatnonzero(val_mask)
            if args.predictions_out is not None:
                val_smiles = val_df["SMILES"].to_numpy()[val_mask]
            for cls in model_classes:
                model = cls()
                if args.predictions_out is None and not args.correlation:
                    metrics = _evaluate_model(model, X_train, y_train, X_val, y_val)
                else:
                    metrics, predictions = _evaluate_model(
                        model, X_train, y_train, X_val, y_val, return_predictions=True,
                    )
                    if args.correlation:
                        corr_preds.setdefault(target, {})[(cls.name, input_label)] = dict(
                            zip(val_index.tolist(), predictions.tolist())
                        )
                    if args.predictions_out is not None:
                        prediction_rows.extend(
                            dict(zip(_PREDICTION_COLUMNS, (
                                smiles, target, cls.name, input_label, args.split, args.seed, float(actual), float(pred),
                            )))
                            for smiles, actual, pred in zip(val_smiles, y_val, predictions)
                        )
                if blind_df is not None:
                    blind_pred = np.full(len(blind_df), np.nan)
                    blind_pred[blind_valid] = model.predict(blind_features[blind_valid])
                    blind_preds.setdefault((cls.name, input_label), {})[target] = blind_pred
                results.append((target, cls.name, input_label, metrics))

    if blind_df is not None:
        out_dir = Path("results")
        out_dir.mkdir(parents=True, exist_ok=True)
        for (name, input_label), by_target in blind_preds.items():
            path = out_dir / f"{name}_{re.sub(r'[^\w+.-]', '_', input_label.replace(' + ', '+'))}_blind_predictions.csv"
            blind_df.select(["Molecule_Name", "SMILES"]).with_columns(
                [pl.Series(target, pred) for target, pred in by_target.items()]
            ).write_csv(path)
            print(f"Wrote {len(blind_df)} blind predictions to {path}", file=sys.stderr)

    if args.predictions_out is not None:
        try:
            _write_predictions(args.predictions_out, prediction_rows)
        except (OSError, ValueError) as exc:
            print(str(exc), file=sys.stderr)
            sys.exit(1)
        print(f"Wrote {len(prediction_rows)} validation predictions to {args.predictions_out}", file=sys.stderr)

    reverse = not _LOWER_IS_BETTER[args.sort_by]

    def _sort_key(r: tuple[str, str, str, dict[str, float]]) -> tuple[str, float]:
        value = r[3][args.sort_by]
        return (r[0], float("inf") if np.isnan(value) else -value if reverse else value)

    results.sort(key=_sort_key)
    console.print(
        f"[dim]input[/]="
        + escape(_short(" | ".join(" + ".join(g) for g in args.input)))
        + f"  [dim]split[/]={args.split}  [dim]seed[/]={args.seed}"
    )
    _print_results(results, args.sort_by)
    if args.correlation:
        for target, preds in corr_preds.items():
            if len(preds) < 2:
                print(f"\n--correlation needs at least two (model, input) combinations; got {len(preds)}", file=sys.stderr)
                break
            _print_correlations(target, preds)


if __name__ == "__main__":
    main()
