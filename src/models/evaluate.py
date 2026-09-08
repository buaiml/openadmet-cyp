"""CLI for evaluating CYP challenge models on a train/validation split."""

import argparse
import sys
from pathlib import Path

import numpy as np
import polars as pl

from data_tools.inputs import INPUT_REGISTRY, featurize
from data_tools.load import load_data
from models import REGISTRY, CYPModel
from models.utils import drop_nan_rows

_CYP_TARGETS = [
    "CYP1A2_pIC50_direct_inhibition",
    "CYP2C9_pIC50_direct_inhibition",
    "CYP2D6_pIC50_direct_inhibition",
    "CYP3A4_pIC50_direct_inhibition",
]


def _compute_metrics(actual: pl.Series, predicted: pl.Series) -> dict[str, float]:
    """Compute RMSE, MAE, and R2 from actual and predicted Series."""
    a = [float(v) for v in actual]
    p = [float(v) for v in predicted]
    n = len(a)
    mean_a = sum(a) / n
    ss_res = sum((ai - pi) ** 2 for ai, pi in zip(a, p))
    ss_tot = sum((ai - mean_a) ** 2 for ai in a)
    mae = sum(abs(ai - pi) for ai, pi in zip(a, p)) / n
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0.0 else 0.0
    return {"RMSE": (ss_res / n) ** 0.5, "MAE": mae, "R2": r2}


def _evaluate_model(
    model: CYPModel,
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
) -> dict[str, float]:
    """Fit model on train, evaluate on val, return metrics."""
    model.fit(X_train, y_train)
    preds = pl.Series("prediction", model.predict(X_val).tolist())
    return _compute_metrics(pl.Series("actual", y_val.tolist()), preds)


def _resolve_models(names: list[str]) -> list[type[CYPModel]]:
    """Return model classes for the given names, or all registered models if names == ['all']."""
    if names == ["all"]:
        return list(REGISTRY.values())
    unknown = [n for n in names if n not in REGISTRY]
    if unknown:
        raise ValueError(f"Unknown models: {unknown}. Available: {list(REGISTRY.keys())}")
    return [REGISTRY[n] for n in names]


def _print_results(results: list[tuple[str, str, dict[str, float]]]) -> None:
    """Print a formatted metrics table (Isoform, Model, RMSE, MAE, R2) to stdout."""
    iso_w = max(len(iso) for iso, _, _ in results)
    model_w = max(len(name) for _, name, _ in results)
    print(f"{'Isoform':{iso_w}}  {'Model':{model_w}}  {'RMSE':>8}  {'MAE':>8}  {'R2':>8}")
    print(f"{'-' * iso_w}  {'-' * model_w}  {'-' * 8}  {'-' * 8}  {'-' * 8}")
    for iso, name, m in results:
        print(f"{iso:{iso_w}}  {name:{model_w}}  {m['RMSE']:>8.4f}  {m['MAE']:>8.4f}  {m['R2']:>8.4f}")


def main() -> None:
    """Entry point for the evaluate-models CLI."""
    parser = argparse.ArgumentParser(description="Evaluate CYP models on a train/val split.")
    parser.add_argument("--data-dir", type=Path, default=Path("data"), help="Directory containing data CSVs")
    parser.add_argument("--val-split", type=float, default=0.2, help="Validation fraction")
    parser.add_argument(
        "--split",
        default="random",
        choices=["random", "scaffold", "butina"],
        help="Train/val split strategy (default: random)",
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
        "--isoform-mode",
        default="separate",
        choices=["separate"],
        help="How to handle the 4 CYP isoforms. 'separate' (default): fit and evaluate an "
        "independent model per isoform, sharing one feature matrix.",
    )
    parser.add_argument(
        "--targets",
        nargs="+",
        default=_CYP_TARGETS,
        choices=_CYP_TARGETS,
        help="CYP target column(s) to evaluate (default: all four isoforms)",
    )
    parser.add_argument(
        "--input",
        nargs="+",
        default=["rdkit"],
        help=f"Input featurization(s) to use (hstacked if multiple). Available: {list(INPUT_REGISTRY.keys())}",
    )
    parser.add_argument("--cache-dir", default="data/features", help="Directory for cached feature matrices")
    parser.add_argument("--sort-by", default="MAE", choices=["MAE", "RMSE", "R2"], help="Metric to sort results by")
    args = parser.parse_args()

    if not 0.0 < args.val_split < 1.0:
        print(f"--val-split must be in (0, 1), got {args.val_split}", file=sys.stderr)
        sys.exit(1)

    try:
        model_classes = _resolve_models(args.models)
        unknown_inputs = [n for n in args.input if n not in INPUT_REGISTRY]
        if unknown_inputs:
            raise ValueError(f"Unknown input(s): {unknown_inputs}. Available: {list(INPUT_REGISTRY.keys())}")
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

    # Features don't depend on the target isoform, so compute once and reuse across isoforms.
    train_features = featurize(train_df, args.input, cache_dir)
    val_features = featurize(val_df, args.input, cache_dir)

    results: list[tuple[str, str, dict[str, float]]] = []
    for target in args.targets:
        X_train, y_train = drop_nan_rows(train_features, train_df[target].to_numpy(), label=f"train/{target}")
        X_val, y_val = drop_nan_rows(val_features, val_df[target].to_numpy(), label=f"val/{target}")
        for cls in model_classes:
            metrics = _evaluate_model(cls(), X_train, y_train, X_val, y_val)
            results.append((target, cls.name, metrics))

    reverse = args.sort_by == "R2"
    results.sort(key=lambda r: (r[0], -r[2][args.sort_by] if reverse else r[2][args.sort_by]))
    print(f"isoform-mode={args.isoform_mode}  input={' + '.join(args.input)}  split={args.split}  seed={args.seed}")
    _print_results(results)


if __name__ == "__main__":
    main()
