"""CLI for evaluating CYP challenge models on a train/validation split."""

import argparse
import csv
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


def _resolve_inputs(names: list[str]) -> list[list[str]]:
    """Return the input sets to evaluate: each registered input alone if names == ['all'], else names hstacked."""
    if names == ["all"]:
        return [[n] for n in INPUT_REGISTRY]
    unknown = [n for n in names if n not in INPUT_REGISTRY]
    if unknown:
        raise ValueError(f"Unknown input(s): {unknown}. Available: {list(INPUT_REGISTRY.keys())}")
    return [names]


def _print_results(results: list[tuple[str, str, str, dict[str, float]]]) -> None:
    """Print a formatted metrics table (Isoform, Model, Input, RMSE, MAE, R2) to stdout."""
    iso_w = max(len(iso) for iso, _, _, _ in results)
    model_w = max(len(name) for _, name, _, _ in results)
    input_w = max(len(inp) for _, _, inp, _ in results)
    print(f"{'Isoform':{iso_w}}  {'Model':{model_w}}  {'Input':{input_w}}  {'RMSE':>8}  {'MAE':>8}  {'R2':>8}")
    print(f"{'-' * iso_w}  {'-' * model_w}  {'-' * input_w}  {'-' * 8}  {'-' * 8}  {'-' * 8}")
    for iso, name, inp, m in results:
        print(f"{iso:{iso_w}}  {name:{model_w}}  {inp:{input_w}}  {m['RMSE']:>8.4f}  {m['MAE']:>8.4f}  {m['R2']:>8.4f}")


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
        help="Input featurization(s) to use (hstacked if multiple), or 'all' to evaluate each registered "
        f"input on its own. Available: {list(INPUT_REGISTRY.keys())}",
    )
    parser.add_argument("--cache-dir", default="data/features", help="Directory for cached feature matrices")
    parser.add_argument("--sort-by", default="MAE", choices=["MAE", "RMSE", "R2"], help="Metric to sort results by")
    parser.add_argument(
        "--predictions-out", type=Path,
        help="Write validation predictions to a new CSV (use results/); creates parents, refuses existing files",
    )
    args = parser.parse_args()

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
    for input_names in input_sets:
        input_label = " + ".join(input_names)
        # Features don't depend on the target isoform, so compute once and reuse across isoforms.
        train_features = featurize(train_df, input_names, cache_dir)
        val_features = featurize(val_df, input_names, cache_dir)
        for target in args.targets:
            label = f"{target}, {input_label}"
            X_train, y_train = drop_nan_rows(train_features, train_df[target].to_numpy(), label=f"train/{label}")
            X_val, y_val, val_mask = drop_nan_rows(
                val_features, val_df[target].to_numpy(), label=f"val/{label}", return_mask=True,
            )
            if args.predictions_out is not None:
                val_smiles = val_df["SMILES"].to_numpy()[val_mask]
            for cls in model_classes:
                if args.predictions_out is None:
                    metrics = _evaluate_model(cls(), X_train, y_train, X_val, y_val)
                else:
                    metrics, predictions = _evaluate_model(
                        cls(), X_train, y_train, X_val, y_val, return_predictions=True,
                    )
                    prediction_rows.extend(
                        dict(zip(_PREDICTION_COLUMNS, (
                            smiles, target, cls.name, input_label, args.split, args.seed, float(actual), float(pred),
                        )))
                        for smiles, actual, pred in zip(val_smiles, y_val, predictions)
                    )
                results.append((target, cls.name, input_label, metrics))

    if args.predictions_out is not None:
        try:
            _write_predictions(args.predictions_out, prediction_rows)
        except (OSError, ValueError) as exc:
            print(str(exc), file=sys.stderr)
            sys.exit(1)
        print(f"Wrote {len(prediction_rows)} validation predictions to {args.predictions_out}", file=sys.stderr)

    reverse = args.sort_by == "R2"
    results.sort(key=lambda r: (r[0], -r[3][args.sort_by] if reverse else r[3][args.sort_by]))
    print(f"isoform-mode={args.isoform_mode}  input={' + '.join(args.input)}  split={args.split}  seed={args.seed}")
    _print_results(results)


if __name__ == "__main__":
    main()
