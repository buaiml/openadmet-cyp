"""Data handling shared by every training backend.

Datasets are chosen with `--datasets NAME[:ISO,ISO]` and merged at run time into
one wide table with a head per source x isoform. The challenge is always part of
the final stage: it supplies the four scored heads and the pinned split.
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error

from data_tools.merge_datasets import merge
from data_tools.overlap import add_keys


def slug(specs: list[str]) -> str:
    """Directory-safe name for a list of dataset specs."""
    return "+".join(specs).replace(":", "-").replace(",", "_") or "challenge"


def add_common_arguments(parser: argparse.ArgumentParser) -> None:
    """Flags every backend shares."""
    parser.add_argument(
        "--datasets",
        nargs="+",
        default=None,
        metavar="NAME[:ISO,ISO]",
        help="Train these sources jointly (challenge is always included). Heads are named "
        "{isoform}-{source}, e.g. bindingdb:CYP2D6,CYP1A2 tdi pubchem",
    )
    parser.add_argument("--list-datasets", action="store_true", help="List selectable datasets and exit")
    parser.add_argument("--data-dir", type=Path, default=Path("data"), help="Where the source CSVs live (default: data)")
    parser.add_argument("--split-strategy", default="scaffold", choices=["scaffold", "butina", "random"])
    parser.add_argument("--split-seed", type=int, default=0, help="Seed for the merged table's split (default: 0)")
    parser.add_argument(
        "--target-columns",
        nargs="+",
        default=None,
        help="Heads to train on (default: every head of the selected datasets). Missing labels are masked in the loss.",
    )
    parser.add_argument("--output-dir", type=Path, default=None, help="Directory for outputs (default: results/<model>/<datasets>)")
    parser.add_argument("--epochs", type=int, default=50, help="Number of training epochs (default: 50)")
    parser.add_argument("--seed", type=int, default=42, help="Random seed (default: 42)")


def merge_stage(specs: dict[str, list[str] | None], args: argparse.Namespace):
    """Merge the challenge with the given sources on the pinned split.

    Returns (table, heads, scored, tasks) exactly as data_tools.merge_datasets.merge.
    """
    return merge(
        {"challenge": None, **specs},  # the challenge supplies the scored heads and the pinned split
        data_dir=args.data_dir,
        split_strategy=args.split_strategy,
        split_seed=args.split_seed,
    )


def prepare_joint_split(
    specs: dict[str, list[str] | None], args: argparse.Namespace, parser: argparse.ArgumentParser
) -> tuple[pd.DataFrame, pd.DataFrame, list[str], list[str]]:
    """Train/val frames plus (target columns, scored heads to evaluate) for Python-API backends.

    Train holds every pinned-train molecule with at least one selected label, so
    auxiliary-only molecules still train their head. Val is the pinned val split.
    """
    table, heads, scored, tasks = merge_stage(specs, args)
    target_columns = args.target_columns or heads
    unknown = [c for c in target_columns if c not in heads]
    if unknown:
        parser.error(f"--target-columns not among heads {heads}: {unknown}")
    eval_heads = [h for h in target_columns if h in scored]
    if not eval_heads:
        parser.error(f"--target-columns must include at least one scored head: {scored}")
    if any(tasks[h] == "classification" for h in target_columns):
        print("  binary heads present: training as regression, 0/1 labels as numbers", file=sys.stderr)

    df = pd.DataFrame(add_keys(table).to_dict(as_series=False))  # avoids the pyarrow dependency of .to_pandas()
    has_label = df[target_columns].notna().any(axis=1) & df["SMILES"].notna()
    train_df = df.loc[(df["split"] == "train") & has_label].copy()
    val_df = df.loc[df["split"] == "val"].copy()

    if train_df.empty:
        raise ValueError("No training examples")
    if not val_df[eval_heads].notna().any().any():
        raise ValueError(f"No validation examples for {eval_heads}")

    # Guard against structural leakage between train and validation.
    overlap = set(train_df["inchikey_block"]) & set(val_df["inchikey_block"])
    if overlap:
        raise ValueError(f"Train/validation overlap: {len(overlap)} connectivity blocks")
    return train_df, val_df, target_columns, eval_heads


def write_val_outputs(
    output_dir: Path,
    run_id: str,
    val_df: pd.DataFrame,
    train_df: pd.DataFrame,
    pred_by_head: dict[str, np.ndarray],
    eval_heads: list[str],
    run_info: dict[str, object],
) -> None:
    """Score each scored head on val; write predictions.csv (T4 long format) and metrics.csv."""
    long_frames, metric_rows = [], []
    for head in eval_heads:
        labelled = val_df[head].notna().to_numpy()
        actual = val_df.loc[labelled, head].to_numpy(dtype=float)
        pred = pred_by_head[head][labelled]
        rmse = float(np.sqrt(mean_squared_error(actual, pred)))
        model_id = f"{run_id}_{head}"
        long_frames.append(pd.DataFrame({
            "model_id": model_id,
            "inchikey_block": val_df.loc[labelled, "inchikey_block"].to_numpy(),
            "fold": "val",
            "target": head,
            "pred": pred,
        }))
        metric_rows.append({
            "model_id": model_id,
            "target": head,
            **run_info,
            "n_train": int(train_df[head].notna().sum()),
            "n_val": int(labelled.sum()),
            "val_rmse": rmse,
        })
        print(f"Validation RMSE {head}: {rmse:.4f}")

    prediction_path = output_dir / "predictions.csv"
    pd.concat(long_frames).to_csv(prediction_path, index=False)
    metrics_path = output_dir / "metrics.csv"
    pd.DataFrame(metric_rows).to_csv(metrics_path, index=False)
    print(f"Macro validation RMSE: {np.mean([r['val_rmse'] for r in metric_rows]):.4f}")
    print(f"Predictions saved to: {prediction_path}")
    print(f"Metrics saved to: {metrics_path}")

