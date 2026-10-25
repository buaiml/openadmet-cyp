
"""Fine-tune Uni-Mol for one CYP target using the pinned dataset split."""

import argparse
import random
from pathlib import Path

import numpy as np
import pandas as pd
import polars as pl
import torch
from sklearn.metrics import mean_squared_error
from unimol_tools import MolPredict, MolTrain

from data_tools.overlap import add_keys


TARGETS = [
    "CYP1A2_pIC50_direct_inhibition",
    "CYP2C9_pIC50_direct_inhibition",
    "CYP2D6_pIC50_direct_inhibition",
    "CYP3A4_pIC50_direct_inhibition",
]


def set_seed(seed: int) -> None:
    """Seed the main random number generators."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Fine-tune Uni-Mol on a pinned train/validation split."
    )
    parser.add_argument("--data-path", type=Path, required=True)
    parser.add_argument("--target", choices=TARGETS, required=True)
    parser.add_argument("--learning-rate", type=float, default=2e-4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--output-dir", type=Path, default=Path("results/unimol"))
    parser.add_argument("--use-cuda", action="store_true")
    args = parser.parse_args()

    set_seed(args.seed)

    df = pd.read_csv(args.data_path)

    required = {"SMILES", "split", args.target}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Missing required columns: {sorted(missing)}")

    # Use the repository's standard molecule connectivity identifiers.
    df = add_keys(pl.read_csv(args.data_path)).to_pandas()
    
    train_df = df.loc[
        (df["split"] == "train")
        & df["SMILES"].notna()
        & df[args.target].notna()
    ].copy()

    val_df = df.loc[
        (df["split"] == "val")
        & df["SMILES"].notna()
        & df[args.target].notna()
    ].copy()

    if train_df.empty:
        raise ValueError(f"No training examples for {args.target}")
    if val_df.empty:
        raise ValueError(f"No validation examples for {args.target}")

    # Guard against structural leakage between train and validation.
    overlap = (
        set(train_df["inchikey_block"])
        & set(val_df["inchikey_block"])
    )
    if overlap:
        raise ValueError(
            f"Train/validation overlap: {len(overlap)} connectivity blocks"
        )

    output_dir = (
        args.output_dir
        / args.target
        / f"lr_{args.learning_rate}_seed_{args.seed}"
    )
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Target: {args.target}")
    print(f"Training molecules: {len(train_df)}")
    print(f"Validation molecules: {len(val_df)}")
    print(f"Learning rate: {args.learning_rate}")
    print(f"Seed: {args.seed}")
    print(f"Device: {'CUDA' if args.use_cuda else 'CPU'}")

    trainer = MolTrain(
        task="regression",
        data_type="molecule",
        model_name="unimolv1",
        smiles_col="SMILES",
        target_cols=[args.target],
        learning_rate=args.learning_rate,
        epochs=args.epochs,
        save_path=str(output_dir),
        use_cuda=args.use_cuda,
        kfold=1,
        split_seed=args.seed,
        seed=args.seed,
        metrics="none",
    )

    # No official validation rows are passed to the trainer.
    trainer.fit(train_df[["SMILES", args.target]])

    predictor = MolPredict(load_model=str(output_dir))
    predictions = np.asarray(
        predictor.predict(val_df[["SMILES"]])
    ).reshape(-1)

    if len(predictions) != len(val_df):
        raise RuntimeError(
            "Prediction count does not match validation count."
        )

    if not np.isfinite(predictions).all():
        raise RuntimeError("Non-finite predictions returned by Uni-Mol.")

    actual = val_df[args.target].to_numpy(dtype=float)
    rmse = float(np.sqrt(mean_squared_error(actual, predictions)))

    model_id = (
        f"unimol_{args.target}_lr{args.learning_rate}_seed{args.seed}"
    )

    # T4 long-format predictions.
    output = pd.DataFrame({
        "model_id": model_id,
        "inchikey_block": val_df["inchikey_block"].to_numpy(),
        "fold": "val",
        "target": args.target,
        "pred": predictions,
    })

    prediction_path = output_dir / "predictions.csv"
    output.to_csv(prediction_path, index=False)

    metrics_path = output_dir / "metrics.csv"
    pd.DataFrame([{
        "model_id": model_id,
        "target": args.target,
        "learning_rate": args.learning_rate,
        "seed": args.seed,
        "n_train": len(train_df),
        "n_val": len(val_df),
        "val_rmse": rmse,
    }]).to_csv(metrics_path, index=False)

    print(f"Validation RMSE: {rmse:.4f}")
    print(f"Predictions saved to: {prediction_path}")
    print(f"Metrics saved to: {metrics_path}")


if __name__ == "__main__":
    main()
